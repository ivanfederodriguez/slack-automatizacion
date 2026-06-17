from __future__ import annotations

import json
import sqlite3
import tempfile
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Optional

from slack_personal_agent import AgentApp, AgentConfig, SlackClassification


class FixtureStructuredModel:
    def __init__(self, response: dict[str, Any]) -> None:
        self.response = SlackClassification.model_validate(response)
        self.calls: list[str] = []

    def invoke(self, prompt: str) -> SlackClassification:
        self.calls.append(prompt)
        return self.response


class FixtureSlackClient:
    def __init__(self, fixture: dict[str, Any]) -> None:
        self.fixture = fixture
        self.history_calls: list[dict[str, Any]] = []
        self.reply_calls: list[dict[str, Any]] = []
        self.post_calls: list[dict[str, Any]] = []
        self.users = dict(fixture.get("users") or {})
        my_user_id = str(fixture.get("my_user_id") or "UME")
        message_user = str((fixture.get("message") or {}).get("user") or "")
        conversation_user = str((fixture.get("conversation") or {}).get("user") or "")
        self.users.setdefault(my_user_id, "Ivan Rodriguez")
        if message_user:
            self.users.setdefault(message_user, message_user)
        if conversation_user:
            self.users.setdefault(conversation_user, conversation_user)

    def users_info(self, user: str) -> dict[str, Any]:
        label = self.users.get(user, user)
        return {
            "user": {
                "name": label,
                "profile": {"real_name": label, "display_name": label},
            }
        }

    def auth_test(self) -> dict[str, Any]:
        my_user_id = str(self.fixture.get("my_user_id") or "UME")
        return {"team": "Fixture", "user": "fixture", "user_id": my_user_id}

    def conversations_history(self, **kwargs: Any) -> dict[str, Any]:
        self.history_calls.append(kwargs)
        return {
            "messages": list(self.fixture.get("history_messages") or []),
            "response_metadata": {},
        }

    def conversations_replies(self, **kwargs: Any) -> dict[str, Any]:
        self.reply_calls.append(kwargs)
        return {
            "messages": list(self.fixture.get("thread_messages") or []),
            "response_metadata": {},
        }

    def chat_postMessage(self, **kwargs: Any) -> dict[str, Any]:
        self.post_calls.append(kwargs)
        return {"ok": True, "ts": "fixture-post.000000"}


class DryRunReplayAgentApp(AgentApp):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.skipped_slack_posts: list[dict[str, Any]] = []

    def send_task_acknowledgement(self, task_id: int) -> bool:
        self.skipped_slack_posts.append({"type": "task_ack", "task_id": task_id})
        return True

    def send_context_acknowledgement(
        self,
        task_id: int,
        new_context_text: str,
        *,
        transcribed_audio: bool = False,
    ) -> bool:
        self.skipped_slack_posts.append(
            {
                "type": "context_ack",
                "task_id": task_id,
                "text": new_context_text,
                "transcribed_audio": transcribed_audio,
            }
        )
        return True


@dataclass(frozen=True)
class ReprocessMessageResult:
    fixture_path: Path
    dry_run: bool
    message_text: str
    extracted_urls: list[str]
    model_classification: Optional[SlackClassification]
    final_classification: Optional[SlackClassification]
    requested_action: str
    public_request_text: str
    context_text: str
    task_created: bool
    slack_post_calls: list[dict[str, Any]]
    skipped_slack_posts: list[dict[str, Any]]
    trello_enabled: bool


def _load_fixture(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _fetch_single_row(conn: sqlite3.Connection, query: str, params: tuple[Any, ...]) -> Optional[sqlite3.Row]:
    conn.row_factory = sqlite3.Row
    return conn.execute(query, params).fetchone()


def run_reprocess_message_fixture(
    *,
    config: AgentConfig,
    fixture_path: Path,
    dry_run: bool = False,
) -> ReprocessMessageResult:
    fixture = _load_fixture(fixture_path)
    message = dict(fixture["message"])
    conversation = dict(fixture["conversation"])
    my_user_id = str(fixture.get("my_user_id") or config.my_slack_user_id or "")
    model_response = fixture.get("model_response")
    fixture_model = FixtureStructuredModel(model_response) if model_response else None
    fixture_slack = FixtureSlackClient(fixture)

    with tempfile.TemporaryDirectory(prefix="slack-agent-replay-") as tmp_dir:
        replay_db = str(Path(tmp_dir) / "replay.db") if dry_run else config.db_path
        replay_config = replace(
            config,
            db_path=replay_db,
            trello_enabled=False,
            trello_auto_create=False,
        )
        app_cls = DryRunReplayAgentApp if dry_run else AgentApp
        app = app_cls(
            replay_config,
            slack_client=fixture_slack,
            structured_model_factory=(
                (lambda schema: fixture_model)
                if fixture_model is not None
                else None
            ),
            sleep_fn=lambda _: None,
        )
        app.init_db()
        app.process_message(message, conversation, my_user_id=my_user_id)

        channel_id = conversation["id"]
        message_ts = message["ts"]
        with app.db_connect() as conn:
            processed = _fetch_single_row(
                conn,
                """
                SELECT context_text, classification_json
                FROM processed_messages
                WHERE channel_id = ? AND message_ts = ?
                """,
                (channel_id, message_ts),
            )
            task = _fetch_single_row(
                conn,
                """
                SELECT public_request_text, requested_action, classification_json
                FROM tasks
                WHERE channel_id = ? AND message_ts = ?
                ORDER BY id DESC
                LIMIT 1
                """,
                (channel_id, message_ts),
            )

        final_payload: dict[str, Any] = {}
        if task and task["classification_json"]:
            final_payload = json.loads(task["classification_json"])
        elif processed and processed["classification_json"]:
            final_payload = json.loads(processed["classification_json"])

        final_classification = (
            SlackClassification.model_validate(final_payload)
            if final_payload
            else None
        )

        links = app.extract_message_links(message.get("text") or "")
        skipped_slack_posts = list(getattr(app, "skipped_slack_posts", []))
        return ReprocessMessageResult(
            fixture_path=fixture_path,
            dry_run=dry_run,
            message_text=message.get("text") or "",
            extracted_urls=[link.url for link in links],
            model_classification=fixture_model.response if fixture_model is not None else None,
            final_classification=final_classification,
            requested_action=(task["requested_action"] if task else "") or (
                final_classification.requested_action if final_classification else ""
            ),
            public_request_text=(task["public_request_text"] if task else ""),
            context_text=(processed["context_text"] if processed else "") or "",
            task_created=task is not None,
            slack_post_calls=list(fixture_slack.post_calls),
            skipped_slack_posts=skipped_slack_posts,
            trello_enabled=replay_config.trello_enabled,
        )


def _classification_lines(title: str, classification: Optional[SlackClassification]) -> list[str]:
    if classification is None:
        return [title, "- Sin clasificación disponible."]
    return [
        title,
        f"- category: {classification.category}",
        f"- needs_external_system: {str(classification.needs_external_system).lower()}",
        f"- external_systems: [{', '.join(classification.external_systems)}]",
    ]


def format_reprocess_message_result(
    result: ReprocessMessageResult,
    *,
    show_before_after: bool = False,
) -> str:
    lines = [
        f"Fixture: {result.fixture_path}",
        f"Dry-run: {str(result.dry_run).lower()}",
        "",
        "Texto original:",
        result.message_text or "(sin texto)",
        "",
        "URLs extraídas:",
    ]
    lines.extend(f"- {url}" for url in result.extracted_urls)
    if not result.extracted_urls:
        lines.append("- Sin URLs detectadas.")

    lines.extend(["", "Contexto usado:", result.context_text or "Sin contexto reciente."])

    if show_before_after:
        lines.extend(["", *_classification_lines("Clasificación del modelo:", result.model_classification)])

    lines.extend(["", *_classification_lines("Clasificación final:", result.final_classification)])
    lines.extend(
        [
            "",
            "Acción pedida final:",
            result.requested_action or "Sin acción pedida.",
            "",
            "public_request_text:",
            result.public_request_text or "Sin texto público porque no se creó tarea.",
            "",
            f"Habría creado tarea: {'sí' if result.task_created else 'no'}",
            f"Slack postMessage llamadas: {len(result.slack_post_calls)}",
            f"Slack posts omitidos por dry-run: {len(result.skipped_slack_posts)}",
            f"Trello habilitado en replay: {str(result.trello_enabled).lower()}",
        ]
    )
    return "\n".join(lines)
