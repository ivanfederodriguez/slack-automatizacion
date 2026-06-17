import json
import sqlite3
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import slack_personal_agent
from main import build_parser
import trello_client
from trello_client import TrelloCard, TrelloCardState, TrelloClient, TrelloComment
from slack_personal_agent import (
    AgentApp,
    AgentConfig,
    AudioTranscriptFusion,
    SlackClassification,
    local_model_fit_hint,
    parse_snooze_until,
    validate_trello_api_key_format,
)
from url_enrichment import extract_urls_from_text


def make_config(tmp_path, **overrides):
    env = {
        "SLACK_USER_TOKEN": "xoxp-test-token",
        "MY_SLACK_USER_ID": "UME",
        "DB_PATH": str(tmp_path / "agent.db"),
        "MODEL_PROVIDER": "ollama",
        "OLLAMA_MODEL": "qwen3:4b-instruct",
    }
    env.update(overrides)
    return AgentConfig.from_env(env)


def sample_classification():
    return SlackClassification(
        is_actionable=True,
        summary="Hay que revisar un reporte.",
        requested_action="Comparar el reporte con Salesforce.",
        priority="medium",
        category="salesforce",
        needs_reply=True,
        needs_external_system=True,
        external_systems=["salesforce"],
        missing_information=[],
        suggested_next_step="Abrir Salesforce y validar los números.",
        draft_reply="Dale, lo reviso y te confirmo si coincide.",
    )


def make_classification(**overrides):
    return sample_classification().model_copy(update=overrides)


class FakeStructuredModel:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def invoke(self, prompt):
        self.calls.append(prompt)
        if not self.responses:
            raise AssertionError("No había respuestas falsas disponibles.")
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


class FakeSlackClient:
    def __init__(self, post_error=None):
        self.users = {
            "UME": "Ivan Rodriguez",
            "UOTHER": "Ana Gomez",
        }
        self.list_calls = []
        self.history_calls = []
        self.history_messages = []
        self.history_lookup = {}
        self.reply_calls = []
        self.replies_lookup = {}
        self.post_calls = []
        self.post_error = post_error

    def users_info(self, user):
        return {
            "user": {
                "name": self.users.get(user, user),
                "profile": {"real_name": self.users.get(user, user)},
            }
        }

    def auth_test(self):
        return {"team": "Test", "user": "ivan", "user_id": "UME"}

    def conversations_list(self, **kwargs):
        self.list_calls.append(kwargs)
        return {"channels": [], "response_metadata": {}}

    def conversations_history(self, **kwargs):
        self.history_calls.append(kwargs)
        lookup_key = (
            kwargs.get("channel"),
            kwargs.get("oldest"),
            kwargs.get("latest"),
            kwargs.get("inclusive"),
        )
        messages = self.history_lookup.get(lookup_key, self.history_messages)
        return {"messages": list(messages), "response_metadata": {}}

    def conversations_replies(self, **kwargs):
        self.reply_calls.append(kwargs)
        lookup_key = (kwargs.get("channel"), kwargs.get("ts"))
        messages = self.replies_lookup.get(lookup_key, [])
        return {"messages": list(messages), "response_metadata": {}}

    def chat_postMessage(self, **kwargs):
        self.post_calls.append(kwargs)
        if self.post_error is not None:
            raise self.post_error
        return {"ok": True, "ts": "50.123456"}


class FakeTrelloClient:
    def __init__(self, should_fail=False, card_state=None, card_comments=None, attachment_error=None):
        self.should_fail = should_fail
        self.attachment_error = attachment_error
        self.card_state = card_state or TrelloCardState(
            id="card123",
            name="Test card",
            url="https://trello.com/c/card123",
            list_id="list123",
            list_name="Inbox",
            closed=False,
        )
        self.card_comments = list(card_comments or [])
        self.created_cards = []
        self.comments = []
        self.file_attachments = []
        self.url_attachments = []

    def get_me(self):
        return {"fullName": "Ivan Rodriguez", "username": "ivan"}

    def get_list(self, list_id):
        return type("FakeList", (), {"id": list_id, "name": "Inbox", "board_id": "B1"})()

    def create_card(self, **kwargs):
        if self.should_fail:
            raise RuntimeError("trello down")
        self.created_cards.append(kwargs)
        return TrelloCard(id="card123", name=kwargs["name"], url="https://trello.com/c/card123")

    def get_card(self, card_id):
        if self.should_fail:
            raise RuntimeError("trello down")
        return self.card_state

    def get_card_comments(self, card_id, limit=50):
        if self.should_fail:
            raise RuntimeError("trello down")
        return list(self.card_comments[:limit])

    def add_card_comment(self, card_id, text):
        if self.should_fail:
            raise RuntimeError("trello down")
        self.comments.append({"card_id": card_id, "text": text})

    def attach_file_to_card(self, card_id, file_path, name=None):
        if self.should_fail:
            raise RuntimeError("trello down")
        if self.attachment_error is not None:
            raise self.attachment_error
        self.file_attachments.append({"card_id": card_id, "file_path": file_path, "name": name})
        return f"att-{len(self.file_attachments)}"

    def add_url_attachment_to_card(self, card_id, url, name=None):
        if self.should_fail:
            raise RuntimeError("trello down")
        if self.attachment_error is not None:
            raise self.attachment_error
        self.url_attachments.append({"card_id": card_id, "url": url, "name": name})
        return f"url-{len(self.url_attachments)}"


class FakeTelegramClient:
    def __init__(self, send_error=None, updates=None):
        self.send_error = send_error
        self.sent_messages = []
        self.updates = list(updates or [])

    def send_message(self, text):
        self.sent_messages.append(text)
        if self.send_error is not None:
            raise self.send_error
        return {"ok": True}

    def get_updates(self, offset=None, limit=20):
        return list(self.updates[:limit])


class FakeAudioTranscriber:
    def __init__(self, responses=None, error=None):
        self.responses = list(responses or [])
        self.error = error
        self.calls = []

    def transcribe(self, audio_path):
        self.calls.append(audio_path)
        if self.error is not None:
            raise self.error
        if self.responses:
            return self.responses.pop(0)
        return "Transcripción local de prueba."


class DoctorResult:
    summary = "Proveedor listo."


def get_processed_row(db_path):
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        return conn.execute(
            "SELECT * FROM processed_messages ORDER BY processed_at ASC LIMIT 1"
        ).fetchone()


def count_tasks(db_path):
    with sqlite3.connect(db_path) as conn:
        return conn.execute("SELECT COUNT(*) FROM tasks").fetchone()[0]


def get_task(db_path, where="1 = 1", params=()):
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        return conn.execute(f"SELECT * FROM tasks WHERE {where} ORDER BY id ASC LIMIT 1", params).fetchone()


def get_audio_rows(db_path):
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        return conn.execute("SELECT * FROM audio_transcriptions ORDER BY id ASC").fetchall()


def get_task_events(db_path, event_type=None):
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        if event_type:
            return conn.execute(
                "SELECT * FROM task_events WHERE event_type = ? ORDER BY id ASC",
                (event_type,),
            ).fetchall()
        return conn.execute("SELECT * FROM task_events ORDER BY id ASC").fetchall()


def get_processed_actions(db_path):
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        return conn.execute("SELECT * FROM trello_processed_actions ORDER BY id ASC").fetchall()


def get_file_rows(db_path):
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        return conn.execute("SELECT * FROM slack_file_attachments ORDER BY id ASC").fetchall()


def audio_file(**overrides):
    payload = {
        "id": "F1",
        "name": "audio.m4a",
        "mimetype": "audio/mp4",
        "url_private_download": "https://files.slack.test/audio.m4a",
    }
    payload.update(overrides)
    return payload


def image_file(**overrides):
    payload = {
        "id": "IMG1",
        "name": "captura.png",
        "mimetype": "image/png",
        "filetype": "png",
        "url_private_download": "https://files.slack.test/captura.png",
        "size": 2048,
    }
    payload.update(overrides)
    return payload


def insert_task(
    app,
    *,
    created_at,
    message_ts,
    summary,
    public_request_text=None,
    requested_action,
    has_audio_transcript=0,
    priority,
    category,
    classification,
    sender_label="Ana Gomez",
    conversation_label="DM con Ana Gomez",
    status="new",
    trello_status="created",
    trello_card_id="",
    trello_card_url="",
    trello_last_error="",
):
    with sqlite3.connect(app.config.db_path) as conn:
        conn.execute(
            """
            INSERT INTO tasks (
                created_at,
                channel_id,
                message_ts,
                user_id,
                sender_label,
                conversation_label,
                summary,
                public_request_text,
                requested_action,
                has_audio_transcript,
                priority,
                category,
                status,
                trello_status,
                trello_card_id,
                trello_card_url,
                trello_last_error,
                classification_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                created_at,
                f"D{message_ts}",
                message_ts,
                "UOTHER",
                sender_label,
                conversation_label,
                summary,
                public_request_text or summary,
                requested_action,
                has_audio_transcript,
                priority,
                category,
                status,
                trello_status,
                trello_card_id,
                trello_card_url,
                trello_last_error,
                classification.model_dump_json(),
            ),
        )


def test_dm_actionable_creates_task(tmp_path):
    fake_model = FakeStructuredModel([sample_classification()])
    app = AgentApp(
        make_config(tmp_path),
        slack_client=FakeSlackClient(),
        structured_model_factory=lambda schema: fake_model,
        sleep_fn=lambda _: None,
    )
    app.init_db()

    app.process_message(
        {"ts": "1.0", "text": "Revisás este reporte?", "user": "UOTHER"},
        {"id": "D123", "is_im": True, "user": "UOTHER"},
        my_user_id="UME",
    )

    row = get_processed_row(app.config.db_path)
    assert row["classification_status"] == "done"
    assert row["relevant"] == 1
    assert count_tasks(app.config.db_path) == 1


def test_salesforce_report_request_overrides_research_classification(tmp_path):
    message_text = (
        "Ivo, me armás un informe de altas 2026 por campaña principal/campaña de origen:\n"
        "<https://techo.lightning.force.com/lightning/r/Campaign/7011W000001buEh/view|[IND] Campañas Pauta Digital>\n"
        "- amplify\n"
        "- orgánico web"
    )
    fake_model = FakeStructuredModel(
        [
            make_classification(
                summary="Micaela pide un informe de altas 2026.",
                requested_action="Armar un informe de altas 2026 por campaña principal/campaña de origen.",
                category="research",
                needs_external_system=False,
                external_systems=[],
            )
        ]
    )
    app = AgentApp(
        make_config(tmp_path),
        slack_client=FakeSlackClient(),
        structured_model_factory=lambda schema: fake_model,
        sleep_fn=lambda _: None,
    )
    app.init_db()

    app.process_message(
        {"ts": "1.05", "text": message_text, "user": "UOTHER"},
        {"id": "D123", "is_im": True, "user": "UOTHER"},
        my_user_id="UME",
    )

    task = get_task(app.config.db_path)
    classification = json.loads(task["classification_json"])
    assert count_tasks(app.config.db_path) == 1
    assert task["category"] == "salesforce"
    assert classification["category"] == "salesforce"
    assert classification["needs_external_system"] is True
    assert "salesforce" in classification["external_systems"]
    assert "Campañas/fuentes solicitadas:" in task["public_request_text"]
    assert "- [IND] Campañas Pauta Digital" in task["public_request_text"]
    assert "  - amplify" in task["public_request_text"]


def test_actionable_task_can_auto_sync_to_trello(tmp_path):
    fake_model = FakeStructuredModel([sample_classification()])
    fake_trello = FakeTrelloClient()
    app = AgentApp(
        make_config(
            tmp_path,
            TRELLO_ENABLED="true",
            TRELLO_API_KEY="key",
            TRELLO_TOKEN="token",
            TRELLO_LIST_ID="list123",
        ),
        slack_client=FakeSlackClient(),
        structured_model_factory=lambda schema: fake_model,
        trello_client=fake_trello,
        sleep_fn=lambda _: None,
    )
    app.init_db()

    app.process_message(
        {"ts": "1.1", "text": "Revisás este reporte?", "user": "UOTHER"},
        {"id": "D123", "is_im": True, "user": "UOTHER"},
        my_user_id="UME",
    )

    with sqlite3.connect(app.config.db_path) as conn:
        conn.row_factory = sqlite3.Row
        task = conn.execute("SELECT trello_status, trello_card_url FROM tasks LIMIT 1").fetchone()
    assert task["trello_status"] == "created"
    assert task["trello_card_url"] == "https://trello.com/c/card123"
    assert len(fake_trello.created_cards) == 1


def test_new_dm_task_sends_automatic_ack_without_mention(tmp_path):
    fake_model = FakeStructuredModel([sample_classification()])
    fake_slack = FakeSlackClient()
    app = AgentApp(
        make_config(tmp_path),
        slack_client=fake_slack,
        structured_model_factory=lambda schema: fake_model,
        sleep_fn=lambda _: None,
    )
    app.init_db()

    app.process_message(
        {"ts": "1.2", "text": "Revisás este reporte?", "user": "UOTHER"},
        {"id": "D123", "is_im": True, "user": "UOTHER"},
        my_user_id="UME",
    )

    task = get_task(app.config.db_path)
    assert task["acknowledged_at"]
    assert fake_slack.post_calls[0]["channel"] == "D123"
    assert fake_slack.post_calls[0]["thread_ts"] == "1.2"
    text = fake_slack.post_calls[0]["text"]
    assert "\n\n*Pedido registrado:*" in text
    assert "*Sistema:* Salesforce" in text
    assert "*Estado:* Pendiente de revisión" in text
    assert "Lo dejé registrado para revisarlo" in text
    assert "<@UOTHER>" not in text


def test_new_private_channel_task_ack_mentions_requester(tmp_path):
    fake_model = FakeStructuredModel([sample_classification()])
    fake_slack = FakeSlackClient()
    app = AgentApp(
        make_config(tmp_path),
        slack_client=fake_slack,
        structured_model_factory=lambda schema: fake_model,
        sleep_fn=lambda _: None,
    )
    app.init_db()

    app.process_message(
        {"ts": "1.3", "text": "ivan, revisás este reporte?", "user": "UOTHER"},
        {"id": "G123", "name": "finanzas", "is_private": True},
        my_user_id="UME",
    )

    assert fake_slack.post_calls[0]["text"].startswith("<@UOTHER> Dale, lo tomo.")
    assert "*Pedido registrado:*" in fake_slack.post_calls[0]["text"]


def test_new_group_dm_task_ack_mentions_requester(tmp_path):
    fake_model = FakeStructuredModel([sample_classification()])
    fake_slack = FakeSlackClient()
    app = AgentApp(
        make_config(tmp_path),
        slack_client=fake_slack,
        structured_model_factory=lambda schema: fake_model,
        sleep_fn=lambda _: None,
    )
    app.init_db()

    app.process_message(
        {"ts": "1.4", "text": "Revisás este reporte?", "user": "UOTHER"},
        {"id": "GMPIM", "name": "mpdm-test", "is_mpim": True},
        my_user_id="UME",
    )

    assert fake_slack.post_calls[0]["text"].startswith("<@UOTHER> Dale, lo tomo.")
    assert "*Pedido registrado:*" in fake_slack.post_calls[0]["text"]


def test_send_task_acknowledgement_uses_public_request_text_without_truncation(tmp_path):
    fake_slack = FakeSlackClient()
    app = AgentApp(
        make_config(tmp_path),
        slack_client=fake_slack,
        structured_model_factory=lambda schema: FakeStructuredModel([]),
        sleep_fn=lambda _: None,
    )
    app.init_db()
    long_request = (
        "Modificar los links de formularios en la actualización compartida, verificando que cada "
        "link apunte al formulario correcto y que no se rompa el flujo de carga para el equipo."
    )
    insert_task(
        app,
        created_at="2026-06-11T12:00:00+00:00",
        message_ts="1.45",
        summary="Luciana Santos pide a Ivan Rodríguez que modifique los links de formularios...",
        public_request_text=long_request,
        requested_action="Modificar los links de formularios en la actualización compartida.",
        priority="medium",
        category="admin",
        classification=make_classification(
            summary="Luciana Santos pide a Ivan Rodríguez que modifique los links de formularios...",
            requested_action="Modificar los links de formularios en la actualización compartida.",
            category="admin",
        ),
    )

    app.send_task_acknowledgement(1)

    text = fake_slack.post_calls[0]["text"]
    assert "*Pedido registrado:*" in text
    assert long_request in text
    assert "Luciana Santos pide a Ivan Rodríguez" not in text
    assert "..." not in text
    assert "compact_text" not in text


def test_send_task_acknowledgement_uses_audio_copy_when_audio_was_transcribed(tmp_path):
    fake_slack = FakeSlackClient()
    app = AgentApp(
        make_config(tmp_path),
        slack_client=fake_slack,
        structured_model_factory=lambda schema: FakeStructuredModel([]),
        sleep_fn=lambda _: None,
    )
    app.init_db()
    insert_task(
        app,
        created_at="2026-06-11T12:00:00+00:00",
        message_ts="1.46",
        summary="Explicar en lenguaje simple la API de Salesforce.",
        public_request_text="Explicar en lenguaje simple la API de Salesforce.",
        requested_action="Explicar en lenguaje simple la API de Salesforce.",
        priority="medium",
        category="communications",
        classification=make_classification(
            summary="Explicar en lenguaje simple la API de Salesforce.",
            requested_action="Explicar en lenguaje simple la API de Salesforce.",
            category="communications",
        ),
    )
    with sqlite3.connect(app.config.db_path) as conn:
        conn.execute(
            "UPDATE tasks SET has_audio_transcript = 1 WHERE id = 1",
        )

    app.send_task_acknowledgement(1)

    text = fake_slack.post_calls[0]["text"]
    assert "Transcribí el audio y lo dejé registrado para revisarlo" in text


def test_thread_context_updates_existing_task_without_duplicate_and_confirms(tmp_path):
    fake_model = FakeStructuredModel([sample_classification()])
    fake_slack = FakeSlackClient()
    fake_trello = FakeTrelloClient()
    app = AgentApp(
        make_config(
            tmp_path,
            TRELLO_ENABLED="true",
            TRELLO_API_KEY="key",
            TRELLO_TOKEN="token",
            TRELLO_LIST_ID="list123",
        ),
        slack_client=fake_slack,
        structured_model_factory=lambda schema: fake_model,
        trello_client=fake_trello,
        sleep_fn=lambda _: None,
    )
    app.init_db()
    conversation = {"id": "D123", "is_im": True, "user": "UOTHER"}

    app.process_message(
        {"ts": "1.5", "text": "Revisás este reporte?", "user": "UOTHER"},
        conversation,
        my_user_id="UME",
    )
    app.process_message(
        {
            "ts": "1.6",
            "thread_ts": "1.5",
            "text": "Además, usá el reporte de mayo como referencia.",
            "user": "UOTHER",
        },
        conversation,
        my_user_id="UME",
    )

    with sqlite3.connect(app.config.db_path) as conn:
        conn.row_factory = sqlite3.Row
        processed_context = conn.execute(
            "SELECT classification_status FROM processed_messages WHERE message_ts = '1.6'"
        ).fetchone()
        context_events = conn.execute(
            "SELECT details_json FROM task_events WHERE event_type = 'context_added'"
        ).fetchall()
        task = conn.execute("SELECT last_context_ack_at FROM tasks LIMIT 1").fetchone()

    assert count_tasks(app.config.db_path) == 1
    assert len(fake_model.calls) == 1
    assert processed_context["classification_status"] == "context_added"
    assert task["last_context_ack_at"]
    assert "*Pedido actualizado:*" in fake_slack.post_calls[1]["text"]
    assert "Buenísimo, gracias. Lo sumo al pedido." in fake_slack.post_calls[1]["text"]
    assert len(context_events) == 1
    assert fake_trello.comments[0]["card_id"] == "card123"
    assert "reporte de mayo" in fake_trello.comments[0]["text"]


def test_consecutive_dm_messages_from_same_sender_group_into_one_task(tmp_path):
    fake_model = FakeStructuredModel([sample_classification()])
    fake_slack = FakeSlackClient()
    app = AgentApp(
        make_config(tmp_path, CASE_GROUPING_WINDOW_MINUTES="15"),
        slack_client=fake_slack,
        structured_model_factory=lambda schema: fake_model,
        sleep_fn=lambda _: None,
    )
    app.init_db()
    conversation = {"id": "D123", "is_im": True, "user": "UOTHER"}

    app.process_message(
        {"ts": "1.55", "text": "Revisás este reporte?", "user": "UOTHER"},
        conversation,
        my_user_id="UME",
    )
    app.process_message(
        {"ts": "1.56", "text": "La versión buena es la de junio.", "user": "UOTHER"},
        conversation,
        my_user_id="UME",
    )

    with sqlite3.connect(app.config.db_path) as conn:
        conn.row_factory = sqlite3.Row
        second_message = conn.execute(
            "SELECT classification_status FROM processed_messages WHERE message_ts = '1.56'"
        ).fetchone()
        context_events = conn.execute(
            "SELECT details_json FROM task_events WHERE event_type = 'context_added'"
        ).fetchall()

    assert count_tasks(app.config.db_path) == 1
    assert len(fake_model.calls) == 1
    assert second_message["classification_status"] == "context_added"
    assert len(context_events) == 1
    assert "La versión buena" in context_events[0]["details_json"]
    assert "*Pedido actualizado:*" in fake_slack.post_calls[1]["text"]
    assert "Buenísimo, gracias. Lo sumo al pedido." in fake_slack.post_calls[1]["text"]


def test_recent_context_excludes_previous_day_without_thread(tmp_path):
    zone = ZoneInfo("America/Argentina/Cordoba")
    current_ts = f"{datetime(2026, 6, 17, 0, 30, tzinfo=zone).timestamp():.6f}"
    yesterday_ts = f"{datetime(2026, 6, 16, 23, 45, tzinfo=zone).timestamp():.6f}"
    fake_slack = FakeSlackClient()
    fake_slack.history_messages = [
        {"ts": yesterday_ts, "text": "Ayer era otro pedido.", "user": "UOTHER"},
    ]
    app = AgentApp(
        make_config(
            tmp_path,
            CONTEXT_MAX_AGE_MINUTES="120",
            LOCAL_TIMEZONE="America/Argentina/Cordoba",
        ),
        slack_client=fake_slack,
        structured_model_factory=lambda schema: FakeStructuredModel([]),
        sleep_fn=lambda _: None,
    )

    context = app.fetch_recent_context("D123", current_ts)

    assert "Ayer era otro pedido" not in context
    assert context == ""


def test_recent_context_includes_same_day_message_within_window(tmp_path):
    zone = ZoneInfo("America/Argentina/Cordoba")
    current_ts = f"{datetime(2026, 6, 17, 10, 0, tzinfo=zone).timestamp():.6f}"
    previous_ts = f"{datetime(2026, 6, 17, 9, 30, tzinfo=zone).timestamp():.6f}"
    fake_slack = FakeSlackClient()
    fake_slack.history_messages = [
        {"ts": previous_ts, "text": "Usá también el reporte de mayo.", "user": "UOTHER"},
    ]
    app = AgentApp(
        make_config(
            tmp_path,
            CONTEXT_MAX_AGE_MINUTES="120",
            LOCAL_TIMEZONE="America/Argentina/Cordoba",
        ),
        slack_client=fake_slack,
        structured_model_factory=lambda schema: FakeStructuredModel([]),
        sleep_fn=lambda _: None,
    )

    context = app.fetch_recent_context("D123", current_ts)

    assert "Ana Gomez: Usá también el reporte de mayo." in context


def test_duplicate_thread_context_does_not_send_confirmation(tmp_path):
    fake_model = FakeStructuredModel([sample_classification()])
    fake_slack = FakeSlackClient()
    app = AgentApp(
        make_config(tmp_path),
        slack_client=fake_slack,
        structured_model_factory=lambda schema: fake_model,
        sleep_fn=lambda _: None,
    )
    app.init_db()
    conversation = {"id": "D123", "is_im": True, "user": "UOTHER"}

    app.process_message(
        {"ts": "1.7", "text": "Revisás este reporte?", "user": "UOTHER"},
        conversation,
        my_user_id="UME",
    )
    app.process_message(
        {
            "ts": "1.8",
            "thread_ts": "1.7",
            "text": "Revisás este reporte?",
            "user": "UOTHER",
        },
        conversation,
        my_user_id="UME",
    )

    processed_context = get_processed_row(app.config.db_path)
    with sqlite3.connect(app.config.db_path) as conn:
        status = conn.execute(
            "SELECT classification_status FROM processed_messages WHERE message_ts = '1.8'"
        ).fetchone()[0]

    assert processed_context["message_ts"] == "1.7"
    assert status == "context_duplicate"
    assert count_tasks(app.config.db_path) == 1
    assert len(fake_slack.post_calls) == 1


def test_audio_disabled_does_not_change_flow(tmp_path):
    fake_model = FakeStructuredModel([sample_classification()])
    app = AgentApp(
        make_config(tmp_path, AUDIO_TRANSCRIPTION_ENABLED="false"),
        slack_client=FakeSlackClient(),
        structured_model_factory=lambda schema: fake_model,
        audio_transcriber=FakeAudioTranscriber(error=AssertionError("should not transcribe")),
        sleep_fn=lambda _: None,
    )
    app.init_db()

    app.process_message(
        {
            "ts": "1.9",
            "text": "Revisás este reporte?",
            "user": "UOTHER",
            "files": [audio_file(transcription_text="audio ignorado")],
        },
        {"id": "D123", "is_im": True, "user": "UOTHER"},
        my_user_id="UME",
    )

    assert "Audios transcriptos" not in fake_model.calls[0]
    assert get_audio_rows(app.config.db_path) == []


def test_slack_audio_transcript_only_is_used_before_classification(tmp_path):
    fake_model = FakeStructuredModel([sample_classification()])
    app = AgentApp(
        make_config(tmp_path, LOCAL_WHISPER_ENABLED="false"),
        slack_client=FakeSlackClient(),
        structured_model_factory=lambda schema: fake_model,
        sleep_fn=lambda _: None,
    )
    app.init_db()

    app.process_message(
        {
            "ts": "2.1",
            "text": "",
            "user": "UOTHER",
            "files": [audio_file(transcription_text="Necesito revisar el reporte de audio.")],
        },
        {"id": "D123", "is_im": True, "user": "UOTHER"},
        my_user_id="UME",
    )

    assert "Audios transcriptos:" in fake_model.calls[0]
    assert "Necesito revisar el reporte de audio." in fake_model.calls[0]
    rows = get_audio_rows(app.config.db_path)
    assert rows[0]["transcription_status"] == "slack_only"
    assert rows[0]["selected_transcript_text"] == "Necesito revisar el reporte de audio."


def test_local_whisper_audio_transcript_only_is_used_before_classification(tmp_path):
    fake_model = FakeStructuredModel([sample_classification()])
    fake_transcriber = FakeAudioTranscriber(["Necesito revisar el reporte local."])
    app = AgentApp(
        make_config(tmp_path, SLACK_AUDIO_TRANSCRIPTS_ENABLED="false"),
        slack_client=FakeSlackClient(),
        structured_model_factory=lambda schema: fake_model,
        audio_transcriber=fake_transcriber,
        sleep_fn=lambda _: None,
    )
    app.init_db()
    app.download_audio_attachment = lambda attachment: tmp_path / "audio.m4a"

    app.process_message(
        {"ts": "2.2", "text": "", "user": "UOTHER", "files": [audio_file()]},
        {"id": "D123", "is_im": True, "user": "UOTHER"},
        my_user_id="UME",
    )

    assert fake_transcriber.calls == [tmp_path / "audio.m4a"]
    assert "Necesito revisar el reporte local." in fake_model.calls[0]
    assert get_audio_rows(app.config.db_path)[0]["transcription_status"] == "local_only"


def test_slack_and_local_audio_transcripts_are_fused(tmp_path):
    fake_model = FakeStructuredModel([
        AudioTranscriptFusion(transcript="Necesito revisar el reporte fusionado."),
        sample_classification(),
    ])
    app = AgentApp(
        make_config(tmp_path),
        slack_client=FakeSlackClient(),
        structured_model_factory=lambda schema: fake_model,
        audio_transcriber=FakeAudioTranscriber(["Necesito revisar reporte local."]),
        sleep_fn=lambda _: None,
    )
    app.init_db()
    app.download_audio_attachment = lambda attachment: tmp_path / "audio.m4a"

    app.process_message(
        {
            "ts": "2.3",
            "text": "",
            "user": "UOTHER",
            "files": [audio_file(transcription_text="Necesito revisar el reporte Slack.")],
        },
        {"id": "D123", "is_im": True, "user": "UOTHER"},
        my_user_id="UME",
    )

    assert "Transcripción de Slack" in fake_model.calls[0]
    assert "Necesito revisar el reporte fusionado." in fake_model.calls[1]
    row = get_audio_rows(app.config.db_path)[0]
    assert row["transcription_status"] == "fused"
    assert row["fused_transcript_text"] == "Necesito revisar el reporte fusionado."


def test_audio_fusion_failure_falls_back_to_local(tmp_path):
    fake_model = FakeStructuredModel([RuntimeError("fusion down"), sample_classification()])
    app = AgentApp(
        make_config(tmp_path),
        slack_client=FakeSlackClient(),
        structured_model_factory=lambda schema: fake_model,
        audio_transcriber=FakeAudioTranscriber(["Texto local más confiable."]),
        sleep_fn=lambda _: None,
    )
    app.init_db()
    app.download_audio_attachment = lambda attachment: tmp_path / "audio.m4a"

    app.process_message(
        {
            "ts": "2.4",
            "text": "",
            "user": "UOTHER",
            "files": [audio_file(transcription_text="Texto Slack.")],
        },
        {"id": "D123", "is_im": True, "user": "UOTHER"},
        my_user_id="UME",
    )

    assert "Texto local más confiable." in fake_model.calls[1]
    row = get_audio_rows(app.config.db_path)[0]
    assert row["transcription_status"] == "local_only"
    assert row["selected_transcript_text"] == "Texto local más confiable."
    assert row["transcription_error"] == "fusion down"


def test_multiple_audio_transcripts_are_concatenated_in_order(tmp_path):
    fake_model = FakeStructuredModel([sample_classification()])
    app = AgentApp(
        make_config(tmp_path, LOCAL_WHISPER_ENABLED="false"),
        slack_client=FakeSlackClient(),
        structured_model_factory=lambda schema: fake_model,
        sleep_fn=lambda _: None,
    )
    app.init_db()

    app.process_message(
        {
            "ts": "2.5",
            "text": "",
            "user": "UOTHER",
            "files": [
                audio_file(id="F1", name="uno.m4a", transcription_text="Primer audio."),
                audio_file(id="F2", name="dos.m4a", transcription_text="Segundo audio."),
            ],
        },
        {"id": "D123", "is_im": True, "user": "UOTHER"},
        my_user_id="UME",
    )

    prompt = fake_model.calls[0]
    assert "[Audio 1]: Primer audio." in prompt
    assert "[Audio 2]: Segundo audio." in prompt
    assert prompt.index("[Audio 1]") < prompt.index("[Audio 2]")


def test_text_and_audio_are_combined_before_classification(tmp_path):
    fake_model = FakeStructuredModel([sample_classification()])
    app = AgentApp(
        make_config(tmp_path, LOCAL_WHISPER_ENABLED="false"),
        slack_client=FakeSlackClient(),
        structured_model_factory=lambda schema: fake_model,
        sleep_fn=lambda _: None,
    )
    app.init_db()

    app.process_message(
        {
            "ts": "2.6",
            "text": "Este es el contexto escrito.",
            "user": "UOTHER",
            "files": [audio_file(transcription_text="Este es el contexto hablado.")],
        },
        {"id": "D123", "is_im": True, "user": "UOTHER"},
        my_user_id="UME",
    )

    prompt = fake_model.calls[0]
    assert "Texto original:" in prompt
    assert "Este es el contexto escrito." in prompt
    assert "Audios transcriptos:" in prompt
    assert "Este es el contexto hablado." in prompt


def test_audio_in_existing_thread_adds_context_without_duplicate(tmp_path):
    fake_model = FakeStructuredModel([sample_classification()])
    fake_slack = FakeSlackClient()
    app = AgentApp(
        make_config(tmp_path, LOCAL_WHISPER_ENABLED="false"),
        slack_client=fake_slack,
        structured_model_factory=lambda schema: fake_model,
        sleep_fn=lambda _: None,
    )
    app.init_db()
    conversation = {"id": "D123", "is_im": True, "user": "UOTHER"}

    app.process_message(
        {"ts": "2.7", "text": "Revisás este reporte?", "user": "UOTHER"},
        conversation,
        my_user_id="UME",
    )
    app.process_message(
        {
            "ts": "2.8",
            "thread_ts": "2.7",
            "text": "",
            "user": "UOTHER",
            "files": [audio_file(transcription_text="Sumo el audio con el detalle.")],
        },
        conversation,
        my_user_id="UME",
    )

    assert count_tasks(app.config.db_path) == 1
    assert len(fake_model.calls) == 1
    assert "*Pedido actualizado:*" in fake_slack.post_calls[1]["text"]
    assert "transcribí el audio" in fake_slack.post_calls[1]["text"]
    audio_row = get_audio_rows(app.config.db_path)[0]
    assert audio_row["task_id"] == 1
    assert audio_row["selected_transcript_text"] == "Sumo el audio con el detalle."


def test_audio_download_failure_is_audited_and_does_not_break(tmp_path):
    fake_model = FakeStructuredModel([])
    app = AgentApp(
        make_config(tmp_path),
        slack_client=FakeSlackClient(),
        structured_model_factory=lambda schema: fake_model,
        audio_transcriber=FakeAudioTranscriber(),
        sleep_fn=lambda _: None,
    )
    app.init_db()
    app.download_audio_attachment = lambda attachment: (_ for _ in ()).throw(RuntimeError("download failed"))

    app.process_message(
        {"ts": "2.9", "text": "", "user": "UOTHER", "files": [audio_file()]},
        {"id": "D123", "is_im": True, "user": "UOTHER"},
        my_user_id="UME",
    )

    row = get_audio_rows(app.config.db_path)[0]
    processed = get_processed_row(app.config.db_path)
    assert row["transcription_status"] == "failed"
    assert row["transcription_error"] == "download failed"
    assert processed["classification_status"] == "ignored"
    assert count_tasks(app.config.db_path) == 0


def test_audio_missing_local_whisper_dependency_is_audited_and_does_not_break(tmp_path):
    fake_model = FakeStructuredModel([])
    app = AgentApp(
        make_config(tmp_path),
        slack_client=FakeSlackClient(),
        structured_model_factory=lambda schema: fake_model,
        audio_transcriber=FakeAudioTranscriber(
            error=RuntimeError("Falta instalar `faster-whisper` u `openai-whisper`.")
        ),
        sleep_fn=lambda _: None,
    )
    app.init_db()
    app.download_audio_attachment = lambda attachment: tmp_path / "audio.m4a"

    app.process_message(
        {"ts": "2.95", "text": "", "user": "UOTHER", "files": [audio_file()]},
        {"id": "D123", "is_im": True, "user": "UOTHER"},
        my_user_id="UME",
    )

    row = get_audio_rows(app.config.db_path)[0]
    processed = get_processed_row(app.config.db_path)
    assert row["transcription_status"] == "failed"
    assert "faster-whisper" in row["transcription_error"]
    assert processed["classification_status"] == "ignored"
    assert count_tasks(app.config.db_path) == 0


def test_detect_visual_attachment_accepts_png(tmp_path):
    app = AgentApp(
        make_config(tmp_path),
        slack_client=FakeSlackClient(),
        structured_model_factory=lambda schema: FakeStructuredModel([]),
        sleep_fn=lambda _: None,
    )

    attachments = app.detect_visual_attachments({"channel": "D1", "ts": "1.0", "files": [image_file()]})

    assert len(attachments) == 1
    assert attachments[0].mime_type == "image/png"


def test_detect_visual_attachment_accepts_jpeg(tmp_path):
    app = AgentApp(
        make_config(tmp_path),
        slack_client=FakeSlackClient(),
        structured_model_factory=lambda schema: FakeStructuredModel([]),
        sleep_fn=lambda _: None,
    )

    attachments = app.detect_visual_attachments(
        {"channel": "D1", "ts": "1.0", "files": [image_file(mimetype="image/jpeg", filetype="jpg", name="foto.jpg")]}
    )

    assert len(attachments) == 1
    assert attachments[0].filename == "foto.jpg"


def test_detect_visual_attachment_ignores_non_images(tmp_path):
    app = AgentApp(
        make_config(tmp_path),
        slack_client=FakeSlackClient(),
        structured_model_factory=lambda schema: FakeStructuredModel([]),
        sleep_fn=lambda _: None,
    )

    attachments = app.detect_visual_attachments(
        {"channel": "D1", "ts": "1.0", "files": [{"id": "DOC1", "name": "archivo.txt", "mimetype": "text/plain"}]}
    )

    assert attachments == []


def test_new_trello_card_uploads_slack_image_attachment(tmp_path):
    fake_model = FakeStructuredModel([sample_classification()])
    fake_trello = FakeTrelloClient()
    app = AgentApp(
        make_config(
            tmp_path,
            TRELLO_ENABLED="true",
            TRELLO_API_KEY="key",
            TRELLO_TOKEN="token",
            TRELLO_LIST_ID="list123",
        ),
        slack_client=FakeSlackClient(),
        structured_model_factory=lambda schema: fake_model,
        trello_client=fake_trello,
        sleep_fn=lambda _: None,
    )
    app.init_db()
    downloaded = tmp_path / "captura.png"
    downloaded.write_bytes(b"png")
    app.download_visual_attachment = lambda attachment: downloaded

    app.process_message(
        {
            "ts": "2.96",
            "text": "Revisás esta captura con el error?",
            "user": "UOTHER",
            "files": [image_file()],
        },
        {"id": "D123", "is_im": True, "user": "UOTHER"},
        my_user_id="UME",
    )

    rows = get_file_rows(app.config.db_path)
    assert len(fake_trello.created_cards) == 1
    assert len(fake_trello.file_attachments) == 1
    assert rows[0]["attachment_status"] == "attached"
    assert rows[0]["trello_card_id"] == "card123"
    assert any("Imagen adjuntada desde Slack: captura.png" in comment["text"] for comment in fake_trello.comments)
    assert downloaded.exists() is False


def test_context_image_attaches_to_existing_trello_card(tmp_path):
    fake_model = FakeStructuredModel([sample_classification()])
    fake_trello = FakeTrelloClient()
    fake_slack = FakeSlackClient()
    app = AgentApp(
        make_config(
            tmp_path,
            TRELLO_ENABLED="true",
            TRELLO_API_KEY="key",
            TRELLO_TOKEN="token",
            TRELLO_LIST_ID="list123",
        ),
        slack_client=fake_slack,
        structured_model_factory=lambda schema: fake_model,
        trello_client=fake_trello,
        sleep_fn=lambda _: None,
    )
    app.init_db()
    downloaded = tmp_path / "captura-contexto.png"
    downloaded.write_bytes(b"png")
    app.download_visual_attachment = lambda attachment: downloaded
    conversation = {"id": "D123", "is_im": True, "user": "UOTHER"}

    app.process_message(
        {"ts": "2.97", "text": "Revisás este error?", "user": "UOTHER"},
        conversation,
        my_user_id="UME",
    )
    app.process_message(
        {
            "ts": "2.98",
            "thread_ts": "2.97",
            "text": "Te dejo la captura.",
            "user": "UOTHER",
            "files": [image_file(id="IMG2", name="detalle.png")],
        },
        conversation,
        my_user_id="UME",
    )

    rows = get_file_rows(app.config.db_path)
    assert count_tasks(app.config.db_path) == 1
    assert len(fake_trello.file_attachments) == 1
    assert rows[0]["attachment_status"] == "attached"
    assert fake_trello.file_attachments[0]["card_id"] == "card123"


def test_visual_attachment_download_failure_is_audited_without_breaking(tmp_path):
    fake_model = FakeStructuredModel([sample_classification()])
    fake_trello = FakeTrelloClient()
    app = AgentApp(
        make_config(
            tmp_path,
            TRELLO_ENABLED="true",
            TRELLO_API_KEY="key",
            TRELLO_TOKEN="token",
            TRELLO_LIST_ID="list123",
        ),
        slack_client=FakeSlackClient(),
        structured_model_factory=lambda schema: fake_model,
        trello_client=fake_trello,
        sleep_fn=lambda _: None,
    )
    app.init_db()
    app.download_visual_attachment = lambda attachment: (_ for _ in ()).throw(RuntimeError("download failed"))

    app.process_message(
        {
            "ts": "2.99",
            "text": "Revisás esta captura con el error?",
            "user": "UOTHER",
            "files": [image_file(id="IMG3")],
        },
        {"id": "D123", "is_im": True, "user": "UOTHER"},
        my_user_id="UME",
    )

    rows = get_file_rows(app.config.db_path)
    assert count_tasks(app.config.db_path) == 1
    assert rows[0]["attachment_status"] == "failed"
    assert rows[0]["attachment_error"] == "download failed"


def test_visual_attachment_trello_upload_failure_is_audited_without_breaking(tmp_path):
    fake_model = FakeStructuredModel([sample_classification()])
    fake_trello = FakeTrelloClient(attachment_error=RuntimeError("trello attach down"))
    app = AgentApp(
        make_config(
            tmp_path,
            TRELLO_ENABLED="true",
            TRELLO_API_KEY="key",
            TRELLO_TOKEN="token",
            TRELLO_LIST_ID="list123",
        ),
        slack_client=FakeSlackClient(),
        structured_model_factory=lambda schema: fake_model,
        trello_client=fake_trello,
        sleep_fn=lambda _: None,
    )
    app.init_db()
    downloaded = tmp_path / "captura-falla.png"
    downloaded.write_bytes(b"png")
    app.download_visual_attachment = lambda attachment: downloaded

    app.process_message(
        {
            "ts": "3.01",
            "text": "Revisás esta captura con el error?",
            "user": "UOTHER",
            "files": [image_file(id="IMG4")],
        },
        {"id": "D123", "is_im": True, "user": "UOTHER"},
        my_user_id="UME",
    )

    rows = get_file_rows(app.config.db_path)
    assert count_tasks(app.config.db_path) == 1
    assert rows[0]["attachment_status"] == "failed"
    assert rows[0]["attachment_error"] == "trello attach down"
    assert any("No pude adjuntar la imagen desde Slack" in comment["text"] for comment in fake_trello.comments)


def test_visual_attachment_sync_does_not_duplicate_existing_attachment(tmp_path):
    fake_model = FakeStructuredModel([sample_classification()])
    fake_trello = FakeTrelloClient()
    app = AgentApp(
        make_config(
            tmp_path,
            TRELLO_ENABLED="true",
            TRELLO_API_KEY="key",
            TRELLO_TOKEN="token",
            TRELLO_LIST_ID="list123",
        ),
        slack_client=FakeSlackClient(),
        structured_model_factory=lambda schema: fake_model,
        trello_client=fake_trello,
        sleep_fn=lambda _: None,
    )
    app.init_db()
    downloaded = tmp_path / "captura-dup.png"
    downloaded.write_bytes(b"png")
    app.download_visual_attachment = lambda attachment: downloaded

    app.process_message(
        {
            "ts": "3.02",
            "text": "Revisás esta captura con el error?",
            "user": "UOTHER",
            "files": [image_file(id="IMG5")],
        },
        {"id": "D123", "is_im": True, "user": "UOTHER"},
        my_user_id="UME",
    )

    assert app.sync_visual_attachments_for_task(1, "card123") == 0
    assert len(fake_trello.file_attachments) == 1


def test_visual_attachment_link_mode_comments_metadata_without_downloading(tmp_path):
    fake_model = FakeStructuredModel([sample_classification()])
    fake_trello = FakeTrelloClient()
    app = AgentApp(
        make_config(
            tmp_path,
            TRELLO_ENABLED="true",
            TRELLO_API_KEY="key",
            TRELLO_TOKEN="token",
            TRELLO_LIST_ID="list123",
            TRELLO_IMAGE_ATTACHMENT_MODE="link",
        ),
        slack_client=FakeSlackClient(),
        structured_model_factory=lambda schema: fake_model,
        trello_client=fake_trello,
        sleep_fn=lambda _: None,
    )
    app.init_db()
    app.download_visual_attachment = lambda attachment: (_ for _ in ()).throw(AssertionError("no debería descargar"))

    app.process_message(
        {
            "ts": "3.03",
            "text": "Revisás esta captura con el error?",
            "user": "UOTHER",
            "files": [image_file(id="IMG6")],
        },
        {"id": "D123", "is_im": True, "user": "UOTHER"},
        my_user_id="UME",
    )

    rows = get_file_rows(app.config.db_path)
    assert rows[0]["attachment_status"] == "linked"
    assert fake_trello.file_attachments == []
    assert any("Imagen recibida desde Slack: captura.png" in comment["text"] for comment in fake_trello.comments)
    assert any("URL privada:" in comment["text"] for comment in fake_trello.comments)


def test_channel_without_mention_is_ignored(tmp_path):
    fake_model = FakeStructuredModel([])
    app = AgentApp(
        make_config(tmp_path),
        slack_client=FakeSlackClient(),
        structured_model_factory=lambda schema: fake_model,
        sleep_fn=lambda _: None,
    )
    app.init_db()

    app.process_message(
        {"ts": "2.0", "text": "Reporte listo para revisar", "user": "UOTHER"},
        {"id": "C123", "name": "general", "is_private": False},
        my_user_id="UME",
    )

    row = get_processed_row(app.config.db_path)
    assert row["classification_status"] == "ignored"
    assert row["relevant"] == 0
    assert fake_model.calls == []
    assert count_tasks(app.config.db_path) == 0


def test_public_channel_is_ignored_even_with_direct_mention(tmp_path):
    fake_model = FakeStructuredModel([])
    app = AgentApp(
        make_config(tmp_path),
        slack_client=FakeSlackClient(),
        structured_model_factory=lambda schema: fake_model,
        sleep_fn=lambda _: None,
    )
    app.init_db()

    app.process_message(
        {"ts": "2.5", "text": "<@UME> revisá esto", "user": "UOTHER"},
        {"id": "C123", "name": "general", "is_private": False},
        my_user_id="UME",
    )

    row = get_processed_row(app.config.db_path)
    assert row["classification_status"] == "ignored"
    assert row["relevance_reason"] == "Los canales públicos están fuera de alcance."
    assert fake_model.calls == []


def test_private_channel_alias_mention_is_relevant(tmp_path):
    fake_model = FakeStructuredModel([sample_classification()])
    app = AgentApp(
        make_config(tmp_path),
        slack_client=FakeSlackClient(),
        structured_model_factory=lambda schema: fake_model,
        sleep_fn=lambda _: None,
    )
    app.init_db()

    app.process_message(
        {"ts": "2.7", "text": "ivan, revisás este número?", "user": "UOTHER"},
        {"id": "G123", "name": "finanzas", "is_private": True},
        my_user_id="UME",
    )

    row = get_processed_row(app.config.db_path)
    assert row["classification_status"] == "done"
    assert row["relevant"] == 1
    assert count_tasks(app.config.db_path) == 1


def test_self_message_is_ignored_when_flag_is_disabled(tmp_path):
    fake_model = FakeStructuredModel([])
    app = AgentApp(
        make_config(tmp_path, INCLUDE_SELF_FOR_TEST="false"),
        slack_client=FakeSlackClient(),
        structured_model_factory=lambda schema: fake_model,
        sleep_fn=lambda _: None,
    )
    app.init_db()

    app.process_message(
        {"ts": "3.0", "text": "<@UME> prueba", "user": "UME"},
        {"id": "C123", "name": "general", "is_private": False},
        my_user_id="UME",
    )

    row = get_processed_row(app.config.db_path)
    assert row["classification_status"] == "ignored"
    assert row["relevance_reason"] == "Mensaje enviado por Ivan."
    assert fake_model.calls == []


def test_self_message_can_be_processed_for_testing(tmp_path):
    fake_model = FakeStructuredModel([sample_classification()])
    app = AgentApp(
        make_config(tmp_path, INCLUDE_SELF_FOR_TEST="true"),
        slack_client=FakeSlackClient(),
        structured_model_factory=lambda schema: fake_model,
        sleep_fn=lambda _: None,
    )
    app.init_db()

    app.process_message(
        {"ts": "4.0", "text": "<@UME> revisar reporte de prueba", "user": "UME"},
        {"id": "G555", "name": "privado-pruebas", "is_private": True},
        my_user_id="UME",
    )

    row = get_processed_row(app.config.db_path)
    assert row["classification_status"] == "done"
    assert row["relevant"] == 1
    assert count_tasks(app.config.db_path) == 1


def test_failed_classification_is_retried_without_losing_message(tmp_path):
    fake_model = FakeStructuredModel([RuntimeError("ollama down"), sample_classification()])
    app = AgentApp(
        make_config(tmp_path),
        slack_client=FakeSlackClient(),
        structured_model_factory=lambda schema: fake_model,
        sleep_fn=lambda _: None,
    )
    app.init_db()

    app.process_message(
        {"ts": "5.0", "text": "Comparás esto con Salesforce?", "user": "UOTHER"},
        {"id": "D123", "is_im": True, "user": "UOTHER"},
        my_user_id="UME",
    )

    row = get_processed_row(app.config.db_path)
    assert row["classification_status"] == "failed"
    assert count_tasks(app.config.db_path) == 0

    recovered = app.retry_failed_messages(limit=10)
    row = get_processed_row(app.config.db_path)
    assert recovered == 1
    assert row["classification_status"] == "done"
    assert count_tasks(app.config.db_path) == 1


def test_trello_failure_marks_task_as_failed(tmp_path):
    fake_model = FakeStructuredModel([sample_classification()])
    fake_trello = FakeTrelloClient(should_fail=True)
    app = AgentApp(
        make_config(
            tmp_path,
            TRELLO_ENABLED="true",
            TRELLO_API_KEY="key",
            TRELLO_TOKEN="token",
            TRELLO_LIST_ID="list123",
        ),
        slack_client=FakeSlackClient(),
        structured_model_factory=lambda schema: fake_model,
        trello_client=fake_trello,
        sleep_fn=lambda _: None,
    )
    app.init_db()

    app.process_message(
        {"ts": "5.1", "text": "Comparás esto con Salesforce?", "user": "UOTHER"},
        {"id": "D123", "is_im": True, "user": "UOTHER"},
        my_user_id="UME",
    )

    with sqlite3.connect(app.config.db_path) as conn:
        conn.row_factory = sqlite3.Row
        task = conn.execute("SELECT trello_status, trello_last_error FROM tasks LIMIT 1").fetchone()
    assert task["trello_status"] == "failed"
    assert "trello down" in task["trello_last_error"]


def test_trello_waiting_comment_sends_question_to_dm_and_dedupes(tmp_path):
    fake_slack = FakeSlackClient()
    fake_trello = FakeTrelloClient(
        card_comments=[
            TrelloComment(
                id="action1",
                text="Pedir: ¿Me pasás el link del registro y el usuario?",
                date="2026-06-11T12:05:00.000Z",
            )
        ]
    )
    app = AgentApp(
        make_config(tmp_path, TRELLO_ENABLED="true"),
        slack_client=fake_slack,
        structured_model_factory=lambda schema: FakeStructuredModel([]),
        trello_client=fake_trello,
        sleep_fn=lambda _: None,
    )
    app.init_db()
    insert_task(
        app,
        created_at="2026-06-11T12:00:00+00:00",
        message_ts="49.0",
        summary="Pedir datos para avanzar",
        public_request_text="Revisar el pedido cuando llegue la información faltante.",
        requested_action="Revisar el pedido cuando llegue la información faltante.",
        priority="medium",
        category="admin",
        classification=make_classification(summary="Pedir datos para avanzar"),
        trello_card_id="card123",
    )

    assert app.sync_trello_waiting_requests() == 1
    assert app.sync_trello_waiting_requests() == 0

    task = get_task(app.config.db_path)
    text = fake_slack.post_calls[0]["text"]
    assert task["status"] == "waiting_for_requester"
    assert task["waiting_trello_action_id"] == "action1"
    assert task["waiting_request_text"] == "¿Me pasás el link del registro y el usuario?"
    assert task["waiting_request_message_ts"] == "50.123456"
    assert "<@UOTHER>" not in text
    assert "¿Me pasás el link del registro y el usuario?" in text
    assert "Pedir:" not in text
    assert len(fake_slack.post_calls) == 1
    assert len(get_task_events(app.config.db_path, "waiting_requested")) == 1


def test_trello_waiting_comment_mentions_requester_in_private_channel(tmp_path):
    fake_slack = FakeSlackClient()
    fake_trello = FakeTrelloClient(
        card_comments=[
            TrelloComment(
                id="action2",
                text="Pedir: Pasame el ID de la oportunidad.",
                date="2026-06-11T12:05:00.000Z",
            )
        ]
    )
    app = AgentApp(
        make_config(tmp_path, TRELLO_ENABLED="true"),
        slack_client=fake_slack,
        structured_model_factory=lambda schema: FakeStructuredModel([]),
        trello_client=fake_trello,
        sleep_fn=lambda _: None,
    )
    app.init_db()
    insert_task(
        app,
        created_at="2026-06-11T12:00:00+00:00",
        message_ts="49.1",
        summary="Pedir ID",
        public_request_text="Revisar una oportunidad cuando llegue el ID.",
        requested_action="Revisar una oportunidad cuando llegue el ID.",
        priority="medium",
        category="salesforce",
        classification=make_classification(summary="Pedir ID"),
        sender_label="Ana Gomez",
        conversation_label="#ventas",
        trello_card_id="card123",
    )
    with sqlite3.connect(app.config.db_path) as conn:
        conn.execute(
            "UPDATE tasks SET channel_id = 'G123', requester_user_id = 'UOTHER' WHERE id = 1"
        )

    assert app.sync_trello_waiting_requests() == 1

    assert fake_slack.post_calls[0]["text"].startswith("<@UOTHER> Para poder avanzar")


def test_trello_reply_comment_sends_exact_text_to_slack(tmp_path):
    fake_slack = FakeSlackClient()
    fake_trello = FakeTrelloClient(
        card_comments=[
            TrelloComment(
                id="reply1",
                text="Responder: Te paso el link correcto: https://example.com",
                date="2026-06-11T12:05:00.000Z",
            )
        ]
    )
    app = AgentApp(
        make_config(tmp_path, TRELLO_ENABLED="true"),
        slack_client=fake_slack,
        structured_model_factory=lambda schema: FakeStructuredModel([]),
        trello_client=fake_trello,
        sleep_fn=lambda _: None,
    )
    app.init_db()
    insert_task(
        app,
        created_at="2026-06-11T12:00:00+00:00",
        message_ts="49.15",
        summary="Mandar link",
        public_request_text="Mandar el link correcto al requester.",
        requested_action="Mandar el link correcto.",
        priority="medium",
        category="admin",
        classification=make_classification(summary="Mandar link"),
        trello_card_id="card123",
    )

    assert app.sync_trello_reply_commands() == 1
    assert fake_slack.post_calls[0]["text"] == "Te paso el link correcto: https://example.com"
    assert get_task(app.config.db_path)["status"] == "new"
    assert fake_trello.comments[-1]["text"] == "Respuesta enviada a Slack."
    actions = get_processed_actions(app.config.db_path)
    assert actions[0]["trello_action_id"] == "reply1"
    assert actions[0]["status"] == "processed"


def test_trello_reply_multiline_preserves_newlines(tmp_path):
    fake_slack = FakeSlackClient()
    fake_trello = FakeTrelloClient(
        card_comments=[
            TrelloComment(
                id="reply2",
                text="Responder:\nTe paso el link correcto: https://example.com\n\nAvisame si te sigue fallando.",
                date="2026-06-11T12:05:00.000Z",
            )
        ]
    )
    app = AgentApp(
        make_config(tmp_path, TRELLO_ENABLED="true"),
        slack_client=fake_slack,
        structured_model_factory=lambda schema: FakeStructuredModel([]),
        trello_client=fake_trello,
        sleep_fn=lambda _: None,
    )
    app.init_db()
    insert_task(
        app,
        created_at="2026-06-11T12:00:00+00:00",
        message_ts="49.16",
        summary="Mandar link",
        requested_action="Mandar link correcto.",
        priority="medium",
        category="admin",
        classification=make_classification(summary="Mandar link"),
        trello_card_id="card123",
    )

    assert app.sync_trello_reply_commands() == 1
    assert fake_slack.post_calls[0]["text"] == (
        "Te paso el link correcto: https://example.com\n\nAvisame si te sigue fallando."
    )


def test_trello_reply_sync_dedupes_processed_comment(tmp_path):
    fake_slack = FakeSlackClient()
    fake_trello = FakeTrelloClient(
        card_comments=[
            TrelloComment(
                id="reply3",
                text="Responder: Ya te lo mandé por acá.",
                date="2026-06-11T12:05:00.000Z",
            )
        ]
    )
    app = AgentApp(
        make_config(tmp_path, TRELLO_ENABLED="true"),
        slack_client=fake_slack,
        structured_model_factory=lambda schema: FakeStructuredModel([]),
        trello_client=fake_trello,
        sleep_fn=lambda _: None,
    )
    app.init_db()
    insert_task(
        app,
        created_at="2026-06-11T12:00:00+00:00",
        message_ts="49.17",
        summary="Responder rápido",
        requested_action="Responder rápido.",
        priority="medium",
        category="communications",
        classification=make_classification(summary="Responder rápido"),
        trello_card_id="card123",
    )

    assert app.sync_trello_reply_commands() == 1
    assert app.sync_trello_reply_commands() == 0
    assert len(fake_slack.post_calls) == 1


def test_trello_reply_can_mark_task_responded_if_enabled(tmp_path):
    fake_slack = FakeSlackClient()
    fake_trello = FakeTrelloClient(
        card_comments=[
            TrelloComment(
                id="reply4",
                text="Responder: Ya quedó enviado.",
                date="2026-06-11T12:05:00.000Z",
            )
        ]
    )
    app = AgentApp(
        make_config(tmp_path, TRELLO_ENABLED="true", TRELLO_REPLY_MARK_RESPONDED="true"),
        slack_client=fake_slack,
        structured_model_factory=lambda schema: FakeStructuredModel([]),
        trello_client=fake_trello,
        sleep_fn=lambda _: None,
    )
    app.init_db()
    insert_task(
        app,
        created_at="2026-06-11T12:00:00+00:00",
        message_ts="49.18",
        summary="Responder rápido",
        requested_action="Responder rápido.",
        priority="medium",
        category="communications",
        classification=make_classification(summary="Responder rápido"),
        trello_card_id="card123",
    )

    assert app.sync_trello_reply_commands() == 1
    task = get_task(app.config.db_path)
    assert task["status"] == "responded"
    assert task["reply_ts"] == "50.123456"


def test_trello_reply_failure_is_audited_and_comments_back_in_trello(tmp_path):
    fake_slack = FakeSlackClient(post_error=RuntimeError("slack down"))
    fake_trello = FakeTrelloClient(
        card_comments=[
            TrelloComment(
                id="reply5",
                text="Responder: Te lo mando ahora.",
                date="2026-06-11T12:05:00.000Z",
            )
        ]
    )
    app = AgentApp(
        make_config(tmp_path, TRELLO_ENABLED="true"),
        slack_client=fake_slack,
        structured_model_factory=lambda schema: FakeStructuredModel([]),
        trello_client=fake_trello,
        sleep_fn=lambda _: None,
    )
    app.init_db()
    insert_task(
        app,
        created_at="2026-06-11T12:00:00+00:00",
        message_ts="49.19",
        summary="Responder rápido",
        requested_action="Responder rápido.",
        priority="medium",
        category="communications",
        classification=make_classification(summary="Responder rápido"),
        trello_card_id="card123",
    )

    assert app.sync_trello_reply_commands() == 0
    task = get_task(app.config.db_path)
    assert task["status"] == "new"
    assert task["trello_last_error"] == "slack down"
    assert any("No pude enviar la respuesta a Slack: slack down" in comment["text"] for comment in fake_trello.comments)
    actions = get_processed_actions(app.config.db_path)
    assert actions[0]["status"] == "failed"


def test_trello_reply_in_channel_does_not_add_automatic_mention(tmp_path):
    fake_slack = FakeSlackClient()
    fake_trello = FakeTrelloClient(
        card_comments=[
            TrelloComment(
                id="reply6",
                text="Responder: Te paso el dato por acá.",
                date="2026-06-11T12:05:00.000Z",
            )
        ]
    )
    app = AgentApp(
        make_config(tmp_path, TRELLO_ENABLED="true"),
        slack_client=fake_slack,
        structured_model_factory=lambda schema: FakeStructuredModel([]),
        trello_client=fake_trello,
        sleep_fn=lambda _: None,
    )
    app.init_db()
    insert_task(
        app,
        created_at="2026-06-11T12:00:00+00:00",
        message_ts="49.20",
        summary="Responder en canal",
        requested_action="Responder en canal.",
        priority="medium",
        category="communications",
        classification=make_classification(summary="Responder en canal"),
        sender_label="Ana Gomez",
        conversation_label="#ventas",
        trello_card_id="card123",
    )
    with sqlite3.connect(app.config.db_path) as conn:
        conn.execute("UPDATE tasks SET channel_id = 'G123', requester_user_id = 'UOTHER' WHERE id = 1")

    assert app.sync_trello_reply_commands() == 1
    assert fake_slack.post_calls[0]["text"] == "Te paso el dato por acá."
    assert "<@UOTHER>" not in fake_slack.post_calls[0]["text"]


def test_waiting_requester_reply_clears_waiting_and_adds_context(tmp_path):
    fake_model = FakeStructuredModel([sample_classification()])
    fake_slack = FakeSlackClient()
    fake_trello = FakeTrelloClient(
        card_comments=[
            TrelloComment(
                id="action3",
                text="Pedir: Pasame el link del registro.",
                date="2026-06-11T12:05:00.000Z",
            )
        ]
    )
    app = AgentApp(
        make_config(tmp_path, TRELLO_ENABLED="true"),
        slack_client=fake_slack,
        structured_model_factory=lambda schema: fake_model,
        trello_client=fake_trello,
        sleep_fn=lambda _: None,
    )
    app.init_db()
    conversation = {"id": "D123", "is_im": True, "user": "UOTHER"}
    app.process_message(
        {"ts": "49.2", "text": "Revisás este registro?", "user": "UOTHER"},
        conversation,
        my_user_id="UME",
    )
    with sqlite3.connect(app.config.db_path) as conn:
        conn.execute("UPDATE tasks SET trello_card_id = 'card123', trello_status = 'created' WHERE id = 1")

    assert app.sync_trello_waiting_requests() == 1
    app.process_message(
        {"ts": "49.3", "thread_ts": "49.2", "text": "Este es el link: https://example.com/registro", "user": "UOTHER"},
        conversation,
        my_user_id="UME",
    )

    task = get_task(app.config.db_path)
    status = sqlite3.connect(app.config.db_path).execute(
        "SELECT classification_status FROM processed_messages WHERE message_ts = '49.3'"
    ).fetchone()[0]
    assert count_tasks(app.config.db_path) == 1
    assert status == "context_added"
    assert task["status"] == "new"
    assert task["waiting_cleared_at"]
    assert any("Respuesta recibida desde Slack: Este es el link" in comment["text"] for comment in fake_trello.comments)
    assert len(get_task_events(app.config.db_path, "waiting_cleared")) == 1
    assert "*Pedido actualizado:*" in fake_slack.post_calls[-1]["text"]


def test_trello_done_ignores_waiting_task_even_when_due_complete(tmp_path):
    fake_slack = FakeSlackClient()
    fake_trello = FakeTrelloClient(
        card_state=TrelloCardState(
            id="card123",
            name="Reporte",
            url="https://trello.com/c/card123",
            list_id="list123",
            list_name="Inbox",
            closed=False,
            due_complete=True,
        )
    )
    app = AgentApp(
        make_config(tmp_path, TRELLO_ENABLED="true", FINAL_REPLY_MODE="slack_auto"),
        slack_client=fake_slack,
        structured_model_factory=lambda schema: FakeStructuredModel([]),
        trello_client=fake_trello,
        sleep_fn=lambda _: None,
    )
    app.init_db()
    insert_task(
        app,
        created_at="2026-06-11T12:00:00+00:00",
        message_ts="49.4",
        summary="Revisar reporte",
        public_request_text="Revisar el reporte cuando llegue la información faltante.",
        requested_action="Revisar reporte",
        priority="medium",
        category="data",
        status="waiting_for_requester",
        classification=make_classification(summary="Revisar reporte"),
        trello_card_id="card123",
    )

    assert app.sync_trello_done_tasks() == 0
    assert get_task(app.config.db_path)["status"] == "waiting_for_requester"
    assert fake_slack.post_calls == []


def test_trello_done_slack_auto_sends_final_reply_without_telegram(tmp_path):
    fake_slack = FakeSlackClient()
    fake_telegram = FakeTelegramClient()
    fake_trello = FakeTrelloClient(
        card_state=TrelloCardState(
            id="card123",
            name="Reporte",
            url="https://trello.com/c/card123",
            list_id="list123",
            list_name="Inbox",
            closed=False,
            due_complete=True,
        )
    )
    public_request = (
        "Modificar los links de formularios en la actualización compartida, verificando cada formulario "
        "y dejando el recorrido listo para el equipo sin recortar este texto largo."
    )
    app = AgentApp(
        make_config(
            tmp_path,
            TRELLO_ENABLED="true",
            FINAL_REPLY_MODE="slack_auto",
            TELEGRAM_ENABLED="true",
            TELEGRAM_BOT_TOKEN="bot-token",
            TELEGRAM_CHAT_ID="123",
        ),
        slack_client=fake_slack,
        structured_model_factory=lambda schema: FakeStructuredModel([]),
        trello_client=fake_trello,
        telegram_client=fake_telegram,
        sleep_fn=lambda _: None,
    )
    app.init_db()
    insert_task(
        app,
        created_at="2026-06-11T12:00:00+00:00",
        message_ts="49.5",
        summary="Luciana Santos pide a Ivan Rodríguez que modifique los links...",
        public_request_text=public_request,
        requested_action="Modificar links",
        priority="medium",
        category="admin",
        classification=make_classification(summary="Luciana Santos pide a Ivan Rodríguez que modifique los links..."),
        trello_card_id="card123",
    )

    assert app.sync_trello_done_tasks() == 1

    task = get_task(app.config.db_path)
    text = fake_slack.post_calls[0]["text"]
    assert task["status"] == "responded"
    assert task["reply_ts"] == "50.123456"
    assert "Listo, ya quedó resuelto." in text
    assert "*Pedido resuelto:*" in text
    assert public_request in text
    assert "Luciana Santos pide a Ivan Rodríguez" not in text
    assert "..." not in text
    assert fake_telegram.sent_messages == []
    assert len(get_task_events(app.config.db_path, "slack_auto_final_reply_sent")) == 1


def test_trello_done_marks_done_pending_reply_and_sends_telegram(tmp_path):
    fake_trello = FakeTrelloClient(
        card_state=TrelloCardState(
            id="card123",
            name="Reporte",
            url="https://trello.com/c/card123",
            list_id="done123",
            list_name="Hecho",
            closed=False,
        )
    )
    fake_telegram = FakeTelegramClient()
    app = AgentApp(
        make_config(
            tmp_path,
            TRELLO_ENABLED="true",
            TRELLO_DONE_MODE="list",
            TRELLO_API_KEY="key",
            TRELLO_TOKEN="token",
            TELEGRAM_ENABLED="true",
            TELEGRAM_BOT_TOKEN="bot-token",
            TELEGRAM_CHAT_ID="123",
        ),
        slack_client=FakeSlackClient(),
        structured_model_factory=lambda schema: FakeStructuredModel([]),
        trello_client=fake_trello,
        telegram_client=fake_telegram,
        sleep_fn=lambda _: None,
    )
    app.init_db()
    insert_task(
        app,
        created_at="2026-06-11T12:00:00+00:00",
        message_ts="50.0",
        summary="Validar reporte mensual",
        requested_action="Comparar el reporte",
        priority="medium",
        category="data",
        classification=make_classification(summary="Validar reporte mensual"),
        trello_card_id="card123",
        trello_card_url="https://trello.com/c/card123",
    )

    synced = app.sync_trello_done_tasks()

    task = get_task(app.config.db_path)
    assert synced == 1
    assert task["status"] == "done_pending_reply"
    assert task["done_pending_reply_at"]
    assert task["final_reply_suggestion"] == (
        "Ana, ya quedó resuelto lo que me pediste sobre Validar reporte mensual."
    )
    assert task["telegram_notified_at"]
    assert "Tarea #1 lista para respuesta final" in fake_telegram.sent_messages[0]
    assert "Solicitante: Ana Gomez" in fake_telegram.sent_messages[0]
    assert "/send 1" in fake_telegram.sent_messages[0]
    assert "/edit 1 texto" in fake_telegram.sent_messages[0]
    assert "/nosend 1" in fake_telegram.sent_messages[0]


def test_trello_done_mode_check_accepts_due_complete(tmp_path):
    fake_slack = FakeSlackClient()
    fake_trello = FakeTrelloClient(
        card_state=TrelloCardState(
            id="card123",
            name="Reporte",
            url="https://trello.com/c/card123",
            list_id="list123",
            list_name="Inbox",
            closed=False,
            due_complete=True,
        )
    )
    app = AgentApp(
        make_config(tmp_path, TRELLO_ENABLED="true", TRELLO_DONE_MODE="check"),
        slack_client=fake_slack,
        structured_model_factory=lambda schema: FakeStructuredModel([]),
        trello_client=fake_trello,
        sleep_fn=lambda _: None,
    )
    app.init_db()
    insert_task(
        app,
        created_at="2026-06-11T12:00:00+00:00",
        message_ts="50.2",
        summary="Validar reporte mensual",
        requested_action="Comparar el reporte",
        priority="medium",
        category="data",
        classification=make_classification(summary="Validar reporte mensual"),
        trello_card_id="card123",
    )

    assert app.sync_trello_done_tasks() == 1

    task = get_task(app.config.db_path)
    assert task["status"] == "done_pending_reply"
    assert fake_slack.post_calls == []


def test_trello_done_mode_check_ignores_incomplete_due(tmp_path):
    fake_trello = FakeTrelloClient(
        card_state=TrelloCardState(
            id="card123",
            name="Reporte",
            url="https://trello.com/c/card123",
            list_id="done123",
            list_name="Hecho",
            closed=False,
            due_complete=False,
        )
    )
    app = AgentApp(
        make_config(tmp_path, TRELLO_ENABLED="true", TRELLO_DONE_MODE="check"),
        slack_client=FakeSlackClient(),
        structured_model_factory=lambda schema: FakeStructuredModel([]),
        trello_client=fake_trello,
        sleep_fn=lambda _: None,
    )
    app.init_db()
    insert_task(
        app,
        created_at="2026-06-11T12:00:00+00:00",
        message_ts="50.3",
        summary="Validar reporte mensual",
        requested_action="Comparar el reporte",
        priority="medium",
        category="data",
        classification=make_classification(summary="Validar reporte mensual"),
        trello_card_id="card123",
    )

    assert app.sync_trello_done_tasks() == 0
    assert get_task(app.config.db_path)["status"] == "new"


def test_trello_done_mode_list_keeps_existing_behavior(tmp_path):
    fake_trello = FakeTrelloClient(
        card_state=TrelloCardState(
            id="card123",
            name="Reporte",
            url="https://trello.com/c/card123",
            list_id="done123",
            list_name="Hecho",
            closed=False,
            due_complete=False,
        )
    )
    app = AgentApp(
        make_config(tmp_path, TRELLO_ENABLED="true", TRELLO_DONE_MODE="list"),
        slack_client=FakeSlackClient(),
        structured_model_factory=lambda schema: FakeStructuredModel([]),
        trello_client=fake_trello,
        sleep_fn=lambda _: None,
    )
    app.init_db()
    insert_task(
        app,
        created_at="2026-06-11T12:00:00+00:00",
        message_ts="50.4",
        summary="Validar reporte mensual",
        requested_action="Comparar el reporte",
        priority="medium",
        category="data",
        classification=make_classification(summary="Validar reporte mensual"),
        trello_card_id="card123",
    )

    assert app.sync_trello_done_tasks() == 1
    assert get_task(app.config.db_path)["status"] == "done_pending_reply"


def test_trello_done_mode_list_or_check_accepts_list_or_check(tmp_path):
    for message_ts, card_state in [
        (
            "50.5",
            TrelloCardState(
                id="card123",
                name="Reporte",
                url="https://trello.com/c/card123",
                list_id="list123",
                list_name="Inbox",
                closed=False,
                due_complete=True,
            ),
        ),
        (
            "50.6",
            TrelloCardState(
                id="card123",
                name="Reporte",
                url="https://trello.com/c/card123",
                list_id="done123",
                list_name="Hecho",
                closed=False,
                due_complete=False,
            ),
        ),
    ]:
        case_tmp_path = tmp_path / message_ts
        case_tmp_path.mkdir()
        fake_trello = FakeTrelloClient(card_state=card_state)
        app = AgentApp(
            make_config(case_tmp_path, TRELLO_ENABLED="true", TRELLO_DONE_MODE="list_or_check"),
            slack_client=FakeSlackClient(),
            structured_model_factory=lambda schema: FakeStructuredModel([]),
            trello_client=fake_trello,
            sleep_fn=lambda _: None,
        )
        app.init_db()
        insert_task(
            app,
            created_at="2026-06-11T12:00:00+00:00",
            message_ts=message_ts,
            summary="Validar reporte mensual",
            requested_action="Comparar el reporte",
            priority="medium",
            category="data",
            classification=make_classification(summary="Validar reporte mensual"),
            trello_card_id="card123",
        )

        assert app.sync_trello_done_tasks() == 1
        assert get_task(app.config.db_path)["status"] == "done_pending_reply"


def test_trello_done_mode_checklist_uses_configured_item(tmp_path):
    fake_trello = FakeTrelloClient(
        card_state=TrelloCardState(
            id="card123",
            name="Reporte",
            url="https://trello.com/c/card123",
            list_id="list123",
            list_name="Inbox",
            closed=False,
            checklist_items=(
                {"name": "Hecho", "state": "complete"},
                {"name": "Avisar", "state": "incomplete"},
            ),
        )
    )
    app = AgentApp(
        make_config(tmp_path, TRELLO_ENABLED="true", TRELLO_DONE_MODE="checklist"),
        slack_client=FakeSlackClient(),
        structured_model_factory=lambda schema: FakeStructuredModel([]),
        trello_client=fake_trello,
        sleep_fn=lambda _: None,
    )
    app.init_db()
    insert_task(
        app,
        created_at="2026-06-11T12:00:00+00:00",
        message_ts="50.7",
        summary="Validar reporte mensual",
        requested_action="Comparar el reporte",
        priority="medium",
        category="data",
        classification=make_classification(summary="Validar reporte mensual"),
        trello_card_id="card123",
    )

    assert app.sync_trello_done_tasks() == 1
    assert get_task(app.config.db_path)["status"] == "done_pending_reply"


def test_trello_done_mode_checklist_ignores_incomplete_item(tmp_path):
    fake_trello = FakeTrelloClient(
        card_state=TrelloCardState(
            id="card123",
            name="Reporte",
            url="https://trello.com/c/card123",
            list_id="list123",
            list_name="Inbox",
            closed=False,
            checklist_items=({"name": "Hecho", "state": "incomplete"},),
        )
    )
    app = AgentApp(
        make_config(tmp_path, TRELLO_ENABLED="true", TRELLO_DONE_MODE="checklist"),
        slack_client=FakeSlackClient(),
        structured_model_factory=lambda schema: FakeStructuredModel([]),
        trello_client=fake_trello,
        sleep_fn=lambda _: None,
    )
    app.init_db()
    insert_task(
        app,
        created_at="2026-06-11T12:00:00+00:00",
        message_ts="50.8",
        summary="Validar reporte mensual",
        requested_action="Comparar el reporte",
        priority="medium",
        category="data",
        classification=make_classification(summary="Validar reporte mensual"),
        trello_card_id="card123",
    )

    assert app.sync_trello_done_tasks() == 0
    assert get_task(app.config.db_path)["status"] == "new"


def test_trello_client_get_card_reads_due_complete_and_checklists(monkeypatch):
    captured = {}

    class FakeResponse:
        status_code = 200

        def json(self):
            return {
                "id": "card123",
                "name": "Reporte",
                "url": "https://trello.com/c/card123",
                "idList": "list123",
                "closed": False,
                "dueComplete": True,
                "list": {"name": "Inbox"},
                "checklists": [
                    {
                        "checkItems": [
                            {"name": "Hecho", "state": "complete"},
                            {"name": "Avisar", "state": "incomplete"},
                        ]
                    }
                ],
            }

    def fake_request(method, url, **kwargs):
        captured["params"] = kwargs["params"]
        return FakeResponse()

    monkeypatch.setattr(trello_client.requests, "request", fake_request)

    card = TrelloClient("key", "token").get_card("card123")

    assert captured["params"]["fields"] == "name,url,idList,closed,dueComplete"
    assert captured["params"]["checklists"] == "all"
    assert card.due_complete is True
    assert card.checklist_items == (
        {"name": "Hecho", "state": "complete"},
        {"name": "Avisar", "state": "incomplete"},
    )


def test_trello_client_get_card_comments_reads_comment_actions(monkeypatch):
    captured = {}

    class FakeResponse:
        status_code = 200

        def json(self):
            return [
                {
                    "id": "action1",
                    "date": "2026-06-11T12:05:00.000Z",
                    "idMemberCreator": "member1",
                    "data": {"text": "Pedir: Pasame el link."},
                    "memberCreator": {"username": "ivan", "fullName": "Ivan Rodriguez"},
                }
            ]

    def fake_request(method, url, **kwargs):
        captured["url"] = url
        captured["params"] = kwargs["params"]
        return FakeResponse()

    monkeypatch.setattr(trello_client.requests, "request", fake_request)

    comments = TrelloClient("key", "token").get_card_comments("card123", limit=5)

    assert captured["url"].endswith("/cards/card123/actions")
    assert captured["params"]["filter"] == "commentCard"
    assert captured["params"]["limit"] == "5"
    assert comments == [
        TrelloComment(
            id="action1",
            text="Pedir: Pasame el link.",
            date="2026-06-11T12:05:00.000Z",
            member_id="member1",
            member_username="ivan",
            member_full_name="Ivan Rodriguez",
        )
    ]


def test_trello_done_channel_final_reply_uses_mention(tmp_path):
    fake_trello = FakeTrelloClient(
        card_state=TrelloCardState(
            id="card123",
            name="Reporte",
            url="https://trello.com/c/card123",
            list_id="done123",
            list_name="Hecho",
            closed=False,
        )
    )
    app = AgentApp(
        make_config(
            tmp_path,
            TRELLO_ENABLED="true",
            TRELLO_DONE_MODE="list",
            TRELLO_API_KEY="key",
            TRELLO_TOKEN="token",
            TELEGRAM_ENABLED="true",
            TELEGRAM_BOT_TOKEN="bot-token",
            TELEGRAM_CHAT_ID="123",
        ),
        slack_client=FakeSlackClient(),
        structured_model_factory=lambda schema: FakeStructuredModel([]),
        trello_client=fake_trello,
        telegram_client=FakeTelegramClient(),
        sleep_fn=lambda _: None,
    )
    app.init_db()
    insert_task(
        app,
        created_at="2026-06-11T12:00:00+00:00",
        message_ts="50.1",
        summary="Validar forecast",
        requested_action="Comparar forecast",
        priority="medium",
        category="data",
        classification=make_classification(summary="Validar forecast"),
        sender_label="Ana Gomez",
        conversation_label="#finanzas",
        trello_card_id="card123",
        trello_card_url="https://trello.com/c/card123",
    )
    with sqlite3.connect(app.config.db_path) as conn:
        conn.execute(
            """
            UPDATE tasks
            SET channel_id = 'G123',
                requester_user_id = 'UOTHER',
                requester_label = 'Ana Gomez'
            WHERE id = 1
            """
        )

    app.sync_trello_done_tasks()

    task = get_task(app.config.db_path)
    assert task["final_reply_suggestion"] == (
        "<@UOTHER>, ya quedó resuelto lo que me pediste sobre Validar forecast."
    )


def test_telegram_edit_updates_manual_reply(tmp_path):
    app = AgentApp(
        make_config(tmp_path),
        slack_client=FakeSlackClient(),
        structured_model_factory=lambda schema: FakeStructuredModel([]),
        sleep_fn=lambda _: None,
    )
    app.init_db()
    insert_task(
        app,
        created_at="2026-06-11T12:00:00+00:00",
        message_ts="51.0",
        summary="Responder cierre",
        requested_action="Cerrar pedido",
        priority="medium",
        category="communications",
        classification=make_classification(summary="Responder cierre"),
    )
    app.mark_task_done_pending_reply(1, "Ana, ya quedó resuelto.")

    assert app.handle_telegram_command("/edit 1 Quedó resuelto, gracias por avisar.")

    task = get_task(app.config.db_path)
    assert task["manual_reply"] == "Quedó resuelto, gracias por avisar."


def test_telegram_send_sends_approved_final_reply_to_slack(tmp_path):
    fake_slack = FakeSlackClient()
    app = AgentApp(
        make_config(tmp_path),
        slack_client=fake_slack,
        structured_model_factory=lambda schema: FakeStructuredModel([]),
        sleep_fn=lambda _: None,
    )
    app.init_db()
    insert_task(
        app,
        created_at="2026-06-11T12:00:00+00:00",
        message_ts="52.0",
        summary="Responder cierre",
        requested_action="Cerrar pedido",
        priority="medium",
        category="communications",
        classification=make_classification(summary="Responder cierre"),
    )
    app.mark_task_done_pending_reply(1, "Ana, ya quedó resuelto.")
    app.edit_task_manual_reply(1, "Listo, quedó resuelto.")

    assert app.handle_telegram_command("/send 1")

    task = get_task(app.config.db_path)
    assert task["status"] == "responded"
    assert task["reply_sent_at"]
    assert fake_slack.post_calls == [
        {
            "channel": "D52.0",
            "thread_ts": "52.0",
            "text": "Listo, quedó resuelto.",
            "unfurl_links": False,
            "unfurl_media": False,
        }
    ]


def test_telegram_nosend_marks_done_without_slack_reply(tmp_path):
    fake_slack = FakeSlackClient()
    app = AgentApp(
        make_config(tmp_path),
        slack_client=fake_slack,
        structured_model_factory=lambda schema: FakeStructuredModel([]),
        sleep_fn=lambda _: None,
    )
    app.init_db()
    insert_task(
        app,
        created_at="2026-06-11T12:00:00+00:00",
        message_ts="53.0",
        summary="Responder cierre",
        requested_action="Cerrar pedido",
        priority="medium",
        category="communications",
        classification=make_classification(summary="Responder cierre"),
    )
    app.mark_task_done_pending_reply(1, "Ana, ya quedó resuelto.")

    assert app.handle_telegram_command("/nosend 1")

    task = get_task(app.config.db_path)
    assert task["status"] == "done"
    assert fake_slack.post_calls == []


def test_telegram_failure_is_saved_without_breaking_done_sync(tmp_path):
    fake_trello = FakeTrelloClient(
        card_state=TrelloCardState(
            id="card123",
            name="Reporte",
            url="https://trello.com/c/card123",
            list_id="done123",
            list_name="Hecho",
            closed=False,
        )
    )
    app = AgentApp(
        make_config(
            tmp_path,
            TRELLO_ENABLED="true",
            TRELLO_DONE_MODE="list",
            TRELLO_API_KEY="key",
            TRELLO_TOKEN="token",
            TELEGRAM_ENABLED="true",
            TELEGRAM_BOT_TOKEN="bot-token",
            TELEGRAM_CHAT_ID="123",
        ),
        slack_client=FakeSlackClient(),
        structured_model_factory=lambda schema: FakeStructuredModel([]),
        trello_client=fake_trello,
        telegram_client=FakeTelegramClient(send_error=RuntimeError("telegram down")),
        sleep_fn=lambda _: None,
    )
    app.init_db()
    insert_task(
        app,
        created_at="2026-06-11T12:00:00+00:00",
        message_ts="54.0",
        summary="Validar reporte mensual",
        requested_action="Comparar el reporte",
        priority="medium",
        category="data",
        classification=make_classification(summary="Validar reporte mensual"),
        trello_card_id="card123",
        trello_card_url="https://trello.com/c/card123",
    )

    assert app.sync_trello_done_tasks() == 1

    task = get_task(app.config.db_path)
    assert task["status"] == "done_pending_reply"
    assert task["telegram_notified_at"] is None
    assert task["telegram_error"] == "telegram down"


def test_slack_ack_failure_is_saved_without_breaking_processing(tmp_path):
    fake_model = FakeStructuredModel([sample_classification()])
    fake_slack = FakeSlackClient(post_error=RuntimeError("slack down"))
    app = AgentApp(
        make_config(tmp_path),
        slack_client=fake_slack,
        structured_model_factory=lambda schema: fake_model,
        sleep_fn=lambda _: None,
    )
    app.init_db()

    app.process_message(
        {"ts": "55.0", "text": "Revisás este reporte?", "user": "UOTHER"},
        {"id": "D123", "is_im": True, "user": "UOTHER"},
        my_user_id="UME",
    )

    task = get_task(app.config.db_path)
    row = get_processed_row(app.config.db_path)
    assert row["classification_status"] == "done"
    assert task["acknowledged_at"] is None
    assert task["ack_error"] == "slack down"


def test_trello_client_can_be_built_without_list_id_for_listing(tmp_path):
    fake_trello = FakeTrelloClient()
    app = AgentApp(
        make_config(
            tmp_path,
            TRELLO_ENABLED="true",
            TRELLO_API_KEY="abcd1234abcd1234abcd1234abcd1234",
            TRELLO_TOKEN="token",
        ),
        slack_client=FakeSlackClient(),
        structured_model_factory=lambda schema: FakeStructuredModel([sample_classification()]),
        trello_client=fake_trello,
        sleep_fn=lambda _: None,
    )
    assert app.get_trello_client() is fake_trello


def test_relevant_message_prompt_includes_recent_context(tmp_path):
    fake_slack = FakeSlackClient()
    fake_slack.history_messages = [
        {"ts": "10.0", "text": "Necesitamos validar el reporte antes de las 5", "user": "UOTHER"},
        {"ts": "11.0", "text": "Yo veo la parte de Salesforce", "user": "UME"},
    ]
    fake_model = FakeStructuredModel([sample_classification()])
    app = AgentApp(
        make_config(tmp_path),
        slack_client=fake_slack,
        structured_model_factory=lambda schema: fake_model,
        sleep_fn=lambda _: None,
    )
    app.init_db()

    app.process_message(
        {"ts": "12.0", "text": "ivo, podés mirar el número final?", "user": "UOTHER"},
        {"id": "G123", "name": "finanzas", "is_private": True},
        my_user_id="UME",
    )

    prompt = fake_model.calls[0]
    assert "Contexto reciente de la conversación" in prompt
    assert "Necesitamos validar el reporte antes de las 5" in prompt
    assert "Yo veo la parte de Salesforce" in prompt


def test_list_conversations_only_requests_private_and_dm_types(tmp_path):
    fake_slack = FakeSlackClient()
    app = AgentApp(
        make_config(tmp_path),
        slack_client=fake_slack,
        structured_model_factory=lambda schema: FakeStructuredModel([sample_classification()]),
        sleep_fn=lambda _: None,
    )

    app.list_conversations()
    assert fake_slack.list_calls[0]["types"] == "private_channel,im,mpim"


def test_config_defaults_to_local_provider_and_light_model(tmp_path):
    config = make_config(tmp_path)
    assert config.model_provider == "ollama"
    assert config.ollama_model == "qwen3:4b-instruct"


def test_groq_provider_can_be_selected(tmp_path):
    config = make_config(
        tmp_path,
        MODEL_PROVIDER="groq",
        GROQ_API_KEY="groq-test-key",
        GROQ_MODEL="openai/gpt-oss-20b",
    )
    assert config.model_provider == "groq"
    assert config.groq_api_key == "groq-test-key"
    assert config.groq_model == "openai/gpt-oss-20b"


def test_custom_aliases_are_loaded_from_env(tmp_path):
    config = make_config(tmp_path, MY_MENTION_ALIASES="ivan, ivo, ivo rodriguez")
    assert config.my_mention_aliases == ("ivan", "ivo", "ivo rodriguez")


def test_reply_send_mode_defaults_to_safe_false(tmp_path):
    config = make_config(tmp_path)
    assert config.slack_send_approved_replies is False


def test_reply_send_mode_can_be_enabled_from_env(tmp_path):
    config = make_config(tmp_path, SLACK_SEND_APPROVED_REPLIES="true")
    assert config.slack_send_approved_replies is True


def test_case_grouping_and_sync_worker_config_defaults(tmp_path):
    config = make_config(tmp_path)
    assert config.case_grouping_window_minutes == 15
    assert config.final_reply_mode == "telegram_approval"
    assert config.trello_waiting_enabled is True
    assert config.trello_waiting_comment_prefix == "Pedir:"
    assert config.trello_waiting_auto_clear is True
    assert config.trello_reply_enabled is True
    assert config.trello_reply_comment_prefix == "Responder:"
    assert config.trello_reply_mark_responded is False
    assert config.slack_image_attachments_enabled is True
    assert config.trello_attach_slack_images is True
    assert config.trello_image_attachment_mode == "upload"
    assert config.sync_worker_seconds == 60
    assert config.sync_waiting_enabled is True
    assert config.sync_trello_done_enabled is True
    assert config.sync_telegram_poll_enabled is True


def test_doctor_warns_about_slack_chat_write_scope(tmp_path, capsys):
    app = AgentApp(
        make_config(tmp_path, LOCAL_WHISPER_ENABLED="false"),
        slack_client=FakeSlackClient(),
        structured_model_factory=lambda schema: FakeStructuredModel([DoctorResult()]),
        sleep_fn=lambda _: None,
    )

    assert app.doctor() is True

    output = capsys.readouterr().out
    assert "chat.postMessage" in output
    assert "chat:write" in output
    assert "chat:write:bot" in output


def test_doctor_warns_when_slack_auto_final_replies_are_enabled(tmp_path, capsys):
    app = AgentApp(
        make_config(tmp_path, LOCAL_WHISPER_ENABLED="false", FINAL_REPLY_MODE="slack_auto"),
        slack_client=FakeSlackClient(),
        structured_model_factory=lambda schema: FakeStructuredModel([DoctorResult()]),
        sleep_fn=lambda _: None,
    )

    assert app.doctor() is True

    output = capsys.readouterr().out
    assert "FINAL_REPLY_MODE=slack_auto" in output
    assert "directo a Slack" in output


def test_doctor_warns_about_slack_files_read_for_visual_attachments(tmp_path, capsys):
    app = AgentApp(
        make_config(tmp_path, LOCAL_WHISPER_ENABLED="false", TRELLO_ENABLED="false"),
        slack_client=FakeSlackClient(),
        structured_model_factory=lambda schema: FakeStructuredModel([DoctorResult()]),
        sleep_fn=lambda _: None,
    )

    assert app.doctor() is True

    output = capsys.readouterr().out
    assert "files:read" in output
    assert "imágenes de Slack" in output


def test_doctor_warns_when_local_whisper_enabled_without_backend(tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(slack_personal_agent, "detect_local_whisper_backend", lambda: "")
    app = AgentApp(
        make_config(tmp_path, LOCAL_WHISPER_ENABLED="true"),
        slack_client=FakeSlackClient(),
        structured_model_factory=lambda schema: FakeStructuredModel([DoctorResult()]),
        sleep_fn=lambda _: None,
    )

    assert app.doctor() is False

    output = capsys.readouterr().out
    assert "LOCAL_WHISPER_ENABLED=true" in output
    assert "faster-whisper" in output


def test_atlassian_token_is_rejected_as_trello_api_key():
    error = validate_trello_api_key_format("ATATT3xFfGF0Zvzy-example-token")
    assert error is not None
    assert "Atlassian Account" in error


def test_trello_auth_url_rejects_wrong_key_type(tmp_path):
    app = AgentApp(
        make_config(tmp_path, TRELLO_API_KEY="ATATT3xFfGF0Zvzy-example-token"),
        slack_client=FakeSlackClient(),
        structured_model_factory=lambda schema: FakeStructuredModel([sample_classification()]),
        sleep_fn=lambda _: None,
    )
    try:
        app.trello_auth_url()
    except Exception as exc:
        assert "Atlassian Account" in str(exc)
    else:
        raise AssertionError("Se esperaba error por key inválida.")


def test_large_local_model_gets_warning():
    level, hint = local_model_fit_hint("gpt-oss:20b")
    assert level == "warning"
    assert "14 GB" in hint


def test_main_parser_accepts_brief_command():
    args = build_parser().parse_args(["brief", "--limit", "3"])
    assert args.command == "brief"
    assert args.limit == 3


def test_main_parser_accepts_review_command():
    args = build_parser().parse_args(["review", "--limit", "2"])
    assert args.command == "review"
    assert args.limit == 2


def test_main_parser_accepts_approve_reply_command():
    args = build_parser().parse_args(["approve-reply", "12", "--send"])
    assert args.command == "approve-reply"
    assert args.task_id == "12"
    assert args.send


def test_main_parser_accepts_review_send_mode():
    args = build_parser().parse_args(["review", "--send-approved-replies"])
    assert args.command == "review"
    assert args.send_approved_replies


def test_main_parser_accepts_transcribe_audio_command():
    args = build_parser().parse_args(["transcribe-audio", "/tmp/audio.m4a"])
    assert args.command == "transcribe-audio"
    assert args.task_id == "/tmp/audio.m4a"


def test_main_parser_accepts_trello_waiting_sync_command():
    args = build_parser().parse_args(["trello-waiting-sync", "--limit", "7"])
    assert args.command == "trello-waiting-sync"
    assert args.limit == 7


def test_main_parser_accepts_trello_reply_sync_command():
    args = build_parser().parse_args(["trello-reply-sync", "--limit", "9"])
    assert args.command == "trello-reply-sync"
    assert args.limit == 9


def test_transcribe_audio_path_uses_injected_transcriber(tmp_path):
    fake_transcriber = FakeAudioTranscriber(["Texto transcripto local."])
    app = AgentApp(
        make_config(tmp_path),
        slack_client=FakeSlackClient(),
        structured_model_factory=lambda schema: FakeStructuredModel([]),
        audio_transcriber=fake_transcriber,
        sleep_fn=lambda _: None,
    )
    audio_path = tmp_path / "audio.m4a"

    assert app.transcribe_audio_path(audio_path) == "Texto transcripto local."
    assert fake_transcriber.calls == [audio_path]


def test_parse_snooze_until_accepts_relative_values():
    now = datetime(2026, 6, 11, 15, 0, tzinfo=timezone.utc)
    assert parse_snooze_until("4h", now=now) == datetime(2026, 6, 11, 19, 0, tzinfo=timezone.utc)
    assert parse_snooze_until("2d", now=now) == datetime(2026, 6, 13, 15, 0, tzinfo=timezone.utc)
    assert parse_snooze_until("mañana", now=now) == datetime(2026, 6, 12, 15, 0, tzinfo=timezone.utc)


def test_brief_groups_tasks_by_operational_signals(tmp_path):
    app = AgentApp(
        make_config(tmp_path),
        slack_client=FakeSlackClient(),
        structured_model_factory=lambda schema: FakeStructuredModel([]),
        sleep_fn=lambda _: None,
    )
    app.init_db()
    now = datetime(2026, 6, 11, 15, 0, tzinfo=timezone.utc)

    insert_task(
        app,
        created_at="2026-06-11T12:00:00+00:00",
        message_ts="30.0",
        summary="Validar oportunidades contra Salesforce",
        requested_action="Comparar oportunidades y responder",
        priority="urgent",
        category="salesforce",
        classification=make_classification(
            summary="Validar oportunidades contra Salesforce",
            requested_action="Comparar oportunidades y responder",
            priority="urgent",
            category="salesforce",
            needs_reply=True,
            external_systems=["salesforce"],
            draft_reply="Lo reviso hoy y te confirmo.",
        ),
    )
    insert_task(
        app,
        created_at="2026-06-07T12:00:00+00:00",
        message_ts="31.0",
        summary="Revisar issue de sincronización",
        requested_action="Mirar error de deploy",
        priority="high",
        category="software",
        classification=make_classification(
            summary="Revisar issue de sincronización",
            requested_action="Mirar error de deploy",
            priority="high",
            category="software",
            missing_information=["Falta link al log"],
            external_systems=["github"],
            suggested_next_step="Pedir el log del deploy.",
        ),
    )
    insert_task(
        app,
        created_at="2026-06-07T10:00:00+00:00",
        message_ts="32.0",
        summary="Ordenar agenda de reunión",
        requested_action="Confirmar próximos pasos",
        priority="medium",
        category="meeting",
        classification=make_classification(
            summary="Ordenar agenda de reunión",
            requested_action="Confirmar próximos pasos",
            priority="medium",
            category="meeting",
            needs_reply=False,
            external_systems=[],
        ),
    )

    brief = app.build_brief(section_limit=5, now=now)

    assert "Resumen del agente" in brief
    assert "Tareas abiertas: 3" in brief
    assert "Nuevas tareas detectadas hoy" in brief
    assert "Validar oportunidades contra Salesforce" in brief
    assert "Prioridad alta/urgente" in brief
    assert "Tareas bloqueadas" in brief
    assert "Falta link al log" in brief
    assert "Necesitan respuesta" in brief
    assert "Lo reviso hoy y te confirmo." in brief
    assert "Relacionadas con Salesforce" in brief
    assert "Relacionadas con software" in brief
    assert "Viejas sin mover" in brief
    assert "Decidir si #2 sigue abierta" in brief


def test_brief_handles_empty_task_list(tmp_path):
    app = AgentApp(
        make_config(tmp_path),
        slack_client=FakeSlackClient(),
        structured_model_factory=lambda schema: FakeStructuredModel([]),
        sleep_fn=lambda _: None,
    )

    brief = app.build_brief(now=datetime(2026, 6, 11, 15, 0, tzinfo=timezone.utc))

    assert "Tareas abiertas: 0" in brief
    assert "No hay tareas abiertas en SQLite." in brief
    assert "Sin tareas nuevas hoy." in brief


def test_review_can_approve_existing_draft(tmp_path):
    fake_slack = FakeSlackClient()
    app = AgentApp(
        make_config(tmp_path),
        slack_client=fake_slack,
        structured_model_factory=lambda schema: FakeStructuredModel([]),
        input_fn=lambda prompt: "a",
        sleep_fn=lambda _: None,
    )
    app.init_db()
    insert_task(
        app,
        created_at="2026-06-11T12:00:00+00:00",
        message_ts="40.0",
        summary="Responder a Lara",
        requested_action="Confirmar revisión",
        priority="medium",
        category="communications",
        classification=make_classification(
            summary="Responder a Lara",
            requested_action="Confirmar revisión",
            category="communications",
            draft_reply="Lo reviso y te aviso.",
        ),
    )

    reviewed = app.review_tasks(limit=1)

    with sqlite3.connect(app.config.db_path) as conn:
        conn.row_factory = sqlite3.Row
        task = conn.execute(
            "SELECT status, reply_approved_at, manual_reply FROM tasks LIMIT 1"
        ).fetchone()
    assert reviewed == 1
    assert task["status"] == "reply_approved"
    assert task["reply_approved_at"]
    assert task["manual_reply"] == "Lo reviso y te aviso."
    assert fake_slack.post_calls == []


def test_review_can_approve_and_send_reply_when_explicitly_enabled(tmp_path):
    fake_slack = FakeSlackClient()
    app = AgentApp(
        make_config(tmp_path),
        slack_client=fake_slack,
        structured_model_factory=lambda schema: FakeStructuredModel([]),
        input_fn=lambda prompt: "a",
        sleep_fn=lambda _: None,
    )
    app.init_db()
    insert_task(
        app,
        created_at="2026-06-11T12:00:00+00:00",
        message_ts="40.5",
        summary="Responder a Lara",
        requested_action="Confirmar revisión",
        priority="medium",
        category="communications",
        classification=make_classification(
            summary="Responder a Lara",
            requested_action="Confirmar revisión",
            category="communications",
            draft_reply="Lo reviso y te aviso.",
        ),
    )

    reviewed = app.review_tasks(limit=1, send_replies=True)

    with sqlite3.connect(app.config.db_path) as conn:
        conn.row_factory = sqlite3.Row
        task = conn.execute(
            "SELECT status, reply_approved_at, reply_sent_at, reply_ts, manual_reply FROM tasks LIMIT 1"
        ).fetchone()
        events = conn.execute(
            "SELECT event_type FROM task_events WHERE task_id = 1 ORDER BY id ASC"
        ).fetchall()

    assert reviewed == 1
    assert task["status"] == "responded"
    assert task["reply_approved_at"]
    assert task["reply_sent_at"]
    assert task["reply_ts"] == "50.123456"
    assert task["manual_reply"] == "Lo reviso y te aviso."
    assert fake_slack.post_calls == [
        {
            "channel": "D40.5",
            "thread_ts": "40.5",
            "text": "Lo reviso y te aviso.",
            "unfurl_links": False,
            "unfurl_media": False,
        }
    ]
    assert [event["event_type"] for event in events] == ["reply_approved", "reply_sent"]


def test_approve_reply_can_send_existing_approved_reply_by_task_id(tmp_path):
    fake_slack = FakeSlackClient()
    app = AgentApp(
        make_config(tmp_path),
        slack_client=fake_slack,
        structured_model_factory=lambda schema: FakeStructuredModel([]),
        sleep_fn=lambda _: None,
    )
    app.init_db()
    insert_task(
        app,
        created_at="2026-06-11T12:00:00+00:00",
        message_ts="40.7",
        summary="Responder a Lara",
        requested_action="Confirmar revisión",
        priority="medium",
        category="communications",
        classification=make_classification(
            summary="Responder a Lara",
            requested_action="Confirmar revisión",
            category="communications",
            draft_reply="Borrador inicial.",
        ),
    )
    app.mark_task_reply_approved(1, "Respuesta ya aprobada.")

    assert app.approve_reply(1, send=True)

    with sqlite3.connect(app.config.db_path) as conn:
        conn.row_factory = sqlite3.Row
        task = conn.execute("SELECT status, reply_sent_at, reply_ts FROM tasks LIMIT 1").fetchone()

    assert task["status"] == "responded"
    assert task["reply_sent_at"]
    assert task["reply_ts"] == "50.123456"
    assert fake_slack.post_calls[0]["text"] == "Respuesta ya aprobada."


def test_approve_reply_records_slack_error_and_keeps_task_pending_send(tmp_path):
    fake_slack = FakeSlackClient(post_error=RuntimeError("missing_scope"))
    app = AgentApp(
        make_config(tmp_path),
        slack_client=fake_slack,
        structured_model_factory=lambda schema: FakeStructuredModel([]),
        sleep_fn=lambda _: None,
    )
    app.init_db()
    insert_task(
        app,
        created_at="2026-06-11T12:00:00+00:00",
        message_ts="40.8",
        summary="Responder a Lara",
        requested_action="Confirmar revisión",
        priority="medium",
        category="communications",
        classification=make_classification(
            summary="Responder a Lara",
            requested_action="Confirmar revisión",
            category="communications",
            draft_reply="Lo reviso y te aviso.",
        ),
    )

    assert app.approve_reply(1, send=True) is False

    with sqlite3.connect(app.config.db_path) as conn:
        conn.row_factory = sqlite3.Row
        task = conn.execute(
            "SELECT status, reply_sent_at, reply_error, manual_reply FROM tasks LIMIT 1"
        ).fetchone()
        events = conn.execute(
            "SELECT event_type FROM task_events WHERE task_id = 1 ORDER BY id ASC"
        ).fetchall()

    assert task["status"] == "reply_approved"
    assert task["reply_sent_at"] is None
    assert task["reply_error"] == "missing_scope"
    assert task["manual_reply"] == "Lo reviso y te aviso."
    assert fake_slack.post_calls[0]["thread_ts"] == "40.8"
    assert [event["event_type"] for event in events] == ["reply_approved", "reply_failed"]


def test_review_can_send_reply_when_enabled_from_env(tmp_path):
    fake_slack = FakeSlackClient()
    app = AgentApp(
        make_config(tmp_path, SLACK_SEND_APPROVED_REPLIES="true"),
        slack_client=fake_slack,
        structured_model_factory=lambda schema: FakeStructuredModel([]),
        input_fn=lambda prompt: "a",
        sleep_fn=lambda _: None,
    )
    app.init_db()
    insert_task(
        app,
        created_at="2026-06-11T12:00:00+00:00",
        message_ts="40.9",
        summary="Responder a Lara",
        requested_action="Confirmar revisión",
        priority="medium",
        category="communications",
        classification=make_classification(
            summary="Responder a Lara",
            requested_action="Confirmar revisión",
            category="communications",
            draft_reply="Lo reviso y te aviso.",
        ),
    )

    reviewed = app.review_tasks(limit=1)

    with sqlite3.connect(app.config.db_path) as conn:
        conn.row_factory = sqlite3.Row
        task = conn.execute(
            "SELECT status, reply_sent_at, reply_ts FROM tasks LIMIT 1"
        ).fetchone()

    assert reviewed == 1
    assert task["status"] == "responded"
    assert task["reply_sent_at"]
    assert task["reply_ts"] == "50.123456"
    assert fake_slack.post_calls == [
        {
            "channel": "D40.9",
            "thread_ts": "40.9",
            "text": "Lo reviso y te aviso.",
            "unfurl_links": False,
            "unfurl_media": False,
        }
    ]


def test_review_can_edit_open_trello_and_mark_done(tmp_path):
    inputs = iter(["t", "d"])
    opened = []
    app = AgentApp(
        make_config(tmp_path),
        slack_client=FakeSlackClient(),
        structured_model_factory=lambda schema: FakeStructuredModel([]),
        input_fn=lambda prompt: next(inputs),
        open_url_fn=lambda url: opened.append(url) or True,
        sleep_fn=lambda _: None,
    )
    app.init_db()
    insert_task(
        app,
        created_at="2026-06-11T12:00:00+00:00",
        message_ts="41.0",
        summary="Revisar card",
        requested_action="Abrir Trello y resolver",
        priority="high",
        category="software",
        classification=make_classification(
            summary="Revisar card",
            requested_action="Abrir Trello y resolver",
            priority="high",
            category="software",
        ),
        trello_card_url="https://trello.com/c/card123",
    )

    reviewed = app.review_tasks(limit=1)

    with sqlite3.connect(app.config.db_path) as conn:
        status = conn.execute("SELECT status FROM tasks LIMIT 1").fetchone()[0]
    assert opened == ["https://trello.com/c/card123"]
    assert reviewed == 1
    assert status == "done"


def test_review_can_edit_draft_and_snooze_next_task(tmp_path):
    inputs = iter(["e", "Respuesta editada.", "s", "1d"])
    app = AgentApp(
        make_config(tmp_path),
        slack_client=FakeSlackClient(),
        structured_model_factory=lambda schema: FakeStructuredModel([]),
        input_fn=lambda prompt: next(inputs),
        sleep_fn=lambda _: None,
    )
    app.init_db()
    insert_task(
        app,
        created_at="2026-06-11T10:00:00+00:00",
        message_ts="42.0",
        summary="Responder campaña",
        requested_action="Responder con ajustes",
        priority="high",
        category="communications",
        classification=make_classification(
            summary="Responder campaña",
            requested_action="Responder con ajustes",
            priority="high",
            category="communications",
            draft_reply="Borrador inicial.",
        ),
    )
    insert_task(
        app,
        created_at="2026-06-11T11:00:00+00:00",
        message_ts="43.0",
        summary="Mirar reporte luego",
        requested_action="Revisar mañana",
        priority="medium",
        category="research",
        classification=make_classification(
            summary="Mirar reporte luego",
            requested_action="Revisar mañana",
            priority="medium",
            category="research",
        ),
    )

    reviewed = app.review_tasks(limit=2)

    with sqlite3.connect(app.config.db_path) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute("SELECT status, manual_reply, snoozed_until FROM tasks ORDER BY id ASC").fetchall()
    assert reviewed == 2
    assert rows[0]["status"] == "reply_approved"
    assert rows[0]["manual_reply"] == "Respuesta editada."
    assert rows[1]["status"] == "snoozed"
    assert rows[1]["snoozed_until"]


def test_brief_and_review_skip_future_snoozed_tasks(tmp_path):
    app = AgentApp(
        make_config(tmp_path),
        slack_client=FakeSlackClient(),
        structured_model_factory=lambda schema: FakeStructuredModel([]),
        sleep_fn=lambda _: None,
    )
    app.init_db()
    insert_task(
        app,
        created_at="2026-06-11T12:00:00+00:00",
        message_ts="44.0",
        summary="Tarea pospuesta",
        requested_action="Volver luego",
        priority="urgent",
        category="admin",
        classification=make_classification(
            summary="Tarea pospuesta",
            requested_action="Volver luego",
            priority="urgent",
            category="admin",
        ),
    )
    app.snooze_task(1, datetime(2026, 6, 12, 15, 0, tzinfo=timezone.utc))

    now = datetime(2026, 6, 11, 15, 0, tzinfo=timezone.utc)
    brief = app.build_brief(now=now)
    review_rows = app.fetch_tasks_for_review(now=now)

    assert "Tareas abiertas: 0" in brief
    assert review_rows == []


def test_extract_urls_from_text_deduplicates_and_cleans_slack_links():
    text = (
        "Mirá <https://example.com/reporte|reporte> y también "
        "https://example.com/reporte). "
        "Además https://github.com/acme/platform/issues/42."
    )

    urls = extract_urls_from_text(text)

    assert urls == [
        "https://example.com/reporte",
        "https://github.com/acme/platform/issues/42",
    ]


def test_extract_salesforce_url_from_slack_link_drops_visible_text():
    text = (
        "<https://techo.lightning.force.com/lightning/r/Campaign/7011W000001buEh/view|"
        "[IND] Campañas Pauta Digital>"
    )

    urls = extract_urls_from_text(text)

    assert urls == [
        "https://techo.lightning.force.com/lightning/r/Campaign/7011W000001buEh/view",
    ]
    assert "Connect your Salesforce account" not in urls[0]
    assert "[IND] Campañas Pauta Digital" not in urls[0]


def test_prompt_includes_public_url_enrichment(tmp_path):
    fake_model = FakeStructuredModel([sample_classification()])
    app = AgentApp(
        make_config(tmp_path),
        slack_client=FakeSlackClient(),
        structured_model_factory=lambda schema: fake_model,
        public_preview_fetcher=lambda url: (
            "Reporte Q2",
            "Dashboard público con métricas trimestrales.",
            "resolved",
            "",
        ),
        sleep_fn=lambda _: None,
    )
    app.init_db()

    app.process_message(
        {
            "ts": "20.0",
            "text": "ivan, revisás esto? https://example.com/report",
            "user": "UOTHER",
        },
        {"id": "G123", "name": "finanzas", "is_private": True},
        my_user_id="UME",
    )

    prompt = fake_model.calls[0]
    assert "URLs detectadas:" in prompt
    assert "Tipo: public_web" in prompt
    assert "Dashboard público con métricas trimestrales." in prompt

    stored_links = app.get_message_links("G123", "20.0")
    assert len(stored_links) == 1
    assert stored_links[0]["status"] == "resolved"
    assert stored_links[0]["summary"] == "Dashboard público con métricas trimestrales."


def test_slack_permalink_is_resolved_into_prompt_context(tmp_path):
    fake_slack = FakeSlackClient()
    permalink_ts = "1710000000.123456"
    permalink_url = "https://acme.slack.com/archives/C111/p1710000000123456"
    parent_message = {
        "ts": permalink_ts,
        "text": "Necesito que valides el tablero antes de compartirlo.",
        "user": "UOTHER",
        "thread_ts": permalink_ts,
        "reply_count": 1,
    }
    fake_slack.history_lookup[("C111", permalink_ts, permalink_ts, True)] = [parent_message]
    fake_slack.replies_lookup[("C111", permalink_ts)] = [
        parent_message,
        {"ts": "1710000001.000000", "text": "Lo ideal es tenerlo hoy.", "user": "UOTHER"},
    ]

    fake_model = FakeStructuredModel([sample_classification()])
    app = AgentApp(
        make_config(tmp_path),
        slack_client=fake_slack,
        structured_model_factory=lambda schema: fake_model,
        sleep_fn=lambda _: None,
    )
    app.init_db()

    app.process_message(
        {
            "ts": "21.0",
            "text": f"ivo, podés revisar esto? {permalink_url}",
            "user": "UOTHER",
        },
        {"id": "G777", "name": "ops", "is_private": True},
        my_user_id="UME",
    )

    prompt = fake_model.calls[0]
    assert "Tipo: slack_message" in prompt
    assert "Necesito que valides el tablero antes de compartirlo." in prompt
    assert "Lo ideal es tenerlo hoy." in prompt

    stored_links = app.get_message_links("G777", "21.0")
    assert stored_links[0]["url_type"] == "slack_message"
    assert stored_links[0]["status"] == "resolved"


def test_google_workspace_link_is_marked_as_requires_integration(tmp_path):
    fake_model = FakeStructuredModel([sample_classification()])
    app = AgentApp(
        make_config(tmp_path),
        slack_client=FakeSlackClient(),
        structured_model_factory=lambda schema: fake_model,
        sleep_fn=lambda _: None,
    )
    app.init_db()

    app.process_message(
        {
            "ts": "22.0",
            "text": "ivan, revisás esta planilla? https://docs.google.com/spreadsheets/d/abc123/edit",
            "user": "UOTHER",
        },
        {"id": "G999", "name": "equipo", "is_private": True},
        my_user_id="UME",
    )

    prompt = fake_model.calls[0]
    assert "Tipo: google_sheet" in prompt
    assert "requires_integration" in prompt
    assert "Google Workspace" in prompt


def test_trello_reference_skips_card_creation(tmp_path):
    fake_model = FakeStructuredModel([sample_classification()])
    fake_trello = FakeTrelloClient()
    app = AgentApp(
        make_config(
            tmp_path,
            TRELLO_ENABLED="true",
            TRELLO_API_KEY="abcd1234abcd1234abcd1234abcd1234",
            TRELLO_TOKEN="token",
            TRELLO_LIST_ID="list123",
        ),
        slack_client=FakeSlackClient(),
        structured_model_factory=lambda schema: fake_model,
        trello_client=fake_trello,
        sleep_fn=lambda _: None,
    )
    app.init_db()

    app.process_message(
        {
            "ts": "23.0",
            "text": "Revisás esta card? https://trello.com/c/abc12345/titulo-de-prueba",
            "user": "UOTHER",
        },
        {"id": "D123", "is_im": True, "user": "UOTHER"},
        my_user_id="UME",
    )

    with sqlite3.connect(app.config.db_path) as conn:
        conn.row_factory = sqlite3.Row
        task = conn.execute("SELECT trello_status, trello_last_error FROM tasks LIMIT 1").fetchone()

    assert task["trello_status"] == "skipped"
    assert "ya referencia un recurso de Trello" in task["trello_last_error"]
    assert fake_trello.created_cards == []
