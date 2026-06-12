import sqlite3
from datetime import datetime, timezone

from main import build_parser
from trello_client import TrelloCard
from slack_personal_agent import (
    AgentApp,
    AgentConfig,
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
    def __init__(self, should_fail=False):
        self.should_fail = should_fail
        self.created_cards = []

    def get_me(self):
        return {"fullName": "Ivan Rodriguez", "username": "ivan"}

    def get_list(self, list_id):
        return type("FakeList", (), {"id": list_id, "name": "Inbox", "board_id": "B1"})()

    def create_card(self, **kwargs):
        if self.should_fail:
            raise RuntimeError("trello down")
        self.created_cards.append(kwargs)
        return TrelloCard(id="card123", name=kwargs["name"], url="https://trello.com/c/card123")


def get_processed_row(db_path):
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        return conn.execute(
            "SELECT * FROM processed_messages ORDER BY processed_at ASC LIMIT 1"
        ).fetchone()


def count_tasks(db_path):
    with sqlite3.connect(db_path) as conn:
        return conn.execute("SELECT COUNT(*) FROM tasks").fetchone()[0]


def insert_task(
    app,
    *,
    created_at,
    message_ts,
    summary,
    requested_action,
    priority,
    category,
    classification,
    sender_label="Ana Gomez",
    conversation_label="DM con Ana Gomez",
    status="new",
    trello_status="created",
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
                requested_action,
                priority,
                category,
                status,
                trello_status,
                trello_card_url,
                trello_last_error,
                classification_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                created_at,
                f"D{message_ts}",
                message_ts,
                "UOTHER",
                sender_label,
                conversation_label,
                summary,
                requested_action,
                priority,
                category,
                status,
                trello_status,
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
    assert args.task_id == 12
    assert args.send


def test_main_parser_accepts_review_send_mode():
    args = build_parser().parse_args(["review", "--send-approved-replies"])
    assert args.command == "review"
    assert args.send_approved_replies


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
