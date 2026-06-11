from __future__ import annotations

import json
import os
import re
import requests
import sqlite3
import time
import unicodedata
import webbrowser
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Literal, Optional, Protocol, TypedDict

from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field
from rich import print
from rich.markup import escape
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError

from autostart import install_launch_agents, uninstall_launch_agents
from trello_client import TrelloCard, TrelloClient, TrelloError, build_trello_token_url
from url_enrichment import (
    MessageLink,
    classify_url,
    domain_from_url,
    extract_urls_from_text,
    format_links_for_prompt,
    parse_html_preview,
)


DB_PATH = "slack_agent.db"
DEFAULT_PROVIDER = "ollama"
DEFAULT_OLLAMA_MODEL = "qwen3:4b-instruct"
DEFAULT_OLLAMA_BASE_URL = "http://127.0.0.1:11434"
DEFAULT_GROQ_MODEL = "openai/gpt-oss-20b"
MAX_MESSAGE_LINKS = 5
MAX_PUBLIC_PREVIEW_BYTES = 65536


class ConfigError(RuntimeError):
    """Raised when required configuration is missing or invalid."""


class SlackClassification(BaseModel):
    is_actionable: bool = Field(description="Si el mensaje requiere acción de Ivan.")
    summary: str = Field(description="Resumen breve del mensaje.")
    requested_action: str = Field(description="Qué esperan que Ivan haga.")
    priority: Literal["low", "medium", "high", "urgent"]
    category: Literal[
        "salesforce",
        "data",
        "software",
        "research",
        "meeting",
        "admin",
        "fundraising",
        "communications",
        "other",
    ]
    needs_reply: bool
    needs_external_system: bool
    external_systems: list[str] = Field(default_factory=list)
    missing_information: list[str] = Field(default_factory=list)
    suggested_next_step: str
    draft_reply: str


class DoctorCheck(BaseModel):
    status: Literal["ok"]
    summary: str


class AgentState(TypedDict, total=False):
    message: dict[str, Any]
    conversation: dict[str, Any]
    my_user_id: str
    sender_label: str
    conversation_label: str
    relevant: bool
    relevance_reason: str
    message_links: list[MessageLink]
    classification: Optional[SlackClassification]


class StructuredModel(Protocol):
    def invoke(self, prompt: str) -> Any:
        ...


class TrelloClientProtocol(Protocol):
    def get_me(self) -> dict[str, Any]:
        ...

    def get_list(self, list_id: str) -> Any:
        ...

    def create_card(
        self,
        *,
        list_id: str,
        name: str,
        desc: str,
        pos: str = "top",
        member_ids: Optional[list[str]] = None,
        label_ids: Optional[list[str]] = None,
    ) -> TrelloCard:
        ...


PublicPreviewFetcher = Callable[[str], tuple[str, str, str, str]]


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_iso_datetime(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed


def now_slack_ts() -> str:
    return f"{time.time():.6f}"


def parse_bool(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def coerce_string_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    return []


def compact_text(value: Any, max_length: int = 140) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    if len(text) <= max_length:
        return text
    return text[: max_length - 3].rstrip() + "..."


def parse_snooze_until(value: str, now: Optional[datetime] = None) -> Optional[datetime]:
    raw = normalize_for_matching(value)
    current = now or datetime.now(timezone.utc)
    if current.tzinfo is None:
        current = current.replace(tzinfo=timezone.utc)
    if not raw:
        raw = "1d"
    if raw in {"manana", "mañana", "tomorrow"}:
        return current + timedelta(days=1)

    match = re.fullmatch(r"(\d+)\s*([hdw])", raw)
    if match:
        amount = int(match.group(1))
        unit = match.group(2)
        if unit == "h":
            return current + timedelta(hours=amount)
        if unit == "d":
            return current + timedelta(days=amount)
        if unit == "w":
            return current + timedelta(weeks=amount)

    parsed = parse_iso_datetime(raw)
    if parsed:
        return parsed
    return None


@dataclass(frozen=True)
class AgentConfig:
    slack_user_token: str
    my_slack_user_id: str = ""
    my_mention_aliases: tuple[str, ...] = ("ivan", "ivo")
    poll_seconds: int = 300
    slack_sleep_seconds: float = 1.2
    include_self_for_test: bool = False
    model_provider: Literal["ollama", "groq"] = DEFAULT_PROVIDER
    ollama_model: str = DEFAULT_OLLAMA_MODEL
    ollama_base_url: str = DEFAULT_OLLAMA_BASE_URL
    groq_model: str = DEFAULT_GROQ_MODEL
    groq_api_key: str = ""
    slack_send_approved_replies: bool = False
    trello_enabled: bool = False
    trello_auto_create: bool = True
    trello_api_key: str = ""
    trello_token: str = ""
    trello_list_id: str = ""
    trello_member_ids: tuple[str, ...] = ()
    trello_label_ids: tuple[str, ...] = ()
    trello_card_position: str = "top"
    db_path: str = DB_PATH

    @classmethod
    def from_env(cls, env: Optional[dict[str, str]] = None) -> "AgentConfig":
        env_map = env if env is not None else os.environ
        get = env_map.get

        slack_user_token = (get("SLACK_USER_TOKEN", "") or "").strip()
        if not slack_user_token:
            raise ConfigError("Falta SLACK_USER_TOKEN en el entorno o en .env.")

        model_provider = (get("MODEL_PROVIDER", DEFAULT_PROVIDER) or DEFAULT_PROVIDER).strip().lower()
        if model_provider not in {"ollama", "groq"}:
            raise ConfigError("MODEL_PROVIDER debe ser 'ollama' o 'groq'.")

        poll_seconds_raw = (get("POLL_SECONDS", "300") or "300").strip()
        sleep_seconds_raw = (get("SLACK_SLEEP_SECONDS", "1.2") or "1.2").strip()

        try:
            poll_seconds = int(poll_seconds_raw)
        except ValueError as exc:
            raise ConfigError("POLL_SECONDS debe ser un entero.") from exc

        try:
            slack_sleep_seconds = float(sleep_seconds_raw)
        except ValueError as exc:
            raise ConfigError("SLACK_SLEEP_SECONDS debe ser un número.") from exc

        mention_aliases = tuple(
            alias
            for alias in (
                normalize_for_matching(part)
                for part in (get("MY_MENTION_ALIASES", "ivan,ivo") or "ivan,ivo").split(",")
            )
            if alias
        )
        if not mention_aliases:
            mention_aliases = ("ivan", "ivo")

        trello_member_ids = tuple(
            member_id.strip()
            for member_id in (get("TRELLO_MEMBER_IDS", "") or "").split(",")
            if member_id.strip()
        )
        trello_label_ids = tuple(
            label_id.strip()
            for label_id in (get("TRELLO_LABEL_IDS", "") or "").split(",")
            if label_id.strip()
        )

        return cls(
            slack_user_token=slack_user_token,
            my_slack_user_id=(get("MY_SLACK_USER_ID", "") or "").strip(),
            my_mention_aliases=mention_aliases,
            poll_seconds=poll_seconds,
            slack_sleep_seconds=slack_sleep_seconds,
            include_self_for_test=parse_bool(get("INCLUDE_SELF_FOR_TEST", "false") or "false"),
            model_provider=model_provider,  # type: ignore[arg-type]
            ollama_model=(get("OLLAMA_MODEL", DEFAULT_OLLAMA_MODEL) or DEFAULT_OLLAMA_MODEL).strip(),
            ollama_base_url=(get("OLLAMA_BASE_URL", DEFAULT_OLLAMA_BASE_URL) or DEFAULT_OLLAMA_BASE_URL).strip(),
            groq_model=(get("GROQ_MODEL", DEFAULT_GROQ_MODEL) or DEFAULT_GROQ_MODEL).strip(),
            groq_api_key=(get("GROQ_API_KEY", "") or "").strip(),
            slack_send_approved_replies=parse_bool(get("SLACK_SEND_APPROVED_REPLIES", "false") or "false"),
            trello_enabled=parse_bool(get("TRELLO_ENABLED", "false") or "false"),
            trello_auto_create=parse_bool(get("TRELLO_AUTO_CREATE", "true") or "true"),
            trello_api_key=(get("TRELLO_API_KEY", "") or "").strip(),
            trello_token=(get("TRELLO_TOKEN", "") or "").strip(),
            trello_list_id=(get("TRELLO_LIST_ID", "") or "").strip(),
            trello_member_ids=trello_member_ids,
            trello_label_ids=trello_label_ids,
            trello_card_position=(get("TRELLO_CARD_POSITION", "top") or "top").strip(),
            db_path=(get("DB_PATH", DB_PATH) or DB_PATH).strip(),
        )


def local_model_fit_hint(model_name: str) -> tuple[str, str]:
    lowered = model_name.lower()
    known_hints = {
        "qwen3:4b": ("ok", "Modelo liviano y razonable para esta Mac; ronda los 2.5 GB en Ollama."),
        "qwen3:4b-instruct": ("ok", "Modelo liviano y razonable para esta Mac; ronda los 2.5 GB en Ollama."),
        "llama3.1:8b": ("caution", "Alternativa viable, pero más lenta y pesada; ronda los 4.9 GB cuantizado."),
        "llama3.1:8b-instruct": ("caution", "Alternativa viable, pero más lenta y pesada; ronda los 4.9 GB cuantizado."),
        "gpt-oss:20b": ("warning", "Modelo muy pesado para 18 GB unificados; ronda los 14 GB y deja poco margen al sistema."),
    }
    if lowered in known_hints:
        return known_hints[lowered]

    if "fp16" in lowered or "bf16" in lowered:
        return "warning", "La variante elegida usa precisión alta y es poco recomendable para 18 GB unificados."

    size_match = re.search(r":(\d+(?:\.\d+)?)b", lowered)
    if not size_match:
        return "unknown", "No tengo una heurística precisa para este modelo; verificá consumo real con `doctor`."

    size_b = float(size_match.group(1))
    if size_b >= 14:
        return "warning", "Por tamaño parece demasiado grande para esta Mac si va a quedar corriendo en background."
    if size_b >= 8:
        return "caution", "Podría funcionar, pero esperá más latencia y menos margen de memoria."
    if size_b >= 4:
        return "ok", "Por tamaño debería ser razonable para esta Mac en un uso de clasificación liviana."
    return "ok", "Modelo muy liviano; buen candidato para correr en segundo plano."


def normalize_for_matching(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text)
    without_accents = "".join(char for char in normalized if not unicodedata.combining(char))
    return without_accents.lower().strip()


def validate_trello_api_key_format(value: str) -> Optional[str]:
    key = value.strip()
    if not key:
        return "Falta TRELLO_API_KEY."
    if key.startswith("ATATT"):
        return (
            "El valor cargado en TRELLO_API_KEY parece ser un token de Atlassian Account, "
            "no una API key de Trello Power-Up."
        )
    if "=" in key:
        return (
            "TRELLO_API_KEY no debería contener '='. Eso sugiere que pegaste una URL completa "
            "o una credencial del tipo incorrecto."
        )
    if len(key) < 20:
        return "TRELLO_API_KEY parece demasiado corto para ser una API key válida de Trello."
    return None


class AgentApp:
    def __init__(
        self,
        config: AgentConfig,
        slack_client: Optional[WebClient] = None,
        structured_model_factory: Optional[Callable[[type[BaseModel]], StructuredModel]] = None,
        trello_client: Optional[TrelloClientProtocol] = None,
        public_preview_fetcher: Optional[PublicPreviewFetcher] = None,
        input_fn: Callable[[str], str] = input,
        open_url_fn: Callable[[str], bool] = webbrowser.open,
        sleep_fn: Callable[[float], None] = time.sleep,
    ) -> None:
        self.config = config
        self.slack = slack_client or WebClient(token=config.slack_user_token)
        self._structured_model_factory = structured_model_factory or self._build_structured_model
        self._trello_client = trello_client
        self._public_preview_fetcher = public_preview_fetcher or self.fetch_public_url_preview
        self._input = input_fn
        self._open_url = open_url_fn
        self._classifier: Optional[StructuredModel] = None
        self._graph = None
        self._user_cache: dict[str, str] = {}
        self._sleep = sleep_fn

    def db_connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.config.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def init_db(self) -> None:
        with self.db_connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS conversations (
                    channel_id TEXT PRIMARY KEY,
                    channel_name TEXT,
                    channel_type TEXT,
                    is_private INTEGER,
                    is_im INTEGER,
                    is_mpim INTEGER,
                    last_seen_ts TEXT DEFAULT '0'
                )
                """
            )

            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS processed_messages (
                    channel_id TEXT NOT NULL,
                    message_ts TEXT NOT NULL,
                    processed_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    user_id TEXT,
                    sender_label TEXT,
                    conversation_label TEXT,
                    raw_text TEXT,
                    relevant INTEGER,
                    relevance_reason TEXT,
                    context_text TEXT,
                    classification_status TEXT NOT NULL DEFAULT 'pending',
                    classification_json TEXT,
                    classification_error TEXT,
                    PRIMARY KEY (channel_id, message_ts)
                )
                """
            )

            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS tasks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at TEXT NOT NULL,
                    channel_id TEXT NOT NULL,
                    message_ts TEXT NOT NULL,
                    user_id TEXT,
                    sender_label TEXT,
                    conversation_label TEXT,
                    summary TEXT,
                    requested_action TEXT,
                    priority TEXT,
                    category TEXT,
                    status TEXT DEFAULT 'new',
                    classification_json TEXT
                )
                """
            )

            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS message_links (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    channel_id TEXT NOT NULL,
                    message_ts TEXT NOT NULL,
                    url TEXT NOT NULL,
                    domain TEXT NOT NULL,
                    url_type TEXT NOT NULL,
                    title TEXT,
                    summary TEXT,
                    status TEXT NOT NULL,
                    error TEXT,
                    fetched_at TEXT NOT NULL,
                    metadata_json TEXT,
                    UNIQUE(channel_id, message_ts, url)
                )
                """
            )

            self._ensure_column(conn, "processed_messages", "updated_at", "updated_at TEXT NOT NULL DEFAULT ''")
            self._ensure_column(
                conn,
                "processed_messages",
                "classification_status",
                "classification_status TEXT NOT NULL DEFAULT 'pending'",
            )
            self._ensure_column(conn, "processed_messages", "classification_error", "classification_error TEXT")
            self._ensure_column(conn, "processed_messages", "context_text", "context_text TEXT")
            self._ensure_column(conn, "tasks", "trello_status", "trello_status TEXT NOT NULL DEFAULT 'pending'")
            self._ensure_column(conn, "tasks", "trello_card_id", "trello_card_id TEXT")
            self._ensure_column(conn, "tasks", "trello_card_url", "trello_card_url TEXT")
            self._ensure_column(conn, "tasks", "trello_last_error", "trello_last_error TEXT")
            self._ensure_column(conn, "tasks", "trello_synced_at", "trello_synced_at TEXT")
            self._ensure_column(conn, "tasks", "reviewed_at", "reviewed_at TEXT")
            self._ensure_column(conn, "tasks", "snoozed_until", "snoozed_until TEXT")
            self._ensure_column(conn, "tasks", "reply_approved_at", "reply_approved_at TEXT")
            self._ensure_column(conn, "tasks", "reply_sent_at", "reply_sent_at TEXT")
            self._ensure_column(conn, "tasks", "reply_ts", "reply_ts TEXT")
            self._ensure_column(conn, "tasks", "reply_error", "reply_error TEXT")
            self._ensure_column(conn, "tasks", "manual_reply", "manual_reply TEXT")

            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS task_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    task_id INTEGER NOT NULL,
                    created_at TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    details_json TEXT,
                    FOREIGN KEY(task_id) REFERENCES tasks(id)
                )
                """
            )

            conn.execute(
                """
                CREATE UNIQUE INDEX IF NOT EXISTS idx_tasks_channel_message
                ON tasks(channel_id, message_ts)
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_processed_retry_status
                ON processed_messages(classification_status, processed_at)
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_message_links_message
                ON message_links(channel_id, message_ts)
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_task_events_task
                ON task_events(task_id, created_at)
                """
            )

    def _ensure_column(self, conn: sqlite3.Connection, table: str, column_name: str, definition: str) -> None:
        columns = {row["name"] for row in conn.execute(f"PRAGMA table_info({table})")}
        if column_name not in columns:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {definition}")

    def record_task_event(self, task_id: int, event_type: str, details: Optional[dict[str, Any]] = None) -> None:
        with self.db_connect() as conn:
            conn.execute(
                """
                INSERT INTO task_events (task_id, created_at, event_type, details_json)
                VALUES (?, ?, ?, ?)
                """,
                (
                    task_id,
                    now_iso(),
                    event_type,
                    json.dumps(details or {}, ensure_ascii=False, sort_keys=True),
                ),
            )

    def _build_structured_model(self, schema: type[BaseModel]) -> StructuredModel:
        if self.config.model_provider == "ollama":
            try:
                from langchain_ollama import ChatOllama
            except ImportError as exc:
                raise RuntimeError(
                    "Falta instalar `langchain-ollama`. Corré `pip install -r requirements.txt`."
                ) from exc

            model = ChatOllama(
                model=self.config.ollama_model,
                base_url=self.config.ollama_base_url,
                temperature=0,
            )
            return model.with_structured_output(schema)

        if not self.config.groq_api_key:
            raise RuntimeError("MODEL_PROVIDER=groq requiere GROQ_API_KEY configurada.")

        try:
            from langchain_groq import ChatGroq
        except ImportError as exc:
            raise RuntimeError(
                "Falta instalar `langchain-groq`. Corré `pip install -r requirements.txt`."
            ) from exc

        model = ChatGroq(
            model=self.config.groq_model,
            api_key=self.config.groq_api_key,
            temperature=0,
        )
        return model.with_structured_output(schema)

    def get_classifier(self) -> StructuredModel:
        if self._classifier is None:
            self._classifier = self._structured_model_factory(SlackClassification)
        return self._classifier

    def has_complete_trello_config(self) -> bool:
        return bool(
            self.config.trello_api_key
            and self.config.trello_token
            and self.config.trello_list_id
        )

    def has_trello_auth_config(self) -> bool:
        return bool(
            self.config.trello_api_key
            and self.config.trello_token
        )

    def trello_api_key_error(self) -> Optional[str]:
        return validate_trello_api_key_format(self.config.trello_api_key)

    def get_trello_client(self) -> TrelloClientProtocol:
        if self._trello_client is None:
            if not self.has_trello_auth_config():
                raise TrelloError(
                    "Faltan TRELLO_API_KEY o TRELLO_TOKEN."
                )
            key_error = self.trello_api_key_error()
            if key_error:
                raise TrelloError(key_error)
            self._trello_client = TrelloClient(
                api_key=self.config.trello_api_key,
                token=self.config.trello_token,
            )
        return self._trello_client

    def slack_call(self, method: Callable[..., Any], **kwargs: Any) -> Any:
        while True:
            try:
                return method(**kwargs)
            except SlackApiError as exc:
                if exc.response.get("error") == "ratelimited":
                    retry_after = int(exc.response.headers.get("Retry-After", "30"))
                    print(f"[yellow]Slack rate limit. Esperando {retry_after}s...[/yellow]")
                    self._sleep(retry_after)
                    continue
                raise

    def get_my_user_id(self) -> str:
        if self.config.my_slack_user_id:
            return self.config.my_slack_user_id

        auth = self.slack_call(self.slack.auth_test)
        user_id = auth["user_id"]
        print(f"[yellow]MY_SLACK_USER_ID no estaba en .env. Usando detectado:[/yellow] {user_id}")
        return user_id

    def conv_type(self, conversation: dict[str, Any]) -> str:
        if conversation.get("is_im"):
            return "im"
        if conversation.get("is_mpim"):
            return "mpim"
        if conversation.get("is_private"):
            return "private_channel"
        return "public_channel"

    def user_label(self, user_id: Optional[str]) -> str:
        if not user_id:
            return "unknown"
        if user_id in self._user_cache:
            return self._user_cache[user_id]

        try:
            response = self.slack_call(self.slack.users_info, user=user_id)
            user = response.get("user", {})
            profile = user.get("profile", {})
            label = (
                profile.get("real_name")
                or profile.get("display_name")
                or user.get("name")
                or user_id
            )
        except Exception:
            label = user_id

        self._user_cache[user_id] = label
        return label

    def conversation_label(self, conversation: dict[str, Any]) -> str:
        if conversation.get("is_im"):
            return f"DM con {self.user_label(conversation.get('user'))}"
        if conversation.get("is_mpim"):
            return conversation.get("name") or conversation.get("id")
        return f"#{conversation.get('name') or conversation.get('id')}"

    def upsert_conversation(self, conversation: dict[str, Any], last_seen_ts: Optional[str] = None) -> None:
        with self.db_connect() as conn:
            conn.execute(
                """
                INSERT INTO conversations (
                    channel_id,
                    channel_name,
                    channel_type,
                    is_private,
                    is_im,
                    is_mpim,
                    last_seen_ts
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(channel_id) DO UPDATE SET
                    channel_name=excluded.channel_name,
                    channel_type=excluded.channel_type,
                    is_private=excluded.is_private,
                    is_im=excluded.is_im,
                    is_mpim=excluded.is_mpim,
                    last_seen_ts=COALESCE(?, conversations.last_seen_ts)
                """,
                (
                    conversation["id"],
                    self.conversation_label(conversation),
                    self.conv_type(conversation),
                    int(bool(conversation.get("is_private"))),
                    int(bool(conversation.get("is_im"))),
                    int(bool(conversation.get("is_mpim"))),
                    last_seen_ts or "0",
                    last_seen_ts,
                ),
            )

    def get_last_seen_ts(self, channel_id: str) -> str:
        with self.db_connect() as conn:
            row = conn.execute(
                "SELECT last_seen_ts FROM conversations WHERE channel_id = ?",
                (channel_id,),
            ).fetchone()
        return row["last_seen_ts"] if row and row["last_seen_ts"] else "0"

    def set_last_seen_ts(self, channel_id: str, ts_value: str) -> None:
        with self.db_connect() as conn:
            conn.execute(
                "UPDATE conversations SET last_seen_ts = ? WHERE channel_id = ?",
                (ts_value, channel_id),
            )

    def get_processed_status(self, channel_id: str, message_ts: str) -> Optional[str]:
        with self.db_connect() as conn:
            row = conn.execute(
                """
                SELECT classification_status
                FROM processed_messages
                WHERE channel_id = ? AND message_ts = ?
                """,
                (channel_id, message_ts),
            ).fetchone()
        if not row:
            return None
        return row["classification_status"]

    def upsert_processed_relevance(
        self,
        *,
        channel_id: str,
        message_ts: str,
        user_id: Optional[str],
        sender_label: str,
        conversation_label: str,
        raw_text: str,
        relevant: bool,
        relevance_reason: str,
        context_text: str,
    ) -> None:
        classification_status = "pending" if relevant else "ignored"
        timestamp = now_iso()

        with self.db_connect() as conn:
            conn.execute(
                """
                INSERT INTO processed_messages (
                    channel_id,
                    message_ts,
                    processed_at,
                    updated_at,
                    user_id,
                    sender_label,
                    conversation_label,
                    raw_text,
                    relevant,
                    relevance_reason,
                    context_text,
                    classification_status,
                    classification_json,
                    classification_error
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, NULL)
                ON CONFLICT(channel_id, message_ts) DO UPDATE SET
                    updated_at=excluded.updated_at,
                    user_id=excluded.user_id,
                    sender_label=excluded.sender_label,
                    conversation_label=excluded.conversation_label,
                    raw_text=excluded.raw_text,
                    relevant=excluded.relevant,
                    relevance_reason=excluded.relevance_reason,
                    context_text=excluded.context_text,
                    classification_status=excluded.classification_status,
                    classification_error=NULL
                """,
                (
                    channel_id,
                    message_ts,
                    timestamp,
                    timestamp,
                    user_id,
                    sender_label,
                    conversation_label,
                    raw_text,
                    int(relevant),
                    relevance_reason,
                    context_text,
                    classification_status,
                ),
            )

    def mark_processed_done(
        self,
        *,
        channel_id: str,
        message_ts: str,
        classification: SlackClassification,
    ) -> None:
        with self.db_connect() as conn:
            conn.execute(
                """
                UPDATE processed_messages
                SET updated_at = ?,
                    classification_status = 'done',
                    classification_json = ?,
                    classification_error = NULL
                WHERE channel_id = ? AND message_ts = ?
                """,
                (
                    now_iso(),
                    json.dumps(classification.model_dump(), ensure_ascii=False),
                    channel_id,
                    message_ts,
                ),
            )

    def mark_processed_failed(self, *, channel_id: str, message_ts: str, error_message: str) -> None:
        with self.db_connect() as conn:
            conn.execute(
                """
                UPDATE processed_messages
                SET updated_at = ?,
                    classification_status = 'failed',
                    classification_error = ?
                WHERE channel_id = ? AND message_ts = ?
                """,
                (
                    now_iso(),
                    error_message[:1000],
                    channel_id,
                    message_ts,
                ),
            )

    def save_task(
        self,
        *,
        channel_id: str,
        message_ts: str,
        user_id: Optional[str],
        sender_label: str,
        conversation_label: str,
        classification: SlackClassification,
    ) -> bool:
        with self.db_connect() as conn:
            cursor = conn.execute(
                """
                INSERT OR IGNORE INTO tasks (
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
                    classification_json
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    now_iso(),
                    channel_id,
                    message_ts,
                    user_id,
                    sender_label,
                    conversation_label,
                    classification.summary,
                    classification.requested_action,
                    classification.priority,
                    classification.category,
                    "new",
                    "pending",
                    json.dumps(classification.model_dump(), ensure_ascii=False),
                ),
            )
        return cursor.rowcount > 0

    def extract_message_links(self, text: str) -> list[MessageLink]:
        links: list[MessageLink] = []
        for url in extract_urls_from_text(text)[:MAX_MESSAGE_LINKS]:
            url_type, metadata = classify_url(url)
            links.append(
                MessageLink(
                    url=url,
                    domain=domain_from_url(url),
                    url_type=url_type,
                    metadata=metadata,
                )
            )
        return links

    def fetch_public_url_preview(self, url: str) -> tuple[str, str, str, str]:
        try:
            response = requests.get(
                url,
                timeout=(3, 5),
                allow_redirects=True,
                stream=True,
                headers={"User-Agent": "SlackPersonalAgent/1.0"},
            )
        except requests.RequestException as exc:
            return "", "", "error", str(exc)

        if response.status_code in {401, 403}:
            return "", "", "private_or_blocked", f"HTTP {response.status_code}"
        if response.status_code >= 400:
            return "", "", "error", f"HTTP {response.status_code}"

        content_type = (response.headers.get("Content-Type", "") or "").lower()
        if "text/html" not in content_type:
            return "", f"Recurso no HTML ({content_type or 'sin content-type'})", "non_html", ""

        chunks: list[bytes] = []
        size = 0
        try:
            for chunk in response.iter_content(chunk_size=4096):
                if not chunk:
                    continue
                chunks.append(chunk)
                size += len(chunk)
                if size >= MAX_PUBLIC_PREVIEW_BYTES:
                    break
        finally:
            response.close()

        html_text = b"".join(chunks).decode(response.encoding or "utf-8", errors="ignore")
        title, description = parse_html_preview(html_text)
        if title or description:
            summary = description or title
            return title, summary, "resolved", ""
        return "", "No encontré metadata útil en la página.", "fetched", ""

    def build_slack_link_preview(self, link: MessageLink) -> MessageLink:
        channel_id = link.metadata.get("channel_id")
        message_ts = link.metadata.get("message_ts")
        if not channel_id or not message_ts:
            link.status = "unparsed"
            link.summary = "No pude parsear el permalink de Slack."
            return link

        try:
            response = self.slack_call(
                self.slack.conversations_history,
                channel=channel_id,
                oldest=message_ts,
                latest=message_ts,
                inclusive=True,
                limit=1,
            )
            messages = response.get("messages", [])
            if not messages:
                link.status = "not_found"
                link.summary = "No encontré el mensaje referenciado en Slack."
                return link

            message = messages[0]
            sender = self.user_label(message.get("user"))
            message_text = (message.get("text") or "").strip()
            link.title = f"Mensaje de Slack en {channel_id}"
            link.summary = f"{sender}: {message_text[:220]}" if message_text else f"{sender}: (sin texto)"
            link.status = "resolved"
            if message.get("thread_ts") == message.get("ts") and int(message.get("reply_count", 0) or 0) > 0:
                try:
                    replies = self.slack_call(
                        self.slack.conversations_replies,
                        channel=channel_id,
                        ts=message_ts,
                        limit=3,
                    ).get("messages", [])[1:]
                except Exception:
                    replies = []
                if replies:
                    reply_lines = []
                    for reply in replies[:2]:
                        reply_sender = self.user_label(reply.get("user"))
                        reply_lines.append(f"{reply_sender}: {(reply.get('text') or '').strip()[:120]}")
                    link.summary += " | Respuestas: " + " / ".join(reply_lines)
            return link
        except Exception as exc:
            link.status = "error"
            link.error = str(exc)
            link.summary = "No pude resolver el enlace de Slack."
            return link

    def enrich_message_link(self, link: MessageLink) -> MessageLink:
        if link.url_type == "slack_message":
            return self.build_slack_link_preview(link)

        if link.url_type.startswith("trello_"):
            link.status = "resolved"
            link.summary = "El mensaje ya referencia un recurso de Trello; evitá duplicar la tarjeta."
            return link

        if link.url_type in {"google_sheet", "google_doc", "google_slides", "google_drive", "google_docs_resource"}:
            link.status = "requires_integration"
            link.summary = "Recurso de Google Workspace; por ahora conviene tratarlo como referencia privada."
            return link

        if link.url_type == "salesforce":
            link.status = "requires_integration"
            link.summary = "Enlace de Salesforce detectado; probablemente requiere contexto o integración autenticada."
            return link

        if link.url_type.startswith("github_"):
            link.status = "resolved"
            owner = link.metadata.get("owner")
            repo = link.metadata.get("repo")
            number = link.metadata.get("number")
            if link.url_type == "github_issue":
                link.summary = f"Issue de GitHub {owner}/{repo}#{number}."
            elif link.url_type == "github_pr":
                link.summary = f"Pull request de GitHub {owner}/{repo}#{number}."
            elif link.url_type == "github_repo":
                link.summary = f"Repositorio de GitHub {owner}/{repo}."
            else:
                link.summary = "Recurso de GitHub detectado."
            return link

        if link.url_type == "private_resource":
            link.status = "private_or_unknown"
            link.summary = "Enlace privado o local; guardado como referencia sin intentar abrirlo."
            return link

        title, summary, status, error = self._public_preview_fetcher(link.url)
        link.title = title
        link.summary = summary or "URL pública detectada."
        link.status = status
        link.error = error
        return link

    def replace_message_links(self, channel_id: str, message_ts: str, links: list[MessageLink]) -> None:
        with self.db_connect() as conn:
            conn.execute(
                "DELETE FROM message_links WHERE channel_id = ? AND message_ts = ?",
                (channel_id, message_ts),
            )
            for link in links:
                conn.execute(
                    """
                    INSERT INTO message_links (
                        channel_id,
                        message_ts,
                        url,
                        domain,
                        url_type,
                        title,
                        summary,
                        status,
                        error,
                        fetched_at,
                        metadata_json
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        channel_id,
                        message_ts,
                        link.url,
                        link.domain,
                        link.url_type,
                        link.title,
                        link.summary,
                        link.status,
                        link.error,
                        now_iso(),
                        link.metadata_json(),
                    ),
                )

    def get_message_links(self, channel_id: str, message_ts: str) -> list[sqlite3.Row]:
        with self.db_connect() as conn:
            rows = conn.execute(
                """
                SELECT url, domain, url_type, title, summary, status, error, metadata_json
                FROM message_links
                WHERE channel_id = ? AND message_ts = ?
                ORDER BY id ASC
                """,
                (channel_id, message_ts),
            ).fetchall()
        return list(rows)

    def get_message_links_context(self, channel_id: str, message_ts: str) -> str:
        rows = self.get_message_links(channel_id, message_ts)
        links: list[MessageLink] = []
        for row in rows:
            metadata: dict[str, Any] = {}
            raw_metadata = row["metadata_json"] or ""
            if raw_metadata:
                try:
                    metadata = json.loads(raw_metadata)
                except json.JSONDecodeError:
                    metadata = {}
            links.append(
                MessageLink(
                    url=row["url"],
                    domain=row["domain"],
                    url_type=row["url_type"],
                    title=row["title"] or "",
                    summary=row["summary"] or "",
                    status=row["status"] or "detected",
                    error=row["error"] or "",
                    metadata=metadata,
                )
            )
        return format_links_for_prompt(links)

    def message_mentions_me(self, text: str, my_user_id: str) -> bool:
        if f"<@{my_user_id}>" in text:
            return True

        normalized_text = normalize_for_matching(text)
        for alias in self.config.my_mention_aliases:
            if re.search(rf"(?<!\w){re.escape(alias)}(?!\w)", normalized_text):
                return True
        return False

    def get_task_row(self, channel_id: str, message_ts: str) -> Optional[sqlite3.Row]:
        with self.db_connect() as conn:
            row = conn.execute(
                """
                SELECT tasks.*,
                       processed_messages.raw_text,
                       processed_messages.context_text
                FROM tasks
                LEFT JOIN processed_messages
                  ON processed_messages.channel_id = tasks.channel_id
                 AND processed_messages.message_ts = tasks.message_ts
                WHERE tasks.channel_id = ? AND tasks.message_ts = ?
                """,
                (channel_id, message_ts),
            ).fetchone()
        return row

    def get_task_by_id(self, task_id: int) -> Optional[sqlite3.Row]:
        with self.db_connect() as conn:
            row = conn.execute(
                """
                SELECT tasks.*,
                       processed_messages.raw_text,
                       processed_messages.context_text,
                       (
                           SELECT COUNT(*)
                           FROM message_links
                           WHERE message_links.channel_id = tasks.channel_id
                             AND message_links.message_ts = tasks.message_ts
                       ) AS link_count
                FROM tasks
                LEFT JOIN processed_messages
                  ON processed_messages.channel_id = tasks.channel_id
                 AND processed_messages.message_ts = tasks.message_ts
                WHERE tasks.id = ?
                """,
                (task_id,),
            ).fetchone()
        return row

    def message_has_trello_reference(self, channel_id: str, message_ts: str) -> bool:
        with self.db_connect() as conn:
            row = conn.execute(
                """
                SELECT 1
                FROM message_links
                WHERE channel_id = ?
                  AND message_ts = ?
                  AND url_type LIKE 'trello_%'
                LIMIT 1
                """,
                (channel_id, message_ts),
            ).fetchone()
        return row is not None

    def build_trello_card_payload(self, task_row: sqlite3.Row) -> tuple[str, str]:
        summary = task_row["summary"] or "Nueva tarea desde Slack"
        action = task_row["requested_action"] or "Sin acción especificada"
        raw_text = task_row["raw_text"] or ""
        context_text = task_row["context_text"] or ""
        links_text = self.get_message_links_context(task_row["channel_id"], task_row["message_ts"])
        name = summary[:120]
        description = "\n".join(
            [
                f"Conversación: {task_row['conversation_label']}",
                f"Remitente: {task_row['sender_label']}",
                f"Prioridad: {task_row['priority']}",
                f"Categoría: {task_row['category']}",
                "",
                f"Acción pedida: {action}",
                "",
                "Mensaje original:",
                raw_text or "(sin texto)",
                "",
                "Contexto reciente:",
                context_text or "Sin contexto reciente.",
                "",
                "URLs detectadas:",
                links_text,
            ]
        )
        return name, description

    def mark_task_trello_created(self, task_id: int, card: TrelloCard) -> None:
        with self.db_connect() as conn:
            conn.execute(
                """
                UPDATE tasks
                SET trello_status = 'created',
                    trello_card_id = ?,
                    trello_card_url = ?,
                    trello_last_error = NULL,
                    trello_synced_at = ?
                WHERE id = ?
                """,
                (card.id, card.url, now_iso(), task_id),
            )

    def mark_task_trello_failed(self, task_id: int, error_message: str) -> None:
        with self.db_connect() as conn:
            conn.execute(
                """
                UPDATE tasks
                SET trello_status = 'failed',
                    trello_last_error = ?,
                    trello_synced_at = ?
                WHERE id = ?
                """,
                (error_message[:1000], now_iso(), task_id),
            )

    def mark_task_trello_skipped(self, task_id: int, reason: str) -> None:
        with self.db_connect() as conn:
            conn.execute(
                """
                UPDATE tasks
                SET trello_status = 'skipped',
                    trello_last_error = ?,
                    trello_synced_at = ?
                WHERE id = ?
                """,
                (reason[:1000], now_iso(), task_id),
            )

    def sync_task_to_trello(self, task_row: sqlite3.Row) -> bool:
        if not self.config.trello_enabled:
            return False

        if task_row["trello_card_id"]:
            return False

        if not self.config.trello_list_id:
            self.mark_task_trello_failed(task_row["id"], "Falta TRELLO_LIST_ID para crear cards.")
            return False

        if self.message_has_trello_reference(task_row["channel_id"], task_row["message_ts"]):
            self.mark_task_trello_skipped(task_row["id"], "El mensaje ya referencia un recurso de Trello.")
            return False

        name, description = self.build_trello_card_payload(task_row)
        try:
            client = self.get_trello_client()
            card = client.create_card(
                list_id=self.config.trello_list_id,
                name=name,
                desc=description,
                pos=self.config.trello_card_position,
                member_ids=list(self.config.trello_member_ids),
                label_ids=list(self.config.trello_label_ids),
            )
        except Exception as exc:
            self.mark_task_trello_failed(task_row["id"], str(exc))
            print(f"[yellow]No pude crear la card de Trello para la tarea #{task_row['id']}: {exc}[/yellow]")
            return False

        self.mark_task_trello_created(task_row["id"], card)
        print(f"[green]Trello OK:[/green] tarea #{task_row['id']} -> {card.url}")
        return True

    def sync_task_to_trello_by_message(self, channel_id: str, message_ts: str) -> bool:
        task_row = self.get_task_row(channel_id, message_ts)
        if not task_row:
            return False
        return self.sync_task_to_trello(task_row)

    def sync_pending_trello_tasks(self, limit: int = 20) -> int:
        if not self.config.trello_enabled:
            return 0

        with self.db_connect() as conn:
            rows = conn.execute(
                """
                SELECT tasks.*,
                       processed_messages.raw_text,
                       processed_messages.context_text
                FROM tasks
                LEFT JOIN processed_messages
                  ON processed_messages.channel_id = tasks.channel_id
                 AND processed_messages.message_ts = tasks.message_ts
                WHERE tasks.trello_card_id IS NULL
                  AND tasks.trello_status IN ('pending', 'failed')
                ORDER BY tasks.id ASC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()

        synced = 0
        for row in rows:
            synced += int(self.sync_task_to_trello(row))
        return synced

    def list_conversations(self) -> list[dict[str, Any]]:
        conversations: list[dict[str, Any]] = []
        cursor: Optional[str] = None

        while True:
            response = self.slack_call(
                self.slack.conversations_list,
                types="private_channel,im,mpim",
                limit=200,
                cursor=cursor,
                exclude_archived=True,
            )
            conversations.extend(response.get("channels", []))
            cursor = response.get("response_metadata", {}).get("next_cursor")
            if not cursor:
                break

        return conversations

    def fetch_recent_context(self, channel_id: str, before_ts: str, limit: int = 6) -> str:
        response = self.slack_call(
            self.slack.conversations_history,
            channel=channel_id,
            latest=before_ts,
            inclusive=False,
            limit=limit,
        )
        messages = sorted(
            (message for message in response.get("messages", []) if self.is_human_message(message)),
            key=lambda message: float(message.get("ts", "0")),
        )
        lines = []
        for message in messages:
            text = (message.get("text") or "").strip()
            if not text:
                continue
            lines.append(f"- {self.user_label(message.get('user'))}: {text}")
        return "\n".join(lines)

    def call_history(self, channel_id: str, oldest: str) -> list[dict[str, Any]]:
        messages: list[dict[str, Any]] = []
        cursor: Optional[str] = None

        while True:
            response = self.slack_call(
                self.slack.conversations_history,
                channel=channel_id,
                oldest=oldest,
                inclusive=False,
                limit=15,
                cursor=cursor,
            )
            messages.extend(response.get("messages", []))
            cursor = response.get("response_metadata", {}).get("next_cursor")
            if not cursor:
                break

        return sorted(messages, key=lambda message: float(message.get("ts", "0")))

    def is_human_message(self, message: dict[str, Any]) -> bool:
        if message.get("subtype"):
            return False
        if not message.get("text"):
            return False
        return True

    def build_classification_prompt(
        self,
        *,
        text: str,
        sender_label: str,
        conversation_label: str,
        relevance_reason: str,
        context_text: str,
        links_text: str,
    ) -> str:
        context_block = context_text or "Sin contexto reciente."
        links_block = links_text or "Sin URLs detectadas."
        return f"""
Sos el asistente personal de trabajo de Ivan Rodríguez.

Analizá este mensaje de Slack y devolvé una clasificación estructurada.

Datos:
- Conversación: {conversation_label}
- Remitente: {sender_label}
- Razón de relevancia: {relevance_reason}

Contexto reciente de la conversación:
{context_block}

URLs detectadas:
{links_block}

Mensaje:
{text}

Reglas:
- No inventes datos.
- Si es un DM casual sin pedido concreto, is_actionable=false.
- Si hay un pedido de trabajo o seguimiento, is_actionable=true.
- Identificá qué esperan que haga Ivan.
- Si falta información para actuar, listala.
- El draft_reply debe ser breve, profesional, natural y en español argentino neutro.
- No propongas ejecutar acciones riesgosas sin aprobación humana.
""".strip()

    def invoke_classification(
        self,
        *,
        text: str,
        sender_label: str,
        conversation_label: str,
        relevance_reason: str,
        context_text: str,
        links_text: str,
    ) -> SlackClassification:
        prompt = self.build_classification_prompt(
            text=text,
            sender_label=sender_label,
            conversation_label=conversation_label,
            relevance_reason=relevance_reason,
            context_text=context_text,
            links_text=links_text,
        )
        return self.get_classifier().invoke(prompt)

    def decide_relevance(self, state: AgentState) -> AgentState:
        message = state["message"]
        conversation = state["conversation"]
        my_user_id = state["my_user_id"]
        text = message.get("text", "")
        sender_id = message.get("user")

        if sender_id == my_user_id and not self.config.include_self_for_test:
            return {
                **state,
                "relevant": False,
                "relevance_reason": "Mensaje enviado por Ivan.",
            }

        if conversation.get("is_im") or conversation.get("is_mpim"):
            return {
                **state,
                "relevant": True,
                "relevance_reason": "DM o grupo DM entrante.",
            }

        if self.conv_type(conversation) == "public_channel":
            return {
                **state,
                "relevant": False,
                "relevance_reason": "Los canales públicos están fuera de alcance.",
            }

        if self.message_mentions_me(text, my_user_id):
            return {
                **state,
                "relevant": True,
                "relevance_reason": "Mensaje en canal privado mencionando a Ivan por @ o alias textual.",
            }

        return {
            **state,
            "relevant": False,
            "relevance_reason": "No es DM ni menciona a Ivan en un canal privado.",
        }

    def extract_urls(self, state: AgentState) -> AgentState:
        if not state.get("relevant"):
            return state

        message_text = state["message"].get("text", "")
        return {
            **state,
            "message_links": self.extract_message_links(message_text),
        }

    def enrich_urls(self, state: AgentState) -> AgentState:
        if not state.get("relevant"):
            return state

        enriched_links = [self.enrich_message_link(link) for link in state.get("message_links", [])]
        return {
            **state,
            "message_links": enriched_links,
        }

    def persist_relevance(self, state: AgentState) -> AgentState:
        message = state["message"]
        conversation = state["conversation"]
        context_text = ""
        if state.get("relevant"):
            context_text = self.fetch_recent_context(conversation["id"], message["ts"])
        self.upsert_processed_relevance(
            channel_id=conversation["id"],
            message_ts=message["ts"],
            user_id=message.get("user"),
            sender_label=state["sender_label"],
            conversation_label=state["conversation_label"],
            raw_text=message.get("text", ""),
            relevant=bool(state.get("relevant")),
            relevance_reason=state.get("relevance_reason", ""),
            context_text=context_text,
        )
        self.replace_message_links(
            conversation["id"],
            message["ts"],
            state.get("message_links", []),
        )
        return state

    def maybe_classify(self, state: AgentState) -> AgentState:
        if not state.get("relevant"):
            return state

        classification = self.invoke_classification(
            text=state["message"].get("text", ""),
            sender_label=state["sender_label"],
            conversation_label=state["conversation_label"],
            relevance_reason=state.get("relevance_reason", ""),
            context_text=self.get_message_context(
                channel_id=state["conversation"]["id"],
                message_ts=state["message"]["ts"],
            ),
            links_text=self.get_message_links_context(
                channel_id=state["conversation"]["id"],
                message_ts=state["message"]["ts"],
            ),
        )
        return {
            **state,
            "classification": classification,
        }

    def get_message_context(self, channel_id: str, message_ts: str) -> str:
        with self.db_connect() as conn:
            row = conn.execute(
                """
                SELECT context_text
                FROM processed_messages
                WHERE channel_id = ? AND message_ts = ?
                """,
                (channel_id, message_ts),
            ).fetchone()
        if not row:
            return ""
        return row["context_text"] or ""

    def persist_classification(self, state: AgentState) -> AgentState:
        if not state.get("relevant"):
            return state

        classification = state.get("classification")
        if classification is None:
            return state

        message = state["message"]
        conversation = state["conversation"]

        self.mark_processed_done(
            channel_id=conversation["id"],
            message_ts=message["ts"],
            classification=classification,
        )

        if classification.is_actionable:
            inserted = self.save_task(
                channel_id=conversation["id"],
                message_ts=message["ts"],
                user_id=message.get("user"),
                sender_label=state["sender_label"],
                conversation_label=state["conversation_label"],
                classification=classification,
            )
            if inserted and self.config.trello_enabled and self.config.trello_auto_create:
                self.sync_task_to_trello_by_message(conversation["id"], message["ts"])

            print("\n[bold cyan]Nueva tarea detectada[/bold cyan]")
            print(f"[bold]Conversación:[/bold] {state['conversation_label']}")
            print(f"[bold]De:[/bold] {state['sender_label']}")
            print(f"[bold]Resumen:[/bold] {classification.summary}")
            print(f"[bold]Prioridad:[/bold] {classification.priority}")
            print(f"[bold]Categoría:[/bold] {classification.category}")
            print(f"[bold]Acción:[/bold] {classification.requested_action}")
            print(f"[bold]Falta:[/bold] {', '.join(classification.missing_information) or 'Nada'}")
            print(f"[bold]Borrador:[/bold] {classification.draft_reply}\n")

        return state

    def build_graph(self):
        if self._graph is not None:
            return self._graph

        graph = StateGraph(AgentState)
        graph.add_node("decide_relevance", self.decide_relevance)
        graph.add_node("extract_urls", self.extract_urls)
        graph.add_node("enrich_urls", self.enrich_urls)
        graph.add_node("persist_relevance", self.persist_relevance)
        graph.add_node("maybe_classify", self.maybe_classify)
        graph.add_node("persist_classification", self.persist_classification)

        graph.add_edge(START, "decide_relevance")
        graph.add_edge("decide_relevance", "extract_urls")
        graph.add_edge("extract_urls", "enrich_urls")
        graph.add_edge("enrich_urls", "persist_relevance")
        graph.add_edge("persist_relevance", "maybe_classify")
        graph.add_edge("maybe_classify", "persist_classification")
        graph.add_edge("persist_classification", END)

        self._graph = graph.compile()
        return self._graph

    def process_message(self, message: dict[str, Any], conversation: dict[str, Any], my_user_id: str) -> None:
        sender_label = self.user_label(message.get("user"))
        conversation_label = self.conversation_label(conversation)
        state: AgentState = {
            "message": message,
            "conversation": conversation,
            "my_user_id": my_user_id,
            "sender_label": sender_label,
            "conversation_label": conversation_label,
        }

        try:
            self.build_graph().invoke(state)
        except Exception as exc:
            self.mark_processed_failed(
                channel_id=conversation["id"],
                message_ts=message["ts"],
                error_message=str(exc),
            )
            print(
                f"[yellow]Clasificación falló para {conversation_label} ({message['ts']}): "
                f"{exc}[/yellow]"
            )

    def fetch_retryable_messages(self, limit: int = 25) -> list[sqlite3.Row]:
        with self.db_connect() as conn:
            rows = conn.execute(
                """
                SELECT channel_id,
                       message_ts,
                       user_id,
                       sender_label,
                       conversation_label,
                       raw_text,
                       context_text,
                       relevance_reason,
                       classification_status
                FROM processed_messages
                WHERE relevant = 1
                  AND classification_status IN ('pending', 'failed')
                ORDER BY processed_at ASC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return list(rows)

    def retry_failed_messages(self, limit: int = 25) -> int:
        retry_rows = self.fetch_retryable_messages(limit=limit)
        recovered = 0

        for row in retry_rows:
            try:
                classification = self.invoke_classification(
                    text=row["raw_text"] or "",
                    sender_label=row["sender_label"] or "unknown",
                    conversation_label=row["conversation_label"] or row["channel_id"],
                    relevance_reason=row["relevance_reason"] or "Mensaje previamente marcado como relevante.",
                    context_text=row["context_text"] or "",
                    links_text=self.get_message_links_context(row["channel_id"], row["message_ts"]),
                )
                self.mark_processed_done(
                    channel_id=row["channel_id"],
                    message_ts=row["message_ts"],
                    classification=classification,
                )
                if classification.is_actionable:
                    self.save_task(
                        channel_id=row["channel_id"],
                        message_ts=row["message_ts"],
                        user_id=row["user_id"],
                        sender_label=row["sender_label"] or "unknown",
                        conversation_label=row["conversation_label"] or row["channel_id"],
                        classification=classification,
                    )
                recovered += 1
            except Exception as exc:
                self.mark_processed_failed(
                    channel_id=row["channel_id"],
                    message_ts=row["message_ts"],
                    error_message=str(exc),
                )

        if recovered:
            print(f"[green]Reintentos recuperados:[/green] {recovered}")
        return recovered

    def bootstrap(self) -> None:
        self.init_db()
        my_user_id = self.get_my_user_id()
        conversations = self.list_conversations()
        baseline_ts = now_slack_ts()

        for conversation in conversations:
            self.upsert_conversation(conversation, last_seen_ts=baseline_ts)

        print("[bold green]Bootstrap completo[/bold green]")
        print(f"User ID monitoreado: {my_user_id}")
        print(f"Conversaciones registradas: {len(conversations)}")
        print(f"Baseline ts: {baseline_ts}")
        print("[yellow]No se procesó historial viejo. Desde ahora leerá mensajes nuevos.[/yellow]")

    def poll_once(self) -> None:
        self.init_db()
        my_user_id = self.get_my_user_id()
        conversations = self.list_conversations()
        print(f"[green]Conversaciones accesibles:[/green] {len(conversations)}")

        self.retry_failed_messages(limit=25)
        if self.config.trello_enabled and self.config.trello_auto_create:
            self.sync_pending_trello_tasks(limit=25)
        total_new = 0

        for conversation in conversations:
            self.upsert_conversation(conversation)
            channel_id = conversation["id"]
            last_seen_ts = self.get_last_seen_ts(channel_id)

            try:
                messages = self.call_history(channel_id, oldest=last_seen_ts)
            except SlackApiError as exc:
                print(
                    f"[yellow]No pude leer {self.conversation_label(conversation)}: "
                    f"{exc.response.get('error')}[/yellow]"
                )
                self._sleep(self.config.slack_sleep_seconds)
                continue

            max_ts = last_seen_ts
            if messages:
                print(
                    f"[dim]{self.conversation_label(conversation)}: "
                    f"{len(messages)} mensaje(s) nuevo(s)[/dim]"
                )

            for message in messages:
                ts_value = message.get("ts")
                if not ts_value:
                    continue

                max_ts = max(max_ts, ts_value, key=lambda item: float(item))
                if not self.is_human_message(message):
                    continue

                status = self.get_processed_status(channel_id, ts_value)
                if status in {"done", "ignored"}:
                    continue

                self.process_message(message, conversation, my_user_id=my_user_id)
                total_new += 1

            self.set_last_seen_ts(channel_id, max_ts)
            self._sleep(self.config.slack_sleep_seconds)

        self.retry_failed_messages(limit=25)
        if self.config.trello_enabled and self.config.trello_auto_create:
            self.sync_pending_trello_tasks(limit=25)
        print(f"[green]Poll terminado.[/green] Mensajes humanos nuevos procesados: {total_new}")

    def fetch_open_tasks_for_brief(self, limit: int = 200) -> list[sqlite3.Row]:
        with self.db_connect() as conn:
            rows = conn.execute(
                """
                SELECT tasks.*,
                       processed_messages.raw_text,
                       processed_messages.context_text,
                       (
                           SELECT COUNT(*)
                           FROM message_links
                           WHERE message_links.channel_id = tasks.channel_id
                             AND message_links.message_ts = tasks.message_ts
                       ) AS link_count
                FROM tasks
                LEFT JOIN processed_messages
                  ON processed_messages.channel_id = tasks.channel_id
                 AND processed_messages.message_ts = tasks.message_ts
                WHERE COALESCE(tasks.status, 'new') NOT IN (
                    'done',
                    'ignored',
                    'dismissed',
                    'archived',
                    'responded'
                )
                ORDER BY tasks.created_at DESC, tasks.id DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return list(rows)

    def task_brief_info(self, row: sqlite3.Row, now_local: datetime) -> dict[str, Any]:
        try:
            classification = json.loads(row["classification_json"] or "{}")
        except json.JSONDecodeError:
            classification = {}

        created_at = parse_iso_datetime(row["created_at"])
        created_local = created_at.astimezone(now_local.tzinfo) if created_at else None
        age_days = (now_local.date() - created_local.date()).days if created_local else 0
        snoozed_until = parse_iso_datetime(row["snoozed_until"])
        snoozed_local = snoozed_until.astimezone(now_local.tzinfo) if snoozed_until else None
        external_systems = [
            item.lower()
            for item in coerce_string_list(classification.get("external_systems"))
        ]
        missing_information = coerce_string_list(classification.get("missing_information"))

        return {
            "id": row["id"],
            "created_at": created_at,
            "created_local": created_local,
            "age_days": age_days,
            "sender_label": row["sender_label"] or "unknown",
            "conversation_label": row["conversation_label"] or row["channel_id"],
            "summary": row["summary"] or "Sin resumen",
            "requested_action": row["requested_action"] or "Sin acción especificada",
            "priority": (row["priority"] or "medium").lower(),
            "category": (row["category"] or "other").lower(),
            "status": row["status"] or "new",
            "trello_status": row["trello_status"] or "pending",
            "trello_card_url": row["trello_card_url"] or "",
            "trello_last_error": row["trello_last_error"] or "",
            "reviewed_at": row["reviewed_at"] or "",
            "snoozed_until": snoozed_until,
            "snoozed_local": snoozed_local,
            "reply_approved_at": row["reply_approved_at"] or "",
            "reply_sent_at": row["reply_sent_at"] or "",
            "reply_ts": row["reply_ts"] or "",
            "reply_error": row["reply_error"] or "",
            "manual_reply": row["manual_reply"] or "",
            "needs_reply": bool(classification.get("needs_reply")),
            "needs_external_system": bool(classification.get("needs_external_system")),
            "external_systems": external_systems,
            "missing_information": missing_information,
            "suggested_next_step": classification.get("suggested_next_step") or "",
            "draft_reply": classification.get("draft_reply") or "",
            "link_count": int(row["link_count"] or 0),
        }

    def task_brief_line(self, task: dict[str, Any]) -> str:
        created_local = task.get("created_local")
        created_text = created_local.strftime("%Y-%m-%d %H:%M") if created_local else "fecha desconocida"
        summary = compact_text(task["summary"], 110)
        requested_action = compact_text(task["requested_action"], 130)
        line = (
            f"- #{task['id']} {task['priority']}/{task['category']}: {summary} "
            f"({task['sender_label']} en {task['conversation_label']}, {created_text})"
        )
        details = [f"Acción: {requested_action}"]
        if task["missing_information"]:
            details.append(f"Falta: {compact_text(', '.join(task['missing_information']), 120)}")
        if task["trello_status"]:
            details.append(f"Trello: {task['trello_status']}")
        if task["status"] == "reply_approved" and not task["reply_sent_at"]:
            details.append("Respuesta aprobada sin enviar")
        if task["reply_error"]:
            details.append(f"Error Slack: {compact_text(task['reply_error'], 90)}")
        return line + "\n  " + " | ".join(details)

    def add_brief_section(
        self,
        lines: list[str],
        title: str,
        tasks: list[dict[str, Any]],
        *,
        empty_text: str,
        limit: int,
    ) -> None:
        lines.append("")
        lines.append(title)
        if not tasks:
            lines.append(f"- {empty_text}")
            return
        for task in tasks[:limit]:
            lines.append(self.task_brief_line(task))
        if len(tasks) > limit:
            lines.append(f"- Y {len(tasks) - limit} más.")

    def build_brief(
        self,
        *,
        section_limit: int = 8,
        scan_limit: int = 200,
        stale_days: int = 3,
        now: Optional[datetime] = None,
    ) -> str:
        self.init_db()
        now_utc = now or datetime.now(timezone.utc)
        if now_utc.tzinfo is None:
            now_utc = now_utc.replace(tzinfo=timezone.utc)
        now_local = now_utc.astimezone()
        section_limit = max(1, section_limit)
        rows = self.fetch_open_tasks_for_brief(limit=max(scan_limit, section_limit))
        tasks = [
            task
            for task in (self.task_brief_info(row, now_local) for row in rows)
            if not task["snoozed_until"] or task["snoozed_until"] <= now_utc
        ]

        priority_order = {"urgent": 0, "high": 1, "medium": 2, "low": 3}
        tasks_by_priority = sorted(
            tasks,
            key=lambda task: (
                priority_order.get(task["priority"], 9),
                -(task["created_at"].timestamp() if task["created_at"] else 0),
            ),
        )
        today_tasks = [
            task for task in tasks
            if task["created_local"] and task["created_local"].date() == now_local.date()
        ]
        urgent_tasks = [
            task for task in tasks_by_priority
            if task["priority"] in {"urgent", "high"}
        ]
        blocked_tasks = [task for task in tasks_by_priority if task["missing_information"]]
        reply_tasks = [
            task
            for task in tasks_by_priority
            if task["needs_reply"] and task["draft_reply"] and task["status"] != "reply_approved"
        ]
        approved_reply_tasks = [
            task
            for task in tasks_by_priority
            if task["status"] == "reply_approved" and not task["reply_sent_at"]
        ]
        salesforce_tasks = [
            task for task in tasks_by_priority
            if task["category"] == "salesforce" or "salesforce" in task["external_systems"]
        ]
        software_tasks = [
            task for task in tasks_by_priority
            if task["category"] == "software" or "github" in task["external_systems"]
        ]
        stale_tasks = [
            task for task in tasks_by_priority
            if task["status"] == "new" and task["age_days"] >= stale_days
        ]
        trello_failed = [
            task for task in tasks_by_priority
            if task["trello_status"] == "failed"
        ]
        category_counts = Counter(task["category"] for task in tasks)

        lines = [
            "Resumen del agente",
            f"Fecha local: {now_local.strftime('%Y-%m-%d %H:%M')}",
            f"Tareas abiertas: {len(tasks)}",
        ]
        if tasks:
            counters = [
                f"nuevas hoy {len(today_tasks)}",
                f"alta/urgente {len(urgent_tasks)}",
                f"bloqueadas {len(blocked_tasks)}",
                f"necesitan respuesta {len(reply_tasks)}",
                f"aprobadas sin enviar {len(approved_reply_tasks)}",
            ]
            lines.append("Estado: " + " | ".join(counters))
            lines.append(
                "Categorías: "
                + ", ".join(f"{category} {count}" for category, count in category_counts.most_common())
            )
        else:
            lines.append("No hay tareas abiertas en SQLite.")

        self.add_brief_section(
            lines,
            "Nuevas tareas detectadas hoy",
            today_tasks,
            empty_text="Sin tareas nuevas hoy.",
            limit=section_limit,
        )
        self.add_brief_section(
            lines,
            "Prioridad alta/urgente",
            urgent_tasks,
            empty_text="Sin tareas high o urgent.",
            limit=section_limit,
        )
        self.add_brief_section(
            lines,
            "Tareas bloqueadas",
            blocked_tasks,
            empty_text="No hay tareas con información faltante.",
            limit=section_limit,
        )
        self.add_brief_section(
            lines,
            "Necesitan respuesta",
            reply_tasks,
            empty_text="No hay borradores pendientes de respuesta.",
            limit=section_limit,
        )
        self.add_brief_section(
            lines,
            "Respuestas aprobadas sin enviar",
            approved_reply_tasks,
            empty_text="No hay respuestas aprobadas pendientes de envío.",
            limit=section_limit,
        )
        self.add_brief_section(
            lines,
            "Relacionadas con Salesforce",
            salesforce_tasks,
            empty_text="Sin tareas de Salesforce.",
            limit=section_limit,
        )
        self.add_brief_section(
            lines,
            "Relacionadas con software",
            software_tasks,
            empty_text="Sin tareas de software.",
            limit=section_limit,
        )
        self.add_brief_section(
            lines,
            f"Viejas sin mover ({stale_days}+ días)",
            stale_tasks,
            empty_text="No hay tareas viejas sin mover.",
            limit=section_limit,
        )

        lines.append("")
        lines.append("Sugerencias")
        suggestions: list[str] = []
        for task in blocked_tasks[:2]:
            suggestions.append(
                f"- Pedir contexto para #{task['id']}: "
                f"{compact_text(', '.join(task['missing_information']), 130)}"
            )
        for task in reply_tasks[:2]:
            suggestions.append(
                f"- Responder a {task['sender_label']} por #{task['id']}: "
                f"{compact_text(task['draft_reply'], 170)}"
            )
        for task in approved_reply_tasks[:2]:
            suggestions.append(
                f"- Enviar respuesta aprobada de #{task['id']} con "
                f"`python main.py approve-reply {task['id']} --send`."
            )
        for task in trello_failed[:2]:
            suggestions.append(
                f"- Revisar Trello en #{task['id']}: "
                f"{compact_text(task['trello_last_error'] or 'falló la sincronización', 140)}"
            )
        for task in stale_tasks[:2]:
            suggestions.append(f"- Decidir si #{task['id']} sigue abierta o se puede cerrar/mover.")
        if not suggestions:
            suggestions.append("- No veo acciones de seguimiento urgentes con los datos actuales.")
        lines.extend(suggestions[:section_limit])

        return "\n".join(lines)

    def print_brief(self, limit: int = 8) -> None:
        print(self.build_brief(section_limit=limit))

    def fetch_tasks_for_review(
        self,
        limit: int = 10,
        now: Optional[datetime] = None,
        *,
        include_approved_replies: bool = False,
    ) -> list[sqlite3.Row]:
        self.init_db()
        current = now or datetime.now(timezone.utc)
        if current.tzinfo is None:
            current = current.replace(tzinfo=timezone.utc)
        closed_statuses = [
            "done",
            "ignored",
            "dismissed",
            "archived",
            "responded",
        ]
        if not include_approved_replies:
            closed_statuses.append("reply_approved")
        status_placeholders = ", ".join("?" for _ in closed_statuses)
        with self.db_connect() as conn:
            rows = conn.execute(
                f"""
                SELECT tasks.*,
                       processed_messages.raw_text,
                       processed_messages.context_text,
                       (
                           SELECT COUNT(*)
                           FROM message_links
                           WHERE message_links.channel_id = tasks.channel_id
                             AND message_links.message_ts = tasks.message_ts
                       ) AS link_count
                FROM tasks
                LEFT JOIN processed_messages
                  ON processed_messages.channel_id = tasks.channel_id
                 AND processed_messages.message_ts = tasks.message_ts
                WHERE COALESCE(tasks.status, 'new') NOT IN ({status_placeholders})
                  AND (
                    tasks.snoozed_until IS NULL
                    OR tasks.snoozed_until = ''
                    OR tasks.snoozed_until <= ?
                  )
                ORDER BY
                    CASE LOWER(COALESCE(tasks.priority, 'medium'))
                        WHEN 'urgent' THEN 0
                        WHEN 'high' THEN 1
                        WHEN 'medium' THEN 2
                        WHEN 'low' THEN 3
                        ELSE 4
                    END,
                    tasks.created_at ASC,
                    tasks.id ASC
                LIMIT ?
                """,
                (*closed_statuses, current.isoformat(), limit),
            ).fetchall()
        return list(rows)

    def mark_task_status(self, task_id: int, status: str) -> None:
        with self.db_connect() as conn:
            conn.execute(
                """
                UPDATE tasks
                SET status = ?,
                    reviewed_at = ?,
                    snoozed_until = CASE WHEN ? != 'snoozed' THEN NULL ELSE snoozed_until END
                WHERE id = ?
                """,
                (status, now_iso(), status, task_id),
            )
        self.record_task_event(task_id, "status_changed", {"status": status})

    def mark_task_reply_approved(self, task_id: int, reply_text: str) -> None:
        with self.db_connect() as conn:
            conn.execute(
                """
                UPDATE tasks
                SET status = 'reply_approved',
                    reviewed_at = ?,
                    reply_approved_at = ?,
                    manual_reply = ?,
                    reply_error = NULL,
                    snoozed_until = NULL
                WHERE id = ?
                """,
                (now_iso(), now_iso(), reply_text, task_id),
            )
        self.record_task_event(task_id, "reply_approved", {"reply_text": reply_text})

    def mark_task_reply_sent(self, task_id: int, reply_ts: str) -> None:
        with self.db_connect() as conn:
            conn.execute(
                """
                UPDATE tasks
                SET status = 'responded',
                    reviewed_at = ?,
                    reply_sent_at = ?,
                    reply_ts = ?,
                    reply_error = NULL,
                    snoozed_until = NULL
                WHERE id = ?
                """,
                (now_iso(), now_iso(), reply_ts, task_id),
            )
        self.record_task_event(task_id, "reply_sent", {"reply_ts": reply_ts})

    def mark_task_reply_failed(self, task_id: int, error_message: str) -> None:
        with self.db_connect() as conn:
            conn.execute(
                """
                UPDATE tasks
                SET reviewed_at = ?,
                    reply_error = ?
                WHERE id = ?
                """,
                (now_iso(), error_message[:1000], task_id),
            )
        self.record_task_event(task_id, "reply_failed", {"error": error_message[:1000]})

    def reply_text_for_task(self, task_row: sqlite3.Row) -> str:
        if task_row["manual_reply"]:
            return task_row["manual_reply"].strip()

        try:
            classification = json.loads(task_row["classification_json"] or "{}")
        except json.JSONDecodeError:
            classification = {}
        return str(classification.get("draft_reply") or "").strip()

    def send_approved_reply(self, task_id: int) -> bool:
        self.init_db()
        task_row = self.get_task_by_id(task_id)
        if not task_row:
            raise RuntimeError(f"No encontré la tarea #{task_id}.")

        reply_text = self.reply_text_for_task(task_row)
        if not reply_text:
            raise RuntimeError(f"La tarea #{task_id} no tiene respuesta aprobada para enviar.")

        try:
            response = self.slack_call(
                self.slack.chat_postMessage,
                channel=task_row["channel_id"],
                thread_ts=task_row["message_ts"],
                text=reply_text,
                unfurl_links=False,
                unfurl_media=False,
            )
        except Exception as exc:
            self.mark_task_reply_failed(task_id, str(exc))
            print(f"[yellow]No pude enviar la respuesta de la tarea #{task_id}: {exc}[/yellow]")
            return False

        reply_ts = str(response.get("ts") or "")
        self.mark_task_reply_sent(task_id, reply_ts)
        print(f"[green]Respuesta enviada en Slack para tarea #{task_id}.[/green]")
        return True

    def approve_reply(self, task_id: int, reply_text: Optional[str] = None, *, send: bool = False) -> bool:
        self.init_db()
        task_row = self.get_task_by_id(task_id)
        if not task_row:
            raise RuntimeError(f"No encontré la tarea #{task_id}.")

        approved_text = (reply_text or self.reply_text_for_task(task_row)).strip()
        if not approved_text:
            raise RuntimeError(f"La tarea #{task_id} no tiene borrador para aprobar.")

        self.mark_task_reply_approved(task_id, approved_text)
        if send:
            return self.send_approved_reply(task_id)
        return True

    def snooze_task(self, task_id: int, snoozed_until: datetime) -> None:
        if snoozed_until.tzinfo is None:
            snoozed_until = snoozed_until.replace(tzinfo=timezone.utc)
        with self.db_connect() as conn:
            conn.execute(
                """
                UPDATE tasks
                SET status = 'snoozed',
                    reviewed_at = ?,
                    snoozed_until = ?
                WHERE id = ?
                """,
                (now_iso(), snoozed_until.astimezone(timezone.utc).isoformat(), task_id),
            )
        self.record_task_event(
            task_id,
            "snoozed",
            {"snoozed_until": snoozed_until.astimezone(timezone.utc).isoformat()},
        )

    def build_review_task_text(self, task: dict[str, Any]) -> str:
        created_local = task.get("created_local")
        created_text = created_local.strftime("%Y-%m-%d %H:%M") if created_local else "fecha desconocida"
        lines = [
            "",
            f"Tarea #{task['id']} [{task['priority']}/{task['category']}]",
            f"De: {task['sender_label']}",
            f"En: {task['conversation_label']}",
            f"Fecha: {created_text}",
            f"Estado: {task['status']} | Trello: {task['trello_status']}",
            "",
            f"Resumen: {task['summary']}",
            f"Acción: {task['requested_action']}",
        ]
        if task["suggested_next_step"]:
            lines.append(f"Siguiente paso: {task['suggested_next_step']}")
        if task["missing_information"]:
            lines.append(f"Falta: {', '.join(task['missing_information'])}")
        if task["draft_reply"]:
            lines.extend(["", "Borrador:", task["draft_reply"]])
        if task["manual_reply"] and task["manual_reply"] != task["draft_reply"]:
            lines.extend(["", "Respuesta aprobada:", task["manual_reply"]])
        if task["reply_error"]:
            lines.extend(["", f"Error de envío Slack: {task['reply_error']}"])
        if task["trello_card_url"]:
            lines.extend(["", f"Trello: {task['trello_card_url']}"])
        return "\n".join(lines)

    def review_tasks(self, limit: int = 10, *, send_replies: bool = False) -> int:
        now_local = datetime.now(timezone.utc).astimezone()
        send_mode = send_replies or self.config.slack_send_approved_replies
        rows = self.fetch_tasks_for_review(limit=limit, include_approved_replies=send_mode)
        if not rows:
            print("[dim]No hay tareas pendientes para review.[/dim]")
            return 0

        reviewed = 0
        print("[bold magenta]Review de tareas[/bold magenta]")
        if send_mode:
            print("[yellow]Modo envío activo: aprobar o editar también publica la respuesta en Slack.[/yellow]")
            print("[dim]Acciones: a=aprobar y enviar, e=editar y enviar, i=ignorar, t=abrir Trello, s=snooze, d=done, enter=saltar, q=salir[/dim]")
        else:
            print("[dim]Acciones: a=aprobar borrador, e=editar, i=ignorar, t=abrir Trello, s=snooze, d=done, enter=saltar, q=salir[/dim]")

        for row in rows:
            task = self.task_brief_info(row, now_local)
            print(escape(self.build_review_task_text(task)))

            while True:
                action = self._input("\nAcción > ").strip().lower()
                if action in {"q", "quit", "salir"}:
                    print("[yellow]Review interrumpido.[/yellow]")
                    return reviewed
                if action == "":
                    print("[dim]Saltada.[/dim]")
                    break
                if action == "a":
                    reply_text = task["manual_reply"] or task["draft_reply"]
                    if not reply_text:
                        print("[yellow]Esta tarea no tiene borrador para aprobar. Usá [e] para escribir uno.[/yellow]")
                        continue
                    sent = self.approve_reply(task["id"], reply_text, send=send_mode)
                    if send_mode and sent:
                        print(f"[green]Borrador aprobado y enviado para tarea #{task['id']}.[/green]")
                    elif send_mode:
                        print(f"[yellow]Borrador aprobado, pero el envío falló para tarea #{task['id']}.[/yellow]")
                    else:
                        print(f"[green]Borrador aprobado para tarea #{task['id']}.[/green]")
                    reviewed += 1
                    break
                if action == "e":
                    edited = self._input("Borrador editado > ").strip()
                    if not edited:
                        print("[yellow]No guardé cambios porque el borrador quedó vacío.[/yellow]")
                        continue
                    sent = self.approve_reply(task["id"], edited, send=send_mode)
                    if send_mode and sent:
                        print(f"[green]Borrador editado, aprobado y enviado para tarea #{task['id']}.[/green]")
                    elif send_mode:
                        print(f"[yellow]Borrador editado y aprobado, pero el envío falló para tarea #{task['id']}.[/yellow]")
                    else:
                        print(f"[green]Borrador editado y aprobado para tarea #{task['id']}.[/green]")
                    reviewed += 1
                    break
                if action == "i":
                    self.mark_task_status(task["id"], "ignored")
                    print(f"[green]Tarea #{task['id']} ignorada.[/green]")
                    reviewed += 1
                    break
                if action == "d":
                    self.mark_task_status(task["id"], "done")
                    print(f"[green]Tarea #{task['id']} marcada como done.[/green]")
                    reviewed += 1
                    break
                if action == "s":
                    raw_snooze = self._input("Snooze hasta (1d, 4h, 1w, mañana, YYYY-MM-DD) > ").strip()
                    snoozed_until = parse_snooze_until(raw_snooze)
                    if not snoozed_until:
                        print("[yellow]No entendí ese snooze. Probá con 4h, 1d, 1w o YYYY-MM-DD.[/yellow]")
                        continue
                    self.snooze_task(task["id"], snoozed_until)
                    local_until = snoozed_until.astimezone().strftime("%Y-%m-%d %H:%M")
                    print(f"[green]Tarea #{task['id']} pospuesta hasta {local_until}.[/green]")
                    reviewed += 1
                    break
                if action == "t":
                    if not task["trello_card_url"]:
                        print("[yellow]Esta tarea todavía no tiene URL de Trello.[/yellow]")
                        continue
                    opened = self._open_url(task["trello_card_url"])
                    if opened:
                        print(f"[green]Abrí Trello:[/green] {task['trello_card_url']}")
                        self.record_task_event(task["id"], "trello_opened", {"url": task["trello_card_url"]})
                    else:
                        print(f"[yellow]No pude abrir Trello automáticamente:[/yellow] {task['trello_card_url']}")
                    continue

                print("[yellow]Acción no reconocida. Usá a/e/i/t/s/d, enter para saltar o q para salir.[/yellow]")

        print(f"[green]Review terminado.[/green] Tareas actualizadas: {reviewed}")
        return reviewed

    def print_review(self, limit: int = 10, *, send_replies: bool = False) -> None:
        self.review_tasks(limit=limit, send_replies=send_replies)

    def print_tasks(self, limit: int = 20) -> None:
        self.init_db()
        with self.db_connect() as conn:
            rows = conn.execute(
                """
                SELECT id,
                       created_at,
                       priority,
                       category,
                       sender_label,
                       conversation_label,
                       summary,
                       requested_action,
                       status,
                       trello_status,
                       trello_card_url
                FROM tasks
                ORDER BY id DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()

        if not rows:
            print("[dim]No hay tareas guardadas todavía.[/dim]")
            return

        print("\n[bold magenta]Tareas recientes[/bold magenta]")
        for row in rows:
            print(
                f"\n[bold]#{row['id']}[/bold] "
                f"[{row['status']}] [{row['priority']}/{row['category']}]"
            )
            print(f"[bold]Fecha:[/bold] {row['created_at']}")
            print(f"[bold]De:[/bold] {row['sender_label']}")
            print(f"[bold]En:[/bold] {row['conversation_label']}")
            print(f"[bold]Resumen:[/bold] {row['summary']}")
            print(f"[bold]Acción:[/bold] {row['requested_action']}")
            print(f"[bold]Trello:[/bold] {row['trello_status']}")
            if row["trello_card_url"]:
                print(f"[bold]Card:[/bold] {row['trello_card_url']}")

    def loop(self) -> None:
        print("[bold green]Slack Personal Agent iniciado[/bold green]")
        print(f"Poll cada {self.config.poll_seconds}s")
        print(f"Sleep entre llamadas Slack: {self.config.slack_sleep_seconds}s")

        while True:
            try:
                self.poll_once()
                self.print_tasks(limit=10)
            except KeyboardInterrupt:
                print("[red]Detenido por usuario.[/red]")
                break
            except Exception as exc:
                print(f"[red]Error general:[/red] {exc}")

            self._sleep(self.config.poll_seconds)

    def trello_auth_url(self) -> str:
        if not self.config.trello_api_key:
            raise TrelloError("Falta TRELLO_API_KEY para construir la URL de autorización.")
        key_error = self.trello_api_key_error()
        if key_error:
            raise TrelloError(key_error)
        return build_trello_token_url(self.config.trello_api_key)

    def install_autostart(self) -> None:
        artifacts = install_launch_agents(
            project_dir=Path(__file__).resolve().parent,
            ollama_base_url=self.config.ollama_base_url,
        )
        print("[green]Autostart instalado.[/green]")
        print(f"Ollama plist: {artifacts.ollama_plist}")
        print(f"Agente plist: {artifacts.agent_plist}")
        print(f"Logs: {artifacts.log_dir}")

    def uninstall_autostart(self) -> None:
        artifacts = uninstall_launch_agents(project_dir=Path(__file__).resolve().parent)
        print("[green]Autostart desinstalado.[/green]")
        print(f"LaunchAgents: {artifacts.ollama_plist.parent}")

    def print_trello_lists(self) -> None:
        client = self.get_trello_client()
        boards = client.list_boards_with_lists()
        if not boards:
            print("[yellow]No encontré boards accesibles en Trello.[/yellow]")
            return

        for board in boards:
            print(f"\n[bold]{board.get('name')}[/bold] ({board.get('id')})")
            print(board.get("url"))
            for trello_list in board.get("lists", []):
                print(f"- {trello_list.get('name')} | {trello_list.get('id')}")

    def doctor(self) -> bool:
        self.init_db()
        ok = True

        print("[bold green]Doctor del agente[/bold green]")
        print(f"Proveedor activo: {self.config.model_provider}")

        if self.config.model_provider == "ollama":
            hint_level, hint_text = local_model_fit_hint(self.config.ollama_model)
            color = {"ok": "green", "caution": "yellow", "warning": "red"}.get(hint_level, "white")
            print(f"[{color}]Modelo local: {self.config.ollama_model}[/{color}]")
            print(f"[{color}]Consejo:[/{color}] {hint_text}")
        else:
            print(f"Modelo remoto: {self.config.groq_model}")

        try:
            auth = self.slack_call(self.slack.auth_test)
            print("[green]Slack auth OK[/green]")
            print(f"Workspace: {auth.get('team')}")
            print(f"Usuario: {auth.get('user')}")
            print(f"User ID: {auth.get('user_id')}")
        except Exception as exc:
            ok = False
            print(f"[red]Slack auth falló:[/red] {exc}")

        try:
            conversations = self.list_conversations()
            counts = Counter(self.conv_type(conversation) for conversation in conversations)
            print(f"Conversaciones accesibles: {len(conversations)}")
            print(
                "Detalle: "
                f"public_channel={counts.get('public_channel', 0)}, "
                f"private_channel={counts.get('private_channel', 0)}, "
                f"im={counts.get('im', 0)}, "
                f"mpim={counts.get('mpim', 0)}"
            )
        except Exception as exc:
            ok = False
            print(f"[red]No pude listar conversaciones:[/red] {exc}")

        try:
            doctor_model = self._structured_model_factory(DoctorCheck)
            result = doctor_model.invoke(
                "Respondé con status='ok' y un summary corto indicando que el proveedor está listo."
            )
            print("[green]Inferencia del proveedor OK[/green]")
            print(f"Resumen proveedor: {result.summary}")
        except Exception as exc:
            ok = False
            print(f"[red]Inferencia del proveedor falló:[/red] {exc}")
            if self.config.model_provider == "ollama":
                print(
                    "[yellow]Si estás usando Ollama, verificá `ollama serve` y que el modelo "
                    f"`{self.config.ollama_model}` esté descargado.[/yellow]"
                )

        if self.config.trello_enabled:
            print("[bold]Chequeo Trello[/bold]")
            if not self.has_trello_auth_config():
                ok = False
                print(
                    "[red]Trello está habilitado pero faltan TRELLO_API_KEY o TRELLO_TOKEN.[/red]"
                )
            else:
                key_error = self.trello_api_key_error()
                if key_error:
                    ok = False
                    print(f"[red]{key_error}[/red]")
                    print("[yellow]La API key correcta se genera desde Trello Power-Ups, no desde Atlassian Account API Tokens.[/yellow]")
                    return ok
                try:
                    client = self.get_trello_client()
                    me = client.get_me()
                    print("[green]Trello auth OK[/green]")
                    print(f"Cuenta Trello: {me.get('fullName') or me.get('username')}")
                    if self.config.trello_list_id:
                        trello_list = client.get_list(self.config.trello_list_id)
                        print(f"Lista destino: {trello_list.name} ({trello_list.id})")
                    else:
                        ok = False
                        print("[yellow]Falta TRELLO_LIST_ID. Usá `python main.py trello-lists` para descubrirlo.[/yellow]")
                except Exception as exc:
                    ok = False
                    print(f"[red]Trello falló:[/red] {exc}")
        else:
            print("[dim]Trello no está habilitado.[/dim]")

        return ok
