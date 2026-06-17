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
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field
from rich import print
from rich.markup import escape
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError

from autostart import install_launch_agents, uninstall_launch_agents
from audio_transcription import (
    AudioAttachment,
    AudioTranscriber,
    AudioTranscriptResult,
    LocalWhisperTranscriber,
    choose_audio_transcript,
    combine_text_and_audio,
    detect_audio_attachments,
    detect_local_whisper_backend,
    format_audio_transcripts_for_message,
)
from trello_client import TrelloCard, TrelloCardState, TrelloClient, TrelloComment, TrelloError, build_trello_token_url
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
MAX_SLACK_MESSAGE_LENGTH = 40000
DEFAULT_CONTEXT_MAX_AGE_MINUTES = 120
DEFAULT_LOCAL_TIMEZONE = "America/Argentina/Cordoba"
VISUAL_IMAGE_SUFFIXES = {
    "gif",
    "heic",
    "heif",
    "jpeg",
    "jpg",
    "png",
    "webp",
}


class ConfigError(RuntimeError):
    """Raised when required configuration is missing or invalid."""


class TelegramError(RuntimeError):
    """Raised when Telegram API operations fail."""


class TelegramClient:
    def __init__(self, bot_token: str, chat_id: str, timeout: int = 30) -> None:
        self.bot_token = bot_token
        self.chat_id = chat_id
        self.timeout = timeout

    @property
    def api_base(self) -> str:
        return f"https://api.telegram.org/bot{self.bot_token}"

    def _request(self, method: str, path: str, **kwargs: Any) -> dict[str, Any]:
        response = requests.request(
            method,
            f"{self.api_base}{path}",
            timeout=self.timeout,
            **kwargs,
        )
        if response.status_code >= 400:
            raise TelegramError(f"Telegram devolvió {response.status_code}: {response.text[:500]}")
        payload = response.json()
        if not payload.get("ok"):
            raise TelegramError(str(payload.get("description") or "Telegram respondió ok=false."))
        return payload

    def send_message(self, text: str) -> dict[str, Any]:
        return self._request(
            "POST",
            "/sendMessage",
            json={
                "chat_id": self.chat_id,
                "text": text,
                "disable_web_page_preview": True,
            },
        )

    def get_updates(self, offset: Optional[int] = None, limit: int = 20) -> list[dict[str, Any]]:
        payload = self._request(
            "GET",
            "/getUpdates",
            params={
                "offset": offset,
                "limit": limit,
                "timeout": 0,
            },
        )
        return list(payload.get("result") or [])


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


class AudioTranscriptFusion(BaseModel):
    transcript: str


@dataclass(frozen=True)
class SlackVisualAttachment:
    source_message_id: str
    channel_id: str
    message_ts: str
    file_id: str
    filename: str
    mime_type: str
    filetype: str
    url_private: str
    index: int = 0
    size_bytes: int = 0


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

    def get_card(self, card_id: str) -> TrelloCardState:
        ...

    def add_card_comment(self, card_id: str, text: str) -> None:
        ...

    def get_card_comments(self, card_id: str, limit: int = 50) -> list[TrelloComment]:
        ...

    def attach_file_to_card(self, card_id: str, file_path: Path, name: Optional[str] = None) -> str:
        ...

    def add_url_attachment_to_card(self, card_id: str, url: str, name: Optional[str] = None) -> str:
        ...


class TelegramClientProtocol(Protocol):
    def send_message(self, text: str) -> Any:
        ...

    def get_updates(self, offset: Optional[int] = None, limit: int = 20) -> list[dict[str, Any]]:
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


def parse_slack_timestamp(value: Any) -> Optional[datetime]:
    try:
        return datetime.fromtimestamp(float(value), timezone.utc)
    except (TypeError, ValueError, OSError):
        return None


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


def clean_text(value: Any) -> str:
    return str(value or "").strip()


def row_value(row: Any, key: str, default: Any = "") -> Any:
    try:
        value = row[key]
    except (KeyError, IndexError, TypeError):
        return default
    return default if value is None else value


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
    case_grouping_window_minutes: int = 15
    context_max_age_minutes: int = DEFAULT_CONTEXT_MAX_AGE_MINUTES
    local_timezone: str = DEFAULT_LOCAL_TIMEZONE
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
    trello_done_mode: Literal["check", "list", "checklist", "list_or_check"] = "check"
    trello_done_checklist_item_name: str = "Hecho"
    trello_done_list_id: str = ""
    trello_done_list_names: tuple[str, ...] = ("hecho", "done")
    trello_waiting_enabled: bool = True
    trello_waiting_comment_prefix: str = "Pedir:"
    trello_waiting_auto_clear: bool = True
    trello_reply_enabled: bool = True
    trello_reply_comment_prefix: str = "Responder:"
    trello_reply_mark_responded: bool = False
    telegram_enabled: bool = False
    telegram_bot_token: str = ""
    telegram_chat_id: str = ""
    final_reply_mode: Literal["telegram_approval", "slack_auto"] = "telegram_approval"
    audio_transcription_enabled: bool = True
    slack_audio_transcripts_enabled: bool = True
    local_whisper_enabled: bool = True
    local_whisper_model: str = "tiny"
    local_whisper_language: str = "es"
    local_whisper_device: str = "auto"
    local_whisper_compute_type: str = "auto"
    local_whisper_max_seconds: int = 600
    local_whisper_keep_audio_files: bool = False
    local_whisper_cache_dir: str = "~/Library/Application Support/slack-personal-agent/audio"
    audio_transcript_fusion_enabled: bool = True
    audio_transcript_fusion_model: str = "main"
    slack_image_attachments_enabled: bool = True
    trello_attach_slack_images: bool = True
    trello_image_attachment_mode: Literal["upload", "link"] = "upload"
    slack_image_keep_files: bool = False
    slack_image_cache_dir: str = "~/Library/Application Support/slack-personal-agent/images"
    slack_image_max_bytes: int = 15000000
    sync_worker_seconds: int = 60
    sync_waiting_enabled: bool = True
    sync_trello_done_enabled: bool = True
    sync_telegram_poll_enabled: bool = True
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

        trello_done_mode = (get("TRELLO_DONE_MODE", "check") or "check").strip().lower()
        if trello_done_mode not in {"check", "list", "checklist", "list_or_check"}:
            raise ConfigError("TRELLO_DONE_MODE debe ser check, list, checklist o list_or_check.")

        final_reply_mode = (get("FINAL_REPLY_MODE", "telegram_approval") or "telegram_approval").strip().lower()
        if final_reply_mode not in {"telegram_approval", "slack_auto"}:
            raise ConfigError("FINAL_REPLY_MODE debe ser telegram_approval o slack_auto.")

        trello_image_attachment_mode = (get("TRELLO_IMAGE_ATTACHMENT_MODE", "upload") or "upload").strip().lower()
        if trello_image_attachment_mode not in {"upload", "link"}:
            raise ConfigError("TRELLO_IMAGE_ATTACHMENT_MODE debe ser upload o link.")

        poll_seconds_raw = (get("POLL_SECONDS", "300") or "300").strip()
        sleep_seconds_raw = (get("SLACK_SLEEP_SECONDS", "1.2") or "1.2").strip()
        case_grouping_window_minutes_raw = (
            get("CASE_GROUPING_WINDOW_MINUTES", "15") or "15"
        ).strip()
        context_max_age_minutes_raw = (
            get("CONTEXT_MAX_AGE_MINUTES", str(DEFAULT_CONTEXT_MAX_AGE_MINUTES))
            or str(DEFAULT_CONTEXT_MAX_AGE_MINUTES)
        ).strip()
        local_timezone = (get("LOCAL_TIMEZONE", DEFAULT_LOCAL_TIMEZONE) or DEFAULT_LOCAL_TIMEZONE).strip()
        sync_worker_seconds_raw = (get("SYNC_WORKER_SECONDS", "60") or "60").strip()

        try:
            poll_seconds = int(poll_seconds_raw)
        except ValueError as exc:
            raise ConfigError("POLL_SECONDS debe ser un entero.") from exc

        try:
            slack_sleep_seconds = float(sleep_seconds_raw)
        except ValueError as exc:
            raise ConfigError("SLACK_SLEEP_SECONDS debe ser un número.") from exc

        try:
            case_grouping_window_minutes = int(case_grouping_window_minutes_raw)
        except ValueError as exc:
            raise ConfigError("CASE_GROUPING_WINDOW_MINUTES debe ser un entero.") from exc
        if case_grouping_window_minutes < 0:
            raise ConfigError("CASE_GROUPING_WINDOW_MINUTES debe ser mayor o igual a 0.")

        try:
            context_max_age_minutes = int(context_max_age_minutes_raw)
        except ValueError as exc:
            raise ConfigError("CONTEXT_MAX_AGE_MINUTES debe ser un entero.") from exc
        if context_max_age_minutes < 0:
            raise ConfigError("CONTEXT_MAX_AGE_MINUTES debe ser mayor o igual a 0.")

        try:
            ZoneInfo(local_timezone)
        except ZoneInfoNotFoundError as exc:
            raise ConfigError("LOCAL_TIMEZONE debe ser una timezone IANA válida.") from exc

        try:
            sync_worker_seconds = int(sync_worker_seconds_raw)
        except ValueError as exc:
            raise ConfigError("SYNC_WORKER_SECONDS debe ser un entero.") from exc
        if sync_worker_seconds <= 0:
            raise ConfigError("SYNC_WORKER_SECONDS debe ser mayor a 0.")

        max_audio_seconds_raw = (get("LOCAL_WHISPER_MAX_SECONDS", "600") or "600").strip()
        try:
            local_whisper_max_seconds = int(max_audio_seconds_raw)
        except ValueError as exc:
            raise ConfigError("LOCAL_WHISPER_MAX_SECONDS debe ser un entero.") from exc

        slack_image_max_bytes_raw = (get("SLACK_IMAGE_MAX_BYTES", "15000000") or "15000000").strip()
        try:
            slack_image_max_bytes = int(slack_image_max_bytes_raw)
        except ValueError as exc:
            raise ConfigError("SLACK_IMAGE_MAX_BYTES debe ser un entero.") from exc
        if slack_image_max_bytes <= 0:
            raise ConfigError("SLACK_IMAGE_MAX_BYTES debe ser mayor a 0.")

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
        trello_done_list_names = tuple(
            normalize_for_matching(name)
            for name in (get("TRELLO_DONE_LIST_NAMES", "Hecho,Done") or "Hecho,Done").split(",")
            if normalize_for_matching(name)
        )

        return cls(
            slack_user_token=slack_user_token,
            my_slack_user_id=(get("MY_SLACK_USER_ID", "") or "").strip(),
            my_mention_aliases=mention_aliases,
            poll_seconds=poll_seconds,
            slack_sleep_seconds=slack_sleep_seconds,
            include_self_for_test=parse_bool(get("INCLUDE_SELF_FOR_TEST", "false") or "false"),
            case_grouping_window_minutes=case_grouping_window_minutes,
            context_max_age_minutes=context_max_age_minutes,
            local_timezone=local_timezone,
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
            trello_done_mode=trello_done_mode,  # type: ignore[arg-type]
            trello_done_checklist_item_name=(
                get("TRELLO_DONE_CHECKLIST_ITEM_NAME", "Hecho") or "Hecho"
            ).strip(),
            trello_done_list_id=(get("TRELLO_DONE_LIST_ID", "") or "").strip(),
            trello_done_list_names=trello_done_list_names or ("hecho", "done"),
            trello_waiting_enabled=parse_bool(get("TRELLO_WAITING_ENABLED", "true") or "true"),
            trello_waiting_comment_prefix=(get("TRELLO_WAITING_COMMENT_PREFIX", "Pedir:") or "Pedir:").strip(),
            trello_waiting_auto_clear=parse_bool(get("TRELLO_WAITING_AUTO_CLEAR", "true") or "true"),
            trello_reply_enabled=parse_bool(get("TRELLO_REPLY_ENABLED", "true") or "true"),
            trello_reply_comment_prefix=(get("TRELLO_REPLY_COMMENT_PREFIX", "Responder:") or "Responder:").strip(),
            trello_reply_mark_responded=parse_bool(get("TRELLO_REPLY_MARK_RESPONDED", "false") or "false"),
            telegram_enabled=parse_bool(get("TELEGRAM_ENABLED", "false") or "false"),
            telegram_bot_token=(get("TELEGRAM_BOT_TOKEN", "") or "").strip(),
            telegram_chat_id=(get("TELEGRAM_CHAT_ID", "") or "").strip(),
            final_reply_mode=final_reply_mode,  # type: ignore[arg-type]
            audio_transcription_enabled=parse_bool(get("AUDIO_TRANSCRIPTION_ENABLED", "true") or "true"),
            slack_audio_transcripts_enabled=parse_bool(get("SLACK_AUDIO_TRANSCRIPTS_ENABLED", "true") or "true"),
            local_whisper_enabled=parse_bool(get("LOCAL_WHISPER_ENABLED", "true") or "true"),
            local_whisper_model=(get("LOCAL_WHISPER_MODEL", "tiny") or "tiny").strip(),
            local_whisper_language=(get("LOCAL_WHISPER_LANGUAGE", "es") or "es").strip(),
            local_whisper_device=(get("LOCAL_WHISPER_DEVICE", "auto") or "auto").strip(),
            local_whisper_compute_type=(get("LOCAL_WHISPER_COMPUTE_TYPE", "auto") or "auto").strip(),
            local_whisper_max_seconds=local_whisper_max_seconds,
            local_whisper_keep_audio_files=parse_bool(get("LOCAL_WHISPER_KEEP_AUDIO_FILES", "false") or "false"),
            local_whisper_cache_dir=(
                get(
                    "LOCAL_WHISPER_CACHE_DIR",
                    "~/Library/Application Support/slack-personal-agent/audio",
                )
                or "~/Library/Application Support/slack-personal-agent/audio"
            ).strip(),
            audio_transcript_fusion_enabled=parse_bool(get("AUDIO_TRANSCRIPT_FUSION_ENABLED", "true") or "true"),
            audio_transcript_fusion_model=(get("AUDIO_TRANSCRIPT_FUSION_MODEL", "main") or "main").strip(),
            slack_image_attachments_enabled=parse_bool(get("SLACK_IMAGE_ATTACHMENTS_ENABLED", "true") or "true"),
            trello_attach_slack_images=parse_bool(get("TRELLO_ATTACH_SLACK_IMAGES", "true") or "true"),
            trello_image_attachment_mode=trello_image_attachment_mode,  # type: ignore[arg-type]
            slack_image_keep_files=parse_bool(get("SLACK_IMAGE_KEEP_FILES", "false") or "false"),
            slack_image_cache_dir=(
                get(
                    "SLACK_IMAGE_CACHE_DIR",
                    "~/Library/Application Support/slack-personal-agent/images",
                )
                or "~/Library/Application Support/slack-personal-agent/images"
            ).strip(),
            slack_image_max_bytes=slack_image_max_bytes,
            sync_worker_seconds=sync_worker_seconds,
            sync_waiting_enabled=parse_bool(get("SYNC_WAITING_ENABLED", "true") or "true"),
            sync_trello_done_enabled=parse_bool(get("SYNC_TRELLO_DONE_ENABLED", "true") or "true"),
            sync_telegram_poll_enabled=parse_bool(get("SYNC_TELEGRAM_POLL_ENABLED", "true") or "true"),
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


SALESFORCE_SYSTEM_TERMS = ("salesforce", "crm")
SALESFORCE_OPERATION_TERMS = (
    "informe",
    "reporte",
    "report",
    "dashboard",
    "tablero",
    "altas",
    "campana",
    "campana principal",
    "campana de origen",
    "armar",
    "generar",
    "exportar",
    "consultar",
    "revisar",
)

DONOR_DATA_OPERATION_TERMS = (
    "armar",
    "exportar",
    "actualizar",
    "generar",
    "preparar",
    "pasar",
    "enviar",
    "necesito",
    "me piden",
    "me armas",
    "me armás",
    "listado",
    "informe",
    "base",
    "stock",
)
DONOR_DATA_ENTITY_TERMS = (
    "base de donantes",
    "donantes actualizada",
    "donantes actualizado",
    "base actualizada",
    "stock activo",
    "donante",
    "donantes",
)
DONOR_DATA_FIELD_TERMS = (
    "datos de la persona",
    "datos personales",
    "nya",
    "nombre y apellido",
    "fecha de nacimiento",
    "edad",
    "lugar de residencia",
    "residencia",
    "datos de la donacion",
    "fecha establecida",
    "estado",
    "monto",
    "fecha de finalizacion",
    "campana",
)
NEW_REQUEST_STARTERS = (
    "y por otro lado",
    "por otro lado",
    "otra cosa",
    "ademas",
    "tambien te queria pedir",
    "tambien queria pedir",
    "nuevo pedido",
    "aparte",
)


def has_salesforce_url(message_links: list[MessageLink]) -> bool:
    return any(link.url_type == "salesforce" for link in message_links)


def has_salesforce_text_signal(text: str) -> bool:
    normalized = normalize_for_matching(text)
    if not normalized:
        return False
    has_system = any(term in normalized for term in SALESFORCE_SYSTEM_TERMS)
    has_operation = any(term in normalized for term in SALESFORCE_OPERATION_TERMS)
    return has_system and has_operation


def has_donor_data_request_signal(text: str) -> bool:
    normalized = normalize_for_matching(text)
    if not normalized:
        return False
    has_operation = any(term in normalized for term in DONOR_DATA_OPERATION_TERMS)
    has_core_entity = any(term in normalized for term in DONOR_DATA_ENTITY_TERMS)
    has_donation_dataset = "donacion" in normalized and any(
        term in normalized for term in ("base", "listado", "stock")
    )
    has_entity = has_core_entity or has_donation_dataset
    has_fields = any(term in normalized for term in DONOR_DATA_FIELD_TERMS)
    return has_operation and has_entity and has_fields


def starts_new_request(text: str) -> bool:
    normalized = normalize_for_matching(text)
    if not normalized:
        return False
    return any(normalized.startswith(starter) for starter in NEW_REQUEST_STARTERS)


def improve_donor_data_requested_action(action: str, text: str) -> str:
    normalized = normalize_for_matching(text)
    if not has_donor_data_request_signal(text):
        return action

    if "stock activo" in normalized or "base de donantes" in normalized:
        lead = "Armar un informe/base de donantes activos actualizado"
        if "amplify" in normalized:
            lead += " para Amplify"
        details = []
        if "mirta" in normalized or "buyer techo" in normalized:
            details.append("siguiendo la lógica del perfil de Mirta / análisis buyer techo")
        details.append("incluyendo datos personales y datos de donación solicitados")
        return lead + ", " + ", ".join(details) + "."

    cleaned = clean_text(action)
    if cleaned and "datos personales" not in normalize_for_matching(cleaned):
        return f"{cleaned.rstrip('.')}, incluyendo datos personales y datos de donación solicitados."
    return cleaned


def donor_data_missing_information(existing: list[str]) -> list[str]:
    additions = [
        "Confirmar fecha de corte del stock activo.",
        "Confirmar si la fuente oficial de la base de donantes es Salesforce.",
        "Confirmar si deben incluirse otros campos usados previamente en el perfil de Mirta.",
        "Confirmar formato de entrega: CSV, Excel, Google Sheets o dashboard.",
    ]
    seen = {normalize_for_matching(item) for item in existing}
    missing = list(existing)
    for item in additions:
        if normalize_for_matching(item) not in seen:
            missing.append(item)
    return missing


def improve_salesforce_requested_action(action: str, text: str) -> str:
    cleaned = clean_text(action)
    if not cleaned:
        return cleaned

    normalized_action = normalize_for_matching(cleaned)
    normalized_text = normalize_for_matching(text)
    if (
        "campana" in normalized_text
        and ("salesforce.com" in normalized_text or "force.com" in normalized_text)
        and "campanas indicadas" not in normalized_action
    ):
        cleaned = re.sub(r"\s+en salesforce\.?$", "", cleaned.rstrip("."), flags=re.IGNORECASE)
        cleaned = f"{cleaned.rstrip('.')} para las campañas indicadas en Salesforce."
    elif "salesforce" not in normalized_action and (
        "campana" in normalized_action
        or "campana" in normalized_text
        or "reporte" in normalized_action
        or "informe" in normalized_action
    ):
        cleaned = cleaned.rstrip(".")
        cleaned = f"{cleaned} en Salesforce."

    if (
        "datos personales" in normalized_text
        and "donacion" in normalized_text
        and "datos personales" not in normalize_for_matching(cleaned)
        and "donacion" not in normalize_for_matching(cleaned)
    ):
        cleaned = cleaned.rstrip(".")
        cleaned = f"{cleaned}, incluyendo datos personales y de donación."

    return cleaned


def normalize_classification_with_rules(
    classification: SlackClassification,
    *,
    text: str,
    message_links: list[MessageLink],
) -> SlackClassification:
    """Apply deterministic business rules that should outrank the local model."""
    has_donor_signal = has_donor_data_request_signal(text)
    if not has_salesforce_url(message_links) and not has_salesforce_text_signal(text) and not has_donor_signal:
        return classification

    external_systems = coerce_string_list(classification.external_systems)
    if "salesforce" not in {normalize_for_matching(system) for system in external_systems}:
        external_systems.append("salesforce")
    requested_action = (
        improve_donor_data_requested_action(classification.requested_action, text)
        if has_donor_signal
        else improve_salesforce_requested_action(classification.requested_action, text)
    )
    missing_information = (
        donor_data_missing_information(coerce_string_list(classification.missing_information))
        if has_donor_signal
        else classification.missing_information
    )

    return classification.model_copy(
        update={
            "category": "salesforce",
            "needs_external_system": True,
            "external_systems": external_systems,
            "requested_action": requested_action,
            "missing_information": missing_information,
        }
    )


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
        telegram_client: Optional[TelegramClientProtocol] = None,
        audio_transcriber: Optional[AudioTranscriber] = None,
        public_preview_fetcher: Optional[PublicPreviewFetcher] = None,
        input_fn: Callable[[str], str] = input,
        open_url_fn: Callable[[str], bool] = webbrowser.open,
        sleep_fn: Callable[[float], None] = time.sleep,
    ) -> None:
        self.config = config
        self.slack = slack_client or WebClient(token=config.slack_user_token)
        self._structured_model_factory = structured_model_factory or self._build_structured_model
        self._trello_client = trello_client
        self._telegram_client = telegram_client
        self._audio_transcriber = audio_transcriber
        self._public_preview_fetcher = public_preview_fetcher or self.fetch_public_url_preview
        self._input = input_fn
        self._open_url = open_url_fn
        self._classifier: Optional[StructuredModel] = None
        self._audio_fusion_model: Optional[StructuredModel] = None
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
                    has_audio_transcript INTEGER NOT NULL DEFAULT 0,
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

            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS audio_transcriptions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    source TEXT NOT NULL,
                    source_message_id TEXT NOT NULL,
                    channel_id TEXT NOT NULL,
                    message_ts TEXT NOT NULL,
                    task_id INTEGER,
                    file_id TEXT,
                    filename TEXT,
                    mime_type TEXT,
                    slack_transcript_text TEXT,
                    local_transcript_text TEXT,
                    fused_transcript_text TEXT,
                    selected_transcript_text TEXT,
                    transcription_status TEXT NOT NULL,
                    transcription_error TEXT,
                    created_at TEXT NOT NULL
                )
                """
            )

            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS trello_processed_actions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    task_id INTEGER NOT NULL,
                    trello_card_id TEXT NOT NULL,
                    trello_action_id TEXT NOT NULL,
                    action_type TEXT NOT NULL,
                    command_prefix TEXT NOT NULL,
                    command_text TEXT,
                    processed_at TEXT NOT NULL,
                    status TEXT NOT NULL,
                    error TEXT,
                    FOREIGN KEY(task_id) REFERENCES tasks(id)
                )
                """
            )

            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS slack_file_attachments (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    task_id INTEGER,
                    channel_id TEXT NOT NULL,
                    message_ts TEXT NOT NULL,
                    file_id TEXT NOT NULL,
                    filename TEXT,
                    mime_type TEXT,
                    filetype TEXT,
                    url_private TEXT,
                    local_path TEXT,
                    trello_card_id TEXT,
                    trello_attachment_id TEXT,
                    attachment_status TEXT NOT NULL,
                    attachment_error TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
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
            self._ensure_column(conn, "tasks", "updated_at", "updated_at TEXT")
            self._ensure_column(conn, "tasks", "case_key", "case_key TEXT")
            self._ensure_column(conn, "tasks", "requester_user_id", "requester_user_id TEXT")
            self._ensure_column(conn, "tasks", "requester_label", "requester_label TEXT")
            self._ensure_column(conn, "tasks", "thread_ts", "thread_ts TEXT")
            self._ensure_column(conn, "tasks", "acknowledged_at", "acknowledged_at TEXT")
            self._ensure_column(conn, "tasks", "last_context_ack_at", "last_context_ack_at TEXT")
            self._ensure_column(conn, "tasks", "done_pending_reply_at", "done_pending_reply_at TEXT")
            self._ensure_column(conn, "tasks", "final_reply_suggestion", "final_reply_suggestion TEXT")
            self._ensure_column(conn, "tasks", "ack_error", "ack_error TEXT")
            self._ensure_column(conn, "tasks", "context_ack_error", "context_ack_error TEXT")
            self._ensure_column(conn, "tasks", "telegram_notified_at", "telegram_notified_at TEXT")
            self._ensure_column(conn, "tasks", "telegram_error", "telegram_error TEXT")
            self._ensure_column(conn, "tasks", "public_request_text", "public_request_text TEXT")
            self._ensure_column(conn, "tasks", "has_audio_transcript", "has_audio_transcript INTEGER NOT NULL DEFAULT 0")
            self._ensure_column(conn, "tasks", "waiting_requested_at", "waiting_requested_at TEXT")
            self._ensure_column(conn, "tasks", "waiting_request_text", "waiting_request_text TEXT")
            self._ensure_column(conn, "tasks", "waiting_request_message_ts", "waiting_request_message_ts TEXT")
            self._ensure_column(conn, "tasks", "waiting_trello_action_id", "waiting_trello_action_id TEXT")
            self._ensure_column(conn, "tasks", "waiting_cleared_at", "waiting_cleared_at TEXT")
            self._ensure_column(conn, "tasks", "waiting_error", "waiting_error TEXT")

            conn.execute(
                """
                UPDATE tasks
                SET thread_ts = message_ts
                WHERE thread_ts IS NULL OR thread_ts = ''
                """
            )
            conn.execute(
                """
                UPDATE tasks
                SET case_key = channel_id || ':' || COALESCE(NULLIF(thread_ts, ''), message_ts)
                WHERE case_key IS NULL OR case_key = ''
                """
            )
            conn.execute(
                """
                UPDATE tasks
                SET requester_user_id = user_id
                WHERE requester_user_id IS NULL OR requester_user_id = ''
                """
            )
            conn.execute(
                """
                UPDATE tasks
                SET requester_label = sender_label
                WHERE requester_label IS NULL OR requester_label = ''
                """
            )
            conn.execute(
                """
                UPDATE tasks
                SET public_request_text = COALESCE(public_request_text, summary)
                WHERE public_request_text IS NULL OR public_request_text = ''
                """
            )
            conn.execute(
                """
                UPDATE tasks
                SET has_audio_transcript = COALESCE(has_audio_transcript, 0)
                WHERE has_audio_transcript IS NULL
                """
            )
            conn.execute(
                """
                UPDATE tasks
                SET updated_at = created_at
                WHERE updated_at IS NULL OR updated_at = ''
                """
            )

            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS agent_state (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                )
                """
            )

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
                CREATE INDEX IF NOT EXISTS idx_audio_transcriptions_message
                ON audio_transcriptions(channel_id, message_ts)
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_trello_processed_actions_action
                ON trello_processed_actions(trello_action_id, command_prefix, status)
                """
            )
            conn.execute(
                """
                CREATE UNIQUE INDEX IF NOT EXISTS idx_slack_file_attachments_message_file
                ON slack_file_attachments(channel_id, message_ts, file_id)
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_slack_file_attachments_task
                ON slack_file_attachments(task_id, trello_card_id, attachment_status)
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_task_events_task
                ON task_events(task_id, created_at)
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_tasks_case_key
                ON tasks(case_key)
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_tasks_trello_card_id
                ON tasks(trello_card_id)
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

    def record_trello_processed_action(
        self,
        *,
        task_id: int,
        trello_card_id: str,
        trello_action_id: str,
        action_type: str,
        command_prefix: str,
        command_text: str,
        status: str,
        error: str = "",
    ) -> None:
        with self.db_connect() as conn:
            conn.execute(
                """
                INSERT INTO trello_processed_actions (
                    task_id,
                    trello_card_id,
                    trello_action_id,
                    action_type,
                    command_prefix,
                    command_text,
                    processed_at,
                    status,
                    error
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    task_id,
                    trello_card_id,
                    trello_action_id,
                    action_type,
                    command_prefix,
                    command_text,
                    now_iso(),
                    status,
                    error[:1000],
                ),
            )

    def trello_action_already_processed(self, trello_action_id: str, command_prefix: str) -> bool:
        if not trello_action_id:
            return False
        with self.db_connect() as conn:
            row = conn.execute(
                """
                SELECT 1
                FROM trello_processed_actions
                WHERE trello_action_id = ?
                  AND command_prefix = ?
                  AND status = 'processed'
                LIMIT 1
                """,
                (trello_action_id, command_prefix),
            ).fetchone()
        return row is not None

    def get_agent_state(self, key: str) -> Optional[str]:
        with self.db_connect() as conn:
            row = conn.execute(
                "SELECT value FROM agent_state WHERE key = ?",
                (key,),
            ).fetchone()
        return row["value"] if row else None

    def set_agent_state(self, key: str, value: str) -> None:
        with self.db_connect() as conn:
            conn.execute(
                """
                INSERT INTO agent_state (key, value)
                VALUES (?, ?)
                ON CONFLICT(key) DO UPDATE SET value = excluded.value
                """,
                (key, value),
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

    def get_audio_fusion_model(self) -> StructuredModel:
        if self._audio_fusion_model is None:
            self._audio_fusion_model = self._structured_model_factory(AudioTranscriptFusion)
        return self._audio_fusion_model

    def get_audio_transcriber(self) -> AudioTranscriber:
        if self._audio_transcriber is None:
            self._audio_transcriber = LocalWhisperTranscriber(
                model_name=self.config.local_whisper_model,
                language=self.config.local_whisper_language,
                device=self.config.local_whisper_device,
                compute_type=self.config.local_whisper_compute_type,
            )
        return self._audio_transcriber

    def build_audio_fusion_prompt(self, slack_text: str, local_text: str) -> str:
        return f"""
Tenés dos transcripciones del mismo audio de trabajo.

Transcripción de Slack:
{slack_text}

Transcripción local de Whisper:
{local_text}

Reglas:
- No inventes contenido.
- Combiná ambas transcripciones en la mejor versión final.
- Corregí errores obvios solo si una versión apoya a la otra.
- Preservá nombres propios, IDs, URLs, montos, fechas y acciones.
- Si hay duda, mantené la formulación más conservadora.
- Devolvé solo la transcripción final, sin explicación.
""".strip()

    def build_best_audio_transcript(self, slack_text: str, local_text: str) -> str:
        prompt = self.build_audio_fusion_prompt(slack_text, local_text)
        response = self.get_audio_fusion_model().invoke(prompt)
        if isinstance(response, AudioTranscriptFusion):
            return compact_text(response.transcript, 8000)
        if isinstance(response, str):
            return response.strip()
        if isinstance(response, dict):
            return str(response.get("transcript") or "").strip()
        return str(getattr(response, "transcript", "") or "").strip()

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

    def has_telegram_config(self) -> bool:
        return bool(self.config.telegram_bot_token and self.config.telegram_chat_id)

    def get_telegram_client(self) -> TelegramClientProtocol:
        if self._telegram_client is None:
            if not self.has_telegram_config():
                raise TelegramError("Faltan TELEGRAM_BOT_TOKEN o TELEGRAM_CHAT_ID.")
            self._telegram_client = TelegramClient(
                bot_token=self.config.telegram_bot_token,
                chat_id=self.config.telegram_chat_id,
            )
        return self._telegram_client

    def audio_cache_dir(self) -> Path:
        return Path(self.config.local_whisper_cache_dir).expanduser()

    def download_audio_attachment(self, attachment: AudioAttachment) -> Path:
        if not attachment.url_private:
            raise RuntimeError("El audio no trae url_private ni url_private_download.")

        cache_dir = self.audio_cache_dir()
        cache_dir.mkdir(parents=True, exist_ok=True)
        suffix = Path(attachment.filename).suffix or ".audio"
        target = cache_dir / f"{attachment.file_id or attachment.index}{suffix}"
        response = requests.get(
            attachment.url_private,
            timeout=(5, 60),
            headers={"Authorization": f"Bearer {self.config.slack_user_token}"},
        )
        if response.status_code >= 400:
            raise RuntimeError(f"Slack audio devolvió HTTP {response.status_code}: {response.text[:300]}")
        target.write_bytes(response.content)
        return target

    def save_audio_transcription_result(
        self,
        result: AudioTranscriptResult,
        *,
        task_id: Optional[int] = None,
    ) -> None:
        attachment = result.attachment
        with self.db_connect() as conn:
            conn.execute(
                """
                INSERT INTO audio_transcriptions (
                    source,
                    source_message_id,
                    channel_id,
                    message_ts,
                    task_id,
                    file_id,
                    filename,
                    mime_type,
                    slack_transcript_text,
                    local_transcript_text,
                    fused_transcript_text,
                    selected_transcript_text,
                    transcription_status,
                    transcription_error,
                    created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    attachment.source,
                    attachment.source_message_id,
                    attachment.channel_id,
                    attachment.message_ts,
                    task_id,
                    attachment.file_id,
                    attachment.filename,
                    attachment.mime_type,
                    result.slack_transcript_text,
                    result.local_transcript_text,
                    result.fused_transcript_text,
                    result.selected_transcript_text,
                    result.transcription_status,
                    result.transcription_error[:1000] if result.transcription_error else "",
                    now_iso(),
                ),
            )

    def transcribe_and_fuse_audio(
        self,
        attachment: AudioAttachment,
        *,
        task_id: Optional[int] = None,
    ) -> AudioTranscriptResult:
        slack_text = attachment.slack_transcript_text if self.config.slack_audio_transcripts_enabled else ""
        local_text = ""
        fused_text = ""
        error = ""
        downloaded_path: Optional[Path] = None

        if self.config.local_whisper_enabled:
            try:
                if (
                    attachment.duration_seconds is not None
                    and attachment.duration_seconds > self.config.local_whisper_max_seconds
                ):
                    raise RuntimeError(
                        f"El audio dura {attachment.duration_seconds:.0f}s y supera LOCAL_WHISPER_MAX_SECONDS."
                    )
                downloaded_path = self.download_audio_attachment(attachment)
                local_text = self.get_audio_transcriber().transcribe(downloaded_path)
            except Exception as exc:
                error = str(exc)
            finally:
                if (
                    downloaded_path
                    and downloaded_path.exists()
                    and not self.config.local_whisper_keep_audio_files
                ):
                    downloaded_path.unlink(missing_ok=True)

        if slack_text and local_text and self.config.audio_transcript_fusion_enabled:
            try:
                fused_text = self.build_best_audio_transcript(slack_text, local_text)
            except Exception as exc:
                error = str(exc)

        selected_text, status = choose_audio_transcript(
            slack_text=slack_text,
            local_text=local_text,
            fused_text=fused_text,
        )
        if error and not selected_text:
            status = "failed"
        elif not selected_text:
            status = "missing"

        result = AudioTranscriptResult(
            attachment=attachment,
            slack_transcript_text=slack_text,
            local_transcript_text=local_text,
            fused_transcript_text=fused_text,
            selected_transcript_text=selected_text,
            transcription_status=status,
            transcription_error=error,
        )
        self.save_audio_transcription_result(result, task_id=task_id)
        return result

    def enrich_message_with_audio(
        self,
        message: dict[str, Any],
        conversation: dict[str, Any],
        *,
        task_id: Optional[int] = None,
    ) -> dict[str, Any]:
        if not self.config.audio_transcription_enabled:
            return message

        enriched_message = dict(message)
        enriched_message.setdefault("channel", conversation.get("id", ""))
        attachments = detect_audio_attachments(enriched_message)
        if not attachments:
            return enriched_message

        results = [
            self.transcribe_and_fuse_audio(attachment, task_id=task_id)
            for attachment in attachments
        ]
        audio_text = format_audio_transcripts_for_message(results)
        if not audio_text:
            enriched_message["_audio_transcription_status"] = "missing"
            return enriched_message

        enriched_message["text"] = combine_text_and_audio(enriched_message.get("text", ""), audio_text)
        enriched_message["_audio_transcript_added"] = True
        return enriched_message

    def transcribe_audio_path(self, audio_path: Path) -> str:
        return self.get_audio_transcriber().transcribe(audio_path)

    def transcribe_audio_folder(self, folder_path: Path) -> list[tuple[Path, str]]:
        results = []
        for path in sorted(item for item in folder_path.iterdir() if item.is_file()):
            results.append((path, self.transcribe_audio_path(path)))
        return results

    def image_cache_dir(self) -> Path:
        return Path(self.config.slack_image_cache_dir).expanduser()

    def is_visual_file(self, file_payload: dict[str, Any]) -> bool:
        mimetype = str(file_payload.get("mimetype") or "").lower()
        if mimetype.startswith("image/"):
            return True
        filetype = str(file_payload.get("filetype") or "").lower()
        if filetype in VISUAL_IMAGE_SUFFIXES:
            return True
        name = str(file_payload.get("name") or file_payload.get("title") or "").lower()
        suffix = Path(name).suffix.lstrip(".")
        return suffix in VISUAL_IMAGE_SUFFIXES

    def detect_visual_attachments(self, message: dict[str, Any]) -> list[SlackVisualAttachment]:
        channel_id = str(message.get("channel") or message.get("channel_id") or "")
        message_ts = str(message.get("ts") or "")
        attachments: list[SlackVisualAttachment] = []
        for index, file_payload in enumerate(message.get("files") or [], start=1):
            if not isinstance(file_payload, dict) or not self.is_visual_file(file_payload):
                continue
            file_id = str(file_payload.get("id") or file_payload.get("file_id") or f"image-{index}")
            filename = str(file_payload.get("name") or file_payload.get("title") or file_id)
            size_bytes = int(file_payload.get("size") or 0)
            attachments.append(
                SlackVisualAttachment(
                    source_message_id=f"{channel_id}:{message_ts}",
                    channel_id=channel_id,
                    message_ts=message_ts,
                    file_id=file_id,
                    filename=filename,
                    mime_type=str(file_payload.get("mimetype") or ""),
                    filetype=str(file_payload.get("filetype") or ""),
                    url_private=str(
                        file_payload.get("url_private_download")
                        or file_payload.get("url_private")
                        or ""
                    ),
                    index=index,
                    size_bytes=size_bytes,
                )
            )
        return attachments

    def save_visual_attachment_metadata(
        self,
        attachment: SlackVisualAttachment,
        *,
        task_id: Optional[int] = None,
    ) -> None:
        timestamp = now_iso()
        with self.db_connect() as conn:
            conn.execute(
                """
                INSERT INTO slack_file_attachments (
                    task_id,
                    channel_id,
                    message_ts,
                    file_id,
                    filename,
                    mime_type,
                    filetype,
                    url_private,
                    local_path,
                    trello_card_id,
                    trello_attachment_id,
                    attachment_status,
                    attachment_error,
                    created_at,
                    updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, NULL, NULL, NULL, 'detected', NULL, ?, ?)
                ON CONFLICT(channel_id, message_ts, file_id) DO UPDATE SET
                    task_id = COALESCE(excluded.task_id, slack_file_attachments.task_id),
                    filename = excluded.filename,
                    mime_type = excluded.mime_type,
                    filetype = excluded.filetype,
                    url_private = excluded.url_private,
                    attachment_status = CASE
                        WHEN slack_file_attachments.attachment_status IN ('attached', 'linked')
                            THEN slack_file_attachments.attachment_status
                        ELSE 'detected'
                    END,
                    updated_at = excluded.updated_at
                """,
                (
                    task_id,
                    attachment.channel_id,
                    attachment.message_ts,
                    attachment.file_id,
                    attachment.filename,
                    attachment.mime_type,
                    attachment.filetype,
                    attachment.url_private,
                    timestamp,
                    timestamp,
                ),
            )

    def annotate_message_with_visual_attachments(
        self,
        message: dict[str, Any],
        conversation: dict[str, Any],
        *,
        task_id: Optional[int] = None,
    ) -> dict[str, Any]:
        if not self.config.slack_image_attachments_enabled:
            return message

        enriched_message = dict(message)
        enriched_message.setdefault("channel", conversation.get("id", ""))
        attachments = self.detect_visual_attachments(enriched_message)
        if not attachments:
            return enriched_message

        for attachment in attachments:
            self.save_visual_attachment_metadata(attachment, task_id=task_id)
        enriched_message["_visual_attachments"] = attachments
        return enriched_message

    def assign_visual_attachments_to_task(self, channel_id: str, message_ts: str, task_id: int) -> None:
        with self.db_connect() as conn:
            conn.execute(
                """
                UPDATE slack_file_attachments
                SET task_id = ?,
                    updated_at = ?
                WHERE channel_id = ?
                  AND message_ts = ?
                  AND task_id IS NULL
                """,
                (task_id, now_iso(), channel_id, message_ts),
            )

    def update_visual_attachment_row(
        self,
        *,
        channel_id: str,
        message_ts: str,
        file_id: str,
        status: str,
        local_path: str = "",
        trello_card_id: str = "",
        trello_attachment_id: str = "",
        error: str = "",
    ) -> None:
        with self.db_connect() as conn:
            conn.execute(
                """
                UPDATE slack_file_attachments
                SET local_path = CASE WHEN ? != '' THEN ? ELSE local_path END,
                    trello_card_id = CASE WHEN ? != '' THEN ? ELSE trello_card_id END,
                    trello_attachment_id = CASE WHEN ? != '' THEN ? ELSE trello_attachment_id END,
                    attachment_status = ?,
                    attachment_error = ?,
                    updated_at = ?
                WHERE channel_id = ?
                  AND message_ts = ?
                  AND file_id = ?
                """,
                (
                    local_path,
                    local_path,
                    trello_card_id,
                    trello_card_id,
                    trello_attachment_id,
                    trello_attachment_id,
                    status,
                    error[:1000] if error else None,
                    now_iso(),
                    channel_id,
                    message_ts,
                    file_id,
                ),
            )

    def download_visual_attachment(self, attachment: SlackVisualAttachment) -> Path:
        if not attachment.url_private:
            raise RuntimeError("La imagen no trae url_private ni url_private_download.")
        if attachment.size_bytes and attachment.size_bytes > self.config.slack_image_max_bytes:
            raise RuntimeError(
                f"La imagen supera SLACK_IMAGE_MAX_BYTES ({attachment.size_bytes} bytes)."
            )

        cache_dir = self.image_cache_dir()
        cache_dir.mkdir(parents=True, exist_ok=True)
        suffix = Path(attachment.filename).suffix or ".image"
        target = cache_dir / f"{attachment.file_id or attachment.index}{suffix}"
        response = requests.get(
            attachment.url_private,
            timeout=(5, 60),
            headers={"Authorization": f"Bearer {self.config.slack_user_token}"},
        )
        if response.status_code >= 400:
            raise RuntimeError(f"Slack imagen devolvió HTTP {response.status_code}: {response.text[:300]}")
        if len(response.content) > self.config.slack_image_max_bytes:
            raise RuntimeError(
                f"La imagen descargada supera SLACK_IMAGE_MAX_BYTES ({len(response.content)} bytes)."
            )
        target.write_bytes(response.content)
        self.update_visual_attachment_row(
            channel_id=attachment.channel_id,
            message_ts=attachment.message_ts,
            file_id=attachment.file_id,
            status="downloaded",
            local_path=str(target),
        )
        return target

    def build_visual_attachment_link_comment(self, attachment: SlackVisualAttachment) -> str:
        return "\n".join(
            [
                f"Imagen recibida desde Slack: {attachment.filename}",
                f"Slack file id: {attachment.file_id}",
                f"URL privada: {attachment.url_private}",
                "Puede requerir permisos de Slack para abrirla.",
            ]
        )

    def maybe_comment_trello_card(self, card_id: str, text: str) -> None:
        if not card_id or not text.strip():
            return
        try:
            self.get_trello_client().add_card_comment(card_id, text)
        except Exception as exc:
            print(f"[yellow]No pude comentar en Trello {card_id}: {exc}[/yellow]")

    def sync_visual_attachments_for_task(self, task_id: int, card_id: str) -> int:
        if (
            not self.config.trello_enabled
            or not self.config.trello_attach_slack_images
            or not card_id
        ):
            return 0

        with self.db_connect() as conn:
            rows = conn.execute(
                """
                SELECT *
                FROM slack_file_attachments
                WHERE task_id = ?
                ORDER BY id ASC
                """,
                (task_id,),
            ).fetchall()

        attached = 0
        for row in rows:
            if row["trello_card_id"] == card_id and row["attachment_status"] in {"attached", "linked"}:
                continue

            attachment = SlackVisualAttachment(
                source_message_id=f"{row['channel_id']}:{row['message_ts']}",
                channel_id=row["channel_id"],
                message_ts=row["message_ts"],
                file_id=row["file_id"],
                filename=row["filename"] or row["file_id"],
                mime_type=row["mime_type"] or "",
                filetype=row["filetype"] or "",
                url_private=row["url_private"] or "",
            )
            try:
                if self.config.trello_image_attachment_mode == "link":
                    self.maybe_comment_trello_card(card_id, self.build_visual_attachment_link_comment(attachment))
                    self.update_visual_attachment_row(
                        channel_id=attachment.channel_id,
                        message_ts=attachment.message_ts,
                        file_id=attachment.file_id,
                        status="linked",
                        trello_card_id=card_id,
                    )
                else:
                    downloaded_path = self.download_visual_attachment(attachment)
                    try:
                        trello_attachment_id = self.get_trello_client().attach_file_to_card(
                            card_id,
                            downloaded_path,
                            name=attachment.filename,
                        )
                        self.update_visual_attachment_row(
                            channel_id=attachment.channel_id,
                            message_ts=attachment.message_ts,
                            file_id=attachment.file_id,
                            status="attached",
                            local_path=str(downloaded_path) if self.config.slack_image_keep_files else "",
                            trello_card_id=card_id,
                            trello_attachment_id=trello_attachment_id,
                        )
                        self.maybe_comment_trello_card(card_id, f"Imagen adjuntada desde Slack: {attachment.filename}")
                    finally:
                        if downloaded_path.exists() and not self.config.slack_image_keep_files:
                            downloaded_path.unlink(missing_ok=True)
                attached += 1
            except Exception as exc:
                error_message = str(exc)
                self.update_visual_attachment_row(
                    channel_id=attachment.channel_id,
                    message_ts=attachment.message_ts,
                    file_id=attachment.file_id,
                    status="failed",
                    trello_card_id=card_id,
                    error=error_message,
                )
                self.maybe_comment_trello_card(
                    card_id,
                    f"No pude adjuntar la imagen desde Slack: {attachment.filename}. {error_message[:220]}",
                )
                print(
                    f"[yellow]No pude adjuntar imagen Slack para tarea #{task_id} ({attachment.filename}): "
                    f"{error_message}[/yellow]"
                )
        return attached

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
        thread_ts: str,
        case_key: str,
        user_id: Optional[str],
        sender_label: str,
        conversation_label: str,
        classification: SlackClassification,
        raw_text: str = "",
        has_audio_transcript: bool = False,
    ) -> bool:
        timestamp = now_iso()
        with self.db_connect() as conn:
            cursor = conn.execute(
                """
                INSERT OR IGNORE INTO tasks (
                    created_at,
                    updated_at,
                    channel_id,
                    message_ts,
                    thread_ts,
                    case_key,
                    user_id,
                    requester_user_id,
                    sender_label,
                    requester_label,
                    conversation_label,
                    summary,
                    public_request_text,
                    requested_action,
                    has_audio_transcript,
                    priority,
                    category,
                    status,
                    trello_status,
                    classification_json
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    timestamp,
                    timestamp,
                    channel_id,
                    message_ts,
                    thread_ts,
                    case_key,
                    user_id,
                    user_id,
                    sender_label,
                    sender_label,
                    conversation_label,
                    classification.summary,
                    self.build_public_request_text(
                        task_row={
                            "classification_json": json.dumps(classification.model_dump(), ensure_ascii=False),
                            "sender_label": sender_label,
                            "requester_label": sender_label,
                            "raw_text": raw_text or classification.requested_action,
                            "context_text": "",
                            "requested_action": classification.requested_action,
                        },
                        transcribed_audio=False,
                        context_text="",
                    ),
                    classification.requested_action,
                    int(has_audio_transcript),
                    classification.priority,
                    classification.category,
                    "new",
                    "pending",
                    json.dumps(classification.model_dump(), ensure_ascii=False),
                ),
            )
        return cursor.rowcount > 0

    def mark_processed_context_status(
        self,
        *,
        channel_id: str,
        message_ts: str,
        status: str,
        reason: str,
    ) -> None:
        with self.db_connect() as conn:
            conn.execute(
                """
                UPDATE processed_messages
                SET updated_at = ?,
                    classification_status = ?,
                    relevance_reason = ?,
                    classification_error = NULL
                WHERE channel_id = ? AND message_ts = ?
                """,
                (
                    now_iso(),
                    status,
                    reason,
                    channel_id,
                    message_ts,
                ),
            )

    def normalized_context_fingerprint(self, text: str) -> str:
        normalized = normalize_for_matching(text)
        normalized = re.sub(r"\s+", " ", normalized)
        return normalized.strip()

    def existing_context_fingerprints(self, task_id: int) -> set[str]:
        with self.db_connect() as conn:
            rows = conn.execute(
                """
                SELECT details_json
                FROM task_events
                WHERE task_id = ?
                  AND event_type = 'context_added'
                """,
                (task_id,),
            ).fetchall()

        fingerprints: set[str] = set()
        for row in rows:
            try:
                details = json.loads(row["details_json"] or "{}")
            except json.JSONDecodeError:
                details = {}
            fingerprint = self.normalized_context_fingerprint(details.get("text") or "")
            if fingerprint:
                fingerprints.add(fingerprint)
        return fingerprints

    def message_adds_new_context(self, task_row: sqlite3.Row, text: str) -> bool:
        fingerprint = self.normalized_context_fingerprint(text)
        if not fingerprint:
            return False

        original = self.normalized_context_fingerprint(task_row["raw_text"] or "")
        if fingerprint == original:
            return False

        if fingerprint in self.existing_context_fingerprints(task_row["id"]):
            return False

        return True

    def _strip_public_request_prefix(self, text: str, *, sender_label: str = "", requester_label: str = "") -> str:
        value = clean_text(text)
        if not value:
            return ""

        for prefix in (
            "Petición registrada:",
            "Petición actualizada:",
            "Petición resuelta:",
            "*Pedido registrado:*",
            "*Pedido actualizado:*",
            "*Pedido resuelto:*",
            "Texto original:",
            "Audios transcriptos:",
        ):
            if value.startswith(prefix):
                value = value[len(prefix):].strip()

        value = re.sub(r"^(?:[A-ZÁÉÍÓÚÑ][^:\n]{0,80}?\s+)?(?:pide|solicita|requiere|necesita|quiere|consulta|pregunta|pide a Ivan Rodríguez que)\s+", "", value, flags=re.IGNORECASE)
        value = re.sub(r"^(?:a Ivan Rodríguez que|a Ivan que|que)\s+", "", value, flags=re.IGNORECASE)

        if sender_label:
            value = re.sub(rf"^{re.escape(sender_label)}\s*[:,-]?\s*", "", value, flags=re.IGNORECASE)
        if requester_label and requester_label != sender_label:
            value = re.sub(rf"^{re.escape(requester_label)}\s*[:,-]?\s*", "", value, flags=re.IGNORECASE)

        return clean_text(value)

    def classification_from_task_row(self, task_row: Any) -> dict[str, Any]:
        try:
            return json.loads(row_value(task_row, "classification_json", "") or "{}")
        except json.JSONDecodeError:
            return {}

    def classification_external_systems(self, classification: dict[str, Any]) -> list[str]:
        systems = coerce_string_list(classification.get("external_systems"))
        category = normalize_for_matching(classification.get("category") or "")
        if category == "salesforce" and "salesforce" not in {normalize_for_matching(system) for system in systems}:
            systems.append("salesforce")
        deduped: list[str] = []
        seen: set[str] = set()
        for system in systems:
            normalized = normalize_for_matching(system)
            if normalized and normalized not in seen:
                seen.add(normalized)
                deduped.append(normalized)
        return deduped

    def format_external_systems_for_slack(self, classification: dict[str, Any]) -> str:
        labels = {
            "salesforce": "Salesforce",
            "trello": "Trello",
            "github": "GitHub",
            "google_sheet": "Google Sheets",
            "google_sheets": "Google Sheets",
            "google_drive": "Google Drive",
            "google_doc": "Google Docs",
            "google_docs": "Google Docs",
        }
        systems = self.classification_external_systems(classification)
        return ", ".join(labels.get(system, system.replace("_", " ").title()) for system in systems)

    def is_salesforce_classification(self, classification: dict[str, Any]) -> bool:
        return (
            normalize_for_matching(classification.get("category") or "") == "salesforce"
            or "salesforce" in self.classification_external_systems(classification)
        )

    def strip_slack_link_markup(self, text: str) -> str:
        return re.sub(
            r"<(https?://[^>|]+)(?:\|([^>]+))?>",
            lambda match: clean_text(match.group(2) or match.group(1)),
            text,
        )

    def extract_salesforce_campaign_sections(self, text: str) -> list[tuple[str, list[str]]]:
        sections: list[tuple[str, list[str]]] = []
        current_index: Optional[int] = None
        link_pattern = re.compile(r"<(https?://[^>|]+)(?:\|([^>]+))?>")
        ignored_labels = {"connect your salesforce account"}

        for line in str(text or "").splitlines():
            bullet_match = re.match(r"^\s*([•◦\-*])\s+(.+?)\s*$", line)
            bullet = bullet_match.group(1) if bullet_match else ""
            content = bullet_match.group(2) if bullet_match else line.strip()
            matches = list(link_pattern.finditer(content))
            salesforce_matches = [
                match
                for match in matches
                if classify_url(match.group(1))[0] == "salesforce"
            ]
            if salesforce_matches and (not bullet or bullet == "•" or current_index is None):
                for match in salesforce_matches:
                    label = clean_text(match.group(2) or match.group(1))
                    if normalize_for_matching(label) in ignored_labels:
                        label = clean_text(match.group(1))
                    sections.append((label, []))
                    current_index = len(sections) - 1
                continue

            if bullet_match and current_index is not None:
                label, items = sections[current_index]
                item = self.strip_slack_link_markup(content).strip()
                if item:
                    items.append(item)
                    sections[current_index] = (label, items)
                continue

            if line.strip():
                current_index = None

        deduped: list[tuple[str, list[str]]] = []
        seen: set[str] = set()
        for label, items in sections:
            key = normalize_for_matching(label)
            if not key or key in seen:
                continue
            seen.add(key)
            deduped.append((label, items))
        return deduped

    def salesforce_required_fields_text(self, source_text: str) -> str:
        normalized = normalize_for_matching(source_text)
        groups = [
            (
                "Persona",
                [
                    ("nombre y apellido", ("nombre y apellido", "nombre completo", "nya")),
                    ("fecha de nacimiento o edad", ("fecha de nacimiento", "edad")),
                    ("lugar de residencia", ("lugar de residencia", "residencia")),
                    ("email", ("email", "correo")),
                    ("telefono", ("telefono", "celular")),
                    ("DNI", ("dni", "documento")),
                ],
                "datos personales solicitados",
                ("datos personales", "datos de persona", "datos de la persona"),
            ),
            (
                "Donación",
                [
                    ("fecha establecida", ("fecha establecida",)),
                    ("estado", ("estado",)),
                    ("monto", ("monto", "importe")),
                    ("fecha de finalización", ("fecha de finalizacion", "fecha de finalización")),
                    ("campaña", ("campana", "campaña")),
                ],
                "datos de donación solicitados",
                ("datos de donacion", "datos de donación", "donacion", "donación"),
            ),
        ]

        lines = []
        for group_name, candidates, fallback, fallback_terms in groups:
            fields = [
                label
                for label, terms in candidates
                if any(normalize_for_matching(term) in normalized for term in terms)
            ]
            if fields:
                lines.append(f"- {group_name}: {', '.join(fields)}.")
            elif any(normalize_for_matching(term) in normalized for term in fallback_terms):
                lines.append(f"- {group_name}: {fallback}.")
        return "\n".join(lines)

    def build_donor_stock_public_request_text(self, *, raw_text: str, base_text: str) -> str:
        source_text = "\n".join(piece for piece in (raw_text, base_text) if piece)
        normalized = normalize_for_matching(source_text)
        if not has_donor_data_request_signal(source_text):
            return ""

        lead = "Armar un informe/base de donantes activos actualizado"
        if "amplify" in normalized:
            lead += " para Amplify"
        if "mirta" in normalized or "buyer techo" in normalized:
            lead += ", siguiendo la lógica del perfil de Mirta / análisis buyer techo"
        lead += "."

        objectives = ["Entregar una base actualizada de donantes activos."]
        if "amplify" in normalized:
            objectives.append("Enfocar el informe en el canal Amplify.")
        if "mirta" in normalized or "buyer techo" in normalized or "perfil" in normalized:
            objectives.append("Usar una lógica similar a la utilizada para construir perfiles de donantes anteriores.")

        blocks = [
            lead,
            "\n".join(["Objetivo:", *[f"- {objective}" for objective in objectives]]),
        ]

        fields_text = self.salesforce_required_fields_text(source_text)
        if fields_text:
            blocks.append("\n".join(["Campos requeridos:", fields_text]))

        confirmation_items = [
            "Fecha de corte del stock activo.",
            "Fuente oficial de datos, probablemente Salesforce/CRM.",
            "Si deben incluirse otros campos que se usaron previamente para el perfil de Mirta.",
            "Formato de entrega esperado.",
        ]
        blocks.append(
            "\n".join(
                [
                    "Información a confirmar:",
                    *[f"- {item}" for item in confirmation_items],
                ]
            )
        )
        return "\n\n".join(blocks).strip()

    def build_salesforce_public_request_text(
        self,
        *,
        classification: dict[str, Any],
        raw_text: str,
        base_text: str,
        context_text: str,
    ) -> str:
        source_text = "\n".join(piece for piece in (raw_text, base_text, context_text) if piece)
        normalized_source = normalize_for_matching(source_text)
        if not any(term in normalized_source for term in ("informe", "reporte", "report", "dashboard", "tablero", "altas", "campana")):
            return ""

        campaign_sections = self.extract_salesforce_campaign_sections(raw_text)
        request_text = self.strip_slack_link_markup(base_text).strip()
        request_text = re.sub(r"\s+", " ", request_text).strip()
        if not request_text:
            request_text = "Armar el informe solicitado en Salesforce."
        normalized_request = normalize_for_matching(request_text)
        if campaign_sections and "campanas indicadas" not in normalized_request and "campañas indicadas" not in request_text.lower():
            request_text = re.sub(r"\s+en salesforce\.?$", "", request_text.rstrip("."), flags=re.IGNORECASE)
            request_text = f"{request_text.rstrip('.')} para las campañas indicadas en Salesforce."
        elif "salesforce" not in normalized_request:
            request_text = f"{request_text.rstrip('.')} en Salesforce."
        else:
            request_text = f"{request_text.rstrip('.')}."

        blocks = [request_text]
        if campaign_sections:
            campaign_lines = ["Campañas/fuentes solicitadas:"]
            for label, items in campaign_sections:
                campaign_lines.append(f"- {label}")
                campaign_lines.extend(f"  - {item}" for item in items)
            blocks.append("\n".join(campaign_lines))

        fields_text = self.salesforce_required_fields_text(source_text)
        if fields_text:
            blocks.append("\n".join(["Campos requeridos:", fields_text]))

        return "\n\n".join(blocks).strip()

    def build_public_request_text(
        self,
        *,
        task_row: sqlite3.Row,
        transcribed_audio: bool = False,
        context_text: str = "",
    ) -> str:
        classification = self.classification_from_task_row(task_row)

        sender_label = row_value(task_row, "sender_label", "") or ""
        requester_label = row_value(task_row, "requester_label", "") or sender_label
        raw_text = row_value(task_row, "raw_text", "") or ""
        context_source = context_text or row_value(task_row, "context_text", "") or ""
        if starts_new_request(raw_text):
            context_source = ""
        requested_action = clean_text(classification.get("requested_action") or row_value(task_row, "requested_action", "") or "")
        base_text = requested_action or raw_text
        base_text = self._strip_public_request_prefix(
            base_text,
            sender_label=sender_label,
            requester_label=requester_label,
        )
        if not base_text:
            base_text = self._strip_public_request_prefix(
                raw_text,
                sender_label=sender_label,
                requester_label=requester_label,
            )

        if self.is_salesforce_classification(classification):
            donor_stock_text = self.build_donor_stock_public_request_text(
                raw_text=raw_text,
                base_text=base_text,
            )
            if donor_stock_text:
                return donor_stock_text
            salesforce_text = self.build_salesforce_public_request_text(
                classification=classification,
                raw_text=raw_text,
                base_text=base_text,
                context_text=context_source,
            )
            if salesforce_text:
                return salesforce_text

        pieces = [piece for piece in (base_text, context_source.strip()) if piece]
        public_text = "\n".join(pieces).strip()
        if not public_text:
            public_text = "Pedido recibido."

        return public_text

    def mark_task_acknowledged(self, task_id: int) -> None:
        with self.db_connect() as conn:
            conn.execute(
                """
                UPDATE tasks
                SET acknowledged_at = ?,
                    ack_error = NULL,
                    updated_at = ?
                WHERE id = ?
                """,
                (now_iso(), now_iso(), task_id),
            )
        self.record_task_event(task_id, "slack_ack_sent", {})

    def mark_task_ack_failed(self, task_id: int, error_message: str) -> None:
        with self.db_connect() as conn:
            conn.execute(
                """
                UPDATE tasks
                SET ack_error = ?,
                    updated_at = ?
                WHERE id = ?
                """,
                (error_message[:1000], now_iso(), task_id),
            )
        self.record_task_event(task_id, "slack_ack_failed", {"error": error_message[:1000]})

    def mark_task_context_acknowledged(self, task_id: int) -> None:
        with self.db_connect() as conn:
            conn.execute(
                """
                UPDATE tasks
                SET last_context_ack_at = ?,
                    context_ack_error = NULL,
                    updated_at = ?
                WHERE id = ?
                """,
                (now_iso(), now_iso(), task_id),
            )
        self.record_task_event(task_id, "slack_context_ack_sent", {})

    def mark_task_context_ack_failed(self, task_id: int, error_message: str) -> None:
        with self.db_connect() as conn:
            conn.execute(
                """
                UPDATE tasks
                SET context_ack_error = ?,
                    updated_at = ?
                WHERE id = ?
                """,
                (error_message[:1000], now_iso(), task_id),
            )
        self.record_task_event(task_id, "slack_context_ack_failed", {"error": error_message[:1000]})

    def send_task_acknowledgement(self, task_id: int) -> bool:
        task_row = self.get_task_by_id(task_id)
        if not task_row:
            return False

        mention = ""
        requester_user_id = task_row["requester_user_id"] or task_row["user_id"] or ""
        if self.requester_needs_mention(task_row) and requester_user_id:
            mention = f"<@{requester_user_id}> "
        request_text = task_row["public_request_text"] or self.build_public_request_text(task_row=task_row)
        audio_prefix = (
            "Transcribí el audio y lo dejé registrado para revisarlo."
            if int(task_row["has_audio_transcript"] or 0)
            else "Lo dejé registrado para revisarlo."
        )
        classification = self.classification_from_task_row(task_row)
        systems_text = self.format_external_systems_for_slack(classification)
        missing_information = coerce_string_list(classification.get("missing_information"))
        blocks = [
            f"{mention}Dale, lo tomo. {audio_prefix}",
            f"*Pedido registrado:*\n{request_text}",
        ]
        if systems_text:
            blocks.append(f"*Sistema:* {systems_text}")
        if missing_information:
            blocks.append("*Falta información:*\n" + "\n".join(f"- {item}" for item in missing_information))
        blocks.append("*Estado:* Pendiente de revisión")
        text = "\n\n".join(blocks)
        try:
            self.slack_call(
                self.slack.chat_postMessage,
                channel=task_row["channel_id"],
                thread_ts=task_row["thread_ts"] or task_row["message_ts"],
                text=text,
                unfurl_links=False,
                unfurl_media=False,
            )
        except Exception as exc:
            self.mark_task_ack_failed(task_id, str(exc))
            print(f"[yellow]No pude enviar acuse de recibo Slack para tarea #{task_id}: {exc}[/yellow]")
            return False

        self.mark_task_acknowledged(task_id)
        return True

    def send_context_acknowledgement(self, task_id: int, new_context_text: str, *, transcribed_audio: bool = False) -> bool:
        task_row = self.get_task_by_id(task_id)
        if not task_row:
            return False

        request_text = task_row["public_request_text"] or self.build_public_request_text(
            task_row=task_row,
            transcribed_audio=transcribed_audio,
            context_text=new_context_text,
        )
        prefix = "Buenísimo, transcribí el audio y lo sumo al pedido." if transcribed_audio else "Buenísimo, gracias. Lo sumo al pedido."
        blocks = [
            prefix,
            f"*Pedido actualizado:*\n{request_text}",
        ]
        if new_context_text.strip():
            blocks.append(f"*Nuevo contexto agregado:*\n{new_context_text.strip()}")
        text = "\n\n".join(blocks)
        try:
            self.slack_call(
                self.slack.chat_postMessage,
                channel=task_row["channel_id"],
                thread_ts=task_row["thread_ts"] or task_row["message_ts"],
                text=text,
                unfurl_links=False,
                unfurl_media=False,
            )
        except Exception as exc:
            self.mark_task_context_ack_failed(task_id, str(exc))
            print(f"[yellow]No pude confirmar contexto Slack para tarea #{task_id}: {exc}[/yellow]")
            return False

        self.mark_task_context_acknowledged(task_id)
        return True

    def waiting_request_from_comment(self, comment_text: str) -> str:
        prefix = self.config.trello_waiting_comment_prefix.strip()
        text = str(comment_text or "").strip()
        if not prefix or not text.lower().startswith(prefix.lower()):
            return ""
        return text[len(prefix):].strip()

    def reply_command_from_comment(self, comment_text: str) -> str:
        prefix = self.config.trello_reply_comment_prefix.strip()
        text = str(comment_text or "").strip()
        if not prefix or not text.lower().startswith(prefix.lower()):
            return ""
        return text[len(prefix):].lstrip()

    def build_waiting_request_slack_text(self, task_row: sqlite3.Row, waiting_request_text: str) -> str:
        mention = ""
        requester_user_id = task_row["requester_user_id"] or task_row["user_id"] or ""
        if self.requester_needs_mention(task_row) and requester_user_id:
            mention = f"<@{requester_user_id}> "
        return (
            f"{mention}Para poder avanzar, necesito que me pases esto.\n\n"
            f"*Falta información:*\n{waiting_request_text}\n\n"
            "*Estado:* Esperando información"
        )

    def mark_task_waiting_requested(
        self,
        task_id: int,
        *,
        waiting_request_text: str,
        waiting_request_message_ts: str,
        waiting_trello_action_id: str,
    ) -> None:
        timestamp = now_iso()
        with self.db_connect() as conn:
            conn.execute(
                """
                UPDATE tasks
                SET status = 'waiting_for_requester',
                    waiting_requested_at = ?,
                    waiting_request_text = ?,
                    waiting_request_message_ts = ?,
                    waiting_trello_action_id = ?,
                    waiting_cleared_at = NULL,
                    waiting_error = NULL,
                    updated_at = ?,
                    snoozed_until = NULL
                WHERE id = ?
                """,
                (
                    timestamp,
                    waiting_request_text,
                    waiting_request_message_ts,
                    waiting_trello_action_id,
                    timestamp,
                    task_id,
                ),
            )
        self.record_task_event(
            task_id,
            "waiting_requested",
            {
                "waiting_request_text": waiting_request_text,
                "waiting_request_message_ts": waiting_request_message_ts,
                "waiting_trello_action_id": waiting_trello_action_id,
            },
        )

    def mark_task_waiting_failed(self, task_id: int, error_message: str) -> None:
        with self.db_connect() as conn:
            conn.execute(
                """
                UPDATE tasks
                SET waiting_error = ?,
                    updated_at = ?
                WHERE id = ?
                """,
                (error_message[:1000], now_iso(), task_id),
            )
        self.record_task_event(task_id, "waiting_request_failed", {"error": error_message[:1000]})

    def mark_task_trello_reply_command_failed(self, task_id: int, error_message: str) -> None:
        with self.db_connect() as conn:
            conn.execute(
                """
                UPDATE tasks
                SET trello_last_error = ?,
                    updated_at = ?
                WHERE id = ?
                """,
                (error_message[:1000], now_iso(), task_id),
            )
        self.record_task_event(task_id, "trello_reply_failed", {"error": error_message[:1000]})

    def mark_task_waiting_cleared(self, task_id: int) -> None:
        timestamp = now_iso()
        with self.db_connect() as conn:
            conn.execute(
                """
                UPDATE tasks
                SET status = 'new',
                    waiting_cleared_at = ?,
                    waiting_error = NULL,
                    updated_at = ?
                WHERE id = ?
                """,
                (timestamp, timestamp, task_id),
            )
        self.record_task_event(task_id, "waiting_cleared", {})

    def send_waiting_request_to_slack(
        self,
        task_row: sqlite3.Row,
        *,
        waiting_request_text: str,
        waiting_trello_action_id: str,
    ) -> bool:
        try:
            response = self.slack_call(
                self.slack.chat_postMessage,
                channel=task_row["channel_id"],
                thread_ts=task_row["thread_ts"] or task_row["message_ts"],
                text=self.build_waiting_request_slack_text(task_row, waiting_request_text),
                unfurl_links=False,
                unfurl_media=False,
            )
        except Exception as exc:
            self.mark_task_waiting_failed(task_row["id"], str(exc))
            print(f"[yellow]No pude pedir información en Slack para tarea #{task_row['id']}: {exc}[/yellow]")
            return False

        self.mark_task_waiting_requested(
            task_row["id"],
            waiting_request_text=waiting_request_text,
            waiting_request_message_ts=str(response.get("ts") or ""),
            waiting_trello_action_id=waiting_trello_action_id,
        )
        self.record_trello_processed_action(
            task_id=task_row["id"],
            trello_card_id=task_row["trello_card_id"] or "",
            trello_action_id=waiting_trello_action_id,
            action_type="comment_command",
            command_prefix=self.config.trello_waiting_comment_prefix,
            command_text=waiting_request_text,
            status="processed",
        )
        return True

    def send_trello_reply_to_slack(
        self,
        task_row: sqlite3.Row,
        *,
        reply_text: str,
        trello_action_id: str,
    ) -> bool:
        try:
            response = self.slack_call(
                self.slack.chat_postMessage,
                channel=task_row["channel_id"],
                thread_ts=task_row["thread_ts"] or task_row["message_ts"],
                text=reply_text,
                unfurl_links=False,
                unfurl_media=False,
            )
        except Exception as exc:
            error_message = str(exc)
            self.mark_task_trello_reply_command_failed(task_row["id"], error_message)
            self.record_trello_processed_action(
                task_id=task_row["id"],
                trello_card_id=task_row["trello_card_id"] or "",
                trello_action_id=trello_action_id,
                action_type="comment_command",
                command_prefix=self.config.trello_reply_comment_prefix,
                command_text=reply_text,
                status="failed",
                error=error_message,
            )
            self.maybe_comment_trello_card(
                task_row["trello_card_id"] or "",
                f"No pude enviar la respuesta a Slack: {error_message[:220]}",
            )
            print(f"[yellow]No pude enviar respuesta Trello->Slack para tarea #{task_row['id']}: {exc}[/yellow]")
            return False

        reply_ts = str(response.get("ts") or "")
        self.record_trello_processed_action(
            task_id=task_row["id"],
            trello_card_id=task_row["trello_card_id"] or "",
            trello_action_id=trello_action_id,
            action_type="comment_command",
            command_prefix=self.config.trello_reply_comment_prefix,
            command_text=reply_text,
            status="processed",
        )
        self.record_task_event(
            task_row["id"],
            "trello_reply_sent",
            {
                "reply_ts": reply_ts,
                "trello_action_id": trello_action_id,
            },
        )
        self.maybe_comment_trello_card(task_row["trello_card_id"] or "", "Respuesta enviada a Slack.")
        if self.config.trello_reply_mark_responded and task_row["status"] != "waiting_for_requester":
            self.mark_task_reply_sent(task_row["id"], reply_ts)
        return True

    def mark_task_trello_context_failed(self, task_id: int, error_message: str) -> None:
        with self.db_connect() as conn:
            conn.execute(
                """
                UPDATE tasks
                SET trello_last_error = ?,
                    trello_synced_at = ?,
                    updated_at = ?
                WHERE id = ?
                """,
                (error_message[:1000], now_iso(), now_iso(), task_id),
            )
        self.record_task_event(task_id, "trello_context_failed", {"error": error_message[:1000]})

    def add_context_to_trello_card(self, task_row: sqlite3.Row, text: str, sender_label: str) -> None:
        card_id = task_row["trello_card_id"] or ""
        if not card_id:
            return

        if task_row["status"] == "waiting_for_requester":
            comment = f"Respuesta recibida desde Slack: {text.strip()}"
        else:
            comment = "\n".join(
                [
                    "Contexto agregado desde Slack:",
                    "",
                    f"{sender_label}: {text.strip()}",
                ]
            )
        try:
            self.get_trello_client().add_card_comment(card_id, comment)
        except Exception as exc:
            self.mark_task_trello_context_failed(task_row["id"], str(exc))
            print(f"[yellow]No pude agregar contexto en Trello para tarea #{task_row['id']}: {exc}[/yellow]")
            return

        self.record_task_event(task_row["id"], "trello_context_added", {"card_id": card_id})

    def add_context_to_task(
        self,
        task_row: sqlite3.Row,
        *,
        message: dict[str, Any],
        conversation: dict[str, Any],
        sender_label: str,
        conversation_label: str,
    ) -> bool:
        text = (message.get("text") or "").strip()
        visual_attachments = list(message.get("_visual_attachments") or [])
        has_visual_context = bool(visual_attachments)
        context_text = ""
        try:
            context_text = self.fetch_recent_context(
                conversation["id"],
                message["ts"],
                thread_ts=message.get("thread_ts"),
            )
        except Exception:
            context_text = ""

        links = [self.enrich_message_link(link) for link in self.extract_message_links(text)]
        self.upsert_processed_relevance(
            channel_id=conversation["id"],
            message_ts=message["ts"],
            user_id=message.get("user"),
            sender_label=sender_label,
            conversation_label=conversation_label,
            raw_text=text,
            relevant=True,
            relevance_reason=f"Contexto para tarea existente #{task_row['id']}.",
            context_text=context_text,
        )
        self.replace_message_links(conversation["id"], message["ts"], links)

        for attachment in visual_attachments:
            self.save_visual_attachment_metadata(attachment, task_id=task_row["id"])
        self.assign_visual_attachments_to_task(conversation["id"], message["ts"], task_row["id"])

        if not has_visual_context and not self.message_adds_new_context(task_row, text):
            self.mark_processed_context_status(
                channel_id=conversation["id"],
                message_ts=message["ts"],
                status="context_duplicate",
                reason=f"No aporta contexto nuevo para tarea #{task_row['id']}.",
            )
            return False

        with self.db_connect() as conn:
            conn.execute(
                """
                UPDATE tasks
                SET updated_at = ?,
                    public_request_text = ?,
                    status = CASE WHEN status = 'snoozed' THEN 'new' ELSE status END,
                    snoozed_until = CASE WHEN status = 'snoozed' THEN NULL ELSE snoozed_until END
                WHERE id = ?
                """,
                (
                    now_iso(),
                    (
                        self.build_public_request_text(
                            task_row=task_row,
                            transcribed_audio=bool(message.get("_audio_transcript_added")),
                            context_text=text,
                        )
                        if text
                        else task_row["public_request_text"]
                    ),
                    task_row["id"],
                ),
            )

        self.record_task_event(
            task_row["id"],
            "context_added",
            {
                "channel_id": conversation["id"],
                "message_ts": message["ts"],
                "sender_label": sender_label,
                "text": text,
                "visual_file_ids": [attachment.file_id for attachment in visual_attachments],
            },
        )
        if text:
            self.add_context_to_trello_card(task_row, text, sender_label)
        if task_row["trello_card_id"]:
            self.sync_visual_attachments_for_task(task_row["id"], task_row["trello_card_id"])
        self.mark_processed_context_status(
            channel_id=conversation["id"],
            message_ts=message["ts"],
            status="context_added",
            reason=f"Contexto agregado a tarea #{task_row['id']}.",
        )
        if text:
            self.send_context_acknowledgement(
                task_row["id"],
                text,
                transcribed_audio=bool(message.get("_audio_transcript_added")),
            )
        if (
            task_row["status"] == "waiting_for_requester"
            and self.config.trello_waiting_auto_clear
            and (text or has_visual_context)
        ):
            self.mark_task_waiting_cleared(task_row["id"])
        return True

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

    def get_message_link_objects(self, channel_id: str, message_ts: str) -> list[MessageLink]:
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
        return links

    def get_message_links_context(self, channel_id: str, message_ts: str) -> str:
        links = self.get_message_link_objects(channel_id, message_ts)
        return format_links_for_prompt(links)

    def task_context_additions_text(self, task_id: int, limit: int = 8) -> str:
        with self.db_connect() as conn:
            rows = conn.execute(
                """
                SELECT created_at, details_json
                FROM task_events
                WHERE task_id = ?
                  AND event_type = 'context_added'
                ORDER BY created_at ASC, id ASC
                LIMIT ?
                """,
                (task_id, limit),
            ).fetchall()

        lines = []
        for row in rows:
            try:
                details = json.loads(row["details_json"] or "{}")
            except json.JSONDecodeError:
                details = {}
            sender = details.get("sender_label") or "unknown"
            text = compact_text(details.get("text") or "", 500)
            if text:
                lines.append(f"- {sender}: {text}")
        return "\n".join(lines)

    def message_mentions_me(self, text: str, my_user_id: str) -> bool:
        if f"<@{my_user_id}>" in text:
            return True

        normalized_text = normalize_for_matching(text)
        for alias in self.config.my_mention_aliases:
            if re.search(rf"(?<!\w){re.escape(alias)}(?!\w)", normalized_text):
                return True
        return False

    def message_thread_ts(self, message: dict[str, Any]) -> str:
        return str(message.get("thread_ts") or message.get("ts") or "")

    def build_case_key(self, channel_id: str, thread_ts: str) -> str:
        return f"{channel_id}:{thread_ts}"

    def open_task_statuses_sql(self) -> tuple[str, ...]:
        return (
            "new",
            "snoozed",
            "reply_approved",
            "done_pending_reply",
            "waiting_for_requester",
        )

    def requester_needs_mention(self, task_row: sqlite3.Row) -> bool:
        channel_id = task_row["channel_id"] or ""
        return not channel_id.startswith("D")

    def is_direct_message_conversation(self, conversation: dict[str, Any]) -> bool:
        channel_id = str(conversation.get("id") or "")
        return bool(conversation.get("is_im")) or channel_id.startswith("D")

    def requester_address(self, task_row: sqlite3.Row) -> str:
        requester_user_id = task_row["requester_user_id"] or task_row["user_id"] or ""
        if self.requester_needs_mention(task_row) and requester_user_id:
            return f"<@{requester_user_id}>"
        requester_label = task_row["requester_label"] or task_row["sender_label"] or ""
        first_name = requester_label.split()[0] if requester_label else ""
        return first_name or "Hola"

    def local_zone(self) -> ZoneInfo:
        return ZoneInfo(self.config.local_timezone)

    def is_same_local_day(self, left: datetime, right: datetime) -> bool:
        zone = self.local_zone()
        return left.astimezone(zone).date() == right.astimezone(zone).date()

    def is_message_within_context_window(self, current_ts: str, previous_ts: str) -> bool:
        current_dt = parse_slack_timestamp(current_ts)
        previous_dt = parse_slack_timestamp(previous_ts)
        if not current_dt or not previous_dt:
            return False

        age_minutes = (current_dt - previous_dt).total_seconds() / 60
        if age_minutes < 0:
            return False
        if self.config.context_max_age_minutes == 0:
            return False
        if age_minutes > self.config.context_max_age_minutes:
            return False
        return self.is_same_local_day(current_dt, previous_dt)

    def is_message_within_grouping_window(self, current_ts: str, previous_ts: str) -> bool:
        current_dt = parse_slack_timestamp(current_ts)
        previous_dt = parse_slack_timestamp(previous_ts)
        if not current_dt or not previous_dt:
            return False

        age_minutes = (current_dt - previous_dt).total_seconds() / 60
        if age_minutes < 0:
            return False
        if self.config.case_grouping_window_minutes == 0:
            return False
        if age_minutes > self.config.case_grouping_window_minutes:
            return False
        return self.is_same_local_day(current_dt, previous_dt)

    def find_existing_case_for_message(
        self,
        message: dict[str, Any],
        conversation: dict[str, Any],
        sender_label: str,
    ) -> Optional[sqlite3.Row]:
        channel_id = conversation["id"]
        thread_ts = self.message_thread_ts(message)
        if not thread_ts:
            return None

        statuses = self.open_task_statuses_sql()
        placeholders = ", ".join("?" for _ in statuses)
        with self.db_connect() as conn:
            row = conn.execute(
                f"""
                SELECT tasks.*,
                       processed_messages.raw_text,
                       processed_messages.context_text
                FROM tasks
                LEFT JOIN processed_messages
                  ON processed_messages.channel_id = tasks.channel_id
                 AND processed_messages.message_ts = tasks.message_ts
                WHERE tasks.channel_id = ?
                  AND COALESCE(tasks.thread_ts, tasks.message_ts) = ?
                  AND COALESCE(tasks.status, 'new') IN ({placeholders})
                ORDER BY tasks.id DESC
                LIMIT 1
                """,
                (channel_id, thread_ts, *statuses),
            ).fetchone()
            if row and row["message_ts"] != message.get("ts"):
                return row

            if message.get("thread_ts"):
                return None

            if starts_new_request(message.get("text", "")):
                return None

            requester_user_id = message.get("user") or ""
            if self.config.case_grouping_window_minutes <= 0:
                return None

            recent_cutoff = (
                datetime.now(timezone.utc)
                - timedelta(minutes=self.config.case_grouping_window_minutes)
            ).isoformat()
            if self.is_direct_message_conversation(conversation) and requester_user_id:
                recent_row = conn.execute(
                    f"""
                    SELECT tasks.*,
                           processed_messages.raw_text,
                           processed_messages.context_text
                    FROM tasks
                    LEFT JOIN processed_messages
                      ON processed_messages.channel_id = tasks.channel_id
                     AND processed_messages.message_ts = tasks.message_ts
                    WHERE tasks.channel_id = ?
                      AND COALESCE(tasks.status, 'new') IN ({placeholders})
                      AND COALESCE(tasks.updated_at, tasks.created_at) >= ?
                      AND COALESCE(tasks.requester_user_id, tasks.user_id, '') = ?
                    ORDER BY COALESCE(tasks.updated_at, tasks.created_at) DESC, tasks.id DESC
                    LIMIT 1
                    """,
                    (channel_id, *statuses, recent_cutoff, requester_user_id),
                ).fetchone()
                if (
                    recent_row
                    and recent_row["message_ts"] != message.get("ts")
                    and self.is_message_within_grouping_window(
                        str(message.get("ts") or ""),
                        str(recent_row["message_ts"] or ""),
                    )
                ):
                    return recent_row

            if not self.looks_like_context_message(message.get("text", "")):
                return None

            recent_rows = conn.execute(
                f"""
                SELECT tasks.*,
                       processed_messages.raw_text,
                       processed_messages.context_text
                FROM tasks
                LEFT JOIN processed_messages
                  ON processed_messages.channel_id = tasks.channel_id
                 AND processed_messages.message_ts = tasks.message_ts
                WHERE tasks.channel_id = ?
                  AND COALESCE(tasks.status, 'new') IN ({placeholders})
                  AND COALESCE(tasks.updated_at, tasks.created_at) >= ?
                  AND (
                    COALESCE(tasks.requester_user_id, tasks.user_id, '') = ?
                    OR COALESCE(tasks.requester_label, tasks.sender_label, '') = ?
                  )
                ORDER BY COALESCE(tasks.updated_at, tasks.created_at) DESC, tasks.id DESC
                LIMIT 2
                """,
                (channel_id, *statuses, recent_cutoff, requester_user_id, sender_label),
            ).fetchall()

        eligible_rows = [
            row
            for row in recent_rows
            if row["message_ts"] != message.get("ts")
            and self.is_message_within_grouping_window(
                str(message.get("ts") or ""),
                str(row["message_ts"] or ""),
            )
        ]
        if len(eligible_rows) == 1:
            return eligible_rows[0]
        return None

    def looks_like_context_message(self, text: str) -> bool:
        normalized = normalize_for_matching(text)
        if not normalized:
            return False
        starters = (
            "ademas",
            "tambien",
            "contexto",
            "dato",
            "detalle",
            "sumo",
            "agrego",
            "te paso",
            "aca",
            "ahi",
            "me olvide",
            "por las dudas",
            "el link",
            "la captura",
        )
        return normalized.startswith(starters) or "http://" in normalized or "https://" in normalized

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
        action = task_row["public_request_text"] or task_row["requested_action"] or "Sin acción especificada"
        raw_text = task_row["raw_text"] or ""
        context_text = task_row["context_text"] or ""
        context_additions = self.task_context_additions_text(task_row["id"])
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
                "Contexto agregado:",
                context_additions or "Sin contexto agregado.",
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
        self.sync_visual_attachments_for_task(task_row["id"], card.id)
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

    def is_trello_card_done_by_check(self, card: TrelloCardState) -> bool:
        return bool(card.due_complete)

    def is_trello_card_done_by_list(self, card: TrelloCardState) -> bool:
        if self.config.trello_done_list_id and card.list_id == self.config.trello_done_list_id:
            return True
        normalized_list_name = normalize_for_matching(card.list_name)
        return bool(normalized_list_name and normalized_list_name in self.config.trello_done_list_names)

    def is_trello_card_done_by_checklist(self, card: TrelloCardState) -> bool:
        wanted = normalize_for_matching(self.config.trello_done_checklist_item_name)
        if not wanted:
            return False
        for item in card.checklist_items:
            item_name = normalize_for_matching(str(item.get("name") or ""))
            item_state = normalize_for_matching(str(item.get("state") or ""))
            if item_name == wanted and item_state == "complete":
                return True
        return False

    def is_trello_card_done(self, card: TrelloCardState) -> bool:
        mode = self.config.trello_done_mode
        if mode == "check":
            return self.is_trello_card_done_by_check(card)
        if mode == "list":
            return self.is_trello_card_done_by_list(card)
        if mode == "checklist":
            return self.is_trello_card_done_by_checklist(card)
        if mode == "list_or_check":
            return self.is_trello_card_done_by_check(card) or self.is_trello_card_done_by_list(card)
        return False

    def mark_task_trello_check_failed(self, task_id: int, error_message: str) -> None:
        with self.db_connect() as conn:
            conn.execute(
                """
                UPDATE tasks
                SET trello_last_error = ?,
                    trello_synced_at = ?,
                    updated_at = ?
                WHERE id = ?
                """,
                (error_message[:1000], now_iso(), now_iso(), task_id),
            )
        self.record_task_event(task_id, "trello_done_check_failed", {"error": error_message[:1000]})

    def build_final_reply_suggestion(self, task_row: sqlite3.Row) -> str:
        address = self.requester_address(task_row)
        summary = compact_text(task_row["summary"] or "el pedido", 180)
        return f"{address}, ya quedó resuelto lo que me pediste sobre {summary}."

    def mark_task_done_pending_reply(self, task_id: int, final_reply_suggestion: str) -> bool:
        timestamp = now_iso()
        with self.db_connect() as conn:
            cursor = conn.execute(
                """
                UPDATE tasks
                SET status = 'done_pending_reply',
                    reviewed_at = ?,
                    done_pending_reply_at = COALESCE(NULLIF(done_pending_reply_at, ''), ?),
                    final_reply_suggestion = ?,
                    reply_error = NULL,
                    updated_at = ?,
                    snoozed_until = NULL
                WHERE id = ?
                  AND COALESCE(status, 'new') != 'done_pending_reply'
                """,
                (timestamp, timestamp, final_reply_suggestion, timestamp, task_id),
            )
        if cursor.rowcount > 0:
            self.record_task_event(
                task_id,
                "done_pending_reply",
                {"final_reply_suggestion": final_reply_suggestion},
            )
            return True
        return False

    def mark_task_telegram_notified(self, task_id: int) -> None:
        with self.db_connect() as conn:
            conn.execute(
                """
                UPDATE tasks
                SET telegram_notified_at = ?,
                    telegram_error = NULL,
                    updated_at = ?
                WHERE id = ?
                """,
                (now_iso(), now_iso(), task_id),
            )
        self.record_task_event(task_id, "telegram_notified", {})

    def mark_task_telegram_failed(self, task_id: int, error_message: str) -> None:
        with self.db_connect() as conn:
            conn.execute(
                """
                UPDATE tasks
                SET telegram_error = ?,
                    updated_at = ?
                WHERE id = ?
                """,
                (error_message[:1000], now_iso(), task_id),
            )
        self.record_task_event(task_id, "telegram_failed", {"error": error_message[:1000]})

    def build_telegram_approval_message(self, task_row: sqlite3.Row) -> str:
        context_additions = self.task_context_additions_text(task_row["id"])
        context_parts = [
            f"Original: {compact_text(task_row['raw_text'] or '(sin texto)', 700)}",
        ]
        if task_row["context_text"]:
            context_parts.append(f"Contexto reciente: {compact_text(task_row['context_text'], 700)}")
        if context_additions:
            context_parts.append(f"Agregado: {compact_text(context_additions, 900)}")

        suggestion = task_row["manual_reply"] or task_row["final_reply_suggestion"] or self.build_final_reply_suggestion(task_row)
        return "\n".join(
            [
                f"Tarea #{task_row['id']} lista para respuesta final",
                f"Solicitante: {task_row['requester_label'] or task_row['sender_label'] or 'unknown'}",
                f"Resumen: {task_row['summary'] or 'Sin resumen'}",
                "",
                "Contexto relevante:",
                "\n".join(context_parts),
                "",
                "Respuesta final sugerida:",
                suggestion,
                "",
                "Comandos:",
                f"/send {task_row['id']}",
                f"/edit {task_row['id']} texto",
                f"/nosend {task_row['id']}",
            ]
        )

    def send_done_pending_telegram(self, task_id: int) -> bool:
        task_row = self.get_task_by_id(task_id)
        if not task_row:
            return False

        if not self.config.telegram_enabled:
            self.mark_task_telegram_failed(task_id, "Telegram está deshabilitado.")
            return False

        try:
            client = self.get_telegram_client()
            client.send_message(self.build_telegram_approval_message(task_row))
        except Exception as exc:
            self.mark_task_telegram_failed(task_id, str(exc))
            print(f"[yellow]No pude enviar aprobación Telegram para tarea #{task_id}: {exc}[/yellow]")
            return False

        self.mark_task_telegram_notified(task_id)
        return True

    def sync_trello_reply_commands(self, limit: int = 50) -> int:
        if not self.config.trello_enabled or not self.config.trello_reply_enabled:
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
                WHERE tasks.trello_card_id IS NOT NULL
                  AND tasks.trello_card_id != ''
                  AND COALESCE(tasks.status, 'new') NOT IN (
                    'done',
                    'ignored',
                    'dismissed',
                    'archived'
                  )
                ORDER BY tasks.id ASC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()

        sent = 0
        for row in rows:
            try:
                comments = self.get_trello_client().get_card_comments(row["trello_card_id"], limit=25)
            except Exception as exc:
                self.mark_task_trello_reply_command_failed(row["id"], str(exc))
                print(f"[yellow]No pude leer comentarios Trello para tarea #{row['id']}: {exc}[/yellow]")
                continue

            reply_comment = None
            reply_text = ""
            for comment in sorted(comments, key=lambda item: item.date or "", reverse=True):
                text = self.reply_command_from_comment(comment.text)
                if text and not self.trello_action_already_processed(
                    comment.id,
                    self.config.trello_reply_comment_prefix,
                ):
                    reply_comment = comment
                    reply_text = text
                    break

            if not reply_comment or not reply_text:
                continue

            if self.send_trello_reply_to_slack(
                row,
                reply_text=reply_text,
                trello_action_id=reply_comment.id,
            ):
                sent += 1

        return sent

    def sync_trello_waiting_requests(self, limit: int = 50) -> int:
        if not self.config.trello_enabled or not self.config.trello_waiting_enabled:
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
                WHERE tasks.trello_card_id IS NOT NULL
                  AND tasks.trello_card_id != ''
                  AND COALESCE(tasks.status, 'new') NOT IN (
                    'done',
                    'ignored',
                    'dismissed',
                    'archived',
                    'responded',
                    'done_pending_reply'
                  )
                ORDER BY tasks.id ASC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()

        requested = 0
        for row in rows:
            if row["status"] == "waiting_for_requester":
                continue
            try:
                comments = self.get_trello_client().get_card_comments(row["trello_card_id"], limit=25)
            except Exception as exc:
                self.mark_task_waiting_failed(row["id"], str(exc))
                print(f"[yellow]No pude leer comentarios Trello para tarea #{row['id']}: {exc}[/yellow]")
                continue

            waiting_comment = None
            waiting_text = ""
            for comment in sorted(comments, key=lambda item: item.date or "", reverse=True):
                text = self.waiting_request_from_comment(comment.text)
                if text:
                    waiting_comment = comment
                    waiting_text = text
                    break

            if not waiting_comment or not waiting_text:
                continue
            if waiting_comment.id and (
                waiting_comment.id == (row["waiting_trello_action_id"] or "")
                or self.trello_action_already_processed(
                    waiting_comment.id,
                    self.config.trello_waiting_comment_prefix,
                )
            ):
                continue

            if self.send_waiting_request_to_slack(
                row,
                waiting_request_text=waiting_text,
                waiting_trello_action_id=waiting_comment.id,
            ):
                requested += 1

        return requested

    def build_slack_auto_final_reply_text(self, task_row: sqlite3.Row) -> str:
        public_request_text = task_row["public_request_text"] or self.build_public_request_text(task_row=task_row)
        return "\n\n".join(
            [
                "Listo, ya quedó resuelto.",
                f"*Pedido resuelto:*\n{public_request_text}",
                "*Estado:* Resuelto",
            ]
        )

    def send_slack_auto_final_reply(self, task_row: sqlite3.Row) -> bool:
        try:
            response = self.slack_call(
                self.slack.chat_postMessage,
                channel=task_row["channel_id"],
                thread_ts=task_row["thread_ts"] or task_row["message_ts"],
                text=self.build_slack_auto_final_reply_text(task_row),
                unfurl_links=False,
                unfurl_media=False,
            )
        except Exception as exc:
            self.mark_task_reply_failed(task_row["id"], str(exc))
            print(f"[yellow]No pude enviar cierre automático Slack para tarea #{task_row['id']}: {exc}[/yellow]")
            return False

        reply_ts = str(response.get("ts") or "")
        self.mark_task_reply_sent(task_row["id"], reply_ts)
        self.record_task_event(task_row["id"], "slack_auto_final_reply_sent", {"reply_ts": reply_ts})
        return True

    def sync_trello_done_tasks(self, limit: int = 50) -> int:
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
                WHERE tasks.trello_card_id IS NOT NULL
                  AND tasks.trello_card_id != ''
                  AND COALESCE(tasks.status, 'new') NOT IN (
                    'done',
                    'ignored',
                    'dismissed',
                    'archived',
                    'responded',
                    'done_pending_reply'
                  )
                ORDER BY tasks.id ASC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()

        moved = 0
        for row in rows:
            if row["status"] == "waiting_for_requester":
                continue
            try:
                card = self.get_trello_client().get_card(row["trello_card_id"])
            except Exception as exc:
                self.mark_task_trello_check_failed(row["id"], str(exc))
                print(f"[yellow]No pude chequear Trello para tarea #{row['id']}: {exc}[/yellow]")
                continue

            if not self.is_trello_card_done(card):
                continue

            if self.config.final_reply_mode == "slack_auto":
                moved += int(self.send_slack_auto_final_reply(row))
                continue

            final_reply = self.build_final_reply_suggestion(row)
            if self.mark_task_done_pending_reply(row["id"], final_reply):
                self.send_done_pending_telegram(row["id"])
                moved += 1

        return moved

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

    def fetch_recent_context(
        self,
        channel_id: str,
        before_ts: str,
        limit: int = 6,
        thread_ts: Optional[str] = None,
    ) -> str:
        explicit_thread = bool(thread_ts and thread_ts != before_ts)
        if explicit_thread:
            response = self.slack_call(
                self.slack.conversations_replies,
                channel=channel_id,
                ts=thread_ts,
                limit=limit + 1,
            )
        else:
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
            message_ts = str(message.get("ts") or "")
            if not message_ts or float(message_ts) >= float(before_ts):
                continue
            if not explicit_thread and not self.is_message_within_context_window(before_ts, message_ts):
                continue
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
        if message.get("text"):
            return True
        return bool(
            message.get("files")
            and (self.config.audio_transcription_enabled or self.config.slack_image_attachments_enabled)
        )

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
        new_request_rule = (
            "- El mensaje actual parece iniciar un pedido separado; no mezcles detalles del contexto previo salvo evidencia clara."
            if starts_new_request(text)
            else ""
        )
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
- Si el mensaje pide armar, revisar, generar, exportar, validar o consultar un informe/reporte/dashboard usando links o datos de Salesforce, la categoría debe ser "salesforce".
- Si hay URLs de Salesforce, agregá "salesforce" en external_systems.
- Si el pedido requiere consultar Salesforce o usar campañas/registros de Salesforce, needs_external_system=true.
- No clasifiques como "research" un pedido operativo de extracción, validación o armado de informe desde Salesforce.
- Si el mensaje pide armar, actualizar, exportar, generar o preparar una base, listado, informe o stock de donantes, donaciones, altas, bajas o campañas, clasificalo como "salesforce" si requiere consultar datos del CRM/Salesforce.
- No clasifiques como "research" pedidos operativos de extracción de datos, armado de base o informes de donantes, aunque el mensaje mencione análisis, perfiles, buyer persona, buyer techo o investigación.
- "research" se reserva para búsqueda, análisis conceptual, benchmarking, lectura de fuentes externas o investigación sin extracción operativa de datos internos.
- Si el pedido incluye campos como nombre y apellido, fecha de nacimiento, edad, residencia, monto, estado, fecha establecida, fecha de finalización o campaña, asumí que es una tarea de base/datos internos.
- Si no hay link de Salesforce pero el pedido menciona base de donantes, stock activo, datos personales y datos de donación, marcá needs_external_system=true y agregá "salesforce" a external_systems, salvo que el mensaje indique explícitamente otra fuente.
{new_request_rule}
- El campo requested_action debe quedar redactado para que otro agente pueda ejecutar la tarea sin leer todo el hilo.
- En requested_action incluí qué hay que hacer, período temporal si aparece, segmentación solicitada, fuentes o campañas indicadas y campos requeridos.
- Ejemplo de requested_action bueno: "Armar un informe de altas 2026 por campaña principal/campaña de origen para las campañas indicadas en Salesforce, incluyendo datos de la persona —nombre y apellido, fecha de nacimiento/edad, residencia— y datos de donación —fecha establecida, estado, monto, fecha de finalización y campaña—."
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

        if not str(text or "").strip():
            return {
                **state,
                "relevant": False,
                "relevance_reason": "Mensaje sin texto transcripto.",
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
        if state.get("relevant") and not starts_new_request(message.get("text", "")):
            context_text = self.fetch_recent_context(
                conversation["id"],
                message["ts"],
                thread_ts=message.get("thread_ts"),
            )
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

        message_text = state["message"].get("text", "")
        classification = self.invoke_classification(
            text=message_text,
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
        classification = normalize_classification_with_rules(
            classification,
            text=message_text,
            message_links=state.get("message_links", []),
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
            thread_ts = self.message_thread_ts(message)
            case_key = self.build_case_key(conversation["id"], thread_ts)
            inserted = self.save_task(
                channel_id=conversation["id"],
                message_ts=message["ts"],
                thread_ts=thread_ts,
                case_key=case_key,
                user_id=message.get("user"),
                sender_label=state["sender_label"],
                conversation_label=state["conversation_label"],
                classification=classification,
                raw_text=message.get("text", ""),
                has_audio_transcript=bool(message.get("_audio_transcript_added")),
            )
            if inserted:
                task_row = self.get_task_row(conversation["id"], message["ts"])
                if task_row:
                    self.assign_visual_attachments_to_task(conversation["id"], message["ts"], task_row["id"])
                    self.send_task_acknowledgement(task_row["id"])
                if self.config.trello_enabled and self.config.trello_auto_create:
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
        try:
            existing_case = None
            if message.get("user") != my_user_id or self.config.include_self_for_test:
                existing_case = self.find_existing_case_for_message(message, conversation, sender_label)

            message = self.annotate_message_with_visual_attachments(
                message,
                conversation,
                task_id=existing_case["id"] if existing_case else None,
            )
            message = self.enrich_message_with_audio(
                message,
                conversation,
                task_id=existing_case["id"] if existing_case else None,
            )
            state: AgentState = {
                "message": message,
                "conversation": conversation,
                "my_user_id": my_user_id,
                "sender_label": sender_label,
                "conversation_label": conversation_label,
            }

            if message.get("user") != my_user_id or self.config.include_self_for_test:
                if existing_case is None:
                    existing_case = self.find_existing_case_for_message(message, conversation, sender_label)
                if existing_case:
                    self.add_context_to_task(
                        existing_case,
                        message=message,
                        conversation=conversation,
                        sender_label=sender_label,
                        conversation_label=conversation_label,
                    )
                    return
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
                classification = normalize_classification_with_rules(
                    classification,
                    text=row["raw_text"] or "",
                    message_links=self.get_message_link_objects(row["channel_id"], row["message_ts"]),
                )
                self.mark_processed_done(
                    channel_id=row["channel_id"],
                    message_ts=row["message_ts"],
                    classification=classification,
                )
                if classification.is_actionable:
                    thread_ts = row["message_ts"]
                    inserted = self.save_task(
                        channel_id=row["channel_id"],
                        message_ts=row["message_ts"],
                        thread_ts=thread_ts,
                        case_key=self.build_case_key(row["channel_id"], thread_ts),
                        user_id=row["user_id"],
                        sender_label=row["sender_label"] or "unknown",
                        conversation_label=row["conversation_label"] or row["channel_id"],
                        classification=classification,
                        raw_text=row["raw_text"] or "",
                        has_audio_transcript=False,
                    )
                    if inserted:
                        task_row = self.get_task_row(row["channel_id"], row["message_ts"])
                        if task_row:
                            self.assign_visual_attachments_to_task(row["channel_id"], row["message_ts"], task_row["id"])
                            self.send_task_acknowledgement(task_row["id"])
                        if self.config.trello_enabled and self.config.trello_auto_create:
                            self.sync_task_to_trello_by_message(row["channel_id"], row["message_ts"])
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
        if self.config.trello_enabled:
            self.sync_trello_reply_commands(limit=25)
            if self.config.sync_waiting_enabled:
                self.sync_trello_waiting_requests(limit=25)
            self.sync_trello_done_tasks(limit=25)
        if self.config.sync_telegram_poll_enabled and self.config.final_reply_mode == "telegram_approval":
            self.poll_telegram_updates(limit=25)
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
                if status in {"done", "ignored", "context_added", "context_duplicate"}:
                    continue

                self.process_message(message, conversation, my_user_id=my_user_id)
                total_new += 1

            self.set_last_seen_ts(channel_id, max_ts)
            self._sleep(self.config.slack_sleep_seconds)

        self.retry_failed_messages(limit=25)
        if self.config.trello_enabled and self.config.trello_auto_create:
            self.sync_pending_trello_tasks(limit=25)
        if self.config.trello_enabled:
            self.sync_trello_reply_commands(limit=25)
            if self.config.sync_waiting_enabled:
                self.sync_trello_waiting_requests(limit=25)
            self.sync_trello_done_tasks(limit=25)
        if self.config.sync_telegram_poll_enabled and self.config.final_reply_mode == "telegram_approval":
            self.poll_telegram_updates(limit=25)
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
            "done_pending_reply_at": row["done_pending_reply_at"] or "",
            "final_reply_suggestion": row["final_reply_suggestion"] or "",
            "telegram_error": row["telegram_error"] or "",
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
        if task["status"] == "done_pending_reply":
            details.append("Respuesta final pendiente de Telegram")
        if task["reply_error"]:
            details.append(f"Error Slack: {compact_text(task['reply_error'], 90)}")
        if task["telegram_error"]:
            details.append(f"Error Telegram: {compact_text(task['telegram_error'], 90)}")
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
                    updated_at = ?,
                    snoozed_until = CASE WHEN ? != 'snoozed' THEN NULL ELSE snoozed_until END
                WHERE id = ?
                """,
                (status, now_iso(), now_iso(), status, task_id),
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
                    updated_at = ?,
                    snoozed_until = NULL
                WHERE id = ?
                """,
                (now_iso(), now_iso(), reply_text, now_iso(), task_id),
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
                    updated_at = ?,
                    snoozed_until = NULL
                WHERE id = ?
                """,
                (now_iso(), now_iso(), reply_ts, now_iso(), task_id),
            )
        self.record_task_event(task_id, "reply_sent", {"reply_ts": reply_ts})

    def mark_task_reply_failed(self, task_id: int, error_message: str) -> None:
        with self.db_connect() as conn:
            conn.execute(
                """
                UPDATE tasks
                SET reviewed_at = ?,
                    reply_error = ?,
                    updated_at = ?
                WHERE id = ?
                """,
                (now_iso(), error_message[:1000], now_iso(), task_id),
            )
        self.record_task_event(task_id, "reply_failed", {"error": error_message[:1000]})

    def reply_text_for_task(self, task_row: sqlite3.Row) -> str:
        if task_row["manual_reply"]:
            return task_row["manual_reply"].strip()

        if task_row["final_reply_suggestion"]:
            return task_row["final_reply_suggestion"].strip()

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

    def edit_task_manual_reply(self, task_id: int, reply_text: str) -> None:
        with self.db_connect() as conn:
            conn.execute(
                """
                UPDATE tasks
                SET manual_reply = ?,
                    reply_error = NULL,
                    updated_at = ?
                WHERE id = ?
                """,
                (reply_text, now_iso(), task_id),
            )
        self.record_task_event(task_id, "manual_reply_edited", {"reply_text": reply_text})

    def handle_telegram_command(self, command_text: str) -> bool:
        self.init_db()
        raw = command_text.strip()
        if not raw:
            return False

        parts = raw.split(maxsplit=2)
        command = parts[0].split("@", 1)[0].lower()
        if command not in {"/send", "/edit", "/nosend"}:
            return False
        if len(parts) < 2 or not parts[1].isdigit():
            return False

        task_id = int(parts[1])
        task_row = self.get_task_by_id(task_id)
        if not task_row:
            raise RuntimeError(f"No encontré la tarea #{task_id}.")

        if command == "/edit":
            if len(parts) < 3 or not parts[2].strip():
                raise RuntimeError("/edit requiere texto.")
            self.edit_task_manual_reply(task_id, parts[2].strip())
            return True

        if command == "/send":
            return self.send_approved_reply(task_id)

        if command == "/nosend":
            self.mark_task_status(task_id, "done")
            self.record_task_event(task_id, "final_reply_suppressed", {})
            return True

        return False

    def poll_telegram_updates(self, limit: int = 20) -> int:
        if self.config.final_reply_mode != "telegram_approval":
            return 0
        if not self.config.telegram_enabled or not self.has_telegram_config():
            return 0

        self.init_db()
        raw_offset = self.get_agent_state("telegram_update_offset")
        offset = int(raw_offset) if raw_offset and raw_offset.isdigit() else None
        try:
            updates = self.get_telegram_client().get_updates(offset=offset, limit=limit)
        except Exception as exc:
            print(f"[yellow]No pude leer updates de Telegram: {exc}[/yellow]")
            return 0

        handled = 0
        next_offset = offset
        for update in updates:
            update_id = update.get("update_id")
            if isinstance(update_id, int):
                next_offset = max(next_offset or 0, update_id + 1)

            message = update.get("message") or update.get("edited_message") or {}
            chat = message.get("chat") or {}
            if str(chat.get("id") or "") != str(self.config.telegram_chat_id):
                continue

            text = message.get("text") or ""
            if not text:
                continue

            try:
                handled += int(self.handle_telegram_command(text))
            except Exception as exc:
                print(f"[yellow]No pude procesar comando Telegram `{text}`: {exc}[/yellow]")

        if next_offset is not None:
            self.set_agent_state("telegram_update_offset", str(next_offset))
        return handled

    def snooze_task(self, task_id: int, snoozed_until: datetime) -> None:
        if snoozed_until.tzinfo is None:
            snoozed_until = snoozed_until.replace(tzinfo=timezone.utc)
        with self.db_connect() as conn:
            conn.execute(
                """
                UPDATE tasks
                SET status = 'snoozed',
                    reviewed_at = ?,
                    snoozed_until = ?,
                    updated_at = ?
                WHERE id = ?
                """,
                (now_iso(), snoozed_until.astimezone(timezone.utc).isoformat(), now_iso(), task_id),
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
            sync_worker_seconds=self.config.sync_worker_seconds,
            sync_waiting_enabled=self.config.sync_waiting_enabled,
            sync_trello_done_enabled=self.config.sync_trello_done_enabled,
            sync_telegram_poll_enabled=self.config.sync_telegram_poll_enabled,
            final_reply_mode=self.config.final_reply_mode,
        )
        print("[green]Autostart instalado.[/green]")
        print(f"Ollama plist: {artifacts.ollama_plist}")
        print(f"Agente plist: {artifacts.agent_plist}")
        print(f"Sync plist: {artifacts.sync_plist}")
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
            print("[bold]Chequeo Slack escritura[/bold]")
            print(
                "[yellow]Los acuses automáticos, confirmaciones de contexto y respuestas aprobadas "
                "usan chat.postMessage; el token necesita el scope chat:write o chat:write:bot. "
                "Si Slack devuelve missing_scope con needed=chat:write:bot, reinstalá/actualizá "
                "la app o el token con ese permiso.[/yellow]"
            )
            if self.config.final_reply_mode == "slack_auto":
                print(
                    "[yellow]FINAL_REPLY_MODE=slack_auto está activo: cuando Trello esté marcado "
                    "como hecho, el cierre final se enviará directo a Slack.[/yellow]"
                )
            if self.config.slack_image_attachments_enabled and self.config.trello_attach_slack_images:
                print(
                    "[yellow]Las imágenes de Slack usan URLs privadas; si querés adjuntarlas a Trello "
                    "el token necesita poder leer archivos (`files:read`).[/yellow]"
                )
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

        if self.config.local_whisper_enabled:
            print("[bold]Chequeo Whisper local[/bold]")
            whisper_backend = detect_local_whisper_backend()
            if whisper_backend:
                print(f"[green]Whisper local OK:[/green] {whisper_backend}")
            else:
                ok = False
                print(
                    "[red]LOCAL_WHISPER_ENABLED=true pero no encontré faster-whisper ni "
                    "openai-whisper instalados.[/red]"
                )
                print(
                    "[yellow]Instalá dependencias con `pip install -r requirements.txt` "
                    "o desactivá LOCAL_WHISPER_ENABLED=false si solo querés usar transcripts de Slack.[/yellow]"
                )
        else:
            print("[dim]Whisper local deshabilitado.[/dim]")

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
                    if self.config.trello_waiting_enabled or self.config.trello_reply_enabled:
                        print("[bold]Chequeo Trello comentarios[/bold]")
                        with self.db_connect() as conn:
                            sample = conn.execute(
                                """
                                SELECT trello_card_id
                                FROM tasks
                                WHERE trello_card_id IS NOT NULL AND trello_card_id != ''
                                ORDER BY id DESC
                                LIMIT 1
                                """
                            ).fetchone()
                        if sample:
                            client.get_card_comments(sample["trello_card_id"], limit=1)
                            print("[green]Lectura de comentarios Trello OK[/green]")
                        else:
                            print(
                                "[yellow]No hay cards locales para probar lectura de comentarios. "
                                "Los comandos Trello por comentario requieren token Trello con scope read.[/yellow]"
                            )
                    if self.config.trello_attach_slack_images:
                        print(
                            f"[yellow]Adjuntos visuales a Trello activos en modo "
                            f"{self.config.trello_image_attachment_mode}; imágenes grandes pueden fallar "
                            f"si superan SLACK_IMAGE_MAX_BYTES={self.config.slack_image_max_bytes}.[/yellow]"
                        )
                except Exception as exc:
                    ok = False
                    print(f"[red]Trello falló:[/red] {exc}")
        else:
            print("[dim]Trello no está habilitado.[/dim]")

        return ok
