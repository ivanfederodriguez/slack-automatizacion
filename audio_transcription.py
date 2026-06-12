from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Protocol


AUDIO_FILETYPES = {
    "aac",
    "aiff",
    "amr",
    "flac",
    "m4a",
    "mp3",
    "mp4",
    "oga",
    "ogg",
    "opus",
    "wav",
    "webm",
}


class AudioTranscriber(Protocol):
    def transcribe(self, audio_path: Path) -> str:
        ...


@dataclass(frozen=True)
class AudioAttachment:
    source: str
    source_message_id: str
    channel_id: str
    message_ts: str
    file_id: str
    filename: str
    mime_type: str
    url_private: str
    slack_transcript_text: str = ""
    index: int = 0
    duration_seconds: Optional[float] = None


@dataclass(frozen=True)
class AudioTranscriptResult:
    attachment: AudioAttachment
    slack_transcript_text: str = ""
    local_transcript_text: str = ""
    fused_transcript_text: str = ""
    selected_transcript_text: str = ""
    transcription_status: str = "missing"
    transcription_error: str = ""


class LocalWhisperTranscriber:
    def __init__(
        self,
        *,
        model_name: str = "tiny",
        language: str = "es",
        device: str = "auto",
        compute_type: str = "auto",
    ) -> None:
        self.model_name = model_name
        self.language = language
        self.device = device
        self.compute_type = compute_type
        self._model: Any = None
        self._backend = ""

    def transcribe(self, audio_path: Path) -> str:
        if self._model is None:
            self._load_model()

        if self._backend == "faster_whisper":
            segments, _info = self._model.transcribe(str(audio_path), language=self.language or None)
            return clean_transcript_text(" ".join(segment.text for segment in segments))

        result = self._model.transcribe(str(audio_path), language=self.language or None)
        return clean_transcript_text(result.get("text") or "")

    def _load_model(self) -> None:
        try:
            from faster_whisper import WhisperModel

            compute_type = "default" if self.compute_type == "auto" else self.compute_type
            self._model = WhisperModel(
                self.model_name,
                device=self.device,
                compute_type=compute_type,
            )
            self._backend = "faster_whisper"
            return
        except ImportError:
            pass

        try:
            import whisper
        except ImportError as exc:
            raise RuntimeError(
                "Falta instalar `faster-whisper` u `openai-whisper` para transcribir audio local."
            ) from exc

        kwargs: dict[str, Any] = {}
        if self.device and self.device != "auto":
            kwargs["device"] = self.device
        self._model = whisper.load_model(self.model_name, **kwargs)
        self._backend = "openai_whisper"


def clean_transcript_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def is_audio_file(file_payload: dict[str, Any]) -> bool:
    mimetype = str(file_payload.get("mimetype") or "").lower()
    if mimetype.startswith("audio/"):
        return True

    filetype = str(file_payload.get("filetype") or "").lower()
    if filetype in AUDIO_FILETYPES:
        return True

    name = str(file_payload.get("name") or file_payload.get("title") or "").lower()
    suffix = Path(name).suffix.lstrip(".")
    return suffix in AUDIO_FILETYPES


def extract_slack_transcript_text(file_payload: dict[str, Any]) -> str:
    for key in (
        "transcription_text",
        "transcript_text",
        "transcription_preview",
        "transcript",
        "transcription",
    ):
        value = file_payload.get(key)
        text = transcript_text_from_value(value)
        if text:
            return text

    for key, value in file_payload.items():
        lowered = str(key).lower()
        if "transcript" not in lowered and "transcription" not in lowered:
            continue
        text = transcript_text_from_value(value)
        if text:
            return text
    return ""


def transcript_text_from_value(value: Any) -> str:
    if isinstance(value, str):
        return clean_transcript_text(value)
    if isinstance(value, dict):
        for key in ("text", "transcript", "transcription", "preview", "content"):
            text = transcript_text_from_value(value.get(key))
            if text:
                return text
    if isinstance(value, list):
        parts = [transcript_text_from_value(item) for item in value]
        return clean_transcript_text(" ".join(part for part in parts if part))
    return ""


def detect_audio_attachments(message: dict[str, Any], *, source: str = "slack") -> list[AudioAttachment]:
    channel_id = str(message.get("channel") or message.get("channel_id") or "")
    message_ts = str(message.get("ts") or "")
    attachments = []
    for index, file_payload in enumerate(message.get("files") or [], start=1):
        if not isinstance(file_payload, dict) or not is_audio_file(file_payload):
            continue
        file_id = str(file_payload.get("id") or file_payload.get("file_id") or f"audio-{index}")
        filename = str(file_payload.get("name") or file_payload.get("title") or file_id)
        attachments.append(
            AudioAttachment(
                source=source,
                source_message_id=f"{channel_id}:{message_ts}",
                channel_id=channel_id,
                message_ts=message_ts,
                file_id=file_id,
                filename=filename,
                mime_type=str(file_payload.get("mimetype") or ""),
                url_private=str(
                    file_payload.get("url_private_download")
                    or file_payload.get("url_private")
                    or ""
                ),
                slack_transcript_text=extract_slack_transcript_text(file_payload),
                index=index,
                duration_seconds=extract_duration_seconds(file_payload),
            )
        )
    return attachments


def extract_duration_seconds(file_payload: dict[str, Any]) -> Optional[float]:
    for key in ("duration_seconds", "duration_secs", "duration"):
        value = file_payload.get(key)
        if isinstance(value, (int, float)):
            return float(value)
    duration_ms = file_payload.get("duration_ms")
    if isinstance(duration_ms, (int, float)):
        return float(duration_ms) / 1000
    return None


def format_audio_transcripts_for_message(results: list[AudioTranscriptResult]) -> str:
    lines = []
    for display_index, result in enumerate(results, start=1):
        text = clean_transcript_text(result.selected_transcript_text)
        if text:
            lines.append(f"[Audio {display_index}]: {text}")
    if not lines:
        return ""
    return "Audios transcriptos:\n" + "\n".join(lines)


def combine_text_and_audio(original_text: str, audio_text: str) -> str:
    original = clean_transcript_text(original_text)
    audio = audio_text.strip()
    if original and audio:
        return f"Texto original:\n{original}\n\n{audio}"
    if audio:
        return audio
    return original


def choose_audio_transcript(
    *,
    slack_text: str,
    local_text: str,
    fused_text: str = "",
) -> tuple[str, str]:
    slack_clean = clean_transcript_text(slack_text)
    local_clean = clean_transcript_text(local_text)
    fused_clean = clean_transcript_text(fused_text)
    if fused_clean:
        return fused_clean, "fused"
    if slack_clean and local_clean:
        return local_clean, "local_only"
    if slack_clean:
        return slack_clean, "slack_only"
    if local_clean:
        return local_clean, "local_only"
    return "", "missing"
