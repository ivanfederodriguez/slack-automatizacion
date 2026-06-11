from __future__ import annotations

import html
import json
import re
from dataclasses import dataclass, field
from typing import Any, Optional
from urllib.parse import urlparse


SLACK_LINK_RE = re.compile(r"<(https?://[^>|]+)(?:\|[^>]+)?>")
PLAIN_URL_RE = re.compile(r"https?://[^\s<>()]+")
TITLE_RE = re.compile(r"<title[^>]*>(.*?)</title>", re.IGNORECASE | re.DOTALL)
META_DESC_RE = re.compile(
    r'<meta[^>]+name=["\']description["\'][^>]+content=["\'](.*?)["\']',
    re.IGNORECASE | re.DOTALL,
)


@dataclass
class MessageLink:
    url: str
    domain: str
    url_type: str
    title: str = ""
    summary: str = ""
    status: str = "detected"
    error: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def metadata_json(self) -> str:
        return json.dumps(self.metadata, ensure_ascii=False, sort_keys=True)


def clean_extracted_url(url: str) -> str:
    cleaned = url.strip()
    if "|" in cleaned:
        cleaned = cleaned.split("|", 1)[0]
    while cleaned and cleaned[-1] in ".,);]>":
        cleaned = cleaned[:-1]
    return cleaned


def extract_urls_from_text(text: str) -> list[str]:
    seen: set[str] = set()
    urls: list[str] = []

    for match in SLACK_LINK_RE.finditer(text):
        url = clean_extracted_url(match.group(1))
        if url and url not in seen:
            seen.add(url)
            urls.append(url)

    for match in PLAIN_URL_RE.finditer(text):
        url = clean_extracted_url(match.group(0))
        if url and url not in seen:
            seen.add(url)
            urls.append(url)

    return urls


def domain_from_url(url: str) -> str:
    parsed = urlparse(url)
    return (parsed.netloc or "").lower()


def is_privateish_host(host: str) -> bool:
    host = host.lower()
    return (
        host in {"localhost", "127.0.0.1", "0.0.0.0"}
        or host.endswith(".local")
        or host.endswith(".internal")
        or host.endswith(".lan")
    )


def classify_url(url: str) -> tuple[str, dict[str, Any]]:
    parsed = urlparse(url)
    host = (parsed.netloc or "").lower()
    path = parsed.path or ""
    parts = [part for part in path.split("/") if part]

    if host.endswith("slack.com") and len(parts) >= 3 and parts[0] == "archives":
        metadata: dict[str, Any] = {"channel_id": parts[1]}
        permalink_token = parts[2]
        if permalink_token.startswith("p") and permalink_token[1:].isdigit():
            digits = permalink_token[1:]
            if len(digits) > 6:
                metadata["message_ts"] = f"{digits[:-6]}.{digits[-6:]}"
        return "slack_message", metadata

    if host == "trello.com":
        if len(parts) >= 2 and parts[0] == "c":
            return "trello_card", {"short_link": parts[1]}
        if len(parts) >= 2 and parts[0] == "b":
            return "trello_board", {"short_link": parts[1]}
        if len(parts) >= 2 and parts[0] == "l":
            return "trello_list", {"short_link": parts[1]}
        return "trello_resource", {}

    if host == "docs.google.com":
        if len(parts) >= 1 and parts[0] == "spreadsheets":
            return "google_sheet", {}
        if len(parts) >= 1 and parts[0] == "document":
            return "google_doc", {}
        if len(parts) >= 1 and parts[0] == "presentation":
            return "google_slides", {}
        return "google_docs_resource", {}

    if host == "drive.google.com":
        return "google_drive", {}

    if host.endswith("salesforce.com") or host.endswith("force.com"):
        return "salesforce", {}

    if host == "github.com":
        if len(parts) >= 4 and parts[2] == "issues" and parts[3].isdigit():
            return "github_issue", {"owner": parts[0], "repo": parts[1], "number": parts[3]}
        if len(parts) >= 4 and parts[2] == "pull" and parts[3].isdigit():
            return "github_pr", {"owner": parts[0], "repo": parts[1], "number": parts[3]}
        if len(parts) >= 2:
            return "github_repo", {"owner": parts[0], "repo": parts[1]}
        return "github_resource", {}

    if is_privateish_host(host):
        return "private_resource", {}

    return "public_web", {}


def parse_html_preview(html_text: str) -> tuple[str, str]:
    title_match = TITLE_RE.search(html_text)
    title = ""
    if title_match:
        title = html.unescape(re.sub(r"\s+", " ", title_match.group(1))).strip()

    desc_match = META_DESC_RE.search(html_text)
    description = ""
    if desc_match:
        description = html.unescape(re.sub(r"\s+", " ", desc_match.group(1))).strip()

    return title, description


def format_links_for_prompt(links: list[MessageLink]) -> str:
    if not links:
        return "Sin URLs detectadas."

    chunks = []
    for link in links:
        chunks.append(
            "\n".join(
                [
                    f"- Tipo: {link.url_type}",
                    f"  URL: {link.url}",
                    f"  Estado: {link.status}",
                    f"  Interpretación: {link.summary or link.title or 'Sin interpretación adicional.'}",
                ]
            )
        )
    return "\n".join(chunks)
