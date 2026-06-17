from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import requests


TRELLO_API_BASE = "https://api.trello.com/1"


class TrelloError(RuntimeError):
    """Raised when Trello API operations fail."""


@dataclass(frozen=True)
class TrelloCard:
    id: str
    name: str
    url: str


@dataclass(frozen=True)
class TrelloCardState:
    id: str
    name: str
    url: str
    list_id: str
    list_name: str
    closed: bool
    due_complete: bool = False
    checklist_done: bool = False
    checklist_items: tuple[dict[str, str], ...] = ()


@dataclass(frozen=True)
class TrelloComment:
    id: str
    text: str
    date: str
    member_id: str = ""
    member_username: str = ""
    member_full_name: str = ""


@dataclass(frozen=True)
class TrelloList:
    id: str
    name: str
    board_id: str


class TrelloClient:
    def __init__(self, api_key: str, token: str, timeout: int = 30) -> None:
        self.api_key = api_key
        self.token = token
        self.timeout = timeout

    def _request(
        self,
        method: str,
        path: str,
        *,
        params: Optional[dict[str, Any]] = None,
        json_body: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        query = {
            "key": self.api_key,
            "token": self.token,
        }
        if params:
            query.update(params)

        response = requests.request(
            method,
            f"{TRELLO_API_BASE}{path}",
            params=query,
            json=json_body,
            timeout=self.timeout,
            headers={"Accept": "application/json"},
        )
        if response.status_code >= 400:
            raise TrelloError(f"Trello devolvió {response.status_code}: {response.text[:500]}")
        return response.json()

    def get_me(self) -> dict[str, Any]:
        return self._request("GET", "/members/me")

    def get_list(self, list_id: str) -> TrelloList:
        payload = self._request("GET", f"/lists/{list_id}")
        return TrelloList(
            id=payload["id"],
            name=payload["name"],
            board_id=payload["idBoard"],
        )

    def list_boards_with_lists(self) -> list[dict[str, Any]]:
        return self._request(
            "GET",
            "/members/me/boards",
            params={
                "fields": "name,url",
                "lists": "open",
                "list_fields": "name",
            },
        )

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
        body: dict[str, Any] = {
            "idList": list_id,
            "name": name,
            "desc": desc,
            "pos": pos,
        }
        if member_ids:
            body["idMembers"] = member_ids
        if label_ids:
            body["idLabels"] = label_ids

        payload = self._request("POST", "/cards", json_body=body)
        return TrelloCard(
            id=payload["id"],
            name=payload["name"],
            url=payload["url"],
        )

    def update_card(
        self,
        card_id: str,
        *,
        name: Optional[str] = None,
        desc: Optional[str] = None,
    ) -> TrelloCardState:
        body: dict[str, Any] = {}
        if name is not None:
            body["name"] = name
        if desc is not None:
            body["desc"] = desc
        if not body:
            return self.get_card(card_id)

        self._request("PUT", f"/cards/{card_id}", json_body=body)
        return self.get_card(card_id)

    def get_card(self, card_id: str) -> TrelloCardState:
        payload = self._request(
            "GET",
            f"/cards/{card_id}",
            params={
                "fields": "name,url,idList,closed,dueComplete",
                "list": "true",
                "list_fields": "name",
                "checklists": "all",
                "checkItem_fields": "name,state",
            },
        )
        card_list = payload.get("list") or {}
        checklist_items = []
        for checklist in payload.get("checklists") or []:
            for item in checklist.get("checkItems") or []:
                checklist_items.append(
                    {
                        "name": str(item.get("name") or ""),
                        "state": str(item.get("state") or ""),
                    }
                )
        return TrelloCardState(
            id=payload["id"],
            name=payload.get("name") or "",
            url=payload.get("url") or "",
            list_id=payload.get("idList") or "",
            list_name=card_list.get("name") or "",
            closed=bool(payload.get("closed")),
            due_complete=bool(payload.get("dueComplete")),
            checklist_items=tuple(checklist_items),
        )

    def add_card_comment(self, card_id: str, text: str) -> None:
        self._request(
            "POST",
            f"/cards/{card_id}/actions/comments",
            json_body={"text": text},
        )

    def get_card_comments(self, card_id: str, limit: int = 50) -> list[TrelloComment]:
        payload = self._request(
            "GET",
            f"/cards/{card_id}/actions",
            params={
                "filter": "commentCard",
                "limit": str(limit),
                "fields": "date,data,idMemberCreator",
                "memberCreator": "true",
                "memberCreator_fields": "username,fullName",
            },
        )
        comments = []
        for action in payload or []:
            data = action.get("data") or {}
            member = action.get("memberCreator") or {}
            comments.append(
                TrelloComment(
                    id=str(action.get("id") or ""),
                    text=str(data.get("text") or ""),
                    date=str(action.get("date") or ""),
                    member_id=str(action.get("idMemberCreator") or ""),
                    member_username=str(member.get("username") or ""),
                    member_full_name=str(member.get("fullName") or ""),
                )
            )
        return comments

    def attach_file_to_card(self, card_id: str, file_path: Path, name: Optional[str] = None) -> str:
        query = {
            "key": self.api_key,
            "token": self.token,
        }
        if name:
            query["name"] = name

        with file_path.open("rb") as handle:
            response = requests.post(
                f"{TRELLO_API_BASE}/cards/{card_id}/attachments",
                params=query,
                files={"file": (name or file_path.name, handle)},
                timeout=self.timeout,
                headers={"Accept": "application/json"},
            )
        if response.status_code >= 400:
            raise TrelloError(f"Trello devolvió {response.status_code}: {response.text[:500]}")
        payload = response.json()
        return str(payload.get("id") or "")

    def add_url_attachment_to_card(self, card_id: str, url: str, name: Optional[str] = None) -> str:
        body: dict[str, Any] = {"url": url}
        if name:
            body["name"] = name
        payload = self._request(
            "POST",
            f"/cards/{card_id}/attachments",
            json_body=body,
        )
        return str(payload.get("id") or "")


def build_trello_token_url(api_key: str) -> str:
    return (
        "https://trello.com/1/authorize"
        f"?expiration=never&scope=read,write&response_type=token&key={api_key}"
    )
