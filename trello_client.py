from __future__ import annotations

from dataclasses import dataclass
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

    def get_card(self, card_id: str) -> TrelloCardState:
        payload = self._request(
            "GET",
            f"/cards/{card_id}",
            params={
                "fields": "name,url,idList,closed",
                "list": "true",
                "list_fields": "name",
            },
        )
        card_list = payload.get("list") or {}
        return TrelloCardState(
            id=payload["id"],
            name=payload.get("name") or "",
            url=payload.get("url") or "",
            list_id=payload.get("idList") or "",
            list_name=card_list.get("name") or "",
            closed=bool(payload.get("closed")),
        )

    def add_card_comment(self, card_id: str, text: str) -> None:
        self._request(
            "POST",
            f"/cards/{card_id}/actions/comments",
            json_body={"text": text},
        )


def build_trello_token_url(api_key: str) -> str:
    return (
        "https://trello.com/1/authorize"
        f"?expiration=never&scope=read,write&response_type=token&key={api_key}"
    )
