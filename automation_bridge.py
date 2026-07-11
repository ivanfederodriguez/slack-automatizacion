from __future__ import annotations

import json
import sqlite3
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any


TASK_AUTOMATION_COLUMNS: dict[str, str] = {
    "automation_status": "automation_status TEXT NOT NULL DEFAULT 'pending'",
    "automation_owner": "automation_owner TEXT",
    "automation_lease_until": "automation_lease_until TEXT",
    "automation_attempts": "automation_attempts INTEGER NOT NULL DEFAULT 0",
    "automation_started_at": "automation_started_at TEXT",
    "automation_finished_at": "automation_finished_at TEXT",
    "automation_run_id": "automation_run_id INTEGER",
    "automation_result_text": "automation_result_text TEXT",
    "automation_artifacts_json": "automation_artifacts_json TEXT",
    "automation_error": "automation_error TEXT",
    "clarification_question": "clarification_question TEXT",
    "clarification_answer": "clarification_answer TEXT",
    "clarification_requested_at": "clarification_requested_at TEXT",
    "clarification_answered_at": "clarification_answered_at TEXT",
}


def now_iso() -> str:
    return datetime.now(UTC).isoformat()


@dataclass(frozen=True, slots=True)
class OutboxDelivery:
    id: int
    event_key: str
    task_id: int
    event_type: str
    payload: dict[str, Any]
    channel_id: str
    thread_ts: str
    message_ts: str

    @property
    def client_msg_id(self) -> str:
        return str(uuid.uuid5(uuid.NAMESPACE_URL, self.event_key))


class AutomationBridge:
    def __init__(self, db_path: str | Path, *, worker_id: str | None = None) -> None:
        self.db_path = Path(db_path)
        self.worker_id = worker_id or f"slack-{uuid.uuid4()}"

    def connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path, timeout=10)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA busy_timeout = 10000")
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    def initialize(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as connection:
            connection.execute("PRAGMA journal_mode = WAL")
            task_columns = {
                str(row[1]) for row in connection.execute("PRAGMA table_info(tasks)").fetchall()
            }
            if not task_columns:
                return
            for name, definition in TASK_AUTOMATION_COLUMNS.items():
                if name not in task_columns:
                    connection.execute(f"ALTER TABLE tasks ADD COLUMN {definition}")
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS automation_outbox (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_key TEXT NOT NULL UNIQUE,
                    task_id INTEGER NOT NULL,
                    producer TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    routing_channel TEXT NOT NULL DEFAULT 'slack',
                    payload_json TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'pending',
                    attempts INTEGER NOT NULL DEFAULT 0,
                    available_at TEXT NOT NULL,
                    locked_at TEXT,
                    locked_by TEXT,
                    sent_at TEXT,
                    external_id TEXT,
                    last_error TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    FOREIGN KEY(task_id) REFERENCES tasks(id)
                );
                CREATE INDEX IF NOT EXISTS idx_tasks_automation_pending
                ON tasks(category, automation_status, automation_lease_until, id);
                CREATE INDEX IF NOT EXISTS idx_automation_outbox_pending
                ON automation_outbox(routing_channel, status, available_at, id);
                """
            )

    def claim_next(self) -> OutboxDelivery | None:
        self.initialize()
        now = now_iso()
        stale = (datetime.now(UTC) - timedelta(minutes=15)).isoformat()
        with self.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                """
                SELECT automation_outbox.*, tasks.channel_id, tasks.thread_ts, tasks.message_ts
                FROM automation_outbox
                JOIN tasks ON tasks.id = automation_outbox.task_id
                WHERE automation_outbox.routing_channel = 'slack'
                  AND automation_outbox.event_type IN (
                    'salesforce_completed', 'salesforce_summary_ready'
                  )
                  AND automation_outbox.available_at <= ?
                  AND (
                    automation_outbox.status = 'pending'
                    OR (
                      automation_outbox.status = 'processing'
                      AND coalesce(automation_outbox.locked_at, '') < ?
                    )
                  )
                ORDER BY automation_outbox.id ASC
                LIMIT 1
                """,
                (now, stale),
            ).fetchone()
            if row is None:
                connection.rollback()
                return None
            connection.execute(
                """
                UPDATE automation_outbox
                SET status = 'processing', locked_at = ?, locked_by = ?,
                    attempts = attempts + 1, updated_at = ?
                WHERE id = ?
                """,
                (now, self.worker_id, now, row["id"]),
            )
            connection.commit()
        return OutboxDelivery(
            id=int(row["id"]),
            event_key=str(row["event_key"]),
            task_id=int(row["task_id"]),
            event_type=str(row["event_type"]),
            payload=json.loads(row["payload_json"] or "{}"),
            channel_id=str(row["channel_id"]),
            thread_ts=str(row["thread_ts"] or row["message_ts"]),
            message_ts=str(row["message_ts"]),
        )

    def mark_sent(self, delivery: OutboxDelivery, external_id: str) -> None:
        now = now_iso()
        with self.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute(
                """
                UPDATE automation_outbox
                SET status = 'sent', sent_at = ?, external_id = ?, last_error = NULL,
                    locked_at = NULL, locked_by = NULL, updated_at = ?
                WHERE id = ? AND locked_by = ?
                """,
                (now, external_id, now, delivery.id, self.worker_id),
            )
            if delivery.event_type == "salesforce_completed":
                connection.execute(
                    """
                    UPDATE tasks
                    SET status = 'responded', reply_sent_at = ?, reply_ts = ?,
                        reply_error = NULL, updated_at = ?
                    WHERE id = ?
                    """,
                    (now, external_id, now, delivery.task_id),
                )
            connection.commit()

    def mark_failed(self, delivery: OutboxDelivery, error: str) -> None:
        now = datetime.now(UTC)
        with self.connect() as connection:
            row = connection.execute(
                "SELECT attempts FROM automation_outbox WHERE id = ?", (delivery.id,)
            ).fetchone()
            attempts = int(row[0]) if row else 1
            terminal = attempts >= 5
            delay = min(30, 2**min(attempts, 4))
            connection.execute(
                """
                UPDATE automation_outbox
                SET status = ?, available_at = ?, last_error = ?,
                    locked_at = NULL, locked_by = NULL, updated_at = ?
                WHERE id = ?
                """,
                (
                    "failed" if terminal else "pending",
                    (now + timedelta(minutes=delay)).isoformat(),
                    error[:4000],
                    now.isoformat(),
                    delivery.id,
                ),
            )
            connection.execute(
                "UPDATE tasks SET reply_error = ?, updated_at = ? WHERE id = ?",
                (error[:1000], now.isoformat(), delivery.task_id),
            )
