"""SQLite-хранилище собранных новостей.

Единая схema под задачу дедупликации: id, title, text, source, published_at.
PK по id обеспечивает идемпотентный upsert — повторный сбор не плодит дубли.
"""
from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Iterable, Iterator


SCHEMA = """
CREATE TABLE IF NOT EXISTS news (
    id           TEXT PRIMARY KEY,   -- "<source>:<msg_id>"
    title        TEXT NOT NULL,
    text         TEXT NOT NULL,
    source       TEXT NOT NULL,      -- @channel
    published_at TEXT NOT NULL       -- ISO-8601 UTC
);
CREATE INDEX IF NOT EXISTS idx_news_published ON news(published_at);

-- Чекпоинт инкрементального сбора: последний обработанный msg_id на канал.
CREATE TABLE IF NOT EXISTS ingest_state (
    source      TEXT PRIMARY KEY,
    last_msg_id INTEGER NOT NULL
);
"""


@dataclass
class NewsItem:
    id: str
    title: str
    text: str
    source: str
    published_at: datetime  # aware UTC

    def as_row(self) -> tuple[str, str, str, str, str]:
        dt = self.published_at.astimezone(timezone.utc)
        return (self.id, self.title, self.text, self.source, dt.isoformat())


@contextmanager
def connect(db_path: str) -> Iterator[sqlite3.Connection]:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        conn.executescript(SCHEMA)
        yield conn
        conn.commit()
    finally:
        conn.close()


def upsert_news(conn: sqlite3.Connection, items: Iterable[NewsItem]) -> int:
    rows = [it.as_row() for it in items]
    if not rows:
        return 0
    conn.executemany(
        """
        INSERT INTO news (id, title, text, source, published_at)
        VALUES (?, ?, ?, ?, ?)
        ON CONFLICT(id) DO UPDATE SET
            title=excluded.title,
            text=excluded.text,
            source=excluded.source,
            published_at=excluded.published_at
        """,
        rows,
    )
    return len(rows)


def get_last_msg_id(conn: sqlite3.Connection, source: str) -> int:
    cur = conn.execute(
        "SELECT last_msg_id FROM ingest_state WHERE source = ?", (source,)
    )
    row = cur.fetchone()
    return int(row["last_msg_id"]) if row else 0


def set_last_msg_id(conn: sqlite3.Connection, source: str, msg_id: int) -> None:
    conn.execute(
        """
        INSERT INTO ingest_state (source, last_msg_id) VALUES (?, ?)
        ON CONFLICT(source) DO UPDATE SET last_msg_id=excluded.last_msg_id
        """,
        (source, msg_id),
    )


def count_news(conn: sqlite3.Connection) -> int:
    return int(conn.execute("SELECT COUNT(*) FROM news").fetchone()[0])
