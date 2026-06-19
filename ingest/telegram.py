"""Сбор постов из публичных telegram-каналов через Telethon (MTProto).

Инкрементальный: на канал хранится last_msg_id, докачиваются только новые посты.
Маппинг сообщения -> единая схема (id, title, text, source, published_at).

Запуск:  python -m ingest.telegram --config config.yaml
Креды берутся из .env (TG_API_ID, TG_API_HASH, TG_SESSION).
"""
from __future__ import annotations

import argparse
import os

import yaml
from dotenv import load_dotenv
from telethon.sync import TelegramClient
from telethon.tl.types import Message

from .store import (
    NewsItem,
    connect,
    count_news,
    get_last_msg_id,
    set_last_msg_id,
    upsert_news,
)


def _split_title_text(raw: str) -> tuple[str, str]:
    """Заголовок = первая непустая строка поста, text = весь пост."""
    text = (raw or "").strip()
    first_line = next((ln.strip() for ln in text.splitlines() if ln.strip()), "")
    return first_line[:300], text


def _norm_source(channel: str) -> str:
    name = channel.strip()
    if name.startswith("https://t.me/"):
        name = name.rsplit("/", 1)[-1]
    return name if name.startswith("@") else f"@{name}"


def _message_to_item(msg: Message, source: str) -> NewsItem | None:
    raw = msg.message  # текст поста; None у чисто-медийных постов
    if not raw or not raw.strip():
        return None
    title, text = _split_title_text(raw)
    return NewsItem(
        id=f"{source}:{msg.id}",
        title=title,
        text=text,
        source=source,
        published_at=msg.date,  # Telethon отдаёт aware UTC datetime
    )


def collect(config: dict) -> int:
    load_dotenv()
    api_id = os.environ["TG_API_ID"]
    api_hash = os.environ["TG_API_HASH"]
    session = os.environ.get("TG_SESSION", "news_scraper")

    cfg = config["ingest"]
    db_path = cfg["db_path"]
    history_limit = int(cfg.get("history_limit", 500))

    total = 0
    with connect(db_path) as conn, TelegramClient(session, int(api_id), api_hash) as client:
        for channel in cfg["channels"]:
            source = _norm_source(channel)
            last_id = get_last_msg_id(conn, source)
            items: list[NewsItem] = []
            max_seen = last_id

            # min_id=last_id -> только сообщения новее чекпоинта (инкремент).
            # limit ограничивает первый (полный) проход.
            for msg in client.iter_messages(
                channel, limit=history_limit, min_id=last_id
            ):
                max_seen = max(max_seen, msg.id)
                item = _message_to_item(msg, source)
                if item is not None:
                    items.append(item)

            n = upsert_news(conn, items)
            if max_seen > last_id:
                set_last_msg_id(conn, source, max_seen)
            total += n
            print(f"[{source}] +{n} постов (last_msg_id -> {max_seen})")

        print(f"Всего новостей в БД: {count_news(conn)}")
    return total


def main() -> None:
    ap = argparse.ArgumentParser(description="Сбор новостей из telegram-каналов")
    ap.add_argument("--config", default="config.yaml")
    args = ap.parse_args()
    with open(args.config, encoding="utf-8") as f:
        config = yaml.safe_load(f)
    collect(config)


if __name__ == "__main__":
    main()
