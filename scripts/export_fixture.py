"""Экспорт таблицы news из SQLite в JSON-фикстуру для воспроизводимой проверки.

news.db в git не попадает (gitignore) и зависит от момента сбора — запуск
ingest позже даст другие новости. Фикстура замораживает конкретный набор,
чтобы работодатель прогнал `python -m dedup.pipeline --input <fixture>` и
получил ровно тот же clusters.json, что и мы.

Схема совпадает с tests/fixtures/news_sample.json:
    [{id, title, text, source, published_at}, ...]

Запуск:
    python scripts/export_fixture.py --db news.db \
        --out tests/fixtures/news_100k.json --limit 100000
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3


def export(db_path: str, out_path: str, limit: int | None) -> None:
    conn = sqlite3.connect(db_path)
    try:
        # ORDER BY id -> детерминированный порядок независимо от вставки.
        sql = "SELECT id, title, text, source, published_at FROM news ORDER BY id"
        if limit:
            sql += f" LIMIT {int(limit)}"
        rows = conn.execute(sql).fetchall()
    finally:
        conn.close()

    cols = ["id", "title", "text", "source", "published_at"]
    data = [dict(zip(cols, r)) for r in rows]

    # Канонический JSON (sort_keys) -> стабильный SHA при одинаковых данных.
    payload = json.dumps(data, ensure_ascii=False, sort_keys=True, indent=2)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(payload)

    sha = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    sha_path = f"{out_path}.sha256"
    with open(sha_path, "w", encoding="utf-8") as f:
        f.write(f"{sha}  {out_path}\n")

    print(f"Экспортировано {len(data)} новостей -> {out_path}")
    print(f"SHA-256: {sha}")
    print(f"Чек-сумма записана в {sha_path}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Экспорт news.db в JSON-фикстуру")
    ap.add_argument("--db", default="news.db")
    ap.add_argument("--out", default="tests/fixtures/news_100k.json")
    ap.add_argument("--limit", type=int, default=None, help="макс. число новостей")
    args = ap.parse_args()
    export(args.db, args.out, args.limit)


if __name__ == "__main__":
    main()
