"""Export the news table from SQLite to a JSON fixture for reproducible verification.

news.db is not tracked in git (gitignored) and depends on when it was
collected — running ingest again later would produce different news.
The fixture freezes a specific dataset so an employer can run
`python -m dedup.pipeline --input <fixture>` and get exactly the same
clusters.json that we did.

Schema matches tests/fixtures/news_sample.json:
    [{id, title, text, source, published_at}, ...]

Usage:
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
        # ORDER BY id -> deterministic order regardless of insertion.
        sql = "SELECT id, title, text, source, published_at FROM news ORDER BY id"
        if limit:
            sql += f" LIMIT {int(limit)}"
        rows = conn.execute(sql).fetchall()
    finally:
        conn.close()

    cols = ["id", "title", "text", "source", "published_at"]
    data = [dict(zip(cols, r)) for r in rows]

    # Canonical JSON (sort_keys) -> stable SHA for identical data.
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
