"""Scrapes public telegram channels via t.me/s/<channel> (no credentials needed).

Saves posts to tests/fixtures/news_sample.json in the dedup schema.

Usage:
    python scripts/scrape_fixture.py
    python scripts/scrape_fixture.py --channels meduzalive rian_ru bbcrussian --limit 150
"""
from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import requests
from bs4 import BeautifulSoup

DEFAULT_CHANNELS = ["tass_agency", "rbc_news", "interfax_russia", "kommersant", "vedomosti", "ria_novosti", "interfaxonline"]
OUT_PATH = Path("tests/fixtures/news_sample.json")
HEADERS = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"}


def scrape_channel(username: str, limit: int = 150) -> list[dict]:
    posts: list[dict] = []
    before: int | None = None
    username = username.lstrip("@")

    while len(posts) < limit:
        url = f"https://t.me/s/{username}"
        if before:
            url += f"?before={before}"

        try:
            r = requests.get(url, headers=HEADERS, timeout=10, verify=False)
            r.raise_for_status()
        except Exception as e:
            print(f"  [{username}] fetch error: {e}")
            break

        soup = BeautifulSoup(r.text, "html.parser")
        msgs = soup.select(".tgme_widget_message")
        if not msgs:
            break

        ids_this_page: list[int] = []
        for m in msgs:
            text_el = m.select_one(".tgme_widget_message_text")
            date_el = m.select_one(".tgme_widget_message_date time")
            link_el = m.select_one("a.tgme_widget_message_date")

            if not text_el or not link_el:
                continue

            raw = text_el.get_text(separator="\n").strip()
            if not raw:
                continue

            href = link_el.get("href", "")
            m_id = re.search(r"/(\d+)$", href)
            msg_id = int(m_id.group(1)) if m_id else 0
            ids_this_page.append(msg_id)

            lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
            title = lines[0][:300] if lines else ""

            posts.append({
                "id": f"@{username}:{msg_id}",
                "title": title,
                "text": raw,
                "source": f"@{username}",
                "published_at": date_el["datetime"] if date_el else None,
            })

        if not ids_this_page:
            break
        before = min(ids_this_page)
        time.sleep(0.5)  # polite delay

    return posts[:limit]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--channels", nargs="+", default=DEFAULT_CHANNELS)
    ap.add_argument("--limit", type=int, default=150, help="постов на канал")
    ap.add_argument("--out", default=str(OUT_PATH))
    args = ap.parse_args()

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    all_posts: list[dict] = []
    for ch in args.channels:
        print(f"Скрапинг @{ch.lstrip('@')} ...")
        posts = scrape_channel(ch, limit=args.limit)
        print(f"  → {len(posts)} постов")
        all_posts.extend(posts)

    out = Path(args.out)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(all_posts, f, ensure_ascii=False, indent=2)

    print(f"\nИтого: {len(all_posts)} постов → {out}")


if __name__ == "__main__":
    main()
