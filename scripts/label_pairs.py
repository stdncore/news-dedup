"""Interactive labeling of news pairs for eval.

Shows candidate pairs (close in time, different sources),
you decide: same event or not.

Controls:
  1  — duplicate (same event)
  0  — not a duplicate
  s  — skip
  q  — quit, save

Saves to tests/fixtures/labeled_pairs.json:
  [{"id1": ..., "id2": ..., "label": 1}, ...]

Run:
    python scripts/label_pairs.py
    python scripts/label_pairs.py --limit 80  # how many pairs to label
"""
from __future__ import annotations

import argparse
import json
import re
import random
import textwrap
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path

FIXTURE = Path("tests/fixtures/news_sample.json")
OUT = Path("tests/fixtures/labeled_pairs.json")

_DIGEST_MARKERS = re.compile(
    r"главные новости|кратко:|дайджест|итоги дня|важное за|что случилось"
    r"|больше новостей к этому часу|прямо сейчас в эфире|в программе «|слушайте в эфире",
    flags=re.IGNORECASE,
)
_BULLET_RE = re.compile(r"[▪•▸►\-🟢🔴🔵🟡]\s?")


def _is_digest(text: str) -> bool:
    if _DIGEST_MARKERS.search(text):
        return True
    if len(_BULLET_RE.findall(text)) >= 3 and len(text) > 250:
        return True
    return False
WINDOW_HOURS = 6  # show pairs within this window


def parse_dt(s: str | None) -> datetime:
    if not s:
        return datetime(2000, 1, 1, tzinfo=timezone.utc)
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def candidate_pairs(posts: list[dict], window_h: int, max_pairs: int) -> list[tuple[dict, dict]]:
    """Pairs: different sources + published within window_h hours of each other."""
    pairs = []
    for a, b in combinations(posts, 2):
        if a["source"] == b["source"]:
            continue
        dt_a = parse_dt(a["published_at"])
        dt_b = parse_dt(b["published_at"])
        diff = abs((dt_a - dt_b).total_seconds()) / 3600
        if diff <= window_h:
            pairs.append((a, b, diff))
    # sort by time difference (closer in time — more likely duplicates)
    pairs.sort(key=lambda x: x[2])
    # shuffle a bit so they're not all from the same chunk
    top = pairs[: max_pairs * 3]
    random.shuffle(top)
    return [(a, b) for a, b, _ in top[:max_pairs]]


_EMOJI_RE = re.compile(
    "[\U00010000-\U0010ffff"
    "\U0001f300-\U0001f9ff"
    "☀-⛿✀-➿"
    "■-◿⬀-⯿"
    "▪▸►▶⚡🔵🟢🟡🔴🟠🎙📁✔️]+",
    flags=re.UNICODE,
)


def _clean(text: str) -> str:
    return _EMOJI_RE.sub("", text).strip()


def show_pair(idx: int, total: int, a: dict, b: dict) -> None:
    sep = "─" * 60
    print(f"\n{sep}")
    print(f"Pair {idx}/{total}")
    print(sep)
    print(f"[A] {a['source']}  {a['published_at']}")
    print(textwrap.fill(_clean(a["text"])[:400], width=70))
    print()
    print(f"[B] {b['source']}  {b['published_at']}")
    print(textwrap.fill(_clean(b["text"])[:400], width=70))
    print(sep)
    print("1=duplicate  0=not a duplicate  s=skip  q=quit")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fixture", default=str(FIXTURE))
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--limit", type=int, default=100)
    ap.add_argument("--window", type=int, default=WINDOW_HOURS)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)

    posts = json.load(open(args.fixture, encoding="utf-8"))
    before = len(posts)
    posts = [p for p in posts if not _is_digest(p.get("text", ""))]
    print(f"Loaded {before} posts, after digest filter: {len(posts)}")

    pairs = candidate_pairs(posts, args.window, args.limit)
    print(f"Candidates to label: {len(pairs)} (window {args.window}h, different sources)")

    out_path = Path(args.out)
    # load already-labeled pairs if the file exists
    labeled: list[dict] = []
    if out_path.exists():
        labeled = json.load(open(out_path, encoding="utf-8"))
        done_ids = {(r["id1"], r["id2"]) for r in labeled}
        pairs = [(a, b) for a, b in pairs if (a["id"], b["id"]) not in done_ids]
        print(f"Already labeled: {len(labeled)}, candidates remaining: {len(pairs)}")

    try:
        for i, (a, b) in enumerate(pairs, 1):
            show_pair(i, len(pairs), a, b)
            while True:
                ch = input("> ").strip().lower()
                if ch == "q":
                    raise KeyboardInterrupt
                if ch == "s":
                    break
                if ch in ("0", "1"):
                    labeled.append({
                        "id1": a["id"],
                        "id2": b["id"],
                        "label": int(ch),
                        "source1": a["source"],
                        "source2": b["source"],
                    })
                    break
                print("Enter 1, 0, s or q")
    except KeyboardInterrupt:
        pass

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(labeled, f, ensure_ascii=False, indent=2)

    n_dup = sum(r["label"] for r in labeled)
    print(f"\nSaved {len(labeled)} pairs → {out_path}")
    print(f"Duplicates: {n_dup}  Non-duplicates: {len(labeled) - n_dup}")


if __name__ == "__main__":
    main()
