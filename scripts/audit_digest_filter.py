"""Audit the FP rate of the digest filter.

Loads the fixture, runs is_digest(), takes a random sample of discarded
posts and prompts for manual labeling (y = true digest, n = incorrectly discarded).
Saves the result to tests/fixtures/digest_audit.json.

Usage:
    python scripts/audit_digest_filter.py --sample 50
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

FIXTURE = Path("tests/fixtures/news_sample.json")
AUDIT_OUT = Path("tests/fixtures/digest_audit.json")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fixture", default=str(FIXTURE))
    ap.add_argument("--out", default=str(AUDIT_OUT))
    ap.add_argument("--sample", type=int, default=50, help="Number of posts to label")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    from dedup.text import is_digest

    posts = json.loads(Path(args.fixture).read_text(encoding="utf-8"))
    total = len(posts)
    filtered = [p for p in posts if is_digest(p.get("text", ""))]
    kept = total - len(filtered)

    print(f"Total posts: {total}")
    print(f"Discarded by filter: {len(filtered)} ({100*len(filtered)/total:.1f}%)")
    print(f"Kept: {kept}\n")

    if not filtered:
        print("Nothing was discarded — no audit needed.")
        return

    rng = random.Random(args.seed)
    sample = rng.sample(filtered, min(args.sample, len(filtered)))
    print(f"Labeling {len(sample)} randomly sampled discarded posts.")
    print("y = true digest (filter is correct), n = false positive (FP)\n")
    print("-" * 60)

    results = []
    fp_count = 0
    for i, post in enumerate(sample, 1):
        text = (post.get("text") or "").strip()
        preview = text[:300].replace("\n", " ")
        print(f"\n[{i}/{len(sample)}] id={post.get('id')}  source={post.get('source')}")
        print(f"  {preview}{'...' if len(text) > 300 else ''}")
        while True:
            ans = input("  Digest? [y/n/q]: ").strip().lower()
            if ans in ("y", "n", "q"):
                break
            print("  Enter y, n, or q (quit)")
        if ans == "q":
            print("Aborted.")
            break
        is_true_digest = ans == "y"
        if not is_true_digest:
            fp_count += 1
        results.append({
            "id": post.get("id"),
            "source": post.get("source"),
            "is_true_digest": is_true_digest,
            "text_preview": text[:200],
        })

    if results:
        fp_rate = fp_count / len(results)
        print(f"\n--- Result ---")
        print(f"Labeled: {len(results)}  FP: {fp_count}  FP rate: {fp_rate:.1%}")
        Path(args.out).write_text(
            json.dumps(
                {"fp_count": fp_count, "labeled": len(results), "fp_rate": fp_rate, "items": results},
                ensure_ascii=False, indent=2,
            ),
            encoding="utf-8",
        )
        print(f"Saved to {args.out}")
        print(f"\nAdd to README: digest filter FP rate: {fp_count} of {len(results)} ({fp_rate:.1%})")


if __name__ == "__main__":
    main()
