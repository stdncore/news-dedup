"""Embedding throughput benchmark — gate before running on 100k.

Measures docs/sec on the real model and current CPU, extrapolates to 100k.
Decision rule: >=10 docs/sec -> 100k in <3h, proceed as-is; <10 -> a single
run on CPU is not viable, need a quantized/smaller model (with tau recalibration).

Usage:
    python scripts/benchmark_embed.py --n 5000 --batch-size 256
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import yaml

from dedup.embed import build_embeddings

FIXTURE = Path("tests/fixtures/news_sample.json")


def main() -> None:
    ap = argparse.ArgumentParser(description="Embedding throughput benchmark")
    ap.add_argument("--n", type=int, default=5000, help="how many documents to run")
    ap.add_argument("--batch-size", type=int, default=None, help="override batch_size")
    ap.add_argument("--fixture", default=None, help="path to JSON fixture (default: news_sample.json)")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path("config.yaml").read_text(encoding="utf-8"))["dedup"]
    batch_size = args.batch_size or cfg.get("batch_size", 256)

    fixture_path = Path(args.fixture) if args.fixture else FIXTURE
    posts = json.loads(fixture_path.read_text(encoding="utf-8"))
    titles_all = [p.get("title", "") for p in posts]
    texts_all = [p.get("text", "") for p in posts]
    base = len(posts)

    # Cycle unique texts up to N — throughput per-doc, content doesn't matter.
    idx = [i % base for i in range(args.n)]
    titles = [titles_all[i] for i in idx]
    texts = [texts_all[i] for i in idx]

    print(f"Model: {cfg['model']}  N={args.n}  batch_size={batch_size}")
    print("Warming up (cold start includes model loading)...")

    t0 = time.perf_counter()
    build_embeddings(
        titles, texts,
        model_name=cfg["model"],
        max_chars=cfg.get("max_chars", 1500),
        batch_size=batch_size,
        query_prefix=cfg.get("query_prefix", ""),
        seed=cfg.get("seed", 42),
        # no cache — pure measurement
    )
    dt = time.perf_counter() - t0

    rate = args.n / dt
    eta_100k_min = 100_000 / rate / 60
    print(f"\n=== RESULT ===")
    print(f"Time (incl. model loading): {dt:.1f} s")
    print(f"Throughput: {rate:.1f} docs/sec")
    print(f"ETA 100k: {eta_100k_min:.1f} min ({eta_100k_min/60:.1f} h)")
    verdict = "OK — proceed as-is" if rate >= 10 else "STOP — need a quantized/smaller model + tau recalibration"
    print(f"Gate (>=10 docs/sec): {verdict}")


if __name__ == "__main__":
    main()
