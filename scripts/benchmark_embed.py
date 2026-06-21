"""Бенчмарк throughput эмбеддингов — gate перед прогоном на 100k.

Измеряет docs/sec на реальной модели и текущем CPU, экстраполирует на 100k.
Решение: >=10 docs/sec -> 100k за <3ч, идём как есть; <10 -> разовый прогон
на CPU не жизнеспособен, нужен квант/меньшая модель (с рекалибровкой tau).

Запуск:
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
    ap = argparse.ArgumentParser(description="Бенчмарк throughput эмбеддингов")
    ap.add_argument("--n", type=int, default=5000, help="сколько документов гонять")
    ap.add_argument("--batch-size", type=int, default=None, help="override batch_size")
    ap.add_argument("--fixture", default=None, help="путь к JSON-фикстуре (по умолч. news_sample.json)")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path("config.yaml").read_text(encoding="utf-8"))["dedup"]
    batch_size = args.batch_size or cfg.get("batch_size", 256)

    fixture_path = Path(args.fixture) if args.fixture else FIXTURE
    posts = json.loads(fixture_path.read_text(encoding="utf-8"))
    titles_all = [p.get("title", "") for p in posts]
    texts_all = [p.get("text", "") for p in posts]
    base = len(posts)

    # Циклируем уникальные тексты до N — throughput per-doc, состав не важен.
    idx = [i % base for i in range(args.n)]
    titles = [titles_all[i] for i in idx]
    texts = [texts_all[i] for i in idx]

    print(f"Модель: {cfg['model']}  N={args.n}  batch_size={batch_size}")
    print("Прогрев (cold start включает загрузку модели)...")

    t0 = time.perf_counter()
    build_embeddings(
        titles, texts,
        model_name=cfg["model"],
        max_chars=cfg.get("max_chars", 1500),
        batch_size=batch_size,
        query_prefix=cfg.get("query_prefix", ""),
        seed=cfg.get("seed", 42),
        # без кэша — чистое измерение
    )
    dt = time.perf_counter() - t0

    rate = args.n / dt
    eta_100k_min = 100_000 / rate / 60
    print(f"\n=== РЕЗУЛЬТАТ ===")
    print(f"Время (вкл. загрузку модели): {dt:.1f} c")
    print(f"Throughput: {rate:.1f} docs/sec")
    print(f"ETA 100k: {eta_100k_min:.1f} мин ({eta_100k_min/60:.1f} ч)")
    verdict = "OK — идём как есть" if rate >= 10 else "СТОП — нужен квант/меньшая модель + рекалибровка tau"
    print(f"Gate (>=10 docs/sec): {verdict}")


if __name__ == "__main__":
    main()
