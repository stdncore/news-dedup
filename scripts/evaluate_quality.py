"""Оценка качества кластеризации по экспертной разметке пар.

Читает clusters.json (результат pipeline) и labeled_pairs.json,
вычисляет pairwise Precision / Recall / F1 + ARI + структурные метрики.

Запуск:
    python scripts/evaluate_quality.py
    python scripts/evaluate_quality.py --clusters clusters.json --pairs tests/fixtures/labeled_pairs.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from dedup.eval import cluster_size_stats, evaluate


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clusters", default="clusters.json")
    ap.add_argument("--pairs", default="tests/fixtures/labeled_pairs.json")
    args = ap.parse_args()

    result = json.loads(Path(args.clusters).read_text(encoding="utf-8"))
    pairs = json.loads(Path(args.pairs).read_text(encoding="utf-8"))

    # id -> cluster_id
    id_to_cluster: dict[str, int] = {
        r["id"]: int(r["cluster_id"]) for r in result["per_news"]
    }

    # Фильтруем пары где оба ID есть в результате
    valid_pairs = [p for p in pairs if p["id1"] in id_to_cluster and p["id2"] in id_to_cluster]
    missing = len(pairs) - len(valid_pairs)
    if missing:
        print(f"[warn] {missing} пар пропущено — ID не найдены в clusters.json")

    # Строим векторы меток для pairwise F1
    true_labels, pred_labels = [], []
    tp = fp = fn = tn = 0
    for p in valid_pairs:
        gold = int(p["label"])  # 1 = дубль, 0 = не дубль
        same = 1 if id_to_cluster[p["id1"]] == id_to_cluster[p["id2"]] else 0
        true_labels.append(gold)
        pred_labels.append(same)
        if gold == 1 and same == 1:
            tp += 1
        elif gold == 0 and same == 1:
            fp += 1
        elif gold == 1 and same == 0:
            fn += 1
        else:
            tn += 1

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    # Структурные метрики по всем кластерам
    all_labels = [int(r["cluster_id"]) for r in result["per_news"]]
    stats = cluster_size_stats(all_labels)

    print("=" * 50)
    print("КАЧЕСТВО ПО ЭКСПЕРТНОЙ РАЗМЕТКЕ")
    print("=" * 50)
    print(f"Размеченных пар:   {len(valid_pairs)}  (из {len(pairs)})")
    print(f"  TP={tp}  FP={fp}  FN={fn}  TN={tn}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1:        {f1:.4f}")
    print()
    print("СТРУКТУРНЫЕ МЕТРИКИ (100k)")
    print("=" * 50)
    print(f"  Новостей:          {stats['n_items']:>7}")
    print(f"  Кластеров:         {stats['n_clusters']:>7}")
    print(f"  Синглтонов:        {stats['n_singletons']:>7}")
    print(f"  Макс. кластер:     {stats['max_cluster']:>7}")
    print(f"  Ср. размер:        {stats['mean_cluster']:>7.2f}")
    print(f"  Dedup rate:        {stats['dedup_rate']:>7.1%}")
    print()

    # Итоговая строка для отчёта
    print(f"Pairwise F1 = {f1:.4f}  |  dedup rate = {stats['dedup_rate']:.1%}  |  кластеров = {stats['n_clusters']}")


if __name__ == "__main__":
    main()
