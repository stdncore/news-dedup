"""Оценка качества кластеризации.

Две ситуации:
  * есть ground-truth метки событий -> pairwise P/R/F1 + ARI;
  * нет разметки -> структурные sanity-метрики (распределение размеров).
"""
from __future__ import annotations

from itertools import combinations

import numpy as np


def pairwise_prf(true_labels: list[int], pred_labels: list[int]) -> dict[str, float]:
    """Pairwise precision/recall/F1: считаем пары в одном кластере.

    TP — пара в одном кластере и в truth, и в предсказании.
    """
    def same_cluster_pairs(labels):
        groups: dict[int, list[int]] = {}
        for i, l in enumerate(labels):
            groups.setdefault(l, []).append(i)
        pairs = set()
        for members in groups.values():
            for a, b in combinations(sorted(members), 2):
                pairs.add((a, b))
        return pairs

    t = same_cluster_pairs(true_labels)
    p = same_cluster_pairs(pred_labels)
    tp = len(t & p)
    precision = tp / len(p) if p else 0.0
    recall = tp / len(t) if t else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def adjusted_rand(true_labels: list[int], pred_labels: list[int]) -> float:
    from sklearn.metrics import adjusted_rand_score

    return float(adjusted_rand_score(true_labels, pred_labels))


def evaluate(true_labels: list[int], pred_labels: list[int]) -> dict[str, float]:
    out = pairwise_prf(true_labels, pred_labels)
    out["ari"] = adjusted_rand(true_labels, pred_labels)
    return out


def cluster_size_stats(pred_labels: list[int]) -> dict[str, float]:
    """Без разметки: структурные метрики качества."""
    labels = np.asarray(pred_labels)
    _, counts = np.unique(labels, return_counts=True)
    return {
        "n_items": int(labels.size),
        "n_clusters": int(counts.size),
        "n_singletons": int((counts == 1).sum()),
        "max_cluster": int(counts.max()) if counts.size else 0,
        "mean_cluster": float(counts.mean()) if counts.size else 0.0,
        "dedup_rate": float(1 - counts.size / labels.size) if labels.size else 0.0,
    }
