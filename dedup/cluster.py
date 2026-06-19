"""Кластеризация: FAISS ANN -> рёбра по порогу косинуса + окну времени ->
компоненты связности (union-find) -> выбор канонической новости.

Масштабируется на 100k+: ANN-поиск соседей вместо O(n^2) матрицы.
"""
from __future__ import annotations

from datetime import datetime

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components


def ann_neighbors(
    emb: np.ndarray,
    top_k: int = 20,
    hnsw_m: int = 32,
    ef_search: int = 64,
) -> tuple[np.ndarray, np.ndarray]:
    """top-k соседей по косинусу (inner product на L2-norm векторах).

    До 50k — IndexFlatIP (точный, стабильный на macOS).
    Свыше 50k — HNSW для масштаба.
    Возвращает (sims, idx): обе (N, top_k+1), включая саму точку.
    """
    import faiss

    n, dim = emb.shape
    k = min(top_k + 1, n)

    if n <= 50_000:
        index = faiss.IndexFlatIP(dim)
    else:
        index = faiss.IndexHNSWFlat(dim, hnsw_m, faiss.METRIC_INNER_PRODUCT)
        index.hnsw.efSearch = ef_search

    index.add(emb)
    sims, idx = index.search(emb, k)
    return sims, idx


def build_edges(
    sims: np.ndarray,
    idx: np.ndarray,
    published_at: list[datetime],
    cosine_threshold: float,
    time_window_hours: float,
) -> list[tuple[int, int]]:
    """Рёбра между новостями: косинус >= порог И разница времени <= окна."""
    window = time_window_hours * 3600.0
    ts = np.array([d.timestamp() for d in published_at])
    edges: set[tuple[int, int]] = set()
    n = sims.shape[0]
    for i in range(n):
        for sim, j in zip(sims[i], idx[i]):
            if j == i or j < 0:
                continue
            if sim < cosine_threshold:
                continue
            if abs(ts[i] - ts[j]) > window:
                continue
            edges.add((min(i, int(j)), max(i, int(j))))
    return sorted(edges)


def connected_clusters(n: int, edges: list[tuple[int, int]]) -> np.ndarray:
    """Метки кластеров через компоненты связности графа рёбер."""
    if edges:
        rows, cols = zip(*edges)
        data = np.ones(len(edges))
        graph = coo_matrix((data, (rows, cols)), shape=(n, n))
    else:
        graph = coo_matrix((n, n))
    _, labels = connected_components(graph, directed=False)
    return labels


def pick_canonical(
    labels: np.ndarray,
    published_at: list[datetime],
    text_len: list[int],
    strategy: str = "earliest",
) -> dict[int, int]:
    """Для каждого cluster_id выбрать индекс канонической новости."""
    best: dict[int, int] = {}
    for i, lab in enumerate(labels):
        lab = int(lab)
        if lab not in best:
            best[lab] = i
            continue
        cur = best[lab]
        if strategy == "longest":
            if text_len[i] > text_len[cur]:
                best[lab] = i
        else:  # earliest
            if published_at[i] < published_at[cur]:
                best[lab] = i
    return best
