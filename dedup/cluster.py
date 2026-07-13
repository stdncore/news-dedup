"""Clustering: FAISS ANN -> edges by cosine threshold + time window ->
connected components (union-find) -> pick the canonical news item.

Scales to 100k+: ANN neighbor search instead of an O(n^2) matrix.
"""
from __future__ import annotations

import os

# macOS: multiple libomp copies (faiss + torch + sklearn) -> segfault on
# OpenMP fork inside faiss.search. Allow it before importing faiss.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from datetime import datetime

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components


def ann_neighbors(
    emb: np.ndarray,
    top_k: int = 20,
    nlist: int = 256,
    nprobe: int = 32,
) -> tuple[np.ndarray, np.ndarray]:
    """Top-k cosine neighbors (inner product on L2-normalized vectors).

    Up to 50k — IndexFlatIP (exact, stable on macOS).
    Above 50k — IndexIVFFlat: for a one-off static index it builds faster
    and uses less RAM than HNSW (HNSW only pays off with many repeated
    queries, whereas here it's a single batch pass).
    Returns (sims, idx): both (N, top_k+1), including the point itself.
    """
    import faiss

    # macOS: faiss libomp and torch libomp conflict when faiss.search spawns
    # a parallel OpenMP team -> segfault. Forcing faiss to one thread avoids
    # spawning a team (work runs in the calling thread), so no crash. Search
    # is slower, but IVFFlat with nprobe still isn't a full scan.
    faiss.omp_set_num_threads(1)

    n, dim = emb.shape
    k = min(top_k + 1, n)

    if n <= 50_000:
        index = faiss.IndexFlatIP(dim)
        index.add(emb)
    else:
        quantizer = faiss.IndexFlatIP(dim)
        # nlist must not exceed the number of points; for stable training
        # FAISS wants at least ~39*nlist training vectors.
        nlist = min(nlist, max(1, n // 39))
        index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT)
        index.train(emb)
        index.add(emb)
        index.nprobe = nprobe

    sims, idx = index.search(emb, k)
    return sims, idx


def build_edges(
    sims: np.ndarray,
    idx: np.ndarray,
    published_at: list[datetime],
    cosine_threshold: float,
    time_window_hours: float,
) -> list[tuple[int, int]]:
    """Edges between news items: cosine >= threshold AND time difference <= window."""
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


def build_weighted_edges(
    sims: np.ndarray,
    idx: np.ndarray,
    published_at: list[datetime],
    cosine_threshold: float,
    time_window_hours: float,
) -> dict[tuple[int, int], float]:
    """Like build_edges, but returns {(i,j): max_cosine} for Louvain.

    Edge weight = maximum cosine between i and j (symmetrizing the ANN output).
    """
    window = time_window_hours * 3600.0
    ts = np.array([d.timestamp() for d in published_at])
    weights: dict[tuple[int, int], float] = {}
    n = sims.shape[0]
    for i in range(n):
        for sim, j in zip(sims[i], idx[i]):
            j = int(j)
            if j == i or j < 0 or sim < cosine_threshold:
                continue
            if abs(ts[i] - ts[j]) > window:
                continue
            a, b = (i, j) if i < j else (j, i)
            if sim > weights.get((a, b), 0.0):
                weights[(a, b)] = float(sim)
    return weights


def connected_clusters(n: int, edges: list[tuple[int, int]]) -> np.ndarray:
    """Cluster labels via connected components of the edge graph."""
    if edges:
        rows, cols = zip(*edges)
        data = np.ones(len(edges))
        graph = coo_matrix((data, (rows, cols)), shape=(n, n))
    else:
        graph = coo_matrix((n, n))
    _, labels = connected_components(graph, directed=False)
    return labels


def louvain_clusters(
    n: int,
    weighted_edges: dict[tuple[int, int], float],
    resolution: float = 1.0,
    seed: int = 42,
) -> np.ndarray:
    """Cluster labels via Louvain communities on the weighted graph.

    Unlike connected components, Louvain cuts weak "bridges" between dense
    groups. This removes transitive over-merging: a hot news topic (drone
    attack, war) would otherwise merge adjacent events into a mega-cluster
    through a chain of similar posts. Edge weight = cosine (connection
    strength); modularity is optimized so that weak inter-event bridges end
    up at the cut boundary. resolution>1 splits more aggressively.
    """
    import networkx as nx
    from networkx.algorithms.community import louvain_communities

    g = nx.Graph()
    g.add_nodes_from(range(n))
    for (i, j), w in weighted_edges.items():
        g.add_edge(i, j, weight=float(w))
    comms = louvain_communities(g, weight="weight", resolution=resolution, seed=seed)
    labels = np.zeros(n, dtype=int)
    for cid, members in enumerate(comms):
        for m in members:
            labels[m] = cid
    return labels


def pick_canonical(
    labels: np.ndarray,
    published_at: list[datetime],
    text_len: list[int],
    strategy: str = "earliest",
) -> dict[int, int]:
    """For each cluster_id, pick the index of the canonical news item."""
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
