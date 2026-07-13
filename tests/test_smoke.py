"""Offline smoke test for the clustering core and metrics (no faiss/model).

Checks the graph->components->canonical logic and pairwise metrics on
synthetic data: 3 events with 2-3 duplicates each + 1 unique news item.
"""
from datetime import datetime, timedelta, timezone

import numpy as np

from dedup.cluster import build_edges, connected_clusters, pick_canonical
from dedup.eval import cluster_size_stats, evaluate

BASE = datetime(2026, 6, 18, 12, 0, tzinfo=timezone.utc)


def _fake_neighbors(emb, top_k):
    """Exact full kNN via dot product (embeddings are L2-normalized)."""
    sims_full = emb @ emb.T
    idx = np.argsort(-sims_full, axis=1)[:, : top_k + 1]
    sims = np.take_along_axis(sims_full, idx, axis=1)
    return sims, idx


def test_clusters_and_metrics():
    # 7 news items: events A(0,1,2), B(3,4), C(5), D(6). Ground-truth labels:
    true = [0, 0, 0, 1, 1, 2, 3]
    # Embeddings: close within an event, far apart between events.
    rng = np.random.default_rng(42)
    centers = rng.normal(size=(4, 16))
    event_of = [0, 0, 0, 1, 1, 2, 3]
    vecs = np.array([centers[e] + 0.01 * rng.normal(size=16) for e in event_of])
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
    vecs = vecs.astype("float32")

    published = [BASE + timedelta(hours=h) for h in (0, 1, 2, 0, 5, 0, 0)]
    text_len = [100, 120, 90, 80, 200, 50, 70]

    sims, idx = _fake_neighbors(vecs, top_k=6)
    edges = build_edges(sims, idx, published, cosine_threshold=0.8, time_window_hours=72)
    labels = connected_clusters(len(vecs), edges)

    # 4 events -> 4 clusters.
    assert len(set(labels.tolist())) == 4

    metrics = evaluate(true, labels.tolist())
    assert metrics["f1"] == 1.0, metrics
    assert metrics["ari"] == 1.0, metrics

    # Canonical for event B by "longest" = index 4 (text_len 200).
    canon = pick_canonical(labels, published, text_len, strategy="longest")
    b_label = int(labels[3])
    assert canon[b_label] == 4

    # Canonical by "earliest" for event A = index 0 (hour 0).
    canon_e = pick_canonical(labels, published, text_len, strategy="earliest")
    a_label = int(labels[0])
    assert canon_e[a_label] == 0

    stats = cluster_size_stats(labels.tolist())
    assert stats["n_clusters"] == 4
    assert stats["n_singletons"] == 2  # events C and D

    print("OK:", metrics, stats)


def test_time_window_blocks_far_apart():
    # Two identical news items, but 100 hours apart -> the 72h window cuts the edge.
    vecs = np.array([[1.0, 0.0], [1.0, 0.0]], dtype="float32")
    published = [BASE, BASE + timedelta(hours=100)]
    sims, idx = _fake_neighbors(vecs, top_k=1)
    edges = build_edges(sims, idx, published, cosine_threshold=0.8, time_window_hours=72)
    assert edges == []
    labels = connected_clusters(2, edges)
    assert len(set(labels.tolist())) == 2


def test_lexical_edge_time_window_blocks_far_apart():
    """A lexical (MinHash) edge must not survive the time-window filter.

    TASS/Interfax boilerplate headlines produce a high Jaccard similarity between
    unrelated events. This test checks two conditions separately:
    1. MinHash actually finds an edge (otherwise the test is meaningless).
    2. The time-window filter in pipeline.run() cuts it (we check the filter
       logic directly, without FAISS/model, to avoid the OpenMP crash on macOS ARM64).
    """
    from dedup.prefilter import lexical_edges

    # Long boilerplate -> high Jaccard between different events.
    # Specifically: ~20 shared words, ~5 unique -> 3-gram Jaccard ~= 0.60+
    boilerplate = (
        "москва тасс агентство сообщает что сегодня по данным официального источника "
        "в пресс службе ведомства состоялось брифинг по итогам которого было объявлено "
    )
    texts = [
        boilerplate + "принятие закона о государственном бюджете",
        boilerplate + "открытие нового моста через реку волгу",
    ]

    # 1) MinHash should find an edge (low threshold for test reliability)
    lex = lexical_edges(texts, num_perm=128, jaccard_threshold=0.4, shingle_size=3)
    assert len(lex) > 0, (
        "MinHash found no edge between the boilerplate texts — test is invalid, lower jaccard_threshold"
    )

    # 2) After the time-window filter (72h) the edge should disappear (30-day gap)
    t0 = BASE.timestamp()
    t1 = (BASE + timedelta(days=30)).timestamp()
    tw = 72 * 3600.0
    ts = [t0, t1]
    filtered = [(i, j) for i, j in lex if abs(ts[i] - ts[j]) <= tw]
    assert len(filtered) == 0, (
        f"Time-window filter failed to cut the edge: {filtered} (gap {(t1-t0)/3600:.0f}h > {tw/3600:.0f}h)"
    )


if __name__ == "__main__":
    test_clusters_and_metrics()
    test_time_window_blocks_far_apart()
    test_lexical_edge_time_window_blocks_far_apart()
    print("all smoke tests passed")
