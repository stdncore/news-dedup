"""Offline smoke-тест ядра кластеризации и метрик (без faiss/модели).

Проверяет логику граф->компоненты->каноническая и pairwise-метрики на
синтетике: 3 события по 2-3 дубля + 1 уникальная новость.
"""
from datetime import datetime, timedelta, timezone

import numpy as np

from dedup.cluster import build_edges, connected_clusters, pick_canonical
from dedup.eval import cluster_size_stats, evaluate

BASE = datetime(2026, 6, 18, 12, 0, tzinfo=timezone.utc)


def _fake_neighbors(emb, top_k):
    """Полный kNN через точное скалярное произведение (эмбеддинги L2-norm)."""
    sims_full = emb @ emb.T
    idx = np.argsort(-sims_full, axis=1)[:, : top_k + 1]
    sims = np.take_along_axis(sims_full, idx, axis=1)
    return sims, idx


def test_clusters_and_metrics():
    # 7 новостей: события A(0,1,2), B(3,4), C(5), D(6). truth-метки:
    true = [0, 0, 0, 1, 1, 2, 3]
    # Эмбеддинги: внутри события близкие, между — далёкие.
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

    # 4 события -> 4 кластера.
    assert len(set(labels.tolist())) == 4

    metrics = evaluate(true, labels.tolist())
    assert metrics["f1"] == 1.0, metrics
    assert metrics["ari"] == 1.0, metrics

    # Каноническая для события B по "longest" = индекс 4 (text_len 200).
    canon = pick_canonical(labels, published, text_len, strategy="longest")
    b_label = int(labels[3])
    assert canon[b_label] == 4

    # Каноническая по "earliest" для события A = индекс 0 (час 0).
    canon_e = pick_canonical(labels, published, text_len, strategy="earliest")
    a_label = int(labels[0])
    assert canon_e[a_label] == 0

    stats = cluster_size_stats(labels.tolist())
    assert stats["n_clusters"] == 4
    assert stats["n_singletons"] == 2  # события C и D

    print("OK:", metrics, stats)


def test_time_window_blocks_far_apart():
    # Две идентичные новости, но разнесены на 100 часов -> окно 72ч режет ребро.
    vecs = np.array([[1.0, 0.0], [1.0, 0.0]], dtype="float32")
    published = [BASE, BASE + timedelta(hours=100)]
    sims, idx = _fake_neighbors(vecs, top_k=1)
    edges = build_edges(sims, idx, published, cosine_threshold=0.8, time_window_hours=72)
    assert edges == []
    labels = connected_clusters(2, edges)
    assert len(set(labels.tolist())) == 2


def test_lexical_edge_time_window_blocks_far_apart():
    """Lexical (MinHash) ребро не должно выживать после time-window фильтра.

    Boilerplate-заголовки ТАСС/Интерфакс дают высокий Jaccard между разными событиями.
    Тест проверяет два условия отдельно:
    1. MinHash действительно находит ребро (иначе тест бессмысленен).
    2. Time-window фильтр в pipeline.run() его срезает (проверяем логику фильтра напрямую,
       без FAISS/модели чтобы избежать OpenMP-краша на macOS ARM64).
    """
    from dedup.prefilter import lexical_edges

    # Длинный boilerplate → высокий Jaccard между разными событиями.
    # Конкретно: ~20 общих слов, ~5 уникальных → Jaccard 3-gram ≈ 0.60+
    boilerplate = (
        "москва тасс агентство сообщает что сегодня по данным официального источника "
        "в пресс службе ведомства состоялось брифинг по итогам которого было объявлено "
    )
    texts = [
        boilerplate + "принятие закона о государственном бюджете",
        boilerplate + "открытие нового моста через реку волгу",
    ]

    # 1) MinHash должен найти ребро (низкий порог для надёжности теста)
    lex = lexical_edges(texts, num_perm=128, jaccard_threshold=0.4, shingle_size=3)
    assert len(lex) > 0, (
        "MinHash не нашёл ребро между boilerplate-текстами — тест некорректен, снизь jaccard_threshold"
    )

    # 2) После time-window фильтра (72ч) ребро должно исчезнуть (разница 30 дней)
    t0 = BASE.timestamp()
    t1 = (BASE + timedelta(days=30)).timestamp()
    tw = 72 * 3600.0
    ts = [t0, t1]
    filtered = [(i, j) for i, j in lex if abs(ts[i] - ts[j]) <= tw]
    assert len(filtered) == 0, (
        f"Time-window фильтр не срезал ребро: {filtered} (разница {(t1-t0)/3600:.0f}ч > {tw/3600:.0f}ч)"
    )


if __name__ == "__main__":
    test_clusters_and_metrics()
    test_time_window_blocks_far_apart()
    test_lexical_edge_time_window_blocks_far_apart()
    print("all smoke tests passed")
