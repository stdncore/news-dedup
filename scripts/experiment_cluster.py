"""Quick clustering experiment WITHOUT recomputing embeddings.

Embeddings are pulled from the cache by id; only the ANN -> edges ->
clustering -> evaluation against labeled_pairs stages are run. Lets you
compare connected components vs Louvain in minutes instead of hours.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from dedup.cluster import ann_neighbors, build_edges, connected_clusters


def load_state():
    cfg = yaml.safe_load(Path("config.yaml").read_text(encoding="utf-8"))["dedup"]
    result = json.loads(Path("clusters.json").read_text(encoding="utf-8"))
    news = {p["id"]: p for p in json.loads(Path("tests/fixtures/news_100k.json").read_text(encoding="utf-8"))}

    # id order matches the run exactly (per_news preserves df order).
    ids = [r["id"] for r in result["per_news"]]
    cache = np.load(cfg["embeddings_cache"], allow_pickle=True)
    vec = {str(k): v for k, v in zip(cache["ids"], cache["vectors"])}

    ids = [i for i in ids if i in vec]
    emb = np.vstack([vec[i] for i in ids]).astype("float32")
    published = pd.to_datetime([news[i]["published_at"] for i in ids], utc=True)
    published = [d.to_pydatetime() for d in published]
    return cfg, ids, emb, published


def eval_pairs(ids, labels):
    pairs = json.loads(Path("tests/fixtures/labeled_pairs.json").read_text(encoding="utf-8"))
    pos = {i: l for i, l in zip(ids, labels)}
    tp = fp = fn = tn = 0
    for p in pairs:
        if p["id1"] not in pos or p["id2"] not in pos:
            continue
        gold = int(p["label"])
        same = pos[p["id1"]] == pos[p["id2"]]
        if gold and same: tp += 1
        elif not gold and same: fp += 1
        elif gold and not same: fn += 1
        else: tn += 1
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return dict(tp=tp, fp=fp, fn=fn, tn=tn, precision=prec, recall=rec, f1=f1)


def structure(labels):
    labels = np.asarray(labels)
    _, counts = np.unique(labels, return_counts=True)
    return dict(
        n_clusters=int(counts.size),
        max_cluster=int(counts.max()),
        singletons=int((counts == 1).sum()),
        dedup_rate=float(1 - counts.size / labels.size),
    )


def louvain_clusters(n, edges, sims_lookup, resolution=1.0, seed=42):
    """Louvain communities on a weighted graph (weight = cosine similarity)."""
    import networkx as nx
    from networkx.algorithms.community import louvain_communities

    g = nx.Graph()
    g.add_nodes_from(range(n))
    for (i, j), w in edges.items():
        g.add_edge(i, j, weight=float(w))
    comms = louvain_communities(g, weight="weight", resolution=resolution, seed=seed)
    labels = np.zeros(n, dtype=int)
    for cid, members in enumerate(comms):
        for m in members:
            labels[m] = cid
    return labels


def main():
    t0 = time.perf_counter()
    cfg, ids, emb, published = load_state()
    n = len(ids)
    print(f"Loaded {n} posts from cache in {time.perf_counter()-t0:.1f}s")

    ann = cfg.get("ann", {})
    sims, idx = ann_neighbors(emb, top_k=ann.get("top_k", 50),
                              nlist=ann.get("nlist", 256), nprobe=ann.get("nprobe", 32))
    print(f"FAISS ready in {time.perf_counter()-t0:.1f}s")

    tau = cfg.get("cosine_threshold", 0.75)
    tw = cfg.get("time_window_hours", 72)

    # Edges as a list + a weight dict (for Louvain).
    edge_list = build_edges(sims, idx, published, cosine_threshold=tau, time_window_hours=tw)
    window = tw * 3600.0
    ts = np.array([d.timestamp() for d in published])
    eweight = {}
    for i in range(n):
        for sim, j in zip(sims[i], idx[i]):
            j = int(j)
            if j == i or j < 0 or sim < tau:
                continue
            if abs(ts[i] - ts[j]) > window:
                continue
            a, b = min(i, j), max(i, j)
            eweight[(a, b)] = max(eweight.get((a, b), 0.0), float(sim))
    print(f"Edges: {len(edge_list)}")

    # Baseline: connected components.
    cc = connected_clusters(n, edge_list)
    print("\n=== connected components (baseline) ===")
    print("  structure:", structure(cc))
    print("  pairwise: ", eval_pairs(ids, cc))

    # Louvain across several resolution values.
    for res in (0.5, 1.0, 1.5, 2.0):
        lv = louvain_clusters(n, eweight, sims, resolution=res)
        print(f"\n=== Louvain resolution={res} ===")
        print("  structure:", structure(lv))
        print("  pairwise: ", eval_pairs(ids, lv))

    print(f"\nTotal {time.perf_counter()-t0:.1f}s")


if __name__ == "__main__":
    main()
