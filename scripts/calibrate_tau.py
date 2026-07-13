"""Calibrate the cosine threshold tau on labeled pairs.

Algorithm:
  1. Load the fixture + labeled_pairs.
  2. Compute embeddings once (cached in embeddings_cache.npy).
  3. Build ANN neighbors once.
  4. For each tau in the grid: build edges -> clusters -> pairwise F1 against labels.
  5. Print a table and recommend the best tau.

Usage:
    python scripts/calibrate_tau.py
    python scripts/calibrate_tau.py --taus 0.75 0.80 0.85 0.90 0.95
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import yaml

FIXTURE = Path("tests/fixtures/news_sample.json")
LABELS_FILE = Path("tests/fixtures/labeled_pairs.json")
EMB_CACHE = Path("embeddings_cache.npy")
CONFIG = Path("config.yaml")

DEFAULT_TAUS = [0.70, 0.75, 0.80, 0.83, 0.85, 0.87, 0.90, 0.92, 0.95]


def load_fixture_filtered(path: Path) -> list[dict]:
    from dedup.text import is_digest

    posts = json.loads(path.read_text(encoding="utf-8"))
    return [p for p in posts if not is_digest(p.get("text", ""))]


def get_embeddings(posts: list[dict], cfg: dict) -> np.ndarray:
    if EMB_CACHE.exists():
        emb = np.load(EMB_CACHE)
        if emb.shape[0] == len(posts):
            print(f"Loaded embeddings from cache {EMB_CACHE} ({emb.shape})")
            return emb
        print("Cache is stale (size mismatch), recomputing...")

    from dedup.embed import build_embeddings

    dcfg = cfg["dedup"]
    titles = [p.get("title", "") for p in posts]
    texts = [p.get("text", "") for p in posts]
    emb = build_embeddings(
        titles,
        texts,
        model_name=dcfg["model"],
        max_chars=dcfg.get("max_chars", 1500),
        batch_size=dcfg.get("batch_size", 64),
        query_prefix=dcfg.get("query_prefix", ""),
        seed=dcfg.get("seed", 42),
    )
    np.save(EMB_CACHE, emb)
    print(f"Embeddings saved to {EMB_CACHE} ({emb.shape})")
    return emb


def eval_tau(
    tau: float,
    sims: np.ndarray,
    idx: np.ndarray,
    published: list,
    time_window_hours: float,
    id2pos: dict[str, int],
    labeled: list[dict],
) -> dict[str, float]:
    from dedup.cluster import build_edges, connected_clusters

    edges = build_edges(sims, idx, published, cosine_threshold=tau, time_window_hours=time_window_hours)
    n = sims.shape[0]
    labels = connected_clusters(n, edges)

    tp = fp = fn = 0
    for pair in labeled:
        i = id2pos.get(pair["id1"])
        j = id2pos.get(pair["id2"])
        if i is None or j is None:
            continue
        same_pred = labels[i] == labels[j]
        same_true = pair["label"] == 1
        if same_pred and same_true:
            tp += 1
        elif same_pred and not same_true:
            fp += 1
        elif not same_pred and same_true:
            fn += 1

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    n_clusters = len(set(labels.tolist()))
    return {
        "tau": tau,
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "tp": tp, "fp": fp, "fn": fn,
        "n_clusters": n_clusters,
        "n_edges": len(edges),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--taus", nargs="+", type=float, default=DEFAULT_TAUS)
    ap.add_argument("--config", default=str(CONFIG))
    ap.add_argument("--fixture", default=str(FIXTURE))
    ap.add_argument("--labels", default=str(LABELS_FILE))
    ap.add_argument("--test-ratio", type=float, default=0.3,
                    help="Fraction of pairs held out for testing (0 = no split, all pairs go to train)")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    dcfg = cfg["dedup"]

    posts = load_fixture_filtered(Path(args.fixture))
    print(f"Posts after filtering: {len(posts)}")

    labeled = json.loads(Path(args.labels).read_text(encoding="utf-8"))
    n_dup = sum(r["label"] for r in labeled)
    print(f"Labeled pairs: {len(labeled)}  (duplicates: {n_dup}, non-duplicates: {len(labeled)-n_dup})")

    if args.test_ratio > 0:
        import random
        rng = random.Random(dcfg.get("seed", 42))
        shuffled = labeled[:]
        rng.shuffle(shuffled)
        split = int(len(shuffled) * (1 - args.test_ratio))
        train_pairs = shuffled[:split]
        test_pairs = shuffled[split:]
        print(f"Train pairs: {len(train_pairs)}, Test pairs: {len(test_pairs)}")
    else:
        train_pairs = labeled
        test_pairs = None
        print("Split disabled (--test-ratio 0): all pairs go to train")

    id2pos = {p["id"]: i for i, p in enumerate(posts)}

    from datetime import datetime, timezone

    def parse_dt(s):
        if not s:
            return datetime(2000, 1, 1, tzinfo=timezone.utc)
        return datetime.fromisoformat(s.replace("Z", "+00:00"))

    published = [parse_dt(p.get("published_at")) for p in posts]

    emb = get_embeddings(posts, cfg)

    from dedup.cluster import ann_neighbors

    ann = dcfg.get("ann", {})
    sims, idx = ann_neighbors(
        emb,
        top_k=ann.get("top_k", 20),
        hnsw_m=ann.get("hnsw_m", 32),
        ef_search=ann.get("ef_search", 64),
    )
    print(f"ANN ready. Tau grid: {args.taus}\n")

    tw = dcfg.get("time_window_hours", 72)

    # Grid search over train_pairs
    results = []
    for tau in args.taus:
        r = eval_tau(tau, sims, idx, published, tw, id2pos, train_pairs)
        results.append(r)

    header = f"{'tau':>6}  {'F1':>6}  {'P':>6}  {'R':>6}  {'TP':>4}  {'FP':>4}  {'FN':>4}  {'clusters':>10}  {'edges':>7}"
    print(f"\n--- TRAIN ({len(train_pairs)} pairs) ---")
    print(header)
    print("-" * 72)
    best = max(results, key=lambda r: r["f1"])
    for r in results:
        marker = " <-- best" if r["tau"] == best["tau"] else ""
        print(
            f"{r['tau']:>6.2f}  {r['f1']:>6.3f}  {r['precision']:>6.3f}  {r['recall']:>6.3f}"
            f"  {r['tp']:>4}  {r['fp']:>4}  {r['fn']:>4}  {r['n_clusters']:>10}  {r['n_edges']:>7}{marker}"
        )

    print(f"\nBest tau on train = {best['tau']}  (F1={best['f1']:.3f})")

    if test_pairs:
        final = eval_tau(best["tau"], sims, idx, published, tw, id2pos, test_pairs)
        print(f"\n--- HELD-OUT TEST ({len(test_pairs)} pairs) ---")
        print(header)
        print("-" * 72)
        print(
            f"{final['tau']:>6.2f}  {final['f1']:>6.3f}  {final['precision']:>6.3f}  {final['recall']:>6.3f}"
            f"  {final['tp']:>4}  {final['fp']:>4}  {final['fn']:>4}  {final['n_clusters']:>10}  {final['n_edges']:>7}"
        )
        print(f"\nFinal (held-out) F1 = {final['f1']:.3f}  P={final['precision']:.3f}  R={final['recall']:.3f}")

    print(f"\nUpdate config.yaml: cosine_threshold: {best['tau']}")


if __name__ == "__main__":
    main()
