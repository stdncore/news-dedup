"""Калибровка порога косинуса tau на размеченных парах.

Алгоритм:
  1. Загружает fixture + labeled_pairs.
  2. Вычисляет эмбеддинги один раз (кешируется в embeddings_cache.npy).
  3. Строит ANN-соседей один раз.
  4. Для каждого tau из сетки: строит рёбра -> кластеры -> pairwise F1 по меткам.
  5. Выводит таблицу и рекомендует лучший tau.

Запуск:
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
    import re

    _DIGEST_MARKERS = re.compile(
        r"главные новости|кратко:|дайджест|итоги дня|важное за|что случилось"
        r"|больше новостей к этому часу|прямо сейчас в эфире|в программе «|слушайте в эфире",
        flags=re.IGNORECASE,
    )
    _BULLET_RE = re.compile(r"[▪•▸►\-🟢🔴🔵🟡]\s?")

    def _is_digest(text: str) -> bool:
        if _DIGEST_MARKERS.search(text):
            return True
        if len(_BULLET_RE.findall(text)) >= 3 and len(text) > 250:
            return True
        return False

    posts = json.loads(path.read_text(encoding="utf-8"))
    return [p for p in posts if not _is_digest(p.get("text", ""))]


def get_embeddings(posts: list[dict], cfg: dict) -> np.ndarray:
    if EMB_CACHE.exists():
        emb = np.load(EMB_CACHE)
        if emb.shape[0] == len(posts):
            print(f"Загружены эмбеддинги из кеша {EMB_CACHE} ({emb.shape})")
            return emb
        print("Кеш устарел (размер не совпадает), пересчитываем...")

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
    print(f"Эмбеддинги сохранены в {EMB_CACHE} ({emb.shape})")
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
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    dcfg = cfg["dedup"]

    posts = load_fixture_filtered(Path(args.fixture))
    print(f"Постов после фильтра: {len(posts)}")

    labeled = json.loads(Path(args.labels).read_text(encoding="utf-8"))
    n_dup = sum(r["label"] for r in labeled)
    print(f"Размеченных пар: {len(labeled)}  (дублей: {n_dup}, не-дублей: {len(labeled)-n_dup})")

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
    print(f"ANN готов. Сетка tau: {args.taus}\n")

    results = []
    for tau in args.taus:
        r = eval_tau(tau, sims, idx, published, dcfg.get("time_window_hours", 72), id2pos, labeled)
        results.append(r)

    print(f"{'tau':>6}  {'F1':>6}  {'P':>6}  {'R':>6}  {'TP':>4}  {'FP':>4}  {'FN':>4}  {'кластеров':>10}  {'рёбер':>7}")
    print("-" * 72)
    best = max(results, key=lambda r: r["f1"])
    for r in results:
        marker = " <-- best" if r["tau"] == best["tau"] else ""
        print(
            f"{r['tau']:>6.2f}  {r['f1']:>6.3f}  {r['precision']:>6.3f}  {r['recall']:>6.3f}"
            f"  {r['tp']:>4}  {r['fp']:>4}  {r['fn']:>4}  {r['n_clusters']:>10}  {r['n_edges']:>7}{marker}"
        )

    print(f"\nЛучший tau = {best['tau']}  (F1={best['f1']:.3f})")
    print(f"Обнови config.yaml: cosine_threshold: {best['tau']}")


if __name__ == "__main__":
    main()
