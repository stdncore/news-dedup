"""Оркестрация дедупликации: SQLite -> пре-фильтр -> эмбеддинги -> ANN ->
граф -> кластеры -> каноническая -> JSON.

Запуск:  python -m dedup.pipeline --config config.yaml
"""
from __future__ import annotations

import argparse
import json
import re
from datetime import datetime

import numpy as np
import pandas as pd
import yaml

from .cluster import (
    ann_neighbors,
    build_edges,
    connected_clusters,
    pick_canonical,
)
from .embed import build_embeddings
from .prefilter import lexical_edges


def load_news(db_path: str) -> pd.DataFrame:
    import sqlite3

    conn = sqlite3.connect(db_path)
    try:
        df = pd.read_sql_query(
            "SELECT id, title, text, source, published_at FROM news ORDER BY published_at",
            conn,
        )
    finally:
        conn.close()
    df["published_at"] = pd.to_datetime(df["published_at"], utc=True)
    return df.reset_index(drop=True)


_DIGEST_MARKERS = re.compile(
    r"главные новости|кратко:|дайджест|итоги дня|важное за|что случилось"
    r"|больше новостей к этому часу|прямо сейчас в эфире|в программе «|слушайте в эфире",
    flags=re.IGNORECASE,
)
_BULLET_RE = re.compile(r"[▪•▸►\-🟢🔴🔵🟡]\s?")


def _is_digest(text: str) -> bool:
    """True если пост — дайджест/сводка нескольких событий."""
    if _DIGEST_MARKERS.search(text):
        return True
    if len(_BULLET_RE.findall(text)) >= 3 and len(text) > 250:
        return True
    return False


def load_fixture(path: str) -> pd.DataFrame:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    df = pd.DataFrame(data)
    df["published_at"] = pd.to_datetime(df["published_at"], utc=True)
    df["title"] = df["title"].fillna("")
    df["text"] = df["text"].fillna("")
    before = len(df)
    df = df[~df["text"].apply(_is_digest)].reset_index(drop=True)
    print(f"Фильтр дайджестов: убрано {before - len(df)} постов, осталось {len(df)}")
    return df.sort_values("published_at").reset_index(drop=True)


def run(config: dict, df: pd.DataFrame | None = None, input_file: str | None = None) -> pd.DataFrame:
    cfg = config["dedup"]
    if df is None:
        if input_file:
            df = load_fixture(input_file)
        else:
            df = load_news(cfg["db_path"])
    n = len(df)
    if n == 0:
        raise SystemExit("В БД нет новостей — сначала запусти ingest.")

    titles = df["title"].tolist()
    texts = df["text"].tolist()
    published = [d.to_pydatetime() for d in df["published_at"]]
    text_len = df["text"].str.len().tolist()

    # 1) Лексический пре-фильтр -> достоверные рёбра.
    pf = cfg.get("prefilter", {})
    lex = []
    if pf.get("enabled", True):
        lex = lexical_edges(
            [f"{t} {x}" for t, x in zip(titles, texts)],
            num_perm=pf.get("num_perm", 256),
            jaccard_threshold=pf.get("jaccard_threshold", 0.7),
            shingle_size=pf.get("shingle_size", 5),
        )
        print(f"Лексический пре-фильтр: {len(lex)} рёбер")

    # 2) Эмбеддинги.
    emb = build_embeddings(
        titles,
        texts,
        model_name=cfg["model"],
        max_chars=cfg.get("max_chars", 1500),
        batch_size=cfg.get("batch_size", 64),
        query_prefix=cfg.get("query_prefix", ""),
        seed=cfg.get("seed", 42),
    )

    # 3) ANN + 4/5) рёбра по порогу косинуса и окну времени.
    ann = cfg.get("ann", {})
    sims, idx = ann_neighbors(
        emb,
        top_k=ann.get("top_k", 20),
        hnsw_m=ann.get("hnsw_m", 32),
        ef_search=ann.get("ef_search", 64),
    )
    sem = build_edges(
        sims,
        idx,
        published,
        cosine_threshold=cfg.get("cosine_threshold", 0.85),
        time_window_hours=cfg.get("time_window_hours", 72),
    )
    print(f"Семантические рёбра: {len(sem)}")

    # 6) Компоненты связности.
    edges = sorted(set(lex) | set(sem))
    labels = connected_clusters(n, edges)
    df = df.copy()
    df["cluster_id"] = labels

    # 7) Каноническая новость на кластер.
    canon = pick_canonical(
        labels, published, text_len, strategy=cfg.get("canonical", "earliest")
    )
    df["is_canonical"] = [canon[int(l)] == i for i, l in enumerate(labels)]

    n_clusters = len(set(labels.tolist()))
    print(f"Новостей: {n}  кластеров: {n_clusters}  дублей убрано: {n - n_clusters}")

    # 8) Вывод.
    _write_output(df, cfg.get("output", "clusters.json"))
    return df


def _write_output(df: pd.DataFrame, path: str) -> None:
    groups: dict[str, list[str]] = {}
    for cid, gid in zip(df["cluster_id"], df["id"]):
        groups.setdefault(str(int(cid)), []).append(gid)
    payload = {
        "clusters": groups,
        "per_news": df[["id", "cluster_id", "is_canonical"]].to_dict("records"),
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, default=str)
    print(f"Результат записан в {path}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Дедупликация новостей")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--input", default=None, help="JSON fixture вместо SQLite")
    args = ap.parse_args()
    with open(args.config, encoding="utf-8") as f:
        config = yaml.safe_load(f)
    run(config, input_file=args.input)


if __name__ == "__main__":
    main()
