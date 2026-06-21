"""Оркестрация дедупликации: SQLite -> пре-фильтр -> эмбеддинги -> ANN ->
граф -> кластеры -> каноническая -> JSON.

Запуск:  python -m dedup.pipeline --config config.yaml
"""
from __future__ import annotations

import argparse
import json
import resource
import sys
from datetime import datetime

import numpy as np
import pandas as pd
import yaml

from .cluster import (
    ann_neighbors,
    build_weighted_edges,
    connected_clusters,
    louvain_clusters,
    pick_canonical,
)
from .embed import build_embeddings
from .prefilter import lexical_edges
from .text import is_digest


def _log_rss(phase: str) -> None:
    """Пиковая RSS после фазы. macOS отдаёт байты, Linux — килобайты."""
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    mb = rss / (1024 ** 2) if sys.platform == "darwin" else rss / 1024
    print(f"[mem] после {phase}: peak RSS {mb:.0f} MB")


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




def load_fixture(path: str) -> pd.DataFrame:
    # .gz поддержан: фикстура 100k в репозитории хранится сжатой (133MB -> 31MB).
    if path.endswith(".gz"):
        import gzip

        with gzip.open(path, "rt", encoding="utf-8") as f:
            data = json.load(f)
    else:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    df = pd.DataFrame(data)
    df["published_at"] = pd.to_datetime(df["published_at"], utc=True)
    df["title"] = df["title"].fillna("")
    df["text"] = df["text"].fillna("")
    before = len(df)
    df = df[~df["text"].apply(is_digest)].reset_index(drop=True)
    print(f"Фильтр дайджестов: убрано {before - len(df)} постов, осталось {len(df)}")
    # Фильтр постов где текст — только URL (без содержательного контента).
    import re as _re
    _url_only = _re.compile(r"^https?://\S+$")
    before = len(df)
    df = df[~df["text"].str.strip().apply(lambda t: bool(_url_only.match(t)))].reset_index(drop=True)
    print(f"Фильтр URL-only: убрано {before - len(df)} постов, осталось {len(df)}")
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

    ids = df["id"].astype(str).tolist()
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
        # Доля пар к n — сигнал blowup'а на boilerplate (агентские футеры).
        print(f"Лексический пре-фильтр: {len(lex)} рёбер ({len(lex) / max(n, 1):.2f}/новость)")

    # 2) Эмбеддинги (инкрементальный кэш по id переживает краш прогона).
    emb = build_embeddings(
        titles,
        texts,
        model_name=cfg["model"],
        max_chars=cfg.get("max_chars", 1500),
        batch_size=cfg.get("batch_size", 64),
        query_prefix=cfg.get("query_prefix", ""),
        seed=cfg.get("seed", 42),
        ids=ids,
        cache_path=cfg.get("embeddings_cache"),
    )
    _log_rss("эмбеддинги")

    # 3) ANN + 4/5) рёбра по порогу косинуса и окну времени.
    ann = cfg.get("ann", {})
    sims, idx = ann_neighbors(
        emb,
        top_k=ann.get("top_k", 20),
        nlist=ann.get("nlist", 256),
        nprobe=ann.get("nprobe", 32),
    )
    _log_rss("FAISS индекс")
    cosine_threshold = cfg.get("cosine_threshold", 0.85)
    tw_hours = cfg.get("time_window_hours", 72)
    sem_w = build_weighted_edges(
        sims, idx, published,
        cosine_threshold=cosine_threshold,
        time_window_hours=tw_hours,
    )
    print(f"Семантические рёбра: {len(sem_w)}")

    # 6) Кластеризация графа рёбер.
    # Lexical edges тоже фильтруем по времени: boilerplate-заголовки (ТАСС, Интерфакс)
    # иначе склеят события, разнесённые на месяцы. Лексическим рёбрам даём
    # высокий вес (1.0) — это высокоточные MinHash-совпадения.
    tw = tw_hours * 3600.0
    ts = [d.timestamp() for d in published]
    lex = [(i, j) for i, j in lex if abs(ts[i] - ts[j]) <= tw]
    weighted = dict(sem_w)
    for i, j in lex:
        a, b = (i, j) if i < j else (j, i)
        weighted[(a, b)] = max(weighted.get((a, b), 0.0), 1.0)

    method = cfg.get("clustering", "louvain")
    if method == "louvain":
        labels = louvain_clusters(
            n, weighted,
            resolution=cfg.get("louvain_resolution", 1.0),
            seed=cfg.get("seed", 42),
        )
        print(f"Кластеризация: Louvain (resolution={cfg.get('louvain_resolution', 1.0)})")
    else:
        labels = connected_clusters(n, sorted(weighted.keys()))
        print("Кластеризация: connected components")
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
