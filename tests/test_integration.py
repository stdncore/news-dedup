"""Integration tests with the real model.

Require the deepvk/USER-bge-m3 model to be downloaded (~560 MB).
Run: pytest tests/test_integration.py -m integration -v

Marked with @pytest.mark.integration and skipped in CI by default.
Use conftest.py or pytest.ini to register the marker.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

FIXTURE = Path("tests/fixtures/news_sample.json")
LABELS = Path("tests/fixtures/labeled_pairs.json")

pytest.importorskip("sentence_transformers", reason="sentence-transformers not installed")


@pytest.mark.integration
def test_real_model_known_duplicates():
    """Pairs with label=1 should have cosine > 0.80 on the real model.

    Verifies that query_prefix="" gives correct similarity for the doc-doc task.
    If the cosine is below the threshold, the query_prefix may be wrong or the
    model may have degraded.
    """
    import yaml
    from dedup.embed import build_embeddings

    posts = json.loads(FIXTURE.read_text(encoding="utf-8"))
    labeled = json.loads(LABELS.read_text(encoding="utf-8"))

    # Take 5 positive pairs (label=1)
    pos_pairs = [p for p in labeled if p["label"] == 1][:5]
    assert pos_pairs, "No positive pairs in labeled_pairs.json"

    # Collect the needed ids
    needed_ids = {p["id1"] for p in pos_pairs} | {p["id2"] for p in pos_pairs}
    id2post = {p["id"]: p for p in posts if p["id"] in needed_ids}
    missing = needed_ids - set(id2post)
    if missing:
        pytest.skip(f"Posts from labeled_pairs not found in fixture: {missing}")

    cfg = yaml.safe_load(Path("config.yaml").read_text(encoding="utf-8"))["dedup"]

    all_ids = list(id2post)
    titles = [id2post[i].get("title", "") for i in all_ids]
    texts = [id2post[i].get("text", "") for i in all_ids]
    pos_map = {id_: idx for idx, id_ in enumerate(all_ids)}

    emb = build_embeddings(
        titles, texts,
        model_name=cfg["model"],
        max_chars=cfg.get("max_chars", 1500),
        batch_size=cfg.get("batch_size", 64),
        query_prefix=cfg.get("query_prefix", ""),
        seed=cfg.get("seed", 42),
    )

    threshold = cfg.get("cosine_threshold", 0.75)
    failed = []
    for pair in pos_pairs:
        i, j = pos_map[pair["id1"]], pos_map[pair["id2"]]
        sim = float(np.dot(emb[i], emb[j]))
        if sim < threshold:
            failed.append((pair["id1"], pair["id2"], sim))

    assert not failed, (
        f"The following positive pairs have cosine < {threshold} — "
        f"the query_prefix may be wrong or the model may have degraded:\n"
        + "\n".join(f"  {a} <-> {b}: {s:.3f}" for a, b, s in failed)
    )
