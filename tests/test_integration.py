"""Integration-тесты с реальной моделью.

Требуют скачанной модели deepvk/USER-bge-m3 (~560 MB).
Запуск: pytest tests/test_integration.py -m integration -v

Маркируются @pytest.mark.integration и пропускаются в CI по умолчанию.
Используй conftest.py или pytest.ini для регистрации маркера.
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
    """Пары с label=1 должны иметь cosine > 0.80 на реальной модели.

    Проверяет что query_prefix="" даёт корректное сходство для doc-doc задачи.
    Если косинус < порога — возможен неверный query_prefix или деградация модели.
    """
    import yaml
    from dedup.embed import build_embeddings

    posts = json.loads(FIXTURE.read_text(encoding="utf-8"))
    labeled = json.loads(LABELS.read_text(encoding="utf-8"))

    # Берём 5 позитивных пар (label=1)
    pos_pairs = [p for p in labeled if p["label"] == 1][:5]
    assert pos_pairs, "Нет позитивных пар в labeled_pairs.json"

    # Собираем нужные id
    needed_ids = {p["id1"] for p in pos_pairs} | {p["id2"] for p in pos_pairs}
    id2post = {p["id"]: p for p in posts if p["id"] in needed_ids}
    missing = needed_ids - set(id2post)
    if missing:
        pytest.skip(f"Посты из labeled_pairs не найдены в fixture: {missing}")

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

    failed = []
    for pair in pos_pairs:
        i, j = pos_map[pair["id1"]], pos_map[pair["id2"]]
        sim = float(np.dot(emb[i], emb[j]))
        if sim < 0.80:
            failed.append((pair["id1"], pair["id2"], sim))

    assert not failed, (
        f"Следующие позитивные пары имеют cosine < 0.80 — "
        f"возможен неверный query_prefix или деградация модели:\n"
        + "\n".join(f"  {a} <-> {b}: {s:.3f}" for a, b, s in failed)
    )
