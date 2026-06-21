"""Извлечение именованных сущностей (NER) через natasha для NE-gating.

Идея: два поста с похожими эмбеддингами, но разными сущностями (персоны,
организации, локации) — это разные события с общим boilerplate-языком
(агентские шаблоны ТАСС/Интерфакс), а не дубли. Совпадение хотя бы одной
сущности — необходимое условие для семантического ребра в "серой зоне"
косинуса.

Извлечение тяжёлое (slovnet-модель ~5 мин на 80k), поэтому кэшируется по id
в .npz рядом с эмбеддингами.
"""
from __future__ import annotations

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np

# Метки сущностей, которые различают события. ORG специально не берём как
# жёсткий сигнал в одиночку — агентства (ТАСС/РИА) сами ORG и зашумляют.
_ENTITY_TYPES = {"PER", "LOC"}

_MODELS = None


def _load_models():
    """Ленивая инициализация natasha (singleton): сегментер + NER + морфо."""
    global _MODELS
    if _MODELS is None:
        from natasha import (
            Doc,
            MorphVocab,
            NewsEmbedding,
            NewsNERTagger,
            Segmenter,
        )

        emb = NewsEmbedding()
        _MODELS = {
            "Doc": Doc,
            "segmenter": Segmenter(),
            "ner": NewsNERTagger(emb),
            "morph": MorphVocab(),
        }
    return _MODELS


def extract_entities(text: str) -> frozenset[str]:
    """Множество нормализованных сущностей (PER, LOC) из текста, lower-case.

    Нормализация (span.normal) приводит словоформы к лемме: "Собянина" и
    "Собянин" → один токен, иначе морфология русского ломает пересечение.
    """
    if not text:
        return frozenset()
    m = _load_models()
    doc = m["Doc"](text)
    doc.segment(m["segmenter"])
    doc.tag_ner(m["ner"])
    out: set[str] = set()
    for span in doc.spans:
        if span.type not in _ENTITY_TYPES:
            continue
        span.normalize(m["morph"])
        out.add((span.normal or span.text).lower())
    return frozenset(out)


def build_entities(
    texts: list[str],
    ids: list[str] | None = None,
    cache_path: str | None = None,
) -> list[frozenset[str]]:
    """Сущности для каждого текста, с инкрементальным кэшем по id.

    Без ids/cache_path — простой проход без кэша. Возвращает список в порядке
    входа, элемент i — frozenset сущностей текста i.
    """
    if ids is None or cache_path is None:
        return [extract_entities(t) for t in texts]

    cache = _load_cache(cache_path)
    todo = [(i, t) for i, (cid, t) in enumerate(zip(ids, texts)) if cid not in cache]
    if todo:
        print(f"NER: {len(cache)} из кэша, считаем {len(todo)}")
        for n, (i, t) in enumerate(todo, 1):
            cache[ids[i]] = extract_entities(t)
            if n % 5000 == 0:
                _save_cache(cache_path, cache)
                print(f"  NER кэш: {len(cache)}/{len(ids)}")
        _save_cache(cache_path, cache)
    else:
        print(f"NER: все {len(ids)} из кэша")

    return [cache[cid] for cid in ids]


def _save_cache(cache_path: str, cache: dict[str, frozenset[str]]) -> None:
    """Атомарная запись: список сущностей через '|' разделитель в object-массиве."""
    keys = list(cache.keys())
    vals = np.asarray(["|".join(sorted(cache[k])) for k in keys], dtype=object)
    tmp = f"{cache_path}.tmp.npz"
    np.savez(tmp, ids=np.asarray(keys, dtype=object), ents=vals)
    os.replace(tmp, cache_path)


def _load_cache(cache_path: str) -> dict[str, frozenset[str]]:
    if not os.path.exists(cache_path):
        return {}
    try:
        data = np.load(cache_path, allow_pickle=True)
        return {
            str(k): frozenset(s.split("|") if s else [])
            for k, s in zip(data["ids"], data["ents"])
        }
    except Exception:
        return {}
