"""Named entity recognition (NER) via natasha, for NE-gating.

Idea: two posts with similar embeddings but different entities (persons,
organizations, locations) are different events sharing boilerplate language
(TASS/Interfax wire-service templates), not duplicates. A match on at least
one entity is a necessary condition for a semantic edge in the "gray zone"
of cosine similarity.

Extraction is heavy (slovnet model, ~5 min for 80k), so it's cached by id
in a .npz file next to the embeddings.
"""
from __future__ import annotations

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np

# Entity labels that distinguish events. ORG is deliberately excluded as a
# standalone hard signal — wire agencies (TASS/RIA) are themselves ORGs and add noise.
_ENTITY_TYPES = {"PER", "LOC"}

_MODELS = None


def _load_models():
    """Lazy natasha initialization (singleton): segmenter + NER + morphology."""
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
    """Set of normalized entities (PER, LOC) from the text, lower-cased.

    Normalization (span.normal) reduces word forms to their lemma: "Собянина"
    and "Собянин" -> one token; otherwise Russian morphology breaks the intersection.
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
    """Entities for each text, with an incremental cache keyed by id.

    Without ids/cache_path — a plain pass with no caching. Returns a list in
    input order; element i is the frozenset of entities for text i.
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
    """Atomic write: entity list joined with '|' into an object array."""
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
