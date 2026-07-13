"""Semantic embeddings via sentence-transformers (deepvk/USER-bge-m3).

Vectors are L2-normalized -> cosine similarity = dot product,
which allows searching for neighbors via a FAISS inner-product index.

At large volumes (100k) an incremental cache by id is supported:
embeddings are written in chunks, so a crash doesn't lose what's already computed.
"""
from __future__ import annotations

import os

# macOS pulls in several copies of libomp (torch + faiss + sklearn) -> double
# OpenMP initialization -> segfault. Allow coexistence before the dylib import.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np


def _cpu_threads() -> int:
    """Actual number of available cores (affinity), not host cpu_count.

    Inside a container/cgroup, os.cpu_count() lies -> thread oversubscription
    and contention. sched_getaffinity gives the real quota (Linux);
    it's absent on macOS -> fall back to cpu_count.
    """
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:
        return os.cpu_count() or 1


def _build_docs(
    titles: list[str], texts: list[str], max_chars: int, query_prefix: str
) -> list[str]:
    # title reinforces the event signal; text is truncated for speed/memory.
    return [
        f"{query_prefix}{(t or '').strip()}. {(body or '')[:max_chars].strip()}"
        for t, body in zip(titles, texts)
    ]


def _save_cache(cache_path: str, ids: list[str], vectors: np.ndarray) -> None:
    """Atomic cache write: tmp -> os.replace (a crash doesn't corrupt the file).

    np.savez appends .npz itself if the name lacks it, so we keep tmp
    already with .npz so the real path matches what we rename.
    """
    tmp = f"{cache_path}.tmp.npz"
    np.savez(tmp, ids=np.asarray(ids, dtype=object), vectors=vectors)
    os.replace(tmp, cache_path)


def _load_cache(cache_path: str) -> dict[str, np.ndarray]:
    """Read the cache as {id: vector}; empty dict if the file is missing/corrupt."""
    if not os.path.exists(cache_path):
        return {}
    try:
        data = np.load(cache_path, allow_pickle=True)
        return {str(k): v for k, v in zip(data["ids"], data["vectors"])}
    except Exception:
        return {}  # corrupt cache — recompute from scratch


def build_embeddings(
    titles: list[str],
    texts: list[str],
    model_name: str = "deepvk/USER-bge-m3",
    max_chars: int = 1500,
    batch_size: int = 64,
    query_prefix: str = "",
    seed: int = 42,
    ids: list[str] | None = None,
    cache_path: str | None = None,
    chunk_size: int = 2000,
) -> np.ndarray:
    """Return an (N, dim) float32 matrix, L2-normalized, in input order.

    If ids and cache_path are given — incremental mode: only ids missing
    from the cache are computed, and the cache is appended chunk by chunk
    (atomic write). A crash at 90k of 100k -> a rerun finishes the rest.
    """
    import torch
    from sentence_transformers import SentenceTransformer

    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.set_num_threads(_cpu_threads())

    docs = _build_docs(titles, texts, max_chars, query_prefix)

    # Simple path without a cache (small volumes / one-off runs).
    if ids is None or cache_path is None:
        model = SentenceTransformer(model_name)
        emb = model.encode(
            docs,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,  # L2-norm
        )
        return emb.astype("float32")

    # Incremental mode with an id-based cache.
    cache = _load_cache(cache_path)
    todo = [(i, d) for i, (cid, d) in enumerate(zip(ids, docs)) if cid not in cache]
    if todo:
        print(f"Эмбеддинги: {len(cache)} из кэша, считаем {len(todo)}")
        model = SentenceTransformer(model_name)
        for start in range(0, len(todo), chunk_size):
            chunk = todo[start : start + chunk_size]
            vecs = model.encode(
                [d for _, d in chunk],
                batch_size=batch_size,
                show_progress_bar=True,
                convert_to_numpy=True,
                normalize_embeddings=True,
            ).astype("float32")
            for (idx, _), v in zip(chunk, vecs):
                cache[ids[idx]] = v
            # Append to the cache after every chunk — makes the run resumable.
            all_ids = list(cache.keys())
            _save_cache(cache_path, all_ids, np.vstack([cache[k] for k in all_ids]))
            print(f"  кэш: {len(cache)}/{len(ids)}")
    else:
        print(f"Эмбеддинги: все {len(ids)} из кэша")

    # Assemble the matrix strictly in the order of the input ids.
    return np.vstack([cache[cid] for cid in ids]).astype("float32")
