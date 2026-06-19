"""Семантические эмбеддинги через sentence-transformers (deepvk/USER-bge-m3).

Вектора L2-нормализуются -> косинусная близость = скалярное произведение,
что позволяет искать соседей через FAISS inner-product индекс.
"""
from __future__ import annotations

import numpy as np


def build_embeddings(
    titles: list[str],
    texts: list[str],
    model_name: str = "deepvk/USER-bge-m3",
    max_chars: int = 1500,
    batch_size: int = 64,
    query_prefix: str = "",
    seed: int = 42,
) -> np.ndarray:
    """Вернуть матрицу (N, dim) float32, L2-normalized."""
    import torch
    from sentence_transformers import SentenceTransformer

    torch.manual_seed(seed)
    np.random.seed(seed)

    # title усиливает сигнал события; обрезаем текст для скорости/памяти.
    docs = [
        f"{query_prefix}{(t or '').strip()}. {(body or '')[:max_chars].strip()}"
        for t, body in zip(titles, texts)
    ]

    model = SentenceTransformer(model_name)
    emb = model.encode(
        docs,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,  # L2-norm
    )
    return emb.astype("float32")
