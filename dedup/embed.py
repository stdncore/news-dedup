"""Семантические эмбеддинги через sentence-transformers (deepvk/USER-bge-m3).

Вектора L2-нормализуются -> косинусная близость = скалярное произведение,
что позволяет искать соседей через FAISS inner-product индекс.

На больших объёмах (100k) поддерживается инкрементальный кэш по id:
эмбеддинги пишутся чанками, краш не теряет уже посчитанное.
"""
from __future__ import annotations

import os

# macOS тащит несколько копий libomp (torch + faiss + sklearn) -> двойная
# инициализация OpenMP -> segfault. Разрешаем сосуществование до импорта dylib.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np


def _cpu_threads() -> int:
    """Реальное число доступных ядер (affinity), не host-cpu_count.

    Внутри контейнера/cgroup os.cpu_count() врёт -> переподписка потоков
    и contention. sched_getaffinity даёт фактическую квоту (Linux);
    на macOS его нет -> fallback на cpu_count.
    """
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:
        return os.cpu_count() or 1


def _build_docs(
    titles: list[str], texts: list[str], max_chars: int, query_prefix: str
) -> list[str]:
    # title усиливает сигнал события; обрезаем текст для скорости/памяти.
    return [
        f"{query_prefix}{(t or '').strip()}. {(body or '')[:max_chars].strip()}"
        for t, body in zip(titles, texts)
    ]


def _save_cache(cache_path: str, ids: list[str], vectors: np.ndarray) -> None:
    """Атомарная запись кэша: tmp -> os.replace (краш не бьёт файл).

    np.savez сам дописывает .npz если имени его нет, поэтому tmp держим
    уже с .npz, чтобы реальный путь совпал с тем, что мы переименовываем.
    """
    tmp = f"{cache_path}.tmp.npz"
    np.savez(tmp, ids=np.asarray(ids, dtype=object), vectors=vectors)
    os.replace(tmp, cache_path)


def _load_cache(cache_path: str) -> dict[str, np.ndarray]:
    """Прочитать кэш как {id: vector}; пустой dict если файла нет/битый."""
    if not os.path.exists(cache_path):
        return {}
    try:
        data = np.load(cache_path, allow_pickle=True)
        return {str(k): v for k, v in zip(data["ids"], data["vectors"])}
    except Exception:
        return {}  # битый кэш — считаем заново


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
    """Вернуть матрицу (N, dim) float32, L2-normalized в порядке входа.

    Если заданы ids и cache_path — инкрементальный режим: считаются только
    отсутствующие в кэше id, кэш дописывается чанками (atomic write).
    Краш на 90k из 100k -> повторный запуск досчитает остаток.
    """
    import torch
    from sentence_transformers import SentenceTransformer

    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.set_num_threads(_cpu_threads())

    docs = _build_docs(titles, texts, max_chars, query_prefix)

    # Простой путь без кэша (малые объёмы / разовые прогоны).
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

    # Инкрементальный режим с кэшем по id.
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
            # Дописываем кэш после каждого чанка — выживаемость прогона.
            all_ids = list(cache.keys())
            _save_cache(cache_path, all_ids, np.vstack([cache[k] for k in all_ids]))
            print(f"  кэш: {len(cache)}/{len(ids)}")
    else:
        print(f"Эмбеддинги: все {len(ids)} из кэша")

    # Собираем матрицу строго в порядке входных id.
    return np.vstack([cache[cid] for cid in ids]).astype("float32")
