"""Lexical pre-filter for near-duplicates via MinHash + LSH (datasketch).

A cheap recall-oriented candidate generator: catches copy-paste and
near-identical texts before the semantic stage. Config follows the
production HuggingFace reference: 256 permutations, Jaccard 0.7, 5-word shingles.

Returns edges (i, j) - candidate index pairs, which are then merged directly
into the graph (lexical duplicates are considered reliable).
"""
from __future__ import annotations

from datasketch import MinHash, MinHashLSH

from .text import word_tokens


def _shingles(tokens: list[str], k: int) -> set[str]:
    if len(tokens) < k:
        return {" ".join(tokens)} if tokens else set()
    return {" ".join(tokens[i : i + k]) for i in range(len(tokens) - k + 1)}


def _minhash(shingles: set[str], num_perm: int) -> MinHash:
    m = MinHash(num_perm=num_perm)
    for sh in shingles:
        m.update(sh.encode("utf-8"))
    return m


def lexical_edges(
    texts: list[str],
    num_perm: int = 256,
    jaccard_threshold: float = 0.7,
    shingle_size: int = 5,
) -> list[tuple[int, int]]:
    """Index pairs with estimated Jaccard >= threshold."""
    lsh = MinHashLSH(threshold=jaccard_threshold, num_perm=num_perm)
    minhashes: list[MinHash] = []
    for i, text in enumerate(texts):
        sh = _shingles(word_tokens(text), shingle_size)
        m = _minhash(sh, num_perm)
        minhashes.append(m)
        lsh.insert(str(i), m)

    edges: set[tuple[int, int]] = set()
    for i, m in enumerate(minhashes):
        for key in lsh.query(m):
            j = int(key)
            if j != i:
                edges.add((min(i, j), max(i, j)))
    return sorted(edges)
