"""PGS-axis chunking helpers for ``05_score_prs``.

Full-catalog mean_impute densifies ``plan ⋈ weights ⋈ afreq`` before joining dose.
At ~337M weight rows that join shuffles tens–hundreds of GB on a single node.
Chunking bounds each densify so weights can broadcast (or at least SMJ a smaller set).
"""
from __future__ import annotations

from typing import Iterable, Sequence


def chunk_fixed(items: Sequence, size: int) -> list[list]:
    """Split ``items`` into contiguous chunks of at most ``size``.

    ``size <= 0`` yields a single chunk containing all items (baseline / no chunking).
    """
    if not items:
        return []
    if size <= 0:
        return [list(items)]
    return [list(items[i : i + size]) for i in range(0, len(items), size)]


def chunk_by_weight_budget(
    pgs_rows: Sequence[dict],
    *,
    weight_count_key: str = "n_weights",
    max_weight_rows: int = 3_000_000,
) -> list[list[dict]]:
    """Greedy-pack PGS rows so each chunk's Σ weight rows stays ≤ ``max_weight_rows``.

    Each dict must carry ``weight_count_key`` (int). A single PGS larger than the
    budget still gets its own chunk (never split mid-PGS). ``max_weight_rows <= 0``
    disables packing and returns one chunk.
    """
    if not pgs_rows:
        return []
    if max_weight_rows <= 0:
        return [list(pgs_rows)]

    chunks: list[list[dict]] = []
    cur: list[dict] = []
    cur_w = 0
    for row in pgs_rows:
        w = int(row.get(weight_count_key) or 0)
        if cur and cur_w + w > max_weight_rows:
            chunks.append(cur)
            cur, cur_w = [], 0
        cur.append(row)
        cur_w += w
    if cur:
        chunks.append(cur)
    return chunks


def iter_chunk_ids(chunks: Iterable[Sequence[dict]], id_key: str = "pgs_id") -> list[list[str]]:
    """Extract id lists from packed chunk dicts (for logging / predicates)."""
    return [[str(r[id_key]) for r in ch] for ch in chunks]
