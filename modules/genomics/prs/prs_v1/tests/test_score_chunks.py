"""Unit tests for ``lib/score_chunks.py``."""
from __future__ import annotations

from score_chunks import chunk_by_weight_budget, chunk_fixed, iter_chunk_ids


def test_chunk_fixed_all_at_once():
    items = [{"pgs_id": f"P{i}"} for i in range(5)]
    assert chunk_fixed(items, 0) == [items]
    assert chunk_fixed([], 3) == []


def test_chunk_fixed_splits():
    items = list(range(7))
    assert chunk_fixed(items, 3) == [[0, 1, 2], [3, 4, 5], [6]]


def test_weight_budget_packs_and_isolates_oversize():
    rows = [
        {"pgs_id": "a", "n_weights": 2_000_000},
        {"pgs_id": "b", "n_weights": 2_000_000},
        {"pgs_id": "c", "n_weights": 500_000},
        {"pgs_id": "d", "n_weights": 4_000_000},  # alone
    ]
    chunks = chunk_by_weight_budget(rows, max_weight_rows=3_000_000)
    assert [iter_chunk_ids([ch])[0] for ch in chunks] == [
        ["a"],
        ["b", "c"],
        ["d"],
    ]


def test_weight_budget_disabled():
    rows = [{"pgs_id": "a", "n_weights": 1}, {"pgs_id": "b", "n_weights": 1}]
    assert chunk_by_weight_budget(rows, max_weight_rows=0) == [rows]
