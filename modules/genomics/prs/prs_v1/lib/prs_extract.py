"""Orientation-canonical union + dense dose fill for the distributed PRS scorer.

The gVCF kernel (``gvcf_dose``) extracts the dose of a catalog row's ``effect``
allele. For the dense ``dosage`` store we keep ONE canonical dose per variant —
the dose of the lexicographically-greater allele (``allele_hi``) — and orient each
PGS's weight via a signed weight + a per-PGS offset (the function_prs panel-scorer
trick), so a variant is stored once regardless of which allele a PGS counts:

    canonical dose d = dose(allele_hi)
    PGS counts allele_hi (not flipped): contribution = w · d
    PGS counts allele_lo (flipped):     contribution = w · (2 − d) = (−w)·d + 2w
      → signed_weight = −w, offset += 2w

Dense fill follows function_prs's three-way rule (score_kernel.py:198-244):
real dose kept; covered-but-no-informative-dose → 0 (confirmed zero); truly-missing
(no record covered the position) → 2·AF(allele_hi) (mean-impute vs panel afreq).
"""
from __future__ import annotations

import numpy as np


def canonical(effect: str, other: str) -> tuple[str, str, bool]:
    """Return (allele_lo, allele_hi, flip). flip = the PGS counts allele_lo, so
    the stored dose (of allele_hi) must be flipped (2 − d) to get effect dose."""
    if effect <= other:
        lo, hi = effect, other
    else:
        lo, hi = other, effect
    flip = (effect == lo) and (effect != other)
    return lo, hi, flip


def variant_id(chrom, pos, effect: str, other: str) -> str:
    lo, hi, _ = canonical(effect, other)
    return f"{chrom}:{pos}:{lo}:{hi}"


def signed_weight(w: float, flip: bool) -> float:
    return -w if flip else w


def offset_contrib(w: float, flip: bool) -> float:
    return 2.0 * w if flip else 0.0


class UnionCatalog:
    """A `_Catalog` duck-type for gvcf_dose: effect=allele_hi, other=allele_lo, so
    the kernel returns dose(allele_hi) = the canonical dose we store."""
    __slots__ = ("n_var", "chrom", "pos", "effect", "other", "variant_id")

    def __init__(self, chrom, pos, effect, other, vid):
        self.chrom = chrom; self.pos = pos
        self.effect = effect; self.other = other
        self.variant_id = vid; self.n_var = len(pos)


def build_union_catalog(rows) -> UnionCatalog:
    """rows: iterable of (chrom, pos, allele_a, allele_b). Dedup to orientation-
    canonical variants; catalog.effect = allele_hi, catalog.other = allele_lo."""
    seen: dict[str, tuple] = {}
    for chrom, pos, a, b in rows:
        lo, hi, _ = canonical(a, b)
        vid = f"{chrom}:{pos}:{lo}:{hi}"
        seen[vid] = (str(chrom), int(pos), hi, lo)
    vids = sorted(seen)
    chrom = np.array([seen[v][0] for v in vids])
    pos = np.array([seen[v][1] for v in vids], dtype=np.int64)
    hi = np.array([seen[v][2] for v in vids])
    lo = np.array([seen[v][3] for v in vids])
    return UnionCatalog(chrom, pos, hi, lo, np.array(vids))


def dense_fill(dose: np.ndarray, had_record: np.ndarray, af_hi) -> np.ndarray:
    """function_prs three-way fill → a dense dose vector (no NaN).
    real dose kept · (nan & had_record) → 0 · (nan & not had_record) → 2·AF(hi)."""
    out = dose.astype(np.float64).copy()
    nan = np.isnan(out)
    out[nan & had_record] = 0.0
    imp = nan & (~had_record)
    out[imp] = 2.0 * np.asarray(af_hi, dtype=np.float64)[imp]
    return out
