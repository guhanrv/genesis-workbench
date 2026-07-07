"""Effect-oriented union + dense dose fill for the distributed PRS scorer.

The gVCF kernel (``gvcf_dose``) extracts the dose of a catalog row's ``effect``
allele. We store dose **oriented to the effect allele** per ``(chrom,pos,effect,other)``
variant, and each PGS weight is plain (``raw = Σ dose·weight``) — matching
function_prs's USER-side scoring exactly.

Why not a single canonical dose + signed-weight/offset (the panel-scorer trick)?
Because the gVCF kernel returns dose 0 for a REF block whose FASTA base is *neither*
catalog allele ("0 copies of both alleles", a catalog-vs-reference disagreement).
That correct value breaks the ``dose_effect = 2 − dose_canonical`` flip the offset
trick assumes — a single canonical dose can't encode "0 of both" for both
orientations. Validated: the canonical approach diverged from the naive
``Σ dose(effect)·weight`` on Amy's real gVCF; the effect-oriented store matches it
exactly. The signed-weight trick is safe only on the panel side (clean pgen dose).

Dedup is by ``(chrom,pos,effect,other)``: PGS that count the same allele at a
position share a stored dose; the rare cross-PGS opposite-orientation case stores
both — correct, and still near the ~19M-variant union in practice.

Dense fill matches function_prs's three-way rule (score_kernel.py:198-244):
real dose kept; covered-but-no-informative-dose → 0 (confirmed zero); truly-missing
→ 2·AF(effect) (mean-impute vs panel afreq).
"""
from __future__ import annotations

import numpy as np


def variant_id(chrom, pos, effect: str, other: str) -> str:
    """Effect-oriented id: the stored dose is the dose of ``effect``."""
    return f"{chrom}:{pos}:{effect}:{other}"


class UnionCatalog:
    """A `_Catalog` duck-type for gvcf_dose. catalog.effect/other are the PGS's
    effect/other, so the kernel returns the dose of the effect allele."""
    __slots__ = ("n_var", "chrom", "pos", "effect", "other", "variant_id")

    def __init__(self, chrom, pos, effect, other, vid):
        self.chrom = chrom; self.pos = pos
        self.effect = effect; self.other = other
        self.variant_id = vid; self.n_var = len(pos)


def build_union_catalog(rows) -> UnionCatalog:
    """rows: iterable of (chrom, pos, effect, other). Dedup by
    (chrom,pos,effect,other); the kernel extracts dose(effect) per row."""
    seen: dict[str, tuple] = {}
    for chrom, pos, effect, other in rows:
        vid = f"{chrom}:{pos}:{effect}:{other}"
        seen[vid] = (str(chrom), int(pos), str(effect), str(other))
    vids = sorted(seen)
    chrom = np.array([seen[v][0] for v in vids])
    pos = np.array([seen[v][1] for v in vids], dtype=np.int64)
    eff = np.array([seen[v][2] for v in vids])
    oth = np.array([seen[v][3] for v in vids])
    return UnionCatalog(chrom, pos, eff, oth, np.array(vids))


def dense_fill(dose: np.ndarray, had_record: np.ndarray, af_effect) -> np.ndarray:
    """function_prs three-way fill → a dense dose vector (no NaN).
    real dose kept · (nan & had_record) → 0 · (nan & not had_record) → 2·AF(effect)."""
    out = dose.astype(np.float64).copy()
    nan = np.isnan(out)
    out[nan & had_record] = 0.0
    imp = nan & (~had_record)
    out[imp] = 2.0 * np.asarray(af_effect, dtype=np.float64)[imp]
    return out
