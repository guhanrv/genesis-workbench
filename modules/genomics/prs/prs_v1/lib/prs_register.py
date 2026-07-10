"""Pure parsers that turn PGS-catalog curation into rows for the three
catalog-side Delta stores — ``pgs_registry``, ``pgs_weights``, ``pgs_panel_ref``.

Kept free of Spark/dbutils so it can be unit-tested at $0 off-cluster (the
notebook ``01_register_catalog`` is a thin wrapper: read curation → call these →
``createDataFrame`` → MERGE).

Design notes
------------
* **Weights are effect-oriented** to match ``prs_extract``/``gvcf_dose`` and the
  scorer (``raw = Σ dose·weight``): ``variant_id = chrom:pos:effect:other`` and a
  plain ``weight``. Rows at the same ``variant_id`` within one PGS are summed
  (a scorefile can list a variant twice).
* **Panel reference: computed OR curated.** By default ``ref_01_build_panel_ref`` COMPUTES
  ``pgs_panel_ref`` (per-PGS × superpop mean/sd) by scoring the reference panel per PGS —
  self-contained, scales to 100+ PGS, no offline plink2. As an OVERRIDE, put frozen stats
  in ``prs.yaml``'s ``reference_distribution`` and ``panel_ref_rows`` (below) writes them
  directly ($0) — same ``pgs_panel_ref`` table, same ``(pgs_id, superpop, panel_version)``
  key, so the two paths are interchangeable behind this seam. ``panel_version`` is the
  curation's ``version`` (both paths must stamp the same one for the scorer's z-join).
* **weight_sha** is the sha256 of the raw scorefile bytes — the content address
  that scopes a single-PGS restatement to its own column (registry/weights/panel_ref
  for one PGS all share it; the other 100 stay valid).
"""
from __future__ import annotations

import gzip
import hashlib


# superpop key (as curated in prs.yaml reference_distribution) — stored verbatim as
# pgs_panel_ref.superpop; the scorer joins it BY EQUALITY to the PCA module's
# most_similar_pop (05_score_prs). Those are the HGDP+1kGP panel SuperPop CODES
# (from the .psam SuperPop column, carried through the PCA basis + RF classifier),
# so the curated keys must be the same codes or the z-join silently misses.
_PANEL_SUPERPOPS = ("AFR", "AMR", "CSA", "EAS", "EUR", "MID")


def compute_weight_sha(scorefile_path) -> str:
    """sha256 of the raw scorefile bytes (content address for the weights)."""
    h = hashlib.sha256()
    with open(scorefile_path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_scorefile(scorefile_path):
    """Harmonized PGS-catalog scoring file → list of (chrom, pos, effect, other, weight).

    Uses harmonized columns when present (hm_chr/hm_pos), else the author columns
    (chr_name/chr_position); other-allele falls back reference_allele → hm_inferOtherAllele
    (the PGS-catalog harmonizer's inferred other allele — some scorefiles carry ONLY this).
    ``chrom`` is stripped of a leading ``chr`` (ensembl style, matching gvcf_dose/prs_extract).
    Rows with no position (or no resolvable other allele) are skipped.
    """
    rows = []
    with gzip.open(scorefile_path, "rt") as f:
        hdr = None
        ix = {}
        for line in f:
            if line.startswith("#"):
                continue
            c = line.rstrip("\n").split("\t")
            if hdr is None:
                hdr = c
                ix = {k: i for i, k in enumerate(hdr)}
                cch = "hm_chr" if "hm_chr" in ix else "chr_name"
                cps = "hm_pos" if "hm_pos" in ix else "chr_position"
                coa = next((col for col in ("other_allele", "reference_allele",
                                            "hm_inferOtherAllele") if col in ix), None)
                if coa is None:
                    raise ValueError(
                        f"{scorefile_path}: no other/reference/hm_inferOtherAllele column "
                        f"(cols={hdr}) — cannot form chrom:pos:effect:other variant_id")
                continue
            if not c[ix[cps]] or not c[ix[coa]]:
                continue
            rows.append((
                str(c[ix[cch]]).replace("chr", ""),
                int(c[ix[cps]]),
                c[ix["effect_allele"]],
                c[ix[coa]],
                float(c[ix["effect_weight"]]),
            ))
    return rows


_PALINDROMIC = ({"A", "T"}, {"C", "G"})


def is_palindromic(effect: str, other: str) -> bool:
    """Strand-ambiguous SNP (A/T or C/G): the effect allele can't be strand-resolved
    without extra info, so the extracted dose may be for the wrong strand."""
    return {str(effect).upper(), str(other).upper()} in _PALINDROMIC


def weights_rows(pgs_id: str, scorefile_rows, weight_sha: str, drop_palindromic: bool = True):
    """Effect-oriented pgs_weights rows, deduped by variant_id (weights summed).

    Returns list of dicts: pgs_id, variant_id, effect_allele, other_allele, weight, weight_sha.

    ``drop_palindromic`` (default True): skip A/T and C/G SNPs — they're strand-ambiguous, so a
    gVCF-extracted dose can silently be for the wrong strand. This matches pgsc_calc/plink2 (and
    function_prs) drop-mode. Parity-validated: dropping them makes this pipeline's raw score match
    function_prs to machine epsilon on PGS000004 (the 31 palindromic vars there flip the sign).
    """
    acc: dict[str, dict] = {}
    for chrom, pos, effect, other, w in scorefile_rows:
        if drop_palindromic and is_palindromic(effect, other):
            continue
        vid = f"{chrom}:{pos}:{effect}:{other}"
        r = acc.get(vid)
        if r is None:
            acc[vid] = {
                "pgs_id": pgs_id,
                "variant_id": vid,
                "effect_allele": str(effect),
                "other_allele": str(other),
                "weight": float(w),
                "weight_sha": weight_sha,
            }
        else:
            r["weight"] += float(w)
    return [acc[v] for v in sorted(acc)]


def registry_row(entry: dict, weight_sha: str, n_variants: int, weight_path: str) -> dict:
    """One pgs_registry row from a prs.yaml ``scores`` entry (registered_at set by the notebook)."""
    return {
        "pgs_id": entry["pgs_id"],
        "score_id": entry.get("id"),
        "disease": entry.get("disease"),
        "direction": entry.get("direction"),
        "body_system": ";".join(entry.get("functional_categories", []) or []) or None,
        "hr_per_sd": (float(entry["hr_per_sd"]) if entry.get("hr_per_sd") is not None else None),
        "clinical_model": entry.get("clinical_model"),
        "training_ancestries": ";".join(entry.get("training_ancestries", []) or []) or None,
        "weight_sha": weight_sha,
        "n_variants": int(n_variants),
        "weight_path": weight_path,
    }


def panel_ref_rows(entry: dict, panel_version: str, weight_sha: str):
    """pgs_panel_ref rows from the curated (frozen) reference_distribution.

    Returns one row per superpop present: pgs_id, superpop, mean, sd, quantiles(None),
    n_panel(None), panel_version, weight_sha. Superpop key stored verbatim as curated.
    """
    dist = entry.get("reference_distribution") or {}
    out = []
    for sp in _PANEL_SUPERPOPS:
        d = dist.get(sp)
        if not d or d.get("mean") is None or d.get("sd") is None:
            continue
        out.append({
            "pgs_id": entry["pgs_id"],
            "superpop": sp,
            "mean": float(d["mean"]),
            "sd": float(d["sd"]),
            "quantiles": None,
            "n_panel": None,
            "panel_version": panel_version,
            "weight_sha": weight_sha,
        })
    return out
