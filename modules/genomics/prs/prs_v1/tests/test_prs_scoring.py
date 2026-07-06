"""Pure-Python reference + unit test for the PRS scoring math in
``notebooks/01_score_prs.py``.

The Spark/Glow notebook can only run on a Databricks cluster, so this test
re-implements the *allele-orientation rule* — the part with real risk of a sign
bug — in plain Python and checks it against a hand-worked example. The reference
``effect_dosage`` / ``score_samples`` here mirror the notebook's logic exactly:

    effect_allele == alt  -> eff_dosage = alt_state
    effect_allele == ref  -> eff_dosage = 2 - alt_state
    allele mismatch       -> variant dropped
    alt_state < 0         -> genotype missing, dropped

Run: ``python -m pytest test_prs_scoring.py``  (no Spark/Glow needed).
"""
from __future__ import annotations


def effect_dosage(alt_state, effect_allele: str, ref: str, alt: str) -> float | None:
    """Dosage of the effect allele for one sample at one variant, or None to drop.

    ``alt_state`` is the per-sample alt-allele dosage — an integer 0/1/2 for hard
    genotype calls, OR a continuous float in [0, 2] for imputed dosage (DS/HDS).
    Missing is normalized to ``None`` upstream in the ingest step."""
    if alt_state is None:
        return None  # missing genotype/dosage
    if effect_allele == alt:
        return float(alt_state)
    if effect_allele == ref:
        return 2.0 - float(alt_state)
    return None  # effect allele matches neither ref nor alt -> drop


def score_samples(variants, sample_ids):
    """variants: list of dicts {ref, alt, effect_allele, weight, states: [per-sample alt dosage]}.
    Returns {sample_id: {"prs_raw": float, "n_variants_matched": int}}."""
    out = {s: {"prs_raw": 0.0, "n_variants_matched": 0} for s in sample_ids}
    for v in variants:
        for sid, state in zip(sample_ids, v["states"]):
            d = effect_dosage(state, v["effect_allele"], v["ref"], v["alt"])
            if d is None:
                continue
            out[sid]["prs_raw"] += d * v["weight"]
            out[sid]["n_variants_matched"] += 1
    return out


# --- worked example -------------------------------------------------------
SAMPLES = ["s1", "s2"]
VARIANTS = [
    # effect allele == ALT: eff_dosage == alt_state
    {"ref": "G", "alt": "A", "effect_allele": "A", "weight": 0.5, "states": [2, 0]},
    # effect allele == REF: eff_dosage == 2 - alt_state
    {"ref": "C", "alt": "T", "effect_allele": "C", "weight": 1.0, "states": [1, 2]},
    # allele mismatch -> dropped for everyone
    {"ref": "A", "alt": "G", "effect_allele": "X", "weight": 9.9, "states": [2, 2]},
    # missing genotype for s2 (null from ingest) -> dropped only for s2
    {"ref": "T", "alt": "C", "effect_allele": "C", "weight": 2.0, "states": [1, None]},
]


def test_effect_dosage_orientation():
    assert effect_dosage(2, "A", "G", "A") == 2.0          # alt, homozygous
    assert effect_dosage(1, "C", "C", "T") == 1.0          # ref, het -> 2-1
    assert effect_dosage(0, "C", "C", "T") == 2.0          # ref, hom-ref alt_state=0 -> 2
    assert effect_dosage(2, "X", "A", "G") is None         # mismatch
    assert effect_dosage(None, "C", "T", "C") is None      # missing (null)


def test_effect_dosage_continuous():
    # imputed dosage (DS/HDS) is continuous, not just 0/1/2
    assert abs(effect_dosage(0.7, "A", "G", "A") - 0.7) < 1e-9        # alt, dosage as-is
    assert abs(effect_dosage(1.3, "C", "C", "T") - 0.7) < 1e-9        # ref, 2 - 1.3


def test_score_samples_worked_example():
    res = score_samples(VARIANTS, SAMPLES)
    # s1: 0.5*2 + 1.0*(2-1) + (mismatch skip) + 2.0*(2-1) = 1.0 + 1.0 + 2.0 = 4.0 over 3 variants
    assert res["s1"]["prs_raw"] == 4.0
    assert res["s1"]["n_variants_matched"] == 3
    # s2: 0.5*0 + 1.0*(2-2) + (mismatch skip) + (missing skip) = 0.0 over 2 variants
    assert res["s2"]["prs_raw"] == 0.0
    assert res["s2"]["n_variants_matched"] == 2


if __name__ == "__main__":
    test_effect_dosage_orientation()
    test_score_samples_worked_example()
    print("PRS scoring reference tests passed.")
