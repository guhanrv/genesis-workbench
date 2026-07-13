"""Pure-Python reference + unit test for the PRS scoring math in
``notebooks/05_score_prs.py``.

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


# --- z-calibration regression tests ---------------------------------------
# Two bugs were found scoring the 6×196 cohort at genome scale; both silently miscalibrated the
# panel-normalized z. These guard against reverting either. See notebooks/05_score_prs.py and
# ref_00_build_panel_stats.py (the panel is built under mean-imputation, missing→2·AF).
import pathlib

_SCORER = pathlib.Path(__file__).resolve().parent.parent / "notebooks" / "05_score_prs.py"


def test_scorer_default_is_mean_impute():
    """Bug (a): the panel z-reference is built under mean-imputation, so members must be scored the
    same way. A `drop` default computes member raw and panel mean under different missing policies →
    miscalibrated z. Guard the shipped default."""
    src = _SCORER.read_text()
    assert 'dbutils.widgets.text("missing_mode", "mean_impute"' in src, (
        "05_score_prs default missing_mode must stay mean_impute (matches the mean-imputed panel).")


def test_scorer_restricts_to_panel_matched_set():
    """Bug (b): the member must be scored over EXACTLY the panel-matched variant set (inner-join
    pgs_panel_afreq). A left-join lets the member sum off-panel variants the panel never scored →
    raw on a larger scale than the panel distribution → inflated z. Guard the inner join."""
    src = _SCORER.read_text()
    assert '.join(afreq, "variant_id", "inner")' in src, (
        "05_score_prs mean_impute path must inner-join afreq (restrict member to the panel-matched set).")


def score_mean_impute(weights, panel_af, member_dose):
    """Pure-Python reference of the scorer's mean_impute aggregate (the calibrated path).

    Sums over EXACTLY the panel-matched set (variant_ids in ``panel_af``): a covered variant uses its
    real dose; a missing panel-matched variant is imputed to ``2·AF`` (matching how the panel itself
    was built); an OFF-panel variant (not in ``panel_af``) is EXCLUDED — the panel has no distribution
    to standardize it against. ``weights``/``panel_af``/``member_dose`` are {variant_id: value}."""
    raw, matched = 0.0, 0
    for vid, w in weights.items():
        if vid not in panel_af:
            continue                                # off-panel → excluded
        if vid in member_dose:
            d = member_dose[vid]; matched += 1      # covered → real dose (counts as coverage)
        else:
            d = 2.0 * panel_af[vid]                 # missing panel variant → 2·AF (panel policy)
        raw += d * w
    return {"raw": raw, "n_variants_matched": matched}


def _buggy_score_left_join(weights, panel_af, member_dose):
    """The pre-fix behavior: off-panel variants the member covers ALSO contribute (left-join) →
    raw on a larger scale than the panel distribution."""
    raw = 0.0
    for vid, w in weights.items():
        if vid in member_dose:
            raw += member_dose[vid] * w
        elif vid in panel_af:
            raw += 2.0 * panel_af[vid] * w
    return raw


def test_mean_impute_excludes_offpanel_and_imputes_missing():
    weights = {"1:100:A:G": 1.0, "1:200:A:G": 1.0, "1:300:A:G": 50.0}  # 1:300 is heavy + OFF-panel
    panel_af = {"1:100:A:G": 0.5, "1:200:A:G": 0.5}                    # panel matched only the first two
    member = {"1:100:A:G": 2.0, "1:300:A:G": 2.0}                      # covers 1:100 & off-panel 1:300; 1:200 missing
    r = score_mean_impute(weights, panel_af, member)
    # 1:100 covered (2·1) + 1:200 missing→2·0.5·1 (=1) + 1:300 off-panel EXCLUDED  = 3.0
    assert r["raw"] == 3.0
    assert r["n_variants_matched"] == 1                                # only 1:100 is real coverage


def test_offpanel_inflation_regression():
    """The heavy off-panel variant would blow up z if included (bug b). Standardized against the
    panel distribution (built over the matched set only), the fixed raw stays bounded."""
    weights = {"1:100:A:G": 1.0, "1:200:A:G": 1.0, "1:300:A:G": 50.0}
    panel_af = {"1:100:A:G": 0.5, "1:200:A:G": 0.5}
    member = {"1:100:A:G": 2.0, "1:200:A:G": 2.0, "1:300:A:G": 2.0}
    panel_mean, panel_sd = 2.0, 1.0                                    # panel's dist over the 2 matched variants
    raw_fixed = score_mean_impute(weights, panel_af, member)["raw"]    # 2 + 2 = 4 (off-panel excluded)
    raw_buggy = _buggy_score_left_join(weights, panel_af, member)      # 4 + 100 = 104 (off-panel 50·2)
    z_fixed = (raw_fixed - panel_mean) / panel_sd
    z_buggy = (raw_buggy - panel_mean) / panel_sd
    assert abs(z_fixed) < 5, f"fixed z should be bounded, got {z_fixed}"
    assert abs(z_buggy) > 50, f"buggy (off-panel-included) z should be wildly inflated, got {z_buggy}"


if __name__ == "__main__":
    test_effect_dosage_orientation()
    test_effect_dosage_continuous()
    test_score_samples_worked_example()
    test_scorer_default_is_mean_impute()
    test_scorer_restricts_to_panel_matched_set()
    test_mean_impute_excludes_offpanel_and_imputes_missing()
    test_offpanel_inflation_regression()
    print("PRS scoring reference + z-calibration regression tests passed.")
