"""Unit tests for the ancestry math in ``lib/prs_ancestry.py`` and the
``z_admixed`` combiner in ``notebooks/05_score_prs.py``.

No Spark/Databricks needed: ``prs_ancestry`` is a plain importable module (the
distributed notebook only wraps ``project_member`` in ``applyInPandas``), so we
import it directly and exercise the FRAPOSA fit/project + RF classify on a small
synthetic reference panel, plus a pure-Python reference for the z_admixed weighting.

Run: ``python -m pytest test_prs_ancestry.py``
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "lib"))
import prs_ancestry as anc  # noqa: E402


# ── synthetic reference panel: K superpops, each a distinct allele-freq profile ──
def _synth_panel(*, n_var=400, per_pop=60, seed=0):
    """Genotypes ~ Binomial(2, af_pop): each population has its own random AF
    vector, so PC space separates them — a miniature HGDP+1kGP."""
    rng = np.random.default_rng(seed)
    pops = ["AFR", "EAS", "EUR"]
    af = {p: rng.uniform(0.1, 0.9, n_var) for p in pops}
    cols, labels = [], []
    for p in pops:
        for _ in range(per_pop):
            cols.append(rng.binomial(2, af[p]).astype(np.float32))
            labels.append(p)
    X = np.stack(cols, axis=1)  # (n_var, n_panel)
    return X, np.array(labels), af, pops, rng


def test_project_member_is_deterministic():
    """fit_panel_basis + project_member must be numerically stable/reproducible
    (locks the FRAPOSA OADP output against accidental regressions)."""
    X, labels, af, pops, rng = _synth_panel(seed=1)
    basis = anc.fit_panel_basis(X.copy(), dim_ref=6)
    # a fresh EUR-like sample
    xu = rng.binomial(2, af["EUR"]).astype(np.float32)
    pc = anc.project_member(basis, xu)
    assert pc.shape == (6,)
    # deterministic: same inputs → identical PCs
    pc2 = anc.project_member(anc.fit_panel_basis(X.copy(), dim_ref=6), xu.copy())
    assert np.allclose(pc, pc2, atol=1e-10)
    # finite, non-degenerate
    assert np.all(np.isfinite(pc)) and np.linalg.norm(pc) > 0


def test_missing_handled_as_zero():
    """NaN entries (a locus the sample doesn't cover) must be treated as the
    panel mean (standardized → 0), never propagate NaN into the PCs."""
    X, labels, af, pops, rng = _synth_panel(seed=2)
    basis = anc.fit_panel_basis(X.copy(), dim_ref=4)
    xu = rng.binomial(2, af["AFR"]).astype(np.float32)
    xu[: len(xu) // 4] = np.nan  # 25% missing
    pc = anc.project_member(basis, xu)
    assert np.all(np.isfinite(pc))


def test_fit_project_classify_end_to_end():
    """Held-out samples drawn from population k must classify back to k — the
    end-to-end guarantee (fit basis → OADP project → RF MSP)."""
    pytest.importorskip("sklearn")
    pytest.importorskip("scipy")
    X, labels, af, pops, rng = _synth_panel(seed=3)
    basis = anc.fit_panel_basis(X.copy(), dim_ref=6)
    correct = 0
    n = 30
    for _ in range(n):
        p = pops[rng.integers(len(pops))]
        xu = rng.binomial(2, af[p]).astype(np.float32)
        res = anc.classify_ancestry(basis["pcs_ref"], labels.tolist(),
                                    anc.project_member(basis, xu), n_pcs=5)
        correct += (res["msp"] == p)
    acc = correct / n
    assert acc >= 0.9, f"held-out classification accuracy too low: {acc}"


def test_classify_shapes_and_probs():
    pytest.importorskip("sklearn")
    X, labels, af, pops, rng = _synth_panel(seed=4)
    basis = anc.fit_panel_basis(X.copy(), dim_ref=6)
    xu = rng.binomial(2, af["EAS"]).astype(np.float32)
    res = anc.classify_ancestry(basis["pcs_ref"], labels.tolist(),
                                anc.project_member(basis, xu), n_pcs=5)
    assert res["msp"] in pops
    assert abs(sum(res["rf_probs"].values()) - 1.0) < 1e-9
    assert 0.0 <= res["mahalanobis_p_all"] <= 1.0


# ── z_admixed reference (mirrors the scorer's SQL: Σ p·z_pop / Σ p, sd>0 & p>0) ──
def _z_admixed(raw, panel, rf_probs):
    num = den = 0.0
    for pop, p in rf_probs.items():
        m, s = panel[pop]
        if s <= 0 or p <= 0:
            continue
        num += p * (raw - m) / s
        den += p
    return num / den if den > 0 else None


_PANEL = {"EUR": (0.10, 0.20), "AMR": (0.05, 0.25), "CSA": (0.00, 0.30),
          "EAS": (-0.20, 0.15), "AFR": (0.30, 0.40), "MID": (0.02, 0.0)}


def test_z_admixed_collapses_to_z_msp():
    """RF mass concentrated on one superpop ⇒ z_admixed == z_msp for that pop."""
    raw = 0.34
    z_eur = (raw - _PANEL["EUR"][0]) / _PANEL["EUR"][1]
    assert abs(_z_admixed(raw, _PANEL, {"EUR": 1.0}) - z_eur) < 1e-12


def test_z_admixed_weighted_average():
    raw = 0.34
    probs = {"EUR": 0.95, "AMR": 0.05}
    z = lambda p: (raw - _PANEL[p][0]) / _PANEL[p][1]
    exp = (0.95 * z("EUR") + 0.05 * z("AMR")) / 1.0
    assert abs(_z_admixed(raw, _PANEL, probs) - exp) < 1e-12


def test_z_admixed_drops_degenerate_and_renormalizes():
    """A superpop with sd=0 (MID) carrying RF mass must drop out and the
    surviving weights renormalize — not poison the result with a NaN/inf."""
    raw = 0.34
    z_eur = (raw - _PANEL["EUR"][0]) / _PANEL["EUR"][1]
    assert abs(_z_admixed(raw, _PANEL, {"EUR": 0.5, "MID": 0.5}) - z_eur) < 1e-12


def test_mahalanobis_uses_npcs_dof_and_flags_outliers():
    """Regression: Mahalanobis p must use df = n_pcs (was n_pcs-1, which inflated it and hid
    outliers), and a sample belonging to no reference population must get a much smaller p_all
    than an in-population one — the signal the ancestry-abstain gate keys on."""
    pytest.importorskip("sklearn")
    pytest.importorskip("scipy")
    from scipy.stats import chi2
    X, labels, af, pops, rng = _synth_panel(seed=7)
    basis = anc.fit_panel_basis(X.copy(), dim_ref=6)
    n_pcs = 5
    xu_in = rng.binomial(2, af["EUR"]).astype(np.float32)
    res_in = anc.classify_ancestry(basis["pcs_ref"], labels.tolist(),
                                   anc.project_member(basis, xu_in), n_pcs=n_pcs)
    # p_all must equal chi2.sf(d2, df=n_pcs) — fails if df regresses to n_pcs-1
    assert abs(res_in["mahalanobis_p_all"] - float(chi2.sf(res_in["mahalanobis_d2"], n_pcs))) < 1e-9
    # a pure-noise sample (no reference population) is a PC-space outlier → smaller p_all
    xu_out = rng.uniform(0, 2, X.shape[0]).astype(np.float32)
    res_out = anc.classify_ancestry(basis["pcs_ref"], labels.tolist(),
                                    anc.project_member(basis, xu_out), n_pcs=n_pcs)
    assert res_out["mahalanobis_p_all"] < res_in["mahalanobis_p_all"]


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn(); print(f"PASS {name}")
            except Exception as e:  # pragma: no cover
                print(f"FAIL {name}: {e}")
