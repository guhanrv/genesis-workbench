"""Off-cluster unit tests for the consolidated FRAPOSA lib (``lib/fraposa.py``).

Ports the historical ``prs/tests/test_prs_ancestry.py`` ancestry-math coverage onto the
single consolidated module and adds the fit-core invariants. No Spark/Databricks: everything
here is pure numpy (+ sklearn/scipy for the classifier), so ``python -m pytest`` runs it.

Run: ``python -m pytest test_fraposa.py``
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "lib"))
import fraposa as fr  # noqa: E402


# ── synthetic reference panel: K superpops, each a distinct allele-freq profile ──
def _synth_panel(*, n_var=400, per_pop=60, seed=0):
    """Genotypes ~ Binomial(2, af_pop): each population has its own random AF vector,
    so PC space separates them — a miniature HGDP+1kGP."""
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
    """fit_panel_basis + project_member must be numerically stable / reproducible."""
    X, labels, af, pops, rng = _synth_panel(seed=1)
    basis = fr.fit_panel_basis(X.copy(), dim_ref=6)
    xu = rng.binomial(2, af["EUR"]).astype(np.float32)
    pc = fr.project_member(basis, xu)
    assert pc.shape == (6,)
    pc2 = fr.project_member(fr.fit_panel_basis(X.copy(), dim_ref=6), xu.copy())
    assert np.allclose(pc, pc2, atol=1e-10)
    assert np.all(np.isfinite(pc)) and np.linalg.norm(pc) > 0


def test_missing_handled_as_zero():
    """NaN entries (a locus the sample doesn't cover) → panel mean (standardized 0),
    never propagate NaN into the PCs."""
    X, labels, af, pops, rng = _synth_panel(seed=2)
    basis = fr.fit_panel_basis(X.copy(), dim_ref=4)
    xu = rng.binomial(2, af["AFR"]).astype(np.float32)
    xu[: len(xu) // 4] = np.nan
    pc = fr.project_member(basis, xu)
    assert np.all(np.isfinite(pc))


def test_fit_project_classify_end_to_end():
    """Held-out samples drawn from population k must classify back to k."""
    pytest.importorskip("sklearn")
    pytest.importorskip("scipy")
    X, labels, af, pops, rng = _synth_panel(seed=3)
    basis = fr.fit_panel_basis(X.copy(), dim_ref=6)
    correct = 0
    n = 30
    for _ in range(n):
        p = pops[rng.integers(len(pops))]
        xu = rng.binomial(2, af[p]).astype(np.float32)
        res = fr.classify_ancestry(basis["pcs_ref"], labels.tolist(),
                                   fr.project_member(basis, xu), n_pcs=5)
        correct += (res["msp"] == p)
    assert correct / n >= 0.9, f"held-out classification accuracy too low: {correct / n}"


def test_classify_shapes_and_probs():
    pytest.importorskip("sklearn")
    X, labels, af, pops, rng = _synth_panel(seed=4)
    basis = fr.fit_panel_basis(X.copy(), dim_ref=6)
    xu = rng.binomial(2, af["EAS"]).astype(np.float32)
    res = fr.classify_ancestry(basis["pcs_ref"], labels.tolist(),
                               fr.project_member(basis, xu), n_pcs=5)
    assert res["msp"] in pops
    assert abs(sum(res["rf_probs"].values()) - 1.0) < 1e-9
    assert 0.0 <= res["mahalanobis_p_all"] <= 1.0


def test_mahalanobis_uses_npcs_dof_and_flags_outliers():
    """Mahalanobis p must use df = n_pcs (not n_pcs-1), and an out-of-panel sample must get
    a much smaller p_all than an in-population one — the signal the abstain gate keys on."""
    pytest.importorskip("sklearn")
    pytest.importorskip("scipy")
    from scipy.stats import chi2
    X, labels, af, pops, rng = _synth_panel(seed=7)
    basis = fr.fit_panel_basis(X.copy(), dim_ref=6)
    n_pcs = 5
    xu_in = rng.binomial(2, af["EUR"]).astype(np.float32)
    res_in = fr.classify_ancestry(basis["pcs_ref"], labels.tolist(),
                                  fr.project_member(basis, xu_in), n_pcs=n_pcs)
    assert abs(res_in["mahalanobis_p_all"] - float(chi2.sf(res_in["mahalanobis_d2"], n_pcs))) < 1e-9
    # A genuinely out-of-panel sample must score MORE extreme (smaller p_all) than an
    # in-population one. The original prs test drew the "outlier" as rng.uniform(0,2), but
    # under unpinned numpy/sklearn/scipy that draw can land inside the panel cloud and the
    # assertion flips (reproduced against the ORIGINAL lib too — an env/version artifact, not
    # a logic change; see design §9 on pinning). Use an unambiguously far sample: push every
    # locus to a fixed extreme so its PCs sit well outside every superpop's Gaussian.
    xu_out = np.full(X.shape[0], 2.0, dtype=np.float32)
    xu_out[::2] = 0.0
    res_out = fr.classify_ancestry(basis["pcs_ref"], labels.tolist(),
                                   fr.project_member(basis, xu_out), n_pcs=n_pcs)
    assert res_out["mahalanobis_p_all"] < res_in["mahalanobis_p_all"]


# ── fit-core invariants (the pca_fit-side contract fraposa.fit_basis now owns) ──

def test_fit_basis_shapes_and_dims():
    """fit_basis returns slot/sample-aligned arrays at the requested dims."""
    X, labels, af, pops, rng = _synth_panel(seed=8)
    n_var, n_panel = X.shape
    b = fr.fit_basis(X.copy(), dim_ref=6, dim_online=24)
    assert b["U_on"].shape == (n_var, 24)
    assert b["s_on"].shape == (24,)
    assert b["V_on"].shape == (n_panel, 24)
    assert b["pcs_ref"].shape == (n_panel, 6)
    assert b["mean"].shape == (n_var, 1) and b["std"].shape == (n_var, 1)
    assert np.all(np.isfinite(b["U_on"])) and np.all(np.isfinite(b["pcs_ref"]))


def test_canonicalize_signs_is_gauge_only():
    """Sign-canonicalization must be a pure gauge choice: it changes per-component signs but
    preserves the fit (U·diag(s)·Vᵀ reconstruction and |values| unchanged), and is idempotent."""
    X, labels, af, pops, rng = _synth_panel(seed=9)
    b_raw = fr.fit_basis(X.copy(), dim_ref=6, dim_online=24, canonicalize=False)
    b_can = fr.fit_basis(X.copy(), dim_ref=6, dim_online=24, canonicalize=True)
    # each component differs only by a global ±1
    for j in range(b_raw["V_on"].shape[1]):
        col_raw, col_can = b_raw["V_on"][:, j], b_can["V_on"][:, j]
        assert np.allclose(col_raw, col_can) or np.allclose(col_raw, -col_can)
    # singular values (gauge-invariant) identical
    assert np.array_equal(b_raw["s_on"], b_can["s_on"])
    # canonical form: each component's largest-|entry| in V is positive
    V = b_can["V_on"]
    idx = np.argmax(np.abs(V), axis=0)
    assert np.all(V[idx, np.arange(V.shape[1])] >= 0)
    # idempotent
    b_can2 = fr.fit_basis(X.copy(), dim_ref=6, dim_online=24, canonicalize=True)
    assert np.array_equal(b_can["V_on"], b_can2["V_on"])


def test_fit_panel_basis_does_not_canonicalize():
    """fit_panel_basis preserves the historical prs-side contract: NO sign canonicalization
    (so it stays byte-identical to the original prs_ancestry.fit_panel_basis)."""
    X, labels, af, pops, rng = _synth_panel(seed=10)
    b_wrap = fr.fit_panel_basis(X.copy(), dim_ref=6)
    b_raw = fr.fit_basis(X.copy(), dim_ref=6, dim_online=6 * 4, canonicalize=False)
    for k in ("U_on", "s_on", "V_on", "pcs_ref", "mean", "std"):
        assert np.array_equal(b_wrap[k], b_raw[k]), k
    assert b_wrap["dim_stu"] == 12 and b_wrap["dim_ref"] == 6


def test_svd_online_and_procrustes_primitives_finite():
    """The OADP primitives run and return finite, correctly-shaped results."""
    X, labels, af, pops, rng = _synth_panel(seed=11)
    basis = fr.fit_panel_basis(X.copy(), dim_ref=4)
    b = ((rng.binomial(2, af["EUR"]).astype(np.float32).reshape(-1, 1)
          - basis["mean"]) / basis["std"])[:, 0]
    s_aug, V_aug = fr.svd_online(basis["U_on"], basis["s_on"], basis["V_on"], b)
    assert s_aug.shape[0] == basis["s_on"].shape[0] + 1
    assert V_aug.shape[0] == basis["V_on"].shape[0] + 1
    assert np.all(np.isfinite(s_aug)) and np.all(np.isfinite(V_aug))
    # procrustes recovers an exact rotation up to the fit
    Ymat = rng.standard_normal((20, 3))
    R, rho, c = fr.procrustes(Ymat, Ymat)
    assert np.allclose(rho * Ymat @ R + c, Ymat, atol=1e-8)


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn(); print(f"PASS {name}")
            except Exception as e:  # pragma: no cover
                print(f"FAIL {name}: {e}")
