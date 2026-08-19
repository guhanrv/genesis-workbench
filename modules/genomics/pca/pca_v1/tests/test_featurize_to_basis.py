"""Off-cluster unit tests for ``lib/featurize_to_basis.py`` (design §9.1).

Proves the Design-A guarantees the model relies on: order-invariance, correct fill per
``impute_mode``, orientation flip, non-basis-locus drop, and the fit/serve compatibility
that lets the same transform run in the panel fit and at serving. Pure numpy — no Spark.

Run: ``python -m pytest test_featurize_to_basis.py``
"""
from __future__ import annotations

import os
import random
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "lib"))
import featurize_to_basis as ftb  # noqa: E402


def _basis(n=20, *, effect_is_alt=True, seed=0):
    """A small slot-ordered basis dict (effect_allele == alt by default, the panel-built
    convention). Frozen mean/std chosen so standardization is easy to check by hand."""
    rng = np.random.default_rng(seed)
    chrom = np.array([str((i % 22) + 1) for i in range(n)], dtype=object)
    pos = np.arange(1, n + 1, dtype=np.int64)
    ref = np.array(["A"] * n, dtype=object)
    alt = np.array(["G"] * n, dtype=object)
    vids = np.array([f"{chrom[i]}:{pos[i]}:A:G" for i in range(n)], dtype=object)
    effect = alt.copy() if effect_is_alt else ref.copy()
    other = ref.copy() if effect_is_alt else alt.copy()
    mean = rng.uniform(0.2, 1.8, n)
    std = rng.uniform(0.3, 0.9, n)
    return {"variant_id": vids, "effect_allele": effect, "other_allele": other,
            "ref": ref, "alt": alt, "mean": mean, "std": std}, vids, mean, std


def test_order_invariance():
    """Same observed pairs in any order → identical vector (Design A's core guarantee)."""
    basis, vids, mean, std = _basis(seed=1)
    pairs = [(vids[i], float(i % 3)) for i in range(0, 20, 2)]
    z1, n1 = ftb.featurize_to_basis(pairs, dict(basis), impute_mode="zero")
    shuf = pairs[:]
    random.Random(42).shuffle(shuf)
    z2, n2 = ftb.featurize_to_basis(shuf, dict(basis), impute_mode="zero")
    assert np.array_equal(z1, z2) and n1 == n2 == len(pairs)


def test_non_basis_loci_dropped():
    """Observed loci not in the basis are ignored and don't count toward coverage."""
    basis, vids, mean, std = _basis(seed=2)
    pairs = [(vids[0], 2.0), ("99:12345:A:G", 1.0), (vids[5], 0.0)]
    z, n_cov = ftb.featurize_to_basis(pairs, dict(basis), impute_mode="at_mean")
    assert n_cov == 2  # only the two real basis loci


def test_missing_slots_filled_zero_vs_at_mean():
    """Unobserved slots: 'zero' → (0-mean)/std; 'at_mean' → 0. Observed slots identical."""
    basis, vids, mean, std = _basis(seed=3)
    obs_idx = [0, 1, 2, 3, 4]
    pairs = [(vids[i], 2.0) for i in obs_idx]
    z_zero, _ = ftb.featurize_to_basis(pairs, dict(basis), impute_mode="zero")
    z_mean, _ = ftb.featurize_to_basis(pairs, dict(basis), impute_mode="at_mean")
    miss = [i for i in range(20) if i not in obs_idx]
    assert np.allclose(z_zero[miss], (0.0 - mean[miss]) / std[miss])
    assert np.allclose(z_mean[miss], 0.0)
    assert np.array_equal(z_zero[obs_idx], z_mean[obs_idx])
    # observed slot value is standardized effect dose
    assert np.allclose(z_zero[obs_idx], (2.0 - mean[obs_idx]) / std[obs_idx])


def test_orientation_flip_when_effect_is_ref():
    """When effect_allele == ref, ALT dose d must be oriented to 2 - d before standardizing."""
    basis, vids, mean, std = _basis(effect_is_alt=False, seed=4)
    d = 1.4
    z, n_cov = ftb.featurize_to_basis([(vids[0], d)], dict(basis), impute_mode="at_mean")
    assert n_cov == 1
    assert np.isclose(z[0], ((2.0 - d) - mean[0]) / std[0])


def test_effect_is_alt_no_flip():
    basis, vids, mean, std = _basis(effect_is_alt=True, seed=5)
    d = 1.4
    z, _ = ftb.featurize_to_basis([(vids[0], d)], dict(basis), impute_mode="at_mean")
    assert np.isclose(z[0], (d - mean[0]) / std[0])


def test_full_coverage_matches_direct_standardize():
    """A fully-observed sample featurizes to exactly (dose - mean)/std at every slot."""
    basis, vids, mean, std = _basis(seed=6)
    dose = np.array([float(i % 3) for i in range(20)])
    pairs = [(vids[i], dose[i]) for i in range(20)]
    z, n_cov = ftb.featurize_to_basis(pairs, dict(basis), impute_mode="zero")
    assert n_cov == 20
    assert np.allclose(z, (dose - mean) / std)


def test_bad_impute_mode_raises():
    basis, vids, mean, std = _basis(seed=7)
    with pytest.raises(ValueError):
        ftb.featurize_to_basis([(vids[0], 1.0)], dict(basis), impute_mode="bogus")


def test_fit_serve_parity_against_project_member_standardize():
    """The load-bearing test (design §9.2): featurize_to_basis(at_mean) reproduces the
    standardized vector fraposa.project_member computes internally, so fit and serve use the
    SAME transform. (Tolerance ~1e-6: project_member casts mean/std to float32; featurize
    standardizes in float64 — a Phase-2 dtype-pin note, not a logic difference.)"""
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "lib"))
    import fraposa as fr
    rng = np.random.default_rng(11)
    n_var, n_panel = 300, 90
    af = rng.uniform(0.1, 0.9, n_var)
    X = rng.binomial(2, af.reshape(-1, 1), size=(n_var, n_panel)).astype(np.float32)
    basis = fr.fit_panel_basis(X.copy(), dim_ref=6)

    chrom = np.array([str((i % 22) + 1) for i in range(n_var)], dtype=object)
    pos = np.arange(1, n_var + 1, dtype=np.int64)
    alt = np.array(["G"] * n_var, dtype=object)
    ref = np.array(["A"] * n_var, dtype=object)
    vids = np.array([f"{chrom[i]}:{pos[i]}:A:G" for i in range(n_var)], dtype=object)
    fbasis = {"variant_id": vids, "effect_allele": alt, "ref": ref, "alt": alt,
              "mean": basis["mean"], "std": basis["std"]}

    xu = rng.binomial(2, af).astype(np.float32)
    miss = set(rng.choice(n_var, size=40, replace=False).tolist())
    # project_member's internal standardized vector b
    xu_nan = xu.astype(np.float64).copy()
    for i in miss:
        xu_nan[i] = np.nan
    m32, s32 = basis["mean"].astype(np.float32), basis["std"].astype(np.float32)
    Xu = xu_nan.astype(np.float32).reshape(-1, 1)
    b_internal = ((Xu - m32) / s32)
    b_internal[np.isnan(Xu)] = 0.0
    b_internal = b_internal[:, 0].astype(np.float64)

    pairs = [(vids[i], float(xu[i])) for i in range(n_var) if i not in miss]
    z, n_cov = ftb.featurize_to_basis(pairs, dict(fbasis), impute_mode="at_mean")
    assert n_cov == n_var - 40
    assert np.max(np.abs(z - b_internal)) < 1e-6


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn(); print(f"PASS {name}")
            except Exception as e:  # pragma: no cover
                print(f"FAIL {name}: {e}")
