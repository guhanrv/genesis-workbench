"""Pure-NumPy reference + test for the PCA approach in ``notebooks/01_compute_pca.py``.

The Spark notebook only runs on a cluster, so this validates the *method* the
notebook implements — mean-impute missing dosages, then PCA — by showing that on a
synthetic two-population genotype matrix, **PC1 separates the populations** (the
whole point of ancestry PCA). Mirrors the notebook's imputation + centering.

Run: ``python -m pytest test_pca_reference.py``  (needs only numpy).
"""
from __future__ import annotations

import numpy as np


def mean_impute(dosages: np.ndarray) -> np.ndarray:
    """dosages: samples × variants, missing encoded as -1 (as glow.genotype_states does).
    Replace missing with the per-variant mean of observed calls (notebook's rule)."""
    out = dosages.astype(float).copy()
    for j in range(out.shape[1]):
        col = out[:, j]
        observed = col[col >= 0]
        m = observed.mean() if observed.size else 0.0
        col[col < 0] = m
        out[:, j] = col
    return out


def pca_top_components(matrix: np.ndarray, k: int) -> np.ndarray:
    """samples × variants → samples × k principal-component scores (SVD on centered data)."""
    centered = matrix - matrix.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    return centered @ vt[:k].T


def _synthetic_two_populations(seed: int = 0):
    rng = np.random.default_rng(seed)
    n_per, n_snps = 30, 200
    # Two populations with different alt-allele frequencies per SNP.
    afs_a = rng.uniform(0.05, 0.5, n_snps)
    afs_b = np.clip(afs_a + rng.uniform(-0.4, 0.4, n_snps), 0.01, 0.99)
    pop_a = rng.binomial(2, afs_a, size=(n_per, n_snps))
    pop_b = rng.binomial(2, afs_b, size=(n_per, n_snps))
    dosages = np.vstack([pop_a, pop_b]).astype(float)
    labels = np.array([0] * n_per + [1] * n_per)
    # sprinkle missing calls (-1)
    miss = rng.random(dosages.shape) < 0.02
    dosages[miss] = -1
    return dosages, labels


def test_mean_impute_removes_missing():
    d = np.array([[2, -1], [0, 1], [-1, 1]], dtype=float)
    imp = mean_impute(d)
    assert (imp >= 0).all()
    assert imp[2, 0] == 1.0   # mean of observed {2,0}
    assert imp[0, 1] == 1.0   # mean of observed {1,1}


def test_pc1_separates_two_populations():
    dosages, labels = _synthetic_two_populations()
    pcs = pca_top_components(mean_impute(dosages), k=2)
    pc1 = pcs[:, 0]
    mean_a, mean_b = pc1[labels == 0].mean(), pc1[labels == 1].mean()
    # PC1 means of the two populations must be well separated relative to spread
    spread = pc1.std()
    assert abs(mean_a - mean_b) > spread, (mean_a, mean_b, spread)
    # a sign-threshold at the midpoint classifies populations near-perfectly
    thresh = (mean_a + mean_b) / 2
    pred = (pc1 > thresh).astype(int)
    acc = max((pred == labels).mean(), (pred != labels).mean())
    assert acc > 0.9, acc


if __name__ == "__main__":
    test_mean_impute_removes_missing()
    test_pc1_separates_two_populations()
    print("PCA reference tests passed.")
