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


def randomized_svd(M: np.ndarray, k: int, n_oversample: int = 10, n_power_iter: int = 2,
                   seed: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Halko (2011) randomized SVD — the pure-NumPy reference for ``lib/pca_fit``'s ``randomized``
    backend. Returns approximate top-k ``(U (m×k), s (k,), Vt (k×n))`` of ``M`` without ever forming
    the n×n Gram: sketch the range with a random Ω, refine with power iterations, then an exact SVD of
    the small projected ``B = QᵀM``. The Spark backend is this same recipe with the matvecs distributed."""
    rng = np.random.default_rng(seed)
    m, n = M.shape
    ell = min(k + n_oversample, min(m, n))
    Q, _ = np.linalg.qr(M @ rng.standard_normal((n, ell)))       # sketch range(M)
    for _ in range(n_power_iter):                                # power iterations sharpen the spectrum
        Q, _ = np.linalg.qr(M @ (M.T @ Q))
    Ub, s, Vt = np.linalg.svd(Q.T @ M, full_matrices=False)      # exact SVD of the small B = QᵀM
    return (Q @ Ub)[:, :k], s[:k], Vt[:k]


def pca_top_components_randomized(matrix: np.ndarray, k: int) -> np.ndarray:
    """samples × variants → samples × k PC scores via randomized SVD on centered data (mirrors
    ``pca_top_components`` but with the matrix-free method the ``randomized`` backend uses)."""
    centered = matrix - matrix.mean(axis=0, keepdims=True)
    _, _, vt = randomized_svd(centered, k)
    return centered @ vt.T


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


def _low_rank_with_gap(seed: int = 0):
    """A matrix with a planted rank-6 structure and a clean spectral gap to a low noise floor, so the
    top-6 singular vectors are individually well-defined (unlike the near-degenerate two-population
    spectrum). This is the regime where RSVD must reproduce exact SVD axis-for-axis."""
    rng = np.random.default_rng(seed)
    m, n = 200, 500
    qa, _ = np.linalg.qr(rng.standard_normal((m, 6)))       # orthonormal left factors
    qb, _ = np.linalg.qr(rng.standard_normal((n, 6)))       # orthonormal right factors
    s = np.array([60.0, 45.0, 32.0, 22.0, 15.0, 9.0])       # separated singular values ≫ noise floor
    return qa @ np.diag(s) @ qb.T + 0.05 * rng.standard_normal((m, n))


def test_randomized_svd_matches_exact():
    """With a real spectral gap, randomized SVD must recover exact SVD's top-k singular values and
    each principal axis (up to sign) — the equivalence that lets it replace the driver eigh."""
    M = _low_rank_with_gap()
    k = 6
    _, s_r, vt_r = randomized_svd(M, k)
    _, s_x, vt_x = np.linalg.svd(M, full_matrices=False)
    s_x, vt_x = s_x[:k], vt_x[:k]
    assert np.allclose(s_r, s_x, rtol=1e-2), (s_r, s_x)      # top-k singular values match
    cos = np.abs(np.sum(vt_r * vt_x, axis=1))               # each axis matches up to sign
    assert (cos > 0.99).all(), cos


def test_randomized_reconstruction_matches_exact():
    """The rank-k reconstruction is basis-independent (sign- and rotation-invariant), so it is the
    cleanest statement of equivalence — but only well-posed with a gap at the k/(k+1) boundary (the
    two-population spectrum has none past PC1). On the gapped matrix RSVD's rank-k reconstruction must
    match exact SVD's, which is exactly the low-rank approximation PCA consumes."""
    M = _low_rank_with_gap()
    k = 6
    ur, sr, vtr = randomized_svd(M, k)
    ux, sx, vtx = np.linalg.svd(M, full_matrices=False)
    rec_r = (ur * sr) @ vtr
    rec_x = (ux[:, :k] * sx[:k]) @ vtx[:k]
    rel = np.linalg.norm(rec_r - rec_x) / np.linalg.norm(rec_x)
    assert rel < 1e-2, rel


def test_pc1_separates_two_populations_randomized():
    """Same population-separation guarantee as the exact PCA, through the matrix-free RSVD path."""
    dosages, labels = _synthetic_two_populations()
    pcs = pca_top_components_randomized(mean_impute(dosages), k=2)
    pc1 = pcs[:, 0]
    mean_a, mean_b = pc1[labels == 0].mean(), pc1[labels == 1].mean()
    spread = pc1.std()
    assert abs(mean_a - mean_b) > spread, (mean_a, mean_b, spread)
    thresh = (mean_a + mean_b) / 2
    pred = (pc1 > thresh).astype(int)
    acc = max((pred == labels).mean(), (pred != labels).mean())
    assert acc > 0.9, acc


if __name__ == "__main__":
    test_mean_impute_removes_missing()
    test_pc1_separates_two_populations()
    test_randomized_svd_matches_exact()
    test_randomized_reconstruction_matches_exact()
    test_pc1_separates_two_populations_randomized()
    print("PCA reference tests passed.")
