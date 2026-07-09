"""FRAPOSA ancestry — panel PCA basis fit (one-time) + per-sample OADP projection (distributable).

Ported faithfully from `function_pca` (the validated single-node engine):
  - `oadp.py`            → svd_online / procrustes / procrustes_diffdim (verbatim)
  - `refit._run_fraposa_oadp_core` → split at its natural seam into `fit_panel_basis`
                           (standardize + SVD + loadings — the one-time panel work) and
                           `project_member` (standardize a new sample with the panel's
                           mean/std, OADP online-SVD, Procrustes-align back — cheap, runs
                           once per sample so it distributes across samples)
  - `ancestry.classify_ancestry` → RF (MSP) + Mahalanobis outlier (verbatim)

`fit_panel_basis(X_panel) + project_member(basis, Xu)` reproduces `_run_fraposa_oadp_core`
bit-for-bit (same ops, same order) — see the parity test. The point of the split is that the
3942×3942 eigendecomposition happens ONCE (fixed reference basis, à la pgsc_calc) and every
new sample is just an online-SVD update against the broadcast basis.

Pure numpy + sklearn/scipy for classify. No plink2, no fraposa_pgsc.
"""

from __future__ import annotations

import numpy as np


# ── OADP primitives (verbatim from function_pca/oadp.py) ────────────────────────

def svd_online(U1: np.ndarray, d1: np.ndarray, V1: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Online SVD update: append b (length p) to the data, return (s_aug, V_aug).

    U1: (p, k) left singular vectors; d1: (k,) singular values; V1: (n, k) right
    singular vectors; b: (p,) new column, standardised the same way as X1.
    Returns (s_aug, V_aug) for [X1 | b] — shapes (k+1,), (n+1, k+1).
    Direct port of fraposa.svd_online (Zhang 2020)."""
    n, k = V1.shape
    assert U1.shape[1] == k
    assert d1.shape[0] == k

    b = b.reshape((-1, 1))
    b_tilde = b - U1 @ (U1.T @ b)
    nrm = float(np.sqrt(np.sum(b_tilde ** 2)))
    if nrm < 1e-12:
        b_tilde = np.zeros_like(b_tilde)
    else:
        b_tilde = b_tilde / nrm

    R_top = np.concatenate([np.diag(d1), U1.T @ b], axis=1)
    R_bot = np.concatenate([np.zeros((1, k)), b_tilde.T @ b], axis=1)
    R = np.concatenate([R_top, R_bot], axis=0)
    _Ur, d2, R_Vt = np.linalg.svd(R, full_matrices=False)

    V_new = np.zeros((k + 1, n + 1))
    V_new[:k, :n] = V1.T
    V_new[k, n] = 1.0
    V_aug = (R_Vt @ V_new).T
    return d2, V_aug


def procrustes(Y: np.ndarray, X: np.ndarray) -> tuple[np.ndarray, float, np.ndarray]:
    """Find best (R, ρ, c) such that ρ X R + c ≈ Y. Same dim Y, X."""
    X = np.array(X, dtype=np.float64, copy=True)
    Y = np.array(Y, dtype=np.float64, copy=True)
    X_mean = X.mean(axis=0)
    Y_mean = Y.mean(axis=0)
    X -= X_mean
    Y -= Y_mean
    C = Y.T @ X
    U, s, Vt = np.linalg.svd(C, full_matrices=False)
    trXX = float(np.sum(X ** 2))
    trS = float(np.sum(s))
    R = Vt.T @ U.T
    rho = trS / max(trXX, 1e-12)
    c = Y_mean - rho * X_mean @ R
    return R, rho, c


def procrustes_diffdim(
    Y: np.ndarray, X: np.ndarray,
    *, n_iter_max: int = 10000, epsilon_min: float = 1e-6,
) -> tuple[np.ndarray, float, np.ndarray]:
    """Procrustes when X has more dims than Y (X.shape[1] > Y.shape[1])."""
    X = np.array(X, dtype=np.float64, copy=True)
    Y = np.array(Y, dtype=np.float64, copy=True)
    n_X, p_X = X.shape
    n_Y, p_Y = Y.shape
    assert n_X == n_Y
    assert p_X >= p_Y
    if p_X == p_Y:
        return procrustes(Y, X)
    Z = np.zeros((n_X, p_X - p_Y))
    R = np.eye(p_X)
    rho = 1.0
    c = np.zeros(p_X)
    for _ in range(n_iter_max):
        W = np.hstack([Y, Z])
        R, rho, c = procrustes(W, X)
        X_new = X @ R * rho + c
        Z_new = X_new[:, p_Y:]
        Z_new_centered = Z_new - Z_new.mean(axis=0)
        Z_diff = Z_new - Z
        denom = float(np.sum(Z_new_centered ** 2))
        if denom < 1e-12:
            break
        eps = float(np.sum(Z_diff ** 2) / denom)
        if eps < epsilon_min:
            break
        Z = Z_new
    return R, rho, c


# ── FRAPOSA standardize + eig (verbatim from function_pca/refit.py) ─────────────

def _fraposa_standardize(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-variant empirical mean and std (ddof=0). Modifies X in place; missing → 0.
    Matches FRAPOSA's standardize() exactly (np.std default ddof=0)."""
    is_miss = np.isnan(X)
    p = X.shape[0]
    mean = np.zeros(p, dtype=np.float64)
    std = np.zeros(p, dtype=np.float64)
    for i in range(p):
        row = X[i, :][~is_miss[i, :]]
        if row.size > 0:
            mean[i] = float(np.mean(row))
            std[i] = float(np.std(row))
    std[std == 0] = 1.0
    X -= mean.astype(np.float32).reshape(-1, 1)
    X /= std.astype(np.float32).reshape(-1, 1)
    X[is_miss] = 0.0
    return mean.reshape(-1, 1), std.reshape(-1, 1)


def _svd_eigcov(XTX: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Eigendecomposition of XTX → (s, V), s = sqrt(eigenvalues), descending."""
    ssq, V = np.linalg.eigh(XTX)
    s = np.sqrt(np.abs(ssq))
    return s[::-1], V.T[::-1].T


# ── The fit/project seam (factored from _run_fraposa_oadp_core) ─────────────────

def fit_panel_basis(X: np.ndarray, *, dim_ref: int = 4) -> dict:
    """One-time: fit the fixed FRAPOSA reference basis from the panel dose matrix.

    X: (n_var, n_panel) float32 ALT-dose, NaN for missing. Standardized in place.
    Returns a broadcast-safe basis dict — the panel eigendecomposition happens here
    and only here; `project_member` reuses it for every sample.
    """
    dim_stu = dim_ref * 2
    dim_online = dim_stu * 2

    mean, std = _fraposa_standardize(X)             # X standardized in place
    XTX = (X.T @ X).astype(np.float64)
    s, V = _svd_eigcov(XTX)
    s = s.astype(np.float64); V = V.astype(np.float64)
    U = X @ (V[:, :dim_online] / s[:dim_online])    # (n_var, dim_online)
    pcs_ref = V[:, :dim_ref] * s[:dim_ref]          # (n_panel, dim_ref)

    return {
        "U_on": U[:, :dim_online].astype(np.float64),   # (n_var, dim_online)
        "s_on": s[:dim_online],                          # (dim_online,)
        "V_on": V[:, :dim_online],                       # (n_panel, dim_online)
        "pcs_ref": pcs_ref,                              # (n_panel, dim_ref)
        "mean": mean,                                    # (n_var, 1)
        "std": std,                                      # (n_var, 1)
        "dim_ref": dim_ref,
        "dim_stu": dim_stu,
    }


def project_member(basis: dict, Xu: np.ndarray) -> np.ndarray:
    """Per-sample (distributable): OADP-project one sample onto the fixed basis.

    Xu: (n_var,) or (n_var, 1) float32/float64 ALT-dose, NaN for missing, in the
    SAME variant order as the panel `X` passed to `fit_panel_basis`. Returns the
    sample's PCs (dim_ref,). Standardizes with the panel's mean/std (missing → 0),
    then FRAPOSA OADP: online-SVD append + Procrustes-align back to the ref PCs.
    """
    mean, std = basis["mean"], basis["std"]
    dim_stu, dim_ref = basis["dim_stu"], basis["dim_ref"]

    Xu = np.asarray(Xu, dtype=np.float32).reshape(-1, 1)
    is_miss = np.isnan(Xu)
    Xu = (Xu - mean.astype(np.float32)) / std.astype(np.float32)
    Xu[is_miss] = 0.0
    b = Xu[:, 0].astype(np.float64)

    s_aug, V_aug = svd_online(basis["U_on"], basis["s_on"], basis["V_on"], b)
    s_aug = s_aug[:dim_stu]; V_aug = V_aug[:, :dim_stu]
    pcs_aug = V_aug * s_aug
    R, rho, c = procrustes_diffdim(basis["pcs_ref"], pcs_aug[:-1, :])
    return (pcs_aug[-1:, :] @ R * rho + c).flatten()[:dim_ref]


# ── Classification (verbatim from function_pca/ancestry.py) ─────────────────────

def classify_ancestry(
    panel_pcs: np.ndarray,           # (n_panel, k) full panel PCs
    panel_superpops: list,           # length n_panel — SuperPop labels
    user_pcs: np.ndarray,            # (k,) or (1, k) query PCs
    *, n_pcs: int = 5,
    panel_unrelated: np.ndarray | None = None,  # bool (n_panel,); None = all
) -> dict:
    """pgsc_calc's RF classifier (MSP = argmax) + Mahalanobis outlier diagnostic.

    Trains RandomForestClassifier(random_state=32) on the unrelated panel PCs and
    reports Mahalanobis_P_ALL for the sample. Returns a plain dict (broadcast-safe).
    """
    from scipy.stats import chi2
    from sklearn.covariance import EmpiricalCovariance
    from sklearn.ensemble import RandomForestClassifier

    if user_pcs.ndim == 1:
        user_pcs = user_pcs[None, :]
    X_panel = panel_pcs[:, :n_pcs]
    X_user = user_pcs[:, :n_pcs]

    train_mask = panel_unrelated if panel_unrelated is not None else np.ones(len(panel_superpops), dtype=bool)
    train_pops = np.asarray([s for s, m in zip(panel_superpops, train_mask) if m])
    train_pcs = X_panel[train_mask]

    cov_all = EmpiricalCovariance().fit(train_pcs)
    d2 = float(cov_all.mahalanobis(X_user)[0])
    p_all = float(chi2.sf(d2, n_pcs - 1))

    clf = RandomForestClassifier(random_state=32)
    clf.fit(train_pcs, train_pops)
    probs = clf.predict_proba(X_user)[0]
    pop_probs = {str(c): float(p) for c, p in zip(clf.classes_, probs)}
    msp = str(clf.classes_[int(np.argmax(probs))])

    return {
        "msp": msp,
        "rf_probs": pop_probs,
        "mahalanobis_d2": d2,
        "mahalanobis_p_all": p_all,
        "n_pcs_used": n_pcs,
    }


def fit_rf(panel_pcs: np.ndarray, panel_superpops: list, *, n_pcs: int = 5,
           panel_unrelated: np.ndarray | None = None):
    """Fit the pgsc_calc RF once (driver) so it can be broadcast + reused across
    samples via `predict_proba`. Returns (clf, EmpiricalCovariance, classes)."""
    from sklearn.covariance import EmpiricalCovariance
    from sklearn.ensemble import RandomForestClassifier

    train_mask = panel_unrelated if panel_unrelated is not None else np.ones(len(panel_superpops), dtype=bool)
    train_pops = np.asarray([s for s, m in zip(panel_superpops, train_mask) if m])
    train_pcs = panel_pcs[train_mask, :n_pcs]
    clf = RandomForestClassifier(random_state=32).fit(train_pcs, train_pops)
    cov = EmpiricalCovariance().fit(train_pcs)
    return clf, cov
