# Vendored from pca_v1/lib/fraposa.py for the prs bundle addPyFile path. Canonical copy: pca_v1.
"""FRAPOSA ancestry — THE single fit + project + classify implementation.

This module is the one home for the whole FRAPOSA path:

  - the driver-side panel fit — standardize → Gram ``XᵀX`` → ``eigh`` → loadings
    ``U = X·(V/s)``, plus eigenvector sign-canonicalization.
  - the per-sample OADP projection (``svd_online`` / ``procrustes`` / ``project_member``).
  - the RF + Mahalanobis classifier (``classify_ancestry`` / ``fit_rf``).

Everything here is **pure numpy** (+ sklearn/scipy, imported lazily inside the classifier
functions only), so it imports and runs off-cluster and is unit-tested directly. The
distributed pieces — the Spark LD-prune and the ``distributed`` / ``randomized`` fit
backends — stay in ``pca_fit.py`` (they are Spark-specific and never run at inference);
``pca_fit``'s driver backend delegates its numpy fit to :func:`fit_basis` here so the
panel eigendecomposition has exactly one implementation.

**Provenance.** The primitives (``svd_online`` / ``procrustes`` / ``procrustes_diffdim``)
are ported verbatim from the validated reference OADP engine (Zhang 2020). The fit is the
FRAPOSA panel fit; ``project_member`` is its OADP online-SVD + Procrustes-back seam;
``classify_ancestry`` is pgsc_calc's RF (MSP = argmax) + Mahalanobis outlier diagnostic.
The math is unchanged from the two source copies. The one *intentional* unification: the
fit now canonicalizes eigenvector signs by default (:func:`canonicalize_signs`) — this was
already done for the production PCA basis in ``pca_fit`` and is a reproducibility-only
gauge choice (a consistent per-component sign flip of ``V``/``pcs_ref``/``U`` preserves the
fit exactly), so it does not change any distribution the classifier sees.

NOTE: a golden-vector parity test against the original reference output is still not
committed — the tests here cover determinism / finiteness / round-trip / classification.
Add the parity fixture before relying on byte-exact equivalence to the historical basis.

────────────────────────────────────────────────────────────────────────────────────────
THIRD-PARTY LICENSES
────────────────────────────────────────────────────────────────────────────────────────
This file is a DERIVED WORK. It contains code copied and adapted from two upstream
projects, whose licenses are reproduced below in full as those licenses require.

════════════════════════════════════════════════════════════════════════════════════════
1. FRAPOSA — https://github.com/daviddaiweizhang/fraposa (MIT License)
   Copyright (c) 2019 Daiwei Zhang
   Reference: Zhang D, Dey R, Lee S. "Fast and robust ancestry prediction using principal
   component analysis." Bioinformatics 36(11):3439-3446, 2020.

   Derived here: ``svd_online``, ``procrustes``, ``procrustes_diffdim`` (ported verbatim);
   ``standardize`` and ``eig_cov`` (the FRAPOSA panel fit); the OADP online-SVD +
   Procrustes-back seam in ``oadp_project`` / ``project_member``.

   MODIFICATIONS from upstream FRAPOSA:
     - The projection seam was split into ``oadp_project`` (takes an already-standardized
       vector) and ``project_member`` (standardizes, then delegates), so the registered
       model's ``predict`` composes the same featurization the fit uses. The math is
       unchanged; only the function boundary moved.
     - ``fit_basis`` optionally applies ``canonicalize_signs`` (default on), a
       reproducibility-only gauge fix absent upstream: eigenvector signs are arbitrary, so
       a rebuild could otherwise flip PC signs. A consistent per-component sign flip of
       V/pcs_ref/U preserves the fit exactly. ``fit_panel_basis`` keeps the upstream
       behaviour (no canonicalization).
     - Small numerical guards added (zero-norm and near-zero-denominator early exits).

   Permission is hereby granted, free of charge, to any person obtaining a copy of this
   software and associated documentation files (the "Software"), to deal in the Software
   without restriction, including without limitation the rights to use, copy, modify,
   merge, publish, distribute, sublicense, and/or sell copies of the Software, and to
   permit persons to whom the Software is furnished to do so, subject to the following
   conditions:

   The above copyright notice and this permission notice shall be included in all copies
   or substantial portions of the Software.

   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
   INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A
   PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
   HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF
   CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE
   OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

════════════════════════════════════════════════════════════════════════════════════════
2. pgsc_calc — https://github.com/PGScatalog/pgsc_calc (Apache License 2.0)
   Copyright (c) PGS Catalog / EMBL-EBI

   Derived here: ``classify_ancestry`` and ``fit_rf`` — the RandomForest
   most-similar-population classifier (MSP = argmax of ``predict_proba``,
   ``random_state=32``) plus the ``EmpiricalCovariance`` Mahalanobis outlier diagnostic.

   MODIFICATIONS from upstream pgsc_calc (noted per Apache-2.0 §4(b)):
     - The Mahalanobis p-value uses ``chi2.sf(d2, n_pcs)`` rather than ``n_pcs - 1``
       degrees of freedom. A Mahalanobis d² against a Gaussian fitted in n_pcs dimensions
       is ~χ²(n_pcs); the smaller dof inflates the survival function and makes outliers
       look less extreme than they are.
     - ``fit_rf`` was factored out so the forest and covariance are fit ONCE on the driver
       and reused across samples (upstream refits per invocation). Same estimator, same
       seed, same training mask.

   Licensed under the Apache License, Version 2.0 (the "License"); you may not use this
   file except in compliance with the License. You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software distributed under
   the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
   KIND, either express or implied. See the License for the specific language governing
   permissions and limitations under the License.
────────────────────────────────────────────────────────────────────────────────────────
"""
from __future__ import annotations

import numpy as np


# ── OADP primitives (verbatim from the reference OADP engine, Zhang 2020) ────────────

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


# ── FRAPOSA standardize + eig (verbatim from the reference FRAPOSA fit) ─────────────

def standardize(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-variant empirical mean and std (ddof=0). Modifies X in place; missing → 0.

    Matches FRAPOSA's ``standardize()`` exactly (``np.std`` default ddof=0). X is
    (n_var, n_samples), NaN = missing. Returns ``(mean, std)`` each shaped (n_var, 1);
    zero-variance rows get std=1 so the divide is a no-op instead of inf/nan.
"""
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


def eig_cov(XTX: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Eigendecomposition of the samples² Gram XTX → (s, V), s = sqrt(eigenvalues),
    both sorted DESCENDING."""
    ssq, V = np.linalg.eigh(XTX)
    s = np.sqrt(np.abs(ssq))
    return s[::-1], V.T[::-1].T


def canonicalize_signs(V_on, pcs_ref, U_on):
    """Fix the arbitrary per-component sign of the decomposition so the basis is
    REPRODUCIBLE across rebuilds / LAPACK-BLAS builds. eigh/svd only determine each
    eigenvector up to sign; without a convention a rebuilt basis can flip PC signs (and,
    in a near-degenerate block, the classifier's PC4/PC5 rotate). Convention: make each
    component's largest-|value| sample-space entry positive. V_on / pcs_ref / U_on are
    column-aligned per component, so flipping a component in all three together preserves
    the fit exactly (scores↔loadings stay mutually consistent) while pinning the sign.
    NOTE: this does NOT make backends bit-identical to each other or to the historical
    basis — it only makes each backend deterministic run-to-run.
    (Formerly ``pca_fit._canonicalize_signs``.)"""
    k = V_on.shape[1]
    idx = np.argmax(np.abs(V_on), axis=0)
    signs = np.sign(V_on[idx, np.arange(k)])
    signs[signs == 0] = 1.0
    V_on = V_on * signs
    U_on = U_on * signs
    pcs_ref = pcs_ref * signs[:pcs_ref.shape[1]]
    return V_on, pcs_ref, U_on


# ── The fit core (one implementation; pca_fit's driver backend delegates here) ──────

def fit_basis(X: np.ndarray, *, dim_ref: int, dim_online: int, canonicalize: bool = True) -> dict:
    """Fit the fixed FRAPOSA basis from a standardized-in-place panel dose matrix.

    X: (n_var, n_panel) float32 ALT-dose, NaN = missing. **Standardized in place.**
    ``dim_ref`` = # reference PCs the classifier consumes; ``dim_online`` = # PCs the
    OADP online-SVD carries (must be ≥ dim_ref; callers that clamp to rank pass both
    already-clamped — see ``pca_fit.fit_pca_model``). Returns a broadcast-safe basis dict.

    This is the exact driver fit shared by both entry points: standardize (per-variant
    mean/std, missing→0) → Gram ``XᵀX`` (one BLAS gemm) → ``eigh`` (descending) → loadings
    ``U = X·(V/s)`` → optional sign-canonicalization. The panel eigendecomposition happens
    here and only here; :func:`project_member` reuses the result for every sample."""
    mean, std = standardize(X)                       # X standardized in place
    XTX = (X.T @ X).astype(np.float64)               # (n_panel × n_panel), single BLAS gemm
    s, V = eig_cov(XTX)
    s = s.astype(np.float64); V = V.astype(np.float64)
    s_on = s[:dim_online]; V_on = V[:, :dim_online]  # (n_panel × dim_online)
    pcs_ref = V[:, :dim_ref] * s[:dim_ref]           # (n_panel × dim_ref)
    U_on = (X @ (V_on / s_on)).astype(np.float64)    # (n_var × dim_online)
    if canonicalize:
        V_on, pcs_ref, U_on = canonicalize_signs(V_on, pcs_ref, U_on)
    return {
        "U_on": U_on,                                # (n_var, dim_online)
        "s_on": s_on,                                # (dim_online,)
        "V_on": V_on,                                # (n_panel, dim_online)
        "pcs_ref": pcs_ref,                          # (n_panel, dim_ref)
        "mean": mean,                                # (n_var, 1)
        "std": std,                                  # (n_var, 1)
        "dim_ref": dim_ref,
        "dim_stu": dim_ref * 2,
    }


def fit_panel_basis(X: np.ndarray, *, dim_ref: int = 4) -> dict:
    """One-time convenience fit: FRAPOSA reference basis from the panel dose matrix.

    Thin wrapper over :func:`fit_basis` that reproduces the historical
    the FRAPOSA contract EXACTLY — ``dim_online = dim_ref * 4`` (i.e.
    ``dim_stu * 2`` with ``dim_stu = dim_ref * 2``), no rank clamping (callers with tiny
    panels should use ``pca_fit.fit_pca_model`` which clamps), and **no sign
    canonicalization** (the historical prs-side fit did not canonicalize; only the
    production PCA basis in ``pca_fit`` did). Standardizes X in place."""
    dim_stu = dim_ref * 2
    dim_online = dim_stu * 2
    return fit_basis(X, dim_ref=dim_ref, dim_online=dim_online, canonicalize=False)


def oadp_project(basis: dict, z: np.ndarray) -> np.ndarray:
    """OADP-project an ALREADY-STANDARDIZED sample vector onto the fixed basis.

    ``z``: (n_var,) or (n_var, 1) float, the standardized (missing already imputed)
    sample vector — exactly what :func:`featurize_to_basis.featurize_to_basis` returns, in
    the SAME variant order as the panel X passed to :func:`fit_basis`. Returns the sample's
    PCs (dim_ref,) via FRAPOSA OADP: online-SVD append + Procrustes-align back to the ref
    PCs. This is the projection half of :func:`project_member`, split out so the model's
    ``predict`` can compose ``featurize_to_basis`` (align/orient/impute/standardize) with
    this — the SAME transform the fit uses — making fit and serve provably identical."""
    dim_stu, dim_ref = basis["dim_stu"], basis["dim_ref"]
    b = np.asarray(z, dtype=np.float64).reshape(-1)

    s_aug, V_aug = svd_online(basis["U_on"], basis["s_on"], basis["V_on"], b)
    s_aug = s_aug[:dim_stu]; V_aug = V_aug[:, :dim_stu]
    pcs_aug = V_aug * s_aug
    R, rho, c = procrustes_diffdim(basis["pcs_ref"], pcs_aug[:-1, :])
    return (pcs_aug[-1:, :] @ R * rho + c).flatten()[:dim_ref]


def project_member(basis: dict, Xu: np.ndarray) -> np.ndarray:
    """Per-sample (distributable): OADP-project one sample onto the fixed basis.

    Xu: (n_var,) or (n_var, 1) float32/float64 ALT-dose, NaN for missing, in the SAME
    variant order as the panel X passed to :func:`fit_basis`. Returns the sample's PCs
    (dim_ref,). Standardizes with the panel's mean/std (missing → 0, i.e. the ``at_mean``
    impute convention), then OADP-projects via :func:`oadp_project`. Byte-identical to the
    the reference OADP projection (the standardize block is unchanged; the
    projection is factored into ``oadp_project``)."""
    mean, std = basis["mean"], basis["std"]

    Xu = np.asarray(Xu, dtype=np.float32).reshape(-1, 1)
    is_miss = np.isnan(Xu)
    Xu = (Xu - mean.astype(np.float32)) / std.astype(np.float32)
    Xu[is_miss] = 0.0
    b = Xu[:, 0].astype(np.float64)
    return oadp_project(basis, b)


# ── Classification (verbatim from pgsc_calc's RF + Mahalanobis) ─────────────────────

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
    # Mahalanobis d^2 of a point vs a Gaussian fitted in n_pcs dimensions ~ chi^2(n_pcs).
    # (Was n_pcs-1, which inflates the survival function and makes outliers look less extreme.)
    p_all = float(chi2.sf(d2, n_pcs))

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
    samples via ``predict_proba``. Returns ``(clf, EmpiricalCovariance)``.
"""
    from sklearn.covariance import EmpiricalCovariance
    from sklearn.ensemble import RandomForestClassifier

    train_mask = panel_unrelated if panel_unrelated is not None else np.ones(len(panel_superpops), dtype=bool)
    train_pops = np.asarray([s for s, m in zip(panel_superpops, train_mask) if m])
    train_pcs = panel_pcs[train_mask, :n_pcs]
    clf = RandomForestClassifier(random_state=32).fit(train_pcs, train_pops)
    cov = EmpiricalCovariance().fit(train_pcs)
    return clf, cov
