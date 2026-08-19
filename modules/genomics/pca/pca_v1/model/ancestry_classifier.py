"""ancestry_classifier — ``mlflow.pyfunc`` model: principal components → ancestry label.

The RandomForest + Mahalanobis covariance are fit **once** at model-build on the frozen
panel ``(pcs_ref, superpops)`` and loaded (never re-fit) at score time.

    predict input   : { pc1..pck : double }  (+ optional coverage_frac : double)
                      one row per sample; k = n_pcs (5, pgsc_calc convention)
    predict output  : { most_similar_pop : string,
                        rf_probs         : json string (map<string,double>),
                        mahalanobis_p    : double,
                        is_outlier       : bool }

``is_outlier = (mahalanobis_p < outlier_alpha) OR (coverage_frac < min_coverage_frac)``.
The coverage term only fires when a ``coverage_frac`` column is supplied (it comes from the
``ancestry_pca`` model, design §4.4) — absent it, only the Mahalanobis gate applies.

**Why pyfunc, not bare ``mlflow.sklearn``.** The stated output contract (Mahalanobis p +
the two-trigger outlier gate) is more than an sklearn estimator's ``predict``/
``predict_proba`` can express, and ``is_outlier`` needs ``coverage_frac`` which is not an
RF feature. So the model is a thin pyfunc holding the fit-once RandomForest + the
EmpiricalCovariance; both are standard sklearn objects (pickle-serialized exactly as
``mlflow.sklearn`` would) — this is NOT the ``allow_pickle`` numpy-array anti-pattern §5.1
warns about. The classify math is the shared ``fraposa`` code (RF random_state=32,
Mahalanobis df = n_pcs), so it stays consistent with the historical classifier.

Logged via MLflow code-based logging (``python_model=<this file>``).
"""
from __future__ import annotations

import json
import pickle

import numpy as np
import pandas as pd
import mlflow.pyfunc

CLASSIFIER_PKL = "classifier.pkl"   # {clf, cov, classes, n_pcs, outlier_alpha, min_coverage_frac}

OUT_MSP = "most_similar_pop"
OUT_RF_PROBS = "rf_probs"
OUT_MAHALANOBIS_P = "mahalanobis_p"
OUT_IS_OUTLIER = "is_outlier"
IN_COVERAGE = "coverage_frac"       # optional input column (from ancestry_pca)


def build_classifier_state(pcs_ref, superpops, *, n_pcs=5, outlier_alpha=0.001,
                           min_coverage_frac=0.1, panel_unrelated=None):
    """Fit the RF + EmpiricalCovariance ONCE and return the picklable state dict the model
    loads. ``pcs_ref`` (n_panel, ≥n_pcs) and ``superpops`` (n_panel,) are the frozen panel
    inputs from the basis fit; king.cutoff was already applied at basis build, so the whole
    panel is the training set (``panel_unrelated=None``). Uses the shared ``fraposa.fit_rf``
    so the estimator matches the historical classifier exactly."""
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lib"))
    import fraposa
    clf, cov = fraposa.fit_rf(np.asarray(pcs_ref), list(superpops), n_pcs=n_pcs,
                              panel_unrelated=panel_unrelated)
    return {
        "clf": clf, "cov": cov, "classes": [str(c) for c in clf.classes_],
        "n_pcs": int(n_pcs), "outlier_alpha": float(outlier_alpha),
        "min_coverage_frac": float(min_coverage_frac),
    }


class AncestryClassifierModel(mlflow.pyfunc.PythonModel):
    """PCs → ancestry label + outlier gate. See module docstring."""

    def load_context(self, context):
        from scipy.stats import chi2  # noqa: F401 (explicit for the serving env)
        with open(context.artifacts["classifier"], "rb") as f:
            st = pickle.load(f)
        self.clf = st["clf"]
        self.cov = st["cov"]
        self.classes = st["classes"]
        self.n_pcs = int(st["n_pcs"])
        self.outlier_alpha = float(st["outlier_alpha"])
        self.min_coverage_frac = float(st["min_coverage_frac"])

    def _pc_matrix(self, model_input):
        """Extract the (n, n_pcs) PC matrix from pc1..pc{n_pcs} columns."""
        cols = [f"pc{k + 1}" for k in range(self.n_pcs)]
        missing = [c for c in cols if c not in model_input.columns]
        if missing:
            raise ValueError(f"ancestry_classifier input missing PC columns {missing}; "
                             f"expected pc1..pc{self.n_pcs}")
        return model_input[cols].to_numpy(dtype=np.float64)

    def predict(self, context, model_input, params=None):
        from scipy.stats import chi2
        if not isinstance(model_input, pd.DataFrame):
            model_input = pd.DataFrame(model_input)

        X = self._pc_matrix(model_input)
        coverage = (model_input[IN_COVERAGE].to_numpy(dtype=np.float64)
                    if IN_COVERAGE in model_input.columns else None)

        probs = self.clf.predict_proba(X)               # (n, n_classes)
        d2 = self.cov.mahalanobis(X)                     # (n,)
        p_all = chi2.sf(d2, self.n_pcs)                  # df = n_pcs (matches fraposa.classify)

        out_rows = []
        for i in range(len(X)):
            pr = {str(c): float(p) for c, p in zip(self.clf.classes_, probs[i])}
            msp = str(self.clf.classes_[int(np.argmax(probs[i]))])
            is_out = bool(p_all[i] < self.outlier_alpha)
            if coverage is not None:
                is_out = is_out or bool(coverage[i] < self.min_coverage_frac)
            out_rows.append({
                OUT_MSP: msp,
                OUT_RF_PROBS: json.dumps(pr),
                OUT_MAHALANOBIS_P: float(p_all[i]),
                OUT_IS_OUTLIER: is_out,
            })
        return pd.DataFrame(out_rows, columns=[OUT_MSP, OUT_RF_PROBS, OUT_MAHALANOBIS_P, OUT_IS_OUTLIER])


# Code-based logging entry point (the repo idiom — see netsolp_wrapper.py).
mlflow.models.set_model(AncestryClassifierModel())
