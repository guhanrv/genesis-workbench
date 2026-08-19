"""Off-cluster tests for the two MLflow models (design §9.2–9.4).

The load-bearing test is **fit/serve parity**: the panel fit and the model's serve-path
predict, run on the same synthetic sample, must produce the SAME PC vector — this proves
Design A's guarantee holds in code. Plus log_model → load_model → predict round-trips for
both models and the classifier's outlier gate.

Uses a local file-based MLflow tracking store (no Databricks/UC). Run:
``python -m pytest test_models.py``
"""
from __future__ import annotations

import json
import os
import sys

os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")

import numpy as np
import pandas as pd
import pytest

os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")

_LIB = os.path.join(os.path.dirname(__file__), "..", "lib")
_MODEL = os.path.join(os.path.dirname(__file__), "..", "model")
sys.path.insert(0, _LIB)
sys.path.insert(0, _MODEL)

import fraposa as fr          # noqa: E402
import featurize_to_basis as ftb  # noqa: E402
import ancestry_pca as apca   # noqa: E402
import ancestry_classifier as aclf  # noqa: E402

mlflow = pytest.importorskip("mlflow")


# ── shared synthetic panel + a slot-ordered basis_loci table ──
def _panel_and_basis(*, n_var=300, per_pop=60, dim_ref=6, seed=0):
    rng = np.random.default_rng(seed)
    pops = ["AFR", "EAS", "EUR"]
    af = {p: rng.uniform(0.1, 0.9, n_var) for p in pops}
    cols, labels = [], []
    for p in pops:
        for _ in range(per_pop):
            cols.append(rng.binomial(2, af[p]).astype(np.float32))
            labels.append(p)
    X = np.stack(cols, axis=1)                       # (n_var, n_panel)
    basis = fr.fit_basis(X.copy(), dim_ref=dim_ref, dim_online=dim_ref * 4, canonicalize=True)

    chrom = np.array([str((i % 22) + 1) for i in range(n_var)], dtype=object)
    pos = np.arange(1, n_var + 1, dtype=np.int64)
    ref = np.array(["A"] * n_var, dtype=object)
    alt = np.array(["G"] * n_var, dtype=object)
    vids = np.array([f"{chrom[i]}:{pos[i]}:A:G" for i in range(n_var)], dtype=object)
    loci = pd.DataFrame({
        "slot_idx": np.arange(n_var), "variant_id": vids,
        "chrom": chrom, "pos": pos, "ref": ref, "alt": alt,
        "effect_allele": alt, "other_allele": ref,       # panel convention: effect == alt
        "mean": basis["mean"].reshape(-1), "std": basis["std"].reshape(-1),
    })
    return X, np.array(labels), af, pops, basis, loci, vids, dim_ref


def _write_pca_artifacts(tmpdir, basis, loci, *, dim_ref, impute_mode="at_mean"):
    """Write the three ancestry_pca artifacts (parquet / npz / json) to tmpdir."""
    loci_path = os.path.join(tmpdir, apca.BASIS_LOCI_PARQUET)
    npz_path = os.path.join(tmpdir, apca.LOADINGS_NPZ)
    params_path = os.path.join(tmpdir, apca.PARAMS_JSON)
    loci.to_parquet(loci_path)
    np.savez(npz_path, U_on=basis["U_on"], s_on=basis["s_on"], V_on=basis["V_on"],
             pcs_ref=basis["pcs_ref"])
    with open(params_path, "w") as f:
        json.dump({"n_pcs": 5, "dim_ref": dim_ref, "dim_stu": basis["dim_stu"],
                   "dim_online": len(basis["s_on"]), "panel_version": "test",
                   "impute_mode": impute_mode, "min_coverage_frac": 0.1}, f)
    return {"basis_loci": loci_path, "loadings": npz_path, "params": params_path}


class _Ctx:
    """Minimal stand-in for the MLflow PythonModelContext (just .artifacts)."""
    def __init__(self, artifacts):
        self.artifacts = artifacts


def test_fit_serve_parity():
    """THE test (§9.2): featurize+project through the model == project_member on the fit path,
    for the SAME fully-observed sample. impute_mode 'at_mean' makes the model's fill match
    project_member's missing→panel-mean convention, so the two are equal up to the float32/64
    seam already characterized in test_featurize_to_basis."""
    X, labels, af, pops, basis, loci, vids, dim_ref = _panel_and_basis(seed=1)
    model = apca.AncestryPCAModel()

    import tempfile
    with tempfile.TemporaryDirectory() as td:
        arts = _write_pca_artifacts(td, basis, loci, dim_ref=dim_ref, impute_mode="at_mean")
        model.load_context(_Ctx(arts))

        rng = np.random.default_rng(99)
        for p in pops:
            xu = rng.binomial(2, af[p]).astype(np.float32)      # fully observed
            # fit-path reference: project_member standardizes then OADP-projects
            pc_fit = fr.project_member(basis, xu.copy())
            # serve-path: model featurizes (at_mean) then oadp_projects
            row = pd.DataFrame([{apca.IN_VARIANT_IDS: list(vids),
                                 apca.IN_DOSES: [float(x) for x in xu]}])
            out = model.predict(None, row)
            pc_serve = out[[f"pc{k+1}" for k in range(dim_ref)]].to_numpy()[0]
            assert np.max(np.abs(pc_fit - pc_serve)) < 1e-5, (p, pc_fit, pc_serve)
            assert out[apca.OUT_COVERAGE].iloc[0] == 1.0        # fully observed


def test_pca_coverage_and_missing():
    """Partial coverage → coverage_frac < 1; non-basis loci dropped; still finite PCs."""
    X, labels, af, pops, basis, loci, vids, dim_ref = _panel_and_basis(seed=2)
    model = apca.AncestryPCAModel()
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        model.load_context(_Ctx(_write_pca_artifacts(td, basis, loci, dim_ref=dim_ref)))
        rng = np.random.default_rng(7)
        xu = rng.binomial(2, af["EUR"]).astype(np.float32)
        keep = list(range(0, len(vids), 2))            # observe half
        vv = [vids[i] for i in keep] + ["99:1:A:G"]    # + one non-basis locus
        dd = [float(xu[i]) for i in keep] + [2.0]
        out = model.predict(None, pd.DataFrame([{apca.IN_VARIANT_IDS: vv, apca.IN_DOSES: dd}]))
        assert abs(out[apca.OUT_COVERAGE].iloc[0] - len(keep) / len(vids)) < 1e-9
        assert np.all(np.isfinite(out[[f"pc{k+1}" for k in range(dim_ref)]].to_numpy()))


def test_pca_log_load_predict_roundtrip(tmp_path):
    """log_model → load_model → predict reproduces the in-process model output."""
    X, labels, af, pops, basis, loci, vids, dim_ref = _panel_and_basis(seed=3)
    import tempfile
    mlflow.set_tracking_uri(f"file://{tmp_path}/mlruns")
    with tempfile.TemporaryDirectory() as td:
        arts = _write_pca_artifacts(td, basis, loci, dim_ref=dim_ref)
        rng = np.random.default_rng(5)
        xu = rng.binomial(2, af["AFR"]).astype(np.float32)
        row = pd.DataFrame([{apca.IN_VARIANT_IDS: list(vids),
                             apca.IN_DOSES: [float(x) for x in xu]}])

        in_proc = apca.AncestryPCAModel()
        in_proc.load_context(_Ctx(arts))
        expected = in_proc.predict(None, row)

        with mlflow.start_run():
            info = mlflow.pyfunc.log_model(
                artifact_path="model",
                python_model=os.path.join(_MODEL, "ancestry_pca.py"),
                artifacts=arts,
                code_paths=[_LIB],
            )
        loaded = mlflow.pyfunc.load_model(info.model_uri)
        got = loaded.predict(row)
        pc_cols = [f"pc{k+1}" for k in range(dim_ref)]
        assert np.allclose(expected[pc_cols].to_numpy(), got[pc_cols].to_numpy(), atol=1e-9)
        assert np.allclose(expected[apca.OUT_COVERAGE], got[apca.OUT_COVERAGE])


# ── classifier ──
def _make_classifier(basis, labels, **kw):
    st = aclf.build_classifier_state(basis["pcs_ref"], labels, n_pcs=5, **kw)
    import tempfile, pickle
    td = tempfile.mkdtemp()
    pkl = os.path.join(td, aclf.CLASSIFIER_PKL)
    with open(pkl, "wb") as f:
        pickle.dump(st, f)
    model = aclf.AncestryClassifierModel()
    model.load_context(_Ctx({"classifier": pkl}))
    return model, pkl


def test_classifier_predict_and_outlier_gate():
    pytest.importorskip("sklearn")
    pytest.importorskip("scipy")
    X, labels, af, pops, basis, loci, vids, dim_ref = _panel_and_basis(seed=4)
    model, _ = _make_classifier(basis, labels, outlier_alpha=0.001, min_coverage_frac=0.1)

    # an in-population sample projects to its cluster → correct MSP, not an outlier
    rng = np.random.default_rng(21)
    xu = rng.binomial(2, af["EUR"]).astype(np.float32)
    pc = fr.project_member(basis, xu)
    row = {f"pc{k+1}": float(pc[k]) for k in range(5)}
    row[aclf.IN_COVERAGE] = 0.9
    out = model.predict(None, pd.DataFrame([row]))
    assert out[aclf.OUT_MSP].iloc[0] in pops
    assert abs(sum(json.loads(out[aclf.OUT_RF_PROBS].iloc[0]).values()) - 1.0) < 1e-9
    assert not out[aclf.OUT_IS_OUTLIER].iloc[0]

    # low coverage alone must trip the outlier gate even for an in-cluster sample
    row_lowcov = dict(row); row_lowcov[aclf.IN_COVERAGE] = 0.01
    out2 = model.predict(None, pd.DataFrame([row_lowcov]))
    assert out2[aclf.OUT_IS_OUTLIER].iloc[0]


def test_classifier_matches_fraposa_classify():
    """The model's MSP/probs/p must equal the shared fraposa.classify_ancestry (fit-once RF,
    df=n_pcs), so swapping the per-run re-fit for the loaded model changes nothing."""
    pytest.importorskip("sklearn")
    X, labels, af, pops, basis, loci, vids, dim_ref = _panel_and_basis(seed=6)
    model, _ = _make_classifier(basis, labels)
    rng = np.random.default_rng(33)
    xu = rng.binomial(2, af["EAS"]).astype(np.float32)
    pc = fr.project_member(basis, xu)
    ref = fr.classify_ancestry(basis["pcs_ref"], labels.tolist(), pc, n_pcs=5)
    row = {f"pc{k+1}": float(pc[k]) for k in range(5)}
    out = model.predict(None, pd.DataFrame([row]))
    assert out[aclf.OUT_MSP].iloc[0] == ref["msp"]
    assert abs(out[aclf.OUT_MAHALANOBIS_P].iloc[0] - ref["mahalanobis_p_all"]) < 1e-9
    got_probs = json.loads(out[aclf.OUT_RF_PROBS].iloc[0])
    assert got_probs == pytest.approx(ref["rf_probs"])


def test_classifier_log_load_roundtrip(tmp_path):
    pytest.importorskip("sklearn")
    X, labels, af, pops, basis, loci, vids, dim_ref = _panel_and_basis(seed=8)
    model, pkl = _make_classifier(basis, labels)
    rng = np.random.default_rng(44)
    pc = fr.project_member(basis, rng.binomial(2, af["AFR"]).astype(np.float32))
    row = pd.DataFrame([{**{f"pc{k+1}": float(pc[k]) for k in range(5)}, aclf.IN_COVERAGE: 0.8}])
    expected = model.predict(None, row)

    mlflow.set_tracking_uri(f"file://{tmp_path}/mlruns")
    with mlflow.start_run():
        info = mlflow.pyfunc.log_model(
            artifact_path="clf",
            python_model=os.path.join(_MODEL, "ancestry_classifier.py"),
            artifacts={"classifier": pkl},
            code_paths=[_LIB],
        )
    loaded = mlflow.pyfunc.load_model(info.model_uri)
    got = loaded.predict(row)
    assert got[aclf.OUT_MSP].iloc[0] == expected[aclf.OUT_MSP].iloc[0]
    assert abs(got[aclf.OUT_MAHALANOBIS_P].iloc[0] - expected[aclf.OUT_MAHALANOBIS_P].iloc[0]) < 1e-9
    assert got[aclf.OUT_IS_OUTLIER].iloc[0] == expected[aclf.OUT_IS_OUTLIER].iloc[0]


def test_log_ancestry_models_end_to_end(tmp_path):
    """The builder helper (log_ancestry_models) logs both models with correct signatures, and
    they load + chain (ancestry_pca PCs → ancestry_classifier) — the shape ref_01 produces."""
    pytest.importorskip("sklearn")
    import log_ancestry_models as lm
    X, labels, af, pops, basis, loci, vids, dim_ref = _panel_and_basis(seed=12)
    n_var = len(vids)
    kept_pd = loci[["variant_id", "chrom", "pos", "ref", "alt"]].copy()
    kept_pd.insert(0, "vidx", np.arange(n_var))
    fit = {"kept_pd": kept_pd, "n_var": n_var, "dim_ref": dim_ref, "dim_stu": basis["dim_stu"],
           "dim_online": len(basis["s_on"]), "mean": basis["mean"], "std": basis["std"],
           "s_on": basis["s_on"], "V_on": basis["V_on"], "pcs_ref": basis["pcs_ref"], "U_on": basis["U_on"]}

    mlflow.set_tracking_uri(f"file://{tmp_path}/mlruns")
    with mlflow.start_run():
        info_pca = lm.log_ancestry_pca(fit, panel_version="test", uc_model_name="c.s.ancestry_pca")
        info_clf = lm.log_ancestry_classifier(fit, np.array(labels), uc_model_name="c.s.ancestry_classifier")

    rng = np.random.default_rng(77)
    xu = rng.binomial(2, af["EUR"]).astype(np.float32)
    row = pd.DataFrame([{apca.IN_VARIANT_IDS: list(vids), apca.IN_DOSES: [float(x) for x in xu]}])
    pca_out = mlflow.pyfunc.load_model(info_pca.model_uri).predict(row)
    assert pca_out[apca.OUT_COVERAGE].iloc[0] == 1.0

    pcrow = {f"pc{k+1}": float(pca_out[f"pc{k+1}"].iloc[0]) for k in range(5)}
    pcrow[aclf.IN_COVERAGE] = float(pca_out[apca.OUT_COVERAGE].iloc[0])
    clf_out = mlflow.pyfunc.load_model(info_clf.model_uri).predict(pd.DataFrame([pcrow]))
    assert clf_out[aclf.OUT_MSP].iloc[0] in pops
    assert not clf_out[aclf.OUT_IS_OUTLIER].iloc[0]


if __name__ == "__main__":
    # Standalone runner (no pytest): supply a real temp dir for tests that take `tmp_path`,
    # so they actually run here too rather than being skipped.
    import pathlib
    import tempfile as _tempfile
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                if "tmp_path" in fn.__code__.co_varnames:
                    with _tempfile.TemporaryDirectory() as _td:
                        fn(pathlib.Path(_td))
                else:
                    fn()
                print(f"PASS {name}")
            except Exception as e:  # pragma: no cover
                print(f"FAIL {name}: {e}")
