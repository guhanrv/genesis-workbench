"""Build + log the ancestry_pca and ancestry_classifier MLflow models from a fit result.

Called by the reference-basis notebook (``ref_01_build_basis``) after ``pca_fit.prune_and_fit``
returns the fitted arrays. Keeps the notebook thin and puts the artifact-shaping / log_model
wiring in one importable, reviewable place. The two models replace the old ``pca_basis.npz``
Volume artifact (design §5): ``ancestry_pca`` (Design-A pyfunc) and ``ancestry_classifier``
(fit-once RF + covariance pyfunc).

Serialization (design §5.1): basis_loci → parquet, loadings → numeric-only npz
(``allow_pickle=False``), params → JSON, classifier state → pickle. All logged as MLflow
artifacts behind a signature, so the on-disk format is no longer the consumer contract.
"""
from __future__ import annotations

import json
import os
import pickle
import tempfile

import numpy as np
import pandas as pd


# module dirs, so log_model can point python_model= at the wrapper files and code_paths= at the lib.
_MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
_LIB_DIR = os.path.join(_MODEL_DIR, "..", "lib")

# The model's load_context does flat imports (`import featurize_to_basis`, `import fraposa`), so those
# files must land at the TOP of the model's code/ dir. Passing the lib DIRECTORY to code_paths nests
# them as code/lib/*.py (not importable flat → ModuleNotFoundError when loaded on another cluster).
# Pass the individual FILES instead so they land as code/featurize_to_basis.py, code/fraposa.py.
_CODE_FILES = [
    os.path.join(_LIB_DIR, "featurize_to_basis.py"),
    os.path.join(_LIB_DIR, "fraposa.py"),
]


def build_basis_loci(kept_pd: pd.DataFrame, mean: np.ndarray, std: np.ndarray) -> pd.DataFrame:
    """Assemble the typed ``basis_loci`` table (design §4.2) from the pruned loci + fit stats.

    ``kept_pd`` has columns vidx/chrom/pos/ref/alt in stable slot order (the fit's variant
    order). effect_allele = alt, other_allele = ref (the panel's pgenlib dose counts ALT, so
    the basis is oriented to ALT — matches ``featurize_to_basis``'s panel convention).
    variant_id = ``chrom:pos:ref:alt`` (canonical, design §4.1)."""
    n = len(kept_pd)
    chrom = kept_pd["chrom"].astype(str).to_numpy()
    pos = kept_pd["pos"].to_numpy(np.int64)
    ref = kept_pd["ref"].astype(str).to_numpy()
    alt = kept_pd["alt"].astype(str).to_numpy()
    return pd.DataFrame({
        "slot_idx": np.arange(n, dtype=np.int64),
        "variant_id": [f"{chrom[i]}:{pos[i]}:{ref[i]}:{alt[i]}" for i in range(n)],
        "chrom": chrom, "pos": pos, "ref": ref, "alt": alt,
        "effect_allele": alt, "other_allele": ref,
        "mean": np.asarray(mean, dtype=np.float64).reshape(-1),
        "std": np.asarray(std, dtype=np.float64).reshape(-1),
    })


def build_reference_pcs(fit: dict, superpops, *, n_pcs: int = 2) -> pd.DataFrame:
    """The panel's (PC1, PC2, superpop) cloud for the app's PCA scatter background (§7.6).

    ``fit["pcs_ref"]`` is (n_panel, dim_ref); ``superpops`` is the length-n_panel label array.
    Returns a DataFrame with pc1..pc{n_pcs} + superpop, which ref_01 writes to the
    ``ancestry_pca_reference_pcs`` table the results endpoint reads. Kept to PC1/PC2 (n_pcs=2)
    — cheap, precomputed, high-value (design §7.6)."""
    pcs = np.asarray(fit["pcs_ref"])
    k = min(n_pcs, pcs.shape[1])
    cols = {f"pc{j + 1}": pcs[:, j].astype(float) for j in range(k)}
    cols["superpop"] = [str(s) for s in superpops]
    return pd.DataFrame(cols)


def log_ancestry_pca(fit: dict, *, panel_version: str, uc_model_name: str,
                     n_pcs: int = 5, impute_mode: str = "zero",
                     min_coverage_frac: float = 0.1, registered_model_name: str | None = None):
    """Log the ancestry_pca pyfunc from a ``pca_fit.prune_and_fit`` result. Returns the
    logged ``ModelInfo``. Writes the three artifacts (parquet / npz / json) to a temp dir,
    then ``mlflow.pyfunc.log_model`` with an explicit signature. ``registered_model_name``
    (UC 3-level) registers into Unity Catalog when set."""
    import mlflow
    from mlflow.models import ModelSignature
    from mlflow.types.schema import Schema, ColSpec, Array, DataType

    import ancestry_pca as apca  # from _MODEL_DIR (added to sys.path by the notebook)

    loci = build_basis_loci(fit["kept_pd"], fit["mean"], fit["std"])
    dim_ref = int(fit["dim_ref"])

    tmp = tempfile.mkdtemp(prefix="ancestry_pca_")
    loci_path = os.path.join(tmp, apca.BASIS_LOCI_PARQUET)
    npz_path = os.path.join(tmp, apca.LOADINGS_NPZ)
    params_path = os.path.join(tmp, apca.PARAMS_JSON)
    loci.to_parquet(loci_path)
    # numeric-only, allow_pickle=False-safe (no object arrays) — fixes the old npz pickle issue.
    np.savez(npz_path, U_on=fit["U_on"], s_on=fit["s_on"], V_on=fit["V_on"], pcs_ref=fit["pcs_ref"])
    with open(params_path, "w") as f:
        json.dump({"n_pcs": int(n_pcs), "dim_ref": dim_ref, "dim_stu": int(fit["dim_stu"]),
                   "dim_online": int(fit["dim_online"]), "panel_version": panel_version,
                   "impute_mode": impute_mode, "min_coverage_frac": float(min_coverage_frac)}, f)

    # signature: one row per sample — list-typed observed loci in, PCs + coverage out.
    in_schema = Schema([
        ColSpec(DataType.string, apca.IN_SAMPLE_ID, required=False),
        ColSpec(Array(DataType.string), apca.IN_VARIANT_IDS),
        ColSpec(Array(DataType.double), apca.IN_DOSES),
    ])
    out_schema = Schema(
        [ColSpec(DataType.double, f"pc{k + 1}") for k in range(dim_ref)]
        + [ColSpec(DataType.double, apca.OUT_COVERAGE)]
    )
    signature = ModelSignature(inputs=in_schema, outputs=out_schema)

    return mlflow.pyfunc.log_model(
        artifact_path="ancestry_pca",   # 'artifact_path' (MLflow 2.x); 'name' is the 3.x rename
        python_model=os.path.join(_MODEL_DIR, "ancestry_pca.py"),
        artifacts={"basis_loci": loci_path, "loadings": npz_path, "params": params_path},
        code_paths=_CODE_FILES,
        signature=signature,
        registered_model_name=registered_model_name,
    )


def log_ancestry_classifier(fit: dict, superpops, *, uc_model_name: str, n_pcs: int = 5,
                            outlier_alpha: float = 0.001, min_coverage_frac: float = 0.1,
                            registered_model_name: str | None = None):
    """Log the ancestry_classifier pyfunc (fit-once RF + covariance) from the fit's
    ``pcs_ref`` + panel ``superpops``. Returns the logged ``ModelInfo``."""
    import mlflow
    from mlflow.models import ModelSignature
    from mlflow.types.schema import Schema, ColSpec, DataType

    import ancestry_classifier as aclf

    state = aclf.build_classifier_state(
        fit["pcs_ref"], superpops, n_pcs=n_pcs, outlier_alpha=outlier_alpha,
        min_coverage_frac=min_coverage_frac)

    tmp = tempfile.mkdtemp(prefix="ancestry_clf_")
    pkl_path = os.path.join(tmp, aclf.CLASSIFIER_PKL)
    with open(pkl_path, "wb") as f:
        pickle.dump(state, f)

    in_schema = Schema(
        [ColSpec(DataType.double, f"pc{k + 1}") for k in range(n_pcs)]
        + [ColSpec(DataType.double, aclf.IN_COVERAGE, required=False)]
    )
    out_schema = Schema([
        ColSpec(DataType.string, aclf.OUT_MSP),
        ColSpec(DataType.string, aclf.OUT_RF_PROBS),
        ColSpec(DataType.double, aclf.OUT_MAHALANOBIS_P),
        ColSpec(DataType.boolean, aclf.OUT_IS_OUTLIER),
    ])
    signature = ModelSignature(inputs=in_schema, outputs=out_schema)

    return mlflow.pyfunc.log_model(
        artifact_path="ancestry_classifier",   # 'artifact_path' (MLflow 2.x); 'name' is the 3.x rename
        python_model=os.path.join(_MODEL_DIR, "ancestry_classifier.py"),
        artifacts={"classifier": pkl_path},
        code_paths=_CODE_FILES,
        signature=signature,
        registered_model_name=registered_model_name,
    )
