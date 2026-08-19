"""ancestry_pca — custom ``mlflow.pyfunc`` model: dosage → principal components (Design A).

This is the portable, versioned replacement for the ``pca_basis.npz``-on-a-Volume artifact
(decisions D1/D6, design §5.1). It owns the full inference transform for a new sample:

    predict input   : per sample, the OBSERVED (variant_id, dose) loci only, any order
                      (one DataFrame row per sample; see the signature below)
    predict internal: 1. look up variant_id → slot against OWN basis_loci  (drop non-basis)
                      2. orient dose to the slot's effect_allele
                      3. missing basis slots → fill_value (impute_mode; v1 default "zero")
                      4. standardize with OWN frozen mean/std
                      5. OADP-project (svd_online + Procrustes) → PCs
                      6. coverage_frac = n_covered / N_basis_loci
    predict output  : { pc1..pck : double, coverage_frac : double }  (one row per sample)

Steps 1–4 are the shared ``featurize_to_basis`` (the SAME function the fit pipeline calls to
build panel vectors), step 5 is the shared ``fraposa.oadp_project`` — so fit and serve are
provably identical (no train/serve skew). The caller passes *names*, never a slot order, so
it is structurally impossible to align to the wrong slots (Design A).

**Serialization (fixes the old ``allow_pickle=True``):**
  - ``basis_loci`` → parquet artifact (slot_idx, variant_id, chrom, pos, ref, alt,
    effect_allele, other_allele, mean, std) — the typed coordinate system (design §4.2).
  - loadings ``U_on/s_on/V_on`` + ``pcs_ref`` → numeric-only ``.npz`` with
    ``allow_pickle=False``.
  - params (n_pcs, dim_ref, dim_stu, dim_online, panel_version, impute_mode,
    min_coverage_frac) → JSON.
An MLflow signature wraps it so the on-disk format stops being the consumer contract.

This module is logged with MLflow **code-based logging** (``python_model=<this file>``), the
same pattern the repo's other pyfunc models use (e.g. netsolp_wrapper.py) — the class is
NOT cloudpickled from a notebook ``__main__`` scope.
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
import mlflow.pyfunc


# artifact filenames inside the logged model (kept as constants so the builder and the
# loader agree on exactly one set of names).
BASIS_LOCI_PARQUET = "basis_loci.parquet"
LOADINGS_NPZ = "loadings.npz"
PARAMS_JSON = "params.json"

# predict input/output column names (also used to build the MLflow signature at log time).
IN_SAMPLE_ID = "sample_id"
IN_VARIANT_IDS = "variant_ids"   # array<string>  — observed loci for this sample
IN_DOSES = "doses"               # array<double>  — ALT-allele dosage, aligned to variant_ids
OUT_COVERAGE = "coverage_frac"


class AncestryPCAModel(mlflow.pyfunc.PythonModel):
    """Design-A ancestry-PCA pyfunc. See module docstring."""

    def load_context(self, context):
        import pyarrow.parquet as pq  # noqa: F401 (kept explicit for the serving env)

        # --- params (JSON) ---
        with open(context.artifacts["params"]) as f:
            self.params = json.load(f)
        self.dim_ref = int(self.params["dim_ref"])
        self.dim_stu = int(self.params["dim_stu"])
        self.impute_mode = self.params.get("impute_mode", "zero")
        self.min_coverage_frac = float(self.params.get("min_coverage_frac", 0.0))
        self.n_pcs = int(self.params.get("n_pcs", self.dim_ref))

        # --- basis_loci (parquet) → the slot-ordered coordinate system ---
        loci = pd.read_parquet(context.artifacts["basis_loci"]).sort_values("slot_idx")
        self.n_basis = len(loci)
        # basis dict shaped for featurize_to_basis (slot-ordered arrays) + fraposa.oadp_project.
        self._basis_arrays = {
            "variant_id": loci["variant_id"].to_numpy(dtype=object),
            "effect_allele": loci["effect_allele"].to_numpy(dtype=object),
            "other_allele": loci["other_allele"].to_numpy(dtype=object),
            "ref": loci["ref"].to_numpy(dtype=object),
            "alt": loci["alt"].to_numpy(dtype=object),
            "mean": loci["mean"].to_numpy(dtype=np.float64),
            "std": loci["std"].to_numpy(dtype=np.float64),
        }

        # --- loadings (numeric-only npz, allow_pickle=False) ---
        with np.load(context.artifacts["loadings"], allow_pickle=False) as z:
            self._proj_basis = {
                "U_on": z["U_on"], "s_on": z["s_on"], "V_on": z["V_on"],
                "pcs_ref": z["pcs_ref"], "dim_ref": self.dim_ref, "dim_stu": self.dim_stu,
            }

        # precompute the variant_id → slot lookup once (reused for every row).
        import featurize_to_basis as _ftb
        _ftb.build_basis_index(self._basis_arrays)
        self._ftb = _ftb

        import fraposa as _fr
        self._fr = _fr

    def _predict_one(self, variant_ids, doses):
        """One sample's observed (variant_id, dose) pairs → (pcs (dim_ref,), coverage_frac)."""
        pairs = zip(
            [] if variant_ids is None else list(variant_ids),
            [] if doses is None else list(doses),
        )
        z, n_covered = self._ftb.featurize_to_basis(
            pairs, self._basis_arrays, impute_mode=self.impute_mode)
        pcs = self._fr.oadp_project(self._proj_basis, z)
        coverage_frac = (n_covered / self.n_basis) if self.n_basis else 0.0
        return pcs, coverage_frac

    def predict(self, context, model_input, params=None):
        # Accept a DataFrame (one row per sample) or a list of record dicts.
        if not isinstance(model_input, pd.DataFrame):
            model_input = pd.DataFrame(model_input)

        pc_cols = [f"pc{k + 1}" for k in range(self.dim_ref)]
        out_rows = []
        for _, row in model_input.iterrows():
            pcs, cov = self._predict_one(row.get(IN_VARIANT_IDS), row.get(IN_DOSES))
            rec = {pc_cols[k]: float(pcs[k]) for k in range(self.dim_ref)}
            rec[OUT_COVERAGE] = float(cov)
            out_rows.append(rec)
        return pd.DataFrame(out_rows, columns=pc_cols + [OUT_COVERAGE])


# Code-based logging entry point (the repo idiom — see netsolp_wrapper.py): MLflow imports
# this file and picks up the model instance set here rather than cloudpickling a class from a
# notebook __main__ scope.
mlflow.models.set_model(AncestryPCAModel())
