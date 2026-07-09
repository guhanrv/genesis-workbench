"""Shared PCA fit — the pure computation behind both pca_v1 entry points.

Given a QC'd per-variant dose table (one row per variant: ``vidx, chrom, pos, ref, alt,
dose`` where ``dose`` is a float array over samples, NaN = missing), this does the
cohort-agnostic work: a distributed Hail-style windowed-r² **LD prune**, then a
driver-side **FRAPOSA fit** (per-variant standardize → samples² Gram ``XᵀX`` → ``eigh`` →
loadings ``U = X·(V/s)``), and writes ONE projectable model ``pca_basis.npz``
{loadings U/s/V, per-variant mean/std, loci, reference PC scores ``pcs_ref``, sample ids,
superpop labels, provenance}. Optionally also materializes the per-sample scores
(``pcs_ref``) to a Delta table for consumers that just want covariates (GWAS).

Both entry points call this with the SAME contract; they differ only in how they
produce the QC'd dose table (the source-specific adapter):
  - ``ref_01_build_basis`` — pgenlib read of the frozen HGDP+1kGP panel (labeled),
  - ``01_compute_pca``          — Glow-ingested cohort dosage (in-cohort, unlabeled).

The fit is verbatim the FRAPOSA ``lib/prs_ancestry.fit_panel_basis`` used by prs to
project members, so a basis built here is byte-identical to the validated science and
prs's projection/classification is unchanged.

Scale note: the fit collects the pruned matrix (n_var × n_samples) to the driver for the
BLAS ``eigh`` — fine at reference/cohort scale (samples² tractable, the same regime
``spark.ml.PCA`` assumed), and required to emit projectable variant loadings. A very large
in-cohort GWAS PCA (many thousands of samples) should run this on a classic driver sized
like the reference job, or move to a distributed-Gram backend — a documented follow-up,
not a silent regression.
"""
from __future__ import annotations

import os
import shutil
import tempfile

import numpy as np
import pandas as pd
import pyspark.sql.functions as F
from pyspark.sql.types import StructType, StructField, StringType, LongType


def fit_pca_model(
    spark,
    qc_df,
    *,
    sample_ids,
    superpops,
    dim_ref: int,
    r2: float,
    window_bp: int,
    panel_version: str,
    out_path: str,
    chunk_bp: int = 25_000_000,
    scores_table: str | None = None,
    local_npz: str | None = None,
):
    """LD-prune ``qc_df`` → FRAPOSA fit → write the projectable model npz to ``out_path``.

    ``sample_ids`` / ``superpops`` are the dose columns' sample order + labels (superpops
    may be an empty array for an unlabeled cohort). Returns a small summary dict.
    """
    n_panel = len(sample_ids)
    dim_stu = dim_ref * 2
    dim_online = dim_stu * 2

    # --- 1. Distributed LD prune (per chrom × chunk; Hail-style windowed greedy r²) ---
    PRUNE_SCHEMA = StructType([
        StructField("vidx", LongType()), StructField("chrom", StringType()),
        StructField("pos", LongType()), StructField("ref", StringType()), StructField("alt", StringType())])

    def _prune_chrom(pdf):
        import numpy as _np
        from bisect import bisect_left
        pdf = pdf.sort_values("pos")
        pos = pdf["pos"].to_numpy()
        # each group is one chunk_bp position slice of a chromosome (not a whole chromosome), so D is
        # bounded regardless of chromosome size. stack without .tolist()'s Python list-of-lists
        # transient and standardize in-place in float32 to keep peak memory ~2×D.
        D = _np.stack(pdf["dose"].to_numpy()).astype(_np.float32, copy=False)   # (m, n_samples)
        mean = _np.nanmean(D, 1, keepdims=True).astype(_np.float32)
        sd = _np.nanstd(D, 1, keepdims=True).astype(_np.float32); sd[sd == 0] = 1
        inv = 1.0 / D.shape[1]
        Z = D - mean; Z /= sd; _np.nan_to_num(Z, copy=False)
        del D, mean, sd                                             # free dose matrix before the r² loop
        kpos, kidx = [], []
        keep = _np.zeros(len(pdf), dtype=bool)
        for i in range(len(pdf)):
            lo = bisect_left(kpos, pos[i] - window_bp)
            if lo < len(kidx):
                rr = (Z[kidx[lo:]].astype(_np.float64) @ Z[i].astype(_np.float64) * inv) ** 2
                if (rr >= r2).any():
                    continue
            kpos.append(int(pos[i])); kidx.append(i); keep[i] = True
        return pdf.loc[keep, ["vidx", "chrom", "pos", "ref", "alt"]]

    # Partition each chromosome into chunk_bp position slices so no single prune task materializes a
    # whole chromosome's dose matrix. Hail ld_prune's stage-1 "local prune per partition"; the only
    # approximation vs a whole-chromosome prune is at slice seams (immaterial for a PCA basis).
    qc_chunked = qc_df.withColumn("chunk", (F.col("pos") / F.lit(chunk_bp)).cast("long"))
    kept = qc_chunked.groupBy("chrom", "chunk").applyInPandas(_prune_chrom, schema=PRUNE_SCHEMA)
    kept = kept.orderBy(F.col("chrom").cast("int"), "pos")            # stable variant order
    kept_pd = kept.toPandas()
    n_var = len(kept_pd)
    order_by_vidx = {int(v): i for i, v in enumerate(kept_pd["vidx"].tolist())}
    print(f"pruned loci: {n_var}")

    # --- 2. Fit on the driver — collect the pruned panel (small) → BLAS XᵀX → eigh → loadings ---
    # The pruned panel is small, so collect it once and fit on the driver: XᵀX is a single BLAS gemm,
    # and we avoid re-scanning the multi-GB QC store. The standardize/eigh/loadings are verbatim
    # lib/prs_ancestry.fit_panel_basis (FRAPOSA). (Requires driver.maxResultSize ≥ ~4g for the collect.)
    kept_rows = qc_df.join(F.broadcast(kept.select("vidx")), "vidx").select("vidx", "dose").collect()
    vidx_arr = np.fromiter((r["vidx"] for r in kept_rows), np.int64, len(kept_rows))
    X = np.stack([np.asarray(r["dose"], dtype=np.float32) for r in kept_rows])   # (n_var, n_samples), NaN=missing
    X = X[np.argsort([order_by_vidx[int(v)] for v in vidx_arr])]                 # stable (chrom, pos) order
    assert X.shape[0] == n_var, f"collected {X.shape[0]} != pruned {n_var}"
    del kept_rows, vidx_arr

    # FRAPOSA standardize: per-variant mean/std (ddof=0), cast to float32, missing → 0.
    is_miss = np.isnan(X)
    mean = np.zeros(n_var, np.float64); std = np.zeros(n_var, np.float64)
    for i in range(n_var):
        row = X[i, :][~is_miss[i, :]]
        if row.size:
            mean[i] = float(np.mean(row)); std[i] = float(np.std(row))
    std[std == 0] = 1.0
    X -= mean.astype(np.float32).reshape(-1, 1)
    X /= std.astype(np.float32).reshape(-1, 1)
    X[is_miss] = 0.0
    mean = mean.reshape(-1, 1); std = std.reshape(-1, 1)

    XTX = (X.T @ X).astype(np.float64)                               # (n_panel × n_panel), single BLAS gemm
    ssq, V = np.linalg.eigh(XTX)
    s_all = np.sqrt(np.abs(ssq))[::-1]; V_all = V.T[::-1].T          # descending
    s_on = s_all[:dim_online]; V_on = V_all[:, :dim_online]          # (n_panel × dim_online)
    pcs_ref = V_all[:, :dim_ref] * s_all[:dim_ref]                   # (n_panel × dim_ref)
    U_on = (X @ (V_on / s_on)).astype(np.float64)                    # (n_var × dim_online)
    print(f"eigendecomposition (driver): top s = {np.round(s_on[:dim_ref], 1)}")

    # --- 3. Persist the ONE projectable model (same shape as lib/prs_ancestry.fit_panel_basis) ---
    local_npz = local_npz or os.path.join(tempfile.gettempdir(), "pca_basis.npz")
    np.savez(local_npz,
             loci_chrom=kept_pd["chrom"].astype(str).to_numpy(),
             loci_pos=kept_pd["pos"].to_numpy(np.int64),
             loci_ref=kept_pd["ref"].astype(str).to_numpy(),
             loci_alt=kept_pd["alt"].astype(str).to_numpy(),
             U_on=U_on, s_on=s_on, V_on=V_on, pcs_ref=pcs_ref, mean=mean, std=std,
             dim_ref=dim_ref, dim_stu=dim_stu,
             panel_iids=np.array(sample_ids), superpops=np.asarray(superpops),
             panel_version=np.array(panel_version))
    shutil.copyfile(local_npz, out_path)                            # out_path is a /Volumes FUSE path
    print(f"wrote basis → {out_path} | n_loci={n_var} n_panel={n_panel} "
          f"dim_online={dim_online} panel_version={panel_version}")

    # --- 4. Optional: materialize per-sample scores for covariate consumers (GWAS) ---
    if scores_table:
        cols = ["sample_id"] + [f"PC{j + 1}" for j in range(dim_ref)]
        rows = [(str(sample_ids[i]), *[float(pcs_ref[i, j]) for j in range(dim_ref)]) for i in range(n_panel)]
        spark.createDataFrame(rows, cols).write.mode("overwrite").option(
            "overwriteSchema", "true").saveAsTable(scores_table)
        print(f"wrote scores → {scores_table} ({n_panel} samples × {dim_ref} PCs)")

    return {"n_var": int(n_var), "n_panel": int(n_panel), "dim_online": int(dim_online),
            "panel_version": panel_version, "out_path": out_path}
