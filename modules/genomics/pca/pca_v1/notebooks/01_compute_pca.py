# Databricks notebook source
# MAGIC %md
# MAGIC # PCA step 2 — population-structure PCA (in-cohort entry point)
# MAGIC
# MAGIC The **in-cohort** adapter: reads the Glow-ingested dosage Delta from `00_ingest_vcf`, QC's to
# MAGIC common biallelic SNPs, and hands a per-variant dose table to the shared **`lib/pca_fit`** — the
# MAGIC exact same LD-prune + FRAPOSA fit the **reference** adapter (`ref_01_build_basis`) uses.
# MAGIC One fit, one output contract: it emits a projectable model npz (loadings/mean/std/loci/scores)
# MAGIC plus a per-sample scores table — the PCs a GWAS adjusts for (genesis's GWAS runs unadjusted).
# MAGIC
# MAGIC No Glow, no RDD API. The fit does a driver-side `eigh` on the samples² Gram (the projectable
# MAGIC basis needs variant loadings, which `spark.ml.PCA` can't emit in this orientation); it collects
# MAGIC the pruned matrix to the driver — fine at cohort scale, but a very large cohort should run on a
# MAGIC classic driver like the reference job (see `lib/pca_fit`'s scale note). A fully distributed-Gram
# MAGIC backend for the huge-cohort regime is a documented follow-up.

# COMMAND ----------

dbutils.widgets.text("catalog", "genesis_workbench", "Catalog")
dbutils.widgets.text("schema", "genesis_schema", "Schema")
dbutils.widgets.text("n_components", "10", "Number of principal components")
dbutils.widgets.text("maf_cutoff", "0.05", "Minor-allele-frequency cutoff")
dbutils.widgets.text("max_variants", "0", "Cap SNPs used (0 = all; M is unbounded here)")
dbutils.widgets.text("r2", "0.05", "LD-prune r² threshold (prune if ≥)")
dbutils.widgets.text("window_bp", "1000000", "LD-prune window (bp)")
dbutils.widgets.text("model_dir", "", "Volume dir to write the projectable model npz (pca_model_<run>.npz)")
dbutils.widgets.text("backend", "driver", "PCA fit backend: driver (default) | distributed (large cohorts, classic cluster)")
dbutils.widgets.text("mlflow_run_id", "", "MLflow Run ID")

catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")
n_components = int(dbutils.widgets.get("n_components"))
maf_cutoff = float(dbutils.widgets.get("maf_cutoff"))
max_variants = int(dbutils.widgets.get("max_variants"))
r2 = float(dbutils.widgets.get("r2"))
window_bp = int(dbutils.widgets.get("window_bp"))
model_dir = dbutils.widgets.get("model_dir")
backend = dbutils.widgets.get("backend").strip() or "driver"
mlflow_run_id = dbutils.widgets.get("mlflow_run_id")

# COMMAND ----------

import os
import sys
import numpy as np
import pyspark.sql.functions as F

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "..", "lib")))
import pca_fit                                   # the shared prune + FRAPOSA fit (same as ref_01_build_basis)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 1. Dosage (from ingest) → QC'd per-variant dose table (the shared-fit hand-off)
# MAGIC Same contract the reference adapter (`ref_01_build_basis`) produces: one row per variant
# MAGIC with a float dose array over samples (NaN = missing). Keep biallelic common SNPs (MAF ≥ cutoff;
# MAGIC biallelic already enforced at ingest), then hand off to `lib/pca_fit` — identical LD-prune +
# MAGIC FRAPOSA fit as the reference basis, so this in-cohort path and the reference path are one engine.

# COMMAND ----------

run = mlflow_run_id.replace("-", "_")
dosage = spark.table(f"{catalog}.{schema}.pca_dosage_{run}")

# canonical sample order (identical across all variant rows) → the fit's dose-column labels
sample_ids = list(dosage.select("sample_ids").first()["sample_ids"])
N = len(sample_ids)

freq = dosage.select(
    "chrom", "pos", "states",
    F.expr("aggregate(filter(states, x -> x is not null), cast(0.0 as double), (a, x) -> a + x)").alias("alt_sum"),
    F.expr("size(filter(states, x -> x is not null))").alias("n_called"),
).withColumn("af", F.col("alt_sum") / (2 * F.col("n_called")))

common = freq.where((F.col("n_called") > 0) & (F.least(F.col("af"), 1 - F.col("af")) >= F.lit(maf_cutoff)))
if max_variants and max_variants > 0:
    common = common.limit(max_variants)

# per-variant dose table matching lib/pca_fit's contract: dose = states with nulls → NaN (the fit's
# missing convention); ref/alt are placeholders (unused for an in-cohort fit — no cross-strand
# projection of out-of-sample members); vidx = a stable row id for the prune/collect join.
qc = (common
      .withColumn("vidx", F.monotonically_increasing_id())
      .withColumn("dose", F.expr("transform(states, x -> coalesce(cast(x as double), double('nan')))"))
      .withColumn("ref", F.lit("N")).withColumn("alt", F.lit("N"))
      .select("vidx", "chrom", "pos", "ref", "alt", "dose"))

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2. Prune + fit → projectable model (shared `lib/pca_fit`) + per-sample scores for GWAS
# MAGIC One fit, one output contract. The model npz is projectable (loadings/mean/std) like the
# MAGIC reference basis; `scores_table` also materializes per-sample PCs — the covariates a GWAS adjusts
# MAGIC for. NOTE: the fit collects the pruned matrix to the driver; fine at cohort scale, but a very
# MAGIC large cohort should run this on a classic driver (like the reference job) — see lib/pca_fit.

# COMMAND ----------

out_path = f"{model_dir.rstrip('/')}/pca_model_{run}.npz"
scores_table = f"{catalog}.{schema}.pca_components_{run}"
pca_fit.fit_pca_model(
    spark, qc,
    sample_ids=sample_ids, superpops=np.array([]),      # in-cohort: unlabeled (no reference superpops)
    dim_ref=min(n_components, N), r2=r2, window_bp=window_bp,
    panel_version=f"cohort_{run}", out_path=out_path,
    scores_table=scores_table, backend=backend,
)
print(f"cohort PCA → model {out_path} + scores {scores_table}")
