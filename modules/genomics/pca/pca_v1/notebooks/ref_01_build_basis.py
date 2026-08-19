# Databricks notebook source
# MAGIC %md
# MAGIC # Build the frozen FRAPOSA PCA basis (reference cohort) — Spark-native, distributed (classic; NOT serverless)
# MAGIC
# MAGIC pca_v1's **reference-cohort** entry point: fits PCA on the frozen HGDP+1kGP panel and emits
# MAGIC **both** a Volume `pca_basis.npz` (gVCF extract / npz fallback) **and** two MLflow pyfuncs
# MAGIC (`ancestry_pca` + `ancestry_classifier` at `@champion`). PRS still extracts member gVCFs with
# MAGIC `gvcf_dose`; the models own align/orient/impute/project so fit and serve cannot drift.
# MAGIC The sibling `01_compute_pca` is the **in-cohort** entry point (same sample-covariance
# MAGIC eigendecomposition, scores-only, for GWAS covariates); Task 2 collapses the two into one engine.
# MAGIC **Entirely in Spark — no plink2, no binaries**, plus a **Hail-style windowed-r² LD prune**
# MAGIC (distributed, per chromosome). The fit is verbatim FRAPOSA `fit_panel_basis`, so member
# MAGIC projection/classification downstream is unchanged.
# MAGIC
# MAGIC Pipeline (all distributed except the tiny driver-side eigendecomposition):
# MAGIC   1. **QC read** — pgenlib reads the panel pgen in variant-index BLOCKS (`mapInPandas`), subsets to the
# MAGIC      king.cutoff-unrelated samples, and keeps autosomal biallelic non-palindromic common SNVs
# MAGIC      (MAF ≥ maf, missing ≤ geno). Writes per-variant dose arrays to a transient Delta table. pgenlib is
# MAGIC      used ONLY here, in a simple blocked read — the heavy steps below never touch it.
# MAGIC   2. **LD prune** — per chromosome (`applyInPandas`, distributed): sliding `window_bp` window; drop a
# MAGIC      variant whose r² ≥ `r2` with an already-kept variant (greedy MIS, à la Hail `ld_prune` / plink2
# MAGIC      `--indep-pairwise`), keeping the earlier/higher-MAF one.
# MAGIC   3. **Fit** — the pruned panel is small (n_var × n_panel), so collect it to the driver and fit in
# MAGIC      numpy: standardize (per-variant mean/std, missing→0), Gram `XᵀX` (samples×samples) as one BLAS
# MAGIC      gemm → `eigh` → `V`, `s`, and loadings `U = X·(V/s)`. Verbatim `lib/prs_ancestry.fit_panel_basis`.
# MAGIC      This reproduces `lib/prs_ancestry.fit_panel_basis` — so `project_member`/`classify` are unchanged.
# MAGIC   4. **Persist** `pca_basis.npz` (loci, U, s, V, pcs_ref, mean, std, superpops) to the Volume.

# COMMAND ----------

dbutils.widgets.text("catalog", "genesis_workbench", "Catalog")
dbutils.widgets.text("schema", "genesis_schema", "Schema")
dbutils.widgets.text("panel_pgen", "", "Panel .pgen path (Volume)")
dbutils.widgets.text("panel_pvar_parquet", "", "Panel .pvar.parquet (CHROM,POS,REF,ALT in pgen order)")
dbutils.widgets.text("panel_psam", "", "Panel .psam (IID + SuperPop)")
dbutils.widgets.text("king_cutoff", "", "king.cutoff.out.id (related samples to exclude)")
dbutils.widgets.text("out_path", "", "Volume path to write pca_basis.npz")
dbutils.widgets.text("pca_model_name", "ancestry_pca", "UC model name for the PCA pyfunc")
dbutils.widgets.text("classifier_model_name", "ancestry_classifier", "UC model name for the classifier")
dbutils.widgets.text("experiment_name", "dbx_genesis_workbench_modules", "MLflow experiment tag")
dbutils.widgets.text("user_email", "a@b.com", "User Id/Email")
dbutils.widgets.text("sql_warehouse_id", "w123", "SQL Warehouse Id")
dbutils.widgets.text("n_pcs", "5", "PCs the classifier uses (pgsc_calc convention)")
dbutils.widgets.text("impute_mode", "at_mean", "Missing-locus fill at serve: zero | at_mean | mean_af (at_mean = historical FRAPOSA)")
dbutils.widgets.text("outlier_alpha", "0.001", "Mahalanobis P below this => PC-space outlier")
dbutils.widgets.text("min_coverage_frac", "0.1", "Min fraction of basis loci a sample must cover")
dbutils.widgets.text("dim_ref", "10", "Reference PC dimension (classify uses first 5)")
dbutils.widgets.text("maf", "0.05", "Min MAF")
dbutils.widgets.text("geno", "0.1", "Max per-variant missingness")
dbutils.widgets.text("r2", "0.05", "LD-prune r² threshold (prune if ≥)")
dbutils.widgets.text("window_bp", "1000000", "LD-prune window (bp)")
dbutils.widgets.text("block_size", "20000", "pgenlib variants per QC-read task")
dbutils.widgets.text("panel_version", "pgsc_HGDP+1kGP_v2_nohwe", "Basis provenance tag")

# COMMAND ----------

# Resolve the genesis_workbench wheel (for set_mlflow_experiment at model-registration time).
_catalog0 = dbutils.widgets.get("catalog"); _schema0 = dbutils.widgets.get("schema")
gwb_library_path = None
for lib in dbutils.fs.ls(f"/Volumes/{_catalog0}/{_schema0}/libraries"):
    if lib.name.startswith("genesis_workbench"):
        gwb_library_path = lib.path.replace("dbfs:", "")
print(f"Genesis Workbench library wheel: {gwb_library_path}")

# COMMAND ----------

# MAGIC %pip install pgenlib==0.94.1 zstandard==0.23.0 mlflow==2.22.0 scikit-learn==1.3.0 scipy==1.11.1 databricks-sdk==0.50.0 databricks-sql-connector==4.0.3 {gwb_library_path}
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

import os
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pyspark.sql.functions as F
from pyspark.sql.types import (StructType, StructField, StringType, LongType, IntegerType,
                               FloatType, ArrayType)

catalog = dbutils.widgets.get("catalog"); schema = dbutils.widgets.get("schema")
panel_pgen = dbutils.widgets.get("panel_pgen"); pvar_parquet = dbutils.widgets.get("panel_pvar_parquet")
panel_psam = dbutils.widgets.get("panel_psam"); king_cutoff = dbutils.widgets.get("king_cutoff")
out_path = dbutils.widgets.get("out_path"); dim_ref = int(dbutils.widgets.get("dim_ref"))
pca_model_name = dbutils.widgets.get("pca_model_name")
classifier_model_name = dbutils.widgets.get("classifier_model_name")
experiment_name = dbutils.widgets.get("experiment_name")
user_email = dbutils.widgets.get("user_email")
sql_warehouse_id = dbutils.widgets.get("sql_warehouse_id")
n_pcs = int(dbutils.widgets.get("n_pcs"))
impute_mode = dbutils.widgets.get("impute_mode").strip() or "at_mean"
outlier_alpha = float(dbutils.widgets.get("outlier_alpha"))
min_coverage_frac = float(dbutils.widgets.get("min_coverage_frac"))
MAF = float(dbutils.widgets.get("maf")); GENO = float(dbutils.widgets.get("geno"))
R2 = float(dbutils.widgets.get("r2")); WINDOW_BP = int(dbutils.widgets.get("window_bp"))
BLOCK = int(dbutils.widgets.get("block_size")); panel_version = dbutils.widgets.get("panel_version")
assert panel_pgen and pvar_parquet and panel_psam and king_cutoff and out_path, "paths are required"
spark.sql(f"USE CATALOG {catalog}"); spark.sql(f"USE SCHEMA {schema}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 1. Driver: unrelated-sample mask + SuperPop labels + pgen variant table

# COMMAND ----------

# psam (all pgen samples, in order) → IID + SuperPop; king.cutoff → related IIDs to drop.
psam = pd.read_csv(panel_psam, sep="\t")
psam.columns = [c.lstrip("#") for c in psam.columns]
all_iids = psam["IID"].astype(str).tolist()
sp_by_iid = dict(zip(psam["IID"].astype(str), psam["SuperPop"].astype(str)))
related = set()
with open(king_cutoff) as f:
    for line in f:
        parts = line.rstrip("\n").split("\t")
        related.add(parts[-1] if len(parts) > 1 else parts[0])   # IID column (last)
unrel_pos = np.array([i for i, s in enumerate(all_iids) if s not in related], dtype=np.int64)
unrel_iids = [all_iids[i] for i in unrel_pos]
superpops = np.array([sp_by_iid[s] for s in unrel_iids])
n_panel = len(unrel_iids); n_all = len(all_iids)
uniq, cnts = np.unique(superpops, return_counts=True)
print(f"panel: {n_all} total, {n_panel} unrelated | { {k: int(v) for k, v in zip(uniq, cnts)} }")

# pgen variant table (pgen order) — CHROM already 'chr'-stripped in the parquet. Read as ARROW
# (columnar), NOT pandas: at ~10^8 panel variants, pandas `.astype(str).to_numpy()` on chrom/ref/alt
# materializes ~15 GB of Python allele strings on the driver, which is what forced driver.memory down
# and the QC concurrency to local[*,2]. Arrow keeps them as packed bytes+offsets (~hundreds of MB/col)
# and preserves the FULL allele strings, so indels / single-char non-ACGT (e.g. 'N') filter EXACTLY as
# the string path did — byte-identical. Mirrors prs ref_00_build_panel_stats (Arrow, strings, no int codes).
pv_tbl = pq.read_table(pvar_parquet, columns=["CHROM", "POS", "REF", "ALT"])
n_pgen = pv_tbl.num_rows
print(f"pgen variants: {n_pgen}")

PGEN_B = spark.sparkContext.broadcast(panel_pgen)
UNREL_B = spark.sparkContext.broadcast(unrel_pos)
# pos → int64 numpy (small); chrom/ref/alt → contiguous Arrow arrays (compact, full strings), sliced
# per block in the QC read. Broadcast is ~2 GB (vs ~15 GB of Python-object arrays before).
PV_B = spark.sparkContext.broadcast({
    "chrom": pv_tbl["CHROM"].combine_chunks(),
    "pos": pv_tbl["POS"].to_numpy(zero_copy_only=False).astype(np.int64),
    "ref": pv_tbl["REF"].combine_chunks(),
    "alt": pv_tbl["ALT"].combine_chunks()})

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2. Distributed QC read (pgenlib blocks → transient Delta of per-variant dose arrays)

# COMMAND ----------

# QC = autosomal biallelic non-palindromic SNVs, MAF ≥ maf, missing ≤ geno. NO Hardy-Weinberg
# filter — deliberately. Pooled HWE across a multi-ancestry panel (HGDP+1kGP spans 6 superpops)
# rejects exactly the high-Fst, ancestry-informative variants PCA needs (Wahlund effect: real
# between-population allele-frequency structure reads as a pooled HWE departure). Measured on this
# panel: the variants a pooled-HWE filter drops skew ~2.7× higher in per-population Fst than the
# kept set — i.e. it discards ancestry signal, not noise. Dropping HWE grows the basis from ~47k to
# ~157k loci and is what the validated pipeline uses (panel_version pgsc_HGDP+1kGP_v2_nohwe). Do NOT
# re-add an HWE filter here without re-checking that Fst delta.
_PAL = ({"A", "T"}, {"C", "G"})
QC_SCHEMA = StructType([
    StructField("vidx", LongType()), StructField("chrom", StringType()), StructField("pos", LongType()),
    StructField("ref", StringType()), StructField("alt", StringType()),
    StructField("dose", ArrayType(FloatType())),   # length n_panel (unrelated), NaN = missing
])

def _qc_block(itr):
    import numpy as _np
    from pgenlib import PgenReader
    pv = PV_B.value; unrel = UNREL_B.value
    rd = PgenReader(PGEN_B.value.encode())
    n_all_ = rd.get_raw_sample_ct()
    try:
        for pdf in itr:
            for _, row in pdf.iterrows():
                start = int(row["start"]); stop = min(start + BLOCK, n_pgen)
                buf = _np.empty((stop - start, n_all_), dtype=_np.float32)
                rd.read_dosages_range(start, stop, buf, sample_maj=0)
                buf = buf[:, unrel]                     # subset to unrelated samples
                buf[buf < -0.5] = _np.nan
                # slice this block's metadata once (Arrow → py lists), not a per-variant scalar deref
                chrom_b = pv["chrom"].slice(start, stop - start).to_pylist()
                ref_b = pv["ref"].slice(start, stop - start).to_pylist()
                alt_b = pv["alt"].slice(start, stop - start).to_pylist()
                pos_b = pv["pos"][start:stop]
                out = []
                for k in range(stop - start):
                    c = chrom_b[k]; r = ref_b[k]; a = alt_b[k]
                    if not c or not c.isdigit() or int(c) < 1 or int(c) > 22:  # autosome
                        continue
                    if len(r) != 1 or len(a) != 1:                        # biallelic SNV
                        continue
                    if {r.upper(), a.upper()} in _PAL:                    # non-palindromic
                        continue
                    d = buf[k]; valid = ~_np.isnan(d)
                    nv = int(valid.sum())
                    if nv == 0 or (1 - nv / len(d)) > GENO:               # missingness
                        continue
                    af = float(d[valid].sum()) / (2.0 * nv)
                    if min(af, 1 - af) < MAF:                             # MAF
                        continue
                    out.append((start + k, c, int(pos_b[k]), r, a, [float(x) for x in d]))
                if out:
                    yield pd.DataFrame(out, columns=["vidx", "chrom", "pos", "ref", "alt", "dose"])
    finally:
        rd.close()

blocks = spark.createDataFrame(
    [(s,) for s in range(0, n_pgen, BLOCK)], ["start"]).repartition(max(1, n_pgen // BLOCK))
qc = blocks.mapInPandas(_qc_block, schema=QC_SCHEMA)
qc.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable("_pca_qc_dose")
qc = spark.table("_pca_qc_dose")
print(f"QC-passing variants: {qc.count()}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3. Prune + fit → the projectable model (shared `lib/pca_fit`)
# MAGIC The QC'd per-variant dose table is the source-agnostic hand-off: from here the LD-prune,
# MAGIC FRAPOSA standardize/eigh, and npz persist are the SAME code the in-cohort path (`01_compute_pca`)
# MAGIC uses — `lib/pca_fit.fit_pca_model`. This notebook is now just the reference-cohort adapter.

# COMMAND ----------

import sys
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "..", "lib")))
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "..", "model")))
import pca_fit

try:
    fit = pca_fit.fit_pca_model(
        spark, qc,
        sample_ids=unrel_iids, superpops=superpops,
        dim_ref=dim_ref, r2=R2, window_bp=WINDOW_BP,
        panel_version=panel_version, out_path=out_path,
    )
finally:
    spark.sql("DROP TABLE IF EXISTS _pca_qc_dose")   # transient QC store — cleaned up even on fit failure

print(f"npz: {out_path} | {fit['n_var']} loci · {fit['n_panel']} panel samples · dim_ref={fit['dim_ref']}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 4. Register `ancestry_pca` + `ancestry_classifier` at `@champion`
# MAGIC Same fitted arrays as the npz. PRS loads these when present; npz remains the gVCF-extract catalog
# MAGIC and the fallback if the registry is empty.

# COMMAND ----------

import mlflow
from mlflow import MlflowClient
from genesis_workbench.workbench import initialize
from genesis_workbench.models import set_mlflow_experiment
import log_ancestry_models as lm

_token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().getOrElse(None)
initialize(core_catalog_name=catalog, core_schema_name=schema,
           sql_warehouse_id=sql_warehouse_id, token=_token)

set_mlflow_experiment(experiment_tag=experiment_name, user_email=user_email, shared=True)
mlflow.set_registry_uri("databricks-uc")
mlflow.set_tracking_uri("databricks")

pca_uc = f"{catalog}.{schema}.{pca_model_name}"
clf_uc = f"{catalog}.{schema}.{classifier_model_name}"

with mlflow.start_run(run_name=f"ancestry-basis-{panel_version}") as run:
    mlflow.log_params({"panel_version": panel_version, "n_loci": fit["n_var"],
                       "n_panel": fit["n_panel"], "dim_ref": fit["dim_ref"],
                       "n_pcs": n_pcs, "impute_mode": impute_mode,
                       "outlier_alpha": outlier_alpha, "min_coverage_frac": min_coverage_frac,
                       "npz_path": out_path})
    lm.log_ancestry_pca(
        fit, panel_version=panel_version, uc_model_name=pca_uc, n_pcs=n_pcs,
        impute_mode=impute_mode, min_coverage_frac=min_coverage_frac,
        registered_model_name=pca_uc)
    lm.log_ancestry_classifier(
        fit, superpops, uc_model_name=clf_uc, n_pcs=n_pcs,
        outlier_alpha=outlier_alpha, min_coverage_frac=min_coverage_frac,
        registered_model_name=clf_uc)
    print(f"registered {pca_uc} and {clf_uc} | run {run.info.run_id}")

_client = MlflowClient()
for uc in (pca_uc, clf_uc):
    v = max(int(mv.version) for mv in _client.search_model_versions(f"name = '{uc}'"))
    _client.set_registered_model_alias(uc, "champion", v)
    print(f"  {uc}@champion -> v{v}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 5. Persist the panel PC1/PC2 cloud (app ancestry scatter)

# COMMAND ----------

ref_pcs = lm.build_reference_pcs(fit, superpops, n_pcs=2)
ref_tbl = f"{catalog}.{schema}.ancestry_pca_reference_pcs"
(spark.createDataFrame(ref_pcs)
 .withColumn("panel_version", F.lit(panel_version))
 .write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(ref_tbl))
print(f"wrote reference PC cloud ({len(ref_pcs)} panel samples) → {ref_tbl}")
