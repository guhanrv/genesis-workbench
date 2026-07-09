# Databricks notebook source
# MAGIC %md
# MAGIC # Build the frozen FRAPOSA PCA basis — Spark-native, distributed (classic; NOT serverless)
# MAGIC
# MAGIC Produces `pca_basis.npz` (the fixed HGDP+1kGP basis `05_build_sample_ancestry` projects members onto)
# MAGIC **entirely in Spark — no plink2, no binaries.** This is the ancestry analogue of `pgs_panel_ref`: a
# MAGIC one-time reference artifact, but now built with the same distributed-PCA pattern genesis's
# MAGIC `pca_v1/01_compute_pca` and Databricks' `cspray` use (Spark ML PCA = a distributed sample-covariance
# MAGIC eigendecomposition), plus a **Hail-style windowed-r² LD prune** done distributed (per chromosome).
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
dbutils.widgets.text("dim_ref", "10", "Reference PC dimension (classify uses first 5)")
dbutils.widgets.text("maf", "0.05", "Min MAF")
dbutils.widgets.text("geno", "0.1", "Max per-variant missingness")
dbutils.widgets.text("r2", "0.05", "LD-prune r² threshold (prune if ≥)")
dbutils.widgets.text("window_bp", "1000000", "LD-prune window (bp)")
dbutils.widgets.text("block_size", "20000", "pgenlib variants per QC-read task")
dbutils.widgets.text("panel_version", "pgsc_HGDP+1kGP_v1", "Basis provenance tag")

# COMMAND ----------

# MAGIC %pip install pgenlib zstandard
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

import os
import numpy as np
import pandas as pd
import pyspark.sql.functions as F
from pyspark.sql.types import (StructType, StructField, StringType, LongType, IntegerType,
                               FloatType, ArrayType)

catalog = dbutils.widgets.get("catalog"); schema = dbutils.widgets.get("schema")
panel_pgen = dbutils.widgets.get("panel_pgen"); pvar_parquet = dbutils.widgets.get("panel_pvar_parquet")
panel_psam = dbutils.widgets.get("panel_psam"); king_cutoff = dbutils.widgets.get("king_cutoff")
out_path = dbutils.widgets.get("out_path"); dim_ref = int(dbutils.widgets.get("dim_ref"))
MAF = float(dbutils.widgets.get("maf")); GENO = float(dbutils.widgets.get("geno"))
R2 = float(dbutils.widgets.get("r2")); WINDOW_BP = int(dbutils.widgets.get("window_bp"))
BLOCK = int(dbutils.widgets.get("block_size")); panel_version = dbutils.widgets.get("panel_version")
assert panel_pgen and pvar_parquet and panel_psam and king_cutoff and out_path, "paths are required"
spark.sql(f"USE CATALOG {catalog}"); spark.sql(f"USE SCHEMA {schema}")
dim_stu = dim_ref * 2; dim_online = dim_stu * 2

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

# pgen variant table (pgen order) — CHROM already 'chr'-stripped in the parquet.
pvar = spark.read.parquet(pvar_parquet).toPandas()
pvar["vidx"] = np.arange(len(pvar), dtype=np.int64)
n_pgen = len(pvar)
print(f"pgen variants: {n_pgen}")

PGEN_B = spark.sparkContext.broadcast(panel_pgen)
UNREL_B = spark.sparkContext.broadcast(unrel_pos)
# broadcast pgen-ordered chrom/pos/ref/alt for the QC-read tasks
PV_B = spark.sparkContext.broadcast({
    "chrom": pvar["CHROM"].astype(str).to_numpy(), "pos": pvar["POS"].to_numpy(np.int64),
    "ref": pvar["REF"].astype(str).to_numpy(), "alt": pvar["ALT"].astype(str).to_numpy()})

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2. Distributed QC read (pgenlib blocks → transient Delta of per-variant dose arrays)

# COMMAND ----------

_PAL = ({"A", "T"}, {"C", "G"})
QC_SCHEMA = StructType([
    StructField("vidx", LongType()), StructField("chrom", StringType()), StructField("pos", LongType()),
    StructField("ref", StringType()), StructField("alt", StringType()),
    StructField("dose", ArrayType(FloatType())),   # length n_panel (unrelated), NaN = missing
])

def _qc_block(itr):
    import numpy as _np
    from pgenlib import PgenReader
    pv = PV_B.value; unrel = UNREL_B.value; n_all_ = None
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
                out = []
                for k in range(stop - start):
                    gi = start + k
                    c = str(pv["chrom"][gi]); r = str(pv["ref"][gi]); a = str(pv["alt"][gi])
                    if not c.isdigit() or int(c) < 1 or int(c) > 22:      # autosome
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
                    out.append((gi, c, int(pv["pos"][gi]), r, a, [float(x) for x in d]))
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
# MAGIC ### 3. Distributed LD prune (per chromosome; Hail-style windowed greedy r²)

# COMMAND ----------

PRUNE_SCHEMA = StructType([
    StructField("vidx", LongType()), StructField("chrom", StringType()),
    StructField("pos", LongType()), StructField("ref", StringType()), StructField("alt", StringType())])

def _prune_chrom(pdf):
    import numpy as _np
    from bisect import bisect_left
    pdf = pdf.sort_values("pos")
    pos = pdf["pos"].to_numpy()
    # each group is one CHUNK_BP position slice of a chromosome (not a whole chromosome), so D is
    # bounded (~0.5 GB) regardless of chromosome size. stack without .tolist()'s Python list-of-lists
    # transient and standardize in-place in float32 to keep peak memory ~2×D.
    D = _np.stack(pdf["dose"].to_numpy()).astype(_np.float32, copy=False)   # (m, n_panel)
    mean = _np.nanmean(D, 1, keepdims=True).astype(_np.float32)
    sd = _np.nanstd(D, 1, keepdims=True).astype(_np.float32); sd[sd == 0] = 1
    inv = 1.0 / D.shape[1]
    Z = D - mean; Z /= sd; _np.nan_to_num(Z, copy=False)
    del D, mean, sd                                             # free dose matrix before the r² loop
    kpos, kidx = [], []
    keep = _np.zeros(len(pdf), dtype=bool)
    for i in range(len(pdf)):
        lo = bisect_left(kpos, pos[i] - WINDOW_BP)
        if lo < len(kidx):
            r2 = (Z[kidx[lo:]].astype(_np.float64) @ Z[i].astype(_np.float64) * inv) ** 2
            if (r2 >= R2).any():
                continue
        kpos.append(int(pos[i])); kidx.append(i); keep[i] = True
    out = pdf.loc[keep, ["vidx", "chrom", "pos", "ref", "alt"]]
    return out

# Partition each chromosome into CHUNK_BP position slices so no single prune task materializes a
# whole chromosome's dose matrix (chr2 ≈ 461k×3330 ≈ 6 GB → OOM). This is Hail ld_prune's stage-1
# "local prune per partition": greedy windowed prune within each slice. The only approximation vs a
# whole-chromosome prune is at slice seams (variants in the first WINDOW_BP of a slice don't see kept
# variants just across the boundary → a few extra retained loci), which is immaterial for a PCA basis.
CHUNK_BP = 25_000_000
qc_chunked = qc.withColumn("chunk", (F.col("pos") / F.lit(CHUNK_BP)).cast("long"))
kept = qc_chunked.groupBy("chrom", "chunk").applyInPandas(_prune_chrom, schema=PRUNE_SCHEMA)
kept = kept.orderBy(F.col("chrom").cast("int"), "pos")            # stable variant order
kept_pd = kept.toPandas()
n_var = len(kept_pd)
order_by_vidx = {int(v): i for i, v in enumerate(kept_pd["vidx"].tolist())}
print(f"pruned loci: {n_var}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 4. Fit on the driver — collect the pruned panel (small) → BLAS XᵀX → eigh → loadings

# COMMAND ----------

# The pruned panel is small (n_var × n_panel ≈ 2 GB), so collect it once and fit on the driver:
# XᵀX is a single BLAS gemm — not a distributed treeReduce of n_var np.outer calls — and we avoid
# re-scanning the multi-GB QC store for a separate loadings pass. The standardize/eigh/loadings below
# are verbatim lib/prs_ancestry.fit_panel_basis (FRAPOSA), so the basis is identical to the validated
# science. (Requires spark.driver.maxResultSize ≥ ~4g for the ~2 GB collect.)
kept_rows = qc.join(F.broadcast(kept.select("vidx")), "vidx").select("vidx", "dose").collect()
vidx_arr = np.fromiter((r["vidx"] for r in kept_rows), np.int64, len(kept_rows))
X = np.stack([np.asarray(r["dose"], dtype=np.float32) for r in kept_rows])   # (n_var, n_panel), NaN=missing
X = X[np.argsort([order_by_vidx[int(v)] for v in vidx_arr])]                 # stable (chrom, pos) order
assert X.shape[0] == n_var, f"collected {X.shape[0]} != pruned {n_var}"
del kept_rows, vidx_arr

# FRAPOSA standardize: per-variant mean/std (ddof=0), cast to float32, missing → 0 (verbatim fit_panel_basis).
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
U_on = (X @ (V_on / s_on)).astype(np.float64)                   # (n_var × dim_online)
print(f"eigendecomposition (driver): top s = {np.round(s_on[:dim_ref], 1)}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 5. Persist pca_basis.npz (same shape as lib/prs_ancestry.fit_panel_basis output)

# COMMAND ----------

local_npz = "/local_disk0/pca_basis.npz"
np.savez(local_npz,
         loci_chrom=kept_pd["chrom"].astype(str).to_numpy(),
         loci_pos=kept_pd["pos"].to_numpy(np.int64),
         loci_ref=kept_pd["ref"].astype(str).to_numpy(),
         loci_alt=kept_pd["alt"].astype(str).to_numpy(),
         U_on=U_on, s_on=s_on, V_on=V_on, pcs_ref=pcs_ref, mean=mean, std=std,
         dim_ref=dim_ref, dim_stu=dim_stu,
         panel_iids=np.array(unrel_iids), superpops=superpops, panel_version=np.array(panel_version))
dbutils.fs.cp("file:" + local_npz, out_path)
spark.sql("DROP TABLE IF EXISTS _pca_qc_dose")
print(f"wrote basis → {out_path} | n_loci={n_var} n_panel={n_panel} "
      f"dim_online={dim_online} panel_version={panel_version}")
