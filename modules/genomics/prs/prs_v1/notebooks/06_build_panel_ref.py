# Databricks notebook source
# MAGIC %md
# MAGIC # PRS panel reference — COMPUTE per-PGS `pgs_panel_ref` on the reference panel (classic; NOT serverless)
# MAGIC
# MAGIC Scores the frozen HGDP+1kGP panel against every registered PGS and derives the per-superpop
# MAGIC `{mean, sd}` the scorer standardizes against (`z_msp` / `z_admixed`). This is the **compute-from-panel**
# MAGIC builder the register lib anticipated — it makes `reference_distribution` a pipeline OUTPUT instead of
# MAGIC an offline curation (no plink2, no hand-pasted stats), so z scales to 100+ PGS on-cluster.
# MAGIC
# MAGIC **Spark-native, no binaries** (pgenlib for the pgen dose; distributed scoring in Spark), mirroring
# MAGIC `04_build_panel_afreq` / the PCA basis build. Panel scoring genome-wide is the one worker-scalable
# MAGIC job here (panel × union is large) — TITRATE `num_workers` up for a faster one-time build.
# MAGIC
# MAGIC Pipeline: registered `pgs_weights` → union variants matched to the panel `.pvar` (effect-orientation)
# MAGIC → pgenlib reads each variant's panel dose ONCE (missing → 2·AF, matching plink2 `--score`
# MAGIC mean-imputation) → distributed `aggregateByKey` sums `weight·dose` per (PGS × panel sample) into a
# MAGIC per-PGS score vector → group panel samples by SuperPop → `{mean, sd}` → MERGE `pgs_panel_ref`.
# MAGIC Keyed on `(pgs_id, superpop, panel_version)` + `weight_sha`, so it restates one PGS safely.
# MAGIC
# MAGIC Requires a prior `00_register_catalog` run (reads `pgs_weights` / `pgs_registry`).

# COMMAND ----------

dbutils.widgets.text("catalog", "genesis_workbench", "Catalog")
dbutils.widgets.text("schema", "genesis_schema", "Schema")
dbutils.widgets.text("panel_pgen", "", "Panel .pgen path (Volume)")
dbutils.widgets.text("panel_pvar_parquet", "", "Panel .pvar.parquet (CHROM,POS,REF,ALT in pgen order)")
dbutils.widgets.text("panel_psam", "", "Panel .psam (IID + SuperPop, pgen order)")
dbutils.widgets.text("king_cutoff", "", "king.cutoff.out.id (related IIDs to drop; empty = use all panel samples)")
dbutils.widgets.text("panel_version", "", "panel_version to stamp (MUST match the curation version the scorer normalizes against)")
dbutils.widgets.text("pgs_ids", "", "Restrict to these PGS (comma-sep; empty = all registered)")
dbutils.widgets.text("block_size", "20000", "pgenlib variants per read task")

catalog = dbutils.widgets.get("catalog"); schema = dbutils.widgets.get("schema")
panel_pgen = dbutils.widgets.get("panel_pgen"); pvar_parquet = dbutils.widgets.get("panel_pvar_parquet")
panel_psam = dbutils.widgets.get("panel_psam"); king_cutoff = dbutils.widgets.get("king_cutoff").strip()
panel_version = dbutils.widgets.get("panel_version").strip()
block_size = int(dbutils.widgets.get("block_size"))
assert panel_pgen and pvar_parquet and panel_psam and panel_version, "panel paths + panel_version are required"

# COMMAND ----------

# MAGIC %pip install pgenlib
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

import os
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pyspark.sql.functions as F
from pyspark.sql.types import StructType, StructField, StringType, DoubleType, LongType, IntegerType, ArrayType
from delta.tables import DeltaTable

catalog = dbutils.widgets.get("catalog"); schema = dbutils.widgets.get("schema")
panel_pgen = dbutils.widgets.get("panel_pgen"); pvar_parquet = dbutils.widgets.get("panel_pvar_parquet")
panel_psam = dbutils.widgets.get("panel_psam"); king_cutoff = dbutils.widgets.get("king_cutoff").strip()
panel_version = dbutils.widgets.get("panel_version").strip()
block_size = int(dbutils.widgets.get("block_size"))
pgs_filter = [x.strip() for x in dbutils.widgets.get("pgs_ids").split(",") if x.strip()]
spark.sql(f"USE CATALOG {catalog}"); spark.sql(f"USE SCHEMA {schema}")

# COMMAND ----------

# MAGIC %md ### 1. Panel samples → SuperPop (pgen order), optional king-cutoff to unrelated

# COMMAND ----------

psam = pd.read_csv(panel_psam, sep="\t")
psam.columns = [c.lstrip("#") for c in psam.columns]
all_iids = psam["IID"].astype(str).to_numpy()
superpop_all = psam["SuperPop"].astype(str).to_numpy()          # pgen sample order
keep_mask = np.ones(len(all_iids), dtype=bool)
if king_cutoff:
    related = set()
    with open(king_cutoff) as f:
        for line in f:
            p = line.rstrip("\n").split("\t"); related.add(p[-1] if len(p) > 1 else p[0])
    keep_mask = np.array([iid not in related for iid in all_iids])
superpops = superpop_all[keep_mask]                             # labels for the kept panel samples
n_panel = int(keep_mask.sum())
uniq, cnts = np.unique(superpops, return_counts=True)
print(f"panel: {len(all_iids)} total, {n_panel} used | {dict(zip(uniq.tolist(), cnts.tolist()))}")

# COMMAND ----------

# MAGIC %md ### 2. Registered weights → union variants matched to the panel pgen (effect-orientation)

# COMMAND ----------

reg = spark.table("pgs_registry").select("pgs_id", "weight_sha")
if pgs_filter:
    reg = reg.where(F.col("pgs_id").isin(pgs_filter))
sha_by_pgs = {r["pgs_id"]: r["weight_sha"] for r in reg.collect()}
if not sha_by_pgs:
    dbutils.notebook.exit("0 — no registered PGS to score")

wq = spark.table("pgs_weights").where(F.col("pgs_id").isin(list(sha_by_pgs)))
uni_pd = (wq.select("variant_id").distinct()
          .select(F.split("variant_id", ":").alias("p"))
          .select(F.col("p")[0].alias("chrom"), F.col("p")[1].cast("long").alias("pos"),
                  F.col("p")[2].alias("effect"), F.col("p")[3].alias("other"))).toPandas()

# pvar (driver) → global pgen index (gidx) of each union position; vectorised integer-key membership
# (no per-row Python string over the ~10^8-row pvar — same approach as 04_build_panel_afreq).
pv = pq.read_table(pvar_parquet, columns=["CHROM", "POS", "REF", "ALT"])
import pyarrow.compute as pc
chrom_np = pv["CHROM"].to_numpy(zero_copy_only=False).astype(str)
pos_np = pv["POS"].to_numpy().astype(np.int64)
_cats = sorted(set(chrom_np.tolist()) | set(uni_pd["chrom"].astype(str).tolist()))
pv_key = (pd.Categorical(chrom_np, categories=_cats).codes.astype(np.int64) << 40) | pos_np
uni_key = ((pd.Categorical(uni_pd["chrom"].astype(str), categories=_cats).codes.astype(np.int64) << 40)
           | uni_pd["pos"].astype(np.int64).to_numpy())
gidx = np.nonzero(np.isin(pv_key, uni_key))[0]
pvsub = pd.DataFrame({"gidx": gidx, "chrom": chrom_np[gidx], "pos": pos_np[gidx],
                      "ref": pc.take(pv["REF"], gidx).to_numpy(zero_copy_only=False),
                      "alt": pc.take(pv["ALT"], gidx).to_numpy(zero_copy_only=False)})
m = uni_pd.merge(pvsub, on=["chrom", "pos"])
eff = m["effect"].str.upper(); oth = m["other"].str.upper(); rf = m["ref"].str.upper(); al = m["alt"].str.upper()
is_alt = (eff == al) & (oth == rf); is_ref = (eff == rf) & (oth == al)
m = m[is_alt | is_ref].copy(); m["effect_is_alt"] = is_alt[is_alt | is_ref].values
m["variant_id"] = m["chrom"] + ":" + m["pos"].astype(str) + ":" + m["effect"] + ":" + m["other"]
matched = m[["gidx", "variant_id", "effect_is_alt"]].drop_duplicates("variant_id")
print(f"matched union↔pgen: {len(matched):,} variants")

# variant → its (pgs_id, weight) list (a variant may score into several PGS); one row per gidx.
vw = (spark.createDataFrame(matched).join(wq.select("pgs_id", "variant_id", "weight"), "variant_id")
      .groupBy("gidx", "effect_is_alt")
      .agg(F.collect_list(F.struct("pgs_id", "weight")).alias("pw")))

# COMMAND ----------

# MAGIC %md ### 3. Distributed panel score: pgenlib dose per variant → Σ weight·dose per (PGS × sample)

# COMMAND ----------

KEEP_B = spark.sparkContext.broadcast(keep_mask)
PGEN_B = spark.sparkContext.broadcast(panel_pgen)
n_blocks = max(1, vw.count() // block_size)

def _score_part(rows):
    import pgenlib, numpy as _np
    reader = pgenlib.PgenReader(PGEN_B.value.encode())
    n_all = reader.get_raw_sample_ct(); buf = _np.empty(n_all, dtype=_np.int8)
    keep = KEEP_B.value
    try:
        for r in rows:
            reader.read(int(r["gidx"]), buf)
            d = buf[keep].astype(_np.float64)               # dose (kept samples), -1 = missing
            valid = d >= 0
            af_alt = float(d[valid].sum()) / (2.0 * int(valid.sum())) if valid.any() else 0.0
            dose_alt = _np.where(valid, d, 2.0 * af_alt)     # mean-impute missing → 2·AF (plink2 --score)
            dose_eff = dose_alt if r["effect_is_alt"] else (2.0 - dose_alt)
            for pw in r["pw"]:
                yield (pw["pgs_id"], dose_eff * float(pw["weight"]))
    finally:
        reader.close()

# aggregateByKey with map-side combine: each partition pre-sums per PGS → only ~n_pgs vectors shuffle.
scores = (vw.rdd.repartition(n_blocks).mapPartitions(_score_part)
          .aggregateByKey(np.zeros(n_panel), lambda a, v: a + v, lambda a, b: a + b)
          .collectAsMap())                                   # {pgs_id: panel score vector (n_panel)}
print(f"scored {len(scores)} PGS on the panel")

# COMMAND ----------

# MAGIC %md ### 4. Per-superpop {mean, sd} → MERGE pgs_panel_ref (keyed pgs_id, superpop, panel_version)

# COMMAND ----------

rows = []
for pgs_id, vec in scores.items():
    for sp in np.unique(superpops):
        s = vec[superpops == sp]
        if s.size == 0:
            continue
        rows.append((pgs_id, str(sp), float(np.mean(s)),
                     float(np.std(s, ddof=1)) if s.size > 1 else 0.0,
                     None, int(s.size), panel_version, sha_by_pgs[pgs_id]))

PANEL_SCHEMA = StructType([
    StructField("pgs_id", StringType()), StructField("superpop", StringType()),
    StructField("mean", DoubleType()), StructField("sd", DoubleType()),
    StructField("quantiles", ArrayType(DoubleType())), StructField("n_panel", IntegerType()),
    StructField("panel_version", StringType()), StructField("weight_sha", StringType())])
ref_df = spark.createDataFrame(rows, PANEL_SCHEMA)

spark.sql("""CREATE TABLE IF NOT EXISTS pgs_panel_ref (
    pgs_id STRING, superpop STRING, mean DOUBLE, sd DOUBLE,
    quantiles ARRAY<DOUBLE>, n_panel INT, panel_version STRING, weight_sha STRING) USING DELTA""")
(DeltaTable.forName(spark, f"{catalog}.{schema}.pgs_panel_ref").alias("t")
 .merge(ref_df.alias("s"),
        "t.pgs_id = s.pgs_id AND t.superpop = s.superpop AND t.panel_version = s.panel_version")
 .whenMatchedUpdateAll().whenNotMatchedInsertAll().execute())
print(f"MERGEd {len(rows)} pgs_panel_ref rows ({len(scores)} PGS × superpops) @ panel_version={panel_version}")
