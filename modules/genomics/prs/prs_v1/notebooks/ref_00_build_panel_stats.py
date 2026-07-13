# Databricks notebook source
# MAGIC %md
# MAGIC # Build panel stats — `pgs_panel_afreq` + `pgs_panel_ref` in ONE reference-panel scan (classic; NOT serverless)
# MAGIC
# MAGIC Both panel-reference tables come from the **same** pgenlib pass over the HGDP+1kGP panel at the
# MAGIC registered PGS-union loci, so this builds them together (one scan, not two):
# MAGIC * **`pgs_panel_afreq`** — per-variant panel **effect-allele frequency** (for `2·AF` mean-imputation of
# MAGIC   truly-missing variants, plink2 `--score` / `--read-freq` semantics). This is a *byproduct*: the panel
# MAGIC   scoring already computes `af_alt` per variant to impute, so afreq is free.
# MAGIC * **`pgs_panel_ref`** — per-PGS × superpop `{mean, sd, quantiles}` the scorer standardizes against
# MAGIC   (`z_msp` / `z_admixed` + percentile lookups). `reference_distribution` becomes a pipeline OUTPUT
# MAGIC   (no offline plink2, no hand-pasted stats). Keyed `(pgs_id, superpop, panel_version)` + `weight_sha`.
# MAGIC
# MAGIC **Driver-bounded at genome-wide scale.** The union↔pgen match runs **per chromosome** (Arrow compute
# MAGIC on the pvar CHROM column — no 10^8-row Python string array; only ~union/22 union rows on the driver at
# MAGIC once), staging matches to Delta. The scan is distributed (`mapPartitions`, map-side combine); only the
# MAGIC per-PGS score vectors (`n_pgs × n_panel` ≈ a few MB) land on the driver. So a genome-wide union can't
# MAGIC OOM the driver regardless of node size — the whole-union `toPandas` + full-pvar string array were the
# MAGIC ceilings and are gone.
# MAGIC
# MAGIC Requires a prior `01_register_catalog` run (reads `pgs_weights` / `pgs_registry`). Panel scoring is the
# MAGIC one worker-scalable job here (panel × union is large) — TITRATE `num_workers` up for a faster build.

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

# COMMAND ----------
# MAGIC %pip install pgenlib==0.94.1
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

import os
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pyarrow.compute as pc
import pyspark.sql.functions as F
from pyspark import StorageLevel
from pyspark.sql.types import (StructType, StructField, StringType, DoubleType, LongType,
                               IntegerType, BooleanType, ArrayType)
from delta.tables import DeltaTable

catalog = dbutils.widgets.get("catalog"); schema = dbutils.widgets.get("schema")
panel_pgen = dbutils.widgets.get("panel_pgen"); pvar_parquet = dbutils.widgets.get("panel_pvar_parquet")
panel_psam = dbutils.widgets.get("panel_psam"); king_cutoff = dbutils.widgets.get("king_cutoff").strip()
panel_version = dbutils.widgets.get("panel_version").strip()
block_size = int(dbutils.widgets.get("block_size"))
pgs_filter = [x.strip() for x in dbutils.widgets.get("pgs_ids").split(",") if x.strip()]
assert panel_pgen and pvar_parquet and panel_psam and panel_version, "panel paths + panel_version are required"
spark.sql(f"USE CATALOG {catalog}"); spark.sql(f"USE SCHEMA {schema}")

QUANTILE_PROBS = [0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]  # stored in pgs_panel_ref.quantiles

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

# MAGIC %md
# MAGIC ### 2. Registered weights → union variants matched to the panel pgen (per-chrom, driver-bounded)
# MAGIC For each union variant find its GLOBAL pgen index (`gidx`) + whether the effect allele is the pgen ALT.
# MAGIC Looped per chromosome so the driver never holds the whole union or a full-pvar Python string array;
# MAGIC matches are staged to `_panel_match` (Delta). See the module notes — this was the OOM ceiling.

# COMMAND ----------

reg = spark.table("pgs_registry").select("pgs_id", "weight_sha")
if pgs_filter:
    reg = reg.where(F.col("pgs_id").isin(pgs_filter))
sha_by_pgs = {r["pgs_id"]: r["weight_sha"] for r in reg.collect()}
if not sha_by_pgs:
    dbutils.notebook.exit("0 — no registered PGS to score")

wq = spark.table("pgs_weights").where(F.col("pgs_id").isin(list(sha_by_pgs)))
uni = (wq.select("variant_id").distinct().select(F.split("variant_id", ":").alias("p"))
       .select(F.col("p")[0].alias("chrom"), F.col("p")[1].cast("long").alias("pos"),
               F.col("p")[2].alias("effect"), F.col("p")[3].alias("other")))

MATCH_SCHEMA = StructType([
    StructField("gidx", LongType()), StructField("variant_id", StringType()),
    StructField("effect_is_alt", BooleanType())])

# pvar read ONCE (Arrow columnar, ~GB not tens of GB); per-chrom membership via pyarrow.compute so the
# ~10^8-row CHROM column never becomes a numpy object array (that .astype(str) was ~5-6 GB of py strings).
pv = pq.read_table(pvar_parquet, columns=["CHROM", "POS", "REF", "ALT"])
chroms = sorted(r["chrom"] for r in uni.select("chrom").distinct().collect())
n_matched, first = 0, True
for ch in chroms:
    sel = pc.indices_nonzero(pc.equal(pv["CHROM"], ch)).to_numpy()      # this chrom's GLOBAL pgen indices
    if sel.size == 0:
        continue
    pvsub = pd.DataFrame({
        "gidx": sel.astype(np.int64),
        "pos": pc.take(pv["POS"], sel).to_numpy().astype(np.int64),
        "ref": pc.take(pv["REF"], sel).to_numpy(zero_copy_only=False),
        "alt": pc.take(pv["ALT"], sel).to_numpy(zero_copy_only=False)})
    uc = uni.where(F.col("chrom") == ch).toPandas()                     # ~union/22 rows, bounded
    if uc.empty:
        continue
    m = uc.merge(pvsub, on="pos")                                       # both restricted to chrom ch
    if m.empty:
        continue
    eff = m["effect"].str.upper(); oth = m["other"].str.upper()
    rf = m["ref"].str.upper(); al = m["alt"].str.upper()
    is_alt = (eff == al) & (oth == rf); is_ref = (eff == rf) & (oth == al)
    m = m[is_alt | is_ref].copy()
    if m.empty:
        continue
    m["effect_is_alt"] = is_alt[is_alt | is_ref].values
    m["variant_id"] = m["chrom"] + ":" + m["pos"].astype(str) + ":" + m["effect"] + ":" + m["other"]
    mc = m[["gidx", "variant_id", "effect_is_alt"]].drop_duplicates("variant_id")
    (spark.createDataFrame(mc, MATCH_SCHEMA).write.mode("overwrite" if first else "append")
     .option("overwriteSchema", "true").saveAsTable("_panel_match"))
    n_matched += len(mc); first = False
if first:
    spark.createDataFrame([], MATCH_SCHEMA).write.mode("overwrite").option(
        "overwriteSchema", "true").saveAsTable("_panel_match")
matched_df = spark.table("_panel_match")
print(f"matched union↔pgen: {n_matched:,} variants across {len(chroms)} chroms")

# COMMAND ----------

# MAGIC %md ### 3. variant → its (pgs_id, weight) list (one row per pgen variant)

# COMMAND ----------

vw = (matched_df.join(wq.select("pgs_id", "variant_id", "weight"), "variant_id")
      .groupBy("gidx", "variant_id", "effect_is_alt")
      .agg(F.collect_list(F.struct("pgs_id", "weight")).alias("pw")))

# COMMAND ----------

# MAGIC %md
# MAGIC ### 4. ONE panel scan → afreq (per variant) + per-(PGS×sample) score vectors (map-side combine)

# COMMAND ----------

KEEP_B = spark.sparkContext.broadcast(keep_mask)
PGEN_B = spark.sparkContext.broadcast(panel_pgen)
n_blocks = max(1, vw.count() // block_size)

def _scan_part(rows):
    """pgenlib read per variant → yield ('A', variant_id, af_effect, None) for afreq, and accumulate
    per-PGS Σ weight·dose vectors (pre-summed per partition), yielded once as ('S', pgs_id, None, vec)."""
    import pgenlib, numpy as _np
    reader = pgenlib.PgenReader(PGEN_B.value.encode())
    n_all = reader.get_raw_sample_ct(); buf = _np.empty(n_all, dtype=_np.int8)
    keep = KEEP_B.value; n_keep = int(keep.sum())
    acc = {}
    try:
        for r in rows:
            reader.read(int(r["gidx"]), buf)
            d = buf[keep].astype(_np.float64)                    # kept-sample dose; -1 = missing
            valid = d >= 0
            af_alt = float(d[valid].sum()) / (2.0 * int(valid.sum())) if valid.any() else 0.0
            dose_alt = _np.where(valid, d, 2.0 * af_alt)          # mean-impute missing → 2·AF
            eia = bool(r["effect_is_alt"])
            dose_eff = dose_alt if eia else (2.0 - dose_alt)
            af_eff = af_alt if eia else (1.0 - af_alt)
            yield ("A", r["variant_id"], float(af_eff), None)     # afreq row (per variant)
            for pw in r["pw"]:
                v = acc.get(pw["pgs_id"])
                if v is None:
                    v = _np.zeros(n_keep); acc[pw["pgs_id"]] = v
                v += dose_eff * float(pw["weight"])               # map-side combine
        for pgs_id, vec in acc.items():
            yield ("S", pgs_id, None, vec.tolist())
    finally:
        reader.close()

# scan ONCE; persist so both outputs (afreq write + score aggregate) reuse it without re-reading the pgen
scanned = vw.rdd.repartition(n_blocks).mapPartitions(_scan_part).persist(StorageLevel.MEMORY_AND_DISK)

# afreq: per-variant effect-allele frequency → pgs_panel_afreq
AFREQ_SCHEMA = StructType([StructField("variant_id", StringType()), StructField("af_effect", DoubleType())])
afreq_df = spark.createDataFrame(scanned.filter(lambda t: t[0] == "A").map(lambda t: (t[1], t[2])), AFREQ_SCHEMA)
spark.sql("CREATE TABLE IF NOT EXISTS pgs_panel_afreq (variant_id STRING, af_effect DOUBLE) USING DELTA")
afreq_df.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable("pgs_panel_afreq")
n_afreq = spark.table("pgs_panel_afreq").count()
print(f"wrote {n_afreq:,} panel afreq rows → pgs_panel_afreq")

# scores: per-partition partial vectors summed per PGS → {pgs_id: panel score vector (n_panel)}
scores = (scanned.filter(lambda t: t[0] == "S").map(lambda t: (t[1], np.array(t[3])))
          .aggregateByKey(np.zeros(n_panel), lambda a, v: a + v, lambda a, b: a + b).collectAsMap())
scanned.unpersist()
print(f"scored {len(scores)} PGS on the panel")

# COMMAND ----------

# MAGIC %md ### 5. Per-superpop {mean, sd, quantiles} → MERGE pgs_panel_ref

# COMMAND ----------

rows = []
for pgs_id, vec in scores.items():
    for sp in np.unique(superpops):
        s = vec[superpops == sp]
        if s.size == 0:
            continue
        rows.append((pgs_id, str(sp), float(np.mean(s)),
                     float(np.std(s, ddof=1)) if s.size > 1 else 0.0,
                     [float(q) for q in np.quantile(s, QUANTILE_PROBS)] if s.size > 1 else None,
                     int(s.size), panel_version, sha_by_pgs[pgs_id]))

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
print(f"MERGEd {len(rows)} pgs_panel_ref rows ({len(scores)} PGS × superpops) @ panel_version={panel_version} "
      f"(quantiles at probs {QUANTILE_PROBS})")
