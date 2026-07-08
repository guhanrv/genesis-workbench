# Databricks notebook source
# MAGIC %md
# MAGIC # Build `pgs_panel_afreq` — panel allele frequencies via pgenlib (distributed; NO plink2)
# MAGIC
# MAGIC Computes, for each registered PGS-union variant, the panel frequency of its **effect allele** from
# MAGIC the HGDP+1kGP reference pgen — using **pgenlib** (`%pip`, no binary/init-script), distributed over
# MAGIC variant blocks with `mapInPandas`. Parity-validated $0 locally: pgenlib AF == `plink2 --freq`
# MAGIC ALT_FREQS (max|Δ|≈5e-7 on chr22). Output feeds `dense_fill(af_effect=…)` for `2·AF` mean-imputation
# MAGIC of truly-missing variants (pgsc_calc/plink2 `--read-freq` semantics).
# MAGIC
# MAGIC Classic cluster only (pgenlib random-access reads the .pgen). NOTE: pgenlib reading the .pgen from a
# MAGIC UC Volume (FUSE) is the piece to validate on-cluster — validate on a small `pgs_ids` first.

# COMMAND ----------

dbutils.widgets.text("catalog", "genesis_workbench", "Catalog")
dbutils.widgets.text("schema", "genesis_schema", "Schema")
dbutils.widgets.text("panel_pgen", "", "Panel .pgen path (Volume)")
dbutils.widgets.text("panel_pvar_parquet", "", "Panel .pvar.parquet (CHROM,POS,REF,ALT in pgen order)")
dbutils.widgets.text("pgs_ids", "", "Restrict to these PGS' variants (comma-sep; empty = all registered)")
dbutils.widgets.text("block_size", "20000", "pgenlib variant indices per task")

catalog = dbutils.widgets.get("catalog"); schema = dbutils.widgets.get("schema")

# COMMAND ----------
# MAGIC %pip install pgenlib
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

import os
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pyarrow.compute as pc
import pyspark.sql.functions as F
from pyspark.sql.types import StructType, StructField, StringType, DoubleType

catalog = dbutils.widgets.get("catalog"); schema = dbutils.widgets.get("schema")
panel_pgen = dbutils.widgets.get("panel_pgen"); pvar_parquet = dbutils.widgets.get("panel_pvar_parquet")
pgs_filter = [x.strip() for x in dbutils.widgets.get("pgs_ids").split(",") if x.strip()]
block_size = int(dbutils.widgets.get("block_size"))
spark.sql(f"USE CATALOG {catalog}"); spark.sql(f"USE SCHEMA {schema}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 1. Map the PGS-union variants → pgen variant indices (driver-side, via the ordered pvar)
# MAGIC The pvar.parquet row order == pgen variant index. Restrict to the union's (chrom,pos) first so the
# MAGIC join stays small, then match allele sets and record the effect-allele orientation.

# COMMAND ----------

wq = spark.table("pgs_weights")
if pgs_filter:
    wq = wq.where(F.col("pgs_id").isin(pgs_filter))
# union variants (effect-oriented ids): chrom:pos:effect:other
uni = (wq.select("variant_id").distinct().select(F.split("variant_id", ":").alias("p"))
       .select(F.col("p")[0].alias("chrom"), F.col("p")[1].cast("long").alias("pos"),
               F.col("p")[2].alias("effect"), F.col("p")[3].alias("other")))
uni_pd = uni.toPandas()
want_pos = set(zip(uni_pd["chrom"].astype(str), uni_pd["pos"].astype("int64")))
print(f"union variants: {len(uni_pd):,} ({len(want_pos):,} distinct positions)")

# ordered pvar (driver) → keep only rows at union positions, with their global pgen index
pv = pq.read_table(pvar_parquet, columns=["CHROM", "POS", "REF", "ALT"])
chrom_np = pv["CHROM"].to_numpy(zero_copy_only=False).astype(str)
pos_np = pv["POS"].to_numpy()
# boolean mask for union positions (vectorised set-membership via pandas)
key = pd.Series(list(map(lambda cp: f"{cp[0]}:{cp[1]}", zip(chrom_np, pos_np))))
want_key = {f"{c}:{p}" for c, p in want_pos}
mask = key.isin(want_key).to_numpy()
gidx = np.nonzero(mask)[0]
pvsub = pd.DataFrame({"gidx": gidx, "chrom": chrom_np[gidx], "pos": pos_np[gidx],
                      "ref": pv["REF"].to_numpy(zero_copy_only=False)[gidx],
                      "alt": pv["ALT"].to_numpy(zero_copy_only=False)[gidx]})
print(f"pvar rows at union positions: {len(pvsub):,}")

# match union (effect,other) to pvar (ref,alt); record whether effect == ALT (else effect == REF)
m = uni_pd.merge(pvsub, on=["chrom", "pos"])
eff = m["effect"].str.upper(); oth = m["other"].str.upper(); rf = m["ref"].str.upper(); al = m["alt"].str.upper()
is_alt = (eff == al) & (oth == rf)
is_ref = (eff == rf) & (oth == al)
m = m[is_alt | is_ref].copy()
m["effect_is_alt"] = is_alt[is_alt | is_ref].values
m["variant_id"] = m["chrom"] + ":" + m["pos"].astype(str) + ":" + m["effect"] + ":" + m["other"]
matched = m[["gidx", "variant_id", "effect_is_alt"]].drop_duplicates("variant_id")
print(f"matched union↔pgen: {len(matched):,} variants")

# COMMAND ----------

# MAGIC %md ### 2. pgenlib AF per variant (distributed over index blocks) → orient to effect → store

# COMMAND ----------

n_blocks = max(1, (len(matched) + block_size - 1) // block_size)
tasks = spark.createDataFrame(matched).repartition(n_blocks)
PGEN_B = spark.sparkContext.broadcast(panel_pgen)

OUT = StructType([StructField("variant_id", StringType()), StructField("af_effect", DoubleType())])

def _afreq_block(itr):
    import pgenlib, numpy as _np, pandas as _pd
    reader = pgenlib.PgenReader(PGEN_B.value.encode())
    n = reader.get_raw_sample_ct(); buf = _np.empty(n, dtype=_np.int8)
    try:
        for pdf in itr:
            out = []
            for _, r in pdf.iterrows():
                reader.read(int(r["gidx"]), buf)
                valid = buf >= 0
                af_alt = float(buf[valid].sum()) / (2.0 * int(valid.sum())) if valid.any() else 0.0
                af_effect = af_alt if r["effect_is_alt"] else (1.0 - af_alt)
                out.append((r["variant_id"], af_effect))
            if out:
                yield _pd.DataFrame(out, columns=["variant_id", "af_effect"])
    finally:
        reader.close()

afreq = tasks.mapInPandas(_afreq_block, schema=OUT)
spark.sql("""CREATE TABLE IF NOT EXISTS pgs_panel_afreq (variant_id STRING, af_effect DOUBLE) USING DELTA""")
afreq.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable("pgs_panel_afreq")
n = spark.table("pgs_panel_afreq").count()
print(f"wrote {n:,} panel afreq rows → {catalog}.{schema}.pgs_panel_afreq")
