# Databricks notebook source
# MAGIC %md
# MAGIC # PRS extract — per-sample gVCF → dosage store (classic; per-sample parallel)
# MAGIC
# MAGIC Extracts each sample's **effect-allele dosage** at the union of registered catalog variants and
# MAGIC MERGE-upserts the long `dosage(sample_id, variant_id, dose)` store. Extraction is the real cost
# MAGIC (one gVCF pass/sample), so it is done **once** and reused; the scorer never re-reads a VCF.
# MAGIC
# MAGIC **gVCF path (this notebook):** the ported pysam END-block+FASTA kernel (`lib/gvcf_dose.py`) — plink2
# MAGIC and Glow both drop gVCF `END=` REF blocks (~70% coverage loss), so this cannot be Glow/SQL. Only
# MAGIC **covered** (`had_record`) variants are written (real dose, or 0 for a covered REF block); truly-missing
# MAGIC variants are omitted so the scorer treats them as 0 and `n_variants_matched` = true coverage.
# MAGIC *(Hard-called / imputed VCF is the Glow `DS/HDS/GT` path in `00_ingest_vcf.py`; adapting its output to
# MAGIC this long store is a follow-up — this notebook implements the primary gVCF path.)*
# MAGIC
# MAGIC **Distribution:** one Spark task per sample (`mapInPandas`); the union catalog + precomputed FASTA
# MAGIC ref-base array are broadcast once. Runs on the classic extract cluster (fixed `num_workers`,
# MAGIC started minimal + titrated — see job.yml). **Not serverless** (pysam + binary VCF I/O).

# COMMAND ----------

dbutils.widgets.text("catalog", "genesis_workbench", "Catalog")
dbutils.widgets.text("schema", "genesis_schema", "Schema")
dbutils.widgets.text("vcf_dir", "", "Dir of gVCFs (each *.vcf.gz / *.g.vcf.gz = one sample)")
dbutils.widgets.text("vcf_paths", "", "Explicit gVCF paths (comma-sep; overrides vcf_dir)")
dbutils.widgets.text("fasta_path", "", "GRCh38 FASTA (.fna.bgz) for REF-block resolution")
dbutils.widgets.text("pgs_ids", "", "Restrict union to these PGS' variants (comma-sep; empty = all registered)")
dbutils.widgets.text("reextract", "false", "true = re-extract samples already in dosage (needed after add-PGS)")

catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")

# COMMAND ----------

# MAGIC %pip install pysam
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

import os
import glob
import numpy as np
import pandas as pd
import pysam
import pyspark.sql.functions as F
from pyspark.sql.types import StructType, StructField, StringType, DoubleType
from delta.tables import DeltaTable

catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")
vcf_dir = dbutils.widgets.get("vcf_dir")
vcf_paths_arg = dbutils.widgets.get("vcf_paths")
fasta_path = dbutils.widgets.get("fasta_path")
pgs_filter = [x.strip() for x in dbutils.widgets.get("pgs_ids").split(",") if x.strip()]
reextract = dbutils.widgets.get("reextract").strip().lower() == "true"

spark.sql(f"USE CATALOG {catalog}")
spark.sql(f"USE SCHEMA {schema}")

# ship the kernel + extractor to executors (import-time deps: numpy, pysam)
lib_dir = os.path.abspath(os.path.join(os.getcwd(), "..", "lib"))
for m in ("gvcf_dose.py", "prs_extract.py"):
    spark.sparkContext.addPyFile(os.path.join(lib_dir, m))
import sys; sys.path.append(lib_dir)
import gvcf_dose as kern
import prs_extract as ext

# COMMAND ----------

# MAGIC %md
# MAGIC ### 1. Union catalog from the registered weights (the positions to extract)

# COMMAND ----------

wq = spark.table("pgs_weights")
if pgs_filter:
    wq = wq.where(F.col("pgs_id").isin(pgs_filter))

# distinct (chrom,pos,effect,other) drives the extraction. variant_id = chrom:pos:effect:other,
# so split it back out. (For genome-wide PGS this collect is large — chunk by chrom to titrate.)
uv = (wq.select("variant_id").distinct()
        .select(F.split("variant_id", ":").alias("p"))
        .select(F.col("p")[0].alias("chrom"), F.col("p")[1].cast("long").alias("pos"),
                F.col("p")[2].alias("effect"), F.col("p")[3].alias("other")))
rows = [(r["chrom"], r["pos"], r["effect"], r["other"]) for r in uv.collect()]
ucat = ext.build_union_catalog(rows)
print(f"union catalog: {ucat.n_var} variants" + (f" (restricted to {pgs_filter})" if pgs_filter else ""))

# precompute FASTA ref base per catalog position ONCE on the driver, broadcast the array
fasta_ref = kern.build_catalog_fasta_ref(ucat, fasta_path)
UCAT_B = spark.sparkContext.broadcast(ucat)
FASTA_B = spark.sparkContext.broadcast(fasta_ref)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2. Sample manifest + sample-granularity incrementality
# MAGIC sample_id is read from each gVCF header (authoritative); samples already in `dosage` are skipped
# MAGIC unless `reextract=true`. (Add-sample = new files only; re-run = 0 work.)

# COMMAND ----------

if vcf_paths_arg.strip():
    paths = [p.strip() for p in vcf_paths_arg.split(",") if p.strip()]
else:
    paths = sorted(glob.glob(os.path.join(vcf_dir, "*.vcf.gz")) + glob.glob(os.path.join(vcf_dir, "*.g.vcf.gz")))

manifest = []
for p in paths:
    with pysam.VariantFile(p) as vf:
        sid = list(vf.header.samples)[0]
    manifest.append((sid, p))
print(f"{len(manifest)} gVCF(s) found")

if not reextract and spark.catalog.tableExists("dosage"):
    have = {r["sample_id"] for r in spark.table("dosage").select("sample_id").distinct().collect()}
    before = len(manifest)
    manifest = [(s, p) for (s, p) in manifest if s not in have]
    print(f"incremental: skipping {before - len(manifest)} already-extracted sample(s); {len(manifest)} to extract")

if not manifest:
    dbutils.notebook.exit("0 — nothing to extract (all samples present; use reextract=true to force)")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3. Distribute: one task per sample → (sample_id, variant_id, dose) → MERGE dosage

# COMMAND ----------

OUT_SCHEMA = StructType([
    StructField("sample_id", StringType()),
    StructField("variant_id", StringType()),
    StructField("dose", DoubleType()),
])

def _extract_partition(itr):
    import prs_extract as _e, gvcf_dose as _k  # resolved from addPyFile on the executor
    uc = UCAT_B.value
    fa = FASTA_B.value
    for pdf in itr:
        for _, r in pdf.iterrows():
            sid, dose_rows = _e.sample_dosage_rows(r["vcf_path"], uc, fa, kernel=_k)
            if dose_rows:
                out = pd.DataFrame(dose_rows, columns=["variant_id", "dose"])
                out.insert(0, "sample_id", sid)
                yield out

n_tasks = max(1, len(manifest))
mdf = spark.createDataFrame(manifest, ["sample_id", "vcf_path"]).repartition(n_tasks)
extracted = mdf.mapInPandas(_extract_partition, schema=OUT_SCHEMA)

# stage then MERGE (idempotent on (sample_id, variant_id))
extracted.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable("_dosage_stage")
stage = spark.table("_dosage_stage")
(DeltaTable.forName(spark, f"{catalog}.{schema}.dosage").alias("t")
 .merge(stage.alias("s"), "t.sample_id = s.sample_id AND t.variant_id = s.variant_id")
 .whenMatchedUpdateAll().whenNotMatchedInsertAll().execute())

n_rows = stage.count()
n_samp = stage.select("sample_id").distinct().count()
print(f"extracted {n_rows} dosage rows across {n_samp} sample(s) → {catalog}.{schema}.dosage")
spark.sql("DROP TABLE IF EXISTS _dosage_stage")
