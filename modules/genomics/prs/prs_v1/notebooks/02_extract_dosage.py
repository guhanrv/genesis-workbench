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
# MAGIC **Distribution (default, `shard_by_chrom=true`):** the union is built and broadcast **one chromosome
# MAGIC at a time**, and each chrom is extracted across all samples (one task per sample) before the next.
# MAGIC The driver therefore never holds more than a single chromosome's catalog (~union/22), so a
# MAGIC genome-wide union (tens of millions of variants) can't OOM the driver regardless of node size —
# MAGIC the whole-union `toPandas` was the ceiling. `shard_by_chrom=false` is the legacy whole-genome walk
# MAGIC (one task per sample, whole union broadcast at once) — only for small unions. Runs on the classic
# MAGIC extract cluster (**not serverless** — pysam + binary VCF I/O).

# COMMAND ----------

dbutils.widgets.text("catalog", "genesis_workbench", "Catalog")
dbutils.widgets.text("schema", "genesis_schema", "Schema")
dbutils.widgets.text("vcf_dir", "", "Dir of gVCFs (each *.vcf.gz / *.g.vcf.gz = one sample)")
dbutils.widgets.text("vcf_paths", "", "Explicit gVCF paths (comma-sep; overrides vcf_dir)")
dbutils.widgets.text("fasta_path", "", "GRCh38 FASTA (.fna.bgz) for REF-block resolution")
dbutils.widgets.text("fasta_ref_cache_dir", "", "Volume dir to cache the FASTA-ref array (skips the ~30s rebuild on repeat runs)")
dbutils.widgets.text("pgs_ids", "", "Restrict union to these PGS' variants (comma-sep; empty = all registered)")
dbutils.widgets.text("reextract", "false", "true = re-extract samples already in dosage (needed after add-PGS)")
dbutils.widgets.text("shard_by_chrom", "true", "Shard extraction per chrom — bounds driver memory to one chrom's catalog")

catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")

# COMMAND ----------

# MAGIC %pip install pysam
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

import os
import glob
import hashlib
from pathlib import Path
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
fasta_ref_cache_dir = dbutils.widgets.get("fasta_ref_cache_dir").strip()
pgs_filter = [x.strip() for x in dbutils.widgets.get("pgs_ids").split(",") if x.strip()]
reextract = dbutils.widgets.get("reextract").strip().lower() == "true"
shard_by_chrom = dbutils.widgets.get("shard_by_chrom").strip().lower() == "true"

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
# MAGIC ### 1. Sample manifest + sample-granularity incrementality
# MAGIC Resolve the gVCFs FIRST (cheap) so a no-op run exits before building any catalog. `sample_id` is read
# MAGIC from each gVCF header (authoritative); samples already in `dosage` are skipped unless `reextract=true`.

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
# MAGIC ### 2. Union catalog from the registered weights (the positions to extract)
# MAGIC `variant_id = chrom:pos:effect:other`, split back out. The catalog is built per chromosome (below), so
# MAGIC the driver holds only one chrom's arrays at a time — a genome-wide union never lands whole on the driver.

# COMMAND ----------

wq = spark.table("pgs_weights")
if pgs_filter:
    wq = wq.where(F.col("pgs_id").isin(pgs_filter))

# variant_id split, reused per chrom (chrom = the first ':' field)
uv_all = (wq.select("variant_id").distinct()
            .select(F.split("variant_id", ":").alias("p"))
            .select(F.col("p")[0].alias("chrom"), F.col("p")[1].cast("long").alias("pos"),
                    F.col("p")[2].alias("effect"), F.col("p")[3].alias("other")))

# fasta-ref cache is content-addressed on the registered (pgs, weight_sha) set + FASTA — identical across
# runs, so a repeat/incremental run reads the cached .npz instead of rebuilding. Key computed ONCE here;
# the per-chrom filename appends the chrom (see _chrom_fasta_ref). Volumes are FUSE-mounted and don't
# support the random seek() np.savez/load need, so cache on LOCAL disk and sync with dbutils.fs.cp.
_cache_key = None
if fasta_ref_cache_dir:
    sel = [f"{r['pgs_id']}:{r['weight_sha']}" for r in
           wq.select("pgs_id", "weight_sha").distinct().orderBy("pgs_id", "weight_sha").collect()]
    _cache_key = hashlib.sha256(("|".join(sel) + "|" + os.path.basename(fasta_path)).encode()).hexdigest()[:16]

def _chrom_fasta_ref(catc, tag):
    """FASTA ref-base array for one chrom's catalog, with a per-(tag, PGS-set, FASTA) content-addressed
    cache. tag = chrom (bounded path) or 'all' (legacy). Returns the |S1 ref array."""
    cache_path = None
    if _cache_key:
        fname = f"fastaref_{tag}_{catc.n_var}_{_cache_key}.npz"
        cache_path = Path("/tmp") / fname
        _vol = fasta_ref_cache_dir.rstrip("/") + "/" + fname
        try:
            dbutils.fs.cp(_vol, "file:" + str(cache_path))          # warm: pull cached array
        except Exception:
            pass                                                     # cold: build below
    fr = kern.build_catalog_fasta_ref(catc, fasta_path, cache_path=cache_path)
    if _cache_key and cache_path is not None and cache_path.exists():
        try:
            dbutils.fs.mkdirs(fasta_ref_cache_dir)
            dbutils.fs.cp("file:" + str(cache_path), fasta_ref_cache_dir.rstrip("/") + "/" + cache_path.name)
        except Exception as e:
            print("warn: fasta-ref cache push failed (non-fatal):", e)
    return fr

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3. Extract → `_dosage_stage` → MERGE `dosage`
# MAGIC Bounded path: loop chroms; per chrom build+broadcast that chrom's catalog, extract all samples (one
# MAGIC task each) into `_dosage_stage`, then unpersist the broadcast before the next chrom. One final MERGE
# MAGIC (idempotent on `(sample_id, variant_id)`) folds the stage into `dosage`.

# COMMAND ----------

OUT_SCHEMA = StructType([
    StructField("sample_id", StringType()),
    StructField("variant_id", StringType()),
    StructField("dose", DoubleType()),
])

# per-iteration broadcast (reassigned each chrom); the mapInPandas closure reads it at action time
_CHROM_B = None

def _extract_one_chrom(itr):
    """One task per sample: walk this chrom's gVCF region against the broadcast chrom catalog."""
    import prs_extract as _e, gvcf_dose as _k
    ch, pos, eff, oth, vid, fa = _CHROM_B.value
    sub = _e.UnionCatalog(ch, pos, eff, oth, vid)
    for pdf in itr:
        for _, r in pdf.iterrows():
            sid, dose_rows = _e.sample_dosage_rows(r["vcf_path"], sub, fa, kernel=_k)
            if dose_rows:
                out = pd.DataFrame(dose_rows, columns=["variant_id", "dose"]); out.insert(0, "sample_id", sid)
                yield out

def _extract_whole(itr):
    """Legacy (shard_by_chrom=false): one task per sample, whole-genome walk against the full union."""
    import prs_extract as _e, gvcf_dose as _k
    uc = _CHROM_B.value[0]; fa = _CHROM_B.value[1]
    for pdf in itr:
        for _, r in pdf.iterrows():
            sid, dose_rows = _e.sample_dosage_rows(r["vcf_path"], uc, fa, kernel=_k)
            if dose_rows:
                out = pd.DataFrame(dose_rows, columns=["variant_id", "dose"]); out.insert(0, "sample_id", sid)
                yield out

n_samp = len(manifest)
mdf_samples = spark.createDataFrame(manifest, ["sample_id", "vcf_path"])

if shard_by_chrom:
    chroms = sorted(r["chrom"] for r in uv_all.select("chrom").distinct().collect())
    print(f"bounded per-chrom extraction: {len(chroms)} chroms × {n_samp} sample(s)")
    total_var = 0
    for i, ch in enumerate(chroms):
        cpd = uv_all.where(F.col("chrom") == ch).toPandas()          # one chrom only — bounded
        catc = ext.build_union_catalog(zip(cpd["chrom"], cpd["pos"], cpd["effect"], cpd["other"]))
        frc = _chrom_fasta_ref(catc, ch)
        total_var += catc.n_var
        _CHROM_B = spark.sparkContext.broadcast(
            (catc.chrom, catc.pos, catc.effect, catc.other, catc.variant_id, frc))
        extracted = mdf_samples.repartition(n_samp).mapInPandas(_extract_one_chrom, schema=OUT_SCHEMA)
        extracted.write.mode("overwrite" if i == 0 else "append").option(
            "overwriteSchema", "true").saveAsTable("_dosage_stage")
        _CHROM_B.unpersist()
        print(f"  chr{ch}: {catc.n_var:,} variants extracted for {n_samp} sample(s)")
    print(f"union total: {total_var:,} variants across {len(chroms)} chroms")
else:
    UNION_WARN = 20_000_000
    upd = uv_all.toPandas()                                          # whole union on driver (legacy)
    if len(upd) > UNION_WARN:
        print(f"WARNING: shard_by_chrom=false builds the whole {len(upd):,}-variant union on the driver "
              f"(> {UNION_WARN:,}); use shard_by_chrom=true for genome-wide unions.")
    ucat = ext.build_union_catalog(zip(upd["chrom"], upd["pos"], upd["effect"], upd["other"]))
    fr = _chrom_fasta_ref(ucat, "all")
    _CHROM_B = spark.sparkContext.broadcast((ucat, fr))
    print(f"whole-genome walk: {ucat.n_var:,} variants × {n_samp} sample(s)")
    extracted = mdf_samples.repartition(n_samp).mapInPandas(_extract_whole, schema=OUT_SCHEMA)
    extracted.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable("_dosage_stage")
    _CHROM_B.unpersist()

# COMMAND ----------

# stage → MERGE dosage (idempotent on (sample_id, variant_id))
stage = spark.table("_dosage_stage")
(DeltaTable.forName(spark, f"{catalog}.{schema}.dosage").alias("t")
 .merge(stage.alias("s"), "t.sample_id = s.sample_id AND t.variant_id = s.variant_id")
 .whenMatchedUpdateAll().whenNotMatchedInsertAll().execute())

n_rows = spark.table("dosage").count()
n_s = spark.table("dosage").select("sample_id").distinct().count()
print(f"dosage: {n_rows:,} rows across {n_s} sample(s)")
