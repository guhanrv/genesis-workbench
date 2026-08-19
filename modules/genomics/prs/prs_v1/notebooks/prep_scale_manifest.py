# Databricks notebook source
# MAGIC %md
# MAGIC # Prep — synthetic scale sample manifest (no physical clones / no bcftools)
# MAGIC
# MAGIC Writes a Delta manifest that maps many **logical** `sample_id`s to one physical
# MAGIC gVCF for cardinality / fan-out benchmarks. Production remains 1:1 elsewhere;
# MAGIC this notebook always emits `synthetic=true` and refuses unsafe configs.
# MAGIC
# MAGIC Columns: `sample_id`, `source_sample_id`, `vcf_path`, `source_fingerprint`,
# MAGIC `synthetic`, `scale_run_id`, `batch_id`. IDs come from `spark.range()` (not a
# MAGIC Python N-loop). Source fingerprint is a single streaming sha256 of the file.

# COMMAND ----------

dbutils.widgets.text(
    "source_vcf",
    "/Volumes/dev_exploration_sandbox/genesis_workbench/prs_data/incoming/amy.vcf.gz",
    "Source gVCF (.vcf.gz)",
)
dbutils.widgets.text(
    "manifest_path",
    "/Volumes/dev_exploration_sandbox/genesis_workbench/prs_data/scale/manifest_synth",
    "Delta path for sample manifest",
)
dbutils.widgets.text("n_samples", "10", "Number of logical sample_ids")
dbutils.widgets.text("id_prefix", "synth_amy", "Logical ID prefix → {prefix}_01 … (reserved)")
dbutils.widgets.text("scale_run_id", "scale_dev", "Scale run id stamped on every row")
dbutils.widgets.text("batch_size", "25", "Logical samples per batch_id (override for thinned-loci runs)")
dbutils.widgets.text("overwrite", "true", "Overwrite existing Delta path")

# COMMAND ----------

import os
import sys

import pyspark.sql.functions as F

lib_dir = os.path.abspath(os.path.join(os.getcwd(), "..", "lib"))
if lib_dir not in sys.path:
    sys.path.append(lib_dir)
from scale_manifest import (
    MANIFEST_COLUMNS,
    batch_id_for_index,
    build_source_fingerprint_metadata,
    fingerprint_column_value,
    logical_sample_id,
    read_vcf_header_sample_id,
    streaming_file_sha256,
    validate_synthetic_manifest_config,
)

source_vcf = dbutils.widgets.get("source_vcf").strip()
manifest_path = dbutils.widgets.get("manifest_path").strip().rstrip("/")
n_samples = int(dbutils.widgets.get("n_samples"))
id_prefix = dbutils.widgets.get("id_prefix").strip()
scale_run_id = dbutils.widgets.get("scale_run_id").strip()
batch_size = int(dbutils.widgets.get("batch_size"))
overwrite = dbutils.widgets.get("overwrite").strip().lower() == "true"

cfg = validate_synthetic_manifest_config(
    n_samples=n_samples,
    id_prefix=id_prefix,
    scale_run_id=scale_run_id,
    batch_size=batch_size,
    synthetic=True,
)
n_samples = cfg["n_samples"]
id_prefix = cfg["id_prefix"]
scale_run_id = cfg["scale_run_id"]
batch_size = cfg["batch_size"]
width = cfg["id_width"]

assert source_vcf.endswith(".vcf.gz") or source_vcf.endswith(".g.vcf.gz") or source_vcf.endswith(".vcf"), (
    f"source_vcf must be a VCF path, got {source_vcf!r}"
)
assert os.path.exists(source_vcf), f"missing source: {source_vcf}"
assert manifest_path, "manifest_path must be non-empty"

print(
    f"config: n_samples={n_samples} id_prefix={id_prefix!r} width={width} "
    f"batch_size={batch_size} scale_run_id={scale_run_id!r} overwrite={overwrite}"
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 1. Validate single-sample header + streaming sha256 (once)

# COMMAND ----------

source_sample_id = read_vcf_header_sample_id(source_vcf)
size_bytes = os.path.getsize(source_vcf)
print(f"source sample_id={source_sample_id!r}  size={size_bytes:,} bytes  path={source_vcf}")

print("computing streaming sha256 (single pass)…")
sha256_hex = streaming_file_sha256(source_vcf)
fp_meta = build_source_fingerprint_metadata(
    vcf_path=source_vcf,
    sha256_hex=sha256_hex,
    size_bytes=size_bytes,
    source_sample_id=source_sample_id,
)
source_fingerprint = fingerprint_column_value(fp_meta)
print(f"source_fingerprint={source_fingerprint}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2. Materialize logical IDs with ``spark.range`` → Delta

# COMMAND ----------

# 1-based indices via spark.range(1, n+1) — no Python loop of N ids on the driver.
idx = spark.range(1, n_samples + 1).withColumnRenamed("id", "idx")

# Deterministic pad width matches lib.logical_sample_id / prep_scale_clones.
pad = width
manifest = (
    idx.withColumn(
        "sample_id",
        F.format_string(f"%s_%0{pad}d", F.lit(id_prefix), F.col("idx")),
    )
    .withColumn("source_sample_id", F.lit(source_sample_id))
    .withColumn("vcf_path", F.lit(source_vcf))
    .withColumn("source_fingerprint", F.lit(source_fingerprint))
    .withColumn("synthetic", F.lit(True))
    .withColumn("scale_run_id", F.lit(scale_run_id))
    .withColumn(
        "batch_id",
        F.format_string(
            "batch_%05d",
            ((F.col("idx") - F.lit(1)) / F.lit(batch_size)).cast("int"),
        ),
    )
    .select(*MANIFEST_COLUMNS)
)

# Spot-check first/last against pure helpers (driver holds two rows only).
first_sid = logical_sample_id(id_prefix, 1, n_samples)
last_sid = logical_sample_id(id_prefix, n_samples, n_samples)
first_batch = batch_id_for_index(1, batch_size)
last_batch = batch_id_for_index(n_samples, batch_size)
_check = (
    idx.withColumn("sample_id", F.format_string(f"%s_%0{pad}d", F.lit(id_prefix), F.col("idx")))
    .withColumn(
        "batch_id",
        F.format_string("batch_%05d", ((F.col("idx") - F.lit(1)) / F.lit(batch_size)).cast("int")),
    )
    .where(F.col("idx").isin(1, n_samples))
    .collect()
)
by_idx = {int(r["idx"]): r for r in _check}
assert by_idx[1]["sample_id"] == first_sid and by_idx[1]["batch_id"] == first_batch
assert by_idx[n_samples]["sample_id"] == last_sid and by_idx[n_samples]["batch_id"] == last_batch

writer = manifest.write.format("delta").mode("overwrite" if overwrite else "errorifexists")
if overwrite:
    writer = writer.option("overwriteSchema", "true")
writer.save(manifest_path)

n_written = spark.read.format("delta").load(manifest_path).count()
assert n_written == n_samples, f"expected {n_samples} rows, got {n_written}"
print(f"wrote {n_written} synthetic manifest rows → {manifest_path}")
print(f"  sample_id range: {first_sid} … {last_sid}")
print(f"  batch_id range:  {first_batch} … {last_batch}")
print(f"  source_sample_id={source_sample_id!r} fingerprint={source_fingerprint}")

dbutils.notebook.exit(manifest_path)
