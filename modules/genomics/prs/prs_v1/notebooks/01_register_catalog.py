# Databricks notebook source
# MAGIC %md
# MAGIC # PRS register — curation → pgs_registry / pgs_weights / pgs_panel_ref
# MAGIC
# MAGIC Populates the three **catalog-side** stores from `prs.yaml` (the single source of truth) + the
# MAGIC harmonized PGS-catalog scoring files. This is the **"Add a PGS"** entry point: register → append
# MAGIC weights → append the frozen panel reference. Pure parsing lives in `lib/prs_register.py`
# MAGIC (unit-tested off-cluster); this notebook is only the Spark/Delta wrapper.
# MAGIC
# MAGIC - **weights** are effect-oriented (`variant_id = chr:pos:effect:other`, plain `weight`) — matches
# MAGIC   the extractor + scorer (`raw = Σ dose·weight`).
# MAGIC - **panel_ref** is a **curation read** of `reference_distribution` (function_prs's FROZEN HGDP+1kGP
# MAGIC   per-superpop mean/sd) — no 1000G panel-scoring job needed. `panel_version` = curation `version`.
# MAGIC - **weight_sha** (sha256 of the scorefile) scopes a restatement to one PGS: re-registering a PGS
# MAGIC   replaces only its weights and refreshes only its registry/panel rows; the other 100 stay valid.
# MAGIC
# MAGIC Runs on the small classic register cluster (see job.yml). No serverless.

# COMMAND ----------

# MAGIC %pip install pyyaml

# COMMAND ----------

dbutils.widgets.text("catalog", "genesis_workbench", "Catalog")
dbutils.widgets.text("schema", "genesis_schema", "Schema")
dbutils.widgets.text("catalog_config_path", "", "Path to prs.yaml (curation source of truth)")
dbutils.widgets.text("scorefile_dir", "", "Dir holding <scoring_file> harmonized scorefiles")
dbutils.widgets.text("pgs_ids", "", "PGS ids to register (comma-sep; empty = all in prs.yaml)")

catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")

# COMMAND ----------

import os
import yaml
import pyspark.sql.functions as F
from pyspark.sql.types import (StructType, StructField, StringType, DoubleType, LongType,
                               ArrayType, TimestampType)
from delta.tables import DeltaTable

# import the pure parsers from the module lib (staged next to the notebooks in the workspace)
import sys
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "..", "lib")))
import prs_register as R

spark.sql(f"USE CATALOG {catalog}")
spark.sql(f"USE SCHEMA {schema}")

config_path = dbutils.widgets.get("catalog_config_path")
scorefile_dir = dbutils.widgets.get("scorefile_dir")
want = [x.strip() for x in dbutils.widgets.get("pgs_ids").split(",") if x.strip()]

# COMMAND ----------

cur = yaml.safe_load(open(config_path))
panel_version = cur["version"]
scores = cur["scores"]
if want:
    scores = [s for s in scores if s["pgs_id"] in want]
print(f"curation version={panel_version}; registering {len(scores)} score(s)")

# COMMAND ----------

# --- Explicit schemas (dicts carry Nones → need typed createDataFrame) ---
REG_SCHEMA = StructType([
    StructField("pgs_id", StringType()), StructField("score_id", StringType()),
    StructField("disease", StringType()), StructField("direction", StringType()),
    StructField("body_system", StringType()), StructField("hr_per_sd", DoubleType()),
    StructField("clinical_model", StringType()), StructField("training_ancestries", StringType()),
    StructField("weight_sha", StringType()), StructField("n_variants", LongType()),
    StructField("weight_path", StringType()),
])
WGT_SCHEMA = StructType([
    StructField("pgs_id", StringType()), StructField("variant_id", StringType()),
    StructField("effect_allele", StringType()), StructField("other_allele", StringType()),
    StructField("weight", DoubleType()), StructField("weight_sha", StringType()),
])
PANEL_SCHEMA = StructType([
    StructField("pgs_id", StringType()), StructField("superpop", StringType()),
    StructField("mean", DoubleType()), StructField("sd", DoubleType()),
    StructField("quantiles", ArrayType(DoubleType())), StructField("n_panel", LongType()),
    StructField("panel_version", StringType()), StructField("weight_sha", StringType()),
])

# COMMAND ----------

# --- Parse each scorefile → the three stores. Weights (the only large store: a single genome-wide
#     PGS can be millions of variants, and 100+ of them overflow the driver) are STREAMED to Delta in
#     bounded batches, so the driver never holds more than ~one big PGS of rows at once. Registry/panel
#     are one-row-ish per PGS, so they stay in driver lists and MERGE once at the end. ---
FLUSH_ROWS = 2_000_000        # append pgs_weights once the buffer crosses this (bounds driver memory)

registry_rows, panel_rows, skipped = [], [], []
registered_pgs = [s["pgs_id"] for s in scores
                  if os.path.exists(os.path.join(scorefile_dir, s["scoring_file"]))]
if not registered_pgs:
    dbutils.notebook.exit("0 — nothing registered (no scorefiles found)")

# pgs_weights is restate-safe: drop the registered PGS' old rows up front, then stream fresh rows in.
spark.sql(f"DELETE FROM pgs_weights WHERE pgs_id IN ({', '.join(repr(p) for p in registered_pgs)})")

_buf, _n_flushed = [], 0
def _flush_weights():
    global _buf, _n_flushed
    if not _buf:
        return
    spark.createDataFrame(_buf, WGT_SCHEMA).write.mode("append").saveAsTable("pgs_weights")
    _n_flushed += len(_buf)
    _buf = []

for s in scores:
    pgs_id = s["pgs_id"]
    path = os.path.join(scorefile_dir, s["scoring_file"])
    if not os.path.exists(path):
        skipped.append((pgs_id, s["scoring_file"]))
        continue
    sha = R.compute_weight_sha(path)
    wr = R.weights_rows(pgs_id, R.parse_scorefile(path), sha)
    pr = R.panel_ref_rows(s, panel_version=panel_version, weight_sha=sha)
    registry_rows.append(R.registry_row(s, weight_sha=sha, n_variants=len(wr), weight_path=path))
    panel_rows.extend(pr)
    _buf.extend(wr)
    print(f"  {pgs_id}: {len(wr)} weights, {len(pr)} panel rows, sha={sha[:12]}…")
    if len(_buf) >= FLUSH_ROWS:
        _flush_weights()
_flush_weights()   # final partial batch

if skipped:
    print(f"WARNING: {len(skipped)} scorefile(s) missing in {scorefile_dir} — skipped: {skipped}")
print(f"pgs_weights: streamed {_n_flushed} weight rows across {len(registered_pgs)} PGS")

# COMMAND ----------

reg_df = spark.createDataFrame(registry_rows, REG_SCHEMA).withColumn("registered_at", F.current_timestamp())
panel_df = spark.createDataFrame(panel_rows, PANEL_SCHEMA)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Write — registry/panel via keyed MERGE (weights already streamed above)
# MAGIC Weights were replaced per-PGS as they parsed (restate-safe: only the registered `pgs_id`s' rows
# MAGIC were dropped/rewritten). Registry (key `pgs_id`) and panel (key `pgs_id,superpop,panel_version`)
# MAGIC MERGE-upsert. All idempotent: re-running the same curation is a no-op change.

# COMMAND ----------

# pgs_registry: one row per pgs_id.
(DeltaTable.forName(spark, f"{catalog}.{schema}.pgs_registry").alias("t")
 .merge(reg_df.alias("s"), "t.pgs_id = s.pgs_id")
 .whenMatchedUpdateAll().whenNotMatchedInsertAll().execute())

# pgs_panel_ref: keyed on (pgs_id, superpop, panel_version).
(DeltaTable.forName(spark, f"{catalog}.{schema}.pgs_panel_ref").alias("t")
 .merge(panel_df.alias("s"),
        "t.pgs_id = s.pgs_id AND t.superpop = s.superpop AND t.panel_version = s.panel_version")
 .whenMatchedUpdateAll().whenNotMatchedInsertAll().execute())

# COMMAND ----------

print(f"registered {len(registered_pgs)} PGS @ panel_version={panel_version}")
for t in ["pgs_registry", "pgs_weights", "pgs_panel_ref"]:
    print(f"  {t}: {spark.table(t).count()} rows total")
