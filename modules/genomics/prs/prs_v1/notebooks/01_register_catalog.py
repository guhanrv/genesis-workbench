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
# MAGIC - **panel_ref** is a **curation read** of `reference_distribution` (the reference implementation's FROZEN HGDP+1kGP
# MAGIC   per-superpop mean/sd) — no 1000G panel-scoring job needed. `panel_version` = curation `version`.
# MAGIC - **weight_sha** (sha256 of the scorefile) scopes a restatement to one PGS: re-registering a PGS
# MAGIC   replaces only its weights and refreshes only its registry/panel rows; the other 100 stay valid.
# MAGIC
# MAGIC Runs on the small classic register cluster (see job.yml). No serverless.

# COMMAND ----------

# MAGIC %pip install pyyaml==6.0.1

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
import pandas as pd
import pyspark.sql.functions as F
from pyspark.sql.types import (StructType, StructField, StringType, DoubleType, LongType,
                               ArrayType, TimestampType)
from delta.tables import DeltaTable

# import the pure parsers from the module lib (driver import for schemas; addPyFile ships it to the
# executors so the distributed scorefile parse below can `import prs_register` in each task).
import sys
lib_dir = os.path.abspath(os.path.join(os.getcwd(), "..", "lib"))
sys.path.append(lib_dir)
spark.sparkContext.addPyFile(os.path.join(lib_dir, "prs_register.py"))
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

# --- Weights parse: DISTRIBUTED, one Spark task per scorefile (mapInPandas). Parsing 100+ gzip
#     scorefiles (some genome-wide, millions of lines) on the DRIVER was serial — ~1 core of N, the
#     module's slowest task by far. Ship the paths to executors instead: each parses its file (gzip +
#     effect-orientation + palindromic drop, all in prs_register) and emits its weight rows, written to
#     pgs_weights in ONE distributed pass. Nothing accumulates on the driver, so this both parallelizes
#     the parse AND removes the driver-memory ceiling (no batching needed). ---
present = [(s["pgs_id"], os.path.join(scorefile_dir, s["scoring_file"])) for s in scores
           if os.path.exists(os.path.join(scorefile_dir, s["scoring_file"]))]
registered_pgs = [p for p, _ in present]
skipped = [(s["pgs_id"], s["scoring_file"]) for s in scores
           if not os.path.exists(os.path.join(scorefile_dir, s["scoring_file"]))]
if not registered_pgs:
    dbutils.notebook.exit("0 — nothing registered (no scorefiles found)")

_WGT_COLS = ["pgs_id", "variant_id", "effect_allele", "other_allele", "weight", "weight_sha"]

def _parse_files(itr):
    """One task per scorefile path: sha + parse + effect-oriented dedup → weight-row DataFrame."""
    import prs_register as _R           # shipped via addPyFile
    for pdf in itr:
        for _, r in pdf.iterrows():
            sha = _R.compute_weight_sha(r["path"])
            wr = _R.weights_rows(r["pgs_id"], _R.parse_scorefile(r["path"]), sha)
            if wr:
                yield pd.DataFrame(wr, columns=_WGT_COLS)

# one partition per file → up to (num cores) files parsed concurrently
paths_df = spark.createDataFrame(present, ["pgs_id", "path"]).repartition(len(present))
weights_df = paths_df.mapInPandas(_parse_files, schema=WGT_SCHEMA)

# pgs_weights is restate-safe: drop the registered PGS' old rows, then write the fresh set in one pass.
spark.sql(f"DELETE FROM pgs_weights WHERE pgs_id IN ({', '.join(repr(p) for p in registered_pgs)})")
weights_df.write.mode("append").saveAsTable("pgs_weights")

if skipped:
    print(f"WARNING: {len(skipped)} scorefile(s) missing in {scorefile_dir} — skipped: {skipped}")

# COMMAND ----------

# --- Registry + panel rows (small, one-ish per PGS) built on the driver from the curation + the
#     just-written weights. n_variants / weight_sha per PGS are read back from pgs_weights (distributed
#     groupBy, not a driver parse), so registry reflects exactly what was written (post dedup/drop). ---
nv = {r["pgs_id"]: (r["n_variants"], r["weight_sha"]) for r in
      spark.table("pgs_weights").where(F.col("pgs_id").isin(registered_pgs))
      .groupBy("pgs_id").agg(F.count("*").alias("n_variants"),
                             F.first("weight_sha").alias("weight_sha")).collect()}

# Fail loudly (don't silently drop): a PGS whose scorefile was found but parsed to 0 usable weight rows
# would otherwise just vanish from the registry (the `if pgs_id not in nv: continue` below). That hides
# a malformed/empty scorefile or an all-palindromic-drop. Surface it prominently.
_zero_var = sorted({p for p, _ in present} - set(nv))
if _zero_var:
    print(f"WARNING: {len(_zero_var)} scorefile(s) parsed to 0 usable variants and are being DROPPED "
          f"from the registry — inspect for malformed/empty files, wrong columns, or all-palindromic "
          f"variants: {_zero_var}")

registry_rows, panel_rows = [], []
path_by_pgs = dict(present)
for s in scores:
    pgs_id = s["pgs_id"]
    if pgs_id not in nv:
        continue
    n_variants, sha = nv[pgs_id]
    registry_rows.append(R.registry_row(s, weight_sha=sha, n_variants=n_variants, weight_path=path_by_pgs[pgs_id]))
    panel_rows.extend(R.panel_ref_rows(s, panel_version=panel_version, weight_sha=sha))
print(f"pgs_weights: wrote {sum(v[0] for v in nv.values())} weight rows across {len(registered_pgs)} PGS")

reg_df = spark.createDataFrame(registry_rows, REG_SCHEMA).withColumn("registered_at", F.current_timestamp())
panel_df = spark.createDataFrame(panel_rows, PANEL_SCHEMA)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Write — registry/panel via keyed MERGE (weights written distributed above)
# MAGIC Weights were replaced for the registered `pgs_id`s (restate-safe: only their rows were
# MAGIC dropped/rewritten). Registry (key `pgs_id`) and panel (key `pgs_id,superpop,panel_version`)
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
