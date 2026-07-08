# Databricks notebook source
# MAGIC %md
# MAGIC # PRS scorer — SQL join-aggregate over the reconcile plan (classic; NOT serverless yet)
# MAGIC
# MAGIC Reads `_prs_reconcile_plan` (written by `02_reconcile`) and scores **only those (sample, pgs) cells** —
# MAGIC so a run can never touch the full grid. Default mechanism is a plain **SQL join-aggregate** (predictable,
# MAGIC observable cost); `applyInPandas` is a benchmark-gated alternative, not used here.
# MAGIC
# MAGIC ```
# MAGIC raw = Σ_variant  dose · weight    (dose is effect-oriented per (chrom,pos,effect,other))
# MAGIC z_msp = (raw − panel_mean[pgs,msp]) / panel_sd[pgs,msp]      (reference-panel normalized; no cohort re-rank)
# MAGIC ```
# MAGIC Runs on a classic cluster started minimal + titrated (see job.yml). MERGE-upserts the flat `prs_scores`.

# COMMAND ----------

dbutils.widgets.text("catalog", "genesis_workbench", "Catalog")
dbutils.widgets.text("schema", "genesis_schema", "Schema")
catalog = dbutils.widgets.get("catalog"); schema = dbutils.widgets.get("schema")

# COMMAND ----------

import math
import pyspark.sql.functions as F
from delta.tables import DeltaTable

spark.sql(f"USE CATALOG {catalog}"); spark.sql(f"USE SCHEMA {schema}")
spark.conf.set("spark.sql.shuffle.partitions", "auto")  # bounded by classic cluster size

plan = spark.table("_prs_reconcile_plan")                 # sample_id, pgs_id, weight_sha, panel_version
if plan.limit(1).count() == 0:
    dbutils.notebook.exit("0")  # reconcile found nothing; scorer is a no-op

planned_samples = plan.select("sample_id").distinct()
planned_pgs = plan.select("pgs_id", "weight_sha").distinct()

# COMMAND ----------

# MAGIC %md
# MAGIC ### 1. Raw score — join only the planned samples' dosage to the planned PGS' weights
# MAGIC Restricting both sides to the plan keeps add-sample (one row) / add-PGS (one column) tiny; a semi-join
# MAGIC back to the plan drops any (sample,pgs) product not actually requested.

# COMMAND ----------

# Incremental fast path: when the plan targets few samples (add-sample / small batch), prune
# `dosage` with an explicit sample_id predicate so Liquid Clustering (CLUSTER BY sample_id) SKIPS
# files instead of scanning the whole store. Stage-0 measured that a broadcast join here re-scans
# all dosage rows even to score one new sample; the predicate turns that full scan into a file-skip.
# Full backfills (many planned samples) skip the predicate and read all files (correct + fastest).
SAMPLE_PREDICATE_MAX = 200
_planned_ids = [r["sample_id"] for r in planned_samples.limit(SAMPLE_PREDICATE_MAX + 1).collect()]
if 0 < len(_planned_ids) <= SAMPLE_PREDICATE_MAX:
    dose = spark.table("dosage").where(F.col("sample_id").isin(_planned_ids))            # file-skip to planned samples
else:
    dose = spark.table("dosage").join(F.broadcast(planned_samples), "sample_id")         # backfill: read all
wts = spark.table("pgs_weights").join(F.broadcast(planned_pgs), ["pgs_id", "weight_sha"])  # only planned pgs@sha

raw = (
    dose.join(wts, "variant_id")
    .groupBy("sample_id", "pgs_id", "weight_sha")
    .agg(F.sum(F.col("dose") * F.col("weight")).alias("raw_score"),
         F.count(F.lit(1)).alias("n_variants_matched"))
    # keep only the exact cells the plan asked for (sparse plans compute nothing extra)
    .join(plan, ["sample_id", "pgs_id", "weight_sha"])
)

# coverage vs the PGS' total variant count (from registry)
nvar = spark.table("pgs_registry").select("pgs_id", F.col("n_variants").alias("pgs_nvar"))
raw = (raw.join(nvar, "pgs_id", "left")
       .withColumn("coverage_pct", F.when(F.col("pgs_nvar") > 0, 100.0 * F.col("n_variants_matched") / F.col("pgs_nvar")))
       .withColumn("small_score", F.col("n_variants_matched") < F.lit(1000)))

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2. Reference-panel normalization — z_msp/percentile_msp against the FROZEN panel (per sample's MSP)
# MAGIC `sample_ancestry` (from the PCA module: sample_id → most_similar_pop) picks which panel superpop row to
# MAGIC standardize against. Frozen panel ⇒ a new sample's z needs no cohort re-rank. (Admixed PRSmix+ z is a
# MAGIC documented follow-up — needs the RF-posterior weighting from `_admixed_score`.)

# COMMAND ----------

anc = spark.table("sample_ancestry").select("sample_id", F.col("most_similar_pop").alias("msp"))
ref = spark.table("pgs_panel_ref").select(
    "pgs_id", F.col("superpop").alias("msp"), "panel_version",
    F.col("mean").alias("ref_mean"), F.col("sd").alias("ref_sd"))

scored = (
    raw  # already carries panel_version (joined from the plan above)
    .join(anc, "sample_id", "left")
    .join(ref, ["pgs_id", "msp", "panel_version"], "left")
    .withColumn("z_msp", F.when(F.col("ref_sd") > 0, (F.col("raw_score") - F.col("ref_mean")) / F.col("ref_sd")))
)

# percentile from z via the normal CDF (parametric); empirical-from-panel-quantiles is a refinement.
@F.udf("double")
def _norm_cdf_pct(z):
    return None if z is None else 100.0 * 0.5 * (1.0 + math.erf(float(z) / math.sqrt(2.0)))

scored = scored.withColumn("percentile_msp", _norm_cdf_pct(F.col("z_msp")))

# registry metadata + result columns (admixed/clinical/concordance left null here — later stages fill them)
reg = spark.table("pgs_registry").select("pgs_id", "score_id", "disease", "direction", "hr_per_sd", "clinical_model")
out = (
    scored.join(reg, "pgs_id", "left")
    .withColumn("used_ancestry", F.col("msp"))
    .withColumn("z_admixed", F.lit(None).cast("double"))
    .withColumn("percentile_admixed", F.lit(None).cast("double"))
    .withColumn("integrated_z_source", F.lit("msp"))
    .withColumn("integrated_risk_10yr", F.lit(None).cast("double"))
    .withColumn("clinical_risk_10yr", F.lit(None).cast("double"))
    .withColumn("risk_category", F.lit(None).cast("string"))
    .withColumn("concordance_verdict", F.lit(None).cast("string"))
    .withColumn("computed_at", F.current_timestamp())
    .select("sample_id", "pgs_id", "weight_sha", "panel_version", "raw_score", "n_variants_matched",
            "coverage_pct", "small_score", F.col("msp").alias("most_similar_pop"), "used_ancestry",
            "z_msp", "percentile_msp", "z_admixed", "percentile_admixed", "integrated_z_source",
            "integrated_risk_10yr", "clinical_risk_10yr", "risk_category", "concordance_verdict", "computed_at")
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3. MERGE-upsert into the flat `prs_scores` cell-store (MERGE-safe: no Glow structs)

# COMMAND ----------

out.createOrReplaceTempView("_scored")
tgt = DeltaTable.forName(spark, f"{catalog}.{schema}.prs_scores")
(tgt.alias("t").merge(spark.table("_scored").alias("s"), "t.sample_id = s.sample_id AND t.pgs_id = s.pgs_id")
   .whenMatchedUpdateAll()
   .whenNotMatchedInsertAll()
   .execute())

n = spark.table("_scored").count()
print(f"scored + upserted {n} (sample,pgs) cells into {catalog}.{schema}.prs_scores")
