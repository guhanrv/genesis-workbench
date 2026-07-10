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
dbutils.widgets.text("missing_mode", "drop", "drop = missing→0 (tested default) | mean_impute = missing→2·AF from pgs_panel_afreq")
catalog = dbutils.widgets.get("catalog"); schema = dbutils.widgets.get("schema")
missing_mode = dbutils.widgets.get("missing_mode").strip().lower()

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
# MAGIC ### 1. Raw score — join the planned samples' dosage to the planned PGS' weights
# MAGIC `dose` is pruned to the plan's samples and `wts` to the plan's PGS, so the aggregate spans that
# MAGIC grid; the FINAL left-join onto the plan (below) keeps exactly the planned cells and fills any
# MAGIC zero-coverage ones (raw=0) so each planned cell is written once. For single-mode plans (add-sample
# MAGIC / add-PGS / restate-one) the grid IS the plan, so nothing extra is aggregated.

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

# Missing-variant handling (pgsc_calc/plink2 semantics):
#   drop (default): a PGS variant not covered by the sample contributes 0 (inner join). Correct for
#     high-coverage WGS gVCF where missing is rare; this is the parity-validated path.
#   mean_impute: missing → 2·AF(effect) from the frozen panel (pgs_panel_afreq), matching function_prs
#     mean_impute / plink2 --read-freq. Densifies the join (every planned cell × its PGS' variants), so
#     it's costlier — worth it for hard-called / low-coverage cohorts where missing is common.
_use_impute = missing_mode == "mean_impute" and spark.catalog.tableExists("pgs_panel_afreq")
if missing_mode == "mean_impute" and not _use_impute:
    print("WARN: missing_mode=mean_impute but pgs_panel_afreq absent → falling back to drop")

if _use_impute:
    afreq = spark.table("pgs_panel_afreq").select("variant_id", "af_effect")
    dose_sv = dose.select("sample_id", "variant_id", F.col("dose").alias("_d"))
    agg = (
        plan.join(wts, ["pgs_id", "weight_sha"])                       # densify: cell × PGS' variants
        .join(afreq, "variant_id", "left")
        .join(dose_sv, ["sample_id", "variant_id"], "left")
        .withColumn("dose_f", F.coalesce(F.col("_d"), 2.0 * F.col("af_effect"), F.lit(0.0)))  # missing→2·AF (→0 if no panel AF)
        .groupBy("sample_id", "pgs_id", "weight_sha")
        .agg(F.sum(F.col("dose_f") * F.col("weight")).alias("raw_score"),
             F.sum(F.col("_d").isNotNull().cast("int")).alias("n_variants_matched"))  # matched = real coverage
    )
else:
    agg = (
        dose.join(wts, "variant_id")
        .groupBy("sample_id", "pgs_id", "weight_sha")
        .agg(F.sum(F.col("dose") * F.col("weight")).alias("raw_score"),
             F.count(F.lit(1)).alias("n_variants_matched"))
    )

# Left-join the aggregate onto the plan so EVERY planned cell is written exactly once — a cell whose
# sample shares no variant with the PGS (zero coverage) still lands as raw=0, matched=0 instead of
# vanishing (an inner join would drop it, and reconcile would then re-plan it every run forever).
raw = (
    plan.join(agg, ["sample_id", "pgs_id", "weight_sha"], "left")
    .withColumn("raw_score", F.coalesce(F.col("raw_score"), F.lit(0.0)))
    .withColumn("n_variants_matched", F.coalesce(F.col("n_variants_matched"), F.lit(0)))
)

# coverage vs the PGS' total variant count (from registry)
nvar = spark.table("pgs_registry").select("pgs_id", F.col("n_variants").alias("pgs_nvar"))
raw = (raw.join(nvar, "pgs_id", "left")
       .withColumn("coverage_pct", F.when(F.col("pgs_nvar") > 0, 100.0 * F.col("n_variants_matched") / F.col("pgs_nvar")))
       .withColumn("small_score", F.col("n_variants_matched") < F.lit(1000)))

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2. Reference-panel normalization — z_msp (single MSP) + z_admixed (RF-posterior-weighted)
# MAGIC `sample_ancestry` (from the ancestry module) gives `most_similar_pop` → the panel superpop row to
# MAGIC standardize against (`z_msp`), plus `rf_probs` → the continuous-ancestry `z_admixed`. Frozen panel ⇒
# MAGIC a new sample's z needs no cohort re-rank.

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

# ── z_admixed — continuous-ancestry (PRSmix-style) normalization ─────────────────────────────
# Standardize raw against EACH panel superpop's distribution and weight by the RF ancestry
# posterior (rf_probs, persisted by 05_build_sample_ancestry):  z_admixed = Σ_pop P_RF(pop)·z_pop.
# Uses the SAME PGS + the existing pgs_panel_ref (per-superpop mean/sd), so it needs no bespoke
# per-ancestry PGS catalog. For a non-admixed sample (RF ~1.0 on its MSP) it collapses to z_msp;
# for an admixed sample it blends the superpop references. Degenerate rows (sd=0 / prob=0) drop and
# the surviving weights renormalize (mass redistribution) — matching function_prs._admixed_score.
# (PRSmix+ with DISTINCT per-ancestry PGS variants + MyOme β/caPRS weighting is a curation-heavy
#  extension deliberately left out of the core scorer.)
probs = (spark.table("sample_ancestry")
         .select("sample_id", F.from_json(F.col("rf_probs"), "map<string,double>").alias("_p"))
         .select("sample_id", F.explode("_p").alias("superpop", "prob")))
ref_all = spark.table("pgs_panel_ref").select(
    "pgs_id", "superpop", "panel_version", F.col("mean").alias("amean"), F.col("sd").alias("asd"))
adm = (
    raw.select("sample_id", "pgs_id", "panel_version", "raw_score")
    .join(probs, "sample_id")
    .join(ref_all, ["pgs_id", "superpop", "panel_version"])
    .where((F.col("asd") > 0) & (F.col("prob") > 0))
    .withColumn("z_anc", (F.col("raw_score") - F.col("amean")) / F.col("asd"))
    .groupBy("sample_id", "pgs_id")
    .agg((F.sum(F.col("prob") * F.col("z_anc")) / F.sum(F.col("prob"))).alias("z_admixed"))
)
scored = (scored.join(adm, ["sample_id", "pgs_id"], "left")
          .withColumn("percentile_admixed", _norm_cdf_pct(F.col("z_admixed"))))

# registry metadata + result columns (clinical/concordance left null here — later stages fill them)
reg = spark.table("pgs_registry").select("pgs_id", "score_id", "disease", "direction", "hr_per_sd", "clinical_model")
out = (
    scored.join(reg, "pgs_id", "left")
    .withColumn("used_ancestry", F.col("msp"))
    .withColumn("integrated_z_source", F.when(F.col("z_admixed").isNotNull(), F.lit("admixed")).otherwise(F.lit("msp")))
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
