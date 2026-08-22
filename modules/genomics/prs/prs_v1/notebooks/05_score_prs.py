# Databricks notebook source
# MAGIC %md
# MAGIC # PRS scorer — SQL join-aggregate over the reconcile plan (classic; NOT serverless yet)
# MAGIC
# MAGIC Reads `_prs_reconcile_plan` (written by `03_reconcile`) and scores **only those (sample, pgs) cells** —
# MAGIC so a run can never touch the full grid. Default mechanism is a plain **SQL join-aggregate** (predictable,
# MAGIC observable cost); `applyInPandas` is a benchmark-gated alternative, not used here.
# MAGIC
# MAGIC ```
# MAGIC raw = Σ_variant  dose · weight    (dose is effect-oriented per (chrom,pos,effect,other))
# MAGIC z_msp = (raw − panel_mean[pgs,msp]) / panel_sd[pgs,msp]      (reference-panel normalized; no cohort re-rank)
# MAGIC ```
# MAGIC Runs on a classic cluster started minimal + titrated (see job.yml). MERGE-upserts the flat `prs_scores`.
# MAGIC
# MAGIC **Sample-axis chunking (default 10):** n=10 × 197 PGS densify was ~164 GB shuffle
# MAGIC with no spill on 4 workers; n=100 all-at-once wrote ~1.5 TB and spilled ~3 TB.
# MAGIC Each sample batch re-filters `dosage` (Liquid Clustering file-skip) so the join
# MAGIC stays in the measured no-spill shape. Set `sample_chunk_size=0` for all-at-once.
# MAGIC PGS-axis chunking (`pgs_chunk_size` / `max_weight_rows_per_chunk`) is optional and
# MAGIC default-off — a 3M weight budget was 4.4× slower at n=10 (re-reads dosage).

# COMMAND ----------

dbutils.widgets.text("catalog", "genesis_workbench", "Catalog")
dbutils.widgets.text("schema", "genesis_schema", "Schema")
# Default mean_impute: pgs_panel_ref (the z reference) is built by ref_00_build_panel_stats under
# mean-imputation (missing→2·AF), so members MUST be scored the same way or z is miscalibrated (raw
# and panel_mean would be computed under different missing policies). At WGS coverage the raw-score
# effect vs drop is negligible. Use drop only for a raw-only run with no panel/z.
dbutils.widgets.text("missing_mode", "mean_impute", "mean_impute = missing→2·AF from pgs_panel_afreq (matches panel; calibrated z) | drop = missing→0 (raw-only)")
# 0 = score the full planned PGS set in one join (legacy / baseline A/B). >0 = fixed PGS count per chunk.
dbutils.widgets.text("pgs_chunk_size", "0", "Fixed # PGS per score chunk (0 = use weight budget or all-at-once)")
# When pgs_chunk_size=0: greedy-pack PGS until Σ weight rows hits this budget. 0 disables packing.
dbutils.widgets.text("max_weight_rows_per_chunk", "0", "Weight-row budget per chunk when pgs_chunk_size=0 (0 = no PGS chunking)")
# 0 = all planned samples in one join (n=100 spilled). 10 = measured no-spill shape.
dbutils.widgets.text("sample_chunk_size", "10", "Max samples per score join (0 = all-at-once)")
# `auto` is the measured no-spill setting; a fixed count is A/B-only. Pinning too low grows
# per-task memory, which is what OOM-killed the standard-memory node A/B.
dbutils.widgets.text("shuffle_partitions", "auto", "spark.sql.shuffle.partitions (auto = AQE picks)")
catalog = dbutils.widgets.get("catalog"); schema = dbutils.widgets.get("schema")
missing_mode = dbutils.widgets.get("missing_mode").strip().lower()
pgs_chunk_size = int(dbutils.widgets.get("pgs_chunk_size") or "0")
max_weight_rows_per_chunk = int(dbutils.widgets.get("max_weight_rows_per_chunk") or "0")
sample_chunk_size = int(dbutils.widgets.get("sample_chunk_size") or "0")
shuffle_partitions = dbutils.widgets.get("shuffle_partitions").strip() or "auto"

# COMMAND ----------

import math
import os
import sys
import time

import pyspark.sql.functions as F
from delta.tables import DeltaTable
from pyspark.sql import DataFrame

spark.sql(f"USE CATALOG {catalog}"); spark.sql(f"USE SCHEMA {schema}")
spark.conf.set("spark.sql.shuffle.partitions", shuffle_partitions)  # bounded by classic cluster size

plan = spark.table("_prs_reconcile_plan")                 # sample_id, pgs_id, weight_sha, panel_version
if plan.limit(1).count() == 0:
    dbutils.notebook.exit("0")  # reconcile found nothing; scorer is a no-op

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "..", "lib")))
from prs_readiness import assert_scoring_ready
from score_chunks import chunk_by_weight_budget, chunk_fixed

assert_scoring_ready(spark, catalog, schema, plan, require_models=False)

planned_samples = plan.select("sample_id").distinct()
planned_pgs = plan.select("pgs_id", "weight_sha").distinct()

# COMMAND ----------

# MAGIC %md
# MAGIC ### 0. Sample batches + optional PGS chunks
# MAGIC Each sample batch file-skips `dosage` and MERGE-upserts independently.

# COMMAND ----------

WEIGHTS_BROADCAST_MAX_ROWS = 3_000_000
SAMPLE_PREDICATE_MAX = 200  # isin() file-skip; larger batches use a broadcast sample join

_sample_ids = [r["sample_id"] for r in planned_samples.select("sample_id").collect()]
_sample_chunks = chunk_fixed(_sample_ids, sample_chunk_size)

_use_impute = missing_mode == "mean_impute" and spark.catalog.tableExists("pgs_panel_afreq")
if missing_mode == "mean_impute" and not _use_impute:
    print("WARN: missing_mode=mean_impute but pgs_panel_afreq absent → falling back to drop")

afreq = (
    spark.table("pgs_panel_afreq").select("variant_id", "af_effect")
    if _use_impute else None
)


def _dose_for_samples(sids):
    """Prune dosage to one sample batch (Liquid Clustering file-skip when small)."""
    if 0 < len(sids) <= SAMPLE_PREDICATE_MAX:
        return spark.table("dosage").where(F.col("sample_id").isin(sids))
    batch = spark.createDataFrame([(s,) for s in sids], schema="sample_id STRING")
    return spark.table("dosage").join(F.broadcast(batch), "sample_id")

# Weight counts for budget packing (cheap agg over planned PGS only).
_wcount = (
    spark.table("pgs_weights")
    .join(F.broadcast(planned_pgs), ["pgs_id", "weight_sha"])
    .groupBy("pgs_id", "weight_sha")
    .count()
    .withColumnRenamed("count", "n_weights")
)
_pgs_rows = [r.asDict() for r in _wcount.collect()]
_pgs_rows.sort(key=lambda r: (-int(r["n_weights"]), r["pgs_id"]))

if pgs_chunk_size > 0:
    _chunks = chunk_fixed(_pgs_rows, pgs_chunk_size)
    _chunk_mode = f"fixed_size={pgs_chunk_size}"
elif max_weight_rows_per_chunk > 0 and len(_pgs_rows) > 1:
    _chunks = chunk_by_weight_budget(
        _pgs_rows, max_weight_rows=max_weight_rows_per_chunk
    )
    _chunk_mode = f"weight_budget={max_weight_rows_per_chunk}"
else:
    _chunks = chunk_fixed(_pgs_rows, 0)
    _chunk_mode = "all_at_once"

print(
    f"score chunks: samples={len(_sample_ids)} sample_batches={len(_sample_chunks)} "
    f"(size={sample_chunk_size or 'all'}) pgs_mode={_chunk_mode} n_pgs={len(_pgs_rows)} "
    f"n_pgs_chunks={len(_chunks)} missing_mode={'mean_impute' if _use_impute else 'drop'}"
)
for i, ch in enumerate(_chunks):
    print(
        f"  pgs_chunk[{i}] n_pgs={len(ch)} n_weights={sum(int(r['n_weights']) for r in ch)} "
        f"pgs={[r['pgs_id'] for r in ch[:5]]}{'…' if len(ch) > 5 else ''}"
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ### 1–3. Per-chunk raw score → z normalization → MERGE

# COMMAND ----------

_has_outlier = "is_outlier" in spark.table("sample_ancestry").columns
anc = spark.table("sample_ancestry").select(
    "sample_id", F.col("most_similar_pop").alias("msp"),
    (F.col("is_outlier") if _has_outlier else F.lit(False)).alias("is_outlier"))
ref = spark.table("pgs_panel_ref").select(
    "pgs_id", F.col("superpop").alias("msp"), "panel_version",
    F.col("mean").alias("ref_mean"), F.col("sd").alias("ref_sd"))
probs = (spark.table("sample_ancestry")
         .select("sample_id", F.from_json(F.col("rf_probs"), "map<string,double>").alias("_p"))
         .select("sample_id", F.explode("_p").alias("superpop", "prob")))
ref_all = spark.table("pgs_panel_ref").select(
    "pgs_id", "superpop", "panel_version", F.col("mean").alias("amean"), F.col("sd").alias("asd"))
nvar = spark.table("pgs_registry").select("pgs_id", F.col("n_variants").alias("pgs_nvar"))
reg = spark.table("pgs_registry").select(
    "pgs_id", "score_id", "disease", "direction", "hr_per_sd", "clinical_model")

@F.udf("double")
def _norm_cdf_pct(z):
    return None if z is None else 100.0 * 0.5 * (1.0 + math.erf(float(z) / math.sqrt(2.0)))


def _raw_agg(plan_c: DataFrame, wts_c: DataFrame, dose_c: DataFrame, *, broadcast_wts: bool) -> DataFrame:
    """Join-aggregate raw scores for one (sample-batch × PGS-chunk)."""
    dose_sv = dose_c.select("sample_id", "variant_id", F.col("dose").alias("_d"))
    if _use_impute:
        wts_panel = wts_c.join(afreq, "variant_id", "inner")
        _wp = F.broadcast(wts_panel) if broadcast_wts else wts_panel
        # densify: cell × panel-matched variants, then left-join dose (missing→2·AF)
        agg = (
            plan_c.join(_wp, ["pgs_id", "weight_sha"])
            .join(dose_sv, ["sample_id", "variant_id"], "left")
            .withColumn("dose_f", F.coalesce(F.col("_d"), 2.0 * F.col("af_effect")))
            .groupBy("sample_id", "pgs_id", "weight_sha")
            .agg(
                F.sum(F.col("dose_f") * F.col("weight")).alias("raw_score"),
                F.sum(F.col("_d").isNotNull().cast("int")).alias("n_variants_matched"),
            )
        )
    else:
        _wts = F.broadcast(wts_c) if broadcast_wts else wts_c
        agg = (
            dose_c.join(_wts, "variant_id")
            .groupBy("sample_id", "pgs_id", "weight_sha")
            .agg(
                F.sum(F.col("dose") * F.col("weight")).alias("raw_score"),
                F.count(F.lit(1)).alias("n_variants_matched"),
            )
        )

    raw = (
        plan_c.join(agg, ["sample_id", "pgs_id", "weight_sha"], "left")
        .withColumn("raw_score", F.coalesce(F.col("raw_score"), F.lit(0.0)))
        .withColumn("n_variants_matched", F.coalesce(F.col("n_variants_matched"), F.lit(0)))
    )
    return (
        raw.join(nvar, "pgs_id", "left")
        .withColumn(
            "coverage_pct",
            F.when(F.col("pgs_nvar") > 0, 100.0 * F.col("n_variants_matched") / F.col("pgs_nvar")),
        )
        .withColumn("small_score", F.col("n_variants_matched") < F.lit(1000))
    )


def _normalize(raw: DataFrame) -> DataFrame:
    scored = (
        raw.join(anc, "sample_id", "left")
        .join(ref, ["pgs_id", "msp", "panel_version"], "left")
        .withColumn(
            "z_msp",
            F.when(F.col("ref_sd") > 0, (F.col("raw_score") - F.col("ref_mean")) / F.col("ref_sd")),
        )
        .withColumn("percentile_msp", _norm_cdf_pct(F.col("z_msp")))
    )
    adm = (
        raw.select("sample_id", "pgs_id", "panel_version", "raw_score")
        .join(probs, "sample_id")
        .join(ref_all, ["pgs_id", "superpop", "panel_version"])
        .where((F.col("asd") > 0) & (F.col("prob") > 0))
        .withColumn("z_anc", (F.col("raw_score") - F.col("amean")) / F.col("asd"))
        .groupBy("sample_id", "pgs_id")
        .agg((F.sum(F.col("prob") * F.col("z_anc")) / F.sum(F.col("prob"))).alias("z_admixed"))
    )
    scored = (
        scored.join(adm, ["sample_id", "pgs_id"], "left")
        .withColumn("percentile_admixed", _norm_cdf_pct(F.col("z_admixed")))
    )
    for _zc in ("z_msp", "percentile_msp", "z_admixed", "percentile_admixed"):
        scored = scored.withColumn(_zc, F.when(~F.col("is_outlier"), F.col(_zc)))
    return (
        scored.join(reg, "pgs_id", "left")
        .withColumn("used_ancestry", F.col("msp"))
        .withColumn(
            "integrated_z_source",
            F.when(F.col("z_admixed").isNotNull(), F.lit("admixed")).otherwise(F.lit("msp")),
        )
        .withColumn("integrated_risk_10yr", F.lit(None).cast("double"))
        .withColumn("clinical_risk_10yr", F.lit(None).cast("double"))
        .withColumn("risk_category", F.lit(None).cast("string"))
        .withColumn("concordance_verdict", F.lit(None).cast("string"))
        .withColumn("computed_at", F.current_timestamp())
        .select(
            "sample_id", "pgs_id", "weight_sha", "panel_version", "raw_score",
            "n_variants_matched", "coverage_pct", "small_score",
            F.col("msp").alias("most_similar_pop"), "used_ancestry",
            "z_msp", "percentile_msp", "z_admixed", "percentile_admixed",
            "integrated_z_source", "integrated_risk_10yr", "clinical_risk_10yr",
            "risk_category", "concordance_verdict", "computed_at",
        )
    )


tgt = DeltaTable.forName(spark, f"{catalog}.{schema}.prs_scores")
total_cells = 0
t_all = time.time()
n_joins = len(_sample_chunks) * len(_chunks)
join_i = 0

for si, sids in enumerate(_sample_chunks):
    dose_c = _dose_for_samples(sids)
    plan_s = plan.where(F.col("sample_id").isin(sids)) if len(sids) <= SAMPLE_PREDICATE_MAX else (
        plan.join(F.broadcast(spark.createDataFrame([(s,) for s in sids], "sample_id STRING")), "sample_id")
    )
    for ci, ch in enumerate(_chunks):
        join_i += 1
        t0 = time.time()
        chunk_keys = spark.createDataFrame(
            [(r["pgs_id"], r["weight_sha"]) for r in ch],
            schema="pgs_id STRING, weight_sha STRING",
        )
        plan_c = plan_s.join(F.broadcast(chunk_keys), ["pgs_id", "weight_sha"])
        wts_c = spark.table("pgs_weights").join(F.broadcast(chunk_keys), ["pgs_id", "weight_sha"])
        n_w = sum(int(r["n_weights"]) for r in ch)
        broadcastable = n_w <= WEIGHTS_BROADCAST_MAX_ROWS

        out = _normalize(_raw_agg(plan_c, wts_c, dose_c, broadcast_wts=broadcastable))
        out.createOrReplaceTempView("_scored_chunk")
        (
            tgt.alias("t")
            .merge(
                spark.table("_scored_chunk").alias("s"),
                "t.sample_id = s.sample_id AND t.pgs_id = s.pgs_id",
            )
            .whenMatchedUpdateAll()
            .whenNotMatchedInsertAll()
            .execute()
        )
        # Count the PLAN, not `_scored_chunk`: the view is a lazy plan over the whole densify
        # join, so counting it re-runs that join after the MERGE already consumed it. Every
        # step from plan_c to `out` is a left join on keys unique in the right side, so the
        # scored row count always equals the planned cell count.
        n = plan_c.count()
        total_cells += n
        print(
            f"join[{join_i}/{n_joins}] sample_batch={si} n_samples={len(sids)} "
            f"pgs_chunk={ci} n_pgs={len(ch)} n_weights={n_w} cells={n} "
            f"broadcast={broadcastable} wall_s={time.time() - t0:.1f}"
        )

print(
    f"scored + upserted {total_cells} (sample,pgs) cells into {catalog}.{schema}.prs_scores "
    f"in {n_joins} joins ({time.time() - t_all:.1f}s)"
)
dbutils.notebook.exit(str(total_cells))
