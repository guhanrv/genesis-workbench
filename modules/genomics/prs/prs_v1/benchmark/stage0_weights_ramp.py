# Databricks notebook source
# MAGIC %md
# MAGIC # Stage-0 WEIGHTS ramp — find the non-broadcast shuffle breakpoint (classic; single-node)
# MAGIC
# MAGIC The 1k *sample* ramp (`stage0_genomewide_ramp.py`) showed **zero spill at 1e9 rows** because AQE
# MAGIC broadcasts a single-PGS weight side, so the dosage never shuffles. The real shuffle/spill cliff is
# MAGIC therefore on the **weights axis**: as `n_pgs` grows, `pgs_weights` eventually exceeds the broadcast
# MAGIC threshold and the `dose ⋈ weights` join flips to a **shuffle** of the dosage side. This probes that.
# MAGIC
# MAGIC **Cost is bounded by keeping the dosage side small and fixed** (`fixed_samples`, default 128 → 128M
# MAGIC rows): once weights stop broadcasting it is the *dosage* that shuffles, so a small dosage side caps
# MAGIC the cost while still exposing the crossover. We ramp only `n_pgs` (rebuilding weights each rung),
# MAGIC full-backfill, and watch `shuffle_mb` / `spill_mb`. Guarded on time + weight-row count. Not serverless.
# MAGIC
# MAGIC Scorer / reconcile / profiler are **verbatim** from the genome-wide ramp (faithful benchmark).

# COMMAND ----------

dbutils.widgets.text("catalog", "dev_exploration_sandbox", "Catalog")
dbutils.widgets.text("schema", "prs_w_bench", "Bench schema (scratch)")
dbutils.widgets.text("synth_loci", "1000000", "Synthetic UNION loci (dosage width)")
dbutils.widgets.text("fixed_samples", "128", "Samples held FIXED across the ramp (bounds dosage-shuffle cost)")
dbutils.widgets.text("weight_per_pgs", "480000", "Loci each PGS covers (n_pgs × this ≈ total weight rows)")
dbutils.widgets.text("n_pgs_ramp", "1,2,4,8,16,32", "PGS counts to ramp through (the weights axis; fine near the broadcast threshold)")
dbutils.widgets.text("max_minutes", "30", "GUARD: stop the ramp once cumulative wall exceeds this")
dbutils.widgets.text("max_weight_rows", "40000000", "GUARD: skip a rung whose weights would exceed this many rows")
dbutils.widgets.text("dbu_per_node_hr", "2.0", "DBU/hr per node (refine from billing)")
dbutils.widgets.text("dollar_per_dbu", "0.15", "$/DBU (refine from billing)")

catalog = dbutils.widgets.get("catalog"); schema = dbutils.widgets.get("schema")
synth_loci = int(dbutils.widgets.get("synth_loci")); fixed_samples = int(dbutils.widgets.get("fixed_samples"))
weight_per_pgs = int(dbutils.widgets.get("weight_per_pgs"))
pgs_ramp = [int(x) for x in dbutils.widgets.get("n_pgs_ramp").split(",") if x.strip()]
max_minutes = float(dbutils.widgets.get("max_minutes")); max_weight_rows = int(dbutils.widgets.get("max_weight_rows"))
dbu_per_node_hr = float(dbutils.widgets.get("dbu_per_node_hr")); dollar_per_dbu = float(dbutils.widgets.get("dollar_per_dbu"))
PANEL_VERSION = "w-v1"

# COMMAND ----------

import time, json, requests
import pyspark.sql.functions as F
from delta.tables import DeltaTable

spark.sql(f"CREATE SCHEMA IF NOT EXISTS {catalog}.{schema}")
spark.sql(f"USE CATALOG {catalog}"); spark.sql(f"USE SCHEMA {schema}")
spark.conf.set("spark.sql.shuffle.partitions", "auto")

def now(): return time.time()
def cluster_shape():
    infos = spark.sparkContext._jsc.sc().statusTracker().getExecutorInfos()
    return max(0, len(infos) - 1), spark.sparkContext.defaultParallelism

# --- Spark stage profiler: diff completed stages across a phase → task/scan/shuffle/spill ---
def _stages_json():
    try:
        ui = spark.sparkContext.uiWebUrl; app = spark.sparkContext.applicationId
        return requests.get(f"{ui}/api/v1/applications/{app}/stages?status=complete", timeout=60).json()
    except Exception as e:
        return {"_error": str(e)}
def _stage_keys():
    js = _stages_json()
    return {(s["stageId"], s.get("attemptId", 0)) for s in js} if isinstance(js, list) else set()
def _profile(before):
    js = _stages_json()
    if not isinstance(js, list): return {"_error": (js or {}).get("_error", "stages api n/a")}
    new = [s for s in js if (s["stageId"], s.get("attemptId", 0)) not in before]
    if not new: return {"note": "no new stages"}
    def g(s, k): return s.get(k, 0) or 0
    return {"task_s": round(sum(g(s, "executorRunTime") for s in new) / 1000.0, 1),
            "input_mb": round(sum(g(s, "inputBytes") for s in new) / 1e6, 1),
            "shuffle_mb": round(sum(g(s, "shuffleReadBytes") + g(s, "shuffleWriteBytes") for s in new) / 1e6, 1),
            "spill_mb": round(sum(g(s, "memoryBytesSpilled") + g(s, "diskBytesSpilled") for s in new) / 1e6, 1),
            "n_tasks": sum(g(s, "numTasks") for s in new)}

# COMMAND ----------

# MAGIC %md ### Stores + fixed sample cohort + union loci (built ONCE; weights rebuilt per rung)

# COMMAND ----------

for t in ["dosage", "pgs_weights", "pgs_registry", "pgs_panel_ref", "sample_ancestry", "prs_scores", "_prs_reconcile_plan", "_gw_loci"]:
    spark.sql(f"DROP TABLE IF EXISTS {t}")
spark.sql("CREATE TABLE dosage (sample_id STRING, variant_id STRING, dose DOUBLE) USING DELTA CLUSTER BY (sample_id)")
spark.sql("CREATE TABLE prs_scores (sample_id STRING, pgs_id STRING, weight_sha STRING, panel_version STRING, raw_score DOUBLE, n_variants_matched INT, coverage_pct DOUBLE, small_score BOOLEAN, most_similar_pop STRING, used_ancestry STRING, z_msp DOUBLE, percentile_msp DOUBLE, z_admixed DOUBLE, percentile_admixed DOUBLE, integrated_z_source STRING, integrated_risk_10yr DOUBLE, clinical_risk_10yr DOUBLE, risk_category STRING, concordance_verdict STRING, computed_at TIMESTAMP) USING DELTA PARTITIONED BY (pgs_id)")
spark.sql("CREATE TABLE IF NOT EXISTS stage0_w_metrics (run_ts TIMESTAMP, phase STRING, n_samples LONG, n_dosage_rows LONG, n_weight_rows LONG, n_cells LONG, wall_clock_s DOUBLE, task_s DOUBLE, input_mb DOUBLE, shuffle_mb DOUBLE, spill_mb DOUBLE, est_cost_usd DOUBLE) USING DELTA")

# union loci
spark.range(synth_loci).select(F.concat_ws(":", F.lit("1"), (F.col("id") + 1).cast("string"), F.lit("A"), F.lit("G")).alias("variant_id")).write.mode("overwrite").saveAsTable("_gw_loci")

# fixed sample cohort (built once) — dosage width = synth_loci, so dosage rows = fixed_samples × synth_loci
s = spark.range(fixed_samples).select(F.concat(F.lit("SAMP"), F.format_string("%07d", F.col("id"))).alias("sample_id"))
(s.crossJoin(spark.table("_gw_loci")).withColumn("dose", F.pmod(F.abs(F.hash(F.concat("sample_id", "variant_id"))), F.lit(3)).cast("double"))).write.mode("append").saveAsTable("dosage")
(s.withColumn("most_similar_pop", F.lit("European"))).write.mode("append").saveAsTable("sample_ancestry")
n_dose_fixed = spark.table("dosage").count()
print(f"fixed cohort: {fixed_samples} samples × {synth_loci:,} loci = {n_dose_fixed:,} dosage rows")

# COMMAND ----------

# MAGIC %md ### build_weights(n_pgs) — rebuilds pgs_weights/registry/panel for the rung (models real PGS overlap)

# COMMAND ----------

def build_weights(n_pgs):
    dens_ppm = min(1_000_000, int(1_000_000 * weight_per_pgs / float(synth_loci)))
    wparts = []
    for k in range(n_pgs):
        base = spark.table("_gw_loci")
        if dens_ppm < 1_000_000:
            base = base.where((F.abs(F.hash(F.concat(F.col("variant_id"), F.lit(f"|{k}")))) % 1_000_000) < dens_ppm)
        wparts.append(base
            .withColumn("pgs_id", F.lit(f"GWPGS{k:03d}")).withColumn("effect_allele", F.lit("A")).withColumn("other_allele", F.lit("G"))
            .withColumn("weight", ((F.abs(F.hash(F.concat(F.col("variant_id"), F.lit(f"w{k}")))) % 2000) - 1000) / 1000.0)
            .withColumn("weight_sha", F.lit(f"gwsha{k:03d}"))
            .select("pgs_id", "variant_id", "effect_allele", "other_allele", "weight", "weight_sha"))
    w = wparts[0]
    for x in wparts[1:]: w = w.unionByName(x)
    w.write.mode("overwrite").saveAsTable("pgs_weights")
    # Warm the catalog stats so the planner's broadcast decision uses the TRUE size (the prior run's
    # confound: a fresh overwrite left stats cold → AQE never broadcast even a tiny weights table).
    spark.sql("ANALYZE TABLE pgs_weights COMPUTE STATISTICS")
    (spark.table("pgs_weights").groupBy("pgs_id", "weight_sha").agg(F.count("*").alias("n_variants"))
        .withColumn("score_id", F.col("pgs_id")).withColumn("disease", F.lit("gw")).withColumn("direction", F.lit("risk"))
        .withColumn("body_system", F.lit("gw")).withColumn("hr_per_sd", F.lit(1.5)).withColumn("clinical_model", F.lit(None).cast("string"))
        .withColumn("training_ancestries", F.lit("European")).withColumn("weight_path", F.lit("synthetic")).withColumn("registered_at", F.current_timestamp())
     ).write.mode("overwrite").saveAsTable("pgs_registry")
    (spark.table("pgs_registry").select("pgs_id", "weight_sha").crossJoin(spark.createDataFrame([("European",)], ["superpop"]))
        .withColumn("mean", F.lit(0.0)).withColumn("sd", F.lit(1.0)).withColumn("quantiles", F.lit(None).cast("array<double>"))
        .withColumn("n_panel", F.lit(3202)).withColumn("panel_version", F.lit(PANEL_VERSION))
        .select("pgs_id", "superpop", "mean", "sd", "quantiles", "n_panel", "panel_version", "weight_sha")
     ).write.mode("overwrite").saveAsTable("pgs_panel_ref")
    n_w = spark.table("pgs_weights").count()
    size_bytes = spark.sql("DESCRIBE DETAIL pgs_weights").select("sizeInBytes").collect()[0][0]
    return n_w, int(size_bytes or 0)

# COMMAND ----------

# MAGIC %md ### scorer + reconcile + timed/profiled phase (verbatim from stage0_genomewide_ramp)

# COMMAND ----------

def reconcile():
    registry = spark.table("pgs_registry").select("pgs_id", "weight_sha")
    desired = spark.table("dosage").select("sample_id").distinct().crossJoin(registry).withColumn("panel_version", F.lit(PANEL_VERSION))
    existing = spark.table("prs_scores").select("sample_id", "pgs_id", F.col("weight_sha").alias("e_sha"), F.col("panel_version").alias("e_pv"))
    plan = (desired.join(existing, ["sample_id", "pgs_id"], "left")
            .where((F.col("e_sha").isNull()) | (F.col("e_sha") != F.col("weight_sha")) | (F.col("e_pv").isNull()) | (F.col("e_pv") != F.col("panel_version")))
            .select("sample_id", "pgs_id", "weight_sha", "panel_version"))
    plan.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable("_prs_reconcile_plan")
    return spark.table("_prs_reconcile_plan")

def score(plan):
    planned_samples = plan.select("sample_id").distinct()
    planned_pgs = plan.select("pgs_id", "weight_sha").distinct()
    SAMPLE_PREDICATE_MAX = 200
    _ids = [r["sample_id"] for r in planned_samples.limit(SAMPLE_PREDICATE_MAX + 1).collect()]
    if 0 < len(_ids) <= SAMPLE_PREDICATE_MAX:
        dose = spark.table("dosage").where(F.col("sample_id").isin(_ids))
    else:
        dose = spark.table("dosage").join(F.broadcast(planned_samples), "sample_id")
    wts = spark.table("pgs_weights").join(F.broadcast(planned_pgs), ["pgs_id", "weight_sha"])   # AQE decides broadcast vs shuffle
    raw = (dose.join(wts, "variant_id").groupBy("sample_id", "pgs_id", "weight_sha")
           .agg(F.sum(F.col("dose") * F.col("weight")).alias("raw_score"), F.count(F.lit(1)).alias("n_variants_matched"))
           .join(plan, ["sample_id", "pgs_id", "weight_sha"]))
    nvar = spark.table("pgs_registry").select("pgs_id", F.col("n_variants").alias("pgs_nvar"))
    raw = (raw.join(nvar, "pgs_id", "left").withColumn("coverage_pct", F.when(F.col("pgs_nvar") > 0, 100.0 * F.col("n_variants_matched") / F.col("pgs_nvar"))).withColumn("small_score", F.col("n_variants_matched") < F.lit(1000)))
    anc = spark.table("sample_ancestry").select("sample_id", F.col("most_similar_pop").alias("msp"))
    ref = spark.table("pgs_panel_ref").select("pgs_id", F.col("superpop").alias("msp"), "panel_version", F.col("mean").alias("ref_mean"), F.col("sd").alias("ref_sd"))
    scored = (raw.join(anc, "sample_id", "left").join(ref, ["pgs_id", "msp", "panel_version"], "left")
              .withColumn("z_msp", F.when(F.col("ref_sd") > 0, (F.col("raw_score") - F.col("ref_mean")) / F.col("ref_sd"))))
    out = (scored.withColumn("used_ancestry", F.col("msp")).withColumn("percentile_msp", F.lit(None).cast("double"))
           .withColumn("z_admixed", F.lit(None).cast("double")).withColumn("percentile_admixed", F.lit(None).cast("double"))
           .withColumn("integrated_z_source", F.lit("msp")).withColumn("integrated_risk_10yr", F.lit(None).cast("double"))
           .withColumn("clinical_risk_10yr", F.lit(None).cast("double")).withColumn("risk_category", F.lit(None).cast("string"))
           .withColumn("concordance_verdict", F.lit(None).cast("string")).withColumn("computed_at", F.current_timestamp())
           .select("sample_id", "pgs_id", "weight_sha", "panel_version", "raw_score", "n_variants_matched", "coverage_pct", "small_score",
                   F.col("msp").alias("most_similar_pop"), "used_ancestry", "z_msp", "percentile_msp", "z_admixed", "percentile_admixed",
                   "integrated_z_source", "integrated_risk_10yr", "clinical_risk_10yr", "risk_category", "concordance_verdict", "computed_at"))
    out.createOrReplaceTempView("_scored")
    (DeltaTable.forName(spark, f"{catalog}.{schema}.prs_scores").alias("t")
       .merge(spark.table("_scored").alias("s"), "t.sample_id = s.sample_id AND t.pgs_id = s.pgs_id")
       .whenMatchedUpdateAll().whenNotMatchedInsertAll().execute())

RESULTS = []
def run_phase(phase, n_samples):
    n_exec, cores = cluster_shape(); before = _stage_keys(); t0 = now()
    plan = reconcile(); n_cells = plan.count()
    if n_cells > 0: score(plan)
    wall = now() - t0; prof = _profile(before)
    n_dose = spark.table("dosage").count(); n_w = spark.table("pgs_weights").count()
    cost = (1 + n_exec) * dbu_per_node_hr * (wall / 3600.0) * dollar_per_dbu
    (spark.createDataFrame([(phase, n_samples, n_dose, n_w, n_cells, wall, prof.get("task_s"), prof.get("input_mb"), prof.get("shuffle_mb"), prof.get("spill_mb"), cost)],
        "phase string, n_samples long, n_dosage_rows long, n_weight_rows long, n_cells long, wall_clock_s double, task_s double, input_mb double, shuffle_mb double, spill_mb double, est_cost_usd double")
        .withColumn("run_ts", F.current_timestamp())
        .select("run_ts", "phase", "n_samples", "n_dosage_rows", "n_weight_rows", "n_cells", "wall_clock_s", "task_s", "input_mb", "shuffle_mb", "spill_mb", "est_cost_usd")
     ).write.mode("append").saveAsTable("stage0_w_metrics")
    RESULTS.append({"phase": phase, "n_samples": n_samples, "n_dosage_rows": n_dose, "n_weight_rows": n_w, "n_cells": n_cells,
                    "wall_clock_s": round(wall, 1), "est_cost_usd": round(cost, 4), "profile": prof})
    print(f"[{phase}] n_w={n_w:,} dose_rows={n_dose:,} cells={n_cells} wall={wall:.1f}s "
          f"| task_s={prof.get('task_s')} in_mb={prof.get('input_mb')} shuf_mb={prof.get('shuffle_mb')} spill_mb={prof.get('spill_mb')} ${cost:.4f}")
    return wall

# COMMAND ----------

# MAGIC %md ### RAMP: grow n_pgs (guarded) — watch shuffle_mb flip from ~broadcast to full dosage shuffle

# COMMAND ----------

bcast_threshold = spark.conf.get("spark.sql.autoBroadcastJoinThreshold")
print(f"spark.sql.autoBroadcastJoinThreshold = {bcast_threshold} (weights broadcast below this)")

cum0 = now(); stopped = None
for NP in pgs_ramp:
    if NP * weight_per_pgs > max_weight_rows:
        stopped = f"weight-row guard: {NP}×{weight_per_pgs} > max_weight_rows={max_weight_rows}"; break
    n_w, size_bytes = build_weights(NP)
    spark.sql("TRUNCATE TABLE prs_scores")                 # fresh full backfill so rungs are comparable
    print(f"  pgs{NP}: {n_w:,} weight rows, sizeInBytes={size_bytes/1e6:.1f} MB")
    run_phase(f"wramp_pgs{NP}", fixed_samples)
    RESULTS[-1]["weight_bytes_mb"] = round(size_bytes / 1e6, 1)   # attach the size that drove the broadcast decision
    if (now() - cum0) / 60.0 > max_minutes:
        stopped = f"time guard: cumulative > {max_minutes} min"; break

print("\n=== stage0_w_metrics ===")
spark.table("stage0_w_metrics").orderBy("run_ts").show(200, truncate=False)
dbutils.notebook.exit(json.dumps({"synth_loci": synth_loci, "fixed_samples": fixed_samples,
                                  "weight_per_pgs": weight_per_pgs, "n_pgs_ramp": pgs_ramp,
                                  "autoBroadcastJoinThreshold": bcast_threshold,
                                  "stopped": stopped, "phases": RESULTS}))
