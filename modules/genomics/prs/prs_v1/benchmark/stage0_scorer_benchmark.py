# Databricks notebook source
# MAGIC %md
# MAGIC # Stage-0 scorer benchmark — SQL join-aggregate titration (classic; NOT serverless)
# MAGIC
# MAGIC The plan's **gate**: measure the distributed scorer's cost/throughput on a REAL cohort, pick the
# MAGIC smallest classic cluster that meets the need, confirm incrementality bounds steady-state cost, and
# MAGIC derive the **`dbu_per_cell`** that the reconcile kill-switch uses for its cost estimate — before any
# MAGIC full backfill runs.
# MAGIC
# MAGIC **Data (PHI-free, reuses what's already in the workspace):** explode the existing 3202-sample 1000G
# MAGIC `pca_dosage` (Glow wide `states[]`/`sample_ids[]`) into the long `dosage` store, and synthesize
# MAGIC `pgs_weights` at those same loci for `n_pgs` scores. The join-aggregate cost depends on SCALE
# MAGIC (dosage rows × weight rows), not on allele identity, so synthetic weights give a faithful throughput
# MAGIC benchmark. Extrapolate `$/cell` to the 48.4M-weight × N-sample backfill.
# MAGIC
# MAGIC **How to titrate:** run `mode=prep` ONCE (single-node), then `mode=score` at each rung of the
# MAGIC num_workers ladder (see benchmark/STAGE0_RUNPLAN.md). Each score run appends a row per phase to
# MAGIC `stage0_metrics`. Nothing runs on serverless; every run is a fixed-size classic job cluster.

# COMMAND ----------

dbutils.widgets.text("catalog", "dev_exploration_sandbox", "Catalog")
dbutils.widgets.text("schema", "prs_stage0_bench", "Bench schema (scratch)")
dbutils.widgets.text("source_dosage", "dev_exploration_sandbox.genesis_workbench.pca_dosage_e0cbd8839f114af595973ab9c49e2214", "Wide dosage to explode")
dbutils.widgets.text("mode", "prep", "prep | score | full")
dbutils.widgets.text("n_pgs", "5", "Synthetic PGS count (columns)")
dbutils.widgets.text("density_pct", "60", "Percent of loci each PGS weights (0-100)")
dbutils.widgets.text("max_loci", "50000", "Cap loci in prep (0 = all; start small, extrapolate $/cell)")
dbutils.widgets.text("dbu_per_node_hr", "2.0", "DBU/hr per node (c3d-highmem-8 ~2; refine from billing)")
dbutils.widgets.text("dollar_per_dbu", "0.15", "$/DBU (jobs compute; refine from billing)")

catalog = dbutils.widgets.get("catalog"); schema = dbutils.widgets.get("schema")
source_dosage = dbutils.widgets.get("source_dosage")
mode = dbutils.widgets.get("mode").strip().lower()
n_pgs = int(dbutils.widgets.get("n_pgs")); density_pct = int(dbutils.widgets.get("density_pct"))
max_loci = int(dbutils.widgets.get("max_loci"))
dbu_per_node_hr = float(dbutils.widgets.get("dbu_per_node_hr")); dollar_per_dbu = float(dbutils.widgets.get("dollar_per_dbu"))

# COMMAND ----------

import time, math
import pyspark.sql.functions as F
from delta.tables import DeltaTable

spark.sql(f"CREATE SCHEMA IF NOT EXISTS {catalog}.{schema}")
spark.sql(f"USE CATALOG {catalog}"); spark.sql(f"USE SCHEMA {schema}")
spark.conf.set("spark.sql.shuffle.partitions", "auto")

def cluster_shape():
    infos = spark.sparkContext._jsc.sc().statusTracker().getExecutorInfos()
    n_exec = max(0, len(infos) - 1)      # minus driver
    cores = spark.sparkContext.defaultParallelism
    return n_exec, cores

def now(): return time.time()

# COMMAND ----------

# MAGIC %md ### DDL — the five stores + a metrics table (idempotent)

# COMMAND ----------

spark.sql("CREATE TABLE IF NOT EXISTS pgs_registry (pgs_id STRING, score_id STRING, disease STRING, direction STRING, body_system STRING, hr_per_sd DOUBLE, clinical_model STRING, training_ancestries STRING, weight_sha STRING, n_variants BIGINT, weight_path STRING, registered_at TIMESTAMP) USING DELTA")
spark.sql("CREATE TABLE IF NOT EXISTS pgs_weights (pgs_id STRING, variant_id STRING, effect_allele STRING, other_allele STRING, weight DOUBLE, weight_sha STRING) USING DELTA")
spark.sql("CREATE TABLE IF NOT EXISTS pgs_panel_ref (pgs_id STRING, superpop STRING, mean DOUBLE, sd DOUBLE, quantiles ARRAY<DOUBLE>, n_panel INT, panel_version STRING, weight_sha STRING) USING DELTA")
spark.sql("CREATE TABLE IF NOT EXISTS dosage (sample_id STRING, variant_id STRING, dose DOUBLE) USING DELTA CLUSTER BY (sample_id)")
spark.sql("CREATE TABLE IF NOT EXISTS sample_ancestry (sample_id STRING, most_similar_pop STRING) USING DELTA")
spark.sql("CREATE TABLE IF NOT EXISTS prs_scores (sample_id STRING, pgs_id STRING, weight_sha STRING, panel_version STRING, raw_score DOUBLE, n_variants_matched INT, coverage_pct DOUBLE, small_score BOOLEAN, most_similar_pop STRING, used_ancestry STRING, z_msp DOUBLE, percentile_msp DOUBLE, z_admixed DOUBLE, percentile_admixed DOUBLE, integrated_z_source STRING, integrated_risk_10yr DOUBLE, clinical_risk_10yr DOUBLE, risk_category STRING, concordance_verdict STRING, computed_at TIMESTAMP) USING DELTA PARTITIONED BY (pgs_id)")
spark.sql("CREATE TABLE IF NOT EXISTS stage0_metrics (run_ts TIMESTAMP, phase STRING, mechanism STRING, n_exec INT, cores INT, n_cells BIGINT, n_dosage_rows BIGINT, n_weight_rows BIGINT, wall_clock_s DOUBLE, cells_per_s DOUBLE, est_dbu DOUBLE, est_cost_usd DOUBLE, dbu_per_cell DOUBLE) USING DELTA")

PANEL_VERSION = "bench-v1"

# COMMAND ----------

# MAGIC %md
# MAGIC ### PREP (run once, single-node): explode pca_dosage → long dosage + synth weights/panel/ancestry
# MAGIC The join-aggregate scales with dosage-rows × weight-rows; synthetic (deterministic, hash-based)
# MAGIC weights at the real loci reproduce that scale faithfully. Skipped unless `mode` is prep/full.

# COMMAND ----------

if mode in ("prep", "full"):
    t0 = now()
    src = spark.table(source_dosage)  # chrom, pos, states[], sample_ids[] (one row per locus)
    if max_loci > 0:
        src = src.limit(max_loci)     # cap loci → bound dosage rows for a cheap first run
    long = (src
        .select("chrom", "pos", F.arrays_zip("sample_ids", "states").alias("z"))
        .select("chrom", "pos", F.explode("z").alias("z"))
        .select(
            F.col("z.sample_ids").alias("sample_id"),
            F.concat_ws(":", F.regexp_replace(F.col("chrom"), "chr", ""), F.col("pos").cast("string"),
                        F.lit("A"), F.lit("G")).alias("variant_id"),
            F.col("z.states").cast("double").alias("dose"))
        .where(F.col("dose").isNotNull()))
    # (re)create clustered so the A/B re-measure exercises Liquid Clustering file-skipping
    spark.sql("DROP TABLE IF EXISTS dosage")
    spark.sql("CREATE TABLE dosage (sample_id STRING, variant_id STRING, dose DOUBLE) USING DELTA CLUSTER BY (sample_id)")
    long.write.mode("append").saveAsTable("dosage")
    spark.sql("OPTIMIZE dosage")   # cluster the written data by sample_id
    n_dose = spark.table("dosage").count()
    n_samp = spark.table("dosage").select("sample_id").distinct().count()
    vids = spark.table("dosage").select("variant_id").distinct()
    n_vid = vids.count()
    print(f"dosage: {n_dose:,} rows | {n_samp} samples × {n_vid:,} loci  (build {now()-t0:.1f}s)")

    # synthetic pgs_weights: each PGS weights ~density_pct% of loci, deterministic weight from hash
    wparts = []
    for k in range(n_pgs):
        w = (vids
             .where((F.abs(F.hash(F.concat(F.col("variant_id"), F.lit(f"|{k}")))) % 100) < density_pct)
             .withColumn("pgs_id", F.lit(f"BENCHPGS{k:03d}"))
             .withColumn("effect_allele", F.lit("A")).withColumn("other_allele", F.lit("G"))
             .withColumn("weight", ((F.abs(F.hash(F.concat(F.col("variant_id"), F.lit(f"w{k}")))) % 2000) - 1000) / 1000.0)
             .withColumn("weight_sha", F.lit(f"benchsha{k:03d}"))
             .select("pgs_id", "variant_id", "effect_allele", "other_allele", "weight", "weight_sha"))
        wparts.append(w)
    weights = wparts[0]
    for w in wparts[1:]:
        weights = weights.unionByName(w)
    weights.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable("pgs_weights")
    reg = (spark.table("pgs_weights").groupBy("pgs_id", "weight_sha").agg(F.count("*").alias("n_variants"))
           .withColumn("score_id", F.col("pgs_id")).withColumn("disease", F.lit("bench"))
           .withColumn("direction", F.lit("risk")).withColumn("body_system", F.lit("bench"))
           .withColumn("hr_per_sd", F.lit(1.5)).withColumn("clinical_model", F.lit(None).cast("string"))
           .withColumn("training_ancestries", F.lit("European")).withColumn("weight_path", F.lit("synthetic"))
           .withColumn("registered_at", F.current_timestamp()))
    reg.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable("pgs_registry")

    # panel ref (one superpop row per pgs) + sample_ancestry (all European for the bench)
    superpops = ["European", "African", "East_Asian", "South_Asian", "Hispanic_Latino", "Admixed"]
    (spark.table("pgs_registry").select("pgs_id", "weight_sha")
        .crossJoin(spark.createDataFrame([(s,) for s in superpops], ["superpop"]))
        .withColumn("mean", F.lit(0.0)).withColumn("sd", F.lit(1.0))
        .withColumn("quantiles", F.lit(None).cast("array<double>")).withColumn("n_panel", F.lit(3202))
        .withColumn("panel_version", F.lit(PANEL_VERSION))
        .select("pgs_id", "superpop", "mean", "sd", "quantiles", "n_panel", "panel_version", "weight_sha")
     ).write.mode("overwrite").option("overwriteSchema", "true").saveAsTable("pgs_panel_ref")
    (spark.table("dosage").select("sample_id").distinct().withColumn("most_similar_pop", F.lit("European"))
     ).write.mode("overwrite").option("overwriteSchema", "true").saveAsTable("sample_ancestry")

    n_w = spark.table("pgs_weights").count()
    print(f"pgs_weights: {n_w:,} rows across {n_pgs} PGS (~{density_pct}% density) | prep total {now()-t0:.1f}s")
    print("PREP done. Now run mode=score at each num_workers rung.")

# COMMAND ----------

# MAGIC %md
# MAGIC ### The scorer — reconcile plan + join-aggregate + normalize + MERGE (identical to 02/01), TIMED
# MAGIC Factored so each benchmark phase (full / add_pgs / add_sample / rerun) times the same code path.

# COMMAND ----------

def reconcile(sample_filter=None, pgs_filter=None):
    registry = spark.table("pgs_registry").select("pgs_id", "weight_sha")
    if pgs_filter: registry = registry.where(F.col("pgs_id").isin(pgs_filter))
    samples = spark.table("dosage").select("sample_id").distinct()
    if sample_filter: samples = samples.where(F.col("sample_id").isin(sample_filter))
    desired = samples.crossJoin(registry).withColumn("panel_version", F.lit(PANEL_VERSION))
    existing = spark.table("prs_scores").select("sample_id", "pgs_id",
                 F.col("weight_sha").alias("e_sha"), F.col("panel_version").alias("e_pv"))
    plan = (desired.join(existing, ["sample_id", "pgs_id"], "left")
            .where((F.col("e_sha").isNull()) | (F.col("e_sha") != F.col("weight_sha"))
                   | (F.col("e_pv").isNull()) | (F.col("e_pv") != F.col("panel_version")))
            .select("sample_id", "pgs_id", "weight_sha", "panel_version"))
    plan.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable("_prs_reconcile_plan")
    return spark.table("_prs_reconcile_plan")

def score(plan):
    planned_samples = plan.select("sample_id").distinct()
    planned_pgs = plan.select("pgs_id", "weight_sha").distinct()
    # incremental fast path (mirrors 01_score_prs): small plan → predicate → Liquid-Clustering file-skip
    SAMPLE_PREDICATE_MAX = 200
    _ids = [r["sample_id"] for r in planned_samples.limit(SAMPLE_PREDICATE_MAX + 1).collect()]
    if 0 < len(_ids) <= SAMPLE_PREDICATE_MAX:
        dose = spark.table("dosage").where(F.col("sample_id").isin(_ids))
    else:
        dose = spark.table("dosage").join(F.broadcast(planned_samples), "sample_id")
    wts = spark.table("pgs_weights").join(F.broadcast(planned_pgs), ["pgs_id", "weight_sha"])
    raw = (dose.join(wts, "variant_id").groupBy("sample_id", "pgs_id", "weight_sha")
           .agg(F.sum(F.col("dose") * F.col("weight")).alias("raw_score"),
                F.count(F.lit(1)).alias("n_variants_matched"))
           .join(plan, ["sample_id", "pgs_id", "weight_sha"]))
    nvar = spark.table("pgs_registry").select("pgs_id", F.col("n_variants").alias("pgs_nvar"))
    raw = (raw.join(nvar, "pgs_id", "left")
           .withColumn("coverage_pct", F.when(F.col("pgs_nvar") > 0, 100.0 * F.col("n_variants_matched") / F.col("pgs_nvar")))
           .withColumn("small_score", F.col("n_variants_matched") < F.lit(1000)))
    anc = spark.table("sample_ancestry").select("sample_id", F.col("most_similar_pop").alias("msp"))
    ref = spark.table("pgs_panel_ref").select("pgs_id", F.col("superpop").alias("msp"), "panel_version",
             F.col("mean").alias("ref_mean"), F.col("sd").alias("ref_sd"))
    scored = (raw.join(anc, "sample_id", "left").join(ref, ["pgs_id", "msp", "panel_version"], "left")
              .withColumn("z_msp", F.when(F.col("ref_sd") > 0, (F.col("raw_score") - F.col("ref_mean")) / F.col("ref_sd"))))
    out = (scored.withColumn("used_ancestry", F.col("msp"))
           .withColumn("percentile_msp", F.lit(None).cast("double"))
           .withColumn("z_admixed", F.lit(None).cast("double")).withColumn("percentile_admixed", F.lit(None).cast("double"))
           .withColumn("integrated_z_source", F.lit("msp")).withColumn("integrated_risk_10yr", F.lit(None).cast("double"))
           .withColumn("clinical_risk_10yr", F.lit(None).cast("double")).withColumn("risk_category", F.lit(None).cast("string"))
           .withColumn("concordance_verdict", F.lit(None).cast("string")).withColumn("computed_at", F.current_timestamp())
           .select("sample_id", "pgs_id", "weight_sha", "panel_version", "raw_score", "n_variants_matched",
                   "coverage_pct", "small_score", F.col("msp").alias("most_similar_pop"), "used_ancestry",
                   "z_msp", "percentile_msp", "z_admixed", "percentile_admixed", "integrated_z_source",
                   "integrated_risk_10yr", "clinical_risk_10yr", "risk_category", "concordance_verdict", "computed_at"))
    out.createOrReplaceTempView("_scored")
    tgt = DeltaTable.forName(spark, f"{catalog}.{schema}.prs_scores")
    (tgt.alias("t").merge(spark.table("_scored").alias("s"), "t.sample_id = s.sample_id AND t.pgs_id = s.pgs_id")
       .whenMatchedUpdateAll().whenNotMatchedInsertAll().execute())

RESULTS = []   # collected per-phase metrics; returned via notebook.exit for machine-readable retrieval

def run_phase(phase, mechanism="sql", sample_filter=None, pgs_filter=None):
    n_exec, cores = cluster_shape()
    t0 = now()
    plan = reconcile(sample_filter=sample_filter, pgs_filter=pgs_filter)
    n_cells = plan.count()
    if n_cells > 0:
        score(plan)
    wall = now() - t0
    n_dose = spark.table("dosage").count(); n_w = spark.table("pgs_weights").count()
    nodes = 1 + n_exec
    dbu = nodes * dbu_per_node_hr * (wall / 3600.0)
    cost = dbu * dollar_per_dbu
    cps = (n_cells / wall) if wall > 0 else 0.0
    dpc = (dbu / n_cells) if n_cells > 0 else 0.0
    row = [(phase, mechanism, n_exec, cores, n_cells, n_dose, n_w, wall, cps, dbu, cost, dpc)]
    (spark.createDataFrame(row, "phase string, mechanism string, n_exec int, cores int, n_cells long, "
                           "n_dosage_rows long, n_weight_rows long, wall_clock_s double, cells_per_s double, "
                           "est_dbu double, est_cost_usd double, dbu_per_cell double")
        .withColumn("run_ts", F.current_timestamp())
        .select("run_ts", "phase", "mechanism", "n_exec", "cores", "n_cells", "n_dosage_rows",
                "n_weight_rows", "wall_clock_s", "cells_per_s", "est_dbu", "est_cost_usd", "dbu_per_cell")
     ).write.mode("append").saveAsTable("stage0_metrics")
    RESULTS.append({"phase": phase, "n_exec": n_exec, "cores": cores, "n_cells": n_cells,
                    "n_dosage_rows": n_dose, "n_weight_rows": n_w, "wall_clock_s": round(wall, 2),
                    "cells_per_s": round(cps, 1), "est_dbu": round(dbu, 4),
                    "est_cost_usd": round(cost, 4), "dbu_per_cell": dpc})
    print(f"[{phase}] cells={n_cells:,} wall={wall:.1f}s cells/s={cps:,.0f} "
          f"nodes={nodes} est_cost=${cost:.4f} dbu/cell={dpc:.3e}")
    return wall, n_cells

# COMMAND ----------

# MAGIC %md ### RUN the benchmark phases (mode=score or full)

# COMMAND ----------

if mode in ("score", "full"):
    # Reset to the PREP baseline so every num_workers rung scores the identical grid (the add_pgs /
    # add_sample phases below mutate the bench; undo any prior run's increments first).
    base_ids = ", ".join(f"'BENCHPGS{k:03d}'" for k in range(n_pgs))
    spark.sql("DELETE FROM dosage WHERE sample_id = 'BENCH_NEW_SAMPLE'")
    spark.sql("DELETE FROM sample_ancestry WHERE sample_id = 'BENCH_NEW_SAMPLE'")
    spark.sql(f"DELETE FROM pgs_weights WHERE pgs_id NOT IN ({base_ids})")
    spark.sql(f"DELETE FROM pgs_registry WHERE pgs_id NOT IN ({base_ids})")
    spark.sql(f"DELETE FROM pgs_panel_ref WHERE pgs_id NOT IN ({base_ids})")
    spark.sql("TRUNCATE TABLE prs_scores")   # fresh backfill each score run so wall-clock is comparable

    # 1) FULL backfill: all samples × all PGS (the big matmul, once)
    run_phase("full_backfill")

    # 2) INCREMENTAL add-PGS: register one more synthetic column, score only it (~1 column of cells)
    vids = spark.table("dosage").select("variant_id").distinct()
    newk = spark.table("pgs_registry").count()
    neww = (vids.where((F.abs(F.hash(F.concat(F.col("variant_id"), F.lit(f"|{newk}")))) % 100) < density_pct)
            .withColumn("pgs_id", F.lit(f"BENCHPGS{newk:03d}")).withColumn("effect_allele", F.lit("A"))
            .withColumn("other_allele", F.lit("G"))
            .withColumn("weight", ((F.abs(F.hash(F.concat(F.col("variant_id"), F.lit(f"w{newk}")))) % 2000) - 1000) / 1000.0)
            .withColumn("weight_sha", F.lit(f"benchsha{newk:03d}"))
            .select("pgs_id", "variant_id", "effect_allele", "other_allele", "weight", "weight_sha"))
    neww.write.mode("append").saveAsTable("pgs_weights")
    (neww.groupBy("pgs_id", "weight_sha").agg(F.count("*").alias("n_variants"))
        .withColumn("score_id", F.col("pgs_id")).withColumn("disease", F.lit("bench")).withColumn("direction", F.lit("risk"))
        .withColumn("body_system", F.lit("bench")).withColumn("hr_per_sd", F.lit(1.5)).withColumn("clinical_model", F.lit(None).cast("string"))
        .withColumn("training_ancestries", F.lit("European")).withColumn("weight_path", F.lit("synthetic")).withColumn("registered_at", F.current_timestamp())
        .select("pgs_id","score_id","disease","direction","body_system","hr_per_sd","clinical_model","training_ancestries","weight_sha","n_variants","weight_path","registered_at")
     ).write.mode("append").saveAsTable("pgs_registry")
    (spark.table("pgs_registry").where(F.col("pgs_id") == f"BENCHPGS{newk:03d}").select("pgs_id","weight_sha")
        .crossJoin(spark.createDataFrame([("European",)], ["superpop"]))
        .withColumn("mean", F.lit(0.0)).withColumn("sd", F.lit(1.0)).withColumn("quantiles", F.lit(None).cast("array<double>"))
        .withColumn("n_panel", F.lit(3202)).withColumn("panel_version", F.lit(PANEL_VERSION))
        .select("pgs_id","superpop","mean","sd","quantiles","n_panel","panel_version","weight_sha")
     ).write.mode("append").saveAsTable("pgs_panel_ref")
    run_phase("add_pgs")

    # 3) INCREMENTAL add-sample: clone one sample's dosage under a new id, score only it (~1 row of cells)
    victim = spark.table("dosage").select("sample_id").limit(1).collect()[0]["sample_id"]
    (spark.table("dosage").where(F.col("sample_id") == victim)
        .withColumn("sample_id", F.lit("BENCH_NEW_SAMPLE"))).write.mode("append").saveAsTable("dosage")
    spark.sql("INSERT INTO sample_ancestry VALUES ('BENCH_NEW_SAMPLE','European')")
    run_phase("add_sample")

    # 4) RE-RUN: reconcile should find 0 cells (idempotent → no work)
    run_phase("rerun_idempotent")

    print("\n=== stage0_metrics (this + prior runs) ===")
    spark.table("stage0_metrics").orderBy("run_ts").show(200, truncate=False)

    import json as _json
    dbutils.notebook.exit(_json.dumps({"max_loci": max_loci, "n_pgs": n_pgs, "density_pct": density_pct,
                                       "phases": RESULTS}))
