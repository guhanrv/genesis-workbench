# Databricks notebook source
# MAGIC %md
# MAGIC # Publish PRS results + log summary metrics to MLflow
# MAGIC
# MAGIC Terminal step of the scoring DAG. Reads the fixed `prs_scores` cell-store (written by
# MAGIC `05_score_prs`) and logs summary metrics on the MLflow run — coverage, how many cells carry a
# MAGIC reference-panel `z_msp` / admixed `z_admixed`, and the raw-score distribution. Mirrors
# MAGIC `pca_v1/06_save_results`. NOTE: because scoring is incremental (MERGE-upserted), these are
# MAGIC **store-wide totals** over the whole `prs_scores` table (all runs to date), NOT just the cells
# MAGIC this run computed — the metric keys are `store_*` to make that explicit. (Per-run wide result
# MAGIC tables are gone — results live in `prs_scores`.)

# COMMAND ----------

dbutils.widgets.text("catalog", "genesis_workbench", "Catalog")
dbutils.widgets.text("schema", "genesis_schema", "Schema")
dbutils.widgets.text("pgs_ids", "", "Restrict summary to these PGS (comma-sep; empty = all in prs_scores)")
dbutils.widgets.text("mlflow_run_id", "", "MLflow Run ID")
dbutils.widgets.text("user_email", "a@b.com", "User Email")

catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")
pgs_filter = [x.strip() for x in dbutils.widgets.get("pgs_ids").split(",") if x.strip()]
mlflow_run_id = dbutils.widgets.get("mlflow_run_id")

# COMMAND ----------

# MAGIC %pip install mlflow==2.22.0

# COMMAND ----------

import pyspark.sql.functions as F

df = spark.table(f"{catalog}.{schema}.prs_scores")
if pgs_filter:
    df = df.where(F.col("pgs_id").isin(pgs_filter))

agg = df.agg(
    F.count("*").alias("n_cells"),
    F.countDistinct("sample_id").alias("n_samples"),
    F.countDistinct("pgs_id").alias("n_pgs"),
    F.mean("raw_score").alias("mean_raw"),
    F.stddev_pop("raw_score").alias("sd_raw"),
    F.mean("coverage_pct").alias("mean_coverage_pct"),
    F.mean(F.col("small_score").cast("double")).alias("frac_small_score"),
    F.sum(F.col("z_msp").isNotNull().cast("int")).alias("n_z_msp"),
    F.sum(F.col("z_admixed").isNotNull().cast("int")).alias("n_z_admixed"),
).first()

print(f"prs_scores: {agg['n_cells']} cells · {agg['n_samples']} samples × {agg['n_pgs']} PGS")
print(f"raw: mean={agg['mean_raw']}, sd={agg['sd_raw']} | mean coverage={agg['mean_coverage_pct']}%")
print(f"normalized: z_msp on {agg['n_z_msp']} cells, z_admixed on {agg['n_z_admixed']} cells | "
      f"small_score frac={agg['frac_small_score']}")

# COMMAND ----------

import mlflow

mlflow.set_registry_uri("databricks-uc")
mlflow.set_tracking_uri("databricks")

if mlflow_run_id.strip():
    with mlflow.start_run(run_id=mlflow_run_id):
        mlflow.log_param("results_table", f"{catalog}.{schema}.prs_scores")
        # store_* prefix: these are cumulative store totals, not this run's cells (incremental MERGE)
        for k in ("n_cells", "n_samples", "n_pgs", "n_z_msp", "n_z_admixed"):
            mlflow.log_metric(f"store_{k}", int(agg[k]))
        for k in ("mean_raw", "sd_raw", "mean_coverage_pct", "frac_small_score"):
            if agg[k] is not None:
                mlflow.log_metric(f"store_{k}", float(agg[k]))
        mlflow.set_tag("job_status", "prs_complete")
    print("PRS results published — MLflow run updated")
else:
    print("no mlflow_run_id — metrics printed only (manual run)")
