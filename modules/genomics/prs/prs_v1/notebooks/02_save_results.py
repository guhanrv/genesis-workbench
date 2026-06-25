# Databricks notebook source
# MAGIC %md
# MAGIC # Save PRS Results and Update MLflow
# MAGIC
# MAGIC Reads the per-sample PRS Delta table, computes cohort summary statistics,
# MAGIC and records them on the MLflow run.

# COMMAND ----------

dbutils.widgets.text("catalog", "genesis_workbench", "Catalog")
dbutils.widgets.text("schema", "genesis_schema", "Schema")
dbutils.widgets.text("pgs_id", "", "PGS id")
dbutils.widgets.text("mlflow_run_id", "", "MLflow Run ID")
dbutils.widgets.text("user_email", "a@b.com", "User Email")

catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")
pgs_id = dbutils.widgets.get("pgs_id")
mlflow_run_id = dbutils.widgets.get("mlflow_run_id")

# COMMAND ----------

# MAGIC %pip install mlflow==2.22.0

# COMMAND ----------

import pyspark.sql.functions as F

results_table = f"prs_scores_{mlflow_run_id.replace('-', '_')}"
df = spark.table(f"{catalog}.{schema}.{results_table}")

agg = df.agg(
    F.count("*").alias("n_samples"),
    F.mean("prs_raw").alias("mean_raw"),
    F.stddev_pop("prs_raw").alias("sd_raw"),
    F.mean("n_variants_matched").alias("mean_variants_matched"),
    F.min("n_variants_matched").alias("min_variants_matched"),
).first()

n_samples = agg["n_samples"]
print(f"PRS samples scored: {n_samples}")
print(f"mean raw={agg['mean_raw']}, sd raw={agg['sd_raw']}")
print(f"variants matched per sample: mean={agg['mean_variants_matched']}, min={agg['min_variants_matched']}")

# COMMAND ----------

import mlflow

mlflow.set_registry_uri("databricks-uc")
mlflow.set_tracking_uri("databricks")

with mlflow.start_run(run_id=mlflow_run_id):
    mlflow.log_param("pgs_id", pgs_id)
    mlflow.log_param("results_table", f"{catalog}.{schema}.{results_table}")
    mlflow.log_metric("n_samples_scored", n_samples)
    if agg["mean_raw"] is not None:
        mlflow.log_metric("mean_prs_raw", float(agg["mean_raw"]))
    if agg["mean_variants_matched"] is not None:
        mlflow.log_metric("mean_variants_matched", float(agg["mean_variants_matched"]))
    if agg["min_variants_matched"] is not None:
        mlflow.log_metric("min_variants_matched", float(agg["min_variants_matched"]))
    mlflow.set_tag("job_status", "prs_complete")

print("PRS scoring complete — MLflow run updated")
