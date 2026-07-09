# Databricks notebook source
# MAGIC %md
# MAGIC ### Update MLflow Run Status
# MAGIC
# MAGIC Lightweight notebook called as a success/failure task in job workflows.
# MAGIC Updates the `job_status` tag on the MLflow run to reflect the final outcome.

# COMMAND ----------

dbutils.widgets.text("mlflow_run_id", "", "MLflow Run ID")
dbutils.widgets.text("job_status", "failed", "Job Status")

mlflow_run_id = dbutils.widgets.get("mlflow_run_id")
job_status = dbutils.widgets.get("job_status")

# No-op when no run id is supplied (e.g. a manual/ad-hoc job run not launched via the
# framework's batch-model invocation) — the mark tasks then simply do nothing.
if not mlflow_run_id.strip():
    dbutils.notebook.exit("no mlflow_run_id — skipping status update")

# COMMAND ----------

import mlflow

mlflow.set_registry_uri("databricks-uc")
mlflow.set_tracking_uri("databricks")

with mlflow.start_run(run_id=mlflow_run_id):
    mlflow.set_tag("job_status", job_status)

print(f"Updated MLflow run {mlflow_run_id}: job_status={job_status}")
