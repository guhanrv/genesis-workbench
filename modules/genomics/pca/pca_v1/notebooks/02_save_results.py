# Databricks notebook source
# MAGIC %md
# MAGIC # Save PCA Results and Update MLflow
# MAGIC
# MAGIC Reads the per-sample principal-component Delta table and records summary
# MAGIC metrics on the MLflow run.

# COMMAND ----------

dbutils.widgets.text("catalog", "genesis_workbench", "Catalog")
dbutils.widgets.text("schema", "genesis_schema", "Schema")
dbutils.widgets.text("mlflow_run_id", "", "MLflow Run ID")
dbutils.widgets.text("user_email", "a@b.com", "User Email")

catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")
mlflow_run_id = dbutils.widgets.get("mlflow_run_id")

# COMMAND ----------

# MAGIC %pip install mlflow==2.22.0

# COMMAND ----------

results_table = f"pca_components_{mlflow_run_id.replace('-', '_')}"
df = spark.table(f"{catalog}.{schema}.{results_table}")

n_samples = df.count()
n_pcs = len([c for c in df.columns if c.startswith("PC")])
print(f"PCA: {n_samples} samples × {n_pcs} components")

# COMMAND ----------

import mlflow

mlflow.set_registry_uri("databricks-uc")
mlflow.set_tracking_uri("databricks")

with mlflow.start_run(run_id=mlflow_run_id):
    mlflow.log_param("results_table", f"{catalog}.{schema}.{results_table}")
    mlflow.log_metric("n_samples", n_samples)
    mlflow.log_metric("n_components", n_pcs)
    mlflow.set_tag("job_status", "pca_complete")

print("PCA complete — MLflow run updated")
