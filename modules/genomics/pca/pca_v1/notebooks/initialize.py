# Databricks notebook source

# COMMAND ----------

dbutils.widgets.text("catalog", "genesis_workbench", "Catalog")
dbutils.widgets.text("schema", "genesis_schema", "Schema")
dbutils.widgets.text("pca_compute_job_id", "1234", "PCA Compute Job ID")
dbutils.widgets.text("user_email", "a@b.com", "Email of the user running the deploy")
dbutils.widgets.text("sql_warehouse_id", "8f210e00850a2c16", "SQL Warehouse Id")
dbutils.widgets.text("databricks_app_names", "genesis-workbench:mcp-genesis-workbench", "Databricks App Names (colon/comma-separated, UI + MCP)")

catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")

# COMMAND ----------

# MAGIC %pip install databricks-sdk==0.50.0 databricks-sql-connector==4.0.3 mlflow==2.22.0

# COMMAND ----------

gwb_library_path = None
for lib in dbutils.fs.ls(f"/Volumes/{catalog}/{schema}/libraries"):
    if lib.name.startswith("genesis_workbench"):
        gwb_library_path = lib.path.replace("dbfs:", "")
print(gwb_library_path)

# COMMAND ----------

# MAGIC %pip install {gwb_library_path} --force-reinstall
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")
pca_compute_job_id = dbutils.widgets.get("pca_compute_job_id")
user_email = dbutils.widgets.get("user_email")
sql_warehouse_id = dbutils.widgets.get("sql_warehouse_id")

# COMMAND ----------

from genesis_workbench.workbench import initialize

databricks_token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().getOrElse(None)
initialize(core_catalog_name=catalog, core_schema_name=schema, sql_warehouse_id=sql_warehouse_id, token=databricks_token)

# COMMAND ----------

spark.sql(f"USE CATALOG {catalog}")
spark.sql(f"USE SCHEMA {schema}")

# COMMAND ----------

# Reuse the GWAS sample VCF as a runnable example cohort for PCA.
example_vcf = f"/Volumes/{catalog}/{schema}/gwas_data/sample_vcf/ALL.chr6.shapeit2_integrated_snvindels_v2a_27022019.GRCh38.phased.vcf.gz"

query = f"""
    MERGE INTO settings AS target
    USING (
        SELECT * FROM VALUES
            ('pca_compute_job_id', '{pca_compute_job_id}', 'genomics'),
            ('pca_sample_vcf_path', '{example_vcf}', 'genomics')
        AS src(key, value, module)
    ) AS source
    ON target.key = source.key AND target.module = source.module
    WHEN MATCHED THEN UPDATE SET target.value = source.value
    WHEN NOT MATCHED THEN INSERT (key, value, module) VALUES (source.key, source.value, source.module)
"""
spark.sql(query)

# COMMAND ----------

from genesis_workbench.workbench import set_app_permissions_for_job
import os

_app_names_raw = dbutils.widgets.get("databricks_app_names")
os.environ["DATABRICKS_APP_NAMES"] = ",".join(
    [n.strip() for n in _app_names_raw.replace(":", ",").split(",") if n.strip()]
)  # UI + MCP
set_app_permissions_for_job(job_id=pca_compute_job_id, user_email=user_email)

# COMMAND ----------

# Register the PCA workflow as a batch model so it appears in the Deployed Models tab
from genesis_workbench.models import register_batch_model

register_batch_model(
    model_name="pca",
    model_display_name="Ancestry / Population-structure PCA",
    model_description="Glow/Spark per-sample principal components from a cohort VCF — population-structure covariates for GWAS and a basis for ancestry analysis",
    model_category="genomics",
    module="genomics",
    job_id=pca_compute_job_id,
    job_name="pca_compute",
    cluster_type="CPU",
    added_by=user_email,
)

print("Genomics PCA module initialization complete")
