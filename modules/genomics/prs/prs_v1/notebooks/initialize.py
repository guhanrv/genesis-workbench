# Databricks notebook source

# COMMAND ----------

dbutils.widgets.text("catalog", "genesis_workbench", "Catalog")
dbutils.widgets.text("schema", "genesis_schema", "Schema")
dbutils.widgets.text("prs_scoring_job_id", "1234", "PRS Scoring Job ID")
dbutils.widgets.text("user_email", "a@b.com", "Email of the user running the deploy")
dbutils.widgets.text("sql_warehouse_id", "8f210e00850a2c16", "SQL Warehouse Id")
dbutils.widgets.text("example_pgs_id", "PGS000004", "Example PGS Catalog id")
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
prs_scoring_job_id = dbutils.widgets.get("prs_scoring_job_id")
user_email = dbutils.widgets.get("user_email")
sql_warehouse_id = dbutils.widgets.get("sql_warehouse_id")
pgs_id = dbutils.widgets.get("example_pgs_id")

# COMMAND ----------

from genesis_workbench.workbench import initialize

databricks_token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().getOrElse(None)
initialize(core_catalog_name=catalog, core_schema_name=schema, sql_warehouse_id=sql_warehouse_id, token=databricks_token)

# COMMAND ----------

spark.sql(f"USE CATALOG {catalog}")
spark.sql(f"USE SCHEMA {schema}")

# COMMAND ----------

example_scorefile = f"/Volumes/{catalog}/{schema}/prs_reference/scorefiles/{pgs_id}_hmPOS_GRCh38.txt.gz"
# Reuse the GWAS sample VCF as a runnable example cohort to score.
example_vcf = f"/Volumes/{catalog}/{schema}/gwas_data/sample_vcf/ALL.chr6.shapeit2_integrated_snvindels_v2a_27022019.GRCh38.phased.vcf.gz"

query = f"""
    MERGE INTO settings AS target
    USING (
        SELECT * FROM VALUES
            ('prs_scoring_job_id', '{prs_scoring_job_id}', 'genomics'),
            ('prs_sample_scorefile_path', '{example_scorefile}', 'genomics'),
            ('prs_sample_vcf_path', '{example_vcf}', 'genomics'),
            ('prs_example_pgs_id', '{pgs_id}', 'genomics')
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
set_app_permissions_for_job(job_id=prs_scoring_job_id, user_email=user_email)

# COMMAND ----------

# Register the PRS workflow as a batch model so it appears in the Deployed Models tab
from genesis_workbench.models import register_batch_model

register_batch_model(
    model_name="prs",
    model_display_name="Polygenic Risk Scoring",
    model_description="Glow/Spark PGS Catalog scoring — per-sample polygenic risk scores from a VCF and a PGS Catalog scoring file",
    model_category="genomics",
    module="genomics",
    job_id=prs_scoring_job_id,
    job_name="prs_scoring",
    cluster_type="CPU",
    added_by=user_email,
)

print("Genomics PRS module initialization complete")
