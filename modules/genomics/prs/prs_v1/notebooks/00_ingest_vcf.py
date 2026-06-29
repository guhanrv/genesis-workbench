# Databricks notebook source
# MAGIC %md
# MAGIC # PRS step 1 — VCF → dosage Delta (Glow ingest)
# MAGIC
# MAGIC The **only** Glow / classic-cluster step. Glow is a JVM Spark extension and
# MAGIC cannot run on serverless, so we isolate it here: read the VCF, derive
# MAGIC per-sample alt-allele dosage (`glow.genotype_states`), and write a plain
# MAGIC Delta table. The downstream scoring step reads that table and runs entirely
# MAGIC on serverless (no Glow).

# COMMAND ----------

dbutils.widgets.text("catalog", "genesis_workbench", "Catalog")
dbutils.widgets.text("schema", "genesis_schema", "Schema")
dbutils.widgets.text("vcf_path", "", "VCF file path (cohort to score)")
dbutils.widgets.text("mlflow_run_id", "", "MLflow Run ID")

catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")

# COMMAND ----------

glow_whl_path = None
for lib in dbutils.fs.ls(f"/Volumes/{catalog}/{schema}/libraries"):
    if lib.name.startswith("glow") and lib.name.endswith(".whl"):
        glow_whl_path = lib.path.replace("dbfs:", "")
print(f"Glow wheel: {glow_whl_path}")

# COMMAND ----------

# MAGIC %pip install {glow_whl_path} --force-reinstall
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")
vcf_path = dbutils.widgets.get("vcf_path")
mlflow_run_id = dbutils.widgets.get("mlflow_run_id")

# COMMAND ----------

import glow
import pyspark.sql.functions as F

spark = glow.register(spark)

# `glow.genotype_states` returns alt-allele dosage (0/1/2; -1 = missing), aligned
# element-wise to `genotypes.sampleId`. Glow's `start` is 0-based → 1-based pos is start+1.
dosage = (
    spark.read.format("vcf").load(vcf_path)
    .where(F.size("alternateAlleles") == 1)  # biallelic sites only
    .select(
        F.regexp_replace(F.col("contigName"), "chr", "").alias("chrom"),
        (F.col("start") + 1).alias("pos"),
        F.col("referenceAllele").alias("ref"),
        F.col("alternateAlleles")[0].alias("alt"),
        glow.genotype_states(F.col("genotypes")).alias("states"),
        F.col("genotypes.sampleId").alias("sample_ids"),
    )
)

dosage_table = f"{catalog}.{schema}.prs_dosage_{mlflow_run_id.replace('-', '_')}"
dosage.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(dosage_table)
print(f"Wrote {spark.table(dosage_table).count()} biallelic variants → {dosage_table}")
