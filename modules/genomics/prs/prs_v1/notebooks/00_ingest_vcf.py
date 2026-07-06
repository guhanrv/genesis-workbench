# Databricks notebook source
# MAGIC %md
# MAGIC # PRS step 1 — VCF → dosage Delta (Glow ingest)
# MAGIC
# MAGIC The **only** Glow / classic-cluster step. Glow is a JVM Spark extension and
# MAGIC cannot run on serverless, so it is isolated here: read the VCF, derive a
# MAGIC per-sample **alt-allele dosage** array, and write a plain Delta table the
# MAGIC serverless scoring step consumes.
# MAGIC
# MAGIC **Dosage source (important for imputed cohorts).** Imputed genotypes are often
# MAGIC dosage-only — `DS` (diploid dosage) or `HDS` (haploid dosage pair) — with **no
# MAGIC hard `GT`**, which `glow.genotype_states` cannot read. We therefore pick, in
# MAGIC order (or per the `dosage_field` param): **DS → HDS(summed) → GT** (hard calls
# MAGIC via `genotype_states`). Missing values are normalized to `null`. Dosage is
# MAGIC continuous (0–2) and is the more accurate input for scoring anyway.

# COMMAND ----------

dbutils.widgets.text("catalog", "genesis_workbench", "Catalog")
dbutils.widgets.text("schema", "genesis_schema", "Schema")
dbutils.widgets.text("vcf_path", "", "VCF file path (cohort to score)")
dbutils.widgets.text("dosage_field", "auto", "Dosage source: auto | DS | HDS | GT")
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
dosage_field = dbutils.widgets.get("dosage_field")
mlflow_run_id = dbutils.widgets.get("mlflow_run_id")

# COMMAND ----------

import glow
import pyspark.sql.functions as F

spark = glow.register(spark)

variants = spark.read.format("vcf").load(vcf_path).where(F.size("alternateAlleles") == 1)  # biallelic

# --- choose the per-sample dosage source ------------------------------------
# Prefer dosage (DS/HDS) when present — imputed cohorts often lack hard GT.
gt_elem = variants.schema["genotypes"].dataType.elementType
avail = {f.name.lower(): f.name for f in gt_elem.fields}
pref = (dosage_field or "auto").strip().lower()

def _states_expr():
    if pref == "ds" or (pref == "auto" and "ds" in avail):
        return "DS", F.expr(f"transform(genotypes, g -> cast(g.`{avail['ds']}` as double))")
    if pref == "hds" or (pref == "auto" and "hds" in avail):
        f = avail["hds"]
        return "HDS", F.expr(
            f"transform(genotypes, g -> case when g.`{f}` is null then null "
            f"else aggregate(g.`{f}`, cast(0.0 as double), (a, x) -> a + cast(x as double)) end)"
        )
    # GT hard calls via Glow; genotype_states returns 0/1/2 (-1 = missing) → null
    return "GT", F.expr(
        "transform(genotype_states(genotypes), x -> case when x < 0 then null else cast(x as double) end)"
    )

kind, states = _states_expr()
print(f"dosage source: {kind}")

# `states` is a per-sample alt-allele dosage array (double; null = missing),
# aligned element-wise to `genotypes.sampleId`. Glow's `start` is 0-based → +1.
dosage = variants.select(
    F.regexp_replace(F.col("contigName"), "chr", "").alias("chrom"),
    (F.col("start") + 1).alias("pos"),
    F.col("referenceAllele").alias("ref"),
    F.col("alternateAlleles")[0].alias("alt"),
    states.alias("states"),
    F.col("genotypes.sampleId").alias("sample_ids"),
)

dosage_table = f"{catalog}.{schema}.prs_dosage_{mlflow_run_id.replace('-', '_')}"
dosage.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(dosage_table)
print(f"Wrote {spark.table(dosage_table).count()} biallelic variants ({kind} dosage) → {dosage_table}")
