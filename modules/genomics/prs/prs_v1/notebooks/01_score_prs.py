# Databricks notebook source
# MAGIC %md
# MAGIC # Polygenic Risk Score (PRS) scoring with Glow
# MAGIC
# MAGIC Scores every sample in a VCF against a [PGS Catalog](https://www.pgscatalog.org/)
# MAGIC scoring file:
# MAGIC
# MAGIC ```
# MAGIC PRS(sample) = Σ_variants  dosage_of_effect_allele(sample) × effect_weight
# MAGIC ```
# MAGIC
# MAGIC Spark/Glow-native — mirrors the GWAS submodule's modality (no PLINK / pgsc_calc):
# MAGIC 1. Read VCF as a Glow DataFrame, derive per-sample dosage via `glow.genotype_states`.
# MAGIC 2. Join variants to the scoring file on (chrom, pos), orienting dosage to the effect allele.
# MAGIC 3. Sum `dosage × weight` per sample, then standardize within the cohort (z-score + percentile).

# COMMAND ----------

dbutils.widgets.text("catalog", "genesis_workbench", "Catalog")
dbutils.widgets.text("schema", "genesis_schema", "Schema")
dbutils.widgets.text("sql_warehouse_id", "w123", "SQL Warehouse Id")
dbutils.widgets.text("vcf_path", "", "VCF file path (cohort to score)")
dbutils.widgets.text("scorefile_path", "", "PGS Catalog scoring file (.txt/.txt.gz, harmonized)")
dbutils.widgets.text("pgs_id", "", "PGS Catalog score id (e.g. PGS000004)")
dbutils.widgets.text("mlflow_run_id", "", "MLflow Run ID")
dbutils.widgets.text("user_email", "a@b.com", "User Email")

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
scorefile_path = dbutils.widgets.get("scorefile_path")
pgs_id = dbutils.widgets.get("pgs_id")
mlflow_run_id = dbutils.widgets.get("mlflow_run_id")

# COMMAND ----------

import glow
import pyspark.sql.functions as F
from pyspark.sql import Window

spark = glow.register(spark)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 1. Read VCF → per-variant, per-sample dosage
# MAGIC `glow.genotype_states` returns the alternate-allele dosage (0/1/2; -1 = missing),
# MAGIC aligned element-wise to `genotypes.sampleId`. Glow's `start` is 0-based, so the
# MAGIC 1-based VCF/PGS position is `start + 1`.

# COMMAND ----------

variants = spark.read.format("vcf").load(vcf_path)
dosage = variants.select(
    F.regexp_replace(F.col("contigName"), "chr", "").alias("chrom"),
    (F.col("start") + 1).alias("pos"),
    F.col("referenceAllele").alias("ref"),
    F.col("alternateAlleles")[0].alias("alt"),
    glow.genotype_states(F.col("genotypes")).alias("states"),
    F.col("genotypes.sampleId").alias("sample_ids"),
).where(F.size("alternateAlleles") == 1)  # biallelic sites only

raw_table = f"prs_raw_vcf_{mlflow_run_id.replace('-', '_')}"
dosage.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(f"{catalog}.{schema}.{raw_table}")
dosage = spark.table(f"{catalog}.{schema}.{raw_table}")
print(f"Loaded {dosage.count()} biallelic variants from {vcf_path}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2. Parse the PGS Catalog scoring file
# MAGIC PGS Catalog files are tab-delimited with `#`-prefixed metadata. We use the
# MAGIC harmonized coordinates (`hm_chr`/`hm_pos`) when present, else the author-reported
# MAGIC `chr_name`/`chr_position`.

# COMMAND ----------

score_raw = (
    spark.read.option("sep", "\t").option("comment", "#").option("header", "true")
    .csv(scorefile_path)
)
cols = score_raw.columns
chr_col = "hm_chr" if "hm_chr" in cols else "chr_name"
pos_col = "hm_pos" if "hm_pos" in cols else "chr_position"
other_col = "other_allele" if "other_allele" in cols else "reference_allele"

score = score_raw.select(
    F.regexp_replace(F.col(chr_col).cast("string"), "chr", "").alias("chrom"),
    F.col(pos_col).cast("long").alias("pos"),
    F.col("effect_allele").alias("effect_allele"),
    F.col(other_col).alias("other_allele"),
    F.col("effect_weight").cast("double").alias("weight"),
).where(F.col("pos").isNotNull() & F.col("weight").isNotNull())

n_score = score.count()
print(f"Scoring file {pgs_id}: {n_score} weighted variants")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3. Join + orient dosage to the effect allele, sum per sample
# MAGIC If the effect allele is the ALT allele, the effect dosage is the alt dosage;
# MAGIC if it is the REF allele, it is `2 - alt_dosage`. Strand/allele mismatches and
# MAGIC missing genotypes (state < 0) are dropped.

# COMMAND ----------

joined = dosage.join(score, on=["chrom", "pos"], how="inner")

per_sample = (
    joined.select(
        F.explode(F.arrays_zip("sample_ids", "states")).alias("z"),
        "effect_allele", "other_allele", "ref", "alt", "weight",
    )
    .select(
        F.col("z.sample_ids").alias("sample_id"),
        F.when(F.col("z.states") < 0, F.lit(None).cast("double"))
        .when(F.col("effect_allele") == F.col("alt"), F.col("z.states").cast("double"))
        .when(F.col("effect_allele") == F.col("ref"), F.lit(2.0) - F.col("z.states").cast("double"))
        .otherwise(F.lit(None).cast("double")).alias("eff_dosage"),
        F.col("weight"),
    )
    .where(F.col("eff_dosage").isNotNull())
    .withColumn("contrib", F.col("eff_dosage") * F.col("weight"))
)

prs = per_sample.groupBy("sample_id").agg(
    F.sum("contrib").alias("prs_raw"),
    F.count("*").alias("n_variants_matched"),
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 4. Standardize within the cohort (z-score + percentile) and persist

# COMMAND ----------

stats = prs.agg(F.mean("prs_raw").alias("mu"), F.stddev_pop("prs_raw").alias("sd")).first()
mu = float(stats["mu"]) if stats["mu"] is not None else 0.0
sd = float(stats["sd"]) if stats["sd"] not in (None, 0.0) else 1.0

prs = (
    prs.withColumn("pgs_id", F.lit(pgs_id))
    .withColumn("prs_z", (F.col("prs_raw") - F.lit(mu)) / F.lit(sd))
    .withColumn("prs_percentile", F.round(F.percent_rank().over(Window.orderBy("prs_raw")) * 100, 2))
)

results_table = f"prs_scores_{mlflow_run_id.replace('-', '_')}"
prs.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(f"{catalog}.{schema}.{results_table}")

n_samples = prs.count()
print(f"Scored {n_samples} samples against {pgs_id} → {catalog}.{schema}.{results_table}")
print(f"cohort mean(raw)={mu:.4f} sd(raw)={sd:.4f}")
