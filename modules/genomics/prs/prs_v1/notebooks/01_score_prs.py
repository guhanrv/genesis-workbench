# Databricks notebook source
# MAGIC %md
# MAGIC # PRS step 2 — score against a PGS Catalog file (serverless)
# MAGIC
# MAGIC Runs entirely on **serverless** — no Glow. Reads the dosage Delta table
# MAGIC produced by `00_ingest_vcf` and scores every sample against a
# MAGIC [PGS Catalog](https://www.pgscatalog.org/) file:
# MAGIC
# MAGIC ```
# MAGIC PRS(sample) = Σ_variants  dosage_of_effect_allele(sample) × effect_weight
# MAGIC ```
# MAGIC
# MAGIC Join on `(chrom, pos)`, orient dosage to the effect allele
# MAGIC (`effect==alt → dosage`, `effect==ref → 2−dosage`; mismatches/missing dropped),
# MAGIC sum per sample, standardize within the cohort (z-score + percentile).

# COMMAND ----------

dbutils.widgets.text("catalog", "genesis_workbench", "Catalog")
dbutils.widgets.text("schema", "genesis_schema", "Schema")
dbutils.widgets.text("scorefile_path", "", "PGS Catalog scoring file (.txt/.txt.gz, harmonized)")
dbutils.widgets.text("pgs_id", "", "PGS Catalog score id (e.g. PGS000004)")
dbutils.widgets.text("mlflow_run_id", "", "MLflow Run ID")

catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")
scorefile_path = dbutils.widgets.get("scorefile_path")
pgs_id = dbutils.widgets.get("pgs_id")
mlflow_run_id = dbutils.widgets.get("mlflow_run_id")

# COMMAND ----------

import pyspark.sql.functions as F
from pyspark.sql import Window

# COMMAND ----------

# MAGIC %md
# MAGIC ### 1. Dosage (from the Glow ingest step) + scoring file

# COMMAND ----------

dosage = spark.table(f"{catalog}.{schema}.prs_dosage_{mlflow_run_id.replace('-', '_')}")

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

print(f"Scoring file {pgs_id}: {score.count()} weighted variants")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2. Join + orient dosage to the effect allele, sum per sample

# COMMAND ----------

joined = dosage.join(score, on=["chrom", "pos"], how="inner")

per_sample = (
    joined.select(
        F.explode(F.arrays_zip("sample_ids", "states")).alias("z"),
        "effect_allele", "ref", "alt", "weight",
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
# MAGIC ### 3. Standardize within the cohort and persist

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
print(f"Scored {prs.count()} samples against {pgs_id} → {catalog}.{schema}.{results_table}")
print(f"cohort mean(raw)={mu:.4f} sd(raw)={sd:.4f}")
