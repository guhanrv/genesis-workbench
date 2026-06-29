# Databricks notebook source
# MAGIC %md
# MAGIC # PCA step 2 — population-structure PCA (serverless, distributed)
# MAGIC
# MAGIC Runs entirely on **serverless** — no Glow, no RDD API. Reads the dosage Delta
# MAGIC table from `00_ingest_vcf` and computes per-sample principal components (the
# MAGIC covariates a GWAS should adjust for — genesis's GWAS currently runs unadjusted).
# MAGIC
# MAGIC **Distributed by construction:** we orient the matrix as **variants-as-rows ×
# MAGIC samples-as-features**, so `spark.ml.PCA` builds an `N×N` (samples²) covariance —
# MAGIC small regardless of how many variants — distributed across the variant rows. The
# MAGIC per-sample coordinates are the principal components themselves (`model.pc`,
# MAGIC shape `numFeatures(N) × k`), so no `transform`/collect of a giant matrix is
# MAGIC needed. (The earlier samples-as-rows orientation would have decomposed a
# MAGIC variant×variant matrix on the driver — that's what we avoid.)

# COMMAND ----------

dbutils.widgets.text("catalog", "genesis_workbench", "Catalog")
dbutils.widgets.text("schema", "genesis_schema", "Schema")
dbutils.widgets.text("n_components", "10", "Number of principal components")
dbutils.widgets.text("maf_cutoff", "0.05", "Minor-allele-frequency cutoff")
dbutils.widgets.text("max_variants", "0", "Cap SNPs used (0 = all; M is unbounded here)")
dbutils.widgets.text("mlflow_run_id", "", "MLflow Run ID")

catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")
n_components = int(dbutils.widgets.get("n_components"))
maf_cutoff = float(dbutils.widgets.get("maf_cutoff"))
max_variants = int(dbutils.widgets.get("max_variants"))
mlflow_run_id = dbutils.widgets.get("mlflow_run_id")

# COMMAND ----------

import pyspark.sql.functions as F
from pyspark.ml.feature import PCA
from pyspark.ml.linalg import Vectors, VectorUDT

# COMMAND ----------

# MAGIC %md
# MAGIC ### 1. Dosage (from ingest) → keep biallelic common SNPs

# COMMAND ----------

dosage = spark.table(f"{catalog}.{schema}.pca_dosage_{mlflow_run_id.replace('-', '_')}")

# canonical sample order (identical across all variant rows)
sample_ids = list(dosage.select("sample_ids").first()["sample_ids"])
N = len(sample_ids)

freq = dosage.select(
    "states", "sample_ids",
    F.expr("aggregate(filter(states, x -> x >= 0), 0, (a, x) -> a + x)").alias("alt_sum"),
    F.expr("size(filter(states, x -> x >= 0))").alias("n_called"),
).withColumn("af", F.col("alt_sum") / (2 * F.col("n_called")))

common = freq.where((F.col("n_called") > 0) & (F.least(F.col("af"), 1 - F.col("af")) >= F.lit(maf_cutoff)))
if max_variants and max_variants > 0:
    common = common.limit(max_variants)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2. One centered, mean-imputed feature vector per variant (length N = #samples)

# COMMAND ----------

@F.udf(VectorUDT())
def center_impute(states):
    obs = [float(x) for x in states if x is not None and x >= 0]
    m = sum(obs) / len(obs) if obs else 0.0
    return Vectors.dense([0.0 if (x is None or x < 0) else float(x) - m for x in states])

variant_rows = common.select(center_impute(F.col("states")).alias("features"))

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3. Fit Spark ML PCA; per-sample coords are the principal components (model.pc)

# COMMAND ----------

k = min(n_components, N)
model = PCA(k=k, inputCol="features", outputCol="pcs").fit(variant_rows)
explained = [round(float(x), 4) for x in model.explainedVariance]
print(f"explained variance ratio (top {k}): {explained}")

# model.pc is a DenseMatrix of shape (numFeatures = N samples) × k.
# Row i = sample i's coordinates on the k PCs.
pc = model.pc.toArray()  # numpy (N × k), small (driver-side, N = #samples)
rows_out = [(sample_ids[i], *[float(pc[i][j]) for j in range(k)]) for i in range(N)]
schema_cols = ["sample_id"] + [f"PC{j + 1}" for j in range(k)]
pcs = spark.createDataFrame(rows_out, schema_cols)

results_table = f"pca_components_{mlflow_run_id.replace('-', '_')}"
pcs.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(f"{catalog}.{schema}.{results_table}")
print(f"Wrote {pcs.count()} samples × {k} PCs → {catalog}.{schema}.{results_table}")
