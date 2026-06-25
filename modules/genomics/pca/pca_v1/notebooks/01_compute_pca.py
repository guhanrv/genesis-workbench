# Databricks notebook source
# MAGIC %md
# MAGIC # Ancestry / population-structure PCA with Glow + Spark ML
# MAGIC
# MAGIC Computes per-sample principal components from a cohort VCF — the standard
# MAGIC population-structure covariates a GWAS adjusts for (the `gwas` submodule
# MAGIC currently runs *unadjusted*; these PCs close that gap and also give a basis
# MAGIC for ancestry analysis).
# MAGIC
# MAGIC Spark/Glow-native, mirroring the GWAS modality:
# MAGIC 1. Glow read VCF → per-sample alt-allele dosage (`glow.genotype_states`).
# MAGIC 2. Keep biallelic common SNPs (MAF ≥ cutoff), downsample to `max_variants`
# MAGIC    (Spark ML's PCA covariance is dense, so columns must stay < 65535).
# MAGIC 3. Mean-impute missing dosage, assemble a per-sample sparse vector, fit
# MAGIC    `pyspark.ml.feature.PCA` and emit `PC1..PCk` per sample.

# COMMAND ----------

dbutils.widgets.text("catalog", "genesis_workbench", "Catalog")
dbutils.widgets.text("schema", "genesis_schema", "Schema")
dbutils.widgets.text("sql_warehouse_id", "w123", "SQL Warehouse Id")
dbutils.widgets.text("vcf_path", "", "VCF file path (cohort)")
dbutils.widgets.text("n_components", "10", "Number of principal components")
dbutils.widgets.text("maf_cutoff", "0.05", "Minor-allele-frequency cutoff")
dbutils.widgets.text("max_variants", "50000", "Max SNPs used for PCA (< 65535)")
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
n_components = int(dbutils.widgets.get("n_components"))
maf_cutoff = float(dbutils.widgets.get("maf_cutoff"))
max_variants = int(dbutils.widgets.get("max_variants"))
mlflow_run_id = dbutils.widgets.get("mlflow_run_id")

# COMMAND ----------

import glow
import pyspark.sql.functions as F
from pyspark.sql.window import Window
from pyspark.ml.feature import PCA
from pyspark.ml.functions import vector_to_array
from pyspark.ml.linalg import Vectors, VectorUDT

spark = glow.register(spark)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 1. Read VCF, keep biallelic common SNPs, downsample

# COMMAND ----------

variants = spark.read.format("vcf").load(vcf_path)
snps = (
    variants
    .where(F.size("alternateAlleles") == 1)
    .where((F.length("referenceAllele") == 1) & (F.length(F.col("alternateAlleles")[0]) == 1))
    .select(
        F.concat_ws(":", F.col("contigName"), F.col("start")).alias("variant_key"),
        glow.genotype_states(F.col("genotypes")).alias("states"),
        F.col("genotypes.sampleId").alias("sample_ids"),
    )
)

# alt-allele frequency from non-missing calls; keep common SNPs (maf >= cutoff)
freq = snps.select(
    "variant_key", "states", "sample_ids",
    F.expr("aggregate(filter(states, x -> x >= 0), 0, (a, x) -> a + x)").alias("alt_sum"),
    F.expr("size(filter(states, x -> x >= 0))").alias("n_called"),
)
freq = freq.withColumn("af", F.col("alt_sum") / (2 * F.col("n_called")))
common = freq.where(
    (F.col("n_called") > 0)
    & (F.least(F.col("af"), 1 - F.col("af")) >= F.lit(maf_cutoff))
)

# deterministic downsample to <= max_variants (Spark ML PCA column cap)
common = common.withColumn("vidx", F.row_number().over(Window.orderBy("variant_key")) - 1)
common = common.where(F.col("vidx") < max_variants)
n_variants = common.count()
print(f"PCA on {n_variants} common biallelic SNPs (maf >= {maf_cutoff}, capped at {max_variants})")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2. Long form, mean-impute missing dosage

# COMMAND ----------

long = common.select(
    "vidx",
    F.explode(F.arrays_zip("sample_ids", "states")).alias("z"),
).select("vidx", F.col("z.sample_ids").alias("sample_id"), F.col("z.states").cast("double").alias("state"))

means = long.where(F.col("state") >= 0).groupBy("vidx").agg(F.mean("state").alias("m"))
imp = (
    long.join(means, "vidx")
    .withColumn("d", F.when(F.col("state") < 0, F.col("m")).otherwise(F.col("state")))
    .select("sample_id", "vidx", "d")
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3. Assemble per-sample sparse vector, fit Spark ML PCA

# COMMAND ----------

@F.udf(VectorUDT())
def to_vector(pairs, size):
    d = {int(p["vidx"]): float(p["d"]) for p in pairs if p["d"] is not None}
    return Vectors.sparse(int(size), sorted(d.items()))

per_sample = imp.groupBy("sample_id").agg(F.collect_list(F.struct("vidx", "d")).alias("pairs"))
vecdf = per_sample.withColumn("features", to_vector(F.col("pairs"), F.lit(n_variants))).select("sample_id", "features")

k = min(n_components, n_variants)
pca = PCA(k=k, inputCol="features", outputCol="pcs").fit(vecdf)
explained = [float(x) for x in pca.explainedVariance]
print(f"explained variance ratio (top {k}): {[round(x, 4) for x in explained]}")

pcs = (
    pca.transform(vecdf)
    .select("sample_id", vector_to_array("pcs").alias("pc"))
    .select("sample_id", *[F.col("pc")[i].alias(f"PC{i + 1}") for i in range(k)])
)

results_table = f"pca_components_{mlflow_run_id.replace('-', '_')}"
pcs.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(f"{catalog}.{schema}.{results_table}")
print(f"Wrote {pcs.count()} samples × {k} PCs → {catalog}.{schema}.{results_table}")
