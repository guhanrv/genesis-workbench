# Databricks notebook source
# MAGIC %md
# MAGIC # PRS ingest — hard-called / imputed VCF → long effect-oriented `dosage` (Glow path)
# MAGIC
# MAGIC The non-gVCF counterpart to `00_extract_dosage` (pysam gVCF path). For hard-called or imputed
# MAGIC cohorts (DS/HDS/GT, **no `END=` REF blocks**), Glow reads dosage distributedly. We then orient the
# MAGIC ALT dosage to each catalog PGS's **effect allele** and write the SAME long
# MAGIC `dosage(sample_id, variant_id, dose)` store the gVCF path writes — so scoring is identical downstream.
# MAGIC
# MAGIC Glow is a JVM Spark extension → **classic cluster only** (never serverless). The orientation logic
# MAGIC mirrors `prs_extract.orient_alt_dose` (unit-tested off-cluster); palindromic (A/T,C/G) SNPs dropped.
# MAGIC
# MAGIC NOTE: the Glow read itself is validated only on a classic cluster (JVM); the orientation math is
# MAGIC the unit-tested part.

# COMMAND ----------

dbutils.widgets.text("catalog", "genesis_workbench", "Catalog")
dbutils.widgets.text("schema", "genesis_schema", "Schema")
dbutils.widgets.text("vcf_path", "", "Hard-called/imputed VCF (cohort to ingest)")
dbutils.widgets.text("dosage_field", "auto", "Dosage source: auto | DS | HDS | GT")
dbutils.widgets.text("pgs_ids", "", "Restrict to these PGS' variants (comma-sep; empty = all registered)")

catalog = dbutils.widgets.get("catalog"); schema = dbutils.widgets.get("schema")

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

catalog = dbutils.widgets.get("catalog"); schema = dbutils.widgets.get("schema")
vcf_path = dbutils.widgets.get("vcf_path"); dosage_field = dbutils.widgets.get("dosage_field")
pgs_filter = [x.strip() for x in dbutils.widgets.get("pgs_ids").split(",") if x.strip()]

import glow
import pyspark.sql.functions as F
from delta.tables import DeltaTable
spark = glow.register(spark)
spark.sql(f"USE CATALOG {catalog}"); spark.sql(f"USE SCHEMA {schema}")

# COMMAND ----------

# MAGIC %md ### 1. Glow read → per-sample ALT dosage (DS → HDS → GT), exploded long

# COMMAND ----------

variants = spark.read.format("vcf").load(vcf_path).where(F.size("alternateAlleles") == 1)  # biallelic
gt_elem = variants.schema["genotypes"].dataType.elementType
avail = {f.name.lower(): f.name for f in gt_elem.fields}
pref = (dosage_field or "auto").strip().lower()

def _states_expr():
    if pref == "ds" or (pref == "auto" and "ds" in avail):
        return "DS", F.expr(f"transform(genotypes, g -> cast(g.`{avail['ds']}` as double))")
    if pref == "hds" or (pref == "auto" and "hds" in avail):
        f = avail["hds"]
        return "HDS", F.expr(f"transform(genotypes, g -> case when g.`{f}` is null then null "
                             f"else aggregate(g.`{f}`, cast(0.0 as double), (a, x) -> a + cast(x as double)) end)")
    return "GT", F.expr("transform(genotype_states(genotypes), x -> case when x < 0 then null else cast(x as double) end)")

kind, states = _states_expr()
print(f"dosage source: {kind}")

# one row per (variant, sample): sample_id + ALT dosage (Glow start is 0-based → +1; strip chr → ensembl)
long_var = (variants.select(
        F.regexp_replace(F.col("contigName"), "chr", "").alias("chrom"),
        (F.col("start") + 1).alias("pos"),
        F.col("referenceAllele").alias("ref"),
        F.col("alternateAlleles")[0].alias("alt"),
        F.arrays_zip(F.col("genotypes.sampleId").alias("sid"), states.alias("st")).alias("z"))
    .select("chrom", "pos", "ref", "alt", F.explode("z").alias("z"))
    .select("chrom", "pos", "ref", "alt",
            F.col("z.sampleId").alias("sample_id"), F.col("z.st").alias("alt_dose"))
    .where(F.col("alt_dose").isNotNull()))

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2. Orient ALT dosage → effect-allele dose (mirrors prs_extract.orient_alt_dose), drop palindromic
# MAGIC effect==alt & other==ref → alt_dose · effect==ref & other==alt → 2−alt_dose · else drop · A/T,C/G drop

# COMMAND ----------

wq = spark.table("pgs_weights")
if pgs_filter:
    wq = wq.where(F.col("pgs_id").isin(pgs_filter))
cat = (wq.select("variant_id").distinct().select(F.split("variant_id", ":").alias("p"))
       .select(F.col("p")[0].alias("c"), F.col("p")[1].cast("long").alias("p2"),
               F.col("p")[2].alias("effect"), F.col("p")[3].alias("other")))

e, o, r, a, d = F.upper("effect"), F.upper("other"), F.upper("ref"), F.upper("alt"), F.col("alt_dose")
pair = F.array_sort(F.array(e, o))
palin = pair.isin([["A", "T"], ["C", "G"]])
dose_expr = (F.when(palin, F.lit(None).cast("double"))
             .when((e == a) & (o == r), d)
             .when((e == r) & (o == a), F.lit(2.0) - d)
             .otherwise(F.lit(None).cast("double")))

oriented = (long_var.join(cat, (long_var.chrom == cat.c) & (long_var.pos == cat.p2))
            .withColumn("dose", dose_expr)
            .where(F.col("dose").isNotNull())
            .select("sample_id",
                    F.concat_ws(":", F.col("effect"), F.col("other")).alias("_eo"),
                    F.concat_ws(":", cat.c, cat.p2.cast("string"), F.col("effect"), F.col("other")).alias("variant_id"),
                    "dose")
            .select("sample_id", "variant_id", "dose"))

# COMMAND ----------

# MAGIC %md ### 3. MERGE into the shared long `dosage` store (idempotent on sample_id, variant_id)

# COMMAND ----------

oriented.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable("_dosage_stage_hc")
stage = spark.table("_dosage_stage_hc")
(DeltaTable.forName(spark, f"{catalog}.{schema}.dosage").alias("t")
 .merge(stage.alias("s"), "t.sample_id = s.sample_id AND t.variant_id = s.variant_id")
 .whenMatchedUpdateAll().whenNotMatchedInsertAll().execute())
print(f"ingested {stage.count()} (sample,variant) hard-called dosage rows across "
      f"{stage.select('sample_id').distinct().count()} samples → {catalog}.{schema}.dosage")
spark.sql("DROP TABLE IF EXISTS _dosage_stage_hc")
