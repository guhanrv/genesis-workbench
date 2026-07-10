# Databricks notebook source
# MAGIC %md
# MAGIC # PRS store setup — the five persistent Delta tables
# MAGIC
# MAGIC Idempotent DDL for the distributed/incremental scorer's fixed-name stores (extends the
# MAGIC `clinvar_variants` / `acmg_gene_panel` reference-table precedent). All schemas are **flat scalars**
# MAGIC — no Glow VCF-INFO structs — so `MERGE INTO` is safe here (the struct-drift failure that made
# MAGIC genesis retreat from MERGE on results cannot occur on these).
# MAGIC
# MAGIC Stores:
# MAGIC 1. `pgs_registry`      — catalog of available scores (one row per PGS).
# MAGIC 2. `pgs_weights`       — long, signed per-variant weights (append per new PGS).
# MAGIC 3. `pgs_panel_ref`     — FROZEN per-PGS × superpop reference distributions (enables per-sample z with no re-rank).
# MAGIC 4. `dosage`            — per-sample alt-allele dosage at union variants (sparse; incrementally extended).
# MAGIC 5. `prs_scores`        — the results cell-store, keyed (sample_id, pgs_id, weight_sha, panel_version). MERGE-upsert.

# COMMAND ----------

dbutils.widgets.text("catalog", "genesis_workbench", "Catalog")
dbutils.widgets.text("schema", "genesis_schema", "Schema")
dbutils.widgets.text("sql_warehouse_id", "w123", "SQL Warehouse Id")

catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")

# COMMAND ----------

spark.sql(f"USE CATALOG {catalog}")
spark.sql(f"USE SCHEMA {schema}")

# COMMAND ----------

# 1. Registry — one row per available score. `weight_sha` versions the weight file so a
#    single-PGS restatement never invalidates the other 100 (per-PGS scoping).
spark.sql("""
CREATE TABLE IF NOT EXISTS pgs_registry (
    pgs_id STRING, score_id STRING, disease STRING, direction STRING, body_system STRING,
    hr_per_sd DOUBLE, clinical_model STRING, training_ancestries STRING,
    weight_sha STRING, n_variants BIGINT, weight_path STRING, registered_at TIMESTAMP
) USING DELTA
""")

# 2. Weights — long, EFFECT-ORIENTED: variant_id = chr:pos:effect:other and the stored
#    dose is the dose of `effect`, so raw = Σ dose·weight (plain weight, no signed/offset).
#    (The signed-weight/offset trick is panel-side only — it's unsafe on gVCF-extracted
#    user dose where a REF block's FASTA base is neither catalog allele; validated on real data.)
spark.sql("""
CREATE TABLE IF NOT EXISTS pgs_weights (
    pgs_id STRING, variant_id STRING, effect_allele STRING, other_allele STRING,
    weight DOUBLE, weight_sha STRING
) USING DELTA
""")

# 3. Panel reference — FROZEN per (pgs_id, superpop). Computed once per PGS; lets a new sample be
#    normalized (z/percentile) against a fixed panel with zero cohort re-ranking.
spark.sql("""
CREATE TABLE IF NOT EXISTS pgs_panel_ref (
    pgs_id STRING, superpop STRING, mean DOUBLE, sd DOUBLE,
    quantiles ARRAY<DOUBLE>, n_panel INT, panel_version STRING, weight_sha STRING
) USING DELTA
""")

# 4. Dosage — per-sample effect-allele dosage at covered union variants (real dose / covered-REF→0;
#    truly-missing omitted → scorer treats as 0). Incrementally extended via anti-join on missing
#    (sample_id, variant_id). CLUSTER BY sample_id (Liquid Clustering): add-sample / small-batch
#    scoring file-skips to just those samples instead of scanning the whole store (Stage-0 finding).
spark.sql("""
CREATE TABLE IF NOT EXISTS dosage (
    sample_id STRING, variant_id STRING, dose DOUBLE
) USING DELTA
CLUSTER BY (sample_id)
""")

# 5. Scores cell-store — the results. Flat scalars → MERGE-safe. Partitioned by pgs_id
#    (add-PGS = new partition; disease/pgs queries pruned).
spark.sql("""
CREATE TABLE IF NOT EXISTS prs_scores (
    sample_id STRING, pgs_id STRING, weight_sha STRING, panel_version STRING,
    raw_score DOUBLE, n_variants_matched INT, coverage_pct DOUBLE, small_score BOOLEAN,
    most_similar_pop STRING, used_ancestry STRING,
    z_msp DOUBLE, percentile_msp DOUBLE, z_admixed DOUBLE, percentile_admixed DOUBLE,
    integrated_z_source STRING, integrated_risk_10yr DOUBLE, clinical_risk_10yr DOUBLE,
    risk_category STRING, concordance_verdict STRING, computed_at TIMESTAMP
) USING DELTA
PARTITIONED BY (pgs_id)
""")

# 6. Ancestry — per-sample most-similar-population from the frozen PCA basis (pca_v1 builds the basis;
#    05_build_sample_ancestry projects+classifies). The scorer left-joins this on sample_id to pick the panel superpop row
#    for z_msp/percentile_msp; declared here so it always exists (empty ⇒ scorer degrades to null z).
spark.sql("""
CREATE TABLE IF NOT EXISTS sample_ancestry (
    sample_id STRING, most_similar_pop STRING, mahalanobis_p_all DOUBLE,
    n_loci_covered BIGINT, n_loci_basis BIGINT, panel_version STRING,
    rf_probs STRING, computed_at TIMESTAMP
) USING DELTA
""")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Data governance (PHI)
# MAGIC `dosage` / `prs_scores` / `sample_ancestry` hold per-sample genotype-derived data (PHI). Access is
# MAGIC controlled at the Unity Catalog **catalog/schema** level (the deploying identity owns the schema;
# MAGIC grant read to consumers explicitly). We additionally **tag** these tables (`data_classification=PHI`
# MAGIC + a comment) so UC lineage / discovery / policy tooling can find and govern them. Column masks /
# MAGIC row filters are org-specific UC policies applied out-of-band (a masking UDF + `ALTER TABLE … SET
# MAGIC MASK`); this module marks the classification so a policy is applied before production PHI lands,
# MAGIC rather than hardcoding one.

# COMMAND ----------

_PHI = {
    "dosage": "PHI: per-sample effect-allele dosage (genotype-derived).",
    "prs_scores": "PHI: per-sample polygenic risk scores.",
    "sample_ancestry": "PHI: per-sample inferred ancestry.",
}
for t, desc in _PHI.items():
    spark.sql(f"ALTER TABLE {t} SET TBLPROPERTIES ('data_classification' = 'PHI')")   # idempotent
    spark.sql(f"COMMENT ON TABLE {t} IS '{desc}'")

# COMMAND ----------

for t in ["pgs_registry", "pgs_weights", "pgs_panel_ref", "dosage", "prs_scores", "sample_ancestry"]:
    n = spark.table(t).count()
    print(f"  {catalog}.{schema}.{t}: exists, {n} rows")
print("PRS stores ready.")
