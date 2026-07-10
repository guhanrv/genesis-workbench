# Databricks notebook source
# MAGIC %md
# MAGIC # PRS store maintenance — OPTIMIZE + VACUUM + stats (storage-safety guardrail)
# MAGIC
# MAGIC The plan's storage-safety item: MERGE-heavy Delta tables accumulate small files + old versions.
# MAGIC This compacts them and reclaims space. Safe to schedule (idempotent, cheap on classic single-node).
# MAGIC
# MAGIC - `dosage` is Liquid-Clustered by `sample_id` → `OPTIMIZE` reclusters (file-skipping on add-sample).
# MAGIC - `prs_scores` is partitioned by `pgs_id` → `OPTIMIZE` compacts per partition.
# MAGIC - `VACUUM` drops files older than the retention window (Delta time-travel horizon).
# MAGIC
# MAGIC **VACUUM is destructive to time-travel history** older than the window — keep `retention_hours`
# MAGIC ≥ 168 (7 days, Delta's safety default) unless you knowingly want a shorter horizon.

# COMMAND ----------

dbutils.widgets.text("catalog", "genesis_workbench", "Catalog")
dbutils.widgets.text("schema", "genesis_schema", "Schema")
dbutils.widgets.text("retention_hours", "168", "VACUUM retention window (hours; >=168 recommended)")
dbutils.widgets.text("run_vacuum", "true", "Run VACUUM (false = OPTIMIZE + ANALYZE only)")

catalog = dbutils.widgets.get("catalog"); schema = dbutils.widgets.get("schema")
retention_hours = int(dbutils.widgets.get("retention_hours"))
run_vacuum = dbutils.widgets.get("run_vacuum").strip().lower() == "true"

# COMMAND ----------

spark.sql(f"USE CATALOG {catalog}"); spark.sql(f"USE SCHEMA {schema}")

STORES = ["pgs_registry", "pgs_weights", "pgs_panel_ref", "dosage", "prs_scores"]

for t in STORES:
    if not spark.catalog.tableExists(t):
        print(f"  {t}: absent, skipping")
        continue
    print(f"OPTIMIZE {t} …")
    # Clustered/partitioned tables cluster/compact automatically; a plain OPTIMIZE is correct for both
    # (Liquid Clustering ignores ZORDER, and prs_scores is partitioned — no ZORDER key needed).
    spark.sql(f"OPTIMIZE {t}")
    spark.sql(f"ANALYZE TABLE {t} COMPUTE STATISTICS")   # refresh stats for the cost-based optimizer

# COMMAND ----------

if run_vacuum:
    for t in STORES:
        if spark.catalog.tableExists(t):
            print(f"VACUUM {t} RETAIN {retention_hours} HOURS …")
            spark.sql(f"VACUUM {t} RETAIN {retention_hours} HOURS")
else:
    print("VACUUM skipped (run_vacuum=false)")

# COMMAND ----------

for t in STORES:
    if spark.catalog.tableExists(t):
        d = spark.sql(f"DESCRIBE DETAIL {t}").first()
        print(f"  {t}: {d['numFiles']} files, {round((d['sizeInBytes'] or 0)/1e6, 1)} MB")
print("maintenance complete.")
