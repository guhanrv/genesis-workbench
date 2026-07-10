# Databricks notebook source
# MAGIC %md
# MAGIC # PRS reconcile driver — the incremental brain + cost kill-switch
# MAGIC
# MAGIC Computes the **desired** `(sample × registered-PGS)` grid, anti-joins the existing `prs_scores`
# MAGIC (matching on `weight_sha` + `panel_version`), and emits **only the missing/stale cells** as a
# MAGIC plan table the scorer consumes. This is the single point that decides how much compute runs.
# MAGIC
# MAGIC **Hard guardrails (enforced here, not by discipline):**
# MAGIC - **Dry-run by default** (`apply=false`): prints the plan + estimated cost and writes nothing to score.
# MAGIC - **`max_cells` cap**: refuses to emit a runnable plan larger than the cap unless `apply=true` AND
# MAGIC   `confirm_large=true`. A stray "score everything" can never fire silently.
# MAGIC - Idempotent: re-running with no changes yields **0 cells** (proven in the design smoke tests).

# COMMAND ----------

dbutils.widgets.text("catalog", "genesis_workbench", "Catalog")
dbutils.widgets.text("schema", "genesis_schema", "Schema")
dbutils.widgets.text("samples", "", "Sample ids (comma-sep; empty = all in dosage store)")
dbutils.widgets.text("pgs_ids", "", "PGS ids (comma-sep; empty = all registered)")
dbutils.widgets.text("panel_version", "", "Panel version to score against (empty = latest in pgs_panel_ref)")
dbutils.widgets.text("apply", "false", "false = dry-run (plan only); true = write runnable plan")
dbutils.widgets.text("max_cells", "100000", "Refuse to emit a runnable plan larger than this")
dbutils.widgets.text("confirm_large", "false", "Must be true to exceed max_cells")
dbutils.widgets.text("dbu_per_cell", "0.0", "Calibrated cost/cell from Stage-0 benchmark (0 = uncalibrated)")

catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")
apply = dbutils.widgets.get("apply").strip().lower() == "true"
confirm_large = dbutils.widgets.get("confirm_large").strip().lower() == "true"
max_cells = int(dbutils.widgets.get("max_cells"))
dbu_per_cell = float(dbutils.widgets.get("dbu_per_cell"))

# COMMAND ----------

import pyspark.sql.functions as F

spark.sql(f"USE CATALOG {catalog}")
spark.sql(f"USE SCHEMA {schema}")

def _csv(name):
    v = dbutils.widgets.get(name).strip()
    return [x.strip() for x in v.split(",") if x.strip()] if v else None

sample_filter = _csv("samples")
pgs_filter = _csv("pgs_ids")
panel_version = dbutils.widgets.get("panel_version").strip()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Desired grid = (samples) × (registered PGS, at their current weight_sha) × panel_version

# COMMAND ----------

registry = spark.table("pgs_registry").select("pgs_id", "weight_sha")
if pgs_filter:
    registry = registry.where(F.col("pgs_id").isin(pgs_filter))

# panel_version to normalize against: explicit param, else the version curated for the CURRENTLY-
# registered weights — join pgs_panel_ref on (pgs_id, weight_sha), NOT a lexicographic max over all
# historical versions (string max mis-picks across digit boundaries, e.g. 'v10' < 'v2'). max() here
# only tiebreaks the rare case where the same weights were re-curated under several versions.
if panel_version:
    pv = F.lit(panel_version)
else:
    pv_tbl = (spark.table("pgs_panel_ref").select("pgs_id", "weight_sha", "panel_version").distinct()
              .join(registry, ["pgs_id", "weight_sha"])
              .groupBy("pgs_id").agg(F.max("panel_version").alias("panel_version")))
    registry = registry.join(pv_tbl, "pgs_id", "left")

samples = spark.table("dosage").select("sample_id").distinct()
if sample_filter:
    samples = samples.where(F.col("sample_id").isin(sample_filter))

desired = samples.crossJoin(registry)
if panel_version:
    desired = desired.withColumn("panel_version", pv)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Missing/stale = desired ⟂ prs_scores on (sample_id, pgs_id, weight_sha, panel_version)
# MAGIC A cell is recomputed iff absent, or its stored `weight_sha`/`panel_version` differs (restated PGS).

# COMMAND ----------

existing = spark.table("prs_scores").select(
    "sample_id", "pgs_id",
    F.col("weight_sha").alias("e_sha"), F.col("panel_version").alias("e_pv"),
)
# stale iff NOT (stored weight_sha AND panel_version both null-safe-equal the desired). Null-safe (<=>)
# so a PGS with NO panel_ref row (desired panel_version = NULL) still converges: NULL<=>NULL is a match,
# so an already-scored NULL-panel_version cell is NOT re-planned every run (idempotent).
plan = (
    desired.join(existing, ["sample_id", "pgs_id"], "left")
    .where(~(F.col("e_sha").eqNullSafe(F.col("weight_sha")) & F.col("e_pv").eqNullSafe(F.col("panel_version"))))
    .select("sample_id", "pgs_id", "weight_sha", "panel_version")
)

n_cells = plan.count()
n_samples = plan.select("sample_id").distinct().count()
n_pgs = plan.select("pgs_id").distinct().count()
est = f"{n_cells * dbu_per_cell:.2f} DBU" if dbu_per_cell > 0 else "UNCALIBRATED (run Stage-0 benchmark to set dbu_per_cell)"

print("=" * 64)
print(f"RECONCILE PLAN: {n_cells} cells to compute  ({n_samples} samples × {n_pgs} pgs touched)")
print(f"  estimated cost: {est}")
print(f"  mode: {'APPLY' if apply else 'DRY-RUN (no plan written)'}   max_cells={max_cells}")
print("=" * 64)

# COMMAND ----------

# --- kill-switch --------------------------------------------------------------
# The scorer (next task) reads _prs_reconcile_plan, so we ALWAYS (over)write it here —
# writing it EMPTY on a no-op/dry-run makes the scorer a guaranteed no-op and can never
# re-run a stale plan left by a prior apply.
empty_plan = plan.limit(0)

if n_cells == 0:
    empty_plan.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable("_prs_reconcile_plan")
    print("Nothing to do — prs_scores is already reconciled (idempotent). Wrote empty plan.")
    dbutils.notebook.exit("0")

if not apply:
    empty_plan.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable("_prs_reconcile_plan")
    print("DRY-RUN: wrote EMPTY plan (scorer will no-op). Re-run with apply=true to score. Nothing scored.")
    dbutils.notebook.exit(str(n_cells))

if n_cells > max_cells and not confirm_large:
    raise ValueError(
        f"REFUSING: plan has {n_cells} cells > max_cells={max_cells}. "
        f"This guards against an accidental full-grid backfill. "
        f"To proceed intentionally, set confirm_large=true (and make sure the cluster is sized/approved)."
    )

# COMMAND ----------

# Persist the runnable plan; the scorer (05_score_prs) reads THIS table, so it only ever
# touches the cells reconcile approved — never the full grid.
plan.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable("_prs_reconcile_plan")
print(f"Wrote runnable plan: {catalog}.{schema}._prs_reconcile_plan ({n_cells} cells). Scorer will process only these.")
