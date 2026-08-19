"""Readiness gate for a PRS scoring run (Direction B).

Fails loud if the planned cells cannot be calibrated: missing ancestry, missing weights,
or a panel_version mismatch. Importable off-cluster (Spark session required for table checks).
"""
from __future__ import annotations


def assert_scoring_ready(spark, catalog: str, schema: str, plan, *, require_models: bool = False):
    """Raise ValueError if score cannot produce calibrated z for ``plan``.

    ``plan`` is a Spark DataFrame with at least ``panel_version`` (and usually
    ``pgs_id`` / ``weight_sha``). ``require_models`` is False so an npz-only workspace
    still scores; set True once @champion models are mandatory.
    """
    spark.sql(f"USE CATALOG {catalog}")
    spark.sql(f"USE SCHEMA {schema}")

    missing_tables = [t for t in ("pgs_weights", "pgs_panel_ref", "sample_ancestry", "dosage")
                      if not spark.catalog.tableExists(t)]
    if missing_tables:
        raise ValueError(
            f"PRS stores not ready: missing {missing_tables}. Run 00_setup_stores + register + "
            f"extract + ancestry before scoring.")

    versions = [r["panel_version"] for r in plan.select("panel_version").distinct().collect()]
    if not versions:
        return  # empty plan — scorer is a no-op anyway

    pref = spark.table("pgs_panel_ref")
    for v in versions:
        n = pref.where(pref.panel_version == v).limit(1).count()
        if n == 0:
            raise ValueError(
                f"pgs_panel_ref has no rows for panel_version={v!r}. Run ref_00_build_panel_stats "
                f"with that version (must match curation) before scoring, or z will be null.")

    if require_models:
        pca_name = f"{catalog}.{schema}.ancestry_pca"
        try:
            from mlflow import MlflowClient
            c = MlflowClient()
            c.get_model_version_by_alias(pca_name, "champion")
        except Exception as e:
            raise ValueError(
                f"ancestry_pca@champion is required but not registered ({pca_name}): {e}") from e
