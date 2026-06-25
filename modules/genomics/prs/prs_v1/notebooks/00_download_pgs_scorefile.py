# Databricks notebook source
# MAGIC %md
# MAGIC # Download an example PGS Catalog scoring file
# MAGIC
# MAGIC Pulls one harmonized (GRCh38) scoring file from the PGS Catalog FTP into the
# MAGIC `prs_reference` volume, so the scoring workflow has a runnable example out of
# MAGIC the box. Users point `scorefile_path` at any PGS Catalog harmonized file.

# COMMAND ----------

dbutils.widgets.text("catalog", "genesis_workbench", "Catalog")
dbutils.widgets.text("schema", "genesis_schema", "Schema")
dbutils.widgets.text("example_pgs_id", "PGS000004", "Example PGS Catalog id (GRCh38)")

catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")
pgs_id = dbutils.widgets.get("example_pgs_id")

# COMMAND ----------

import os
import subprocess

ref_dir = f"/Volumes/{catalog}/{schema}/prs_reference/scorefiles"
os.makedirs(ref_dir, exist_ok=True)

# PGS Catalog harmonized scoring file (positions lifted to GRCh38).
fname = f"{pgs_id}_hmPOS_GRCh38.txt.gz"
url = f"https://ftp.ebi.ac.uk/pub/databases/spot/pgs/scores/{pgs_id}/ScoringFiles/Harmonized/{fname}"
dest = os.path.join(ref_dir, fname)

if os.path.exists(dest):
    print(f"{fname} already present, skipping download")
else:
    print(f"Downloading {url} ...")
    try:
        subprocess.run(
            ["curl", "-L", "--fail", "--retry", "5", "--retry-delay", "10",
             "--connect-timeout", "30", "-o", dest, url],
            check=True,
        )
        print(f"Downloaded → {dest}")
    except Exception:
        if os.path.exists(dest):
            os.remove(dest)  # don't leave a partial a re-run would treat as complete
        raise

# COMMAND ----------

print(f"Example scoring file available at: {dest}")
