# Databricks notebook source
# MAGIC %md
# MAGIC # Ancestry — per-sample MSP against the FROZEN panel PCA basis (classic; distributed)
# MAGIC
# MAGIC Writes `sample_ancestry(sample_id, most_similar_pop)` — the table the scorer joins to pick which
# MAGIC panel-superpop row normalizes each sample (`z_msp`/`percentile_msp`). Mirrors pgsc_calc's ancestry
# MAGIC step: **FRAPOSA OADP** projection onto a fixed HGDP+1kGP PC basis + a **RandomForest** MSP classifier.
# MAGIC
# MAGIC **Distribution boundary (the point of this notebook):**
# MAGIC   - the fixed basis (QC + LD-prune + panel eigendecomposition) is a **one-time reference artifact**
# MAGIC     (`pca_basis.npz` on the Volume — built by the pca module, `pca_v1/ref_01_build_basis`, à la
# MAGIC     pgsc_calc's reference), NOT
# MAGIC     recomputed per run;
# MAGIC   - the **heavy per-sample work is distributed**: gVCF dose extraction at the basis loci (chrom-sharded,
# MAGIC     the same kernel as `02_extract_dosage`) + the OADP projection (`applyInPandas`, **numpy-only** on
# MAGIC     executors — `project_member` pulls in no sklearn), so it scales across samples;
# MAGIC   - the RF classify is O(n_samples × 5 PCs) — trained once and run on the **driver** over the collected
# MAGIC     per-sample PCs (keeps sklearn/scipy off the executors; trivial even at biobank scale).
# MAGIC
# MAGIC Incremental: a sample already in `sample_ancestry` is skipped unless `reclassify=true`. Classic cluster,
# MAGIC minimal + titrated (see job.yml). **Not serverless** (pysam + binary VCF I/O).

# COMMAND ----------

dbutils.widgets.text("catalog", "genesis_workbench", "Catalog")
dbutils.widgets.text("schema", "genesis_schema", "Schema")
dbutils.widgets.text("basis_path", "", "Volume path to pca_basis.npz (from pca_v1/ref_01_build_basis)")
dbutils.widgets.text("vcf_dir", "", "Dir of gVCFs (each *.vcf.gz / *.g.vcf.gz = one sample)")
dbutils.widgets.text("vcf_paths", "", "Explicit gVCF paths (comma-sep; overrides vcf_dir)")
dbutils.widgets.text("fasta_path", "", "GRCh38 FASTA (.fna.bgz) for REF-block resolution")
dbutils.widgets.text("fasta_ref_cache_dir", "", "Volume dir to cache the FASTA-ref array (skips the rebuild on repeat runs)")
dbutils.widgets.text("shard_by_chrom", "true", "Shard extraction (sample × chrom) across chroms/cores")
dbutils.widgets.text("reclassify", "false", "true = recompute samples already in sample_ancestry")
dbutils.widgets.text("outlier_alpha", "0.001", "Mahalanobis P below this => PC-space outlier (RF can't be trusted)")
dbutils.widgets.text("min_coverage_frac", "0.1", "Fraction of basis loci a sample must cover, else flagged outlier")

catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")
outlier_alpha = float(dbutils.widgets.get("outlier_alpha"))
min_coverage_frac = float(dbutils.widgets.get("min_coverage_frac"))

# COMMAND ----------

# MAGIC %pip install pysam scikit-learn scipy
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

import os
import glob
import hashlib
from pathlib import Path
import numpy as np
import pandas as pd
import pysam
import pyspark.sql.functions as F
from pyspark.sql.types import StructType, StructField, StringType, DoubleType, LongType, BooleanType
from delta.tables import DeltaTable

catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")
basis_path = dbutils.widgets.get("basis_path").strip()
vcf_dir = dbutils.widgets.get("vcf_dir")
vcf_paths_arg = dbutils.widgets.get("vcf_paths")
fasta_path = dbutils.widgets.get("fasta_path")
fasta_ref_cache_dir = dbutils.widgets.get("fasta_ref_cache_dir").strip()
shard_by_chrom = dbutils.widgets.get("shard_by_chrom").strip().lower() == "true"
reclassify = dbutils.widgets.get("reclassify").strip().lower() == "true"

spark.sql(f"USE CATALOG {catalog}")
spark.sql(f"USE SCHEMA {schema}")

# ship the extraction kernel + the FRAPOSA project/classify lib to executors.
# prs_ancestry keeps sklearn/scipy imports INSIDE classify_ancestry/fit_rf, so importing it on a
# (numpy-only) executor for project_member is safe — no sklearn needed out there.
lib_dir = os.path.abspath(os.path.join(os.getcwd(), "..", "lib"))
for m in ("gvcf_dose.py", "prs_extract.py", "prs_ancestry.py"):
    spark.sparkContext.addPyFile(os.path.join(lib_dir, m))
import sys; sys.path.append(lib_dir)
import gvcf_dose as kern
import prs_extract as ext
import prs_ancestry as anc

# COMMAND ----------

# MAGIC %md
# MAGIC ### 1. Load the frozen PCA basis (one-time reference artifact) + build the extraction catalog
# MAGIC `np.load` reads a zip (needs seek), which Volume FUSE rejects — so copy the npz to node-local /tmp
# MAGIC first (a sequential read FUSE allows), then load. The basis is ~19 MB → broadcast to every task.

# COMMAND ----------

if not basis_path:
    raise ValueError("basis_path is required — point it at pca_basis.npz (see pca_v1/ref_01_build_basis)")
_local_basis = "/tmp/" + os.path.basename(basis_path)
dbutils.fs.cp(basis_path, "file:" + _local_basis)
_b = np.load(_local_basis, allow_pickle=True)

# --- artifact contract: fail fast if the pca-built basis npz drifted (see pca_v1/lib/pca_fit) ---
# The basis crosses a module boundary as a Volume artifact, so its schema is an implicit contract.
# Assert the keys we consume + a matching schema_version, so a producer change breaks here at load
# with a clear message, not silently mid-projection.
_EXPECTED_BASIS_SCHEMA = "1"
_REQUIRED_KEYS = {"loci_chrom", "loci_pos", "loci_ref", "loci_alt", "U_on", "s_on", "V_on",
                  "pcs_ref", "mean", "std", "dim_ref", "dim_stu", "superpops", "panel_version"}
_missing = _REQUIRED_KEYS - set(_b.files)
if _missing:
    raise ValueError(f"basis {basis_path} is missing keys {sorted(_missing)} — incompatible with this "
                     f"consumer. Rebuild via pca_v1/ref_01_build_basis (BASIS_SCHEMA_VERSION={_EXPECTED_BASIS_SCHEMA}).")
_got_schema = str(_b["schema_version"]) if "schema_version" in _b.files else "0"
if _got_schema != _EXPECTED_BASIS_SCHEMA:
    raise ValueError(f"basis schema_version={_got_schema} != expected {_EXPECTED_BASIS_SCHEMA} — the pca "
                     f"basis contract changed. Rebuild via pca_v1/ref_01_build_basis or update this consumer.")

chrom, pos, ref, alt = _b["loci_chrom"], _b["loci_pos"], _b["loci_ref"], _b["loci_alt"]
n_loci = len(pos)
dim_ref = int(_b["dim_ref"])
panel_version = str(_b["panel_version"]) if "panel_version" in _b.files else "pca_v1"
superpops = _b["superpops"].tolist()
pcs_ref = _b["pcs_ref"]
# broadcast-safe basis dict for project_member (numpy-only)
basis = {"U_on": _b["U_on"], "s_on": _b["s_on"], "V_on": _b["V_on"], "pcs_ref": pcs_ref,
         "mean": _b["mean"], "std": _b["std"], "dim_ref": dim_ref, "dim_stu": int(_b["dim_stu"])}
print(f"basis: {n_loci} loci · {len(superpops)} panel samples · dim_ref={dim_ref} · panel_version={panel_version}")

# catalog: effect=ALT, other=REF → sample_dosage_rows yields ALT-dose, matching the panel's pgenlib dose
# (mean/std the basis standardizes against). variant_id = chrom:pos:ALT:REF.
rows = [(str(chrom[i]), int(pos[i]), str(alt[i]), str(ref[i])) for i in range(n_loci)]
ucat = ext.build_union_catalog(rows)
basis_vids = [f"{chrom[i]}:{pos[i]}:{alt[i]}:{ref[i]}" for i in range(n_loci)]

# FASTA ref-base per locus (once on the driver; content-addressed cache — same pattern as 02_extract_dosage)
cache_path = None; _vol_cache = None
if fasta_ref_cache_dir:
    key = hashlib.sha256((os.path.basename(basis_path) + "|" + os.path.basename(fasta_path)).encode()).hexdigest()[:16]
    fname = f"pcaref_{n_loci}_{key}.npz"
    cache_path = Path("/tmp") / fname
    _vol_cache = fasta_ref_cache_dir.rstrip("/") + "/" + fname
    try:
        dbutils.fs.cp(_vol_cache, "file:" + str(cache_path)); print("fasta-ref cache: pulled from volume")
    except Exception:
        pass
fasta_ref = kern.build_catalog_fasta_ref(ucat, fasta_path, cache_path=cache_path)
if _vol_cache and cache_path is not None and cache_path.exists():
    try:
        dbutils.fs.mkdirs(fasta_ref_cache_dir); dbutils.fs.cp("file:" + str(cache_path), _vol_cache)
    except Exception as e:
        print("warn: fasta-ref cache push failed (non-fatal):", e)

BASIS_B = spark.sparkContext.broadcast(basis)
VIDX_B = spark.sparkContext.broadcast({v: i for i, v in enumerate(basis_vids)})
if shard_by_chrom:
    BYCHROM_B = spark.sparkContext.broadcast(ext.split_catalog_by_chrom(ucat, fasta_ref))
    catalog_chroms = sorted(BYCHROM_B.value.keys())
    print(f"sharding by chrom: {len(catalog_chroms)} chroms")
else:
    UCAT_B = spark.sparkContext.broadcast(ucat)
    FASTA_B = spark.sparkContext.broadcast(fasta_ref)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2. Sample manifest + incrementality (skip samples already classified)

# COMMAND ----------

if vcf_paths_arg.strip():
    paths = [p.strip() for p in vcf_paths_arg.split(",") if p.strip()]
else:
    paths = sorted(glob.glob(os.path.join(vcf_dir, "*.vcf.gz")) + glob.glob(os.path.join(vcf_dir, "*.g.vcf.gz")))

manifest = []
for p in paths:
    with pysam.VariantFile(p) as vf:
        sid = list(vf.header.samples)[0]
    manifest.append((sid, p))
print(f"{len(manifest)} gVCF(s) found")

if not reclassify and spark.catalog.tableExists("sample_ancestry"):
    have = {r["sample_id"] for r in spark.table("sample_ancestry").select("sample_id").distinct().collect()}
    before = len(manifest)
    manifest = [(s, p) for (s, p) in manifest if s not in have]
    print(f"incremental: skipping {before - len(manifest)} already-classified; {len(manifest)} to classify")

if not manifest:
    dbutils.notebook.exit("0 — nothing to classify (use reclassify=true to force)")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3. Distribute: extract dose at basis loci → OADP-project each sample (numpy-only executors)

# COMMAND ----------

DOSE_SCHEMA = StructType([
    StructField("sample_id", StringType()),
    StructField("variant_id", StringType()),
    StructField("dose", DoubleType()),
])

def _extract_sample(itr):                        # one task per sample
    import prs_extract as _e, gvcf_dose as _k
    uc = UCAT_B.value; fa = FASTA_B.value
    for pdf in itr:
        for _, r in pdf.iterrows():
            sid, dose_rows = _e.sample_dosage_rows(r["vcf_path"], uc, fa, kernel=_k)
            if dose_rows:
                out = pd.DataFrame(dose_rows, columns=["variant_id", "dose"]); out.insert(0, "sample_id", sid)
                yield out

def _extract_sample_chrom(itr):                  # one task per (sample × chrom)
    import prs_extract as _e, gvcf_dose as _k
    bc = BYCHROM_B.value
    for pdf in itr:
        for _, r in pdf.iterrows():
            ch, cpos, eff, oth, vid, fa = bc[r["chrom"]]
            sub = _e.UnionCatalog(ch, cpos, eff, oth, vid)
            sid, dose_rows = _e.sample_dosage_rows(r["vcf_path"], sub, fa, kernel=_k)
            if dose_rows:
                out = pd.DataFrame(dose_rows, columns=["variant_id", "dose"]); out.insert(0, "sample_id", sid)
                yield out

if shard_by_chrom:
    tasks = [(sid, p, ch) for (sid, p) in manifest for ch in catalog_chroms]
    n_tasks = max(1, len(tasks))
    mdf = spark.createDataFrame(tasks, ["sample_id", "vcf_path", "chrom"]).repartition(n_tasks)
    dose = mdf.mapInPandas(_extract_sample_chrom, schema=DOSE_SCHEMA)
    print(f"distributing {len(manifest)} sample(s) × {len(catalog_chroms)} chrom = {n_tasks} tasks")
else:
    n_tasks = max(1, len(manifest))
    mdf = spark.createDataFrame(manifest, ["sample_id", "vcf_path"]).repartition(n_tasks)
    dose = mdf.mapInPandas(_extract_sample, schema=DOSE_SCHEMA)

# One row per sample: gather its dose into Xu (basis-loci order, missing → NaN → standardize→0),
# FRAPOSA-OADP project onto the broadcast basis. numpy-only — no sklearn on the executor.
PC_COLS = [f"pc{k}" for k in range(dim_ref)]
PROJ_SCHEMA = StructType(
    [StructField("sample_id", StringType())]
    + [StructField(c, DoubleType()) for c in PC_COLS]
    + [StructField("n_covered", LongType())]
)

def _project(pdf):
    import numpy as _np, prs_ancestry as _a
    sid = pdf["sample_id"].iloc[0]
    idx = VIDX_B.value; N = len(idx)
    Xu = _np.full(N, _np.nan, dtype=_np.float32)
    for vid, d in zip(pdf["variant_id"].to_numpy(), pdf["dose"].to_numpy()):
        j = idx.get(vid)
        if j is not None:
            Xu[j] = d
    ncov = int(_np.sum(~_np.isnan(Xu)))
    pc = _a.project_member(BASIS_B.value, Xu)
    return pd.DataFrame([[sid, *[float(pc[k]) for k in range(len(PC_COLS))], ncov]],
                        columns=["sample_id", *PC_COLS, "n_covered"])

proj = dose.groupBy("sample_id").applyInPandas(_project, schema=PROJ_SCHEMA)
proj_rows = proj.collect()   # (n_samples × ~dim_ref) — tiny, safe to collect even at biobank scale
print(f"projected {len(proj_rows)} sample(s)")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 4. Classify on the driver (RF trained once on the panel) → MERGE `sample_ancestry`
# MAGIC `king.cutoff` was already applied when the basis was built, so the whole panel is the unrelated
# MAGIC training set (`panel_unrelated=None`). MSP = RF argmax; Mahalanobis_P_ALL flags PC-space outliers.

# COMMAND ----------

import json
from scipy.stats import chi2

N_PCS = 5  # pgsc_calc classifies on 5 PCs
clf, cov = anc.fit_rf(pcs_ref, superpops, n_pcs=N_PCS)   # once, on the driver

# is_outlier gate: RF argmax ALWAYS returns a superpop, even for a sample that belongs to no
# reference population (non-human/contaminated, an ancestry absent from HGDP+1kGP) or that covers
# almost no basis loci (projects to ~origin). Flag those so the scorer refuses to report a calibrated
# z against the wrong panel, instead of silently normalizing. Two independent triggers:
#   - PC-space outlier: Mahalanobis_P_ALL < outlier_alpha (sample far from every superpop cloud)
#   - low coverage: covered/basis < min_coverage_frac (projection is unreliable / near-origin)
out_rows = []
for r in proj_rows:
    pc5 = np.array([r[f"pc{k}"] for k in range(N_PCS)], dtype=np.float64)[None, :]
    probs = clf.predict_proba(pc5)[0]
    msp = str(clf.classes_[int(np.argmax(probs))])
    d2 = float(cov.mahalanobis(pc5)[0])
    p_all = float(chi2.sf(d2, N_PCS))                      # df = n_pcs (was N_PCS-1, a bug)
    cov_frac = (int(r["n_covered"]) / n_loci) if n_loci else 0.0
    is_outlier = bool(p_all < outlier_alpha or cov_frac < min_coverage_frac)
    rf_probs = {str(c): float(p) for c, p in zip(clf.classes_, probs)}
    out_rows.append((r["sample_id"], msp, p_all, int(r["n_covered"]), n_loci,
                     panel_version, json.dumps(rf_probs), is_outlier))

OUT_SCHEMA = StructType([
    StructField("sample_id", StringType()),
    StructField("most_similar_pop", StringType()),
    StructField("mahalanobis_p_all", DoubleType()),
    StructField("n_loci_covered", LongType()),
    StructField("n_loci_basis", LongType()),
    StructField("panel_version", StringType()),
    StructField("rf_probs", StringType()),
    StructField("is_outlier", BooleanType()),
])
out = (spark.createDataFrame(out_rows, OUT_SCHEMA)
       .withColumn("computed_at", F.current_timestamp()))

spark.sql("""
CREATE TABLE IF NOT EXISTS sample_ancestry (
  sample_id STRING, most_similar_pop STRING, mahalanobis_p_all DOUBLE,
  n_loci_covered BIGINT, n_loci_basis BIGINT, panel_version STRING,
  rf_probs STRING, is_outlier BOOLEAN, computed_at TIMESTAMP
) USING DELTA
""")
# tolerate a pre-existing table created before is_outlier was added
spark.conf.set("spark.databricks.delta.schema.autoMerge.enabled", "true")
out.createOrReplaceTempView("_anc")
(DeltaTable.forName(spark, f"{catalog}.{schema}.sample_ancestry").alias("t")
 .merge(spark.table("_anc").alias("s"), "t.sample_id = s.sample_id")
 .whenMatchedUpdateAll().whenNotMatchedInsertAll().execute())

n_out = sum(1 for r in out_rows if r[7])
for r in out_rows:
    flag = "  ⚠ OUTLIER (z will be withheld)" if r[7] else ""
    print(f"  {r[0]}: MSP={r[1]}  p_all={r[2]:.3g}  covered={r[3]}/{r[4]}{flag}")
print(f"MERGE-upserted {len(out_rows)} sample(s) → {catalog}.{schema}.sample_ancestry ({n_out} flagged outlier)")
