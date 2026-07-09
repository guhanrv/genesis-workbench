# Databricks notebook source
# MAGIC %md
# MAGIC # Build the frozen FRAPOSA PCA basis (one-time reference; classic; NOT serverless)
# MAGIC
# MAGIC Produces `pca_basis.npz` — the fixed HGDP+1kGP PC basis that `05_build_sample_ancestry` projects
# MAGIC members onto. This is the ancestry analogue of `pgs_panel_ref`/`pgs_panel_afreq`: a **one-time,
# MAGIC self-regenerable reference artifact** built from the panel on the Volume (nothing depends on a local
# MAGIC file). Mirrors pgsc_calc's reference build and `function_pca` bit-for-bit (the fit is the same numpy
# MAGIC `prs_ancestry.fit_panel_basis` used in the validated local run: 47k loci, RF self-acc 1.0, LOO 60/60).
# MAGIC
# MAGIC Pipeline (all on the driver — this is a fixed reference, not per-sample work):
# MAGIC   1. plink2 QC the panel (pgsc_calc FILTER_VARIANTS defaults + `--remove king.cutoff` ⇒ the survivors
# MAGIC      are the unrelated training set, so `sample_ancestry` needs no separate unrelated mask)
# MAGIC   2. plink2 LD-prune (`--indep-pairwise 1000 50 0.05 --exclude range high-LD`) → fixed PCA loci
# MAGIC   3. drop palindromic SNVs (pgsc_calc PCA_ELIGIBLE; also keeps panel/member reads consistent — the
# MAGIC      member gVCF kernel drops palindromic too)
# MAGIC   4. pgenlib reads panel ALT-dose at the loci → FRAPOSA fit → save loci + U/s/V/pcs_ref/mean/std +
# MAGIC      panel IIDs + SuperPop labels to the Volume
# MAGIC
# MAGIC **Requires plink2 (v2.00a5+) on the runner.** Glow can't QC/LD-prune, so this classic step needs the
# MAGIC plink2 binary staged on the cluster (`plink2_path` widget). If plink2 isn't available, run this once
# MAGIC locally and stage the npz — the artifact is identical (the fit is the shared numpy lib).

# COMMAND ----------

dbutils.widgets.text("catalog", "genesis_workbench", "Catalog")
dbutils.widgets.text("schema", "genesis_schema", "Schema")
dbutils.widgets.text("panel_dir", "", "Volume dir with GRCh38_HGDP+1kGP_ALL.{pgen,pvar.zst,psam} + king.cutoff")
dbutils.widgets.text("high_ld_path", "", "high-LD-regions-hg38-GRCh38.txt (Volume/workspace path)")
dbutils.widgets.text("out_path", "", "Volume path to write pca_basis.npz")
dbutils.widgets.text("plink2_path", "plink2", "plink2 binary (v2.00a5+)")
dbutils.widgets.text("dim_ref", "10", "Reference PC dimension (classify uses first 5)")
dbutils.widgets.text("panel_version", "pgsc_HGDP+1kGP_v1", "Basis provenance tag")

catalog = dbutils.widgets.get("catalog"); schema = dbutils.widgets.get("schema")
panel_dir = dbutils.widgets.get("panel_dir").rstrip("/")
high_ld_path = dbutils.widgets.get("high_ld_path").strip()
out_path = dbutils.widgets.get("out_path").strip()
plink2 = dbutils.widgets.get("plink2_path").strip()
dim_ref = int(dbutils.widgets.get("dim_ref"))
panel_version = dbutils.widgets.get("panel_version").strip()

# COMMAND ----------

# MAGIC %pip install pgenlib zstandard
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

import os, subprocess, time, gzip
import numpy as np

catalog = dbutils.widgets.get("catalog"); schema = dbutils.widgets.get("schema")
panel_dir = dbutils.widgets.get("panel_dir").rstrip("/")
high_ld_path = dbutils.widgets.get("high_ld_path").strip()
out_path = dbutils.widgets.get("out_path").strip()
plink2 = dbutils.widgets.get("plink2_path").strip()
dim_ref = int(dbutils.widgets.get("dim_ref"))
panel_version = dbutils.widgets.get("panel_version").strip()
assert panel_dir and out_path and high_ld_path, "panel_dir, high_ld_path, out_path are required"

lib_dir = os.path.abspath(os.path.join(os.getcwd(), "..", "lib"))
import sys; sys.path.append(lib_dir)
import prs_ancestry as anc

# ── inlined driver-only readers (faithful copies from function_pca.refit; kept OUT of the shared
#    prs_ancestry lib so that lib stays numpy-only-importable on executors — pgenlib lives only here) ──

def _read_text(path):
    if path.endswith(".gz"):
        with gzip.open(path, "rt") as f: return f.read()
    if path.endswith(".zst"):
        import zstandard
        with open(path, "rb") as f:
            return zstandard.ZstdDecompressor().decompress(f.read(), max_output_size=10 << 30).decode()
    with open(path, "rt") as f: return f.read()

def _read_pgen_subset_by_id(pgen_prefix, variant_ids):
    """Read named variants (by ID, caller order) from a pgen → (sample_ids, X (n_var, n_sam) ALT-dose,
    NaN=missing). Direct port of function_pca.refit._read_pgen_subset_by_id."""
    from pgenlib import PgenReader
    psam_path = pgen_prefix + ".psam"
    pvar_path = pgen_prefix + ".pvar" if os.path.exists(pgen_prefix + ".pvar") else pgen_prefix + ".pvar.zst"
    sample_ids = []
    with open(psam_path) as f:
        header = f.readline().lstrip("#").rstrip("\n").split("\t"); iid_col = header.index("IID")
        for line in f:
            if line.strip(): sample_ids.append(line.rstrip("\n").split("\t")[iid_col])
    id_to_idx = {}; id_col = 2; pgen_idx = 0
    for line in _read_text(pvar_path).splitlines():
        if line.startswith("##"): continue
        if line.startswith("#"):
            id_col = line.lstrip("#").split("\t").index("ID"); continue
        parts = line.split("\t", id_col + 2); id_to_idx[parts[id_col]] = pgen_idx; pgen_idx += 1
    n_pgen_var = pgen_idx
    indices = np.empty(len(variant_ids), dtype=np.uint32); missing = []
    for i, vid in enumerate(variant_ids):
        idx = id_to_idx.get(vid)
        if idx is None: missing.append(vid)
        else: indices[i] = idx
    if missing: raise KeyError(f"{len(missing)} variant IDs not in {os.path.basename(pvar_path)} (e.g. {missing[:3]})")
    n_sam = len(sample_ids)
    X = np.empty((len(variant_ids), n_sam), dtype=np.float32)
    reader = PgenReader(str(pgen_prefix + ".pgen").encode(), raw_sample_ct=n_sam, variant_ct=n_pgen_var)
    try: reader.read_dosages_list(indices, X, sample_maj=0)
    finally: reader.close()
    X[X < -0.5] = np.nan
    return sample_ids, X

_PAL = ({"A", "T"}, {"C", "G"})
def _palindromic(r, a): return {r.upper(), a.upper()} in _PAL

# COMMAND ----------

# MAGIC %md
# MAGIC ### 1. Stage panel to local NVMe (plink2 + pgenlib want local random-access files)

# COMMAND ----------

local = "/local_disk0/pca_panel"; os.makedirs(local, exist_ok=True)
PANEL = "GRCh38_HGDP+1kGP_ALL"; KING = "GRCh38_HGDP+1kGP.king.cutoff.out.id"
for base in (f"{PANEL}.pgen", f"{PANEL}.pvar.zst", f"{PANEL}.psam", KING):
    dst = os.path.join(local, base)
    if not os.path.exists(dst):
        t = time.time(); dbutils.fs.cp(f"{panel_dir}/{base}", "file:" + dst)
        print(f"  staged {base} ({os.path.getsize(dst)/1e9:.2f} GB) in {time.time()-t:.0f}s")
local_high_ld = "/local_disk0/high-LD-regions.txt"
dbutils.fs.cp(high_ld_path, "file:" + local_high_ld)
panel_prefix = os.path.join(local, PANEL)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2. plink2 QC (pgsc_calc FILTER_VARIANTS + --remove king.cutoff) then LD-prune

# COMMAND ----------

qc = os.path.join(local, "panel_qc")
if not os.path.exists(qc + ".pgen"):
    t = time.time()
    subprocess.run([plink2, "--pfile", panel_prefix, "vzs",
                    "--remove", os.path.join(local, KING),
                    "--max-alleles", "2", "--snps-only", "just-acgt", "--rm-dup", "exclude-all",
                    "--autosome", "--maf", "0.05", "--hwe", "0.0001", "--geno", "0.1", "--mind", "0.1",
                    "--make-pgen", "vzs", "--freq", "zs", "--threads", "8", "--out", qc],
                   check=True, capture_output=True)
    print(f"QC: {time.time()-t:.0f}s")

pruned = os.path.join(local, "panel_pruned")
t = time.time()
subprocess.run([plink2, "--pfile", qc, "vzs", "--indep-pairwise", "1000", "50", "0.05",
                "--exclude", "range", local_high_ld, "--threads", "8", "--out", pruned],
               check=True, capture_output=True)
prune_in = open(pruned + ".prune.in").read().split()
print(f"LD-prune: {time.time()-t:.0f}s | {len(prune_in)} loci")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3. Drop palindromic → read panel ALT-dose → FRAPOSA fit → save to Volume

# COMMAND ----------

id2key = {}
for line in _read_text(qc + ".pvar.zst").splitlines():
    if line.startswith("#"): continue
    p = line.split("\t"); id2key[p[2]] = (p[0], int(p[1]), p[3], p[4])   # #CHROM POS ID REF ALT
loci_ids, loci = [], []
for vid in prune_in:
    c, pos, r, a = id2key[vid]
    if _palindromic(r, a): continue
    loci_ids.append(vid); loci.append((c, pos, r, a))
print(f"loci after palindromic drop: {len(loci)} (dropped {len(prune_in) - len(loci)})")

t = time.time()
panel_iids, X = _read_pgen_subset_by_id(qc, loci_ids)
print(f"panel dose: {X.shape} in {time.time()-t:.0f}s (missing frac {np.isnan(X).mean():.4f})")

# SuperPop labels aligned to panel_iids (from the FULL panel psam)
hdr = open(panel_prefix + ".psam").readline().lstrip("#").rstrip().split("\t")
iid_c, sp_c = hdr.index("IID"), hdr.index("SuperPop")
sp_map = {}
with open(panel_prefix + ".psam") as f:
    f.readline()
    for line in f:
        p = line.rstrip().split("\t"); sp_map[p[iid_c]] = p[sp_c]
superpops = np.array([sp_map[i] for i in panel_iids])
uniq, cnts = np.unique(superpops, return_counts=True)
print(f"panel (post-QC, unrelated): {len(panel_iids)} samples | { {k: int(v) for k, v in zip(uniq, cnts)} }")

basis = anc.fit_panel_basis(X, dim_ref=dim_ref)   # standardizes X in place
# quick self-check: RF training accuracy on the panel PCs (should be ~1.0 if PCs separate superpops)
clf, _cov = anc.fit_rf(basis["pcs_ref"], superpops.tolist(), n_pcs=5)
print(f"RF self-accuracy on panel PCs: {(clf.predict(basis['pcs_ref'][:, :5]) == superpops).mean():.3f}")

# COMMAND ----------

local_npz = "/local_disk0/pca_basis.npz"
np.savez(local_npz,
         loci_chrom=np.array([c for c, _, _, _ in loci]),
         loci_pos=np.array([p for _, p, _, _ in loci], dtype=np.int64),
         loci_ref=np.array([r for _, _, r, _ in loci]),
         loci_alt=np.array([a for _, _, _, a in loci]),
         U_on=basis["U_on"], s_on=basis["s_on"], V_on=basis["V_on"],
         pcs_ref=basis["pcs_ref"], mean=basis["mean"], std=basis["std"],
         dim_ref=basis["dim_ref"], dim_stu=basis["dim_stu"],
         panel_iids=np.array(panel_iids), superpops=superpops,
         panel_version=np.array(panel_version))
dbutils.fs.cp("file:" + local_npz, out_path)
print(f"wrote basis → {out_path} ({os.path.getsize(local_npz)/1e6:.1f} MB) | "
      f"n_loci={len(loci)} n_panel={len(panel_iids)} panel_version={panel_version}")
