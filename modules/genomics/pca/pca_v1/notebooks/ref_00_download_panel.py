# Databricks notebook source
# MAGIC %md
# MAGIC # Download the pgsc HGDP+1kGP reference panel → Volume (self-regenerable, on-cluster)
# MAGIC
# MAGIC Fetches the PGS Catalog reference panel and stages the GRCh38 pgen+psam to a Volume, so the panel
# MAGIC allele-freq store (`04_build_panel_afreq`) is regenerable entirely on the workspace — no 12 GB
# MAGIC local upload. Source: `https://ftp.ebi.ac.uk/pub/databases/spot/pgs/reference/pgsc_HGDP+1kGP_v1.tar.zst`
# MAGIC (gnomAD-derived HGDP+1kGP, processed by the PGS Catalog; same panel that produced the frozen
# MAGIC reference distributions). Decompress via Python `zstandard`+`tarfile` (no zstd binary needed).
# MAGIC Classic cluster with local disk + outbound internet.

# COMMAND ----------

dbutils.widgets.text("dest_volume_dir", "", "Volume dir to stage the panel into")
dbutils.widgets.text("url", "https://ftp.ebi.ac.uk/pub/databases/spot/pgs/resources/pgsc_HGDP+1kGP_v1.tar.zst", "Panel tar.zst URL")

# COMMAND ----------
# MAGIC %pip install zstandard
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

import os, shutil, subprocess, time
import zstandard, tarfile

dest = dbutils.widgets.get("dest_volume_dir").rstrip("/")
url = dbutils.widgets.get("url")
assert dest, "set dest_volume_dir"
# Extract pgen (pgenlib genotypes) + psam (metadata incl. SuperPop labels) + pvar.zst (variant table) +
# king.cutoff (related-sample exclude list). king.cutoff is staged to the Volume so the Spark-native
# PCA-ancestry basis (ref_01_build_basis excludes related samples via king.cutoff) is regenerable
# server-side with nothing depending on a local file. pvar.zst is extracted only to DERIVE pvar.parquet
# (step 3) — the parquet is what the builders consume, so pvar.zst itself is NOT staged to the Volume.
WANT = ("GRCh38_HGDP+1kGP_ALL.pgen", "GRCh38_HGDP+1kGP_ALL.psam", "GRCh38_HGDP+1kGP_ALL.pvar.zst",
        "GRCh38_HGDP+1kGP.king.cutoff.out.id")
TO_VOLUME = ("GRCh38_HGDP+1kGP_ALL.pgen", "GRCh38_HGDP+1kGP_ALL.psam",
             "GRCh38_HGDP+1kGP.king.cutoff.out.id")   # + pvar.parquet built below (pvar.zst stays local)
local = "/local_disk0/panel"; os.makedirs(local, exist_ok=True)
archive = "/local_disk0/pgsc_panel.tar.zst"

# COMMAND ----------

# 1. download the archive to local disk (resumable curl; ~15 GB)
if not os.path.exists(archive):
    t = time.time()
    print(f"downloading {url} → {archive} …")
    subprocess.run(["curl", "-L", "--fail", "--retry", "5", "--retry-delay", "10", "-C", "-",
                    "-o", archive, url], check=True)
    print(f"downloaded {os.path.getsize(archive)/1e9:.1f} GB in {time.time()-t:.0f}s")

# COMMAND ----------

# 2. stream-decompress (zstd) + extract ONLY the GRCh38 pgen+psam (streaming tar → low memory)
t = time.time()
with open(archive, "rb") as fh:
    reader = zstandard.ZstdDecompressor().stream_reader(fh)
    with tarfile.open(fileobj=reader, mode="r|") as tar:
        for m in tar:
            base = os.path.basename(m.name)
            if base in WANT and m.isfile():
                print(f"  extracting {base} ({m.size/1e9:.1f} GB)")
                with tar.extractfile(m) as src, open(os.path.join(local, base), "wb") as out:
                    shutil.copyfileobj(src, out, length=1 << 24)
print(f"extracted in {time.time()-t:.0f}s: {os.listdir(local)}")

# COMMAND ----------

# 3. derive pvar.parquet SERVER-SIDE from the extracted .pvar.zst (index-map for the afreq builder;
#    reproduces scripts/10_validate/calibrate_prs_with_1000g.load_or_cache_pvar exactly → CHROM,POS,REF,ALT
#    in pgen order, CHROM without 'chr'). No local upload — fully self-regenerable from the pgsc source.
import pandas as pd
pvar_zst = os.path.join(local, "GRCh38_HGDP+1kGP_ALL.pvar.zst")
pvar_tsv = os.path.join(local, "GRCh38_HGDP+1kGP_ALL.pvar")
with open(pvar_zst, "rb") as fi, open(pvar_tsv, "wb") as fo:
    zstandard.ZstdDecompressor().copy_stream(fi, fo, write_size=1 << 24)
header_lineno = None
with open(pvar_tsv) as f:
    for n, line in enumerate(f):
        if line.startswith("#CHROM"):
            header_lineno = n; break
assert header_lineno is not None, "no #CHROM header in pvar"
pvar = pd.read_csv(pvar_tsv, sep="\t", skiprows=header_lineno,
                   usecols=["#CHROM", "POS", "REF", "ALT"],
                   dtype={"#CHROM": "string", "POS": "int64", "REF": "string", "ALT": "string"},
                   low_memory=False).rename(columns={"#CHROM": "CHROM"})
pvar["CHROM"] = pvar["CHROM"].astype(str).str.replace(r"^chr", "", regex=True)
parquet_local = os.path.join(local, "GRCh38_HGDP+1kGP_ALL.pvar.parquet")
pvar.to_parquet(parquet_local, index=False)
print(f"built pvar.parquet server-side: {len(pvar):,} variants")

# 4. copy pgen + psam + the derived parquet to the Volume via dbutils.fs.cp (reliable for large
#    files → Volumes; plain shutil.copyfile over the FUSE mount stalls on the 12 GB pgen).
os.makedirs(dest, exist_ok=True)
for base in list(TO_VOLUME) + ["GRCh38_HGDP+1kGP_ALL.pvar.parquet"]:
    src = os.path.join(local, base); dst = os.path.join(dest, base)
    t = time.time()
    dbutils.fs.cp("file:" + src, dst)
    print(f"  {base}: {os.path.getsize(src)/1e9:.2f} GB → {dst}  ({time.time()-t:.0f}s)")

print("panel staged. Files on Volume:")
for f in dbutils.fs.ls(dest):
    print(f"  {f.name}  {f.size/1e9:.2f} GB")
