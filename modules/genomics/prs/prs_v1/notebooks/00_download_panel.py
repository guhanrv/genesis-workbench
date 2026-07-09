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
dbutils.widgets.text("url", "https://ftp.ebi.ac.uk/pub/databases/spot/pgs/reference/pgsc_HGDP+1kGP_v1.tar.zst", "Panel tar.zst URL")

# COMMAND ----------
# MAGIC %pip install zstandard
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

import os, shutil, subprocess, time
import zstandard, tarfile

dest = dbutils.widgets.get("dest_volume_dir").rstrip("/")
url = dbutils.widgets.get("url")
assert dest, "set dest_volume_dir"
WANT = ("GRCh38_HGDP+1kGP_ALL.pgen", "GRCh38_HGDP+1kGP_ALL.psam")   # pgenlib needs the .pgen; psam for metadata
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

# 3. copy the panel files to the Volume via FUSE (sequential write — handles large files, unlike the Files API)
os.makedirs(dest, exist_ok=True)
for base in WANT:
    src = os.path.join(local, base); dst = os.path.join(dest, base)
    t = time.time(); shutil.copyfile(src, dst)
    print(f"  {base}: {os.path.getsize(dst)/1e9:.2f} GB → {dst}  ({time.time()-t:.0f}s)")

print("panel staged. Files on Volume:")
for f in dbutils.fs.ls(dest):
    print(f"  {f.name}  {f.size/1e9:.2f} GB")
