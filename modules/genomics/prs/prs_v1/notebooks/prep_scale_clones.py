# Databricks notebook source
# MAGIC %md
# MAGIC # Prep — 10 header-rewritten clones of Amy's gVCF (scale ladder rung 1)
# MAGIC
# MAGIC Copies a source gVCF N times into `dest_dir`, renames each file, and rewrites the
# MAGIC VCF sample header so extract sees distinct `sample_id`s (`amy_01` … `amy_10`).
# MAGIC Genotypes are identical — this is an I/O / task-fanout / MERGE stress test, not biology.
# MAGIC
# MAGIC Uses `bcftools reheader -s` (header-only; sample names live only on the `#CHROM` line).
# MAGIC Then `bcftools index -t` for a fresh `.tbi`.
# MAGIC
# MAGIC **bcftools:** no runtime micromamba/conda. Stage a pre-built
# MAGIC `bcftools=1.21` + `htslib=1.21` linux-x86_64 prefix tar.gz on the UC libraries
# MAGIC Volume and pass its path via `bcftools_archive_path` (widgets cannot interpolate
# MAGIC `${catalog}`).
# MAGIC
# MAGIC **pysam:** pinned `pysam==0.22.1` — not installed in this notebook. Supply via
# MAGIC cluster/task `libraries:` when job-wired; interactive clusters must already have it.

# COMMAND ----------

dbutils.widgets.text("source_vcf", "/Volumes/dev_exploration_sandbox/genesis_workbench/prs_data/incoming/amy.vcf.gz", "Source gVCF (.vcf.gz)")
dbutils.widgets.text("dest_dir", "/Volumes/dev_exploration_sandbox/genesis_workbench/prs_data/incoming/scale10", "Destination directory")
dbutils.widgets.text("n_clones", "10", "Number of clones")
dbutils.widgets.text("id_prefix", "amy", "Sample/file ID prefix → {prefix}_01 …")
dbutils.widgets.text("overwrite", "true", "Overwrite existing clone files")
# Empty default: catalog/schema cannot be interpolated in notebook widgets — operator must set
# e.g. /Volumes/<catalog>/<schema>/libraries/bcftools-1.21-htslib-1.21-linux-x86_64.tar.gz
dbutils.widgets.text(
    "bcftools_archive_path",
    "",
    "UC Volume path to pinned bcftools=1.21+htslib=1.21 linux-x86_64 tar.gz (required)",
)

source_vcf = dbutils.widgets.get("source_vcf").strip()
dest_dir = dbutils.widgets.get("dest_dir").rstrip("/")
n_clones = int(dbutils.widgets.get("n_clones"))
id_prefix = dbutils.widgets.get("id_prefix").strip() or "amy"
overwrite = dbutils.widgets.get("overwrite").strip().lower() == "true"
bcftools_archive_path = dbutils.widgets.get("bcftools_archive_path").strip()

assert n_clones >= 1 and n_clones <= 1000, "n_clones out of sane range"
assert source_vcf.endswith(".vcf.gz") or source_vcf.endswith(".g.vcf.gz")
assert bcftools_archive_path, (
    "bcftools_archive_path is required. Stage a pre-approved "
    "bcftools-1.21-htslib-1.21-linux-x86_64.tar.gz on the core libraries Volume "
    "(offline: micromamba create -p … bcftools=1.21 htslib=1.21; tar czf …) and set "
    "the widget to that /Volumes/... path. Widgets cannot interpolate catalog/schema."
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 1. Stage pinned bcftools from UC Volume → /local_disk0
# MAGIC
# MAGIC Expects a pre-built conda-prefix tarball (`bcftools=1.21`, `htslib=1.21`, linux-x86_64).
# MAGIC Extract once per node; refuse path-traversal members; verify version.

# COMMAND ----------

import os
import shutil
import subprocess
import tarfile
from pathlib import Path

# Dependency pin (not installed here): pysam==0.22.1 — must be present on the cluster
# (future job task libraries: - pypi: package: pysam==0.22.1). No %pip at runtime.

source_vcf = dbutils.widgets.get("source_vcf").strip()
dest_dir = dbutils.widgets.get("dest_dir").rstrip("/")
n_clones = int(dbutils.widgets.get("n_clones"))
id_prefix = dbutils.widgets.get("id_prefix").strip() or "amy"
overwrite = dbutils.widgets.get("overwrite").strip().lower() == "true"
bcftools_archive_path = dbutils.widgets.get("bcftools_archive_path").strip()

BCFTOOLS_PREFIX = Path("/local_disk0/bcftools")
BCFTOOLS = str(BCFTOOLS_PREFIX / "bin" / "bcftools")
_LOCAL_ARCHIVE = Path("/local_disk0/bcftools-1.21-htslib-1.21-linux-x86_64.tar.gz")
_EXPECTED_VERSION_PREFIX = "1.21"


def _assert_safe_tar_members(tar: tarfile.TarFile, dest: Path) -> None:
    """Reject absolute paths, .. components, and resolved paths that escape dest."""
    dest_resolved = dest.resolve()
    for member in tar.getmembers():
        name = member.name
        if not name or name.startswith("/") or name.startswith("\\"):
            raise ValueError(f"unsafe absolute/empty archive member path: {name!r}")
        # Normalize separators; reject any .. path segment
        parts = Path(name.replace("\\", "/")).parts
        if ".." in parts:
            raise ValueError(f"path traversal in archive member: {name!r}")
        if member.issym() or member.islnk():
            link_parts = Path(member.linkname.replace("\\", "/")).parts
            if (
                not member.linkname
                or member.linkname.startswith(("/", "\\"))
                or ".." in link_parts
            ):
                raise ValueError(
                    f"unsafe archive link target: {name!r} -> {member.linkname!r}"
                )
        target = (dest / name).resolve()
        try:
            target.relative_to(dest_resolved)
        except ValueError as exc:
            raise ValueError(f"archive member escapes extract dir: {name!r}") from exc


def _bcftools_version_ok(bcftools_bin: str) -> str:
    first = subprocess.check_output([bcftools_bin, "--version"], text=True).splitlines()[0]
    # e.g. "bcftools 1.21"
    tokens = first.split()
    ver = tokens[1] if len(tokens) >= 2 and tokens[0].lower() == "bcftools" else tokens[-1]
    if not ver.startswith(_EXPECTED_VERSION_PREFIX):
        raise RuntimeError(
            f"bcftools version must begin with {_EXPECTED_VERSION_PREFIX!r}; got {first!r}"
        )
    return first


def ensure_bcftools_from_volume_archive(archive_path: str) -> str:
    """Copy pinned prefix tarball from UC Volume → /local_disk0 and extract once per node."""
    if os.path.isfile(BCFTOOLS):
        line = _bcftools_version_ok(BCFTOOLS)
        print(f"bcftools already staged: {line}")
        return BCFTOOLS

    if not archive_path:
        raise FileNotFoundError(
            "bcftools_archive_path is empty — set the widget to the UC Volume tar.gz path"
        )
    if not os.path.isfile(archive_path):
        raise FileNotFoundError(
            f"bcftools archive absent at {archive_path!r}. "
            "Upload a pre-approved linux-x86_64 prefix tarball "
            "(bcftools=1.21 + htslib=1.21; no install-time scripts) to the libraries Volume "
            "and pass that path via bcftools_archive_path."
        )

    os.makedirs("/local_disk0", exist_ok=True)
    print(f"copying {archive_path} → {_LOCAL_ARCHIVE}")
    shutil.copy2(archive_path, _LOCAL_ARCHIVE)

    BCFTOOLS_PREFIX.mkdir(parents=True, exist_ok=True)
    with tarfile.open(_LOCAL_ARCHIVE, "r:gz") as tar:
        _assert_safe_tar_members(tar, BCFTOOLS_PREFIX)
        tar.extractall(path=BCFTOOLS_PREFIX)

    if not os.path.isfile(BCFTOOLS):
        raise FileNotFoundError(
            f"bcftools missing after extract at {BCFTOOLS} "
            f"(archive layout should be a conda-style prefix with bin/bcftools)"
        )
    line = _bcftools_version_ok(BCFTOOLS)
    print(f"bcftools: {line}")
    return BCFTOOLS


BCFTOOLS = ensure_bcftools_from_volume_archive(bcftools_archive_path)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2. Peek source sample id + stage dest dir

# COMMAND ----------

import pysam

assert os.path.exists(source_vcf), f"missing source: {source_vcf}"
src_tbi = source_vcf + ".tbi"
if not os.path.exists(src_tbi):
    print(f"WARNING: no .tbi next to source ({src_tbi}); extract still works (full scan), index clones after reheader")

with pysam.VariantFile(source_vcf) as vf:
    samples = list(vf.header.samples)
    assert len(samples) == 1, f"expected single-sample gVCF, got {samples}"
    old_sid = samples[0]
    # Flavor check (informational)
    hdr = str(vf.header)
    is_gatk = "<NON_REF>" in hdr or "ID=NON_REF" in hdr
    print(f"source sample_id={old_sid!r}  gatk_NON_REF={is_gatk}  path={source_vcf}")
    print(f"source size={os.path.getsize(source_vcf):,} bytes")

dbutils.fs.mkdirs(dest_dir)
print("dest:", dest_dir)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3. Copy → reheader → index for each clone

# COMMAND ----------

width = max(2, len(str(n_clones)))
results = []

for i in range(1, n_clones + 1):
    new_sid = f"{id_prefix}_{i:0{width}d}"
    out_vcf = f"{dest_dir}/{new_sid}.vcf.gz"
    out_tbi = out_vcf + ".tbi"
    if (not overwrite) and os.path.exists(out_vcf) and os.path.exists(out_tbi):
        print(f"skip existing {new_sid}")
        results.append((new_sid, out_vcf, "skipped"))
        continue

    # samples file for bcftools reheader -s : "old_name new_name"
    samp_map = f"/tmp/reheader_{new_sid}.txt"
    with open(samp_map, "w") as f:
        f.write(f"{old_sid} {new_sid}\n")

    tmp_out = f"/tmp/{new_sid}.vcf.gz"
    cmd = [BCFTOOLS, "reheader", "-s", samp_map, "-o", tmp_out, source_vcf]
    print(f"[{i}/{n_clones}] reheader {old_sid} → {new_sid}")
    subprocess.check_call(cmd)
    subprocess.check_call([BCFTOOLS, "index", "-t", "-f", tmp_out])

    # Move onto the Volume (FUSE path)
    shutil.copyfile(tmp_out, out_vcf)
    shutil.copyfile(tmp_out + ".tbi", out_tbi)
    os.remove(tmp_out)
    os.remove(tmp_out + ".tbi")
    os.remove(samp_map)

    # Verify
    with pysam.VariantFile(out_vcf) as vf:
        got = list(vf.header.samples)
    assert got == [new_sid], f"header mismatch for {out_vcf}: {got}"
    results.append((new_sid, out_vcf, "ok"))
    print(f"  wrote {out_vcf} ({os.path.getsize(out_vcf):,} B) sample={got}")

print(f"\nDone: {len(results)} clones in {dest_dir}")
for sid, path, st in results:
    print(f"  {st:8s}  {sid}  →  {path}")

# Paths list for prs_scoring vcf_paths widget
vcf_paths = ",".join(p for _, p, _ in results)
print("\nvcf_paths=")
print(vcf_paths)
dbutils.notebook.exit(vcf_paths)
