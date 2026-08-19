"""Synthetic scale-test sample-manifest helpers (import-light; no Spark / no bcftools).

Production PRS stays 1:1 (header sample_id ↔ unique gVCF). These helpers only support
**synthetic** manifests that map many logical ``sample_id``s to one physical source for
Delta fan-out benchmarks. Callers must refuse production modality and unsafe configs
before writing a Delta path.
"""
from __future__ import annotations

import hashlib
import re
from typing import Any, Iterable, Mapping, Sequence

# Logical IDs must use one of these prefixes so synthetic cohorts cannot collide with
# production member IDs that lack the marker.
RESERVED_ID_PREFIXES: tuple[str, ...] = ("synth_", "scale_")

MANIFEST_COLUMNS: tuple[str, ...] = (
    "sample_id",
    "source_sample_id",
    "vcf_path",
    "source_fingerprint",
    "synthetic",
    "scale_run_id",
    "batch_id",
)

_MAX_N_SAMPLES_HARD = 100_000
_ID_PREFIX_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_]*$")
_SCALE_RUN_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.\-]*$")


def id_width(n_samples: int) -> int:
    """Zero-pad width for 1-based indices (at least 2, matching prep_scale_clones)."""
    if n_samples < 1:
        raise ValueError(f"n_samples must be >= 1, got {n_samples}")
    return max(2, len(str(int(n_samples))))


def logical_sample_id(id_prefix: str, index: int, n_samples: int) -> str:
    """Deterministic logical ID: ``{prefix}_{i:0{width}d}`` (1-based ``index``)."""
    prefix = (id_prefix or "").strip()
    if not prefix:
        raise ValueError("id_prefix must be non-empty")
    if index < 1 or index > n_samples:
        raise ValueError(f"index must be in [1, {n_samples}], got {index}")
    return f"{prefix}_{index:0{id_width(n_samples)}d}"


def batch_id_for_index(index: int, batch_size: int) -> str:
    """Deterministic batch id for a 1-based sample index: ``batch_{k:05d}`` (0-based k)."""
    if index < 1:
        raise ValueError(f"index must be >= 1, got {index}")
    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}")
    k = (index - 1) // int(batch_size)
    return f"batch_{k:05d}"


def assert_unique_nonempty_ids(sample_ids: Sequence[str]) -> None:
    """Fail if any logical id is empty/blank or duplicated."""
    seen: set[str] = set()
    for sid in sample_ids:
        s = (sid or "").strip()
        if not s:
            raise ValueError("logical sample_id must be non-empty")
        if s in seen:
            raise ValueError(f"duplicate logical sample_id: {s!r}")
        seen.add(s)


def assert_reserved_id_prefix(
    id_prefix: str,
    *,
    reserved_prefixes: Sequence[str] = RESERVED_ID_PREFIXES,
) -> str:
    """Require a reserved synthetic prefix for isolation from production member IDs."""
    prefix = (id_prefix or "").strip()
    if not prefix:
        raise ValueError("id_prefix must be non-empty")
    if not _ID_PREFIX_RE.match(prefix):
        raise ValueError(
            f"id_prefix {prefix!r} must match {_ID_PREFIX_RE.pattern} "
            "(letter start, alnum/underscore only)"
        )
    # Bare markers (synth / scale) or longer names that start with reserved_
    # (e.g. synth_amy, scale_run1).
    bare = {p.rstrip("_") for p in reserved_prefixes}
    if prefix in bare or any(prefix.startswith(p) for p in reserved_prefixes):
        return prefix
    raise ValueError(
        f"id_prefix {prefix!r} must start with a reserved synthetic marker "
        f"{tuple(reserved_prefixes)} for production isolation"
    )


def assert_synthetic_only(*, synthetic: bool) -> None:
    """Refuse production modality — this generator never writes synthetic=false rows."""
    if not synthetic:
        raise ValueError(
            "REFUSING: production modality (synthetic=false). "
            "Many logical IDs may share one gVCF only under synthetic=true scale manifests."
        )


def validate_synthetic_manifest_config(
    *,
    n_samples: int,
    id_prefix: str,
    scale_run_id: str,
    batch_size: int,
    synthetic: bool = True,
    max_n_samples: int = _MAX_N_SAMPLES_HARD,
    reserved_prefixes: Sequence[str] = RESERVED_ID_PREFIXES,
) -> dict[str, Any]:
    """Validate and normalize config for a synthetic-only scale manifest.

    Returns a dict with normalized ``n_samples``, ``id_prefix``, ``scale_run_id``,
    ``batch_size``, ``synthetic``, ``id_width``. Raises ``ValueError`` on unsafe /
    production conditions.
    """
    assert_synthetic_only(synthetic=synthetic)

    n = int(n_samples)
    if n < 1:
        raise ValueError(f"n_samples must be >= 1, got {n}")
    hard = int(max_n_samples)
    if hard < 1 or hard > _MAX_N_SAMPLES_HARD:
        raise ValueError(f"max_n_samples must be in [1, {_MAX_N_SAMPLES_HARD}], got {hard}")
    if n > hard:
        raise ValueError(f"n_samples={n} exceeds max_n_samples={hard}")

    prefix = assert_reserved_id_prefix(id_prefix, reserved_prefixes=reserved_prefixes)

    run_id = (scale_run_id or "").strip()
    if not run_id:
        raise ValueError("scale_run_id must be non-empty")
    if not _SCALE_RUN_RE.match(run_id):
        raise ValueError(
            f"scale_run_id {run_id!r} must match {_SCALE_RUN_RE.pattern}"
        )

    bs = int(batch_size)
    if bs < 1:
        raise ValueError(f"batch_size must be >= 1, got {bs}")
    if bs > n:
        # Still valid (one batch); normalize is fine to leave as-is.
        pass

    return {
        "n_samples": n,
        "id_prefix": prefix,
        "scale_run_id": run_id,
        "batch_size": bs,
        "synthetic": True,
        "id_width": id_width(n),
    }


def estimate_fanout_rows(canonical_variant_rows: int, n_pending_samples: int) -> int:
    """Estimated dosage rows = canonical variants × pending logical samples."""
    v = int(canonical_variant_rows)
    s = int(n_pending_samples)
    if v < 0 or s < 0:
        raise ValueError(
            f"canonical_variant_rows and n_pending_samples must be >= 0, got {v}, {s}"
        )
    return v * s


def assert_row_estimate_within_limit(
    estimated_rows: int,
    max_fanout_rows: int,
    *,
    confirm_large: bool = False,
) -> None:
    """Refuse oversized fan-out unless ``confirm_large=true`` (mirrors reconcile guardrails)."""
    est = int(estimated_rows)
    cap = int(max_fanout_rows)
    if cap < 1:
        raise ValueError(f"max_fanout_rows must be >= 1, got {cap}")
    if est < 0:
        raise ValueError(f"estimated_rows must be >= 0, got {est}")
    if est > cap and not confirm_large:
        raise ValueError(
            f"REFUSING: estimated fan-out rows {est} > max_fanout_rows={cap}. "
            f"To proceed intentionally, set confirm_large=true."
        )


def build_source_fingerprint_metadata(
    *,
    vcf_path: str,
    sha256_hex: str,
    size_bytes: int,
    source_sample_id: str,
) -> dict[str, Any]:
    """Metadata for the single physical source behind a synthetic manifest."""
    path = (vcf_path or "").strip()
    if not path:
        raise ValueError("vcf_path must be non-empty")
    digest = (sha256_hex or "").strip().lower()
    if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
        raise ValueError(f"sha256_hex must be 64 lowercase hex chars, got {sha256_hex!r}")
    sid = (source_sample_id or "").strip()
    if not sid:
        raise ValueError("source_sample_id must be non-empty")
    sz = int(size_bytes)
    if sz < 0:
        raise ValueError(f"size_bytes must be >= 0, got {sz}")
    return {
        "vcf_path": path,
        "sha256": digest,
        "size_bytes": sz,
        "source_sample_id": sid,
    }


def fingerprint_column_value(meta: Mapping[str, Any]) -> str:
    """Value written to the manifest ``source_fingerprint`` column (content sha256)."""
    return str(meta["sha256"])


def streaming_file_sha256(path: str, *, chunk_size: int = 1 << 20) -> str:
    """Stream file bytes once; return lowercase hex digest. No bcftools."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def parse_vcf_header_sample_ids(header_lines: Iterable[str]) -> list[str]:
    """Extract sample IDs from VCF header text lines (through ``#CHROM``). Pure."""
    chrom_line = None
    for line in header_lines:
        if line.startswith("#CHROM"):
            chrom_line = line.rstrip("\n\r")
            break
    if chrom_line is None:
        raise ValueError("VCF header has no #CHROM line")
    parts = chrom_line.split("\t")
    if len(parts) < 10:
        raise ValueError(
            f"VCF #CHROM line has no sample columns (got {len(parts)} fields)"
        )
    return [p for p in parts[9:] if p != ""]


def assert_single_sample_ids(sample_ids: Sequence[str]) -> str:
    """Require exactly one sample in the VCF header; return it."""
    ids = list(sample_ids)
    if len(ids) != 1:
        raise ValueError(
            f"expected exactly one sample in VCF header, found {len(ids)} ({ids[:5]!r})"
        )
    sid = (ids[0] or "").strip()
    if not sid:
        raise ValueError("VCF header sample_id is empty")
    return sid


def read_vcf_header_sample_id(path: str) -> str:
    """Read gzip-or-plain VCF header until ``#CHROM``; return the single sample id.

    Uses only the stdlib (gzip) — no bcftools / pysam required for the contract check.
    """
    import gzip

    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8", errors="replace") as f:
        samples = parse_vcf_header_sample_ids(f)
    return assert_single_sample_ids(samples)
