"""Pure unit tests for ``lib/scale_manifest.py`` (no Spark / no bcftools)."""
from __future__ import annotations

import hashlib
import os
import sys
import tempfile

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "lib"))
from scale_manifest import (
    MANIFEST_COLUMNS,
    assert_reserved_id_prefix,
    assert_row_estimate_within_limit,
    assert_single_sample_ids,
    assert_synthetic_only,
    assert_unique_nonempty_ids,
    batch_id_for_index,
    build_source_fingerprint_metadata,
    estimate_fanout_rows,
    fingerprint_column_value,
    id_width,
    logical_sample_id,
    parse_vcf_header_sample_ids,
    read_vcf_header_sample_id,
    streaming_file_sha256,
    validate_synthetic_manifest_config,
)


def test_manifest_columns_contract():
    assert MANIFEST_COLUMNS == (
        "sample_id",
        "source_sample_id",
        "vcf_path",
        "source_fingerprint",
        "synthetic",
        "scale_run_id",
        "batch_id",
    )


def test_validate_config_ok():
    cfg = validate_synthetic_manifest_config(
        n_samples=10,
        id_prefix="synth_amy",
        scale_run_id="run_1",
        batch_size=3,
        synthetic=True,
    )
    assert cfg["n_samples"] == 10
    assert cfg["id_prefix"] == "synth_amy"
    assert cfg["synthetic"] is True
    assert cfg["id_width"] == 2
    assert cfg["batch_size"] == 3


def test_validate_config_rejects_production_modality():
    with pytest.raises(ValueError, match="synthetic=false"):
        validate_synthetic_manifest_config(
            n_samples=10,
            id_prefix="synth_amy",
            scale_run_id="run_1",
            batch_size=10,
            synthetic=False,
        )


def test_validate_config_rejects_unreserved_prefix():
    with pytest.raises(ValueError, match="reserved"):
        validate_synthetic_manifest_config(
            n_samples=10,
            id_prefix="member",
            scale_run_id="run_1",
            batch_size=10,
            synthetic=True,
        )


def test_validate_config_rejects_bad_n_and_empty_run():
    with pytest.raises(ValueError, match="n_samples"):
        validate_synthetic_manifest_config(
            n_samples=0,
            id_prefix="synth_amy",
            scale_run_id="r1",
            batch_size=1,
        )
    with pytest.raises(ValueError, match="scale_run_id"):
        validate_synthetic_manifest_config(
            n_samples=1,
            id_prefix="synth_amy",
            scale_run_id="  ",
            batch_size=1,
        )
    with pytest.raises(ValueError, match="exceeds"):
        validate_synthetic_manifest_config(
            n_samples=101,
            id_prefix="scale",
            scale_run_id="r1",
            batch_size=10,
            max_n_samples=100,
        )


def test_assert_synthetic_only():
    assert_synthetic_only(synthetic=True)
    with pytest.raises(ValueError, match="production"):
        assert_synthetic_only(synthetic=False)


def test_reserved_prefixes_are_synthetic_only():
    assert assert_reserved_id_prefix("scale") == "scale"
    assert assert_reserved_id_prefix("synth") == "synth"
    assert assert_reserved_id_prefix("synth_x") == "synth_x"
    with pytest.raises(ValueError, match="reserved"):
        assert_reserved_id_prefix("amy")


def test_deterministic_logical_ids_and_batches():
    assert id_width(10) == 2
    assert id_width(100) == 3
    assert id_width(1000) == 4
    assert logical_sample_id("synth_amy", 1, 10) == "synth_amy_01"
    assert logical_sample_id("synth_amy", 10, 10) == "synth_amy_10"
    assert logical_sample_id("scale_amy", 7, 1000) == "scale_amy_0007"

    assert batch_id_for_index(1, 1000) == "batch_00000"
    assert batch_id_for_index(1000, 1000) == "batch_00000"
    assert batch_id_for_index(1001, 1000) == "batch_00001"
    assert batch_id_for_index(2500, 1000) == "batch_00002"

    ids = [logical_sample_id("scale", i, 5) for i in range(1, 6)]
    assert_unique_nonempty_ids(ids)
    with pytest.raises(ValueError, match="duplicate"):
        assert_unique_nonempty_ids(["a", "a"])
    with pytest.raises(ValueError, match="non-empty"):
        assert_unique_nonempty_ids(["ok", "  "])


def test_row_estimate_guardrails():
    assert estimate_fanout_rows(1_000_000, 10) == 10_000_000
    assert_row_estimate_within_limit(100, max_fanout_rows=100, confirm_large=False)
    with pytest.raises(ValueError, match="REFUSING"):
        assert_row_estimate_within_limit(101, max_fanout_rows=100, confirm_large=False)
    assert_row_estimate_within_limit(101, max_fanout_rows=100, confirm_large=True)


def test_source_fingerprint_metadata():
    digest = "a" * 64
    meta = build_source_fingerprint_metadata(
        vcf_path="/tmp/amy.vcf.gz",
        sha256_hex=digest.upper(),
        size_bytes=12,
        source_sample_id="Amy",
    )
    assert meta["sha256"] == digest
    assert fingerprint_column_value(meta) == digest
    with pytest.raises(ValueError, match="sha256"):
        build_source_fingerprint_metadata(
            vcf_path="/tmp/x.vcf.gz",
            sha256_hex="deadbeef",
            size_bytes=1,
            source_sample_id="s1",
        )


def test_streaming_sha256_and_header_parse():
    payload = b"hello-scale-manifest\n"
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "blob.bin")
        with open(p, "wb") as f:
            f.write(payload)
        assert streaming_file_sha256(p) == hashlib.sha256(payload).hexdigest()

    header = [
        "##fileformat=VCFv4.2\n",
        "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tOnlyOne\n",
    ]
    assert parse_vcf_header_sample_ids(header) == ["OnlyOne"]
    assert assert_single_sample_ids(["OnlyOne"]) == "OnlyOne"
    with pytest.raises(ValueError, match="exactly one"):
        assert_single_sample_ids(["a", "b"])
    with pytest.raises(ValueError, match="#CHROM"):
        parse_vcf_header_sample_ids(["##fileformat=VCFv4.2\n"])


def test_read_plain_vcf_header_sample_id():
    with tempfile.TemporaryDirectory() as td:
        vcf = os.path.join(td, "tiny.vcf")
        with open(vcf, "w", encoding="utf-8") as f:
            f.write(
                "##fileformat=VCFv4.2\n"
                "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tSrcSid\n"
                "chr1\t1\t.\tA\tG\t.\tPASS\t.\tGT\t0/1\n"
            )
        assert read_vcf_header_sample_id(vcf) == "SrcSid"
