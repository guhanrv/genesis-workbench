"""Path-safety checks for the bcftools Volume-archive extract in prep_scale_clones."""
from __future__ import annotations

import io
import os
import tarfile
import tempfile
import unittest
from pathlib import Path

NOTEBOOK = Path(__file__).resolve().parents[1] / "notebooks" / "prep_scale_clones.py"


def _load_assert_safe_tar_members():
    """Exec only `_assert_safe_tar_members` from the notebook (no dbutils)."""
    text = NOTEBOOK.read_text()
    start = text.index("def _assert_safe_tar_members(")
    rest = text[start + 1 :]
    rel = rest.index("\ndef ")
    chunk = text[start : start + 1 + rel]
    ns: dict = {"tarfile": tarfile, "Path": Path, "os": os}
    exec(chunk, ns)  # noqa: S102 — load helper under test from sibling notebook
    return ns["_assert_safe_tar_members"]


def _tar_with_members(names_and_data: list[tuple[str, bytes]]) -> tarfile.TarFile:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for name, data in names_and_data:
            info = tarfile.TarInfo(name=name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    buf.seek(0)
    return tarfile.open(fileobj=buf, mode="r:gz")


class TestBcftoolsArchivePathSafety(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._check_members = staticmethod(_load_assert_safe_tar_members())

    def test_accepts_normal_prefix_layout(self):
        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp)
            with _tar_with_members([("bin/bcftools", b"x"), ("lib/libhts.so", b"y")]) as tar:
                self._check_members(tar, dest)

    def test_rejects_dotdot_member(self):
        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp)
            with _tar_with_members([("../evil", b"x")]) as tar:
                with self.assertRaisesRegex(ValueError, "path traversal"):
                    self._check_members(tar, dest)

    def test_rejects_absolute_member(self):
        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp)
            with _tar_with_members([("/etc/passwd", b"x")]) as tar:
                with self.assertRaisesRegex(ValueError, "unsafe absolute"):
                    self._check_members(tar, dest)

    def test_rejects_escaping_symlink(self):
        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w:gz") as tar:
            info = tarfile.TarInfo(name="lib/libhts.so")
            info.type = tarfile.SYMTYPE
            info.linkname = "../../outside"
            tar.addfile(info)
        buf.seek(0)
        with tempfile.TemporaryDirectory() as tmp:
            with tarfile.open(fileobj=buf, mode="r:gz") as tar:
                with self.assertRaisesRegex(ValueError, "unsafe archive link"):
                    self._check_members(tar, Path(tmp))


if __name__ == "__main__":
    unittest.main()
