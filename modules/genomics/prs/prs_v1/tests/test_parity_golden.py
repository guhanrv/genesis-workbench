"""Golden-value PARITY tests (scaffold) — plink2 raw-score parity + FRAPOSA basis parity.

The scoring kernel and the ancestry fit are justified throughout by parity with external tools
(plink2 --score / pgsc_calc, FRAPOSA), but that parity is currently UNVERIFIED — there are no committed
golden fixtures. These tests are the concrete slot: they run and assert once the fixtures below are
committed, and skip (loudly) until then. This makes the parity claim enforceable instead of aspirational.

Fixtures to commit under ``tests/fixtures/`` (small, public/synthetic data ONLY — never member gVCFs):

  raw-score parity (plink2):
    - ``fixtures/tiny.g.vcf.gz`` (+ .tbi)   a small single-sample GRCh38 gVCF (public/synthetic)
    - ``fixtures/tiny.scorefile.txt``        a small PGS scorefile (chr, pos, effect, other, weight)
    - ``fixtures/tiny.sscore``               plink2 output to reproduce, generated with e.g.:
          plink2 --vcf tiny.g.vcf.gz dosage=DS --score tiny.scorefile.txt 1 2 3 header cols=+scoresums
    The test loads the scorefile + gVCF, runs the SAME dose kernel (lib/gvcf_dose + lib/prs_extract),
    and asserts the summed raw score equals the plink2 .sscore SCORE1_SUM within a tight tolerance.

  FRAPOSA basis/projection parity:
    - ``fixtures/golden_basis.npz``          a basis built by the reference FRAPOSA engine on a small panel
    - ``fixtures/golden_projection.json``    expected PCs for a held-out sample projected onto that basis
    The test asserts lib/prs_ancestry.fit_panel_basis + project_member reproduce these within atol
    (after sign-canonicalization), making the "reproduces the reference bit-for-bit" docstring enforceable.

Run: ``python -m pytest test_parity_golden.py``
"""
from __future__ import annotations

import os

try:
    import pytest
except ModuleNotFoundError:  # allow `python test_parity_golden.py` without pytest
    pytest = None

_FIX = os.path.join(os.path.dirname(__file__), "fixtures")
_RAW_FIX = [os.path.join(_FIX, f) for f in ("tiny.g.vcf.gz", "tiny.scorefile.txt", "tiny.sscore")]
_ANC_FIX = [os.path.join(_FIX, f) for f in ("golden_basis.npz", "golden_projection.json")]
RAW_TOL = 1e-6
ANC_ATOL = 1e-6


def _have(paths):
    return all(os.path.exists(p) for p in paths)


def _skip(msg):
    if pytest:
        pytest.skip(msg)
    raise _Skipped(msg)


class _Skipped(Exception):
    pass


def test_raw_score_parity_vs_plink2():
    if not _have(_RAW_FIX):
        _skip("plink2 golden fixtures not committed — see module docstring to generate "
              "fixtures/tiny.{g.vcf.gz,scorefile.txt,sscore}. Raw-score parity is UNVERIFIED until then.")
    # TODO(parity): load tiny.scorefile via lib/prs_register, extract dose via lib/prs_extract+gvcf_dose
    # over tiny.g.vcf.gz, sum weight*dose, and assert == plink2 SCORE1_SUM within RAW_TOL.
    raise AssertionError("fixtures present but parity assertion not implemented — implement per docstring")


def test_fraposa_basis_parity():
    if not _have(_ANC_FIX):
        _skip("FRAPOSA golden fixtures not committed — see module docstring for golden_basis.npz + "
              "golden_projection.json. The 'reproduces the reference' claim is UNVERIFIED until then.")
    # TODO(parity): fit_panel_basis on the golden panel, project the held-out sample, assert PCs match
    # golden_projection within ANC_ATOL after sign-canonicalization.
    raise AssertionError("fixtures present but parity assertion not implemented — implement per docstring")


if __name__ == "__main__":
    for fn in (test_raw_score_parity_vs_plink2, test_fraposa_basis_parity):
        try:
            fn(); print(f"RAN {fn.__name__}")
        except _Skipped as e:
            print(f"SKIP {fn.__name__}: {e}")
        except Exception as e:
            print(f"TODO {fn.__name__}: {e}")
