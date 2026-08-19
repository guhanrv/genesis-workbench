# Vendored from pca_v1/lib/featurize_to_basis.py for the prs bundle. Canonical copy: pca_v1.
"""featurize_to_basis — the ONE fit/serve-shared featurization transform (design §5.1).

Turns a sample's **observed** ``(variant_id, dose)`` pairs into the standardized,
basis-aligned length-N vector the FRAPOSA projection consumes. This is the load-bearing
half of Design A (decision D6): the caller passes *names*, never a slot order, so it is
structurally impossible for a caller to align to the wrong slots or standardize with the
wrong mean/std. The SAME function runs

  - inside the ``ancestry_pca`` model at serving (one call per incoming sample), and
  - in the fit pipeline to build the panel's own vectors,

so fit and serve are provably identical (no train/serve skew). Pure numpy — imports and
runs off-cluster; unit-tested directly.

Steps (design §5.1, 1–4 — projection/coverage-gate steps 5–6 live in the model/fraposa):
  1. look up each observed ``variant_id`` → ``slot_idx`` against the basis (drop non-basis loci);
  2. orient the ALT-allele dose to the slot's ``effect_allele`` (2−dose flip when effect==ref);
  3. fill unobserved basis slots per ``impute_mode``;
  4. standardize ``(x − mean) / std`` with the basis's frozen per-slot mean/std.

Returns ``(z, n_covered)`` — ``z`` the float64 length-N standardized vector, ``n_covered``
the number of basis slots the sample actually observed (feeds ``coverage_frac`` upstream).

**Input contract.** ``variant_id = "chrom:pos:ref:alt"`` (canonical ``dosage.variant_id``,
design §4.1) and ``dose`` = ALT-allele dosage in [0, 2]. The basis is keyed by the SAME
canonical id, so a match is an exact string equality; orientation to ``effect_allele`` then
handles the (rare, for a panel-built basis where effect==alt) ref/alt sign. Palindromic SNPs
are dropped at *build* time (design §4.2), so they never appear in the basis and serving
inherits the drop through the lookup — this function does not re-check strand.

**impute_mode** (design §8.2; ``fill_value`` for unobserved slots):
  - ``"zero"`` (v1 DEFAULT) — raw dose 0 → ``z = (0 − mean) / std``. The "absent ≈ confident
    hom-ref ≈ 0 copies of ALT" assumption; valid only for uniform high-quality WGS (D9). This
    is a DELIBERATE v1 change from the historical FRAPOSA behavior below.
  - ``"at_mean"`` — impute at the panel mean, i.e. standardized value 0 (the observation
    contributes nothing to the projection). **This reproduces the historical
    ``fraposa.project_member`` missing-handling exactly** (it set standardized-missing → 0).
  - ``"mean_af"`` — raw dose 2·AF → standardize. With a frozen basis ``mean`` equals the
    panel's ``2·AF_effect``, so ``mean_af`` coincides with ``at_mean`` (z = 0) unless an
    explicit per-slot ``af`` is supplied via the basis (``af`` array); then it uses ``2·af``.
"""
from __future__ import annotations

import numpy as np

_IMPUTE_MODES = ("zero", "at_mean", "mean_af")


def build_basis_index(basis: dict) -> dict:
    """Precompute the ``variant_id → slot_idx`` lookup once (e.g. before broadcasting the
    basis to executors), so per-sample featurization is a dict get, not a rebuild.

    ``basis`` must carry equal-length, slot-ordered numpy/array fields: ``variant_id``,
    ``effect_allele``, ``ref``, ``alt``, ``mean``, ``std``. Returns the same dict with a
    ``"vid_to_slot"`` entry added (idempotent — reuses an existing one)."""
    if "vid_to_slot" not in basis:
        vids = basis["variant_id"]
        basis["vid_to_slot"] = {str(v): i for i, v in enumerate(vids)}
    return basis


def featurize_to_basis(observed_pairs, basis: dict, *, impute_mode: str = "zero"):
    """Observed ``(variant_id, dose)`` pairs → standardized length-N basis vector.

    Parameters
    ----------
    observed_pairs : iterable of (variant_id, dose)
        variant_id = "chrom:pos:ref:alt" (canonical); dose = ALT-allele dosage in [0, 2].
        Observed loci only, ANY order. Non-basis loci are dropped. If two pairs map to the
        same slot (should not happen for a canonical one-row-per-(sample,variant) source),
        the last one wins.
    basis : dict
        Slot-ordered arrays, all length N: ``variant_id``, ``effect_allele``, ``ref``,
        ``alt``, ``mean`` (N, or (N,1)), ``std`` (N, or (N,1)). Optionally ``af`` (N,) for
        ``impute_mode="mean_af"``. A ``"vid_to_slot"`` lookup is built on demand.
    impute_mode : {"zero", "at_mean", "mean_af"}
        How to fill unobserved basis slots (see module docstring). Default "zero" (v1).

    Returns
    -------
    (z, n_covered) : (np.ndarray float64 shape (N,), int)
        ``z`` is the standardized, basis-aligned vector ready for FRAPOSA projection;
        ``n_covered`` is how many basis slots the sample observed.
    """
    if impute_mode not in _IMPUTE_MODES:
        raise ValueError(f"impute_mode must be one of {_IMPUTE_MODES}, got {impute_mode!r}")

    build_basis_index(basis)
    vid_to_slot = basis["vid_to_slot"]
    N = len(basis["variant_id"])
    mean = np.asarray(basis["mean"], dtype=np.float64).reshape(-1)
    std = np.asarray(basis["std"], dtype=np.float64).reshape(-1)
    effect = basis["effect_allele"]
    alt = basis["alt"]

    # raw effect-oriented dose per slot; NaN marks "not yet observed" so we can distinguish
    # a covered slot from an imputed one before applying impute_mode.
    raw = np.full(N, np.nan, dtype=np.float64)
    for vid, dose in observed_pairs:
        j = vid_to_slot.get(str(vid))
        if j is None:                                 # locus not in the basis → dropped
            continue
        d = float(dose)
        # orient ALT-allele dose → effect-allele dose (design §5.1 step 2). For a basis built
        # from the panel, effect_allele == alt so this is the identity; the flip covers the
        # general case (effect == ref) and keeps fit/serve orientation in one place.
        if str(effect[j]) == str(alt[j]):
            raw[j] = d
        else:                                         # effect is the ref allele → diploid flip
            raw[j] = 2.0 - d
    covered = ~np.isnan(raw)
    n_covered = int(covered.sum())

    # standardize observed slots with the frozen panel mean/std (design §5.1 step 4).
    z = np.zeros(N, dtype=np.float64)
    z[covered] = (raw[covered] - mean[covered]) / std[covered]

    # fill unobserved slots per impute_mode (design §8.2).
    miss = ~covered
    if impute_mode == "zero":
        # raw dose 0 → standardized (0 - mean)/std
        z[miss] = (0.0 - mean[miss]) / std[miss]
    elif impute_mode == "at_mean":
        # impute at the panel mean → standardized 0 (historical FRAPOSA behavior)
        z[miss] = 0.0
    else:  # "mean_af"
        af = basis.get("af")
        if af is None:
            # no explicit AF → panel mean already equals 2·AF_effect, so this is at-mean (z=0)
            z[miss] = 0.0
        else:
            af = np.asarray(af, dtype=np.float64).reshape(-1)
            z[miss] = (2.0 * af[miss] - mean[miss]) / std[miss]

    return z, n_covered
