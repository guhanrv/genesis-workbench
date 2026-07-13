"""the reference gVCF-dose module — gVCF END-block expansion + per-target dose
extraction.

Adapted from a validated single-node pysam scorer. Two layers:

* Per-target dose primitives — ``_compute_dose_for_target`` and friends —
  decode the right dose from a single VCF/gVCF record + (chrom, pos, ea,
  oa) target. Handles GT vs HDS and ref-blocks (``END=``), and the
  REF/ALT-swap encoding (effect allele may be the record's REF or ALT).
  NOTE: strand-flip is NOT handled — the effect/other alleles are matched
  literally, with no reverse-complement. Inputs must be forward-strand,
  GRCh38-harmonized (PGS Catalog hmPOS); palindromic (A/T, C/G) variants
  are dropped at registration so an ambiguous-strand call can't mis-match.
* Lockstep walk — ``_extract_dose_vector`` walks the user's VCF in one
  pass against a sorted catalog, expanding gVCF reference blocks as it
  goes. Returns a per-target dose array aligned with the catalog.

Plus the catalog-FASTA precompute (``build_catalog_fasta_ref``) which is
a one-time setup the walker consumes.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np


# ──────────────────────────────── dose extraction ───────────────────────────


_NAN = float("nan")


def _gt_dose_for_allele(
    gt: tuple[int | None, ...] | None,
    alleles: list[str],
    target_allele: str,
) -> float:
    """Count copies of ``target_allele`` in a record's GT.

    ``alleles`` is ``[REF, ALT_1, ALT_2, …]`` from the record. Returns
    ``NaN`` if the GT is missing (None present) or the target allele
    isn't represented in the record's allele set.
    """
    if gt is None or any(a is None for a in gt):
        return _NAN
    try:
        target_idx = alleles.index(target_allele)
    except ValueError:
        return _NAN
    return float(sum(1 for a in gt if a == target_idx))


def _hds_dose_for_allele(
    hds: tuple[float | None, ...] | None,
    alleles: list[str],
    target_allele: str,
) -> float:
    """Count copies of ``target_allele`` from haploid dosages (HDS field).

    HDS is ``(p_alt_1_h1, p_alt_1_h2, p_alt_2_h1, …)`` for biallelic
    sites; ``len(hds) == len(alts) × ploidy``. For typical biallelic
    diploid sites, ``hds = (p_h1, p_h2)`` where each entry is the
    posterior probability of ALT on that haplotype. Sum to get expected
    ALT dose. If the user's REF is the target, dose = 2 - ALT_dose.
    """
    if not hds or any(h is None for h in hds):
        return _NAN
    # Multiallelic (incl. a symbolic <NON_REF> ALT): HDS has len(alts)×ploidy entries, so summing all
    # of them would fold other-ALT / <NON_REF> posteriors into this ALT's dose (overcount), and the
    # 2 - alt_dose REF orientation would then also be wrong. Drop instead of miscompute — the variant
    # becomes missing and is mean-imputed (2·AF) downstream. Only biallelic (REF + 1 ALT) is trusted.
    if len(alleles) != 2:
        return _NAN
    alt_dose = float(sum(hds))
    if target_allele == alleles[0]:
        return 2.0 - alt_dose
    if target_allele == alleles[1]:
        return alt_dose
    return _NAN


def _record_carries_trustworthy_dose(rec, *, is_hds: bool = False) -> bool:
    """Whether a VCF record gives us trustworthy genotype information.

    Returns False when the record fails any of:
      * ``FILTER`` is non-empty AND non-``PASS`` (e.g. ``FAIL``,
        ``LowDP``, ``LowGQ`` — sequencing.com 30x WGS gVCFs flag
        low-depth/quality regions this way; these are "we couldn't
        confidently call", NOT "homozygous REF").
      * For ``GT``-mode (``is_hds=False``): ``GT`` is missing (``./.``
        or any ``None`` element), or the GT field is absent.
      * For ``HDS``-mode (``is_hds=True``): ``HDS`` is missing/empty
        (any element is None). Imputed bundles often omit GT entirely
        and only carry HDS — checking GT presence would skip every
        record. We trust HDS values directly.

    Returning False from this check causes :func:`_extract_dose_vector`
    to skip the record entirely — the position is left as not-covered
    so the downstream impute step (mean-impute against panel afreq, or
    drop, depending on ``missing_mode``) handles it. This matches what
    ``bcftools view -f PASS`` + plink2 ``--score --read-freq`` does on
    the bcftools+plink2 path: low-quality / no-call records produce
    "missing" pgen genotypes which plink2 then mean-imputes.

    Without this filter, a REF block with ``FILTER=FAIL, GT=./.``
    (very common in 30x WGS gVCFs over low-depth regions) would be
    treated as "user is homozygous REF" by pysam and "missing →
    mean-impute" by plink2, producing real per-PGS score divergence
    (e.g. ~1e-3 on a small-effect-size PGS where one such position
    sits inside a low-depth REF block).
    """
    # FILTER check: empty list / no filter set / single PASS all pass.
    filters = list(rec.filter) if rec.filter else []
    if filters and filters != ["PASS"]:
        return False
    sample = rec.samples[0]
    if is_hds:
        # HDS-mode: trust dosage directly. Imputed VCFs commonly omit GT.
        hds = sample.get("HDS")
        if hds is None:
            return False
        try:
            if any(h is None for h in hds):
                return False
        except TypeError:
            return False
        return True
    # GT-mode: any missing element (None) or no GT at all → no info.
    gt = sample.get("GT")
    if gt is None or any(a is None for a in gt):
        return False
    return True


def _compute_dose_for_target(
    record,
    effect_allele: str,
    other_allele: str,
    *,
    is_hds: bool,
    fasta_ref_at_pos: str | None = None,
) -> float:
    """Derive the user's dose of ``effect_allele`` at this record's position.

    Handles three cases on a single record:

      * Variant call where the record's allele set includes ``effect_allele``
        → count copies via GT (or HDS for imputed bundles). If the alleles
        don't include effect or other, returns NaN (caller decides whether
        to treat as 0 — observed-but-different-variant — or mean-impute).
      * REF block (no ALT). The record covers our target position via
        ``END=``, but ``rec.ref`` is the FASTA base at the BLOCK START
        (rec.pos), not at the target. For multi-bp REF blocks the FASTA
        base at the actual target can differ from ``rec.ref``. We accept
        the caller-supplied ``fasta_ref_at_pos`` and use it preferentially;
        without it we fall back to ``rec.ref`` (correct only for
        single-base blocks, the behaviour before the FASTA-aware fix).
      * REF block REF-of-position matches effect → dose = 2.
        REF block REF-of-position matches other → dose = 0.
        REF block REF-of-position is neither → user is confirmed REF/REF
        for some allele the catalog doesn't know about (true catalog vs
        reference disagreement). They have 0 copies of effect — return 0,
        not NaN, so the caller doesn't mistakenly mean-impute. The latter
        was the smoking gun on small PGSes (PGS001819 had 32% of catalog
        positions flagged as "missing" because rec.ref of the enclosing
        block disagreed with the catalog's effect/other; mean-imputing
        them inflated the score by +1.18 vs plink2).
    """
    rec_ref = record.ref or ""
    rec_alts = list(record.alts) if record.alts else []
    alleles = [rec_ref] + rec_alts

    if rec_alts:
        sample = record.samples[0]
        if is_hds and "HDS" in record.format:
            dose = _hds_dose_for_allele(
                sample.get("HDS"), alleles, effect_allele,
            )
            if dose == dose:  # not NaN
                return dose
        # Fall back to GT (covers GT-only records and HDS records that
        # also carry GT — most do).
        return _gt_dose_for_allele(
            sample.get("GT"), alleles, effect_allele,
        )

    # REF block. Use the FASTA REF base at the actual target position when
    # the caller supplied one; otherwise fall back to rec.ref (which is
    # only correct for 1-bp blocks).
    ref_at_target = (fasta_ref_at_pos or rec_ref).upper()
    if ref_at_target == effect_allele:
        return 2.0
    if ref_at_target == other_allele:
        return 0.0
    # FASTA REF at this position is neither the catalog's effect nor its
    # other. The user is REF/REF for whatever the FASTA says — they have
    # 0 copies of effect. Returning 0 (vs NaN) keeps the caller from
    # mean-imputing what's actually a confirmed zero contribution.
    return 0.0


# ────────────────────────────────── catalog FASTA precompute ────────────────


def build_catalog_fasta_ref(
    catalog: "_Catalog",
    fasta_path: Path,
    cache_path: Path | None = None,
) -> np.ndarray:
    """Return a ``(catalog.n_var,)`` ``S1`` numpy array of the FASTA REF
    base at every catalog position.

    This is **catalog-level data** — same for every user — so we compute
    it once per (catalog, fasta) pair and (optionally) cache it on disk.
    Replaces the per-user, per-chrom ``fa.fetch().upper()`` calls that
    previously cost ~3-5 s/chrom × 22 chroms × N users on cold runs.

    The returned array is index-aligned with ``catalog.chrom``,
    ``catalog.pos``, etc. Bases are stored as single bytes (``S1``) for
    O(1) numpy comparison against effect/other allele bytes — the
    pre-1.0 path used Python string indexing inside the inner record
    loop, which dominated CPU on dense (10 M+ position) catalogs.

    Positions whose chromosome is missing from the FASTA, or that fall
    outside the chrom length, get ``b'N'``. Downstream comparisons
    against effect/other (``A``/``C``/``G``/``T``) all return False at
    those rows, which collapses to dose=0 in REF-block handling — same
    behaviour as the legacy code returning NaN for missing FASTA refs
    AND skipping mean-imputation when the position WAS covered by a
    REF block.
    """
    import pysam

    if cache_path is not None and cache_path.exists():
        try:
            blob = np.load(cache_path, allow_pickle=False)
            arr = blob["fasta_ref"]
            if arr.shape == (catalog.n_var,) and arr.dtype.kind == "S":
                print(f"  pysam_score: catalog fasta-ref cache hit "
                      f"({cache_path.name}, {arr.size:,} rows)")
                return arr.astype("|S1")
        except Exception:
            pass

    fa = pysam.FastaFile(str(fasta_path))
    out = np.full(catalog.n_var, b"N", dtype="|S1")

    # Group catalog rows by chrom via numpy (orders of magnitude faster
    # than the Python ``setdefault`` loop on million-row catalogs — the
    # loop's per-iteration object-array indexing was the dominant cost
    # in profiling on the genome-wide 7.8M-position catalog).
    chrom_str = catalog.chrom.astype(str)
    unique_chroms = np.unique(chrom_str)

    for chrom in unique_chroms:
        rows = np.where(chrom_str == chrom)[0]
        if rows.size == 0:
            continue
        try:
            seq = fa.fetch(str(chrom))
        except KeyError:
            try:
                alt = (f"chr{chrom}" if not str(chrom).startswith("chr")
                       else str(chrom)[3:])
                seq = fa.fetch(alt)
            except KeyError:
                continue
        # Single 250MB-class allocation per chrom: bytes-encoded upper-
        # case sequence backed by a numpy ``|S1`` view (zero-copy gather).
        seq_bytes = seq.upper().encode("ascii")
        seq_len = len(seq_bytes)
        positions = catalog.pos[rows]                # int64, fancy-indexed
        in_range = (positions >= 1) & (positions <= seq_len)
        valid_pos = positions[in_range] - 1
        valid_rows = rows[in_range]
        if valid_pos.size:
            buf = np.frombuffer(seq_bytes, dtype="|S1")
            out[valid_rows] = buf[valid_pos]
    fa.close()

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(cache_path, fasta_ref=out)
        print(f"  pysam_score: built catalog fasta-ref cache "
              f"→ {cache_path.name} ({out.size:,} rows)")
    return out


# ────────────────────────────────── lockstep walk ───────────────────────────


def _extract_dose_vector(
    vcf_path: Path,
    catalog: _Catalog,
    *,
    is_hds: bool = False,
    fasta_path: Path | None = None,
    fasta_ref_arr: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, str]:
    """Return ``(dose_vector, had_record_mask, sample_iid)`` for every catalog row.

    ``dose_vector`` has shape ``(catalog.n_var,)``, dtype ``float64``,
    with NaN where the user has no record covering the position. Positions
    that ARE covered by a record but produce no informative dose (e.g. an
    upstream indel with non-matching alleles) are also left NaN — caller
    interprets these via ``had_record_mask``.

    ``had_record_mask`` (bool array, same shape) marks rows where at least
    one VCF record covered the catalog position. The caller can use this
    to distinguish "truly missing → mean-impute" from "covered but no
    informative dose returned → leave NaN" cases.

    Walks the VCF chromosome by chromosome via ``vcf.fetch(chrom)``.
    ``record.stop`` honours ``END=`` so REF blocks are handled in O(1)
    per target without any pre-expansion.

    ``fasta_ref_arr`` (preferred): a precomputed ``(catalog.n_var,)`` S1
    numpy array of the FASTA REF base at every catalog position, built
    by :func:`build_catalog_fasta_ref`. When supplied, REF block dose
    computation is fully vectorised — one ``np.where`` per REF block
    instead of a Python for-loop over targets. Speedup is dramatic on
    dense (10 M+ position) catalogs against 30x WGS gVCFs (which are
    dominated by long REF blocks each covering 100s of catalog positions).

    ``fasta_path`` (legacy): if ``fasta_ref_arr`` isn't supplied, the
    function loads each chromosome sequence on demand via pysam and
    indexes Python strings inside the inner loop. Correct but ~3-10×
    slower on large catalogs.
    """
    import pysam

    # Group target row indices by chromosome, sorted by position.
    chrom_to_indices: dict[str, list[int]] = {}
    for i in range(catalog.n_var):
        chrom_to_indices.setdefault(str(catalog.chrom[i]), []).append(i)
    for chrom in chrom_to_indices:
        chrom_to_indices[chrom].sort(key=lambda i: int(catalog.pos[i]))

    dose = np.full(catalog.n_var, _NAN, dtype=np.float64)
    had_record = np.zeros(catalog.n_var, dtype=bool)
    sample_iid = ""

    # Precompute single-byte effect/other allele arrays for vectorised
    # REF-block dose computation. SNPs collapse to b'A'/b'C'/b'G'/b'T'.
    # Multi-bp indels collapse to their first byte; the FASTA at the
    # target position is single-byte too, so multi-bp catalog alleles
    # never match → dose=0, which is mathematically the same answer the
    # per-record path gives (a REF block can't tell us about an indel).
    eff_b1: np.ndarray | None = None
    if fasta_ref_arr is not None:
        eff_b1 = np.array(
            [(s[:1] if s else "").encode("ascii", errors="replace")
             for s in catalog.effect.tolist()],
            dtype="|S1",
        )

    vf = pysam.VariantFile(str(vcf_path))
    sample_iid = list(vf.header.samples)[0] if vf.header.samples else ""

    # Legacy on-demand FASTA only if no precomputed array — avoids the
    # multi-GB chromosome-sequence allocation in the hot path.
    fa = (pysam.FastaFile(str(fasta_path))
          if (fasta_path is not None and fasta_ref_arr is None) else None)

    # Probe the first chromosome key style ("1" vs "chr1") in the VCF.
    vcf_chrom_styles: list[str] = []
    try:
        for c in vf.header.contigs:
            vcf_chrom_styles.append(c)
            if len(vcf_chrom_styles) >= 5:
                break
    except Exception:
        pass
    use_chr_prefix = any(c.startswith("chr") for c in vcf_chrom_styles)

    for chrom, indices in chrom_to_indices.items():
        if not indices:
            continue
        vcf_chrom = f"chr{chrom}" if use_chr_prefix else chrom

        # Sorted target positions on this chromosome (in original row order
        # -> we keep the parallel ``pos_for_idx`` and ``idx_arr`` arrays).
        pos_for_idx = np.array(
            [int(catalog.pos[i]) for i in indices], dtype=np.int64,
        )
        idx_arr = np.array(indices, dtype=np.int64)
        first_pos = int(pos_for_idx[0])
        last_pos = int(pos_for_idx[-1])

        # Cache the chromosome FASTA sequence on first access — pysam's
        # per-call fetch overhead is significant; one big fetch + slice
        # is ~100× faster than per-position fetch on large catalogs (3 M
        # positions over 22 chromosomes). ``cached_seq`` is a string of
        # the entire chromosome; we index it as ``cached_seq[pos - 1]`` to
        # get the 1-based FASTA REF base at any position on this chrom.
        cached_seq: str | None = None
        if fa is not None:
            try:
                cached_seq = fa.fetch(chrom).upper()
            except KeyError:
                # Try the alternate prefix style (chr/no-chr) before giving up.
                try:
                    alt = (f"chr{chrom}" if not chrom.startswith("chr")
                           else chrom[3:])
                    cached_seq = fa.fetch(alt).upper()
                except KeyError:
                    cached_seq = None

        # Per-chrom slice of the catalog FASTA REF as a Python str,
        # indexed by k (target position within this chrom). Lets the
        # per-record fallback path do a Python str char access (~100 ns)
        # instead of numpy gather + bytes-to-str decode (~5 µs) per call
        # — ~50× faster, which matters because small REF blocks (window
        # < VEC_THRESHOLD) take this path. Trivial memory: 1 byte ×
        # n_targets_on_chrom ≈ 100 KB even for genome-wide catalogs.
        chrom_fasta_str: str | None = None
        if fasta_ref_arr is not None:
            chrom_fasta_str = (
                fasta_ref_arr[idx_arr].tobytes().decode("ascii")
            )

        if vcf_chrom not in vf.header.contigs:
            # Contig genuinely absent from this gVCF's header (rare on autosomes; common for
            # chrM/X/Y). Legitimately skip — nothing to fetch.
            continue
        try:
            rec_iter = vf.fetch(vcf_chrom, first_pos - 1, last_pos)
        except (ValueError, OSError) as e:
            # The contig IS in the header but fetch failed → almost always a missing/corrupt tabix
            # index (or I/O error), NOT "chromosome absent". Surface it: silently continuing would
            # drop the whole chromosome and silently undercount coverage for this sample.
            raise RuntimeError(
                f"fetch failed for contig {vcf_chrom!r} (present in header) — missing or corrupt "
                f"index for this gVCF? Original error: {e}"
            ) from e

        # For each record, binary-search the target window that falls inside
        # [rec_start, rec_end]. We don't maintain a one-shot pointer past
        # processed targets because **multiple records can legitimately
        # overlap the same target**:
        #
        #   * An upstream indel (e.g. a 14 bp deletion at pos X-3 with
        #     stop X+11) is returned by ``fetch`` for any target at X..X+10,
        #     even though its alleles (REF=ATCG…, ALT=A) don't match the
        #     SNP scoring entry (effect=T other=C).
        #   * The matching SNP record at pos X (REF=C, ALT=T) follows.
        #
        # If we permanently advanced past target X after the indel returned
        # NaN, the SNP record would never get a chance and we'd silently
        # miss ~hundreds of variants per mega-PGS on HDS-imputed BCFs
        # (this is the exact "drop mode" divergence we hit vs plink2 on
        # PGS001925). Binary search per record is O(log n) and the inner
        # window is typically 1 target (variant calls) or many (REF
        # blocks) — both fast.
        # Vectorisation kicks in only when (hi - lo) >= this threshold.
        # Below it, the per-record Python loop is faster because numpy
        # fancy-index/where setup costs (~5-10 µs each) dominate the
        # 1-2 dose computations the small window would do. Empirically
        # crossover is around k=4 for typical 30x WGS gVCFs (most
        # variant call records cover 1 catalog target; most short REF
        # blocks cover 0-3; only long REF blocks on dense catalogs
        # routinely cover ≥4, and that's where vectorisation amortises).
        VEC_THRESHOLD = 4
        for rec in rec_iter:
            rec_start = rec.pos                      # 1-based POS
            rec_end = rec.stop                       # honours END=
            lo = int(np.searchsorted(pos_for_idx, rec_start, side="left"))
            hi = int(np.searchsorted(pos_for_idx, rec_end + 1, side="left"))
            if lo >= hi:
                continue

            # Skip records that don't carry trustworthy genotype info
            # (FILTER ≠ PASS or GT missing). Treating them as "covered"
            # would set had_record=True → confirmed_zero step → dose=0,
            # but the right semantics is "no information here, fall
            # through to mean-imputation against panel afreq" — which
            # is what plink2 + --read-freq does for missing pgen calls.
            if not _record_carries_trustworthy_dose(rec, is_hds=is_hds):
                continue

            # ── REF-block fast path: vectorised dose update via FASTA ──
            # When fasta_ref_arr is precomputed AND the window is wide
            # enough, a single np.where covers every target in the REF
            # block — no Python for-loop, no per-target FASTA char
            # lookup, no per-target string compare. This is the dominant
            # cost on dense catalogs against 30x WGS gVCFs.
            #
            # Semantics match the per-record path exactly:
            #   FASTA[target] == effect → dose = 2  (homozygous effect REF)
            #   otherwise              → dose = 0  (homozygous other-or-
            #                                       confirmed-non-effect)
            # No explicit "fasta == other → 0" branch needed — dose=0
            # already means "no contribution"; np.where(... 2.0, 0.0)
            # collapses both "matches other" and "matches neither" to
            # the same numerical answer.
            if (fasta_ref_arr is not None and not rec.alts
                    and (hi - lo) >= VEC_THRESHOLD):
                window_rows = idx_arr[lo:hi]
                had_record[window_rows] = True
                nan_mask = np.isnan(dose[window_rows])
                if not nan_mask.any():
                    continue
                fill_rows = window_rows[nan_mask]
                fill_fasta = fasta_ref_arr[fill_rows]
                fill_eff = eff_b1[fill_rows]
                dose[fill_rows] = np.where(fill_fasta == fill_eff, 2.0, 0.0)
                continue

            # Per-record path: variant call records (need per-record
            # GT/HDS lookup), small REF blocks (vectorisation overhead
            # would lose), AND legacy fasta_path-only mode.
            #
            # OPTIMISATION: the FASTA REF lookup is only used by
            # _compute_dose_for_target when rec.alts is empty (REF
            # block). For variant call records the function ignores
            # fasta_ref_at_pos. We hoist the rec.alts check out of the
            # inner loop so variant call records skip the expensive
            # numpy gather / Python str index entirely. This is critical
            # on 30x WGS gVCFs where ~10⁸ variant call records cost
            # ~5 µs each in the FASTA gather (~500 s of pure waste on a
            # genome-wide catalog).
            is_ref_block = not rec.alts
            for k in range(lo, hi):
                row = int(idx_arr[k])
                # Mark coverage even if this specific record produces no
                # informative dose — caller uses had_record to distinguish
                # "truly missing" from "covered but no allele match".
                had_record[row] = True
                # Keep the FIRST valid dose seen at this row. A REF block
                # gives the correct REF/REF signal; a later variant call
                # at the same position (rare in proper gVCFs but possible
                # in HDS-imputed BCFs with co-located indels and SNPs)
                # would overwrite it — we don't want that, so we only
                # write when current dose is still NaN.
                if not np.isnan(dose[row]):
                    continue
                fasta_ref: str | None = None
                if is_ref_block:
                    if chrom_fasta_str is not None:
                        ch = chrom_fasta_str[k]
                        if ch != "N":
                            fasta_ref = ch
                    elif cached_seq is not None:
                        target_pos = int(pos_for_idx[k])
                        if 1 <= target_pos <= len(cached_seq):
                            fasta_ref = cached_seq[target_pos - 1]
                d = _compute_dose_for_target(
                    rec,
                    str(catalog.effect[row]),
                    str(catalog.other[row]),
                    is_hds=is_hds,
                    fasta_ref_at_pos=fasta_ref,
                )
                if d == d:                          # non-NaN
                    dose[row] = d
                # else: leave NaN — a later record (e.g. the actual SNP
                # following an upstream indel) may resolve it.
    vf.close()
    if fa is not None:
        fa.close()
    return dose, had_record, sample_iid


