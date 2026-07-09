# Stage-0 findings — profiled scorer benchmark + worker-scaling curve

Measured on a single-user classic cluster, node `c3d-highmem-8-lssd` (8 cores/64 GB),
DBR 15.4, classic job clusters. Data: 1000G `pca_dosage` exploded to the long store, capped at
**30,000 loci × 3,202 samples ≈ 96M dosage rows**, 5 synthetic PGS (~60% density ≈ 30k weights each).
Profiling via the driver Spark REST API (completed-stage diff per phase). Total suite cost ≈ **$0.30**
(5 runs, each ≈ 8–10 min single-/multi-node; bench schema torn down after). DBU $ at default
2 DBU/hr·node × $0.15 — wall-clock / bytes / cell-counts are exact, absolute $ approximate.

## 1. Where compute actually goes (single-node, profiled)

| phase | cells | wall | task_s (Σcores) | input_MB | shuffle_MB | spill_MB |
|---|---|---|---|---|---|---|
| full_backfill | 16,010 | 32.4s | 69.6 | 6.3 | 5.0 | 0 |
| add_pgs | 3,202 | 19.9s | 36.7 | 1.0 | 1.9 | 0 |
| add_sample | 6 | 13.8s | 17.9 | 0.4 | 0.8 | 0 |
| add_batch_x50 | 300 | 15.2s | 24.9 | 3.4 | 21.6 | 0 |
| **rerun_idempotent** (0 cells, no scoring) | 0 | **8.7s** | 10.3 | 0.0 | 0.3 | 0 |

**The cost is NOT data movement.** Three facts nail it:
- **Scan is free**: `input_MB = 6.3` for the whole 96M-row backfill. Dosage dictionary-compresses to
  ~MBs (3202 samples, 30k variants, few dose values). There is almost nothing to read.
- **No spill** anywhere → memory is not a constraint at this scale.
- **~8.7s fixed floor**: the score-free `rerun` phase still takes 8.7s. That's reconcile's cross-join +
  writing `_prs_reconcile_plan` + several **sequential Spark jobs** (each with launch latency) + the
  Delta MERGE. `task_s=69.6` over 8 cores ≈ 8.7s of *parallel* compute, but wall is 32s → the balance
  is **serial orchestration**, not compute.

## 2. Batching is the steady-state win (~45×)

| onboarding | samples | cells | wall | per-sample |
|---|---|---|---|---|
| per-member (add_sample) | 1 | 6 | 13.8s | 13.8s |
| **batched (add_batch_x50)** | **50** | 300 | **15.2s** | **0.30s** |

Scoring **50 members costs +1.4s over scoring 1** → **~45× cheaper per member**. The ~8.7s floor is paid
**once per run**, so amortizing it across a batch is the single biggest steady-state lever. **Onboarding
should batch pending members into one scoring run**, not run per-member.

## 3. Worker-scaling curve (B) — full_backfill, identical 96M-row grid

| workers | nodes | wall | cells/s | task_s |
|---|---|---|---|---|
| 0 (single-node) | 1 | **32.4s** | 494 | 69.6 |
| 1 | 2 | 35.9s | 446 | 72.6 |
| 2 | 3 | 37.9s | 422 | 70.1 |
| 4 | 5 | 42.9s | 374 | 68.4 |

**Adding workers makes it monotonically SLOWER.** `task_s` is ~constant (~70) — the total work doesn't grow —
so more nodes just add distribution/coordination overhead with no parallelism to exploit (the data is MBs).
**At cohort scale, do not add workers — single-node is fastest and cheapest.**

## 4. Recommendations for the "big leagues"

1. **Batch onboarding.** Score all pending new members in one run (≈45× cheaper/member than per-member).
   Same for new PGS: register several, score in one pass. **Implemented**: the reconcile+scorer are
   already batch (add_batch_x50 used the identical code path); the lever is the trigger, so
   `prs_scoring.job.yml` now has a file-arrival trigger on `prs_landing_zone` that batches arrivals
   (`min_time_between_triggers_seconds`/`wait_after_last_change_seconds`) — PAUSED until coordinated
   with any upstream file-delivery producer (which it should replace, not run alongside).
2. **Right-size to single-node until data is genome-wide.** Workers only pay off once the *parallel* work
   (`task_s / cores`) exceeds the ~8s serial floor — i.e., when dosage is GBs, not MBs. At cohort/chr-scale
   the scorer is orchestration-bound; a single-node classic cluster wins. Re-run this curve at genome-wide
   scale (dosage GBs) before sizing the one-time full backfill.
3. **The fixed floor is orchestration, not compute.** If per-run latency ever matters, reduce the number of
   sequential Spark actions per phase (e.g., skip materializing `_prs_reconcile_plan` for tiny plans; cache
   the plan; fold reconcile+score actions) — but batching already sidesteps it.
4. **Option A (Liquid Clustering by sample_id) is scale-insurance.** No measurable effect here (nothing to
   skip when dosage is 6 MB), but correct and necessary once dosage is GBs and the scan dominates. Kept.
5. **`dbu_per_cell` for the reconcile kill-switch:** ~1.1e-6 DBU/cell from full_backfill (scale-dependent —
   a genome-wide PGS column costs more per cell because more variants are scanned). Seed the widget with it.
6. **Extraction, not scoring, is the real production cost** (large WGS gVCFs, minutes/sample). Benchmark it
   separately on real gVCFs (PHI/data-governance decision); the scorer backfill is single-digit dollars.

## Genome-wide ramp — sample scaling at 1M loci (`stage0_genomewide_ramp.py`)

Probes the regime the cohort test couldn't: 1,000,000 synthetic loci (≈ one small genome-wide PGS),
1 PGS, **sample count ramped 1→128 in one guarded single-node run** (12.3 min, ≈$0.06, no guard hit).

| N samples | dosage rows | full-backfill wall | task_s | input_MB | shuffle_MB | spill_MB |
|---|---|---|---|---|---|---|
| 1 | 1M | 15.2s | 14.3 | 11.8 | 14.7 | 0 |
| 8 | 8M | 14.2s | 15.7 | 0.3 | 25.5 | 0 |
| 16 | 16M | 15.8s | 18.9 | 0.3 | 25.5 | 0 |
| 32 | 32M | 18.4s | 25.4 | 0.5 | 25.5 | 0 |
| 64 | 64M | 24.5s | 40.4 | 1.3 | 25.5 | 0 |
| 128 | 128M | 26.9s | 69.2 | 2.5 | 25.5 | 0 |
| **add_sample** (1 new → 128M store) | 129M | **14.9s** | 24.1 | 5.5 | 14.8 | 0 |
| **add_batch_x50** (50 new) | 179M | 24.1s | 42.8 | 1.8 | 25.5 | 0 |

Findings:
- **Scan stays cheap even at 128M rows** (`input_MB` ≤ 12). Dosage is inherently dictionary-compressible
  (variant_ids repeat across samples, dose ∈ {0,1,2}) → MBs on disk. **A "GB-scan" regime does NOT arise at
  1M loci × 128 samples** — it needs the full ~19M-variant union and/or thousands of samples.
- **No spill at any rung** — a single 64 GB node handles 128M rows fine.
- **The regime shift is CPU/`task_s`, not scan/shuffle/spill.** `task_s` grows with samples (15→69 from
  N=8→128); wall is floor-bound (~14s) up to ~N=16, then grows with work. N=128 full backfill = **27s**.
- **Shuffle is ~flat at ~25 MB (weights-bound)** — the 1M-row weight side dominates the shuffle; the dosage
  side stays modest.
- **Option A's file-skip PAYS OFF here** (it didn't at 30k loci): `add_sample` onto a 128M-row store is
  **14.9s** with `task_s`≈24 — i.e. it scores just the new sample (≈ scoring 1 sample fresh), NOT a
  128M-row re-scan (which would be `task_s`≈69, ~27s). Liquid Clustering + the predicate prune to the new
  sample's files once the store spans many files.
- **Batching still ~31× at scale**: 50 new samples = 24.1s vs 1 new = 14.9s → ~0.48s/sample batched.

Net: the **scorer is remarkably cheap on modest hardware** — genome-wide single-PGS × 128 samples backfills
in ~27s single-node and onboards a member in ~15s, no spill. The "big leagues" cost is **extraction**
(per-sample gVCF, minutes each), not scoring. The single-node ceiling (where spill / workers finally
matter) is beyond 128M compressible rows — reachable only with the full multi-PGS union (tens of millions
of loci) and/or thousands of samples; that rung is a further gated step.

## REAL extraction at scale (6 consented member WGS gVCFs)

The pivot from synthetic scoring to the real cost driver. Ran the actual pysam extraction on 6 real
WGS gVCFs (~200–300 MB each, on a UC Volume) against a genome-wide 1M-locus catalog, single-node
8-core. (Extraction walk is gVCF-READ-bound — measured locally: walk@100k=37s vs walk@1M=42s — so a
1M-locus catalog is representative of full genome-wide extraction *time*.)

| phase | samples | mode | tasks | dosage rows | wall |
|---|---|---|---|---|---|
| full6 | 6 | sample | 6 | 5.62M | 134.8s |
| full6 | 6 | **sharded (sample×chrom)** | 132 | 5.62M | **88.9s** |
| onboard1 | 1 | sample | 1 | 928k | 61.2s |
| onboard1 | 1 | **sharded** | 22 | 928k | **19.1s** |

fasta-ref build = 31.5s one-time (reads the 842 MB FASTA from the Volume; amortized across all samples).
Sharded and sample modes produce **identical row counts** → sharding is correct on-cluster.

Findings:
- **Extraction is I/O + BGZF-decompress bound on Volume gVCF reads** (~1 sample isolated = 61s ≈ 5 MB/s of
  gVCF; local SSD was 40s — FUSE adds overhead). 6 concurrent sample-tasks on one node = 135s (worse than
  the ~46s a compute-bound model predicts) → **contention** when many large concurrent reads hit one node's
  FUSE mount.
- **Chrom-sharding wins, biggest on onboarding**: add-one-member **61s → 19s (3.2×)** (22 chrom tasks spread
  the single gVCF's decompress across cores + smaller reads reduce contention); full-6 **135s → 89s (1.5×)**.
  This is the incremental path the design optimizes, so the onboarding win is the one that matters.
- **Scaling lever = NODES** (FUSE bandwidth + cores), not just cores-per-node — extraction is
  embarrassingly parallel across samples; add nodes to raise aggregate read bandwidth.
- **Extraction (~1 core-min/sample) dominates; scoring is ~free.** This confirms the whole cost model:
  extract once (~a minute/sample, sharded ~20s), reuse forever; the distributed scorer then costs cents.
  End-to-end onboarding of a new member ≈ ~20s extract (sharded) + seconds to score.

Rough cohort extrapolation: extraction wall ≈ N_samples × ~60 core-s / (cluster cores), I/O-contention
capped per node → e.g. ~1000 members on ~8 nodes ≈ tens of minutes, once. Cheap, and paid once.

## Caveat — small-scale regime
Every number here is the **cohort/chr-scale (MB) regime**, where the scorer is orchestration-bound and
workers don't help. At **genome-wide scale (GB dosage, 48M-weight PGS that won't broadcast)** the profile
flips to scan/shuffle-bound and workers *will* scale — that regime needs genome-wide data to measure and is
gated (cost + data staging). The curve here proves the small-scale regime and sets the batching + sizing
policy for cohort operation.
