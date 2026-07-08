# Stage-0 findings — profiled scorer benchmark + worker-scaling curve

Measured on `dev_exploration_sandbox` (single-user), node `c3d-highmem-8-lssd` (8 cores/64 GB),
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
   Same for new PGS: register several, score in one pass.
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

## Caveat — small-scale regime
Every number here is the **cohort/chr-scale (MB) regime**, where the scorer is orchestration-bound and
workers don't help. At **genome-wide scale (GB dosage, 48M-weight PGS that won't broadcast)** the profile
flips to scan/shuffle-bound and workers *will* scale — that regime needs genome-wide data to measure and is
gated (cost + data staging). The curve here proves the small-scale regime and sets the batching + sizing
policy for cohort operation.
