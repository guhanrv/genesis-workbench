# Scale-ladder findings — sample ramp to 1,000 samples (1 billion dosage rows)

Extends the Stage-0 genome-wide ramp (`stage0_genomewide_ramp.py`) past its prior 128-sample stop to
**N = 1,000 samples × 1,000,000 loci = 1 billion dosage rows**, single-node `c3d-highmem-8-lssd`
(8 cores / 64 GB), DBR 15.4, 1 synthetic PGS (~480k weights). One guarded single-node run, ~17 min
wall (incl. cluster start), **~$0.02 DBU / ~$0.27 real**. Profiled via the driver Spark REST API
(completed-stage diff per phase). Bench schema (`dev_exploration_sandbox.prs_gw_bench`) torn down after.

Ramp `128,256,512,1000` then `add_sample` (+1) and `add_batch_x50` (+50) at the 1k store.

## The numbers

| phase | N | dosage rows | wall | task_s | input_MB | shuffle_MB | spill_MB |
|---|---|---|---|---|---|---|---|
| full_backfill | 128 | 128 M | 33.5s | 147 | 462 | **2198** | 0 |
| full_backfill | 256 | 256 M | 32.5s | 131 | 490 | 12.5 | 0 |
| full_backfill | 512 | 512 M | 50.3s | 280 | 969 | 12.6 | 0 |
| full_backfill | **1000** | **1.0 B** | **83.7s** | 527 | 1876 | 12.8 | **0** |
| add_sample (+1) | 1001 | 1.001 B | 27.4s | 118 | 5.5 | 14.8 | 0 |
| add_batch_x50 | 1051 | 1.051 B | 35.9s | 133 | 8.9 | 12.5 | 0 |

## 1. No spill at 1 billion rows — and AQE broadcast is why

**`spill_mb = 0` at every rung, including 1e9 rows on a 64 GB node.** The shuffle/spill breakpoint the
128-sample run couldn't reach **did not appear at 1k**. The `shuffle_MB` column shows the mechanism:
N=128 shuffled **2.2 GB** (the full dosage side of `dose ⋈ weights` on `variant_id`), but N=256→1000
all shuffle a flat **~12 MB**. That is **Adaptive Query Execution flipping the score-join to a broadcast**
once it learned the weights side (480k rows, 1 PGS) is broadcast-small — after which the billion dosage
rows **stream through un-shuffled**, so there is nothing to spill regardless of sample count.

## 2. The real breakpoint is the WEIGHTS axis, not the sample axis

This relocates the "1.7 T ⋈ 336 M shuffle" concern. Ramping **samples** alone never triggers the big
shuffle, because a single-PGS weight side always broadcasts. The shuffle/spill cliff lives on the
**weights axis**: many PGS → hundreds of millions of weight rows that **can't** broadcast → the dosage
side must shuffle on `variant_id`. Implication: a 10k-**sample** rung would likely *also* show no spill
(just more scan / `task_s`); the informative next rung is **ramping `n_pgs`** (the harness's
`n_pgs` × `weight_per_pgs` widgets) until the weight side exceeds the broadcast threshold and forces the
non-broadcast shuffle join. That is the regime that stresses shuffle, skew, and the Delta MERGE.

## 3. Sample-axis scaling is smooth (no cliff)

Full-backfill `task_s` grows ~linearly with samples (256→1000: 131→527); wall 33→84s. No knee, no spill.
A single 64 GB node backfills 1,000 samples genome-wide-per-PGS in **84s**.

## 4. Incrementality holds at 1e9 rows (the production path)

`add_sample` scored **1 new member against a 1.001-billion-row store in 27.4s reading 5.5 MB** — it did
**not** re-scan the billion rows. Liquid-clustering file-skip + the sample predicate prune to the new
sample's files, exactly as the design intends, now confirmed at 1e9 scale (it only had 128M-row evidence
before). `add_batch_x50` = 35.9s for 50 new samples → **~0.17 s/sample batched** vs 27.4s for 1 → the
~30–45× batching win persists at 1k.

## Caveats / scope
- **Sample axis isolated on purpose.** 1 PGS / 480k weights keeps the variable clean; it also means the
  weight side broadcasts (§1–2). Prod has many PGS whose union won't broadcast — untested here.
- **Synthetic dosage** (join-aggregate cost is scale-driven, not identity-driven — validated by the
  earlier Stage-0 suite). This measures the **scorer**, not extraction (the real per-sample cost driver,
  separately gated on real gVCFs / PHI).
- **1M loci, not the full ~19M-variant union.** The loci-width axis is a further multiplier not combined
  here.

## Weights-axis ramp (`stage0_weights_ramp.py`)

Follow-up to §2: hold samples FIXED at 128 (128M dosage rows — bounds the dosage-shuffle cost) and ramp
`n_pgs` 1→8→32→64 (weights 480k→30.7M rows), full-backfill each rung. Single-node `c3d-highmem-8-lssd`,
~13 min, ~$0.03 DBU. Bench schema torn down after.

| phase | n_pgs | weight rows | cells | wall | task_s | input_MB | shuffle_MB | spill_MB |
|---|---|---|---|---|---|---|---|---|
| wramp_pgs1 | 1 | 480 K | 128 | 33.0s | 147 | 464 | 2197 | 0 |
| wramp_pgs8 | 8 | 3.84 M | 1024 | 46.9s | 285 | 24.5 | 2291 | 0 |
| wramp_pgs32 | 32 | 15.4 M | 4096 | 83.3s | 576 | 98 | 2612 | 0 |
| wramp_pgs64 | 64 | 30.7 M | 8192 | 133.9s | 975 | 196 | 3040 | **0** |

Findings:
- **Still zero spill** — even at 30.7M weights × 128M dosage forcing a **~3 GB shuffle** on a single 64 GB
  node. The single-node ceiling is higher than expected on the weights axis too.
- **Cost scales ~linearly with weight rows** — `task_s` 147→975, wall 33→134s. Clean linear trajectory.
- **Prod extrapolation:** ~336M weight rows ≈ 11× the 30M tested → ~11× `task_s` (~2.9 hr parallel work)
  and ~11× shuffle volume, which almost certainly **spills** on 64 GB. So on the weights axis, **workers
  do pay off at prod scale** — the opposite of the sample axis (§3), where they never did.

**Methodological caveat (first run, cold stats):** `shuffle_MB` stayed high at *every* rung, including
`pgs1`, whereas the *sample* ramp broadcast that same 480k/1-PGS config (~12 MB shuffle). Suspected cause:
`pgs_weights` rebuilt every rung → cold stats → no broadcast. A warm-stats rerun tested that fix.

### Warm-stats rerun (`ANALYZE TABLE` per rung; finer ramp; `sizeInBytes` recorded)

| n_pgs | weight rows | weight size | shuffle_MB | spill_MB | wall | task_s |
|---|---|---|---|---|---|---|
| 1 | 480 K | **3.1 MB** | 2197 | 0 | 41s | 204 |
| 2 | 961 K | 6.1 MB | 2211 | 0 | 38s | 204 |
| 4 | 1.9 M | 12.3 MB | 2237 | 0 | 40s | 231 |
| 8 | 3.8 M | 24.5 MB | 2291 | 0 | 47s | 280 |
| 16 | 7.7 M | 49 MB | 2398 | 0 | 60s | 387 |
| 32 | 15.4 M | 98 MB | 2612 | 0 | 88s | 597 |

`autoBroadcastJoinThreshold = 10 MB`. **The warm-stats fix did NOT produce a broadcast→shuffle flip.**
Shuffle is **~2.2 GB flat from `pgs1`** — even at a **3.1 MB** weights table, well under the 10 MB
threshold with warm stats. Conclusions:
- **The dose side drives the shuffle, not the weights.** Shuffle ≈ constant ~2.2 GB (the 128M-row dose
  side sort-merged on `variant_id`), + only ~400 MB as weights grow 3→98 MB. "Where do weights stop
  broadcasting" is the **wrong frame** for this query shape — the dose side shuffles regardless.
- **Why it differs from the sample ramp** (where 480k weights broadcast → 12 MB shuffle + map-side partial
  agg): there weights are built ONCE and reused; here the planner estimates the *derived* `wts` relation
  (`pgs_weights ⋈ broadcast(planned_pgs)`), not the base table, so `ANALYZE TABLE` on the base didn't flip
  it to broadcast. A **planner stats-estimation artifact of the derived relation**, not a byte threshold.
- **Still no spill** through 15.4M weights / 98 MB / 2.6 GB shuffle single-node; cost ~linear in weights.
- **Reconciles the 6×196 observation:** shuffle here is dose-side-dominated; at 6 samples (~5.6M dose rows)
  it is ~23× smaller → negligible regardless of PGS count. "6×196 didn't shuffle" = the tiny-dose regime.

**To settle the broadcast question, the next step is `EXPLAIN FORMATTED` on the scorer** (BroadcastHashJoin
vs SortMergeJoin, and the size the optimizer assigns the `wts` relation) — a near-free query, not another
ramp. Forcing broadcast (e.g. `F.broadcast(wts)` when small, or an explicit size hint) would then let the
dose stream + map-side aggregate, collapsing the shuffle — worth testing as a scorer optimization.

### EXPLAIN — mechanism settled (`prs_explain` run)

Reproduced the scorer join+agg on small data (128 samples × 200k loci, 1 PGS = 0.7 MB weights, warm stats),
captured the FINAL executed (post-AQE) physical plan, default vs `F.broadcast(wts)`:

| variant | `dose ⋈ wts` | dose shuffled? | Exchanges |
|---|---|---|---|
| default scorer | **SortMergeJoin** | **yes** | 5 |
| `F.broadcast(wts)` | **BroadcastHashJoin** | **no** | 4 |

- Even at **0.7 MB** weights (70× under the 10 MB threshold, warm stats) the join is a **SortMergeJoin** →
  dose shuffles. Root cause: `wts` is a **derived relation** (`pgs_weights ⋈ broadcast(planned_pgs)`), and
  the optimizer over-estimates a *join output's* size, so it never auto-broadcasts — and **AQE did not
  convert it** (SMJ survived into the final plan). This is why every weights-ramp rung shuffled ~2.2 GB.
- **Fix confirmed:** `F.broadcast(wts)` → BroadcastHashJoin, dose-side Exchange gone, `sum()` becomes
  map-side partial → the shuffle collapses. **Caveat:** only valid while `wts` fits broadcast memory
  (few PGS / sample batches) — a full ~200-PGS / hundreds-of-M-row weight union won't broadcast, so the
  large full-backfill still sort-merges (chunk it by sample batch instead).

**Scorer optimization to land:** broadcast `wts` when its estimated/known size is under threshold (guarded
`F.broadcast` or a size check), so small-PGS and batched-onboarding scoring stream the dose instead of
shuffling it — the common steady-state path.

### Broadcast fix — LANDED + verified

`notebooks/05_score_prs.py` now guards the drop-path join: `F.broadcast(wts)` when the planned weights are
≤ `WEIGHTS_BROADCAST_MAX_ROWS` (3M). Verified on-cluster (`prs_verify_merge` run): small weights →
**BroadcastHashJoin, SMJ=0** (dose streams, no shuffle). Large unions exceed the guard and stay a shuffle
join (chunk by sample batch). [Verify caveat: the run's cases both landed <3M so only the broadcast branch
was re-exercised; the >3M→SMJ branch is already established by the EXPLAIN + weights ramp.]

### Spill / workers ramp — single-node does NOT spill even at 256 PGS (`prs_spill_bench` run)

Fixed 128 samples (128M dose), ramp `n_pgs` 64→128→256:

| n_pgs | weight rows | weight size | shuffle_MB | spill_MB | wall | task_s |
|---|---|---|---|---|---|---|
| 64 | 30.7 M | 201 MB | 3052 | **0** | 155 s | 1041 |
| 128 | 61.5 M | 392 MB | 4012 | **0** | 282 s | 2051 |
| 256 | 122.9 M | 784 MB | 5960 | **0** | 624 s | 4693 |

**No spill even at 123M weights / 784 MB / ~6 GB shuffle on a single 64 GB node.** So on the weights axis
**workers are a wall-time lever, not a memory necessity** up through this scale — cost scales cleanly
linearly (`task_s` doubles with weights, no cliff). At 256 PGS a 128-sample backfill is ~10 min single-node.

### MERGE into a 200M-row `prs_scores` — no spill; onboard cheap; update ≈ 2.5× insert (`prs_merge_bench` run)

Target: 200M rows (1M samples × 200 PGS), partitioned by `pgs_id`. (Synthetic target compressed to 1.4 GB
via constant columns — under-represents real byte size, but the MERGE mechanics/partitioning are faithful.)

| op | rows | wall | shuffle_MB | spill_MB |
|---|---|---|---|---|
| onboard **1 member** (+200) | 200 | **21 s** | 0.3 | 0 |
| onboard batch 50k (+10M) | 10 M | 103 s | 6395 | 0 |
| **update** 50k existing (10M) | 10 M | 265 s | 12171 | 0 |

Onboarding a single member into a 200M-row table = 21 s (partition prune, ~0 shuffle). **Updates cost
~2.5× inserts** (matched-file rewrite) → re-scoring after a weight change is pricier than adding members.
No spill.

### EXTRACTION at scale — 6 real gVCFs → 256 DISTINCT cold files (`prs_extract_bench` run)

The dominant cost driver, previously only at N=6. Replicated 6 real-scale WGS gVCFs to **256 distinct
physical files (72.4 GB > 64 GB node RAM → every read cold, no page-cache cheat)**, ran the REAL kernel
(`gvcf_dose`+`prs_extract`, sample-mode whole-genome walk, 1 task/file) on a single 8-core node against a
synthetic genome-wide 1M-locus catalog. Copy 136 s; extract 256 = **52.9 min**. Replicas torn down.

| metric | value |
|---|---|
| per-sample core-time (at 256-scale, cold) | **98.7 s** (~1.65 core-min) |
| per-node throughput | **4.8 samples/min** / **22.8 MB/s** aggregate cold FUSE read |
| spill | **0** (extraction streams) |
| covered loci/sample | 938,749 (matches STAGE0's ~937k → catalog faithful) |

- **Contention is real but modest:** per-sample rose 61 s (STAGE0 isolated) → 99 s at 256-scale = **~1.6×**
  from concurrent FUSE reads; aggregate 8-way read = 22.8 MB/s (≈5× a single 4.6 MB/s stream, not 8× —
  FUSE saturates below core count). No spill; node-bandwidth is the ceiling → **nodes are the lever**.
- **Sharded (sample×chrom) would be ~1.5–3.2× faster** (STAGE0) — these sample-mode numbers are a
  conservative upper bound.

**Extrapolation (sample-mode, cold, contention-included; sharded ≈ 2× better):**

| cohort | core-hours (one-time) | wall (fleet) | est. compute cost |
|---|---|---|---|
| 1k | ~27 | 3.5 h @1 node / ~1 h @4 nodes | ~$3 (~$1.5 sharded) |
| 100k | ~2,740 | ~7 h @50 nodes | ~$290 (~$150 sharded) |
| 1M | ~27,400 | ~7 h @500 nodes / ~35 h @100 nodes | ~$2,900 (~$1,500 sharded) |

Extraction is embarrassingly parallel across samples and never spills, so wall scales ~linearly with nodes.

## Next steps (each separately gated)
1. **Sharded extraction rung** — re-run the 256-file test with `shard_by_chrom` to confirm the ~2× win at
   scale (halves the extrapolations above).
2. **10k-sample scorer rung** — 1e10 rows; needs workers for build throughput; per §2 expect scan-bound.
3. **Real `prs_scores` MERGE size** — re-run the merge test with realistic (non-constant) columns so the
   target reflects true byte size, not the 1.4 GB synthetic floor.
