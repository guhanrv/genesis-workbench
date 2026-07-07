# Stage-0 scorer benchmark — run plan (GATED on explicit go-ahead)

The plan's benchmark gate. Goal: **pick the smallest classic cluster the scorer needs**, decide
**SQL join-aggregate vs `applyInPandas`**, confirm **incrementality bounds steady-state cost**, and derive
**`dbu_per_cell`** for the reconcile kill-switch — all from measured numbers, before any real backfill.

Everything runs on **fixed-size classic job clusters** (single-node first, titrated up one rung at a time).
**Nothing on serverless.** Harness: `benchmark/stage0_scorer_benchmark.py`.

## What it measures (PHI-free)
Reuses the workspace's existing **3202-sample 1000G `pca_dosage`** (Glow wide `states[]`/`sample_ids[]`),
exploded into the long `dosage` store, with **synthetic `pgs_weights`** at those loci. The join-aggregate
cost scales with `dosage_rows × weight_rows`, not allele identity — so this is a faithful *throughput*
benchmark. `max_loci` caps the first run cheap; extrapolate `$/cell` to the 48.4M-weight × N-sample backfill.

Per run it appends one `stage0_metrics` row per phase:
`full_backfill` (all samples × all PGS) · `add_pgs` (~1 column) · `add_sample` (~1 row) · `rerun_idempotent` (0 cells).
Columns: `n_exec, cores, n_cells, wall_clock_s, cells_per_s, est_dbu, est_cost_usd, dbu_per_cell`.

## Cluster (same known-valid single-node config as the smokes)
`c3d-highmem-8-lssd`, `SINGLE_USER`, `spark_version=15.4.x-scala2.12`, ephemeral job cluster.
Titration = re-submit with a larger `num_workers` (`0` single-node → `1` → `2` → `4`), everything else fixed.

## Ladder (STOP after each rung; approve the next)
| step | mode | cluster | params | purpose | est. cost* |
|---|---|---|---|---|---|
| 0 | prep | single-node | `max_loci=50000` | build bench dosage+weights ONCE | ~$0.5–2 |
| 1 | score | single-node (`num_workers=0`) | — | baseline throughput + `$/cell` | ~$0.5–1 |
| 2 | score | `num_workers=1` | — | first titration rung | ~$0.5–1 |
| 3 | score | `num_workers=2` | — | curve | ~$1 |
| 4 | score | `num_workers=4` | — | curve (stop at smallest that meets need) | ~$1–2 |
| 5 | (optional) prep+score | `max_loci=0` (all loci) | — | full chr21 scale, best rung only | TBD from curve |

\* Rough; the harness reports actual `est_cost_usd`. Refine `dbu_per_node_hr` / `dollar_per_dbu` from billing.
**Total to a mechanism+size decision: a few dollars.** The full genome-wide backfill is **estimated** from
step-1's `$/cell` and **approved separately** — never run blind.

## How to run a rung (copy-paste; requires go-ahead)
```bash
# import the harness once
databricks -p dev-exploration workspace import \
  --file benchmark/stage0_scorer_benchmark.py --language PYTHON --format SOURCE --overwrite \
  /Users/<you>/stage0_scorer_benchmark

# STEP 0 — prep once (single-node). STEP 1+ — mode=score, bump num_workers each rung.
databricks -p dev-exploration api post /api/2.1/jobs/runs/submit --json '{
  "run_name": "stage0_prep",
  "tasks": [{"task_key":"s0","notebook_task":{
      "notebook_path":"/Users/<you>/stage0_scorer_benchmark",
      "base_parameters":{"mode":"prep","max_loci":"50000","n_pgs":"5","density_pct":"60"}},
    "new_cluster":{"spark_version":"15.4.x-scala2.12","node_type_id":"c3d-highmem-8-lssd","num_workers":0,
      "data_security_mode":"SINGLE_USER",
      "spark_conf":{"spark.databricks.cluster.profile":"singleNode","spark.master":"local[*, 4]"},
      "custom_tags":{"ResourceClass":"SingleNode","work_type":"prs_stage0"}}}]}'
# then mode=score at num_workers 0,1,2,4 (drop the singleNode spark_conf + set num_workers>0 for multi-node).
```

## Reading the result
```sql
SELECT phase, n_exec, cores, n_cells, wall_clock_s, cells_per_s, est_cost_usd, dbu_per_cell
FROM   dev_exploration_sandbox.prs_stage0_bench.stage0_metrics ORDER BY run_ts;
```
Decide from the numbers:
- **Cluster size** = the smallest `n_exec` where `cells_per_s` stops improving materially (knee of the curve).
- **Mechanism** = SQL, unless it's inadequate — then add an `applyInPandas` variant and compare the same way.
- **Incrementality** = `add_pgs` / `add_sample` cells ≪ `full_backfill`, and `rerun_idempotent` = 0 cells → steady-state cost is increment-bounded (the cost-model claim).
- **`dbu_per_cell`** (from `full_backfill`) → set the reconcile widget so its dry-run cost estimate is calibrated.
- **Full-backfill $** = `dbu_per_cell × (48.4M-equiv weighted cells)`, presented for approval before step 5.

## Teardown (after the decision)
```sql
DROP SCHEMA IF EXISTS dev_exploration_sandbox.prs_stage0_bench CASCADE;
```
Keep it only while iterating on the ladder; the `stage0_metrics` numbers are what matter — copy them out first.

## Not covered here (separate, also gated)
**Extraction** cost (per-sample gVCF → dosage) is the real production cost driver but needs real WGS gVCFs
(PHI). Benchmark it separately on member gVCFs at small N, or synthesize larger gVCFs — do **not** stage
member data into the shared sandbox without a data-governance decision.
