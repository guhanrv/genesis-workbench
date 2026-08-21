# Amy full-union scale ladder (10 / 100 / 1k × 197 PGS)

Distinct from `SCALE_LADDER_1K_FINDINGS.md` (Stage-0, 1M loci, 1 PGS). This is
the production-shaped store: Amy gVCF `SQA7J737`, ~17.2M dosage rows/sample,
197 PGS, `mean_impute`.

Dollars are **illustrative at $0.60/node-hr** (`system.billing` is not readable
here). Fan-out cluster: 4 workers + driver, `c3d-standard-8-lssd`. Score
cluster: 4 workers + driver, `c3d-highmem-8-lssd`, unless noted. Cost on a row
is **exec node-hours** unless setup is called out. Fan-out is **not** production
extract COGS.

## 1k correctness (2026-08-21)

`prs_scores` for `synth_amy1k_*`: **197,000 cells**, 1,000 samples, 197 PGS.
Mean coverage **94.86%**, min **52.17%** (same as n=10 / source).

| Check | Result |
|---|---|
| Clones vs each other | 195/197 PGS exact on `ROUND(raw,8)` + match-count; other 2 have max raw spread **5×10⁻¹²** (float noise) |
| `synth_amy1k_0001` vs `synth_amy10_01` | 197/197 raw and n_matched; max \|Δraw\| 1×10⁻¹³ |
| vs `SQA7J737` | 197/197 raw and n_matched |
| z vs source | Differs only on **PGS000001** (`prs_curation_dev_v2`) and **PGS000004** (`valtest_stats`); source still `prs_curation_196_v1`. Same stale-panel note as n=100. Ancestry EUR throughout. |

## Locked score config

**4 workers, `sample_chunk_size=10`, mean_impute, PGS chunking off.**  
n=100: 3,826 s, 0 spill, **~$0.032/sample**.  
1k resume: 32,653 s for 875 new samples, 0 spill, **~96 samples/h**.

Do **not** use: PGS weight-budget chunking (4.4× slower at n=10); all-at-once at
n≥100 (1.5 TB shuffle / ~3 TB spill); `sample_chunk_size=25` at the 1k table
(~400 GB shuffle + spill; cancelled after 125 samples).

## Fan-out (synthetic)

| N | Rows | Exec | Nodes | Est. $ | Rate |
|---|---|---|---|---|---|
| 10 | 172 M | 204 s | 5 | $0.17 | 843k rows/s |
| 100 | 1.72 B | 1,156 s | 5 | $0.96 | 1.49M rows/s |
| 1,000 | 17.2 B | 33,925 s + 261 s setup | 5 | **$28.50** | ~3× slower than n=100 linear |

1k fan-out run: `385678097411173`.

## Score

| Config | N | Exec | Est. $ | Keep? |
|---|---|---|---|---|
| 1 node, all-at-once | 10 | 3,627 s | $0.60 | baseline only |
| 1 node, PGS 3M chunks | 10 | 16,099 s | $2.68 | no |
| 4 workers, all-at-once | 10 | 517 s | $0.43 | workers help |
| 4 workers, chunk=10 | 100 | 3,826 s | $3.19 | **yes — lock** |
| 4 workers, all-at-once | 100 | 10,684 s | $8.90 | no — spill |
| 4 workers, chunk=25 | 1k (125 cells) | 15,419 s then cancel | $13.04 | no — spill |
| 4 workers, chunk=10 resume | 1k (875 new) | 32,653 s + 261 s setup | $27.43 | **yes** |

Runs: n=10 W1 `682582972754812`; n=10 W4 `172832081298152`; n=100 chunk=10
`107977029748806`; 1k chunk=25 cancelled `1107301303743633`; 1k chunk=10
`317736927459014`.

Full 1k score from empty at the locked pace ≈ **10.4 h / ~$31**.

## This 1k experiment invoice

| Piece | Est. $ |
|---|---|
| 1k fan-out | $28.50 |
| 1k score resume (chunk=10) | $27.43 |
| Cancelled chunk=25 | $13.04 |
| **Paid this rung** | **~$69** |
| If chunk=10 from the start (no 25) | **~$59** |

## Production extract / ancestry (not fan-out)

| Job | N | Exec | Est. $/sample |
|---|---|---|---|
| gVCF extract, 197-PGS union | 6 real | 2,010 s, 1× highmem | $0.056 |
| gVCF extract, Amy clones | 10 | 2,209 s, 1× highmem | $0.037 |
| `build_sample_ancestry` | 10 new | 186 s | $0.003 |

Do not mix skip/no-op extracts or Stage-0 1M-locus benches into this COGS.

**Member-facing batched compute is still ~$0.09–0.14/sample (ceiling $0.15).**
1k scoring did not change the ~$0.03 score line. Fan-out must not appear in
that number.

## 10k

Not run. Fan-out already lost time-invariance at 1k. Score at chunk=10 is
sequential 10-sample batches (~10 h wall per 1k on this 5-node fleet). Treat
10k as a budget decision, not the default next job.
