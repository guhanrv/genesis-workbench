# PRS — Polygenic Risk Scoring (`prs_v1`)

Distributed, **incremental**, multi-PGS × multi-sample polygenic risk scoring on the
genesis Spark/Delta stack. Scores a growing set of samples against many
[PGS Catalog](https://www.pgscatalog.org/) scoring files, materializes results in a
MERGE-upserted Delta cell-store, and makes **add-PGS / add-sample / restate-PGS** cost
≈ one column / one row rather than a full recompute.

```
raw(sample, pgs) = Σ_variants  dose_of_effect_allele(sample) × weight        (effect-oriented)
z_msp            = (raw − panel_mean[pgs, MSP]) / panel_sd[pgs, MSP]         (frozen reference panel)
z_admixed        = Σ_superpop  P_RF(superpop) · (raw − mean[pgs,pop]) / sd[pgs,pop]   (continuous ancestry)
```

Normalization is against a **frozen HGDP+1kGP reference panel** (per-superpop), so a new
sample's z needs no cohort re-rank — unlike naive in-cohort standardization.

## Data governance / PHI (deployer responsibility)

This pipeline processes **individual genomic data (PHI)**. The code cannot enforce consent or
data-use policy — that is the **deploying organization's responsibility**. Before running against
real samples: ensure appropriate consent and IRB/data-use agreements are in place; keep the
`dosage` / `prs_scores` / `sample_ancestry` stores under Unity Catalog governance (they are tagged
`data_classification=PHI` in `00_setup_stores`; apply column masks / row filters and `GRANT`s per
your policy — the module does not ship grants). Do **not** stage member genomes into a shared
sandbox without a governance decision. The reference panel (HGDP+1kGP) and any figures in
`benchmark/` use public or synthetic/real-scale data only — no member-identifiable data is committed.

## Persistent Delta stores (`00_setup_stores`)

| store | grain | role |
|---|---|---|
| `pgs_registry` | pgs_id | catalog of available scores + metadata |
| `pgs_weights` | pgs_id × variant | long effect-oriented per-variant weights (`variant_id = chrom:pos:effect:other`) |
| `pgs_panel_ref` | pgs_id × superpop | **frozen** per-superpop mean/sd/quantiles → z with no re-rank |
| `pgs_panel_afreq` | variant | panel allele freq for 2·AF mean-imputation of missing variants |
| `dosage` | sample × variant | per-sample effect-allele dosage; Liquid-clustered by `sample_id`; incrementally extended |
| `sample_ancestry` | sample | most-similar-pop + RF posterior (`rf_probs`) from the PCA basis |
| `prs_scores` | sample × pgs | **the results cell-store** — flat scalars (MERGE-safe), partitioned by `pgs_id` |

## What it deploys — the `prs_scoring` job

Six tasks, all on **classic** job clusters (**never serverless** — unbounded autoscale is
the #1 bill risk). Light/extract start single-node; **`prs_score_cluster` is 4 workers**
(n=10/n=100 A/B). Fan-out is a separate 4-worker cluster, skipped in production. Ephemeral.

```
setup_stores → register → extract_dosage → build_sample_ancestry
                                      └→ optional synthetic manifest fan-out
                                                   └→ reconcile → score_prs → save_results → mark_success/failure
```

- **`00_setup_stores`** — idempotent DDL for the stores above.
- **`01_register_catalog`** — curation (`prs.yaml` + scorefiles) → `pgs_registry` / `pgs_weights` / `pgs_panel_ref`. Palindromic (A/T, C/G) SNPs dropped (strand-ambiguous).
- **`02_extract_dosage`** — per-sample **gVCF** → `dosage`, one Spark task per (sample × chrom). Uses the ported pysam **END-block + FASTA** kernel (`lib/gvcf_dose.py`): plink2 and Glow both drop gVCF `END=` REF blocks (~70% coverage loss), so this can't be Glow/SQL. Incremental: skips samples already extracted. *(Non-gVCF cohorts use the sibling engine `00_ingest_vcf` — Glow `DS/HDS/GT`, imputed-dosage or regular hard-called VCF — which orients + MERGEs into the SAME `dosage` store, so everything downstream is identical.)*
- **`03_reconcile`** — the incremental brain + cost kill-switch. Anti-joins the desired `(sample × pgs)` grid vs `prs_scores` → emits **only missing/stale cells**. **Dry-run by default** (`apply=false`): prints the cell count + estimated cost and scores nothing; a plan larger than `max_cells` additionally needs `confirm_large=true`. A stray run can never fire the full grid.
- **`04_build_sample_ancestry`** — OADP-projects each sample onto the frozen FRAPOSA PCA basis (`lib/prs_ancestry.py`) → RF most-similar-pop → `sample_ancestry`. Extraction distributed like `extract_dosage`; RF classify on the driver.
- **`materialize_scale_manifest`** — disabled unless `sample_manifest_path` is supplied and
  `confirm_synthetic_fanout=true`. Extracts each canonical gVCF once, then uses bounded Spark
  batches to copy canonical dosage/ancestry rows to synthetic logical IDs. This measures Delta,
  reconcile, and scoring scale without creating duplicate gVCFs or rereading one file N times.
- **`05_score_prs`** — SQL join-aggregate over the reconcile plan → `raw`, then reference-panel `z_msp` + RF-posterior-weighted `z_admixed` → MERGE `prs_scores`. Default `missing_mode=mean_impute`. Default `sample_chunk_size=10` (n=100 all-at-once spilled; n=10 did not). Optional PGS-axis chunking stays off.
- **`06_save_results`** — read `prs_scores` → log cohort summary metrics (coverage, z_msp/z_admixed counts, raw distribution) to the MLflow run. `mark_success` / `mark_failure` then set the run's `job_status` (framework convention).

## Two ingest engines (both write the shared `dosage` store)

| cohort | engine | notebook |
|---|---|---|
| **gVCF** (`END=` REF blocks) | pysam END-block + FASTA kernel | `02_extract_dosage` (in the scoring DAG) |
| **imputed dosage / regular hard-called VCF** (`DS`/`HDS`/`GT`) | Glow, distributed | `00_ingest_vcf` (guards against gVCF input) |

## Companion jobs

- **`prs_reference_setup`** — one-time scoring-reference builder (run on demand): `ref_00_build_panel_stats` computes **both** panel tables in a **single** pgenlib scan of the reference panel at the PGS-union loci — `pgs_panel_afreq` (per-variant panel effect-AF for mean-imputing missing variants; a byproduct of the scan) and `pgs_panel_ref` (per-PGS × superpop `{mean,sd,quantiles}` by scoring the panel per PGS — Spark-native, so `z_msp`/`z_admixed` need no offline plink2 curation and scale to 100+ PGS; leave `prs.yaml`'s `reference_distribution` empty to use it, or populate it to override with frozen stats). The union↔pgen match is driver-bounded per chromosome, so a genome-wide union won't OOM. Reads the HGDP+1kGP panel from the **pca** module's `pca_reference` volume (pca owns the ancestry reference). The frozen PCA basis itself (`pca_basis.npz`) is built by **`pca_v1`** (`ref_01_build_basis`, Spark-native: distributed QC + Hail-style windowed-r² LD-prune + FRAPOSA eigendecomposition) and consumed here as a read-only artifact — see `pca_v1`. Run `pca_reference_setup` before this. `00_download_pgs_scorefile` runs in `prs_initial_setup_job`.
- **`prs_maintenance`** — scheduled (paused) `maintain_stores`: OPTIMIZE + VACUUM + ANALYZE so MERGE versioning doesn't balloon.

## Key parameters (`prs_scoring`)

| param | meaning |
|---|---|
| `catalog` / `schema` | target Unity Catalog location |
| `vcf_dir` / `vcf_paths` | gVCFs to extract (dir, or explicit list) |
| `scorefile_dir` / `catalog_config_path` | curation inputs (scorefiles + `prs.yaml`) |
| `pgs_ids` / `samples` | scope the run to specific scores / samples (empty = all) |
| `basis_path` | `pca_basis.npz` for ancestry (default `${var.prs_pca_basis}`) |
| `apply` / `max_cells` / `confirm_large` | reconcile guardrails — **dry-run by default** |
| `missing_mode` | `mean_impute` (default; 2·AF from `pgs_panel_afreq`) or `drop` |
| `pgs_chunk_size` / `max_weight_rows_per_chunk` | PGS-axis bound (default off; n=10 A/B was slower) |
| `sample_chunk_size` | max samples per densify join (default **10**, the no-spill shape; `0` = all-at-once) |
| `sample_manifest_path` / `confirm_synthetic_fanout` | synthetic-only sample-axis benchmark; empty/false in production |
| `fanout_batch_size` / `max_fanout_rows` | bounded fan-out writes and pre-write row-count kill switch |

## Deploy

Requires `core` (catalog/schema, the `libraries` volume with the Glow JAR + `genesis_workbench`
wheel, and `settings`). From the repo root:

```
./deploy.sh genomics <cloud> --only-submodule prs/prs_v1
```

## Notes / scope

- **Incrementality is the cost control.** The full grid is scored once (backfill); thereafter
  reconcile computes only the increment. Guardrails are enforced in code (dry-run default,
  `max_cells` cap), not left to discipline.
- **A synthetic manifest is not an extraction benchmark.** Many logical IDs can share one
  canonical gVCF only for scale testing. True production extraction throughput requires distinct
  physical gVCFs; the manifest rung measures downstream sample cardinality.
- `variant_id` is orientation-canonical (`chrom:pos:effect:other`) in both `dosage` and
  `pgs_weights`, so the join can't silently miss on strand.
- **Reference tests** (`tests/`, `python -m pytest`) are dependency-light and run without a
  cluster: `test_prs_scoring.py` (allele-orientation math) and `test_prs_ancestry.py` (FRAPOSA
  fit/project + RF classify on a synthetic panel, and the `z_admixed` combiner).
- PRSmix+ with *distinct per-ancestry PGS variants* per trait (+ MyOme β/caPRS weighting) and
  clinical absolute-risk integration are deliberately out of scope for the generic module.
