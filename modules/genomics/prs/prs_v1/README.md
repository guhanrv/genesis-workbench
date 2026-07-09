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

Six tasks, all on **classic** job clusters (fixed `num_workers`, started minimal + titrated;
**never serverless** — its unbounded autoscale is the #1 bill risk). Ephemeral (auto-terminate).

```
setup_stores → register → extract_dosage → reconcile ──────────┐
                                        └→ build_sample_ancestry ┴→ score_prs → save_results → mark_success/failure
```

- **`00_setup_stores`** — idempotent DDL for the stores above.
- **`00_register_catalog`** — curation (`prs.yaml` + scorefiles) → `pgs_registry` / `pgs_weights` / `pgs_panel_ref`. Palindromic (A/T, C/G) SNPs dropped (strand-ambiguous).
- **`00_extract_dosage`** — per-sample **gVCF** → `dosage`, one Spark task per (sample × chrom). Uses the ported pysam **END-block + FASTA** kernel (`lib/gvcf_dose.py`): plink2 and Glow both drop gVCF `END=` REF blocks (~70% coverage loss), so this can't be Glow/SQL. Incremental: skips samples already extracted. *(Non-gVCF cohorts use the sibling engine `00_ingest_vcf` — Glow `DS/HDS/GT`, imputed-dosage or regular hard-called VCF — which orients + MERGEs into the SAME `dosage` store, so everything downstream is identical.)*
- **`02_reconcile`** — the incremental brain + cost kill-switch. Anti-joins the desired `(sample × pgs)` grid vs `prs_scores` → emits **only missing/stale cells**. **Dry-run by default** (`apply=false`): prints the cell count + estimated cost and scores nothing; a plan larger than `max_cells` additionally needs `confirm_large=true`. A stray run can never fire the full grid.
- **`05_build_sample_ancestry`** — OADP-projects each sample onto the frozen FRAPOSA PCA basis (`lib/prs_ancestry.py`) → RF most-similar-pop → `sample_ancestry`. Extraction distributed like `extract_dosage`; RF classify on the driver.
- **`01_score_prs`** — SQL join-aggregate over the reconcile plan → `raw`, then reference-panel `z_msp` + RF-posterior-weighted `z_admixed` → MERGE `prs_scores`. Default `missing_mode=drop`; `mean_impute` uses `pgs_panel_afreq` (2·AF).
- **`02_save_results`** — read `prs_scores` → log cohort summary metrics (coverage, z_msp/z_admixed counts, raw distribution) to the MLflow run. `mark_success` / `mark_failure` then set the run's `job_status` (framework convention).

## Two ingest engines (both write the shared `dosage` store)

| cohort | engine | notebook |
|---|---|---|
| **gVCF** (`END=` REF blocks) | pysam END-block + FASTA kernel | `00_extract_dosage` (in the scoring DAG) |
| **imputed dosage / regular hard-called VCF** (`DS`/`HDS`/`GT`) | Glow, distributed | `00_ingest_vcf` (guards against gVCF input) |

## Companion jobs

- **`prs_reference_setup`** — one-time scoring-reference builder (run on demand): `04_build_panel_afreq` → `pgs_panel_afreq` (panel allele frequencies for mean-imputing missing PGS variants). It reads the HGDP+1kGP panel from the **pca** module's `pca_reference` volume (pca owns the ancestry reference). The frozen PCA basis itself (`pca_basis.npz`) is built by **`pca_v1`** (`ref_01_build_basis`, Spark-native: distributed QC + Hail-style windowed-r² LD-prune + FRAPOSA eigendecomposition) and consumed here as a read-only artifact — see `pca_v1`. Run `pca_reference_setup` before this. `00_download_pgs_scorefile` runs in `prs_initial_setup_job`.
- **`prs_maintenance`** — scheduled (paused) `03_maintain_stores`: OPTIMIZE + VACUUM + ANALYZE so MERGE versioning doesn't balloon.

## Key parameters (`prs_scoring`)

| param | meaning |
|---|---|
| `catalog` / `schema` | target Unity Catalog location |
| `vcf_dir` / `vcf_paths` | gVCFs to extract (dir, or explicit list) |
| `scorefile_dir` / `catalog_config_path` | curation inputs (scorefiles + `prs.yaml`) |
| `pgs_ids` / `samples` | scope the run to specific scores / samples (empty = all) |
| `basis_path` | `pca_basis.npz` for ancestry (default `${var.prs_pca_basis}`) |
| `apply` / `max_cells` / `confirm_large` | reconcile guardrails — **dry-run by default** |
| `missing_mode` | `drop` (default) or `mean_impute` (2·AF from `pgs_panel_afreq`) |

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
- `variant_id` is orientation-canonical (`chrom:pos:effect:other`) in both `dosage` and
  `pgs_weights`, so the join can't silently miss on strand.
- **Reference tests** (`tests/`, `python -m pytest`) are dependency-light and run without a
  cluster: `test_prs_scoring.py` (allele-orientation math) and `test_prs_ancestry.py` (FRAPOSA
  fit/project + RF classify on a synthetic panel, and the `z_admixed` combiner).
- PRSmix+ with *distinct per-ancestry PGS variants* per trait (+ MyOme β/caPRS weighting) and
  clinical absolute-risk integration are deliberately out of scope for the generic module.
