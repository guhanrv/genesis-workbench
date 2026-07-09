# PCA — Ancestry / Population-structure PCA (`pca_v1`)

The population-structure PCA module. **One PCA engine** (`lib/pca_fit`) with two source
adapters: an **in-cohort** path (structure within a scored cohort — the covariates a GWAS
should adjust for; the `gwas` submodule currently runs **unadjusted**) and a **reference**
path (the frozen HGDP+1kGP basis the `prs` module projects members onto for ancestry
classification). Both emit the same **projectable model** contract; the reference path also
serves `prs` as a read-only Volume artifact.

## What it deploys

- **`pca_compute`** job — **in-cohort** PCA, two steps so Glow is isolated:
  - `00_ingest_vcf.py` — **classic cluster + Glow** (the only Glow step): reads the VCF,
    keeps biallelic SNPs, derives per-sample dosage, writes `pca_dosage_<run>`. Dosage source
    (`dosage_field`, default `auto`): **DS → HDS(summed) → GT** — imputed, dosage-only cohorts
    (no hard `GT`) work directly. Missing → `null`.
  - `01_compute_pca.py` → `02_save_results.py` — QC to common SNPs (MAF ≥ cutoff), then hand
    off to **`lib/pca_fit`** (LD-prune → FRAPOSA standardize → samples² `eigh` → loadings).
    Emits a projectable model `pca_model_<run>.npz` **and** `pca_components_<run>`
    (`sample_id, PC1..PCk`) — the covariates for GWAS.
- **`pca_reference_setup`** job — **reference** basis (one-time, on demand):
  - `ref_00_download_panel.py` → `ref_01_build_basis.py` — stage the pgsc HGDP+1kGP panel, then
    the SAME `lib/pca_fit` (pgenlib QC read this time) → `pca_basis.npz` on the `pca_reference`
    volume. Spark-native, **no plink2**. This is the ancestry basis `prs`'s
    `05_build_sample_ancestry` projects members onto.
- **`pca_initial_setup_job`** — registers the workflow.
- Volumes: `pca_data`, `pca_results`, `pca_reference` (panel + basis, read cross-module by `prs`).

## Parameters (`pca_compute`)

| param | default | meaning |
|---|---|---|
| `vcf_path` | — | cohort VCF |
| `n_components` | 10 | number of PCs to emit |
| `maf_cutoff` | 0.05 | minor-allele-frequency filter |
| `max_variants` | 0 | optional SNP cap for runtime (0 = all; variant count is unbounded with this orientation) |

## Deploy

Requires `core` (catalog/schema, the `libraries` volume with the Glow JAR + wheel,
`settings`). From the repo root:

```
./deploy.sh genomics <cloud> --only-submodule pca/pca_v1
```

## Notes / scope

- Two flows, one engine: in-cohort (`pca_compute`) and reference (`pca_reference_setup`)
  both call `lib/pca_fit`. Projecting members onto the reference basis + ancestry
  classification lives in `prs` (it's PRS-domain); `gwas` consumes the in-cohort scores.
- `lib/pca_fit` collects the pruned matrix to the driver for the `eigh` (required to emit
  projectable variant loadings, which `spark.ml.PCA` can't in this orientation). The
  covariance is `N×N` (samples²), so this is cohort/reference scale — fine on the reference
  cluster and on serverless for modest cohorts. A very large in-cohort GWAS PCA should run
  on a classic driver like the reference job, or move to a distributed-Gram backend (a
  tracked follow-up). LD-prune is now built in (`r2`/`window_bp`).
- The mean-impute + PCA method has a dependency-light reference + unit test in
  `tests/test_pca_reference.py` (numpy only): it confirms PC1 separates a synthetic
  two-population cohort. The Spark notebook itself runs only on a cluster.
