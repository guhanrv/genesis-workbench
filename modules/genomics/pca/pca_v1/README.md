# PCA — Ancestry / Population-structure PCA (`pca_v1`)

Computes per-sample principal components from a cohort VCF, entirely in Spark/Glow
— the population-structure covariates a GWAS should adjust for. (The `gwas`
submodule currently runs **unadjusted** logistic regression; these PCs close that
gap and also provide a basis for ancestry analysis.)

## What it deploys

- **`pca_compute`** job — `01_compute_pca.py` → `02_save_results.py`. Reads a VCF
  with Glow, derives per-sample dosage (`glow.genotype_states`), keeps biallelic
  common SNPs (MAF ≥ cutoff), downsamples to `max_variants` (Spark ML's PCA
  covariance is dense, so columns must stay < 65535), mean-imputes missing dosage,
  assembles a per-sample sparse vector, and fits `pyspark.ml.feature.PCA`. Output:
  `pca_components_<run>` Delta table (`sample_id, PC1..PCk`).
- **`pca_initial_setup_job`** — registers the workflow.
- Volumes: `pca_data`, `pca_results`.

## Parameters (`pca_compute`)

| param | default | meaning |
|---|---|---|
| `vcf_path` | — | cohort VCF |
| `n_components` | 10 | number of PCs to emit |
| `maf_cutoff` | 0.05 | minor-allele-frequency filter |
| `max_variants` | 50000 | SNP cap (Spark ML PCA requires < 65535 columns) |

## Deploy

Requires `core` (catalog/schema, the `libraries` volume with the Glow JAR + wheel,
`settings`). From the repo root:

```
./deploy.sh genomics <cloud> --only-submodule pca/pca_v1
```

## Notes / scope

- **In-cohort** PCA (structure within the scored cohort). Projecting samples onto a
  fixed external reference panel (e.g. HGDP+1kGP) for absolute ancestry labels is a
  downstream extension, not included here.
- Downsampling to `max_variants` is a uniform deterministic cut ordered by variant
  key — not LD-pruning. For production ancestry PCA, LD-prune upstream.
- The mean-impute + PCA method has a dependency-light reference + unit test in
  `tests/test_pca_reference.py` (numpy only): it confirms PC1 separates a synthetic
  two-population cohort. The Spark notebook itself runs only on a cluster.
