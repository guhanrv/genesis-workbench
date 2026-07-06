# PCA — Ancestry / Population-structure PCA (`pca_v1`)

Computes per-sample principal components from a cohort VCF, entirely in Spark/Glow
— the population-structure covariates a GWAS should adjust for. (The `gwas`
submodule currently runs **unadjusted** logistic regression; these PCs close that
gap and also provide a basis for ancestry analysis.)

## What it deploys

- **`pca_compute`** job, two steps so Glow is isolated and the rest runs serverless:
  - `00_ingest_vcf.py` — **classic cluster + Glow** (the only Glow step): reads the VCF,
    keeps biallelic SNPs, derives per-sample dosage, writes `pca_dosage_<run>`. Dosage source
    is chosen (`dosage_field` param, default `auto`): **DS → HDS(summed) → GT** — so imputed,
    dosage-only cohorts (no hard `GT`) work directly. Missing → `null`.
  - `01_compute_pca.py` → `02_save_results.py` — **serverless** (no Glow, no RDD): keeps
    common SNPs (MAF ≥ cutoff), mean-imputes/centers, and fits `pyspark.ml.feature.PCA`
    with the matrix oriented **variants-as-rows × samples-as-features** so the covariance
    is `N×N` (samples², small) regardless of variant count — distributed across variant
    rows, no driver blow-up. Per-sample coordinates are read directly from `model.pc`.
    Output: `pca_components_<run>` (`sample_id, PC1..PCk`).
- **`pca_initial_setup_job`** — registers the workflow.
- Volumes: `pca_data`, `pca_results`.

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

- **In-cohort** PCA (structure within the scored cohort). Projecting samples onto a
  fixed external reference panel (e.g. HGDP+1kGP) for absolute ancestry labels is a
  downstream extension, not included here.
- The covariance is `N×N` (samples²), so #samples should stay well under Spark ML's
  65535-feature limit (true for any real cohort); the **variant** count is unbounded.
  `max_variants` is an optional runtime cap, not a correctness constraint. For
  production ancestry PCA, LD-prune upstream.
- The mean-impute + PCA method has a dependency-light reference + unit test in
  `tests/test_pca_reference.py` (numpy only): it confirms PC1 separates a synthetic
  two-population cohort. The Spark notebook itself runs only on a cluster.
