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
| `max_variants` | 0 | optional SNP cap for runtime (0 = all; applied deterministically after an order-by) |
| `r2` | 0.05 | LD-prune r² threshold (prune a variant with r² ≥ this to a kept one in-window) |
| `window_bp` | 1000000 | LD-prune window (bp) |
| `backend` | `driver` | fit backend: `driver` (collect + `eigh`), `distributed` (samples² Gram via RowMatrix; large cohorts), or `randomized` (matrix-free RSVD; biobank scale — never forms the Gram) |

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
- **Both entry points run on classic single-node clusters** (not serverless): the `driver` backend
  collects the pruned matrix to the driver for the `eigh` (needed to emit projectable variant
  loadings, which `spark.ml.PCA` can't in this orientation), and the `distributed` backend uses the
  RDD/MLlib API — neither works on serverless. Only `pca_compute`'s `save_results`/`mark_*` stay
  serverless (they just read a table / log MLflow).
- Three backends, one output contract: `driver` (default) for cohort/reference scale; `distributed`
  (`backend=distributed`) computes the samples² Gram via `RowMatrix` + loadings via a distributed map,
  never collecting the full variant × sample matrix — its win over `driver` is relieving the
  *variant*-axis driver-memory ceiling for large in-cohort GWAS PCA. The samples² Gram itself must
  still fit the driver (~tens of thousands of samples — the intrinsic limit of a samples² PCA), which
  is the wall both hit at **biobank scale** (a 100k² Gram is ~80 GB, 500k² ~2 TB — infeasible to build
  *or* eigh). `randomized` (`backend=randomized`) is the matrix-free path for that regime: Halko RSVD
  over distributed matvecs against X that **never forms the Gram** — a seeded random sketch, a few
  power iterations (`rsvd_power_iter`, default 2) and oversampling (`rsvd_oversample`, default 10),
  then a small `svd` on the driver. It lifts the *sample*-axis ceiling to biobank size in a fixed ~4–6
  Spark passes (vs Lanczos/IRAM's O(k) sequential matvecs — RSVD is chosen because this harness is
  orchestration-bound; an out-of-core IRAM backend is a further follow-up). Approximate top-k PCA, not
  bit-identical, so the reference basis stays on `driver`. The
  `pca_compute_cluster` is single-node (`num_workers: 0`) by default — `distributed` already helps
  there; **titrate `num_workers` up for horizontal shuffle/compute scale**. LD-prune is built in
  (`r2`/`window_bp`); QC is autosomal-only.
- The mean-impute + PCA method has a dependency-light reference + unit test in
  `tests/test_pca_reference.py` (numpy only): it confirms PC1 separates a synthetic
  two-population cohort. The Spark notebook itself runs only on a cluster.
