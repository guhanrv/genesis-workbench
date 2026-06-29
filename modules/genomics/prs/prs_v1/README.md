# PRS — Polygenic Risk Scoring (`prs_v1`)

Scores every sample in a VCF against a [PGS Catalog](https://www.pgscatalog.org/)
scoring file, entirely in Spark/Glow — the same modality as the `gwas` submodule
(no PLINK, no `pgsc_calc`).

```
PRS(sample) = Σ_variants  dosage_of_effect_allele(sample) × effect_weight
```

## What it deploys

- **`prs_scoring`** job, two steps so Glow is isolated and the rest runs serverless:
  - `00_ingest_vcf.py` — **classic cluster + Glow** (the only Glow step): reads the VCF,
    derives per-sample alt-allele dosage (`glow.genotype_states`), writes `prs_dosage_<run>`.
  - `01_score_prs.py` → `02_save_results.py` — **serverless** (no Glow): joins the dosage
    table to the scoring file on `(chrom, pos)`, orients dosage to the **effect allele**
    (`effect==alt → dosage`, `effect==ref → 2-dosage`, mismatches/missing dropped), sums
    `dosage × weight` per sample, standardizes within the cohort. Output: `prs_scores_<run>`
    (`sample_id, pgs_id, prs_raw, n_variants_matched, prs_z, prs_percentile`).
- **`prs_initial_setup_job`** — downloads one harmonized GRCh38 PGS Catalog scoring
  file (default `PGS000004`) into the `prs_reference` volume and registers the
  workflow.
- Volumes: `prs_reference`, `prs_data`, `prs_results`.

## Parameters (`prs_scoring`)

| param | meaning |
|---|---|
| `vcf_path` | cohort VCF to score |
| `scorefile_path` | PGS Catalog scoring file (harmonized `*_hmPOS_GRCh38.txt.gz`) |
| `pgs_id` | PGS Catalog id, recorded on the output + MLflow run |

## Deploy

Requires `core` (provides the catalog/schema, the `libraries` volume with the Glow
JAR + wheel, and `settings`). From the repo root:

```
./deploy.sh genomics <cloud> --only-submodule prs/prs_v1
```

## Notes / scope

- Biallelic sites only; positions matched on `(chrom, pos)` after stripping `chr`.
  Glow's `start` is 0-based, so the 1-based scoring-file position is `start + 1`.
- Standardization is **within the scored cohort**. Ancestry-specific reference
  distributions (e.g. PRSmix+/percentiles vs an external panel) are a downstream
  extension, not included here.
- The allele-orientation math has a dependency-free reference + unit test in
  `tests/test_prs_scoring.py` (`python -m pytest`), since the Spark notebook itself
  only runs on a cluster.
