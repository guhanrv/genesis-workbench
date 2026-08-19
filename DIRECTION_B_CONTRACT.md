# Direction B contract — ours as core, their packaging

**Branch:** `feature/genomics-pca-prs` (this repo). Do not start from `gwb_repo_share`.

## Core (ours — do not regress)

- Member genomes enter via `gvcf_dose` END-block + FASTA expansion (`02_extract_dosage`). Glow `00_ingest_vcf` is only for imputed / hard-called VCF. `allow_gvcf=false`.
- `03_reconcile` dry-run default + `max_cells` / `confirm_large`.
- PRS/PCA compute on classic SINGLE_USER clusters (no serverless default for extract / ancestry / score / basis).
- GWAS `pca_scores_table` covariates and **no** `%pip install glow --force-reinstall`.

## Packaging (theirs — ported)

- One numpy FRAPOSA home: `pca_v1/lib/fraposa.py` (licenses in-file).
- `ref_01_build_basis` still writes `pca_basis.npz` **and** registers `ancestry_pca` + `ancestry_classifier` at `@champion`.
- Ancestry serve prefers those models when present; npz + `gvcf_dose` extract remains the fallback.
- Fail-loud empty PGS overlap after extract.
- Score/readiness gate: models (or npz), `pgs_weights`, `pgs_panel_ref` for the planned `panel_version`.
- Declarative app-SP `CAN_MANAGE_RUN` on deploy.
- README citations for FRAPOSA / pgsc_calc / PGS Catalog / GIAB.

## Explicit non-goals

- Glow-only ingest as the member-gVCF path.
- Serverless scoring by default.
- Dropping `pca_scores_table` or restoring glow `--force-reinstall`.
- Replacing reconcile with “just MERGE and hope”.
