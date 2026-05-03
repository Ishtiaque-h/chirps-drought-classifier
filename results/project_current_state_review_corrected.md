# Project Current-State Review — Corrected

Generated UTC: 2026-05-03T00:46:12.003797+00:00

Repository reviewed: `Ishtiaque-h/chirps-drought-classifier`  
Default branch reviewed: `v2.0`

## Correction

The previous review was incomplete. I missed the committed SPI-6 lead-6 artifact in:

```text
results/seasonal/seasonal_spi6_lead6_experiment_scores.txt
```

The corrected review below includes that result.

## Current state summary

The project is now a rigorous drought predictability-limit study rather than only a Central Valley classifier. Current implemented tracks include:

- canonical leakage-safe SPI-1 lead-1 forecasting,
- multi-region XGBoost evaluation with mask sensitivity,
- SPI-12 regionalization with PCA, k-means, run theory, and Niño3.4/SOI/PDO correlations,
- canonical climate-index ingestion for Niño3.4, PDO, and SOI,
- leakage-safe SPI-3 lead-3 and SPI-6 lead-6 seasonal experiments,
- MJO/AR/IVT atmospheric feature experiments,
- EDL uncertainty quantification.

## Key reviewed result artifacts

- `results/corrected_enso_spatial_checkpoint/forecast_skill_scores.txt`
- `results/multiregion/multiregion_summary.csv`
- `results/regionalization/*/zone_climate_index_correlations.csv`
- `results/report/seasonal_spi3_experiment_scores.txt`
- `results/seasonal/seasonal_spi6_lead6_experiment_scores.txt`
- `results/atmos/atmos_feature_xgb_experiment_scores.txt`
- `results/edl/edl_metrics.txt`

## Corrected seasonal long-lead assessment

The seasonal long-lead pipeline is scientifically important because it enforces:

```text
lead_months >= target_spi_window
```

unless `--allow-overlap` is explicitly passed.

### SPI-3 lead-3

Artifact:

```text
results/report/seasonal_spi3_experiment_scores.txt
```

Key BSS values:

- Persistence SPI-3: `-0.7810`
- XGBoost raw: `-0.4243`
- XGBoost isotonic: `-0.1271`

Interpretation: leakage-safe SPI-3 lead-3 remains below climatology after calibration.

### SPI-6 lead-6

Artifact:

```text
results/seasonal/seasonal_spi6_lead6_experiment_scores.txt
```

Key BSS values:

- Persistence SPI-3: `-0.5927`
- XGBoost raw: `-0.1956`
- XGBoost isotonic: `-0.1428`

Interpretation: leakage-safe SPI-6 lead-6 also remains below climatology, strengthening the predictability-limit narrative.

## Updated scientific conclusion

The evidence supports this thesis:

> A rigorous, leakage-free drought ML evaluation across hydroclimates shows that probability skill is limited relative to monthly climatology; mechanism diagnostics explain why skill varies across regions and teleconnection regimes.

## Updated organization note

Seasonal results are currently split:

```text
results/report/seasonal_spi3_experiment_scores.txt
results/seasonal/seasonal_spi6_lead6_experiment_scores.txt
```

For clarity, mirror or move SPI-3 lead-3 into:

```text
results/seasonal/seasonal_spi3_lead3_experiment_scores.txt
```

Then keep all long-lead seasonal artifacts under `results/seasonal/`.

## Recommended next implementation step

Create a robust master evidence aggregation script that scans all committed result folders, especially:

```text
results/corrected_enso_spatial_checkpoint/
results/multiregion/
results/regionalization/
results/seasonal/
results/report/
results/atmos/
results/edl/
```

and writes:

```text
results/project_current_state_summary.csv
results/project_current_state_review.md
```

This should be generated from artifacts automatically rather than manually inferred, so committed files are not missed.
