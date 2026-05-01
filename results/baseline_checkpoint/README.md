# Baseline Checkpoint — Post-Fix Canonical Result Set

**Date:** 2026-04-27  
**Branch:** copilot/improve-project-based-on-analysis  
**Status:** Frozen baseline — do not overwrite; compare future experiments against these values.

---

## Summary

This checkpoint captures the canonical post-fix evaluation results for the
1-month-ahead Central Valley drought forecast pipeline (CHIRPS v3, 1991–2026).

It supersedes earlier runs in which the calibration study produced invalid rows
(`val_BS_selected = inf`, empty `best_calibration`). Those failures were caused
by timestamp-alignment issues between validation-set predictions and the monthly
groupby aggregation. The corrected pipeline now produces well-formed calibration
results.

---

## Key Findings

### Calibration study (`calib_study_results.csv`)

| Model       | Best calibration | Val BS (selected) | Test BS  | Test BSS  |
|-------------|-----------------|-------------------|----------|-----------|
| XGB         | isotonic        | 0.06473           | 0.07905  | −0.22983  |
| XGB-Spatial | isotonic        | 0.06443           | 0.07455  | −0.15977  |

**Interpretation:**
- Isotonic regression is consistently selected over Platt scaling on the validation set.
- Both calibrated models remain below climatology (BSS < 0).
- XGB-Spatial is the best ML option, but its advantage over climatology is small
  and not statistically significant (p > 0.05 in the paired bootstrap test).

### Season-stratified BSS

- **MAM (spring):** Multiple models show positive BSS (~0.18–0.31), indicating
  meaningful conditional skill during the critical wet-to-dry transition months.
- **Other seasons:** BSS is negative or near zero; model skill is absent outside spring.

### ENSO-stratified BSS

Previously empty (`stratifier,group,n_months` header only) due to a bug: when
Niño 3.4 was not a training feature, the ENSO-phase column was set to "Unavailable"
and the stratification was skipped entirely.

**Fix applied:** `evaluate_forecast_skill.py` now loads `climate_indices_monthly.csv`
for ENSO-phase assignment even when `nino34_lag1` was not included as a training feature.
Rerun `evaluate_forecast_skill.py` after running `download_climate_indices.py` to
obtain populated ENSO-stratified results.

---

## Next Experiments

1. **ENSO/PDO-enhanced training** (primary)
   - Run `download_climate_indices.py` to fetch Niño 3.4 and PDO monthly series.
   - Run `build_dataset_forecast.py` to merge ENSO/PDO lags into the feature matrix.
   - Retrain all models; evaluate with the same split and monthly BSS protocol.
   - **Hypothesis:** Exogenous climate drivers move monthly dry-class BSS above climatology.

2. **Feature ablation**
   - Run `scripts/run_feature_ablation.py` (new) to quantify the marginal BSS
     contribution of each feature group (SPI lags, precip lags, seasonality, ENSO, PDO).

3. **ENSO-stratified diagnostics**
   - After step 1, rerun `evaluate_forecast_skill.py` to populate
     `forecast_skill_stratified_enso.csv` with meaningful group counts.

---

## What NOT to change

- Do not add new model architectures yet; current evidence points to
  information-content limits, not model-capacity limits.
- Do not tune XGBoost hyperparameters; the BS gap vs climatology is not a
  tuning problem.
