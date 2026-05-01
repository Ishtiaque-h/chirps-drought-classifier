# Corrected ENSO + XGBoost-Spatial Checkpoint

**Date:** 2026-05-01  
**Status:** Current canonical corrected result set.

This checkpoint supersedes the 2026-04-27 baseline checkpoint. It includes:

- PDO sentinel handling fixed (`-9.9`, `-99.99`, `-999` -> missing)
- trailing missing PDO values left missing rather than forward-filled
- Niño3.4 absolute SST converted to monthly anomalies using 1991-2020 climatology
- active forecast dataset built with `--climate-features nino34`
- LogReg, RF, XGBoost, and XGBoost-Spatial retrained on the corrected ENSO-only schema
- XGBoost-Spatial evaluated from raw probabilities in the main skill table and recalibrated only in the calibration study
- ConvLSTM arrays rebuilt with the corrected `t+1` target alignment and ConvLSTM retrained

## Headline

The corrected ENSO + spatial model nearly ties climatology but does not beat it reliably.

| Model | Calibration | Test BS | Test BSS | 95% CI |
|---|---|---:|---:|---|
| XGB | isotonic | 0.06839 | -0.06395 | [-0.2165, 0.0492] |
| XGB-Spatial | isotonic | 0.06394 | +0.00525 | [-0.0622, 0.0733] |
| Climatology | reference | 0.0643 | 0.00000 | - |

XGB-Spatial has the best ranking skill so far (`ROC-AUC = 0.7432`), but the calibrated BSS confidence interval crosses zero and the paired bootstrap test vs climatology is not significant (`p = 0.881`).

The corrected-target ConvLSTM is valid again but remains below climatology
(`BSS = -0.3565`, `ROC-AUC = 0.5886`) and does not change the headline.

## Included Files

- `forecast_skill_scores.txt`
- `forecast_skill_bss_hss_table.csv`
- `calib_study_results.csv`
- `forecast_skill_stratified_season.csv`
- `forecast_skill_stratified_enso.csv`
- `calib_study_stratified_season.csv`

## Interpretation

This checkpoint supports a refined conclusion: corrected ENSO anomalies and local spatial context add useful ranking information and reduce Brier error, but they do not establish statistically reliable positive probability skill over climatology for 1-month-ahead SPI-1 drought prediction in the Central Valley.
