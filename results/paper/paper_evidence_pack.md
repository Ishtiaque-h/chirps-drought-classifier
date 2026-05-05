# Paper Evidence Pack

This folder consolidates completed experiments into manuscript-facing tables and figures. It does not retrain models.

## Core Interpretation

The strongest current claim is a predictability and evaluation audit: lag-based ML detects drought-relevant structure, but it rarely converts that signal into robust calibrated monthly BSS over climatology once leakage-safe targets, validation-only calibration, regional masks, independent precipitation validation, and monthly bootstrap uncertainty are enforced.

## Key Counts

- Master evidence rows: 59
- Headline rows: 36
- Robust-positive rows in master evidence: 1
- Temporal holdouts with positive BSS: 0/5
- Best canonical Central Valley calibrated checkpoint: XGB-Spatial BSS +0.005 (CI -0.062 to +0.073)
- Best selected operational checkpoint: CPC NMME probability forecast selected SPI-1 dry fraction lead 1 BSS +0.131 (CI -0.304 to +0.338)
- Best raw CPC NMME probability checkpoint: SPI-6 dry fraction lead 6 BSS +0.035 (CI -0.430 to +0.255)
- Seasonal regional robust-positive exception: mediterranean_spain_basin_masked SPI-6 lead-6 BSS +0.078, but signal_flag=positive_calibration_shift

## Generated Files

- `table01_master_evidence.csv/md`: all consolidated evidence rows.
- `table02_headline_results.csv/md`: compact manuscript headline table.
- `table03_mask_methods.csv/md`: source-cited mask methods and retained-cell fractions.
- `table04_temporal_robustness.csv/md`: rolling holdout control for test-period non-representativeness.
- `table05_seasonal_signal_audit.csv/md`: seasonal BSS interpreted with event-tracking diagnostics.
- `table06_regionalization_mechanism.csv/md`: SPI-12 mechanism evidence joined with zone-level forecast diagnostics.
- `fig01_headline_bss_forest.png`: headline BSS forest plot.
- `fig02_multiregion_bss_forest.png`: multi-region selected BSS forest plot.
- `fig03_seasonal_bss_vs_tracking.png`: seasonal BSS vs event-tracking correlation.
- `fig04_temporal_holdout_bss.png`: temporal holdout BSS forest plot.
- `fig05_mask_retention.png`: retained-cell fractions for source-cited masks.
- `manuscript_results_discussion_draft.md`: prose draft for Results and Discussion.
- `methods_sources_and_evidence_index.md`: source/citation and claim-to-evidence checklist.

## Manuscript Guardrails

- Do not claim a broadly positive operational forecast model.
- Treat positive point estimates with confidence intervals crossing zero as hypothesis-generating.
- Treat the Mediterranean Spain SPI-6 seasonal robust-positive row as a calibration-shift exception unless follow-up diagnostics show stronger temporal tracking.
- Use regionalization and SHAP as mechanism evidence, not proof of calibrated forecast skill.
