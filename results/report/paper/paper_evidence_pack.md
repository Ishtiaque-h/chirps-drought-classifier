# Paper Evidence Pack

This folder consolidates completed experiments into manuscript-facing tables and figures. It does not retrain models.

## Core Interpretation

The strongest current claim is two-tiered: lag-based ML rarely converts drought-relevant structure into robust calibrated SPI dry-fraction BSS once leakage-safe evaluation controls are enforced, while land-surface dry-fraction targets are more predictable. CFSv2 root-zone soil-moisture forecasts are robustly positive in Central Valley, but the current monthly-mean extraction does not establish added value over same-target persistence and does not generalize cleanly across the added regions.

## Key Counts

- Master evidence rows: 78
- Headline rows: 55
- Robust-positive rows in master evidence: 6
- Temporal holdouts with positive BSS: 0/5
- Best canonical Central Valley calibrated checkpoint: XGB-Spatial BSS +0.005 (CI -0.062 to +0.073)
- Best selected operational checkpoint: CPC NMME probability forecast selected SPI-1 dry fraction lead 1 BSS +0.131 (CI -0.314 to +0.325)
- Best raw CPC NMME probability checkpoint: SPI-6 dry fraction lead 6 BSS +0.035 (CI -0.444 to +0.268)
- Best CFSv2 land-surface checkpoint: NCEI CFSv2 soil-moisture forecast selected California Central Valley lead 1 BSS +0.630 (CI +0.477 to +0.762)
- Best same-target land-surface persistence checkpoint: Southern Great Plains lead 1 BSS +0.695 (CI +0.414 to +0.851)
- Seasonal regional robust-positive exception: mediterranean_spain_basin_masked SPI-6 lead-6 BSS +0.078, but signal_flag=positive_calibration_shift
- Memory-target checkpoint: best selected SPI-6 lead-6 XGBoost BSS +0.040 (CI -0.020 to +0.082); soil-memory selected BSS -0.158
- Evaluation-inflation audit: strict monthly SPI-1 BSS -0.023; invalid random-row monthly BSS +0.995; invalid overlapping SPI-3 lead-1 monthly BSS +0.674
- Transition-target diagnostic: rectangular Central Valley eligible-only onset BSS +0.104 (CI +0.025 to +0.167), but SGP basin onset BSS -0.084 (CI -0.147 to -0.016)
- Land-surface added-value diagnostic: Central Valley delta BS (CFSv2 minus raw persistence) +0.001 with CI crossing zero; Southern Great Plains delta BS +0.020 (CI +0.003 to +0.036), favoring persistence

## Generated Files

- `table01_master_evidence.csv`: all consolidated evidence rows.
- `table02_headline_results.csv`: compact manuscript headline table.
- `table03_mask_methods.csv`: source-cited mask methods and retained-cell fractions.
- `table04_temporal_robustness.csv`: rolling holdout control for test-period non-representativeness.
- `table05_seasonal_signal_audit.csv`: seasonal BSS interpreted with event-tracking diagnostics.
- `table06_regionalization_mechanism.csv`: SPI-12 mechanism evidence joined with zone-level forecast diagnostics.
- `table07_evaluation_inflation_audit.csv`: invalid-protocol audit showing skill inflation from random row splits, pixel-level inference, and overlapping SPI targets.
- `table08_transition_target_summary.csv`: onset/termination eligible-pixel transition diagnostic and replication checks.
- `table09_landsurface_added_value.csv`: CFSv2 root-zone soil-moisture added-value diagnostics against same-target persistence.
- `fig01_headline_bss_forest.png`: headline BSS forest plot.
- `fig02_multiregion_bss_forest.png`: multi-region selected BSS forest plot.
- `fig03_seasonal_bss_vs_tracking.png`: seasonal BSS vs event-tracking correlation.
- `fig04_temporal_holdout_bss.png`: temporal holdout BSS forest plot.
- `fig05_mask_retention.png`: retained-cell fractions for source-cited masks.
- `manuscript_methods_draft.md`: source-cited Methods draft tied to the current evidence tables.
- `manuscript_claims_audit.md`: allowed-claims and overclaim-risk checklist.
- `manuscript_results_discussion_draft.md`: prose draft for Results and Discussion.
- `methods_sources_and_evidence_index.md`: source/citation and claim-to-evidence checklist.

## Manuscript Guardrails

- Do not generalize the positive land-surface benchmark to precipitation-index skill, all leads, all regions, or deployment readiness.
- Do not claim CFSv2 root-zone soil-moisture added value over persistence from the current monthly-mean extraction.
- Treat transition-target onset as a target-design diagnostic; it does not survive basin/regional replication.
- Treat positive point estimates with confidence intervals crossing zero as hypothesis-generating.
- Treat the Mediterranean Spain SPI-6 seasonal robust-positive row as a calibration-shift exception unless follow-up diagnostics show stronger temporal tracking.
- Use regionalization and SHAP as mechanism evidence, not proof of calibrated forecast skill.
