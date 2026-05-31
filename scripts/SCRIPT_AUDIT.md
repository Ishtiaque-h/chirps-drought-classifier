# Script Audit

This audit separates scientific methods code from local manuscript-preparation
helpers. A script is not public just because an internal note once referenced
it. Public scripts should let another researcher reproduce the claims; local
helpers may stay ignored in the working tree.

## Keep Public

### Core CHIRPS/SPI Pipeline

- `_dl_one_chirps_v3_monthly.sh`
- `download_chirps_v3_monthly.sh`
- `clip_to_cvalley_monthly.py`
- `make_spi_labels.py`
- `download_climate_indices.py`
- `build_dataset_forecast.py`
- `feature_config.py`
- `train_forecast_logreg.py`
- `train_forecast_rf.py`
- `train_forecast_xgboost.py`
- `train_forecast_xgb_spatial.py`
- `evaluate_forecast_skill.py`

### Reported Model And Feature Experiments

- `build_dataset_convlstm.py`
- `train_forecast_convlstm.py`
- `tune_forecast_convlstm.py`
- `run_edl_experiment.py`
- `run_feature_ablation.py`
- `download_era5_land_met_monthly.py`
- `run_met_feature_experiment.py`
- `run_met_spatial_feature_experiment.py`
- `download_era5_land_soil_moisture_monthly.py`
- `run_soil_moisture_feature_experiment.py`
- `download_era5_ivt_monthly.py`
- `build_ivt_monthly_features.py`
- `download_mjo_rmm.py`
- `run_atmos_feature_experiment.py`

These remain public while the manuscript discusses them as comparison,
sensitivity, or negative-result checkpoints. If those sections are removed from
the final paper, these can be revisited.

### Robustness, Validation, And Mechanism Analyses

- `region_config.py`
- `run_multiregion_xgb_experiment.py`
- `build_region_masks.py`
- `build_basin_masks.py`
- `analyze_multiregion_mechanisms.py`
- `run_temporal_robustness_audit.py`
- `download_era5_land_monthly.py`
- `validate_era5_spi.py`
- `validate_chirps_prism_cvalley.py`
- `validate_usdm.py`
- `evaluate_regional_forecast.py`
- `plot_case_study.py`
- `plot_spatial_skill.py`
- `xgb_shap_forecast_analysis.py`
- `analyze_spi12_regionalization.py`

### Seasonal, Operational, And Land-Surface Benchmarks

- `build_dataset_seasonal.py`
- `run_spi3_seasonal_experiment.py`
- `run_seasonal_longlead_experiment.py`
- `run_climate_index_sensitivity_experiment.py`
- `prepare_cpc_nmme_precip_anomaly_inputs.py`
- `prepare_cpc_nmme_precip_probability_inputs.py`
- `prepare_ncei_cfsv2_precip_forecast_inputs.py`
- `run_operational_precip_benchmark.py`
- `run_landsurface_forecast_benchmark.py`
- `audit_landsurface_forecast_archives.py`
- `run_gefsv12_landsurface_benchmark.py`
- `run_gefsv12_landsurface_sensitivity.py`
- `run_gefsv12_landsurface_stack_benchmark.py`
- `run_c3s_ecmwf_landsurface_benchmark.py`
- `run_landsurface_domain_transfer_benchmark.py`
- `run_memory_target_experiment.py`
- `run_transition_target_experiment.py`
- `run_evaluation_inflation_audit.py`

### Manuscript Result Generators

- `generate_master_results.py`
- `generate_manuscript_results.py`

These are the only public aggregation scripts. They create curated CSV/figure
artifacts under `results/report/` and `results/report/paper/`.

## Local-Only Helpers

The following scripts only assemble or summarize completed outputs. They are
kept in the workspace for convenience and are ignored by `.gitignore`:

- `build_seasonal_regional_summary.py`
- `audit_seasonal_regional_signal.py`
- `summarize_transition_experiments.py`
- `diagnose_landsurface_added_value.py`
- `build_regionalization_mechanism_tables.py`

If a helper becomes necessary for the final reproducibility protocol, either
merge its logic into `generate_manuscript_results.py` or promote it explicitly
in this audit.
