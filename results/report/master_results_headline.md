# Master Results Headline Table

Generated from current project outputs by `scripts/build_master_results_table.py`.
BSS is monthly dry-fraction Brier Skill Score against calendar-month climatology.

| category | region | target | lead_months | model | calibration | n_test_months | bs_model | bss_vs_climatology | bss_ci_low | bss_ci_high | claim_status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| central_valley_calibrated_checkpoint | California Central Valley | SPI-1 dry fraction | 1 | XGB-Spatial | isotonic | 63 | 0.064 | 0.005 | -0.062 | 0.073 | positive_uncertain |
| central_valley_calibrated_checkpoint | California Central Valley | SPI-1 dry fraction | 1 | XGB | isotonic | 63 | 0.068 | -0.064 | -0.216 | 0.049 | not_distinguishable_from_climatology |
| central_valley_feature_extension | California Central Valley | SPI-1 dry fraction | 1 | XGB-Spatial + gridded temperature/VPD selected | selected | 63 | 0.067 | -0.047 | -0.131 | 0.023 | not_distinguishable_from_climatology |
| central_valley_feature_extension | California Central Valley | SPI-1 dry fraction | 1 | XGBoost + regional temperature/VPD selected | selected | 63 | 0.070 | -0.092 | -0.177 | -0.033 | robust_negative |
| central_valley_feature_extension | California Central Valley | SPI-1 dry fraction | 1 | XGBoost + soil moisture selected | selected | 63 | 0.074 | -0.158 | -0.385 | -0.049 | robust_negative |
| central_valley_feature_extension | California Central Valley | SPI-1 dry fraction | 1 | XGBoost + MJO/IVT selected | selected | 39 | 0.105 | -0.333 | -1.044 | -0.083 | robust_negative |
| central_valley_model_suite_raw | California Central Valley | SPI-1 dry fraction | 1 | Climatological baseline | raw_or_model_default | 63 | 0.064 | 0.000 |  |  | reference |
| central_valley_model_suite_raw | California Central Valley | SPI-1 dry fraction | 1 | Random Forest | raw_or_model_default | 63 | 0.069 | -0.068 | -0.405 | 0.097 | not_distinguishable_from_climatology |
| central_valley_model_suite_raw | California Central Valley | SPI-1 dry fraction | 1 | XGBoost-Spatial | raw_or_model_default | 63 | 0.072 | -0.115 | -0.769 | 0.223 | not_distinguishable_from_climatology |
| central_valley_model_suite_raw | California Central Valley | SPI-1 dry fraction | 1 | ConvLSTM | raw_or_model_default | 63 | 0.087 | -0.356 | -1.272 | 0.026 | not_distinguishable_from_climatology |
| central_valley_seasonal_target | California Central Valley | SPI-3 dry fraction | 3 | Seasonal SPI-3 lead-3 xgb_isotonic | xgb_isotonic | 59 | 0.084 | 0.036 | -0.127 | 0.140 | positive_uncertain |
| central_valley_seasonal_target | California Central Valley | SPI-3 dry fraction | 6 | Seasonal SPI-3 lead-6 xgb_isotonic | xgb_isotonic | 62 | 0.093 | -0.108 | -0.332 | 0.058 | not_distinguishable_from_climatology |
| central_valley_seasonal_target | California Central Valley | SPI-6 dry fraction | 6 | Seasonal SPI-6 lead-6 xgb_isotonic | xgb_isotonic | 62 | 0.141 | -0.110 | -0.350 | 0.031 | not_distinguishable_from_climatology |
| central_valley_uncertainty | California Central Valley | SPI-1 dry fraction | 1 | EDL MLP selected | selected | 63 | 0.072 | -0.116 | -0.207 | -0.059 | robust_negative |
| feature_ablation | California Central Valley | SPI-1 dry fraction | 1 | XGBoost ablation: enso | raw | 63 |  | -0.145 |  |  | negative_no_ci |
| feature_ablation | California Central Valley | SPI-1 dry fraction | 1 | XGBoost ablation: all_features | raw | 63 |  | -0.272 |  |  | negative_no_ci |
| feature_ablation | California Central Valley | SPI-1 dry fraction | 1 | XGBoost ablation: pr_lags | raw | 63 |  | -0.455 |  |  | negative_no_ci |
| feature_ablation | California Central Valley | SPI-1 dry fraction | 1 | XGBoost ablation: seasonality | raw | 63 |  | -0.463 |  |  | negative_no_ci |
| multi_region_selected_checkpoint | California Central Valley | SPI-1 dry fraction | 1 | spatial XGBoost | isotonic | 63 | 0.066 | -0.026 | -0.124 | 0.047 | not_distinguishable_from_climatology |
| multi_region_selected_checkpoint | California Central Valley (basin-mask sensitivity) | SPI-1 dry fraction | 1 | tabular XGBoost | isotonic | 63 | 0.070 | -0.021 | -0.129 | 0.042 | not_distinguishable_from_climatology |
| multi_region_selected_checkpoint | Horn of Africa bounding box (country-mask sensitivity) | SPI-1 dry fraction | 1 | spatial XGBoost | isotonic | 63 | 0.014 | -0.031 | -0.302 | 0.320 | not_distinguishable_from_climatology |
| multi_region_selected_checkpoint | Mediterranean Spain bounding box | SPI-1 dry fraction | 1 | tabular XGBoost | isotonic | 63 | 0.046 | 0.044 | -0.151 | 0.241 | positive_uncertain |
| multi_region_selected_checkpoint | Mediterranean Spain bounding box (basin-mask sensitivity) | SPI-1 dry fraction | 1 | spatial XGBoost | isotonic | 63 | 0.047 | -0.085 | -0.234 | 0.034 | not_distinguishable_from_climatology |
| multi_region_selected_checkpoint | Mediterranean Spain bounding box (country-mask sensitivity) | SPI-1 dry fraction | 1 | spatial XGBoost | isotonic | 63 | 0.048 | 0.024 | -0.199 | 0.305 | positive_uncertain |
| multi_region_selected_checkpoint | Murray-Darling Basin bounding box (basin-mask sensitivity) | SPI-1 dry fraction | 1 | tabular XGBoost | platt | 63 | 0.046 | -0.639 | -1.308 | -0.278 | robust_negative |
| multi_region_selected_checkpoint | Southern Great Plains | SPI-1 dry fraction | 1 | spatial XGBoost | isotonic | 63 | 0.052 | -0.082 | -0.148 | -0.002 | robust_negative |
| multi_region_selected_checkpoint | Southern Great Plains (basin-mask sensitivity) | SPI-1 dry fraction | 1 | spatial XGBoost | isotonic | 63 | 0.055 | 0.010 | -0.098 | 0.150 | positive_uncertain |
| operational_dynamical_benchmark | California Central Valley | SPI-1 dry fraction | 1 | CPC NMME anomaly forecast selected | selected | 63 | 0.064 | 0.002 | -0.438 | 0.239 | positive_uncertain |
| operational_dynamical_benchmark | California Central Valley | SPI-1 dry fraction | 1 | CPC NMME probability forecast selected | selected | 52 | 0.067 | 0.131 | -0.304 | 0.338 | positive_uncertain |
| operational_dynamical_benchmark | California Central Valley | SPI-1 dry fraction | 1 | CPC NMME probability forecast raw | raw | 52 | 0.096 | -0.248 | -1.223 | 0.095 | not_distinguishable_from_climatology |
| operational_dynamical_benchmark | California Central Valley | SPI-3 dry fraction | 3 | CPC NMME anomaly forecast selected | selected | 59 | 0.080 | 0.086 | -0.225 | 0.261 | positive_uncertain |
| operational_dynamical_benchmark | California Central Valley | SPI-3 dry fraction | 3 | CPC NMME probability forecast raw | raw | 51 | 0.094 | 0.007 | -0.549 | 0.243 | positive_uncertain |
| operational_dynamical_benchmark | California Central Valley | SPI-3 dry fraction | 3 | CPC NMME probability forecast selected | selected | 51 | 0.115 | -0.212 | -1.148 | 0.176 | not_distinguishable_from_climatology |
| operational_dynamical_benchmark | California Central Valley | SPI-6 dry fraction | 6 | CPC NMME anomaly forecast selected | selected | 62 | 0.171 | -0.344 | -1.130 | 0.113 | not_distinguishable_from_climatology |
| operational_dynamical_benchmark | California Central Valley | SPI-6 dry fraction | 6 | CPC NMME probability forecast raw | raw | 49 | 0.101 | 0.035 | -0.430 | 0.255 | positive_uncertain |
| operational_dynamical_benchmark | California Central Valley | SPI-6 dry fraction | 6 | CPC NMME probability forecast selected | selected | 49 | 0.147 | -0.409 | -1.693 | 0.163 | not_distinguishable_from_climatology |
