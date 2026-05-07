| scope | transition | model | n_months | bss_vs_eligible_climatology | bss_ci_low | bss_ci_high | spearman | claim_status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cvalley | onset | all-pixel XGBoost isotonic, eligible score | 63 | +0.052 | -0.041 | +0.118 | +0.396 | positive_uncertain |
| cvalley | onset | eligible-only XGBoost isotonic | 63 | +0.104 | +0.025 | +0.167 | +0.402 | robust_positive |
| cvalley_basin_masked | onset | all-pixel XGBoost isotonic, eligible score | 63 | +0.011 | -0.119 | +0.087 | +0.389 | positive_uncertain |
| cvalley_basin_masked | onset | eligible-only XGBoost isotonic | 63 | -0.027 | -0.181 | +0.058 | +0.341 | not_distinguishable_from_climatology |
| mediterranean_spain_basin_masked | onset | all-pixel XGBoost isotonic, eligible score | 63 | -0.128 | -0.364 | +0.021 | +0.021 | not_distinguishable_from_climatology |
| mediterranean_spain_basin_masked | onset | eligible-only XGBoost isotonic | 63 | -0.035 | -0.124 | +0.016 | +0.014 | not_distinguishable_from_climatology |
| southern_great_plains_basin_masked | onset | all-pixel XGBoost isotonic, eligible score | 63 | -0.087 | -0.156 | -0.026 | +0.010 | robust_negative |
| southern_great_plains_basin_masked | onset | eligible-only XGBoost isotonic | 63 | -0.084 | -0.147 | -0.016 | +0.113 | robust_negative |
| cvalley | termination | all-pixel XGBoost isotonic, eligible score | 51 | -1.368 | -3.498 | -0.372 | -0.001 | robust_negative |
| cvalley | termination | eligible-only XGBoost isotonic | 51 | -0.031 | -0.285 | +0.178 | +0.152 | not_distinguishable_from_climatology |
