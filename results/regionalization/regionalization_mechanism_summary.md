# SPI-12 Regionalization Mechanism Summary

Method context: diagnostic SPI-12 regionalization follows the same broad structure as recent CHIRPS drought regionalization work: SPI-12, PCA/k-means zones, run-theory drought metrics, and climate-index correlation checks. These diagnostics do not alter the frozen SPI-1 lead-1 forecast target.

## Zone Mechanism Highlights

| run_slug | zone | pixel_fraction | max_duration_months | mean_intensity | top_index | top_lag_months | top_pearson_r |
| --- | --- | --- | --- | --- | --- | --- | --- |
| cvalley | 2 | 0.143 | 26.000 | 0.249 | pdo | 0 | 0.346 |
| cvalley | 1 | 0.317 | 21.000 | 0.374 | nino34 | 6 | 0.290 |
| cvalley | 0 | 0.290 | 21.000 | 0.456 | pdo | 6 | 0.254 |
| cvalley | 3 | 0.250 | 17.000 | 0.195 | pdo | 0 | 0.172 |
| cvalley_basin_masked | 2 | 0.251 | 22.000 | 0.416 | pdo | 6 | 0.301 |
| cvalley_basin_masked | 0 | 0.309 | 20.000 | 0.333 | pdo | 6 | 0.294 |
| cvalley_basin_masked | 1 | 0.206 | 21.000 | 0.412 | nino34 | 6 | 0.281 |
| cvalley_basin_masked | 3 | 0.234 | 20.000 | 0.457 | pdo | 6 | 0.231 |
| horn_of_africa_country_masked | 0 | 0.385 | 6.000 | 0.185 | nino34 | 6 | 0.542 |
| horn_of_africa_country_masked | 3 | 0.274 | 19.000 | 0.247 | nino34 | 6 | 0.356 |
| horn_of_africa_country_masked | 1 | 0.208 | 12.000 | 0.239 | nino34 | 0 | -0.289 |
| horn_of_africa_country_masked | 2 | 0.133 | 13.000 | 0.286 | pdo | 0 | -0.168 |
| mediterranean_spain_basin_masked | 3 | 0.176 | 11.000 | 0.306 | pdo | 0 | 0.187 |
| mediterranean_spain_basin_masked | 0 | 0.332 | 11.000 | 0.299 | soi | 0 | 0.124 |
| mediterranean_spain_basin_masked | 2 | 0.279 | 10.000 | 0.278 | nino34 | 1 | 0.118 |
| mediterranean_spain_basin_masked | 1 | 0.214 | 12.000 | 0.280 | nino34 | 3 | 0.094 |
| murray_darling_basin_masked | 0 | 0.199 | 14.000 | 0.349 | nino34 | 1 | -0.460 |
| murray_darling_basin_masked | 3 | 0.218 | 25.000 | 0.207 | soi | 1 | 0.393 |
| murray_darling_basin_masked | 2 | 0.302 | 14.000 | 0.262 | nino34 | 3 | -0.358 |
| murray_darling_basin_masked | 1 | 0.282 | 7.000 | 0.178 | nino34 | 3 | -0.215 |
| southern_great_plains_basin_masked | 3 | 0.208 | 15.000 | 0.224 | nino34 | 6 | 0.501 |
| southern_great_plains_basin_masked | 1 | 0.298 | 17.000 | 0.366 | nino34 | 3 | 0.491 |
| southern_great_plains_basin_masked | 0 | 0.218 | 29.000 | 0.308 | pdo | 6 | 0.453 |
| southern_great_plains_basin_masked | 2 | 0.277 | 13.000 | 0.334 | pdo | 3 | 0.386 |

## Zone Forecast Diagnostics

| run_slug | model | zone | obs_dry_mean | pred_selected_dry_mean | selected_bias | selected_corr | selected_bss_vs_zone_climatology |
| --- | --- | --- | --- | --- | --- | --- | --- |
| cvalley | spatial | 3 | 0.187 | 0.143 | -0.043 | 0.261 | -0.014 |
| cvalley | spatial | 0 | 0.186 | 0.140 | -0.046 | 0.257 | -0.030 |
| cvalley | spatial | 1 | 0.158 | 0.133 | -0.025 | 0.346 | -0.035 |
| cvalley | spatial | 2 | 0.159 | 0.144 | -0.015 | 0.098 | -0.114 |
| cvalley | tabular | 3 | 0.187 | 0.147 | -0.040 | 0.230 | -0.018 |
| cvalley | tabular | 0 | 0.186 | 0.145 | -0.041 | 0.194 | -0.062 |
| cvalley | tabular | 1 | 0.158 | 0.139 | -0.019 | 0.246 | -0.088 |
| cvalley | tabular | 2 | 0.159 | 0.159 | -0.000 | 0.154 | -0.148 |
| cvalley_basin_masked | spatial | 1 | 0.195 | 0.143 | -0.052 | 0.259 | -0.010 |
| cvalley_basin_masked | spatial | 3 | 0.195 | 0.138 | -0.057 | 0.287 | -0.017 |
| cvalley_basin_masked | spatial | 2 | 0.168 | 0.122 | -0.046 | 0.383 | -0.033 |
| cvalley_basin_masked | spatial | 0 | 0.172 | 0.132 | -0.039 | 0.322 | -0.041 |
| cvalley_basin_masked | tabular | 1 | 0.195 | 0.136 | -0.059 | 0.291 | -0.005 |
| cvalley_basin_masked | tabular | 3 | 0.195 | 0.133 | -0.062 | 0.301 | -0.017 |
| cvalley_basin_masked | tabular | 2 | 0.168 | 0.117 | -0.051 | 0.399 | -0.029 |
| cvalley_basin_masked | tabular | 0 | 0.172 | 0.127 | -0.045 | 0.346 | -0.032 |
| horn_of_africa_country_masked | spatial | 2 | 0.098 | 0.079 | -0.019 | 0.209 | 0.150 |
| horn_of_africa_country_masked | spatial | 1 | 0.111 | 0.079 | -0.032 | 0.205 | -0.014 |
| horn_of_africa_country_masked | spatial | 3 | 0.140 | 0.083 | -0.057 | 0.145 | -0.050 |
| horn_of_africa_country_masked | spatial | 0 | 0.111 | 0.075 | -0.036 | 0.156 | -0.125 |
| horn_of_africa_country_masked | tabular | 2 | 0.098 | 0.079 | -0.019 | 0.192 | 0.144 |
| horn_of_africa_country_masked | tabular | 1 | 0.111 | 0.080 | -0.031 | 0.212 | -0.010 |
| horn_of_africa_country_masked | tabular | 3 | 0.140 | 0.083 | -0.056 | 0.085 | -0.066 |
| horn_of_africa_country_masked | tabular | 0 | 0.111 | 0.075 | -0.036 | 0.110 | -0.146 |
| mediterranean_spain_basin_masked | spatial | 0 | 0.183 | 0.151 | -0.032 | -0.105 | -0.054 |
| mediterranean_spain_basin_masked | spatial | 1 | 0.123 | 0.154 | 0.031 | -0.079 | -0.074 |
| mediterranean_spain_basin_masked | spatial | 2 | 0.125 | 0.148 | 0.024 | -0.115 | -0.110 |
| mediterranean_spain_basin_masked | spatial | 3 | 0.127 | 0.159 | 0.032 | -0.180 | -0.122 |
| mediterranean_spain_basin_masked | tabular | 0 | 0.183 | 0.143 | -0.041 | -0.047 | -0.065 |
| mediterranean_spain_basin_masked | tabular | 1 | 0.123 | 0.148 | 0.025 | -0.064 | -0.087 |
| mediterranean_spain_basin_masked | tabular | 2 | 0.125 | 0.142 | 0.018 | -0.113 | -0.138 |
| mediterranean_spain_basin_masked | tabular | 3 | 0.127 | 0.151 | 0.025 | -0.176 | -0.143 |
| murray_darling_basin_masked | spatial | 0 | 0.116 | 0.229 | 0.113 | -0.230 | -0.355 |
| murray_darling_basin_masked | spatial | 3 | 0.102 | 0.230 | 0.128 | -0.197 | -0.446 |
| murray_darling_basin_masked | spatial | 1 | 0.124 | 0.231 | 0.106 | -0.307 | -0.446 |
| murray_darling_basin_masked | spatial | 2 | 0.094 | 0.230 | 0.137 | -0.246 | -0.603 |
| murray_darling_basin_masked | tabular | 0 | 0.116 | 0.228 | 0.112 | -0.246 | -0.334 |
| murray_darling_basin_masked | tabular | 1 | 0.124 | 0.230 | 0.105 | -0.245 | -0.398 |
| murray_darling_basin_masked | tabular | 3 | 0.102 | 0.230 | 0.128 | -0.233 | -0.444 |
| murray_darling_basin_masked | tabular | 2 | 0.094 | 0.229 | 0.136 | -0.235 | -0.568 |
| southern_great_plains_basin_masked | spatial | 0 | 0.222 | 0.132 | -0.090 | 0.147 | 0.027 |
| southern_great_plains_basin_masked | spatial | 1 | 0.237 | 0.133 | -0.104 | 0.228 | 0.006 |
| southern_great_plains_basin_masked | spatial | 2 | 0.235 | 0.127 | -0.108 | 0.236 | -0.013 |
| southern_great_plains_basin_masked | spatial | 3 | 0.187 | 0.128 | -0.059 | 0.159 | -0.030 |
| southern_great_plains_basin_masked | tabular | 0 | 0.222 | 0.128 | -0.094 | 0.042 | -0.005 |
| southern_great_plains_basin_masked | tabular | 1 | 0.237 | 0.128 | -0.109 | 0.153 | -0.032 |
| southern_great_plains_basin_masked | tabular | 3 | 0.187 | 0.124 | -0.062 | 0.167 | -0.037 |
| southern_great_plains_basin_masked | tabular | 2 | 0.235 | 0.123 | -0.112 | 0.114 | -0.054 |
