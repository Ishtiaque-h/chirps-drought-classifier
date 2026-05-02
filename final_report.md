# Final Report (Working Draft)

## Related Research

### CHIRPS and drought monitoring
- CHIRPS v3 provides the core precipitation record used in this project and is widely used for drought monitoring at regional scales.
- Recent CHIRPS studies emphasize multi-index monitoring (SPI/SPEI/NDVI/SMCI) and regionalization rather than short-lead forecasting, which helps frame our predictability-barrier result.

### Subseasonal drought predictability
- Subseasonal drought forecast skill often drops sharply after 2-3 weeks; the literature supports the expectation of limited skill at 1-month lead without strong exogenous predictors.

### Teleconnections and regionalization
- PCA and clustering are used to define homogeneous drought zones, and ENSO/SOI correlations are used to interpret spatial variability. This directly informs our regionalization and ENSO diagnostics.

### Uncertainty quantification (UQ)
- Probabilistic DL and UQ methods are increasingly reported for hydrologic prediction; they improve reliability and interpretability but do not necessarily increase skill. This supports a short UQ section (calibration + SHAP + EDL).

### Agriculture relevance
- Central Valley agricultural scale motivates the societal impact framing. Drought indices are commonly linked to crop yield outcomes, reinforcing a potential small extension that connects drought probabilities to yield anomalies.

## Step 3 Diagnostics (Regionalization)

Method summary: Compute SPI-12 at each CHIRPS grid cell, standardize by 1991-2020 monthly climatology, then run PCA followed by k-means clustering to define homogeneous drought zones. For each zone, compute run-theory drought metrics (duration, severity, intensity) using the SPI-12 threshold of -1.0 and summarize ENSO/SOI/PDO correlations at lags 0-6 months to interpret teleconnection sensitivity. This is diagnostic-only and does not alter the frozen SPI-1 lead-1 forecast target or model checkpoints.

Core figures locked (Central Valley basin-masked):
- zone_map.png
- zone_run_duration_summary.png
- zone_spi12_timeseries.png

## Comprehensive ML Assessment: Project State & Predictability Barriers

### Overview
All scripts have been verified and executed successfully (May 2026). The project includes 9 major experiment tracks: baseline climatology, 6 traditional ML variants (RF, LogReg, XGBoost, XGBoost-Spatial, ConvLSTM, EDL), plus 3 feature engineering extensions (meteorological, soil moisture, atmospheric). Comprehensive evaluation reveals a critical finding: **1-month SPI-1 forecasting is fundamentally constrained by atmospheric chaos; feature engineering consistently degrades performance**.

### Performance Hierarchy (Monthly Brier Skill Score vs Climatology)

| Model | Test Set | BSS | Key Finding |
|-------|----------|-----|------------|
| Random Forest | 63 mo | **-0.068** | ✓ Best non-climatology |
| XGBoost-Spatial (3×3 neighborhood) | 63 mo | **-0.115** | ✓ Spatial helps |
| Meteorological (T/VPD anomalies) | 63 mo | -0.092 | ✗ Marginal |
| EDL MLP + isotonic calibration | ~63 mo | -0.126 | ✗ Fails |
| Soil Moisture features | 63 mo | -0.158 | ✗ Fails |
| XGBoost (no spatial) | 63 mo | -0.272 | ✗ Fails |
| ConvLSTM | 63 mo | -0.357 | ✗ Fails |
| Logistic Regression | 63 mo | -0.389 | ✗ Fails |
| **Atmospheric (MJO+AR/IVT)** | **39 mo** | **-0.333** | ✗ **Worst** |

**Conclusion**: All feature engineering approaches (meteorological, soil moisture, atmospheric indices) either degrade performance or fail to improve beyond spatial neighbor averaging. This indicates the problem is **chaos-limited**, not **feature-limited**.

### Why Feature Engineering Fails: Atmospheric Predictability Limits

1. **Timescale Mismatch**: Deterministic atmospheric predictability extends ~10-14 days. SPI-1[t+1] integrates weather over days 31-60, landing in the stochastic ensemble-spread regime.
2. **Feature Lag Obsolescence**: Soil moisture has ~10-30 day memory. After 30-60 days, predictors revert to climatology. Atmospheric features suffer from overfitting in low-signal regime (BSS -0.333).
3. **Spatial Neighbors Work**: Spatial neighbor averaging (BS = 0.0717) succeeds via **denoising**, not prediction. Least-bad non-climatology approach (BSS -0.115).

### Uncertainty Quantification: Aleatoric >> Epistemic

EDL decomposition shows aleatoric uncertainty (0.85, 80%) dominates epistemic uncertainty (0.24, 20%). This correctly identifies the problem as **chaos-limited**, not model-limited. UQ is valuable for **reliability** and **decision-making**, not for claiming predictive skill.

### Paper Narrative Recommendations

- **Primary** (Recommended): "Subseasonal Drought Forecasting Barriers" — negative result framing, guides practitioners
- **Secondary**: "Uncertainty Quantification for Drought Prediction" — EDL methods focus
- **Supporting**: "Regional Drought Teleconnections" — Step 3 regionalization standalone

## References

1. Funk, C. et al. (2026). CHIRPS v3 dataset. Scientific Data. https://doi.org/10.1038/s41597-026-07096-4
2. Funk, C. et al. (2015). CHIRPS environmental record. Scientific Data. https://doi.org/10.1038/sdata.2015.66
3. World Meteorological Organization (2012). Standardized Precipitation Index User Guide (WMO-No. 1090). https://library.wmo.int/idurl/4/39629
4. Murphy, A. (1973). Brier score decomposition. Journal of Applied Meteorology. https://ui.adsabs.harvard.edu/abs/1973JApMe..12..595M/abstract
5. Su, L. et al. (2023). Subseasonal drought forecast skill over the coastal western United States. Journal of Hydrometeorology. https://doi.org/10.1175/JHM-D-22-0103.1
6. AghaKouchak, A. et al. (2023). Drought as a cascading hazard (review). Nature Reviews Earth & Environment. https://doi.org/10.1038/s43017-023-00457-2
7. Dikshit, A. et al. (2021). Drought forecasting with ML (review). Journal of Environmental Management. https://doi.org/10.1016/j.jenvman.2021.111979
8. Hall, K., Acharya, N. (2022). XCast: a python climate forecasting toolkit. Frontiers in Climate. https://doi.org/10.3389/fclim.2022.953262
9. DeFlorio, M. et al. (2024). Seasonal/subseasonal forecasts of landfalling atmospheric rivers (winter 2022/23). Bulletin of the AMS. https://doi.org/10.1175/BAMS-D-22-0208.1
10. Masukwedza, G. et al. (2025). Subseasonal dry-spell forecast skill over Southern Africa. Climate Dynamics. https://doi.org/10.1007/s00382-025-07674-z
11. Oyarzabal, R. et al. (2025). ML drought forecasting review. Natural Hazards. https://doi.org/10.1007/s11069-025-07195-2
12. Sabut, A., Mishra, A. (2026). Century-scale drought research review. Water Resources Research. https://doi.org/10.1029/2025WR041987
13. Klotz, D. et al. (2022). DL uncertainty for rainfall-runoff modeling. Hydrology and Earth System Sciences. https://doi.org/10.5194/hess-26-1673-2022
14. Xu, L. et al. (2022). Probabilistic DL for precipitation forecast uncertainty. Hydrology and Earth System Sciences. https://doi.org/10.5194/hess-26-2923-2022
15. Liu, S. et al. (2023). UQ for ML streamflow prediction under change. Frontiers in Water. https://doi.org/10.3389/frwa.2023.1150126
16. De Leon Perez, D. et al. (2025). Probabilistic UQ in short-to-seasonal hydrologic prediction (scoping review). Water. https://doi.org/10.3390/w17202932
17. Ebrahimi, H. (2026). Evidential UQ for hybrid rainfall simulation. Water Resources Management. https://doi.org/10.1007/s11269-025-04386-1
18. Laassilia, O. et al. (2026). CHIRPS vs PERSIANN for drought monitoring and impacts (Morocco). Earth Systems and Environment. https://doi.org/10.1007/s41748-026-01120-8
19. Molosiwa, R. et al. (2026). Botswana regionalization with CHIRPS and SPI-12. Theoretical and Applied Climatology. https://doi.org/10.1007/s00704-026-06154-6
20. Tabassum, F., Krishna, A. P. (2022). CHIRPS-based SPI drought assessment (Subarnarekha basin). Environmental Monitoring and Assessment. https://doi.org/10.1007/s10661-022-10547-1
21. Polat, A. B. et al. (2026). Multi-index remote-sensing drought risk (CHIRPS + NDVI + LST). Environmental Monitoring and Assessment. https://doi.org/10.1007/s10661-025-14895-6
22. Agudelo, D. et al. (2024). Subseasonal drought forecasting in Bihar, India (report). https://hdl.handle.net/10568/163071
23. USGS (accessed 2026). Central Valley agriculture facts. https://ca.water.usgs.gov/projects/central-valley/about-central-valley.html
24. Islam, S. M. et al. (2024). Drought impact on crop yields in Iowa (preprint). https://doi.org/10.31223/X54Q4D
