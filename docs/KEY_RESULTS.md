# Key Results — CHIRPS Drought Classifier (California Central Valley)

> **Portfolio-ready summary of the six most significant outputs.**  
> Full details and figures: [`reports/PORTFOLIO.md`](../reports/PORTFOLIO.md)  
> Research assessment: [`ANALYSIS.md`](ANALYSIS.md)

---

## Best Model

| Property | Value |
|----------|-------|
| **Model** | XGBoost-Spatial (XGBoost + 3×3 neighbourhood features) |
| **Test period** | 2021–2026 · 63 independent monthly test steps |
| **Brier Score (dry)** | 0.0666  *(climatology ref = 0.0646)* |
| **Brier Skill Score** | −0.030  *(95% CI spans zero)* |
| **ROC-AUC (dry)** | 0.68 |
| **Region** | California Central Valley · 35.4°–40.6°N / 122.5°–119.0°W |

**Key finding:** No model outperforms climatology in Brier Skill Score. The
ROC-AUC of ~0.68 confirms learnable ranking signal exists, but models cannot
translate it into calibrated probability improvements — a **predictability
barrier** driven by the chaotic nature of monthly Mediterranean precipitation.

---

## Significant Outputs

| # | Output | Location |
|---|--------|----------|
| 1 | Best-model summary metrics | [`reports/tables/best_model_metrics.csv`](../reports/tables/best_model_metrics.csv) |
| 2 | Confusion matrix | [`reports/figures/confusion_matrix.png`](../reports/figures/confusion_matrix.png) |
| 3 | Model comparison table | [`reports/tables/model_comparison.csv`](../reports/tables/model_comparison.csv) |
| 4 | Drought risk map | [`reports/figures/drought_risk_map.png`](../reports/figures/drought_risk_map.png) |
| 5 | Feature importance / SHAP | [`reports/figures/feature_importance_shap.png`](../reports/figures/feature_importance_shap.png) |
| 6 | Time-series trend chart | [`reports/figures/time_series_trend.png`](../reports/figures/time_series_trend.png) |

---

## Most Important Findings

1. **XGBoost-Spatial is the best model** (BSS = −0.03, BS = 0.0666 vs. climatology
   BS = 0.0646), but the difference is not statistically significant at 63 test months.
2. **ROC-AUC ≈ 0.68** across gradient-boosted models confirms a learnable ranking signal
   — models detect *relative* drought likelihood better than chance.
3. **Ranking ≠ calibrated probability.** Murphy BS decomposition shows near-zero
   resolution: models cannot reliably distinguish drought months from non-drought months
   in advance. Post-hoc calibration cannot overcome a discrimination barrier.
4. **SHAP confirms physical consistency**: `spi1_lag1` + `spi3_lag1` dominate,
   reflecting soil-moisture persistence. Nonlinear threshold behaviour near SPI ≈ −1
   matches known SPI classification dynamics.
5. **Negative result has scientific value**: demonstrating the predictability ceiling
   with rigorous bootstrap evidence redirects effort toward higher-value additions
   (ENSO indices, soil moisture, longer aggregation windows).

---

## Regenerate Portfolio Figures

```bash
# No data files required — uses documented metric values
python scripts/generate_portfolio_figures.py
```
