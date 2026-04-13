# Research Analysis — Central Valley Drought Classifier

> **Comprehensive evaluation and strategic roadmap for the chirps-drought-classifier project**
> Prepared as a combined ML scientist and peer-reviewer assessment.

---

## 1. Current State of the Project

### What has been accomplished

The project implements a complete, reproducible pipeline for **1-month-ahead drought class prediction** in California's Central Valley (1991–2025) using CHIRPS v3.0 satellite precipitation:

| Component | Status | Assessment |
|-----------|--------|------------|
| Data ingestion & SPI computation | ✅ Complete | WMO-standard gamma-fit SPI-1/3/6; scientifically correct |
| Feature engineering | ✅ Complete | SPI + precipitation lags + cyclic month encoding (10 features) |
| Model suite | ✅ Complete | LogReg, RF, XGBoost, XGBoost-Spatial, ConvLSTM |
| Evaluation protocol | ✅ Rigorous | Monthly-level BSS/HSS, bootstrap CI, 3 naive baselines, calibration study |
| Explainability | ✅ Complete | SHAP TreeExplainer (dry/normal/wet classes) |
| Cross-dataset validation | ✅ Complete | ERA5-Land SPI-1 comparison |
| Qualitative validation | ✅ Complete | USDM D1+ consistency check (correctly framed as non-metric) |
| Spatial analysis | ✅ Complete | Per-pixel accuracy maps, Sacramento/San Joaquin sub-regions |
| Case study | ✅ Complete | 2021–22 drought / 2023 atmospheric rivers |

### Key result

> **No model substantially outperforms climatology in monthly BSS for the dry class.**
>
> Closest to climatology: **XGBoost-Spatial** (BSS = −0.031, BS = 0.067 vs. climatology BS = 0.065).
> ROC-AUC shows ranking signal (~0.68), confirming that models detect *relative* drought likelihood but cannot translate this into calibrated probability improvement at the monthly level.

This is a scientifically valid and publishable finding — but only if the analysis is sufficiently thorough to explain **why** the predictability barrier exists and **whether it generalizes** beyond this specific region and feature set.

---

## 2. Dataset Design Assessment

### 2.1 Spatial focus: Central Valley

**Strengths:**
- Agriculturally critical region with clear policy relevance (almond, grape, citrus crops)
- Coherent hydroclimate: Mediterranean regime, winter-dominated precipitation
- CHIRPS performs well here (good gauge density, moderate topographic complexity)

**Limitations:**
- **Too homogeneous for generalizability claims.** Central Valley is a single, relatively uniform precipitation regime. The finding that ML cannot beat climatology may be a property of this specific regime — or it may be universal. A single-region study cannot distinguish these.
- **Limited climate variability.** Central Valley monthly precipitation is strongly seasonal (dry summers, wet winters). The normal class dominates (~60–70% of months), which means climatology is already a strong baseline. The *opportunity* for ML to add skill is inherently limited in low-variability regimes.
- **~6,800 pixels at 0.05° resolution** provides good spatial coverage but creates pseudo-replication risk (correctly addressed by monthly aggregation).

**Verdict:** Central Valley is a defensible starting point, but **the single-region design is the primary limitation for publication impact.**

### 2.2 Temporal design

**Strengths:**
- Train (1991–2016) / val (2017–2020) / test (2021–2025) split is temporal, non-shuffled — gold standard
- 60 independent test months is adequate for bootstrap-based inference
- No target leakage: SPI-1[t+1] depends only on pr[t+1], which is unknown from features at t

**Limitations:**
- Test period (2021–2025) coincidentally includes an extreme drought (2021–22) followed by an extreme wet reversal (2023). This is good for case-study value but means the test set is not climatologically representative — it over-represents extreme events relative to the training distribution.
- The 30-year gamma-fit baseline (1991–2020) may not capture non-stationarity driven by climate change, potentially biasing SPI values in the 2021–2025 test period.

### 2.3 What expanding to additional regions would offer

| Benefit | Detail |
|---------|--------|
| **Generalizability test** | Does "no skill" hold in a different hydroclimate (e.g., semi-arid Great Plains, monsoonal India, Mediterranean Spain)? If yes, the finding is much more significant. If no, what feature or mechanism explains the difference? |
| **Transfer learning** | Train on one region, test on another — demonstrates whether spatial patterns are generalizable or regionally overfit |
| **Statistical power** | Multiple regions multiply the number of independent test months, tightening confidence intervals |
| **Publication impact** | Multi-region studies are far more publishable (reviewer concern: "would this hold elsewhere?" is preemptively answered) |
| **Mechanistic insight** | Comparing skill across regimes (Mediterranean vs. continental vs. monsoonal) reveals which climate properties enable or prevent predictability |

**Recommended expansion regions (ranked by scientific complementarity):**

1. **Murray–Darling Basin, Australia** — analogous semi-arid agricultural region; CHIRPS coverage excellent; different ENSO teleconnection sign
2. **Great Plains, USA (Kansas–Oklahoma)** — continental regime with convective summer precipitation; tests whether skill gap is California-specific
3. **Mediterranean Spain (Ebro/Guadalquivir basins)** — closest hydroclimate analog; tests cross-regime transferability within Mediterranean climates
4. **Horn of Africa (Kenya–Ethiopia)** — CHIRPS was originally designed for this region; bimodal precipitation with strong ENSO dependence

---

## 3. Feature Engineering Assessment

### 3.1 Current feature set

| Feature group | Variables | Adequacy |
|---------------|-----------|----------|
| SPI indices | spi1_lag1–3, spi3_lag1, spi6_lag1 | ✅ Core — captures multi-scale drought memory |
| Raw precipitation | pr_lag1–3 | ✅ Useful — provides absolute magnitude information |
| Seasonality | month_sin, month_cos | ✅ Standard cyclic encoding |
| **Total** | **10 features** | Narrow but defensible for a pure-precipitation study |

### 3.2 What is missing and likely limiting model skill

**High-priority additions (likely to improve skill):**

| Feature | Source | Rationale | Feasibility |
|---------|--------|-----------|-------------|
| **ENSO index (Niño 3.4)** | NOAA | Central Valley precipitation has a well-documented ENSO teleconnection; La Niña winters are consistently drier. This is the single most promising predictor for improving monthly-scale skill. | High — freely available monthly time series |
| **Temperature / VPD anomalies** | ERA5-Land or CPC | Temperature modulates drought severity through evapotranspiration; VPD amplifies agricultural drought even when precipitation is near-normal. | Medium — requires downloading ERA5-Land temperature grid |
| **Pacific Decadal Oscillation (PDO)** | NOAA | Low-frequency modulation of California precipitation on decadal timescales. | High — freely available monthly time series |
| **Atmospheric River count/intensity** | e.g., Gershunov et al. catalog | Central Valley precipitation extremes are driven by atmospheric rivers; their frequency and intensity are potentially predictable at sub-seasonal lead times. | Medium — requires catalog preprocessing |

**Medium-priority additions:**

| Feature | Source | Rationale | Feasibility |
|---------|--------|-----------|-------------|
| **NDVI/EVI anomalies** | MODIS (MOD13A3) | Vegetation response lags precipitation; can serve as an integrated drought indicator | Medium |
| **Soil moisture** | SMAP or ERA5-Land | Direct measure of hydrological drought; provides memory that SPI does not capture | Medium |
| **Topographic features** | SRTM DEM | Elevation, slope, and aspect modulate precipitation and drought susceptibility at sub-regional scales | High (static) |

**Key insight:** The current feature set is **purely endogenous** (all features are derived from the same CHIRPS precipitation dataset used to define the target). Adding **exogenous climate drivers** (ENSO, PDO) would test a fundamentally different hypothesis: can large-scale climate state improve local drought prediction beyond what local precipitation history provides?

### 3.3 Feature engineering recommendations

1. **Add ENSO Niño 3.4 index as a first experiment.** This is the lowest-effort, highest-potential-impact addition. If ENSO does not improve BSS in Central Valley, it strongly supports the "intrinsic predictability barrier" interpretation.
2. **Add temperature/VPD** to test whether drought predictability improves when evaporative demand is included (transition from meteorological to agricultural drought framing).
3. **Conduct a formal ablation study** on the current 10 features to quantify which provide marginal information gain.

---

## 4. Modeling and Evaluation Assessment

### 4.1 Strengths

- **Methodological rigor is publication-grade.** Monthly-level evaluation with bootstrap CI, BS decomposition, calibration study, paired significance tests — this is above-average for the hydrology ML literature.
- **Baseline comparison is exemplary.** Three naive baselines (climatology, persistence, SPI-1 threshold) with an explicit "spatial complexity ladder" is exactly what reviewers want to see.
- **SHAP explainability** confirms physically consistent feature effects (negative anomaly → higher drought probability), ruling out spurious correlations.
- **Cross-dataset validation** (ERA5-Land) is a strong addition that most precipitation-ML papers lack.

### 4.2 Weaknesses

- **Calibration is the bottleneck, not model architecture.** The gap between XGBoost-Spatial (BS = 0.067) and climatology (BS = 0.065) is tiny. Isotonic/Platt calibration has been tried but cannot create skill where discrimination is marginal. This is correct behavior — calibration cannot improve a model that does not resolve events.
- **Model zoo without feature diversity.** Trying LR, RF, XGBoost, ConvLSTM on the same 10 features tests model architecture, not information content. If features lack predictive signal for next month's SPI-1, no architecture will help.
- **ConvLSTM underperforms XGBoost.** This is expected when the spatial neighborhood is already captured by the 3×3 mean features in XGBoost-Spatial. The ConvLSTM's overhead (complexity, training cost) is not justified.
- **HSS = 0 for XGBoost variants** means the model always predicts "normal" at the monthly dominant-class level — the model has learned that the safe bet is always the majority class. This is a common failure mode when class imbalance meets weak signal.

### 4.3 The predictability question

The fundamental question is: **Is monthly SPI-1 in Central Valley inherently unpredictable from past precipitation alone?**

Evidence suggests **yes**, for this specific target and region:
- SPI-1[t+1] = f(pr[t+1]) only. Monthly precipitation in California is primarily driven by synoptic-scale events (atmospheric rivers, frontal systems) that are **chaotic at 1-month lead**.
- The autocorrelation of monthly precipitation in Central Valley is weak (r ≈ 0.1–0.3), consistent with the finding that persistence fails badly (BSS = −0.57).
- Climatology is already well-calibrated because it is based on the correct base rate distribution.

This is a **publishable scientific finding** — but only if framed correctly and supported by evidence from multiple angles (features, regions, time horizons).

---

## 5. Big-Picture Research Questions

### For high-impact publication, the project should address at least 2–3 of these:

| # | Research Question | Impact | Feasibility |
|---|-------------------|--------|-------------|
| 1 | **Does the predictability barrier generalize across hydroclimatic regimes?** Train and evaluate the exact same pipeline in 2–3 additional regions. If the barrier holds, this is a strong negative result with broad implications. If it breaks in some regimes, characterize what makes them different. | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |
| 2 | **Do exogenous climate drivers (ENSO, PDO) improve skill?** Add Niño 3.4 and PDO as features; measure BSS change. This directly tests whether the barrier is due to missing information vs. intrinsic chaos. | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| 3 | **How does skill vary with lead time and temporal aggregation?** Evaluate at seasonal (3-month) and quarterly horizons. SPI-3 as target may be more predictable at seasonal lead. | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| 4 | **Is there conditional skill?** Do models outperform climatology specifically during ENSO warm/cold phases, or during winter (wet season) vs. summer? Stratified BSS analysis. | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| 5 | **Transfer learning: can a model trained on one region predict drought in an analogous region?** Train on Central Valley, test on Ebro Basin (or vice versa). | ⭐⭐⭐⭐ | ⭐⭐⭐ |
| 6 | **What is the minimum feature set for positive skill?** Formal ablation + feature importance analysis to identify which features (if any) provide marginal predictive information. | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| 7 | **Anomaly detection vs. classification:** Frame the problem as detecting anomalous months (outlier detection) rather than 3-class classification. This may be better suited to the low base rate of drought. | ⭐⭐⭐ | ⭐⭐⭐⭐ |

---

## 6. Modern Research Context

### 6.1 Where this work sits in the literature

The project addresses a well-studied problem (ML-based drought prediction) but with unusually rigorous methodology. Most published studies in this space suffer from:

- **Inflated accuracy from spatial pseudo-replication** — this project correctly addresses this
- **No baseline comparison** — this project includes three baselines
- **Target leakage from SPI accumulation windows** — v3 correctly uses SPI-1[t+1]
- **No uncertainty quantification** — this project includes bootstrap CI and calibration analysis

The finding that ML does not outperform climatology at 1-month lead is consistent with:
- Luo et al. (2024, *J. Hydrol.*) — found marginal ML skill for short-lead drought forecasting in China
- AghaKouchak et al. (2023, *Rev. Geophys.*) — argued that precipitation-only indices have limited predictability at sub-seasonal scales
- Dikshit et al. (2021, *Sci. Total Environ.*) — found that additional predictors (ENSO, soil moisture) are needed to exceed climatological baselines

### 6.2 Gaps this project could fill

1. **Rigorous negative result in drought ML** — the literature has a publication bias toward positive results; a methodologically sound "no skill" finding with thorough explanation is valuable
2. **Multi-region predictability comparison** — very few studies compare ML drought skill across diverse hydroclimates using identical methodology
3. **Explicit predictability decomposition** — BS decomposition (reliability/resolution/uncertainty) is rarely reported in drought ML papers; this is a methodological contribution

---

## 7. Strategic Recommendations

### Ranked by impact × feasibility:

| Rank | Action | Impact | Feasibility | Rationale |
|------|--------|--------|-------------|-----------|
| **1** | **Add ENSO (Niño 3.4) and PDO as features** | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | Lowest effort, directly tests the key hypothesis. If it works, you have a positive skill result. If not, you strengthen the negative finding. |
| **2** | **Add 1–2 expansion regions** (Murray–Darling + Great Plains or Spain) | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | Transforms the paper from "regional case study" to "generalizable finding." CHIRPS is global — pipeline reuse is straightforward. |
| **3** | **Stratified skill analysis** (by ENSO phase, season, drought severity) | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | Can be done with existing data. May reveal conditional skill that is masked in overall BSS. |
| **4** | **Seasonal target (SPI-3 at 3-month lead)** | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | Tests whether longer accumulation windows make the problem more predictable. Important for operational utility. |
| **5** | **Ablation study on features** | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | Quick to implement; provides formal feature contribution analysis. |
| **6** | **Add temperature/VPD features** | ⭐⭐⭐ | ⭐⭐⭐ | Tests agricultural drought framing. Requires ERA5-Land download. |
| **7** | **Transfer learning experiment** | ⭐⭐⭐⭐ | ⭐⭐ | High novelty but requires multi-region setup (depends on #2). |

### What NOT to prioritize:

- **More model architectures** (transformers, attention networks) — not useful when the bottleneck is information content, not model capacity
- **Hyperparameter tuning** — XGBoost is already well-tuned; marginal gains won't close the BS gap
- **More complex calibration** — isotonic and Platt have been tried; the gap is too small for calibration to bridge

---

## 8. Recommended Research Narrative

### For a publication-quality paper, the narrative should be:

> **"We systematically evaluate whether machine learning can improve 1-month-ahead drought prediction beyond climatological baselines."**
>
> We build a rigorous, leakage-free pipeline using CHIRPS v3.0 satellite precipitation and WMO-standard SPI-1. All metrics are computed at the monthly level (60 independent test months) with bootstrap uncertainty, using three naive baselines for reference. We test shallow (logistic regression, random forest), gradient-boosted (XGBoost ± spatial features), and deep learning (ConvLSTM) models.
>
> **Key finding:** In California's Central Valley (2021–2025), no model substantially outperforms climatology in Brier Skill Score, despite showing discrimination signal (ROC-AUC ~0.68). Brier Score decomposition reveals that models achieve negligible resolution improvement over the climatological base rate, consistent with the theoretical expectation that single-month precipitation in this Mediterranean regime is largely chaotic at 1-month lead.
>
> **Implications:** (1) ML model accuracy reported without baseline comparison systematically overstates forecast utility. (2) The predictability barrier suggests that improving drought prediction at monthly scale requires either exogenous climate drivers (ENSO, teleconnections) or longer accumulation windows (seasonal SPI-3). (3) The methodology presented here provides a template for rigorous drought ML evaluation that correctly accounts for spatial autocorrelation and class frequency.

This narrative transforms a "negative result" into a **methodological and scientific contribution**.

---

## 9. Actionable Next Steps (Immediate)

1. ✅ **Rewrite README** to reflect the mature project state and ML insights
2. **Add Niño 3.4 feature** — download monthly ENSO index, merge into `build_dataset_forecast.py`, retrain, evaluate
3. **Stratified BSS analysis** — add seasonal and ENSO-phase stratification to `evaluate_forecast_skill.py`
4. **Expand to Murray–Darling Basin** — clone pipeline with new bounding box, run all stages
5. **Draft paper outline** using the narrative framework above
