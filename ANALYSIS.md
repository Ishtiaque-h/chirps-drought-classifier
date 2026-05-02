# Research Analysis — Central Valley Drought Classifier

> **Comprehensive evaluation and strategic roadmap for the chirps-drought-classifier project**
> Prepared as a combined ML scientist and peer-reviewer assessment.

---

## 1. Current State of the Project

### What has been accomplished

The project implements a complete, reproducible pipeline for **1-month-ahead drought class prediction** in California's Central Valley (1991–2026) using CHIRPS v3.0 satellite precipitation:

| Component | Status | Assessment |
|-----------|--------|------------|
| Data ingestion & SPI computation | ✅ Complete | WMO-standard gamma-fit SPI-1/3/6; scientifically correct |
| Feature engineering | ✅ Complete | SPI + precipitation lags + cyclic month encoding + corrected Niño3.4 anomaly lags |
| Model suite | ✅ Complete | LogReg, RF, XGBoost, XGBoost-Spatial, and corrected-target ConvLSTM are current on the corrected ENSO-only schema |
| Evaluation protocol | ✅ Rigorous | Monthly-level BSS/HSS, bootstrap CI, 3 naive baselines, calibration study |
| Calibration study | ✅ Valid (post-fix) | Isotonic regression selected; calibrated XGB-Spatial is effectively tied with climatology |
| Explainability | ✅ Current | Corrected SHAP artifacts now exist for both XGBoost and XGBoost-Spatial |
| Cross-dataset validation | ✅ Complete | ERA5-Land SPI-1 comparison |
| Qualitative validation | ✅ Complete | USDM D1+ consistency check (correctly framed as non-metric) |
| Spatial analysis | ✅ Complete | Per-pixel accuracy maps, Sacramento/San Joaquin sub-regions |
| Case study | ✅ Complete | 2021–22 drought / 2023 atmospheric rivers |
| Season-conditional skill | ✅ Quantified | Raw MAM skill is positive, but bootstrap CI crosses zero and calibration removes the signal |
| ENSO stratification | ✅ Fixed | Niño3.4 is now converted from absolute SST to anomalies; stratified rows populate correctly |
| Feature ablation | ✅ Complete | `scripts/run_feature_ablation.py` uses early-stopped XGBoost predictions and current features |
| Seasonal SPI-3 experiment | ✅ Initial test complete | Leakage-free SPI-3 lead-3 tabular XGBoost remains below climatology after isotonic calibration |
| Temperature/VPD experiment | ✅ Initial test complete | Regional ERA5-Land t2m/VPD anomalies improve raw XGB but still do not beat climatology |
| Soil-moisture experiment | ✅ Initial test complete | Regional ERA5-Land soil-water/root-zone anomaly lags overfit and remain below climatology |
| Multi-region path | ✅ Initial path complete | Region registry + runner now supports Central Valley, Southern Great Plains, and Mediterranean Spain tabular/spatial tests |
| Regional mechanism comparison | ✅ Initial analysis complete | Reproducible diagnostics separate ranking, calibration, test-period shift, persistence, and feature-group gain |
| Region geometry audit | ✅ First-pass country masks complete | Natural Earth country masks quantify rectangular-box contamination and add a masked Spain sensitivity run |

### Key results (corrected ENSO + spatial checkpoint — 2026-05-01)

> **Climate-index preprocessing is now sane.** PDO `-9.9` sentinels are masked,
> recent missing PDO values are not forward-filled, and Niño3.4 absolute SST is
> converted to monthly anomalies using the 1991–2020 climatology. The active
> corrected checkpoint uses Niño3.4 anomaly lags only; PDO is excluded because
> recent PDO values are missing after August 2025.
>
> **The best model is now a practical tie with climatology.** Raw XGB-Spatial
> remains below climatology, but validation-selected isotonic calibration gives a
> tiny positive point estimate (BSS = +0.005). The confidence interval crosses
> zero, so this is not a statistically reliable positive-skill result.

| Model | Best calibration | Test BS | Test BSS vs climatology |
|-------|-----------------|---------|------------------------|
| XGB | isotonic | 0.06839 | −0.064 |
| XGB-Spatial | isotonic | 0.06394 | +0.005 |
| Climatology (reference) | — | 0.0643 | 0.000 |

> **Conditional skill is suggestive but not yet defensible.**
> Raw MAM BSS is positive, but the monthly bootstrap CI crosses zero. Global
> calibration removes most of the MAM gain, and season-specific calibration
> overfits badly with only 12 validation months per season.
>
> **The first seasonal target experiment is also negative.** A leakage-free
> SPI-3 lead-3 setup (features at t, target SPI-3 ending t+3) gives calibrated
> XGBoost BSS = -0.127 with a 95% CI below zero. Longer accumulation alone is
> not enough for positive skill in the current Central Valley tabular setup.
>
> **Temperature/VPD adds signal but not positive skill.** ERA5-Land t2m/VPD
> anomaly lags dominate gain in a separate non-spatial XGBoost experiment and
> move raw BSS close to climatology (BSS = -0.030, CI crossing zero). A gridded
> temperature/VPD + XGBoost-Spatial experiment also remains below climatology
> after validation-selected isotonic calibration (BSS = -0.047, CI crossing
> zero). This is signal, not robust probability skill.
>
> **Regional soil moisture is not the missing ingredient.** ERA5-Land
> soil-water/root-zone anomaly lags dominate split gain in a separate XGBoost
> experiment (~68% gain share), but selected calibrated test BSS is -0.158 with
> the CI below zero. Monthly dry-probability correlation is near zero or
> negative on the test period, so this is overfit land-surface memory rather
> than usable 1-month-ahead SPI-1 skill.
>
> **The first multi-region extension complicates the barrier hypothesis in a
> useful way.**
> `scripts/run_multiregion_xgb_experiment.py` now clips CHIRPS, computes
> region-specific SPI, builds the same SPI-1[t+1] forecast table, and evaluates
> monthly BSS for configured regions. Southern Great Plains tabular/spatial
> XGBoost remains below climatology (`selected BSS ≈ -0.082`, CI just below
> zero). Mediterranean Spain, however, has a positive calibrated point estimate
> (`tabular BSS = +0.044`, `spatial BSS = +0.022`), but both confidence
> intervals cross zero. The current result is therefore not "ML never works";
> it is "positive 1-month SPI-1 skill is region-dependent and not yet robust."
>
> **The mechanism diagnostics identify three different regimes.**
> `scripts/analyze_multiregion_mechanisms.py` shows that Central Valley has the
> strongest ranking signal but under-amplified calibrated probabilities
> (`selected amplitude ratio ≈ 0.21`). Southern Great Plains has weak ranking,
> strong underprediction of test dry frequency, and a train-to-test dry shift
> from 0.160 to 0.208. Mediterranean Spain has the best calibrated point
> estimate, near-zero selected bias, and higher selected-probability correlation
> than climatology, but uncertainty remains too wide for a claim of robust
> positive skill.
>
> **The first geometry sensitivity weakens but does not erase the Spain hint.**
> `scripts/build_region_masks.py` uses Natural Earth country polygons to audit
> rectangular boxes. The mask removes 0.0% of valid Southern Great Plains cells,
> 2.85% of Central Valley cells, and 5.64% of Mediterranean Spain cells. The
> masked Spain spatial run keeps a positive calibrated point estimate
> (`BSS = +0.0235`) but its CI still crosses zero (`[-0.199, +0.305]`). Spain
> remains hypothesis-generating, not a defensible positive-skill result.
>
> **XGB-Spatial is the best current ML option** because it has the best ranking
> skill (ROC-AUC = 0.743) and the best calibrated Brier Score. Current evidence
> still points to information-content limits, not model-capacity limits.

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
- Train (1991–2016) / val (2017–2020) / test (2021–2026) split is temporal, non-shuffled — gold standard
- 63 independent test months is adequate for bootstrap-based inference
- No target leakage: SPI-1[t+1] depends only on pr[t+1], which is unknown from features at t

**Limitations:**
- Test period (2021–2026) coincidentally includes an extreme drought (2021–22) followed by an extreme wet reversal (2023). This is good for case-study value but means the test set is not climatologically representative — it over-represents extreme events relative to the training distribution.
- The 30-year gamma-fit baseline (1991–2020) may not capture non-stationarity driven by climate change, potentially biasing SPI values in the 2021–2026 test period.

### 2.3 What expanding to additional regions would offer

| Benefit | Detail |
|---------|--------|
| **Generalizability test** | Does "no skill" hold in a different hydroclimate (e.g., semi-arid Great Plains, monsoonal India, Mediterranean Spain)? If yes, the finding is much more significant. If no, what feature or mechanism explains the difference? |
| **Transfer learning** | Train on one region, test on another — demonstrates whether spatial patterns are generalizable or regionally overfit |
| **Statistical power** | Multiple regions multiply the number of independent test months, tightening confidence intervals |
| **Publication impact** | Multi-region studies are far more publishable (reviewer concern: "would this hold elsewhere?" is preemptively answered) |
| **Mechanistic insight** | Comparing skill across regimes (Mediterranean vs. continental vs. monsoonal) reveals which climate properties enable or prevent predictability |

**Recommended expansion regions (ranked by scientific complementarity):**

1. **Great Plains, USA (Kansas–Oklahoma)** — first full tabular and spatial tests are complete and negative
2. **Mediterranean Spain (Ebro/Guadalquivir basins)** — full tabular/spatial tests and a Spain country-mask sensitivity are complete; positive calibrated point estimates persist, but CIs cross zero
3. **Murray–Darling Basin, Australia** — analogous semi-arid agricultural region; CHIRPS coverage excellent; different ENSO teleconnection sign
4. **Horn of Africa (Kenya–Ethiopia)** — CHIRPS was originally designed for this region; bimodal precipitation with strong ENSO dependence

---

## 3. Feature Engineering Assessment

### 3.1 Current feature set

| Feature group | Variables | Adequacy |
|---------------|-----------|----------|
| SPI indices | spi1_lag1–3, spi3_lag1, spi6_lag1 | ✅ Core — captures multi-scale drought memory |
| Raw precipitation | pr_lag1–3 | ✅ Useful — provides absolute magnitude information |
| Seasonality | month_sin, month_cos | ✅ Standard cyclic encoding |
| ENSO | nino34_lag1–2 | ✅ Active corrected exogenous anomaly features |
| **Total** | **12 active features** | Narrow but defensible for the corrected ENSO checkpoint |

### 3.2 What is missing and likely limiting model skill

**Candidate and tested additions:**

| Feature | Source | Rationale | Feasibility |
|---------|--------|-----------|-------------|
| **ENSO index (Niño 3.4)** | NOAA | Implemented as monthly anomalies. It improves calibrated XGB-Spatial to a near-tie with climatology but does not produce statistically reliable positive skill. | Complete |
| **Temperature / VPD anomalies** | ERA5-Land or CPC | Temperature modulates drought severity through evapotranspiration; VPD amplifies agricultural drought even when precipitation is near-normal. Regional and gridded initial tests add signal but do not beat climatology. | Complete initial tests |
| **Pacific Decadal Oscillation (PDO)** | NOAA | Low-frequency modulation of California precipitation on decadal timescales. | High — freely available monthly time series |
| **Atmospheric River count/intensity** | e.g., Gershunov et al. catalog | Central Valley precipitation extremes are driven by atmospheric rivers; their frequency and intensity are potentially predictable at sub-seasonal lead times. | Medium — requires catalog preprocessing |

**Medium-priority additions:**

| Feature | Source | Rationale | Feasibility |
|---------|--------|-----------|-------------|
| **NDVI/EVI anomalies** | MODIS (MOD13A3) | Vegetation response lags precipitation; can serve as an integrated drought indicator | Medium |
| **Soil moisture** | SMAP or ERA5-Land | Direct measure of hydrological drought; regional ERA5-Land anomaly lags overfit and do not improve SPI-1 lead-1 BSS. A gridded or SMAP variant is now secondary, not the main next step. | Initial regional test complete |
| **Topographic features** | SRTM DEM | Elevation, slope, and aspect modulate precipitation and drought susceptibility at sub-regional scales | High (static) |

**Key insight:** The current active feature set is no longer purely endogenous: it includes corrected Niño3.4 anomaly lags. That exogenous climate signal improves ranking and brings calibrated XGB-Spatial to a near-tie with climatology, but still does not produce statistically reliable positive BSS.

### 3.3 Feature engineering recommendations

1. **Treat the corrected ENSO experiment as the current checkpoint.** ENSO helps ranking and nearly closes the calibrated BSS gap, but the positive point estimate is not statistically reliable.
2. **Treat temperature/VPD and regional soil moisture as completed negative land-surface tests.** They add model fit, but not reliable probability skill over climatology.
3. **Treat ablation cautiously.** Current ablation shows precipitation lags and seasonality help, while removing ENSO/SPI lags improves this trained XGB model. Because these features are correlated, this should be read as a trained-model diagnostic, not a causal feature-importance statement.
4. **Prioritize new independent information or new regions.** For Central Valley, atmospheric-river/subseasonal predictors are more defensible than more tuning of the same lagged land-surface fields. For the paper, multi-region generalization is higher value.

---

## 4. Modeling and Evaluation Assessment

### 4.1 Strengths

- **Methodological rigor is publication-grade.** Monthly-level evaluation with bootstrap CI, BS decomposition, calibration study, paired significance tests — this is above-average for the hydrology ML literature.
- **Baseline comparison is exemplary.** Three naive baselines (climatology, persistence, SPI-1 threshold) with an explicit "spatial complexity ladder" is exactly what reviewers want to see.
- **SHAP explainability** confirms physically consistent feature effects (negative anomaly → higher drought probability), ruling out spurious correlations.
- **Cross-dataset validation** (ERA5-Land) is a strong addition that most precipitation-ML papers lack.

### 4.2 Weaknesses

- **Calibration is not enough by itself.** The gap between calibrated XGBoost-Spatial (BS = 0.06394) and climatology (BS = 0.0643) is tiny and statistically indistinguishable from zero. Calibration can align probabilities, but it cannot create robust resolution where monthly predictability is weak.
- **Model zoo without feature diversity.** Trying LR, RF, XGBoost, and spatial variants on the same narrow feature family tests model architecture more than information content. If features lack predictive signal for next month's SPI-1, no architecture will help.
- **ConvLSTM remains weaker than XGBoost-Spatial.** After fixing the target alignment and retraining, ConvLSTM improves over the stale artifact but still has negative BSS and lower ranking skill than XGBoost-Spatial.
- **Monthly categorical skill remains weak.** LogReg, XGBoost, XGBoost-Spatial, and ConvLSTM have only small positive HSS, while RF is slightly negative. This is a common failure mode when class imbalance meets weak signal.

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
| 2 | **Do additional exogenous drivers improve skill beyond corrected ENSO?** Temperature/VPD and regional soil moisture are now negative initial tests; remaining Central Valley candidates include atmospheric-river/circulation predictors, vegetation indices, or a carefully handled historical PDO subset. | ⭐⭐⭐⭐ | ⭐⭐ |
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

The finding that ML does not reliably outperform climatology at 1-month lead is consistent with:
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
| **1** | **Improve region geometry beyond country masks** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | Country masks catch rectangular-box contamination, but basin/hydroclimate masks are still needed before final publication claims, especially Spain/Murray-Darling. |
| **2** | **Turn mechanism diagnostics into paper figures** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | The regional contrast is now the main scientific story and should be presented before adding more features. |
| **3** | **Add one more full region** (Murray-Darling or Horn of Africa) | ⭐⭐⭐⭐ | ⭐⭐ | Tests whether the Mediterranean Spain hint is isolated or part of a broader regime pattern. |
| **4** | **Atmospheric-river or subseasonal circulation predictors** | ⭐⭐⭐⭐ | ⭐⭐ | Central Valley monthly extremes are event-driven; AR/circulation predictors are more physically targeted than more lagged land-surface tuning. |
| **5** | **Seasonal target variants with more information** | ⭐⭐⭐ | ⭐⭐⭐ | The first tabular SPI-3 lead-3 run is negative; revisit with spatial features, extra drivers, or additional regions if needed. |
| **6** | **Gridded/SMAP soil-moisture sensitivity** | ⭐⭐ | ⭐⭐ | Regional ERA5-Land soil moisture overfits; only pursue this if a spatial or independent-observation formulation is needed for completeness. |
| **7** | **Refresh corrected explainability as figures evolve** | ⭐⭐ | ⭐⭐⭐⭐ | Current SHAP artifacts are refreshed; rerun only after model/schema changes. |
| **8** | **Regional/seasonal stratified diagnostics with more months** | ⭐⭐⭐⭐ | ⭐⭐⭐ | Current MAM/ENSO hints have CIs crossing zero; more independent months or regions are needed. |
| **9** | **Transfer learning experiment** | ⭐⭐⭐⭐ | ⭐⭐ | High novelty but requires multi-region setup. |

### What NOT to prioritize:

- **More model architectures** (transformers, attention networks) — not useful when the bottleneck is information content, not model capacity
- **Hyperparameter tuning** — XGBoost is already well-tuned; marginal gains won't close the BS gap
- **More complex calibration** — isotonic and Platt have been tried; the gap is too small for calibration to bridge

---

## 8. Recommended Research Narrative

### For a publication-quality paper, the narrative should be:

> **"We systematically evaluate whether machine learning can improve 1-month-ahead drought prediction beyond climatological baselines."**
>
> We build a rigorous, leakage-free pipeline using CHIRPS v3.0 satellite precipitation, WMO-standard SPI-1, corrected Niño3.4 anomaly lags, and local spatial context. All metrics are computed at the monthly level (63 independent test months) with bootstrap uncertainty, using three naive baselines for reference.
>
> **Key finding:** In California's Central Valley (2021–2026), corrected ENSO + XGBoost-Spatial nearly ties climatology in calibrated Brier Skill Score (BSS = +0.005, CI crossing zero), despite showing useful discrimination signal (ROC-AUC = 0.743). Brier Score decomposition still shows only marginal resolution improvement over the climatological base rate, consistent with the theoretical expectation that single-month precipitation in this Mediterranean regime is largely chaotic at 1-month lead.
>
> **Implications:** (1) ML model accuracy reported without baseline comparison systematically overstates forecast utility. (2) ENSO and spatial context add information, but not enough for statistically reliable positive probability skill at this horizon. (3) The methodology presented here provides a template for rigorous drought ML evaluation that correctly accounts for spatial autocorrelation, class frequency, calibration, and uncertainty.

This narrative transforms a "negative result" into a **methodological and scientific contribution**.

---

## 9. Actionable Next Steps (Immediate)

### Already implemented
1. ✅ **Rewritten README** to reflect the mature project state and ML insights
2. ✅ **Calibration study fixed** — isotonic consistently selected; valid BS values now produced
3. ✅ **Climate-index preprocessing fixed** — PDO `-9.9` sentinels are masked, trailing missing
   PDO is not stale-filled, and Niño3.4 absolute SST is converted to 1991–2020 monthly anomalies
4. ✅ **ENSO stratification fixed** — stratified BSS tables now have El Niño / La Niña / Neutral rows
   with monthly bootstrap confidence intervals
5. ✅ **Feature ablation script fixed** — `scripts/run_feature_ablation.py` uses the early-stopped
   XGBoost best iteration and current feature schema
6. ✅ **Corrected checkpoint documented** — `results/corrected_enso_spatial_checkpoint/` freezes
   the corrected ENSO-only + XGB-Spatial result set
7. ✅ **Auxiliary baselines refreshed** — LogReg and RF are retrained on the corrected ENSO-only
   schema and included in the current model-suite table
8. ✅ **Corrected SHAP refreshed** — `scripts/xgb_shap_forecast_analysis.py` now supports
   non-spatial and spatial XGBoost and writes current corrected-schema SHAP artifacts
9. ✅ **Seasonal SPI-3 lead-3 experiment added** — `scripts/run_spi3_seasonal_experiment.py`
   runs a non-overlapping target experiment without overwriting the canonical SPI-1 checkpoint;
   calibrated XGBoost remains below climatology (`BSS = -0.127`, CI below zero)
10. ✅ **ERA5-Land met-feature experiment added** — `scripts/download_era5_land_met_monthly.py`
   downloads t2m/d2m, and `scripts/run_met_feature_experiment.py` tests regional temperature/VPD
   anomaly lags without overwriting the canonical checkpoint; raw BSS improves to `-0.030`,
   but validation-selected calibration remains negative (`BSS = -0.092`)
11. ✅ **Spatialized ERA5-Land met-feature experiment added** —
   `scripts/run_met_spatial_feature_experiment.py` interpolates gridded t2m/VPD anomaly lags
   to the CHIRPS grid and combines them with XGBoost-Spatial features; selected BSS remains
   negative (`BSS = -0.047`, CI crossing zero)
12. ✅ **ERA5-Land soil-moisture experiment added** —
   `scripts/download_era5_land_soil_moisture_monthly.py` downloads volumetric soil-water
   layers, and `scripts/run_soil_moisture_feature_experiment.py` tests regional layer/root-zone
   anomaly lags without overwriting the canonical checkpoint; validation-selected BSS remains
   negative (`BSS = -0.158`, CI below zero)
13. ✅ **Multi-region XGBoost path added** —
   `scripts/region_config.py` defines candidate regions and
   `scripts/run_multiregion_xgb_experiment.py` runs region clipping, parallel SPI fitting,
   dataset build, and monthly BSS evaluation without overwriting canonical artifacts.
   Central Valley parity remains near climatology, and the first full Southern Great Plains
   tabular and spatial runs are negative (`selected BSS = -0.082`, CI just below zero).
14. ✅ **Mediterranean Spain full-region experiment added** —
   tabular and spatial XGBoost both show positive calibrated point estimates
   (`BSS = +0.044` and `+0.022`), but both CIs cross zero. This is the first
   hint of region-dependent positive skill, not yet a defensible positive-skill claim.
15. ✅ **Multi-region mechanism analysis added** —
   `scripts/analyze_multiregion_mechanisms.py` regenerates regional mechanism summaries,
   split dry-frequency stats, feature-group gain shares, BSS CI plots, monthly dry-fraction
   traces, signal-vs-skill plots, and an interpretation report under `results/multiregion/`.
16. ✅ **First-pass region geometry audit added** —
   `scripts/build_region_masks.py` builds Natural Earth country masks and writes diagnostics
   under `results/multiregion/`; Spain loses 5.64% of valid cells, and the masked Spain
   spatial sensitivity remains positive only as an uncertain point estimate (`BSS = +0.0235`).

### Next experiments (priority order)

1. **Improve region geometry beyond country masks**
   Add basin or hydroclimate masks for rectangular regions before making final publication claims.

2. **Promote mechanism diagnostics into publication figures**
   The current BSS CI, monthly dry-fraction, signal-vs-skill, and feature-group plots are
   now central to the paper narrative.

3. **Add one more full region if resources allow**
   Murray-Darling Basin or Horn of Africa would test whether the Mediterranean Spain hint is
   isolated or regime-linked.

4. **Extend seasonal SPI-3 only if needed**
   The first leakage-free tabular SPI-3 lead-3 result is negative. A fair next
   seasonal test would add spatial features or new exogenous drivers rather than
   simply tuning the same tabular model.

5. **Refresh corrected explainability artifacts after any model/schema change**
   ```bash
   python scripts/xgb_shap_forecast_analysis.py --model both
   ```

5. **Feature ablation / sensitivity checks**
   ```bash
   python scripts/run_feature_ablation.py
   ```
