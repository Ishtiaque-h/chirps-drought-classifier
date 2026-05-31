# Research Analysis — Central Valley Drought Classifier

> **Comprehensive evaluation and strategic roadmap for the chirps-drought-classifier project**
> Prepared as a combined ML scientist and peer-reviewer assessment.

Current documentation map: [`docs/README.md`](docs/README.md). Manuscript-facing
claims should be checked against `results/report/paper/` before being copied into a
paper draft.

Current narrative synthesis and literature-informed manuscript strategy:
[`final_report.md`](final_report.md). Tracked related works and collected PDFs:
[`literature/related_works_index.csv`](literature/related_works_index.csv).

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
| PRISM validation | ✅ Added | PRISM basin-mask SPI-1 comparison supports CHIRPS timing signal but shows CHIRPS is drier than PRISM in the test period |
| Spatial analysis | ✅ Complete | Per-pixel accuracy maps, Sacramento/San Joaquin sub-regions |
| Case study | ✅ Complete | 2021–22 drought / 2023 atmospheric rivers |
| Season-conditional skill | ✅ Quantified | Raw MAM skill is positive, but bootstrap CI crosses zero and calibration removes the signal |
| ENSO stratification | ✅ Fixed | Niño3.4 is now converted from absolute SST to anomalies; stratified rows populate correctly |
| Feature ablation | ✅ Complete | `scripts/run_feature_ablation.py` uses early-stopped XGBoost predictions and current features |
| Seasonal SPI-3 experiment | ✅ Initial test complete | Leakage-free SPI-3 lead-3 tabular XGBoost has positive but uncertain calibrated BSS; CI crosses zero |
| Seasonal SPI-6 experiment | ✅ Baseline fixed | SPI-6 lead-6 now uses a target-consistent SPI-6 persistence baseline; XGBoost remains below climatology |
| Temperature/VPD experiment | ✅ Initial test complete | Regional ERA5-Land t2m/VPD anomalies improve raw XGB but still do not beat climatology |
| Soil-moisture experiment | ✅ Initial test complete | Regional ERA5-Land soil-water/root-zone anomaly lags overfit and remain below climatology |
| Memory-target experiment | ✅ Added | Central Valley SPI-6 lead-6 lag/climate XGB has a positive but uncertain selected BSS (`+0.040`, CI crossing zero) with near-zero event tracking; adding ERA5-Land soil-memory lags worsens selected BSS |
| Transition-target experiment | ✅ Replication checked | Rectangular Central Valley onset is small but robust-positive (`BSS = +0.104`), but the signal fails under Central Valley basin masking and added-region checks; termination does not beat the eligible-state baseline |
| Evaluation-inflation audit | ✅ Added | Invalid random row splits and overlapping SPI targets produce very high apparent BSS (`+0.995` and `+0.674` monthly), while the strict chronological monthly audit remains tied with climatology |
| Multi-region path | ✅ Five-region path complete | Region registry + runner now supports Central Valley, Southern Great Plains, Murray-Darling, Mediterranean Spain, and Horn of Africa tabular/spatial tests |
| Regional mechanism comparison | ✅ Initial analysis complete | Reproducible diagnostics separate ranking, calibration, test-period shift, persistence, and feature-group gain |
| Temporal robustness audit | ✅ Added | Five rolling Central Valley holdouts show no positive tabular BSS point estimates; 2021–2026 is not the only weak window |
| SPI-12 regionalization mechanism tables | ✅ Added | Zone-level run metrics, climate-index correlations, and forecast diagnostics are compiled for paper tables |
| Region geometry audit | ✅ Source-cited masks added | Natural Earth country masks plus DWR, EPA, Murray-Darling Basin Authority/data.gov.au, and MITECO masks quantify rectangular-box sensitivity and add masked priority-region runs |
| Master results table | ✅ Added | `scripts/generate_master_results.py` creates a 183-row result table and 88-row headline table from current artifacts |
| Paper evidence pack | ✅ Added | `scripts/generate_manuscript_results.py` consolidates the master, seasonal, temporal, mask, PRISM, and regionalization evidence into manuscript-facing tables and figures under `results/report/paper/` |
| Operational benchmark path | ✅ NMME anomaly + probability benchmarks complete | CPC NMME anomaly and official below-normal probability benchmarks now cover SPI-1 lead-1, SPI-3 lead-3, and SPI-6 lead-6; the best selected probability row is SPI-1 lead-1 (`BSS = +0.131`) and the best raw probability row is SPI-6 lead-6 (`BSS = +0.035`), but all confidence intervals cross zero |
| Forecast-informed land-surface path | ✅ GEFSv12 hindcast added, replicated, transfer-tested, constrained, rare-event scored, persistence-regime audited, cross-product checked, persistence-residual gated, and modern/native archive extension audited | Central Valley CFSv2 RZSM remains robustly positive vs climatology (`BSS = +0.511` in the four-cycle replication); the five-region GEFSv12+persistence stack is robust-positive vs climatology in 4/5 regions and robustly improves selected persistence in 3/5; leave-one-region-out calibration is robust-positive in 5/5 regions and robustly improves transferred persistence in 4/5; monotonic XGBoost improves mean BSS but not universal added value; rare-event q0.80/q0.90 diagnostics show meaningful AP lift with region-dependent event-BSS; the persistence-regime diagnostic shows the largest stack gain when dry memory and wetter GEFSv12 forecasts disagree; the persistence-residual selector confirms that validation-safe monotonic/guarded/threshold rows improve raw persistence in only 2/5 leave-one-region-out regions and pooled monotonic in 3/5, so this is not a universal adaptive selector; SMAP cross-product transfer narrows the claim because direct GEFS transfer is weak/mixed, but validation-only base-rate corrections recover GEFS transfer signal; prediction-base-rate shifting reaches 10/10 robust-positive source-region rows, dry-rate shifting gives more robust persistence-relative added value, and complex target-specific adaptation is weaker on frozen SMAP test months; operational GEFS is accessible, but both the SOILW 0.1-1 m smoke test and the top-layer-compatible SOILL 0-1 m test are robustly negative in Central Valley 2024-2025, and diagnostics show no forecast candidate robustly beats raw persistence on validation; Central Valley SMAP L4 multi-snapshot targets now cover 2020-01 to 2026-03, and SMAP scoring confirms raw persistence is robust (`BSS = +0.825`) while selected operational GEFS stays negative; year-held-out calibration gives weak positive GEFS/persistence stack point estimates versus SMAP climatology but still loses to raw SMAP persistence, so the persistence guard falls back to persistence; C3S seasonal-original single levels is accessible and structurally verified; compact C3S/ECMWF system 51 VSM benchmarks do not provide a positive added-value extension: Central Valley selected C3S is positive but non-robust (`+0.107`, CI `[-0.149, +0.283]`) and loses to selected persistence (`-0.523`), Mediterranean Spain selected C3S is below climatology (`-0.307`, CI `[-0.710, +0.024]`) and also loses to selected persistence (`-0.366`), and Southern Great Plains selected C3S is positive but non-robust (`+0.085`, CI `[-0.089, +0.232]`) while losing strongly to selected persistence (`-1.092`); SubX/IRI currently returns authentication pages in this environment |

### Key results (corrected ENSO + spatial checkpoint — 2026-05-01)

> **Climate-index preprocessing is now sane.** PDO `-9.9` sentinels are masked,
> recent missing PDO values are not forward-filled, and Niño3.4 absolute SST is
> converted to monthly anomalies using the 1991–2020 climatology. The active
> corrected checkpoint uses Niño3.4 anomaly lags only; PDO is excluded because
> recent PDO values are missing after August 2025. A common-valid-period
> sensitivity without PDO tail forward-fill now shows that PDO does not rescue
> Central Valley SPI-1 skill: spatial PDO-only BSS is `-0.114`, and
> Niño3.4+PDO is robustly negative (`-1.385`, CI below zero).
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
> **The seasonal target result is suggestive but not yet defensible as temporal
> forecast skill.** A leakage-free Central Valley SPI-3 lead-3 setup (features
> at t, target SPI-3 ending t+3) gives calibrated XGBoost BSS = +0.036 with a
> 95% CI crossing zero. Regional SPI-3/SPI-6 long-lead tests are now summarized
> in `results/report/seasonal/seasonal_regional_longlead_summary.csv`: most rows are
> negative or uncertain, and the only robust-positive row is Mediterranean
> Spain SPI-6 lead-6 with Niño3.4-only features (BSS = +0.078, CI
> [+0.004, +0.162]). The signal audit in
> `results/report/seasonal/seasonal_regional_signal_audit.csv` flags that row as a
> calibration-shift result, not event tracking (`r = 0.041`, variance ratio =
> 0.104). The SPI-6 persistence baseline has been corrected to use `spi6_lag1`,
> which makes the baseline target-consistent and worse than the old SPI-3 proxy.
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
> **Forecast-informed root-zone soil moisture is promising, but not a broad
> added-value claim yet.** `scripts/run_landsurface_forecast_benchmark.py`
> verifies NOAA NCEI CFSv2 monthly-mean `flxf` soil-water forecasts against an
> ERA5-Land 0-100 cm root-zone soil-moisture dry-fraction target. Central
> Valley CFSv2 remains robustly positive after a strict four-cycle replication
> (`BSS = +0.511`, CI `[+0.292, +0.676]`), but it no longer beats raw
> persistence on point BS. Southern Great Plains CFSv2 is positive but
> uncertain (`+0.413`, CI crossing zero) and is much weaker than persistence;
> Mediterranean Spain is below climatology (`-0.141`). A NOAA GEFSv12
> public-reforecast RZSM checkpoint uses 11-member Wednesday long reforecasts,
> 2000-2016 hindcast calibration, and 2017-2019 frozen testing. The five-region
> GEFSv12+persistence stack is robust-positive against climatology in 4/5
> regions and robustly improves selected persistence by paired delta BS in 3/5
> regions. Leave-one-region-out calibration-transfer is stronger: robust-positive
> in 5/5 regions and robustly better than transferred persistence in 4/5, with
> mean stack BSS `+0.638`. A monotonic XGBoost dry-fraction variant increases
> mean BSS (`+0.669` under leave-one-region-out) but still has only 3/5 robust
> added-value rows. Rare-event q0.80/q0.90 diagnostics show meaningful PR-AUC
> lift, but event-BSS remains region/threshold dependent. This supports target
> reframing toward land-surface drought and transferability of calibration, but
> not universal dynamic-model added value or deployment readiness.
> A persistence-regime diagnostic now clarifies the added-value mechanism:
> under leave-one-region-out calibration, the stack improves transferred
> persistence overall (`delta BS = -0.0063`, CI `[-0.0101, -0.0028]`) and has
> its largest robust gain when antecedent dry memory is high but GEFSv12
> forecasts wetter-than-normal root-zone moisture (`delta BS = -0.0249`, CI
> `[-0.0431, -0.0101]`). The first independent-target validation is now
> complete for the two U.S. regions: against NLDAS Noah `SoilM_0_100cm`
> dry fraction, the validation-selected GEFSv12/persistence stack is
> robust-positive and robustly improves selected persistence in both Central
> Valley and Southern Great Plains. This reduces the ERA5-Land-only target
> vulnerability for U.S. checkpoints. A GLDAS Noah `RootMoist_inst`
> model-product sensitivity now covers all five regions: the
> validation-selected stack is robust-positive in 5/5 regions and robustly
> improves selected persistence in 3/5 regions. This supports target-product
> robustness, but GLDAS is still a model product; it is not the same as global
> independent satellite validation.
>
> **The first memory-target checkpoint is suggestive but not event-tracking
> skill.** `scripts/run_memory_target_experiment.py` tests Central Valley
> SPI-6 lead-6 with lag/climate features and then adds ERA5-Land soil-memory
> lags. Lag/climate XGBoost reaches selected `BSS = +0.040` with a CI crossing
> zero (`[-0.020, +0.082]`), but monthly rank correlation is essentially zero
> (`r = 0.004`) and the amplitude ratio is only `0.076`. Adding soil-memory
> lags worsens selected BSS to `-0.158` even though soil variables dominate
> feature gain. The CPC NMME coverage audit also shows zero overlap with the
> 1991-2016 training period, so NMME cannot be used as a trained ML feature
> without adding hindcasts.
>
> **The evaluation protocol is now experimentally justified.**
> `scripts/run_evaluation_inflation_audit.py` retrains comparable XGBoost
> checkpoints under invalid shortcuts. The strict chronological monthly SPI-1
> audit row stays near climatology (`BSS = -0.023`, CI crossing zero), but an
> invalid random-row split gives monthly `BSS = +0.995` and an overlapping
> SPI-3 lead-1 target gives monthly `BSS = +0.674`. Pixel-level inference also
> gives artificially tight intervals because hundreds of thousands of
> autocorrelated pixels are treated as independent. This is strong evidence that
> the project's strict monthly, leakage-free design is not cosmetic; it prevents
> major skill inflation.
>
> **The multi-region extension is now broad enough for the generalization claim.**
> `scripts/run_multiregion_xgb_experiment.py` now clips CHIRPS, computes
> region-specific SPI, builds the same SPI-1[t+1] forecast table, and evaluates
> monthly BSS for configured regions. Across Central Valley, Southern Great
> Plains, Murray-Darling, Mediterranean Spain, and Horn of Africa, no
> source-cited geometry checkpoint gives statistically robust positive selected
> BSS. Southern Great Plains improves under the EPA ecoregion mask (`BSS =
> +0.010`, CI [-0.098, +0.150]) but remains uncertain. Horn of Africa calibrates
> near climatology (`BSS = -0.031`, CI [-0.302, +0.320]) under a country-mask
> checkpoint. Murray-Darling is strongly negative after official basin masking
> (`BSS = -0.639`, CI [-1.308, -0.278]). Mediterranean Spain's rectangular
> positive point estimate does not survive stricter river-basin district masking
> (`spatial BSS = -0.0846`, CI crossing zero).
>
> **The mechanism diagnostics identify multiple failure modes.**
> `scripts/analyze_multiregion_mechanisms.py` shows that Central Valley has the
> strongest ranking signal but under-amplified calibrated probabilities
> (`selected amplitude ratio ≈ 0.21`). Rectangular Southern Great Plains has
> weak ranking, strong underprediction of test dry frequency, and a train-to-test
> dry shift from 0.160 to 0.208. The EPA ecoregion mask improves Southern Great
> Plains ranking and BSS, but still leaves a wide CI crossing zero. Basin-masked
> Central Valley strengthens raw ranking (`ROC-AUC ≈ 0.72`) but still remains
> below climatology. Murray-Darling has ranking signal (`ROC-AUC ≈ 0.63`) but
> negative selected-probability correlation, high positive bias, and a train-to-test
> dry-frequency drop from 0.147 to 0.109. Horn has weak ranking and a small
> negative selected BSS, but its CI is wide and crosses zero. Basin-masked Spain
> has weak/negative selected-probability correlation, so its rectangular positive
> point estimate was a geometry-sensitive artifact.
>
> **Basin geometry materially changes the sample.**
> `scripts/build_basin_masks.py` uses CA DWR Bulletin 118 groundwater basins for
> Central Valley, US EPA Level III ecoregions for Southern Great Plains, the
> official Murray-Darling Basin Boundary - Water Act 2007, and MITECO terrestrial
> river-basin districts for Spain. The DWR Central Valley mask retains 27.92% of
> valid rectangular cells, the EPA Southern Great Plains ecoregion mask retains
> 78.64%, the Murray-Darling Basin mask retains 52.07%, and the selected Spain
> basin-district mask retains 52.31%. Horn of Africa currently uses a Natural
> Earth country-intersection mask retaining 87.29%; that is a useful regional
> checkpoint but not a hydrologic or livelihood-zone boundary.
>
> **Temporal robustness is now directly tested.**
> `scripts/run_temporal_robustness_audit.py` retrains tabular XGBoost across
> five rolling chronological holdouts. Against train-period monthly climatology,
> none of the five holdouts has positive BSS (`2005–2008: -0.089`,
> `2009–2012: -0.004`, `2013–2016: -0.077`, `2017–2020: -0.010`,
> `2021–2026: -0.025`; all CIs cross zero). This means the weak-skill
> conclusion is not solely an artifact of the 2021–2026 test window. However,
> event-block diagnostics show the main canonical failure is severe
> underprediction during the 2021–2022 drought (`bias = -0.120`, BSS = -0.067),
> while 2023 and 2024–2026 are closer to climatology.
>
> **PRISM validation strengthens the data-source argument but adds a caveat.**
> `scripts/validate_chirps_prism_cvalley.py` downloads official PRISM monthly
> precipitation, clips it to the DWR Central Valley basin union, computes SPI-1
> with a 1991–2020 baseline, and compares it with CHIRPS. CHIRPS and PRISM
> monthly dry fractions are strongly aligned in the test period (`Pearson r =
> 0.816`, `Spearman r = 0.720`), but CHIRPS is drier on average (`test bias
> CHIRPS - PRISM = +0.046`). Evaluated against PRISM SPI-1 dry fraction, the
> current XGB-Spatial probabilities are again tied with climatology
> (`BSS = -0.003`, `Spearman = 0.530`).
>
> **SPI-12 regionalization is useful as mechanism evidence, not as forecast
> proof.** The curated mechanism table in
> `results/report/regionalization/regionalization_mechanism_summary.csv`
> joins zone-level PCA/run-theory/teleconnection summaries to the existing
> multi-region forecast probabilities. Horn of Africa, Murray-Darling,
> and Southern Great Plains show strong zone-level SPI-12 teleconnection
> correlations (e.g., Horn zone 0 Niño3.4 lag-6 `r = 0.542`, Murray-Darling zone
> 0 Niño3.4 lag-1 `r = -0.460`, Southern Great Plains zone 3 Niño3.4 lag-6
> `r = 0.501`). But zone-level SPI-1 forecast BSS remains mostly negative or
> only weakly positive in isolated zones. This motivated the SPI-3/SPI-6
> seasonal audit; the current seasonal results show that target-timescale
> mismatch is only a partial explanation, because longer targets still do not
> produce broad event-tracking skill.
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
- Test period (2021–2026) coincidentally includes an extreme drought (2021–22) followed by an extreme wet reversal (2023). This is good for case-study value but means the test set is not climatologically representative. This is now partly controlled by the rolling temporal audit, which shows no positive tabular BSS point estimate in five chronological holdouts, but event-block inference within 2021–2026 remains limited by only 12–27 months per block.
- The 30-year gamma-fit baseline (1991–2020) may not capture non-stationarity driven by climate change, potentially biasing SPI values in the 2021–2026 test period.

### 2.3 What the completed regional expansion now offers

| Benefit | Detail |
|---------|--------|
| **Generalizability test** | Does "no skill" hold in a different hydroclimate (e.g., semi-arid Great Plains, monsoonal India, Mediterranean Spain)? If yes, the finding is much more significant. If no, what feature or mechanism explains the difference? |
| **Transfer learning** | Train on one region, test on another — demonstrates whether spatial patterns are generalizable or regionally overfit |
| **Statistical power** | Multiple regions multiply the number of independent test months, tightening confidence intervals |
| **Publication impact** | Multi-region studies are far more publishable (reviewer concern: "would this hold elsewhere?" is preemptively answered) |
| **Mechanistic insight** | Comparing skill across regimes (Mediterranean vs. continental vs. monsoonal) reveals which climate properties enable or prevent predictability |

**Completed expansion checkpoints:**

1. **Great Plains, USA (Kansas–Oklahoma)** — rectangular tests are negative, while the EPA ecoregion-masked run improves to a small positive but uncertain spatial point estimate
2. **Mediterranean Spain (Ebro/Guadalquivir basins)** — rectangular and country-mask runs have positive uncertain point estimates, but the stricter basin-district run turns negative
3. **Murray-Darling Basin, Australia** — official basin-mask run is strongly negative despite moderate ranking signal, exposing calibration/test-period shift as a distinct failure mode
4. **Horn of Africa (Djibouti/Eritrea/Ethiopia/Kenya/Somalia intersection)** — country-mask run calibrates close to climatology but does not exceed it; keep a caveat that it is not a basin or livelihood-zone mask

**Verdict:** Additional regions are no longer the bottleneck. The scientific task is now to turn the five-region mechanism comparison into a clear, source-cited paper narrative.

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
| **Pacific Decadal Oscillation (PDO)** | NOAA | Low-frequency modulation of California precipitation on decadal timescales. | Sensitivity complete — common-valid-period PDO-only and Niño3.4+PDO do not improve Central Valley SPI-1 BSS |
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
2. **Separate lagged land-surface features from forecast-informed land-surface targets.** Temperature/VPD and ERA5-Land soil-memory lags are completed negative SPI-1 feature tests; CFSv2 forecast root-zone soil moisture is a narrower target-reframing signal, but the added-value audit does not show robust improvement over same-target persistence.
3. **Treat ablation cautiously.** Current ablation shows precipitation lags and seasonality help, while removing ENSO/SPI lags improves this trained XGB model. Because these features are correlated, this should be read as a trained-model diagnostic, not a causal feature-importance statement.
4. **Prioritize new independent information only if the paper needs a feature-extension section.** For Central Valley, atmospheric-river/subseasonal predictors are more defensible than more tuning of the same lagged land-surface fields. For the paper, the five-region generalization result is now higher value than adding more regions.

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
| 2 | **Do additional exogenous drivers improve skill beyond corrected ENSO?** Temperature/VPD, regional soil moisture, and common-valid PDO sensitivity are negative for SPI-1. Remaining Central Valley candidates should be physically event-informed, such as atmospheric-river/circulation predictors, rather than more low-frequency climate-index variants. | ⭐⭐⭐⭐ | ⭐⭐ |
| 3 | **How does skill vary with lead time and temporal aggregation?** Evaluate at seasonal (3-month) and quarterly horizons. SPI-3 as target may be more predictable at seasonal lead. | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| 4 | **Is there conditional skill?** Do models outperform climatology specifically during ENSO warm/cold phases, or during winter (wet season) vs. summer? Stratified BSS analysis. | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| 5 | **Transfer learning: can a model trained on one region predict drought in an analogous region?** Train on Central Valley, test on Ebro Basin (or vice versa). | ⭐⭐⭐⭐ | ⭐⭐⭐ |
| 6 | **What is the minimum feature set for positive skill?** Formal ablation + feature importance analysis to identify which features (if any) provide marginal predictive information. | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| 7 | **Anomaly detection vs. classification:** Frame the problem as detecting anomalous months (outlier detection) rather than 3-class classification. This may be better suited to the low base rate of drought. | ⭐⭐⭐ | ⭐⭐⭐⭐ |

---

## 6. Modern Research Context

### 6.1 Where this work sits in the literature

The project addresses a well-studied problem (ML-based drought prediction) but with unusually explicit forecast verification. The protocol audit in `literature/literature_protocol_audit.csv` shows that many related studies are not directly comparable to this strict monthly SPI-1 probability task because target scale, lead time, inputs, validation split, metrics, and baselines differ. The main comparability risks are:

- **Inflated accuracy from spatial pseudo-replication** — this project audits monthly versus pixel-level inference
- **Missing climatology-relative baselines** — this project includes climatology, persistence, and operational forecast benchmarks
- **Target leakage from SPI accumulation windows** — the canonical task uses SPI-1[t+1], and the invalid overlapping SPI-3 audit quantifies the risk
- **Limited uncertainty quantification** — this project includes bootstrap CI and calibration analysis

The finding that ML does not reliably outperform climatology at 1-month lead is consistent with a cautious reading of the literature:
- Su et al. (2023, *Journal of Hydrometeorology*) found that subseasonal drought onset/termination skill over the coastal western United States degrades sharply by week 4 ([doi:10.1175/JHM-D-22-0103.1](https://doi.org/10.1175/JHM-D-22-0103.1)).
- AghaKouchak et al. (2023, *Nature Reviews Earth & Environment*) frame drought as a cascading, impact-dependent hazard, supporting the need to test more than a single precipitation-index forecast target before making broad operational claims ([doi:10.1038/s43017-023-00457-2](https://doi.org/10.1038/s43017-023-00457-2)).
- Dikshit et al. (2021, *Journal of Environmental Management*) show that deep-learning drought forecasting can help for SPEI-style targets under different predictor/target designs, which is useful context but not direct evidence that the stricter SPI-1 lead-1 CHIRPS setup here should beat climatology ([doi:10.1016/j.jenvman.2021.111979](https://doi.org/10.1016/j.jenvman.2021.111979)).
- NMME/CFSv2/GEFSv12 provides a more appropriate benchmark than another ML-only architecture because these sources supply actual forecast fields, not only lagged observed predictors. NOAA CPC documents NMME data access and citation requirements ([CPC NMME data](https://www.cpc.ncep.noaa.gov/products/NMME/data.html)), NCEI describes NMME as a global multi-model seasonal forecast archive with precipitation variables ([NCEI NMME](https://www.ncei.noaa.gov/products/weather-climate-models/north-american-multi-model)), and Kirtman et al. (2014) is the core NMME reference ([doi:10.1175/BAMS-D-12-00050.1](https://doi.org/10.1175/BAMS-D-12-00050.1)). The project now includes a NOAA NCEI THREDDS CFSv2 individual-run SPI-3 lead-3 precipitation extraction, a monthly-mean CFSv2 `flxf` soil-moisture benchmark ([NCEI monthly means catalog](https://www.ncei.noaa.gov/thredds/catalog/model-cfs_v2_for_mm/catalog.html)), and a NOAA GEFSv12 reforecast RZSM benchmark using the public AWS archive and Guan et al. (2022) as the dataset reference ([doi:10.1175/MWR-D-21-0245.1](https://doi.org/10.1175/MWR-D-21-0245.1)). The precipitation benchmark remains non-robust; root-zone soil-moisture benchmarks show stronger land-surface predictability, but broad dynamic-model added value over persistence is not established.
- SubX is the analogous subseasonal benchmark path if the paper emphasizes weeks 3-4 or monthly aggregation from subseasonal forecasts; cite Pegion et al. (2019) ([doi:10.1175/BAMS-D-18-0270.1](https://doi.org/10.1175/BAMS-D-18-0270.1)).

### 6.2 Gaps this project could fill

1. **Rigorous negative result in drought ML** — the literature has a publication bias toward positive results; a methodologically sound "no skill" finding with thorough explanation is valuable
2. **Multi-region predictability comparison** — very few studies compare ML drought skill across diverse hydroclimates using identical methodology
3. **Explicit predictability decomposition** — BS decomposition (reliability/resolution/uncertainty) is rarely reported in drought ML papers; this is a methodological contribution

### 6.3 Core data and method references

Primary citations for the methods section:

- **CHIRPS v3:** Funk et al. (2026), *Scientific Data*, CHIRPS Version 3 ([doi:10.1038/s41597-026-07096-4](https://doi.org/10.1038/s41597-026-07096-4)).
- **Original CHIRPS record:** Funk et al. (2015), *Scientific Data*, CHIRPS environmental record ([doi:10.1038/sdata.2015.66](https://doi.org/10.1038/sdata.2015.66)).
- **SPI calculation standard:** World Meteorological Organization (2012), *Standardized Precipitation Index User Guide*, WMO-No. 1090 ([WMO Library](https://library.wmo.int/idurl/4/39629)).
- **Brier Score decomposition:** Murphy (1973), *Journal of Applied Meteorology*, vector partition of the probability/Brier score ([ADS record](https://ui.adsabs.harvard.edu/abs/1973JApMe..12..595M/abstract)).

### 6.4 Boundary and mask source references

The regional mask analyses use source-cited public boundary datasets:

- **Country masks:** Natural Earth 1:50m Admin 0 country polygons via the project GitHub mirror ([GeoJSON](https://github.com/nvkelso/natural-earth-vector/blob/master/geojson/ne_50m_admin_0_countries.geojson)).
- **Central Valley groundwater-basin mask:** California Department of Water Resources Bulletin 118 groundwater basins, selecting Sacramento Valley and San Joaquin Valley basin numbers 5-021 and 5-022 ([FeatureServer](https://gis.water.ca.gov/arcgis/rest/services/Geoscientific/i08_B118_CA_GroundwaterBasins/FeatureServer/0)).
- **Southern Great Plains ecoregion mask:** US EPA Level III Ecoregions of the Continental United States, selecting Level II `SOUTH CENTRAL SEMI-ARID PRAIRIES` features intersecting the configured Southern Great Plains box ([EPA page](https://www.epa.gov/eco-research/level-iii-and-iv-ecoregions-continental-united-states)).
- **Murray-Darling Basin mask:** Murray-Darling Basin Authority / data.gov.au boundary defined under Section 4(1) of the Water Act 2007 ([WFS GeoJSON](https://data.gov.au/geoserver/murray-darling-basin-boundary/wfs?request=GetFeature&typeName=ckan_4ede9aed_5620_47db_a72b_0b3aa0a3ced0&outputFormat=json)).
- **Mediterranean Spain river-basin mask:** MITECO terrestrial river-basin district collection `agua:Demarcaciones_ET`, selecting Ebro, Catalonia internal basins, Jucar, Segura, Andalusia Mediterranean basins, and Guadalquivir ([OGC collection](https://wmts.mapama.gob.es/sig-api/ogc/features/v1/collections/agua%3ADemarcaciones_ET)).

---

## 7. Strategic Recommendations

### Ranked by impact × feasibility:

| Rank | Action | Impact | Feasibility | Rationale |
|------|--------|--------|-------------|-----------|
| **1** | **Write source-cited mask methods** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | The paper should explicitly cite Natural Earth, DWR, EPA, MDBA/data.gov.au, and MITECO boundary sources, report retained-cell fractions, and caveat Horn's country mask. |
| **2** | **Turn mechanism diagnostics into paper figures** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | The five-region, geometry-sensitive comparison is now the main scientific story and should be presented before adding more features. |
| **3** | **Perform final consistency review of claims** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | Ensure every positive point estimate is framed as uncertain and every rectangular result is labeled as a sensitivity rather than a final regional claim. |
| **4** | **Land-surface hindcast/ensemble benchmark** | ⭐⭐⭐⭐⭐ | ⭐⭐ | GEFSv12 RZSM now adds a true public-reforecast benchmark with five-region stack, lead/valid-day sensitivity, leave-one-region-out transfer, monotonic constraints, and rare-event diagnostics; the signal is strong vs climatology but persistence remains a hard regional baseline. |
| **5** | **Atmospheric-river or subseasonal circulation predictors** | ⭐⭐⭐⭐ | ⭐⭐ | Central Valley monthly extremes are event-driven; AR/circulation predictors are more physically targeted than more lagged land-surface tuning. |
| **6** | **Seasonal target variants with more information** | ⭐⭐⭐ | ⭐⭐⭐ | SPI-3 lead-3 is a positive but uncertain hint; revisit with spatial features, operational precipitation forecasts, or additional regions if needed. |
| **7** | **Gridded/SMAP soil-moisture sensitivity** | ⭐⭐ | ⭐⭐ | Regional ERA5-Land soil moisture overfits; only pursue this if a spatial or independent-observation formulation is needed for completeness. |
| **8** | **Refresh corrected explainability as figures evolve** | ⭐⭐ | ⭐⭐⭐⭐ | Current SHAP artifacts are refreshed; rerun only after model/schema changes. |
| **9** | **Regional/seasonal stratified diagnostics with more months** | ⭐⭐⭐⭐ | ⭐⭐⭐ | Current MAM/ENSO hints have CIs crossing zero; more independent months or regions are needed. |
| **10** | **Transfer learning experiment** | ⭐⭐⭐⭐ | ⭐⭐ | High novelty but should wait until the source-cited regional results are written cleanly. |

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
> **Implications:** (1) ML model accuracy reported without climatology-relative baselines can overstate forecast utility. (2) ENSO and spatial context add information, but not enough for statistically reliable positive probability skill at this horizon. (3) The same weak-skill pattern now appears across five source-cited regional checkpoints, although the mechanism differs by region: under-amplification in California, geometry sensitivity in Spain, dry-frequency shift in the Great Plains, calibration shift in Murray-Darling, and near-climatology calibration in Horn of Africa. (4) The methodology presented here provides a template for rigorous drought ML evaluation that correctly accounts for spatial autocorrelation, class frequency, calibration, region geometry, and uncertainty.

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
9. ✅ **Seasonal SPI-3 lead-3 experiment added** — `scripts/run_seasonal_longlead_experiment.py`
   runs a non-overlapping target experiment without overwriting the canonical SPI-1 checkpoint;
   calibrated XGBoost has a positive but uncertain point estimate (`BSS = +0.036`,
   CI crossing zero)
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
   rectangular-screen hint, not a defensible positive-skill claim.
15. ✅ **Multi-region mechanism analysis added** —
   `scripts/analyze_multiregion_mechanisms.py` regenerates regional mechanism summaries,
   split dry-frequency stats, feature-group gain shares, BSS CI plots, monthly dry-fraction
   traces, signal-vs-skill plots, and an interpretation report under `results/multiregion/`.
16. ✅ **First-pass region geometry audit added** —
   `scripts/build_region_masks.py` builds Natural Earth country masks and writes diagnostics
   under `results/multiregion/`; Spain loses 5.64% of valid cells, and the masked Spain
   spatial sensitivity remains positive only as an uncertain point estimate (`BSS = +0.0235`).
17. ✅ **Basin/hydroclimate/ecoregion masks added for priority regions** —
   `scripts/build_basin_masks.py` builds DWR Central Valley groundwater-basin, EPA
   Southern Great Plains ecoregion, and MITECO Spain river-basin district masks.
   Basin-masked Central Valley remains near but below climatology (`best BSS = -0.0206`),
   EPA ecoregion-masked Southern Great Plains improves to a small positive but uncertain
   spatial point estimate (`BSS = +0.0101`, CI crossing zero), and basin-masked Spain
   turns negative (`best BSS = -0.0846`).
18. ✅ **Murray-Darling official basin-mask experiment added** —
   official Water Act 2007 basin geometry retains 52.07% of valid rectangular cells.
   The masked run has moderate ranking (`ROC-AUC ≈ 0.63`) but strongly negative
   selected BSS (`best BSS = -0.639`, CI below zero) because calibration/test-period
   shift biases selected probabilities high.
19. ✅ **Horn of Africa country-mask experiment added** —
   Natural Earth country intersection for Djibouti, Eritrea, Ethiopia, Kenya, and
   Somalia retains 87.29% of valid rectangular cells. The masked run calibrates close
   to climatology but does not beat it (`best BSS = -0.031`, CI crossing zero); this
   should be caveated as a political-region mask rather than a basin or livelihood-zone mask.
20. ✅ **SPI-6 persistence baseline fixed** —
   `scripts/run_seasonal_longlead_experiment.py` now uses `spi{target_spi}_lag1`
   for the persistence baseline. The refreshed SPI-6 lead-6 output has
   target-consistent persistence (`BSS = -0.949`) and calibrated XGBoost still
   below climatology (`BSS = -0.110`, CI crossing zero).
21. ✅ **Master result table added** —
   `scripts/generate_master_results.py` writes
   `results/report/master_results_table.csv` and
   `results/report/master_results_headline.csv`. Canonical SPI-1 rows still
   have no robust positive BSS result; robust positives now occur only in
   land-surface/diagnostic rows.
22. ✅ **Operational/dynamical benchmark added** —
   `scripts/prepare_cpc_nmme_precip_anomaly_inputs.py` preprocesses CPC NMME real-time
   multi-model precipitation anomalies, and
   `scripts/prepare_cpc_nmme_precip_probability_inputs.py` preprocesses official CPC NMME
   below-normal precipitation probabilities.
   `scripts/run_operational_precip_benchmark.py` scores them with the same
   validation-only calibration and monthly BSS protocol. The anomaly SPI-3
   lead-3 row remains the strongest anomaly-based hint (`BSS = +0.086`, CI
   crossing zero). The official probability rows add a cleaner probabilistic
   benchmark, but are still not robust: selected SPI-1 lead-1 has
   `BSS = +0.131` with a CI crossing zero, raw SPI-6 lead-6 has
   `BSS = +0.035` with a CI crossing zero, and isotonic calibration degrades the
   SPI-3/SPI-6 probability rows because the 2019-2020 validation overlap is
   short. CPC probability coverage is also partial for Central Valley dry-season
   targets.
23. ✅ **PDO common-valid-period sensitivity added** —
   `scripts/run_climate_index_sensitivity_experiment.py` compares CHIRPS/SPI-only,
   Niño3.4-only, PDO-only, and Niño3.4+PDO variants after dropping months where
   any climate-index lag is unavailable. On the shared 2021-01 to 2025-09 test
   window, PDO-only is negative and uncertain for spatial XGBoost (`BSS =
   -0.114`) and Niño3.4+PDO is robustly negative (`BSS = -1.385`, CI below
   zero). This closes the immediate concern that excluding PDO may have hidden a
   positive Central Valley SPI-1 signal.
24. ✅ **GEFSv12 RZSM hindcast benchmark added** —
   `scripts/run_gefsv12_landsurface_benchmark.py` extracts NOAA GEFSv12
   reforecast `soilw_bgrnd` root-zone soil moisture from the public AWS archive
   using byte-range GRIB reads. The main Central Valley run uses all 11
   Wednesday long-reforecast members, observed thresholds through 2016,
   hindcast calibration over 2000-2016, and a frozen 2017-2019 test. Selected
   GEFSv12 BSS is robustly positive in Southern Great Plains (`+0.753`, CI
   `[+0.618, +0.843]`) and Mediterranean Spain (`+0.540`, CI `[+0.090,
   +0.757]`), and positive but uncertain in Central Valley (`+0.351`, CI
   `[-0.204, +0.656]`). Same-target persistence remains a serious baseline:
   Southern Great Plains GEFSv12 beats persistence on point BSS, while
   Mediterranean Spain persistence is stronger; paired delta-BS intervals cross
   zero in both regions.
25. ✅ **GEFSv12 Southern Great Plains lead/valid-day sensitivity added** —
   `scripts/run_gefsv12_landsurface_sensitivity.py` sweeps target-month valid
   days 5, 10, 15, and 20 plus 0/1-week initialization lag under the same
   hindcast-calibrated protocol. All 8 rows are robust-positive versus
   climatology (`BSS = +0.621` to `+0.753`), and 7/8 beat selected persistence
   on point BSS. The one exception is the longest-lead corner
   (day 20, lag 1 week), which remains robust versus climatology but is
   essentially tied with persistence.
26. ✅ **GEFSv12+persistence stack benchmark added** —
   `scripts/run_gefsv12_landsurface_stack_benchmark.py` reads the completed
   GEFSv12 hindcast forecast files, refits validation-only isotonic mappings,
   and selects a convex GEFSv12/persistence blend by validation Brier score.
   On the frozen 2017-2019 test period, the five-region validation-selected
   stack is robust-positive versus climatology in 4/5 regions and improves
   selected persistence by paired delta BS in 3/5. Murray-Darling remains
   persistence-dominated despite high climatology-relative BSS.
27. ✅ **Worst-corner GEFSv12 stack sensitivity added** —
   The Southern Great Plains day20/+1w extraction, the weakest single-source
   GEFSv12 row from the lead/valid-day grid, remains robust-positive after
   stacking (`BSS = +0.670`, CI `[+0.401, +0.852]`). Its paired added value over
   selected persistence is uncertain (`delta BS = -0.0029`, CI `[-0.0081,
   +0.0017]`), which keeps the claim conservative.
28. ✅ **Land-surface domain transfer and constrained model added** —
   `scripts/run_landsurface_domain_transfer_benchmark.py` evaluates local,
   pooled, and leave-one-region-out calibration transfer, plus a monotonic
   XGBoost dry-fraction model constrained by GEFSv12 dry anomaly and same-target
   persistence. Leave-one-region-out stack calibration is robust-positive in
   5/5 regions and robustly improves transferred persistence in 4/5; monotonic
   XGBoost has higher mean BSS but only 3/5 robust added-value rows.
29. ✅ **Rare-event land-surface formulation added** —
   The same benchmark defines validation-period q0.80/q0.90 regional extensive
   dry-event thresholds and reports AP, event-BSS, and reliability/resolution.
   Leave-one-region-out q0.80 stack/monotonic rows have about 3x mean AP lift
   and 3/5 robust event-BSS rows; q0.90 has larger AP lift but very small event
   counts and fewer robust event-BSS rows.
30. ✅ **Land-surface persistence-regime diagnostic added** —
   `scripts/analyze_landsurface_persistence_regimes.py` stratifies
   leave-one-region-out land-surface monthly scores by season, antecedent dry
   fraction, GEFSv12 anomaly sign, and memory/forecast agreement. The stack
   robustly improves transferred persistence overall (`delta BS = -0.0063`)
   and most clearly helps when antecedent dry memory is high but GEFSv12
   forecasts wetter-than-normal root-zone moisture (`delta BS = -0.0249`).
31. ✅ **Independent land-surface target audit, NLDAS validation, GLDAS
   sensitivity, and SMAP L4 validation added** —
   `scripts/audit_landsurface_independent_targets.py` records candidate
   external target products and local-file readiness. NLDAS Noah monthly soil
   moisture is downloaded for 1991-2019 and scored for Central Valley and
   Southern Great Plains using `scripts/run_nldas_landsurface_validation.py`.
   `results/report/paper/table18_landsurface_nldas_validation.csv` shows the
   validation-selected GEFSv12/persistence stack is robust-positive and
   robustly improves selected persistence in both U.S. regions.
   `results/report/paper/table19_landsurface_target_product_comparison.csv`
   and `fig07_landsurface_target_product_comparison.png` provide the compact
   ERA5-Land versus NLDAS target-product comparison. GLDAS Noah monthly
   `RootMoist_inst` is downloaded for 2000-2019 and scored for all five
   regions using `scripts/run_gldas_landsurface_validation.py`.
   `results/report/paper/table20_landsurface_gldas_validation.csv` and
   `table21_landsurface_era5_gldas_comparison.csv` show that the
   validation-selected stack remains robust-positive in 5/5 regions under a
   global model-product target, with robust added value over selected
   persistence in 3/5. SMAP L4 SPL4SMGP `sm_rootzone_pctl` is now scored as a
   short-record satellite-assimilated target check over 2015-2019. The
   mid-month proxy in `table22_landsurface_smap_l4_validation.csv` is
   robust-positive in 3/5 regions and robustly improves selected persistence in
   1/5. The early/mid/late snapshot sensitivity in
   `table23_landsurface_smap_l4_snapshot_sensitivity.csv` is robust-positive in
   5/5 and robustly improves selected persistence in 2/5, so it reduces the
   mid-month-proxy concern but still does not support universal dynamic added
   value over persistence. A stricter cross-product transfer diagnostic in
   `table24_landsurface_target_product_transfer.csv` scores ERA5-Land- and
   GLDAS-calibrated probabilities against SMAP without SMAP recalibration:
   GEFS-only rows contain the only robust-positive cross-product cases, but
   source-product selected stacks are robust-positive in 0/10 and no compact
   transfer row robustly improves SMAP same-target persistence. This supports a
   transferable dynamical forecast signal only cautiously; calibration and
   persistence blending are target-product dependent. The follow-up
   calibration-transfer ladder in
   `table25_landsurface_calibration_transfer_ladder.csv` shows that direct
   source-calibrated GEFS transfer is robust-positive in 3/10 rows, while a
   validation-only source-to-SMAP dry-rate shift raises GEFS transfer to 9/10
   and robustly improves SMAP persistence in 3/10. Pooled ERA5-Land+GLDAS
   source calibration gives robust-positive selected stacks in 3/5 SMAP
   regions. This makes calibration/base-rate mismatch the more likely
   cross-product failure mode than complete loss of forecast signal. The
   follow-up adaptation benchmark in `table27_landsurface_target_product_adaptation_benchmark.csv`
   uses validation months only. Product-specific prediction-base-rate shifting
   makes GEFS robust-positive in 10/10 source-region rows, while the observed
   source-to-SMAP dry-rate shift gives 9/10 and more robust added-value rows
   over SMAP persistence (3/10 versus 2/10). Validation-selected seasonal and
   complex target-specific selectors are weaker on frozen SMAP test months,
   making base-rate calibration the publishable contribution and complex
   adaptation a future direction. The SMAP product-residual benchmark
   (`table34`/`table35`) tests the same methods against raw same-target SMAP
   persistence: leading GEFS base-rate rows are robust-positive against SMAP
   climatology in 9/10 to 10/10 rows, but robustly improve raw SMAP persistence
   in only 3/10; threshold/base-rate gating has the same 3/10 added-value
   ceiling, and all-candidate/stack-heavy selectors are weaker. This reinforces
   target-product calibration as the method claim and rejects a stronger
   mature-domain-adaptation claim for now. The year-by-year sensitivity table
   (`table28_landsurface_base_rate_yearly_sensitivity.csv`) shows all leading
   GEFS base-rate methods remain positive against SMAP climatology in 2017,
   2018, and 2019, but 2019 is weak relative to SMAP persistence. The modern
   operational-GEFS replication (`table36`) now checks Central Valley, Southern
   Great Plains, and Mediterranean Spain for SOILW 0.1-1 m and SOILL 0-1 m
   over 2021-2025. Selected operational GEFS is robust-positive in 0/6
   region/soil-mode rows, robust-negative in 2/6, and positive on point BSS in
   only 2/6; the best row is Mediterranean Spain SOILL (`BSS = +0.049`, CI
   `[-1.032, +0.597]`) and still loses to raw persistence. This reduces the
   Central-Valley-only concern. The follow-up native-RZSM archive audit
   (`table37`/`table38`) identifies C3S seasonal-original single levels as the
   cleanest next archive if licence access is resolved: ECMWF system 51 exposes
   native `volumetric_soil_moisture` for 1981-2026, all months, and 24-5160 h
   leads. After CDS licence acceptance, the corrected tiny retrieval probe
   succeeds and returns a 51-member, 4-soil-layer NetCDF for the Central Valley
   test box. SubX/IRI endpoints return authentication pages. The next clean
   experiment was therefore a compact ECMWF system 51 VSM benchmark. It
   confirms access but not a strong positive claim: Central Valley selected C3S
   BSS is `+0.107` over complete 2021-2025 testing, with CI crossing zero, and
   it loses to selected persistence (`BSS vs persistence = -0.523`).
   Mediterranean Spain replication is weaker: selected C3S BSS is `-0.307`
   and also loses to selected persistence (`-0.366`). Southern Great Plains is
   the diagnostic control because GEFSv12 was strongest there; selected C3S is
   still only positive-uncertain (`BSS = +0.085`, CI `[-0.089, +0.232]`) and
   loses strongly to selected persistence (`-1.092`). This argues against a
   C3S-specific persistence-residual selector as the next major method.

### Next experiments / writing priorities

1. **Stop adding target products unless the manuscript specifically needs one**
   NLDAS Noah monthly soil moisture now provides the first independent U.S.
   target validation for Central Valley and Southern Great Plains, GLDAS
   provides the five-region model-product sensitivity check, and SMAP L4
   provides both mid-month and early/mid/late short-record satellite-assimilated
   checks. GLEAM/ESA CCI should remain optional sensitivity checks.
   Reproducible commands are:
   `python scripts/download_nldas_noah_monthly.py --start 1991-01 --end 2019-12`
   followed by
   `python scripts/run_nldas_landsurface_validation.py --regions cvalley southern_great_plains --copy-report`,
   and
   `python scripts/download_gldas_noah_monthly.py --start 2000-01 --end 2019-12`
   followed by
   `python scripts/run_gldas_landsurface_validation.py --copy-report`, and
   `python scripts/download_smap_l4_regional_subsets.py --out-dir data/raw/smap_l4_spl4smgp_midmonth --start 2015-04 --end 2019-12 --snapshot-days 5 15 25`
   followed by
   `python scripts/run_smap_l4_landsurface_validation.py --smap-dir data/raw/smap_l4_spl4smgp_midmonth --out-prefix landsurface_smap_l4_multi3snapshot_gefsv12_validation --target-tag multi3snapshot --copy-report`.

2. **Use the master and support tables as the paper source of truth**
   Build paper tables from `results/report/master_results_headline.csv`; do not
   manually copy numbers from older prose. Use `results/temporal/`,
   `results/validation/prism_*`, and `results/report/regionalization/zone_forecast_diagnostics.csv`
   as supporting diagnostics.

3. **Write the source-cited data/mask-methods subsection**
   Include the boundary sources, selection logic, retained-cell fractions, and why
   each masked run is a cleaner scientific checkpoint than a rectangular bbox. State
   explicitly that Horn is country-intersection geometry, not a hydrologic/livelihood mask.
   Add the PRISM validation method as an independent U.S. precipitation-data check.

4. **Use `results/report/paper/` as the manuscript evidence pack**
   `scripts/generate_manuscript_results.py` now creates the master/headline
   evidence tables plus mask, temporal, seasonal, regionalization, transition,
   land-surface added-value, climate-index, GEFSv12 sensitivity, stack,
   reliability, domain-transfer, rare-event, persistence-regime, and independent
   target-audit tables. This is now the
   highest-level evidence source for manuscript drafting.

5. **Do not add more regions before writing**
   Five checkpoints are enough for the generalization claim; the immediate risk is
   narrative inconsistency, not lack of regional coverage.

6. **Treat the land-surface benchmark as promising but persistence-limited**
   CPC NMME anomaly/probability benchmarks now cover SPI-1 lead-1, SPI-3
   lead-3, and SPI-6 lead-6, and NCEI CFSv2 now adds a true lead-window SPI-3
   lead-3 precipitation benchmark. The CFSv2 raw-amount row is near climatology
   (`BSS = -0.030`, CI crossing zero) and the anomaly row is worse
   (`BSS = -0.303`, CI crossing zero). The CFSv2 root-zone soil-moisture
   benchmark remains robustly positive in Central Valley under four-cycle
   aggregation, but `results/report/paper/table09_landsurface_added_value.csv`
   shows no robust overall added value over raw persistence. Southern Great
   Plains persistence is robustly better than CFSv2. The GEFSv12 RZSM
   hindcast-calibrated rows now replicate robust climatology skill across
   multiple regions. The Southern Great Plains lead/valid-day grid is stable,
   and the five-region validation-selected GEFSv12+persistence stack supports a
   cautious combined forecast-memory benchmark. Domain-transfer and rare-event
   diagnostics strengthen the claim, but persistence dominance in Murray-Darling
   and small 36-month tests remain important limitations.

7. **Treat seasonal regional results as a calibration/target-design audit**
   The expanded seasonal table has one robust-positive BSS row, but its
   near-zero correlation and low variance indicate calibration shift rather
   than useful event timing. A fair next seasonal test should use independent
   forecast precipitation or circulation predictors, not more tuning of the
   same lagged-observation tabular model.

8. **Treat onset transitions as a target-design diagnostic**
   The corrected transition experiment shows a small robust-positive
   rectangular Central Valley onset result only after eligible-pixel scoring and
   eligible-only training (`BSS = +0.104`). It does not survive Central Valley
   basin masking, is robustly negative in Southern Great Plains, and is near
   climatology in Mediterranean Spain. Termination remains non-skillful. This
   is useful evidence about target design, not a paper headline.

8. **Refresh corrected explainability artifacts after any model/schema change**
   ```bash
   python scripts/xgb_shap_forecast_analysis.py --model both
   ```

9. **Feature ablation / sensitivity checks**
   ```bash
   python scripts/run_feature_ablation.py
   ```

## 10. Scientific Strength Upgrades Already in Place

These are the changes that most improve the scientific credibility of the project:

1. **Leakage-safe target design.** The canonical forecast target is now SPI-1[t+1] or SPI-3/6 at non-overlapping leads, so features never reuse the same accumulation window as the label.
2. **Monthly-level inference.** Skill is always summarized over independent test months, not pixels, which avoids pseudo-replication and inflated significance.
3. **Calibration with validation-only selection.** Isotonic and Platt calibration are fit on the validation split only, then frozen before test evaluation.
4. **Multi-region generalization checks.** Central Valley is no longer treated as a one-off result; the same pipeline has been run across Great Plains, Spain, Murray-Darling, and Horn of Africa checkpoints.
5. **Temporal robustness controls.** Five rolling Central Valley holdouts now test whether the 2021–2026 result is an artifact of a single unusual test period.
6. **Independent precipitation-product validation.** PRISM SPI-1 validation checks whether the CHIRPS target itself is driving the conclusion.
7. **Conditional-skill diagnostics.** ENSO- and season-stratified BSS is now computed and saved in `results/report/seasonal/seasonal_monthly_scores_stratified_bss.csv`; regional long-lead seasonal BSS is summarized in `results/report/seasonal/seasonal_regional_longlead_summary.csv`; and `results/report/seasonal/seasonal_regional_signal_audit.csv` separates calibration shifts from temporal tracking.

### What the new seasonal stratification actually says

The seasonal summaries are useful, but they are not yet strong enough to support an operational claim:

- SPI-3 lead-3 shows a small overall isotonic gain at the monthly level, but the confidence interval still crosses zero.
- SPI-3 lead-6 is weaker overall and remains below climatology after calibration.
- The strongest conditional skill appears in El Niño months and some individual season bins, but several of those bins have only 2–3 months, so they should be presented as hypothesis-generating rather than confirmatory.
- Multi-region SPI-3/SPI-6 long-lead tests do not show a broad timescale fix. Southern Great Plains SPI-6 lead-6 is robustly negative, Murray-Darling Niño3.4-only seasonal runs are robustly negative, Horn of Africa country-mask seasonal runs remain below climatology with wide uncertainty, and Mediterranean Spain SPI-6 lead-6 is robust-positive only after isotonic calibration while showing near-zero event correlation.

In practice, this means the project is now scientifically stronger because it can say **where the signal might live** for precipitation SPI, while using the CFSv2 RZSM result as a separate, narrower land-surface target claim: robust against climatology in Central Valley, mixed across added regions, and not demonstrably additive over same-target persistence in the current monthly-mean extraction.
