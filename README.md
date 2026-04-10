# Central Valley Drought Classifier
**One-month-ahead drought forecasting from precipitation using machine learning**

This project forecasts next month's drought class — *Dry*, *Normal*, or *Wet* — for California's Central Valley using WMO-standard SPI indices derived from 35 years (1991–2025) of CHIRPS v3.0 precipitation. The pipeline is built to journal standards: no data leakage, monthly-level evaluation against three naive baselines, and full probabilistic scoring with calibration and uncertainty quantification.

---

## Study Region and Problem

**Central Valley, California** (35.4°–40.6°N, 122.5°–119.0°W, 0.05° grid, ~7,200 pixels).

The region produces roughly 25% of US food supply. Drought is the primary agricultural risk — early warning one month ahead allows pre-season irrigation scheduling and crop-stress mitigation.

**Forecast task:** given the SPI-1/3/6 and raw precipitation history through month *t*, predict whether month *t+1* will be *dry* (SPI-1 ≤ −1), *normal*, or *wet* (SPI-1 ≥ +1).

---

## Data and Labels

| Source | Description | Period | Resolution |
|--------|-------------|--------|------------|
| [CHIRPS v3.0](https://www.chc.ucsb.edu/data/chirps3) | Gauge-satellite blended precipitation | 1991–2025 | 0.05° monthly |

**SPI computation (WMO standard):** for each pixel × calendar month, a gamma distribution is fitted to the 1991–2020 baseline precipitation. All values (1991–2025) are transformed via the gamma CDF and then the inverse normal to produce SPI-1 (single-month), SPI-3, and SPI-6.

**Three-class target:** `drought_label_spi1[t+1]` — the SPI-1 class for the *next* month.
- The target depends only on `pr[t+1]`, which is not in the feature set.
- No accumulation-window overlap with features (unlike SPI-3 or SPI-6 as target).

---

## Feature Set (9 features, all available at forecast time *t*)

| Feature | Description |
|---------|-------------|
| `spi1_lag1/2/3` | SPI-1 at t, t−1, t−2 (recent drought memory) |
| `spi3_lag1` | SPI-3 at t (medium-term drought signal) |
| `spi6_lag1` | SPI-6 at t (long-term accumulated deficit) |
| `pr_lag1/2/3` | Raw precipitation at t, t−1, t−2 |
| `month_sin`, `month_cos` | Cyclic encoding of the *target* month |

---

## Train / Validation / Test Split

| Split | Years | Role |
|-------|-------|------|
| Train | 1991–2016 | Model fitting, climatological baseline estimation |
| Validation | 2017–2020 | Hyperparameter selection, post-hoc calibration fitting |
| Test | 2021–2025 | Final evaluation (60 independent monthly maps, frozen) |

All splits are strict temporal — no shuffling, no future leakage.

---

## Models

Five forecasters were trained and compared at the same feature set:

| Model | Architecture | Notes |
|-------|-------------|-------|
| **Logistic Regression** | Linear, `liblinear` solver | Linear baseline |
| **Random Forest** | 300 trees, balanced class weights | Ensemble, interpretable feature ranking |
| **XGBoost** | Gradient boosting, GPU, balanced weights | Primary model |
| **XGBoost-Spatial** | XGBoost + 3×3 neighbourhood mean features | Adds local spatial context |
| **ConvLSTM** | Spatiotemporal deep learning (2D+time) | Spatial architecture baseline |

Three naive baselines are included as the scientific lower bound:
1. **Climatological** — per-calendar-month class frequencies from training (1991–2016)
2. **Persistence** — predict next month = current month's SPI-1 class
3. **SPI-1 heuristic** — convert current SPI-1 continuously to class probabilities (linear mapping)

---

## Evaluation Methodology

All **primary** metrics are computed at the **monthly level** (60 independent test months, 2021–2025), not the pixel level. Each monthly map contains ~7,200 spatially autocorrelated pixels; treating them as independent samples would inflate significance by a factor of ~100× relative to the true degrees of freedom.

**Metrics:**
- **Brier Score (BS)** — mean squared probability error; lower is better
- **Brier Skill Score (BSS)** — relative improvement over climatological BS; BSS > 0 means the model beats climatology
- **Heidke Skill Score (HSS)** — categorical skill accounting for class frequency
- **ROC-AUC** — ranking skill for dry vs. not-dry (supplementary)
- **Murphy BS decomposition** — splits BS into *reliability* + *resolution* − *uncertainty* to distinguish over-confidence from discriminating power
- **Bootstrap 95% CI** (2000 resamples, monthly block) for all BS/BSS/HSS scores
- **Paired bootstrap significance** — two-sided p-value for model vs. climatology and XGB-Spatial vs. XGB

**Post-hoc calibration study** (no test leakage):
- Three calibrators (*uncalibrated*, *Platt scaling*, *isotonic regression*) are fitted on validation pixels only
- Best method is selected by validation monthly BS, then applied to the frozen test set
- Calibration improves probability sharpness (reliability); the calibrated test results are reported alongside raw skill scores

---

## Primary Skill Results (test set 2021–2025, 60 months)

> Run `scripts/evaluate_forecast_skill.py` after training all models to populate this table.
> BSS > 0 means the model beats the climatological baseline on the dry class.

| Forecaster | BS (dry) | BSS (dry) | BSS 95% CI | HSS (3-class) | ROC-AUC |
|------------|----------|-----------|------------|----------------|---------|
| Climatological baseline | — | 0.000 (ref) | — | — | — |
| Persistence baseline | *(run scripts)* | *(run scripts)* | *(run scripts)* | *(run scripts)* | *(run scripts)* |
| SPI-1 heuristic | *(run scripts)* | *(run scripts)* | *(run scripts)* | *(run scripts)* | *(run scripts)* |
| Logistic Regression | *(run scripts)* | *(run scripts)* | *(run scripts)* | *(run scripts)* | *(run scripts)* |
| Random Forest | *(run scripts)* | *(run scripts)* | *(run scripts)* | *(run scripts)* | *(run scripts)* |
| **XGBoost** | *(run scripts)* | *(run scripts)* | *(run scripts)* | *(run scripts)* | *(run scripts)* |
| **XGBoost-Spatial** | *(run scripts)* | *(run scripts)* | *(run scripts)* | *(run scripts)* | *(run scripts)* |

Additional calibration-study outputs (selection by validation-set BS, decomposition, significance tests) are written to `outputs/calib_study_results.csv`.

---

## What the Model Learned (SHAP Interpretation)

SHAP TreeExplainer was applied to the XGBoost model to attribute the dry-class probability to each feature.

**Key findings:**
- `spi1_lag1` and `spi3_lag1` jointly dominate — the model relies heavily on recent and medium-term drought memory, which is hydrologically intuitive (soil moisture has multi-month persistence).
- `pr_lag1` and `pr_lag2` contribute a secondary signal: raw precipitation reinforces the SPI signal but adds complementary information on absolute moisture supply.
- Seasonal features (`month_sin`, `month_cos`) modulate drought probability near climatological dry/wet transitions (late autumn, early spring).
- The SPI→drought-probability relationship is strongly non-linear near SPI ≈ −1: small additional deficits sharply push the model into high-confidence dry predictions. This matches the known threshold behaviour of SPI-based drought classification.
- `spi6_lag1` contributes mainly for extended dry events (2021–2022), where long-term accumulation deficit reinforced near-term signals — consistent with the multi-year character of that drought.

---

## Limitations and Honest Assessment

- **BSS context:** At 1-month ahead, SPI-1 predictability from precipitation persistence is inherently limited. The bootstrap CI around BSS quantifies whether any positive skill is distinguishable from sampling noise given 60 test months.
- **Small test sample:** 60 months is a small sample for tight confidence intervals. Any claim of statistical significance should be interpreted cautiously.
- **USDM comparison** (`validate_usdm.py`) is a qualitative plausibility check only — USDM integrates soil moisture, streamflow, and observer reports, making it incommensurable with SPI-1.
- **ERA5-Land cross-validation** (`validate_era5_spi.py`) uses the same gamma-SPI methodology over the same domain with an independent precipitation source, giving the closest available out-of-source generalisation estimate.
- Spatial features (XGBoost-Spatial) and the ConvLSTM add neighbourhood information but do not introduce external predictors (sea-surface temperatures, soil moisture, snow cover). Adding such predictors is the highest-value next step for genuine skill improvement.

---

## Recommended Next Steps (ranked)

| Priority | Action | Scientific Benefit | Feasibility |
|----------|--------|-------------------|-------------|
| 1 | Add SST / PDO / ENSO index as external predictors | Likely main source of monthly-scale skill beyond persistence | Medium — public NOAA indices |
| 2 | Increase test window (rolling origin or leave-one-year-out CV) | More stable BSS CI, detects inter-annual skill variation | Medium |
| 3 | Forecast SPI-3 target at 3-month lead | More predictable target; directly useful for seasonal outlook | Low — straightforward feature reuse |
| 4 | Platt-calibrated ensemble (LogReg + RF + XGB) | Reduces model-uncertainty contribution to BS | Low |
| 5 | Spatial deep learning with attention over grid domain | Captures non-local teleconnection signals | High — needs GPU + more data |

---

## Reproducing Results

### Prerequisites

```bash
conda env create -f environment.yml
conda activate chirps-drought
```

### Pipeline (in order)

```bash
# 1. Download and preprocess data
bash scripts/download_chirps_v3_monthly_1991_2025.sh
python scripts/clip_to_cvalley_monthly.py
python scripts/make_spi_labels.py

# 2. Build forecast dataset
python scripts/build_dataset_forecast.py

# 3. Train models
python scripts/train_forecast_logreg.py
python scripts/train_forecast_rf.py
python scripts/train_forecast_xgboost.py
python scripts/train_forecast_xgb_spatial.py   # optional; adds spatial features
python scripts/train_forecast_convlstm.py       # optional; GPU recommended

# 4. Evaluate
python scripts/evaluate_forecast_skill.py       # main skill table + calibration study
python scripts/xgb_shap_forecast_analysis.py    # SHAP interpretation
python scripts/validate_era5_spi.py             # cross-dataset validation
python scripts/plot_spatial_skill.py            # per-pixel skill map
python scripts/plot_case_study.py               # 2021–2022 drought case study
```

### Output files

| File | Description |
|------|-------------|
| `outputs/forecast_skill_bss_hss_table.csv` | Main skill table (all models, with 95% CI) |
| `outputs/forecast_skill_scores.txt` | Human-readable summary |
| `outputs/calib_study_results.csv` | Calibration study: BS decomposition, CI, p-values |
| `outputs/calib_study_reliability_diagram.png` | Reliability diagram (calibrated models) |
| `outputs/calib_study_decomposition_barplot.png` | Murphy BS decomposition barplot |
| `outputs/forecast_reliability_diagram.png` | Raw vs. isotonic-calibrated reliability (pixel level) |
| `outputs/forecast_monthly_cm.png` | Monthly confusion matrix (XGBoost) |
| `outputs/spatial_skill_accuracy.png` | Per-pixel accuracy map |
| `outputs/case_study_2021_2025.png` | Temporal case study 2021–2025 |

---

## Acknowledgement

AI tools (ChatGPT, Gemini) were used to assist in code development and documentation.

---

## Author

Md Ishtiaque Hossain
MSc Candidate, Computer and Information Sciences, University of Delaware
[LinkedIn](https://linkedin.com/in/ishtiaque-h) | [GitHub](https://github.com/Ishtiaque-h)
