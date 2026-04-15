# Central Valley Drought Classifier

**Can machine learning predict next month's drought from satellite precipitation alone?**

We build a leakage-free forecasting pipeline to predict monthly drought classes (*Dry / Normal / Wet*) in California's Central Valley using [CHIRPS v3.0](https://www.chc.ucsb.edu/data/chirps3) satellite precipitation and WMO-standard SPI. Every metric is evaluated at the monthly level — 63 independent test months (2021–2026) — against three naive baselines, with bootstrap confidence intervals.

**Key finding:** No model — from logistic regression to ConvLSTM — outperforms climatology in Brier Skill Score, even though ranking signal exists (ROC-AUC ~0.68). This is not a model failure; it is a **predictability barrier**. Monthly SPI-1 in a Mediterranean climate is fundamentally driven by chaotic synoptic events (atmospheric rivers, frontal passages) at 1-month lead, making the base rate the best available probability estimate.

> Full research assessment, literature context, and strategic roadmap: [`ANALYSIS.md`](ANALYSIS.md)

---

## Problem Statement

Predicting drought 1 month ahead using only past precipitation is a hard problem. The target — SPI-1 at time *t+1* — depends entirely on precipitation at *t+1*, which is unknown. This makes the problem fundamentally different from *reconstruction* (classifying the current month) and sets a high bar for ML to add value beyond climatological base rates.


## Why This Matters

- **Agricultural stakes are real.** Central Valley (35.4°–40.6°N, 122.5°–119.0°W) produces roughly 25% of the U.S. food supply; drought losses reach billions per year. [`[USGS]`](https://ca.water.usgs.gov/projects/central-valley/about-central-valley.html#:~:text=Using%20fewer%20than%201%25%20of,nuts%2C%20and%20other%20table%20foods.) One-month-ahead early warning enables pre-season irrigation scheduling and crop-stress mitigation.
- **Most published drought ML overstates skill.** Label leakage through SPI accumulation windows, spatial pseudo-replication, and missing baselines inflate reported accuracy. This project eliminates all three — and shows what remains. 
- **Negative results have scientific value.** Demonstrating a predictability ceiling with rigorous evidence helps the community redirect effort toward problems where ML *can* add value (e.g., seasonal horizons, exogenous climate drivers). Operational drought early-warning relies on understanding what is and isn't predictable.

---

## Pipeline

```mermaid
graph TD;
    A["CHIRPS v3.0 Monthly (1991-2025)"] --> B["Clip to Central Valley"];
    B --> C["Gamma-fit SPI per pixel × calendar month"];
    C --> D["SPI-1 / SPI-3 / SPI-6"];
    D --> E["Target: SPI-1 drought class at t+1"];
    D --> F["Features: SPI lags + pr lags + month"];
    E --> G["Forecast dataset (zero leakage)"];
    F --> G;
    G --> H["Model suite: LogReg / RF / XGBoost / XGBoost-Spatial / ConvLSTM"];
    H --> I["Skill evaluation (monthly BSS / HSS, bootstrap CI)"];
    H --> J["SHAP explainability"];
    H --> K["Cross-dataset validation (ERA5-Land)"];
    H --> L["Spatial skill maps / Case studies"];
```

**Data:** [CHIRPS v3.0](https://www.chc.ucsb.edu/data/chirps3) — 0.05° (~5 km), monthly, 1991–2026, ~7,200 pixels over Central Valley.

**Temporal split (strictly chronological, no shuffling):**

| Split | Period | Purpose |
|-------|--------|---------|
| Train | 1991–2016 | 26 years of model learning |
| Validation | 2017–2020 | Hyperparameter selection + post-hoc calibration fitting |
| Test | 2021–2026 | 63 months, never seen during training or calibration |

### Features and Target

**Forecast task:** given the precipitation and SPI history through month *t*, predict whether month *t+1* will be *dry* (SPI-1 ≤ −1), *normal*, or *wet* (SPI-1 ≥ +1).

| Feature | What it captures |
|---------|-----------------|
| `spi1_lag1/2/3` | Recent drought state (1-month SPI memory at t, t−1, t−2) |
| `spi3_lag1` | Medium-term precipitation accumulation |
| `spi6_lag1` | Longer-term hydrological drought context |
| `pr_lag1/2/3` | Raw precipitation absolute magnitude (SPI is relative to climatology) |
| `month_sin`, `month_cos` | Cyclic encoding of the *target* month |

**Leakage-free target design:** The target is `SPI-1[t+1]`, which depends *only* on `pr[t+1]` — unknown at prediction time. All features are derived from time *t* or earlier. There is **zero accumulation-window overlap** between features and target. This eliminates the data leakage present in SPI-3-based targets (where 2 of 3 accumulation months overlap with features).

**However**, the feature set is purely endogenous: every feature derives from the same CHIRPS precipitation used to define the target. No exogenous climate driver (ENSO, PDO), no temperature/VPD, no soil moisture, no teleconnection index is included. This is both a limitation and a controlled experimental choice — it isolates the question: can local precipitation history predict next month's drought class?

---

### Models

| Model | Architecture | Key Design Choice  | Hypothesis |
|-------|--------------|--------------------|------------|
| **Logistic Regression** | Linear, `lbfgs` solver | StandardScaler, balanced class weights | Can linear relationships between lagged SPI/pr and next-month drought explain the variance? |
| **Random Forest** | Bagged trees (500 trees, max_depth=20 ) | balanced_subsample weighting | Do nonlinear feature interactions (thresholds, conditional dependencies) improve discrimination? |
| **XGBoost** | Gradient-boosted trees (hist, early-stopped) | balanced sample weights, multi:softprob | Does boosting extract additional weak signals from noisy tabular data? |
| **XGBoost-Spatial** | XGBoost + 3×3 neighbourhood mean features | Adds 4 local spatial context features (spi1/3/6, pr) | Does local spatial context (3×3 mean) add information beyond the pixel-level features? |
| **ConvLSTM** | 2-layer ConvLSTM → 1×1 Conv head | Spatiotemporal deep learning; 4 channels × 3 months | Can learned spatiotemporal filters (3-month sequences × 2D spatial) discover patterns invisible to tabular models? |

Three **naive baselines** set the scientific lower bound:
1. **Climatological** — per-calendar-month class frequencies from training (1991–2016)
2. **Persistence** — predict next month = current month's SPI-1 class
3. **SPI-1 heuristic** — map current SPI-1 continuously to class probabilities

---

## Evaluation

All primary metrics are computed at the **monthly level** (63 independent test months), not the pixel level. Each monthly map contains ~7,200 spatially autocorrelated pixels; treating them as independent samples would inflate significance by ~100×.

- **Brier Score (BS)** — mean squared probability error; lower is better
- **Brier Skill Score (BSS)** — relative improvement over climatological BS; BSS > 0 means the model beats climatology
- **Heidke Skill Score (HSS)** — categorical skill accounting for class frequency
- **ROC-AUC** — ranking skill for dry vs. not-dry
- **Murphy BS decomposition** — splits BS into *reliability* + *resolution* − *uncertainty* to distinguish over-confidence from lack of discriminating power
- **Bootstrap 95% CI** (2,000 resamples, monthly block) for all scores
- **Paired bootstrap significance** — two-sided p-value for model vs. climatology

**Post-hoc calibration study** (no test leakage): three calibrators (uncalibrated, Platt scaling, isotonic regression) are fitted on validation pixels only. The best method by validation BS is applied to the frozen test set.

---

## Results

### Skill Scores (63 test months, monthly level)

| Forecaster | Brier Score (dry) | BSS vs. climatology | HSS (3-class) | ROC-AUC (dry) |
|---|---|---|---|---|
| **Climatological baseline** | **0.0646** (ref) | 0.0 | 0.00 | — |
| Persistence | 0.1011 | −0.57 | 0.09 | 0.56 |
| SPI-1 heuristic | 0.0949 | −0.47 | 0.09 | 0.56 |
| Logistic Regression | 0.0874 | −0.35 | 0.15 | 0.81 |
| Random Forest | 0.0820 | −0.27 | 0.11 | 0.60 |
| XGBoost | 0.0687 | −0.06 | 0.00 | 0.67 |
| **XGBoost-Spatial** | **0.0666** | **−0.03** | 0.00 | **0.68** |
| ConvLSTM | 0.0823 | −0.27 | 0.22 | 0.52 |

> BSS > 0 would mean the model beats climatology. No model crosses this threshold.
> XGBoost-Spatial comes closest (BSS = -0.03), with 95% CI spanning zero — indicating the difference from climatology is not statistically significant.

- **LogReg BSS = −0.35:** Linear relationships explain almost nothing beyond climatology.
- **RF BSS = −0.27:** Nonlinear interactions improve slightly, but not significantly.
- **XGBoost BSS = −0.06:** Boosting finds marginal additional signal.
- **XGBoost-Spatial BSS = −0.03:** Spatial context adds a tiny, statistically insignificant increment.
- **ConvLSTM BSS = −0.27:** Deep spatiotemporal learning does not help — it overfits with only ~300 training windows.

### What This Tells US

1. **Models detect drought signal.** ROC-AUC ~0.68 means the models rank months by drought risk better than chance — there *is* learnable structure in the features.
2. **But ranking ≠ calibrated probability.** The convergence of all five models to negative BSS is itself the key finding. It demonstrates that the models cannot produce probability estimates more reliable than the climatological base rate. The signal exists but is too weak to improve upon always predicting "25% chance of drought."
3. **The bottleneck is resolution, not reliability.** Brier Score decomposition (Murphy 1973) shows models achieve near-zero resolution — they cannot reliably distinguish drought months from non-drought months in advance. Post-hoc calibration (isotonic/Platt) was tested and confirmed: you cannot calibrate your way past a discrimination barrier.
4. **This is physically consistent.** SPI-1 autocorrelation in Central Valley is weak (r ≈ 0.1–0.3). Persistence fails badly (BSS = −0.57). Monthly precipitation is governed by atmospheric rivers and frontal passages — chaotic processes at 1-month lead.

### What the Model Learned (SHAP)

SHAP TreeExplainer on XGBoost attributes the dry-class probability to each feature:

- **`spi1_lag1` and `spi3_lag1` jointly dominate** — the model relies on recent and medium-term drought memory, which is hydrologically intuitive (soil moisture has multi-month persistence).
- **`pr_lag1`/`pr_lag2` provide a secondary signal** — raw precipitation reinforces SPI but adds complementary information on absolute moisture supply.
- **Seasonal features** (`month_sin`, `month_cos`) modulate drought probability near climatological dry/wet transitions (late autumn, early spring).
- **Nonlinear threshold near SPI ≈ −1** — small additional deficits sharply push the model into high-confidence dry predictions, matching the known threshold behaviour of SPI-based drought classification.
- **`spi6_lag1` contributes mainly for extended dry events** (2021–2022), where long-term accumulation deficit reinforced near-term signals — consistent with the multi-year character of that drought.

The model *understands* drought dynamics. It simply cannot overcome the chaotic nature of next-month precipitation.

---

## Evaluation and Validation

- **Three naive baselines** (climatology, persistence, SPI-1 heuristic) provide the necessary reference points
- **Bootstrap confidence intervals** (2,000 iterations) quantify uncertainty on all skill scores
- **Brier Score decomposition** pinpoints *where* models fail (resolution vs. reliability)
- **Post-hoc calibration study** with frozen test evaluation — no test leakage in the calibration pipeline
- **Spatial skill maps:** Per-pixel accuracy over 2021-2026, with Sacramento Valley (lat > 38 deg) and San Joaquin Valley (lat < 38 deg) sub-regions annotated.
- **Case study:** 2021-22 drought and 2023 atmospheric river events, showing model captures the correct directional signal.
- **Cross-dataset validation** `validate_era5_spi.py` computes SPI-1 from ERA5-Land precipitation using the same gamma-fit methodology, then compares model predictions against this independent product.
- **USDM comparison framed as qualitative** — `validate_usdm.py` overlays the model's dry fraction against USDM D1+ area. USDM integrates soil moisture and streamflow, so this is a plausibility check — not a skill metric. It avoids treating a composite drought index as ground truth for a precipitation-only forecast.

---

## Limitations and Assessment

- **BSS context:** At 1-month lead, SPI-1 predictability from precipitation persistence is inherently limited. The bootstrap CI quantifies whether any positive skill is distinguishable from sampling noise given 63 test months.
- **Small test sample:** 63 months is adequate for bootstrap-based inference but imposes wide confidence intervals. Any claim of significance should be interpreted cautiously.
- **Single region:** The predictability barrier may be specific to Central Valley's Mediterranean hydroclimate, or it may be universal. A single-region study cannot distinguish these.
- **Purely endogenous features:** All predictors derive from CHIRPS precipitation. Large-scale climate drivers (ENSO, PDO) and other variables (temperature, soil moisture) are not included — these represent the highest-value additions for genuine skill improvement.
- **Test period extremes:** 2021–2026 includes a historic drought followed by extreme wet reversal, which is not climatologically representative.

---

## Research Directions

Highest-impact next steps (see [`ANALYSIS.md`](ANALYSIS.md) for the full strategic roadmap):

| Priority | Direction | What it tests |
|----------|-----------|---------------|
| 1 | **Add ENSO / PDO as external predictors** | Can large-scale climate state break the predictability barrier? |
| 2 | **Expand to additional regions** | Does "no skill" generalize across hydroclimates? |
| 3 | **Stratified BSS by season and ENSO phase** | Is there conditional skill masked in overall averages? |
| 4 | **Seasonal target (SPI-3 at 3-month lead)** | Do longer aggregation windows improve predictability? |
| 5 | **Platt-calibrated ensemble** (LogReg + RF + XGB) | Can model averaging reduce the uncertainty component of BS? |

---

## Reproducing Results

### Prerequisites

```bash
conda env create -f environment.yml
conda activate chirps-drought
```

### Pipeline (in order)

```bash
# 1. Download and preprocess
bash scripts/download_chirps_v3_monthly_1991_2025.sh
python scripts/clip_to_cvalley_monthly.py
python scripts/make_spi_labels.py

# 2. Build forecast dataset
python scripts/build_dataset_forecast.py

# 3. Train models
python scripts/train_forecast_logreg.py
python scripts/train_forecast_rf.py
python scripts/train_forecast_xgboost.py
python scripts/train_forecast_xgb_spatial.py    # adds 3x3 neighbourhood features
python scripts/train_forecast_convlstm.py        # optional, GPU recommended

# 4. Evaluate and interpret
python scripts/evaluate_forecast_skill.py        # skill table + calibration study
python scripts/xgb_shap_forecast_analysis.py     # SHAP interpretation
python scripts/validate_era5_spi.py              # cross-dataset validation
python scripts/validate_usdm.py                  # USDM plausibility check
python scripts/plot_spatial_skill.py             # per-pixel skill map
python scripts/plot_case_study.py                # 2021-2025 case study
```

### Key output files

| File | Description |
|------|-------------|
| `outputs/forecast_skill_bss_hss_table.csv` | Main skill table (all models, with 95% CI) |
| `outputs/calib_study_results.csv` | Calibration study: BS decomposition, CI, p-values |
| `outputs/calib_study_reliability_diagram.png` | Reliability diagram (calibrated models) |
| `outputs/calib_study_decomposition_barplot.png` | Murphy BS decomposition barplot |
| `outputs/spatial_skill_accuracy.png` | Per-pixel accuracy map |
| `outputs/case_study_2021_2026.png` | Temporal case study 2021–2026 |

---

## Acknowledgement

AI tools (ChatGPT, Gemini, GitHub Copilot) were used ethically for project design, code development, and documentation.

---

## Author

Md Ishtiaque Hossain \
MSc, Computer and Information Sciences \
University of Delaware \
[LinkedIn](https://linkedin.com/in/ishtiaque-h) · [GitHub](https://github.com/Ishtiaque-h)