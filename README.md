# Central Valley Drought Classifier

**Can machine learning predict next month's drought from satellite precipitation alone?**

We build a leakage-free forecasting pipeline to predict monthly drought classes (*Dry / Normal / Wet*) in California's Central Valley using [CHIRPS v3.0](https://www.chc.ucsb.edu/data/chirps3) satellite precipitation and WMO-standard SPI. Every metric is evaluated at the monthly level — 63 independent test months (2021–2026) — against three naive baselines, with bootstrap confidence intervals.

**Key finding:** No model — from logistic regression to ConvLSTM — outperforms climatology in Brier Skill Score, even though ranking signal exists (LogReg ROC-AUC = 0.80). This is not a model failure; it is a **predictability barrier**. Monthly SPI-1 in a Mediterranean climate is fundamentally driven by chaotic synoptic events (atmospheric rivers, frontal passages) at 1-month lead, making the base rate the best available probability estimate.

> Full research assessment, literature context, and strategic roadmap: [`ANALYSIS.md`](ANALYSIS.md)

---

## Problem Statement

Predicting drought 1 month ahead using only past precipitation is a hard problem. The target — SPI-1 at time *t+1* — depends entirely on precipitation at *t+1*, which is unknown. This makes the problem fundamentally different from *reconstruction* (classifying the current month) and sets a high bar for ML to add value beyond climatological base rates.


## Why This Matters

- **Agricultural stakes are real.** Central Valley (35.4°–40.6°N, 122.5°–119.0°W) produces roughly 25% of the U.S. food supply; drought losses reach billions per year. [`[USGS]`](https://ca.water.usgs.gov/projects/central-valley/about-central-valley.html) One-month-ahead early warning enables pre-season irrigation scheduling and crop-stress mitigation.
- **Most published drought ML overstates skill.** Label leakage through SPI accumulation windows, spatial pseudo-replication, and missing baselines inflate reported accuracy. This project eliminates all three — and shows what remains. 
- **Negative results have scientific value.** Demonstrating a predictability ceiling with rigorous evidence helps the community redirect effort toward problems where ML *can* add value (e.g., seasonal horizons, exogenous climate drivers). Operational drought early-warning relies on understanding what is and isn't predictable.

---

## Pipeline

```mermaid
graph TD;
    A["CHIRPS v3.0 Monthly (1991-2026)"] --> B["Clip to Central Valley"];
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
| `nino34_lag1/2` *(optional)* | ENSO Niño3.4 climate-state signal at feature month and previous month |
| `pdo_lag1/2` *(optional)* | Pacific Decadal Oscillation state at feature month and previous month |

**Leakage-free target design:** The target is `SPI-1[t+1]`, which depends *only* on `pr[t+1]` — unknown at prediction time. All features are derived from time *t* or earlier. There is **zero accumulation-window overlap** between features and target. This eliminates the data leakage present in SPI-3-based targets (where 2 of 3 accumulation months overlap with features).

By default the pipeline runs with endogenous CHIRPS-only features.  
If `data/processed/climate_indices_monthly.csv` exists, ENSO/PDO lag features are added automatically (still leakage-safe because all features are at time *t* or earlier).

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
1. **Climatological** — per-calendar-month class frequencies from training (1991–2016); always the same probability for each month regardless of recent precipitation
2. **Persistence** — predict next month = current month's SPI-1 class; assumes drought state does not change
3. **SPI-1 threshold** — map current SPI-1 continuously to class probabilities via sigmoid; exploits the fact that current drought state correlates with next month (weak autocorrelation)

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

**Post-hoc calibration study** (no test leakage): After model training, three calibration methods are tested — (1) no calibration, (2) Platt scaling (logistic recalibration), (3) isotonic regression (nonparametric recalibration) — to adjust predicted probabilities to match observed frequencies. Calibrators are fitted on validation set only; the best-performing method is applied to the frozen test set to assess whether probability estimates can be improved without retraining the core model.

---

## Results

### Skill Scores (63 test months, monthly level)

| Forecaster | Brier Score (dry) | BSS vs. climatology | HSS (3-class) | ROC-AUC (dry) |
|---|---|---|---|---|
| **Climatological baseline** | **0.0643** (ref) | 0.0 | 0.00 | — |
| Persistence | 0.1018 | −0.58 | 0.08 | 0.51 |
| SPI-1 threshold | 0.0960 | −0.49 | 0.08 | 0.51 |
| Logistic Regression | 0.0895 | −0.39 | 0.15 | 0.80 |
| Random Forest | 0.0807 | −0.26 | 0.10 | 0.61 |
| XGBoost (no spatial) | 0.0836 | −0.30 | 0.14 | 0.72 |
| **XGBoost-Spatial** | **0.0667** | **−0.04** | 0.00 | **0.70** |
| ConvLSTM | 0.1023 | −0.59 | 0.06 | 0.41 |

> BSS > 0 would mean the model beats climatology. No model crosses this threshold.
> XGBoost-Spatial comes closest (BSS = −0.04), with 95% CI spanning zero — indicating the difference from climatology is not statistically significant. Full confidence intervals in [results/report/forecast_skill_bss_hss_table.csv](results/report/forecast_skill_bss_hss_table.csv).

- **LogReg BSS = −0.39:** Linear relationships explain almost nothing beyond climatology.
- **RF BSS = −0.26:** Nonlinear interactions improve slightly, but not significantly.
- **XGBoost BSS = −0.30:** Boosting finds marginal additional signal.
- **XGBoost-Spatial BSS = −0.04:** Spatial context adds a tiny, statistically insignificant increment.
- **ConvLSTM BSS = −0.59:** Deep spatiotemporal learning underperforms — likely due to overfitting with limited training windows (~300).

### What This Tells Us

1. **Models detect drought signal.** ROC-AUC ~0.70 means the models rank months by drought risk better than chance — there *is* learnable structure in the features.
2. **But ranking ≠ calibrated probability.** All five models converge to negative BSS — the key finding. They cannot produce probability estimates more reliable than climatology. The signal is too weak to beat "always predict 25% chance of drought."
3. **The bottleneck is resolution, not reliability.** Brier Score decomposition (Murphy 1973) shows near-zero resolution — models cannot distinguish drought months in advance. Post-hoc calibration (isotonic/Platt) was tested and confirmed: you cannot calibrate past a discrimination barrier.
4. **This is physically consistent.** SPI-1 autocorrelation in Central Valley is weak (r ≈ 0.1–0.3). Monthly precipitation is dominated by chaotic synoptic events (atmospheric rivers) at 1-month lead.

### What the Model Learned (SHAP)

SHAP TreeExplainer on XGBoost attributes the dry-class probability to each feature:

- **`spi1_lag1` and `spi3_lag1` jointly dominate** — the model relies on recent and medium-term drought memory, which is hydrologically intuitive (soil moisture has multi-month persistence).
- **`pr_lag1`/`pr_lag2` provide a secondary signal** — raw precipitation reinforces SPI but adds complementary information on absolute moisture supply.
- **Seasonal features** (`month_sin`, `month_cos`) modulate drought probability near climatological dry/wet transitions (late autumn, early spring).
- **Nonlinear threshold near SPI ≈ −1** — small additional deficits sharply push the model into high-confidence dry predictions, matching the known threshold behaviour of SPI-based drought classification.
- **`spi6_lag1` contributes mainly for extended dry events** (2021–2022), where long-term accumulation deficit reinforced near-term signals — consistent with the multi-year character of that drought.

The model *understands* drought dynamics. It simply cannot overcome the chaotic nature of next-month precipitation.

### Regional Forecast Evaluation

Beyond pixel-level skill, we evaluate the model's ability to predict the **dominant drought class at the Central Valley scale**. For each month, we compute the fraction of pixels predicted as dry/normal/wet and compare the dominant class to observations. See [results/regional/](results/regional/) for dominant-class accuracy and class fraction time series.

---

## Evaluation Methods

- **Three naive baselines** — climatology, persistence, SPI-1 threshold
- **Bootstrap confidence intervals** (2,000 iterations) on all skill scores
- **Brier Score decomposition** (Murphy 1973) — identifies whether failures are due to reliability or resolution
- **Post-hoc calibration** — Platt scaling and isotonic regression tested on validation set, applied to frozen test set
- **Stratified skill diagnostics** — season-wise BSS and ENSO-phase BSS tables
- **Spatial skill maps** — per-pixel accuracy over 2021-2026
- **Case studies** — 2021–22 drought and 2023 atmospheric rivers
- **Cross-dataset validation** — ERA5-Land SPI-1 comparison
- **Qualitative validation** — USDM D1+ plausibility check (not a metric)

---

## Limitations

- **1-month lead is fundamentally hard:** Monthly precipitation is dominated by chaotic synoptic weather; SPI-1 autocorrelation is weak.
- **Small test set:** 63 months yields wide confidence intervals; positive skill claims require CI spanning zero.
- **Single region:** Cannot generalize to other hydroclimates without regional expansion.
- **Endogenous features only:** No ENSO, PDO, temperature, or soil moisture. These are the highest-value next additions.
- **Test period non-representative:** 2021–2026 is extreme (historic drought → extreme wet).

---

## Next Steps

Highest-impact directions (see [`ANALYSIS.md`](ANALYSIS.md) for full roadmap):

1. **Add ENSO / PDO** — Can large-scale climate state break the predictability barrier?
2. **Expand to other regions** — Does the barrier generalize across hydroclimates?
3. **Seasonal target (SPI-3, 3-month lead)** — Do longer windows improve predictability?
4. **Temperature + VPD** — Does evaporative demand help?
5. **Conditional skill** — Is there signal hidden in season or ENSO phase?

---

## Reproducing Results

### Prerequisites

```bash
conda env create -f environment.yml
conda activate drought-classifier
```

### Pipeline (in order)

```bash
# 1. Download and preprocess
bash scripts/download_chirps_v3_monthly.sh
python scripts/clip_to_cvalley_monthly.py
python scripts/make_spi_labels.py
python scripts/download_climate_indices.py   # optional: creates ENSO/PDO monthly file

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
python scripts/evaluate_regional_forecast.py     # regional (Central Valley) dominant class accuracy
python scripts/xgb_shap_forecast_analysis.py     # SHAP interpretation
python scripts/validate_era5_spi.py              # cross-dataset validation
python scripts/validate_usdm.py                  # USDM plausibility check
python scripts/plot_spatial_skill.py             # per-pixel skill map
python scripts/plot_case_study.py                # 2021-2026 case study
```

### Results (reproduced outputs)

**Key results are saved to the `results/` folder by category:**

- **[results/report/](results/report/)** — Main skill table, calibration study, reliability diagrams, case study
- **[results/report/](results/report/)** — Includes season/ENSO-stratified BSS CSV tables
- **[results/spatial/](results/spatial/)** — Per-pixel accuracy maps
- **[results/xgboost/](results/xgboost/)** — XGBoost feature importance, confusion matrix, SHAP interpretation
- **[results/validation/](results/validation/)** — ERA5-Land and USDM cross-dataset validation
- **[results/regional/](results/regional/)** — Regional (Central Valley) forecast evaluation

---

## Acknowledgement

AI tools (ChatGPT, Gemini, GitHub Copilot) were used ethically for research & analysis, code review, and documentation.

---

## Author

Md Ishtiaque Hossain \
MSc, Computer and Information Sciences \
University of Delaware \
[LinkedIn](https://linkedin.com/in/ishtiaque-h) · [GitHub](https://github.com/Ishtiaque-h)
