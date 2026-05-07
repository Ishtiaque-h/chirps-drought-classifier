# Central Valley Drought Classifier

**Can machine learning predict next month's drought from satellite precipitation alone?**

We build a leakage-free forecasting pipeline to predict monthly drought classes (*Dry / Normal / Wet*) in California's Central Valley using [CHIRPS v3.0](https://www.chc.ucsb.edu/data/chirps3) satellite precipitation and WMO-standard SPI. Every metric is evaluated at the monthly level — 63 independent test months (2021–2026) — against three naive baselines, with bootstrap confidence intervals.

**Key finding:** Corrected ENSO-anomaly features and local spatial context bring XGBoost-Spatial almost exactly to climatology (calibrated BSS = +0.005), but the confidence interval still crosses zero. Ranking signal is real (XGBoost-Spatial ROC-AUC = 0.74), yet reliable probability skill remains statistically indistinguishable from the climatological base rate. This is not just a model-capacity problem; it points to a **predictability barrier** for 1-month-ahead SPI-1 in California's Mediterranean hydroclimate.

> Full research assessment, literature context, and strategic roadmap: [`ANALYSIS.md`](ANALYSIS.md)
> Current narrative synthesis and manuscript strategy: [`final_report.md`](final_report.md)
> Documentation map and current source-of-truth guide: [`docs/README.md`](docs/README.md)

---

## Problem Statement

Predicting drought 1 month ahead using only past precipitation is a hard problem. The target — SPI-1 at time *t+1* — depends entirely on precipitation at *t+1*, which is unknown. This makes the problem fundamentally different from *reconstruction* (classifying the current month) and sets a high bar for ML to add value beyond climatological base rates.


## Why This Matters

- **Agricultural stakes are real.** Central Valley (35.4°–40.6°N, 122.5°–119.0°W) produces roughly 25% of the U.S. food supply; drought losses reach billions per year. [`[USGS]`](https://ca.water.usgs.gov/projects/central-valley/about-central-valley.html) One-month-ahead early warning enables pre-season irrigation scheduling and crop-stress mitigation.
- **Drought-ML skill is often hard to compare across protocols.** SPI accumulation-window overlap, spatial pseudo-replication, and missing climatology baselines can inflate apparent skill. This project audits those risks directly and shows what remains under the strict monthly protocol.
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
    H --> K["Cross-dataset validation (ERA5-Land / PRISM)"];
    H --> L["Spatial skill maps / Case studies"];
```

**Data:** [CHIRPS v3.0](https://www.chc.ucsb.edu/data/chirps3) — 0.05° (~5 km), monthly, 1991–2026, ~7,200 pixels over Central Valley.

Primary manuscript citations to carry forward: CHIRPS v3
([Funk et al., 2026](https://doi.org/10.1038/s41597-026-07096-4)),
the original CHIRPS record
([Funk et al., 2015](https://doi.org/10.1038/sdata.2015.66)),
SPI guidance
([WMO-No. 1090](https://library.wmo.int/idurl/4/39629)),
and Murphy's Brier Score decomposition
([Murphy, 1973](https://ui.adsabs.harvard.edu/abs/1973JApMe..12..595M/abstract)).
Independent U.S. precipitation validation uses PRISM monthly precipitation from
the PRISM Climate Group at Oregon State University
([official data portal](https://prism.oregonstate.edu/?id=US);
[Daly et al., 2008](https://doi.org/10.1002/joc.1688)).

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
| `nino34_lag1/2` *(active corrected checkpoint)* | ENSO Niño3.4 monthly anomaly at feature month and previous month |
| `pdo_lag1/2` *(optional, not active checkpoint)* | Pacific Decadal Oscillation state at feature month and previous month |

**Leakage-free target design:** The target is `SPI-1[t+1]`, which depends *only* on `pr[t+1]` — unknown at prediction time. All features are derived from time *t* or earlier. There is **zero accumulation-window overlap** between features and target. The evaluation-inflation audit separately shows how an overlapping SPI-3 lead-1 target can create apparent skill from shared accumulation months.

By default the pipeline can run with endogenous CHIRPS-only features. The current corrected checkpoint uses Niño3.4 anomaly lags (`--climate-features nino34`) and excludes PDO because the NOAA PDO file has missing recent months after August 2025. Climate features remain leakage-safe because all model features are at time *t* or earlier.

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

Paper-ready summary tables are now generated from the current artifacts:

```bash
python scripts/generate_master_results.py
```

The consolidated outputs are
[results/report/master_results_table.csv](results/report/master_results_table.csv)
and the compact manuscript table
[results/report/master_results_headline.csv](results/report/master_results_headline.csv).
For the canonical SPI-1 precipitation target, the table contains **no robust
positive BSS result**: all positive SPI point estimates have confidence
intervals crossing zero. The separate CFSv2 root-zone soil-moisture benchmark
is the first robust-positive land-surface target result and is reported below.

### Skill Scores (63 test months, monthly level)

Current corrected checkpoint: Niño3.4 anomaly lags + all tabular baselines retrained on the corrected feature schema. XGBoost-Spatial is the strongest raw ML model by ranking skill; Random Forest has the lowest raw Brier Score among uncalibrated ML models, but still does not beat climatology.

| Forecaster | Brier Score (dry) | BSS vs. climatology | HSS (3-class) | ROC-AUC (dry) |
|---|---|---|---|---|
| **Climatological baseline** | **0.0643** (ref) | 0.0 | 0.00 | — |
| Persistence | 0.1018 | −0.58 | 0.08 | 0.51 |
| SPI-1 threshold | 0.0960 | −0.49 | 0.08 | 0.51 |
| Logistic Regression | 0.0893 | −0.39 | 0.12 | 0.84 |
| Random Forest | 0.0686 | −0.07 | −0.04 | 0.66 |
| XGBoost (ENSO-only) | 0.0817 | −0.27 | 0.08 | 0.68 |
| **XGBoost-Spatial (ENSO-only)** | **0.0717** | **−0.12** | 0.04 | **0.74** |
| ConvLSTM *(corrected target)* | 0.0872 | −0.36 | 0.09 | 0.59 |

> Raw BSS > 0 would mean the model beats climatology. No raw model crosses this threshold.
> Post-hoc isotonic calibration brings XGBoost-Spatial to **BSS = +0.005** (95% CI [−0.062, +0.073]), which is statistically indistinguishable from climatology. Manuscript-facing intervals are consolidated in [results/report/paper/table02_headline_results.csv](results/report/paper/table02_headline_results.csv).

- **XGBoost BSS = −0.27:** Corrected Niño3.4 anomalies help relative to the contaminated ENSO/PDO run, but do not beat climatology.
- **XGBoost-Spatial raw BSS = −0.12:** Spatial neighborhood features improve over non-spatial XGBoost and raise ROC-AUC to 0.74.
- **Random Forest raw BSS = −0.07:** The strongest raw Brier Score among uncalibrated ML models is still below climatology.
- **XGBoost-Spatial calibrated BSS = +0.005:** The best current model is effectively tied with climatology; the positive point estimate is not significant.
- **ConvLSTM corrected BSS = −0.36:** The target-aligned ConvLSTM improves over the stale artifact but still underperforms XGBoost-Spatial and climatology.

### What This Tells Us

1. **Models detect drought signal.** ROC-AUC = 0.74 for XGBoost-Spatial means the model ranks dry-risk months better than chance.
2. **But ranking ≠ calibrated probability.** The best calibrated BSS is +0.005 with a confidence interval crossing zero. This is a practical tie with climatology, not a positive-skill result.
3. **Spatial context helps, but only marginally.** Spatial features reduce Brier Score and improve ranking, but do not yet produce statistically reliable probability skill.
4. **This is physically consistent.** SPI-1 autocorrelation in Central Valley is weak (r ≈ 0.1–0.3). Monthly precipitation is dominated by chaotic synoptic events (atmospheric rivers) at 1-month lead.

### What the Model Learned

Current corrected XGBoost-Spatial SHAP importance is dominated by the corrected climate and seasonal terms, with spatial neighborhood features adding secondary information:

- **`nino34_lag1/2` and `month_sin/cos` dominate SHAP importance** — large-scale climate state and target-month seasonality carry the strongest split signal.
- **Spatial neighborhood means add useful ranking information** — `spi1_nbr_mean`, `spi3_nbr_mean`, `spi6_nbr_mean`, and `pr_nbr_mean` improve ROC-AUC and reduce Brier error relative to non-spatial XGBoost.
- **Precipitation lags remain useful in ablation** — removing `pr_lag1/2/3` worsens raw XGB BSS, even though the tree-gain ranking emphasizes climate/season terms.
- **Correlated feature groups complicate interpretation** — ablation shows removing ENSO or SPI lags can improve the trained non-spatial XGB BSS, so feature importance should be read as model behavior, not causal attribution.

The model captures real structure, but the structure is not strong enough to produce statistically reliable probability skill over climatology.

### Seasonal Target Experiments

Leakage-free seasonal experiments now test longer accumulation targets with
`lead >= SPI window`. For SPI-3 lead-3, features at month `t` predict the
SPI-3 class ending at `t+3`, so the target accumulation window is
`pr[t+1..t+3]` with no overlap.

| Seasonal target | Forecaster | Dry BS | BSS vs. climatology | 95% CI |
|---|---|---:|---:|---|
| SPI-3 lead-3 | Climatology | 0.08765 | 0.00000 | — |
| SPI-3 lead-3 | Persistence SPI-3 | 0.15798 | −0.80245 | [−1.55296, −0.36279] |
| SPI-3 lead-3 | XGBoost isotonic | 0.08449 | +0.03604 | [−0.12663, +0.13441] |
| SPI-3 lead-6 | XGBoost isotonic | 0.09329 | −0.10782 | [−0.32744, +0.04534] |
| SPI-6 lead-6 | Persistence SPI-6 | 0.24745 | −0.94918 | [−2.01903, −0.36532] |
| SPI-6 lead-6 | XGBoost isotonic | 0.14092 | −0.11009 | [−0.37336, +0.03511] |

For Central Valley, SPI-3 lead-3 is the only seasonal target with a positive
point estimate, but its confidence interval crosses zero. It is a useful
hypothesis-generating signal, not a defensible positive-skill claim. The SPI-6
lead-6 persistence baseline is now target-consistent (`spi6_lag1`, not the old
SPI-3 proxy).

The same runner now supports source-cited regional masks and climate-feature
schemas, including the Horn of Africa country-mask caveat. The manuscript-facing
seasonal audit is consolidated in
[`results/report/paper/table05_seasonal_signal_audit.csv`](results/report/paper/table05_seasonal_signal_audit.csv).
The only robust-positive regional seasonal row so far is Mediterranean Spain
SPI-6 lead-6 with Niño3.4-only features (`BSS = +0.078`, 95% CI
`[+0.004, +0.162]`), but the signal audit classifies it as a calibration-shift
result rather than temporal event tracking (`r = 0.041`, calibrated variance
ratio `0.104`). This does not overturn the main predictability-barrier result.

### Land-Surface Driver Experiments

A separate ERA5-Land experiment adds regional 2m temperature and VPD monthly
anomaly lags at `t` and `t-1` to the canonical SPI-1 lead-1 target. These
features are leakage-safe because they are known at the feature month, not the
target month.

| ERA5-Land met-feature forecaster | Dry BS | BSS vs. climatology | 95% CI |
|---|---:|---:|---|
| Climatology | 0.06428 | 0.00000 | — |
| XGBoost + met raw | 0.06622 | −0.03017 | [−0.45397, 0.21753] |
| XGBoost + met isotonic | 0.07020 | −0.09217 | [−0.17743, −0.03138] |
| XGBoost + met Platt | 0.07284 | −0.13322 | [−0.24330, −0.06439] |
| XGBoost + met selected | 0.07020 | −0.09217 | [−0.17743, −0.03138] |
| XGBoost-Spatial + gridded met selected | 0.06729 | −0.04678 | [−0.13215, 0.02511] |

Regional VPD/temperature anomalies are highly used by the model and improve
raw non-spatial XGBoost substantially relative to the corrected ENSO-only XGB
run, but they still do not establish positive BSS. Spatialized ERA5-Land
temperature/VPD anomalies also remain below climatology when added to the
XGBoost-Spatial feature set. Validation-selected calibration chose isotonic in
both met-feature experiments and degraded test-period skill, which is another
sign of distribution shift in the 2021-2026 test period.

A second ERA5-Land experiment adds regional volumetric soil-water anomaly lags
for layers 1-3 plus a 0-100 cm root-zone approximation. This tests land-surface
memory directly, but the result is worse than the met-feature run:

| ERA5-Land soil-moisture forecaster | Dry BS | BSS vs. climatology | 95% CI |
|---|---:|---:|---|
| Climatology | 0.06428 | 0.00000 | — |
| XGBoost + soil raw | 0.10184 | −0.58431 | [−1.41972, −0.25733] |
| XGBoost + soil isotonic | 0.07444 | −0.15801 | [−0.37426, −0.04884] |
| XGBoost + soil Platt | 0.07200 | −0.12012 | [−0.21919, −0.05756] |
| XGBoost + soil selected | 0.07444 | −0.15801 | [−0.37426, −0.04884] |

Soil-moisture features consume most of the trained model's split gain (~68%),
but test-period monthly dry-probability correlation is near zero or negative.
That is a strong overfitting warning: regional soil moisture does not add
usable SPI-1 lead-1 probability skill in this setup.

### Memory-Target Checkpoint

Because the related literature suggests better skill for memory-bearing drought
targets, the project now includes a Central Valley SPI-6 lead-6 checkpoint:

```bash
python scripts/run_memory_target_experiment.py --target-spi 6 --lead-months 6 --copy-report
```

| SPI-6 lead-6 checkpoint | Test months | BSS vs. climatology | 95% CI | Diagnostic |
|---|---:|---:|---:|---|
| Lag/climate XGBoost selected | 62 | +0.040 | [−0.020, +0.082] | `r = 0.004`, amplitude ratio `0.076` |
| Soil-memory XGBoost selected | 62 | −0.158 | [−0.410, +0.012] | soil variables dominate gain but do not generalize |
| CPC NMME anomaly selected | 62 | −0.344 | [−1.190, +0.104] | external benchmark only |
| CPC NMME probability selected | 49 | −0.409 | [−1.766, +0.168] | partial coverage |
| SPI-6 persistence | 62 | −0.949 | [−2.019, −0.329] | robustly worse than climatology |

This is an important negative/suggestive result. Longer-memory SPI-6 improves
the lag/climate point estimate, but the selected probabilities mostly track a
calibration shift rather than month-to-month drought events. ERA5-Land
soil-memory lags do not solve the problem. The NMME coverage audit also shows
that the real-time CPC archive has zero overlap with the 1991-2016 training
period, so a trained forecast-informed ML model needs hindcast or ensemble
archives, not only the real-time files currently cached here.

### Forecast-Informed Land-Surface Benchmark

The strongest positive direction is now a separate land-surface target, not the
canonical SPI-1 precipitation target. `scripts/run_landsurface_forecast_benchmark.py`
verifies NOAA NCEI CFSv2 monthly-mean `flxf` soil-water forecasts against an
ERA5-Land 0-100 cm root-zone soil-moisture dry-fraction target. Dry thresholds
and climatology use the 1991-2016 train period; CFSv2 forecast signals are
mapped to dry-fraction probability with validation-only isotonic calibration.

```bash
python scripts/run_landsurface_forecast_benchmark.py \
  --start-target 2017-01 \
  --copy-report

python scripts/run_landsurface_forecast_benchmark.py \
  --region cvalley \
  --run-hours 0 6 12 18 \
  --start-target 2017-01 \
  --copy-report
```

| Land-surface checkpoint | Test months | BSS vs. climatology | 95% CI | Diagnostic |
|---|---:|---:|---:|---|
| Central Valley CFSv2 RZSM selected, 18 UTC | 42 | +0.630 | [+0.477, +0.762] | strongest CFSv2 row |
| Central Valley CFSv2 RZSM selected, 4-cycle mean | 39 | +0.511 | [+0.292, +0.676] | strict all-cycle replication |
| Central Valley RZSM persistence selected, 4-cycle months | 39 | +0.468 | [+0.264, +0.605] | persistence also robust |
| Southern Great Plains CFSv2 RZSM selected, 4-cycle mean | 39 | +0.413 | [-0.010, +0.633] | positive but uncertain; worse than persistence |
| Southern Great Plains RZSM persistence raw | 39 | +0.695 | [+0.414, +0.851] | strongest land-memory row |
| Mediterranean Spain CFSv2 RZSM selected, 4-cycle mean | 39 | -0.141 | [-1.344, +0.468] | no replication |

This is now a more nuanced result. Root-zone soil-moisture dry fraction is much
more predictable than SPI-1 in some regions, and CFSv2 RZSM remains robustly
positive in Central Valley after the four-cycle replication. But the added
regions show mixed CFSv2 value: Southern Great Plains is dominated by
persistence, and Mediterranean Spain does not replicate. The defensible claim is
therefore target reframing toward land-surface drought, not broad CFSv2 added
value, precipitation SPI skill, all leads, all regions, or deployment readiness.
The explicit added-value diagnostic is saved at
[`results/report/paper/table09_landsurface_added_value.csv`](results/report/paper/table09_landsurface_added_value.csv);
it shows no robust overall CFSv2 improvement over raw persistence, and Southern
Great Plains persistence is robustly better.
Detailed coverage-gap files are generated locally under `results/report/` when
the benchmark is rerun.

### Regional Forecast Evaluation

Beyond pixel-level skill, we evaluate the model's ability to predict the **dominant drought class at the Central Valley scale**. For each month, we compute the fraction of pixels predicted as dry/normal/wet and compare the dominant class to observations. These local diagnostic outputs are generated under `results/regional/`; the manuscript-facing conclusions are consolidated in `results/report/paper/`.

### Multi-Region Evaluation Path

The region-aware runner now supports rectangular CHIRPS regions plus
country/basin/ecoregion mask sensitivities without overwriting the canonical
Central Valley checkpoint:

```bash
python scripts/run_multiregion_xgb_experiment.py --list-regions
python scripts/run_multiregion_xgb_experiment.py --region cvalley --model both --copy-report
python scripts/run_multiregion_xgb_experiment.py --region southern_great_plains --model both --spi-n-jobs 8 --copy-report
python scripts/build_region_masks.py --copy-report
python scripts/run_multiregion_xgb_experiment.py --region mediterranean_spain --model both --country-mask --rebuild-dataset --copy-report
python scripts/build_basin_masks.py --copy-report
python scripts/run_multiregion_xgb_experiment.py --region cvalley --model both --basin-mask --rebuild-dataset --copy-report
python scripts/run_multiregion_xgb_experiment.py --region southern_great_plains --model both --basin-mask --rebuild-dataset --copy-report
python scripts/run_multiregion_xgb_experiment.py --region murray_darling --prepare-grid-only --rebuild-pr --rebuild-spi --spi-n-jobs 8
python scripts/build_basin_masks.py --copy-report
python scripts/run_multiregion_xgb_experiment.py --region murray_darling --model both --basin-mask --rebuild-dataset --copy-report
python scripts/run_multiregion_xgb_experiment.py --region horn_of_africa --prepare-grid-only --rebuild-pr --rebuild-spi --spi-n-jobs 8
python scripts/build_region_masks.py --copy-report
python scripts/run_multiregion_xgb_experiment.py --region horn_of_africa --model both --country-mask --rebuild-dataset --copy-report
python scripts/run_multiregion_xgb_experiment.py --region mediterranean_spain --model both --basin-mask --rebuild-dataset --copy-report
```

The runner clips CHIRPS, computes region-specific SPI, builds the same
SPI-1[t+1] forecast table, and evaluates monthly dry-fraction BSS. Parallel SPI
fitting is available through `--spi-n-jobs`; `--grid-stride` is available only
for smoke tests and should not be treated as a scientific result.
The manuscript-facing multi-region and mechanism evidence is consolidated in
[`results/report/paper/table02_headline_results.csv`](results/report/paper/table02_headline_results.csv),
[`results/report/paper/table03_mask_methods.csv`](results/report/paper/table03_mask_methods.csv),
and [`results/report/paper/table06_regionalization_mechanism.csv`](results/report/paper/table06_regionalization_mechanism.csv).
The mask audits and reproducible mechanism analysis are:

```bash
python scripts/build_region_masks.py --copy-report
python scripts/build_basin_masks.py --copy-report
python scripts/analyze_multiregion_mechanisms.py
```

They write the region-mask diagnostics, BSS comparison, monthly dry-fraction
traces, signal-vs-skill scatter, feature-group gain summary, and a short
interpretation report under local generated result folders.

Current regional result:

| Region / model | Dry BS | BSS vs. climatology | 95% CI | Note |
|---|---:|---:|---|---|
| Central Valley / spatial XGB retrain | 0.06598 | −0.02644 | [−0.12449, 0.04704] | Parity check: near climatology, consistent with frozen checkpoint |
| Central Valley basin-mask / tabular XGB | 0.06972 | −0.02065 | [−0.12852, 0.04233] | DWR groundwater-basin mask strengthens ranking but not probability skill |
| Southern Great Plains / tabular XGB | 0.05164 | −0.08218 | [−0.14450, −0.00066] | First full contrasting-region test; still below climatology |
| Southern Great Plains / spatial XGB | 0.05163 | −0.08200 | [−0.14790, −0.00190] | Spatial context does not close the gap |
| Southern Great Plains ecoregion-mask / spatial XGB | 0.05544 | +0.01013 | [−0.09777, 0.15022] | EPA South Central Semi-Arid Prairies mask improves the point estimate, but the CI crosses zero |
| Murray-Darling basin-mask / tabular XGB | 0.04633 | −0.63945 | [−1.30820, −0.27846] | Official Water Act basin mask; ranking exists but calibration/test-period shift fails badly |
| Horn of Africa country-mask / spatial XGB | 0.01385 | −0.03111 | [−0.30241, 0.31967] | Near climatology after calibration, but not positive skill; country mask is not a livelihood-zone mask |
| Mediterranean Spain / tabular XGB | 0.04587 | +0.04403 | [−0.15104, 0.24096] | Positive point estimate, not statistically reliable |
| Mediterranean Spain / spatial XGB | 0.04692 | +0.02212 | [−0.15139, 0.17647] | Spatial context weakens the point estimate |
| Mediterranean Spain country-mask / spatial XGB | 0.04827 | +0.02350 | [−0.19910, 0.30486] | Positive point estimate survives masking but remains highly uncertain |
| Mediterranean Spain basin-mask / spatial XGB | 0.04738 | −0.08459 | [−0.23434, 0.03410] | Hydrologic-district mask turns the Spain hint negative |

This no longer reads as a simple universal "no skill" result, but the cleanest
geometry-aware checkpoint still does not support robust positive skill. Central
Valley has ranking signal but weak calibrated skill, rectangular Southern Great
Plains is below climatology, the EPA ecoregion-masked Southern Great Plains
checkpoint improves to a small positive but uncertain point estimate,
Murray-Darling fails strongly after calibration/test-period shift, Horn of
Africa calibrates near climatology but not above it, and the earlier positive
Mediterranean Spain point estimate does not survive the stricter basin-district
mask.

The geometry audit is now explicit. Natural Earth country masks remove 0.0% of
valid Southern Great Plains cells, 1.61% of Murray-Darling cells, 2.85% of
Central Valley cells, 5.64% of Mediterranean Spain cells, and 12.71% of Horn of
Africa cells. Stricter basin/hydroclimate masks materially change the sample:
the DWR Central Valley groundwater-basin mask retains 27.92% of valid Central
Valley cells, the EPA Southern Great Plains ecoregion mask retains 78.64%, the
official Murray-Darling Basin mask retains 52.07%, and the selected Spain
river-basin district mask retains 52.31%. Those masked runs are the cleaner
interpretation checkpoints; Horn remains a country-intersection checkpoint with
a boundary-definition caveat.

Boundary and mask sources are cited in the reproducible diagnostics and in this
README for paper traceability: Natural Earth 1:50m country polygons
([GitHub mirror](https://github.com/nvkelso/natural-earth-vector/blob/master/geojson/ne_50m_admin_0_countries.geojson)),
California DWR Bulletin 118 groundwater basins
([FeatureServer](https://gis.water.ca.gov/arcgis/rest/services/Geoscientific/i08_B118_CA_GroundwaterBasins/FeatureServer/0)),
US EPA Level III ecoregions
([EPA data page](https://www.epa.gov/eco-research/level-iii-and-iv-ecoregions-continental-united-states)),
Murray-Darling Basin Authority / data.gov.au basin boundary
([WFS GeoJSON](https://data.gov.au/geoserver/murray-darling-basin-boundary/wfs?request=GetFeature&typeName=ckan_4ede9aed_5620_47db_a72b_0b3aa0a3ced0&outputFormat=json)),
and MITECO terrestrial river-basin districts
([OGC collection](https://wmts.mapama.gob.es/sig-api/ogc/features/v1/collections/agua%3ADemarcaciones_ET)).

### Operational Forecast Benchmark Path

The operational benchmark path compares external dynamical precipitation
forecasts against the same monthly climatology reference. The reproducible
Central Valley NMME run is:

```bash
python scripts/run_operational_precip_benchmark.py --write-template
python scripts/prepare_cpc_nmme_precip_anomaly_inputs.py --copy-report
python scripts/run_operational_precip_benchmark.py \
  --forecast-csv outputs/nmme_cpc_cvalley_lead1_forecast.csv \
  --copy-report
python scripts/prepare_cpc_nmme_precip_anomaly_inputs.py \
  --lead-months 3 \
  --start-target 2018-07 \
  --out-file outputs/nmme_cpc_cvalley_lead3_forecast.csv \
  --copy-report
python scripts/run_operational_precip_benchmark.py \
  --forecast-csv outputs/nmme_cpc_cvalley_lead3_forecast.csv \
  --dataset data/processed/dataset_seasonal_spi3_lead3.parquet \
  --target-spi 3 \
  --lead-months 3 \
  --output-prefix operational_nmme_cpc_spi3_lead3 \
  --copy-report
python scripts/prepare_cpc_nmme_precip_anomaly_inputs.py \
  --lead-months 6 \
  --start-target 2018-10 \
  --out-file outputs/nmme_cpc_cvalley_lead6_forecast.csv \
  --copy-report
python scripts/run_operational_precip_benchmark.py \
  --forecast-csv outputs/nmme_cpc_cvalley_lead6_forecast.csv \
  --dataset data/processed/dataset_seasonal_spi6_lead6.parquet \
  --target-spi 6 \
  --lead-months 6 \
  --output-prefix operational_nmme_cpc_spi6_lead6 \
  --copy-report
python scripts/prepare_cpc_nmme_precip_probability_inputs.py \
  --lead-months 3 \
  --dataset data/processed/dataset_seasonal_spi3_lead3.parquet \
  --out-file outputs/nmme_cpc_prob_cvalley_lead3_forecast.csv \
  --copy-report
python scripts/run_operational_precip_benchmark.py \
  --forecast-csv outputs/nmme_cpc_prob_cvalley_lead3_forecast.csv \
  --dataset data/processed/dataset_seasonal_spi3_lead3.parquet \
  --target-spi 3 \
  --lead-months 3 \
  --output-prefix operational_nmme_cpc_prob_spi3_lead3 \
  --copy-report
```

The forecast CSV should contain `target_time` plus one of
`forecast_prob_dry`, `forecast_pr_anom`, or `forecast_pr`. This lets SubX,
NMME, GEFS, ECMWF/SEAS5, or similar precipitation forecasts be scored with the
same validation-only calibration and monthly BSS protocol used everywhere else
in the project.

The implemented operational checkpoints now include CPC NMME real-time
multi-model precipitation anomalies, official CPC NMME below-normal
precipitation probabilities, and a NOAA NCEI THREDDS CFSv2 individual-run
precipitation extraction over the Central Valley bounding box. The CPC
probability files start in 2019 and have missing Central Valley values for some
dry-season targets, so their test coverage is smaller than the anomaly
benchmarks. The NCEI CFSv2 extraction uses true lead-window precipitation for
SPI-3 lead-3, but the accessible 6-hour individual-run aggregation starts in
2016 and is too short-horizon for a complete SPI-6 accumulation window.

| Operational benchmark | Test months | BSS vs. climatology | 95% CI |
|---|---:|---:|---|
| CPC NMME SPI-1 lead-1 probability + isotonic | 52 | +0.131 | [−0.304, +0.338] |
| CPC NMME SPI-3 lead-3 anomaly + isotonic | 59 | +0.086 | [−0.225, +0.261] |
| CPC NMME SPI-6 lead-6 probability raw | 49 | +0.035 | [−0.430, +0.255] |
| CPC NMME SPI-3 lead-3 probability raw | 51 | +0.007 | [−0.549, +0.243] |
| CPC NMME SPI-1 lead-1 anomaly + isotonic | 63 | +0.002 | [−0.438, +0.239] |
| NCEI CFSv2 SPI-3 lead-3 accumulated precipitation + isotonic | 55 | −0.030 | [−0.166, +0.059] |
| NCEI CFSv2 SPI-3 lead-3 precipitation anomaly + isotonic | 55 | −0.303 | [−1.202, +0.200] |
| CPC NMME SPI-6 lead-6 anomaly + isotonic | 62 | −0.344 | [−1.130, +0.113] |
| CPC NMME SPI-3 lead-3 probability + isotonic | 51 | −0.212 | [−1.148, +0.176] |
| CPC NMME SPI-6 lead-6 probability + isotonic | 49 | −0.409 | [−1.693, +0.163] |

The precipitation operational benchmark is scientifically useful, but it does
not create a robust positive-skill result. The official raw probabilities have
weak positive SPI-3/SPI-6 point estimates, while validation-only isotonic
calibration helps SPI-1 but overfits the short probability validation window
for SPI-3/SPI-6. The separate land-surface benchmark above is the first robust
positive forecast-informed result, and it changes the candidate paper claim
toward root-zone soil-moisture drought rather than precipitation SPI.
Artifacts are saved at
local `results/report/` files when the benchmark scripts are run. Public
manuscript-facing operational and land-surface rows are consolidated in
[`results/report/paper/table02_headline_results.csv`](results/report/paper/table02_headline_results.csv)
and [`results/report/paper/table09_landsurface_added_value.csv`](results/report/paper/table09_landsurface_added_value.csv).

Use the CPC/NMME data page
([NOAA CPC](https://www.cpc.ncep.noaa.gov/products/NMME/data.html)), NCEI NMME
access notes ([NOAA NCEI](https://www.ncei.noaa.gov/products/weather-climate-models/north-american-multi-model)),
the CPC probability NetCDF archive
([NOAA CPC FTP](https://ftp.cpc.ncep.noaa.gov/NMME/prob/netcdf/)),
the NCEI THREDDS CFSv2 precipitation catalog
([NOAA NCEI THREDDS](https://www.ncei.noaa.gov/thredds/catalog/model-nmme_cfs_v2_pr_6h_agg/files/catalog.html)),
the NCEI CFSv2 monthly means catalog
([NOAA NCEI THREDDS](https://www.ncei.noaa.gov/thredds/catalog/model-cfs_v2_for_mm/catalog.html)),
and cite Kirtman et al. (2014)
([doi:10.1175/BAMS-D-12-00050.1](https://doi.org/10.1175/BAMS-D-12-00050.1)).
SubX is the alternative subseasonal benchmark; cite Pegion et al. (2019)
([doi:10.1175/BAMS-D-18-0270.1](https://doi.org/10.1175/BAMS-D-18-0270.1)).

---

## Evaluation Methods

- **Three naive baselines** — climatology, persistence, SPI-1 threshold
- **Bootstrap confidence intervals** (2,000 iterations) on all skill scores
- **Brier Score decomposition** (Murphy 1973) — identifies whether failures are due to reliability or resolution
- **Post-hoc calibration** — Platt scaling and isotonic regression tested on validation set, applied to frozen test set
- **Stratified skill diagnostics** — season-wise and ENSO-phase BSS tables with monthly bootstrap intervals
- **Spatial skill maps** — per-pixel accuracy over 2021-2026
- **Case studies** — 2021–22 drought and 2023 atmospheric rivers
- **Cross-dataset validation** — ERA5-Land SPI-1 comparison
- **Independent U.S. precipitation validation** — PRISM SPI-1 comparison over the DWR Central Valley basin mask
- **Temporal robustness audit** — rolling chronological Central Valley holdouts and 2021-2026 event-block diagnostics
- **Evaluation-inflation audit** — invalid random row splits, pixel-level inference, and overlapping SPI targets tested as methodological stress cases
- **Qualitative validation** — USDM D1+ plausibility check (not a metric)

### Evaluation-Inflation Audit

The project now directly tests how much apparent skill can be created by common
but invalid evaluation shortcuts:

```bash
python scripts/run_evaluation_inflation_audit.py --copy-report
```

| Audit row | Inference | BSS vs. climatology | 95% CI | Status |
|---|---|---:|---:|---|
| Strict SPI-1 lead-1 chronological | monthly | −0.023 | [−0.117, +0.073] | valid primary protocol |
| Random SPI-1 row split | monthly | +0.995 | [+0.993, +0.997] | invalid split |
| Overlapping SPI-3 lead-1 | monthly | +0.674 | [+0.407, +0.820] | invalid target |
| Strict SPI-1 lead-1 chronological | pixel | −0.019 | [−0.020, −0.018] | invalid pixel inference |

This is central to the paper claim. Randomly splitting spatial pixels across
train/validation/test or allowing SPI accumulation windows to overlap can make
the task look highly predictable. The strict monthly, chronological,
leakage-free protocol removes that inflation.

The literature protocol audit at
[`literature/literature_protocol_audit.csv`](literature/literature_protocol_audit.csv)
keeps this critique narrow: prior drought-ML studies are often not directly
comparable because target scale, lead time, inputs, validation split, metrics,
and baselines differ. We do not claim that the broader literature is invalid.

---

## Limitations

- **1-month lead is fundamentally hard:** Monthly precipitation is dominated by chaotic synoptic weather; SPI-1 autocorrelation is weak.
- **Small test set:** 63 months yields wide confidence intervals; positive skill claims require a confidence interval that excludes zero.
- **Regional geometry:** Central Valley, Southern Great Plains, Murray-Darling, Spain, and Horn of Africa now have source-cited geometry checkpoints; Horn still uses political-country geometry rather than a basin or livelihood-zone mask.
- **Limited exogenous drivers:** Corrected Niño3.4 anomaly lags are included; PDO is excluded from the active checkpoint because recent PDO values are missing. Regional/gridded ERA5-Land temperature/VPD and regional ERA5-Land soil-memory lags do not beat climatology for SPI-1. The separate CFSv2 forecast-informed root-zone soil-moisture target is robust against climatology in Central Valley, but added-region checks are mixed and the current added-value audit does not show robust improvement over same-target persistence.
- **Test period non-representative:** 2021–2026 is extreme (historic drought -> extreme wet). This is now partly controlled by `results/temporal/`: five rolling tabular holdouts all remain at or below climatology, but the canonical event-block analysis still has only 12-27 months per block.
- **CHIRPS/PRISM differences:** PRISM validation shows strong test-period agreement (`Pearson r = 0.816`, `Spearman r = 0.720`) but CHIRPS is drier on average over the basin (`CHIRPS - PRISM dry-fraction bias = +0.046`), so target-data uncertainty should be acknowledged.

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
python scripts/download_era5_land_met_monthly.py  # optional: t2m/d2m for VPD experiment
python scripts/download_era5_land_soil_moisture_monthly.py  # optional: soil-water experiment

# 2. Build forecast dataset
python scripts/build_dataset_forecast.py --climate-features nino34
python scripts/build_dataset_convlstm.py          # optional ConvLSTM arrays

# 3. Train corrected checkpoint models
python scripts/train_forecast_logreg.py
python scripts/train_forecast_rf.py
python scripts/train_forecast_xgboost.py
python scripts/train_forecast_xgb_spatial.py    # adds 3x3 neighbourhood features
python scripts/train_forecast_convlstm.py        # optional, GPU recommended

# 4. Evaluate and interpret
python scripts/evaluate_forecast_skill.py        # skill table + calibration study
python scripts/run_spi3_seasonal_experiment.py   # optional leakage-free SPI-3 lead-3 experiment
python scripts/run_met_feature_experiment.py      # optional ERA5-Land temperature/VPD experiment
python scripts/run_met_spatial_feature_experiment.py  # optional gridded met + spatial XGB experiment
python scripts/run_soil_moisture_feature_experiment.py  # optional ERA5-Land soil-moisture experiment
python scripts/run_multiregion_xgb_experiment.py --list-regions
python scripts/run_multiregion_xgb_experiment.py --region southern_great_plains --model both --spi-n-jobs 8 --copy-report
python scripts/build_region_masks.py --copy-report
python scripts/run_multiregion_xgb_experiment.py --region mediterranean_spain --model both --country-mask --rebuild-dataset --copy-report
python scripts/build_basin_masks.py --copy-report
python scripts/run_seasonal_longlead_experiment.py --list-regions
python scripts/run_multiregion_xgb_experiment.py --region cvalley --model both --basin-mask --rebuild-dataset --copy-report
python scripts/run_multiregion_xgb_experiment.py --region southern_great_plains --model both --basin-mask --rebuild-dataset --copy-report
python scripts/run_multiregion_xgb_experiment.py --region murray_darling --prepare-grid-only --rebuild-pr --rebuild-spi --spi-n-jobs 8
python scripts/build_basin_masks.py --copy-report
python scripts/run_multiregion_xgb_experiment.py --region murray_darling --model both --basin-mask --rebuild-dataset --copy-report
python scripts/run_multiregion_xgb_experiment.py --region horn_of_africa --prepare-grid-only --rebuild-pr --rebuild-spi --spi-n-jobs 8
python scripts/build_region_masks.py --copy-report
python scripts/run_multiregion_xgb_experiment.py --region horn_of_africa --model both --country-mask --rebuild-dataset --copy-report
python scripts/run_multiregion_xgb_experiment.py --region mediterranean_spain --model both --basin-mask --rebuild-dataset --copy-report
python scripts/analyze_multiregion_mechanisms.py
python scripts/evaluate_regional_forecast.py     # regional (Central Valley) dominant class accuracy
python scripts/xgb_shap_forecast_analysis.py --model both  # SHAP interpretation
python scripts/validate_era5_spi.py              # cross-dataset validation
python scripts/validate_chirps_prism_cvalley.py  # PRISM basin validation
python scripts/run_temporal_robustness_audit.py  # rolling holdout sensitivity
python scripts/validate_usdm.py                  # USDM plausibility check
python scripts/plot_spatial_skill.py             # per-pixel skill map
python scripts/plot_case_study.py                # 2021-2026 case study
python scripts/generate_master_results.py
python scripts/generate_manuscript_results.py
```

### Results (reproduced outputs)

- **[results/report/paper/](results/report/paper/)** — consolidated evidence pack: master/headline tables, mask methods, temporal robustness, seasonal signal audit, regionalization mechanism table, transition diagnostics, land-surface added-value table, and manuscript-facing figures.
- **[results/report/master_results_table.csv](results/report/master_results_table.csv)** and **[results/report/master_results_headline.csv](results/report/master_results_headline.csv)** — compact source tables feeding the paper pack.

---

## Acknowledgement

AI tools (ChatGPT, Gemini, GitHub Copilot) were used ethically for research & analysis, code review, and documentation.

---

## Author

Md Ishtiaque Hossain \
MSc, Computer and Information Sciences \
University of Delaware \
[LinkedIn](https://linkedin.com/in/ishtiaque-h) · [GitHub](https://github.com/Ishtiaque-h)
