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

**Leakage-free target design:** The target is `SPI-1[t+1]`, which depends *only* on `pr[t+1]` — unknown at prediction time. All features are derived from time *t* or earlier. There is **zero accumulation-window overlap** between features and target. This eliminates the data leakage present in SPI-3-based targets (where 2 of 3 accumulation months overlap with features).

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
python scripts/build_master_results_table.py
```

The consolidated outputs are
[results/report/master_results_table.csv](results/report/master_results_table.csv)
and the compact manuscript table
[results/report/master_results_headline.csv](results/report/master_results_headline.csv).
As of the current run, that table contains **no robust positive BSS result**:
all positive point estimates have confidence intervals crossing zero.

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
> Post-hoc isotonic calibration brings XGBoost-Spatial to **BSS = +0.005** (95% CI [−0.062, +0.073]), which is statistically indistinguishable from climatology. Full intervals are in [results/report/forecast_skill_bss_hss_table.csv](results/report/forecast_skill_bss_hss_table.csv) and [results/report/calib_study_results.csv](results/report/calib_study_results.csv).

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
schemas, including the Horn of Africa country-mask caveat. The compiled table is
[`results/seasonal/seasonal_regional_longlead_summary.csv`](results/seasonal/seasonal_regional_longlead_summary.csv),
with a companion signal audit at
[`results/seasonal/seasonal_regional_signal_audit.csv`](results/seasonal/seasonal_regional_signal_audit.csv).
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

### Regional Forecast Evaluation

Beyond pixel-level skill, we evaluate the model's ability to predict the **dominant drought class at the Central Valley scale**. For each month, we compute the fraction of pixels predicted as dry/normal/wet and compare the dominant class to observations. See [results/regional/](results/regional/) for dominant-class accuracy and class fraction time series.

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
The compact comparison table is saved at
[results/multiregion/regional_mechanism_summary.csv](results/multiregion/regional_mechanism_summary.csv).
The mask audits and reproducible mechanism analysis are:

```bash
python scripts/build_region_masks.py --copy-report
python scripts/build_basin_masks.py --copy-report
python scripts/analyze_multiregion_mechanisms.py
```

They write the region-mask diagnostics, BSS comparison, monthly dry-fraction
traces, signal-vs-skill scatter, feature-group gain summary, and a short
interpretation report under
[results/multiregion/](results/multiregion/).

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
python scripts/build_nmme_cpc_forecast_csv.py --copy-report
python scripts/run_operational_precip_benchmark.py \
  --forecast-csv outputs/nmme_cpc_cvalley_lead1_forecast.csv \
  --copy-report
python scripts/build_nmme_cpc_forecast_csv.py \
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
python scripts/build_nmme_cpc_forecast_csv.py \
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
python scripts/build_nmme_cpc_prob_forecast_csv.py \
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

The implemented operational checkpoints now include both CPC NMME real-time
multi-model precipitation anomalies and official CPC NMME below-normal
precipitation probabilities over the Central Valley bounding box. The
probability files start in 2019 and have missing Central Valley values for some
dry-season targets, so their test coverage is smaller than the anomaly
benchmarks.

| Operational benchmark | Test months | BSS vs. climatology | 95% CI |
|---|---:|---:|---|
| CPC NMME SPI-1 lead-1 probability + isotonic | 52 | +0.131 | [−0.304, +0.338] |
| CPC NMME SPI-3 lead-3 anomaly + isotonic | 59 | +0.086 | [−0.225, +0.261] |
| CPC NMME SPI-6 lead-6 probability raw | 49 | +0.035 | [−0.430, +0.255] |
| CPC NMME SPI-3 lead-3 probability raw | 51 | +0.007 | [−0.549, +0.243] |
| CPC NMME SPI-1 lead-1 anomaly + isotonic | 63 | +0.002 | [−0.438, +0.239] |
| CPC NMME SPI-6 lead-6 anomaly + isotonic | 62 | −0.344 | [−1.130, +0.113] |
| CPC NMME SPI-3 lead-3 probability + isotonic | 51 | −0.212 | [−1.148, +0.176] |
| CPC NMME SPI-6 lead-6 probability + isotonic | 49 | −0.409 | [−1.693, +0.163] |

The operational benchmark is scientifically useful, but it does not create a
robust positive-skill result. The official raw probabilities have weak positive
SPI-3/SPI-6 point estimates, while validation-only isotonic calibration helps
SPI-1 but overfits the short probability validation window for SPI-3/SPI-6.
Artifacts are saved at
[results/report/nmme_cpc_cvalley_lead1_forecast.csv](results/report/nmme_cpc_cvalley_lead1_forecast.csv),
[results/report/operational_precip_benchmark_monthly_scores.csv](results/report/operational_precip_benchmark_monthly_scores.csv),
[results/report/operational_precip_benchmark_scores.txt](results/report/operational_precip_benchmark_scores.txt),
[results/report/operational_nmme_cpc_spi3_lead3_monthly_scores.csv](results/report/operational_nmme_cpc_spi3_lead3_monthly_scores.csv),
and
[results/report/operational_nmme_cpc_spi6_lead6_monthly_scores.csv](results/report/operational_nmme_cpc_spi6_lead6_monthly_scores.csv),
plus the probability benchmark files
[results/report/operational_nmme_cpc_prob_spi1_lead1_monthly_scores.csv](results/report/operational_nmme_cpc_prob_spi1_lead1_monthly_scores.csv),
[results/report/operational_nmme_cpc_prob_spi3_lead3_monthly_scores.csv](results/report/operational_nmme_cpc_prob_spi3_lead3_monthly_scores.csv),
and
[results/report/operational_nmme_cpc_prob_spi6_lead6_monthly_scores.csv](results/report/operational_nmme_cpc_prob_spi6_lead6_monthly_scores.csv).

Use the CPC/NMME data page
([NOAA CPC](https://www.cpc.ncep.noaa.gov/products/NMME/data.html)), NCEI NMME
access notes ([NOAA NCEI](https://www.ncei.noaa.gov/products/weather-climate-models/north-american-multi-model)),
the CPC probability NetCDF archive
([NOAA CPC FTP](https://ftp.cpc.ncep.noaa.gov/NMME/prob/netcdf/)),
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
- **Qualitative validation** — USDM D1+ plausibility check (not a metric)

---

## Limitations

- **1-month lead is fundamentally hard:** Monthly precipitation is dominated by chaotic synoptic weather; SPI-1 autocorrelation is weak.
- **Small test set:** 63 months yields wide confidence intervals; positive skill claims require a confidence interval that excludes zero.
- **Regional geometry:** Central Valley, Southern Great Plains, Murray-Darling, Spain, and Horn of Africa now have source-cited geometry checkpoints; Horn still uses political-country geometry rather than a basin or livelihood-zone mask.
- **Limited exogenous drivers:** Corrected Niño3.4 anomaly lags are included; PDO is excluded from the active checkpoint because recent PDO values are missing. Regional/gridded ERA5-Land temperature/VPD and regional ERA5-Land soil moisture have been tested separately, but none beat climatology.
- **Test period non-representative:** 2021–2026 is extreme (historic drought -> extreme wet). This is now partly controlled by `results/temporal/`: five rolling tabular holdouts all remain at or below climatology, but the canonical event-block analysis still has only 12-27 months per block.
- **CHIRPS/PRISM differences:** PRISM validation shows strong test-period agreement (`Pearson r = 0.816`, `Spearman r = 0.720`) but CHIRPS is drier on average over the basin (`CHIRPS - PRISM dry-fraction bias = +0.046`), so target-data uncertainty should be acknowledged.

---

## Next Steps

Highest-impact directions (see [`ANALYSIS.md`](ANALYSIS.md) for full roadmap):

1. **Draft from the paper evidence pack** — `results/paper/` now contains the consolidated master evidence table, headline table, source-cited mask-methods table, temporal robustness table, seasonal signal audit, regionalization mechanism table, and five manuscript-facing figures.
2. **Stop expanding regions for now** — Five hydroclimate checkpoints are enough to support the generalization claim; additional regions would add cost before the narrative is tightened.
3. **Only extend operational benchmarks if needed** — CPC NMME anomaly and probability benchmarks now cover SPI-1 lead-1, SPI-3 lead-3, and SPI-6 lead-6. A further follow-up should use full hindcast/ensemble NetCDF or SubX only if the paper needs a stronger operational comparison.
4. **Reframe the target only if pursuing positive skill** — SPI-12 regionalization shows teleconnection signal at longer drought-memory timescales, but the current SPI-3/SPI-6 regional seasonal audit does not yet show reliable event tracking. The next fair target-reframing step should add independent forecast precipitation/circulation inputs or zone-level targets, not only tune the same lagged-observation model.

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
python scripts/build_seasonal_regional_summary.py
python scripts/audit_seasonal_regional_signal.py
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
python scripts/build_regionalization_mechanism_tables.py
python scripts/validate_usdm.py                  # USDM plausibility check
python scripts/plot_spatial_skill.py             # per-pixel skill map
python scripts/plot_case_study.py                # 2021-2026 case study
python scripts/build_master_results_table.py
python scripts/build_paper_evidence_pack.py
```

### Results (reproduced outputs)

**Key results are saved to the `results/` folder by category:**

- **[results/report/](results/report/)** — Main skill table, calibration study, reliability diagrams, SHAP summaries, seasonal SPI-3, met-feature, and soil-moisture experiments
- **[results/report/](results/report/)** — Includes season/ENSO-stratified BSS CSV tables
- **[results/paper/](results/paper/)** — Consolidated paper evidence pack: master/headline tables, mask methods, temporal robustness, seasonal signal audit, regionalization mechanism table, and manuscript-facing figures
- **[results/spatial/](results/spatial/)** — Per-pixel accuracy maps
- **[outputs/](outputs/)** — Full model artifacts, probability arrays, feature-importance plots, and detailed SHAP dependence plots
- **[results/validation/](results/validation/)** — ERA5-Land, PRISM, and USDM validation/consistency checks
- **[results/temporal/](results/temporal/)** — Rolling temporal robustness and event-block diagnostics
- **[results/seasonal/](results/seasonal/)** — Seasonal long-lead and regional seasonal signal audits
- **[results/regionalization/](results/regionalization/)** — SPI-12 regionalization and zone-level mechanism tables
- **[results/regional/](results/regional/)** — Regional (Central Valley) forecast evaluation
- **[results/multiregion/](results/multiregion/)** — Region-aware XGBoost comparison artifacts

---

## Acknowledgement

AI tools (ChatGPT, Gemini, GitHub Copilot) were used ethically for research & analysis, code review, and documentation.

---

## Author

Md Ishtiaque Hossain \
MSc, Computer and Information Sciences \
University of Delaware \
[LinkedIn](https://linkedin.com/in/ishtiaque-h) · [GitHub](https://github.com/Ishtiaque-h)
