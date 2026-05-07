# Final Report: CHIRPS Drought Forecasting Project (Working Draft)

Last updated: 2026-05-06

This report is the narrative synthesis for manuscript planning. The numerical
source of truth remains the generated CSV tables in `results/report/paper/` and
`results/report/`; regenerate those before changing any quantitative claim.

## Current Research Position

The strongest defensible claim is not that the current ML system is a broadly
skillful drought forecaster. The defensible claim is:

> A leakage-free, monthly block-evaluated drought-forecasting benchmark shows
> that lag-based ML can detect drought-relevant ranking signal, but it rarely
> converts that signal into robust calibrated probability skill over
> climatology for 1-month-ahead meteorological SPI-1. The result persists under
> corrected climate indices, source-cited regional masks, temporal robustness
> checks, independent PRISM validation, seasonal target experiments, and
> operational NMME/CFSv2 precipitation comparisons. A separate forecast-informed
> CFSv2 root-zone soil-moisture benchmark is robustly positive in Central
> Valley, but regional replication is mixed and same-target persistence can be
> stronger. This suggests target reframing toward land-surface drought is
> promising, but not yet a broad CFSv2 added-value claim.

This is scientifically useful if framed as a predictability and evaluation
audit. It is weak if framed as a new high-performing ML model.

## Current Evidence State

- Best canonical Central Valley checkpoint: XGB-Spatial with validation-only
  isotonic calibration gives BSS `+0.005`, CI `[-0.062, +0.073]`.
- Temporal robustness: 0 of 5 rolling Central Valley holdouts has positive BSS.
- PRISM validation: CHIRPS and PRISM dry-fraction timing agrees strongly, but
  the current XGB-Spatial probabilities remain tied with climatology against
  PRISM (`BSS = -0.003`).
- Multi-region evaluation: no source-cited mask checkpoint gives robust
  positive selected BSS across Central Valley, Southern Great Plains,
  Murray-Darling, Mediterranean Spain, and Horn of Africa.
- Seasonal/regional exception: Mediterranean Spain SPI-6 lead-6 with Nino3.4
  features has robust positive BSS (`+0.078`), but the signal audit flags it as
  calibration shift rather than event tracking (`r = 0.041`, variance ratio
  `0.104`).
- Operational benchmark: CPC NMME probability and anomaly rows produce positive
  point estimates in some cases, but all current CIs cross zero.
- Forecast-informed land-surface benchmark: Central Valley CFSv2 RZSM remains
  robustly positive under strict four-cycle aggregation (`BSS = +0.511`, CI
  `[+0.292, +0.676]`), but it does not beat raw persistence on point BS.
  Southern Great Plains CFSv2 is positive but uncertain (`+0.413`, CI crosses
  zero) and weaker than persistence; Mediterranean Spain is below climatology
  (`-0.141`).
- Memory-target checkpoint: Central Valley SPI-6 lead-6 lag/climate XGBoost has
  a positive but uncertain selected point estimate (`BSS = +0.040`, CI
  `[-0.020, +0.082]`), but it has essentially no monthly event tracking
  (`r = 0.004`, amplitude ratio `0.076`). Adding ERA5-Land soil-memory lags
  worsens selected BSS to `-0.158`, despite soil variables dominating gain.
- CPC NMME real-time products cannot yet be used as trained ML features under
  the strict split: SPI-6 lead-6 probability coverage starts at target
  2019-07 and anomaly coverage at 2018-10, leaving zero overlap with the
  1991-2016 training period. They remain external benchmarks unless hindcasts
  are added.
- Evaluation-inflation audit: the strict chronological monthly SPI-1 audit row
  stays near climatology (`BSS = -0.023`, CI `[-0.117, +0.073]`), while
  invalid shortcuts look highly skillful: random row split monthly `BSS =
  +0.995`, and overlapping SPI-3 lead-1 monthly `BSS = +0.674`. Pixel-level
  inference also produces artificially tiny intervals because it treats
  hundreds of thousands of spatially autocorrelated pixels as independent.
- Regionalization: SPI-12 zones reveal strong teleconnection signals in some
  regions, but those mechanisms do not reliably become calibrated SPI-1
  forecast skill.


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

## Literature Review

The collected related-work index is
`literature/related_works_index.csv`. Local PDFs are stored under
`literature/papers/` 
For a paper-by-paper support map and implementable next hypotheses, see
`literature/literature_synthesis_memo.md`. For the protocol-comparability
audit requested before finalizing the manuscript claim, see
`literature/literature_protocol_audit.csv` and
`literature/literature_protocol_audit.md`.

### Forecast Verification and Operational Baselines

Becker and van den Dool (2016) provide the appropriate NMME probability-forecast
benchmarking frame: Brier skill, reliability, and resolution should be assessed
against cross-validated climatology, and the Brier score admits the standard
reliability/resolution decomposition. This validates our use of BSS and
reliability-oriented calibration checks for CPC NMME probability products.

Carrao et al. (2018) show that seasonal SPI forecast skill depends on region,
season, lead time, and SPI accumulation period, and that skill should be judged
against climatological benchmarks. This supports our decision to treat positive
seasonal rows as conditional evidence, not as broad proof of skill.

Su et al. (2023) directly strengthens our western-U.S. predictability argument:
SubX-driven drought onset and termination skill is usable mainly at weeks 1-2,
limited by week 3, and mostly absent by week 4 for many drought severities. A
monthly SPI-1 lead-1 target is therefore expected to be difficult. Importantly,
their evaluation is explicitly probabilistic (including debiased Brier skill
score), aligning with our BSS-first framing.

### Memory, Target Choice, and Physical Predictability

Sutanto and Van Lanen (2022) show that hydrological drought forecasts outperform
meteorological ones where catchment memory is strong. This is important because
our main target, SPI-1, has little physical memory and depends on next-month
precipitation. It supports a stronger future hypothesis: skill is more likely
for memory-bearing targets such as root-zone soil moisture, streamflow drought,
SPI-6/SPI-12, or event duration/recovery than for SPI-1 dry fraction.

Lesinger and Tian (2025) show a successful modern direction: recursive deep
learning plus dynamic-model forecasts can skillfully predict root-zone soil
moisture drought at subseasonal leads. The key inputs are antecedent root-zone
soil moisture and dynamic-model forecast information. This does not suggest
that a bigger neural net on CHIRPS lags will solve our current problem; it
suggests a target/input pivot if we want a positive forecast model.

Lesinger et al. (2024) similarly supports using land-surface variables and
dynamic forecast inputs. Their SubX flash-drought study shows that evaporative
demand, soil moisture, and flash-drought predictability degrade with lead and
vary by target; in their abstract, lead week-1 forcing-variable ACC is moderate
to high (~0.70–0.95), but by weeks 3–4 predictability is low for all forcing
variables (ACC < 0.5).

Malloy and Kirtman (2023) strengthens the timescale-mismatch framing for
teleconnections: they find the lag between BSISO-related circulation anomalies
and Great Plains rainfall anomalies is about 2 weeks, and explicitly describe
this as a subseasonal “forecast of opportunity.” This supports our caution that
teleconnection signal can be real but episodic and not automatically transferrable
to stable 1-month-ahead SPI-1 probability skill.

### Machine Learning Drought Forecasting

Hwang et al. (2019), SubseasonalClimateUSA, DroughtSet, and DroughtCast all
point in the same direction: successful ML forecast papers generally use richer
inputs than lagged precipitation alone, including dynamic forecasts, soil
moisture, temperature, humidity/VPD, remote sensing, static geography, and
explicit benchmark comparisons. This supports our current limitation statement:
the project is not underpowered because it lacks yet another architecture; it is
limited because the canonical target/input combination is low-information.

The 2025 ML drought review by Osman et al. reinforces three reviewer-relevant
gaps: input-variable choice, lead-time dependence, and regional transferability.
Our project now addresses regional transferability and lead-time sensitivity
better than many ML-only drought papers, includes NOAA NCEI CFSv2
forecast-informed precipitation benchmarks, and now adds the more relevant
land-surface benchmark implied by recent root-zone soil-moisture work. The
remaining limitation is not simple replication anymore: the four-cycle and
added-region checks show mixed CFSv2 added value and strong persistence.

The protocol audit adds an important guardrail: prior positive studies should
not be dismissed as leaky unless their methods are audited in detail. Many are
simply different tasks. CHIRPS/SPI studies often target annual SPI-12 or
multi-month SPI regression, while operational and modern ML studies use
dynamic forecasts, land-surface states, remote sensing, benchmark datasets, or
deterministic metrics. The manuscript should therefore compare protocols rather
than claim the broader literature is invalid.

### SHAP, Regionalization, and Uncertainty

Ozupek et al. (2025) supports using SHAP/LIME for drought interpretability, but
also warns indirectly that lagged drought-index terms often dominate. Our SHAP
results should therefore be presented as model-behavior diagnostics and checked
against ablations, not as causal attribution.

Molosiwa et al. (2026) is the closest methodological support for our CHIRPS
SPI-12 regionalization path. It justifies using long-term CHIRPS SPI,
homogeneous drought zones, run-theory summaries, and teleconnection diagnostics.
For our paper, regionalization is strongest as mechanism evidence explaining
where teleconnection signal exists and why it often fails to become SPI-1 skill.

Schreck et al. (2024) shows evidential deep learning can be useful in
Earth-system applications when uncertainty is evaluated with calibration and
error-association diagnostics. Shen et al. (2024) provides the caution: EDL
uncertainty can be unreliable without rigorous validation. Therefore EDL should
not become a central claim unless we compare it against simpler calibrated
probability and ensemble baselines.

## Strengthened Hypothesis Options

### Primary Manuscript Hypothesis

Strict evaluation reveals a predictable gap between apparent ML signal and
robust calibrated drought probability skill. In lag-only meteorological SPI-1
forecasting, ranking signal can exist without positive Brier skill once
climatology, temporal blocking, calibration, masks, independent precipitation
validation, and test-period uncertainty are enforced.

This hypothesis is already well supported by the current project outputs.

### Stronger Positive-Skill Hypothesis

Drought predictability improves when the target and inputs contain physical
memory and forecast information: root-zone soil moisture, streamflow drought,
SPI-6/SPI-12, dynamic-model forecasts, and antecedent land-surface states should
outperform CHIRPS-only SPI-1 in regions/seasons with stronger memory or
teleconnection coupling.

This is scientifically stronger if the goal is a positive forecast model, but it
requires a clear pivot and additional data engineering.

### Regionalization Hypothesis

SPI-12 regionalization can identify teleconnection-sensitive drought regimes,
but teleconnection sensitivity alone is insufficient for calibrated SPI-1
forecast skill. Forecast skill requires target-scale alignment and event
tracking, not only strong climate-index correlation.

This is partially supported now and useful for discussion.


### Positioning Against Prior Work (What This Study Adds)

The literature supports three core points that make this study publishable if stated clearly:

1. **Subseasonal limits are real.** Subseasonal drought skill drops after weeks 3-4 in multiple regions, which is consistent with the weak 1-month-ahead skill we observe here (Su et al., 2023; Masukwedza et al., 2025). This frames the result as an expected physical limit rather than a model failure.
2. **Protocol comparability matters.** The literature audit shows that prior positive drought-ML studies often differ in target accumulation scale, lead time, input information, split design, metric, and baseline. Our work tests a narrower hard case rather than refuting CHIRPS/SPI drought forecasting in general.
3. **Reliability and uncertainty matter more than point accuracy.** Reviews emphasize probabilistic calibration and uncertainty quantification, but our results show that improved reliability does not automatically yield positive skill. We explicitly separate ranking skill from calibrated probability skill and show that calibration cannot create resolution where the signal is weak.

In short, this project is positioned as a **predictability audit**: it tests whether ML adds reliable probability skill over climatology under leakage-safe targets, monthly-level inference, and source-cited regional masks. That methodological framing is the scientific claim, not a particular model’s accuracy.

## Step 3 Diagnostics (Regionalization)

Method summary: Compute SPI-12 at each CHIRPS grid cell, standardize by 1991-2020 monthly climatology, then run PCA followed by k-means clustering to define homogeneous drought zones. For each zone, compute run-theory drought metrics (duration, severity, intensity) using the SPI-12 threshold of -1.0 and summarize ENSO/SOI/PDO correlations at lags 0-6 months to interpret teleconnection sensitivity. This is diagnostic-only and does not alter the frozen SPI-1 lead-1 forecast target or model checkpoints.

Core figures locked (Central Valley basin-masked):
- zone_map.png
- zone_run_duration_summary.png
- zone_spi12_timeseries.png

The compiled mechanism table now shows strong SPI-12 teleconnection structure in several regions, but weak conversion into SPI-1 lead-1 forecast skill. Examples: Horn of Africa zone 0 has Niño3.4 lag-6 correlation `r = 0.542`; Murray-Darling zone 0 has Niño3.4 lag-1 correlation `r = -0.460`; Southern Great Plains zone 3 has Niño3.4 lag-6 correlation `r = 0.501`. Zone-level forecast BSS is still mostly negative or only weakly positive in isolated zones, supporting the interpretation that the project is seeing real climate-memory signal at longer drought timescales but not robust monthly SPI-1 probability skill.

## Comprehensive ML Assessment: Project State & Predictability Barriers

### Overview
The May 2026 reproducibility checkpoint includes the canonical Central Valley benchmark, calibration study, EDL uncertainty experiment, feature-extension experiments, leakage-safe seasonal targets, five-region geometry-sensitive evaluation, temporal robustness audit, PRISM validation, operational precipitation benchmarks, a forecast-informed land-surface benchmark, and SPI-12 regionalization diagnostics. The current manuscript-facing source of truth is `results/report/paper/`, generated by `scripts/generate_manuscript_results.py`; the canonical monthly SPI-1 source table remains `results/report/master_results_headline.csv`. The critical finding is: **no canonical monthly SPI-1 experiment produces robust positive BSS over climatology**. The new positive result is different and narrower: land-surface root-zone dry fraction is predictable, but the current CFSv2 monthly-mean extraction does not establish added value over persistence.

### Performance Hierarchy (Monthly Brier Skill Score vs Climatology)

| Model | Test Set | BSS | Key Finding |
|-------|----------|-----|------------|
| CFSv2 RZSM anomaly selected | 42 mo | **+0.630** | Robust positive land-surface target result; not a precipitation-SPI claim |
| CFSv2 RZSM four-cycle selected | 39 mo | **+0.511** | Central Valley all-cycle replication remains robust positive |
| Southern Great Plains RZSM persistence raw | 39 mo | **+0.695** | Strongest land-surface row; persistence dominates CFSv2 |
| Southern Great Plains CFSv2 RZSM four-cycle | 39 mo | **+0.413** | Positive but uncertain; worse than persistence |
| Mediterranean Spain CFSv2 RZSM four-cycle | 39 mo | -0.141 | Added-region non-replication |
| ERA5-Land RZSM persistence raw | 42 mo | **+0.537** | Strong same-target memory baseline, but CI crosses zero |
| CPC NMME SPI-1 lead-1 probability + isotonic | 52 mo | **+0.131** | Largest operational point estimate, but partial coverage and CI crosses zero |
| CPC NMME SPI-3 lead-3 precipitation anomaly | 59 mo | **+0.086** | Strongest forecast-informed hint, but CI crosses zero |
| Mediterranean Spain SPI-6 lead-6 Niño3.4-only | 62 mo | **+0.078** | Robust BSS, but audit flags calibration shift (`r = 0.041`, low variance), not event tracking |
| Mediterranean Spain rectangular tabular | 63 mo | **+0.044** | Positive point estimate, but not robust and geometry-sensitive |
| CPC NMME SPI-6 lead-6 probability raw | 49 mo | **+0.035** | Official raw probability has weak positive point estimate; CI crosses zero |
| Seasonal SPI-3 lead-3 + isotonic | 59 mo | **+0.036** | Positive seasonal hint, but CI crosses zero |
| Southern Great Plains ecoregion mask | 63 mo | **+0.010** | Positive regional hint, but CI crosses zero |
| XGB-Spatial + isotonic calibration | 63 mo | **+0.005** | Best Central Valley probability checkpoint, but CI crosses zero |
| CPC NMME lead-1 precipitation anomaly | 63 mo | **+0.002** | Operational benchmark is also tied with climatology; CI crosses zero |
| NCEI CFSv2 SPI-3 lead-3 raw accumulated precipitation | 55 mo | -0.030 | True lead-window CFSv2 precipitation benchmark; near climatology, CI crosses zero |
| Random Forest raw | 63 mo | -0.068 | Strongest raw Central Valley Brier score among classical models |
| XGBoost-Spatial raw | 63 mo | -0.115 | Strong ranking signal, weak calibrated probability skill |
| EDL MLP selected | 63 mo | -0.116 | Uncertainty experiment underperforms climatology |
| SPI-6 lead-6 XGBoost isotonic | 62 mo | -0.110 | Target-consistent persistence fixed; model still below climatology |
| Soil moisture selected | 63 mo | -0.158 | Land-surface memory overfits this target |
| MJO/IVT selected | 39 mo | -0.333 | Shorter-period atmospheric feature test is robustly negative |
| Murray-Darling basin mask | 63 mo | -0.639 | Strong calibration/test-period shift failure |

**Conclusion**: The project should not claim a broadly positive forecast model. It should claim a rigorous predictability audit: ranking signals exist, but calibrated probability skill over climatology is not robust under monthly inference, source-cited regional masks, validation-only calibration, and signal-vs-calibration diagnostics.

### Temporal Robustness and Data-Source Validation

The test-period limitation is now controlled rather than only acknowledged. `scripts/run_temporal_robustness_audit.py` retrains tabular XGBoost across five chronological Central Valley holdouts. Against train-month climatology, all five BSS point estimates are non-positive: `2005-2008 = -0.089`, `2009-2012 = -0.004`, `2013-2016 = -0.077`, `2017-2020 = -0.010`, and `2021-2026 = -0.025`; all confidence intervals cross zero. The weak-skill conclusion is therefore not simply caused by the 2021-2026 drought/wet reversal. Within the canonical test, the main event-scale failure is underprediction during the 2021-2022 drought (`bias = -0.120`).

`scripts/validate_chirps_prism_cvalley.py` now downloads official PRISM 4 km monthly precipitation, clips to the DWR Central Valley basin union, computes PRISM SPI-1 with a 1991-2020 baseline, and compares it with CHIRPS. CHIRPS and PRISM dry fractions agree strongly during 2021-2026 (`Pearson r = 0.816`, `Spearman r = 0.720`), but CHIRPS is drier on average (`CHIRPS - PRISM dry-fraction bias = +0.046`). Evaluated directly against PRISM SPI-1 dry fraction, XGB-Spatial remains tied with climatology (`BSS = -0.003`, `Spearman = 0.530`). This strengthens the claim that the no-skill result is not only a CHIRPS artifact, while requiring an explicit data-source uncertainty caveat.

### Why Feature Engineering Fails: Atmospheric Predictability Limits

1. **Timescale Mismatch**: SPI-1[t+1] is a full next-month precipitation accumulation predicted from lagged monthly conditions, so much of the verifying precipitation lies beyond deterministic weather predictability.
2. **Feature Lag Obsolescence**: Monthly soil-moisture and atmospheric lag features can carry memory, but the current test-period evidence shows overfitting or reversion toward climatology before they add calibrated probability skill.
3. **Spatial Neighbors Help Ranking**: Spatial neighbor averaging improves raw XGBoost ranking and reduces overfitting, but even the calibrated spatial checkpoint is only `BSS = +0.005` with a confidence interval crossing zero.

### Uncertainty Quantification: Useful, But Not Yet a Claim

EDL decomposition shows high uncertainty, but the current EDL result should be framed cautiously. The selected EDL monthly BSS is `-0.116` with a CI below zero, and the monthly uncertainty signal has weak Spearman correlation with absolute forecast error (`total_u ≈ 0.03`, `epistemic_u ≈ 0.12`). EDL is therefore an exploratory uncertainty diagnostic, not a paper-ready uncertainty-quantification contribution by itself.

### Operational Benchmark Path

The operational comparison now uses CPC NMME real-time products, official CPC
below-normal precipitation probabilities, and a NOAA NCEI THREDDS CFSv2
individual-run precipitation extraction. All are scored against the same
monthly dry-fraction climatology after optional validation-only calibration:

```bash
python scripts/prepare_cpc_nmme_precip_anomaly_inputs.py --copy-report
python scripts/run_operational_precip_benchmark.py \
  --forecast-csv outputs/nmme_cpc_cvalley_lead1_forecast.csv \
  --copy-report
python scripts/prepare_cpc_nmme_precip_probability_inputs.py \
  --lead-months 3 \
  --dataset data/processed/dataset_seasonal_spi3_lead3.parquet \
  --out-file outputs/nmme_cpc_prob_cvalley_lead3_forecast.csv \
  --copy-report
python scripts/prepare_ncei_cfsv2_precip_forecast_inputs.py \
  --target-spi 3 \
  --lead-months 3 \
  --run-hours 18 \
  --copy-report
```

The anomaly NetCDF coverage starts at target month 2018-05, so calibration uses 32 validation months (2018-05 to 2020-12) and evaluation uses 63 test months (2021-01 to 2026-03). The selected isotonic NMME anomaly benchmark has `BSS = +0.002` with a confidence interval crossing zero. The raw NMME dry signal has modest rank correlation with observed monthly dry fraction (`Spearman ≈ 0.267`), but that signal does not translate into robust Brier skill. This is scientifically useful because it shows that even adding a real dynamical precipitation forecast input does not yet produce robust monthly dry-fraction probability skill in this Central Valley setup. SubX remains the corresponding subseasonal benchmark if the target is reframed around weeks 3-4.

Seasonal NMME anomaly checkpoints add a more target-aligned operational test. They are produced with `--lead-months 3` / `--target-spi 3` and `--lead-months 6` / `--target-spi 6` using the seasonal datasets. SPI-3 lead-3 improves the point estimate (`BSS = +0.086`, CI crossing zero), while SPI-6 lead-6 is below climatology (`BSS = -0.344`, CI crossing zero). This is scientifically useful because it shows that forecast-informed precipitation helps most at the SPI-3 seasonal target, but still does not produce robust positive probability skill.

The new NCEI CFSv2 extraction is stricter about lead-window alignment for
SPI-3 lead-3: it accumulates individual-run CFSv2 precipitation over the full
target SPI-3 window before scoring. The raw accumulated precipitation row is
near climatology (`BSS = -0.030`, CI `[-0.166, +0.059]`), while the forecast
anomaly row is worse (`BSS = -0.303`, CI `[-1.202, +0.200]`). This reduces the
remaining operational-precipitation ambiguity: a true lead-window dynamical
precipitation input still does not yield robust Central Valley SPI-3 probability
skill. The NCEI 6-hour individual-run aggregation starts in 2016 and has about
a five-month horizon, so it is not a complete SPI-6 lead-6 accumulation source.

Official CPC probability checkpoints are a stronger probabilistic benchmark but
do not change the conclusion. The selected SPI-1 lead-1 probability row has
`BSS = +0.131` with a confidence interval crossing zero (`[-0.304, +0.338]`)
over 52 test months. Raw probability rows for SPI-3 lead-3 and SPI-6 lead-6 are
weakly positive (`+0.007` and `+0.035`) but uncertain; isotonic calibration
degrades the SPI-3/SPI-6 probability rows because the CPC probability archive
only gives 15-18 validation months before the 2021+ test period. The probability
benchmark also has missing Central Valley dry-season targets, so it is a
partial-coverage operational comparison rather than a replacement for the
all-month anomaly benchmark.

The forecast-informed land-surface benchmark is the first robust positive
operational-style result, but the replication changes the interpretation.
`scripts/run_landsurface_forecast_benchmark.py` extracts NCEI CFSv2
monthly-mean `flxf` soil-water forecasts, approximates 0-100 cm root-zone soil
moisture from the first three CFSv2 soil layers, and verifies against an
ERA5-Land root-zone soil-moisture dry-fraction target. The original Central
Valley 18 UTC row has `BSS = +0.630`; the stricter four-cycle Central Valley
replication remains robustly positive (`BSS = +0.511`, CI
`[+0.292, +0.676]`). However, CFSv2 does not beat raw persistence on point BS
in the four-cycle Central Valley subset, Southern Great Plains persistence is
much stronger than CFSv2, and Mediterranean Spain does not replicate. This is
best described as land-surface target predictability with uncertain added value
from CFSv2 beyond persistence. The explicit added-value diagnostic in
`results/report/paper/table09_landsurface_added_value.csv` confirms this:
overall Central Valley CFSv2 is nearly tied with raw persistence (`delta BS =
+0.001`, CI crossing zero), Southern Great Plains persistence is robustly
better (`delta BS = +0.020`, CI `[+0.003, +0.036]`), and Mediterranean Spain
does not show CFSv2 skill over climatology or persistence.

The first memory-target checkpoint is now complete. `scripts/run_memory_target_experiment.py`
tests Central Valley SPI-6 lead-6 with lag/climate features and then with
ERA5-Land soil-water/root-zone anomaly lags. Lag/climate XGBoost reaches a
positive but uncertain selected BSS (`+0.040`, CI `[-0.020, +0.082]`), but the
monthly Spearman correlation is near zero and the calibrated probability
amplitude is strongly damped. Soil-memory features dominate gain but degrade
the validation-selected test result (`BSS = -0.158`, CI `[-0.410, +0.012]`).
This means the literature-informed hypothesis is only partially supported for
lag-only targets: longer-memory SPI helps the Brier point estimate slightly,
but observed soil-memory lags alone do not supply robust event-tracking
forecast skill. The positive land-surface result needs dynamic forecast soil
moisture, not just lagged ERA5-Land memory.

The transition-target checkpoint is useful, but replication weakens it as a
positive claim. `scripts/run_transition_target_experiment.py` derives SPI-1
lead-1 onset and termination events from the existing forecast tables and now
reports an eligible-pixel evaluation so current-state gating is not counted as
forecast skill. The rectangular Central Valley onset run is small but
robust-positive after eligible-only training (`BSS = +0.104`, CI
`[+0.025, +0.167]`, monthly Spearman `0.402`). However, the signal does not
survive the Central Valley basin mask (`BSS = -0.027`, CI crossing zero), is
robustly negative in the Southern Great Plains basin mask (`BSS = -0.084`, CI
`[-0.147, -0.016]`), and remains near climatology in Mediterranean Spain
(`BSS = -0.035`, CI crossing zero). Termination also does not beat the
eligible-state climatology (`BSS = -0.031`, CI crossing zero). This should
therefore be treated as a target-design diagnostic, not a headline positive
SPI result.

The evaluation-inflation audit is now the strongest methodological support for
the paper. `scripts/run_evaluation_inflation_audit.py` shows that invalid
protocols can manufacture excellent-looking skill: random row splitting gives
monthly `BSS = +0.995`, and an overlapping SPI-3 lead-1 target gives monthly
`BSS = +0.674`. The valid strict SPI-1 monthly audit row remains tied with
climatology (`BSS = -0.023`). This directly supports the manuscript framing:
many drought-ML workflows can overstate predictability if they split spatial
pixels randomly, score at pixel level, or let SPI accumulation windows overlap
between features and targets.

### Paper Narrative Recommendations

- **Primary** (Recommended): "Limits of Lag-Based ML for Monthly Drought Forecasting" — rigorous negative result and multi-region predictability audit
- **Secondary**: "Land-Surface Drought Predictability vs Persistence" — use CFSv2 root-zone soil-moisture forecasts and same-target persistence to show that target reframing can produce skill where precipitation SPI does not, while testing whether dynamic forecasts add value over land memory
- **Exploratory target-design diagnostic**: "Drought-Onset Transitions" — rectangular Central Valley onset has a small positive checkpoint, but the signal does not survive basin/regional replication
- **Supporting**: "Regional Drought Teleconnections" — Step 3 regionalization standalone
- **Most defensible next positive-skill direction**: move to a richer hindcast/ensemble land-surface forecast archive if the paper needs a stronger positive component; the current CFSv2 monthly-mean extraction does not show robust added value over persistence.

## Recommendation

Do not add another standalone neural architecture now. The most valuable next
step is one of these two paths:

1. If the paper is framed as an evaluation/predictability audit, add a compact
   evaluation-inflation experiment showing how pixel-level or leakage-prone
   evaluation would overstate skill relative to the current monthly
   leakage-free protocol. This step is now complete and should become a central
   methods/results table rather than another appendix-only diagnostic.

2. If the paper needs a stronger positive forecast component, the next step is
   not more regional replication of the same CFSv2 extraction. The completed
   four-cycle and added-region checks show that persistence is a major baseline.
   A fuller hindcast/ensemble or SubX/GEFS/ECMWF land-surface path is needed to
   test dynamic added value beyond land memory.

EDL, ConvLSTM tuning, and SHAP expansion are secondary. They should support the
chosen research claim, not define it.

## Citation Links to Carry Forward

- CHIRPS v3: https://doi.org/10.1038/s41597-026-07096-4
- CHIRPS SPI-12 regionalization analogue: https://doi.org/10.1007/s00704-026-06154-6
- NMME probabilistic BSS/reliability: https://doi.org/10.1175/JCLI-D-14-00862.1
- ECMWF S4 seasonal SPI forecasting: https://doi.org/10.3390/cli6020048
- Western U.S. SubX drought skill: https://doi.org/10.1175/JHM-D-22-0103.1
- SubX flash drought skill: https://doi.org/10.1175/JHM-D-23-0124.1
- Catchment memory and drought forecast skill: https://doi.org/10.1038/s41598-022-06553-5
- Hybrid DL-dynamic soil-moisture drought forecasts: https://doi.org/10.1038/s41467-025-62761-3
- DroughtCast: https://doi.org/10.3389/fdata.2021.773478
- Subseasonal ML western U.S.: https://arxiv.org/abs/1809.07394
- SubseasonalClimateUSA: https://openreview.net/forum?id=pWkrU6raMt
- DroughtSet: https://arxiv.org/abs/2412.15075
- XAI drought forecasting: https://doi.org/10.1007/s00477-025-03007-y
- EDL for Earth-system uncertainty: https://doi.org/10.48550/arXiv.2309.13207
- EDL caution: https://doi.org/10.48550/arXiv.2402.06160
- ML drought forecasting review: https://doi.org/10.1016/j.crm.2025.100758

## References

1. Funk, C. et al. (2026). CHIRPS v3 dataset. Scientific Data. https://doi.org/10.1038/s41597-026-07096-4
2. Funk, C. et al. (2015). CHIRPS environmental record. Scientific Data. https://doi.org/10.1038/sdata.2015.66
3. World Meteorological Organization (2012). Standardized Precipitation Index User Guide (WMO-No. 1090). https://library.wmo.int/idurl/4/39629
4. Murphy, A. (1973). Brier score decomposition. Journal of Applied Meteorology. https://ui.adsabs.harvard.edu/abs/1973JApMe..12..595M/abstract
5. Su, L. et al. (2023). Subseasonal drought forecast skill over the coastal western United States. Journal of Hydrometeorology. https://doi.org/10.1175/JHM-D-22-0103.1
5a. Kirtman, B. et al. (2014). The North American Multi-Model Ensemble. Bulletin of the American Meteorological Society. https://doi.org/10.1175/BAMS-D-12-00050.1
5b. Pegion, K. et al. (2019). The Subseasonal Experiment (SubX). Bulletin of the American Meteorological Society. https://doi.org/10.1175/BAMS-D-18-0270.1
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
25. PRISM Climate Group, Oregon State University. PRISM Climate Data. https://prism.oregonstate.edu/?id=US
26. Daly, C. et al. (2008). Physiographically sensitive mapping of climatological temperature and precipitation. International Journal of Climatology. https://doi.org/10.1002/joc.1688
