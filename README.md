# Central Valley Drought Classifier

**Can machine learning predict next month's drought from satellite precipitation alone?**

We build a leakage-free forecasting pipeline to predict monthly drought classes (*Dry / Normal / Wet*) in California's Central Valley using [CHIRPS v3.0](https://www.chc.ucsb.edu/data/chirps3) satellite precipitation and WMO-standard SPI. Every metric is evaluated at the monthly level — 60 independent test months (2021–2025) — against three naive baselines, with bootstrap confidence intervals.

**Key finding:** No model — from logistic regression to ConvLSTM — outperforms climatology in Brier Skill Score, even though ranking signal exists (ROC-AUC ~0.68). This is not a model failure; it is a **predictability barrier**. Monthly SPI-1 in a Mediterranean climate is fundamentally driven by chaotic synoptic events (atmospheric rivers, frontal passages) at 1-month lead, making the base rate the best available probability estimate.

> Full research assessment, literature context, and strategic roadmap: [`ANALYSIS.md`](ANALYSIS.md)

---

## Why This Matters

- **Agricultural stakes are real.** Central Valley produces over half of U.S. fruits and vegetables; drought losses reach billions per year. Knowing what *cannot* be predicted is as operationally valuable as a positive forecast.
- **Most published drought ML overstates skill.** Label leakage through SPI accumulation windows, spatial pseudo-replication, and missing baselines inflate reported accuracy. This project eliminates all three — and shows what remains.
- **Negative results have scientific value.** Demonstrating a predictability ceiling with rigorous evidence helps the community redirect effort toward problems where ML *can* add value (e.g., seasonal horizons, exogenous climate drivers).

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

**Data:** CHIRPS v3.0, 0.05° (~5 km), monthly, 1991–2025

**Temporal split (strictly chronological, no shuffling):**

| Split | Period | Purpose |
|-------|--------|---------|
| Train | 1991–2016 | 26 years of model learning |
| Validation | 2017–2020 | Calibration selection only |
| Test | 2021–2025 | 60 months, never seen during training or calibration |

**Leakage-free target design:** The target is `SPI-1[t+1]`, which depends *only* on `pr[t+1]` — unknown at prediction time. All features are derived from time *t* or earlier. Unlike SPI-3-based targets used in much of the literature, there is **zero accumulation-window overlap** between features and target.

---

## Results

### Skill Scores (60 test months, monthly level)

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
> XGBoost-Spatial comes closest (BSS = −0.03); 95% bootstrap CI spans zero.

### ML Interpretation

1. **Models detect drought signal.** ROC-AUC ~0.68 means the models rank months by drought risk better than chance — there *is* learnable structure in the features.
2. **But ranking ≠ calibrated probability.** BSS stays negative because the models cannot produce probability estimates more reliable than the climatological base rate. The signal exists but is too weak to improve upon always predicting "25% chance of drought."
3. **The bottleneck is resolution, not reliability.** Brier Score decomposition (Murphy 1973) shows models achieve near-zero resolution — they cannot reliably distinguish drought months from non-drought months in advance. Post-hoc calibration (isotonic/Platt) was tested and confirmed: you cannot calibrate your way past a discrimination barrier.
4. **This is physically consistent.** SPI-1 autocorrelation in Central Valley is weak (r ≈ 0.1–0.3). Persistence fails badly (BSS = −0.57). Monthly precipitation is governed by atmospheric rivers and frontal passages — chaotic processes at 1-month lead.

### SHAP Explainability

SHAP TreeExplainer confirms the model learned real hydroclimatic relationships, not artifacts:
- Negative `spi1_lag1` strongly increases dry-class probability — past drought carries forward
- A sharp nonlinear threshold near SPI-1 ≈ 0 mirrors the physical transition from normal to deficit
- `spi3_lag1` provides medium-term drought memory; seasonal features capture wet/dry season transitions

The model *understands* drought dynamics. It simply cannot overcome the chaotic nature of next-month precipitation.

---

## What Makes This Rigorous

Most drought ML studies report high accuracy but few test whether they actually beat a naive baseline. This project implements practices that are uncommon in the literature but essential for valid conclusions:

- **SPI-1 target** eliminates the accumulation-window leakage that inflates accuracy in SPI-3-based studies
- **Monthly-level metrics** treat the 60 independent months as the effective sample size — not ~400k spatially autocorrelated pixels
- **Three naive baselines** (climatology, persistence, SPI-1 heuristic) provide the necessary reference points
- **Bootstrap confidence intervals** (2,000 iterations) quantify uncertainty on all skill scores
- **Brier Score decomposition** pinpoints *where* models fail (resolution vs. reliability)
- **Cross-dataset validation** against ERA5-Land SPI-1 confirms results are not an artifact of the CHIRPS product
- **USDM comparison framed as qualitative** — correctly avoids treating a composite drought index as ground truth for a precipitation-only forecast

---

## Reproducibility

The full pipeline is reproducible from raw CHIRPS data. Scripts in `scripts/` run sequentially:

1. Download & clip CHIRPS → 2. Compute SPI & drought labels → 3. Build tabular forecast dataset → 4. Train models (LogReg, RF, XGBoost, XGBoost-Spatial, ConvLSTM) → 5. SHAP analysis → 6. Skill evaluation & calibration study → 7. ERA5-Land validation → 8. Spatial maps & case studies

See `scripts/` for the complete, ordered pipeline. Environment setup: `environment.yml` / `requirements.txt`.

---

## Limitations and Research Directions

**Current scope limitations:**
- **Single region** — the predictability barrier may be specific to Central Valley's Mediterranean hydroclimate, or it may be universal. A single-region study cannot distinguish these.
- **Purely endogenous features** — all predictors derive from CHIRPS precipitation. Large-scale climate drivers (ENSO, PDO) and other variables (temperature, soil moisture) are not included.
- **Test period extremes** — 2021–2025 includes a historic drought followed by extreme wet reversal, which is not climatologically representative.

**Highest-impact next steps** (see [`ANALYSIS.md`](ANALYSIS.md) for the full strategic roadmap):

| Direction | What it tests | Why it matters |
|-----------|---------------|----------------|
| **Add ENSO (Niño 3.4) as a feature** | Can large-scale climate state break the predictability barrier? | If ENSO doesn't help, it strengthens the "intrinsic chaos" interpretation |
| **Expand to additional regions** (Murray–Darling, Great Plains) | Does "no skill" generalize across hydroclimates? | Transforms a regional finding into a general scientific result |
| **Stratified BSS by season and ENSO phase** | Is there conditional skill masked in overall averages? | Reveals whether some conditions are predictable even if the mean is not |
| **Seasonal target (SPI-3 at 3-month lead)** | Do longer aggregation windows improve predictability? | Aligns with operational drought forecasting timescales |

---

## Acknowledgement

AI tools (ChatGPT, Gemini, GitHub Copilot) were used ethically for project design, code development, and documentation.

---

## Author

Md Ishtiaque Hossain \
MSc, Computer and Information Sciences \
University of Delaware \
[LinkedIn](https://linkedin.com/in/ishtiaque-h) · [GitHub](https://github.com/Ishtiaque-h)
