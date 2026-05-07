# Literature synthesis memo (supports claims + next experiments)

Date: 2026-05-06

## What the new full texts strengthen (direct, quotable support)

1. **A 1‑month lead for meteorological drought is often beyond the “useful” subseasonal window.**
   - Su et al. (2023) evaluate SubX-driven drought onset/termination and report usable skill mainly at **weeks 1–2**, **limited** skill at **week 3**, and **essentially no skill at week 4** across many severities.
   - Lesinger et al. (2024) evaluate SubX predictors relevant to flash drought and report moderate–high ACC at **lead week 1** for ETo forcing variables, but **low predictability by weeks 3–4** (ACC < 0.5 for forcing variables in their abstract).

2. **Probabilistic verification needs BSS + reliability/resolution framing (not just accuracy/AUC).**
   - Becker & van den Dool (2016) assess NMME tercile probabilities using **cross-validated hindcasts** with **Brier skill score**, and explicitly discuss **reliability** and **resolution** (including the standard Brier decomposition).

3. **Teleconnection signal can be real but “episodic” (forecast-of-opportunity) and at shorter lags.**
   - Malloy & Kirtman (2023) find the lag between BSISO-related circulation anomalies and Great Plains rainfall anomalies is about **2 weeks**, and explicitly describe this as a **forecast of opportunity**. This supports a timescale-mismatch interpretation: strong teleconnection diagnostics do not guarantee stable **monthly** SPI‑1 lead‑1 probability skill.

4. **Target memory matters: low-memory SPI‑1 is the hard case.**
   - Sutanto & Van Lanen (2022) attribute hydrological drought forecast performance to **catchment memory**, supporting the claim that weak SPI‑1 skill can be physically plausible and motivating memory-bearing targets (SPI‑6/SPI‑12, soil moisture drought, etc.).

## How this maps to our manuscript positioning

- **Strengthens the current primary claim (predictability + evaluation audit):**
  - Literature supports both the **expected skill decay beyond ~weeks 2–4** and the need for **probabilistic, climatology-relative verification**.
  - This makes our “ranking signal without robust BSS” result easier to defend as a physically consistent outcome under strict evaluation.

- **Enables a stronger *positive-skill* hypothesis without overclaiming today:**
  - Positive skill is more likely when the target and inputs encode **physical memory** and/or **dynamic forecast information** (Lesinger & Tian 2025; Hwang et al. 2019; SubseasonalClimateUSA).

## Stronger, implementable hypotheses (minimal engineering)

### H1 — Transition targets are more predictable than monthly state
Instead of predicting “dry fraction next month” (state), predict **event transitions**:
- **Onset:** non-dry → dry
- **Termination:** dry → non-dry

Why this is defendable:
- Su et al. (2023) explicitly studies onset/termination and reports different skill behavior by lead and event type.

Why this is implementable here:
- We already compute SPI labels per pixel and have the leakage-safe time indexing; onset/termination can be derived from the existing label time series without new data sources.

Status after implementation:
- Rectangular Central Valley onset has a small robust-positive result only under
  the stricter eligible-pixel framing: eligible-only XGBoost isotonic `BSS =
  +0.104`, CI `[+0.025, +0.167]`, monthly Spearman `0.402`.
- Replication weakens the hypothesis: Central Valley basin-masked onset is
  negative/uncertain (`BSS = -0.027`), Southern Great Plains basin-masked onset
  is robust-negative (`BSS = -0.084`), and Mediterranean Spain basin-masked
  onset is near climatology (`BSS = -0.035`).
- Termination does not beat the eligible-state climatology: eligible-only
  XGBoost isotonic `BSS = -0.031`, CI crosses zero.
- Interpretation: onset remains a useful target-design diagnostic, but not a
  manuscript headline positive-skill result.

### H2 — Skill is conditional (forecast-of-opportunity), not stationary
Evaluate BSS *conditioned on regime*, e.g.:
- ENSO phase (already partially supported by existing stratified tables)
- season
- stronger/stricter “teleconnection-active” months (thresholded indices)

Why this is defendable:
- Malloy & Kirtman (2023) provide a clear forecast-of-opportunity mechanism framing.

Why this is implementable here:
- The repo already has climate-index download + stratification machinery; this can be extended with minimal additional tables.

## Concrete “next runs” using current scripts

- **Seasonal memory-bearing targets (already implemented):**
  - Use `scripts/run_seasonal_longlead_experiment.py` to scan regions/masks for SPI‑3 lead‑3 and SPI‑6 lead‑6 skill, then apply the existing event-tracking diagnostics to avoid mistaking calibration shift for true skill.

- **Land-surface + dynamic-model direction (data permitting):**
  - Use the existing ERA5-Land download/feature scripts as a stepping stone toward the Lesinger et al. (2024) / Lesinger & Tian (2025) setup; the literature suggests the win is *forecast-informed* land-surface prediction rather than “more CHIRPS lags.”

## Key citations now locally available

These are downloaded and available under `literature/papers/` (see `literature/related_works_index.csv`):
- Becker & van den Dool (2016) — NMME probabilistic BSS + reliability/resolution baseline
- Su et al. (2023) — SubX drought onset/termination skill decay by week 3–4
- Lesinger et al. (2024) — SubX predictors (ETo/RZSM) lose predictability by weeks 3–4
- Malloy & Kirtman (2023) — BSISO teleconnection lag ~2 weeks (forecast-of-opportunity)
