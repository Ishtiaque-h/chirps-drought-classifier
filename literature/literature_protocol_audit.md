# Literature Protocol Audit

This audit is a guardrail for manuscript claims. It is not a systematic review
and it should not be used to label individual papers as invalid unless their
methods are audited in full. The purpose is narrower: compare target, lead,
predictors, validation, and metrics so our claims do not imply that successful
CHIRPS/SPI or drought-ML studies solved the exact task used here.

Machine-readable table: `literature/literature_protocol_audit.csv`.

## Main Findings

1. **Successful CHIRPS/SPI studies often target higher-memory quantities.**
   Annual SPI-12, SPI-3/SPI-6/SPI-12 regression, and regionalization are not
   equivalent to monthly SPI-1 dry-fraction probability at lead 1.

2. **Successful forecast-informed studies use richer information.**
   Seasonal/subseasonal studies usually use dynamical forecast systems,
   ensemble forecasts, land-surface states, remote sensing, or multi-driver
   benchmark datasets. They are not direct evidence that CHIRPS-lag-only SPI-1
   should beat climatology.

3. **The fair critique is protocol comparability, not blanket leakage.**
   We can show in our own audit that random row splits, pixel-level inference,
   and overlapping SPI accumulation windows inflate apparent skill. We should
   not claim that most prior work is leaky without a formal paper-by-paper
   methods audit.

4. **The strongest positive direction is target/input reframing.**
   Recent strong work supports root-zone soil moisture, hydrological drought,
   onset/termination, dynamic-model forecasts, and benchmark-first evaluation.

## Wording To Use

Use:

> Prior drought-ML studies are often not directly comparable because target
> accumulation scale, lead time, input information, validation split, metric,
> and climatology-relative baselines differ. Our contribution is a strict
> leakage-safe probability-skill audit for one hard target class: monthly
> SPI-1 dry fraction at lead 1, plus controlled extensions.

Avoid:

> Most drought-ML papers are leaky.

Avoid:

> CHIRPS drought forecasting is not possible.

Use instead:

> CHIRPS-derived monthly SPI-1 dry-fraction probability is hard to improve over
> climatology using lagged observed predictors alone; other targets and
> forecast-informed inputs can be more predictable.

## How This Changes The Paper Claim

The manuscript should not be framed as a broad refutation of CHIRPS drought
forecasting. It should be framed as a targeted predictability/evaluation audit:

- canonical target: monthly SPI-1 dry fraction at lead 1;
- strict design: no accumulation-window overlap, chronological split,
  validation-only calibration, monthly block inference;
- comparison: climatology, persistence, operational forecast benchmarks,
  temporal holdouts, source-cited masks, and target-reframing diagnostics;
- conclusion: lag-based ML rarely converts ranking signal into robust
  calibrated BSS for this target, while memory-bearing land-surface targets are
  more promising but still need persistence-relative forecast evidence.
