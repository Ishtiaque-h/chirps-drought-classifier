#!/usr/bin/env python
"""
Cross-dataset validation: CHIRPS-trained model vs. ERA5-Land SPI-1.

Methodological rationale (Q1/Q2 journal standard):
  The primary quantitative external validation uses ERA5-Land monthly
  precipitation (independent reanalysis product) to compute SPI-1 over
  the same domain and period.  Because both CHIRPS and ERA5-Land are
  precipitation-derived, the comparison is methodologically consistent:
  any skill score (BSS, HSS) reflects the model's ability to generalise
  across precipitation datasets, not cross-product definitional differences.

  Contrast with USDM: USDM D1+ integrates soil moisture, streamflow, and
  observer reports, so correlating it with a pure-precipitation SPI is
  confounded by data-source differences and should only be treated as a
  qualitative consistency check (see validate_usdm.py).

Workflow:
  1. Load ERA5-Land monthly total precipitation over Central Valley bbox.
     (Expected file: data/processed/era5_land_monthly_cvalley_1991_2025.nc
      with variable 'tp', units m/month → converted to mm/month)
     If the ERA5 file is absent, the script prints instructions and exits.
  2. Compute SPI-1 using the same gamma-fit methodology as make_spi_labels.py
     (1991–2020 baseline, per-calendar-month gamma fit, zero-probability handling).
  3. Derive drought_label_spi1_era5 from ERA5 SPI-1 (same ±1 thresholds).
  4. Compute monthly regional dominant class from ERA5 labels.
  5. Load the XGBoost model, compute predictions on the CHIRPS test set.
  6. Compare monthly dominant-class predictions against ERA5-Land labels.
  7. Report BSS and HSS; save comparison plot and metrics text.

ERA5-Land download instructions (CDS API):
  import cdsapi
  c = cdsapi.Client()
  c.retrieve('reanalysis-era5-land-monthly-means', {
      'product_type': 'monthly_averaged_reanalysis',
      'variable': 'total_precipitation',
      'year':  [str(y) for y in range(1991, 2026)],
      'month': [f'{m:02d}' for m in range(1, 13)],
      'time':  '00:00',
      'area':  [40.6, -122.5, 35.4, -119.0],   # N, W, S, E
      'format': 'netcdf',
  }, 'data/processed/era5_land_monthly_cvalley_1991_2025.nc')

Inputs:
  data/processed/era5_land_monthly_cvalley_1991_2025.nc  (ERA5-Land, user-provided)
  data/processed/dataset_forecast.parquet
  outputs/forecast_xgb_model.json

Outputs:
  outputs/era5_validation_metrics.txt
  outputs/era5_validation_comparison.png
"""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import xarray as xr
import xgboost as xgb
import matplotlib.pyplot as plt
from scipy.stats import gamma as gamma_dist, norm

ERA5_FILE  = Path("data/processed/era5_land_monthly_cvalley_1991_2025.nc")
DATA       = Path("data/processed/dataset_forecast.parquet")
MODEL_PATH = Path("outputs/forecast_xgb_model.json")
OUT_DIR    = Path("outputs"); OUT_DIR.mkdir(exist_ok=True)

BASELINE_START = 1991
BASELINE_END   = 2020
TEST_START     = 2021

FEATURES = [
    "spi1_lag1", "spi1_lag2", "spi1_lag3",
    "spi3_lag1", "spi6_lag1",
    "pr_lag1", "pr_lag2", "pr_lag3",
    "month_sin", "month_cos",
]
TARGET = "target_label"
LABEL_MAP     = {-1: 0, 0: 1, 1: 2}
INV_LABEL_MAP = {v: k for k, v in LABEL_MAP.items()}
CLASSES       = [-1, 0, 1]

# ── check ERA5 file ───────────────────────────────────────────────────────────
if not ERA5_FILE.exists():
    print(
        f"ERA5-Land file not found: {ERA5_FILE}\n\n"
        "To generate it, run the following using the CDS API:\n\n"
        "  import cdsapi\n"
        "  c = cdsapi.Client()\n"
        "  c.retrieve('reanalysis-era5-land-monthly-means', {\n"
        "      'product_type': 'monthly_averaged_reanalysis',\n"
        "      'variable': 'total_precipitation',\n"
        "      'year':  [str(y) for y in range(1991, 2026)],\n"
        "      'month': [f'{m:02d}' for m in range(1, 13)],\n"
        "      'time':  '00:00',\n"
        "      'area':  [40.6, -122.5, 35.4, -119.0],\n"
        "      'format': 'netcdf',\n"
        "  }, 'data/processed/era5_land_monthly_cvalley_1991_2025.nc')\n\n"
        "Place the downloaded file at the path above and re-run this script."
    )
    sys.exit(0)

# ── load ERA5-Land ────────────────────────────────────────────────────────────
print("Loading ERA5-Land precipitation...")
era5_ds = xr.open_dataset(ERA5_FILE).load()

# variable may be 'tp' or 'total_precipitation'
tp_var = "tp" if "tp" in era5_ds else "total_precipitation"
tp = era5_ds[tp_var]

# ERA5-Land monthly-means product: 'tp' is m (accumulated over the month).
# Convert m → mm by multiplying by 1000.
# The 'm d**-1' (per-day mean) product would need an additional × days-in-month
# step; that product is not the monthly-means download, so we do not handle it
# here — users should use 'monthly_averaged_reanalysis' (not 'monthly_averaged_
# reanalysis_by_hour_of_day') as specified in the download instructions above.
if tp.attrs.get("units", "m") == "m":
    tp = tp * 1000.0
    tp.attrs["units"] = "mm"

# standardise coordinate names
if "longitude" not in tp.dims and "lon" in tp.dims:
    tp = tp.rename({"lon": "longitude", "lat": "latitude"})

times   = pd.DatetimeIndex(tp.time.values)
months  = times.month
nlat    = tp.sizes["latitude"]
nlon    = tp.sizes["longitude"]
ntimes  = len(times)

baseline_mask = (times.year >= BASELINE_START) & (times.year <= BASELINE_END)
tp_vals       = tp.values  # (time, lat, lon)

# ── reuse SPI function from make_spi_labels.py ────────────────────────────────
def precip_to_spi(series: np.ndarray, baseline_mask: np.ndarray) -> np.ndarray:
    """Gamma-fit SPI (identical to make_spi_labels.py)."""
    base  = series[baseline_mask]
    base  = base[~np.isnan(base)]
    spi   = np.full(len(series), np.nan)
    if len(base) < 10:
        return spi
    p_zero  = np.mean(base == 0)
    nonzero = base[base > 0]
    if len(nonzero) < 5:
        return spi
    fit_alpha, _, fit_beta = gamma_dist.fit(nonzero, floc=0)
    for i, val in enumerate(series):
        if np.isnan(val):
            continue
        if val == 0:
            cdf_val = p_zero
        else:
            cdf_val = p_zero + (1.0 - p_zero) * gamma_dist.cdf(val, fit_alpha, scale=fit_beta)
        cdf_val = np.clip(cdf_val, 1e-6, 1 - 1e-6)
        spi[i]  = norm.ppf(cdf_val)
    return spi

# ── compute ERA5-Land SPI-1 ───────────────────────────────────────────────────
print("Computing ERA5-Land SPI-1 (this may take a few minutes)...")
spi1_era5 = np.full_like(tp_vals, np.nan, dtype=np.float32)

for m in range(1, 13):
    month_idx    = np.where(months == m)[0]
    base_idx     = np.where(baseline_mask & (months == m))[0]
    mask_month   = np.zeros(ntimes, dtype=bool)
    mask_month[base_idx] = True
    base_within  = mask_month[month_idx]

    for j in range(nlat):
        for k in range(nlon):
            series_m         = tp_vals[month_idx, j, k].astype(np.float64)
            spi_m            = precip_to_spi(series_m, base_within)
            spi1_era5[month_idx, j, k] = spi_m.astype(np.float32)
    print(f"  ERA5 SPI-1 month {m:02d}/12 done")

# drought label
label_era5 = np.where(spi1_era5 <= -1.0, -1,
              np.where(spi1_era5 >=  1.0,  1, 0)).astype(np.int8)
label_era5[np.isnan(spi1_era5)] = 0

# ── aggregate ERA5 labels to monthly regional dominant class ──────────────────
print("Aggregating ERA5 to monthly dominant class...")
test_mask  = times.year >= TEST_START
test_times = times[test_mask]

era5_monthly = {}
for t, ts in zip(np.where(test_mask)[0], test_times):
    flat   = label_era5[t].ravel()
    flat   = flat[~np.isnan(spi1_era5[t].ravel())]
    if len(flat) == 0:
        continue
    vals, counts = np.unique(flat, return_counts=True)
    dominant     = int(vals[counts.argmax()])
    era5_monthly[ts.to_period("M").to_timestamp()] = dominant

era5_series = pd.Series(era5_monthly, name="era5_dominant")

# ── load model and get XGBoost predictions ────────────────────────────────────
print("Loading XGBoost model and CHIRPS test predictions...")
df   = pd.read_parquet(DATA)
df["year"] = df["year"].astype(int)
test = df[df["year"] >= TEST_START].copy()

assert MODEL_PATH.exists(), f"Model not found: {MODEL_PATH}"
model = xgb.Booster()
model.load_model(MODEL_PATH.as_posix())

dtest = xgb.DMatrix(test[FEATURES], feature_names=FEATURES)
probs = model.predict(dtest)
test["pred_label"] = [INV_LABEL_MAP[i] for i in probs.argmax(axis=1)]

# monthly dominant prediction and observation from CHIRPS
test["month_dt"] = (
    pd.to_datetime(test["time"]) + pd.DateOffset(months=1)
).dt.to_period("M").dt.to_timestamp()

chirps_monthly = test.groupby("month_dt").agg(
    chirps_pred    = ("pred_label",  lambda s: int(s.mode()[0])),
    chirps_obs_dry_frac = (TARGET,  lambda s: (s == -1).mean()),
    chirps_pred_dry_frac= ("pred_label", lambda s: (s == -1).mean()),
).reset_index()

# ── merge and compare ─────────────────────────────────────────────────────────
merged = chirps_monthly.set_index("month_dt").join(
    era5_series, how="inner"
).dropna()

print(f"Overlapping test months: {len(merged)}")
if len(merged) < 5:
    print("Too few overlapping months. Check ERA5 file time range.")
    sys.exit(0)

# BSS
def brier_score(y_true_bin, prob):
    return float(np.mean((prob - y_true_bin) ** 2))

obs_dry_era5  = (merged["era5_dominant"] == -1).astype(float).values
pred_dry_frac = merged["chirps_pred_dry_frac"].values

# climatological reference: training-set dry fraction
df_train   = df[df["year"] <= 2016]
clim_dry_p = (df_train[TARGET] == -1).mean()
bs_clim    = brier_score(obs_dry_era5, np.full(len(obs_dry_era5), clim_dry_p))
bs_model   = brier_score(obs_dry_era5, pred_dry_frac)
bss_era5   = 1.0 - bs_model / bs_clim if bs_clim > 0 else np.nan

# HSS
def hss(y_true, y_pred, classes):
    from sklearn.metrics import confusion_matrix as cm
    mat   = cm(y_true, y_pred, labels=classes)
    total = mat.sum()
    if total == 0:
        return np.nan
    correct  = np.diag(mat).sum()
    expected = (mat.sum(axis=1) * mat.sum(axis=0)).sum() / total
    denom    = total - expected
    return float((correct - expected) / denom) if denom > 0 else np.nan

hss_val = hss(merged["era5_dominant"].values,
              merged["chirps_pred"].values, CLASSES)

print(f"\nERA5-Land SPI-1 cross-dataset validation")
print(f"  BSS (dry class, ref=climatology): {bss_era5:.4f}")
print(f"  HSS (3-class monthly dominant)  : {hss_val:.4f}")

# ── plot ──────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(merged.index, merged["chirps_pred_dry_frac"],
        label="Model — CHIRPS dry fraction", linewidth=1.8)
ax.plot(merged.index,
        (merged["era5_dominant"] == -1).astype(float),
        label="ERA5-Land SPI-1 — dry class (0/1)", linewidth=1.8,
        linestyle="--", drawstyle="steps-post")
ax.set_xlabel("Month (target)")
ax.set_ylabel("Fraction / indicator")
ax.set_title(
    "Cross-dataset validation: CHIRPS-based model vs. ERA5-Land SPI-1\n"
    f"Central Valley 2021–2025  |  BSS={bss_era5:.3f}  HSS={hss_val:.3f}",
    fontsize=10,
)
ax.legend()
ax.set_ylim(-0.05, 1.05)
fig.tight_layout()
plot_path = OUT_DIR / "era5_validation_comparison.png"
fig.savefig(plot_path, dpi=150, bbox_inches="tight")
plt.close()
print("Wrote:", plot_path)

# ── text metrics ──────────────────────────────────────────────────────────────
metrics_txt = (
    "ERA5-Land Cross-Dataset Validation — Central Valley 2021–2025\n"
    "=" * 60 + "\n"
    "Methodology: SPI-1 computed from ERA5-Land monthly precipitation using\n"
    "  the same gamma-fit / WMO-threshold approach as the CHIRPS training data.\n"
    "  Both products are precipitation-based → skill scores are directly comparable.\n\n"
    f"Overlapping test months       : {len(merged)}\n"
    f"Brier Score (model, dry)      : {bs_model:.4f}\n"
    f"Brier Score (climatology ref) : {bs_clim:.4f}\n"
    f"Brier Skill Score (BSS)       : {bss_era5:.4f}  (>0 = better than climatology)\n"
    f"Heidke Skill Score (HSS)      : {hss_val:.4f}  (>0 = better than random)\n"
)
print(metrics_txt)
(OUT_DIR / "era5_validation_metrics.txt").write_text(metrics_txt)
print("Wrote:", OUT_DIR / "era5_validation_metrics.txt")
