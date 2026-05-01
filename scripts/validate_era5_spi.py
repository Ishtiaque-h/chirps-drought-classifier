#!/usr/bin/env python
"""
Cross-dataset validation: CHIRPS-trained model vs. ERA5-Land SPI-1.

Methodological rationale:
  The primary quantitative external validation uses ERA5-Land monthly
  precipitation (independent reanalysis product) to compute SPI-1 over
  the same domain and period. Because both CHIRPS and ERA5-Land are
  precipitation-derived, the comparison is methodologically consistent:
  any skill score (BSS, HSS) reflects the model's ability to generalise
  across precipitation datasets, not cross-product definitional differences.

  Contrast with USDM: USDM D1+ integrates soil moisture, streamflow, and
  observer reports, so correlating it with a pure-precipitation SPI is
  confounded by data-source differences and should only be treated as a
  qualitative consistency check (see validate_usdm.py).

Workflow:
  1. Load ERA5-Land monthly total precipitation over Central Valley bbox.
     (Expected file: data/processed/era5_land_monthly_cvalley_<START_YEAR>_<CURRENT_YEAR>.nc
      with variable 'tp', units m/month → converted to mm/month)
  2. Compute SPI-1 using the same gamma-fit methodology as make_spi_labels.py
     (1991–2020 baseline, per-calendar-month gamma fit, zero-probability handling).
  3. Derive drought_label_spi1_era5 from ERA5 SPI-1 (same ±1 thresholds).
  4. Compute monthly regional dominant class from ERA5 labels.
  5. Load the XGBoost model, compute predictions on the CHIRPS test set.
  6. Compare monthly dominant-class predictions against ERA5-Land labels.
  7. Report BSS and HSS; save comparison plot and metrics text.

Inputs:
  data/processed/era5_land_monthly_cvalley_<START_YEAR>_<CURRENT_YEAR>.nc
  data/processed/dataset_forecast.parquet
  outputs/forecast_xgb_model.json

Outputs:
  outputs/era5_validation_metrics.txt
  outputs/era5_validation_comparison.png
"""
from pathlib import Path
from datetime import datetime, timezone
import sys
import subprocess
import numpy as np
import pandas as pd
import xarray as xr
import xgboost as xgb
import matplotlib.pyplot as plt
from scipy.stats import gamma as gamma_dist, norm
from feature_config import get_feature_columns

PROJECT_ROOT = Path(__file__).resolve().parents[1]

START_YEAR = 1991
CURRENT_YEAR = datetime.now(timezone.utc).year

ERA5_FILE = PROJECT_ROOT / "data" / "processed" / f"era5_land_monthly_cvalley_{START_YEAR}_{CURRENT_YEAR}.nc"
DATA = PROJECT_ROOT / "data" / "processed" / "dataset_forecast.parquet"
MODEL_PATH = PROJECT_ROOT / "outputs" / "forecast_xgb_model.json"
OUT_DIR = PROJECT_ROOT / "outputs"
OUT_DIR.mkdir(exist_ok=True)

DOWNLOAD_SCRIPT = PROJECT_ROOT / "scripts" / "download_era5_land_monthly.py"

BASELINE_START = 1991
BASELINE_END   = 2020
TEST_START     = 2021

TARGET = "target_label"
LABEL_MAP     = {-1: 0, 0: 1, 1: 2}
INV_LABEL_MAP = {v: k for k, v in LABEL_MAP.items()}
CLASSES       = [-1, 0, 1]

def ensure_era5_file() -> None:
    """
    Ensure the ERA5 file exists.
    If missing, automatically run the download script.
    """
    if ERA5_FILE.exists():
        return

    print(f"ERA5-Land file not found: {ERA5_FILE}")
    print("Running download script automatically...")

    if not DOWNLOAD_SCRIPT.exists():
        raise FileNotFoundError(
            f"Download script not found: {DOWNLOAD_SCRIPT}"
        )

    try:
        subprocess.run(
            [sys.executable, str(DOWNLOAD_SCRIPT)],
            cwd=PROJECT_ROOT,
            check=True,
        )
    except subprocess.CalledProcessError as e:
        raise RuntimeError(
            "Automatic ERA5 download failed. "
            "Please check the download script output above."
        ) from e

    if not ERA5_FILE.exists():
        raise FileNotFoundError(
            "Download script finished, but the ERA5 file was still not created:\n"
            f"{ERA5_FILE}"
        )

    print("ERA5-Land file downloaded successfully.")


def choose_precip_var(ds: xr.Dataset) -> str:
    """Return the ERA5 precipitation variable name."""
    for candidate in ["tp", "total_precipitation"]:
        if candidate in ds.data_vars:
            return candidate
    raise ValueError(
        f"Could not find precipitation variable in ERA5 file. "
        f"Available variables: {list(ds.data_vars)}"
    )


def combine_first_over_dim(da: xr.DataArray, dim: str) -> xr.DataArray:
    """
    Merge slices across an extra dimension by taking the first non-null value.
    Useful when a malformed open_mfdataset result created an outer source dimension
    and aligned the real timestamps on another axis.
    """
    if dim not in da.dims:
        return da

    merged = da.isel({dim: 0})
    for i in range(1, da.sizes[dim]):
        merged = merged.combine_first(da.isel({dim: i}))
    return merged


def standardize_era5_tp(ds: xr.Dataset) -> xr.DataArray:
    """
    Standardize ERA5 precipitation to exactly:
        (time, latitude, longitude)

    Handles:
    - lon/lat -> longitude/latitude
    - valid_time -> time
    - expver dimension
    - malformed 4D arrays from bad nested concat:
        (time, valid_time, latitude, longitude)
      where outer 'time' is really a source-file axis
    """
    tp_var = choose_precip_var(ds)
    tp = ds[tp_var]

    # Rename spatial coordinates if needed
    rename_map = {}
    if "lon" in tp.dims:
        rename_map["lon"] = "longitude"
    if "lat" in tp.dims:
        rename_map["lat"] = "latitude"
    if rename_map:
        tp = tp.rename(rename_map)

    # Collapse expver if present
    if "expver" in tp.dims:
        tp = tp.mean(dim="expver", skipna=True)

    # Case 1: only valid_time exists -> rename to time
    if "valid_time" in tp.dims and "time" not in tp.dims:
        tp = tp.rename({"valid_time": "time"})

    # Case 2: both time and valid_time exist
    # Usually this means the file was concatenated incorrectly and the outer
    # 'time' dimension is actually a file/source axis.
    elif "valid_time" in tp.dims and "time" in tp.dims:
        tp = tp.rename({"time": "source_file", "valid_time": "time"})
        tp = combine_first_over_dim(tp, "source_file")

    # Squeeze harmless singleton dims
    extra_dims = [d for d in tp.dims if d not in {"time", "latitude", "longitude"}]
    for dim in extra_dims:
        if tp.sizes[dim] == 1:
            tp = tp.squeeze(dim, drop=True)

    # If one extra non-singleton dimension still exists, try to merge it
    extra_dims = [d for d in tp.dims if d not in {"time", "latitude", "longitude"}]
    if len(extra_dims) == 1:
        tp = combine_first_over_dim(tp, extra_dims[0])

    required_dims = {"time", "latitude", "longitude"}
    if not required_dims.issubset(set(tp.dims)):
        raise ValueError(
            f"Unexpected ERA5 dimensions after standardization: {tp.dims}, shape={tp.shape}"
        )

    tp = tp.transpose("time", "latitude", "longitude")

    if tp.ndim != 3:
        raise ValueError(
            f"Expected 3D precipitation array, got dims={tp.dims}, shape={tp.shape}"
        )

    # Sort by time and drop duplicate timestamps if any
    time_index = pd.DatetimeIndex(tp["time"].values)
    keep = ~time_index.duplicated()
    tp = tp.isel(time=keep).sortby("time")

    return tp


def precip_to_spi(series: np.ndarray, baseline_mask: np.ndarray) -> np.ndarray:
    """Gamma-fit SPI (same method as make_spi_labels.py)."""
    base = series[baseline_mask]
    base = base[~np.isnan(base)]

    spi = np.full(len(series), np.nan, dtype=np.float64)

    if len(base) < 10:
        return spi

    p_zero = np.mean(base == 0)
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
            cdf_val = p_zero + (1.0 - p_zero) * gamma_dist.cdf(
                val, fit_alpha, scale=fit_beta
            )

        cdf_val = np.clip(cdf_val, 1e-6, 1 - 1e-6)
        spi[i] = norm.ppf(cdf_val)

    return spi


def brier_score(y_true_bin: np.ndarray, prob: np.ndarray) -> float:
    return float(np.mean((prob - y_true_bin) ** 2))


def hss(y_true: np.ndarray, y_pred: np.ndarray, classes: list[int]) -> float:
    from sklearn.metrics import confusion_matrix as cm

    mat = cm(y_true, y_pred, labels=classes)
    total = mat.sum()
    if total == 0:
        return np.nan

    correct = np.diag(mat).sum()
    expected = (mat.sum(axis=1) * mat.sum(axis=0)).sum() / total
    denom = total - expected

    return float((correct - expected) / denom) if denom > 0 else np.nan


# ── check ERA5 file ───────────────────────────────────────────────────────────
ensure_era5_file()

# ── load ERA5-Land ────────────────────────────────────────────────────────────
print("Loading ERA5-Land precipitation...")
era5_ds = xr.open_dataset(ERA5_FILE).load()
tp = standardize_era5_tp(era5_ds)

print(f"Standardized ERA5 dims: {tp.dims}")
print(f"Standardized ERA5 shape: {tp.shape}")
print(f"ERA5 time range: {pd.Timestamp(tp.time.values[0])} -> {pd.Timestamp(tp.time.values[-1])}")

# Convert precipitation units if necessary
# ERA5-Land monthly-means total precipitation is typically in meters.
units = str(tp.attrs.get("units", "")).strip().lower()
if units in {"m", "meter", "meters", "metre", "metres"} or units == "":
    tp = tp * 1000.0
    tp.attrs["units"] = "mm"

times  = pd.DatetimeIndex(tp.time.values)
months = times.month
nlat   = tp.sizes["latitude"]
nlon   = tp.sizes["longitude"]
ntimes = len(times)

baseline_mask = (times.year >= BASELINE_START) & (times.year <= BASELINE_END)
tp_vals = tp.values.astype(np.float64)  # (time, lat, lon)

# ── compute ERA5-Land SPI-1 ───────────────────────────────────────────────────
print("Computing ERA5-Land SPI-1 (this may take a few minutes)...")
spi1_era5 = np.full(tp_vals.shape, np.nan, dtype=np.float32)

for m in range(1, 13):
    month_idx = np.where(months == m)[0]
    base_idx  = np.where(baseline_mask & (months == m))[0]

    month_baseline_mask = np.zeros(len(month_idx), dtype=bool)
    month_baseline_positions = np.where(np.isin(month_idx, base_idx))[0]
    month_baseline_mask[month_baseline_positions] = True

    for j in range(nlat):
        for k in range(nlon):
            series_m = tp_vals[month_idx, j, k]
            spi_m = precip_to_spi(series_m, month_baseline_mask)
            spi1_era5[month_idx, j, k] = spi_m.astype(np.float32)

    print(f"  ERA5 SPI-1 month {m:02d}/12 done")

# drought labels
label_era5 = np.where(
    spi1_era5 <= -1.0, -1,
    np.where(spi1_era5 >= 1.0, 1, 0)
).astype(np.int8)

# keep masked / missing locations neutral for dominant-class aggregation
label_era5[np.isnan(spi1_era5)] = 0

# ── aggregate ERA5 labels to monthly regional dominant class ──────────────────
print("Aggregating ERA5 to monthly dominant class...")
test_mask = times.year >= TEST_START
test_times = times[test_mask]

era5_monthly = {}
for t_idx, ts in zip(np.where(test_mask)[0], test_times):
    valid_mask = ~np.isnan(spi1_era5[t_idx].ravel())
    flat = label_era5[t_idx].ravel()[valid_mask]

    if len(flat) == 0:
        continue

    vals, counts = np.unique(flat, return_counts=True)
    dominant = int(vals[counts.argmax()])
    era5_monthly[ts.to_period("M").to_timestamp()] = dominant

era5_series = pd.Series(era5_monthly, name="era5_dominant")

# ── load model and get XGBoost predictions ────────────────────────────────────
print("Loading XGBoost model and CHIRPS test predictions...")
df = pd.read_parquet(DATA)
df["year"] = df["year"].astype(int)
FEATURES = get_feature_columns(df.columns)
test = df[df["year"] >= TEST_START].copy()

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

model = xgb.Booster()
model.load_model(MODEL_PATH.as_posix())

dtest = xgb.DMatrix(test[FEATURES], feature_names=FEATURES)
probs = model.predict(dtest)
test["pred_label"] = [INV_LABEL_MAP[i] for i in probs.argmax(axis=1)]

test["month_dt"] = (
    pd.to_datetime(test["time"]) + pd.DateOffset(months=1)
).dt.to_period("M").dt.to_timestamp()

chirps_monthly = test.groupby("month_dt").agg(
    chirps_pred         = ("pred_label", lambda s: int(s.mode()[0])),
    chirps_obs_dry_frac = (TARGET, lambda s: (s == -1).mean()),
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
obs_dry_era5  = (merged["era5_dominant"] == -1).astype(float).values
pred_dry_frac = merged["chirps_pred_dry_frac"].values

df_train   = df[df["year"] <= 2016]
clim_dry_p = (df_train[TARGET] == -1).mean()

bs_clim  = brier_score(obs_dry_era5, np.full(len(obs_dry_era5), clim_dry_p))
bs_model = brier_score(obs_dry_era5, pred_dry_frac)
bss_era5 = 1.0 - bs_model / bs_clim if bs_clim > 0 else np.nan

# HSS
hss_val = hss(
    merged["era5_dominant"].values,
    merged["chirps_pred"].values,
    CLASSES
)

print("\nERA5-Land SPI-1 cross-dataset validation")
print(f"  BSS (dry class, ref=climatology): {bss_era5:.4f}")
print(f"  HSS (3-class monthly dominant)  : {hss_val:.4f}")

# ── plot ──────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(
    merged.index,
    merged["chirps_pred_dry_frac"],
    label="Model — CHIRPS dry fraction",
    linewidth=1.8
)
ax.plot(
    merged.index,
    (merged["era5_dominant"] == -1).astype(float),
    label="ERA5-Land SPI-1 — dry class (0/1)",
    linewidth=1.8,
    linestyle="--",
    drawstyle="steps-post"
)
ax.set_xlabel("Month (target)")
ax.set_ylabel("Fraction / indicator")
ax.set_title(
    "Cross-dataset validation: CHIRPS-based model vs. ERA5-Land SPI-1\n"
    f"Central Valley {TEST_START}–{CURRENT_YEAR}  |  BSS={bss_era5:.3f}  HSS={hss_val:.3f}",
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
    f"ERA5-Land Cross-Dataset Validation — Central Valley {TEST_START}–{CURRENT_YEAR}\n"
    + "=" * 63 + "\n"
    + "Methodology: SPI-1 computed from ERA5-Land monthly precipitation using\n"
    + "  the same gamma-fit / WMO-threshold approach as the CHIRPS training data.\n"
    + "  Both products are precipitation-based → skill scores are directly comparable.\n\n"
    + f"Overlapping test months       : {len(merged)}\n"
    + f"Brier Score (model, dry)      : {bs_model:.4f}\n"
    + f"Brier Score (climatology ref) : {bs_clim:.4f}\n"
    + f"Brier Skill Score (BSS)       : {bss_era5:.4f}  (>0 = better than climatology)\n"
    + f"Heidke Skill Score (HSS)      : {hss_val:.4f}  (>0 = better than random)\n"
)

print(metrics_txt)
metrics_path = OUT_DIR / "era5_validation_metrics.txt"
metrics_path.write_text(metrics_txt)
print("Wrote:", metrics_path)
