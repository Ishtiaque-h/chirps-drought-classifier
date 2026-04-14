#!/usr/bin/env python
"""
Compute WMO-standard Standardized Precipitation Index (SPI) labels.

For each calendar month and each pixel, fit a gamma distribution to the
1991–2020 baseline, transform values to SPI via the normal quantile function,
then compute SPI-1, SPI-3, and SPI-6.

WMO thresholds:
  SPI <= -1.0  →  dry   (-1)
  SPI >= +1.0  →  wet   (+1)
  otherwise    →  normal ( 0)

Input:
  data/processed/chirps_v3_monthly_cvalley_1991_2026.nc

Output:
  data/processed/chirps_v3_monthly_cvalley_spi_1991_2026.nc
    Variables: spi1, spi3, spi6, drought_label_spi1, drought_label_spi3

Note on drought_label_spi1:
  This is the scientifically preferred target for 1-month-ahead forecasting.
  Using SPI-1[t+1] as the target eliminates the accumulation-window overlap
  present when SPI-3[t+1] is used as the target (since SPI-3[t+1] shares two
  of its three accumulation months with features spi3[t] and pr_lag1/2).
  See: McKee et al. (1993); Dikshit et al. (2021, Sci. Total Environ.).
"""
from pathlib import Path
import numpy as np
import xarray as xr
from scipy.stats import gamma as gamma_dist, norm

IN_FILE = Path("data/processed/chirps_v3_monthly_cvalley_1991_2026.nc")
OUT_DIR = Path("data/processed")
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_FILE = OUT_DIR / "chirps_v3_monthly_cvalley_spi_1991_2026.nc"

BASELINE_START = "1991-01-01"
BASELINE_END   = "2020-12-31"

# ---------- load data ----------
print("Loading", IN_FILE)
ds = xr.open_dataset(IN_FILE).load()
pr = ds["pr"]  # (time, latitude, longitude)

times = pr.time.values
nlat  = pr.sizes["latitude"]
nlon  = pr.sizes["longitude"]
ntimes = len(times)

# ---------- helper: raw → SPI for a 1-D monthly time series ----------
def precip_to_spi(series: np.ndarray, baseline_mask: np.ndarray) -> np.ndarray:
    """
    Fit a gamma distribution to baseline values and return SPI for all values.

    Parameters
    ----------
    series        : 1-D array of length T (full period, may contain NaN)
    baseline_mask : boolean mask of length T, True for baseline years

    Returns
    -------
    spi : 1-D float array of length T
    """
    base = series[baseline_mask]
    base = base[~np.isnan(base)]
    spi = np.full(len(series), np.nan)

    if len(base) < 10:          # too few valid points → all NaN
        return spi

    # Probability of zero (some months can be dry)
    p_zero = np.mean(base == 0)
    nonzero = base[base > 0]

    if len(nonzero) < 5:        # almost all zeros → all NaN
        return spi

    # Fit gamma to nonzero baseline values; fix location at 0 (standard SPI)
    fit_alpha, _, fit_beta = gamma_dist.fit(nonzero, floc=0)

    for i, val in enumerate(series):
        if np.isnan(val):
            continue
        if val == 0:
            # cumulative probability up to zero
            cdf_val = p_zero
        else:
            # P(X <= val) = p_zero + (1-p_zero) * Gamma_CDF(val)
            cdf_val = p_zero + (1.0 - p_zero) * gamma_dist.cdf(val, fit_alpha, scale=fit_beta)
        # Clamp to avoid ppf(0) / ppf(1) = ±inf
        cdf_val = np.clip(cdf_val, 1e-6, 1 - 1e-6)
        spi[i] = norm.ppf(cdf_val)

    return spi


# ---------- pre-compute baseline mask (same for every pixel) ----------
import pandas as pd
time_pd = pd.DatetimeIndex(times)
baseline_mask = (time_pd.year >= 1991) & (time_pd.year <= 2020)
months_arr    = time_pd.month  # 1–12

# ---------- compute SPI-1 pixel-by-pixel per calendar month ----------
print("Computing SPI-1 (this may take a few minutes)...")
pr_vals = pr.values  # (time, lat, lon)

spi1_vals = np.full_like(pr_vals, np.nan, dtype=np.float32)

for m in range(1, 13):
    month_idx = np.where(months_arr == m)[0]
    base_idx  = np.where(baseline_mask & (months_arr == m))[0]
    mask_for_month = np.zeros(ntimes, dtype=bool)
    mask_for_month[base_idx] = True

    for j in range(nlat):
        for k in range(nlon):
            series_m = pr_vals[month_idx, j, k]
            # baseline sub-mask within this month's subset
            base_within_m = mask_for_month[month_idx]
            spi_m = precip_to_spi(series_m, base_within_m)
            spi1_vals[month_idx, j, k] = spi_m.astype(np.float32)
    print(f"  month {m:02d}/12 done")

print("SPI-1 done.  Computing rolling sums for SPI-3 and SPI-6...")

# ---------- SPI-3 and SPI-6 via rolling window on raw precipitation ----------
# Roll precipitation before applying the SPI transform, per the standard approach
def rolling_sum(arr3d: np.ndarray, window: int) -> np.ndarray:
    """Rolling sum along axis=0; leading (window-1) time steps → NaN."""
    out = np.full_like(arr3d, np.nan, dtype=np.float64)
    for t in range(window - 1, arr3d.shape[0]):
        out[t] = np.nansum(arr3d[t - window + 1 : t + 1], axis=0)
        # If any of the window slices are NaN, result should be NaN
        has_nan = np.any(np.isnan(arr3d[t - window + 1 : t + 1]), axis=0)
        out[t][has_nan] = np.nan
    return out

pr3_vals = rolling_sum(pr_vals.astype(np.float64), 3)
pr6_vals = rolling_sum(pr_vals.astype(np.float64), 6)

def compute_spi_from_rolled(rolled_vals, window):
    """Compute SPI on rolled (accumulated) precipitation."""
    spi_out = np.full_like(rolled_vals, np.nan, dtype=np.float32)
    for m in range(1, 13):
        month_idx = np.where(months_arr == m)[0]
        base_idx  = np.where(baseline_mask & (months_arr == m))[0]
        mask_for_month = np.zeros(ntimes, dtype=bool)
        mask_for_month[base_idx] = True
        base_within_m = mask_for_month[month_idx]
        for j in range(nlat):
            for k in range(nlon):
                series_m = rolled_vals[month_idx, j, k]
                spi_m = precip_to_spi(series_m, base_within_m)
                spi_out[month_idx, j, k] = spi_m.astype(np.float32)
        print(f"  SPI-{window} month {m:02d}/12 done")
    return spi_out

print("Computing SPI-3...")
spi3_vals = compute_spi_from_rolled(pr3_vals, 3)

print("Computing SPI-6...")
spi6_vals = compute_spi_from_rolled(pr6_vals, 6)

# ---------- drought labels from SPI-1 (primary forecast target) ----------
# SPI-1[t+1] is the scientifically preferred target for 1-month-ahead
# forecasting because it has zero accumulation-window overlap with the
# feature set (which includes spi1[t], spi3[t]=f(pr[t-2..t]), pr[t]).
label1_vals = np.where(spi1_vals <= -1.0, -1,
              np.where(spi1_vals >=  1.0,  1, 0)).astype(np.int8)
label1_vals[np.isnan(spi1_vals)] = 0

# ---------- drought labels from SPI-3 (kept for reference / v2 compatibility) ----------
label_vals = np.where(spi3_vals <= -1.0, -1,
             np.where(spi3_vals >=  1.0,  1, 0)).astype(np.int8)
label_vals[np.isnan(spi3_vals)] = 0

# ---------- pack into xarray Dataset ----------
coords = {"time": pr.time, "latitude": pr.latitude, "longitude": pr.longitude}
dims   = ("time", "latitude", "longitude")

spi1_da    = xr.DataArray(spi1_vals,   coords=coords, dims=dims, name="spi1",
                           attrs={"long_name": "SPI-1 (1-month SPI)", "units": "dimensionless"})
spi3_da    = xr.DataArray(spi3_vals,   coords=coords, dims=dims, name="spi3",
                           attrs={"long_name": "SPI-3 (3-month SPI)", "units": "dimensionless"})
spi6_da    = xr.DataArray(spi6_vals,   coords=coords, dims=dims, name="spi6",
                           attrs={"long_name": "SPI-6 (6-month SPI)", "units": "dimensionless"})
label1_da  = xr.DataArray(label1_vals, coords=coords, dims=dims, name="drought_label_spi1",
                           attrs={"long_name": "Drought label from SPI-1 (dry=-1, normal=0, wet=1)",
                                  "units": "1",
                                  "threshold": "SPI-1 <= -1 = dry; >= 1 = wet",
                                  "note": "Primary forecast target: SPI-1[t+1] has zero "
                                          "accumulation-window overlap with feature set at t."})
label_da   = xr.DataArray(label_vals,  coords=coords, dims=dims, name="drought_label_spi3",
                           attrs={"long_name": "Drought label from SPI-3 (dry=-1, normal=0, wet=1)",
                                  "units": "1", "threshold": "SPI-3 <= -1 = dry; >= 1 = wet"})

out_ds = xr.Dataset({"spi1": spi1_da, "spi3": spi3_da, "spi6": spi6_da,
                      "drought_label_spi1": label1_da,
                      "drought_label_spi3": label_da})

enc = {v: {"zlib": True, "complevel": 4} for v in ["spi1", "spi3", "spi6"]}
enc["drought_label_spi1"] = {"zlib": True, "complevel": 4, "dtype": "int8"}
enc["drought_label_spi3"] = {"zlib": True, "complevel": 4, "dtype": "int8"}

print("Saving to", OUT_FILE)
out_ds.to_netcdf(OUT_FILE, encoding=enc)

# ---------- summary ----------
counts1 = {k: int((label1_vals == v).sum()) for k, v in {"dry": -1, "normal": 0, "wet": 1}.items()}
counts3 = {k: int((label_vals  == v).sum()) for k, v in {"dry": -1, "normal": 0, "wet": 1}.items()}
print("Wrote:", OUT_FILE)
print("drought_label_spi1 counts (all grid-cells × months):", counts1)
print("drought_label_spi3 counts (all grid-cells × months):", counts3)
print("SPI-1 range: [{:.2f}, {:.2f}]".format(float(np.nanmin(spi1_vals)),
                                               float(np.nanmax(spi1_vals))))
print("SPI-3 range: [{:.2f}, {:.2f}]".format(float(np.nanmin(spi3_vals)),
                                               float(np.nanmax(spi3_vals))))
