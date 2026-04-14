#!/usr/bin/env python
"""
Build a model-ready tabular forecasting dataset from CHIRPS SPI labels.

Key design choice (eliminates label leakage):
  TARGET at time t  →  drought_label_spi1[t+1]   (predict *next* month's SPI-1 class)
  FEATURES at time t  →  only information available before the target month

Why SPI-1 as target (not SPI-3)?
  SPI-3[t+1] = f(pr[t-1], pr[t], pr[t+1]).  Because features include pr[t], pr[t-1],
  and spi3[t] = f(pr[t-2..t]), the model only needs to infer pr[t+1] to reconstruct
  SPI-3[t+1] — making the task trivially easy and accuracy artificially high.
  SPI-1[t+1] = f(pr[t+1]) only, so features at t carry zero accumulation-window
  information about the target.  This is the standard choice in published 1-month-ahead
  drought forecasting literature (Dikshit et al. 2021, Sci. Total Environ.).

Features:
  spi1_lag1, spi1_lag2, spi1_lag3   — SPI-1 at t, t-1, t-2
  spi3_lag1                         — SPI-3 at t (3-month window ending at t)
  spi6_lag1                         — SPI-6 at t (6-month window ending at t)
  pr_lag1, pr_lag2, pr_lag3         — raw precipitation at t, t-1, t-2
  month_sin, month_cos              — cyclic seasonality of the TARGET month
  year                              — calendar year of the TARGET month

Input:
  data/processed/chirps_v3_monthly_cvalley_1991_2026.nc
  data/processed/chirps_v3_monthly_cvalley_spi_1991_2026.nc

Output:
  data/processed/dataset_forecast.parquet
  data/processed/dataset_forecast_sample.csv   (first 10 k rows)
"""
from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr

PROCESSED   = Path("data/processed")
PR_FILE     = PROCESSED / "chirps_v3_monthly_cvalley_1991_2026.nc"
SPI_FILE    = PROCESSED / "chirps_v3_monthly_cvalley_spi_1991_2026.nc"
OUT_PARQUET = PROCESSED / "dataset_forecast.parquet"
OUT_SAMPLE  = PROCESSED / "dataset_forecast_sample.csv"

# ---------- load ----------
print("Loading datasets...")
pr_ds  = xr.open_dataset(PR_FILE).load()
spi_ds = xr.open_dataset(SPI_FILE).load()

lat_name = "latitude" if "latitude" in pr_ds.coords else "lat"
lon_name = "longitude" if "longitude" in pr_ds.coords else "lon"

pr    = pr_ds["pr"]                     # (time, lat, lon)
spi1  = spi_ds["spi1"]
spi3  = spi_ds["spi3"]
spi6  = spi_ds["spi6"]
label = spi_ds["drought_label_spi1"]   # int8 — SPI-1 class (primary forecast target)

# ensure matching time axes
spi1  = spi1.sel(time=pr.time)
spi3  = spi3.sel(time=pr.time)
spi6  = spi6.sel(time=pr.time)
label = label.sel(time=pr.time)

# ---------- TARGET: shift label back by 1 so that target[t] = label[t+1] ----------
# xarray shift(time=-1) moves data backward by 1 step: target_at_t = label_at_{t+1}
# Using drought_label_spi1[t+1]: SPI-1 depends only on pr[t+1], so features at t
# contain zero accumulation-window information about the target.
target = label.shift(time=-1)   # shift=-1: target[t] = label[t+1]
target.name = "target_label"

# ---------- FEATURES (all lagged, no future information) ----------
# "lag1" = current month value, "lag2" = previous month, etc.
spi1_lag1 = spi1.copy(); spi1_lag1.name = "spi1_lag1"
spi1_lag2 = spi1.shift(time=1);  spi1_lag2.name = "spi1_lag2"
spi1_lag3 = spi1.shift(time=2);  spi1_lag3.name = "spi1_lag3"
spi3_lag1 = spi3.copy(); spi3_lag1.name = "spi3_lag1"
spi6_lag1 = spi6.copy(); spi6_lag1.name = "spi6_lag1"
pr_lag1   = pr.copy();   pr_lag1.name = "pr_lag1"
pr_lag2   = pr.shift(time=1);  pr_lag2.name = "pr_lag2"
pr_lag3   = pr.shift(time=2);  pr_lag3.name = "pr_lag3"

# ---------- stack spatial dims → flat DataFrame ----------
ds_stacked = xr.Dataset({
    "spi1_lag1": spi1_lag1,
    "spi1_lag2": spi1_lag2,
    "spi1_lag3": spi1_lag3,
    "spi3_lag1": spi3_lag1,
    "spi6_lag1": spi6_lag1,
    "pr_lag1":   pr_lag1,
    "pr_lag2":   pr_lag2,
    "pr_lag3":   pr_lag3,
    "target_label": target,
}).stack(pixel=(lat_name, lon_name))

ds_flat = ds_stacked.reset_index("pixel")
df = ds_flat.to_dataframe()
if "time" not in df.columns:
    df = df.reset_index()

# ---------- drop rows where target is NaN (last time step after shift, or masked) ----------
feat_cols = ["spi1_lag1", "spi1_lag2", "spi1_lag3",
             "spi3_lag1", "spi6_lag1",
             "pr_lag1", "pr_lag2", "pr_lag3"]
df = df.dropna(subset=["target_label"] + feat_cols).copy()

# ---------- time features of the TARGET month ----------
target_time = pd.to_datetime(df["time"]) + pd.DateOffset(months=1)
df["month"] = target_time.dt.month
df["year"]  = target_time.dt.year
df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12.0)
df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12.0)

# rename coords
df = df.rename(columns={lat_name: "latitude", lon_name: "longitude"})

# cast target
df["target_label"] = df["target_label"].astype(np.int8)

# reorder
cols = (
    ["time", "year", "month", "month_sin", "month_cos", "latitude", "longitude"]
    + feat_cols
    + ["target_label"]
)
df = df[cols]

# ---------- save ----------
OUT_PARQUET.parent.mkdir(parents=True, exist_ok=True)
df.to_parquet(OUT_PARQUET, index=False)
df.head(10_000).to_csv(OUT_SAMPLE, index=False)

print("Wrote:", OUT_PARQUET, f"(rows={len(df):,}, cols={df.shape[1]})")
print("Wrote:", OUT_SAMPLE)
print("Class distribution:\n", df["target_label"].value_counts().sort_index())
