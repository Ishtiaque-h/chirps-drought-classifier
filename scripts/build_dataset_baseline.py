#!/usr/bin/env python
"""
Build a model-ready tabular dataset from CHIRPS anomalies + labels.

Inputs:
  data/processed/chirps_v3_monthly_cvalley_1991_2025.nc        (pr)
  data/processed/chirps_v3_monthly_cvalley_anom_1991_2025.nc   (pr_anom)
  data/processed/chirps_v3_monthly_cvalley_labels_1991_2025.nc (drought_label)

Output:
  data/processed/dataset_baseline.parquet
  data/processed/dataset_baseline_sample.csv   (first ~10k rows preview)
"""
from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr

# ---- paths ----
PROCESSED = Path("data/processed")
OUT_PARQUET = PROCESSED / "dataset_baseline.parquet"
OUT_SAMPLE = PROCESSED / "dataset_baseline_sample.csv"

# ---- load inputs (small enough to fit in memory) ----
pr_ds   = xr.open_dataset(PROCESSED / "chirps_v3_monthly_cvalley_1991_2025.nc").load()
anom_ds = xr.open_dataset(PROCESSED / "chirps_v3_monthly_cvalley_anom_1991_2025.nc").load()
lab_ds  = xr.open_dataset(PROCESSED / "chirps_v3_monthly_cvalley_labels_1991_2025.nc").load()

# variable names and coord names
lat_name = "latitude" if "latitude" in pr_ds.coords else "lat"
lon_name = "longitude" if "longitude" in pr_ds.coords else "lon"

pr      = pr_ds["pr"]                        # (time, lat, lon)
pr_anom = anom_ds["pr_anom"]                 # (time, lat, lon)
label   = lab_ds["drought_label"]            # (time, lat, lon)

# ensure identical coords/time
pr_anom = pr_anom.sel(time=pr.time)
label   = label.sel(time=pr.time)

# ---- create lag features along time per grid cell ----
# xarray shift handles broadcasting across lat/lon
anom_lag1 = pr_anom.shift(time=1)
anom_lag3 = pr_anom.shift(time=3)

# ---- build a DataFrame by stacking spatial dims ----
# ---- build a DataFrame by stacking spatial dims ----
ds_stacked = xr.Dataset({
    "pr": pr,
    "pr_anom": pr_anom,
    "anom_lag1": anom_lag1,
    "anom_lag3": anom_lag3,
    "drought_label": label,
}).stack(pixel=(lat_name, lon_name))

# Expand the 'pixel' MultiIndex into separate latitude/longitude coords
ds_flat = ds_stacked.reset_index('pixel')   # now 'latitude'/'longitude' are coords, index is just 'time'

# Safe: to_dataframe() then bring 'time' out if needed
df = ds_flat.to_dataframe()
if 'time' not in df.columns:  # usually it's still an index, so make it a column
    df = df.reset_index()

# drop rows where target is NaN (e.g., first lags or masked cells)
df = df.dropna(subset=["drought_label"]).copy()

# ---- time features ----
df["month"] = pd.to_datetime(df["time"]).dt.month
df["year"]  = pd.to_datetime(df["time"]).dt.year
# cyclic month encoding (helps linear models)
df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12.0)
df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12.0)

# rename coords to clean column names
df = df.rename(columns={lat_name: "latitude", lon_name: "longitude"})

# reorder columns
cols = [
    "time", "year", "month", "month_sin", "month_cos",
    "latitude", "longitude",
    "pr", "pr_anom", "anom_lag1", "anom_lag3",
    "drought_label"
]
df = df[cols]

# cast target to small int
df["drought_label"] = df["drought_label"].astype(np.int8)

# ---- save outputs ----
OUT_PARQUET.parent.mkdir(parents=True, exist_ok=True)
df.to_parquet(OUT_PARQUET, index=False)
df.head(10000).to_csv(OUT_SAMPLE, index=False)

print("Wrote:", OUT_PARQUET, f"(rows={len(df):,}, cols={df.shape[1]})")
print("Wrote:", OUT_SAMPLE)

