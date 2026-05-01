#!/usr/bin/env python
"""
Build a 4-D spatiotemporal dataset for the ConvLSTM drought forecast model.

Design
------
Each sample is a sequence of seq_len=3 consecutive months ending at time t,
with four channels per month: [spi1, spi3, spi6, pr_norm].
The target is the spatial drought-class grid at t+1
(drought_label_spi1 shifted back by one, matching build_dataset_forecast.py).

Channel normalisation
  spi1, spi3, spi6 — already zero-centred (~N(0,1)); no further scaling needed.
  pr_norm          — divided by the 99th-percentile of training-set values so it
                     falls in [0, ~1] without distorting the heavy tail.

Time split (matches train_forecast_xgboost.py)
  Train :  year ≤ 2016
  Val   :  2017–2020
  Test  :  year ≥ 2021

Target encoding
  drought_label_spi1 ∈ {-1, 0, 1}  →  {0, 1, 2}

Outputs (written to data/processed/)
  convlstm_X_train.npy  — float32, shape (N_train, seq_len, C, lat, lon)
  convlstm_y_train.npy  — int64,   shape (N_train, lat, lon)
  convlstm_X_val.npy
  convlstm_y_val.npy
  convlstm_X_test.npy
  convlstm_y_test.npy
  convlstm_meta.npz     — lat, lon, pr_scale, split_years, test feature/target times

Inputs
  data/processed/chirps_v3_monthly_cvalley_1991_2026.nc
  data/processed/chirps_v3_monthly_cvalley_spi_1991_2026.nc
"""
from pathlib import Path
import numpy as np
import xarray as xr

BASE_DIR = Path(__file__).resolve().parents[1]
PROC     = BASE_DIR / "data" / "processed"
PR_FILE  = PROC / "chirps_v3_monthly_cvalley_1991_2026.nc"
SPI_FILE = PROC / "chirps_v3_monthly_cvalley_spi_1991_2026.nc"

SEQ_LEN    = 3          # number of lag months in each input window
LABEL_MAP  = {-1: 0, 0: 1, 1: 2}
TRAIN_END  = 2016       # inclusive
VAL_END    = 2020       # inclusive; test = year >= 2021
TARGET_ALIGNMENT_VERSION = "t_plus_1_single_shift_v2"

# --------------------------------------------------------------------------
print("Loading data ...")
pr_ds  = xr.open_dataset(PR_FILE).load()
spi_ds = xr.open_dataset(SPI_FILE).load()

pr    = pr_ds["pr"].astype("float32")                # (time, lat, lon)
spi1  = spi_ds["spi1"].astype("float32")
spi3  = spi_ds["spi3"].astype("float32")
spi6  = spi_ds["spi6"].astype("float32")
label = spi_ds["drought_label_spi1"].astype("float32")

# Align time axes
times = pr.time.values
spi1  = spi1.sel(time=times)
spi3  = spi3.sel(time=times)
spi6  = spi6.sel(time=times)
label = label.sel(time=times)

lat = pr.latitude.values
lon = pr.longitude.values
T   = len(times)

print(f"  Spatial grid : {len(lat)} lat × {len(lon)} lon")
print(f"  Time steps   : {T}  ({str(times[0])[:7]} → {str(times[-1])[:7]})")

# --------------------------------------------------------------------------
# Convert to numpy (time, lat, lon)
pr_np    = pr.values        # raw mm month⁻¹
spi1_np  = spi1.values
spi3_np  = spi3.values
spi6_np  = spi6.values
label_np = label.values     # float (NaN outside mask), values in {-1, 0, 1}

# Encode target:  target[t] = label[t+1]
# shift(-1): move data back by one → target_np[t] = label_np[t+1]
target_np = np.empty_like(label_np)
target_np[:-1] = label_np[1:]
target_np[-1]  = np.nan

# Replace NaN label with a sentinel (-99) before int conversion — masked pixels
# will be ignored during loss computation via ignore_index in CrossEntropyLoss.
label_valid = np.where(np.isnan(target_np), -99, target_np).astype("int64")
# Remap {-1, 0, 1} → {0, 1, 2}; keep -99 as-is (invalid / out-of-mask pixels)
remap = np.vectorize(lambda x: LABEL_MAP.get(x, -99))
label_encoded = remap(label_valid)

# --------------------------------------------------------------------------
# Decide per-timestep train / val / test membership
import pandas as pd
time_years = pd.DatetimeIndex(times).year  # length T
target_years = np.empty(T, dtype=int)
target_years[:-1] = time_years[1:]
target_years[-1] = -1

def is_valid_sequence_end(t: int) -> bool:
    """True if we can build a seq_len window ending at t and have a target at t+1."""
    return (t >= SEQ_LEN - 1) and (t < T - 1)

# Split by target year (t+1), matching tabular forecast dataset conventions.
train_idx = [t for t in range(T) if is_valid_sequence_end(t) and target_years[t] <= TRAIN_END]
val_idx   = [t for t in range(T) if is_valid_sequence_end(t) and TRAIN_END < target_years[t] <= VAL_END]
test_idx  = [t for t in range(T) if is_valid_sequence_end(t) and target_years[t] > VAL_END]

print(f"  Train windows: {len(train_idx)}  "
      f"Val windows: {len(val_idx)}  "
      f"Test windows: {len(test_idx)}")

# --------------------------------------------------------------------------
# Compute pr normalisation scale from training set only
pr_train_vals = np.concatenate([pr_np[t - SEQ_LEN + 1: t + 1] for t in train_idx])
pr_scale = float(np.nanpercentile(pr_train_vals, 99))
pr_scale  = max(pr_scale, 1e-6)          # guard against all-zero edge case
print(f"  pr 99th-pct (train) : {pr_scale:.2f} mm")

pr_norm_np = pr_np / pr_scale            # values in [0, ~1]; heavy tail > 1 is fine

# --------------------------------------------------------------------------
def build_arrays(indices):
    """Return (X, y) arrays for the given sequence-end indices."""
    n = len(indices)
    nlat, nlon = len(lat), len(lon)
    X = np.empty((n, SEQ_LEN, 4, nlat, nlon), dtype="float32")
    y = np.empty((n, nlat, nlon), dtype="int64")
    for i, t in enumerate(indices):
        for s in range(SEQ_LEN):
            ts = t - (SEQ_LEN - 1 - s)   # oldest → newest channel
            X[i, s, 0] = spi1_np[ts]
            X[i, s, 1] = spi3_np[ts]
            X[i, s, 2] = spi6_np[ts]
            X[i, s, 3] = pr_norm_np[ts]
        y[i] = label_encoded[t]           # label_encoded[t] is already shifted to target t+1
    return X, y

print("Building arrays ...")
X_train, y_train = build_arrays(train_idx)
X_val,   y_val   = build_arrays(val_idx)
X_test,  y_test  = build_arrays(test_idx)

print(f"  X_train {X_train.shape}  y_train {y_train.shape}")
print(f"  X_val   {X_val.shape}    y_val   {y_val.shape}")
print(f"  X_test  {X_test.shape}   y_test  {y_test.shape}")

# Replace NaN inputs with 0.0 (masked ocean / outside-boundary pixels already carry
# NaN from the source NetCDF; treated as zero-padding for the CNN receptive field)
for arr in (X_train, X_val, X_test):
    np.nan_to_num(arr, copy=False, nan=0.0)

# --------------------------------------------------------------------------
print("Saving arrays ...")
np.save(PROC / "convlstm_X_train.npy", X_train)
np.save(PROC / "convlstm_y_train.npy", y_train)
np.save(PROC / "convlstm_X_val.npy",   X_val)
np.save(PROC / "convlstm_y_val.npy",   y_val)
np.save(PROC / "convlstm_X_test.npy",  X_test)
np.save(PROC / "convlstm_y_test.npy",  y_test)

np.savez(
    PROC / "convlstm_meta.npz",
    lat=lat, lon=lon,
    pr_scale=pr_scale,
    train_end=TRAIN_END,
    val_end=VAL_END,
    seq_len=SEQ_LEN,
    channels=["spi1", "spi3", "spi6", "pr_norm"],
    label_map=list(LABEL_MAP.items()),
    target_alignment_version=TARGET_ALIGNMENT_VERSION,
    test_feature_times=np.array([times[t] for t in test_idx]),
    test_target_times=np.array([times[t + 1] for t in test_idx]),
)

print("Done.  Files written to", PROC)
