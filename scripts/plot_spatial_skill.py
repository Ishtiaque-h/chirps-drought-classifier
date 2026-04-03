#!/usr/bin/env python
"""
Spatial skill maps for the 1-month-ahead drought forecast.

For each 0.05° pixel in the Central Valley, compute the fraction of test
months (2021–2025) where the dominant predicted class matches the true
drought class.  Save the result as a NetCDF (for GIS use) and a heatmap PNG.

The spatial skill map reveals which sub-regions the model forecasts well
or poorly, which is expected in any remote sensing / hydrology paper.
Sub-regional patterns can be linked to known agro-climatic gradients:
  - Sacramento Valley (northern half, approx. lat > 38°): wetter, more
    variable winter precipitation.
  - San Joaquin Valley (southern half, approx. lat < 38°): drier, more
    drought-prone.

Inputs:
  data/processed/dataset_forecast.parquet
  outputs/forecast_xgb_model.json

Outputs:
  outputs/spatial_skill_accuracy.nc    — per-pixel forecast accuracy (0–1)
  outputs/spatial_skill_accuracy.png   — heatmap
"""
from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr
import xgboost as xgb
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

DATA       = Path("data/processed/dataset_forecast.parquet")
MODEL_PATH = Path("outputs/forecast_xgb_model.json")
OUT_DIR    = Path("outputs"); OUT_DIR.mkdir(exist_ok=True)

FEATURES = [
    "spi1_lag1", "spi1_lag2", "spi1_lag3",
    "spi3_lag1", "spi6_lag1",
    "pr_lag1", "pr_lag2", "pr_lag3",
    "month_sin", "month_cos",
]
TARGET    = "target_label"
INV_MAP   = {0: -1, 1: 0, 2: 1}

# ── load data and model ───────────────────────────────────────────────────────
print("Loading dataset and model...")
df = pd.read_parquet(DATA)
df["year"] = df["year"].astype(int)
test = df[df["year"] >= 2021].copy()

assert MODEL_PATH.exists(), f"Model not found: {MODEL_PATH}. Run train_forecast_xgboost.py first."
model = xgb.Booster()
model.load_model(MODEL_PATH.as_posix())

dtest = xgb.DMatrix(test[FEATURES], feature_names=FEATURES)
probs = model.predict(dtest)
test["pred_label"] = [INV_MAP[i] for i in probs.argmax(axis=1)]
test["correct"]    = (test["pred_label"] == test[TARGET]).astype(int)

# ── per-pixel accuracy ────────────────────────────────────────────────────────
print("Computing per-pixel accuracy...")
pixel_acc = (
    test.groupby(["latitude", "longitude"])["correct"]
    .mean()
    .rename("accuracy")
    .reset_index()
)

# ── pivot to 2-D grid ─────────────────────────────────────────────────────────
lats = np.sort(pixel_acc["latitude"].unique())
lons = np.sort(pixel_acc["longitude"].unique())

acc_grid = np.full((len(lats), len(lons)), np.nan)
lat_idx  = {v: i for i, v in enumerate(lats)}
lon_idx  = {v: i for i, v in enumerate(lons)}

for _, row in pixel_acc.iterrows():
    i = lat_idx[row["latitude"]]
    j = lon_idx[row["longitude"]]
    acc_grid[i, j] = row["accuracy"]

# ── save as NetCDF ────────────────────────────────────────────────────────────
da = xr.DataArray(
    acc_grid,
    coords={"latitude": lats, "longitude": lons},
    dims=["latitude", "longitude"],
    name="forecast_accuracy",
    attrs={
        "long_name": "Per-pixel 1-month-ahead forecast accuracy (2021–2025)",
        "units":     "fraction",
        "note":      "Fraction of test months where XGBoost dominant class == SPI-1 truth",
    },
)
nc_path = OUT_DIR / "spatial_skill_accuracy.nc"
da.to_netcdf(nc_path)
print("Wrote:", nc_path)

# ── plot ──────────────────────────────────────────────────────────────────────
print("Plotting spatial skill map...")
fig, ax = plt.subplots(figsize=(8, 9))

lon2d, lat2d = np.meshgrid(lons, lats)

# diverging colormap centered approximately at random-class performance (0.33 for 3 classes)
# using RdYlGn: red = poor skill (~0), green = high skill (~1)
cmap  = plt.cm.RdYlGn
vmin, vmax = 0.0, 1.0
pc = ax.pcolormesh(lon2d, lat2d, acc_grid, cmap=cmap, vmin=vmin, vmax=vmax,
                   shading="auto")
plt.colorbar(pc, ax=ax, label="Forecast accuracy (fraction of months correct)")

# annotate sub-regions
ax.axhline(38.0, color="navy", lw=1.2, linestyle="--", alpha=0.7)
ax.text(-121.9, 38.15, "Sacramento Valley (N)", color="navy", fontsize=8,
        ha="left", va="bottom")
ax.text(-121.9, 37.85, "San Joaquin Valley (S)", color="navy", fontsize=8,
        ha="left", va="top")

ax.set_xlabel("Longitude (°E)")
ax.set_ylabel("Latitude (°N)")
ax.set_title(
    "Per-pixel 1-month-ahead drought forecast accuracy\n"
    "Central Valley, California (2021–2025 test period)\n"
    "Target: SPI-1 drought class (dry / normal / wet)",
    fontsize=9,
)

fig.tight_layout()
png_path = OUT_DIR / "spatial_skill_accuracy.png"
fig.savefig(png_path, dpi=150, bbox_inches="tight")
plt.close()
print("Wrote:", png_path)

# ── summary statistics ────────────────────────────────────────────────────────
valid = acc_grid[~np.isnan(acc_grid)]
print(f"\nSpatial accuracy summary:")
print(f"  Mean  : {valid.mean():.3f}")
print(f"  Median: {np.median(valid):.3f}")
print(f"  Std   : {valid.std():.3f}")
print(f"  Min   : {valid.min():.3f}")
print(f"  Max   : {valid.max():.3f}")
