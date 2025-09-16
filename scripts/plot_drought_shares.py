#!/usr/bin/env python
"""
Compute the monthly fraction of grid cells labeled dry/normal/wet (1991–2025)
and save both a CSV and a simple stacked-area plot.

Inputs:
  data/processed/chirps_v3_monthly_cvalley_labels_1991_2025.nc
Outputs:
  outputs/drought_shares.csv
  outputs/drought_shares_stacked.png
"""
from pathlib import Path
import xarray as xr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

LABELS_FILE = Path("data/processed/chirps_v3_monthly_cvalley_labels_1991_2025.nc")
OUT_DIR = Path("outputs")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# --- load numeric labels (-1=dry, 0=normal, 1=wet) ---
ds = xr.open_dataset(LABELS_FILE)
lab = ds["drought_label"]  # dims: time, latitude, longitude (names may vary but include 'time')

# detect spatial dims automatically (everything except time)
spatial_dims = [d for d in lab.dims if d != "time"]
if not spatial_dims:
    raise SystemExit("No spatial dimensions found on drought_label variable.")

# --- compute fractions per month ---
# Boolean masks; mean over space gives fraction (skipna handles any masked cells)
frac_dry = (lab == -1).mean(dim=spatial_dims, skipna=True)
frac_nor = (lab == 0).mean(dim=spatial_dims, skipna=True)
frac_wet = (lab == 1).mean(dim=spatial_dims, skipna=True)

# turn into a tidy DataFrame
df = pd.DataFrame({
    "time": pd.to_datetime(frac_dry["time"].values),
    "dry": frac_dry.values.astype(float),
    "normal": frac_nor.values.astype(float),
    "wet": frac_wet.values.astype(float),
}).set_index("time").sort_index()

# save CSV
csv_path = OUT_DIR / "drought_shares.csv"
df.to_csv(csv_path, float_format="%.6f")

# --- plot (stacked area) ---
fig, ax = plt.subplots(figsize=(11, 4))
ax.stackplot(df.index, df["dry"], df["normal"], df["wet"], labels=["dry","normal","wet"])
ax.set_ylabel("Fraction of area")
ax.set_title("Central Valley monthly drought-class shares (1991–2025)")
ax.set_xlim(df.index.min(), df.index.max())
ax.set_ylim(0, 1)
ax.legend(loc="upper right", ncol=3, frameon=False)
ax.grid(True, alpha=0.25)

png_path = OUT_DIR / "drought_shares_stacked.png"
fig.tight_layout()
fig.savefig(png_path, dpi=150)
plt.close(fig)

print("Wrote:", csv_path)
print("Wrote:", png_path)
print("Sample rows:\n", df.head(3))

