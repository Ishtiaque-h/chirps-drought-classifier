#!/usr/bin/env python
"""
Plot a drought-class map (dry/normal/wet) for a given month (YYYY-MM).

Input:
  data/processed/chirps_v3_monthly_cvalley_labels_1991_2025.nc
Output:
  outputs/drought_map_YYYY-MM.png
Usage:
  python scripts/plot_drought_map.py 2014-01
"""
from pathlib import Path
import sys
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm

LABELS = Path("data/processed/chirps_v3_monthly_cvalley_labels_1991_2025.nc")
OUTDIR = Path("outputs"); OUTDIR.mkdir(parents=True, exist_ok=True)

# ---- parse month arg ----
ym = sys.argv[1] if len(sys.argv) > 1 else "2014-01"  # default example
# standardize to first day of month
time_key = f"{ym}-01"

# ---- load labels ----
ds = xr.open_dataset(LABELS)
lab = ds["drought_label"]  # -1=dry, 0=normal, 1=wet

# ---- select month ----
try:
    sel = lab.sel(time=time_key)
except Exception:
    # helpful error with available range
    t0 = str(lab.time.values[0])[:10]
    t1 = str(lab.time.values[-1])[:10]
    raise SystemExit(f"Month {ym} not found. Available range: {t0} .. {t1}")

# get coord names
lat_name = "latitude" if "latitude" in sel.coords else "lat"
lon_name = "longitude" if "longitude" in sel.coords else "lon"

lat = sel[lat_name].values
lon = sel[lon_name].values
data = sel.values  # 2D array

# ---- colormap (dry/normal/wet) ----
# values are -1, 0, 1
bounds = [-1.5, -0.5, 0.5, 1.5]
cmap = ListedColormap(["#d73027", "#fdae61", "#1a9850"])  # dry, normal, wet
norm = BoundaryNorm(bounds, cmap.N)

# detect latitude order for correct display
origin = "lower" if lat[0] < lat[-1] else "upper"

# ---- plot ----
fig, ax = plt.subplots(figsize=(6.5, 6))
im = ax.imshow(
    data,
    cmap=cmap,
    norm=norm,
    extent=[lon.min(), lon.max(), lat.min(), lat.max()],
    origin=origin,
    interpolation="nearest",
    aspect="auto",
)

ax.set_title(f"Central Valley drought classes — {ym}")
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")
ax.grid(True, alpha=0.2, linestyle=":")

# custom legend
from matplotlib.patches import Patch
legend_patches = [
    Patch(color="#d73027", label="dry (-1)"),
    Patch(color="#fdae61", label="normal (0)"),
    Patch(color="#1a9850", label="wet (1)"),
]
ax.legend(handles=legend_patches, loc="lower left", frameon=False)

out_path = OUTDIR / f"drought_map_{ym}.png"
fig.tight_layout()
fig.savefig(out_path, dpi=150)
plt.close(fig)

print("Wrote:", out_path)

