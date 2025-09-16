#!/usr/bin/env python
from pathlib import Path
import xarray as xr
import numpy as np
from dask.diagnostics import ProgressBar

IN_FILE  = Path("data/processed/chirps_v3_monthly_cvalley_1991_2025.nc")
OUT_DIR  = Path("data/processed"); OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_FILE = OUT_DIR / "chirps_v3_monthly_cvalley_labels_1991_2025.nc"

# Load the clipped regional dataset fully (it's small enough)
ds = xr.open_dataset(IN_FILE).load()
pr = ds["pr"]  # (time, latitude, longitude)

# 1) Baseline distribution (1991–2020), by calendar month
base = pr.sel(time=slice("1991-01-01", "2020-12-31"))

# Compute both percentiles at once to avoid coord conflicts
q = base.groupby("time.month").quantile([0.20, 0.80], dim="time", skipna=True)  # (month, quantile, lat, lon)
p20 = q.sel(quantile=0.20).drop_vars("quantile")  # remove the coord
p80 = q.sel(quantile=0.80).drop_vars("quantile")

# Fallback for all-NaN cells: use monthly mean as a backup threshold
clim = base.groupby("time.month").mean("time")
p20 = p20.where(~np.isnan(p20), clim)
p80 = p80.where(~np.isnan(p80), clim)

# 2) Target window for labels (1991–2025)
target = pr.sel(time=slice("1991-01-01", "2025-12-31"))

# Align monthly percentiles to each target month (broadcast by month)
t_month = target["time"].dt.month
p20m = p20.sel(month=t_month)
p80m = p80.sel(month=t_month)

# 3) Make labels: -1=dry, 0=normal, +1=wet (and a string version)
label_num = xr.where(target < p20m, -1, xr.where(target > p80m, 1, 0)).astype(np.int8)
label_num.name = "drought_label"

label_str = xr.where(label_num == -1, "dry", xr.where(label_num == 1, "wet", "normal"))
label_str = label_str.astype(object)  # netCDF will store as variable-length strings
label_str.name = "drought_label_str"

# 4) Save labels + thresholds so we keep provenance
out = xr.Dataset(
    {
        "drought_label": label_num,
        "drought_label_str": label_str,
        "pr_p20": p20,
        "pr_p80": p80,
    }
)

# Small compression for numeric arrays; strings are stored as variable-length
enc = {
    "drought_label": {"zlib": True, "complevel": 4},
    "pr_p20": {"zlib": True, "complevel": 4},
    "pr_p80": {"zlib": True, "complevel": 4},
    # (no compression for drought_label_str to keep it simple)
}

with ProgressBar():
    out.to_netcdf(OUT_FILE, encoding=enc)

# Print a quick summary (region-wide counts)
counts = {k: int((label_num == v).sum().values) for k, v in {"dry": -1, "normal": 0, "wet": 1}.items()}
print("Wrote:", OUT_FILE)
print("Counts (all grid-cells x months):", counts)

