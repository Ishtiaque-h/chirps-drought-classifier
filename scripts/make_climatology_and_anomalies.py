#!/usr/bin/env python
from pathlib import Path
import xarray as xr
from dask.diagnostics import ProgressBar

IN_FILE  = Path("data/processed/chirps_v3_monthly_cvalley_1991_2025.nc")
OUT_DIR  = Path("data/processed"); OUT_DIR.mkdir(parents=True, exist_ok=True)
CLIM_OUT = OUT_DIR / "chirps_v3_monthly_cvalley_clim_1991_2020.nc"
ANOM_OUT = OUT_DIR / "chirps_v3_monthly_cvalley_anom_1991_2025.nc"

# 1) Load the regional dataset (time, lat, lon)
ds = xr.open_dataset(IN_FILE).load()
pr = ds["pr"]

# 2) Baseline climatology = monthly mean over 1991–2020
base = pr.sel(time=slice("1991-01-01", "2020-12-31"))
clim = base.groupby("time.month").mean("time")  # (month, lat, lon)
clim.name = "pr_clim"

# 3) Anomalies for 1991–2025 = (month value) - (monthly climatology)
# use slice through end of 2025; extra months are ignored
target = pr.sel(time=slice("1991-01-01", "2025-12-31"))
anom = target.groupby("time.month") - clim
anom.name = "pr_anom"

# 4) Save with light compression + a tiny progress bar
enc = {"pr_clim": dict(zlib=True, complevel=4), "pr_anom": dict(zlib=True, complevel=4)}

with ProgressBar():
    xr.Dataset({"pr_clim": clim}).to_netcdf(CLIM_OUT, encoding={"pr_clim": enc["pr_clim"]}, compute=True)

with ProgressBar():
    xr.Dataset({"pr_anom": anom}).to_netcdf(ANOM_OUT, encoding={"pr_anom": enc["pr_anom"]}, compute=True)

print("Wrote:", CLIM_OUT)
print("  dims:", {k: int(v) for k, v in xr.open_dataset(CLIM_OUT).sizes.items()})
print("Wrote:", ANOM_OUT)
anom_ds = xr.open_dataset(ANOM_OUT)
print("  dims:", {k: int(v) for k, v in anom_ds.sizes.items()})
print("  time span:", str(anom_ds.time[0].values), "→", str(anom_ds.time[-1].values))

