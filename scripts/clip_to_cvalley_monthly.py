#!/usr/bin/env python
from pathlib import Path
import xarray as xr
from dask.diagnostics import ProgressBar

IN_DIR = Path("data/raw/chirps_v3/monthly")
OUT_DIR = Path("data/processed"); OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_FILE = OUT_DIR / "chirps_v3_monthly_cvalley_1991_2025.nc"

# Central Valley bbox
lat_min, lat_max = 35.4, 40.6
lon_min, lon_max = -122.5, -119.0

files = sorted(IN_DIR.glob("chirps-v3.0.*.monthly.nc"))
if not files:
    raise SystemExit("No input files found in data/raw/chirps_v3/monthly")

# add time chunks so progress updates smoothly
ds = xr.open_mfdataset(files, combine="by_coords", chunks={"time": 12})

lat_name = "latitude" if "latitude" in ds.coords else "lat"
lon_name = "longitude" if "longitude" in ds.coords else "lon"

# latitude slice (handles ascending/descending)
a, b = sorted((lat_min, lat_max))
if float(ds[lat_name][0]) > float(ds[lat_name][-1]):
    ds = ds.sel({lat_name: slice(b, a)})
else:
    ds = ds.sel({lat_name: slice(a, b)})

# longitude slice (dataset is -180..180 from your check)
c, d = sorted((lon_min, lon_max))
ds = ds.sel({lon_name: slice(c, d)})

# choose precip var, rename to 'pr'
var = next((v for v in ds.data_vars if v.lower().startswith(("precip","pr"))), list(ds.data_vars)[0])
ds = ds[[var]].rename({var: "pr"})

# compress and compute with a progress bar
encoding = {"pr": dict(zlib=True, complevel=4)}
delayed = ds.to_netcdf(OUT_FILE, encoding=encoding, compute=False)
with ProgressBar():
    delayed.compute()

print("Wrote:", OUT_FILE)
print("Dims:", {k: int(v) for k, v in ds.sizes.items()})
