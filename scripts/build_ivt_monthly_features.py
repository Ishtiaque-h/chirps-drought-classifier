#!/usr/bin/env python
"""
Build monthly IVT anomaly features from ERA5 and save to ar_ivt_monthly.csv.

Output:
  data/processed/ar_ivt_monthly.csv

Columns:
  time,
  ivt_mean_anom_lag1, ivt_mean_anom_lag2,
  ivt_p90_anom_lag1, ivt_p90_anom_lag2
"""
from __future__ import annotations

from argparse import ArgumentParser, Namespace
from datetime import datetime, timezone
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import xarray as xr


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED = PROJECT_ROOT / "data" / "processed"

START_YEAR = 1991
CURRENT_YEAR = datetime.now(timezone.utc).year
IVT_FILE = PROCESSED / f"era5_ivt_monthly_cvalley_{START_YEAR}_{CURRENT_YEAR}.nc"
DOWNLOAD_SCRIPT = PROJECT_ROOT / "scripts" / "download_era5_ivt_monthly.py"
OUT_FILE = PROCESSED / "ar_ivt_monthly.csv"


def parse_args() -> Namespace:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument(
        "--download-if-missing",
        action="store_true",
        help="Run download_era5_ivt_monthly.py if the IVT NetCDF is missing.",
    )
    return parser.parse_args()


def ensure_ivt_file(download_if_missing: bool) -> None:
    if IVT_FILE.exists():
        return
    if not download_if_missing:
        raise FileNotFoundError(
            f"ERA5 IVT file not found: {IVT_FILE}\n"
            f"Run {DOWNLOAD_SCRIPT} or pass --download-if-missing."
        )
    print("ERA5 IVT file not found. Running downloader...")
    subprocess.run([sys.executable, str(DOWNLOAD_SCRIPT)], cwd=PROJECT_ROOT, check=True)
    if not IVT_FILE.exists():
        raise FileNotFoundError(f"Download finished, but file is still missing: {IVT_FILE}")


def choose_var(ds: xr.Dataset, candidates: list[str]) -> str:
    for candidate in candidates:
        if candidate in ds.data_vars:
            return candidate
    lower_map = {name.lower(): name for name in ds.data_vars}
    for candidate in candidates:
        if candidate.lower() in lower_map:
            return lower_map[candidate.lower()]
    raise ValueError(f"Could not find any of {candidates}; available={list(ds.data_vars)}")


def standardize_da(ds: xr.Dataset, var_name: str) -> xr.DataArray:
    da = ds[var_name]
    rename_map = {}
    if "valid_time" in da.dims:
        rename_map["valid_time"] = "time"
    if "lat" in da.dims:
        rename_map["lat"] = "latitude"
    if "lon" in da.dims:
        rename_map["lon"] = "longitude"
    if rename_map:
        da = da.rename(rename_map)

    if "expver" in da.dims:
        da = da.mean(dim="expver", skipna=True)

    extra_dims = [d for d in da.dims if d not in {"time", "latitude", "longitude"}]
    for dim in extra_dims:
        if da.sizes[dim] == 1:
            da = da.squeeze(dim, drop=True)
    extra_dims = [d for d in da.dims if d not in {"time", "latitude", "longitude"}]
    if extra_dims:
        raise ValueError(f"Unexpected dimensions for {var_name}: {da.dims}")

    da = da.transpose("time", "latitude", "longitude")
    time_index = pd.DatetimeIndex(da["time"].values)
    da = da.isel(time=~time_index.duplicated()).sortby("time")
    return da


def main() -> None:
    args = parse_args()
    ensure_ivt_file(args.download_if_missing)

    print(f"Loading IVT data: {IVT_FILE}")
    ds = xr.open_dataset(IVT_FILE).load()

    east_var = choose_var(
        ds,
        [
            "vertical_integral_of_eastward_water_vapour_flux",
            "viwve",
        ],
    )
    north_var = choose_var(
        ds,
        [
            "vertical_integral_of_northward_water_vapour_flux",
            "viwvn",
        ],
    )

    east = standardize_da(ds, east_var)
    north = standardize_da(ds, north_var)
    north = north.sel(time=east.time)

    ivt = np.sqrt(east ** 2 + north ** 2)

    ivt_mean = ivt.mean(dim=["latitude", "longitude"], skipna=True)
    ivt_p90 = ivt.quantile(0.9, dim=["latitude", "longitude"], skipna=True)

    frame = pd.DataFrame(
        {
            "time": pd.to_datetime(ivt_mean["time"].values).to_period("M").to_timestamp(),
            "ivt_mean": ivt_mean.values.astype(float),
            "ivt_p90": ivt_p90.values.astype(float),
        }
    ).sort_values("time")

    base = frame[(frame["time"].dt.year >= 1991) & (frame["time"].dt.year <= 2020)].copy()
    clim = (
        base.assign(month=base["time"].dt.month)
        .groupby("month")[["ivt_mean", "ivt_p90"]]
        .mean()
        .rename(columns={"ivt_mean": "ivt_mean_clim", "ivt_p90": "ivt_p90_clim"})
    )

    frame["month"] = frame["time"].dt.month
    frame = frame.merge(clim, on="month", how="left")
    frame["ivt_mean_anom"] = frame["ivt_mean"] - frame["ivt_mean_clim"]
    frame["ivt_p90_anom"] = frame["ivt_p90"] - frame["ivt_p90_clim"]

    frame = frame.set_index("time").sort_index()
    frame["ivt_mean_anom_lag1"] = frame["ivt_mean_anom"]
    frame["ivt_mean_anom_lag2"] = frame["ivt_mean_anom"].shift(1)
    frame["ivt_p90_anom_lag1"] = frame["ivt_p90_anom"]
    frame["ivt_p90_anom_lag2"] = frame["ivt_p90_anom"].shift(1)

    out_cols = [
        "ivt_mean_anom_lag1",
        "ivt_mean_anom_lag2",
        "ivt_p90_anom_lag1",
        "ivt_p90_anom_lag2",
    ]
    out = frame.reset_index()[["time"] + out_cols]

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_FILE, index=False)
    print(f"Wrote: {OUT_FILE} rows={len(out):,}")
    print(out.tail(6).to_string(index=False))


if __name__ == "__main__":
    main()
