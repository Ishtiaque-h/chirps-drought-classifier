#!/usr/bin/env python
"""
Download ERA5 monthly IVT components for Central Valley.

Output:
  data/processed/era5_ivt_monthly_cvalley_<START_YEAR>_<CURRENT_YEAR>.nc
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import glob
import os
import shutil
import zipfile

import cdsapi
import pandas as pd
import xarray as xr


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "processed"
DATA_DIR.mkdir(parents=True, exist_ok=True)

START_YEAR = 1991
AREA_BBOX = [40.6, -122.5, 35.4, -119.0]  # N, W, S, E
VARIABLES = [
    "vertical_integral_of_eastward_water_vapour_flux",
    "vertical_integral_of_northward_water_vapour_flux",
]


def get_current_year_and_available_months() -> tuple[int, list[str]]:
    today = datetime.now(timezone.utc)
    current_year = today.year
    last_available_month = today.month - 1 if today.day >= 6 else today.month - 2
    if last_available_month < 1:
        return current_year, []
    return current_year, [f"{m:02d}" for m in range(1, last_available_month + 1)]


def extracted_or_plain_netcdfs(payload_path: Path, temp_dir: Path) -> list[str]:
    temp_dir.mkdir(parents=True, exist_ok=True)
    if zipfile.is_zipfile(payload_path):
        with zipfile.ZipFile(payload_path, "r") as zip_ref:
            zip_ref.extractall(temp_dir)
        files = glob.glob(str(temp_dir / "*.nc"))
        files.extend(glob.glob(str(temp_dir / "*.netcdf")))
        if not files:
            files.extend(glob.glob(str(temp_dir / "*")))
        return files
    return [str(payload_path)]


def standardize_dataset(ds: xr.Dataset) -> xr.Dataset:
    rename_map = {}
    if "valid_time" in ds.dims:
        rename_map["valid_time"] = "time"
    if "lat" in ds.dims:
        rename_map["lat"] = "latitude"
    if "lon" in ds.dims:
        rename_map["lon"] = "longitude"
    if rename_map:
        ds = ds.rename(rename_map)

    if "expver" in ds.dims:
        ds = ds.mean(dim="expver", skipna=True)
    if "expver" in ds.variables and "expver" not in ds.dims:
        ds = ds.drop_vars("expver")

    return ds


def download_request(
    client: cdsapi.Client,
    years: list[str],
    months: list[str],
    out_path: Path,
) -> None:
    if not years or not months:
        return

    client.retrieve(
        "reanalysis-era5-single-levels-monthly-means",
        {
            "product_type": "monthly_averaged_reanalysis",
            "variable": VARIABLES,
            "year": years,
            "month": months,
            "time": "00:00",
            "area": AREA_BBOX,
            "format": "netcdf",
        },
        str(out_path),
    )


def main() -> None:
    current_year, current_year_months = get_current_year_and_available_months()
    historical_years = [str(y) for y in range(START_YEAR, current_year)]
    all_months = [f"{m:02d}" for m in range(1, 13)]

    payload_hist = DATA_DIR / f"era5_ivt_monthly_cvalley_{START_YEAR}_{current_year - 1}.zip"
    payload_curr = DATA_DIR / f"era5_ivt_monthly_cvalley_{current_year}.zip"
    target_file = DATA_DIR / f"era5_ivt_monthly_cvalley_{START_YEAR}_{current_year}.nc"

    temp_hist = DATA_DIR / f"temp_era5_ivt_{START_YEAR}_{current_year - 1}"
    temp_curr = DATA_DIR / f"temp_era5_ivt_{current_year}"

    extracted_files: list[str] = []
    client = cdsapi.Client()

    print("Starting ERA5 IVT downloads from Copernicus CDS...")
    print(f"Variables: {VARIABLES}")
    print(f"Project start year: {START_YEAR}")
    print(f"Current year: {current_year}")
    print(f"Current-year available months: {current_year_months or 'none yet'}")

    try:
        if historical_years:
            print(f"Downloading historical ERA5 IVT data: {START_YEAR} to {current_year - 1}")
            download_request(client, historical_years, all_months, payload_hist)
            extracted_files.extend(extracted_or_plain_netcdfs(payload_hist, temp_hist))

        if current_year_months:
            print(f"Downloading current-year ERA5 IVT months: {current_year_months}")
            download_request(client, [str(current_year)], current_year_months, payload_curr)
            extracted_files.extend(extracted_or_plain_netcdfs(payload_curr, temp_curr))

        if not extracted_files:
            raise RuntimeError("No NetCDF files were produced by CDS downloads.")

        print(f"Found extracted files: {extracted_files}")
        datasets = []
        for fp in extracted_files:
            ds = xr.open_dataset(fp)
            datasets.append(standardize_dataset(ds))

        out = xr.concat(datasets, dim="time").sortby("time")
        time_index = pd.Index(out["time"].values)
        out = out.isel(time=~time_index.duplicated())
        out.to_netcdf(target_file)
        print(f"Successfully saved: {target_file}")

    finally:
        print("Cleaning up temporary files...")
        for path in [temp_hist, temp_curr]:
            if path.exists():
                shutil.rmtree(path)
        for path in [payload_hist, payload_curr]:
            if path.exists():
                os.remove(path)

    print("Process finished successfully.")


if __name__ == "__main__":
    main()
