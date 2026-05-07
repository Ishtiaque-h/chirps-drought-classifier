#!/usr/bin/env python
"""
Download ERA5-Land monthly volumetric soil water layers for a configured region.

This supports isolated soil-moisture feature experiments without changing the
canonical CHIRPS/SPI forecast pipeline.

Output:
  data/processed/era5_land_soil_moisture_monthly_<region>_<START_YEAR>_<CURRENT_YEAR>.nc
"""
from __future__ import annotations

from argparse import ArgumentParser, Namespace
from datetime import datetime, timezone
from pathlib import Path
import glob
import os
import shutil
import zipfile

import cdsapi
import pandas as pd
import xarray as xr

from region_config import resolve_region


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "processed"
DATA_DIR.mkdir(parents=True, exist_ok=True)

START_YEAR = 1991
VARIABLES = [
    "volumetric_soil_water_layer_1",
    "volumetric_soil_water_layer_2",
    "volumetric_soil_water_layer_3",
    "volumetric_soil_water_layer_4",
]


def parse_args() -> Namespace:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--region", default="cvalley")
    parser.add_argument("--start-year", type=int, default=START_YEAR)
    parser.add_argument(
        "--end-year",
        type=int,
        default=None,
        help=(
            "Final year to download. Defaults to the current available ERA5-Land year. "
            "Use 2025 to avoid current-year ERA5T access restrictions."
        ),
    )
    parser.add_argument(
        "--out-file",
        type=Path,
        default=None,
        help="Defaults to data/processed/era5_land_soil_moisture_monthly_<region>_<start>_<current>.nc.",
    )
    return parser.parse_args()


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
    area_bbox: list[float],
) -> None:
    if not years or not months:
        return

    client.retrieve(
        "reanalysis-era5-land-monthly-means",
        {
            "product_type": "monthly_averaged_reanalysis",
            "variable": VARIABLES,
            "year": years,
            "month": months,
            "time": "00:00",
            "area": area_bbox,
            "data_format": "netcdf",
        },
        str(out_path),
    )


def main() -> None:
    args = parse_args()
    region = resolve_region(args.region)
    area_bbox = [region.lat_max, region.lon_min, region.lat_min, region.lon_max]  # N, W, S, E
    current_year, current_year_months = get_current_year_and_available_months()
    end_year = args.end_year or current_year
    if end_year > current_year:
        raise ValueError(f"--end-year {end_year} is in the future relative to current year {current_year}.")
    historical_last_year = min(end_year, current_year - 1)
    historical_years = [str(y) for y in range(args.start_year, historical_last_year + 1)]
    include_current_year = end_year >= current_year
    all_months = [f"{m:02d}" for m in range(1, 13)]

    payload_hist = DATA_DIR / f"era5_land_soil_moisture_monthly_{region.slug}_{args.start_year}_{historical_last_year}.zip"
    payload_curr = DATA_DIR / f"era5_land_soil_moisture_monthly_{region.slug}_{current_year}.zip"
    target_file = args.out_file or (
        DATA_DIR / f"era5_land_soil_moisture_monthly_{region.slug}_{args.start_year}_{end_year}.nc"
    )

    temp_hist = DATA_DIR / f"temp_era5_soil_{region.slug}_{args.start_year}_{historical_last_year}"
    temp_curr = DATA_DIR / f"temp_era5_soil_{region.slug}_{current_year}"

    extracted_files: list[str] = []
    client = cdsapi.Client()

    print("Starting ERA5-Land soil-moisture downloads from Copernicus CDS...")
    print(f"Region: {region.slug} ({region.name})")
    print(f"Area bbox [N, W, S, E]: {area_bbox}")
    print(f"Variables: {VARIABLES}")
    print(f"Project start year: {args.start_year}")
    print(f"Requested end year: {end_year}")
    print(f"Current year: {current_year}")
    print(f"Current-year available months: {current_year_months or 'none yet'}")

    try:
        if historical_years:
            print(f"Downloading historical ERA5-Land soil moisture: {args.start_year} to {current_year - 1}")
            download_request(client, historical_years, all_months, payload_hist, area_bbox)
            extracted_files.extend(extracted_or_plain_netcdfs(payload_hist, temp_hist))

        if include_current_year and current_year_months:
            print(f"Downloading current-year ERA5-Land soil months: {current_year_months}")
            download_request(client, [str(current_year)], current_year_months, payload_curr, area_bbox)
            extracted_files.extend(extracted_or_plain_netcdfs(payload_curr, temp_curr))
        elif include_current_year:
            print("No current-year monthly means appear available yet.")

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
