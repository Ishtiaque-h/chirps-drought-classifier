"""
ERA5 vs. ERA5T: The official, finalized ERA5 dataset usually lags about 2 to 3 months behind the current date.
For the most recent months (like early 2026), Copernicus uses a preliminary dataset called ERA5T (Near Real-Time).

Merge Failure: When the requested date range crosses the boundary between the historical ERA5 data and
the preliminary ERA5T data, the CDS backend tries to stitch them together on the fly.

The Corruption: When the new CDS API infrastructure converts this mixed-stream data from GRIB to NetCDF,
it often wraps it improperly or prepends stream headers.

Xarray error:  Because the standard HDF5/NetCDF magic signature is displaced from the top of the file by this bad formatting,
Xarray throws an esception.

Solution: Split download

Download ERA5-Land monthly total precipitation for the Central Valley,
automatically covering:

- all full years from START_YEAR through last calendar year
- all currently available months of the current year

Why split the download?
When requests cross the boundary between finalized ERA5-Land data and
near-real-time ERA5-Land-T data, CDS NetCDF conversion can sometimes
produce malformed files. Splitting historical years and current-year
months avoids that problem.

Output:
  data/processed/era5_land_monthly_cvalley_<START_YEAR>_<CURRENT_YEAR>.nc
"""

from pathlib import Path
from datetime import datetime, timezone
import cdsapi
import xarray as xr
import zipfile
import glob
import shutil
import os
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "processed"
DATA_DIR.mkdir(parents=True, exist_ok=True)

START_YEAR = 1991
AREA_BBOX = [40.6, -122.5, 35.4, -119.0]  # N, W, S, E


def get_current_year_and_available_months() -> tuple[int, list[str]]:
    """
    Determine the current year and the monthly ERA5-Land months that should
    be available in CDS.

    ECMWF states monthly ERA5-Land-T updates are generally available about
    5 days after the end of the month. So:
      - on day 6 or later, assume previous month is available
      - before day 6, be conservative and stop two months back
    """
    today = datetime.now(timezone.utc)
    current_year = today.year

    if today.day >= 6:
        last_available_month = today.month - 1
    else:
        last_available_month = today.month - 2

    if last_available_month < 1:
        available_months = []
    else:
        available_months = [f"{m:02d}" for m in range(1, last_available_month + 1)]

    return current_year, available_months


def extract_zip_to_temp(zip_path: Path, temp_dir: Path) -> list[str]:
    """Extract CDS zip payload and return extracted NetCDF files."""
    temp_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(temp_dir)
    return glob.glob(str(temp_dir / "*.nc"))


def standardize_dataset(ds: xr.Dataset) -> xr.Dataset:
    """
    Standardize dimensions/coords before concatenation.
    """
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
    """Run one CDS request if there is anything to request."""
    if not years or not months:
        return

    client.retrieve(
        "reanalysis-era5-land-monthly-means",
        {
            "product_type": "monthly_averaged_reanalysis",
            "variable": "total_precipitation",
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

    zip_hist = DATA_DIR / f"era5_land_monthly_cvalley_{START_YEAR}_{current_year - 1}.zip"
    zip_curr = DATA_DIR / f"era5_land_monthly_cvalley_{current_year}.zip"
    target_file = DATA_DIR / f"era5_land_monthly_cvalley_{START_YEAR}_{current_year}.nc"

    temp_hist = DATA_DIR / f"temp_era5_{START_YEAR}_{current_year - 1}"
    temp_curr = DATA_DIR / f"temp_era5_{current_year}"

    extracted_files: list[str] = []
    client = cdsapi.Client()

    print("Starting downloads from Copernicus CDS...")
    print(f"Project start year: {START_YEAR}")
    print(f"Current year: {current_year}")
    print(f"Current-year available months: {current_year_months or 'none yet'}")

    try:
        # 1) Historical finalized years
        if historical_years:
            print(f"Downloading finalized historical ERA5-Land: {START_YEAR} to {current_year - 1}")
            download_request(client, historical_years, all_months, zip_hist)

            print("Extracting historical archive...")
            extracted_files.extend(extract_zip_to_temp(zip_hist, temp_hist))
        else:
            print("No historical years to download.")

        # 2) Current-year near-real-time months
        if current_year_months:
            print(f"Downloading current-year ERA5-Land/ERA5-Land-T months: {current_year_months}")
            download_request(client, [str(current_year)], current_year_months, zip_curr)

            print("Extracting current-year archive...")
            extracted_files.extend(extract_zip_to_temp(zip_curr, temp_curr))
        else:
            print("No current-year monthly means appear available yet.")

        if not extracted_files:
            raise RuntimeError("No NetCDF files were extracted from CDS downloads.")

        print(f"Found extracted files: {extracted_files}")
        print("Merging data using xarray...")

        datasets = []
        for fp in extracted_files:
            ds = xr.open_dataset(fp)
            ds = standardize_dataset(ds)
            datasets.append(ds)

        era5_ds = xr.concat(datasets, dim="time")
        era5_ds = era5_ds.sortby("time")

        # Drop duplicate months if any
        time_index = pd.Index(era5_ds["time"].values)
        era5_ds = era5_ds.isel(time=~time_index.duplicated())

        era5_ds.to_netcdf(target_file)
        print(f"Successfully combined and saved to: {target_file}")

    finally:
        print("Cleaning up temporary files...")
        for path in [temp_hist, temp_curr]:
            if path.exists():
                shutil.rmtree(path)

        for path in [zip_hist, zip_curr]:
            if path.exists():
                os.remove(path)

    print("Process finished successfully!")


if __name__ == "__main__":
    main()