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

1. Download 1991 to 2025 (This will pull purely from the stable ERA5 archive).

2. Download 2026 (This will pull purely from the ERA5T archive).

3. Use xarray.open_mfdataset() to smoothly join them together in Python.
"""
import cdsapi
import xarray as xr
import zipfile
import os
import glob
import shutil
import pandas as pd

c = cdsapi.Client()
area_bbox = [40.6, -122.5, 35.4, -119.0]

zip_1991_2025 = 'data/processed/era5_land_monthly_cvalley_1991_2025.nc'
zip_2026 = 'data/processed/era5_land_monthly_cvalley_2026.nc'

print("Starting downloads from Copernicus CDS...")

c.retrieve('reanalysis-era5-land-monthly-means', {
    'product_type': 'monthly_averaged_reanalysis',
    'variable': 'total_precipitation',
    'year': [str(y) for y in range(1991, 2026)],
    'month': [f'{m:02d}' for m in range(1, 13)],
    'time': '00:00',
    'area': area_bbox,
    'format': 'netcdf',
}, zip_1991_2025)

c.retrieve('reanalysis-era5-land-monthly-means', {
    'product_type': 'monthly_averaged_reanalysis',
    'variable': 'total_precipitation',
    'year': ['2026'],
    'month': ['01', '02'],
    'time': '00:00',
    'area': area_bbox,
    'format': 'netcdf',
}, zip_2026)

print("Downloads complete. Extracting ZIP archives...")

temp_dir_1 = 'data/processed/temp_era5_1991_2025'
temp_dir_2 = 'data/processed/temp_era5_2026'
os.makedirs(temp_dir_1, exist_ok=True)
os.makedirs(temp_dir_2, exist_ok=True)

with zipfile.ZipFile(zip_1991_2025, 'r') as zip_ref:
    zip_ref.extractall(temp_dir_1)
with zipfile.ZipFile(zip_2026, 'r') as zip_ref:
    zip_ref.extractall(temp_dir_2)

actual_nc_files = []
actual_nc_files.extend(glob.glob(f"{temp_dir_1}/*.nc"))
actual_nc_files.extend(glob.glob(f"{temp_dir_2}/*.nc"))

print(f"Found extracted files: {actual_nc_files}")
print("Merging data using xarray...")

datasets = []
for fp in actual_nc_files:
    ds = xr.open_dataset(fp)

    if "valid_time" in ds.dims:
        ds = ds.rename({"valid_time": "time"})
    elif "valid_time" in ds.coords:
        ds = ds.rename({"valid_time": "time"})

    if "expver" in ds.dims:
        ds = ds.mean(dim="expver", skipna=True)

    if "expver" in ds.variables and "expver" not in ds.dims:
        ds = ds.drop_vars("expver")

    datasets.append(ds)

era5_ds = xr.concat(datasets, dim="time")
era5_ds = era5_ds.sortby("time")

time_index = pd.Index(era5_ds["time"].values)
era5_ds = era5_ds.isel(time=~time_index.duplicated())

target_file = 'data/processed/era5_land_monthly_cvalley_1991_2026.nc'
era5_ds.to_netcdf(target_file)

print(f"Successfully combined and saved to: {target_file}")

print("Cleaning up temporary files...")
shutil.rmtree(temp_dir_1)
shutil.rmtree(temp_dir_2)
os.remove(zip_1991_2025)
os.remove(zip_2026)

print("Process finished flawlessly!")
