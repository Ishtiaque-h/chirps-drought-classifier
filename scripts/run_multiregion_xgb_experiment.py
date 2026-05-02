#!/usr/bin/env python
"""
Run a leakage-free XGBoost drought-forecast experiment for a named region.

This region-aware evaluation path is deliberately non-destructive:
  - canonical Central Valley files are reused by default for --region cvalley
  - new region products are written under data/processed/regions/<region>/
  - model artifacts are written under outputs/multiregion/<region>/

Workflow per region:
  1. Clip global CHIRPS v3 monthly files to a rectangular region bbox.
  2. Compute WMO-style SPI-1/3/6 and SPI-1 drought labels per pixel.
  3. Build the same SPI-1[t+1] forecast table used by the canonical pipeline.
  4. Train/evaluate tabular and/or 3x3-neighbourhood XGBoost.

Primary metric:
  Monthly dry-fraction Brier Skill Score (BSS) vs train-period monthly
  climatology, with monthly bootstrap confidence intervals.
"""
from __future__ import annotations

from argparse import ArgumentParser, Namespace
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
import json
import re
import shutil
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
import xgboost as xgb
from joblib import Parallel, delayed
from scipy.stats import gamma as gamma_dist, norm
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.utils.class_weight import compute_sample_weight

from feature_config import get_feature_columns
from region_config import REGIONS, Region, region_table, resolve_region


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_CHIRPS = PROJECT_ROOT / "data" / "raw" / "chirps_v3" / "monthly"
PROCESSED = PROJECT_ROOT / "data" / "processed"
REGION_DATA_ROOT = PROCESSED / "regions"
OUT_ROOT = PROJECT_ROOT / "outputs" / "multiregion"
REPORT_ROOT = PROJECT_ROOT / "results" / "multiregion"
CLIMATE_FILE = PROCESSED / "climate_indices_monthly.csv"

BASELINE_START_YEAR = 1991
BASELINE_END_YEAR = 2020
TRAIN_END_YEAR = 2016
VAL_START_YEAR = 2017
VAL_END_YEAR = 2020
TEST_START_YEAR = 2021
MISSING_SENTINELS = (-9.9, -99.99, -999.0)
LABEL_MAP = {-1: 0, 0: 1, 1: 2}
INV_LABEL_MAP = {v: k for k, v in LABEL_MAP.items()}
TARGET = "target_label"
SPATIAL_FEATURES = [
    "spi1_nbr_mean",
    "spi3_nbr_mean",
    "spi6_nbr_mean",
    "pr_nbr_mean",
]


def parse_args() -> Namespace:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--region", default="cvalley", help="Region slug or alias.")
    parser.add_argument("--list-regions", action="store_true", help="List configured regions and exit.")
    parser.add_argument(
        "--model",
        choices=["tabular", "spatial", "both"],
        default="both",
        help="Which XGBoost feature set to evaluate.",
    )
    parser.add_argument(
        "--climate-features",
        choices=["none", "nino34", "pdo", "all"],
        default="nino34",
        help="Optional climate-index features to merge into the forecast table.",
    )
    parser.add_argument("--start-year", type=int, default=BASELINE_START_YEAR)
    parser.add_argument(
        "--end-year",
        type=int,
        default=None,
        help="Last CHIRPS year to use. Defaults to the latest local raw CHIRPS year.",
    )
    parser.add_argument("--rebuild-pr", action="store_true", help="Rebuild clipped CHIRPS region file.")
    parser.add_argument("--rebuild-spi", action="store_true", help="Recompute region SPI file.")
    parser.add_argument("--rebuild-dataset", action="store_true", help="Rebuild region forecast parquet.")
    parser.add_argument(
        "--grid-stride",
        type=int,
        default=1,
        help=(
            "Keep every Nth latitude/longitude after region clipping. "
            "Use only for smoke tests; stride > 1 is not a publication result."
        ),
    )
    parser.add_argument(
        "--no-canonical-cvalley",
        action="store_true",
        help="For cvalley, do not reuse canonical data/processed files.",
    )
    parser.add_argument("--n-bootstrap", type=int, default=2000)
    parser.add_argument(
        "--spi-n-jobs",
        type=int,
        default=1,
        help="Parallel jobs for SPI fitting. Use >1 for large full-resolution regions.",
    )
    parser.add_argument("--num-boost-round", type=int, default=2000)
    parser.add_argument("--early-stopping-rounds", type=int, default=50)
    parser.add_argument("--verbose-eval", type=int, default=100)
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Stop after clipping/SPI/dataset build; do not train models.",
    )
    parser.add_argument(
        "--prepare-grid-only",
        action="store_true",
        help="Stop after clipping CHIRPS and computing SPI; do not build a forecast table.",
    )
    parser.add_argument(
        "--copy-report",
        action="store_true",
        help="Copy score text, summary CSV, and feature plots into results/multiregion/.",
    )
    parser.add_argument(
        "--country-mask",
        action="store_true",
        help=(
            "Filter the forecast table to the configured country mask and write a "
            "separate '<region>_country_masked' sensitivity run."
        ),
    )
    parser.add_argument(
        "--basin-mask",
        action="store_true",
        help=(
            "Filter the forecast table to the configured basin/hydroclimate mask and write a "
            "separate '<region>_basin_masked' sensitivity run."
        ),
    )
    parser.add_argument(
        "--mask-file",
        type=Path,
        default=None,
        help="Optional mask NetCDF. Defaults to the selected country or basin mask product.",
    )
    parser.add_argument(
        "--mask-var",
        default=None,
        help="Optional mask variable name inside --mask-file. Defaults from --country-mask/--basin-mask.",
    )
    return parser.parse_args()


def available_chirps_files(start_year: int, end_year: int | None) -> list[Path]:
    files = sorted(RAW_CHIRPS.glob("chirps-v3.0.*.monthly.nc"))
    selected = []
    for path in files:
        match = re.search(r"chirps-v3\.0\.(\d{4})\.monthly\.nc$", path.name)
        if not match:
            continue
        year = int(match.group(1))
        if year < start_year:
            continue
        if end_year is not None and year > end_year:
            continue
        selected.append(path)
    if not selected:
        raise FileNotFoundError(f"No CHIRPS monthly files found in {RAW_CHIRPS}")
    return selected


def latest_chirps_year() -> int:
    files = available_chirps_files(BASELINE_START_YEAR, None)
    years = []
    for path in files:
        match = re.search(r"chirps-v3\.0\.(\d{4})\.monthly\.nc$", path.name)
        if match:
            years.append(int(match.group(1)))
    return max(years)


def region_dir(region: Region) -> Path:
    return REGION_DATA_ROOT / region.slug


def output_dir(region: Region) -> Path:
    return OUT_ROOT / region.slug


def report_dir(region: Region) -> Path:
    return REPORT_ROOT / region.slug


def region_paths(region: Region, start_year: int, end_year: int, use_canonical: bool) -> dict[str, Path]:
    if region.slug == "cvalley" and use_canonical:
        return {
            "pr": PROCESSED / f"chirps_v3_monthly_cvalley_{start_year}_{end_year}.nc",
            "spi": PROCESSED / f"chirps_v3_monthly_cvalley_spi_{start_year}_{end_year}.nc",
            "dataset": PROCESSED / "dataset_forecast.parquet",
            "sample": PROCESSED / "dataset_forecast_sample.csv",
        }
    rdir = region_dir(region)
    return {
        "pr": rdir / f"chirps_v3_monthly_{region.slug}_{start_year}_{end_year}.nc",
        "spi": rdir / f"chirps_v3_monthly_{region.slug}_spi_{start_year}_{end_year}.nc",
        "dataset": rdir / f"dataset_forecast_{region.slug}.parquet",
        "sample": rdir / f"dataset_forecast_{region.slug}_sample.csv",
    }


def default_country_mask_file(region: Region) -> Path:
    return REGION_DATA_ROOT / region.slug / "masks" / f"{region.slug}_country_mask.nc"


def default_basin_mask_file(region: Region) -> Path:
    return REGION_DATA_ROOT / region.slug / "masks" / f"{region.slug}_basin_mask.nc"


def masked_dataset_paths(region: Region) -> dict[str, Path]:
    rdir = region_dir(region)
    return {
        "dataset": rdir / f"dataset_forecast_{region.slug}.parquet",
        "sample": rdir / f"dataset_forecast_{region.slug}_sample.csv",
    }


def select_lat_lon(ds: xr.Dataset, region: Region, grid_stride: int = 1) -> xr.Dataset:
    lat_name = "latitude" if "latitude" in ds.coords else "lat"
    lon_name = "longitude" if "longitude" in ds.coords else "lon"

    lat_min, lat_max = sorted((region.lat_min, region.lat_max))
    if float(ds[lat_name][0]) > float(ds[lat_name][-1]):
        ds = ds.sel({lat_name: slice(lat_max, lat_min)})
    else:
        ds = ds.sel({lat_name: slice(lat_min, lat_max)})

    lon_min, lon_max = sorted((region.lon_min, region.lon_max))
    ds = ds.sel({lon_name: slice(lon_min, lon_max)})
    if grid_stride > 1:
        ds = ds.isel({lat_name: slice(None, None, grid_stride), lon_name: slice(None, None, grid_stride)})
    return ds


def clip_chirps(
    region: Region,
    out_file: Path,
    start_year: int,
    end_year: int,
    force: bool,
    grid_stride: int,
) -> None:
    if out_file.exists() and not force:
        print(f"Using existing clipped CHIRPS file: {out_file}")
        return

    files = available_chirps_files(start_year, end_year)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    print(f"Clipping CHIRPS for {region.name}: {len(files)} yearly files")
    print(f"Region bbox lat[{region.lat_min}, {region.lat_max}] lon[{region.lon_min}, {region.lon_max}]")
    if grid_stride > 1:
        print(f"Grid stride smoke mode: keeping every {grid_stride}th lat/lon cell")

    clipped = []
    for idx, file in enumerate(files, start=1):
        print(f"  clipping {file.name} ({idx}/{len(files)})")
        part = xr.open_dataset(file)
        part = select_lat_lon(part, region, grid_stride=grid_stride)
        if part.sizes.get("latitude", part.sizes.get("lat", 0)) == 0 or part.sizes.get(
            "longitude", part.sizes.get("lon", 0)
        ) == 0:
            raise ValueError(f"Region selection produced an empty grid for {region.slug}")
        var = next(
            (v for v in part.data_vars if v.lower().startswith(("precip", "pr"))),
            list(part.data_vars)[0],
        )
        part = part[[var]].rename({var: "pr"})
        if "lat" in part.coords:
            part = part.rename({"lat": "latitude"})
        if "lon" in part.coords:
            part = part.rename({"lon": "longitude"})
        clipped.append(part.load())

    ds = xr.concat(clipped, dim="time").sortby("time")
    time_index = pd.Index(ds["time"].values)
    ds = ds.isel(time=~time_index.duplicated())

    encoding = {"pr": dict(zlib=True, complevel=4)}
    ds.to_netcdf(out_file, encoding=encoding)
    print(f"Wrote clipped CHIRPS: {out_file}")
    print("Dims:", {k: int(v) for k, v in ds.sizes.items()})


def precip_to_spi(series: np.ndarray, baseline_mask: np.ndarray) -> np.ndarray:
    base = series[baseline_mask]
    base = base[~np.isnan(base)]
    spi = np.full(len(series), np.nan, dtype=np.float32)

    if len(base) < 10:
        return spi
    p_zero = np.mean(base == 0)
    nonzero = base[base > 0]
    if len(nonzero) < 5:
        return spi

    fit_alpha, _, fit_beta = gamma_dist.fit(nonzero, floc=0)
    for i, val in enumerate(series):
        if np.isnan(val):
            continue
        if val == 0:
            cdf_val = p_zero
        else:
            cdf_val = p_zero + (1.0 - p_zero) * gamma_dist.cdf(val, fit_alpha, scale=fit_beta)
        cdf_val = np.clip(cdf_val, 1e-6, 1 - 1e-6)
        spi[i] = norm.ppf(cdf_val)
    return spi


def rolling_sum(arr3d: np.ndarray, window: int) -> np.ndarray:
    out = np.full_like(arr3d, np.nan, dtype=np.float64)
    for t in range(window - 1, arr3d.shape[0]):
        block = arr3d[t - window + 1 : t + 1]
        out[t] = np.nansum(block, axis=0)
        has_nan = np.any(np.isnan(block), axis=0)
        out[t][has_nan] = np.nan
    return out


def rolling_sum_1d(series: np.ndarray, window: int) -> np.ndarray:
    out = np.full(len(series), np.nan, dtype=np.float64)
    for t in range(window - 1, len(series)):
        block = series[t - window + 1 : t + 1]
        if np.any(np.isnan(block)):
            continue
        out[t] = np.sum(block)
    return out


def compute_spi_for_pixel(
    series: np.ndarray,
    months_arr: np.ndarray,
    baseline_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    spi1 = np.full(len(series), np.nan, dtype=np.float32)
    spi3 = np.full(len(series), np.nan, dtype=np.float32)
    spi6 = np.full(len(series), np.nan, dtype=np.float32)
    rolled = {
        1: series,
        3: rolling_sum_1d(series, 3),
        6: rolling_sum_1d(series, 6),
    }
    out = {1: spi1, 3: spi3, 6: spi6}
    for window, values in rolled.items():
        for month in range(1, 13):
            month_idx = np.where(months_arr == month)[0]
            base_idx = np.where(baseline_mask & (months_arr == month))[0]
            mask_for_month = np.zeros(len(series), dtype=bool)
            mask_for_month[base_idx] = True
            base_within_month = mask_for_month[month_idx]
            out[window][month_idx] = precip_to_spi(values[month_idx], base_within_month)
    return spi1, spi3, spi6


def compute_spi_parallel(
    pr_vals: np.ndarray,
    months_arr: np.ndarray,
    baseline_mask: np.ndarray,
    n_jobs: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ntimes, nlat, nlon = pr_vals.shape
    n_pixels = nlat * nlon
    flat = pr_vals.reshape(ntimes, n_pixels).T
    print(f"Parallel SPI fitting across {n_pixels:,} pixels with n_jobs={n_jobs}...")
    results = Parallel(n_jobs=n_jobs, verbose=10, batch_size=64)(
        delayed(compute_spi_for_pixel)(flat[i], months_arr, baseline_mask)
        for i in range(n_pixels)
    )
    spi1 = np.empty((ntimes, n_pixels), dtype=np.float32)
    spi3 = np.empty((ntimes, n_pixels), dtype=np.float32)
    spi6 = np.empty((ntimes, n_pixels), dtype=np.float32)
    for i, (pix_spi1, pix_spi3, pix_spi6) in enumerate(results):
        spi1[:, i] = pix_spi1
        spi3[:, i] = pix_spi3
        spi6[:, i] = pix_spi6
    return (
        spi1.reshape(ntimes, nlat, nlon),
        spi3.reshape(ntimes, nlat, nlon),
        spi6.reshape(ntimes, nlat, nlon),
    )


def compute_spi_from_rolled(
    rolled_vals: np.ndarray,
    months_arr: np.ndarray,
    baseline_mask: np.ndarray,
    window: int,
) -> np.ndarray:
    ntimes, nlat, nlon = rolled_vals.shape
    spi_out = np.full_like(rolled_vals, np.nan, dtype=np.float32)
    for month in range(1, 13):
        month_idx = np.where(months_arr == month)[0]
        base_idx = np.where(baseline_mask & (months_arr == month))[0]
        mask_for_month = np.zeros(ntimes, dtype=bool)
        mask_for_month[base_idx] = True
        base_within_month = mask_for_month[month_idx]
        for j in range(nlat):
            for k in range(nlon):
                spi_out[month_idx, j, k] = precip_to_spi(
                    rolled_vals[month_idx, j, k],
                    base_within_month,
                )
        print(f"  SPI-{window} month {month:02d}/12 done")
    return spi_out


def make_spi_labels(pr_file: Path, spi_file: Path, force: bool, n_jobs: int) -> None:
    if spi_file.exists() and not force:
        print(f"Using existing SPI file: {spi_file}")
        return

    print(f"Loading clipped CHIRPS for SPI: {pr_file}")
    ds = xr.open_dataset(pr_file).load()
    pr = ds["pr"]
    pr_vals = pr.values.astype(np.float64)
    times = pd.DatetimeIndex(pr.time.values)
    months_arr = times.month.to_numpy()
    baseline_mask = (times.year >= BASELINE_START_YEAR) & (times.year <= BASELINE_END_YEAR)
    ntimes, nlat, nlon = pr_vals.shape

    if n_jobs > 1:
        spi1_vals, spi3_vals, spi6_vals = compute_spi_parallel(
            pr_vals, months_arr, baseline_mask, n_jobs=n_jobs
        )
    else:
        print(f"Computing SPI-1 for grid {nlat} x {nlon} across {ntimes} months...")
        spi1_vals = np.full_like(pr_vals, np.nan, dtype=np.float32)
        for month in range(1, 13):
            month_idx = np.where(months_arr == month)[0]
            base_idx = np.where(baseline_mask & (months_arr == month))[0]
            mask_for_month = np.zeros(ntimes, dtype=bool)
            mask_for_month[base_idx] = True
            base_within_month = mask_for_month[month_idx]
            for j in range(nlat):
                for k in range(nlon):
                    spi1_vals[month_idx, j, k] = precip_to_spi(
                        pr_vals[month_idx, j, k],
                        base_within_month,
                    )
            print(f"  SPI-1 month {month:02d}/12 done")

        print("Computing SPI-3 and SPI-6 rolling precipitation sums...")
        pr3_vals = rolling_sum(pr_vals, 3)
        pr6_vals = rolling_sum(pr_vals, 6)

        print("Computing SPI-3...")
        spi3_vals = compute_spi_from_rolled(pr3_vals, months_arr, baseline_mask, 3)
        print("Computing SPI-6...")
        spi6_vals = compute_spi_from_rolled(pr6_vals, months_arr, baseline_mask, 6)

    label1_vals = np.full_like(spi1_vals, np.nan, dtype=np.float32)
    finite1 = np.isfinite(spi1_vals)
    label1_vals[finite1] = np.where(
        spi1_vals[finite1] <= -1.0,
        -1,
        np.where(spi1_vals[finite1] >= 1.0, 1, 0),
    )

    label3_vals = np.full_like(spi3_vals, np.nan, dtype=np.float32)
    finite3 = np.isfinite(spi3_vals)
    label3_vals[finite3] = np.where(
        spi3_vals[finite3] <= -1.0,
        -1,
        np.where(spi3_vals[finite3] >= 1.0, 1, 0),
    )

    coords = {"time": pr.time, "latitude": pr.latitude, "longitude": pr.longitude}
    dims = ("time", "latitude", "longitude")
    out_ds = xr.Dataset(
        {
            "spi1": xr.DataArray(spi1_vals, coords=coords, dims=dims),
            "spi3": xr.DataArray(spi3_vals, coords=coords, dims=dims),
            "spi6": xr.DataArray(spi6_vals, coords=coords, dims=dims),
            "drought_label_spi1": xr.DataArray(label1_vals, coords=coords, dims=dims),
            "drought_label_spi3": xr.DataArray(label3_vals, coords=coords, dims=dims),
        }
    )
    enc = {v: {"zlib": True, "complevel": 4} for v in out_ds.data_vars}
    spi_file.parent.mkdir(parents=True, exist_ok=True)
    print(f"Saving SPI labels: {spi_file}")
    out_ds.to_netcdf(spi_file, encoding=enc)

    counts = {
        label: int(np.nansum(label1_vals == value))
        for label, value in {"dry": -1, "normal": 0, "wet": 1}.items()
    }
    print("SPI-1 label counts:", counts)


def load_climate_features(times: pd.DatetimeIndex, climate_features: str) -> tuple[pd.DataFrame, list[str]]:
    if climate_features == "none":
        return pd.DataFrame({"time": times}), []
    if not CLIMATE_FILE.exists():
        raise FileNotFoundError(
            f"Climate feature request '{climate_features}' requires {CLIMATE_FILE}"
        )

    selected = []
    if climate_features in {"all", "nino34"}:
        selected.append("nino34")
    if climate_features in {"all", "pdo"}:
        selected.append("pdo")

    cdf = pd.read_csv(CLIMATE_FILE)
    required = {"time", *selected}
    missing = required.difference(cdf.columns)
    if missing:
        raise ValueError(f"{CLIMATE_FILE} is missing columns: {sorted(missing)}")

    cdf["time"] = pd.to_datetime(cdf["time"]).dt.to_period("M").dt.to_timestamp()
    cdf = (
        cdf[["time"] + selected]
        .sort_values("time")
        .drop_duplicates(subset=["time"], keep="last")
        .set_index("time")
    )
    cdf[selected] = cdf[selected].replace(list(MISSING_SENTINELS), np.nan)
    cdf = cdf.reindex(times).sort_index()
    cdf[selected] = cdf[selected].interpolate(method="time", limit_area="inside")

    exog_cols = []
    if "nino34" in selected:
        cdf["nino34_lag1"] = cdf["nino34"]
        cdf["nino34_lag2"] = cdf["nino34"].shift(1)
        exog_cols.extend(["nino34_lag1", "nino34_lag2"])
    if "pdo" in selected:
        cdf["pdo_lag1"] = cdf["pdo"]
        cdf["pdo_lag2"] = cdf["pdo"].shift(1)
        exog_cols.extend(["pdo_lag1", "pdo_lag2"])

    out = cdf.drop(columns=selected).reset_index().rename(columns={"index": "time"})
    return out[["time"] + exog_cols], exog_cols


def load_grid_mask(mask_file: Path, template: xr.DataArray, var_name: str = "country_mask") -> xr.DataArray:
    if not mask_file.exists():
        raise FileNotFoundError(
            f"Mask file not found: {mask_file}. "
            "Run scripts/build_region_masks.py first or pass --mask-file."
        )

    mask_ds = xr.open_dataset(mask_file)
    try:
        if var_name not in mask_ds:
            raise KeyError(f"Mask file {mask_file} does not contain variable '{var_name}'")
        mask = mask_ds[var_name]
        if "lat" in mask.coords:
            mask = mask.rename({"lat": "latitude"})
        if "lon" in mask.coords:
            mask = mask.rename({"lon": "longitude"})

        try:
            mask = mask.sel(latitude=template["latitude"], longitude=template["longitude"])
        except Exception as exc:
            raise ValueError(
                f"Mask grid in {mask_file} is not aligned with the region CHIRPS grid."
            ) from exc
        return mask.astype(bool).load()
    finally:
        mask_ds.close()


def build_forecast_dataset(
    region: Region,
    pr_file: Path,
    spi_file: Path,
    dataset_file: Path,
    sample_file: Path,
    climate_features: str,
    force: bool,
    mask_file: Path | None = None,
    mask_var: str = "country_mask",
) -> pd.DataFrame:
    if dataset_file.exists() and not force:
        print(f"Using existing forecast dataset: {dataset_file}")
        df = pd.read_parquet(dataset_file)
        df["time"] = pd.to_datetime(df["time"])
        return df

    print(f"Building forecast table for {region.name}")
    pr_ds = xr.open_dataset(pr_file).load()
    spi_ds = xr.open_dataset(spi_file).load()

    pr = pr_ds["pr"]
    spi1 = spi_ds["spi1"].sel(time=pr.time)
    spi3 = spi_ds["spi3"].sel(time=pr.time)
    spi6 = spi_ds["spi6"].sel(time=pr.time)
    label = spi_ds["drought_label_spi1"].sel(time=pr.time)

    grid_mask = None
    if mask_file is not None:
        grid_mask = load_grid_mask(mask_file, pr, var_name=mask_var)
        n_keep = int(grid_mask.sum())
        n_total = int(grid_mask.size)
        print(f"Applying grid mask '{mask_var}': {mask_file} ({n_keep:,}/{n_total:,} grid cells retained)")
        pr = pr.where(grid_mask)
        spi1 = spi1.where(grid_mask)
        spi3 = spi3.where(grid_mask)
        spi6 = spi6.where(grid_mask)
        label = label.where(grid_mask)

    target = label.shift(time=-1)
    target.name = "target_label"

    ds_stacked = xr.Dataset(
        {
            "spi1_lag1": spi1,
            "spi1_lag2": spi1.shift(time=1),
            "spi1_lag3": spi1.shift(time=2),
            "spi3_lag1": spi3,
            "spi6_lag1": spi6,
            "pr_lag1": pr,
            "pr_lag2": pr.shift(time=1),
            "pr_lag3": pr.shift(time=2),
            "target_label": target,
        }
    ).stack(pixel=("latitude", "longitude"))
    if grid_mask is not None:
        flat_mask = grid_mask.stack(pixel=("latitude", "longitude"))
        ds_stacked = ds_stacked.where(flat_mask, drop=True)

    df = ds_stacked.reset_index("pixel").to_dataframe()
    if "time" not in df.columns:
        df = df.reset_index()
    df["time"] = pd.to_datetime(df["time"]).dt.to_period("M").dt.to_timestamp()

    all_times = pd.DatetimeIndex(sorted(df["time"].unique()))
    cdf, exog_cols = load_climate_features(all_times, climate_features)
    if exog_cols:
        df = df.merge(cdf, on="time", how="left")
        print(f"Added climate features: {exog_cols}")

    feat_cols = [
        "spi1_lag1",
        "spi1_lag2",
        "spi1_lag3",
        "spi3_lag1",
        "spi6_lag1",
        "pr_lag1",
        "pr_lag2",
        "pr_lag3",
    ]
    before = len(df)
    df = df.dropna(subset=["target_label"] + feat_cols + exog_cols).copy()
    dropped = before - len(df)
    if dropped:
        print(f"Dropped rows with missing target/features: {dropped:,}")

    target_time = pd.to_datetime(df["time"]) + pd.DateOffset(months=1)
    df["month"] = target_time.dt.month
    df["year"] = target_time.dt.year
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12.0)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12.0)
    df["region"] = region.slug
    df["target_label"] = df["target_label"].astype(np.int8)

    cols = (
        ["region", "time", "year", "month", "month_sin", "month_cos", "latitude", "longitude"]
        + feat_cols
        + exog_cols
        + ["target_label"]
    )
    df = df[cols]
    dataset_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(dataset_file, index=False)
    df.head(10_000).to_csv(sample_file, index=False)
    print(f"Wrote dataset: {dataset_file} rows={len(df):,} cols={df.shape[1]}")
    print("Class distribution:\n", df["target_label"].value_counts().sort_index())
    return df


def build_spatial_feature_frame(
    pr_file: Path,
    spi_file: Path,
    mask_file: Path | None = None,
    mask_var: str = "country_mask",
) -> pd.DataFrame:
    print("Building 3x3 spatial-neighbourhood features...")
    pr_ds = xr.open_dataset(pr_file).load()
    spi_ds = xr.open_dataset(spi_file).load()
    pr = pr_ds["pr"].astype("float32")
    spi1 = spi_ds["spi1"].sel(time=pr.time).astype("float32")
    spi3 = spi_ds["spi3"].sel(time=pr.time).astype("float32")
    spi6 = spi_ds["spi6"].sel(time=pr.time).astype("float32")

    grid_mask = None
    if mask_file is not None:
        grid_mask = load_grid_mask(mask_file, pr, var_name=mask_var)
        n_keep = int(grid_mask.sum())
        n_total = int(grid_mask.size)
        print(f"Applying grid mask '{mask_var}' before neighbourhood rolling: {n_keep:,}/{n_total:,} grid cells retained")
        pr = pr.where(grid_mask)
        spi1 = spi1.where(grid_mask)
        spi3 = spi3.where(grid_mask)
        spi6 = spi6.where(grid_mask)

    def nbr_mean(da: xr.DataArray, name: str) -> xr.DataArray:
        rolled = da.rolling({"latitude": 3, "longitude": 3}, min_periods=1, center=True).mean()
        rolled.name = name
        return rolled

    nbr_ds = xr.Dataset(
        {
            "spi1_nbr_mean": nbr_mean(spi1, "spi1_nbr_mean"),
            "spi3_nbr_mean": nbr_mean(spi3, "spi3_nbr_mean"),
            "spi6_nbr_mean": nbr_mean(spi6, "spi6_nbr_mean"),
            "pr_nbr_mean": nbr_mean(pr, "pr_nbr_mean"),
        }
    ).stack(pixel=("latitude", "longitude"))
    if grid_mask is not None:
        flat_mask = grid_mask.stack(pixel=("latitude", "longitude"))
        nbr_ds = nbr_ds.where(flat_mask, drop=True)

    nbr_df = nbr_ds.reset_index("pixel").to_dataframe()
    if "time" not in nbr_df.columns:
        nbr_df = nbr_df.reset_index()
    nbr_df["time"] = pd.to_datetime(nbr_df["time"]).dt.to_period("M").dt.to_timestamp()
    print(f"Neighbourhood feature table: {nbr_df.shape}")
    return nbr_df[["time", "latitude", "longitude"] + SPATIAL_FEATURES]


def brier(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean((p - y) ** 2))


def bss(y: np.ndarray, p: np.ndarray, ref: np.ndarray) -> float:
    ref_bs = brier(y, ref)
    return float(1.0 - brier(y, p) / ref_bs) if ref_bs > 0 else float("nan")


def bootstrap_bss(
    monthly: pd.DataFrame,
    pred_col: str,
    ref_col: str,
    n_bootstrap: int,
    seed: int = 42,
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    idx = np.arange(len(monthly))
    vals = []
    y = monthly["y_true_dry_frac"].to_numpy()
    p = monthly[pred_col].to_numpy()
    ref = monthly[ref_col].to_numpy()
    for _ in range(n_bootstrap):
        sample = rng.choice(idx, size=len(idx), replace=True)
        vals.append(bss(y[sample], p[sample], ref[sample]))
    lo, hi = np.nanquantile(vals, [0.025, 0.975])
    return float(lo), float(hi)


def monthly_bs(frame: pd.DataFrame, prob_col: str) -> float:
    monthly = (
        frame.groupby("target_time")
        .agg(y_true_dry_frac=("is_dry", "mean"), pred_prob_dry=(prob_col, "mean"))
        .reset_index()
    )
    return brier(monthly["y_true_dry_frac"].to_numpy(), monthly["pred_prob_dry"].to_numpy())


def target_month(series: pd.Series) -> pd.Series:
    return (pd.to_datetime(series) + pd.DateOffset(months=1)).dt.to_period("M").dt.to_timestamp()


def xgb_params() -> dict[str, object]:
    return {
        "objective": "multi:softprob",
        "num_class": 3,
        "eval_metric": "mlogloss",
        "tree_method": "hist",
        "device": "cpu",
        "eta": 0.05,
        "max_depth": 8,
        "min_child_weight": 5,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "lambda": 1.0,
        "alpha": 0.1,
        "seed": 42,
    }


def evaluate_model(
    region: Region,
    df: pd.DataFrame,
    model_name: str,
    features: list[str],
    out_dir: Path,
    args: Namespace,
) -> dict[str, object]:
    df = df.copy()
    df["year"] = df["year"].astype(int)
    train = df[df["year"] <= TRAIN_END_YEAR].copy()
    val = df[(df["year"] >= VAL_START_YEAR) & (df["year"] <= VAL_END_YEAR)].copy()
    test = df[df["year"] >= TEST_START_YEAR].copy()

    if train.empty or val.empty or test.empty:
        raise ValueError(
            f"Empty split for {region.slug}: train={train.shape}, val={val.shape}, test={test.shape}"
        )

    print(f"[{region.slug}/{model_name}] Train {train.shape}  Val {val.shape}  Test {test.shape}")
    print(f"[{region.slug}/{model_name}] Features: {features}")

    y_train_enc = train[TARGET].map(LABEL_MAP).to_numpy()
    y_val_enc = val[TARGET].map(LABEL_MAP).to_numpy()
    y_test_enc = test[TARGET].map(LABEL_MAP).to_numpy()

    dtrain = xgb.DMatrix(
        train[features],
        label=y_train_enc,
        weight=compute_sample_weight(class_weight="balanced", y=y_train_enc),
        feature_names=features,
    )
    dval = xgb.DMatrix(val[features], label=y_val_enc, feature_names=features)
    dtest = xgb.DMatrix(test[features], label=y_test_enc, feature_names=features)

    print(f"[{region.slug}/{model_name}] Training XGBoost...")
    model = xgb.train(
        params=xgb_params(),
        dtrain=dtrain,
        num_boost_round=args.num_boost_round,
        evals=[(dtrain, "train"), (dval, "val")],
        early_stopping_rounds=args.early_stopping_rounds,
        verbose_eval=args.verbose_eval,
    )
    iteration_range = (0, model.best_iteration + 1)
    probs_val = model.predict(dval, iteration_range=iteration_range).reshape(-1, 3)
    probs_test = model.predict(dtest, iteration_range=iteration_range).reshape(-1, 3)

    dry_idx = LABEL_MAP[-1]
    val_raw = probs_val[:, dry_idx]
    test_raw = probs_test[:, dry_idx]

    train["is_dry"] = (train[TARGET] == -1).astype(float)
    val["is_dry"] = (val[TARGET] == -1).astype(float)
    test["is_dry"] = (test[TARGET] == -1).astype(float)
    train_monthly_dry = train.groupby("month")["is_dry"].mean()
    global_dry = float(train["is_dry"].mean())
    val["clim_prob_dry"] = val["month"].map(train_monthly_dry).fillna(global_dry)
    test["clim_prob_dry"] = test["month"].map(train_monthly_dry).fillna(global_dry)
    val["target_time"] = target_month(val["time"])
    test["target_time"] = target_month(test["time"])

    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(val_raw, val["is_dry"].to_numpy())
    val_iso = iso.predict(val_raw)
    test_iso = iso.predict(test_raw)

    platt = LogisticRegression(solver="lbfgs")
    platt.fit(val_raw.reshape(-1, 1), val["is_dry"].to_numpy())
    val_platt = platt.predict_proba(val_raw.reshape(-1, 1))[:, 1]
    test_platt = platt.predict_proba(test_raw.reshape(-1, 1))[:, 1]

    val["xgb_raw_prob_dry"] = val_raw
    val["xgb_isotonic_prob_dry"] = val_iso
    val["xgb_platt_prob_dry"] = val_platt
    test["xgb_raw_prob_dry"] = test_raw
    test["xgb_isotonic_prob_dry"] = test_iso
    test["xgb_platt_prob_dry"] = test_platt

    calibration_cols = {
        "none": "xgb_raw_prob_dry",
        "isotonic": "xgb_isotonic_prob_dry",
        "platt": "xgb_platt_prob_dry",
    }
    val_bs_by_method = {method: monthly_bs(val, col) for method, col in calibration_cols.items()}
    best_method = min(val_bs_by_method, key=val_bs_by_method.get)
    best_col = calibration_cols[best_method]
    test["xgb_selected_prob_dry"] = test[best_col]

    monthly = (
        test.groupby("target_time")
        .agg(
            y_true_dry_frac=("is_dry", "mean"),
            xgb_raw_prob_dry=("xgb_raw_prob_dry", "mean"),
            xgb_isotonic_prob_dry=("xgb_isotonic_prob_dry", "mean"),
            xgb_platt_prob_dry=("xgb_platt_prob_dry", "mean"),
            xgb_selected_prob_dry=("xgb_selected_prob_dry", "mean"),
            clim_prob_dry=("clim_prob_dry", "mean"),
        )
        .reset_index()
    )

    y_month = monthly["y_true_dry_frac"].to_numpy()
    clim = monthly["clim_prob_dry"].to_numpy()
    bs_clim = brier(y_month, clim)
    metrics = {}
    for method, col in {
        "raw": "xgb_raw_prob_dry",
        "isotonic": "xgb_isotonic_prob_dry",
        "platt": "xgb_platt_prob_dry",
        "selected": "xgb_selected_prob_dry",
    }.items():
        pred = monthly[col].to_numpy()
        metrics[method] = {
            "bs": brier(y_month, pred),
            "bss": bss(y_month, pred, clim),
            "ci": bootstrap_bss(monthly, col, "clim_prob_dry", args.n_bootstrap),
        }

    y_pred = np.array([INV_LABEL_MAP[i] for i in probs_test.argmax(axis=1)])
    report = classification_report(
        test[TARGET].to_numpy(),
        y_pred,
        labels=[-1, 0, 1],
        target_names=["dry(-1)", "normal(0)", "wet(+1)"],
        digits=3,
        zero_division=0,
    )
    cm = confusion_matrix(test[TARGET].to_numpy(), y_pred, labels=[-1, 0, 1])
    try:
        roc_auc = float(roc_auc_score(test["is_dry"].to_numpy(), test_raw))
    except ValueError:
        roc_auc = float("nan")

    out_dir.mkdir(parents=True, exist_ok=True)
    monthly_path = out_dir / f"{model_name}_monthly_scores.csv"
    monthly.to_csv(monthly_path, index=False)

    model_path = out_dir / f"{model_name}_model.json"
    model.save_model(model_path.as_posix())

    probs_path = out_dir / f"{model_name}_test_probs.npz"
    np.savez_compressed(
        probs_path,
        probs=probs_test.astype("float32"),
        dry_probs_raw=test_raw.astype("float32"),
        dry_probs_isotonic=test_iso.astype("float32"),
        dry_probs_platt=test_platt.astype("float32"),
        dry_probs_selected=test[best_col].to_numpy(dtype="float32"),
        y_true=test[TARGET].to_numpy(),
        times=test["time"].to_numpy(),
        target_times=test["target_time"].to_numpy(),
        latitude=test["latitude"].to_numpy(),
        longitude=test["longitude"].to_numpy(),
        features=np.array(features),
        best_iteration=model.best_iteration,
        best_calibration=best_method,
    )

    importance = model.get_score(importance_type="gain")
    imp = pd.Series({f: importance.get(f, 0.0) for f in features}).sort_values(ascending=True)
    fi_path = out_dir / f"{model_name}_feature_importance.png"
    fig, ax = plt.subplots(figsize=(8, max(5, 0.28 * len(features))))
    ax.barh(imp.index, imp.values)
    ax.set_xlabel("Gain")
    ax.set_title(f"{region.name} {model_name} feature importance")
    plt.tight_layout()
    fig.savefig(fi_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    score_lines = [
        f"Multi-Region XGBoost Experiment: {region.name}",
        "=" * 72,
        f"Region slug: {region.slug}",
        f"Region bbox: lat[{region.lat_min}, {region.lat_max}] lon[{region.lon_min}, {region.lon_max}]",
        f"Region rationale: {region.rationale}",
        f"Model: {model_name}",
        "Target: SPI-1[t+1] dry class probability",
        f"Rows: train={len(train):,}, val={len(val):,}, test={len(test):,}",
        f"Test months: {monthly['target_time'].nunique()}",
        f"Best iteration: {model.best_iteration}",
        f"Validation monthly BS by calibration: {val_bs_by_method}",
        f"Selected calibration: {best_method}",
        f"Pixel dry ROC-AUC raw: {roc_auc:.5f}",
        f"Features: {features}",
        "",
        "Monthly dry-fraction Brier Scores",
        f"  Climatology        : {bs_clim:.5f}",
    ]
    for method in ["raw", "isotonic", "platt", "selected"]:
        score_lines.append(f"  XGBoost {method:<9}: {metrics[method]['bs']:.5f}")
    score_lines.extend(["", "Brier Skill Score vs monthly climatology"])
    for method in ["raw", "isotonic", "platt", "selected"]:
        ci = metrics[method]["ci"]
        score_lines.append(
            f"  XGBoost {method:<9}: {metrics[method]['bss']:.5f} "
            f"(95% CI [{ci[0]:.5f}, {ci[1]:.5f}])"
        )
    score_lines.extend(
        [
            "",
            "Pixel-level classification report (secondary diagnostic)",
            report,
            "Pixel-level confusion matrix labels [-1, 0, 1]",
            str(cm),
            "",
            "Outputs:",
            f"  {monthly_path}",
            f"  {model_path}",
            f"  {probs_path}",
            f"  {fi_path}",
        ]
    )
    text = "\n".join(score_lines) + "\n"
    scores_path = out_dir / f"{model_name}_scores.txt"
    scores_path.write_text(text)
    print(text)

    return {
        "region": region.slug,
        "region_name": region.name,
        "model": model_name,
        "train_rows": len(train),
        "val_rows": len(val),
        "test_rows": len(test),
        "test_months": int(monthly["target_time"].nunique()),
        "best_iteration": int(model.best_iteration),
        "selected_calibration": best_method,
        "climatology_bs": bs_clim,
        "raw_bs": metrics["raw"]["bs"],
        "raw_bss": metrics["raw"]["bss"],
        "raw_bss_ci_low": metrics["raw"]["ci"][0],
        "raw_bss_ci_high": metrics["raw"]["ci"][1],
        "selected_bs": metrics["selected"]["bs"],
        "selected_bss": metrics["selected"]["bss"],
        "selected_bss_ci_low": metrics["selected"]["ci"][0],
        "selected_bss_ci_high": metrics["selected"]["ci"][1],
        "pixel_dry_roc_auc_raw": roc_auc,
        "scores_path": str(scores_path),
    }


def update_global_summary(rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    summary_path = OUT_ROOT / "multiregion_summary.csv"
    new = pd.DataFrame(rows)
    if summary_path.exists():
        old = pd.read_csv(summary_path)
        combined = pd.concat([old, new], ignore_index=True)
        combined = combined.drop_duplicates(subset=["region", "model"], keep="last")
    else:
        combined = new
    combined = combined.sort_values(["region", "model"])
    combined.to_csv(summary_path, index=False)
    print(f"Updated global multi-region summary: {summary_path}")


def update_region_summary(out_dir: Path, rows: list[dict[str, object]]) -> Path:
    summary_path = out_dir / "summary.csv"
    new = pd.DataFrame(rows)
    if summary_path.exists():
        old = pd.read_csv(summary_path)
        combined = pd.concat([old, new], ignore_index=True)
        combined = combined.drop_duplicates(subset=["region", "model"], keep="last")
    else:
        combined = new
    combined = combined.sort_values(["region", "model"])
    combined.to_csv(summary_path, index=False)
    print(f"Wrote region summary: {summary_path}")
    return summary_path


def copy_report_artifacts(region: Region, out_dir: Path, rows: list[dict[str, object]]) -> None:
    rdir = report_dir(region)
    rdir.mkdir(parents=True, exist_ok=True)
    for path in out_dir.glob("*_scores.txt"):
        shutil.copy2(path, rdir / path.name)
    for path in out_dir.glob("*_monthly_scores.csv"):
        shutil.copy2(path, rdir / path.name)
    for path in out_dir.glob("*_feature_importance.png"):
        shutil.copy2(path, rdir / path.name)
    summary_path = out_dir / "summary.csv"
    if summary_path.exists():
        shutil.copy2(summary_path, rdir / "summary.csv")
    global_summary = OUT_ROOT / "multiregion_summary.csv"
    if global_summary.exists():
        REPORT_ROOT.mkdir(parents=True, exist_ok=True)
        shutil.copy2(global_summary, REPORT_ROOT / "multiregion_summary.csv")
    print(f"Copied report artifacts to: {rdir}")


def write_region_metadata(region: Region, paths: dict[str, Path], out_dir: Path, args: Namespace) -> None:
    args_dict = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    meta = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "region": {
            "slug": region.slug,
            "name": region.name,
            "lat_min": region.lat_min,
            "lat_max": region.lat_max,
            "lon_min": region.lon_min,
            "lon_max": region.lon_max,
            "rationale": region.rationale,
            "mask_countries": list(region.mask_countries),
            "mask_note": region.mask_note,
        },
        "paths": {key: str(value) for key, value in paths.items()},
        "args": args_dict,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "region_metadata.json").write_text(json.dumps(meta, indent=2, sort_keys=True))


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)
    args = parse_args()
    if args.list_regions:
        print(region_table())
        return

    if args.grid_stride < 1:
        raise ValueError("--grid-stride must be >= 1")
    if args.country_mask and args.basin_mask:
        raise ValueError("Choose only one of --country-mask or --basin-mask")

    base_region = resolve_region(args.region)
    source_region = base_region
    if args.grid_stride > 1:
        source_region = replace(
            base_region,
            slug=f"{base_region.slug}_stride{args.grid_stride}",
            name=f"{base_region.name} (grid-stride {args.grid_stride} smoke)",
            rationale=(
                base_region.rationale
                + f" Smoke-test run using every {args.grid_stride}th CHIRPS grid cell; "
                "do not treat as a full-region scientific result."
            ),
        )
    end_year = args.end_year or latest_chirps_year()
    use_canonical = base_region.slug == "cvalley" and args.grid_stride == 1 and not args.no_canonical_cvalley
    paths = region_paths(source_region, args.start_year, end_year, use_canonical)

    mask_file = None
    mask_var = None
    run_region = source_region
    mask_kind = "country" if args.country_mask else "basin" if args.basin_mask else None
    if mask_kind is not None:
        if args.grid_stride > 1 and args.mask_file is None:
            raise ValueError(
                f"--{mask_kind}-mask with --grid-stride requires an explicit --mask-file on the same grid"
            )
        if mask_kind == "country" and not base_region.mask_countries:
            raise ValueError(f"Region {base_region.slug} has no configured mask countries")
        mask_file = args.mask_file or (
            default_country_mask_file(base_region)
            if mask_kind == "country"
            else default_basin_mask_file(base_region)
        )
        mask_var = args.mask_var or ("country_mask" if mask_kind == "country" else "basin_mask")
        masked_slug = f"{source_region.slug}_{mask_kind}_masked"
        run_region = replace(
            source_region,
            slug=masked_slug,
            name=f"{source_region.name} ({mask_kind}-mask sensitivity)",
            rationale=(
                source_region.rationale
                + f" {mask_kind.title()}-mask sensitivity run using "
                f"{mask_file.name}."
            ),
            mask_countries=base_region.mask_countries,
            mask_note=base_region.mask_note,
        )
        paths.update(masked_dataset_paths(run_region))
        paths["mask"] = mask_file

    out_dir = output_dir(run_region)
    write_region_metadata(run_region, paths, out_dir, args)

    print(f"Running multi-region path for {run_region.name} ({run_region.slug})")
    print(f"Source CHIRPS/SPI region: {source_region.name} ({source_region.slug})")
    print(f"Use canonical cvalley files: {use_canonical}")
    print(f"Climate features: {args.climate_features}")
    if mask_file is not None:
        print(f"{mask_kind.title()} mask file: {mask_file}")
        print(f"Mask variable: {mask_var}")

    if not use_canonical:
        clip_chirps(source_region, paths["pr"], args.start_year, end_year, args.rebuild_pr, args.grid_stride)
        make_spi_labels(paths["pr"], paths["spi"], args.rebuild_spi, args.spi_n_jobs)

    if args.prepare_grid_only:
        print("Prepare-grid-only requested; stopping before forecast dataset build.")
        return

    df = build_forecast_dataset(
        region=run_region,
        pr_file=paths["pr"],
        spi_file=paths["spi"],
        dataset_file=paths["dataset"],
        sample_file=paths["sample"],
        climate_features=args.climate_features,
        force=args.rebuild_dataset and (not use_canonical or mask_kind is not None),
        mask_file=mask_file,
        mask_var=mask_var or "country_mask",
    )

    if args.prepare_only:
        print("Prepare-only requested; stopping before model training.")
        return

    model_rows = []
    model_choices = ["tabular", "spatial"] if args.model == "both" else [args.model]

    for model_name in model_choices:
        model_df = df
        features = get_feature_columns(model_df.columns)
        if model_name == "spatial":
            spatial = build_spatial_feature_frame(
                paths["pr"],
                paths["spi"],
                mask_file=mask_file,
                mask_var=mask_var or "country_mask",
            )
            model_df = model_df.merge(
                spatial,
                on=["time", "latitude", "longitude"],
                how="left",
            )
            missing = int(model_df[SPATIAL_FEATURES].isna().sum().sum())
            if missing:
                print(f"Spatial features missing values: {missing:,}; filling with 0.")
                model_df[SPATIAL_FEATURES] = model_df[SPATIAL_FEATURES].fillna(0.0)
            features = get_feature_columns(model_df.columns) + SPATIAL_FEATURES
        model_rows.append(evaluate_model(run_region, model_df, model_name, features, out_dir, args))

    update_region_summary(out_dir, model_rows)
    update_global_summary(model_rows)
    if args.copy_report:
        copy_report_artifacts(run_region, out_dir, model_rows)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        raise SystemExit(130)
