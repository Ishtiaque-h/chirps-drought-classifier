#!/usr/bin/env python
"""Prepare CPC NMME precipitation-anomaly inputs for benchmarking.

This script downloads CPC NMME real-time multi-model ensemble-mean
precipitation anomaly files and extracts a regional mean forecast at a fixed
target lead. The output is intentionally shaped for
run_operational_precip_benchmark.py.

Default design:
  - source: CPC NMME real-time anomalies, ENSMEAN/NMME.prate.*.anom.nc
  - region: California Central Valley bounding box from region_config.py
  - lead: initialization month + 1 month predicts the target month
  - verification period: 2017-01 through the latest target month in the
    canonical dataset_forecast.parquet

Default output:
  outputs/nmme_cpc_cvalley_lead1_forecast.csv
"""
from __future__ import annotations

from argparse import ArgumentParser, Namespace
from pathlib import Path
import shutil
import tempfile

import numpy as np
import pandas as pd
import requests
import xarray as xr

from region_config import resolve_region


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET = PROJECT_ROOT / "data" / "processed" / "dataset_forecast.parquet"
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "nmme_cpc"
OUT_DIR = PROJECT_ROOT / "outputs"
REPORT_DIR = PROJECT_ROOT / "results" / "report"
DEFAULT_OUT = OUT_DIR / "nmme_cpc_cvalley_lead1_forecast.csv"

CPC_BASE_URL = "https://ftp.cpc.ncep.noaa.gov/NMME/realtime_anom/ENSMEAN"
MM_PER_DAY = 86400.0


def parse_args() -> Namespace:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DATASET)
    parser.add_argument("--region", default="cvalley")
    parser.add_argument("--lead-months", type=int, default=1)
    parser.add_argument("--start-target", default="2017-01")
    parser.add_argument(
        "--end-target",
        default=None,
        help="YYYY-MM target month. Defaults to the latest target month in --dataset.",
    )
    parser.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    parser.add_argument("--out-file", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--copy-report", action="store_true")
    parser.add_argument("--force-download", action="store_true")
    parser.add_argument(
        "--strict-missing",
        action="store_true",
        help="Raise on missing CPC NetCDF files instead of skipping those target months.",
    )
    parser.add_argument(
        "--max-months",
        type=int,
        default=None,
        help="Debug option: process only the first N target months.",
    )
    return parser.parse_args()


def canonical_target_end(dataset: Path) -> pd.Timestamp:
    if not dataset.exists():
        raise FileNotFoundError(f"Canonical forecast dataset not found: {dataset}")
    df = pd.read_parquet(dataset, columns=["time"])
    target_time = (
        pd.to_datetime(df["time"]) + pd.DateOffset(months=1)
    ).dt.to_period("M").dt.to_timestamp()
    return pd.Timestamp(target_time.max())


def month_start(value: str | pd.Timestamp) -> pd.Timestamp:
    return pd.Timestamp(value).to_period("M").to_timestamp()


def target_months(start: str, end: str | None, dataset: Path, max_months: int | None) -> pd.DatetimeIndex:
    start_ts = month_start(start)
    end_ts = canonical_target_end(dataset) if end is None else month_start(end)
    if end_ts < start_ts:
        raise ValueError(f"end-target {end_ts:%Y-%m} is before start-target {start_ts:%Y-%m}")
    months = pd.date_range(start_ts, end_ts, freq="MS")
    if max_months is not None:
        months = months[:max_months]
    return months


def init_for_target(target_time: pd.Timestamp, lead_months: int) -> pd.Timestamp:
    if lead_months < 0:
        raise ValueError("--lead-months must be non-negative")
    return target_time - pd.DateOffset(months=lead_months)


def cpc_file_info(init_time: pd.Timestamp) -> tuple[str, str]:
    init_yyyymm = init_time.strftime("%Y%m")
    init_dir = f"{init_yyyymm}0800"
    filename = f"NMME.prate.{init_yyyymm}.ENSMEAN.anom.nc"
    url = f"{CPC_BASE_URL}/{init_dir}/{filename}"
    return filename, url


def download_if_needed(url: str, dest: Path, force: bool, strict_missing: bool) -> bool:
    if dest.exists() and not force:
        return True

    dest.parent.mkdir(parents=True, exist_ok=True)
    headers = {"User-Agent": "chirps-drought-classifier/1.0"}
    with requests.get(url, headers=headers, stream=True, timeout=90) as response:
        if response.status_code == 404 and not strict_missing:
            print(f"  skipping missing CPC NetCDF: {url}")
            return False
        response.raise_for_status()
        with tempfile.NamedTemporaryFile(dir=dest.parent, delete=False) as tmp:
            tmp_path = Path(tmp.name)
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    tmp.write(chunk)
    tmp_path.replace(dest)
    return True


def months_since_1960_to_timestamps(values: np.ndarray) -> pd.DatetimeIndex:
    origin = pd.Timestamp("1960-01-01")
    out = []
    for value in values:
        out.append(origin + pd.DateOffset(months=int(round(float(value)))))
    return pd.DatetimeIndex(out).to_period("M").to_timestamp()


def subset_region(da: xr.DataArray, lat_min: float, lat_max: float, lon_min: float, lon_max: float) -> xr.DataArray:
    lat_name = "lat" if "lat" in da.coords else "latitude"
    lon_name = "lon" if "lon" in da.coords else "longitude"

    lat_values = da[lat_name].values
    lat_slice = slice(lat_max, lat_min) if float(lat_values[0]) > float(lat_values[-1]) else slice(lat_min, lat_max)

    lon_values = da[lon_name].values
    if float(np.nanmin(lon_values)) >= 0.0:
        lon_min_use = lon_min % 360.0
        lon_max_use = lon_max % 360.0
    else:
        lon_min_use = lon_min
        lon_max_use = lon_max

    if lon_min_use <= lon_max_use:
        sub = da.sel({lat_name: lat_slice, lon_name: slice(lon_min_use, lon_max_use)})
    else:
        west = da.sel({lat_name: lat_slice, lon_name: slice(lon_min_use, float(np.nanmax(lon_values)))})
        east = da.sel({lat_name: lat_slice, lon_name: slice(float(np.nanmin(lon_values)), lon_max_use)})
        sub = xr.concat([west, east], dim=lon_name)

    if sub.sizes.get(lat_name, 0) == 0 or sub.sizes.get(lon_name, 0) == 0:
        raise ValueError(
            "NMME regional subset is empty: "
            f"lat[{lat_min}, {lat_max}] lon[{lon_min}, {lon_max}]"
        )
    return sub


def choose_forecast_var(ds: xr.Dataset) -> str:
    for candidate in ["fcst", "anom"]:
        if candidate in ds.data_vars:
            return candidate
    if len(ds.data_vars) == 1:
        return list(ds.data_vars)[0]
    raise ValueError(f"Could not identify NMME forecast variable; available={list(ds.data_vars)}")


def to_mm_day(value: float, units: str) -> float:
    units_clean = units.lower().replace(" ", "")
    # CPC NMME precipitation-rate anomaly files are sometimes labelled mm/d
    # even when the stored values are rate-scale values near 1e-6. Convert those
    # small values to mm/day before trusting the unit string.
    if abs(value) < 1e-3:
        return value * MM_PER_DAY
    if units_clean in {"mm/s", "mmsec-1", "mmsecond-1"}:
        return value * MM_PER_DAY
    if units_clean in {"mm/d", "mm/day", "mmday-1", "mmperday"}:
        return value
    return value


def regional_mean_anomaly_mm_day(path: Path, target_time: pd.Timestamp, region_slug: str) -> float:
    region = resolve_region(region_slug)
    with xr.open_dataset(path, decode_times=False) as ds:
        var_name = choose_forecast_var(ds)
        target_index = months_since_1960_to_timestamps(ds["target"].values)
        matches = np.flatnonzero(target_index == month_start(target_time))
        if len(matches) != 1:
            raise ValueError(
                f"{path.name} does not contain target month {target_time:%Y-%m}; "
                f"available {target_index.min():%Y-%m} to {target_index.max():%Y-%m}"
            )

        da = ds[var_name].isel(target=int(matches[0]))
        sub = subset_region(da, region.lat_min, region.lat_max, region.lon_min, region.lon_max)
        lat_name = "lat" if "lat" in sub.coords else "latitude"
        lon_name = "lon" if "lon" in sub.coords else "longitude"
        weights = np.cos(np.deg2rad(sub[lat_name]))
        mean_value = float(sub.weighted(weights).mean(dim=[lat_name, lon_name], skipna=True).values)
        return to_mm_day(mean_value, str(ds[var_name].attrs.get("units", "")))


def build_rows(args: Namespace) -> pd.DataFrame:
    region = resolve_region(args.region)
    months = target_months(args.start_target, args.end_target, args.dataset, args.max_months)
    print(f"Region: {region.slug} ({region.name})")
    print(f"Lead months: {args.lead_months}")
    print(f"Target months: {months.min():%Y-%m} to {months.max():%Y-%m} ({len(months)} months)")
    print(f"Raw cache: {args.raw_dir}")

    rows = []
    for i, target_time in enumerate(months, start=1):
        init_time = init_for_target(target_time, args.lead_months)
        filename, url = cpc_file_info(init_time)
        local_path = args.raw_dir / filename
        available = download_if_needed(url, local_path, args.force_download, args.strict_missing)
        if not available:
            continue
        anomaly = regional_mean_anomaly_mm_day(local_path, target_time, region.slug)
        rows.append(
            {
                "target_time": target_time,
                "forecast_prob_dry": np.nan,
                "forecast_pr_anom": anomaly,
                "forecast_pr": np.nan,
                "source_model": "CPC_NMME_ENSMEAN",
                "init_time": init_time,
                "lead_months": args.lead_months,
                "region": region.slug,
                "lat_min": region.lat_min,
                "lat_max": region.lat_max,
                "lon_min": region.lon_min,
                "lon_max": region.lon_max,
                "source_url": url,
                "units": "mm/day anomaly converted from source mm/s",
                "notes": "CPC NMME real-time multi-model ensemble-mean precipitation anomaly.",
            }
        )
        if i == 1 or i == len(months) or i % 12 == 0:
            print(f"  {i:>3}/{len(months)} target={target_time:%Y-%m} init={init_time:%Y-%m} anom={anomaly:+.4f} mm/day")

    if not rows:
        raise RuntimeError("No NMME CPC NetCDF target months were processed.")
    skipped = len(months) - len(rows)
    if skipped:
        print(
            f"Processed {len(rows)} months and skipped {skipped} missing NetCDF months. "
            "Early CPC archive months may be GRIB-only."
        )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    df = build_rows(args)
    args.out_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out_file, index=False)
    print(f"Wrote: {args.out_file} rows={len(df):,}")
    print(df.tail(6).to_string(index=False))

    if args.copy_report:
        dest = REPORT_DIR / args.out_file.name
        shutil.copy2(args.out_file, dest)
        print(f"Copied report artifact: {dest}")


if __name__ == "__main__":
    main()
