#!/usr/bin/env python
"""Prepare CPC NMME probabilistic precipitation inputs for benchmarking.

This downloads CPC NMME real-time probabilistic precipitation-tercile NetCDF
files and extracts the regional mean probability of below-normal precipitation
for a fixed target lead. The output is shaped for
run_operational_precip_benchmark.py.

Default design:
  - source: CPC NMME real-time probability NetCDF,
    /NMME/prob/netcdf/prate.YYYYMM.prob.adj.mon.nc
  - predictor: regional mean prob_below, normalized to [0, 1]
  - region: California Central Valley bounding box from region_config.py
  - verification period: 2019 archive start through the latest target month in
    the requested dataset

The forecast is below-normal precipitation probability, not a direct SPI event
probability. The scorer evaluates both the raw probability and a validation-only
isotonic mapping to observed regional dry-fraction probability.
"""
from __future__ import annotations

from argparse import ArgumentParser, Namespace
from pathlib import Path
import shutil
import tempfile

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import requests
import xarray as xr

from region_config import resolve_region


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET = PROJECT_ROOT / "data" / "processed" / "dataset_forecast.parquet"
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "nmme_cpc_prob"
OUT_DIR = PROJECT_ROOT / "outputs"
REPORT_DIR = PROJECT_ROOT / "results" / "report"
DEFAULT_OUT = OUT_DIR / "nmme_cpc_prob_cvalley_lead1_forecast.csv"

CPC_BASE_URL = "https://ftp.cpc.ncep.noaa.gov/NMME/prob/netcdf"
ARCHIVE_START_INIT = pd.Timestamp("2019-01-01")


def parse_args() -> Namespace:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DATASET)
    parser.add_argument("--region", default="cvalley")
    parser.add_argument("--lead-months", type=int, default=1)
    parser.add_argument(
        "--start-target",
        default=None,
        help=(
            "YYYY-MM target month. Defaults to the first target available from "
            "the 2019-01 CPC probability archive for the requested lead."
        ),
    )
    parser.add_argument(
        "--end-target",
        default=None,
        help="YYYY-MM target month. Defaults to the latest target month in --dataset.",
    )
    parser.add_argument(
        "--product-suffix",
        choices=["prob.adj.mon", "prob.mon"],
        default="prob.adj.mon",
        help="CPC probability product suffix. The adjusted monthly product is the default.",
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


def month_start(value: str | pd.Timestamp) -> pd.Timestamp:
    return pd.Timestamp(value).to_period("M").to_timestamp()


def canonical_target_end(dataset: Path, lead_months: int) -> pd.Timestamp:
    if not dataset.exists():
        raise FileNotFoundError(f"Forecast dataset not found: {dataset}")

    schema = set(pq.ParquetFile(dataset).schema.names)
    if "target_time" in schema:
        df = pd.read_parquet(dataset, columns=["target_time"])
        return pd.to_datetime(df["target_time"]).dt.to_period("M").dt.to_timestamp().max()

    if "time" not in schema:
        raise ValueError(f"{dataset} does not contain time or target_time")
    df = pd.read_parquet(dataset, columns=["time"])
    target_time = (
        pd.to_datetime(df["time"]) + pd.DateOffset(months=lead_months)
    ).dt.to_period("M").dt.to_timestamp()
    return pd.Timestamp(target_time.max())


def target_months(
    start: str | None,
    end: str | None,
    dataset: Path,
    lead_months: int,
    max_months: int | None,
) -> pd.DatetimeIndex:
    if lead_months < 0:
        raise ValueError("--lead-months must be non-negative")
    start_ts = (
        month_start(start)
        if start is not None
        else ARCHIVE_START_INIT + pd.DateOffset(months=lead_months)
    )
    end_ts = canonical_target_end(dataset, lead_months) if end is None else month_start(end)
    if end_ts < start_ts:
        raise ValueError(f"end-target {end_ts:%Y-%m} is before start-target {start_ts:%Y-%m}")
    months = pd.date_range(start_ts, end_ts, freq="MS")
    if max_months is not None:
        months = months[:max_months]
    return months


def init_for_target(target_time: pd.Timestamp, lead_months: int) -> pd.Timestamp:
    return month_start(target_time - pd.DateOffset(months=lead_months))


def cpc_file_info(init_time: pd.Timestamp, product_suffix: str) -> tuple[str, str]:
    init_yyyymm = init_time.strftime("%Y%m")
    filename = f"prate.{init_yyyymm}.{product_suffix}.nc"
    url = f"{CPC_BASE_URL}/{filename}"
    return filename, url


def download_if_needed(url: str, dest: Path, force: bool, strict_missing: bool) -> bool:
    if dest.exists() and not force:
        return True

    dest.parent.mkdir(parents=True, exist_ok=True)
    headers = {"User-Agent": "chirps-drought-classifier/1.0"}
    with requests.get(url, headers=headers, stream=True, timeout=90) as response:
        if response.status_code == 404 and not strict_missing:
            print(f"  skipping missing CPC probability NetCDF: {url}")
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
    out = [origin + pd.DateOffset(months=int(round(float(value)))) for value in values]
    return pd.DatetimeIndex(out).to_period("M").to_timestamp()


def subset_region(
    da: xr.DataArray,
    lat_min: float,
    lat_max: float,
    lon_min: float,
    lon_max: float,
) -> xr.DataArray:
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
            "NMME probability regional subset is empty: "
            f"lat[{lat_min}, {lat_max}] lon[{lon_min}, {lon_max}]"
        )
    return sub


def regional_mean_prob_below(path: Path, target_time: pd.Timestamp, region_slug: str) -> float:
    region = resolve_region(region_slug)
    with xr.open_dataset(path, decode_times=False) as ds:
        if "prob_below" not in ds.data_vars:
            raise ValueError(f"{path.name} does not contain prob_below; available={list(ds.data_vars)}")
        if "target" not in ds.coords:
            raise ValueError(f"{path.name} does not contain target coordinate")

        target_index = months_since_1960_to_timestamps(ds["target"].values)
        matches = np.flatnonzero(target_index == month_start(target_time))
        if len(matches) != 1:
            raise ValueError(
                f"{path.name} does not contain target month {target_time:%Y-%m}; "
                f"available {target_index.min():%Y-%m} to {target_index.max():%Y-%m}"
            )

        da = ds["prob_below"].isel(target=int(matches[0]))
        sub = subset_region(da, region.lat_min, region.lat_max, region.lon_min, region.lon_max)
        lat_name = "lat" if "lat" in sub.coords else "latitude"
        lon_name = "lon" if "lon" in sub.coords else "longitude"
        weights = np.cos(np.deg2rad(sub[lat_name]))
        prob_source = float(sub.weighted(weights).mean(dim=[lat_name, lon_name], skipna=True).values)
        source_max = float(ds["prob_below"].max(skipna=True).values)
        scale = 1.0 if source_max <= 1.5 else 0.01
        if not np.isfinite(prob_source):
            raise ValueError(f"{path.name} produced non-finite regional prob_below for {target_time:%Y-%m}")
        return float(np.clip(prob_source * scale, 0.0, 1.0))


def build_rows(args: Namespace) -> pd.DataFrame:
    region = resolve_region(args.region)
    months = target_months(
        args.start_target,
        args.end_target,
        args.dataset,
        args.lead_months,
        args.max_months,
    )
    print(f"Region: {region.slug} ({region.name})")
    print(f"Lead months: {args.lead_months}")
    print(f"CPC product: {args.product_suffix}")
    print(f"Target months: {months.min():%Y-%m} to {months.max():%Y-%m} ({len(months)} months)")
    print(f"Raw cache: {args.raw_dir}")

    rows = []
    for i, target_time in enumerate(months, start=1):
        init_time = init_for_target(target_time, args.lead_months)
        filename, url = cpc_file_info(init_time, args.product_suffix)
        local_path = args.raw_dir / filename
        available = download_if_needed(url, local_path, args.force_download, args.strict_missing)
        if not available:
            continue
        try:
            prob_dry = regional_mean_prob_below(local_path, target_time, region.slug)
        except ValueError as exc:
            if args.strict_missing:
                raise
            print(f"  skipping unusable CPC probability target={target_time:%Y-%m}: {exc}")
            continue
        rows.append(
            {
                "target_time": target_time,
                "forecast_prob_dry": prob_dry,
                "forecast_pr_anom": np.nan,
                "forecast_pr": np.nan,
                "source_model": f"CPC_NMME_{args.product_suffix.upper().replace('.', '_')}",
                "init_time": init_time,
                "lead_months": args.lead_months,
                "region": region.slug,
                "lat_min": region.lat_min,
                "lat_max": region.lat_max,
                "lon_min": region.lon_min,
                "lon_max": region.lon_max,
                "source_url": url,
                "units": "probability; source values normalized from fraction-or-percent range",
                "notes": (
                    "CPC NMME real-time precipitation probability; regional mean "
                    "prob_below used as below-normal/dry predictor."
                ),
            }
        )
        if i == 1 or i == len(months) or i % 12 == 0:
            print(
                f"  {i:>3}/{len(months)} target={target_time:%Y-%m} "
                f"init={init_time:%Y-%m} prob_below={prob_dry:.3f}"
            )

    if not rows:
        raise RuntimeError("No CPC NMME probability NetCDF target months were processed.")
    skipped = len(months) - len(rows)
    if skipped:
        print(f"Processed {len(rows)} months and skipped {skipped} missing NetCDF months.")
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
