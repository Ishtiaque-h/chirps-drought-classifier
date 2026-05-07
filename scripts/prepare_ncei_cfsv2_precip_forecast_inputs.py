#!/usr/bin/env python
"""Prepare NCEI CFSv2 precipitation forecast inputs for benchmarking.

This script extracts true lead-window precipitation forecasts from the official
NOAA NCEI THREDDS CFSv2 individual-run archive. It is intentionally separate
from scoring because data access, lead definitions, units, and ensemble
aggregation are archive-specific. The output is shaped for
run_operational_precip_benchmark.py.

Default design:
  - source: NCEI THREDDS NMME CFS-v2 6-hour precipitation individual files
  - region: California Central Valley bounding box from region_config.py
  - initialization: latest available day within the initialization month
  - ensemble proxy: selected 6-hour run cycles from that initialization day
  - target window: months t+1 ... t+lead, matching leakage-safe seasonal SPI-k
    targets when target_spi == lead_months

The NCEI aggregation discovered here begins in 2016, so this is a
forecast-informed external benchmark rather than a full 1991-2016 trained
forecast-feature dataset.
"""
from __future__ import annotations

from argparse import ArgumentParser, Namespace
from pathlib import Path
import re
import shutil

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import requests
import xarray as xr

from region_config import resolve_region


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED = PROJECT_ROOT / "data" / "processed"
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "nmme_ncei_cfsv2"
OUT_DIR = PROJECT_ROOT / "outputs"
REPORT_DIR = PROJECT_ROOT / "results" / "report"

NCEI_THREDDS = "https://www.ncei.noaa.gov/thredds"
CATALOG_XML = (
    f"{NCEI_THREDDS}/catalog/model-nmme_cfs_v2_pr_6h_agg/files/catalog.xml"
)
OPENDAP_BASE = f"{NCEI_THREDDS}/dodsC"
SECONDS_PER_6H = 6 * 60 * 60
FILE_RE = re.compile(
    r"(?P<path>model-nmme_cfs_v2_pr_6h_agg/files/\d{4}/"
    r"pr_6hour_cfsv2-2011\.(?P<init>\d{10})_"
    r"(?P<start>\d{10})-(?P<end>\d{10})\.nc)"
)


def parse_args() -> Namespace:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--region", default="cvalley")
    parser.add_argument("--target-spi", type=int, choices=[1, 3, 6], default=3)
    parser.add_argument(
        "--lead-months",
        type=int,
        default=None,
        help="Defaults to --target-spi; use lead == target SPI for leakage-safe accumulation targets.",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=None,
        help="Defaults to data/processed/dataset_seasonal_spi<target>_lead<lead>.parquet.",
    )
    parser.add_argument(
        "--start-target",
        default=None,
        help="YYYY-MM target month. Defaults to first month covered by the NCEI CFSv2 archive and dataset.",
    )
    parser.add_argument(
        "--end-target",
        default=None,
        help="YYYY-MM target month. Defaults to last month covered by the NCEI CFSv2 archive and dataset.",
    )
    parser.add_argument(
        "--run-hours",
        nargs="+",
        type=int,
        default=[0, 6, 12, 18],
        choices=[0, 6, 12, 18],
        help="CFSv2 initialization hours on the final calendar day of the initialization month.",
    )
    parser.add_argument("--catalog-cache", type=Path, default=RAW_DIR / "cfsv2_pr_6h_catalog.xml")
    parser.add_argument("--out-file", type=Path, default=None)
    parser.add_argument("--copy-report", action="store_true")
    parser.add_argument("--refresh-catalog", action="store_true")
    parser.add_argument("--max-months", type=int, default=None)
    parser.add_argument("--progress-every", type=int, default=6)
    parser.add_argument(
        "--min-coverage-fraction",
        type=float,
        default=0.95,
        help="Minimum valid 6-hour steps per run relative to the target-window expectation.",
    )
    return parser.parse_args()


def month_start(value: str | pd.Timestamp) -> pd.Timestamp:
    return pd.Timestamp(value).to_period("M").to_timestamp()


def default_dataset(target_spi: int, lead_months: int) -> Path:
    if target_spi == 1 and lead_months == 1:
        return PROCESSED / "dataset_forecast.parquet"
    return PROCESSED / f"dataset_seasonal_spi{target_spi}_lead{lead_months}.parquet"


def default_out_file(region: str, target_spi: int, lead_months: int) -> Path:
    return OUT_DIR / f"ncei_cfsv2_{region}_spi{target_spi}_lead{lead_months}_forecast.csv"


def load_target_months(dataset: Path, lead_months: int) -> pd.DatetimeIndex:
    if not dataset.exists():
        raise FileNotFoundError(f"Target dataset not found: {dataset}")
    schema = set(pq.ParquetFile(dataset).schema.names)
    if "target_time" in schema:
        df = pd.read_parquet(dataset, columns=["target_time"])
        target_time = pd.to_datetime(df["target_time"]).dt.to_period("M").dt.to_timestamp()
    elif "time" in schema:
        df = pd.read_parquet(dataset, columns=["time"])
        target_time = (
            pd.to_datetime(df["time"]) + pd.DateOffset(months=lead_months)
        ).dt.to_period("M").dt.to_timestamp()
    else:
        raise ValueError(f"{dataset} must contain target_time or time")
    return pd.DatetimeIndex(sorted(target_time.dropna().unique()))


def fetch_catalog(cache_path: Path, refresh: bool) -> str:
    if cache_path.exists() and not refresh:
        return cache_path.read_text(encoding="utf-8")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    response = requests.get(CATALOG_XML, timeout=90)
    response.raise_for_status()
    cache_path.write_text(response.text, encoding="utf-8")
    return response.text


def build_catalog_index(catalog_xml: str) -> pd.DataFrame:
    rows = []
    for match in FILE_RE.finditer(catalog_xml):
        init = pd.to_datetime(match.group("init"), format="%Y%m%d%H")
        file_id = f"{match.group('init')}_{match.group('start')}_{match.group('end')}"
        rows.append(
            {
                "path": match.group("path"),
                "file_id": file_id,
                "init_time": init,
                "init_month": init.to_period("M").to_timestamp(),
                "init_date": init.normalize(),
                "init_hour": int(init.hour),
                "url": f"{OPENDAP_BASE}/{match.group('path')}",
            }
        )
    if not rows:
        raise RuntimeError("No CFSv2 precipitation files found in the NCEI THREDDS catalog.")
    index = (
        pd.DataFrame(rows)
        .drop_duplicates("file_id", keep="first")
        .sort_values("init_time")
        .reset_index(drop=True)
    )
    return index


def covered_target_range(index: pd.DataFrame, lead_months: int) -> tuple[pd.Timestamp, pd.Timestamp]:
    min_init = index["init_month"].min()
    max_init = index["init_month"].max()
    return (
        month_start(min_init + pd.DateOffset(months=lead_months)),
        month_start(max_init + pd.DateOffset(months=lead_months)),
    )


def select_target_months(
    dataset_months: pd.DatetimeIndex,
    catalog_index: pd.DataFrame,
    lead_months: int,
    start: str | None,
    end: str | None,
    max_months: int | None,
) -> pd.DatetimeIndex:
    archive_start, archive_end = covered_target_range(catalog_index, lead_months)
    start_ts = month_start(start) if start else max(dataset_months.min(), archive_start)
    end_ts = month_start(end) if end else min(dataset_months.max(), archive_end)
    months = dataset_months[(dataset_months >= start_ts) & (dataset_months <= end_ts)]
    if max_months is not None:
        months = months[:max_months]
    if len(months) == 0:
        raise ValueError(
            "No target months after intersecting dataset and CFSv2 archive coverage: "
            f"dataset={dataset_months.min():%Y-%m}..{dataset_months.max():%Y-%m}, "
            f"archive={archive_start:%Y-%m}..{archive_end:%Y-%m}"
        )
    return pd.DatetimeIndex(months)


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
            "CFSv2 regional subset is empty: "
            f"lat[{lat_min}, {lat_max}] lon[{lon_min}, {lon_max}]"
        )
    return sub


def expected_steps(accum_start: pd.Timestamp, accum_end_exclusive: pd.Timestamp) -> int:
    seconds = (accum_end_exclusive - accum_start).total_seconds()
    return int(round(seconds / SECONDS_PER_6H))


def accumulated_precip_mm(
    url: str,
    region_slug: str,
    accum_start: pd.Timestamp,
    accum_end_exclusive: pd.Timestamp,
) -> tuple[float, int]:
    region = resolve_region(region_slug)
    with xr.open_dataset(url, decode_times=True) as ds:
        if "pr" not in ds.data_vars:
            raise ValueError(f"{url} does not contain variable 'pr'; available={list(ds.data_vars)}")
        time_name = "forecast_time"
        da = ds["pr"].sel({time_name: slice(accum_start, accum_end_exclusive - pd.Timedelta(hours=6))})
        if da.sizes.get(time_name, 0) == 0:
            return float("nan"), 0
        sub = subset_region(da, region.lat_min, region.lat_max, region.lon_min, region.lon_max)
        lat_name = "lat" if "lat" in sub.coords else "latitude"
        lon_name = "lon" if "lon" in sub.coords else "longitude"
        weights = np.cos(np.deg2rad(sub[lat_name]))
        regional_rate = sub.weighted(weights).mean(dim=[lat_name, lon_name], skipna=True)
        # CFSv2 pr is kg m-2 s-1, numerically equivalent to mm s-1 for water.
        amount_mm = float((regional_rate * SECONDS_PER_6H).sum(dim=time_name, skipna=True).values)
        return amount_mm, int(regional_rate.sizes[time_name])


def candidate_runs(
    catalog_index: pd.DataFrame,
    init_month: pd.Timestamp,
    run_hours: list[int],
) -> pd.DataFrame:
    init_month = month_start(init_month)
    month_candidates = catalog_index[
        (catalog_index["init_month"] == init_month)
        & (catalog_index["init_hour"].isin(run_hours))
    ].copy()
    if month_candidates.empty:
        return month_candidates
    latest_available_day = month_candidates["init_date"].max()
    return month_candidates[month_candidates["init_date"] == latest_available_day].sort_values("init_time")


def add_forecast_anomaly(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["target_year"] = pd.to_datetime(out["target_time"]).dt.year
    out["target_month"] = pd.to_datetime(out["target_time"]).dt.month

    normal = out[(out["target_year"] >= 2017) & (out["target_year"] <= 2020)]
    if normal["target_month"].nunique() < 12:
        normal = out[out["target_year"] <= 2020]
    if normal.empty:
        out["forecast_pr_anom"] = np.nan
        out["forecast_anom_reference"] = "unavailable"
        return out

    month_norm = normal.groupby("target_month")["forecast_pr"].mean()
    global_norm = float(normal["forecast_pr"].mean())
    out["forecast_pr_normal"] = out["target_month"].map(month_norm).fillna(global_norm)
    out["forecast_pr_anom"] = out["forecast_pr"] - out["forecast_pr_normal"]
    out["forecast_anom_reference"] = (
        f"CFSv2 forecast monthly normal from {int(normal['target_year'].min())}-"
        f"{int(normal['target_year'].max())}"
    )
    return out.drop(columns=["target_year", "target_month"])


def build_rows(args: Namespace) -> pd.DataFrame:
    lead_months = args.lead_months if args.lead_months is not None else args.target_spi
    dataset = args.dataset or default_dataset(args.target_spi, lead_months)
    region = resolve_region(args.region)
    catalog_xml = fetch_catalog(args.catalog_cache, refresh=args.refresh_catalog)
    catalog_index = build_catalog_index(catalog_xml)
    dataset_months = load_target_months(dataset, lead_months)
    months = select_target_months(
        dataset_months,
        catalog_index,
        lead_months,
        start=args.start_target,
        end=args.end_target,
        max_months=args.max_months,
    )

    print(f"Region: {region.slug} ({region.name})", flush=True)
    print(f"Target: SPI-{args.target_spi} dry fraction at lead {lead_months}", flush=True)
    print(f"Dataset: {dataset}", flush=True)
    print(
        "NCEI CFSv2 catalog coverage: "
        f"{catalog_index['init_time'].min():%Y-%m-%d %H} to {catalog_index['init_time'].max():%Y-%m-%d %H}",
        flush=True,
    )
    print(f"Target months: {months.min():%Y-%m} to {months.max():%Y-%m} ({len(months)} months)", flush=True)
    print(f"Run hours: {', '.join(str(h) for h in sorted(args.run_hours))} UTC", flush=True)

    rows = []
    skipped = []
    for i, target_time in enumerate(months, start=1):
        target_time = month_start(target_time)
        init_month = month_start(target_time - pd.DateOffset(months=lead_months))
        accum_start = init_month + pd.DateOffset(months=1)
        accum_end = target_time + pd.DateOffset(months=1)
        expected = expected_steps(accum_start, accum_end)
        runs = candidate_runs(catalog_index, init_month, sorted(args.run_hours))
        if runs.empty:
            skipped.append((target_time, "no_init_month_runs"))
            continue

        member_amounts = []
        member_steps = []
        member_urls = []
        for run in runs.itertuples(index=False):
            amount, n_steps = accumulated_precip_mm(
                run.url,
                region.slug,
                accum_start,
                accum_end,
            )
            coverage = n_steps / expected if expected > 0 else 0.0
            if np.isfinite(amount) and coverage >= args.min_coverage_fraction:
                member_amounts.append(amount)
                member_steps.append(n_steps)
                member_urls.append(run.url)

        if not member_amounts:
            skipped.append((target_time, "insufficient_valid_steps"))
            continue

        forecast_pr = float(np.mean(member_amounts))
        selected_init_time = runs["init_time"].min()
        selected_init_day = runs["init_date"].max()
        rows.append(
            {
                "target_time": target_time,
                "forecast_prob_dry": np.nan,
                "forecast_pr_anom": np.nan,
                "forecast_pr": forecast_pr,
                "source_model": "NCEI_NMME_CFSv2_PR_6h",
                "init_time": selected_init_time,
                "init_day_used": selected_init_day,
                "lead_months": lead_months,
                "target_spi": args.target_spi,
                "region": region.slug,
                "lat_min": region.lat_min,
                "lat_max": region.lat_max,
                "lon_min": region.lon_min,
                "lon_max": region.lon_max,
                "accum_start": accum_start,
                "accum_end_exclusive": accum_end,
                "n_runs": len(member_amounts),
                "requested_run_hours": " ".join(str(h) for h in sorted(args.run_hours)),
                "mean_steps_per_run": float(np.mean(member_steps)),
                "expected_steps_per_run": expected,
                "coverage_fraction": float(np.mean(member_steps) / expected) if expected > 0 else np.nan,
                "forecast_pr_member_std": float(np.std(member_amounts, ddof=0)),
                "source_url": CATALOG_XML,
                "units": "mm accumulated over target SPI window",
                "notes": (
                    "NCEI THREDDS CFSv2 individual-run precipitation; "
                    "member mean over selected latest-available initialization cycles in the feature month."
                ),
                "member_urls": " ".join(member_urls),
            }
        )
        progress_every = max(1, int(args.progress_every))
        if i == 1 or i == len(months) or i % progress_every == 0:
            print(
                f"  {i:>3}/{len(months)} target={target_time:%Y-%m} "
                f"init={selected_init_day:%Y-%m-%d} runs={len(member_amounts)} "
                f"pr={forecast_pr:.2f} mm coverage={np.mean(member_steps) / expected:.2f}",
                flush=True,
            )

    if not rows:
        raise RuntimeError("No NCEI CFSv2 target months were processed.")
    if skipped:
        print(f"Skipped {len(skipped)} target months; first skipped rows: {skipped[:5]}", flush=True)

    out = pd.DataFrame(rows).sort_values("target_time").reset_index(drop=True)
    out = add_forecast_anomaly(out)
    return out


def main() -> None:
    args = parse_args()
    lead_months = args.lead_months if args.lead_months is not None else args.target_spi
    out_file = args.out_file or default_out_file(args.region, args.target_spi, lead_months)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    df = build_rows(args)
    df.to_csv(out_file, index=False)
    print(f"Wrote {len(df)} rows: {out_file}", flush=True)
    print(
        f"Coverage: {df['target_time'].min():%Y-%m} to {df['target_time'].max():%Y-%m}; "
        f"mean runs={df['n_runs'].mean():.2f}; "
        f"mean coverage={df['coverage_fraction'].mean():.3f}",
        flush=True,
    )
    if args.copy_report:
        shutil.copy2(out_file, REPORT_DIR / out_file.name)


if __name__ == "__main__":
    main()
