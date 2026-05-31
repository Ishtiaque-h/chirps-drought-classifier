#!/usr/bin/env python
"""Forecast-informed land-surface benchmark using GEFSv12 reforecast soil moisture.

This benchmark is the public-reforecast counterpart to the existing CFSv2
monthly-mean land-surface check. It verifies NOAA GEFSv12 retrospective
forecast soil moisture against an ERA5-Land 0-100 cm root-zone soil-moisture
dry-fraction target.

Default design:
  - source: NOAA GEFSv12 reforecast on the public AWS bucket
  - forecast variable: soilw_bgrnd, first three soil layers approximating 0-1 m
  - forecast signal: ensemble-mean root-zone soil moisture valid near the
    middle of each target month
  - target: ERA5-Land monthly root-zone soil-moisture dry fraction
  - split: 1991-2013 observed climatology/threshold period, 2014-2016
    validation-only isotonic calibration, 2017-2019 frozen test period

GEFSv12 retrospective forecasts span 2000-2019, so this uses a separate
hindcast-era split rather than the project's 2021+ canonical test window.
"""
from __future__ import annotations

from argparse import ArgumentParser, Namespace
from pathlib import Path
import re
import shutil
import tempfile
import time

import numpy as np
import pandas as pd
import requests
import xarray as xr

from region_config import resolve_region
from run_landsurface_forecast_benchmark import (
    brier,
    bss,
    bootstrap_bss,
    default_soil_file,
    fit_isotonic,
    month_start,
    observed_rootzone_target,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "gefsv12_reforecast_land"
OUT_DIR = PROJECT_ROOT / "outputs"
REPORT_DIR = PROJECT_ROOT / "results" / "report"

AWS_ROOT = "https://noaa-gefs-retrospective.s3.amazonaws.com"
SOURCE_BUCKET = "s3://noaa-gefs-retrospective/GEFSv12/reforecast"
REQUEST_HEADERS = {"User-Agent": "chirps-drought-classifier/gefsv12-land-benchmark"}
IDX_STEP_RE = re.compile(r":(?P<step>\d+) hour fcst:")
DEFAULT_MEMBERS = ["c00"] + [f"p{i:02d}" for i in range(1, 11)]


def request_with_retry(
    method: str,
    url: str,
    *,
    timeout: tuple[int, int],
    headers: dict[str, str],
    retries: int = 4,
    **kwargs,
) -> requests.Response:
    """Small retry wrapper for long public-archive runs."""
    last_error: requests.RequestException | None = None
    for attempt in range(retries):
        try:
            return requests.request(method, url, timeout=timeout, headers=headers, **kwargs)
        except requests.RequestException as exc:
            last_error = exc
            if attempt == retries - 1:
                break
            time.sleep(1.5 * (attempt + 1))
    assert last_error is not None
    raise last_error


def parse_args() -> Namespace:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--region", default="cvalley")
    parser.add_argument(
        "--soil-file",
        type=Path,
        default=None,
        help="Defaults to data/processed/era5_land_soil_moisture_monthly_<region>_1991_2026.nc.",
    )
    parser.add_argument("--lead-months", type=int, default=1)
    parser.add_argument("--start-target", default="2014-01")
    parser.add_argument("--end-target", default="2019-12")
    parser.add_argument("--validation-start-year", type=int, default=2014)
    parser.add_argument("--validation-end-year", type=int, default=2016)
    parser.add_argument("--test-start-year", type=int, default=2017)
    parser.add_argument("--test-end-year", type=int, default=2019)
    parser.add_argument("--normal-start-year", type=int, default=1991)
    parser.add_argument("--normal-end-year", type=int, default=2013)
    parser.add_argument("--dry-quantile", type=float, default=0.20)
    parser.add_argument(
        "--valid-day",
        type=int,
        default=15,
        help="Day of target month used for GEFSv12 RZSM state verification.",
    )
    parser.add_argument(
        "--init-lag-weeks",
        type=int,
        default=0,
        help=(
            "Use an earlier weekly long GEFSv12 initialization relative to the latest long init "
            "before the target month starts (0=latest, 1=1 week earlier, etc). "
            "This is the primary lead-sensitivity control."
        ),
    )
    parser.add_argument(
        "--members",
        nargs="+",
        default=DEFAULT_MEMBERS,
        help="GEFSv12 members to use, e.g. c00 p01 ... p10.",
    )
    parser.add_argument("--min-members", type=int, default=6)
    parser.add_argument("--cache-dir", type=Path, default=RAW_DIR)
    parser.add_argument("--out-prefix", default=None)
    parser.add_argument("--n-bootstrap", type=int, default=2000)
    parser.add_argument("--copy-report", action="store_true")
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--max-months", type=int, default=None)
    parser.add_argument("--progress-every", type=int, default=6)
    return parser.parse_args()


def default_out_prefix(region_slug: str, valid_day: int, init_lag_weeks: int) -> str:
    region = resolve_region(region_slug)
    lag = f"_lag{init_lag_weeks}w" if int(init_lag_weeks) else ""
    if region.slug == "cvalley":
        return f"landsurface_gefsv12_rzsm_day{valid_day:02d}{lag}"
    return f"landsurface_gefsv12_rzsm_{region.slug}_day{valid_day:02d}{lag}"


def s3_url(key: str) -> str:
    return f"{AWS_ROOT}/{key}"


def key_exists(key: str) -> bool:
    response = request_with_retry("HEAD", s3_url(key), timeout=(10, 30), headers=REQUEST_HEADERS)
    return response.status_code == 200


def soilw_key(init_time: pd.Timestamp, member: str, day_group: str) -> str:
    init = init_time.strftime("%Y%m%d%H")
    return (
        f"GEFSv12/reforecast/{init_time.year:04d}/{init}/{member}/"
        f"{day_group}/soilw_bgrnd_{init}_{member}.grib2"
    )


def member_day_group_exists(init_time: pd.Timestamp, member: str, day_group: str) -> bool:
    return key_exists(soilw_key(init_time, member, day_group) + ".idx")


def latest_long_init_before_target(target_month: pd.Timestamp, members: list[str]) -> pd.Timestamp | None:
    target_start = month_start(target_month)
    # Weekly long GEFSv12 reforecasts are the useful source for target-month
    # midpoints. Search backward from the day before the target month starts.
    for delta in range(1, 36):
        candidate = target_start - pd.Timedelta(days=delta)
        init_time = pd.Timestamp(candidate.year, candidate.month, candidate.day, 0)
        if member_day_group_exists(init_time, members[0], "Days:10-35"):
            return init_time
    return None


def forecast_step_hours(init_time: pd.Timestamp, valid_time: pd.Timestamp) -> int:
    hours = (valid_time - init_time).total_seconds() / 3600.0
    return int(round(hours / 3.0) * 3)


def day_group_for_step(step_hours: int) -> str:
    if step_hours <= 240:
        return "Days:1-10"
    return "Days:10-35"


def idx_cache_path(cache_dir: Path, key: str) -> Path:
    return cache_dir / "idx" / (key.replace("/", "__") + ".idx")


def fetch_text_cached(url: str, path: Path, refresh: bool) -> str | None:
    if path.exists() and not refresh:
        return path.read_text(encoding="utf-8")
    response = request_with_retry("GET", url, timeout=(10, 45), headers=REQUEST_HEADERS)
    if response.status_code == 404:
        return None
    response.raise_for_status()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(response.text, encoding="utf-8")
    return response.text


def parse_idx(idx_text: str) -> list[dict[str, object]]:
    rows = []
    for line in idx_text.splitlines():
        parts = line.split(":")
        if len(parts) < 7:
            continue
        match = IDX_STEP_RE.search(line)
        if not match:
            continue
        rows.append(
            {
                "message": int(parts[0]),
                "offset": int(parts[1]),
                "level": parts[4],
                "step_hours": int(match.group("step")),
                "line": line,
            }
        )
    return rows


def selected_message_ranges(
    key: str,
    target_step: int,
    cache_dir: Path,
    refresh: bool,
) -> tuple[list[tuple[int, int]], int, int]:
    idx_text = fetch_text_cached(s3_url(key + ".idx"), idx_cache_path(cache_dir, key), refresh)
    if idx_text is None:
        return [], target_step, 0
    rows = parse_idx(idx_text)
    if not rows:
        return [], target_step, 0
    available_steps = sorted({int(row["step_hours"]) for row in rows})
    selected_step = min(available_steps, key=lambda step: abs(step - target_step))
    selected = [row for row in rows if int(row["step_hours"]) == selected_step]
    if len(selected) < 3:
        return [], selected_step, len(selected)

    offsets_by_message = {int(row["message"]): int(row["offset"]) for row in rows}
    response = request_with_retry("HEAD", s3_url(key), timeout=(10, 30), headers=REQUEST_HEADERS)
    if response.status_code == 404:
        return [], selected_step, 0
    response.raise_for_status()
    file_size = int(response.headers["Content-Length"])

    ranges = []
    for row in selected:
        message = int(row["message"])
        start = int(row["offset"])
        end = offsets_by_message.get(message + 1, file_size) - 1
        ranges.append((start, end))
    return ranges, selected_step, len(selected)


def download_grib_ranges(key: str, ranges: list[tuple[int, int]], cache_dir: Path, refresh: bool) -> Path:
    digest = f"{key.replace('/', '__')}__{ranges[0][0]}_{ranges[-1][1]}.grib2"
    path = cache_dir / "messages" / digest
    if path.exists() and path.stat().st_size > 0 and not refresh:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    chunks = []
    for start, end in ranges:
        response = request_with_retry(
            "GET",
            s3_url(key),
            headers={**REQUEST_HEADERS, "Range": f"bytes={start}-{end}"},
            timeout=(10, 60),
        )
        response.raise_for_status()
        chunks.append(response.content)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes(b"".join(chunks))
    tmp.replace(path)
    return path


def subset_region(
    da: xr.DataArray,
    lat_min: float,
    lat_max: float,
    lon_min: float,
    lon_max: float,
) -> xr.DataArray:
    lat_name = "latitude" if "latitude" in da.coords else "lat"
    lon_name = "longitude" if "longitude" in da.coords else "lon"
    lat_values = da[lat_name].values
    lat_slice = slice(lat_max, lat_min) if float(lat_values[0]) > float(lat_values[-1]) else slice(lat_min, lat_max)
    lon_values = da[lon_name].values
    lon_min_use = lon_min % 360.0 if float(np.nanmin(lon_values)) >= 0.0 else lon_min
    lon_max_use = lon_max % 360.0 if float(np.nanmin(lon_values)) >= 0.0 else lon_max
    if lon_min_use <= lon_max_use:
        return da.sel({lat_name: lat_slice, lon_name: slice(lon_min_use, lon_max_use)})
    west = da.sel({lat_name: lat_slice, lon_name: slice(lon_min_use, float(np.nanmax(lon_values)))})
    east = da.sel({lat_name: lat_slice, lon_name: slice(float(np.nanmin(lon_values)), lon_max_use)})
    return xr.concat([west, east], dim=lon_name)


def extract_rootzone_mean(path: Path, region_slug: str) -> float:
    region = resolve_region(region_slug)
    with xr.open_dataset(path, engine="cfgrib", backend_kwargs={"indexpath": ""}) as ds:
        if "soilw" not in ds.data_vars:
            raise ValueError(f"{path} does not contain soilw; available={list(ds.data_vars)}")
        da = ds["soilw"]
        depth_name = "depthBelowLandLayer"
        if depth_name not in da.dims:
            raise ValueError(f"{path} soilw has unexpected dimensions: {da.dims}")
        da = subset_region(da, region.lat_min, region.lat_max, region.lon_min, region.lon_max)
        depths = [float(v) for v in da[depth_name].values]
        order = np.argsort(depths)[:3]
        selected = da.isel({depth_name: order})
        weights = xr.DataArray(
            np.array([0.1, 0.3, 0.6], dtype=float),
            dims=[depth_name],
            coords={depth_name: selected[depth_name].values},
        )
        rootzone = (selected * weights).sum(dim=depth_name)
        lat_name = "latitude" if "latitude" in rootzone.coords else "lat"
        lon_name = "longitude" if "longitude" in rootzone.coords else "lon"
        area_weights = np.cos(np.deg2rad(rootzone[lat_name]))
        regional = rootzone.weighted(area_weights).mean(dim=[lat_name, lon_name], skipna=True)
        return float(regional.values)


def forecast_member_value(
    init_time: pd.Timestamp,
    member: str,
    valid_time: pd.Timestamp,
    args: Namespace,
) -> tuple[float | None, dict[str, object]]:
    target_step = forecast_step_hours(init_time, valid_time)
    day_group = day_group_for_step(target_step)
    key = soilw_key(init_time, member, day_group)
    ranges, selected_step, n_messages = selected_message_ranges(
        key,
        target_step,
        cache_dir=args.cache_dir,
        refresh=args.refresh,
    )
    audit = {
        "member": member,
        "key": key,
        "target_step_hours": target_step,
        "selected_step_hours": selected_step,
        "n_layer_messages": n_messages,
    }
    if not ranges:
        audit["reason"] = "missing_or_incomplete_step"
        return None, audit
    local = download_grib_ranges(key, ranges, args.cache_dir, refresh=args.refresh)
    value = extract_rootzone_mean(local, args.region)
    audit["local_file"] = str(local)
    return value, audit


def target_months_from_observed(args: Namespace, observed: pd.DataFrame) -> pd.DatetimeIndex:
    start = month_start(args.start_target)
    end = month_start(args.end_target)
    months = observed.loc[
        (observed["target_time"] >= start)
        & (observed["target_time"] <= end),
        "target_time",
    ].drop_duplicates()
    months = pd.DatetimeIndex(months)
    if args.max_months is not None:
        months = months[: args.max_months]
    if len(months) == 0:
        raise ValueError("No target months selected from observed ERA5-Land target.")
    return months


def build_forecast_rows(args: Namespace, target_months: pd.DatetimeIndex) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    skipped = []
    progress_every = max(1, int(args.progress_every))

    for i, target_time in enumerate(target_months, start=1):
        target_time = month_start(target_time)
        day = min(args.valid_day, int(target_time.days_in_month))
        valid_time = pd.Timestamp(target_time.year, target_time.month, day, 12)
        init_time = latest_long_init_before_target(target_time, args.members)
        if init_time is None:
            skipped.append({"target_time": target_time, "reason": "no_long_init_found"})
            continue
        init_lag_weeks = int(getattr(args, "init_lag_weeks", 0) or 0)
        if init_lag_weeks:
            candidate = init_time - pd.Timedelta(days=7 * init_lag_weeks)
            if member_day_group_exists(candidate, args.members[0], "Days:10-35"):
                init_time = candidate
            else:
                skipped.append(
                    {
                        "target_time": target_time,
                        "base_init_time": init_time,
                        "candidate_init_time": candidate,
                        "reason": f"no_long_init_found_lag{init_lag_weeks}w",
                    }
                )
                continue

        values = []
        member_audits = []
        for member in args.members:
            try:
                value, audit = forecast_member_value(init_time, member, valid_time, args)
            except Exception as exc:
                value = None
                audit = {"member": member, "reason": f"error:{type(exc).__name__}:{exc}"}
            member_audits.append(audit)
            if value is not None and np.isfinite(value):
                values.append(value)

        if len(values) < args.min_members:
            skipped.append(
                {
                    "target_time": target_time,
                    "init_time": init_time,
                    "valid_time": valid_time,
                    "reason": f"insufficient_members_{len(values)}_of_{args.min_members}",
                    "member_audit": repr(member_audits),
                }
            )
            continue

        rows.append(
            {
                "target_time": target_time,
                "valid_time": valid_time,
                "init_time": init_time,
                "lead_days_to_valid": (valid_time - init_time).total_seconds() / 86400.0,
                "lead_months": args.lead_months,
                "region": args.region,
                "source_model": "NOAA_GEFSv12_reforecast_soilw_bgrnd",
                "forecast_rzsm": float(np.mean(values)),
                "forecast_rzsm_member_std": float(np.std(values, ddof=0)),
                "n_members": int(len(values)),
                "requested_members": " ".join(args.members),
                "source_bucket": SOURCE_BUCKET,
                "source_reference": "Guan et al. (2022), doi:10.1175/MWR-D-21-0245.1",
                "units": "GEFSv12 soil wetness fraction",
                "notes": (
                    "GEFSv12 soilw_bgrnd first-meter approximation from the first three "
                    "below-ground layers, weighted as 0-10, 10-40, and 40-100 cm; "
                    "forecast state valid near target-month midpoint."
                ),
            }
        )
        if i == 1 or i == len(target_months) or i % progress_every == 0:
            print(
                f"  {i:>3}/{len(target_months)} target={target_time:%Y-%m} "
                f"init={init_time:%Y-%m-%d} valid={valid_time:%Y-%m-%d %H} "
                f"members={len(values)} rzsm={np.mean(values):.4f}",
                flush=True,
            )

    if not rows:
        raise RuntimeError("No GEFSv12 land-surface target months were processed.")
    forecast = pd.DataFrame(rows).sort_values("target_time").reset_index(drop=True)
    forecast["target_year"] = pd.to_datetime(forecast["target_time"]).dt.year
    forecast["target_month"] = pd.to_datetime(forecast["target_time"]).dt.month
    normal = forecast[
        (forecast["target_year"] >= args.validation_start_year)
        & (forecast["target_year"] <= args.validation_end_year)
    ].copy()
    if normal.empty:
        forecast["forecast_rzsm_normal"] = np.nan
        forecast["forecast_rzsm_anom"] = np.nan
    else:
        month_norm = normal.groupby("target_month")["forecast_rzsm"].mean()
        global_norm = float(normal["forecast_rzsm"].mean())
        forecast["forecast_rzsm_normal"] = forecast["target_month"].map(month_norm).fillna(global_norm)
        forecast["forecast_rzsm_anom"] = forecast["forecast_rzsm"] - forecast["forecast_rzsm_normal"]
    forecast = forecast.drop(columns=["target_year", "target_month"])
    return forecast, pd.DataFrame(skipped)


def score_benchmark(
    observed: pd.DataFrame,
    forecast: pd.DataFrame,
    args: Namespace,
) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    merged = observed.merge(forecast, on="target_time", how="inner", suffixes=("", "_forecast"))
    merged = merged.dropna(
        subset=[
            "forecast_rzsm",
            "forecast_rzsm_anom",
            "persistence_raw_prob_dry",
            "y_true_dry_frac",
            "clim_prob_dry",
        ]
    ).copy()
    if merged.empty:
        raise ValueError("No overlap between observed target and GEFSv12 forecast rows.")

    merged["gefs_raw_dry_signal"] = -merged["forecast_rzsm"].astype(float)
    merged["gefs_anom_dry_signal"] = -merged["forecast_rzsm_anom"].astype(float)
    val = merged[
        (merged["target_year"] >= args.validation_start_year)
        & (merged["target_year"] <= args.validation_end_year)
    ].copy()
    test = merged[
        (merged["target_year"] >= args.test_start_year)
        & (merged["target_year"] <= args.test_end_year)
    ].copy()
    if val.empty:
        raise ValueError("No validation overlap for GEFSv12 land-surface calibration.")
    if test.empty:
        raise ValueError("No test overlap for GEFSv12 land-surface evaluation.")

    val_raw, test_raw = fit_isotonic(val, test, "gefs_raw_dry_signal")
    val_anom, test_anom = fit_isotonic(val, test, "gefs_anom_dry_signal")
    val_pers_iso, test_pers_iso = fit_isotonic(val, test, "persistence_raw_prob_dry")

    val_bs = {
        "gefs_raw_isotonic": brier(val["y_true_dry_frac"], val_raw),
        "gefs_anom_isotonic": brier(val["y_true_dry_frac"], val_anom),
        "persistence_raw": brier(val["y_true_dry_frac"], val["persistence_raw_prob_dry"]),
        "persistence_isotonic": brier(val["y_true_dry_frac"], val_pers_iso),
    }
    gefs_best = min({k: v for k, v in val_bs.items() if k.startswith("gefs_")}, key=val_bs.get)
    persistence_best = min({k: v for k, v in val_bs.items() if k.startswith("persistence_")}, key=val_bs.get)

    test = test.copy()
    test["gefs_raw_isotonic_prob_dry"] = test_raw
    test["gefs_anom_isotonic_prob_dry"] = test_anom
    test["gefs_selected_prob_dry"] = (
        test["gefs_raw_isotonic_prob_dry"]
        if gefs_best == "gefs_raw_isotonic"
        else test["gefs_anom_isotonic_prob_dry"]
    )
    test["persistence_isotonic_prob_dry"] = test_pers_iso
    test["persistence_selected_prob_dry"] = (
        test["persistence_raw_prob_dry"]
        if persistence_best == "persistence_raw"
        else test["persistence_isotonic_prob_dry"]
    )

    score_cols = [
        "gefs_raw_isotonic_prob_dry",
        "gefs_anom_isotonic_prob_dry",
        "gefs_selected_prob_dry",
        "persistence_raw_prob_dry",
        "persistence_isotonic_prob_dry",
        "persistence_selected_prob_dry",
    ]
    y = test["y_true_dry_frac"].to_numpy(dtype=float)
    ref = test["clim_prob_dry"].to_numpy(dtype=float)
    persistence_raw_ref = test["persistence_raw_prob_dry"].to_numpy(dtype=float)
    persistence_selected_ref = test["persistence_selected_prob_dry"].to_numpy(dtype=float)
    score_rows = []
    for i, col in enumerate(score_cols):
        pred = test[col].to_numpy(dtype=float)
        lo, hi = bootstrap_bss(test, col, n_bootstrap=args.n_bootstrap, seed=901 + i)
        score_rows.append(
            {
                "forecast": col.replace("_prob_dry", ""),
                "bs": brier(y, pred),
                "bss": bss(y, pred, ref),
                "bss_ci_low": lo,
                "bss_ci_high": hi,
                "bss_vs_persistence_raw": bss(y, pred, persistence_raw_ref),
                "bss_vs_persistence_selected": bss(y, pred, persistence_selected_ref),
            }
        )
    scores = pd.DataFrame(score_rows)

    gefs_corr = test["gefs_selected_prob_dry"].corr(test["y_true_dry_frac"], method="spearman")
    gefs_raw_corr = test["gefs_raw_dry_signal"].corr(test["y_true_dry_frac"], method="spearman")
    gefs_anom_corr = test["gefs_anom_dry_signal"].corr(test["y_true_dry_frac"], method="spearman")
    pers_corr = test["persistence_selected_prob_dry"].corr(test["y_true_dry_frac"], method="spearman")
    y_std = float(test["y_true_dry_frac"].std(ddof=0))
    gefs_amp = float(test["gefs_selected_prob_dry"].std(ddof=0) / y_std) if y_std > 0 else np.nan
    pers_amp = float(test["persistence_selected_prob_dry"].std(ddof=0) / y_std) if y_std > 0 else np.nan

    lines = [
        "GEFSv12 Forecast-Informed Land-Surface Benchmark",
        "=" * 72,
        "Design: GEFSv12 reforecast soil moisture mapped to ERA5-Land root-zone dry-fraction probability.",
        f"Region: {resolve_region(args.region).name} ({args.region})",
        f"Forecast source: NOAA GEFSv12 reforecast AWS bucket, soilw_bgrnd, members {' '.join(args.members)}.",
        f"Observed target: ERA5-Land root-zone soil moisture dry fraction, q={args.dry_quantile:.2f}, thresholds from {args.normal_start_year}-{args.normal_end_year}.",
        (
            f"Forecast valid day: target-month day {args.valid_day} at 12 UTC; "
            "latest prior long GEFSv12 initialization."
            if int(getattr(args, "init_lag_weeks", 0) or 0) == 0
            else (
                f"Forecast valid day: target-month day {args.valid_day} at 12 UTC; "
                f"weekly long GEFSv12 initialization lag={int(getattr(args, 'init_lag_weeks', 0) or 0)}w."
            )
        ),
        f"Validation months: {val['target_time'].nunique()} ({val['target_time'].min():%Y-%m} to {val['target_time'].max():%Y-%m})",
        f"Test months: {test['target_time'].nunique()} ({test['target_time'].min():%Y-%m} to {test['target_time'].max():%Y-%m})",
        f"Validation BS by method: {val_bs}",
        f"Selected GEFSv12 calibration: {gefs_best}",
        f"Selected persistence calibration: {persistence_best}",
        f"Climatology BS: {brier(y, ref):.5f}",
        f"Spearman corr(GEFSv12 selected prob, observed dry fraction): {gefs_corr:.3f}",
        f"Spearman corr(GEFSv12 raw dry signal, observed dry fraction): {gefs_raw_corr:.3f}",
        f"Spearman corr(GEFSv12 anomaly dry signal, observed dry fraction): {gefs_anom_corr:.3f}",
        f"Spearman corr(persistence selected prob, observed dry fraction): {pers_corr:.3f}",
        f"GEFSv12 selected probability amplitude ratio: {gefs_amp:.3f}",
        f"Persistence selected probability amplitude ratio: {pers_amp:.3f}",
        "",
        "Monthly dry-fraction Brier Skill Score vs climatology:",
    ]
    for row in score_rows:
        lines.append(
            f"  {row['forecast']:<34} BS={row['bs']:.5f} "
            f"BSS={row['bss']:.5f} "
            f"95% CI [{row['bss_ci_low']:.5f}, {row['bss_ci_high']:.5f}]"
        )
    selected = next(row for row in score_rows if row["forecast"] == "gefs_selected")
    lines.extend(
        [
            "",
            "GEFSv12 selected skill against same-target persistence references:",
            f"  vs persistence_raw      : {selected['bss_vs_persistence_raw']:.5f}",
            f"  vs persistence_selected : {selected['bss_vs_persistence_selected']:.5f}",
        ]
    )
    return test, scores, "\n".join(lines) + "\n"


def write_outputs(
    forecast: pd.DataFrame,
    skipped: pd.DataFrame,
    monthly: pd.DataFrame,
    scores: pd.DataFrame,
    score_text: str,
    args: Namespace,
) -> None:
    OUT_DIR.mkdir(exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    outputs = {
        "forecast": OUT_DIR / f"{args.out_prefix}_forecast.csv",
        "skipped": OUT_DIR / f"{args.out_prefix}_skipped_months.csv",
        "monthly": OUT_DIR / f"{args.out_prefix}_monthly_scores.csv",
        "summary": OUT_DIR / f"{args.out_prefix}_summary.csv",
        "scores": OUT_DIR / f"{args.out_prefix}_scores.txt",
    }
    forecast.to_csv(outputs["forecast"], index=False)
    skipped.to_csv(outputs["skipped"], index=False)
    monthly["target_variable"] = "ERA5-Land root-zone soil moisture dry fraction"
    monthly["source_model"] = "NOAA_GEFSv12_reforecast_soilw_bgrnd"
    monthly["benchmark_lead_months"] = args.lead_months
    monthly["region"] = args.region
    monthly.to_csv(outputs["monthly"], index=False)
    scores.to_csv(outputs["summary"], index=False)
    outputs["scores"].write_text(score_text + f"\nMonthly scores: {outputs['monthly']}\n", encoding="utf-8")

    print(score_text)
    print(f"Wrote forecast rows: {outputs['forecast']} rows={len(forecast):,}")
    print(f"Wrote skipped audit: {outputs['skipped']} rows={len(skipped):,}")
    print(f"Wrote monthly scores: {outputs['monthly']} rows={len(monthly):,}")
    print(f"Wrote score summary: {outputs['summary']}")

    if args.copy_report:
        for path in outputs.values():
            shutil.copy2(path, REPORT_DIR / path.name)


def main() -> None:
    args = parse_args()
    args.region = resolve_region(args.region).slug
    if args.soil_file is None:
        args.soil_file = default_soil_file(args.region)
    if args.out_prefix is None:
        args.out_prefix = default_out_prefix(args.region, args.valid_day, int(getattr(args, "init_lag_weeks", 0) or 0))
    args.cache_dir.mkdir(parents=True, exist_ok=True)

    observed = observed_rootzone_target(args)
    months = target_months_from_observed(args, observed)
    print(
        f"Selected target months: {months.min():%Y-%m} to {months.max():%Y-%m} "
        f"({len(months)} months)",
        flush=True,
    )
    forecast, skipped = build_forecast_rows(args, months)
    monthly, scores, score_text = score_benchmark(observed, forecast, args)
    write_outputs(forecast, skipped, monthly, scores, score_text, args)


if __name__ == "__main__":
    main()
