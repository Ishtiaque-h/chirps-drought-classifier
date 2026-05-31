#!/usr/bin/env python
"""Operational GEFS land-surface smoke benchmark.

This script extends the GEFSv12 reforecast land-surface path into the modern
operational GEFS public AWS archive. It is intentionally separated from
`run_gefsv12_landsurface_benchmark.py` because the operational pgrb2b files
expose a different soil-layer subset in the metadata probes used here.

Important caveats:
  - The operational pgrb2b files probed for 2021-2026 include SOILW for
    0.1-0.4 m and 0.4-1 m below ground, but not SOILW for 0-0.1 m.
  - The --soil-mode soilw_0p1_1m default therefore remains a 0.1-1 m
    subsurface approximation, not the same 0-100 cm RZSM approximation used
    for GEFSv12 reforecasts and ERA5-Land targets.
  - The --soil-mode soill_0_1m compatibility test includes the top layer but
    uses SOILL liquid soil moisture rather than SOILW total soil moisture.

The purpose is a modern feasibility/smoke score, not a final deployment claim.
"""

from __future__ import annotations

from argparse import ArgumentParser, Namespace
from dataclasses import dataclass
from pathlib import Path
import re
import shutil
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
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "gefs_operational_land"
OUT_DIR = PROJECT_ROOT / "outputs"
REPORT_DIR = PROJECT_ROOT / "results" / "report" / "landsurface"

AWS_ROOT = "https://noaa-gefs-pds.s3.amazonaws.com"
SOURCE_BUCKET = "s3://noaa-gefs-pds/"
SOURCE_REGISTRY = "https://registry.opendata.aws/noaa-gefs/"
REQUEST_HEADERS = {"User-Agent": "chirps-drought-classifier/operational-gefs-land-benchmark"}
DEFAULT_MEMBERS = ["gec00"] + [f"gep{i:02d}" for i in range(1, 11)]
SOIL_LINE_RE = re.compile(
    r":(?P<variable>SOILW|SOILL):(?P<level>[^:]+ below ground):(?P<step>\d+) hour fcst"
)


@dataclass(frozen=True)
class SoilMode:
    cli_name: str
    grib_variable: str
    data_var: str
    levels: tuple[str, ...]
    thickness_m: tuple[float, ...]
    source_model: str
    signal_label: str
    units: str
    notes: str
    caveat: str


SOIL_MODES = {
    "soilw_0p1_1m": SoilMode(
        cli_name="soilw_0p1_1m",
        grib_variable="SOILW",
        data_var="soilw",
        levels=("0.1-0.4 m below ground", "0.4-1 m below ground"),
        thickness_m=(0.3, 0.6),
        source_model="NOAA_GEFS_operational_pgrb2b_SOILW_0p1_1m",
        signal_label="pgrb2b SOILW 0.1-1 m",
        units="GEFS total volumetric soil water fraction",
        notes=(
            "Operational GEFS pgrb2b SOILW 0.1-0.4 and 0.4-1 m layers, "
            "thickness-weighted and normalized over 0.1-1 m. This is not the "
            "same as the GEFSv12 reforecast 0-100 cm approximation."
        ),
        caveat=(
            "Operational GEFS pgrb2b SOILW 0.1-1 m smoke benchmark; "
            "not directly identical to GEFSv12 reforecast 0-100 cm RZSM."
        ),
    ),
    "soill_0_1m": SoilMode(
        cli_name="soill_0_1m",
        grib_variable="SOILL",
        data_var="soill",
        levels=(
            "0-0.1 m below ground",
            "0.1-0.4 m below ground",
            "0.4-1 m below ground",
        ),
        thickness_m=(0.1, 0.3, 0.6),
        source_model="NOAA_GEFS_operational_pgrb2b_SOILL_0_1m",
        signal_label="pgrb2b SOILL 0-1 m",
        units="GEFS liquid volumetric soil moisture fraction",
        notes=(
            "Operational GEFS pgrb2b SOILL 0-0.1, 0.1-0.4, and 0.4-1 m layers, "
            "thickness-weighted over 0-1 m. This resolves top-layer coverage, "
            "but SOILL is liquid soil moisture and is not identical to SOILW."
        ),
        caveat=(
            "Operational GEFS pgrb2b SOILL 0-1 m compatibility benchmark; "
            "includes the top layer but uses liquid soil moisture rather than SOILW."
        ),
    ),
}


def request_with_retry(
    method: str,
    url: str,
    *,
    timeout: tuple[int, int],
    headers: dict[str, str],
    retries: int = 4,
    **kwargs,
) -> requests.Response:
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
    parser.add_argument("--soil-file", type=Path, default=None)
    parser.add_argument("--lead-months", type=int, default=1)
    parser.add_argument("--start-target", default="2021-01")
    parser.add_argument("--end-target", default="2025-12")
    parser.add_argument("--validation-start-year", type=int, default=2021)
    parser.add_argument("--validation-end-year", type=int, default=2023)
    parser.add_argument("--test-start-year", type=int, default=2024)
    parser.add_argument("--test-end-year", type=int, default=2025)
    parser.add_argument("--normal-start-year", type=int, default=1991)
    parser.add_argument("--normal-end-year", type=int, default=2020)
    parser.add_argument("--dry-quantile", type=float, default=0.20)
    parser.add_argument("--valid-day", type=int, default=15)
    parser.add_argument(
        "--soil-mode",
        choices=sorted(SOIL_MODES),
        default="soilw_0p1_1m",
        help=(
            "Operational GEFS soil-layer extraction mode. "
            "soilw_0p1_1m preserves the original total-water subsurface test; "
            "soill_0_1m tests top-layer-compatible liquid soil moisture."
        ),
    )
    parser.add_argument(
        "--init-days-before-target-start",
        type=int,
        default=7,
        help="Operational 00Z GEFS initialization offset before the target month starts.",
    )
    parser.add_argument("--cycle", default="00")
    parser.add_argument("--members", nargs="+", default=DEFAULT_MEMBERS)
    parser.add_argument("--min-members", type=int, default=6)
    parser.add_argument("--cache-dir", type=Path, default=RAW_DIR)
    parser.add_argument("--out-prefix", default=None)
    parser.add_argument("--n-bootstrap", type=int, default=1000)
    parser.add_argument("--copy-report", action="store_true")
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--max-months", type=int, default=None)
    parser.add_argument("--progress-every", type=int, default=6)
    return parser.parse_args()


def default_out_prefix(
    region_slug: str,
    valid_day: int,
    init_days_before_target_start: int,
    soil_mode: str,
) -> str:
    region = resolve_region(region_slug)
    mode_prefix = "subroot" if soil_mode == "soilw_0p1_1m" else soil_mode
    return (
        f"landsurface_operational_gefs_{mode_prefix}_{region.slug}_"
        f"day{valid_day:02d}_init{init_days_before_target_start:02d}d"
    )


def operational_file_url(init_time: pd.Timestamp, member: str, cycle: str, forecast_hour: int) -> str:
    date = init_time.strftime("%Y%m%d")
    return (
        f"{AWS_ROOT}/gefs.{date}/{cycle}/atmos/pgrb2bp5/"
        f"{member}.t{cycle}z.pgrb2b.0p50.f{forecast_hour:03d}"
    )


def idx_cache_path(cache_dir: Path, url: str) -> Path:
    key = url.replace("https://", "").replace("/", "__")
    return cache_dir / "idx" / f"{key}.idx"


def fetch_idx(url: str, cache_dir: Path, refresh: bool) -> str | None:
    path = idx_cache_path(cache_dir, url)
    if path.exists() and not refresh:
        return path.read_text(encoding="utf-8")
    response = request_with_retry("GET", url + ".idx", timeout=(10, 45), headers=REQUEST_HEADERS)
    if response.status_code == 404:
        return None
    response.raise_for_status()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(response.text, encoding="utf-8")
    return response.text


def parse_idx(idx_text: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for line in idx_text.splitlines():
        parts = line.split(":")
        if len(parts) < 7:
            continue
        try:
            message = int(parts[0])
            offset = int(parts[1])
        except ValueError:
            continue
        rows.append(
            {
                "message": message,
                "offset": offset,
                "variable": parts[3],
                "level": parts[4],
                "line": line,
            }
        )
    return rows


def selected_message_ranges(
    url: str,
    cache_dir: Path,
    refresh: bool,
    soil_mode: SoilMode,
) -> tuple[list[tuple[int, int]], list[str]]:
    idx_text = fetch_idx(url, cache_dir, refresh)
    if idx_text is None:
        return [], []
    rows = parse_idx(idx_text)
    if not rows:
        return [], []

    selected = []
    selected_levels = []
    for row in rows:
        line = str(row["line"])
        match = SOIL_LINE_RE.search(line)
        if not match:
            continue
        variable = match.group("variable").strip()
        level = match.group("level").strip()
        if variable == soil_mode.grib_variable and level in soil_mode.levels:
            selected.append(row)
            selected_levels.append(level)
    if len(selected) < len(soil_mode.levels):
        return [], selected_levels

    offsets_by_message = {int(row["message"]): int(row["offset"]) for row in rows}
    response = request_with_retry("HEAD", url, timeout=(10, 30), headers=REQUEST_HEADERS)
    if response.status_code == 404:
        return [], selected_levels
    response.raise_for_status()
    file_size = int(response.headers["Content-Length"])

    ranges = []
    for row in selected:
        message = int(row["message"])
        start = int(row["offset"])
        end = offsets_by_message.get(message + 1, file_size) - 1
        ranges.append((start, end))
    return ranges, selected_levels


def download_grib_ranges(
    url: str,
    ranges: list[tuple[int, int]],
    cache_dir: Path,
    refresh: bool,
) -> Path:
    digest = url.replace("https://", "").replace("/", "__")
    path = cache_dir / "messages" / f"{digest}__{ranges[0][0]}_{ranges[-1][1]}.grib2"
    if path.exists() and path.stat().st_size > 0 and not refresh:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    chunks = []
    for start, end in ranges:
        response = request_with_retry(
            "GET",
            url,
            headers={**REQUEST_HEADERS, "Range": f"bytes={start}-{end}"},
            timeout=(10, 90),
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


def extract_subroot_mean(path: Path, region_slug: str, soil_mode: SoilMode) -> float:
    region = resolve_region(region_slug)
    with xr.open_dataset(path, engine="cfgrib", backend_kwargs={"indexpath": ""}) as ds:
        if soil_mode.data_var in ds.data_vars:
            data_var = soil_mode.data_var
        elif len(ds.data_vars) == 1:
            data_var = next(iter(ds.data_vars))
        else:
            raise ValueError(
                f"{path} does not contain {soil_mode.data_var}; available={list(ds.data_vars)}"
            )
        da = ds[data_var]
        depth_name = "depthBelowLandLayer"
        if depth_name not in da.dims:
            raise ValueError(f"{path} {data_var} has unexpected dimensions: {da.dims}")
        da = subset_region(da, region.lat_min, region.lat_max, region.lon_min, region.lon_max)
        depths = [float(v) for v in da[depth_name].values]
        order = np.argsort(depths)[: len(soil_mode.levels)]
        selected = da.isel({depth_name: order})
        thickness = np.asarray(soil_mode.thickness_m, dtype=float)
        weights = xr.DataArray(
            thickness / thickness.sum(),
            dims=[depth_name],
            coords={depth_name: selected[depth_name].values},
        )
        subroot = (selected * weights).sum(dim=depth_name)
        lat_name = "latitude" if "latitude" in subroot.coords else "lat"
        lon_name = "longitude" if "longitude" in subroot.coords else "lon"
        area_weights = np.cos(np.deg2rad(subroot[lat_name]))
        regional = subroot.weighted(area_weights).mean(dim=[lat_name, lon_name], skipna=True)
        return float(regional.values)


def forecast_step_hours(init_time: pd.Timestamp, valid_time: pd.Timestamp) -> int:
    hours = (valid_time - init_time).total_seconds() / 3600.0
    return int(round(hours))


def forecast_member_value(
    init_time: pd.Timestamp,
    member: str,
    valid_time: pd.Timestamp,
    args: Namespace,
) -> tuple[float | None, dict[str, object]]:
    soil_mode = SOIL_MODES[args.soil_mode]
    step_hours = forecast_step_hours(init_time, valid_time)
    url = operational_file_url(init_time, member, args.cycle, step_hours)
    ranges, selected_levels = selected_message_ranges(url, args.cache_dir, args.refresh, soil_mode)
    audit = {
        "member": member,
        "url": url,
        "soil_mode": soil_mode.cli_name,
        "grib_variable": soil_mode.grib_variable,
        "selected_step_hours": step_hours,
        "selected_levels": ";".join(selected_levels),
    }
    if not ranges:
        audit["reason"] = f"missing_or_incomplete_{soil_mode.grib_variable.lower()}_layers"
        return None, audit
    local = download_grib_ranges(url, ranges, args.cache_dir, args.refresh)
    value = extract_subroot_mean(local, args.region, soil_mode)
    audit["local_file"] = str(local)
    return value, audit


def target_months_from_observed(args: Namespace, observed: pd.DataFrame) -> pd.DatetimeIndex:
    start = month_start(args.start_target)
    end = month_start(args.end_target)
    months = observed.loc[
        (observed["target_time"] >= start) & (observed["target_time"] <= end),
        "target_time",
    ].drop_duplicates()
    months = pd.DatetimeIndex(months)
    if args.max_months is not None:
        months = months[: args.max_months]
    if len(months) == 0:
        raise ValueError("No target months selected from observed ERA5-Land target.")
    return months


def build_forecast_rows(args: Namespace, target_months: pd.DatetimeIndex) -> tuple[pd.DataFrame, pd.DataFrame]:
    soil_mode = SOIL_MODES[args.soil_mode]
    rows = []
    skipped = []
    progress_every = max(1, int(args.progress_every))

    if int(args.init_days_before_target_start) < 1:
        raise ValueError("--init-days-before-target-start must be >= 1 to avoid target-month leakage.")

    for i, target_time in enumerate(target_months, start=1):
        target_time = month_start(target_time)
        day = min(int(args.valid_day), int(target_time.days_in_month))
        valid_time = pd.Timestamp(target_time.year, target_time.month, day, 12)
        init_time = (target_time - pd.Timedelta(days=int(args.init_days_before_target_start))).replace(
            hour=int(args.cycle),
            minute=0,
            second=0,
            microsecond=0,
        )
        step_hours = forecast_step_hours(init_time, valid_time)
        if step_hours < 0 or step_hours > 840:
            skipped.append(
                {
                    "target_time": target_time,
                    "init_time": init_time,
                    "valid_time": valid_time,
                    "reason": f"unsupported_step_{step_hours}",
                }
            )
            continue

        values = []
        member_audits = []
        for member in args.members:
            try:
                value, audit = forecast_member_value(init_time, member, valid_time, args)
            except Exception as exc:  # noqa: BLE001 - keep month/member audit complete
                value = None
                audit = {"member": member, "reason": f"error:{type(exc).__name__}:{exc}"}
            member_audits.append(audit)
            if value is not None and np.isfinite(value):
                values.append(value)

        if len(values) < int(args.min_members):
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
                "soil_mode": soil_mode.cli_name,
                "source_model": soil_mode.source_model,
                "forecast_subroot_sm": float(np.mean(values)),
                "forecast_subroot_member_std": float(np.std(values, ddof=0)),
                "n_members": int(len(values)),
                "requested_members": " ".join(args.members),
                "source_bucket": SOURCE_BUCKET,
                "source_reference": SOURCE_REGISTRY,
                "units": soil_mode.units,
                "notes": soil_mode.notes,
            }
        )

        if i == 1 or i == len(target_months) or i % progress_every == 0:
            print(
                f"  {i:>3}/{len(target_months)} target={target_time:%Y-%m} "
                f"init={init_time:%Y-%m-%d} valid={valid_time:%Y-%m-%d %H} "
                f"step={step_hours}h members={len(values)} sm={np.mean(values):.4f}",
                flush=True,
            )

    if not rows:
        raise RuntimeError("No operational GEFS land-surface target months were processed.")

    forecast = pd.DataFrame(rows).sort_values("target_time").reset_index(drop=True)
    forecast["target_year"] = pd.to_datetime(forecast["target_time"]).dt.year
    forecast["target_month"] = pd.to_datetime(forecast["target_time"]).dt.month
    normal = forecast[
        (forecast["target_year"] >= args.validation_start_year)
        & (forecast["target_year"] <= args.validation_end_year)
    ].copy()
    if normal.empty:
        forecast["forecast_subroot_normal"] = np.nan
        forecast["forecast_subroot_anom"] = np.nan
    else:
        month_norm = normal.groupby("target_month")["forecast_subroot_sm"].mean()
        global_norm = float(normal["forecast_subroot_sm"].mean())
        forecast["forecast_subroot_normal"] = forecast["target_month"].map(month_norm).fillna(global_norm)
        forecast["forecast_subroot_anom"] = forecast["forecast_subroot_sm"] - forecast["forecast_subroot_normal"]
    forecast = forecast.drop(columns=["target_year", "target_month"])
    return forecast, pd.DataFrame(skipped)


def score_benchmark(
    observed: pd.DataFrame,
    forecast: pd.DataFrame,
    args: Namespace,
) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    soil_mode = SOIL_MODES[args.soil_mode]
    merged = observed.merge(forecast, on="target_time", how="inner", suffixes=("", "_forecast"))
    merged = merged.dropna(
        subset=[
            "forecast_subroot_sm",
            "forecast_subroot_anom",
            "persistence_raw_prob_dry",
            "y_true_dry_frac",
            "clim_prob_dry",
        ]
    ).copy()
    if merged.empty:
        raise ValueError("No overlap between observed target and operational GEFS forecast rows.")

    merged["gefs_raw_dry_signal"] = -merged["forecast_subroot_sm"].astype(float)
    merged["gefs_anom_dry_signal"] = -merged["forecast_subroot_anom"].astype(float)
    val = merged[
        (merged["target_year"] >= args.validation_start_year)
        & (merged["target_year"] <= args.validation_end_year)
    ].copy()
    test = merged[
        (merged["target_year"] >= args.test_start_year)
        & (merged["target_year"] <= args.test_end_year)
    ].copy()
    if val.empty:
        raise ValueError("No validation overlap for operational GEFS land-surface calibration.")
    if test.empty:
        raise ValueError("No test overlap for operational GEFS land-surface evaluation.")

    val_raw, test_raw = fit_isotonic(val, test, "gefs_raw_dry_signal")
    val_anom, test_anom = fit_isotonic(val, test, "gefs_anom_dry_signal")
    val_pers_iso, test_pers_iso = fit_isotonic(val, test, "persistence_raw_prob_dry")

    val_bs = {
        "operational_gefs_raw_isotonic": brier(val["y_true_dry_frac"], val_raw),
        "operational_gefs_anom_isotonic": brier(val["y_true_dry_frac"], val_anom),
        "persistence_raw": brier(val["y_true_dry_frac"], val["persistence_raw_prob_dry"]),
        "persistence_isotonic": brier(val["y_true_dry_frac"], val_pers_iso),
    }
    gefs_best = min(
        {k: v for k, v in val_bs.items() if k.startswith("operational_gefs_")},
        key=val_bs.get,
    )
    pers_best = min({k: v for k, v in val_bs.items() if k.startswith("persistence_")}, key=val_bs.get)

    test = test.copy()
    test["operational_gefs_raw_isotonic_prob_dry"] = test_raw
    test["operational_gefs_anom_isotonic_prob_dry"] = test_anom
    test["persistence_isotonic_prob_dry"] = test_pers_iso
    test["operational_gefs_selected_prob_dry"] = (
        test["operational_gefs_raw_isotonic_prob_dry"]
        if gefs_best == "operational_gefs_raw_isotonic"
        else test["operational_gefs_anom_isotonic_prob_dry"]
    )
    test["persistence_selected_prob_dry"] = (
        test["persistence_raw_prob_dry"]
        if pers_best == "persistence_raw"
        else test["persistence_isotonic_prob_dry"]
    )

    model_cols = {
        "operational_gefs_raw_isotonic": "operational_gefs_raw_isotonic_prob_dry",
        "operational_gefs_anom_isotonic": "operational_gefs_anom_isotonic_prob_dry",
        "operational_gefs_selected": "operational_gefs_selected_prob_dry",
        "persistence_raw": "persistence_raw_prob_dry",
        "persistence_isotonic": "persistence_isotonic_prob_dry",
        "persistence_selected": "persistence_selected_prob_dry",
        "monthly_climatology": "clim_prob_dry",
    }

    rows = []
    for model, col in model_cols.items():
        bs = brier(test["y_true_dry_frac"], test[col])
        bs_ref = brier(test["y_true_dry_frac"], test["clim_prob_dry"])
        bss_value = bss(test["y_true_dry_frac"], test[col], test["clim_prob_dry"])
        if model == "monthly_climatology":
            ci_low = ci_high = 0.0
        else:
            ci_low, ci_high = bootstrap_bss(
                test,
                pred_col=col,
                ref_col="clim_prob_dry",
                n_bootstrap=args.n_bootstrap,
                seed=901,
            )
        delta_bs_vs_persistence = np.nan
        if model.startswith("operational_gefs"):
            delta_bs_vs_persistence = bs - brier(test["y_true_dry_frac"], test["persistence_selected_prob_dry"])
        rows.append(
            {
                "region": args.region,
                "model": model,
                "selected_by_validation": model
                in {gefs_best.replace("operational_gefs_", "operational_gefs_"), pers_best}
                or model in {"operational_gefs_selected", "persistence_selected"},
                "n_validation_months": int(len(val)),
                "n_test_months": int(len(test)),
                "test_start": f"{args.test_start_year}-01",
                "test_end": f"{args.test_end_year}-12",
                "brier_score": bs,
                "brier_score_climatology": bs_ref,
                "bss_vs_climatology": bss_value,
                "bss_ci_low": ci_low,
                "bss_ci_high": ci_high,
                "delta_bs_vs_persistence_selected": delta_bs_vs_persistence,
                "validation_brier_score": val_bs.get(model, np.nan),
                "soil_mode": soil_mode.cli_name,
                "source_model": soil_mode.source_model,
                "source_url": SOURCE_REGISTRY,
                "caveat": soil_mode.caveat,
            }
        )
    summary = pd.DataFrame(rows)
    monthly_cols = [
        "target_time",
        "target_year",
        "target_month",
        "y_true_dry_frac",
        "clim_prob_dry",
        "persistence_raw_prob_dry",
        "persistence_isotonic_prob_dry",
        "persistence_selected_prob_dry",
        "forecast_subroot_sm",
        "forecast_subroot_anom",
        "forecast_subroot_member_std",
        "n_members",
        "lead_days_to_valid",
        "init_time",
        "valid_time",
        "operational_gefs_raw_isotonic_prob_dry",
        "operational_gefs_anom_isotonic_prob_dry",
        "operational_gefs_selected_prob_dry",
    ]
    text = (
        "Operational GEFS land-surface smoke benchmark\n"
        f"Region: {args.region}\n"
        f"Forecast signal: {soil_mode.signal_label}, valid day {args.valid_day}, "
        f"init {args.init_days_before_target_start} days before target month start\n"
        f"Validation: {args.validation_start_year}-{args.validation_end_year}; "
        f"Test: {args.test_start_year}-{args.test_end_year}\n"
        f"Selected operational GEFS mapping: {gefs_best}\n"
        f"Selected persistence mapping: {pers_best}\n"
        f"Source caveat: {soil_mode.caveat}\n"
        "Caveat: this is a modern feasibility/smoke test, not a final deployment claim.\n"
    )
    return summary, test[[c for c in monthly_cols if c in test.columns]].copy(), text


def main() -> None:
    args = parse_args()
    region = resolve_region(args.region)
    args.region = region.slug
    if args.soil_mode not in SOIL_MODES:
        raise ValueError(f"Unknown soil mode: {args.soil_mode}")
    if args.soil_file is None:
        args.soil_file = default_soil_file(region.slug)
    if args.out_prefix is None:
        args.out_prefix = default_out_prefix(
            region.slug,
            args.valid_day,
            args.init_days_before_target_start,
            args.soil_mode,
        )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    args.cache_dir.mkdir(parents=True, exist_ok=True)

    observed_args = Namespace(
        soil_file=args.soil_file,
        lead_months=args.lead_months,
        normal_start_year=args.normal_start_year,
        normal_end_year=args.normal_end_year,
        dry_quantile=args.dry_quantile,
    )
    observed = observed_rootzone_target(observed_args)
    target_months = target_months_from_observed(args, observed)
    forecast, skipped = build_forecast_rows(args, target_months)
    summary, monthly, text = score_benchmark(observed, forecast, args)

    forecast_path = OUT_DIR / f"{args.out_prefix}_forecast.csv"
    skipped_path = OUT_DIR / f"{args.out_prefix}_skipped_months.csv"
    summary_path = OUT_DIR / f"{args.out_prefix}_summary.csv"
    monthly_path = OUT_DIR / f"{args.out_prefix}_monthly_scores.csv"
    scores_path = OUT_DIR / f"{args.out_prefix}_scores.txt"

    forecast.to_csv(forecast_path, index=False)
    skipped.to_csv(skipped_path, index=False)
    summary.to_csv(summary_path, index=False)
    monthly.to_csv(monthly_path, index=False)
    scores_path.write_text(text + "\n" + summary.to_string(index=False) + "\n", encoding="utf-8")

    if args.copy_report:
        for path in [forecast_path, skipped_path, summary_path, monthly_path, scores_path]:
            shutil.copy2(path, REPORT_DIR / path.name)

    print(text)
    print(summary.to_string(index=False))
    print("")
    print(f"Wrote {forecast_path.relative_to(PROJECT_ROOT)}")
    print(f"Wrote {summary_path.relative_to(PROJECT_ROOT)}")
    print(f"Wrote {monthly_path.relative_to(PROJECT_ROOT)}")
    print(f"Wrote {scores_path.relative_to(PROJECT_ROOT)}")
    if len(skipped):
        print(f"Skipped months: {len(skipped)} ({skipped_path.relative_to(PROJECT_ROOT)})")


if __name__ == "__main__":
    main()
