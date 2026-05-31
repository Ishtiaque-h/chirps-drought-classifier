#!/usr/bin/env python
"""Native C3S/ECMWF seasonal VSM land-surface benchmark.

This compact benchmark follows the same target and probability-scoring design
as the existing CFSv2 and GEFSv12 land-surface checks, but uses a cleaner
native root-zone soil-moisture source: C3S seasonal-original single-levels,
ECMWF system 51, variable ``volumetric_soil_moisture``.

Default design:
  - forecast source: C3S seasonal-original single-levels, ECMWF system 51
  - forecast variable: native volumetric soil moisture (NetCDF variable ``vsw``)
  - forecast state: ensemble-mean soil moisture valid near target-month day 15
  - root-zone approximation: soil layers 1-3 weighted as 0-7, 7-28, and
    28-100 cm, matching the ERA5-Land root-zone target approximation
  - target: ERA5-Land 0-100 cm root-zone soil-moisture dry fraction
  - calibration: validation-only isotonic mapping from forecast dry signal to
    observed monthly dry fraction
  - score: frozen test-period monthly Brier Skill Score against climatology

The script intentionally starts as a compact single-region benchmark. It should
be scaled only after the first retrieval/scoring pass is scientifically useful.
"""

from __future__ import annotations

from argparse import ArgumentParser, Namespace
from pathlib import Path
import shutil
import time

import numpy as np
import pandas as pd
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
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "c3s_ecmwf_s51_vsm"
OUT_DIR = PROJECT_ROOT / "outputs"
REPORT_DIR = PROJECT_ROOT / "results" / "report"
PAPER_DIR = REPORT_DIR / "paper"

C3S_DATASET = "seasonal-original-single-levels"
C3S_DATASET_URL = "https://cds.climate.copernicus.eu/datasets/seasonal-original-single-levels"
C3S_DOI = "10.24381/cds.181d637e"


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
    parser.add_argument("--start-target", default="2019-01")
    parser.add_argument("--end-target", default="2022-12")
    parser.add_argument("--validation-start-year", type=int, default=2019)
    parser.add_argument("--validation-end-year", type=int, default=2020)
    parser.add_argument("--test-start-year", type=int, default=2021)
    parser.add_argument("--test-end-year", type=int, default=2022)
    parser.add_argument("--normal-start-year", type=int, default=1991)
    parser.add_argument("--normal-end-year", type=int, default=2016)
    parser.add_argument("--dry-quantile", type=float, default=0.20)
    parser.add_argument(
        "--valid-day",
        type=int,
        default=15,
        help="Day of target month used for C3S VSM state verification.",
    )
    parser.add_argument(
        "--originating-centre",
        default="ecmwf",
        help="C3S originating centre. Default uses the strongest audited candidate.",
    )
    parser.add_argument("--system", default="51", help="C3S seasonal forecast system.")
    parser.add_argument("--cache-dir", type=Path, default=RAW_DIR)
    parser.add_argument("--out-prefix", default=None)
    parser.add_argument("--n-bootstrap", type=int, default=2000)
    parser.add_argument("--copy-report", action="store_true")
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--max-months", type=int, default=None)
    parser.add_argument("--progress-every", type=int, default=6)
    parser.add_argument(
        "--request-sleep",
        type=float,
        default=0.5,
        help="Pause between CDS requests when downloads are needed.",
    )
    parser.add_argument(
        "--download-retries",
        type=int,
        default=3,
        help="Number of attempts for each uncached CDS request.",
    )
    parser.add_argument(
        "--retry-sleep",
        type=float,
        default=30.0,
        help="Base pause, in seconds, before retrying a failed CDS request.",
    )
    parser.add_argument(
        "--fail-on-skipped",
        action="store_true",
        help="Raise an error if any selected target month could not be processed.",
    )
    return parser.parse_args()


def default_out_prefix(args: Namespace) -> str:
    region = resolve_region(args.region)
    return (
        f"landsurface_c3s_ecmwf_s{args.system}_vsm_{region.slug}"
        f"_day{int(args.valid_day):02d}_{args.validation_start_year}_{args.test_end_year}"
    )


def c3s_target_info(target_time: pd.Timestamp, valid_day: int) -> tuple[pd.Timestamp, pd.Timestamp, int]:
    target_time = month_start(target_time)
    init_time = month_start(target_time - pd.DateOffset(months=1))
    valid_day = min(int(valid_day), int(target_time.days_in_month))
    valid_time = pd.Timestamp(target_time.year, target_time.month, valid_day)
    lead_hours = int((valid_time - init_time).total_seconds() // 3600)
    if lead_hours <= 0 or lead_hours % 24 != 0:
        raise ValueError(
            f"C3S seasonal-original VSM lead must be a positive 24-hour step; got {lead_hours} h"
        )
    return init_time, valid_time, lead_hours


def c3s_cache_path(args: Namespace, target_time: pd.Timestamp, lead_hours: int) -> Path:
    region = resolve_region(args.region)
    init_time = month_start(target_time - pd.DateOffset(months=1))
    name = (
        f"c3s_{args.originating_centre}_s{args.system}_vsm_{region.slug}_"
        f"init{init_time:%Y%m%d}_target{month_start(target_time):%Y%m}_l{lead_hours:04d}.nc"
    )
    return args.cache_dir / name


def c3s_area(region_slug: str) -> list[float]:
    region = resolve_region(region_slug)
    # CDS uses [north, west, south, east].
    return [float(region.lat_max), float(region.lon_min), float(region.lat_min), float(region.lon_max)]


def retrieve_c3s_file(
    args: Namespace,
    init_time: pd.Timestamp,
    lead_hours: int,
    target_path: Path,
) -> None:
    if target_path.exists() and target_path.stat().st_size > 0 and not args.refresh:
        return
    import cdsapi  # type: ignore

    request = {
        "originating_centre": args.originating_centre,
        "system": str(args.system),
        "variable": ["volumetric_soil_moisture"],
        "year": [f"{init_time.year:04d}"],
        "month": [f"{init_time.month:02d}"],
        "day": ["01"],
        "leadtime_hour": [str(int(lead_hours))],
        "area": c3s_area(args.region),
        "data_format": "netcdf",
    }
    target_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = target_path.with_suffix(target_path.suffix + ".tmp")
    client = cdsapi.Client(quiet=True, progress=False)
    max_attempts = max(1, int(args.download_retries))
    last_exc: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        if tmp.exists():
            tmp.unlink()
        try:
            client.retrieve(C3S_DATASET, request, str(tmp))
            tmp.replace(target_path)
            last_exc = None
            break
        except Exception as exc:  # noqa: BLE001 - CDS failures include mixed client/server exceptions
            last_exc = exc
            if tmp.exists():
                tmp.unlink()
            if attempt >= max_attempts:
                break
            sleep_seconds = float(args.retry_sleep) * attempt
            print(
                f"  CDS request failed for init={init_time:%Y-%m-%d} "
                f"lead={lead_hours}h attempt={attempt}/{max_attempts}: "
                f"{type(exc).__name__}: {exc}. Retrying in {sleep_seconds:.1f}s.",
                flush=True,
            )
            time.sleep(sleep_seconds)
    if last_exc is not None:
        raise last_exc
    if args.request_sleep:
        time.sleep(float(args.request_sleep))


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
        sub = da.sel({lat_name: lat_slice, lon_name: slice(lon_min_use, lon_max_use)})
    else:
        west = da.sel({lat_name: lat_slice, lon_name: slice(lon_min_use, float(np.nanmax(lon_values)))})
        east = da.sel({lat_name: lat_slice, lon_name: slice(float(np.nanmin(lon_values)), lon_max_use)})
        sub = xr.concat([west, east], dim=lon_name)
    if sub.sizes.get(lat_name, 0) == 0 or sub.sizes.get(lon_name, 0) == 0:
        raise ValueError(
            "C3S regional subset is empty: "
            f"lat[{lat_min}, {lat_max}] lon[{lon_min}, {lon_max}]"
        )
    return sub


def extract_rootzone_members(path: Path, region_slug: str) -> tuple[float, float, int, str]:
    region = resolve_region(region_slug)
    with xr.open_dataset(path) as ds:
        var_name = "vsw" if "vsw" in ds.data_vars else next(iter(ds.data_vars))
        da = ds[var_name]
        if "soilLayer" not in da.dims:
            raise ValueError(f"{path} has no soilLayer dimension; dims={da.dims}")
        da = subset_region(da, region.lat_min, region.lat_max, region.lon_min, region.lon_max)
        da = da.isel(soilLayer=[0, 1, 2])
        weights = xr.DataArray(
            np.array([0.07, 0.21, 0.72], dtype=float),
            dims=["soilLayer"],
            coords={"soilLayer": da["soilLayer"].values},
        )
        rootzone = (da * weights).sum(dim="soilLayer")
        for dim in ["forecast_reference_time", "forecast_period"]:
            if dim in rootzone.dims:
                rootzone = rootzone.squeeze(dim, drop=True)
        lat_name = "latitude" if "latitude" in rootzone.coords else "lat"
        lon_name = "longitude" if "longitude" in rootzone.coords else "lon"
        area_weights = np.cos(np.deg2rad(rootzone[lat_name]))
        regional = rootzone.weighted(area_weights).mean(dim=[lat_name, lon_name], skipna=True)
        if "number" in regional.dims:
            member_values = np.asarray(regional.values, dtype=float)
            member_values = member_values[np.isfinite(member_values)]
            n_members = int(member_values.size)
            if n_members == 0:
                raise ValueError(f"{path} has no finite member values after regional averaging.")
            return (
                float(np.mean(member_values)),
                float(np.std(member_values, ddof=0)),
                n_members,
                var_name,
            )
        value = float(regional.values)
        return value, 0.0, 1, var_name


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
    rows: list[dict[str, object]] = []
    skipped: list[dict[str, object]] = []
    progress_every = max(1, int(args.progress_every))

    for i, target_time in enumerate(target_months, start=1):
        target_time = month_start(target_time)
        try:
            init_time, valid_time, lead_hours = c3s_target_info(target_time, args.valid_day)
            path = c3s_cache_path(args, target_time, lead_hours)
            retrieve_c3s_file(args, init_time, lead_hours, path)
            value, member_std, n_members, var_name = extract_rootzone_members(path, args.region)
        except Exception as exc:  # noqa: BLE001 - keep full skipped-month audit
            skipped.append(
                {
                    "target_time": target_time,
                    "reason": f"{type(exc).__name__}: {exc}",
                }
            )
            continue

        rows.append(
            {
                "target_time": target_time,
                "valid_time": valid_time,
                "init_time": init_time,
                "lead_hours": int(lead_hours),
                "lead_days_to_valid": lead_hours / 24.0,
                "lead_months": args.lead_months,
                "region": args.region,
                "source_model": f"C3S_{args.originating_centre}_seasonal_original_system_{args.system}_vsm",
                "forecast_rzsm": value,
                "forecast_rzsm_member_std": member_std,
                "n_members": n_members,
                "netcdf_variable": var_name,
                "source_dataset": C3S_DATASET,
                "source_url": C3S_DATASET_URL,
                "source_reference": f"C3S seasonal original single levels, doi:{C3S_DOI}",
                "local_file": str(path.relative_to(PROJECT_ROOT)),
                "units": "volumetric soil moisture",
                "notes": (
                    "ECMWF system 51 native VSM, root-zone approximation from soil layers 1-3 "
                    "weighted as 0-7, 7-28, and 28-100 cm; forecast state valid near target-month midpoint."
                ),
            }
        )
        if i == 1 or i == len(target_months) or i % progress_every == 0:
            print(
                f"  {i:>3}/{len(target_months)} target={target_time:%Y-%m} "
                f"init={init_time:%Y-%m-%d} valid={valid_time:%Y-%m-%d} "
                f"lead={lead_hours:>4}h members={n_members} rzsm={value:.4f}",
                flush=True,
            )

    if not rows:
        raise RuntimeError("No C3S/ECMWF VSM target months were processed.")

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
        raise ValueError("No overlap between observed target and C3S/ECMWF forecast rows.")

    merged["c3s_raw_dry_signal"] = -merged["forecast_rzsm"].astype(float)
    merged["c3s_anom_dry_signal"] = -merged["forecast_rzsm_anom"].astype(float)
    val = merged[
        (merged["target_year"] >= args.validation_start_year)
        & (merged["target_year"] <= args.validation_end_year)
    ].copy()
    test = merged[
        (merged["target_year"] >= args.test_start_year)
        & (merged["target_year"] <= args.test_end_year)
    ].copy()
    if val.empty:
        raise ValueError("No validation overlap for C3S/ECMWF VSM calibration.")
    if test.empty:
        raise ValueError("No test overlap for C3S/ECMWF VSM evaluation.")

    val_raw, test_raw = fit_isotonic(val, test, "c3s_raw_dry_signal")
    val_anom, test_anom = fit_isotonic(val, test, "c3s_anom_dry_signal")
    val_pers_iso, test_pers_iso = fit_isotonic(val, test, "persistence_raw_prob_dry")

    val_bs = {
        "c3s_raw_isotonic": brier(val["y_true_dry_frac"], val_raw),
        "c3s_anom_isotonic": brier(val["y_true_dry_frac"], val_anom),
        "persistence_raw": brier(val["y_true_dry_frac"], val["persistence_raw_prob_dry"]),
        "persistence_isotonic": brier(val["y_true_dry_frac"], val_pers_iso),
    }
    c3s_best = min({k: v for k, v in val_bs.items() if k.startswith("c3s_")}, key=val_bs.get)
    persistence_best = min({k: v for k, v in val_bs.items() if k.startswith("persistence_")}, key=val_bs.get)

    test = test.copy()
    test["c3s_raw_isotonic_prob_dry"] = test_raw
    test["c3s_anom_isotonic_prob_dry"] = test_anom
    test["c3s_selected_prob_dry"] = (
        test["c3s_raw_isotonic_prob_dry"]
        if c3s_best == "c3s_raw_isotonic"
        else test["c3s_anom_isotonic_prob_dry"]
    )
    test["persistence_isotonic_prob_dry"] = test_pers_iso
    test["persistence_selected_prob_dry"] = (
        test["persistence_raw_prob_dry"]
        if persistence_best == "persistence_raw"
        else test["persistence_isotonic_prob_dry"]
    )

    score_cols = [
        "c3s_raw_isotonic_prob_dry",
        "c3s_anom_isotonic_prob_dry",
        "c3s_selected_prob_dry",
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
        lo, hi = bootstrap_bss(test, col, n_bootstrap=args.n_bootstrap, seed=1201 + i)
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

    c3s_corr = test["c3s_selected_prob_dry"].corr(test["y_true_dry_frac"], method="spearman")
    c3s_raw_corr = test["c3s_raw_dry_signal"].corr(test["y_true_dry_frac"], method="spearman")
    c3s_anom_corr = test["c3s_anom_dry_signal"].corr(test["y_true_dry_frac"], method="spearman")
    pers_corr = test["persistence_selected_prob_dry"].corr(test["y_true_dry_frac"], method="spearman")
    y_std = float(test["y_true_dry_frac"].std(ddof=0))
    c3s_amp = float(test["c3s_selected_prob_dry"].std(ddof=0) / y_std) if y_std > 0 else np.nan
    pers_amp = float(test["persistence_selected_prob_dry"].std(ddof=0) / y_std) if y_std > 0 else np.nan

    lines = [
        "C3S/ECMWF Native-VSM Land-Surface Benchmark",
        "=" * 72,
        "Design: C3S ECMWF system 51 native VSM mapped to ERA5-Land root-zone dry-fraction probability.",
        f"Region: {resolve_region(args.region).name} ({args.region})",
        f"Forecast source: C3S seasonal-original single levels, {args.originating_centre} system {args.system}.",
        f"Observed target: ERA5-Land root-zone soil moisture dry fraction, q={args.dry_quantile:.2f}, thresholds from {args.normal_start_year}-{args.normal_end_year}.",
        f"Forecast valid day: target-month day {args.valid_day}; initialization: day 1 of previous month.",
        f"Validation months: {val['target_time'].nunique()} ({val['target_time'].min():%Y-%m} to {val['target_time'].max():%Y-%m})",
        f"Test months: {test['target_time'].nunique()} ({test['target_time'].min():%Y-%m} to {test['target_time'].max():%Y-%m})",
        f"Validation BS by method: {val_bs}",
        f"Selected C3S calibration: {c3s_best}",
        f"Selected persistence calibration: {persistence_best}",
        f"Climatology BS: {brier(y, ref):.5f}",
        f"Spearman corr(C3S selected prob, observed dry fraction): {c3s_corr:.3f}",
        f"Spearman corr(C3S raw dry signal, observed dry fraction): {c3s_raw_corr:.3f}",
        f"Spearman corr(C3S anomaly dry signal, observed dry fraction): {c3s_anom_corr:.3f}",
        f"Spearman corr(persistence selected prob, observed dry fraction): {pers_corr:.3f}",
        f"C3S selected probability amplitude ratio: {c3s_amp:.3f}",
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
    selected = next(row for row in score_rows if row["forecast"] == "c3s_selected")
    lines.extend(
        [
            "",
            "C3S selected skill against same-target persistence references:",
            f"  vs persistence_raw      : {selected['bss_vs_persistence_raw']:.5f}",
            f"  vs persistence_selected : {selected['bss_vs_persistence_selected']:.5f}",
        ]
    )
    return test, scores, "\n".join(lines) + "\n"


def yearly_score_summary(monthly: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for year, group in monthly.groupby(pd.to_datetime(monthly["target_time"]).dt.year):
        y = group["y_true_dry_frac"].to_numpy(dtype=float)
        ref = group["clim_prob_dry"].to_numpy(dtype=float)
        c3s = group["c3s_selected_prob_dry"].to_numpy(dtype=float)
        persistence = group["persistence_selected_prob_dry"].to_numpy(dtype=float)
        rows.append(
            {
                "year": int(year),
                "n_months": int(len(group)),
                "obs_dry_frac_mean": float(np.mean(y)),
                "c3s_prob_mean": float(np.mean(c3s)),
                "persistence_prob_mean": float(np.mean(persistence)),
                "c3s_bss_vs_climatology": bss(y, c3s, ref),
                "persistence_bss_vs_climatology": bss(y, persistence, ref),
                "c3s_bss_vs_persistence_selected": bss(y, c3s, persistence),
                "spearman_c3s_prob_obs": group["c3s_selected_prob_dry"].corr(
                    group["y_true_dry_frac"], method="spearman"
                ),
                "spearman_persistence_prob_obs": group["persistence_selected_prob_dry"].corr(
                    group["y_true_dry_frac"], method="spearman"
                ),
            }
        )
    return pd.DataFrame(rows)


def season_label(month: int) -> str:
    if month in (12, 1, 2):
        return "DJF"
    if month in (3, 4, 5):
        return "MAM"
    if month in (6, 7, 8):
        return "JJA"
    return "SON"


def bootstrap_mean_ci(values: pd.Series, seed: int, n_bootstrap: int = 5000) -> tuple[float, float]:
    arr = values.to_numpy(dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size < 2:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    means = np.empty(int(n_bootstrap), dtype=float)
    for i in range(int(n_bootstrap)):
        means[i] = float(np.mean(rng.choice(arr, size=arr.size, replace=True)))
    lo, hi = np.quantile(means, [0.025, 0.975])
    return float(lo), float(hi)


def persistence_regime_diagnostic(monthly: pd.DataFrame) -> pd.DataFrame:
    df = monthly.copy()
    target_time = pd.to_datetime(df["target_time"])
    df["season"] = target_time.dt.month.map(season_label)
    df["target_year"] = target_time.dt.year
    df["bs_c3s"] = (df["c3s_selected_prob_dry"] - df["y_true_dry_frac"]) ** 2
    df["bs_persistence"] = (df["persistence_selected_prob_dry"] - df["y_true_dry_frac"]) ** 2
    df["delta_bs_c3s_minus_persistence"] = df["bs_c3s"] - df["bs_persistence"]
    df["c3s_minus_persistence_prob"] = (
        df["c3s_selected_prob_dry"] - df["persistence_selected_prob_dry"]
    )
    df["persistence_state"] = pd.cut(
        df["persistence_raw_prob_dry"],
        [-np.inf, 0.20, 0.50, np.inf],
        labels=["memory_wet_low_dryfrac", "memory_mixed", "memory_dry_high_dryfrac"],
    )
    df["c3s_anom_sign"] = np.where(
        df["forecast_rzsm_anom"] >= 0.0,
        "c3s_wet_anomaly",
        "c3s_dry_anomaly",
    )
    df["disagreement"] = pd.cut(
        df["c3s_minus_persistence_prob"],
        [-np.inf, -0.10, 0.10, np.inf],
        labels=["c3s_wetter_than_memory", "similar", "c3s_drier_than_memory"],
    )

    rows: list[dict[str, object]] = []
    for grouping in ["season", "target_year", "persistence_state", "c3s_anom_sign", "disagreement"]:
        for group_name, group in df.groupby(grouping, observed=False):
            if group.empty:
                continue
            seed = abs(hash((grouping, str(group_name)))) % (2**32)
            lo, hi = bootstrap_mean_ci(group["delta_bs_c3s_minus_persistence"], seed=seed)
            rows.append(
                {
                    "grouping": grouping,
                    "group": str(group_name),
                    "n_months": int(len(group)),
                    "obs_mean": float(group["y_true_dry_frac"].mean()),
                    "c3s_prob_mean": float(group["c3s_selected_prob_dry"].mean()),
                    "persistence_prob_mean": float(group["persistence_selected_prob_dry"].mean()),
                    "delta_bs_c3s_minus_persistence": float(
                        group["delta_bs_c3s_minus_persistence"].mean()
                    ),
                    "delta_bs_ci_low": lo,
                    "delta_bs_ci_high": hi,
                    "c3s_better_month_fraction": float(
                        (group["delta_bs_c3s_minus_persistence"] < 0.0).mean()
                    ),
                    "spearman_c3s_prob_obs": group["c3s_selected_prob_dry"].corr(
                        group["y_true_dry_frac"], method="spearman"
                    ),
                    "spearman_persistence_prob_obs": group[
                        "persistence_selected_prob_dry"
                    ].corr(group["y_true_dry_frac"], method="spearman"),
                }
            )
    return pd.DataFrame(rows)


def refresh_paper_c3s_summary_table() -> None:
    landsurface_dir = REPORT_DIR / "landsurface"
    candidates = []
    prefix = "landsurface_c3s_ecmwf_s51_vsm_"
    suffix = "_summary.csv"
    for path in sorted(landsurface_dir.glob(f"{prefix}*{suffix}")):
        stem = path.name
        if "_smoke_" in stem:
            continue
        body = stem[len(prefix) : -len(suffix)]
        parts = body.rsplit("_", 2)
        if len(parts) != 3:
            continue
        region_slug, target_start_year, target_end_year = parts
        if not (target_start_year.isdigit() and target_end_year.isdigit()):
            continue
        candidates.append((path, region_slug, int(target_start_year), int(target_end_year)))
    if not candidates:
        return

    current_end_year = max(item[3] for item in candidates)
    rows = []
    for path, region_slug, target_start_year, target_end_year in candidates:
        if target_end_year != current_end_year:
            continue
        table = pd.read_csv(path)
        table.insert(0, "region", region_slug)
        table.insert(1, "target_start_year", target_start_year)
        table.insert(2, "target_end_year", target_end_year)
        rows.append(table)
    if not rows:
        return
    PAPER_DIR.mkdir(parents=True, exist_ok=True)
    combined = pd.concat(rows, ignore_index=True)
    combined.to_csv(PAPER_DIR / "table39_c3s_ecmwf_native_vsm_benchmark.csv", index=False)


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
        "yearly": OUT_DIR / f"{args.out_prefix}_yearly_scores.csv",
        "regimes": OUT_DIR / f"{args.out_prefix}_persistence_regime_diagnostic.csv",
    }
    forecast.to_csv(outputs["forecast"], index=False)
    skipped.to_csv(outputs["skipped"], index=False)
    monthly["target_variable"] = "ERA5-Land root-zone soil moisture dry fraction"
    monthly["source_model"] = f"C3S_{args.originating_centre}_seasonal_original_system_{args.system}_vsm"
    monthly["benchmark_lead_months"] = args.lead_months
    monthly["region"] = args.region
    monthly.to_csv(outputs["monthly"], index=False)
    scores.to_csv(outputs["summary"], index=False)
    yearly_score_summary(monthly).to_csv(outputs["yearly"], index=False)
    persistence_regime_diagnostic(monthly).to_csv(outputs["regimes"], index=False)
    outputs["scores"].write_text(score_text + f"\nMonthly scores: {outputs['monthly']}\n", encoding="utf-8")

    print(score_text)
    print(f"Wrote forecast rows: {outputs['forecast']} rows={len(forecast):,}")
    print(f"Wrote skipped audit: {outputs['skipped']} rows={len(skipped):,}")
    print(f"Wrote monthly scores: {outputs['monthly']} rows={len(monthly):,}")
    print(f"Wrote score summary: {outputs['summary']}")
    print(f"Wrote yearly scores: {outputs['yearly']}")
    print(f"Wrote persistence-regime diagnostic: {outputs['regimes']}")

    if args.copy_report:
        (REPORT_DIR / "landsurface").mkdir(parents=True, exist_ok=True)
        for path in outputs.values():
            shutil.copy2(path, REPORT_DIR / "landsurface" / path.name)
        PAPER_DIR.mkdir(parents=True, exist_ok=True)
        refresh_paper_c3s_summary_table()


def main() -> None:
    args = parse_args()
    args.region = resolve_region(args.region).slug
    if args.soil_file is None:
        args.soil_file = default_soil_file(args.region)
    if args.out_prefix is None:
        args.out_prefix = default_out_prefix(args)
    args.cache_dir.mkdir(parents=True, exist_ok=True)

    observed = observed_rootzone_target(args)
    months = target_months_from_observed(args, observed)
    print(
        f"Selected target months: {months.min():%Y-%m} to {months.max():%Y-%m} "
        f"({len(months)} months)",
        flush=True,
    )
    forecast, skipped = build_forecast_rows(args, months)
    if args.fail_on_skipped and not skipped.empty:
        skipped_preview = skipped.head(10).to_string(index=False)
        raise RuntimeError(
            "C3S/ECMWF run has skipped months under --fail-on-skipped.\n"
            f"{skipped_preview}"
        )
    monthly, scores, score_text = score_benchmark(observed, forecast, args)
    write_outputs(forecast, skipped, monthly, scores, score_text, args)


if __name__ == "__main__":
    main()
