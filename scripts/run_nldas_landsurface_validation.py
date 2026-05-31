#!/usr/bin/env python
"""Validate GEFSv12 land-surface forecasts against NLDAS Noah soil moisture.

This script builds an independent U.S. land-surface dry-fraction target from
NLDAS Noah monthly soil moisture, then scores existing GEFSv12 root-zone
soil-moisture forecasts against that target using the same validation/test
protocol as the ERA5-Land land-surface benchmark.

It is intentionally limited to U.S./North American regions covered by NLDAS.
Use ``scripts/download_nldas_noah_monthly.py`` first to acquire the monthly
NetCDF files with Earthdata/GES DISC credentials.
"""

from __future__ import annotations

from argparse import ArgumentParser, Namespace
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
import xarray as xr

from region_config import resolve_region
from run_gefsv12_landsurface_stack_benchmark import (
    LANDSURFACE_DIR,
    PROJECT_ROOT,
    added_value_status,
    bootstrap_delta_bs,
    choose_convex_weight,
    forecast_path,
)
from run_landsurface_forecast_benchmark import brier, bss, bootstrap_bss, fit_isotonic


RAW_DIR = PROJECT_ROOT / "data" / "raw" / "nldas_noah_monthly"
PROCESSED = PROJECT_ROOT / "data" / "processed"
OUT_DIR = PROJECT_ROOT / "outputs"
REPORT_DIR = PROJECT_ROOT / "results" / "report"
PAPER_DIR = REPORT_DIR / "paper"

NLDAS_REGIONS = {"cvalley", "southern_great_plains"}
DEFAULT_VARIABLE = "SoilM_0_100cm"


def parse_args() -> Namespace:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--regions", nargs="+", default=["cvalley", "southern_great_plains"])
    parser.add_argument("--nldas-dir", type=Path, default=RAW_DIR)
    parser.add_argument("--forecast-dir", type=Path, default=REPORT_DIR)
    parser.add_argument("--variable", default=DEFAULT_VARIABLE)
    parser.add_argument("--start", default="1991-01")
    parser.add_argument("--end", default="2019-12")
    parser.add_argument("--lead-months", type=int, default=1)
    parser.add_argument("--normal-start-year", type=int, default=1991)
    parser.add_argument("--normal-end-year", type=int, default=2016)
    parser.add_argument("--validation-start-year", type=int, default=2000)
    parser.add_argument("--validation-end-year", type=int, default=2016)
    parser.add_argument("--test-start-year", type=int, default=2017)
    parser.add_argument("--test-end-year", type=int, default=2019)
    parser.add_argument("--dry-quantile", type=float, default=0.20)
    parser.add_argument("--weight-steps", type=int, default=101)
    parser.add_argument("--n-bootstrap", type=int, default=2000)
    parser.add_argument("--out-prefix", default="landsurface_nldas_gefsv12_validation")
    parser.add_argument("--copy-report", action="store_true")
    parser.add_argument(
        "--audit-only",
        action="store_true",
        help="Only write the expected-file coverage audit and do not require all raw files.",
    )
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="Score with available months only. Intended for smoke tests, not manuscript claims.",
    )
    return parser.parse_args()


def month_starts(start: str, end: str) -> list[pd.Timestamp]:
    months = pd.date_range(
        pd.Timestamp(start).to_period("M").to_timestamp(),
        pd.Timestamp(end).to_period("M").to_timestamp(),
        freq="MS",
    )
    return [pd.Timestamp(month) for month in months]


def raw_path(raw_dir: Path, month: pd.Timestamp) -> Path:
    return raw_dir / f"{month:%Y}" / f"NLDAS_NOAH0125_M.A{month:%Y%m}.020.nc"


def looks_like_netcdf(path: Path) -> bool:
    if not path.exists() or path.stat().st_size < 1024:
        return False
    with path.open("rb") as handle:
        magic = handle.read(8)
    return magic.startswith(b"CDF") or magic.startswith(b"\x89HDF")


def build_coverage_audit(args: Namespace) -> pd.DataFrame:
    rows = []
    for month in month_starts(args.start, args.end):
        path = raw_path(args.nldas_dir, month)
        rows.append(
            {
                "target_month": month.strftime("%Y-%m"),
                "year": int(month.year),
                "month": int(month.month),
                "expected_file": str(path.relative_to(PROJECT_ROOT)) if path.is_absolute() else str(path),
                "exists": path.exists(),
                "looks_like_netcdf": looks_like_netcdf(path),
                "size_bytes": path.stat().st_size if path.exists() else 0,
            }
        )
    return pd.DataFrame(rows)


def choose_var(ds: xr.Dataset, requested: str) -> str:
    if requested in ds.data_vars:
        return requested
    fallbacks = [requested, "SoilM_0_100cm", "RootMoist", "SMAvail_0_100cm"]
    lower = {name.lower(): name for name in ds.data_vars}
    for name in fallbacks:
        if name.lower() in lower:
            return lower[name.lower()]
    raise ValueError(f"Could not find NLDAS soil-moisture variable. Requested={requested}; available={list(ds.data_vars)}")


def standardize_field(ds: xr.Dataset, variable: str, target_time: pd.Timestamp) -> xr.DataArray:
    var = choose_var(ds, variable)
    da = ds[var]
    rename = {}
    for dim in da.dims:
        low = dim.lower()
        if low in {"lat", "latitude"}:
            rename[dim] = "latitude"
        elif low in {"lon", "longitude"}:
            rename[dim] = "longitude"
        elif low in {"time", "valid_time"}:
            rename[dim] = "time"
    if rename:
        da = da.rename(rename)
    for dim in list(da.dims):
        if dim not in {"time", "latitude", "longitude"} and da.sizes[dim] == 1:
            da = da.squeeze(dim, drop=True)
    extra = [dim for dim in da.dims if dim not in {"time", "latitude", "longitude"}]
    if extra:
        raise ValueError(f"Unexpected NLDAS dimensions for {var}: {da.dims}")
    if "time" in da.dims and da.sizes["time"] == 1:
        da = da.squeeze("time", drop=True)
    if "time" in da.dims:
        raise ValueError(f"Unexpected multi-time monthly NLDAS field: {da.dims}")
    da = da.assign_coords(time=target_time).expand_dims("time")
    lon = da["longitude"]
    if float(lon.max()) > 180.0:
        da = da.assign_coords(longitude=(((lon + 180.0) % 360.0) - 180.0)).sortby("longitude")
    da = da.transpose("time", "latitude", "longitude")
    return da


def subset_region(da: xr.DataArray, region_slug: str) -> xr.DataArray:
    region = resolve_region(region_slug)
    if region.slug not in NLDAS_REGIONS:
        raise ValueError(
            f"NLDAS validation is only configured for {sorted(NLDAS_REGIONS)}; got {region.slug}."
        )
    lat_slice = slice(region.lat_min, region.lat_max)
    if da["latitude"][0] > da["latitude"][-1]:
        lat_slice = slice(region.lat_max, region.lat_min)
    return da.sel(latitude=lat_slice, longitude=slice(region.lon_min, region.lon_max))


def load_nldas_region_cube(region_slug: str, args: Namespace, audit: pd.DataFrame) -> xr.DataArray:
    fields = []
    good_months = audit.loc[audit["looks_like_netcdf"]].copy()
    for row in good_months.itertuples(index=False):
        month = pd.Timestamp(f"{row.target_month}-01")
        path = raw_path(args.nldas_dir, month)
        ds = xr.open_dataset(path)
        try:
            field = standardize_field(ds, args.variable, month)
            fields.append(subset_region(field, region_slug).load())
        finally:
            ds.close()
    if not fields:
        raise FileNotFoundError("No valid NLDAS monthly NetCDF files are available.")
    cube = xr.concat(fields, dim="time").sortby("time")
    cube.name = "nldas_rootzone_sm"
    return cube


def observed_from_cube(cube: xr.DataArray, region_slug: str, args: Namespace) -> tuple[pd.DataFrame, xr.Dataset]:
    years = pd.DatetimeIndex(cube["time"].values).year
    normal_mask = (years >= args.normal_start_year) & (years <= args.normal_end_year)
    if int(normal_mask.sum()) < 12:
        raise ValueError("Not enough NLDAS normal-period months to define dry thresholds.")
    normal = cube.isel(time=normal_mask)
    thresholds = normal.groupby("time.month").quantile(args.dry_quantile, dim="time", skipna=True)

    rows = []
    for i, target_time in enumerate(pd.DatetimeIndex(cube["time"].values)):
        field = cube.isel(time=i)
        threshold = thresholds.sel(month=int(target_time.month))
        valid = np.isfinite(field) & np.isfinite(threshold)
        dry = xr.where(valid, field <= threshold, np.nan)
        rows.append(
            {
                "target_time": pd.Timestamp(target_time).to_period("M").to_timestamp(),
                "target_year": int(target_time.year),
                "target_month": int(target_time.month),
                "y_true_dry_frac": float(dry.mean(dim=["latitude", "longitude"], skipna=True).values),
                "observed_nldas_rzsm": float(field.mean(dim=["latitude", "longitude"], skipna=True).values),
            }
        )
    observed = pd.DataFrame(rows).sort_values("target_time").reset_index(drop=True)
    train = observed.loc[
        (observed["target_year"] >= args.normal_start_year)
        & (observed["target_year"] <= args.normal_end_year)
    ].copy()
    month_clim = train.groupby("target_month")["y_true_dry_frac"].mean()
    global_clim = float(train["y_true_dry_frac"].mean())
    observed["clim_prob_dry"] = observed["target_month"].map(month_clim).fillna(global_clim)
    persistence = observed[["target_time", "y_true_dry_frac"]].rename(
        columns={"target_time": "persistence_time", "y_true_dry_frac": "persistence_raw_prob_dry"}
    )
    observed["persistence_time"] = (
        observed["target_time"] - pd.DateOffset(months=args.lead_months)
    ).dt.to_period("M").dt.to_timestamp()
    observed = observed.merge(persistence, on="persistence_time", how="left")

    target_ds = xr.Dataset({"nldas_rootzone_sm": cube})
    target_ds.attrs.update(
        {
            "source": "NLDAS_NOAH0125_M.2.0",
            "variable": args.variable,
            "region": region_slug,
            "dry_quantile": float(args.dry_quantile),
            "normal_start_year": int(args.normal_start_year),
            "normal_end_year": int(args.normal_end_year),
        }
    )
    return observed, target_ds


def fit_and_score(region: str, observed: pd.DataFrame, args: Namespace) -> tuple[pd.DataFrame, pd.DataFrame]:
    fpath = forecast_path(region, args.forecast_dir)
    if not fpath.exists():
        raise FileNotFoundError(f"GEFSv12 forecast file not found for {region}: {fpath}")
    forecast = pd.read_csv(fpath, parse_dates=["target_time", "valid_time", "init_time"])
    merged = observed.merge(forecast, on="target_time", how="inner", suffixes=("", "_forecast"))
    merged = merged.dropna(
        subset=["forecast_rzsm", "forecast_rzsm_anom", "persistence_raw_prob_dry", "y_true_dry_frac", "clim_prob_dry"]
    ).copy()
    if merged.empty:
        raise ValueError(f"No NLDAS/GEFSv12 overlap for {region}.")
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
    if val.empty or test.empty:
        raise ValueError(f"Region {region} has empty validation or test split after NLDAS/GEFSv12 merge.")

    val_raw, test_raw = fit_isotonic(val, test, "gefs_raw_dry_signal")
    val_anom, test_anom = fit_isotonic(val, test, "gefs_anom_dry_signal")
    val_pers_iso, test_pers_iso = fit_isotonic(val, test, "persistence_raw_prob_dry")
    val = val.copy()
    test = test.copy()
    val["gefs_raw_isotonic_prob_dry"] = val_raw
    val["gefs_anom_isotonic_prob_dry"] = val_anom
    val["persistence_isotonic_prob_dry"] = val_pers_iso
    test["gefs_raw_isotonic_prob_dry"] = test_raw
    test["gefs_anom_isotonic_prob_dry"] = test_anom
    test["persistence_isotonic_prob_dry"] = test_pers_iso

    val_bs = {
        "gefs_raw_isotonic": brier(val["y_true_dry_frac"], val["gefs_raw_isotonic_prob_dry"]),
        "gefs_anom_isotonic": brier(val["y_true_dry_frac"], val["gefs_anom_isotonic_prob_dry"]),
        "persistence_raw": brier(val["y_true_dry_frac"], val["persistence_raw_prob_dry"]),
        "persistence_isotonic": brier(val["y_true_dry_frac"], val["persistence_isotonic_prob_dry"]),
    }
    gefs_best = min({k: v for k, v in val_bs.items() if k.startswith("gefs_")}, key=val_bs.get)
    persistence_best = min({k: v for k, v in val_bs.items() if k.startswith("persistence_")}, key=val_bs.get)
    for frame in [val, test]:
        frame["gefs_selected_prob_dry"] = (
            frame["gefs_raw_isotonic_prob_dry"]
            if gefs_best == "gefs_raw_isotonic"
            else frame["gefs_anom_isotonic_prob_dry"]
        )
        frame["persistence_selected_prob_dry"] = (
            frame["persistence_raw_prob_dry"]
            if persistence_best == "persistence_raw"
            else frame["persistence_isotonic_prob_dry"]
        )

    weight, weight_val_bs = choose_convex_weight(
        val,
        "gefs_selected_prob_dry",
        "persistence_selected_prob_dry",
        args.weight_steps,
    )
    for frame in [val, test]:
        frame["stack_validation_selected_prob_dry"] = (
            weight * frame["gefs_selected_prob_dry"]
            + (1.0 - weight) * frame["persistence_selected_prob_dry"]
        ).clip(0.0, 1.0)

    summary = score_summary(region, test, weight, weight_val_bs, gefs_best, persistence_best, args)
    monthly_cols = [
        "target_time",
        "target_year",
        "target_month",
        "y_true_dry_frac",
        "clim_prob_dry",
        "observed_nldas_rzsm",
        "forecast_rzsm",
        "forecast_rzsm_anom",
        "gefs_selected_prob_dry",
        "persistence_selected_prob_dry",
        "stack_validation_selected_prob_dry",
    ]
    monthly = test[[c for c in monthly_cols if c in test.columns]].copy()
    monthly.insert(0, "region", region)
    return summary, monthly


def score_summary(
    region: str,
    test: pd.DataFrame,
    selected_weight_gefs: float,
    selected_weight_val_bs: float,
    gefs_best: str,
    persistence_best: str,
    args: Namespace,
) -> pd.DataFrame:
    y = test["y_true_dry_frac"].to_numpy(dtype=float)
    clim = test["clim_prob_dry"].to_numpy(dtype=float)
    persistence = test["persistence_selected_prob_dry"].to_numpy(dtype=float)
    bs_clim = brier(y, clim)
    bs_persistence = brier(y, persistence)
    rows = []
    candidates = [
        ("gefs_selected", "gefs_selected_prob_dry"),
        ("persistence_selected", "persistence_selected_prob_dry"),
        ("stack_validation_selected", "stack_validation_selected_prob_dry"),
    ]
    for i, (model, col) in enumerate(candidates):
        pred = test[col].to_numpy(dtype=float)
        ci_low, ci_high = bootstrap_bss(
            test,
            pred_col=col,
            ref_col="clim_prob_dry",
            n_bootstrap=args.n_bootstrap,
            seed=4101 + i,
        )
        delta = brier(y, pred) - bs_persistence
        delta_low, delta_high = bootstrap_delta_bs(
            test,
            candidate_col=col,
            reference_col="persistence_selected_prob_dry",
            n_bootstrap=args.n_bootstrap,
            seed=4201 + i,
        )
        rows.append(
            {
                "region": region,
                "target_product": "NLDAS_NOAH0125_M",
                "target_variable": args.variable,
                "model": model,
                "n_test_months": int(len(test)),
                "bs_climatology": bs_clim,
                "bs_model": brier(y, pred),
                "bs_persistence_selected": bs_persistence,
                "bss_vs_climatology": bss(y, pred, clim),
                "bss_ci_low": ci_low,
                "bss_ci_high": ci_high,
                "bss_vs_persistence_selected": bss(y, pred, persistence),
                "delta_bs_model_minus_persistence_selected": delta,
                "delta_bs_ci_low": delta_low,
                "delta_bs_ci_high": delta_high,
                "added_value_status_selected": added_value_status(delta, delta_low, delta_high),
                "selected_weight_gefs": selected_weight_gefs if model == "stack_validation_selected" else np.nan,
                "selected_weight_persistence": (1.0 - selected_weight_gefs) if model == "stack_validation_selected" else np.nan,
                "selected_weight_val_bs": selected_weight_val_bs if model == "stack_validation_selected" else np.nan,
                "selected_gefs_calibration": gefs_best,
                "selected_persistence_calibration": persistence_best,
                "spearman_model_vs_observed": test[col].corr(test["y_true_dry_frac"], method="spearman"),
            }
        )
    return pd.DataFrame(rows)


def write_outputs(
    summary_parts: list[pd.DataFrame],
    monthly_parts: list[pd.DataFrame],
    coverage: pd.DataFrame,
    args: Namespace,
) -> None:
    OUT_DIR.mkdir(exist_ok=True)
    LANDSURFACE_DIR.mkdir(parents=True, exist_ok=True)
    prefix = args.out_prefix
    coverage_path = OUT_DIR / f"{prefix}_coverage_audit.csv"
    coverage.to_csv(coverage_path, index=False)
    paths = {"coverage": coverage_path}
    if summary_parts:
        summary = pd.concat(summary_parts, ignore_index=True)
        monthly = pd.concat(monthly_parts, ignore_index=True)
        paths["summary"] = OUT_DIR / f"{prefix}_summary.csv"
        paths["monthly"] = OUT_DIR / f"{prefix}_monthly_scores.csv"
        summary.to_csv(paths["summary"], index=False)
        monthly.to_csv(paths["monthly"], index=False)
    if args.copy_report:
        for path in paths.values():
            shutil.copy2(path, LANDSURFACE_DIR / path.name)
    for label, path in paths.items():
        print(f"Wrote {label}: {path}")


def main() -> None:
    args = parse_args()
    regions = [resolve_region(region).slug for region in args.regions]
    unsupported = sorted(set(regions) - NLDAS_REGIONS)
    if unsupported:
        raise SystemExit(f"NLDAS validation only supports {sorted(NLDAS_REGIONS)}; unsupported={unsupported}")

    coverage = build_coverage_audit(args)
    missing = int((~coverage["looks_like_netcdf"]).sum())
    print(
        f"NLDAS coverage: {len(coverage) - missing}/{len(coverage)} monthly NetCDF files available "
        f"for {args.start} to {args.end}."
    )
    if args.audit_only:
        write_outputs([], [], coverage, args)
        return
    if missing and not args.allow_incomplete:
        write_outputs([], [], coverage, args)
        raise SystemExit(
            f"Missing or invalid NLDAS files: {missing}/{len(coverage)}. "
            "Run scripts/download_nldas_noah_monthly.py with Earthdata credentials first, "
            "or pass --allow-incomplete for a non-manuscript smoke test."
        )

    summary_parts = []
    monthly_parts = []
    for region in regions:
        print(f"Processing NLDAS target and GEFSv12 validation for {region}", flush=True)
        cube = load_nldas_region_cube(region, args, coverage)
        observed, target_ds = observed_from_cube(cube, region, args)
        target_path = PROCESSED / f"nldas_noah_rootzone_monthly_{region}_{args.start[:4]}_{args.end[:4]}.nc"
        target_ds.to_netcdf(target_path)
        summary, monthly = fit_and_score(region, observed, args)
        summary["target_file"] = str(target_path.relative_to(PROJECT_ROOT))
        monthly["target_file"] = str(target_path.relative_to(PROJECT_ROOT))
        summary_parts.append(summary)
        monthly_parts.append(monthly)
    write_outputs(summary_parts, monthly_parts, coverage, args)


if __name__ == "__main__":
    main()
