#!/usr/bin/env python
"""Short-record SMAP L4 validation for GEFSv12 land-surface probabilities.

This script scores existing GEFSv12 root-zone soil-moisture forecast
probabilities against regional SMAP L4 SPL4SMGP root-zone soil-moisture
percentile dry fractions. The target is the monthly mean fraction of regional
SMAP grid cells and snapshots with ``sm_rootzone_pctl <= 20``.

The default protocol uses 2015-2016 as a short validation/calibration period
and 2017-2019 as the frozen test period. Because the validation record is short
and the default monthly target is based on weekly snapshots, the result should
be interpreted as satellite-assimilated target validation, not a full
operational deployment benchmark.
"""

from __future__ import annotations

from argparse import ArgumentParser, Namespace
from pathlib import Path
import re
import shutil

import numpy as np
import pandas as pd
import xarray as xr

from region_config import REGIONS, resolve_region
from run_gefsv12_landsurface_stack_benchmark import (
    LANDSURFACE_DIR,
    PROJECT_ROOT,
    added_value_status,
    bootstrap_delta_bs,
    choose_convex_weight,
    forecast_path,
)
from run_landsurface_forecast_benchmark import brier, bss, bootstrap_bss, fit_isotonic


RAW_DIR = PROJECT_ROOT / "data" / "raw" / "smap_l4_spl4smgp"
PROCESSED = PROJECT_ROOT / "data" / "processed"
OUT_DIR = PROJECT_ROOT / "outputs"
REPORT_DIR = PROJECT_ROOT / "results" / "report"
DEFAULT_VARIABLE = "sm_rootzone_pctl"
SUBSET_RE = re.compile(
    r"SMAP_L4_SM_gph_(?P<stamp>\d{8}T\d{6})_.*__(?P<region>[a-z0-9_]+)__"
    r"(?P<variable>[a-z0-9_]+)_y(?P<y0>\d+)-(?P<y1>\d+)_x(?P<x0>\d+)-(?P<x1>\d+)\.nc4$"
)


def parse_args() -> Namespace:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument(
        "--regions",
        nargs="+",
        default=[
            "cvalley",
            "southern_great_plains",
            "mediterranean_spain",
            "murray_darling",
            "horn_of_africa",
        ],
    )
    parser.add_argument("--smap-dir", type=Path, default=RAW_DIR)
    parser.add_argument("--forecast-dir", type=Path, default=REPORT_DIR)
    parser.add_argument("--variable", default=DEFAULT_VARIABLE)
    parser.add_argument("--start", default="2015-04")
    parser.add_argument("--end", default="2019-12")
    parser.add_argument("--lead-months", type=int, default=1)
    parser.add_argument("--validation-start-year", type=int, default=2015)
    parser.add_argument("--validation-end-year", type=int, default=2016)
    parser.add_argument("--test-start-year", type=int, default=2017)
    parser.add_argument("--test-end-year", type=int, default=2019)
    parser.add_argument("--dry-percentile", type=float, default=20.0)
    parser.add_argument(
        "--climatology-mode",
        choices=["constant", "monthly"],
        default="constant",
        help="Constant validation mean is the default because SMAP validation has only ~20 months.",
    )
    parser.add_argument("--weight-steps", type=int, default=101)
    parser.add_argument("--n-bootstrap", type=int, default=2000)
    parser.add_argument("--out-prefix", default="landsurface_smap_l4_gefsv12_validation")
    parser.add_argument(
        "--target-tag",
        default=None,
        help=(
            "Optional tag added to processed SMAP target NetCDF filenames. "
            "Use this when comparing snapshot strategies so one run does not overwrite another."
        ),
    )
    parser.add_argument("--copy-report", action="store_true")
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument(
        "--process-only",
        action="store_true",
        help=(
            "Build SMAP monthly target NetCDFs and coverage audit without "
            "scoring GEFSv12 forecasts. Use this for modern years beyond the "
            "GEFSv12 reforecast window."
        ),
    )
    parser.add_argument("--allow-incomplete", action="store_true")
    return parser.parse_args()


def month_starts(start: str, end: str) -> list[pd.Timestamp]:
    months = pd.date_range(
        pd.Timestamp(start).to_period("M").to_timestamp(),
        pd.Timestamp(end).to_period("M").to_timestamp(),
        freq="MS",
    )
    return [pd.Timestamp(month) for month in months]


def looks_like_hdf(path: Path) -> bool:
    if not path.exists() or path.stat().st_size < 1024:
        return False
    with path.open("rb") as handle:
        return handle.read(8).startswith(b"\x89HDF")


def coordinate_cache_path(raw_dir: Path) -> Path:
    return raw_dir / "coordinates" / "smap_l4_spl4smgp_v008_coordinates.nc4"


def subset_files(raw_dir: Path, region: str, variable: str) -> list[Path]:
    files = sorted((raw_dir / "subsets" / region).glob(f"*/SMAP_L4_SM_gph_*__{region}__{variable}_*.nc4"))
    return [path for path in files if SUBSET_RE.search(path.name)]


def build_coverage_audit(args: Namespace) -> pd.DataFrame:
    months = month_starts(args.start, args.end)
    rows = []
    for region in [resolve_region(region).slug for region in args.regions]:
        files = subset_files(args.smap_dir, region, args.variable)
        by_month: dict[str, list[Path]] = {}
        for path in files:
            match = SUBSET_RE.search(path.name)
            if not match:
                continue
            stamp = pd.to_datetime(match.group("stamp"), format="%Y%m%dT%H%M%S")
            month = stamp.to_period("M").strftime("%Y-%m")
            by_month.setdefault(month, []).append(path)
        for month in months:
            key = month.strftime("%Y-%m")
            present = [path for path in by_month.get(key, []) if looks_like_hdf(path)]
            rows.append(
                {
                    "target_month": key,
                    "region": region,
                    "n_subset_files": len(by_month.get(key, [])),
                    "n_valid_subset_files": len(present),
                    "all_valid": len(present) > 0,
                    "size_bytes": sum(path.stat().st_size for path in present),
                }
            )
    return pd.DataFrame(rows)


def load_coordinate_subset(raw_dir: Path, region: str, y0: int, y1: int, x0: int, x1: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    coords_path = coordinate_cache_path(raw_dir)
    if not coords_path.exists():
        raise FileNotFoundError(
            f"SMAP coordinate cache not found: {coords_path}. "
            "Run scripts/download_smap_l4_regional_subsets.py first."
        )
    ds = xr.open_dataset(coords_path, engine="netcdf4")
    try:
        lat = ds["cell_lat"].isel(y=slice(y0, y1 + 1), x=slice(x0, x1 + 1)).to_numpy()
        lon = ds["cell_lon"].isel(y=slice(y0, y1 + 1), x=slice(x0, x1 + 1)).to_numpy()
    finally:
        ds.close()
    r = resolve_region(region)
    mask = (
        (lat >= r.lat_min)
        & (lat <= r.lat_max)
        & (lon >= r.lon_min)
        & (lon <= r.lon_max)
    )
    return lat, lon, mask


def find_variable(ds: xr.Dataset, variable: str) -> str:
    if variable in ds.data_vars:
        return variable
    suffix = f"_{variable}"
    matches = [name for name in ds.data_vars if name.endswith(suffix)]
    if matches:
        return matches[0]
    raise ValueError(f"Could not find {variable} in SMAP subset. Available: {list(ds.data_vars)}")


def target_from_subsets(region: str, args: Namespace) -> tuple[pd.DataFrame, xr.Dataset]:
    rows = []
    monthly_grids: list[xr.DataArray] = []
    files = subset_files(args.smap_dir, region, args.variable)
    grouped: dict[str, list[Path]] = {}
    for path in files:
        match = SUBSET_RE.search(path.name)
        if not match:
            continue
        stamp = pd.to_datetime(match.group("stamp"), format="%Y%m%dT%H%M%S")
        if stamp < pd.Timestamp(args.start) or stamp > pd.Timestamp(args.end) + pd.offsets.MonthEnd(0):
            continue
        grouped.setdefault(stamp.to_period("M").strftime("%Y-%m"), []).append(path)

    for month in month_starts(args.start, args.end):
        key = month.strftime("%Y-%m")
        paths = sorted(grouped.get(key, []))
        snapshot_arrays = []
        snapshot_dry_fracs = []
        snapshot_mean_pctls = []
        n_region_cells = np.nan
        for path in paths:
            if not looks_like_hdf(path):
                continue
            match = SUBSET_RE.search(path.name)
            if not match:
                continue
            y0, y1, x0, x1 = (int(match.group(name)) for name in ["y0", "y1", "x0", "x1"])
            _, _, mask = load_coordinate_subset(args.smap_dir, region, y0, y1, x0, x1)
            n_region_cells = int(mask.sum())
            ds = xr.open_dataset(path, engine="netcdf4")
            try:
                var = find_variable(ds, args.variable)
                values = ds[var].to_numpy().astype(float)
            finally:
                ds.close()
            values = np.where(values <= -9990.0, np.nan, values)
            values = np.where(mask, values, np.nan)
            if np.isfinite(values).sum() == 0:
                continue
            dry = values <= args.dry_percentile
            snapshot_dry_fracs.append(float(np.nanmean(np.where(np.isfinite(values), dry.astype(float), np.nan))))
            snapshot_mean_pctls.append(float(np.nanmean(values)))
            snapshot_arrays.append(values)
        if snapshot_arrays:
            stacked = np.stack(snapshot_arrays, axis=0)
            valid_count = np.isfinite(stacked).sum(axis=0)
            monthly_field = np.divide(
                np.nansum(stacked, axis=0),
                valid_count,
                out=np.full(stacked.shape[1:], np.nan, dtype=float),
                where=valid_count > 0,
            )
            dry_field = np.where(np.isfinite(monthly_field), monthly_field <= args.dry_percentile, np.nan)
            y_true = float(np.nanmean(dry_field))
            mean_pctl = float(np.nanmean(monthly_field))
            n_snapshots = len(snapshot_arrays)
            template = xr.DataArray(
                monthly_field[np.newaxis, :, :],
                dims=("time", "smap_y", "smap_x"),
                coords={"time": [month]},
                name="smap_rootzone_pctl",
            )
            monthly_grids.append(template)
        else:
            y_true = np.nan
            mean_pctl = np.nan
            n_snapshots = 0
        rows.append(
            {
                "target_time": month,
                "target_year": int(month.year),
                "target_month": int(month.month),
                "y_true_dry_frac": y_true,
                "observed_smap_rootzone_pctl": mean_pctl,
                "n_snapshots": int(n_snapshots),
                "n_region_cells": n_region_cells,
                "snapshot_dry_frac_mean": float(np.nanmean(snapshot_dry_fracs)) if snapshot_dry_fracs else np.nan,
                "snapshot_pctl_mean": float(np.nanmean(snapshot_mean_pctls)) if snapshot_mean_pctls else np.nan,
            }
        )

    observed = pd.DataFrame(rows).sort_values("target_time").reset_index(drop=True)
    observed = observed.dropna(subset=["y_true_dry_frac"]).copy()
    if observed.empty:
        raise FileNotFoundError(f"No valid SMAP subset observations for {region}.")

    val = observed.loc[
        (observed["target_year"] >= args.validation_start_year)
        & (observed["target_year"] <= args.validation_end_year)
    ].copy()
    if val.empty:
        raise ValueError(f"No SMAP validation months available for {region}.")
    if args.climatology_mode == "monthly":
        month_clim = val.groupby("target_month")["y_true_dry_frac"].mean()
        global_clim = float(val["y_true_dry_frac"].mean())
        observed["clim_prob_dry"] = observed["target_month"].map(month_clim).fillna(global_clim)
    else:
        observed["clim_prob_dry"] = float(val["y_true_dry_frac"].mean())

    persistence = observed[["target_time", "y_true_dry_frac"]].rename(
        columns={"target_time": "persistence_time", "y_true_dry_frac": "persistence_raw_prob_dry"}
    )
    observed["persistence_time"] = (
        observed["target_time"] - pd.DateOffset(months=args.lead_months)
    ).dt.to_period("M").dt.to_timestamp()
    observed = observed.merge(persistence, on="persistence_time", how="left")

    if monthly_grids:
        target_ds = xr.Dataset({"smap_rootzone_pctl": xr.concat(monthly_grids, dim="time")})
    else:
        target_ds = xr.Dataset()
    target_ds.attrs.update(
        {
            "source": "SPL4SMGP.008",
            "variable": args.variable,
            "region": region,
            "dry_percentile_threshold": float(args.dry_percentile),
            "monthly_aggregation": "mean of downloaded snapshots; dry target from monthly mean percentile field",
            "climatology_mode": args.climatology_mode,
            "validation_start_year": int(args.validation_start_year),
            "validation_end_year": int(args.validation_end_year),
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
        raise ValueError(f"No SMAP/GEFSv12 overlap for {region}.")
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
    if len(val) < 12 or test.empty:
        raise ValueError(
            f"Region {region} has too few SMAP validation/test months after merge: "
            f"validation={len(val)}, test={len(test)}."
        )

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

    summary = score_summary(region, test, weight, weight_val_bs, gefs_best, persistence_best, len(val), args)
    monthly_cols = [
        "target_time",
        "target_year",
        "target_month",
        "y_true_dry_frac",
        "clim_prob_dry",
        "observed_smap_rootzone_pctl",
        "n_snapshots",
        "n_region_cells",
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
    n_validation_months: int,
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
            seed=7101 + i,
        )
        delta = brier(y, pred) - bs_persistence
        delta_low, delta_high = bootstrap_delta_bs(
            test,
            candidate_col=col,
            reference_col="persistence_selected_prob_dry",
            n_bootstrap=args.n_bootstrap,
            seed=7201 + i,
        )
        rows.append(
            {
                "region": region,
                "target_product": "SPL4SMGP.008",
                "target_variable": f"{args.variable}_le_{args.dry_percentile:g}",
                "model": model,
                "n_validation_months": int(n_validation_months),
                "n_test_months": int(len(test)),
                "monthly_snapshot_proxy": True,
                "climatology_mode": args.climatology_mode,
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
                "mean_snapshots_per_month": float(test["n_snapshots"].mean()) if "n_snapshots" in test else np.nan,
                "target_aggregation": "monthly_mean_percentile_field",
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


def write_processed_targets(
    target_rows: list[dict[str, object]],
    monthly_rows: list[pd.DataFrame],
    coverage: pd.DataFrame,
    args: Namespace,
) -> None:
    OUT_DIR.mkdir(exist_ok=True)
    LANDSURFACE_DIR.mkdir(parents=True, exist_ok=True)
    coverage_path = OUT_DIR / f"{args.out_prefix}_coverage_audit.csv"
    targets_path = OUT_DIR / f"{args.out_prefix}_processed_targets.csv"
    monthly_path = OUT_DIR / f"{args.out_prefix}_monthly_targets.csv"
    coverage.to_csv(coverage_path, index=False)
    pd.DataFrame(target_rows).to_csv(targets_path, index=False)
    if monthly_rows:
        pd.concat(monthly_rows, ignore_index=True).to_csv(monthly_path, index=False)
    else:
        pd.DataFrame().to_csv(monthly_path, index=False)
    if args.copy_report:
        for path in [coverage_path, targets_path, monthly_path]:
            shutil.copy2(path, LANDSURFACE_DIR / path.name)
    print(f"Wrote coverage: {coverage_path}")
    print(f"Wrote processed targets: {targets_path}")
    print(f"Wrote monthly targets: {monthly_path}")


def main() -> None:
    args = parse_args()
    regions = [resolve_region(region).slug for region in args.regions]
    unsupported = sorted(set(regions) - set(REGIONS))
    if unsupported:
        raise SystemExit(f"Unsupported regions: {unsupported}")

    coverage = build_coverage_audit(args)
    grouped = coverage.groupby("region")["n_valid_subset_files"].agg(["sum", "min", "max"]).reset_index()
    print("SMAP subset coverage by region:")
    print(grouped.to_string(index=False))
    if args.audit_only:
        write_outputs([], [], coverage, args)
        return
    missing_months = coverage.loc[coverage["n_valid_subset_files"].eq(0)]
    if not missing_months.empty and not args.allow_incomplete:
        write_outputs([], [], coverage, args)
        raise SystemExit(
            f"Missing SMAP subset files for {len(missing_months)} region-month rows. "
            "Run scripts/download_smap_l4_regional_subsets.py first or pass --allow-incomplete for a smoke test."
        )

    summary_parts = []
    monthly_parts = []
    processed_targets = []
    processed_monthly = []
    for region in regions:
        action = "Processing SMAP target" if args.process_only else "Processing SMAP target and GEFSv12 validation"
        print(f"{action} for {region}", flush=True)
        observed, target_ds = target_from_subsets(region, args)
        tag = f"_{args.target_tag}" if args.target_tag else ""
        target_path = PROCESSED / f"smap_l4_rootzone_pctl_monthly_{region}_{args.start[:4]}_{args.end[:4]}{tag}.nc"
        target_ds.to_netcdf(target_path)
        processed_targets.append(
            {
                "region": region,
                "target_file": str(target_path.relative_to(PROJECT_ROOT)),
                "start": args.start,
                "end": args.end,
                "target_tag": args.target_tag or "",
                "n_months": int(observed["target_time"].nunique()),
                "mean_snapshots_per_month": float(observed["n_snapshots"].mean()),
                "min_snapshots_per_month": int(observed["n_snapshots"].min()),
                "max_snapshots_per_month": int(observed["n_snapshots"].max()),
                "mean_dry_fraction": float(observed["y_true_dry_frac"].mean()),
                "mean_rootzone_percentile": float(observed["observed_smap_rootzone_pctl"].mean()),
            }
        )
        observed_out = observed.copy()
        observed_out.insert(0, "region", region)
        observed_out["target_file"] = str(target_path.relative_to(PROJECT_ROOT))
        processed_monthly.append(observed_out)
        if args.process_only:
            continue
        summary, monthly = fit_and_score(region, observed, args)
        summary["target_file"] = str(target_path.relative_to(PROJECT_ROOT))
        monthly["target_file"] = str(target_path.relative_to(PROJECT_ROOT))
        summary_parts.append(summary)
        monthly_parts.append(monthly)
    if args.process_only:
        write_processed_targets(processed_targets, processed_monthly, coverage, args)
        return
    write_outputs(summary_parts, monthly_parts, coverage, args)


if __name__ == "__main__":
    main()
