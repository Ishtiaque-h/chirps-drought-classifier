#!/usr/bin/env python
"""Conservative GEFSv12 + persistence stack for the land-surface benchmark.

This script tests whether the public GEFSv12 root-zone soil-moisture reforecast
adds useful information beyond same-target ERA5-Land persistence after the
single-source GEFSv12 benchmark has already been calibrated.

The stack is intentionally simple and leakage-safe:

  - rebuild the ERA5-Land root-zone dry-fraction target;
  - read completed GEFSv12 day-15 forecast CSVs when available, or rebuild
    other valid-day/init-lag combinations through the GEFSv12 benchmark helper;
  - fit isotonic calibrators on the validation period only;
  - select the GEFSv12 calibration, persistence calibration, and convex blend
    weight using validation Brier score only;
  - score the frozen test period against climatology and same-target
    persistence.

This is a benchmark/control, not a new high-capacity ML model.
"""

from __future__ import annotations

from argparse import ArgumentParser, Namespace
from pathlib import Path
import re
import shutil

import numpy as np
import pandas as pd

from region_config import resolve_region
from run_landsurface_forecast_benchmark import (
    brier,
    bss,
    bootstrap_bss,
    default_soil_file,
    fit_isotonic,
    observed_rootzone_target,
)
from run_gefsv12_landsurface_benchmark import (
    DEFAULT_MEMBERS,
    RAW_DIR,
    build_forecast_rows,
    target_months_from_observed,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = PROJECT_ROOT / "results" / "report"
LANDSURFACE_DIR = REPORT_DIR / "landsurface"
OUT_DIR = PROJECT_ROOT / "outputs"

SOIL_FILE_RE = re.compile(
    r"era5_land_soil_moisture_monthly_(?P<region>[a-z0-9_]+)_(?P<start>\d{4})_(?P<end>\d{4})\.nc$"
)


def parse_args() -> Namespace:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument(
        "--regions",
        nargs="+",
        default=["cvalley", "southern_great_plains", "mediterranean_spain"],
        help="Region slugs to score.",
    )
    parser.add_argument("--forecast-dir", type=Path, default=REPORT_DIR)
    parser.add_argument("--lead-months", type=int, default=1)
    parser.add_argument("--validation-start-year", type=int, default=2000)
    parser.add_argument("--validation-end-year", type=int, default=2016)
    parser.add_argument("--test-start-year", type=int, default=2017)
    parser.add_argument("--test-end-year", type=int, default=2019)
    parser.add_argument("--normal-start-year", type=int, default=1991)
    parser.add_argument("--normal-end-year", type=int, default=2016)
    parser.add_argument("--dry-quantile", type=float, default=0.20)
    parser.add_argument("--start-target", default="2000-01")
    parser.add_argument("--end-target", default="2019-12")
    parser.add_argument(
        "--valid-day",
        type=int,
        default=15,
        help="Single GEFSv12 target-month valid day used when --valid-days is omitted.",
    )
    parser.add_argument(
        "--valid-days",
        nargs="+",
        type=int,
        default=None,
        help="Optional valid-day sweep. If omitted, uses --valid-day.",
    )
    parser.add_argument(
        "--init-lag-weeks",
        nargs="+",
        type=int,
        default=[0],
        help="Weekly long-init lag(s) relative to the latest long init before target-month start.",
    )
    parser.add_argument("--members", nargs="+", default=DEFAULT_MEMBERS)
    parser.add_argument("--min-members", type=int, default=6)
    parser.add_argument("--cache-dir", type=Path, default=RAW_DIR)
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--max-months", type=int, default=None)
    parser.add_argument("--progress-every", type=int, default=120)
    parser.add_argument("--weight-steps", type=int, default=101)
    parser.add_argument("--n-bootstrap", type=int, default=2000)
    parser.add_argument(
        "--out-prefix",
        default=None,
    )
    parser.add_argument("--copy-report", action="store_true")
    return parser.parse_args()


def forecast_path(region_slug: str, forecast_dir: Path) -> Path:
    region = resolve_region(region_slug).slug
    if region == "cvalley":
        name = "landsurface_gefsv12_rzsm_day15_hindcastcal_forecast.csv"
    else:
        name = f"landsurface_gefsv12_rzsm_{region}_day15_hindcastcal_forecast.csv"
    return forecast_dir / name


def stable_unique_ints(values: list[int]) -> list[int]:
    out: list[int] = []
    seen: set[int] = set()
    for value in values:
        ivalue = int(value)
        if ivalue in seen:
            continue
        seen.add(ivalue)
        out.append(ivalue)
    return out


def default_out_prefix(valid_days: list[int], init_lags: list[int]) -> str:
    if valid_days == [15] and init_lags == [0]:
        return "landsurface_gefsv12_rzsm_stack_day15_hindcastcal"
    return "landsurface_gefsv12_rzsm_stack_lead_validday_sensitivity_hindcastcal"


def resolve_soil_file(region_slug: str) -> Path:
    candidate = default_soil_file(region_slug)
    if candidate.exists():
        return candidate

    processed = PROJECT_ROOT / "data" / "processed"
    matches = list(processed.glob(f"era5_land_soil_moisture_monthly_{region_slug}_*.nc"))
    if not matches:
        raise FileNotFoundError(
            f"No ERA5-Land soil moisture file found for region={region_slug}. "
            f"Tried {candidate}."
        )

    scored: list[tuple[int, Path]] = []
    for path in matches:
        match = SOIL_FILE_RE.search(path.name)
        end_year = int(match.group("end")) if match and match.group("region") == region_slug else -1
        scored.append((end_year, path))
    scored.sort(key=lambda item: (item[0], item[1].name))
    return scored[-1][1]


def choose_convex_weight(
    validation: pd.DataFrame,
    gefs_col: str,
    persistence_col: str,
    weight_steps: int,
) -> tuple[float, float]:
    weights = np.linspace(0.0, 1.0, max(2, int(weight_steps)))
    y = validation["y_true_dry_frac"].to_numpy(dtype=float)
    gefs = validation[gefs_col].to_numpy(dtype=float)
    persistence = validation[persistence_col].to_numpy(dtype=float)
    rows = []
    for weight in weights:
        pred = weight * gefs + (1.0 - weight) * persistence
        rows.append((brier(y, pred), float(weight)))
    rows.sort(key=lambda item: (item[0], abs(item[1] - 0.5)))
    return rows[0][1], rows[0][0]


def bootstrap_delta_bs(
    monthly: pd.DataFrame,
    candidate_col: str,
    reference_col: str,
    n_bootstrap: int,
    seed: int,
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    idx = np.arange(len(monthly))
    y = monthly["y_true_dry_frac"].to_numpy(dtype=float)
    candidate = monthly[candidate_col].to_numpy(dtype=float)
    reference = monthly[reference_col].to_numpy(dtype=float)
    vals = np.empty(n_bootstrap, dtype=float)
    for i in range(n_bootstrap):
        sample = rng.choice(idx, size=len(idx), replace=True)
        vals[i] = brier(y[sample], candidate[sample]) - brier(y[sample], reference[sample])
    lo, hi = np.nanquantile(vals, [0.025, 0.975])
    return float(lo), float(hi)


def added_value_status(delta_bs: float, ci_low: float, ci_high: float) -> str:
    if ci_high < 0:
        return "stack_robust_added_value"
    if ci_low > 0:
        return "persistence_robustly_better"
    if delta_bs < 0:
        return "stack_uncertain_added_value"
    if delta_bs > 0:
        return "persistence_uncertain_better"
    return "tied"


def score_rows(
    region: str,
    test: pd.DataFrame,
    selected_weight_gefs: float,
    selected_weight_val_bs: float,
    gefs_best: str,
    persistence_best: str,
    n_bootstrap: int,
) -> pd.DataFrame:
    y = test["y_true_dry_frac"].to_numpy(dtype=float)
    clim = test["clim_prob_dry"].to_numpy(dtype=float)
    persistence = test["persistence_selected_prob_dry"].to_numpy(dtype=float)
    bs_clim = brier(y, clim)
    bs_persistence = brier(y, persistence)

    candidates = [
        ("gefs_selected", "gefs_selected_prob_dry"),
        ("persistence_selected", "persistence_selected_prob_dry"),
        ("stack_equal_weight", "stack_equal_prob_dry"),
        ("stack_validation_selected", "stack_validation_selected_prob_dry"),
    ]
    rows = []
    for i, (label, col) in enumerate(candidates):
        pred = test[col].to_numpy(dtype=float)
        ci_low, ci_high = bootstrap_bss(
            test,
            col,
            ref_col="clim_prob_dry",
            n_bootstrap=n_bootstrap,
            seed=1201 + i,
        )
        delta = brier(y, pred) - bs_persistence
        delta_low, delta_high = bootstrap_delta_bs(
            test,
            col,
            "persistence_selected_prob_dry",
            n_bootstrap=n_bootstrap,
            seed=1301 + i,
        )
        rows.append(
            {
                "region": region,
                "model": label,
                "n_months": int(len(test)),
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
                "selected_weight_gefs": selected_weight_gefs if label == "stack_validation_selected" else np.nan,
                "selected_weight_persistence": (1.0 - selected_weight_gefs)
                if label == "stack_validation_selected"
                else np.nan,
                "selected_weight_val_bs": selected_weight_val_bs
                if label == "stack_validation_selected"
                else np.nan,
                "selected_gefs_calibration": gefs_best,
                "selected_persistence_calibration": persistence_best,
                "spearman_model_vs_observed": test[col].corr(test["y_true_dry_frac"], method="spearman"),
                "amplitude_ratio_model": (
                    float(test[col].std(ddof=0) / test["y_true_dry_frac"].std(ddof=0))
                    if float(test["y_true_dry_frac"].std(ddof=0)) > 0
                    else np.nan
                ),
            }
        )
    return pd.DataFrame(rows)


def load_or_build_forecast(region: str, observed: pd.DataFrame, args: Namespace) -> tuple[pd.DataFrame, str]:
    use_existing = int(args.valid_day) == 15 and int(args.init_lag_weeks) == 0
    forecast_file = forecast_path(region, args.forecast_dir)
    if use_existing and forecast_file.exists() and not args.refresh:
        forecast = pd.read_csv(forecast_file, parse_dates=["target_time", "valid_time", "init_time"])
        return forecast, str(forecast_file.relative_to(PROJECT_ROOT))

    months = target_months_from_observed(args, observed)
    forecast, _skipped = build_forecast_rows(args, months)
    return forecast, "built_from_cached_or_remote_gefsv12_reforecast"


def run_region(region_slug: str, args: Namespace) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    region = resolve_region(region_slug).slug
    target_args = Namespace(
        soil_file=resolve_soil_file(region),
        lead_months=args.lead_months,
        normal_start_year=args.normal_start_year,
        normal_end_year=args.normal_end_year,
        dry_quantile=args.dry_quantile,
    )
    observed = observed_rootzone_target(target_args)
    combo_args = Namespace(**vars(args))
    combo_args.region = region
    combo_args.soil_file = target_args.soil_file
    forecast, forecast_source = load_or_build_forecast(region, observed, combo_args)
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
        raise ValueError(f"Region {region} has empty validation or test split after merge.")

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
        frame["stack_equal_prob_dry"] = 0.5 * frame["gefs_selected_prob_dry"] + 0.5 * frame["persistence_selected_prob_dry"]
        frame["stack_validation_selected_prob_dry"] = (
            weight * frame["gefs_selected_prob_dry"]
            + (1.0 - weight) * frame["persistence_selected_prob_dry"]
        ).clip(0.0, 1.0)

    monthly_cols = [
        "target_time",
        "target_year",
        "target_month",
        "region",
        "y_true_dry_frac",
        "clim_prob_dry",
        "forecast_rzsm",
        "forecast_rzsm_anom",
        "forecast_rzsm_member_std",
        "n_members",
        "lead_days_to_valid",
        "valid_day",
        "init_lag_weeks",
        "gefs_selected_prob_dry",
        "persistence_selected_prob_dry",
        "stack_equal_prob_dry",
        "stack_validation_selected_prob_dry",
    ]
    test["region"] = region
    test["valid_day"] = int(args.valid_day)
    test["init_lag_weeks"] = int(args.init_lag_weeks)
    monthly = test[[c for c in monthly_cols if c in test.columns]].copy()
    scores = score_rows(
        region,
        test,
        selected_weight_gefs=weight,
        selected_weight_val_bs=weight_val_bs,
        gefs_best=gefs_best,
        persistence_best=persistence_best,
        n_bootstrap=args.n_bootstrap,
    )
    scores["valid_day"] = int(args.valid_day)
    scores["init_lag_weeks"] = int(args.init_lag_weeks)
    meta = {
        "region": region,
        "region_name": resolve_region(region).name,
        "valid_day": int(args.valid_day),
        "init_lag_weeks": int(args.init_lag_weeks),
        "forecast_source": forecast_source,
        "validation_months": int(val["target_time"].nunique()),
        "test_months": int(test["target_time"].nunique()),
        "selected_weight_gefs": float(weight),
        "selected_weight_persistence": float(1.0 - weight),
        "selected_weight_val_bs": float(weight_val_bs),
        "selected_gefs_calibration": gefs_best,
        "selected_persistence_calibration": persistence_best,
    }
    return monthly, scores, meta


def format_text(scores: pd.DataFrame, meta_rows: list[dict[str, object]], args: Namespace) -> str:
    lines = [
        "GEFSv12 + Persistence Land-Surface Stack Benchmark",
        "=" * 72,
        "Design: validation-selected convex blend of GEFSv12 selected probability and same-target persistence selected probability.",
        f"Validation years: {args.validation_start_year}-{args.validation_end_year}",
        f"Test years: {args.test_start_year}-{args.test_end_year}",
        "",
    ]
    for meta in meta_rows:
        region = str(meta["region"])
        region_scores = scores.loc[
            scores["region"].eq(region)
            & scores["valid_day"].eq(int(meta["valid_day"]))
            & scores["init_lag_weeks"].eq(int(meta["init_lag_weeks"]))
        ].copy()
        stack = region_scores.loc[region_scores["model"].eq("stack_validation_selected")].iloc[0]
        gefs = region_scores.loc[region_scores["model"].eq("gefs_selected")].iloc[0]
        pers = region_scores.loc[region_scores["model"].eq("persistence_selected")].iloc[0]
        lines.extend(
            [
                f"{meta['region_name']} ({region})",
                f"  valid day={int(meta['valid_day'])} init lag={int(meta['init_lag_weeks'])}w",
                f"  validation months={meta['validation_months']} test months={meta['test_months']}",
                f"  selected blend weight: GEFSv12={float(meta['selected_weight_gefs']):.2f}, persistence={float(meta['selected_weight_persistence']):.2f}",
                f"  GEFSv12 selected BSS={float(gefs['bss_vs_climatology']):+.3f}; persistence selected BSS={float(pers['bss_vs_climatology']):+.3f}",
                f"  stack selected BSS={float(stack['bss_vs_climatology']):+.3f} "
                f"CI[{float(stack['bss_ci_low']):+.3f}, {float(stack['bss_ci_high']):+.3f}]",
                f"  stack vs selected persistence BSS={float(stack['bss_vs_persistence_selected']):+.3f}; "
                f"delta BS={float(stack['delta_bs_model_minus_persistence_selected']):+.4f} "
                f"CI[{float(stack['delta_bs_ci_low']):+.4f}, {float(stack['delta_bs_ci_high']):+.4f}] "
                f"status={stack['added_value_status_selected']}",
                "",
            ]
        )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    OUT_DIR.mkdir(exist_ok=True)
    LANDSURFACE_DIR.mkdir(parents=True, exist_ok=True)
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    valid_days = stable_unique_ints(args.valid_days if args.valid_days is not None else [args.valid_day])
    init_lags = stable_unique_ints(args.init_lag_weeks)
    if args.out_prefix is None:
        args.out_prefix = default_out_prefix(valid_days, init_lags)

    monthly_parts: list[pd.DataFrame] = []
    score_parts: list[pd.DataFrame] = []
    meta_rows: list[dict[str, object]] = []
    for init_lag in sorted(init_lags):
        for valid_day in sorted(valid_days):
            for region in args.regions:
                combo_args = Namespace(**vars(args))
                combo_args.valid_day = int(valid_day)
                combo_args.init_lag_weeks = int(init_lag)
                print(f"Running region={region} valid_day={valid_day} init_lag={init_lag}w", flush=True)
                monthly, scores, meta = run_region(region, combo_args)
                monthly_parts.append(monthly)
                score_parts.append(scores)
                meta_rows.append(meta)

    monthly_all = pd.concat(monthly_parts, ignore_index=True)
    scores_all = pd.concat(score_parts, ignore_index=True)
    meta = pd.DataFrame(meta_rows)
    text = format_text(scores_all, meta_rows, args)

    outputs = {
        "monthly": OUT_DIR / f"{args.out_prefix}_monthly_scores.csv",
        "summary": OUT_DIR / f"{args.out_prefix}_summary.csv",
        "meta": OUT_DIR / f"{args.out_prefix}_meta.csv",
        "scores": OUT_DIR / f"{args.out_prefix}_scores.txt",
    }
    monthly_all.to_csv(outputs["monthly"], index=False)
    scores_all.to_csv(outputs["summary"], index=False)
    meta.to_csv(outputs["meta"], index=False)
    outputs["scores"].write_text(text + "\n", encoding="utf-8")

    print(text)
    print(f"Wrote monthly scores: {outputs['monthly']} rows={len(monthly_all):,}")
    print(f"Wrote score summary: {outputs['summary']} rows={len(scores_all):,}")
    print(f"Wrote metadata: {outputs['meta']} rows={len(meta):,}")

    if args.copy_report:
        for path in outputs.values():
            shutil.copy2(path, LANDSURFACE_DIR / path.name)


if __name__ == "__main__":
    main()
