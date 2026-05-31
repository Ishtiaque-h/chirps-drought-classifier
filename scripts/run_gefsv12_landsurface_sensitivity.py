#!/usr/bin/env python
"""Sweep GEFSv12 lead/valid-day sensitivity for the land-surface benchmark.

This is the scientifically defensible follow-up to the promising GEFSv12 RZSM
hindcast-calibrated checkpoint: test whether the positive skill is stable
across reasonable choices of

  - valid day (day-of-month within the target month at 12 UTC), and
  - lead (weekly long reforecast initialization lag in weeks relative to the
    latest long init before the target month begins).

It reuses the same target definition and scoring protocol as
`scripts/run_gefsv12_landsurface_benchmark.py` (isotonic calibration on a
validation window and BSS vs climatology on a frozen test window).

Outputs (under outputs/):
  - <out_prefix>_sensitivity_summary.csv
  - <out_prefix>_sensitivity_grid.csv
  - <out_prefix>_sensitivity_monthly_scores.csv
  - <out_prefix>_sensitivity_summary.txt

Use --copy-report to also copy these artifacts to results/report/landsurface/.
"""

from __future__ import annotations

from argparse import ArgumentParser, Namespace
from pathlib import Path
import re
import shutil

import numpy as np
import pandas as pd

from region_config import resolve_region
from run_landsurface_forecast_benchmark import default_soil_file, observed_rootzone_target
from run_gefsv12_landsurface_benchmark import (
    DEFAULT_MEMBERS,
    RAW_DIR,
    score_benchmark,
    target_months_from_observed,
    build_forecast_rows,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = PROJECT_ROOT / "outputs"
REPORT_DIR = PROJECT_ROOT / "results" / "report" / "landsurface"

SOIL_FILE_RE = re.compile(
    r"era5_land_soil_moisture_monthly_(?P<region>[a-z0-9_]+)_(?P<start>\d{4})_(?P<end>\d{4})\.nc$"
)


def parse_args() -> Namespace:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--region", default="southern_great_plains")
    parser.add_argument(
        "--soil-file",
        type=Path,
        default=None,
        help=(
            "Optional ERA5-Land monthly soil moisture file. If omitted, picks the default for the "
            "region, falling back to the latest-matching file under data/processed/."
        ),
    )
    parser.add_argument("--lead-months", type=int, default=1)
    parser.add_argument("--start-target", default="2000-01")
    parser.add_argument("--end-target", default="2019-12")
    parser.add_argument("--validation-start-year", type=int, default=2000)
    parser.add_argument("--validation-end-year", type=int, default=2016)
    parser.add_argument("--test-start-year", type=int, default=2017)
    parser.add_argument("--test-end-year", type=int, default=2019)
    parser.add_argument("--normal-start-year", type=int, default=1991)
    parser.add_argument("--normal-end-year", type=int, default=2016)
    parser.add_argument("--dry-quantile", type=float, default=0.20)
    parser.add_argument(
        "--valid-days",
        nargs="+",
        type=int,
        default=[5, 10, 15, 20],
        help="Target-month days (1-31) to verify the GEFSv12 soil moisture state at 12 UTC.",
    )
    parser.add_argument(
        "--init-lag-weeks",
        nargs="+",
        type=int,
        default=[0, 1],
        help="Weekly long-init lag(s) in weeks relative to the latest long init before the target month starts.",
    )
    parser.add_argument(
        "--members",
        nargs="+",
        default=DEFAULT_MEMBERS,
        help="GEFSv12 members to use, e.g. c00 p01 ... p10.",
    )
    parser.add_argument("--min-members", type=int, default=6)
    parser.add_argument("--cache-dir", type=Path, default=RAW_DIR)
    parser.add_argument(
        "--out-prefix",
        default=None,
        help="Output filename prefix under outputs/ and, with --copy-report, results/report/landsurface/.",
    )
    parser.add_argument("--n-bootstrap", type=int, default=2000)
    parser.add_argument("--copy-report", action="store_true")
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--max-months", type=int, default=None)
    parser.add_argument("--progress-every", type=int, default=12)
    return parser.parse_args()


def resolve_soil_file(region_slug: str, soil_file: Path | None) -> Path:
    if soil_file is not None:
        return soil_file

    candidate = default_soil_file(region_slug)
    if candidate.exists():
        return candidate

    processed = PROJECT_ROOT / "data" / "processed"
    pattern = f"era5_land_soil_moisture_monthly_{region_slug}_*.nc"
    matches = list(processed.glob(pattern))
    if not matches:
        raise FileNotFoundError(
            f"No ERA5-Land soil moisture file found for region={region_slug}. "
            f"Tried {candidate} and {processed / pattern}"
        )

    scored: list[tuple[int, Path]] = []
    for path in matches:
        match = SOIL_FILE_RE.search(path.name)
        end_year = -1
        if match and match.group("region") == region_slug:
            end_year = int(match.group("end"))
        scored.append((end_year, path))

    scored.sort(key=lambda item: (item[0], item[1].name))
    return scored[-1][1]


def default_out_prefix(region_slug: str) -> str:
    region = resolve_region(region_slug)
    return f"landsurface_gefsv12_rzsm_{region.slug}_lead_validday_sensitivity"


def stable_unique_ints(values: list[int]) -> list[int]:
    out: list[int] = []
    seen: set[int] = set()
    for value in values:
        iv = int(value)
        if iv in seen:
            continue
        out.append(iv)
        seen.add(iv)
    return out


def overlap_counts(observed: pd.DataFrame, forecast: pd.DataFrame, args: Namespace) -> tuple[int, int]:
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
        return 0, 0
    val = merged[
        (merged["target_year"] >= args.validation_start_year)
        & (merged["target_year"] <= args.validation_end_year)
    ]
    test = merged[
        (merged["target_year"] >= args.test_start_year)
        & (merged["target_year"] <= args.test_end_year)
    ]
    return int(val["target_time"].nunique()), int(test["target_time"].nunique())


def summarize_combo(
    region: str,
    valid_day: int,
    init_lag_weeks: int,
    forecast: pd.DataFrame,
    skipped: pd.DataFrame,
    monthly: pd.DataFrame,
    scores: pd.DataFrame,
    val_months: int,
    test_months: int,
) -> dict[str, object]:
    out: dict[str, object] = {
        "region": region,
        "valid_day": int(valid_day),
        "init_lag_weeks": int(init_lag_weeks),
        "n_months_forecast": int(len(forecast)),
        "n_months_skipped": int(len(skipped)),
        "n_months_val": int(val_months),
        "n_months_test": int(test_months),
    }

    if monthly.empty or scores.empty:
        out["status"] = "no_scores"
        return out

    def row(name: str) -> pd.Series:
        hit = scores.loc[scores["forecast"].eq(name)]
        if hit.empty:
            raise KeyError(f"Missing score row: {name}")
        return hit.iloc[0]

    gefs = row("gefs_selected")
    pers = row("persistence_selected")

    out.update(
        {
            "gefs_selected_bs": float(gefs["bs"]),
            "gefs_selected_bss": float(gefs["bss"]),
            "gefs_selected_bss_ci_low": float(gefs["bss_ci_low"]),
            "gefs_selected_bss_ci_high": float(gefs["bss_ci_high"]),
            "gefs_selected_bss_vs_persistence_selected": float(gefs["bss_vs_persistence_selected"]),
            "persistence_selected_bs": float(pers["bs"]),
            "persistence_selected_bss": float(pers["bss"]),
            "delta_bs_gefs_minus_persistence_selected": float(gefs["bs"] - pers["bs"]),
        }
    )

    out["lead_days_mean_test"] = float(monthly["lead_days_to_valid"].mean())
    out["lead_days_min_test"] = float(monthly["lead_days_to_valid"].min())
    out["lead_days_max_test"] = float(monthly["lead_days_to_valid"].max())

    corr = monthly["gefs_selected_prob_dry"].corr(monthly["y_true_dry_frac"], method="spearman")
    out["spearman_gefs_selected_vs_observed_test"] = float(corr) if corr is not None else float("nan")

    y_std = float(monthly["y_true_dry_frac"].std(ddof=0))
    p_std = float(monthly["gefs_selected_prob_dry"].std(ddof=0))
    out["amplitude_ratio_gefs_selected_test"] = float(p_std / y_std) if y_std > 0 else float("nan")

    out["status"] = "ok"
    return out


def format_summary_text(summary: pd.DataFrame, args: Namespace) -> str:
    ok = summary.loc[summary["status"].eq("ok")].copy()
    lines = [
        "GEFSv12 Lead / Valid-Day Sensitivity",
        "=" * 72,
        f"Region: {resolve_region(args.region).name} ({args.region})",
        f"Target months: {args.start_target} to {args.end_target}",
        f"Validation years: {args.validation_start_year}-{args.validation_end_year}",
        f"Test years: {args.test_start_year}-{args.test_end_year}",
        f"Valid days: {sorted(stable_unique_ints(args.valid_days))}",
        f"Init lag weeks: {sorted(stable_unique_ints(args.init_lag_weeks))}",
        "",
    ]
    if ok.empty:
        lines.append("No successful combinations produced scores.")
        return "\n".join(lines) + "\n"

    best = ok.sort_values("gefs_selected_bss", ascending=False).iloc[0]
    worst = ok.sort_values("gefs_selected_bss", ascending=True).iloc[0]
    lines.extend(
        [
            "Headline (GEFSv12 selected BSS vs climatology):",
            f"  Best  : day={int(best['valid_day'])} lag={int(best['init_lag_weeks'])}w "
            f"BSS={best['gefs_selected_bss']:+.3f} "
            f"CI[{best['gefs_selected_bss_ci_low']:+.3f}, {best['gefs_selected_bss_ci_high']:+.3f}] "
            f"lead~{best['lead_days_mean_test']:.1f}d test_n={int(best['n_months_test'])}",
            f"  Worst : day={int(worst['valid_day'])} lag={int(worst['init_lag_weeks'])}w "
            f"BSS={worst['gefs_selected_bss']:+.3f} "
            f"CI[{worst['gefs_selected_bss_ci_low']:+.3f}, {worst['gefs_selected_bss_ci_high']:+.3f}] "
            f"lead~{worst['lead_days_mean_test']:.1f}d test_n={int(worst['n_months_test'])}",
            "",
            "Interpretation notes:",
            "  - Stability is suggested if BSS stays positive (or CI overlaps similar values) across most combos.",
            "  - If only one narrow combo is positive, treat it as a fragile/extraction-specific signal.",
            "",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    args.region = resolve_region(args.region).slug
    args.valid_days = stable_unique_ints([int(v) for v in args.valid_days])
    args.init_lag_weeks = stable_unique_ints([int(v) for v in args.init_lag_weeks])

    args.soil_file = resolve_soil_file(args.region, args.soil_file)
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    if args.out_prefix is None:
        args.out_prefix = default_out_prefix(args.region)

    out_summary = OUT_DIR / f"{args.out_prefix}_sensitivity_summary.csv"
    out_grid = OUT_DIR / f"{args.out_prefix}_sensitivity_grid.csv"
    out_monthly = OUT_DIR / f"{args.out_prefix}_sensitivity_monthly_scores.csv"
    out_text = OUT_DIR / f"{args.out_prefix}_sensitivity_summary.txt"
    total_combos = len(args.valid_days) * len(args.init_lag_weeks)

    # Observed target + climatology/persistence are invariant across combos.
    observed = observed_rootzone_target(args)
    months = target_months_from_observed(args, observed)
    print(
        f"Selected target months: {months.min():%Y-%m} to {months.max():%Y-%m} ({len(months)} months)",
        flush=True,
    )

    summary_rows: list[dict[str, object]] = []
    monthly_rows: list[pd.DataFrame] = []

    def checkpoint(done: int) -> None:
        summary = pd.DataFrame(summary_rows).sort_values(["valid_day", "init_lag_weeks"]).reset_index(drop=True)
        grid = (
            summary.loc[summary["status"].eq("ok"), ["valid_day", "init_lag_weeks", "gefs_selected_bss"]]
            .pivot(index="valid_day", columns="init_lag_weeks", values="gefs_selected_bss")
            .sort_index()
        )
        monthly_all = pd.concat(monthly_rows, ignore_index=True) if monthly_rows else pd.DataFrame()

        summary.to_csv(out_summary, index=False)
        grid.to_csv(out_grid)
        if not monthly_all.empty:
            monthly_all.to_csv(out_monthly, index=False)
        out_text.write_text(format_summary_text(summary, args), encoding="utf-8")

        n_ok = int(summary["status"].eq("ok").sum())
        n_err = int(len(summary) - n_ok)
        print(
            f"Checkpoint {done}/{total_combos}: wrote {out_summary.name} (ok={n_ok}, err={n_err})",
            flush=True,
        )

    for init_lag in sorted(args.init_lag_weeks):
        for valid_day in sorted(args.valid_days):
            combo = f"day{valid_day:02d}_lag{init_lag}w"
            print(f"\nRunning combo: {combo}", flush=True)
            combo_args = Namespace(**vars(args))
            combo_args.valid_day = int(valid_day)
            combo_args.init_lag_weeks = int(init_lag)

            try:
                forecast, skipped = build_forecast_rows(combo_args, months)
                monthly, scores, _ = score_benchmark(observed, forecast, combo_args)
                val_months, test_months = overlap_counts(observed, forecast, combo_args)
                record = summarize_combo(
                    combo_args.region,
                    combo_args.valid_day,
                    combo_args.init_lag_weeks,
                    forecast,
                    skipped,
                    monthly,
                    scores,
                    val_months,
                    test_months,
                )
                monthly = monthly.copy()
                monthly["valid_day"] = int(valid_day)
                monthly["init_lag_weeks"] = int(init_lag)
                monthly_rows.append(monthly)
            except Exception as exc:
                record = {
                    "region": combo_args.region,
                    "valid_day": int(valid_day),
                    "init_lag_weeks": int(init_lag),
                    "status": f"error:{type(exc).__name__}:{exc}",
                }
                forecast = pd.DataFrame()
                skipped = pd.DataFrame()

            summary_rows.append(record)
            checkpoint(len(summary_rows))

    print(f"\nWrote sensitivity summary: {out_summary} rows={len(summary_rows):,}")
    print(f"Wrote sensitivity grid: {out_grid}")
    if out_monthly.exists():
        print(f"Wrote sensitivity monthly scores: {out_monthly}")
    print(f"Wrote sensitivity notes: {out_text}")

    if args.copy_report:
        for path in [out_summary, out_grid, out_monthly, out_text]:
            if path.exists():
                shutil.copy2(path, REPORT_DIR / path.name)


if __name__ == "__main__":
    main()
