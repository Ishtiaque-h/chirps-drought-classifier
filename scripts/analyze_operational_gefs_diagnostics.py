#!/usr/bin/env python
"""Diagnose the failed operational GEFS land-surface smoke benchmark.

The operational GEFS 2021-2025 smoke run established archive feasibility, but
the validation-selected isotonic calibration failed badly on the 2024-2025
Central Valley test period. This script does not download new forecast data.
It reads a completed operational GEFS forecast CSV, reconstructs validation and
test predictions, and asks whether the failure is mainly:

  - poor forecast ranking,
  - unstable base-rate/calibration,
  - an unsafe validation selector,
  - or lack of added value over same-target persistence.

The key guardrail is a persistence-safe selector: a forecast candidate must
robustly improve validation Brier score over raw persistence before it can be
selected. Otherwise the selector falls back to raw persistence.
"""

from __future__ import annotations

from argparse import ArgumentParser, Namespace
from dataclasses import dataclass
from pathlib import Path
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


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = PROJECT_ROOT / "outputs"
REPORT_DIR = PROJECT_ROOT / "results" / "report" / "landsurface"


@dataclass(frozen=True)
class Candidate:
    name: str
    val_pred: pd.Series
    test_pred: pd.Series
    formula: str
    selected_weight: float | None = None


def parse_args() -> Namespace:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--region", default="cvalley")
    parser.add_argument(
        "--forecast-csv",
        type=Path,
        default=None,
        help="Completed operational GEFS forecast CSV. Defaults to the 2021-2025 Central Valley 11-member run.",
    )
    parser.add_argument("--soil-file", type=Path, default=None)
    parser.add_argument("--lead-months", type=int, default=1)
    parser.add_argument("--normal-start-year", type=int, default=1991)
    parser.add_argument("--normal-end-year", type=int, default=2020)
    parser.add_argument("--dry-quantile", type=float, default=0.20)
    parser.add_argument("--validation-start-year", type=int, default=2021)
    parser.add_argument("--validation-end-year", type=int, default=2023)
    parser.add_argument("--test-start-year", type=int, default=2024)
    parser.add_argument("--test-end-year", type=int, default=2025)
    parser.add_argument("--n-bootstrap", type=int, default=2000)
    parser.add_argument("--out-prefix", default="landsurface_operational_gefs_subroot_cvalley_2021_2025_diagnostics")
    parser.add_argument("--copy-report", action="store_true")
    return parser.parse_args()


def default_forecast_csv(region_slug: str) -> Path:
    return (
        REPORT_DIR
        / f"landsurface_operational_gefs_subroot_{region_slug}_2021_2025_11member_forecast.csv"
    )


def bootstrap_delta_bs(
    monthly: pd.DataFrame,
    pred_col: str,
    ref_col: str,
    n_bootstrap: int,
    seed: int = 42,
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    idx = np.arange(len(monthly))
    y = monthly["y_true_dry_frac"].to_numpy(dtype=float)
    p = monthly[pred_col].to_numpy(dtype=float)
    ref = monthly[ref_col].to_numpy(dtype=float)
    vals = np.empty(n_bootstrap, dtype=float)
    for i in range(n_bootstrap):
        sample = rng.choice(idx, size=len(idx), replace=True)
        vals[i] = brier(y[sample], p[sample]) - brier(y[sample], ref[sample])
    lo, hi = np.nanquantile(vals, [0.025, 0.975])
    return float(lo), float(hi)


def merge_observed_forecast(args: Namespace) -> pd.DataFrame:
    if args.forecast_csv is None:
        args.forecast_csv = default_forecast_csv(args.region)
    if args.soil_file is None:
        args.soil_file = default_soil_file(args.region)
    if not args.forecast_csv.exists():
        raise FileNotFoundError(f"Operational GEFS forecast CSV not found: {args.forecast_csv}")

    observed_args = Namespace(
        soil_file=args.soil_file,
        lead_months=args.lead_months,
        normal_start_year=args.normal_start_year,
        normal_end_year=args.normal_end_year,
        dry_quantile=args.dry_quantile,
    )
    observed = observed_rootzone_target(observed_args)
    forecast = pd.read_csv(
        args.forecast_csv,
        parse_dates=["target_time", "valid_time", "init_time"],
    )
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
    return merged


def choose_weighted_candidate(
    name: str,
    val_a: pd.Series,
    test_a: pd.Series,
    val_b: pd.Series,
    test_b: pd.Series,
    y_val: pd.Series,
    formula: str,
    weights: np.ndarray | None = None,
) -> Candidate:
    if weights is None:
        weights = np.linspace(0.0, 1.0, 21)
    rows = []
    for weight in weights:
        pred = weight * val_a + (1.0 - weight) * val_b
        rows.append((float(weight), brier(y_val, pred)))
    selected_weight, _ = min(rows, key=lambda item: item[1])
    return Candidate(
        name=name,
        val_pred=selected_weight * val_a + (1.0 - selected_weight) * val_b,
        test_pred=selected_weight * test_a + (1.0 - selected_weight) * test_b,
        formula=formula,
        selected_weight=float(selected_weight),
    )


def build_candidates(val: pd.DataFrame, test: pd.DataFrame) -> list[Candidate]:
    val_raw_iso, test_raw_iso = fit_isotonic(val, test, "gefs_raw_dry_signal")
    val_anom_iso, test_anom_iso = fit_isotonic(val, test, "gefs_anom_dry_signal")
    val_pers_iso, test_pers_iso = fit_isotonic(val, test, "persistence_raw_prob_dry")

    candidates: list[Candidate] = [
        Candidate(
            "monthly_climatology",
            val["clim_prob_dry"].astype(float),
            test["clim_prob_dry"].astype(float),
            "Calendar-month climatology from observed normal period.",
        ),
        Candidate(
            "persistence_raw",
            val["persistence_raw_prob_dry"].astype(float),
            test["persistence_raw_prob_dry"].astype(float),
            "Previous-month observed dry fraction.",
        ),
        Candidate(
            "persistence_isotonic",
            val_pers_iso,
            test_pers_iso,
            "Validation-fitted isotonic mapping of previous-month dry fraction.",
        ),
        Candidate(
            "operational_gefs_raw_isotonic",
            val_raw_iso,
            test_raw_iso,
            "Validation-fitted isotonic mapping of negative operational GEFS soil-moisture signal.",
        ),
        Candidate(
            "operational_gefs_anom_isotonic",
            val_anom_iso,
            test_anom_iso,
            "Validation-fitted isotonic mapping of negative operational GEFS soil-moisture anomaly.",
        ),
    ]

    candidates.extend(
        [
            choose_weighted_candidate(
                "operational_gefs_raw_climatology_shrink",
                val_raw_iso,
                test_raw_iso,
                val["clim_prob_dry"].astype(float),
                test["clim_prob_dry"].astype(float),
                val["y_true_dry_frac"],
                "w * GEFS raw isotonic + (1-w) * climatology; w selected on validation.",
            ),
            choose_weighted_candidate(
                "operational_gefs_anom_climatology_shrink",
                val_anom_iso,
                test_anom_iso,
                val["clim_prob_dry"].astype(float),
                test["clim_prob_dry"].astype(float),
                val["y_true_dry_frac"],
                "w * GEFS anomaly isotonic + (1-w) * climatology; w selected on validation.",
            ),
            choose_weighted_candidate(
                "persistence_isotonic_climatology_shrink",
                val_pers_iso,
                test_pers_iso,
                val["clim_prob_dry"].astype(float),
                test["clim_prob_dry"].astype(float),
                val["y_true_dry_frac"],
                "w * persistence isotonic + (1-w) * climatology; w selected on validation.",
            ),
            choose_weighted_candidate(
                "stack_gefs_raw_persistence_raw",
                val_raw_iso,
                test_raw_iso,
                val["persistence_raw_prob_dry"].astype(float),
                test["persistence_raw_prob_dry"].astype(float),
                val["y_true_dry_frac"],
                "w * GEFS raw isotonic + (1-w) * raw persistence; w selected on validation.",
            ),
            choose_weighted_candidate(
                "stack_gefs_anom_persistence_raw",
                val_anom_iso,
                test_anom_iso,
                val["persistence_raw_prob_dry"].astype(float),
                test["persistence_raw_prob_dry"].astype(float),
                val["y_true_dry_frac"],
                "w * GEFS anomaly isotonic + (1-w) * raw persistence; w selected on validation.",
            ),
        ]
    )
    return candidates


def candidate_predictions_frame(
    base: pd.DataFrame,
    candidates: list[Candidate],
    split: str,
) -> pd.DataFrame:
    out = base[
        [
            "target_time",
            "target_year",
            "target_month",
            "y_true_dry_frac",
            "clim_prob_dry",
            "persistence_raw_prob_dry",
            "forecast_subroot_sm",
            "forecast_subroot_anom",
            "forecast_subroot_member_std",
        ]
    ].copy()
    out["split"] = split
    for candidate in candidates:
        pred = candidate.val_pred if split == "validation" else candidate.test_pred
        out[candidate.name] = np.asarray(pred, dtype=float)
    return out


def score_candidates(
    val_pred: pd.DataFrame,
    test_pred: pd.DataFrame,
    candidates: list[Candidate],
    n_bootstrap: int,
) -> tuple[pd.DataFrame, str, str]:
    rows = []
    val_tmp = val_pred.copy()
    test_tmp = test_pred.copy()

    for candidate in candidates:
        val_bs = brier(val_tmp["y_true_dry_frac"], val_tmp[candidate.name])
        test_bs = brier(test_tmp["y_true_dry_frac"], test_tmp[candidate.name])
        val_bss = bss(val_tmp["y_true_dry_frac"], val_tmp[candidate.name], val_tmp["clim_prob_dry"])
        test_bss = bss(test_tmp["y_true_dry_frac"], test_tmp[candidate.name], test_tmp["clim_prob_dry"])
        if candidate.name == "monthly_climatology":
            test_ci_low = test_ci_high = 0.0
        else:
            test_ci_low, test_ci_high = bootstrap_bss(
                test_tmp,
                pred_col=candidate.name,
                ref_col="clim_prob_dry",
                n_bootstrap=n_bootstrap,
                seed=930,
            )

        val_delta = val_bs - brier(val_tmp["y_true_dry_frac"], val_tmp["persistence_raw"])
        test_delta = test_bs - brier(test_tmp["y_true_dry_frac"], test_tmp["persistence_raw"])
        val_delta_low, val_delta_high = bootstrap_delta_bs(
            val_tmp,
            candidate.name,
            "persistence_raw",
            n_bootstrap=n_bootstrap,
            seed=931,
        )
        test_delta_low, test_delta_high = bootstrap_delta_bs(
            test_tmp,
            candidate.name,
            "persistence_raw",
            n_bootstrap=n_bootstrap,
            seed=932,
        )
        rows.append(
            {
                "candidate": candidate.name,
                "formula": candidate.formula,
                "selected_weight": candidate.selected_weight,
                "validation_brier_score": val_bs,
                "validation_bss_vs_climatology": val_bss,
                "validation_delta_bs_vs_raw_persistence": val_delta,
                "validation_delta_ci_low": val_delta_low,
                "validation_delta_ci_high": val_delta_high,
                "validation_robustly_beats_raw_persistence": bool(val_delta_high < 0.0),
                "test_brier_score": test_bs,
                "test_bss_vs_climatology": test_bss,
                "test_bss_ci_low": test_ci_low,
                "test_bss_ci_high": test_ci_high,
                "test_delta_bs_vs_raw_persistence": test_delta,
                "test_delta_ci_low": test_delta_low,
                "test_delta_ci_high": test_delta_high,
                "test_robustly_beats_raw_persistence": bool(test_delta_high < 0.0),
                "validation_mean_prediction": float(np.nanmean(val_tmp[candidate.name])),
                "test_mean_prediction": float(np.nanmean(test_tmp[candidate.name])),
                "test_spearman_y_pred": float(
                    pd.Series(test_tmp["y_true_dry_frac"]).corr(test_tmp[candidate.name], method="spearman")
                ),
            }
        )
    scores = pd.DataFrame(rows)

    validation_selected = str(scores.sort_values("validation_brier_score").iloc[0]["candidate"])
    safe_pool = scores.loc[
        scores["validation_robustly_beats_raw_persistence"]
        & ~scores["candidate"].eq("monthly_climatology")
    ].copy()
    if safe_pool.empty:
        persistence_safe_selected = "persistence_raw"
    else:
        persistence_safe_selected = str(safe_pool.sort_values("validation_brier_score").iloc[0]["candidate"])
    scores["validation_selected_any"] = scores["candidate"].eq(validation_selected)
    scores["persistence_safe_selected"] = scores["candidate"].eq(persistence_safe_selected)
    return scores.sort_values("validation_brier_score").reset_index(drop=True), validation_selected, persistence_safe_selected


def build_yearly_summary(predictions: pd.DataFrame, selected_any: str, selected_safe: str) -> pd.DataFrame:
    cols = [
        "y_true_dry_frac",
        "clim_prob_dry",
        "persistence_raw",
        selected_any,
        selected_safe,
        "forecast_subroot_sm",
        "forecast_subroot_anom",
    ]
    existing = [col for col in cols if col in predictions.columns]
    existing = list(dict.fromkeys(existing))
    grouped = predictions.groupby(["split", "target_year"], sort=True)[existing].agg(["mean", "std"])
    grouped.columns = ["_".join([str(x) for x in col if x]) for col in grouped.columns]
    return grouped.reset_index()


def build_shift_summary(val_pred: pd.DataFrame, test_pred: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for split, df in [("validation", val_pred), ("test", test_pred)]:
        rows.append(
            {
                "split": split,
                "n_months": int(len(df)),
                "mean_y_true_dry_frac": float(df["y_true_dry_frac"].mean()),
                "std_y_true_dry_frac": float(df["y_true_dry_frac"].std(ddof=0)),
                "mean_climatology": float(df["clim_prob_dry"].mean()),
                "mean_persistence_raw": float(df["persistence_raw"].mean()),
                "mean_forecast_subroot_sm": float(df["forecast_subroot_sm"].mean()),
                "std_forecast_subroot_sm": float(df["forecast_subroot_sm"].std(ddof=0)),
                "mean_forecast_subroot_anom": float(df["forecast_subroot_anom"].mean()),
                "spearman_y_forecast_sm": float(
                    pd.Series(df["y_true_dry_frac"]).corr(df["forecast_subroot_sm"], method="spearman")
                ),
                "spearman_y_forecast_anom": float(
                    pd.Series(df["y_true_dry_frac"]).corr(df["forecast_subroot_anom"], method="spearman")
                ),
                "spearman_y_persistence_raw": float(
                    pd.Series(df["y_true_dry_frac"]).corr(df["persistence_raw"], method="spearman")
                ),
            }
        )
    return pd.DataFrame(rows)


def write_markdown(
    scores: pd.DataFrame,
    shift: pd.DataFrame,
    validation_selected: str,
    safe_selected: str,
    path: Path,
) -> None:
    selected_row = scores.loc[scores["candidate"].eq(validation_selected)].iloc[0]
    safe_row = scores.loc[scores["candidate"].eq(safe_selected)].iloc[0]
    raw_persistence = scores.loc[scores["candidate"].eq("persistence_raw")].iloc[0]
    shift_cols = [
        "split",
        "n_months",
        "mean_y_true_dry_frac",
        "mean_climatology",
        "mean_persistence_raw",
        "mean_forecast_subroot_sm",
        "mean_forecast_subroot_anom",
        "spearman_y_forecast_anom",
        "spearman_y_persistence_raw",
    ]
    shift_small = shift[[col for col in shift_cols if col in shift.columns]].copy()
    shift_table = ["| " + " | ".join(shift_small.columns) + " |"]
    shift_table.append("| " + " | ".join(["---"] * len(shift_small.columns)) + " |")
    for _, row in shift_small.iterrows():
        values = []
        for value in row:
            if isinstance(value, float):
                values.append(f"{value:.3f}")
            else:
                values.append(str(value))
        shift_table.append("| " + " | ".join(values) + " |")
    lines = [
        "# Operational GEFS Land-Surface Diagnostic",
        "",
        "This diagnostic explains the failed Central Valley 2021-2025 operational GEFS smoke test.",
        "",
        "## Selection Result",
        "",
        f"- Validation-selected candidate: `{validation_selected}`.",
        f"- Persistence-safe selected candidate: `{safe_selected}`.",
        f"- Validation-selected test BSS: `{float(selected_row['test_bss_vs_climatology']):+.3f}` "
        f"(CI `{float(selected_row['test_bss_ci_low']):+.3f}` to `{float(selected_row['test_bss_ci_high']):+.3f}`).",
        f"- Persistence-safe test BSS: `{float(safe_row['test_bss_vs_climatology']):+.3f}` "
        f"(CI `{float(safe_row['test_bss_ci_low']):+.3f}` to `{float(safe_row['test_bss_ci_high']):+.3f}`).",
        f"- Raw persistence test BSS: `{float(raw_persistence['test_bss_vs_climatology']):+.3f}` "
        f"(CI `{float(raw_persistence['test_bss_ci_low']):+.3f}` to `{float(raw_persistence['test_bss_ci_high']):+.3f}`).",
        "",
        "## Interpretation",
        "",
        "The operational GEFS archive is accessible, but validation-only isotonic calibration is unstable here. "
        "No candidate robustly improves validation Brier score over raw persistence, so the persistence-safe selector "
        "falls back to raw persistence. That means the current operational-GEFS path should not be scaled as a positive "
        "modern benchmark until depth compatibility and base-rate/calibration transfer are fixed.",
        "",
        "## Shift Summary",
        "",
        "\n".join(shift_table),
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    region = resolve_region(args.region)
    args.region = region.slug
    if args.forecast_csv is None:
        args.forecast_csv = default_forecast_csv(region.slug)
    if args.soil_file is None:
        args.soil_file = default_soil_file(region.slug)

    merged = merge_observed_forecast(args)
    val = merged[
        (merged["target_year"] >= args.validation_start_year)
        & (merged["target_year"] <= args.validation_end_year)
    ].copy()
    test = merged[
        (merged["target_year"] >= args.test_start_year)
        & (merged["target_year"] <= args.test_end_year)
    ].copy()
    if val.empty:
        raise ValueError("No validation rows selected.")
    if test.empty:
        raise ValueError("No test rows selected.")

    candidates = build_candidates(val, test)
    val_pred = candidate_predictions_frame(val, candidates, "validation")
    test_pred = candidate_predictions_frame(test, candidates, "test")
    scores, validation_selected, safe_selected = score_candidates(
        val_pred,
        test_pred,
        candidates,
        n_bootstrap=args.n_bootstrap,
    )
    predictions = pd.concat([val_pred, test_pred], ignore_index=True)
    yearly = build_yearly_summary(predictions, validation_selected, safe_selected)
    shift = build_shift_summary(val_pred, test_pred)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    scores_path = OUT_DIR / f"{args.out_prefix}_candidate_scores.csv"
    predictions_path = OUT_DIR / f"{args.out_prefix}_monthly_predictions.csv"
    yearly_path = OUT_DIR / f"{args.out_prefix}_yearly_summary.csv"
    shift_path = OUT_DIR / f"{args.out_prefix}_shift_summary.csv"
    md_path = OUT_DIR / f"{args.out_prefix}.md"

    scores.to_csv(scores_path, index=False)
    predictions.to_csv(predictions_path, index=False)
    yearly.to_csv(yearly_path, index=False)
    shift.to_csv(shift_path, index=False)
    write_markdown(scores, shift, validation_selected, safe_selected, md_path)

    if args.copy_report:
        for path in [scores_path, predictions_path, yearly_path, shift_path, md_path]:
            shutil.copy2(path, REPORT_DIR / path.name)

    print(f"Validation-selected candidate: {validation_selected}")
    print(f"Persistence-safe selected candidate: {safe_selected}")
    print("")
    print(scores[[
        "candidate",
        "selected_weight",
        "validation_brier_score",
        "validation_delta_bs_vs_raw_persistence",
        "validation_delta_ci_low",
        "validation_delta_ci_high",
        "validation_robustly_beats_raw_persistence",
        "test_bss_vs_climatology",
        "test_bss_ci_low",
        "test_bss_ci_high",
        "test_delta_bs_vs_raw_persistence",
    ]].to_string(index=False))
    print("")
    print(f"Wrote {scores_path.relative_to(PROJECT_ROOT)}")
    print(f"Wrote {md_path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
