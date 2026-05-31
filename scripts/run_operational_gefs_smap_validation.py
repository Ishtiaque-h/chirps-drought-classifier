#!/usr/bin/env python
"""Score operational GEFS soil-moisture forecasts against modern SMAP L4 targets.

This is a target-product validation script for the operational GEFS smoke path.
It reads an already-built operational GEFS forecast CSV and a processed SMAP L4
monthly dry-fraction target, then evaluates leakage-safe validation-only
calibration on a frozen modern test period.

The default Central Valley protocol matches the operational-GEFS smoke test:
validation = 2021-2023 and test = 2024-2025. Results are short-record
satellite-assimilated validation, not deployment proof.
"""

from __future__ import annotations

from argparse import ArgumentParser, Namespace
from pathlib import Path
import shutil

import numpy as np
import pandas as pd

from run_landsurface_forecast_benchmark import brier, bss, bootstrap_bss, fit_isotonic


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = PROJECT_ROOT / "outputs"
REPORT_DIR = PROJECT_ROOT / "results" / "report" / "landsurface"


def parse_args() -> Namespace:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--region", default="cvalley")
    parser.add_argument(
        "--target-csv",
        type=Path,
        default=REPORT_DIR / "landsurface_smap_l4_modern_multi3snapshot_cvalley_target_monthly_targets.csv",
    )
    parser.add_argument(
        "--forecast-csv",
        type=Path,
        default=REPORT_DIR / "landsurface_operational_gefs_soill_0_1m_cvalley_2021_2025_11member_forecast.csv",
    )
    parser.add_argument("--validation-start-year", type=int, default=2021)
    parser.add_argument("--validation-end-year", type=int, default=2023)
    parser.add_argument("--test-start-year", type=int, default=2024)
    parser.add_argument("--test-end-year", type=int, default=2025)
    parser.add_argument(
        "--climatology-mode",
        choices=["constant", "monthly"],
        default="constant",
        help="Use constant validation mean by default because the modern SMAP calibration period is short.",
    )
    parser.add_argument("--weight-steps", type=int, default=101)
    parser.add_argument(
        "--min-yearly-improvement",
        type=float,
        default=0.001,
        help=(
            "Minimum held-out validation-year Brier improvement over raw persistence "
            "required in every validation year before the persistence guard uses "
            "the year-held-out GEFS/persistence stack."
        ),
    )
    parser.add_argument("--n-bootstrap", type=int, default=1000)
    parser.add_argument("--out-prefix", default="landsurface_operational_gefs_soill_0_1m_cvalley_smap_validation")
    parser.add_argument("--copy-report", action="store_true")
    return parser.parse_args()


def bootstrap_delta_bs(
    monthly: pd.DataFrame,
    pred_col: str,
    ref_col: str,
    n_bootstrap: int,
    seed: int,
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    idx = np.arange(len(monthly))
    y = monthly["y_true_dry_frac"].to_numpy(dtype=float)
    pred = monthly[pred_col].to_numpy(dtype=float)
    ref = monthly[ref_col].to_numpy(dtype=float)
    vals = np.empty(n_bootstrap, dtype=float)
    for i in range(n_bootstrap):
        sample = rng.choice(idx, size=len(idx), replace=True)
        vals[i] = brier(y[sample], pred[sample]) - brier(y[sample], ref[sample])
    lo, hi = np.nanquantile(vals, [0.025, 0.975])
    return float(lo), float(hi)


def choose_weight(
    val: pd.DataFrame,
    a_col: str,
    b_col: str,
    weight_steps: int,
) -> tuple[float, float]:
    weights = np.linspace(0.0, 1.0, int(weight_steps))
    rows = []
    y = val["y_true_dry_frac"].to_numpy(dtype=float)
    a = val[a_col].to_numpy(dtype=float)
    b = val[b_col].to_numpy(dtype=float)
    for weight in weights:
        pred = weight * a + (1.0 - weight) * b
        rows.append((float(weight), brier(y, pred)))
    return min(rows, key=lambda item: item[1])


def year_holdout_predictions(val: pd.DataFrame) -> pd.DataFrame:
    parts = []
    years = sorted(int(year) for year in val["target_year"].dropna().unique())
    for year in years:
        train = val.loc[val["target_year"].ne(year)].copy()
        holdout = val.loc[val["target_year"].eq(year)].copy()
        if train.empty or holdout.empty:
            continue
        _, raw = fit_isotonic(train, holdout, "gefs_raw_dry_signal")
        _, anom = fit_isotonic(train, holdout, "gefs_anom_dry_signal")
        _, pers_iso = fit_isotonic(train, holdout, "persistence_raw_prob_dry")
        out = holdout[
            [
                "target_time",
                "target_year",
                "y_true_dry_frac",
                "clim_prob_dry",
                "persistence_raw_prob_dry",
            ]
        ].copy()
        out["operational_gefs_raw_isotonic_prob_dry"] = raw
        out["operational_gefs_anom_isotonic_prob_dry"] = anom
        out["persistence_isotonic_prob_dry"] = pers_iso
        parts.append(out)
    if not parts:
        raise ValueError("Could not build year-held-out validation predictions.")
    return pd.concat(parts, ignore_index=True).sort_values("target_time").reset_index(drop=True)


def select_year_holdout_stack(
    cv: pd.DataFrame,
    weight_steps: int,
    min_yearly_improvement: float,
) -> dict[str, object]:
    cv_bs = {
        "operational_gefs_raw_isotonic": brier(
            cv["y_true_dry_frac"], cv["operational_gefs_raw_isotonic_prob_dry"]
        ),
        "operational_gefs_anom_isotonic": brier(
            cv["y_true_dry_frac"], cv["operational_gefs_anom_isotonic_prob_dry"]
        ),
        "persistence_raw": brier(cv["y_true_dry_frac"], cv["persistence_raw_prob_dry"]),
        "persistence_isotonic": brier(cv["y_true_dry_frac"], cv["persistence_isotonic_prob_dry"]),
        "monthly_climatology": brier(cv["y_true_dry_frac"], cv["clim_prob_dry"]),
    }
    cv_gefs_best = min(
        {k: v for k, v in cv_bs.items() if k.startswith("operational_gefs_")},
        key=cv_bs.get,
    )
    cv_persistence_best = min(
        {k: v for k, v in cv_bs.items() if k.startswith("persistence_")},
        key=cv_bs.get,
    )
    cv_gefs_col = (
        "operational_gefs_raw_isotonic_prob_dry"
        if cv_gefs_best == "operational_gefs_raw_isotonic"
        else "operational_gefs_anom_isotonic_prob_dry"
    )
    cv_persistence_col = (
        "persistence_raw_prob_dry"
        if cv_persistence_best == "persistence_raw"
        else "persistence_isotonic_prob_dry"
    )
    best_weight, best_bs = choose_weight(cv, cv_gefs_col, cv_persistence_col, weight_steps)
    cv = cv.copy()
    cv["cv_stack_selected_prob_dry"] = (
        best_weight * cv[cv_gefs_col] + (1.0 - best_weight) * cv[cv_persistence_col]
    ).clip(0.0, 1.0)
    yearly_delta = {}
    for year, group in cv.groupby("target_year"):
        yearly_delta[int(year)] = brier(
            group["y_true_dry_frac"], group["cv_stack_selected_prob_dry"]
        ) - brier(group["y_true_dry_frac"], group["persistence_raw_prob_dry"])
    passes_guard = all(delta <= -float(min_yearly_improvement) for delta in yearly_delta.values())
    return {
        "cv_brier_scores": cv_bs,
        "cv_gefs_best": cv_gefs_best,
        "cv_gefs_col": cv_gefs_col,
        "cv_persistence_best": cv_persistence_best,
        "cv_persistence_col": cv_persistence_col,
        "cv_stack_weight": float(best_weight),
        "cv_stack_brier_score": float(best_bs),
        "cv_yearly_delta_bs_vs_raw_persistence": yearly_delta,
        "cv_min_yearly_delta_bs_vs_raw_persistence": float(min(yearly_delta.values())),
        "cv_max_yearly_delta_bs_vs_raw_persistence": float(max(yearly_delta.values())),
        "passes_persistence_guard": bool(passes_guard),
    }


def load_merged(args: Namespace) -> pd.DataFrame:
    if not args.target_csv.exists():
        raise FileNotFoundError(f"SMAP target CSV not found: {args.target_csv}")
    if not args.forecast_csv.exists():
        raise FileNotFoundError(f"Operational GEFS forecast CSV not found: {args.forecast_csv}")

    target = pd.read_csv(args.target_csv, parse_dates=["target_time", "persistence_time"])
    forecast = pd.read_csv(args.forecast_csv, parse_dates=["target_time", "valid_time", "init_time"])
    if "region" in target.columns:
        target = target.loc[target["region"].astype(str).eq(args.region)].copy()
    merged = target.merge(forecast, on="target_time", how="inner", suffixes=("", "_forecast"))
    merged = merged.dropna(
        subset=[
            "y_true_dry_frac",
            "persistence_raw_prob_dry",
            "forecast_subroot_sm",
            "forecast_subroot_anom",
        ]
    ).copy()
    if merged.empty:
        raise ValueError("No overlap between SMAP target and operational GEFS forecast rows.")

    val_mask = (merged["target_year"] >= args.validation_start_year) & (
        merged["target_year"] <= args.validation_end_year
    )
    val = merged.loc[val_mask].copy()
    if val.empty:
        raise ValueError("No validation rows after merging SMAP and operational GEFS.")
    if args.climatology_mode == "monthly":
        month_clim = val.groupby("target_month")["y_true_dry_frac"].mean()
        global_clim = float(val["y_true_dry_frac"].mean())
        merged["clim_prob_dry"] = merged["target_month"].map(month_clim).fillna(global_clim)
    else:
        merged["clim_prob_dry"] = float(val["y_true_dry_frac"].mean())

    merged["gefs_raw_dry_signal"] = -merged["forecast_subroot_sm"].astype(float)
    merged["gefs_anom_dry_signal"] = -merged["forecast_subroot_anom"].astype(float)
    return merged


def score(args: Namespace) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    merged = load_merged(args)
    val = merged[
        (merged["target_year"] >= args.validation_start_year)
        & (merged["target_year"] <= args.validation_end_year)
    ].copy()
    test = merged[
        (merged["target_year"] >= args.test_start_year)
        & (merged["target_year"] <= args.test_end_year)
    ].copy()
    if test.empty:
        raise ValueError("No test rows after merging SMAP and operational GEFS.")

    val_raw, test_raw = fit_isotonic(val, test, "gefs_raw_dry_signal")
    val_anom, test_anom = fit_isotonic(val, test, "gefs_anom_dry_signal")
    val_pers_iso, test_pers_iso = fit_isotonic(val, test, "persistence_raw_prob_dry")
    val = val.copy()
    test = test.copy()
    val["operational_gefs_raw_isotonic_prob_dry"] = val_raw
    test["operational_gefs_raw_isotonic_prob_dry"] = test_raw
    val["operational_gefs_anom_isotonic_prob_dry"] = val_anom
    test["operational_gefs_anom_isotonic_prob_dry"] = test_anom
    val["persistence_isotonic_prob_dry"] = val_pers_iso
    test["persistence_isotonic_prob_dry"] = test_pers_iso

    val_bs = {
        "operational_gefs_raw_isotonic": brier(val["y_true_dry_frac"], val["operational_gefs_raw_isotonic_prob_dry"]),
        "operational_gefs_anom_isotonic": brier(val["y_true_dry_frac"], val["operational_gefs_anom_isotonic_prob_dry"]),
        "persistence_raw": brier(val["y_true_dry_frac"], val["persistence_raw_prob_dry"]),
        "persistence_isotonic": brier(val["y_true_dry_frac"], val["persistence_isotonic_prob_dry"]),
    }
    gefs_best = min({k: v for k, v in val_bs.items() if k.startswith("operational_gefs_")}, key=val_bs.get)
    persistence_best = min({k: v for k, v in val_bs.items() if k.startswith("persistence_")}, key=val_bs.get)
    gefs_col = (
        "operational_gefs_raw_isotonic_prob_dry"
        if gefs_best == "operational_gefs_raw_isotonic"
        else "operational_gefs_anom_isotonic_prob_dry"
    )
    persistence_col = (
        "persistence_raw_prob_dry"
        if persistence_best == "persistence_raw"
        else "persistence_isotonic_prob_dry"
    )
    for frame in [val, test]:
        frame["operational_gefs_selected_prob_dry"] = frame[gefs_col]
        frame["persistence_selected_prob_dry"] = frame[persistence_col]

    weight, weight_val_bs = choose_weight(
        val,
        "operational_gefs_selected_prob_dry",
        "persistence_selected_prob_dry",
        args.weight_steps,
    )
    for frame in [val, test]:
        frame["stack_validation_selected_prob_dry"] = (
            weight * frame["operational_gefs_selected_prob_dry"]
            + (1.0 - weight) * frame["persistence_selected_prob_dry"]
        ).clip(0.0, 1.0)

    cv = year_holdout_predictions(val)
    cv_meta = select_year_holdout_stack(
        cv,
        weight_steps=args.weight_steps,
        min_yearly_improvement=args.min_yearly_improvement,
    )
    for frame in [val, test]:
        frame["cv_operational_gefs_selected_prob_dry"] = frame[str(cv_meta["cv_gefs_col"])]
        frame["cv_persistence_selected_prob_dry"] = frame[str(cv_meta["cv_persistence_col"])]
        frame["cv_stack_selected_prob_dry"] = (
            float(cv_meta["cv_stack_weight"]) * frame["cv_operational_gefs_selected_prob_dry"]
            + (1.0 - float(cv_meta["cv_stack_weight"])) * frame["cv_persistence_selected_prob_dry"]
        ).clip(0.0, 1.0)
        frame["persistence_guard_selected_prob_dry"] = (
            frame["cv_stack_selected_prob_dry"]
            if bool(cv_meta["passes_persistence_guard"])
            else frame["persistence_raw_prob_dry"]
        )

    monthly = pd.concat(
        [val.assign(split="validation"), test.assign(split="test")],
        ignore_index=True,
    )
    model_cols = {
        "operational_gefs_raw_isotonic": "operational_gefs_raw_isotonic_prob_dry",
        "operational_gefs_anom_isotonic": "operational_gefs_anom_isotonic_prob_dry",
        "operational_gefs_selected": "operational_gefs_selected_prob_dry",
        "persistence_raw": "persistence_raw_prob_dry",
        "persistence_isotonic": "persistence_isotonic_prob_dry",
        "persistence_selected": "persistence_selected_prob_dry",
        "stack_validation_selected": "stack_validation_selected_prob_dry",
        "cv_operational_gefs_selected": "cv_operational_gefs_selected_prob_dry",
        "cv_persistence_selected": "cv_persistence_selected_prob_dry",
        "cv_stack_selected": "cv_stack_selected_prob_dry",
        "persistence_guard_selected": "persistence_guard_selected_prob_dry",
        "monthly_climatology": "clim_prob_dry",
    }

    rows = []
    for i, (model, col) in enumerate(model_cols.items()):
        bs_model = brier(test["y_true_dry_frac"], test[col])
        bs_clim = brier(test["y_true_dry_frac"], test["clim_prob_dry"])
        if model == "monthly_climatology":
            lo = hi = 0.0
        else:
            lo, hi = bootstrap_bss(
                test,
                pred_col=col,
                ref_col="clim_prob_dry",
                n_bootstrap=args.n_bootstrap,
                seed=940 + i,
            )
        delta_low, delta_high = bootstrap_delta_bs(
            test,
            pred_col=col,
            ref_col="persistence_raw_prob_dry",
            n_bootstrap=args.n_bootstrap,
            seed=950 + i,
        )
        if model == "stack_validation_selected":
            selected_weight_gefs = weight
            validation_score = weight_val_bs
            selection_protocol = "in_sample_validation_stack"
        elif model == "cv_stack_selected":
            selected_weight_gefs = float(cv_meta["cv_stack_weight"])
            validation_score = float(cv_meta["cv_stack_brier_score"])
            selection_protocol = "year_held_out_validation_stack"
        elif model.startswith("cv_"):
            selected_weight_gefs = np.nan
            if model == "cv_operational_gefs_selected":
                validation_score = dict(cv_meta["cv_brier_scores"]).get(str(cv_meta["cv_gefs_best"]), np.nan)
            elif model == "cv_persistence_selected":
                validation_score = dict(cv_meta["cv_brier_scores"]).get(
                    str(cv_meta["cv_persistence_best"]), np.nan
                )
            else:
                validation_score = np.nan
            selection_protocol = "year_held_out_validation"
        elif model == "persistence_guard_selected":
            selected_weight_gefs = (
                float(cv_meta["cv_stack_weight"]) if bool(cv_meta["passes_persistence_guard"]) else 0.0
            )
            validation_score = (
                float(cv_meta["cv_stack_brier_score"])
                if bool(cv_meta["passes_persistence_guard"])
                else dict(cv_meta["cv_brier_scores"]).get("persistence_raw", np.nan)
            )
            selection_protocol = "year_held_out_persistence_guard"
        else:
            selected_weight_gefs = np.nan
            validation_score = val_bs.get(model, np.nan)
            selection_protocol = "in_sample_validation"
        rows.append(
            {
                "region": args.region,
                "target_product": "SMAP_L4_SPL4SMGP.008_modern_multi3snapshot",
                "forecast_file": str(args.forecast_csv),
                "model": model,
                "selected_by_validation": model
                in {
                    gefs_best,
                    persistence_best,
                    "operational_gefs_selected",
                    "persistence_selected",
                    "stack_validation_selected",
                    "cv_operational_gefs_selected",
                    "cv_persistence_selected",
                    "cv_stack_selected",
                    "persistence_guard_selected",
                },
                "n_validation_months": int(len(val)),
                "n_test_months": int(len(test)),
                "test_start": f"{args.test_start_year}-01",
                "test_end": f"{args.test_end_year}-12",
                "brier_score": bs_model,
                "brier_score_climatology": bs_clim,
                "bss_vs_climatology": bss(test["y_true_dry_frac"], test[col], test["clim_prob_dry"]),
                "bss_ci_low": lo,
                "bss_ci_high": hi,
                "delta_bs_vs_raw_persistence": bs_model
                - brier(test["y_true_dry_frac"], test["persistence_raw_prob_dry"]),
                "delta_bs_ci_low": delta_low,
                "delta_bs_ci_high": delta_high,
                "validation_brier_score": validation_score,
                "selected_weight_gefs": selected_weight_gefs,
                "selected_weight_persistence": (
                    1.0 - selected_weight_gefs if np.isfinite(selected_weight_gefs) else np.nan
                ),
                "selected_gefs_calibration": (
                    str(cv_meta["cv_gefs_best"]) if model.startswith("cv_") or model == "persistence_guard_selected" else gefs_best
                ),
                "selected_persistence_calibration": (
                    str(cv_meta["cv_persistence_best"]) if model.startswith("cv_") or model == "persistence_guard_selected" else persistence_best
                ),
                "selection_protocol": selection_protocol,
                "passes_persistence_guard": bool(cv_meta["passes_persistence_guard"])
                if model == "persistence_guard_selected"
                else np.nan,
                "cv_max_yearly_delta_bs_vs_raw_persistence": float(
                    cv_meta["cv_max_yearly_delta_bs_vs_raw_persistence"]
                )
                if model in {"cv_stack_selected", "persistence_guard_selected"}
                else np.nan,
                "cv_min_yearly_delta_bs_vs_raw_persistence": float(
                    cv_meta["cv_min_yearly_delta_bs_vs_raw_persistence"]
                )
                if model in {"cv_stack_selected", "persistence_guard_selected"}
                else np.nan,
                "spearman_model_vs_observed": test[col].corr(test["y_true_dry_frac"], method="spearman"),
                "climatology_mode": args.climatology_mode,
            }
        )
    summary = pd.DataFrame(rows)
    keep_monthly = [
        "split",
        "target_time",
        "target_year",
        "target_month",
        "y_true_dry_frac",
        "observed_smap_rootzone_pctl",
        "n_snapshots",
        "clim_prob_dry",
        "persistence_raw_prob_dry",
        "forecast_subroot_sm",
        "forecast_subroot_anom",
        *model_cols.values(),
    ]
    monthly = monthly[[c for c in dict.fromkeys(keep_monthly) if c in monthly.columns]].copy()
    text = (
        "Operational GEFS vs modern SMAP L4 target validation\n"
        f"Region: {args.region}\n"
        f"Forecast: {args.forecast_csv}\n"
        f"Target: {args.target_csv}\n"
        f"Validation: {args.validation_start_year}-{args.validation_end_year}; "
        f"Test: {args.test_start_year}-{args.test_end_year}\n"
        f"Selected GEFS calibration: {gefs_best}\n"
        f"Selected persistence calibration: {persistence_best}\n"
        f"Selected stack GEFS weight: {weight:.2f}\n"
        f"Year-held-out GEFS calibration: {cv_meta['cv_gefs_best']}\n"
        f"Year-held-out persistence calibration: {cv_meta['cv_persistence_best']}\n"
        f"Year-held-out stack GEFS weight: {float(cv_meta['cv_stack_weight']):.2f}\n"
        f"Persistence guard passed: {bool(cv_meta['passes_persistence_guard'])} "
        f"(max yearly delta BS vs raw persistence = "
        f"{float(cv_meta['cv_max_yearly_delta_bs_vs_raw_persistence']):+.4f})\n"
        "Interpret as short-record satellite-assimilated target validation, not deployment proof.\n"
    )
    return summary, monthly, text


def main() -> None:
    args = parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    summary, monthly, text = score(args)
    summary_path = OUT_DIR / f"{args.out_prefix}_summary.csv"
    monthly_path = OUT_DIR / f"{args.out_prefix}_monthly_scores.csv"
    scores_path = OUT_DIR / f"{args.out_prefix}_scores.txt"
    summary.to_csv(summary_path, index=False)
    monthly.to_csv(monthly_path, index=False)
    scores_path.write_text(text + "\n" + summary.to_string(index=False) + "\n", encoding="utf-8")
    if args.copy_report:
        for path in [summary_path, monthly_path, scores_path]:
            shutil.copy2(path, REPORT_DIR / path.name)
    print(text)
    print(summary.to_string(index=False))
    print("")
    print(f"Wrote {summary_path.relative_to(PROJECT_ROOT)}")
    print(f"Wrote {monthly_path.relative_to(PROJECT_ROOT)}")
    print(f"Wrote {scores_path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
