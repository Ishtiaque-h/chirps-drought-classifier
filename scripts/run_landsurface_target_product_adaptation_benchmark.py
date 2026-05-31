#!/usr/bin/env python
"""Validation-only target-product adaptation benchmark for SMAP RZSM drought.

This script turns the calibration-transfer ladder into a leakage-safe adaptation
benchmark. Candidate probabilities are read from the ladder monthly file. The
adaptation strategy is selected using only SMAP validation months (2015-2016)
and scored on the frozen SMAP test period (2017-2019).

The benchmark distinguishes fixed methods from validation-selected methods:

  - fixed direct transfer;
  - fixed source-to-SMAP dry-rate shift;
  - fixed prediction-to-SMAP base-rate shift;
  - fixed shrunk seasonal prediction-to-SMAP base-rate shift;
  - fixed bias-corrected temperature/base-rate scaling;
  - fixed ridge-logit SMAP recalibration;
  - fixed SMAP validation recalibration;
  - simple selector: direct vs dry-rate shift;
  - base-rate selector: direct vs the base-rate-only shifts;
  - full selector: direct vs dry-rate shift vs SMAP validation recalibration;
  - pooled source calibration and SMAP self-calibration as references.
"""

from __future__ import annotations

from argparse import ArgumentParser, Namespace
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from run_gefsv12_landsurface_stack_benchmark import (
    PROJECT_ROOT,
    added_value_status,
    bootstrap_delta_bs,
)
from run_landsurface_forecast_benchmark import brier, bss, bootstrap_bss


OUT_DIR = PROJECT_ROOT / "outputs"
REPORT_DIR = PROJECT_ROOT / "results" / "report"
LANDSURFACE_DIR = REPORT_DIR / "landsurface"

DEFAULT_LADDER_MONTHLY = (
    LANDSURFACE_DIR
    / "landsurface_calibration_transfer_ladder_smap_multi3snapshot_monthly_scores.csv"
)

FIXED_STRATEGY_LABELS = {
    "source_direct": "Fixed direct transfer",
    "target_prior_shift": "Fixed validation dry-rate shift",
    "prediction_base_rate_shift": "Fixed validation prediction-base-rate shift",
    "seasonal_prediction_base_rate_shift": "Fixed shrunk seasonal base-rate shift",
    "temperature_base_rate_scaling": "Fixed bias-corrected temperature/base-rate scaling",
    "ridge_logit_recalibration": "Fixed ridge-logit SMAP recalibration",
    "smap_validation_recalibration": "Fixed SMAP validation recalibration",
    "pooled_source_calibration": "Pooled source calibration",
    "smap_self_calibrated_reference": "SMAP self-calibrated reference",
}

PRIMARY_MODELS = ["gefs_selected", "stack_validation_selected"]
SOURCE_PRODUCTS = ["era5_land", "gldas_noah"]
EPS = 1.0e-5


def parse_args() -> Namespace:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--ladder-monthly", type=Path, default=DEFAULT_LADDER_MONTHLY)
    parser.add_argument("--validation-start-year", type=int, default=2015)
    parser.add_argument("--validation-end-year", type=int, default=2016)
    parser.add_argument("--test-start-year", type=int, default=2017)
    parser.add_argument("--test-end-year", type=int, default=2019)
    parser.add_argument(
        "--monthly-shrink-k",
        type=float,
        default=6.0,
        help=(
            "Pseudo-month count for shrinking month-specific base-rate logit "
            "offsets toward the global validation offset."
        ),
    )
    parser.add_argument(
        "--ridge-alpha",
        type=float,
        default=1.0,
        help="Ridge penalty for the target-specific logit recalibration candidate.",
    )
    parser.add_argument("--temperature-slope-min", type=float, default=0.25)
    parser.add_argument("--temperature-slope-max", type=float, default=2.0)
    parser.add_argument("--temperature-slope-steps", type=int, default=72)
    parser.add_argument("--n-bootstrap", type=int, default=2000)
    parser.add_argument(
        "--out-prefix",
        default="landsurface_target_product_adaptation_benchmark_smap_multi3snapshot",
    )
    parser.add_argument("--copy-report", action="store_true")
    return parser.parse_args()


def claim_status(bss_value: float, ci_low: float, ci_high: float) -> str:
    if not np.isfinite(bss_value):
        return "not_applicable"
    if np.isfinite(ci_low) and np.isfinite(ci_high):
        if ci_low > 0:
            return "robust_positive"
        if ci_high < 0:
            return "robust_negative"
        if bss_value > 0:
            return "positive_uncertain"
        return "negative_uncertain"
    if bss_value > 0:
        return "positive_no_ci"
    if bss_value < 0:
        return "negative_no_ci"
    return "reference"


def read_monthly(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Ladder monthly file not found: {path}")
    df = pd.read_csv(path, parse_dates=["target_time"])
    required = {
        "region",
        "target_time",
        "target_year",
        "target_month",
        "y_true_dry_frac",
        "smap_clim_prob_dry",
        "smap_persistence_selected_prob_dry",
        "prob_dry",
        "strategy",
        "source_product",
        "model",
    }
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Ladder monthly file is missing columns: {sorted(missing)}")
    return df


def split_years(df: pd.DataFrame, start_year: int, end_year: int) -> pd.DataFrame:
    return df.loc[(df["target_year"] >= start_year) & (df["target_year"] <= end_year)].copy()


def clip_prob(values: pd.Series | np.ndarray | float) -> np.ndarray:
    return np.clip(np.asarray(values, dtype=float), EPS, 1.0 - EPS)


def logit(values: pd.Series | np.ndarray | float) -> np.ndarray:
    p = clip_prob(values)
    return np.log(p / (1.0 - p))


def expit(values: pd.Series | np.ndarray | float) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.asarray(values, dtype=float)))


def apply_logit_offset(prob: pd.Series, offset: float) -> pd.Series:
    shifted = expit(logit(prob) + float(offset))
    return pd.Series(shifted, index=prob.index).clip(0.0, 1.0)


def fit_mean_matching_logit_offset(prob: pd.Series, target_mean: float) -> float:
    target = float(np.clip(target_mean, EPS, 1.0 - EPS))
    p = pd.Series(prob).dropna()
    if p.empty:
        return 0.0
    logits = logit(p.to_numpy(dtype=float))
    lo, hi = -20.0, 20.0
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        mean_mid = float(expit(logits + mid).mean())
        if mean_mid < target:
            lo = mid
        else:
            hi = mid
    return float(0.5 * (lo + hi))


def add_prediction_base_rate_shift(
    monthly: pd.DataFrame,
    args: Namespace,
) -> pd.DataFrame:
    direct = monthly.loc[
        monthly["strategy"].eq("source_direct")
        & monthly["source_product"].isin(SOURCE_PRODUCTS)
        & monthly["model"].isin(PRIMARY_MODELS)
    ].copy()
    rows = []
    for _, group in direct.groupby(["region", "source_product", "model"], sort=True):
        group = group.copy()
        val = split_years(group, args.validation_start_year, args.validation_end_year)
        val = val.dropna(subset=["prob_dry", "y_true_dry_frac"])
        if val.empty:
            continue
        offset = fit_mean_matching_logit_offset(
            val["prob_dry"],
            target_mean=float(val["y_true_dry_frac"].mean()),
        )
        group["prob_dry"] = apply_logit_offset(group["prob_dry"], offset)
        group["strategy"] = "prediction_base_rate_shift"
        group["model_label"] = "prediction_base_rate_" + group["model"].astype(str)
        group["validation_prediction_mean_dry"] = float(val["prob_dry"].mean())
        group["smap_validation_mean_dry_for_prediction_shift"] = float(
            val["y_true_dry_frac"].mean()
        )
        group["prediction_base_rate_logit_offset"] = offset
        rows.append(group)
    if not rows:
        return pd.DataFrame(columns=monthly.columns)
    return pd.concat(rows, ignore_index=True)


def add_seasonal_base_rate_shift(
    monthly: pd.DataFrame,
    args: Namespace,
) -> pd.DataFrame:
    direct = monthly.loc[
        monthly["strategy"].eq("source_direct")
        & monthly["source_product"].isin(SOURCE_PRODUCTS)
        & monthly["model"].isin(PRIMARY_MODELS)
    ].copy()
    rows = []
    for _, group in direct.groupby(["region", "source_product", "model"], sort=True):
        group = group.copy()
        val = split_years(group, args.validation_start_year, args.validation_end_year)
        val = val.dropna(subset=["prob_dry", "y_true_dry_frac"])
        if val.empty:
            continue
        global_offset = fit_mean_matching_logit_offset(
            val["prob_dry"],
            target_mean=float(val["y_true_dry_frac"].mean()),
        )
        offsets: dict[int, float] = {}
        shrink_weights: dict[int, float] = {}
        for month, month_val in val.groupby("target_month", sort=True):
            month_val = month_val.dropna(subset=["prob_dry", "y_true_dry_frac"])
            n_month = int(len(month_val))
            if n_month == 0:
                offsets[int(month)] = global_offset
                shrink_weights[int(month)] = 0.0
                continue
            month_offset = fit_mean_matching_logit_offset(
                month_val["prob_dry"],
                target_mean=float(month_val["y_true_dry_frac"].mean()),
            )
            weight = n_month / (n_month + float(args.monthly_shrink_k))
            offsets[int(month)] = float(weight * month_offset + (1.0 - weight) * global_offset)
            shrink_weights[int(month)] = float(weight)
        group["prob_dry"] = [
            float(apply_logit_offset(pd.Series([prob]), offsets.get(int(month), global_offset)).iloc[0])
            for prob, month in zip(group["prob_dry"], group["target_month"], strict=True)
        ]
        group["strategy"] = "seasonal_prediction_base_rate_shift"
        group["model_label"] = "seasonal_base_rate_" + group["model"].astype(str)
        group["validation_prediction_mean_dry"] = float(val["prob_dry"].mean())
        group["smap_validation_mean_dry_for_prediction_shift"] = float(
            val["y_true_dry_frac"].mean()
        )
        group["prediction_base_rate_logit_offset"] = global_offset
        group["monthly_shrink_k"] = float(args.monthly_shrink_k)
        group["mean_monthly_shrink_weight"] = (
            float(np.mean(list(shrink_weights.values()))) if shrink_weights else np.nan
        )
        rows.append(group)
    if not rows:
        return pd.DataFrame(columns=monthly.columns)
    return pd.concat(rows, ignore_index=True)


def fit_temperature_base_rate_parameters(
    prob: pd.Series,
    y_true: pd.Series,
    args: Namespace,
) -> tuple[float, float, float]:
    logits = logit(prob.to_numpy(dtype=float))
    target_mean = float(pd.Series(y_true).mean())
    slopes = np.linspace(
        float(args.temperature_slope_min),
        float(args.temperature_slope_max),
        int(args.temperature_slope_steps),
    )
    best: tuple[float, float, float] | None = None
    for slope in slopes:
        offset = fit_mean_matching_logit_offset(
            pd.Series(expit(float(slope) * logits)),
            target_mean=target_mean,
        )
        pred = expit(float(slope) * logits + offset)
        score = brier(y_true, pred)
        if best is None or score < best[0]:
            best = (float(score), float(slope), float(offset))
    if best is None:
        raise ValueError("Could not fit temperature/base-rate parameters.")
    return best


def add_temperature_base_rate_scaling(
    monthly: pd.DataFrame,
    args: Namespace,
) -> pd.DataFrame:
    direct = monthly.loc[
        monthly["strategy"].eq("source_direct")
        & monthly["source_product"].isin(SOURCE_PRODUCTS)
        & monthly["model"].isin(PRIMARY_MODELS)
    ].copy()
    rows = []
    for _, group in direct.groupby(["region", "source_product", "model"], sort=True):
        group = group.copy()
        val = split_years(group, args.validation_start_year, args.validation_end_year)
        val = val.dropna(subset=["prob_dry", "y_true_dry_frac"])
        if val.empty:
            continue
        val_bs, slope, offset = fit_temperature_base_rate_parameters(
            val["prob_dry"],
            val["y_true_dry_frac"],
            args,
        )
        group["prob_dry"] = expit(slope * logit(group["prob_dry"].to_numpy(dtype=float)) + offset)
        group["prob_dry"] = group["prob_dry"].clip(0.0, 1.0)
        group["strategy"] = "temperature_base_rate_scaling"
        group["model_label"] = "temperature_base_rate_" + group["model"].astype(str)
        group["temperature_base_rate_validation_bs"] = val_bs
        group["temperature_base_rate_slope"] = slope
        group["temperature_base_rate_offset"] = offset
        group["temperature_base_rate_target_mean"] = float(val["y_true_dry_frac"].mean())
        rows.append(group)
    if not rows:
        return pd.DataFrame(columns=monthly.columns)
    return pd.concat(rows, ignore_index=True)


def add_ridge_logit_recalibration(monthly: pd.DataFrame, args: Namespace) -> pd.DataFrame:
    direct = monthly.loc[
        monthly["strategy"].eq("source_direct")
        & monthly["source_product"].isin(SOURCE_PRODUCTS)
        & monthly["model"].isin(PRIMARY_MODELS)
    ].copy()
    rows = []
    for _, group in direct.groupby(["region", "source_product", "model"], sort=True):
        group = group.copy()
        val = split_years(group, args.validation_start_year, args.validation_end_year)
        val = val.dropna(subset=["prob_dry", "y_true_dry_frac"])
        if len(val) < 8:
            continue
        x = logit(val["prob_dry"].to_numpy(dtype=float)).reshape(-1, 1)
        y = logit(val["y_true_dry_frac"].to_numpy(dtype=float))
        model = Ridge(alpha=float(args.ridge_alpha))
        model.fit(x, y)
        pred_x = logit(group["prob_dry"].to_numpy(dtype=float)).reshape(-1, 1)
        group["prob_dry"] = expit(model.predict(pred_x))
        group["prob_dry"] = group["prob_dry"].clip(0.0, 1.0)
        group["strategy"] = "ridge_logit_recalibration"
        group["model_label"] = "ridge_logit_" + group["model"].astype(str)
        group["ridge_logit_alpha"] = float(args.ridge_alpha)
        group["ridge_logit_intercept"] = float(model.intercept_)
        group["ridge_logit_slope"] = float(model.coef_[0])
        rows.append(group)
    if not rows:
        return pd.DataFrame(columns=monthly.columns)
    return pd.concat(rows, ignore_index=True)


def add_generated_candidate_strategies(monthly: pd.DataFrame, args: Namespace) -> pd.DataFrame:
    generated = [
        add_prediction_base_rate_shift(monthly, args),
        add_seasonal_base_rate_shift(monthly, args),
        add_temperature_base_rate_scaling(monthly, args),
        add_ridge_logit_recalibration(monthly, args),
    ]
    parts = [monthly.copy()] + [part for part in generated if not part.empty]
    return pd.concat(parts, ignore_index=True, sort=False)


def validation_brier(rows: pd.DataFrame) -> float:
    return brier(rows["y_true_dry_frac"].to_numpy(dtype=float), rows["prob_dry"].to_numpy(dtype=float))


def choose_strategy(
    monthly: pd.DataFrame,
    strategies: list[str],
    region: str,
    source_product: str,
    model: str,
    args: Namespace,
) -> tuple[str, dict[str, float]]:
    val = split_years(monthly, args.validation_start_year, args.validation_end_year)
    scores: dict[str, float] = {}
    for strategy in strategies:
        rows = val.loc[
            val["strategy"].eq(strategy)
            & val["region"].eq(region)
            & val["source_product"].eq(source_product)
            & val["model"].eq(model)
        ].dropna(subset=["prob_dry", "y_true_dry_frac"])
        if rows.empty:
            continue
        scores[strategy] = validation_brier(rows)
    if not scores:
        raise ValueError(
            f"No validation candidates for region={region}, source={source_product}, model={model}."
        )
    preference = {strategy: i for i, strategy in enumerate(strategies)}
    selected = sorted(scores, key=lambda strategy: (scores[strategy], preference[strategy]))[0]
    return selected, scores


def fixed_candidate_rows(
    monthly: pd.DataFrame,
    strategies: list[str],
    args: Namespace,
) -> pd.DataFrame:
    keep = monthly.loc[
        monthly["strategy"].isin(strategies)
        & monthly["model"].isin(PRIMARY_MODELS)
    ].copy()
    keep["adaptation_method"] = keep["strategy"].map(FIXED_STRATEGY_LABELS)
    keep["selected_strategy"] = keep["strategy"]
    keep["selection_scope"] = "fixed_method"
    keep["validation_bs_selected"] = np.nan
    return keep


def selector_rows(
    monthly: pd.DataFrame,
    strategies: list[str],
    method_name: str,
    args: Namespace,
) -> pd.DataFrame:
    rows = []
    source_rows = monthly.loc[
        monthly["source_product"].isin(["era5_land", "gldas_noah"])
        & monthly["model"].isin(PRIMARY_MODELS)
    ]
    groups = source_rows[["region", "source_product", "model"]].drop_duplicates()
    for record in groups.to_dict(orient="records"):
        region = str(record["region"])
        source_product = str(record["source_product"])
        model = str(record["model"])
        selected, scores = choose_strategy(
            monthly,
            strategies=strategies,
            region=region,
            source_product=source_product,
            model=model,
            args=args,
        )
        chosen = monthly.loc[
            monthly["strategy"].eq(selected)
            & monthly["region"].eq(region)
            & monthly["source_product"].eq(source_product)
            & monthly["model"].eq(model)
        ].copy()
        chosen["adaptation_method"] = method_name
        chosen["selected_strategy"] = selected
        chosen["selection_scope"] = "validation_selected"
        chosen["validation_bs_selected"] = scores[selected]
        for strategy, score in scores.items():
            chosen[f"validation_bs_{strategy}"] = score
        rows.append(chosen)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def build_adapted_monthly(monthly: pd.DataFrame, args: Namespace) -> pd.DataFrame:
    candidate_monthly = add_generated_candidate_strategies(monthly, args)
    parts = [
        fixed_candidate_rows(
            candidate_monthly,
            strategies=[
                "source_direct",
                "target_prior_shift",
                "prediction_base_rate_shift",
                "seasonal_prediction_base_rate_shift",
                "temperature_base_rate_scaling",
                "ridge_logit_recalibration",
                "smap_validation_recalibration",
                "pooled_source_calibration",
                "smap_self_calibrated_reference",
            ],
            args=args,
        ),
        selector_rows(
            candidate_monthly,
            strategies=["source_direct", "target_prior_shift"],
            method_name="Validation-selected simple adaptation",
            args=args,
        ),
        selector_rows(
            candidate_monthly,
            strategies=[
                "source_direct",
                "target_prior_shift",
                "prediction_base_rate_shift",
                "seasonal_prediction_base_rate_shift",
            ],
            method_name="Validation-selected base-rate adaptation",
            args=args,
        ),
        selector_rows(
            candidate_monthly,
            strategies=["source_direct", "target_prior_shift", "smap_validation_recalibration"],
            method_name="Validation-selected full adaptation",
            args=args,
        ),
        selector_rows(
            candidate_monthly,
            strategies=[
                "source_direct",
                "target_prior_shift",
                "prediction_base_rate_shift",
                "seasonal_prediction_base_rate_shift",
                "ridge_logit_recalibration",
                "temperature_base_rate_scaling",
                "smap_validation_recalibration",
            ],
            method_name="Validation-selected complex adaptation",
            args=args,
        ),
    ]
    out = pd.concat([part for part in parts if not part.empty], ignore_index=True)
    return out.sort_values(
        ["adaptation_method", "source_product", "region", "model", "target_time"]
    ).reset_index(drop=True)


def score_group(group: pd.DataFrame, n_bootstrap: int, seed: int) -> dict[str, object]:
    test = group.copy()
    scoring = test.rename(
        columns={
            "smap_clim_prob_dry": "clim_prob_dry",
            "smap_persistence_selected_prob_dry": "persistence_selected_prob_dry",
        }
    )
    y = scoring["y_true_dry_frac"].to_numpy(dtype=float)
    pred = scoring["prob_dry"].to_numpy(dtype=float)
    clim = scoring["clim_prob_dry"].to_numpy(dtype=float)
    persistence = scoring["persistence_selected_prob_dry"].to_numpy(dtype=float)
    ci_low, ci_high = bootstrap_bss(
        scoring,
        pred_col="prob_dry",
        ref_col="clim_prob_dry",
        n_bootstrap=n_bootstrap,
        seed=seed,
    )
    pers_ci_low, pers_ci_high = bootstrap_bss(
        scoring,
        pred_col="prob_dry",
        ref_col="persistence_selected_prob_dry",
        n_bootstrap=n_bootstrap,
        seed=seed + 100_000,
    )
    delta = brier(y, pred) - brier(y, persistence)
    delta_low, delta_high = bootstrap_delta_bs(
        scoring,
        candidate_col="prob_dry",
        reference_col="persistence_selected_prob_dry",
        n_bootstrap=n_bootstrap,
        seed=seed + 200_000,
    )
    bss_clim = bss(y, pred, clim)
    return {
        "n_test_months": int(len(scoring)),
        "bs_smap_climatology": brier(y, clim),
        "bs_smap_persistence_selected": brier(y, persistence),
        "bs_model": brier(y, pred),
        "bss_vs_smap_climatology": bss_clim,
        "bss_vs_smap_climatology_ci_low": ci_low,
        "bss_vs_smap_climatology_ci_high": ci_high,
        "bss_vs_smap_persistence_selected": bss(y, pred, persistence),
        "bss_vs_smap_persistence_selected_ci_low": pers_ci_low,
        "bss_vs_smap_persistence_selected_ci_high": pers_ci_high,
        "delta_bs_model_minus_smap_persistence_selected": delta,
        "delta_bs_ci_low": delta_low,
        "delta_bs_ci_high": delta_high,
        "claim_status_vs_smap_climatology": claim_status(bss_clim, ci_low, ci_high),
        "added_value_status_vs_smap_persistence": added_value_status(delta, delta_low, delta_high),
        "spearman_model_vs_smap_observed": scoring["prob_dry"].corr(
            scoring["y_true_dry_frac"], method="spearman"
        ),
        "amplitude_ratio_model_vs_smap_observed": float(np.std(pred, ddof=0) / np.std(y, ddof=0))
        if float(np.std(y, ddof=0)) > 0
        else np.nan,
    }


def score_adapted_monthly(adapted: pd.DataFrame, args: Namespace) -> pd.DataFrame:
    test = split_years(adapted, args.test_start_year, args.test_end_year)
    group_cols = [
        "adaptation_method",
        "selection_scope",
        "source_product",
        "region",
        "model",
        "selected_strategy",
    ]
    rows = []
    for i, (keys, group) in enumerate(test.groupby(group_cols, dropna=False, sort=True), start=1):
        row = dict(zip(group_cols, keys, strict=True))
        row.update(score_group(group, n_bootstrap=args.n_bootstrap, seed=18001 + i))
        validation_cols = [
            col
            for col in group.columns
            if col == "validation_bs_selected" or col.startswith("validation_bs_")
        ]
        for col in sorted(set(validation_cols)):
            row[col] = group[col].dropna().iloc[0] if col in group.columns and not group[col].dropna().empty else np.nan
        rows.append(row)
    return pd.DataFrame(rows).sort_values(
        ["adaptation_method", "source_product", "region", "model"]
    ).reset_index(drop=True)


def score_adapted_by_year(adapted: pd.DataFrame, args: Namespace) -> pd.DataFrame:
    test = split_years(adapted, args.test_start_year, args.test_end_year)
    group_cols = [
        "adaptation_method",
        "selection_scope",
        "source_product",
        "region",
        "model",
        "selected_strategy",
        "target_year",
    ]
    rows = []
    for keys, group in test.groupby(group_cols, dropna=False, sort=True):
        y = group["y_true_dry_frac"].to_numpy(dtype=float)
        pred = group["prob_dry"].to_numpy(dtype=float)
        clim = group["smap_clim_prob_dry"].to_numpy(dtype=float)
        persistence = group["smap_persistence_selected_prob_dry"].to_numpy(dtype=float)
        row = dict(zip(group_cols, keys, strict=True))
        row.update(
            {
                "n_test_months": int(len(group)),
                "bs_smap_climatology": brier(y, clim),
                "bs_smap_persistence_selected": brier(y, persistence),
                "bs_model": brier(y, pred),
                "bss_vs_smap_climatology": bss(y, pred, clim),
                "bss_vs_smap_persistence_selected": bss(y, pred, persistence),
                "delta_bs_model_minus_smap_persistence_selected": brier(y, pred)
                - brier(y, persistence),
                "spearman_model_vs_smap_observed": group["prob_dry"].corr(
                    group["y_true_dry_frac"], method="spearman"
                ),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows).sort_values(
        ["adaptation_method", "source_product", "region", "model", "target_year"]
    ).reset_index(drop=True)


def compact_summary(summary: pd.DataFrame) -> pd.DataFrame:
    out = (
        summary.groupby(["adaptation_method", "model"], dropna=False)
        .agg(
            n_rows=("region", "size"),
            n_regions=("region", "nunique"),
            n_sources=("source_product", "nunique"),
            robust_positive_count=(
                "claim_status_vs_smap_climatology",
                lambda s: int((s == "robust_positive").sum()),
            ),
            robust_added_value_count=(
                "added_value_status_vs_smap_persistence",
                lambda s: int((s == "stack_robust_added_value").sum()),
            ),
            mean_bss_vs_smap_climatology=("bss_vs_smap_climatology", "mean"),
            median_bss_vs_smap_climatology=("bss_vs_smap_climatology", "median"),
            mean_bss_vs_smap_persistence=("bss_vs_smap_persistence_selected", "mean"),
            median_bss_vs_smap_persistence=("bss_vs_smap_persistence_selected", "median"),
        )
        .reset_index()
    )
    out["robust_positive_fraction"] = out["robust_positive_count"] / out["n_rows"]
    out["robust_added_value_fraction"] = out["robust_added_value_count"] / out["n_rows"]
    return out.sort_values(["adaptation_method", "model"]).reset_index(drop=True)


def compact_yearly_summary(yearly: pd.DataFrame) -> pd.DataFrame:
    out = (
        yearly.groupby(["adaptation_method", "model", "target_year"], dropna=False)
        .agg(
            n_rows=("region", "size"),
            n_regions=("region", "nunique"),
            n_sources=("source_product", "nunique"),
            positive_bss_count=("bss_vs_smap_climatology", lambda s: int((s > 0).sum())),
            positive_added_value_count=(
                "delta_bs_model_minus_smap_persistence_selected",
                lambda s: int((s < 0).sum()),
            ),
            mean_bss_vs_smap_climatology=("bss_vs_smap_climatology", "mean"),
            median_bss_vs_smap_climatology=("bss_vs_smap_climatology", "median"),
            mean_bss_vs_smap_persistence=("bss_vs_smap_persistence_selected", "mean"),
            median_bss_vs_smap_persistence=("bss_vs_smap_persistence_selected", "median"),
        )
        .reset_index()
    )
    out["positive_bss_fraction"] = out["positive_bss_count"] / out["n_rows"]
    out["positive_added_value_fraction"] = out["positive_added_value_count"] / out["n_rows"]
    return out.sort_values(["adaptation_method", "model", "target_year"]).reset_index(drop=True)


def write_summary_text(compact: pd.DataFrame, path: Path) -> None:
    lines = [
        "Target-product adaptation benchmark against SMAP L4 multi-snapshot target",
        "",
        compact.to_string(index=False),
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    monthly = read_monthly(args.ladder_monthly)
    adapted = build_adapted_monthly(monthly, args)
    summary = score_adapted_monthly(adapted, args)
    compact = compact_summary(summary)
    yearly = score_adapted_by_year(adapted, args)
    yearly_compact = compact_yearly_summary(yearly)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    monthly_path = OUT_DIR / f"{args.out_prefix}_monthly_scores.csv"
    summary_path = OUT_DIR / f"{args.out_prefix}_summary.csv"
    compact_path = OUT_DIR / f"{args.out_prefix}_compact_summary.csv"
    yearly_path = OUT_DIR / f"{args.out_prefix}_yearly_summary.csv"
    yearly_compact_path = OUT_DIR / f"{args.out_prefix}_yearly_compact_summary.csv"
    text_path = OUT_DIR / f"{args.out_prefix}_summary.txt"
    adapted.to_csv(monthly_path, index=False)
    summary.to_csv(summary_path, index=False)
    compact.to_csv(compact_path, index=False)
    yearly.to_csv(yearly_path, index=False)
    yearly_compact.to_csv(yearly_compact_path, index=False)
    write_summary_text(compact, text_path)

    if args.copy_report:
        LANDSURFACE_DIR.mkdir(parents=True, exist_ok=True)
        for path in [monthly_path, summary_path, compact_path, yearly_path, yearly_compact_path, text_path]:
            shutil.copy2(path, LANDSURFACE_DIR / path.name)

    print(f"Wrote {summary_path}")
    print(f"Wrote {compact_path}")
    print(f"Wrote {yearly_compact_path}")
    print(compact.to_string(index=False))


if __name__ == "__main__":
    main()
