#!/usr/bin/env python
"""Domain-transfer benchmark for GEFSv12 land-surface drought probabilities.

This script evaluates whether the GEFSv12 root-zone soil-moisture signal and
same-target persistence calibration transfer across regions. It reuses completed
GEFSv12 forecast extractions and ERA5-Land root-zone dry-fraction targets.

Transfer modes:

  - local: calibrate and select the stack weight on the target region's
    validation years;
  - pooled: calibrate and select the stack weight on all regions' validation
    years, including the target region;
  - leave_one_region_out: calibrate and select the stack weight on all
    validation regions except the target region.

The frozen test window is always the target region's held-out years.
"""

from __future__ import annotations

from argparse import ArgumentParser, Namespace
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import average_precision_score

from region_config import resolve_region
from run_gefsv12_landsurface_stack_benchmark import (
    LANDSURFACE_DIR,
    OUT_DIR,
    PROJECT_ROOT,
    added_value_status,
    bootstrap_delta_bs,
    choose_convex_weight,
    load_or_build_forecast,
    resolve_soil_file,
)
from run_landsurface_forecast_benchmark import (
    brier,
    bss,
    bootstrap_bss,
    fit_isotonic,
    observed_rootzone_target,
)


DEFAULT_REGIONS = [
    "cvalley",
    "southern_great_plains",
    "mediterranean_spain",
    "murray_darling",
    "horn_of_africa",
]

MONOTONIC_XGB_FEATURES: list[tuple[str, int]] = [
    ("gefs_anom_dry_signal", 1),
    ("persistence_raw_prob_dry", 1),
    ("target_month_sin", 0),
    ("target_month_cos", 0),
]

RARE_EVENT_MODELS = {
    "gefs_transfer_selected": "gefs_transfer_selected_prob_dry",
    "persistence_transfer_selected": "persistence_transfer_selected_prob_dry",
    "stack_transfer_selected": "stack_transfer_selected_prob_dry",
    "monotonic_xgb_transfer": "monotonic_xgb_transfer_prob_dry",
}


def parse_args() -> Namespace:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--regions", nargs="+", default=DEFAULT_REGIONS)
    parser.add_argument("--forecast-dir", type=Path, default=PROJECT_ROOT / "results" / "report")
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
    parser.add_argument("--valid-day", type=int, default=15)
    parser.add_argument("--init-lag-weeks", type=int, default=0)
    parser.add_argument("--weight-steps", type=int, default=101)
    parser.add_argument("--n-bootstrap", type=int, default=2000)
    parser.add_argument(
        "--event-quantiles",
        nargs="+",
        type=float,
        default=[0.80, 0.90],
        help="Validation-period regional dry-fraction quantiles used as rare-event thresholds.",
    )
    parser.add_argument("--event-reliability-bins", type=int, default=5)
    parser.add_argument(
        "--skip-monotonic-xgb",
        action="store_true",
        help="Skip the monotonic XGBoost dry-fraction benchmark.",
    )
    parser.add_argument("--out-prefix", default="landsurface_gefsv12_rzsm_domain_transfer_day15_hindcastcal")
    parser.add_argument("--copy-report", action="store_true")
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--max-months", type=int, default=None)
    parser.add_argument("--progress-every", type=int, default=120)
    return parser.parse_args()


def load_region_frame(region_slug: str, args: Namespace) -> pd.DataFrame:
    region = resolve_region(region_slug).slug
    target_args = Namespace(
        soil_file=resolve_soil_file(region),
        lead_months=args.lead_months,
        normal_start_year=args.normal_start_year,
        normal_end_year=args.normal_end_year,
        dry_quantile=args.dry_quantile,
    )
    observed = observed_rootzone_target(target_args)
    forecast_args = Namespace(**vars(args))
    forecast_args.region = region
    forecast_args.soil_file = target_args.soil_file
    forecast, _source = load_or_build_forecast(region, observed, forecast_args)
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
        raise ValueError(f"No merged GEFSv12/ERA5-Land rows for {region}.")
    merged["region"] = region
    merged["gefs_raw_dry_signal"] = -merged["forecast_rzsm"].astype(float)
    merged["gefs_anom_dry_signal"] = -merged["forecast_rzsm_anom"].astype(float)
    month_angle = 2.0 * np.pi * (merged["target_month"].astype(float) - 1.0) / 12.0
    merged["target_month_sin"] = np.sin(month_angle)
    merged["target_month_cos"] = np.cos(month_angle)
    return merged


def fit_monotonic_xgb_predictions(
    source_val: pd.DataFrame,
    target_test: pd.DataFrame,
    seed: int = 1701,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    """Fit a constrained dry-fraction regressor on source validation rows.

    The monotonic constraints encode a minimal hydrologic prior: higher GEFSv12
    dry anomaly signal and higher same-target persistence probability cannot
    reduce the predicted dry fraction. Seasonal terms are unconstrained.
    """

    from xgboost import XGBRegressor

    available = [
        (name, sign)
        for name, sign in MONOTONIC_XGB_FEATURES
        if name in source_val.columns
        and name in target_test.columns
        and source_val[name].notna().all()
        and target_test[name].notna().all()
    ]
    if not available:
        raise ValueError("No monotonic-XGB features are available.")

    feature_names = [name for name, _sign in available]
    constraints = tuple(sign for _name, sign in available)
    model = XGBRegressor(
        objective="reg:squarederror",
        n_estimators=120,
        max_depth=2,
        learning_rate=0.035,
        min_child_weight=8.0,
        subsample=0.85,
        colsample_bytree=1.0,
        reg_lambda=10.0,
        reg_alpha=0.0,
        monotone_constraints=constraints,
        random_state=seed,
        n_jobs=1,
        verbosity=0,
    )
    x_source = source_val[feature_names].to_numpy(dtype=float)
    y_source = source_val["y_true_dry_frac"].to_numpy(dtype=float)
    x_target = target_test[feature_names].to_numpy(dtype=float)
    model.fit(x_source, y_source)
    source_pred = np.clip(model.predict(x_source), 0.0, 1.0)
    target_pred = np.clip(model.predict(x_target), 0.0, 1.0)
    meta = {
        "monotonic_xgb_features": " ".join(feature_names),
        "monotonic_xgb_constraints": " ".join(str(value) for value in constraints),
    }
    return source_pred, target_pred, meta


def fit_transfer_predictions(
    source_val: pd.DataFrame,
    target_test: pd.DataFrame,
    weight_steps: int,
    include_monotonic_xgb: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    source_val = source_val.copy()
    target_test = target_test.copy()

    val_raw, test_raw = fit_isotonic(source_val, target_test, "gefs_raw_dry_signal")
    val_anom, test_anom = fit_isotonic(source_val, target_test, "gefs_anom_dry_signal")
    val_pers_iso, test_pers_iso = fit_isotonic(source_val, target_test, "persistence_raw_prob_dry")

    source_val["gefs_raw_isotonic_prob_dry"] = val_raw
    source_val["gefs_anom_isotonic_prob_dry"] = val_anom
    source_val["persistence_isotonic_prob_dry"] = val_pers_iso
    target_test["gefs_raw_isotonic_prob_dry"] = test_raw
    target_test["gefs_anom_isotonic_prob_dry"] = test_anom
    target_test["persistence_isotonic_prob_dry"] = test_pers_iso

    val_bs = {
        "gefs_raw_isotonic": brier(source_val["y_true_dry_frac"], source_val["gefs_raw_isotonic_prob_dry"]),
        "gefs_anom_isotonic": brier(source_val["y_true_dry_frac"], source_val["gefs_anom_isotonic_prob_dry"]),
        "persistence_raw": brier(source_val["y_true_dry_frac"], source_val["persistence_raw_prob_dry"]),
        "persistence_isotonic": brier(source_val["y_true_dry_frac"], source_val["persistence_isotonic_prob_dry"]),
    }
    gefs_best = min({k: v for k, v in val_bs.items() if k.startswith("gefs_")}, key=val_bs.get)
    persistence_best = min({k: v for k, v in val_bs.items() if k.startswith("persistence_")}, key=val_bs.get)

    for frame in [source_val, target_test]:
        frame["gefs_transfer_selected_prob_dry"] = (
            frame["gefs_raw_isotonic_prob_dry"]
            if gefs_best == "gefs_raw_isotonic"
            else frame["gefs_anom_isotonic_prob_dry"]
        )
        frame["persistence_transfer_selected_prob_dry"] = (
            frame["persistence_raw_prob_dry"]
            if persistence_best == "persistence_raw"
            else frame["persistence_isotonic_prob_dry"]
        )

    weight, weight_val_bs = choose_convex_weight(
        source_val,
        "gefs_transfer_selected_prob_dry",
        "persistence_transfer_selected_prob_dry",
        weight_steps=weight_steps,
    )
    for frame in [source_val, target_test]:
        frame["stack_transfer_selected_prob_dry"] = (
            weight * frame["gefs_transfer_selected_prob_dry"]
            + (1.0 - weight) * frame["persistence_transfer_selected_prob_dry"]
        ).clip(0.0, 1.0)

    monotonic_meta: dict[str, object] = {}
    if include_monotonic_xgb:
        val_mono, test_mono, monotonic_meta = fit_monotonic_xgb_predictions(source_val, target_test)
        source_val["monotonic_xgb_transfer_prob_dry"] = val_mono
        target_test["monotonic_xgb_transfer_prob_dry"] = test_mono

    meta = {
        "selected_weight_gefs": float(weight),
        "selected_weight_persistence": float(1.0 - weight),
        "selected_weight_val_bs": float(weight_val_bs),
        "selected_gefs_calibration": gefs_best,
        "selected_persistence_calibration": persistence_best,
        "source_validation_bs_gefs_raw_isotonic": float(val_bs["gefs_raw_isotonic"]),
        "source_validation_bs_gefs_anom_isotonic": float(val_bs["gefs_anom_isotonic"]),
        "source_validation_bs_persistence_raw": float(val_bs["persistence_raw"]),
        "source_validation_bs_persistence_isotonic": float(val_bs["persistence_isotonic"]),
        **monotonic_meta,
    }
    return source_val, target_test, meta


def score_transfer(
    target_region: str,
    transfer_mode: str,
    source_regions: list[str],
    source_val: pd.DataFrame,
    target_test: pd.DataFrame,
    meta: dict[str, object],
    args: Namespace,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    candidates = [
        ("gefs_transfer_selected", "gefs_transfer_selected_prob_dry"),
        ("persistence_transfer_selected", "persistence_transfer_selected_prob_dry"),
        ("stack_transfer_selected", "stack_transfer_selected_prob_dry"),
    ]
    if "monotonic_xgb_transfer_prob_dry" in target_test.columns:
        candidates.append(("monotonic_xgb_transfer", "monotonic_xgb_transfer_prob_dry"))
    y = target_test["y_true_dry_frac"].to_numpy(dtype=float)
    clim = target_test["clim_prob_dry"].to_numpy(dtype=float)
    persistence = target_test["persistence_transfer_selected_prob_dry"].to_numpy(dtype=float)
    bs_clim = brier(y, clim)
    bs_persistence = brier(y, persistence)

    for i, (model, col) in enumerate(candidates):
        pred = target_test[col].to_numpy(dtype=float)
        ci_low, ci_high = bootstrap_bss(
            target_test,
            pred_col=col,
            ref_col="clim_prob_dry",
            n_bootstrap=args.n_bootstrap,
            seed=1701 + i,
        )
        delta = brier(y, pred) - bs_persistence
        delta_low, delta_high = bootstrap_delta_bs(
            target_test,
            candidate_col=col,
            reference_col="persistence_transfer_selected_prob_dry",
            n_bootstrap=args.n_bootstrap,
            seed=1801 + i,
        )
        rows.append(
            {
                "target_region": target_region,
                "transfer_mode": transfer_mode,
                "source_regions": " ".join(source_regions),
                "model": model,
                "n_source_validation_months": int(source_val["target_time"].nunique()),
                "n_source_validation_rows": int(len(source_val)),
                "n_test_months": int(target_test["target_time"].nunique()),
                "bs_climatology": bs_clim,
                "bs_model": brier(y, pred),
                "bs_persistence_transfer_selected": bs_persistence,
                "bss_vs_climatology": bss(y, pred, clim),
                "bss_ci_low": ci_low,
                "bss_ci_high": ci_high,
                "bss_vs_persistence_transfer_selected": bss(y, pred, persistence),
                "delta_bs_model_minus_persistence_transfer_selected": delta,
                "delta_bs_ci_low": delta_low,
                "delta_bs_ci_high": delta_high,
                "added_value_status": added_value_status(delta, delta_low, delta_high),
                "selected_weight_gefs": meta["selected_weight_gefs"] if model == "stack_transfer_selected" else np.nan,
                "selected_weight_persistence": meta["selected_weight_persistence"] if model == "stack_transfer_selected" else np.nan,
                "selected_gefs_calibration": meta["selected_gefs_calibration"],
                "selected_persistence_calibration": meta["selected_persistence_calibration"],
                "monotonic_xgb_features": meta.get("monotonic_xgb_features", ""),
                "monotonic_xgb_constraints": meta.get("monotonic_xgb_constraints", ""),
                "spearman_model_vs_observed": target_test[col].corr(target_test["y_true_dry_frac"], method="spearman"),
                "amplitude_ratio_model": (
                    float(target_test[col].std(ddof=0) / target_test["y_true_dry_frac"].std(ddof=0))
                    if float(target_test["y_true_dry_frac"].std(ddof=0)) > 0
                    else np.nan
                ),
            }
        )
    return pd.DataFrame(rows)


def build_event_thresholds(
    validation: pd.DataFrame,
    event_quantiles: list[float],
) -> dict[tuple[str, float], dict[str, float]]:
    thresholds: dict[tuple[str, float], dict[str, float]] = {}
    for region, part in validation.groupby("region"):
        y = part["y_true_dry_frac"].astype(float)
        for quantile in event_quantiles:
            threshold = float(y.quantile(float(quantile)))
            event = (y >= threshold).astype(int)
            thresholds[(str(region), float(quantile))] = {
                "threshold": threshold,
                "validation_event_rate": float(event.mean()),
                "validation_event_count": int(event.sum()),
                "validation_month_count": int(len(event)),
            }
    return thresholds


def event_labels(frame: pd.DataFrame, thresholds: dict[tuple[str, float], dict[str, float]], quantile: float) -> np.ndarray:
    labels = []
    for row in frame[["region", "y_true_dry_frac"]].itertuples(index=False):
        threshold = thresholds[(str(row.region), float(quantile))]["threshold"]
        labels.append(int(float(row.y_true_dry_frac) >= threshold))
    return np.asarray(labels, dtype=int)


def bootstrap_event_bss(
    y: np.ndarray,
    pred: np.ndarray,
    ref_prob: float,
    n_bootstrap: int,
    seed: int,
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    idx = np.arange(len(y))
    vals = np.empty(n_bootstrap, dtype=float)
    ref = np.full(len(y), float(ref_prob), dtype=float)
    for i in range(n_bootstrap):
        sample = rng.choice(idx, size=len(idx), replace=True)
        ref_bs = brier(y[sample], ref[sample])
        vals[i] = 1.0 - brier(y[sample], pred[sample]) / ref_bs if ref_bs > 0 else np.nan
    lo, hi = np.nanquantile(vals, [0.025, 0.975])
    return float(lo), float(hi)


def bootstrap_average_precision(
    y: np.ndarray,
    pred: np.ndarray,
    n_bootstrap: int,
    seed: int,
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    idx = np.arange(len(y))
    vals = np.full(n_bootstrap, np.nan, dtype=float)
    for i in range(n_bootstrap):
        sample = rng.choice(idx, size=len(idx), replace=True)
        y_sample = y[sample]
        if len(np.unique(y_sample)) < 2:
            continue
        vals[i] = average_precision_score(y_sample, pred[sample])
    if np.all(np.isnan(vals)):
        return np.nan, np.nan
    lo, hi = np.nanquantile(vals, [0.025, 0.975])
    return float(lo), float(hi)


def binary_reliability_resolution(y: np.ndarray, pred: np.ndarray, n_bins: int) -> dict[str, float]:
    y = np.asarray(y, dtype=float)
    pred = np.asarray(pred, dtype=float)
    base = float(np.mean(y))
    uncertainty = base * (1.0 - base)
    edges = np.linspace(0.0, 1.0, max(2, int(n_bins)) + 1)
    reliability = 0.0
    resolution = 0.0
    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i + 1]
        if i == len(edges) - 2:
            mask = (pred >= lo) & (pred <= hi)
        else:
            mask = (pred >= lo) & (pred < hi)
        if not mask.any():
            continue
        weight = float(mask.mean())
        mean_pred = float(pred[mask].mean())
        mean_obs = float(y[mask].mean())
        reliability += weight * (mean_pred - mean_obs) ** 2
        resolution += weight * (mean_obs - base) ** 2
    return {
        "event_reliability": float(reliability),
        "event_resolution": float(resolution),
        "event_uncertainty": float(uncertainty),
        "event_resolution_minus_reliability": float(resolution - reliability),
    }


def score_rare_events(
    target_region: str,
    transfer_mode: str,
    source_regions: list[str],
    source_scored: pd.DataFrame,
    target_scored: pd.DataFrame,
    thresholds: dict[tuple[str, float], dict[str, float]],
    args: Namespace,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    monthly_parts: list[pd.DataFrame] = []
    summary_rows: list[dict[str, object]] = []
    for quantile in args.event_quantiles:
        q = float(quantile)
        source_event = event_labels(source_scored, thresholds, q)
        target_event = event_labels(target_scored, thresholds, q)
        target_info = thresholds[(target_region, q)]
        target_ref_prob = float(target_info["validation_event_rate"])

        for i, (model, score_col) in enumerate(RARE_EVENT_MODELS.items()):
            if score_col not in source_scored.columns or score_col not in target_scored.columns:
                continue
            source_score = source_scored[score_col].to_numpy(dtype=float)
            target_score = target_scored[score_col].to_numpy(dtype=float)
            if len(np.unique(source_event)) < 2:
                event_prob = np.full(len(target_event), float(np.mean(source_event)), dtype=float)
                calibration = "constant_source_event_rate"
            else:
                calibrator = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
                calibrator.fit(source_score, source_event)
                event_prob = np.clip(calibrator.predict(target_score), 0.0, 1.0)
                calibration = "isotonic_source_validation"

            bs_event_ref = brier(target_event, np.full(len(target_event), target_ref_prob, dtype=float))
            bs_event_model = brier(target_event, event_prob)
            event_bss = 1.0 - bs_event_model / bs_event_ref if bs_event_ref > 0 else np.nan
            bss_low, bss_high = bootstrap_event_bss(
                target_event,
                event_prob,
                target_ref_prob,
                n_bootstrap=args.n_bootstrap,
                seed=2101 + i,
            )
            if len(np.unique(target_event)) < 2:
                ap = np.nan
                ap_low, ap_high = np.nan, np.nan
            else:
                ap = float(average_precision_score(target_event, event_prob))
                ap_low, ap_high = bootstrap_average_precision(
                    target_event,
                    event_prob,
                    n_bootstrap=args.n_bootstrap,
                    seed=2201 + i,
                )
            baseline_ap = float(target_event.mean())
            rel = binary_reliability_resolution(target_event, event_prob, args.event_reliability_bins)

            summary_rows.append(
                {
                    "target_region": target_region,
                    "transfer_mode": transfer_mode,
                    "source_regions": " ".join(source_regions),
                    "event_quantile": q,
                    "event_threshold_dry_fraction": float(target_info["threshold"]),
                    "model": model,
                    "event_calibration": calibration,
                    "n_source_validation_rows": int(len(source_scored)),
                    "n_test_months": int(len(target_event)),
                    "n_test_events": int(target_event.sum()),
                    "test_event_rate": baseline_ap,
                    "validation_event_rate_reference": target_ref_prob,
                    "average_precision": ap,
                    "average_precision_ci_low": ap_low,
                    "average_precision_ci_high": ap_high,
                    "average_precision_lift_over_event_rate": ap / baseline_ap if baseline_ap > 0 else np.nan,
                    "event_brier_model": bs_event_model,
                    "event_brier_climatology": bs_event_ref,
                    "event_bss_vs_validation_climatology": event_bss,
                    "event_bss_ci_low": bss_low,
                    "event_bss_ci_high": bss_high,
                    "event_claim_status": (
                        "robust_positive"
                        if np.isfinite(bss_low) and bss_low > 0
                        else ("positive_uncertain" if np.isfinite(event_bss) and event_bss > 0 else "not_positive")
                    ),
                    **rel,
                }
            )

            monthly = target_scored[
                ["target_time", "target_year", "target_month", "region", "y_true_dry_frac"]
            ].copy()
            monthly["target_region"] = target_region
            monthly["transfer_mode"] = transfer_mode
            monthly["source_regions"] = " ".join(source_regions)
            monthly["event_quantile"] = q
            monthly["event_threshold_dry_fraction"] = float(target_info["threshold"])
            monthly["model"] = model
            monthly["event_observed"] = target_event
            monthly["event_prob"] = event_prob
            monthly["dry_fraction_score"] = target_score
            monthly_parts.append(monthly)

    monthly_out = pd.concat(monthly_parts, ignore_index=True) if monthly_parts else pd.DataFrame()
    summary_out = pd.DataFrame(summary_rows)
    return monthly_out, summary_out


def transfer_source_regions(mode: str, target_region: str, regions: list[str]) -> list[str]:
    if mode == "local":
        return [target_region]
    if mode == "pooled":
        return regions
    if mode == "leave_one_region_out":
        return [region for region in regions if region != target_region]
    raise ValueError(f"Unknown transfer mode: {mode}")


def main() -> None:
    args = parse_args()
    OUT_DIR.mkdir(exist_ok=True)
    LANDSURFACE_DIR.mkdir(parents=True, exist_ok=True)
    regions = [resolve_region(region).slug for region in args.regions]

    frames = []
    for region in regions:
        print(f"Loading region frame: {region}", flush=True)
        frames.append(load_region_frame(region, args))
    data = pd.concat(frames, ignore_index=True)
    val_all = data[
        (data["target_year"] >= args.validation_start_year)
        & (data["target_year"] <= args.validation_end_year)
    ].copy()
    test_all = data[
        (data["target_year"] >= args.test_start_year)
        & (data["target_year"] <= args.test_end_year)
    ].copy()
    if val_all.empty or test_all.empty:
        raise ValueError("Domain-transfer benchmark has empty validation or test data.")
    event_thresholds = build_event_thresholds(val_all, args.event_quantiles)

    monthly_parts: list[pd.DataFrame] = []
    summary_parts: list[pd.DataFrame] = []
    rare_monthly_parts: list[pd.DataFrame] = []
    rare_summary_parts: list[pd.DataFrame] = []
    meta_rows: list[dict[str, object]] = []
    for target_region in regions:
        target_test = test_all.loc[test_all["region"].eq(target_region)].copy()
        if target_test.empty:
            raise ValueError(f"No test rows for target region={target_region}")
        for mode in ["local", "pooled", "leave_one_region_out"]:
            source_regions = transfer_source_regions(mode, target_region, regions)
            source_val = val_all.loc[val_all["region"].isin(source_regions)].copy()
            if source_val.empty:
                raise ValueError(f"No source validation rows for mode={mode}, target={target_region}")
            source_scored, target_scored, meta = fit_transfer_predictions(
                source_val,
                target_test,
                weight_steps=args.weight_steps,
                include_monotonic_xgb=not args.skip_monotonic_xgb,
            )
            target_scored = target_scored.copy()
            target_scored["target_region"] = target_region
            target_scored["transfer_mode"] = mode
            target_scored["source_regions"] = " ".join(source_regions)
            monthly_cols = [
                "target_time",
                "target_year",
                "target_month",
                "target_region",
                "transfer_mode",
                "source_regions",
                "y_true_dry_frac",
                "clim_prob_dry",
                "forecast_rzsm",
                "forecast_rzsm_anom",
                "persistence_raw_prob_dry",
                "gefs_transfer_selected_prob_dry",
                "persistence_transfer_selected_prob_dry",
                "stack_transfer_selected_prob_dry",
                "monotonic_xgb_transfer_prob_dry",
            ]
            monthly_parts.append(target_scored[[c for c in monthly_cols if c in target_scored.columns]].copy())
            summary = score_transfer(target_region, mode, source_regions, source_val, target_scored, meta, args)
            summary_parts.append(summary)
            rare_monthly, rare_summary = score_rare_events(
                target_region,
                mode,
                source_regions,
                source_scored,
                target_scored,
                event_thresholds,
                args,
            )
            if not rare_monthly.empty:
                rare_monthly_parts.append(rare_monthly)
            if not rare_summary.empty:
                rare_summary_parts.append(rare_summary)
            meta_rows.append(
                {
                    "target_region": target_region,
                    "transfer_mode": mode,
                    "source_regions": " ".join(source_regions),
                    **meta,
                }
            )
            stack = summary.loc[summary["model"].eq("stack_transfer_selected")].iloc[0]
            print(
                f"  target={target_region:<24} mode={mode:<20} "
                f"stack_BSS={float(stack['bss_vs_climatology']):+.3f} "
                f"deltaBS={float(stack['delta_bs_model_minus_persistence_transfer_selected']):+.4f} "
                f"status={stack['added_value_status']}",
                flush=True,
            )

    monthly = pd.concat(monthly_parts, ignore_index=True)
    summary = pd.concat(summary_parts, ignore_index=True)
    rare_monthly_all = pd.concat(rare_monthly_parts, ignore_index=True) if rare_monthly_parts else pd.DataFrame()
    rare_summary_all = pd.concat(rare_summary_parts, ignore_index=True) if rare_summary_parts else pd.DataFrame()
    meta = pd.DataFrame(meta_rows)

    outputs = {
        "monthly": OUT_DIR / f"{args.out_prefix}_monthly_scores.csv",
        "summary": OUT_DIR / f"{args.out_prefix}_summary.csv",
        "meta": OUT_DIR / f"{args.out_prefix}_meta.csv",
        "rare_monthly": OUT_DIR / f"{args.out_prefix}_rare_event_monthly_scores.csv",
        "rare_summary": OUT_DIR / f"{args.out_prefix}_rare_event_summary.csv",
    }
    monthly.to_csv(outputs["monthly"], index=False)
    summary.to_csv(outputs["summary"], index=False)
    meta.to_csv(outputs["meta"], index=False)
    rare_monthly_all.to_csv(outputs["rare_monthly"], index=False)
    rare_summary_all.to_csv(outputs["rare_summary"], index=False)

    text_path = OUT_DIR / f"{args.out_prefix}_summary.txt"
    stack = summary.loc[summary["model"].eq("stack_transfer_selected")].copy()
    lines = [
        "GEFSv12 Land-Surface Domain-Transfer Benchmark",
        "=" * 72,
        f"Regions: {' '.join(regions)}",
        f"Validation years: {args.validation_start_year}-{args.validation_end_year}",
        f"Test years: {args.test_start_year}-{args.test_end_year}",
        "",
    ]
    for mode, part in stack.groupby("transfer_mode", sort=False):
        robust = int((part["bss_ci_low"] > 0).sum())
        added = int(part["added_value_status"].eq("stack_robust_added_value").sum())
        lines.append(
            f"{mode}: robust-positive vs climatology {robust}/{len(part)}; "
            f"robust added value vs transferred persistence {added}/{len(part)}; "
            f"mean stack BSS {float(part['bss_vs_climatology'].mean()):+.3f}"
        )
    if not rare_summary_all.empty:
        lines.extend(["", "Rare-event diagnostics:"])
        for (mode, quantile, model), part in rare_summary_all.groupby(
            ["transfer_mode", "event_quantile", "model"], sort=False
        ):
            if model not in {"stack_transfer_selected", "monotonic_xgb_transfer"}:
                continue
            robust = int(part["event_claim_status"].eq("robust_positive").sum())
            mean_ap = float(part["average_precision"].mean())
            mean_lift = float(part["average_precision_lift_over_event_rate"].mean())
            lines.append(
                f"{mode} q{quantile:.2f} {model}: robust event-BSS {robust}/{len(part)}; "
                f"mean AP {mean_ap:.3f}; mean AP lift {mean_lift:.2f}x"
            )
    text_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(text_path.read_text(encoding="utf-8"))
    print(f"Wrote monthly scores: {outputs['monthly']} rows={len(monthly):,}")
    print(f"Wrote summary: {outputs['summary']} rows={len(summary):,}")
    print(f"Wrote metadata: {outputs['meta']} rows={len(meta):,}")
    print(f"Wrote rare-event monthly scores: {outputs['rare_monthly']} rows={len(rare_monthly_all):,}")
    print(f"Wrote rare-event summary: {outputs['rare_summary']} rows={len(rare_summary_all):,}")

    if args.copy_report:
        for path in list(outputs.values()) + [text_path]:
            shutil.copy2(path, LANDSURFACE_DIR / path.name)


if __name__ == "__main__":
    main()
