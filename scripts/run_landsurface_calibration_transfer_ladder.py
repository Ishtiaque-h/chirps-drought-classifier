#!/usr/bin/env python
"""Calibration-transfer ladder for GEFSv12 root-zone drought probabilities.

This diagnostic asks why source-product probabilities transfer poorly to the
SMAP L4 target. It separates four possibilities:

1. direct transfer from ERA5-Land/GLDAS calibrators;
2. validation-only SMAP base-rate adaptation;
3. validation-only SMAP recalibration of source probabilities; and
4. product-invariant calibration learned from ERA5-Land + GLDAS source targets.

The held-out score is always the SMAP L4 early/mid/late multi-snapshot
dry-fraction target for 2017-2019. SMAP test months are never used to fit
calibrators, choose stack weights, or estimate base-rate shifts.
"""

from __future__ import annotations

from argparse import ArgumentParser, Namespace
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
import xarray as xr
from sklearn.isotonic import IsotonicRegression

from region_config import resolve_region
from run_gefsv12_landsurface_stack_benchmark import (
    PROJECT_ROOT,
    added_value_status,
    bootstrap_delta_bs,
    choose_convex_weight,
    forecast_path,
    resolve_soil_file,
)
from run_landsurface_forecast_benchmark import (
    brier,
    bss,
    bootstrap_bss,
    observed_rootzone_target,
)


OUT_DIR = PROJECT_ROOT / "outputs"
REPORT_DIR = PROJECT_ROOT / "results" / "report"
LANDSURFACE_DIR = REPORT_DIR / "landsurface"
PROCESSED = PROJECT_ROOT / "data" / "processed"

EPS = 1.0e-5


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
    parser.add_argument("--sources", nargs="+", default=["era5_land", "gldas_noah"])
    parser.add_argument("--forecast-dir", type=Path, default=REPORT_DIR)
    parser.add_argument("--source-validation-start-year", type=int, default=2000)
    parser.add_argument("--source-validation-end-year", type=int, default=2016)
    parser.add_argument("--smap-validation-start-year", type=int, default=2015)
    parser.add_argument("--smap-validation-end-year", type=int, default=2016)
    parser.add_argument("--test-start-year", type=int, default=2017)
    parser.add_argument("--test-end-year", type=int, default=2019)
    parser.add_argument("--lead-months", type=int, default=1)
    parser.add_argument("--era5-normal-start-year", type=int, default=1991)
    parser.add_argument("--era5-normal-end-year", type=int, default=2016)
    parser.add_argument("--gldas-normal-start-year", type=int, default=2000)
    parser.add_argument("--gldas-normal-end-year", type=int, default=2016)
    parser.add_argument("--dry-quantile", type=float, default=0.20)
    parser.add_argument("--smap-dry-percentile", type=float, default=20.0)
    parser.add_argument("--weight-steps", type=int, default=101)
    parser.add_argument("--n-bootstrap", type=int, default=2000)
    parser.add_argument(
        "--out-prefix",
        default="landsurface_calibration_transfer_ladder_smap_multi3snapshot",
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


def logit(value: float) -> float:
    p = float(np.clip(value, EPS, 1.0 - EPS))
    return float(np.log(p / (1.0 - p)))


def expit(value: np.ndarray | float) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.asarray(value, dtype=float)))


def prior_shift(prob: pd.Series, source_mean: float, target_mean: float) -> pd.Series:
    offset = logit(target_mean) - logit(source_mean)
    shifted = expit(np.log(np.clip(prob.to_numpy(dtype=float), EPS, 1.0 - EPS) / np.clip(1.0 - prob.to_numpy(dtype=float), EPS, 1.0 - EPS)) + offset)
    return pd.Series(shifted, index=prob.index).clip(0.0, 1.0)


def fit_iso(train: pd.DataFrame, signal_col: str) -> IsotonicRegression:
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    iso.fit(train[signal_col].to_numpy(dtype=float), train["y_true_dry_frac"].to_numpy(dtype=float))
    return iso


def predict_iso(model: IsotonicRegression, frame: pd.DataFrame, signal_col: str) -> pd.Series:
    return pd.Series(
        model.predict(frame[signal_col].to_numpy(dtype=float)),
        index=frame.index,
    ).clip(0.0, 1.0)


def month_start_index(values: object) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(pd.to_datetime(values)).to_period("M").to_timestamp()


def add_climatology_and_persistence(
    observed: pd.DataFrame,
    validation_start_year: int,
    validation_end_year: int,
    lead_months: int,
    climatology_mode: str,
) -> pd.DataFrame:
    out = observed.sort_values("target_time").reset_index(drop=True).copy()
    val = out.loc[
        (out["target_year"] >= validation_start_year)
        & (out["target_year"] <= validation_end_year)
    ].copy()
    if val.empty:
        raise ValueError("Cannot define climatology without validation/normal rows.")
    if climatology_mode == "monthly":
        month_clim = val.groupby("target_month")["y_true_dry_frac"].mean()
        global_clim = float(val["y_true_dry_frac"].mean())
        out["clim_prob_dry"] = out["target_month"].map(month_clim).fillna(global_clim)
    else:
        out["clim_prob_dry"] = float(val["y_true_dry_frac"].mean())
    persistence = out[["target_time", "y_true_dry_frac"]].rename(
        columns={"target_time": "persistence_time", "y_true_dry_frac": "persistence_raw_prob_dry"}
    )
    out["persistence_time"] = (
        out["target_time"] - pd.DateOffset(months=lead_months)
    ).dt.to_period("M").dt.to_timestamp()
    out = out.merge(persistence, on="persistence_time", how="left")
    return out


def load_forecast(region_slug: str, forecast_dir: Path) -> pd.DataFrame:
    path = forecast_path(region_slug, forecast_dir)
    if not path.exists():
        raise FileNotFoundError(f"GEFSv12 forecast file not found for {region_slug}: {path}")
    forecast = pd.read_csv(path, parse_dates=["target_time", "valid_time", "init_time"])
    forecast["target_time"] = month_start_index(forecast["target_time"])
    forecast["gefs_raw_dry_signal"] = -forecast["forecast_rzsm"].astype(float)
    forecast["gefs_anom_dry_signal"] = -forecast["forecast_rzsm_anom"].astype(float)
    return forecast


def load_era5_target(region_slug: str, args: Namespace) -> pd.DataFrame:
    soil_file = resolve_soil_file(region_slug)
    obs_args = Namespace(
        soil_file=soil_file,
        lead_months=args.lead_months,
        normal_start_year=args.era5_normal_start_year,
        normal_end_year=args.era5_normal_end_year,
        dry_quantile=args.dry_quantile,
    )
    observed = observed_rootzone_target(obs_args)
    observed["target_time"] = month_start_index(observed["target_time"])
    observed["target_year"] = observed["target_time"].dt.year
    observed["target_month"] = observed["target_time"].dt.month
    observed["target_product"] = "era5_land"
    return observed


def load_gldas_target(region_slug: str, args: Namespace) -> pd.DataFrame:
    path = PROCESSED / f"gldas_noah_rootzone_monthly_{region_slug}_2000_2019.nc"
    if not path.exists():
        raise FileNotFoundError(f"Processed GLDAS target not found: {path}")
    ds = xr.open_dataset(path)
    try:
        da = ds["gldas_rootzone_sm"].load()
    finally:
        ds.close()
    years = pd.DatetimeIndex(da["time"].values).year
    normal_mask = (years >= args.gldas_normal_start_year) & (years <= args.gldas_normal_end_year)
    normal = da.isel(time=normal_mask)
    thresholds = normal.groupby("time.month").quantile(args.dry_quantile, dim="time", skipna=True)
    rows = []
    for i, target_time in enumerate(pd.DatetimeIndex(da["time"].values)):
        field = da.isel(time=i)
        threshold = thresholds.sel(month=int(target_time.month))
        valid = np.isfinite(field) & np.isfinite(threshold)
        dry = xr.where(valid, field <= threshold, np.nan)
        rows.append(
            {
                "target_time": pd.Timestamp(target_time).to_period("M").to_timestamp(),
                "target_year": int(target_time.year),
                "target_month": int(target_time.month),
                "y_true_dry_frac": float(dry.mean(dim=["latitude", "longitude"], skipna=True).values),
            }
        )
    observed = add_climatology_and_persistence(
        pd.DataFrame(rows),
        validation_start_year=args.gldas_normal_start_year,
        validation_end_year=args.gldas_normal_end_year,
        lead_months=args.lead_months,
        climatology_mode="monthly",
    )
    observed["target_product"] = "gldas_noah"
    return observed


def load_smap_target(region_slug: str, args: Namespace) -> pd.DataFrame:
    path = PROCESSED / f"smap_l4_rootzone_pctl_monthly_{region_slug}_2015_2019_multi3snapshot.nc"
    if not path.exists():
        raise FileNotFoundError(f"Processed SMAP multi-snapshot target not found: {path}")
    ds = xr.open_dataset(path)
    try:
        da = ds["smap_rootzone_pctl"].load()
    finally:
        ds.close()
    rows = []
    for i, target_time in enumerate(pd.DatetimeIndex(da["time"].values)):
        field = da.isel(time=i)
        valid = np.isfinite(field)
        dry = xr.where(valid, field <= args.smap_dry_percentile, np.nan)
        rows.append(
            {
                "target_time": pd.Timestamp(target_time).to_period("M").to_timestamp(),
                "target_year": int(target_time.year),
                "target_month": int(target_time.month),
                "y_true_dry_frac": float(dry.mean(dim=[d for d in field.dims], skipna=True).values),
            }
        )
    observed = add_climatology_and_persistence(
        pd.DataFrame(rows).dropna(subset=["y_true_dry_frac"]),
        validation_start_year=args.smap_validation_start_year,
        validation_end_year=args.smap_validation_end_year,
        lead_months=args.lead_months,
        climatology_mode="constant",
    )
    observed["target_product"] = "smap_l4_multi3snapshot"
    return observed


def merge_target_forecast(target: pd.DataFrame, forecast: pd.DataFrame) -> pd.DataFrame:
    merged = target.merge(forecast, on="target_time", how="inner", suffixes=("", "_forecast"))
    merged = merged.dropna(
        subset=[
            "y_true_dry_frac",
            "forecast_rzsm",
            "forecast_rzsm_anom",
            "gefs_raw_dry_signal",
            "gefs_anom_dry_signal",
            "persistence_raw_prob_dry",
        ]
    ).copy()
    if merged.empty:
        raise ValueError("No overlap between target observations and GEFSv12 forecast rows.")
    return merged.sort_values("target_time").reset_index(drop=True)


def split_years(frame: pd.DataFrame, start_year: int, end_year: int) -> pd.DataFrame:
    return frame.loc[
        (frame["target_year"] >= start_year)
        & (frame["target_year"] <= end_year)
    ].copy()


def fit_selected_calibration(
    train: pd.DataFrame,
    apply: pd.DataFrame,
    weight_steps: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    train = train.copy()
    apply = apply.copy()

    raw_iso = fit_iso(train, "gefs_raw_dry_signal")
    anom_iso = fit_iso(train, "gefs_anom_dry_signal")
    pers_iso = fit_iso(train, "persistence_raw_prob_dry")

    train["gefs_raw_isotonic_prob_dry"] = predict_iso(raw_iso, train, "gefs_raw_dry_signal")
    train["gefs_anom_isotonic_prob_dry"] = predict_iso(anom_iso, train, "gefs_anom_dry_signal")
    train["persistence_isotonic_prob_dry"] = predict_iso(pers_iso, train, "persistence_raw_prob_dry")
    apply["gefs_raw_isotonic_prob_dry"] = predict_iso(raw_iso, apply, "gefs_raw_dry_signal")
    apply["gefs_anom_isotonic_prob_dry"] = predict_iso(anom_iso, apply, "gefs_anom_dry_signal")
    apply["persistence_isotonic_prob_dry"] = predict_iso(pers_iso, apply, "persistence_raw_prob_dry")

    val_bs = {
        "gefs_raw_isotonic": brier(train["y_true_dry_frac"], train["gefs_raw_isotonic_prob_dry"]),
        "gefs_anom_isotonic": brier(train["y_true_dry_frac"], train["gefs_anom_isotonic_prob_dry"]),
        "persistence_raw": brier(train["y_true_dry_frac"], train["persistence_raw_prob_dry"]),
        "persistence_isotonic": brier(train["y_true_dry_frac"], train["persistence_isotonic_prob_dry"]),
    }
    gefs_best = min({k: v for k, v in val_bs.items() if k.startswith("gefs_")}, key=val_bs.get)
    persistence_best = min({k: v for k, v in val_bs.items() if k.startswith("persistence_")}, key=val_bs.get)

    for frame in [train, apply]:
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
        train,
        "gefs_selected_prob_dry",
        "persistence_selected_prob_dry",
        weight_steps,
    )
    for frame in [train, apply]:
        frame["stack_validation_selected_prob_dry"] = (
            weight * frame["gefs_selected_prob_dry"]
            + (1.0 - weight) * frame["persistence_selected_prob_dry"]
        ).clip(0.0, 1.0)

    metadata = {
        "selected_gefs_calibration": gefs_best,
        "selected_persistence_calibration": persistence_best,
        "selected_weight_gefs": float(weight),
        "selected_weight_persistence": float(1.0 - weight),
        "selected_weight_val_bs": float(weight_val_bs),
        "validation_bs_gefs_selected": float(
            val_bs["gefs_raw_isotonic" if gefs_best == "gefs_raw_isotonic" else "gefs_anom_isotonic"]
        ),
        "validation_bs_persistence_selected": float(val_bs[persistence_best]),
    }
    return train, apply, metadata


def build_target_persistence_reference(
    smap_merged: pd.DataFrame,
    args: Namespace,
) -> tuple[pd.DataFrame, dict[str, object]]:
    val = split_years(smap_merged, args.smap_validation_start_year, args.smap_validation_end_year)
    test = split_years(smap_merged, args.test_start_year, args.test_end_year)
    _, pred, meta = fit_selected_calibration(val, pd.concat([val, test], ignore_index=True), args.weight_steps)
    keep = pred[["target_time", "persistence_selected_prob_dry"]].rename(
        columns={"persistence_selected_prob_dry": "smap_persistence_selected_prob_dry"}
    )
    return keep, meta


def source_direct_predictions(
    source_product: str,
    source_merged: pd.DataFrame,
    apply_merged: pd.DataFrame,
    args: Namespace,
) -> tuple[pd.DataFrame, dict[str, object]]:
    source_train = split_years(
        source_merged,
        args.source_validation_start_year,
        args.source_validation_end_year,
    )
    _, pred, meta = fit_selected_calibration(source_train, apply_merged, args.weight_steps)
    return pred, meta


def model_columns() -> dict[str, str]:
    return {
        "gefs_selected": "gefs_selected_prob_dry",
        "source_persistence_selected": "persistence_selected_prob_dry",
        "stack_validation_selected": "stack_validation_selected_prob_dry",
    }


def append_strategy_rows(
    rows: list[pd.DataFrame],
    smap_base: pd.DataFrame,
    predictions: pd.DataFrame,
    strategy: str,
    source_product: str,
    source_model_label_prefix: str,
    columns: dict[str, str],
    metadata: dict[str, object],
) -> None:
    base_cols = [
        "region",
        "target_time",
        "target_year",
        "target_month",
        "y_true_dry_frac",
        "smap_clim_prob_dry",
        "smap_persistence_selected_prob_dry",
    ]
    for model, col in columns.items():
        if col not in predictions.columns:
            continue
        part = smap_base[base_cols].merge(
            predictions[["target_time", col]].rename(columns={col: "prob_dry"}),
            on="target_time",
            how="inner",
        )
        part = part.dropna(subset=["prob_dry"])
        if part.empty:
            continue
        part["strategy"] = strategy
        part["source_product"] = source_product
        part["model"] = model
        part["model_label"] = f"{source_model_label_prefix}{model}"
        for key, value in metadata.items():
            part[key] = value
        rows.append(part)


def build_region_rows(
    region: str,
    source_targets: dict[str, pd.DataFrame],
    smap_target: pd.DataFrame,
    forecast: pd.DataFrame,
    pooled_meta: dict[str, object] | None,
    pooled_models: dict[str, object] | None,
    args: Namespace,
) -> list[pd.DataFrame]:
    rows: list[pd.DataFrame] = []
    smap_merged = merge_target_forecast(smap_target, forecast)
    smap_persistence_ref, smap_ref_meta = build_target_persistence_reference(smap_merged, args)
    smap_base = smap_merged.merge(smap_persistence_ref, on="target_time", how="left")
    smap_base = smap_base.rename(columns={"clim_prob_dry": "smap_clim_prob_dry"})

    # Self-calibrated SMAP reference.
    smap_val = split_years(smap_merged, args.smap_validation_start_year, args.smap_validation_end_year)
    smap_apply = split_years(smap_merged, args.smap_validation_start_year, args.test_end_year)
    _, smap_pred, smap_meta = fit_selected_calibration(smap_val, smap_apply, args.weight_steps)
    append_strategy_rows(
        rows,
        smap_base,
        smap_pred,
        strategy="smap_self_calibrated_reference",
        source_product="smap_l4_multi3snapshot",
        source_model_label_prefix="smap_",
        columns={
            "gefs_selected": "gefs_selected_prob_dry",
            "target_persistence_selected": "persistence_selected_prob_dry",
            "stack_validation_selected": "stack_validation_selected_prob_dry",
        },
        metadata={**smap_meta, **{"smap_persistence_calibration": smap_ref_meta["selected_persistence_calibration"]}},
    )

    for source_product, source_target in source_targets.items():
        source_merged = merge_target_forecast(source_target, forecast)
        source_apply = source_merged.loc[
            (source_merged["target_year"] >= args.smap_validation_start_year)
            & (source_merged["target_year"] <= args.test_end_year)
        ].copy()
        source_pred, source_meta = source_direct_predictions(
            source_product,
            source_merged,
            source_apply,
            args,
        )

        # Direct transfer.
        append_strategy_rows(
            rows,
            smap_base,
            source_pred,
            strategy="source_direct",
            source_product=source_product,
            source_model_label_prefix="",
            columns=model_columns(),
            metadata=source_meta,
        )

        # Source-to-SMAP base-rate shift, estimated only on overlapping SMAP validation months.
        source_overlap = source_merged.loc[
            (source_merged["target_year"] >= args.smap_validation_start_year)
            & (source_merged["target_year"] <= args.smap_validation_end_year)
        ].copy()
        smap_overlap = smap_merged.loc[
            (smap_merged["target_year"] >= args.smap_validation_start_year)
            & (smap_merged["target_year"] <= args.smap_validation_end_year)
        ].copy()
        source_mean = float(source_overlap["y_true_dry_frac"].mean())
        target_mean = float(smap_overlap["y_true_dry_frac"].mean())
        shifted = source_pred.copy()
        for col in model_columns().values():
            if col in shifted.columns:
                shifted[col] = prior_shift(shifted[col], source_mean=source_mean, target_mean=target_mean)
        append_strategy_rows(
            rows,
            smap_base,
            shifted,
            strategy="target_prior_shift",
            source_product=source_product,
            source_model_label_prefix="prior_shift_",
            columns=model_columns(),
            metadata={
                **source_meta,
                "source_overlap_mean_dry": source_mean,
                "smap_validation_mean_dry": target_mean,
            },
        )

        # SMAP validation-only recalibration of source probabilities.
        recalibrated = source_pred.copy()
        for model, col in model_columns().items():
            if col not in recalibrated.columns:
                continue
            val_join = smap_merged[["target_time", "y_true_dry_frac"]].merge(
                source_pred[["target_time", col]],
                on="target_time",
                how="inner",
            )
            val_join = val_join.loc[
                (val_join["target_time"].dt.year >= args.smap_validation_start_year)
                & (val_join["target_time"].dt.year <= args.smap_validation_end_year)
            ].dropna()
            if len(val_join) < 8:
                continue
            iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
            iso.fit(val_join[col].to_numpy(dtype=float), val_join["y_true_dry_frac"].to_numpy(dtype=float))
            recalibrated[col] = pd.Series(
                iso.predict(recalibrated[col].to_numpy(dtype=float)),
                index=recalibrated.index,
            ).clip(0.0, 1.0)
        append_strategy_rows(
            rows,
            smap_base,
            recalibrated,
            strategy="smap_validation_recalibration",
            source_product=source_product,
            source_model_label_prefix="smap_recalibrated_",
            columns=model_columns(),
            metadata={**source_meta, "smap_recalibration_months": int(len(smap_overlap))},
        )

    if pooled_models and pooled_meta:
        pooled = smap_merged.copy()
        raw_iso = pooled_models["gefs_raw_isotonic"]
        anom_iso = pooled_models["gefs_anom_isotonic"]
        pers_iso = pooled_models["persistence_isotonic"]
        pooled["pooled_gefs_raw_isotonic_prob_dry"] = predict_iso(raw_iso, pooled, "gefs_raw_dry_signal")
        pooled["pooled_gefs_anom_isotonic_prob_dry"] = predict_iso(anom_iso, pooled, "gefs_anom_dry_signal")
        pooled["pooled_persistence_isotonic_prob_dry"] = predict_iso(pers_iso, pooled, "persistence_raw_prob_dry")
        pooled["pooled_gefs_selected_prob_dry"] = (
            pooled["pooled_gefs_raw_isotonic_prob_dry"]
            if pooled_meta["selected_gefs_calibration"] == "gefs_raw_isotonic"
            else pooled["pooled_gefs_anom_isotonic_prob_dry"]
        )
        pooled["target_persistence_source_calibrated_prob_dry"] = (
            pooled["persistence_raw_prob_dry"]
            if pooled_meta["selected_persistence_calibration"] == "persistence_raw"
            else pooled["pooled_persistence_isotonic_prob_dry"]
        )
        w = float(pooled_meta["selected_weight_gefs"])
        pooled["pooled_stack_selected_prob_dry"] = (
            w * pooled["pooled_gefs_selected_prob_dry"]
            + (1.0 - w) * pooled["target_persistence_source_calibrated_prob_dry"]
        ).clip(0.0, 1.0)
        append_strategy_rows(
            rows,
            smap_base,
            pooled,
            strategy="pooled_source_calibration",
            source_product="era5_land_plus_gldas_noah",
            source_model_label_prefix="pooled_",
            columns={
                "gefs_selected": "pooled_gefs_selected_prob_dry",
                "target_persistence_selected": "target_persistence_source_calibrated_prob_dry",
                "stack_validation_selected": "pooled_stack_selected_prob_dry",
            },
            metadata=pooled_meta,
        )

    return rows


def fit_pooled_source_models(
    all_source_merged: list[pd.DataFrame],
    args: Namespace,
) -> tuple[dict[str, object], dict[str, object]]:
    train = pd.concat(
        [
            split_years(frame, args.source_validation_start_year, args.source_validation_end_year)
            for frame in all_source_merged
        ],
        ignore_index=True,
    ).dropna(
        subset=[
            "y_true_dry_frac",
            "gefs_raw_dry_signal",
            "gefs_anom_dry_signal",
            "persistence_raw_prob_dry",
        ]
    )
    raw_iso = fit_iso(train, "gefs_raw_dry_signal")
    anom_iso = fit_iso(train, "gefs_anom_dry_signal")
    pers_iso = fit_iso(train, "persistence_raw_prob_dry")
    train = train.copy()
    train["gefs_raw_isotonic_prob_dry"] = predict_iso(raw_iso, train, "gefs_raw_dry_signal")
    train["gefs_anom_isotonic_prob_dry"] = predict_iso(anom_iso, train, "gefs_anom_dry_signal")
    train["persistence_isotonic_prob_dry"] = predict_iso(pers_iso, train, "persistence_raw_prob_dry")
    val_bs = {
        "gefs_raw_isotonic": brier(train["y_true_dry_frac"], train["gefs_raw_isotonic_prob_dry"]),
        "gefs_anom_isotonic": brier(train["y_true_dry_frac"], train["gefs_anom_isotonic_prob_dry"]),
        "persistence_raw": brier(train["y_true_dry_frac"], train["persistence_raw_prob_dry"]),
        "persistence_isotonic": brier(train["y_true_dry_frac"], train["persistence_isotonic_prob_dry"]),
    }
    gefs_best = min({k: v for k, v in val_bs.items() if k.startswith("gefs_")}, key=val_bs.get)
    persistence_best = min({k: v for k, v in val_bs.items() if k.startswith("persistence_")}, key=val_bs.get)
    train["gefs_selected_prob_dry"] = (
        train["gefs_raw_isotonic_prob_dry"]
        if gefs_best == "gefs_raw_isotonic"
        else train["gefs_anom_isotonic_prob_dry"]
    )
    train["persistence_selected_prob_dry"] = (
        train["persistence_raw_prob_dry"]
        if persistence_best == "persistence_raw"
        else train["persistence_isotonic_prob_dry"]
    )
    weight, weight_val_bs = choose_convex_weight(
        train,
        "gefs_selected_prob_dry",
        "persistence_selected_prob_dry",
        args.weight_steps,
    )
    models = {
        "gefs_raw_isotonic": raw_iso,
        "gefs_anom_isotonic": anom_iso,
        "persistence_isotonic": pers_iso,
    }
    meta = {
        "selected_gefs_calibration": gefs_best,
        "selected_persistence_calibration": persistence_best,
        "selected_weight_gefs": float(weight),
        "selected_weight_persistence": float(1.0 - weight),
        "selected_weight_val_bs": float(weight_val_bs),
        "pooled_source_training_rows": int(len(train)),
        "validation_bs_gefs_selected": float(
            val_bs["gefs_raw_isotonic" if gefs_best == "gefs_raw_isotonic" else "gefs_anom_isotonic"]
        ),
        "validation_bs_persistence_selected": float(val_bs[persistence_best]),
    }
    return models, meta


def score_monthly(monthly: pd.DataFrame, args: Namespace) -> pd.DataFrame:
    rows = []
    group_cols = ["strategy", "source_product", "region", "model"]
    for i, (keys, group) in enumerate(monthly.groupby(group_cols, sort=True), start=1):
        test = split_years(group, args.test_start_year, args.test_end_year)
        if test.empty:
            continue
        scoring = test.rename(
            columns={
                "smap_clim_prob_dry": "clim_prob_dry",
                "smap_persistence_selected_prob_dry": "persistence_selected_prob_dry",
            }
        ).copy()
        y = scoring["y_true_dry_frac"].to_numpy(dtype=float)
        pred = scoring["prob_dry"].to_numpy(dtype=float)
        clim = scoring["clim_prob_dry"].to_numpy(dtype=float)
        persistence = scoring["persistence_selected_prob_dry"].to_numpy(dtype=float)
        ci_low, ci_high = bootstrap_bss(
            scoring,
            pred_col="prob_dry",
            ref_col="clim_prob_dry",
            n_bootstrap=args.n_bootstrap,
            seed=12001 + i,
        )
        pers_ci_low, pers_ci_high = bootstrap_bss(
            scoring,
            pred_col="prob_dry",
            ref_col="persistence_selected_prob_dry",
            n_bootstrap=args.n_bootstrap,
            seed=13001 + i,
        )
        delta = brier(y, pred) - brier(y, persistence)
        delta_low, delta_high = bootstrap_delta_bs(
            scoring,
            candidate_col="prob_dry",
            reference_col="persistence_selected_prob_dry",
            n_bootstrap=args.n_bootstrap,
            seed=14001 + i,
        )
        row = dict(zip(group_cols, keys, strict=True))
        row.update(
            {
                "n_test_months": int(len(scoring)),
                "bs_smap_climatology": brier(y, clim),
                "bs_smap_persistence_selected": brier(y, persistence),
                "bs_model": brier(y, pred),
                "bss_vs_smap_climatology": bss(y, pred, clim),
                "bss_vs_smap_climatology_ci_low": ci_low,
                "bss_vs_smap_climatology_ci_high": ci_high,
                "bss_vs_smap_persistence_selected": bss(y, pred, persistence),
                "bss_vs_smap_persistence_selected_ci_low": pers_ci_low,
                "bss_vs_smap_persistence_selected_ci_high": pers_ci_high,
                "delta_bs_model_minus_smap_persistence_selected": delta,
                "delta_bs_ci_low": delta_low,
                "delta_bs_ci_high": delta_high,
                "claim_status_vs_smap_climatology": claim_status(
                    bss(y, pred, clim), ci_low, ci_high
                ),
                "added_value_status_vs_smap_persistence": added_value_status(
                    delta, delta_low, delta_high
                ),
                "spearman_model_vs_smap_observed": scoring["prob_dry"].corr(
                    scoring["y_true_dry_frac"], method="spearman"
                ),
                "amplitude_ratio_model_vs_smap_observed": float(np.std(pred, ddof=0) / np.std(y, ddof=0))
                if float(np.std(y, ddof=0)) > 0
                else np.nan,
            }
        )
        for col in [
            "selected_gefs_calibration",
            "selected_persistence_calibration",
            "selected_weight_gefs",
            "selected_weight_persistence",
            "selected_weight_val_bs",
            "source_overlap_mean_dry",
            "smap_validation_mean_dry",
            "pooled_source_training_rows",
            "smap_recalibration_months",
        ]:
            row[col] = group[col].dropna().iloc[0] if col in group.columns and not group[col].dropna().empty else np.nan
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["strategy", "source_product", "region", "model"]).reset_index(drop=True)


def write_summary_text(summary: pd.DataFrame, path: Path) -> None:
    lines = [
        "Land-surface calibration-transfer ladder against SMAP L4 multi-snapshot target",
        "",
    ]
    compact = summary.loc[summary["model"].isin(["gefs_selected", "stack_validation_selected"])].copy()
    if not compact.empty:
        agg = (
            compact.groupby(["strategy", "model"])
            .agg(
                n_rows=("region", "size"),
                robust_positive=("claim_status_vs_smap_climatology", lambda s: int((s == "robust_positive").sum())),
                robust_added_value=("added_value_status_vs_smap_persistence", lambda s: int((s == "stack_robust_added_value").sum())),
                mean_bss=("bss_vs_smap_climatology", "mean"),
                mean_bss_vs_persistence=("bss_vs_smap_persistence_selected", "mean"),
            )
            .reset_index()
        )
        lines.append(agg.to_string(index=False))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    regions = [resolve_region(region).slug for region in args.regions]
    sources = list(dict.fromkeys(args.sources))
    unsupported_sources = sorted(set(sources) - {"era5_land", "gldas_noah"})
    if unsupported_sources:
        raise SystemExit(f"Unsupported sources: {unsupported_sources}")

    target_cache: dict[str, dict[str, pd.DataFrame]] = {}
    forecast_cache: dict[str, pd.DataFrame] = {}
    source_merged_for_pool: list[pd.DataFrame] = []
    for region in regions:
        forecast = load_forecast(region, args.forecast_dir)
        forecast_cache[region] = forecast
        source_targets = {}
        if "era5_land" in sources:
            source_targets["era5_land"] = load_era5_target(region, args)
        if "gldas_noah" in sources:
            source_targets["gldas_noah"] = load_gldas_target(region, args)
        smap_target = load_smap_target(region, args)
        target_cache[region] = {**source_targets, "smap_l4_multi3snapshot": smap_target}
        for source_product, target in source_targets.items():
            merged = merge_target_forecast(target, forecast)
            merged["source_product"] = source_product
            merged["region"] = region
            source_merged_for_pool.append(merged)

    pooled_models, pooled_meta = fit_pooled_source_models(source_merged_for_pool, args)

    monthly_parts = []
    for region in regions:
        print(f"Building calibration-transfer ladder for {region}", flush=True)
        source_targets = {
            source: target_cache[region][source]
            for source in sources
        }
        monthly_parts.extend(
            build_region_rows(
                region=region,
                source_targets=source_targets,
                smap_target=target_cache[region]["smap_l4_multi3snapshot"],
                forecast=forecast_cache[region],
                pooled_meta=pooled_meta,
                pooled_models=pooled_models,
                args=args,
            )
        )
    monthly = pd.concat(monthly_parts, ignore_index=True)
    summary = score_monthly(monthly, args)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    monthly_path = OUT_DIR / f"{args.out_prefix}_monthly_scores.csv"
    summary_path = OUT_DIR / f"{args.out_prefix}_summary.csv"
    text_path = OUT_DIR / f"{args.out_prefix}_summary.txt"
    monthly.to_csv(monthly_path, index=False)
    summary.to_csv(summary_path, index=False)
    write_summary_text(summary, text_path)

    if args.copy_report:
        LANDSURFACE_DIR.mkdir(parents=True, exist_ok=True)
        for path in [monthly_path, summary_path, text_path]:
            shutil.copy2(path, LANDSURFACE_DIR / path.name)

    print(f"Wrote {summary_path}")
    print(f"Wrote {monthly_path}")
    print(f"Wrote {text_path}")
    if args.copy_report:
        print(f"Copied outputs to {LANDSURFACE_DIR}")
    compact_cols = [
        "strategy",
        "source_product",
        "region",
        "model",
        "n_test_months",
        "bss_vs_smap_climatology",
        "bss_vs_smap_climatology_ci_low",
        "bss_vs_smap_climatology_ci_high",
        "bss_vs_smap_persistence_selected",
        "added_value_status_vs_smap_persistence",
        "claim_status_vs_smap_climatology",
    ]
    print(summary[compact_cols].to_string(index=False))


if __name__ == "__main__":
    main()
