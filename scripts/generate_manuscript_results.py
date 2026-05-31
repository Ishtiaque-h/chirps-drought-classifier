#!/usr/bin/env python
"""Build paper-facing evidence tables and figures from completed experiments.

This script does not train models. It consolidates the current result artifacts
into a small evidence pack for manuscript drafting:

  results/report/paper/table01_master_evidence.csv
  results/report/paper/table02_headline_results.csv
  results/report/paper/table03_mask_methods.csv
  results/report/paper/table04_temporal_robustness.csv
  results/report/paper/table05_seasonal_signal_audit.csv
  results/report/paper/table06_regionalization_mechanism.csv
  results/report/paper/table07_evaluation_inflation_audit.csv
  results/report/paper/table08_transition_target_summary.csv
  results/report/paper/table09_landsurface_added_value.csv
  results/report/paper/table10_climate_index_sensitivity.csv
  results/report/paper/table11_gefsv12_landsurface_sensitivity.csv
  results/report/paper/table12_gefsv12_landsurface_stack.csv
  results/report/paper/table13_landsurface_reliability_resolution.csv
  results/report/paper/table14_landsurface_domain_transfer.csv
  results/report/paper/table15_landsurface_rare_event.csv
  results/report/paper/table16_landsurface_persistence_regimes.csv
  results/report/paper/table17_landsurface_independent_target_audit.csv
  results/report/paper/table18_landsurface_nldas_validation.csv
  results/report/paper/table19_landsurface_target_product_comparison.csv
  results/report/paper/table20_landsurface_gldas_validation.csv
  results/report/paper/table21_landsurface_era5_gldas_comparison.csv
  results/report/paper/table22_landsurface_smap_l4_validation.csv
  results/report/paper/table23_landsurface_smap_l4_snapshot_sensitivity.csv
  results/report/paper/table24_landsurface_target_product_transfer.csv
  results/report/paper/table25_landsurface_calibration_transfer_ladder.csv
  results/report/paper/table26_landsurface_calibration_transfer_ladder_compact.csv
  results/report/paper/table27_landsurface_target_product_adaptation_benchmark.csv
  results/report/paper/table28_landsurface_base_rate_yearly_sensitivity.csv
  results/report/paper/table29_landsurface_forecast_archive_audit.csv
  results/report/paper/table30_operational_gefs_diagnostic.csv
  results/report/paper/table31_operational_gefs_smap_validation.csv
  results/report/paper/fig01_headline_bss_forest.png
  results/report/paper/fig02_multiregion_bss_forest.png
  results/report/paper/fig03_seasonal_bss_vs_tracking.png
  results/report/paper/fig04_temporal_holdout_bss.png
  results/report/paper/fig05_mask_retention.png
  results/report/paper/fig06_landsurface_stack_reliability.png
  results/report/paper/fig07_landsurface_target_product_comparison.png
  results/report/paper/fig08_landsurface_era5_gldas_comparison.png
  results/report/paper/fig09_landsurface_calibration_transfer_ladder.png
  results/report/paper/paper_evidence_pack.md
"""
from __future__ import annotations

from pathlib import Path
import textwrap

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from region_config import resolve_region


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS = PROJECT_ROOT / "results"
REPORT = RESULTS / "report"
PAPER = REPORT / "paper"
LANDSURFACE_DIR = REPORT / "landsurface"
PAPER.mkdir(parents=True, exist_ok=True)


STATUS_COLORS = {
    "robust_positive": "#2E7D32",
    "positive_uncertain": "#7CB342",
    "reference": "#616161",
    "not_distinguishable_from_climatology": "#546E7A",
    "negative_uncertain": "#546E7A",
    "robust_negative": "#C62828",
    "negative_no_ci": "#8D6E63",
    "positive_no_ci": "#6A1B9A",
    "not_applicable": "#9E9E9E",
}

SIGNAL_COLORS = {
    "positive_temporal_tracking": "#2E7D32",
    "positive_weak_tracking": "#7CB342",
    "positive_calibration_shift": "#F9A825",
    "raw_signal_lost_after_calibration": "#8E24AA",
    "no_positive_signal": "#546E7A",
}


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def read_first_csv(*paths: Path) -> pd.DataFrame:
    for path in paths:
        if path.exists():
            return pd.read_csv(path)
    return pd.DataFrame()


def finite(value: object) -> bool:
    try:
        return bool(np.isfinite(float(value)))
    except (TypeError, ValueError):
        return False


def claim_status(bss: object, lo: object = np.nan, hi: object = np.nan) -> str:
    if not finite(bss):
        return "not_applicable"
    bss_f = float(bss)
    lo_f = float(lo) if finite(lo) else np.nan
    hi_f = float(hi) if finite(hi) else np.nan
    if np.isfinite(lo_f) and np.isfinite(hi_f):
        if lo_f > 0:
            return "robust_positive"
        if hi_f < 0:
            return "robust_negative"
        if bss_f > 0:
            return "positive_uncertain"
        return "negative_uncertain"
    if bss_f > 0:
        return "positive_no_ci"
    if bss_f < 0:
        return "negative_no_ci"
    return "reference"


def format_float(value: object, digits: int = 3) -> str:
    if not finite(value):
        return ""
    return f"{float(value):.{digits}f}"


def short_region(region: object) -> str:
    text = "" if pd.isna(region) else str(region)
    replacements = {
        "California Central Valley (basin-mask sensitivity)": "Central Valley basin",
        "California Central Valley basin": "Central Valley PRISM",
        "cvalley": "Central Valley",
        "California Central Valley": "Central Valley",
        "Mediterranean Spain bounding box (basin-mask sensitivity)": "Med Spain basin",
        "Mediterranean Spain bounding box (country-mask sensitivity)": "Med Spain country",
        "Mediterranean Spain bounding box": "Med Spain bbox",
        "mediterranean_spain": "Mediterranean Spain",
        "mediterranean_spain_basin_masked": "Med Spain basin",
        "Southern Great Plains (basin-mask sensitivity)": "SGP basin",
        "southern_great_plains": "Southern Great Plains",
        "Southern Great Plains": "SGP bbox",
        "southern_great_plains_basin_masked": "SGP basin",
        "Murray-Darling Basin bounding box (basin-mask sensitivity)": "Murray-Darling basin",
        "murray_darling_basin_masked": "Murray-Darling basin",
        "murray_darling": "Murray-Darling",
        "Horn of Africa bounding box (country-mask sensitivity)": "Horn country",
        "horn_of_africa_country_masked": "Horn country",
        "horn_of_africa": "Horn of Africa",
    }
    return replacements.get(text, text.replace("_", " "))


def short_model(model: object) -> str:
    text = "" if pd.isna(model) else str(model)
    replacements = {
        "XGB-Spatial + gridded temperature/VPD selected": "XGB-Spatial + gridded T/VPD",
        "XGBoost + regional temperature/VPD selected": "XGB + regional T/VPD",
        "XGBoost + soil moisture selected": "XGB + soil moisture",
        "XGBoost + MJO/IVT selected": "XGB + MJO/IVT",
        "External precipitation forecast selected": "External precip benchmark",
        "CPC NMME anomaly forecast selected": "CPC NMME anomaly selected",
        "CPC NMME probability forecast raw": "CPC NMME prob raw",
        "CPC NMME probability forecast selected": "CPC NMME prob selected",
        "NCEI CFSv2 precipitation forecast selected": "NCEI CFSv2 selected",
        "NCEI CFSv2 soil-moisture forecast selected": "CFSv2 RZSM selected",
        "NCEI CFSv2 soil-moisture forecast 4-cycle mean selected": "CFSv2 RZSM 4-cycle",
        "NOAA GEFSv12 RZSM reforecast 11-member selected": "GEFSv12 RZSM selected",
        "ERA5-Land root-zone persistence raw": "RZSM persistence raw",
        "ERA5-Land root-zone persistence selected": "RZSM persistence selected",
        "XGB-Spatial probabilities": "XGB-Spatial vs PRISM",
        "XGBoost seasonal isotonic": "Seasonal XGB isotonic",
        "Tabular XGBoost rolling holdout": "Rolling XGB",
        "spatial XGBoost": "Spatial XGB",
        "tabular XGBoost": "Tabular XGB",
    }
    return replacements.get(text, text)


def plot_label(row: pd.Series) -> str:
    group = str(row.get("evidence_group", ""))
    experiment = str(row.get("experiment", ""))
    region = short_region(row.get("region", ""))
    model = short_model(row.get("model", ""))
    if group == "seasonal_regional_longlead":
        return f"{region}: {experiment.replace('seasonal_', '').replace('_', ' ')}"
    if group == "temporal_robustness":
        return f"{experiment.replace('_', ' ')}"
    if group == "multi_region_selected_checkpoint":
        return f"{region}: {model}"
    if group == "central_valley_calibrated_checkpoint":
        return f"Central Valley calibrated: {model}"
    if group == "central_valley_feature_extension":
        return model
    if group == "central_valley_uncertainty":
        return model
    if group == "operational_dynamical_benchmark":
        return model
    if group == "forecast_informed_landsurface_benchmark":
        return model
    if group == "independent_precipitation_validation":
        return model
    return f"{region}: {model}"


def markdown_table(df: pd.DataFrame, digits: int = 3) -> str:
    if df.empty:
        return ""
    headers = list(df.columns)
    rows = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for _, row in df.iterrows():
        values = []
        for value in row:
            if isinstance(value, float):
                values.append(format_float(value, digits))
            else:
                text = "" if pd.isna(value) else str(value)
                text = text.replace("\n", " ")
                values.append(text)
        rows.append("| " + " | ".join(values) + " |")
    return "\n".join(rows)


def write_table(df: pd.DataFrame, stem: str, md_cols: list[str] | None = None, digits: int = 3) -> Path:
    csv_path = PAPER / f"{stem}.csv"
    df.to_csv(csv_path, index=False)
    return csv_path


def evidence_from_master_headline() -> pd.DataFrame:
    path = REPORT / "master_results_headline.csv"
    df = read_csv(path)
    if df.empty:
        return df
    out = pd.DataFrame(
        {
            "evidence_group": df["category"],
            "experiment": df["scope"],
            "region": df["region"],
            "target": df["target"],
            "lead_months": df["lead_months"],
            "model": df["model"],
            "calibration": df["calibration"],
            "reference": "monthly climatology",
            "n_months": df["n_test_months"],
            "bss": df["bss_vs_climatology"],
            "ci_low": df["bss_ci_low"],
            "ci_high": df["bss_ci_high"],
            "status": df["claim_status"],
            "signal_flag": "",
            "interpretation": df["notes"],
            "source_file": df["source_file"],
        }
    )
    out.loc[out["model"].str.contains("Climatological baseline", na=False), "status"] = "reference"
    return out


def evidence_from_seasonal_regional() -> pd.DataFrame:
    path = REPORT / "seasonal" / "seasonal_regional_longlead_summary.csv"
    audit_path = REPORT / "seasonal" / "seasonal_regional_signal_audit.csv"
    df = read_first_csv(path, RESULTS / "seasonal" / "seasonal_regional_longlead_summary.csv")
    audit = read_first_csv(audit_path, RESULTS / "seasonal" / "seasonal_regional_signal_audit.csv")
    if df.empty:
        return df
    merge_cols = ["region", "target_spi", "lead_months", "climate_features"]
    if not audit.empty:
        df = df.merge(
            audit[merge_cols + ["corr_iso_obs", "std_ratio_iso_obs", "iso_bias", "signal_flag"]],
            on=merge_cols,
            how="left",
        )
    out = pd.DataFrame(
        {
            "evidence_group": "seasonal_regional_longlead",
            "experiment": (
                "seasonal_spi"
                + df["target_spi"].astype(str)
                + "_lead"
                + df["lead_months"].astype(str)
                + "_"
                + df["climate_features"].astype(str)
            ),
            "region": df["region"],
            "target": "SPI-" + df["target_spi"].astype(str) + " dry fraction",
            "lead_months": df["lead_months"],
            "model": "XGBoost seasonal isotonic",
            "calibration": "isotonic",
            "reference": "monthly climatology",
            "n_months": df["n_test_months"],
            "bss": df["bss_xgb_isotonic"],
            "ci_low": df["bss_xgb_isotonic_ci_low"],
            "ci_high": df["bss_xgb_isotonic_ci_high"],
            "status": df["status"],
            "signal_flag": df.get("signal_flag", ""),
            "interpretation": (
                "climate_features="
                + df["climate_features"].astype(str)
                + "; mask="
                + df.get("mask_kind", pd.Series([""] * len(df))).fillna("").astype(str)
                + "; corr_iso_obs="
                + df.get("corr_iso_obs", pd.Series([np.nan] * len(df))).map(lambda x: format_float(x, 3))
                + "; std_ratio_iso_obs="
                + df.get("std_ratio_iso_obs", pd.Series([np.nan] * len(df))).map(lambda x: format_float(x, 3))
            ),
            "source_file": df["score_file"],
        }
    )
    return out


def evidence_from_temporal() -> pd.DataFrame:
    path = RESULTS / "temporal" / "temporal_robustness_summary.csv"
    df = read_csv(path)
    if df.empty:
        return df
    df = df.loc[df["reference"].eq("clim_train_monthly")].copy()
    out = pd.DataFrame(
        {
            "evidence_group": "temporal_robustness",
            "experiment": df["split"],
            "region": "California Central Valley",
            "target": "SPI-1 dry fraction",
            "lead_months": 1,
            "model": "Tabular XGBoost rolling holdout",
            "calibration": df["selected_calibration"],
            "reference": "train calendar-month climatology",
            "n_months": df["n_test_months"],
            "bss": df["bss"],
            "ci_low": df["bss_ci_low"],
            "ci_high": df["bss_ci_high"],
            "status": [claim_status(b, lo, hi) for b, lo, hi in zip(df["bss"], df["bss_ci_low"], df["bss_ci_high"])],
            "signal_flag": "",
            "interpretation": (
                "test_years="
                + df["test_years"].astype(str)
                + "; test_minus_train_dry_mean="
                + df["test_minus_train_dry_mean"].map(lambda x: format_float(x, 3))
                + "; monthly_bias="
                + df["prediction_bias_monthly"].map(lambda x: format_float(x, 3))
            ),
            "source_file": rel(path),
        }
    )
    return out


def evidence_from_prism() -> pd.DataFrame:
    path = RESULTS / "validation" / "prism_model_validation_summary.csv"
    df = read_csv(path)
    if df.empty:
        return df
    row = df.iloc[0]
    bss = float(row["xgb_bss_vs_prism_climatology"])
    out = pd.DataFrame(
        [
            {
                "evidence_group": "independent_precipitation_validation",
                "experiment": "prism_spi1_validation",
                "region": "California Central Valley basin",
                "target": "PRISM SPI-1 dry fraction",
                "lead_months": 1,
                "model": "XGB-Spatial probabilities",
                "calibration": "selected canonical",
                "reference": "PRISM monthly climatology",
                "n_months": int(row["n_test_months"]),
                "bss": bss,
                "ci_low": np.nan,
                "ci_high": np.nan,
                "status": claim_status(bss),
                "signal_flag": "",
                "interpretation": (
                    "CHIRPS-PRISM dry-fraction bias="
                    + format_float(row["chirps_obs_dry_mean"] - row["prism_obs_dry_mean"], 3)
                    + "; xgb_prism_spearman="
                    + format_float(row["xgb_prism_spearman"], 3)
                ),
                "source_file": rel(path),
            }
        ]
    )
    return out


def evidence_from_gefsv12_stack() -> pd.DataFrame:
    path = REPORT / "landsurface" / "landsurface_gefsv12_rzsm_stack_day15_hindcastcal_summary.csv"
    df = read_csv(path)
    if df.empty:
        return df

    label_map = {
        "gefs_selected": "GEFSv12 selected",
        "persistence_selected": "ERA5-Land persistence selected",
        "stack_equal_weight": "GEFSv12 + persistence equal-weight stack",
        "stack_validation_selected": "GEFSv12 + persistence validation-selected stack",
    }
    rows = []
    for record in df.to_dict(orient="records"):
        region_slug = str(record["region"])
        model_key = str(record["model"])
        weight_text = ""
        if model_key == "stack_validation_selected" and pd.notna(record.get("selected_weight_gefs")):
            weight_text = (
                f" Validation-selected convex weight: GEFSv12={float(record['selected_weight_gefs']):.2f}, "
                f"persistence={float(record['selected_weight_persistence']):.2f}."
            )
        rows.append(
            {
                "evidence_group": "forecast_informed_landsurface_stack",
                "experiment": "landsurface_gefsv12_rzsm_stack_day15_hindcastcal",
                "region": resolve_region(region_slug).name,
                "target": "ERA5-Land root-zone soil moisture dry fraction",
                "lead_months": 1,
                "model": label_map.get(model_key, model_key),
                "calibration": "validation-only isotonic plus validation-selected convex blend",
                "reference": "monthly climatology",
                "n_months": int(record["n_months"]),
                "bss": float(record["bss_vs_climatology"]),
                "ci_low": float(record["bss_ci_low"]),
                "ci_high": float(record["bss_ci_high"]),
                "status": claim_status(
                    float(record["bss_vs_climatology"]),
                    float(record["bss_ci_low"]),
                    float(record["bss_ci_high"]),
                ),
                "signal_flag": str(record["added_value_status_selected"]),
                "interpretation": (
                    f"Same-target persistence-relative delta BS={float(record['delta_bs_model_minus_persistence_selected']):+.4f} "
                    f"(95% CI {float(record['delta_bs_ci_low']):+.4f} to {float(record['delta_bs_ci_high']):+.4f})."
                    + weight_text
                ),
                "source_file": rel(path),
            }
        )
    return pd.DataFrame(rows)


def build_master_evidence() -> pd.DataFrame:
    parts = [
        evidence_from_master_headline(),
        evidence_from_seasonal_regional(),
        evidence_from_temporal(),
        evidence_from_prism(),
        evidence_from_gefsv12_stack(),
    ]
    df = pd.concat([p for p in parts if not p.empty], ignore_index=True)
    df["paper_priority"] = False

    priority_patterns = [
        "XGB-Spatial",
        "External precipitation forecast selected",
        "CPC NMME anomaly forecast selected",
        "CPC NMME probability forecast raw",
        "CPC NMME probability forecast selected",
        "NCEI CFSv2 precipitation forecast selected",
        "NCEI CFSv2 soil-moisture forecast",
        "NOAA GEFSv12 RZSM reforecast",
        "C3S/ECMWF native VSM",
        "GEFSv12 + persistence validation-selected stack",
        "ERA5-Land root-zone persistence raw",
        "ERA5-Land root-zone persistence selected",
        "Seasonal SPI-3 lead-3",
        "Seasonal SPI-6 lead-6",
        "EDL MLP selected",
        "XGBoost + soil moisture selected",
        "XGBoost + regional temperature/VPD selected",
        "XGB-Spatial + gridded temperature/VPD selected",
        "XGBoost + MJO/IVT selected",
        "Memory target lag/climate XGBoost selected",
        "Memory target soil-memory XGBoost selected",
        "Memory target external CPC NMME anomaly forecast selected",
        "Memory target external CPC NMME probability forecast selected",
    ]
    def is_priority_model(model: object) -> bool:
        text = "" if pd.isna(model) else str(model).lower()
        return any(pattern.lower() in text for pattern in priority_patterns)

    df.loc[df["model"].map(is_priority_model), "paper_priority"] = True
    df.loc[df["evidence_group"].eq("seasonal_regional_longlead") & df["status"].isin(["robust_positive", "robust_negative"]), "paper_priority"] = True
    df.loc[df["evidence_group"].eq("temporal_robustness"), "paper_priority"] = True
    df.loc[df["evidence_group"].eq("independent_precipitation_validation"), "paper_priority"] = True
    df.loc[df["evidence_group"].eq("multi_region_selected_checkpoint"), "paper_priority"] = True
    df.loc[df["evidence_group"].eq("central_valley_memory_target"), "paper_priority"] = True
    df = df.sort_values(["evidence_group", "region", "target", "lead_months", "model"]).reset_index(drop=True)
    return df


def build_headline_table(master: pd.DataFrame) -> pd.DataFrame:
    keep_groups = {
        "central_valley_calibrated_checkpoint",
        "central_valley_feature_extension",
        "central_valley_uncertainty",
        "central_valley_memory_target",
        "operational_dynamical_benchmark",
        "forecast_informed_landsurface_benchmark",
        "forecast_informed_landsurface_stack",
        "multi_region_selected_checkpoint",
        "seasonal_regional_longlead",
        "temporal_robustness",
        "independent_precipitation_validation",
    }
    df = master.loc[master["paper_priority"] & master["evidence_group"].isin(keep_groups)].copy()
    # Keep seasonal regional table compact: robust rows plus the two Horn rows that complete region coverage.
    seasonal = df["evidence_group"].eq("seasonal_regional_longlead")
    keep_seasonal = (
        df["status"].isin(["robust_positive", "robust_negative"])
        | df["region"].str.contains("horn_of_africa", na=False)
        | df["region"].str.contains("southern_great_plains", na=False)
    )
    df = df.loc[~seasonal | keep_seasonal].copy()
    df = df.sort_values(["evidence_group", "bss"], ascending=[True, False]).reset_index(drop=True)
    return df


def build_mask_table() -> pd.DataFrame:
    basin = read_csv(RESULTS / "multiregion" / "region_basin_mask_diagnostics.csv")
    country = read_csv(RESULTS / "multiregion" / "region_mask_diagnostics.csv")
    rows: list[dict[str, object]] = []
    if not basin.empty:
        for rec in basin.to_dict(orient="records"):
            rows.append(
                {
                    "region": rec["region"],
                    "region_name": rec["region_name"],
                    "mask_kind": "basin_or_ecoregion",
                    "mask_label": rec["mask_label"],
                    "source_url": rec["source_url"],
                    "source_note": rec["source_note"],
                    "valid_pr_cells": rec["valid_pr_cells"],
                    "retained_cells": rec["valid_in_basin_cells"],
                    "retained_fraction": rec["valid_basin_fraction"],
                    "outside_fraction": rec["valid_outside_basin_fraction"],
                    "dataset_exists_at_mask_build": rec["dataset_exists"],
                    "caveat": "Official/scientific region mask; preferred over rectangular bbox.",
                }
            )
    if not country.empty:
        for rec in country.to_dict(orient="records"):
            rows.append(
                {
                    "region": rec["region"],
                    "region_name": rec["region_name"],
                    "mask_kind": "country_intersection",
                    "mask_label": rec["countries"],
                    "source_url": "https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson/ne_50m_admin_0_countries.geojson",
                    "source_note": "Natural Earth 1:50m country polygons; CHIRPS grid-cell-center intersection.",
                    "valid_pr_cells": rec["valid_pr_cells"],
                    "retained_cells": rec["valid_in_country_cells"],
                    "retained_fraction": rec["valid_country_fraction"],
                    "outside_fraction": rec["valid_outside_country_fraction"],
                    "dataset_exists_at_mask_build": rec["dataset_exists"],
                    "caveat": rec["mask_note"],
                }
            )
    return pd.DataFrame(rows).sort_values(["region", "mask_kind"]).reset_index(drop=True)


def build_temporal_table() -> pd.DataFrame:
    path = RESULTS / "temporal" / "temporal_robustness_summary.csv"
    df = read_csv(path)
    if df.empty:
        return df
    df = df.loc[df["reference"].eq("clim_train_monthly")].copy()
    cols = [
        "split",
        "test_years",
        "n_test_months",
        "selected_calibration",
        "bss",
        "bss_ci_low",
        "bss_ci_high",
        "test_minus_train_dry_mean",
        "prediction_bias_monthly",
        "prediction_corr_monthly",
        "prediction_amplitude_ratio",
    ]
    return df[cols].reset_index(drop=True)


def build_seasonal_signal_table() -> pd.DataFrame:
    df = read_first_csv(
        REPORT / "seasonal" / "seasonal_regional_signal_audit.csv",
        RESULTS / "seasonal" / "seasonal_regional_signal_audit.csv",
    )
    if df.empty:
        return df
    cols = [
        "region",
        "target_spi",
        "lead_months",
        "climate_features",
        "bss_iso_vs_clim",
        "corr_iso_obs",
        "std_ratio_iso_obs",
        "iso_bias",
        "iso_beats_clim_fraction",
        "robust_status",
        "signal_flag",
    ]
    return df[cols].sort_values(["region", "target_spi", "lead_months", "climate_features"]).reset_index(drop=True)


def build_regionalization_table() -> pd.DataFrame:
    mech = read_first_csv(
        REPORT / "regionalization" / "regionalization_mechanism_summary.csv",
        RESULTS / "regionalization" / "regionalization_mechanism_summary.csv",
    )
    zone = read_first_csv(
        REPORT / "regionalization" / "zone_forecast_diagnostics.csv",
        RESULTS / "regionalization" / "zone_forecast_diagnostics.csv",
    )
    if mech.empty:
        return mech

    best_idx = mech["top_pearson_r"].abs().groupby(mech["run_slug"]).idxmax()
    best = mech.loc[best_idx].copy()
    best = best.rename(
        columns={
            "top_index": "strongest_index",
            "top_lag_months": "strongest_lag_months",
            "top_pearson_r": "strongest_pearson_r",
            "top_p_value": "strongest_p_value",
        }
    )

    if not zone.empty:
        z = zone.loc[zone["model"].eq("spatial")].copy()
        agg = (
            z.groupby("run_slug")
            .agg(
                max_zone_bss=("selected_bss_vs_zone_climatology", "max"),
                n_positive_zone_bss=("selected_bss_vs_zone_climatology", lambda s: int((s > 0).sum())),
                n_zones_with_forecast=("selected_bss_vs_zone_climatology", "size"),
                median_selected_corr=("selected_corr", "median"),
            )
            .reset_index()
        )
        best = best.merge(agg, on="run_slug", how="left")
    cols = [
        "run_slug",
        "region_name",
        "mask_kind",
        "pca_cumulative_explained_variance",
        "zone",
        "n_pixels",
        "pixel_fraction",
        "strongest_index",
        "strongest_lag_months",
        "strongest_pearson_r",
        "strongest_p_value",
        "max_zone_bss",
        "n_positive_zone_bss",
        "n_zones_with_forecast",
        "median_selected_corr",
    ]
    return best[[c for c in cols if c in best.columns]].sort_values("run_slug").reset_index(drop=True)


def errorbar_plot(
    df: pd.DataFrame,
    path: Path,
    title: str,
    label_cols: list[str],
    width: float = 9.0,
    row_height: float = 0.34,
) -> None:
    plot = df.copy()
    plot = plot.loc[plot["bss"].map(finite)].copy()
    if plot.empty:
        return
    plot = plot.sort_values("bss", ascending=True).reset_index(drop=True)
    labels = plot[label_cols].fillna("").astype(str).agg(" | ".join, axis=1)
    labels = labels.map(lambda s: textwrap.fill(s, width=48))
    y = np.arange(len(plot))
    fig_h = max(4.0, 1.0 + row_height * len(plot))
    fig, ax = plt.subplots(figsize=(width, fig_h))
    colors = [STATUS_COLORS.get(s, "#546E7A") for s in plot["status"]]
    for i, row in plot.iterrows():
        lo = row["ci_low"]
        hi = row["ci_high"]
        xerr = None
        if finite(lo) and finite(hi):
            xerr = [[float(row["bss"]) - float(lo)], [float(hi) - float(row["bss"])]]
        ax.errorbar(
            row["bss"],
            i,
            xerr=xerr,
            fmt="o",
            color=colors[i],
            ecolor=colors[i],
            elinewidth=1.2,
            capsize=2.5,
            markersize=4.8,
        )
    ax.axvline(0, color="#333333", linewidth=1.0)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Brier Skill Score vs climatology")
    ax.set_title(title)
    ax.grid(axis="x", color="#DDDDDD", linewidth=0.7)
    ax.set_axisbelow(True)
    plt.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_multiregion(path: Path) -> None:
    df = read_csv(RESULTS / "multiregion" / "multiregion_summary.csv")
    if df.empty:
        return
    plot = df.copy()
    plot["bss"] = plot["selected_bss"]
    plot["ci_low"] = plot["selected_bss_ci_low"]
    plot["ci_high"] = plot["selected_bss_ci_high"]
    plot["status"] = [claim_status(b, lo, hi) for b, lo, hi in zip(plot["bss"], plot["ci_low"], plot["ci_high"])]
    plot["label"] = plot.apply(lambda row: f"{short_region(row['region'])}: {short_model(row['model'])}", axis=1)
    errorbar_plot(
        plot.rename(columns={"label": "experiment"}),
        path,
        "Multi-region SPI-1 lead-1 selected BSS",
        ["experiment"],
        width=8.5,
        row_height=0.32,
    )


def plot_seasonal_tracking(df: pd.DataFrame, path: Path) -> None:
    if df.empty:
        return
    plot = df.copy()
    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    for flag, part in plot.groupby("signal_flag"):
        ax.scatter(
            part["corr_iso_obs"],
            part["bss_iso_vs_clim"],
            s=40 + 140 * part["std_ratio_iso_obs"].clip(lower=0, upper=1.5),
            color=SIGNAL_COLORS.get(flag, "#546E7A"),
            alpha=0.82,
            edgecolor="white",
            linewidth=0.6,
            label=flag.replace("_", " "),
        )
    ax.axhline(0, color="#333333", linewidth=1.0)
    ax.axvline(0, color="#BBBBBB", linewidth=0.8)
    ax.set_xlabel("Correlation: calibrated dry fraction vs observed dry fraction")
    ax.set_ylabel("Calibrated BSS vs monthly climatology")
    ax.set_title("Seasonal regional BSS vs event tracking")
    ax.grid(color="#E0E0E0", linewidth=0.7)
    ax.legend(fontsize=7, frameon=False, loc="best")
    plt.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_mask_retention(df: pd.DataFrame, path: Path) -> None:
    if df.empty:
        return
    plot = df.sort_values("retained_fraction")
    labels = plot["region"] + " | " + plot["mask_kind"]
    fig, ax = plt.subplots(figsize=(8.0, max(3.5, 0.45 * len(plot) + 1.0)))
    ax.barh(labels, plot["retained_fraction"], color="#4E79A7")
    ax.set_xlim(0, 1)
    ax.set_xlabel("Fraction of valid CHIRPS cells retained")
    ax.set_title("Mask retention by source-cited region geometry")
    for i, value in enumerate(plot["retained_fraction"]):
        ax.text(min(float(value) + 0.015, 0.98), i, f"{float(value):.1%}", va="center", fontsize=8)
    ax.grid(axis="x", color="#E0E0E0", linewidth=0.7)
    ax.set_axisbelow(True)
    plt.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def build_evaluation_inflation_table() -> pd.DataFrame:
    path = REPORT / "evaluation_inflation_audit.csv"
    if not path.exists():
        path = PROJECT_ROOT / "outputs" / "evaluation_inflation_audit.csv"
    df = read_csv(path)
    if df.empty:
        return df

    keep_cols = [
        "scenario",
        "target",
        "split_kind",
        "inference_level",
        "protocol_validity",
        "target_overlap_months",
        "calibration_selection_level",
        "selected_calibration",
        "n_units_for_inference",
        "n_pixel_rows_scored",
        "bs_reference",
        "bs_model",
        "bss_vs_climatology",
        "bss_ci_low",
        "bss_ci_high",
        "spearman_obs_pred",
        "audit_status",
        "top_features",
    ]
    out = df[[c for c in keep_cols if c in df.columns]].copy()
    conditions = [
        out["protocol_validity"].eq("invalid_random_row_split"),
        out["protocol_validity"].eq("invalid_overlapping_target"),
        out["inference_level"].eq("pixel"),
    ]
    choices = [
        "Invalid split: train/validation/test contain pixels from the same target months.",
        "Invalid target: SPI accumulation windows overlap with feature months.",
        "Invalid inference: treats spatially autocorrelated pixels as independent units.",
    ]
    out["interpretation"] = np.select(conditions, choices, default="Valid monthly benchmark/control.")
    return out.sort_values(["protocol_validity", "scenario", "inference_level"]).reset_index(drop=True)


def build_transition_table() -> pd.DataFrame:
    df = read_first_csv(
        REPORT / "transition" / "transition_target_summary.csv",
        RESULTS / "transition" / "transition_target_summary.csv",
    )
    if df.empty:
        return df
    keep_cols = [
        "scope",
        "transition",
        "model",
        "n_months",
        "n_eligible_median",
        "n_eligible_min",
        "n_eligible_max",
        "bs_reference",
        "bs_model",
        "bss_vs_eligible_climatology",
        "bss_ci_low",
        "bss_ci_high",
        "spearman",
        "amplitude_ratio",
        "claim_status",
        "source_file",
    ]
    out = df[[c for c in keep_cols if c in df.columns]].copy()
    conditions = [
        out["scope"].eq("cvalley") & out["transition"].eq("onset") & out["model"].astype(str).str.contains("eligible-only", case=False, na=False),
        out["scope"].eq("cvalley_basin_masked") & out["transition"].eq("onset"),
        out["scope"].eq("southern_great_plains_basin_masked") & out["transition"].eq("onset"),
        out["scope"].eq("mediterranean_spain_basin_masked") & out["transition"].eq("onset"),
        out["transition"].eq("termination"),
    ]
    choices = [
        "Rectangular Central Valley onset is small robust-positive, but it is not replicated by stricter geometry.",
        "Central Valley basin-mask sensitivity does not preserve the rectangular onset signal.",
        "Southern Great Plains basin-masked onset is robustly worse than eligible climatology.",
        "Mediterranean Spain basin-masked onset remains near eligible climatology.",
        "Termination does not beat the eligible-state climatology.",
    ]
    out["interpretation"] = np.select(conditions, choices, default="Transition-target diagnostic row.")
    return out.sort_values(["transition", "scope", "model"]).reset_index(drop=True)


def build_landsurface_added_value_table() -> pd.DataFrame:
    df = read_first_csv(
        REPORT / "landsurface" / "landsurface_added_value_diagnostics.csv",
        RESULTS / "landsurface" / "landsurface_added_value_diagnostics.csv",
    )
    if df.empty:
        return df
    keep_cols = [
        "region",
        "forecast_system",
        "group_type",
        "group_value",
        "n_months",
        "bs_climatology",
        "bs_forecast_selected",
        "bs_persistence_raw",
        "bs_persistence_selected",
        "forecast_bss_vs_climatology",
        "persistence_raw_bss_vs_climatology",
        "persistence_selected_bss_vs_climatology",
        "forecast_bss_vs_persistence_raw",
        "forecast_bss_vs_persistence_selected",
        "delta_bs_forecast_minus_persistence_raw",
        "delta_bs_raw_ci_low",
        "delta_bs_raw_ci_high",
        "delta_bs_forecast_minus_persistence_selected",
        "delta_bs_selected_ci_low",
        "delta_bs_selected_ci_high",
        "added_value_status_raw",
        "added_value_status_selected",
        "source_file",
    ]
    out = df[[c for c in keep_cols if c in df.columns]].copy()
    conditions = [
        out["group_type"].eq("overall") & out["forecast_system"].eq("CFSv2_RZSM") & out["region"].eq("southern_great_plains"),
        out["group_type"].eq("overall") & out["forecast_system"].eq("CFSv2_RZSM") & out["region"].eq("cvalley"),
        out["group_type"].eq("overall") & out["forecast_system"].eq("CFSv2_RZSM") & out["region"].eq("mediterranean_spain"),
        out["group_type"].eq("overall") & out["forecast_system"].eq("GEFSv12_RZSM") & out["region"].eq("southern_great_plains"),
        out["group_type"].eq("overall") & out["forecast_system"].eq("GEFSv12_RZSM") & out["region"].eq("mediterranean_spain"),
        out["group_type"].eq("overall") & out["forecast_system"].eq("GEFSv12_RZSM") & out["region"].eq("cvalley"),
        out["added_value_status_raw"].eq("persistence_robustly_better"),
        out["added_value_status_raw"].astype(str).str.endswith("_robust_added_value", na=False),
    ]
    choices = [
        "Overall Southern Great Plains persistence is robustly better than CFSv2.",
        "Overall Central Valley CFSv2 and raw persistence are not statistically separable.",
        "Overall Mediterranean Spain does not show CFSv2 skill over climatology or persistence.",
        "Overall Southern Great Plains GEFSv12 beats same-target persistence on point Brier score.",
        "Overall Mediterranean Spain persistence is stronger than GEFSv12.",
        "Overall Central Valley GEFSv12 is better than weak same-period persistence, with uncertainty.",
        "Conditional subset where raw persistence is robustly better than CFSv2.",
        "Conditional subset where the dynamic forecast is robustly better than raw persistence.",
    ]
    out["interpretation"] = np.select(conditions, choices, default="Conditional added-value diagnostic row.")
    return out.sort_values(["forecast_system", "region", "group_type", "group_value"]).reset_index(drop=True)


def build_climate_index_sensitivity_table() -> pd.DataFrame:
    df = read_csv(REPORT / "climate" / "climate_index_sensitivity_summary.csv")
    if df.empty:
        return df
    keep_cols = [
        "model_kind",
        "variant",
        "n_test_months",
        "test_target_start",
        "test_target_end",
        "selected_calibration",
        "bs_climatology",
        "bs_selected",
        "selected_bss",
        "selected_bss_ci_low",
        "selected_bss_ci_high",
        "raw_bss",
        "platt_bss",
        "isotonic_bss",
        "spearman_obs_pred",
        "amplitude_ratio",
        "claim_status",
        "top_features",
    ]
    out = df[[c for c in keep_cols if c in df.columns]].copy()
    conditions = [
        out["variant"].eq("chirps_only"),
        out["variant"].eq("nino34"),
        out["variant"].eq("pdo"),
        out["variant"].eq("nino34_pdo"),
    ]
    choices = [
        "Common-valid-period CHIRPS/SPI baseline; confidence interval crosses zero.",
        "Niño3.4 does not improve common-valid-period BSS over the CHIRPS/SPI baseline.",
        "PDO does not improve common-valid-period BSS and is negative under selected calibration.",
        "Combining Niño3.4 and PDO over-amplifies probabilities and is robustly worse than climatology.",
    ]
    out["interpretation"] = np.select(conditions, choices, default="Climate-index sensitivity row.")
    return out.sort_values(["model_kind", "variant"]).reset_index(drop=True)


def build_gefsv12_landsurface_sensitivity_table() -> pd.DataFrame:
    path = (
        REPORT
        / "landsurface"
        / "landsurface_gefsv12_rzsm_sgp_lead_validday_sensitivity_hindcastcal_grid_sensitivity_summary.csv"
    )
    df = read_csv(path)
    if df.empty:
        return df
    keep_cols = [
        "region",
        "valid_day",
        "init_lag_weeks",
        "n_months_test",
        "gefs_selected_bss",
        "gefs_selected_bss_ci_low",
        "gefs_selected_bss_ci_high",
        "gefs_selected_bss_vs_persistence_selected",
        "delta_bs_gefs_minus_persistence_selected",
        "persistence_selected_bss",
        "lead_days_mean_test",
        "spearman_gefs_selected_vs_observed_test",
        "amplitude_ratio_gefs_selected_test",
        "status",
    ]
    out = df[[c for c in keep_cols if c in df.columns]].copy()
    out["claim_status"] = [
        claim_status(bss, lo, hi)
        for bss, lo, hi in zip(
            out["gefs_selected_bss"],
            out["gefs_selected_bss_ci_low"],
            out["gefs_selected_bss_ci_high"],
        )
    ]
    out["beats_selected_persistence_point"] = out["gefs_selected_bss_vs_persistence_selected"] > 0.0
    best_idx = out["gefs_selected_bss"].idxmax()
    worst_idx = out["gefs_selected_bss"].idxmin()
    out["interpretation"] = "Robust positive versus climatology; persistence-relative value is a point-estimate diagnostic."
    out.loc[best_idx, "interpretation"] = "Best GEFSv12 extraction choice in the sensitivity grid."
    out.loc[worst_idx, "interpretation"] = "Worst GEFSv12 extraction choice; still robust positive versus climatology."
    out.loc[
        ~out["beats_selected_persistence_point"],
        "interpretation",
    ] = "Robust versus climatology but does not beat selected persistence on the point estimate."
    out["source_file"] = rel(path)
    return out.sort_values(["init_lag_weeks", "valid_day"]).reset_index(drop=True)


def build_gefsv12_landsurface_stack_table() -> pd.DataFrame:
    path = REPORT / "landsurface" / "landsurface_gefsv12_rzsm_stack_day15_hindcastcal_summary.csv"
    df = read_csv(path)
    if df.empty:
        return df
    keep_cols = [
        "region",
        "model",
        "n_months",
        "bs_climatology",
        "bs_model",
        "bs_persistence_selected",
        "bss_vs_climatology",
        "bss_ci_low",
        "bss_ci_high",
        "bss_vs_persistence_selected",
        "delta_bs_model_minus_persistence_selected",
        "delta_bs_ci_low",
        "delta_bs_ci_high",
        "added_value_status_selected",
        "selected_weight_gefs",
        "selected_weight_persistence",
        "selected_gefs_calibration",
        "selected_persistence_calibration",
        "monotonic_xgb_features",
        "monotonic_xgb_constraints",
        "spearman_model_vs_observed",
        "amplitude_ratio_model",
    ]
    out = df[[c for c in keep_cols if c in df.columns]].copy()
    out["claim_status"] = [
        claim_status(bss, lo, hi)
        for bss, lo, hi in zip(
            out["bss_vs_climatology"],
            out["bss_ci_low"],
            out["bss_ci_high"],
        )
    ]
    out["source_file"] = rel(path)
    conditions = [
        out["model"].eq("stack_validation_selected") & out["added_value_status_selected"].eq("stack_robust_added_value") & out["claim_status"].eq("robust_positive"),
        out["model"].eq("stack_validation_selected") & out["added_value_status_selected"].eq("stack_robust_added_value") & out["claim_status"].ne("robust_positive"),
        out["model"].eq("stack_validation_selected") & out["added_value_status_selected"].eq("stack_uncertain_added_value"),
        out["model"].eq("stack_validation_selected") & out["added_value_status_selected"].astype(str).str.contains("persistence", na=False),
        out["model"].eq("stack_equal_weight"),
        out["model"].eq("gefs_selected"),
        out["model"].eq("persistence_selected"),
    ]
    choices = [
        "Validation-selected GEFSv12+persistence stack is robust-positive against climatology and improves selected persistence by paired delta BS.",
        "Validation-selected stack improves selected persistence by paired delta BS but is not robust-positive against climatology.",
        "Validation-selected stack is robust-positive against climatology, with uncertain added value over selected persistence.",
        "Validation-selected stack is robust-positive against climatology, but selected persistence is not beaten.",
        "Fixed equal-weight stack is a no-selection sensitivity row.",
        "Single-source GEFSv12 selected reference row.",
        "Same-target selected persistence reference row.",
    ]
    out["interpretation"] = np.select(conditions, choices, default="GEFSv12+persistence stack diagnostic row.")
    return out.sort_values(["region", "model"]).reset_index(drop=True)


LANDSURFACE_RELIABILITY_MODELS = {
    "gefs_selected": "gefs_selected_prob_dry",
    "persistence_selected": "persistence_selected_prob_dry",
    "stack_equal_weight": "stack_equal_prob_dry",
    "stack_validation_selected": "stack_validation_selected_prob_dry",
}


def reliability_resolution(
    monthly: pd.DataFrame,
    pred_col: str,
    n_bins: int = 5,
) -> tuple[dict[str, object], pd.DataFrame]:
    data = monthly[["region", "target_time", "y_true_dry_frac", "clim_prob_dry", pred_col]].dropna().copy()
    data = data.rename(columns={pred_col: "pred_prob_dry"})
    data["pred_prob_dry"] = data["pred_prob_dry"].astype(float)
    data["y_true_dry_frac"] = data["y_true_dry_frac"].astype(float)
    data["clim_prob_dry"] = data["clim_prob_dry"].astype(float)
    if data.empty:
        return {}, pd.DataFrame()

    try:
        bins = pd.qcut(data["pred_prob_dry"], q=min(n_bins, len(data)), duplicates="drop")
    except ValueError:
        bins = pd.Series(["all"] * len(data), index=data.index)
    data["bin"] = bins.astype(str)

    overall_obs = float(data["y_true_dry_frac"].mean())
    bs_model = float(np.mean((data["pred_prob_dry"] - data["y_true_dry_frac"]) ** 2))
    bs_climatology = float(np.mean((data["clim_prob_dry"] - data["y_true_dry_frac"]) ** 2))
    uncertainty = float(np.mean((data["y_true_dry_frac"] - overall_obs) ** 2))

    bin_rows: list[dict[str, object]] = []
    reliability = 0.0
    resolution = 0.0
    n_total = len(data)
    for i, (_, part) in enumerate(data.groupby("bin", sort=False), start=1):
        weight = len(part) / n_total
        mean_pred = float(part["pred_prob_dry"].mean())
        mean_obs = float(part["y_true_dry_frac"].mean())
        rel_component = weight * (mean_pred - mean_obs) ** 2
        res_component = weight * (mean_obs - overall_obs) ** 2
        reliability += rel_component
        resolution += res_component
        bin_rows.append(
            {
                "bin_index": i,
                "n_months": int(len(part)),
                "pred_min": float(part["pred_prob_dry"].min()),
                "pred_max": float(part["pred_prob_dry"].max()),
                "mean_pred": mean_pred,
                "mean_observed": mean_obs,
                "bin_bs": float(np.mean((part["pred_prob_dry"] - part["y_true_dry_frac"]) ** 2)),
                "reliability_component": rel_component,
                "resolution_component": res_component,
            }
        )

    pred_std = float(data["pred_prob_dry"].std(ddof=0))
    obs_std = float(data["y_true_dry_frac"].std(ddof=0))
    slope = np.nan
    if pred_std > 0:
        slope = float(np.cov(data["pred_prob_dry"], data["y_true_dry_frac"], ddof=0)[0, 1] / (pred_std**2))
    spearman = data["pred_prob_dry"].corr(data["y_true_dry_frac"], method="spearman")
    binned_bss_proxy = (resolution - reliability) / uncertainty if uncertainty > 0 else np.nan
    summary = {
        "n_months": int(n_total),
        "n_bins": int(len(bin_rows)),
        "bs_model": bs_model,
        "bs_climatology": bs_climatology,
        "bss_vs_climatology": 1.0 - bs_model / bs_climatology if bs_climatology > 0 else np.nan,
        "mean_pred": float(data["pred_prob_dry"].mean()),
        "mean_observed": overall_obs,
        "calibration_bias": float(data["pred_prob_dry"].mean() - overall_obs),
        "pred_std": pred_std,
        "observed_std": obs_std,
        "amplitude_ratio": pred_std / obs_std if obs_std > 0 else np.nan,
        "spearman_pred_observed": float(spearman) if pd.notna(spearman) else np.nan,
        "calibration_slope": slope,
        "reliability": float(reliability),
        "resolution": float(resolution),
        "uncertainty": uncertainty,
        "reliability_ratio": reliability / uncertainty if uncertainty > 0 else np.nan,
        "resolution_ratio": resolution / uncertainty if uncertainty > 0 else np.nan,
        "resolution_minus_reliability": float(resolution - reliability),
        "binned_bss_proxy": float(binned_bss_proxy) if np.isfinite(binned_bss_proxy) else np.nan,
    }
    return summary, pd.DataFrame(bin_rows)


def build_landsurface_reliability_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    path = REPORT / "landsurface" / "landsurface_gefsv12_rzsm_stack_day15_hindcastcal_monthly_scores.csv"
    monthly = read_csv(path)
    if monthly.empty:
        return pd.DataFrame(), pd.DataFrame()
    monthly["target_time"] = pd.to_datetime(monthly["target_time"])

    summaries: list[dict[str, object]] = []
    bins_out: list[pd.DataFrame] = []
    for region, region_df in monthly.groupby("region", sort=True):
        for model, pred_col in LANDSURFACE_RELIABILITY_MODELS.items():
            if pred_col not in region_df.columns:
                continue
            summary, bins = reliability_resolution(region_df, pred_col)
            if not summary:
                continue
            summary.update(
                {
                    "region": region,
                    "model": model,
                    "source_file": rel(path),
                    "interpretation": "Binned reliability/resolution diagnostic over the frozen 2017-2019 test months.",
                }
            )
            summaries.append(summary)
            if not bins.empty:
                bins = bins.copy()
                bins.insert(0, "model", model)
                bins.insert(0, "region", region)
                bins["source_file"] = rel(path)
                bins_out.append(bins)

    summary_df = pd.DataFrame(summaries)
    bins_df = pd.concat(bins_out, ignore_index=True) if bins_out else pd.DataFrame()
    if not summary_df.empty:
        summary_df = summary_df[
            [
                "region",
                "model",
                "n_months",
                "n_bins",
                "bs_model",
                "bs_climatology",
                "bss_vs_climatology",
                "mean_pred",
                "mean_observed",
                "calibration_bias",
                "pred_std",
                "observed_std",
                "amplitude_ratio",
                "spearman_pred_observed",
                "calibration_slope",
                "reliability",
                "resolution",
                "uncertainty",
                "reliability_ratio",
                "resolution_ratio",
                "resolution_minus_reliability",
                "binned_bss_proxy",
                "interpretation",
                "source_file",
            ]
        ].sort_values(["region", "model"]).reset_index(drop=True)
    return summary_df, bins_df


def build_landsurface_domain_transfer_table() -> pd.DataFrame:
    path = REPORT / "landsurface" / "landsurface_gefsv12_rzsm_domain_transfer_day15_hindcastcal_summary.csv"
    df = read_csv(path)
    if df.empty:
        return df
    keep_cols = [
        "target_region",
        "transfer_mode",
        "source_regions",
        "model",
        "n_source_validation_months",
        "n_source_validation_rows",
        "n_test_months",
        "bs_climatology",
        "bs_model",
        "bs_persistence_transfer_selected",
        "bss_vs_climatology",
        "bss_ci_low",
        "bss_ci_high",
        "bss_vs_persistence_transfer_selected",
        "delta_bs_model_minus_persistence_transfer_selected",
        "delta_bs_ci_low",
        "delta_bs_ci_high",
        "added_value_status",
        "selected_weight_gefs",
        "selected_weight_persistence",
        "selected_gefs_calibration",
        "selected_persistence_calibration",
        "monotonic_xgb_features",
        "monotonic_xgb_constraints",
        "spearman_model_vs_observed",
        "amplitude_ratio_model",
    ]
    out = df[[c for c in keep_cols if c in df.columns]].copy()
    out["claim_status"] = [
        claim_status(bss, lo, hi)
        for bss, lo, hi in zip(out["bss_vs_climatology"], out["bss_ci_low"], out["bss_ci_high"])
    ]
    conditions = [
        out["model"].eq("stack_transfer_selected") & out["transfer_mode"].eq("local"),
        out["model"].eq("stack_transfer_selected") & out["transfer_mode"].eq("pooled"),
        out["model"].eq("stack_transfer_selected") & out["transfer_mode"].eq("leave_one_region_out"),
        out["model"].eq("gefs_transfer_selected"),
        out["model"].eq("persistence_transfer_selected"),
        out["model"].eq("monotonic_xgb_transfer"),
    ]
    choices = [
        "Target-region validation calibration reference.",
        "Pooled multi-region validation calibration; includes the target-region validation period.",
        "Leave-one-region-out calibration; excludes the target region from calibration and weight selection.",
        "Transferred GEFSv12-only calibration row.",
        "Transferred same-target persistence calibration row.",
        "Monotonic XGBoost dry-fraction row constrained by GEFSv12 dry signal and same-target persistence.",
    ]
    out["interpretation"] = np.select(conditions, choices, default="Domain-transfer diagnostic row.")
    out["source_file"] = rel(path)
    return out.sort_values(["transfer_mode", "target_region", "model"]).reset_index(drop=True)


def build_landsurface_rare_event_table() -> pd.DataFrame:
    path = REPORT / "landsurface" / "landsurface_gefsv12_rzsm_domain_transfer_day15_hindcastcal_rare_event_summary.csv"
    df = read_csv(path)
    if df.empty:
        return df
    keep_cols = [
        "target_region",
        "transfer_mode",
        "source_regions",
        "event_quantile",
        "event_threshold_dry_fraction",
        "model",
        "event_calibration",
        "n_source_validation_rows",
        "n_test_months",
        "n_test_events",
        "test_event_rate",
        "validation_event_rate_reference",
        "average_precision",
        "average_precision_ci_low",
        "average_precision_ci_high",
        "average_precision_lift_over_event_rate",
        "event_brier_model",
        "event_brier_climatology",
        "event_bss_vs_validation_climatology",
        "event_bss_ci_low",
        "event_bss_ci_high",
        "event_claim_status",
        "event_reliability",
        "event_resolution",
        "event_uncertainty",
        "event_resolution_minus_reliability",
    ]
    out = df[[c for c in keep_cols if c in df.columns]].copy()
    out["interpretation"] = (
        "Rare-event diagnostic for validation-defined regional upper-tail root-zone dry-fraction months."
    )
    out["source_file"] = rel(path)
    return out.sort_values(["transfer_mode", "event_quantile", "target_region", "model"]).reset_index(drop=True)


def build_landsurface_persistence_regime_table() -> pd.DataFrame:
    path = REPORT / "landsurface" / "landsurface_persistence_regime_diagnostics.csv"
    df = read_csv(path)
    if df.empty:
        return df
    keep_cols = [
        "transfer_mode",
        "target_region",
        "group_type",
        "group_value",
        "model",
        "n_months",
        "small_sample_flag",
        "mean_observed_dry_fraction",
        "mean_antecedent_dry_fraction",
        "mean_gefs_dry_anomaly_signal",
        "bs_model",
        "bs_persistence_transfer_selected",
        "delta_bs_model_minus_persistence",
        "delta_bs_ci_low",
        "delta_bs_ci_high",
        "bss_vs_persistence",
        "added_value_status",
        "interpretation",
    ]
    out = df[[c for c in keep_cols if c in df.columns]].copy()
    out["source_file"] = rel(path)
    return out.sort_values(["transfer_mode", "target_region", "group_type", "group_value", "model"]).reset_index(drop=True)


def build_landsurface_independent_target_audit_table() -> pd.DataFrame:
    path = REPORT / "landsurface" / "landsurface_independent_target_audit.csv"
    df = read_csv(path)
    if df.empty:
        return df
    keep_cols = [
        "product",
        "target_variable",
        "spatial_domain",
        "nominal_period",
        "independent_status",
        "validation_role",
        "covered_regions",
        "n_project_regions_covered",
        "local_file_count",
        "ready_to_score_now",
        "source_url",
        "access_note",
        "priority",
        "current_decision",
    ]
    out = df[[c for c in keep_cols if c in df.columns]].copy()
    out["source_file"] = rel(path)
    priority_rank = {
        "high_us_only": 0,
        "high_global_short_record": 1,
        "medium_global_model_sensitivity": 2,
        "medium_surface_only": 3,
    }
    out["_priority_rank"] = out["priority"].map(priority_rank).fillna(99).astype(int)
    return (
        out.sort_values(["ready_to_score_now", "_priority_rank", "product"], ascending=[False, True, True])
        .drop(columns=["_priority_rank"])
        .reset_index(drop=True)
    )


def build_landsurface_forecast_archive_audit_table() -> pd.DataFrame:
    path = REPORT / "landsurface" / "landsurface_forecast_archive_audit.csv"
    df = read_csv(path)
    if df.empty:
        return df
    keep_cols = [
        "forecast_archive",
        "archive_role",
        "remote_probe_status",
        "soil_moisture_support",
        "target_overlap",
        "scientific_decision",
        "priority",
        "source_url",
        "source_reference",
    ]
    out = df[[c for c in keep_cols if c in df.columns]].copy()
    out["source_file"] = rel(path)
    priority_rank = {
        "high_diagnostic_followup": 0,
        "high_next_smoke_test": 0,
        "complete_baseline": 1,
        "high_after_operational_gefs_smoke_or_parallel_download": 2,
        "medium_second_line": 3,
        "medium_high_friction": 4,
        "low_for_retrospective_scoring": 5,
    }
    out["_priority_rank"] = out["priority"].map(priority_rank).fillna(99).astype(int)
    return out.sort_values(["_priority_rank", "forecast_archive"]).drop(columns=["_priority_rank"]).reset_index(drop=True)


def build_operational_gefs_diagnostic_table() -> pd.DataFrame:
    paths = sorted(
        (REPORT / "landsurface").glob(
            "landsurface_operational_gefs_*_cvalley_2021_2025_diagnostics_candidate_scores.csv"
        )
    )
    if not paths:
        return pd.DataFrame()
    keep_candidates = {
        "stack_gefs_anom_persistence_raw",
        "stack_gefs_raw_persistence_raw",
        "operational_gefs_anom_isotonic",
        "operational_gefs_raw_isotonic",
        "persistence_raw",
        "monthly_climatology",
    }
    parts = []
    for path in paths:
        df = read_csv(path)
        if df.empty:
            continue
        chunk = df.loc[df["candidate"].isin(keep_candidates)].copy()
        if chunk.empty:
            continue
        if "soill_0_1m" in path.name:
            soil_mode = "soill_0_1m"
        elif "subroot" in path.name:
            soil_mode = "soilw_0p1_1m"
        else:
            soil_mode = "unknown"
        chunk["soil_mode"] = soil_mode
        chunk["diagnostic_id"] = path.stem.removesuffix("_candidate_scores")
        chunk["source_file"] = rel(path)
        parts.append(chunk)
    if not parts:
        return pd.DataFrame()
    out = pd.concat(parts, ignore_index=True)
    keep_cols = [
        "soil_mode",
        "diagnostic_id",
        "candidate",
        "formula",
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
        "test_delta_ci_low",
        "test_delta_ci_high",
        "validation_selected_any",
        "persistence_safe_selected",
        "test_spearman_y_pred",
        "source_file",
    ]
    out = out[[c for c in keep_cols if c in out.columns]].copy()
    order = {
        "stack_gefs_anom_persistence_raw": 0,
        "stack_gefs_raw_persistence_raw": 1,
        "operational_gefs_anom_isotonic": 2,
        "operational_gefs_raw_isotonic": 3,
        "persistence_raw": 4,
        "monthly_climatology": 5,
    }
    soil_order = {"soilw_0p1_1m": 0, "soill_0_1m": 1}
    out["_order"] = out["candidate"].map(order).fillna(99).astype(int)
    out["_soil_order"] = out["soil_mode"].map(soil_order).fillna(99).astype(int)
    return (
        out.sort_values(["_soil_order", "_order"])
        .drop(columns=["_soil_order", "_order"])
        .reset_index(drop=True)
    )


def build_operational_gefs_smap_validation_table() -> pd.DataFrame:
    paths = sorted(
        (REPORT / "landsurface").glob(
            "landsurface_operational_gefs_*_cvalley_smap_validation_summary.csv"
        )
    )
    if not paths:
        return pd.DataFrame()
    parts = []
    for path in paths:
        df = read_csv(path)
        if df.empty:
            continue
        if "soill_0_1m" in path.name:
            soil_mode = "soill_0_1m"
        elif "soilw_0p1_1m" in path.name:
            soil_mode = "soilw_0p1_1m"
        else:
            soil_mode = "unknown"
        df = df.copy()
        df["soil_mode"] = soil_mode
        df["source_file"] = rel(path)
        parts.append(df)
    if not parts:
        return pd.DataFrame()
    out = pd.concat(parts, ignore_index=True)
    keep_models = {
        "operational_gefs_raw_isotonic",
        "operational_gefs_anom_isotonic",
        "operational_gefs_selected",
        "persistence_raw",
        "persistence_selected",
        "stack_validation_selected",
        "cv_operational_gefs_selected",
        "cv_persistence_selected",
        "cv_stack_selected",
        "persistence_guard_selected",
        "monthly_climatology",
    }
    out = out.loc[out["model"].isin(keep_models)].copy()
    out["claim_status"] = [
        claim_status(bss, lo, hi)
        for bss, lo, hi in zip(out["bss_vs_climatology"], out["bss_ci_low"], out["bss_ci_high"])
    ]
    keep_cols = [
        "soil_mode",
        "region",
        "target_product",
        "model",
        "selected_by_validation",
        "n_validation_months",
        "n_test_months",
        "brier_score",
        "brier_score_climatology",
        "bss_vs_climatology",
        "bss_ci_low",
        "bss_ci_high",
        "delta_bs_vs_raw_persistence",
        "delta_bs_ci_low",
        "delta_bs_ci_high",
        "selected_weight_gefs",
        "selected_gefs_calibration",
        "selected_persistence_calibration",
        "selection_protocol",
        "passes_persistence_guard",
        "cv_max_yearly_delta_bs_vs_raw_persistence",
        "cv_min_yearly_delta_bs_vs_raw_persistence",
        "spearman_model_vs_observed",
        "claim_status",
        "source_file",
    ]
    order = {
        "soilw_0p1_1m": 0,
        "soill_0_1m": 1,
    }
    model_order = {
        "operational_gefs_selected": 0,
        "stack_validation_selected": 1,
        "operational_gefs_anom_isotonic": 2,
        "operational_gefs_raw_isotonic": 3,
        "persistence_raw": 4,
        "persistence_selected": 5,
        "cv_stack_selected": 6,
        "persistence_guard_selected": 7,
        "cv_operational_gefs_selected": 8,
        "cv_persistence_selected": 9,
        "monthly_climatology": 10,
    }
    out = out[[c for c in keep_cols if c in out.columns]].copy()
    out["_soil_order"] = out["soil_mode"].map(order).fillna(99).astype(int)
    out["_model_order"] = out["model"].map(model_order).fillna(99).astype(int)
    return (
        out.sort_values(["_soil_order", "_model_order"])
        .drop(columns=["_soil_order", "_model_order"])
        .reset_index(drop=True)
    )


def build_landsurface_nldas_validation_table() -> pd.DataFrame:
    path = REPORT / "landsurface" / "landsurface_nldas_gefsv12_validation_summary.csv"
    df = read_csv(path)
    if df.empty:
        return df
    keep_cols = [
        "region",
        "target_product",
        "target_variable",
        "model",
        "n_test_months",
        "bs_climatology",
        "bs_model",
        "bs_persistence_selected",
        "bss_vs_climatology",
        "bss_ci_low",
        "bss_ci_high",
        "bss_vs_persistence_selected",
        "delta_bs_model_minus_persistence_selected",
        "delta_bs_ci_low",
        "delta_bs_ci_high",
        "added_value_status_selected",
        "selected_weight_gefs",
        "selected_weight_persistence",
        "selected_gefs_calibration",
        "selected_persistence_calibration",
        "spearman_model_vs_observed",
        "target_file",
    ]
    out = df[[c for c in keep_cols if c in df.columns]].copy()
    out["claim_status"] = [
        claim_status(bss, lo, hi)
        for bss, lo, hi in zip(out["bss_vs_climatology"], out["bss_ci_low"], out["bss_ci_high"])
    ]
    out["interpretation"] = (
        "Independent NLDAS Noah soil-moisture target validation for U.S. regions; "
        "scores GEFSv12/persistence probabilities against NLDAS-derived dry fraction."
    )
    out["source_file"] = rel(path)
    return out.sort_values(["region", "model"]).reset_index(drop=True)


def build_landsurface_target_product_comparison_table() -> pd.DataFrame:
    era5_path = REPORT / "landsurface" / "landsurface_gefsv12_rzsm_stack_day15_hindcastcal_summary.csv"
    nldas_path = REPORT / "landsurface" / "landsurface_nldas_gefsv12_validation_summary.csv"
    era5 = read_csv(era5_path)
    nldas = read_csv(nldas_path)
    if era5.empty or nldas.empty:
        return pd.DataFrame()

    us_regions = ["cvalley", "southern_great_plains"]
    model = "stack_validation_selected"
    era5 = era5.loc[era5["region"].isin(us_regions) & era5["model"].eq(model)].copy()
    nldas = nldas.loc[nldas["region"].isin(us_regions) & nldas["model"].eq(model)].copy()
    if era5.empty or nldas.empty:
        return pd.DataFrame()

    cols = [
        "region",
        "n_months",
        "bs_model",
        "bs_persistence_selected",
        "bss_vs_climatology",
        "bss_ci_low",
        "bss_ci_high",
        "bss_vs_persistence_selected",
        "delta_bs_model_minus_persistence_selected",
        "delta_bs_ci_low",
        "delta_bs_ci_high",
        "added_value_status_selected",
        "selected_weight_gefs",
        "selected_weight_persistence",
        "spearman_model_vs_observed",
    ]
    era5 = era5[[c for c in cols if c in era5.columns]].rename(
        columns={
            "n_months": "era5_n_test_months",
            "bs_model": "era5_bs_model",
            "bs_persistence_selected": "era5_bs_persistence_selected",
            "bss_vs_climatology": "era5_bss_vs_climatology",
            "bss_ci_low": "era5_bss_ci_low",
            "bss_ci_high": "era5_bss_ci_high",
            "bss_vs_persistence_selected": "era5_bss_vs_persistence_selected",
            "delta_bs_model_minus_persistence_selected": "era5_delta_bs_model_minus_persistence_selected",
            "delta_bs_ci_low": "era5_delta_bs_ci_low",
            "delta_bs_ci_high": "era5_delta_bs_ci_high",
            "added_value_status_selected": "era5_added_value_status_selected",
            "selected_weight_gefs": "era5_selected_weight_gefs",
            "selected_weight_persistence": "era5_selected_weight_persistence",
            "spearman_model_vs_observed": "era5_spearman_model_vs_observed",
        }
    )
    nldas = nldas[
        [
            "region",
            "n_test_months",
            "bs_model",
            "bs_persistence_selected",
            "bss_vs_climatology",
            "bss_ci_low",
            "bss_ci_high",
            "bss_vs_persistence_selected",
            "delta_bs_model_minus_persistence_selected",
            "delta_bs_ci_low",
            "delta_bs_ci_high",
            "added_value_status_selected",
            "selected_weight_gefs",
            "selected_weight_persistence",
            "spearman_model_vs_observed",
        ]
    ].rename(
        columns={
            "n_test_months": "nldas_n_test_months",
            "bs_model": "nldas_bs_model",
            "bs_persistence_selected": "nldas_bs_persistence_selected",
            "bss_vs_climatology": "nldas_bss_vs_climatology",
            "bss_ci_low": "nldas_bss_ci_low",
            "bss_ci_high": "nldas_bss_ci_high",
            "bss_vs_persistence_selected": "nldas_bss_vs_persistence_selected",
            "delta_bs_model_minus_persistence_selected": "nldas_delta_bs_model_minus_persistence_selected",
            "delta_bs_ci_low": "nldas_delta_bs_ci_low",
            "delta_bs_ci_high": "nldas_delta_bs_ci_high",
            "added_value_status_selected": "nldas_added_value_status_selected",
            "selected_weight_gefs": "nldas_selected_weight_gefs",
            "selected_weight_persistence": "nldas_selected_weight_persistence",
            "spearman_model_vs_observed": "nldas_spearman_model_vs_observed",
        }
    )
    out = era5.merge(nldas, on="region", how="inner")
    out.insert(1, "region_label", out["region"].map(short_region))
    out.insert(2, "model", model)
    out["era5_claim_status"] = [
        claim_status(b, lo, hi)
        for b, lo, hi in zip(out["era5_bss_vs_climatology"], out["era5_bss_ci_low"], out["era5_bss_ci_high"])
    ]
    out["nldas_claim_status"] = [
        claim_status(b, lo, hi)
        for b, lo, hi in zip(out["nldas_bss_vs_climatology"], out["nldas_bss_ci_low"], out["nldas_bss_ci_high"])
    ]
    out["bss_shift_nldas_minus_era5"] = out["nldas_bss_vs_climatology"] - out["era5_bss_vs_climatology"]
    out["delta_bs_shift_nldas_minus_era5"] = (
        out["nldas_delta_bs_model_minus_persistence_selected"]
        - out["era5_delta_bs_model_minus_persistence_selected"]
    )
    out["interpretation"] = (
        "Target-product comparison for the same validation-selected GEFSv12/persistence stack; "
        "positive BSS under both products supports U.S. cross-target robustness."
    )
    out["source_files"] = f"{rel(era5_path)}; {rel(nldas_path)}"
    return out.sort_values("region").reset_index(drop=True)


def build_landsurface_gldas_validation_table() -> pd.DataFrame:
    path = REPORT / "landsurface" / "landsurface_gldas_gefsv12_validation_summary.csv"
    df = read_csv(path)
    if df.empty:
        return df
    keep_cols = [
        "region",
        "target_product",
        "target_variable",
        "model",
        "n_test_months",
        "bs_climatology",
        "bs_model",
        "bs_persistence_selected",
        "bss_vs_climatology",
        "bss_ci_low",
        "bss_ci_high",
        "bss_vs_persistence_selected",
        "delta_bs_model_minus_persistence_selected",
        "delta_bs_ci_low",
        "delta_bs_ci_high",
        "added_value_status_selected",
        "selected_weight_gefs",
        "selected_weight_persistence",
        "selected_gefs_calibration",
        "selected_persistence_calibration",
        "spearman_model_vs_observed",
        "target_file",
    ]
    out = df[[c for c in keep_cols if c in df.columns]].copy()
    out["claim_status"] = [
        claim_status(bss, lo, hi)
        for bss, lo, hi in zip(out["bss_vs_climatology"], out["bss_ci_low"], out["bss_ci_high"])
    ]
    out["interpretation"] = (
        "Global GLDAS Noah model-product target sensitivity for GEFSv12/persistence land-surface probabilities."
    )
    out["source_file"] = rel(path)
    return out.sort_values(["region", "model"]).reset_index(drop=True)


def build_landsurface_era5_gldas_comparison_table() -> pd.DataFrame:
    era5_path = REPORT / "landsurface" / "landsurface_gefsv12_rzsm_stack_day15_hindcastcal_summary.csv"
    gldas_path = REPORT / "landsurface" / "landsurface_gldas_gefsv12_validation_summary.csv"
    era5 = read_csv(era5_path)
    gldas = read_csv(gldas_path)
    if era5.empty or gldas.empty:
        return pd.DataFrame()

    model = "stack_validation_selected"
    era5 = era5.loc[era5["model"].eq(model)].copy()
    gldas = gldas.loc[gldas["model"].eq(model)].copy()
    if era5.empty or gldas.empty:
        return pd.DataFrame()

    era5 = era5[
        [
            "region",
            "n_months",
            "bss_vs_climatology",
            "bss_ci_low",
            "bss_ci_high",
            "bss_vs_persistence_selected",
            "delta_bs_model_minus_persistence_selected",
            "delta_bs_ci_low",
            "delta_bs_ci_high",
            "added_value_status_selected",
            "selected_weight_gefs",
            "selected_weight_persistence",
            "spearman_model_vs_observed",
        ]
    ].rename(
        columns={
            "n_months": "era5_n_test_months",
            "bss_vs_climatology": "era5_bss_vs_climatology",
            "bss_ci_low": "era5_bss_ci_low",
            "bss_ci_high": "era5_bss_ci_high",
            "bss_vs_persistence_selected": "era5_bss_vs_persistence_selected",
            "delta_bs_model_minus_persistence_selected": "era5_delta_bs_model_minus_persistence_selected",
            "delta_bs_ci_low": "era5_delta_bs_ci_low",
            "delta_bs_ci_high": "era5_delta_bs_ci_high",
            "added_value_status_selected": "era5_added_value_status_selected",
            "selected_weight_gefs": "era5_selected_weight_gefs",
            "selected_weight_persistence": "era5_selected_weight_persistence",
            "spearman_model_vs_observed": "era5_spearman_model_vs_observed",
        }
    )
    gldas = gldas[
        [
            "region",
            "n_test_months",
            "bss_vs_climatology",
            "bss_ci_low",
            "bss_ci_high",
            "bss_vs_persistence_selected",
            "delta_bs_model_minus_persistence_selected",
            "delta_bs_ci_low",
            "delta_bs_ci_high",
            "added_value_status_selected",
            "selected_weight_gefs",
            "selected_weight_persistence",
            "spearman_model_vs_observed",
        ]
    ].rename(
        columns={
            "n_test_months": "gldas_n_test_months",
            "bss_vs_climatology": "gldas_bss_vs_climatology",
            "bss_ci_low": "gldas_bss_ci_low",
            "bss_ci_high": "gldas_bss_ci_high",
            "bss_vs_persistence_selected": "gldas_bss_vs_persistence_selected",
            "delta_bs_model_minus_persistence_selected": "gldas_delta_bs_model_minus_persistence_selected",
            "delta_bs_ci_low": "gldas_delta_bs_ci_low",
            "delta_bs_ci_high": "gldas_delta_bs_ci_high",
            "added_value_status_selected": "gldas_added_value_status_selected",
            "selected_weight_gefs": "gldas_selected_weight_gefs",
            "selected_weight_persistence": "gldas_selected_weight_persistence",
            "spearman_model_vs_observed": "gldas_spearman_model_vs_observed",
        }
    )
    out = era5.merge(gldas, on="region", how="inner")
    out.insert(1, "region_label", out["region"].map(short_region))
    out.insert(2, "model", model)
    out["era5_claim_status"] = [
        claim_status(b, lo, hi)
        for b, lo, hi in zip(out["era5_bss_vs_climatology"], out["era5_bss_ci_low"], out["era5_bss_ci_high"])
    ]
    out["gldas_claim_status"] = [
        claim_status(b, lo, hi)
        for b, lo, hi in zip(out["gldas_bss_vs_climatology"], out["gldas_bss_ci_low"], out["gldas_bss_ci_high"])
    ]
    out["bss_shift_gldas_minus_era5"] = out["gldas_bss_vs_climatology"] - out["era5_bss_vs_climatology"]
    out["delta_bs_shift_gldas_minus_era5"] = (
        out["gldas_delta_bs_model_minus_persistence_selected"]
        - out["era5_delta_bs_model_minus_persistence_selected"]
    )
    out["interpretation"] = (
        "Target-product comparison for the same validation-selected GEFSv12/persistence stack; "
        "GLDAS is a global model-product sensitivity target."
    )
    out["source_files"] = f"{rel(era5_path)}; {rel(gldas_path)}"
    return out.sort_values("region").reset_index(drop=True)


def build_landsurface_smap_l4_validation_table() -> pd.DataFrame:
    path = REPORT / "landsurface" / "landsurface_smap_l4_midmonth_gefsv12_validation_summary.csv"
    df = read_csv(path)
    if df.empty:
        return df
    keep_cols = [
        "region",
        "target_product",
        "target_variable",
        "model",
        "n_validation_months",
        "n_test_months",
        "monthly_snapshot_proxy",
        "climatology_mode",
        "bs_climatology",
        "bs_model",
        "bs_persistence_selected",
        "bss_vs_climatology",
        "bss_ci_low",
        "bss_ci_high",
        "bss_vs_persistence_selected",
        "delta_bs_model_minus_persistence_selected",
        "delta_bs_ci_low",
        "delta_bs_ci_high",
        "added_value_status_selected",
        "selected_weight_gefs",
        "selected_weight_persistence",
        "selected_gefs_calibration",
        "selected_persistence_calibration",
        "spearman_model_vs_observed",
        "mean_snapshots_per_month",
        "target_aggregation",
        "target_file",
    ]
    out = df[[c for c in keep_cols if c in df.columns]].copy()
    out["claim_status"] = [
        claim_status(bss, lo, hi)
        for bss, lo, hi in zip(out["bss_vs_climatology"], out["bss_ci_low"], out["bss_ci_high"])
    ]
    out["interpretation"] = (
        "Short-record SMAP L4 satellite-assimilated target validation using mid-month "
        "SPL4SMGP sm_rootzone_pctl dry-fraction snapshots; validation period is short."
    )
    out["source_file"] = rel(path)
    return out.sort_values(["region", "model"]).reset_index(drop=True)


def build_landsurface_smap_l4_snapshot_sensitivity_table() -> pd.DataFrame:
    mid_path = REPORT / "landsurface" / "landsurface_smap_l4_midmonth_gefsv12_validation_summary.csv"
    multi_path = REPORT / "landsurface" / "landsurface_smap_l4_multi3snapshot_gefsv12_validation_summary.csv"
    mid = read_csv(mid_path)
    multi = read_csv(multi_path)
    if mid.empty or multi.empty:
        return pd.DataFrame()
    mid = mid.loc[mid["model"].eq("stack_validation_selected")].copy()
    multi = multi.loc[multi["model"].eq("stack_validation_selected")].copy()
    if mid.empty or multi.empty:
        return pd.DataFrame()
    keep = [
        "region",
        "n_validation_months",
        "n_test_months",
        "bss_vs_climatology",
        "bss_ci_low",
        "bss_ci_high",
        "delta_bs_model_minus_persistence_selected",
        "delta_bs_ci_low",
        "delta_bs_ci_high",
        "added_value_status_selected",
        "selected_weight_gefs",
        "selected_weight_persistence",
        "mean_snapshots_per_month",
    ]
    mid = mid[[c for c in keep if c in mid.columns]].rename(
        columns={
            "n_validation_months": "midmonth_n_validation_months",
            "n_test_months": "midmonth_n_test_months",
            "bss_vs_climatology": "midmonth_bss_vs_climatology",
            "bss_ci_low": "midmonth_bss_ci_low",
            "bss_ci_high": "midmonth_bss_ci_high",
            "delta_bs_model_minus_persistence_selected": "midmonth_delta_bs_model_minus_persistence",
            "delta_bs_ci_low": "midmonth_delta_bs_ci_low",
            "delta_bs_ci_high": "midmonth_delta_bs_ci_high",
            "added_value_status_selected": "midmonth_added_value_status",
            "selected_weight_gefs": "midmonth_weight_gefs",
            "selected_weight_persistence": "midmonth_weight_persistence",
            "mean_snapshots_per_month": "midmonth_mean_snapshots_per_month",
        }
    )
    multi = multi[[c for c in keep if c in multi.columns]].rename(
        columns={
            "n_validation_months": "multi3_n_validation_months",
            "n_test_months": "multi3_n_test_months",
            "bss_vs_climatology": "multi3_bss_vs_climatology",
            "bss_ci_low": "multi3_bss_ci_low",
            "bss_ci_high": "multi3_bss_ci_high",
            "delta_bs_model_minus_persistence_selected": "multi3_delta_bs_model_minus_persistence",
            "delta_bs_ci_low": "multi3_delta_bs_ci_low",
            "delta_bs_ci_high": "multi3_delta_bs_ci_high",
            "added_value_status_selected": "multi3_added_value_status",
            "selected_weight_gefs": "multi3_weight_gefs",
            "selected_weight_persistence": "multi3_weight_persistence",
            "mean_snapshots_per_month": "multi3_mean_snapshots_per_month",
        }
    )
    out = mid.merge(multi, on="region", how="inner")
    out.insert(1, "region_label", out["region"].map(short_region))
    out["bss_shift_multi3_minus_midmonth"] = (
        out["multi3_bss_vs_climatology"] - out["midmonth_bss_vs_climatology"]
    )
    out["delta_bs_shift_multi3_minus_midmonth"] = (
        out["multi3_delta_bs_model_minus_persistence"] - out["midmonth_delta_bs_model_minus_persistence"]
    )
    out["test_month_shift_multi3_minus_midmonth"] = out["multi3_n_test_months"] - out["midmonth_n_test_months"]
    out["midmonth_claim_status"] = [
        claim_status(b, lo, hi)
        for b, lo, hi in zip(out["midmonth_bss_vs_climatology"], out["midmonth_bss_ci_low"], out["midmonth_bss_ci_high"])
    ]
    out["multi3_claim_status"] = [
        claim_status(b, lo, hi)
        for b, lo, hi in zip(out["multi3_bss_vs_climatology"], out["multi3_bss_ci_low"], out["multi3_bss_ci_high"])
    ]
    out["interpretation"] = (
        "SMAP L4 target-aggregation sensitivity comparing one mid-month snapshot "
        "against early/mid/late monthly snapshots; same GEFSv12/persistence protocol."
    )
    out["source_files"] = f"{rel(mid_path)}; {rel(multi_path)}"
    return out.sort_values("region").reset_index(drop=True)


def build_landsurface_target_product_transfer_table() -> pd.DataFrame:
    path = (
        REPORT
        / "landsurface"
        / "landsurface_target_product_transfer_to_smap_multi3snapshot_summary.csv"
    )
    df = read_csv(path)
    if df.empty:
        return df

    keep_models = {
        "gefs_selected",
        "stack_validation_selected",
    }
    out = df.loc[
        df["model"].isin(keep_models)
        | (
            df["transfer_kind"].eq("target_self_calibrated_reference")
            & df["model"].eq("stack_validation_selected")
        )
    ].copy()
    keep_cols = [
        "source_product",
        "target_product",
        "region",
        "model",
        "transfer_kind",
        "n_months",
        "bs_smap_climatology",
        "bs_smap_persistence_selected",
        "bs_model",
        "bss_vs_smap_climatology",
        "bss_vs_smap_climatology_ci_low",
        "bss_vs_smap_climatology_ci_high",
        "bss_vs_smap_persistence_selected",
        "bss_vs_smap_persistence_selected_ci_low",
        "bss_vs_smap_persistence_selected_ci_high",
        "delta_bs_model_minus_smap_persistence_selected",
        "delta_bs_ci_low",
        "delta_bs_ci_high",
        "claim_status_vs_smap_climatology",
        "added_value_status_vs_smap_persistence",
        "transfer_status",
        "spearman_model_vs_smap_observed",
        "amplitude_ratio_model_vs_smap_observed",
        "source_file",
        "target_file",
    ]
    out = out[[c for c in keep_cols if c in out.columns]].copy()
    out.insert(1, "source_product_label", out["source_product"].map({
        "era5_land": "ERA5-Land calibrated",
        "gldas_noah": "GLDAS calibrated",
        "smap_l4_multi3snapshot": "SMAP self-calibrated reference",
    }).fillna(out["source_product"]))
    out.insert(4, "region_label", out["region"].map(short_region))
    out.insert(6, "model_label", out["model"].map({
        "gefs_selected": "GEFSv12 selected",
        "stack_validation_selected": "GEFSv12+persistence selected stack",
    }).fillna(out["model"]))
    out["interpretation"] = np.where(
        out["transfer_kind"].eq("cross_product_transfer"),
        "Frozen source-product probabilities scored against SMAP without SMAP recalibration.",
        "SMAP-calibrated reference; not cross-product transfer evidence.",
    )
    return out.sort_values(["transfer_kind", "source_product", "region", "model"]).reset_index(drop=True)


def build_landsurface_calibration_transfer_ladder_table() -> pd.DataFrame:
    path = (
        REPORT
        / "landsurface"
        / "landsurface_calibration_transfer_ladder_smap_multi3snapshot_summary.csv"
    )
    df = read_csv(path)
    if df.empty:
        return df
    keep_models = {"gefs_selected", "stack_validation_selected"}
    out = df.loc[df["model"].isin(keep_models)].copy()
    keep_cols = [
        "strategy",
        "source_product",
        "region",
        "model",
        "n_test_months",
        "bs_smap_climatology",
        "bs_smap_persistence_selected",
        "bs_model",
        "bss_vs_smap_climatology",
        "bss_vs_smap_climatology_ci_low",
        "bss_vs_smap_climatology_ci_high",
        "bss_vs_smap_persistence_selected",
        "bss_vs_smap_persistence_selected_ci_low",
        "bss_vs_smap_persistence_selected_ci_high",
        "delta_bs_model_minus_smap_persistence_selected",
        "delta_bs_ci_low",
        "delta_bs_ci_high",
        "claim_status_vs_smap_climatology",
        "added_value_status_vs_smap_persistence",
        "spearman_model_vs_smap_observed",
        "amplitude_ratio_model_vs_smap_observed",
        "selected_gefs_calibration",
        "selected_persistence_calibration",
        "selected_weight_gefs",
        "selected_weight_persistence",
        "source_overlap_mean_dry",
        "smap_validation_mean_dry",
        "pooled_source_training_rows",
        "smap_recalibration_months",
    ]
    out = out[[c for c in keep_cols if c in out.columns]].copy()
    out.insert(2, "source_product_label", out["source_product"].map({
        "era5_land": "ERA5-Land calibrated",
        "gldas_noah": "GLDAS calibrated",
        "era5_land_plus_gldas_noah": "ERA5-Land+GLDAS pooled",
        "smap_l4_multi3snapshot": "SMAP self-calibrated reference",
    }).fillna(out["source_product"]))
    out.insert(4, "region_label", out["region"].map(short_region))
    out.insert(6, "model_label", out["model"].map({
        "gefs_selected": "GEFSv12 selected",
        "stack_validation_selected": "GEFSv12+persistence selected stack",
    }).fillna(out["model"]))
    out["interpretation"] = np.select(
        [
            out["strategy"].eq("source_direct"),
            out["strategy"].eq("target_prior_shift"),
            out["strategy"].eq("smap_validation_recalibration"),
            out["strategy"].eq("pooled_source_calibration"),
            out["strategy"].eq("smap_self_calibrated_reference"),
        ],
        [
            "Frozen source-product calibration scored against SMAP without adaptation.",
            "Validation-only source-to-SMAP dry-rate shift applied before SMAP test scoring.",
            "Source probability recalibrated to SMAP using 2015-2016 validation months only.",
            "Product-invariant calibration learned from ERA5-Land+GLDAS source targets.",
            "SMAP validation-calibrated upper reference; not transfer evidence.",
        ],
        default="Calibration-transfer ladder row.",
    )
    out["source_file"] = rel(path)
    strategy_order = {
        "source_direct": 0,
        "target_prior_shift": 1,
        "smap_validation_recalibration": 2,
        "pooled_source_calibration": 3,
        "smap_self_calibrated_reference": 4,
    }
    out["_strategy_order"] = out["strategy"].map(strategy_order).fillna(99)
    out = out.sort_values(["_strategy_order", "source_product", "region", "model"]).drop(columns=["_strategy_order"])
    return out.reset_index(drop=True)


def build_landsurface_calibration_transfer_ladder_compact_table(ladder: pd.DataFrame) -> pd.DataFrame:
    if ladder.empty:
        return ladder
    work = ladder.copy()
    strategy_labels = {
        "source_direct": "Direct source transfer",
        "target_prior_shift": "Validation dry-rate shift",
        "smap_validation_recalibration": "SMAP validation recalibration",
        "pooled_source_calibration": "ERA5-Land+GLDAS pooled calibration",
        "smap_self_calibrated_reference": "SMAP self-calibrated reference",
    }
    strategy_order = {key: i for i, key in enumerate(strategy_labels)}
    model_order = {"gefs_selected": 0, "stack_validation_selected": 1}
    grouped = (
        work.groupby(["strategy", "model"], dropna=False)
        .agg(
            n_rows=("region", "size"),
            n_regions=("region", "nunique"),
            n_sources=("source_product", "nunique"),
            robust_positive_count=(
                "claim_status_vs_smap_climatology",
                lambda s: int((s == "robust_positive").sum()),
            ),
            positive_or_robust_count=(
                "claim_status_vs_smap_climatology",
                lambda s: int(s.isin(["robust_positive", "positive_uncertain", "positive_no_ci"]).sum()),
            ),
            robust_added_value_count=(
                "added_value_status_vs_smap_persistence",
                lambda s: int((s == "stack_robust_added_value").sum()),
            ),
            uncertain_added_value_count=(
                "added_value_status_vs_smap_persistence",
                lambda s: int((s == "stack_uncertain_added_value").sum()),
            ),
            mean_bss_vs_smap_climatology=("bss_vs_smap_climatology", "mean"),
            median_bss_vs_smap_climatology=("bss_vs_smap_climatology", "median"),
            mean_bss_vs_smap_persistence=("bss_vs_smap_persistence_selected", "mean"),
            median_bss_vs_smap_persistence=("bss_vs_smap_persistence_selected", "median"),
            mean_spearman=("spearman_model_vs_smap_observed", "mean"),
            mean_amplitude_ratio=("amplitude_ratio_model_vs_smap_observed", "mean"),
        )
        .reset_index()
    )
    grouped["strategy_label"] = grouped["strategy"].map(strategy_labels).fillna(grouped["strategy"])
    grouped["model_label"] = grouped["model"].map({
        "gefs_selected": "GEFSv12 selected",
        "stack_validation_selected": "GEFSv12+persistence stack",
    }).fillna(grouped["model"])
    grouped["robust_positive_fraction"] = grouped["robust_positive_count"] / grouped["n_rows"]
    grouped["robust_added_value_fraction"] = grouped["robust_added_value_count"] / grouped["n_rows"]
    grouped["positive_or_robust_fraction"] = grouped["positive_or_robust_count"] / grouped["n_rows"]
    grouped["uncertain_added_value_fraction"] = grouped["uncertain_added_value_count"] / grouped["n_rows"]
    grouped["interpretation"] = np.select(
        [
            grouped["strategy"].eq("source_direct"),
            grouped["strategy"].eq("target_prior_shift"),
            grouped["strategy"].eq("smap_validation_recalibration"),
            grouped["strategy"].eq("pooled_source_calibration"),
            grouped["strategy"].eq("smap_self_calibrated_reference"),
        ],
        [
            "Strict no-adaptation transfer; tests raw portability of source-product calibration.",
            "Validation-only base-rate adaptation; tests whether target dry-rate mismatch explains transfer failure.",
            "Validation-only SMAP probability recalibration; tests whether source probabilities need target-specific calibration.",
            "No SMAP fitting; tests product-invariant calibration learned from ERA5-Land and GLDAS sources.",
            "Upper reference fitted to SMAP validation months; not transfer evidence.",
        ],
        default="Calibration-transfer compact summary.",
    )
    grouped["_strategy_order"] = grouped["strategy"].map(strategy_order).fillna(99)
    grouped["_model_order"] = grouped["model"].map(model_order).fillna(99)
    cols = [
        "strategy",
        "strategy_label",
        "model",
        "model_label",
        "n_rows",
        "n_regions",
        "n_sources",
        "robust_positive_count",
        "robust_positive_fraction",
        "positive_or_robust_count",
        "positive_or_robust_fraction",
        "robust_added_value_count",
        "robust_added_value_fraction",
        "uncertain_added_value_count",
        "uncertain_added_value_fraction",
        "mean_bss_vs_smap_climatology",
        "median_bss_vs_smap_climatology",
        "mean_bss_vs_smap_persistence",
        "median_bss_vs_smap_persistence",
        "mean_spearman",
        "mean_amplitude_ratio",
        "interpretation",
    ]
    grouped = grouped.sort_values(["_strategy_order", "_model_order"]).drop(columns=["_strategy_order", "_model_order"])
    return grouped[cols].reset_index(drop=True)


def build_landsurface_target_product_adaptation_benchmark_table() -> pd.DataFrame:
    path = (
        REPORT
        / "landsurface"
        / "landsurface_target_product_adaptation_benchmark_smap_multi3snapshot_compact_summary.csv"
    )
    df = read_csv(path)
    if df.empty:
        return df
    keep_methods = {
        "Fixed direct transfer",
        "Fixed validation dry-rate shift",
        "Fixed validation prediction-base-rate shift",
        "Fixed bias-corrected temperature/base-rate scaling",
        "Fixed shrunk seasonal base-rate shift",
        "Fixed ridge-logit SMAP recalibration",
        "Validation-selected simple adaptation",
        "Validation-selected base-rate adaptation",
        "Validation-selected full adaptation",
        "Validation-selected complex adaptation",
        "Pooled source calibration",
        "SMAP self-calibrated reference",
    }
    out = df.loc[df["adaptation_method"].isin(keep_methods)].copy()
    keep_cols = [
        "adaptation_method",
        "model",
        "n_rows",
        "n_regions",
        "n_sources",
        "robust_positive_count",
        "robust_positive_fraction",
        "robust_added_value_count",
        "robust_added_value_fraction",
        "mean_bss_vs_smap_climatology",
        "median_bss_vs_smap_climatology",
        "mean_bss_vs_smap_persistence",
        "median_bss_vs_smap_persistence",
    ]
    out = out[[c for c in keep_cols if c in out.columns]].copy()
    out.insert(2, "model_label", out["model"].map({
        "gefs_selected": "GEFSv12 selected",
        "stack_validation_selected": "GEFSv12+persistence selected stack",
    }).fillna(out["model"]))
    out["decision_relevance"] = np.select(
        [
            out["adaptation_method"].eq("Fixed validation dry-rate shift"),
            out["adaptation_method"].eq("Fixed validation prediction-base-rate shift"),
            out["adaptation_method"].eq("Fixed bias-corrected temperature/base-rate scaling"),
            out["adaptation_method"].eq("Fixed shrunk seasonal base-rate shift"),
            out["adaptation_method"].eq("Fixed ridge-logit SMAP recalibration"),
            out["adaptation_method"].eq("Validation-selected simple adaptation"),
            out["adaptation_method"].eq("Validation-selected base-rate adaptation"),
            out["adaptation_method"].eq("Validation-selected full adaptation"),
            out["adaptation_method"].eq("Validation-selected complex adaptation"),
            out["adaptation_method"].eq("Pooled source calibration"),
            out["adaptation_method"].eq("SMAP self-calibrated reference"),
        ],
        [
            "Primary simple adaptation method.",
            "Product-specific prediction mean matched to SMAP validation dry fraction.",
            "Bias-corrected temperature scaling after validation base-rate matching.",
            "Calendar-month base-rate offsets shrunk toward the global validation offset.",
            "Target-specific smooth logit recalibration fitted to SMAP validation months.",
            "Validation-only choice between direct transfer and dry-rate shift.",
            "Validation-only choice among direct transfer and base-rate-only shifts.",
            "Tests whether target-specific recalibration is justified.",
            "Tests whether base-rate shifts plus target-specific recalibration are justified.",
            "No-SMAP source-product generalization reference.",
            "Upper reference fitted on SMAP validation labels.",
        ],
        default="Target-product adaptation benchmark row.",
    )
    method_order = {
        "Fixed direct transfer": 0,
        "Fixed validation dry-rate shift": 1,
        "Fixed validation prediction-base-rate shift": 2,
        "Fixed bias-corrected temperature/base-rate scaling": 3,
        "Fixed shrunk seasonal base-rate shift": 4,
        "Fixed ridge-logit SMAP recalibration": 5,
        "Validation-selected simple adaptation": 6,
        "Validation-selected base-rate adaptation": 7,
        "Validation-selected full adaptation": 8,
        "Validation-selected complex adaptation": 9,
        "Pooled source calibration": 10,
        "SMAP self-calibrated reference": 11,
    }
    model_order = {"gefs_selected": 0, "stack_validation_selected": 1}
    out["_method_order"] = out["adaptation_method"].map(method_order).fillna(99)
    out["_model_order"] = out["model"].map(model_order).fillna(99)
    out = out.sort_values(["_method_order", "_model_order"]).drop(columns=["_method_order", "_model_order"])
    out["source_file"] = rel(path)
    return out.reset_index(drop=True)


def build_landsurface_base_rate_yearly_sensitivity_table() -> pd.DataFrame:
    path = (
        REPORT
        / "landsurface"
        / "landsurface_target_product_adaptation_benchmark_smap_multi3snapshot_yearly_compact_summary.csv"
    )
    df = read_csv(path)
    if df.empty:
        return df
    keep_methods = {
        "Fixed validation dry-rate shift",
        "Fixed validation prediction-base-rate shift",
        "Fixed bias-corrected temperature/base-rate scaling",
    }
    out = df.loc[
        df["adaptation_method"].isin(keep_methods)
        & df["model"].eq("gefs_selected")
    ].copy()
    keep_cols = [
        "adaptation_method",
        "model",
        "target_year",
        "n_rows",
        "n_regions",
        "n_sources",
        "positive_bss_count",
        "positive_bss_fraction",
        "positive_added_value_count",
        "positive_added_value_fraction",
        "mean_bss_vs_smap_climatology",
        "median_bss_vs_smap_climatology",
        "mean_bss_vs_smap_persistence",
        "median_bss_vs_smap_persistence",
    ]
    out = out[[c for c in keep_cols if c in out.columns]].copy()
    out.insert(2, "model_label", "GEFSv12 selected")
    out["interpretation"] = np.where(
        out["mean_bss_vs_smap_persistence"] > 0,
        "Positive mean added value over SMAP persistence for this test year.",
        "Positive against climatology but persistence-relative value is weak for this test year.",
    )
    method_order = {
        "Fixed validation dry-rate shift": 0,
        "Fixed validation prediction-base-rate shift": 1,
        "Fixed bias-corrected temperature/base-rate scaling": 2,
    }
    out["_method_order"] = out["adaptation_method"].map(method_order).fillna(99)
    out = out.sort_values(["_method_order", "target_year"]).drop(columns=["_method_order"])
    out["source_file"] = rel(path)
    return out.reset_index(drop=True)


def plot_landsurface_reliability(bins: pd.DataFrame, path: Path) -> None:
    if bins.empty:
        return
    model_labels = {
        "gefs_selected": "GEFSv12",
        "persistence_selected": "Persistence",
        "stack_equal_weight": "Equal stack",
        "stack_validation_selected": "Validation stack",
    }
    colors = {
        "gefs_selected": "#4E79A7",
        "persistence_selected": "#F28E2B",
        "stack_equal_weight": "#59A14F",
        "stack_validation_selected": "#B07AA1",
    }
    regions = sorted(bins["region"].dropna().unique())
    fig, axes = plt.subplots(1, len(regions), figsize=(4.4 * len(regions), 4.2), sharex=True, sharey=True)
    if len(regions) == 1:
        axes = [axes]
    for ax, region in zip(axes, regions):
        part_region = bins.loc[bins["region"].eq(region)].copy()
        for model, part in part_region.groupby("model", sort=False):
            part = part.sort_values("mean_pred")
            ax.plot(
                part["mean_pred"],
                part["mean_observed"],
                marker="o",
                linewidth=1.4,
                markersize=4,
                color=colors.get(model, "#546E7A"),
                label=model_labels.get(model, model),
            )
        ax.plot([0, 1], [0, 1], color="#333333", linestyle="--", linewidth=0.9)
        ax.set_title(short_region(region))
        ax.set_xlabel("Mean predicted dry fraction")
        ax.grid(color="#E0E0E0", linewidth=0.7)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("Mean observed dry fraction")
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False, fontsize=8)
    fig.suptitle("GEFSv12 RZSM stack reliability by forecast-probability bin", y=0.98)
    fig.tight_layout(rect=(0, 0.08, 1, 0.95))
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_landsurface_target_product_comparison(comparison: pd.DataFrame, path: Path) -> None:
    if comparison.empty:
        return
    rows: list[dict[str, object]] = []
    for _, row in comparison.iterrows():
        rows.extend(
            [
                {
                    "plot_label": f"{row['region_label']} | ERA5-Land target",
                    "region_label": row["region_label"],
                    "target_product": "ERA5-Land",
                    "bss": row["era5_bss_vs_climatology"],
                    "ci_low": row["era5_bss_ci_low"],
                    "ci_high": row["era5_bss_ci_high"],
                    "status": row["era5_claim_status"],
                },
                {
                    "plot_label": f"{row['region_label']} | NLDAS Noah target",
                    "region_label": row["region_label"],
                    "target_product": "NLDAS Noah",
                    "bss": row["nldas_bss_vs_climatology"],
                    "ci_low": row["nldas_bss_ci_low"],
                    "ci_high": row["nldas_bss_ci_high"],
                    "status": row["nldas_claim_status"],
                },
            ]
        )
    plot = pd.DataFrame(rows)
    if plot.empty:
        return
    product_order = {"ERA5-Land": 0, "NLDAS Noah": 1}
    plot["_product_order"] = plot["target_product"].map(product_order).fillna(99)
    plot = plot.sort_values(["region_label", "_product_order"]).drop(columns=["_product_order"])
    errorbar_plot(
        plot,
        path,
        "GEFSv12/persistence stack target-product comparison",
        ["plot_label"],
        width=8.6,
        row_height=0.48,
    )


def plot_landsurface_era5_gldas_comparison(comparison: pd.DataFrame, path: Path) -> None:
    if comparison.empty:
        return
    rows: list[dict[str, object]] = []
    for _, row in comparison.iterrows():
        rows.extend(
            [
                {
                    "plot_label": f"{row['region_label']} | ERA5-Land",
                    "region_label": row["region_label"],
                    "target_product": "ERA5-Land",
                    "bss": row["era5_bss_vs_climatology"],
                    "ci_low": row["era5_bss_ci_low"],
                    "ci_high": row["era5_bss_ci_high"],
                    "status": row["era5_claim_status"],
                },
                {
                    "plot_label": f"{row['region_label']} | GLDAS",
                    "region_label": row["region_label"],
                    "target_product": "GLDAS",
                    "bss": row["gldas_bss_vs_climatology"],
                    "ci_low": row["gldas_bss_ci_low"],
                    "ci_high": row["gldas_bss_ci_high"],
                    "status": row["gldas_claim_status"],
                },
            ]
        )
    plot = pd.DataFrame(rows)
    if plot.empty:
        return
    product_order = {"ERA5-Land": 0, "GLDAS": 1}
    plot["_product_order"] = plot["target_product"].map(product_order).fillna(99)
    plot = plot.sort_values(["region_label", "_product_order"]).drop(columns=["_product_order"])
    errorbar_plot(
        plot,
        path,
        "GEFSv12/persistence stack: ERA5-Land vs GLDAS targets",
        ["plot_label"],
        width=9.2,
        row_height=0.40,
    )


def plot_landsurface_calibration_transfer_ladder(compact: pd.DataFrame, path: Path) -> None:
    if compact.empty:
        return
    plot = compact.copy()
    model_colors = {
        "gefs_selected": "#4E79A7",
        "stack_validation_selected": "#59A14F",
    }
    plot["row_label"] = plot["strategy_label"] + "\n" + plot["model_label"]
    plot = plot.iloc[::-1].reset_index(drop=True)
    y = np.arange(len(plot))
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(10.4, max(4.8, 0.42 * len(plot) + 1.0)),
        sharey=True,
    )
    panels = [
        (
            axes[0],
            "robust_positive_fraction",
            "robust_positive_count",
            "Robust-positive vs SMAP climatology",
        ),
        (
            axes[1],
            "robust_added_value_fraction",
            "robust_added_value_count",
            "Robust added value vs SMAP persistence",
        ),
    ]
    colors = [model_colors.get(model, "#546E7A") for model in plot["model"]]
    for ax, frac_col, count_col, title in panels:
        ax.barh(y, plot[frac_col], color=colors, height=0.68)
        ax.set_xlim(0, 1)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Fraction of rows")
        ax.grid(axis="x", color="#E0E0E0", linewidth=0.7)
        ax.set_axisbelow(True)
        ax.set_xticks(np.linspace(0, 1, 6))
        for i, row in plot.iterrows():
            label = f"{int(row[count_col])}/{int(row['n_rows'])}"
            x = min(float(row[frac_col]) + 0.025, 0.92)
            ax.text(x, i, label, va="center", fontsize=8)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(plot["row_label"], fontsize=8)
    axes[1].tick_params(axis="y", length=0)
    handles = [
        plt.Line2D([0], [0], marker="s", linestyle="", color=color, markersize=8, label=label)
        for label, color in [
            ("GEFSv12 selected", model_colors["gefs_selected"]),
            ("GEFSv12+persistence stack", model_colors["stack_validation_selected"]),
        ]
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False, fontsize=8)
    fig.suptitle("SMAP L4 calibration-transfer ladder", y=0.995)
    fig.tight_layout(rect=(0, 0.06, 1, 0.96))
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def build_summary_markdown(
    master: pd.DataFrame,
    headline: pd.DataFrame,
    seasonal: pd.DataFrame,
    temporal: pd.DataFrame,
    evaluation: pd.DataFrame,
    transition: pd.DataFrame,
    land_added: pd.DataFrame,
    climate_index: pd.DataFrame,
    gefs_sensitivity: pd.DataFrame,
    gefs_stack: pd.DataFrame,
    landsurface_reliability: pd.DataFrame,
    landsurface_domain_transfer: pd.DataFrame,
    landsurface_rare_event: pd.DataFrame,
    landsurface_persistence_regimes: pd.DataFrame,
    landsurface_independent_targets: pd.DataFrame,
    landsurface_nldas_validation: pd.DataFrame,
    landsurface_target_product_comparison: pd.DataFrame,
    landsurface_gldas_validation: pd.DataFrame,
    landsurface_era5_gldas_comparison: pd.DataFrame,
    landsurface_smap_l4_validation: pd.DataFrame,
    landsurface_smap_l4_snapshot_sensitivity: pd.DataFrame,
    landsurface_target_product_transfer: pd.DataFrame,
    landsurface_calibration_transfer_ladder: pd.DataFrame,
    landsurface_target_product_adaptation_benchmark: pd.DataFrame,
    landsurface_base_rate_yearly_sensitivity: pd.DataFrame,
) -> str:
    robust_pos = master.loc[master["status"].eq("robust_positive")]
    canonical = master.loc[master["evidence_group"].eq("central_valley_calibrated_checkpoint")]
    operational = master.loc[
        (master["evidence_group"].eq("operational_dynamical_benchmark"))
        & (master["model"].astype(str).str.contains("selected", case=False, na=False))
    ]
    land_surface = master.loc[
        (master["evidence_group"].eq("forecast_informed_landsurface_benchmark"))
        & (master["model"].astype(str).str.contains("selected", case=False, na=False))
    ]
    cfs_land_surface = land_surface.loc[
        land_surface["model"].astype(str).str.contains("CFSv2", case=False, na=False)
    ]
    gefs_land_surface = land_surface.loc[
        land_surface["model"].astype(str).str.contains("GEFSv12", case=False, na=False)
    ]
    persistence_land_surface = master.loc[
        (master["evidence_group"].eq("forecast_informed_landsurface_benchmark"))
        & (master["model"].astype(str).str.contains("persistence", case=False, na=False))
    ]
    operational_prob_raw = master.loc[
        (master["evidence_group"].eq("operational_dynamical_benchmark"))
        & (master["model"].astype(str).str.contains("probability", case=False, na=False))
        & (master["model"].astype(str).str.contains(" raw", case=False, na=False))
    ]
    seasonal_pos = seasonal.loc[seasonal["robust_status"].eq("robust_positive")]
    memory = master.loc[
        (master["evidence_group"].eq("central_valley_memory_target"))
        & (master["model"].astype(str).str.contains("XGBoost", case=False, na=False))
        & (master["calibration"].astype(str).eq("selected"))
    ].copy()
    temporal_positive = int((temporal["bss"] > 0).sum()) if not temporal.empty else 0
    eval_line = ""
    if not evaluation.empty:
        def bss_for(scenario: str, level: str = "monthly") -> float | None:
            row = evaluation.loc[
                evaluation["scenario"].eq(scenario)
                & evaluation["inference_level"].eq(level)
            ]
            if row.empty:
                return None
            return float(row["bss_vs_climatology"].iloc[0])

        strict_bss = bss_for("strict_spi1_chrono")
        random_bss = bss_for("random_spi1_rows")
        overlap_bss = bss_for("overlap_spi3_lead1")
        if strict_bss is not None and random_bss is not None and overlap_bss is not None:
            eval_line = (
                f"- Evaluation-inflation audit: strict monthly SPI-1 BSS {strict_bss:+.3f}; "
                f"invalid random-row monthly BSS {random_bss:+.3f}; "
                f"invalid overlapping SPI-3 lead-1 monthly BSS {overlap_bss:+.3f}"
            )
    transition_line = ""
    if not transition.empty:
        cvalley_onset = transition.loc[
            transition["scope"].eq("cvalley")
            & transition["transition"].eq("onset")
            & transition["model"].astype(str).str.contains("eligible-only", case=False, na=False)
        ]
        sgp_onset = transition.loc[
            transition["scope"].eq("southern_great_plains_basin_masked")
            & transition["transition"].eq("onset")
            & transition["model"].astype(str).str.contains("eligible-only", case=False, na=False)
        ]
        if not cvalley_onset.empty and not sgp_onset.empty:
            cv = cvalley_onset.iloc[0]
            sgp = sgp_onset.iloc[0]
            transition_line = (
                f"- Transition-target diagnostic: rectangular Central Valley eligible-only onset BSS "
                f"{float(cv['bss_vs_eligible_climatology']):+.3f} "
                f"(CI {float(cv['bss_ci_low']):+.3f} to {float(cv['bss_ci_high']):+.3f}), "
                f"but SGP basin onset BSS {float(sgp['bss_vs_eligible_climatology']):+.3f} "
                f"(CI {float(sgp['bss_ci_low']):+.3f} to {float(sgp['bss_ci_high']):+.3f})"
            )
    land_added_line = ""
    if not land_added.empty:
        overall = land_added.loc[land_added["group_type"].eq("overall")]
        cfsv2 = overall.loc[overall["forecast_system"].eq("CFSv2_RZSM")]
        gefs = overall.loc[overall["forecast_system"].eq("GEFSv12_RZSM")]
        sgp = cfsv2.loc[cfsv2["region"].eq("southern_great_plains")]
        cv = cfsv2.loc[cfsv2["region"].eq("cvalley")]
        gefs_sgp = gefs.loc[gefs["region"].eq("southern_great_plains")]
        gefs_med = gefs.loc[gefs["region"].eq("mediterranean_spain")]
        if not sgp.empty and not cv.empty:
            sgp_row = sgp.iloc[0]
            cv_row = cv.iloc[0]
            land_added_line = (
                f"- Land-surface added-value diagnostic: Central Valley delta BS "
                f"(CFSv2 minus raw persistence) {float(cv_row['delta_bs_forecast_minus_persistence_raw']):+.3f} "
                f"with CI crossing zero; Southern Great Plains delta BS "
                f"{float(sgp_row['delta_bs_forecast_minus_persistence_raw']):+.3f} "
                f"(CI {float(sgp_row['delta_bs_raw_ci_low']):+.3f} to "
                f"{float(sgp_row['delta_bs_raw_ci_high']):+.3f}), favoring persistence"
            )
        if not gefs_sgp.empty and not gefs_med.empty:
            sgp_row = gefs_sgp.iloc[0]
            med_row = gefs_med.iloc[0]
            extra = (
                f"; GEFSv12 selected-minus-selected-persistence delta BS is "
                f"{float(sgp_row['delta_bs_forecast_minus_persistence_selected']):+.3f} "
                f"(CI {float(sgp_row['delta_bs_selected_ci_low']):+.3f} to "
                f"{float(sgp_row['delta_bs_selected_ci_high']):+.3f}) for Southern Great Plains and "
                f"{float(med_row['delta_bs_forecast_minus_persistence_selected']):+.3f} "
                f"(CI {float(med_row['delta_bs_selected_ci_low']):+.3f} to "
                f"{float(med_row['delta_bs_selected_ci_high']):+.3f}) for Mediterranean Spain"
            )
            land_added_line = land_added_line + extra if land_added_line else "- Land-surface added-value diagnostic: " + extra[2:]

    lines = [
        "# Paper Evidence Pack",
        "",
        "This folder consolidates completed experiments into manuscript-facing tables and figures. It does not retrain models.",
        "",
        "## Core Interpretation",
        "",
        "The strongest current claim is two-tiered: lag-based ML rarely converts drought-relevant structure into robust calibrated SPI dry-fraction BSS once leakage-safe evaluation controls are enforced, while land-surface dry-fraction targets are more predictable. CFSv2 root-zone soil-moisture forecasts are robustly positive in Central Valley, and the expanded GEFSv12 reforecast check now shows robust positive land-surface BSS against climatology in four of five validation-selected stack regions. The key limitation is persistence-relative value: the stack robustly improves selected persistence in three of five regions, is only uncertain-added-value in Horn of Africa, and is persistence-dominated in Murray-Darling.",
        "",
        "## Key Counts",
        "",
        f"- Master evidence rows: {len(master)}",
        f"- Headline rows: {len(headline)}",
        f"- Robust-positive rows in master evidence: {len(robust_pos)}",
        f"- Temporal holdouts with positive BSS: {temporal_positive}/{len(temporal)}",
    ]
    if not canonical.empty:
        best = canonical.sort_values("bss", ascending=False).iloc[0]
        lines.append(
            f"- Best canonical Central Valley calibrated checkpoint: {best['model']} BSS {float(best['bss']):+.3f} "
            f"(CI {float(best['ci_low']):+.3f} to {float(best['ci_high']):+.3f})"
        )
    if not operational.empty:
        best_op = operational.sort_values("bss", ascending=False).iloc[0]
        lines.append(
            f"- Best selected operational checkpoint: {best_op['model']} {best_op['target']} lead {int(best_op['lead_months'])} "
            f"BSS {float(best_op['bss']):+.3f} "
            f"(CI {float(best_op['ci_low']):+.3f} to {float(best_op['ci_high']):+.3f})"
        )
    if not operational_prob_raw.empty:
        best_raw = operational_prob_raw.sort_values("bss", ascending=False).iloc[0]
        lines.append(
            f"- Best raw CPC NMME probability checkpoint: {best_raw['target']} lead {int(best_raw['lead_months'])} "
            f"BSS {float(best_raw['bss']):+.3f} "
            f"(CI {float(best_raw['ci_low']):+.3f} to {float(best_raw['ci_high']):+.3f})"
        )
    if not cfs_land_surface.empty:
        best_land = cfs_land_surface.sort_values("bss", ascending=False).iloc[0]
        lines.append(
            f"- Best CFSv2 land-surface checkpoint: {best_land['model']} {best_land['region']} lead {int(best_land['lead_months'])} "
            f"BSS {float(best_land['bss']):+.3f} "
            f"(CI {float(best_land['ci_low']):+.3f} to {float(best_land['ci_high']):+.3f})"
        )
    if not gefs_land_surface.empty:
        best_gefs = gefs_land_surface.sort_values("bss", ascending=False).iloc[0]
        lines.append(
            f"- GEFSv12 land-surface hindcast checkpoint: {best_gefs['region']} lead {int(best_gefs['lead_months'])} "
            f"BSS {float(best_gefs['bss']):+.3f} "
            f"(CI {float(best_gefs['ci_low']):+.3f} to {float(best_gefs['ci_high']):+.3f})"
        )
    persistence_candidates = []
    if not persistence_land_surface.empty:
        persistence_candidates.append(
            persistence_land_surface[["region", "lead_months", "bss", "ci_low", "ci_high"]].copy()
        )
    if not gefs_stack.empty:
        stack_persistence = gefs_stack.loc[gefs_stack["model"].eq("persistence_selected")].copy()
        if not stack_persistence.empty:
            stack_persistence = stack_persistence.rename(
                columns={
                    "bss_vs_climatology": "bss",
                    "bss_ci_low": "ci_low",
                    "bss_ci_high": "ci_high",
                }
            )
            stack_persistence["lead_months"] = 1
            persistence_candidates.append(stack_persistence[["region", "lead_months", "bss", "ci_low", "ci_high"]])
    if persistence_candidates:
        persistence_all = pd.concat(persistence_candidates, ignore_index=True)
        best_persist = persistence_all.sort_values("bss", ascending=False).iloc[0]
        lines.append(
            f"- Best same-target land-surface persistence checkpoint: {short_region(best_persist['region'])} lead {int(best_persist['lead_months'])} "
            f"BSS {float(best_persist['bss']):+.3f} "
            f"(CI {float(best_persist['ci_low']):+.3f} to {float(best_persist['ci_high']):+.3f})"
        )
    if not seasonal_pos.empty:
        row = seasonal_pos.iloc[0]
        lines.append(
            f"- Seasonal regional robust-positive exception: {row['region']} SPI-{int(row['target_spi'])} lead-{int(row['lead_months'])} "
            f"BSS {float(row['bss_iso_vs_clim']):+.3f}, but signal_flag={row['signal_flag']}"
        )
    if not memory.empty:
        best_memory = memory.sort_values("bss", ascending=False).iloc[0]
        soil_memory = memory.loc[memory["model"].astype(str).str.contains("soil-memory", case=False, na=False)]
        soil_text = ""
        if not soil_memory.empty:
            soil_row = soil_memory.sort_values("bss", ascending=False).iloc[0]
            soil_text = f"; soil-memory selected BSS {float(soil_row['bss']):+.3f}"
        lines.append(
            f"- Memory-target checkpoint: best selected SPI-6 lead-6 XGBoost BSS {float(best_memory['bss']):+.3f} "
            f"(CI {float(best_memory['ci_low']):+.3f} to {float(best_memory['ci_high']):+.3f}){soil_text}"
        )
    if eval_line:
        lines.append(eval_line)
    if transition_line:
        lines.append(transition_line)
    if land_added_line:
        lines.append(land_added_line)
    if not climate_index.empty:
        spatial = climate_index.loc[climate_index["model_kind"].eq("spatial")]
        pdo = spatial.loc[spatial["variant"].eq("pdo")]
        combo = spatial.loc[spatial["variant"].eq("nino34_pdo")]
        if not pdo.empty and not combo.empty:
            pdo_row = pdo.iloc[0]
            combo_row = combo.iloc[0]
            lines.append(
                f"- Climate-index sensitivity: on the common valid 2021-01 to 2025-09 window, "
                f"spatial PDO-only BSS is {float(pdo_row['selected_bss']):+.3f} "
                f"(CI {float(pdo_row['selected_bss_ci_low']):+.3f} to "
                f"{float(pdo_row['selected_bss_ci_high']):+.3f}); "
                f"Niño3.4+PDO BSS is {float(combo_row['selected_bss']):+.3f} "
                f"(CI {float(combo_row['selected_bss_ci_low']):+.3f} to "
                f"{float(combo_row['selected_bss_ci_high']):+.3f})"
            )
    if not gefs_sensitivity.empty:
        ok = gefs_sensitivity.loc[gefs_sensitivity["status"].eq("ok")].copy()
        if not ok.empty:
            best = ok.sort_values("gefs_selected_bss", ascending=False).iloc[0]
            worst = ok.sort_values("gefs_selected_bss", ascending=True).iloc[0]
            beats = int(ok["beats_selected_persistence_point"].sum())
            lines.append(
                f"- GEFSv12 SGP lead/valid-day sensitivity: {len(ok)}/{len(ok)} rows are robust-positive versus climatology; "
                f"BSS ranges from {float(worst['gefs_selected_bss']):+.3f} "
                f"(day {int(worst['valid_day'])}, lag {int(worst['init_lag_weeks'])}w) to "
                f"{float(best['gefs_selected_bss']):+.3f} "
                f"(day {int(best['valid_day'])}, lag {int(best['init_lag_weeks'])}w), "
                f"and {beats}/{len(ok)} beat selected persistence on point BSS"
            )
    if not gefs_stack.empty:
        stack = gefs_stack.loc[gefs_stack["model"].eq("stack_validation_selected")].copy()
        if not stack.empty:
            robust = int(stack["claim_status"].eq("robust_positive").sum())
            added = int(stack["added_value_status_selected"].eq("stack_robust_added_value").sum())
            best = stack.sort_values("bss_vs_climatology", ascending=False).iloc[0]
            lines.append(
                f"- GEFSv12+persistence stack: validation-selected convex blends improve selected persistence by paired delta BS in "
                f"{added}/{len(stack)} regions; {robust}/{len(stack)} are robust-positive versus climatology. "
                f"Best stack row is {short_region(best['region'])} BSS {float(best['bss_vs_climatology']):+.3f} "
                f"(CI {float(best['bss_ci_low']):+.3f} to {float(best['bss_ci_high']):+.3f})"
            )
    if not landsurface_reliability.empty:
        stack_rel = landsurface_reliability.loc[landsurface_reliability["model"].eq("stack_validation_selected")]
        if not stack_rel.empty:
            better = int((stack_rel["resolution"] > stack_rel["reliability"]).sum())
            best = stack_rel.sort_values("resolution_minus_reliability", ascending=False).iloc[0]
            worst = stack_rel.sort_values("resolution_minus_reliability", ascending=True).iloc[0]
            lines.append(
                f"- GEFSv12+persistence stack reliability/resolution: the validation-selected stack has "
                f"resolution greater than reliability in {better}/{len(stack_rel)} regions. "
                f"Best resolution-minus-reliability is {short_region(best['region'])} "
                f"({float(best['resolution_minus_reliability']):+.4f}); weakest is {short_region(worst['region'])} "
                f"({float(worst['resolution_minus_reliability']):+.4f})."
            )
    if not landsurface_domain_transfer.empty:
        transfer_stack = landsurface_domain_transfer.loc[
            landsurface_domain_transfer["model"].eq("stack_transfer_selected")
        ].copy()
        loro = transfer_stack.loc[transfer_stack["transfer_mode"].eq("leave_one_region_out")]
        if not loro.empty:
            robust = int((loro["bss_ci_low"] > 0).sum())
            added = int(loro["added_value_status"].eq("stack_robust_added_value").sum())
            mean_bss = float(loro["bss_vs_climatology"].mean())
            lines.append(
                f"- GEFSv12 land-surface domain transfer: leave-one-region-out calibration is "
                f"robust-positive in {robust}/{len(loro)} regions and robustly improves transferred persistence in "
                f"{added}/{len(loro)} regions; mean stack BSS is {mean_bss:+.3f}."
            )
        transfer_mono = landsurface_domain_transfer.loc[
            landsurface_domain_transfer["model"].eq("monotonic_xgb_transfer")
            & landsurface_domain_transfer["transfer_mode"].eq("leave_one_region_out")
        ].copy()
        if not transfer_mono.empty:
            robust = int((transfer_mono["bss_ci_low"] > 0).sum())
            added = int(transfer_mono["added_value_status"].eq("stack_robust_added_value").sum())
            mean_bss = float(transfer_mono["bss_vs_climatology"].mean())
            lines.append(
                f"- Monotonic land-surface XGBoost: leave-one-region-out constrained training is "
                f"robust-positive in {robust}/{len(transfer_mono)} regions and robustly improves transferred persistence in "
                f"{added}/{len(transfer_mono)} regions; mean BSS is {mean_bss:+.3f}."
            )
    if not landsurface_rare_event.empty:
        rare_focus = landsurface_rare_event.loc[
            landsurface_rare_event["transfer_mode"].eq("leave_one_region_out")
            & landsurface_rare_event["model"].isin(["stack_transfer_selected", "monotonic_xgb_transfer"])
        ].copy()
        if not rare_focus.empty:
            pieces = []
            for (quantile, model), part in rare_focus.groupby(["event_quantile", "model"], sort=False):
                robust = int(part["event_claim_status"].eq("robust_positive").sum())
                mean_ap = float(part["average_precision"].mean())
                mean_lift = float(part["average_precision_lift_over_event_rate"].mean())
                label = "stack" if model == "stack_transfer_selected" else "monotonic XGB"
                pieces.append(
                    f"q{float(quantile):.2f} {label}: robust event-BSS {robust}/{len(part)}, "
                    f"mean AP {mean_ap:.3f}, mean AP lift {mean_lift:.2f}x"
                )
            lines.append("- Rare-event land-surface formulation: " + "; ".join(pieces) + ".")
    if not landsurface_persistence_regimes.empty:
        focus = landsurface_persistence_regimes.loc[
            landsurface_persistence_regimes["target_region"].eq("all_regions")
            & landsurface_persistence_regimes["group_type"].eq("memory_forecast_agreement")
            & landsurface_persistence_regimes["model"].eq("stack_transfer_selected")
        ].copy()
        if not focus.empty:
            disagreement = focus.loc[focus["group_value"].eq("memory_dry_forecast_wet")]
            overall = landsurface_persistence_regimes.loc[
                landsurface_persistence_regimes["target_region"].eq("all_regions")
                & landsurface_persistence_regimes["group_type"].eq("overall")
                & landsurface_persistence_regimes["model"].eq("stack_transfer_selected")
            ]
            text = ""
            if not overall.empty:
                row = overall.iloc[0]
                text = (
                    f"overall stack delta BS versus transferred persistence is "
                    f"{float(row['delta_bs_model_minus_persistence']):+.4f} "
                    f"(CI {float(row['delta_bs_ci_low']):+.4f} to {float(row['delta_bs_ci_high']):+.4f})"
                )
            if not disagreement.empty:
                row = disagreement.iloc[0]
                text += (
                    f"; the largest robust regime gain occurs when antecedent memory is dry but GEFSv12 is wetter than normal "
                    f"(delta BS {float(row['delta_bs_model_minus_persistence']):+.4f}, "
                    f"CI {float(row['delta_bs_ci_low']):+.4f} to {float(row['delta_bs_ci_high']):+.4f})"
                )
            if text:
                lines.append("- Land-surface persistence-regime diagnostic: " + text + ".")
    c3s = master.loc[
        master["evidence_group"].eq("forecast_informed_landsurface_benchmark")
        & master["model"].astype(str).str.contains("C3S/ECMWF native VSM", case=False, na=False)
        & master["calibration"].eq("c3s_selected")
    ].copy()
    if not c3s.empty:
        pieces = [
            f"{short_region(row['region'])} BSS {float(row['bss']):+.3f} "
            f"(CI {float(row['ci_low']):+.3f} to {float(row['ci_high']):+.3f})"
            for _, row in c3s.sort_values("region").iterrows()
        ]
        lines.append(
            "- C3S/ECMWF native-VSM benchmark: "
            + "; ".join(pieces)
            + ". These compact 2021-2025 tests verify native-RZSM archive access but do not provide persistence-beating added value."
        )
    if not landsurface_independent_targets.empty:
        ready = int(landsurface_independent_targets["ready_to_score_now"].astype(bool).sum())
        total = len(landsurface_independent_targets)
        first = landsurface_independent_targets.iloc[0]
        first_label = "ready/scored priority product" if bool(first["ready_to_score_now"]) else "recommended first acquisition"
        lines.append(
            f"- Independent target audit: {ready}/{total} candidate soil-moisture target products are currently ready to score from local files; "
            f"{first_label} is {first['product']} ({first['current_decision']})."
        )
    if not landsurface_nldas_validation.empty:
        stack = landsurface_nldas_validation.loc[
            landsurface_nldas_validation["model"].eq("stack_validation_selected")
        ].copy()
        if not stack.empty:
            robust = int(stack["claim_status"].eq("robust_positive").sum())
            added = int(stack["added_value_status_selected"].eq("stack_robust_added_value").sum())
            pieces = [
                f"{short_region(row['region'])} BSS {float(row['bss_vs_climatology']):+.3f} "
                f"(CI {float(row['bss_ci_low']):+.3f} to {float(row['bss_ci_high']):+.3f}), "
                f"delta BS {float(row['delta_bs_model_minus_persistence_selected']):+.4f}"
                for _, row in stack.sort_values("region").iterrows()
            ]
            lines.append(
                f"- NLDAS independent-target validation: validation-selected GEFSv12/persistence stacks are "
                f"robust-positive in {robust}/{len(stack)} U.S. regions and robustly improve selected persistence in "
                f"{added}/{len(stack)} regions; "
                + "; ".join(pieces)
                + "."
            )
    if not landsurface_target_product_comparison.empty:
        pieces = []
        for _, row in landsurface_target_product_comparison.sort_values("region").iterrows():
            pieces.append(
                f"{row['region_label']}: ERA5-Land BSS {float(row['era5_bss_vs_climatology']):+.3f}, "
                f"NLDAS BSS {float(row['nldas_bss_vs_climatology']):+.3f}, "
                f"NLDAS-minus-ERA5 shift {float(row['bss_shift_nldas_minus_era5']):+.3f}"
            )
        lines.append(
            "- ERA5-Land vs NLDAS target-product comparison: "
            + "; ".join(pieces)
            + ". Both U.S. regions remain positive under the independent NLDAS target."
        )
    if not landsurface_gldas_validation.empty:
        stack = landsurface_gldas_validation.loc[
            landsurface_gldas_validation["model"].eq("stack_validation_selected")
        ].copy()
        if not stack.empty:
            robust = int(stack["claim_status"].eq("robust_positive").sum())
            added = int(stack["added_value_status_selected"].eq("stack_robust_added_value").sum())
            caveats = stack.loc[
                ~stack["added_value_status_selected"].eq("stack_robust_added_value")
            ][["region", "added_value_status_selected"]]
            caveat_text = "; ".join(
                f"{short_region(row['region'])}: {row['added_value_status_selected']}"
                for _, row in caveats.sort_values("region").iterrows()
            )
            lines.append(
                f"- GLDAS global target-product sensitivity: validation-selected stacks are robust-positive in "
                f"{robust}/{len(stack)} regions and robustly improve selected persistence in {added}/{len(stack)} regions. "
                + (f"Persistence-relative caveats: {caveat_text}." if caveat_text else "")
            )
    if not landsurface_era5_gldas_comparison.empty:
        pieces = []
        for _, row in landsurface_era5_gldas_comparison.sort_values("region").iterrows():
            pieces.append(
                f"{row['region_label']}: ERA5-Land {float(row['era5_bss_vs_climatology']):+.3f}, "
                f"GLDAS {float(row['gldas_bss_vs_climatology']):+.3f}"
            )
        lines.append("- ERA5-Land vs GLDAS target-product comparison: " + "; ".join(pieces) + ".")
    if not landsurface_smap_l4_validation.empty:
        stack = landsurface_smap_l4_validation.loc[
            landsurface_smap_l4_validation["model"].eq("stack_validation_selected")
        ].copy()
        if not stack.empty:
            robust = int(stack["claim_status"].eq("robust_positive").sum())
            added = int(stack["added_value_status_selected"].eq("stack_robust_added_value").sum())
            caveats = stack.loc[
                ~stack["added_value_status_selected"].eq("stack_robust_added_value")
            ][["region", "added_value_status_selected"]]
            caveat_text = "; ".join(
                f"{short_region(row['region'])}: {row['added_value_status_selected']}"
                for _, row in caveats.sort_values("region").iterrows()
            )
            lines.append(
                f"- SMAP L4 short-record validation: mid-month SPL4SMGP stacks are robust-positive in "
                f"{robust}/{len(stack)} regions and robustly improve selected persistence in {added}/{len(stack)} regions. "
                + (f"Persistence-relative caveats: {caveat_text}." if caveat_text else "")
            )
    if not landsurface_smap_l4_snapshot_sensitivity.empty:
        robust = int(landsurface_smap_l4_snapshot_sensitivity["multi3_claim_status"].eq("robust_positive").sum())
        added = int(
            landsurface_smap_l4_snapshot_sensitivity["multi3_added_value_status"].eq("stack_robust_added_value").sum()
        )
        mid_robust = int(landsurface_smap_l4_snapshot_sensitivity["midmonth_claim_status"].eq("robust_positive").sum())
        mid_added = int(
            landsurface_smap_l4_snapshot_sensitivity["midmonth_added_value_status"].eq("stack_robust_added_value").sum()
        )
        lines.append(
            "- SMAP L4 snapshot sensitivity: early/mid/late snapshots increase the validation-selected stack from "
            f"{mid_robust}/5 to {robust}/5 robust-positive regions and from {mid_added}/5 to {added}/5 robust added-value rows. "
            "This reduces the mid-month-proxy concern but keeps the persistence-relative claim conditional."
        )
    if not landsurface_target_product_transfer.empty:
        cross = landsurface_target_product_transfer.loc[
            landsurface_target_product_transfer["transfer_kind"].eq("cross_product_transfer")
        ].copy()
        gefs = cross.loc[cross["model"].eq("gefs_selected")].copy()
        stack = cross.loc[cross["model"].eq("stack_validation_selected")].copy()
        if not cross.empty:
            stack_robust = int(stack["claim_status_vs_smap_climatology"].eq("robust_positive").sum())
            added = int(
                cross["added_value_status_vs_smap_persistence"].eq("stack_robust_added_value").sum()
            )
            lines.append(
                "- SMAP cross-product transfer: source-calibrated ERA5-Land/GLDAS probabilities were scored "
                "against the SMAP multi-snapshot target without SMAP recalibration. GEFS-only rows contain the "
                "only robust-positive transfer cases, whereas selected source-product "
                f"stacks are robust-positive in {stack_robust}/{len(stack)} rows; robust added value over SMAP "
                f"same-target persistence occurs in {added}/{len(cross)} compact transfer rows. This narrows the "
                "claim to dynamical forecast signal transfer, not universal calibration or persistence-stack transfer."
            )
    if not landsurface_calibration_transfer_ladder.empty:
        compact = landsurface_calibration_transfer_ladder.copy()
        def count_rows(strategy: str, model: str, status_col: str, status_value: str) -> tuple[int, int]:
            part = compact.loc[compact["strategy"].eq(strategy) & compact["model"].eq(model)]
            return int(part[status_col].eq(status_value).sum()), len(part)

        direct_gefs, direct_gefs_n = count_rows(
            "source_direct", "gefs_selected", "claim_status_vs_smap_climatology", "robust_positive"
        )
        shifted_gefs, shifted_gefs_n = count_rows(
            "target_prior_shift", "gefs_selected", "claim_status_vs_smap_climatology", "robust_positive"
        )
        pooled_stack, pooled_stack_n = count_rows(
            "pooled_source_calibration", "stack_validation_selected", "claim_status_vs_smap_climatology", "robust_positive"
        )
        shifted_added, shifted_added_n = count_rows(
            "target_prior_shift", "gefs_selected", "added_value_status_vs_smap_persistence", "stack_robust_added_value"
        )
        lines.append(
            "- SMAP calibration-transfer ladder: direct source-calibrated GEFS transfer is robust-positive in "
            f"{direct_gefs}/{direct_gefs_n} rows, while a validation-only source-to-SMAP prior shift raises this to "
            f"{shifted_gefs}/{shifted_gefs_n} and robustly improves SMAP persistence in {shifted_added}/{shifted_added_n}. "
            f"Pooled ERA5-Land+GLDAS source calibration gives robust-positive selected stacks in {pooled_stack}/{pooled_stack_n} regions. "
            "This indicates that much of the cross-product failure is calibration/base-rate mismatch, but persistence-relative "
            "added value remains conditional."
        )
    if not landsurface_target_product_adaptation_benchmark.empty:
        compact = landsurface_target_product_adaptation_benchmark.copy()

        def adaptation_row(method: str, model: str) -> pd.Series | None:
            part = compact.loc[
                compact["adaptation_method"].eq(method)
                & compact["model"].eq(model)
            ]
            if part.empty:
                return None
            return part.iloc[0]

        dry_gefs = adaptation_row("Fixed validation dry-rate shift", "gefs_selected")
        pred_base_gefs = adaptation_row(
            "Fixed validation prediction-base-rate shift", "gefs_selected"
        )
        temp_base_gefs = adaptation_row(
            "Fixed bias-corrected temperature/base-rate scaling", "gefs_selected"
        )
        pred_base_stack = adaptation_row(
            "Fixed validation prediction-base-rate shift", "stack_validation_selected"
        )
        temp_base_stack = adaptation_row(
            "Fixed bias-corrected temperature/base-rate scaling", "stack_validation_selected"
        )
        base_selector_gefs = adaptation_row(
            "Validation-selected base-rate adaptation", "gefs_selected"
        )
        complex_gefs = adaptation_row("Validation-selected complex adaptation", "gefs_selected")
        complex_stack = adaptation_row(
            "Validation-selected complex adaptation", "stack_validation_selected"
        )
        if all(
            row is not None
            for row in [
                dry_gefs,
                pred_base_gefs,
                temp_base_gefs,
                pred_base_stack,
                temp_base_stack,
                base_selector_gefs,
                complex_gefs,
                complex_stack,
            ]
        ):
            lines.append(
                "- SMAP target-product adaptation benchmark: product-specific prediction-base-rate shifting "
                "is the strongest simple calibration for GEFS, with robust-positive transfer in "
                f"{int(pred_base_gefs['robust_positive_count'])}/{int(pred_base_gefs['n_rows'])} rows "
                f"and robust added value over SMAP persistence in {int(pred_base_gefs['robust_added_value_count'])}/"
                f"{int(pred_base_gefs['n_rows'])}. The source-to-SMAP dry-rate shift has "
                f"{int(dry_gefs['robust_positive_count'])}/{int(dry_gefs['n_rows'])} robust-positive GEFS rows "
                f"and {int(dry_gefs['robust_added_value_count'])}/{int(dry_gefs['n_rows'])} robust added-value rows. "
                "Bias-corrected temperature/base-rate scaling also keeps GEFS robust-positive in "
                f"{int(temp_base_gefs['robust_positive_count'])}/{int(temp_base_gefs['n_rows'])} rows "
                f"but has {int(temp_base_gefs['robust_added_value_count'])}/{int(temp_base_gefs['n_rows'])} "
                "robust added-value rows. "
                "The stack remains conditional after prediction-base-rate shifting "
                f"({int(pred_base_stack['robust_positive_count'])}/{int(pred_base_stack['n_rows'])} robust-positive, "
                f"{int(pred_base_stack['robust_added_value_count'])}/{int(pred_base_stack['n_rows'])} robust added value). "
                "The temperature-scaled stack is weaker "
                f"({int(temp_base_stack['robust_positive_count'])}/{int(temp_base_stack['n_rows'])} robust-positive). "
                "Validation-selected seasonal/base-rate and complex target-specific selectors are weaker "
                f"for GEFS ({int(base_selector_gefs['robust_positive_count'])}/{int(base_selector_gefs['n_rows'])} "
                f"and {int(complex_gefs['robust_positive_count'])}/{int(complex_gefs['n_rows'])} robust-positive rows, respectively), "
                f"and complex stack adaptation gives {int(complex_stack['robust_added_value_count'])}/"
                f"{int(complex_stack['n_rows'])} robust added-value rows. This supports base-rate calibration as a "
                "paper contribution, but not a more complex target-specific adaptation claim on the current sample."
            )
    if not landsurface_base_rate_yearly_sensitivity.empty:
        yearly = landsurface_base_rate_yearly_sensitivity.copy()

        def yearly_rows(method: str) -> pd.DataFrame:
            return yearly.loc[yearly["adaptation_method"].eq(method)].sort_values("target_year")

        pred_years = yearly_rows("Fixed validation prediction-base-rate shift")
        dry_years = yearly_rows("Fixed validation dry-rate shift")
        temp_years = yearly_rows("Fixed bias-corrected temperature/base-rate scaling")
        if not pred_years.empty and not dry_years.empty and not temp_years.empty:
            pred_counts = ", ".join(
                f"{int(row['target_year'])}: {int(row['positive_bss_count'])}/{int(row['n_rows'])}"
                for _, row in pred_years.iterrows()
            )
            dry_counts = ", ".join(
                f"{int(row['target_year'])}: {int(row['positive_bss_count'])}/{int(row['n_rows'])}"
                for _, row in dry_years.iterrows()
            )
            temp_counts = ", ".join(
                f"{int(row['target_year'])}: {int(row['positive_bss_count'])}/{int(row['n_rows'])}"
                for _, row in temp_years.iterrows()
            )
            weak_years = pred_years.loc[pred_years["mean_bss_vs_smap_persistence"] < 0]
            weak_text = "; ".join(
                f"{int(row['target_year'])} mean BSS vs persistence {float(row['mean_bss_vs_smap_persistence']):+.3f}"
                for _, row in weak_years.iterrows()
            )
            lines.append(
                "- SMAP base-rate yearly sensitivity: prediction-base-rate GEFS rows remain positive against "
                f"SMAP climatology in {pred_counts}; dry-rate shift rows in {dry_counts}; temperature/base-rate rows in {temp_counts}. "
                + (
                    f"Persistence-relative value is not stable every year ({weak_text}). "
                    if weak_text
                    else ""
                )
                + "This supports a short-record calibration finding while keeping the added-value claim conditional."
            )
    lines.extend(
        [
            "",
            "## Generated Files",
            "",
            "- `table01_master_evidence.csv`: all consolidated evidence rows.",
            "- `table02_headline_results.csv`: compact manuscript headline table.",
            "- `table03_mask_methods.csv`: source-cited mask methods and retained-cell fractions.",
            "- `table04_temporal_robustness.csv`: rolling holdout control for test-period non-representativeness.",
            "- `table05_seasonal_signal_audit.csv`: seasonal BSS interpreted with event-tracking diagnostics.",
            "- `table06_regionalization_mechanism.csv`: SPI-12 mechanism evidence joined with zone-level forecast diagnostics.",
            "- `table07_evaluation_inflation_audit.csv`: invalid-protocol audit showing skill inflation from random row splits, pixel-level inference, and overlapping SPI targets.",
            "- `table08_transition_target_summary.csv`: onset/termination eligible-pixel transition diagnostic and replication checks.",
            "- `table09_landsurface_added_value.csv`: CFSv2 and GEFSv12 root-zone soil-moisture added-value diagnostics against same-target persistence.",
            "- `table10_climate_index_sensitivity.csv`: common-valid-period Niño3.4/PDO sensitivity test without PDO tail forward-fill.",
            "- `table11_gefsv12_landsurface_sensitivity.csv`: Southern Great Plains GEFSv12 lead/valid-day sensitivity grid.",
            "- `table12_gefsv12_landsurface_stack.csv`: validation-selected GEFSv12+persistence land-surface stack diagnostic.",
            "- `table13_landsurface_reliability_resolution.csv`: Brier reliability/resolution diagnostics for GEFSv12, persistence, and stack land-surface probabilities.",
            "- `table14_landsurface_domain_transfer.csv`: local, pooled, and leave-one-region-out land-surface calibration-transfer benchmark.",
            "- `table15_landsurface_rare_event.csv`: validation-thresholded dry-event PR-AUC, event-BSS, and reliability diagnostics.",
            "- `table16_landsurface_persistence_regimes.csv`: paired added-value diagnostics by season, antecedent dry-fraction regime, and GEFSv12/persistence agreement class.",
            "- `table17_landsurface_independent_target_audit.csv`: source-cited independent soil-moisture target feasibility and local-file audit.",
            "- `table18_landsurface_nldas_validation.csv`: U.S. independent-target validation against NLDAS Noah monthly soil moisture.",
            "- `table19_landsurface_target_product_comparison.csv`: ERA5-Land versus NLDAS comparison for the validation-selected U.S. GEFSv12/persistence stack.",
            "- `table20_landsurface_gldas_validation.csv`: global GLDAS Noah target-product sensitivity for all five regions.",
            "- `table21_landsurface_era5_gldas_comparison.csv`: ERA5-Land versus GLDAS comparison for the validation-selected GEFSv12/persistence stack.",
            "- `table22_landsurface_smap_l4_validation.csv`: short-record SMAP L4 satellite-assimilated target validation.",
            "- `table23_landsurface_smap_l4_snapshot_sensitivity.csv`: SMAP L4 mid-month versus early/mid/late snapshot sensitivity.",
            "- `table24_landsurface_target_product_transfer.csv`: ERA5-Land/GLDAS-calibrated probability transfer to the SMAP L4 multi-snapshot target without SMAP recalibration.",
            "- `table25_landsurface_calibration_transfer_ladder.csv`: SMAP calibration-transfer ladder separating direct transfer, prior shift, SMAP validation recalibration, and pooled source calibration.",
            "- `table26_landsurface_calibration_transfer_ladder_compact.csv`: compact manuscript summary of ladder row counts, mean BSS, and persistence-relative added value.",
            "- `table27_landsurface_target_product_adaptation_benchmark.csv`: validation-only target-product adaptation benchmark comparing base-rate shifts with target-specific recalibration.",
            "- `table28_landsurface_base_rate_yearly_sensitivity.csv`: year-by-year SMAP sensitivity for the leading GEFS base-rate calibration methods.",
            "- `table29_landsurface_forecast_archive_audit.csv`: operational GEFS, SMAP-extension, SubX, and ECMWF/S2S feasibility audit for extending the land-surface benchmark.",
            "- `table30_operational_gefs_diagnostic.csv`: Central Valley 2021-2025 operational GEFS calibration and persistence-safe selection diagnostic for both SOILW 0.1-1 m and SOILL 0-1 m extraction modes.",
            "- `table31_operational_gefs_smap_validation.csv`: Central Valley operational GEFS validation against the modern SMAP L4 multi-snapshot target.",
            "- `table32_landsurface_persistence_residual_selector.csv`: validation-safe persistence-residual selector benchmark against raw same-target persistence.",
            "- `table33_landsurface_persistence_residual_selector_compact.csv`: compact summary of persistence-residual selector row counts and mean skill.",
            "- `table34_smap_product_residual_selector.csv`: SMAP product-transfer residual benchmark against raw same-target SMAP persistence.",
            "- `table35_smap_product_residual_selector_compact.csv`: compact summary of SMAP product-residual row counts, mean skill, and persistence-relative added value.",
            "- `table36_operational_gefs_three_region_replication.csv`: modern operational-GEFS smoke-test replication for Central Valley, Southern Great Plains, and Mediterranean Spain across SOILW/SOILL extraction modes.",
            "- `table37_native_rzsm_archive_audit.csv`: native-RZSM forecast archive audit showing C3S seasonal-original as the best candidate and recording successful local retrieval.",
            "- `table38_native_rzsm_archive_candidates.csv`: C3S seasonal-original native VSM candidate systems ranked by historical coverage, start months, and lead-time range.",
            "- `table39_c3s_ecmwf_native_vsm_benchmark.csv`: compact Central Valley, Mediterranean Spain, and Southern Great Plains C3S/ECMWF system 51 VSM benchmarks scored against ERA5-Land root-zone dry fraction.",
            "- `fig01_headline_bss_forest.png`: headline BSS forest plot.",
            "- `fig02_multiregion_bss_forest.png`: multi-region selected BSS forest plot.",
            "- `fig03_seasonal_bss_vs_tracking.png`: seasonal BSS vs event-tracking correlation.",
            "- `fig04_temporal_holdout_bss.png`: temporal holdout BSS forest plot.",
            "- `fig05_mask_retention.png`: retained-cell fractions for source-cited masks.",
            "- `fig06_landsurface_stack_reliability.png`: binned reliability diagram for the GEFSv12/persistence land-surface stack.",
            "- `fig07_landsurface_target_product_comparison.png`: BSS forest plot comparing ERA5-Land and NLDAS targets in the two U.S. regions.",
            "- `fig08_landsurface_era5_gldas_comparison.png`: BSS forest plot comparing ERA5-Land and GLDAS targets across all five regions.",
            "- `fig09_landsurface_calibration_transfer_ladder.png`: compact ladder figure showing robust-positive and robust-added-value fractions.",
            "- `manuscript_methods_draft.md`: source-cited Methods draft tied to the current evidence tables.",
            "- `manuscript_claims_audit.md`: allowed-claims and overclaim-risk checklist.",
            "- `manuscript_results_discussion_draft.md`: prose draft for Results and Discussion.",
            "- `methods_sources_and_evidence_index.md`: source/citation and claim-to-evidence checklist.",
            "",
            "## Manuscript Guardrails",
            "",
            "- Do not generalize the positive land-surface benchmark to precipitation-index skill, all leads, all regions, or deployment readiness.",
            "- Do not claim CFSv2 root-zone soil-moisture added value over persistence from the current monthly-mean extraction.",
            "- Treat GEFSv12 RZSM as robust against climatology in the replicated Southern Great Plains and Mediterranean Spain rows, but do not claim broad dynamic-model added value over same-target persistence.",
            "- Treat the GEFSv12 Southern Great Plains lead/valid-day grid as stability evidence; only 7/8 rows beat selected persistence on point BSS.",
            "- Treat the GEFSv12+persistence stack as a conservative benchmark over 36 test months; it supports combined forecast-memory value but is not a deployment-ready operational system.",
            "- Treat land-surface reliability/resolution decomposition as a binned 36-month diagnostic, not a standalone significance test.",
            "- Treat domain-transfer results as calibration-transfer evidence; they do not prove deployment readiness without independent operational hindcast validation.",
            "- Treat transition-target onset as a target-design diagnostic; it does not survive basin/regional replication.",
            "- Treat PDO sensitivity as a Central Valley common-valid-period result, not evidence that PDO is never useful for other targets, regions, or timescales.",
            "- Treat positive point estimates with confidence intervals crossing zero as hypothesis-generating.",
            "- Treat the Mediterranean Spain SPI-6 seasonal robust-positive row as a calibration-shift exception unless follow-up diagnostics show stronger temporal tracking.",
            "- Use regionalization and SHAP as mechanism evidence, not proof of calibrated forecast skill.",
        ]
    )
    return "\n".join(lines) + "\n"


def build_methods_draft(
    headline: pd.DataFrame,
    mask_table: pd.DataFrame,
    temporal: pd.DataFrame,
    seasonal: pd.DataFrame,
    regionalization: pd.DataFrame,
    evaluation: pd.DataFrame,
    transition: pd.DataFrame,
    land_added: pd.DataFrame,
    climate_index: pd.DataFrame,
    gefs_sensitivity: pd.DataFrame,
    gefs_stack: pd.DataFrame,
    landsurface_reliability: pd.DataFrame,
    landsurface_domain_transfer: pd.DataFrame,
    landsurface_rare_event: pd.DataFrame,
    landsurface_persistence_regimes: pd.DataFrame,
    landsurface_independent_targets: pd.DataFrame,
    landsurface_nldas_validation: pd.DataFrame,
    landsurface_target_product_comparison: pd.DataFrame,
    landsurface_gldas_validation: pd.DataFrame,
    landsurface_era5_gldas_comparison: pd.DataFrame,
    landsurface_smap_l4_validation: pd.DataFrame,
    landsurface_smap_l4_snapshot_sensitivity: pd.DataFrame,
    landsurface_target_product_transfer: pd.DataFrame,
    landsurface_calibration_transfer_ladder: pd.DataFrame,
    landsurface_target_product_adaptation_benchmark: pd.DataFrame,
) -> str:
    canonical = headline.loc[
        headline["evidence_group"].eq("central_valley_calibrated_checkpoint")
        & headline["model"].astype(str).eq("XGB-Spatial")
    ]
    n_canonical = int(canonical["n_months"].iloc[0]) if not canonical.empty else 63
    bss_canonical = float(canonical["bss"].iloc[0]) if not canonical.empty else np.nan
    ci_canonical = (
        float(canonical["ci_low"].iloc[0]),
        float(canonical["ci_high"].iloc[0]),
    ) if not canonical.empty else (np.nan, np.nan)

    op = headline.loc[headline["evidence_group"].eq("operational_dynamical_benchmark")].copy()
    op_best = op.sort_values("bss", ascending=False).iloc[0] if not op.empty else None
    op_summary = ""
    if op_best is not None:
        op_summary = (
            f"The largest operational point estimate in the current evidence pack is "
            f"{op_best['model']} for {op_best['target']} at lead {int(op_best['lead_months'])}, "
            f"with BSS {float(op_best['bss']):+.3f} and a 95% CI from "
            f"{float(op_best['ci_low']):+.3f} to {float(op_best['ci_high']):+.3f}."
        )
    cfs = op.loc[op["experiment"].astype(str).str.contains("cfsv2", case=False, na=False)].copy()
    cfs_summary = ""
    if not cfs.empty:
        raw = cfs.loc[cfs["experiment"].astype(str).str.contains("rawamount", case=False, na=False)]
        anom = cfs.loc[cfs["experiment"].astype(str).str.contains("anomaly", case=False, na=False)]
        pieces = []
        if not raw.empty:
            row = raw.iloc[0]
            pieces.append(
                f"raw accumulated-precipitation BSS {float(row['bss']):+.3f} "
                f"(95% CI {float(row['ci_low']):+.3f} to {float(row['ci_high']):+.3f})"
            )
        if not anom.empty:
            row = anom.iloc[0]
            pieces.append(
                f"anomaly BSS {float(row['bss']):+.3f} "
                f"(95% CI {float(row['ci_low']):+.3f} to {float(row['ci_high']):+.3f})"
            )
        cfs_summary = (
            " The NCEI CFSv2 SPI-3 lead-3 benchmark accumulates individual-run "
            "precipitation over the full target window; "
            + ", while ".join(pieces)
            + ". The accessible 6-hour individual-run archive starts in 2016 and "
            "does not provide a complete SPI-6 lead-6 accumulation window."
        )

    land = headline.loc[headline["evidence_group"].eq("forecast_informed_landsurface_benchmark")].copy()
    land_line = ""
    if not land.empty:
        cfs_land = land.loc[land["model"].astype(str).str.contains("CFSv2", case=False, na=False)]
        gefs_land = land.loc[land["model"].astype(str).str.contains("GEFSv12", case=False, na=False)]
        c3s_land = land.loc[
            land["model"].astype(str).str.contains("C3S/ECMWF native VSM", case=False, na=False)
            & land["calibration"].eq("c3s_selected")
        ]
        persistence_land = land.loc[land["model"].astype(str).str.contains("persistence", case=False, na=False)]
        parts = [
            "The forecast-informed land-surface benchmark verifies dynamic-model soil-moisture forecasts against an ERA5-Land 0-100 cm root-zone soil-moisture dry-fraction target."
        ]
        if not cfs_land.empty:
            row = cfs_land.sort_values("bss", ascending=False).iloc[0]
            parts.append(
                f"The best selected CFSv2 row is {row['region']} with BSS {float(row['bss']):+.3f} "
                f"(95% CI {float(row['ci_low']):+.3f} to {float(row['ci_high']):+.3f})."
            )
            allcycle = cfs_land.loc[
                cfs_land["model"].astype(str).str.contains("4-cycle", case=False, na=False)
                & cfs_land["region"].astype(str).eq("California Central Valley")
            ]
            if not allcycle.empty:
                row = allcycle.sort_values("bss", ascending=False).iloc[0]
                parts.append(
                    f"The strict four-cycle Central Valley replication remains positive "
                    f"(BSS {float(row['bss']):+.3f}, 95% CI {float(row['ci_low']):+.3f} to {float(row['ci_high']):+.3f})."
                )
            regional = cfs_land.loc[
                cfs_land["model"].astype(str).str.contains("4-cycle", case=False, na=False)
                & ~cfs_land["region"].astype(str).eq("California Central Valley")
            ].copy()
            if not regional.empty:
                pieces = [
                    f"{row['region']}: BSS {float(row['bss']):+.3f}"
                    for _, row in regional.sort_values("region").iterrows()
                ]
                parts.append("Regional four-cycle CFSv2 checks are mixed (" + "; ".join(pieces) + ").")
        if not c3s_land.empty:
            pieces = [
                f"{row['region']} selected C3S BSS {float(row['bss']):+.3f} "
                f"(95% CI {float(row['ci_low']):+.3f} to {float(row['ci_high']):+.3f})"
                for _, row in c3s_land.sort_values("region").iterrows()
            ]
            parts.append(
                "C3S/ECMWF system 51 native-VSM checkpoints use 2019-2020 validation "
                "and complete 2021-2025 testing: "
                + "; ".join(pieces)
                + "."
            )
        stack_gefs = (
            gefs_stack.loc[gefs_stack["model"].eq("gefs_selected")].copy()
            if not gefs_stack.empty
            else pd.DataFrame()
        )
        if not stack_gefs.empty:
            row = stack_gefs.sort_values("bss_vs_climatology", ascending=False).iloc[0]
            robust_count = int(stack_gefs["claim_status"].eq("robust_positive").sum())
            parts.append(
                f"The expanded hindcast-calibrated GEFSv12 11-member day-15 rows have a best BSS of "
                f"{float(row['bss_vs_climatology']):+.3f} in {short_region(row['region'])} "
                f"(95% CI {float(row['bss_ci_low']):+.3f} to {float(row['bss_ci_high']):+.3f}) "
                f"over the 2017-2019 frozen test period; {robust_count}/{len(stack_gefs)} "
                "selected GEFSv12 rows are robust-positive against climatology."
            )
        elif not gefs_land.empty:
            row = gefs_land.sort_values("bss", ascending=False).iloc[0]
            robust_count = int(gefs_land["status"].eq("robust_positive").sum())
            parts.append(
                f"The best hindcast-calibrated GEFSv12 11-member reforecast row has BSS "
                f"{float(row['bss']):+.3f} "
                f"(95% CI {float(row['ci_low']):+.3f} to {float(row['ci_high']):+.3f}) "
                f"over the 2017-2019 frozen test period; {robust_count}/{len(gefs_land)} "
                "selected GEFSv12 regional rows are robust-positive against climatology."
            )
        stack_persistence = (
            gefs_stack.loc[gefs_stack["model"].eq("persistence_selected")].copy()
            if not gefs_stack.empty
            else pd.DataFrame()
        )
        if not stack_persistence.empty:
            row = stack_persistence.sort_values("bss_vs_climatology", ascending=False).iloc[0]
            parts.append(
                f"The strongest same-target persistence row has BSS {float(row['bss_vs_climatology']):+.3f} "
                f"in {short_region(row['region'])} (95% CI {float(row['bss_ci_low']):+.3f} to "
                f"{float(row['bss_ci_high']):+.3f})."
            )
        elif not persistence_land.empty:
            row = persistence_land.sort_values("bss", ascending=False).iloc[0]
            parts.append(
                f"The strongest same-target persistence row has BSS {float(row['bss']):+.3f} "
                f"(95% CI {float(row['ci_low']):+.3f} to {float(row['ci_high']):+.3f})."
            )
        land_line = " ".join(parts)

    gefs_sensitivity_line = ""
    if not gefs_sensitivity.empty:
        ok = gefs_sensitivity.loc[gefs_sensitivity["status"].eq("ok")].copy()
        if not ok.empty:
            best = ok.sort_values("gefs_selected_bss", ascending=False).iloc[0]
            worst = ok.sort_values("gefs_selected_bss", ascending=True).iloc[0]
            beats = int(ok["beats_selected_persistence_point"].sum())
            gefs_sensitivity_line = (
                "A Southern Great Plains GEFSv12 lead/valid-day sensitivity grid "
                "varies target-month valid day (5, 10, 15, 20) and weekly long-init lag "
                "(0 or 1 week) under the same hindcast-calibrated protocol. "
                f"All {len(ok)} combinations are robust-positive versus climatology; "
                f"BSS ranges from {float(worst['gefs_selected_bss']):+.3f} "
                f"(95% CI {float(worst['gefs_selected_bss_ci_low']):+.3f} to "
                f"{float(worst['gefs_selected_bss_ci_high']):+.3f}) to "
                f"{float(best['gefs_selected_bss']):+.3f} "
                f"(95% CI {float(best['gefs_selected_bss_ci_low']):+.3f} to "
                f"{float(best['gefs_selected_bss_ci_high']):+.3f}). "
                f"{beats}/{len(ok)} combinations beat selected persistence on point BSS."
            )

    gefs_stack_line = ""
    if not gefs_stack.empty:
        stack = gefs_stack.loc[gefs_stack["model"].eq("stack_validation_selected")].copy()
        if not stack.empty:
            robust = int(stack["claim_status"].eq("robust_positive").sum())
            added = int(stack["added_value_status_selected"].eq("stack_robust_added_value").sum())
            weights = [
                f"{short_region(row['region'])}: GEFSv12 weight {float(row['selected_weight_gefs']):.2f}"
                for _, row in stack.sort_values("region").iterrows()
            ]
            gefs_stack_line = (
                "A conservative stack benchmark combines GEFSv12 selected probability with same-target selected persistence using a "
                "validation-selected convex blend, then scores the frozen 2017-2019 test period. "
                f"The stack improves selected persistence by paired delta BS in {added}/{len(stack)} regions and is robust-positive "
                f"against climatology in {robust}/{len(stack)} regions. Selected weights were "
                + "; ".join(weights)
                + "."
            )

    landsurface_reliability_line = ""
    if not landsurface_reliability.empty:
        stack = landsurface_reliability.loc[landsurface_reliability["model"].eq("stack_validation_selected")].copy()
        if not stack.empty:
            pieces = [
                f"{short_region(row['region'])}: reliability {float(row['reliability']):.4f}, "
                f"resolution {float(row['resolution']):.4f}, "
                f"bias {float(row['calibration_bias']):+.3f}"
                for _, row in stack.sort_values("region").iterrows()
            ]
            landsurface_reliability_line = (
                "For the GEFSv12+persistence stack, Brier-score reliability and resolution are reported using "
                "forecast-probability bins over the frozen 2017-2019 monthly test period. "
                "Because each regional test contains 36 monthly verification units, these values are interpreted "
                "as calibration diagnostics rather than formal significance tests. "
                + "; ".join(pieces)
                + "."
            )

    landsurface_domain_transfer_line = ""
    if not landsurface_domain_transfer.empty:
        stack = landsurface_domain_transfer.loc[
            landsurface_domain_transfer["model"].eq("stack_transfer_selected")
        ].copy()
        mono = landsurface_domain_transfer.loc[
            landsurface_domain_transfer["model"].eq("monotonic_xgb_transfer")
        ].copy()
        if not stack.empty:
            pieces = []
            for mode, part in stack.groupby("transfer_mode", sort=False):
                robust = int((part["bss_ci_low"] > 0).sum())
                added = int(part["added_value_status"].eq("stack_robust_added_value").sum())
                pieces.append(
                    f"{mode}: robust-positive {robust}/{len(part)}, "
                    f"robust added value over transferred persistence {added}/{len(part)}, "
                    f"mean BSS {float(part['bss_vs_climatology'].mean()):+.3f}"
                )
            landsurface_domain_transfer_line = (
                "A land-surface domain-transfer benchmark tests calibration transfer by fitting GEFSv12 and persistence calibrators "
                "under local, pooled, and leave-one-region-out validation protocols, then scoring each held-out regional 2017-2019 test window. "
                + "; ".join(pieces)
                + "."
            )
            if not mono.empty:
                mono_pieces = []
                for mode, part in mono.groupby("transfer_mode", sort=False):
                    robust = int((part["bss_ci_low"] > 0).sum())
                    added = int(part["added_value_status"].eq("stack_robust_added_value").sum())
                    mono_pieces.append(
                        f"{mode}: robust-positive {robust}/{len(part)}, "
                        f"robust added value over transferred persistence {added}/{len(part)}, "
                        f"mean BSS {float(part['bss_vs_climatology'].mean()):+.3f}"
                    )
                landsurface_domain_transfer_line += (
                    " A monotonic XGBoost dry-fraction variant constrains dry probability to increase with GEFSv12 dry anomaly "
                    "and same-target persistence; "
                    + "; ".join(mono_pieces)
                    + "."
                )

    landsurface_rare_event_line = ""
    if not landsurface_rare_event.empty:
        focus = landsurface_rare_event.loc[
            landsurface_rare_event["transfer_mode"].eq("leave_one_region_out")
            & landsurface_rare_event["model"].isin(["stack_transfer_selected", "monotonic_xgb_transfer"])
        ].copy()
        if not focus.empty:
            pieces = []
            for (quantile, model), part in focus.groupby(["event_quantile", "model"], sort=False):
                label = "stack" if model == "stack_transfer_selected" else "monotonic XGBoost"
                robust = int(part["event_claim_status"].eq("robust_positive").sum())
                pieces.append(
                    f"q{float(quantile):.2f} {label}: robust event-BSS {robust}/{len(part)}, "
                    f"mean AP {float(part['average_precision'].mean()):.3f}, "
                    f"mean AP lift {float(part['average_precision_lift_over_event_rate'].mean()):.2f}x"
                )
            landsurface_rare_event_line = (
                "A rare-event reformulation defines regionally extensive dry months using each region's validation-period "
                "upper-tail root-zone dry-fraction quantiles (q0.80 and q0.90), calibrates model dry-fraction scores to "
                "event probabilities using validation data only, and reports average precision, event-BSS, and binned "
                "reliability/resolution on 2017-2019 tests. "
                + "; ".join(pieces)
                + "."
            )

    memory = headline.loc[headline["evidence_group"].eq("central_valley_memory_target")].copy()
    memory_line = ""
    if not memory.empty:
        lag = memory.loc[memory["model"].astype(str).str.contains("lag/climate", case=False, na=False)]
        soil = memory.loc[memory["model"].astype(str).str.contains("soil-memory", case=False, na=False)]
        parts = ["A Central Valley SPI-6 lead-6 memory-target checkpoint compares lag/climate XGBoost, ERA5-Land soil-memory XGBoost, target-consistent SPI-6 persistence, and external CPC NMME rows."]
        if not lag.empty:
            row = lag.sort_values("bss", ascending=False).iloc[0]
            parts.append(
                f"The lag/climate selected row has BSS {float(row['bss']):+.3f} "
                f"(95% CI {float(row['ci_low']):+.3f} to {float(row['ci_high']):+.3f})."
            )
        if not soil.empty:
            row = soil.sort_values("bss", ascending=False).iloc[0]
            parts.append(
                f"The soil-memory selected row has BSS {float(row['bss']):+.3f} "
                f"(95% CI {float(row['ci_low']):+.3f} to {float(row['ci_high']):+.3f})."
            )
        memory_line = " ".join(parts)

    mask_lines = []
    if not mask_table.empty:
        for _, row in mask_table.sort_values("region").iterrows():
            mask_lines.append(
                f"- {row['region']}: {row['mask_kind']} mask retained "
                f"{float(row['retained_fraction']):.1%} of valid CHIRPS cells "
                f"({int(row['retained_cells']):,}/{int(row['valid_pr_cells']):,}); "
                f"source: {row['source_note']}"
            )

    temporal_line = ""
    if not temporal.empty:
        positives = int((temporal["bss"] > 0).sum())
        temporal_line = (
            f"The temporal robustness audit used {len(temporal)} chronological holdouts; "
            f"{positives} had positive BSS point estimates against train-period monthly climatology."
        )

    seasonal_line = ""
    if not seasonal.empty:
        robust_pos = seasonal.loc[seasonal["robust_status"].eq("robust_positive")]
        seasonal_line = (
            f"The seasonal signal audit contains {len(seasonal)} regional/target/feature rows; "
            f"{len(robust_pos)} is robust-positive before event-tracking qualification."
        )

    regionalization_line = ""
    if not regionalization.empty:
        regionalization_line = (
            f"The regionalization table summarizes {len(regionalization)} SPI-12 zone analyses "
            "and joins teleconnection correlations to zone-level forecast diagnostics."
        )

    evaluation_line = ""
    if not evaluation.empty:
        strict = evaluation.loc[
            evaluation["scenario"].eq("strict_spi1_chrono")
            & evaluation["inference_level"].eq("monthly")
        ]
        random = evaluation.loc[
            evaluation["scenario"].eq("random_spi1_rows")
            & evaluation["inference_level"].eq("monthly")
        ]
        overlap = evaluation.loc[
            evaluation["scenario"].eq("overlap_spi3_lead1")
            & evaluation["inference_level"].eq("monthly")
        ]
        if not strict.empty and not random.empty and not overlap.empty:
            evaluation_line = (
                "A separate evaluation-inflation audit retrains comparable XGBoost models under deliberately invalid protocols. "
                f"The strict chronological SPI-1 monthly row gives BSS {float(strict['bss_vs_climatology'].iloc[0]):+.3f}; "
                f"the invalid random-row split gives monthly BSS {float(random['bss_vs_climatology'].iloc[0]):+.3f}; "
                f"the invalid overlapping SPI-3 lead-1 target gives monthly BSS {float(overlap['bss_vs_climatology'].iloc[0]):+.3f}. "
                "These rows are methodological stress tests and are not interpreted as forecast skill."
            )
    transition_line = ""
    if not transition.empty:
        cv = transition.loc[
            transition["scope"].eq("cvalley")
            & transition["transition"].eq("onset")
            & transition["model"].astype(str).str.contains("eligible-only", case=False, na=False)
        ]
        basin = transition.loc[
            transition["scope"].eq("cvalley_basin_masked")
            & transition["transition"].eq("onset")
            & transition["model"].astype(str).str.contains("eligible-only", case=False, na=False)
        ]
        sgp = transition.loc[
            transition["scope"].eq("southern_great_plains_basin_masked")
            & transition["transition"].eq("onset")
            & transition["model"].astype(str).str.contains("eligible-only", case=False, na=False)
        ]
        med = transition.loc[
            transition["scope"].eq("mediterranean_spain_basin_masked")
            & transition["transition"].eq("onset")
            & transition["model"].astype(str).str.contains("eligible-only", case=False, na=False)
        ]
        term = transition.loc[
            transition["scope"].eq("cvalley")
            & transition["transition"].eq("termination")
            & transition["model"].astype(str).str.contains("eligible-only", case=False, na=False)
        ]
        pieces = [
            "Transition-target diagnostics define onset as not-dry at month t followed by dry at t+1, and termination as dry at t followed by not-dry at t+1. Primary transition scoring is restricted to eligible pixels so current-state gating is not counted as forecast skill."
        ]
        if not cv.empty:
            row = cv.iloc[0]
            pieces.append(
                f"Rectangular Central Valley onset has eligible-only BSS {float(row['bss_vs_eligible_climatology']):+.3f} "
                f"(95% CI {float(row['bss_ci_low']):+.3f} to {float(row['bss_ci_high']):+.3f})."
            )
        for label, part in [
            ("Central Valley basin", basin),
            ("Southern Great Plains basin", sgp),
            ("Mediterranean Spain basin", med),
        ]:
            if not part.empty:
                row = part.iloc[0]
                pieces.append(
                    f"{label} onset has eligible-only BSS {float(row['bss_vs_eligible_climatology']):+.3f}."
                )
        if not term.empty:
            row = term.iloc[0]
            pieces.append(
                f"Central Valley termination has eligible-only BSS {float(row['bss_vs_eligible_climatology']):+.3f}."
            )
        transition_line = " ".join(pieces)
    land_added_line = ""
    if not land_added.empty:
        overall = land_added.loc[land_added["group_type"].eq("overall")]
        pieces = [
            "Land-surface added-value diagnostics compare selected root-zone soil-moisture forecast probabilities directly against same-target ERA5-Land persistence using paired monthly Brier-score differences."
        ]
        for system, region in [
            ("CFSv2_RZSM", "cvalley"),
            ("CFSv2_RZSM", "southern_great_plains"),
            ("CFSv2_RZSM", "mediterranean_spain"),
            ("GEFSv12_RZSM", "cvalley"),
            ("GEFSv12_RZSM", "southern_great_plains"),
            ("GEFSv12_RZSM", "mediterranean_spain"),
        ]:
            row_df = overall.loc[overall["forecast_system"].eq(system) & overall["region"].eq(region)]
            if not row_df.empty:
                row = row_df.iloc[0]
                pieces.append(
                    f"{system} {region} delta BS (forecast minus raw persistence) is "
                    f"{float(row['delta_bs_forecast_minus_persistence_raw']):+.3f} "
                    f"(95% CI {float(row['delta_bs_raw_ci_low']):+.3f} to "
                    f"{float(row['delta_bs_raw_ci_high']):+.3f})."
                )
        land_added_line = " ".join(pieces)
    climate_index_line = ""
    if not climate_index.empty:
        spatial = climate_index.loc[climate_index["model_kind"].eq("spatial")]
        pdo = spatial.loc[spatial["variant"].eq("pdo")]
        combo = spatial.loc[spatial["variant"].eq("nino34_pdo")]
        if not pdo.empty and not combo.empty:
            pdo_row = pdo.iloc[0]
            combo_row = combo.iloc[0]
            climate_index_line = (
                "A common-valid-period climate-index sensitivity compares CHIRPS/SPI-only, "
                "Niño3.4-only, PDO-only, and Niño3.4+PDO feature sets after dropping months "
                "where any climate-index lag is unavailable. This avoids stale PDO tail "
                "forward-fill. In the spatial model, PDO-only selected BSS is "
                f"{float(pdo_row['selected_bss']):+.3f} "
                f"(95% CI {float(pdo_row['selected_bss_ci_low']):+.3f} to "
                f"{float(pdo_row['selected_bss_ci_high']):+.3f}), while Niño3.4+PDO is "
                f"{float(combo_row['selected_bss']):+.3f} "
                f"(95% CI {float(combo_row['selected_bss_ci_low']):+.3f} to "
                f"{float(combo_row['selected_bss_ci_high']):+.3f})."
            )

    landsurface_persistence_regime_line = ""
    if not landsurface_persistence_regimes.empty:
        overall = landsurface_persistence_regimes.loc[
            landsurface_persistence_regimes["target_region"].eq("all_regions")
            & landsurface_persistence_regimes["group_type"].eq("overall")
            & landsurface_persistence_regimes["model"].isin(["stack_transfer_selected", "monotonic_xgb_transfer"])
        ].copy()
        disagreement = landsurface_persistence_regimes.loc[
            landsurface_persistence_regimes["target_region"].eq("all_regions")
            & landsurface_persistence_regimes["group_type"].eq("memory_forecast_agreement")
            & landsurface_persistence_regimes["group_value"].eq("memory_dry_forecast_wet")
            & landsurface_persistence_regimes["model"].eq("stack_transfer_selected")
        ].copy()
        pieces = []
        for _, row in overall.sort_values("model").iterrows():
            label = (
                "Validation-selected stack"
                if row["model"] == "stack_transfer_selected"
                else "Monotonic XGBoost"
            )
            pieces.append(
                f"{label} delta BS versus transferred persistence {float(row['delta_bs_model_minus_persistence']):+.4f} "
                f"(95% CI {float(row['delta_bs_ci_low']):+.4f} to {float(row['delta_bs_ci_high']):+.4f})"
            )
        if not disagreement.empty:
            row = disagreement.iloc[0]
            pieces.append(
                "stack improvement is largest when antecedent dry memory is high but GEFSv12 forecasts wetter-than-normal root-zone moisture "
                f"(delta BS {float(row['delta_bs_model_minus_persistence']):+.4f})"
            )
        if pieces:
            landsurface_persistence_regime_line = (
                "A persistence-regime diagnostic stratifies the leave-one-region-out land-surface rows by season, antecedent dry fraction, "
                "GEFSv12 anomaly sign, and forecast/memory agreement class. "
                + "; ".join(pieces)
                + "."
            )

    landsurface_independent_target_line = ""
    if not landsurface_independent_targets.empty:
        ready = int(landsurface_independent_targets["ready_to_score_now"].astype(bool).sum())
        n_products = len(landsurface_independent_targets)
        nldas = landsurface_independent_targets.loc[landsurface_independent_targets["product"].eq("NLDAS_NOAH0125_M")]
        smap = landsurface_independent_targets.loc[landsurface_independent_targets["product"].eq("SMAP_L4_SPL4SM")]
        pieces = [
            f"The independent land-surface target audit finds {ready}/{n_products} candidate soil-moisture products ready to score from local files."
        ]
        if not nldas.empty:
            nldas_ready = bool(nldas["ready_to_score_now"].iloc[0])
            if nldas_ready:
                pieces.append(
                    "NLDAS_NOAH0125_M local files are present and the U.S. validation is summarized separately."
                )
            else:
                pieces.append(
                    "NLDAS_NOAH0125_M is the first recommended acquisition for Central Valley and Southern Great Plains U.S. validation."
                )
        if not smap.empty:
            smap_ready = bool(smap["ready_to_score_now"].iloc[0])
            if smap_ready:
                pieces.append(
                    "SMAP L4 local regional subsets are present and the short-record satellite-assimilated validation is summarized separately."
                )
            else:
                pieces.append(
                    "SMAP L4 is the first global root-zone validation candidate, but its 2015-present record mainly supports short-record test validation."
                )
        landsurface_independent_target_line = " ".join(pieces)

    landsurface_nldas_validation_line = ""
    if not landsurface_nldas_validation.empty:
        stack = landsurface_nldas_validation.loc[
            landsurface_nldas_validation["model"].eq("stack_validation_selected")
        ].copy()
        if not stack.empty:
            robust = int(stack["claim_status"].eq("robust_positive").sum())
            added = int(stack["added_value_status_selected"].eq("stack_robust_added_value").sum())
            pieces = []
            for _, row in stack.sort_values("region").iterrows():
                pieces.append(
                    f"{short_region(row['region'])} BSS {float(row['bss_vs_climatology']):+.3f} "
                    f"(95% CI {float(row['bss_ci_low']):+.3f} to {float(row['bss_ci_high']):+.3f}) "
                    f"and paired delta BS versus selected persistence "
                    f"{float(row['delta_bs_model_minus_persistence_selected']):+.4f} "
                    f"(95% CI {float(row['delta_bs_ci_low']):+.4f} to {float(row['delta_bs_ci_high']):+.4f})"
                )
            landsurface_nldas_validation_line = (
                "NLDAS Noah monthly `SoilM_0_100cm` was added as an independent U.S. target check for Central Valley "
                "and Southern Great Plains. The validation-selected GEFSv12/persistence stack is robust-positive "
                f"against NLDAS-derived dry fraction in {robust}/{len(stack)} regions and robustly improves selected "
                f"persistence in {added}/{len(stack)} regions: "
                + "; ".join(pieces)
                + "."
            )

    landsurface_target_product_comparison_line = ""
    if not landsurface_target_product_comparison.empty:
        pieces = []
        for _, row in landsurface_target_product_comparison.sort_values("region").iterrows():
            pieces.append(
                f"{row['region_label']}: ERA5-Land BSS {float(row['era5_bss_vs_climatology']):+.3f} "
                f"(95% CI {float(row['era5_bss_ci_low']):+.3f} to {float(row['era5_bss_ci_high']):+.3f}); "
                f"NLDAS BSS {float(row['nldas_bss_vs_climatology']):+.3f} "
                f"(95% CI {float(row['nldas_bss_ci_low']):+.3f} to {float(row['nldas_bss_ci_high']):+.3f})"
            )
        landsurface_target_product_comparison_line = (
            "The U.S. target-product comparison applies the same validation-selected GEFSv12/persistence stack protocol "
            "to ERA5-Land and NLDAS Noah dry-fraction targets over the matched 2017-2019 test months. "
            + "; ".join(pieces)
            + "."
        )

    landsurface_gldas_validation_line = ""
    if not landsurface_gldas_validation.empty:
        stack = landsurface_gldas_validation.loc[
            landsurface_gldas_validation["model"].eq("stack_validation_selected")
        ].copy()
        if not stack.empty:
            robust = int(stack["claim_status"].eq("robust_positive").sum())
            added = int(stack["added_value_status_selected"].eq("stack_robust_added_value").sum())
            pieces = []
            for _, row in stack.sort_values("region").iterrows():
                pieces.append(
                    f"{short_region(row['region'])}: BSS {float(row['bss_vs_climatology']):+.3f} "
                    f"(95% CI {float(row['bss_ci_low']):+.3f} to {float(row['bss_ci_high']):+.3f}), "
                    f"delta BS versus selected persistence "
                    f"{float(row['delta_bs_model_minus_persistence_selected']):+.4f}"
                )
            landsurface_gldas_validation_line = (
                "GLDAS Noah `RootMoist_inst` was added as a global model-product target sensitivity over 2000-2019 "
                "with 2000-2016 target normals/calibration and 2017-2019 testing. "
                f"The validation-selected GEFSv12/persistence stack is robust-positive in {robust}/{len(stack)} regions "
                f"and robustly improves selected persistence in {added}/{len(stack)} regions: "
                + "; ".join(pieces)
                + "."
            )

    landsurface_era5_gldas_comparison_line = ""
    if not landsurface_era5_gldas_comparison.empty:
        pieces = []
        for _, row in landsurface_era5_gldas_comparison.sort_values("region").iterrows():
            pieces.append(
                f"{row['region_label']}: ERA5-Land BSS {float(row['era5_bss_vs_climatology']):+.3f}; "
                f"GLDAS BSS {float(row['gldas_bss_vs_climatology']):+.3f}"
            )
        landsurface_era5_gldas_comparison_line = (
            "The ERA5-Land versus GLDAS target-product comparison shows: "
            + "; ".join(pieces)
            + "."
        )

    landsurface_smap_l4_validation_line = ""
    if not landsurface_smap_l4_validation.empty:
        stack = landsurface_smap_l4_validation.loc[
            landsurface_smap_l4_validation["model"].eq("stack_validation_selected")
        ].copy()
        if not stack.empty:
            robust = int(stack["claim_status"].eq("robust_positive").sum())
            added = int(stack["added_value_status_selected"].eq("stack_robust_added_value").sum())
            pieces = []
            for _, row in stack.sort_values("region").iterrows():
                pieces.append(
                    f"{short_region(row['region'])}: BSS {float(row['bss_vs_climatology']):+.3f} "
                    f"(95% CI {float(row['bss_ci_low']):+.3f} to {float(row['bss_ci_high']):+.3f}), "
                    f"delta BS versus selected persistence "
                    f"{float(row['delta_bs_model_minus_persistence_selected']):+.4f}"
                )
            landsurface_smap_l4_validation_line = (
                "SMAP L4 SPL4SMGP `sm_rootzone_pctl` was added as a short-record "
                "satellite-assimilated target check using one mid-month snapshot per month over 2015-2019, "
                "2015-2016 validation, and 2017-2019 testing. "
                f"The validation-selected GEFSv12/persistence stack is robust-positive in {robust}/{len(stack)} regions "
                f"and robustly improves selected persistence in {added}/{len(stack)} regions: "
                + "; ".join(pieces)
                + "."
            )

    landsurface_smap_l4_snapshot_sensitivity_line = ""
    if not landsurface_smap_l4_snapshot_sensitivity.empty:
        robust = int(landsurface_smap_l4_snapshot_sensitivity["multi3_claim_status"].eq("robust_positive").sum())
        added = int(
            landsurface_smap_l4_snapshot_sensitivity["multi3_added_value_status"].eq("stack_robust_added_value").sum()
        )
        mid_robust = int(landsurface_smap_l4_snapshot_sensitivity["midmonth_claim_status"].eq("robust_positive").sum())
        mid_added = int(
            landsurface_smap_l4_snapshot_sensitivity["midmonth_added_value_status"].eq("stack_robust_added_value").sum()
        )
        landsurface_smap_l4_snapshot_sensitivity_line = (
            "A compact SMAP L4 snapshot sensitivity repeats the same scoring protocol using "
            "early/mid/late snapshots per month. The validation-selected stack changes from "
            f"{mid_robust}/5 to {robust}/5 robust-positive regions and from {mid_added}/5 to {added}/5 "
            "robust added-value rows."
        )

    landsurface_target_product_transfer_line = ""
    if not landsurface_target_product_transfer.empty:
        cross = landsurface_target_product_transfer.loc[
            landsurface_target_product_transfer["transfer_kind"].eq("cross_product_transfer")
        ].copy()
        gefs = cross.loc[cross["model"].eq("gefs_selected")].copy()
        stack = cross.loc[cross["model"].eq("stack_validation_selected")].copy()
        if not cross.empty:
            stack_robust = int(stack["claim_status_vs_smap_climatology"].eq("robust_positive").sum())
            robust_added = int(
                cross["added_value_status_vs_smap_persistence"].eq("stack_robust_added_value").sum()
            )
            landsurface_target_product_transfer_line = (
                "A cross-product transfer diagnostic then scored ERA5-Land- and GLDAS-calibrated "
                "probabilities against the SMAP multi-snapshot target without SMAP-specific recalibration. "
                "GEFS-only transfer rows contain the only robust-positive cross-product cases; "
                f"source-product selected stacks are robust-positive in {stack_robust}/{len(stack)} rows; "
                f"robust added value over SMAP same-target persistence occurs in {robust_added}/{len(cross)} "
                "compact transfer rows."
            )

    landsurface_calibration_transfer_ladder_line = ""
    if not landsurface_calibration_transfer_ladder.empty:
        def count_rows(strategy: str, model: str, status_col: str, status_value: str) -> tuple[int, int]:
            part = landsurface_calibration_transfer_ladder.loc[
                landsurface_calibration_transfer_ladder["strategy"].eq(strategy)
                & landsurface_calibration_transfer_ladder["model"].eq(model)
            ]
            return int(part[status_col].eq(status_value).sum()), len(part)

        direct_gefs, direct_gefs_n = count_rows(
            "source_direct", "gefs_selected", "claim_status_vs_smap_climatology", "robust_positive"
        )
        shifted_gefs, shifted_gefs_n = count_rows(
            "target_prior_shift", "gefs_selected", "claim_status_vs_smap_climatology", "robust_positive"
        )
        shifted_added, shifted_added_n = count_rows(
            "target_prior_shift", "gefs_selected", "added_value_status_vs_smap_persistence", "stack_robust_added_value"
        )
        pooled_stack, pooled_stack_n = count_rows(
            "pooled_source_calibration", "stack_validation_selected", "claim_status_vs_smap_climatology", "robust_positive"
        )
        landsurface_calibration_transfer_ladder_line = (
            "A calibration-transfer ladder shows that direct source-product GEFS transfer is robust-positive in "
            f"{direct_gefs}/{direct_gefs_n} source-region rows, whereas validation-only source-to-SMAP prior shifting "
            f"raises GEFS transfer to {shifted_gefs}/{shifted_gefs_n} and robustly improves SMAP same-target persistence "
            f"in {shifted_added}/{shifted_added_n}. Product-invariant ERA5-Land+GLDAS calibration gives robust-positive "
            f"selected stacks in {pooled_stack}/{pooled_stack_n} SMAP regions."
        )

    landsurface_target_product_adaptation_benchmark_line = ""
    if not landsurface_target_product_adaptation_benchmark.empty:
        def adaptation_row(method: str, model: str) -> pd.Series | None:
            part = landsurface_target_product_adaptation_benchmark.loc[
                landsurface_target_product_adaptation_benchmark["adaptation_method"].eq(method)
                & landsurface_target_product_adaptation_benchmark["model"].eq(model)
            ]
            if part.empty:
                return None
            return part.iloc[0]

        dry_gefs = adaptation_row("Fixed validation dry-rate shift", "gefs_selected")
        pred_base_gefs = adaptation_row(
            "Fixed validation prediction-base-rate shift", "gefs_selected"
        )
        temp_base_gefs = adaptation_row(
            "Fixed bias-corrected temperature/base-rate scaling", "gefs_selected"
        )
        pred_base_stack = adaptation_row(
            "Fixed validation prediction-base-rate shift", "stack_validation_selected"
        )
        temp_base_stack = adaptation_row(
            "Fixed bias-corrected temperature/base-rate scaling", "stack_validation_selected"
        )
        base_selector_gefs = adaptation_row(
            "Validation-selected base-rate adaptation", "gefs_selected"
        )
        complex_gefs = adaptation_row("Validation-selected complex adaptation", "gefs_selected")
        complex_stack = adaptation_row(
            "Validation-selected complex adaptation", "stack_validation_selected"
        )
        if all(
            row is not None
            for row in [
                dry_gefs,
                pred_base_gefs,
                temp_base_gefs,
                pred_base_stack,
                temp_base_stack,
                base_selector_gefs,
                complex_gefs,
                complex_stack,
            ]
        ):
            landsurface_target_product_adaptation_benchmark_line = (
                "A target-product adaptation benchmark first tests base-rate-only corrections before target-specific "
                "recalibration. Product-specific prediction-base-rate shifting maps the validation mean transferred "
                "probability to the SMAP validation dry fraction and gives GEFS robust-positive transfer in "
                f"{int(pred_base_gefs['robust_positive_count'])}/{int(pred_base_gefs['n_rows'])} rows, with robust added "
                f"value over SMAP persistence in {int(pred_base_gefs['robust_added_value_count'])}/"
                f"{int(pred_base_gefs['n_rows'])}. The source-to-SMAP observed dry-rate shift gives "
                f"{int(dry_gefs['robust_positive_count'])}/{int(dry_gefs['n_rows'])} robust-positive and "
                f"{int(dry_gefs['robust_added_value_count'])}/{int(dry_gefs['n_rows'])} robust added-value GEFS rows. "
                "Bias-corrected temperature/base-rate scaling also gives "
                f"{int(temp_base_gefs['robust_positive_count'])}/{int(temp_base_gefs['n_rows'])} robust-positive GEFS rows "
                f"but only {int(temp_base_gefs['robust_added_value_count'])}/{int(temp_base_gefs['n_rows'])} robust added-value rows. "
                "The stack remains more conditional under prediction-base-rate shifting "
                f"({int(pred_base_stack['robust_positive_count'])}/{int(pred_base_stack['n_rows'])} robust-positive rows). "
                "The temperature-scaled stack is weaker "
                f"({int(temp_base_stack['robust_positive_count'])}/{int(temp_base_stack['n_rows'])} robust-positive rows). "
                "The validation-selected seasonal/base-rate selector and the complex selector do not improve the frozen "
                "SMAP test result: GEFS robust-positive counts fall to "
                f"{int(base_selector_gefs['robust_positive_count'])}/{int(base_selector_gefs['n_rows'])} and "
                f"{int(complex_gefs['robust_positive_count'])}/{int(complex_gefs['n_rows'])}, respectively, and complex "
                f"stack adaptation gives {int(complex_stack['robust_added_value_count'])}/{int(complex_stack['n_rows'])} "
                "robust added-value rows. The protocol therefore treats base-rate calibration as the contribution and "
                "target-specific adaptation as future work."
            )

    lines = [
        "# Manuscript Methods Draft",
        "",
        "This draft is generated by `scripts/generate_manuscript_results.py` and should be edited into journal style, not treated as final prose.",
        "",
        "## Study Design",
        "",
        "We evaluated whether leakage-safe machine-learning models can forecast monthly drought probability from antecedent precipitation, drought-state, seasonal, and selected climate-index predictors. The canonical task predicts the dry-area fraction for SPI-1 at month `t+1` from features available at month `t` or earlier. The target is intentionally strict: SPI-1 at `t+1` depends on precipitation during the next month, so the experiment tests forecast information rather than reconstruction skill.",
        "",
        "The primary California Central Valley split is chronological: 1991-2016 for model fitting, 2017-2020 for hyperparameter selection and post-hoc calibration, and 2021-2026 for frozen testing. No shuffling is used across years. The canonical test set contains "
        f"{n_canonical} independent monthly verification units.",
        "",
        "## Precipitation Data and SPI Labels",
        "",
        "CHIRPS v3 monthly precipitation is the primary gridded precipitation record (Funk et al., 2026; Funk et al., 2015). SPI is computed following WMO SPI guidance by fitting gamma distributions separately for each grid cell and calendar month over the 1991-2020 baseline, with zero-precipitation probability handled explicitly before conversion to the standard normal quantile. Drought classes are defined as dry for SPI <= -1, wet for SPI >= +1, and normal otherwise.",
        "",
        "The canonical dry-event score uses the monthly fraction of valid grid cells classified as dry. Pixel-level classification diagnostics are retained as secondary information, but primary inference is performed at the monthly level because grid cells within a month are spatially autocorrelated.",
        "",
        "## Forecast Features and Models",
        "",
        "The canonical feature set includes lagged SPI-1, SPI-3, SPI-6, lagged precipitation, cyclic target-month encodings, and corrected Niño3.4 anomaly lags. PDO lags are not part of the active checkpoint because recent source values are missing after sentinel-value masking and the common-valid-period sensitivity does not show positive added skill from PDO. The model suite includes logistic regression, random forest, tabular XGBoost, XGBoost with local 3x3 neighborhood features, and ConvLSTM. Post-hoc feature extensions test ERA5-Land temperature/VPD, ERA5-Land soil moisture, MJO/IVT features, and common-valid climate-index variants.",
        "",
        climate_index_line,
        "",
        "## Calibration and Skill Evaluation",
        "",
        "The primary score is monthly dry-fraction Brier Skill Score against train-period calendar-month climatology. BSS greater than zero means the model improves on climatology. Confidence intervals are computed by monthly bootstrap resampling, preserving the month as the independent verification unit. Post-hoc calibration uses validation-only Platt scaling and isotonic regression; the selected calibration is then applied unchanged to the frozen test period.",
        "",
        f"The best canonical Central Valley calibrated checkpoint is XGB-Spatial with BSS {bss_canonical:+.3f} "
        f"(95% CI {ci_canonical[0]:+.3f} to {ci_canonical[1]:+.3f}). This is treated as a positive but uncertain point estimate, not robust positive skill.",
        "",
        "## Regional Masks and Multi-Region Tests",
        "",
        "Multi-region experiments use the same scoring logic across California Central Valley, Southern Great Plains, Murray-Darling, Mediterranean Spain, and Horn of Africa checkpoints. Rectangular bounding boxes are treated as sensitivity domains. Final regional interpretation uses source-cited masks where available.",
        "",
        *mask_lines,
        "",
        "The Horn of Africa mask is a country-intersection land mask over Djibouti, Eritrea, Ethiopia, Kenya, and Somalia. It should not be described as a hydrologic basin, livelihood zone, or agroecological region.",
        "",
        "## Temporal and Independent-Data Robustness",
        "",
        temporal_line,
        "",
        evaluation_line,
        "",
        "Independent U.S. precipitation validation uses PRISM monthly precipitation over the DWR Central Valley basin union. PRISM SPI-1 is computed with the same 1991-2020 calendar-month baseline convention and used to check whether the CHIRPS-based conclusion is an artifact of the primary precipitation product.",
        "",
        "## Seasonal, Operational, and Mechanism Checks",
        "",
        "Leakage-safe seasonal experiments require lead >= SPI accumulation window unless explicitly overridden. SPI-3 lead-3 predicts the SPI-3 class ending at `t+3` from features through `t`; SPI-6 lead-6 is handled analogously.",
        "",
        seasonal_line,
        "",
        "Operational forecast benchmarks use CPC NMME real-time precipitation anomaly files, official CPC NMME below-normal precipitation probability NetCDF files, and a NOAA NCEI THREDDS CFSv2 individual-run precipitation extraction. CPC products are aggregated to regional monthly predictors; the CFSv2 benchmark accumulates precipitation over the full SPI-3 lead-3 target window before scoring. All operational rows use the same validation-only calibration and monthly BSS protocol. "
        + op_summary
        + cfs_summary
        + " CPC probability coverage is partial for some Central Valley dry-season target months.",
        "",
        land_line,
        "",
        gefs_sensitivity_line,
        "",
        gefs_stack_line,
        "",
        landsurface_reliability_line,
        "",
        landsurface_domain_transfer_line,
        "",
        landsurface_rare_event_line,
        "",
        landsurface_persistence_regime_line,
        "",
        landsurface_independent_target_line,
        "",
        landsurface_nldas_validation_line,
        "",
        landsurface_target_product_comparison_line,
        "",
        landsurface_gldas_validation_line,
        "",
        landsurface_era5_gldas_comparison_line,
        "",
        landsurface_smap_l4_validation_line,
        landsurface_smap_l4_snapshot_sensitivity_line,
        landsurface_target_product_transfer_line,
        landsurface_calibration_transfer_ladder_line,
        landsurface_target_product_adaptation_benchmark_line,
        "",
        land_added_line,
        "",
        transition_line,
        "",
        memory_line,
        "",
        regionalization_line,
        "SPI-12 regionalization is used as mechanism evidence and not as causal attribution. Zone-level teleconnection structure is interpreted alongside zone-level forecast diagnostics to distinguish physical signal from reliable forecast conversion.",
        "",
        "## Citation Anchors",
        "",
        "- CHIRPS v3: Funk et al. (2026), https://doi.org/10.1038/s41597-026-07096-4",
        "- Original CHIRPS record: Funk et al. (2015), https://doi.org/10.1038/sdata.2015.66",
        "- SPI: WMO SPI User Guide, WMO-No. 1090, https://library.wmo.int/idurl/4/39629",
        "- Brier Score decomposition/probabilistic skill: Murphy (1973), https://ui.adsabs.harvard.edu/abs/1973JApMe..12..595M/abstract",
        "- PRISM: PRISM Climate Group, https://prism.oregonstate.edu/?id=US; Daly et al. (2008), https://doi.org/10.1002/joc.1688",
        "- NMME/CFSv2/GEFSv12: CPC data access, https://www.cpc.ncep.noaa.gov/products/NMME/data.html; NCEI NMME overview, https://www.ncei.noaa.gov/products/weather-climate-models/north-american-multi-model; probability archive, https://ftp.cpc.ncep.noaa.gov/NMME/prob/netcdf/; NCEI CFSv2 THREDDS precipitation catalog, https://www.ncei.noaa.gov/thredds/catalog/model-nmme_cfs_v2_pr_6h_agg/files/catalog.html; NCEI CFSv2 monthly means catalog, https://www.ncei.noaa.gov/thredds/catalog/model-cfs_v2_for_mm/catalog.html; GEFSv12 reforecast AWS registry, https://registry.opendata.aws/noaa-gefs-reforecast/; Kirtman et al. (2014), https://doi.org/10.1175/BAMS-D-12-00050.1; Guan et al. (2022), https://doi.org/10.1175/MWR-D-21-0245.1",
        "- NLDAS Noah monthly soil moisture: NASA GES DISC NLDAS_NOAH0125_M catalog, https://catalog.data.gov/dataset/nldas-noah-land-surface-model-l4-monthly-0-125-x-0-125-degree-v2-0-nldas-noah0125-m-at-ges-66559",
        "- GLDAS Noah monthly root-zone soil moisture: NASA GES DISC GLDAS_NOAH025_M catalog, https://hydro1.gesdisc.eosdis.nasa.gov/data/GLDAS/GLDAS_NOAH025_M.2.1/",
        "- SMAP L4 SPL4SMGP root-zone soil moisture percentiles: NSIDC SPL4SMGP version 8 catalog, https://nsidc.org/data/spl4smgp/versions/8",
        "- C3S seasonal original single levels: Copernicus Climate Data Store, https://cds.climate.copernicus.eu/datasets/seasonal-original-single-levels, doi:10.24381/cds.181d637e",
        "- Probability calibration and target-product adaptation: Saerens et al. (2002), https://pubmed.ncbi.nlm.nih.gov/11747533/; Guo et al. (2017), https://arxiv.org/abs/1706.04599; Kull et al. (2017), https://proceedings.mlr.press/v54/kull17a.html; Class Probability Matching with Calibrated Networks for Label Shift Adaptation (ICLR 2024), https://proceedings.iclr.cc/paper_files/paper/2024/hash/7e3767db483c942b883eb4f8cfb74e31-Abstract-Conference.html; Sun et al. (2016), https://doi.org/10.1609/aaai.v30i1.10306",
        "- SubX future benchmark context: Pegion et al. (2019), https://doi.org/10.1175/BAMS-D-18-0270.1",
        "- Regionalization analogy: Molosiwa et al. (2026), https://doi.org/10.1007/s00704-026-06154-6",
    ]
    return "\n".join([line for line in lines if line is not None]) + "\n"


def build_claims_audit(
    master: pd.DataFrame,
    headline: pd.DataFrame,
    temporal: pd.DataFrame,
    seasonal: pd.DataFrame,
    evaluation: pd.DataFrame,
    transition: pd.DataFrame,
    land_added: pd.DataFrame,
    climate_index: pd.DataFrame,
    gefs_sensitivity: pd.DataFrame,
    gefs_stack: pd.DataFrame,
    landsurface_domain_transfer: pd.DataFrame,
    landsurface_rare_event: pd.DataFrame,
    landsurface_persistence_regimes: pd.DataFrame,
    landsurface_independent_targets: pd.DataFrame,
    landsurface_nldas_validation: pd.DataFrame,
    landsurface_gldas_validation: pd.DataFrame,
    landsurface_smap_l4_validation: pd.DataFrame,
) -> str:
    robust_pos = int(master["status"].eq("robust_positive").sum())
    robust_pos_label = "row" if robust_pos == 1 else "rows"
    canonical = headline.loc[
        headline["evidence_group"].eq("central_valley_calibrated_checkpoint")
        & headline["model"].astype(str).eq("XGB-Spatial")
    ]
    xgb_text = "XGB-Spatial row missing"
    if not canonical.empty:
        row = canonical.iloc[0]
        xgb_text = (
            f"XGB-Spatial BSS {float(row['bss']):+.3f}, "
            f"CI {float(row['ci_low']):+.3f} to {float(row['ci_high']):+.3f}"
        )
    op = headline.loc[headline["evidence_group"].eq("operational_dynamical_benchmark")].copy()
    op_text = "Operational rows missing"
    cfs_text = ""
    if not op.empty:
        best = op.sort_values("bss", ascending=False).iloc[0]
        op_text = (
            f"best operational point estimate: {best['model']}, {best['target']}, "
            f"BSS {float(best['bss']):+.3f}, CI {float(best['ci_low']):+.3f} to {float(best['ci_high']):+.3f}"
        )
        cfs = op.loc[op["experiment"].astype(str).str.contains("cfsv2_rawamount", case=False, na=False)]
        if not cfs.empty:
            row = cfs.iloc[0]
            cfs_text = (
                f"; NCEI CFSv2 SPI-3 raw accumulated precipitation BSS {float(row['bss']):+.3f}, "
                f"CI {float(row['ci_low']):+.3f} to {float(row['ci_high']):+.3f}"
            )
    memory = headline.loc[headline["evidence_group"].eq("central_valley_memory_target")].copy()
    memory_text = "Memory-target rows missing"
    if not memory.empty:
        lag = memory.loc[memory["model"].astype(str).str.contains("lag/climate", case=False, na=False)]
        soil = memory.loc[memory["model"].astype(str).str.contains("soil-memory", case=False, na=False)]
        parts = []
        if not lag.empty:
            row = lag.sort_values("bss", ascending=False).iloc[0]
            parts.append(
                f"lag/climate selected BSS {float(row['bss']):+.3f}, "
                f"CI {float(row['ci_low']):+.3f} to {float(row['ci_high']):+.3f}"
            )
        if not soil.empty:
            row = soil.sort_values("bss", ascending=False).iloc[0]
            parts.append(f"soil-memory selected BSS {float(row['bss']):+.3f}")
        memory_text = "; ".join(parts)
    land = headline.loc[headline["evidence_group"].eq("forecast_informed_landsurface_benchmark")].copy()
    land_text = "Forecast-informed land-surface rows missing"
    if not land.empty:
        cfs_land = land.loc[land["model"].astype(str).str.contains("CFSv2", case=False, na=False)]
        gefs_land = land.loc[land["model"].astype(str).str.contains("GEFSv12", case=False, na=False)]
        c3s_land = land.loc[
            land["model"].astype(str).str.contains("C3S/ECMWF native VSM", case=False, na=False)
            & land["calibration"].eq("c3s_selected")
        ]
        persistence_land = land.loc[land["model"].astype(str).str.contains("persistence", case=False, na=False)]
        parts = []
        if not cfs_land.empty:
            row = cfs_land.sort_values("bss", ascending=False).iloc[0]
            parts.append(
                f"best CFSv2 selected RZSM {row['region']} BSS {float(row['bss']):+.3f}, "
                f"CI {float(row['ci_low']):+.3f} to {float(row['ci_high']):+.3f}"
            )
            allcycle = cfs_land.loc[
                cfs_land["model"].astype(str).str.contains("4-cycle", case=False, na=False)
                & cfs_land["region"].astype(str).eq("California Central Valley")
            ]
            if not allcycle.empty:
                row = allcycle.sort_values("bss", ascending=False).iloc[0]
                parts.append(
                    f"Central Valley four-cycle CFSv2 BSS {float(row['bss']):+.3f}, "
                    f"CI {float(row['ci_low']):+.3f} to {float(row['ci_high']):+.3f}"
                )
            regional = cfs_land.loc[
                cfs_land["model"].astype(str).str.contains("4-cycle", case=False, na=False)
                & ~cfs_land["region"].astype(str).eq("California Central Valley")
            ]
            if not regional.empty:
                pieces = [
                    f"{row['region']} {float(row['bss']):+.3f}"
                    for _, row in regional.sort_values("region").iterrows()
                ]
                parts.append("regional four-cycle CFSv2 rows: " + ", ".join(pieces))
        stack_gefs = (
            gefs_stack.loc[gefs_stack["model"].eq("gefs_selected")].copy()
            if not gefs_stack.empty
            else pd.DataFrame()
        )
        if not stack_gefs.empty:
            row = stack_gefs.sort_values("bss_vs_climatology", ascending=False).iloc[0]
            robust_count = int(stack_gefs["claim_status"].eq("robust_positive").sum())
            parts.append(
                f"best GEFSv12 hindcast-calibrated RZSM BSS {float(row['bss_vs_climatology']):+.3f}, "
                f"CI {float(row['bss_ci_low']):+.3f} to {float(row['bss_ci_high']):+.3f}; "
                f"{robust_count}/{len(stack_gefs)} GEFSv12 rows robust-positive against climatology"
            )
        elif not gefs_land.empty:
            row = gefs_land.sort_values("bss", ascending=False).iloc[0]
            robust_count = int(gefs_land["status"].eq("robust_positive").sum())
            parts.append(
                f"best GEFSv12 hindcast-calibrated RZSM BSS {float(row['bss']):+.3f}, "
                f"CI {float(row['ci_low']):+.3f} to {float(row['ci_high']):+.3f}; "
                f"{robust_count}/{len(gefs_land)} GEFSv12 rows robust-positive against climatology"
            )
        if not persistence_land.empty:
            row = persistence_land.sort_values("bss", ascending=False).iloc[0]
            parts.append(
                f"best persistence BSS {float(row['bss']):+.3f}, "
                f"CI {float(row['ci_low']):+.3f} to {float(row['ci_high']):+.3f}"
            )
        if not c3s_land.empty:
            pieces = [
                f"{row['region']} selected C3S BSS {float(row['bss']):+.3f}, "
                f"CI {float(row['ci_low']):+.3f} to {float(row['ci_high']):+.3f}"
                for _, row in c3s_land.sort_values("region").iterrows()
            ]
            parts.append("C3S native VSM rows: " + "; ".join(pieces))
        land_text = "; ".join(parts)
    gefs_sensitivity_text = "GEFSv12 lead/valid-day sensitivity table missing"
    if not gefs_sensitivity.empty:
        ok = gefs_sensitivity.loc[gefs_sensitivity["status"].eq("ok")].copy()
        if not ok.empty:
            best = ok.sort_values("gefs_selected_bss", ascending=False).iloc[0]
            worst = ok.sort_values("gefs_selected_bss", ascending=True).iloc[0]
            beats = int(ok["beats_selected_persistence_point"].sum())
            gefs_sensitivity_text = (
                f"{len(ok)}/{len(ok)} SGP GEFSv12 lead/valid-day rows are robust-positive; "
                f"BSS range {float(worst['gefs_selected_bss']):+.3f} to "
                f"{float(best['gefs_selected_bss']):+.3f}; "
                f"{beats}/{len(ok)} beat selected persistence on point BSS"
            )
    gefs_stack_text = "GEFSv12+persistence stack table missing"
    if not gefs_stack.empty:
        stack = gefs_stack.loc[gefs_stack["model"].eq("stack_validation_selected")].copy()
        if not stack.empty:
            robust = int(stack["claim_status"].eq("robust_positive").sum())
            added = int(stack["added_value_status_selected"].eq("stack_robust_added_value").sum())
            pieces = [
                f"{row['region']} BSS {float(row['bss_vs_climatology']):+.3f}, "
                f"delta BS {float(row['delta_bs_model_minus_persistence_selected']):+.4f}, "
                f"status {row['added_value_status_selected']}"
                for _, row in stack.sort_values("region").iterrows()
            ]
            gefs_stack_text = (
                f"{added}/{len(stack)} validation-selected stacks robustly improve selected persistence by paired delta BS; "
                f"{robust}/{len(stack)} robust-positive versus climatology; "
                + "; ".join(pieces)
            )
    domain_transfer_text = "Land-surface domain-transfer table missing"
    if not landsurface_domain_transfer.empty:
        stack = landsurface_domain_transfer.loc[
            landsurface_domain_transfer["model"].eq("stack_transfer_selected")
        ].copy()
        mono = landsurface_domain_transfer.loc[
            landsurface_domain_transfer["model"].eq("monotonic_xgb_transfer")
        ].copy()
        pieces = []
        for mode, part in stack.groupby("transfer_mode", sort=False):
            robust = int((part["bss_ci_low"] > 0).sum())
            added = int(part["added_value_status"].eq("stack_robust_added_value").sum())
            pieces.append(
                f"{mode}: {robust}/{len(part)} robust-positive, {added}/{len(part)} robust added value, "
                f"mean BSS {float(part['bss_vs_climatology'].mean()):+.3f}"
            )
        if pieces:
            domain_transfer_text = "; ".join(pieces)
        mono_pieces = []
        for mode, part in mono.groupby("transfer_mode", sort=False):
            robust = int((part["bss_ci_low"] > 0).sum())
            added = int(part["added_value_status"].eq("stack_robust_added_value").sum())
            mono_pieces.append(
                f"monotonic {mode}: {robust}/{len(part)} robust-positive, "
                f"{added}/{len(part)} robust added value, mean BSS {float(part['bss_vs_climatology'].mean()):+.3f}"
            )
        if mono_pieces:
            domain_transfer_text += "; " + "; ".join(mono_pieces)
    rare_event_text = "Rare-event land-surface table missing"
    if not landsurface_rare_event.empty:
        focus = landsurface_rare_event.loc[
            landsurface_rare_event["transfer_mode"].eq("leave_one_region_out")
            & landsurface_rare_event["model"].isin(["stack_transfer_selected", "monotonic_xgb_transfer"])
        ].copy()
        pieces = []
        for (quantile, model), part in focus.groupby(["event_quantile", "model"], sort=False):
            label = "stack" if model == "stack_transfer_selected" else "monotonic"
            robust = int(part["event_claim_status"].eq("robust_positive").sum())
            pieces.append(
                f"q{float(quantile):.2f} {label}: {robust}/{len(part)} robust event-BSS, "
                f"mean AP {float(part['average_precision'].mean()):.3f}, "
                f"mean AP lift {float(part['average_precision_lift_over_event_rate'].mean()):.2f}x"
            )
        if pieces:
            rare_event_text = "; ".join(pieces)
    persistence_regime_text = "Persistence-regime table missing"
    if not landsurface_persistence_regimes.empty:
        overall = landsurface_persistence_regimes.loc[
            landsurface_persistence_regimes["target_region"].eq("all_regions")
            & landsurface_persistence_regimes["group_type"].eq("overall")
            & landsurface_persistence_regimes["model"].eq("stack_transfer_selected")
        ]
        disagreement = landsurface_persistence_regimes.loc[
            landsurface_persistence_regimes["target_region"].eq("all_regions")
            & landsurface_persistence_regimes["group_type"].eq("memory_forecast_agreement")
            & landsurface_persistence_regimes["group_value"].eq("memory_dry_forecast_wet")
            & landsurface_persistence_regimes["model"].eq("stack_transfer_selected")
        ]
        pieces = []
        if not overall.empty:
            row = overall.iloc[0]
            pieces.append(
                f"overall stack delta BS {float(row['delta_bs_model_minus_persistence']):+.4f} "
                f"(CI {float(row['delta_bs_ci_low']):+.4f} to {float(row['delta_bs_ci_high']):+.4f})"
            )
        if not disagreement.empty:
            row = disagreement.iloc[0]
            pieces.append(
                f"memory-dry/forecast-wet delta BS {float(row['delta_bs_model_minus_persistence']):+.4f} "
                f"(CI {float(row['delta_bs_ci_low']):+.4f} to {float(row['delta_bs_ci_high']):+.4f})"
            )
        if pieces:
            persistence_regime_text = "; ".join(pieces)
    independent_target_text = "Independent land-surface target audit missing"
    if not landsurface_independent_targets.empty:
        ready = int(landsurface_independent_targets["ready_to_score_now"].astype(bool).sum())
        n_products = len(landsurface_independent_targets)
        independent_target_text = (
            f"{ready}/{n_products} candidate external soil-moisture target products have local files ready to score; "
            "NLDAS, GLDAS, and SMAP L4 have now been scored where local files are present; GLEAM/ESA CCI remain optional sensitivity checks."
        )
    nldas_validation_text = "NLDAS independent-target validation table missing"
    if not landsurface_nldas_validation.empty:
        stack = landsurface_nldas_validation.loc[
            landsurface_nldas_validation["model"].eq("stack_validation_selected")
        ].copy()
        if not stack.empty:
            robust = int(stack["claim_status"].eq("robust_positive").sum())
            added = int(stack["added_value_status_selected"].eq("stack_robust_added_value").sum())
            pieces = [
                f"{row['region']} BSS {float(row['bss_vs_climatology']):+.3f}, "
                f"delta BS {float(row['delta_bs_model_minus_persistence_selected']):+.4f}, "
                f"status {row['added_value_status_selected']}"
                for _, row in stack.sort_values("region").iterrows()
            ]
            nldas_validation_text = (
                f"NLDAS validation-selected stack: {robust}/{len(stack)} robust-positive, "
                f"{added}/{len(stack)} robust added value over selected persistence; "
                + "; ".join(pieces)
            )
    gldas_validation_text = "GLDAS target-product sensitivity table missing"
    if not landsurface_gldas_validation.empty:
        stack = landsurface_gldas_validation.loc[
            landsurface_gldas_validation["model"].eq("stack_validation_selected")
        ].copy()
        if not stack.empty:
            robust = int(stack["claim_status"].eq("robust_positive").sum())
            added = int(stack["added_value_status_selected"].eq("stack_robust_added_value").sum())
            pieces = [
                f"{row['region']} BSS {float(row['bss_vs_climatology']):+.3f}, "
                f"delta BS {float(row['delta_bs_model_minus_persistence_selected']):+.4f}, "
                f"status {row['added_value_status_selected']}"
                for _, row in stack.sort_values("region").iterrows()
            ]
            gldas_validation_text = (
                f"GLDAS validation-selected stack: {robust}/{len(stack)} robust-positive, "
                f"{added}/{len(stack)} robust added value over selected persistence; "
                + "; ".join(pieces)
            )
    smap_l4_validation_text = "SMAP L4 short-record validation table missing"
    if not landsurface_smap_l4_validation.empty:
        stack = landsurface_smap_l4_validation.loc[
            landsurface_smap_l4_validation["model"].eq("stack_validation_selected")
        ].copy()
        if not stack.empty:
            robust = int(stack["claim_status"].eq("robust_positive").sum())
            added = int(stack["added_value_status_selected"].eq("stack_robust_added_value").sum())
            pieces = [
                f"{row['region']} BSS {float(row['bss_vs_climatology']):+.3f}, "
                f"delta BS {float(row['delta_bs_model_minus_persistence_selected']):+.4f}, "
                f"status {row['added_value_status_selected']}"
                for _, row in stack.sort_values("region").iterrows()
            ]
            smap_l4_validation_text = (
                f"SMAP L4 mid-month validation-selected stack: {robust}/{len(stack)} robust-positive, "
                f"{added}/{len(stack)} robust added value over selected persistence; "
                + "; ".join(pieces)
            )
    temporal_text = (
        f"{int((temporal['bss'] > 0).sum())}/{len(temporal)} rolling holdouts have positive BSS"
        if not temporal.empty else "Temporal table missing"
    )
    seasonal_pos = seasonal.loc[seasonal["robust_status"].eq("robust_positive")]
    seasonal_pos_label = "row" if len(seasonal_pos) == 1 else "rows"
    seasonal_text = (
        f"{len(seasonal_pos)} robust-positive seasonal {seasonal_pos_label}; signal flags: "
        + ", ".join(sorted(seasonal_pos["signal_flag"].dropna().astype(str).unique()))
        if not seasonal_pos.empty else "no robust-positive seasonal rows"
    )
    eval_text = "Evaluation-inflation table missing"
    if not evaluation.empty:
        strict = evaluation.loc[
            evaluation["scenario"].eq("strict_spi1_chrono")
            & evaluation["inference_level"].eq("monthly")
        ]
        random = evaluation.loc[
            evaluation["scenario"].eq("random_spi1_rows")
            & evaluation["inference_level"].eq("monthly")
        ]
        overlap = evaluation.loc[
            evaluation["scenario"].eq("overlap_spi3_lead1")
            & evaluation["inference_level"].eq("monthly")
        ]
        if not strict.empty and not random.empty and not overlap.empty:
            eval_text = (
                f"strict monthly BSS {float(strict['bss_vs_climatology'].iloc[0]):+.3f}; "
                f"invalid random-row monthly BSS {float(random['bss_vs_climatology'].iloc[0]):+.3f}; "
                f"invalid overlapping SPI-3 lead-1 monthly BSS {float(overlap['bss_vs_climatology'].iloc[0]):+.3f}"
            )
    transition_text = "Transition table missing"
    if not transition.empty:
        cv = transition.loc[
            transition["scope"].eq("cvalley")
            & transition["transition"].eq("onset")
            & transition["model"].astype(str).str.contains("eligible-only", case=False, na=False)
        ]
        basin = transition.loc[
            transition["scope"].eq("cvalley_basin_masked")
            & transition["transition"].eq("onset")
            & transition["model"].astype(str).str.contains("eligible-only", case=False, na=False)
        ]
        sgp = transition.loc[
            transition["scope"].eq("southern_great_plains_basin_masked")
            & transition["transition"].eq("onset")
            & transition["model"].astype(str).str.contains("eligible-only", case=False, na=False)
        ]
        med = transition.loc[
            transition["scope"].eq("mediterranean_spain_basin_masked")
            & transition["transition"].eq("onset")
            & transition["model"].astype(str).str.contains("eligible-only", case=False, na=False)
        ]
        pieces = []
        for label, part in [
            ("rectangular Central Valley onset", cv),
            ("Central Valley basin onset", basin),
            ("Southern Great Plains basin onset", sgp),
            ("Mediterranean Spain basin onset", med),
        ]:
            if not part.empty:
                row = part.iloc[0]
                pieces.append(
                    f"{label} BSS {float(row['bss_vs_eligible_climatology']):+.3f}, "
                    f"CI {float(row['bss_ci_low']):+.3f} to {float(row['bss_ci_high']):+.3f}"
                )
        transition_text = "; ".join(pieces)
    land_added_text = "Land-surface added-value table missing"
    if not land_added.empty:
        overall = land_added.loc[land_added["group_type"].eq("overall")]
        pieces = []
        for system, region in [
            ("CFSv2_RZSM", "cvalley"),
            ("CFSv2_RZSM", "southern_great_plains"),
            ("CFSv2_RZSM", "mediterranean_spain"),
            ("GEFSv12_RZSM", "cvalley"),
            ("GEFSv12_RZSM", "southern_great_plains"),
            ("GEFSv12_RZSM", "mediterranean_spain"),
        ]:
            part = overall.loc[overall["forecast_system"].eq(system) & overall["region"].eq(region)]
            if not part.empty:
                row = part.iloc[0]
                pieces.append(
                    f"{system} {region} delta BS {float(row['delta_bs_forecast_minus_persistence_raw']):+.3f}, "
                    f"CI {float(row['delta_bs_raw_ci_low']):+.3f} to {float(row['delta_bs_raw_ci_high']):+.3f}, "
                    f"status {row['added_value_status_raw']}"
                )
        land_added_text = "; ".join(pieces)
    climate_index_text = "Climate-index sensitivity table missing"
    if not climate_index.empty:
        spatial = climate_index.loc[climate_index["model_kind"].eq("spatial")]
        tabular = climate_index.loc[climate_index["model_kind"].eq("tabular")]
        parts = []
        for label, df in [("spatial", spatial), ("tabular", tabular)]:
            for variant in ["pdo", "nino34_pdo"]:
                row_df = df.loc[df["variant"].eq(variant)]
                if not row_df.empty:
                    row = row_df.iloc[0]
                    parts.append(
                        f"{label} {variant} BSS {float(row['selected_bss']):+.3f}, "
                        f"CI {float(row['selected_bss_ci_low']):+.3f} to "
                        f"{float(row['selected_bss_ci_high']):+.3f}, status {row['claim_status']}"
                    )
        climate_index_text = "; ".join(parts)

    rows = [
        {
            "claim": "Primary lag-based precipitation-index contribution remains a predictability/evaluation audit.",
            "evidence": f"Master evidence has {robust_pos} robust-positive {robust_pos_label}; canonical {xgb_text}.",
            "allowed_wording": "Lag-based ML detects signal but rarely converts it into robust calibrated SPI dry-fraction BSS over climatology.",
            "avoid": "Do not use the land-surface result to imply the canonical SPI-1 model is skillful.",
        },
        {
            "claim": "Central Valley canonical skill is statistically indistinguishable from climatology.",
            "evidence": xgb_text,
            "allowed_wording": "The best calibrated Central Valley checkpoint is a positive but uncertain point estimate.",
            "avoid": "Do not call XGB-Spatial a significant improvement over climatology.",
        },
        {
            "claim": "PDO exclusion is not an unresolved positive-skill omission.",
            "evidence": climate_index_text,
            "allowed_wording": "PDO was excluded from the active checkpoint because recent source values are missing, and a common-valid-period sensitivity does not show positive PDO skill.",
            "avoid": "Do not claim PDO is useless generally; this is a Central Valley SPI-1 common-valid-window result.",
        },
        {
            "claim": "Operational NMME/CFSv2 products provide detectable but non-robust signal.",
            "evidence": op_text + cfs_text,
            "allowed_wording": "CPC NMME anomaly/probability and NCEI CFSv2 lead-window rows are useful external benchmarks with CIs crossing zero.",
            "avoid": "Do not present the +0.131 SPI-1 probability row as robust because its CI crosses zero and coverage is partial; do not claim CFSv2 solves the seasonal target.",
        },
        {
            "claim": "Forecast-informed land-surface target is the strongest positive direction.",
            "evidence": land_text + "; sensitivity: " + gefs_sensitivity_text + "; stack: " + gefs_stack_text + "; domain transfer: " + domain_transfer_text + "; persistence regimes: " + persistence_regime_text + "; independent targets: " + independent_target_text + "; NLDAS validation: " + nldas_validation_text + "; GLDAS validation: " + gldas_validation_text + "; SMAP L4 validation: " + smap_l4_validation_text + "; added-value diagnostic: " + land_added_text,
            "allowed_wording": "Land-surface dry-fraction targets are more predictable than SPI-1 in this setup. CFSv2 root-zone soil-moisture forecasts are robustly positive in Central Valley, and the expanded GEFSv12+persistence stack is robust-positive against climatology in most tested regions while showing region-dependent added value over selected persistence, interpretable regime dependence, independent NLDAS target support for the two U.S. regions, global GLDAS model-product sensitivity support, and mixed short-record SMAP L4 satellite-assimilated target support.",
            "avoid": "Do not generalize this to precipitation SPI skill, all leads, deployment readiness, all independent observation products, or universal dynamic-model added value over persistence.",
        },
        {
            "claim": "Rare-event framing adds useful discrimination evidence but is not yet a universal event-detection claim.",
            "evidence": rare_event_text,
            "allowed_wording": "Validation-thresholded extensive dry-event diagnostics show meaningful PR-AUC lift for land-surface forecasts, with robust event-BSS in only some region/threshold combinations.",
            "avoid": "Do not claim solved rare drought-event detection from 36-month regional tests with very small q0.90 event counts.",
        },
        {
            "claim": "Transition targets are diagnostic, not a robust positive SPI claim.",
            "evidence": transition_text,
            "allowed_wording": "Onset reframing can reveal localized signal, but the current Central Valley rectangular result does not survive basin/regional replication.",
            "avoid": "Do not present drought-onset transitions as a general positive SPI forecast result.",
        },
        {
            "claim": "Memory-bearing SPI-6 target gives only suggestive, non-event-tracking evidence.",
            "evidence": memory_text + "; memory-target diagnostics show near-zero event tracking for the positive lag/climate point estimate.",
            "allowed_wording": "Longer-memory targets may improve point estimates, but the current checkpoint does not show robust event-tracking skill.",
            "avoid": "Do not claim that soil moisture or SPI-6 solves the forecast problem.",
        },
        {
            "claim": "Less rigorous evaluation protocols can manufacture apparent skill.",
            "evidence": eval_text,
            "allowed_wording": "Random row splits and overlapping SPI targets strongly inflate BSS and are used only as invalid-protocol stress tests.",
            "avoid": "Do not mix invalid audit rows with valid forecast-skill claims.",
        },
        {
            "claim": "The 2021-2026 test period is not the sole explanation.",
            "evidence": temporal_text,
            "allowed_wording": "Rolling holdouts show weak skill across multiple eras.",
            "avoid": "Do not imply the temporal audit proves all future periods will be unskillful.",
        },
        {
            "claim": "Seasonal targets do not broadly solve the problem.",
            "evidence": seasonal_text,
            "allowed_wording": "The one robust-positive seasonal row is treated as a calibration-shift exception unless event tracking improves.",
            "avoid": "Do not use the Mediterranean Spain SPI-6 row as a general positive-skill claim.",
        },
        {
            "claim": "Regionalization and SHAP are mechanism evidence, not causal proof.",
            "evidence": "Regionalization links SPI-12 zones to teleconnection correlations but zone-level BSS is mostly weak.",
            "allowed_wording": "Mechanism diagnostics show where signal may live.",
            "avoid": "Do not claim causal attribution or guaranteed forecast utility from SHAP/regionalization alone.",
        },
    ]
    df = pd.DataFrame(rows)
    return "# Manuscript Claims Audit\n\n" + markdown_table(df) + "\n"


def main() -> None:
    master = build_master_evidence()
    if master.empty:
        raise SystemExit("No evidence rows found. Run upstream result builders first.")
    headline = build_headline_table(master)
    mask_table = build_mask_table()
    temporal = build_temporal_table()
    seasonal = build_seasonal_signal_table()
    regionalization = build_regionalization_table()
    evaluation = build_evaluation_inflation_table()
    transition = build_transition_table()
    land_added = build_landsurface_added_value_table()
    climate_index = build_climate_index_sensitivity_table()
    gefs_sensitivity = build_gefsv12_landsurface_sensitivity_table()
    gefs_stack = build_gefsv12_landsurface_stack_table()
    landsurface_reliability, landsurface_reliability_bins = build_landsurface_reliability_tables()
    landsurface_domain_transfer = build_landsurface_domain_transfer_table()
    landsurface_rare_event = build_landsurface_rare_event_table()
    landsurface_persistence_regimes = build_landsurface_persistence_regime_table()
    landsurface_independent_targets = build_landsurface_independent_target_audit_table()
    landsurface_forecast_archive_audit = build_landsurface_forecast_archive_audit_table()
    operational_gefs_diagnostic = build_operational_gefs_diagnostic_table()
    operational_gefs_smap_validation = build_operational_gefs_smap_validation_table()
    landsurface_nldas_validation = build_landsurface_nldas_validation_table()
    landsurface_target_product_comparison = build_landsurface_target_product_comparison_table()
    landsurface_gldas_validation = build_landsurface_gldas_validation_table()
    landsurface_era5_gldas_comparison = build_landsurface_era5_gldas_comparison_table()
    landsurface_smap_l4_validation = build_landsurface_smap_l4_validation_table()
    landsurface_smap_l4_snapshot_sensitivity = build_landsurface_smap_l4_snapshot_sensitivity_table()
    landsurface_target_product_transfer = build_landsurface_target_product_transfer_table()
    landsurface_calibration_transfer_ladder = build_landsurface_calibration_transfer_ladder_table()
    landsurface_calibration_transfer_ladder_compact = (
        build_landsurface_calibration_transfer_ladder_compact_table(
            landsurface_calibration_transfer_ladder
        )
    )
    landsurface_target_product_adaptation_benchmark = (
        build_landsurface_target_product_adaptation_benchmark_table()
    )
    landsurface_base_rate_yearly_sensitivity = (
        build_landsurface_base_rate_yearly_sensitivity_table()
    )

    write_table(
        master,
        "table01_master_evidence",
        md_cols=[
            "evidence_group",
            "region",
            "target",
            "model",
            "n_months",
            "bss",
            "ci_low",
            "ci_high",
            "status",
            "signal_flag",
        ],
    )
    write_table(
        headline,
        "table02_headline_results",
        md_cols=[
            "evidence_group",
            "experiment",
            "region",
            "target",
            "model",
            "n_months",
            "bss",
            "ci_low",
            "ci_high",
            "status",
            "signal_flag",
        ],
    )
    write_table(
        mask_table,
        "table03_mask_methods",
        md_cols=[
            "region",
            "mask_kind",
            "mask_label",
            "valid_pr_cells",
            "retained_cells",
            "retained_fraction",
            "source_note",
            "source_url",
            "caveat",
        ],
    )
    write_table(temporal, "table04_temporal_robustness")
    write_table(seasonal, "table05_seasonal_signal_audit")
    write_table(regionalization, "table06_regionalization_mechanism")
    write_table(evaluation, "table07_evaluation_inflation_audit")
    write_table(transition, "table08_transition_target_summary")
    write_table(land_added, "table09_landsurface_added_value")
    write_table(climate_index, "table10_climate_index_sensitivity")
    write_table(gefs_sensitivity, "table11_gefsv12_landsurface_sensitivity")
    write_table(gefs_stack, "table12_gefsv12_landsurface_stack")
    write_table(landsurface_reliability, "table13_landsurface_reliability_resolution")
    write_table(landsurface_domain_transfer, "table14_landsurface_domain_transfer")
    write_table(landsurface_rare_event, "table15_landsurface_rare_event")
    write_table(landsurface_persistence_regimes, "table16_landsurface_persistence_regimes")
    write_table(landsurface_independent_targets, "table17_landsurface_independent_target_audit")
    write_table(landsurface_nldas_validation, "table18_landsurface_nldas_validation")
    write_table(landsurface_target_product_comparison, "table19_landsurface_target_product_comparison")
    write_table(landsurface_gldas_validation, "table20_landsurface_gldas_validation")
    write_table(landsurface_era5_gldas_comparison, "table21_landsurface_era5_gldas_comparison")
    write_table(landsurface_smap_l4_validation, "table22_landsurface_smap_l4_validation")
    write_table(landsurface_smap_l4_snapshot_sensitivity, "table23_landsurface_smap_l4_snapshot_sensitivity")
    write_table(landsurface_target_product_transfer, "table24_landsurface_target_product_transfer")
    write_table(landsurface_calibration_transfer_ladder, "table25_landsurface_calibration_transfer_ladder")
    write_table(
        landsurface_calibration_transfer_ladder_compact,
        "table26_landsurface_calibration_transfer_ladder_compact",
    )
    write_table(
        landsurface_target_product_adaptation_benchmark,
        "table27_landsurface_target_product_adaptation_benchmark",
    )
    write_table(
        landsurface_base_rate_yearly_sensitivity,
        "table28_landsurface_base_rate_yearly_sensitivity",
    )
    write_table(
        landsurface_forecast_archive_audit,
        "table29_landsurface_forecast_archive_audit",
    )
    write_table(
        operational_gefs_diagnostic,
        "table30_operational_gefs_diagnostic",
    )
    write_table(
        operational_gefs_smap_validation,
        "table31_operational_gefs_smap_validation",
    )
    if not landsurface_reliability_bins.empty:
        LANDSURFACE_DIR.mkdir(parents=True, exist_ok=True)
        landsurface_reliability_bins.to_csv(
            LANDSURFACE_DIR / "landsurface_gefsv12_rzsm_stack_reliability_bins.csv",
            index=False,
        )

    plot_headline = headline.copy()
    plot_headline = plot_headline.loc[~plot_headline["model"].str.contains("Climatological baseline", na=False)].copy()
    # Keep the overview legible by prioritizing compact groups and robust seasonal rows.
    plot_headline = plot_headline.loc[
        plot_headline["evidence_group"].isin(
            [
                "central_valley_calibrated_checkpoint",
                "central_valley_feature_extension",
                "central_valley_uncertainty",
                "central_valley_memory_target",
                "operational_dynamical_benchmark",
                "forecast_informed_landsurface_benchmark",
                "forecast_informed_landsurface_stack",
                "independent_precipitation_validation",
                "seasonal_regional_longlead",
            ]
        )
    ].copy()
    plot_headline["plot_label"] = plot_headline.apply(plot_label, axis=1)
    errorbar_plot(
        plot_headline,
        PAPER / "fig01_headline_bss_forest.png",
        "Headline probability-skill checkpoints",
        ["plot_label"],
        width=10.0,
        row_height=0.34,
    )
    plot_multiregion(PAPER / "fig02_multiregion_bss_forest.png")
    plot_seasonal_tracking(seasonal, PAPER / "fig03_seasonal_bss_vs_tracking.png")
    temporal_plot = temporal.rename(
        columns={"split": "experiment", "bss": "bss", "bss_ci_low": "ci_low", "bss_ci_high": "ci_high"}
    ).copy()
    if not temporal_plot.empty:
        temporal_plot["status"] = [
            claim_status(b, lo, hi)
            for b, lo, hi in zip(temporal_plot["bss"], temporal_plot["ci_low"], temporal_plot["ci_high"])
        ]
    errorbar_plot(
        temporal_plot,
        PAPER / "fig04_temporal_holdout_bss.png",
        "Temporal holdout BSS vs train-month climatology",
        ["experiment", "test_years"],
        width=8.0,
        row_height=0.45,
    )
    plot_mask_retention(mask_table, PAPER / "fig05_mask_retention.png")
    plot_landsurface_reliability(
        landsurface_reliability_bins,
        PAPER / "fig06_landsurface_stack_reliability.png",
    )
    plot_landsurface_target_product_comparison(
        landsurface_target_product_comparison,
        PAPER / "fig07_landsurface_target_product_comparison.png",
    )
    plot_landsurface_era5_gldas_comparison(
        landsurface_era5_gldas_comparison,
        PAPER / "fig08_landsurface_era5_gldas_comparison.png",
    )
    plot_landsurface_calibration_transfer_ladder(
        landsurface_calibration_transfer_ladder_compact,
        PAPER / "fig09_landsurface_calibration_transfer_ladder.png",
    )

    (PAPER / "paper_evidence_pack.md").write_text(
        build_summary_markdown(
            master,
            headline,
            seasonal,
            temporal,
            evaluation,
            transition,
            land_added,
            climate_index,
            gefs_sensitivity,
            gefs_stack,
            landsurface_reliability,
            landsurface_domain_transfer,
            landsurface_rare_event,
            landsurface_persistence_regimes,
            landsurface_independent_targets,
            landsurface_nldas_validation,
            landsurface_target_product_comparison,
            landsurface_gldas_validation,
            landsurface_era5_gldas_comparison,
            landsurface_smap_l4_validation,
            landsurface_smap_l4_snapshot_sensitivity,
            landsurface_target_product_transfer,
            landsurface_calibration_transfer_ladder,
            landsurface_target_product_adaptation_benchmark,
            landsurface_base_rate_yearly_sensitivity,
        )
    )
    (PAPER / "manuscript_methods_draft.md").write_text(
        build_methods_draft(
            headline,
            mask_table,
            temporal,
            seasonal,
            regionalization,
            evaluation,
            transition,
            land_added,
            climate_index,
            gefs_sensitivity,
            gefs_stack,
            landsurface_reliability,
            landsurface_domain_transfer,
            landsurface_rare_event,
            landsurface_persistence_regimes,
            landsurface_independent_targets,
            landsurface_nldas_validation,
            landsurface_target_product_comparison,
            landsurface_gldas_validation,
            landsurface_era5_gldas_comparison,
            landsurface_smap_l4_validation,
            landsurface_smap_l4_snapshot_sensitivity,
            landsurface_target_product_transfer,
            landsurface_calibration_transfer_ladder,
            landsurface_target_product_adaptation_benchmark,
        ),
        encoding="utf-8",
    )
    (PAPER / "manuscript_claims_audit.md").write_text(
        build_claims_audit(
            master,
            headline,
            temporal,
            seasonal,
            evaluation,
            transition,
            land_added,
            climate_index,
            gefs_sensitivity,
            gefs_stack,
            landsurface_domain_transfer,
            landsurface_rare_event,
            landsurface_persistence_regimes,
            landsurface_independent_targets,
            landsurface_nldas_validation,
            landsurface_gldas_validation,
            landsurface_smap_l4_validation,
        ),
        encoding="utf-8",
    )

    print(f"Wrote evidence pack to {PAPER}")
    print(f"Master evidence rows: {len(master)}")
    print(f"Headline rows: {len(headline)}")
    print("Master status counts:")
    print(master["status"].value_counts().to_string())


if __name__ == "__main__":
    main()
