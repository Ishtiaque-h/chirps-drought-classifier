#!/usr/bin/env python
"""Build paper-facing evidence tables and figures from completed experiments.

This script does not train models. It consolidates the current result artifacts
into a small evidence pack for manuscript drafting:

  results/paper/table01_master_evidence.csv
  results/paper/table02_headline_results.csv
  results/paper/table03_mask_methods.csv
  results/paper/table04_temporal_robustness.csv
  results/paper/table05_seasonal_signal_audit.csv
  results/paper/table06_regionalization_mechanism.csv
  results/paper/fig01_headline_bss_forest.png
  results/paper/fig02_multiregion_bss_forest.png
  results/paper/fig03_seasonal_bss_vs_tracking.png
  results/paper/fig04_temporal_holdout_bss.png
  results/paper/fig05_mask_retention.png
  results/paper/paper_evidence_pack.md
"""
from __future__ import annotations

from pathlib import Path
import textwrap

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS = PROJECT_ROOT / "results"
REPORT = RESULTS / "report"
PAPER = RESULTS / "paper"
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
        "California Central Valley": "Central Valley",
        "Mediterranean Spain bounding box (basin-mask sensitivity)": "Med Spain basin",
        "Mediterranean Spain bounding box (country-mask sensitivity)": "Med Spain country",
        "Mediterranean Spain bounding box": "Med Spain bbox",
        "mediterranean_spain_basin_masked": "Med Spain basin",
        "Southern Great Plains (basin-mask sensitivity)": "SGP basin",
        "Southern Great Plains": "SGP bbox",
        "southern_great_plains_basin_masked": "SGP basin",
        "Murray-Darling Basin bounding box (basin-mask sensitivity)": "Murray-Darling basin",
        "murray_darling_basin_masked": "Murray-Darling basin",
        "Horn of Africa bounding box (country-mask sensitivity)": "Horn country",
        "horn_of_africa_country_masked": "Horn country",
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


def write_table(df: pd.DataFrame, stem: str, md_cols: list[str] | None = None, digits: int = 3) -> tuple[Path, Path]:
    csv_path = PAPER / f"{stem}.csv"
    md_path = PAPER / f"{stem}.md"
    df.to_csv(csv_path, index=False)
    display = df[md_cols] if md_cols else df
    md_path.write_text(markdown_table(display, digits=digits) + "\n")
    return csv_path, md_path


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
    path = RESULTS / "seasonal" / "seasonal_regional_longlead_summary.csv"
    audit_path = RESULTS / "seasonal" / "seasonal_regional_signal_audit.csv"
    df = read_csv(path)
    audit = read_csv(audit_path)
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


def build_master_evidence() -> pd.DataFrame:
    parts = [
        evidence_from_master_headline(),
        evidence_from_seasonal_regional(),
        evidence_from_temporal(),
        evidence_from_prism(),
    ]
    df = pd.concat([p for p in parts if not p.empty], ignore_index=True)
    df["paper_priority"] = False

    priority_patterns = [
        "XGB-Spatial",
        "External precipitation forecast selected",
        "CPC NMME anomaly forecast selected",
        "CPC NMME probability forecast raw",
        "CPC NMME probability forecast selected",
        "Seasonal SPI-3 lead-3",
        "Seasonal SPI-6 lead-6",
        "EDL MLP selected",
        "XGBoost + soil moisture selected",
        "XGBoost + regional temperature/VPD selected",
        "XGB-Spatial + gridded temperature/VPD selected",
        "XGBoost + MJO/IVT selected",
    ]
    def is_priority_model(model: object) -> bool:
        text = "" if pd.isna(model) else str(model).lower()
        return any(pattern.lower() in text for pattern in priority_patterns)

    df.loc[df["model"].map(is_priority_model), "paper_priority"] = True
    df.loc[df["evidence_group"].eq("seasonal_regional_longlead") & df["status"].isin(["robust_positive", "robust_negative"]), "paper_priority"] = True
    df.loc[df["evidence_group"].eq("temporal_robustness"), "paper_priority"] = True
    df.loc[df["evidence_group"].eq("independent_precipitation_validation"), "paper_priority"] = True
    df.loc[df["evidence_group"].eq("multi_region_selected_checkpoint"), "paper_priority"] = True
    df = df.sort_values(["evidence_group", "region", "target", "lead_months", "model"]).reset_index(drop=True)
    return df


def build_headline_table(master: pd.DataFrame) -> pd.DataFrame:
    keep_groups = {
        "central_valley_calibrated_checkpoint",
        "central_valley_feature_extension",
        "central_valley_uncertainty",
        "operational_dynamical_benchmark",
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
    df = read_csv(RESULTS / "seasonal" / "seasonal_regional_signal_audit.csv")
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
    mech = read_csv(RESULTS / "regionalization" / "regionalization_mechanism_summary.csv")
    zone = read_csv(RESULTS / "regionalization" / "zone_forecast_diagnostics.csv")
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


def build_summary_markdown(master: pd.DataFrame, headline: pd.DataFrame, seasonal: pd.DataFrame, temporal: pd.DataFrame) -> str:
    robust_pos = master.loc[master["status"].eq("robust_positive")]
    canonical = master.loc[master["evidence_group"].eq("central_valley_calibrated_checkpoint")]
    operational = master.loc[
        (master["evidence_group"].eq("operational_dynamical_benchmark"))
        & (master["model"].astype(str).str.contains("selected", case=False, na=False))
    ]
    operational_prob_raw = master.loc[
        (master["evidence_group"].eq("operational_dynamical_benchmark"))
        & (master["model"].astype(str).str.contains("probability", case=False, na=False))
        & (master["model"].astype(str).str.contains(" raw", case=False, na=False))
    ]
    seasonal_pos = seasonal.loc[seasonal["robust_status"].eq("robust_positive")]
    temporal_positive = int((temporal["bss"] > 0).sum()) if not temporal.empty else 0

    lines = [
        "# Paper Evidence Pack",
        "",
        "This folder consolidates completed experiments into manuscript-facing tables and figures. It does not retrain models.",
        "",
        "## Core Interpretation",
        "",
        "The strongest current claim is a predictability and evaluation audit: lag-based ML detects drought-relevant structure, but it rarely converts that signal into robust calibrated monthly BSS over climatology once leakage-safe targets, validation-only calibration, regional masks, independent precipitation validation, and monthly bootstrap uncertainty are enforced.",
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
    if not seasonal_pos.empty:
        row = seasonal_pos.iloc[0]
        lines.append(
            f"- Seasonal regional robust-positive exception: {row['region']} SPI-{int(row['target_spi'])} lead-{int(row['lead_months'])} "
            f"BSS {float(row['bss_iso_vs_clim']):+.3f}, but signal_flag={row['signal_flag']}"
        )
    lines.extend(
        [
            "",
            "## Generated Files",
            "",
            "- `table01_master_evidence.csv/md`: all consolidated evidence rows.",
            "- `table02_headline_results.csv/md`: compact manuscript headline table.",
            "- `table03_mask_methods.csv/md`: source-cited mask methods and retained-cell fractions.",
            "- `table04_temporal_robustness.csv/md`: rolling holdout control for test-period non-representativeness.",
            "- `table05_seasonal_signal_audit.csv/md`: seasonal BSS interpreted with event-tracking diagnostics.",
            "- `table06_regionalization_mechanism.csv/md`: SPI-12 mechanism evidence joined with zone-level forecast diagnostics.",
            "- `fig01_headline_bss_forest.png`: headline BSS forest plot.",
            "- `fig02_multiregion_bss_forest.png`: multi-region selected BSS forest plot.",
            "- `fig03_seasonal_bss_vs_tracking.png`: seasonal BSS vs event-tracking correlation.",
            "- `fig04_temporal_holdout_bss.png`: temporal holdout BSS forest plot.",
            "- `fig05_mask_retention.png`: retained-cell fractions for source-cited masks.",
            "- `manuscript_results_discussion_draft.md`: prose draft for Results and Discussion.",
            "- `methods_sources_and_evidence_index.md`: source/citation and claim-to-evidence checklist.",
            "",
            "## Manuscript Guardrails",
            "",
            "- Do not claim a broadly positive operational forecast model.",
            "- Treat positive point estimates with confidence intervals crossing zero as hypothesis-generating.",
            "- Treat the Mediterranean Spain SPI-6 seasonal robust-positive row as a calibration-shift exception unless follow-up diagnostics show stronger temporal tracking.",
            "- Use regionalization and SHAP as mechanism evidence, not proof of calibrated forecast skill.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    master = build_master_evidence()
    if master.empty:
        raise SystemExit("No evidence rows found. Run upstream result builders first.")
    headline = build_headline_table(master)
    mask_table = build_mask_table()
    temporal = build_temporal_table()
    seasonal = build_seasonal_signal_table()
    regionalization = build_regionalization_table()

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

    plot_headline = headline.copy()
    plot_headline = plot_headline.loc[~plot_headline["model"].str.contains("Climatological baseline", na=False)].copy()
    # Keep the overview legible by prioritizing compact groups and robust seasonal rows.
    plot_headline = plot_headline.loc[
        plot_headline["evidence_group"].isin(
            [
                "central_valley_calibrated_checkpoint",
                "central_valley_feature_extension",
                "central_valley_uncertainty",
                "operational_dynamical_benchmark",
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

    (PAPER / "paper_evidence_pack.md").write_text(
        build_summary_markdown(master, headline, seasonal, temporal)
    )

    print(f"Wrote evidence pack to {PAPER}")
    print(f"Master evidence rows: {len(master)}")
    print(f"Headline rows: {len(headline)}")
    print("Master status counts:")
    print(master["status"].value_counts().to_string())


if __name__ == "__main__":
    main()
