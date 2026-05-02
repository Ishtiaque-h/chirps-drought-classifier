#!/usr/bin/env python
"""
Analyze why multi-region SPI-1 forecast skill differs across regions.

This script consumes completed outputs from run_multiregion_xgb_experiment.py
and creates reproducible mechanism diagnostics:
  - selected/raw BSS and confidence intervals
  - monthly dry-fraction bias, correlation, and amplitude diagnostics
  - train/validation/test dry-fraction regime shifts and lag-1 persistence
  - XGBoost feature-gain shares by physical feature group
  - compact figures for report review

The goal is scientific interpretation, not another model run.
"""
from __future__ import annotations

from pathlib import Path
import math
import shutil

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT_ROOT = PROJECT_ROOT / "outputs" / "multiregion"
REPORT_ROOT = PROJECT_ROOT / "results" / "multiregion"
DATA_ROOT = PROJECT_ROOT / "data" / "processed"
REGION_DATA_ROOT = DATA_ROOT / "regions"

SUMMARY_PATH = OUT_ROOT / "multiregion_summary.csv"
FULL_SUMMARY_REPORT = REPORT_ROOT / "multiregion_summary.csv"

SPLITS = {
    "train": (None, 2016),
    "validation": (2017, 2020),
    "test": (2021, None),
}


def corr(a: pd.Series | np.ndarray, b: pd.Series | np.ndarray) -> float:
    a_arr = np.asarray(a, dtype=float)
    b_arr = np.asarray(b, dtype=float)
    mask = np.isfinite(a_arr) & np.isfinite(b_arr)
    if mask.sum() < 3:
        return float("nan")
    if np.nanstd(a_arr[mask]) == 0 or np.nanstd(b_arr[mask]) == 0:
        return float("nan")
    return float(np.corrcoef(a_arr[mask], b_arr[mask])[0, 1])


def brier(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean((np.asarray(p, dtype=float) - np.asarray(y, dtype=float)) ** 2))


def safe_std_ratio(pred: np.ndarray, obs: np.ndarray) -> float:
    pred_std = float(np.nanstd(pred, ddof=1))
    obs_std = float(np.nanstd(obs, ddof=1))
    return pred_std / obs_std if obs_std > 0 else float("nan")


def model_monthly_path(region: str, model: str) -> Path:
    return OUT_ROOT / region / f"{model}_monthly_scores.csv"


def model_path(region: str, model: str) -> Path:
    return OUT_ROOT / region / f"{model}_model.json"


def dataset_path(region: str) -> Path:
    if region == "cvalley":
        return DATA_ROOT / "dataset_forecast.parquet"
    return REGION_DATA_ROOT / region / f"dataset_forecast_{region}.parquet"


def load_summary() -> pd.DataFrame:
    if not SUMMARY_PATH.exists():
        raise FileNotFoundError(f"Missing multi-region summary: {SUMMARY_PATH}")
    summary = pd.read_csv(SUMMARY_PATH)
    summary = summary[~summary["region"].str.contains("_stride", regex=False)].copy()
    return summary.sort_values(["region", "model"]).reset_index(drop=True)


def monthly_diagnostics(summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for row in summary.itertuples(index=False):
        monthly = pd.read_csv(model_monthly_path(row.region, row.model), parse_dates=["target_time"])
        y = monthly["y_true_dry_frac"].to_numpy(dtype=float)
        clim = monthly["clim_prob_dry"].to_numpy(dtype=float)
        raw = monthly["xgb_raw_prob_dry"].to_numpy(dtype=float)
        selected = monthly["xgb_selected_prob_dry"].to_numpy(dtype=float)
        iso = monthly["xgb_isotonic_prob_dry"].to_numpy(dtype=float)
        platt = monthly["xgb_platt_prob_dry"].to_numpy(dtype=float)

        for label, pred in {
            "climatology": clim,
            "raw": raw,
            "isotonic": iso,
            "platt": platt,
            "selected": selected,
        }.items():
            rows.append(
                {
                    "region": row.region,
                    "region_name": row.region_name,
                    "model": row.model,
                    "forecast": label,
                    "bs": brier(y, pred),
                    "bias_mean_pred_minus_obs": float(np.nanmean(pred - y)),
                    "mae": float(np.nanmean(np.abs(pred - y))),
                    "corr_with_observed": corr(y, pred),
                    "std_pred": float(np.nanstd(pred, ddof=1)),
                    "std_observed": float(np.nanstd(y, ddof=1)),
                    "amplitude_ratio_pred_to_obs": safe_std_ratio(pred, y),
                    "mean_pred": float(np.nanmean(pred)),
                    "mean_observed": float(np.nanmean(y)),
                    "min_pred": float(np.nanmin(pred)),
                    "max_pred": float(np.nanmax(pred)),
                    "min_observed": float(np.nanmin(y)),
                    "max_observed": float(np.nanmax(y)),
                }
            )
    return pd.DataFrame(rows)


def mechanism_summary(summary: pd.DataFrame, diagnostics: pd.DataFrame) -> pd.DataFrame:
    selected = diagnostics[diagnostics["forecast"] == "selected"].copy()
    raw = diagnostics[diagnostics["forecast"] == "raw"].copy()
    clim = diagnostics[diagnostics["forecast"] == "climatology"].copy()
    key_cols = ["region", "region_name", "model"]
    merged = summary.merge(
        selected[
            key_cols
            + [
                "bs",
                "bias_mean_pred_minus_obs",
                "corr_with_observed",
                "amplitude_ratio_pred_to_obs",
                "mean_pred",
                "mean_observed",
                "std_observed",
            ]
        ].rename(
            columns={
                "bs": "selected_monthly_bs_check",
                "bias_mean_pred_minus_obs": "selected_bias",
                "corr_with_observed": "corr_selected_with_observed",
                "amplitude_ratio_pred_to_obs": "selected_amplitude_ratio",
                "mean_pred": "mean_selected_prob_dry",
                "mean_observed": "mean_observed_dry_frac",
                "std_observed": "std_observed_dry_frac",
            }
        ),
        on=key_cols,
        how="left",
    )
    merged = merged.merge(
        raw[key_cols + ["corr_with_observed", "amplitude_ratio_pred_to_obs"]].rename(
            columns={
                "corr_with_observed": "corr_raw_with_observed",
                "amplitude_ratio_pred_to_obs": "raw_amplitude_ratio",
            }
        ),
        on=key_cols,
        how="left",
    )
    merged = merged.merge(
        clim[key_cols + ["corr_with_observed", "amplitude_ratio_pred_to_obs", "mean_pred"]].rename(
            columns={
                "corr_with_observed": "corr_clim_with_observed",
                "amplitude_ratio_pred_to_obs": "clim_amplitude_ratio",
                "mean_pred": "mean_clim_prob_dry",
            }
        ),
        on=key_cols,
        how="left",
    )
    cols = [
        "region",
        "region_name",
        "model",
        "selected_bss",
        "selected_bss_ci_low",
        "selected_bss_ci_high",
        "raw_bss",
        "pixel_dry_roc_auc_raw",
        "mean_observed_dry_frac",
        "std_observed_dry_frac",
        "mean_clim_prob_dry",
        "mean_selected_prob_dry",
        "selected_bias",
        "corr_clim_with_observed",
        "corr_raw_with_observed",
        "corr_selected_with_observed",
        "clim_amplitude_ratio",
        "raw_amplitude_ratio",
        "selected_amplitude_ratio",
    ]
    return merged[cols].sort_values(["region", "model"]).reset_index(drop=True)


def split_name_from_year(year: int) -> str:
    if year <= 2016:
        return "train"
    if year <= 2020:
        return "validation"
    return "test"


def lag1_autocorr(values: pd.Series) -> float:
    values = values.astype(float)
    if len(values) < 3:
        return float("nan")
    return corr(values.iloc[1:].to_numpy(), values.iloc[:-1].to_numpy())


def split_dry_stats(regions: list[str]) -> pd.DataFrame:
    rows = []
    for region in regions:
        path = dataset_path(region)
        if not path.exists():
            print(f"Skipping split dry stats for missing dataset: {path}")
            continue
        print(f"Loading regional forecast table for split stats: {path}")
        df = pd.read_parquet(path, columns=["time", "year", "month", "target_label"])
        df["target_time"] = (
            pd.to_datetime(df["time"]) + pd.DateOffset(months=1)
        ).dt.to_period("M").dt.to_timestamp()
        df["target_year"] = df["target_time"].dt.year
        df["split"] = df["target_year"].map(split_name_from_year)
        df["is_dry"] = (df["target_label"] == -1).astype(float)
        monthly = (
            df.groupby(["split", "target_time", "month"], observed=True)
            .agg(dry_frac=("is_dry", "mean"), n_pixels=("is_dry", "size"))
            .reset_index()
            .sort_values("target_time")
        )
        for split, sub in monthly.groupby("split", sort=False):
            dry = sub["dry_frac"].astype(float)
            month_means = sub.groupby("month")["dry_frac"].mean()
            rows.append(
                {
                    "region": region,
                    "split": split,
                    "n_months": int(sub["target_time"].nunique()),
                    "mean_dry_frac": float(dry.mean()),
                    "std_dry_frac": float(dry.std(ddof=1)),
                    "min_dry_frac": float(dry.min()),
                    "max_dry_frac": float(dry.max()),
                    "lag1_autocorr_dry_frac": lag1_autocorr(dry.reset_index(drop=True)),
                    "months_dry_frac_ge_0_25": int((dry >= 0.25).sum()),
                    "months_dry_frac_ge_0_50": int((dry >= 0.50).sum()),
                    "months_dry_frac_ge_0_75": int((dry >= 0.75).sum()),
                    "seasonal_cycle_std": float(month_means.std(ddof=1)),
                    "seasonal_cycle_range": float(month_means.max() - month_means.min()),
                    "mean_pixels_per_month": float(sub["n_pixels"].mean()),
                }
            )
    out = pd.DataFrame(rows)
    split_order = {"train": 0, "validation": 1, "test": 2}
    out["split_order"] = out["split"].map(split_order)
    out = out.sort_values(["region", "split_order"]).drop(columns="split_order")
    return out.reset_index(drop=True)


def feature_group(feature: str) -> str:
    if feature.startswith("nino34"):
        return "enso"
    if feature.startswith("pdo"):
        return "pdo"
    if feature.startswith("month_"):
        return "seasonality"
    if feature.endswith("_nbr_mean"):
        return "spatial_context"
    if feature.startswith("spi"):
        return "spi_lags"
    if feature.startswith("pr_"):
        return "precip_lags"
    return "other"


def feature_gain_groups(summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for row in summary.itertuples(index=False):
        path = model_path(row.region, row.model)
        if not path.exists():
            print(f"Skipping feature gains for missing model: {path}")
            continue
        booster = xgb.Booster()
        booster.load_model(path.as_posix())
        gains = booster.get_score(importance_type="gain")
        if not gains:
            continue
        gain_df = pd.DataFrame(
            [{"feature": feature, "gain": gain, "group": feature_group(feature)} for feature, gain in gains.items()]
        )
        total = gain_df["gain"].sum()
        grouped = gain_df.groupby("group", as_index=False)["gain"].sum()
        for group_row in grouped.itertuples(index=False):
            rows.append(
                {
                    "region": row.region,
                    "region_name": row.region_name,
                    "model": row.model,
                    "feature_group": group_row.group,
                    "gain": float(group_row.gain),
                    "gain_share": float(group_row.gain / total) if total > 0 else float("nan"),
                    "n_features_with_gain": int((gain_df["group"] == group_row.group).sum()),
                }
            )
    return pd.DataFrame(rows).sort_values(["region", "model", "feature_group"]).reset_index(drop=True)


def best_model_by_region(summary: pd.DataFrame) -> pd.DataFrame:
    idx = summary.groupby("region")["selected_bss"].idxmax()
    return summary.loc[idx].sort_values("region").reset_index(drop=True)


def plot_bss(summary: pd.DataFrame, out_path: Path) -> None:
    plot_df = summary.sort_values(["region", "model"]).reset_index(drop=True)
    labels = [f"{r}\n{m}" for r, m in zip(plot_df["region"], plot_df["model"])]
    y = plot_df["selected_bss"].to_numpy()
    lo = plot_df["selected_bss_ci_low"].to_numpy()
    hi = plot_df["selected_bss_ci_high"].to_numpy()
    x = np.arange(len(plot_df))

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.errorbar(x, y, yerr=[y - lo, hi - y], fmt="o", capsize=4, color="#2f5d8c")
    ax.axhline(0, color="black", linewidth=1, linestyle="--")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("Selected calibrated BSS vs climatology")
    ax.set_title("Multi-region monthly dry-fraction skill")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_monthly_series(best: pd.DataFrame, out_path: Path) -> None:
    n = len(best)
    fig, axes = plt.subplots(n, 1, figsize=(12, 3.1 * n), sharex=True)
    if n == 1:
        axes = [axes]
    for ax, row in zip(axes, best.itertuples(index=False)):
        monthly = pd.read_csv(model_monthly_path(row.region, row.model), parse_dates=["target_time"])
        ax.plot(monthly["target_time"], monthly["y_true_dry_frac"], color="#202020", linewidth=1.6, label="Observed")
        ax.plot(monthly["target_time"], monthly["clim_prob_dry"], color="#888888", linewidth=1.2, label="Climatology")
        ax.plot(
            monthly["target_time"],
            monthly["xgb_selected_prob_dry"],
            color="#1f77b4",
            linewidth=1.4,
            label=f"Selected XGB ({row.model})",
        )
        ax.set_ylim(-0.03, 1.03)
        ax.set_ylabel("Dry fraction")
        ax.set_title(f"{row.region_name}: selected BSS={row.selected_bss:.3f}")
        ax.grid(alpha=0.2)
    axes[0].legend(loc="upper right", ncol=3, fontsize=9)
    axes[-1].set_xlabel("Target month")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_signal_vs_skill(summary: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    markers = {"tabular": "o", "spatial": "s"}
    for model, sub in summary.groupby("model"):
        ax.scatter(
            sub["pixel_dry_roc_auc_raw"],
            sub["selected_bss"],
            label=model,
            marker=markers.get(model, "o"),
            s=70,
        )
        for row in sub.itertuples(index=False):
            ax.annotate(row.region, (row.pixel_dry_roc_auc_raw, row.selected_bss), xytext=(4, 4), textcoords="offset points", fontsize=8)
    ax.axhline(0, color="black", linestyle="--", linewidth=1)
    ax.set_xlabel("Pixel-level dry ROC-AUC (raw)")
    ax.set_ylabel("Selected calibrated BSS")
    ax.set_title("Ranking signal does not guarantee probability skill")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_feature_groups(feature_groups: pd.DataFrame, best: pd.DataFrame, out_path: Path) -> None:
    best_keys = set(zip(best["region"], best["model"]))
    fg = feature_groups[
        feature_groups.apply(lambda row: (row["region"], row["model"]) in best_keys, axis=1)
    ].copy()
    if fg.empty:
        return
    pivot = fg.pivot_table(
        index=["region", "model"],
        columns="feature_group",
        values="gain_share",
        fill_value=0.0,
    )
    preferred = ["enso", "seasonality", "spi_lags", "precip_lags", "spatial_context", "pdo", "other"]
    cols = [c for c in preferred if c in pivot.columns] + [c for c in pivot.columns if c not in preferred]
    pivot = pivot[cols]
    labels = [f"{idx[0]}\n{idx[1]}" for idx in pivot.index]

    fig, ax = plt.subplots(figsize=(9, 5))
    bottom = np.zeros(len(pivot))
    colors = {
        "enso": "#4c78a8",
        "seasonality": "#f58518",
        "spi_lags": "#54a24b",
        "precip_lags": "#b279a2",
        "spatial_context": "#e45756",
        "pdo": "#72b7b2",
        "other": "#bab0ac",
    }
    for col in pivot.columns:
        vals = pivot[col].to_numpy()
        ax.bar(labels, vals, bottom=bottom, label=col, color=colors.get(col))
        bottom += vals
    ax.set_ylabel("Gain share")
    ax.set_ylim(0, 1)
    ax.set_title("Feature-group gain share for best model per region")
    ax.legend(loc="center left", bbox_to_anchor=(1, 0.5), fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def write_report(
    summary: pd.DataFrame,
    mechanism: pd.DataFrame,
    split_stats: pd.DataFrame,
    out_path: Path,
) -> None:
    best = best_model_by_region(summary)
    lines = [
        "Multi-Region Mechanism Analysis",
        "=" * 72,
        "Scope: completed full-resolution regions only; grid-stride smoke tests are excluded.",
        "Primary target: monthly regional dry-fraction probability for SPI-1[t+1].",
        "",
        "Headline by best selected BSS per region:",
    ]
    for row in best.itertuples(index=False):
        lines.append(
            f"- {row.region_name} ({row.model}): selected BSS={row.selected_bss:.3f} "
            f"[{row.selected_bss_ci_low:.3f}, {row.selected_bss_ci_high:.3f}], "
            f"raw ROC-AUC={row.pixel_dry_roc_auc_raw:.3f}"
        )

    lines.extend(["", "Mechanism diagnostics:"])
    for region, sub in mechanism.groupby("region", sort=False):
        best_row = sub.loc[sub["selected_bss"].idxmax()]
        split_sub = split_stats[(split_stats["region"] == region) & (split_stats["split"].isin(["train", "test"]))]
        shift = ""
        if set(split_sub["split"]) >= {"train", "test"}:
            train_mean = float(split_sub.loc[split_sub["split"] == "train", "mean_dry_frac"].iloc[0])
            test_mean = float(split_sub.loc[split_sub["split"] == "test", "mean_dry_frac"].iloc[0])
            shift = f"; train->test dry mean {train_mean:.3f}->{test_mean:.3f}"
        lines.append(
            f"- {best_row['region_name']}: selected-probability correlation="
            f"{best_row['corr_selected_with_observed']:.3f}, amplitude ratio="
            f"{best_row['selected_amplitude_ratio']:.3f}, selected bias="
            f"{best_row['selected_bias']:.3f}{shift}."
        )

    lines.extend(
        [
            "",
            "Scientific interpretation:",
            "- Central Valley has the strongest raw ranking signal, but selected probabilities still do not beat climatology robustly.",
            "- Southern Great Plains has weak raw ranking and a test-period seasonal climatology mismatch; both tabular and spatial XGB remain below climatology.",
            "- Mediterranean Spain has the best calibrated point estimate, but the confidence interval crosses zero; this is a hypothesis-generating regional hint, not a positive-skill claim.",
            "- ENSO and seasonality dominate gain in all regions, so model fit is mostly large-scale/seasonal. That does not guarantee calibrated probability skill.",
            "",
            "Recommended next actions:",
            "1. Add basin/land masks for rectangular regions before publication-level regional claims.",
            "2. Run one additional regime, preferably Murray-Darling or Horn of Africa, if compute budget allows.",
            "3. Add regional diagnostic plots to the paper narrative before adding more predictors.",
        ]
    )
    out_path.write_text("\n".join(lines) + "\n")


def copy_to_report(paths: list[Path]) -> None:
    REPORT_ROOT.mkdir(parents=True, exist_ok=True)
    for path in paths:
        if path.exists():
            shutil.copy2(path, REPORT_ROOT / path.name)


def main() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    REPORT_ROOT.mkdir(parents=True, exist_ok=True)

    summary = load_summary()
    diagnostics = monthly_diagnostics(summary)
    mechanism = mechanism_summary(summary, diagnostics)
    regions = sorted(summary["region"].unique())
    split_stats = split_dry_stats(regions)
    feature_groups = feature_gain_groups(summary)
    best = best_model_by_region(summary)

    outputs = {
        "summary": OUT_ROOT / "multiregion_summary_full_regions.csv",
        "diagnostics": OUT_ROOT / "regional_monthly_probability_diagnostics.csv",
        "mechanism": OUT_ROOT / "regional_mechanism_summary.csv",
        "split_stats": OUT_ROOT / "regional_split_dry_stats.csv",
        "feature_groups": OUT_ROOT / "regional_feature_gain_groups.csv",
        "report": OUT_ROOT / "regional_mechanism_report.txt",
        "bss_fig": OUT_ROOT / "multiregion_selected_bss_ci.png",
        "series_fig": OUT_ROOT / "multiregion_monthly_dry_fraction.png",
        "scatter_fig": OUT_ROOT / "multiregion_signal_vs_skill.png",
        "feature_fig": OUT_ROOT / "multiregion_feature_gain_groups.png",
    }

    summary.to_csv(outputs["summary"], index=False)
    diagnostics.to_csv(outputs["diagnostics"], index=False)
    mechanism.to_csv(outputs["mechanism"], index=False)
    split_stats.to_csv(outputs["split_stats"], index=False)
    feature_groups.to_csv(outputs["feature_groups"], index=False)

    plot_bss(summary, outputs["bss_fig"])
    plot_monthly_series(best, outputs["series_fig"])
    plot_signal_vs_skill(summary, outputs["scatter_fig"])
    plot_feature_groups(feature_groups, best, outputs["feature_fig"])
    write_report(summary, mechanism, split_stats, outputs["report"])

    report_paths = list(outputs.values())
    copy_to_report(report_paths)
    if FULL_SUMMARY_REPORT.exists():
        # Keep the legacy summary name focused on full-resolution regions.
        summary.to_csv(FULL_SUMMARY_REPORT, index=False)

    print("Wrote multi-region mechanism outputs:")
    for path in outputs.values():
        print(f"  {path}")
    print(f"Copied report-ready artifacts to {REPORT_ROOT}")
    print()
    print((outputs["report"]).read_text())


if __name__ == "__main__":
    main()
