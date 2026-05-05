#!/usr/bin/env python
"""Build paper-ready master result tables from current project outputs.

The project has many experiment-specific score files. This script creates a
single machine-readable table with consistent columns, plus a compact headline
table for manuscript drafting. It does not train models or change source data.
"""
from __future__ import annotations

from argparse import ArgumentParser, Namespace
from pathlib import Path
import re

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = PROJECT_ROOT / "outputs"
RESULTS = PROJECT_ROOT / "results"
REPORT_DIR = RESULTS / "report"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

N_BOOTSTRAP = 2000
MONTHLY_SEED = 42


def parse_args() -> Namespace:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--n-bootstrap", type=int, default=N_BOOTSTRAP)
    parser.add_argument("--out-dir", type=Path, default=REPORT_DIR)
    return parser.parse_args()


def brier(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean((np.asarray(p, dtype=float) - np.asarray(y, dtype=float)) ** 2))


def bss_from_arrays(y: np.ndarray, p: np.ndarray, ref: np.ndarray) -> float:
    ref_bs = brier(y, ref)
    return float(1.0 - brier(y, p) / ref_bs) if ref_bs > 0 else float("nan")


def bootstrap_bss(
    y: np.ndarray,
    p: np.ndarray,
    ref: np.ndarray,
    n_bootstrap: int,
    seed: int,
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    idx = np.arange(len(y))
    vals = np.empty(n_bootstrap, dtype=float)
    for i in range(n_bootstrap):
        sample = rng.choice(idx, size=len(idx), replace=True)
        vals[i] = bss_from_arrays(y[sample], p[sample], ref[sample])
    lo, hi = np.nanquantile(vals, [0.025, 0.975])
    return float(lo), float(hi)


def parse_ci(value: object) -> tuple[float, float]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return float("nan"), float("nan")
    text = str(value).strip()
    match = re.search(r"\[([^,\]]+),\s*([^\]]+)\]", text)
    if not match:
        return float("nan"), float("nan")
    try:
        return float(match.group(1)), float(match.group(2))
    except ValueError:
        return float("nan"), float("nan")


def read_canonical_ref_bs() -> float:
    path = OUTPUTS / "forecast_skill_bss_hss_table.csv"
    if not path.exists():
        return float("nan")
    df = pd.read_csv(path)
    row = df.loc[df["Forecaster"].str.contains("Climatological", na=False), "BS_dry"]
    return float(row.iloc[0]) if not row.empty else float("nan")


def claim_status(bss: float, ci_low: float, ci_high: float) -> str:
    if not np.isfinite(bss):
        return "not_applicable"
    if np.isfinite(ci_low) and np.isfinite(ci_high):
        if bss > 0 and ci_low > 0:
            return "robust_positive"
        if ci_high < 0:
            return "robust_negative"
        if bss > 0:
            return "positive_uncertain"
        return "not_distinguishable_from_climatology" if ci_high >= 0 else "negative_uncertain"
    if bss > 0:
        return "positive_no_ci"
    if bss < 0:
        return "negative_no_ci"
    return "reference"


def add_row(rows: list[dict[str, object]], **kwargs: object) -> None:
    base = {
        "category": "",
        "scope": "",
        "region": "",
        "target": "",
        "lead_months": np.nan,
        "model": "",
        "calibration": "",
        "n_test_months": np.nan,
        "bs_reference": np.nan,
        "bs_model": np.nan,
        "bss_vs_climatology": np.nan,
        "bss_ci_low": np.nan,
        "bss_ci_high": np.nan,
        "roc_auc_dry": np.nan,
        "hss": np.nan,
        "delta_bss": np.nan,
        "claim_status": "",
        "headline": False,
        "notes": "",
        "source_file": "",
    }
    base.update(kwargs)
    if not base["claim_status"]:
        base["claim_status"] = claim_status(
            float(base["bss_vs_climatology"])
            if pd.notna(base["bss_vs_climatology"]) else float("nan"),
            float(base["bss_ci_low"])
            if pd.notna(base["bss_ci_low"]) else float("nan"),
            float(base["bss_ci_high"])
            if pd.notna(base["bss_ci_high"]) else float("nan"),
        )
    rows.append(base)


def numeric_bss(value: object) -> float:
    text = str(value).replace("(ref)", "").strip()
    if text in {"-", "—", ""}:
        return 0.0
    return float(text)


def add_canonical_suite(rows: list[dict[str, object]]) -> None:
    path = OUTPUTS / "forecast_skill_bss_hss_table.csv"
    if not path.exists():
        return
    df = pd.read_csv(path)
    ref_bs = float(df.loc[df["Forecaster"].str.contains("Climatological"), "BS_dry"].iloc[0])
    for record in df.to_dict(orient="records"):
        ci_low, ci_high = parse_ci(record.get("BSS_dry_95CI"))
        auc = record.get("ROC-AUC_dry")
        hss = record.get("HSS")
        add_row(
            rows,
            category="central_valley_model_suite_raw",
            scope="canonical_spi1_lead1",
            region="California Central Valley",
            target="SPI-1 dry fraction",
            lead_months=1,
            model=record["Forecaster"],
            calibration="raw_or_model_default",
            n_test_months=63,
            bs_reference=ref_bs,
            bs_model=float(record["BS_dry"]),
            bss_vs_climatology=numeric_bss(record["BSS_dry"]),
            bss_ci_low=ci_low,
            bss_ci_high=ci_high,
            roc_auc_dry=float(auc) if str(auc) not in {"—", "nan"} else np.nan,
            hss=float(hss) if str(hss) not in {"—", "nan"} else np.nan,
            headline=record["Forecaster"] in {
                "Climatological baseline",
                "XGBoost-Spatial",
                "Random Forest",
                "ConvLSTM",
            },
            notes="Monthly-level score; pixel metrics are secondary.",
            source_file=str(path.relative_to(PROJECT_ROOT)),
        )


def add_calibration_study(rows: list[dict[str, object]]) -> None:
    path = OUTPUTS / "calib_study_results.csv"
    if not path.exists():
        return
    df = pd.read_csv(path)
    ref_bs = read_canonical_ref_bs()
    for record in df.to_dict(orient="records"):
        ci_low, ci_high = parse_ci(record.get("test_BSS_95CI"))
        add_row(
            rows,
            category="central_valley_calibrated_checkpoint",
            scope="canonical_spi1_lead1",
            region="California Central Valley",
            target="SPI-1 dry fraction",
            lead_months=1,
            model=record["model"],
            calibration=str(record["best_calibration"]),
            n_test_months=63,
            bs_reference=ref_bs,
            bs_model=float(record["test_BS"]),
            bss_vs_climatology=float(record["test_BSS"]),
            bss_ci_low=ci_low,
            bss_ci_high=ci_high,
            headline=True,
            notes=(
                "Validation-selected post-hoc calibration; this is the primary "
                "Central Valley probability-skill checkpoint."
            ),
            source_file=str(path.relative_to(PROJECT_ROOT)),
        )


def monthly_forecast_rows(
    path: Path,
    pred_cols: dict[str, str],
    category: str,
    scope: str,
    region: str,
    target: str,
    lead_months: int,
    model_prefix: str,
    rows: list[dict[str, object]],
    n_bootstrap: int,
    headline_methods: set[str],
    note: str = "",
) -> None:
    if not path.exists():
        return
    df = pd.read_csv(path)
    y = df["y_true_dry_frac"].to_numpy(dtype=float)
    ref = df["clim_prob_dry"].to_numpy(dtype=float)
    bs_ref = brier(y, ref)
    for method, col in pred_cols.items():
        if col not in df.columns:
            continue
        p = df[col].to_numpy(dtype=float)
        ci_low, ci_high = bootstrap_bss(
            y, p, ref, n_bootstrap=n_bootstrap, seed=MONTHLY_SEED + len(rows)
        )
        add_row(
            rows,
            category=category,
            scope=scope,
            region=region,
            target=target,
            lead_months=lead_months,
            model=f"{model_prefix} {method}".strip(),
            calibration=method,
            n_test_months=int(len(df)),
            bs_reference=bs_ref,
            bs_model=brier(y, p),
            bss_vs_climatology=bss_from_arrays(y, p, ref),
            bss_ci_low=ci_low,
            bss_ci_high=ci_high,
            headline=method in headline_methods,
            notes=note,
            source_file=str(path.relative_to(PROJECT_ROOT)),
        )


def add_feature_extensions(rows: list[dict[str, object]], n_bootstrap: int) -> None:
    experiments = [
        (
            OUTPUTS / "met_feature_xgb_monthly_scores.csv",
            "central_valley_feature_extension",
            "XGBoost + regional temperature/VPD",
            "ERA5-Land regional t2m/VPD anomaly lags.",
        ),
        (
            OUTPUTS / "met_spatial_xgb_monthly_scores.csv",
            "central_valley_feature_extension",
            "XGB-Spatial + gridded temperature/VPD",
            "ERA5-Land gridded t2m/VPD anomaly lags interpolated to CHIRPS grid.",
        ),
        (
            OUTPUTS / "soil_moisture_xgb_monthly_scores.csv",
            "central_valley_feature_extension",
            "XGBoost + soil moisture",
            "ERA5-Land soil-water/root-zone anomaly lags.",
        ),
        (
            OUTPUTS / "atmos_feature_xgb_monthly_scores.csv",
            "central_valley_feature_extension",
            "XGBoost + MJO/IVT",
            "MJO RMM monthly features plus optional ERA5 IVT anomalies; shorter test period.",
        ),
    ]
    pred_cols = {
        "raw": "xgb_raw_prob_dry",
        "isotonic": "xgb_isotonic_prob_dry",
        "platt": "xgb_platt_prob_dry",
        "selected": "xgb_selected_prob_dry",
    }
    for path, category, prefix, note in experiments:
        monthly_forecast_rows(
            path=path,
            pred_cols=pred_cols,
            category=category,
            scope="canonical_spi1_lead1_feature_extension",
            region="California Central Valley",
            target="SPI-1 dry fraction",
            lead_months=1,
            model_prefix=prefix,
            rows=rows,
            n_bootstrap=n_bootstrap,
            headline_methods={"selected"},
            note=note,
        )


def add_edl(rows: list[dict[str, object]], n_bootstrap: int) -> None:
    path = OUTPUTS / "edl_uncertainty_monthly.csv"
    if not path.exists():
        return
    df = pd.read_csv(path)
    abs_err = (df["edl_selected_prob_dry"] - df["y_true_dry_frac"]).abs()
    total_corr = df["total_u"].corr(abs_err, method="spearman")
    epistemic_corr = df["epistemic_u"].corr(abs_err, method="spearman")
    pred_cols = {
        "raw": "edl_raw_prob_dry",
        "isotonic": "edl_isotonic_prob_dry",
        "platt": "edl_platt_prob_dry",
        "selected": "edl_selected_prob_dry",
    }
    monthly_forecast_rows(
        path=path,
        pred_cols=pred_cols,
        category="central_valley_uncertainty",
        scope="canonical_spi1_lead1_edl",
        region="California Central Valley",
        target="SPI-1 dry fraction",
        lead_months=1,
        model_prefix="EDL MLP",
        rows=rows,
        n_bootstrap=n_bootstrap,
        headline_methods={"selected"},
        note=(
            "Dirichlet EDL uncertainty; selected monthly uncertainty Spearman "
            f"corr(abs error): total={total_corr:.3f}, epistemic={epistemic_corr:.3f}."
        ),
    )


def add_seasonal(rows: list[dict[str, object]], n_bootstrap: int) -> None:
    specs = [
        (OUTPUTS / "seasonal_spi3_lead3_monthly_scores.csv", 3, 3),
        (OUTPUTS / "seasonal_spi3_lead6_monthly_scores.csv", 3, 6),
        (OUTPUTS / "seasonal_spi6_lead6_monthly_scores.csv", 6, 6),
    ]
    for path, spi, lead in specs:
        pred_cols = {
            f"persistence_spi{spi}": "persistence_prob_dry",
            "xgb_raw": "xgb_prob_dry",
            "xgb_isotonic": "xgb_cal_prob_dry",
        }
        monthly_forecast_rows(
            path=path,
            pred_cols=pred_cols,
            category="central_valley_seasonal_target",
            scope=f"seasonal_spi{spi}_lead{lead}",
            region="California Central Valley",
            target=f"SPI-{spi} dry fraction",
            lead_months=lead,
            model_prefix=f"Seasonal SPI-{spi} lead-{lead}",
            rows=rows,
            n_bootstrap=n_bootstrap,
            headline_methods={"xgb_isotonic"},
            note="Leakage-safe seasonal target with lead >= SPI accumulation window.",
        )


def add_operational_benchmark(rows: list[dict[str, object]], n_bootstrap: int) -> None:
    path = OUTPUTS / "operational_precip_benchmark_monthly_scores.csv"
    if not path.exists():
        path = REPORT_DIR / "operational_precip_benchmark_monthly_scores.csv"
    if not path.exists():
        return
    pred_cols = {
        "raw": "operational_raw_prob_dry",
        "isotonic": "operational_isotonic_prob_dry",
        "selected": "operational_selected_prob_dry",
    }
    monthly_forecast_rows(
        path=path,
        pred_cols=pred_cols,
        category="operational_dynamical_benchmark",
        scope="canonical_spi1_lead1_external_precip",
        region="California Central Valley",
        target="SPI-1 dry fraction",
        lead_months=1,
        model_prefix="External precipitation forecast",
        rows=rows,
        n_bootstrap=n_bootstrap,
        headline_methods={"selected"},
        note=(
            "External operational/dynamical precipitation forecast mapped to "
            "monthly dry-fraction probability using validation-only calibration."
        ),
    )


def add_multiregion(rows: list[dict[str, object]]) -> None:
    path = RESULTS / "multiregion" / "multiregion_summary.csv"
    if not path.exists():
        path = OUTPUTS / "multiregion" / "multiregion_summary.csv"
    if not path.exists():
        return
    df = pd.read_csv(path)
    df = df[~df["region"].astype(str).str.contains("_stride", regex=False)].copy()
    best_idx = df.groupby("region")["selected_bss"].idxmax()
    headline_keys = set(zip(df.loc[best_idx, "region"], df.loc[best_idx, "model"]))
    for record in df.to_dict(orient="records"):
        add_row(
            rows,
            category="multi_region_selected_checkpoint",
            scope="multi_region_spi1_lead1",
            region=str(record["region_name"]),
            target="SPI-1 dry fraction",
            lead_months=1,
            model=f"{record['model']} XGBoost",
            calibration=str(record["selected_calibration"]),
            n_test_months=int(record["test_months"]),
            bs_reference=float(record["climatology_bs"]),
            bs_model=float(record["selected_bs"]),
            bss_vs_climatology=float(record["selected_bss"]),
            bss_ci_low=float(record["selected_bss_ci_low"]),
            bss_ci_high=float(record["selected_bss_ci_high"]),
            roc_auc_dry=float(record["pixel_dry_roc_auc_raw"]),
            headline=(record["region"], record["model"]) in headline_keys,
            notes=f"Region slug={record['region']}; raw BSS={float(record['raw_bss']):.3f}.",
            source_file=str(path.relative_to(PROJECT_ROOT)),
        )


def add_ablation(rows: list[dict[str, object]]) -> None:
    path = OUTPUTS / "feature_ablation_results.csv"
    if not path.exists():
        return
    df = pd.read_csv(path)
    for record in df.to_dict(orient="records"):
        features = record.get("ablated_features", "")
        if pd.isna(features):
            features = ""
        add_row(
            rows,
            category="feature_ablation",
            scope="canonical_spi1_lead1_xgb_raw",
            region="California Central Valley",
            target="SPI-1 dry fraction",
            lead_months=1,
            model=f"XGBoost ablation: {record['group']}",
            calibration="raw",
            n_test_months=63,
            bss_vs_climatology=float(record["bss"]),
            delta_bss=float(record["delta_bss"]),
            headline=record["group"] in {"all_features", "enso", "seasonality", "pr_lags"},
            notes=f"Ablated features: {features}",
            source_file=str(path.relative_to(PROJECT_ROOT)),
        )


def markdown_table(df: pd.DataFrame, cols: list[str]) -> str:
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join(["---"] * len(cols)) + " |"
    lines = [header, sep]
    for _, row in df.iterrows():
        vals = []
        for col in cols:
            value = row[col]
            if isinstance(value, float):
                if np.isnan(value):
                    vals.append("")
                elif col in {"bss_vs_climatology", "bss_ci_low", "bss_ci_high", "bs_model", "bs_reference"}:
                    vals.append(f"{value:.3f}")
                else:
                    vals.append(f"{value:.3f}")
            else:
                vals.append(str(value))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines) + "\n"


def write_outputs(rows: list[dict[str, object]], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    sort_cols = ["category", "scope", "region", "model"]
    df = df.sort_values(sort_cols).reset_index(drop=True)
    full_path = out_dir / "master_results_table.csv"
    df.to_csv(full_path, index=False)

    headline = df[df["headline"]].copy()
    headline = headline.sort_values(
        ["category", "scope", "region", "bss_vs_climatology"],
        ascending=[True, True, True, False],
    ).reset_index(drop=True)
    headline_path = out_dir / "master_results_headline.csv"
    headline.to_csv(headline_path, index=False)

    md_cols = [
        "category",
        "region",
        "target",
        "lead_months",
        "model",
        "calibration",
        "n_test_months",
        "bs_model",
        "bss_vs_climatology",
        "bss_ci_low",
        "bss_ci_high",
        "claim_status",
    ]
    md = [
        "# Master Results Headline Table",
        "",
        "Generated from current project outputs by `scripts/build_master_results_table.py`.",
        "BSS is monthly dry-fraction Brier Skill Score against calendar-month climatology.",
        "",
        markdown_table(headline[md_cols], md_cols),
    ]
    md_path = out_dir / "master_results_headline.md"
    md_path.write_text("\n".join(md), encoding="utf-8")

    print(f"Wrote {full_path} rows={len(df):,}")
    print(f"Wrote {headline_path} rows={len(headline):,}")
    print(f"Wrote {md_path}")


def main() -> None:
    args = parse_args()
    rows: list[dict[str, object]] = []
    add_canonical_suite(rows)
    add_calibration_study(rows)
    add_feature_extensions(rows, args.n_bootstrap)
    add_edl(rows, args.n_bootstrap)
    add_seasonal(rows, args.n_bootstrap)
    add_operational_benchmark(rows, args.n_bootstrap)
    add_multiregion(rows)
    add_ablation(rows)
    write_outputs(rows, args.out_dir)


if __name__ == "__main__":
    main()
