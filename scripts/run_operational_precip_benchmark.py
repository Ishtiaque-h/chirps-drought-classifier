#!/usr/bin/env python
"""Evaluate external dynamical precipitation forecasts against the SPI target.

This is an adapter for SubX, NMME, GEFS, ECMWF/SEAS5, or other operational
forecast systems after their precipitation forecasts have been preprocessed to a
monthly regional CSV. It deliberately keeps data access separate from scoring:
different archives use different calendars, units, ensemble dimensions, and
lead definitions, but the verification protocol here stays fixed.

Required forecast CSV columns:
  target_time
  and one of:
    forecast_prob_dry  - direct dry-event probability in [0, 1]
    forecast_pr_anom   - forecast precipitation anomaly; lower means drier
    forecast_pr        - forecast precipitation amount; lower means drier

Outputs:
  outputs/operational_precip_benchmark_monthly_scores.csv
  outputs/operational_precip_benchmark_scores.txt
"""
from __future__ import annotations

from argparse import ArgumentParser, Namespace
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET = PROJECT_ROOT / "data" / "processed" / "dataset_forecast.parquet"
OUT_DIR = PROJECT_ROOT / "outputs"
REPORT_DIR = PROJECT_ROOT / "results" / "report"
OUT_DIR.mkdir(exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

N_BOOTSTRAP = 2000


def parse_args() -> Namespace:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument(
        "--forecast-csv",
        type=Path,
        default=None,
        help="Monthly external forecast CSV. See --write-template for required columns.",
    )
    parser.add_argument("--dataset", type=Path, default=DATASET)
    parser.add_argument("--n-bootstrap", type=int, default=N_BOOTSTRAP)
    parser.add_argument("--copy-report", action="store_true")
    parser.add_argument(
        "--write-template",
        action="store_true",
        help="Write a forecast CSV template and exit.",
    )
    return parser.parse_args()


def write_template(path: Path) -> None:
    template = pd.DataFrame(
        {
            "target_time": pd.date_range("2017-01-01", periods=3, freq="MS"),
            "forecast_prob_dry": [np.nan, np.nan, np.nan],
            "forecast_pr_anom": [np.nan, np.nan, np.nan],
            "forecast_pr": [np.nan, np.nan, np.nan],
            "source_model": ["NMME_or_SubX_model_name"] * 3,
            "init_time": [pd.NaT, pd.NaT, pd.NaT],
            "lead_months": [1, 1, 1],
            "notes": [
                "Fill exactly one of forecast_prob_dry, forecast_pr_anom, or forecast_pr.",
                "Use target_time for the month being predicted.",
                "Aggregate ensembles/lead windows before running this scorer.",
            ],
        }
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    template.to_csv(path, index=False)
    print(f"Wrote template: {path}")


def brier(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean((np.asarray(p, dtype=float) - np.asarray(y, dtype=float)) ** 2))


def bss(y: np.ndarray, p: np.ndarray, ref: np.ndarray) -> float:
    ref_bs = brier(y, ref)
    return float(1.0 - brier(y, p) / ref_bs) if ref_bs > 0 else float("nan")


def bootstrap_bss(monthly: pd.DataFrame, pred_col: str, n_bootstrap: int, seed: int = 42) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    idx = np.arange(len(monthly))
    y = monthly["y_true_dry_frac"].to_numpy(dtype=float)
    ref = monthly["clim_prob_dry"].to_numpy(dtype=float)
    pred = monthly[pred_col].to_numpy(dtype=float)
    vals = np.empty(n_bootstrap, dtype=float)
    for i in range(n_bootstrap):
        sample = rng.choice(idx, size=len(idx), replace=True)
        vals[i] = bss(y[sample], pred[sample], ref[sample])
    lo, hi = np.nanquantile(vals, [0.025, 0.975])
    return float(lo), float(hi)


def load_observed_monthly(dataset: Path) -> pd.DataFrame:
    if not dataset.exists():
        raise FileNotFoundError(f"Forecast dataset not found: {dataset}")
    df = pd.read_parquet(dataset, columns=["time", "target_label"])
    df["target_time"] = (
        pd.to_datetime(df["time"]) + pd.DateOffset(months=1)
    ).dt.to_period("M").dt.to_timestamp()
    df["target_year"] = df["target_time"].dt.year
    df["target_month"] = df["target_time"].dt.month
    df["is_dry"] = (df["target_label"] == -1).astype(float)

    monthly = (
        df.groupby(["target_time", "target_year", "target_month"], observed=True)
        .agg(y_true_dry_frac=("is_dry", "mean"), n_pixels=("is_dry", "size"))
        .reset_index()
        .sort_values("target_time")
    )

    train = monthly[monthly["target_year"] <= 2016].copy()
    if train.empty:
        raise ValueError("No training months available to build climatology.")
    month_clim = train.groupby("target_month")["y_true_dry_frac"].mean()
    global_clim = float(train["y_true_dry_frac"].mean())
    monthly["clim_prob_dry"] = monthly["target_month"].map(month_clim).fillna(global_clim)
    return monthly


def load_forecast(path: Path) -> tuple[pd.DataFrame, str, str]:
    if path is None:
        raise ValueError("Pass --forecast-csv, or use --write-template to create a template.")
    if not path.exists():
        raise FileNotFoundError(f"Forecast CSV not found: {path}")

    fcst = pd.read_csv(path)
    if "target_time" not in fcst.columns:
        raise ValueError(f"{path} must contain target_time")
    fcst["target_time"] = pd.to_datetime(fcst["target_time"]).dt.to_period("M").dt.to_timestamp()
    fcst = fcst.sort_values("target_time").drop_duplicates("target_time", keep="last")

    if "forecast_prob_dry" in fcst.columns and fcst["forecast_prob_dry"].notna().any():
        fcst["forecast_prob_dry"] = fcst["forecast_prob_dry"].clip(0.0, 1.0)
        return fcst, "forecast_prob_dry", "probability"

    if "forecast_pr_anom" in fcst.columns and fcst["forecast_pr_anom"].notna().any():
        fcst["dry_signal"] = -fcst["forecast_pr_anom"].astype(float)
        return fcst, "dry_signal", "precip_anomaly_signal"

    if "forecast_pr" in fcst.columns and fcst["forecast_pr"].notna().any():
        fcst["dry_signal"] = -fcst["forecast_pr"].astype(float)
        return fcst, "dry_signal", "precip_amount_signal"

    raise ValueError(
        f"{path} must provide forecast_prob_dry, forecast_pr_anom, or forecast_pr with non-missing values."
    )


def fit_candidates(
    merged: pd.DataFrame,
    predictor_col: str,
    predictor_kind: str,
) -> tuple[pd.DataFrame, str, dict[str, float], dict[str, object]]:
    val = merged[(merged["target_year"] >= 2017) & (merged["target_year"] <= 2020)].copy()
    test = merged[merged["target_year"] >= 2021].copy()
    if val.empty:
        raise ValueError("No validation overlap with forecast CSV for 2017-2020 calibration.")
    if test.empty:
        raise ValueError("No test overlap with forecast CSV for 2021+ evaluation.")
    coverage = {
        "validation_months": int(val["target_time"].nunique()),
        "validation_start": val["target_time"].min(),
        "validation_end": val["target_time"].max(),
        "test_months": int(test["target_time"].nunique()),
        "test_start": test["target_time"].min(),
        "test_end": test["target_time"].max(),
        "predictor_kind": predictor_kind,
    }

    candidates: dict[str, pd.Series] = {}
    if predictor_kind == "probability":
        candidates["raw"] = test[predictor_col].clip(0.0, 1.0)

    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    iso.fit(val[predictor_col].to_numpy(dtype=float), val["y_true_dry_frac"].to_numpy(dtype=float))
    candidates["isotonic"] = pd.Series(
        iso.predict(test[predictor_col].to_numpy(dtype=float)),
        index=test.index,
    ).clip(0.0, 1.0)

    val_bs: dict[str, float] = {}
    if predictor_kind == "probability":
        val_bs["raw"] = brier(
            val["y_true_dry_frac"].to_numpy(dtype=float),
            val[predictor_col].clip(0.0, 1.0).to_numpy(dtype=float),
        )
    val_iso = iso.predict(val[predictor_col].to_numpy(dtype=float))
    val_bs["isotonic"] = brier(val["y_true_dry_frac"].to_numpy(dtype=float), val_iso)
    best_method = min(val_bs, key=val_bs.get)

    for method, pred in candidates.items():
        test[f"operational_{method}_prob_dry"] = pred
    test["operational_selected_prob_dry"] = test[f"operational_{best_method}_prob_dry"]
    return test, best_method, val_bs, coverage


def write_scores(
    monthly: pd.DataFrame,
    best_method: str,
    val_bs: dict[str, float],
    coverage: dict[str, object],
    n_bootstrap: int,
    source: Path,
) -> None:
    pred_cols = [c for c in monthly.columns if c.startswith("operational_") and c.endswith("_prob_dry")]
    rows = []
    y = monthly["y_true_dry_frac"].to_numpy(dtype=float)
    ref = monthly["clim_prob_dry"].to_numpy(dtype=float)
    bs_ref = brier(y, ref)
    selected_corr = monthly["operational_selected_prob_dry"].corr(
        monthly["y_true_dry_frac"], method="spearman"
    )
    signal_corr = (
        monthly["dry_signal"].corr(monthly["y_true_dry_frac"], method="spearman")
        if "dry_signal" in monthly.columns
        else np.nan
    )
    y_std = float(monthly["y_true_dry_frac"].std(ddof=0))
    pred_std = float(monthly["operational_selected_prob_dry"].std(ddof=0))
    amplitude_ratio = pred_std / y_std if y_std > 0 else np.nan
    for col in pred_cols:
        ci = bootstrap_bss(monthly, col, n_bootstrap=n_bootstrap, seed=101 + len(rows))
        rows.append(
            {
                "forecast": col.replace("operational_", "").replace("_prob_dry", ""),
                "bs": brier(y, monthly[col].to_numpy(dtype=float)),
                "bss": bss(y, monthly[col].to_numpy(dtype=float), ref),
                "bss_ci_low": ci[0],
                "bss_ci_high": ci[1],
            }
        )

    monthly_path = OUT_DIR / "operational_precip_benchmark_monthly_scores.csv"
    score_path = OUT_DIR / "operational_precip_benchmark_scores.txt"
    monthly.to_csv(monthly_path, index=False)

    lines = [
        "Operational/Dynamical Precipitation Benchmark",
        "=" * 64,
        f"Forecast source CSV: {source}",
        "Design: external monthly precipitation forecast mapped to dry-fraction probability.",
        "Calibration: validation-only mapping from forecast signal/probability to observed monthly dry fraction.",
        f"Predictor kind: {coverage['predictor_kind']}",
        (
            "Validation months: "
            f"{coverage['validation_months']} "
            f"({pd.Timestamp(coverage['validation_start']):%Y-%m} to "
            f"{pd.Timestamp(coverage['validation_end']):%Y-%m})"
        ),
        f"Validation BS by method: {val_bs}",
        f"Selected method: {best_method}",
        (
            "Test months: "
            f"{coverage['test_months']} "
            f"({pd.Timestamp(coverage['test_start']):%Y-%m} to "
            f"{pd.Timestamp(coverage['test_end']):%Y-%m})"
        ),
        f"Climatology BS: {bs_ref:.5f}",
        f"Spearman corr(selected prob, observed dry fraction): {selected_corr:.3f}",
        f"Spearman corr(raw dry signal, observed dry fraction): {signal_corr:.3f}",
        f"Selected probability amplitude ratio: {amplitude_ratio:.3f}",
        "",
        "Monthly dry-fraction Brier Skill Score vs climatology:",
    ]
    for row in rows:
        lines.append(
            f"  {row['forecast']:<12} BS={row['bs']:.5f} "
            f"BSS={row['bss']:.5f} "
            f"95% CI [{row['bss_ci_low']:.5f}, {row['bss_ci_high']:.5f}]"
        )
    lines.extend(["", f"Monthly scores: {monthly_path}"])
    score_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


def main() -> None:
    args = parse_args()
    if args.write_template:
        write_template(OUT_DIR / "operational_precip_benchmark_template.csv")
        return

    observed = load_observed_monthly(args.dataset)
    forecast, predictor_col, predictor_kind = load_forecast(args.forecast_csv)
    merged = observed.merge(forecast, on="target_time", how="inner")
    merged = merged.dropna(subset=[predictor_col, "y_true_dry_frac", "clim_prob_dry"]).copy()
    if merged.empty:
        raise ValueError("Forecast CSV has no overlapping target_time rows with the observed dataset.")

    test_monthly, best_method, val_bs, coverage = fit_candidates(
        merged, predictor_col, predictor_kind
    )
    write_scores(test_monthly, best_method, val_bs, coverage, args.n_bootstrap, args.forecast_csv)

    if args.copy_report:
        for name in [
            "operational_precip_benchmark_monthly_scores.csv",
            "operational_precip_benchmark_scores.txt",
        ]:
            shutil.copy2(OUT_DIR / name, REPORT_DIR / name)


if __name__ == "__main__":
    main()
