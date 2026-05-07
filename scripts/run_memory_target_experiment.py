#!/usr/bin/env python
"""Run a memory-target drought forecast experiment.

This experiment tests the literature-motivated hypothesis that forecast skill
should improve when the drought target has physical memory. The default target
is Central Valley SPI-6 dry fraction at lead 6, which is leakage-safe because
features at month t predict the SPI-6 accumulation ending at t+6.

The script compares:
  - lag_climate: CHIRPS/SPI lags + month + optional climate-index lags
  - soil_memory: lag_climate + ERA5-Land soil-water/root-zone anomaly lags

CPC NMME real-time forecasts are audited as external operational benchmarks.
They are not used as ML training features by default because the real-time
archive available in this project starts after the 1991-2016 training period.
"""
from __future__ import annotations

from argparse import ArgumentParser, Namespace
from pathlib import Path
import shutil

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_sample_weight

from feature_config import BASE_FEATURES


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED = PROJECT_ROOT / "data" / "processed"
OUT_DIR = PROJECT_ROOT / "outputs"
REPORT_DIR = PROJECT_ROOT / "results" / "report"
OUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

LABEL_MAP = {-1: 0, 0: 1, 1: 2}
INV_LABEL_MAP = {v: k for k, v in LABEL_MAP.items()}

SOIL_FEATURES = [
    "swvl1_anom_lag1",
    "swvl1_anom_lag2",
    "swvl2_anom_lag1",
    "swvl2_anom_lag2",
    "swvl3_anom_lag1",
    "swvl3_anom_lag2",
    "rootzone_sm_anom_lag1",
    "rootzone_sm_anom_lag2",
]

CLIMATE_FEATURES_NINO34 = ["nino34_lag1", "nino34_lag2"]
CLIMATE_FEATURES_PDO = ["pdo_lag1", "pdo_lag2"]


def parse_args() -> Namespace:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--target-spi", type=int, choices=[3, 6], default=6)
    parser.add_argument("--lead-months", type=int, default=6)
    parser.add_argument(
        "--seasonal-dataset",
        type=Path,
        default=None,
        help="Defaults to data/processed/dataset_seasonal_spi<target>_lead<lead>.parquet.",
    )
    parser.add_argument(
        "--soil-dataset",
        type=Path,
        default=PROCESSED / "dataset_forecast_soil_moisture.parquet",
    )
    parser.add_argument(
        "--nmme-prob-csv",
        type=Path,
        default=REPORT_DIR / "nmme_cpc_prob_cvalley_lead6_forecast.csv",
    )
    parser.add_argument(
        "--nmme-anom-csv",
        type=Path,
        default=REPORT_DIR / "nmme_cpc_cvalley_lead6_forecast.csv",
    )
    parser.add_argument(
        "--nmme-prob-monthly",
        type=Path,
        default=REPORT_DIR / "operational_nmme_cpc_prob_spi6_lead6_monthly_scores.csv",
    )
    parser.add_argument(
        "--nmme-anom-monthly",
        type=Path,
        default=REPORT_DIR / "operational_nmme_cpc_spi6_lead6_monthly_scores.csv",
    )
    parser.add_argument("--include-pdo", action="store_true")
    parser.add_argument("--n-bootstrap", type=int, default=2000)
    parser.add_argument("--output-prefix", default=None)
    parser.add_argument("--copy-report", action="store_true")
    return parser.parse_args()


def brier(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean((np.asarray(p, dtype=float) - np.asarray(y, dtype=float)) ** 2))


def bss(y: np.ndarray, p: np.ndarray, ref: np.ndarray) -> float:
    ref_bs = brier(y, ref)
    return float(1.0 - brier(y, p) / ref_bs) if ref_bs > 0 else float("nan")


def bootstrap_bss(
    monthly: pd.DataFrame,
    pred_col: str,
    ref_col: str,
    n_bootstrap: int,
    seed: int,
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    idx = np.arange(len(monthly))
    y = monthly["y_true_dry_frac"].to_numpy(dtype=float)
    p = monthly[pred_col].to_numpy(dtype=float)
    ref = monthly[ref_col].to_numpy(dtype=float)
    vals = np.empty(n_bootstrap, dtype=float)
    for i in range(n_bootstrap):
        sample = rng.choice(idx, size=len(idx), replace=True)
        vals[i] = bss(y[sample], p[sample], ref[sample])
    lo, hi = np.nanquantile(vals, [0.025, 0.975])
    return float(lo), float(hi)


def amplitude_ratio(y: pd.Series, p: pd.Series) -> float:
    y_std = float(y.std(ddof=0))
    p_std = float(p.std(ddof=0))
    return p_std / y_std if y_std > 0 else float("nan")


def default_seasonal_dataset(target_spi: int, lead_months: int) -> Path:
    return PROCESSED / f"dataset_seasonal_spi{target_spi}_lead{lead_months}.parquet"


def load_memory_dataset(args: Namespace) -> pd.DataFrame:
    seasonal_path = args.seasonal_dataset or default_seasonal_dataset(args.target_spi, args.lead_months)
    if not seasonal_path.exists():
        raise FileNotFoundError(f"Seasonal dataset not found: {seasonal_path}")
    if not args.soil_dataset.exists():
        raise FileNotFoundError(f"Soil-moisture dataset not found: {args.soil_dataset}")

    print(f"Loading seasonal target dataset: {seasonal_path}")
    df = pd.read_parquet(seasonal_path)
    df["time"] = pd.to_datetime(df["time"]).dt.to_period("M").dt.to_timestamp()
    df["target_time"] = pd.to_datetime(df["target_time"]).dt.to_period("M").dt.to_timestamp()

    print(f"Loading soil-memory features: {args.soil_dataset}")
    soil_cols = ["time"] + SOIL_FEATURES
    soil = pd.read_parquet(args.soil_dataset, columns=soil_cols)
    soil["time"] = pd.to_datetime(soil["time"]).dt.to_period("M").dt.to_timestamp()
    soil = soil.drop_duplicates("time", keep="last")

    merged = df.merge(soil, on="time", how="left")
    missing = int(merged[SOIL_FEATURES].isna().any(axis=1).sum())
    if missing:
        print(f"Rows with missing soil-memory features before drop: {missing:,}")
        merged = merged.dropna(subset=SOIL_FEATURES).copy()

    target_col = f"target_label_spi{args.target_spi}"
    if target_col not in merged.columns:
        raise ValueError(f"Missing target column: {target_col}")
    if args.lead_months < args.target_spi:
        raise ValueError("Use lead-months >= target-spi for a leakage-safe memory target.")
    merged["year"] = pd.to_datetime(merged["target_time"]).dt.year.astype(int)
    merged["month"] = pd.to_datetime(merged["target_time"]).dt.month.astype(int)
    return merged


def active_climate_features(columns: list[str], include_pdo: bool) -> list[str]:
    features = [c for c in CLIMATE_FEATURES_NINO34 if c in columns]
    if include_pdo:
        features.extend([c for c in CLIMATE_FEATURES_PDO if c in columns])
    return features


def monthly_bs(frame: pd.DataFrame, prob_col: str) -> float:
    monthly = (
        frame.groupby("target_time", observed=True)
        .agg(y_true_dry_frac=("is_dry", "mean"), pred_prob_dry=(prob_col, "mean"))
        .reset_index()
    )
    return brier(
        monthly["y_true_dry_frac"].to_numpy(dtype=float),
        monthly["pred_prob_dry"].to_numpy(dtype=float),
    )


def add_climatology(train: pd.DataFrame, target_col: str, frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    train_dry = train.assign(is_dry=(train[target_col] == -1).astype(float))
    month_clim = train_dry.groupby("month")["is_dry"].mean()
    global_clim = float(train_dry["is_dry"].mean())
    out["clim_prob_dry"] = out["month"].map(month_clim).fillna(global_clim)
    return out


def train_variant(
    df: pd.DataFrame,
    target_spi: int,
    lead_months: int,
    variant: str,
    features: list[str],
    n_bootstrap: int,
    output_prefix: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    target_col = f"target_label_spi{target_spi}"
    persistence_feature = f"spi{target_spi}_lag1"
    if persistence_feature not in df.columns:
        raise ValueError(f"Missing target-consistent persistence feature: {persistence_feature}")

    train = df[df["year"] <= 2016].copy()
    val = df[(df["year"] >= 2017) & (df["year"] <= 2020)].copy()
    test = df[df["year"] >= 2021].copy()
    if train.empty or val.empty or test.empty:
        raise ValueError(f"Bad split for {variant}: train={train.shape} val={val.shape} test={test.shape}")

    train = add_climatology(train, target_col, train)
    val = add_climatology(train, target_col, val)
    test = add_climatology(train, target_col, test)

    y_train_enc = train[target_col].map(LABEL_MAP).to_numpy()
    y_val_enc = val[target_col].map(LABEL_MAP).to_numpy()
    y_test_enc = test[target_col].map(LABEL_MAP).to_numpy()

    dtrain = xgb.DMatrix(
        train[features],
        label=y_train_enc,
        weight=compute_sample_weight(class_weight="balanced", y=y_train_enc),
        feature_names=features,
    )
    dval = xgb.DMatrix(val[features], label=y_val_enc, feature_names=features)
    dtest = xgb.DMatrix(test[features], label=y_test_enc, feature_names=features)

    params = {
        "objective": "multi:softprob",
        "num_class": 3,
        "eval_metric": "mlogloss",
        "tree_method": "hist",
        "device": "cpu",
        "eta": 0.05,
        "max_depth": 8,
        "min_child_weight": 5,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "lambda": 1.0,
        "alpha": 0.1,
    }

    print(f"\nTraining {variant} with {len(features)} features")
    print(f"Train {train.shape}  Val {val.shape}  Test {test.shape}")
    model = xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=2000,
        evals=[(dtrain, "train"), (dval, "val")],
        early_stopping_rounds=50,
        verbose_eval=100,
    )
    iteration_range = (0, model.best_iteration + 1)
    probs_val = model.predict(dval, iteration_range=iteration_range).reshape(-1, 3)
    probs_test = model.predict(dtest, iteration_range=iteration_range).reshape(-1, 3)

    dry_idx = LABEL_MAP[-1]
    val_raw = probs_val[:, dry_idx]
    test_raw = probs_test[:, dry_idx]
    val["is_dry"] = (val[target_col] == -1).astype(float)
    test["is_dry"] = (test[target_col] == -1).astype(float)

    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    iso.fit(val_raw, val["is_dry"].to_numpy(dtype=float))
    val_iso = iso.predict(val_raw)
    test_iso = iso.predict(test_raw)

    platt = LogisticRegression(solver="lbfgs")
    platt.fit(val_raw.reshape(-1, 1), val["is_dry"].to_numpy(dtype=float))
    val_platt = platt.predict_proba(val_raw.reshape(-1, 1))[:, 1]
    test_platt = platt.predict_proba(test_raw.reshape(-1, 1))[:, 1]

    val[f"{variant}_raw_prob_dry"] = val_raw
    val[f"{variant}_isotonic_prob_dry"] = val_iso
    val[f"{variant}_platt_prob_dry"] = val_platt
    test[f"{variant}_raw_prob_dry"] = test_raw
    test[f"{variant}_isotonic_prob_dry"] = test_iso
    test[f"{variant}_platt_prob_dry"] = test_platt

    cal_cols = {
        "raw": f"{variant}_raw_prob_dry",
        "isotonic": f"{variant}_isotonic_prob_dry",
        "platt": f"{variant}_platt_prob_dry",
    }
    val_bs = {method: monthly_bs(val, col) for method, col in cal_cols.items()}
    best_method = min(val_bs, key=val_bs.get)
    test[f"{variant}_selected_prob_dry"] = test[cal_cols[best_method]]
    test[f"{variant}_persistence_prob_dry"] = (test[persistence_feature] <= -1.0).astype(float)

    group_cols = {
        "y_true_dry_frac": ("is_dry", "mean"),
        "clim_prob_dry": ("clim_prob_dry", "mean"),
        f"{variant}_raw_prob_dry": (f"{variant}_raw_prob_dry", "mean"),
        f"{variant}_isotonic_prob_dry": (f"{variant}_isotonic_prob_dry", "mean"),
        f"{variant}_platt_prob_dry": (f"{variant}_platt_prob_dry", "mean"),
        f"{variant}_selected_prob_dry": (f"{variant}_selected_prob_dry", "mean"),
        f"{variant}_persistence_prob_dry": (f"{variant}_persistence_prob_dry", "mean"),
    }
    monthly = test.groupby("target_time", observed=True).agg(**group_cols).reset_index()

    y_pred = np.array([INV_LABEL_MAP[i] for i in probs_test.argmax(axis=1)])
    report = classification_report(
        test[target_col].to_numpy(),
        y_pred,
        target_names=["dry(-1)", "normal(0)", "wet(+1)"],
        digits=3,
    )
    cm = confusion_matrix(test[target_col].to_numpy(), y_pred, labels=[-1, 0, 1])

    model_path = OUT_DIR / f"{output_prefix}_{variant}_xgb_model.json"
    probs_path = OUT_DIR / f"{output_prefix}_{variant}_xgb_test_probs.npz"
    fi_csv = OUT_DIR / f"{output_prefix}_{variant}_feature_importance.csv"
    fi_png = OUT_DIR / f"{output_prefix}_{variant}_feature_importance.png"
    model.save_model(model_path.as_posix())
    np.savez_compressed(
        probs_path,
        probs=probs_test.astype("float32"),
        y_true=test[target_col].to_numpy(),
        target_times=test["target_time"].to_numpy(),
        latitude=test["latitude"].to_numpy(),
        longitude=test["longitude"].to_numpy(),
        features=np.array(features),
        best_iteration=model.best_iteration,
        best_calibration=best_method,
    )

    importance = model.get_score(importance_type="gain")
    imp = pd.DataFrame(
        {"feature": features, "gain": [importance.get(f, 0.0) for f in features]}
    ).sort_values("gain", ascending=False)
    imp.to_csv(fi_csv, index=False)
    fig, ax = plt.subplots(figsize=(8, max(4.5, 0.28 * len(features))))
    plot_imp = imp.sort_values("gain", ascending=True)
    ax.barh(plot_imp["feature"], plot_imp["gain"])
    ax.set_xlabel("Gain")
    ax.set_title(f"{variant}: SPI-{target_spi} lead-{lead_months} feature importance")
    plt.tight_layout()
    fig.savefig(fi_png, dpi=150, bbox_inches="tight")
    plt.close(fig)

    summary_rows = []
    pred_cols = {
        "persistence": f"{variant}_persistence_prob_dry",
        "raw": f"{variant}_raw_prob_dry",
        "isotonic": f"{variant}_isotonic_prob_dry",
        "platt": f"{variant}_platt_prob_dry",
        "selected": f"{variant}_selected_prob_dry",
    }
    y = monthly["y_true_dry_frac"].to_numpy(dtype=float)
    ref = monthly["clim_prob_dry"].to_numpy(dtype=float)
    for i, (method, col) in enumerate(pred_cols.items()):
        p = monthly[col].to_numpy(dtype=float)
        lo, hi = bootstrap_bss(monthly, col, "clim_prob_dry", n_bootstrap, seed=300 + i)
        summary_rows.append(
            {
                "experiment": f"memory_target_spi{target_spi}_lead{lead_months}",
                "variant": variant,
                "model": "target-consistent persistence" if method == "persistence" else "XGBoost",
                "calibration": method if method != "persistence" else "none",
                "target": f"SPI-{target_spi} dry fraction",
                "lead_months": lead_months,
                "n_test_months": int(len(monthly)),
                "bs_reference": brier(y, ref),
                "bs_model": brier(y, p),
                "bss_vs_climatology": bss(y, p, ref),
                "bss_ci_low": lo,
                "bss_ci_high": hi,
                "spearman_obs_pred": monthly[col].corr(monthly["y_true_dry_frac"], method="spearman"),
                "amplitude_ratio": amplitude_ratio(monthly["y_true_dry_frac"], monthly[col]),
                "bias": float(monthly[col].mean() - monthly["y_true_dry_frac"].mean()),
                "best_validation_calibration": best_method,
                "validation_monthly_bs_raw": val_bs["raw"],
                "validation_monthly_bs_isotonic": val_bs["isotonic"],
                "validation_monthly_bs_platt": val_bs["platt"],
                "n_features": len(features),
                "top_features": "; ".join(imp.head(8)["feature"].tolist()),
                "source_file": str((OUT_DIR / f"{output_prefix}_{variant}_monthly_scores.csv").relative_to(PROJECT_ROOT)),
            }
        )

    monthly.to_csv(OUT_DIR / f"{output_prefix}_{variant}_monthly_scores.csv", index=False)
    print(report)
    print("Confusion matrix labels [-1, 0, 1]")
    print(cm)
    return monthly, pd.DataFrame(summary_rows)


def operational_row_from_monthly(
    path: Path,
    model_prefix: str,
    target_spi: int,
    lead_months: int,
    n_bootstrap: int,
) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    pred_cols = {
        "raw": "operational_raw_prob_dry",
        "isotonic": "operational_isotonic_prob_dry",
        "selected": "operational_selected_prob_dry",
    }
    rows = []
    y = df["y_true_dry_frac"].to_numpy(dtype=float)
    ref = df["clim_prob_dry"].to_numpy(dtype=float)
    for i, (method, col) in enumerate(pred_cols.items()):
        if col not in df.columns:
            continue
        p = df[col].to_numpy(dtype=float)
        lo, hi = bootstrap_bss(df, col, "clim_prob_dry", n_bootstrap, seed=500 + i)
        rows.append(
            {
                "experiment": f"memory_target_spi{target_spi}_lead{lead_months}",
                "variant": "external_operational",
                "model": model_prefix,
                "calibration": method,
                "target": f"SPI-{target_spi} dry fraction",
                "lead_months": lead_months,
                "n_test_months": int(len(df)),
                "bs_reference": brier(y, ref),
                "bs_model": brier(y, p),
                "bss_vs_climatology": bss(y, p, ref),
                "bss_ci_low": lo,
                "bss_ci_high": hi,
                "spearman_obs_pred": df[col].corr(df["y_true_dry_frac"], method="spearman"),
                "amplitude_ratio": amplitude_ratio(df["y_true_dry_frac"], df[col]),
                "bias": float(df[col].mean() - df["y_true_dry_frac"].mean()),
                "best_validation_calibration": "",
                "validation_monthly_bs_raw": np.nan,
                "validation_monthly_bs_isotonic": np.nan,
                "validation_monthly_bs_platt": np.nan,
                "n_features": np.nan,
                "top_features": "",
                "source_file": str(path.relative_to(PROJECT_ROOT)),
            }
        )
    return pd.DataFrame(rows)


def audit_nmme_coverage(
    observed_monthly: pd.DataFrame,
    nmme_prob_csv: Path,
    nmme_anom_csv: Path,
    output_prefix: str,
) -> pd.DataFrame:
    rows = []
    for label, path in [
        ("CPC NMME probability", nmme_prob_csv),
        ("CPC NMME anomaly", nmme_anom_csv),
    ]:
        if not path.exists():
            rows.append(
                {
                    "source": label,
                    "forecast_csv": str(path.relative_to(PROJECT_ROOT)),
                    "available": False,
                    "first_target": "",
                    "last_target": "",
                    "train_months": 0,
                    "validation_months": 0,
                    "test_months": 0,
                    "strict_split_training_usable": False,
                    "interpretation": "forecast CSV is missing",
                }
            )
            continue
        fcst = pd.read_csv(path)
        fcst["target_time"] = pd.to_datetime(fcst["target_time"]).dt.to_period("M").dt.to_timestamp()
        merged = observed_monthly[["target_time", "target_year"]].merge(
            fcst[["target_time"]].drop_duplicates(), on="target_time", how="inner"
        )
        train_n = int((merged["target_year"] <= 2016).sum())
        val_n = int(((merged["target_year"] >= 2017) & (merged["target_year"] <= 2020)).sum())
        test_n = int((merged["target_year"] >= 2021).sum())
        usable = train_n > 0
        rows.append(
            {
                "source": label,
                "forecast_csv": str(path.relative_to(PROJECT_ROOT)),
                "available": True,
                "first_target": fcst["target_time"].min().strftime("%Y-%m"),
                "last_target": fcst["target_time"].max().strftime("%Y-%m"),
                "train_months": train_n,
                "validation_months": val_n,
                "test_months": test_n,
                "strict_split_training_usable": usable,
                "interpretation": (
                    "usable as an ML feature under the strict split"
                    if usable
                    else "no overlap with 1991-2016 training period; use as external benchmark unless hindcasts are added"
                ),
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(OUT_DIR / f"{output_prefix}_nmme_coverage_audit.csv", index=False)
    return out


def write_scores(
    output_prefix: str,
    target_spi: int,
    lead_months: int,
    features_by_variant: dict[str, list[str]],
    summary: pd.DataFrame,
    coverage: pd.DataFrame,
) -> None:
    summary_path = OUT_DIR / f"{output_prefix}_summary.csv"
    scores_path = OUT_DIR / f"{output_prefix}_scores.txt"
    summary.to_csv(summary_path, index=False)

    selected = summary[
        (summary["calibration"].eq("selected"))
        | (summary["model"].eq("target-consistent persistence"))
    ].copy()
    selected = selected.sort_values("bss_vs_climatology", ascending=False)

    lines = [
        f"Memory Target Experiment: SPI-{target_spi} Lead-{lead_months}",
        "=" * 72,
        "Design: features at t predict leakage-safe memory target at t+lead.",
        "Primary question: does adding ERA5-Land soil-memory improve calibrated BSS?",
        "",
        "Feature sets:",
    ]
    for name, features in features_by_variant.items():
        lines.append(f"  {name}: {len(features)} features")
        lines.append(f"    {features}")
    lines.extend(["", "CPC NMME coverage audit:"])
    for row in coverage.to_dict(orient="records"):
        lines.append(
            "  "
            f"{row['source']}: targets {row['first_target']} to {row['last_target']}; "
            f"train={row['train_months']}, val={row['validation_months']}, test={row['test_months']}; "
            f"{row['interpretation']}"
        )
    lines.extend(["", "Selected/persistence rows:"])
    for row in selected.to_dict(orient="records"):
        lines.append(
            f"  {row['variant']:<22} {row['model']:<32} {row['calibration']:<9} "
            f"n={int(row['n_test_months']):>2} "
            f"BS={row['bs_model']:.5f} "
            f"BSS={row['bss_vs_climatology']:+.5f} "
            f"95% CI [{row['bss_ci_low']:+.5f}, {row['bss_ci_high']:+.5f}] "
            f"rho={row['spearman_obs_pred']:+.3f} "
            f"amp={row['amplitude_ratio']:.3f}"
        )
    lines.extend(["", "Outputs:", f"  {summary_path}", f"  {scores_path}"])
    scores_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


def main() -> None:
    args = parse_args()
    output_prefix = args.output_prefix or f"memory_target_spi{args.target_spi}_lead{args.lead_months}"
    df = load_memory_dataset(args)

    climate_features = active_climate_features(list(df.columns), include_pdo=args.include_pdo)
    base_features = BASE_FEATURES + climate_features
    missing_base = [c for c in base_features if c not in df.columns]
    if missing_base:
        raise ValueError(f"Missing requested base features: {missing_base}")

    features_by_variant = {
        "lag_climate": base_features,
        "soil_memory": base_features + SOIL_FEATURES,
    }

    target_col = f"target_label_spi{args.target_spi}"
    observed_monthly = (
        df.assign(is_dry=(df[target_col] == -1).astype(float), target_year=df["year"])
        .groupby(["target_time", "target_year"], observed=True)
        .agg(y_true_dry_frac=("is_dry", "mean"))
        .reset_index()
    )

    monthly_parts = []
    summary_parts = []
    for variant, features in features_by_variant.items():
        monthly, summary = train_variant(
            df=df,
            target_spi=args.target_spi,
            lead_months=args.lead_months,
            variant=variant,
            features=features,
            n_bootstrap=args.n_bootstrap,
            output_prefix=output_prefix,
        )
        monthly_parts.append(monthly)
        summary_parts.append(summary)

    combined_monthly = monthly_parts[0]
    for monthly in monthly_parts[1:]:
        keep = [c for c in monthly.columns if c not in {"y_true_dry_frac", "clim_prob_dry"}]
        combined_monthly = combined_monthly.merge(monthly[keep], on="target_time", how="outer")
    combined_monthly.to_csv(OUT_DIR / f"{output_prefix}_monthly_scores.csv", index=False)

    coverage = audit_nmme_coverage(
        observed_monthly=observed_monthly,
        nmme_prob_csv=args.nmme_prob_csv,
        nmme_anom_csv=args.nmme_anom_csv,
        output_prefix=output_prefix,
    )

    summary_parts.append(
        operational_row_from_monthly(
            args.nmme_prob_monthly,
            "CPC NMME probability forecast",
            args.target_spi,
            args.lead_months,
            args.n_bootstrap,
        )
    )
    summary_parts.append(
        operational_row_from_monthly(
            args.nmme_anom_monthly,
            "CPC NMME anomaly forecast",
            args.target_spi,
            args.lead_months,
            args.n_bootstrap,
        )
    )
    summary = pd.concat([part for part in summary_parts if not part.empty], ignore_index=True)
    write_scores(output_prefix, args.target_spi, args.lead_months, features_by_variant, summary, coverage)

    if args.copy_report:
        for suffix in [
            "summary.csv",
            "scores.txt",
            "monthly_scores.csv",
            "nmme_coverage_audit.csv",
        ]:
            shutil.copy2(OUT_DIR / f"{output_prefix}_{suffix}", REPORT_DIR / f"{output_prefix}_{suffix}")


if __name__ == "__main__":
    main()
