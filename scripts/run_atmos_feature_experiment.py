#!/usr/bin/env python
"""
XGBoost experiment with MJO phase/amplitude and optional AR/IVT predictors.

This is non-destructive:
  - reads data/processed/dataset_forecast.parquet
  - writes data/processed/dataset_forecast_atmos.parquet
  - saves outputs/atmos_feature_xgb_* artifacts

Feature design:
  - Canonical target remains SPI-1 class at t+1.
  - MJO features are monthly aggregates from daily RMM indices.
  - Optional AR/IVT features are merged from a monthly CSV if present.
"""
from __future__ import annotations

from argparse import ArgumentParser, Namespace
from pathlib import Path
import subprocess
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_sample_weight

from feature_config import get_feature_columns


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED = PROJECT_ROOT / "data" / "processed"
OUT_DIR = PROJECT_ROOT / "outputs"
OUT_DIR.mkdir(exist_ok=True)

DATA = PROCESSED / "dataset_forecast.parquet"
ATMOS_DATASET = PROCESSED / "dataset_forecast_atmos.parquet"

MJO_FILE = PROCESSED / "mjo_rmm_monthly.csv"
MJO_DOWNLOAD = PROJECT_ROOT / "scripts" / "download_mjo_rmm.py"

AR_IVT_FILE = PROCESSED / "ar_ivt_monthly.csv"

TARGET = "target_label"
LABEL_MAP = {-1: 0, 0: 1, 1: 2}
INV_LABEL_MAP = {v: k for k, v in LABEL_MAP.items()}

MJO_FEATURES = [
    "mjo_amp_mean",
    "mjo_phase_sin",
    "mjo_phase_cos",
    "mjo_active_frac",
]


def parse_args() -> Namespace:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument(
        "--download-mjo-if-missing",
        action="store_true",
        help="Run download_mjo_rmm.py if the MJO monthly file is missing.",
    )
    parser.add_argument(
        "--rebuild-dataset",
        action="store_true",
        help="Rebuild dataset_forecast_atmos.parquet even if it exists.",
    )
    parser.add_argument(
        "--require-ar-ivt",
        action="store_true",
        help="Fail if ar_ivt_monthly.csv is not present.",
    )
    parser.add_argument(
        "--n-bootstrap",
        type=int,
        default=2000,
        help="Monthly bootstrap iterations for BSS confidence intervals.",
    )
    return parser.parse_args()


def ensure_mjo_file(download_if_missing: bool) -> None:
    if MJO_FILE.exists():
        return
    if not download_if_missing:
        raise FileNotFoundError(
            f"MJO monthly file not found: {MJO_FILE}\n"
            f"Run {MJO_DOWNLOAD} or pass --download-mjo-if-missing."
        )
    print("MJO monthly file not found. Running downloader...")
    subprocess.run([sys.executable, str(MJO_DOWNLOAD)], cwd=PROJECT_ROOT, check=True)
    if not MJO_FILE.exists():
        raise FileNotFoundError(f"Download finished, but file is still missing: {MJO_FILE}")


def load_mjo_features() -> pd.DataFrame:
    mjo = pd.read_csv(MJO_FILE)
    missing = {"time", *MJO_FEATURES}.difference(mjo.columns)
    if missing:
        raise ValueError(f"{MJO_FILE} is missing required columns: {sorted(missing)}")
    mjo["time"] = pd.to_datetime(mjo["time"]).dt.to_period("M").dt.to_timestamp()
    return mjo[["time"] + MJO_FEATURES].sort_values("time")


def load_ar_ivt_features(require: bool) -> tuple[pd.DataFrame, list[str]]:
    if not AR_IVT_FILE.exists():
        if require:
            raise FileNotFoundError(f"AR/IVT feature file not found: {AR_IVT_FILE}")
        print(f"AR/IVT file not found; proceeding without: {AR_IVT_FILE}")
        return pd.DataFrame({"time": []}), []

    df = pd.read_csv(AR_IVT_FILE)
    if "time" not in df.columns:
        raise ValueError(f"{AR_IVT_FILE} must contain a 'time' column")

    df["time"] = pd.to_datetime(df["time"]).dt.to_period("M").dt.to_timestamp()
    feature_cols = [
        c for c in df.columns
        if c != "time" and pd.api.types.is_numeric_dtype(df[c])
    ]
    if not feature_cols:
        raise ValueError(f"{AR_IVT_FILE} has no numeric feature columns")

    return df[["time"] + feature_cols].sort_values("time"), feature_cols


def build_dataset(require_ar_ivt: bool) -> tuple[pd.DataFrame, list[str]]:
    print(f"Loading canonical dataset: {DATA}")
    df = pd.read_parquet(DATA)
    df["time"] = pd.to_datetime(df["time"]).dt.to_period("M").dt.to_timestamp()

    mjo = load_mjo_features()
    ar_ivt, ar_ivt_cols = load_ar_ivt_features(require_ar_ivt)

    out = df.merge(mjo, on="time", how="left")
    if ar_ivt_cols:
        out = out.merge(ar_ivt, on="time", how="left")

    missing = int(out[MJO_FEATURES].isna().any(axis=1).sum())
    if missing:
        print(f"Rows with missing MJO features before dropna: {missing:,}")

    drop_cols = MJO_FEATURES + ar_ivt_cols
    out = out.dropna(subset=drop_cols).copy()
    out.to_parquet(ATMOS_DATASET, index=False)
    out.head(10_000).to_csv(ATMOS_DATASET.with_suffix(".sample.csv"), index=False)
    print(f"Wrote: {ATMOS_DATASET} rows={len(out):,} cols={out.shape[1]}")

    return out, ar_ivt_cols


def brier(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean((p - y) ** 2))


def bss(y: np.ndarray, p: np.ndarray, ref: np.ndarray) -> float:
    ref_bs = brier(y, ref)
    return float(1.0 - brier(y, p) / ref_bs) if ref_bs > 0 else float("nan")


def bootstrap_bss(
    monthly: pd.DataFrame,
    pred_col: str,
    ref_col: str,
    n_bootstrap: int,
    seed: int = 42,
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    idx = np.arange(len(monthly))
    vals = []
    y = monthly["y_true_dry_frac"].to_numpy()
    p = monthly[pred_col].to_numpy()
    ref = monthly[ref_col].to_numpy()
    for _ in range(n_bootstrap):
        sample = rng.choice(idx, size=len(idx), replace=True)
        vals.append(bss(y[sample], p[sample], ref[sample]))
    lo, hi = np.nanquantile(vals, [0.025, 0.975])
    return float(lo), float(hi)


def target_month(series: pd.Series) -> pd.Series:
    return (pd.to_datetime(series) + pd.DateOffset(months=1)).dt.to_period("M").dt.to_timestamp()


def monthly_bs(frame: pd.DataFrame, prob_col: str) -> float:
    monthly = (
        frame.groupby("target_time")
        .agg(
            y_true_dry_frac=("is_dry", "mean"),
            pred_prob_dry=(prob_col, "mean"),
        )
        .reset_index()
    )
    return brier(monthly["y_true_dry_frac"].to_numpy(), monthly["pred_prob_dry"].to_numpy())


def run_experiment(df: pd.DataFrame, ar_ivt_cols: list[str], n_bootstrap: int) -> None:
    df = df.copy()
    df["year"] = df["year"].astype(int)
    features = get_feature_columns(df.columns) + MJO_FEATURES + ar_ivt_cols

    train = df[df["year"] <= 2016].copy()
    val = df[(df["year"] >= 2017) & (df["year"] <= 2020)].copy()
    test = df[df["year"] >= 2021].copy()

    print(f"Train {train.shape}  Val {val.shape}  Test {test.shape}")
    print(f"Features: {features}")

    y_train_enc = train[TARGET].map(LABEL_MAP).to_numpy()
    y_val_enc = val[TARGET].map(LABEL_MAP).to_numpy()
    y_test_enc = test[TARGET].map(LABEL_MAP).to_numpy()

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

    print("Training XGBoost with MJO/AR/IVT features...")
    model = xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=2000,
        evals=[(dtrain, "train"), (dval, "val")],
        early_stopping_rounds=50,
        verbose_eval=100,
    )
    print(f"Best iteration: {model.best_iteration}")
    iteration_range = (0, model.best_iteration + 1)

    probs_val = model.predict(dval, iteration_range=iteration_range).reshape(-1, 3)
    probs = model.predict(dtest, iteration_range=iteration_range).reshape(-1, 3)

    y_pred = np.array([INV_LABEL_MAP[i] for i in probs.argmax(axis=1)])
    report = classification_report(
        test[TARGET].to_numpy(),
        y_pred,
        target_names=["dry(-1)", "normal(0)", "wet(+1)"],
        digits=3,
    )
    print(report)

    train["is_dry"] = (train[TARGET] == -1).astype(float)
    val["is_dry"] = (val[TARGET] == -1).astype(float)
    test["is_dry"] = (test[TARGET] == -1).astype(float)
    train_monthly_dry = train.groupby("month")["is_dry"].mean()
    global_dry = float(train["is_dry"].mean())
    val["clim_prob_dry"] = val["month"].map(train_monthly_dry).fillna(global_dry)
    test["clim_prob_dry"] = test["month"].map(train_monthly_dry).fillna(global_dry)

    val["target_time"] = target_month(val["time"])
    test["target_time"] = target_month(test["time"])

    dry_idx = LABEL_MAP[-1]
    val_raw = probs_val[:, dry_idx]
    test_raw = probs[:, dry_idx]

    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(val_raw, val["is_dry"].to_numpy())
    val_iso = iso.predict(val_raw)
    test_iso = iso.predict(test_raw)

    platt = LogisticRegression(solver="lbfgs")
    platt.fit(val_raw.reshape(-1, 1), val["is_dry"].to_numpy())
    val_platt = platt.predict_proba(val_raw.reshape(-1, 1))[:, 1]
    test_platt = platt.predict_proba(test_raw.reshape(-1, 1))[:, 1]

    val["xgb_raw_prob_dry"] = val_raw
    val["xgb_isotonic_prob_dry"] = val_iso
    val["xgb_platt_prob_dry"] = val_platt
    test["xgb_raw_prob_dry"] = test_raw
    test["xgb_isotonic_prob_dry"] = test_iso
    test["xgb_platt_prob_dry"] = test_platt

    calibration_cols = {
        "none": "xgb_raw_prob_dry",
        "isotonic": "xgb_isotonic_prob_dry",
        "platt": "xgb_platt_prob_dry",
    }
    val_bs_by_method = {
        method: monthly_bs(val, col)
        for method, col in calibration_cols.items()
    }
    best_method = min(val_bs_by_method, key=val_bs_by_method.get)
    best_col = calibration_cols[best_method]
    test["xgb_selected_prob_dry"] = test[best_col]

    monthly = (
        test.groupby("target_time")
        .agg(
            y_true_dry_frac=("is_dry", "mean"),
            xgb_raw_prob_dry=("xgb_raw_prob_dry", "mean"),
            xgb_isotonic_prob_dry=("xgb_isotonic_prob_dry", "mean"),
            xgb_platt_prob_dry=("xgb_platt_prob_dry", "mean"),
            xgb_selected_prob_dry=("xgb_selected_prob_dry", "mean"),
            clim_prob_dry=("clim_prob_dry", "mean"),
        )
        .reset_index()
    )

    y = monthly["y_true_dry_frac"].to_numpy()
    clim = monthly["clim_prob_dry"].to_numpy()
    raw = monthly["xgb_raw_prob_dry"].to_numpy()
    iso_pred = monthly["xgb_isotonic_prob_dry"].to_numpy()
    platt_pred = monthly["xgb_platt_prob_dry"].to_numpy()
    selected = monthly["xgb_selected_prob_dry"].to_numpy()

    bs_clim = brier(y, clim)
    bs_raw = brier(y, raw)
    bs_iso = brier(y, iso_pred)
    bs_platt = brier(y, platt_pred)
    bs_selected = brier(y, selected)
    bss_raw = bss(y, raw, clim)
    bss_iso = bss(y, iso_pred, clim)
    bss_platt = bss(y, platt_pred, clim)
    bss_selected = bss(y, selected, clim)
    raw_ci = bootstrap_bss(monthly, "xgb_raw_prob_dry", "clim_prob_dry", n_bootstrap)
    iso_ci = bootstrap_bss(monthly, "xgb_isotonic_prob_dry", "clim_prob_dry", n_bootstrap)
    platt_ci = bootstrap_bss(monthly, "xgb_platt_prob_dry", "clim_prob_dry", n_bootstrap)
    selected_ci = bootstrap_bss(monthly, "xgb_selected_prob_dry", "clim_prob_dry", n_bootstrap)

    monthly_path = OUT_DIR / "atmos_feature_xgb_monthly_scores.csv"
    monthly.to_csv(monthly_path, index=False)

    model_path = OUT_DIR / "atmos_feature_xgb_model.json"
    model.save_model(model_path.as_posix())

    probs_path = OUT_DIR / "atmos_feature_xgb_test_probs.npz"
    np.savez_compressed(
        probs_path,
        probs=probs.astype("float32"),
        dry_probs_raw=test_raw.astype("float32"),
        dry_probs_isotonic=test_iso.astype("float32"),
        dry_probs_platt=test_platt.astype("float32"),
        dry_probs_selected=test[best_col].to_numpy(dtype="float32"),
        y_true=test[TARGET].to_numpy(),
        times=test["time"].to_numpy(),
        target_times=test["target_time"].to_numpy(),
        latitude=test["latitude"].to_numpy(),
        longitude=test["longitude"].to_numpy(),
        features=np.array(features),
        best_iteration=model.best_iteration,
        best_calibration=best_method,
    )

    importance = model.get_score(importance_type="gain")
    imp = pd.Series([importance.get(f, 0.0) for f in features], index=features)
    imp = imp.sort_values(ascending=True)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(imp.index, imp.values)
    ax.set_xlabel("Gain")
    ax.set_title("XGBoost + MJO/AR/IVT features")
    plt.tight_layout()
    fi_path = OUT_DIR / "atmos_feature_xgb_feature_importance.png"
    fig.savefig(fi_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    cm = confusion_matrix(test[TARGET].to_numpy(), y_pred, labels=[-1, 0, 1])
    metrics = (
        "Atmospheric Feature Experiment (MJO + optional AR/IVT)\n"
        f"{'=' * 60}\n"
        "Design: canonical SPI-1[t+1] target with monthly MJO phase/amplitude "
        "features (and optional AR/IVT indices) at feature month t.\n"
        f"Test months: {monthly['target_time'].nunique()}\n"
        f"Best iteration: {model.best_iteration}\n"
        f"Validation monthly BS by calibration: {val_bs_by_method}\n"
        f"Selected calibration: {best_method}\n"
        f"Features: {features}\n\n"
        "Monthly dry-fraction Brier Scores\n"
        f"  Climatology           : {bs_clim:.5f}\n"
        f"  XGBoost + atmos raw   : {bs_raw:.5f}\n"
        f"  XGBoost + atmos isotonic: {bs_iso:.5f}\n"
        f"  XGBoost + atmos Platt : {bs_platt:.5f}\n"
        f"  XGBoost + atmos selected: {bs_selected:.5f}\n\n"
        "Brier Skill Score vs monthly climatology\n"
        f"  XGBoost + atmos raw   : {bss_raw:.5f} "
        f"(95% CI [{raw_ci[0]:.5f}, {raw_ci[1]:.5f}])\n"
        f"  XGBoost + atmos isotonic: {bss_iso:.5f} "
        f"(95% CI [{iso_ci[0]:.5f}, {iso_ci[1]:.5f}])\n"
        f"  XGBoost + atmos Platt : {bss_platt:.5f} "
        f"(95% CI [{platt_ci[0]:.5f}, {platt_ci[1]:.5f}])\n"
        f"  XGBoost + atmos selected: {bss_selected:.5f} "
        f"(95% CI [{selected_ci[0]:.5f}, {selected_ci[1]:.5f}])\n\n"
        "Pixel-level classification report (secondary diagnostic)\n"
        f"{report}\n"
        "Pixel-level confusion matrix labels [-1, 0, 1]\n"
        f"{cm}\n\n"
        "Outputs:\n"
        f"  {monthly_path}\n"
        f"  {model_path}\n"
        f"  {probs_path}\n"
        f"  {fi_path}\n"
    )
    scores_path = OUT_DIR / "atmos_feature_xgb_experiment_scores.txt"
    scores_path.write_text(metrics)
    print(metrics)


def main() -> None:
    args = parse_args()
    ensure_mjo_file(args.download_mjo_if_missing)
    if ATMOS_DATASET.exists() and not args.rebuild_dataset:
        print(f"Loading existing atmos-feature dataset: {ATMOS_DATASET}")
        df = pd.read_parquet(ATMOS_DATASET)
        df["time"] = pd.to_datetime(df["time"])
        base = set(get_feature_columns(df.columns))
        meta = {
            "time",
            "year",
            "month",
            "month_sin",
            "month_cos",
            "latitude",
            "longitude",
            "region",
            "target_label",
        }
        ar_ivt_cols = [
            c for c in df.columns
            if c not in base and c not in meta and c not in MJO_FEATURES
        ]
    else:
        df, ar_ivt_cols = build_dataset(args.require_ar_ivt)
    run_experiment(df, ar_ivt_cols, args.n_bootstrap)


if __name__ == "__main__":
    main()
