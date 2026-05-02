#!/usr/bin/env python
"""Run leakage-safe seasonal long-lead XGBoost experiments for SPI-3/SPI-6 targets."""
from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_sample_weight

from feature_config import get_feature_columns
from build_dataset_seasonal import _build_one

import xarray as xr

PROCESSED = Path("data/processed")
OUT_DIR = Path("outputs")
OUT_DIR.mkdir(exist_ok=True)

PR_FILE = PROCESSED / "chirps_v3_monthly_cvalley_1991_2026.nc"
SPI_FILE = PROCESSED / "chirps_v3_monthly_cvalley_spi_1991_2026.nc"

LABEL_MAP = {-1: 0, 0: 1, 1: 2}
INV_LABEL_MAP = {v: k for k, v in LABEL_MAP.items()}


def parse_args() -> ArgumentParser:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--target-spi", type=int, choices=[3, 6], default=3)
    parser.add_argument("--lead-months", type=int, default=3)
    parser.add_argument(
        "--climate-features",
        choices=["nino34", "pdo", "all", "none"],
        default="nino34",
    )
    parser.add_argument("--rebuild-dataset", action="store_true")
    parser.add_argument("--n-bootstrap", type=int, default=1000)
    return parser.parse_args()


def brier(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean((p - y) ** 2))


def bss(y: np.ndarray, p: np.ndarray, ref: np.ndarray) -> float:
    ref_bs = brier(y, ref)
    if ref_bs <= 0:
        return float("nan")
    return 1.0 - brier(y, p) / ref_bs


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


def _load_or_build_dataset(target_spi: int, lead_months: int, climate_features: str, rebuild_dataset: bool) -> pd.DataFrame:
    dataset_path = PROCESSED / f"dataset_seasonal_spi{target_spi}_lead{lead_months}.parquet"
    if dataset_path.exists() and not rebuild_dataset:
        print(f"Loading existing dataset: {dataset_path}")
        df = pd.read_parquet(dataset_path)
        df["time"] = pd.to_datetime(df["time"])
        df["target_time"] = pd.to_datetime(df["target_time"])
        return df

    print("Building dataset from source grids...")
    pr_ds = xr.open_dataset(PR_FILE).load()
    spi_ds = xr.open_dataset(SPI_FILE).load()

    lat_name = "latitude" if "latitude" in pr_ds.coords else "lat"
    lon_name = "longitude" if "longitude" in pr_ds.coords else "lon"

    pr = pr_ds["pr"].sel(time=spi_ds.time)
    spi1 = spi_ds["spi1"].sel(time=pr.time)
    spi3 = spi_ds["spi3"].sel(time=pr.time)
    spi6 = spi_ds["spi6"].sel(time=pr.time)
    spi_target = spi3 if target_spi == 3 else spi6

    df = _build_one(
        pr=pr,
        spi1=spi1,
        spi3=spi3,
        spi6=spi6,
        spi_target=spi_target,
        lead=lead_months,
        spi_idx=target_spi,
        lat_name=lat_name,
        lon_name=lon_name,
        climate_features=climate_features,
    )
    dataset_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(dataset_path, index=False)
    df.head(10000).to_csv(dataset_path.with_suffix(".sample.csv"), index=False)
    print(f"Wrote: {dataset_path} rows={len(df):,}")
    return df


def run_experiment(df: pd.DataFrame, target_spi: int, lead_months: int, n_bootstrap: int) -> None:
    target_col = f"target_label_spi{target_spi}"
    target_value_col = f"target_spi{target_spi}"

    if "year" not in df.columns:
        df["year"] = pd.to_datetime(df["target_time"]).dt.year
    df["year"] = df["year"].astype(int)

    features = get_feature_columns(df.columns)

    train = df[df["year"] <= 2016].copy()
    val = df[(df["year"] >= 2017) & (df["year"] <= 2020)].copy()
    test = df[df["year"] >= 2021].copy()

    print(f"Train {train.shape}  Val {val.shape}  Test {test.shape}")
    print(f"Features: {features}")

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
        "eta": 0.05,
        "max_depth": 8,
        "min_child_weight": 5,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "lambda": 1.0,
        "alpha": 0.1,
    }

    print(f"Training seasonal SPI-{target_spi} lead-{lead_months} XGBoost...")
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

    probs_cal = np.zeros_like(probs, dtype="float32")
    for k in range(3):
        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(probs_val[:, k], (y_val_enc == k).astype(int))
        probs_cal[:, k] = iso.predict(probs[:, k]).astype("float32")
    row_sum = probs_cal.sum(axis=1, keepdims=True)
    row_sum[row_sum <= 0] = 1.0
    probs_cal = probs_cal / row_sum

    y_pred = np.array([INV_LABEL_MAP[i] for i in probs.argmax(axis=1)])
    report = classification_report(
        test[target_col].to_numpy(),
        y_pred,
        target_names=["dry(-1)", "normal(0)", "wet(+1)"],
        digits=3,
    )

    train["is_dry"] = (train[target_col] == -1).astype(float)
    test["is_dry"] = (test[target_col] == -1).astype(float)
    train_monthly_dry = train.groupby("month")["is_dry"].mean()
    global_dry = float(train["is_dry"].mean())
    test["clim_prob_dry"] = test["month"].map(train_monthly_dry).fillna(global_dry)
    test["xgb_prob_dry"] = probs[:, LABEL_MAP[-1]]
    test["xgb_cal_prob_dry"] = probs_cal[:, LABEL_MAP[-1]]
    test["persistence_prob_dry"] = (test["spi3_lag1"] <= -1.0).astype(float)

    monthly = (
        test.groupby("target_time")
        .agg(
            y_true_dry_frac=("is_dry", "mean"),
            xgb_prob_dry=("xgb_prob_dry", "mean"),
            xgb_cal_prob_dry=("xgb_cal_prob_dry", "mean"),
            clim_prob_dry=("clim_prob_dry", "mean"),
            persistence_prob_dry=("persistence_prob_dry", "mean"),
        )
        .reset_index()
    )

    bs_clim = brier(monthly["y_true_dry_frac"].to_numpy(), monthly["clim_prob_dry"].to_numpy())
    bs_xgb = brier(monthly["y_true_dry_frac"].to_numpy(), monthly["xgb_prob_dry"].to_numpy())
    bs_xgb_cal = brier(monthly["y_true_dry_frac"].to_numpy(), monthly["xgb_cal_prob_dry"].to_numpy())
    bs_persist = brier(monthly["y_true_dry_frac"].to_numpy(), monthly["persistence_prob_dry"].to_numpy())

    bss_xgb = bss(monthly["y_true_dry_frac"].to_numpy(), monthly["xgb_prob_dry"].to_numpy(), monthly["clim_prob_dry"].to_numpy())
    bss_xgb_cal = bss(monthly["y_true_dry_frac"].to_numpy(), monthly["xgb_cal_prob_dry"].to_numpy(), monthly["clim_prob_dry"].to_numpy())
    bss_persist = bss(monthly["y_true_dry_frac"].to_numpy(), monthly["persistence_prob_dry"].to_numpy(), monthly["clim_prob_dry"].to_numpy())

    xgb_ci = bootstrap_bss(monthly, "xgb_prob_dry", "clim_prob_dry", n_bootstrap)
    xgb_cal_ci = bootstrap_bss(monthly, "xgb_cal_prob_dry", "clim_prob_dry", n_bootstrap)
    persist_ci = bootstrap_bss(monthly, "persistence_prob_dry", "clim_prob_dry", n_bootstrap)

    prefix = f"seasonal_spi{target_spi}_lead{lead_months}"
    monthly_path = OUT_DIR / f"{prefix}_monthly_scores.csv"
    model_path = OUT_DIR / f"{prefix}_xgb_model.json"
    probs_path = OUT_DIR / f"{prefix}_xgb_test_probs.npz"
    fi_path = OUT_DIR / f"{prefix}_xgb_feature_importance.png"
    scores_path = OUT_DIR / f"{prefix}_experiment_scores.txt"

    monthly.to_csv(monthly_path, index=False)
    model.save_model(model_path.as_posix())
    np.savez_compressed(
        probs_path,
        probs=probs.astype("float32"),
        probs_calibrated=probs_cal.astype("float32"),
        y_true=test[target_col].to_numpy(),
        y_true_value=test[target_value_col].to_numpy(),
        times=test["time"].to_numpy(),
        target_times=test["target_time"].to_numpy(),
        latitude=test["latitude"].to_numpy(),
        longitude=test["longitude"].to_numpy(),
        features=np.array(features),
        best_iteration=model.best_iteration,
        lead_months=lead_months,
        target_spi=target_spi,
    )

    importance = model.get_score(importance_type="gain")
    imp = pd.Series([importance.get(f, 0.0) for f in features], index=features).sort_values(ascending=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(imp.index, imp.values)
    ax.set_xlabel("Gain")
    ax.set_title(f"Seasonal SPI-{target_spi} lead-{lead_months} XGBoost feature importance")
    plt.tight_layout()
    fig.savefig(fi_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    cm = confusion_matrix(test[target_col].to_numpy(), y_pred, labels=[-1, 0, 1])
    metrics = (
        f"Seasonal SPI-{target_spi} Lead-{lead_months} XGBoost Experiment\n"
        f"{'=' * 64}\n"
        f"Design: features at t, target SPI-{target_spi} class at t+{lead_months}\n"
        f"Leakage-safe condition enforced by builder: lead >= SPI window unless overlap allowed\n"
        f"Test months: {monthly['target_time'].nunique()}\n"
        f"Best iteration: {model.best_iteration}\n"
        f"Features: {features}\n\n"
        f"Monthly dry-fraction Brier Scores\n"
        f"  Climatology       : {bs_clim:.5f}\n"
        f"  Persistence SPI-3 : {bs_persist:.5f}\n"
        f"  XGBoost           : {bs_xgb:.5f}\n"
        f"  XGBoost isotonic  : {bs_xgb_cal:.5f}\n\n"
        f"Brier Skill Score vs monthly climatology\n"
        f"  Persistence SPI-3 : {bss_persist:.5f} (95% CI [{persist_ci[0]:.5f}, {persist_ci[1]:.5f}])\n"
        f"  XGBoost           : {bss_xgb:.5f} (95% CI [{xgb_ci[0]:.5f}, {xgb_ci[1]:.5f}])\n"
        f"  XGBoost isotonic  : {bss_xgb_cal:.5f} (95% CI [{xgb_cal_ci[0]:.5f}, {xgb_cal_ci[1]:.5f}])\n\n"
        f"Pixel-level classification report\n"
        f"{report}\n"
        f"Pixel-level confusion matrix labels [-1, 0, 1]\n"
        f"{cm}\n\n"
        f"Outputs:\n"
        f"  {monthly_path}\n"
        f"  {model_path}\n"
        f"  {probs_path}\n"
        f"  {fi_path}\n"
    )
    scores_path.write_text(metrics)
    print(metrics)


def main() -> None:
    args = parse_args()
    if args.lead_months < args.target_spi:
        raise ValueError(
            f"Use lead_months >= target_spi ({args.target_spi}) for leakage-safe seasonal setup."
        )

    df = _load_or_build_dataset(
        target_spi=args.target_spi,
        lead_months=args.lead_months,
        climate_features=args.climate_features,
        rebuild_dataset=args.rebuild_dataset,
    )
    run_experiment(df, args.target_spi, args.lead_months, args.n_bootstrap)


if __name__ == "__main__":
    main()
