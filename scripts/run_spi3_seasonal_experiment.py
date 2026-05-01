#!/usr/bin/env python
"""
Leakage-free seasonal SPI-3 forecast experiment.

This is intentionally separate from the canonical SPI-1 checkpoint.  It tests
whether a longer accumulation target is more predictable without overwriting
data/processed/dataset_forecast.parquet or the main output artifacts.

Forecast design:
  FEATURES at month t
  TARGET   SPI-3 drought class at month t + lead_months

With the default lead_months=3, the target SPI-3 window uses precipitation from
t+1, t+2, and t+3.  That avoids the SPI-3 leakage problem described in
build_dataset_forecast.py, where SPI-3[t+1] overlaps with features at t.

Outputs:
  data/processed/dataset_forecast_spi3_lead3.parquet
  outputs/seasonal_spi3_xgb_model.json
  outputs/seasonal_spi3_xgb_test_probs.npz
  outputs/seasonal_spi3_xgb_feature_importance.png
  outputs/seasonal_spi3_monthly_scores.csv
  outputs/seasonal_spi3_experiment_scores.txt
"""
from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
import xgboost as xgb
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_sample_weight

from feature_config import get_feature_columns

PROCESSED = Path("data/processed")
OUT_DIR = Path("outputs")
OUT_DIR.mkdir(exist_ok=True)

PR_FILE = PROCESSED / "chirps_v3_monthly_cvalley_1991_2026.nc"
SPI_FILE = PROCESSED / "chirps_v3_monthly_cvalley_spi_1991_2026.nc"
CLIMATE_FILE = PROCESSED / "climate_indices_monthly.csv"
MISSING_SENTINELS = (-9.9, -99.99, -999.0)

TARGET = "target_label"
LABEL_MAP = {-1: 0, 0: 1, 1: 2}
INV_LABEL_MAP = {v: k for k, v in LABEL_MAP.items()}


def parse_args() -> ArgumentParser:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument(
        "--lead-months",
        type=int,
        default=3,
        help="Forecast lead in months. Default 3 gives leakage-free SPI-3.",
    )
    parser.add_argument(
        "--climate-features",
        choices=["nino34", "pdo", "all", "none"],
        default="nino34",
        help="Optional climate features to merge. Default: nino34.",
    )
    parser.add_argument(
        "--rebuild-dataset",
        action="store_true",
        help="Rebuild the SPI-3 experiment parquet even if it already exists.",
    )
    parser.add_argument(
        "--n-bootstrap",
        type=int,
        default=2000,
        help="Monthly bootstrap iterations for BSS confidence intervals.",
    )
    return parser.parse_args()


def climate_lags(all_times: pd.DatetimeIndex, climate_features: str) -> pd.DataFrame:
    if climate_features == "none":
        return pd.DataFrame({"time": all_times})
    if not CLIMATE_FILE.exists():
        print(f"Climate file not found; running without climate features: {CLIMATE_FILE}")
        return pd.DataFrame({"time": all_times})

    selected = []
    if climate_features in {"all", "nino34"}:
        selected.append("nino34")
    if climate_features in {"all", "pdo"}:
        selected.append("pdo")

    cdf = pd.read_csv(CLIMATE_FILE)
    missing = {"time", *selected}.difference(cdf.columns)
    if missing:
        raise ValueError(f"{CLIMATE_FILE} is missing required columns: {sorted(missing)}")

    cdf["time"] = pd.to_datetime(cdf["time"]).dt.to_period("M").dt.to_timestamp()
    cdf = (
        cdf[["time"] + selected]
        .sort_values("time")
        .drop_duplicates(subset=["time"], keep="last")
        .set_index("time")
    )
    cdf[selected] = cdf[selected].replace(list(MISSING_SENTINELS), np.nan)
    cdf = cdf.reindex(all_times).sort_index()
    cdf[selected] = cdf[selected].interpolate(method="time", limit_area="inside")

    out_cols = []
    if "nino34" in selected:
        cdf["nino34_lag1"] = cdf["nino34"]
        cdf["nino34_lag2"] = cdf["nino34"].shift(1)
        out_cols.extend(["nino34_lag1", "nino34_lag2"])
    if "pdo" in selected:
        cdf["pdo_lag1"] = cdf["pdo"]
        cdf["pdo_lag2"] = cdf["pdo"].shift(1)
        out_cols.extend(["pdo_lag1", "pdo_lag2"])

    return cdf[out_cols].reset_index().rename(columns={"index": "time"})


def build_dataset(lead_months: int, climate_features: str, out_path: Path) -> pd.DataFrame:
    print("Loading CHIRPS/SPI grids...")
    pr_ds = xr.open_dataset(PR_FILE).load()
    spi_ds = xr.open_dataset(SPI_FILE).load()

    lat_name = "latitude" if "latitude" in pr_ds.coords else "lat"
    lon_name = "longitude" if "longitude" in pr_ds.coords else "lon"

    pr = pr_ds["pr"].sel(time=spi_ds.time)
    spi1 = spi_ds["spi1"].sel(time=pr.time)
    spi3 = spi_ds["spi3"].sel(time=pr.time)
    spi6 = spi_ds["spi6"].sel(time=pr.time)

    target_spi3 = spi3.shift(time=-lead_months)
    target_label = xr.where(
        target_spi3 <= -1.0,
        -1,
        xr.where(target_spi3 >= 1.0, 1, 0),
    ).where(target_spi3.notnull())
    target_label.name = TARGET
    target_spi3.name = "target_spi3"

    ds = xr.Dataset({
        "spi1_lag1": spi1,
        "spi1_lag2": spi1.shift(time=1),
        "spi1_lag3": spi1.shift(time=2),
        "spi3_lag1": spi3,
        "spi6_lag1": spi6,
        "pr_lag1": pr,
        "pr_lag2": pr.shift(time=1),
        "pr_lag3": pr.shift(time=2),
        "target_spi3": target_spi3,
        TARGET: target_label,
    }).stack(pixel=(lat_name, lon_name))

    df = ds.reset_index("pixel").to_dataframe()
    if "time" not in df.columns:
        df = df.reset_index()
    df["time"] = pd.to_datetime(df["time"]).dt.to_period("M").dt.to_timestamp()
    df = df.rename(columns={lat_name: "latitude", lon_name: "longitude"})

    all_times = pd.DatetimeIndex(sorted(df["time"].unique()))
    cdf = climate_lags(all_times, climate_features)
    exog_cols = [c for c in cdf.columns if c != "time"]
    if exog_cols:
        df = df.merge(cdf, on="time", how="left")
        print(f"Added climate features: {exog_cols}")
    else:
        print("No climate features added.")

    feat_cols = [
        "spi1_lag1", "spi1_lag2", "spi1_lag3",
        "spi3_lag1", "spi6_lag1",
        "pr_lag1", "pr_lag2", "pr_lag3",
    ]
    before = len(df)
    df = df.dropna(subset=[TARGET, "target_spi3"] + feat_cols + exog_cols).copy()
    print(f"Dropped rows with missing feature/target values: {before - len(df):,}")

    target_time = pd.to_datetime(df["time"]) + pd.DateOffset(months=lead_months)
    df["target_time"] = target_time.dt.to_period("M").dt.to_timestamp()
    df["month"] = df["target_time"].dt.month
    df["year"] = df["target_time"].dt.year
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12.0)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12.0)
    df[TARGET] = df[TARGET].astype(np.int8)

    cols = (
        ["time", "target_time", "year", "month", "month_sin", "month_cos",
         "latitude", "longitude"]
        + feat_cols
        + exog_cols
        + ["target_spi3", TARGET]
    )
    df = df[cols]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    df.head(10_000).to_csv(out_path.with_suffix(".sample.csv"), index=False)
    print(f"Wrote: {out_path} rows={len(df):,} cols={df.shape[1]}")
    print("Class distribution:")
    print(df[TARGET].value_counts().sort_index())
    return df


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


def run_experiment(df: pd.DataFrame, lead_months: int, n_bootstrap: int) -> None:
    df = df.copy()
    df["year"] = df["year"].astype(int)
    features = get_feature_columns(df.columns)

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
        "device": "cuda",
        "eta": 0.05,
        "max_depth": 8,
        "min_child_weight": 5,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "lambda": 1.0,
        "alpha": 0.1,
    }

    print("Training seasonal SPI-3 XGBoost...")
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
        test[TARGET].to_numpy(),
        y_pred,
        target_names=["dry(-1)", "normal(0)", "wet(+1)"],
        digits=3,
    )
    print(report)

    train["is_dry"] = (train[TARGET] == -1).astype(float)
    test["is_dry"] = (test[TARGET] == -1).astype(float)
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
    bs_xgb_cal = brier(
        monthly["y_true_dry_frac"].to_numpy(),
        monthly["xgb_cal_prob_dry"].to_numpy(),
    )
    bs_persist = brier(
        monthly["y_true_dry_frac"].to_numpy(),
        monthly["persistence_prob_dry"].to_numpy(),
    )
    bss_xgb = bss(
        monthly["y_true_dry_frac"].to_numpy(),
        monthly["xgb_prob_dry"].to_numpy(),
        monthly["clim_prob_dry"].to_numpy(),
    )
    bss_xgb_cal = bss(
        monthly["y_true_dry_frac"].to_numpy(),
        monthly["xgb_cal_prob_dry"].to_numpy(),
        monthly["clim_prob_dry"].to_numpy(),
    )
    bss_persist = bss(
        monthly["y_true_dry_frac"].to_numpy(),
        monthly["persistence_prob_dry"].to_numpy(),
        monthly["clim_prob_dry"].to_numpy(),
    )
    xgb_ci = bootstrap_bss(monthly, "xgb_prob_dry", "clim_prob_dry", n_bootstrap)
    xgb_cal_ci = bootstrap_bss(monthly, "xgb_cal_prob_dry", "clim_prob_dry", n_bootstrap)
    persist_ci = bootstrap_bss(monthly, "persistence_prob_dry", "clim_prob_dry", n_bootstrap)

    monthly_path = OUT_DIR / "seasonal_spi3_monthly_scores.csv"
    monthly.to_csv(monthly_path, index=False)

    model_path = OUT_DIR / "seasonal_spi3_xgb_model.json"
    model.save_model(model_path.as_posix())
    probs_path = OUT_DIR / "seasonal_spi3_xgb_test_probs.npz"
    np.savez_compressed(
        probs_path,
        probs=probs.astype("float32"),
        probs_calibrated=probs_cal.astype("float32"),
        y_true=test[TARGET].to_numpy(),
        times=test["time"].to_numpy(),
        target_times=test["target_time"].to_numpy(),
        latitude=test["latitude"].to_numpy(),
        longitude=test["longitude"].to_numpy(),
        features=np.array(features),
        best_iteration=model.best_iteration,
        lead_months=lead_months,
    )

    importance = model.get_score(importance_type="gain")
    imp = pd.Series([importance.get(f, 0.0) for f in features], index=features)
    imp = imp.sort_values(ascending=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(imp.index, imp.values)
    ax.set_xlabel("Gain")
    ax.set_title(f"Seasonal SPI-3 lead-{lead_months} XGBoost — Feature importance")
    plt.tight_layout()
    fi_path = OUT_DIR / "seasonal_spi3_xgb_feature_importance.png"
    fig.savefig(fi_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    cm = confusion_matrix(test[TARGET].to_numpy(), y_pred, labels=[-1, 0, 1])
    metrics = (
        f"Seasonal SPI-3 Lead-{lead_months} XGBoost Experiment\n"
        f"{'=' * 60}\n"
        f"Design: features at t, target SPI-3 class at t+{lead_months}\n"
        f"No-overlap target window for lead=3: pr[t+1..t+3]\n"
        f"Test months: {monthly['target_time'].nunique()}\n"
        f"Best iteration: {model.best_iteration}\n"
        f"Features: {features}\n\n"
        f"Monthly dry-fraction Brier Scores\n"
        f"  Climatology          : {bs_clim:.5f}\n"
        f"  Persistence SPI-3    : {bs_persist:.5f}\n"
        f"  XGBoost              : {bs_xgb:.5f}\n\n"
        f"  XGBoost isotonic     : {bs_xgb_cal:.5f}\n\n"
        f"Brier Skill Score vs monthly climatology\n"
        f"  Persistence SPI-3    : {bss_persist:.5f} "
        f"(95% CI [{persist_ci[0]:.5f}, {persist_ci[1]:.5f}])\n"
        f"  XGBoost              : {bss_xgb:.5f} "
        f"(95% CI [{xgb_ci[0]:.5f}, {xgb_ci[1]:.5f}])\n\n"
        f"  XGBoost isotonic     : {bss_xgb_cal:.5f} "
        f"(95% CI [{xgb_cal_ci[0]:.5f}, {xgb_cal_ci[1]:.5f}])\n\n"
        f"Pixel-level classification report (secondary diagnostic)\n"
        f"{report}\n"
        f"Pixel-level confusion matrix labels [-1, 0, 1]\n"
        f"{cm}\n\n"
        f"Outputs:\n"
        f"  {monthly_path}\n"
        f"  {model_path}\n"
        f"  {probs_path}\n"
        f"  {fi_path}\n"
    )
    scores_path = OUT_DIR / "seasonal_spi3_experiment_scores.txt"
    scores_path.write_text(metrics)
    print(metrics)


def main() -> None:
    args = parse_args()
    if args.lead_months < 3:
        raise ValueError(
            "Use lead_months >= 3 for a leakage-free SPI-3 experiment; "
            "SPI-3 at shorter leads overlaps with features at t."
        )

    dataset_path = PROCESSED / f"dataset_forecast_spi3_lead{args.lead_months}.parquet"
    if dataset_path.exists() and not args.rebuild_dataset:
        print(f"Loading existing experiment dataset: {dataset_path}")
        df = pd.read_parquet(dataset_path)
        df["time"] = pd.to_datetime(df["time"])
        df["target_time"] = pd.to_datetime(df["target_time"])
    else:
        df = build_dataset(args.lead_months, args.climate_features, dataset_path)

    run_experiment(df, args.lead_months, args.n_bootstrap)


if __name__ == "__main__":
    main()
