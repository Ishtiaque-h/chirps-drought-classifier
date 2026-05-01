#!/usr/bin/env python
"""
XGBoost experiment with ERA5-Land temperature and VPD anomaly features.

This experiment is non-destructive:
  - it reads the canonical dataset_forecast.parquet
  - it writes dataset_forecast_met.parquet
  - it saves separate model/probability/metric artifacts under outputs/

Feature design:
  ERA5-Land 2m temperature and dewpoint are converted to regional monthly
  mean temperature and VPD.  Calendar-month anomalies are computed relative to
  1991-2020.  Lag-1 is the feature month t; lag-2 is t-1.  The target remains
  the canonical SPI-1 class at t+1, so the features are leakage-safe.
"""
from __future__ import annotations

from argparse import ArgumentParser
from datetime import datetime, timezone
from pathlib import Path
import subprocess
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
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

START_YEAR = 1991
CURRENT_YEAR = datetime.now(timezone.utc).year
DATA = PROCESSED / "dataset_forecast.parquet"
MET_FILE = PROCESSED / f"era5_land_met_monthly_cvalley_{START_YEAR}_{CURRENT_YEAR}.nc"
MET_DATASET = PROCESSED / "dataset_forecast_met.parquet"
DOWNLOAD_SCRIPT = PROJECT_ROOT / "scripts" / "download_era5_land_met_monthly.py"

TARGET = "target_label"
LABEL_MAP = {-1: 0, 0: 1, 1: 2}
INV_LABEL_MAP = {v: k for k, v in LABEL_MAP.items()}
MET_FEATURES = [
    "t2m_anom_lag1",
    "t2m_anom_lag2",
    "vpd_anom_lag1",
    "vpd_anom_lag2",
]


def parse_args() -> ArgumentParser:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument(
        "--download-if-missing",
        action="store_true",
        help="Run download_era5_land_met_monthly.py if the met NetCDF is missing.",
    )
    parser.add_argument(
        "--rebuild-dataset",
        action="store_true",
        help="Rebuild dataset_forecast_met.parquet even if it already exists.",
    )
    parser.add_argument(
        "--n-bootstrap",
        type=int,
        default=2000,
        help="Monthly bootstrap iterations for BSS confidence intervals.",
    )
    return parser.parse_args()


def ensure_met_file(download_if_missing: bool) -> None:
    if MET_FILE.exists():
        return
    if not download_if_missing:
        raise FileNotFoundError(
            f"ERA5-Land met file not found: {MET_FILE}\n"
            f"Run {DOWNLOAD_SCRIPT} or pass --download-if-missing."
        )
    print(f"ERA5-Land met file not found: {MET_FILE}")
    print("Running met download script...")
    subprocess.run([sys.executable, str(DOWNLOAD_SCRIPT)], cwd=PROJECT_ROOT, check=True)
    if not MET_FILE.exists():
        raise FileNotFoundError(f"Download finished, but file is still missing: {MET_FILE}")


def choose_var(ds: xr.Dataset, candidates: list[str]) -> str:
    for candidate in candidates:
        if candidate in ds.data_vars:
            return candidate
    lower_map = {name.lower(): name for name in ds.data_vars}
    for candidate in candidates:
        if candidate.lower() in lower_map:
            return lower_map[candidate.lower()]
    raise ValueError(f"Could not find any of {candidates}; available={list(ds.data_vars)}")


def standardize_met_da(ds: xr.Dataset, var_name: str) -> xr.DataArray:
    da = ds[var_name]
    rename_map = {}
    if "valid_time" in da.dims:
        rename_map["valid_time"] = "time"
    if "lat" in da.dims:
        rename_map["lat"] = "latitude"
    if "lon" in da.dims:
        rename_map["lon"] = "longitude"
    if rename_map:
        da = da.rename(rename_map)

    if "expver" in da.dims:
        da = da.mean(dim="expver", skipna=True)

    extra_dims = [d for d in da.dims if d not in {"time", "latitude", "longitude"}]
    for dim in extra_dims:
        if da.sizes[dim] == 1:
            da = da.squeeze(dim, drop=True)
    extra_dims = [d for d in da.dims if d not in {"time", "latitude", "longitude"}]
    if extra_dims:
        raise ValueError(f"Unexpected dimensions for {var_name}: {da.dims}")

    da = da.transpose("time", "latitude", "longitude")
    time_index = pd.DatetimeIndex(da["time"].values)
    da = da.isel(time=~time_index.duplicated()).sortby("time")
    return da


def to_celsius(da: xr.DataArray) -> xr.DataArray:
    units = str(da.attrs.get("units", "")).strip().lower()
    values = da.values
    if units in {"k", "kelvin"} or np.nanmedian(values) > 100:
        return da - 273.15
    return da


def saturation_vapor_pressure_kpa(temp_c: xr.DataArray) -> xr.DataArray:
    return 0.6108 * np.exp((17.27 * temp_c) / (temp_c + 237.3))


def build_met_monthly_features() -> pd.DataFrame:
    print(f"Loading ERA5-Land met data: {MET_FILE}")
    ds = xr.open_dataset(MET_FILE).load()
    t_var = choose_var(ds, ["t2m", "2m_temperature"])
    d_var = choose_var(ds, ["d2m", "2m_dewpoint_temperature"])

    t2m_c = to_celsius(standardize_met_da(ds, t_var))
    d2m_c = to_celsius(standardize_met_da(ds, d_var))
    d2m_c = d2m_c.sel(time=t2m_c.time)

    vpd_kpa = saturation_vapor_pressure_kpa(t2m_c) - saturation_vapor_pressure_kpa(d2m_c)
    vpd_kpa = vpd_kpa.clip(min=0.0)

    t2m_reg = t2m_c.mean(dim=["latitude", "longitude"], skipna=True)
    vpd_reg = vpd_kpa.mean(dim=["latitude", "longitude"], skipna=True)

    met = pd.DataFrame({
        "time": pd.to_datetime(t2m_reg["time"].values).to_period("M").to_timestamp(),
        "t2m_c": t2m_reg.values.astype(float),
        "vpd_kpa": vpd_reg.values.astype(float),
    }).sort_values("time")

    base = met[(met["time"].dt.year >= 1991) & (met["time"].dt.year <= 2020)].copy()
    clim = (
        base.assign(month=base["time"].dt.month)
        .groupby("month")[["t2m_c", "vpd_kpa"]]
        .mean()
        .rename(columns={"t2m_c": "t2m_clim", "vpd_kpa": "vpd_clim"})
    )
    met["month"] = met["time"].dt.month
    met = met.merge(clim, on="month", how="left")
    met["t2m_anom"] = met["t2m_c"] - met["t2m_clim"]
    met["vpd_anom"] = met["vpd_kpa"] - met["vpd_clim"]

    met = met[["time", "t2m_anom", "vpd_anom"]].set_index("time").sort_index()
    met["t2m_anom_lag1"] = met["t2m_anom"]
    met["t2m_anom_lag2"] = met["t2m_anom"].shift(1)
    met["vpd_anom_lag1"] = met["vpd_anom"]
    met["vpd_anom_lag2"] = met["vpd_anom"].shift(1)
    met = met[MET_FEATURES].reset_index()

    print("ERA5-Land met feature range:")
    print(met.describe())
    return met


def build_dataset() -> pd.DataFrame:
    print(f"Loading canonical dataset: {DATA}")
    df = pd.read_parquet(DATA)
    df["time"] = pd.to_datetime(df["time"]).dt.to_period("M").dt.to_timestamp()
    met = build_met_monthly_features()
    out = df.merge(met, on="time", how="left")
    missing = int(out[MET_FEATURES].isna().any(axis=1).sum())
    if missing:
        print(f"Rows with missing met features before dropna: {missing:,}")
    out = out.dropna(subset=MET_FEATURES).copy()
    out.to_parquet(MET_DATASET, index=False)
    out.head(10_000).to_csv(MET_DATASET.with_suffix(".sample.csv"), index=False)
    print(f"Wrote: {MET_DATASET} rows={len(out):,} cols={out.shape[1]}")
    return out


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


def monthly_dry_frame(frame: pd.DataFrame, prob_col: str) -> pd.DataFrame:
    return (
        frame.groupby("target_time")
        .agg(
            y_true_dry_frac=("is_dry", "mean"),
            pred_prob_dry=(prob_col, "mean"),
            clim_prob_dry=("clim_prob_dry", "mean"),
        )
        .reset_index()
    )


def monthly_bs(frame: pd.DataFrame, prob_col: str) -> float:
    monthly = monthly_dry_frame(frame, prob_col)
    return brier(monthly["y_true_dry_frac"].to_numpy(), monthly["pred_prob_dry"].to_numpy())


def run_experiment(df: pd.DataFrame, n_bootstrap: int) -> None:
    df = df.copy()
    df["year"] = df["year"].astype(int)
    features = get_feature_columns(df.columns) + MET_FEATURES

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

    print("Training XGBoost with ERA5-Land met features...")
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

    monthly_path = OUT_DIR / "met_feature_xgb_monthly_scores.csv"
    monthly.to_csv(monthly_path, index=False)

    model_path = OUT_DIR / "met_feature_xgb_model.json"
    model.save_model(model_path.as_posix())

    probs_path = OUT_DIR / "met_feature_xgb_test_probs.npz"
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
    ax.set_title("XGBoost + ERA5-Land temperature/VPD features")
    plt.tight_layout()
    fi_path = OUT_DIR / "met_feature_xgb_feature_importance.png"
    fig.savefig(fi_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    cm = confusion_matrix(test[TARGET].to_numpy(), y_pred, labels=[-1, 0, 1])
    metrics = (
        "ERA5-Land Temperature/VPD Feature Experiment\n"
        f"{'=' * 60}\n"
        "Design: canonical SPI-1[t+1] target, regional ERA5-Land t2m/VPD "
        "monthly anomaly lags at t and t-1.\n"
        f"Test months: {monthly['target_time'].nunique()}\n"
        f"Best iteration: {model.best_iteration}\n"
        f"Validation monthly BS by calibration: {val_bs_by_method}\n"
        f"Selected calibration: {best_method}\n"
        f"Features: {features}\n\n"
        "Monthly dry-fraction Brier Scores\n"
        f"  Climatology          : {bs_clim:.5f}\n"
        f"  XGBoost + met raw    : {bs_raw:.5f}\n"
        f"  XGBoost + met isotonic: {bs_iso:.5f}\n"
        f"  XGBoost + met Platt  : {bs_platt:.5f}\n"
        f"  XGBoost + met selected: {bs_selected:.5f}\n\n"
        "Brier Skill Score vs monthly climatology\n"
        f"  XGBoost + met raw    : {bss_raw:.5f} "
        f"(95% CI [{raw_ci[0]:.5f}, {raw_ci[1]:.5f}])\n"
        f"  XGBoost + met isotonic: {bss_iso:.5f} "
        f"(95% CI [{iso_ci[0]:.5f}, {iso_ci[1]:.5f}])\n"
        f"  XGBoost + met Platt  : {bss_platt:.5f} "
        f"(95% CI [{platt_ci[0]:.5f}, {platt_ci[1]:.5f}])\n"
        f"  XGBoost + met selected: {bss_selected:.5f} "
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
    scores_path = OUT_DIR / "met_feature_xgb_experiment_scores.txt"
    scores_path.write_text(metrics)
    print(metrics)


def main() -> None:
    args = parse_args()
    ensure_met_file(args.download_if_missing)
    if MET_DATASET.exists() and not args.rebuild_dataset:
        print(f"Loading existing met-feature dataset: {MET_DATASET}")
        df = pd.read_parquet(MET_DATASET)
        df["time"] = pd.to_datetime(df["time"])
    else:
        df = build_dataset()
    run_experiment(df, args.n_bootstrap)


if __name__ == "__main__":
    main()
