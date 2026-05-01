#!/usr/bin/env python
"""
XGBoost-Spatial experiment with gridded ERA5-Land temperature/VPD anomalies.

This is an isolated experiment. It does not overwrite:
  - data/processed/dataset_forecast.parquet
  - outputs/xgb_spatial_model.json
  - outputs/xgb_spatial_test_probs.npz

Feature design:
  - Canonical target remains SPI-1 class at t+1.
  - CHIRPS neighborhood features are the same 3x3 means used by
    train_forecast_xgb_spatial.py at feature month t.
  - ERA5-Land t2m and VPD anomalies are computed per ERA5 grid cell relative
    to 1991-2020 calendar-month climatology, interpolated to the CHIRPS grid,
    and used at t and t-1.
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
PR_FILE = PROCESSED / "chirps_v3_monthly_cvalley_1991_2026.nc"
SPI_FILE = PROCESSED / "chirps_v3_monthly_cvalley_spi_1991_2026.nc"
MET_FILE = PROCESSED / f"era5_land_met_monthly_cvalley_{START_YEAR}_{CURRENT_YEAR}.nc"
DOWNLOAD_SCRIPT = PROJECT_ROOT / "scripts" / "download_era5_land_met_monthly.py"

TARGET = "target_label"
LABEL_MAP = {-1: 0, 0: 1, 1: 2}
INV_LABEL_MAP = {v: k for k, v in LABEL_MAP.items()}

CHIRPS_SPATIAL_FEATURES = [
    "spi1_nbr_mean",
    "spi3_nbr_mean",
    "spi6_nbr_mean",
    "pr_nbr_mean",
]
MET_SPATIAL_FEATURES = [
    "t2m_anom_grid_lag1",
    "t2m_anom_grid_lag2",
    "vpd_anom_grid_lag1",
    "vpd_anom_grid_lag2",
]


def parse_args() -> ArgumentParser:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument(
        "--download-if-missing",
        action="store_true",
        help="Run download_era5_land_met_monthly.py if the met NetCDF is missing.",
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
    subprocess.run([sys.executable, str(DOWNLOAD_SCRIPT)], cwd=PROJECT_ROOT, check=True)
    if not MET_FILE.exists():
        raise FileNotFoundError(f"Download finished, but file is still missing: {MET_FILE}")


def choose_var(ds: xr.Dataset, candidates: list[str]) -> str:
    for candidate in candidates:
        if candidate in ds.data_vars:
            return candidate
    lower = {name.lower(): name for name in ds.data_vars}
    for candidate in candidates:
        if candidate.lower() in lower:
            return lower[candidate.lower()]
    raise ValueError(f"Could not find any of {candidates}; available={list(ds.data_vars)}")


def standardize_da(ds: xr.Dataset, var_name: str) -> xr.DataArray:
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
    if units in {"k", "kelvin"} or np.nanmedian(da.values) > 100:
        return da - 273.15
    return da


def saturation_vapor_pressure_kpa(temp_c: xr.DataArray) -> xr.DataArray:
    return 0.6108 * np.exp((17.27 * temp_c) / (temp_c + 237.3))


def fill_spatial_nans(da: xr.DataArray) -> xr.DataArray:
    """Fill ERA5-Land land/sea-mask gaps from nearest valid spatial neighbors."""
    original_lat = da["latitude"]
    original_lon = da["longitude"]
    filled = (
        da.sortby("latitude").sortby("longitude")
        .interpolate_na("longitude", method="nearest", fill_value="extrapolate")
        .interpolate_na("latitude", method="nearest", fill_value="extrapolate")
    )
    return filled.sel(latitude=original_lat, longitude=original_lon)


def nbr_mean(da: xr.DataArray, name: str) -> xr.DataArray:
    lat_name = "latitude" if "latitude" in da.coords else "lat"
    lon_name = "longitude" if "longitude" in da.coords else "lon"
    out = da.rolling({lat_name: 3, lon_name: 3}, min_periods=1, center=True).mean()
    out.name = name
    return out


def flat_dataset(ds: xr.Dataset) -> pd.DataFrame:
    lat_name = "latitude" if "latitude" in ds.coords else "lat"
    lon_name = "longitude" if "longitude" in ds.coords else "lon"
    flat = ds.stack(pixel=(lat_name, lon_name)).reset_index("pixel").to_dataframe()
    if "time" not in flat.columns:
        flat = flat.reset_index()
    flat = flat.rename(columns={lat_name: "latitude", lon_name: "longitude"})
    flat["time"] = pd.to_datetime(flat["time"]).dt.to_period("M").dt.to_timestamp()
    return flat


def build_chirps_spatial_features() -> pd.DataFrame:
    print("Building CHIRPS neighborhood features...")
    pr_ds = xr.open_dataset(PR_FILE).load()
    spi_ds = xr.open_dataset(SPI_FILE).load()

    pr = pr_ds["pr"].astype("float32")
    spi1 = spi_ds["spi1"].astype("float32").sel(time=pr.time)
    spi3 = spi_ds["spi3"].astype("float32").sel(time=pr.time)
    spi6 = spi_ds["spi6"].astype("float32").sel(time=pr.time)

    ds = xr.Dataset({
        "spi1_nbr_mean": nbr_mean(spi1, "spi1_nbr_mean"),
        "spi3_nbr_mean": nbr_mean(spi3, "spi3_nbr_mean"),
        "spi6_nbr_mean": nbr_mean(spi6, "spi6_nbr_mean"),
        "pr_nbr_mean": nbr_mean(pr, "pr_nbr_mean"),
    })
    out = flat_dataset(ds)
    print(f"  CHIRPS spatial feature table: {out.shape}")
    return out


def build_met_spatial_features() -> pd.DataFrame:
    print("Building gridded ERA5-Land temperature/VPD anomaly features...")
    pr_ds = xr.open_dataset(PR_FILE)
    target_lat = pr_ds["latitude"]
    target_lon = pr_ds["longitude"]

    met_ds = xr.open_dataset(MET_FILE).load()
    t_var = choose_var(met_ds, ["t2m", "2m_temperature"])
    d_var = choose_var(met_ds, ["d2m", "2m_dewpoint_temperature"])

    t2m_c = fill_spatial_nans(to_celsius(standardize_da(met_ds, t_var)))
    d2m_c = fill_spatial_nans(to_celsius(standardize_da(met_ds, d_var))).sel(time=t2m_c.time)
    vpd_kpa = (
        saturation_vapor_pressure_kpa(t2m_c)
        - saturation_vapor_pressure_kpa(d2m_c)
    ).clip(min=0.0)

    baseline = (t2m_c["time"].dt.year >= 1991) & (t2m_c["time"].dt.year <= 2020)
    t2m_clim = t2m_c.where(baseline, drop=True).groupby("time.month").mean("time")
    vpd_clim = vpd_kpa.where(baseline, drop=True).groupby("time.month").mean("time")

    t2m_anom = t2m_c.groupby("time.month") - t2m_clim
    vpd_anom = vpd_kpa.groupby("time.month") - vpd_clim

    # xarray interpolation is simplest and safest when both axes are ascending.
    t2m_anom = t2m_anom.sortby("latitude").sortby("longitude")
    vpd_anom = vpd_anom.sortby("latitude").sortby("longitude")
    target_lat_sorted = target_lat.sortby(target_lat)
    target_lon_sorted = target_lon.sortby(target_lon)

    def interp_to_chirps(da: xr.DataArray) -> xr.DataArray:
        linear = da.interp(
            latitude=target_lat_sorted,
            longitude=target_lon_sorted,
            method="linear",
        )
        nearest = da.interp(
            latitude=target_lat_sorted,
            longitude=target_lon_sorted,
            method="nearest",
        )
        # Linear interpolation leaves a thin NaN rim when CHIRPS cell centers sit
        # just outside the ERA5 coordinate span. Nearest is used only as a rim
        # fallback so we keep the full canonical CHIRPS grid.
        return linear.combine_first(nearest).sel(latitude=target_lat, longitude=target_lon)

    t2m_interp = interp_to_chirps(t2m_anom)
    vpd_interp = interp_to_chirps(vpd_anom)

    ds = xr.Dataset({
        "t2m_anom_grid_lag1": t2m_interp.astype("float32"),
        "t2m_anom_grid_lag2": t2m_interp.shift(time=1).astype("float32"),
        "vpd_anom_grid_lag1": vpd_interp.astype("float32"),
        "vpd_anom_grid_lag2": vpd_interp.shift(time=1).astype("float32"),
    })
    out = flat_dataset(ds)
    print(f"  ERA5 spatial-met feature table: {out.shape}")
    print(out[MET_SPATIAL_FEATURES].describe())
    return out


def build_feature_frame() -> pd.DataFrame:
    print(f"Loading canonical dataset: {DATA}")
    df = pd.read_parquet(DATA)
    df["time"] = pd.to_datetime(df["time"]).dt.to_period("M").dt.to_timestamp()
    df["year"] = df["year"].astype(int)

    chirps_sp = build_chirps_spatial_features()
    met_sp = build_met_spatial_features()

    keys = ["time", "latitude", "longitude"]
    out = df.merge(chirps_sp[keys + CHIRPS_SPATIAL_FEATURES], on=keys, how="left")
    out = out.merge(met_sp[keys + MET_SPATIAL_FEATURES], on=keys, how="left")

    missing_spatial = int(out[CHIRPS_SPATIAL_FEATURES].isna().sum().sum())
    missing_met_rows = int(out[MET_SPATIAL_FEATURES].isna().any(axis=1).sum())
    if missing_spatial:
        print(f"  Warning: {missing_spatial:,} missing CHIRPS spatial values; filling with 0.")
        out[CHIRPS_SPATIAL_FEATURES] = out[CHIRPS_SPATIAL_FEATURES].fillna(0.0)
    if missing_met_rows:
        print(f"  Rows with missing met spatial values before dropna: {missing_met_rows:,}")
    out = out.dropna(subset=MET_SPATIAL_FEATURES).copy()
    print(f"  Final experiment frame: {out.shape}")
    return out


def brier(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean((p - y) ** 2))


def bss(y: np.ndarray, p: np.ndarray, ref: np.ndarray) -> float:
    ref_bs = brier(y, ref)
    return float(1.0 - brier(y, p) / ref_bs) if ref_bs > 0 else float("nan")


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


def add_calibrated_dry_probs(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    probs_val: np.ndarray,
    probs_test: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame, str, dict[str, float]]:
    train = train.copy()
    val = val.copy()
    test = test.copy()

    train["is_dry"] = (train[TARGET] == -1).astype(float)
    val["is_dry"] = (val[TARGET] == -1).astype(float)
    test["is_dry"] = (test[TARGET] == -1).astype(float)
    val["target_time"] = target_month(val["time"])
    test["target_time"] = target_month(test["time"])

    train_monthly_dry = train.groupby("month")["is_dry"].mean()
    global_dry = float(train["is_dry"].mean())
    val["clim_prob_dry"] = val["month"].map(train_monthly_dry).fillna(global_dry)
    test["clim_prob_dry"] = test["month"].map(train_monthly_dry).fillna(global_dry)

    dry_idx = LABEL_MAP[-1]
    val_raw = probs_val[:, dry_idx]
    test_raw = probs_test[:, dry_idx]

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
    val_bs_by_method = {method: monthly_bs(val, col) for method, col in calibration_cols.items()}
    best_method = min(val_bs_by_method, key=val_bs_by_method.get)
    test["xgb_selected_prob_dry"] = test[calibration_cols[best_method]]
    return val, test, best_method, val_bs_by_method


def run_experiment(df: pd.DataFrame, n_bootstrap: int) -> None:
    features = get_feature_columns(df.columns) + CHIRPS_SPATIAL_FEATURES + MET_SPATIAL_FEATURES
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

    print("Training XGBoost-Spatial with gridded ERA5-Land met features...")
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
    probs_test = model.predict(dtest, iteration_range=iteration_range).reshape(-1, 3)

    val, test, best_method, val_bs_by_method = add_calibrated_dry_probs(
        train, val, test, probs_val, probs_test
    )

    y_pred = np.array([INV_LABEL_MAP[i] for i in probs_test.argmax(axis=1)])
    report = classification_report(
        test[TARGET].to_numpy(),
        y_pred,
        target_names=["dry(-1)", "normal(0)", "wet(+1)"],
        digits=3,
    )
    print(report)

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
    metrics = {}
    for method, col in {
        "raw": "xgb_raw_prob_dry",
        "isotonic": "xgb_isotonic_prob_dry",
        "platt": "xgb_platt_prob_dry",
        "selected": "xgb_selected_prob_dry",
    }.items():
        pred = monthly[col].to_numpy()
        metrics[method] = {
            "bs": brier(y, pred),
            "bss": bss(y, pred, clim),
            "ci": bootstrap_bss(monthly, col, "clim_prob_dry", n_bootstrap),
        }
    bs_clim = brier(y, clim)

    monthly_path = OUT_DIR / "met_spatial_xgb_monthly_scores.csv"
    monthly.to_csv(monthly_path, index=False)

    model_path = OUT_DIR / "met_spatial_xgb_model.json"
    model.save_model(model_path.as_posix())

    probs_path = OUT_DIR / "met_spatial_xgb_test_probs.npz"
    selected_col = {
        "none": "xgb_raw_prob_dry",
        "isotonic": "xgb_isotonic_prob_dry",
        "platt": "xgb_platt_prob_dry",
    }[best_method]
    np.savez_compressed(
        probs_path,
        probs=probs_test.astype("float32"),
        dry_probs_raw=test["xgb_raw_prob_dry"].to_numpy(dtype="float32"),
        dry_probs_isotonic=test["xgb_isotonic_prob_dry"].to_numpy(dtype="float32"),
        dry_probs_platt=test["xgb_platt_prob_dry"].to_numpy(dtype="float32"),
        dry_probs_selected=test[selected_col].to_numpy(dtype="float32"),
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
    imp = pd.Series({f: importance.get(f, 0.0) for f in features}).sort_values(ascending=True)
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.barh(imp.index, imp.values)
    ax.set_xlabel("Gain")
    ax.set_title("XGBoost-Spatial + gridded ERA5-Land temperature/VPD features")
    plt.tight_layout()
    fi_path = OUT_DIR / "met_spatial_xgb_feature_importance.png"
    fig.savefig(fi_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    cm = confusion_matrix(test[TARGET].to_numpy(), y_pred, labels=[-1, 0, 1])
    lines = [
        "XGBoost-Spatial + Gridded ERA5-Land Temperature/VPD Experiment",
        "=" * 72,
        "Design: canonical SPI-1[t+1] target; CHIRPS 3x3 neighborhood features; "
        "ERA5-Land t2m/VPD per-grid-cell anomaly lags interpolated to CHIRPS at t and t-1.",
        f"Test months: {monthly['target_time'].nunique()}",
        f"Best iteration: {model.best_iteration}",
        f"Validation monthly BS by calibration: {val_bs_by_method}",
        f"Selected calibration: {best_method}",
        f"Features: {features}",
        "",
        "Monthly dry-fraction Brier Scores",
        f"  Climatology              : {bs_clim:.5f}",
    ]
    for method in ["raw", "isotonic", "platt", "selected"]:
        label = f"XGB-Spatial + met {method}"
        lines.append(f"  {label:<27}: {metrics[method]['bs']:.5f}")
    lines.extend(["", "Brier Skill Score vs monthly climatology"])
    for method in ["raw", "isotonic", "platt", "selected"]:
        label = f"XGB-Spatial + met {method}"
        ci = metrics[method]["ci"]
        lines.append(
            f"  {label:<27}: {metrics[method]['bss']:.5f} "
            f"(95% CI [{ci[0]:.5f}, {ci[1]:.5f}])"
        )
    lines.extend([
        "",
        "Pixel-level classification report (secondary diagnostic)",
        report,
        "Pixel-level confusion matrix labels [-1, 0, 1]",
        str(cm),
        "",
        "Outputs:",
        f"  {monthly_path}",
        f"  {model_path}",
        f"  {probs_path}",
        f"  {fi_path}",
    ])
    text = "\n".join(lines) + "\n"
    scores_path = OUT_DIR / "met_spatial_xgb_experiment_scores.txt"
    scores_path.write_text(text)
    print(text)


def main() -> None:
    args = parse_args()
    ensure_met_file(args.download_if_missing)
    df = build_feature_frame()
    run_experiment(df, args.n_bootstrap)


if __name__ == "__main__":
    main()
