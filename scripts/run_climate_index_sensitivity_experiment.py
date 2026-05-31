#!/usr/bin/env python
"""Run common-valid-period ENSO/PDO sensitivity tests.

This experiment closes a narrow reviewer-facing question: does excluding PDO
from the active checkpoint materially change the Central Valley SPI-1 lead-1
conclusion? It compares CHIRPS-only, Niño3.4-only, PDO-only, and
Niño3.4+PDO feature sets on the same rows/months, without forward-filling
missing climate-index tails.

Outputs:
  results/report/climate/climate_index_sensitivity_summary.csv
  results/report/climate/climate_index_sensitivity_monthly_scores.csv
  results/report/climate/climate_index_sensitivity_summary.md
"""
from __future__ import annotations

from argparse import ArgumentParser, Namespace
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import xgboost as xgb
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.utils.class_weight import compute_sample_weight

from feature_config import BASE_FEATURES


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED = PROJECT_ROOT / "data" / "processed"
DATASET = PROCESSED / "dataset_forecast.parquet"
CLIMATE = PROCESSED / "climate_indices_monthly.csv"
PR_FILE = PROCESSED / "chirps_v3_monthly_cvalley_1991_2026.nc"
SPI_FILE = PROCESSED / "chirps_v3_monthly_cvalley_spi_1991_2026.nc"
OUT_DIR = PROJECT_ROOT / "results" / "report" / "climate"

TARGET = "target_label"
LABEL_MAP = {-1: 0, 0: 1, 1: 2}
MISSING_SENTINELS = (-9.9, -99.99, -999.0, -9999.0)
CLIMATE_LAGS = ["nino34_lag1", "nino34_lag2", "pdo_lag1", "pdo_lag2"]
SPATIAL_FEATURES = ["spi1_nbr_mean", "spi3_nbr_mean", "spi6_nbr_mean", "pr_nbr_mean"]
VARIANTS = {
    "chirps_only": [],
    "nino34": ["nino34_lag1", "nino34_lag2"],
    "pdo": ["pdo_lag1", "pdo_lag2"],
    "nino34_pdo": CLIMATE_LAGS,
}


def parse_args() -> Namespace:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DATASET)
    parser.add_argument("--climate-file", type=Path, default=CLIMATE)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument(
        "--model-kind",
        choices=["tabular", "spatial", "both"],
        default="spatial",
        help="Spatial uses the same 3x3 neighborhood features as the canonical XGB-Spatial checkpoint.",
    )
    parser.add_argument("--n-bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-boost-round", type=int, default=2000)
    parser.add_argument("--early-stopping-rounds", type=int, default=50)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--verbose-eval", type=int, default=200)
    return parser.parse_args()


def project_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def add_target_month(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["time"] = pd.to_datetime(out["time"]).dt.to_period("M").dt.to_timestamp()
    out["target_month"] = (
        out["time"] + pd.DateOffset(months=1)
    ).dt.to_period("M").dt.to_timestamp()
    out["month"] = out["target_month"].dt.month
    out["year"] = out["target_month"].dt.year
    return out


def read_climate_lags(climate_file: Path, feature_times: pd.DatetimeIndex) -> pd.DataFrame:
    if not climate_file.exists():
        raise FileNotFoundError(f"Climate index file not found: {climate_file}")
    cdf = pd.read_csv(climate_file)
    required = {"time", "nino34", "pdo"}
    missing = required.difference(cdf.columns)
    if missing:
        raise ValueError(f"{climate_file} is missing required columns: {sorted(missing)}")

    cdf["time"] = pd.to_datetime(cdf["time"]).dt.to_period("M").dt.to_timestamp()
    cdf = (
        cdf[["time", "nino34", "pdo"]]
        .sort_values("time")
        .drop_duplicates("time", keep="last")
        .set_index("time")
    )
    cdf[["nino34", "pdo"]] = cdf[["nino34", "pdo"]].replace(
        list(MISSING_SENTINELS), np.nan
    )
    cdf = cdf.reindex(feature_times).sort_index()
    cdf[["nino34", "pdo"]] = cdf[["nino34", "pdo"]].interpolate(
        method="time", limit_area="inside"
    )
    cdf["nino34_lag1"] = cdf["nino34"]
    cdf["nino34_lag2"] = cdf["nino34"].shift(1)
    cdf["pdo_lag1"] = cdf["pdo"]
    cdf["pdo_lag2"] = cdf["pdo"].shift(1)
    return cdf[CLIMATE_LAGS].reset_index().rename(columns={"index": "time"})


def load_base_frame(dataset: Path, climate_file: Path) -> pd.DataFrame:
    dataset = project_path(dataset)
    climate_file = project_path(climate_file)
    df = pd.read_parquet(dataset)
    df = add_target_month(df)

    # Rebuild climate lags from the current source table so stale columns in an
    # older dataset artifact cannot affect the sensitivity result.
    df = df.drop(columns=[c for c in CLIMATE_LAGS if c in df.columns], errors="ignore")
    feature_times = pd.DatetimeIndex(sorted(pd.to_datetime(df["time"]).unique()))
    climate_lags = read_climate_lags(climate_file, feature_times)
    df = df.merge(climate_lags, on="time", how="left")

    needed = [TARGET] + BASE_FEATURES + CLIMATE_LAGS
    before = len(df)
    df = df.dropna(subset=needed).copy()
    print(
        "Applied common valid ENSO/PDO row mask: "
        f"{before:,} -> {len(df):,} rows"
    )
    return df


def add_spatial_features(df: pd.DataFrame) -> pd.DataFrame:
    print("Building 3x3 spatial-neighborhood features...")
    pr_ds = xr.open_dataset(PR_FILE).load()
    spi_ds = xr.open_dataset(SPI_FILE).load()

    pr = pr_ds["pr"].astype("float32")
    spi1 = spi_ds["spi1"].astype("float32").sel(time=pr.time)
    spi3 = spi_ds["spi3"].astype("float32").sel(time=pr.time)
    spi6 = spi_ds["spi6"].astype("float32").sel(time=pr.time)

    lat_name = "latitude" if "latitude" in pr.coords else "lat"
    lon_name = "longitude" if "longitude" in pr.coords else "lon"

    def nbr_mean(da: xr.DataArray, name: str) -> xr.DataArray:
        rolled = da.rolling({lat_name: 3, lon_name: 3}, min_periods=1, center=True).mean()
        rolled.name = name
        return rolled

    nbr_ds = xr.Dataset(
        {
            "spi1_nbr_mean": nbr_mean(spi1, "spi1_nbr_mean"),
            "spi3_nbr_mean": nbr_mean(spi3, "spi3_nbr_mean"),
            "spi6_nbr_mean": nbr_mean(spi6, "spi6_nbr_mean"),
            "pr_nbr_mean": nbr_mean(pr, "pr_nbr_mean"),
        }
    ).stack(pixel=(lat_name, lon_name))
    nbr_df = nbr_ds.reset_index("pixel").to_dataframe()
    if "time" not in nbr_df.columns:
        nbr_df = nbr_df.reset_index()
    nbr_df = nbr_df.rename(columns={lat_name: "latitude", lon_name: "longitude"})
    nbr_df["time"] = pd.to_datetime(nbr_df["time"]).dt.to_period("M").dt.to_timestamp()

    out = df.merge(
        nbr_df[["time", "latitude", "longitude"] + SPATIAL_FEATURES],
        on=["time", "latitude", "longitude"],
        how="left",
    )
    missing = int(out[SPATIAL_FEATURES].isna().sum().sum())
    if missing:
        print(f"Warning: filling {missing:,} missing spatial feature values with 0")
        out[SPATIAL_FEATURES] = out[SPATIAL_FEATURES].fillna(0.0)
    return out


def brier(y: np.ndarray, p: np.ndarray) -> float:
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    return float(np.mean((p - y) ** 2))


def bss(y: np.ndarray, p: np.ndarray, ref: np.ndarray) -> float:
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
        vals[i] = bss(y[sample], p[sample], ref[sample])
    lo, hi = np.nanquantile(vals, [0.025, 0.975])
    return float(lo), float(hi)


def claim_status(point: float, lo: float, hi: float) -> str:
    if not np.isfinite(point):
        return "not_applicable"
    if np.isfinite(lo) and np.isfinite(hi):
        if lo > 0:
            return "robust_positive"
        if hi < 0:
            return "robust_negative"
        if point > 0:
            return "positive_uncertain"
        return "negative_uncertain"
    return "positive_no_ci" if point > 0 else "negative_no_ci"


def train_monthly_climatology(train: pd.DataFrame) -> pd.Series:
    dry = train.assign(is_dry=(train[TARGET] == -1).astype(float))
    return dry.groupby("month")["is_dry"].mean()


def monthly_scores(
    frame: pd.DataFrame,
    pred: np.ndarray,
    train_clim: pd.Series,
) -> pd.DataFrame:
    tmp = frame[["target_month", "month", TARGET]].copy()
    tmp["y_dry"] = (tmp[TARGET] == -1).astype(float)
    tmp["pred_prob_dry"] = pred
    global_clim = float(train_clim.mean())
    tmp["clim_prob_dry"] = tmp["month"].map(train_clim).fillna(global_clim)
    return (
        tmp.groupby("target_month", as_index=False)
        .agg(
            month=("month", "first"),
            y_true_dry_frac=("y_dry", "mean"),
            pred_prob_dry=("pred_prob_dry", "mean"),
            clim_prob_dry=("clim_prob_dry", "mean"),
            n_pixels=("y_dry", "size"),
        )
        .sort_values("target_month")
    )


def monthly_bs_bss(monthly: pd.DataFrame) -> tuple[float, float, float]:
    y = monthly["y_true_dry_frac"].to_numpy(float)
    pred = monthly["pred_prob_dry"].to_numpy(float)
    ref = monthly["clim_prob_dry"].to_numpy(float)
    return brier(y, ref), brier(y, pred), bss(y, pred, ref)


def calibrate_predictions(
    y_val_enc: np.ndarray,
    val_raw: np.ndarray,
    test_raw: np.ndarray,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    y_val_dry = (y_val_enc == LABEL_MAP[-1]).astype(int)
    out: dict[str, tuple[np.ndarray, np.ndarray]] = {
        "raw": (val_raw, test_raw),
    }

    if len(np.unique(y_val_dry)) > 1:
        platt = LogisticRegression(solver="lbfgs", max_iter=1000)
        platt.fit(val_raw.reshape(-1, 1), y_val_dry)
        out["platt"] = (
            platt.predict_proba(val_raw.reshape(-1, 1))[:, 1],
            platt.predict_proba(test_raw.reshape(-1, 1))[:, 1],
        )

        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(val_raw, y_val_dry)
        out["isotonic"] = (iso.predict(val_raw), iso.predict(test_raw))

    return out


def amplitude_ratio(obs: np.ndarray, pred: np.ndarray) -> float:
    obs_std = float(np.std(obs))
    if obs_std == 0:
        return float("nan")
    return float(np.std(pred) / obs_std)


def top_features(model: xgb.Booster, features: list[str], n: int = 8) -> str:
    score = model.get_score(importance_type="gain")
    ranking = sorted(
        ((feature, float(score.get(feature, 0.0))) for feature in features),
        key=lambda item: item[1],
        reverse=True,
    )
    return "; ".join(feature for feature, gain in ranking[:n] if gain > 0)


def train_and_score(
    df: pd.DataFrame,
    variant: str,
    model_kind: str,
    args: Namespace,
) -> tuple[dict[str, object], pd.DataFrame]:
    climate_features = VARIANTS[variant]
    features = BASE_FEATURES + climate_features
    if model_kind == "spatial":
        features = features + SPATIAL_FEATURES

    train = df[df["year"] <= 2016].copy()
    val = df[(df["year"] >= 2017) & (df["year"] <= 2020)].copy()
    test = df[df["year"] >= 2021].copy()
    if train.empty or val.empty or test.empty:
        raise ValueError(
            f"Bad split for {variant}/{model_kind}: "
            f"train={train.shape}, val={val.shape}, test={test.shape}"
        )

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
        "device": args.device,
        "eta": 0.05,
        "max_depth": 8,
        "min_child_weight": 5,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "lambda": 1.0,
        "alpha": 0.1,
        "seed": args.seed,
    }
    print(
        f"\nTraining {model_kind} / {variant}: "
        f"features={len(features)} train={train.shape} val={val.shape} test={test.shape}"
    )
    model = xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=args.num_boost_round,
        evals=[(dtrain, "train"), (dval, "val")],
        early_stopping_rounds=args.early_stopping_rounds,
        verbose_eval=args.verbose_eval,
    )
    iteration_range = (0, int(model.best_iteration) + 1)
    val_probs = model.predict(dval, iteration_range=iteration_range).reshape(-1, 3)
    test_probs = model.predict(dtest, iteration_range=iteration_range).reshape(-1, 3)

    calibrations = calibrate_predictions(
        y_val_enc=y_val_enc,
        val_raw=val_probs[:, LABEL_MAP[-1]],
        test_raw=test_probs[:, LABEL_MAP[-1]],
    )
    train_clim = train_monthly_climatology(train)

    val_scores: dict[str, tuple[float, float, float]] = {}
    test_scores: dict[str, tuple[float, float, float]] = {}
    monthly_frames = []
    for calibration, (val_pred, test_pred) in calibrations.items():
        val_monthly = monthly_scores(val, val_pred, train_clim)
        test_monthly = monthly_scores(test, test_pred, train_clim)
        val_scores[calibration] = monthly_bs_bss(val_monthly)
        test_scores[calibration] = monthly_bs_bss(test_monthly)
        month_out = test_monthly.copy()
        month_out["variant"] = variant
        month_out["model_kind"] = model_kind
        month_out["calibration"] = calibration
        month_out["brier_gain_vs_clim"] = (
            (month_out["clim_prob_dry"] - month_out["y_true_dry_frac"]) ** 2
            - (month_out["pred_prob_dry"] - month_out["y_true_dry_frac"]) ** 2
        )
        monthly_frames.append(month_out)

    selected_calibration = min(
        val_scores,
        key=lambda name: val_scores[name][1],
    )
    selected_test = monthly_frames[
        list(calibrations.keys()).index(selected_calibration)
    ].copy()

    y = selected_test["y_true_dry_frac"].to_numpy(float)
    pred = selected_test["pred_prob_dry"].to_numpy(float)
    ref = selected_test["clim_prob_dry"].to_numpy(float)
    ci_low, ci_high = bootstrap_bss(
        y,
        pred,
        ref,
        n_bootstrap=args.n_bootstrap,
        seed=args.seed,
    )
    selected_bs_ref, selected_bs_model, selected_bss = monthly_bs_bss(selected_test)

    raw_bss = test_scores.get("raw", (np.nan, np.nan, np.nan))[2]
    platt_bss = test_scores.get("platt", (np.nan, np.nan, np.nan))[2]
    iso_bss = test_scores.get("isotonic", (np.nan, np.nan, np.nan))[2]
    row = {
        "model_kind": model_kind,
        "variant": variant,
        "climate_features": ",".join(climate_features) if climate_features else "none",
        "n_features": len(features),
        "n_train_rows": len(train),
        "n_val_rows": len(val),
        "n_test_rows": len(test),
        "n_test_months": int(selected_test["target_month"].nunique()),
        "test_target_start": selected_test["target_month"].min().date().isoformat(),
        "test_target_end": selected_test["target_month"].max().date().isoformat(),
        "best_iteration": int(model.best_iteration),
        "selected_calibration": selected_calibration,
        "bs_climatology": selected_bs_ref,
        "bs_selected": selected_bs_model,
        "selected_bss": selected_bss,
        "selected_bss_ci_low": ci_low,
        "selected_bss_ci_high": ci_high,
        "raw_bss": raw_bss,
        "platt_bss": platt_bss,
        "isotonic_bss": iso_bss,
        "val_raw_bss": val_scores.get("raw", (np.nan, np.nan, np.nan))[2],
        "val_platt_bss": val_scores.get("platt", (np.nan, np.nan, np.nan))[2],
        "val_isotonic_bss": val_scores.get("isotonic", (np.nan, np.nan, np.nan))[2],
        "spearman_obs_pred": float(pd.Series(y).corr(pd.Series(pred), method="spearman")),
        "amplitude_ratio": amplitude_ratio(y, pred),
        "claim_status": claim_status(selected_bss, ci_low, ci_high),
        "top_features": top_features(model, features),
    }
    all_monthly = pd.concat(monthly_frames, ignore_index=True)
    return row, all_monthly


def markdown_summary(summary: pd.DataFrame) -> str:
    cols = [
        "model_kind",
        "variant",
        "n_test_months",
        "selected_calibration",
        "selected_bss",
        "selected_bss_ci_low",
        "selected_bss_ci_high",
        "spearman_obs_pred",
        "amplitude_ratio",
        "claim_status",
    ]
    view = summary[cols].copy()
    for col in [
        "selected_bss",
        "selected_bss_ci_low",
        "selected_bss_ci_high",
        "spearman_obs_pred",
        "amplitude_ratio",
    ]:
        view[col] = view[col].map(lambda x: f"{float(x):+.3f}" if pd.notna(x) else "")
    table_lines = [
        "| " + " | ".join(view.columns) + " |",
        "| " + " | ".join(["---"] * len(view.columns)) + " |",
    ]
    for _, row in view.iterrows():
        table_lines.append(
            "| " + " | ".join("" if pd.isna(value) else str(value) for value in row) + " |"
        )
    lines = [
        "# Climate Index Sensitivity",
        "",
        "Common-valid-period comparison of CHIRPS-only, Niño3.4-only, PDO-only, "
        "and Niño3.4+PDO features. Rows use the same months after masking the "
        "current climate-index table without tail forward-fill.",
        "",
        "\n".join(table_lines),
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    args.out_dir = project_path(args.out_dir)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    df = load_base_frame(args.dataset, args.climate_file)
    model_kinds = ["tabular", "spatial"] if args.model_kind == "both" else [args.model_kind]
    if "spatial" in model_kinds:
        df = add_spatial_features(df)

    rows = []
    monthly_parts = []
    for model_kind in model_kinds:
        for variant in ["chirps_only", "nino34", "pdo", "nino34_pdo"]:
            row, monthly = train_and_score(df, variant, model_kind, args)
            rows.append(row)
            monthly_parts.append(monthly)

    summary = pd.DataFrame(rows).sort_values(["model_kind", "variant"]).reset_index(drop=True)
    monthly_scores_out = pd.concat(monthly_parts, ignore_index=True)

    summary_path = args.out_dir / "climate_index_sensitivity_summary.csv"
    monthly_path = args.out_dir / "climate_index_sensitivity_monthly_scores.csv"
    md_path = args.out_dir / "climate_index_sensitivity_summary.md"

    summary.to_csv(summary_path, index=False)
    monthly_scores_out.to_csv(monthly_path, index=False)
    md_path.write_text(markdown_summary(summary))

    print(f"\nWrote {summary_path}")
    print(f"Wrote {monthly_path}")
    print(f"Wrote {md_path}")
    print(summary[["model_kind", "variant", "selected_calibration", "selected_bss", "selected_bss_ci_low", "selected_bss_ci_high", "claim_status"]].to_string(index=False))


if __name__ == "__main__":
    main()
