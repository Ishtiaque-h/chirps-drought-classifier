#!/usr/bin/env python
"""
SHAP analysis for the corrected drought forecast XGBoost models.

Uses TreeExplainer on the current leakage-free tabular forecast dataset.  The
script can explain either the non-spatial XGBoost model, the XGBoost-Spatial
model, or both.  Spatial mode rebuilds the same 3x3 neighbourhood mean features
used by train_forecast_xgb_spatial.py without importing that training script.

Inputs:
  data/processed/dataset_forecast.parquet
  outputs/forecast_xgb_model.json
  outputs/xgb_spatial_model.json

Outputs:
  outputs/*_shap_importance_forecast.csv
  outputs/*_shap_summary_bar_forecast.png
  outputs/*_shap_beeswarm_dry_forecast.png
  outputs/*_shap_dependence_<feature>_dry.png
"""
from argparse import ArgumentParser
from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
from feature_config import get_feature_columns

DATA        = Path("data/processed/dataset_forecast.parquet")
PR_FILE     = Path("data/processed/chirps_v3_monthly_cvalley_1991_2026.nc")
SPI_FILE    = Path("data/processed/chirps_v3_monthly_cvalley_spi_1991_2026.nc")
OUT_DIR     = Path("outputs"); OUT_DIR.mkdir(exist_ok=True)
XGB_MODEL   = OUT_DIR / "forecast_xgb_model.json"
SP_MODEL    = OUT_DIR / "xgb_spatial_model.json"
SP_PROBS    = OUT_DIR / "xgb_spatial_test_probs.npz"
XGB_PROBS   = OUT_DIR / "forecast_xgb_test_probs.npz"

TARGET = "target_label"

label_map = {-1: 0, 0: 1, 1: 2}
DRY_IDX   = label_map[-1]   # 0
FEATURES_SPATIAL = [
    "spi1_nbr_mean",
    "spi3_nbr_mean",
    "spi6_nbr_mean",
    "pr_nbr_mean",
]


def parse_args() -> ArgumentParser:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        choices=["xgb", "xgb-spatial", "both"],
        default="xgb-spatial",
        help="Model artifact to explain. Default: xgb-spatial.",
    )
    parser.add_argument(
        "--n-per-class",
        type=int,
        default=700,
        help="Stratified test-set sample size per drought class.",
    )
    return parser.parse_args()


def load_dataset() -> pd.DataFrame:
    print("Loading dataset...")
    df = pd.read_parquet(DATA)
    df["time"] = pd.to_datetime(df["time"])
    df["year"] = df["year"].astype(int)
    return df


def add_spatial_features(df: pd.DataFrame) -> pd.DataFrame:
    print("Building spatial-neighbourhood features...")
    pr_ds = xr.open_dataset(PR_FILE).load()
    spi_ds = xr.open_dataset(SPI_FILE).load()

    pr = pr_ds["pr"].astype("float32")
    spi1 = spi_ds["spi1"].astype("float32").sel(time=pr.time.values)
    spi3 = spi_ds["spi3"].astype("float32").sel(time=pr.time.values)
    spi6 = spi_ds["spi6"].astype("float32").sel(time=pr.time.values)

    lat_name = "latitude" if "latitude" in pr.coords else "lat"
    lon_name = "longitude" if "longitude" in pr.coords else "lon"

    def nbr_mean(da: xr.DataArray, name: str) -> xr.DataArray:
        out = da.rolling({lat_name: 3, lon_name: 3}, min_periods=1, center=True).mean()
        out.name = name
        return out

    nbr_ds = xr.Dataset({
        "spi1_nbr_mean": nbr_mean(spi1, "spi1_nbr_mean"),
        "spi3_nbr_mean": nbr_mean(spi3, "spi3_nbr_mean"),
        "spi6_nbr_mean": nbr_mean(spi6, "spi6_nbr_mean"),
        "pr_nbr_mean": nbr_mean(pr, "pr_nbr_mean"),
    }).stack(pixel=(lat_name, lon_name))

    nbr_df = nbr_ds.reset_index("pixel").to_dataframe()
    if "time" not in nbr_df.columns:
        nbr_df = nbr_df.reset_index()
    nbr_df = nbr_df.rename(columns={lat_name: "latitude", lon_name: "longitude"})
    nbr_df["time"] = pd.to_datetime(nbr_df["time"])

    out = df.merge(
        nbr_df[["time", "latitude", "longitude"] + FEATURES_SPATIAL],
        on=["time", "latitude", "longitude"],
        how="left",
    )
    missing = int(out[FEATURES_SPATIAL].isna().sum().sum())
    if missing:
        print(f"  Warning: {missing:,} missing spatial feature values; filling with 0.")
        out[FEATURES_SPATIAL] = out[FEATURES_SPATIAL].fillna(0.0)
    print(f"  Spatial dataset shape: {out.shape}")
    return out


def saved_features(npz_path: Path) -> list[str] | None:
    if not npz_path.exists():
        return None
    with np.load(npz_path, allow_pickle=True) as arr:
        if "features" not in arr.files:
            return None
        return [str(f) for f in arr["features"].tolist()]


def model_config(model_name: str) -> tuple[Path, Path, str, str]:
    if model_name == "xgb":
        return XGB_MODEL, XGB_PROBS, "xgb", "XGBoost"
    if model_name == "xgb-spatial":
        return SP_MODEL, SP_PROBS, "xgb_spatial", "XGBoost-Spatial"
    raise ValueError(model_name)


def feature_columns(df: pd.DataFrame, model_name: str, npz_path: Path) -> list[str]:
    features = saved_features(npz_path)
    if features is not None:
        missing = [f for f in features if f not in df.columns]
        if missing:
            raise ValueError(f"Saved feature list contains missing columns: {missing}")
        return features
    features = get_feature_columns(df.columns)
    if model_name == "xgb-spatial":
        features = features + FEATURES_SPATIAL
    return features


def stratified_sample(test: pd.DataFrame, n_per_class: int) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    parts = []
    for _, grp in test.groupby(TARGET):
        n = min(len(grp), n_per_class)
        idx = rng.choice(len(grp), size=n, replace=False)
        parts.append(grp.iloc[idx])
    return pd.concat(parts, ignore_index=True)


def shap_values_by_class(model: xgb.Booster, X_sample: pd.DataFrame) -> list[np.ndarray]:
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)
    print("shap_values shape:", np.array(shap_values).shape)
    if isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
        return [shap_values[:, :, c] for c in range(shap_values.shape[2])]
    return list(shap_values)


def safe_feature_name(feature: str) -> str:
    return feature.replace("/", "_").replace(" ", "_").replace("-", "_")


def run_one(model_name: str, base_df: pd.DataFrame, n_per_class: int) -> None:
    model_path, npz_path, prefix, title = model_config(model_name)
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    df = add_spatial_features(base_df) if model_name == "xgb-spatial" else base_df.copy()
    features = feature_columns(df, model_name, npz_path)
    test = df[df["year"] >= 2021]

    print(f"Loading model from: {model_path}")
    model = xgb.Booster()
    model.load_model(model_path.as_posix())

    sample = stratified_sample(test, n_per_class)
    X_sample = sample[features].reset_index(drop=True)
    print(
        f"{title} SHAP sample: {X_sample.shape}  "
        f"classes: {sample[TARGET].value_counts().sort_index().to_dict()}"
    )

    print("Computing SHAP values...")
    sv_list = shap_values_by_class(model, X_sample)
    sv_dry = sv_list[DRY_IDX]

    mean_abs = np.mean([np.abs(sv) for sv in sv_list], axis=0)
    global_imp = (
        pd.Series(mean_abs.mean(axis=0), index=features, name="mean_abs_shap")
        .sort_values(ascending=False)
    )
    dry_imp = (
        pd.Series(np.abs(sv_dry).mean(axis=0), index=features, name="dry_mean_abs_shap")
        .sort_values(ascending=False)
    )
    imp_df = (
        pd.concat([global_imp, dry_imp], axis=1)
        .fillna(0.0)
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index()
        .rename(columns={"index": "feature"})
    )
    imp_path = OUT_DIR / f"{prefix}_shap_importance_forecast.csv"
    imp_df.to_csv(imp_path, index=False)

    print("Saving global SHAP bar plot...")
    top_imp = global_imp.sort_values(ascending=True).tail(15)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(top_imp.index, top_imp.values)
    ax.set_xlabel("Mean |SHAP value| (averaged over all classes)")
    ax.set_title(f"{title} — Global SHAP importance")
    plt.tight_layout()
    bar_path = OUT_DIR / f"{prefix}_shap_summary_bar_forecast.png"
    plt.savefig(bar_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print("Saving SHAP beeswarm (dry class)...")
    plt.figure()
    shap.summary_plot(sv_dry, X_sample, feature_names=features, show=False, max_display=15)
    plt.title(f"{title} SHAP — Dry class contributions")
    beeswarm_path = OUT_DIR / f"{prefix}_shap_beeswarm_dry_forecast.png"
    plt.tight_layout()
    plt.savefig(beeswarm_path, dpi=150, bbox_inches="tight")
    plt.close()

    preferred = [
        "nino34_lag1",
        "nino34_lag2",
        "month_sin",
        "spi1_lag1",
        "spi3_lag1",
        "pr_lag1",
        "spi1_nbr_mean",
    ]
    for feature in [f for f in preferred if f in features]:
        print(f"Saving SHAP dependence plot — {feature}...")
        plt.figure()
        shap.dependence_plot(
            feature, sv_dry, X_sample,
            feature_names=features, interaction_index=None, show=False,
        )
        plt.title(f"SHAP dependence — {feature} (dry class)")
        dep_path = OUT_DIR / f"{prefix}_shap_dependence_{safe_feature_name(feature)}_dry.png"
        plt.tight_layout()
        plt.savefig(dep_path, dpi=150, bbox_inches="tight")
        compat_name = {
            "spi1_lag1": "xgb_shap_dependence_spi1_dry.png",
            "spi3_lag1": "xgb_shap_dependence_spi3_dry.png",
        }.get(feature)
        if prefix == "xgb" and compat_name is not None:
            plt.savefig(OUT_DIR / compat_name, dpi=150, bbox_inches="tight")
        plt.close()

    print("Wrote:", imp_path)
    print("Wrote:", bar_path)
    print("Wrote:", beeswarm_path)


def main() -> None:
    args = parse_args()
    base_df = load_dataset()
    model_names = ["xgb", "xgb-spatial"] if args.model == "both" else [args.model]
    for model_name in model_names:
        run_one(model_name, base_df, args.n_per_class)
    print("Done.")


if __name__ == "__main__":
    main()
