#!/usr/bin/env python
"""
Compile SPI-12 regionalization diagnostics into paper-ready mechanism tables.

This script connects the diagnostic SPI-12 drought zones to the frozen
multi-region SPI-1 lead-1 forecast outputs. It does not retrain models. It
summarizes:
  1. PCA/k-means zone diagnostics, run-theory drought metrics, and strongest
     climate-index correlations.
  2. Zone-stratified forecast diagnostics for existing multi-region test
     probability files.

Outputs:
  results/regionalization/regionalization_mechanism_summary.csv
  results/regionalization/zone_forecast_diagnostics.csv
  results/regionalization/regionalization_mechanism_summary.md
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REG_RESULTS = PROJECT_ROOT / "results" / "regionalization"
REG_OUTPUTS = PROJECT_ROOT / "outputs" / "regionalization"
MULTIREGION_OUTPUTS = PROJECT_ROOT / "outputs" / "multiregion"
PROCESSED = PROJECT_ROOT / "data" / "processed"

TARGET = "target_label"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--regionalization-dir", type=Path, default=REG_RESULTS)
    parser.add_argument("--regionalization-output-dir", type=Path, default=REG_OUTPUTS)
    parser.add_argument("--multiregion-output-dir", type=Path, default=MULTIREGION_OUTPUTS)
    parser.add_argument("--out-dir", type=Path, default=REG_RESULTS)
    parser.add_argument("--models", nargs="+", default=["spatial", "tabular"])
    return parser.parse_args()


def round_coord(values: pd.Series | np.ndarray) -> np.ndarray:
    return np.round(np.asarray(values, dtype="float64"), 5)


def dataset_path_for_run(run_slug: str) -> Path | None:
    if run_slug == "cvalley":
        path = PROCESSED / "dataset_forecast.parquet"
    else:
        path = PROCESSED / "regions" / run_slug / f"dataset_forecast_{run_slug}.parquet"
    return path if path.exists() else None


def load_zone_frame(netcdf_path: Path) -> pd.DataFrame:
    ds = xr.open_dataset(netcdf_path)
    zone = ds["zone"].load()
    zdf = zone.to_dataframe(name="zone").reset_index()
    zdf = zdf[np.isfinite(zdf["zone"])].copy()
    zdf["zone"] = zdf["zone"].astype(int)
    zdf = zdf[zdf["zone"] >= 0].copy()
    zdf["lat_key"] = round_coord(zdf["latitude"])
    zdf["lon_key"] = round_coord(zdf["longitude"])
    return zdf[["lat_key", "lon_key", "zone"]]


def load_metadata(run_dir: Path) -> dict[str, object]:
    path = run_dir / "regionalization_metadata.json"
    return json.loads(path.read_text()) if path.exists() else {}


def strongest_correlations(corr_path: Path) -> pd.DataFrame:
    if not corr_path.exists():
        return pd.DataFrame()
    corr = pd.read_csv(corr_path)
    if corr.empty or "pearson_r" not in corr:
        return pd.DataFrame()
    lag_col = "index_lag_months" if "index_lag_months" in corr.columns else "lag_months"
    rows = []
    for zone, group in corr.dropna(subset=["pearson_r"]).groupby("zone"):
        top = group.iloc[group["pearson_r"].abs().argmax()]
        rows.append(
            {
                "zone": int(zone),
                "top_index": top.get("index", ""),
                "top_lag_months": int(top.get(lag_col, -1)),
                "top_pearson_r": float(top["pearson_r"]),
                "top_p_value": float(top.get("p_value", np.nan)),
            }
        )
    return pd.DataFrame(rows)


def mechanism_rows(run_dir: Path) -> list[dict[str, object]]:
    run_slug = run_dir.name
    metadata = load_metadata(run_dir)
    pca_path = run_dir / "pca_explained_variance.csv"
    counts_path = run_dir / "zone_pixel_counts.csv"
    metrics_path = run_dir / "zone_run_metrics.csv"
    corr_path = run_dir / "zone_climate_index_correlations.csv"

    if not counts_path.exists():
        return []

    pca_cum = np.nan
    if pca_path.exists():
        pca = pd.read_csv(pca_path)
        if "cumulative_explained_variance_ratio" in pca and not pca.empty:
            pca_cum = float(pca["cumulative_explained_variance_ratio"].iloc[-1])

    counts = pd.read_csv(counts_path)
    metrics = pd.read_csv(metrics_path) if metrics_path.exists() else pd.DataFrame({"zone": counts["zone"]})
    topcorr = strongest_correlations(corr_path)
    merged = counts.merge(metrics, on="zone", how="left").merge(topcorr, on="zone", how="left")

    region_meta = metadata.get("region", {}) if isinstance(metadata, dict) else {}
    args_meta = metadata.get("args", {}) if isinstance(metadata, dict) else {}
    rows = []
    for _, row in merged.iterrows():
        rows.append(
            {
                "run_slug": run_slug,
                "region": region_meta.get("slug", args_meta.get("region", "")) if isinstance(region_meta, dict) else "",
                "region_name": region_meta.get("name", "") if isinstance(region_meta, dict) else "",
                "mask_kind": args_meta.get("mask_kind", "") if isinstance(args_meta, dict) else "",
                "n_clusters": args_meta.get("n_clusters", np.nan) if isinstance(args_meta, dict) else np.nan,
                "pca_cumulative_explained_variance": pca_cum,
                "zone": int(row["zone"]),
                "n_pixels": int(row.get("n_pixels", 0)),
                "pixel_fraction": float(row.get("pixel_fraction", np.nan)),
                "n_drought_runs": int(row.get("n_runs", 0)) if pd.notna(row.get("n_runs", np.nan)) else np.nan,
                "mean_duration_months": float(row.get("mean_duration_months", np.nan)),
                "max_duration_months": float(row.get("max_duration_months", np.nan)),
                "mean_intensity": float(row.get("mean_intensity", np.nan)),
                "min_spi12_during_runs": float(row.get("min_spi12_during_runs", np.nan)),
                "top_index": row.get("top_index", ""),
                "top_lag_months": row.get("top_lag_months", np.nan),
                "top_pearson_r": row.get("top_pearson_r", np.nan),
                "top_p_value": row.get("top_p_value", np.nan),
            }
        )
    return rows


def assign_zones(df: pd.DataFrame, zone_frame: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["lat_key"] = round_coord(out["latitude"])
    out["lon_key"] = round_coord(out["longitude"])
    out = out.merge(zone_frame, on=["lat_key", "lon_key"], how="left")
    return out.drop(columns=["lat_key", "lon_key"])


def train_zone_climatology(run_slug: str, zone_frame: pd.DataFrame) -> pd.DataFrame | None:
    path = dataset_path_for_run(run_slug)
    if path is None:
        print(f"  No dataset found for {run_slug}; skipping zone climatology")
        return None
    cols = ["year", "month", "latitude", "longitude", TARGET]
    print(f"  Loading train labels for {run_slug}: {path.relative_to(PROJECT_ROOT)}")
    df = pd.read_parquet(path, columns=cols)
    df = assign_zones(df, zone_frame)
    df = df[df["zone"].notna()].copy()
    df["zone"] = df["zone"].astype(int)
    train = df[df["year"] <= 2016].copy()
    if train.empty:
        return None
    train["is_dry"] = (train[TARGET] == -1).astype(float)
    clim = (
        train.groupby(["zone", "month"], as_index=False)
        .agg(zone_clim_prob_dry=("is_dry", "mean"), n_train_pixels=("is_dry", "size"))
    )
    return clim


def brier_score(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean((y - p) ** 2))


def bss(y: np.ndarray, p: np.ndarray, ref: np.ndarray) -> float:
    ref_bs = brier_score(y, ref)
    return float(1.0 - brier_score(y, p) / ref_bs) if ref_bs > 0 else float("nan")


def safe_corr(a: pd.Series, b: pd.Series) -> float:
    if len(a) < 3 or a.std(ddof=0) == 0 or b.std(ddof=0) == 0:
        return float("nan")
    return float(a.corr(b))


def amplitude_ratio(obs: pd.Series, pred: pd.Series) -> float:
    denom = float(obs.std(ddof=0))
    if denom <= 0:
        return float("nan")
    return float(pred.std(ddof=0) / denom)


def zone_forecast_rows(
    run_slug: str,
    model_name: str,
    probs_path: Path,
    zone_frame: pd.DataFrame,
    clim: pd.DataFrame | None,
) -> list[dict[str, object]]:
    print(f"  Zone forecast diagnostics: {run_slug}/{model_name}")
    z = np.load(probs_path, allow_pickle=True)
    frame = pd.DataFrame(
        {
            "target_time": pd.to_datetime(z["target_times"]),
            "latitude": z["latitude"].astype(float),
            "longitude": z["longitude"].astype(float),
            "y_true": z["y_true"].astype(int),
            "dry_prob_raw": z["dry_probs_raw"].astype(float),
            "dry_prob_selected": z["dry_probs_selected"].astype(float),
        }
    )
    frame = assign_zones(frame, zone_frame)
    frame = frame[frame["zone"].notna()].copy()
    if frame.empty:
        return []
    frame["zone"] = frame["zone"].astype(int)
    frame["is_dry"] = (frame["y_true"] == -1).astype(float)
    frame["month"] = frame["target_time"].dt.month.astype(int)

    monthly = (
        frame.groupby(["zone", "target_time", "month"], as_index=False)
        .agg(
            obs_dry_frac=("is_dry", "mean"),
            pred_raw_dry_frac=("dry_prob_raw", "mean"),
            pred_selected_dry_frac=("dry_prob_selected", "mean"),
            n_test_pixels=("is_dry", "size"),
        )
        .sort_values(["zone", "target_time"])
    )
    if clim is not None and not clim.empty:
        monthly = monthly.merge(clim, on=["zone", "month"], how="left")
    else:
        monthly["zone_clim_prob_dry"] = monthly.groupby("zone")["obs_dry_frac"].transform("mean")
        monthly["n_train_pixels"] = np.nan

    rows = []
    best_calibration = str(z["best_calibration"]) if "best_calibration" in z.files else ""
    for zone, group in monthly.groupby("zone"):
        g = group.dropna(subset=["zone_clim_prob_dry"]).copy()
        if g.empty:
            continue
        y = g["obs_dry_frac"].to_numpy()
        pred = g["pred_selected_dry_frac"].to_numpy()
        raw = g["pred_raw_dry_frac"].to_numpy()
        ref = g["zone_clim_prob_dry"].to_numpy()
        rows.append(
            {
                "run_slug": run_slug,
                "model": model_name,
                "zone": int(zone),
                "selected_calibration": best_calibration,
                "n_test_months": int(g["target_time"].nunique()),
                "mean_test_pixels_per_month": float(g["n_test_pixels"].mean()),
                "n_train_pixels_for_climatology": float(g["n_train_pixels"].sum()) if "n_train_pixels" in g else np.nan,
                "obs_dry_mean": float(g["obs_dry_frac"].mean()),
                "pred_selected_dry_mean": float(g["pred_selected_dry_frac"].mean()),
                "pred_raw_dry_mean": float(g["pred_raw_dry_frac"].mean()),
                "zone_clim_dry_mean": float(g["zone_clim_prob_dry"].mean()),
                "selected_bias": float((g["pred_selected_dry_frac"] - g["obs_dry_frac"]).mean()),
                "raw_bias": float((g["pred_raw_dry_frac"] - g["obs_dry_frac"]).mean()),
                "selected_corr": safe_corr(g["obs_dry_frac"], g["pred_selected_dry_frac"]),
                "raw_corr": safe_corr(g["obs_dry_frac"], g["pred_raw_dry_frac"]),
                "selected_amplitude_ratio": amplitude_ratio(g["obs_dry_frac"], g["pred_selected_dry_frac"]),
                "raw_amplitude_ratio": amplitude_ratio(g["obs_dry_frac"], g["pred_raw_dry_frac"]),
                "selected_bs": brier_score(y, pred),
                "raw_bs": brier_score(y, raw),
                "zone_clim_bs": brier_score(y, ref),
                "selected_bss_vs_zone_climatology": bss(y, pred, ref),
                "raw_bss_vs_zone_climatology": bss(y, raw, ref),
                "months_obs_dry_frac_ge_0_25": int((g["obs_dry_frac"] >= 0.25).sum()),
                "months_obs_dry_frac_ge_0_50": int((g["obs_dry_frac"] >= 0.50).sum()),
            }
        )
    return rows


def write_markdown(mechanism: pd.DataFrame, zone_diag: pd.DataFrame, out_dir: Path) -> None:
    def markdown_table(df: pd.DataFrame) -> str:
        if df.empty:
            return ""
        work = df.copy()
        headers = list(work.columns)
        rows = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
        for _, row in work.iterrows():
            vals = []
            for value in row:
                if isinstance(value, float):
                    vals.append(f"{value:.3f}")
                else:
                    vals.append(str(value))
            rows.append("| " + " | ".join(vals) + " |")
        return "\n".join(rows)

    lines = [
        "# SPI-12 Regionalization Mechanism Summary",
        "",
        "Method context: diagnostic SPI-12 regionalization follows the same broad structure as recent CHIRPS drought regionalization work: SPI-12, PCA/k-means zones, run-theory drought metrics, and climate-index correlation checks. These diagnostics do not alter the frozen SPI-1 lead-1 forecast target.",
        "",
        "## Zone Mechanism Highlights",
        "",
    ]
    if not mechanism.empty:
        top = mechanism.copy()
        top["abs_top_r"] = top["top_pearson_r"].abs()
        top = top.sort_values(["run_slug", "abs_top_r"], ascending=[True, False])
        lines.append(
            markdown_table(
                top[
                    [
                        "run_slug",
                        "zone",
                        "pixel_fraction",
                        "max_duration_months",
                        "mean_intensity",
                        "top_index",
                        "top_lag_months",
                        "top_pearson_r",
                    ]
                ].round(3)
            )
        )
    lines.extend(["", "## Zone Forecast Diagnostics", ""])
    if not zone_diag.empty:
        selected = zone_diag.sort_values(
            ["run_slug", "model", "selected_bss_vs_zone_climatology"],
            ascending=[True, True, False],
        )
        lines.append(
            markdown_table(
                selected[
                    [
                        "run_slug",
                        "model",
                        "zone",
                        "obs_dry_mean",
                        "pred_selected_dry_mean",
                        "selected_bias",
                        "selected_corr",
                        "selected_bss_vs_zone_climatology",
                    ]
                ].round(3)
            )
        )
    (out_dir / "regionalization_mechanism_summary.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    mechanism_all: list[dict[str, object]] = []
    zone_diag_all: list[dict[str, object]] = []

    for run_dir in sorted(args.regionalization_dir.iterdir()):
        if not run_dir.is_dir():
            continue
        run_slug = run_dir.name
        netcdf_path = args.regionalization_output_dir / run_slug / "spi12_regionalization.nc"
        if not netcdf_path.exists():
            continue
        print(f"Processing {run_slug}")
        mechanism_all.extend(mechanism_rows(run_dir))
        zone_frame = load_zone_frame(netcdf_path)
        clim = train_zone_climatology(run_slug, zone_frame)
        for model_name in args.models:
            probs_path = args.multiregion_output_dir / run_slug / f"{model_name}_test_probs.npz"
            if probs_path.exists():
                zone_diag_all.extend(zone_forecast_rows(run_slug, model_name, probs_path, zone_frame, clim))

    mechanism = pd.DataFrame(mechanism_all)
    zone_diag = pd.DataFrame(zone_diag_all)
    mechanism_path = args.out_dir / "regionalization_mechanism_summary.csv"
    zone_diag_path = args.out_dir / "zone_forecast_diagnostics.csv"
    mechanism.to_csv(mechanism_path, index=False)
    zone_diag.to_csv(zone_diag_path, index=False)
    write_markdown(mechanism, zone_diag, args.out_dir)

    print(f"Wrote {mechanism_path}")
    print(f"Wrote {zone_diag_path}")
    print(f"Wrote {args.out_dir / 'regionalization_mechanism_summary.md'}")


if __name__ == "__main__":
    main()
