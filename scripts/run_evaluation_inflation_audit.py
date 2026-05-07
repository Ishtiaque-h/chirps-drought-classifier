#!/usr/bin/env python
"""Audit how common evaluation shortcuts inflate drought forecast skill.

The manuscript claim depends on strict verification choices: chronological
splits, leakage-safe targets, and monthly-level inference. This script makes
that choice visible by comparing the strict setup against deliberately invalid
or less defensible alternatives:

  1. strict SPI-1 lead-1, chronological split, monthly inference
  2. same strict model scored at pixel level, treating pixels as independent
  3. SPI-1 lead-1 with random row split, which mixes pixels from the same
     target months across train/validation/test
  4. overlapping SPI-3 lead-1 target, where feature windows and target
     accumulation windows share precipitation months
  5. leakage-safe SPI-3 lead-3 target as the target-aligned control

The invalid rows are not candidate forecast models. They quantify overclaim
risk and should be used as methodological evidence.
"""
from __future__ import annotations

from argparse import ArgumentParser, Namespace
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
import xarray as xr
import xgboost as xgb
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.utils.class_weight import compute_sample_weight

from build_dataset_seasonal import _build_one
from feature_config import BASE_FEATURES


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED = PROJECT_ROOT / "data" / "processed"
OUT_DIR = PROJECT_ROOT / "outputs"
REPORT_DIR = PROJECT_ROOT / "results" / "report"
AUDIT_DATA_DIR = PROCESSED / "audit"
OUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)
AUDIT_DATA_DIR.mkdir(parents=True, exist_ok=True)

FORECAST_DATASET = PROCESSED / "dataset_forecast.parquet"
STRICT_SPI3_DATASET = PROCESSED / "dataset_seasonal_spi3_lead3.parquet"
OVERLAP_SPI3_DATASET = AUDIT_DATA_DIR / "dataset_overlap_spi3_lead1.parquet"
PR_FILE = PROCESSED / "chirps_v3_monthly_cvalley_1991_2026.nc"
SPI_FILE = PROCESSED / "chirps_v3_monthly_cvalley_spi_1991_2026.nc"

LABEL_MAP = {-1: 0, 0: 1, 1: 2}
NINO34_FEATURES = ["nino34_lag1", "nino34_lag2"]


def parse_args() -> Namespace:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--forecast-dataset", type=Path, default=FORECAST_DATASET)
    parser.add_argument("--strict-spi3-dataset", type=Path, default=STRICT_SPI3_DATASET)
    parser.add_argument("--overlap-spi3-dataset", type=Path, default=OVERLAP_SPI3_DATASET)
    parser.add_argument("--rebuild-overlap-dataset", action="store_true")
    parser.add_argument("--n-bootstrap", type=int, default=2000)
    parser.add_argument("--n-pixel-bootstrap", type=int, default=300)
    parser.add_argument("--random-seed", type=int, default=20260506)
    parser.add_argument("--output-prefix", default="evaluation_inflation_audit")
    parser.add_argument("--copy-report", action="store_true")
    return parser.parse_args()


def brier(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean((np.asarray(p, dtype=float) - np.asarray(y, dtype=float)) ** 2))


def bss(y: np.ndarray, p: np.ndarray, ref: np.ndarray) -> float:
    ref_bs = brier(y, ref)
    return float(1.0 - brier(y, p) / ref_bs) if ref_bs > 0 else float("nan")


def bootstrap_arrays(
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


def add_target_time(df: pd.DataFrame, lead_months: int) -> pd.DataFrame:
    out = df.copy()
    if "target_time" in out.columns:
        out["target_time"] = pd.to_datetime(out["target_time"]).dt.to_period("M").dt.to_timestamp()
    else:
        out["target_time"] = (
            pd.to_datetime(out["time"]) + pd.DateOffset(months=lead_months)
        ).dt.to_period("M").dt.to_timestamp()
    out["target_year"] = out["target_time"].dt.year.astype(int)
    out["target_month"] = out["target_time"].dt.month.astype(int)
    return out


def feature_columns(df: pd.DataFrame) -> list[str]:
    missing = [col for col in BASE_FEATURES if col not in df.columns]
    if missing:
        raise ValueError(f"Missing base features: {missing}")
    return BASE_FEATURES + [col for col in NINO34_FEATURES if col in df.columns]


def prepare_frame(path: Path, target_col: str, lead_months: int) -> tuple[pd.DataFrame, list[str]]:
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_parquet(path)
    df["time"] = pd.to_datetime(df["time"]).dt.to_period("M").dt.to_timestamp()
    df = add_target_time(df, lead_months=lead_months)
    features = feature_columns(df)
    df = df.dropna(subset=features + [target_col, "target_time"]).copy()
    df[target_col] = df[target_col].astype(np.int8)
    return df, features


def build_overlap_spi3_dataset(path: Path, rebuild: bool) -> None:
    if path.exists() and not rebuild:
        return
    print("Building overlapping SPI-3 lead-1 audit dataset...")
    if not PR_FILE.exists() or not SPI_FILE.exists():
        raise FileNotFoundError(f"Missing CHIRPS/SPI source files: {PR_FILE}, {SPI_FILE}")

    pr_ds = xr.open_dataset(PR_FILE).load()
    spi_ds = xr.open_dataset(SPI_FILE).load()
    lat_name = "latitude" if "latitude" in pr_ds.coords else "lat"
    lon_name = "longitude" if "longitude" in pr_ds.coords else "lon"
    pr = pr_ds["pr"].sel(time=spi_ds.time)
    spi1 = spi_ds["spi1"].sel(time=pr.time)
    spi3 = spi_ds["spi3"].sel(time=pr.time)
    spi6 = spi_ds["spi6"].sel(time=pr.time)

    df = _build_one(
        pr=pr,
        spi1=spi1,
        spi3=spi3,
        spi6=spi6,
        spi_target=spi3,
        lead=1,
        spi_idx=3,
        lat_name=lat_name,
        lon_name=lon_name,
        climate_features="nino34",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    df.head(10_000).to_csv(path.with_suffix(".sample.csv"), index=False)
    print(f"Wrote {path} rows={len(df):,}")


def split_chronological(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train = df[df["target_year"] <= 2016].copy()
    val = df[(df["target_year"] >= 2017) & (df["target_year"] <= 2020)].copy()
    test = df[df["target_year"] >= 2021].copy()
    return train, val, test


def split_random_rows(
    df: pd.DataFrame,
    seed: int,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(df))
    n_train = int(len(df) * train_frac)
    n_val = int(len(df) * val_frac)
    train_idx = order[:n_train]
    val_idx = order[n_train:n_train + n_val]
    test_idx = order[n_train + n_val:]
    return df.iloc[train_idx].copy(), df.iloc[val_idx].copy(), df.iloc[test_idx].copy()


def add_climatology(train: pd.DataFrame, frames: list[pd.DataFrame], target_col: str) -> list[pd.DataFrame]:
    train_dry = train.assign(is_dry=(train[target_col] == -1).astype(float))
    month_clim = train_dry.groupby("target_month")["is_dry"].mean()
    global_clim = float(train_dry["is_dry"].mean())
    out = []
    for frame in frames:
        copy = frame.copy()
        copy["is_dry"] = (copy[target_col] == -1).astype(float)
        copy["clim_prob_dry"] = copy["target_month"].map(month_clim).fillna(global_clim)
        out.append(copy)
    return out


def train_xgb(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    features: list[str],
    target_col: str,
) -> tuple[pd.DataFrame, dict[str, object]]:
    y_train = train[target_col].map(LABEL_MAP).to_numpy()
    y_val = val[target_col].map(LABEL_MAP).to_numpy()
    y_test = test[target_col].map(LABEL_MAP).to_numpy()
    dtrain = xgb.DMatrix(
        train[features],
        label=y_train,
        weight=compute_sample_weight(class_weight="balanced", y=y_train),
        feature_names=features,
    )
    dval = xgb.DMatrix(val[features], label=y_val, feature_names=features)
    dtest = xgb.DMatrix(test[features], label=y_test, feature_names=features)
    params = {
        "objective": "multi:softprob",
        "num_class": 3,
        "eval_metric": "mlogloss",
        "tree_method": "hist",
        "device": "cpu",
        "eta": 0.05,
        "max_depth": 7,
        "min_child_weight": 5,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "lambda": 1.0,
        "alpha": 0.1,
    }
    model = xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=1400,
        evals=[(dtrain, "train"), (dval, "val")],
        early_stopping_rounds=40,
        verbose_eval=100,
    )
    iteration_range = (0, model.best_iteration + 1)
    dry_idx = LABEL_MAP[-1]
    val_raw = model.predict(dval, iteration_range=iteration_range).reshape(-1, 3)[:, dry_idx]
    test_raw = model.predict(dtest, iteration_range=iteration_range).reshape(-1, 3)[:, dry_idx]

    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    iso.fit(val_raw, val["is_dry"].to_numpy(dtype=float))
    val_iso = iso.predict(val_raw)
    test_iso = iso.predict(test_raw)

    platt = LogisticRegression(solver="lbfgs")
    platt.fit(val_raw.reshape(-1, 1), val["is_dry"].to_numpy(dtype=float))
    val_platt = platt.predict_proba(val_raw.reshape(-1, 1))[:, 1]
    test_platt = platt.predict_proba(test_raw.reshape(-1, 1))[:, 1]

    scored = test.copy()
    scored["raw_prob_dry"] = test_raw
    scored["isotonic_prob_dry"] = test_iso
    scored["platt_prob_dry"] = test_platt

    val_scored = val.copy()
    val_scored["raw_prob_dry"] = val_raw
    val_scored["isotonic_prob_dry"] = val_iso
    val_scored["platt_prob_dry"] = val_platt
    metadata = {
        "best_iteration": int(model.best_iteration),
        "feature_count": len(features),
        "top_features": "; ".join(
            pd.Series(
                {f: model.get_score(importance_type="gain").get(f, 0.0) for f in features}
            ).sort_values(ascending=False).head(6).index.tolist()
        ),
    }
    return scored, {"val": val_scored, **metadata}


def monthly_scores(frame: pd.DataFrame, pred_col: str) -> pd.DataFrame:
    return (
        frame.groupby("target_time", observed=True)
        .agg(
            y_true_dry_frac=("is_dry", "mean"),
            pred_prob_dry=(pred_col, "mean"),
            clim_prob_dry=("clim_prob_dry", "mean"),
            n_pixels=("is_dry", "size"),
        )
        .reset_index()
        .sort_values("target_time")
    )


def select_calibration(
    val_scored: pd.DataFrame,
    selection_level: str,
) -> tuple[str, dict[str, float]]:
    candidates = {
        "raw": "raw_prob_dry",
        "isotonic": "isotonic_prob_dry",
        "platt": "platt_prob_dry",
    }
    scores: dict[str, float] = {}
    if selection_level == "monthly":
        for method, col in candidates.items():
            monthly = monthly_scores(val_scored, col)
            scores[method] = brier(monthly["y_true_dry_frac"].to_numpy(), monthly["pred_prob_dry"].to_numpy())
    elif selection_level == "pixel":
        y = val_scored["is_dry"].to_numpy(dtype=float)
        for method, col in candidates.items():
            scores[method] = brier(y, val_scored[col].to_numpy(dtype=float))
    else:
        raise ValueError(f"Unknown selection_level: {selection_level}")
    return min(scores, key=scores.get), scores


def evaluate_level(
    scored: pd.DataFrame,
    pred_col: str,
    scenario: dict[str, object],
    inference_level: str,
    n_bootstrap: int,
    seed: int,
) -> dict[str, object]:
    if inference_level == "monthly":
        unit = monthly_scores(scored, pred_col)
        y = unit["y_true_dry_frac"].to_numpy(dtype=float)
        p = unit["pred_prob_dry"].to_numpy(dtype=float)
        ref = unit["clim_prob_dry"].to_numpy(dtype=float)
        n_units = int(len(unit))
        n_rows = int(unit["n_pixels"].sum())
        spearman = pd.Series(p).corr(pd.Series(y), method="spearman")
    elif inference_level == "pixel":
        y = scored["is_dry"].to_numpy(dtype=float)
        p = scored[pred_col].to_numpy(dtype=float)
        ref = scored["clim_prob_dry"].to_numpy(dtype=float)
        n_units = int(len(scored))
        n_rows = int(len(scored))
        spearman = pd.Series(p).corr(pd.Series(y), method="spearman")
    else:
        raise ValueError(f"Unknown inference_level: {inference_level}")
    lo, hi = bootstrap_arrays(y, p, ref, n_bootstrap=n_bootstrap, seed=seed)
    out = {
        **scenario,
        "inference_level": inference_level,
        "n_units_for_inference": n_units,
        "n_pixel_rows_scored": n_rows,
        "bs_reference": brier(y, ref),
        "bs_model": brier(y, p),
        "bss_vs_climatology": bss(y, p, ref),
        "bss_ci_low": lo,
        "bss_ci_high": hi,
        "spearman_obs_pred": float(spearman) if pd.notna(spearman) else np.nan,
        "pred_mean": float(np.mean(p)),
        "obs_mean": float(np.mean(y)),
        "bias": float(np.mean(p) - np.mean(y)),
    }
    return out


def run_scenario(
    name: str,
    df: pd.DataFrame,
    features: list[str],
    target_col: str,
    split_kind: str,
    target_description: str,
    target_overlap_months: int,
    protocol_validity: str,
    calibration_selection_level: str,
    n_bootstrap: int,
    n_pixel_bootstrap: int,
    seed: int,
) -> tuple[list[dict[str, object]], pd.DataFrame]:
    if split_kind == "chronological":
        train, val, test = split_chronological(df)
    elif split_kind == "random_rows":
        train, val, test = split_random_rows(df, seed=seed)
    else:
        raise ValueError(split_kind)
    train, val, test = add_climatology(train, [train, val, test], target_col)

    print(f"\n=== {name} ===")
    print(f"Split={split_kind}; train={train.shape}; val={val.shape}; test={test.shape}")
    scored, meta = train_xgb(train, val, test, features, target_col)
    best_method, val_bs = select_calibration(meta["val"], calibration_selection_level)
    pred_col = f"{best_method}_prob_dry"
    scored["selected_prob_dry"] = scored[pred_col]

    scenario_base = {
        "scenario": name,
        "target": target_description,
        "split_kind": split_kind,
        "protocol_validity": protocol_validity,
        "target_overlap_months": target_overlap_months,
        "calibration_selection_level": calibration_selection_level,
        "selected_calibration": best_method,
        "validation_bs_raw": val_bs["raw"],
        "validation_bs_isotonic": val_bs["isotonic"],
        "validation_bs_platt": val_bs["platt"],
        "best_iteration": meta["best_iteration"],
        "feature_count": meta["feature_count"],
        "top_features": meta["top_features"],
    }
    rows = [
        evaluate_level(
            scored,
            "selected_prob_dry",
            scenario_base,
            "monthly",
            n_bootstrap=n_bootstrap,
            seed=seed + 11,
        ),
        evaluate_level(
            scored,
            "selected_prob_dry",
            scenario_base,
            "pixel",
            n_bootstrap=n_pixel_bootstrap,
            seed=seed + 17,
        ),
    ]
    monthly = monthly_scores(scored, "selected_prob_dry")
    monthly["scenario"] = name
    monthly["selected_calibration"] = best_method
    return rows, monthly


def status_for_row(row: pd.Series) -> str:
    if row["inference_level"] == "pixel":
        return "invalid_pixel_inference"
    if str(row["protocol_validity"]).startswith("invalid"):
        return "invalid_protocol"
    bss_value = float(row["bss_vs_climatology"])
    lo = float(row["bss_ci_low"])
    hi = float(row["bss_ci_high"])
    if lo > 0:
        return "robust_positive"
    if hi < 0:
        return "robust_negative"
    if bss_value > 0:
        return "positive_uncertain"
    return "not_distinguishable_from_climatology"


def write_outputs(
    summary: pd.DataFrame,
    monthly: pd.DataFrame,
    output_prefix: str,
    copy_report: bool,
) -> None:
    summary["audit_status"] = summary.apply(status_for_row, axis=1)
    summary_path = OUT_DIR / f"{output_prefix}.csv"
    monthly_path = OUT_DIR / f"{output_prefix}_monthly_scores.csv"
    scores_path = OUT_DIR / f"{output_prefix}_summary.txt"
    summary.to_csv(summary_path, index=False)
    monthly.to_csv(monthly_path, index=False)

    display = summary.sort_values(
        ["protocol_validity", "scenario", "inference_level"],
        ascending=[True, True, True],
    )
    lines = [
        "Evaluation Inflation Audit",
        "=" * 72,
        "Invalid rows are methodological stress tests, not candidate forecasts.",
        "",
    ]
    for row in display.to_dict(orient="records"):
        lines.append(
            f"{row['scenario']:<30} {row['inference_level']:<7} "
            f"{row['protocol_validity']:<24} "
            f"BSS={row['bss_vs_climatology']:+.4f} "
            f"95% CI [{row['bss_ci_low']:+.4f}, {row['bss_ci_high']:+.4f}] "
            f"n={int(row['n_units_for_inference']):,} "
            f"cal={row['selected_calibration']} "
            f"status={row['audit_status']}"
        )
    lines.extend(
        [
            "",
            "Outputs:",
            f"  {summary_path}",
            f"  {monthly_path}",
        ]
    )
    scores_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))

    if copy_report:
        for path in [summary_path, monthly_path, scores_path]:
            shutil.copy2(path, REPORT_DIR / path.name)


def main() -> None:
    args = parse_args()
    build_overlap_spi3_dataset(args.overlap_spi3_dataset, args.rebuild_overlap_dataset)

    strict_spi1, strict_spi1_features = prepare_frame(args.forecast_dataset, "target_label", lead_months=1)
    overlap_spi3, overlap_features = prepare_frame(args.overlap_spi3_dataset, "target_label_spi3", lead_months=1)
    strict_spi3, strict_spi3_features = prepare_frame(args.strict_spi3_dataset, "target_label_spi3", lead_months=3)

    all_rows: list[dict[str, object]] = []
    all_monthly = []
    specs = [
        (
            "strict_spi1_chrono",
            strict_spi1,
            strict_spi1_features,
            "target_label",
            "chronological",
            "SPI-1 lead-1 dry fraction",
            0,
            "valid_primary_protocol",
            "monthly",
        ),
        (
            "random_spi1_rows",
            strict_spi1,
            strict_spi1_features,
            "target_label",
            "random_rows",
            "SPI-1 lead-1 dry fraction",
            0,
            "invalid_random_row_split",
            "pixel",
        ),
        (
            "overlap_spi3_lead1",
            overlap_spi3,
            overlap_features,
            "target_label_spi3",
            "chronological",
            "SPI-3 lead-1 dry fraction",
            2,
            "invalid_overlapping_target",
            "monthly",
        ),
        (
            "strict_spi3_lead3",
            strict_spi3,
            strict_spi3_features,
            "target_label_spi3",
            "chronological",
            "SPI-3 lead-3 dry fraction",
            0,
            "valid_target_aligned_control",
            "monthly",
        ),
    ]
    for i, spec in enumerate(specs):
        rows, monthly = run_scenario(
            *spec,
            n_bootstrap=args.n_bootstrap,
            n_pixel_bootstrap=args.n_pixel_bootstrap,
            seed=args.random_seed + 1000 * i,
        )
        all_rows.extend(rows)
        all_monthly.append(monthly)

    summary = pd.DataFrame(all_rows)
    monthly = pd.concat(all_monthly, ignore_index=True)
    write_outputs(summary, monthly, args.output_prefix, args.copy_report)


if __name__ == "__main__":
    main()
