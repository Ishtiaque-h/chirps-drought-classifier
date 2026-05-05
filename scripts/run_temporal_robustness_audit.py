#!/usr/bin/env python
"""
Temporal robustness audit for the canonical Central Valley SPI-1 lead-1 task.

This script tests whether the headline 2021-2026 result is an artifact of a
single non-representative holdout period. It retrains the same tabular XGBoost
design across rolling chronological splits, applies validation-only calibration,
and evaluates monthly dry-fraction BSS against multiple climatology references.

Outputs:
  results/temporal/temporal_robustness_monthly_predictions.csv
  results/temporal/temporal_robustness_summary.csv
  results/temporal/temporal_robustness_event_blocks.csv
  results/temporal/temporal_robustness_audit.txt
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.utils.class_weight import compute_sample_weight

from feature_config import get_feature_columns


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET = PROJECT_ROOT / "data" / "processed" / "dataset_forecast.parquet"
OUT_DIR = PROJECT_ROOT / "results" / "temporal"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET = "target_label"
LABEL_MAP = {-1: 0, 0: 1, 1: 2}
DRY_IDX = LABEL_MAP[-1]


@dataclass(frozen=True)
class SplitSpec:
    name: str
    train_end: int
    val_start: int
    val_end: int
    test_start: int
    test_end: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DATASET)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--n-bootstrap", type=int, default=1000)
    parser.add_argument("--num-boost-round", type=int, default=900)
    parser.add_argument("--early-stopping-rounds", type=int, default=40)
    parser.add_argument("--max-depth", type=int, default=6)
    parser.add_argument("--eta", type=float, default=0.05)
    parser.add_argument("--nthread", type=int, default=0)
    parser.add_argument(
        "--max-train-rows",
        type=int,
        default=0,
        help=(
            "Optional deterministic cap for training rows per split. "
            "0 means use all rows. Use only for smoke tests."
        ),
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def default_splits() -> list[SplitSpec]:
    return [
        SplitSpec("holdout_2005_2008", 2000, 2001, 2004, 2005, 2008),
        SplitSpec("holdout_2009_2012", 2004, 2005, 2008, 2009, 2012),
        SplitSpec("holdout_2013_2016", 2008, 2009, 2012, 2013, 2016),
        SplitSpec("holdout_2017_2020", 2012, 2013, 2016, 2017, 2020),
        SplitSpec("canonical_2021_2026", 2016, 2017, 2020, 2021, 2026),
    ]


def brier_score(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean((y - p) ** 2))


def bss(y: np.ndarray, p: np.ndarray, ref: np.ndarray) -> float:
    bs_model = brier_score(y, p)
    bs_ref = brier_score(y, ref)
    return float(1.0 - bs_model / bs_ref) if bs_ref > 0 else float("nan")


def bootstrap_bss(
    y: np.ndarray,
    p: np.ndarray,
    ref: np.ndarray,
    n_bootstrap: int,
    seed: int,
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    n = len(y)
    vals = np.empty(n_bootstrap, dtype=float)
    for i in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        vals[i] = bss(y[idx], p[idx], ref[idx])
    return float(np.nanpercentile(vals, 2.5)), float(np.nanpercentile(vals, 97.5))


def safe_corr(a: pd.Series, b: pd.Series) -> float:
    if len(a) < 3 or a.std(ddof=0) == 0 or b.std(ddof=0) == 0:
        return float("nan")
    return float(a.corr(b))


def amplitude_ratio(obs: pd.Series, pred: pd.Series) -> float:
    denom = float(obs.std(ddof=0))
    if denom <= 0:
        return float("nan")
    return float(pred.std(ddof=0) / denom)


def add_target_time(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["time"] = pd.to_datetime(out["time"])
    out["target_time"] = out["time"] + pd.DateOffset(months=1)
    out["target_year"] = out["target_time"].dt.year.astype(int)
    out["target_month"] = out["target_time"].dt.month.astype(int)
    return out


def monthly_observed_from_pixels(df: pd.DataFrame) -> pd.DataFrame:
    tmp = df[["target_time", "target_year", "target_month", TARGET]].copy()
    tmp["is_dry"] = (tmp[TARGET] == -1).astype(float)
    return (
        tmp.groupby(["target_time", "target_year", "target_month"], as_index=False)
        .agg(obs_dry_frac=("is_dry", "mean"), n_pixels=("is_dry", "size"))
        .sort_values("target_time")
    )


def prior_climatology(monthly_all: pd.DataFrame, target_time: pd.Timestamp, month: int, window_years: int | None) -> float:
    prior = monthly_all[
        (monthly_all["target_time"] < target_time)
        & (monthly_all["target_month"] == month)
    ]
    if window_years is not None:
        cutoff = target_time - pd.DateOffset(years=window_years)
        prior = prior[prior["target_time"] >= cutoff]
    if not prior.empty:
        return float(prior["obs_dry_frac"].mean())
    prior_all = monthly_all[monthly_all["target_time"] < target_time]
    return float(prior_all["obs_dry_frac"].mean()) if not prior_all.empty else float(monthly_all["obs_dry_frac"].mean())


def attach_climatology_references(
    monthly: pd.DataFrame,
    train_pixels: pd.DataFrame,
    monthly_all: pd.DataFrame,
) -> pd.DataFrame:
    train_tmp = train_pixels[["target_month", TARGET]].copy()
    train_tmp["is_dry"] = (train_tmp[TARGET] == -1).astype(float)
    train_monthly = train_tmp.groupby("target_month")["is_dry"].mean()
    train_global = float(train_tmp["is_dry"].mean())

    fixed_1991_2020 = monthly_all[
        (monthly_all["target_year"] >= 1991)
        & (monthly_all["target_year"] <= 2020)
    ].groupby("target_month")["obs_dry_frac"].mean()
    fixed_global = float(monthly_all[
        (monthly_all["target_year"] >= 1991)
        & (monthly_all["target_year"] <= 2020)
    ]["obs_dry_frac"].mean())

    out = monthly.copy()
    out["clim_train_monthly"] = out["target_month"].map(train_monthly).fillna(train_global)
    out["clim_fixed_1991_2020"] = out["target_month"].map(fixed_1991_2020).fillna(fixed_global)
    out["clim_expanding_prior"] = [
        prior_climatology(monthly_all, t, m, None)
        for t, m in zip(out["target_time"], out["target_month"], strict=True)
    ]
    out["clim_rolling_15yr_prior"] = [
        prior_climatology(monthly_all, t, m, 15)
        for t, m in zip(out["target_time"], out["target_month"], strict=True)
    ]
    out["clim_rolling_30yr_prior"] = [
        prior_climatology(monthly_all, t, m, 30)
        for t, m in zip(out["target_time"], out["target_month"], strict=True)
    ]
    return out


def split_frame(df: pd.DataFrame, spec: SplitSpec) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train = df[df["target_year"] <= spec.train_end].copy()
    val = df[(df["target_year"] >= spec.val_start) & (df["target_year"] <= spec.val_end)].copy()
    test = df[(df["target_year"] >= spec.test_start) & (df["target_year"] <= spec.test_end)].copy()
    return train, val, test


def maybe_cap_train(train: pd.DataFrame, max_rows: int, seed: int) -> pd.DataFrame:
    if max_rows <= 0 or len(train) <= max_rows:
        return train
    return train.sample(n=max_rows, random_state=seed).sort_values(["target_time", "latitude", "longitude"])


def xgb_params(args: argparse.Namespace) -> dict[str, object]:
    params: dict[str, object] = {
        "objective": "multi:softprob",
        "num_class": 3,
        "eval_metric": "mlogloss",
        "tree_method": "hist",
        "eta": args.eta,
        "max_depth": args.max_depth,
        "min_child_weight": 5,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "lambda": 1.0,
        "alpha": 0.1,
        "seed": args.seed,
    }
    if args.nthread > 0:
        params["nthread"] = args.nthread
    return params


def fit_calibrators(
    val_raw: np.ndarray,
    y_val_dry: np.ndarray,
    test_raw: np.ndarray,
) -> dict[str, np.ndarray]:
    preds = {"none": test_raw.clip(0.0, 1.0)}

    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(val_raw, y_val_dry)
    preds["isotonic"] = iso.predict(test_raw).clip(0.0, 1.0)

    platt = LogisticRegression(solver="lbfgs")
    platt.fit(val_raw.reshape(-1, 1), y_val_dry)
    preds["platt"] = platt.predict_proba(test_raw.reshape(-1, 1))[:, 1].clip(0.0, 1.0)
    return preds


def select_calibration(
    val: pd.DataFrame,
    val_raw: np.ndarray,
    y_val_dry: np.ndarray,
) -> tuple[str, dict[str, float]]:
    candidates: dict[str, np.ndarray] = {"none": val_raw.clip(0.0, 1.0)}
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(val_raw, y_val_dry)
    candidates["isotonic"] = iso.predict(val_raw).clip(0.0, 1.0)
    platt = LogisticRegression(solver="lbfgs")
    platt.fit(val_raw.reshape(-1, 1), y_val_dry)
    candidates["platt"] = platt.predict_proba(val_raw.reshape(-1, 1))[:, 1].clip(0.0, 1.0)

    tmp = val[["target_time", TARGET]].copy()
    tmp["obs"] = y_val_dry
    scores: dict[str, float] = {}
    obs_monthly = tmp.groupby("target_time")["obs"].mean()
    for method, pred in candidates.items():
        tmp[method] = pred
        pred_monthly = tmp.groupby("target_time")[method].mean().reindex(obs_monthly.index)
        scores[method] = brier_score(obs_monthly.to_numpy(), pred_monthly.to_numpy())
    best = min(scores, key=scores.get)
    return best, scores


def run_split(
    df: pd.DataFrame,
    monthly_all: pd.DataFrame,
    spec: SplitSpec,
    features: list[str],
    args: argparse.Namespace,
    split_idx: int,
) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    train, val, test = split_frame(df, spec)
    if train.empty or val.empty or test.empty:
        raise ValueError(f"Empty split {spec}: train={train.shape}, val={val.shape}, test={test.shape}")

    train_fit = maybe_cap_train(train, args.max_train_rows, args.seed + split_idx)
    print(
        f"[{spec.name}] Train {train_fit.shape} (source {train.shape})  "
        f"Val {val.shape}  Test {test.shape}"
    )

    y_train = train_fit[TARGET].map(LABEL_MAP).to_numpy()
    y_val = val[TARGET].map(LABEL_MAP).to_numpy()
    y_test = test[TARGET].map(LABEL_MAP).to_numpy()

    dtrain = xgb.DMatrix(
        train_fit[features],
        label=y_train,
        weight=compute_sample_weight(class_weight="balanced", y=y_train),
        feature_names=features,
    )
    dval = xgb.DMatrix(val[features], label=y_val, feature_names=features)
    dtest = xgb.DMatrix(test[features], label=y_test, feature_names=features)

    model = xgb.train(
        params=xgb_params(args),
        dtrain=dtrain,
        num_boost_round=args.num_boost_round,
        evals=[(dtrain, "train"), (dval, "val")],
        early_stopping_rounds=args.early_stopping_rounds,
        verbose_eval=100,
    )
    iteration_range = (0, model.best_iteration + 1)
    val_probs = model.predict(dval, iteration_range=iteration_range).reshape(-1, 3)
    test_probs = model.predict(dtest, iteration_range=iteration_range).reshape(-1, 3)
    val_raw = val_probs[:, DRY_IDX]
    test_raw = test_probs[:, DRY_IDX]
    y_val_dry = (val[TARGET].to_numpy() == -1).astype(int)

    best_calibration, val_bs = select_calibration(val, val_raw, y_val_dry)
    calibrated = fit_calibrators(val_raw, y_val_dry, test_raw)

    test_monthly = test[["target_time", "target_year", "target_month", TARGET]].copy()
    test_monthly["obs"] = (test_monthly[TARGET] == -1).astype(float)
    for method, pred in calibrated.items():
        test_monthly[f"prob_{method}"] = pred
    monthly = (
        test_monthly.groupby(["target_time", "target_year", "target_month"], as_index=False)
        .agg(
            obs_dry_frac=("obs", "mean"),
            xgb_raw_prob_dry=("prob_none", "mean"),
            xgb_isotonic_prob_dry=("prob_isotonic", "mean"),
            xgb_platt_prob_dry=("prob_platt", "mean"),
            n_pixels=("obs", "size"),
        )
        .sort_values("target_time")
    )
    monthly["selected_calibration"] = best_calibration
    monthly["xgb_selected_prob_dry"] = monthly[f"xgb_{best_calibration if best_calibration != 'none' else 'raw'}_prob_dry"]
    monthly = attach_climatology_references(monthly, train, monthly_all)
    monthly["split"] = spec.name
    monthly["train_end"] = spec.train_end
    monthly["validation_years"] = f"{spec.val_start}-{spec.val_end}"
    monthly["test_years"] = f"{spec.test_start}-{spec.test_end}"

    rows: list[dict[str, object]] = []
    y = monthly["obs_dry_frac"].to_numpy()
    p = monthly["xgb_selected_prob_dry"].to_numpy()
    train_obs_mean = float(train[TARGET].eq(-1).mean())
    val_obs_mean = float(val[TARGET].eq(-1).mean())
    test_obs_mean = float(test[TARGET].eq(-1).mean())

    for ref_col in [
        "clim_train_monthly",
        "clim_expanding_prior",
        "clim_rolling_15yr_prior",
        "clim_rolling_30yr_prior",
        "clim_fixed_1991_2020",
    ]:
        ref = monthly[ref_col].to_numpy()
        ci_low, ci_high = bootstrap_bss(y, p, ref, args.n_bootstrap, args.seed + 1000 + split_idx)
        rows.append(
            {
                "split": spec.name,
                "train_years": f"1991-{spec.train_end}",
                "validation_years": f"{spec.val_start}-{spec.val_end}",
                "test_years": f"{spec.test_start}-{spec.test_end}",
                "n_test_months": int(len(monthly)),
                "n_train_rows": int(len(train)),
                "n_train_rows_fit": int(len(train_fit)),
                "n_val_rows": int(len(val)),
                "n_test_rows": int(len(test)),
                "best_iteration": int(model.best_iteration),
                "selected_calibration": best_calibration,
                "validation_bs_none": val_bs["none"],
                "validation_bs_isotonic": val_bs["isotonic"],
                "validation_bs_platt": val_bs["platt"],
                "reference": ref_col,
                "bs_model": brier_score(y, p),
                "bs_reference": brier_score(y, ref),
                "bss": bss(y, p, ref),
                "bss_ci_low": ci_low,
                "bss_ci_high": ci_high,
                "obs_dry_mean_train_pixels": train_obs_mean,
                "obs_dry_mean_val_pixels": val_obs_mean,
                "obs_dry_mean_test_pixels": test_obs_mean,
                "test_minus_train_dry_mean": test_obs_mean - train_obs_mean,
                "test_minus_val_dry_mean": test_obs_mean - val_obs_mean,
                "pred_dry_mean_test_monthly": float(monthly["xgb_selected_prob_dry"].mean()),
                "obs_dry_mean_test_monthly": float(monthly["obs_dry_frac"].mean()),
                "prediction_bias_monthly": float((monthly["xgb_selected_prob_dry"] - monthly["obs_dry_frac"]).mean()),
                "prediction_corr_monthly": safe_corr(monthly["obs_dry_frac"], monthly["xgb_selected_prob_dry"]),
                "prediction_amplitude_ratio": amplitude_ratio(monthly["obs_dry_frac"], monthly["xgb_selected_prob_dry"]),
            }
        )

    return monthly, rows


def event_block_rows(monthly_all_splits: pd.DataFrame) -> pd.DataFrame:
    canonical = monthly_all_splits[monthly_all_splits["split"] == "canonical_2021_2026"].copy()
    if canonical.empty:
        return pd.DataFrame()
    blocks = [
        ("drought_2021_2022", "2021-01-01", "2022-12-01"),
        ("wet_reversal_2023", "2023-01-01", "2023-12-01"),
        ("late_test_2024_2026", "2024-01-01", "2026-03-01"),
    ]
    rows = []
    for name, start, end in blocks:
        block = canonical[
            (canonical["target_time"] >= pd.Timestamp(start))
            & (canonical["target_time"] <= pd.Timestamp(end))
        ]
        if block.empty:
            continue
        y = block["obs_dry_frac"].to_numpy()
        p = block["xgb_selected_prob_dry"].to_numpy()
        ref = block["clim_train_monthly"].to_numpy()
        rows.append(
            {
                "block": name,
                "date_start": start,
                "date_end": end,
                "n_months": int(len(block)),
                "obs_dry_mean": float(block["obs_dry_frac"].mean()),
                "pred_dry_mean": float(block["xgb_selected_prob_dry"].mean()),
                "clim_dry_mean": float(block["clim_train_monthly"].mean()),
                "bias": float((block["xgb_selected_prob_dry"] - block["obs_dry_frac"]).mean()),
                "bs_model": brier_score(y, p),
                "bs_reference": brier_score(y, ref),
                "bss_vs_train_monthly_climatology": bss(y, p, ref),
                "corr": safe_corr(block["obs_dry_frac"], block["xgb_selected_prob_dry"]),
                "amplitude_ratio": amplitude_ratio(block["obs_dry_frac"], block["xgb_selected_prob_dry"]),
            }
        )
    return pd.DataFrame(rows)


def write_notes(summary: pd.DataFrame, event_blocks: pd.DataFrame, out_dir: Path) -> None:
    primary = summary[summary["reference"] == "clim_train_monthly"].copy()
    n_positive = int((primary["bss"] > 0).sum())
    n_robust_positive = int((primary["bss_ci_low"] > 0).sum())
    n_robust_negative = int((primary["bss_ci_high"] < 0).sum())

    lines = [
        "Temporal Robustness Audit",
        "=" * 72,
        "",
        "Design: rolling chronological Central Valley tabular XGBoost checkpoints.",
        "Each split uses validation-only calibration selection and monthly dry-fraction BSS.",
        "Primary reference: train-period calendar-month climatology.",
        "",
        f"Splits evaluated: {primary['split'].nunique()}",
        f"Positive BSS point estimates: {n_positive}/{len(primary)}",
        f"Robust positive BSS intervals: {n_robust_positive}/{len(primary)}",
        f"Robust negative BSS intervals: {n_robust_negative}/{len(primary)}",
        "",
        "Primary split summary:",
        primary[
            [
                "split",
                "test_years",
                "n_test_months",
                "selected_calibration",
                "bss",
                "bss_ci_low",
                "bss_ci_high",
                "test_minus_train_dry_mean",
                "prediction_bias_monthly",
                "prediction_corr_monthly",
            ]
        ].round(4).to_string(index=False),
    ]
    if not event_blocks.empty:
        lines.extend(
            [
                "",
                "Canonical 2021-2026 event blocks:",
                event_blocks.round(4).to_string(index=False),
            ]
        )
    (out_dir / "temporal_robustness_audit.txt").write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {args.dataset}")
    df = pd.read_parquet(args.dataset)
    df = add_target_time(df)
    features = get_feature_columns(df.columns)
    monthly_all = monthly_observed_from_pixels(df)

    monthly_outputs: list[pd.DataFrame] = []
    summary_rows: list[dict[str, object]] = []
    for i, spec in enumerate(default_splits()):
        monthly, rows = run_split(df, monthly_all, spec, features, args, i)
        monthly_outputs.append(monthly)
        summary_rows.extend(rows)

    monthly_all_splits = pd.concat(monthly_outputs, ignore_index=True)
    summary = pd.DataFrame(summary_rows)
    event_blocks = event_block_rows(monthly_all_splits)

    monthly_path = args.out_dir / "temporal_robustness_monthly_predictions.csv"
    summary_path = args.out_dir / "temporal_robustness_summary.csv"
    events_path = args.out_dir / "temporal_robustness_event_blocks.csv"
    monthly_all_splits.to_csv(monthly_path, index=False)
    summary.to_csv(summary_path, index=False)
    event_blocks.to_csv(events_path, index=False)
    write_notes(summary, event_blocks, args.out_dir)

    print(f"Wrote {monthly_path}")
    print(f"Wrote {summary_path}")
    print(f"Wrote {events_path}")
    print(f"Wrote {args.out_dir / 'temporal_robustness_audit.txt'}")


if __name__ == "__main__":
    main()
