#!/usr/bin/env python
"""Run a transition-target drought forecast experiment (SPI-1 lead-1).

This experiment tests whether forecasting *transitions* is easier/more skillful
than forecasting monthly state under the same leakage-safe setup.

Targets (binary events, per pixel):
  - onset:       not-dry at month t  -> dry at month t+1
  - termination: dry at month t      -> not-dry at month t+1

Inputs:
  - data/processed/dataset_forecast.parquet by default
  - or any region forecast parquet built by run_multiregion_xgb_experiment.py

Evaluation:
  - Monthly aggregation is the primary verification unit (independent months).
  - Scores: Brier Score and Brier Skill Score (BSS) vs monthly climatology.
  - Additional reference: state-conditioned monthly climatology (Markov-style).

Outputs (written to outputs/):
  - transition_<kind>_monthly_scores.csv
  - transition_<kind>_eligible_monthly_scores.csv
  - transition_<kind>_xgb_model.json
  - transition_<kind>_conditional_xgb_model.json
  - transition_<kind>_xgb_test_probs.npz
  - transition_<kind>_xgb_feature_importance.png
  - transition_<kind>_conditional_xgb_feature_importance.png
  - transition_<kind>_experiment_scores.txt
"""

from __future__ import annotations

from argparse import ArgumentParser, Namespace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.isotonic import IsotonicRegression
from sklearn.utils.class_weight import compute_sample_weight

from feature_config import get_feature_columns


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA = PROJECT_ROOT / "data" / "processed" / "dataset_forecast.parquet"
OUT_DIR = PROJECT_ROOT / "outputs"
OUT_DIR.mkdir(exist_ok=True)


def project_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def slugify(value: str) -> str:
    return (
        value.strip()
        .lower()
        .replace("-", "_")
        .replace("/", "_")
        .replace(" ", "_")
    )


def parse_args() -> Namespace:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument(
        "--transition",
        choices=["onset", "termination"],
        default="onset",
        help="Transition event to forecast (default: onset).",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DATA,
        help=(
            "Forecast parquet to use. Defaults to the canonical Central Valley "
            "data/processed/dataset_forecast.parquet."
        ),
    )
    parser.add_argument(
        "--scope",
        default="cvalley",
        help=(
            "Short label written into outputs. Use a region/mask slug for "
            "regional replication runs."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUT_DIR,
        help="Directory for transition artifacts. Defaults to outputs/.",
    )
    parser.add_argument(
        "--n-bootstrap",
        type=int,
        default=2000,
        help="Bootstrap iterations for monthly BSS confidence intervals.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for bootstrap resampling.",
    )
    parser.add_argument(
        "--output-prefix",
        default=None,
        help=(
            "Optional output file prefix. Defaults to transition_<kind> for "
            "canonical cvalley and transition_<scope>_<kind> otherwise."
        ),
    )
    parser.add_argument(
        "--num-boost-round",
        type=int,
        default=2000,
        help="Maximum XGBoost boosting rounds.",
    )
    parser.add_argument(
        "--early-stopping-rounds",
        type=int,
        default=50,
        help="XGBoost early-stopping patience.",
    )
    parser.add_argument(
        "--verbose-eval",
        type=int,
        default=100,
        help="XGBoost logging interval.",
    )
    return parser.parse_args()


def brier(y: np.ndarray, p: np.ndarray) -> float:
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    return float(np.mean((p - y) ** 2))


def bss(y: np.ndarray, p: np.ndarray, ref: np.ndarray) -> float:
    ref_bs = brier(y, ref)
    return float(1.0 - brier(y, p) / ref_bs) if ref_bs > 0 else float("nan")


def bootstrap_bss(
    monthly: pd.DataFrame,
    pred_col: str,
    ref_col: str,
    n_bootstrap: int,
    seed: int,
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    idx = np.arange(len(monthly))
    y = monthly["y_true_event_frac"].to_numpy(dtype=float)
    p = monthly[pred_col].to_numpy(dtype=float)
    ref = monthly[ref_col].to_numpy(dtype=float)
    vals = np.empty(n_bootstrap, dtype=float)
    for i in range(n_bootstrap):
        sample = rng.choice(idx, size=len(idx), replace=True)
        vals[i] = bss(y[sample], p[sample], ref[sample])
    lo, hi = np.nanquantile(vals, [0.025, 0.975])
    return float(lo), float(hi)


def bootstrap_bss_arrays(
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


def amplitude_ratio(y: pd.Series, p: pd.Series) -> float:
    y_std = float(y.std(ddof=0))
    p_std = float(p.std(ddof=0))
    return p_std / y_std if y_std > 0 else float("nan")


def build_transition_labels(df: pd.DataFrame, kind: str) -> pd.DataFrame:
    out = df.copy()

    out["current_is_dry"] = out["spi1_lag1"].astype(float) <= -1.0
    out["next_is_dry"] = out["target_label"].astype(int) == -1

    if kind == "onset":
        out["eligible"] = ~out["current_is_dry"]
        out["event"] = (out["eligible"] & out["next_is_dry"]).astype(int)
    elif kind == "termination":
        out["eligible"] = out["current_is_dry"]
        out["event"] = (out["eligible"] & ~out["next_is_dry"]).astype(int)
    else:
        raise ValueError(f"Unknown transition kind: {kind}")

    return out


def add_eligible_climatology(train_eligible: pd.DataFrame, test_eligible: pd.DataFrame) -> pd.DataFrame:
    out = test_eligible.copy()
    month_clim = train_eligible.groupby("month", observed=True)["event"].mean()
    global_clim = float(train_eligible["event"].mean())
    out["eligible_clim_prob_event"] = out["month"].map(month_clim).fillna(global_clim)
    return out


def aggregate_monthly(
    frame: pd.DataFrame,
    pred_cols: dict[str, str],
    ref_col: str,
) -> pd.DataFrame:
    agg: dict[str, tuple[str, str]] = {
        "y_true_event_frac": ("event", "mean"),
        "n_pixels": ("event", "size"),
        "reference_prob_event": (ref_col, "mean"),
    }
    for out_col, source_col in pred_cols.items():
        agg[out_col] = (source_col, "mean")
    return (
        frame.groupby("target_time", observed=True)
        .agg(**agg)
        .reset_index()
        .sort_values("target_time")
    )


def score_line(
    monthly: pd.DataFrame,
    pred_col: str,
    ref_col: str,
    n_bootstrap: int,
    seed: int,
) -> dict[str, float]:
    y = monthly["y_true_event_frac"].to_numpy(dtype=float)
    p = monthly[pred_col].to_numpy(dtype=float)
    ref = monthly[ref_col].to_numpy(dtype=float)
    ci_low, ci_high = bootstrap_bss_arrays(y, p, ref, n_bootstrap, seed)
    return {
        "bs": brier(y, p),
        "ref_bs": brier(y, ref),
        "bss": bss(y, p, ref),
        "ci_low": ci_low,
        "ci_high": ci_high,
        "spearman": float(
            monthly["y_true_event_frac"].corr(monthly[pred_col], method="spearman")
        ),
        "amp_ratio": amplitude_ratio(monthly["y_true_event_frac"], monthly[pred_col]),
    }


def plot_importance(model: xgb.Booster, features: list[str], title: str, out_path: Path) -> None:
    importance = model.get_score(importance_type="gain")
    imp = (
        pd.Series([importance.get(f, 0.0) for f in features], index=features)
        .sort_values(ascending=True)
    )
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(imp.index, imp.values)
    ax.set_xlabel("Gain")
    ax.set_title(title)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def add_climatology_baselines(train: pd.DataFrame, test: pd.DataFrame, kind: str) -> pd.DataFrame:
    out = test.copy()

    # Unconditional monthly climatology of event frequency.
    month_clim = train.groupby("month", observed=True)["event"].mean()
    global_clim = float(train["event"].mean())
    out["clim_prob_event"] = out["month"].map(month_clim).fillna(global_clim)

    # State-conditioned climatology (Markov-style sanity baseline).
    if kind == "onset":
        eligible_train = train.loc[~train["current_is_dry"].astype(bool)]
        eligible_test = ~out["current_is_dry"].astype(bool)
        fallback = float(eligible_train["event"].mean()) if not eligible_train.empty else global_clim
        cond = eligible_train.groupby("month", observed=True)["event"].mean()
        out["clim_cond_prob_event"] = 0.0
        out.loc[eligible_test, "clim_cond_prob_event"] = (
            out.loc[eligible_test, "month"].map(cond).fillna(fallback)
        )
    elif kind == "termination":
        eligible_train = train.loc[train["current_is_dry"].astype(bool)]
        eligible_test = out["current_is_dry"].astype(bool)
        fallback = float(eligible_train["event"].mean()) if not eligible_train.empty else global_clim
        cond = eligible_train.groupby("month", observed=True)["event"].mean()
        out["clim_cond_prob_event"] = 0.0
        out.loc[eligible_test, "clim_cond_prob_event"] = (
            out.loc[eligible_test, "month"].map(cond).fillna(fallback)
        )
    else:
        raise ValueError(f"Unknown transition kind: {kind}")

    return out


def main() -> None:
    args = parse_args()
    dataset = project_path(args.dataset)
    out_dir = project_path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    scope = slugify(args.scope)
    default_dataset = DATA.resolve()
    if args.output_prefix is not None:
        prefix = args.output_prefix
    elif dataset.resolve() == default_dataset and scope == "cvalley":
        prefix = f"transition_{args.transition}"
    else:
        prefix = f"transition_{scope}_{args.transition}"

    if not dataset.exists():
        raise FileNotFoundError(
            f"Dataset not found: {dataset}. Build the forecast parquet first."
        )

    print(f"Loading dataset: {dataset}")
    df = pd.read_parquet(dataset)
    df["year"] = df["year"].astype(int)

    # Align a canonical target-month timestamp for monthly aggregation.
    df["time"] = pd.to_datetime(df["time"]).dt.to_period("M").dt.to_timestamp()
    df["target_time"] = (df["time"] + pd.DateOffset(months=1)).dt.to_period("M").dt.to_timestamp()

    df = build_transition_labels(df, args.transition)

    features = get_feature_columns(df.columns)

    train = df[df["year"] <= 2016].copy()
    val = df[(df["year"] >= 2017) & (df["year"] <= 2020)].copy()
    test = df[df["year"] >= 2021].copy()

    if train.empty or val.empty or test.empty:
        raise ValueError(
            f"Empty split(s): train={len(train):,} val={len(val):,} test={len(test):,}."
        )

    print(
        f"Scope={scope}  Transition={args.transition}  "
        f"Train={train.shape}  Val={val.shape}  Test={test.shape}"
    )
    print(f"Features ({len(features)}): {features}")

    y_train = train["event"].to_numpy(dtype=int)
    y_val = val["event"].to_numpy(dtype=int)
    y_test = test["event"].to_numpy(dtype=int)

    dtrain = xgb.DMatrix(
        train[features],
        label=y_train,
        weight=compute_sample_weight(class_weight="balanced", y=y_train),
        feature_names=features,
    )
    dval = xgb.DMatrix(val[features], label=y_val, feature_names=features)
    dtest = xgb.DMatrix(test[features], label=y_test, feature_names=features)

    params = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "tree_method": "hist",
        "eta": 0.05,
        "max_depth": 7,
        "min_child_weight": 10,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "lambda": 1.0,
        "alpha": 0.0,
        "seed": args.seed,
    }

    print("Training binary XGBoost...")
    model = xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=args.num_boost_round,
        evals=[(dtrain, "train"), (dval, "val")],
        early_stopping_rounds=args.early_stopping_rounds,
        verbose_eval=args.verbose_eval,
    )
    best_iter = int(model.best_iteration)
    print(f"Best iteration: {best_iter}")

    iteration_range = (0, best_iter + 1)
    p_val = model.predict(dval, iteration_range=iteration_range).astype("float32")
    p_test = model.predict(dtest, iteration_range=iteration_range).astype("float32")

    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(p_val, y_val)
    p_test_cal = iso.predict(p_test).astype("float32")

    test = test.copy()
    test["xgb_prob_event"] = p_test
    test["xgb_cal_prob_event"] = p_test_cal

    test = add_climatology_baselines(train=train, test=test, kind=args.transition)

    monthly = (
        test.groupby("target_time", observed=True)
        .agg(
            y_true_event_frac=("event", "mean"),
            xgb_prob_event=("xgb_prob_event", "mean"),
            xgb_cal_prob_event=("xgb_cal_prob_event", "mean"),
            clim_prob_event=("clim_prob_event", "mean"),
            clim_cond_prob_event=("clim_cond_prob_event", "mean"),
        )
        .reset_index()
        .sort_values("target_time")
    )

    # Scores
    y = monthly["y_true_event_frac"].to_numpy(dtype=float)
    bs_clim = brier(y, monthly["clim_prob_event"].to_numpy(dtype=float))
    bs_clim_cond = brier(y, monthly["clim_cond_prob_event"].to_numpy(dtype=float))
    bs_xgb = brier(y, monthly["xgb_prob_event"].to_numpy(dtype=float))
    bs_xgb_cal = brier(y, monthly["xgb_cal_prob_event"].to_numpy(dtype=float))

    bss_xgb = bss(y, monthly["xgb_prob_event"].to_numpy(dtype=float), monthly["clim_prob_event"].to_numpy(dtype=float))
    bss_xgb_cal = bss(y, monthly["xgb_cal_prob_event"].to_numpy(dtype=float), monthly["clim_prob_event"].to_numpy(dtype=float))
    bss_xgb_vs_cond = bss(y, monthly["xgb_prob_event"].to_numpy(dtype=float), monthly["clim_cond_prob_event"].to_numpy(dtype=float))
    bss_xgb_cal_vs_cond = bss(y, monthly["xgb_cal_prob_event"].to_numpy(dtype=float), monthly["clim_cond_prob_event"].to_numpy(dtype=float))

    ci_xgb = bootstrap_bss(monthly, "xgb_prob_event", "clim_prob_event", args.n_bootstrap, args.seed)
    ci_xgb_cal = bootstrap_bss(monthly, "xgb_cal_prob_event", "clim_prob_event", args.n_bootstrap, args.seed)
    ci_xgb_vs_cond = bootstrap_bss(
        monthly,
        "xgb_prob_event",
        "clim_cond_prob_event",
        args.n_bootstrap,
        args.seed,
    )
    ci_xgb_cal_vs_cond = bootstrap_bss(
        monthly,
        "xgb_cal_prob_event",
        "clim_cond_prob_event",
        args.n_bootstrap,
        args.seed,
    )

    # Event-tracking diagnostics (monthly)
    spearman_raw = float(monthly["y_true_event_frac"].corr(monthly["xgb_prob_event"], method="spearman"))
    spearman_cal = float(monthly["y_true_event_frac"].corr(monthly["xgb_cal_prob_event"], method="spearman"))
    amp_raw = amplitude_ratio(monthly["y_true_event_frac"], monthly["xgb_prob_event"])
    amp_cal = amplitude_ratio(monthly["y_true_event_frac"], monthly["xgb_cal_prob_event"])

    # Save artifacts
    monthly_path = out_dir / f"{prefix}_monthly_scores.csv"
    eligible_monthly_path = out_dir / f"{prefix}_eligible_monthly_scores.csv"
    model_path = out_dir / f"{prefix}_xgb_model.json"
    conditional_model_path = out_dir / f"{prefix}_conditional_xgb_model.json"
    probs_path = out_dir / f"{prefix}_xgb_test_probs.npz"
    fi_path = out_dir / f"{prefix}_xgb_feature_importance.png"
    conditional_fi_path = out_dir / f"{prefix}_conditional_xgb_feature_importance.png"
    scores_path = out_dir / f"{prefix}_experiment_scores.txt"

    # Eligible-only evaluation isolates the non-trivial transition probability:
    # onset is scored only where the current pixel is not dry, and termination
    # only where the current pixel is dry. This avoids treating current-state
    # gating as forecast skill.
    train_eligible = train.loc[train["eligible"].astype(bool)].copy()
    val_eligible = val.loc[val["eligible"].astype(bool)].copy()
    test_eligible = test.loc[test["eligible"].astype(bool)].copy()
    test_eligible = add_eligible_climatology(train_eligible, test_eligible)

    eligible_monthly_existing = aggregate_monthly(
        test_eligible,
        pred_cols={
            "xgb_prob_event": "xgb_prob_event",
            "xgb_cal_prob_event": "xgb_cal_prob_event",
        },
        ref_col="eligible_clim_prob_event",
    )

    if train_eligible["event"].nunique() < 2 or val_eligible["event"].nunique() < 2:
        raise ValueError(
            "Eligible-only train/validation split has fewer than two event classes; "
            f"train classes={train_eligible['event'].nunique()}, "
            f"val classes={val_eligible['event'].nunique()}."
        )

    dtrain_cond = xgb.DMatrix(
        train_eligible[features],
        label=train_eligible["event"].to_numpy(dtype=int),
        weight=compute_sample_weight(
            class_weight="balanced",
            y=train_eligible["event"].to_numpy(dtype=int),
        ),
        feature_names=features,
    )
    dval_cond = xgb.DMatrix(
        val_eligible[features],
        label=val_eligible["event"].to_numpy(dtype=int),
        feature_names=features,
    )
    dtest_cond = xgb.DMatrix(
        test_eligible[features],
        label=test_eligible["event"].to_numpy(dtype=int),
        feature_names=features,
    )
    print("Training eligible-only binary XGBoost...")
    conditional_model = xgb.train(
        params=params,
        dtrain=dtrain_cond,
        num_boost_round=args.num_boost_round,
        evals=[(dtrain_cond, "train"), (dval_cond, "val")],
        early_stopping_rounds=args.early_stopping_rounds,
        verbose_eval=args.verbose_eval,
    )
    conditional_best_iter = int(conditional_model.best_iteration)
    conditional_range = (0, conditional_best_iter + 1)
    p_val_cond = conditional_model.predict(
        dval_cond,
        iteration_range=conditional_range,
    ).astype("float32")
    p_test_cond = conditional_model.predict(
        dtest_cond,
        iteration_range=conditional_range,
    ).astype("float32")

    iso_cond = IsotonicRegression(out_of_bounds="clip")
    iso_cond.fit(p_val_cond, val_eligible["event"].to_numpy(dtype=int))
    p_test_cond_cal = iso_cond.predict(p_test_cond).astype("float32")
    test_eligible["conditional_xgb_prob_event"] = p_test_cond
    test_eligible["conditional_xgb_cal_prob_event"] = p_test_cond_cal

    eligible_monthly_conditional = aggregate_monthly(
        test_eligible,
        pred_cols={
            "conditional_xgb_prob_event": "conditional_xgb_prob_event",
            "conditional_xgb_cal_prob_event": "conditional_xgb_cal_prob_event",
        },
        ref_col="eligible_clim_prob_event",
    )
    eligible_monthly = eligible_monthly_existing.merge(
        eligible_monthly_conditional[
            [
                "target_time",
                "conditional_xgb_prob_event",
                "conditional_xgb_cal_prob_event",
            ]
        ],
        on="target_time",
        how="left",
    )
    monthly.insert(0, "scope", scope)
    monthly.insert(1, "transition", args.transition)
    eligible_monthly.insert(0, "scope", scope)
    eligible_monthly.insert(1, "transition", args.transition)
    eligible_monthly.to_csv(eligible_monthly_path, index=False)

    monthly.to_csv(monthly_path, index=False)
    model.save_model(model_path.as_posix())
    conditional_model.save_model(conditional_model_path.as_posix())
    np.savez_compressed(
        probs_path,
        probs=p_test,
        probs_calibrated=p_test_cal,
        y_true=y_test,
        times=test["time"].to_numpy(),
        target_times=test["target_time"].to_numpy(),
        latitude=test["latitude"].to_numpy(),
        longitude=test["longitude"].to_numpy(),
        features=np.array(features),
        best_iteration=best_iter,
        conditional_best_iteration=conditional_best_iter,
        transition=args.transition,
        scope=scope,
        dataset=str(dataset),
    )

    plot_importance(
        model,
        features,
        f"Transition target ({args.transition}) — all-pixel XGBoost feature importance",
        fi_path,
    )
    plot_importance(
        conditional_model,
        features,
        f"Transition target ({args.transition}) — eligible-only XGBoost feature importance",
        conditional_fi_path,
    )

    eligible_existing_raw = score_line(
        eligible_monthly,
        "xgb_prob_event",
        "reference_prob_event",
        args.n_bootstrap,
        args.seed,
    )
    eligible_existing_cal = score_line(
        eligible_monthly,
        "xgb_cal_prob_event",
        "reference_prob_event",
        args.n_bootstrap,
        args.seed,
    )
    eligible_cond_raw = score_line(
        eligible_monthly,
        "conditional_xgb_prob_event",
        "reference_prob_event",
        args.n_bootstrap,
        args.seed,
    )
    eligible_cond_cal = score_line(
        eligible_monthly,
        "conditional_xgb_cal_prob_event",
        "reference_prob_event",
        args.n_bootstrap,
        args.seed,
    )

    eligible_counts = eligible_monthly["n_pixels"].astype(int)

    metrics = (
        f"Transition-target SPI-1 lead-1 experiment\n"
        f"{'=' * 64}\n"
        f"Scope: {scope}\n"
        f"Dataset: {dataset}\n"
        f"Transition: {args.transition}\n"
        f"Test months: {monthly['target_time'].nunique()}\n"
        f"Features: {features}\n"
        f"Best iteration: {best_iter}\n\n"
        f"Monthly event-fraction Brier Scores\n"
        f"  Climatology (uncond) : {bs_clim:.6f}\n"
        f"  Climatology (cond)   : {bs_clim_cond:.6f}\n"
        f"  XGBoost              : {bs_xgb:.6f}\n"
        f"  XGBoost isotonic     : {bs_xgb_cal:.6f}\n\n"
        f"Brier Skill Score vs monthly climatology (unconditional)\n"
        f"  XGBoost          : {bss_xgb:+.6f} (95% CI [{ci_xgb[0]:+.6f}, {ci_xgb[1]:+.6f}])\n"
        f"  XGBoost isotonic : {bss_xgb_cal:+.6f} (95% CI [{ci_xgb_cal[0]:+.6f}, {ci_xgb_cal[1]:+.6f}])\n\n"
        f"Brier Skill Score vs state-conditioned monthly climatology (sanity baseline)\n"
        f"  XGBoost          : {bss_xgb_vs_cond:+.6f} (95% CI [{ci_xgb_vs_cond[0]:+.6f}, {ci_xgb_vs_cond[1]:+.6f}])\n"
        f"  XGBoost isotonic : {bss_xgb_cal_vs_cond:+.6f} (95% CI [{ci_xgb_cal_vs_cond[0]:+.6f}, {ci_xgb_cal_vs_cond[1]:+.6f}])\n\n"
        f"Monthly event tracking\n"
        f"  Spearman(raw)    : {spearman_raw:+.4f}\n"
        f"  Spearman(cal)    : {spearman_cal:+.4f}\n"
        f"  Amp ratio(raw)   : {amp_raw:.4f}\n"
        f"  Amp ratio(cal)   : {amp_cal:.4f}\n\n"
        f"Eligible-only monthly transition evaluation\n"
        f"  Months with eligible pixels : {eligible_monthly['target_time'].nunique()}\n"
        f"  Eligible pixels per month   : median={eligible_counts.median():.0f}, "
        f"min={eligible_counts.min():.0f}, max={eligible_counts.max():.0f}\n"
        f"  Existing all-pixel XGBoost raw      : "
        f"BSS={eligible_existing_raw['bss']:+.6f} "
        f"(95% CI [{eligible_existing_raw['ci_low']:+.6f}, {eligible_existing_raw['ci_high']:+.6f}]), "
        f"Spearman={eligible_existing_raw['spearman']:+.4f}, "
        f"Amp ratio={eligible_existing_raw['amp_ratio']:.4f}\n"
        f"  Existing all-pixel XGBoost isotonic : "
        f"BSS={eligible_existing_cal['bss']:+.6f} "
        f"(95% CI [{eligible_existing_cal['ci_low']:+.6f}, {eligible_existing_cal['ci_high']:+.6f}]), "
        f"Spearman={eligible_existing_cal['spearman']:+.4f}, "
        f"Amp ratio={eligible_existing_cal['amp_ratio']:.4f}\n"
        f"  Eligible-only XGBoost raw           : "
        f"BSS={eligible_cond_raw['bss']:+.6f} "
        f"(95% CI [{eligible_cond_raw['ci_low']:+.6f}, {eligible_cond_raw['ci_high']:+.6f}]), "
        f"Spearman={eligible_cond_raw['spearman']:+.4f}, "
        f"Amp ratio={eligible_cond_raw['amp_ratio']:.4f}\n"
        f"  Eligible-only XGBoost isotonic      : "
        f"BSS={eligible_cond_cal['bss']:+.6f} "
        f"(95% CI [{eligible_cond_cal['ci_low']:+.6f}, {eligible_cond_cal['ci_high']:+.6f}]), "
        f"Spearman={eligible_cond_cal['spearman']:+.4f}, "
        f"Amp ratio={eligible_cond_cal['amp_ratio']:.4f}\n"
        f"  Eligible-only best iteration        : {conditional_best_iter}\n\n"
        f"Outputs:\n"
        f"  {monthly_path}\n"
        f"  {eligible_monthly_path}\n"
        f"  {model_path}\n"
        f"  {conditional_model_path}\n"
        f"  {probs_path}\n"
        f"  {fi_path}\n"
        f"  {conditional_fi_path}\n"
    )
    scores_path.write_text(metrics)
    print(metrics)


if __name__ == "__main__":
    main()
