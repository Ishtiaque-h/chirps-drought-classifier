#!/usr/bin/env python
"""
Feature ablation study for the 1-month-ahead drought forecast.

Quantifies the marginal contribution of each feature group by evaluating
the trained XGBoost model with one group of features zeroed out (ablated)
at a time, using the same temporal split and monthly-level BSS protocol as
evaluate_forecast_skill.py.

Feature groups:
  spi_lags   — spi1_lag1, spi1_lag2, spi1_lag3, spi3_lag1, spi6_lag1
  pr_lags    — pr_lag1, pr_lag2, pr_lag3
  seasonality — month_sin, month_cos
  enso       — nino34_lag1, nino34_lag2   (if present in dataset)
  pdo        — pdo_lag1, pdo_lag2         (if present in dataset)

Each ablation replaces the feature values with their per-feature training-set
mean (a neutral, non-leaking replacement) so the model sees a valid input.

Inputs:
  data/processed/dataset_forecast.parquet
  outputs/forecast_xgb_model.json

Outputs:
  outputs/feature_ablation_results.csv
  outputs/feature_ablation_bss_barplot.png
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb

from feature_config import get_feature_columns

DATA       = Path("data/processed/dataset_forecast.parquet")
MODEL_PATH = Path("outputs/forecast_xgb_model.json")
OUT_DIR    = Path("outputs")
OUT_CSV    = OUT_DIR / "feature_ablation_results.csv"
OUT_FIG    = OUT_DIR / "feature_ablation_bss_barplot.png"

LABEL_MAP = {-1: 0, 0: 1, 1: 2}
CLASSES   = [-1, 0, 1]

# ── load data ────────────────────────────────────────────────────────────────
print("Loading dataset...")
df = pd.read_parquet(DATA)
df["year"] = df["year"].astype(int)
FEATURES = get_feature_columns(df.columns)

train = df[df["year"] <= 2016]
val   = df[(df["year"] >= 2017) & (df["year"] <= 2020)]
test  = df[df["year"] >= 2021].copy()

test["month_dt"] = (
    pd.to_datetime(test["time"]) + pd.DateOffset(months=1)
).dt.to_period("M").dt.to_timestamp()

assert MODEL_PATH.exists(), f"Model not found at {MODEL_PATH}. Run train_forecast_xgboost.py first."
model = xgb.Booster()
model.load_model(MODEL_PATH.as_posix())

# ── climatological baseline ───────────────────────────────────────────────────
train["month_num"] = (
    pd.to_datetime(train["time"]) + pd.DateOffset(months=1)
).dt.month
clim_probs_by_month: dict = {}
for m in range(1, 13):
    subset = train[train["month_num"] == m]["target_label"]
    freqs  = subset.value_counts(normalize=True)
    clim_probs_by_month[m] = {c: float(freqs.get(c, 0.0)) for c in CLASSES}

test_target_month = pd.to_datetime(test["month_dt"]).dt.month
test["clim_prob_dry"] = test_target_month.map(
    lambda m: clim_probs_by_month.get(m, {}).get(-1, 1 / 3)
).values


# ── helper functions ──────────────────────────────────────────────────────────

def brier_score(y_true_bin: np.ndarray, prob: np.ndarray) -> float:
    return float(np.mean((prob - y_true_bin) ** 2))


def bss_score(bs_model: float, bs_ref: float) -> float:
    if bs_ref == 0:
        return float("nan")
    return 1.0 - bs_model / bs_ref


def monthly_bss(test_df: pd.DataFrame, prob_col: str) -> float:
    """Aggregate pixel probabilities to monthly level and compute BSS vs climatology."""
    monthly = test_df.groupby("month_dt").agg(
        y_true_dry_frac=("target_label", lambda s: (s == -1).mean()),
        pred_dry_frac   =(prob_col, "mean"),
        clim_dry_frac   =("clim_prob_dry", "mean"),
    )
    obs   = monthly["y_true_dry_frac"].values.astype(float)
    pred  = monthly["pred_dry_frac"].values
    clim  = monthly["clim_dry_frac"].values
    return bss_score(brier_score(obs, pred), brier_score(obs, clim))


# ── compute training-set mean for neutral fill ───────────────────────────────
train_means = train[FEATURES].mean()

# ── define feature groups ─────────────────────────────────────────────────────
BASE_GROUPS: dict[str, list[str]] = {
    "spi_lags":    [f for f in FEATURES if f.startswith("spi")],
    "pr_lags":     [f for f in FEATURES if f.startswith("pr_")],
    "seasonality": [f for f in FEATURES if f.startswith("month_")],
}
OPTIONAL_GROUPS: dict[str, list[str]] = {
    "enso":        [f for f in FEATURES if f.startswith("nino")],
    "pdo":         [f for f in FEATURES if f.startswith("pdo")],
}
# include optional groups only when the features are actually present
ALL_GROUPS = {
    name: feats for name, feats in {**BASE_GROUPS, **OPTIONAL_GROUPS}.items()
    if feats  # skip if no matching columns
}

# ── all-features baseline ─────────────────────────────────────────────────────
print("Computing all-features baseline...")
dtest_full = xgb.DMatrix(test[FEATURES], feature_names=FEATURES)
probs_full = model.predict(dtest_full)          # (n, 3)
test["prob_dry_full"] = probs_full[:, 0]
bss_full = monthly_bss(test, "prob_dry_full")
print(f"  All features: BSS = {bss_full:.4f}")

# ── ablation loop ─────────────────────────────────────────────────────────────
rows: list[dict] = [{"group": "all_features", "ablated_features": "", "bss": bss_full,
                     "delta_bss": 0.0, "n_features_ablated": 0}]

for group_name, feats_to_ablate in ALL_GROUPS.items():
    print(f"  Ablating {group_name}: {feats_to_ablate}")
    X_abl = test[FEATURES].copy()
    for f in feats_to_ablate:
        X_abl[f] = float(train_means[f])  # replace with training mean
    dtest_abl = xgb.DMatrix(X_abl, feature_names=FEATURES)
    probs_abl = model.predict(dtest_abl)
    col_name  = f"prob_dry_{group_name}"
    test[col_name] = probs_abl[:, 0]
    bss_abl   = monthly_bss(test, col_name)
    delta     = bss_abl - bss_full          # negative = feature group is helpful
    print(f"    BSS = {bss_abl:.4f}  delta = {delta:+.4f}")
    rows.append({
        "group":              group_name,
        "ablated_features":   ", ".join(feats_to_ablate),
        "bss":                round(bss_abl, 5),
        "delta_bss":          round(delta, 5),
        "n_features_ablated": len(feats_to_ablate),
    })

results_df = pd.DataFrame(rows)
OUT_DIR.mkdir(exist_ok=True)
results_df.to_csv(OUT_CSV, index=False)
print(f"\nWrote: {OUT_CSV}")
print(results_df.to_string(index=False))

# ── bar-plot ──────────────────────────────────────────────────────────────────
ablation_rows = results_df[results_df["group"] != "all_features"].sort_values("delta_bss")

colors = ["#d62728" if d > 0 else "#2171b5" for d in ablation_rows["delta_bss"]]
fig, ax = plt.subplots(figsize=(max(6, len(ablation_rows) * 1.5 + 2), 4))
bars = ax.barh(ablation_rows["group"], ablation_rows["delta_bss"], color=colors, alpha=0.85)
ax.axvline(0, color="black", lw=0.8)
ax.set_xlabel("ΔBSS (ablated − full model)  |  negative = feature group helps")
ax.set_title(
    "Feature ablation — marginal BSS contribution\n"
    f"(full model BSS = {bss_full:.4f} vs climatology; test 2021–2026)"
)
for bar, val in zip(bars, ablation_rows["delta_bss"]):
    ax.text(
        val + (0.002 if val >= 0 else -0.002),
        bar.get_y() + bar.get_height() / 2,
        f"{val:+.4f}",
        va="center",
        ha="left" if val >= 0 else "right",
        fontsize=9,
    )
fig.tight_layout()
fig.savefig(OUT_FIG, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Wrote: {OUT_FIG}")
