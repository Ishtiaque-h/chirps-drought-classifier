#!/usr/bin/env python
"""
Regional-level accuracy evaluation of the forecast XGBoost model.

For each month in the test period (2021–2026):
  - Compute the fraction of pixels predicted as dry / normal / wet
  - Compute the fraction of ground-truth pixels in each class
  - Report the % of months where the dominant predicted class matches truth

Outputs:
  outputs/regional_forecast_comparison.png  — stacked area chart pred vs actual
  outputs/regional_forecast_metrics.txt     — month-level accuracy summary

Inputs:
  data/processed/dataset_forecast.parquet
  outputs/forecast_xgb_model.json
"""
from pathlib import Path
import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

DATA    = Path("data/processed/dataset_forecast.parquet")
MODEL   = Path("outputs/forecast_xgb_model.json")
OUT_DIR = Path("outputs"); OUT_DIR.mkdir(exist_ok=True)

FEATURES = [
    "spi1_lag1", "spi1_lag2", "spi1_lag3",
    "spi3_lag1", "spi6_lag1",
    "pr_lag1", "pr_lag2", "pr_lag3",
    "month_sin", "month_cos",
]
TARGET = "target_label"

# ---- load ----
print("Loading data and model...")
df = pd.read_parquet(DATA)
df["year"] = df["year"].astype(int)
test = df[df["year"] >= 2021].copy()

model = xgb.Booster()
model.load_model(MODEL.as_posix())

dtest = xgb.DMatrix(test[FEATURES], feature_names=FEATURES)
probs = model.predict(dtest)
test["pred_label"] = probs.argmax(axis=1)
inv_map = {0: -1, 1: 0, 2: 1}
test["pred_label"] = test["pred_label"].map(inv_map)

# target month timestamp
test["month_dt"] = (
    pd.to_datetime(test["time"]) + pd.DateOffset(months=1)
).dt.to_period("M").dt.to_timestamp()

# ---- monthly spatial aggregation ----
def class_fractions(series: pd.Series) -> pd.Series:
    n = len(series)
    return pd.Series({
        "dry_frac":    (series == -1).sum() / n,
        "normal_frac": (series ==  0).sum() / n,
        "wet_frac":    (series ==  1).sum() / n,
    })

pred_monthly  = test.groupby("month_dt")["pred_label"].apply(class_fractions).unstack()
truth_monthly = test.groupby("month_dt")[TARGET].apply(class_fractions).unstack()

# ---- dominant class per month ----
pred_dominant  = pred_monthly.idxmax(axis=1).str.replace("_frac", "")
truth_dominant = truth_monthly.idxmax(axis=1).str.replace("_frac", "")
match = (pred_dominant == truth_dominant)
dominant_acc = match.mean()

print(f"Months in test set : {len(pred_monthly)}")
print(f"Dominant class accuracy: {dominant_acc:.1%}")
print(f"  Correct months: {match.sum()} / {len(match)}")

# ---- full metrics text ----
metrics_lines = [
    "Regional Forecast Evaluation — Central Valley 2021–2026",
    "=" * 55,
    f"Test months                    : {len(pred_monthly)}",
    f"Dominant-class accuracy        : {dominant_acc:.1%}",
    f"Correct months                 : {match.sum()} / {len(match)}",
    "",
    "Monthly class fractions (predicted vs. actual):",
    pred_monthly.assign(source="predicted").to_string(),
    "",
    truth_monthly.assign(source="actual").to_string(),
    "",
    "Dominant class match per month:",
    pd.DataFrame({"predicted": pred_dominant, "actual": truth_dominant,
                  "match": match}).to_string(),
]
metrics_txt = "\n".join(metrics_lines)
print(metrics_txt)
(OUT_DIR / "regional_forecast_metrics.txt").write_text(metrics_txt)

# ---- stacked area chart: predicted vs. actual ----
colors = {"dry_frac": "#d73027", "normal_frac": "#fee090", "wet_frac": "#4575b4"}
labels = {"dry_frac": "Dry", "normal_frac": "Normal", "wet_frac": "Wet"}

fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True)

for ax, (data, title) in zip(axes, [(pred_monthly, "Predicted"), (truth_monthly, "Actual")]):
    ax.stackplot(
        data.index,
        data["dry_frac"], data["normal_frac"], data["wet_frac"],
        labels=["Dry", "Normal", "Wet"],
        colors=[colors["dry_frac"], colors["normal_frac"], colors["wet_frac"]],
        alpha=0.85,
    )
    ax.set_ylabel("Fraction of region")
    ax.set_title(f"{title} drought class shares — Central Valley 2021–2026")
    ax.set_ylim(0, 1)
    ax.legend(loc="upper right", fontsize=9)

axes[-1].set_xlabel("Month (target)")
fig.suptitle(
    f"Forecast XGBoost: Regional drought composition\n"
    f"Dominant-class accuracy = {dominant_acc:.1%}",
    fontsize=12,
)
fig.tight_layout()
fig.savefig(OUT_DIR / "regional_forecast_comparison.png", dpi=150, bbox_inches="tight")
plt.close()

print("Wrote:", OUT_DIR / "regional_forecast_comparison.png")
print("Wrote:", OUT_DIR / "regional_forecast_metrics.txt")
