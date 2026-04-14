#!/usr/bin/env python
"""
Case study: 2021–2022 multi-year drought and 2023 atmospheric river events.

The test period (2021–2026) contains two climatically notable sequences:
  1. 2021–2022: one of California's most severe multi-year droughts on record,
     driven by consecutive La Niña winters and below-normal Sierra snowpack.
  2. 2023 Jan–Mar: a series of atmospheric rivers that reversed the drought,
     delivering above-normal precipitation and widespread flooding.

This case study plots the regional-mean model dry-class probability together
with the observed SPI-1 time series.  Annotating the plot with these known
events demonstrates that the model captures the physically correct signal
— a key requirement for applied hydrology journals.

Inputs:
  data/processed/dataset_forecast.parquet
  data/processed/chirps_v3_monthly_cvalley_spi_1991_2026.nc
  outputs/forecast_xgb_model.json

Outputs:
  outputs/case_study_2021_2026.png
"""
from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr
import xgboost as xgb
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

DATA       = Path("data/processed/dataset_forecast.parquet")
SPI_FILE   = Path("data/processed/chirps_v3_monthly_cvalley_spi_1991_2025.nc")
MODEL_PATH = Path("outputs/forecast_xgb_model.json")
OUT_DIR    = Path("outputs"); OUT_DIR.mkdir(exist_ok=True)

FEATURES = [
    "spi1_lag1", "spi1_lag2", "spi1_lag3",
    "spi3_lag1", "spi6_lag1",
    "pr_lag1", "pr_lag2", "pr_lag3",
    "month_sin", "month_cos",
]
TARGET  = "target_label"
INV_MAP = {0: -1, 1: 0, 2: 1}

# ── load data ─────────────────────────────────────────────────────────────────
print("Loading dataset and model...")
df = pd.read_parquet(DATA)
df["year"] = df["year"].astype(int)
test = df[df["year"] >= 2021].copy()

assert MODEL_PATH.exists(), f"Model not found: {MODEL_PATH}. Run train_forecast_xgboost.py first."
model = xgb.Booster()
model.load_model(MODEL_PATH.as_posix())

dtest = xgb.DMatrix(test[FEATURES], feature_names=FEATURES)
probs = model.predict(dtest)

test["prob_dry"]    = probs[:, 0]   # class 0 = dry (-1) in XGBoost encoding
test["prob_normal"] = probs[:, 1]
test["prob_wet"]    = probs[:, 2]
test["pred_label"]  = [INV_MAP[i] for i in probs.argmax(axis=1)]

# target month timestamp
test["month_dt"] = (
    pd.to_datetime(test["time"]) + pd.DateOffset(months=1)
).dt.to_period("M").dt.to_timestamp()

# ── monthly aggregation ───────────────────────────────────────────────────────
monthly = test.groupby("month_dt").agg(
    prob_dry_mean    = ("prob_dry",    "mean"),
    prob_wet_mean    = ("prob_wet",    "mean"),
    obs_dry_frac     = (TARGET,        lambda s: (s == -1).mean()),
    obs_wet_frac     = (TARGET,        lambda s: (s ==  1).mean()),
    pred_dry_frac    = ("pred_label",  lambda s: (s == -1).mean()),
).reset_index()

# ── load regional-mean SPI-1 for reference ─────────────────────────────────────
if SPI_FILE.exists():
    spi_ds   = xr.open_dataset(SPI_FILE).load()
    spi1_da  = spi_ds["spi1"]
    spi1_mean = spi1_da.mean(dim=["latitude", "longitude"])
    spi1_df  = spi1_mean.to_dataframe(name="spi1_mean").reset_index()
    spi1_df["month_dt"] = pd.to_datetime(spi1_df["time"]).dt.to_period("M").dt.to_timestamp()
    # shift to target month: regional observed SPI-1 at t+1
    spi1_df  = spi1_df.sort_values("month_dt")
    spi1_df["spi1_target"] = spi1_df["spi1_mean"].shift(-1)
    spi1_test = spi1_df[spi1_df["month_dt"].dt.year >= 2021].set_index("month_dt")
    monthly   = monthly.set_index("month_dt").join(
        spi1_test[["spi1_target"]], how="left"
    ).reset_index()
else:
    monthly["spi1_target"] = np.nan

# ── annotate event periods ────────────────────────────────────────────────────
# 2021–2022 drought: Jan 2021 – Dec 2022 (extend to start of Jan 2023 to include Dec 2022)
drought_start = pd.Timestamp("2021-01-01")
drought_end   = pd.Timestamp("2023-01-01")

# 2023 atmospheric river events: Jan 2023 – Mar 2023
ar_start = pd.Timestamp("2023-01-01")
ar_end   = pd.Timestamp("2023-03-01")

# ── plot ──────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 1, figsize=(13, 9), sharex=True,
                          gridspec_kw={"hspace": 0.08})

t = monthly["month_dt"]

# ─ upper panel: dry-class probability and SPI-1 ─
ax1 = axes[0]
ax1.fill_between(t, monthly["prob_dry_mean"], alpha=0.25, color="#d73027",
                 label="Model P(dry) — shaded area")
ax1.plot(t, monthly["prob_dry_mean"], color="#d73027", linewidth=1.8,
         label="Model P(dry) — mean over region")

if "spi1_target" in monthly.columns and not monthly["spi1_target"].isna().all():
    ax1_r = ax1.twinx()
    ax1_r.plot(t, monthly["spi1_target"], color="steelblue", linewidth=1.4,
               linestyle="--", label="Observed SPI-1 (regional mean, t+1)")
    ax1_r.axhline(-1, color="steelblue", lw=0.8, linestyle=":", alpha=0.6)
    ax1_r.axhline( 1, color="steelblue", lw=0.8, linestyle=":", alpha=0.6)
    ax1_r.set_ylabel("Observed SPI-1", color="steelblue")
    ax1_r.tick_params(axis="y", labelcolor="steelblue")
    ax1_r.legend(loc="upper right", fontsize=8)

ax1.set_ylabel("P(dry) — model probability")
ax1.set_ylim(-0.05, 1.05)
ax1.set_title(
    "Case study: 2021–2025 drought forecast — Central Valley\n"
    "1-month-ahead XGBoost model (target: SPI-1 class)",
    fontsize=10,
)
ax1.legend(loc="upper left", fontsize=8)

# ─ lower panel: observed vs. predicted dry fraction ─
ax2 = axes[1]
ax2.plot(t, monthly["obs_dry_frac"],  color="#d73027", linewidth=1.8,
         label="Observed dry fraction (SPI-1)")
ax2.plot(t, monthly["pred_dry_frac"], color="#fc8d59", linewidth=1.8,
         linestyle="--", label="Predicted dry fraction")
ax2.plot(t, monthly["obs_wet_frac"],  color="#4575b4", linewidth=1.4,
         alpha=0.7, label="Observed wet fraction")
ax2.set_ylabel("Fraction of region")
ax2.set_ylim(-0.05, 1.05)
ax2.legend(loc="upper right", fontsize=8)
ax2.set_xlabel("Month (target)")

# ─ annotate drought and AR event periods on both panels ─
for ax in axes:
    ax.axvspan(drought_start, drought_end, color="khaki", alpha=0.35, zorder=0)
    ax.axvspan(ar_start,      ar_end,      color="lightblue", alpha=0.45, zorder=0)

# legend patches for shaded regions
drought_patch = mpatches.Patch(color="khaki",     alpha=0.5, label="2021–2022 multi-year drought")
ar_patch      = mpatches.Patch(color="lightblue", alpha=0.6, label="2023 atmospheric rivers")
fig.legend(handles=[drought_patch, ar_patch], loc="lower center",
           ncol=2, fontsize=9, bbox_to_anchor=(0.5, -0.02))

fig.tight_layout(rect=[0, 0.03, 1, 1])
out_path = OUT_DIR / "case_study_2021_2026.png"
fig.savefig(out_path, dpi=150, bbox_inches="tight")
plt.close()
print("Wrote:", out_path)
