#!/usr/bin/env python
"""
XGBoost forecast model augmented with spatial-neighbourhood features.

Motivation
----------
The pixel-level tabular model (train_forecast_xgboost.py) treats every grid cell
independently.  Real drought systems are spatially coherent: a dry anomaly in
neighbouring cells strongly predicts drought at the target cell.  This script
adds a 3×3-neighbourhood mean for each SPI channel (computed from the gridded
NetCDF) as additional features and retrains XGBoost.

Spatial features added (one per SPI channel)
  spi1_nbr_mean  — 3×3 rolling mean of SPI-1 around each pixel at lag-1
  spi3_nbr_mean  — same for SPI-3
  spi6_nbr_mean  — same for SPI-6
  pr_nbr_mean    — same for raw precipitation at lag-1

These are computed with xarray rolling (min_periods=1, centre=True) so every
pixel has a value regardless of boundary position.

Time split: train ≤ 2016, val 2017–2020, test ≥ 2021  (matches all other scripts)

Outputs
  outputs/xgb_spatial_metrics.txt
  outputs/xgb_spatial_cm.png
  outputs/xgb_spatial_feature_importance.png
  outputs/xgb_spatial_model.json
    outputs/xgb_spatial_test_probs.npz   (calibrated `proba` + uncalibrated `proba_raw`)

Inputs
  data/processed/dataset_forecast.parquet
  data/processed/chirps_v3_monthly_cvalley_1991_2026.nc
  data/processed/chirps_v3_monthly_cvalley_spi_1991_2026.nc
"""
from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import (
    classification_report, confusion_matrix, ConfusionMatrixDisplay,
)
from sklearn.utils.class_weight import compute_sample_weight
from feature_config import get_feature_columns

BASE_DIR   = Path(__file__).resolve().parents[1]
PROC       = BASE_DIR / "data" / "processed"
OUT_DIR    = BASE_DIR / "outputs"; OUT_DIR.mkdir(exist_ok=True)

PARQUET    = PROC / "dataset_forecast.parquet"
PR_FILE    = PROC / "chirps_v3_monthly_cvalley_1991_2026.nc"
SPI_FILE   = PROC / "chirps_v3_monthly_cvalley_spi_1991_2026.nc"

FEATURES_SPATIAL = [
    "spi1_nbr_mean",
    "spi3_nbr_mean",
    "spi6_nbr_mean",
    "pr_nbr_mean",
]
TARGET    = "target_label"
LABEL_MAP = {-1: 0, 0: 1, 1: 2}

# --------------------------------------------------------------------------
# 1. Build spatial neighbourhood features from the gridded NetCDF
# --------------------------------------------------------------------------
print("Building spatial-neighbourhood features ...")
pr_ds  = xr.open_dataset(PR_FILE).load()
spi_ds = xr.open_dataset(SPI_FILE).load()

pr   = pr_ds["pr"].astype("float32")
spi1 = spi_ds["spi1"].astype("float32")
spi3 = spi_ds["spi3"].astype("float32")
spi6 = spi_ds["spi6"].astype("float32")

times = pr.time.values
spi1  = spi1.sel(time=times)
spi3  = spi3.sel(time=times)
spi6  = spi6.sel(time=times)

lat_name = "latitude" if "latitude" in pr.coords else "lat"
lon_name = "longitude" if "longitude" in pr.coords else "lon"

# 3×3 neighbourhood mean: rolling window of size 3 in both spatial dims
def nbr_mean(da: xr.DataArray, name: str) -> xr.DataArray:
    rolled = (
        da.rolling({lat_name: 3, lon_name: 3}, min_periods=1, center=True)
          .mean()
    )
    rolled.name = name
    return rolled

spi1_nbr = nbr_mean(spi1, "spi1_nbr_mean")
spi3_nbr = nbr_mean(spi3, "spi3_nbr_mean")
spi6_nbr = nbr_mean(spi6, "spi6_nbr_mean")
pr_nbr   = nbr_mean(pr,   "pr_nbr_mean")

# Stack to a flat DataFrame keyed by (time, latitude, longitude)
nbr_ds = xr.Dataset({
    "spi1_nbr_mean": spi1_nbr,
    "spi3_nbr_mean": spi3_nbr,
    "spi6_nbr_mean": spi6_nbr,
    "pr_nbr_mean":   pr_nbr,
}).stack(pixel=(lat_name, lon_name))

nbr_df = nbr_ds.reset_index("pixel").to_dataframe()
if "time" not in nbr_df.columns:
    nbr_df = nbr_df.reset_index()

nbr_df = nbr_df.rename(columns={lat_name: "latitude", lon_name: "longitude"})
nbr_df["time"] = pd.to_datetime(nbr_df["time"])

print(f"  Neighbourhood feature table: {nbr_df.shape}")

# --------------------------------------------------------------------------
# 2. Load the tabular forecasting dataset and merge neighbourhood features
# --------------------------------------------------------------------------
print("Loading tabular dataset ...")
df = pd.read_parquet(PARQUET)
df["time"] = pd.to_datetime(df["time"])
df["year"] = df["year"].astype(int)

# The neighbourhood features are keyed at t (lag-1 time step of the target),
# which in build_dataset_forecast.py is the "time" column.
df = df.merge(
    nbr_df[["time", "latitude", "longitude"] + FEATURES_SPATIAL],
    on=["time", "latitude", "longitude"],
    how="left",
)

missing = df[FEATURES_SPATIAL].isna().sum().sum()
if missing > 0:
    print(f"  Warning: {missing} NaN values in spatial features — filling with 0")
    df[FEATURES_SPATIAL] = df[FEATURES_SPATIAL].fillna(0.0)

print(f"  Merged dataset shape: {df.shape}")
FEATURES = get_feature_columns(df.columns) + FEATURES_SPATIAL

# --------------------------------------------------------------------------
# 3. Train / val / test split
# --------------------------------------------------------------------------
train = df[df["year"] <= 2016]
val   = df[(df["year"] >= 2017) & (df["year"] <= 2020)]
test  = df[df["year"] >= 2021]

print(f"  Train {train.shape}  Val {val.shape}  Test {test.shape}")

X_train, y_train = train[FEATURES], train[TARGET]
X_val,   y_val   = val[FEATURES],   val[TARGET]
X_test,  y_test  = test[FEATURES],  test[TARGET]

y_train_enc = y_train.map(LABEL_MAP).values
y_val_enc   = y_val.map(LABEL_MAP).values
y_test_enc  = y_test.map(LABEL_MAP).values

# --------------------------------------------------------------------------
# 4. Train XGBoost
# --------------------------------------------------------------------------
train_weights = compute_sample_weight(class_weight="balanced", y=y_train_enc)

dtrain = xgb.DMatrix(X_train, label=y_train_enc, weight=train_weights, feature_names=FEATURES)
dval   = xgb.DMatrix(X_val,   label=y_val_enc,   feature_names=FEATURES)
dtest  = xgb.DMatrix(X_test,  label=y_test_enc,  feature_names=FEATURES)

params = {
    "objective":        "multi:softprob",
    "num_class":        3,
    "eval_metric":      "mlogloss",
    "tree_method":      "hist",
    "device":           "cuda",
    "eta":              0.05,
    "max_depth":        8,
    "min_child_weight": 5,
    "subsample":        0.9,
    "colsample_bytree": 0.9,
    "lambda":           1.0,
    "alpha":            0.1,
}

print("Training XGBoost (spatial features) ...")
model = xgb.train(
    params=params,
    dtrain=dtrain,
    num_boost_round=2000,
    evals=[(dtrain, "train"), (dval, "val")],
    early_stopping_rounds=50,
    verbose_eval=200,
)

# --------------------------------------------------------------------------
# 5. Calibrate probabilities (isotonic on validation set)
# --------------------------------------------------------------------------
print("Calibrating XGBoost-Spatial probabilities (validation-set isotonic) ...")
proba_val = model.predict(dval).reshape(-1, 3)
proba_test_raw = model.predict(dtest).reshape(-1, 3)

iso_models = []
proba_test_cal = np.zeros_like(proba_test_raw, dtype="float32")
for k in range(3):
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(proba_val[:, k], (y_val_enc == k).astype(int))
    proba_test_cal[:, k] = iso.predict(proba_test_raw[:, k]).astype("float32")
    iso_models.append(iso)

# Ensure calibrated probabilities form a valid simplex row-wise.
row_sum = proba_test_cal.sum(axis=1, keepdims=True)
row_sum[row_sum <= 0] = 1.0
proba_test = (proba_test_cal / row_sum).astype("float32")

# --------------------------------------------------------------------------
# 6. Evaluate
# --------------------------------------------------------------------------
inv_label_map = {v: k for k, v in LABEL_MAP.items()}

y_pred_enc = proba_test.argmax(axis=1)
y_pred     = np.array([inv_label_map[p] for p in y_pred_enc])
y_true     = y_test.values

report = classification_report(
    y_true, y_pred,
    target_names=["Dry (−1)", "Normal (0)", "Wet (+1)"],
)
print(report)

acc  = (y_pred == y_true).mean()
f1s  = {}
for cls_name, cls_val in [("dry", -1), ("normal", 0), ("wet", 1)]:
    from sklearn.metrics import f1_score
    f1s[cls_name] = f1_score(y_true == cls_val, y_pred == cls_val, average="binary")

metrics_text = (
    f"XGBoost + Spatial Neighbours — Test Metrics\n"
    f"{'='*50}\n"
    f"Overall Accuracy : {acc:.4f}\n"
    f"F1 Dry (−1)      : {f1s['dry']:.4f}\n"
    f"F1 Normal (0)    : {f1s['normal']:.4f}\n"
    f"F1 Wet (+1)      : {f1s['wet']:.4f}\n\n"
    f"{report}\n"
    f"Features used    : {FEATURES}\n"
    f"Spatial window   : 3×3 neighbourhood mean (min_periods=1, centre=True)\n"
)
print(metrics_text)

(OUT_DIR / "xgb_spatial_metrics.txt").write_text(metrics_text)

# --------------------------------------------------------------------------
# 7. Confusion matrix
# --------------------------------------------------------------------------
cm  = confusion_matrix(y_true, y_pred, labels=[-1, 0, 1])
fig, ax = plt.subplots(figsize=(5, 4))
disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Dry (−1)", "Normal (0)", "Wet (+1)"],
)
disp.plot(ax=ax, colorbar=False, cmap="Blues")
ax.set_title("XGBoost + Spatial Features — Test Confusion Matrix")
plt.tight_layout()
fig.savefig(OUT_DIR / "xgb_spatial_cm.png", dpi=150)
plt.close(fig)

# --------------------------------------------------------------------------
# 8. Feature importance
# --------------------------------------------------------------------------
importance = model.get_score(importance_type="gain")
imp_df = (
    pd.DataFrame.from_dict(importance, orient="index", columns=["gain"])
    .reset_index().rename(columns={"index": "feature"})
    .sort_values("gain", ascending=False)
)

fig, ax = plt.subplots(figsize=(8, 6))
ax.barh(imp_df["feature"][:15][::-1], imp_df["gain"][:15][::-1], color="steelblue")
ax.set_xlabel("Gain")
ax.set_title("XGBoost + Spatial Features — Top-15 Feature Importance")
plt.tight_layout()
fig.savefig(OUT_DIR / "xgb_spatial_feature_importance.png", dpi=150)
plt.close(fig)

# --------------------------------------------------------------------------
# 9. Save model and calibrated probabilities
# --------------------------------------------------------------------------
model.save_model(str(OUT_DIR / "xgb_spatial_model.json"))

np.savez(
    OUT_DIR / "xgb_spatial_test_probs.npz",
    proba=proba_test.astype("float32"),
    proba_raw=proba_test_raw.astype("float32"),
    y_true=y_true,
    features=FEATURES,
)

# Save raw validation probabilities so that evaluate_forecast_skill.py can
# fit and compare alternative calibrators (Platt, isotonic) using validation-
# only fitting with a frozen test set.
np.savez(
    OUT_DIR / "xgb_spatial_val_probs.npz",
    proba_raw=proba_val.astype("float32"),
    y_val_enc=y_val_enc,
)

print("Saved outputs to", OUT_DIR)
