#!/usr/bin/env python
"""
XGBoost classifier on the SPI-based forecasting dataset.

- Target: drought_label_spi1 at t+1 (next month's SPI-1 drought class)
- Features: SPI-1/3/6 lags, raw precipitation lags, cyclic month
- Time split: train <= 2016, val 2017–2020, test >= 2021

Outputs:
  outputs/forecast_xgb_metrics.txt
  outputs/forecast_xgb_cm.png
  outputs/forecast_xgb_feature_importance.png
  outputs/forecast_xgb_model.json
  outputs/forecast_xgb_test_probs.npz   (raw softmax probabilities for calibration)
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.utils.class_weight import compute_sample_weight
from feature_config import get_feature_columns

DATA    = Path("data/processed/dataset_forecast.parquet")
OUT_DIR = Path("outputs"); OUT_DIR.mkdir(exist_ok=True)

df = pd.read_parquet(DATA)
df["year"] = df["year"].astype(int)

train = df[df["year"] <= 2016]
val   = df[(df["year"] >= 2017) & (df["year"] <= 2020)]
test  = df[df["year"] >= 2021]

FEATURES = get_feature_columns(df.columns)
TARGET = "target_label"

X_train, y_train = train[FEATURES], train[TARGET]
X_val,   y_val   = val[FEATURES],   val[TARGET]
X_test,  y_test  = test[FEATURES],  test[TARGET]

print(f"Train: {X_train.shape}  Val: {X_val.shape}  Test: {X_test.shape}")

# XGBoost requires labels 0, 1, 2
label_map     = {-1: 0, 0: 1, 1: 2}
inv_label_map = {v: k for k, v in label_map.items()}

y_train_enc = y_train.map(label_map).values
y_val_enc   = y_val.map(label_map).values
y_test_enc  = y_test.map(label_map).values

train_weights = compute_sample_weight(class_weight="balanced", y=y_train_enc)

dtrain = xgb.DMatrix(X_train, label=y_train_enc, weight=train_weights, feature_names=FEATURES)
dval   = xgb.DMatrix(X_val,   label=y_val_enc,   feature_names=FEATURES)
dtest  = xgb.DMatrix(X_test,  label=y_test_enc,  feature_names=FEATURES)

params = {
    "objective":       "multi:softprob",
    "num_class":       3,
    "eval_metric":     "mlogloss",
    "tree_method":     "hist",
    "device":          "cuda",
    "eta":             0.05,
    "max_depth":       8,
    "min_child_weight": 5,
    "subsample":       0.9,
    "colsample_bytree": 0.9,
    "lambda":          1.0,
    "alpha":           0.1,
}

print("Training XGBoost...")
model = xgb.train(
    params=params,
    dtrain=dtrain,
    num_boost_round=2000,
    evals=[(dtrain, "train"), (dval, "val")],
    early_stopping_rounds=50,
    verbose_eval=100,
)
print(f"Best iteration: {model.best_iteration}")

# ---------- evaluate ----------
probs    = model.predict(dtest, iteration_range=(0, model.best_iteration + 1))
y_pred_enc = probs.argmax(axis=1)
y_pred   = pd.Series(y_pred_enc).map(inv_label_map).values

report = classification_report(
    y_test, y_pred,
    target_names=["dry(-1)", "normal(0)", "wet(+1)"],
    digits=3,
)
print(report)

cm = confusion_matrix(y_test, y_pred, labels=[-1, 0, 1], normalize="true")
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["dry", "normal", "wet"])
disp.plot(cmap="Blues", values_format=".2f")
plt.title("Forecast XGBoost — Normalized Confusion Matrix")
cm_path = OUT_DIR / "forecast_xgb_cm.png"
plt.savefig(cm_path, dpi=150, bbox_inches="tight")
plt.close()

# ---------- feature importance (gain) ----------
score = model.get_score(importance_type="gain")
imp_values = [score.get(f, 0.0) for f in FEATURES]
imp = pd.Series(imp_values, index=FEATURES).sort_values(ascending=True)
ax  = imp.plot(kind="barh", figsize=(7, 5))
ax.set_xlabel("Gain-based feature importance")
ax.set_title("Forecast XGBoost — Feature importance (gain)")
fi_path = OUT_DIR / "forecast_xgb_feature_importance.png"
plt.tight_layout()
plt.savefig(fi_path, dpi=150, bbox_inches="tight")
plt.close()

# ---------- save ----------
metrics_path = OUT_DIR / "forecast_xgb_metrics.txt"
with open(metrics_path, "w") as f:
    f.write(report)

model_path = OUT_DIR / "forecast_xgb_model.json"
model.save_model(model_path.as_posix())

# save raw softmax probabilities for downstream calibration (evaluate_forecast_skill.py)
probs_path = OUT_DIR / "forecast_xgb_test_probs.npz"
test_times = test["time"].values
np.savez_compressed(
    probs_path,
    probs=probs,          # shape (n_rows, 3): columns = [dry_prob, normal_prob, wet_prob]
    y_true=y_test.values,
    times=test_times,
    latitude=test["latitude"].values,
    longitude=test["longitude"].values,
    features=np.array(FEATURES),
    best_iteration=model.best_iteration,
)

print("Wrote:", metrics_path)
print("Wrote:", cm_path)
print("Wrote:", fi_path)
print("Wrote model:", model_path)
print("Wrote probs:", probs_path)
