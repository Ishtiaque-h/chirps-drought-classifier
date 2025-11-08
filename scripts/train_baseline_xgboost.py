#!/usr/bin/env python
"""
XGBoost baseline for drought classes (dry/normal/wet).

- Uses same time split as previous models:
    train: year <= 2016
    val:   2017–2020
    test:  year >= 2021
- Uses features:
    pr, pr_anom, anom_lag1, anom_lag3, month_sin, month_cos
- Handles missing values natively (XGBoost treats NaN as "missing").
- Saves:
    outputs/xgb_baseline_metrics.txt
    outputs/xgb_baseline_cm.png
    outputs/xgb_baseline_feature_importance.png
    outputs/xgb_baseline_model.json
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

import xgboost as xgb

DATA = Path("data/processed/dataset_baseline.parquet")
OUT_DIR = Path("outputs")
OUT_DIR.mkdir(exist_ok=True)

# ---------- load & split ----------
df = pd.read_parquet(DATA)
df["year"] = df["year"].astype(int)

train = df[df["year"] <= 2016]
val   = df[(df["year"] >= 2017) & (df["year"] <= 2020)]
test  = df[df["year"] >= 2021]

features = ["pr", "pr_anom", "anom_lag1", "anom_lag3", "month_sin", "month_cos"]
target = "drought_label"

X_train, y_train = train[features], train[target]
X_val,   y_val   = val[features],   val[target]
X_test,  y_test  = test[features],  test[target]

print("Train shape:", X_train.shape)
print("Val shape:  ", X_val.shape)
print("Test shape: ", X_test.shape)

# XGBoost expects class labels as 0,1,2. Map:
label_map = {-1: 0, 0: 1, 1: 2}
inv_label_map = {v: k for k, v in label_map.items()}

y_train_enc = y_train.map(label_map).values
y_val_enc   = y_val.map(label_map).values
y_test_enc  = y_test.map(label_map).values

# ---------- DMatrix ----------
dtrain = xgb.DMatrix(X_train, label=y_train_enc, feature_names=features)
dval   = xgb.DMatrix(X_val,   label=y_val_enc,   feature_names=features)
dtest  = xgb.DMatrix(X_test,  label=y_test_enc,  feature_names=features)

# ---------- XGBoost params ----------
# NOTE: For GPU build, you can change tree_method to "gpu_hist".
params = {
    "objective": "multi:softprob",
    "num_class": 3,
    "eval_metric": "mlogloss",
    "tree_method": "hist",
    "device": "cuda",       # for GPU support    
    "eta": 0.05,
    "max_depth": 8,
    "min_child_weight": 5,
    "subsample": 0.9,
    "colsample_bytree": 0.9,
    "lambda": 1.0,           # L2 regularization
    "alpha": 0.0,            # L1 regularization
}

num_boost_round = 2000
early_stopping_rounds = 50

evals = [(dtrain, "train"), (dval, "val")]

print("Training XGBoost...")
model = xgb.train(
    params=params,
    dtrain=dtrain,
    num_boost_round=num_boost_round,
    evals=evals,
    early_stopping_rounds=early_stopping_rounds,
    verbose_eval=50,
)

print(f"Best iteration: {model.best_iteration}")

# ---------- predictions on test ----------
probs = model.predict(dtest, iteration_range=(0, model.best_iteration + 1))
y_pred_enc = probs.argmax(axis=1)
y_pred = pd.Series(y_pred_enc).map(inv_label_map).values

# ---------- metrics ----------
report = classification_report(
    y_test, y_pred,
    target_names=["dry(-1)", "normal(0)", "wet(1)"],
    digits=3,
)
print(report)

# confusion matrix (normalized by true class)
cm = confusion_matrix(y_test, y_pred, labels=[-1, 0, 1], normalize="true")
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["dry", "normal", "wet"])
disp.plot(cmap="Blues", values_format=".2f")
plt.title("XGBoost baseline — Normalized Confusion Matrix")
cm_path = OUT_DIR / "xgb_baseline_cm.png"
plt.savefig(cm_path, dpi=150, bbox_inches="tight")
plt.close()

# ---------- feature importance ----------
# use "gain" importance (average gain of splits using the feature)
score = model.get_score(importance_type="gain")
imp_values = [score.get(f, 0.0) for f in features]
imp = pd.Series(imp_values, index=features).sort_values(ascending=True)

ax = imp.plot(kind="barh", figsize=(6, 4))
ax.set_xlabel("Gain-based feature importance")
ax.set_title("XGBoost baseline — Feature importance")
fi_path = OUT_DIR / "xgb_baseline_feature_importance.png"
plt.tight_layout()
plt.savefig(fi_path, dpi=150, bbox_inches="tight")
plt.close()

# ---------- save metrics & model ----------
metrics_path = OUT_DIR / "xgb_baseline_metrics.txt"
with open(metrics_path, "w") as f:
    f.write(report)

model_path = OUT_DIR / "xgb_baseline_model.json"
model.save_model(model_path.as_posix())

print("Wrote:", metrics_path)
print("Wrote:", cm_path)
print("Wrote:", fi_path)
print("Wrote model:", model_path)
