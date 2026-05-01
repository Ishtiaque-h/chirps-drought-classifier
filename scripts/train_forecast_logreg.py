#!/usr/bin/env python
"""
Multinomial logistic regression on the SPI-based forecasting dataset.

- Target: drought_label_spi1 at t+1 (next month's SPI-1 drought class)
- Features: SPI-1/3/6 lags, raw precipitation lags, cyclic month
- Time split: train <= 2016, val 2017–2020, test >= 2021

Outputs:
  outputs/forecast_logreg_metrics.txt
  outputs/forecast_logreg_val_metrics.txt
  outputs/forecast_logreg_cm.png
  outputs/forecast_logreg_model.pkl
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
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
print("NaN counts (train):\n", X_train.isna().sum())

pipe = Pipeline([
    ("impute", SimpleImputer(strategy="median")),
    ("scale",  StandardScaler()),
    ("clf",    LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        solver="lbfgs",
        C=1.0,
    )),
])

pipe.fit(X_train, y_train)
joblib.dump(pipe, OUT_DIR / "forecast_logreg_model.pkl")

# ---- validation-set metrics (model monitoring, not reported as primary result) ----
y_val_pred  = pipe.predict(X_val)
val_report  = classification_report(
    y_val, y_val_pred,
    target_names=["dry(-1)", "normal(0)", "wet(+1)"],
    digits=3,
)
print("--- Validation set ---")
print(val_report)
val_metrics_path = OUT_DIR / "forecast_logreg_val_metrics.txt"
with open(val_metrics_path, "w") as f:
    f.write(val_report)
print("Wrote:", val_metrics_path)

y_pred = pipe.predict(X_test)
report = classification_report(
    y_test, y_pred,
    target_names=["dry(-1)", "normal(0)", "wet(+1)"],
    digits=3,
)
print(report)

cm = confusion_matrix(y_test, y_pred, labels=[-1, 0, 1], normalize="true")
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["dry", "normal", "wet"])
disp.plot(cmap="Blues", values_format=".2f")
plt.title("Forecast Logistic Regression — Normalized Confusion Matrix")
cm_path = OUT_DIR / "forecast_logreg_cm.png"
plt.savefig(cm_path, dpi=150, bbox_inches="tight")
plt.close()

metrics_path = OUT_DIR / "forecast_logreg_metrics.txt"
with open(metrics_path, "w") as f:
    f.write(report)

print("Wrote:", metrics_path)
print("Wrote:", cm_path)
print("Wrote:", OUT_DIR / "forecast_logreg_model.pkl")
