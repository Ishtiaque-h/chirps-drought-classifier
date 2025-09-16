#!/usr/bin/env python
"""
Train a baseline multinomial logistic regression to classify
dry / normal / wet labels using CHIRPS features, with NaN-safe preprocessing.

Input:  data/processed/dataset_baseline.parquet
Output: outputs/baseline_logreg_metrics.txt + confusion matrix PNG
"""
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

DATA = Path("data/processed/dataset_baseline.parquet")
OUT = Path("outputs"); OUT.mkdir(exist_ok=True)

# --- load ---
df = pd.read_parquet(DATA)
df["year"] = df["year"].astype(int)

# --- time split ---
train = df[df["year"] <= 2016]
val   = df[(df["year"] >= 2017) & (df["year"] <= 2020)]
test  = df[df["year"] >= 2021]

features = ["pr","pr_anom","anom_lag1","anom_lag3","month_sin","month_cos"]
target   = "drought_label"

X_train, y_train = train[features], train[target]
X_val,   y_val   = val[features],   val[target]
X_test,  y_test  = test[features],  test[target]

# (optional) quick sanity peek on NaNs
print("NaN counts (train):\n", X_train.isna().sum())

# --- pipeline: impute (median from train) -> scale -> logistic regression ---
pipe = Pipeline(steps=[
    ("impute", SimpleImputer(strategy="median")),
    ("scale",  StandardScaler()),
    ("clf",    LogisticRegression(max_iter=500, multi_class="multinomial", class_weight="balanced")),
])

pipe.fit(X_train, y_train)

# --- evaluate on test ---
y_pred = pipe.predict(X_test)
report = classification_report(y_test, y_pred, target_names=["dry(-1)","normal(0)","wet(1)"], digits=3)
print(report)

# confusion matrix (normalized by true class)
cm = confusion_matrix(y_test, y_pred, labels=[-1,0,1], normalize="true")
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["dry","normal","wet"])
disp.plot(cmap="Blues", values_format=".2f")
plt.title("Baseline Logistic Regression — Normalized Confusion Matrix")
cm_path = OUT / "baseline_logreg_cm.png"
plt.savefig(cm_path, dpi=150, bbox_inches="tight"); plt.close()

# save metrics
metrics_path = OUT / "baseline_logreg_metrics.txt"
with open(metrics_path, "w") as f:
    f.write(report)
print("Wrote:", metrics_path)
print("Wrote:", cm_path)
