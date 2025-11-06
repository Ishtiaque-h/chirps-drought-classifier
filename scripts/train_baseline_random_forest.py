#!/usr/bin/env python
"""
Random Forest baseline on FULL dataset (no subsampling).
Time split: train<=2016, val 2017–2020, test>=2021
Outputs:
  outputs/rf_full_metrics.txt
  outputs/rf_full_cm.png
  outputs/rf_full_feature_importance.png
"""
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

DATA = Path("data/processed/dataset_baseline.parquet")
OUT  = Path("outputs"); OUT.mkdir(exist_ok=True)

df = pd.read_parquet(DATA)
df["year"] = df["year"].astype(int)

# time-based split
train = df[df["year"] <= 2016]
val   = df[(df["year"] >= 2017) & (df["year"] <= 2020)]
test  = df[df["year"] >= 2021]

features = ["pr","pr_anom","anom_lag1","anom_lag3","month_sin","month_cos"]
target   = "drought_label"

X_train, y_train = train[features], train[target]
X_val,   y_val   = val[features],   val[target]
X_test,  y_test  = test[features],  test[target]

pipe = Pipeline(steps=[
    ("impute", SimpleImputer(strategy="median")),
    ("rf", RandomForestClassifier(
        n_estimators=1000,        # more trees now that we have compute
        max_depth=24,             # moderate to avoid overfitting
        min_samples_leaf=4,
        n_jobs=-1,                # all cores
        class_weight="balanced_subsample",
        random_state=42,
    )),
])

print("Fitting RandomForest on full training set:", X_train.shape)
pipe.fit(X_train, y_train)

# Evaluate on test
y_pred = pipe.predict(X_test)
report = classification_report(
    y_test, y_pred,
    target_names=["dry(-1)","normal(0)","wet(1)"], digits=3
)
print(report)

# Confusion matrix (normalized by true class)
cm = confusion_matrix(y_test, y_pred, labels=[-1,0,1], normalize="true")
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["dry","normal","wet"])
disp.plot(cmap="Blues", values_format=".2f")
plt.title("Random Forest (full data) — Normalized Confusion Matrix")
cm_path = OUT / "rf_full_cm.png"
plt.savefig(cm_path, dpi=150, bbox_inches="tight"); plt.close()

# Feature importance
rf = pipe.named_steps["rf"]
imp = pd.Series(rf.feature_importances_, index=features).sort_values(ascending=False)
ax = imp.plot(kind="barh", figsize=(6,4))
ax.invert_yaxis()
ax.set_xlabel("Feature importance")
ax.set_title("Random Forest (full data) — Feature importance")
fi_path = OUT / "rf_full_feature_importance.png"
plt.tight_layout(); plt.savefig(fi_path, dpi=150, bbox_inches="tight"); plt.close()

# Save metrics
metrics_path = OUT / "rf_full_metrics.txt"
with open(metrics_path, "w") as f:
    f.write(report)

print("Wrote:", metrics_path)
print("Wrote:", cm_path)
print("Wrote:", fi_path)

