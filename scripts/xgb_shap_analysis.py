#!/usr/bin/env python
"""
SHAP analysis for the XGBoost drought classifier, focused on
the probability of the "dry" class (drought_label = -1).

Uses:
  - data/processed/dataset_baseline.parquet
  - outputs/xgb_baseline_model.json

Outputs:
  - outputs/xgb_shap_summary_bar_dry.png
  - outputs/xgb_shap_summary_beeswarm_dry.png
  - outputs/xgb_shap_dependence_pr_anom_dry.png
"""

from pathlib import Path
import numpy as np
import pandas as pd
import xgboost as xgb
import shap
import matplotlib.pyplot as plt

# ---------------- paths ----------------
DATA = Path("data/processed/dataset_baseline.parquet")
MODEL_PATH = Path("outputs/xgb_baseline_model.json")
OUT_DIR = Path("outputs")
OUT_DIR.mkdir(exist_ok=True)

# ---------------- load data ----------------
print("Loading dataset...")
df = pd.read_parquet(DATA)
df["year"] = df["year"].astype(int)

test = df[df["year"] >= 2021]

features = ["pr", "pr_anom", "anom_lag1", "anom_lag3", "month_sin", "month_cos"]
target = "drought_label"

X_test = test[features]
y_test = test[target]

print("Test shape:", X_test.shape)

label_map = {-1: 0, 0: 1, 1: 2}   # dry -> index 0
DRY_IDX = label_map[-1]

# ---------------- load trained XGBoost model ----------------
print("Loading XGBoost model from:", MODEL_PATH)
model = xgb.Booster()
model.load_model(MODEL_PATH.as_posix())

# ---------------- prediction function for SHAP ----------------
def predict_dry_proba(data_np: np.ndarray) -> np.ndarray:
    """
    Return probability of the 'dry' class for each sample.
    Shape: (n_samples,)
    """
    d = xgb.DMatrix(data_np, feature_names=features)
    probs = model.predict(d)  # (n_samples, 3)
    dry_probs = probs[:, DRY_IDX]  # (n_samples,)
    return dry_probs

# ---------------- sample data for SHAP ----------------
N_PER_CLASS = 700   # 700 per class => ~2100 rows total
rng = np.random.default_rng(42)

parts = []
for cls, grp in test.groupby(target):
    n = min(len(grp), N_PER_CLASS)
    idx = rng.choice(len(grp), size=n, replace=False)
    parts.append(grp.iloc[idx])

sample = pd.concat(parts, ignore_index=True)
X_sample = sample[features].reset_index(drop=True)

print("SHAP sample shape:", X_sample.shape)
print("Class counts in SHAP sample:", sample[target].value_counts().sort_index().to_dict())

X_sample_np = X_sample.values

BACKGROUND_SIZE = 300
background = X_sample_np[:BACKGROUND_SIZE, :]

# ---------------- KernelExplainer ----------------
print("Creating KernelExplainer for dry-class probability...")
explainer = shap.KernelExplainer(predict_dry_proba, background)

print("Computing SHAP values for sample (this can take some minutes)...")
# shap_values has shape (n_samples, n_features) because output is scalar
shap_values = explainer.shap_values(X_sample_np)

print("shap_values shape:", np.array(shap_values).shape)

# ---------------- Global summary plots ----------------
print("Saving SHAP summary (bar) plot for dry class...")
plt.figure()
shap.summary_plot(
    shap_values,
    X_sample,
    feature_names=features,
    plot_type="bar",
    show=False,
)
plt.title("XGBoost SHAP — Global importance for dry probability")
bar_path = OUT_DIR / "xgb_shap_summary_bar_dry.png"
plt.tight_layout()
plt.savefig(bar_path, dpi=150, bbox_inches="tight")
plt.close()

print("Saving SHAP summary (beeswarm) plot for dry class...")
plt.figure()
shap.summary_plot(
    shap_values,
    X_sample,
    feature_names=features,
    show=False,
)
beeswarm_path = OUT_DIR / "xgb_shap_summary_beeswarm_dry.png"
plt.tight_layout()
plt.savefig(beeswarm_path, dpi=150, bbox_inches="tight")
plt.close()

# ---------------- Dependence plot for pr_anom ----------------
print("Saving SHAP dependence plot for 'pr_anom' (dry probability)...")
plt.figure()
shap.dependence_plot(
    "pr_anom",
    shap_values,
    X_sample,
    feature_names=features,
    interaction_index=None,
    show=False,
)
plt.title("SHAP dependence — pr_anom (effect on dry probability)")
dep_path = OUT_DIR / "xgb_shap_dependence_pr_anom_dry.png"
plt.tight_layout()
plt.savefig(dep_path, dpi=150, bbox_inches="tight")
plt.close()

print("Wrote:", bar_path)
print("Wrote:", beeswarm_path)
print("Wrote:", dep_path)
print("Done.")
