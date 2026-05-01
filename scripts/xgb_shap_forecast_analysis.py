#!/usr/bin/env python
"""
SHAP analysis for the XGBoost drought forecast model.

Uses TreeExplainer (exact, fast) instead of KernelExplainer.
Analyses all 3 classes (dry, normal, wet) plus detailed dry-class plots.

Inputs:
  data/processed/dataset_forecast.parquet
  outputs/forecast_xgb_model.json

Outputs:
  outputs/xgb_shap_summary_bar_forecast.png       — global bar (mean |SHAP|)
  outputs/xgb_shap_beeswarm_dry_forecast.png      — beeswarm for dry class
  outputs/xgb_shap_dependence_spi3_dry.png        — SPI-3 lag vs. dry SHAP
  outputs/xgb_shap_dependence_spi1_dry.png        — SPI-1 lag vs. dry SHAP
"""
from pathlib import Path
import numpy as np
import pandas as pd
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
from feature_config import get_feature_columns

DATA       = Path("data/processed/dataset_forecast.parquet")
MODEL_PATH = Path("outputs/forecast_xgb_model.json")
OUT_DIR    = Path("outputs"); OUT_DIR.mkdir(exist_ok=True)

TARGET = "target_label"

label_map = {-1: 0, 0: 1, 1: 2}
DRY_IDX   = label_map[-1]   # 0

# ---------- load ----------
print("Loading dataset...")
df = pd.read_parquet(DATA)
df["year"] = df["year"].astype(int)
FEATURES = get_feature_columns(df.columns)
test = df[df["year"] >= 2021]

print("Loading model from:", MODEL_PATH)
model = xgb.Booster()
model.load_model(MODEL_PATH.as_posix())

# ---------- stratified sample ----------
N_PER_CLASS = 700
rng = np.random.default_rng(42)
parts = []
for cls, grp in test.groupby(TARGET):
    n   = min(len(grp), N_PER_CLASS)
    idx = rng.choice(len(grp), size=n, replace=False)
    parts.append(grp.iloc[idx])

sample   = pd.concat(parts, ignore_index=True)
X_sample = sample[FEATURES].reset_index(drop=True)
print(f"SHAP sample: {X_sample.shape}  classes: {sample[TARGET].value_counts().sort_index().to_dict()}")

# ---------- TreeExplainer (exact, no approximation needed) ----------
print("Creating TreeExplainer...")
explainer = shap.TreeExplainer(model)

print("Computing SHAP values...")
# shap_values shape: (n_samples, n_features, n_classes)
shap_values = explainer.shap_values(X_sample)
print("shap_values shape:", np.array(shap_values).shape)

# shap_values is a list of 3 arrays (one per class) or a 3-D array
# normalise to list-of-arrays for compatibility
if isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
    sv_list = [shap_values[:, :, c] for c in range(shap_values.shape[2])]
else:
    sv_list = list(shap_values)

sv_dry = sv_list[DRY_IDX]   # (n_samples, n_features) for dry class

# ---------- 1. Global bar chart (mean |SHAP| across all classes) ----------
print("Saving global SHAP bar plot...")
mean_abs = np.mean([np.abs(sv) for sv in sv_list], axis=0)  # (n_samples, n_features)
global_imp = pd.Series(mean_abs.mean(axis=0), index=FEATURES).sort_values(ascending=True)

fig, ax = plt.subplots(figsize=(8, 5))
ax.barh(global_imp.index, global_imp.values)
ax.set_xlabel("Mean |SHAP value| (averaged over all classes)")
ax.set_title("Forecast XGBoost — Global SHAP importance")
plt.tight_layout()
bar_path = OUT_DIR / "xgb_shap_summary_bar_forecast.png"
plt.savefig(bar_path, dpi=150, bbox_inches="tight")
plt.close()

# ---------- 2. Beeswarm for dry class ----------
print("Saving SHAP beeswarm (dry class)...")
plt.figure()
shap.summary_plot(sv_dry, X_sample, feature_names=FEATURES, show=False)
plt.title("XGBoost SHAP — Dry class contributions")
beeswarm_path = OUT_DIR / "xgb_shap_beeswarm_dry_forecast.png"
plt.tight_layout()
plt.savefig(beeswarm_path, dpi=150, bbox_inches="tight")
plt.close()

# ---------- 3. Dependence plot: spi3_lag1 vs. dry SHAP ----------
print("Saving SHAP dependence plot — spi3_lag1...")
plt.figure()
shap.dependence_plot(
    "spi3_lag1", sv_dry, X_sample,
    feature_names=FEATURES, interaction_index=None, show=False,
)
plt.title("SHAP dependence — SPI-3 lag (effect on dry probability)")
dep_spi3_path = OUT_DIR / "xgb_shap_dependence_spi3_dry.png"
plt.tight_layout()
plt.savefig(dep_spi3_path, dpi=150, bbox_inches="tight")
plt.close()

# ---------- 4. Dependence plot: spi1_lag1 vs. dry SHAP ----------
print("Saving SHAP dependence plot — spi1_lag1...")
plt.figure()
shap.dependence_plot(
    "spi1_lag1", sv_dry, X_sample,
    feature_names=FEATURES, interaction_index=None, show=False,
)
plt.title("SHAP dependence — SPI-1 lag (effect on dry probability)")
dep_spi1_path = OUT_DIR / "xgb_shap_dependence_spi1_dry.png"
plt.tight_layout()
plt.savefig(dep_spi1_path, dpi=150, bbox_inches="tight")
plt.close()

print("Wrote:", bar_path)
print("Wrote:", beeswarm_path)
print("Wrote:", dep_spi3_path)
print("Wrote:", dep_spi1_path)
print("Done.")
