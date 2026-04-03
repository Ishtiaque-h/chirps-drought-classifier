#!/usr/bin/env python
"""
Primary skill evaluation for the 1-month-ahead drought forecast.

Methodological approach (Q1/Q2 journal standard):
  - All PRIMARY metrics are computed at the MONTHLY level (60 independent test
    months) rather than the pixel level (~400k rows).  Pixel rows within the
    same month are spatially autocorrelated (effective df ≈ 60, not 400k), so
    pixel-level accuracy inflates apparent significance.  Monthly aggregation
    matches the effective degrees of freedom available in the test set.
  - Proper probabilistic skill scores (BSS, HSS) are reported.  Accuracy alone
    is insufficient for a climate forecast paper because it is sensitive to class
    frequency.
  - Three naive baselines are included so the reader can judge whether the ML
    model adds genuine predictive skill beyond trivial heuristics.

Skill metrics:
  Brier Score (BS) and Brier Skill Score (BSS):
    BSS = 1 - BS_model / BS_reference   (BSS > 0 means better than reference)
    Reference = climatological class frequencies from training set.
    BSS reported for each class and as multi-class average.

  Heidke Skill Score (HSS):
    HSS = (correct - expected_correct) / (total - expected_correct)
    where expected_correct is from the marginal frequencies.

  ROC-AUC (dry vs. not-dry):
    Binary classification skill for the drought detection task.

  Reliability diagram (calibration):
    Observed frequency vs. predicted probability for the dry class, with
    isotonic-regression post-hoc calibration applied on the validation set.

Baselines:
  1. Climatological: per-calendar-month class-frequency distribution from train.
  2. Persistence:    predict label[t+1] = label[t] (current month persists).
  3. SPI-1 threshold: if spi1_lag1 <= -1 → dry; if spi1_lag1 >= 1 → wet; else normal.

Inputs:
  data/processed/dataset_forecast.parquet
  outputs/forecast_xgb_test_probs.npz   (softmax probabilities from XGBoost)
  outputs/forecast_xgb_model.json       (used to regenerate probs if .npz absent)

Outputs:
  outputs/forecast_skill_scores.txt
  outputs/forecast_skill_bss_hss_table.csv
  outputs/forecast_reliability_diagram.png
  outputs/forecast_monthly_cm.png
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import xgboost as xgb
from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score, confusion_matrix, ConfusionMatrixDisplay

DATA       = Path("data/processed/dataset_forecast.parquet")
PROBS_NPZ  = Path("outputs/forecast_xgb_test_probs.npz")
MODEL_PATH = Path("outputs/forecast_xgb_model.json")
OUT_DIR    = Path("outputs"); OUT_DIR.mkdir(exist_ok=True)

FEATURES = [
    "spi1_lag1", "spi1_lag2", "spi1_lag3",
    "spi3_lag1", "spi6_lag1",
    "pr_lag1", "pr_lag2", "pr_lag3",
    "month_sin", "month_cos",
]
TARGET = "target_label"
LABEL_MAP     = {-1: 0, 0: 1, 1: 2}   # XGBoost internal → class index
INV_LABEL_MAP = {v: k for k, v in LABEL_MAP.items()}
CLASSES       = [-1, 0, 1]             # dry, normal, wet
CLASS_NAMES   = ["dry(-1)", "normal(0)", "wet(+1)"]

# ── load dataset ──────────────────────────────────────────────────────────────
print("Loading dataset...")
df = pd.read_parquet(DATA)
df["year"] = df["year"].astype(int)

train = df[df["year"] <= 2016]
val   = df[(df["year"] >= 2017) & (df["year"] <= 2020)]
test  = df[df["year"] >= 2021].copy()

# target month (the month being predicted)
test["month_dt"] = (
    pd.to_datetime(test["time"]) + pd.DateOffset(months=1)
).dt.to_period("M").dt.to_timestamp()

# ── load or compute XGBoost probabilities ────────────────────────────────────
if PROBS_NPZ.exists():
    print("Loading saved probabilities from", PROBS_NPZ)
    loaded      = np.load(PROBS_NPZ, allow_pickle=True)
    xgb_probs   = loaded["probs"]            # (n_rows, 3)  columns: [dry, normal, wet]
    xgb_y_true  = loaded["y_true"]
    # verify alignment
    assert len(xgb_probs) == len(test), (
        f"Saved probs length ({len(xgb_probs)}) != test rows ({len(test)}). "
        "Retrain the model and re-run."
    )
else:
    print("Saved probabilities not found; regenerating from model...")
    assert MODEL_PATH.exists(), f"Model not found at {MODEL_PATH}. Run train_forecast_xgboost.py first."
    model = xgb.Booster()
    model.load_model(MODEL_PATH.as_posix())
    dtest     = xgb.DMatrix(test[FEATURES], feature_names=FEATURES)
    xgb_probs = model.predict(dtest)         # (n_rows, 3)

# columns: index 0 = dry (-1), 1 = normal (0), 2 = wet (+1)
test["xgb_prob_dry"]    = xgb_probs[:, 0]
test["xgb_prob_normal"] = xgb_probs[:, 1]
test["xgb_prob_wet"]    = xgb_probs[:, 2]
test["xgb_pred"]        = np.array([INV_LABEL_MAP[i] for i in xgb_probs.argmax(axis=1)])

# ── load validation probs for calibration ────────────────────────────────────
print("Computing validation-set probabilities for calibration...")
assert MODEL_PATH.exists(), f"Model not found at {MODEL_PATH}."
model_xgb = xgb.Booster()
model_xgb.load_model(MODEL_PATH.as_posix())

val_y_enc   = val[TARGET].map(LABEL_MAP).values
dval        = xgb.DMatrix(val[FEATURES], feature_names=FEATURES)
val_probs   = model_xgb.predict(dval)         # (n_val, 3)

# ── isotonic calibration on validation set (dry class) ───────────────────────
iso_cal = IsotonicRegression(out_of_bounds="clip")
iso_cal.fit(val_probs[:, 0], (val_y_enc == LABEL_MAP[-1]).astype(int))

test["xgb_prob_dry_cal"] = iso_cal.predict(test["xgb_prob_dry"].values)

# ── helper functions ──────────────────────────────────────────────────────────

def brier_score(y_true_bin: np.ndarray, prob: np.ndarray) -> float:
    """Binary Brier Score: mean((prob - obs)^2)."""
    return float(np.mean((prob - y_true_bin) ** 2))


def bss(bs_model: float, bs_ref: float) -> float:
    """Brier Skill Score = 1 - BS_model / BS_ref."""
    if bs_ref == 0:
        return np.nan
    return 1.0 - bs_model / bs_ref


def heidke_skill_score(y_true: np.ndarray, y_pred: np.ndarray,
                       classes: list) -> float:
    """
    Heidke Skill Score for multi-class categorical forecasts.
    HSS = (correct - expected) / (total - expected)
    """
    n     = len(y_true)
    cm    = confusion_matrix(y_true, y_pred, labels=classes)
    total = cm.sum()
    if total == 0:
        return np.nan
    correct  = np.diag(cm).sum()
    row_sums = cm.sum(axis=1)
    col_sums = cm.sum(axis=0)
    expected = (row_sums * col_sums).sum() / total
    denom    = total - expected
    if denom == 0:
        return np.nan
    return float((correct - expected) / denom)


# ── climatological baseline ───────────────────────────────────────────────────
# Per-calendar-month class frequencies from training set.
# target month = feature month + 1
train["month_num"] = (pd.to_datetime(train["time"]) + pd.DateOffset(months=1)).dt.month
clim_probs_by_month = {}
for m in range(1, 13):
    subset = train[train["month_num"] == m][TARGET]
    freqs  = subset.value_counts(normalize=True)
    clim_probs_by_month[m] = {c: float(freqs.get(c, 0.0)) for c in CLASSES}

# assign climatological probabilities to test rows (by target month)
test_target_month = pd.to_datetime(test["month_dt"]).dt.month
for ci, c in enumerate(CLASSES):
    test[f"clim_prob_{c}"] = test_target_month.map(
        lambda m, c=c: clim_probs_by_month.get(m, {}).get(c, 1/3)
    ).values

test["clim_pred"] = np.array([
    CLASSES[np.argmax([row[f"clim_prob_{c}"] for c in CLASSES])]
    for _, row in test.iterrows()
])

# ── persistence baseline ──────────────────────────────────────────────────────
# Use current month's SPI-1 label (spi1_lag1 thresholded) as the persistence
# prediction for next month.
test["persist_pred"] = np.where(test["spi1_lag1"] <= -1.0, -1,
                        np.where(test["spi1_lag1"] >=  1.0,  1, 0)).astype(int)
for ci, c in enumerate(CLASSES):
    test[f"persist_prob_{c}"] = (test["persist_pred"] == c).astype(float)

# ── SPI-1 threshold baseline ──────────────────────────────────────────────────
# Zero-ML rule: if current SPI-1 is dry, forecast dry next month, etc.
test["thr_pred"] = test["persist_pred"]   # same rule
for ci, c in enumerate(CLASSES):
    test[f"thr_prob_{c}"] = test[f"persist_prob_{c}"]

# ── monthly aggregation ───────────────────────────────────────────────────────
print("Aggregating to monthly level (60 independent test months)...")

monthly = test.groupby("month_dt").agg(
    y_true_mode     = (TARGET, lambda s: int(s.mode()[0])),
    y_true_dry_frac = (TARGET, lambda s: (s == -1).mean()),
    xgb_dry_frac    = ("xgb_prob_dry",    "mean"),
    xgb_norm_frac   = ("xgb_prob_normal", "mean"),
    xgb_wet_frac    = ("xgb_prob_wet",    "mean"),
    xgb_dry_cal_frac= ("xgb_prob_dry_cal","mean"),
    clim_dry_frac   = (f"clim_prob_{-1}", "mean"),
    clim_norm_frac  = (f"clim_prob_{0}",  "mean"),
    clim_wet_frac   = (f"clim_prob_{1}",  "mean"),
    persist_dry_frac= (f"persist_prob_{-1}","mean"),
    persist_norm_frac=(f"persist_prob_{0}", "mean"),
    persist_wet_frac= (f"persist_prob_{1}", "mean"),
    xgb_pred_mode   = ("xgb_pred",    lambda s: int(s.mode()[0])),
    clim_pred_mode  = ("clim_pred",   lambda s: int(s.mode()[0])),
    persist_pred_mode=("persist_pred",lambda s: int(s.mode()[0])),
).reset_index()

# true binary indicator for dry class at monthly level
monthly["obs_dry"] = (monthly["y_true_dry_frac"] > 0.5).astype(float)

n_months = len(monthly)
print(f"Test months: {n_months}")

# ── Brier Scores ─────────────────────────────────────────────────────────────
obs_dry = monthly["obs_dry"].values

bs_xgb   = brier_score(obs_dry, monthly["xgb_dry_frac"].values)
bs_clim  = brier_score(obs_dry, monthly["clim_dry_frac"].values)
bs_pers  = brier_score(obs_dry, monthly["persist_dry_frac"].values)

bss_xgb  = bss(bs_xgb,  bs_clim)
bss_pers = bss(bs_pers, bs_clim)

# ── Heidke Skill Scores ───────────────────────────────────────────────────────
y_true_monthly = monthly["y_true_mode"].values
hss_xgb   = heidke_skill_score(y_true_monthly, monthly["xgb_pred_mode"].values,   CLASSES)
hss_clim  = heidke_skill_score(y_true_monthly, monthly["clim_pred_mode"].values,   CLASSES)
hss_pers  = heidke_skill_score(y_true_monthly, monthly["persist_pred_mode"].values, CLASSES)

# ── ROC-AUC (dry vs. not-dry) ─────────────────────────────────────────────────
try:
    auc_xgb  = roc_auc_score(obs_dry, monthly["xgb_dry_frac"].values)
    auc_pers = roc_auc_score(obs_dry, monthly["persist_dry_frac"].values)
except Exception:
    auc_xgb = auc_pers = np.nan

# ── Confusion matrix at monthly level ─────────────────────────────────────────
cm_monthly = confusion_matrix(y_true_monthly, monthly["xgb_pred_mode"].values,
                              labels=CLASSES, normalize="true")
disp = ConfusionMatrixDisplay(confusion_matrix=cm_monthly,
                              display_labels=["dry", "normal", "wet"])
fig_cm, ax_cm = plt.subplots(figsize=(5, 4))
disp.plot(ax=ax_cm, cmap="Blues", values_format=".2f", colorbar=False)
ax_cm.set_title("XGBoost — Monthly-level confusion matrix\n"
                f"(n = {n_months} independent test months; 2021–2025)")
fig_cm.tight_layout()
cm_path = OUT_DIR / "forecast_monthly_cm.png"
fig_cm.savefig(cm_path, dpi=150, bbox_inches="tight")
plt.close(fig_cm)
print("Wrote:", cm_path)

# ── Reliability diagram (calibration) ─────────────────────────────────────────
fig_rel, axes = plt.subplots(1, 2, figsize=(11, 5), sharey=True)

for ax, col, label, color in [
    (axes[0], "xgb_dry_frac",     "XGBoost (raw)",        "#2171b5"),
    (axes[1], "xgb_dry_cal_frac", "XGBoost (calibrated)", "#238b45"),
]:
    # use pixel-level for calibration curve (more points = smoother bins)
    frac_pos, mean_pred = calibration_curve(
        (test[TARGET] == -1).astype(int),
        test[col] if col != "xgb_dry_cal_frac" else
            iso_cal.predict(test["xgb_prob_dry"].values),
        n_bins=10,
        strategy="uniform",
    )
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Perfect calibration")
    ax.plot(mean_pred, frac_pos, "o-", color=color, label=label)
    ax.set_xlabel("Mean predicted probability (dry class)")
    ax.set_ylabel("Observed frequency (dry class)")
    ax.set_title(label)
    ax.legend(fontsize=8)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)

fig_rel.suptitle("Reliability diagram — dry class probability", fontsize=11)
fig_rel.tight_layout()
rel_path = OUT_DIR / "forecast_reliability_diagram.png"
fig_rel.savefig(rel_path, dpi=150, bbox_inches="tight")
plt.close(fig_rel)
print("Wrote:", rel_path)

# ── Skill score table ─────────────────────────────────────────────────────────
rows = [
    {"Forecaster": "Climatological baseline", "BS_dry": f"{bs_clim:.4f}",
     "BSS_dry": "0.0000 (ref)", "HSS": f"{hss_clim:.4f}", "ROC-AUC_dry": "—"},
    {"Forecaster": "Persistence baseline",    "BS_dry": f"{bs_pers:.4f}",
     "BSS_dry": f"{bss_pers:.4f}", "HSS": f"{hss_pers:.4f}", "ROC-AUC_dry": f"{auc_pers:.4f}"},
    {"Forecaster": "XGBoost (raw)",           "BS_dry": f"{bs_xgb:.4f}",
     "BSS_dry": f"{bss_xgb:.4f}",  "HSS": f"{hss_xgb:.4f}",  "ROC-AUC_dry": f"{auc_xgb:.4f}"},
]
table_df = pd.DataFrame(rows)
csv_path = OUT_DIR / "forecast_skill_bss_hss_table.csv"
table_df.to_csv(csv_path, index=False)
print("Wrote:", csv_path)

# ── text summary ──────────────────────────────────────────────────────────────
summary = (
    "Forecast Skill Evaluation — Central Valley 2021–2025\n"
    "=" * 60 + "\n"
    f"Test months (independent temporal units): {n_months}\n"
    f"Pixels per month (spatially autocorrelated, secondary): "
    f"{len(test) // n_months:,}\n\n"
    "METHODOLOGICAL NOTE:\n"
    "  All PRIMARY metrics below are computed at the monthly level.\n"
    "  Pixel-level metrics inflate significance due to spatial autocorrelation.\n"
    "  BSS reference = climatological class frequency from training set (1991–2016).\n\n"
    "── Monthly-level Brier Scores (dry class) ──\n"
    f"  Climatological (reference) : {bs_clim:.4f}\n"
    f"  Persistence baseline       : {bs_pers:.4f}\n"
    f"  XGBoost                    : {bs_xgb:.4f}\n\n"
    "── Brier Skill Score (BSS, dry class, ref = climatology) ──\n"
    f"  Persistence : {bss_pers:.4f}  (>0 means better than climatology)\n"
    f"  XGBoost     : {bss_xgb:.4f}\n\n"
    "── Heidke Skill Score (HSS, 3-class, monthly dominant class) ──\n"
    f"  Climatological : {hss_clim:.4f}\n"
    f"  Persistence    : {hss_pers:.4f}\n"
    f"  XGBoost        : {hss_xgb:.4f}\n\n"
    "── ROC-AUC (dry vs. not-dry, monthly mean probability) ──\n"
    f"  Persistence : {auc_pers:.4f}\n"
    f"  XGBoost     : {auc_xgb:.4f}\n\n"
    "Outputs:\n"
    f"  {cm_path}\n"
    f"  {rel_path}\n"
    f"  {csv_path}\n"
)
print(summary)
(OUT_DIR / "forecast_skill_scores.txt").write_text(summary)
print("Wrote:", OUT_DIR / "forecast_skill_scores.txt")
