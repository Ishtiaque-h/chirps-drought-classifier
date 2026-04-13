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
  2. Persistence:    predict label[t+1] = label[t] (current month drought class persists).
  3. SPI-1 threshold heuristic: map current SPI-1 continuously to class probabilities
     (dry/wet increase with |SPI-1|; normal dominates near SPI-1≈0).

Inputs:
  data/processed/dataset_forecast.parquet
  outputs/forecast_xgb_test_probs.npz   (softmax probabilities from XGBoost)
  outputs/forecast_xgb_model.json       (used to regenerate probs if .npz absent)
  outputs/forecast_logreg_model.pkl     (optional; adds LogReg row to skill table)
  outputs/forecast_rf_model.pkl         (optional; adds RF row to skill table)
  outputs/xgb_spatial_test_probs.npz    (optional; adds XGBoost-Spatial row)
  outputs/convlstm_test_probs.npz       (optional; adds ConvLSTM row)
  data/processed/convlstm_meta.npz      (lat/lon/test_feature_times for ConvLSTM)

Outputs:
  outputs/forecast_skill_scores.txt
  outputs/forecast_skill_bss_hss_table.csv
  outputs/forecast_reliability_diagram.png
  outputs/forecast_monthly_cm.png
  outputs/calib_study_results.csv         (calibration study, new)
  outputs/calib_study_reliability_diagram.png  (calibration study, new)
  outputs/calib_study_decomposition_barplot.png (calibration study, new)
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import xgboost as xgb
import joblib
from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression as _LogisticReg  # Platt scaling
from sklearn.metrics import roc_auc_score, confusion_matrix, ConfusionMatrixDisplay

DATA          = Path("data/processed/dataset_forecast.parquet")
PROBS_NPZ     = Path("outputs/forecast_xgb_test_probs.npz")
MODEL_PATH    = Path("outputs/forecast_xgb_model.json")
LOGREG_MODEL  = Path("outputs/forecast_logreg_model.pkl")
RF_MODEL      = Path("outputs/forecast_rf_model.pkl")
XGB_SPATIAL_NPZ  = Path("outputs/xgb_spatial_test_probs.npz")
CONVLSTM_NPZ     = Path("outputs/convlstm_test_probs.npz")
CONVLSTM_META    = Path("data/processed/convlstm_meta.npz")
OUT_DIR       = Path("outputs"); OUT_DIR.mkdir(exist_ok=True)

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
SPI_HEURISTIC_SCALE = 2.0      # Linear scale: SPI=±1 -> 0.5; |SPI|>2 clips to probability 1.0
N_BOOTSTRAP_ITERATIONS = 2000  # Standard bootstrap count for stable percentile CIs

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

# ── Logistic Regression probabilities (if model available) ───────────────────
if LOGREG_MODEL.exists():
    print("Loading Logistic Regression model for skill comparison...")
    logreg_pipe = joblib.load(LOGREG_MODEL)
    lr_classes  = list(logreg_pipe.classes_)           # e.g. [-1, 0, 1]
    lr_probs    = logreg_pipe.predict_proba(test[FEATURES])  # (n, 3)
    for ci, c in enumerate(CLASSES):
        col_idx = lr_classes.index(c)
        test[f"lr_prob_{c}"] = lr_probs[:, col_idx]
    test["lr_pred"] = logreg_pipe.predict(test[FEATURES])
    HAS_LOGREG = True
else:
    print("WARNING: Logistic Regression model not found at", LOGREG_MODEL)
    HAS_LOGREG = False

# ── Random Forest probabilities (if model available) ─────────────────────────
if RF_MODEL.exists():
    print("Loading Random Forest model for skill comparison...")
    rf_pipe    = joblib.load(RF_MODEL)
    rf_classes = list(rf_pipe.classes_)
    rf_probs   = rf_pipe.predict_proba(test[FEATURES])
    for ci, c in enumerate(CLASSES):
        col_idx = rf_classes.index(c)
        test[f"rf_prob_{c}"] = rf_probs[:, col_idx]
    test["rf_pred"] = rf_pipe.predict(test[FEATURES])
    HAS_RF = True
else:
    print("WARNING: Random Forest model not found at", RF_MODEL)
    HAS_RF = False

# ── XGBoost-Spatial probabilities (if available) ─────────────────────────────
if XGB_SPATIAL_NPZ.exists():
    print("Loading XGBoost-Spatial probabilities from", XGB_SPATIAL_NPZ)
    sp_loaded    = np.load(XGB_SPATIAL_NPZ, allow_pickle=True)
    sp_probs     = sp_loaded["proba"]          # (n_rows, 3)  columns: [dry, normal, wet]
    assert len(sp_probs) == len(test), (
        f"XGB-Spatial probs length ({len(sp_probs)}) != test rows ({len(test)}). "
        "Retrain the spatial model and re-run."
    )
    test["xgb_spatial_prob_dry"]    = sp_probs[:, 0]
    test["xgb_spatial_prob_normal"] = sp_probs[:, 1]
    test["xgb_spatial_prob_wet"]    = sp_probs[:, 2]
    test["xgb_spatial_pred"]        = np.array(
        [INV_LABEL_MAP[i] for i in sp_probs.argmax(axis=1)]
    )
    HAS_XGB_SPATIAL = True
else:
    print("WARNING: XGBoost-Spatial probs not found at", XGB_SPATIAL_NPZ)
    HAS_XGB_SPATIAL = False

# ── ConvLSTM probabilities (if available) ────────────────────────────────────
HAS_CONVLSTM = False
if CONVLSTM_NPZ.exists() and CONVLSTM_META.exists():
    cl_meta = np.load(CONVLSTM_META, allow_pickle=True)
    if "test_feature_times" not in cl_meta:
        print("WARNING: convlstm_meta.npz is missing 'test_feature_times'. "
              "Re-run build_dataset_convlstm.py to regenerate meta.")
    else:
        print("Loading ConvLSTM probabilities from", CONVLSTM_NPZ)
        cl_loaded  = np.load(CONVLSTM_NPZ, allow_pickle=True)
        cl_proba   = cl_loaded["proba"]                       # (N_test, 3, lat, lon)
        cl_lat     = cl_meta["lat"]                           # (nlat,)
        cl_lon     = cl_meta["lon"]                           # (nlon,)
        cl_times   = pd.to_datetime(cl_meta["test_feature_times"])  # (N_test,)

        N_cl, _, nlat, nlon = cl_proba.shape
        # Compute per-pixel argmax prediction: (N_test, nlat, nlon) encoded {0,1,2}
        cl_pred_enc = cl_proba.argmax(axis=1)
        _inv_enc    = {0: -1, 1: 0, 2: 1}

        def nearest_grid_indices(grid_vals: np.ndarray, query_vals: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            """Return nearest original-grid indices and absolute diffs for each query."""
            sort_idx = np.argsort(grid_vals)
            grid_sorted = grid_vals[sort_idx]
            pos = np.searchsorted(grid_sorted, query_vals)
            pos = np.clip(pos, 1, len(grid_sorted) - 1)
            left = grid_sorted[pos - 1]
            right = grid_sorted[pos]
            use_right = np.abs(query_vals - right) < np.abs(query_vals - left)
            nearest_pos = np.where(use_right, pos, pos - 1)
            nearest_orig_idx = sort_idx[nearest_pos]
            nearest_diff = np.abs(query_vals - grid_vals[nearest_orig_idx])
            return nearest_orig_idx, nearest_diff

        def coord_tolerance(grid_vals: np.ndarray) -> float:
            uniq = np.unique(np.sort(grid_vals))
            if len(uniq) <= 1:
                return 1e-6
            steps = np.diff(uniq)
            steps = steps[steps > 0]
            if len(steps) == 0:
                return 1e-6
            return float(steps.min() / 2.0 + 1e-9)

        # Align by nearest time + nearest lat/lon gridpoint (with tolerance)
        test["time"] = pd.to_datetime(test["time"])
        cl_time_index = pd.DatetimeIndex(cl_times)
        test_time_index = pd.DatetimeIndex(test["time"])
        t_idx_exact = cl_time_index.get_indexer(test_time_index)
        t_idx_nearest = cl_time_index.get_indexer(
            test_time_index, method="nearest", tolerance=pd.Timedelta(days=31)
        )
        n_time_fallback = int(((t_idx_exact == -1) & (t_idx_nearest != -1)).sum())
        if n_time_fallback > 0:
            print(
                f"  NOTE: {n_time_fallback} rows had no exact ConvLSTM time match; "
                "used nearest available time within 31 days."
            )
        t_idx = pd.Series(t_idx_nearest, index=test.index)

        for _col in ["convlstm_prob_dry", "convlstm_prob_normal", "convlstm_prob_wet"]:
            test[_col] = np.nan
        test["convlstm_pred"] = np.nan

        has_time = t_idx.notna().values
        if has_time.any():
            lat_q = test.loc[has_time, "latitude"].to_numpy()
            lon_q = test.loc[has_time, "longitude"].to_numpy()

            lat_idx, lat_diff = nearest_grid_indices(cl_lat, lat_q)
            lon_idx, lon_diff = nearest_grid_indices(cl_lon, lon_q)

            lat_tol = coord_tolerance(cl_lat)
            lon_tol = coord_tolerance(cl_lon)
            within_tol = (lat_diff <= lat_tol) & (lon_diff <= lon_tol)

            matched_rows = test.index[has_time][within_tol]
            if len(matched_rows) > 0:
                time_idx_arr = t_idx.loc[matched_rows].astype(int).to_numpy()
                lat_idx_arr = lat_idx[within_tol]
                lon_idx_arr = lon_idx[within_tol]

                matched_probs = cl_proba[time_idx_arr, :, lat_idx_arr, lon_idx_arr]
                matched_pred_enc = cl_pred_enc[time_idx_arr, lat_idx_arr, lon_idx_arr]

                test.loc[matched_rows, "convlstm_prob_dry"] = matched_probs[:, 0]
                test.loc[matched_rows, "convlstm_prob_normal"] = matched_probs[:, 1]
                test.loc[matched_rows, "convlstm_prob_wet"] = matched_probs[:, 2]
                test.loc[matched_rows, "convlstm_pred"] = np.vectorize(_inv_enc.get)(matched_pred_enc)

        n_missing = int(test["convlstm_prob_dry"].isna().sum())
        if n_missing > 0:
            print(f"  WARNING: {n_missing} test rows had no ConvLSTM match; "
                  "filling with uninformative prior (1/3).")
            for _col in ["convlstm_prob_dry", "convlstm_prob_normal", "convlstm_prob_wet"]:
                test[_col] = test[_col].fillna(1 / 3)
        test["convlstm_pred"] = test["convlstm_pred"].fillna(0).astype(int)
        HAS_CONVLSTM = True
else:
    if not CONVLSTM_NPZ.exists():
        print("WARNING: ConvLSTM probs not found at", CONVLSTM_NPZ)
    if not CONVLSTM_META.exists():
        print("WARNING: ConvLSTM meta not found at", CONVLSTM_META)

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
# Zero-ML heuristic baseline: convert current SPI-1 to probabilistic class
# forecast (distinct from persistence's hard class carry-over).
# - dry probability increases as SPI-1 becomes more negative
# - wet probability increases as SPI-1 becomes more positive
# - normal receives residual probability near neutral conditions
spi_now = test["spi1_lag1"].values.astype(float)
# SPI_HEURISTIC_SCALE controls linear SPI→probability mapping:
# prob ≈ |SPI| / SPI_HEURISTIC_SCALE before clipping (SPI=±1 → 0.5, SPI=±2 → 1.0).
thr_prob_dry = np.clip((-spi_now) / SPI_HEURISTIC_SCALE, 0.0, 1.0)
thr_prob_wet = np.clip((spi_now) / SPI_HEURISTIC_SCALE, 0.0, 1.0)
thr_prob_normal = np.clip(1.0 - thr_prob_dry - thr_prob_wet, 0.0, 1.0)
thr_norm = thr_prob_dry + thr_prob_normal + thr_prob_wet
thr_prob_dry /= thr_norm
thr_prob_normal /= thr_norm
thr_prob_wet /= thr_norm

test["thr_prob_-1"] = thr_prob_dry
test["thr_prob_0"]  = thr_prob_normal
test["thr_prob_1"]  = thr_prob_wet
thr_argmax = np.vstack([thr_prob_dry, thr_prob_normal, thr_prob_wet]).T.argmax(axis=1)
test["thr_pred"] = np.array([CLASSES[i] for i in thr_argmax], dtype=int)

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
    thr_dry_frac    = (f"thr_prob_{-1}", "mean"),
    thr_norm_frac   = (f"thr_prob_{0}",  "mean"),
    thr_wet_frac    = (f"thr_prob_{1}",  "mean"),
    xgb_pred_mode   = ("xgb_pred",    lambda s: int(s.mode()[0])),
    clim_pred_mode  = ("clim_pred",   lambda s: int(s.mode()[0])),
    persist_pred_mode=("persist_pred",lambda s: int(s.mode()[0])),
    thr_pred_mode   = ("thr_pred",    lambda s: int(s.mode()[0])),
    **({f"lr_dry_frac":  (f"lr_prob_{-1}", "mean"),
        f"lr_pred_mode": ("lr_pred", lambda s: int(s.mode()[0]))}
       if HAS_LOGREG else {}),
    **({f"rf_dry_frac":  (f"rf_prob_{-1}", "mean"),
        f"rf_pred_mode": ("rf_pred", lambda s: int(s.mode()[0]))}
       if HAS_RF else {}),
    **({"xgb_spatial_dry_frac":  ("xgb_spatial_prob_dry", "mean"),
        "xgb_spatial_pred_mode": ("xgb_spatial_pred", lambda s: int(s.mode()[0]))}
       if HAS_XGB_SPATIAL else {}),
    **({"convlstm_dry_frac":  ("convlstm_prob_dry", "mean"),
        "convlstm_pred_mode": ("convlstm_pred", lambda s: int(s.mode()[0]))}
       if HAS_CONVLSTM else {}),
).reset_index()

# Monthly dry-event target for probabilistic scoring:
# use observed monthly dry fraction directly (0..1) to avoid arbitrary majority threshold.
obs_dry_frac = monthly["y_true_dry_frac"].astype(float).values

# Binary monthly dry event (for ROC-AUC only): dominant class is dry.
monthly["obs_dry_bin"] = (monthly["y_true_mode"] == -1).astype(int)

n_months = len(monthly)
print(f"Test months: {n_months}")

# ── Brier Scores ─────────────────────────────────────────────────────────────
bs_xgb   = brier_score(obs_dry_frac, monthly["xgb_dry_frac"].values)
bs_clim  = brier_score(obs_dry_frac, monthly["clim_dry_frac"].values)
bs_pers  = brier_score(obs_dry_frac, monthly["persist_dry_frac"].values)
bs_thr   = brier_score(obs_dry_frac, monthly["thr_dry_frac"].values)

bss_xgb  = bss(bs_xgb,  bs_clim)
bss_pers = bss(bs_pers, bs_clim)
bss_thr  = bss(bs_thr,  bs_clim)

# ── Heidke Skill Scores ───────────────────────────────────────────────────────
y_true_monthly = monthly["y_true_mode"].values
hss_xgb   = heidke_skill_score(y_true_monthly, monthly["xgb_pred_mode"].values,   CLASSES)
hss_clim  = heidke_skill_score(y_true_monthly, monthly["clim_pred_mode"].values,   CLASSES)
hss_pers  = heidke_skill_score(y_true_monthly, monthly["persist_pred_mode"].values, CLASSES)
hss_thr   = heidke_skill_score(y_true_monthly, monthly["thr_pred_mode"].values, CLASSES)

# ── ROC-AUC (dry vs. not-dry) ─────────────────────────────────────────────────
obs_dry_bin = monthly["obs_dry_bin"].values
try:
    auc_xgb  = roc_auc_score(obs_dry_bin, monthly["xgb_dry_frac"].values)
    auc_pers = roc_auc_score(obs_dry_bin, monthly["persist_dry_frac"].values)
    auc_thr  = roc_auc_score(obs_dry_bin, monthly["thr_dry_frac"].values)
except Exception:
    auc_xgb = auc_pers = auc_thr = np.nan

# ── LogReg and RF metrics (if models were loaded) ─────────────────────────────
if HAS_LOGREG:
    bs_lr    = brier_score(obs_dry_frac, monthly["lr_dry_frac"].values)
    bss_lr   = bss(bs_lr, bs_clim)
    hss_lr   = heidke_skill_score(y_true_monthly, monthly["lr_pred_mode"].values, CLASSES)
    try:
        auc_lr = roc_auc_score(obs_dry_bin, monthly["lr_dry_frac"].values)
    except Exception:
        auc_lr = np.nan

if HAS_RF:
    bs_rf    = brier_score(obs_dry_frac, monthly["rf_dry_frac"].values)
    bss_rf   = bss(bs_rf, bs_clim)
    hss_rf   = heidke_skill_score(y_true_monthly, monthly["rf_pred_mode"].values, CLASSES)
    try:
        auc_rf = roc_auc_score(obs_dry_bin, monthly["rf_dry_frac"].values)
    except Exception:
        auc_rf = np.nan

# ── XGBoost-Spatial and ConvLSTM metrics ─────────────────────────────────────
if HAS_XGB_SPATIAL:
    bs_sp   = brier_score(obs_dry_frac, monthly["xgb_spatial_dry_frac"].values)
    bss_sp  = bss(bs_sp, bs_clim)
    hss_sp  = heidke_skill_score(
        y_true_monthly, monthly["xgb_spatial_pred_mode"].values, CLASSES
    )
    try:
        auc_sp = roc_auc_score(obs_dry_bin, monthly["xgb_spatial_dry_frac"].values)
    except Exception:
        auc_sp = np.nan

if HAS_CONVLSTM:
    bs_cl   = brier_score(obs_dry_frac, monthly["convlstm_dry_frac"].values)
    bss_cl  = bss(bs_cl, bs_clim)
    hss_cl  = heidke_skill_score(
        y_true_monthly, monthly["convlstm_pred_mode"].values, CLASSES
    )
    try:
        auc_cl = roc_auc_score(obs_dry_bin, monthly["convlstm_dry_frac"].values)
    except Exception:
        auc_cl = np.nan

# ── Bootstrap uncertainty intervals (monthly block bootstrap) ──────────────────
def _bootstrap_ci(values: np.ndarray, alpha: float = 0.05) -> tuple[float, float]:
    """Return two-sided percentile confidence interval for bootstrap samples."""
    lo = float(np.nanpercentile(values, 100 * (alpha / 2)))
    hi = float(np.nanpercentile(values, 100 * (1 - alpha / 2)))
    return lo, hi

def bootstrap_metric(metric_fn, n_months: int, n_boot: int = 2000, seed: int = 42) -> tuple[float, float]:
    """Bootstrap a monthly metric and return its percentile CI.

    metric_fn must accept a 1-D integer index array (resampled month indices)
    and return a scalar metric value computed on that resample.

    Returns
    -------
    tuple[float, float]
        (lower_bound, upper_bound) percentile confidence interval.
    """
    rng = np.random.default_rng(seed)
    vals = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        idx = rng.integers(0, n_months, size=n_months)
        vals[i] = metric_fn(idx)
    return _bootstrap_ci(vals)

def fmt_ci(ci: tuple[float, float]) -> str:
    """Format confidence interval tuple as a compact [lo, hi] string."""
    return f"[{ci[0]:.4f}, {ci[1]:.4f}]"

bss_ci_pers = bootstrap_metric(lambda i: bss(
    brier_score(obs_dry_frac[i], monthly["persist_dry_frac"].values[i]),
    brier_score(obs_dry_frac[i], monthly["clim_dry_frac"].values[i])
), n_months, n_boot=N_BOOTSTRAP_ITERATIONS, seed=101)
bss_ci_thr = bootstrap_metric(lambda i: bss(
    brier_score(obs_dry_frac[i], monthly["thr_dry_frac"].values[i]),
    brier_score(obs_dry_frac[i], monthly["clim_dry_frac"].values[i])
), n_months, n_boot=N_BOOTSTRAP_ITERATIONS, seed=102)
bss_ci_xgb = bootstrap_metric(lambda i: bss(
    brier_score(obs_dry_frac[i], monthly["xgb_dry_frac"].values[i]),
    brier_score(obs_dry_frac[i], monthly["clim_dry_frac"].values[i])
), n_months, n_boot=N_BOOTSTRAP_ITERATIONS, seed=103)
hss_ci_clim = bootstrap_metric(lambda i: heidke_skill_score(
    y_true_monthly[i], monthly["clim_pred_mode"].values[i], CLASSES
), n_months, n_boot=N_BOOTSTRAP_ITERATIONS, seed=111)
hss_ci_pers = bootstrap_metric(lambda i: heidke_skill_score(
    y_true_monthly[i], monthly["persist_pred_mode"].values[i], CLASSES
), n_months, n_boot=N_BOOTSTRAP_ITERATIONS, seed=112)
hss_ci_thr = bootstrap_metric(lambda i: heidke_skill_score(
    y_true_monthly[i], monthly["thr_pred_mode"].values[i], CLASSES
), n_months, n_boot=N_BOOTSTRAP_ITERATIONS, seed=113)
hss_ci_xgb = bootstrap_metric(lambda i: heidke_skill_score(
    y_true_monthly[i], monthly["xgb_pred_mode"].values[i], CLASSES
), n_months, n_boot=N_BOOTSTRAP_ITERATIONS, seed=114)

# Bootstrap CIs for optional models (computed only when the model was loaded)
if HAS_XGB_SPATIAL:
    bss_ci_sp = bootstrap_metric(lambda i: bss(
        brier_score(obs_dry_frac[i], monthly["xgb_spatial_dry_frac"].values[i]),
        brier_score(obs_dry_frac[i], monthly["clim_dry_frac"].values[i])
    ), n_months, n_boot=N_BOOTSTRAP_ITERATIONS, seed=104)
    hss_ci_sp = bootstrap_metric(lambda i: heidke_skill_score(
        y_true_monthly[i], monthly["xgb_spatial_pred_mode"].values[i], CLASSES
    ), n_months, n_boot=N_BOOTSTRAP_ITERATIONS, seed=115)

if HAS_LOGREG:
    bss_ci_lr = bootstrap_metric(lambda i: bss(
        brier_score(obs_dry_frac[i], monthly["lr_dry_frac"].values[i]),
        brier_score(obs_dry_frac[i], monthly["clim_dry_frac"].values[i])
    ), n_months, n_boot=N_BOOTSTRAP_ITERATIONS, seed=105)
    hss_ci_lr = bootstrap_metric(lambda i: heidke_skill_score(
        y_true_monthly[i], monthly["lr_pred_mode"].values[i], CLASSES
    ), n_months, n_boot=N_BOOTSTRAP_ITERATIONS, seed=116)

if HAS_RF:
    bss_ci_rf = bootstrap_metric(lambda i: bss(
        brier_score(obs_dry_frac[i], monthly["rf_dry_frac"].values[i]),
        brier_score(obs_dry_frac[i], monthly["clim_dry_frac"].values[i])
    ), n_months, n_boot=N_BOOTSTRAP_ITERATIONS, seed=106)
    hss_ci_rf = bootstrap_metric(lambda i: heidke_skill_score(
        y_true_monthly[i], monthly["rf_pred_mode"].values[i], CLASSES
    ), n_months, n_boot=N_BOOTSTRAP_ITERATIONS, seed=117)

if HAS_CONVLSTM:
    bss_ci_cl = bootstrap_metric(lambda i: bss(
        brier_score(obs_dry_frac[i], monthly["convlstm_dry_frac"].values[i]),
        brier_score(obs_dry_frac[i], monthly["clim_dry_frac"].values[i])
    ), n_months, n_boot=N_BOOTSTRAP_ITERATIONS, seed=107)
    hss_ci_cl = bootstrap_metric(lambda i: heidke_skill_score(
        y_true_monthly[i], monthly["convlstm_pred_mode"].values[i], CLASSES
    ), n_months, n_boot=N_BOOTSTRAP_ITERATIONS, seed=118)

# ═══════════════════════════════════════════════════════════════════════════════
# CALIBRATION STUDY — XGBoost and XGBoost-Spatial                (NEW SECTION)
# ─────────────────────────────────────────────────────────────────────────────
# Protocol (no test leakage):
#   1. Fit three calibrators (none / Platt / isotonic) on pixel-level validation
#      data (2017–2020).
#   2. Select best method per model by validation monthly-aggregated Brier Score.
#   3. Apply best calibrator to frozen test set; evaluate at monthly level.
#   4. Report BS, BSS, Murphy decomposition, bootstrap 95 % CI, and paired
#      bootstrap significance tests vs climatology and between XGB variants.
# New output files produced in this section:
#   outputs/calib_study_results.csv
#   outputs/calib_study_reliability_diagram.png
#   outputs/calib_study_decomposition_barplot.png
# ═══════════════════════════════════════════════════════════════════════════════


def bs_decomp(obs: np.ndarray, prob: np.ndarray, n_bins: int = 10) -> dict:
    """Murphy 1973 Brier Score decomposition: BS = reliability − resolution + uncertainty.

    Parameters
    ----------
    obs  : 1-D array of observed outcomes (fractions or binary, values in [0, 1]).
    prob : 1-D array of forecast probabilities (clipped to [0, 1]).
    n_bins : number of equal-width forecast bins.

    Returns dict with keys: reliability, resolution, uncertainty, bs_check.
    ``bs_check`` (= reliability − resolution + uncertainty) should closely match
    ``brier_score(obs, prob)``; any small difference is the discretization residual.
    """
    obs  = np.asarray(obs,  dtype=float)
    prob = np.clip(np.asarray(prob, dtype=float), 0.0, 1.0)
    n    = len(obs)
    o_bar = float(obs.mean())
    edges = np.linspace(0.0, 1.0 + 1e-9, n_bins + 1)
    bidx  = np.clip(np.digitize(prob, edges) - 1, 0, n_bins - 1)
    rel = res = 0.0
    for k in range(n_bins):
        mask = bidx == k
        if not mask.any():
            continue
        f_k = float(prob[mask].mean())
        o_k = float(obs[mask].mean())
        n_k = int(mask.sum())
        rel += (n_k / n) * (f_k - o_k) ** 2   # reliability: forecast vs obs within bin
        res += (n_k / n) * (o_k - o_bar) ** 2  # resolution: obs within bin vs climatology
    # Generalized uncertainty term uses observed variance: mean((obs - o_bar)^2).
    # Here obs are monthly dry-area fractions (not strictly binary).
    unc = float(np.mean((obs - o_bar) ** 2))    # uncertainty: intrinsic variability
    return dict(reliability=rel, resolution=res, uncertainty=unc, bs_check=rel - res + unc)


# ── Calibration method helpers ─────────────────────────────────────────────────

def _fit_platt(vp: np.ndarray, vo: np.ndarray) -> _LogisticReg:
    """Platt scaling: fit logistic regression on raw validation scores (pixel level)."""
    lr = _LogisticReg(C=1.0, solver="lbfgs", max_iter=1000)
    lr.fit(vp.reshape(-1, 1), vo)
    return lr


def _apply_platt(m: _LogisticReg, p: np.ndarray) -> np.ndarray:
    """Apply fitted Platt model to new raw probabilities."""
    return m.predict_proba(p.reshape(-1, 1))[:, 1]


def _fit_isotonic_cal(vp: np.ndarray, vo: np.ndarray) -> IsotonicRegression:
    """Fit isotonic regression calibration on raw validation scores (pixel level)."""
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(vp, vo)
    return iso


def _apply_isotonic_cal(m: IsotonicRegression, p: np.ndarray) -> np.ndarray:
    """Apply fitted isotonic model to new raw probabilities."""
    return m.predict(p)


# ── Paired bootstrap significance test ────────────────────────────────────────

def paired_boot_pvalue(sq_a: np.ndarray, sq_b: np.ndarray,
                       n_boot: int = 2000, seed: int = 300) -> float:
    """Two-sided paired bootstrap p-value for H0: E[BS_a] = E[BS_b].

    sq_a and sq_b are per-month squared forecast errors for two competing models.
    Uses the same bootstrap sample (same month indices) for both models at each draw.
    """
    rng      = np.random.default_rng(seed)
    obs_diff = float(sq_a.mean() - sq_b.mean())
    diffs    = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        idx      = rng.integers(0, len(sq_a), len(sq_a))
        diffs[i] = float(sq_a[idx].mean() - sq_b[idx].mean())
    diffs_c = diffs - diffs.mean()  # center under null hypothesis
    return float(np.mean(np.abs(diffs_c) >= np.abs(obs_diff)))


# ── Validation monthly aggregation ────────────────────────────────────────────
# Target month for each validation pixel (feature month + 1), matching test split.
_val_mo_keys = (
    pd.to_datetime(val["time"]) + pd.DateOffset(months=1)
).dt.to_period("M").dt.to_timestamp()

# Binary dry indicator and monthly dry-area fraction for the validation set.
_val_obs_bin = (val[TARGET] == -1).astype(int).values
_val_obs_mo  = (
    pd.Series((val[TARGET] == -1).astype(float).values, index=_val_mo_keys)
    .groupby(_val_mo_keys).mean()
)  # monthly dry-area fraction, validation set

# ── Optional: build XGB-Spatial validation probabilities ──────────────────────
# Requires the saved spatial model and the same gridded NetCDF files used during
# training (train_forecast_xgb_spatial.py).
_XGB_SP_MDL_PATH = Path("outputs/xgb_spatial_model.json")
_SPI_NC_PATH     = Path("data/processed/chirps_v3_monthly_cvalley_spi_1991_2025.nc")
_PR_NC_PATH      = Path("data/processed/chirps_v3_monthly_cvalley_1991_2025.nc")
_SPATIAL_FEAT    = ["spi1_nbr_mean", "spi3_nbr_mean", "spi6_nbr_mean", "pr_nbr_mean"]
_FEAT_WITH_SP    = FEATURES + _SPATIAL_FEAT  # full feature list including spatial

_sp_val_dry = None  # (n_val_pixels,) dry-class prob — filled below when possible

if (HAS_XGB_SPATIAL and _XGB_SP_MDL_PATH.exists()
        and _SPI_NC_PATH.exists() and _PR_NC_PATH.exists()):
    try:
        import xarray as xr
        print("  Calibration study: computing XGB-Spatial validation probabilities ...")
        _pr_ds  = xr.open_dataset(_PR_NC_PATH).load()
        _spi_ds = xr.open_dataset(_SPI_NC_PATH).load()
        _pra    = _pr_ds["pr"].astype("float32")
        _ts     = _pra.time.values
        _s1     = _spi_ds["spi1"].sel(time=_ts).astype("float32")
        _s3     = _spi_ds["spi3"].sel(time=_ts).astype("float32")
        _s6     = _spi_ds["spi6"].sel(time=_ts).astype("float32")
        if "latitude" in _pra.coords:
            _lat_coord = "latitude"
        elif "lat" in _pra.coords:
            _lat_coord = "lat"
        else:
            raise ValueError("No latitude coordinate found (expected 'latitude' or 'lat').")

        if "longitude" in _pra.coords:
            _lon_coord = "longitude"
        elif "lon" in _pra.coords:
            _lon_coord = "lon"
        else:
            raise ValueError("No longitude coordinate found (expected 'longitude' or 'lon').")

        def _compute_nbr_mean(da, nm):
            """3×3 neighbourhood rolling mean (mirrors train_forecast_xgb_spatial.py)."""
            return da.rolling(
                {_lat_coord: 3, _lon_coord: 3}, min_periods=1, center=True
            ).mean().rename(nm)

        _nds = xr.Dataset({
            "spi1_nbr_mean": _compute_nbr_mean(_s1, "spi1_nbr_mean"),
            "spi3_nbr_mean": _compute_nbr_mean(_s3, "spi3_nbr_mean"),
            "spi6_nbr_mean": _compute_nbr_mean(_s6, "spi6_nbr_mean"),
            "pr_nbr_mean":   _compute_nbr_mean(_pra, "pr_nbr_mean"),
        }).stack(pixel=(_lat_coord, _lon_coord))
        _ndf = _nds.reset_index("pixel").to_dataframe()
        if "time" not in _ndf.columns:
            _ndf = _ndf.reset_index()
        _ndf = _ndf.rename(columns={_lat_coord: "latitude", _lon_coord: "longitude"})
        _ndf["time"] = pd.to_datetime(_ndf["time"])

        _vsp = val.merge(
            _ndf[["time", "latitude", "longitude"] + _SPATIAL_FEAT],
            on=["time", "latitude", "longitude"], how="left",
        )
        _vsp[_SPATIAL_FEAT] = _vsp[_SPATIAL_FEAT].fillna(0.0)

        _spm  = xgb.Booster()
        _spm.load_model(_XGB_SP_MDL_PATH.as_posix())
        _dvsp = xgb.DMatrix(_vsp[_FEAT_WITH_SP], feature_names=_FEAT_WITH_SP)
        _sp_val_dry = _spm.predict(_dvsp).reshape(-1, 3)[:, 0]
        print(f"  XGB-Spatial val probs ready ({len(_sp_val_dry)} pixels).")
    except Exception as _exc:
        print(f"  WARNING: Calibration study — XGB-Spatial val probs unavailable ({_exc}).")
        _sp_val_dry = None


# ── Assemble (model_label, val_dry_pixel_probs, test_dry_pixel_probs) list ────
_test_mo_keys = pd.to_datetime(test["month_dt"])
_calib_targets = [("XGB", val_probs[:, 0], test["xgb_prob_dry"].values)]
if HAS_XGB_SPATIAL and _sp_val_dry is not None:
    _calib_targets.append(
        ("XGB-Spatial", _sp_val_dry, test["xgb_spatial_prob_dry"].values)
    )


# ── For each model: compare calibration methods, select best by val BS ────────
# NOTE:
#   We report only probabilistic skill (BS/BSS + decomposition) for calibrated
#   variants. Monotonic post-hoc calibration typically preserves hard-class
#   argmax predictions (HSS) and ranking-based ROC-AUC, though edge cases may exist.
calib_study_rows: list = []   # rows collected into CSV at end of section
_best_test_mo: dict   = {}    # model_label → best-calibrated monthly test dry fracs

print("\n── Calibration Study ────────────────────────────────────────────────────")
for _mlbl, _vdp, _tdp in _calib_targets:
    _best_mname = None
    _best_val_bs = np.inf
    _fitted: dict = {}

    for _mname in ("none", "platt", "isotonic"):
        # --- fit calibrator on pixel-level validation data ---
        if _mname == "none":
            _fitted["none"] = None
            _vdp_c = _vdp.copy()
        elif _mname == "platt":
            _fitted["platt"] = _fit_platt(_vdp, _val_obs_bin)
            _vdp_c = _apply_platt(_fitted["platt"], _vdp)
        else:
            _fitted["isotonic"] = _fit_isotonic_cal(_vdp, _val_obs_bin)
            _vdp_c = _apply_isotonic_cal(_fitted["isotonic"], _vdp)

        # --- aggregate to validation months and score ---
        _vmo  = pd.Series(_vdp_c, index=_val_mo_keys).groupby(_val_mo_keys).mean()
        _cidx = _vmo.index.intersection(_val_obs_mo.index)
        _vbs  = brier_score(_val_obs_mo.loc[_cidx].values, _vmo.loc[_cidx].values)
        if _vbs < _best_val_bs:
            _best_val_bs = _vbs
            _best_mname  = _mname

    # --- apply best calibrator to test pixels ---
    if _best_mname == "none":
        _tdp_c = _tdp.copy()
    elif _best_mname == "platt":
        _tdp_c = _apply_platt(_fitted["platt"], _tdp)
    else:
        _tdp_c = _apply_isotonic_cal(_fitted["isotonic"], _tdp)

    # --- aggregate to test months (aligned to the monthly DataFrame index) ---
    _tmo = (
        pd.Series(_tdp_c, index=_test_mo_keys)
        .groupby(_test_mo_keys).mean()
        .reindex(monthly["month_dt"]).values
    )
    _best_test_mo[_mlbl] = _tmo

    # --- test-set metrics ---
    _bs_t  = brier_score(obs_dry_frac, _tmo)
    _bss_t = bss(_bs_t, bs_clim)
    _d     = bs_decomp(obs_dry_frac, _tmo)
    calib_study_rows.append({
        "model":            _mlbl,
        "best_calibration": _best_mname,
        "val_BS_selected":  round(_best_val_bs, 5),
        "test_BS":          round(_bs_t,  5),
        "test_BSS":         round(_bss_t, 5),
        "reliability":      round(_d["reliability"], 5),
        "resolution":       round(_d["resolution"],  5),
        "uncertainty":      round(_d["uncertainty"],  5),
    })
    print(f"  {_mlbl}: best_calib={_best_mname}  val_BS={_best_val_bs:.5f}  "
          f"test_BS={_bs_t:.5f}  test_BSS={_bss_t:.5f}")
    print(f"    Decomp — reliability={_d['reliability']:.5f}  "
          f"resolution={_d['resolution']:.5f}  uncertainty={_d['uncertainty']:.5f}")


# ── Bootstrap CI and paired significance tests ─────────────────────────────────
_clim_sq = (obs_dry_frac - monthly["clim_dry_frac"].values) ** 2  # per-month sq errors (climatology)
for _ci, _crow in enumerate(calib_study_rows):
    _mo  = _best_test_mo[_crow["model"]]
    _sq  = (obs_dry_frac - _mo) ** 2

    # bootstrap CI for BS (use default-arg capture to avoid closure over loop var)
    _bs_ci = bootstrap_metric(
        lambda idx, _m=_mo: float(np.mean((obs_dry_frac[idx] - _m[idx]) ** 2)),
        n_months, n_boot=N_BOOTSTRAP_ITERATIONS, seed=200 + _ci,
    )
    # bootstrap CI for BSS (relative to climatology)
    _bss_ci = bootstrap_metric(
        lambda idx, _m=_mo: bss(
            float(np.mean((obs_dry_frac[idx] - _m[idx]) ** 2)),
            float(np.mean((obs_dry_frac[idx] - monthly["clim_dry_frac"].values[idx]) ** 2)),
        ), n_months, n_boot=N_BOOTSTRAP_ITERATIONS, seed=210 + _ci,
    )
    # paired test: this model vs climatology
    _p_clim = paired_boot_pvalue(_sq, _clim_sq, n_boot=N_BOOTSTRAP_ITERATIONS, seed=220 + _ci)
    calib_study_rows[_ci]["test_BS_95CI"]  = fmt_ci(_bs_ci)
    calib_study_rows[_ci]["test_BSS_95CI"] = fmt_ci(_bss_ci)
    calib_study_rows[_ci]["p_vs_clim"]     = round(_p_clim, 4)

# paired test between XGB-Spatial and XGB (if both are present)
_p_sp_vs_xgb = np.nan
if "XGB" in _best_test_mo and "XGB-Spatial" in _best_test_mo:
    _xsq = (obs_dry_frac - _best_test_mo["XGB"]) ** 2
    _ssq = (obs_dry_frac - _best_test_mo["XGB-Spatial"]) ** 2
    _p_sp_vs_xgb = paired_boot_pvalue(_ssq, _xsq, n_boot=N_BOOTSTRAP_ITERATIONS, seed=230)
    print(f"  Paired test (XGB-Spatial vs XGB): p = {_p_sp_vs_xgb:.4f}")

# build and save results CSV
_cdf = pd.DataFrame(calib_study_rows)
if not np.isnan(_p_sp_vs_xgb) and len(_cdf) >= 2:
    # annotate XGB-Spatial row with its significance vs XGB
    _cdf["p_vs_XGB"] = np.nan
    _cdf.loc[_cdf["model"] == "XGB-Spatial", "p_vs_XGB"] = round(float(_p_sp_vs_xgb), 4)
calib_csv_path = OUT_DIR / "calib_study_results.csv"
_cdf.to_csv(calib_csv_path, index=False)
print(f"  Wrote: {calib_csv_path}")


# ── Figure 1: Reliability diagram for calibrated XGB models (monthly level) ───
def _rel_curve(obs_arr: np.ndarray, pred_arr: np.ndarray,
               n_bins: int = 5) -> tuple:
    """Bin-average reliability curve for fractional observations at monthly level.

    Returns (mean_predicted, observed_fraction) arrays for non-empty bins.
    """
    edges = np.linspace(0.0, 1.0 + 1e-9, n_bins + 1)
    bidx  = np.clip(np.digitize(pred_arr, edges) - 1, 0, n_bins - 1)
    mp, of = [], []
    for k in range(n_bins):
        mask = bidx == k
        if mask.any():
            mp.append(float(pred_arr[mask].mean()))
            of.append(float(obs_arr[mask].mean()))
    return np.array(mp), np.array(of)


_fig_rel2, _ax_rel2 = plt.subplots(figsize=(6, 5))
_ax_rel2.plot([0, 1], [0, 1], "k--", lw=1.2, label="Perfect calibration")

# climatology reference curve
_mp_cl, _of_cl = _rel_curve(obs_dry_frac, monthly["clim_dry_frac"].values)
_ax_rel2.plot(_mp_cl, _of_cl, "s-", color="#aaaaaa", label="Climatology", ms=6)

# calibrated model curves
_rel2_colors = {"XGB": "#2171b5", "XGB-Spatial": "#238b45"}
for _rlbl, _rmo in _best_test_mo.items():
    _mp_r, _of_r = _rel_curve(obs_dry_frac, _rmo)
    _ax_rel2.plot(_mp_r, _of_r, "o-",
                  color=_rel2_colors.get(_rlbl, "#d62728"),
                  label=f"{_rlbl} (calibrated)", ms=6)

_ax_rel2.set_xlabel("Mean predicted probability (dry)")
_ax_rel2.set_ylabel("Observed dry-area fraction")
_ax_rel2.set_title("Reliability diagram — calibrated models\n(60 independent test months)")
_ax_rel2.legend(fontsize=9)
_ax_rel2.set_xlim(0, 1)
_ax_rel2.set_ylim(0, 1)
_fig_rel2.tight_layout()
_rel2_path = OUT_DIR / "calib_study_reliability_diagram.png"
_fig_rel2.savefig(_rel2_path, dpi=150, bbox_inches="tight")
plt.close(_fig_rel2)
print(f"  Wrote: {_rel2_path}")


# ── Figure 2: Brier Score decomposition barplot ────────────────────────────────
_clim_d = bs_decomp(obs_dry_frac, monthly["clim_dry_frac"].values)
_decomp_models: dict = {"Climatology": _clim_d}
for _dlbl, _dmo in _best_test_mo.items():
    _decomp_models[_dlbl] = bs_decomp(obs_dry_frac, _dmo)

_bar_lbls = list(_decomp_models.keys())
_n_bars   = len(_bar_lbls)
_x_pos    = np.arange(_n_bars)
_w        = 0.25
_comp_clr = {"reliability": "#e6550d", "resolution": "#31a354", "uncertainty": "#3182bd"}

_fig_dc, _ax_dc = plt.subplots(figsize=(max(6, 2 * _n_bars + 2), 4))
for _ki, _comp in enumerate(("reliability", "resolution", "uncertainty")):
    _vals = [_decomp_models[_lbl][_comp] for _lbl in _bar_lbls]
    _ax_dc.bar(_x_pos + (_ki - 1) * _w, _vals, _w,
               label=_comp.capitalize(), color=_comp_clr[_comp], alpha=0.85)

_ax_dc.set_xticks(_x_pos)
_ax_dc.set_xticklabels(_bar_lbls, rotation=15, ha="right")
_ax_dc.set_ylabel("Score component")
_ax_dc.set_title("Brier Score decomposition (test months)\n"
                 "BS = reliability − resolution + uncertainty")
_ax_dc.legend(fontsize=9)
_ax_dc.axhline(0, color="black", lw=0.7)
_fig_dc.tight_layout()
_dc_path = OUT_DIR / "calib_study_decomposition_barplot.png"
_fig_dc.savefig(_dc_path, dpi=150, bbox_inches="tight")
plt.close(_fig_dc)
print(f"  Wrote: {_dc_path}")

# ── End of Calibration Study section ──────────────────────────────────────────

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
    (axes[0], "xgb_prob_dry",     "XGBoost (raw)",        "#2171b5"),
    (axes[1], "xgb_prob_dry_cal", "XGBoost (calibrated)", "#238b45"),
]:
    # use pixel-level for calibration curve (more points = smoother bins)
    y_true_bin = (test[TARGET] == -1).astype(int)
    if col == "xgb_prob_dry_cal":
        y_prob = iso_cal.predict(test["xgb_prob_dry"].values)
    else:
        y_prob = test[col].values
    frac_pos, mean_pred = calibration_curve(
        y_true_bin,
        y_prob,
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
     "BSS_dry": "0.0000 (ref)", "BSS_dry_95CI": "—",
     "HSS": f"{hss_clim:.4f}", "HSS_95CI": fmt_ci(hss_ci_clim), "ROC-AUC_dry": "—"},
    {"Forecaster": "Persistence baseline",    "BS_dry": f"{bs_pers:.4f}",
     "BSS_dry": f"{bss_pers:.4f}", "BSS_dry_95CI": fmt_ci(bss_ci_pers),
     "HSS": f"{hss_pers:.4f}", "HSS_95CI": fmt_ci(hss_ci_pers), "ROC-AUC_dry": f"{auc_pers:.4f}"},
    {"Forecaster": "SPI-1 threshold baseline", "BS_dry": f"{bs_thr:.4f}",
     "BSS_dry": f"{bss_thr:.4f}", "BSS_dry_95CI": fmt_ci(bss_ci_thr),
     "HSS": f"{hss_thr:.4f}", "HSS_95CI": fmt_ci(hss_ci_thr), "ROC-AUC_dry": f"{auc_thr:.4f}"},
    {"Forecaster": "XGBoost (no spatial)",    "BS_dry": f"{bs_xgb:.4f}",
     "BSS_dry": f"{bss_xgb:.4f}",  "BSS_dry_95CI": fmt_ci(bss_ci_xgb),
     "HSS": f"{hss_xgb:.4f}", "HSS_95CI": fmt_ci(hss_ci_xgb), "ROC-AUC_dry": f"{auc_xgb:.4f}"},
]
if HAS_XGB_SPATIAL:
    rows.append(
        {"Forecaster": "XGBoost-Spatial",     "BS_dry": f"{bs_sp:.4f}",
         "BSS_dry": f"{bss_sp:.4f}", "BSS_dry_95CI": fmt_ci(bss_ci_sp),
         "HSS": f"{hss_sp:.4f}", "HSS_95CI": fmt_ci(hss_ci_sp), "ROC-AUC_dry": f"{auc_sp:.4f}"}
    )
if HAS_CONVLSTM:
    rows.append(
        {"Forecaster": "ConvLSTM",            "BS_dry": f"{bs_cl:.4f}",
         "BSS_dry": f"{bss_cl:.4f}", "BSS_dry_95CI": fmt_ci(bss_ci_cl),
         "HSS": f"{hss_cl:.4f}", "HSS_95CI": fmt_ci(hss_ci_cl), "ROC-AUC_dry": f"{auc_cl:.4f}"}
    )
if HAS_LOGREG:
    rows.append(
        {"Forecaster": "Logistic Regression", "BS_dry": f"{bs_lr:.4f}",
         "BSS_dry": f"{bss_lr:.4f}", "BSS_dry_95CI": fmt_ci(bss_ci_lr),
         "HSS": f"{hss_lr:.4f}", "HSS_95CI": fmt_ci(hss_ci_lr), "ROC-AUC_dry": f"{auc_lr:.4f}"}
    )
if HAS_RF:
    rows.append(
        {"Forecaster": "Random Forest",       "BS_dry": f"{bs_rf:.4f}",
         "BSS_dry": f"{bss_rf:.4f}", "BSS_dry_95CI": fmt_ci(bss_ci_rf),
         "HSS": f"{hss_rf:.4f}", "HSS_95CI": fmt_ci(hss_ci_rf), "ROC-AUC_dry": f"{auc_rf:.4f}"}
    )
table_df = pd.DataFrame(rows)
csv_path = OUT_DIR / "forecast_skill_bss_hss_table.csv"
table_df.to_csv(csv_path, index=False)
print("Wrote:", csv_path)

# ── text summary ──────────────────────────────────────────────────────────────
summary = (
    "Forecast Skill Evaluation — Central Valley 2021–2025\n"
    + "=" * 60 + "\n"
    f"Test months (independent temporal units): {n_months}\n"
    f"Pixels per month (spatially autocorrelated, secondary): "
    f"{len(test) // n_months:,}\n\n"
    "METHODOLOGICAL NOTE:\n"
    "  All PRIMARY metrics below are computed at the monthly level.\n"
    "  Pixel-level metrics inflate significance due to spatial autocorrelation.\n"
    "  BSS reference = climatological class frequency from training set (1991–2016).\n\n"
    "  Dry-event Brier target = observed monthly dry-area fraction (not a majority threshold).\n"
    "  ROC-AUC uses binary 'dry-dominant month' event for ranking diagnostics only.\n\n"
    "Spatial complexity ladder:\n"
    "  Climatology → Persistence → SPI-1 threshold heuristic → XGBoost (no spatial)\n"
    "    → XGBoost-Spatial (spatial features) → ConvLSTM (spatial architecture)\n\n"
    "── Monthly-level Brier Scores (dry class) ──\n"
    f"  Climatological (reference) : {bs_clim:.4f}\n"
    f"  Persistence baseline       : {bs_pers:.4f}\n"
    f"  SPI-1 threshold baseline   : {bs_thr:.4f}\n"
    f"  XGBoost (no spatial)       : {bs_xgb:.4f}\n"
    + (f"  XGBoost-Spatial            : {bs_sp:.4f}\n" if HAS_XGB_SPATIAL else "")
    + (f"  ConvLSTM                   : {bs_cl:.4f}\n" if HAS_CONVLSTM   else "")
    + (f"  Logistic Regression        : {bs_lr:.4f}\n" if HAS_LOGREG     else "")
    + (f"  Random Forest              : {bs_rf:.4f}\n" if HAS_RF         else "")
    + "\n── Brier Skill Score (BSS, dry class, ref = climatology) ──\n"
    f"  Persistence         : {bss_pers:.4f}  (95% CI {fmt_ci(bss_ci_pers)})\n"
    f"  SPI-1 threshold     : {bss_thr:.4f}  (95% CI {fmt_ci(bss_ci_thr)})\n"
    f"  XGBoost (no spatial): {bss_xgb:.4f}  (95% CI {fmt_ci(bss_ci_xgb)})\n"
    + (f"  XGBoost-Spatial     : {bss_sp:.4f}  (95% CI {fmt_ci(bss_ci_sp)})\n" if HAS_XGB_SPATIAL else "")
    + (f"  ConvLSTM            : {bss_cl:.4f}  (95% CI {fmt_ci(bss_ci_cl)})\n" if HAS_CONVLSTM   else "")
    + (f"  Logistic Regression : {bss_lr:.4f}  (95% CI {fmt_ci(bss_ci_lr)})\n" if HAS_LOGREG     else "")
    + (f"  Random Forest       : {bss_rf:.4f}  (95% CI {fmt_ci(bss_ci_rf)})\n" if HAS_RF         else "")
    + "\n── Heidke Skill Score (HSS, 3-class, monthly dominant class) ──\n"
    f"  Climatological      : {hss_clim:.4f}  (95% CI {fmt_ci(hss_ci_clim)})\n"
    f"  Persistence         : {hss_pers:.4f}  (95% CI {fmt_ci(hss_ci_pers)})\n"
    f"  SPI-1 threshold     : {hss_thr:.4f}  (95% CI {fmt_ci(hss_ci_thr)})\n"
    f"  XGBoost (no spatial): {hss_xgb:.4f}  (95% CI {fmt_ci(hss_ci_xgb)})\n"
    + (f"  XGBoost-Spatial     : {hss_sp:.4f}  (95% CI {fmt_ci(hss_ci_sp)})\n" if HAS_XGB_SPATIAL else "")
    + (f"  ConvLSTM            : {hss_cl:.4f}  (95% CI {fmt_ci(hss_ci_cl)})\n" if HAS_CONVLSTM   else "")
    + (f"  Logistic Regression : {hss_lr:.4f}  (95% CI {fmt_ci(hss_ci_lr)})\n" if HAS_LOGREG     else "")
    + (f"  Random Forest       : {hss_rf:.4f}  (95% CI {fmt_ci(hss_ci_rf)})\n" if HAS_RF         else "")
    + "\n── ROC-AUC (dry vs. not-dry, monthly mean probability) ──\n"
    f"  Persistence         : {auc_pers:.4f}\n"
    f"  SPI-1 threshold     : {auc_thr:.4f}\n"
    f"  XGBoost (no spatial): {auc_xgb:.4f}\n"
    + (f"  XGBoost-Spatial     : {auc_sp:.4f}\n" if HAS_XGB_SPATIAL else "")
    + (f"  ConvLSTM            : {auc_cl:.4f}\n" if HAS_CONVLSTM   else "")
    + (f"  Logistic Regression : {auc_lr:.4f}\n" if HAS_LOGREG     else "")
    + (f"  Random Forest       : {auc_rf:.4f}\n" if HAS_RF         else "")
    + "\nOutputs:\n"
    f"  {cm_path}\n"
    f"  {rel_path}\n"
    f"  {csv_path}\n"
    f"  {calib_csv_path} (calibration study)\n"
    f"  {_rel2_path} (calibration study)\n"
    f"  {_dc_path} (calibration study)\n"
)
print(summary)
(OUT_DIR / "forecast_skill_scores.txt").write_text(summary)
print("Wrote:", OUT_DIR / "forecast_skill_scores.txt")
