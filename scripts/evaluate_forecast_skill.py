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
  2. Persistence:    predict label[t+1] = label[t] (current month SPI-1 class persists).
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
  outputs/forecast_calibration_comparison.csv
  outputs/forecast_bs_decomposition.csv
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
from sklearn.linear_model import LogisticRegression as _PlattLR
from sklearn.metrics import roc_auc_score, confusion_matrix, ConfusionMatrixDisplay

DATA          = Path("data/processed/dataset_forecast.parquet")
PROBS_NPZ     = Path("outputs/forecast_xgb_test_probs.npz")
MODEL_PATH    = Path("outputs/forecast_xgb_model.json")
LOGREG_MODEL  = Path("outputs/forecast_logreg_model.pkl")
RF_MODEL      = Path("outputs/forecast_rf_model.pkl")
XGB_SPATIAL_NPZ     = Path("outputs/xgb_spatial_test_probs.npz")
XGB_SPATIAL_VAL_NPZ = Path("outputs/xgb_spatial_val_probs.npz")
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
SPI_HEURISTIC_SCALE = 2.0
N_BOOTSTRAP_ITERATIONS = 2000

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

# ── Platt (logistic) scaling on validation set (dry class) ───────────────────
platt_cal = _PlattLR(max_iter=1000, solver="lbfgs", C=1.0)
platt_cal.fit(
    val_probs[:, 0].reshape(-1, 1),
    (val_y_enc == LABEL_MAP[-1]).astype(int),
)
test["xgb_prob_dry_platt"] = platt_cal.predict_proba(
    test["xgb_prob_dry"].values.reshape(-1, 1)
)[:, 1]

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
    sp_probs     = sp_loaded["proba"]          # (n_rows, 3)  iso-calibrated
    sp_probs_raw = sp_loaded["proba_raw"] if "proba_raw" in sp_loaded else sp_probs
    assert len(sp_probs) == len(test), (
        f"XGB-Spatial probs length ({len(sp_probs)}) != test rows ({len(test)}). "
        "Retrain the spatial model and re-run."
    )
    test["xgb_spatial_prob_dry"]     = sp_probs[:, 0]
    test["xgb_spatial_prob_normal"]  = sp_probs[:, 1]
    test["xgb_spatial_prob_wet"]     = sp_probs[:, 2]
    test["xgb_spatial_prob_dry_raw"] = sp_probs_raw[:, 0]
    test["xgb_spatial_pred"]         = np.array(
        [INV_LABEL_MAP[i] for i in sp_probs.argmax(axis=1)]
    )
    HAS_XGB_SPATIAL = True
else:
    print("WARNING: XGBoost-Spatial probs not found at", XGB_SPATIAL_NPZ)
    HAS_XGB_SPATIAL = False

# ── XGBoost-Spatial calibration study (validation-only fitting) ──────────────
# Requires xgb_spatial_val_probs.npz produced by train_forecast_xgb_spatial.py.
# Fits isotonic regression and Platt scaling independently on the validation set
# and applies them to the frozen test raw probabilities to avoid leakage.
HAS_XGB_SPATIAL_CAL = False
if HAS_XGB_SPATIAL and XGB_SPATIAL_VAL_NPZ.exists():
    print("Fitting XGB-Spatial calibrators from validation probs...")
    sp_val       = np.load(XGB_SPATIAL_VAL_NPZ, allow_pickle=True)
    sp_val_raw   = sp_val["proba_raw"]       # (n_val, 3) uncalibrated val probs
    sp_val_y_enc = sp_val["y_val_enc"]       # encoded val labels
    sp_val_bin   = (sp_val_y_enc == LABEL_MAP[-1]).astype(int)

    iso_sp = IsotonicRegression(out_of_bounds="clip")
    iso_sp.fit(sp_val_raw[:, 0], sp_val_bin)
    test["xgb_spatial_prob_dry_iso"] = iso_sp.predict(sp_probs_raw[:, 0])

    platt_sp = _PlattLR(max_iter=1000, solver="lbfgs", C=1.0)
    platt_sp.fit(sp_val_raw[:, 0].reshape(-1, 1), sp_val_bin)
    test["xgb_spatial_prob_dry_platt"] = platt_sp.predict_proba(
        sp_probs_raw[:, 0].reshape(-1, 1)
    )[:, 1]
    HAS_XGB_SPATIAL_CAL = True
    print("  XGB-Spatial: isotonic and Platt calibrators fitted on validation set.")
elif HAS_XGB_SPATIAL:
    print(
        "NOTE: xgb_spatial_val_probs.npz not found. Re-run "
        "train_forecast_xgb_spatial.py to enable XGB-Spatial calibration study."
    )

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


def brier_decompose(
    y_true_bin: np.ndarray, prob: np.ndarray, n_bins: int = 10
) -> dict:
    """Murphy (1973) Brier Score decomposition into reliability, resolution,
    and uncertainty components.

    BS = reliability - resolution + uncertainty

    Lower reliability (calibration error) and higher resolution (discrimination)
    both improve BS.  Uncertainty is irreducible given the base rate.

    Parameters
    ----------
    y_true_bin : array of 0/1 observed outcomes
    prob       : array of forecast probabilities for the positive class
    n_bins     : number of equally-spaced probability bins (default 10)

    Returns
    -------
    dict with keys: bs, reliability, resolution, uncertainty
    """
    o_bar = float(y_true_bin.mean())
    uncertainty = o_bar * (1.0 - o_bar)

    bins    = np.linspace(0.0, 1.0 + 1e-9, n_bins + 1)
    bin_idx = np.clip(np.digitize(prob, bins) - 1, 0, n_bins - 1)

    reliability = 0.0
    resolution  = 0.0
    n = len(y_true_bin)
    for k in range(n_bins):
        mask = bin_idx == k
        n_k  = int(mask.sum())
        if n_k == 0:
            continue
        f_k = float(prob[mask].mean())
        o_k = float(y_true_bin[mask].mean())
        reliability += n_k * (f_k - o_k) ** 2
        resolution  += n_k * (o_k - o_bar) ** 2

    reliability /= n
    resolution  /= n
    return {
        "bs":          float(np.mean((prob - y_true_bin) ** 2)),
        "reliability": reliability,
        "resolution":  resolution,
        "uncertainty": uncertainty,
    }


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
# Use ±2 SPI as soft saturation bounds for this simple heuristic map.
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
    y_true_mode      = (TARGET, lambda s: int(s.mode()[0])),
    y_true_dry_frac  = (TARGET, lambda s: (s == -1).mean()),
    xgb_dry_frac     = ("xgb_prob_dry",       "mean"),
    xgb_norm_frac    = ("xgb_prob_normal",     "mean"),
    xgb_wet_frac     = ("xgb_prob_wet",        "mean"),
    xgb_dry_cal_frac = ("xgb_prob_dry_cal",    "mean"),
    xgb_dry_platt_frac=("xgb_prob_dry_platt",  "mean"),
    clim_dry_frac    = (f"clim_prob_{-1}",     "mean"),
    clim_norm_frac   = (f"clim_prob_{0}",      "mean"),
    clim_wet_frac    = (f"clim_prob_{1}",      "mean"),
    persist_dry_frac = (f"persist_prob_{-1}",  "mean"),
    persist_norm_frac= (f"persist_prob_{0}",   "mean"),
    persist_wet_frac = (f"persist_prob_{1}",   "mean"),
    thr_dry_frac     = (f"thr_prob_{-1}",      "mean"),
    thr_norm_frac    = (f"thr_prob_{0}",       "mean"),
    thr_wet_frac     = (f"thr_prob_{1}",       "mean"),
    xgb_pred_mode    = ("xgb_pred",    lambda s: int(s.mode()[0])),
    clim_pred_mode   = ("clim_pred",   lambda s: int(s.mode()[0])),
    persist_pred_mode= ("persist_pred",lambda s: int(s.mode()[0])),
    thr_pred_mode    = ("thr_pred",    lambda s: int(s.mode()[0])),
    **({f"lr_dry_frac":  (f"lr_prob_{-1}", "mean"),
        f"lr_pred_mode": ("lr_pred", lambda s: int(s.mode()[0]))}
       if HAS_LOGREG else {}),
    **({f"rf_dry_frac":  (f"rf_prob_{-1}", "mean"),
        f"rf_pred_mode": ("rf_pred", lambda s: int(s.mode()[0]))}
       if HAS_RF else {}),
    **({"xgb_spatial_dry_frac":     ("xgb_spatial_prob_dry",      "mean"),
        "xgb_spatial_dry_raw_frac": ("xgb_spatial_prob_dry_raw",  "mean"),
        "xgb_spatial_pred_mode":    ("xgb_spatial_pred", lambda s: int(s.mode()[0]))}
       if HAS_XGB_SPATIAL else {}),
    **({"xgb_spatial_dry_iso_frac":   ("xgb_spatial_prob_dry_iso",   "mean"),
        "xgb_spatial_dry_platt_frac": ("xgb_spatial_prob_dry_platt", "mean")}
       if HAS_XGB_SPATIAL_CAL else {}),
    **({"convlstm_dry_frac":  ("convlstm_prob_dry", "mean"),
        "convlstm_pred_mode": ("convlstm_pred", lambda s: int(s.mode()[0]))}
       if HAS_CONVLSTM else {}),
).reset_index()

# Monthly dry-event target for probabilistic scoring:
# use observed monthly dry fraction directly (0..1) to avoid arbitrary majority threshold.
monthly["obs_dry_frac"] = monthly["y_true_dry_frac"].astype(float)

# Binary monthly dry event (for ROC-AUC only): dominant class is dry.
monthly["obs_dry_bin"] = (monthly["y_true_mode"] == -1).astype(int)

n_months = len(monthly)
print(f"Test months: {n_months}")

# ── Brier Scores ─────────────────────────────────────────────────────────────
obs_dry_frac = monthly["obs_dry_frac"].values

bs_xgb       = brier_score(obs_dry_frac, monthly["xgb_dry_frac"].values)
bs_xgb_iso   = brier_score(obs_dry_frac, monthly["xgb_dry_cal_frac"].values)
bs_xgb_platt = brier_score(obs_dry_frac, monthly["xgb_dry_platt_frac"].values)
bs_clim      = brier_score(obs_dry_frac, monthly["clim_dry_frac"].values)
bs_pers      = brier_score(obs_dry_frac, monthly["persist_dry_frac"].values)
bs_thr       = brier_score(obs_dry_frac, monthly["thr_dry_frac"].values)

bss_xgb       = bss(bs_xgb,       bs_clim)
bss_xgb_iso   = bss(bs_xgb_iso,   bs_clim)
bss_xgb_platt = bss(bs_xgb_platt, bs_clim)
bss_pers      = bss(bs_pers, bs_clim)
bss_thr       = bss(bs_thr,  bs_clim)

# ── Brier Score decomposition (Murphy 1973) ───────────────────────────────────
# BS = reliability - resolution + uncertainty
# Lower reliability (calibration error) + higher resolution = better score.
decomp_clim  = brier_decompose(obs_dry_frac, monthly["clim_dry_frac"].values)
decomp_xgb   = brier_decompose(obs_dry_frac, monthly["xgb_dry_frac"].values)
decomp_xgb_iso   = brier_decompose(obs_dry_frac, monthly["xgb_dry_cal_frac"].values)
decomp_xgb_platt = brier_decompose(obs_dry_frac, monthly["xgb_dry_platt_frac"].values)
decomp_pers  = brier_decompose(obs_dry_frac, monthly["persist_dry_frac"].values)
decomp_thr   = brier_decompose(obs_dry_frac, monthly["thr_dry_frac"].values)

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
    decomp_lr = brier_decompose(obs_dry_frac, monthly["lr_dry_frac"].values)
    try:
        auc_lr = roc_auc_score(obs_dry_bin, monthly["lr_dry_frac"].values)
    except Exception:
        auc_lr = np.nan

if HAS_RF:
    bs_rf    = brier_score(obs_dry_frac, monthly["rf_dry_frac"].values)
    bss_rf   = bss(bs_rf, bs_clim)
    hss_rf   = heidke_skill_score(y_true_monthly, monthly["rf_pred_mode"].values, CLASSES)
    decomp_rf = brier_decompose(obs_dry_frac, monthly["rf_dry_frac"].values)
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
    decomp_sp = brier_decompose(obs_dry_frac, monthly["xgb_spatial_dry_frac"].values)
    try:
        auc_sp = roc_auc_score(obs_dry_bin, monthly["xgb_spatial_dry_frac"].values)
    except Exception:
        auc_sp = np.nan

if HAS_XGB_SPATIAL_CAL:
    bs_sp_iso   = brier_score(obs_dry_frac, monthly["xgb_spatial_dry_iso_frac"].values)
    bs_sp_platt = brier_score(obs_dry_frac, monthly["xgb_spatial_dry_platt_frac"].values)
    bss_sp_iso   = bss(bs_sp_iso,   bs_clim)
    bss_sp_platt = bss(bs_sp_platt, bs_clim)
    decomp_sp_raw   = brier_decompose(
        obs_dry_frac, monthly["xgb_spatial_dry_raw_frac"].values
    )
    decomp_sp_iso   = brier_decompose(
        obs_dry_frac, monthly["xgb_spatial_dry_iso_frac"].values
    )
    decomp_sp_platt = brier_decompose(
        obs_dry_frac, monthly["xgb_spatial_dry_platt_frac"].values
    )

if HAS_CONVLSTM:
    bs_cl   = brier_score(obs_dry_frac, monthly["convlstm_dry_frac"].values)
    bss_cl  = bss(bs_cl, bs_clim)
    hss_cl  = heidke_skill_score(
        y_true_monthly, monthly["convlstm_pred_mode"].values, CLASSES
    )
    decomp_cl = brier_decompose(obs_dry_frac, monthly["convlstm_dry_frac"].values)
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
bss_ci_xgb_iso = bootstrap_metric(lambda i: bss(
    brier_score(obs_dry_frac[i], monthly["xgb_dry_cal_frac"].values[i]),
    brier_score(obs_dry_frac[i], monthly["clim_dry_frac"].values[i])
), n_months, n_boot=N_BOOTSTRAP_ITERATIONS, seed=104)
bss_ci_xgb_platt = bootstrap_metric(lambda i: bss(
    brier_score(obs_dry_frac[i], monthly["xgb_dry_platt_frac"].values[i]),
    brier_score(obs_dry_frac[i], monthly["clim_dry_frac"].values[i])
), n_months, n_boot=N_BOOTSTRAP_ITERATIONS, seed=105)
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
if HAS_XGB_SPATIAL:
    bss_ci_sp = bootstrap_metric(lambda i: bss(
        brier_score(obs_dry_frac[i], monthly["xgb_spatial_dry_frac"].values[i]),
        brier_score(obs_dry_frac[i], monthly["clim_dry_frac"].values[i])
    ), n_months, n_boot=N_BOOTSTRAP_ITERATIONS, seed=121)
    hss_ci_sp = bootstrap_metric(lambda i: heidke_skill_score(
        y_true_monthly[i], monthly["xgb_spatial_pred_mode"].values[i], CLASSES
    ), n_months, n_boot=N_BOOTSTRAP_ITERATIONS, seed=122)
if HAS_XGB_SPATIAL_CAL:
    bss_ci_sp_iso = bootstrap_metric(lambda i: bss(
        brier_score(obs_dry_frac[i], monthly["xgb_spatial_dry_iso_frac"].values[i]),
        brier_score(obs_dry_frac[i], monthly["clim_dry_frac"].values[i])
    ), n_months, n_boot=N_BOOTSTRAP_ITERATIONS, seed=123)
    bss_ci_sp_platt = bootstrap_metric(lambda i: bss(
        brier_score(obs_dry_frac[i], monthly["xgb_spatial_dry_platt_frac"].values[i]),
        brier_score(obs_dry_frac[i], monthly["clim_dry_frac"].values[i])
    ), n_months, n_boot=N_BOOTSTRAP_ITERATIONS, seed=124)
if HAS_LOGREG:
    bss_ci_lr = bootstrap_metric(lambda i: bss(
        brier_score(obs_dry_frac[i], monthly["lr_dry_frac"].values[i]),
        brier_score(obs_dry_frac[i], monthly["clim_dry_frac"].values[i])
    ), n_months, n_boot=N_BOOTSTRAP_ITERATIONS, seed=131)
    hss_ci_lr = bootstrap_metric(lambda i: heidke_skill_score(
        y_true_monthly[i], monthly["lr_pred_mode"].values[i], CLASSES
    ), n_months, n_boot=N_BOOTSTRAP_ITERATIONS, seed=132)
if HAS_RF:
    bss_ci_rf = bootstrap_metric(lambda i: bss(
        brier_score(obs_dry_frac[i], monthly["rf_dry_frac"].values[i]),
        brier_score(obs_dry_frac[i], monthly["clim_dry_frac"].values[i])
    ), n_months, n_boot=N_BOOTSTRAP_ITERATIONS, seed=141)
    hss_ci_rf = bootstrap_metric(lambda i: heidke_skill_score(
        y_true_monthly[i], monthly["rf_pred_mode"].values[i], CLASSES
    ), n_months, n_boot=N_BOOTSTRAP_ITERATIONS, seed=142)
if HAS_CONVLSTM:
    bss_ci_cl = bootstrap_metric(lambda i: bss(
        brier_score(obs_dry_frac[i], monthly["convlstm_dry_frac"].values[i]),
        brier_score(obs_dry_frac[i], monthly["clim_dry_frac"].values[i])
    ), n_months, n_boot=N_BOOTSTRAP_ITERATIONS, seed=151)
    hss_ci_cl = bootstrap_metric(lambda i: heidke_skill_score(
        y_true_monthly[i], monthly["convlstm_pred_mode"].values[i], CLASSES
    ), n_months, n_boot=N_BOOTSTRAP_ITERATIONS, seed=152)

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
# 4-panel layout: XGB raw / XGB isotonic / XGB Platt / XGB-Spatial iso-calibrated
# (pixel-level observations give smoother calibration curves than monthly)
y_true_bin_px = (test[TARGET] == -1).astype(int)

_rel_panels = [
    ("xgb_prob_dry",       "XGBoost (raw)",           "#2171b5"),
    ("xgb_prob_dry_cal",   "XGBoost + Isotonic",       "#238b45"),
    ("xgb_prob_dry_platt", "XGBoost + Platt",          "#d94801"),
]
if HAS_XGB_SPATIAL:
    _rel_panels.append(
        ("xgb_spatial_prob_dry", "XGBoost-Spatial (iso)", "#756bb1")
    )

n_panels = len(_rel_panels)
fig_rel, axes = plt.subplots(1, n_panels, figsize=(4.5 * n_panels, 5), sharey=True)
if n_panels == 1:
    axes = [axes]

for ax, (col, label, color) in zip(axes, _rel_panels):
    y_prob = test[col].values
    frac_pos, mean_pred = calibration_curve(
        y_true_bin_px, y_prob, n_bins=10, strategy="uniform"
    )
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Perfect calibration")
    ax.plot(mean_pred, frac_pos, "o-", color=color, label=label)
    ax.set_xlabel("Mean predicted probability (dry class)")
    ax.set_ylabel("Observed frequency (dry class)")
    ax.set_title(label)
    ax.legend(fontsize=8)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)

fig_rel.suptitle("Reliability diagram — dry class probability (pixel-level)", fontsize=11)
fig_rel.tight_layout()
rel_path = OUT_DIR / "forecast_reliability_diagram.png"
fig_rel.savefig(rel_path, dpi=150, bbox_inches="tight")
plt.close(fig_rel)
print("Wrote:", rel_path)

# ── Calibration comparison table (CSV) ────────────────────────────────────────
cal_rows = [
    {"Model": "XGBoost (no spatial)", "Calibration": "None (raw)",
     "BS": f"{bs_xgb:.4f}", "BSS": f"{bss_xgb:.4f}",
     "BSS_95CI": fmt_ci(bss_ci_xgb),
     "Reliability": f"{decomp_xgb['reliability']:.4f}",
     "Resolution":  f"{decomp_xgb['resolution']:.4f}",
     "Uncertainty": f"{decomp_xgb['uncertainty']:.4f}"},
    {"Model": "XGBoost (no spatial)", "Calibration": "Isotonic (val-fitted)",
     "BS": f"{bs_xgb_iso:.4f}", "BSS": f"{bss_xgb_iso:.4f}",
     "BSS_95CI": fmt_ci(bss_ci_xgb_iso),
     "Reliability": f"{decomp_xgb_iso['reliability']:.4f}",
     "Resolution":  f"{decomp_xgb_iso['resolution']:.4f}",
     "Uncertainty": f"{decomp_xgb_iso['uncertainty']:.4f}"},
    {"Model": "XGBoost (no spatial)", "Calibration": "Platt (val-fitted)",
     "BS": f"{bs_xgb_platt:.4f}", "BSS": f"{bss_xgb_platt:.4f}",
     "BSS_95CI": fmt_ci(bss_ci_xgb_platt),
     "Reliability": f"{decomp_xgb_platt['reliability']:.4f}",
     "Resolution":  f"{decomp_xgb_platt['resolution']:.4f}",
     "Uncertainty": f"{decomp_xgb_platt['uncertainty']:.4f}"},
]
if HAS_XGB_SPATIAL_CAL:
    cal_rows += [
        {"Model": "XGBoost-Spatial", "Calibration": "Raw",
         "BS": f"{brier_score(obs_dry_frac, monthly['xgb_spatial_dry_raw_frac'].values):.4f}",
         "BSS": f"{bss(brier_score(obs_dry_frac, monthly['xgb_spatial_dry_raw_frac'].values), bs_clim):.4f}",
         "BSS_95CI": "—",
         "Reliability": f"{decomp_sp_raw['reliability']:.4f}",
         "Resolution":  f"{decomp_sp_raw['resolution']:.4f}",
         "Uncertainty": f"{decomp_sp_raw['uncertainty']:.4f}"},
        {"Model": "XGBoost-Spatial", "Calibration": "Isotonic (val-fitted)",
         "BS": f"{bs_sp_iso:.4f}", "BSS": f"{bss_sp_iso:.4f}",
         "BSS_95CI": fmt_ci(bss_ci_sp_iso),
         "Reliability": f"{decomp_sp_iso['reliability']:.4f}",
         "Resolution":  f"{decomp_sp_iso['resolution']:.4f}",
         "Uncertainty": f"{decomp_sp_iso['uncertainty']:.4f}"},
        {"Model": "XGBoost-Spatial", "Calibration": "Platt (val-fitted)",
         "BS": f"{bs_sp_platt:.4f}", "BSS": f"{bss_sp_platt:.4f}",
         "BSS_95CI": fmt_ci(bss_ci_sp_platt),
         "Reliability": f"{decomp_sp_platt['reliability']:.4f}",
         "Resolution":  f"{decomp_sp_platt['resolution']:.4f}",
         "Uncertainty": f"{decomp_sp_platt['uncertainty']:.4f}"},
    ]
elif HAS_XGB_SPATIAL:
    cal_rows.append(
        {"Model": "XGBoost-Spatial", "Calibration": "Isotonic (train-script)",
         "BS": f"{bs_sp:.4f}", "BSS": f"{bss_sp:.4f}",
         "BSS_95CI": fmt_ci(bss_ci_sp),
         "Reliability": f"{decomp_sp['reliability']:.4f}",
         "Resolution":  f"{decomp_sp['resolution']:.4f}",
         "Uncertainty": f"{decomp_sp['uncertainty']:.4f}"},
    )
cal_df = pd.DataFrame(cal_rows)
cal_csv_path = OUT_DIR / "forecast_calibration_comparison.csv"
cal_df.to_csv(cal_csv_path, index=False)
print("Wrote:", cal_csv_path)

# ── Brier Score decomposition table (CSV) ────────────────────────────────────
def _decomp_row(name: str, d: dict, bss_val: float, ci: tuple | None = None) -> dict:
    return {
        "Forecaster": name,
        "BS":          f"{d['bs']:.4f}",
        "BSS":         f"{bss_val:.4f}",
        "BSS_95CI":    fmt_ci(ci) if ci is not None else "—",
        "Reliability": f"{d['reliability']:.4f}",
        "Resolution":  f"{d['resolution']:.4f}",
        "Uncertainty": f"{d['uncertainty']:.4f}",
    }

decomp_rows = [
    _decomp_row("Climatological baseline", decomp_clim, 0.0),
    _decomp_row("Persistence baseline",    decomp_pers, bss_pers,   bss_ci_pers),
    _decomp_row("SPI-1 threshold",         decomp_thr,  bss_thr,    bss_ci_thr),
    _decomp_row("XGBoost (raw)",           decomp_xgb,  bss_xgb,    bss_ci_xgb),
    _decomp_row("XGBoost + Isotonic",      decomp_xgb_iso,   bss_xgb_iso,   bss_ci_xgb_iso),
    _decomp_row("XGBoost + Platt",         decomp_xgb_platt, bss_xgb_platt, bss_ci_xgb_platt),
]
if HAS_XGB_SPATIAL:
    decomp_rows.append(_decomp_row("XGBoost-Spatial (iso)", decomp_sp, bss_sp, bss_ci_sp))
if HAS_XGB_SPATIAL_CAL:
    decomp_rows.append(_decomp_row("XGBoost-Spatial + Isotonic", decomp_sp_iso,   bss_sp_iso,   bss_ci_sp_iso))
    decomp_rows.append(_decomp_row("XGBoost-Spatial + Platt",    decomp_sp_platt, bss_sp_platt, bss_ci_sp_platt))
if HAS_CONVLSTM:
    decomp_rows.append(_decomp_row("ConvLSTM", decomp_cl, bss_cl, bss_ci_cl))
if HAS_LOGREG:
    decomp_rows.append(_decomp_row("Logistic Regression", decomp_lr, bss_lr, bss_ci_lr))
if HAS_RF:
    decomp_rows.append(_decomp_row("Random Forest", decomp_rf, bss_rf, bss_ci_rf))

decomp_df = pd.DataFrame(decomp_rows)
decomp_csv_path = OUT_DIR / "forecast_bs_decomposition.csv"
decomp_df.to_csv(decomp_csv_path, index=False)
print("Wrote:", decomp_csv_path)

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
    {"Forecaster": "XGBoost (no spatial)",     "BS_dry": f"{bs_xgb:.4f}",
     "BSS_dry": f"{bss_xgb:.4f}", "BSS_dry_95CI": fmt_ci(bss_ci_xgb),
     "HSS": f"{hss_xgb:.4f}", "HSS_95CI": fmt_ci(hss_ci_xgb), "ROC-AUC_dry": f"{auc_xgb:.4f}"},
    {"Forecaster": "XGBoost + Isotonic",       "BS_dry": f"{bs_xgb_iso:.4f}",
     "BSS_dry": f"{bss_xgb_iso:.4f}", "BSS_dry_95CI": fmt_ci(bss_ci_xgb_iso),
     "HSS": f"{hss_xgb:.4f}", "HSS_95CI": fmt_ci(hss_ci_xgb), "ROC-AUC_dry": f"{auc_xgb:.4f}"},
    {"Forecaster": "XGBoost + Platt",          "BS_dry": f"{bs_xgb_platt:.4f}",
     "BSS_dry": f"{bss_xgb_platt:.4f}", "BSS_dry_95CI": fmt_ci(bss_ci_xgb_platt),
     "HSS": f"{hss_xgb:.4f}", "HSS_95CI": fmt_ci(hss_ci_xgb), "ROC-AUC_dry": f"{auc_xgb:.4f}"},
]
if HAS_XGB_SPATIAL:
    rows.append(
        {"Forecaster": "XGBoost-Spatial",      "BS_dry": f"{bs_sp:.4f}",
         "BSS_dry": f"{bss_sp:.4f}", "BSS_dry_95CI": fmt_ci(bss_ci_sp),
         "HSS": f"{hss_sp:.4f}", "HSS_95CI": fmt_ci(hss_ci_sp), "ROC-AUC_dry": f"{auc_sp:.4f}"}
    )
if HAS_XGB_SPATIAL_CAL:
    rows.append(
        {"Forecaster": "XGBoost-Spatial + Isotonic", "BS_dry": f"{bs_sp_iso:.4f}",
         "BSS_dry": f"{bss_sp_iso:.4f}", "BSS_dry_95CI": fmt_ci(bss_ci_sp_iso),
         "HSS": f"{hss_sp:.4f}", "HSS_95CI": fmt_ci(hss_ci_sp), "ROC-AUC_dry": f"{auc_sp:.4f}"}
    )
    rows.append(
        {"Forecaster": "XGBoost-Spatial + Platt",    "BS_dry": f"{bs_sp_platt:.4f}",
         "BSS_dry": f"{bss_sp_platt:.4f}", "BSS_dry_95CI": fmt_ci(bss_ci_sp_platt),
         "HSS": f"{hss_sp:.4f}", "HSS_95CI": fmt_ci(hss_ci_sp), "ROC-AUC_dry": f"{auc_sp:.4f}"}
    )
if HAS_CONVLSTM:
    rows.append(
        {"Forecaster": "ConvLSTM",             "BS_dry": f"{bs_cl:.4f}",
         "BSS_dry": f"{bss_cl:.4f}", "BSS_dry_95CI": fmt_ci(bss_ci_cl),
         "HSS": f"{hss_cl:.4f}", "HSS_95CI": fmt_ci(hss_ci_cl), "ROC-AUC_dry": f"{auc_cl:.4f}"}
    )
if HAS_LOGREG:
    rows.append(
        {"Forecaster": "Logistic Regression",  "BS_dry": f"{bs_lr:.4f}",
         "BSS_dry": f"{bss_lr:.4f}", "BSS_dry_95CI": fmt_ci(bss_ci_lr),
         "HSS": f"{hss_lr:.4f}", "HSS_95CI": fmt_ci(hss_ci_lr), "ROC-AUC_dry": f"{auc_lr:.4f}"}
    )
if HAS_RF:
    rows.append(
        {"Forecaster": "Random Forest",        "BS_dry": f"{bs_rf:.4f}",
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
    + f"Test months (independent temporal units): {n_months}\n"
    + f"Pixels per month (spatially autocorrelated, secondary): "
    + f"{len(test) // n_months:,}\n\n"
    + "METHODOLOGICAL NOTE:\n"
    + "  All PRIMARY metrics below are computed at the monthly level.\n"
    + "  Pixel-level metrics inflate significance due to spatial autocorrelation.\n"
    + "  BSS reference = climatological class frequency from training set (1991–2016).\n\n"
    + "  Dry-event Brier target = observed monthly dry-area fraction (not a majority threshold).\n"
    + "  ROC-AUC uses binary 'dry-dominant month' event for ranking diagnostics only.\n\n"
    + "  Calibration: isotonic and Platt scalers fitted on 2017–2020 validation set only.\n"
    + "  Test set (2021–2025) is frozen throughout; no leakage into calibration.\n\n"
    + "  BS decomposition (Murphy 1973): BS = reliability - resolution + uncertainty.\n"
    + "  Lower reliability (calibration error) and higher resolution both improve BS.\n\n"
    + "Spatial complexity ladder:\n"
    + "  Climatology → Persistence → SPI-1 threshold heuristic → XGBoost (no spatial)\n"
    + "    → XGBoost-Spatial (spatial features) → ConvLSTM (spatial architecture)\n\n"
    + "── Monthly-level Brier Scores (dry class) ──\n"
    + f"  Climatological (reference) : {bs_clim:.4f}\n"
    + f"  Persistence baseline       : {bs_pers:.4f}\n"
    + f"  SPI-1 threshold baseline   : {bs_thr:.4f}\n"
    + f"  XGBoost (no spatial) raw   : {bs_xgb:.4f}\n"
    + f"  XGBoost + Isotonic cal     : {bs_xgb_iso:.4f}\n"
    + f"  XGBoost + Platt cal        : {bs_xgb_platt:.4f}\n"
    + (f"  XGBoost-Spatial (iso)      : {bs_sp:.4f}\n" if HAS_XGB_SPATIAL else "")
    + (f"  XGBoost-Spatial + Platt    : {bs_sp_platt:.4f}\n" if HAS_XGB_SPATIAL_CAL else "")
    + (f"  ConvLSTM                   : {bs_cl:.4f}\n" if HAS_CONVLSTM   else "")
    + (f"  Logistic Regression        : {bs_lr:.4f}\n" if HAS_LOGREG     else "")
    + (f"  Random Forest              : {bs_rf:.4f}\n" if HAS_RF         else "")
    + "\n── Brier Skill Score (BSS, dry class, ref = climatology) ──\n"
    + f"  Persistence              : {bss_pers:.4f}  (95% CI {fmt_ci(bss_ci_pers)})\n"
    + f"  SPI-1 threshold          : {bss_thr:.4f}  (95% CI {fmt_ci(bss_ci_thr)})\n"
    + f"  XGBoost raw              : {bss_xgb:.4f}  (95% CI {fmt_ci(bss_ci_xgb)})\n"
    + f"  XGBoost + Isotonic       : {bss_xgb_iso:.4f}  (95% CI {fmt_ci(bss_ci_xgb_iso)})\n"
    + f"  XGBoost + Platt          : {bss_xgb_platt:.4f}  (95% CI {fmt_ci(bss_ci_xgb_platt)})\n"
    + (f"  XGBoost-Spatial (iso)    : {bss_sp:.4f}  (95% CI {fmt_ci(bss_ci_sp)})\n" if HAS_XGB_SPATIAL else "")
    + (f"  XGBoost-Spatial + Platt  : {bss_sp_platt:.4f}  (95% CI {fmt_ci(bss_ci_sp_platt)})\n" if HAS_XGB_SPATIAL_CAL else "")
    + (f"  ConvLSTM                 : {bss_cl:.4f}  (95% CI {fmt_ci(bss_ci_cl)})\n" if HAS_CONVLSTM   else "")
    + (f"  Logistic Regression      : {bss_lr:.4f}  (95% CI {fmt_ci(bss_ci_lr)})\n" if HAS_LOGREG     else "")
    + (f"  Random Forest            : {bss_rf:.4f}  (95% CI {fmt_ci(bss_ci_rf)})\n" if HAS_RF         else "")
    + "\n── Heidke Skill Score (HSS, 3-class, monthly dominant class) ──\n"
    + f"  Climatological           : {hss_clim:.4f}  (95% CI {fmt_ci(hss_ci_clim)})\n"
    + f"  Persistence              : {hss_pers:.4f}  (95% CI {fmt_ci(hss_ci_pers)})\n"
    + f"  SPI-1 threshold          : {hss_thr:.4f}  (95% CI {fmt_ci(hss_ci_thr)})\n"
    + f"  XGBoost (no spatial)     : {hss_xgb:.4f}  (95% CI {fmt_ci(hss_ci_xgb)})\n"
    + (f"  XGBoost-Spatial          : {hss_sp:.4f}  (95% CI {fmt_ci(hss_ci_sp)})\n" if HAS_XGB_SPATIAL else "")
    + (f"  ConvLSTM                 : {hss_cl:.4f}  (95% CI {fmt_ci(hss_ci_cl)})\n" if HAS_CONVLSTM   else "")
    + (f"  Logistic Regression      : {hss_lr:.4f}  (95% CI {fmt_ci(hss_ci_lr)})\n" if HAS_LOGREG     else "")
    + (f"  Random Forest            : {hss_rf:.4f}  (95% CI {fmt_ci(hss_ci_rf)})\n" if HAS_RF         else "")
    + "\n── ROC-AUC (dry vs. not-dry, monthly mean probability) ──\n"
    + f"  Persistence              : {auc_pers:.4f}\n"
    + f"  SPI-1 threshold          : {auc_thr:.4f}\n"
    + f"  XGBoost (no spatial)     : {auc_xgb:.4f}\n"
    + (f"  XGBoost-Spatial          : {auc_sp:.4f}\n" if HAS_XGB_SPATIAL else "")
    + (f"  ConvLSTM                 : {auc_cl:.4f}\n" if HAS_CONVLSTM   else "")
    + (f"  Logistic Regression      : {auc_lr:.4f}\n" if HAS_LOGREG     else "")
    + (f"  Random Forest            : {auc_rf:.4f}\n" if HAS_RF         else "")
    + "\n── Brier Score Decomposition (Murphy 1973) ──\n"
    + "  Format: reliability / resolution / uncertainty  (BS = rel - res + unc)\n"
    + f"  Climatological  : {decomp_clim['reliability']:.4f} / {decomp_clim['resolution']:.4f} / {decomp_clim['uncertainty']:.4f}\n"
    + f"  XGBoost raw     : {decomp_xgb['reliability']:.4f} / {decomp_xgb['resolution']:.4f} / {decomp_xgb['uncertainty']:.4f}\n"
    + f"  XGBoost Isotonic: {decomp_xgb_iso['reliability']:.4f} / {decomp_xgb_iso['resolution']:.4f} / {decomp_xgb_iso['uncertainty']:.4f}\n"
    + f"  XGBoost Platt   : {decomp_xgb_platt['reliability']:.4f} / {decomp_xgb_platt['resolution']:.4f} / {decomp_xgb_platt['uncertainty']:.4f}\n"
    + (f"  XGB-Spatial iso : {decomp_sp['reliability']:.4f} / {decomp_sp['resolution']:.4f} / {decomp_sp['uncertainty']:.4f}\n" if HAS_XGB_SPATIAL else "")
    + "\nOutputs:\n"
    + f"  {cm_path}\n"
    + f"  {rel_path}\n"
    + f"  {csv_path}\n"
    + f"  {cal_csv_path}\n"
    + f"  {decomp_csv_path}\n"
)
print(summary)
(OUT_DIR / "forecast_skill_scores.txt").write_text(summary)
print("Wrote:", OUT_DIR / "forecast_skill_scores.txt")
