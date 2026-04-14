#!/usr/bin/env python
"""
Qualitative consistency check: forecast model vs. US Drought Monitor (USDM).

NOTE — methodological framing:
  USDM D1+ is a composite index that integrates soil moisture, streamflow,
  Palmer PDSI, and expert observer reports.  It is NOT equivalent to
  SPI-1 ≤ -1 from CHIRPS precipitation alone.  A Pearson correlation between
  the model's dry fraction and USDM D1+ area is therefore *not* a validation
  metric — it is confounded by data-source and methodology differences.

  This script is retained as a **qualitative plausibility check** only.
  The primary quantitative external validation uses ERA5-Land SPI-1
  (see validate_era5_spi.py), because both CHIRPS and ERA5-Land are
  precipitation-derived products, making skill scores directly comparable.

The USDM provides weekly county-level drought statistics for the contiguous US.
This script downloads the county-level time series for the Central Valley
counties directly from the USDM public API, aggregates to monthly, and
overlays the model's regional dry-fraction predictions over 2021–2026.

Central Valley counties (FIPS):
  Fresno      06019
  Kern        06029
  Kings       06031
  Madera      06039
  Merced      06047
  San Joaquin 06077
  Stanislaus  06099
  Tulare      06107

USDM drought categories (area percentage):
  None (no drought), D0 (abnormally dry), D1–D4 (moderate–exceptional drought)
  D1+ is mapped to "dry" for comparison with model's dry class.

Inputs:
  data/processed/dataset_forecast.parquet
  outputs/forecast_xgb_model.json

Outputs:
  outputs/usdm_consistency.png
  outputs/usdm_consistency_notes.txt
"""
from pathlib import Path
import io
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
import urllib.error
import urllib.parse
import urllib.request

BASE_DIR = Path(__file__).resolve().parents[1]
OUT_DIR = BASE_DIR / "outputs"; OUT_DIR.mkdir(exist_ok=True)
DATA    = BASE_DIR / "data/processed/dataset_forecast.parquet"
MODEL   = BASE_DIR / "outputs/forecast_xgb_model.json"

FEATURES = [
    "spi1_lag1", "spi1_lag2", "spi1_lag3",
    "spi3_lag1", "spi6_lag1",
    "pr_lag1", "pr_lag2", "pr_lag3",
    "month_sin", "month_cos",
]
TARGET = "target_label"
label_map = {-1: 0, 0: 1, 1: 2}

# ---- Central Valley county FIPS codes ----
CV_FIPS = {
    "Fresno":       "06019",
    "Kern":         "06029",
    "Kings":        "06031",
    "Madera":       "06039",
    "Merced":       "06047",
    "San Joaquin":  "06077",
    "Stanislaus":   "06099",
    "Tulare":       "06107",
}

# -----------------------------------------------------------------------
# 1. Download USDM county statistics from USDM public REST API
# -----------------------------------------------------------------------
def fetch_usdm_county(fips: str) -> pd.DataFrame:
    """
    Fetch weekly USDM statistics for one county via the USDM public data API.
    Returns a DataFrame with columns: date, None, D0, D1, D2, D3, D4
    where values are the percent area in each category.
    """
    params = urllib.parse.urlencode({
        "aoi": fips,
        "StartDate": "2021-01-01T00:00:00Z",
        "EndDate": "2026-03-01T00:00:00Z",
        "statisticsType": 1,
    })
    url = (
        "https://usdmdataservices.unl.edu/api/CountyStatistics/GetDroughtSeverityStatisticsByAreaPercent"
        f"?{params}"
    )
    print(f"    URL: {url}")
    try:
        with urllib.request.urlopen(url, timeout=30) as resp:
            status = getattr(resp, "status", None)
            content_type = resp.headers.get("Content-Type", "")
            raw_bytes = resp.read()
        print(f"    Status: {status}")
        print(f"    Content-Type: {content_type}")

        if status != 200:
            print(f"  Warning: skipping FIPS {fips} due to HTTP status {status}")
            return pd.DataFrame()

        raw = raw_bytes.decode("utf-8", errors="replace")
        content_type_l = content_type.lower()
        try:
            if "json" in content_type_l:
                df = pd.read_json(io.StringIO(raw))
            elif "csv" in content_type_l:
                df = pd.read_csv(io.StringIO(raw))
            else:
                print(
                    f"  Warning: skipping FIPS {fips} due to unsupported content type: {content_type}"
                )
                return pd.DataFrame()
        except ValueError as e:
            preview = raw[:200].replace("\n", " ")
            print(f"  Warning: parse failed for FIPS {fips}: {e}")
            print(f"  Body preview (first 200 chars): {preview}")
            return pd.DataFrame()

        if "releaseDate" in df.columns:
            df["date"] = pd.to_datetime(df["releaseDate"])
        elif "MapDate" in df.columns:
            df["date"] = pd.to_datetime(df["MapDate"], format="%Y%m%d", errors="coerce")
        elif "ValidStart" in df.columns:
            df["date"] = pd.to_datetime(df["ValidStart"], errors="coerce")
        else:
            print(
                f"  Warning: no recognized date column for FIPS {fips}. "
                f"Columns: {list(df.columns)}"
            )
            return pd.DataFrame()

        df = df.dropna(subset=["date"])
        return df
    except urllib.error.HTTPError as e:
        content_type = e.headers.get("Content-Type", "") if e.headers else ""
        body_preview = ""
        try:
            body_preview = e.read().decode("utf-8", errors="replace")[:200].replace("\n", " ")
        except Exception:
            body_preview = "<unavailable>"
        print(f"    Status: {e.code}")
        print(f"    Content-Type: {content_type}")
        print(f"  Warning: skipping FIPS {fips} due to HTTP status {e.code}")
        print(f"  Body preview (first 200 chars): {body_preview}")
        return pd.DataFrame()
    except Exception as e:
        print(f"  Warning: could not fetch FIPS {fips}: {e}")
        return pd.DataFrame()


print("Downloading USDM data for Central Valley counties...")
frames = []
for name, fips in CV_FIPS.items():
    print(f"  {name} ({fips})")
    df_c = fetch_usdm_county(fips)
    if not df_c.empty:
        df_c["county"] = name
        frames.append(df_c)

if not frames:
    print("No USDM data could be downloaded (network may be unavailable). "
          "Skipping USDM validation. Please run this script with internet access.")
    import sys; sys.exit(0)

usdm_raw = pd.concat(frames, ignore_index=True)
print(f"Downloaded {len(usdm_raw)} rows across {usdm_raw['county'].nunique()} counties.")

# -----------------------------------------------------------------------
# 2. Aggregate to monthly: mean area fraction across all counties
#    then pick the worst (D1+ extent = D1+D2+D3+D4) per month
# -----------------------------------------------------------------------
# Column names may vary; identify them robustly
d_cols = [c for c in usdm_raw.columns if c.upper() in {"D1", "D2", "D3", "D4"}]
none_col = [c for c in usdm_raw.columns if c.upper() in {"NONE", "D-1", "NO DROUGHT"}]

if not d_cols:
    print(
        f"Could not identify D1–D4 columns in USDM response. "
        f"Available columns: {list(usdm_raw.columns)}. "
        "Skipping USDM consistency check."
    )
    import sys; sys.exit(0)

usdm_raw["month"] = usdm_raw["date"].dt.to_period("M")
# Average across counties per week
weekly_cv = usdm_raw.groupby("date")[d_cols].mean()
if none_col:
    weekly_cv[none_col[0]] = usdm_raw.groupby("date")[none_col[0]].mean()

# Resample to monthly (mean)
weekly_cv.index = pd.to_datetime(weekly_cv.index)
monthly_usdm = weekly_cv.resample("MS").mean()

# Drought class columns from this endpoint are cumulative in practice
# (D1 means D1+, D2 means D2+, etc.), so D1+ is the D1 column itself.
if "D1" in monthly_usdm.columns:
    monthly_usdm["d1plus_frac"] = monthly_usdm["D1"] / 100.0
else:
    # Fallback for unexpected schemas where classes are mutually exclusive.
    monthly_usdm["d1plus_frac"] = monthly_usdm[d_cols].sum(axis=1) / 100.0

print(f"USDM monthly records: {len(monthly_usdm)}")

# -----------------------------------------------------------------------
# 3. Load model and get test-set predictions (2021+)
# -----------------------------------------------------------------------
print("Loading forecast model and dataset...")
df = pd.read_parquet(DATA)
df["year"] = df["year"].astype(int)
test = df[df["year"] >= 2021].copy()

model = xgb.Booster()
model.load_model(MODEL.as_posix())

dtest = xgb.DMatrix(test[FEATURES], feature_names=FEATURES)
probs = model.predict(dtest)
test["pred_label"] = probs.argmax(axis=1)
# map back: 0->-1(dry), 1->0(normal), 2->1(wet)
inv_map = {0: -1, 1: 0, 2: 1}
test["pred_label"] = test["pred_label"].map(inv_map)

# Monthly dry fraction from model
test["month_dt"] = pd.to_datetime(test["time"]) + pd.DateOffset(months=1)
test["month_dt"] = test["month_dt"].dt.to_period("M").dt.to_timestamp()
monthly_model = (
    test.assign(is_dry=(test["pred_label"] == -1).astype(float))
    .groupby("month_dt")["is_dry"]
    .mean()
    .rename("model_dry_frac")
)

# -----------------------------------------------------------------------
# 4. Align and compare
# -----------------------------------------------------------------------
monthly_usdm.index = monthly_usdm.index.to_period("M").to_timestamp()
merged = pd.DataFrame({
    "model_dry_frac": monthly_model,
    "usdm_d1plus_frac": monthly_usdm["d1plus_frac"],
}).dropna()

if len(merged) < 5:
    print("Insufficient overlapping data for comparison. Check USDM download.")
    import sys; sys.exit(0)

corr = merged["model_dry_frac"].corr(merged["usdm_d1plus_frac"])
print(f"Pearson correlation (model dry frac vs USDM D1+ frac): {corr:.3f}")
print(f"Comparison months: {len(merged)}")
print("NOTE: This correlation is a qualitative plausibility indicator only.")
print("      USDM integrates soil moisture/streamflow/observer reports; it is")
print("      not directly comparable to SPI-1 from precipitation alone.")

# -----------------------------------------------------------------------
# 5. Plot (qualitative consistency check)
# -----------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(merged.index, merged["model_dry_frac"],   label="Model — dry fraction (SPI-1 based)",
        linewidth=1.8)
ax.plot(merged.index, merged["usdm_d1plus_frac"], label="USDM — D1+ area fraction (composite index)",
        linewidth=1.8, linestyle="--")
ax.set_xlabel("Month")
ax.set_ylabel("Fraction of region")
ax.set_title(
    "Qualitative consistency check: Model dry fraction vs. USDM D1+ extent\n"
    "Central Valley 2021–2026 (different data sources — not a skill validation)",
    fontsize=10,
)
ax.legend()
ax.set_ylim(0, 1)
fig.tight_layout()
fig.savefig(OUT_DIR / "usdm_consistency.png", dpi=150, bbox_inches="tight")
plt.close()

metrics_txt = (
    f"USDM Consistency Check — Central Valley 2021–2026\n"
    f"{'='*55}\n"
    f"METHODOLOGICAL NOTE:\n"
    f"  This is a qualitative plausibility check, NOT a skill validation.\n"
    f"  USDM D1+ integrates soil moisture, streamflow, Palmer PDSI, and\n"
    f"  expert reports — fundamentally different from SPI-1 (precipitation only).\n"
    f"  Any correlation is confounded by data-source differences.\n"
    f"  For quantitative validation use validate_era5_spi.py (ERA5-Land SPI-1).\n"
    f"{'='*55}\n"
    f"Comparison months : {len(merged)}\n"
    f"Pearson r         : {corr:.4f}\n"
    f"Model dry frac    : mean={merged['model_dry_frac'].mean():.3f}  "
    f"std={merged['model_dry_frac'].std():.3f}\n"
    f"USDM D1+ frac     : mean={merged['usdm_d1plus_frac'].mean():.3f}  "
    f"std={merged['usdm_d1plus_frac'].std():.3f}\n"
)
print(metrics_txt)
(OUT_DIR / "usdm_consistency_notes.txt").write_text(metrics_txt)
print("Wrote:", OUT_DIR / "usdm_consistency.png")
print("Wrote:", OUT_DIR / "usdm_consistency_notes.txt")
