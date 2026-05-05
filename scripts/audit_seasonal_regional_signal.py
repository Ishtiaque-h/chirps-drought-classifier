#!/usr/bin/env python
"""
Audit whether seasonal long-lead BSS comes from event tracking or calibration.

Positive BSS over climatology can arise from useful temporal discrimination, but
it can also arise from shrinking an overconfident/biased baseline toward the
test-period mean. This script reads the monthly score files produced by
run_seasonal_longlead_experiment.py and summarizes correlation, bias, variance,
and month-level Brier gains.

Outputs:
  results/seasonal/seasonal_regional_signal_audit.csv
  results/seasonal/seasonal_regional_signal_audit.md
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SUMMARY_PATH = PROJECT_ROOT / "results" / "seasonal" / "seasonal_regional_longlead_summary.csv"
OUT_DIR = PROJECT_ROOT / "results" / "seasonal"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def brier_score(pred: pd.Series, obs: pd.Series) -> float:
    return float(np.mean((pred.to_numpy(dtype=float) - obs.to_numpy(dtype=float)) ** 2))


def bss(score: float, reference: float) -> float:
    if reference == 0:
        return np.nan
    return float(1.0 - score / reference)


def corr(pred: pd.Series, obs: pd.Series) -> float:
    if pred.nunique(dropna=True) <= 1 or obs.nunique(dropna=True) <= 1:
        return np.nan
    return float(pred.corr(obs))


def variance_ratio(pred: pd.Series, obs: pd.Series) -> float:
    obs_std = float(obs.std(ddof=0))
    if obs_std == 0:
        return np.nan
    return float(pred.std(ddof=0) / obs_std)


def monthly_path_from_score(score_file: str) -> Path:
    path = PROJECT_ROOT / score_file
    return path.with_name(path.name.replace("_experiment_scores.txt", "_monthly_scores.csv"))


def classify_signal(row: pd.Series) -> str:
    if row["bss_iso_vs_clim"] > 0 and row["corr_iso_obs"] < 0.1 and row["std_ratio_iso_obs"] < 0.2:
        return "positive_calibration_shift"
    if row["bss_iso_vs_clim"] > 0 and row["corr_iso_obs"] >= 0.2:
        return "positive_temporal_tracking"
    if row["bss_iso_vs_clim"] > 0:
        return "positive_weak_tracking"
    if row["bss_iso_vs_clim"] < 0 and row["corr_raw_obs"] > row["corr_iso_obs"] + 0.1:
        return "raw_signal_lost_after_calibration"
    return "no_positive_signal"


def audit_row(row: pd.Series) -> dict[str, object] | None:
    monthly_path = monthly_path_from_score(str(row["score_file"]))
    if not monthly_path.exists():
        return None

    df = pd.read_csv(monthly_path, parse_dates=["target_time"])
    obs = df["y_true_dry_frac"]
    clim = df["clim_prob_dry"]
    raw = df["xgb_prob_dry"]
    iso = df["xgb_cal_prob_dry"]
    pers = df["persistence_prob_dry"]

    bs_clim = brier_score(clim, obs)
    bs_raw = brier_score(raw, obs)
    bs_iso = brier_score(iso, obs)
    bs_pers = brier_score(pers, obs)
    iso_gain = (clim - obs) ** 2 - (iso - obs) ** 2

    out: dict[str, object] = {
        "region": row["region"],
        "target_spi": int(row["target_spi"]),
        "lead_months": int(row["lead_months"]),
        "climate_features": row["climate_features"],
        "n_months": len(df),
        "target_start": df["target_time"].min().date().isoformat(),
        "target_end": df["target_time"].max().date().isoformat(),
        "obs_mean": float(obs.mean()),
        "obs_std": float(obs.std(ddof=0)),
        "clim_mean": float(clim.mean()),
        "raw_mean": float(raw.mean()),
        "iso_mean": float(iso.mean()),
        "clim_bias": float((clim - obs).mean()),
        "raw_bias": float((raw - obs).mean()),
        "iso_bias": float((iso - obs).mean()),
        "bs_clim": bs_clim,
        "bs_persistence": bs_pers,
        "bs_raw": bs_raw,
        "bs_iso": bs_iso,
        "bss_persistence_vs_clim": bss(bs_pers, bs_clim),
        "bss_raw_vs_clim": bss(bs_raw, bs_clim),
        "bss_iso_vs_clim": bss(bs_iso, bs_clim),
        "corr_clim_obs": corr(clim, obs),
        "corr_raw_obs": corr(raw, obs),
        "corr_iso_obs": corr(iso, obs),
        "std_ratio_clim_obs": variance_ratio(clim, obs),
        "std_ratio_raw_obs": variance_ratio(raw, obs),
        "std_ratio_iso_obs": variance_ratio(iso, obs),
        "iso_beats_clim_months": int((iso_gain > 0).sum()),
        "iso_beats_clim_fraction": float((iso_gain > 0).mean()),
        "iso_gain_sum": float(iso_gain.sum()),
        "robust_status": row.get("status", ""),
        "monthly_score_file": str(monthly_path.relative_to(PROJECT_ROOT)),
    }
    out["signal_flag"] = classify_signal(pd.Series(out))
    return out


def markdown_table(df: pd.DataFrame) -> str:
    headers = list(df.columns)
    rows = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for _, row in df.iterrows():
        values = []
        for value in row:
            if isinstance(value, float):
                values.append(f"{value:.3f}")
            else:
                values.append(str(value))
        rows.append("| " + " | ".join(values) + " |")
    return "\n".join(rows)


def main() -> None:
    if not SUMMARY_PATH.exists():
        raise SystemExit(f"Missing {SUMMARY_PATH}. Run build_seasonal_regional_summary.py first.")

    summary = pd.read_csv(SUMMARY_PATH)
    rows = [out for _, row in summary.iterrows() if (out := audit_row(row)) is not None]
    if not rows:
        raise SystemExit("No monthly score files found for seasonal regional audit.")

    audit = pd.DataFrame(rows).sort_values(["region", "target_spi", "lead_months", "climate_features"])
    csv_path = OUT_DIR / "seasonal_regional_signal_audit.csv"
    md_path = OUT_DIR / "seasonal_regional_signal_audit.md"
    audit.to_csv(csv_path, index=False)

    display_cols = [
        "region",
        "target_spi",
        "lead_months",
        "climate_features",
        "bss_iso_vs_clim",
        "corr_iso_obs",
        "std_ratio_iso_obs",
        "iso_bias",
        "iso_beats_clim_fraction",
        "robust_status",
        "signal_flag",
    ]
    md = [
        "# Seasonal Regional Signal Audit",
        "",
        "Positive BSS is interpreted alongside correlation, variance ratio, and bias to separate temporal tracking from calibration shifts.",
        "",
        markdown_table(audit[display_cols].round(4)),
        "",
        "Source monthly score files are listed in the CSV.",
    ]
    md_path.write_text("\n".join(md) + "\n")
    print(f"Wrote {csv_path}")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
