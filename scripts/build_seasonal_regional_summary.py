#!/usr/bin/env python
"""
Collect seasonal long-lead experiment score files into one comparison table.

The seasonal runner can now produce Central Valley and multi-region outputs
under both historical filenames and climate-feature-suffixed filenames. This
script parses the score files plus probability metadata so the paper narrative
does not depend on manually copied BSS numbers.

Outputs:
  results/seasonal/seasonal_regional_longlead_summary.csv
  results/seasonal/seasonal_regional_longlead_summary.md
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = PROJECT_ROOT / "outputs"
RESULTS = PROJECT_ROOT / "results" / "seasonal"
RESULTS.mkdir(parents=True, exist_ok=True)


SCORE_PATTERN = re.compile(
    r"seasonal_spi(?P<target_spi>\d+)_lead(?P<lead>\d+)"
    r"(?P<climate_suffix>_climate_(?P<climate_features>[A-Za-z0-9_]+))?"
    r"_experiment_scores\.txt$"
)


def parse_float_triplet(line: str) -> tuple[float, float, float] | None:
    match = re.search(
        r":\s*(?P<bss>[-+0-9.]+)\s*\(95% CI \[(?P<lo>[-+0-9.]+),\s*(?P<hi>[-+0-9.]+)\]\)",
        line,
    )
    if not match:
        return None
    return float(match.group("bss")), float(match.group("lo")), float(match.group("hi"))


def infer_region(path: Path) -> str:
    parts = path.relative_to(OUTPUTS).parts
    if len(parts) >= 3 and parts[0] == "seasonal":
        return parts[1]
    return "cvalley"


def infer_climate_from_probs(probs_path: Path, fallback: str | None) -> str:
    if fallback:
        return fallback
    if not probs_path.exists():
        return "unknown"
    z = np.load(probs_path, allow_pickle=True)
    features = set(str(f) for f in z.get("features", []))
    has_nino = {"nino34_lag1", "nino34_lag2"}.issubset(features)
    has_pdo = {"pdo_lag1", "pdo_lag2"}.issubset(features)
    if has_nino and has_pdo:
        return "all"
    if has_nino:
        return "nino34"
    if has_pdo:
        return "pdo"
    return "none"


def metadata_from_probs(probs_path: Path) -> dict[str, object]:
    if not probs_path.exists():
        return {}
    z = np.load(probs_path, allow_pickle=True)
    out: dict[str, object] = {}
    if "target_times" in z.files:
        target_times = pd.to_datetime(z["target_times"])
        out["target_time_start"] = target_times.min().date().isoformat()
        out["target_time_end"] = target_times.max().date().isoformat()
        out["n_test_months_from_probs"] = int(len(pd.unique(target_times)))
    if "best_iteration" in z.files:
        out["best_iteration"] = int(z["best_iteration"])
    if "features" in z.files:
        out["n_features"] = int(len(z["features"]))
        out["features"] = ", ".join(str(f) for f in z["features"])
    return out


def parse_score_file(path: Path) -> dict[str, object] | None:
    match = SCORE_PATTERN.search(path.name)
    if not match:
        return None

    prefix = path.name.replace("_experiment_scores.txt", "")
    probs_path = path.with_name(prefix + "_xgb_test_probs.npz")
    text = path.read_text()

    row: dict[str, object] = {
        "region": infer_region(path),
        "score_file": str(path.relative_to(PROJECT_ROOT)),
        "target_spi": int(match.group("target_spi")),
        "lead_months": int(match.group("lead")),
        "climate_features": infer_climate_from_probs(probs_path, match.group("climate_features")),
    }
    row.update(metadata_from_probs(probs_path))

    for line in text.splitlines():
        if line.startswith("Region:"):
            row["region_label"] = line.split(":", 1)[1].strip()
        elif line.startswith("Mask kind:"):
            row["mask_kind"] = line.split(":", 1)[1].strip()
        elif line.startswith("Climate features:"):
            row["climate_features"] = line.split(":", 1)[1].strip()
        elif line.startswith("Test months:"):
            row["n_test_months"] = int(line.split(":", 1)[1].strip())
        elif "  Climatology" in line:
            row["bs_climatology"] = float(line.split(":", 1)[1].strip())
        elif "  Persistence SPI-" in line and "95% CI" not in line:
            row["bs_persistence"] = float(line.split(":", 1)[1].strip())
        elif "  XGBoost           :" in line and "95% CI" not in line:
            row["bs_xgb_raw"] = float(line.split(":", 1)[1].strip())
        elif "  XGBoost isotonic" in line and "95% CI" not in line:
            row["bs_xgb_isotonic"] = float(line.split(":", 1)[1].strip())
        elif "  Persistence SPI-" in line and "95% CI" in line:
            parsed = parse_float_triplet(line)
            if parsed:
                row["bss_persistence"], row["bss_persistence_ci_low"], row["bss_persistence_ci_high"] = parsed
        elif "  XGBoost           :" in line and "95% CI" in line:
            parsed = parse_float_triplet(line)
            if parsed:
                row["bss_xgb_raw"], row["bss_xgb_raw_ci_low"], row["bss_xgb_raw_ci_high"] = parsed
        elif "  XGBoost isotonic" in line and "95% CI" in line:
            parsed = parse_float_triplet(line)
            if parsed:
                row["bss_xgb_isotonic"], row["bss_xgb_isotonic_ci_low"], row["bss_xgb_isotonic_ci_high"] = parsed
    return row


def status(row: pd.Series) -> str:
    bss = row.get("bss_xgb_isotonic")
    lo = row.get("bss_xgb_isotonic_ci_low")
    hi = row.get("bss_xgb_isotonic_ci_high")
    if pd.isna(bss):
        return "missing"
    if lo > 0:
        return "robust_positive"
    if hi < 0:
        return "robust_negative"
    if bss > 0:
        return "positive_uncertain"
    return "negative_uncertain"


def markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return ""
    headers = list(df.columns)
    rows = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for _, row in df.iterrows():
        vals = []
        for value in row:
            if isinstance(value, float):
                vals.append(f"{value:.3f}")
            else:
                vals.append(str(value))
        rows.append("| " + " | ".join(vals) + " |")
    return "\n".join(rows)


def main() -> None:
    score_files = sorted(OUTPUTS.glob("seasonal_spi*_lead*_experiment_scores.txt"))
    score_files.extend(sorted((OUTPUTS / "seasonal").glob("*/*_experiment_scores.txt")))

    rows = [row for path in score_files if (row := parse_score_file(path)) is not None]
    df = pd.DataFrame(rows)
    if df.empty:
        raise SystemExit("No seasonal score files found.")
    df["status"] = df.apply(status, axis=1)
    df = df.sort_values(["region", "target_spi", "lead_months", "climate_features"])

    csv_path = RESULTS / "seasonal_regional_longlead_summary.csv"
    md_path = RESULTS / "seasonal_regional_longlead_summary.md"
    df.to_csv(csv_path, index=False)

    display_cols = [
        "region",
        "target_spi",
        "lead_months",
        "climate_features",
        "n_test_months",
        "bss_xgb_isotonic",
        "bss_xgb_isotonic_ci_low",
        "bss_xgb_isotonic_ci_high",
        "status",
        "target_time_start",
        "target_time_end",
    ]
    md = [
        "# Seasonal Regional Long-Lead Summary",
        "",
        markdown_table(df[[c for c in display_cols if c in df.columns]].round(4)),
        "",
        "Source files are listed in the CSV.",
    ]
    md_path.write_text("\n".join(md) + "\n")
    print(f"Wrote {csv_path}")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
