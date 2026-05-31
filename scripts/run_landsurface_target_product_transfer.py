#!/usr/bin/env python
"""Cross-product transfer diagnostic for land-surface drought probabilities.

This script tests a deliberately hard question:

    Do probabilities calibrated on one root-zone soil-moisture target product
    retain value when scored against an independent SMAP L4 target, without
    recalibrating on SMAP?

It uses completed monthly score files only. No model is trained here. The
diagnostic merges frozen ERA5-Land- or GLDAS-calibrated probabilities onto the
SMAP L4 multi-snapshot target months and scores them against SMAP climatology
and same-target SMAP persistence. A SMAP self-calibrated row is included only
as an upper-reference, not as transfer evidence.
"""

from __future__ import annotations

from argparse import ArgumentParser, Namespace
from pathlib import Path
import shutil

import numpy as np
import pandas as pd

from run_gefsv12_landsurface_stack_benchmark import (
    added_value_status,
    bootstrap_delta_bs,
)
from run_landsurface_forecast_benchmark import brier, bss, bootstrap_bss


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = PROJECT_ROOT / "outputs"
REPORT_DIR = PROJECT_ROOT / "results" / "report"
LANDSURFACE_DIR = REPORT_DIR / "landsurface"

DEFAULT_TARGET = (
    LANDSURFACE_DIR
    / "landsurface_smap_l4_multi3snapshot_gefsv12_validation_monthly_scores.csv"
)

SOURCE_FILES = {
    "era5_land": LANDSURFACE_DIR
    / "landsurface_gefsv12_rzsm_stack_day15_hindcastcal_monthly_scores.csv",
    "gldas_noah": LANDSURFACE_DIR
    / "landsurface_gldas_gefsv12_validation_monthly_scores.csv",
    "smap_l4_multi3snapshot": DEFAULT_TARGET,
}

MODEL_COLUMNS = {
    "gefs_selected": "gefs_selected_prob_dry",
    "source_persistence_selected": "persistence_selected_prob_dry",
    "stack_equal_weight": "stack_equal_prob_dry",
    "stack_validation_selected": "stack_validation_selected_prob_dry",
}

SELF_MODEL_RENAMES = {
    "source_persistence_selected": "target_persistence_selected",
}


def parse_args() -> Namespace:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sources",
        nargs="+",
        default=["era5_land", "gldas_noah"],
        choices=sorted(SOURCE_FILES),
        help="Calibration-source products to transfer to SMAP.",
    )
    parser.add_argument(
        "--include-smap-self-reference",
        action="store_true",
        help="Include SMAP-calibrated model rows as an upper-reference.",
    )
    parser.add_argument("--target-file", type=Path, default=DEFAULT_TARGET)
    parser.add_argument(
        "--out-prefix",
        default="landsurface_target_product_transfer_to_smap_multi3snapshot",
    )
    parser.add_argument("--n-bootstrap", type=int, default=2000)
    parser.add_argument("--copy-report", action="store_true")
    return parser.parse_args()


def claim_status(bss_value: float, ci_low: float, ci_high: float) -> str:
    if not np.isfinite(bss_value):
        return "not_applicable"
    if np.isfinite(ci_low) and np.isfinite(ci_high):
        if ci_low > 0:
            return "robust_positive"
        if ci_high < 0:
            return "robust_negative"
        if bss_value > 0:
            return "positive_uncertain"
        return "negative_uncertain"
    if bss_value > 0:
        return "positive_no_ci"
    if bss_value < 0:
        return "negative_no_ci"
    return "reference"


def transfer_status(skill: str, added_value: str, transfer_kind: str) -> str:
    if transfer_kind == "target_self_calibrated_reference":
        return "self_calibrated_reference"
    if skill == "robust_positive" and added_value == "stack_robust_added_value":
        return "transfers_with_robust_added_value"
    if skill == "robust_positive":
        return "transfers_vs_climatology_only"
    if skill == "positive_uncertain":
        return "weak_or_sample_limited_transfer"
    if skill == "robust_negative":
        return "does_not_transfer"
    return "inconclusive_or_negative_transfer"


def read_target(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"SMAP target monthly score file not found: {path}")
    target = pd.read_csv(path, parse_dates=["target_time"])
    required = {
        "region",
        "target_time",
        "target_year",
        "target_month",
        "y_true_dry_frac",
        "clim_prob_dry",
        "persistence_selected_prob_dry",
    }
    missing = required.difference(target.columns)
    if missing:
        raise ValueError(f"SMAP target file is missing required columns: {sorted(missing)}")
    return target


def read_source(source_product: str) -> pd.DataFrame:
    path = SOURCE_FILES[source_product]
    if not path.exists():
        raise FileNotFoundError(f"Source monthly score file not found for {source_product}: {path}")
    source = pd.read_csv(path, parse_dates=["target_time"])
    if "region" not in source.columns or "target_time" not in source.columns:
        raise ValueError(f"Source file lacks region/target_time columns: {path}")
    source["source_product"] = source_product
    source["source_file"] = str(path.relative_to(PROJECT_ROOT))
    return source


def available_model_columns(source: pd.DataFrame) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    for model, column in MODEL_COLUMNS.items():
        if column in source.columns:
            out.append((model, column))
    if not out:
        raise ValueError("No recognized probability columns found in source file.")
    return out


def build_monthly_rows(
    target: pd.DataFrame,
    source: pd.DataFrame,
    source_product: str,
    target_file: Path,
    transfer_kind: str,
) -> pd.DataFrame:
    target_cols = [
        "region",
        "target_time",
        "target_year",
        "target_month",
        "y_true_dry_frac",
        "clim_prob_dry",
        "persistence_selected_prob_dry",
    ]
    target_small = target[target_cols].rename(
        columns={
            "clim_prob_dry": "smap_clim_prob_dry",
            "persistence_selected_prob_dry": "smap_persistence_selected_prob_dry",
        }
    )

    rows: list[pd.DataFrame] = []
    for model, column in available_model_columns(source):
        source_small = source[
            ["region", "target_time", "source_file", column]
        ].rename(columns={column: "transferred_prob_dry"})
        merged = target_small.merge(source_small, on=["region", "target_time"], how="inner")
        if merged.empty:
            continue
        output_model = SELF_MODEL_RENAMES.get(model, model) if transfer_kind.startswith("target_") else model
        merged["model"] = output_model
        merged["source_product"] = source_product
        merged["target_product"] = "smap_l4_multi3snapshot"
        merged["target_file"] = str(target_file.relative_to(PROJECT_ROOT))
        merged["transfer_kind"] = transfer_kind
        rows.append(merged)

    if not rows:
        return pd.DataFrame()
    out = pd.concat(rows, ignore_index=True)
    out = out.dropna(
        subset=[
            "y_true_dry_frac",
            "smap_clim_prob_dry",
            "smap_persistence_selected_prob_dry",
            "transferred_prob_dry",
        ]
    )
    return out.sort_values(["source_product", "region", "model", "target_time"]).reset_index(drop=True)


def score_group(group: pd.DataFrame, n_bootstrap: int, seed: int) -> dict[str, object]:
    y = group["y_true_dry_frac"].to_numpy(dtype=float)
    pred = group["transferred_prob_dry"].to_numpy(dtype=float)
    clim = group["smap_clim_prob_dry"].to_numpy(dtype=float)
    persistence = group["smap_persistence_selected_prob_dry"].to_numpy(dtype=float)

    scoring = group.rename(
        columns={
            "smap_clim_prob_dry": "clim_prob_dry",
            "smap_persistence_selected_prob_dry": "persistence_selected_prob_dry",
        }
    ).copy()

    bss_clim = bss(y, pred, clim)
    bss_clim_lo, bss_clim_hi = bootstrap_bss(
        scoring,
        pred_col="transferred_prob_dry",
        ref_col="clim_prob_dry",
        n_bootstrap=n_bootstrap,
        seed=seed,
    )
    bss_persistence = bss(y, pred, persistence)
    bss_pers_lo, bss_pers_hi = bootstrap_bss(
        scoring,
        pred_col="transferred_prob_dry",
        ref_col="persistence_selected_prob_dry",
        n_bootstrap=n_bootstrap,
        seed=seed + 100_000,
    )
    delta_bs = brier(y, pred) - brier(y, persistence)
    delta_lo, delta_hi = bootstrap_delta_bs(
        scoring,
        candidate_col="transferred_prob_dry",
        reference_col="persistence_selected_prob_dry",
        n_bootstrap=n_bootstrap,
        seed=seed + 200_000,
    )

    skill = claim_status(bss_clim, bss_clim_lo, bss_clim_hi)
    added = added_value_status(delta_bs, delta_lo, delta_hi)
    model_std = float(np.std(pred, ddof=0))
    obs_std = float(np.std(y, ddof=0))
    return {
        "n_months": int(len(group)),
        "bs_smap_climatology": brier(y, clim),
        "bs_smap_persistence_selected": brier(y, persistence),
        "bs_model": brier(y, pred),
        "bss_vs_smap_climatology": bss_clim,
        "bss_vs_smap_climatology_ci_low": bss_clim_lo,
        "bss_vs_smap_climatology_ci_high": bss_clim_hi,
        "bss_vs_smap_persistence_selected": bss_persistence,
        "bss_vs_smap_persistence_selected_ci_low": bss_pers_lo,
        "bss_vs_smap_persistence_selected_ci_high": bss_pers_hi,
        "delta_bs_model_minus_smap_persistence_selected": delta_bs,
        "delta_bs_ci_low": delta_lo,
        "delta_bs_ci_high": delta_hi,
        "claim_status_vs_smap_climatology": skill,
        "added_value_status_vs_smap_persistence": added,
        "transfer_status": transfer_status(
            skill,
            added,
            str(group["transfer_kind"].iloc[0]),
        ),
        "spearman_model_vs_smap_observed": group["transferred_prob_dry"].corr(
            group["y_true_dry_frac"], method="spearman"
        ),
        "amplitude_ratio_model_vs_smap_observed": model_std / obs_std if obs_std > 0 else np.nan,
    }


def score_monthly(monthly: pd.DataFrame, n_bootstrap: int) -> pd.DataFrame:
    if monthly.empty:
        return monthly
    rows: list[dict[str, object]] = []
    group_cols = ["source_product", "target_product", "region", "model", "transfer_kind"]
    for i, (keys, group) in enumerate(monthly.groupby(group_cols, sort=True), start=1):
        row = dict(zip(group_cols, keys, strict=True))
        row.update(score_group(group, n_bootstrap=n_bootstrap, seed=9100 + i * 37))
        row["source_file"] = group["source_file"].iloc[0]
        row["target_file"] = group["target_file"].iloc[0]
        rows.append(row)
    return pd.DataFrame(rows).sort_values(
        ["transfer_kind", "source_product", "region", "model"]
    ).reset_index(drop=True)


def write_summary_text(summary: pd.DataFrame, path: Path) -> None:
    lines = [
        "Land-surface target-product transfer to SMAP L4 multi-snapshot target",
        "",
        "Design: source-product probabilities are frozen before scoring against SMAP. "
        "The SMAP self-calibrated rows, when present, are references only.",
        "",
    ]
    transfer = summary.loc[summary["transfer_kind"].eq("cross_product_transfer")]
    if not transfer.empty:
        robust = int(transfer["claim_status_vs_smap_climatology"].eq("robust_positive").sum())
        added = int(
            transfer["added_value_status_vs_smap_persistence"].eq("stack_robust_added_value").sum()
        )
        lines.append(
            f"Cross-product transfer rows: {len(transfer)}; robust-positive vs SMAP climatology: "
            f"{robust}; robust added value vs SMAP persistence: {added}."
        )
        by_source = (
            transfer.groupby("source_product")
            .agg(
                n_rows=("model", "size"),
                robust_positive=("claim_status_vs_smap_climatology", lambda s: int((s == "robust_positive").sum())),
                robust_added_value=("added_value_status_vs_smap_persistence", lambda s: int((s == "stack_robust_added_value").sum())),
                mean_bss_vs_clim=("bss_vs_smap_climatology", "mean"),
                mean_bss_vs_persistence=("bss_vs_smap_persistence_selected", "mean"),
            )
            .reset_index()
        )
        lines.append("")
        lines.append("By source product:")
        lines.append(by_source.to_string(index=False))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if args.copy_report:
        LANDSURFACE_DIR.mkdir(parents=True, exist_ok=True)

    target = read_target(args.target_file)
    monthly_parts: list[pd.DataFrame] = []

    for source_product in args.sources:
        if source_product == "smap_l4_multi3snapshot":
            continue
        source = read_source(source_product)
        monthly_parts.append(
            build_monthly_rows(
                target,
                source,
                source_product,
                args.target_file,
                transfer_kind="cross_product_transfer",
            )
        )

    if args.include_smap_self_reference:
        source = read_source("smap_l4_multi3snapshot")
        monthly_parts.append(
            build_monthly_rows(
                target,
                source,
                "smap_l4_multi3snapshot",
                args.target_file,
                transfer_kind="target_self_calibrated_reference",
            )
        )

    monthly = pd.concat([p for p in monthly_parts if not p.empty], ignore_index=True)
    if monthly.empty:
        raise RuntimeError("No monthly transfer rows were produced.")
    summary = score_monthly(monthly, n_bootstrap=args.n_bootstrap)

    monthly_path = OUT_DIR / f"{args.out_prefix}_monthly_scores.csv"
    summary_path = OUT_DIR / f"{args.out_prefix}_summary.csv"
    text_path = OUT_DIR / f"{args.out_prefix}_summary.txt"
    monthly.to_csv(monthly_path, index=False)
    summary.to_csv(summary_path, index=False)
    write_summary_text(summary, text_path)

    if args.copy_report:
        for path in [monthly_path, summary_path, text_path]:
            shutil.copy2(path, LANDSURFACE_DIR / path.name)

    print(f"Wrote {summary_path}")
    print(f"Wrote {monthly_path}")
    print(f"Wrote {text_path}")
    if args.copy_report:
        print(f"Copied outputs to {LANDSURFACE_DIR}")

    compact_cols = [
        "source_product",
        "region",
        "model",
        "transfer_kind",
        "n_months",
        "bss_vs_smap_climatology",
        "bss_vs_smap_climatology_ci_low",
        "bss_vs_smap_climatology_ci_high",
        "bss_vs_smap_persistence_selected",
        "added_value_status_vs_smap_persistence",
        "transfer_status",
    ]
    print(summary[compact_cols].to_string(index=False))


if __name__ == "__main__":
    main()
