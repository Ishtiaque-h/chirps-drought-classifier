#!/usr/bin/env python
"""SMAP product-transfer persistence-residual benchmark.

This script asks a focused question:

    After product-specific base-rate adaptation, can GEFSv12 RZSM probabilities
    beat raw same-target SMAP persistence on frozen test months?

It reads the SMAP multi-snapshot target-product adaptation monthly file, rebuilds
raw one-month SMAP persistence from the target series, and fits only
validation-safe selectors on 2015-2016 before scoring 2017-2019.

The benchmark is intentionally simple because the SMAP validation window is
short. It is a guardrail for the target-product calibration claim, not a new
deployment system.
"""

from __future__ import annotations

from argparse import ArgumentParser, Namespace
from pathlib import Path
import shutil

import numpy as np
import pandas as pd

from run_gefsv12_landsurface_stack_benchmark import (
    LANDSURFACE_DIR,
    OUT_DIR,
    PROJECT_ROOT,
    added_value_status,
    bootstrap_delta_bs,
)
from run_landsurface_forecast_benchmark import brier, bss, bootstrap_bss


PAPER_DIR = PROJECT_ROOT / "results" / "report" / "paper"
DEFAULT_MONTHLY = (
    LANDSURFACE_DIR
    / "landsurface_target_product_adaptation_benchmark_smap_multi3snapshot_monthly_scores.csv"
)

GEFS_METHODS = {
    "direct_gefs": ("Fixed direct transfer", "gefs_selected"),
    "dryrate_gefs": ("Fixed validation dry-rate shift", "gefs_selected"),
    "predrate_gefs": ("Fixed validation prediction-base-rate shift", "gefs_selected"),
    "temp_base_gefs": ("Fixed bias-corrected temperature/base-rate scaling", "gefs_selected"),
    "smap_recal_gefs": ("Fixed SMAP validation recalibration", "gefs_selected"),
}

STACK_METHODS = {
    "direct_stack": ("Fixed direct transfer", "stack_validation_selected"),
    "dryrate_stack": ("Fixed validation dry-rate shift", "stack_validation_selected"),
    "predrate_stack": ("Fixed validation prediction-base-rate shift", "stack_validation_selected"),
    "temp_base_stack": ("Fixed bias-corrected temperature/base-rate scaling", "stack_validation_selected"),
    "smap_recal_stack": ("Fixed SMAP validation recalibration", "stack_validation_selected"),
}

BASE_RATE_CANDIDATES = ["dryrate_gefs", "predrate_gefs", "temp_base_gefs"]
ALL_CANDIDATES = {**GEFS_METHODS, **STACK_METHODS}
REFERENCE_COL = "smap_persistence_raw_prob_dry"


def parse_args() -> Namespace:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--monthly", type=Path, default=DEFAULT_MONTHLY)
    parser.add_argument("--validation-start-year", type=int, default=2015)
    parser.add_argument("--validation-end-year", type=int, default=2016)
    parser.add_argument("--test-start-year", type=int, default=2017)
    parser.add_argument("--test-end-year", type=int, default=2019)
    parser.add_argument("--n-bootstrap", type=int, default=2000)
    parser.add_argument(
        "--out-prefix",
        default="landsurface_smap_product_residual_benchmark_multi3snapshot",
    )
    parser.add_argument("--copy-report", action="store_true")
    return parser.parse_args()


def load_monthly(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"SMAP adaptation monthly file not found: {path}")
    df = pd.read_csv(path, parse_dates=["target_time"])
    required = {
        "region",
        "target_time",
        "target_year",
        "target_month",
        "y_true_dry_frac",
        "smap_clim_prob_dry",
        "prob_dry",
        "adaptation_method",
        "source_product",
        "model",
    }
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Monthly file is missing required columns: {sorted(missing)}")
    return df


def split_years(df: pd.DataFrame, start_year: int, end_year: int) -> pd.DataFrame:
    return df.loc[(df["target_year"] >= start_year) & (df["target_year"] <= end_year)].copy()


def raw_smap_persistence(base: pd.DataFrame) -> pd.DataFrame:
    """Compute one-month persistence from each region's SMAP target series."""
    keys = ["region", "target_time", "target_year", "target_month", "y_true_dry_frac", "smap_clim_prob_dry"]
    unique = base[keys].drop_duplicates().sort_values(["region", "target_time"]).copy()
    persist = unique[["region", "target_time", "y_true_dry_frac"]].rename(
        columns={"target_time": "persistence_time", "y_true_dry_frac": REFERENCE_COL}
    )
    unique["persistence_time"] = (
        unique["target_time"] - pd.DateOffset(months=1)
    ).dt.to_period("M").dt.to_timestamp()
    return unique.merge(persist, on=["region", "persistence_time"], how="left")


def build_candidate_frame(monthly: pd.DataFrame, region: str, source_product: str) -> pd.DataFrame:
    base = raw_smap_persistence(monthly.loc[monthly["region"].eq(region)].copy())
    out = base.copy()
    subset = monthly.loc[
        monthly["region"].eq(region) & monthly["source_product"].eq(source_product)
    ].copy()
    for label, (method, model) in ALL_CANDIDATES.items():
        part = subset.loc[
            subset["adaptation_method"].eq(method) & subset["model"].eq(model),
            ["target_time", "prob_dry"],
        ].drop_duplicates("target_time")
        if part.empty:
            continue
        out = out.merge(part.rename(columns={"prob_dry": label}), on="target_time", how="left")
    out["region"] = region
    out["source_product"] = source_product
    return out.sort_values("target_time").reset_index(drop=True)


def available_candidates(frame: pd.DataFrame) -> dict[str, str]:
    return {
        name: name
        for name in ALL_CANDIDATES
        if name in frame.columns and frame[name].notna().any()
    }


def fit_best(
    val: pd.DataFrame,
    candidates: dict[str, str],
    candidate_names: list[str] | None = None,
) -> dict[str, object]:
    names = candidate_names or list(candidates)
    names = [name for name in names if name in candidates]
    rows = [(brier(val["y_true_dry_frac"], val[REFERENCE_COL]), "smap_persistence_raw", REFERENCE_COL)]
    for name in names:
        rows.append((brier(val["y_true_dry_frac"], val[candidates[name]]), name, candidates[name]))
    rows.sort(key=lambda item: (item[0], item[1] != "smap_persistence_raw", item[1]))
    bs, name, col = rows[0]
    return {"selected_name": name, "selected_col": col, "validation_bs": float(bs)}


def fit_guarded_best(
    val: pd.DataFrame,
    candidates: dict[str, str],
    candidate_names: list[str],
    n_bootstrap: int,
) -> dict[str, object]:
    best = fit_best(val, candidates, candidate_names=candidate_names)
    if best["selected_name"] == "smap_persistence_raw":
        return {
            **best,
            "guard_passed": False,
            "applied_name": "smap_persistence_raw",
            "applied_col": REFERENCE_COL,
            "delta_bs_ci_low": 0.0,
            "delta_bs_ci_high": 0.0,
        }
    ci_low, ci_high = bootstrap_delta_bs(
        val,
        candidate_col=str(best["selected_col"]),
        reference_col=REFERENCE_COL,
        n_bootstrap=n_bootstrap,
        seed=3301,
    )
    guard_passed = bool(ci_high < 0.0)
    return {
        **best,
        "guard_passed": guard_passed,
        "applied_name": str(best["selected_name"]) if guard_passed else "smap_persistence_raw",
        "applied_col": str(best["selected_col"]) if guard_passed else REFERENCE_COL,
        "delta_bs_ci_low": float(ci_low),
        "delta_bs_ci_high": float(ci_high),
    }


def threshold_grid(val: pd.DataFrame, candidate_col: str) -> np.ndarray:
    gap = (val[candidate_col].astype(float) - val[REFERENCE_COL].astype(float)).abs().to_numpy()
    finite = gap[np.isfinite(gap)]
    if finite.size == 0:
        return np.array([np.inf])
    qs = np.nanquantile(finite, np.linspace(0.0, 0.90, 12))
    return np.array(sorted({0.0, *[float(x) for x in qs if np.isfinite(x)], np.inf}), dtype=float)


def fit_threshold_gate(
    val: pd.DataFrame,
    candidates: dict[str, str],
    candidate_names: list[str],
) -> dict[str, object]:
    y = val["y_true_dry_frac"].to_numpy(dtype=float)
    ref = val[REFERENCE_COL].to_numpy(dtype=float)
    rows: list[tuple[float, str, str, float, float]] = [
        (brier(y, ref), "smap_persistence_raw", "none", np.inf, 0.0)
    ]
    for name in candidate_names:
        if name not in candidates:
            continue
        col = candidates[name]
        cand = val[col].to_numpy(dtype=float)
        diff = cand - ref
        for direction in ["any", "forecast_drier", "forecast_wetter"]:
            for threshold in threshold_grid(val, col):
                if direction == "any":
                    mask = np.abs(diff) >= threshold
                elif direction == "forecast_drier":
                    mask = diff >= threshold
                else:
                    mask = diff <= -threshold
                pred = np.where(mask, cand, ref)
                rows.append((brier(y, pred), name, direction, float(threshold), float(mask.mean())))
    rows.sort(key=lambda item: (item[0], item[1] != "smap_persistence_raw", item[1], item[2], item[3]))
    bs, name, direction, threshold, dyn_frac = rows[0]
    return {
        "selected_name": name,
        "selected_col": candidates.get(name, REFERENCE_COL),
        "direction": direction,
        "threshold": threshold,
        "validation_bs": float(bs),
        "dynamic_fraction_validation": float(dyn_frac),
    }


def apply_threshold_gate(test: pd.DataFrame, meta: dict[str, object]) -> tuple[pd.Series, pd.Series]:
    if meta["selected_name"] == "smap_persistence_raw":
        pred = test[REFERENCE_COL].astype(float)
        selected = pd.Series("smap_persistence_raw", index=test.index, dtype="object")
        return pred, selected
    cand = test[str(meta["selected_col"])].astype(float).to_numpy()
    ref = test[REFERENCE_COL].astype(float).to_numpy()
    diff = cand - ref
    direction = str(meta["direction"])
    threshold = float(meta["threshold"])
    if direction == "any":
        mask = np.abs(diff) >= threshold
    elif direction == "forecast_drier":
        mask = diff >= threshold
    elif direction == "forecast_wetter":
        mask = diff <= -threshold
    else:
        mask = np.zeros(len(test), dtype=bool)
    selected = pd.Series(np.where(mask, str(meta["selected_name"]), "smap_persistence_raw"), index=test.index)
    pred = pd.Series(np.where(mask, cand, ref), index=test.index)
    return pred.clip(0.0, 1.0), selected


def score_group(
    frame: pd.DataFrame,
    model_cols: dict[str, str],
    n_bootstrap: int,
    seed_offset: int,
) -> pd.DataFrame:
    rows = []
    y = frame["y_true_dry_frac"].to_numpy(dtype=float)
    clim = frame["smap_clim_prob_dry"].to_numpy(dtype=float)
    ref = frame[REFERENCE_COL].to_numpy(dtype=float)
    for i, (model, col) in enumerate(model_cols.items()):
        if col not in frame.columns:
            continue
        pred = frame[col].to_numpy(dtype=float)
        ci_low, ci_high = bootstrap_bss(
            frame,
            pred_col=col,
            ref_col="smap_clim_prob_dry",
            n_bootstrap=n_bootstrap,
            seed=3401 + seed_offset + i,
        )
        delta = brier(y, pred) - brier(y, ref)
        delta_low, delta_high = bootstrap_delta_bs(
            frame,
            candidate_col=col,
            reference_col=REFERENCE_COL,
            n_bootstrap=n_bootstrap,
            seed=3501 + seed_offset + i,
        )
        rows.append(
            {
                "model": model,
                "n_test_months": int(len(frame)),
                "bs_smap_climatology": brier(y, clim),
                "bs_raw_smap_persistence": brier(y, ref),
                "bs_model": brier(y, pred),
                "bss_vs_smap_climatology": bss(y, pred, clim),
                "bss_ci_low": ci_low,
                "bss_ci_high": ci_high,
                "bss_vs_raw_smap_persistence": bss(y, pred, ref),
                "delta_bs_model_minus_raw_smap_persistence": delta,
                "delta_bs_ci_low": delta_low,
                "delta_bs_ci_high": delta_high,
                "added_value_status_vs_raw_smap_persistence": added_value_status(delta, delta_low, delta_high),
                "spearman_model_vs_smap_observed": frame[col].corr(frame["y_true_dry_frac"], method="spearman"),
            }
        )
    return pd.DataFrame(rows)


def compact_summary(summary: pd.DataFrame) -> pd.DataFrame:
    return (
        summary.groupby("model", dropna=False)
        .agg(
            n_rows=("region", "size"),
            n_regions=("region", "nunique"),
            n_sources=("source_product", "nunique"),
            robust_positive_count=("bss_ci_low", lambda s: int((s > 0).sum())),
            robust_added_value_count=(
                "added_value_status_vs_raw_smap_persistence",
                lambda s: int((s == "stack_robust_added_value").sum()),
            ),
            mean_bss_vs_smap_climatology=("bss_vs_smap_climatology", "mean"),
            median_bss_vs_smap_climatology=("bss_vs_smap_climatology", "median"),
            mean_bss_vs_raw_smap_persistence=("bss_vs_raw_smap_persistence", "mean"),
            median_bss_vs_raw_smap_persistence=("bss_vs_raw_smap_persistence", "median"),
            mean_delta_bs_vs_raw_smap_persistence=("delta_bs_model_minus_raw_smap_persistence", "mean"),
        )
        .reset_index()
        .sort_values(["robust_added_value_count", "mean_delta_bs_vs_raw_smap_persistence"], ascending=[False, True])
    )


def main() -> None:
    args = parse_args()
    monthly = load_monthly(args.monthly)
    sources = ["era5_land", "gldas_noah"]
    regions = sorted(monthly["region"].dropna().unique())

    monthly_parts: list[pd.DataFrame] = []
    summary_parts: list[pd.DataFrame] = []
    meta_rows: list[dict[str, object]] = []

    for region in regions:
        for source_product in sources:
            frame = build_candidate_frame(monthly, region, source_product)
            candidates = available_candidates(frame)
            required = {"dryrate_gefs", "predrate_gefs", "temp_base_gefs"}
            if not required.intersection(candidates):
                continue
            frame = frame.dropna(subset=["y_true_dry_frac", "smap_clim_prob_dry", REFERENCE_COL]).copy()
            val = split_years(frame, args.validation_start_year, args.validation_end_year)
            test = split_years(frame, args.test_start_year, args.test_end_year)
            if val.empty or test.empty:
                continue

            best_base = fit_best(val, candidates, BASE_RATE_CANDIDATES)
            guarded_base = fit_guarded_best(val, candidates, BASE_RATE_CANDIDATES, args.n_bootstrap)
            threshold_base = fit_threshold_gate(val, candidates, BASE_RATE_CANDIDATES)
            best_all = fit_best(val, candidates)
            guarded_all = fit_guarded_best(val, candidates, list(candidates), args.n_bootstrap)

            test = test.copy()
            test["best_base_rate_prob_dry"] = test[str(best_base["selected_col"])].astype(float)
            test["guarded_base_rate_prob_dry"] = test[str(guarded_base["applied_col"])].astype(float)
            gate_pred, gate_selected = apply_threshold_gate(test, threshold_base)
            test["threshold_base_rate_gate_prob_dry"] = gate_pred
            test["threshold_base_rate_gate_selected_model"] = gate_selected
            test["best_all_prob_dry"] = test[str(best_all["selected_col"])].astype(float)
            test["guarded_all_prob_dry"] = test[str(guarded_all["applied_col"])].astype(float)
            test["best_base_rate_selected_model"] = str(best_base["selected_name"])
            test["guarded_base_rate_selected_model"] = str(guarded_base["applied_name"])
            test["best_all_selected_model"] = str(best_all["selected_name"])
            test["guarded_all_selected_model"] = str(guarded_all["applied_name"])
            test["region"] = region
            test["source_product"] = source_product

            score_cols = {
                "smap_persistence_raw": REFERENCE_COL,
                "best_base_rate": "best_base_rate_prob_dry",
                "guarded_base_rate": "guarded_base_rate_prob_dry",
                "threshold_base_rate_gate": "threshold_base_rate_gate_prob_dry",
                "best_all_candidates": "best_all_prob_dry",
                "guarded_all_candidates": "guarded_all_prob_dry",
            }
            for name in [
                "direct_gefs",
                "dryrate_gefs",
                "predrate_gefs",
                "temp_base_gefs",
                "smap_recal_gefs",
                "direct_stack",
                "dryrate_stack",
                "predrate_stack",
                "temp_base_stack",
                "smap_recal_stack",
            ]:
                if name in candidates:
                    score_cols[name] = name

            scored = score_group(test, score_cols, args.n_bootstrap, seed_offset=100 * len(summary_parts))
            scored.insert(0, "source_product", source_product)
            scored.insert(0, "region", region)
            summary_parts.append(scored)

            meta_rows.extend(
                [
                    {"region": region, "source_product": source_product, "selector": "best_base_rate", **best_base},
                    {"region": region, "source_product": source_product, "selector": "guarded_base_rate", **guarded_base},
                    {"region": region, "source_product": source_product, "selector": "threshold_base_rate_gate", **threshold_base},
                    {"region": region, "source_product": source_product, "selector": "best_all_candidates", **best_all},
                    {"region": region, "source_product": source_product, "selector": "guarded_all_candidates", **guarded_all},
                ]
            )

            keep_cols = [
                "region",
                "source_product",
                "target_time",
                "target_year",
                "target_month",
                "y_true_dry_frac",
                "smap_clim_prob_dry",
                REFERENCE_COL,
                *[name for name in ALL_CANDIDATES if name in test.columns],
                "best_base_rate_prob_dry",
                "guarded_base_rate_prob_dry",
                "threshold_base_rate_gate_prob_dry",
                "best_all_prob_dry",
                "guarded_all_prob_dry",
                "best_base_rate_selected_model",
                "guarded_base_rate_selected_model",
                "threshold_base_rate_gate_selected_model",
                "best_all_selected_model",
                "guarded_all_selected_model",
            ]
            monthly_parts.append(test[[c for c in keep_cols if c in test.columns]].copy())

            row = scored.loc[scored["model"].eq("guarded_base_rate")].iloc[0]
            print(
                f"region={region:<24} source={source_product:<10} "
                f"guarded_base_deltaBS={float(row['delta_bs_model_minus_raw_smap_persistence']):+.4f} "
                f"status={row['added_value_status_vs_raw_smap_persistence']}",
                flush=True,
            )

    if not summary_parts:
        raise RuntimeError("No SMAP product-residual groups were scored.")

    OUT_DIR.mkdir(exist_ok=True)
    LANDSURFACE_DIR.mkdir(parents=True, exist_ok=True)
    PAPER_DIR.mkdir(parents=True, exist_ok=True)

    monthly_out = pd.concat(monthly_parts, ignore_index=True)
    summary = pd.concat(summary_parts, ignore_index=True)
    meta = pd.DataFrame(meta_rows)
    compact = compact_summary(summary)

    paths = {
        "monthly": OUT_DIR / f"{args.out_prefix}_monthly_scores.csv",
        "summary": OUT_DIR / f"{args.out_prefix}_summary.csv",
        "compact": OUT_DIR / f"{args.out_prefix}_compact_summary.csv",
        "meta": OUT_DIR / f"{args.out_prefix}_selector_meta.csv",
        "text": OUT_DIR / f"{args.out_prefix}_summary.txt",
    }
    monthly_out.to_csv(paths["monthly"], index=False)
    summary.to_csv(paths["summary"], index=False)
    compact.to_csv(paths["compact"], index=False)
    meta.to_csv(paths["meta"], index=False)

    lines = [
        "SMAP Product-Transfer Persistence-Residual Benchmark",
        "=" * 72,
        f"Validation years: {args.validation_start_year}-{args.validation_end_year}",
        f"Test years: {args.test_start_year}-{args.test_end_year}",
        f"Reference: {REFERENCE_COL}",
        "",
        compact.to_string(index=False),
    ]
    paths["text"].write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(paths["text"].read_text(encoding="utf-8"))

    if args.copy_report:
        for path in paths.values():
            shutil.copy2(path, LANDSURFACE_DIR / path.name)
        shutil.copy2(paths["summary"], PAPER_DIR / "table34_smap_product_residual_selector.csv")
        shutil.copy2(paths["compact"], PAPER_DIR / "table35_smap_product_residual_selector_compact.csv")
        print(f"Copied paper table: {PAPER_DIR / 'table34_smap_product_residual_selector.csv'}")
        print(f"Copied paper table: {PAPER_DIR / 'table35_smap_product_residual_selector_compact.csv'}")


if __name__ == "__main__":
    main()
