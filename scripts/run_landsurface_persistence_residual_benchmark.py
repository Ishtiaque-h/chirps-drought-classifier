#!/usr/bin/env python
"""Persistence-residual benchmark for GEFSv12 land-surface drought forecasts.

The purpose of this script is narrow: test whether validation-safe, physically
interpretable rules can identify months where forecast-informed RZSM
probabilities add information beyond same-target land-memory persistence.

It reuses the GEFSv12 domain-transfer setup:

  - validation years are used to calibrate GEFS, persistence, stack, and
    optional monotonic-XGB probabilities;
  - the frozen target-region test years are then scored against climatology and
    raw same-target persistence;
  - adaptive selectors are trained only on validation rows and default back to
    raw persistence when they have no evidence that a dynamic forecast should
    be trusted.

This is a guardrail benchmark, not a deployment model.
"""

from __future__ import annotations

from argparse import ArgumentParser, Namespace
from pathlib import Path
import shutil

import numpy as np
import pandas as pd

from region_config import resolve_region
from run_gefsv12_landsurface_stack_benchmark import (
    LANDSURFACE_DIR,
    OUT_DIR,
    PROJECT_ROOT,
    added_value_status,
    bootstrap_delta_bs,
)
from run_landsurface_domain_transfer_benchmark import (
    DEFAULT_REGIONS,
    fit_transfer_predictions,
    load_region_frame,
    transfer_source_regions,
)
from run_landsurface_forecast_benchmark import brier, bss, bootstrap_bss


PAPER_DIR = PROJECT_ROOT / "results" / "report" / "paper"
REFERENCE_COL = "persistence_raw_prob_dry"
DISAGREEMENT_COL = "gefs_transfer_selected_prob_dry"


def parse_args() -> Namespace:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--regions", nargs="+", default=DEFAULT_REGIONS)
    parser.add_argument("--forecast-dir", type=Path, default=PROJECT_ROOT / "results" / "report")
    parser.add_argument("--lead-months", type=int, default=1)
    parser.add_argument("--start-target", default="2000-01")
    parser.add_argument("--end-target", default="2019-12")
    parser.add_argument("--validation-start-year", type=int, default=2000)
    parser.add_argument("--validation-end-year", type=int, default=2016)
    parser.add_argument("--test-start-year", type=int, default=2017)
    parser.add_argument("--test-end-year", type=int, default=2019)
    parser.add_argument("--normal-start-year", type=int, default=1991)
    parser.add_argument("--normal-end-year", type=int, default=2016)
    parser.add_argument("--dry-quantile", type=float, default=0.20)
    parser.add_argument("--valid-day", type=int, default=15)
    parser.add_argument("--init-lag-weeks", type=int, default=0)
    parser.add_argument("--weight-steps", type=int, default=101)
    parser.add_argument("--n-bootstrap", type=int, default=2000)
    parser.add_argument("--min-regime-rows", type=int, default=24)
    parser.add_argument("--skip-monotonic-xgb", action="store_true")
    parser.add_argument(
        "--out-prefix",
        default="landsurface_persistence_residual_benchmark_day15_hindcastcal",
    )
    parser.add_argument("--copy-report", action="store_true")
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--max-months", type=int, default=None)
    parser.add_argument("--progress-every", type=int, default=120)
    return parser.parse_args()


def candidate_columns(frame: pd.DataFrame) -> dict[str, str]:
    candidates = {
        "persistence_raw": "persistence_raw_prob_dry",
        "persistence_transfer_selected": "persistence_transfer_selected_prob_dry",
        "gefs_transfer_selected": "gefs_transfer_selected_prob_dry",
        "stack_transfer_selected": "stack_transfer_selected_prob_dry",
    }
    if "monotonic_xgb_transfer_prob_dry" in frame.columns:
        candidates["monotonic_xgb_transfer"] = "monotonic_xgb_transfer_prob_dry"
    return {name: col for name, col in candidates.items() if col in frame.columns}


def add_selector_features(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["forecast_minus_persistence"] = (
        out[DISAGREEMENT_COL].astype(float) - out[REFERENCE_COL].astype(float)
    )
    out["abs_forecast_persistence_gap"] = out["forecast_minus_persistence"].abs()
    out["forecast_wetter_than_memory"] = out["forecast_minus_persistence"] < 0.0
    out["forecast_drier_than_memory"] = out["forecast_minus_persistence"] > 0.0
    return out


def fit_best_single(source: pd.DataFrame, candidates: dict[str, str]) -> dict[str, object]:
    scores = [
        (brier(source["y_true_dry_frac"], source[col]), name, col)
        for name, col in candidates.items()
    ]
    scores.sort(key=lambda item: (item[0], item[1] != "persistence_raw", item[1]))
    bs, name, col = scores[0]
    return {
        "selected_name": name,
        "selected_col": col,
        "validation_bs": float(bs),
    }


def apply_best_single(target: pd.DataFrame, meta: dict[str, object]) -> pd.Series:
    return target[str(meta["selected_col"])].astype(float).clip(0.0, 1.0)


def fit_guarded_best(
    source: pd.DataFrame,
    candidates: dict[str, str],
    n_bootstrap: int,
) -> dict[str, object]:
    best = fit_best_single(source, candidates)
    if best["selected_name"] == "persistence_raw":
        return {
            **best,
            "guard_passed": False,
            "delta_bs_vs_raw_persistence": 0.0,
            "delta_bs_ci_low": 0.0,
            "delta_bs_ci_high": 0.0,
            "applied_name": "persistence_raw",
            "applied_col": REFERENCE_COL,
        }

    delta = brier(source["y_true_dry_frac"], source[str(best["selected_col"])]) - brier(
        source["y_true_dry_frac"],
        source[REFERENCE_COL],
    )
    ci_low, ci_high = bootstrap_delta_bs(
        source,
        candidate_col=str(best["selected_col"]),
        reference_col=REFERENCE_COL,
        n_bootstrap=n_bootstrap,
        seed=2601,
    )
    guard_passed = bool(ci_high < 0.0)
    return {
        **best,
        "guard_passed": guard_passed,
        "delta_bs_vs_raw_persistence": float(delta),
        "delta_bs_ci_low": float(ci_low),
        "delta_bs_ci_high": float(ci_high),
        "applied_name": str(best["selected_name"]) if guard_passed else "persistence_raw",
        "applied_col": str(best["selected_col"]) if guard_passed else REFERENCE_COL,
    }


def apply_guarded_best(target: pd.DataFrame, meta: dict[str, object]) -> pd.Series:
    return target[str(meta["applied_col"])].astype(float).clip(0.0, 1.0)


def threshold_grid(source: pd.DataFrame) -> np.ndarray:
    gap = source["abs_forecast_persistence_gap"].to_numpy(dtype=float)
    finite = gap[np.isfinite(gap)]
    if finite.size == 0:
        return np.array([np.inf], dtype=float)
    quantiles = np.nanquantile(finite, np.linspace(0.0, 0.95, 20))
    values = sorted({0.0, *[float(x) for x in quantiles if np.isfinite(x)], np.inf})
    return np.array(values, dtype=float)


def gate_mask(frame: pd.DataFrame, direction: str, threshold: float) -> np.ndarray:
    diff = frame["forecast_minus_persistence"].to_numpy(dtype=float)
    if direction == "any":
        return np.abs(diff) >= threshold
    if direction == "forecast_drier":
        return diff >= threshold
    if direction == "forecast_wetter":
        return diff <= -threshold
    raise ValueError(f"Unknown gate direction: {direction}")


def fit_threshold_gate(source: pd.DataFrame, candidates: dict[str, str]) -> dict[str, object]:
    dynamic_candidates = {
        name: col
        for name, col in candidates.items()
        if name not in {"persistence_raw", "persistence_transfer_selected"}
    }
    if not dynamic_candidates:
        return {
            "selected_name": "persistence_raw",
            "selected_col": REFERENCE_COL,
            "direction": "none",
            "threshold": np.inf,
            "validation_bs": brier(source["y_true_dry_frac"], source[REFERENCE_COL]),
            "dynamic_fraction_validation": 0.0,
        }

    y = source["y_true_dry_frac"].to_numpy(dtype=float)
    ref = source[REFERENCE_COL].to_numpy(dtype=float)
    rows: list[tuple[float, str, str, float, float]] = [
        (brier(y, ref), "persistence_raw", "none", np.inf, 0.0)
    ]
    for name, col in dynamic_candidates.items():
        cand = source[col].to_numpy(dtype=float)
        for direction in ["any", "forecast_drier", "forecast_wetter"]:
            for threshold in threshold_grid(source):
                mask = gate_mask(source, direction, threshold)
                pred = np.where(mask, cand, ref)
                rows.append((brier(y, pred), name, direction, float(threshold), float(mask.mean())))
    rows.sort(key=lambda item: (item[0], item[1] != "persistence_raw", item[1], item[2], item[3]))
    bs, name, direction, threshold, dyn_frac = rows[0]
    return {
        "selected_name": name,
        "selected_col": candidates.get(name, REFERENCE_COL),
        "direction": direction,
        "threshold": threshold,
        "validation_bs": float(bs),
        "dynamic_fraction_validation": float(dyn_frac),
    }


def apply_threshold_gate(target: pd.DataFrame, meta: dict[str, object]) -> pd.Series:
    if meta["selected_name"] == "persistence_raw":
        return target[REFERENCE_COL].astype(float).clip(0.0, 1.0)
    mask = gate_mask(target, str(meta["direction"]), float(meta["threshold"]))
    cand = target[str(meta["selected_col"])].to_numpy(dtype=float)
    ref = target[REFERENCE_COL].to_numpy(dtype=float)
    return pd.Series(np.where(mask, cand, ref), index=target.index).clip(0.0, 1.0)


def bin_edges(values: pd.Series) -> tuple[float, float]:
    arr = values.to_numpy(dtype=float)
    lo, hi = np.nanquantile(arr, [1.0 / 3.0, 2.0 / 3.0])
    if not np.isfinite(lo) or not np.isfinite(hi) or lo >= hi:
        lo = float(np.nanmedian(arr))
        hi = lo
    return float(lo), float(hi)


def memory_bin(values: pd.Series, lo: float, hi: float) -> pd.Series:
    return pd.cut(
        values.astype(float),
        bins=[-np.inf, lo, hi, np.inf],
        labels=["low_memory", "mid_memory", "high_memory"],
        include_lowest=True,
    ).astype(str)


def disagreement_bin(diff: pd.Series, threshold: float) -> pd.Series:
    arr = diff.astype(float)
    out = pd.Series("near_memory", index=diff.index, dtype="object")
    out.loc[arr >= threshold] = "forecast_drier"
    out.loc[arr <= -threshold] = "forecast_wetter"
    return out


def add_regime_labels(
    frame: pd.DataFrame,
    memory_lo: float,
    memory_hi: float,
    disagreement_threshold: float,
) -> pd.Series:
    mem = memory_bin(frame[REFERENCE_COL], memory_lo, memory_hi)
    dis = disagreement_bin(frame["forecast_minus_persistence"], disagreement_threshold)
    return mem + "|" + dis


def fit_regime_selector(
    source: pd.DataFrame,
    candidates: dict[str, str],
    min_regime_rows: int,
) -> dict[str, object]:
    source = source.copy()
    mem_lo, mem_hi = bin_edges(source[REFERENCE_COL])
    disagreement_threshold = float(np.nanquantile(source["abs_forecast_persistence_gap"], 0.50))
    if not np.isfinite(disagreement_threshold):
        disagreement_threshold = 0.0
    source["selector_regime"] = add_regime_labels(source, mem_lo, mem_hi, disagreement_threshold)
    default_bs = brier(source["y_true_dry_frac"], source[REFERENCE_COL])
    choices: dict[str, str] = {}
    validation_rows: list[dict[str, object]] = []
    for regime, part in source.groupby("selector_regime", sort=True):
        if len(part) < min_regime_rows:
            choices[str(regime)] = "persistence_raw"
            validation_rows.append(
                {
                    "regime": regime,
                    "n_validation_rows": int(len(part)),
                    "selected_name": "persistence_raw",
                    "validation_bs": brier(part["y_true_dry_frac"], part[REFERENCE_COL]),
                    "reason": "small_regime_default",
                }
            )
            continue
        scores = [
            (brier(part["y_true_dry_frac"], part[col]), name)
            for name, col in candidates.items()
        ]
        scores.sort(key=lambda item: (item[0], item[1] != "persistence_raw", item[1]))
        bs, selected = scores[0]
        choices[str(regime)] = selected
        validation_rows.append(
            {
                "regime": regime,
                "n_validation_rows": int(len(part)),
                "selected_name": selected,
                "validation_bs": float(bs),
                "reason": "validation_min_bs",
            }
        )
    return {
        "memory_lo": mem_lo,
        "memory_hi": mem_hi,
        "disagreement_threshold": disagreement_threshold,
        "choices": choices,
        "validation_bs": float(default_bs),
        "validation_rows": validation_rows,
    }


def apply_regime_selector(
    target: pd.DataFrame,
    candidates: dict[str, str],
    meta: dict[str, object],
) -> tuple[pd.Series, pd.Series]:
    regimes = add_regime_labels(
        target,
        float(meta["memory_lo"]),
        float(meta["memory_hi"]),
        float(meta["disagreement_threshold"]),
    )
    choices: dict[str, str] = dict(meta["choices"])  # type: ignore[arg-type]
    pred = target[REFERENCE_COL].astype(float).copy()
    selected = pd.Series("persistence_raw", index=target.index, dtype="object")
    for regime, idx in regimes.groupby(regimes).groups.items():
        name = choices.get(str(regime), "persistence_raw")
        col = candidates.get(name, REFERENCE_COL)
        pred.loc[idx] = target.loc[idx, col].astype(float)
        selected.loc[idx] = name
    return pred.clip(0.0, 1.0), selected


def score_models(
    target_region: str,
    transfer_mode: str,
    source_regions: list[str],
    target: pd.DataFrame,
    candidates: dict[str, str],
    selector_meta: dict[str, dict[str, object]],
    n_bootstrap: int,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    scoring_cols = {
        **candidates,
        "best_single_validation": "best_single_validation_prob_dry",
        "guarded_best_validation": "guarded_best_validation_prob_dry",
        "residual_threshold_gate": "residual_threshold_gate_prob_dry",
        "residual_regime_selector": "residual_regime_selector_prob_dry",
    }
    y = target["y_true_dry_frac"].to_numpy(dtype=float)
    clim = target["clim_prob_dry"].to_numpy(dtype=float)
    raw_persistence = target[REFERENCE_COL].to_numpy(dtype=float)
    for i, (model, col) in enumerate(scoring_cols.items()):
        if col not in target.columns:
            continue
        pred = target[col].to_numpy(dtype=float)
        ci_low, ci_high = bootstrap_bss(
            target,
            pred_col=col,
            ref_col="clim_prob_dry",
            n_bootstrap=n_bootstrap,
            seed=2401 + i,
        )
        delta = brier(y, pred) - brier(y, raw_persistence)
        delta_low, delta_high = bootstrap_delta_bs(
            target,
            candidate_col=col,
            reference_col=REFERENCE_COL,
            n_bootstrap=n_bootstrap,
            seed=2501 + i,
        )
        meta = selector_meta.get(model, {})
        rows.append(
            {
                "target_region": target_region,
                "transfer_mode": transfer_mode,
                "source_regions": " ".join(source_regions),
                "model": model,
                "n_test_months": int(target["target_time"].nunique()),
                "bs_climatology": brier(y, clim),
                "bs_raw_persistence": brier(y, raw_persistence),
                "bs_model": brier(y, pred),
                "bss_vs_climatology": bss(y, pred, clim),
                "bss_ci_low": ci_low,
                "bss_ci_high": ci_high,
                "bss_vs_raw_persistence": bss(y, pred, raw_persistence),
                "delta_bs_model_minus_raw_persistence": delta,
                "delta_bs_ci_low": delta_low,
                "delta_bs_ci_high": delta_high,
                "added_value_status_vs_raw_persistence": added_value_status(delta, delta_low, delta_high),
                "dynamic_fraction_test": dynamic_fraction(target, model),
                "selector_detail": selector_detail(meta),
                "spearman_model_vs_observed": target[col].corr(target["y_true_dry_frac"], method="spearman"),
            }
        )
    return pd.DataFrame(rows)


def dynamic_fraction(target: pd.DataFrame, model: str) -> float:
    if model == "best_single_validation":
        return float((target["best_single_selected_model"] != "persistence_raw").mean())
    if model == "guarded_best_validation":
        return float((target["guarded_best_selected_model"] != "persistence_raw").mean())
    if model == "residual_threshold_gate":
        return float((target["residual_threshold_selected_model"] != "persistence_raw").mean())
    if model == "residual_regime_selector":
        return float((target["residual_regime_selected_model"] != "persistence_raw").mean())
    if model in {"gefs_transfer_selected", "stack_transfer_selected", "monotonic_xgb_transfer"}:
        return 1.0
    return 0.0


def selector_detail(meta: dict[str, object]) -> str:
    if not meta:
        return ""
    if "direction" in meta:
        return (
            f"{meta.get('selected_name')} if {meta.get('direction')} "
            f"threshold={float(meta.get('threshold', np.nan)):.4f}"
        )
    if "selected_name" in meta:
        if "guard_passed" in meta:
            return (
                f"best={meta.get('selected_name')}; applied={meta.get('applied_name')}; "
                f"guard_passed={meta.get('guard_passed')}; "
                f"validation_delta_ci=[{float(meta.get('delta_bs_ci_low', np.nan)):.4f}, "
                f"{float(meta.get('delta_bs_ci_high', np.nan)):.4f}]"
            )
        return str(meta["selected_name"])
    if "choices" in meta:
        choices = dict(meta["choices"])  # type: ignore[arg-type]
        counts = pd.Series(list(choices.values())).value_counts().sort_index()
        return "; ".join(f"{name}:{int(count)}" for name, count in counts.items())
    return ""


def compact_summary(summary: pd.DataFrame) -> pd.DataFrame:
    return (
        summary.groupby(["transfer_mode", "model"], dropna=False)
        .agg(
            n_regions=("target_region", "nunique"),
            n_rows=("target_region", "size"),
            robust_positive_count=("bss_ci_low", lambda s: int((s > 0).sum())),
            robust_added_value_count=(
                "added_value_status_vs_raw_persistence",
                lambda s: int((s == "stack_robust_added_value").sum()),
            ),
            mean_bss_vs_climatology=("bss_vs_climatology", "mean"),
            median_bss_vs_climatology=("bss_vs_climatology", "median"),
            mean_bss_vs_raw_persistence=("bss_vs_raw_persistence", "mean"),
            median_bss_vs_raw_persistence=("bss_vs_raw_persistence", "median"),
            mean_delta_bs_vs_raw_persistence=("delta_bs_model_minus_raw_persistence", "mean"),
            mean_dynamic_fraction_test=("dynamic_fraction_test", "mean"),
        )
        .reset_index()
    )


def main() -> None:
    args = parse_args()
    OUT_DIR.mkdir(exist_ok=True)
    LANDSURFACE_DIR.mkdir(parents=True, exist_ok=True)
    PAPER_DIR.mkdir(parents=True, exist_ok=True)
    regions = [resolve_region(region).slug for region in args.regions]

    frames = []
    for region in regions:
        print(f"Loading region frame: {region}", flush=True)
        frames.append(load_region_frame(region, args))
    data = pd.concat(frames, ignore_index=True)
    val_all = data[
        (data["target_year"] >= args.validation_start_year)
        & (data["target_year"] <= args.validation_end_year)
    ].copy()
    test_all = data[
        (data["target_year"] >= args.test_start_year)
        & (data["target_year"] <= args.test_end_year)
    ].copy()
    if val_all.empty or test_all.empty:
        raise ValueError("Persistence-residual benchmark has empty validation or test data.")

    monthly_parts: list[pd.DataFrame] = []
    summary_parts: list[pd.DataFrame] = []
    regime_parts: list[pd.DataFrame] = []

    for target_region in regions:
        target_test = test_all.loc[test_all["region"].eq(target_region)].copy()
        if target_test.empty:
            raise ValueError(f"No test rows for target region={target_region}")
        for mode in ["local", "pooled", "leave_one_region_out"]:
            source_regions = transfer_source_regions(mode, target_region, regions)
            source_val = val_all.loc[val_all["region"].isin(source_regions)].copy()
            if source_val.empty:
                raise ValueError(f"No source validation rows for mode={mode}, target={target_region}")

            source_scored, target_scored, transfer_meta = fit_transfer_predictions(
                source_val,
                target_test,
                weight_steps=args.weight_steps,
                include_monotonic_xgb=not args.skip_monotonic_xgb,
            )
            source_scored = add_selector_features(source_scored)
            target_scored = add_selector_features(target_scored)
            candidates = candidate_columns(target_scored)

            best_meta = fit_best_single(source_scored, candidate_columns(source_scored))
            guarded_best_meta = fit_guarded_best(source_scored, candidate_columns(source_scored), args.n_bootstrap)
            threshold_meta = fit_threshold_gate(source_scored, candidate_columns(source_scored))
            regime_meta = fit_regime_selector(
                source_scored,
                candidate_columns(source_scored),
                min_regime_rows=args.min_regime_rows,
            )

            target_scored["best_single_validation_prob_dry"] = apply_best_single(target_scored, best_meta)
            target_scored["guarded_best_validation_prob_dry"] = apply_guarded_best(target_scored, guarded_best_meta)
            target_scored["residual_threshold_gate_prob_dry"] = apply_threshold_gate(target_scored, threshold_meta)
            regime_pred, regime_selected = apply_regime_selector(target_scored, candidates, regime_meta)
            target_scored["residual_regime_selector_prob_dry"] = regime_pred
            target_scored["residual_regime_selected_model"] = regime_selected
            target_scored["residual_threshold_selected_model"] = np.where(
                target_scored["residual_threshold_gate_prob_dry"].to_numpy(dtype=float)
                == target_scored[REFERENCE_COL].to_numpy(dtype=float),
                "persistence_raw",
                str(threshold_meta["selected_name"]),
            )
            target_scored["target_region"] = target_region
            target_scored["transfer_mode"] = mode
            target_scored["source_regions"] = " ".join(source_regions)
            target_scored["best_single_selected_model"] = str(best_meta["selected_name"])
            target_scored["guarded_best_selected_model"] = str(guarded_best_meta["applied_name"])
            target_scored["threshold_gate_selected_model_validation"] = str(threshold_meta["selected_name"])
            target_scored["threshold_gate_direction"] = str(threshold_meta["direction"])
            target_scored["threshold_gate_threshold"] = float(threshold_meta["threshold"])
            target_scored["selected_weight_gefs"] = transfer_meta.get("selected_weight_gefs", np.nan)
            target_scored["selected_weight_persistence"] = transfer_meta.get("selected_weight_persistence", np.nan)

            selector_meta = {
                "best_single_validation": best_meta,
                "guarded_best_validation": guarded_best_meta,
                "residual_threshold_gate": threshold_meta,
                "residual_regime_selector": regime_meta,
            }
            summary = score_models(
                target_region,
                mode,
                source_regions,
                target_scored,
                candidates,
                selector_meta,
                args.n_bootstrap,
            )
            summary_parts.append(summary)

            regime_table = pd.DataFrame(regime_meta["validation_rows"])
            if not regime_table.empty:
                regime_table.insert(0, "source_regions", " ".join(source_regions))
                regime_table.insert(0, "transfer_mode", mode)
                regime_table.insert(0, "target_region", target_region)
                regime_parts.append(regime_table)

            keep_cols = [
                "target_time",
                "target_year",
                "target_month",
                "target_region",
                "transfer_mode",
                "source_regions",
                "y_true_dry_frac",
                "clim_prob_dry",
                REFERENCE_COL,
                "persistence_transfer_selected_prob_dry",
                "gefs_transfer_selected_prob_dry",
                "stack_transfer_selected_prob_dry",
                "monotonic_xgb_transfer_prob_dry",
                "best_single_validation_prob_dry",
                "guarded_best_validation_prob_dry",
                "residual_threshold_gate_prob_dry",
                "residual_regime_selector_prob_dry",
                "best_single_selected_model",
                "guarded_best_selected_model",
                "residual_threshold_selected_model",
                "residual_regime_selected_model",
                "threshold_gate_selected_model_validation",
                "threshold_gate_direction",
                "threshold_gate_threshold",
                "forecast_rzsm_anom",
                "forecast_minus_persistence",
                "abs_forecast_persistence_gap",
                "selected_weight_gefs",
                "selected_weight_persistence",
            ]
            monthly_parts.append(target_scored[[c for c in keep_cols if c in target_scored.columns]].copy())

            gate = summary.loc[summary["model"].eq("residual_threshold_gate")].iloc[0]
            regime = summary.loc[summary["model"].eq("residual_regime_selector")].iloc[0]
            print(
                f"  target={target_region:<24} mode={mode:<20} "
                f"threshold_deltaBS={float(gate['delta_bs_model_minus_raw_persistence']):+.4f} "
                f"threshold_status={gate['added_value_status_vs_raw_persistence']} "
                f"regime_deltaBS={float(regime['delta_bs_model_minus_raw_persistence']):+.4f} "
                f"regime_status={regime['added_value_status_vs_raw_persistence']}",
                flush=True,
            )

    monthly = pd.concat(monthly_parts, ignore_index=True)
    summary = pd.concat(summary_parts, ignore_index=True)
    regimes = pd.concat(regime_parts, ignore_index=True) if regime_parts else pd.DataFrame()
    compact = compact_summary(summary)

    outputs = {
        "monthly": OUT_DIR / f"{args.out_prefix}_monthly_scores.csv",
        "summary": OUT_DIR / f"{args.out_prefix}_summary.csv",
        "compact": OUT_DIR / f"{args.out_prefix}_compact_summary.csv",
        "regimes": OUT_DIR / f"{args.out_prefix}_regime_choices.csv",
    }
    monthly.to_csv(outputs["monthly"], index=False)
    summary.to_csv(outputs["summary"], index=False)
    compact.to_csv(outputs["compact"], index=False)
    regimes.to_csv(outputs["regimes"], index=False)

    text_path = OUT_DIR / f"{args.out_prefix}_summary.txt"
    lines = [
        "GEFSv12 Persistence-Residual Land-Surface Benchmark",
        "=" * 72,
        f"Regions: {' '.join(regions)}",
        f"Validation years: {args.validation_start_year}-{args.validation_end_year}",
        f"Test years: {args.test_start_year}-{args.test_end_year}",
        f"Primary reference: {REFERENCE_COL}",
        "",
    ]
    for mode, part in compact.groupby("transfer_mode", sort=False):
        lines.append(f"{mode}:")
        for row in part.sort_values(["robust_added_value_count", "mean_delta_bs_vs_raw_persistence"], ascending=[False, True]).itertuples(index=False):
            lines.append(
                f"  {row.model}: robust-positive {int(row.robust_positive_count)}/{int(row.n_rows)}, "
                f"robust added value vs raw persistence {int(row.robust_added_value_count)}/{int(row.n_rows)}, "
                f"mean deltaBS {float(row.mean_delta_bs_vs_raw_persistence):+.4f}, "
                f"mean BSS {float(row.mean_bss_vs_climatology):+.3f}"
            )
        lines.append("")
    text_path.write_text("\n".join(lines), encoding="utf-8")

    print(text_path.read_text(encoding="utf-8"))
    print(f"Wrote monthly scores: {outputs['monthly']} rows={len(monthly):,}")
    print(f"Wrote summary: {outputs['summary']} rows={len(summary):,}")
    print(f"Wrote compact summary: {outputs['compact']} rows={len(compact):,}")
    print(f"Wrote regime choices: {outputs['regimes']} rows={len(regimes):,}")

    if args.copy_report:
        for path in list(outputs.values()) + [text_path]:
            shutil.copy2(path, LANDSURFACE_DIR / path.name)
        shutil.copy2(outputs["summary"], PAPER_DIR / "table32_landsurface_persistence_residual_selector.csv")
        shutil.copy2(outputs["compact"], PAPER_DIR / "table33_landsurface_persistence_residual_selector_compact.csv")
        print(f"Copied paper table: {PAPER_DIR / 'table32_landsurface_persistence_residual_selector.csv'}")
        print(f"Copied paper table: {PAPER_DIR / 'table33_landsurface_persistence_residual_selector_compact.csv'}")


if __name__ == "__main__":
    main()
