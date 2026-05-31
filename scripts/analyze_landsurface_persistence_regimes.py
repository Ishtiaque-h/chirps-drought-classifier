#!/usr/bin/env python
"""Persistence-regime diagnostics for land-surface drought forecasts.

This diagnostic asks when forecast-informed root-zone soil-moisture
probabilities add value beyond same-target land-memory persistence. It uses the
monthly outputs from ``run_landsurface_domain_transfer_benchmark.py`` and
reports paired Brier-score differences by season, antecedent dry-fraction
regime, forecast-anomaly sign, and memory/forecast agreement class.
"""

from __future__ import annotations

from argparse import ArgumentParser, Namespace
from pathlib import Path
import shutil

import numpy as np
import pandas as pd

from run_gefsv12_landsurface_stack_benchmark import LANDSURFACE_DIR, PROJECT_ROOT, added_value_status
from run_landsurface_forecast_benchmark import brier


REPORT_DIR = PROJECT_ROOT / "results" / "report"
PAPER_DIR = REPORT_DIR / "paper"
DEFAULT_MONTHLY = (
    REPORT_DIR
    / "landsurface"
    / "landsurface_gefsv12_rzsm_domain_transfer_day15_hindcastcal_monthly_scores.csv"
)

MODEL_COLUMNS = {
    "gefs_transfer_selected": "gefs_transfer_selected_prob_dry",
    "stack_transfer_selected": "stack_transfer_selected_prob_dry",
    "monotonic_xgb_transfer": "monotonic_xgb_transfer_prob_dry",
}
REFERENCE_COL = "persistence_transfer_selected_prob_dry"


def parse_args() -> Namespace:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--monthly-scores", type=Path, default=DEFAULT_MONTHLY)
    parser.add_argument(
        "--transfer-modes",
        nargs="+",
        default=["leave_one_region_out"],
        help="Transfer modes to analyze from the monthly score file.",
    )
    parser.add_argument("--n-bootstrap", type=int, default=2000)
    parser.add_argument("--min-group-months", type=int, default=6)
    parser.add_argument(
        "--out-prefix",
        default="landsurface_persistence_regime_diagnostics",
        help="Output prefix written under outputs/ and optionally results/report/landsurface/.",
    )
    parser.add_argument("--copy-report", action="store_true")
    return parser.parse_args()


def season(month: int) -> str:
    if month in {12, 1, 2}:
        return "DJF"
    if month in {3, 4, 5}:
        return "MAM"
    if month in {6, 7, 8}:
        return "JJA"
    return "SON"


def memory_regime(value: float) -> str:
    if not np.isfinite(value):
        return "missing"
    if value < 0.10:
        return "low_antecedent_dry_fraction"
    if value < 0.30:
        return "moderate_antecedent_dry_fraction"
    return "high_antecedent_dry_fraction"


def forecast_anomaly_regime(signal: float) -> str:
    if not np.isfinite(signal):
        return "missing"
    if signal > 0:
        return "forecast_drier_than_normal"
    if signal < 0:
        return "forecast_wetter_than_normal"
    return "forecast_near_normal"


def agreement_regime(memory: float, signal: float) -> str:
    if not np.isfinite(memory) or not np.isfinite(signal):
        return "missing"
    memory_dry = memory >= 0.20
    memory_wet = memory < 0.10
    forecast_dry = signal > 0
    forecast_wet = signal < 0
    if memory_dry and forecast_dry:
        return "memory_and_forecast_dry"
    if memory_wet and forecast_wet:
        return "memory_and_forecast_wet"
    if memory_dry and forecast_wet:
        return "memory_dry_forecast_wet"
    if memory_wet and forecast_dry:
        return "memory_wet_forecast_dry"
    return "moderate_or_mixed_memory"


def bootstrap_delta_bs(
    frame: pd.DataFrame,
    candidate_col: str,
    reference_col: str,
    n_bootstrap: int,
    seed: int,
) -> tuple[float, float]:
    y = frame["y_true_dry_frac"].to_numpy(dtype=float)
    cand = frame[candidate_col].to_numpy(dtype=float)
    ref = frame[reference_col].to_numpy(dtype=float)
    rng = np.random.default_rng(seed)
    idx = np.arange(len(frame))
    vals = np.full(n_bootstrap, np.nan, dtype=float)
    for i in range(n_bootstrap):
        sample = rng.choice(idx, size=len(idx), replace=True)
        vals[i] = brier(y[sample], cand[sample]) - brier(y[sample], ref[sample])
    lo, hi = np.nanquantile(vals, [0.025, 0.975])
    return float(lo), float(hi)


def prepare_frame(path: Path, transfer_modes: list[str]) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Monthly score file not found: {path}")
    df = pd.read_csv(path, parse_dates=["target_time"])
    required = {
        "target_region",
        "transfer_mode",
        "target_month",
        "y_true_dry_frac",
        "forecast_rzsm_anom",
        "persistence_raw_prob_dry",
        REFERENCE_COL,
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns in {path}: {missing}")
    df = df.loc[df["transfer_mode"].isin(transfer_modes)].copy()
    if df.empty:
        raise ValueError(f"No rows remain after transfer-mode filter: {transfer_modes}")
    df["season"] = df["target_month"].astype(int).map(season)
    df["gefs_dry_anomaly_signal"] = -df["forecast_rzsm_anom"].astype(float)
    df["antecedent_memory_regime"] = df["persistence_raw_prob_dry"].astype(float).map(memory_regime)
    df["forecast_anomaly_regime"] = df["gefs_dry_anomaly_signal"].map(forecast_anomaly_regime)
    df["memory_forecast_agreement"] = [
        agreement_regime(memory, signal)
        for memory, signal in zip(df["persistence_raw_prob_dry"], df["gefs_dry_anomaly_signal"])
    ]
    return df


def score_group(
    frame: pd.DataFrame,
    transfer_mode: str,
    target_region: str,
    group_type: str,
    group_value: str,
    model: str,
    candidate_col: str,
    min_group_months: int,
    n_bootstrap: int,
    seed: int,
) -> dict[str, object]:
    y = frame["y_true_dry_frac"].to_numpy(dtype=float)
    candidate = frame[candidate_col].to_numpy(dtype=float)
    reference = frame[REFERENCE_COL].to_numpy(dtype=float)
    bs_model = brier(y, candidate)
    bs_reference = brier(y, reference)
    delta = bs_model - bs_reference
    lo, hi = bootstrap_delta_bs(
        frame,
        candidate_col=candidate_col,
        reference_col=REFERENCE_COL,
        n_bootstrap=n_bootstrap,
        seed=seed,
    )
    status = added_value_status(delta, lo, hi)
    return {
        "transfer_mode": transfer_mode,
        "target_region": target_region,
        "group_type": group_type,
        "group_value": group_value,
        "model": model,
        "n_months": int(len(frame)),
        "small_sample_flag": bool(len(frame) < min_group_months),
        "mean_observed_dry_fraction": float(np.mean(y)),
        "mean_antecedent_dry_fraction": float(frame["persistence_raw_prob_dry"].mean()),
        "mean_gefs_dry_anomaly_signal": float(frame["gefs_dry_anomaly_signal"].mean()),
        "bs_model": bs_model,
        "bs_persistence_transfer_selected": bs_reference,
        "delta_bs_model_minus_persistence": delta,
        "delta_bs_ci_low": lo,
        "delta_bs_ci_high": hi,
        "bss_vs_persistence": 1.0 - bs_model / bs_reference if bs_reference > 0 else np.nan,
        "added_value_status": status,
        "interpretation": interpret_row(status, len(frame), min_group_months),
    }


def interpret_row(status: str, n_months: int, min_group_months: int) -> str:
    sample = " Small-group diagnostic; treat as suggestive." if n_months < min_group_months else ""
    if status == "stack_robust_added_value":
        return "Candidate has robust paired Brier-score improvement over persistence." + sample
    if status == "persistence_robustly_better":
        return "Same-target persistence is robustly better than the candidate." + sample
    if status == "candidate_better_uncertain":
        return "Candidate has lower Brier score than persistence, but the paired CI crosses zero." + sample
    if status == "persistence_better_uncertain":
        return "Persistence has lower Brier score than the candidate, but the paired CI crosses zero." + sample
    return "Candidate and persistence are effectively tied within uncertainty." + sample


def score_regimes(df: pd.DataFrame, args: Namespace) -> pd.DataFrame:
    group_specs = [
        ("overall", None),
        ("season", "season"),
        ("antecedent_memory_regime", "antecedent_memory_regime"),
        ("forecast_anomaly_regime", "forecast_anomaly_regime"),
        ("memory_forecast_agreement", "memory_forecast_agreement"),
    ]
    rows: list[dict[str, object]] = []
    seed = 3101
    for transfer_mode, mode_part in df.groupby("transfer_mode", sort=False):
        regions = ["all_regions", *sorted(mode_part["target_region"].unique())]
        for target_region in regions:
            region_part = mode_part if target_region == "all_regions" else mode_part.loc[
                mode_part["target_region"].eq(target_region)
            ]
            if region_part.empty:
                continue
            for group_type, group_col in group_specs:
                if group_col is None:
                    grouped = [("all", region_part)]
                else:
                    grouped = list(region_part.groupby(group_col, sort=True))
                for group_value, group_part in grouped:
                    if group_part.empty:
                        continue
                    for model, candidate_col in MODEL_COLUMNS.items():
                        if candidate_col not in group_part.columns or group_part[candidate_col].isna().all():
                            continue
                        rows.append(
                            score_group(
                                group_part.dropna(
                                    subset=[
                                        "y_true_dry_frac",
                                        candidate_col,
                                        REFERENCE_COL,
                                        "persistence_raw_prob_dry",
                                        "gefs_dry_anomaly_signal",
                                    ]
                                ),
                                transfer_mode=str(transfer_mode),
                                target_region=str(target_region),
                                group_type=group_type,
                                group_value=str(group_value),
                                model=model,
                                candidate_col=candidate_col,
                                min_group_months=args.min_group_months,
                                n_bootstrap=args.n_bootstrap,
                                seed=seed,
                            )
                        )
                        seed += 1
    return pd.DataFrame(rows)


def write_markdown_summary(summary: pd.DataFrame, path: Path) -> None:
    lines = [
        "# Land-Surface Persistence-Regime Diagnostic",
        "",
        "Paired Brier-score differences are computed as candidate minus same-target persistence; negative values mean the candidate improved over persistence.",
        "",
    ]
    focus = summary.loc[
        summary["target_region"].eq("all_regions")
        & summary["group_type"].eq("overall")
        & summary["model"].isin(["stack_transfer_selected", "monotonic_xgb_transfer"])
    ].copy()
    if not focus.empty:
        lines.append("## Overall")
        for row in focus.sort_values(["transfer_mode", "model"]).itertuples(index=False):
            lines.append(
                f"- {row.transfer_mode} {row.model}: delta BS {row.delta_bs_model_minus_persistence:+.4f} "
                f"(95% CI {row.delta_bs_ci_low:+.4f} to {row.delta_bs_ci_high:+.4f}); "
                f"{row.added_value_status}."
            )
        lines.append("")

    agreement = summary.loc[
        summary["target_region"].eq("all_regions")
        & summary["group_type"].eq("memory_forecast_agreement")
        & summary["model"].eq("stack_transfer_selected")
    ].copy()
    if not agreement.empty:
        lines.append("## Memory/Forecast Agreement")
        for row in agreement.sort_values("group_value").itertuples(index=False):
            lines.append(
                f"- {row.group_value}: n={row.n_months}, delta BS {row.delta_bs_model_minus_persistence:+.4f} "
                f"(95% CI {row.delta_bs_ci_low:+.4f} to {row.delta_bs_ci_high:+.4f}); "
                f"{row.added_value_status}."
            )
        lines.append("")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    out_dir = PROJECT_ROOT / "outputs"
    out_dir.mkdir(exist_ok=True)
    LANDSURFACE_DIR.mkdir(parents=True, exist_ok=True)
    PAPER_DIR.mkdir(parents=True, exist_ok=True)

    df = prepare_frame(args.monthly_scores, args.transfer_modes)
    summary = score_regimes(df, args)
    if summary.empty:
        raise SystemExit("No persistence-regime rows were produced.")

    summary = summary.sort_values(
        ["transfer_mode", "target_region", "group_type", "group_value", "model"]
    ).reset_index(drop=True)
    out_csv = out_dir / f"{args.out_prefix}.csv"
    out_md = out_dir / f"{args.out_prefix}.md"
    summary.to_csv(out_csv, index=False)
    write_markdown_summary(summary, out_md)

    if args.copy_report:
        report_csv = LANDSURFACE_DIR / out_csv.name
        report_md = LANDSURFACE_DIR / out_md.name
        paper_csv = PAPER_DIR / "table16_landsurface_persistence_regimes.csv"
        shutil.copy2(out_csv, report_csv)
        shutil.copy2(out_md, report_md)
        shutil.copy2(out_csv, paper_csv)

    print(out_md.read_text(encoding="utf-8"))
    print(f"Wrote persistence-regime summary: {out_csv} rows={len(summary):,}")
    if args.copy_report:
        print(f"Copied report table: {LANDSURFACE_DIR / out_csv.name}")
        print(f"Copied paper table: {PAPER_DIR / 'table16_landsurface_persistence_regimes.csv'}")


if __name__ == "__main__":
    main()
