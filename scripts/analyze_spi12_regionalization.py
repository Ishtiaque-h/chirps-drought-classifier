#!/usr/bin/env python
"""
SPI-12 regionalization and drought run-theory diagnostics.

This diagnostic script is intentionally separate from the frozen SPI-1 lead-1
forecasting pipeline. The forecast dataset remains built from SPI-1/3/6 and
precipitation lags; SPI-12 is computed here only to characterize longer-term
hydroclimatic zones and drought-run behavior inside a region.

Methodological consistency with make_spi_labels.py:
  - 1991-2020 default climatological baseline.
  - gamma distribution fitted separately by calendar month and grid cell.
  - zero-precipitation probability handled before normal-quantile transform.
  - trailing accumulation window is computed before the SPI transform.

Typical usage:
  python scripts/analyze_spi12_regionalization.py --region cvalley --mask-kind basin --n-jobs 8
  python scripts/analyze_spi12_regionalization.py --region southern_great_plains --n-clusters 4 --n-jobs 8

Outputs:
  outputs/regionalization/<run_slug>/spi12_regionalization.nc
  results/regionalization/<run_slug>/zone_map.png
  results/regionalization/<run_slug>/zone_spi12_timeseries.csv
  results/regionalization/<run_slug>/zone_drought_runs.csv
  results/regionalization/<run_slug>/zone_run_metrics.csv
  results/regionalization/<run_slug>/zone_climate_index_correlations.csv
  results/regionalization/<run_slug>/regionalization_method_notes.txt
"""
from __future__ import annotations

from argparse import ArgumentParser, Namespace
from datetime import datetime, timezone
from pathlib import Path
import json
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from joblib import Parallel, delayed
from scipy.stats import gamma as gamma_dist, norm, pearsonr
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from region_config import REGIONS, Region, region_table, resolve_region


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED = PROJECT_ROOT / "data" / "processed"
REGION_DATA_ROOT = PROCESSED / "regions"
CLIMATE_FILE = PROCESSED / "climate_indices_monthly.csv"
OUT_ROOT = PROJECT_ROOT / "outputs" / "regionalization"
REPORT_ROOT = PROJECT_ROOT / "results" / "regionalization"

DEFAULT_START_YEAR = 1991
DEFAULT_END_YEAR = 2026
DEFAULT_BASELINE_START_YEAR = 1991
DEFAULT_BASELINE_END_YEAR = 2020
SPI_WINDOW = 12
MISSING_SENTINELS = (-9.9, -99.99, -999.0)


def write_netcdf_with_optional_compression(
    ds: xr.Dataset,
    out_file: Path,
    encoding: dict[str, dict[str, object]] | None = None,
) -> None:
    """
    Write NetCDF with compression when the active backend supports it.

    The project environment includes netCDF4, so compressed writes should work
    in normal runs. This fallback keeps smoke tests usable in lightweight
    environments where xarray falls back to the scipy backend.
    """
    try:
        ds.to_netcdf(out_file, encoding=encoding or {})
    except ValueError as exc:
        if "unexpected encoding for scipy backend" not in str(exc):
            raise
        print("Active NetCDF backend does not support compression; writing without compression.")
        ds.to_netcdf(out_file)


def parse_args() -> Namespace:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--region", default="cvalley", help="Configured region slug or alias.")
    parser.add_argument("--list-regions", action="store_true", help="List configured regions and exit.")
    parser.add_argument("--start-year", type=int, default=DEFAULT_START_YEAR)
    parser.add_argument("--end-year", type=int, default=DEFAULT_END_YEAR)
    parser.add_argument("--baseline-start-year", type=int, default=DEFAULT_BASELINE_START_YEAR)
    parser.add_argument("--baseline-end-year", type=int, default=DEFAULT_BASELINE_END_YEAR)
    parser.add_argument(
        "--pr-file",
        type=Path,
        default=None,
        help="Optional monthly precipitation NetCDF. Defaults to the configured regional CHIRPS file.",
    )
    parser.add_argument(
        "--mask-kind",
        choices=["none", "country", "basin"],
        default="none",
        help="Optional existing grid mask to apply before SPI-12 regionalization.",
    )
    parser.add_argument(
        "--mask-file",
        type=Path,
        default=None,
        help="Optional mask NetCDF. Defaults to data/processed/regions/<region>/masks/<region>_<mask-kind>_mask.nc.",
    )
    parser.add_argument(
        "--mask-var",
        default=None,
        help="Optional mask variable name. Defaults to country_mask or basin_mask from --mask-kind.",
    )
    parser.add_argument("--n-clusters", type=int, default=4, help="Number of k-means zones.")
    parser.add_argument("--n-components", type=int, default=3, help="PCA components before k-means.")
    parser.add_argument(
        "--min-valid-fraction",
        type=float,
        default=0.85,
        help="Minimum valid SPI-12 fraction required for a pixel to be clustered.",
    )
    parser.add_argument(
        "--drought-threshold",
        type=float,
        default=-1.0,
        help="Zone-mean SPI-12 threshold defining drought runs.",
    )
    parser.add_argument("--n-jobs", type=int, default=1, help="Parallel jobs for SPI-12 fitting.")
    parser.add_argument(
        "--index-file",
        type=Path,
        default=CLIMATE_FILE,
        help="Climate index CSV with a monthly time column. Defaults to data/processed/climate_indices_monthly.csv.",
    )
    parser.add_argument(
        "--index-columns",
        nargs="+",
        default=["nino34", "soi", "pdo"],
        help="Climate-index columns to correlate with zone-mean SPI-12 if present.",
    )
    parser.add_argument(
        "--index-lags",
        type=int,
        nargs="+",
        default=[0, 1, 3, 6],
        help="Positive lags mean the climate index leads zone SPI-12 by that many months.",
    )
    parser.add_argument(
        "--recompute",
        action="store_true",
        help="Recompute SPI-12 and regionalization even if output NetCDF already exists.",
    )
    return parser.parse_args()


def default_pr_file(region: Region, start_year: int, end_year: int) -> Path:
    if region.slug == "cvalley":
        return PROCESSED / f"chirps_v3_monthly_cvalley_{start_year}_{end_year}.nc"
    return REGION_DATA_ROOT / region.slug / f"chirps_v3_monthly_{region.slug}_{start_year}_{end_year}.nc"


def default_mask_file(region: Region, mask_kind: str) -> Path:
    return REGION_DATA_ROOT / region.slug / "masks" / f"{region.slug}_{mask_kind}_mask.nc"


def run_slug(region: Region, mask_kind: str) -> str:
    return region.slug if mask_kind == "none" else f"{region.slug}_{mask_kind}_masked"


def normalize_pr_dataarray(ds: xr.Dataset) -> xr.DataArray:
    if "pr" in ds.data_vars:
        pr = ds["pr"]
    else:
        var = list(ds.data_vars)[0]
        print(f"No 'pr' variable found; using first data variable: {var}")
        pr = ds[var]
    rename = {}
    if "lat" in pr.coords:
        rename["lat"] = "latitude"
    if "lon" in pr.coords:
        rename["lon"] = "longitude"
    if rename:
        pr = pr.rename(rename)
    required = {"time", "latitude", "longitude"}
    missing = required.difference(pr.dims)
    if missing:
        raise ValueError(f"Precipitation variable must have dimensions {sorted(required)}; missing {sorted(missing)}")
    return pr.sortby("time")


def load_grid_mask(mask_file: Path, pr: xr.DataArray, mask_var: str) -> xr.DataArray:
    if not mask_file.exists():
        raise FileNotFoundError(
            f"Mask file not found: {mask_file}. Run scripts/build_region_masks.py or "
            "scripts/build_basin_masks.py first, or pass --mask-file."
        )
    mask_ds = xr.open_dataset(mask_file)
    try:
        if mask_var not in mask_ds:
            raise KeyError(f"{mask_file} does not contain mask variable '{mask_var}'")
        mask = mask_ds[mask_var]
        rename = {}
        if "lat" in mask.coords:
            rename["lat"] = "latitude"
        if "lon" in mask.coords:
            rename["lon"] = "longitude"
        if rename:
            mask = mask.rename(rename)
        try:
            mask = mask.sel(latitude=pr["latitude"], longitude=pr["longitude"])
        except Exception as exc:
            raise ValueError(f"Mask grid is not aligned with the precipitation grid: {mask_file}") from exc
        return mask.astype(bool).load()
    finally:
        mask_ds.close()


def rolling_sum_1d(series: np.ndarray, window: int) -> np.ndarray:
    out = np.full(len(series), np.nan, dtype=np.float64)
    for t in range(window - 1, len(series)):
        block = series[t - window + 1 : t + 1]
        if np.any(np.isnan(block)):
            continue
        out[t] = np.sum(block)
    return out


def precip_to_spi(series: np.ndarray, baseline_mask: np.ndarray) -> np.ndarray:
    """
    Match make_spi_labels.py SPI convention for one calendar-month subset:
    fit gamma to nonzero baseline values, include p_zero, and normal-quantile transform.
    """
    base = series[baseline_mask]
    base = base[~np.isnan(base)]
    spi = np.full(len(series), np.nan, dtype=np.float32)

    if len(base) < 10:
        return spi

    p_zero = np.mean(base == 0)
    nonzero = base[base > 0]
    if len(nonzero) < 5:
        return spi

    fit_alpha, _, fit_beta = gamma_dist.fit(nonzero, floc=0)

    for i, val in enumerate(series):
        if np.isnan(val):
            continue
        if val == 0:
            cdf_val = p_zero
        else:
            cdf_val = p_zero + (1.0 - p_zero) * gamma_dist.cdf(val, fit_alpha, scale=fit_beta)
        cdf_val = np.clip(cdf_val, 1e-6, 1.0 - 1e-6)
        spi[i] = norm.ppf(cdf_val)

    return spi


def compute_spi12_for_pixel(
    series: np.ndarray,
    months_arr: np.ndarray,
    baseline_mask: np.ndarray,
) -> np.ndarray:
    rolled = rolling_sum_1d(series.astype(np.float64), SPI_WINDOW)
    spi12 = np.full(len(series), np.nan, dtype=np.float32)

    for month in range(1, 13):
        month_idx = np.where(months_arr == month)[0]
        base_idx = np.where(baseline_mask & (months_arr == month))[0]
        mask_for_month = np.zeros(len(series), dtype=bool)
        mask_for_month[base_idx] = True
        base_within_month = mask_for_month[month_idx]
        spi12[month_idx] = precip_to_spi(rolled[month_idx], base_within_month)

    return spi12


def compute_spi12_grid(
    pr: xr.DataArray,
    baseline_start_year: int,
    baseline_end_year: int,
    n_jobs: int,
) -> xr.DataArray:
    pr_vals = pr.values.astype(np.float64)
    times = pd.DatetimeIndex(pr["time"].values)
    months_arr = times.month.to_numpy()
    baseline_mask = (times.year >= baseline_start_year) & (times.year <= baseline_end_year)

    if baseline_mask.sum() < 120:
        raise ValueError(
            f"Baseline {baseline_start_year}-{baseline_end_year} has only {int(baseline_mask.sum())} months. "
            "SPI fitting requires a multi-year climatology."
        )

    ntimes, nlat, nlon = pr_vals.shape
    n_pixels = nlat * nlon
    flat = pr_vals.reshape(ntimes, n_pixels).T

    print(
        f"Computing diagnostic SPI-12 for {n_pixels:,} pixels, {ntimes} months, "
        f"baseline={baseline_start_year}-{baseline_end_year}, n_jobs={n_jobs}"
    )

    if n_jobs > 1:
        results = Parallel(n_jobs=n_jobs, verbose=10, batch_size=64)(
            delayed(compute_spi12_for_pixel)(flat[i], months_arr, baseline_mask)
            for i in range(n_pixels)
        )
    else:
        results = []
        for i in range(n_pixels):
            if i and i % 1000 == 0:
                print(f"  SPI-12 pixels done: {i:,}/{n_pixels:,}")
            results.append(compute_spi12_for_pixel(flat[i], months_arr, baseline_mask))

    spi_vals = np.asarray(results, dtype=np.float32).T.reshape(ntimes, nlat, nlon)
    spi12 = xr.DataArray(
        spi_vals,
        coords={"time": pr["time"], "latitude": pr["latitude"], "longitude": pr["longitude"]},
        dims=("time", "latitude", "longitude"),
        name="spi12",
        attrs={
            "long_name": "SPI-12 diagnostic drought-memory index",
            "units": "dimensionless",
            "window_months": SPI_WINDOW,
            "baseline_years": f"{baseline_start_year}-{baseline_end_year}",
            "note": "Diagnostic-only variable; not part of the frozen SPI-1 lead-1 forecast target.",
        },
    )
    return spi12


def cluster_spi12(
    spi12: xr.DataArray,
    n_clusters: int,
    n_components: int,
    min_valid_fraction: float,
) -> tuple[xr.DataArray, pd.DataFrame, pd.DataFrame]:
    stacked = spi12.stack(pixel=("latitude", "longitude"))
    X = stacked.transpose("pixel", "time").values.astype(np.float64)

    valid_counts = np.isfinite(X).sum(axis=1)
    min_valid = int(np.ceil(min_valid_fraction * X.shape[1]))
    std = np.nanstd(X, axis=1)
    valid_pixels = (valid_counts >= min_valid) & np.isfinite(std) & (std > 0)

    n_valid = int(valid_pixels.sum())
    if n_valid < n_clusters:
        raise ValueError(
            f"Only {n_valid} pixels meet min_valid_fraction={min_valid_fraction}; "
            f"cannot fit {n_clusters} clusters."
        )

    X_valid = X[valid_pixels].copy()
    row_mean = np.nanmean(X_valid, axis=1)
    missing = ~np.isfinite(X_valid)
    if missing.any():
        row_idx, _ = np.where(missing)
        X_valid[missing] = row_mean[row_idx]

    row_std = X_valid.std(axis=1)
    row_std[row_std == 0] = 1.0
    X_scaled = (X_valid - row_mean[:, None]) / row_std[:, None]

    n_components_actual = max(1, min(n_components, X_scaled.shape[0], X_scaled.shape[1]))
    pca = PCA(n_components=n_components_actual, random_state=42)
    X_pca = pca.fit_transform(X_scaled)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=20)
    labels_valid = kmeans.fit_predict(X_pca)

    full_labels = np.full(stacked.sizes["pixel"], -1, dtype=np.int16)
    full_labels[valid_pixels] = labels_valid.astype(np.int16)

    zones = xr.DataArray(
        full_labels,
        coords={"pixel": stacked["pixel"]},
        dims=("pixel",),
        name="zone",
        attrs={
            "description": "K-means hydroclimate zones from standardized SPI-12 time series after PCA",
            "n_clusters": n_clusters,
            "n_pca_components": n_components_actual,
            "min_valid_fraction": min_valid_fraction,
        },
    ).unstack("pixel")
    zones = zones.transpose("latitude", "longitude")

    pca_df = pd.DataFrame(
        {
            "component": np.arange(1, n_components_actual + 1),
            "explained_variance_ratio": pca.explained_variance_ratio_,
            "cumulative_explained_variance_ratio": np.cumsum(pca.explained_variance_ratio_),
        }
    )

    zone_counts = pd.Series(labels_valid).value_counts().sort_index()
    count_df = pd.DataFrame(
        {
            "zone": zone_counts.index.astype(int),
            "n_pixels": zone_counts.values.astype(int),
            "pixel_fraction": zone_counts.values / zone_counts.values.sum(),
        }
    )

    print(
        f"Clustered {n_valid:,} valid pixels into {n_clusters} zones; "
        f"PCA components={n_components_actual}, cumulative explained variance="
        f"{pca_df['cumulative_explained_variance_ratio'].iloc[-1]:.3f}"
    )
    return zones, pca_df, count_df


def zone_spi12_timeseries(spi12: xr.DataArray, zones: xr.DataArray) -> pd.DataFrame:
    rows = []
    zone_ids = sorted(int(z) for z in np.unique(zones.values) if z >= 0)

    for zone in zone_ids:
        mask = zones == zone
        n_pixels = int(mask.sum())
        series = spi12.where(mask).mean(dim=("latitude", "longitude"), skipna=True)
        tmp = pd.DataFrame(
            {
                "time": pd.to_datetime(series["time"].values).to_period("M").to_timestamp(),
                "zone": zone,
                "n_pixels": n_pixels,
                "spi12_mean": series.values.astype(float),
            }
        )
        rows.append(tmp)

    return pd.concat(rows, ignore_index=True)


def find_drought_runs(values: np.ndarray, times: np.ndarray, threshold: float) -> list[dict[str, object]]:
    runs = []
    start = None

    def is_drought_value(value: float) -> bool:
        return np.isfinite(value) and value <= threshold

    for i, value in enumerate(values):
        if is_drought_value(float(value)):
            if start is None:
                start = i
        else:
            if start is not None:
                runs.append(run_record(values, times, start, i, threshold))
                start = None

    if start is not None:
        runs.append(run_record(values, times, start, len(values), threshold))

    return runs


def run_record(
    values: np.ndarray,
    times: np.ndarray,
    start: int,
    end_exclusive: int,
    threshold: float,
) -> dict[str, object]:
    block = values[start:end_exclusive].astype(float)
    severity = float(np.nansum(threshold - block))
    duration = int(end_exclusive - start)
    return {
        "start_time": pd.Timestamp(times[start]).to_period("M").to_timestamp(),
        "end_time": pd.Timestamp(times[end_exclusive - 1]).to_period("M").to_timestamp(),
        "duration_months": duration,
        "severity_threshold_deficit": severity,
        "mean_intensity": severity / duration if duration else np.nan,
        "minimum_spi12": float(np.nanmin(block)) if duration else np.nan,
    }


def run_theory_tables(zone_ts: pd.DataFrame, threshold: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    run_rows = []
    metric_rows = []

    for zone, sub in zone_ts.groupby("zone", sort=True):
        sub = sub.sort_values("time")
        runs = find_drought_runs(
            sub["spi12_mean"].to_numpy(dtype=float),
            sub["time"].to_numpy(),
            threshold,
        )

        for idx, run in enumerate(runs, start=1):
            run_rows.append({"zone": int(zone), "run_id": idx, **run})

        if runs:
            duration = np.array([r["duration_months"] for r in runs], dtype=float)
            severity = np.array([r["severity_threshold_deficit"] for r in runs], dtype=float)
            intensity = np.array([r["mean_intensity"] for r in runs], dtype=float)
            minimum = np.array([r["minimum_spi12"] for r in runs], dtype=float)
            metric_rows.append(
                {
                    "zone": int(zone),
                    "n_runs": len(runs),
                    "mean_duration_months": float(np.mean(duration)),
                    "median_duration_months": float(np.median(duration)),
                    "max_duration_months": int(np.max(duration)),
                    "mean_severity_threshold_deficit": float(np.mean(severity)),
                    "median_severity_threshold_deficit": float(np.median(severity)),
                    "max_severity_threshold_deficit": float(np.max(severity)),
                    "mean_intensity": float(np.mean(intensity)),
                    "median_intensity": float(np.median(intensity)),
                    "min_spi12_during_runs": float(np.min(minimum)),
                }
            )
        else:
            metric_rows.append(
                {
                    "zone": int(zone),
                    "n_runs": 0,
                    "mean_duration_months": np.nan,
                    "median_duration_months": np.nan,
                    "max_duration_months": np.nan,
                    "mean_severity_threshold_deficit": np.nan,
                    "median_severity_threshold_deficit": np.nan,
                    "max_severity_threshold_deficit": np.nan,
                    "mean_intensity": np.nan,
                    "median_intensity": np.nan,
                    "min_spi12_during_runs": np.nan,
                }
            )

    return pd.DataFrame(run_rows), pd.DataFrame(metric_rows)


def climate_index_correlations(
    zone_ts: pd.DataFrame,
    index_file: Path,
    index_columns: list[str],
    index_lags: list[int],
) -> pd.DataFrame:
    if not index_file.exists():
        print(f"Climate index file not found; skipping correlations: {index_file}")
        return pd.DataFrame()

    cdf = pd.read_csv(index_file)
    if "time" not in cdf.columns:
        raise ValueError(f"Climate index file must contain a 'time' column: {index_file}")

    cdf["time"] = pd.to_datetime(cdf["time"]).dt.to_period("M").dt.to_timestamp()
    cdf = cdf.sort_values("time").drop_duplicates("time", keep="last").set_index("time")
    cdf = cdf.replace(list(MISSING_SENTINELS), np.nan)

    requested = [col for col in index_columns if col in cdf.columns]
    missing = sorted(set(index_columns).difference(cdf.columns))
    if missing:
        print(f"Skipping missing climate-index columns: {missing}")
    if not requested:
        print("No requested climate-index columns are present; skipping correlations.")
        return pd.DataFrame()

    time_index = pd.DatetimeIndex(sorted(zone_ts["time"].unique()))
    cdf = cdf.reindex(time_index).sort_index()
    cdf[requested] = cdf[requested].interpolate(method="time", limit_area="inside")

    rows = []
    for zone, sub in zone_ts.groupby("zone", sort=True):
        sub = sub.sort_values("time").set_index("time").reindex(time_index)
        spi = sub["spi12_mean"].to_numpy(dtype=float)

        for col in requested:
            for lag in index_lags:
                idx = cdf[col].shift(lag).to_numpy(dtype=float)
                valid = np.isfinite(spi) & np.isfinite(idx)
                n = int(valid.sum())
                if n >= 3 and np.nanstd(spi[valid]) > 0 and np.nanstd(idx[valid]) > 0:
                    r, p = pearsonr(idx[valid], spi[valid])
                    r = float(r)
                    p = float(p)
                else:
                    r = np.nan
                    p = np.nan
                rows.append(
                    {
                        "zone": int(zone),
                        "index": col,
                        "index_lag_months": int(lag),
                        "pearson_r": r,
                        "p_value": p,
                        "n_months": n,
                        "interpretation": "positive means higher index values are associated with wetter SPI-12",
                    }
                )

    return pd.DataFrame(rows)


def plot_zone_map(zones: xr.DataArray, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    plot_da = zones.where(zones >= 0)
    im = ax.pcolormesh(
        zones["longitude"],
        zones["latitude"],
        plot_da,
        shading="auto",
        cmap="tab10",
        vmin=-0.5,
        vmax=float(np.nanmax(plot_da.values)) + 0.5 if np.isfinite(plot_da.values).any() else 1.0,
    )
    zone_ids = sorted(int(z) for z in np.unique(zones.values) if z >= 0)
    cbar = fig.colorbar(im, ax=ax, ticks=zone_ids)
    cbar.set_label("SPI-12 zone")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title("SPI-12 PCA + k-means regionalization")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_zone_timeseries(zone_ts: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 5))
    for zone, sub in zone_ts.groupby("zone", sort=True):
        ax.plot(sub["time"], sub["spi12_mean"], linewidth=1.4, label=f"Zone {zone}")
    ax.axhline(-1.0, linestyle="--", linewidth=1, label="SPI-12 = -1")
    ax.axhline(0.0, linewidth=0.8)
    ax.set_xlabel("Month")
    ax.set_ylabel("Zone mean SPI-12")
    ax.set_title("Zone-mean SPI-12 time series")
    ax.grid(alpha=0.2)
    ax.legend(ncol=2, fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_run_summary(metrics: pd.DataFrame, out_path: Path) -> None:
    if metrics.empty:
        return
    fig, ax = plt.subplots(figsize=(8, 4.8))
    x = np.arange(len(metrics))
    labels = [f"Zone {int(z)}" for z in metrics["zone"]]
    ax.bar(x, metrics["mean_duration_months"].fillna(0.0).to_numpy())
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=0)
    ax.set_ylabel("Mean drought-run duration (months)")
    ax.set_title("SPI-12 run-theory duration by zone")
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def write_method_notes(
    out_path: Path,
    region: Region,
    args: Namespace,
    pr_file: Path,
    mask_file: Path | None,
    pca_df: pd.DataFrame,
    zone_counts: pd.DataFrame,
) -> None:
    lines = [
        "SPI-12 Regionalization Diagnostics",
        "=" * 72,
        "",
        "Purpose:",
        (
            "This analysis is diagnostic-only. It does not change the frozen SPI-1 lead-1 "
            "forecast target, feature table, model checkpoint, calibration protocol, or "
            "multi-region BSS evaluation."
        ),
        "",
        "Region:",
        f"  Slug: {region.slug}",
        f"  Name: {region.name}",
        f"  Rationale: {region.rationale}",
        f"  Precipitation file: {pr_file}",
        f"  Mask file: {mask_file if mask_file is not None else 'none'}",
        "",
        "SPI-12 computation:",
        f"  Accumulation window: {SPI_WINDOW} months",
        f"  Baseline years: {args.baseline_start_year}-{args.baseline_end_year}",
        "  Gamma fitting: per pixel x calendar month, matching make_spi_labels.py convention",
        "  Zero handling: empirical p_zero plus gamma CDF for nonzero precipitation",
        "",
        "Regionalization:",
        f"  K-means zones: {args.n_clusters}",
        f"  PCA components requested: {args.n_components}",
        f"  PCA components used: {len(pca_df)}",
        (
            "  Cumulative explained variance: "
            f"{pca_df['cumulative_explained_variance_ratio'].iloc[-1]:.4f}"
            if not pca_df.empty
            else "  Cumulative explained variance: unavailable"
        ),
        f"  Minimum valid fraction: {args.min_valid_fraction}",
        "",
        "Zone pixel counts:",
    ]
    for row in zone_counts.itertuples(index=False):
        lines.append(f"  Zone {int(row.zone)}: {int(row.n_pixels):,} pixels ({row.pixel_fraction:.3%})")
    lines.extend(
        [
            "",
            "Run theory:",
            f"  Drought threshold: zone-mean SPI-12 <= {args.drought_threshold}",
            "  Severity definition: sum(threshold - SPI-12) across drought months",
            "  Intensity definition: severity / duration",
            "",
            "Climate-index correlations:",
            f"  Index file: {args.index_file}",
            f"  Requested columns: {args.index_columns}",
            f"  Positive lags mean the climate index leads SPI-12: {args.index_lags}",
            "",
            f"Created UTC: {datetime.now(timezone.utc).isoformat()}",
        ]
    )
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)

    args = parse_args()
    if args.list_regions:
        print(region_table())
        return

    region = resolve_region(args.region)
    this_run = run_slug(region, args.mask_kind)
    pr_file = args.pr_file or default_pr_file(region, args.start_year, args.end_year)

    if args.n_clusters < 2:
        raise ValueError("--n-clusters must be at least 2")
    if args.n_components < 1:
        raise ValueError("--n-components must be at least 1")
    if not 0 < args.min_valid_fraction <= 1:
        raise ValueError("--min-valid-fraction must be in (0, 1]")
    if args.mask_kind != "none" and args.mask_file is None:
        args.mask_file = default_mask_file(region, args.mask_kind)
    mask_var = args.mask_var or (f"{args.mask_kind}_mask" if args.mask_kind != "none" else None)

    out_dir = OUT_ROOT / this_run
    report_dir = REPORT_ROOT / this_run
    out_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    diag_nc = out_dir / "spi12_regionalization.nc"

    if diag_nc.exists() and not args.recompute:
        print(f"Using existing diagnostic NetCDF: {diag_nc}")
        diag = xr.open_dataset(diag_nc).load()
        spi12 = diag["spi12"]
        zones = diag["zone"]
        pca_df = pd.read_csv(report_dir / "pca_explained_variance.csv")
        zone_counts = pd.read_csv(report_dir / "zone_pixel_counts.csv")
    else:
        if not pr_file.exists():
            raise FileNotFoundError(
                f"Precipitation file not found: {pr_file}. Run the multiregion preparation first "
                "or pass --pr-file."
            )

        print(f"Loading precipitation: {pr_file}")
        pr_ds = xr.open_dataset(pr_file).load()
        try:
            pr = normalize_pr_dataarray(pr_ds)

            mask = None
            if args.mask_kind != "none":
                print(f"Applying {args.mask_kind} mask: {args.mask_file}")
                mask = load_grid_mask(args.mask_file, pr, mask_var)
                n_keep = int(mask.sum())
                n_total = int(mask.size)
                print(f"Mask retains {n_keep:,}/{n_total:,} grid cells ({n_keep / n_total:.2%})")
                pr = pr.where(mask)

            spi12 = compute_spi12_grid(
                pr=pr,
                baseline_start_year=args.baseline_start_year,
                baseline_end_year=args.baseline_end_year,
                n_jobs=args.n_jobs,
            )
            zones, pca_df, zone_counts = cluster_spi12(
                spi12=spi12,
                n_clusters=args.n_clusters,
                n_components=args.n_components,
                min_valid_fraction=args.min_valid_fraction,
            )

            out_ds = xr.Dataset(
                {
                    "spi12": spi12.astype("float32"),
                    "zone": zones.astype("int16"),
                },
                attrs={
                    "region": region.slug,
                    "region_name": region.name,
                    "run_slug": this_run,
                    "diagnostic_only": "true",
                    "created_utc": datetime.now(timezone.utc).isoformat(),
                    "precipitation_file": str(pr_file.relative_to(PROJECT_ROOT) if pr_file.is_relative_to(PROJECT_ROOT) else pr_file),
                    "mask_kind": args.mask_kind,
                    "mask_file": str(args.mask_file.relative_to(PROJECT_ROOT) if args.mask_file and args.mask_file.is_relative_to(PROJECT_ROOT) else args.mask_file),
                    "baseline_years": f"{args.baseline_start_year}-{args.baseline_end_year}",
                },
            )
            encoding = {
                "spi12": {"zlib": True, "complevel": 4},
                "zone": {"zlib": True, "complevel": 4, "dtype": "int16"},
            }
            write_netcdf_with_optional_compression(out_ds, diag_nc, encoding=encoding)
            print(f"Wrote diagnostic NetCDF: {diag_nc}")
        finally:
            pr_ds.close()

    zone_ts = zone_spi12_timeseries(spi12, zones)
    drought_runs, run_metrics = run_theory_tables(zone_ts, threshold=args.drought_threshold)
    corr_df = climate_index_correlations(
        zone_ts=zone_ts,
        index_file=args.index_file,
        index_columns=args.index_columns,
        index_lags=args.index_lags,
    )

    zone_ts_path = report_dir / "zone_spi12_timeseries.csv"
    drought_runs_path = report_dir / "zone_drought_runs.csv"
    run_metrics_path = report_dir / "zone_run_metrics.csv"
    pca_path = report_dir / "pca_explained_variance.csv"
    zone_counts_path = report_dir / "zone_pixel_counts.csv"
    corr_path = report_dir / "zone_climate_index_correlations.csv"
    metadata_path = report_dir / "regionalization_metadata.json"
    notes_path = report_dir / "regionalization_method_notes.txt"

    zone_ts.to_csv(zone_ts_path, index=False)
    drought_runs.to_csv(drought_runs_path, index=False)
    run_metrics.to_csv(run_metrics_path, index=False)
    pca_df.to_csv(pca_path, index=False)
    zone_counts.to_csv(zone_counts_path, index=False)
    corr_df.to_csv(corr_path, index=False)

    plot_zone_map(zones, report_dir / "zone_map.png")
    plot_zone_timeseries(zone_ts, report_dir / "zone_spi12_timeseries.png")
    plot_run_summary(run_metrics, report_dir / "zone_run_duration_summary.png")

    write_method_notes(
        notes_path,
        region=region,
        args=args,
        pr_file=pr_file,
        mask_file=args.mask_file if args.mask_kind != "none" else None,
        pca_df=pca_df,
        zone_counts=zone_counts,
    )

    metadata = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "run_slug": this_run,
        "region": region.__dict__,
        "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "outputs": {
            "diagnostic_netcdf": str(diag_nc),
            "zone_spi12_timeseries": str(zone_ts_path),
            "zone_drought_runs": str(drought_runs_path),
            "zone_run_metrics": str(run_metrics_path),
            "pca_explained_variance": str(pca_path),
            "zone_pixel_counts": str(zone_counts_path),
            "zone_climate_index_correlations": str(corr_path),
            "notes": str(notes_path),
        },
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")

    print("Wrote SPI-12 regionalization diagnostics:")
    for path in [
        diag_nc,
        zone_ts_path,
        drought_runs_path,
        run_metrics_path,
        pca_path,
        zone_counts_path,
        corr_path,
        notes_path,
        report_dir / "zone_map.png",
        report_dir / "zone_spi12_timeseries.png",
        report_dir / "zone_run_duration_summary.png",
    ]:
        print(f"  {path}")


if __name__ == "__main__":
    main()
