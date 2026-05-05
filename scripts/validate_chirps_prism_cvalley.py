#!/usr/bin/env python
"""
Validate Central Valley CHIRPS SPI-1 labels against PRISM monthly precipitation.

The script downloads/caches official PRISM 4 km monthly precipitation NetCDF
grids, clips them to the Central Valley groundwater-basin geometry, computes
PRISM SPI-1 with the same 1991-2020 calendar-month baseline convention, and
compares PRISM drought labels with CHIRPS SPI-1 labels and the current
XGBoost-Spatial forecast probabilities.

Outputs:
  data/processed/prism_ppt_monthly_cvalley_basin_1991_2026.nc
  data/processed/prism_spi1_cvalley_basin_1991_2026.nc
  results/validation/prism_chirps_monthly_comparison.csv
  results/validation/prism_validation_metrics.txt
  results/validation/prism_chirps_dry_fraction_comparison.png

PRISM source:
  https://services.nacse.org/prism/data/get/us/4km/ppt/YYYYMM?format=nc
"""
from __future__ import annotations

import argparse
import json
import tempfile
import time
import zipfile
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from scipy.stats import gamma, norm, pearsonr, spearmanr
from shapely import contains_xy
from shapely.geometry import shape
from shapely.ops import unary_union


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "prism" / "monthly" / "ppt"
PROCESSED = PROJECT_ROOT / "data" / "processed"
RESULTS = PROJECT_ROOT / "results" / "validation"
RAW_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED.mkdir(parents=True, exist_ok=True)
RESULTS.mkdir(parents=True, exist_ok=True)

CHIRPS_SPI = PROCESSED / "chirps_v3_monthly_cvalley_spi_1991_2026.nc"
CV_GEOMETRY = PROJECT_ROOT / "data" / "metadata" / "dwr" / "central_valley_b118_groundwater_basins.geojson"
XGB_SPATIAL_PROBS = PROJECT_ROOT / "outputs" / "xgb_spatial_test_probs.npz"

PRISM_URL = "https://services.nacse.org/prism/data/get/us/4km/ppt/{yyyymm}?format=nc"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", default="1991-01", help="First PRISM month, YYYY-MM.")
    parser.add_argument("--end", default="2026-03", help="Last PRISM month, YYYY-MM.")
    parser.add_argument("--baseline-start-year", type=int, default=1991)
    parser.add_argument("--baseline-end-year", type=int, default=2020)
    parser.add_argument("--sleep-seconds", type=float, default=0.15)
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Re-download PRISM zip files even if cached.",
    )
    parser.add_argument(
        "--force-rebuild",
        action="store_true",
        help="Rebuild processed PRISM NetCDF/SPI even if cached.",
    )
    parser.add_argument("--chirps-spi", type=Path, default=CHIRPS_SPI)
    parser.add_argument("--geometry", type=Path, default=CV_GEOMETRY)
    parser.add_argument("--xgb-spatial-probs", type=Path, default=XGB_SPATIAL_PROBS)
    return parser.parse_args()


def month_range(start: str, end: str) -> pd.DatetimeIndex:
    return pd.date_range(pd.Timestamp(start + "-01"), pd.Timestamp(end + "-01"), freq="MS")


def output_paths(start_year: int, end_year: int) -> tuple[Path, Path]:
    ppt = PROCESSED / f"prism_ppt_monthly_cvalley_basin_{start_year}_{end_year}.nc"
    spi = PROCESSED / f"prism_spi1_cvalley_basin_{start_year}_{end_year}.nc"
    return ppt, spi


def load_geometry(path: Path):
    data = json.loads(path.read_text())
    geoms = [shape(feature["geometry"]) for feature in data["features"] if feature.get("geometry")]
    return unary_union(geoms)


def geometry_mask(lat: np.ndarray, lon: np.ndarray, geom) -> xr.DataArray:
    lon2d, lat2d = np.meshgrid(lon, lat)
    mask = contains_xy(geom, lon2d, lat2d)
    return xr.DataArray(mask, dims=("latitude", "longitude"), coords={"latitude": lat, "longitude": lon})


def download_prism_month(month: pd.Timestamp, force: bool, sleep_seconds: float) -> Path:
    yyyymm = month.strftime("%Y%m")
    out = RAW_DIR / f"prism_ppt_us_25m_{yyyymm}.zip"
    if out.exists() and out.stat().st_size > 0 and not force:
        return out

    url = PRISM_URL.format(yyyymm=yyyymm)
    print(f"Downloading PRISM {yyyymm}")
    request = Request(url, headers={"User-Agent": "chirps-drought-classifier research validation"})
    try:
        with urlopen(request, timeout=120) as response:
            payload = response.read()
    except (HTTPError, URLError) as exc:
        raise RuntimeError(f"Failed to download PRISM {yyyymm} from {url}: {exc}") from exc

    out.write_bytes(payload)
    if sleep_seconds > 0:
        time.sleep(sleep_seconds)
    return out


def read_prism_zip_subset(zip_path: Path, geom, bounds: tuple[float, float, float, float]) -> xr.DataArray:
    min_lon, min_lat, max_lon, max_lat = bounds
    with zipfile.ZipFile(zip_path) as zf:
        nc_members = [name for name in zf.namelist() if name.endswith(".nc")]
        if not nc_members:
            raise ValueError(f"No NetCDF file found in {zip_path}")
        with tempfile.TemporaryDirectory() as tmp:
            zf.extract(nc_members[0], tmp)
            ds = xr.open_dataset(Path(tmp) / nc_members[0]).load()

    var_name = "Band1"
    if var_name not in ds:
        data_vars = [name for name in ds.data_vars if name.lower() != "crs"]
        if not data_vars:
            raise ValueError(f"No precipitation variable found in {zip_path}")
        var_name = data_vars[0]

    da = ds[var_name].astype("float32").rename("ppt")
    da = da.rename({"lat": "latitude", "lon": "longitude"})
    da = da.where(da > -9990)
    da = da.sel(
        latitude=slice(min_lat - 0.3, max_lat + 0.3),
        longitude=slice(min_lon - 0.3, max_lon + 0.3),
    )
    # PRISM NetCDF files from different production vintages can differ only in
    # floating-point coordinate precision. Round before concat so identical grid
    # cells align exactly instead of creating duplicate lat/lon coordinates.
    da = da.assign_coords(
        latitude=np.round(da["latitude"].values.astype("float64"), 6),
        longitude=np.round(da["longitude"].values.astype("float64"), 6),
    )
    mask = geometry_mask(da["latitude"].values, da["longitude"].values, geom)
    return da.where(mask)


def build_prism_precip(months: pd.DatetimeIndex, args: argparse.Namespace, ppt_path: Path) -> xr.Dataset:
    geom = load_geometry(args.geometry)
    bounds = geom.bounds
    arrays: list[xr.DataArray] = []
    for month in months:
        zip_path = download_prism_month(month, args.force_download, args.sleep_seconds)
        da = read_prism_zip_subset(zip_path, geom, bounds)
        da = da.expand_dims(time=[month])
        arrays.append(da)
    ppt = xr.concat(arrays, dim="time", join="exact")
    ppt["time"] = months
    ds = xr.Dataset({"ppt": ppt})
    ds.attrs.update(
        {
            "source": "PRISM 4km monthly precipitation via NACSE web service",
            "region": "California Central Valley groundwater-basin union",
            "geometry": str(args.geometry.relative_to(PROJECT_ROOT)),
            "units": "mm/month",
        }
    )
    ds.to_netcdf(ppt_path)
    return ds


def compute_spi1(
    pr: xr.DataArray,
    baseline_start_year: int,
    baseline_end_year: int,
) -> xr.DataArray:
    times = pd.DatetimeIndex(pd.to_datetime(pr["time"].values))
    years = times.year
    months = times.month
    values = pr.values.astype("float64")
    out = np.full(values.shape, np.nan, dtype="float32")
    n_time, n_lat, n_lon = values.shape
    flat = values.reshape(n_time, n_lat * n_lon)

    for month in range(1, 13):
        idx = np.where(months == month)[0]
        base_idx = idx[(years[idx] >= baseline_start_year) & (years[idx] <= baseline_end_year)]
        if len(base_idx) < 8:
            continue
        month_values = flat[idx, :]
        base_values = flat[base_idx, :]
        month_out = np.full(month_values.shape, np.nan, dtype="float32")
        for pixel in range(month_values.shape[1]):
            base = base_values[:, pixel]
            base = base[np.isfinite(base)]
            if len(base) < 8:
                continue
            base = np.clip(base, 0.0, None)
            positive = base[base > 0]
            p_zero = 1.0 - (len(positive) / len(base))
            vals = month_values[:, pixel]
            finite = np.isfinite(vals)
            if not finite.any():
                continue
            vals_clip = np.clip(vals[finite], 0.0, None)
            if len(positive) < 4 or float(np.nanstd(positive)) <= 0:
                ranks = np.searchsorted(np.sort(base), vals_clip, side="right")
                cdf = (ranks + 0.5) / (len(base) + 1.0)
            else:
                try:
                    a, loc, scale = gamma.fit(positive, floc=0)
                    cdf = p_zero + (1.0 - p_zero) * gamma.cdf(vals_clip, a, loc=loc, scale=scale)
                except Exception:
                    ranks = np.searchsorted(np.sort(base), vals_clip, side="right")
                    cdf = (ranks + 0.5) / (len(base) + 1.0)
            cdf = np.clip(cdf, 1e-6, 1.0 - 1e-6)
            z = np.full(vals.shape, np.nan, dtype="float32")
            z[finite] = norm.ppf(cdf).astype("float32")
            month_out[:, pixel] = z
        out[idx, :, :] = month_out.reshape(len(idx), n_lat, n_lon)
        print(f"  PRISM SPI-1 month {month:02d}/12 done")

    spi = xr.DataArray(
        out,
        dims=pr.dims,
        coords=pr.coords,
        name="spi1",
        attrs={"baseline_years": f"{baseline_start_year}-{baseline_end_year}"},
    )
    return spi


def dry_fraction_from_spi(spi: xr.DataArray) -> pd.DataFrame:
    dry = (spi <= -1.0).where(np.isfinite(spi))
    frac = dry.mean(dim=[d for d in dry.dims if d != "time"], skipna=True)
    precip_or_spi_mean = spi.mean(dim=[d for d in spi.dims if d != "time"], skipna=True)
    df = pd.DataFrame(
        {
            "month_dt": pd.to_datetime(frac["time"].values),
            "dry_frac": frac.values.astype(float),
            "spi1_mean": precip_or_spi_mean.values.astype(float),
        }
    )
    return df


def precip_mean(pr: xr.DataArray) -> pd.DataFrame:
    mean = pr.mean(dim=[d for d in pr.dims if d != "time"], skipna=True)
    return pd.DataFrame(
        {
            "month_dt": pd.to_datetime(mean["time"].values),
            "precip_mean": mean.values.astype(float),
        }
    )


def load_chirps_monthly(chirps_spi_path: Path, geom) -> pd.DataFrame:
    ds = xr.open_dataset(chirps_spi_path).load()
    spi = ds["spi1"].astype("float32")
    mask = geometry_mask(spi["latitude"].values, spi["longitude"].values, geom)
    spi = spi.where(mask)
    dry = dry_fraction_from_spi(spi).rename(
        columns={"dry_frac": "chirps_dry_frac", "spi1_mean": "chirps_spi1_mean"}
    )
    return dry


def pair_stats(a: pd.Series, b: pd.Series) -> dict[str, float]:
    valid = a.notna() & b.notna()
    av = a[valid].astype(float)
    bv = b[valid].astype(float)
    if len(av) < 3:
        return {"pearson_r": np.nan, "spearman_r": np.nan, "rmse": np.nan, "bias": np.nan}
    return {
        "pearson_r": float(pearsonr(av, bv).statistic),
        "spearman_r": float(spearmanr(av, bv).statistic),
        "rmse": float(np.sqrt(np.mean((av - bv) ** 2))),
        "bias": float((av - bv).mean()),
    }


def brier_score(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean((y - p) ** 2))


def bss(y: np.ndarray, p: np.ndarray, ref: np.ndarray) -> float:
    ref_bs = brier_score(y, ref)
    return float(1.0 - brier_score(y, p) / ref_bs) if ref_bs > 0 else float("nan")


def load_xgb_spatial_monthly(path: Path) -> pd.DataFrame:
    z = np.load(path, allow_pickle=True)
    times = pd.to_datetime(z["times"]) + pd.DateOffset(months=1)
    probs = z["proba"][:, 0].astype(float)
    y_true = z["y_true"].astype(int)
    df = pd.DataFrame(
        {
            "month_dt": times,
            "xgb_spatial_prob_dry": probs,
            "chirps_model_target_dry": (y_true == -1).astype(float),
        }
    )
    return (
        df.groupby("month_dt", as_index=False)
        .agg(
            xgb_spatial_prob_dry=("xgb_spatial_prob_dry", "mean"),
            chirps_model_target_dry_frac=("chirps_model_target_dry", "mean"),
        )
        .sort_values("month_dt")
    )


def write_metrics(
    comparison: pd.DataFrame,
    model_eval: pd.DataFrame,
    args: argparse.Namespace,
) -> None:
    all_stats = pair_stats(comparison["chirps_dry_frac"], comparison["prism_dry_frac"])
    test = comparison[comparison["month_dt"] >= pd.Timestamp("2021-01-01")]
    test_stats = pair_stats(test["chirps_dry_frac"], test["prism_dry_frac"])

    rows = [
        "PRISM Cross-Dataset Validation - Central Valley Basin",
        "=" * 72,
        "",
        "PRISM source: official NACSE/PRISM 4 km monthly precipitation web service.",
        f"PRISM months: {comparison['month_dt'].min().date()} to {comparison['month_dt'].max().date()}",
        f"SPI baseline: {args.baseline_start_year}-{args.baseline_end_year}",
        "Geometry: DWR Central Valley groundwater basin union.",
        "",
        "CHIRPS vs PRISM monthly dry-fraction agreement:",
        f"  All overlap Pearson r   : {all_stats['pearson_r']:.3f}",
        f"  All overlap Spearman r  : {all_stats['spearman_r']:.3f}",
        f"  All overlap RMSE        : {all_stats['rmse']:.3f}",
        f"  All overlap bias CHIRPS-PRISM: {all_stats['bias']:.3f}",
        f"  Test Pearson r          : {test_stats['pearson_r']:.3f}",
        f"  Test Spearman r         : {test_stats['spearman_r']:.3f}",
        f"  Test RMSE               : {test_stats['rmse']:.3f}",
        f"  Test bias CHIRPS-PRISM  : {test_stats['bias']:.3f}",
    ]

    for threshold in [0.25, 0.50]:
        agree = (
            (test["chirps_dry_frac"] >= threshold)
            == (test["prism_dry_frac"] >= threshold)
        ).mean()
        rows.append(f"  Test agreement dry_frac >= {threshold:.2f}: {agree:.3f}")

    if not model_eval.empty:
        rows.extend(["", "XGBoost-Spatial forecast evaluated against PRISM SPI-1 dry fraction:"])
        for key, value in model_eval.iloc[0].items():
            if isinstance(value, float):
                rows.append(f"  {key}: {value:.5f}")
            else:
                rows.append(f"  {key}: {value}")

    (RESULTS / "prism_validation_metrics.txt").write_text("\n".join(rows) + "\n")


def model_against_prism(comparison: pd.DataFrame, xgb_path: Path) -> pd.DataFrame:
    if not xgb_path.exists():
        return pd.DataFrame()
    model = load_xgb_spatial_monthly(xgb_path)
    merged = comparison.merge(model, on="month_dt", how="inner")
    train = comparison[comparison["month_dt"].dt.year <= 2016].copy()
    test = merged[merged["month_dt"].dt.year >= 2021].copy()
    if train.empty or test.empty:
        return pd.DataFrame()
    month_clim = train.groupby(train["month_dt"].dt.month)["prism_dry_frac"].mean()
    global_clim = float(train["prism_dry_frac"].mean())
    test["prism_clim_train_monthly"] = test["month_dt"].dt.month.map(month_clim).fillna(global_clim)

    y = test["prism_dry_frac"].to_numpy()
    p_model = test["xgb_spatial_prob_dry"].to_numpy()
    p_chirps = test["chirps_dry_frac"].to_numpy()
    ref = test["prism_clim_train_monthly"].to_numpy()
    return pd.DataFrame(
        [
            {
                "n_test_months": int(len(test)),
                "prism_obs_dry_mean": float(test["prism_dry_frac"].mean()),
                "chirps_obs_dry_mean": float(test["chirps_dry_frac"].mean()),
                "xgb_prob_dry_mean": float(test["xgb_spatial_prob_dry"].mean()),
                "prism_clim_dry_mean": float(test["prism_clim_train_monthly"].mean()),
                "xgb_bs_vs_prism": brier_score(y, p_model),
                "prism_clim_bs": brier_score(y, ref),
                "xgb_bss_vs_prism_climatology": bss(y, p_model, ref),
                "chirps_label_bs_vs_prism": brier_score(y, p_chirps),
                "chirps_label_bss_vs_prism_climatology": bss(y, p_chirps, ref),
                "xgb_prism_spearman": float(spearmanr(test["xgb_spatial_prob_dry"], test["prism_dry_frac"]).statistic),
            }
        ]
    )


def plot_comparison(comparison: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(comparison["month_dt"], comparison["chirps_dry_frac"], label="CHIRPS SPI-1 dry fraction", lw=1.5)
    ax.plot(comparison["month_dt"], comparison["prism_dry_frac"], label="PRISM SPI-1 dry fraction", lw=1.5)
    ax.axvspan(pd.Timestamp("2021-01-01"), comparison["month_dt"].max(), color="0.9", alpha=0.6, label="Test period")
    ax.set_ylabel("Dry fraction")
    ax.set_ylim(-0.02, 1.02)
    ax.set_title("Central Valley basin dry fraction: CHIRPS vs PRISM")
    ax.legend(loc="upper left", ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(RESULTS / "prism_chirps_dry_fraction_comparison.png", dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    months = month_range(args.start, args.end)
    start_year = int(months[0].year)
    end_year = int(months[-1].year)
    ppt_path, spi_path = output_paths(start_year, end_year)

    if ppt_path.exists() and not args.force_rebuild:
        print(f"Loading cached {ppt_path}")
        prism_ds = xr.open_dataset(ppt_path).load()
    else:
        prism_ds = build_prism_precip(months, args, ppt_path)

    if spi_path.exists() and not args.force_rebuild:
        print(f"Loading cached {spi_path}")
        prism_spi = xr.open_dataset(spi_path)["spi1"].load()
    else:
        print("Computing PRISM SPI-1")
        prism_spi = compute_spi1(prism_ds["ppt"], args.baseline_start_year, args.baseline_end_year)
        xr.Dataset({"spi1": prism_spi}).to_netcdf(spi_path)

    geom = load_geometry(args.geometry)
    prism_dry = dry_fraction_from_spi(prism_spi).rename(
        columns={"dry_frac": "prism_dry_frac", "spi1_mean": "prism_spi1_mean"}
    )
    prism_pr = precip_mean(prism_ds["ppt"]).rename(columns={"precip_mean": "prism_precip_mean"})
    chirps = load_chirps_monthly(args.chirps_spi, geom)

    comparison = (
        prism_dry.merge(prism_pr, on="month_dt", how="left")
        .merge(chirps, on="month_dt", how="inner")
        .sort_values("month_dt")
    )
    comparison["year"] = comparison["month_dt"].dt.year.astype(int)
    comparison["month"] = comparison["month_dt"].dt.month.astype(int)
    comparison_path = RESULTS / "prism_chirps_monthly_comparison.csv"
    comparison.to_csv(comparison_path, index=False)

    model_eval = model_against_prism(comparison, args.xgb_spatial_probs)
    if not model_eval.empty:
        model_eval.to_csv(RESULTS / "prism_model_validation_summary.csv", index=False)

    write_metrics(comparison, model_eval, args)
    plot_comparison(comparison)

    print(f"Wrote {ppt_path}")
    print(f"Wrote {spi_path}")
    print(f"Wrote {comparison_path}")
    print(f"Wrote {RESULTS / 'prism_validation_metrics.txt'}")
    print(f"Wrote {RESULTS / 'prism_chirps_dry_fraction_comparison.png'}")


if __name__ == "__main__":
    main()
