#!/usr/bin/env python
"""Forecast-informed land-surface benchmark using CFSv2 soil moisture.

This benchmark addresses the project's remaining forecast-informed
land-surface gap without changing the canonical SPI-1 experiment. It verifies
NOAA/NCEI CFSv2 monthly-mean forecast soil moisture against an observed
ERA5-Land root-zone soil-moisture dry-fraction target over the Central Valley.

Design:
  - observed target: ERA5-Land 0-100 cm root-zone soil moisture dry fraction
    using train-period calendar-month/grid-cell quantile thresholds
  - forecast input: NCEI CFSv2 monthly-mean flxf GRIB2 soil moisture
  - calibration: validation-only isotonic mapping from forecast dry signal to
    observed monthly dry fraction
  - score: test-period monthly Brier Skill Score against train-period
    calendar-month climatology

The NCEI monthly-mean CFSv2 archive is not a complete 1991-2016 ML training
hindcast in this workflow, so this is an external forecast benchmark rather
than a trained forecast-feature model.
"""
from __future__ import annotations

from argparse import ArgumentParser, Namespace
from pathlib import Path
import re
import shutil
import warnings

import numpy as np
import pandas as pd
import requests
import xarray as xr
from sklearn.isotonic import IsotonicRegression

from region_config import resolve_region


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED = PROJECT_ROOT / "data" / "processed"
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "nmme_ncei_cfsv2_land"
OUT_DIR = PROJECT_ROOT / "outputs"
REPORT_DIR = PROJECT_ROOT / "results" / "report"

START_YEAR = 1991
CURRENT_YEAR = 2026

NCEI_THREDDS = "https://www.ncei.noaa.gov/thredds"
NCEI_CFSV2_MM_ROOT = f"{NCEI_THREDDS}/catalog/model-cfs_v2_for_mm"
NCEI_FILESERVER_ROOT = f"{NCEI_THREDDS}/fileServer/model-cfs_v2_for_mm"
SOURCE_CATALOG = f"{NCEI_CFSV2_MM_ROOT}/catalog.xml"

CATALOG_REF_RE = re.compile(
    r'xlink:href="(?P<href>[^"]+/catalog\.xml)"[^>]*xlink:title="(?P<title>[^"]+)"'
)
REQUEST_HEADERS = {"User-Agent": "chirps-drought-classifier/landsurface-benchmark"}


def parse_args() -> Namespace:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--region", default="cvalley")
    parser.add_argument(
        "--soil-file",
        type=Path,
        default=None,
        help="Defaults to data/processed/era5_land_soil_moisture_monthly_<region>_1991_2026.nc.",
    )
    parser.add_argument("--lead-months", type=int, default=1)
    parser.add_argument("--start-target", default="2017-01")
    parser.add_argument("--end-target", default=None)
    parser.add_argument("--normal-start-year", type=int, default=1991)
    parser.add_argument("--normal-end-year", type=int, default=2016)
    parser.add_argument("--dry-quantile", type=float, default=0.20)
    parser.add_argument(
        "--run-hours",
        nargs="+",
        type=int,
        default=[18],
        choices=[0, 6, 12, 18],
        help=(
            "CFSv2 initialization cycles from the latest available initialization day. "
            "Use all four cycles for an ensemble proxy; default keeps the benchmark small."
        ),
    )
    parser.add_argument("--cache-dir", type=Path, default=RAW_DIR)
    parser.add_argument(
        "--out-prefix",
        default=None,
        help="Output filename prefix under outputs/ and, with --copy-report, results/report/.",
    )
    parser.add_argument("--n-bootstrap", type=int, default=2000)
    parser.add_argument("--copy-report", action="store_true")
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--max-months", type=int, default=None)
    parser.add_argument(
        "--min-members",
        type=int,
        default=None,
        help="Minimum valid CFSv2 cycles required per target month. Defaults to the number of requested run hours.",
    )
    parser.add_argument("--progress-every", type=int, default=6)
    return parser.parse_args()


def default_soil_file(region_slug: str) -> Path:
    region = resolve_region(region_slug)
    return PROCESSED / f"era5_land_soil_moisture_monthly_{region.slug}_{START_YEAR}_{CURRENT_YEAR}.nc"


def default_out_prefix(region_slug: str, lead_months: int, run_hours: list[int]) -> str:
    region = resolve_region(region_slug)
    suffix = ""
    if sorted(run_hours) == [0, 6, 12, 18]:
        suffix = "_allcycles"
    elif sorted(run_hours) != [18]:
        suffix = "_" + "-".join(f"{h:02d}" for h in sorted(run_hours)) + "utc"
    if region.slug == "cvalley":
        return f"landsurface_cfsv2_rzsm_lead{lead_months}{suffix}"
    return f"landsurface_cfsv2_rzsm_{region.slug}_lead{lead_months}{suffix}"


def month_start(value: str | pd.Timestamp) -> pd.Timestamp:
    return pd.Timestamp(value).to_period("M").to_timestamp()


def brier(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean((np.asarray(p, dtype=float) - np.asarray(y, dtype=float)) ** 2))


def bss(y: np.ndarray, p: np.ndarray, ref: np.ndarray) -> float:
    ref_bs = brier(y, ref)
    return float(1.0 - brier(y, p) / ref_bs) if ref_bs > 0 else float("nan")


def bootstrap_bss(
    monthly: pd.DataFrame,
    pred_col: str,
    ref_col: str = "clim_prob_dry",
    n_bootstrap: int = 2000,
    seed: int = 42,
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    idx = np.arange(len(monthly))
    y = monthly["y_true_dry_frac"].to_numpy(dtype=float)
    p = monthly[pred_col].to_numpy(dtype=float)
    ref = monthly[ref_col].to_numpy(dtype=float)
    vals = np.empty(n_bootstrap, dtype=float)
    for i in range(n_bootstrap):
        sample = rng.choice(idx, size=len(idx), replace=True)
        vals[i] = bss(y[sample], p[sample], ref[sample])
    lo, hi = np.nanquantile(vals, [0.025, 0.975])
    return float(lo), float(hi)


def choose_var(ds: xr.Dataset, candidates: list[str]) -> str:
    for candidate in candidates:
        if candidate in ds.data_vars:
            return candidate
    lower = {name.lower(): name for name in ds.data_vars}
    for candidate in candidates:
        if candidate.lower() in lower:
            return lower[candidate.lower()]
    raise ValueError(f"Could not find any of {candidates}; available={list(ds.data_vars)}")


def standardize_soil_layer(ds: xr.Dataset, var_name: str) -> xr.DataArray:
    da = ds[var_name]
    rename = {}
    if "valid_time" in da.dims:
        rename["valid_time"] = "time"
    if "lat" in da.dims:
        rename["lat"] = "latitude"
    if "lon" in da.dims:
        rename["lon"] = "longitude"
    if rename:
        da = da.rename(rename)
    if "expver" in da.dims:
        da = da.mean(dim="expver", skipna=True)
    extra = [dim for dim in da.dims if dim not in {"time", "latitude", "longitude"}]
    for dim in extra:
        if da.sizes[dim] == 1:
            da = da.squeeze(dim, drop=True)
    extra = [dim for dim in da.dims if dim not in {"time", "latitude", "longitude"}]
    if extra:
        raise ValueError(f"Unexpected dimensions for {var_name}: {da.dims}")
    da = da.transpose("time", "latitude", "longitude")
    time_index = pd.DatetimeIndex(da["time"].values)
    da = da.isel(time=~time_index.duplicated()).sortby("time")
    return da


def observed_rootzone_target(args: Namespace) -> pd.DataFrame:
    if not args.soil_file.exists():
        raise FileNotFoundError(f"ERA5-Land soil-moisture file not found: {args.soil_file}")
    ds = xr.open_dataset(args.soil_file).load()
    layers = {
        "swvl1": standardize_soil_layer(
            ds, choose_var(ds, ["swvl1", "volumetric_soil_water_layer_1"])
        ),
        "swvl2": standardize_soil_layer(
            ds, choose_var(ds, ["swvl2", "volumetric_soil_water_layer_2"])
        ),
        "swvl3": standardize_soil_layer(
            ds, choose_var(ds, ["swvl3", "volumetric_soil_water_layer_3"])
        ),
    }
    ds.close()

    # ERA5-Land layers: 0-7, 7-28, and 28-100 cm.
    rootzone = layers["swvl1"] * 0.07 + layers["swvl2"] * 0.21 + layers["swvl3"] * 0.72
    rootzone.name = "rootzone_sm"
    rootzone = rootzone.sortby("time")
    rootzone["time"] = pd.to_datetime(rootzone["time"].values).to_period("M").to_timestamp()

    years = pd.DatetimeIndex(rootzone["time"].values).year
    normal_mask = (years >= args.normal_start_year) & (years <= args.normal_end_year)
    if int(normal_mask.sum()) < 12:
        raise ValueError("Not enough normal-period months to define dry thresholds.")
    normal = rootzone.isel(time=normal_mask)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="All-NaN slice encountered", category=RuntimeWarning)
        thresholds = normal.groupby("time.month").quantile(
            args.dry_quantile,
            dim="time",
            skipna=True,
        )

    dry_fraction_rows = []
    lat_name = "latitude"
    lon_name = "longitude"
    for i, target_time in enumerate(pd.DatetimeIndex(rootzone["time"].values)):
        field = rootzone.isel(time=i)
        threshold = thresholds.sel(month=int(target_time.month))
        valid = np.isfinite(field) & np.isfinite(threshold)
        dry = xr.where(valid, field <= threshold, np.nan)
        dry_fraction = float(dry.mean(dim=[lat_name, lon_name], skipna=True).values)
        regional_mean = float(field.mean(dim=[lat_name, lon_name], skipna=True).values)
        dry_fraction_rows.append(
            {
                "target_time": month_start(target_time),
                "target_year": int(target_time.year),
                "target_month": int(target_time.month),
                "y_true_dry_frac": dry_fraction,
                "observed_rzsm": regional_mean,
            }
        )

    observed = pd.DataFrame(dry_fraction_rows).sort_values("target_time").reset_index(drop=True)
    train = observed[observed["target_year"] <= args.normal_end_year].copy()
    month_clim = train.groupby("target_month")["y_true_dry_frac"].mean()
    global_clim = float(train["y_true_dry_frac"].mean())
    observed["clim_prob_dry"] = observed["target_month"].map(month_clim).fillna(global_clim)

    persistence = observed[["target_time", "y_true_dry_frac"]].rename(
        columns={"target_time": "persistence_time", "y_true_dry_frac": "persistence_raw_prob_dry"}
    )
    observed["persistence_time"] = (
        observed["target_time"] - pd.DateOffset(months=args.lead_months)
    ).dt.to_period("M").dt.to_timestamp()
    observed = observed.merge(persistence, on="persistence_time", how="left")
    return observed


def read_catalog(url: str, cache_file: Path, refresh: bool) -> str | None:
    if cache_file.exists() and not refresh:
        return cache_file.read_text(encoding="utf-8")
    response = requests.get(url, timeout=(15, 45), headers=REQUEST_HEADERS)
    if response.status_code == 404:
        return None
    response.raise_for_status()
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    cache_file.write_text(response.text, encoding="utf-8")
    return response.text


def catalog_refs(xml: str | None) -> list[tuple[str, str]]:
    if not xml:
        return []
    return [(m.group("href"), m.group("title")) for m in CATALOG_REF_RE.finditer(xml)]


def catalog_xml_url(*parts: str) -> str:
    clean = "/".join(part.strip("/") for part in parts if part)
    return f"{NCEI_CFSV2_MM_ROOT}/{clean}/catalog.xml" if clean else SOURCE_CATALOG


def catalog_cache_file(cache_dir: Path, *parts: str) -> Path:
    clean = "_".join(part.strip("/") for part in parts if part) or "root"
    return cache_dir / "catalogs" / f"{clean}.xml"


def latest_init_runs(
    init_month: pd.Timestamp,
    run_hours: list[int],
    cache_dir: Path,
    refresh: bool,
) -> list[pd.Timestamp]:
    year = f"{init_month.year:04d}"
    yyyymm = f"{init_month.year:04d}{init_month.month:02d}"
    month_xml = read_catalog(
        catalog_xml_url(year, yyyymm),
        catalog_cache_file(cache_dir, year, yyyymm),
        refresh,
    )
    day_refs = catalog_refs(month_xml)
    if not day_refs:
        return []
    latest_day = sorted(title for _, title in day_refs if re.fullmatch(r"\d{8}", title))[-1]
    day_xml = read_catalog(
        catalog_xml_url(year, yyyymm, latest_day),
        catalog_cache_file(cache_dir, year, yyyymm, latest_day),
        refresh,
    )
    hour_refs = catalog_refs(day_xml)
    available = {
        int(title[-2:]): pd.to_datetime(title, format="%Y%m%d%H")
        for _, title in hour_refs
        if re.fullmatch(r"\d{10}", title)
    }
    return [available[hour] for hour in sorted(run_hours) if hour in available]


def flxf_file_url(init_time: pd.Timestamp, target_time: pd.Timestamp) -> str:
    year = f"{init_time.year:04d}"
    yyyymm = f"{init_time.year:04d}{init_time.month:02d}"
    yyyymmdd = f"{init_time.year:04d}{init_time.month:02d}{init_time.day:02d}"
    init = f"{init_time.year:04d}{init_time.month:02d}{init_time.day:02d}{init_time.hour:02d}"
    target = f"{target_time.year:04d}{target_time.month:02d}"
    filename = f"flxf.01.{init}.{target}.avrg.grib.grb2"
    return f"{NCEI_FILESERVER_ROOT}/{year}/{yyyymm}/{yyyymmdd}/{init}/{filename}"


def download_file(url: str, cache_dir: Path, refresh: bool) -> Path | None:
    filename = url.rsplit("/", 1)[-1]
    path = cache_dir / filename
    if path.exists() and path.stat().st_size > 0 and not refresh:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    response = requests.get(url, timeout=(15, 90), headers=REQUEST_HEADERS)
    if response.status_code == 404:
        return None
    response.raise_for_status()
    tmp.write_bytes(response.content)
    tmp.replace(path)
    return path


def subset_region(
    da: xr.DataArray,
    lat_min: float,
    lat_max: float,
    lon_min: float,
    lon_max: float,
) -> xr.DataArray:
    lat_name = "lat" if "lat" in da.coords else "latitude"
    lon_name = "lon" if "lon" in da.coords else "longitude"

    lat_values = da[lat_name].values
    lat_slice = slice(lat_max, lat_min) if float(lat_values[0]) > float(lat_values[-1]) else slice(lat_min, lat_max)

    lon_values = da[lon_name].values
    if float(np.nanmin(lon_values)) >= 0.0:
        lon_min_use = lon_min % 360.0
        lon_max_use = lon_max % 360.0
    else:
        lon_min_use = lon_min
        lon_max_use = lon_max

    if lon_min_use <= lon_max_use:
        sub = da.sel({lat_name: lat_slice, lon_name: slice(lon_min_use, lon_max_use)})
    else:
        west = da.sel({lat_name: lat_slice, lon_name: slice(lon_min_use, float(np.nanmax(lon_values)))})
        east = da.sel({lat_name: lat_slice, lon_name: slice(float(np.nanmin(lon_values)), lon_max_use)})
        sub = xr.concat([west, east], dim=lon_name)

    if sub.sizes.get(lat_name, 0) == 0 or sub.sizes.get(lon_name, 0) == 0:
        raise ValueError(
            "CFSv2 regional subset is empty: "
            f"lat[{lat_min}, {lat_max}] lon[{lon_min}, {lon_max}]"
        )
    return sub


def extract_cfsv2_rootzone_soil_moisture(path: Path, region_slug: str) -> float:
    region = resolve_region(region_slug)
    backend_kwargs = {
        "filter_by_keys": {"shortName": "soilw"},
        "indexpath": "",
    }
    with xr.open_dataset(path, engine="cfgrib", backend_kwargs=backend_kwargs) as ds:
        if "soilw" not in ds.data_vars:
            raise ValueError(f"{path} does not contain CFSv2 soilw; available={list(ds.data_vars)}")
        da = ds["soilw"]
        if "depthBelowLandLayer" not in da.dims:
            raise ValueError(f"{path} soilw has unexpected dimensions: {da.dims}")
        da = subset_region(da, region.lat_min, region.lat_max, region.lon_min, region.lon_max)
        depth_name = "depthBelowLandLayer"
        depths = [float(v) for v in da[depth_name].values]
        order = np.argsort(depths)[:3]
        selected = da.isel({depth_name: order})
        weights = xr.DataArray(
            np.array([0.1, 0.3, 0.6], dtype=float),
            dims=[depth_name],
            coords={depth_name: selected[depth_name].values},
        )
        rootzone = (selected * weights).sum(dim=depth_name)
        lat_name = "lat" if "lat" in rootzone.coords else "latitude"
        lon_name = "lon" if "lon" in rootzone.coords else "longitude"
        area_weights = np.cos(np.deg2rad(rootzone[lat_name]))
        regional = rootzone.weighted(area_weights).mean(dim=[lat_name, lon_name], skipna=True)
        return float(regional.values)


def build_forecast_rows(args: Namespace, target_months: pd.DatetimeIndex) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, object]] = []
    skipped: list[dict[str, object]] = []
    progress_every = max(1, int(args.progress_every))

    for i, target_time in enumerate(target_months, start=1):
        target_time = month_start(target_time)
        init_month = month_start(target_time - pd.DateOffset(months=args.lead_months))
        init_runs = latest_init_runs(
            init_month,
            sorted(args.run_hours),
            cache_dir=args.cache_dir,
            refresh=args.refresh,
        )
        if not init_runs:
            skipped.append(
                {
                    "target_time": target_time,
                    "init_month": init_month,
                    "reason": "no_month_or_hour_catalog",
                }
            )
            continue

        member_values = []
        member_urls = []
        for init_time in init_runs:
            url = flxf_file_url(init_time, target_time)
            local = download_file(url, args.cache_dir, refresh=args.refresh)
            if local is None:
                skipped.append(
                    {
                        "target_time": target_time,
                        "init_month": init_month,
                        "init_time": init_time,
                        "reason": "missing_target_month_file",
                    }
                )
                continue
            value = extract_cfsv2_rootzone_soil_moisture(local, args.region)
            if np.isfinite(value):
                member_values.append(value)
                member_urls.append(url)

        if not member_values:
            skipped.append(
                {
                    "target_time": target_time,
                    "init_month": init_month,
                    "reason": "no_valid_member_values",
                }
            )
            continue
        min_members = args.min_members if args.min_members is not None else len(set(args.run_hours))
        if len(member_values) < min_members:
            skipped.append(
                {
                    "target_time": target_time,
                    "init_month": init_month,
                    "reason": f"insufficient_members_{len(member_values)}_of_{min_members}",
                }
            )
            continue

        rows.append(
            {
                "target_time": target_time,
                "init_month": init_month,
                "init_time_first": min(init_runs),
                "init_time_last": max(init_runs),
                "lead_months": args.lead_months,
                "region": args.region,
                "source_model": "NCEI_CFSv2_monthly_flxf_soilw",
                "forecast_rzsm": float(np.mean(member_values)),
                "forecast_rzsm_member_std": float(np.std(member_values, ddof=0)),
                "n_members": int(len(member_values)),
                "requested_run_hours": " ".join(str(h) for h in sorted(args.run_hours)),
                "source_catalog": SOURCE_CATALOG,
                "source_urls": " ".join(member_urls),
                "units": "volumetric soil moisture proportion",
                "notes": (
                    "CFSv2 flxf soilw first-meter approximation from the first three "
                    "depthBelowLandLayer records, weighted as 0-10, 10-40, and 40-100 cm."
                ),
            }
        )
        if i == 1 or i == len(target_months) or i % progress_every == 0:
            print(
                f"  {i:>3}/{len(target_months)} target={target_time:%Y-%m} "
                f"init={init_month:%Y-%m} members={len(member_values)} "
                f"rzsm={np.mean(member_values):.4f}",
                flush=True,
            )

    if not rows:
        raise RuntimeError("No CFSv2 land-surface target months were processed.")

    forecast = pd.DataFrame(rows).sort_values("target_time").reset_index(drop=True)
    forecast["target_year"] = pd.to_datetime(forecast["target_time"]).dt.year
    forecast["target_month"] = pd.to_datetime(forecast["target_time"]).dt.month
    normal = forecast[(forecast["target_year"] >= 2017) & (forecast["target_year"] <= 2020)].copy()
    if normal["target_month"].nunique() < 12:
        normal = forecast[forecast["target_year"] <= 2020].copy()
    if normal.empty:
        forecast["forecast_rzsm_normal"] = np.nan
        forecast["forecast_rzsm_anom"] = np.nan
    else:
        month_norm = normal.groupby("target_month")["forecast_rzsm"].mean()
        global_norm = float(normal["forecast_rzsm"].mean())
        forecast["forecast_rzsm_normal"] = forecast["target_month"].map(month_norm).fillna(global_norm)
        forecast["forecast_rzsm_anom"] = forecast["forecast_rzsm"] - forecast["forecast_rzsm_normal"]
    forecast = forecast.drop(columns=["target_year", "target_month"])

    skipped_df = pd.DataFrame(skipped)
    return forecast, skipped_df


def target_months_from_observed(args: Namespace, observed: pd.DataFrame) -> pd.DatetimeIndex:
    start = month_start(args.start_target)
    end = month_start(args.end_target) if args.end_target else observed["target_time"].max()
    months = observed.loc[
        (observed["target_time"] >= start)
        & (observed["target_time"] <= end),
        "target_time",
    ].drop_duplicates()
    months = pd.DatetimeIndex(months)
    if args.max_months is not None:
        months = months[: args.max_months]
    if len(months) == 0:
        raise ValueError("No target months selected from observed ERA5-Land target.")
    return months


def fit_isotonic(val: pd.DataFrame, test: pd.DataFrame, signal_col: str) -> tuple[pd.Series, pd.Series]:
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    iso.fit(val[signal_col].to_numpy(dtype=float), val["y_true_dry_frac"].to_numpy(dtype=float))
    val_pred = pd.Series(iso.predict(val[signal_col].to_numpy(dtype=float)), index=val.index).clip(0.0, 1.0)
    test_pred = pd.Series(iso.predict(test[signal_col].to_numpy(dtype=float)), index=test.index).clip(0.0, 1.0)
    return val_pred, test_pred


def score_benchmark(
    observed: pd.DataFrame,
    forecast: pd.DataFrame,
    args: Namespace,
) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    merged = observed.merge(forecast, on="target_time", how="inner", suffixes=("", "_forecast"))
    merged = merged.dropna(
        subset=[
            "forecast_rzsm",
            "forecast_rzsm_anom",
            "persistence_raw_prob_dry",
            "y_true_dry_frac",
            "clim_prob_dry",
        ]
    ).copy()
    if merged.empty:
        raise ValueError("No overlap between observed target and CFSv2 forecast rows.")

    merged["cfsv2_raw_dry_signal"] = -merged["forecast_rzsm"].astype(float)
    merged["cfsv2_anom_dry_signal"] = -merged["forecast_rzsm_anom"].astype(float)
    val = merged[(merged["target_year"] >= 2017) & (merged["target_year"] <= 2020)].copy()
    test = merged[merged["target_year"] >= 2021].copy()
    if val.empty:
        raise ValueError("No 2017-2020 validation overlap for CFSv2 land-surface calibration.")
    if test.empty:
        raise ValueError("No 2021+ test overlap for CFSv2 land-surface evaluation.")

    val_raw, test_raw = fit_isotonic(val, test, "cfsv2_raw_dry_signal")
    val_anom, test_anom = fit_isotonic(val, test, "cfsv2_anom_dry_signal")
    val_pers_iso, test_pers_iso = fit_isotonic(val, test, "persistence_raw_prob_dry")

    val_bs = {
        "cfsv2_raw_isotonic": brier(val["y_true_dry_frac"], val_raw),
        "cfsv2_anom_isotonic": brier(val["y_true_dry_frac"], val_anom),
        "persistence_raw": brier(val["y_true_dry_frac"], val["persistence_raw_prob_dry"]),
        "persistence_isotonic": brier(val["y_true_dry_frac"], val_pers_iso),
    }
    cfs_best = min(
        {k: v for k, v in val_bs.items() if k.startswith("cfsv2_")},
        key=val_bs.get,
    )
    persistence_best = min(
        {k: v for k, v in val_bs.items() if k.startswith("persistence_")},
        key=val_bs.get,
    )

    test = test.copy()
    test["cfsv2_raw_isotonic_prob_dry"] = test_raw
    test["cfsv2_anom_isotonic_prob_dry"] = test_anom
    test["cfsv2_selected_prob_dry"] = (
        test["cfsv2_raw_isotonic_prob_dry"]
        if cfs_best == "cfsv2_raw_isotonic"
        else test["cfsv2_anom_isotonic_prob_dry"]
    )
    test["persistence_isotonic_prob_dry"] = test_pers_iso
    test["persistence_selected_prob_dry"] = (
        test["persistence_raw_prob_dry"]
        if persistence_best == "persistence_raw"
        else test["persistence_isotonic_prob_dry"]
    )

    score_cols = [
        "cfsv2_raw_isotonic_prob_dry",
        "cfsv2_anom_isotonic_prob_dry",
        "cfsv2_selected_prob_dry",
        "persistence_raw_prob_dry",
        "persistence_isotonic_prob_dry",
        "persistence_selected_prob_dry",
    ]
    y = test["y_true_dry_frac"].to_numpy(dtype=float)
    ref = test["clim_prob_dry"].to_numpy(dtype=float)
    persistence_raw_ref = test["persistence_raw_prob_dry"].to_numpy(dtype=float)
    persistence_selected_ref = test["persistence_selected_prob_dry"].to_numpy(dtype=float)
    score_rows = []
    for i, col in enumerate(score_cols):
        pred = test[col].to_numpy(dtype=float)
        lo, hi = bootstrap_bss(test, col, n_bootstrap=args.n_bootstrap, seed=301 + i)
        score_rows.append(
            {
                "forecast": col.replace("_prob_dry", ""),
                "bs": brier(y, pred),
                "bss": bss(y, pred, ref),
                "bss_ci_low": lo,
                "bss_ci_high": hi,
                "bss_vs_persistence_raw": bss(y, pred, persistence_raw_ref),
                "bss_vs_persistence_selected": bss(y, pred, persistence_selected_ref),
            }
        )
    scores = pd.DataFrame(score_rows)

    cfs_corr = test["cfsv2_selected_prob_dry"].corr(test["y_true_dry_frac"], method="spearman")
    cfs_raw_corr = test["cfsv2_raw_dry_signal"].corr(test["y_true_dry_frac"], method="spearman")
    cfs_anom_corr = test["cfsv2_anom_dry_signal"].corr(test["y_true_dry_frac"], method="spearman")
    pers_corr = test["persistence_selected_prob_dry"].corr(test["y_true_dry_frac"], method="spearman")
    y_std = float(test["y_true_dry_frac"].std(ddof=0))
    cfs_amp = float(test["cfsv2_selected_prob_dry"].std(ddof=0) / y_std) if y_std > 0 else np.nan
    pers_amp = float(test["persistence_selected_prob_dry"].std(ddof=0) / y_std) if y_std > 0 else np.nan

    lines = [
        "Forecast-Informed Land-Surface Benchmark",
        "=" * 68,
        "Design: CFSv2 monthly forecast soil moisture mapped to ERA5-Land root-zone dry-fraction probability.",
        f"Region: {resolve_region(args.region).name} ({args.region})",
        f"Lead: {args.lead_months} month(s)",
        (
            "Observed target: ERA5-Land root-zone soil moisture dry fraction, "
            f"q={args.dry_quantile:.2f} thresholds from "
            f"{args.normal_start_year}-{args.normal_end_year} by calendar month and grid cell."
        ),
        (
            "Forecast source: NOAA NCEI THREDDS CFSv2 monthly means, flxf soilw, "
            "first-meter approximation from the first three soil layers."
        ),
        f"Run hours: {' '.join(str(h) for h in sorted(args.run_hours))} UTC on latest available init day",
        (
            "Validation months: "
            f"{val['target_time'].nunique()} "
            f"({val['target_time'].min():%Y-%m} to {val['target_time'].max():%Y-%m})"
        ),
        (
            "Test months: "
            f"{test['target_time'].nunique()} "
            f"({test['target_time'].min():%Y-%m} to {test['target_time'].max():%Y-%m})"
        ),
        f"Validation BS by method: {val_bs}",
        f"Selected CFSv2 calibration: {cfs_best}",
        f"Selected persistence calibration: {persistence_best}",
        f"Climatology BS: {brier(y, ref):.5f}",
        f"Spearman corr(CFSv2 selected prob, observed dry fraction): {cfs_corr:.3f}",
        f"Spearman corr(CFSv2 raw dry signal, observed dry fraction): {cfs_raw_corr:.3f}",
        f"Spearman corr(CFSv2 anomaly dry signal, observed dry fraction): {cfs_anom_corr:.3f}",
        f"Spearman corr(persistence selected prob, observed dry fraction): {pers_corr:.3f}",
        f"CFSv2 selected probability amplitude ratio: {cfs_amp:.3f}",
        f"Persistence selected probability amplitude ratio: {pers_amp:.3f}",
        "",
        "Monthly dry-fraction Brier Skill Score vs climatology:",
    ]
    for row in score_rows:
        lines.append(
            f"  {row['forecast']:<34} BS={row['bs']:.5f} "
            f"BSS={row['bss']:.5f} "
            f"95% CI [{row['bss_ci_low']:.5f}, {row['bss_ci_high']:.5f}]"
        )
    cfs_selected = next(row for row in score_rows if row["forecast"] == "cfsv2_selected")
    lines.extend(
        [
            "",
            "CFSv2 selected skill against same-target persistence references:",
            f"  vs persistence_raw      : {cfs_selected['bss_vs_persistence_raw']:.5f}",
            f"  vs persistence_selected : {cfs_selected['bss_vs_persistence_selected']:.5f}",
        ]
    )
    return test, scores, "\n".join(lines) + "\n"


def write_outputs(
    forecast: pd.DataFrame,
    skipped: pd.DataFrame,
    monthly: pd.DataFrame,
    scores: pd.DataFrame,
    score_text: str,
    args: Namespace,
) -> None:
    OUT_DIR.mkdir(exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    outputs = {
        "forecast": OUT_DIR / f"{args.out_prefix}_forecast.csv",
        "skipped": OUT_DIR / f"{args.out_prefix}_skipped_months.csv",
        "monthly": OUT_DIR / f"{args.out_prefix}_monthly_scores.csv",
        "summary": OUT_DIR / f"{args.out_prefix}_summary.csv",
        "scores": OUT_DIR / f"{args.out_prefix}_scores.txt",
    }
    forecast.to_csv(outputs["forecast"], index=False)
    skipped.to_csv(outputs["skipped"], index=False)
    monthly["target_variable"] = "ERA5-Land root-zone soil moisture dry fraction"
    monthly["source_model"] = "NCEI_CFSv2_monthly_flxf_soilw"
    monthly["benchmark_lead_months"] = args.lead_months
    monthly["region"] = args.region
    monthly.to_csv(outputs["monthly"], index=False)
    scores.to_csv(outputs["summary"], index=False)
    outputs["scores"].write_text(score_text + f"\nMonthly scores: {outputs['monthly']}\n", encoding="utf-8")

    print(score_text)
    print(f"Wrote forecast rows: {outputs['forecast']} rows={len(forecast):,}")
    print(f"Wrote skipped audit: {outputs['skipped']} rows={len(skipped):,}")
    print(f"Wrote monthly scores: {outputs['monthly']} rows={len(monthly):,}")
    print(f"Wrote score summary: {outputs['summary']}")

    if args.copy_report:
        for path in outputs.values():
            shutil.copy2(path, REPORT_DIR / path.name)


def main() -> None:
    args = parse_args()
    args.region = resolve_region(args.region).slug
    if args.soil_file is None:
        args.soil_file = default_soil_file(args.region)
    if args.out_prefix is None:
        args.out_prefix = default_out_prefix(args.region, args.lead_months, args.run_hours)
    observed = observed_rootzone_target(args)
    months = target_months_from_observed(args, observed)
    print(
        f"Selected target months: {months.min():%Y-%m} to {months.max():%Y-%m} "
        f"({len(months)} months)",
        flush=True,
    )
    forecast, skipped = build_forecast_rows(args, months)
    monthly, scores, score_text = score_benchmark(observed, forecast, args)
    write_outputs(forecast, skipped, monthly, scores, score_text, args)


if __name__ == "__main__":
    main()
