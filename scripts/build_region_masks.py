#!/usr/bin/env python
"""
Build country-mask diagnostics for configured multi-region CHIRPS grids.

The multi-region experiments intentionally started with rectangular bounding
boxes. This script adds a dependency-light geometry audit using Natural Earth
country polygons and matplotlib.path point-in-polygon tests. It does not alter
existing model outputs; it quantifies whether country/land geometry is likely
to affect the completed regional results.
"""
from __future__ import annotations

from argparse import ArgumentParser, Namespace
from pathlib import Path
import json
import shutil
import urllib.request

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.path import Path as MplPath
import numpy as np
import pandas as pd
import xarray as xr

from region_config import REGIONS, Region, resolve_region


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED = PROJECT_ROOT / "data" / "processed"
REGION_DATA_ROOT = PROCESSED / "regions"
METADATA_ROOT = PROJECT_ROOT / "data" / "metadata" / "natural_earth"
OUT_ROOT = PROJECT_ROOT / "outputs" / "multiregion"
REPORT_ROOT = PROJECT_ROOT / "results" / "multiregion"

COUNTRY_GEOJSON_URL = (
    "https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/"
    "geojson/ne_50m_admin_0_countries.geojson"
)
COUNTRY_GEOJSON = METADATA_ROOT / "ne_50m_admin_0_countries.geojson"

DEFAULT_REGIONS = ("cvalley", "southern_great_plains", "mediterranean_spain")
DEFAULT_START_YEAR = 1991
DEFAULT_END_YEAR = 2026
ROUND_DECIMALS = 6


def parse_args() -> Namespace:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument(
        "--regions",
        nargs="+",
        default=list(DEFAULT_REGIONS),
        help="Region slugs or aliases to audit. Defaults to completed full-resolution regions.",
    )
    parser.add_argument(
        "--include-all-configured",
        action="store_true",
        help="Audit every configured region that has local CHIRPS/dataset files.",
    )
    parser.add_argument("--start-year", type=int, default=DEFAULT_START_YEAR)
    parser.add_argument("--end-year", type=int, default=DEFAULT_END_YEAR)
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Redownload Natural Earth country GeoJSON even if cached locally.",
    )
    parser.add_argument(
        "--copy-report",
        action="store_true",
        help="Copy diagnostics CSV and figure into results/multiregion/.",
    )
    parser.add_argument(
        "--fail-missing",
        action="store_true",
        help="Raise an error if a requested region lacks local data files.",
    )
    return parser.parse_args()


def region_paths(region: Region, start_year: int, end_year: int) -> dict[str, Path]:
    if region.slug == "cvalley":
        return {
            "pr": PROCESSED / f"chirps_v3_monthly_cvalley_{start_year}_{end_year}.nc",
            "dataset": PROCESSED / "dataset_forecast.parquet",
        }

    rdir = REGION_DATA_ROOT / region.slug
    return {
        "pr": rdir / f"chirps_v3_monthly_{region.slug}_{start_year}_{end_year}.nc",
        "dataset": rdir / f"dataset_forecast_{region.slug}.parquet",
    }


def mask_dir(region: Region) -> Path:
    return REGION_DATA_ROOT / region.slug / "masks"


def download_country_geojson(force: bool = False) -> Path:
    if COUNTRY_GEOJSON.exists() and not force:
        print(f"Using cached Natural Earth countries: {COUNTRY_GEOJSON}")
        return COUNTRY_GEOJSON

    METADATA_ROOT.mkdir(parents=True, exist_ok=True)
    print(f"Downloading Natural Earth countries from {COUNTRY_GEOJSON_URL}")
    with urllib.request.urlopen(COUNTRY_GEOJSON_URL, timeout=120) as response:
        payload = response.read()
    COUNTRY_GEOJSON.write_bytes(payload)
    print(f"Wrote Natural Earth countries: {COUNTRY_GEOJSON} ({len(payload):,} bytes)")
    return COUNTRY_GEOJSON


def feature_country_names(feature: dict[str, object]) -> set[str]:
    props = feature.get("properties") or {}
    if not isinstance(props, dict):
        return set()
    keys = ("ADMIN", "NAME", "NAME_EN", "BRK_NAME", "SOVEREIGNT")
    return {str(props[key]) for key in keys if props.get(key)}


def select_country_features(features: list[dict[str, object]], countries: tuple[str, ...]) -> list[dict[str, object]]:
    requested = {country.casefold() for country in countries}
    selected = [
        feature
        for feature in features
        if {name.casefold() for name in feature_country_names(feature)} & requested
    ]
    if selected:
        return selected

    available = sorted({name for feature in features for name in feature_country_names(feature)})
    nearby = ", ".join(available[:30])
    raise ValueError(
        f"No Natural Earth features matched {countries}. "
        f"First available country names: {nearby}"
    )


def iter_polygon_rings(geometry: dict[str, object]):
    geom_type = geometry.get("type")
    coords = geometry.get("coordinates") or []
    if geom_type == "Polygon":
        yield coords
    elif geom_type == "MultiPolygon":
        for polygon in coords:
            yield polygon


def ring_contains_points(ring: list[list[float]], points: np.ndarray) -> np.ndarray:
    arr = np.asarray(ring, dtype=float)
    if arr.ndim != 2 or arr.shape[0] < 3 or arr.shape[1] < 2:
        return np.zeros(points.shape[0], dtype=bool)
    return MplPath(arr[:, :2], closed=True).contains_points(points, radius=1e-10)


def geometry_contains_points(geometry: dict[str, object], points: np.ndarray) -> np.ndarray:
    mask = np.zeros(points.shape[0], dtype=bool)
    for rings in iter_polygon_rings(geometry):
        if not rings:
            continue
        polygon_mask = ring_contains_points(rings[0], points)
        if not polygon_mask.any():
            continue
        for hole in rings[1:]:
            polygon_mask &= ~ring_contains_points(hole, points)
        mask |= polygon_mask
    return mask


def country_mask(
    features: list[dict[str, object]],
    countries: tuple[str, ...],
    lon2d: np.ndarray,
    lat2d: np.ndarray,
) -> np.ndarray:
    points = np.column_stack([lon2d.ravel(), lat2d.ravel()])
    selected = select_country_features(features, countries)
    flat = np.zeros(points.shape[0], dtype=bool)
    for feature in selected:
        geometry = feature.get("geometry") or {}
        if isinstance(geometry, dict):
            flat |= geometry_contains_points(geometry, points)
    return flat.reshape(lat2d.shape)


def read_country_features(path: Path) -> list[dict[str, object]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    features = data.get("features", [])
    if not isinstance(features, list):
        raise ValueError(f"Unexpected GeoJSON feature collection in {path}")
    return features


def load_pr_grid(path: Path) -> tuple[xr.Dataset, xr.DataArray]:
    ds = xr.open_dataset(path)
    if "pr" in ds.data_vars:
        pr = ds["pr"]
    else:
        pr = ds[list(ds.data_vars)[0]]
    if "lat" in pr.coords:
        pr = pr.rename({"lat": "latitude"})
    if "lon" in pr.coords:
        pr = pr.rename({"lon": "longitude"})
    return ds, pr


def coordinate_lookup(lat2d: np.ndarray, lon2d: np.ndarray, mask: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "latitude_key": np.round(lat2d.ravel(), ROUND_DECIMALS),
            "longitude_key": np.round(lon2d.ravel(), ROUND_DECIMALS),
            "country_mask": mask.ravel(),
        }
    )


def dataset_pixel_diagnostics(dataset_path: Path, lookup: pd.DataFrame) -> dict[str, float | int]:
    if not dataset_path.exists():
        return {
            "dataset_exists": False,
            "dataset_rows": 0,
            "dataset_pixels": 0,
            "dataset_pixels_in_country": 0,
            "dataset_pixels_outside_country": 0,
            "dataset_country_fraction": np.nan,
            "dataset_rows_in_country": 0,
            "dataset_rows_outside_country": 0,
            "dataset_row_country_fraction": np.nan,
        }

    coords = pd.read_parquet(dataset_path, columns=["latitude", "longitude"])
    coords["latitude_key"] = coords["latitude"].round(ROUND_DECIMALS)
    coords["longitude_key"] = coords["longitude"].round(ROUND_DECIMALS)
    counts = (
        coords.groupby(["latitude_key", "longitude_key"], observed=True)
        .size()
        .reset_index(name="n_rows")
    )
    merged = counts.merge(lookup, on=["latitude_key", "longitude_key"], how="left")
    missing = int(merged["country_mask"].isna().sum())
    if missing:
        raise ValueError(
            f"{dataset_path} has {missing} coordinate pairs not found in the CHIRPS grid."
        )
    merged["country_mask"] = merged["country_mask"].astype(bool)

    pixels = int(len(merged))
    pixels_in = int(merged["country_mask"].sum())
    rows = int(merged["n_rows"].sum())
    rows_in = int(merged.loc[merged["country_mask"], "n_rows"].sum())
    return {
        "dataset_exists": True,
        "dataset_rows": rows,
        "dataset_pixels": pixels,
        "dataset_pixels_in_country": pixels_in,
        "dataset_pixels_outside_country": pixels - pixels_in,
        "dataset_country_fraction": pixels_in / pixels if pixels else np.nan,
        "dataset_rows_in_country": rows_in,
        "dataset_rows_outside_country": rows - rows_in,
        "dataset_row_country_fraction": rows_in / rows if rows else np.nan,
    }


def write_mask_products(
    region: Region,
    pr: xr.DataArray,
    valid_pr: np.ndarray,
    mask: np.ndarray,
    lat2d: np.ndarray,
    lon2d: np.ndarray,
) -> tuple[Path, Path]:
    out_dir = mask_dir(region)
    out_dir.mkdir(parents=True, exist_ok=True)
    nc_path = out_dir / f"{region.slug}_country_mask.nc"
    csv_path = out_dir / f"{region.slug}_country_mask_pixels.csv"

    ds = xr.Dataset(
        data_vars={
            "country_mask": (
                ("latitude", "longitude"),
                mask.astype(np.int8),
                {"description": "1 where grid-cell center falls inside configured Natural Earth countries"},
            ),
            "valid_pr_any": (
                ("latitude", "longitude"),
                valid_pr.astype(np.int8),
                {"description": "1 where CHIRPS precipitation is non-missing in at least one month"},
            ),
        },
        coords={
            "latitude": pr["latitude"].values,
            "longitude": pr["longitude"].values,
        },
        attrs={
            "region": region.slug,
            "region_name": region.name,
            "mask_countries": ", ".join(region.mask_countries),
            "mask_note": region.mask_note,
            "source": COUNTRY_GEOJSON_URL,
        },
    )
    ds.to_netcdf(nc_path)

    pd.DataFrame(
        {
            "latitude": lat2d.ravel(),
            "longitude": lon2d.ravel(),
            "country_mask": mask.ravel(),
            "valid_pr_any": valid_pr.ravel(),
            "valid_in_country": (mask & valid_pr).ravel(),
            "valid_outside_country": ((~mask) & valid_pr).ravel(),
        }
    ).to_csv(csv_path, index=False)

    return nc_path, csv_path


def audit_region(region: Region, features: list[dict[str, object]], args: Namespace) -> dict[str, object] | None:
    if not region.mask_countries:
        message = f"Skipping {region.slug}: no mask_countries configured."
        if args.fail_missing:
            raise ValueError(message)
        print(message)
        return None

    paths = region_paths(region, args.start_year, args.end_year)
    missing = [name for name, path in paths.items() if not path.exists()]
    if missing:
        message = (
            f"Skipping {region.slug}: missing "
            + ", ".join(f"{name}={paths[name]}" for name in missing)
        )
        if args.fail_missing:
            raise FileNotFoundError(message)
        print(message)
        return None

    print(f"\nAuditing {region.name} ({region.slug})")
    print(f"  Countries: {', '.join(region.mask_countries)}")
    ds, pr = load_pr_grid(paths["pr"])
    try:
        lats = pr["latitude"].values
        lons = pr["longitude"].values
        lon2d, lat2d = np.meshgrid(lons, lats)
        valid_pr = pr.notnull().any("time").values.astype(bool)
        mask = country_mask(features, region.mask_countries, lon2d, lat2d)

        lookup = coordinate_lookup(lat2d, lon2d, mask)
        dataset_diag = dataset_pixel_diagnostics(paths["dataset"], lookup)
        nc_path, csv_path = write_mask_products(region, pr, valid_pr, mask, lat2d, lon2d)
    finally:
        ds.close()

    grid_cells = int(valid_pr.size)
    valid_cells = int(valid_pr.sum())
    mask_cells = int(mask.sum())
    valid_in = int((mask & valid_pr).sum())
    valid_out = int(((~mask) & valid_pr).sum())
    country_valid_fraction = valid_in / valid_cells if valid_cells else np.nan
    outside_fraction = valid_out / valid_cells if valid_cells else np.nan

    print(
        "  Valid CHIRPS cells in country mask: "
        f"{valid_in:,}/{valid_cells:,} ({country_valid_fraction:.3%}); "
        f"outside={valid_out:,} ({outside_fraction:.3%})"
    )
    if dataset_diag["dataset_exists"]:
        print(
            "  Forecast dataset pixels in country mask: "
            f"{dataset_diag['dataset_pixels_in_country']:,}/{dataset_diag['dataset_pixels']:,} "
            f"({dataset_diag['dataset_country_fraction']:.3%})"
        )
    print(f"  Wrote mask: {nc_path}")
    print(f"  Wrote pixel table: {csv_path}")

    return {
        "region": region.slug,
        "region_name": region.name,
        "countries": "; ".join(region.mask_countries),
        "mask_note": region.mask_note,
        "pr_file": str(paths["pr"].relative_to(PROJECT_ROOT)),
        "dataset_file": str(paths["dataset"].relative_to(PROJECT_ROOT)),
        "mask_file": str(nc_path.relative_to(PROJECT_ROOT)),
        "pixel_file": str(csv_path.relative_to(PROJECT_ROOT)),
        "lat_min": float(np.nanmin(lats)),
        "lat_max": float(np.nanmax(lats)),
        "lon_min": float(np.nanmin(lons)),
        "lon_max": float(np.nanmax(lons)),
        "grid_cells": grid_cells,
        "valid_pr_cells": valid_cells,
        "country_mask_cells": mask_cells,
        "valid_in_country_cells": valid_in,
        "valid_outside_country_cells": valid_out,
        "valid_country_fraction": country_valid_fraction,
        "valid_outside_country_fraction": outside_fraction,
        **dataset_diag,
    }


def plot_masks(rows: list[dict[str, object]], out_path: Path) -> None:
    if not rows:
        return
    n = len(rows)
    fig, axes = plt.subplots(1, n, figsize=(5.2 * n, 4.2), squeeze=False)

    for ax, row in zip(axes.ravel(), rows):
        mask_ds = xr.open_dataset(PROJECT_ROOT / str(row["mask_file"]))
        try:
            country = mask_ds["country_mask"].values.astype(bool)
            valid = mask_ds["valid_pr_any"].values.astype(bool)
            lats = mask_ds["latitude"].values
            lons = mask_ds["longitude"].values
            lon2d, lat2d = np.meshgrid(lons, lats)

            valid_in = valid & country
            valid_out = valid & ~country
            country_no_pr = country & ~valid

            ax.scatter(lon2d[valid], lat2d[valid], s=2, color="#d0d0d0", linewidths=0)
            if country_no_pr.any():
                ax.scatter(
                    lon2d[country_no_pr],
                    lat2d[country_no_pr],
                    s=2,
                    color="#9ecae1",
                    linewidths=0,
                )
            if valid_out.any():
                ax.scatter(
                    lon2d[valid_out],
                    lat2d[valid_out],
                    s=3,
                    color="#d95f02",
                    linewidths=0,
                )
            ax.scatter(lon2d[valid_in], lat2d[valid_in], s=2, color="#1b9e77", linewidths=0)

            retained = float(row["valid_country_fraction"])
            ax.set_title(f"{row['region']}\nvalid retained: {retained:.1%}", fontsize=10)
            ax.set_xlabel("Longitude")
            ax.set_ylabel("Latitude")
            ax.set_xlim(float(row["lon_min"]), float(row["lon_max"]))
            ax.set_ylim(float(row["lat_min"]), float(row["lat_max"]))
            ax.grid(True, color="#eeeeee", linewidth=0.6)
        finally:
            mask_ds.close()

    handles = [
        Line2D([0], [0], marker="o", color="w", label="valid in country", markerfacecolor="#1b9e77", markersize=6),
        Line2D([0], [0], marker="o", color="w", label="valid outside country", markerfacecolor="#d95f02", markersize=6),
        Line2D([0], [0], marker="o", color="w", label="country without CHIRPS valid PR", markerfacecolor="#9ecae1", markersize=6),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False)
    fig.suptitle("Country-mask audit for completed multi-region grids", y=0.98, fontsize=13)
    fig.tight_layout(rect=(0, 0.08, 1, 0.94))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def write_geometry_notes(rows: list[dict[str, object]], out_path: Path) -> None:
    lines = [
        "Region Geometry Diagnostics",
        "===========================",
        "",
        "Country masks use Natural Earth 1:50m country polygons and CHIRPS grid-cell centers.",
        "They are a first-pass rectangular-bbox audit, not a replacement for basin polygons.",
        "",
    ]
    for row in rows:
        valid_fraction = float(row["valid_country_fraction"])
        outside_fraction = float(row["valid_outside_country_fraction"])
        dataset_fraction = float(row["dataset_country_fraction"])
        lines.extend(
            [
                f"{row['region']} - {row['region_name']}",
                f"  Countries: {row['countries']}",
                (
                    "  Valid CHIRPS cells retained: "
                    f"{int(row['valid_in_country_cells']):,}/{int(row['valid_pr_cells']):,} "
                    f"({valid_fraction:.2%}); outside country: "
                    f"{int(row['valid_outside_country_cells']):,} ({outside_fraction:.2%})."
                ),
                (
                    "  Forecast dataset pixels retained: "
                    f"{int(row['dataset_pixels_in_country']):,}/{int(row['dataset_pixels']):,} "
                    f"({dataset_fraction:.2%})."
                ),
                f"  Interpretation: {interpret_row(row)}",
                f"  Caveat: {row['mask_note']}",
                "",
            ]
        )
    out_path.write_text("\n".join(lines), encoding="utf-8")


def interpret_row(row: dict[str, object]) -> str:
    outside_fraction = float(row["valid_outside_country_fraction"])
    if outside_fraction == 0:
        return "Country geometry does not remove any valid CHIRPS cells from this bbox."
    if outside_fraction < 0.01:
        return "Country geometry removes a negligible number of valid cells; retraining is not urgent."
    if outside_fraction < 0.05:
        return "Country geometry removes a small but measurable cell fraction; masked sensitivity is useful."
    return "Country geometry removes enough valid cells that a masked rerun should precede final claims."


def copy_reports(paths: list[Path]) -> None:
    REPORT_ROOT.mkdir(parents=True, exist_ok=True)
    for path in paths:
        if path.exists():
            shutil.copy2(path, REPORT_ROOT / path.name)
            print(f"Copied report artifact: {REPORT_ROOT / path.name}")


def main() -> None:
    args = parse_args()
    region_names = list(REGIONS) if args.include_all_configured else args.regions
    regions = [resolve_region(name) for name in region_names]

    geojson_path = download_country_geojson(force=args.force_download)
    features = read_country_features(geojson_path)
    print(f"Loaded {len(features):,} Natural Earth country features")

    rows = []
    for region in regions:
        row = audit_region(region, features, args)
        if row is not None:
            rows.append(row)

    if not rows:
        raise SystemExit("No regions were audited.")

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    diagnostics_path = OUT_ROOT / "region_mask_diagnostics.csv"
    figure_path = OUT_ROOT / "region_country_masks.png"
    notes_path = OUT_ROOT / "region_geometry_notes.txt"

    df = pd.DataFrame(rows)
    df.to_csv(diagnostics_path, index=False)
    plot_masks(rows, figure_path)
    write_geometry_notes(rows, notes_path)

    print(f"\nWrote diagnostics: {diagnostics_path}")
    print(f"Wrote figure: {figure_path}")
    print(f"Wrote notes: {notes_path}")

    if args.copy_report:
        copy_reports([diagnostics_path, figure_path, notes_path])


if __name__ == "__main__":
    main()
