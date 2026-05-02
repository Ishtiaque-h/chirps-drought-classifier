#!/usr/bin/env python
"""
Build basin/hydroclimate masks for high-priority multi-region experiments.

Country masks are useful first-pass checks, but publication-level regional
claims need boundaries closer to the scientific object being studied. This
script builds:
  - Central Valley: CA DWR Bulletin 118 Sacramento Valley and San Joaquin
    Valley groundwater basin/subbasin polygons.
  - Mediterranean Spain: MITECO river basin district polygons for Ebro,
    Catalonia internal basins, Jucar, Segura, Andalusia Mediterranean basins,
    and Guadalquivir.
  - Southern Great Plains: US EPA Level III ecoregion polygons in the South
    Central Semi-Arid Prairies Level II region.

The outputs are mask NetCDF files that can be passed to
run_multiregion_xgb_experiment.py with --basin-mask.
"""
from __future__ import annotations

from argparse import ArgumentParser, Namespace
from pathlib import Path
import json
import shutil
import tempfile
import unicodedata
import urllib.parse
import urllib.request
import zipfile

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import shapefile
from shapely import intersects_xy, union_all
from shapely.geometry import box
from shapely.geometry import shape
from shapely.ops import transform
from pyproj import CRS, Transformer
import xarray as xr

from region_config import Region, resolve_region


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED = PROJECT_ROOT / "data" / "processed"
REGION_DATA_ROOT = PROCESSED / "regions"
METADATA_ROOT = PROJECT_ROOT / "data" / "metadata"
OUT_ROOT = PROJECT_ROOT / "outputs" / "multiregion"
REPORT_ROOT = PROJECT_ROOT / "results" / "multiregion"

DWR_B118_QUERY_URL = (
    "https://gis.water.ca.gov/arcgis/rest/services/Geoscientific/"
    "i08_B118_CA_GroundwaterBasins/FeatureServer/0/query"
)
MITECO_OGC_ITEMS_URL = (
    "https://wmts.mapama.gob.es/sig-api/ogc/features/v1/collections/"
    "agua%3ADemarcaciones_ET/items"
)
EPA_LEVEL3_ZIP_URL = (
    "https://dmap-prod-oms-edc.s3.us-east-1.amazonaws.com/ORD/Ecoregions/us/"
    "us_eco_l3_state_boundaries.zip"
)
EPA_ECOREGION_PAGE = (
    "https://www.epa.gov/eco-research/level-iii-and-iv-ecoregions-continental-united-states"
)

DEFAULT_REGIONS = ("cvalley", "southern_great_plains", "mediterranean_spain")
DEFAULT_START_YEAR = 1991
DEFAULT_END_YEAR = 2026
ROUND_DECIMALS = 6

CENTRAL_VALLEY_BASIN_NUMBERS = ("5-021", "5-022")
SPAIN_BASIN_DISTRICTS = (
    "EBRO",
    "CUENCAS INTERNAS DE CATALUNA",
    "JUCAR",
    "SEGURA",
    "CUENCAS MEDITERRANEAS ANDALUZAS",
    "GUADALQUIVIR",
)
SOUTHERN_GREAT_PLAINS_LEVEL2 = "SOUTH CENTRAL SEMI-ARID PRAIRIES"


def parse_args() -> Namespace:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument(
        "--regions",
        nargs="+",
        default=list(DEFAULT_REGIONS),
        help="Region slugs or aliases to audit. Defaults to the three full-resolution priority regions.",
    )
    parser.add_argument("--start-year", type=int, default=DEFAULT_START_YEAR)
    parser.add_argument("--end-year", type=int, default=DEFAULT_END_YEAR)
    parser.add_argument("--force-download", action="store_true", help="Refresh boundary GeoJSON caches.")
    parser.add_argument("--copy-report", action="store_true", help="Copy diagnostics into results/multiregion/.")
    parser.add_argument("--fail-missing", action="store_true", help="Raise if a requested region lacks data.")
    return parser.parse_args()


def normalize_name(value: object) -> str:
    text = unicodedata.normalize("NFKD", str(value).upper())
    return "".join(ch for ch in text if not unicodedata.combining(ch)).strip()


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


def request_json(url: str, params: dict[str, object] | None = None, timeout: int = 180) -> dict[str, object]:
    full_url = url if params is None else url + "?" + urllib.parse.urlencode(params)
    request = urllib.request.Request(full_url, headers={"User-Agent": "chirps-drought-classifier/1.0"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def write_geojson(path: Path, features: list[dict[str, object]], source_url: str, source_note: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "type": "FeatureCollection",
        "features": features,
        "properties": {
            "source_url": source_url,
            "source_note": source_note,
        },
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def download_file(url: str, path: Path, force: bool) -> Path:
    if path.exists() and not force:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": "chirps-drought-classifier/1.0"})
    with urllib.request.urlopen(request, timeout=240) as response:
        path.write_bytes(response.read())
    return path


def load_cached_geojson(path: Path) -> list[dict[str, object]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    features = data.get("features", [])
    if not isinstance(features, list):
        raise ValueError(f"Unexpected GeoJSON in {path}")
    return features


def pyshp_shape_to_geojson(shape_obj: shapefile.Shape) -> dict[str, object]:
    return shape_obj.__geo_interface__


def central_valley_features(force: bool) -> tuple[list[dict[str, object]], Path, str, str]:
    out_path = METADATA_ROOT / "dwr" / "central_valley_b118_groundwater_basins.geojson"
    source_note = (
        "CA DWR Bulletin 118 groundwater basin polygons; selected Basin_Number "
        "5-021 Sacramento Valley and 5-022 San Joaquin Valley."
    )
    if out_path.exists() and not force:
        return load_cached_geojson(out_path), out_path, DWR_B118_QUERY_URL, source_note

    where = " OR ".join(f"Basin_Number = '{number}'" for number in CENTRAL_VALLEY_BASIN_NUMBERS)
    data = request_json(
        DWR_B118_QUERY_URL,
        {
            "where": where,
            "outFields": "*",
            "returnGeometry": "true",
            "outSR": "4326",
            "f": "geojson",
            "resultRecordCount": 2000,
        },
    )
    features = data.get("features", [])
    if not features:
        raise ValueError("DWR Bulletin 118 query returned no Central Valley basin features")
    write_geojson(out_path, features, DWR_B118_QUERY_URL, source_note)
    return features, out_path, DWR_B118_QUERY_URL, source_note


def spain_basin_features(force: bool) -> tuple[list[dict[str, object]], Path, str, str]:
    out_path = METADATA_ROOT / "miteco" / "mediterranean_spain_river_basin_districts.geojson"
    source_note = (
        "MITECO terrestrial river basin districts; selected Ebro, Catalonia internal "
        "basins, Jucar, Segura, Andalusia Mediterranean basins, and Guadalquivir."
    )
    if out_path.exists() and not force:
        return load_cached_geojson(out_path), out_path, MITECO_OGC_ITEMS_URL, source_note

    data = request_json(
        MITECO_OGC_ITEMS_URL,
        {"f": "application/geo+json", "limit": 100},
    )
    wanted = {normalize_name(name) for name in SPAIN_BASIN_DISTRICTS}
    features = []
    for feature in data.get("features", []):
        props = feature.get("properties") or {}
        name = normalize_name(props.get("nom_demar", ""))
        if name in wanted:
            features.append(feature)
    found = {normalize_name((feature.get("properties") or {}).get("nom_demar", "")) for feature in features}
    missing = wanted.difference(found)
    if missing:
        raise ValueError(f"MITECO basin district query did not return expected names: {sorted(missing)}")
    write_geojson(out_path, features, MITECO_OGC_ITEMS_URL, source_note)
    return features, out_path, MITECO_OGC_ITEMS_URL, source_note


def southern_great_plains_features(region: Region, force: bool) -> tuple[list[dict[str, object]], Path, str, str]:
    out_path = METADATA_ROOT / "epa" / "southern_great_plains_level2_ecoregions.geojson"
    source_note = (
        "US EPA Level III ecoregions with state boundaries; selected features in "
        f"NA_L2NAME={SOUTHERN_GREAT_PLAINS_LEVEL2} that intersect the configured "
        "Southern Great Plains bbox."
    )
    if out_path.exists() and not force:
        return load_cached_geojson(out_path), out_path, EPA_LEVEL3_ZIP_URL, source_note

    zip_path = METADATA_ROOT / "epa" / "us_eco_l3_state_boundaries.zip"
    download_file(EPA_LEVEL3_ZIP_URL, zip_path, force=force)
    bbox_ll = box(region.lon_min, region.lat_min, region.lon_max, region.lat_max)
    features: list[dict[str, object]] = []

    with tempfile.TemporaryDirectory() as tmpdir:
        with zipfile.ZipFile(zip_path) as archive:
            archive.extractall(tmpdir)
        shp_path = next(Path(tmpdir).rglob("*.shp"))
        source_crs = CRS.from_wkt(shp_path.with_suffix(".prj").read_text())
        to_source = Transformer.from_crs("EPSG:4326", source_crs, always_xy=True).transform
        to_wgs84 = Transformer.from_crs(source_crs, "EPSG:4326", always_xy=True).transform
        bbox_source = transform(to_source, bbox_ll)

        reader = shapefile.Reader(str(shp_path))
        for shape_record in reader.iterShapeRecords():
            record = shape_record.record.as_dict()
            if normalize_name(record.get("NA_L2NAME", "")) != SOUTHERN_GREAT_PLAINS_LEVEL2:
                continue
            geom_source = shape(pyshp_shape_to_geojson(shape_record.shape))
            if not geom_source.intersects(bbox_source):
                continue
            geom_wgs84 = transform(to_wgs84, geom_source)
            features.append(
                {
                    "type": "Feature",
                    "properties": record,
                    "geometry": geom_wgs84.__geo_interface__,
                }
            )

    if not features:
        raise ValueError("EPA ecoregion selection returned no Southern Great Plains features")
    write_geojson(out_path, features, EPA_LEVEL3_ZIP_URL, source_note)
    return features, out_path, EPA_LEVEL3_ZIP_URL, source_note


def boundary_features(region: Region, force: bool) -> tuple[list[dict[str, object]], Path, str, str, str]:
    if region.slug == "cvalley":
        features, path, source_url, source_note = central_valley_features(force)
        return features, path, source_url, source_note, "Central Valley DWR Bulletin 118 groundwater basins"
    if region.slug == "southern_great_plains":
        features, path, source_url, source_note = southern_great_plains_features(region, force)
        return features, path, source_url, source_note, "Southern Great Plains EPA South Central Semi-Arid Prairies"
    if region.slug == "mediterranean_spain":
        features, path, source_url, source_note = spain_basin_features(force)
        return features, path, source_url, source_note, "Mediterranean/eastern-southern Spain river basin districts"
    raise KeyError(f"No basin-mask specification exists for region '{region.slug}'")


def geometry_from_features(features: list[dict[str, object]]):
    geometries = []
    for feature in features:
        geom = feature.get("geometry")
        if geom:
            shaped = shape(geom)
            if not shaped.is_empty:
                geometries.append(shaped)
    if not geometries:
        raise ValueError("Boundary feature collection has no usable geometries")
    return union_all(geometries)


def load_pr_grid(path: Path) -> tuple[xr.Dataset, xr.DataArray]:
    ds = xr.open_dataset(path)
    pr = ds["pr"] if "pr" in ds.data_vars else ds[list(ds.data_vars)[0]]
    if "lat" in pr.coords:
        pr = pr.rename({"lat": "latitude"})
    if "lon" in pr.coords:
        pr = pr.rename({"lon": "longitude"})
    return ds, pr


def dataset_pixel_diagnostics(dataset_path: Path, lat2d: np.ndarray, lon2d: np.ndarray, mask: np.ndarray) -> dict[str, object]:
    if not dataset_path.exists():
        return {
            "dataset_exists": False,
            "dataset_rows": 0,
            "dataset_pixels": 0,
            "dataset_pixels_in_basin": 0,
            "dataset_pixels_outside_basin": 0,
            "dataset_basin_fraction": np.nan,
            "dataset_rows_in_basin": 0,
            "dataset_rows_outside_basin": 0,
            "dataset_row_basin_fraction": np.nan,
        }
    lookup = pd.DataFrame(
        {
            "latitude_key": np.round(lat2d.ravel(), ROUND_DECIMALS),
            "longitude_key": np.round(lon2d.ravel(), ROUND_DECIMALS),
            "basin_mask": mask.ravel(),
        }
    )
    coords = pd.read_parquet(dataset_path, columns=["latitude", "longitude"])
    coords["latitude_key"] = coords["latitude"].round(ROUND_DECIMALS)
    coords["longitude_key"] = coords["longitude"].round(ROUND_DECIMALS)
    counts = (
        coords.groupby(["latitude_key", "longitude_key"], observed=True)
        .size()
        .reset_index(name="n_rows")
    )
    merged = counts.merge(lookup, on=["latitude_key", "longitude_key"], how="left")
    missing = int(merged["basin_mask"].isna().sum())
    if missing:
        raise ValueError(f"{dataset_path} has {missing} coordinate pairs not found in the CHIRPS grid.")
    merged["basin_mask"] = merged["basin_mask"].astype(bool)
    pixels = int(len(merged))
    pixels_in = int(merged["basin_mask"].sum())
    rows = int(merged["n_rows"].sum())
    rows_in = int(merged.loc[merged["basin_mask"], "n_rows"].sum())
    return {
        "dataset_exists": True,
        "dataset_rows": rows,
        "dataset_pixels": pixels,
        "dataset_pixels_in_basin": pixels_in,
        "dataset_pixels_outside_basin": pixels - pixels_in,
        "dataset_basin_fraction": pixels_in / pixels if pixels else np.nan,
        "dataset_rows_in_basin": rows_in,
        "dataset_rows_outside_basin": rows - rows_in,
        "dataset_row_basin_fraction": rows_in / rows if rows else np.nan,
    }


def write_mask_products(
    region: Region,
    pr: xr.DataArray,
    valid_pr: np.ndarray,
    basin_mask: np.ndarray,
    lat2d: np.ndarray,
    lon2d: np.ndarray,
    boundary_path: Path,
    source_url: str,
    source_note: str,
    mask_label: str,
) -> tuple[Path, Path]:
    out_dir = mask_dir(region)
    out_dir.mkdir(parents=True, exist_ok=True)
    nc_path = out_dir / f"{region.slug}_basin_mask.nc"
    csv_path = out_dir / f"{region.slug}_basin_mask_pixels.csv"

    ds = xr.Dataset(
        data_vars={
            "basin_mask": (
                ("latitude", "longitude"),
                basin_mask.astype(np.int8),
                {"description": f"1 where grid-cell center falls inside {mask_label}"},
            ),
            "valid_pr_any": (
                ("latitude", "longitude"),
                valid_pr.astype(np.int8),
                {"description": "1 where CHIRPS precipitation is non-missing in at least one month"},
            ),
        },
        coords={"latitude": pr["latitude"].values, "longitude": pr["longitude"].values},
        attrs={
            "region": region.slug,
            "region_name": region.name,
            "mask_label": mask_label,
            "boundary_file": str(boundary_path.relative_to(PROJECT_ROOT)),
            "source_url": source_url,
            "source_note": source_note,
        },
    )
    ds.to_netcdf(nc_path)

    pd.DataFrame(
        {
            "latitude": lat2d.ravel(),
            "longitude": lon2d.ravel(),
            "basin_mask": basin_mask.ravel(),
            "valid_pr_any": valid_pr.ravel(),
            "valid_in_basin": (basin_mask & valid_pr).ravel(),
            "valid_outside_basin": ((~basin_mask) & valid_pr).ravel(),
        }
    ).to_csv(csv_path, index=False)
    return nc_path, csv_path


def audit_region(region: Region, args: Namespace) -> dict[str, object] | None:
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

    features, boundary_path, source_url, source_note, mask_label = boundary_features(region, args.force_download)
    geom = geometry_from_features(features)
    feature_names = sorted(
        {
            str((feature.get("properties") or {}).get("Basin_Subbasin_Name")
                or (feature.get("properties") or {}).get("nom_demar")
                or (feature.get("properties") or {}).get("US_L3NAME")
                or feature.get("id"))
            for feature in features
        }
    )

    print(f"\nAuditing {region.name} ({region.slug})")
    print(f"  Mask: {mask_label}")
    print(f"  Boundary features: {len(features):,}")
    ds, pr = load_pr_grid(paths["pr"])
    try:
        lats = pr["latitude"].values
        lons = pr["longitude"].values
        lon2d, lat2d = np.meshgrid(lons, lats)
        valid_pr = pr.notnull().any("time").values.astype(bool)
        basin_mask = intersects_xy(geom, lon2d.ravel(), lat2d.ravel()).reshape(lat2d.shape)
        dataset_diag = dataset_pixel_diagnostics(paths["dataset"], lat2d, lon2d, basin_mask)
        nc_path, csv_path = write_mask_products(
            region,
            pr,
            valid_pr,
            basin_mask,
            lat2d,
            lon2d,
            boundary_path,
            source_url,
            source_note,
            mask_label,
        )
    finally:
        ds.close()

    grid_cells = int(valid_pr.size)
    valid_cells = int(valid_pr.sum())
    mask_cells = int(basin_mask.sum())
    valid_in = int((basin_mask & valid_pr).sum())
    valid_out = int(((~basin_mask) & valid_pr).sum())
    valid_fraction = valid_in / valid_cells if valid_cells else np.nan
    outside_fraction = valid_out / valid_cells if valid_cells else np.nan
    print(
        "  Valid CHIRPS cells in basin mask: "
        f"{valid_in:,}/{valid_cells:,} ({valid_fraction:.3%}); "
        f"outside={valid_out:,} ({outside_fraction:.3%})"
    )
    if dataset_diag["dataset_exists"]:
        print(
            "  Forecast dataset pixels in basin mask: "
            f"{dataset_diag['dataset_pixels_in_basin']:,}/{dataset_diag['dataset_pixels']:,} "
            f"({dataset_diag['dataset_basin_fraction']:.3%})"
        )
    print(f"  Wrote mask: {nc_path}")
    print(f"  Wrote pixel table: {csv_path}")

    return {
        "region": region.slug,
        "region_name": region.name,
        "mask_label": mask_label,
        "source_url": source_url,
        "source_note": source_note,
        "boundary_file": str(boundary_path.relative_to(PROJECT_ROOT)),
        "selected_boundary_features": "; ".join(feature_names),
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
        "basin_mask_cells": mask_cells,
        "valid_in_basin_cells": valid_in,
        "valid_outside_basin_cells": valid_out,
        "valid_basin_fraction": valid_fraction,
        "valid_outside_basin_fraction": outside_fraction,
        **dataset_diag,
    }


def interpret_row(row: dict[str, object]) -> str:
    outside_fraction = float(row["valid_outside_basin_fraction"])
    if outside_fraction < 0.01:
        return "Basin geometry is nearly identical to the current valid-cell sample."
    if outside_fraction < 0.10:
        return "Basin geometry removes a measurable fraction; masked rerun is useful."
    return "Basin geometry materially changes the sample; masked rerun should be treated as the cleaner checkpoint."


def plot_masks(rows: list[dict[str, object]], out_path: Path) -> None:
    if not rows:
        return
    n = len(rows)
    fig, axes = plt.subplots(1, n, figsize=(5.3 * n, 4.3), squeeze=False)
    for ax, row in zip(axes.ravel(), rows):
        mask_ds = xr.open_dataset(PROJECT_ROOT / str(row["mask_file"]))
        try:
            basin = mask_ds["basin_mask"].values.astype(bool)
            valid = mask_ds["valid_pr_any"].values.astype(bool)
            lats = mask_ds["latitude"].values
            lons = mask_ds["longitude"].values
            lon2d, lat2d = np.meshgrid(lons, lats)
            valid_in = valid & basin
            valid_out = valid & ~basin
            basin_no_pr = basin & ~valid

            ax.scatter(lon2d[valid], lat2d[valid], s=2, color="#d0d0d0", linewidths=0)
            if basin_no_pr.any():
                ax.scatter(lon2d[basin_no_pr], lat2d[basin_no_pr], s=2, color="#9ecae1", linewidths=0)
            if valid_out.any():
                ax.scatter(lon2d[valid_out], lat2d[valid_out], s=3, color="#d95f02", linewidths=0)
            ax.scatter(lon2d[valid_in], lat2d[valid_in], s=2, color="#1b9e77", linewidths=0)
            retained = float(row["valid_basin_fraction"])
            ax.set_title(f"{row['region']}\nvalid retained: {retained:.1%}", fontsize=10)
            ax.set_xlabel("Longitude")
            ax.set_ylabel("Latitude")
            ax.set_xlim(float(row["lon_min"]), float(row["lon_max"]))
            ax.set_ylim(float(row["lat_min"]), float(row["lat_max"]))
            ax.grid(True, color="#eeeeee", linewidth=0.6)
        finally:
            mask_ds.close()

    handles = [
        Line2D([0], [0], marker="o", color="w", label="valid in basin", markerfacecolor="#1b9e77", markersize=6),
        Line2D([0], [0], marker="o", color="w", label="valid outside basin", markerfacecolor="#d95f02", markersize=6),
        Line2D([0], [0], marker="o", color="w", label="basin without CHIRPS valid PR", markerfacecolor="#9ecae1", markersize=6),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False)
    fig.suptitle("Basin/hydroclimate mask audit for priority regions", y=0.98, fontsize=13)
    fig.tight_layout(rect=(0, 0.08, 1, 0.94))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def write_notes(rows: list[dict[str, object]], out_path: Path) -> None:
    lines = [
        "Basin Geometry Diagnostics",
        "==========================",
        "",
        "Masks use official basin/district/ecoregion polygons and CHIRPS grid-cell centers.",
        "They are stricter scientific region definitions than rectangular bounding boxes.",
        "",
    ]
    for row in rows:
        valid_fraction = float(row["valid_basin_fraction"])
        outside_fraction = float(row["valid_outside_basin_fraction"])
        dataset_fraction = float(row["dataset_basin_fraction"])
        lines.extend(
            [
                f"{row['region']} - {row['region_name']}",
                f"  Mask: {row['mask_label']}",
                f"  Source: {row['source_note']}",
                f"  Source URL: {row['source_url']}",
                (
                    "  Valid CHIRPS cells retained: "
                    f"{int(row['valid_in_basin_cells']):,}/{int(row['valid_pr_cells']):,} "
                    f"({valid_fraction:.2%}); outside basin: "
                    f"{int(row['valid_outside_basin_cells']):,} ({outside_fraction:.2%})."
                ),
                (
                    "  Forecast dataset pixels retained: "
                    f"{int(row['dataset_pixels_in_basin']):,}/{int(row['dataset_pixels']):,} "
                    f"({dataset_fraction:.2%})."
                ),
                f"  Interpretation: {interpret_row(row)}",
                "",
            ]
        )
    out_path.write_text("\n".join(lines), encoding="utf-8")


def copy_reports(paths: list[Path]) -> None:
    REPORT_ROOT.mkdir(parents=True, exist_ok=True)
    for path in paths:
        if path.exists():
            shutil.copy2(path, REPORT_ROOT / path.name)
            print(f"Copied report artifact: {REPORT_ROOT / path.name}")


def main() -> None:
    args = parse_args()
    regions = [resolve_region(name) for name in args.regions]
    rows = []
    for region in regions:
        row = audit_region(region, args)
        if row is not None:
            rows.append(row)
    if not rows:
        raise SystemExit("No basin masks were built.")

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    diagnostics_path = OUT_ROOT / "region_basin_mask_diagnostics.csv"
    figure_path = OUT_ROOT / "region_basin_masks.png"
    notes_path = OUT_ROOT / "region_basin_geometry_notes.txt"

    pd.DataFrame(rows).to_csv(diagnostics_path, index=False)
    plot_masks(rows, figure_path)
    write_notes(rows, notes_path)

    print(f"\nWrote diagnostics: {diagnostics_path}")
    print(f"Wrote figure: {figure_path}")
    print(f"Wrote notes: {notes_path}")
    if args.copy_report:
        copy_reports([diagnostics_path, figure_path, notes_path])


if __name__ == "__main__":
    main()
