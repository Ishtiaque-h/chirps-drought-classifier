#!/usr/bin/env python
"""Download regional SMAP L4 root-zone soil-moisture percentile subsets.

SMAP L4 SPL4SMGP is a 3-hourly global HDF5 product. Full-granule download is
too large for this project, so this script uses NASA CMR for granule discovery
and NASA Earthdata OPeNDAP to download only the y/x EASE-Grid slices needed for
the configured project regions.

Default snapshots are weekly within each month (days 5, 12, 19, 26 at 01:30
UTC). This is a short-record target validation proxy, not a full 3-hourly
monthly average.
"""

from __future__ import annotations

from argparse import ArgumentParser, Namespace
from calendar import monthrange
from pathlib import Path
import netrc
import os
import re
import shutil
import subprocess
from urllib.parse import quote

import numpy as np
import pandas as pd
import requests
import xarray as xr

from region_config import REGIONS, resolve_region


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "smap_l4_spl4smgp"
CMR_URL = "https://cmr.earthdata.nasa.gov/search/granules.json"
COLLECTION_CONCEPT_ID = "C3480440870-NSIDC_CPRD"
SHORT_NAME = "SPL4SMGP"
VERSION = "008"
DEFAULT_VARIABLE = "sm_rootzone_pctl"
USER_AGENT = "chirps-drought-classifier/smap-l4-short-record-validation"


class DownloadError(RuntimeError):
    pass


def parse_args() -> Namespace:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument(
        "--regions",
        nargs="+",
        default=[
            "cvalley",
            "southern_great_plains",
            "mediterranean_spain",
            "murray_darling",
            "horn_of_africa",
        ],
    )
    parser.add_argument("--start", default="2015-04", help="First target month, YYYY-MM.")
    parser.add_argument("--end", default="2019-12", help="Last target month, YYYY-MM.")
    parser.add_argument("--out-dir", type=Path, default=RAW_DIR)
    parser.add_argument("--variable", default=DEFAULT_VARIABLE)
    parser.add_argument("--snapshot-days", nargs="+", type=int, default=[5, 12, 19, 26])
    parser.add_argument("--snapshot-time", default="013000", help="SMAP center time, HHMMSS.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-downloads", type=int, default=None)
    parser.add_argument(
        "--coordinate-granule-date",
        default="2017-01-15",
        help="Representative date used to download the static SMAP grid coordinates.",
    )
    return parser.parse_args()


def month_starts(start: str, end: str) -> list[pd.Timestamp]:
    months = pd.date_range(
        pd.Timestamp(start).to_period("M").to_timestamp(),
        pd.Timestamp(end).to_period("M").to_timestamp(),
        freq="MS",
    )
    return [pd.Timestamp(month) for month in months]


def has_netrc_credentials() -> bool:
    try:
        auths = netrc.netrc()
    except (FileNotFoundError, netrc.NetrcParseError):
        return False
    try:
        return bool(auths.authenticators("urs.earthdata.nasa.gov"))
    except (FileNotFoundError, netrc.NetrcParseError):
        return False


def looks_like_hdf(path: Path) -> bool:
    if not path.exists() or path.stat().st_size < 1024:
        return False
    with path.open("rb") as handle:
        return handle.read(8).startswith(b"\x89HDF")


def build_session() -> requests.Session:
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})
    return session


def discover_granule(session: requests.Session, date: pd.Timestamp, snapshot_time: str) -> dict[str, str]:
    start = f"{date:%Y-%m-%d}T00:00:00Z"
    end = f"{date:%Y-%m-%d}T23:59:59Z"
    params = {
        "short_name": SHORT_NAME,
        "version": VERSION,
        "temporal": f"{start},{end}",
        "page_size": 20,
    }
    response = session.get(CMR_URL, params=params, timeout=60)
    response.raise_for_status()
    entries = response.json().get("feed", {}).get("entry", [])
    target_token = f"{date:%Y%m%d}T{snapshot_time}"
    for entry in entries:
        granule_id = entry.get("producer_granule_id", "")
        if target_token not in granule_id:
            continue
        opendap = None
        download = None
        for link in entry.get("links", []):
            href = link.get("href", "")
            title = link.get("title", "")
            if "OPeNDAP" in title and href.startswith("https://"):
                opendap = href
            if title.startswith("Download") and href.endswith(".h5"):
                download = href
        if opendap:
            return {
                "granule_id": granule_id,
                "opendap_url": opendap,
                "download_url": download or "",
                "time_start": entry.get("time_start", ""),
                "time_end": entry.get("time_end", ""),
            }
    raise DownloadError(f"No {SHORT_NAME} v{VERSION} granule found for {target_token}.")


def constraint_for_variable(variable: str, y0: int, y1: int, x0: int, x1: int) -> str:
    expression = f"/Geophysical_Data/{variable}[{y0}:{y1}][{x0}:{x1}]"
    return "?" + quote(expression, safe=":,")


def coordinate_constraint() -> str:
    pieces = [
        "x%5B0:3855%5D",
        "y%5B0:1623%5D",
        "cell_lat%5B0:1623%5D%5B0:3855%5D",
        "cell_lon%5B0:1623%5D%5B0:3855%5D",
    ]
    return "?" + ",".join(pieces)


def download_opendap_subset(opendap_url: str, constraint: str, destination: Path, overwrite: bool) -> str:
    if destination.exists() and looks_like_hdf(destination) and not overwrite:
        return "cached"
    if not has_netrc_credentials():
        raise DownloadError("Earthdata credentials are required in ~/.netrc for SMAP OPeNDAP downloads.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp = destination.with_suffix(destination.suffix + ".tmp")
    cookies = destination.parent / ".earthdata_cookies"
    url = opendap_url + ".nc4" + constraint
    cmd = [
        "curl",
        "--globoff",
        "-L",
        "--fail",
        "--silent",
        "--show-error",
        "--netrc-file",
        str(Path.home() / ".netrc"),
        "-c",
        str(cookies),
        "-b",
        str(cookies),
        "--connect-timeout",
        "30",
        "--max-time",
        "300",
        "--retry",
        "3",
        "--retry-delay",
        "2",
        "-o",
        str(tmp),
        url,
    ]
    try:
        subprocess.run(cmd, check=True)
        if not looks_like_hdf(tmp):
            head = tmp.read_bytes()[:256].decode("utf-8", errors="ignore")
            raise DownloadError(f"Downloaded SMAP subset is not HDF5: {destination}; head={head!r}")
        shutil.move(tmp, destination)
        return "downloaded"
    except Exception:
        tmp.unlink(missing_ok=True)
        raise


def coordinate_cache_path(out_dir: Path) -> Path:
    return out_dir / "coordinates" / "smap_l4_spl4smgp_v008_coordinates.nc4"


def relpath(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def ensure_coordinates(args: Namespace, session: requests.Session) -> Path:
    path = coordinate_cache_path(args.out_dir)
    if path.exists() and looks_like_hdf(path) and not args.overwrite:
        return path
    date = pd.Timestamp(args.coordinate_granule_date)
    granule = discover_granule(session, date, args.snapshot_time)
    status = download_opendap_subset(
        granule["opendap_url"],
        coordinate_constraint(),
        path,
        overwrite=args.overwrite,
    )
    print(f"Coordinate cache: {status} -> {path}", flush=True)
    return path


def region_index_boxes(coords_path: Path, regions: list[str]) -> dict[str, dict[str, int]]:
    ds = xr.open_dataset(coords_path, engine="netcdf4")
    try:
        lat = ds["cell_lat"].to_numpy()
        lon = ds["cell_lon"].to_numpy()
        boxes = {}
        for slug in regions:
            region = resolve_region(slug)
            mask = (
                (lat >= region.lat_min)
                & (lat <= region.lat_max)
                & (lon >= region.lon_min)
                & (lon <= region.lon_max)
            )
            yy, xx = np.where(mask)
            if len(yy) == 0:
                raise DownloadError(f"No SMAP grid cells found for region={region.slug}.")
            boxes[region.slug] = {
                "y0": int(yy.min()),
                "y1": int(yy.max()),
                "x0": int(xx.min()),
                "x1": int(xx.max()),
                "n_bbox_cells": int((yy.max() - yy.min() + 1) * (xx.max() - xx.min() + 1)),
                "n_region_cells": int(mask.sum()),
            }
        return boxes
    finally:
        ds.close()


def snapshot_dates(month: pd.Timestamp, days: list[int]) -> list[pd.Timestamp]:
    last_day = monthrange(int(month.year), int(month.month))[1]
    out = []
    for day in days:
        if 1 <= day <= last_day:
            out.append(pd.Timestamp(year=int(month.year), month=int(month.month), day=int(day)))
    return out


def local_subset_path(
    out_dir: Path,
    region: str,
    date: pd.Timestamp,
    snapshot_time: str,
    granule_id: str,
    variable: str,
    box: dict[str, int],
) -> Path:
    safe_granule = re.sub(r"[^A-Za-z0-9_.-]+", "_", granule_id)
    stem = (
        f"{safe_granule}__{region}__{variable}"
        f"_y{box['y0']:04d}-{box['y1']:04d}_x{box['x0']:04d}-{box['x1']:04d}.nc4"
    )
    return out_dir / "subsets" / region / f"{date:%Y}" / stem


def main() -> None:
    args = parse_args()
    args.out_dir = args.out_dir.resolve()
    regions = [resolve_region(region).slug for region in args.regions]
    session = build_session()
    coords = ensure_coordinates(args, session)
    boxes = region_index_boxes(coords, regions)
    months = month_starts(args.start, args.end)

    tasks = []
    granule_cache: dict[tuple[str, str], dict[str, str]] = {}
    for month in months:
        for date in snapshot_dates(month, args.snapshot_days):
            key = (date.strftime("%Y-%m-%d"), args.snapshot_time)
            if key not in granule_cache:
                granule_cache[key] = discover_granule(session, date, args.snapshot_time)
            granule = granule_cache[key]
            for region in regions:
                box = boxes[region]
                path = local_subset_path(
                    args.out_dir,
                    region,
                    date,
                    args.snapshot_time,
                    granule["granule_id"],
                    args.variable,
                    box,
                )
                tasks.append((month, date, region, granule, box, path))

    if args.max_downloads is not None:
        tasks = tasks[: args.max_downloads]

    print(f"SMAP subset tasks: {len(tasks)}")
    print(f"Output directory: {args.out_dir}")
    for region, box in boxes.items():
        print(
            f"{region}: y={box['y0']}:{box['y1']} x={box['x0']}:{box['x1']} "
            f"region_cells={box['n_region_cells']} bbox_cells={box['n_bbox_cells']}",
            flush=True,
        )
    if args.dry_run:
        for month, date, region, granule, box, path in tasks[:12]:
            print(f"{month:%Y-%m} {date:%Y-%m-%d} {region}: {granule['granule_id']} -> {path}")
        if len(tasks) > 12:
            print(f"... {len(tasks) - 12} additional tasks")
        return

    rows = []
    counts: dict[str, int] = {}
    for i, (month, date, region, granule, box, path) in enumerate(tasks, start=1):
        status = download_opendap_subset(
            granule["opendap_url"],
            constraint_for_variable(args.variable, box["y0"], box["y1"], box["x0"], box["x1"]),
            path,
            overwrite=args.overwrite,
        )
        counts[status] = counts.get(status, 0) + 1
        rows.append(
            {
                "target_month": month.strftime("%Y-%m"),
                "snapshot_date": date.strftime("%Y-%m-%d"),
                "snapshot_time": args.snapshot_time,
                "region": region,
                "granule_id": granule["granule_id"],
                "time_start": granule["time_start"],
                "time_end": granule["time_end"],
                "variable": args.variable,
                "y0": box["y0"],
                "y1": box["y1"],
                "x0": box["x0"],
                "x1": box["x1"],
                "n_region_cells": box["n_region_cells"],
                "local_file": relpath(path),
                "status": status,
                "size_bytes": path.stat().st_size if path.exists() else 0,
            }
        )
        print(f"[{i:04d}/{len(tasks):04d}] {month:%Y-%m} {date:%Y-%m-%d} {region}: {status}", flush=True)

    manifest = pd.DataFrame(rows)
    manifest_path = args.out_dir / "smap_l4_regional_subset_manifest.csv"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    if manifest_path.exists() and not args.overwrite:
        old = pd.read_csv(manifest_path)
        manifest = pd.concat([old, manifest], ignore_index=True)
        manifest = manifest.drop_duplicates(
            subset=["target_month", "snapshot_date", "snapshot_time", "region", "variable"],
            keep="last",
        )
    manifest.to_csv(manifest_path, index=False)
    print("Done:", counts)
    print(f"Wrote manifest: {manifest_path}")


if __name__ == "__main__":
    try:
        main()
    except (DownloadError, requests.RequestException) as exc:
        raise SystemExit(str(exc)) from exc
