#!/usr/bin/env python
"""
Download monthly Niño3.4, PDO, and SOI indices and save one unified climate table.

Output:
  data/processed/climate_indices_monthly.csv

Columns:
  time, nino34, pdo, soi, plus nino34_sst when the Niño3.4 source provides
  absolute SST rather than anomalies.

The downloader targets NOAA PSL-style monthly text files:
  - data rows are year + 12 monthly values
  - missing sentinels are commonly -9.9, -99.99, -999, or -9999

This script is the canonical climate-index ingestion step for the project.
Downstream scripts should read data/processed/climate_indices_monthly.csv rather
than downloading climate indices independently.
"""
from __future__ import annotations

from argparse import ArgumentParser, Namespace
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import urlopen
import socket

import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = BASE_DIR / "data" / "raw" / "climate_indices"
OUT_FILE = BASE_DIR / "data" / "processed" / "climate_indices_monthly.csv"

MISSING_SENTINELS = (-9.9, -99.99, -999.0, -9999.0)
NINO34_CLIM_START = 1991
NINO34_CLIM_END = 2020

NINO34_URLS = [
    "https://psl.noaa.gov/data/correlation/nina34.data",
    "https://www.psl.noaa.gov/data/correlation/nina34.data",
]

PDO_URLS = [
    "https://psl.noaa.gov/data/correlation/pdo.data",
    "https://www.psl.noaa.gov/data/correlation/pdo.data",
]

# SOI source locations have moved/varied across NOAA PSL pages.
# Try the correlation-data location first, then older GCOS time-series paths.
SOI_URLS = [
    "https://psl.noaa.gov/data/correlation/soi.data",
    "https://www.psl.noaa.gov/data/correlation/soi.data",
    "https://psl.noaa.gov/gcos_wgsp/Timeseries/Data/soi.data",
    "https://www.psl.noaa.gov/gcos_wgsp/Timeseries/Data/soi.data",
    "https://psl.noaa.gov/gcos_wgsp/Timeseries/Data/soi.long.data",
    "https://www.psl.noaa.gov/gcos_wgsp/Timeseries/Data/soi.long.data",
]


def parse_args() -> Namespace:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--out-file", type=Path, default=OUT_FILE)
    parser.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Download source text files even if cached raw files already exist.",
    )
    parser.add_argument(
        "--keep-full-period",
        action="store_true",
        help="Keep the full downloaded time range. By default the output is clipped to the common available period.",
    )
    return parser.parse_args()


def _fetch_text(urls: list[str]) -> tuple[str, str]:
    last_err = None
    failed: list[str] = []
    for url in urls:
        try:
            with urlopen(url, timeout=60) as response:
                return response.read().decode("utf-8", errors="replace"), url
        except (URLError, HTTPError, TimeoutError, socket.timeout) as exc:
            last_err = exc
            failed.append(f"{url} -> {exc}")
            continue
    raise RuntimeError(
        "Failed to download from all mirrors. Tried:\n  "
        + "\n  ".join(failed)
        + f"\nLast error: {last_err}\n\n"
        "If NOAA has changed the SOI URL again, manually download a PSL-style "
        "SOI monthly file to data/raw/climate_indices/soi.data and rerun this script."
    )


def _load_or_download_text(raw_path: Path, urls: list[str], force_download: bool) -> str:
    if raw_path.exists() and not force_download:
        print(f"Using cached source file: {raw_path}")
        return raw_path.read_text(encoding="utf-8", errors="replace")

    text, source_url = _fetch_text(urls)
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    raw_path.write_text(text, encoding="utf-8")
    print(f"Downloaded {source_url}")
    print(f"Wrote cached source file: {raw_path}")
    return text


def _parse_psl_data(text: str, value_name: str) -> pd.DataFrame:
    rows = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue

        parts = stripped.split()
        if len(parts) != 13:
            continue

        try:
            year = int(float(parts[0]))
            values = [float(value) for value in parts[1:13]]
        except ValueError:
            continue

        if year < 1800 or year > 2200:
            continue

        for month, value in enumerate(values, start=1):
            if any(np.isclose(value, missing) for missing in MISSING_SENTINELS):
                value = np.nan
            rows.append(
                {
                    "time": pd.Timestamp(year=year, month=month, day=1),
                    value_name: value,
                }
            )

    if not rows:
        raise ValueError(f"Could not parse monthly values for {value_name}")

    df = pd.DataFrame(rows).sort_values("time").drop_duplicates("time", keep="last")
    df = df.reset_index(drop=True)

    last_valid_idx = df[value_name].last_valid_index()
    if last_valid_idx is None:
        raise ValueError(f"No valid monthly values parsed for {value_name}")

    return df.loc[:last_valid_idx].reset_index(drop=True)


def _nino34_to_anomaly(df: pd.DataFrame) -> pd.DataFrame:
    """Convert absolute Niño3.4 SST values to monthly anomalies when needed."""
    out = df.copy()
    finite = out["nino34"].dropna()
    if finite.empty:
        raise ValueError("No valid Niño3.4 values available for anomaly conversion")

    # PSL nina34.data is absolute SST (~24-30 C). If a source already contains
    # anomalies, values should be centered near zero and this is left unchanged.
    if finite.abs().median() <= 10:
        print("Niño3.4 appears to already be an anomaly series; leaving unchanged.")
        return out

    baseline = out[
        (out["time"].dt.year >= NINO34_CLIM_START)
        & (out["time"].dt.year <= NINO34_CLIM_END)
    ].copy()
    monthly_clim = baseline.groupby(baseline["time"].dt.month)["nino34"].mean()

    if monthly_clim.isna().any() or len(monthly_clim) != 12:
        raise ValueError(
            "Cannot compute Niño3.4 monthly climatology for "
            f"{NINO34_CLIM_START}-{NINO34_CLIM_END}"
        )

    out["nino34_sst"] = out["nino34"]
    out["nino34"] = out["nino34"] - out["time"].dt.month.map(monthly_clim)
    print(
        "Converted Niño3.4 absolute SST to monthly anomalies using "
        f"{NINO34_CLIM_START}-{NINO34_CLIM_END} climatology."
    )
    return out


def _download_index(raw_dir: Path, name: str, urls: list[str], force_download: bool) -> pd.DataFrame:
    raw_path = raw_dir / f"{name}.data"
    text = _load_or_download_text(raw_path, urls, force_download=force_download)
    return _parse_psl_data(text, name)


def _common_valid_period(df: pd.DataFrame, cols: list[str]) -> tuple[pd.Timestamp, pd.Timestamp]:
    starts = []
    ends = []
    for col in cols:
        valid = df.loc[df[col].notna(), "time"]
        if valid.empty:
            raise ValueError(f"No valid values found for {col}")
        starts.append(valid.min())
        ends.append(valid.max())
    return max(starts), min(ends)


def _print_summary(df: pd.DataFrame, cols: list[str]) -> None:
    print("\nClimate-index summary:")
    for col in cols:
        if col not in df:
            continue
        valid = df.loc[df[col].notna(), "time"]
        if valid.empty:
            print(f"  {col}: no valid values")
            continue
        print(
            f"  {col}: valid {valid.min().date()} through {valid.max().date()} "
            f"({int(df[col].isna().sum())} missing rows retained)"
        )
    print("\nTail:")
    print(df.tail(12).to_string(index=False))


def main() -> None:
    args = parse_args()

    print("Preparing Niño3.4...")
    nino_df = _download_index(args.raw_dir, "nino34", NINO34_URLS, args.force_download)
    nino_df = _nino34_to_anomaly(nino_df)

    print("\nPreparing PDO...")
    pdo_df = _download_index(args.raw_dir, "pdo", PDO_URLS, args.force_download)

    print("\nPreparing SOI...")
    soi_df = _download_index(args.raw_dir, "soi", SOI_URLS, args.force_download)

    nino_cols = ["time", "nino34"] + (["nino34_sst"] if "nino34_sst" in nino_df.columns else [])
    df = (
        nino_df[nino_cols]
        .merge(pdo_df[["time", "pdo"]], on="time", how="outer")
        .merge(soi_df[["time", "soi"]], on="time", how="outer")
        .sort_values("time")
        .reset_index(drop=True)
    )

    df["time"] = pd.to_datetime(df["time"])
    df = df.set_index("time")
    interp_cols = [col for col in ["nino34", "nino34_sst", "pdo", "soi"] if col in df.columns]
    df[interp_cols] = df[interp_cols].interpolate(method="time", limit_area="inside")
    df = df.reset_index()

    if not args.keep_full_period:
        start, end = _common_valid_period(df, ["nino34", "pdo", "soi"])
        df = df[(df["time"] >= start) & (df["time"] <= end)].copy()
        print(f"\nClipped output to common valid period: {start.date()} through {end.date()}")

    args.out_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out_file, index=False)
    print(f"\nWrote: {args.out_file}")
    _print_summary(df, ["nino34", "pdo", "soi"])


if __name__ == "__main__":
    main()
