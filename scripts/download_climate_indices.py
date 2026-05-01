#!/usr/bin/env python
"""
Download monthly Niño3.4 and PDO indices and save a unified climate table.

Output:
  data/processed/climate_indices_monthly.csv
  columns: time, nino34, pdo, plus nino34_sst when the source provides absolute SST

The downloader targets NOAA PSL-style `.data` files:
  - first line: start_year end_year
  - subsequent lines: year + 12 monthly values
  - missing sentinel often: -9.9, -99.99, or -999
"""
from __future__ import annotations

from pathlib import Path
from urllib.request import urlopen
from urllib.error import URLError, HTTPError
import socket
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]
OUT_FILE = BASE_DIR / "data" / "processed" / "climate_indices_monthly.csv"
MISSING_SENTINELS = (-9.9, -99.99, -999.0)
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


def _fetch_text(urls: list[str]) -> str:
    last_err = None
    for u in urls:
        try:
            with urlopen(u, timeout=30) as r:
                return r.read().decode("utf-8", errors="replace")
        except (URLError, HTTPError, TimeoutError, socket.timeout) as e:
            last_err = e
            continue
    raise RuntimeError(f"Failed to download from all mirrors: {urls}. Last error: {last_err}")


def _parse_psl_data(text: str, value_name: str) -> pd.DataFrame:
    rows = []
    for line in text.splitlines():
        s = line.strip()
        if not s:
            continue
        parts = s.split()
        # data rows are: year + 12 monthly values
        if len(parts) != 13:
            continue
        try:
            year = int(float(parts[0]))
            vals = [float(v) for v in parts[1:13]]
        except ValueError:
            continue
        for m, v in enumerate(vals, start=1):
            if any(np.isclose(v, missing) for missing in MISSING_SENTINELS):
                v = np.nan
            rows.append({"time": pd.Timestamp(year=year, month=m, day=1), value_name: v})

    if not rows:
        raise ValueError(f"Could not parse monthly values for {value_name}")
    df = pd.DataFrame(rows).sort_values("time").reset_index(drop=True)

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

    # PSL nina34.data is absolute SST (~24-30 C). If an anomaly source is used
    # later, values should already be centered near zero and this becomes a no-op.
    if finite.abs().median() <= 10:
        print("Niño3.4 appears to already be an anomaly series; leaving unchanged.")
        return out

    base = out[
        (out["time"].dt.year >= NINO34_CLIM_START)
        & (out["time"].dt.year <= NINO34_CLIM_END)
    ].copy()
    monthly_clim = base.groupby(base["time"].dt.month)["nino34"].mean()
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


def main() -> None:
    print("Downloading Niño3.4...")
    nino_txt = _fetch_text(NINO34_URLS)
    nino_df = _parse_psl_data(nino_txt, "nino34")
    nino_df = _nino34_to_anomaly(nino_df)

    print("Downloading PDO...")
    pdo_txt = _fetch_text(PDO_URLS)
    pdo_df = _parse_psl_data(pdo_txt, "pdo")

    nino_cols = ["time", "nino34"] + (
        ["nino34_sst"] if "nino34_sst" in nino_df.columns else []
    )
    df = nino_df[nino_cols].merge(pdo_df, on="time", how="outer").sort_values("time")
    # time interpolation requires a DatetimeIndex (not a plain "time" column)
    df["time"] = pd.to_datetime(df["time"])
    df = df.set_index("time")
    interp_cols = [c for c in ["nino34", "nino34_sst", "pdo"] if c in df.columns]
    df[interp_cols] = df[interp_cols].interpolate(
        method="time", limit_area="inside"
    )
    df = df.reset_index()

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_FILE, index=False)
    print("Wrote:", OUT_FILE)
    for col in ["nino34", "pdo"]:
        valid = df.loc[df[col].notna(), "time"]
        print(
            f"  {col}: valid {valid.min().date()} through {valid.max().date()} "
            f"({int(df[col].isna().sum())} missing rows retained)"
        )
    print(df.tail(12).to_string(index=False))


if __name__ == "__main__":
    main()
