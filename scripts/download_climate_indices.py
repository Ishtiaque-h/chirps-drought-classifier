#!/usr/bin/env python
"""
Download monthly Niño3.4 and PDO indices and save a unified climate table.

Output:
  data/processed/climate_indices_monthly.csv
  columns: time, nino34, pdo

The downloader targets NOAA PSL-style `.data` files:
  - first line: start_year end_year
  - subsequent lines: year + 12 monthly values
  - missing sentinel often: -99.99 or -999
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
            if np.isclose(v, -99.99) or np.isclose(v, -999.0):
                v = np.nan
            rows.append({"time": pd.Timestamp(year=year, month=m, day=1), value_name: v})

    if not rows:
        raise ValueError(f"Could not parse monthly values for {value_name}")
    return pd.DataFrame(rows).sort_values("time").reset_index(drop=True)


def main() -> None:
    print("Downloading Niño3.4...")
    nino_txt = _fetch_text(NINO34_URLS)
    nino_df = _parse_psl_data(nino_txt, "nino34")

    print("Downloading PDO...")
    pdo_txt = _fetch_text(PDO_URLS)
    pdo_df = _parse_psl_data(pdo_txt, "pdo")

    df = nino_df.merge(pdo_df, on="time", how="outer").sort_values("time")
    df[["nino34", "pdo"]] = df[["nino34", "pdo"]].interpolate(
        method="time", limit_direction="both"
    )

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_FILE, index=False)
    print("Wrote:", OUT_FILE)
    print(df.tail(12).to_string(index=False))


if __name__ == "__main__":
    main()
