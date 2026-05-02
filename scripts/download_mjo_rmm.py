#!/usr/bin/env python
"""
Download the BoM daily RMM MJO index and build monthly features.

Output:
  data/processed/mjo_rmm_monthly.csv

Columns:
  time, mjo_rmm1_mean, mjo_rmm2_mean, mjo_amp_mean,
  mjo_phase_angle, mjo_phase_sin, mjo_phase_cos,
  mjo_active_days, mjo_active_frac

Notes:
  - Uses the BoM "rmm.74toRealtime.txt" daily RMM index.
  - Monthly phase is derived from the vector-mean RMM1/RMM2.
  - Active days are those with daily amplitude >= 1.0.
"""
from __future__ import annotations

from argparse import ArgumentParser, Namespace
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
import socket

import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = BASE_DIR / "data" / "raw" / "mjo"
OUT_FILE = BASE_DIR / "data" / "processed" / "mjo_rmm_monthly.csv"

RMM_URLS = [
    "https://www.bom.gov.au/climate/mjo/graphics/rmm.74toRealtime.txt",
    "http://www.bom.gov.au/climate/mjo/graphics/rmm.74toRealtime.txt",
]


def parse_args() -> Namespace:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--out-file", type=Path, default=OUT_FILE)
    parser.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Download the daily RMM file even if cached locally.",
    )
    return parser.parse_args()


def _fetch_text(urls: list[str]) -> tuple[str, str]:
    last_err = None
    failed: list[str] = []
    for url in urls:
        try:
            req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urlopen(req, timeout=60) as response:
                return response.read().decode("utf-8", errors="replace"), url
        except (URLError, HTTPError, TimeoutError, socket.timeout) as exc:
            last_err = exc
            failed.append(f"{url} -> {exc}")
            continue
    raise RuntimeError(
        "Failed to download from all mirrors. Tried:\n  "
        + "\n  ".join(failed)
        + f"\nLast error: {last_err}"
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


def _parse_rmm(text: str) -> pd.DataFrame:
    rows = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.lower().startswith("year"):
            continue
        if stripped.startswith("#"):
            continue

        parts = stripped.split()
        if len(parts) < 5:
            continue

        try:
            year = int(parts[0])
            month = int(parts[1])
            day = int(parts[2])
            rmm1 = float(parts[3])
            rmm2 = float(parts[4])
        except ValueError:
            continue

        if year < 1800 or year > 2200:
            continue

        rows.append(
            {
                "time": pd.Timestamp(year=year, month=month, day=day),
                "rmm1": rmm1,
                "rmm2": rmm2,
            }
        )

    if not rows:
        raise ValueError("Could not parse any daily RMM entries")

    df = pd.DataFrame(rows).sort_values("time").drop_duplicates("time", keep="last")
    df = df.reset_index(drop=True)
    return df


def _phase_angle(rmm1: np.ndarray, rmm2: np.ndarray) -> np.ndarray:
    return np.arctan2(rmm2, rmm1)


def _phase_number(angle: np.ndarray) -> np.ndarray:
    # Convert radians (-pi, pi] to 1..8 phase index.
    # Phase 1 centered at angle 0 (positive RMM1), then CCW.
    phase = (np.floor(((angle + np.pi) / (2 * np.pi)) * 8) + 1).astype(int)
    phase = np.where(phase == 9, 8, phase)
    return phase


def build_monthly_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["amp"] = np.sqrt(df["rmm1"] ** 2 + df["rmm2"] ** 2)
    df["month"] = df["time"].dt.to_period("M").dt.to_timestamp()

    def agg_month(group: pd.DataFrame) -> pd.Series:
        rmm1_mean = float(group["rmm1"].mean())
        rmm2_mean = float(group["rmm2"].mean())
        amp_mean = float(np.sqrt(rmm1_mean ** 2 + rmm2_mean ** 2))
        angle = float(_phase_angle(np.array([rmm1_mean]), np.array([rmm2_mean]))[0])
        phase = int(_phase_number(np.array([angle]))[0])
        active_days = int((group["amp"] >= 1.0).sum())
        return pd.Series(
            {
                "mjo_rmm1_mean": rmm1_mean,
                "mjo_rmm2_mean": rmm2_mean,
                "mjo_amp_mean": amp_mean,
                "mjo_phase_angle": angle,
                "mjo_phase_sin": float(np.sin(angle)),
                "mjo_phase_cos": float(np.cos(angle)),
                "mjo_phase": phase,
                "mjo_active_days": active_days,
                "mjo_active_frac": float(active_days) / float(len(group)),
            }
        )

    monthly = df.groupby("month", as_index=False).apply(agg_month)
    monthly = monthly.rename(columns={"month": "time"})
    return monthly.sort_values("time").reset_index(drop=True)


def main() -> None:
    args = parse_args()
    raw_path = args.raw_dir / "rmm.74toRealtime.txt"

    print("Downloading daily RMM index...")
    text = _load_or_download_text(raw_path, RMM_URLS, args.force_download)

    print("Parsing daily RMM values...")
    daily = _parse_rmm(text)

    print("Building monthly MJO features...")
    monthly = build_monthly_features(daily)

    args.out_file.parent.mkdir(parents=True, exist_ok=True)
    monthly.to_csv(args.out_file, index=False)
    print(f"Wrote: {args.out_file} rows={len(monthly):,}")
    print("Tail:")
    print(monthly.tail(6).to_string(index=False))


if __name__ == "__main__":
    main()
