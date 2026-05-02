#!/usr/bin/env python
"""Build leakage-safe seasonal datasets for SPI-k targets at configurable lead times."""
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

PROCESSED = Path("data/processed")
PR_FILE = PROCESSED / "chirps_v3_monthly_cvalley_1991_2026.nc"
SPI_FILE = PROCESSED / "chirps_v3_monthly_cvalley_spi_1991_2026.nc"
CLIMATE_FILE = PROCESSED / "climate_indices_monthly.csv"
MISSING_SENTINELS = (-9.9, -99.99, -999.0)


def parse_args():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument(
        "--leads",
        nargs="+",
        type=int,
        default=[3, 6],
        help="Lead times (months ahead) to generate targets for.",
    )
    parser.add_argument(
        "--spi-indices",
        nargs="+",
        type=int,
        default=[3, 6],
        help="Target SPI windows (e.g., 3 6).",
    )
    parser.add_argument(
        "--climate-features",
        choices=["all", "nino34", "pdo", "none"],
        default="all",
        help="Optional climate features to merge from climate_indices_monthly.csv.",
    )
    parser.add_argument(
        "--allow-overlap",
        action="store_true",
        help=(
            "Allow overlapping accumulation windows. By default, lead < spi_index "
            "is skipped to avoid leakage-like overlap in SPI windows."
        ),
    )
    return parser.parse_args()


def _climate_lags(all_times: pd.DatetimeIndex, climate_features: str) -> pd.DataFrame:
    if climate_features == "none" or not CLIMATE_FILE.exists():
        return pd.DataFrame({"time": all_times})

    selected = []
    if climate_features in {"all", "nino34"}:
        selected.append("nino34")
    if climate_features in {"all", "pdo"}:
        selected.append("pdo")

    cdf = pd.read_csv(CLIMATE_FILE)
    required = {"time", *selected}
    missing = required.difference(cdf.columns)
    if missing:
        raise ValueError(f"{CLIMATE_FILE} missing required columns: {sorted(missing)}")

    cdf["time"] = pd.to_datetime(cdf["time"]).dt.to_period("M").dt.to_timestamp()
    cdf = (
        cdf[["time"] + selected]
        .sort_values("time")
        .drop_duplicates(subset=["time"], keep="last")
        .set_index("time")
    )
    cdf[selected] = cdf[selected].replace(list(MISSING_SENTINELS), np.nan)
    cdf = cdf.reindex(all_times).sort_index()
    cdf[selected] = cdf[selected].interpolate(method="time", limit_area="inside")

    out_cols = []
    if "nino34" in selected:
        cdf["nino34_lag1"] = cdf["nino34"]
        cdf["nino34_lag2"] = cdf["nino34"].shift(1)
        out_cols.extend(["nino34_lag1", "nino34_lag2"])
    if "pdo" in selected:
        cdf["pdo_lag1"] = cdf["pdo"]
        cdf["pdo_lag2"] = cdf["pdo"].shift(1)
        out_cols.extend(["pdo_lag1", "pdo_lag2"])

    return cdf[out_cols].reset_index().rename(columns={"index": "time"})


def _build_one(pr, spi1, spi3, spi6, spi_target, lead: int, spi_idx: int, lat_name: str, lon_name: str, climate_features: str) -> pd.DataFrame:
    target_value = spi_target.shift(time=-lead)
    target_label = xr.where(
        target_value <= -1.0,
        -1,
        xr.where(target_value >= 1.0, 1, 0),
    ).where(target_value.notnull())
    target_label.name = f"target_label_spi{spi_idx}"
    target_value.name = f"target_spi{spi_idx}"

    ds = xr.Dataset(
        {
            "spi1_lag1": spi1,
            "spi1_lag2": spi1.shift(time=1),
            "spi1_lag3": spi1.shift(time=2),
            "spi3_lag1": spi3,
            "spi6_lag1": spi6,
            "pr_lag1": pr,
            "pr_lag2": pr.shift(time=1),
            "pr_lag3": pr.shift(time=2),
            target_value.name: target_value,
            target_label.name: target_label,
        }
    ).stack(pixel=(lat_name, lon_name))

    df = ds.reset_index("pixel").to_dataframe()
    if "time" not in df.columns:
        df = df.reset_index()
    df["time"] = pd.to_datetime(df["time"]).dt.to_period("M").dt.to_timestamp()
    df = df.rename(columns={lat_name: "latitude", lon_name: "longitude"})

    all_times = pd.DatetimeIndex(sorted(df["time"].unique()))
    climate_df = _climate_lags(all_times, climate_features)
    exog_cols = [c for c in climate_df.columns if c != "time"]
    if exog_cols:
        df = df.merge(climate_df, on="time", how="left")

    feat_cols = [
        "spi1_lag1", "spi1_lag2", "spi1_lag3",
        "spi3_lag1", "spi6_lag1",
        "pr_lag1", "pr_lag2", "pr_lag3",
    ]
    df = df.dropna(subset=feat_cols + exog_cols + [target_label.name, target_value.name]).copy()

    target_time = pd.to_datetime(df["time"]) + pd.DateOffset(months=lead)
    df["target_time"] = target_time.dt.to_period("M").dt.to_timestamp()
    df["month"] = df["target_time"].dt.month
    df["year"] = df["target_time"].dt.year
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12.0)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12.0)
    df[target_label.name] = df[target_label.name].astype(np.int8)
    df["lead"] = lead

    cols = (
        [
            "time", "target_time", "lead", "year", "month", "month_sin", "month_cos",
            "latitude", "longitude",
        ]
        + feat_cols
        + exog_cols
        + [target_value.name, target_label.name]
    )
    return df[cols]


def main():
    args = parse_args()
    print("Building seasonal forecasting datasets...")
    print(f"  SPI indices: {args.spi_indices}")
    print(f"  Lead times: {args.leads}")

    pr_ds = xr.open_dataset(PR_FILE).load()
    spi_ds = xr.open_dataset(SPI_FILE).load()

    lat_name = "latitude" if "latitude" in pr_ds.coords else "lat"
    lon_name = "longitude" if "longitude" in pr_ds.coords else "lon"

    pr = pr_ds["pr"].sel(time=spi_ds.time)
    spi1 = spi_ds["spi1"].sel(time=pr.time)
    spi3 = spi_ds["spi3"].sel(time=pr.time)
    spi6 = spi_ds["spi6"].sel(time=pr.time)

    spi_map = {3: spi3, 6: spi6}
    built = []
    for spi_idx in args.spi_indices:
        if spi_idx not in spi_map:
            print(f"Skipping unsupported SPI index: {spi_idx}")
            continue
        for lead in args.leads:
            if (not args.allow_overlap) and lead < spi_idx:
                print(f"Skipping SPI-{spi_idx} lead-{lead}: overlap blocked (use --allow-overlap to force).")
                continue

            print(f"\n--- Building SPI-{spi_idx} lead-{lead} ---")
            df = _build_one(
                pr=pr,
                spi1=spi1,
                spi3=spi3,
                spi6=spi6,
                spi_target=spi_map[spi_idx],
                lead=lead,
                spi_idx=spi_idx,
                lat_name=lat_name,
                lon_name=lon_name,
                climate_features=args.climate_features,
            )

            out_file = PROCESSED / f"dataset_seasonal_spi{spi_idx}_lead{lead}.parquet"
            out_sample = PROCESSED / f"dataset_seasonal_spi{spi_idx}_lead{lead}.sample.csv"
            df.to_parquet(out_file, index=False)
            df.head(10000).to_csv(out_sample, index=False)

            label_col = f"target_label_spi{spi_idx}"
            print(f"Wrote: {out_file} (rows={len(df):,}, cols={df.shape[1]})")
            print("Class distribution:")
            print(df[label_col].value_counts().sort_index())
            built.append((spi_idx, lead, len(df)))

    print("\nDone.")
    for spi_idx, lead, nrows in built:
        print(f"  SPI-{spi_idx} lead-{lead}: {nrows:,} rows")


if __name__ == "__main__":
    main()
