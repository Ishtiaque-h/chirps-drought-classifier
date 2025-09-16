#!/usr/bin/env bash
set -euo pipefail

OUTDIR="$1"
YEAR="$2"

URL="https://data.chc.ucsb.edu/products/CHIRPS/v3.0/monthly/global/netcdf/by_year/chirps-v3.0.${YEAR}.monthly.nc"
OUT="${OUTDIR}/chirps-v3.0.${YEAR}.monthly.nc"

if [[ -f "$OUT" ]]; then
  echo "Skip ${YEAR} (already exists)"
  exit 0
fi

echo "Downloading ${YEAR} ..."
curl -s -S -fL --retry 3 --continue-at - -o "$OUT" "$URL"

# quick sanity check
file "$OUT" | grep -qi "NetCDF" || echo "Warning: $OUT not recognized as NetCDF"

