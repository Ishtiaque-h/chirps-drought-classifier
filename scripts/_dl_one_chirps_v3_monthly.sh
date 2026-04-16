#!/usr/bin/env bash
set -euo pipefail

OUTDIR="$1"
YEAR="$2"

URL="https://data.chc.ucsb.edu/products/CHIRPS/v3.0/monthly/global/netcdf/by_year/chirps-v3.0.${YEAR}.monthly.nc"
OUT="${OUTDIR}/chirps-v3.0.${YEAR}.monthly.nc"
TMP="${OUT}.part"

is_valid_netcdf() {
  local f="$1"
  file "$f" | grep -qiE "NetCDF|Hierarchical Data Format|HDF"
}

if [[ -f "$OUT" ]] && is_valid_netcdf "$OUT"; then
  echo "Skip ${YEAR} (already valid)"
  exit 0
fi

echo "Downloading ${YEAR} ..."
curl -sS -fL \
  --retry 5 \
  --retry-delay 2 \
  --connect-timeout 30 \
  --max-time 1800 \
  --continue-at - \
  -o "$TMP" \
  "$URL"

mv "$TMP" "$OUT"

if ! is_valid_netcdf "$OUT"; then
  rm -f "$OUT"
  echo "ERROR: ${OUT} is not a valid NetCDF/HDF file" >&2
  exit 1
fi