#!/usr/bin/env bash
set -euo pipefail

OUTDIR="data/raw/chirps_v3/monthly"
mkdir -p "$OUTDIR"

HELPER="scripts/_dl_one_chirps_v3_monthly.sh"

START_YEAR=1991
END_YEAR="$(date +%Y)"

seq "$END_YEAR" -1 "$START_YEAR" | xargs -n1 -P 6 -I{} "$HELPER" "$OUTDIR" {}
echo "All files downloaded into $OUTDIR"