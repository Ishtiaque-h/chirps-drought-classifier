#!/usr/bin/env bash
set -euo pipefail

OUTDIR="data/raw/chirps_v3/monthly"
mkdir -p "$OUTDIR"

HELPER="scripts/_dl_one_chirps_v3_monthly.sh"
chmod +x "$HELPER"

# run 6 downloads in parallel; change -P for more/less
seq 2025 -1 1991 | xargs -n1 -P 6 -I{} "$HELPER" "$OUTDIR" {}
echo "All files downloaded into $OUTDIR"
