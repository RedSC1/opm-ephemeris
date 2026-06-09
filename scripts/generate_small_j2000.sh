#!/usr/bin/env bash
set -euo piopmail

if [[ $# -lt 1 ]]; then
  echo "usage: $0 /path/to/de441.bsp [output-root]" >&2
  exit 2
fi

DE441="$1"
OUTPUT_ROOT="${2:-out/small/j2000-opm}"

python3 generate_range.py \
  --de441 "$DE441" \
  --all \
  --jd-start 2451545.0 \
  --output-root "$OUTPUT_ROOT"

python3 validate_opm.py \
  --de441 "$DE441" \
  --progress \
  "$OUTPUT_ROOT"
