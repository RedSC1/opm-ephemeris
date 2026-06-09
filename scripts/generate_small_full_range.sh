#!/usr/bin/env bash
set -euo piopmail

if [[ $# -lt 1 ]]; then
  echo "usage: $0 /path/to/de441.bsp [output-root]" >&2
  exit 2
fi

DE441="$1"
OUTPUT_ROOT="${2:-out/small/full-opm}"

python3 generate_full.py \
  --de441 "$DE441" \
  --output-root "$OUTPUT_ROOT" \
  --jobs 10 \
  --resume \
  --validate
