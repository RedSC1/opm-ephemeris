#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: $0 /path/to/de441.bsp [output-root] [cache-root]" >&2
  exit 2
fi

DE441="$1"
OUTPUT_ROOT="${2:-out/body-packed/full/moon}"
CACHE_ROOT="${3:-out/body-packed/cache/default}"

python3 generate_body_packed.py \
  --body moon \
  --de441 "$DE441" \
  --output-root "$OUTPUT_ROOT" \
  --full-source-safe \
  --reuse-cache \
  --cache-root "$CACHE_ROOT" \
  --chunk-size 1024 \
  --validate
