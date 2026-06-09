#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: $0 /path/to/de441.bsp [output-root] [cache-root]" >&2
  exit 2
fi

DE441="$1"
OUTPUT_ROOT="${2:-out/body-packed/seven-century}"
CACHE_ROOT="${3:-out/body-packed/cache/seven-century}"

python3 generate_body_packed.py \
  --all \
  --de441 "$DE441" \
  --output-root "$OUTPUT_ROOT" \
  --start-index -3 \
  --end-index 3 \
  --reuse-cache \
  --cache-root "$CACHE_ROOT" \
  --chunk-size 1024 \
  --validate
