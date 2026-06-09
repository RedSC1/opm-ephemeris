#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: $0 /path/to/de441.bsp [pef-root]" >&2
  exit 2
fi

DE441="$1"
PEF_ROOT="${2:-out/body-packed/full}"

python3 validate_pef.py \
  --de441 "$DE441" \
  --progress \
  "$PEF_ROOT"
