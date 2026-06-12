#!/usr/bin/env python3
"""Compute strict-safe OPM coverage limits for a DE-style BSP source.

Fast diagnostic: this mirrors only the generator's current range-safety math.
It does not sample, fit, or generate ephemerides.
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

from jplephem.spk import SPK

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from opm_demo.body_configs import DEFAULT_BODY_ORDER  # noqa: E402
from opm_demo.generator import (  # noqa: E402
    JD_J2000,
    body_config_for_generation,
    source_bounds,
    validate_requested_range,
)


def fixed_limits(source_start: float, source_end: float, segment_days: float, expansion: float) -> tuple[float, float, str]:
    pad = segment_days * expansion
    first_k = math.ceil((source_start + pad - JD_J2000) / segment_days)
    last_k = math.floor((source_end - pad - JD_J2000) / segment_days)
    start = JD_J2000 + first_k * segment_days
    end = JD_J2000 + last_k * segment_days
    note = f"fixed grid: d={segment_days:g}, f={expansion:g}, pad={pad:g}, k={first_k}..{last_k}"
    return start, end, note


def event_margin_limits(source_start: float, source_end: float, period_days: float, edge_margin_days: float, expansion: float) -> tuple[float, float, str]:
    margin = max(edge_margin_days + period_days * expansion, period_days * (1.0 + expansion))
    start = source_start + margin
    end = source_end - margin
    note = f"event margin: period={period_days:.9f}, edge={edge_margin_days:g}, f={expansion:g}, margin={margin:.9f}"
    return start, end, note


def body_limits(body: str, source_start: float, source_end: float) -> tuple[str, float, float, str]:
    cfg = body_config_for_generation(body)
    expansion = float(cfg.segment_domain_expansion_fraction)
    if cfg.method in {"raw_xyz_cheb", "fixed_frame_shape"}:
        if cfg.segment_days is None:
            raise RuntimeError(f"{body}: missing segment_days")
        start, end, note = fixed_limits(source_start, source_end, float(cfg.segment_days), expansion)
    elif cfg.method in {"mean_apsis_frame_shape", "mean_lunar_apsis_frame_shape"}:
        period = float(cfg.clock.period_days or 0.0)
        start, end, note = event_margin_limits(source_start, source_end, period, float(cfg.edge_margin_days), expansion)
    else:
        raise RuntimeError(f"{body}: unsupported method {cfg.method}")
    return body, start, end, f"{cfg.method}; {note}"


def main() -> int:
    parser = argparse.ArgumentParser(description="Compute strict-safe coverage range for OPM generation without generating ephemerides")
    parser.add_argument("--de441", "--spk", dest="spk", type=Path, required=True, help="DE-style BSP/SPK source file, e.g. de441.bsp")
    parser.add_argument("--bodies", default=",".join(DEFAULT_BODY_ORDER), help="comma-separated bodies; default: all")
    parser.add_argument("--shard-years", type=float, default=600.0)
    args = parser.parse_args()

    bodies = [x.strip() for x in args.bodies.split(",") if x.strip()]
    shard_days = float(args.shard_years) * 365.25

    with SPK.open(str(args.spk)) as spk:
        source_start, source_end = source_bounds(spk)
        rows = [body_limits(body, source_start, source_end) for body in bodies]
        safe_start = max(row[1] for row in rows)
        safe_end = min(row[2] for row in rows)
        safe_days = safe_end - safe_start
        full_shards = int(math.floor(safe_days / shard_days))
        tail_days = safe_days - full_shards * shard_days

        print(f"SOURCE_START {source_start:.9f}")
        print(f"SOURCE_END   {source_end:.9f}")
        print()
        print("body        safe_start         safe_end           left_loss    right_loss   rule")
        for body, start, end, note in rows:
            print(f"{body:<10} {start:16.9f} {end:16.9f} {start-source_start:11.3f} {source_end-end:11.3f}  {note}")
        print()
        print(f"ALL_SAFE_START {safe_start:.9f}")
        print(f"ALL_SAFE_END   {safe_end:.9f}")
        print(f"ALL_SAFE_DAYS  {safe_days:.9f}")
        print(f"ALL_SAFE_YEARS {safe_days / 365.25:.6f}")
        print(f"FULL_600Y_SHARDS {full_shards}")
        print(f"TAIL_DAYS        {tail_days:.9f}")
        print(f"TAIL_YEARS       {tail_days / 365.25:.6f}")
        print(f"SUGGESTED_ARGS   --jd-start {safe_start:.9f} --days {safe_days:.9f}")

        validate_requested_range(spk, bodies, safe_start, safe_days, range_safety="strict")
        print("STRICT_CHECK PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
