#!/usr/bin/env python3
"""Benchmark OPM reconstruction throughput against direct DE441 reads."""
from __future__ import annotations

import argparse
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from jplephem.spk import SPK

import opm_demo.orbit_model as orbit_model
import opm_demo.moon_model as moon_model
from opm_demo.format import body_name_from_id
from opm_demo.validator import read_opm, reconstruct_positions, truth_positions


def iter_opm_files(root: Path) -> list[Path]:
    if root.is_file():
        return [root]
    return sorted(root.rglob("*.opm"))


def elapsed_ms(fn) -> tuple[float, object]:
    start = time.perf_counter()
    result = fn()
    return (time.perf_counter() - start) * 1000.0, result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark OPM read and reconstruction speed")
    parser.add_argument("--de441", type=Path, required=True, help="path to DE441 BSP file")
    parser.add_argument("--opm-root", type=Path, required=True, help="OPM file, century directory, or full-range root")
    parser.add_argument("--samples", type=int, default=100000)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--no-crc", action="store_true", help="skip CRC64 validation when reading OPM files")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    orbit_model.set_de441_path(args.de441)
    moon_model.set_de441_path(args.de441)
    rng = np.random.default_rng(args.seed)
    files = iter_opm_files(args.opm_root)
    if not files:
        raise SystemExit("no .opm files found")

    by_body: dict[str, list[Path]] = defaultdict(list)
    for path in files:
        opm = read_opm(path, check_crc=not args.no_crc)
        body = body_name_from_id(opm.header.body_id)
        if body != "sun":
            by_body[body].append(path)

    print("method body samples wall_ms ns_per_position")
    with SPK.open(str(args.de441)) as spk:
        for body, paths in sorted(by_body.items()):
            path = sorted(paths)[len(paths) // 2]
            open_ms, opm = elapsed_ms(lambda: read_opm(path, check_crc=not args.no_crc))
            start = opm.header.coverage_start_jd
            end = opm.header.coverage_start_jd + opm.header.coverage_span_days
            jds = rng.uniform(start, end, size=args.samples).astype(np.float64)
            opm_ms, _ = elapsed_ms(lambda: reconstruct_positions(opm, jds))
            de441_ms, _ = elapsed_ms(lambda: truth_positions(spk, opm, jds))
            print(f"OPM-open {body} 1 {open_ms:.3f} {open_ms * 1_000_000.0:.1f}")
            print(f"OPM-eval {body} {args.samples} {opm_ms:.3f} {opm_ms * 1_000_000.0 / args.samples:.1f}")
            print(f"DE441-eval {body} {args.samples} {de441_ms:.3f} {de441_ms * 1_000_000.0 / args.samples:.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
