#!/usr/bin/env python3
"""Run cross-century random JD accuracy tests for generated PEF files."""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
from jplephem.spk import SPK

import pef_demo.orbit_model as orbit_model
import pef_demo.moon_model as moon_model
from pef_demo.format import body_name_from_id
from pef_demo.validator import read_pef, reconstruct_positions, truth_positions


def iter_pef_files(root: Path) -> list[Path]:
    if root.is_file():
        return [root]
    return sorted(root.rglob("*.pef"))


def choose_samples(pef, samples_per_file: int, rng: np.random.Generator) -> np.ndarray:
    header = pef.header
    start = header.coverage_start_jd
    end = header.coverage_start_jd + header.coverage_span_days
    return rng.uniform(start, end, size=samples_per_file).astype(np.float64)


def summarize(values: list[np.ndarray]) -> tuple[int, float, float, float, float]:
    if not values:
        return 0, float("nan"), float("nan"), float("nan"), float("nan")
    err = np.concatenate(values)
    return (
        int(len(err)),
        float(np.percentile(err, 50)),
        float(np.percentile(err, 95)),
        float(np.percentile(err, 99)),
        float(np.max(err)),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Random JD PEF-vs-DE441 accuracy test across generated files")
    parser.add_argument("--de441", type=Path, required=True, help="path to DE441 BSP file")
    parser.add_argument("--pef-root", type=Path, required=True, help="PEF file, century directory, or full-range root")
    parser.add_argument("--samples", type=int, default=10000, help="total random samples per body")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--no-crc", action="store_true", help="skip CRC64 validation when reading PEF files")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    orbit_model.set_de441_path(args.de441)
    moon_model.set_de441_path(args.de441)
    rng = np.random.default_rng(args.seed)
    files = iter_pef_files(args.pef_root)
    if not files:
        raise SystemExit("no .pef files found")

    by_body: dict[str, list[Path]] = defaultdict(list)
    for path in files:
        pef = read_pef(path, check_crc=not args.no_crc)
        by_body[body_name_from_id(pef.header.body_id)].append(path)

    errors: dict[str, list[np.ndarray]] = defaultdict(list)
    with SPK.open(str(args.de441)) as spk:
        for body, paths in sorted(by_body.items()):
            if body == "sun":
                continue
            per_file = max(1, int(np.ceil(args.samples / len(paths))))
            remaining = args.samples
            for path in sorted(paths):
                if remaining <= 0:
                    break
                pef = read_pef(path, check_crc=not args.no_crc)
                count = min(per_file, remaining)
                jds = choose_samples(pef, count, rng)
                recon = reconstruct_positions(pef, jds)
                truth = truth_positions(spk, pef, jds)
                errors[body].append(orbit_model.angular_errors_arcsec(truth, recon))
                remaining -= count

    print("body samples p50_arcsec p95_arcsec p99_arcsec max_arcsec")
    failures = 0
    for body in sorted(errors):
        count, p50, p95, p99, max_err = summarize(errors[body])
        if max_err > 0.001:
            failures += 1
        print(f"{body} {count} {p50:.9g} {p95:.9g} {p99:.9g} {max_err:.9g}")
    print(f"{('PASS' if failures == 0 else 'FAIL')}: {len(errors) - failures}/{len(errors)} bodies below 0.001 arcsec max")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
