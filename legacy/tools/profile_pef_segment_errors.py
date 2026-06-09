#!/usr/bin/env python3
"""Profile per-segment PEF validation errors against DE441.

This helper answers whether the validation max is driven by a small tail of
segments/samples.  It uses the same reconstruction and truth-provider code as
validate_pef.py, but keeps per-segment p50/p95/p99/max summaries and reports the
worst segments.
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
from jplephem.spk import SPK

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pef_demo.moon_model as moon_proto  # noqa: E402
import pef_demo.orbit_model as proto  # noqa: E402
from pef_demo import validator  # noqa: E402

JD_J2000 = 2451545.0
DAYS_PER_JULIAN_YEAR = 365.25
AXIS_COUNT = 3


def percentile_text(values: np.ndarray, percentiles: list[float]) -> str:
    return " ".join(f"p{p:g}={np.percentile(values, p):.9g}" for p in percentiles)


def profile_segments(
    spk: SPK,
    pef: validator.PefFile,
    *,
    nodes_per_segment: int,
    chunk_size: int,
    progress: bool,
) -> dict[str, np.ndarray]:
    h = pef.header
    coeffs = pef.qcoeffs.astype(np.float64) * pef.quant_steps[None, None, :]
    clock = validator.mercury_clock(pef) or validator.moon_clock(pef)
    params = validator.frame_params_for_segments(pef, clock) if pef.frame_coeffs is not None else None
    provider, closeable = validator.truth_position_provider(spk, pef)

    seg_indices_all: list[np.ndarray] = []
    seg_years_all: list[np.ndarray] = []
    seg_p50_all: list[np.ndarray] = []
    seg_p95_all: list[np.ndarray] = []
    seg_p99_all: list[np.ndarray] = []
    seg_max_all: list[np.ndarray] = []
    seg_max_jd_all: list[np.ndarray] = []
    seg_max_tau_all: list[np.ndarray] = []
    seg_max_dist_all: list[np.ndarray] = []
    sample_err_parts: list[np.ndarray] = []

    try:
        for start in range(0, h.segment_count, chunk_size):
            stop = min(start + chunk_size, h.segment_count)
            segment_indices, jds, a, b = validator.segment_chunk_nodes(pef, start, stop, nodes_per_segment, clock)
            if len(segment_indices) == 0:
                continue
            width = b - a
            expanded_a = a - h.expansion * width
            expanded_b = b + h.expansion * width
            tau = (2.0 * jds - expanded_a[:, None] - expanded_b[:, None]) / (expanded_b - expanded_a)[:, None]
            recon = validator.reconstruct_segment_nodes(pef, segment_indices, tau, coeffs, params)
            truth = provider.position(jds.reshape(-1)).reshape(recon.shape)
            err = proto.angular_errors_arcsec(truth.reshape((-1, AXIS_COUNT)), recon.reshape((-1, AXIS_COUNT))).reshape(
                (len(segment_indices), nodes_per_segment)
            )
            max_node = np.argmax(err, axis=1)
            row = np.arange(len(segment_indices))
            max_truth = truth[row, max_node]

            seg_indices_all.append(segment_indices)
            seg_years_all.append((0.5 * (a + b) - JD_J2000) / DAYS_PER_JULIAN_YEAR)
            seg_p50_all.append(np.percentile(err, 50, axis=1))
            seg_p95_all.append(np.percentile(err, 95, axis=1))
            seg_p99_all.append(np.percentile(err, 99, axis=1))
            seg_max_all.append(np.max(err, axis=1))
            seg_max_jd_all.append(jds[row, max_node])
            seg_max_tau_all.append(tau[row, max_node])
            seg_max_dist_all.append(np.linalg.norm(max_truth, axis=1))
            sample_err_parts.append(err.reshape(-1))

            if progress:
                print(f"  profiled {stop}/{h.segment_count} segments", flush=True)
    finally:
        validator.close_if_needed(closeable)

    return {
        "segment_index": np.concatenate(seg_indices_all),
        "year": np.concatenate(seg_years_all),
        "p50": np.concatenate(seg_p50_all),
        "p95": np.concatenate(seg_p95_all),
        "p99": np.concatenate(seg_p99_all),
        "max": np.concatenate(seg_max_all),
        "max_jd": np.concatenate(seg_max_jd_all),
        "max_tau": np.concatenate(seg_max_tau_all),
        "max_dist_km": np.concatenate(seg_max_dist_all),
        "sample_errors": np.concatenate(sample_err_parts),
    }


def print_tail_counts(seg_max: np.ndarray, sample_errors: np.ndarray, thresholds: list[float]) -> None:
    print("tail counts")
    for threshold in thresholds:
        seg_count = int(np.count_nonzero(seg_max >= threshold))
        sample_count = int(np.count_nonzero(sample_errors >= threshold))
        print(
            f"  >= {threshold:.9g} arcsec: "
            f"segments {seg_count:,}/{len(seg_max):,} ({seg_count / len(seg_max) * 100:.4f}%), "
            f"samples {sample_count:,}/{len(sample_errors):,} ({sample_count / len(sample_errors) * 100:.4f}%)"
        )
    print()


def print_worst_segments(result: dict[str, np.ndarray], top: int) -> None:
    order = np.argsort(result["max"])[::-1]
    print(f"top {min(top, len(order))} worst segments")
    print("rank seg year max_arcsec p99_arcsec p95_arcsec p50_arcsec max_tau max_jd max_dist_km")
    for rank, idx in enumerate(order[:top], 1):
        print(
            f"{rank:4d} "
            f"{int(result['segment_index'][idx]):7d} "
            f"{result['year'][idx]:+10.1f} "
            f"{result['max'][idx]:.9g} "
            f"{result['p99'][idx]:.9g} "
            f"{result['p95'][idx]:.9g} "
            f"{result['p50'][idx]:.9g} "
            f"{result['max_tau'][idx]:+8.4f} "
            f"{result['max_jd'][idx]:.6f} "
            f"{result['max_dist_km'][idx]:.3f}"
        )
    print()


def print_year_bins(result: dict[str, np.ndarray], bins: int) -> None:
    years = result["year"]
    seg_max = result["max"]
    edges = np.linspace(float(np.min(years)), float(np.max(years)), bins + 1)
    print(f"year-bin segment max summary ({bins} bins)")
    print("bin year_start year_end count max p99 p95 median")
    for i in range(bins):
        mask = (years >= edges[i]) & (years < edges[i + 1] if i + 1 < len(edges) - 1 else years <= edges[i + 1])
        if not np.any(mask):
            continue
        values = seg_max[mask]
        print(
            f"{i:3d} {edges[i]:+10.1f} {edges[i + 1]:+10.1f} {len(values):7d} "
            f"{np.max(values):.9g} {np.percentile(values, 99):.9g} {np.percentile(values, 95):.9g} {np.median(values):.9g}"
        )
    print()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pef", type=Path, help="PEF file to profile")
    parser.add_argument("--de441", type=Path, required=True, help="path to DE441 BSP file")
    parser.add_argument("--nodes-per-segment", type=int, default=32)
    parser.add_argument("--chunk-size", type=int, default=4096)
    parser.add_argument("--top", type=int, default=50)
    parser.add_argument("--year-bins", type=int, default=24)
    parser.add_argument("--thresholds", default="0.0005,0.0006,0.0007,0.0008,0.0009,0.001", help="comma-separated arcsec thresholds")
    parser.add_argument("--progress", action="store_true")
    parser.add_argument("--no-crc", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    proto.set_de441_path(args.de441)
    moon_proto.set_de441_path(args.de441)
    pef = validator.read_pef(args.pef, check_crc=not args.no_crc)
    with SPK.open(str(args.de441)) as spk:
        result = profile_segments(spk, pef, nodes_per_segment=args.nodes_per_segment, chunk_size=args.chunk_size, progress=args.progress)

    sample_errors = result["sample_errors"]
    seg_max = result["max"]
    print(f"file: {args.pef}")
    print(f"segments: {len(seg_max):,}  nodes/segment: {args.nodes_per_segment}  samples: {len(sample_errors):,}")
    print("sample error percentiles:", percentile_text(sample_errors, [50, 90, 95, 99, 99.5, 99.9, 99.99]), f"max={np.max(sample_errors):.9g}")
    print("segment max percentiles:", percentile_text(seg_max, [50, 90, 95, 99, 99.5, 99.9, 99.99]), f"max={np.max(seg_max):.9g}")
    print()
    thresholds = [float(x) for x in args.thresholds.split(",") if x.strip()]
    print_tail_counts(seg_max, sample_errors, thresholds)
    print_worst_segments(result, args.top)
    print_year_bins(result, args.year_bins)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
