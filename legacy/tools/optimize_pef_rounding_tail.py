#!/usr/bin/env python3
"""Optimize quantized rounding for the current high-error segment tail.

This is a diagnostic driver around optimize_pef_segment_rounding.py.  It first
profiles a PEF file, selects segments whose current validation-node max exceeds a
threshold, applies local +/-1 coefficient rounding search to those segments, and
reports the before/after global error distribution without writing a new PEF.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from jplephem.spk import SPK

REPO_ROOT = Path(__file__).resolve().parents[1]
TOOLS_DIR = REPO_ROOT / "tools"
for path in (REPO_ROOT, TOOLS_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import pef_demo.moon_model as moon_proto  # noqa: E402
import pef_demo.orbit_model as proto  # noqa: E402
from pef_demo import validator  # noqa: E402
import optimize_pef_segment_rounding as opt_round  # noqa: E402

JD_J2000 = 2451545.0
DAYS_PER_JULIAN_YEAR = 365.25
AXIS_COUNT = 3


def percentile_text(values: np.ndarray) -> str:
    return (
        f"p50={np.percentile(values, 50):.9g} "
        f"p90={np.percentile(values, 90):.9g} "
        f"p95={np.percentile(values, 95):.9g} "
        f"p99={np.percentile(values, 99):.9g} "
        f"p99.5={np.percentile(values, 99.5):.9g} "
        f"p99.9={np.percentile(values, 99.9):.9g} "
        f"max={np.max(values):.9g}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pef", type=Path)
    parser.add_argument("--de441", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=0.0007)
    parser.add_argument("--nodes-per-segment", type=int, default=32)
    parser.add_argument("--chunk-size", type=int, default=4096)
    parser.add_argument("--max-passes", type=int, default=4)
    parser.add_argument("--radius", type=int, default=1)
    parser.add_argument("--limit", type=int, default=0, help="optimize only the worst N selected segments; 0 means all selected")
    parser.add_argument("--objective", choices=["max", "p99max", "guarded"], default="max", help="rounding objective for selected segments")
    parser.add_argument("--progress", action="store_true")
    parser.add_argument("--no-crc", action="store_true")
    args = parser.parse_args()

    proto.set_de441_path(args.de441)
    moon_proto.set_de441_path(args.de441)
    pef = validator.read_pef(args.pef, check_crc=not args.no_crc)
    clock = validator.mercury_clock(pef) or validator.moon_clock(pef)
    params = validator.frame_params_for_segments(pef, clock) if pef.frame_coeffs is not None else None
    if params is None:
        raise SystemExit("this diagnostic expects frame parameters")

    all_initial_parts: list[np.ndarray] = []
    selected: list[tuple[float, int, np.ndarray, np.ndarray]] = []

    with SPK.open(str(args.de441)) as spk:
        provider, closeable = validator.truth_position_provider(spk, pef)
        try:
            for start in range(0, pef.header.segment_count, args.chunk_size):
                stop = min(start + args.chunk_size, pef.header.segment_count)
                segment_indices, jds, a, b = validator.segment_chunk_nodes(pef, start, stop, args.nodes_per_segment, clock)
                if len(segment_indices) == 0:
                    continue
                width = b - a
                expanded_a = a - pef.header.expansion * width
                expanded_b = b + pef.header.expansion * width
                tau = (2.0 * jds - expanded_a[:, None] - expanded_b[:, None]) / (expanded_b - expanded_a)[:, None]
                coeffs = pef.qcoeffs.astype(np.float64) * pef.quant_steps[None, None, :]
                recon = validator.reconstruct_segment_nodes(pef, segment_indices, tau, coeffs, params)
                truth = provider.position(jds.reshape(-1)).reshape(recon.shape)
                err = proto.angular_errors_arcsec(truth.reshape((-1, AXIS_COUNT)), recon.reshape((-1, AXIS_COUNT))).reshape(
                    (len(segment_indices), args.nodes_per_segment)
                )
                all_initial_parts.append(err.reshape(-1))
                max_by_segment = np.max(err, axis=1)
                for local_i, seg in enumerate(segment_indices):
                    if max_by_segment[local_i] >= args.threshold:
                        selected.append((float(max_by_segment[local_i]), int(seg), jds[local_i].copy(), truth[local_i].copy()))
                if args.progress:
                    print(f"  scanned {stop}/{pef.header.segment_count}; selected={len(selected)}", flush=True)

            selected.sort(reverse=True, key=lambda item: item[0])
            if args.limit > 0:
                selected = selected[: args.limit]

            optimized_by_segment: dict[int, np.ndarray] = {}
            optimized_summaries: list[tuple[int, float, float, int]] = []
            for idx, (current_max, seg, jds_one, truth_one) in enumerate(selected, 1):
                result = opt_round.optimize_segment(
                    pef,
                    params,
                    seg,
                    jds_one,
                    truth_one,
                    max_passes=args.max_passes,
                    radius=args.radius,
                    objective_mode=args.objective,
                    min_improvement=1e-12,
                )
                best_err = result["best_err"]
                optimized_by_segment[seg] = best_err  # type: ignore[assignment]
                optimized_summaries.append((seg, current_max, float(np.max(best_err)), len(result["changes"])))
                if args.progress and (idx % 50 == 0 or idx == len(selected)):
                    print(f"  optimized {idx}/{len(selected)} selected segments", flush=True)
        finally:
            validator.close_if_needed(closeable)

    initial_errors = np.concatenate(all_initial_parts)

    # Rebuild global sample vector by replacing only selected segment errors.  This
    # preserves all non-tail sample errors exactly and avoids a second full truth pass.
    rebuilt_parts: list[np.ndarray] = []
    selected_set = set(optimized_by_segment)
    cursor_selected: dict[int, np.ndarray] = optimized_by_segment
    with SPK.open(str(args.de441)) as spk:
        provider, closeable = validator.truth_position_provider(spk, pef)
        try:
            for start in range(0, pef.header.segment_count, args.chunk_size):
                stop = min(start + args.chunk_size, pef.header.segment_count)
                segment_indices, jds, a, b = validator.segment_chunk_nodes(pef, start, stop, args.nodes_per_segment, clock)
                if len(segment_indices) == 0:
                    continue
                width = b - a
                expanded_a = a - pef.header.expansion * width
                expanded_b = b + pef.header.expansion * width
                tau = (2.0 * jds - expanded_a[:, None] - expanded_b[:, None]) / (expanded_b - expanded_a)[:, None]
                coeffs = pef.qcoeffs.astype(np.float64) * pef.quant_steps[None, None, :]
                recon = validator.reconstruct_segment_nodes(pef, segment_indices, tau, coeffs, params)
                truth = provider.position(jds.reshape(-1)).reshape(recon.shape)
                err = proto.angular_errors_arcsec(truth.reshape((-1, AXIS_COUNT)), recon.reshape((-1, AXIS_COUNT))).reshape(
                    (len(segment_indices), args.nodes_per_segment)
                )
                for local_i, seg_np in enumerate(segment_indices):
                    seg = int(seg_np)
                    if seg in selected_set:
                        rebuilt_parts.append(cursor_selected[seg])
                    else:
                        rebuilt_parts.append(err[local_i])
        finally:
            validator.close_if_needed(closeable)
    optimized_errors = np.concatenate(rebuilt_parts)

    print(f"file: {args.pef}")
    print(f"threshold: {args.threshold:.9g}; selected segments optimized: {len(selected):,}")
    print("initial global:  ", percentile_text(initial_errors))
    print("optimized global:", percentile_text(optimized_errors))
    print()
    if optimized_summaries:
        before = np.asarray([x[1] for x in optimized_summaries])
        after = np.asarray([x[2] for x in optimized_summaries])
        print("selected-tail max summary")
        print("  before:", percentile_text(before))
        print("  after: ", percentile_text(after))
        print()
        print("top optimized selected segments")
        print("seg before_max after_max delta changes")
        for seg, before_max, after_max, changes in optimized_summaries[:30]:
            print(f"{seg:7d} {before_max:.9g} {after_max:.9g} {after_max-before_max:+.9g} {changes:7d}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
