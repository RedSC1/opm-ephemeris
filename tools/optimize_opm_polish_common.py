#!/usr/bin/env python3
"""Test p99-guarded rounding after relaxed residual quantization.

This diagnostic keeps the existing OPM format/model table, reloads the unquantized
residual coefficient cache, requantizes it at one or more candidate quant bases,
then runs local +/- coefficient rounding search only on selected high-tail
segments.  It reports estimated body-packed size and before/after validation tail
statistics without writing a new OPM.
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
from jplephem.spk import SPK

REPO_ROOT = Path(__file__).resolve().parents[1]
TOOLS_DIR = REPO_ROOT / "tools"
for path in (REPO_ROOT, TOOLS_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import opm_demo.moon_model as moon_proto  # noqa: E402
import opm_demo.orbit_model as proto  # noqa: E402
from opm_demo import validator  # noqa: E402
from opm_demo.packing import degree_quant_steps  # noqa: E402
import optimize_opm_segment_rounding as opt_round  # noqa: E402

AXIS_COUNT = 3


def parse_csv_floats(text: str) -> list[float]:
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def zigzag_widths(qcoeffs: np.ndarray) -> np.ndarray:
    widths = np.zeros((qcoeffs.shape[1], qcoeffs.shape[2]), dtype=np.uint8)
    for axis in range(qcoeffs.shape[1]):
        for degree in range(qcoeffs.shape[2]):
            values = qcoeffs[:, axis, degree]
            encoded = np.where(values >= 0, values * 2, -values * 2 - 1).astype(np.uint64)
            widths[axis, degree] = max(1, int(np.max(encoded)).bit_length())
    return widths


def payload_size_for_widths(segment_count: int, widths: np.ndarray) -> int:
    axis_bits = widths.astype(np.int64).sum(axis=1)
    return int(sum((segment_count * int(bits) + 7) // 8 for bits in axis_bits))


def percentile_text(values: np.ndarray) -> str:
    return (
        f"p50={np.percentile(values, 50):.9g} "
        f"p95={np.percentile(values, 95):.9g} "
        f"p99={np.percentile(values, 99):.9g} "
        f"p99.5={np.percentile(values, 99.5):.9g} "
        f"p99.9={np.percentile(values, 99.9):.9g} "
        f"max={np.max(values):.9g}"
    )


def validate_candidate(
    spk: SPK,
    opm: validator.OpmFile,
    coeffs: np.ndarray,
    params: np.ndarray | None,
    clock: object | None,
    *,
    nodes_per_segment: int,
    chunk_size: int,
    select_threshold: float,
    progress: bool,
    error_metric: str = "angular",
) -> tuple[np.ndarray, list[tuple[float, int, np.ndarray, np.ndarray]]]:
    h = opm.header
    errors_by_segment = np.full((h.segment_count, nodes_per_segment), np.nan, dtype=np.float32)
    selected: list[tuple[float, int, np.ndarray, np.ndarray]] = []
    provider, closeable = validator.truth_position_provider(spk, opm)
    try:
        for start in range(0, h.segment_count, chunk_size):
            stop = min(start + chunk_size, h.segment_count)
            segment_indices, jds, a, b = validator.segment_chunk_nodes(opm, start, stop, nodes_per_segment, clock)
            if len(segment_indices) == 0:
                continue
            width = b - a
            expanded_a = a - h.expansion * width
            expanded_b = b + h.expansion * width
            tau = (2.0 * jds - expanded_a[:, None] - expanded_b[:, None]) / (expanded_b - expanded_a)[:, None]
            recon = validator.reconstruct_segment_nodes(opm, segment_indices, tau, coeffs, params)
            truth = provider.position(jds.reshape(-1)).reshape(recon.shape)
            if error_metric == "km":
                err = np.linalg.norm(recon - truth, axis=2)
            elif error_metric == "angular":
                err = proto.angular_errors_arcsec(truth.reshape((-1, AXIS_COUNT)), recon.reshape((-1, AXIS_COUNT))).reshape(
                    (len(segment_indices), nodes_per_segment)
                )
            else:
                raise ValueError(f"unknown error metric {error_metric}")
            errors_by_segment[segment_indices] = err.astype(np.float32)
            max_by_segment = np.max(err, axis=1)
            for local_i, seg_np in enumerate(segment_indices):
                if max_by_segment[local_i] >= select_threshold:
                    selected.append((float(max_by_segment[local_i]), int(seg_np), jds[local_i].copy(), truth[local_i].copy()))
            if progress:
                print(f"  validated {stop}/{h.segment_count}; selected={len(selected)}", flush=True)
    finally:
        validator.close_if_needed(closeable)
    return errors_by_segment, selected


def finite_errors(errors_by_segment: np.ndarray) -> np.ndarray:
    values = errors_by_segment.reshape(-1)
    return values[np.isfinite(values)]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("opm", type=Path, help="baseline OPM supplying header/model/clock tables")
    parser.add_argument("--cache", type=Path, required=True, help="unquantized residual coefficient .npz cache")
    parser.add_argument("--de441", type=Path, required=True)
    parser.add_argument("--quant-bases", default="0.00032,0.00034,0.00036")
    parser.add_argument("--quant-pattern", default=None, help="override cache quant pattern")
    parser.add_argument("--nodes-per-segment", type=int, default=32)
    parser.add_argument("--chunk-size", type=int, default=4096)
    parser.add_argument("--select-threshold", type=float, default=0.001, help="optimize segments whose initial max is at least this value")
    parser.add_argument("--target", type=float, default=0.001, help="tail threshold used in counts")
    parser.add_argument("--limit", type=int, default=0, help="optimize only worst N selected segments; 0 means all selected")
    parser.add_argument("--max-passes", type=int, default=4)
    parser.add_argument("--radius", type=int, default=1)
    parser.add_argument("--objective", choices=["max", "p99max", "guarded", "topk_guarded", "topk_ranked"], default="guarded")
    parser.add_argument("--tail-topk", type=int, default=4, help="top-K tail mean guard for topk_guarded objective")
    parser.add_argument("--min-improvement", type=float, default=1e-12)
    parser.add_argument("--no-width-increase", action="store_true", help="reject rounding changes that exceed the initial global width table")
    parser.add_argument("--progress", action="store_true")
    parser.add_argument("--no-crc", action="store_true")
    args = parser.parse_args()

    proto.set_de441_path(args.de441)
    moon_proto.set_de441_path(args.de441)
    opm = validator.read_opm(args.opm, check_crc=not args.no_crc)
    clock = validator.mercury_clock(opm) or validator.moon_clock(opm)
    params = validator.frame_params_for_segments(opm, clock) if opm.frame_coeffs is not None else None

    with np.load(args.cache, allow_pickle=False) as data:
        residual_coeffs = np.asarray(data["coeffs"], dtype=np.float64)
        degree = int(data["residual_degree"].item())
        cache_pattern = str(data["quant_pattern"].item())
    if degree != opm.header.residual_degree:
        raise SystemExit(f"cache degree {degree} does not match OPM degree {opm.header.residual_degree}")
    if residual_coeffs.shape[:2] != (opm.header.segment_count, AXIS_COUNT):
        raise SystemExit("cache coefficient shape does not match OPM")

    pattern = args.quant_pattern or cache_pattern
    quant_bases = parse_csv_floats(args.quant_bases)
    overhead = opm.header.file_size - opm.header.payload_size

    print(f"baseline: {args.opm}")
    print(f"cache: {args.cache}")
    print(f"segments={opm.header.segment_count:,} degree={degree} nodes/segment={args.nodes_per_segment}")
    print(f"baseline file={opm.header.file_size / 1024 / 1024:.3f} MiB payload={opm.header.payload_size / 1024 / 1024:.3f} MiB width_sum={int(opm.widths.sum())}")
    print()

    for quant_base in quant_bases:
        steps = degree_quant_steps(degree, quant_base, pattern).astype(np.float32).astype(np.float64)
        qcoeffs = np.round(residual_coeffs / steps[None, None, :]).astype(np.int64)
        widths = zigzag_widths(qcoeffs)
        payload_size = payload_size_for_widths(opm.header.segment_count, widths)
        file_size = payload_size + overhead
        size_delta = opm.header.file_size - file_size
        candidate_opm = replace(opm, quant_steps=steps, widths=widths, qcoeffs=qcoeffs)
        coeffs = qcoeffs.astype(np.float64) * steps[None, None, :]

        print(f"quant_base={quant_base:.9g} pattern={pattern}")
        print(
            f"  estimated size: file={file_size / 1024 / 1024:.3f} MiB "
            f"payload={payload_size / 1024 / 1024:.3f} MiB save={size_delta / 1024 / 1024:.3f} MiB "
            f"width_sum={int(widths.sum())} axis_bits={tuple(int(x) for x in widths.sum(axis=1))}"
        )

        with SPK.open(str(args.de441)) as spk:
            initial_by_segment, selected = validate_candidate(
                spk,
                candidate_opm,
                coeffs,
                params,
                clock,
                nodes_per_segment=args.nodes_per_segment,
                chunk_size=args.chunk_size,
                select_threshold=args.select_threshold,
                progress=args.progress,
            )
        initial_errors = finite_errors(initial_by_segment)
        initial_segmax = np.nanmax(initial_by_segment, axis=1)
        initial_segmax = initial_segmax[np.isfinite(initial_segmax)]
        print(f"  initial:   {percentile_text(initial_errors)}")
        print(
            f"             seg>={args.target:.9g}: {int(np.count_nonzero(initial_segmax >= args.target)):,}  "
            f"samples>={args.target:.9g}: {int(np.count_nonzero(initial_errors >= args.target)):,}  "
            f"selected>={args.select_threshold:.9g}: {len(selected):,}"
        )

        selected.sort(reverse=True, key=lambda item: item[0])
        if args.limit > 0:
            selected = selected[: args.limit]
        width_limits = widths if args.no_width_increase else None
        optimized_by_segment = initial_by_segment.copy()
        qcoeffs_optimized = qcoeffs.copy()
        optimized_summaries: list[tuple[int, float, float, int]] = []
        for idx, (current_max, seg, jds_one, truth_one) in enumerate(selected, 1):
            result = opt_round.optimize_segment(
                candidate_opm,
                params,  # type: ignore[arg-type]
                seg,
                jds_one,
                truth_one,
                max_passes=args.max_passes,
                radius=args.radius,
                objective_mode=args.objective,
                min_improvement=args.min_improvement,
                width_limits=width_limits,
                tail_topk=args.tail_topk,
            )
            best_err = np.asarray(result["best_err"], dtype=np.float32)
            best_q = np.asarray(result["best_q"], dtype=np.int64)
            optimized_by_segment[seg] = best_err
            qcoeffs_optimized[seg] = best_q
            optimized_summaries.append((seg, current_max, float(np.max(best_err)), len(result["changes"])))
            if args.progress and (idx % 50 == 0 or idx == len(selected)):
                print(f"  optimized {idx}/{len(selected)} selected segments", flush=True)

        opt_widths = zigzag_widths(qcoeffs_optimized)
        opt_payload_size = payload_size_for_widths(opm.header.segment_count, opt_widths)
        opt_file_size = opt_payload_size + overhead
        optimized_errors = finite_errors(optimized_by_segment)
        optimized_segmax = np.nanmax(optimized_by_segment, axis=1)
        optimized_segmax = optimized_segmax[np.isfinite(optimized_segmax)]
        print(f"  optimized: {percentile_text(optimized_errors)}")
        print(
            f"             seg>={args.target:.9g}: {int(np.count_nonzero(optimized_segmax >= args.target)):,}  "
            f"samples>={args.target:.9g}: {int(np.count_nonzero(optimized_errors >= args.target)):,}  "
            f"optimized_segments={len(optimized_summaries):,}"
        )
        print(
            f"  optimized estimated size: file={opt_file_size / 1024 / 1024:.3f} MiB "
            f"save={((opm.header.file_size - opt_file_size) / 1024 / 1024):.3f} MiB "
            f"width_sum={int(opt_widths.sum())} axis_bits={tuple(int(x) for x in opt_widths.sum(axis=1))}"
        )
        if optimized_summaries:
            before = np.asarray([x[1] for x in optimized_summaries])
            after = np.asarray([x[2] for x in optimized_summaries])
            changed = sum(1 for x in optimized_summaries if x[3] > 0)
            print(
                f"  selected max before: {percentile_text(before)}\n"
                f"  selected max after:  {percentile_text(after)}\n"
                f"  changed segments: {changed:,}/{len(optimized_summaries):,}"
            )
            print("  worst optimized selected segments: seg before after delta changes")
            for seg, before_max, after_max, changes in optimized_summaries[:10]:
                print(f"    {seg:7d} {before_max:.9g} {after_max:.9g} {after_max-before_max:+.9g} {changes:5d}")
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
