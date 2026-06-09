#!/usr/bin/env python3
"""Write a PEF candidate after relaxed quantization and guarded rounding.

This is a writer-side tuning diagnostic: it keeps the baseline PEF model/clock
metadata, requantizes cached residual coefficients at a candidate quant base,
optimizes high-tail segments with width-safe p99-guarded rounding, and writes a
new PEF file with the optimized qcoeffs.  The PEF format and reader are unchanged.
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

import pef_demo.moon_model as moon_proto  # noqa: E402
import pef_demo.orbit_model as proto  # noqa: E402
from pef_demo import generator, validator  # noqa: E402
from pef_demo.body_configs import CONFIGS, QuantConfig  # noqa: E402
from pef_demo.packing import degree_quant_steps  # noqa: E402
import optimize_pef_segment_rounding as opt_round  # noqa: E402
from optimize_pef_relaxed_quant_tail import (  # noqa: E402
    finite_errors,
    payload_size_for_widths,
    percentile_text,
    validate_candidate,
    zigzag_widths,
)

AXIS_COUNT = 3


def boundaries_from_pef(pef: validator.PefFile, clock: object | None) -> np.ndarray:
    bounds = [validator.segment_bounds(pef.header, i, clock) for i in range(pef.header.segment_count)]
    return np.asarray([bounds[0][0]] + [b for _, b in bounds], dtype=np.float64)


def model_table_from_pef(pef: validator.PefFile) -> bytes:
    return generator.pack_model_table(pef.shape_x, pef.shape_y, pef.frame_coeffs)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pef", type=Path, help="baseline PEF supplying header/model/clock tables")
    parser.add_argument("--cache", type=Path, required=True, help="unquantized residual coefficient .npz cache")
    parser.add_argument("--de441", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--quant-base", type=float, default=0.00034)
    parser.add_argument("--quant-pattern", default=None)
    parser.add_argument("--nodes-per-segment", type=int, default=32)
    parser.add_argument("--chunk-size", type=int, default=4096)
    parser.add_argument("--select-threshold", type=float, default=0.00085)
    parser.add_argument("--target", type=float, default=0.001)
    parser.add_argument("--max-passes", type=int, default=4)
    parser.add_argument("--radius", type=int, default=1)
    parser.add_argument("--objective", choices=["max", "p99max", "guarded"], default="guarded")
    parser.add_argument("--min-improvement", type=float, default=1e-12)
    parser.add_argument("--allow-width-increase", action="store_true")
    parser.add_argument("--progress", action="store_true")
    parser.add_argument("--no-crc", action="store_true")
    args = parser.parse_args()

    proto.set_de441_path(args.de441)
    moon_proto.set_de441_path(args.de441)

    pef = validator.read_pef(args.pef, check_crc=not args.no_crc)
    clock = validator.mercury_clock(pef) or validator.moon_clock(pef)
    params = validator.frame_params_for_segments(pef, clock) if pef.frame_coeffs is not None else None
    if params is None:
        raise SystemExit("this diagnostic expects a framed reference-shape PEF")

    with np.load(args.cache, allow_pickle=False) as data:
        residual_coeffs = np.asarray(data["coeffs"], dtype=np.float64)
        degree = int(data["residual_degree"].item())
        cache_pattern = str(data["quant_pattern"].item())
    if degree != pef.header.residual_degree:
        raise SystemExit(f"cache degree {degree} does not match PEF degree {pef.header.residual_degree}")
    if residual_coeffs.shape != pef.qcoeffs.shape:
        raise SystemExit(f"cache coeff shape {residual_coeffs.shape} does not match PEF qcoeff shape {pef.qcoeffs.shape}")

    pattern = args.quant_pattern or cache_pattern
    steps = degree_quant_steps(degree, args.quant_base, pattern).astype(np.float32).astype(np.float64)
    qcoeffs = np.round(residual_coeffs / steps[None, None, :]).astype(np.int64)
    widths = zigzag_widths(qcoeffs)
    payload_size = payload_size_for_widths(pef.header.segment_count, widths)
    overhead = pef.header.file_size - pef.header.payload_size
    print(f"baseline: {args.pef}")
    print(f"output: {args.output}")
    print(
        f"initial candidate q={args.quant_base:.9g} pattern={pattern}: "
        f"file={(payload_size + overhead) / 1024 / 1024:.3f} MiB "
        f"save={(pef.header.file_size - payload_size - overhead) / 1024 / 1024:.3f} MiB "
        f"width_sum={int(widths.sum())} axis_bits={tuple(int(x) for x in widths.sum(axis=1))}"
    )

    candidate_pef = replace(pef, quant_steps=steps, widths=widths, qcoeffs=qcoeffs)
    coeffs = qcoeffs.astype(np.float64) * steps[None, None, :]
    with SPK.open(str(args.de441)) as spk:
        initial_by_segment, selected = validate_candidate(
            spk,
            candidate_pef,
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
    print(f"initial validation: {percentile_text(initial_errors)}")
    print(
        f"  seg>={args.target:.9g}: {int(np.count_nonzero(initial_segmax >= args.target)):,}  "
        f"samples>={args.target:.9g}: {int(np.count_nonzero(initial_errors >= args.target)):,}  "
        f"selected>={args.select_threshold:.9g}: {len(selected):,}"
    )

    selected.sort(reverse=True, key=lambda item: item[0])
    width_limits = None if args.allow_width_increase else widths
    optimized_by_segment = initial_by_segment.copy()
    qcoeffs_optimized = qcoeffs.copy()
    optimized_summaries: list[tuple[int, float, float, int]] = []
    for idx, (current_max, seg, jds_one, truth_one) in enumerate(selected, 1):
        result = opt_round.optimize_segment(
            candidate_pef,
            params,
            seg,
            jds_one,
            truth_one,
            max_passes=args.max_passes,
            radius=args.radius,
            objective_mode=args.objective,
            min_improvement=args.min_improvement,
            width_limits=width_limits,
        )
        best_err = np.asarray(result["best_err"], dtype=np.float32)
        best_q = np.asarray(result["best_q"], dtype=np.int64)
        optimized_by_segment[seg] = best_err
        qcoeffs_optimized[seg] = best_q
        optimized_summaries.append((seg, current_max, float(np.max(best_err)), len(result["changes"])))
        if args.progress and (idx % 50 == 0 or idx == len(selected)):
            print(f"  optimized {idx}/{len(selected)} selected segments", flush=True)

    opt_widths, payload = generator.pack_qcoeffs(qcoeffs_optimized)
    opt_payload_size = len(payload)
    optimized_errors = finite_errors(optimized_by_segment)
    optimized_segmax = np.nanmax(optimized_by_segment, axis=1)
    optimized_segmax = optimized_segmax[np.isfinite(optimized_segmax)]
    print(f"optimized validation sample: {percentile_text(optimized_errors)}")
    print(
        f"  seg>={args.target:.9g}: {int(np.count_nonzero(optimized_segmax >= args.target)):,}  "
        f"samples>={args.target:.9g}: {int(np.count_nonzero(optimized_errors >= args.target)):,}  "
        f"optimized_segments={len(optimized_summaries):,}"
    )
    print(
        f"optimized size estimate: file={(opt_payload_size + overhead) / 1024 / 1024:.3f} MiB "
        f"save={(pef.header.file_size - opt_payload_size - overhead) / 1024 / 1024:.3f} MiB "
        f"width_sum={int(opt_widths.sum())} axis_bits={tuple(int(x) for x in opt_widths.sum(axis=1))}"
    )

    base_cfg = CONFIGS[validator.body_name_from_id(pef.header.body_id)]
    cfg = replace(
        base_cfg,
        clock=replace(
            base_cfg.clock,
            period_days=float(pef.header.period_days),
            phase_start_jd=float(pef.header.phase_start_jd),
        ),
        quant=QuantConfig(float(args.quant_base), pattern),
        segment_domain_expansion_fraction=float(pef.header.expansion),
    )
    packed = generator.PackedBody(
        cfg=cfg,
        boundaries=boundaries_from_pef(pef, clock),
        quant_steps=steps,
        widths=opt_widths,
        qcoeffs=qcoeffs_optimized,
        payload=payload,
        model_table=model_table_from_pef(pef),
        clock_table=pef.clock_table,
        p50=float(np.percentile(optimized_errors, 50)),
        p95=float(np.percentile(optimized_errors, 95)),
        p99=float(np.percentile(optimized_errors, 99)),
        max_err=float(np.max(optimized_errors)),
    )
    size = generator.write_pef_file(
        args.output,
        packed,
        pef.header.source_start_jd,
        pef.header.source_end_jd,
        pef.header.coverage_start_jd,
        pef.header.coverage_span_days,
    )
    print(f"wrote {args.output} ({size / 1024 / 1024:.3f} MiB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
