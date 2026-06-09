#!/usr/bin/env python3
"""Pilot global no-threshold guarded rounding on the worst segment tail.

This diagnostic avoids a fixed validation-error threshold.  It requantizes cached
residual coefficients, validates all segments, sorts segments by their current
max error, and then walks the worst N segments.  A segment adjustment is accepted
when it improves that segment, does not increase the global count above the
initial p99 budget, and does not increase the width table.
"""
from __future__ import annotations

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
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
from opm_demo import generator, validator  # noqa: E402
from opm_demo.body_configs import CONFIGS, QuantConfig  # noqa: E402
from opm_demo.packing import degree_quant_steps  # noqa: E402
import optimize_opm_segment_rounding as opt_round  # noqa: E402
from optimize_opm_polish_common import (  # noqa: E402
    finite_errors,
    payload_size_for_widths,
    percentile_text,
    validate_candidate,
    zigzag_widths,
)

AXIS_COUNT = 3

_WORKER_OPM: validator.OpmFile | None = None
_WORKER_PARAMS: np.ndarray | None = None
_WORKER_PROVIDER: object | None = None
_WORKER_CLOSEABLE: object | None = None
_WORKER_CLOCK: object | None = None
_WORKER_WIDTHS: np.ndarray | None = None
_WORKER_NODES_PER_SEGMENT = 0
_WORKER_MAX_PASSES = 0
_WORKER_RADIUS = 0
_WORKER_OBJECTIVE = "topk_ranked"
_WORKER_MIN_IMPROVEMENT = 0.0
_WORKER_TAIL_TOPK = 0
_WORKER_ERROR_METRIC = "angular"


def init_process_worker(
    opm_path: str,
    cache_path: str | None,
    de441_path: str,
    quant_base: float,
    quant_pattern: str | None,
    requant_existing: bool,
    no_crc: bool,
    nodes_per_segment: int,
    max_passes: int,
    radius: int,
    objective: str,
    min_improvement: float,
    tail_topk: int,
    error_metric: str,
) -> None:
    global _WORKER_OPM, _WORKER_PARAMS, _WORKER_PROVIDER, _WORKER_CLOSEABLE, _WORKER_CLOCK, _WORKER_WIDTHS
    global _WORKER_NODES_PER_SEGMENT, _WORKER_MAX_PASSES, _WORKER_RADIUS, _WORKER_OBJECTIVE, _WORKER_MIN_IMPROVEMENT, _WORKER_TAIL_TOPK, _WORKER_ERROR_METRIC

    proto.set_de441_path(Path(de441_path))
    moon_proto.set_de441_path(Path(de441_path))
    opm = validator.read_opm(Path(opm_path), check_crc=not no_crc)
    clock = validator.mercury_clock(opm) or validator.moon_clock(opm)
    params = validator.frame_params_for_segments(opm, clock) if opm.frame_coeffs is not None else None
    if params is None and opm.header.model_kind != validator.MODEL_RAW_XYZ_CHEB:
        raise RuntimeError("this diagnostic expects raw Cheb or a framed reference-shape OPM")
    if cache_path:
        with np.load(Path(cache_path), allow_pickle=False) as data:
            residual_coeffs = np.asarray(data["coeffs"], dtype=np.float64)
            degree = int(data["residual_degree"].item())
            cache_pattern = str(data["quant_pattern"].item())
        pattern = quant_pattern or cache_pattern
        steps = degree_quant_steps(degree, quant_base, pattern).astype(np.float32).astype(np.float64)
        qcoeffs = np.round(residual_coeffs / steps[None, None, :]).astype(np.int64)
        widths = zigzag_widths(qcoeffs)
        _WORKER_OPM = replace(opm, quant_steps=steps, widths=widths, qcoeffs=qcoeffs)
    elif requant_existing:
        if quant_pattern is None:
            raise RuntimeError("--quant-pattern is required with --requant-existing")
        coeffs = opm.qcoeffs.astype(np.float64) * opm.quant_steps[None, None, :]
        steps = degree_quant_steps(opm.header.residual_degree, quant_base, quant_pattern).astype(np.float32).astype(np.float64)
        qcoeffs = np.round(coeffs / steps[None, None, :]).astype(np.int64)
        widths = zigzag_widths(qcoeffs)
        _WORKER_OPM = replace(opm, quant_steps=steps, widths=widths, qcoeffs=qcoeffs)
    else:
        widths = opm.widths
        _WORKER_OPM = opm
    _WORKER_PARAMS = params
    _WORKER_CLOCK = clock
    _WORKER_WIDTHS = widths
    spk = SPK.open(de441_path)
    _WORKER_PROVIDER, _WORKER_CLOSEABLE = validator.truth_position_provider(spk, _WORKER_OPM)
    _WORKER_NODES_PER_SEGMENT = nodes_per_segment
    _WORKER_MAX_PASSES = max_passes
    _WORKER_RADIUS = radius
    _WORKER_OBJECTIVE = objective
    _WORKER_MIN_IMPROVEMENT = min_improvement
    _WORKER_TAIL_TOPK = tail_topk
    _WORKER_ERROR_METRIC = error_metric


def optimize_one_segment_process(args_tuple: tuple[int, np.ndarray]) -> tuple[int, np.ndarray, np.ndarray, list[object], float, float]:
    seg, before_err = args_tuple
    if _WORKER_OPM is None or _WORKER_PROVIDER is None or _WORKER_WIDTHS is None:
        raise RuntimeError("process worker was not initialized")
    return optimize_one_segment(
        _WORKER_OPM,
        _WORKER_PARAMS,
        _WORKER_PROVIDER,
        _WORKER_CLOCK,
        _WORKER_WIDTHS,
        seg,
        before_err,
        nodes_per_segment=_WORKER_NODES_PER_SEGMENT,
        max_passes=_WORKER_MAX_PASSES,
        radius=_WORKER_RADIUS,
        objective=_WORKER_OBJECTIVE,
        min_improvement=_WORKER_MIN_IMPROVEMENT,
        tail_topk=_WORKER_TAIL_TOPK,
        error_metric=_WORKER_ERROR_METRIC,
    )


def optimize_one_segment(
    opm: validator.OpmFile,
    params: np.ndarray | None,
    provider: object,
    clock: object | None,
    widths: np.ndarray,
    seg: int,
    before_err: np.ndarray,
    *,
    nodes_per_segment: int,
    max_passes: int,
    radius: int,
    objective: str,
    min_improvement: float,
    tail_topk: int,
    error_metric: str,
) -> tuple[int, np.ndarray, np.ndarray, list[object], float, float]:
    before_max = float(np.max(before_err))
    jds = opt_round.segment_eval_nodes(opm, seg, nodes_per_segment, clock)
    truth = provider.position(jds)
    result = opt_round.optimize_segment(
        opm,
        params,
        seg,
        jds,
        truth,
        max_passes=max_passes,
        radius=radius,
        objective_mode=objective,
        min_improvement=min_improvement,
        width_limits=widths,
        tail_topk=tail_topk,
        error_metric=error_metric,
    )
    best_err = np.asarray(result["best_err"], dtype=np.float32)
    best_q = np.asarray(result["best_q"], dtype=np.int64)
    changes = list(result["changes"])
    after_max = float(np.max(best_err))
    return seg, best_err, best_q, changes, before_max, after_max


def summarize_segmax(values: np.ndarray) -> str:
    return (
        f"p50={np.percentile(values, 50):.9g} "
        f"p95={np.percentile(values, 95):.9g} "
        f"p99={np.percentile(values, 99):.9g} "
        f"p99.5={np.percentile(values, 99.5):.9g} "
        f"p99.9={np.percentile(values, 99.9):.9g} "
        f"max={np.max(values):.9g}"
    )


def boundaries_from_opm(opm: validator.OpmFile, clock: object | None) -> np.ndarray:
    bounds = [validator.segment_bounds(opm.header, i, clock) for i in range(opm.header.segment_count)]
    return np.asarray([bounds[0][0]] + [b for _, b in bounds], dtype=np.float64)


def model_table_from_opm(opm: validator.OpmFile) -> bytes:
    return generator.pack_model_table(opm.shape_x, opm.shape_y, opm.frame_coeffs)


def config_from_opm(opm: validator.OpmFile, quant_base: float | None, quant_pattern: str | None) -> object:
    base_cfg = CONFIGS[validator.body_name_from_id(opm.header.body_id)]
    shape_degree = None if opm.header.reference_shape_degree == 255 else int(opm.header.reference_shape_degree)
    cfg = replace(
        base_cfg,
        clock=replace(
            base_cfg.clock,
            period_days=float(opm.header.period_days),
            phase_start_jd=float(opm.header.phase_start_jd),
        ),
        residual_degree=int(opm.header.residual_degree),
        shape_degree=shape_degree,
        edge_margin_days=float(opm.header.edge_margin_days),
        apsis_step_days=float(opm.header.event_search_step_days),
        segment_domain_expansion_fraction=float(opm.header.expansion),
    )
    if quant_base is not None and quant_pattern is not None:
        cfg = replace(cfg, quant=QuantConfig(float(quant_base), quant_pattern))
    return cfg


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("opm", type=Path)
    parser.add_argument("--cache", type=Path, default=None, help="optional unquantized residual coefficient cache; omit to optimize the OPM's existing qcoeffs")
    parser.add_argument("--de441", type=Path, required=True)
    parser.add_argument("--requant-existing", action="store_true", help="approximate scan: requantize reconstructed coefficients from the input OPM instead of requiring a residual cache")
    parser.add_argument("--quant-base", type=float, default=0.00034)
    parser.add_argument("--quant-pattern", default=None)
    parser.add_argument("--nodes-per-segment", type=int, default=32)
    parser.add_argument("--chunk-size", type=int, default=4096)
    parser.add_argument("--limit", type=int, default=5000, help="process only the worst N initial segments")
    parser.add_argument("--max-passes", type=int, default=4)
    parser.add_argument("--radius", type=int, default=1)
    parser.add_argument("--objective", choices=["guarded", "topk_ranked"], default="topk_ranked")
    parser.add_argument("--tail-topk", type=int, default=4)
    parser.add_argument("--min-improvement", type=float, default=1e-12)
    parser.add_argument("--progress-every", type=int, default=250)
    parser.add_argument("--jobs", type=int, default=1, help="optimize this many segments concurrently; acceptance remains ordered")
    parser.add_argument("--output", type=Path, default=None, help="write optimized qcoeffs to this OPM path after the pilot")
    parser.add_argument("--error-metric", choices=["angular", "km"], default="angular", help="optimize angular arcsec errors, or linear km errors for raw Sun anchor")
    parser.add_argument("--no-crc", action="store_true")
    args = parser.parse_args()
    if args.jobs < 1:
        raise SystemExit("--jobs must be >= 1")

    proto.set_de441_path(args.de441)
    moon_proto.set_de441_path(args.de441)
    opm = validator.read_opm(args.opm, check_crc=not args.no_crc)
    clock = validator.mercury_clock(opm) or validator.moon_clock(opm)
    params = validator.frame_params_for_segments(opm, clock) if opm.frame_coeffs is not None else None
    if params is None and opm.header.model_kind != validator.MODEL_RAW_XYZ_CHEB:
        raise SystemExit("this diagnostic expects raw Cheb or a framed reference-shape OPM")

    overhead = opm.header.file_size - opm.header.payload_size
    if args.cache is not None:
        with np.load(args.cache, allow_pickle=False) as data:
            residual_coeffs = np.asarray(data["coeffs"], dtype=np.float64)
            degree = int(data["residual_degree"].item())
            cache_pattern = str(data["quant_pattern"].item())
        if residual_coeffs.shape[:2] != (opm.header.segment_count, AXIS_COUNT):
            raise SystemExit(
                f"cache coefficient shape {residual_coeffs.shape} does not match "
                f"OPM segments/axes {(opm.header.segment_count, AXIS_COUNT)}"
            )
        pattern = args.quant_pattern or cache_pattern
        steps = degree_quant_steps(degree, args.quant_base, pattern).astype(np.float32).astype(np.float64)
        qcoeffs = np.round(residual_coeffs / steps[None, None, :]).astype(np.int64)
        widths = zigzag_widths(qcoeffs)
        candidate_opm = replace(opm, quant_steps=steps, widths=widths, qcoeffs=qcoeffs)
        source_text = f"requantized cache quant_base={args.quant_base:.9g} pattern={pattern}"
    elif args.requant_existing:
        if args.quant_pattern is None:
            raise SystemExit("--quant-pattern is required with --requant-existing")
        steps = degree_quant_steps(opm.header.residual_degree, args.quant_base, args.quant_pattern).astype(np.float32).astype(np.float64)
        source_coeffs = opm.qcoeffs.astype(np.float64) * opm.quant_steps[None, None, :]
        qcoeffs = np.round(source_coeffs / steps[None, None, :]).astype(np.int64)
        widths = zigzag_widths(qcoeffs)
        candidate_opm = replace(opm, quant_steps=steps, widths=widths, qcoeffs=qcoeffs)
        pattern = args.quant_pattern
        source_text = f"requantized existing OPM quant_base={args.quant_base:.9g} pattern={pattern}"
    else:
        steps = opm.quant_steps
        qcoeffs = opm.qcoeffs
        widths = opm.widths
        candidate_opm = opm
        pattern = None
        source_text = "existing OPM qcoeffs/quant_steps"
    payload_size = payload_size_for_widths(opm.header.segment_count, widths)
    coeffs = qcoeffs.astype(np.float64) * steps[None, None, :]

    print(f"baseline: {args.opm}")
    print(f"source={source_text} objective={args.objective} topK={args.tail_topk}")
    print(
        f"size estimate: file={(payload_size + overhead) / 1024 / 1024:.3f} MiB "
        f"save={(opm.header.file_size - payload_size - overhead) / 1024 / 1024:.3f} MiB "
        f"width_sum={int(widths.sum())} axis_bits={tuple(int(x) for x in widths.sum(axis=1))}"
    )

    with SPK.open(str(args.de441)) as spk:
        # Use threshold 0 to collect all errors, but not all truth arrays.
        initial_by_segment, _ = validate_candidate(
            spk,
            candidate_opm,
            coeffs,
            params,
            clock,
            nodes_per_segment=args.nodes_per_segment,
            chunk_size=args.chunk_size,
            select_threshold=float("inf"),
            progress=False,
            error_metric=args.error_metric,
        )

    initial_errors = finite_errors(initial_by_segment)
    initial_p99_budget = float(np.percentile(initial_errors, 99))
    initial_above_budget = int(np.count_nonzero(initial_errors > initial_p99_budget))
    initial_segmax = np.nanmax(initial_by_segment, axis=1)
    order = np.argsort(initial_segmax)[::-1]
    if args.limit > 0:
        order = order[: args.limit]

    print(f"initial global: {percentile_text(initial_errors)}")
    print(f"initial segment max: {summarize_segmax(initial_segmax)}")
    print(f"global p99 budget={initial_p99_budget:.9g}; count_above_budget={initial_above_budget:,}; processing={len(order):,} worst segments")

    optimized_by_segment = initial_by_segment.copy()
    qcoeffs_optimized = qcoeffs.copy()
    current_above_budget = initial_above_budget
    accepted = 0
    rejected_budget = 0
    rejected_nochange = 0
    before_after: list[tuple[int, float, float, int]] = []

    def accept_result(idx: int, result_tuple: tuple[int, np.ndarray, np.ndarray, list[object], float, float]) -> None:
        nonlocal accepted, rejected_budget, rejected_nochange, current_above_budget
        seg, best_err, best_q, changes, before_max, after_max = result_tuple
        before_err = optimized_by_segment[seg].copy()
        if not changes or after_max >= before_max:
            rejected_nochange += 1
        else:
            old_above = int(np.count_nonzero(before_err > initial_p99_budget))
            new_above = int(np.count_nonzero(best_err > initial_p99_budget))
            trial_above = current_above_budget - old_above + new_above
            if trial_above <= initial_above_budget:
                optimized_by_segment[seg] = best_err
                qcoeffs_optimized[seg] = best_q
                current_above_budget = trial_above
                accepted += 1
                before_after.append((seg, before_max, after_max, len(changes)))
            else:
                rejected_budget += 1
        if args.progress_every and (idx % args.progress_every == 0 or idx == len(order)):
            current_segmax = np.nanmax(optimized_by_segment, axis=1)
            print(
                f"  processed {idx:,}/{len(order):,}; accepted={accepted:,}; "
                f"current_max={np.max(current_segmax):.9g}; "
                f"top100_mean={np.mean(np.partition(current_segmax, len(current_segmax)-100)[-100:]):.9g}; "
                f"count_above_budget={current_above_budget:,}",
                flush=True,
            )

    if args.jobs <= 1:
        with SPK.open(str(args.de441)) as spk:
            provider, closeable = validator.truth_position_provider(spk, candidate_opm)
            try:
                for idx, seg_np in enumerate(order, 1):
                    seg = int(seg_np)
                    before_err = optimized_by_segment[seg].copy()
                    result_tuple = optimize_one_segment(
                        candidate_opm,
                        params,
                        provider,
                        clock,
                        widths,
                        seg,
                        before_err,
                        nodes_per_segment=args.nodes_per_segment,
                        max_passes=args.max_passes,
                        radius=args.radius,
                        objective=args.objective,
                        min_improvement=args.min_improvement,
                        tail_topk=args.tail_topk,
                        error_metric=args.error_metric,
                    )
                    accept_result(idx, result_tuple)
            finally:
                validator.close_if_needed(closeable)
    else:
        indexed_order = [(idx, int(seg_np)) for idx, seg_np in enumerate(order, 1)]
        pending_results: dict[int, tuple[int, np.ndarray, np.ndarray, list[object], float, float]] = {}
        next_to_accept = 1
        with ProcessPoolExecutor(
            max_workers=args.jobs,
            initializer=init_process_worker,
            initargs=(
                str(args.opm),
                None if args.cache is None else str(args.cache),
                str(args.de441),
                args.quant_base,
                args.quant_pattern,
                args.requant_existing,
                args.no_crc,
                args.nodes_per_segment,
                args.max_passes,
                args.radius,
                args.objective,
                args.min_improvement,
                args.tail_topk,
                args.error_metric,
            ),
        ) as executor:
            futures = {
                executor.submit(optimize_one_segment_process, (seg, optimized_by_segment[seg].copy())): idx
                for idx, seg in indexed_order
            }
            for future in as_completed(futures):
                idx = futures[future]
                pending_results[idx] = future.result()
                while next_to_accept in pending_results:
                    accept_result(next_to_accept, pending_results.pop(next_to_accept))
                    next_to_accept += 1

    optimized_errors = finite_errors(optimized_by_segment)
    optimized_segmax = np.nanmax(optimized_by_segment, axis=1)
    opt_widths = zigzag_widths(qcoeffs_optimized)
    opt_payload = payload_size_for_widths(opm.header.segment_count, opt_widths)
    print()
    print(f"accepted={accepted:,}; rejected_nochange={rejected_nochange:,}; rejected_budget={rejected_budget:,}")
    print(f"optimized global: {percentile_text(optimized_errors)}")
    print(f"optimized segment max: {summarize_segmax(optimized_segmax)}")
    print(
        f"optimized size estimate: file={(opt_payload + overhead) / 1024 / 1024:.3f} MiB "
        f"save={(opm.header.file_size - opt_payload - overhead) / 1024 / 1024:.3f} MiB "
        f"width_sum={int(opt_widths.sum())} axis_bits={tuple(int(x) for x in opt_widths.sum(axis=1))}"
    )
    if args.output is not None:
        opt_widths_packed, payload = generator.pack_qcoeffs(qcoeffs_optimized)
        if not np.array_equal(opt_widths_packed, opt_widths):
            raise SystemExit("packed width table does not match optimized width estimate")
        output_quant_base = args.quant_base if args.cache is not None else None
        output_quant_pattern = pattern if args.cache is not None else None
        cfg = config_from_opm(candidate_opm, output_quant_base, output_quant_pattern)
        packed = generator.PackedBody(
            cfg=cfg,
            boundaries=boundaries_from_opm(candidate_opm, clock),
            quant_steps=candidate_opm.quant_steps,
            widths=opt_widths_packed,
            qcoeffs=qcoeffs_optimized,
            payload=payload,
            model_table=model_table_from_opm(candidate_opm),
            clock_table=candidate_opm.clock_table,
            p50=float(np.percentile(optimized_errors, 50)),
            p95=float(np.percentile(optimized_errors, 95)),
            p99=float(np.percentile(optimized_errors, 99)),
            max_err=float(np.max(optimized_errors)),
        )
        size = generator.write_opm_file(
            args.output,
            packed,
            candidate_opm.header.source_start_jd,
            candidate_opm.header.source_end_jd,
            candidate_opm.header.coverage_start_jd,
            candidate_opm.header.coverage_span_days,
        )
        print(f"wrote {args.output} ({size / 1024 / 1024:.3f} MiB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
