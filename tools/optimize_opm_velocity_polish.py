#!/usr/bin/env python3
"""Velocity-aware local rounding polish for existing OPM files.

This diagnostic does not write a new OPM.  It starts from the already-quantized
coefficients in an OPM file, selects high-velocity-error segments, and tries
small +/- integer coefficient moves that reduce derivative residuals while
keeping native position residuals guarded.
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
from jplephem.spk import SPK

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import opm_demo.moon_model as moon_model  # noqa: E402
import opm_demo.orbit_model as orbit_model  # noqa: E402
from opm_demo import generator, validator  # noqa: E402
from optimize_opm_global_tail import boundaries_from_opm, config_from_opm, model_table_from_opm  # noqa: E402
from optimize_opm_polish_common import payload_size_for_widths, zigzag_widths  # noqa: E402

AXIS_COUNT = 3
SECONDS_PER_DAY = 86400.0


def q_value_width(value: int) -> int:
    encoded = value * 2 if value >= 0 else -value * 2 - 1
    return max(1, int(encoded).bit_length())


def percentile_text(values: np.ndarray) -> str:
    return (
        f"p50={np.percentile(values, 50):.9g} "
        f"p95={np.percentile(values, 95):.9g} "
        f"p99={np.percentile(values, 99):.9g} "
        f"max={np.max(values):.9g}"
    )


def segment_nodes(opm: validator.OpmFile, segment: int, nodes_per_segment: int, clock: object | None) -> tuple[np.ndarray, float, float]:
    h = opm.header
    coverage_end = h.coverage_start_jd + h.coverage_span_days
    a, b = validator.segment_bounds(h, segment, clock)
    lo = max(a, h.coverage_start_jd)
    hi = min(b, coverage_end)
    if hi <= lo:
        return np.empty((0,), dtype=np.float64), a, b
    return orbit_model.cheb_nodes(lo, hi, nodes_per_segment), a, b


def precompute_segment(opm: validator.OpmFile, params: np.ndarray | None, segment: int, jds: np.ndarray, a: float, b: float) -> dict[str, np.ndarray | float | None]:
    h = opm.header
    width = b - a
    expanded_a = a - h.expansion * width
    expanded_b = b + h.expansion * width
    scale = 2.0 / (expanded_b - expanded_a)
    tau = (2.0 * jds - expanded_a - expanded_b) / (expanded_b - expanded_a)
    basis = np.polynomial.chebyshev.chebvander(tau, h.residual_degree)
    if h.residual_degree > 0:
        dbasis = np.polynomial.chebyshev.chebvander(tau, h.residual_degree - 1)
    else:
        dbasis = np.empty((len(tau), 0), dtype=np.float64)
    shape_x = shape_y = dshape_x = dshape_y = None
    frame_params = None
    if h.model_kind != validator.MODEL_RAW_XYZ_CHEB:
        if opm.shape_x is None or opm.shape_y is None or params is None:
            raise ValueError(f"{opm.path}: orbital-frame model missing model table")
        shape_x = orbit_model.cheb_eval(opm.shape_x, tau)
        shape_y = orbit_model.cheb_eval(opm.shape_y, tau)
        dshape_x = np.polynomial.chebyshev.chebval(tau, validator.cheb_derivative_coeffs(opm.shape_x)) * scale
        dshape_y = np.polynomial.chebyshev.chebval(tau, validator.cheb_derivative_coeffs(opm.shape_y)) * scale
        frame_params = params[segment : segment + 1]
    return {
        "basis": basis,
        "dbasis": dbasis,
        "scale": scale,
        "shape_x": shape_x,
        "shape_y": shape_y,
        "dshape_x": dshape_x,
        "dshape_y": dshape_y,
        "frame_params": frame_params,
    }


def reconstruct_from_q(opm: validator.OpmFile, qcoeffs: np.ndarray, pre: dict[str, np.ndarray | float | None]) -> tuple[np.ndarray, np.ndarray]:
    h = opm.header
    coeffs = qcoeffs.astype(np.float64) * opm.quant_steps[None, :]
    dcoeffs = validator.cheb_derivative_coeffs(coeffs)
    basis = pre["basis"]
    dbasis = pre["dbasis"]
    scale = float(pre["scale"])
    assert isinstance(basis, np.ndarray)
    assert isinstance(dbasis, np.ndarray)
    pos = np.column_stack([basis @ coeffs[axis] for axis in range(AXIS_COUNT)])
    vel = np.column_stack([dbasis @ dcoeffs[axis] * scale for axis in range(AXIS_COUNT)])
    if h.model_kind == validator.MODEL_RAW_XYZ_CHEB:
        return pos, vel
    shape_x = pre["shape_x"]
    shape_y = pre["shape_y"]
    dshape_x = pre["dshape_x"]
    dshape_y = pre["dshape_y"]
    frame_params = pre["frame_params"]
    assert isinstance(shape_x, np.ndarray)
    assert isinstance(shape_y, np.ndarray)
    assert isinstance(dshape_x, np.ndarray)
    assert isinstance(dshape_y, np.ndarray)
    assert isinstance(frame_params, np.ndarray)
    aligned_pos = np.empty((1, pos.shape[0], AXIS_COUNT), dtype=np.float64)
    aligned_vel = np.empty_like(aligned_pos)
    aligned_pos[0, :, 0] = shape_x + pos[:, 0]
    aligned_pos[0, :, 1] = shape_y + pos[:, 1]
    aligned_pos[0, :, 2] = pos[:, 2]
    aligned_vel[0, :, 0] = dshape_x + vel[:, 0]
    aligned_vel[0, :, 1] = dshape_y + vel[:, 1]
    aligned_vel[0, :, 2] = vel[:, 2]
    return (
        validator.unalign_positions_batched(aligned_pos, frame_params)[0],
        validator.unalign_positions_batched(aligned_vel, frame_params)[0],
    )


def segment_errors(opm: validator.OpmFile, qcoeffs: np.ndarray, pre: dict[str, np.ndarray | float | None], truth_pos: np.ndarray, truth_vel: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pos, vel = reconstruct_from_q(opm, qcoeffs, pre)
    pos_err = np.linalg.norm(pos - truth_pos, axis=1)
    vel_err = np.linalg.norm(vel - truth_vel, axis=1) * 1_000_000.0 / SECONDS_PER_DAY
    return pos_err, vel_err


def velocity_objective(vel_err: np.ndarray, mode: str) -> float:
    if mode == "p99":
        return float(np.percentile(vel_err, 99))
    if mode == "max":
        return float(np.max(vel_err))
    if mode == "p99max":
        return float(np.percentile(vel_err, 99) + 0.1 * np.max(vel_err))
    raise ValueError(f"unknown objective {mode}")


def optimize_segment(
    opm: validator.OpmFile,
    params: np.ndarray | None,
    segment: int,
    jds: np.ndarray,
    a: float,
    b: float,
    truth_pos: np.ndarray,
    truth_vel: np.ndarray,
    *,
    max_passes: int,
    radius: int,
    objective_mode: str,
    pos_guard_km: float,
    pos_guard_rel: float,
    min_improvement: float,
    no_width_increase: bool,
) -> dict[str, object]:
    pre = precompute_segment(opm, params, segment, jds, a, b)
    q = opm.qcoeffs[segment].copy()
    initial_q = q.copy()
    pos_err, vel_err = segment_errors(opm, q, pre, truth_pos, truth_vel)
    best_pos = pos_err
    best_vel = vel_err
    best_obj = velocity_objective(best_vel, objective_mode)
    initial_pos_max = float(np.max(pos_err))
    initial_pos_p99 = float(np.percentile(pos_err, 99))
    pos_max_limit = initial_pos_max + pos_guard_km + abs(initial_pos_max) * pos_guard_rel
    pos_p99_limit = initial_pos_p99 + pos_guard_km + abs(initial_pos_p99) * pos_guard_rel
    width_limits = opm.widths if no_width_increase else None
    changes: list[tuple[int, int, int, float, float]] = []
    candidates = [(axis, degree) for axis in range(AXIS_COUNT) for degree in range(opm.header.residual_degree + 1)]
    for _pass_idx in range(max_passes):
        improved = False
        for axis, degree in sorted(candidates, key=lambda item: (item[1], item[0]), reverse=True):
            current_value = int(q[axis, degree])
            local_best_delta = 0
            local_best_obj = best_obj
            local_best_pos = best_pos
            local_best_vel = best_vel
            for delta in range(-radius, radius + 1):
                if delta == 0:
                    continue
                trial_value = current_value + delta
                if width_limits is not None and q_value_width(trial_value) > int(width_limits[axis, degree]):
                    continue
                q[axis, degree] = trial_value
                trial_pos, trial_vel = segment_errors(opm, q, pre, truth_pos, truth_vel)
                if float(np.max(trial_pos)) > pos_max_limit:
                    continue
                if float(np.percentile(trial_pos, 99)) > pos_p99_limit:
                    continue
                trial_obj = velocity_objective(trial_vel, objective_mode)
                if trial_obj + min_improvement < local_best_obj:
                    local_best_delta = delta
                    local_best_obj = trial_obj
                    local_best_pos = trial_pos
                    local_best_vel = trial_vel
            q[axis, degree] = current_value
            if local_best_delta:
                q[axis, degree] = current_value + local_best_delta
                best_obj = local_best_obj
                best_pos = local_best_pos
                best_vel = local_best_vel
                changes.append((axis, degree, local_best_delta, float(np.percentile(best_vel, 99)), float(np.max(best_pos))))
                improved = True
        if not improved:
            break
    return {
        "segment": segment,
        "initial_q": initial_q,
        "best_q": q,
        "initial_pos": pos_err,
        "initial_vel": vel_err,
        "best_pos": best_pos,
        "best_vel": best_vel,
        "changes": changes,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("opm", type=Path)
    parser.add_argument("--de441", type=Path, required=True)
    parser.add_argument("--nodes-per-segment", type=int, default=64)
    parser.add_argument("--chunk-size", type=int, default=4096)
    parser.add_argument("--select", choices=["p99", "max"], default="p99")
    parser.add_argument("--limit", type=int, default=24)
    parser.add_argument("--segments", default="", help="comma-separated segment indices; overrides --limit selection")
    parser.add_argument("--max-passes", type=int, default=3)
    parser.add_argument("--radius", type=int, default=1)
    parser.add_argument("--objective", choices=["p99", "max", "p99max"], default="p99max")
    parser.add_argument("--pos-guard-km", type=float, default=0.0)
    parser.add_argument("--pos-guard-rel", type=float, default=0.0)
    parser.add_argument("--min-improvement", type=float, default=1e-9)
    parser.add_argument("--no-width-increase", action="store_true")
    parser.add_argument("--progress", action="store_true")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--no-crc", action="store_true")
    args = parser.parse_args()

    orbit_model.set_de441_path(args.de441)
    moon_model.set_de441_path(args.de441)
    opm = validator.read_opm(args.opm, check_crc=not args.no_crc)
    clock = validator.mercury_clock(opm) or validator.moon_clock(opm)
    params = validator.frame_params_for_segments(opm, clock) if opm.frame_coeffs is not None else None
    coeffs = opm.qcoeffs.astype(np.float64) * opm.quant_steps[None, None, :]
    dcoeffs = validator.cheb_derivative_coeffs(coeffs)

    segment_scores: list[tuple[float, int, float, float]] = []
    initial_pos_parts: list[np.ndarray] = []
    initial_vel_parts: list[np.ndarray] = []
    with SPK.open(str(args.de441)) as spk:
        provider, closeable = validator.truth_position_provider(spk, opm)
        try:
            if not hasattr(provider, "velocity"):
                raise SystemExit(f"{args.opm}: truth provider does not expose velocity")
            for start in range(0, opm.header.segment_count, args.chunk_size):
                stop = min(start + args.chunk_size, opm.header.segment_count)
                jds, pos_err, vel_err_km_day = validator.native_residual_segment_chunk(
                    provider,
                    opm,
                    coeffs,
                    dcoeffs,
                    params,
                    clock,
                    start,
                    stop,
                    args.nodes_per_segment,
                )
                if len(jds) == 0:
                    continue
                rows = len(pos_err) // args.nodes_per_segment
                pos_2d = pos_err.reshape((rows, args.nodes_per_segment))
                vel_2d = (vel_err_km_day * 1_000_000.0 / SECONDS_PER_DAY).reshape((rows, args.nodes_per_segment))
                segment_indices, _, _, _ = validator.segment_chunk_nodes(opm, start, stop, args.nodes_per_segment, clock)
                for local_idx, segment in enumerate(segment_indices):
                    score = float(np.percentile(vel_2d[local_idx], 99) if args.select == "p99" else np.max(vel_2d[local_idx]))
                    segment_scores.append((score, int(segment), float(np.max(pos_2d[local_idx])), float(np.max(vel_2d[local_idx]))))
                initial_pos_parts.append(pos_err)
                initial_vel_parts.append(vel_err_km_day * 1_000_000.0 / SECONDS_PER_DAY)
                if args.progress:
                    print(f"scanned {stop}/{opm.header.segment_count}", flush=True)

            if args.segments:
                selected_segments = [int(text.strip()) for text in args.segments.split(",") if text.strip()]
            else:
                segment_scores.sort(reverse=True, key=lambda item: item[0])
                selected_segments = [segment for _score, segment, _pos_max, _vel_max in (segment_scores if args.limit <= 0 else segment_scores[: args.limit])]

            qcoeffs_opt = opm.qcoeffs.copy()
            results = []
            for idx, segment in enumerate(selected_segments, 1):
                jds, a, b = segment_nodes(opm, segment, args.nodes_per_segment, clock)
                if len(jds) == 0:
                    continue
                truth_pos = provider.position(jds)
                truth_vel = provider.velocity(jds)
                result = optimize_segment(
                    opm,
                    params,
                    segment,
                    jds,
                    a,
                    b,
                    truth_pos,
                    truth_vel,
                    max_passes=args.max_passes,
                    radius=args.radius,
                    objective_mode=args.objective,
                    pos_guard_km=args.pos_guard_km,
                    pos_guard_rel=args.pos_guard_rel,
                    min_improvement=args.min_improvement,
                    no_width_increase=args.no_width_increase,
                )
                results.append(result)
                qcoeffs_opt[segment] = np.asarray(result["best_q"], dtype=np.int64)
                if args.progress:
                    print(f"optimized {idx}/{len(selected_segments)} segment={segment}", flush=True)
        finally:
            validator.close_if_needed(closeable)

    initial_pos = np.concatenate(initial_pos_parts)
    initial_vel = np.concatenate(initial_vel_parts)
    print(f"file: {args.opm}")
    print(f"segments={opm.header.segment_count:,} selected={len(results):,} nodes/segment={args.nodes_per_segment} objective={args.objective}")
    print(f"position guard: max<=initial+{args.pos_guard_km:g}km+{args.pos_guard_rel:g}rel p99 same")
    print(f"initial all position km: {percentile_text(initial_pos)}")
    print(f"initial all velocity mm/s: {percentile_text(initial_vel)}")
    print()
    print("selected velocity-aware polish")
    print("seg initial_pos_max best_pos_max pos_delta initial_vel_p99 best_vel_p99 vel_p99_delta initial_vel_max best_vel_max changes")
    all_initial_vel = []
    all_best_vel = []
    all_initial_pos = []
    all_best_pos = []
    changed_segments = 0
    for result in results:
        segment = int(result["segment"])
        initial_pos_seg = np.asarray(result["initial_pos"])
        best_pos_seg = np.asarray(result["best_pos"])
        initial_vel_seg = np.asarray(result["initial_vel"])
        best_vel_seg = np.asarray(result["best_vel"])
        changes = result["changes"]
        changed_segments += int(len(changes) > 0)
        all_initial_vel.append(initial_vel_seg)
        all_best_vel.append(best_vel_seg)
        all_initial_pos.append(initial_pos_seg)
        all_best_pos.append(best_pos_seg)
        initial_vel_p99 = float(np.percentile(initial_vel_seg, 99))
        best_vel_p99 = float(np.percentile(best_vel_seg, 99))
        initial_vel_max = float(np.max(initial_vel_seg))
        best_vel_max = float(np.max(best_vel_seg))
        initial_pos_max = float(np.max(initial_pos_seg))
        best_pos_max = float(np.max(best_pos_seg))
        print(
            f"{segment:7d} {initial_pos_max:.9g} {best_pos_max:.9g} {best_pos_max-initial_pos_max:+.9g} "
            f"{initial_vel_p99:.9g} {best_vel_p99:.9g} {best_vel_p99-initial_vel_p99:+.9g} "
            f"{initial_vel_max:.9g} {best_vel_max:.9g} {len(changes):7d}"
        )
    if results:
        print()
        print(f"changed segments: {changed_segments}/{len(results)}")
        print(f"selected initial velocity mm/s: {percentile_text(np.concatenate(all_initial_vel))}")
        print(f"selected best velocity mm/s:    {percentile_text(np.concatenate(all_best_vel))}")
        print(f"selected initial position km:   {percentile_text(np.concatenate(all_initial_pos))}")
        print(f"selected best position km:      {percentile_text(np.concatenate(all_best_pos))}")

    opt_widths, payload = generator.pack_qcoeffs(qcoeffs_opt)
    overhead = opm.header.file_size - opm.header.payload_size
    opt_payload_est = payload_size_for_widths(opm.header.segment_count, opt_widths)
    print(
        f"optimized size estimate: file={(opt_payload_est + overhead) / 1024 / 1024:.3f} MiB "
        f"delta={((opt_payload_est + overhead) - opm.header.file_size) / 1024 / 1024:+.3f} MiB "
        f"width_sum={int(opt_widths.sum())} axis_bits={tuple(int(x) for x in opt_widths.sum(axis=1))}"
    )
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        packed = generator.PackedBody(
            cfg=config_from_opm(opm, None, None),
            boundaries=boundaries_from_opm(opm, clock),
            quant_steps=opm.quant_steps,
            widths=opt_widths,
            qcoeffs=qcoeffs_opt,
            payload=payload,
            model_table=model_table_from_opm(opm),
            clock_table=opm.clock_table,
            p50=float(np.percentile(initial_pos, 50)),
            p95=float(np.percentile(initial_pos, 95)),
            p99=float(np.percentile(initial_pos, 99)),
            max_err=float(np.max(initial_pos)),
        )
        size = generator.write_opm_file(
            args.output,
            packed,
            opm.header.source_start_jd,
            opm.header.source_end_jd,
            opm.header.coverage_start_jd,
            opm.header.coverage_span_days,
        )
        print(f"wrote {args.output} ({size / 1024 / 1024:.3f} MiB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
