#!/usr/bin/env python3
"""Optimize EMB rounding against heliocentric EMB using a fitted Sun PEF anchor.

This keeps the EMB quantization/width table unchanged, but adjusts per-segment
integer coefficients by small +/- steps when doing so improves the tail of:

    (EMB_pef - Sun_pef) vs (EMB_DE441 - Sun_DE441)

The Sun term is reconstructed from the supplied Sun PEF, so the fitted/quantized
Sun anchor error is included in the metric used to polish EMB.
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

import pef_demo.orbit_model as proto  # noqa: E402
from pef_demo import generator, validator  # noqa: E402
from pef_demo.body_configs import CONFIGS  # noqa: E402
from optimize_pef_relaxed_quant_tail import finite_errors, payload_size_for_widths, percentile_text, zigzag_widths  # noqa: E402
from optimize_pef_global_tail_pilot import boundaries_from_pef, summarize_segmax  # noqa: E402
from optimize_pef_segment_rounding import q_value_width, topk_mean, is_better_candidate  # noqa: E402

AXIS_COUNT = 3

_WORKER_EMB: validator.PefFile | None = None
_WORKER_SUN: validator.PefFile | None = None
_WORKER_PARAMS: np.ndarray | None = None
_WORKER_CLOCK: object | None = None
_WORKER_WIDTHS: np.ndarray | None = None
_WORKER_DE441: str | None = None
_WORKER_SPK: SPK | None = None
_WORKER_EMB_PROVIDER: validator.BaryProvider | None = None
_WORKER_SUN_PROVIDER: validator.BaryProvider | None = None
_WORKER_NODES = 32
_WORKER_MAX_PASSES = 4
_WORKER_RADIUS = 1
_WORKER_OBJECTIVE = "topk_ranked"
_WORKER_MIN_IMPROVEMENT = 1e-12
_WORKER_TAIL_TOPK = 4


def emb_config_from_pef(pef: validator.PefFile) -> object:
    base_cfg = CONFIGS["emb"]
    shape_degree = None if pef.header.reference_shape_degree == 255 else int(pef.header.reference_shape_degree)
    return replace(
        base_cfg,
        clock=replace(
            base_cfg.clock,
            period_days=float(pef.header.period_days),
            phase_start_jd=float(pef.header.phase_start_jd),
        ),
        residual_degree=int(pef.header.residual_degree),
        shape_degree=shape_degree,
        edge_margin_days=float(pef.header.edge_margin_days),
        apsis_step_days=float(pef.header.event_search_step_days),
        segment_domain_expansion_fraction=float(pef.header.expansion),
    )


def sun_position_from_pef(sun: validator.PefFile, jds: np.ndarray) -> np.ndarray:
    h = sun.header
    if h.model_kind != validator.MODEL_RAW_XYZ_CHEB:
        return validator.reconstruct_positions(sun, jds)
    coeffs = sun.qcoeffs.astype(np.float64) * sun.quant_steps[None, None, :]
    out = np.empty((len(jds), AXIS_COUNT), dtype=np.float64)
    coverage_end = h.coverage_start_jd + h.coverage_span_days
    d = h.segment_days if h.segment_addressing_kind == validator.SEGMENT_FIXED_DAYS else h.period_days
    seg_indices = np.floor((jds - h.phase_start_jd) / d).astype(np.int64)
    seg_indices = np.clip(seg_indices, 0, h.segment_count - 1)
    for si in np.unique(seg_indices):
        mask = seg_indices == si
        a, b = validator.segment_bounds(h, int(si), None)
        lo = max(a, h.coverage_start_jd)
        hi = min(b, coverage_end)
        in_seg = mask & (jds >= lo) & (jds <= hi if int(si) == h.segment_count - 1 else jds < hi)
        if not np.any(in_seg):
            # Formula addressing can put an endpoint on the neighboring segment;
            # fall back to the clipped segment for those rare boundary samples.
            in_seg = mask
        tau = validator.normalize_expanded(jds[in_seg], a, b, h.expansion)
        out[in_seg] = np.column_stack([proto.cheb_eval(coeffs[int(si), axis], tau) for axis in range(AXIS_COUNT)])
    return out


def emb_reconstruct_from_q(
    pef: validator.PefFile,
    params: np.ndarray,
    segment: int,
    qcoeffs: np.ndarray,
    jds: np.ndarray,
    basis: np.ndarray | None = None,
    shape_x: np.ndarray | None = None,
    shape_y: np.ndarray | None = None,
) -> np.ndarray:
    h = pef.header
    clock = validator.mercury_clock(pef) or validator.moon_clock(pef)
    a, b = validator.segment_bounds(h, segment, clock)
    tau = validator.normalize_expanded(jds, a, b, h.expansion)
    if basis is None:
        basis = np.polynomial.chebyshev.chebvander(tau, h.residual_degree)
    if shape_x is None:
        if pef.shape_x is None:
            raise ValueError("EMB PEF missing shape_x")
        shape_x = proto.cheb_eval(pef.shape_x, tau)
    if shape_y is None:
        if pef.shape_y is None:
            raise ValueError("EMB PEF missing shape_y")
        shape_y = proto.cheb_eval(pef.shape_y, tau)
    coeffs = qcoeffs.astype(np.float64) * pef.quant_steps[None, :]
    aligned = np.empty((1, len(jds), AXIS_COUNT), dtype=np.float64)
    aligned[0, :, 0] = shape_x + basis @ coeffs[0]
    aligned[0, :, 1] = shape_y + basis @ coeffs[1]
    aligned[0, :, 2] = basis @ coeffs[2]
    return validator.unalign_positions_batched(aligned, params[segment : segment + 1])[0]


def composite_errors(
    emb: validator.PefFile,
    sun: validator.PefFile,
    params: np.ndarray,
    segment: int,
    qcoeffs: np.ndarray,
    jds: np.ndarray,
    truth_helio: np.ndarray,
    basis: np.ndarray,
    shape_x: np.ndarray,
    shape_y: np.ndarray,
    sun_pef_pos: np.ndarray,
) -> np.ndarray:
    emb_pos = emb_reconstruct_from_q(emb, params, segment, qcoeffs, jds, basis, shape_x, shape_y)
    candidate = emb_pos - sun_pef_pos
    return proto.angular_errors_arcsec(truth_helio, candidate)


def optimize_segment_composite(
    emb: validator.PefFile,
    sun: validator.PefFile,
    params: np.ndarray,
    segment: int,
    jds: np.ndarray,
    truth_helio: np.ndarray,
    sun_pef_pos: np.ndarray,
    *,
    max_passes: int,
    radius: int,
    objective_mode: str,
    min_improvement: float,
    width_limits: np.ndarray,
    tail_topk: int,
) -> dict[str, object]:
    h = emb.header
    a, b = validator.segment_bounds(h, segment, validator.mercury_clock(emb) or validator.moon_clock(emb))
    tau = validator.normalize_expanded(jds, a, b, h.expansion)
    basis = np.polynomial.chebyshev.chebvander(tau, h.residual_degree)
    if emb.shape_x is None or emb.shape_y is None:
        raise ValueError("EMB PEF missing reference shape")
    shape_x = proto.cheb_eval(emb.shape_x, tau)
    shape_y = proto.cheb_eval(emb.shape_y, tau)

    q = emb.qcoeffs[segment].copy()
    initial_q = q.copy()
    err = composite_errors(emb, sun, params, segment, q, jds, truth_helio, basis, shape_x, shape_y, sun_pef_pos)
    best_err = err
    changes: list[tuple[int, int, int, float]] = []
    candidates = [(axis, degree) for axis in range(AXIS_COUNT) for degree in range(h.residual_degree + 1)]

    for _pass_idx in range(max_passes):
        improved = False
        if objective_mode == "topk_ranked":
            ranked: list[tuple[float, int, int, int]] = []
            current_tail = topk_mean(best_err, tail_topk)
            for axis, degree in candidates:
                current_value = int(q[axis, degree])
                for delta in range(-radius, radius + 1):
                    if delta == 0:
                        continue
                    trial_value = current_value + delta
                    if q_value_width(trial_value) > int(width_limits[axis, degree]):
                        continue
                    q[axis, degree] = trial_value
                    trial_err = composite_errors(emb, sun, params, segment, q, jds, truth_helio, basis, shape_x, shape_y, sun_pef_pos)
                    ranked.append((current_tail - topk_mean(trial_err, tail_topk), axis, degree, delta))
                    q[axis, degree] = current_value
            ranked.sort(reverse=True, key=lambda item: (item[0], item[2], item[1]))
            for _score, axis, degree, delta in ranked:
                current_value = int(q[axis, degree])
                trial_value = current_value + delta
                if q_value_width(trial_value) > int(width_limits[axis, degree]):
                    continue
                q[axis, degree] = trial_value
                trial_err = composite_errors(emb, sun, params, segment, q, jds, truth_helio, basis, shape_x, shape_y, sun_pef_pos)
                if is_better_candidate(trial_err, best_err, objective_mode, min_improvement, tail_topk):
                    best_err = trial_err
                    changes.append((axis, degree, delta, float(np.max(best_err))))
                    improved = True
                else:
                    q[axis, degree] = current_value
        else:
            raise ValueError(f"unsupported objective {objective_mode}")
        if not improved:
            break

    return {"initial_err": err, "best_err": best_err, "best_q": q, "initial_q": initial_q, "changes": changes}


def segment_nodes(pef: validator.PefFile, segment: int, nodes_per_segment: int, clock: object | None) -> np.ndarray:
    h = pef.header
    a, b = validator.segment_bounds(h, segment, clock)
    lo = max(a, h.coverage_start_jd)
    hi = min(b, h.coverage_start_jd + h.coverage_span_days)
    return proto.cheb_nodes(lo, hi, nodes_per_segment)


def validate_composite_by_segment(
    emb: validator.PefFile,
    sun: validator.PefFile,
    coeffs: np.ndarray,
    params: np.ndarray,
    clock: object | None,
    de441_path: Path,
    nodes_per_segment: int,
    chunk_size: int,
) -> np.ndarray:
    out = np.full((emb.header.segment_count, nodes_per_segment), np.nan, dtype=np.float32)
    emb_candidate = replace(emb, qcoeffs=np.rint(coeffs / emb.quant_steps[None, None, :]).astype(np.int64))
    with SPK.open(str(de441_path)) as spk:
        emb_provider = validator.BaryProvider(spk, validator.SPK_TARGET_IDS["emb"])
        sun_provider = validator.BaryProvider(spk, validator.SPK_TARGET_IDS["sun"])
        for start in range(0, emb.header.segment_count, chunk_size):
            stop = min(start + chunk_size, emb.header.segment_count)
            idx, jds, a, b = validator.segment_chunk_nodes(emb, start, stop, nodes_per_segment, clock)
            if len(idx) == 0:
                continue
            width = b - a
            ea = a - emb.header.expansion * width
            eb = b + emb.header.expansion * width
            tau = (2.0 * jds - ea[:, None] - eb[:, None]) / (eb - ea)[:, None]
            emb_pos = validator.reconstruct_segment_nodes(emb_candidate, idx, tau, coeffs, params)
            flat_jds = jds.reshape(-1)
            sun_pef = sun_position_from_pef(sun, flat_jds).reshape(emb_pos.shape)
            truth = (emb_provider.position(flat_jds) - sun_provider.position(flat_jds)).reshape(emb_pos.shape)
            cand = emb_pos - sun_pef
            err = proto.angular_errors_arcsec(truth.reshape((-1, AXIS_COUNT)), cand.reshape((-1, AXIS_COUNT))).reshape((len(idx), nodes_per_segment))
            out[idx] = err.astype(np.float32)
    return out


def init_worker(
    emb_path: str,
    sun_path: str,
    de441_path: str,
    no_crc: bool,
    nodes: int,
    max_passes: int,
    radius: int,
    objective: str,
    min_improvement: float,
    tail_topk: int,
) -> None:
    global _WORKER_EMB, _WORKER_SUN, _WORKER_PARAMS, _WORKER_CLOCK, _WORKER_WIDTHS, _WORKER_DE441, _WORKER_SPK, _WORKER_EMB_PROVIDER, _WORKER_SUN_PROVIDER
    global _WORKER_NODES, _WORKER_MAX_PASSES, _WORKER_RADIUS, _WORKER_OBJECTIVE, _WORKER_MIN_IMPROVEMENT, _WORKER_TAIL_TOPK
    proto.set_de441_path(Path(de441_path))
    emb = validator.read_pef(Path(emb_path), check_crc=not no_crc)
    sun = validator.read_pef(Path(sun_path), check_crc=not no_crc)
    clock = validator.mercury_clock(emb) or validator.moon_clock(emb)
    params = validator.frame_params_for_segments(emb, clock)
    _WORKER_EMB = emb
    _WORKER_SUN = sun
    _WORKER_PARAMS = params
    _WORKER_CLOCK = clock
    _WORKER_WIDTHS = emb.widths
    _WORKER_DE441 = de441_path
    _WORKER_SPK = SPK.open(de441_path)
    _WORKER_EMB_PROVIDER = validator.BaryProvider(_WORKER_SPK, validator.SPK_TARGET_IDS["emb"])
    _WORKER_SUN_PROVIDER = validator.BaryProvider(_WORKER_SPK, validator.SPK_TARGET_IDS["sun"])
    _WORKER_NODES = nodes
    _WORKER_MAX_PASSES = max_passes
    _WORKER_RADIUS = radius
    _WORKER_OBJECTIVE = objective
    _WORKER_MIN_IMPROVEMENT = min_improvement
    _WORKER_TAIL_TOPK = tail_topk


def optimize_one_process(args_tuple: tuple[int, np.ndarray]) -> tuple[int, np.ndarray, np.ndarray, list[object], float, float]:
    seg, before_err = args_tuple
    if _WORKER_EMB is None or _WORKER_SUN is None or _WORKER_PARAMS is None or _WORKER_WIDTHS is None or _WORKER_EMB_PROVIDER is None or _WORKER_SUN_PROVIDER is None:
        raise RuntimeError("worker not initialized")
    jds = segment_nodes(_WORKER_EMB, seg, _WORKER_NODES, _WORKER_CLOCK)
    truth = _WORKER_EMB_PROVIDER.position(jds) - _WORKER_SUN_PROVIDER.position(jds)
    sun_pef_pos = sun_position_from_pef(_WORKER_SUN, jds)
    result = optimize_segment_composite(
        _WORKER_EMB,
        _WORKER_SUN,
        _WORKER_PARAMS,
        seg,
        jds,
        truth,
        sun_pef_pos,
        max_passes=_WORKER_MAX_PASSES,
        radius=_WORKER_RADIUS,
        objective_mode=_WORKER_OBJECTIVE,
        min_improvement=_WORKER_MIN_IMPROVEMENT,
        width_limits=_WORKER_WIDTHS,
        tail_topk=_WORKER_TAIL_TOPK,
    )
    best_err = np.asarray(result["best_err"], dtype=np.float32)
    best_q = np.asarray(result["best_q"], dtype=np.int64)
    changes = list(result["changes"])
    return seg, best_err, best_q, changes, float(np.max(before_err)), float(np.max(best_err))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("emb", type=Path)
    parser.add_argument("--sun-pef", type=Path, required=True)
    parser.add_argument("--de441", type=Path, required=True)
    parser.add_argument("--nodes-per-segment", type=int, default=32)
    parser.add_argument("--chunk-size", type=int, default=4096)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--progress-every", type=int, default=1000)
    parser.add_argument("--max-passes", type=int, default=4)
    parser.add_argument("--radius", type=int, default=1)
    parser.add_argument("--objective", choices=["topk_ranked"], default="topk_ranked")
    parser.add_argument("--tail-topk", type=int, default=4)
    parser.add_argument("--min-improvement", type=float, default=1e-12)
    parser.add_argument("--p99-slack-abs", type=float, default=1e-8, help="accept old-budget failures if true global p99 stays within p99_floor + this slack")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--no-crc", action="store_true")
    args = parser.parse_args()

    emb = validator.read_pef(args.emb, check_crc=not args.no_crc)
    sun = validator.read_pef(args.sun_pef, check_crc=not args.no_crc)
    clock = validator.mercury_clock(emb) or validator.moon_clock(emb)
    params = validator.frame_params_for_segments(emb, clock)
    coeffs = emb.qcoeffs.astype(np.float64) * emb.quant_steps[None, None, :]
    overhead = emb.header.file_size - emb.header.payload_size

    print(f"baseline: {args.emb}")
    print(f"sun anchor: {args.sun_pef}")
    print(f"source=existing EMB qcoeffs/quant_steps composite=(EMB_pef - Sun_pef) vs (EMB_DE441 - Sun_DE441)")
    payload_size = payload_size_for_widths(emb.header.segment_count, emb.widths)
    print(
        f"size estimate: file={(payload_size + overhead) / 1024 / 1024:.3f} MiB "
        f"save={(emb.header.file_size - payload_size - overhead) / 1024 / 1024:.3f} MiB "
        f"width_sum={int(emb.widths.sum())} axis_bits={tuple(int(x) for x in emb.widths.sum(axis=1))}"
    )

    initial_by_segment = validate_composite_by_segment(emb, sun, coeffs, params, clock, args.de441, args.nodes_per_segment, args.chunk_size)
    initial_errors = finite_errors(initial_by_segment)
    p99_budget = float(np.percentile(initial_errors, 99))
    above_budget = int(np.count_nonzero(initial_errors > p99_budget))
    segmax = np.nanmax(initial_by_segment, axis=1)
    order = np.argsort(segmax)[::-1]
    if args.limit > 0:
        order = order[: args.limit]
    print(f"initial global: {percentile_text(initial_errors)}")
    print(f"initial segment max: {summarize_segmax(segmax)}")
    print(f"global p99 budget={p99_budget:.9g}; count_above_budget={above_budget:,}; processing={len(order):,} worst segments")

    optimized_by_segment = initial_by_segment.copy()
    qcoeffs_opt = emb.qcoeffs.copy()
    current_above = above_budget
    p99_floor = p99_budget
    accepted = rejected_nochange = rejected_budget = accepted_p99_slack = 0
    deferred_budget: list[tuple[int, np.ndarray, np.ndarray, list[object], float, float]] = []

    def accept(idx: int, result_tuple: tuple[int, np.ndarray, np.ndarray, list[object], float, float], *, defer_budget: bool = True) -> None:
        nonlocal accepted, rejected_nochange, rejected_budget, current_above, p99_floor, accepted_p99_slack
        seg, best_err, best_q, changes, before_max, after_max = result_tuple
        before_err = optimized_by_segment[seg].copy()
        if not changes or after_max >= before_max:
            rejected_nochange += 1
        else:
            old_above = int(np.count_nonzero(before_err > p99_budget))
            new_above = int(np.count_nonzero(best_err > p99_budget))
            trial_above = current_above - old_above + new_above
            accept_candidate = trial_above <= above_budget
            accepted_by_slack = False
            if not accept_candidate:
                trial_by_segment = optimized_by_segment.copy()
                trial_by_segment[seg] = best_err
                trial_p99 = float(np.percentile(finite_errors(trial_by_segment), 99))
                if trial_p99 <= p99_floor + args.p99_slack_abs:
                    accept_candidate = True
                    accepted_by_slack = True
                    p99_floor = min(p99_floor, trial_p99)
            if accept_candidate:
                optimized_by_segment[seg] = best_err
                qcoeffs_opt[seg] = best_q
                current_above = trial_above
                accepted += 1
                if accepted_by_slack:
                    accepted_p99_slack += 1
            else:
                rejected_budget += 1
                if defer_budget:
                    deferred_budget.append(result_tuple)
        if args.progress_every and (idx % args.progress_every == 0 or idx == len(order)):
            cur_segmax = np.nanmax(optimized_by_segment, axis=1)
            print(
                f"  processed {idx:,}/{len(order):,}; accepted={accepted:,}; "
                f"p99_slack={accepted_p99_slack:,}; current_max={np.max(cur_segmax):.9g}; "
                f"top100_mean={np.mean(np.partition(cur_segmax, len(cur_segmax)-100)[-100:]):.9g}; "
                f"count_above_budget={current_above:,}; p99_floor={p99_floor:.9g}",
                flush=True,
            )

    indexed_order = [(idx, int(seg)) for idx, seg in enumerate(order, 1)]
    if args.jobs <= 1:
        init_worker(str(args.emb), str(args.sun_pef), str(args.de441), args.no_crc, args.nodes_per_segment, args.max_passes, args.radius, args.objective, args.min_improvement, args.tail_topk)
        for idx, seg in indexed_order:
            accept(idx, optimize_one_process((seg, optimized_by_segment[seg].copy())))
    else:
        pending: dict[int, tuple[int, np.ndarray, np.ndarray, list[object], float, float]] = {}
        next_to_accept = 1
        with ProcessPoolExecutor(
            max_workers=args.jobs,
            initializer=init_worker,
            initargs=(
                str(args.emb), str(args.sun_pef), str(args.de441), args.no_crc, args.nodes_per_segment,
                args.max_passes, args.radius, args.objective, args.min_improvement, args.tail_topk,
            ),
        ) as executor:
            futures = {executor.submit(optimize_one_process, (seg, optimized_by_segment[seg].copy())): idx for idx, seg in indexed_order}
            for future in as_completed(futures):
                idx = futures[future]
                pending[idx] = future.result()
                while next_to_accept in pending:
                    accept(next_to_accept, pending.pop(next_to_accept))
                    next_to_accept += 1

    if deferred_budget:
        current_p99 = float(np.percentile(finite_errors(optimized_by_segment), 99))
        p99_floor = min(p99_floor, current_p99)
        accepted_before_revisit = accepted
        rejected_before_revisit = rejected_budget
        for result_tuple in sorted(deferred_budget, key=lambda item: item[4] - item[5], reverse=True):
            seg, best_err, best_q, changes, before_max, after_max = result_tuple
            # A later acceptance may already make this stale; only replay candidates
            # that still improve the current segment max.
            current_err = optimized_by_segment[seg]
            if not changes or float(np.max(best_err)) >= float(np.max(current_err)):
                continue
            trial_by_segment = optimized_by_segment.copy()
            trial_by_segment[seg] = best_err
            trial_p99 = float(np.percentile(finite_errors(trial_by_segment), 99))
            if trial_p99 <= p99_floor + args.p99_slack_abs:
                old_above = int(np.count_nonzero(current_err > p99_budget))
                new_above = int(np.count_nonzero(best_err > p99_budget))
                optimized_by_segment[seg] = best_err
                qcoeffs_opt[seg] = best_q
                current_above = current_above - old_above + new_above
                p99_floor = min(p99_floor, trial_p99)
                accepted += 1
                accepted_p99_slack += 1
                rejected_budget -= 1
        print(
            f"revisit budget rejects: accepted={accepted - accepted_before_revisit:,}; "
            f"remaining_rejected_budget={rejected_budget:,}; "
            f"p99_slack_total={accepted_p99_slack:,}; p99_floor={p99_floor:.9g}",
            flush=True,
        )

    optimized_errors = finite_errors(optimized_by_segment)
    opt_segmax = np.nanmax(optimized_by_segment, axis=1)
    opt_widths = zigzag_widths(qcoeffs_opt)
    opt_payload = payload_size_for_widths(emb.header.segment_count, opt_widths)
    print()
    print(f"accepted={accepted:,}; rejected_nochange={rejected_nochange:,}; rejected_budget={rejected_budget:,}")
    print(f"optimized global: {percentile_text(optimized_errors)}")
    print(f"optimized segment max: {summarize_segmax(opt_segmax)}")
    print(
        f"optimized size estimate: file={(opt_payload + overhead) / 1024 / 1024:.3f} MiB "
        f"save={(emb.header.file_size - opt_payload - overhead) / 1024 / 1024:.3f} MiB "
        f"width_sum={int(opt_widths.sum())} axis_bits={tuple(int(x) for x in opt_widths.sum(axis=1))}"
    )

    if args.output is not None:
        opt_widths_packed, payload = generator.pack_qcoeffs(qcoeffs_opt)
        if not np.array_equal(opt_widths_packed, opt_widths):
            raise SystemExit("packed width table does not match optimized width estimate")
        packed = generator.PackedBody(
            cfg=emb_config_from_pef(emb),
            boundaries=boundaries_from_pef(emb, clock),
            quant_steps=emb.quant_steps,
            widths=opt_widths_packed,
            qcoeffs=qcoeffs_opt,
            payload=payload,
            model_table=generator.pack_model_table(emb.shape_x, emb.shape_y, emb.frame_coeffs),
            clock_table=emb.clock_table,
            p50=float(np.percentile(optimized_errors, 50)),
            p95=float(np.percentile(optimized_errors, 95)),
            p99=float(np.percentile(optimized_errors, 99)),
            max_err=float(np.max(optimized_errors)),
        )
        size = generator.write_pef_file(args.output, packed, emb.header.source_start_jd, emb.header.source_end_jd, emb.header.coverage_start_jd, emb.header.coverage_span_days)
        print(f"wrote {args.output} ({size / 1024 / 1024:.3f} MiB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
