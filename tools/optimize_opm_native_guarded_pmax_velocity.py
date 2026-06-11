#!/usr/bin/env python3
"""Optimize a native-vector OPM with the guarded/refined pmax machinery.

This is the native-metric counterpart of optimize_opm_ssb_sun_anchor_pmax.py:
it keeps the file's storage vector and error metric unchanged, but uses the
stronger pmax scoring grid, local peak refinement, shifted guard grid, and
capped lexicographic acceptance developed for the SSB Sun-anchor optimizer.
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
from optimize_opm_global_tail import boundaries_from_opm, config_from_opm, model_table_from_opm, summarize_segmax  # noqa: E402
from optimize_opm_polish_common import finite_errors, payload_size_for_widths, percentile_text, zigzag_widths  # noqa: E402
from optimize_opm_segment_rounding import errors_for_q, q_value_width, topk_mean  # noqa: E402
from optimize_opm_ssb_sun_anchor_pmax import (  # noqa: E402
    capped_lex_better,
    golden_section_max,
    guard_segment_nodes,
    segment_nodes,
)

AXIS_COUNT = 3

_WORKER_OPM: validator.OpmFile | None = None
_WORKER_PARAMS: np.ndarray | None = None
_WORKER_CLOCK: object | None = None
_WORKER_WIDTHS: np.ndarray | None = None
_WORKER_SPK: SPK | None = None
_WORKER_PROVIDER: object | None = None
_WORKER_NODES = 32
_WORKER_NODE_GRID = "cheb-center-uniform-endpoints"
_WORKER_MAX_PASSES = 4
_WORKER_RADIUS = 1
_WORKER_OBJECTIVE = "capped_lex_guarded_ceiling"
_WORKER_MIN_IMPROVEMENT = 1e-12
_WORKER_TAIL_TOPK = 4
_WORKER_REFINE_PEAKS = 3
_WORKER_GUARD_GRID = "shifted"
_WORKER_PMAX_CAP = 7e-4
_WORKER_ERROR_METRIC = "angular"


def objective_score(err: np.ndarray, tail_topk: int) -> tuple[float, float, float]:
    return (float(np.max(err)), topk_mean(err, tail_topk), float(np.percentile(err, 99)))


def native_errors(
    opm: validator.OpmFile,
    params: np.ndarray | None,
    segment: int,
    qcoeffs: np.ndarray,
    jds: np.ndarray,
    truth: np.ndarray,
    basis: np.ndarray,
    shape_x: np.ndarray | None,
    shape_y: np.ndarray | None,
    error_metric: str,
) -> np.ndarray:
    return errors_for_q(opm, params, segment, qcoeffs, jds, truth, basis, shape_x, shape_y, error_metric)


def initial_native_errors_at_jds(
    opm: validator.OpmFile,
    params: np.ndarray | None,
    segment: int,
    jds: np.ndarray,
    provider: object,
    error_metric: str,
) -> np.ndarray:
    h = opm.header
    a, b = validator.segment_bounds(h, segment, validator.mercury_clock(opm) or validator.moon_clock(opm))
    tau = validator.normalize_expanded(jds, a, b, h.expansion)
    basis = np.polynomial.chebyshev.chebvander(tau, h.residual_degree)
    shape_x = proto.cheb_eval(opm.shape_x, tau) if opm.shape_x is not None else None
    shape_y = proto.cheb_eval(opm.shape_y, tau) if opm.shape_y is not None else None
    truth = provider.position(jds)
    return native_errors(opm, params, segment, opm.qcoeffs[segment], jds, truth, basis, shape_x, shape_y, error_metric)


def append_refined_peak_nodes_native(
    opm: validator.OpmFile,
    params: np.ndarray | None,
    segment: int,
    base_jds: np.ndarray,
    provider: object,
    refine_peaks: int,
    error_metric: str,
) -> np.ndarray:
    if refine_peaks <= 0 or len(base_jds) < 3:
        return base_jds
    order = np.argsort(base_jds)
    jds = np.asarray(base_jds[order], dtype=np.float64)
    errs = initial_native_errors_at_jds(opm, params, segment, jds, provider, error_metric)
    local: list[int] = []
    for i in range(1, len(jds) - 1):
        if errs[i] >= errs[i - 1] and errs[i] >= errs[i + 1]:
            local.append(i)
    if not local:
        local = list(range(len(jds)))
    ranked = sorted(local, key=lambda i: float(errs[i]), reverse=True)
    refined: list[float] = []
    used: set[int] = set()
    for i in ranked:
        if len(refined) >= refine_peaks:
            break
        if i in used:
            continue
        used.add(i)
        left_i = max(0, i - 1)
        right_i = min(len(jds) - 1, i + 1)
        lo = float(jds[left_i])
        hi = float(jds[right_i])
        if hi <= lo:
            refined.append(float(jds[i]))
            continue

        def err_at(x: float) -> float:
            return float(
                initial_native_errors_at_jds(
                    opm,
                    params,
                    segment,
                    np.asarray([x], dtype=np.float64),
                    provider,
                    error_metric,
                )[0]
            )

        refined.append(golden_section_max(err_at, lo, hi))
    while len(refined) < refine_peaks:
        refined.append(float(jds[int(np.argmax(errs))]))
    return np.sort(np.concatenate([jds, np.asarray(refined, dtype=np.float64)]))


def optimize_segment_native_guarded(
    opm: validator.OpmFile,
    params: np.ndarray | None,
    segment: int,
    jds: np.ndarray,
    truth: np.ndarray,
    guard_jds: np.ndarray | None = None,
    guard_truth: np.ndarray | None = None,
    *,
    max_passes: int,
    radius: int,
    objective_mode: str,
    min_improvement: float,
    width_limits: np.ndarray,
    tail_topk: int,
    guard_slack_abs: float = 2e-5,
    guard_slack_rel: float = 0.05,
    pmax_cap: float = 7e-4,
    error_metric: str = "angular",
) -> dict[str, object]:
    h = opm.header
    a, b = validator.segment_bounds(h, segment, validator.mercury_clock(opm) or validator.moon_clock(opm))
    tau = validator.normalize_expanded(jds, a, b, h.expansion)
    basis = np.polynomial.chebyshev.chebvander(tau, h.residual_degree)
    shape_x = proto.cheb_eval(opm.shape_x, tau) if opm.shape_x is not None else None
    shape_y = proto.cheb_eval(opm.shape_y, tau) if opm.shape_y is not None else None

    if guard_jds is not None:
        if guard_truth is None:
            raise ValueError("guard truth is required when guard_jds is supplied")
        guard_tau = validator.normalize_expanded(guard_jds, a, b, h.expansion)
        guard_basis = np.polynomial.chebyshev.chebvander(guard_tau, h.residual_degree)
        guard_shape_x = proto.cheb_eval(opm.shape_x, guard_tau) if opm.shape_x is not None else None
        guard_shape_y = proto.cheb_eval(opm.shape_y, guard_tau) if opm.shape_y is not None else None
    else:
        guard_basis = None
        guard_shape_x = None
        guard_shape_y = None

    q = opm.qcoeffs[segment].copy()
    initial_q = q.copy()
    err = native_errors(opm, params, segment, q, jds, truth, basis, shape_x, shape_y, error_metric)
    if guard_jds is not None and guard_basis is not None and guard_truth is not None:
        guard_err = native_errors(opm, params, segment, q, guard_jds, guard_truth, guard_basis, guard_shape_x, guard_shape_y, error_metric)
    else:
        guard_err = None
    best_err = err
    best_guard_err = guard_err
    changes: list[tuple[int, int, int, float]] = []
    candidates = [(axis, degree) for axis in range(AXIS_COUNT) for degree in range(h.residual_degree + 1)]

    for _pass_idx in range(max_passes):
        improved = False
        if objective_mode in {"topk_ranked", "pmax_ranked", "capped_lex_guarded_ceiling"}:
            ranked: list[tuple[float, float, int, int, int]] = []
            current_max = float(np.max(best_err))
            current_tail = topk_mean(best_err, tail_topk)
            max_first = objective_mode == "pmax_ranked" or (
                objective_mode == "capped_lex_guarded_ceiling" and current_max > pmax_cap
            )
            for axis, degree in candidates:
                current_value = int(q[axis, degree])
                for delta in range(-radius, radius + 1):
                    if delta == 0:
                        continue
                    trial_value = current_value + delta
                    if q_value_width(trial_value) > int(width_limits[axis, degree]):
                        continue
                    q[axis, degree] = trial_value
                    trial_err = native_errors(opm, params, segment, q, jds, truth, basis, shape_x, shape_y, error_metric)
                    trial_max = float(np.max(trial_err))
                    trial_tail = topk_mean(trial_err, tail_topk)
                    if max_first:
                        ranked.append((current_max - trial_max, current_tail - trial_tail, axis, degree, delta))
                    else:
                        ranked.append((current_tail - trial_tail, current_max - trial_max, axis, degree, delta))
                    q[axis, degree] = current_value
            ranked.sort(reverse=True, key=lambda item: (item[0], item[1], item[3], item[2]))
            for _score, _secondary, axis, degree, delta in ranked:
                current_value = int(q[axis, degree])
                trial_value = current_value + delta
                if q_value_width(trial_value) > int(width_limits[axis, degree]):
                    continue
                q[axis, degree] = trial_value
                trial_err = native_errors(opm, params, segment, q, jds, truth, basis, shape_x, shape_y, error_metric)
                trial_guard_err = None
                if objective_mode == "capped_lex_guarded_ceiling" and guard_jds is not None and guard_basis is not None and guard_truth is not None:
                    trial_guard_err = native_errors(
                        opm,
                        params,
                        segment,
                        q,
                        guard_jds,
                        guard_truth,
                        guard_basis,
                        guard_shape_x,
                        guard_shape_y,
                        error_metric,
                    )

                if objective_mode == "pmax_ranked":
                    trial_max = float(np.max(trial_err))
                    best_max = float(np.max(best_err))
                    better = trial_max + min_improvement < best_max
                    if not better and abs(trial_max - best_max) <= min_improvement:
                        better = topk_mean(trial_err, tail_topk) + min_improvement < topk_mean(best_err, tail_topk)
                elif objective_mode == "capped_lex_guarded_ceiling":
                    better = capped_lex_better(trial_err, best_err, min_improvement, tail_topk, pmax_cap)
                    if better and trial_guard_err is not None and best_guard_err is not None:
                        guard_ceiling = float(np.max(best_guard_err)) * (1.0 + guard_slack_rel) + guard_slack_abs
                        better = float(np.max(trial_guard_err)) <= guard_ceiling
                else:
                    trial = objective_score(trial_err, tail_topk)
                    best = objective_score(best_err, tail_topk)
                    better = (
                        trial[1] + min_improvement < best[1]
                        or (trial[1] <= best[1] + min_improvement and trial[0] + min_improvement < best[0])
                        or (trial[1] <= best[1] + min_improvement and trial[0] <= best[0] + min_improvement and trial[2] + min_improvement < best[2])
                    )
                if better:
                    best_err = trial_err
                    if trial_guard_err is not None:
                        best_guard_err = trial_guard_err
                    changes.append((axis, degree, delta, float(np.max(best_err))))
                    improved = True
                else:
                    q[axis, degree] = current_value
        else:
            raise ValueError(f"unsupported objective {objective_mode}")
        if not improved:
            break

    return {"initial_err": err, "best_err": best_err, "best_q": q, "initial_q": initial_q, "changes": changes}


def validate_native_by_segment(
    opm: validator.OpmFile,
    coeffs: np.ndarray,
    params: np.ndarray | None,
    clock: object | None,
    de441_path: Path,
    nodes_per_segment: int,
    chunk_size: int,
    node_grid: str = "cheb",
    refine_peaks: int = 0,
    error_metric: str = "angular",
) -> np.ndarray:
    base_nodes = segment_nodes(opm, 0, nodes_per_segment, clock, node_grid)
    nodes_per_row = len(base_nodes) + max(0, refine_peaks)
    out = np.full((opm.header.segment_count, nodes_per_row), np.nan, dtype=np.float32)
    candidate_opm = replace(opm, qcoeffs=np.rint(coeffs / opm.quant_steps[None, None, :]).astype(np.int64))
    with SPK.open(str(de441_path)) as spk:
        provider, closeable = validator.truth_position_provider(spk, candidate_opm)
        try:
            for start in range(0, opm.header.segment_count, chunk_size):
                stop = min(start + chunk_size, opm.header.segment_count)
                indices: list[int] = []
                bounds: list[tuple[float, float]] = []
                nodes_parts: list[np.ndarray] = []
                for si in range(start, stop):
                    jds = segment_nodes(opm, si, nodes_per_segment, clock, node_grid)
                    if len(jds) == 0:
                        continue
                    if refine_peaks > 0:
                        jds = append_refined_peak_nodes_native(candidate_opm, params, si, jds, provider, refine_peaks, error_metric)
                    a, b = validator.segment_bounds(opm.header, si, clock)
                    indices.append(si)
                    bounds.append((a, b))
                    nodes_parts.append(jds)
                if not indices:
                    continue
                by_len: dict[int, list[int]] = {}
                for pos, part in enumerate(nodes_parts):
                    by_len.setdefault(len(part), []).append(pos)
                for positions in by_len.values():
                    sub_indices = [indices[pos] for pos in positions]
                    sub_bounds = [bounds[pos] for pos in positions]
                    jds = np.vstack([nodes_parts[pos] for pos in positions])
                    idx = np.asarray(sub_indices, dtype=np.int64)
                    bound_arr = np.asarray(sub_bounds, dtype=np.float64)
                    a = bound_arr[:, 0]
                    b = bound_arr[:, 1]
                    width = b - a
                    ea = a - opm.header.expansion * width
                    eb = b + opm.header.expansion * width
                    tau = (2.0 * jds - ea[:, None] - eb[:, None]) / (eb - ea)[:, None]
                    recon = validator.reconstruct_segment_nodes(candidate_opm, idx, tau, coeffs, params)
                    truth = provider.position(jds.reshape(-1)).reshape(recon.shape)
                    if error_metric == "km":
                        err = np.linalg.norm(recon - truth, axis=2)
                    elif error_metric == "angular":
                        err = proto.angular_errors_arcsec(truth.reshape((-1, AXIS_COUNT)), recon.reshape((-1, AXIS_COUNT))).reshape(jds.shape)
                    else:
                        raise ValueError(f"unknown error metric {error_metric}")
                    if err.shape[1] > out.shape[1]:
                        expanded = np.full((out.shape[0], err.shape[1]), np.nan, dtype=out.dtype)
                        expanded[:, : out.shape[1]] = out
                        out = expanded
                    out[idx, : err.shape[1]] = err.astype(np.float32)
        finally:
            validator.close_if_needed(closeable)
    return out


def init_worker(
    opm_path: str,
    de441_path: str,
    no_crc: bool,
    nodes: int,
    node_grid: str,
    max_passes: int,
    radius: int,
    objective: str,
    min_improvement: float,
    tail_topk: int,
    refine_peaks: int,
    guard_grid: str,
    pmax_cap: float,
    error_metric: str,
) -> None:
    global _WORKER_OPM, _WORKER_PARAMS, _WORKER_CLOCK, _WORKER_WIDTHS, _WORKER_SPK, _WORKER_PROVIDER
    global _WORKER_NODES, _WORKER_NODE_GRID, _WORKER_MAX_PASSES, _WORKER_RADIUS, _WORKER_OBJECTIVE, _WORKER_MIN_IMPROVEMENT, _WORKER_TAIL_TOPK, _WORKER_REFINE_PEAKS, _WORKER_GUARD_GRID, _WORKER_PMAX_CAP, _WORKER_ERROR_METRIC
    proto.set_de441_path(Path(de441_path))
    moon_proto.set_de441_path(Path(de441_path))
    opm = validator.read_opm(Path(opm_path), check_crc=not no_crc)
    clock = validator.mercury_clock(opm) or validator.moon_clock(opm)
    params = validator.frame_params_for_segments(opm, clock) if opm.frame_coeffs is not None else None
    if params is None and opm.header.model_kind != validator.MODEL_RAW_XYZ_CHEB:
        raise RuntimeError("this optimizer expects raw Cheb or a framed reference-shape OPM")
    _WORKER_OPM = opm
    _WORKER_PARAMS = params
    _WORKER_CLOCK = clock
    _WORKER_WIDTHS = opm.widths
    _WORKER_SPK = SPK.open(de441_path)
    _WORKER_PROVIDER, _ = validator.truth_position_provider(_WORKER_SPK, opm)
    _WORKER_NODES = nodes
    _WORKER_NODE_GRID = node_grid
    _WORKER_MAX_PASSES = max_passes
    _WORKER_RADIUS = radius
    _WORKER_OBJECTIVE = objective
    _WORKER_MIN_IMPROVEMENT = min_improvement
    _WORKER_TAIL_TOPK = tail_topk
    _WORKER_REFINE_PEAKS = refine_peaks
    _WORKER_GUARD_GRID = guard_grid
    _WORKER_PMAX_CAP = pmax_cap
    _WORKER_ERROR_METRIC = error_metric


def optimize_one_process(args_tuple: tuple[int, np.ndarray]) -> tuple[int, np.ndarray, np.ndarray, list[object], float, float]:
    seg, before_err = args_tuple
    if _WORKER_OPM is None or _WORKER_WIDTHS is None or _WORKER_PROVIDER is None:
        raise RuntimeError("worker not initialized")
    jds = segment_nodes(_WORKER_OPM, seg, _WORKER_NODES, _WORKER_CLOCK, _WORKER_NODE_GRID)
    if _WORKER_REFINE_PEAKS > 0:
        jds = append_refined_peak_nodes_native(
            _WORKER_OPM,
            _WORKER_PARAMS,
            seg,
            jds,
            _WORKER_PROVIDER,
            _WORKER_REFINE_PEAKS,
            _WORKER_ERROR_METRIC,
        )
    truth = _WORKER_PROVIDER.position(jds)
    guard_jds = guard_segment_nodes(_WORKER_OPM, seg, _WORKER_NODES, _WORKER_CLOCK, _WORKER_GUARD_GRID)
    if len(guard_jds):
        guard_truth = _WORKER_PROVIDER.position(guard_jds)
    else:
        guard_jds = None
        guard_truth = None
    result = optimize_segment_native_guarded(
        _WORKER_OPM,
        _WORKER_PARAMS,
        seg,
        jds,
        truth,
        guard_jds,
        guard_truth,
        max_passes=_WORKER_MAX_PASSES,
        radius=_WORKER_RADIUS,
        objective_mode=_WORKER_OBJECTIVE,
        min_improvement=_WORKER_MIN_IMPROVEMENT,
        width_limits=_WORKER_WIDTHS,
        tail_topk=_WORKER_TAIL_TOPK,
        pmax_cap=_WORKER_PMAX_CAP,
        error_metric=_WORKER_ERROR_METRIC,
    )
    best_err = np.asarray(result["best_err"], dtype=np.float32)
    best_q = np.asarray(result["best_q"], dtype=np.int64)
    changes = list(result["changes"])
    return seg, best_err, best_q, changes, float(np.nanmax(before_err)), float(np.max(best_err))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("opm", type=Path)
    parser.add_argument("--de441", type=Path, required=True)
    parser.add_argument("--nodes-per-segment", type=int, default=32)
    parser.add_argument("--node-grid", choices=["cheb", "cheb-center", "cheb-center-uniform", "cheb-center-uniform-endpoints", "cheb-center-uniform-endpoint-band"], default="cheb-center-uniform-endpoints")
    parser.add_argument("--refine-peaks", type=int, default=3)
    parser.add_argument("--chunk-size", type=int, default=4096)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--progress-every", type=int, default=1000)
    parser.add_argument("--max-passes", type=int, default=4)
    parser.add_argument("--radius", type=int, default=1)
    parser.add_argument("--objective", choices=["topk_ranked", "pmax_ranked", "capped_lex_guarded_ceiling"], default="capped_lex_guarded_ceiling")
    parser.add_argument("--guard-grid", choices=["none", "shifted"], default="shifted")
    parser.add_argument("--accept-policy", choices=["budget", "pmax_first"], default="pmax_first")
    parser.add_argument("--tail-topk", type=int, default=4)
    parser.add_argument("--min-improvement", type=float, default=1e-12)
    parser.add_argument("--pmax-cap", type=float, default=7e-4)
    parser.add_argument("--p99-slack-abs", type=float, default=1e-8)
    parser.add_argument("--error-metric", choices=["angular", "km"], default="angular")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--no-crc", action="store_true")
    args = parser.parse_args()
    if args.objective == "capped_lex_guarded_ceiling":
        if args.guard_grid != "shifted":
            raise SystemExit("capped_lex_guarded_ceiling requires --guard-grid shifted")
        if args.accept_policy != "pmax_first":
            raise SystemExit("capped_lex_guarded_ceiling requires --accept-policy pmax_first")

    proto.set_de441_path(args.de441)
    moon_proto.set_de441_path(args.de441)
    opm = validator.read_opm(args.opm, check_crc=not args.no_crc)
    body_name = validator.body_name_from_id(opm.header.body_id)
    clock = validator.mercury_clock(opm) or validator.moon_clock(opm)
    params = validator.frame_params_for_segments(opm, clock) if opm.frame_coeffs is not None else None
    if params is None and opm.header.model_kind != validator.MODEL_RAW_XYZ_CHEB:
        raise SystemExit("this optimizer expects raw Cheb or a framed reference-shape OPM")
    coeffs = opm.qcoeffs.astype(np.float64) * opm.quant_steps[None, None, :]
    overhead = opm.header.file_size - opm.header.payload_size

    print(f"baseline: {args.opm}")
    print(f"body: {body_name}")
    print(f"source=existing {body_name} qcoeffs/quant_steps native metric={args.error_metric}")
    payload_size = payload_size_for_widths(opm.header.segment_count, opm.widths)
    print(
        f"size estimate: file={(payload_size + overhead) / 1024 / 1024:.3f} MiB "
        f"save={(opm.header.file_size - payload_size - overhead) / 1024 / 1024:.3f} MiB "
        f"width_sum={int(opm.widths.sum())} axis_bits={tuple(int(x) for x in opm.widths.sum(axis=1))}"
    )

    initial_by_segment = validate_native_by_segment(
        opm,
        coeffs,
        params,
        clock,
        args.de441,
        args.nodes_per_segment,
        args.chunk_size,
        args.node_grid,
        args.refine_peaks,
        args.error_metric,
    )
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
    qcoeffs_opt = opm.qcoeffs.copy()
    current_above = above_budget
    p99_floor = p99_budget
    accepted = rejected_nochange = rejected_budget = accepted_p99_slack = 0
    deferred_budget: list[tuple[int, np.ndarray, np.ndarray, list[object], float, float]] = []

    def assign_segment_errors(target: np.ndarray, seg: int, err: np.ndarray) -> None:
        target[seg] = np.nan
        target[seg, : len(err)] = err

    def accept(idx: int, result_tuple: tuple[int, np.ndarray, np.ndarray, list[object], float, float], *, defer_budget: bool = True) -> None:
        nonlocal accepted, rejected_nochange, rejected_budget, current_above, p99_floor, accepted_p99_slack
        seg, best_err, best_q, changes, before_max, after_max = result_tuple
        before_err = optimized_by_segment[seg].copy()
        if not changes or after_max >= before_max:
            rejected_nochange += 1
        elif args.accept_policy == "pmax_first":
            assign_segment_errors(optimized_by_segment, seg, best_err)
            qcoeffs_opt[seg] = best_q
            old_above = int(np.count_nonzero(before_err > p99_budget))
            new_above = int(np.count_nonzero(best_err > p99_budget))
            current_above = current_above - old_above + new_above
            accepted += 1
        else:
            old_above = int(np.count_nonzero(before_err > p99_budget))
            new_above = int(np.count_nonzero(best_err > p99_budget))
            trial_above = current_above - old_above + new_above
            accept_candidate = trial_above <= above_budget
            accepted_by_slack = False
            if not accept_candidate:
                trial_by_segment = optimized_by_segment.copy()
                assign_segment_errors(trial_by_segment, seg, best_err)
                trial_p99 = float(np.percentile(finite_errors(trial_by_segment), 99))
                if trial_p99 <= p99_floor + args.p99_slack_abs:
                    accept_candidate = True
                    accepted_by_slack = True
                    p99_floor = min(p99_floor, trial_p99)
            if accept_candidate:
                assign_segment_errors(optimized_by_segment, seg, best_err)
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
            top_n = min(100, len(cur_segmax))
            top_mean = float(np.mean(np.partition(cur_segmax, len(cur_segmax) - top_n)[-top_n:]))
            print(
                f"  processed {idx:,}/{len(order):,}; accepted={accepted:,}; "
                f"p99_slack={accepted_p99_slack:,}; current_max={np.max(cur_segmax):.9g}; "
                f"top100_mean={top_mean:.9g}; "
                f"count_above_budget={current_above:,}; p99_floor={p99_floor:.9g}",
                flush=True,
            )

    indexed_order = [(idx, int(seg)) for idx, seg in enumerate(order, 1)]
    if args.jobs <= 1:
        init_worker(
            str(args.opm),
            str(args.de441),
            args.no_crc,
            args.nodes_per_segment,
            args.node_grid,
            args.max_passes,
            args.radius,
            args.objective,
            args.min_improvement,
            args.tail_topk,
            args.refine_peaks,
            args.guard_grid,
            args.pmax_cap,
            args.error_metric,
        )
        for idx, seg in indexed_order:
            accept(idx, optimize_one_process((seg, optimized_by_segment[seg].copy())))
    else:
        pending: dict[int, tuple[int, np.ndarray, np.ndarray, list[object], float, float]] = {}
        next_to_accept = 1
        with ProcessPoolExecutor(
            max_workers=args.jobs,
            initializer=init_worker,
            initargs=(
                str(args.opm), str(args.de441), args.no_crc, args.nodes_per_segment, args.node_grid,
                args.max_passes, args.radius, args.objective, args.min_improvement, args.tail_topk,
                args.refine_peaks, args.guard_grid, args.pmax_cap, args.error_metric,
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
        for result_tuple in sorted(deferred_budget, key=lambda item: item[4] - item[5], reverse=True):
            seg, best_err, best_q, changes, before_max, after_max = result_tuple
            current_err = optimized_by_segment[seg]
            if not changes or float(np.max(best_err)) >= float(np.nanmax(current_err)):
                continue
            trial_by_segment = optimized_by_segment.copy()
            assign_segment_errors(trial_by_segment, seg, best_err)
            trial_p99 = float(np.percentile(finite_errors(trial_by_segment), 99))
            if trial_p99 <= p99_floor + args.p99_slack_abs:
                old_above = int(np.count_nonzero(current_err > p99_budget))
                new_above = int(np.count_nonzero(best_err > p99_budget))
                assign_segment_errors(optimized_by_segment, seg, best_err)
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
    opt_payload = payload_size_for_widths(opm.header.segment_count, opt_widths)
    print()
    print(f"accepted={accepted:,}; rejected_nochange={rejected_nochange:,}; rejected_budget={rejected_budget:,}")
    print(f"optimized global: {percentile_text(optimized_errors)}")
    print(f"optimized segment max: {summarize_segmax(opt_segmax)}")
    print(
        f"optimized size estimate: file={(opt_payload + overhead) / 1024 / 1024:.3f} MiB "
        f"save={(opm.header.file_size - opt_payload - overhead) / 1024 / 1024:.3f} MiB "
        f"width_sum={int(opt_widths.sum())} axis_bits={tuple(int(x) for x in opt_widths.sum(axis=1))}"
    )

    if args.output is not None:
        opt_widths_packed, payload = generator.pack_qcoeffs(qcoeffs_opt)
        if not np.array_equal(opt_widths_packed, opt_widths):
            raise SystemExit("packed width table does not match optimized width estimate")
        packed = generator.PackedBody(
            cfg=config_from_opm(opm, None, None),
            boundaries=boundaries_from_opm(opm, clock),
            quant_steps=opm.quant_steps,
            widths=opt_widths_packed,
            qcoeffs=qcoeffs_opt,
            payload=payload,
            model_table=model_table_from_opm(opm),
            clock_table=opm.clock_table,
            p50=float(np.percentile(optimized_errors, 50)),
            p95=float(np.percentile(optimized_errors, 95)),
            p99=float(np.percentile(optimized_errors, 99)),
            max_err=float(np.max(optimized_errors)),
        )
        size = generator.write_opm_file(args.output, packed, opm.header.source_start_jd, opm.header.source_end_jd, opm.header.coverage_start_jd, opm.header.coverage_span_days)
        print(f"wrote {args.output} ({size / 1024 / 1024:.3f} MiB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
