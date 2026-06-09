#!/usr/bin/env python3
"""Optimize an SSB-centered body against heliocentric error using a fitted Sun OPM anchor.

This keeps the body's quantization/width table unchanged, but adjusts per-segment
integer coefficients by small +/- steps when doing so improves the tail of:

    (Body_opm - Sun_opm) vs (Body_DE441 - Sun_DE441)

The Sun term is reconstructed from the supplied Sun OPM, so the fitted/quantized
Sun anchor error is included in the metric used to polish SSB-centered bodies.
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

import opm_demo.orbit_model as proto  # noqa: E402
from opm_demo import generator, validator  # noqa: E402
from opm_demo.body_configs import CONFIGS  # noqa: E402
from optimize_opm_polish_common import finite_errors, payload_size_for_widths, percentile_text, zigzag_widths  # noqa: E402
from optimize_opm_global_tail import boundaries_from_opm, summarize_segmax  # noqa: E402
from optimize_opm_segment_rounding import q_value_width, topk_mean, is_better_candidate  # noqa: E402

AXIS_COUNT = 3

_WORKER_BODY: validator.OpmFile | None = None
_WORKER_SUN: validator.OpmFile | None = None
_WORKER_PARAMS: np.ndarray | None = None
_WORKER_CLOCK: object | None = None
_WORKER_WIDTHS: np.ndarray | None = None
_WORKER_BODY_NAME = ""
_WORKER_SPK: SPK | None = None
_WORKER_BODY_PROVIDER: validator.BaryProvider | None = None
_WORKER_SUN_PROVIDER: validator.BaryProvider | None = None
_WORKER_NODES = 32
_WORKER_NODE_GRID = "cheb"
_WORKER_MAX_PASSES = 4
_WORKER_RADIUS = 1
_WORKER_OBJECTIVE = "topk_ranked"
_WORKER_MIN_IMPROVEMENT = 1e-12
_WORKER_TAIL_TOPK = 4
_WORKER_REFINE_PEAKS = 0
_WORKER_GUARD_GRID = "none"
_WORKER_PMAX_CAP = 7e-4


def body_config_from_opm(opm: validator.OpmFile) -> object:
    body = validator.body_name_from_id(opm.header.body_id)
    base_cfg = CONFIGS[body]
    shape_degree = None if opm.header.reference_shape_degree == 255 else int(opm.header.reference_shape_degree)
    return replace(
        base_cfg,
        clock=replace(
            base_cfg.clock,
            period_days=float(opm.header.period_days) if opm.header.period_days else base_cfg.clock.period_days,
            phase_start_jd=float(opm.header.phase_start_jd) if opm.header.phase_start_jd else base_cfg.clock.phase_start_jd,
        ),
        residual_degree=int(opm.header.residual_degree),
        shape_degree=shape_degree,
        edge_margin_days=float(opm.header.edge_margin_days),
        apsis_step_days=float(opm.header.event_search_step_days),
        segment_domain_expansion_fraction=float(opm.header.expansion),
    )


def sun_position_from_opm(sun: validator.OpmFile, jds: np.ndarray) -> np.ndarray:
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
            in_seg = mask
        tau = validator.normalize_expanded(jds[in_seg], a, b, h.expansion)
        out[in_seg] = np.column_stack([proto.cheb_eval(coeffs[int(si), axis], tau) for axis in range(AXIS_COUNT)])
    return out


def reconstruct_from_q(
    opm: validator.OpmFile,
    params: np.ndarray,
    segment: int,
    qcoeffs: np.ndarray,
    jds: np.ndarray,
    basis: np.ndarray | None = None,
    shape_x: np.ndarray | None = None,
    shape_y: np.ndarray | None = None,
) -> np.ndarray:
    h = opm.header
    clock = validator.mercury_clock(opm) or validator.moon_clock(opm)
    a, b = validator.segment_bounds(h, segment, clock)
    tau = validator.normalize_expanded(jds, a, b, h.expansion)
    if basis is None:
        basis = np.polynomial.chebyshev.chebvander(tau, h.residual_degree)
    if h.model_kind == validator.MODEL_RAW_XYZ_CHEB:
        coeffs = qcoeffs.astype(np.float64) * opm.quant_steps[None, :]
        return np.column_stack([basis @ coeffs[axis] for axis in range(AXIS_COUNT)])
    if h.model_kind not in {validator.MODEL_FIXED_FRAME_SHAPE, validator.MODEL_MEAN_APSIS_FRAME_SHAPE, validator.MODEL_MEAN_LUNAR_APSIS_FRAME_SHAPE}:
        raise ValueError(f"unsupported model_kind={h.model_kind}")
    if shape_x is None:
        if opm.shape_x is None:
            raise ValueError(f"{opm.path}: missing shape_x")
        shape_x = proto.cheb_eval(opm.shape_x, tau)
    if shape_y is None:
        if opm.shape_y is None:
            raise ValueError(f"{opm.path}: missing shape_y")
        shape_y = proto.cheb_eval(opm.shape_y, tau)
    coeffs = qcoeffs.astype(np.float64) * opm.quant_steps[None, :]
    aligned = np.empty((1, len(jds), AXIS_COUNT), dtype=np.float64)
    aligned[0, :, 0] = shape_x + basis @ coeffs[0]
    aligned[0, :, 1] = shape_y + basis @ coeffs[1]
    aligned[0, :, 2] = basis @ coeffs[2]
    return validator.unalign_positions_batched(aligned, params[segment : segment + 1])[0]


def composite_errors(
    body: validator.OpmFile,
    sun: validator.OpmFile,
    params: np.ndarray,
    segment: int,
    qcoeffs: np.ndarray,
    jds: np.ndarray,
    truth_helio: np.ndarray,
    basis: np.ndarray,
    shape_x: np.ndarray | None,
    shape_y: np.ndarray | None,
    sun_opm_pos: np.ndarray,
) -> np.ndarray:
    body_pos = reconstruct_from_q(body, params, segment, qcoeffs, jds, basis, shape_x, shape_y)
    candidate = body_pos - sun_opm_pos
    return proto.angular_errors_arcsec(truth_helio, candidate)


def objective_score(err: np.ndarray, tail_topk: int) -> tuple[float, float, float]:
    return (float(np.max(err)), topk_mean(err, tail_topk), float(np.percentile(err, 99)))


def lex_better(trial_err: np.ndarray, best_err: np.ndarray, min_improvement: float, tail_topk: int) -> bool:
    trial = objective_score(trial_err, tail_topk)
    best = objective_score(best_err, tail_topk)
    if trial[0] + min_improvement < best[0]:
        return True
    if trial[0] <= best[0] + min_improvement and trial[1] + min_improvement < best[1]:
        return True
    if trial[0] <= best[0] + min_improvement and trial[1] <= best[1] + min_improvement and trial[2] + min_improvement < best[2]:
        return True
    return False


def capped_lex_better(trial_err: np.ndarray, best_err: np.ndarray, min_improvement: float, tail_topk: int, pmax_cap: float) -> bool:
    trial = objective_score(trial_err, tail_topk)
    best = objective_score(best_err, tail_topk)
    if best[0] > pmax_cap:
        return lex_better(trial_err, best_err, min_improvement, tail_topk)
    if trial[0] > pmax_cap + min_improvement:
        return False
    if trial[1] + min_improvement < best[1]:
        return True
    if trial[1] <= best[1] + min_improvement and trial[0] + min_improvement < best[0]:
        return True
    if trial[1] <= best[1] + min_improvement and trial[0] <= best[0] + min_improvement and trial[2] + min_improvement < best[2]:
        return True
    return False


def optimize_segment_composite(
    body: validator.OpmFile,
    sun: validator.OpmFile,
    params: np.ndarray,
    segment: int,
    jds: np.ndarray,
    truth_helio: np.ndarray,
    sun_opm_pos: np.ndarray,
    guard_jds: np.ndarray | None = None,
    guard_truth_helio: np.ndarray | None = None,
    guard_sun_opm_pos: np.ndarray | None = None,
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
) -> dict[str, object]:
    h = body.header
    a, b = validator.segment_bounds(h, segment, validator.mercury_clock(body) or validator.moon_clock(body))
    tau = validator.normalize_expanded(jds, a, b, h.expansion)
    basis = np.polynomial.chebyshev.chebvander(tau, h.residual_degree)
    shape_x = proto.cheb_eval(body.shape_x, tau) if body.shape_x is not None else None
    shape_y = proto.cheb_eval(body.shape_y, tau) if body.shape_y is not None else None

    if guard_jds is not None:
        if guard_truth_helio is None or guard_sun_opm_pos is None:
            raise ValueError("guard truth and Sun positions are required when guard_jds is supplied")
        guard_tau = validator.normalize_expanded(guard_jds, a, b, h.expansion)
        guard_basis = np.polynomial.chebyshev.chebvander(guard_tau, h.residual_degree)
        guard_shape_x = proto.cheb_eval(body.shape_x, guard_tau) if body.shape_x is not None else None
        guard_shape_y = proto.cheb_eval(body.shape_y, guard_tau) if body.shape_y is not None else None
    else:
        guard_basis = None
        guard_shape_x = None
        guard_shape_y = None

    q = body.qcoeffs[segment].copy()
    initial_q = q.copy()
    err = composite_errors(body, sun, params, segment, q, jds, truth_helio, basis, shape_x, shape_y, sun_opm_pos)
    if guard_jds is not None and guard_basis is not None and guard_truth_helio is not None and guard_sun_opm_pos is not None:
        guard_err = composite_errors(body, sun, params, segment, q, guard_jds, guard_truth_helio, guard_basis, guard_shape_x, guard_shape_y, guard_sun_opm_pos)
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
                    trial_err = composite_errors(body, sun, params, segment, q, jds, truth_helio, basis, shape_x, shape_y, sun_opm_pos)
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
                trial_err = composite_errors(body, sun, params, segment, q, jds, truth_helio, basis, shape_x, shape_y, sun_opm_pos)
                trial_guard_err = None
                if objective_mode == "capped_lex_guarded_ceiling" and guard_jds is not None and guard_basis is not None and guard_truth_helio is not None and guard_sun_opm_pos is not None:
                    trial_guard_err = composite_errors(body, sun, params, segment, q, guard_jds, guard_truth_helio, guard_basis, guard_shape_x, guard_shape_y, guard_sun_opm_pos)

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
                    better = is_better_candidate(trial_err, best_err, "topk_ranked", min_improvement, tail_topk)
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


def initial_composite_errors_at_jds(
    body: validator.OpmFile,
    sun: validator.OpmFile,
    params: np.ndarray,
    segment: int,
    jds: np.ndarray,
    body_provider: validator.BaryProvider,
    sun_provider: validator.BaryProvider,
) -> np.ndarray:
    h = body.header
    a, b = validator.segment_bounds(h, segment, validator.mercury_clock(body) or validator.moon_clock(body))
    tau = validator.normalize_expanded(jds, a, b, h.expansion)
    basis = np.polynomial.chebyshev.chebvander(tau, h.residual_degree)
    shape_x = proto.cheb_eval(body.shape_x, tau) if body.shape_x is not None else None
    shape_y = proto.cheb_eval(body.shape_y, tau) if body.shape_y is not None else None
    truth = body_provider.position(jds) - sun_provider.position(jds)
    sun_opm_pos = sun_position_from_opm(sun, jds)
    return composite_errors(body, sun, params, segment, body.qcoeffs[segment], jds, truth, basis, shape_x, shape_y, sun_opm_pos)


def golden_section_max(fn: object, lo: float, hi: float, iterations: int = 18) -> float:
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return 0.5 * (lo + hi)
    invphi = (np.sqrt(5.0) - 1.0) / 2.0
    invphi2 = (3.0 - np.sqrt(5.0)) / 2.0
    x1 = lo + invphi2 * (hi - lo)
    x2 = lo + invphi * (hi - lo)
    f1 = float(fn(x1))
    f2 = float(fn(x2))
    for _ in range(iterations):
        if f1 < f2:
            lo = x1
            x1 = x2
            f1 = f2
            x2 = lo + invphi * (hi - lo)
            f2 = float(fn(x2))
        else:
            hi = x2
            x2 = x1
            f2 = f1
            x1 = lo + invphi2 * (hi - lo)
            f1 = float(fn(x1))
    return 0.5 * (lo + hi)


def append_refined_peak_nodes(
    body: validator.OpmFile,
    sun: validator.OpmFile,
    params: np.ndarray,
    segment: int,
    base_jds: np.ndarray,
    body_provider: validator.BaryProvider,
    sun_provider: validator.BaryProvider,
    refine_peaks: int,
) -> np.ndarray:
    if refine_peaks <= 0 or len(base_jds) < 3:
        return base_jds
    order = np.argsort(base_jds)
    jds = np.asarray(base_jds[order], dtype=np.float64)
    errs = initial_composite_errors_at_jds(body, sun, params, segment, jds, body_provider, sun_provider)
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
            return float(initial_composite_errors_at_jds(
                body,
                sun,
                params,
                segment,
                np.asarray([x], dtype=np.float64),
                body_provider,
                sun_provider,
            )[0])

        refined.append(golden_section_max(err_at, lo, hi))
    while len(refined) < refine_peaks:
        refined.append(float(jds[int(np.argmax(errs))]))
    return np.sort(np.concatenate([jds, np.asarray(refined, dtype=np.float64)]))


def center_dense_nodes(a: float, b: float, n: int) -> np.ndarray:
    k = np.arange(n, dtype=np.float64)
    u = -1.0 + 2.0 * (k + 0.5) / float(n)
    tau = np.sign(u) * np.abs(u) ** 2
    return 0.5 * (a + b) + 0.5 * (b - a) * tau


def uniform_nodes(a: float, b: float, n: int) -> np.ndarray:
    k = np.arange(n, dtype=np.float64)
    tau = -1.0 + 2.0 * (k + 0.5) / float(n)
    return 0.5 * (a + b) + 0.5 * (b - a) * tau


def endpoint_nodes(a: float, b: float) -> np.ndarray:
    return np.asarray([a, b], dtype=np.float64)


def shifted_uniform_nodes(a: float, b: float, n: int, offset: float = 0.25) -> np.ndarray:
    k = np.arange(n, dtype=np.float64)
    frac = (k + offset) / float(n)
    return a + (b - a) * frac


def shifted_center_dense_nodes(a: float, b: float, n: int, offset: float = 0.25) -> np.ndarray:
    k = np.arange(n, dtype=np.float64)
    u = -1.0 + 2.0 * ((k + offset) / float(n))
    tau = np.sign(u) * np.abs(u) ** 2
    return 0.5 * (a + b) + 0.5 * (b - a) * tau


def endpoint_near_nodes(a: float, b: float) -> np.ndarray:
    width = b - a
    frac = np.asarray([1.0 / 1024.0, 1.0 / 256.0, 1.0 / 64.0], dtype=np.float64)
    nodes = np.concatenate([a + width * frac, b - width * frac])
    return nodes


def endpoint_band_nodes(a: float, b: float) -> np.ndarray:
    width = b - a
    frac = np.asarray([1.0 / 2048.0, 1.0 / 1024.0, 1.0 / 512.0, 1.0 / 256.0, 1.0 / 128.0, 1.0 / 64.0, 1.0 / 32.0], dtype=np.float64)
    nodes = np.concatenate([a + width * frac, b - width * frac])
    return nodes


def shifted_endpoint_band_nodes(a: float, b: float) -> np.ndarray:
    width = b - a
    frac = np.asarray([1.5 / 2048.0, 1.5 / 1024.0, 1.5 / 512.0, 1.5 / 256.0, 1.5 / 128.0, 1.5 / 64.0], dtype=np.float64)
    nodes = np.concatenate([a + width * frac, b - width * frac])
    return nodes


def segment_nodes(opm: validator.OpmFile, segment: int, nodes_per_segment: int, clock: object | None, node_grid: str = "cheb") -> np.ndarray:
    h = opm.header
    a, b = validator.segment_bounds(h, segment, clock)
    lo = max(a, h.coverage_start_jd)
    hi = min(b, h.coverage_start_jd + h.coverage_span_days)
    if node_grid == "cheb":
        return proto.cheb_nodes(lo, hi, nodes_per_segment)
    if node_grid == "cheb-center":
        nodes = np.concatenate([
            proto.cheb_nodes(lo, hi, nodes_per_segment),
            center_dense_nodes(lo, hi, nodes_per_segment),
        ])
        return np.unique(np.sort(nodes))
    if node_grid == "cheb-center-uniform":
        nodes = np.concatenate([
            proto.cheb_nodes(lo, hi, nodes_per_segment),
            center_dense_nodes(lo, hi, nodes_per_segment),
            uniform_nodes(lo, hi, nodes_per_segment * 2),
        ])
        return np.unique(np.sort(nodes))
    if node_grid == "cheb-center-uniform-endpoints":
        nodes = np.concatenate([
            proto.cheb_nodes(lo, hi, nodes_per_segment),
            center_dense_nodes(lo, hi, nodes_per_segment),
            uniform_nodes(lo, hi, nodes_per_segment * 2),
            endpoint_nodes(lo, hi),
        ])
        return np.unique(np.sort(nodes))
    if node_grid == "cheb-center-uniform-endpoint-band":
        nodes = np.concatenate([
            proto.cheb_nodes(lo, hi, nodes_per_segment),
            center_dense_nodes(lo, hi, nodes_per_segment),
            uniform_nodes(lo, hi, nodes_per_segment * 2),
            endpoint_nodes(lo, hi),
            endpoint_band_nodes(lo, hi),
        ])
        return np.unique(np.sort(nodes))
    raise ValueError(f"unknown node grid {node_grid}")


def guard_segment_nodes(opm: validator.OpmFile, segment: int, nodes_per_segment: int, clock: object | None, guard_grid: str) -> np.ndarray:
    if guard_grid == "none":
        return np.empty((0,), dtype=np.float64)
    h = opm.header
    a, b = validator.segment_bounds(h, segment, clock)
    lo = max(a, h.coverage_start_jd)
    hi = min(b, h.coverage_start_jd + h.coverage_span_days)
    if hi <= lo:
        return np.empty((0,), dtype=np.float64)
    if guard_grid == "shifted":
        nodes = np.concatenate([
            shifted_center_dense_nodes(lo, hi, nodes_per_segment, 0.25),
            shifted_center_dense_nodes(lo, hi, nodes_per_segment, 0.75),
            shifted_uniform_nodes(lo, hi, nodes_per_segment * 2, 0.25),
            shifted_uniform_nodes(lo, hi, nodes_per_segment * 2, 0.75),
            endpoint_near_nodes(lo, hi),
            endpoint_band_nodes(lo, hi),
            shifted_endpoint_band_nodes(lo, hi),
        ])
        nodes = nodes[(nodes >= lo) & (nodes <= hi)]
        return np.unique(np.sort(nodes))
    raise ValueError(f"unknown guard grid {guard_grid}")


def validate_composite_by_segment(
    body: validator.OpmFile,
    sun: validator.OpmFile,
    body_name: str,
    coeffs: np.ndarray,
    params: np.ndarray,
    clock: object | None,
    de441_path: Path,
    nodes_per_segment: int,
    chunk_size: int,
    node_grid: str = "cheb",
    refine_peaks: int = 0,
) -> np.ndarray:
    base_nodes = segment_nodes(body, 0, nodes_per_segment, clock, node_grid)
    nodes_per_row = len(base_nodes) + max(0, refine_peaks)
    out = np.full((body.header.segment_count, nodes_per_row), np.nan, dtype=np.float32)
    candidate_body = replace(body, qcoeffs=np.rint(coeffs / body.quant_steps[None, None, :]).astype(np.int64))
    with SPK.open(str(de441_path)) as spk:
        body_provider = validator.BaryProvider(spk, validator.SPK_TARGET_IDS[body_name])
        sun_provider = validator.BaryProvider(spk, validator.SPK_TARGET_IDS["sun"])
        for start in range(0, body.header.segment_count, chunk_size):
            stop = min(start + chunk_size, body.header.segment_count)
            indices: list[int] = []
            bounds: list[tuple[float, float]] = []
            nodes_parts: list[np.ndarray] = []
            for si in range(start, stop):
                jds = segment_nodes(body, si, nodes_per_segment, clock, node_grid)
                if len(jds) == 0:
                    continue
                if refine_peaks > 0:
                    jds = append_refined_peak_nodes(body, sun, params, si, jds, body_provider, sun_provider, refine_peaks)
                a, b = validator.segment_bounds(body.header, si, clock)
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
                ea = a - body.header.expansion * width
                eb = b + body.header.expansion * width
                tau = (2.0 * jds - ea[:, None] - eb[:, None]) / (eb - ea)[:, None]
                body_pos = validator.reconstruct_segment_nodes(candidate_body, idx, tau, coeffs, params)
                flat_jds = jds.reshape(-1)
                sun_opm = sun_position_from_opm(sun, flat_jds).reshape(body_pos.shape)
                truth = (body_provider.position(flat_jds) - sun_provider.position(flat_jds)).reshape(body_pos.shape)
                cand = body_pos - sun_opm
                err = proto.angular_errors_arcsec(truth.reshape((-1, AXIS_COUNT)), cand.reshape((-1, AXIS_COUNT))).reshape(jds.shape)
                if err.shape[1] > out.shape[1]:
                    expanded = np.full((out.shape[0], err.shape[1]), np.nan, dtype=out.dtype)
                    expanded[:, : out.shape[1]] = out
                    out = expanded
                out[idx, : err.shape[1]] = err.astype(np.float32)
    return out


def init_worker(
    body_path: str,
    sun_path: str,
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
) -> None:
    global _WORKER_BODY, _WORKER_SUN, _WORKER_PARAMS, _WORKER_CLOCK, _WORKER_WIDTHS, _WORKER_BODY_NAME, _WORKER_SPK, _WORKER_BODY_PROVIDER, _WORKER_SUN_PROVIDER
    global _WORKER_NODES, _WORKER_NODE_GRID, _WORKER_MAX_PASSES, _WORKER_RADIUS, _WORKER_OBJECTIVE, _WORKER_MIN_IMPROVEMENT, _WORKER_TAIL_TOPK, _WORKER_REFINE_PEAKS, _WORKER_GUARD_GRID, _WORKER_PMAX_CAP
    proto.set_de441_path(Path(de441_path))
    body = validator.read_opm(Path(body_path), check_crc=not no_crc)
    sun = validator.read_opm(Path(sun_path), check_crc=not no_crc)
    body_name = validator.body_name_from_id(body.header.body_id)
    if body.header.storage_vector_id != validator.STORAGE_SSB_TO_BODY:
        raise RuntimeError(f"{body.path}: expected SSB-centered body, got storage_vector_id={body.header.storage_vector_id}")
    clock = validator.mercury_clock(body) or validator.moon_clock(body)
    params = validator.frame_params_for_segments(body, clock) if body.frame_coeffs is not None else None
    if params is None and body.header.model_kind != validator.MODEL_RAW_XYZ_CHEB:
        raise RuntimeError("this optimizer expects raw Cheb or a framed reference-shape OPM")
    _WORKER_BODY = body
    _WORKER_SUN = sun
    _WORKER_PARAMS = params
    _WORKER_CLOCK = clock
    _WORKER_WIDTHS = body.widths
    _WORKER_BODY_NAME = body_name
    _WORKER_SPK = SPK.open(de441_path)
    _WORKER_BODY_PROVIDER = validator.BaryProvider(_WORKER_SPK, validator.SPK_TARGET_IDS[body_name])
    _WORKER_SUN_PROVIDER = validator.BaryProvider(_WORKER_SPK, validator.SPK_TARGET_IDS["sun"])
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


def optimize_one_process(args_tuple: tuple[int, np.ndarray]) -> tuple[int, np.ndarray, np.ndarray, list[object], float, float]:
    seg, before_err = args_tuple
    if _WORKER_BODY is None or _WORKER_SUN is None or _WORKER_PARAMS is None or _WORKER_WIDTHS is None or _WORKER_BODY_PROVIDER is None or _WORKER_SUN_PROVIDER is None:
        raise RuntimeError("worker not initialized")
    jds = segment_nodes(_WORKER_BODY, seg, _WORKER_NODES, _WORKER_CLOCK, _WORKER_NODE_GRID)
    if _WORKER_REFINE_PEAKS > 0:
        jds = append_refined_peak_nodes(
            _WORKER_BODY,
            _WORKER_SUN,
            _WORKER_PARAMS,
            seg,
            jds,
            _WORKER_BODY_PROVIDER,
            _WORKER_SUN_PROVIDER,
            _WORKER_REFINE_PEAKS,
        )
    truth = _WORKER_BODY_PROVIDER.position(jds) - _WORKER_SUN_PROVIDER.position(jds)
    sun_opm_pos = sun_position_from_opm(_WORKER_SUN, jds)
    guard_jds = guard_segment_nodes(_WORKER_BODY, seg, _WORKER_NODES, _WORKER_CLOCK, _WORKER_GUARD_GRID)
    if len(guard_jds):
        guard_truth = _WORKER_BODY_PROVIDER.position(guard_jds) - _WORKER_SUN_PROVIDER.position(guard_jds)
        guard_sun_opm_pos = sun_position_from_opm(_WORKER_SUN, guard_jds)
    else:
        guard_jds = None
        guard_truth = None
        guard_sun_opm_pos = None
    result = optimize_segment_composite(
        _WORKER_BODY,
        _WORKER_SUN,
        _WORKER_PARAMS,
        seg,
        jds,
        truth,
        sun_opm_pos,
        guard_jds,
        guard_truth,
        guard_sun_opm_pos,
        max_passes=_WORKER_MAX_PASSES,
        radius=_WORKER_RADIUS,
        objective_mode=_WORKER_OBJECTIVE,
        min_improvement=_WORKER_MIN_IMPROVEMENT,
        width_limits=_WORKER_WIDTHS,
        tail_topk=_WORKER_TAIL_TOPK,
        pmax_cap=_WORKER_PMAX_CAP,
    )
    best_err = np.asarray(result["best_err"], dtype=np.float32)
    best_q = np.asarray(result["best_q"], dtype=np.int64)
    changes = list(result["changes"])
    return seg, best_err, best_q, changes, float(np.nanmax(before_err)), float(np.max(best_err))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("body_opm", type=Path)
    parser.add_argument("--sun-opm", type=Path, required=True)
    parser.add_argument("--de441", type=Path, required=True)
    parser.add_argument("--nodes-per-segment", type=int, default=32)
    parser.add_argument("--node-grid", choices=["cheb", "cheb-center", "cheb-center-uniform", "cheb-center-uniform-endpoints", "cheb-center-uniform-endpoint-band"], default="cheb-center-uniform-endpoints", help="pmax scoring grid; cheb-center adds center-dense tau nodes, cheb-center-uniform also adds 2x uniform tau nodes, -endpoints includes clipped segment endpoints, and -endpoint-band adds dense near-endpoint nodes")
    parser.add_argument("--refine-peaks", type=int, default=3, help="append this many locally refined peak JDs per segment to the pmax scoring grid")
    parser.add_argument("--chunk-size", type=int, default=4096)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--progress-every", type=int, default=1000)
    parser.add_argument("--max-passes", type=int, default=4)
    parser.add_argument("--radius", type=int, default=1)
    parser.add_argument("--objective", choices=["topk_ranked", "pmax_ranked", "capped_lex_guarded_ceiling"], default="capped_lex_guarded_ceiling")
    parser.add_argument("--guard-grid", choices=["none", "shifted"], default="shifted", help="optional shifted validation grid used by capped_lex_guarded_ceiling")
    parser.add_argument("--accept-policy", choices=["budget", "pmax_first"], default="pmax_first", help="global segment acceptance policy; pmax_first accepts any no-size-increase segment pmax reduction")
    parser.add_argument("--tail-topk", type=int, default=4)
    parser.add_argument("--min-improvement", type=float, default=1e-12)
    parser.add_argument("--pmax-cap", type=float, default=7e-4, help="pmax ceiling for capped_lex_guarded_ceiling before switching back to topK-first")
    parser.add_argument("--p99-slack-abs", type=float, default=1e-8, help="accept old-budget failures if true global p99 stays within p99_floor + this slack")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--no-crc", action="store_true")
    args = parser.parse_args()
    if args.objective == "capped_lex_guarded_ceiling":
        if args.guard_grid != "shifted":
            raise SystemExit("capped_lex_guarded_ceiling requires --guard-grid shifted")
        if args.accept_policy != "pmax_first":
            raise SystemExit("capped_lex_guarded_ceiling requires --accept-policy pmax_first")

    body = validator.read_opm(args.body_opm, check_crc=not args.no_crc)
    sun = validator.read_opm(args.sun_opm, check_crc=not args.no_crc)
    body_name = validator.body_name_from_id(body.header.body_id)
    if body.header.storage_vector_id != validator.STORAGE_SSB_TO_BODY:
        raise SystemExit(f"{args.body_opm}: expected SSB-centered body, got storage_vector_id={body.header.storage_vector_id}")
    clock = validator.mercury_clock(body) or validator.moon_clock(body)
    params = validator.frame_params_for_segments(body, clock) if body.frame_coeffs is not None else None
    if params is None and body.header.model_kind != validator.MODEL_RAW_XYZ_CHEB:
        raise SystemExit("this optimizer expects raw Cheb or a framed reference-shape OPM")
    coeffs = body.qcoeffs.astype(np.float64) * body.quant_steps[None, None, :]
    overhead = body.header.file_size - body.header.payload_size

    print(f"baseline: {args.body_opm}")
    print(f"body: {body_name}")
    print(f"sun anchor: {args.sun_opm}")
    print(f"source=existing {body_name} qcoeffs/quant_steps composite=({body_name}_opm - Sun_opm) vs ({body_name}_DE441 - Sun_DE441)")
    payload_size = payload_size_for_widths(body.header.segment_count, body.widths)
    print(
        f"size estimate: file={(payload_size + overhead) / 1024 / 1024:.3f} MiB "
        f"save={(body.header.file_size - payload_size - overhead) / 1024 / 1024:.3f} MiB "
        f"width_sum={int(body.widths.sum())} axis_bits={tuple(int(x) for x in body.widths.sum(axis=1))}"
    )

    initial_by_segment = validate_composite_by_segment(
        body,
        sun,
        body_name,
        coeffs,
        params,
        clock,
        args.de441,
        args.nodes_per_segment,
        args.chunk_size,
        args.node_grid,
        args.refine_peaks,
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
    qcoeffs_opt = body.qcoeffs.copy()
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
            str(args.body_opm),
            str(args.sun_opm),
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
                str(args.body_opm), str(args.sun_opm), str(args.de441), args.no_crc, args.nodes_per_segment, args.node_grid,
                args.max_passes, args.radius, args.objective, args.min_improvement, args.tail_topk, args.refine_peaks, args.guard_grid, args.pmax_cap,
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
    opt_payload = payload_size_for_widths(body.header.segment_count, opt_widths)
    print()
    print(f"accepted={accepted:,}; rejected_nochange={rejected_nochange:,}; rejected_budget={rejected_budget:,}")
    print(f"optimized global: {percentile_text(optimized_errors)}")
    print(f"optimized segment max: {summarize_segmax(opt_segmax)}")
    print(
        f"optimized size estimate: file={(opt_payload + overhead) / 1024 / 1024:.3f} MiB "
        f"save={(body.header.file_size - opt_payload - overhead) / 1024 / 1024:.3f} MiB "
        f"width_sum={int(opt_widths.sum())} axis_bits={tuple(int(x) for x in opt_widths.sum(axis=1))}"
    )

    if args.output is not None:
        opt_widths_packed, payload = generator.pack_qcoeffs(qcoeffs_opt)
        if not np.array_equal(opt_widths_packed, opt_widths):
            raise SystemExit("packed width table does not match optimized width estimate")
        if body.shape_x is None and body.shape_y is None and body.frame_coeffs is None:
            model_table = b""
        else:
            model_table = generator.pack_model_table(body.shape_x, body.shape_y, body.frame_coeffs)
        packed = generator.PackedBody(
            cfg=body_config_from_opm(body),
            boundaries=boundaries_from_opm(body, clock),
            quant_steps=body.quant_steps,
            widths=opt_widths_packed,
            qcoeffs=qcoeffs_opt,
            payload=payload,
            model_table=model_table,
            clock_table=body.clock_table,
            p50=float(np.percentile(optimized_errors, 50)),
            p95=float(np.percentile(optimized_errors, 95)),
            p99=float(np.percentile(optimized_errors, 99)),
            max_err=float(np.max(optimized_errors)),
        )
        size = generator.write_opm_file(args.output, packed, body.header.source_start_jd, body.header.source_end_jd, body.header.coverage_start_jd, body.header.coverage_span_days)
        print(f"wrote {args.output} ({size / 1024 / 1024:.3f} MiB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
