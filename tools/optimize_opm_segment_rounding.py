#!/usr/bin/env python3
"""Prototype quantization-aware rounding for selected OPM segments.

The normal writer rounds each coefficient independently.  This diagnostic keeps
format/reader/quantization unchanged, but locally adjusts integer coefficients by
small +/- steps when doing so lowers validation-node max error for a segment.
It is intentionally a prototype: it only optimizes selected segments in memory
and reports the potential max-error reduction.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from jplephem.spk import SPK

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import opm_demo.moon_model as moon_proto  # noqa: E402
import opm_demo.orbit_model as proto  # noqa: E402
from opm_demo import validator  # noqa: E402

JD_J2000 = 2451545.0
DAYS_PER_JULIAN_YEAR = 365.25
AXIS_COUNT = 3


def parse_csv_ints(text: str) -> list[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def cheb_nodes(a: float, b: float, n: int) -> np.ndarray:
    k = np.arange(n)
    x = np.cos(np.pi * (k + 0.5) / n)
    return np.sort(0.5 * (a + b) + 0.5 * (b - a) * x)


def segment_eval_nodes(opm: validator.OpmFile, segment: int, nodes_per_segment: int, clock: object | None) -> np.ndarray:
    h = opm.header
    coverage_end = h.coverage_start_jd + h.coverage_span_days
    a, b = validator.segment_bounds(h, segment, clock)
    lo = max(a, h.coverage_start_jd)
    hi = min(b, coverage_end)
    return cheb_nodes(lo, hi, nodes_per_segment)


def normalize_expanded(jd: np.ndarray, a: float, b: float, expansion: float) -> np.ndarray:
    width = b - a
    ea = a - expansion * width
    eb = b + expansion * width
    return (2.0 * jd - ea - eb) / (eb - ea)


def reconstruct_from_q(
    opm: validator.OpmFile,
    params: np.ndarray | None,
    segment: int,
    qcoeffs: np.ndarray,
    jds: np.ndarray,
    basis_residual: np.ndarray | None = None,
    shape_x_values: np.ndarray | None = None,
    shape_y_values: np.ndarray | None = None,
) -> np.ndarray:
    h = opm.header
    clock = validator.mercury_clock(opm) or validator.moon_clock(opm)
    a, b = validator.segment_bounds(h, segment, clock)
    tau = normalize_expanded(jds, a, b, h.expansion)
    if basis_residual is None:
        basis_residual = np.polynomial.chebyshev.chebvander(tau, h.residual_degree)
    coeffs = qcoeffs.astype(np.float64) * opm.quant_steps[None, :]
    if h.model_kind == validator.MODEL_RAW_XYZ_CHEB:
        return np.column_stack([basis_residual @ coeffs[axis] for axis in range(AXIS_COUNT)])
    if opm.shape_x is None or opm.shape_y is None or params is None:
        raise ValueError("this diagnostic expects raw Cheb or a reference-shape OPM")
    if shape_x_values is None:
        shape_x_values = proto.cheb_eval(opm.shape_x, tau)
    if shape_y_values is None:
        shape_y_values = proto.cheb_eval(opm.shape_y, tau)
    aligned = np.empty((1, len(jds), AXIS_COUNT), dtype=np.float64)
    aligned[0, :, 0] = shape_x_values + basis_residual @ coeffs[0]
    aligned[0, :, 1] = shape_y_values + basis_residual @ coeffs[1]
    aligned[0, :, 2] = basis_residual @ coeffs[2]
    return validator.unalign_positions_batched(aligned, params[segment : segment + 1])[0]


def errors_for_q(
    opm: validator.OpmFile,
    params: np.ndarray | None,
    segment: int,
    qcoeffs: np.ndarray,
    jds: np.ndarray,
    truth: np.ndarray,
    basis_residual: np.ndarray,
    shape_x_values: np.ndarray | None,
    shape_y_values: np.ndarray | None,
    error_metric: str = "angular",
) -> np.ndarray:
    recon = reconstruct_from_q(opm, params, segment, qcoeffs, jds, basis_residual, shape_x_values, shape_y_values)
    if error_metric == "km":
        return np.linalg.norm(recon - truth, axis=1)
    if error_metric == "angular":
        return proto.angular_errors_arcsec(truth, recon)
    raise ValueError(f"unknown error metric {error_metric}")


def objective(err: np.ndarray, mode: str) -> float:
    if mode in {"max", "guarded", "topk_guarded", "topk_ranked"}:
        return float(np.max(err))
    if mode == "p99max":
        return float(np.max(err) + 0.25 * np.percentile(err, 99))
    raise ValueError(f"unknown objective {mode}")


def topk_mean(err: np.ndarray, k: int) -> float:
    if k <= 0:
        return float(np.mean(err))
    kk = min(int(k), len(err))
    return float(np.mean(np.partition(err, len(err) - kk)[len(err) - kk :]))


def is_better_candidate(
    trial_err: np.ndarray,
    best_err: np.ndarray,
    mode: str,
    min_improvement: float,
    tail_topk: int = 0,
) -> bool:
    trial_max = float(np.max(trial_err))
    best_max = float(np.max(best_err))
    if mode in {"guarded", "topk_ranked"}:
        trial_p99 = float(np.percentile(trial_err, 99))
        best_p99 = float(np.percentile(best_err, 99))
        return trial_max + min_improvement < best_max and trial_p99 <= best_p99 + 1e-15
    if mode == "topk_guarded":
        trial_p99 = float(np.percentile(trial_err, 99))
        best_p99 = float(np.percentile(best_err, 99))
        trial_tail = topk_mean(trial_err, tail_topk)
        best_tail = topk_mean(best_err, tail_topk)
        return (
            trial_max + min_improvement < best_max
            and trial_p99 <= best_p99 + 1e-15
            and trial_tail <= best_tail + 1e-15
        )
    return objective(trial_err, mode) + min_improvement < objective(best_err, mode)


def q_value_width(value: int) -> int:
    encoded = value * 2 if value >= 0 else -value * 2 - 1
    return max(1, int(encoded).bit_length())


def optimize_segment(
    opm: validator.OpmFile,
    params: np.ndarray | None,
    segment: int,
    jds: np.ndarray,
    truth: np.ndarray,
    *,
    max_passes: int,
    radius: int,
    objective_mode: str,
    min_improvement: float,
    width_limits: np.ndarray | None = None,
    tail_topk: int = 0,
    error_metric: str = "angular",
) -> dict[str, object]:
    h = opm.header
    a, b = validator.segment_bounds(h, segment, validator.mercury_clock(opm) or validator.moon_clock(opm))
    tau = normalize_expanded(jds, a, b, h.expansion)
    basis = np.polynomial.chebyshev.chebvander(tau, h.residual_degree)
    shape_x_values = proto.cheb_eval(opm.shape_x, tau) if opm.shape_x is not None else None
    shape_y_values = proto.cheb_eval(opm.shape_y, tau) if opm.shape_y is not None else None

    q = opm.qcoeffs[segment].copy()
    initial_q = q.copy()
    err = errors_for_q(opm, params, segment, q, jds, truth, basis, shape_x_values, shape_y_values, error_metric)
    best_obj = objective(err, objective_mode)
    best_err = err
    changes: list[tuple[int, int, int, float]] = []

    candidates = [(axis, degree) for axis in range(AXIS_COUNT) for degree in range(h.residual_degree + 1)]
    for pass_idx in range(max_passes):
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
                    if width_limits is not None and q_value_width(trial_value) > int(width_limits[axis, degree]):
                        continue
                    q[axis, degree] = trial_value
                    trial_err = errors_for_q(opm, params, segment, q, jds, truth, basis, shape_x_values, shape_y_values, error_metric)
                    # Positive score means this parameter direction improves the current top-K tail.
                    score = current_tail - topk_mean(trial_err, tail_topk)
                    ranked.append((score, axis, degree, delta))
                    q[axis, degree] = current_value
            ranked.sort(reverse=True, key=lambda item: (item[0], item[2], item[1]))
            for _score, axis, degree, delta in ranked:
                current_value = int(q[axis, degree])
                trial_value = current_value + delta
                if width_limits is not None and q_value_width(trial_value) > int(width_limits[axis, degree]):
                    continue
                q[axis, degree] = trial_value
                trial_err = errors_for_q(opm, params, segment, q, jds, truth, basis, shape_x_values, shape_y_values, error_metric)
                if is_better_candidate(trial_err, best_err, objective_mode, min_improvement, tail_topk):
                    best_obj = objective(trial_err, objective_mode)
                    best_err = trial_err
                    changes.append((axis, degree, delta, float(np.max(best_err))))
                    improved = True
                else:
                    q[axis, degree] = current_value
        else:
            # Try high-degree first, then low-degree; high degree often changes local shape
            # without moving the whole segment as much.
            for axis, degree in sorted(candidates, key=lambda x: (x[1], x[0]), reverse=True):
                current_value = int(q[axis, degree])
                local_best_obj = best_obj
                local_best_delta = 0
                local_best_err = best_err
                for delta in range(-radius, radius + 1):
                    if delta == 0:
                        continue
                    trial_value = current_value + delta
                    if width_limits is not None and q_value_width(trial_value) > int(width_limits[axis, degree]):
                        continue
                    q[axis, degree] = trial_value
                    trial_err = errors_for_q(opm, params, segment, q, jds, truth, basis, shape_x_values, shape_y_values, error_metric)
                    if is_better_candidate(trial_err, local_best_err, objective_mode, min_improvement, tail_topk):
                        local_best_obj = objective(trial_err, objective_mode)
                        local_best_delta = delta
                        local_best_err = trial_err
                q[axis, degree] = current_value
                if local_best_delta:
                    q[axis, degree] = current_value + local_best_delta
                    best_obj = local_best_obj
                    best_err = local_best_err
                    changes.append((axis, degree, local_best_delta, float(np.max(best_err))))
                    improved = True
        if not improved:
            break

    return {
        "segment": segment,
        "initial_err": err,
        "best_err": best_err,
        "initial_q": initial_q,
        "best_q": q,
        "changes": changes,
    }


def print_err_summary(prefix: str, err: np.ndarray) -> None:
    print(
        f"{prefix} p50={np.percentile(err,50):.9g} "
        f"p95={np.percentile(err,95):.9g} "
        f"p99={np.percentile(err,99):.9g} "
        f"max={np.max(err):.9g}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("opm", type=Path)
    parser.add_argument("--de441", type=Path, required=True)
    parser.add_argument("--segments", required=True, help="comma-separated segment indices")
    parser.add_argument("--nodes-per-segment", type=int, default=32)
    parser.add_argument("--max-passes", type=int, default=4)
    parser.add_argument("--radius", type=int, default=1, help="try q coefficient deltas in [-radius, radius]")
    parser.add_argument("--objective", choices=["max", "p99max", "guarded"], default="max")
    parser.add_argument("--min-improvement", type=float, default=1e-12)
    parser.add_argument("--top-changes", type=int, default=12)
    parser.add_argument("--no-crc", action="store_true")
    args = parser.parse_args()

    proto.set_de441_path(args.de441)
    moon_proto.set_de441_path(args.de441)
    opm = validator.read_opm(args.opm, check_crc=not args.no_crc)
    clock = validator.mercury_clock(opm) or validator.moon_clock(opm)
    params = validator.frame_params_for_segments(opm, clock) if opm.frame_coeffs is not None else None
    if params is None:
        raise SystemExit("this diagnostic expects frame parameters")
    segments = parse_csv_ints(args.segments)

    all_initial: list[np.ndarray] = []
    all_best: list[np.ndarray] = []
    results: list[dict[str, object]] = []
    with SPK.open(str(args.de441)) as spk:
        provider, closeable = validator.truth_position_provider(spk, opm)
        try:
            for seg in segments:
                jds = segment_eval_nodes(opm, seg, args.nodes_per_segment, clock)
                truth = provider.position(jds)
                result = optimize_segment(
                    opm,
                    params,
                    seg,
                    jds,
                    truth,
                    max_passes=args.max_passes,
                    radius=args.radius,
                    objective_mode=args.objective,
                    min_improvement=args.min_improvement,
                )
                results.append(result)
                all_initial.append(result["initial_err"])  # type: ignore[arg-type]
                all_best.append(result["best_err"])  # type: ignore[arg-type]
        finally:
            validator.close_if_needed(closeable)

    print(f"file: {args.opm}")
    print(f"segments: {','.join(str(s) for s in segments)} nodes/segment={args.nodes_per_segment} objective={args.objective}")
    print()
    print_err_summary("initial aggregate", np.concatenate(all_initial))
    print_err_summary("best aggregate   ", np.concatenate(all_best))
    print()
    print("per-segment quantization-aware rounding result")
    print("seg year initial_max best_max delta changes max_abs_q_delta changed_coeffs")
    for result in results:
        seg = int(result["segment"])
        a, b = validator.segment_bounds(opm.header, seg, clock)
        year = (0.5 * (a + b) - JD_J2000) / DAYS_PER_JULIAN_YEAR
        initial_err = result["initial_err"]  # type: ignore[assignment]
        best_err = result["best_err"]  # type: ignore[assignment]
        initial_q = result["initial_q"]  # type: ignore[assignment]
        best_q = result["best_q"]  # type: ignore[assignment]
        q_delta = np.asarray(best_q) - np.asarray(initial_q)
        changes = result["changes"]  # type: ignore[assignment]
        changed = int(np.count_nonzero(q_delta))
        max_abs_delta = int(np.max(np.abs(q_delta))) if changed else 0
        print(
            f"{seg:7d} {year:+10.1f} "
            f"{np.max(initial_err):.9g} {np.max(best_err):.9g} {np.max(best_err)-np.max(initial_err):+.9g} "
            f"{len(changes):7d} {max_abs_delta:15d} {changed:14d}"
        )
        if args.top_changes and changes:
            text = "; ".join(
                f"{('xyz'[axis])}{degree}:{delta:+d}->max{max_err:.6g}"
                for axis, degree, delta, max_err in changes[: args.top_changes]
            )
            print(f"        first changes: {text}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
