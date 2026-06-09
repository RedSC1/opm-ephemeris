#!/usr/bin/env python3
"""Refit selected PEF segments to diagnose fit-vs-quantization max errors.

For each selected segment this script keeps the existing PEF reference shape and
frame model, then refits only that segment's residual coefficients under a grid
of residual degrees, fit-node oversampling factors, and segment-domain expansion
fractions.  It reports unquantized vs quantized validation errors, making it
clear whether worst validation spikes are approximation error or quantization
error.
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

import pef_demo.moon_model as moon_proto  # noqa: E402
import pef_demo.orbit_model as proto  # noqa: E402
from pef_demo import validator  # noqa: E402
from pef_demo.packing import degree_quant_steps  # noqa: E402

JD_J2000 = 2451545.0
DAYS_PER_JULIAN_YEAR = 365.25
AXIS_COUNT = 3


def parse_csv_ints(text: str) -> list[int]:
    out: list[int] = []
    for part in text.split(","):
        part = part.strip()
        if part:
            out.append(int(part))
    return out


def parse_csv_floats(text: str) -> list[float]:
    out: list[float] = []
    for part in text.split(","):
        part = part.strip()
        if part:
            out.append(float(part))
    return out


def cheb_nodes(a: float, b: float, n: int) -> np.ndarray:
    k = np.arange(n)
    x = np.cos(np.pi * (k + 0.5) / n)
    return np.sort(0.5 * (a + b) + 0.5 * (b - a) * x)


def expanded_bounds(a: float, b: float, expansion: float) -> tuple[float, float]:
    width = b - a
    return a - expansion * width, b + expansion * width


def normalize_expanded(jd: np.ndarray, a: float, b: float, expansion: float) -> np.ndarray:
    ea, eb = expanded_bounds(a, b, expansion)
    return (2.0 * jd - ea - eb) / (eb - ea)


def fit_segment_coeffs(
    provider: object,
    pef: validator.PefFile,
    params: np.ndarray,
    segment: int,
    degree: int,
    oversample: int,
    expansion: float,
) -> np.ndarray:
    if pef.shape_x is None or pef.shape_y is None:
        raise ValueError("this diagnostic currently expects a reference-shape model")
    a, b = validator.segment_bounds(pef.header, segment, validator.mercury_clock(pef) or validator.moon_clock(pef))
    fit_nodes = (max(pef.header.reference_shape_degree, degree) + 1) * oversample
    fa, fb = expanded_bounds(a, b, expansion)
    nodes = cheb_nodes(fa, fb, fit_nodes)
    tau = normalize_expanded(nodes, a, b, expansion)
    pos = provider.position(nodes)
    plane_u, plane_v, apsis_angle = params[segment]
    aligned_truth = moon_proto.align_positions(pos, float(plane_u), float(plane_v), float(apsis_angle))
    values = np.empty((fit_nodes, AXIS_COUNT), dtype=np.float64)
    values[:, 0] = aligned_truth[:, 0] - proto.cheb_eval(pef.shape_x, tau)
    values[:, 1] = aligned_truth[:, 1] - proto.cheb_eval(pef.shape_y, tau)
    values[:, 2] = aligned_truth[:, 2]
    return np.vstack([np.polynomial.chebyshev.chebfit(tau, values[:, axis], degree) for axis in range(AXIS_COUNT)])


def reconstruct_segment(
    pef: validator.PefFile,
    params: np.ndarray,
    segment: int,
    coeffs: np.ndarray,
    jds: np.ndarray,
    expansion: float,
) -> np.ndarray:
    if pef.shape_x is None or pef.shape_y is None:
        raise ValueError("this diagnostic currently expects a reference-shape model")
    a, b = validator.segment_bounds(pef.header, segment, validator.mercury_clock(pef) or validator.moon_clock(pef))
    tau = normalize_expanded(jds, a, b, expansion)
    aligned = np.empty((1, len(jds), AXIS_COUNT), dtype=np.float64)
    aligned[0, :, 0] = proto.cheb_eval(pef.shape_x, tau) + proto.cheb_eval(coeffs[0], tau)
    aligned[0, :, 1] = proto.cheb_eval(pef.shape_y, tau) + proto.cheb_eval(coeffs[1], tau)
    aligned[0, :, 2] = proto.cheb_eval(coeffs[2], tau)
    return validator.unalign_positions_batched(aligned, params[segment : segment + 1])[0]


def current_segment_recon(
    pef: validator.PefFile,
    params: np.ndarray,
    segment: int,
    jds: np.ndarray,
) -> np.ndarray:
    coeffs = pef.qcoeffs[segment].astype(np.float64) * pef.quant_steps[None, :]
    return reconstruct_segment(pef, params, segment, coeffs, jds, pef.header.expansion)


def segment_eval_nodes(pef: validator.PefFile, segment: int, nodes_per_segment: int, clock: object | None) -> np.ndarray:
    h = pef.header
    coverage_end = h.coverage_start_jd + h.coverage_span_days
    a, b = validator.segment_bounds(h, segment, clock)
    lo = max(a, h.coverage_start_jd)
    hi = min(b, coverage_end)
    return cheb_nodes(lo, hi, nodes_per_segment)


def summarize(values: np.ndarray) -> tuple[float, float, float, float]:
    return (float(np.percentile(values, 50)), float(np.percentile(values, 95)), float(np.percentile(values, 99)), float(np.max(values)))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pef", type=Path)
    parser.add_argument("--de441", type=Path, required=True)
    parser.add_argument("--segments", required=True, help="comma-separated segment indices to test")
    parser.add_argument("--degrees", default="24,25,26")
    parser.add_argument("--expansions", default="0,0.005,0.01,0.02,0.05")
    parser.add_argument("--oversamples", default="3")
    parser.add_argument("--nodes-per-segment", type=int, default=32)
    parser.add_argument("--quant-base-km", type=float, default=0.00025)
    parser.add_argument("--quant-pattern", default="flat")
    parser.add_argument("--no-crc", action="store_true")
    args = parser.parse_args()

    proto.set_de441_path(args.de441)
    moon_proto.set_de441_path(args.de441)
    pef = validator.read_pef(args.pef, check_crc=not args.no_crc)
    clock = validator.mercury_clock(pef) or validator.moon_clock(pef)
    params = validator.frame_params_for_segments(pef, clock) if pef.frame_coeffs is not None else None
    if params is None:
        raise SystemExit("this diagnostic expects a PEF with frame parameters")

    segments = parse_csv_ints(args.segments)
    degrees = parse_csv_ints(args.degrees)
    expansions = parse_csv_floats(args.expansions)
    oversamples = parse_csv_ints(args.oversamples)

    with SPK.open(str(args.de441)) as spk:
        provider, closeable = validator.truth_position_provider(spk, pef)
        try:
            eval_jds_by_segment = {seg: segment_eval_nodes(pef, seg, args.nodes_per_segment, clock) for seg in segments}
            truth_by_segment = {seg: provider.position(eval_jds_by_segment[seg]) for seg in segments}

            current_err_parts = []
            print("current stored PEF errors for selected segments")
            print("seg year max p99 p95 p50 max_tau")
            for seg in segments:
                jds = eval_jds_by_segment[seg]
                truth = truth_by_segment[seg]
                recon = current_segment_recon(pef, params, seg, jds)
                err = proto.angular_errors_arcsec(truth, recon)
                current_err_parts.append(err)
                a, b = validator.segment_bounds(pef.header, seg, clock)
                tau = normalize_expanded(jds, a, b, pef.header.expansion)
                imax = int(np.argmax(err))
                year = (0.5 * (a + b) - JD_J2000) / DAYS_PER_JULIAN_YEAR
                print(
                    f"{seg:7d} {year:+10.1f} {np.max(err):.9g} {np.percentile(err,99):.9g} "
                    f"{np.percentile(err,95):.9g} {np.percentile(err,50):.9g} {tau[imax]:+8.4f}"
                )
            current_all = np.concatenate(current_err_parts)
            p50, p95, p99, mx = summarize(current_all)
            print(f"CURRENT selected aggregate: p50={p50:.9g} p95={p95:.9g} p99={p99:.9g} max={mx:.9g}")
            print()

            print("refit grid aggregate errors")
            print("degree oversample expansion unquant_p50 unquant_p95 unquant_p99 unquant_max quant_p50 quant_p95 quant_p99 quant_max quant_minus_unquant_max")
            for degree in degrees:
                steps = degree_quant_steps(degree, args.quant_base_km, args.quant_pattern).astype(np.float32).astype(np.float64)
                for oversample in oversamples:
                    for expansion in expansions:
                        unquant_parts = []
                        quant_parts = []
                        for seg in segments:
                            coeffs = fit_segment_coeffs(provider, pef, params, seg, degree, oversample, expansion)
                            qcoeffs = np.round(coeffs / steps[None, :]).astype(np.int64)
                            quant_coeffs = qcoeffs.astype(np.float64) * steps[None, :]
                            jds = eval_jds_by_segment[seg]
                            truth = truth_by_segment[seg]
                            recon_unquant = reconstruct_segment(pef, params, seg, coeffs, jds, expansion)
                            recon_quant = reconstruct_segment(pef, params, seg, quant_coeffs, jds, expansion)
                            unquant_parts.append(proto.angular_errors_arcsec(truth, recon_unquant))
                            quant_parts.append(proto.angular_errors_arcsec(truth, recon_quant))
                        unquant = np.concatenate(unquant_parts)
                        quant = np.concatenate(quant_parts)
                        uq50, uq95, uq99, uqmax = summarize(unquant)
                        q50, q95, q99, qmax = summarize(quant)
                        print(
                            f"{degree:6d} {oversample:10d} {expansion:9.4f} "
                            f"{uq50:.9g} {uq95:.9g} {uq99:.9g} {uqmax:.9g} "
                            f"{q50:.9g} {q95:.9g} {q99:.9g} {qmax:.9g} {qmax-uqmax:+.9g}"
                        )
            print()

            print("best per segment among grid by quantized max")
            print("seg current_max best_degree best_oversample best_expansion best_unquant_max best_quant_max")
            for seg in segments:
                best: tuple[float, int, int, float, float] | None = None
                current_err = proto.angular_errors_arcsec(truth_by_segment[seg], current_segment_recon(pef, params, seg, eval_jds_by_segment[seg]))
                for degree in degrees:
                    steps = degree_quant_steps(degree, args.quant_base_km, args.quant_pattern).astype(np.float32).astype(np.float64)
                    for oversample in oversamples:
                        for expansion in expansions:
                            coeffs = fit_segment_coeffs(provider, pef, params, seg, degree, oversample, expansion)
                            qcoeffs = np.round(coeffs / steps[None, :]).astype(np.int64)
                            quant_coeffs = qcoeffs.astype(np.float64) * steps[None, :]
                            jds = eval_jds_by_segment[seg]
                            truth = truth_by_segment[seg]
                            uq = proto.angular_errors_arcsec(truth, reconstruct_segment(pef, params, seg, coeffs, jds, expansion))
                            qe = proto.angular_errors_arcsec(truth, reconstruct_segment(pef, params, seg, quant_coeffs, jds, expansion))
                            item = (float(np.max(qe)), degree, oversample, expansion, float(np.max(uq)))
                            if best is None or item[0] < best[0]:
                                best = item
                assert best is not None
                print(f"{seg:7d} {np.max(current_err):.9g} {best[1]:11d} {best[2]:15d} {best[3]:14.4f} {best[4]:.9g} {best[0]:.9g}")
        finally:
            validator.close_if_needed(closeable)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
