#!/usr/bin/env python3
"""Analyze long-time drift in PEF residual coefficient caches.

This is a read-only/offline helper for body-packed tuning.  It loads a
``*-residual-coeffs.npz`` cache, fits low-degree Chebyshev trends across segment
midpoints for each axis/degree residual coefficient column, subtracts the fitted
trend, and estimates how much the unified axis-degree bit-width table would
shrink if that trend were moved into the reference model.
"""
from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pef_demo.packing import degree_quant_steps, zigzag_bit_length  # noqa: E402

AXIS_NAMES = ("x", "y", "z")
JD_J2000 = 2451545.0
DAYS_PER_JULIAN_YEAR = 365.25


@dataclass(frozen=True)
class ColumnStats:
    axis: int
    degree: int
    current_width: int
    simulated_width: int
    linear_r2: float
    fit_r2: float
    cheb3_r2: float | None
    rms_before_km: float
    rms_after_km: float
    peak_period_years: tuple[float, ...]

    @property
    def saved_bits(self) -> int:
        return self.current_width - self.simulated_width


def scalar(data: np.lib.npyio.NpzFile, name: str):
    return data[name].item()


def parse_axes(text: str) -> tuple[int, ...]:
    axes: list[int] = []
    for part in text.split(","):
        name = part.strip().lower()
        if not name:
            continue
        if name not in AXIS_NAMES:
            raise argparse.ArgumentTypeError(f"unknown axis {name!r}; use x,y,z")
        axes.append(AXIS_NAMES.index(name))
    if not axes:
        raise argparse.ArgumentTypeError("at least one axis is required")
    return tuple(dict.fromkeys(axes))


def bit_widths(qcoeffs: np.ndarray) -> np.ndarray:
    widths = np.zeros(qcoeffs.shape[1:], dtype=np.uint8)
    for axis in range(qcoeffs.shape[1]):
        for degree in range(qcoeffs.shape[2]):
            values = qcoeffs[:, axis, degree]
            widths[axis, degree] = max(zigzag_bit_length(int(v)) for v in values)
    return widths


def quantize(coeffs: np.ndarray, steps: np.ndarray) -> np.ndarray:
    return np.round(coeffs / steps[None, None, :]).astype(np.int64)


def fit_trend(values: np.ndarray, tnorm: np.ndarray, degree: int) -> tuple[np.ndarray, float, float]:
    basis = np.polynomial.chebyshev.chebvander(tnorm, degree)
    coeff, *_ = np.linalg.lstsq(basis, values, rcond=None)
    fitted = basis @ coeff
    residual = values - fitted
    rms_before = float(np.sqrt(np.mean(values * values)))
    rms_after = float(np.sqrt(np.mean(residual * residual)))
    centered = values - float(np.mean(values))
    ss_tot = float(np.dot(centered, centered))
    ss_res = float(np.dot(residual, residual))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0.0 else 1.0
    return fitted, r2, rms_after if rms_before >= 0.0 else rms_after


def r2_for_degree(values: np.ndarray, tnorm: np.ndarray, degree: int) -> tuple[float, float]:
    _fitted, r2, rms_after = fit_trend(values, tnorm, degree)
    return r2, rms_after


def dominant_periods(values: np.ndarray, sample_spacing_years: float, count: int) -> tuple[float, ...]:
    if count <= 0 or len(values) < 4 or sample_spacing_years <= 0.0:
        return ()
    centered = values - float(np.mean(values))
    spectrum = np.fft.rfft(centered)
    freq = np.fft.rfftfreq(len(centered), d=sample_spacing_years)
    if len(freq) <= 1:
        return ()
    power = np.abs(spectrum) ** 2
    power[0] = 0.0
    positive = np.nonzero(freq > 0.0)[0]
    if len(positive) == 0:
        return ()
    take = min(count, len(positive))
    idx = positive[np.argpartition(power[positive], -take)[-take:]]
    idx = idx[np.argsort(power[idx])[::-1]]
    return tuple(float(1.0 / freq[i]) for i in idx if freq[i] > 0.0)


def format_periods(periods: Iterable[float]) -> str:
    values = list(periods)
    if not values:
        return "-"
    return ",".join(f"{p:.0f}y" for p in values)


def load_cache(path: Path) -> tuple[dict[str, object], np.ndarray, np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        meta = {
            "body": str(scalar(data, "body")),
            "center": str(scalar(data, "center")),
            "method": str(scalar(data, "method")),
            "jd_start": float(scalar(data, "jd_start")),
            "jd_end": float(scalar(data, "jd_end")),
            "residual_degree": int(scalar(data, "residual_degree")),
            "quant_base_km": float(scalar(data, "quant_base_km")),
            "quant_pattern": str(scalar(data, "quant_pattern")),
        }
        boundaries = np.asarray(data["boundaries"], dtype=np.float64)
        coeffs = np.asarray(data["coeffs"], dtype=np.float64)
    degree = int(meta["residual_degree"])
    if coeffs.ndim != 3 or coeffs.shape[1] != 3 or coeffs.shape[2] != degree + 1:
        raise ValueError(f"unexpected coeffs shape {coeffs.shape}; expected (segments, 3, {degree + 1})")
    if boundaries.shape != (coeffs.shape[0] + 1,):
        raise ValueError(f"unexpected boundaries shape {boundaries.shape}; expected ({coeffs.shape[0] + 1},)")
    steps = degree_quant_steps(degree, float(meta["quant_base_km"]), str(meta["quant_pattern"])).astype(np.float32).astype(np.float64)
    return meta, boundaries, coeffs, steps


def analyze(
    coeffs: np.ndarray,
    boundaries: np.ndarray,
    steps: np.ndarray,
    *,
    fit_degree: int,
    drift_axes: tuple[int, ...],
    max_report_degree: int | None,
    fft_peaks: int,
) -> tuple[np.ndarray, np.ndarray, list[ColumnStats]]:
    tmids = 0.5 * (boundaries[:-1] + boundaries[1:])
    tnorm = np.polynomial.chebyshev.chebpts1(len(tmids)) if len(tmids) == 1 else 2.0 * (tmids - tmids[0]) / (tmids[-1] - tmids[0]) - 1.0
    years = (tmids - JD_J2000) / DAYS_PER_JULIAN_YEAR
    sample_spacing_years = float(np.median(np.diff(years))) if len(years) > 1 else 0.0

    q_before = quantize(coeffs, steps)
    current_widths = bit_widths(q_before)

    simulated_coeffs = coeffs.copy()
    stats: list[ColumnStats] = []
    stop_degree = coeffs.shape[2] - 1 if max_report_degree is None else min(max_report_degree, coeffs.shape[2] - 1)
    for axis in range(coeffs.shape[1]):
        for degree in range(stop_degree + 1):
            values = coeffs[:, axis, degree]
            fitted, fit_r2, rms_after = fit_trend(values, tnorm, fit_degree)
            linear_r2, _linear_rms_after = r2_for_degree(values, tnorm, 1)
            cheb3_r2 = None
            if fit_degree < 3:
                cheb3_r2, _cheb3_rms_after = r2_for_degree(values, tnorm, 3)
            rms_before = float(np.sqrt(np.mean(values * values)))
            if axis in drift_axes:
                simulated_coeffs[:, axis, degree] = values - fitted
            q_after_col = np.round((values - fitted) / steps[degree]).astype(np.int64) if axis in drift_axes else q_before[:, axis, degree]
            simulated_width = max(zigzag_bit_length(int(v)) for v in q_after_col)
            stats.append(
                ColumnStats(
                    axis=axis,
                    degree=degree,
                    current_width=int(current_widths[axis, degree]),
                    simulated_width=int(simulated_width),
                    linear_r2=float(linear_r2),
                    fit_r2=float(fit_r2),
                    cheb3_r2=None if cheb3_r2 is None else float(cheb3_r2),
                    rms_before_km=rms_before,
                    rms_after_km=float(rms_after),
                    peak_period_years=dominant_periods(values, sample_spacing_years, fft_peaks),
                )
            )

    q_after = quantize(simulated_coeffs, steps)
    simulated_widths = bit_widths(q_after)
    return current_widths, simulated_widths, stats


def print_width_matrix(title: str, widths: np.ndarray) -> None:
    print(title)
    for axis, name in enumerate(AXIS_NAMES):
        values = " ".join(f"{int(v):2d}" for v in widths[axis])
        print(f"  {name}: {values}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cache", type=Path, help="residual coefficient .npz cache")
    parser.add_argument("--fit-degree", type=int, default=2, help="Chebyshev degree to subtract for the simulation")
    parser.add_argument("--axes", type=parse_axes, default=parse_axes("x,y"), help="comma-separated axes whose trends are moved into the reference model")
    parser.add_argument("--max-degree", type=int, default=None, help="highest coefficient degree to analyze/report; default is all")
    parser.add_argument("--top", type=int, default=25, help="number of most important columns to print")
    parser.add_argument("--fft-peaks", type=int, default=3, help="dominant FFT periods to print per reported column")
    parser.add_argument("--show-widths", action="store_true", help="print full current/simulated width matrices")
    args = parser.parse_args(argv)

    if args.fit_degree < 0:
        parser.error("--fit-degree must be non-negative")

    meta, boundaries, coeffs, steps = load_cache(args.cache)
    current_widths, simulated_widths, stats = analyze(
        coeffs,
        boundaries,
        steps,
        fit_degree=args.fit_degree,
        drift_axes=args.axes,
        max_report_degree=args.max_degree,
        fft_peaks=args.fft_peaks,
    )

    segments = coeffs.shape[0]
    current_bits_per_segment = int(np.sum(current_widths))
    simulated_bits_per_segment = int(np.sum(simulated_widths))
    saved_bits_per_segment = current_bits_per_segment - simulated_bits_per_segment
    current_payload_bytes = math.ceil(segments * current_bits_per_segment / 8)
    simulated_payload_bytes = math.ceil(segments * simulated_bits_per_segment / 8)
    saved_bytes = current_payload_bytes - simulated_payload_bytes

    years_start = (float(boundaries[0]) - JD_J2000) / DAYS_PER_JULIAN_YEAR
    years_end = (float(boundaries[-1]) - JD_J2000) / DAYS_PER_JULIAN_YEAR
    print(f"cache: {args.cache}")
    print(f"body: {meta['body']} center={meta['center']} method={meta['method']}")
    print(f"segments: {segments:,}  residual_degree: {meta['residual_degree']}  quant: {meta['quant_base_km']} km {meta['quant_pattern']}")
    print(f"years relative to J2000: {years_start:.1f} .. {years_end:.1f}")
    print(f"drift simulation: subtract Cheb{args.fit_degree} on axes {','.join(AXIS_NAMES[a] for a in args.axes)}")
    print()
    print("payload-width estimate")
    print(f"  current width sum:   {current_bits_per_segment:,} bits/segment")
    print(f"  simulated width sum: {simulated_bits_per_segment:,} bits/segment")
    print(f"  saved:               {saved_bits_per_segment:,} bits/segment")
    print(f"  current payload:     {current_payload_bytes / 1024.0:.3f} KiB")
    print(f"  simulated payload:   {simulated_payload_bytes / 1024.0:.3f} KiB")
    print(f"  payload saved:       {saved_bytes / 1024.0:.3f} KiB = {saved_bytes / (1024.0 * 1024.0):.3f} MiB")
    print()

    if args.show_widths:
        print_width_matrix("current widths", current_widths)
        print_width_matrix("simulated widths", simulated_widths)
        print()

    stats_by_saved = sorted(stats, key=lambda s: (s.saved_bits, s.current_width, s.fit_r2), reverse=True)
    print(f"top {min(args.top, len(stats_by_saved))} columns by saved width")
    print("axis deg width->sim saved linear_R2 fit_R2 cheb3_R2 rms_km->after_km peaks")
    for item in stats_by_saved[: args.top]:
        cheb3 = "-" if item.cheb3_r2 is None else f"{item.cheb3_r2:.3f}"
        print(
            f"{AXIS_NAMES[item.axis]:>4} {item.degree:>3d} "
            f"{item.current_width:>2d}->{item.simulated_width:<2d} {item.saved_bits:>+3d} "
            f"{item.linear_r2:>8.3f} {item.fit_r2:>7.3f} {cheb3:>8} "
            f"{item.rms_before_km:>12.3g}->{item.rms_after_km:<12.3g} "
            f"{format_periods(item.peak_period_years)}"
        )

    print()
    stats_by_width = sorted(stats, key=lambda s: (s.current_width, s.fit_r2), reverse=True)
    print(f"top {min(args.top, len(stats_by_width))} columns by current width")
    print("axis deg width->sim saved linear_R2 fit_R2 cheb3_R2 rms_km->after_km peaks")
    for item in stats_by_width[: args.top]:
        cheb3 = "-" if item.cheb3_r2 is None else f"{item.cheb3_r2:.3f}"
        print(
            f"{AXIS_NAMES[item.axis]:>4} {item.degree:>3d} "
            f"{item.current_width:>2d}->{item.simulated_width:<2d} {item.saved_bits:>+3d} "
            f"{item.linear_r2:>8.3f} {item.fit_r2:>7.3f} {cheb3:>8} "
            f"{item.rms_before_km:>12.3g}->{item.rms_after_km:<12.3g} "
            f"{format_periods(item.peak_period_years)}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
