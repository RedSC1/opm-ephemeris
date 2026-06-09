#!/usr/bin/env python3
"""Find residual coefficient columns whose global bit width is set by outliers.

PEF body-packed payloads use one width per axis/degree column for the whole
file.  This helper loads a residual coefficient cache, quantizes it exactly like
writer code, and reports whether each wide column's maximum width is supported by
many segments or by a small tail of segments.
"""
from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pef_demo.packing import degree_quant_steps  # noqa: E402

AXIS_NAMES = ("x", "y", "z")
JD_J2000 = 2451545.0
DAYS_PER_JULIAN_YEAR = 365.25


@dataclass(frozen=True)
class ColumnOutlierStats:
    axis: int
    degree: int
    width: int
    width_without_1: int
    width_without_10: int
    width_without_100: int
    width_without_1000: int
    count_at_width: int
    count_ge_width_minus_1: int
    count_ge_width_minus_2: int
    count_ge_width_minus_3: int
    q_abs_max: int
    q_abs_p999: float
    q_abs_p99: float
    top_segments: tuple[tuple[int, float, int, int], ...]

    @property
    def segments_to_save_1_bit(self) -> int:
        return self.count_at_width

    @property
    def segments_to_save_2_bits(self) -> int:
        return self.count_ge_width_minus_1

    @property
    def segments_to_save_3_bits(self) -> int:
        return self.count_ge_width_minus_2


def scalar(data: np.lib.npyio.NpzFile, name: str):
    return data[name].item()


def zigzag_lengths(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.int64)
    encoded = np.where(values >= 0, values * 2, -values * 2 - 1).astype(np.uint64)
    # np.frexp returns exponent e such that encoded = m * 2**e, with m in [0.5, 1).
    # For positive integers this is the bit length; zeros are handled as width 1.
    lengths = np.frexp(encoded.astype(np.float64))[1].astype(np.int16)
    lengths[encoded == 0] = 1
    return lengths


def quantize(coeffs: np.ndarray, steps: np.ndarray) -> np.ndarray:
    return np.round(coeffs / steps[None, None, :]).astype(np.int64)


def width_after_ignoring(lengths: np.ndarray, ignore_count: int) -> int:
    if ignore_count <= 0:
        return int(np.max(lengths))
    if ignore_count >= len(lengths):
        return 0
    # kth largest after removing ignore_count largest is partition index ignore_count
    descending = np.partition(lengths, len(lengths) - ignore_count - 1)
    return int(np.max(descending[: len(lengths) - ignore_count]))


def load_cache(path: Path) -> tuple[dict[str, object], np.ndarray, np.ndarray]:
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
    steps = degree_quant_steps(degree, float(meta["quant_base_km"]), str(meta["quant_pattern"])).astype(np.float32).astype(np.float64)
    qcoeffs = quantize(coeffs, steps)
    return meta, boundaries, qcoeffs


def top_segment_info(values: np.ndarray, lengths: np.ndarray, boundaries: np.ndarray, count: int) -> tuple[tuple[int, float, int, int], ...]:
    if count <= 0:
        return ()
    order = np.lexsort((-np.abs(values), -lengths))
    # lexsort sorts ascending; with negative keys, first rows are largest length/abs.
    rows: list[tuple[int, float, int, int]] = []
    for idx in order[:count]:
        seg = int(idx)
        year = (0.5 * (float(boundaries[seg]) + float(boundaries[seg + 1])) - JD_J2000) / DAYS_PER_JULIAN_YEAR
        rows.append((seg, year, int(values[seg]), int(lengths[seg])))
    return tuple(rows)


def analyze_columns(qcoeffs: np.ndarray, boundaries: np.ndarray, top_segments: int) -> list[ColumnOutlierStats]:
    out: list[ColumnOutlierStats] = []
    for axis in range(qcoeffs.shape[1]):
        for degree in range(qcoeffs.shape[2]):
            values = qcoeffs[:, axis, degree]
            lengths = zigzag_lengths(values)
            width = int(np.max(lengths))
            abs_values = np.abs(values)
            out.append(
                ColumnOutlierStats(
                    axis=axis,
                    degree=degree,
                    width=width,
                    width_without_1=width_after_ignoring(lengths, 1),
                    width_without_10=width_after_ignoring(lengths, 10),
                    width_without_100=width_after_ignoring(lengths, 100),
                    width_without_1000=width_after_ignoring(lengths, 1000),
                    count_at_width=int(np.count_nonzero(lengths >= width)),
                    count_ge_width_minus_1=int(np.count_nonzero(lengths >= max(1, width - 1))),
                    count_ge_width_minus_2=int(np.count_nonzero(lengths >= max(1, width - 2))),
                    count_ge_width_minus_3=int(np.count_nonzero(lengths >= max(1, width - 3))),
                    q_abs_max=int(np.max(abs_values)),
                    q_abs_p999=float(np.percentile(abs_values, 99.9)),
                    q_abs_p99=float(np.percentile(abs_values, 99.0)),
                    top_segments=top_segment_info(values, lengths, boundaries, top_segments),
                )
            )
    return out


def summarize_potential(stats: list[ColumnOutlierStats], segments: int, thresholds: tuple[int, ...]) -> None:
    print("tail-removal thought experiment")
    print("  This is not a proposed format; it only shows how much width is held by a tail.")
    for threshold in thresholds:
        saved_bits_1 = 0
        saved_bits_2 = 0
        saved_bits_3 = 0
        columns_1 = 0
        columns_2 = 0
        columns_3 = 0
        for s in stats:
            if s.segments_to_save_1_bit <= threshold:
                saved_bits_1 += 1
                columns_1 += 1
            if s.segments_to_save_2_bits <= threshold:
                saved_bits_2 += 2
                columns_2 += 1
            if s.segments_to_save_3_bits <= threshold:
                saved_bits_3 += 3
                columns_3 += 1
        saved_kib_1 = segments * saved_bits_1 / 8.0 / 1024.0
        saved_kib_2 = segments * saved_bits_2 / 8.0 / 1024.0
        saved_kib_3 = segments * saved_bits_3 / 8.0 / 1024.0
        print(
            f"  <= {threshold:g} offending segments: "
            f"1-bit {saved_bits_1:4d} bits/seg across {columns_1:2d} cols = {saved_kib_1:.3f} KiB; "
            f"2-bit {saved_bits_2:4d} bits/seg across {columns_2:2d} cols = {saved_kib_2:.3f} KiB; "
            f"3-bit {saved_bits_3:4d} bits/seg across {columns_3:2d} cols = {saved_kib_3:.3f} KiB"
        )
    print()


def print_table(title: str, rows: list[ColumnOutlierStats], segments: int, limit: int) -> None:
    print(title)
    print("axis deg w w/o1/10/100/1000 n@w n>=w-1 n>=w-2 n>=w-3 max|q| p99.9|q| p99|q| top_segments(seg@year:q/w)")
    for s in rows[:limit]:
        top = "; ".join(f"{seg}@{year:+.0f}y:{q}/{w}" for seg, year, q, w in s.top_segments)
        print(
            f"{AXIS_NAMES[s.axis]:>4} {s.degree:>3d} {s.width:>2d} "
            f"{s.width_without_1:>2d}/{s.width_without_10:>2d}/{s.width_without_100:>2d}/{s.width_without_1000:>2d} "
            f"{s.count_at_width:>7,d} {s.count_ge_width_minus_1:>8,d} {s.count_ge_width_minus_2:>8,d} {s.count_ge_width_minus_3:>8,d} "
            f"{s.q_abs_max:>12,d} {s.q_abs_p999:>11.0f} {s.q_abs_p99:>9.0f} "
            f"{top}"
        )
    print()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cache", type=Path, help="residual coefficient .npz cache")
    parser.add_argument("--top", type=int, default=30, help="number of columns per table")
    parser.add_argument("--top-segments", type=int, default=3, help="top segment examples per column")
    parser.add_argument("--rare-thresholds", default="1,10,100,1000,10000", help="comma-separated n@width thresholds for summary")
    args = parser.parse_args(argv)

    meta, boundaries, qcoeffs = load_cache(args.cache)
    segments = qcoeffs.shape[0]
    years_start = (float(boundaries[0]) - JD_J2000) / DAYS_PER_JULIAN_YEAR
    years_end = (float(boundaries[-1]) - JD_J2000) / DAYS_PER_JULIAN_YEAR
    stats = analyze_columns(qcoeffs, boundaries, args.top_segments)
    width_sum = sum(s.width for s in stats)
    payload_kib = math.ceil(segments * width_sum / 8) / 1024.0

    print(f"cache: {args.cache}")
    print(f"body: {meta['body']} center={meta['center']} method={meta['method']}")
    print(f"segments: {segments:,} residual_degree: {meta['residual_degree']} quant: {meta['quant_base_km']} km {meta['quant_pattern']}")
    print(f"years relative to J2000: {years_start:.1f} .. {years_end:.1f}")
    print(f"width sum: {width_sum:,} bits/segment  payload: {payload_kib:.3f} KiB")
    print()

    thresholds = tuple(int(x) for x in args.rare_thresholds.split(",") if x.strip())
    summarize_potential(stats, segments, thresholds)

    by_rare_max = sorted(
        stats,
        key=lambda s: (
            s.width_without_1000 < s.width,
            s.width_without_100 < s.width,
            s.width_without_10 < s.width,
            s.width_without_1 < s.width,
            s.width,
            -s.count_at_width,
        ),
        reverse=True,
    )
    print_table(f"top {min(args.top, len(stats))} columns where small tails affect width", by_rare_max, segments, args.top)

    by_width = sorted(stats, key=lambda s: (s.width, -s.count_at_width), reverse=True)
    print_table(f"top {min(args.top, len(stats))} columns by current width", by_width, segments, args.top)

    by_single_bit_cost = sorted(stats, key=lambda s: (s.count_at_width, -s.width, s.count_ge_width_minus_1))
    print_table(f"top {min(args.top, len(stats))} cheapest columns to lower by 1 bit", by_single_bit_cost, segments, args.top)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
