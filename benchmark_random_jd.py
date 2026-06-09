#!/usr/bin/env python3
"""Random JD accuracy and speed benchmark for OPM files.

This benchmark samples floating-point Julian Dates across the requested coverage,
reconstructs vectors from OPM, compares against DE441, and reports error
percentiles together with warm-cache evaluation timing.

It is a random-access benchmark, not a replacement for deterministic per-segment
validation: narrow pmax spikes can be missed by random samples.
"""
from __future__ import annotations

import argparse
import csv
import platform
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
from jplephem.spk import SPK

import opm_demo.moon_model as moon_model
import opm_demo.orbit_model as orbit_model
from opm_demo.format import STORAGE_EARTH_TO_MOON, STORAGE_SSB_TO_BODY, STORAGE_SUN_TO_BODY
from opm_demo.validator import (
    OpmFile,
    body_name_from_id,
    close_if_needed,
    read_opm,
    reconstruct_positions,
    truth_position_provider,
)

SSB_SUN_ANCHOR_BODIES = {"emb", "mars", "jupiter", "saturn", "uranus", "neptune", "pluto"}
NATIVE_ARCSEC_BODIES = {"mercury", "venus", "moon"}


@dataclass(frozen=True)
class Interval:
    start: float
    end: float

    @property
    def length(self) -> float:
        return self.end - self.start


@dataclass(frozen=True)
class BodyBenchmarkResult:
    body: str
    metric: str
    samples: int
    files: int
    segments: int
    size_kib: float
    jd_start: float
    jd_end: float
    p50: float
    p95: float
    p99: float
    pmax: float
    opm_ms: float
    opm_us_per_eval: float
    de441_ms: float
    de441_us_per_eval: float
    speedup: float


class MultiOpmBody:
    def __init__(self, body: str, opms: list[OpmFile]) -> None:
        if not opms:
            raise ValueError(f"no OPM files for {body}")
        self.body = body
        self.opms = sorted(opms, key=lambda p: (p.header.coverage_start_jd, p.path.as_posix()))
        self.intervals = [
            Interval(p.header.coverage_start_jd, p.header.coverage_start_jd + p.header.coverage_span_days)
            for p in self.opms
        ]
        self.segment_count = sum(p.header.segment_count for p in self.opms)
        self.size_kib = sum(p.header.file_size for p in self.opms) / 1024.0

    def assert_non_overlapping(self) -> None:
        prev: tuple[Interval, Path] | None = None
        for interval, opm in zip(self.intervals, self.opms):
            if prev is not None and interval.start < prev[0].end - 1e-9:
                raise ValueError(
                    f"overlapping {self.body} OPM coverages under the requested root:\n"
                    f"  {prev[1]} [{prev[0].start}, {prev[0].end}]\n"
                    f"  {opm.path} [{interval.start}, {interval.end}]\n"
                    "Pass a more specific --opm-root containing one candidate per body, or a non-overlapping shard set."
                )
            prev = (interval, opm.path)

    def eval(self, jds: np.ndarray) -> np.ndarray:
        jds = np.asarray(jds, dtype=np.float64)
        out = np.empty((len(jds), 3), dtype=np.float64)
        filled = np.zeros(len(jds), dtype=bool)
        for opm, interval in zip(self.opms, self.intervals):
            mask = (jds >= interval.start) & (jds <= interval.end) & ~filled
            if not np.any(mask):
                continue
            out[mask] = reconstruct_positions(opm, jds[mask])
            filled[mask] = True
        if not np.all(filled):
            missing = jds[~filled]
            raise ValueError(
                f"{self.body}: {len(missing)} sampled JDs are outside available OPM coverage; "
                f"first missing JD={missing[0]:.9f}"
            )
        return out

    def representative_opm_for(self, jd: float) -> OpmFile:
        for opm, interval in zip(self.opms, self.intervals):
            if interval.start <= jd <= interval.end:
                return opm
        raise ValueError(f"{self.body}: no OPM covers JD {jd}")


class MultiTruthBody:
    def __init__(self, spk: SPK, mopm: MultiOpmBody) -> None:
        self.providers: list[tuple[Interval, object]] = []
        self.closeables: list[object] = []
        for opm, interval in zip(mopm.opms, mopm.intervals):
            provider, closeable = truth_position_provider(spk, opm)
            self.providers.append((interval, provider))
            if closeable is not None:
                self.closeables.append(closeable)

    def close(self) -> None:
        for closeable in self.closeables:
            close_if_needed(closeable)
        self.closeables.clear()

    def eval(self, jds: np.ndarray) -> np.ndarray:
        jds = np.asarray(jds, dtype=np.float64)
        out = np.empty((len(jds), 3), dtype=np.float64)
        filled = np.zeros(len(jds), dtype=bool)
        for interval, provider in self.providers:
            mask = (jds >= interval.start) & (jds <= interval.end) & ~filled
            if not np.any(mask):
                continue
            out[mask] = provider.position(jds[mask])
            filled[mask] = True
        if not np.all(filled):
            missing = jds[~filled]
            raise ValueError(f"truth provider missing JD {missing[0]:.9f}")
        return out


def iter_opm_paths(root: Path) -> list[Path]:
    if root.is_file():
        return [root]
    paths: list[Path] = []
    for path in sorted(root.rglob("*.opm")):
        rel = path.relative_to(root)
        if any(part.startswith(".") for part in rel.parts[:-1]):
            continue
        paths.append(path)
    return paths


def percentile(values: np.ndarray, q: float) -> float:
    return float(np.percentile(values, q))


def km_errors(truth: np.ndarray, recon: np.ndarray) -> np.ndarray:
    return np.linalg.norm(truth - recon, axis=1)


def angular_errors_arcsec(truth: np.ndarray, recon: np.ndarray) -> np.ndarray:
    return orbit_model.angular_errors_arcsec(truth, recon)


def merge_intervals(intervals: Iterable[Interval]) -> list[Interval]:
    sorted_intervals = sorted((i for i in intervals if i.end > i.start), key=lambda x: x.start)
    if not sorted_intervals:
        return []
    merged = [sorted_intervals[0]]
    for cur in sorted_intervals[1:]:
        last = merged[-1]
        if cur.start <= last.end + 1e-9:
            merged[-1] = Interval(last.start, max(last.end, cur.end))
        else:
            merged.append(cur)
    return merged


def intersect_two_interval_lists(a: list[Interval], b: list[Interval]) -> list[Interval]:
    out: list[Interval] = []
    ia = ib = 0
    while ia < len(a) and ib < len(b):
        start = max(a[ia].start, b[ib].start)
        end = min(a[ia].end, b[ib].end)
        if end > start:
            out.append(Interval(start, end))
        if a[ia].end < b[ib].end:
            ia += 1
        else:
            ib += 1
    return out


def clip_intervals(intervals: list[Interval], jd_start: float | None, jd_end: float | None) -> list[Interval]:
    out = intervals
    if jd_start is not None:
        out = [Interval(max(i.start, jd_start), i.end) for i in out]
    if jd_end is not None:
        out = [Interval(i.start, min(i.end, jd_end)) for i in out]
    return [i for i in out if i.end > i.start]


def sample_from_intervals(intervals: list[Interval], samples: int, rng: np.random.Generator, mode: str) -> np.ndarray:
    if samples <= 0:
        raise ValueError("--samples must be positive")
    lengths = np.asarray([i.length for i in intervals], dtype=np.float64)
    total = float(lengths.sum())
    if total <= 0.0:
        raise ValueError("empty benchmark interval")
    if mode == "stratified":
        u = (np.arange(samples, dtype=np.float64) + rng.random(samples)) / samples * total
    elif mode == "uniform":
        u = rng.random(samples) * total
    else:
        raise ValueError(f"unknown sampling mode {mode}")
    cumulative = np.cumsum(lengths)
    idx = np.searchsorted(cumulative, u, side="right")
    prev = np.concatenate(([0.0], cumulative[:-1]))
    starts = np.asarray([i.start for i in intervals], dtype=np.float64)
    return starts[idx] + (u - prev[idx])


def best_timing_ms(fn: Callable[[], object], repeats: int) -> float:
    best = float("inf")
    for _ in range(max(1, repeats)):
        start = time.perf_counter()
        fn()
        elapsed = (time.perf_counter() - start) * 1000.0
        best = min(best, elapsed)
    return best


def resolve_metric(body: str, requested: str, has_sun: bool) -> str:
    if requested != "auto":
        return requested
    if body == "sun":
        return "km"
    if body in SSB_SUN_ANCHOR_BODIES and has_sun:
        return "composite-arcsec"
    return "native-arcsec"


def benchmark_body(
    *,
    body: str,
    mopm: MultiOpmBody,
    sun_opm: MultiOpmBody | None,
    spk: SPK,
    samples: int,
    rng: np.random.Generator,
    sampling: str,
    metric: str,
    jd_start: float | None,
    jd_end: float | None,
    timing_repeats: int,
) -> BodyBenchmarkResult:
    intervals = merge_intervals(mopm.intervals)
    if metric == "composite-arcsec":
        if sun_opm is None:
            raise ValueError(f"{body}: composite metric requires a Sun OPM in --opm-root")
        intervals = intersect_two_interval_lists(intervals, merge_intervals(sun_opm.intervals))
    intervals = clip_intervals(intervals, jd_start, jd_end)
    if not intervals:
        raise ValueError(f"{body}: no OPM coverage remains after applying requested JD range")

    jds = sample_from_intervals(intervals, samples, rng, sampling)
    sampled_start = float(np.min(jds))
    sampled_end = float(np.max(jds))

    truth = MultiTruthBody(spk, mopm)
    sun_truth = MultiTruthBody(spk, sun_opm) if metric == "composite-arcsec" and sun_opm is not None else None
    try:
        opm_vec = mopm.eval(jds)
        truth_vec = truth.eval(jds)
        if metric == "composite-arcsec":
            assert sun_opm is not None and sun_truth is not None
            opm_vec = opm_vec - sun_opm.eval(jds)
            truth_vec = truth_vec - sun_truth.eval(jds)
            errors = angular_errors_arcsec(truth_vec, opm_vec)
        elif metric == "native-arcsec":
            errors = angular_errors_arcsec(truth_vec, opm_vec)
        elif metric == "km":
            errors = km_errors(truth_vec, opm_vec)
        else:
            raise ValueError(f"unknown metric {metric}")

        def opm_eval() -> object:
            if metric == "composite-arcsec":
                assert sun_opm is not None
                return mopm.eval(jds) - sun_opm.eval(jds)
            return mopm.eval(jds)

        def de441_eval() -> object:
            if metric == "composite-arcsec":
                assert sun_truth is not None
                return truth.eval(jds) - sun_truth.eval(jds)
            return truth.eval(jds)

        # Warm cache before the measured loops.
        opm_eval()
        de441_eval()
        opm_ms = best_timing_ms(opm_eval, timing_repeats)
        de441_ms = best_timing_ms(de441_eval, timing_repeats)
    finally:
        truth.close()
        if sun_truth is not None:
            sun_truth.close()

    speedup = de441_ms / opm_ms if opm_ms > 0.0 else float("inf")
    return BodyBenchmarkResult(
        body=body,
        metric=metric,
        samples=samples,
        files=len(mopm.opms) + (len(sun_opm.opms) if metric == "composite-arcsec" and sun_opm is not None else 0),
        segments=mopm.segment_count,
        size_kib=mopm.size_kib,
        jd_start=sampled_start,
        jd_end=sampled_end,
        p50=percentile(errors, 50.0),
        p95=percentile(errors, 95.0),
        p99=percentile(errors, 99.0),
        pmax=float(np.max(errors)),
        opm_ms=opm_ms,
        opm_us_per_eval=opm_ms * 1000.0 / samples,
        de441_ms=de441_ms,
        de441_us_per_eval=de441_ms * 1000.0 / samples,
        speedup=speedup,
    )


def parse_body_filter(args: argparse.Namespace) -> set[str] | None:
    requested: list[str] = []
    for item in args.body or []:
        requested.extend(part.strip().lower() for part in item.split(",") if part.strip())
    if args.bodies:
        requested.extend(part.strip().lower() for part in args.bodies.split(",") if part.strip())
    if not requested or "all" in requested:
        return None
    return set(requested)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Random JD OPM accuracy and speed benchmark")
    parser.add_argument("--de441", type=Path, required=True, help="path to DE441 BSP file")
    parser.add_argument("--opm-root", type=Path, required=True, help="OPM file or directory containing .opm files")
    parser.add_argument("--samples", type=int, default=10000, help="number of random JD samples per body")
    parser.add_argument("--seed", type=int, default=1, help="random seed")
    parser.add_argument("--sampling", choices=["stratified", "uniform"], default="stratified", help="JD sampling method")
    parser.add_argument("--jd-start", type=float, default=None, help="optional benchmark range start JD")
    parser.add_argument("--jd-end", type=float, default=None, help="optional benchmark range end JD")
    parser.add_argument("--days", type=float, default=None, help="optional benchmark range span in days; requires --jd-start")
    parser.add_argument("--body", action="append", help="body to benchmark; may be repeated or comma-separated")
    parser.add_argument("--bodies", help="comma-separated bodies to benchmark, or all")
    parser.add_argument(
        "--metric",
        choices=["auto", "native-arcsec", "composite-arcsec", "km"],
        default="auto",
        help="accuracy metric; auto uses km for Sun, composite arcsec for SSB bodies when Sun is available, native arcsec otherwise",
    )
    parser.add_argument("--timing-repeats", type=int, default=3, help="timing repeats; best warm-cache time is reported")
    parser.add_argument("--no-crc", action="store_true", help="skip CRC64 validation when reading OPM files")
    parser.add_argument("--csv", type=Path, help="optional CSV output path")
    return parser.parse_args()


def print_results(results: list[BodyBenchmarkResult]) -> None:
    print(
        "body metric samples files segments size_KiB jd_min jd_max "
        "p50 p95 p99 pmax opm_ms opm_us de441_ms de441_us speedup"
    )
    for r in results:
        print(
            f"{r.body} {r.metric} {r.samples} {r.files} {r.segments} {r.size_kib:.3f} "
            f"{r.jd_start:.6f} {r.jd_end:.6f} "
            f"{r.p50:.9g} {r.p95:.9g} {r.p99:.9g} {r.pmax:.9g} "
            f"{r.opm_ms:.3f} {r.opm_us_per_eval:.3f} "
            f"{r.de441_ms:.3f} {r.de441_us_per_eval:.3f} {r.speedup:.3f}"
        )


def write_csv(path: Path, results: list[BodyBenchmarkResult]) -> None:
    fieldnames = list(BodyBenchmarkResult.__dataclass_fields__.keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            writer.writerow({name: getattr(result, name) for name in fieldnames})


def main() -> int:
    args = parse_args()
    if args.days is not None:
        if args.jd_start is None:
            raise SystemExit("--days requires --jd-start")
        args.jd_end = args.jd_start + args.days
    if args.jd_start is not None and args.jd_end is not None and args.jd_end <= args.jd_start:
        raise SystemExit("--jd-end must be greater than --jd-start")

    orbit_model.set_de441_path(args.de441)
    moon_model.set_de441_path(args.de441)

    paths = iter_opm_paths(args.opm_root)
    if not paths:
        raise SystemExit(f"no .opm files found under {args.opm_root}")

    by_body_paths: dict[str, list[Path]] = defaultdict(list)
    for path in paths:
        opm = read_opm(path, check_crc=not args.no_crc)
        by_body_paths[body_name_from_id(opm.header.body_id)].append(path)

    body_filter = parse_body_filter(args)
    loaded: dict[str, MultiOpmBody] = {}
    for body, body_paths in sorted(by_body_paths.items()):
        need_sun_for_filtered_composite = body == "sun" and args.metric in {"auto", "composite-arcsec"}
        if body_filter is not None and body not in body_filter and not need_sun_for_filtered_composite:
            continue
        opms = [read_opm(path, check_crc=not args.no_crc) for path in sorted(body_paths)]
        mopm = MultiOpmBody(body, opms)
        mopm.assert_non_overlapping()
        loaded[body] = mopm

    selected_bodies = sorted(body for body in loaded if body != "sun")
    if body_filter is not None:
        selected_bodies = sorted(body for body in body_filter if body in loaded and body != "sun")
        missing = sorted(body for body in body_filter if body not in loaded)
        if missing:
            raise SystemExit(f"requested bodies not found in --opm-root: {', '.join(missing)}")
    if body_filter is not None and "sun" in body_filter and "sun" in loaded:
        selected_bodies = ["sun"] + selected_bodies
    elif body_filter is None and "sun" in loaded:
        selected_bodies = ["sun"] + selected_bodies
    if not selected_bodies:
        raise SystemExit("no benchmarkable bodies selected")

    print(f"# Random JD Accuracy and Speed Benchmark")
    print(f"# opm_root={args.opm_root}")
    print(f"# de441={args.de441}")
    print(f"# samples={args.samples} seed={args.seed} sampling={args.sampling} timing_repeats={args.timing_repeats}")
    if args.jd_start is not None or args.jd_end is not None:
        print(f"# requested_range=[{args.jd_start}, {args.jd_end}]")
    print(f"# python={platform.python_version()} platform={platform.platform()} machine={platform.machine()}")
    print("# metric units: km for km, arcsec for native-arcsec/composite-arcsec")

    rng = np.random.default_rng(args.seed)
    results: list[BodyBenchmarkResult] = []
    with SPK.open(str(args.de441)) as spk:
        for body in selected_bodies:
            metric = resolve_metric(body, args.metric, "sun" in loaded)
            sun_opm = loaded.get("sun") if metric == "composite-arcsec" else None
            result = benchmark_body(
                body=body,
                mopm=loaded[body],
                sun_opm=sun_opm,
                spk=spk,
                samples=args.samples,
                rng=rng,
                sampling=args.sampling,
                metric=metric,
                jd_start=args.jd_start,
                jd_end=args.jd_end,
                timing_repeats=args.timing_repeats,
            )
            results.append(result)

    print_results(results)
    if args.csv:
        write_csv(args.csv, results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
