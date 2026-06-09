#!/usr/bin/env python3
"""Generate body-packed PEF files with separate reference-shape and residual stages."""
from __future__ import annotations

import argparse
import concurrent.futures
import math
import subprocess
import sys
from pathlib import Path

from jplephem.spk import SPK

import pef_demo.moon_model as moon_model
import pef_demo.orbit_model as orbit_model
from pef_demo.body_configs import BodyConfig, DEFAULT_BODY_ORDER
from pef_demo.body_packed_configs import BODY_PACKED_CONFIGS
from pef_demo.format import CENTURY_DAYS, JD_J2000, SPK_TARGET_IDS
from pef_demo.generator import expanded_bounds, fixed_bounds, mercury_clock, moon_century_clock, source_bounds, write_pef_file
from pef_demo.reference_shape import (
    fit_reference_shape,
    load_reference_shape_cache,
    save_reference_shape_cache,
)
from pef_demo.residual_fit import (
    fit_residual_coefficients,
    fit_residuals,
    load_residual_coeff_cache,
    pack_residual_coefficients,
    save_residual_coeff_cache,
)


def century_start(index: int) -> float:
    return JD_J2000 + index * CENTURY_DAYS


def safe_coverage_from_bounds(
    bounds: list[tuple[float, float]],
    cfg: BodyConfig,
    source_start: float,
    source_end: float,
) -> tuple[float, float]:
    eps = 1e-8
    safe: list[tuple[float, float]] = []
    min_start = source_start + cfg.edge_margin_days
    max_end = source_end - cfg.edge_margin_days
    for a, b in bounds:
        a = float(a)
        b = float(b)
        if b <= a + eps:
            continue
        expanded_start, expanded_end = expanded_bounds(a, b, cfg.segment_domain_expansion_fraction)
        if expanded_start < source_start - eps or expanded_end > source_end + eps:
            continue
        if a < min_start - eps or b > max_end + eps:
            continue
        safe.append((a, b))
    if not safe:
        raise ValueError(f"{cfg.body}: no complete safe segments inside source range")
    for prev, cur in zip(safe, safe[1:]):
        if abs(prev[1] - cur[0]) > eps:
            raise ValueError(f"{cfg.body}: non-contiguous safe segments: {prev[1]} -> {cur[0]}")
    start = safe[0][0]
    end = safe[-1][1]
    return float(start), float(end - start)


def venus_safe_coverage(cfg: BodyConfig, source_start: float, source_end: float, de441_path: Path) -> tuple[float, float]:
    orbit_model.set_de441_path(de441_path)
    provider = orbit_model.HelioProvider(SPK_TARGET_IDS["venus"])
    try:
        bounds = orbit_model.find_apsis_segments(provider, source_start, source_end, cfg.apsis_step_days, "mean-apsis")
    finally:
        provider.close()
    return safe_coverage_from_bounds(bounds, cfg, source_start, source_end)


def body_safe_coverage(body: str, source_start: float, source_end: float, de441_path: Path) -> tuple[float, float]:
    cfg = BODY_PACKED_CONFIGS[body]
    if cfg.segment_days is not None:
        return safe_coverage_from_bounds(fixed_bounds(source_start, source_end, cfg.segment_days), cfg, source_start, source_end)
    if body == "venus":
        return venus_safe_coverage(cfg, source_start, source_end, de441_path)
    if body == "mercury":
        return safe_coverage_from_bounds(mercury_clock().bounds(source_start, source_end), cfg, source_start, source_end)
    if body == "moon":
        return safe_coverage_from_bounds(moon_century_clock().bounds(source_start, source_end), cfg, source_start, source_end)
    raise ValueError(f"{body}: --full-source-safe is not implemented for this segment model")


def cache_safe_number(value: float) -> str:
    return f"{int(round(float(value) * 1_000_000.0)):+d}ud"


def reference_shape_cache_path(cache_root: Path, body: str, jd_start: float, jd_end: float, node_oversample: int) -> Path:
    name = f"{body}-{cache_safe_number(jd_start)}-{cache_safe_number(jd_end)}-n{int(node_oversample)}-reference-shape.npz"
    return cache_root / name


def residual_coeff_cache_path(cache_root: Path, body: str, jd_start: float, jd_end: float, node_oversample: int) -> Path:
    cfg = BODY_PACKED_CONFIGS[body]
    name = (
        f"{body}-{cache_safe_number(jd_start)}-{cache_safe_number(jd_end)}-"
        f"n{int(node_oversample)}-d{cfg.residual_degree}-residual-coeffs.npz"
    )
    return cache_root / name


def run_command(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, text=True, check=False)


def build_one(args: tuple[str, float, float, Path, Path, int, int, Path | None, bool, bool]) -> tuple[str, int, str]:
    body, jd_start, days, output_root, de441_path, node_oversample, chunk_size, cache_root, reuse_cache, force_cache = args
    orbit_model.set_de441_path(de441_path)
    moon_model.set_de441_path(de441_path)
    out = output_root / f"{body}.pef"
    jd_end = jd_start + days
    lines: list[str] = []
    with SPK.open(str(de441_path)) as spk:
        src_start, src_end = source_bounds(spk)
        shape = None
        cache_path = None
        if cache_root is not None:
            cache_path = reference_shape_cache_path(cache_root, body, jd_start, jd_end, node_oversample)
            if reuse_cache and not force_cache:
                shape = load_reference_shape_cache(cache_path, body, jd_start, jd_end, node_oversample)
                if shape is not None:
                    lines.append(f"loaded {body} reference shape cache: {cache_path}")
                else:
                    lines.append(f"reference shape cache miss: {cache_path}")
        if shape is None:
            lines.append(f"fitting {body} reference shape...")
            shape = fit_reference_shape(spk, body, jd_start, jd_end, node_oversample)
            if cache_path is not None:
                save_reference_shape_cache(cache_path, shape, jd_start=jd_start, jd_end=jd_end, node_oversample=node_oversample)
                lines.append(f"  saved reference shape cache: {cache_path}")
        lines.append(f"  shape segments={shape.segment_count}")
        if cache_root is None:
            lines.append(f"fitting {body} residuals...")
            packed = fit_residuals(spk, shape, jd_start, jd_end, node_oversample, chunk_size=chunk_size)
        else:
            coeff_cache_path = residual_coeff_cache_path(cache_root, body, jd_start, jd_end, node_oversample)
            coeff_fit = None
            if reuse_cache and not force_cache:
                coeff_fit = load_residual_coeff_cache(coeff_cache_path, shape, jd_start, jd_end, node_oversample)
                if coeff_fit is not None:
                    lines.append(f"loaded {body} residual coefficient cache: {coeff_cache_path}")
                else:
                    lines.append(f"residual coefficient cache miss: {coeff_cache_path}")
            if coeff_fit is None:
                chunk_note = f" chunk_size={chunk_size}" if chunk_size > 0 else ""
                lines.append(f"fitting {body} residual coefficients...{chunk_note}")
                coeff_fit = fit_residual_coefficients(spk, shape, jd_start, jd_end, node_oversample, chunk_size=chunk_size)
                save_residual_coeff_cache(coeff_cache_path, coeff_fit, jd_start=jd_start, jd_end=jd_end, node_oversample=node_oversample)
                lines.append(f"  saved residual coefficient cache: {coeff_cache_path}")
            lines.append(f"packing {body} residuals...")
            packed = pack_residual_coefficients(spk, shape, coeff_fit, jd_start, jd_end, node_oversample)
        size = write_pef_file(out, packed, src_start, src_end, jd_start, days)
    status = "PASS" if packed.max_err <= 0.001 or body == "sun" else "MISS"
    lines.append(
        f"  wrote {out} size={size / 1024.0:.3f} KiB segments={packed.segment_count} "
        f"p99={packed.p99:.6g}\" max={packed.max_err:.6g}\" {status}"
    )
    return body, 0 if status != "MISS" else 1, "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate one body-wide PEF file per body")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--body", choices=DEFAULT_BODY_ORDER)
    group.add_argument("--all", action="store_true", help="write all configured bodies")
    parser.add_argument("--de441", type=Path, required=True, help="path to DE441 BSP file")
    parser.add_argument("--output-root", type=Path, default=Path("out/body-packed/current"))
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--node-oversample", type=int, default=3)
    parser.add_argument("--jd-start", type=float, help="coverage start JD; overrides --start-index")
    parser.add_argument("--days", type=float, help="coverage span days; overrides --end-index")
    parser.add_argument("--start-index", type=int, default=0, help="first J2000-relative century index")
    parser.add_argument("--end-index", type=int, default=0, help="last J2000-relative century index")
    parser.add_argument("--full-source-safe", action="store_true", help="choose complete segments inside DE441 source coverage")
    parser.add_argument("--cache-root", type=Path, default=Path("out/body-packed/cache/default"), help="reference-shape and residual-coefficient cache directory")
    parser.add_argument("--reuse-cache", action="store_true", help="load/save reference-shape caches when available")
    parser.add_argument("--force-cache", action="store_true", help="recompute and overwrite reference-shape caches")
    parser.add_argument("--chunk-size", type=int, default=0, help="chunk size for vectorized residual coefficient fitting; 0 keeps the legacy loop")
    parser.add_argument("--validate", action="store_true", help="run validate_pef.py after generation")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    bodies = DEFAULT_BODY_ORDER if args.all else [args.body]
    if args.full_source_safe:
        if args.jd_start is not None or args.days is not None:
            raise SystemExit("--full-source-safe cannot be combined with --jd-start/--days")
        with SPK.open(str(args.de441)) as spk:
            source_start, source_end = source_bounds(spk)
        coverage_by_body = {body: body_safe_coverage(body, source_start, source_end, args.de441) for body in bodies}
    else:
        if args.jd_start is not None or args.days is not None:
            if args.jd_start is None or args.days is None:
                raise SystemExit("--jd-start and --days must be supplied together")
            jd_start = float(args.jd_start)
            days = float(args.days)
        else:
            jd_start = century_start(int(args.start_index))
            days = century_start(int(args.end_index) + 1) - jd_start
        coverage_by_body = {body: (jd_start, days) for body in bodies}

    if args.force_cache and not args.reuse_cache:
        raise SystemExit("--force-cache requires --reuse-cache")
    if args.chunk_size < 0:
        raise SystemExit("--chunk-size must be non-negative")

    args.output_root.mkdir(parents=True, exist_ok=True)
    cache_root = args.cache_root if args.reuse_cache else None
    tasks = [
        (
            body,
            start,
            span,
            args.output_root,
            args.de441,
            int(args.node_oversample),
            int(args.chunk_size),
            cache_root,
            bool(args.reuse_cache),
            bool(args.force_cache),
        )
        for body, (start, span) in coverage_by_body.items()
    ]

    if len(set(coverage_by_body.values())) == 1:
        only_start, only_days = next(iter(coverage_by_body.values()))
        print(f"body_packed bodies={len(tasks)} coverage={only_start:.9f}..{only_start + only_days:.9f}")
    else:
        print(f"body_packed bodies={len(tasks)} coverage=per-body-full-source-safe")
    failures = 0
    if args.jobs <= 1 or len(tasks) == 1:
        for task in tasks:
            body, code, output = build_one(task)
            print(output, flush=True)
            if code != 0:
                failures += 1
                print(f"FAIL {body}: exit={code}")
    else:
        with concurrent.futures.ProcessPoolExecutor(max_workers=args.jobs) as executor:
            for body, code, output in executor.map(build_one, tasks):
                print(output, flush=True)
                if code != 0:
                    failures += 1
                    print(f"FAIL {body}: exit={code}")

    if args.validate and failures == 0:
        validation = run_command([
            sys.executable,
            "validate_pef.py",
            "--de441",
            str(args.de441),
            "--progress",
            str(args.output_root),
        ])
        if validation.returncode != 0:
            failures += 1

    print(f"{('PASS' if failures == 0 else 'FAIL')}: {len(tasks) - failures}/{len(tasks)} body-packed files generated")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
