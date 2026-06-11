#!/usr/bin/env python3
"""Dense native OPM residual diagnostics with Kammeyer-style SVG plots.

This script compares each OPM file in its native storage-vector convention
against DE441.  It reports position residual magnitudes in km and analytic
velocity residual magnitudes derived from the OPM Chebyshev model.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import sys
import xml.sax.saxutils as xml_escape

import numpy as np
from jplephem.spk import SPK

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import opm_demo.moon_model as moon_model  # noqa: E402
import opm_demo.orbit_model as orbit_model  # noqa: E402
from opm_demo import validator  # noqa: E402

AU_KM = 149597870.7
SECONDS_PER_DAY = 86400.0
AXIS_COUNT = 3
DEFAULT_BODIES = "sun,mercury,venus,emb,moon,mars,jupiter,saturn,uranus,neptune,pluto"
BODY_LABELS = {
    "sun": "Sun",
    "mercury": "Mercury",
    "venus": "Venus",
    "emb": "EMB",
    "moon": "Moon",
    "mars": "Mars",
    "jupiter": "Jupiter",
    "saturn": "Saturn",
    "uranus": "Uranus",
    "neptune": "Neptune",
    "pluto": "Pluto",
}


@dataclass(frozen=True)
class ResidualSummary:
    samples: int
    pos_p50_km: float
    pos_p95_km: float
    pos_p99_km: float
    pos_max_km: float
    pos_worst_jd: float
    vel_p50_mm_s: float
    vel_p95_mm_s: float
    vel_p99_mm_s: float
    vel_max_mm_s: float
    vel_worst_jd: float
    vel_max_au_day: float


def pct(values: np.ndarray, q: float) -> float:
    return float(np.percentile(values, q))


def downsample(jds: np.ndarray, values: np.ndarray, max_points: int) -> tuple[np.ndarray, np.ndarray]:
    if len(jds) <= max_points:
        return jds, values
    idx = np.linspace(0, len(jds) - 1, max_points).round().astype(np.int64)
    idx = np.unique(idx)
    return jds[idx], values[idx]


def polyline(points: list[tuple[float, float]]) -> str:
    return " ".join(f"{x:.2f},{y:.2f}" for x, y in points)


def write_svg(path: Path, body: str, jds: np.ndarray, values: np.ndarray, *, title: str, y_label: str, max_points: int) -> None:
    width, height = 920, 520
    left, right, top, bottom = 112, 32, 60, 86
    plot_w = width - left - right
    plot_h = height - top - bottom
    x0, x1 = float(jds[0]), float(jds[-1])
    ymax = max(float(np.max(values)), 1e-15) * 1.08

    def sx(x: float) -> float:
        return left + (x - x0) / (x1 - x0) * plot_w

    def sy(y: float) -> float:
        return top + (1.0 - y / ymax) * plot_h

    sjd, sval = downsample(jds, values, max_points)
    pts = [(sx(float(x)), sy(float(y))) for x, y in zip(sjd, sval)]
    y_ticks = [0.0, ymax / 4, ymax / 2, 3 * ymax / 4, ymax]
    x_ticks = [x0 + (x1 - x0) * f for f in (0, 0.25, 0.5, 0.75, 1.0)]

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width/2:.1f}" y="24" font-family="Times New Roman" font-size="24" text-anchor="middle">{xml_escape.escape(title)}</text>',
        f'<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" stroke="#333"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" stroke="#333"/>',
    ]
    for t in y_ticks:
        y = sy(t)
        lines.append(f'<line x1="{left}" y1="{y:.2f}" x2="{left + plot_w}" y2="{y:.2f}" stroke="#ddd"/>')
        lines.append(f'<text x="{left - 8}" y="{y + 4:.2f}" font-family="Times New Roman" font-size="17" text-anchor="end">{t:.4g}</text>')
    for t in x_ticks:
        x = sx(t)
        lines.append(f'<line x1="{x:.2f}" y1="{top + plot_h}" x2="{x:.2f}" y2="{top + plot_h + 5}" stroke="#333"/>')
        lines.append(f'<text x="{x:.2f}" y="{top + plot_h + 22}" font-family="Times New Roman" font-size="17" text-anchor="middle">{t:.0f}</text>')
    lines.extend([
        f'<polyline points="{polyline(pts)}" fill="none" stroke="#000" stroke-width="1.5" opacity="0.9"/>',
        f'<text x="{left + plot_w/2:.1f}" y="{height - 18}" font-family="Times New Roman" font-size="19" text-anchor="middle">Julian Date</text>',
        f'<text transform="translate(18,{top + plot_h/2:.1f}) rotate(-90)" font-family="Times New Roman" font-size="19" text-anchor="middle">{xml_escape.escape(y_label)}</text>',
        '</svg>',
    ])
    path.write_text("\n".join(lines) + "\n")


def summarize(jds: np.ndarray, pos_err_km: np.ndarray, vel_err_km_day: np.ndarray) -> ResidualSummary:
    vel_mm_s = vel_err_km_day * 1_000_000.0 / SECONDS_PER_DAY
    vel_au_day = vel_err_km_day / AU_KM
    pos_i = int(np.argmax(pos_err_km))
    vel_i = int(np.argmax(vel_mm_s))
    return ResidualSummary(
        samples=len(jds),
        pos_p50_km=pct(pos_err_km, 50),
        pos_p95_km=pct(pos_err_km, 95),
        pos_p99_km=pct(pos_err_km, 99),
        pos_max_km=float(np.max(pos_err_km)),
        pos_worst_jd=float(jds[pos_i]),
        vel_p50_mm_s=pct(vel_mm_s, 50),
        vel_p95_mm_s=pct(vel_mm_s, 95),
        vel_p99_mm_s=pct(vel_mm_s, 99),
        vel_max_mm_s=float(np.max(vel_mm_s)),
        vel_worst_jd=float(jds[vel_i]),
        vel_max_au_day=float(np.max(vel_au_day)),
    )


def compute_body_residuals(
    spk: SPK,
    opm: validator.OpmFile,
    *,
    nodes_per_segment: int,
    segment_chunk_size: int,
    include_endpoints: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    coeffs = opm.qcoeffs.astype(np.float64) * opm.quant_steps[None, None, :]
    dcoeffs = validator.cheb_derivative_coeffs(coeffs)
    clock = validator.mercury_clock(opm) or validator.moon_clock(opm)
    params = validator.frame_params_for_segments(opm, clock) if opm.frame_coeffs is not None else None
    provider, closeable = validator.truth_position_provider(spk, opm)
    jds_parts: list[np.ndarray] = []
    pos_parts: list[np.ndarray] = []
    vel_parts: list[np.ndarray] = []
    try:
        for start in range(0, opm.header.segment_count, segment_chunk_size):
            stop = min(start + segment_chunk_size, opm.header.segment_count)
            jds, pos_err, vel_err = validator.native_residual_segment_chunk(
                provider,
                opm,
                coeffs,
                dcoeffs,
                params,
                clock,
                start,
                stop,
                nodes_per_segment,
                include_endpoints=include_endpoints,
            )
            if len(jds):
                jds_parts.append(jds)
                pos_parts.append(pos_err)
                vel_parts.append(vel_err)
    finally:
        validator.close_if_needed(closeable)
    if not jds_parts:
        raise ValueError(f"{opm.path}: no diagnostic samples inside coverage")
    return np.concatenate(jds_parts), np.concatenate(pos_parts), np.concatenate(vel_parts)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Dense native OPM residual diagnostics against DE441")
    p.add_argument("--de441", type=Path, required=True)
    p.add_argument("--opm-root", type=Path, required=True)
    p.add_argument("--nodes-per-segment", type=int, default=512)
    p.add_argument("--bodies", default=DEFAULT_BODIES)
    p.add_argument("--segment-chunk-size", type=int, default=4096)
    p.add_argument("--include-endpoints", action="store_true", default=True, help="include segment endpoints in addition to Chebyshev nodes")
    p.add_argument("--no-endpoints", dest="include_endpoints", action="store_false", help="use only Chebyshev center nodes")
    p.add_argument("--plot-dir", type=Path)
    p.add_argument("--plot-max-points", type=int, default=6000)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    orbit_model.set_de441_path(args.de441)
    moon_model.set_de441_path(args.de441)
    bodies = [b.strip().lower() for b in args.bodies.split(",") if b.strip()]
    if args.plot_dir is not None:
        args.plot_dir.mkdir(parents=True, exist_ok=True)
    print(f"# opm_root={args.opm_root}")
    print(f"# de441={args.de441} nodes_per_segment={args.nodes_per_segment} include_endpoints={args.include_endpoints}")
    print("body samples pos_p50_km pos_p95_km pos_p99_km pos_max_km pos_worst_jd vel_p50_mm_s vel_p95_mm_s vel_p99_mm_s vel_max_mm_s vel_worst_jd vel_max_au_day")
    with SPK.open(str(args.de441)) as spk:
        for body in bodies:
            opm = validator.read_opm(args.opm_root / f"{body}.opm")
            jds, pos_err, vel_err = compute_body_residuals(
                spk,
                opm,
                nodes_per_segment=args.nodes_per_segment,
                segment_chunk_size=args.segment_chunk_size,
                include_endpoints=args.include_endpoints,
            )
            order = np.argsort(jds)
            jds = jds[order]
            pos_err = pos_err[order]
            vel_err = vel_err[order]
            s = summarize(jds, pos_err, vel_err)
            print(
                f"{body} {s.samples} "
                f"{s.pos_p50_km:.9g} {s.pos_p95_km:.9g} {s.pos_p99_km:.9g} {s.pos_max_km:.9g} {s.pos_worst_jd:.9f} "
                f"{s.vel_p50_mm_s:.9g} {s.vel_p95_mm_s:.9g} {s.vel_p99_mm_s:.9g} {s.vel_max_mm_s:.9g} {s.vel_worst_jd:.9f} {s.vel_max_au_day:.9g}",
                flush=True,
            )
            if args.plot_dir is not None:
                label = BODY_LABELS.get(body, body.title())
                write_svg(
                    args.plot_dir / f"{body}-native-position-km.svg",
                    body,
                    jds,
                    pos_err,
                    title=f"{label} native position residual",
                    y_label="position residual (km)",
                    max_points=args.plot_max_points,
                )
                write_svg(
                    args.plot_dir / f"{body}-native-velocity-mm-s.svg",
                    body,
                    jds,
                    vel_err * 1_000_000.0 / SECONDS_PER_DAY,
                    title=f"{label} native velocity residual",
                    y_label="velocity residual (mm/s)",
                    max_points=args.plot_max_points,
                )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
