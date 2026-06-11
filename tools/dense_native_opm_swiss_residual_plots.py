#!/usr/bin/env python3
"""Dense native-vector residual plots comparing OPM and Swiss Ephemeris.

This is a diagnostic companion to dense_native_residual_diagnostics.py.  It
uses the OPM native storage-vector convention for each body, compares both OPM
and Swiss Ephemeris against DE441 on the same OPM segment grid, and writes
Kammeyer-style two-color SVG plots for position and velocity residuals.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import sys
import xml.sax.saxutils as xml_escape

import numpy as np
import swisseph as swe
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
DEFAULT_BODIES = "sun,moon,mercury,venus,mars,jupiter,saturn,uranus,neptune,pluto"
SWISS_BASE_FLAGS = swe.FLG_SWIEPH | swe.FLG_XYZ | swe.FLG_EQUATORIAL | swe.FLG_J2000 | swe.FLG_TRUEPOS | swe.FLG_ICRS | swe.FLG_SPEED
SWISS_BODIES = {
    "sun": swe.SUN,
    "moon": swe.MOON,
    "mercury": swe.MERCURY,
    "venus": swe.VENUS,
    "mars": swe.MARS,
    "jupiter": swe.JUPITER,
    "saturn": swe.SATURN,
    "uranus": swe.URANUS,
    "neptune": swe.NEPTUNE,
    "pluto": swe.PLUTO,
}
BODY_LABELS = {
    "sun": "Sun",
    "moon": "Moon",
    "mercury": "Mercury",
    "venus": "Venus",
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
    opm_pos_p99_km: float
    opm_pos_max_km: float
    swiss_pos_p99_km: float
    swiss_pos_max_km: float
    opm_vel_p99_mm_s: float
    opm_vel_max_mm_s: float
    swiss_vel_p99_mm_s: float
    swiss_vel_max_mm_s: float


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


def write_svg(
    path: Path,
    body: str,
    jds: np.ndarray,
    opm_values: np.ndarray,
    swiss_values: np.ndarray,
    *,
    title: str,
    y_label: str,
    max_points: int,
) -> None:
    width, height = 920, 520
    left, right, top, bottom = 112, 32, 60, 86
    plot_w = width - left - right
    plot_h = height - top - bottom
    x0, x1 = float(jds[0]), float(jds[-1])
    ymax = max(float(np.max(opm_values)), float(np.max(swiss_values)), 1e-15) * 1.08

    def sx(x: float) -> float:
        return left + (x - x0) / (x1 - x0) * plot_w

    def sy(y: float) -> float:
        return top + (1.0 - y / ymax) * plot_h

    sjd, sopm = downsample(jds, opm_values, max_points)
    _, sswiss = downsample(jds, swiss_values, max_points)
    opm_pts = [(sx(float(x)), sy(float(y))) for x, y in zip(sjd, sopm)]
    swiss_pts = [(sx(float(x)), sy(float(y))) for x, y in zip(sjd, sswiss)]
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
        f'<polyline points="{polyline(swiss_pts)}" fill="none" stroke="#d95f02" stroke-width="1.5" opacity="0.80"/>',
        f'<polyline points="{polyline(opm_pts)}" fill="none" stroke="#1b9e77" stroke-width="1.5" opacity="0.88"/>',
        f'<text x="{left + plot_w/2:.1f}" y="{height - 18}" font-family="Times New Roman" font-size="19" text-anchor="middle">Julian Date</text>',
        f'<text transform="translate(18,{top + plot_h/2:.1f}) rotate(-90)" font-family="Times New Roman" font-size="19" text-anchor="middle">{xml_escape.escape(y_label)}</text>',
        f'<line x1="{width - 190}" y1="{top + 8}" x2="{width - 156}" y2="{top + 8}" stroke="#1b9e77" stroke-width="2"/>',
        f'<text x="{width - 150}" y="{top + 12}" font-family="Times New Roman" font-size="16">OPM</text>',
        f'<line x1="{width - 190}" y1="{top + 28}" x2="{width - 156}" y2="{top + 28}" stroke="#d95f02" stroke-width="2"/>',
        f'<text x="{width - 150}" y="{top + 32}" font-family="Times New Roman" font-size="16">Swiss Ephemeris</text>',
        '</svg>',
    ])
    path.write_text("\n".join(lines) + "\n")


def swiss_native_state(body: str, jds: np.ndarray, storage_vector_id: int) -> tuple[np.ndarray, np.ndarray]:
    flags = SWISS_BASE_FLAGS
    if storage_vector_id == validator.STORAGE_SSB_TO_BODY:
        flags |= swe.FLG_BARYCTR
    elif storage_vector_id == validator.STORAGE_SUN_TO_BODY:
        flags |= swe.FLG_HELCTR
    elif storage_vector_id == validator.STORAGE_EARTH_TO_MOON:
        pass
    else:
        raise ValueError(f"unsupported native storage vector for Swiss comparison: {storage_vector_id}")

    swe_body = SWISS_BODIES[body]
    pos = np.empty((len(jds), AXIS_COUNT), dtype=np.float64)
    vel = np.empty((len(jds), AXIS_COUNT), dtype=np.float64)
    for i, jd in enumerate(jds):
        xx, retflag = swe.calc(float(jd), swe_body, flags)
        if retflag < 0:
            raise RuntimeError(f"Swiss calc failed {body} jd={jd}: {xx}")
        pos[i] = np.asarray(xx[:3], dtype=np.float64) * AU_KM
        vel[i] = np.asarray(xx[3:6], dtype=np.float64) * AU_KM
    return pos, vel


def compute_body_residuals(
    spk: SPK,
    opm: validator.OpmFile,
    body: str,
    *,
    nodes_per_segment: int,
    segment_chunk_size: int,
    include_endpoints: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    coeffs = opm.qcoeffs.astype(np.float64) * opm.quant_steps[None, None, :]
    dcoeffs = validator.cheb_derivative_coeffs(coeffs)
    clock = validator.mercury_clock(opm) or validator.moon_clock(opm)
    params = validator.frame_params_for_segments(opm, clock) if opm.frame_coeffs is not None else None
    provider, closeable = validator.truth_position_provider(spk, opm)
    jds_parts: list[np.ndarray] = []
    opm_pos_parts: list[np.ndarray] = []
    opm_vel_parts: list[np.ndarray] = []
    swiss_pos_parts: list[np.ndarray] = []
    swiss_vel_parts: list[np.ndarray] = []
    try:
        for start in range(0, opm.header.segment_count, segment_chunk_size):
            stop = min(start + segment_chunk_size, opm.header.segment_count)
            segment_indices, jds, a, b = validator.segment_chunk_nodes(
                opm,
                start,
                stop,
                nodes_per_segment,
                clock,
                include_endpoints=include_endpoints,
            )
            if len(segment_indices) == 0:
                continue
            width = b - a
            expanded_a = a - opm.header.expansion * width
            expanded_b = b + opm.header.expansion * width
            scale = 2.0 / (expanded_b - expanded_a)
            tau = (2.0 * jds - expanded_a[:, None] - expanded_b[:, None]) / (expanded_b - expanded_a)[:, None]
            flat_jds = jds.reshape(-1)
            opm_pos = validator.reconstruct_segment_nodes(opm, segment_indices, tau, coeffs, params).reshape((-1, AXIS_COUNT))
            opm_vel = validator.reconstruct_segment_node_velocities(opm, segment_indices, tau, dcoeffs, params, scale).reshape((-1, AXIS_COUNT))
            truth_pos = provider.position(flat_jds)
            truth_vel = provider.velocity(flat_jds)
            swiss_pos, swiss_vel = swiss_native_state(body, flat_jds, opm.header.storage_vector_id)
            jds_parts.append(flat_jds)
            opm_pos_parts.append(np.linalg.norm(opm_pos - truth_pos, axis=1))
            opm_vel_parts.append(np.linalg.norm(opm_vel - truth_vel, axis=1))
            swiss_pos_parts.append(np.linalg.norm(swiss_pos - truth_pos, axis=1))
            swiss_vel_parts.append(np.linalg.norm(swiss_vel - truth_vel, axis=1))
    finally:
        validator.close_if_needed(closeable)
    if not jds_parts:
        raise ValueError(f"{opm.path}: no diagnostic samples inside coverage")
    return (
        np.concatenate(jds_parts),
        np.concatenate(opm_pos_parts),
        np.concatenate(opm_vel_parts),
        np.concatenate(swiss_pos_parts),
        np.concatenate(swiss_vel_parts),
    )


def summarize(jds: np.ndarray, opm_pos: np.ndarray, opm_vel: np.ndarray, swiss_pos: np.ndarray, swiss_vel: np.ndarray) -> ResidualSummary:
    opm_vel_mm_s = opm_vel * 1_000_000.0 / SECONDS_PER_DAY
    swiss_vel_mm_s = swiss_vel * 1_000_000.0 / SECONDS_PER_DAY
    return ResidualSummary(
        samples=len(jds),
        opm_pos_p99_km=pct(opm_pos, 99),
        opm_pos_max_km=float(np.max(opm_pos)),
        swiss_pos_p99_km=pct(swiss_pos, 99),
        swiss_pos_max_km=float(np.max(swiss_pos)),
        opm_vel_p99_mm_s=pct(opm_vel_mm_s, 99),
        opm_vel_max_mm_s=float(np.max(opm_vel_mm_s)),
        swiss_vel_p99_mm_s=pct(swiss_vel_mm_s, 99),
        swiss_vel_max_mm_s=float(np.max(swiss_vel_mm_s)),
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Dense native-vector residual plots for OPM vs Swiss Ephemeris")
    p.add_argument("--de441", type=Path, required=True)
    p.add_argument("--opm-root", type=Path, required=True)
    p.add_argument("--swiss-ephe", type=Path, required=True)
    p.add_argument("--nodes-per-segment", type=int, default=512)
    p.add_argument("--bodies", default=DEFAULT_BODIES)
    p.add_argument("--segment-chunk-size", type=int, default=4096)
    p.add_argument("--include-endpoints", action="store_true", default=True)
    p.add_argument("--no-endpoints", dest="include_endpoints", action="store_false")
    p.add_argument("--plot-dir", type=Path, required=True)
    p.add_argument("--plot-max-points", type=int, default=6000)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    swe.set_ephe_path(str(args.swiss_ephe))
    orbit_model.set_de441_path(args.de441)
    moon_model.set_de441_path(args.de441)
    args.plot_dir.mkdir(parents=True, exist_ok=True)
    bodies = [b.strip().lower() for b in args.bodies.split(",") if b.strip()]
    print(f"# opm_root={args.opm_root}")
    print(f"# swiss_ephe={args.swiss_ephe} flags={SWISS_BASE_FLAGS}")
    print(f"# de441={args.de441} nodes_per_segment={args.nodes_per_segment} include_endpoints={args.include_endpoints}")
    print("body samples opm_pos_p99_km opm_pos_max_km swiss_pos_p99_km swiss_pos_max_km opm_vel_p99_mm_s opm_vel_max_mm_s swiss_vel_p99_mm_s swiss_vel_max_mm_s")
    with SPK.open(str(args.de441)) as spk:
        for body in bodies:
            opm = validator.read_opm(args.opm_root / f"{body}.opm")
            jds, opm_pos, opm_vel, swiss_pos, swiss_vel = compute_body_residuals(
                spk,
                opm,
                body,
                nodes_per_segment=args.nodes_per_segment,
                segment_chunk_size=args.segment_chunk_size,
                include_endpoints=args.include_endpoints,
            )
            order = np.argsort(jds)
            jds = jds[order]
            opm_pos = opm_pos[order]
            opm_vel = opm_vel[order]
            swiss_pos = swiss_pos[order]
            swiss_vel = swiss_vel[order]
            s = summarize(jds, opm_pos, opm_vel, swiss_pos, swiss_vel)
            print(
                f"{body} {s.samples} "
                f"{s.opm_pos_p99_km:.9g} {s.opm_pos_max_km:.9g} {s.swiss_pos_p99_km:.9g} {s.swiss_pos_max_km:.9g} "
                f"{s.opm_vel_p99_mm_s:.9g} {s.opm_vel_max_mm_s:.9g} {s.swiss_vel_p99_mm_s:.9g} {s.swiss_vel_max_mm_s:.9g}",
                flush=True,
            )
            label = BODY_LABELS.get(body, body.title())
            write_svg(
                args.plot_dir / f"{body}-native-position-opm-vs-swiss-km.svg",
                body,
                jds,
                opm_pos,
                swiss_pos,
                title=f"{label} native position residual",
                y_label="position residual (km)",
                max_points=args.plot_max_points,
            )
            write_svg(
                args.plot_dir / f"{body}-native-velocity-opm-vs-swiss-mm-s.svg",
                body,
                jds,
                opm_vel * 1_000_000.0 / SECONDS_PER_DAY,
                swiss_vel * 1_000_000.0 / SECONDS_PER_DAY,
                title=f"{label} native velocity residual",
                y_label="velocity residual (mm/s)",
                max_points=args.plot_max_points,
            )
    swe.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
