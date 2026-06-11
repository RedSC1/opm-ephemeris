#!/usr/bin/env python3
"""Dense OPM-vs-Swiss geocentric comparison with optional SVG plots.

Uses OPM segment boundaries as a deterministic dense JD grid and evaluates both
OPM and Swiss Ephemeris on the exact same points.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import sys
import math
import xml.sax.saxutils as xml_escape

import numpy as np
import swisseph as swe
from jplephem.spk import SPK

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import opm_demo.orbit_model as orbit_model  # noqa: E402
from opm_demo import validator  # noqa: E402

AXIS_COUNT = 3
SSB = 0
AU_KM = 149597870.7
DE441_EMRAT = 81.300568221497215
SWISS_FLAGS = swe.FLG_SWIEPH | swe.FLG_XYZ | swe.FLG_EQUATORIAL | swe.FLG_J2000 | swe.FLG_TRUEPOS | swe.FLG_ICRS

BODY_TARGETS = {
    "sun": 10,
    "moon": 301,
    "mercury": 199,
    "venus": 299,
    "mars": 4,
    "jupiter": 5,
    "saturn": 6,
    "uranus": 7,
    "neptune": 8,
    "pluto": 9,
}
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
GRID_SOURCE = {
    "sun": "sun",
    "moon": "moon",
    "mercury": "mercury",
    "venus": "venus",
    "mars": "mars",
    "jupiter": "jupiter",
    "saturn": "saturn",
    "uranus": "uranus",
    "neptune": "neptune",
    "pluto": "pluto",
}


@dataclass(frozen=True)
class BaryProvider:
    spk: SPK
    target_id: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "segments", sorted(
            [s for s in self.spk.segments if s.center == SSB and s.target == self.target_id],
            key=lambda s: s.start_jd,
        ))
        if not self.segments:
            raise RuntimeError(f"No SPK segment center=0 target={self.target_id}")

    def position(self, jd_arr: np.ndarray) -> np.ndarray:
        tdb = np.asarray(jd_arr, dtype=np.float64)
        out = np.full((len(tdb), AXIS_COUNT), np.nan, dtype=np.float64)
        for seg in self.segments:
            mask = (tdb >= seg.start_jd) & (tdb <= seg.end_jd)
            if np.any(mask):
                out[mask] = seg.compute(tdb[mask]).T
        if np.any(~np.isfinite(out[:, 0])):
            raise RuntimeError(f"Missing SPK coverage for target {self.target_id}")
        return out


@dataclass(frozen=True)
class RelativeProvider:
    spk: SPK
    center_id: int
    target_id: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "segments", sorted(
            [s for s in self.spk.segments if s.center == self.center_id and s.target == self.target_id],
            key=lambda s: s.start_jd,
        ))
        if not self.segments:
            raise RuntimeError(f"No SPK segment center={self.center_id} target={self.target_id}")

    def position(self, jd_arr: np.ndarray) -> np.ndarray:
        tdb = np.asarray(jd_arr, dtype=np.float64)
        out = np.full((len(tdb), AXIS_COUNT), np.nan, dtype=np.float64)
        for seg in self.segments:
            mask = (tdb >= seg.start_jd) & (tdb <= seg.end_jd)
            if np.any(mask):
                out[mask] = seg.compute(tdb[mask]).T
        if np.any(~np.isfinite(out[:, 0])):
            raise RuntimeError(f"Missing SPK coverage for center={self.center_id} target={self.target_id}")
        return out


def de441_barycentric(spk: SPK, target_id: int, jds: np.ndarray) -> np.ndarray:
    if target_id in {10, 3, 4, 5, 6, 7, 8, 9}:
        return BaryProvider(spk, target_id).position(jds)
    if target_id == 399:
        return BaryProvider(spk, 3).position(jds) + RelativeProvider(spk, 3, 399).position(jds)
    if target_id == 301:
        return BaryProvider(spk, 3).position(jds) + RelativeProvider(spk, 3, 301).position(jds)
    if target_id == 199:
        return BaryProvider(spk, 1).position(jds) + RelativeProvider(spk, 1, 199).position(jds)
    if target_id == 299:
        return BaryProvider(spk, 2).position(jds) + RelativeProvider(spk, 2, 299).position(jds)
    raise RuntimeError(f"unsupported DE441 target id {target_id}")


def de441_geocentric(spk: SPK, body: str, jds: np.ndarray) -> np.ndarray:
    earth = de441_barycentric(spk, 399, jds)
    target = de441_barycentric(spk, BODY_TARGETS[body], jds)
    return target - earth


def opm_barycentric(opms: dict[str, validator.OpmFile], jds: np.ndarray, body: str, emrat: float) -> np.ndarray:
    sun = validator.reconstruct_positions(opms["sun"], jds)
    if body == "sun":
        return sun
    if body == "mercury":
        return sun + validator.reconstruct_positions(opms["mercury"], jds)
    if body == "venus":
        return sun + validator.reconstruct_positions(opms["venus"], jds)
    if body == "emb":
        return validator.reconstruct_positions(opms["emb"], jds)
    if body == "earth":
        emb = validator.reconstruct_positions(opms["emb"], jds)
        moon_geo = validator.reconstruct_positions(opms["moon"], jds)
        return emb - moon_geo / (1.0 + emrat)
    if body == "moon":
        earth = opm_barycentric(opms, jds, "earth", emrat)
        return earth + validator.reconstruct_positions(opms["moon"], jds)
    if body in {"mars", "jupiter", "saturn", "uranus", "neptune", "pluto"}:
        return validator.reconstruct_positions(opms[body], jds)
    raise RuntimeError(f"unsupported OPM body {body}")


def opm_geocentric(opms: dict[str, validator.OpmFile], body: str, jds: np.ndarray, emrat: float) -> np.ndarray:
    if body == "moon":
        return validator.reconstruct_positions(opms["moon"], jds)
    earth = opm_barycentric(opms, jds, "earth", emrat)
    target = opm_barycentric(opms, jds, body, emrat)
    return target - earth


def swiss_geocentric(body: str, jds: np.ndarray) -> np.ndarray:
    swe_body = SWISS_BODIES[body]
    out = np.empty((len(jds), AXIS_COUNT), dtype=np.float64)
    for i, jd in enumerate(jds):
        xx, retflag = swe.calc(float(jd), swe_body, SWISS_FLAGS)
        if retflag < 0:
            raise RuntimeError(f"Swiss calc failed {body} jd={jd}: {xx}")
        out[i] = np.asarray(xx[:3], dtype=np.float64) * AU_KM
    return out


def angular_errors_arcsec(truth: np.ndarray, candidate: np.ndarray) -> np.ndarray:
    cross = np.linalg.norm(np.cross(truth, candidate), axis=1)
    dot = np.sum(truth * candidate, axis=1)
    return np.arctan2(cross, dot) * (180.0 / np.pi) * 3600.0


def pct(values: np.ndarray, q: float) -> float:
    return float(np.percentile(values, q))


def load_opms(root: Path) -> dict[str, validator.OpmFile]:
    out = {}
    for body in ["sun", "mercury", "venus", "emb", "mars", "jupiter", "saturn", "uranus", "neptune", "pluto", "moon"]:
        out[body] = validator.read_opm(root / f"{body}.opm")
    return out


def body_grid(opms: dict[str, validator.OpmFile], body: str, nodes_per_segment: int) -> np.ndarray:
    opm = opms[GRID_SOURCE[body]]
    clock = validator.mercury_clock(opm) or validator.moon_clock(opm)
    parts = []
    for seg in range(opm.header.segment_count):
        a, b = validator.segment_bounds(opm.header, seg, clock)
        lo = max(a, opm.header.coverage_start_jd)
        hi = min(b, opm.header.coverage_start_jd + opm.header.coverage_span_days)
        if hi <= lo:
            continue
        nodes = orbit_model.cheb_nodes(lo, hi, nodes_per_segment)
        nodes = np.concatenate([nodes, np.asarray([lo, hi], dtype=np.float64)])
        parts.append(nodes)
    return np.unique(np.sort(np.concatenate(parts)))


def summarize(err: np.ndarray) -> tuple[float, float, float, float, int]:
    imax = int(np.argmax(err))
    return pct(err, 50), pct(err, 95), pct(err, 99), float(np.max(err)), imax


def downsample(jds: np.ndarray, values: np.ndarray, max_points: int) -> tuple[np.ndarray, np.ndarray]:
    if len(jds) <= max_points:
        return jds, values
    idx = np.linspace(0, len(jds) - 1, max_points).round().astype(np.int64)
    idx = np.unique(idx)
    return jds[idx], values[idx]


def polyline(points: list[tuple[float, float]]) -> str:
    return " ".join(f"{x:.2f},{y:.2f}" for x, y in points)


def write_svg(path: Path, body: str, jds: np.ndarray, opm_err: np.ndarray, swiss_err: np.ndarray, max_points: int) -> None:
    width, height = 920, 520
    left, right, top, bottom = 112, 32, 60, 86
    plot_w = width - left - right
    plot_h = height - top - bottom
    x0, x1 = float(jds[0]), float(jds[-1])
    ymax = max(float(np.max(opm_err)), float(np.max(swiss_err)), 1e-9) * 1.08

    def sx(x: float) -> float:
        return left + (x - x0) / (x1 - x0) * plot_w

    def sy(y: float) -> float:
        return top + (1.0 - y / ymax) * plot_h

    sjd, sopm = downsample(jds, opm_err, max_points)
    _, ssw = downsample(jds, swiss_err, max_points)
    opm_pts = [(sx(float(x)), sy(float(y))) for x, y in zip(sjd, sopm)]
    sw_pts = [(sx(float(x)), sy(float(y))) for x, y in zip(sjd, ssw)]

    y_ticks = [0.0, ymax / 4, ymax / 2, 3 * ymax / 4, ymax]
    x_ticks = [x0 + (x1 - x0) * f for f in (0, 0.25, 0.5, 0.75, 1.0)]
    title = f"{BODY_LABELS.get(body, body)} dense geocentric angular error"
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width/2:.1f}" y="24" font-family="Arial" font-size="24" text-anchor="middle">{xml_escape.escape(title)}</text>',
        f'<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" stroke="#333"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" stroke="#333"/>',
    ]
    for t in y_ticks:
        y = sy(t)
        lines.append(f'<line x1="{left}" y1="{y:.2f}" x2="{left + plot_w}" y2="{y:.2f}" stroke="#ddd"/>')
        lines.append(f'<text x="{left - 8}" y="{y + 4:.2f}" font-family="Arial" font-size="17" text-anchor="end">{t:.4g}</text>')
    for t in x_ticks:
        x = sx(t)
        lines.append(f'<line x1="{x:.2f}" y1="{top + plot_h}" x2="{x:.2f}" y2="{top + plot_h + 5}" stroke="#333"/>')
        lines.append(f'<text x="{x:.2f}" y="{top + plot_h + 22}" font-family="Arial" font-size="17" text-anchor="middle">{t:.0f}</text>')
    lines.extend([
        f'<polyline points="{polyline(sw_pts)}" fill="none" stroke="#d95f02" stroke-width="1.6" opacity="0.85"/>',
        f'<polyline points="{polyline(opm_pts)}" fill="none" stroke="#1b9e77" stroke-width="1.6" opacity="0.9"/>',
        f'<text x="{left + plot_w/2:.1f}" y="{height - 18}" font-family="Arial" font-size="19" text-anchor="middle">Julian Date</text>',
        f'<text transform="translate(18,{top + plot_h/2:.1f}) rotate(-90)" font-family="Arial" font-size="19" text-anchor="middle">angular error (arcsec)</text>',
        f'<line x1="{width - 190}" y1="{top + 8}" x2="{width - 156}" y2="{top + 8}" stroke="#1b9e77" stroke-width="2"/>',
        f'<text x="{width - 150}" y="{top + 12}" font-family="Arial" font-size="16">OPM</text>',
        f'<line x1="{width - 190}" y1="{top + 28}" x2="{width - 156}" y2="{top + 28}" stroke="#d95f02" stroke-width="2"/>',
        f'<text x="{width - 150}" y="{top + 32}" font-family="Arial" font-size="16">Swiss Ephemeris</text>',
        '</svg>',
    ])
    path.write_text("\n".join(lines) + "\n")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--de441", type=Path, required=True)
    p.add_argument("--opm-root", type=Path, required=True)
    p.add_argument("--swiss-ephe", type=Path, required=True, help="Path to Swiss Ephemeris ephemeris files")
    p.add_argument("--nodes-per-segment", type=int, default=512)
    p.add_argument("--bodies", default="sun,moon,mercury,venus,mars,jupiter,saturn,uranus,neptune,pluto")
    p.add_argument("--emrat", type=float, default=DE441_EMRAT)
    p.add_argument("--plot-dir", type=Path)
    p.add_argument("--plot-max-points", type=int, default=6000)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    swe.set_ephe_path(str(args.swiss_ephe))
    opms = load_opms(args.opm_root)
    bodies = [b.strip().lower() for b in args.bodies.split(",") if b.strip()]
    if args.plot_dir is not None:
        args.plot_dir.mkdir(parents=True, exist_ok=True)
    print(f"# opm_root={args.opm_root}")
    print(f"# swiss_ephe={args.swiss_ephe} flags={SWISS_FLAGS}")
    print(f"# de441={args.de441} nodes_per_segment={args.nodes_per_segment} emrat={args.emrat:.17g}")
    print("body samples swiss_p50 swiss_p95 swiss_p99 swiss_max swiss_worst_jd opm_p50 opm_p95 opm_p99 opm_max opm_worst_jd max_ratio_opm_over_swiss")
    with SPK.open(str(args.de441)) as spk:
        for body in bodies:
            jds = body_grid(opms, body, args.nodes_per_segment)
            truth = de441_geocentric(spk, body, jds)
            sw = swiss_geocentric(body, jds)
            op = opm_geocentric(opms, body, jds, args.emrat)
            sw_err = angular_errors_arcsec(truth, sw)
            op_err = angular_errors_arcsec(truth, op)
            sw50, sw95, sw99, swmax, swi = summarize(sw_err)
            op50, op95, op99, opmax, opi = summarize(op_err)
            print(
                f"{body} {len(jds)} "
                f"{sw50:.9g} {sw95:.9g} {sw99:.9g} {swmax:.9g} {jds[swi]:.9f} "
                f"{op50:.9g} {op95:.9g} {op99:.9g} {opmax:.9g} {jds[opi]:.9f} "
                f"{opmax / swmax:.9g}",
                flush=True,
            )
            if args.plot_dir is not None:
                write_svg(args.plot_dir / f"{body}-dense-error.svg", body, jds, op_err, sw_err, args.plot_max_points)
    swe.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
