"""Geocentric Moon modeling helpers for the OPM demo."""
from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
from jplephem.spk import SPK
from scipy.signal import argrelextrema

DE441_PATH: Path | None = None
JD_J2000 = 2451545.0
CENTURY_DAYS = 36525.0
ARCSEC_PER_RAD = 206264.80624709636
EARTH_ID = 399
MOON_ID = 301
EARTH_MOON_BARYCENTER_ID = 3


@dataclass(frozen=True)
class SegmentData:
    jd0: float
    jd1: float
    tmid: float
    nodes: np.ndarray
    pos: np.ndarray
    radius_km: float
    plane_u_best: float
    plane_v_best: float
    apsis_angle_best: float
    coeff_best: np.ndarray


@dataclass(frozen=True)
class TimeModel:
    name: str
    coeff_plane_u: np.ndarray
    coeff_plane_v: np.ndarray
    coeff_apsis_angle: np.ndarray
    eval_fn: Callable[[np.ndarray, np.ndarray], np.ndarray]


def set_de441_path(path: str | Path) -> None:
    global DE441_PATH
    DE441_PATH = Path(path)


def _require_de441_path() -> Path:
    if DE441_PATH is None:
        raise RuntimeError("DE441 path is not set; pass --de441 or call set_de441_path()")
    return DE441_PATH


def normalize_time(jd: np.ndarray | float, start: float, end: float) -> np.ndarray:
    return (2.0 * np.asarray(jd, dtype=np.float64) - start - end) / (end - start)


def cheb_nodes(a: float, b: float, n: int) -> np.ndarray:
    k = np.arange(n)
    x = np.cos(np.pi * (k + 0.5) / n)
    return np.sort(0.5 * (a + b) + 0.5 * (b - a) * x)


def cheb_fit(x: np.ndarray, y: np.ndarray, degree: int) -> np.ndarray:
    return np.polynomial.chebyshev.chebfit(x, y, degree)


def cheb_eval(c: np.ndarray, x: np.ndarray) -> np.ndarray:
    return np.polynomial.chebyshev.chebval(x, c)


class GeoMoonProvider:
    def __init__(self) -> None:
        self.spk = SPK.open(str(_require_de441_path()))
        self.moon_segments = sorted(
            [s for s in self.spk.segments if s.center == EARTH_MOON_BARYCENTER_ID and s.target == MOON_ID],
            key=lambda s: s.start_jd,
        )
        self.earth_segments = sorted(
            [s for s in self.spk.segments if s.center == EARTH_MOON_BARYCENTER_ID and s.target == EARTH_ID],
            key=lambda s: s.start_jd,
        )

    def close(self) -> None:
        self.spk.close()

    def position(self, jd_arr: np.ndarray) -> np.ndarray:
        tdb = np.asarray(jd_arr, dtype=np.float64)
        out = np.zeros((len(tdb), 3), dtype=np.float64)
        for moon_seg, earth_seg in zip(self.moon_segments, self.earth_segments):
            mask = (tdb >= moon_seg.start_jd) & (tdb <= moon_seg.end_jd)
            if not np.any(mask):
                continue
            xm = moon_seg.compute(tdb[mask])
            xe = earth_seg.compute(tdb[mask])
            out[mask, 0] = xm[0] - xe[0]
            out[mask, 1] = xm[1] - xe[1]
            out[mask, 2] = xm[2] - xe[2]
        return out


def plane_frame(plane_u: float, plane_v: float) -> np.ndarray:
    den_inv = 1.0 / (1.0 + plane_u * plane_u + plane_v * plane_v)
    z_axis = np.array([2.0 * plane_u * den_inv, -2.0 * plane_v * den_inv, (1.0 - plane_u * plane_u - plane_v * plane_v) * den_inv])
    x_axis = np.array([(1.0 + plane_v * plane_v - plane_u * plane_u) * den_inv, 2.0 * plane_v * plane_u * den_inv, -2.0 * plane_u * den_inv])
    y_axis = np.array([2.0 * plane_v * plane_u * den_inv, (1.0 - plane_v * plane_v + plane_u * plane_u) * den_inv, 2.0 * plane_v * den_inv])
    return np.column_stack([x_axis, y_axis, z_axis])


def normal_to_plane_uv(normal: np.ndarray) -> tuple[float, float]:
    w = np.asarray(normal, dtype=np.float64)
    w = w / np.linalg.norm(w)
    if w[2] < 0:
        w = -w
    denom = 1.0 + w[2]
    if denom < 1e-15:
        raise ValueError("normal too close to south pole")
    return float(w[0] / denom), float(-w[1] / denom)


def fit_best_frame_params(pos: np.ndarray) -> tuple[float, float, float]:
    _, _, vh = np.linalg.svd(pos, full_matrices=False)
    normal = vh[-1]
    if normal[2] < 0:
        normal = -normal
    plane_u, plane_v = normal_to_plane_uv(normal)
    local = pos @ plane_frame(plane_u, plane_v)
    radii = np.linalg.norm(pos, axis=1)
    apsis_vec = local[int(np.argmin(radii))]
    apsis_angle = float(math.atan2(apsis_vec[1], apsis_vec[0]))
    return plane_u, plane_v, apsis_angle


def align_positions(pos: np.ndarray, plane_u: float, plane_v: float, apsis_angle: float) -> np.ndarray:
    local = pos @ plane_frame(plane_u, plane_v)
    c = math.cos(apsis_angle)
    s = math.sin(apsis_angle)
    x = c * local[:, 0] + s * local[:, 1]
    y = -s * local[:, 0] + c * local[:, 1]
    return np.column_stack([x, y, local[:, 2]])


def unalign_positions(aligned: np.ndarray, plane_u: float, plane_v: float, apsis_angle: float) -> np.ndarray:
    c = math.cos(apsis_angle)
    s = math.sin(apsis_angle)
    local_x = c * aligned[:, 0] - s * aligned[:, 1]
    local_y = s * aligned[:, 0] + c * aligned[:, 1]
    local = np.column_stack([local_x, local_y, aligned[:, 2]])
    return local @ plane_frame(plane_u, plane_v).T


def angular_errors_arcsec(truth: np.ndarray, recon: np.ndarray) -> np.ndarray:
    diff = np.linalg.norm(truth - recon, axis=1)
    radius = np.linalg.norm(truth, axis=1)
    return np.degrees(np.arctan2(diff, radius)) * 3600.0


def find_perigee_times(provider: GeoMoonProvider, jd_start: float, jd_end: float, step_days: float) -> list[float]:
    grid = np.arange(jd_start, jd_end + 0.5 * step_days, step_days)
    radii = np.linalg.norm(provider.position(grid), axis=1)
    order = max(2, int(round(3.0 / step_days)))
    idx = argrelextrema(radii, np.less, order=order)[0]
    return [float(grid[i]) for i in idx if jd_start < grid[i] < jd_end]


def find_perigee_segments(provider: GeoMoonProvider, jd_start: float, jd_end: float, step_days: float, mode: str) -> list[tuple[float, float]]:
    events = find_perigee_times(provider, jd_start, jd_end, step_days)
    if len(events) < 2:
        return []
    if mode == "true-perigee":
        boundaries = events
    elif mode == "mean-perigee":
        idx = np.arange(len(events), dtype=np.float64)
        slope, intercept = np.polyfit(idx, np.asarray(events, dtype=np.float64), 1)
        boundaries = [float(intercept + slope * i) for i in idx]
    else:
        raise ValueError(f"unknown segment mode: {mode}")
    return [(boundaries[i], boundaries[i + 1]) for i in range(len(boundaries) - 1) if boundaries[i + 1] > boundaries[i] + 5.0]


def build_segments(args: argparse.Namespace) -> list[SegmentData]:
    provider = GeoMoonProvider()
    try:
        jd_end = args.jd_start + args.days
        boundaries = find_perigee_segments(provider, args.jd_start, jd_end, args.perigee_step_days, args.segment_mode)
        if args.max_segments:
            boundaries = boundaries[: args.max_segments]
        print(f"segments={len(boundaries)} complete perigee-to-perigee intervals")
        n_nodes = (args.max_degree + 1) * args.node_oversample
        segments: list[SegmentData] = []
        for idx, (a, b) in enumerate(boundaries):
            nodes = cheb_nodes(a, b, n_nodes)
            pos = provider.position(nodes)
            plane_u, plane_v, apsis_angle = fit_best_frame_params(pos)
            aligned = align_positions(pos, plane_u, plane_v, apsis_angle)
            tau = normalize_time(nodes, a, b)
            coeff = np.vstack([cheb_fit(tau, aligned[:, axis], args.max_degree) for axis in range(3)])
            segments.append(SegmentData(a, b, 0.5 * (a + b), nodes, pos, float(np.median(np.linalg.norm(pos, axis=1))), plane_u, plane_v, apsis_angle, coeff))
            if (idx + 1) % 100 == 0:
                print(f"  built {idx + 1}/{len(boundaries)}")
        return segments
    finally:
        provider.close()


def fit_cheb_model(tnorm: np.ndarray, values: np.ndarray, degree: int) -> TimeModel:
    return TimeModel(
        name=f"cheb{degree}",
        coeff_plane_u=cheb_fit(tnorm, values[:, 0], degree),
        coeff_plane_v=cheb_fit(tnorm, values[:, 1], degree),
        coeff_apsis_angle=cheb_fit(tnorm, values[:, 2], degree),
        eval_fn=lambda coeff, t: cheb_eval(coeff, t),
    )


def build_time_models(segments: list[SegmentData], args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray, list[TimeModel]]:
    tmids = np.array([s.tmid for s in segments])
    tnorm = normalize_time(tmids, tmids[0], tmids[-1])
    values = np.column_stack([
        np.array([s.plane_u_best for s in segments]),
        np.array([s.plane_v_best for s in segments]),
        np.unwrap(np.array([s.apsis_angle_best for s in segments])),
    ])
    models = [fit_cheb_model(tnorm, values, d) for d in args.cheb_model_degrees]
    return tnorm, values, models


def eval_model(model: TimeModel, tnorm: np.ndarray) -> np.ndarray:
    return np.column_stack([
        model.eval_fn(model.coeff_plane_u, tnorm),
        model.eval_fn(model.coeff_plane_v, tnorm),
        model.eval_fn(model.coeff_apsis_angle, tnorm),
    ])


def fit_aligned_coeffs_for_params(segments: list[SegmentData], params: np.ndarray, max_degree: int) -> np.ndarray:
    coeffs = np.zeros((len(segments), 3, max_degree + 1), dtype=np.float64)
    for si, seg in enumerate(segments):
        plane_u, plane_v, apsis_angle = params[si]
        aligned = align_positions(seg.pos, float(plane_u), float(plane_v), float(apsis_angle))
        tau = normalize_time(seg.nodes, seg.jd0, seg.jd1)
        for axis in range(3):
            coeffs[si, axis] = cheb_fit(tau, aligned[:, axis], max_degree)
    return coeffs
