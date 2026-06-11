#!/usr/bin/env python3
"""Prototype normal-vector local frame for OPM generation.

This script compares the current OPM1 stereographic plane parameters
(plane_u, plane_v, apsis_angle) against a proposed unit-normal plus apsis
frame (normal_x, normal_y, normal_z, apsis_angle) without changing the OPM
binary format. It runs the initial fit in memory and reports geocentric
angular error metrics and estimated payload sizes.
"""
from __future__ import annotations

import argparse
import math
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
from jplephem.spk import SPK

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import opm_demo.generator as gen  # noqa: E402
import opm_demo.orbit_model as proto  # noqa: E402
import opm_demo.moon_model as moon_proto  # noqa: E402
from opm_demo.body_configs import DEFAULT_BODY_ORDER, BodyConfig  # noqa: E402
from opm_demo.format import OPM_BODY_IDS  # noqa: E402
from optimize_opm_segment_rounding import q_value_width, topk_mean  # noqa: E402
from optimize_opm_ssb_sun_anchor_pmax import capped_lex_better, golden_section_max  # noqa: E402

AXIS_COUNT = 3
ARCSEC_PER_RAD = 206264.80624709636
FrameMode = Literal["stereo", "normal"]


@dataclass
class FitResult:
    body: str
    mode: str
    cfg: BodyConfig
    boundaries: np.ndarray
    quant_steps: np.ndarray
    widths: np.ndarray
    qcoeffs: np.ndarray
    shape_x: np.ndarray | None
    shape_y: np.ndarray | None
    frame_coeffs: np.ndarray | None
    eval_jds: np.ndarray
    truth_native: np.ndarray
    recon_native: np.ndarray
    native_p50: float
    native_p95: float
    native_p99: float
    native_max: float
    payload_bytes: int
    model_bytes: int


def angular_errors_arcsec(truth: np.ndarray, recon: np.ndarray) -> np.ndarray:
    cross = np.linalg.norm(np.cross(truth, recon), axis=1)
    dot = np.einsum("ij,ij->i", truth, recon)
    return np.arctan2(cross, dot) * ARCSEC_PER_RAD


def summarize(err: np.ndarray) -> tuple[float, float, float, float]:
    return (float(np.percentile(err, 50)), float(np.percentile(err, 95)), float(np.percentile(err, 99)), float(np.max(err)))


def normal_from_plane_uv(plane_u: float, plane_v: float) -> np.ndarray:
    den = 1.0 + plane_u * plane_u + plane_v * plane_v
    return np.asarray([2.0 * plane_u / den, -2.0 * plane_v / den, (1.0 - plane_u * plane_u - plane_v * plane_v) / den], dtype=np.float64)


def plane_uv_from_normal(normal: np.ndarray) -> tuple[float, float]:
    n = np.asarray(normal, dtype=np.float64)
    norm = float(np.linalg.norm(n))
    if norm <= 0.0 or not math.isfinite(norm):
        raise ValueError("invalid normal")
    n = n / norm
    if n[2] < 0.0:
        n = -n
    denom = 1.0 + float(n[2])
    if denom < 1e-15:
        raise ValueError("normal too close to stereographic singularity")
    return float(n[0] / denom), float(-n[1] / denom)


def align_positions_normal(pos: np.ndarray, normal: np.ndarray, apsis_angle: float) -> np.ndarray:
    u, v = plane_uv_from_normal(normal)
    return proto.align_positions(pos, u, v, apsis_angle)


def unalign_positions_normal(aligned: np.ndarray, normal: np.ndarray, apsis_angle: float) -> np.ndarray:
    u, v = plane_uv_from_normal(normal)
    return proto.unalign_positions(aligned, u, v, apsis_angle)


def fit_cheb1_values(tmids: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if len(tmids) == 1:
        coeffs = np.column_stack([values[0], np.zeros(values.shape[1], dtype=np.float64)]).T
        return np.zeros(1, dtype=np.float64), coeffs
    tnorm = proto.normalize_time(tmids, tmids[0], tmids[-1])
    coeffs = np.vstack([proto.cheb_fit(tnorm, values[:, axis], 1) for axis in range(values.shape[1])])
    return tnorm, coeffs


def eval_coeffs(coeffs: np.ndarray, tnorm: np.ndarray, *, normalize_normal: bool) -> np.ndarray:
    values = np.column_stack([proto.cheb_eval(coeffs[i], tnorm) for i in range(coeffs.shape[0])])
    if normalize_normal:
        normals = values[:, :3]
        norms = np.linalg.norm(normals, axis=1)
        values[:, :3] = normals / norms[:, None]
    return values


def align_positions_mode(pos: np.ndarray, params: np.ndarray, mode: FrameMode) -> np.ndarray:
    if mode == "stereo":
        return proto.align_positions(pos, float(params[0]), float(params[1]), float(params[2]))
    return align_positions_normal(pos, params[:3], float(params[3]))


def unalign_positions_mode(aligned: np.ndarray, params: np.ndarray, mode: FrameMode) -> np.ndarray:
    if mode == "stereo":
        return proto.unalign_positions(aligned, float(params[0]), float(params[1]), float(params[2]))
    return unalign_positions_normal(aligned, params[:3], float(params[3]))


def fit_result_from_segments(
    *,
    body: str,
    cfg: BodyConfig,
    segments: list,
    provider,
    jd_start: float,
    jd_end: float,
    node_oversample: int,
    mode: FrameMode,
) -> FitResult:
    tmids = np.asarray([s.tmid for s in segments], dtype=np.float64)
    stereo_values = np.column_stack([
        np.asarray([s.plane_u_best for s in segments], dtype=np.float64),
        np.asarray([s.plane_v_best for s in segments], dtype=np.float64),
        np.unwrap(np.asarray([s.apsis_angle_best for s in segments], dtype=np.float64)),
    ])
    if mode == "stereo":
        frame_values = stereo_values
        normalize_normal = False
    else:
        normals = np.asarray([normal_from_plane_uv(float(u), float(v)) for u, v, _ in stereo_values], dtype=np.float64)
        alpha = stereo_values[:, 2].copy()
        flips = 0
        for i in range(1, len(normals)):
            if float(np.dot(normals[i], normals[i - 1])) < 0.0:
                normals[i] = -normals[i]
                alpha[i] += math.pi
                flips += 1
        if flips:
            alpha = np.unwrap(alpha)
        frame_values = np.column_stack([normals, alpha])
        normalize_normal = True

    tnorm_frames, frame_coeffs = fit_cheb1_values(tmids, frame_values)
    params = eval_coeffs(frame_coeffs, tnorm_frames, normalize_normal=normalize_normal)

    max_degree = max(int(cfg.shape_degree or 0), int(cfg.residual_degree))
    fit_nodes = (max_degree + 1) * node_oversample
    shape_degree = int(cfg.shape_degree or 0)
    residual_degree = int(cfg.residual_degree)

    aligned_coeffs = np.zeros((len(segments), AXIS_COUNT, shape_degree + 1), dtype=np.float64)
    for si, seg in enumerate(segments):
        tau = gen.normalize_time_expanded(seg.nodes, seg.jd0, seg.jd1, cfg.segment_domain_expansion_fraction)
        aligned = align_positions_mode(seg.pos, params[si], mode)
        for axis in range(AXIS_COUNT):
            aligned_coeffs[si, axis] = proto.cheb_fit(tau, aligned[:, axis], shape_degree)
    shape_x = np.mean(aligned_coeffs[:, 0, :], axis=0)
    shape_y = np.mean(aligned_coeffs[:, 1, :], axis=0)

    quant_steps = gen.opm_quant_steps(residual_degree, cfg.quant.base_km, cfg.quant.pattern)
    qcoeffs = []
    eval_jds_parts = []
    truth_parts = []
    recon_parts = []
    eval_nodes = max(32, (residual_degree + 1) * node_oversample)
    for si, seg in enumerate(segments):
        tau = gen.normalize_time_expanded(seg.nodes, seg.jd0, seg.jd1, cfg.segment_domain_expansion_fraction)
        aligned_truth = align_positions_mode(seg.pos, params[si], mode)
        coeffs = np.vstack([
            proto.cheb_fit(tau, aligned_truth[:, 0] - proto.cheb_eval(shape_x, tau), residual_degree),
            proto.cheb_fit(tau, aligned_truth[:, 1] - proto.cheb_eval(shape_y, tau), residual_degree),
            proto.cheb_fit(tau, aligned_truth[:, 2], residual_degree),
        ])
        quantized, reconstructed_coeffs = gen.quantize_coeffs(coeffs, quant_steps)
        qcoeffs.append(quantized)

        a = max(seg.jd0, jd_start)
        b = min(seg.jd1, jd_end)
        ej = proto.cheb_nodes(a, b, eval_nodes)
        etau = gen.normalize_time_expanded(ej, seg.jd0, seg.jd1, cfg.segment_domain_expansion_fraction)
        aligned = np.column_stack([
            proto.cheb_eval(shape_x, etau) + proto.cheb_eval(reconstructed_coeffs[0], etau),
            proto.cheb_eval(shape_y, etau) + proto.cheb_eval(reconstructed_coeffs[1], etau),
            proto.cheb_eval(reconstructed_coeffs[2], etau),
        ])
        recon = unalign_positions_mode(aligned, params[si], mode)
        eval_jds_parts.append(ej)
        truth_parts.append(provider.position(ej))
        recon_parts.append(recon)

    eval_jds = np.concatenate(eval_jds_parts)
    truth = np.vstack(truth_parts)
    recon = np.vstack(recon_parts)
    err = angular_errors_arcsec(truth, recon)
    qarr = np.stack(qcoeffs, axis=0)
    widths, payload = gen.pack_qcoeffs(qarr)
    p50, p95, p99, max_err = summarize(err)
    boundaries = np.asarray([segments[0].jd0] + [s.jd1 for s in segments], dtype=np.float64)
    model_bytes = (2 * (shape_degree + 1) + frame_coeffs.size) * 8
    return FitResult(body, mode, cfg, boundaries, quant_steps, widths, qarr, shape_x, shape_y, frame_coeffs, eval_jds, truth, recon, p50, p95, p99, max_err, len(payload), model_bytes)


def raw_sun_fit(spk: SPK, jd_start: float, jd_end: float, node_oversample: int) -> FitResult:
    body = "sun"
    cfg = gen.body_config_for_generation(body)
    provider = gen.BaryProvider(spk, gen.SPK_TARGET_IDS[body])
    packed = gen.fit_raw_sun(provider, cfg, jd_start, jd_end, node_oversample)
    coeffs = packed.qcoeffs.astype(np.float64) * packed.quant_steps[None, None, :]
    eval_nodes = max(32, (cfg.residual_degree + 1) * node_oversample)
    eval_jds_parts = []
    truth_parts = []
    recon_parts = []
    for si, (a, b) in enumerate(zip(packed.boundaries[:-1], packed.boundaries[1:])):
        lo = max(float(a), jd_start)
        hi = min(float(b), jd_end)
        if hi <= lo:
            continue
        jds = proto.cheb_nodes(lo, hi, eval_nodes)
        tau = gen.normalize_time_expanded(jds, float(a), float(b), cfg.segment_domain_expansion_fraction)
        recon = np.column_stack([proto.cheb_eval(coeffs[si, axis], tau) for axis in range(AXIS_COUNT)])
        eval_jds_parts.append(jds)
        truth_parts.append(provider.position(jds))
        recon_parts.append(recon)
    eval_jds = np.concatenate(eval_jds_parts)
    truth = np.vstack(truth_parts)
    recon = np.vstack(recon_parts)
    return FitResult(body, "raw", cfg, packed.boundaries, packed.quant_steps, packed.widths, packed.qcoeffs, None, None, None, eval_jds, truth, recon, packed.p50, packed.p95, packed.p99, packed.max_err, len(packed.payload), 0)


def frame_params_for_result(res: FitResult) -> np.ndarray | None:
    if res.frame_coeffs is None:
        return None
    mids = 0.5 * (res.boundaries[:-1] + res.boundaries[1:])
    if len(mids) == 1:
        tnorm = np.zeros(1, dtype=np.float64)
    else:
        tnorm = proto.normalize_time(mids, mids[0], mids[-1])
    return eval_coeffs(res.frame_coeffs, tnorm, normalize_normal=(res.mode == "normal"))


def reconstruct_native_at(res: FitResult, jds: np.ndarray) -> np.ndarray:
    jds = np.asarray(jds, dtype=np.float64)
    seg_idx = np.searchsorted(res.boundaries, jds, side="right") - 1
    seg_idx = np.clip(seg_idx, 0, len(res.boundaries) - 2)
    a = res.boundaries[seg_idx]
    b = res.boundaries[seg_idx + 1]
    width = b - a
    expanded_a = a - res.cfg.segment_domain_expansion_fraction * width
    expanded_b = b + res.cfg.segment_domain_expansion_fraction * width
    tau = (2.0 * jds - expanded_a - expanded_b) / (expanded_b - expanded_a)
    coeffs = res.qcoeffs.astype(np.float64) * res.quant_steps[None, None, :]
    segment_coeffs = coeffs[seg_idx]
    if res.shape_x is None or res.shape_y is None or res.frame_coeffs is None:
        return np.column_stack([proto.cheb_eval(segment_coeffs[i, axis], tau[i]) for i in range(len(jds)) for axis in []]) if False else np.asarray([
            [proto.cheb_eval(segment_coeffs[i, axis], tau[i]) for axis in range(AXIS_COUNT)]
            for i in range(len(jds))
        ], dtype=np.float64)
    params = frame_params_for_result(res)
    out = np.empty((len(jds), AXIS_COUNT), dtype=np.float64)
    for i, si in enumerate(seg_idx):
        aligned = np.asarray([
            proto.cheb_eval(res.shape_x, tau[i]) + proto.cheb_eval(segment_coeffs[i, 0], tau[i]),
            proto.cheb_eval(res.shape_y, tau[i]) + proto.cheb_eval(segment_coeffs[i, 1], tau[i]),
            proto.cheb_eval(segment_coeffs[i, 2], tau[i]),
        ], dtype=np.float64)[None, :]
        out[i] = unalign_positions_mode(aligned, params[int(si)], res.mode)[0]
    return out


def truth_native_at(spk: SPK, body: str, jds: np.ndarray) -> np.ndarray:
    cfg = gen.body_config_for_generation(body)
    if body == "moon":
        provider = moon_proto.GeoMoonProvider()
        try:
            return provider.position(jds)
        finally:
            provider.close()
    if cfg.center == "sun":
        provider = proto.HelioProvider(gen.SPK_TARGET_IDS[body])
        try:
            return provider.position(jds)
        finally:
            provider.close()
    provider = gen.BaryProvider(spk, gen.SPK_TARGET_IDS[body])
    return provider.position(jds)


def build_segments_for_body(body: str, cfg: BodyConfig, jd_start: float, jd_end: float, node_oversample: int) -> tuple[list, object, object | None]:
    if cfg.method == "fixed_frame_shape":
        spk = SPK.open(str(_DE441_PATH))
        provider = gen.BaryProvider(spk, gen.SPK_TARGET_IDS[body])
        bounds = gen.fixed_bounds(jd_start, jd_end, float(cfg.segment_days))
        fit_nodes = (max(int(cfg.shape_degree or 0), int(cfg.residual_degree)) + 1) * node_oversample
        segments = []
        for a, b in bounds:
            nodes = gen.cheb_nodes_expanded(a, b, fit_nodes, cfg.segment_domain_expansion_fraction)
            pos = provider.position(nodes)
            plane_u, plane_v, apsis_angle = proto.fit_best_frame_params(pos)
            segments.append(proto.SegmentData(a, b, 0.5 * (a + b), nodes, pos, float(np.median(np.linalg.norm(pos, axis=1))), plane_u, plane_v, apsis_angle, np.empty((0,))))
        return segments, provider, spk

    if cfg.method == "mean_apsis_frame_shape":
        clock = gen.mercury_clock() if body == "mercury" else None
        pargs = argparse.Namespace(
            body=body,
            jd_start=jd_start - cfg.edge_margin_days,
            days=(jd_end - jd_start) + 2.0 * cfg.edge_margin_days,
            apsis_step_days=cfg.apsis_step_days,
            segment_mode="mean-apsis",
            max_degree=cfg.shape_degree,
            node_oversample=node_oversample,
            max_segments=0,
            cheb_model_degrees=[1],
            fourier_harmonics=[],
            residual_degrees=[cfg.residual_degree],
        )
        old_cheb_nodes = proto.cheb_nodes
        old_find = proto.find_apsis_segments

        def patched_nodes(a: float, b: float, n: int) -> np.ndarray:
            fa, fb = gen.expanded_bounds(a, b, cfg.segment_domain_expansion_fraction)
            return old_cheb_nodes(fa, fb, n)

        def corrected_segments(provider, start: float, end: float, step_days: float, mode: str):
            if clock is None or mode != "mean-apsis":
                return old_find(provider, start, end, step_days, mode)
            return clock.bounds(start, end)

        proto.set_de441_path(_DE441_PATH)
        proto.cheb_nodes = patched_nodes
        proto.find_apsis_segments = corrected_segments
        try:
            segments_all = proto.build_segments(pargs)
        finally:
            proto.cheb_nodes = old_cheb_nodes
            proto.find_apsis_segments = old_find
        segments = [s for s in segments_all if s.jd1 > jd_start and s.jd0 < jd_end]
        provider = proto.HelioProvider(gen.SPK_TARGET_IDS[body])
        return segments, provider, provider

    if cfg.method == "mean_lunar_apsis_frame_shape":
        clock = gen.moon_century_clock()
        pargs = argparse.Namespace(
            jd_start=jd_start,
            days=jd_end - jd_start,
            perigee_step_days=cfg.apsis_step_days,
            segment_mode="mean-perigee",
            max_degree=cfg.shape_degree,
            node_oversample=node_oversample,
            max_segments=0,
            cheb_model_degrees=[1],
            residual_degrees=[cfg.residual_degree],
        )
        old_cheb_nodes = moon_proto.cheb_nodes
        old_find = moon_proto.find_perigee_segments

        def patched_nodes(a: float, b: float, n: int) -> np.ndarray:
            fa, fb = gen.expanded_bounds(a, b, cfg.segment_domain_expansion_fraction)
            return old_cheb_nodes(fa, fb, n)

        def corrected_segments(provider, start: float, end: float, step_days: float, mode: str):
            return clock.bounds(start, end)

        moon_proto.set_de441_path(_DE441_PATH)
        moon_proto.cheb_nodes = patched_nodes
        moon_proto.find_perigee_segments = corrected_segments
        try:
            segments_all = moon_proto.build_segments(pargs)
        finally:
            moon_proto.cheb_nodes = old_cheb_nodes
            moon_proto.find_perigee_segments = old_find
        segments = [s for s in segments_all if s.jd1 > jd_start and s.jd0 < jd_end]
        provider = moon_proto.GeoMoonProvider()
        return segments, provider, provider

    raise ValueError(f"unsupported method for prototype: {cfg.method}")


def dense_grid_jds(res: FitResult, points: int, kind: str, jd_start: float, jd_end: float) -> np.ndarray:
    if points <= 0:
        return np.empty(0, dtype=np.float64)
    if points < 2:
        raise ValueError("--dense-grid must be at least 2")
    parts = []
    for a0, b0 in zip(res.boundaries[:-1], res.boundaries[1:]):
        a = max(float(a0), jd_start)
        b = min(float(b0), jd_end)
        if b <= a:
            continue
        if kind == "uniform":
            x = np.linspace(-1.0, 1.0, points, dtype=np.float64)
        elif kind == "lobatto":
            k = np.arange(points, dtype=np.float64)
            x = -np.cos(np.pi * k / float(points - 1))
        elif kind == "both":
            uniform = np.linspace(-1.0, 1.0, points, dtype=np.float64)
            k = np.arange(points, dtype=np.float64)
            lobatto = -np.cos(np.pi * k / float(points - 1))
            x = np.unique(np.concatenate([uniform, lobatto]))
        else:
            raise ValueError(f"unknown dense grid kind: {kind}")
        parts.append(0.5 * (a + b) + 0.5 * (b - a) * x)
    return np.concatenate(parts)


def locate_worst(res: FitResult, jds: np.ndarray, err: np.ndarray) -> tuple[int, float, float, float]:
    idx = int(np.argmax(err))
    jd = float(jds[idx])
    seg_idx = int(np.searchsorted(res.boundaries, jd, side="right") - 1)
    seg_idx = max(0, min(seg_idx, len(res.boundaries) - 2))
    a = float(res.boundaries[seg_idx])
    b = float(res.boundaries[seg_idx + 1])
    tau = (2.0 * jd - a - b) / (b - a)
    return seg_idx, jd, tau, float(err[idx])


def geocentric_vectors_at(spk: SPK, results: dict[str, FitResult], mode_for_body: dict[str, str], body: str, jds: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    sun_recon = reconstruct_native_at(results[mode_for_body.get("sun", "raw") + ":sun"], jds)
    moon_mode = mode_for_body.get("moon", "stereo")
    emb_mode = mode_for_body.get("emb", "stereo")
    moon_recon = reconstruct_native_at(results[moon_mode + ":moon"], jds)
    emb_recon = reconstruct_native_at(results[emb_mode + ":emb"], jds)
    moon_truth = truth_native_at(spk, "moon", jds)
    emb_truth = truth_native_at(spk, "emb", jds)
    earth_truth = emb_truth - moon_truth / (1.0 + 81.300568221497215)
    earth_recon = emb_recon - moon_recon / (1.0 + 81.300568221497215)
    if body == "moon":
        return moon_truth, reconstruct_native_at(results[mode_for_body[body] + ":" + body], jds)
    if body == "sun":
        return truth_native_at(spk, "sun", jds) - earth_truth, reconstruct_native_at(results[mode_for_body.get("sun", "raw") + ":sun"], jds) - earth_recon
    if body in {"mercury", "venus"}:
        return truth_native_at(spk, "sun", jds) + truth_native_at(spk, body, jds) - earth_truth, sun_recon + reconstruct_native_at(results[mode_for_body[body] + ":" + body], jds) - earth_recon
    return truth_native_at(spk, body, jds) - earth_truth, reconstruct_native_at(results[mode_for_body[body] + ":" + body], jds) - earth_recon


def geocentric_vectors(spk: SPK, results: dict[str, FitResult], mode_for_body: dict[str, str], body: str) -> tuple[np.ndarray, np.ndarray]:
    res = results[mode_for_body[body] + ":" + body]
    return geocentric_vectors_at(spk, results, mode_for_body, body, res.eval_jds)


def dense_summary_line(
    de441_path: str,
    results: dict[str, FitResult],
    body: str,
    mode: str,
    points: int,
    kind: str,
    jd_start: float,
    jd_end: float,
) -> str:
    global _DE441_PATH
    _DE441_PATH = Path(de441_path)
    proto.set_de441_path(_DE441_PATH)
    moon_proto.set_de441_path(_DE441_PATH)
    gen.proto.set_de441_path(_DE441_PATH)
    gen.moon_proto.set_de441_path(_DE441_PATH)
    with SPK.open(de441_path) as spk:
        res = results[f"{mode}:{body}"]
        jds = dense_grid_jds(res, points, kind, jd_start, jd_end)
        native_truth = truth_native_at(spk, body, jds)
        native_recon = reconstruct_native_at(res, jds)
        native_err = angular_errors_arcsec(native_truth, native_recon)
        np50, np95, np99, nmax = summarize(native_err)
        nseg, njd, ntau, _ = locate_worst(res, jds, native_err)
        mode_map = {"sun": "raw", "moon": mode, "emb": mode, body: mode}
        geo_truth, geo_recon = geocentric_vectors_at(spk, results, mode_map, body, jds)
        geo_err = angular_errors_arcsec(geo_truth, geo_recon)
        gp50, gp95, gp99, gmax = summarize(geo_err)
        gseg, gjd, gtau, _ = locate_worst(res, jds, geo_err)
        return f"{body} {mode} {len(jds)} {np50:.9g} {np95:.9g} {np99:.9g} {nmax:.9g} {nseg} {njd:.9f} {ntau:.9g} {gp50:.9g} {gp95:.9g} {gp99:.9g} {gmax:.9g} {gseg} {gjd:.9f} {gtau:.9g}"


def polish_segment_nodes(a: float, b: float, n: int) -> np.ndarray:
    cheb = proto.cheb_nodes(a, b, n)
    k = np.arange(n, dtype=np.float64)
    u = -1.0 + 2.0 * (k + 0.5) / float(n)
    center = 0.5 * (a + b) + 0.5 * (b - a) * np.sign(u) * np.abs(u) ** 2
    ku = np.arange(n * 2, dtype=np.float64)
    uniform = a + (b - a) * (ku + 0.5) / float(n * 2)
    return np.unique(np.sort(np.concatenate([cheb, center, uniform, np.asarray([a, b], dtype=np.float64)])))


def polish_guard_nodes(a: float, b: float, n: int) -> np.ndarray:
    width = b - a
    k = np.arange(n, dtype=np.float64)
    ku = np.arange(n * 2, dtype=np.float64)
    parts = []
    for offset in (0.25, 0.75):
        u = -1.0 + 2.0 * ((k + offset) / float(n))
        parts.append(0.5 * (a + b) + 0.5 * width * np.sign(u) * np.abs(u) ** 2)
        parts.append(a + width * ((ku + offset) / float(n * 2)))
    near = np.asarray([1.0 / 1024.0, 1.0 / 256.0, 1.0 / 64.0], dtype=np.float64)
    band = np.asarray([1.0 / 2048.0, 1.0 / 1024.0, 1.0 / 512.0, 1.0 / 256.0, 1.0 / 128.0, 1.0 / 64.0, 1.0 / 32.0], dtype=np.float64)
    sband = np.asarray([1.5 / 2048.0, 1.5 / 1024.0, 1.5 / 512.0, 1.5 / 256.0, 1.5 / 128.0, 1.5 / 64.0], dtype=np.float64)
    for frac in (near, band, sband):
        parts.append(np.concatenate([a + width * frac, b - width * frac]))
    nodes = np.concatenate(parts)
    nodes = nodes[(nodes >= a) & (nodes <= b)]
    return np.unique(np.sort(nodes))


def segment_recon_from_q(res: FitResult, params: np.ndarray | None, si: int, qtrial: np.ndarray, jds: np.ndarray) -> np.ndarray:
    a = float(res.boundaries[si])
    b = float(res.boundaries[si + 1])
    tau = gen.normalize_time_expanded(jds, a, b, res.cfg.segment_domain_expansion_fraction)
    basis = np.polynomial.chebyshev.chebvander(tau, int(res.cfg.residual_degree))
    seg_coeffs = qtrial.astype(np.float64) * res.quant_steps[None, :]
    if res.shape_x is None or res.shape_y is None or params is None:
        return np.column_stack([basis @ seg_coeffs[axis] for axis in range(AXIS_COUNT)])
    aligned = np.column_stack([
        proto.cheb_eval(res.shape_x, tau) + basis @ seg_coeffs[0],
        proto.cheb_eval(res.shape_y, tau) + basis @ seg_coeffs[1],
        basis @ seg_coeffs[2],
    ])
    return unalign_positions_mode(aligned, params[si], res.mode)


def prod_metric_truth_and_sun(res: FitResult, sun: FitResult, jds: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
    if res.body in {"mercury", "venus", "moon"}:
        return truth_native_at_path(res.body, jds), None
    with SPK.open(str(_DE441_PATH)) as spk:
        body_truth = gen.BaryProvider(spk, gen.SPK_TARGET_IDS[res.body]).position(jds)
        sun_truth = gen.BaryProvider(spk, gen.SPK_TARGET_IDS["sun"]).position(jds)
    return body_truth - sun_truth, reconstruct_native_at(sun, jds)


def prod_metric_errors(res: FitResult, sun: FitResult, params: np.ndarray | None, si: int, qtrial: np.ndarray, jds: np.ndarray, truth: np.ndarray, sun_recon: np.ndarray | None) -> np.ndarray:
    recon = segment_recon_from_q(res, params, si, qtrial, jds)
    if sun_recon is not None:
        recon = recon - sun_recon
    return angular_errors_arcsec(truth, recon)


def append_refined_nodes_prod(res: FitResult, sun: FitResult, params: np.ndarray | None, si: int, base_jds: np.ndarray, truth: np.ndarray, sun_recon: np.ndarray | None, refine_peaks: int) -> np.ndarray:
    if refine_peaks <= 0 or len(base_jds) < 3:
        return base_jds
    q = res.qcoeffs[si]
    errs = prod_metric_errors(res, sun, params, si, q, base_jds, truth, sun_recon)
    local = [i for i in range(1, len(base_jds) - 1) if errs[i] >= errs[i - 1] and errs[i] >= errs[i + 1]] or list(range(len(base_jds)))
    refined: list[float] = []
    for i in sorted(local, key=lambda idx: float(errs[idx]), reverse=True)[:refine_peaks]:
        lo = float(base_jds[max(0, i - 1)])
        hi = float(base_jds[min(len(base_jds) - 1, i + 1)])
        if hi <= lo:
            refined.append(float(base_jds[i]))
            continue
        def err_at(x: float) -> float:
            jd = np.asarray([x], dtype=np.float64)
            t, s = prod_metric_truth_and_sun(res, sun, jd)
            return float(prod_metric_errors(res, sun, params, si, q, jd, t, s)[0])
        refined.append(golden_section_max(err_at, lo, hi))
    return np.unique(np.sort(np.concatenate([base_jds, np.asarray(refined, dtype=np.float64)])))


def polish_result_production_like(res: FitResult, sun: FitResult, top_segments: int, nodes_per_segment: int, max_passes: int, radius: int, jd_start: float, jd_end: float) -> None:
    if top_segments <= 0:
        return
    params = frame_params_for_result(res)
    seg_errs: list[tuple[float, int, np.ndarray, np.ndarray, np.ndarray | None]] = []
    for si, (a0, b0) in enumerate(zip(res.boundaries[:-1], res.boundaries[1:])):
        a = max(float(a0), jd_start)
        b = min(float(b0), jd_end)
        if b <= a:
            continue
        jds0 = polish_segment_nodes(a, b, nodes_per_segment)
        truth0, sun0 = prod_metric_truth_and_sun(res, sun, jds0)
        jds = append_refined_nodes_prod(res, sun, params, si, jds0, truth0, sun0, 3)
        truth, sun_recon = prod_metric_truth_and_sun(res, sun, jds)
        err = prod_metric_errors(res, sun, params, si, res.qcoeffs[si], jds, truth, sun_recon)
        seg_errs.append((float(np.max(err)), si, jds, truth, sun_recon))
    seg_errs.sort(reverse=True, key=lambda x: x[0])
    limit = len(seg_errs) if top_segments < 0 else min(top_segments, len(seg_errs))
    for _maxerr, si, jds, truth, sun_recon in seg_errs[:limit]:
        a = max(float(res.boundaries[si]), jd_start)
        b = min(float(res.boundaries[si + 1]), jd_end)
        guard_jds = polish_guard_nodes(a, b, nodes_per_segment)
        guard_truth, guard_sun = prod_metric_truth_and_sun(res, sun, guard_jds)
        q = res.qcoeffs[si].copy()
        best_err = prod_metric_errors(res, sun, params, si, q, jds, truth, sun_recon)
        best_guard = prod_metric_errors(res, sun, params, si, q, guard_jds, guard_truth, guard_sun)
        candidates = [(axis, degree) for axis in range(AXIS_COUNT) for degree in range(int(res.cfg.residual_degree) + 1)]
        for _pass_idx in range(max_passes):
            improved = False
            current_max = float(np.max(best_err))
            current_tail = topk_mean(best_err, 4)
            max_first = current_max > 7e-4
            ranked: list[tuple[float, float, int, int, int]] = []
            for axis, degree in candidates:
                current = int(q[axis, degree])
                for delta in range(-radius, radius + 1):
                    if delta == 0:
                        continue
                    trial_value = current + delta
                    if q_value_width(trial_value) > int(res.widths[axis, degree]):
                        continue
                    q[axis, degree] = trial_value
                    trial_err = prod_metric_errors(res, sun, params, si, q, jds, truth, sun_recon)
                    trial_max = float(np.max(trial_err))
                    trial_tail = topk_mean(trial_err, 4)
                    ranked.append(((current_max - trial_max) if max_first else (current_tail - trial_tail), (current_tail - trial_tail) if max_first else (current_max - trial_max), axis, degree, delta))
                    q[axis, degree] = current
            ranked.sort(reverse=True, key=lambda item: (item[0], item[1], item[3], item[2]))
            for _score, _secondary, axis, degree, delta in ranked:
                current = int(q[axis, degree])
                trial_value = current + delta
                if q_value_width(trial_value) > int(res.widths[axis, degree]):
                    continue
                q[axis, degree] = trial_value
                trial_err = prod_metric_errors(res, sun, params, si, q, jds, truth, sun_recon)
                better = capped_lex_better(trial_err, best_err, 1e-12, 4, 7e-4)
                if better:
                    trial_guard = prod_metric_errors(res, sun, params, si, q, guard_jds, guard_truth, guard_sun)
                    guard_ceiling = float(np.max(best_guard)) * 1.05 + 2e-5
                    better = float(np.max(trial_guard)) <= guard_ceiling
                if better:
                    best_err = trial_err
                    best_guard = trial_guard
                    improved = True
                else:
                    q[axis, degree] = current
            if not improved:
                break
        res.qcoeffs[si] = q
    res.widths, payload = gen.pack_qcoeffs(res.qcoeffs)
    res.payload_bytes = len(payload)
    res.recon_native = reconstruct_native_at(res, res.eval_jds)
    err = angular_errors_arcsec(res.truth_native, res.recon_native)
    res.native_p50, res.native_p95, res.native_p99, res.native_max = summarize(err)


def truth_native_at_path(body: str, jds: np.ndarray) -> np.ndarray:
    if body == "moon":
        provider = moon_proto.GeoMoonProvider()
        try:
            return provider.position(jds)
        finally:
            provider.close()
    cfg = gen.body_config_for_generation(body)
    if cfg.center == "sun":
        provider = proto.HelioProvider(gen.SPK_TARGET_IDS[body])
        try:
            return provider.position(jds)
        finally:
            provider.close()
    with SPK.open(str(_DE441_PATH)) as spk:
        provider = gen.BaryProvider(spk, gen.SPK_TARGET_IDS[body])
        return provider.position(jds)


_DE441_PATH: Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--de441", type=Path, required=True)
    parser.add_argument("--jd-start", type=float, default=2451545.0)
    parser.add_argument("--days", type=float, default=36525.0)
    parser.add_argument("--bodies", default="mercury,venus,moon,emb,mars,jupiter,saturn,uranus,neptune,pluto")
    parser.add_argument("--node-oversample", type=int, default=2)
    parser.add_argument("--dense-grid", type=int, default=0, help="Evaluate each segment on this many dense grid points after fitting")
    parser.add_argument("--dense-grid-kind", choices=("uniform", "lobatto", "both"), default="both")
    parser.add_argument("--dense-bodies", default="", help="Comma-separated body subset for dense output; defaults to --bodies")
    parser.add_argument("--dense-workers", type=int, default=1, help="Parallel workers for dense body/mode evaluation")
    parser.add_argument("--polish-top", type=int, default=0, help="Production-like polish this many worst segments per body/mode before reporting; -1 means all")
    parser.add_argument("--polish-grid", type=int, default=32, help="Nodes per segment for production-like polish")
    parser.add_argument("--polish-passes", type=int, default=1)
    parser.add_argument("--polish-radius", type=int, default=1)
    args = parser.parse_args()

    global _DE441_PATH
    _DE441_PATH = args.de441
    gen.proto.set_de441_path(args.de441)
    gen.moon_proto.set_de441_path(args.de441)

    jd_start = args.jd_start
    jd_end = args.jd_start + args.days
    bodies = [b.strip() for b in args.bodies.split(",") if b.strip()]
    if "sun" not in bodies:
        all_bodies = ["sun"] + bodies
    else:
        all_bodies = bodies

    results: dict[str, FitResult] = {}
    with SPK.open(str(args.de441)) as spk:
        sun = raw_sun_fit(spk, jd_start, jd_end, args.node_oversample)
        results["raw:sun"] = sun
        print(f"fit sun raw segments={sun.boundaries.size - 1} native_max={sun.native_max:.6g} payload={sun.payload_bytes}", flush=True)

        for body in bodies:
            if body == "sun":
                continue
            cfg = gen.body_config_for_generation(body)
            segments, provider, closeable = build_segments_for_body(body, cfg, jd_start, jd_end, args.node_oversample)
            try:
                for mode in ("stereo", "normal"):
                    res = fit_result_from_segments(body=body, cfg=cfg, segments=segments, provider=provider, jd_start=jd_start, jd_end=jd_end, node_oversample=args.node_oversample, mode=mode)  # type: ignore[arg-type]
                    if args.polish_top:
                        polish_result_production_like(res, sun, args.polish_top, args.polish_grid, args.polish_passes, args.polish_radius, jd_start, jd_end)
                    results[f"{mode}:{body}"] = res
                    print(f"fit {body} {mode} segments={len(segments)} native_max={res.native_max:.6g} payload={res.payload_bytes} model={res.model_bytes}", flush=True)
            finally:
                if hasattr(closeable, "close"):
                    closeable.close()

        print("\nbody mode native_p50 native_p95 native_p99 native_max geo_p50 geo_p95 geo_p99 geo_max payload_bytes model_bytes")
        for body in bodies:
            for mode in ("stereo", "normal"):
                if body == "sun":
                    res = results["raw:sun"]
                    geo_truth, geo_recon = geocentric_vectors(spk, results, {"sun": "raw", "moon": "stereo", "emb": "stereo"}, "sun")
                else:
                    res = results[f"{mode}:{body}"]
                    mode_map = {"sun": "raw", "moon": mode, "emb": mode, body: mode}
                    geo_truth, geo_recon = geocentric_vectors(spk, results, mode_map, body)
                geo = angular_errors_arcsec(geo_truth, geo_recon)
                gp50, gp95, gp99, gmax = summarize(geo)
                print(f"{body} {mode} {res.native_p50:.9g} {res.native_p95:.9g} {res.native_p99:.9g} {res.native_max:.9g} {gp50:.9g} {gp95:.9g} {gp99:.9g} {gmax:.9g} {res.payload_bytes} {res.model_bytes}")

        if args.dense_grid:
            dense_bodies = [b.strip() for b in args.dense_bodies.split(",") if b.strip()] if args.dense_bodies else bodies
            dense_jobs = [(body, mode) for body in dense_bodies if body != "sun" for mode in ("stereo", "normal")]
            print("\ndense_body mode samples native_p50 native_p95 native_p99 native_max native_seg native_jd native_tau geo_p50 geo_p95 geo_p99 geo_max geo_seg geo_jd geo_tau")
            if args.dense_workers > 1:
                dense_results: dict[tuple[str, str], str] = {}
                with ProcessPoolExecutor(max_workers=args.dense_workers) as executor:
                    future_map = {
                        executor.submit(dense_summary_line, str(args.de441), results, body, mode, args.dense_grid, args.dense_grid_kind, jd_start, jd_end): (body, mode)
                        for body, mode in dense_jobs
                    }
                    for future in as_completed(future_map):
                        key = future_map[future]
                        dense_results[key] = future.result()
                        print(dense_results[key], flush=True)
            else:
                for body, mode in dense_jobs:
                    print(dense_summary_line(str(args.de441), results, body, mode, args.dense_grid, args.dense_grid_kind, jd_start, jd_end), flush=True)


if __name__ == "__main__":
    main()
