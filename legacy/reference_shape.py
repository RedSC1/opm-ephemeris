"""First-stage reference-shape fitting for body-packed OPM generation."""
from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
from jplephem.spk import SPK

from opm_demo.body_configs import BodyConfig
from opm_demo.body_packed_configs import BODY_PACKED_CONFIGS
import opm_demo.generator as gen
import opm_demo.orbit_model as proto
import opm_demo.moon_model as moon_proto
from opm_demo.format import SPK_TARGET_IDS


@dataclass(frozen=True)
class ReferenceShapeFit:
    cfg: BodyConfig
    boundaries: np.ndarray
    frame_coeffs: np.ndarray | None
    shape_x: np.ndarray | None
    shape_y: np.ndarray | None
    frame_params: np.ndarray | None
    clock_table: bytes

    @property
    def segment_count(self) -> int:
        return int(len(self.boundaries) - 1)


def normalized_config(body: str) -> BodyConfig:
    base = BODY_PACKED_CONFIGS[body]
    cfg = replace(
        base,
        segment_domain_expansion_fraction=float(np.float32(base.segment_domain_expansion_fraction)),
    )
    if body == "mercury":
        clock = gen.mercury_clock()
        cfg = replace(cfg, clock=replace(cfg.clock, period_days=clock.period_days, phase_start_jd=clock.phase_start_jd))
    elif body == "moon":
        clock = gen.moon_century_clock()
        cfg = replace(cfg, clock=replace(cfg.clock, period_days=clock.period_days, phase_start_jd=clock.phase_start_jd))
    return cfg


def reference_shape_fit_degree(cfg: BodyConfig) -> int:
    if cfg.method == "raw_xyz_cheb":
        return cfg.residual_degree
    if cfg.method == "fixed_frame_shape":
        return max(cfg.shape_degree or 0, cfg.residual_degree)
    if cfg.shape_degree is None:
        raise ValueError(f"{cfg.body}: shape_degree is required for {cfg.method}")
    return cfg.shape_degree


def _optional_array(value: np.ndarray | None) -> np.ndarray:
    if value is None:
        return np.empty((0,), dtype=np.float64)
    return np.asarray(value, dtype=np.float64)


def _cache_string(data: np.lib.npyio.NpzFile, name: str) -> str:
    return str(data[name].item())


def _cache_float(data: np.lib.npyio.NpzFile, name: str) -> float:
    return float(data[name].item())


def _cache_int(data: np.lib.npyio.NpzFile, name: str) -> int:
    return int(data[name].item())


def _float_matches(a: float | None, b: float | None, atol: float = 1e-9) -> bool:
    if a is None or b is None:
        return a is None and b is None
    return abs(float(a) - float(b)) <= atol


def save_reference_shape_cache(
    path: Path,
    shape: ReferenceShapeFit,
    *,
    jd_start: float,
    jd_end: float,
    node_oversample: int,
) -> None:
    cfg = shape.cfg
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        cache_version=np.asarray(1, dtype=np.int32),
        body=np.asarray(cfg.body),
        center=np.asarray(cfg.center),
        method=np.asarray(cfg.method),
        jd_start=np.asarray(float(jd_start), dtype=np.float64),
        jd_end=np.asarray(float(jd_end), dtype=np.float64),
        node_oversample=np.asarray(int(node_oversample), dtype=np.int32),
        reference_fit_degree=np.asarray(reference_shape_fit_degree(cfg), dtype=np.int32),
        residual_degree=np.asarray(cfg.residual_degree, dtype=np.int32),
        shape_degree=np.asarray(-1 if cfg.shape_degree is None else cfg.shape_degree, dtype=np.int32),
        segment_domain_expansion_fraction=np.asarray(cfg.segment_domain_expansion_fraction, dtype=np.float64),
        clock_kind=np.asarray(cfg.clock.kind),
        clock_period_days=np.asarray(np.nan if cfg.clock.period_days is None else cfg.clock.period_days, dtype=np.float64),
        clock_phase_start_jd=np.asarray(np.nan if cfg.clock.phase_start_jd is None else cfg.clock.phase_start_jd, dtype=np.float64),
        quant_base_km=np.asarray(cfg.quant.base_km, dtype=np.float64),
        quant_pattern=np.asarray(cfg.quant.pattern),
        has_frame_coeffs=np.asarray(shape.frame_coeffs is not None, dtype=np.bool_),
        has_shape_x=np.asarray(shape.shape_x is not None, dtype=np.bool_),
        has_shape_y=np.asarray(shape.shape_y is not None, dtype=np.bool_),
        has_frame_params=np.asarray(shape.frame_params is not None, dtype=np.bool_),
        boundaries=np.asarray(shape.boundaries, dtype=np.float64),
        frame_coeffs=_optional_array(shape.frame_coeffs),
        shape_x=_optional_array(shape.shape_x),
        shape_y=_optional_array(shape.shape_y),
        frame_params=_optional_array(shape.frame_params),
        clock_table=np.frombuffer(shape.clock_table, dtype=np.uint8),
    )


def load_reference_shape_cache(
    path: Path,
    body: str,
    jd_start: float,
    jd_end: float,
    node_oversample: int,
) -> ReferenceShapeFit | None:
    if not path.exists():
        return None
    cfg = normalized_config(body)
    try:
        with np.load(path, allow_pickle=False) as data:
            if _cache_int(data, "cache_version") != 1:
                return None
            if _cache_string(data, "body") != cfg.body:
                return None
            if _cache_string(data, "center") != cfg.center or _cache_string(data, "method") != cfg.method:
                return None
            if not _float_matches(_cache_float(data, "jd_start"), float(jd_start)):
                return None
            if not _float_matches(_cache_float(data, "jd_end"), float(jd_end)):
                return None
            if _cache_int(data, "node_oversample") != int(node_oversample):
                return None
            if _cache_int(data, "reference_fit_degree") != reference_shape_fit_degree(cfg):
                return None
            if _cache_int(data, "shape_degree") != (-1 if cfg.shape_degree is None else cfg.shape_degree):
                return None
            if not _float_matches(_cache_float(data, "segment_domain_expansion_fraction"), cfg.segment_domain_expansion_fraction):
                return None
            if _cache_string(data, "clock_kind") != cfg.clock.kind:
                return None
            cached_period = _cache_float(data, "clock_period_days")
            cached_phase = _cache_float(data, "clock_phase_start_jd")
            cached_period_opt = None if np.isnan(cached_period) else cached_period
            cached_phase_opt = None if np.isnan(cached_phase) else cached_phase
            if not _float_matches(cached_period_opt, cfg.clock.period_days):
                return None
            if not _float_matches(cached_phase_opt, cfg.clock.phase_start_jd):
                return None

            frame_coeffs = np.asarray(data["frame_coeffs"], dtype=np.float64) if bool(data["has_frame_coeffs"].item()) else None
            shape_x = np.asarray(data["shape_x"], dtype=np.float64) if bool(data["has_shape_x"].item()) else None
            shape_y = np.asarray(data["shape_y"], dtype=np.float64) if bool(data["has_shape_y"].item()) else None
            frame_params = np.asarray(data["frame_params"], dtype=np.float64) if bool(data["has_frame_params"].item()) else None
            clock_table = np.asarray(data["clock_table"], dtype=np.uint8).tobytes()
            boundaries = np.asarray(data["boundaries"], dtype=np.float64)
    except (OSError, KeyError, ValueError):
        return None
    return ReferenceShapeFit(cfg, boundaries, frame_coeffs, shape_x, shape_y, frame_params, clock_table)


def _fit_frame_model_from_params(tmids: np.ndarray, params: np.ndarray):
    tnorm = proto.normalize_time(tmids, tmids[0], tmids[-1]) if len(tmids) > 1 else np.zeros(1, dtype=np.float64)
    values = np.column_stack([params[:, 0], params[:, 1], np.unwrap(params[:, 2])])
    model = proto.fit_cheb_model(tnorm, values, 1)
    fitted = proto.eval_model(model, tnorm)
    coeffs = np.vstack([model.coeff_plane_u, model.coeff_plane_v, model.coeff_apsis_angle])
    return coeffs, fitted


def _mean_shape(coeffs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    shape = proto.mean_shape_from_coeffs(coeffs[:, :2, :], "mean")
    return shape[0], shape[1]


def fit_raw_reference_shape(spk: SPK, cfg: BodyConfig, jd_start: float, jd_end: float) -> ReferenceShapeFit:
    assert cfg.segment_days is not None
    boundaries = np.asarray([a for a, _ in gen.fixed_bounds(jd_start, jd_end, cfg.segment_days)] + [gen.fixed_bounds(jd_start, jd_end, cfg.segment_days)[-1][1]], dtype=np.float64)
    return ReferenceShapeFit(cfg, boundaries, None, None, None, None, b"")


def fit_fixed_frame_reference_shape(
    provider: gen.BaryProvider,
    cfg: BodyConfig,
    jd_start: float,
    jd_end: float,
    node_oversample: int,
) -> ReferenceShapeFit:
    assert cfg.segment_days is not None and cfg.shape_degree is not None
    bounds = gen.fixed_bounds(jd_start, jd_end, cfg.segment_days)
    max_degree = max(cfg.shape_degree, cfg.residual_degree)
    fit_nodes = (max_degree + 1) * node_oversample

    tmids = np.asarray([0.5 * (a + b) for a, b in bounds], dtype=np.float64)
    best_params = []
    for a, b in bounds:
        nodes = gen.cheb_nodes_expanded(a, b, fit_nodes, cfg.segment_domain_expansion_fraction)
        best_params.append(proto.fit_best_frame_params(provider.position(nodes)))
    frame_coeffs, params = _fit_frame_model_from_params(tmids, np.asarray(best_params, dtype=np.float64))

    aligned_coeffs = np.zeros((len(bounds), gen.AXIS_COUNT, cfg.shape_degree + 1), dtype=np.float64)
    for si, (a, b) in enumerate(bounds):
        nodes = gen.cheb_nodes_expanded(a, b, fit_nodes, cfg.segment_domain_expansion_fraction)
        tau = gen.normalize_time_expanded(nodes, a, b, cfg.segment_domain_expansion_fraction)
        plane_u, plane_v, apsis_angle = params[si]
        aligned = proto.align_positions(provider.position(nodes), float(plane_u), float(plane_v), float(apsis_angle))
        for axis in range(gen.AXIS_COUNT):
            aligned_coeffs[si, axis] = proto.cheb_fit(tau, aligned[:, axis], cfg.shape_degree)

    shape_x, shape_y = _mean_shape(aligned_coeffs)
    boundaries = np.asarray([bounds[0][0]] + [b for _, b in bounds], dtype=np.float64)
    return ReferenceShapeFit(cfg, boundaries, frame_coeffs, shape_x, shape_y, params, b"")


def fit_helio_apsis_reference_shape(
    cfg: BodyConfig,
    jd_start: float,
    jd_end: float,
    node_oversample: int,
) -> ReferenceShapeFit:
    assert cfg.shape_degree is not None
    clock = gen.mercury_clock() if cfg.body == "mercury" else None
    pargs = argparse.Namespace(
        body=cfg.body,
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
    old_find_apsis_segments = proto.find_apsis_segments

    def patched_nodes(a: float, b: float, n: int) -> np.ndarray:
        fa, fb = gen.expanded_bounds(a, b, cfg.segment_domain_expansion_fraction)
        return old_cheb_nodes(fa, fb, n)

    def corrected_segments(provider, start: float, end: float, step_days: float, mode: str):
        if clock is None or mode != "mean-apsis":
            return old_find_apsis_segments(provider, start, end, step_days, mode)
        return clock.bounds(start, end)

    proto.cheb_nodes = patched_nodes
    proto.find_apsis_segments = corrected_segments
    try:
        segments_all = proto.build_segments(pargs)
    finally:
        proto.cheb_nodes = old_cheb_nodes
        proto.find_apsis_segments = old_find_apsis_segments

    segments = [s for s in segments_all if s.jd1 > jd_start and s.jd0 < jd_end]
    if not segments:
        raise RuntimeError(f"no {cfg.body} segments overlap requested range")
    if segments[0].jd0 > jd_start or segments[-1].jd1 < jd_end:
        raise RuntimeError(f"{cfg.body} mean-apsis segments do not cover requested range")

    tnorm, _values, models = proto.build_time_models(segments, pargs)
    model = next(m for m in models if m.name == "cheb1")
    params = proto.eval_model(model, tnorm)
    frame_coeffs = np.vstack([model.coeff_plane_u, model.coeff_plane_v, model.coeff_apsis_angle])

    coeffs = np.zeros((len(segments), gen.AXIS_COUNT, cfg.shape_degree + 1), dtype=np.float64)
    for si, seg in enumerate(segments):
        plane_u, plane_v, apsis_angle = params[si]
        tau = gen.normalize_time_expanded(seg.nodes, seg.jd0, seg.jd1, cfg.segment_domain_expansion_fraction)
        aligned = proto.align_positions(seg.pos, float(plane_u), float(plane_v), float(apsis_angle))
        for axis in range(gen.AXIS_COUNT):
            coeffs[si, axis] = proto.cheb_fit(tau, aligned[:, axis], cfg.shape_degree)

    shape_x, shape_y = _mean_shape(coeffs)
    boundaries = np.asarray([segments[0].jd0] + [s.jd1 for s in segments], dtype=np.float64)
    clock_table = b""
    if clock is not None:
        file_event_index_start = int(round((boundaries[0] - clock.phase_start_jd) / clock.period_days))
        clock_table = clock.to_clock_table(file_event_index_start)
    return ReferenceShapeFit(cfg, boundaries, frame_coeffs, shape_x, shape_y, params, clock_table)


def fit_lunar_apsis_reference_shape(cfg: BodyConfig, jd_start: float, jd_end: float, node_oversample: int) -> ReferenceShapeFit:
    assert cfg.shape_degree is not None
    clock = gen.moon_century_clock()
    pargs = argparse.Namespace(
        jd_start=jd_start - cfg.edge_margin_days,
        days=(jd_end - jd_start) + 2.0 * cfg.edge_margin_days,
        perigee_step_days=cfg.apsis_step_days,
        segment_mode="mean-perigee",
        max_degree=cfg.shape_degree,
        node_oversample=node_oversample,
        max_segments=0,
        cheb_model_degrees=[1],
        residual_degrees=[cfg.residual_degree],
    )
    old_cheb_nodes = moon_proto.cheb_nodes
    old_find_perigee_segments = moon_proto.find_perigee_segments

    def patched_nodes(a: float, b: float, n: int) -> np.ndarray:
        fa, fb = gen.expanded_bounds(a, b, cfg.segment_domain_expansion_fraction)
        return old_cheb_nodes(fa, fb, n)

    def corrected_segments(provider, start: float, end: float, step_days: float, mode: str):
        if mode != "mean-perigee":
            return old_find_perigee_segments(provider, start, end, step_days, mode)
        return clock.bounds(start, end)

    moon_proto.cheb_nodes = patched_nodes
    moon_proto.find_perigee_segments = corrected_segments
    try:
        segments_all = moon_proto.build_segments(pargs)
    finally:
        moon_proto.cheb_nodes = old_cheb_nodes
        moon_proto.find_perigee_segments = old_find_perigee_segments

    segments = gen.select_full_segments(segments_all, jd_start, jd_end, cfg.body)
    tnorm, _values, models = moon_proto.build_time_models(segments, pargs)
    model = next(m for m in models if m.name == "cheb1")
    params = moon_proto.eval_model(model, tnorm)
    frame_coeffs = np.vstack([model.coeff_plane_u, model.coeff_plane_v, model.coeff_apsis_angle])

    coeffs = np.zeros((len(segments), gen.AXIS_COUNT, cfg.shape_degree + 1), dtype=np.float64)
    for si, seg in enumerate(segments):
        plane_u, plane_v, apsis_angle = params[si]
        tau = gen.normalize_time_expanded(seg.nodes, seg.jd0, seg.jd1, cfg.segment_domain_expansion_fraction)
        aligned = moon_proto.align_positions(seg.pos, float(plane_u), float(plane_v), float(apsis_angle))
        for axis in range(gen.AXIS_COUNT):
            coeffs[si, axis] = moon_proto.cheb_fit(tau, aligned[:, axis], cfg.shape_degree)

    shape_x = np.mean(coeffs[:, 0, :], axis=0)
    shape_y = np.mean(coeffs[:, 1, :], axis=0)
    boundaries = np.asarray([segments[0].jd0] + [s.jd1 for s in segments], dtype=np.float64)
    return ReferenceShapeFit(cfg, boundaries, frame_coeffs, shape_x, shape_y, params, clock.to_clock_table())


def fit_reference_shape(spk: SPK, body: str, jd_start: float, jd_end: float, node_oversample: int) -> ReferenceShapeFit:
    cfg = normalized_config(body)
    if cfg.method == "raw_xyz_cheb":
        return fit_raw_reference_shape(spk, cfg, jd_start, jd_end)
    if cfg.method == "fixed_frame_shape":
        return fit_fixed_frame_reference_shape(gen.BaryProvider(spk, SPK_TARGET_IDS[body]), cfg, jd_start, jd_end, node_oversample)
    if cfg.method == "mean_apsis_frame_shape":
        return fit_helio_apsis_reference_shape(cfg, jd_start, jd_end, node_oversample)
    if cfg.method == "mean_lunar_apsis_frame_shape":
        return fit_lunar_apsis_reference_shape(cfg, jd_start, jd_end, node_oversample)
    raise NotImplementedError(f"unsupported body/method: {body} {cfg.method}")
