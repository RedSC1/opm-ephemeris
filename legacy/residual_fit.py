"""Second-stage residual fitting for body-packed OPM generation."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
from jplephem.spk import SPK

from opm_demo.body_configs import BodyConfig
from opm_demo.format import SPK_TARGET_IDS
import opm_demo.generator as gen
import opm_demo.orbit_model as proto
import opm_demo.moon_model as moon_proto
from opm_demo.reference_shape import ReferenceShapeFit


class PositionProvider(Protocol):
    def position(self, jd_arr: np.ndarray) -> np.ndarray: ...


@dataclass(frozen=True)
class ResidualCoefficientFit:
    cfg: BodyConfig
    boundaries: np.ndarray
    coeffs: np.ndarray
    model_table: bytes
    clock_table: bytes


class _ByteArray(np.ndarray):
    pass


def _provider_for_shape(spk: SPK, shape: ReferenceShapeFit) -> tuple[PositionProvider, object | None, object]:
    cfg = shape.cfg
    if cfg.center == "ssb":
        return gen.BaryProvider(spk, SPK_TARGET_IDS[cfg.body]), None, proto
    if cfg.center == "sun":
        provider = proto.HelioProvider(SPK_TARGET_IDS[cfg.body])
        return provider, provider, proto
    if cfg.center == "earth" and cfg.body == "moon":
        provider = moon_proto.GeoMoonProvider()
        return provider, provider, moon_proto
    raise ValueError(f"unsupported provider for {cfg.body}/{cfg.center}")


def _close_if_needed(closeable: object | None) -> None:
    if closeable is not None and hasattr(closeable, "close"):
        closeable.close()


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


def save_residual_coeff_cache(
    path: Path,
    coeff_fit: ResidualCoefficientFit,
    *,
    jd_start: float,
    jd_end: float,
    node_oversample: int,
) -> None:
    cfg = coeff_fit.cfg
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
        residual_degree=np.asarray(cfg.residual_degree, dtype=np.int32),
        shape_degree=np.asarray(-1 if cfg.shape_degree is None else cfg.shape_degree, dtype=np.int32),
        segment_domain_expansion_fraction=np.asarray(cfg.segment_domain_expansion_fraction, dtype=np.float64),
        clock_kind=np.asarray(cfg.clock.kind),
        clock_period_days=np.asarray(np.nan if cfg.clock.period_days is None else cfg.clock.period_days, dtype=np.float64),
        clock_phase_start_jd=np.asarray(np.nan if cfg.clock.phase_start_jd is None else cfg.clock.phase_start_jd, dtype=np.float64),
        quant_base_km=np.asarray(cfg.quant.base_km, dtype=np.float64),
        quant_pattern=np.asarray(cfg.quant.pattern),
        boundaries=np.asarray(coeff_fit.boundaries, dtype=np.float64),
        coeffs=np.asarray(coeff_fit.coeffs, dtype=np.float64),
        model_table=np.frombuffer(coeff_fit.model_table, dtype=np.uint8),
        clock_table=np.frombuffer(coeff_fit.clock_table, dtype=np.uint8),
    )


def load_residual_coeff_cache(
    path: Path,
    shape: ReferenceShapeFit,
    jd_start: float,
    jd_end: float,
    node_oversample: int,
) -> ResidualCoefficientFit | None:
    if not path.exists():
        return None
    cfg = shape.cfg
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
            if _cache_int(data, "residual_degree") != cfg.residual_degree:
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
            boundaries = np.asarray(data["boundaries"], dtype=np.float64)
            if boundaries.shape != shape.boundaries.shape or not np.allclose(boundaries, shape.boundaries, rtol=0.0, atol=1e-8):
                return None
            coeffs = np.asarray(data["coeffs"], dtype=np.float64)
            if coeffs.shape != (shape.segment_count, gen.AXIS_COUNT, cfg.residual_degree + 1):
                return None
            model_table = np.asarray(data["model_table"], dtype=np.uint8).tobytes()
            clock_table = np.asarray(data["clock_table"], dtype=np.uint8).tobytes()
    except (OSError, KeyError, ValueError):
        return None
    return ResidualCoefficientFit(cfg, boundaries, coeffs, model_table, clock_table)


def cheb_fit_matrix(tau: np.ndarray, degree: int) -> np.ndarray:
    vander = np.polynomial.chebyshev.chebvander(tau, degree)
    return np.linalg.pinv(vander)


def chunk_nodes_and_tau(a: np.ndarray, b: np.ndarray, n: int, expansion: float) -> tuple[np.ndarray, np.ndarray]:
    tau = proto.cheb_nodes(-1.0, 1.0, n)
    width = b - a
    fa = a - expansion * width
    fb = b + expansion * width
    nodes = 0.5 * (fa[:, None] + fb[:, None]) + 0.5 * (fb[:, None] - fa[:, None]) * tau[None, :]
    return nodes, tau


def fit_coeff_chunk(fit_matrix: np.ndarray, values: np.ndarray) -> np.ndarray:
    return np.einsum("cn,sna->sac", fit_matrix, values)


def fit_residual_coefficients_chunked(
    spk: SPK,
    shape: ReferenceShapeFit,
    jd_start: float,
    jd_end: float,
    node_oversample: int,
    chunk_size: int,
) -> ResidualCoefficientFit:
    cfg = shape.cfg
    provider, closeable, model = _provider_for_shape(spk, shape)
    try:
        if cfg.method == "raw_xyz_cheb":
            fit_nodes = (cfg.residual_degree + 1) * node_oversample
            _sample_nodes, tau = chunk_nodes_and_tau(np.asarray([0.0]), np.asarray([1.0]), fit_nodes, cfg.segment_domain_expansion_fraction)
            fit_matrix = cheb_fit_matrix(tau, cfg.residual_degree)
            coeff_chunks = []
            for start in range(0, shape.segment_count, chunk_size):
                stop = min(start + chunk_size, shape.segment_count)
                a = np.asarray(shape.boundaries[start:stop], dtype=np.float64)
                b = np.asarray(shape.boundaries[start + 1:stop + 1], dtype=np.float64)
                nodes, _tau = chunk_nodes_and_tau(a, b, fit_nodes, cfg.segment_domain_expansion_fraction)
                pos = provider.position(nodes.reshape(-1)).reshape((stop - start, fit_nodes, gen.AXIS_COUNT))
                coeff_chunks.append(fit_coeff_chunk(fit_matrix, pos))
            coeffs = np.concatenate(coeff_chunks, axis=0)
            return ResidualCoefficientFit(cfg, shape.boundaries, coeffs, b"", b"")

        if shape.shape_x is None or shape.shape_y is None or shape.frame_coeffs is None or shape.frame_params is None:
            raise ValueError(f"{cfg.body}: shape model is incomplete")
        max_degree = max(cfg.shape_degree or 0, cfg.residual_degree)
        fit_nodes = (max_degree + 1) * node_oversample
        _sample_nodes, tau = chunk_nodes_and_tau(np.asarray([0.0]), np.asarray([1.0]), fit_nodes, cfg.segment_domain_expansion_fraction)
        fit_matrix = cheb_fit_matrix(tau, cfg.residual_degree)
        shape_x_values = model.cheb_eval(shape.shape_x, tau)
        shape_y_values = model.cheb_eval(shape.shape_y, tau)
        coeff_chunks = []
        for start in range(0, shape.segment_count, chunk_size):
            stop = min(start + chunk_size, shape.segment_count)
            a = np.asarray(shape.boundaries[start:stop], dtype=np.float64)
            b = np.asarray(shape.boundaries[start + 1:stop + 1], dtype=np.float64)
            nodes, _tau = chunk_nodes_and_tau(a, b, fit_nodes, cfg.segment_domain_expansion_fraction)
            pos = provider.position(nodes.reshape(-1)).reshape((stop - start, fit_nodes, gen.AXIS_COUNT))
            values = np.empty_like(pos)
            for local_si, global_si in enumerate(range(start, stop)):
                plane_u, plane_v, apsis_angle = shape.frame_params[global_si]
                aligned = model.align_positions(pos[local_si], float(plane_u), float(plane_v), float(apsis_angle))
                values[local_si, :, 0] = aligned[:, 0] - shape_x_values
                values[local_si, :, 1] = aligned[:, 1] - shape_y_values
                values[local_si, :, 2] = aligned[:, 2]
            coeff_chunks.append(fit_coeff_chunk(fit_matrix, values))
    finally:
        _close_if_needed(closeable)

    coeffs = np.concatenate(coeff_chunks, axis=0)
    model_table = gen.pack_model_table(shape.shape_x, shape.shape_y, shape.frame_coeffs)
    return ResidualCoefficientFit(cfg, shape.boundaries, coeffs, model_table, shape.clock_table)


def fit_residual_coefficients(
    spk: SPK,
    shape: ReferenceShapeFit,
    jd_start: float,
    jd_end: float,
    node_oversample: int,
    chunk_size: int = 0,
) -> ResidualCoefficientFit:
    if chunk_size > 0:
        return fit_residual_coefficients_chunked(spk, shape, jd_start, jd_end, node_oversample, chunk_size)

    cfg = shape.cfg
    provider, closeable, model = _provider_for_shape(spk, shape)
    try:
        if cfg.method == "raw_xyz_cheb":
            degree = cfg.residual_degree
            fit_nodes = (degree + 1) * node_oversample
            coeff_parts = []
            for a, b in zip(shape.boundaries[:-1], shape.boundaries[1:]):
                a = float(a)
                b = float(b)
                nodes = gen.cheb_nodes_expanded(a, b, fit_nodes, cfg.segment_domain_expansion_fraction)
                tau = gen.normalize_time_expanded(nodes, a, b, cfg.segment_domain_expansion_fraction)
                pos = provider.position(nodes)
                coeff_parts.append(np.vstack([model.cheb_fit(tau, pos[:, axis], degree) for axis in range(gen.AXIS_COUNT)]))
            coeffs = np.stack(coeff_parts, axis=0)
            return ResidualCoefficientFit(cfg, shape.boundaries, coeffs, b"", b"")

        if shape.shape_x is None or shape.shape_y is None or shape.frame_coeffs is None or shape.frame_params is None:
            raise ValueError(f"{cfg.body}: shape model is incomplete")
        max_degree = max(cfg.shape_degree or 0, cfg.residual_degree)
        fit_nodes = (max_degree + 1) * node_oversample
        coeff_parts = []
        for si, (a, b) in enumerate(zip(shape.boundaries[:-1], shape.boundaries[1:])):
            a = float(a)
            b = float(b)
            plane_u, plane_v, apsis_angle = shape.frame_params[si]
            nodes = gen.cheb_nodes_expanded(a, b, fit_nodes, cfg.segment_domain_expansion_fraction)
            tau = gen.normalize_time_expanded(nodes, a, b, cfg.segment_domain_expansion_fraction)
            pos = provider.position(nodes)
            aligned_truth = model.align_positions(pos, float(plane_u), float(plane_v), float(apsis_angle))
            coeff_parts.append(np.vstack([
                model.cheb_fit(tau, aligned_truth[:, 0] - model.cheb_eval(shape.shape_x, tau), cfg.residual_degree),
                model.cheb_fit(tau, aligned_truth[:, 1] - model.cheb_eval(shape.shape_y, tau), cfg.residual_degree),
                model.cheb_fit(tau, aligned_truth[:, 2], cfg.residual_degree),
            ]))
    finally:
        _close_if_needed(closeable)

    coeffs = np.stack(coeff_parts, axis=0)
    model_table = gen.pack_model_table(shape.shape_x, shape.shape_y, shape.frame_coeffs)
    return ResidualCoefficientFit(cfg, shape.boundaries, coeffs, model_table, shape.clock_table)


def pack_residual_coefficients(
    spk: SPK,
    shape: ReferenceShapeFit,
    coeff_fit: ResidualCoefficientFit,
    jd_start: float,
    jd_end: float,
    node_oversample: int,
) -> gen.PackedBody:
    cfg = shape.cfg
    if cfg.body != coeff_fit.cfg.body or cfg.residual_degree != coeff_fit.cfg.residual_degree:
        raise ValueError(f"{cfg.body}: residual coefficient cache config mismatch")
    if coeff_fit.coeffs.shape != (shape.segment_count, gen.AXIS_COUNT, cfg.residual_degree + 1):
        raise ValueError(f"{cfg.body}: bad residual coefficient shape {coeff_fit.coeffs.shape}")

    quant_steps = gen.opm_quant_steps(cfg.residual_degree, cfg.quant.base_km, cfg.quant.pattern)
    qcoeffs = []
    reconstructed_parts = []
    for coeffs in coeff_fit.coeffs:
        quantized, reconstructed_coeffs = gen.quantize_coeffs(coeffs, quant_steps)
        qcoeffs.append(quantized)
        reconstructed_parts.append(reconstructed_coeffs)
    qarr = np.stack(qcoeffs, axis=0)
    widths, payload = gen.pack_qcoeffs(qarr)

    provider, closeable, model = _provider_for_shape(spk, shape)
    try:
        eval_nodes = max(32, (cfg.residual_degree + 1) * node_oversample)
        truth_parts = []
        recon_parts = []
        for si, (a, b) in enumerate(zip(shape.boundaries[:-1], shape.boundaries[1:])):
            a = float(a)
            b = float(b)
            ej = model.cheb_nodes(max(a, jd_start), min(b, jd_end), eval_nodes)
            etau = gen.normalize_time_expanded(ej, a, b, cfg.segment_domain_expansion_fraction)
            reconstructed_coeffs = reconstructed_parts[si]
            if cfg.method == "raw_xyz_cheb":
                recon = np.column_stack([model.cheb_eval(reconstructed_coeffs[axis], etau) for axis in range(gen.AXIS_COUNT)])
            else:
                if shape.shape_x is None or shape.shape_y is None or shape.frame_params is None:
                    raise ValueError(f"{cfg.body}: shape model is incomplete")
                plane_u, plane_v, apsis_angle = shape.frame_params[si]
                aligned = np.column_stack([
                    model.cheb_eval(shape.shape_x, etau) + model.cheb_eval(reconstructed_coeffs[0], etau),
                    model.cheb_eval(shape.shape_y, etau) + model.cheb_eval(reconstructed_coeffs[1], etau),
                    model.cheb_eval(reconstructed_coeffs[2], etau),
                ])
                recon = model.unalign_positions(aligned, float(plane_u), float(plane_v), float(apsis_angle))
            truth_parts.append(provider.position(ej))
            recon_parts.append(recon)
    finally:
        _close_if_needed(closeable)

    p50, p95, p99, max_err = gen.summarize_errors(np.vstack(truth_parts), np.vstack(recon_parts))
    return gen.PackedBody(cfg, shape.boundaries, quant_steps, widths, qarr, payload, coeff_fit.model_table, coeff_fit.clock_table, p50, p95, p99, max_err)


def fit_residuals(
    spk: SPK,
    shape: ReferenceShapeFit,
    jd_start: float,
    jd_end: float,
    node_oversample: int,
    chunk_size: int = 0,
) -> gen.PackedBody:
    coeff_fit = fit_residual_coefficients(spk, shape, jd_start, jd_end, node_oversample, chunk_size=chunk_size)
    return pack_residual_coefficients(spk, shape, coeff_fit, jd_start, jd_end, node_oversample)
