#!/usr/bin/env python3
"""Read and validate OPM1 coverage files against DE441.

This module parses OPM1 minor-0 files produced by opm_demo.generator,
reconstructs positions from the packed payload/model table, and reports
angular-error percentiles against DE441.
"""
from __future__ import annotations

import argparse
import math
import struct
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from jplephem.spk import SPK

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import opm_demo.packing as pack  # noqa: E402
import opm_demo.orbit_model as proto  # noqa: E402
import opm_demo.moon_model as moon_proto  # noqa: E402
from opm_demo.generator import BaryProvider
from opm_demo.format import HEADER_STRUCT, MAGIC, ENDIAN_TAG, FIXED_HEADER_SIZE  # noqa: E402
from opm_demo.format import HEADER_CRC64_OFFSET, PAYLOAD_CRC64_OFFSET, crc64_ecma  # noqa: E402
from opm_demo.format import OPM_BODY_IDS, SPK_TARGET_IDS, MOON_CLOCK_TABLE_STRUCT  # noqa: E402
from opm_demo.format import FRAME_NONE, FRAME_CHEB1_PLANE_APSIS, FRAME_CHEB1_NORMAL_APSIS  # noqa: E402


AXIS_COUNT = 3
SEGMENT_FIXED_DAYS = 0
SEGMENT_AFFINE_PERIOD_PHASE = 1
MODEL_RAW_XYZ_CHEB = 1
MODEL_MEAN_APSIS_FRAME_SHAPE = 2
MODEL_MEAN_LUNAR_APSIS_FRAME_SHAPE = 3
MODEL_FIXED_FRAME_SHAPE = 4
CLOCK_CORRECTION_NONE = 0
CLOCK_CORRECTION_CHEB_EVENT_TIME_F64 = 1
CLOCK_CORRECTION_CENTURY_I16_LINEAR_TABLE = 2
STORAGE_SSB_TO_BODY = 2
STORAGE_SUN_TO_BODY = 3
STORAGE_EARTH_TO_MOON = 4


@dataclass(frozen=True)
class ParsedMercuryClock:
    degree: int
    coeff_count: int
    file_event_index_start: float
    event_index_scale: float
    coeff: np.ndarray

    @classmethod
    def from_bytes(cls, data: bytes) -> "ParsedMercuryClock":
        degree, coeff_count, _reserved, start, scale = struct.unpack("<HHIdd", data[:24])
        coeff = np.frombuffer(data[24 : 24 + coeff_count * 8], dtype="<f8").copy()
        if len(coeff) != coeff_count:
            raise ValueError("truncated Mercury clock table")
        return cls(int(degree), int(coeff_count), float(start), float(scale), coeff)

    def correction_for_indices(self, indices: np.ndarray | int | float) -> np.ndarray:
        global_indices = self.file_event_index_start + np.asarray(indices, dtype=np.float64)
        x = 2.0 * global_indices / self.event_index_scale - 1.0
        return np.polynomial.chebyshev.chebval(x, self.coeff)


@dataclass(frozen=True)
class ParsedMoonClock:
    domain_start_jd: float
    century_days: float
    quantum_seconds: float
    table_count: int
    interpolation_kind: int
    statistic_kind: int
    values: np.ndarray

    @classmethod
    def from_bytes(cls, data: bytes) -> "ParsedMoonClock":
        domain_start, century_days, quantum_seconds, count, interpolation, statistic, _reserved = MOON_CLOCK_TABLE_STRUCT.unpack(
            data[: MOON_CLOCK_TABLE_STRUCT.size]
        )
        start = MOON_CLOCK_TABLE_STRUCT.size
        values = np.frombuffer(data[start : start + count * 2], dtype="<i2").copy()
        if len(values) != count:
            raise ValueError("truncated Moon clock table")
        return cls(float(domain_start), float(century_days), float(quantum_seconds), int(count), int(interpolation), int(statistic), values)

    @property
    def quantum_days(self) -> float:
        return self.quantum_seconds / 86400.0

    def eval(self, jd: np.ndarray | float) -> np.ndarray:
        if self.interpolation_kind != 1:
            raise ValueError(f"unsupported Moon clock interpolation {self.interpolation_kind}")
        x = (np.asarray(jd, dtype=np.float64) - self.domain_start_jd) / self.century_days
        k = np.floor(x).astype(np.int64)
        u = x - k
        k = np.clip(k, 0, len(self.values) - 2)
        u = np.where((x < 0) | (x > len(self.values) - 1), 0.0, u)
        vals = self.values.astype(np.float64) * self.quantum_days
        return vals[k] + (vals[k + 1] - vals[k]) * u


@dataclass(frozen=True)
class OpmHeader:
    magic: bytes
    endian_tag: int
    header_minor_version: int
    fixed_header_size: int
    header_size: int
    source_start_jd: float
    source_end_jd: float
    coverage_start_jd: float
    coverage_span_days: float
    body_id: int
    center_id: int
    storage_vector_id: int
    model_kind: int
    clock_kind: int
    frame_kind: int
    reference_shape_kind: int
    residual_encoding_kind: int
    segment_addressing_kind: int
    axis_count: int
    segment_count: int
    segment_days: float
    period_days: float
    phase_start_jd: float
    edge_margin_days: float
    event_search_step_days: float
    expansion: float
    residual_degree: int
    reference_shape_degree: int
    quant_table_count: int
    width_table_count: int
    clock_correction_kind: int
    clock_correction_degree: int
    clock_table_offset: int
    clock_table_size: int
    quant_table_offset: int
    quant_table_size: int
    width_table_offset: int
    width_table_size: int
    model_table_offset: int
    model_table_size: int
    payload_offset: int
    payload_size: int
    file_size: int
    header_crc64: int
    payload_crc64: int


@dataclass(frozen=True)
class OpmFile:
    path: Path
    header: OpmHeader
    quant_steps: np.ndarray
    widths: np.ndarray
    qcoeffs: np.ndarray
    shape_x: np.ndarray | None
    shape_y: np.ndarray | None
    frame_coeffs: np.ndarray | None
    clock_table: bytes


def parse_header(data: bytes) -> OpmHeader:
    f = HEADER_STRUCT.unpack(data[:FIXED_HEADER_SIZE])
    return OpmHeader(
        magic=f[0],
        endian_tag=f[1],
        header_minor_version=f[2],
        fixed_header_size=f[4],
        header_size=f[5],
        source_start_jd=f[10],
        source_end_jd=f[11],
        coverage_start_jd=f[12],
        coverage_span_days=f[13],
        body_id=f[14],
        center_id=f[15],
        storage_vector_id=f[16],
        model_kind=f[19],
        clock_kind=f[20],
        frame_kind=f[21],
        reference_shape_kind=f[22],
        residual_encoding_kind=f[23],
        segment_addressing_kind=f[24],
        axis_count=f[25],
        segment_count=f[26],
        segment_days=f[27],
        period_days=f[28],
        phase_start_jd=f[29],
        edge_margin_days=f[30],
        event_search_step_days=f[31],
        expansion=float(f[32]),
        residual_degree=f[35],
        reference_shape_degree=f[36],
        quant_table_count=f[37],
        width_table_count=f[38],
        clock_correction_kind=f[39],
        clock_correction_degree=f[40],
        clock_table_offset=f[46],
        clock_table_size=f[47],
        quant_table_offset=f[48],
        quant_table_size=f[49],
        width_table_offset=f[50],
        width_table_size=f[51],
        model_table_offset=f[52],
        model_table_size=f[53],
        payload_offset=f[54],
        payload_size=f[55],
        file_size=f[56],
        header_crc64=f[57],
        payload_crc64=f[58],
    )


def body_name_from_id(body_id: int) -> str:
    for name, value in OPM_BODY_IDS.items():
        if value == body_id:
            return name
    raise ValueError(f"unknown OPM body id {body_id}")


def verify_crc64(data: bytes, header: OpmHeader, path: Path) -> None:
    if header.payload_crc64:
        actual_payload = crc64_ecma(data[FIXED_HEADER_SIZE:])
        if actual_payload != header.payload_crc64:
            raise ValueError(f"{path}: payload CRC64 mismatch header={header.payload_crc64:016x} actual={actual_payload:016x}")
    if header.header_crc64:
        header_bytes = bytearray(data[:FIXED_HEADER_SIZE])
        struct.pack_into("<Q", header_bytes, HEADER_CRC64_OFFSET, 0)
        actual_header = crc64_ecma(bytes(header_bytes))
        if actual_header != header.header_crc64:
            raise ValueError(f"{path}: header CRC64 mismatch header={header.header_crc64:016x} actual={actual_header:016x}")


def read_opm(path: Path, *, check_crc: bool = True) -> OpmFile:
    data = path.read_bytes()
    h = parse_header(data)
    if h.magic != MAGIC:
        raise ValueError(f"{path}: bad magic {h.magic!r}")
    if h.endian_tag != ENDIAN_TAG:
        raise ValueError(f"{path}: bad endian tag")
    if h.fixed_header_size != FIXED_HEADER_SIZE:
        raise ValueError(f"{path}: unsupported fixed header size {h.fixed_header_size}")
    if h.file_size != len(data):
        raise ValueError(f"{path}: file size mismatch header={h.file_size} actual={len(data)}")
    if check_crc:
        verify_crc64(data, h, path)
    if h.axis_count != AXIS_COUNT:
        raise ValueError(f"{path}: unsupported axis_count={h.axis_count}")

    quant = np.frombuffer(data[h.quant_table_offset : h.quant_table_offset + h.quant_table_size], dtype="<f4").astype(np.float64)
    widths = np.frombuffer(data[h.width_table_offset : h.width_table_offset + h.width_table_size], dtype=np.uint8).reshape((AXIS_COUNT, h.quant_table_count))
    if len(quant) != h.residual_degree + 1:
        raise ValueError(f"{path}: quant table count mismatch")
    if widths.shape != (AXIS_COUNT, h.residual_degree + 1):
        raise ValueError(f"{path}: width table shape mismatch")

    payload = data[h.payload_offset : h.payload_offset + h.payload_size]
    qcoeffs = np.zeros((h.segment_count, AXIS_COUNT, h.residual_degree + 1), dtype=np.int64)
    cursor = 0
    for axis in range(AXIS_COUNT):
        bit_count = int(np.sum(widths[axis].astype(np.int64))) * h.segment_count
        byte_count = (bit_count + 7) // 8
        stream = payload[cursor : cursor + byte_count]
        cursor += byte_count
        arrays = pack.unpack_degree_major(stream, h.segment_count, h.residual_degree, widths[axis])
        for si, arr in enumerate(arrays):
            qcoeffs[si, axis, :] = arr
    if cursor != len(payload):
        raise ValueError(f"{path}: payload split mismatch used={cursor} size={len(payload)}")

    shape_x = shape_y = None
    frame_coeffs = None
    if h.model_table_size:
        model = np.frombuffer(data[h.model_table_offset : h.model_table_offset + h.model_table_size], dtype="<f8").copy()
        if h.reference_shape_degree == 255:
            raise ValueError(f"{path}: model table present but no reference shape degree")
        shape_count = h.reference_shape_degree + 1
        if h.frame_kind == FRAME_CHEB1_PLANE_APSIS:
            frame_rows = 3
        elif h.frame_kind == FRAME_CHEB1_NORMAL_APSIS:
            frame_rows = 4
        else:
            raise ValueError(f"{path}: unsupported frame_kind={h.frame_kind}")
        need = 2 * shape_count + frame_rows * 2
        if len(model) != need:
            raise ValueError(f"{path}: model table f64 count mismatch got={len(model)} expected={need}")
        shape_x = model[:shape_count]
        shape_y = model[shape_count : 2 * shape_count]
        frame_coeffs = model[2 * shape_count :].reshape((frame_rows, 2))

    clock_table = data[h.clock_table_offset : h.clock_table_offset + h.clock_table_size] if h.clock_table_size else b""
    return OpmFile(path, h, quant, widths, qcoeffs, shape_x, shape_y, frame_coeffs, clock_table)


def expanded_bounds(a: float, b: float, expansion: float) -> tuple[float, float]:
    pad = float(expansion) * (b - a)
    return a - pad, b + pad


def normalize_expanded(jd: np.ndarray, a: float, b: float, expansion: float) -> np.ndarray:
    ea, eb = expanded_bounds(a, b, expansion)
    return proto.normalize_time(jd, ea, eb)


def mercury_clock(opm: OpmFile) -> ParsedMercuryClock | None:
    h = opm.header
    if h.clock_correction_kind == CLOCK_CORRECTION_CHEB_EVENT_TIME_F64:
        return ParsedMercuryClock.from_bytes(opm.clock_table)
    return None


def moon_clock(opm: OpmFile) -> ParsedMoonClock | None:
    h = opm.header
    if h.clock_correction_kind == CLOCK_CORRECTION_CENTURY_I16_LINEAR_TABLE:
        return ParsedMoonClock.from_bytes(opm.clock_table)
    return None


def segment_bounds(
    h: OpmHeader,
    segment_index: int,
    clock: ParsedMercuryClock | ParsedMoonClock | None = None,
) -> tuple[float, float]:
    if h.clock_correction_kind == CLOCK_CORRECTION_CHEB_EVENT_TIME_F64:
        if not isinstance(clock, ParsedMercuryClock):
            raise ValueError("Mercury clock correction requested but missing parsed clock")
        base0 = h.phase_start_jd + (clock.file_event_index_start + segment_index) * h.period_days
        base1 = h.phase_start_jd + (clock.file_event_index_start + segment_index + 1) * h.period_days
        return (
            float(base0 + clock.correction_for_indices(segment_index)),
            float(base1 + clock.correction_for_indices(segment_index + 1)),
        )
    if h.clock_correction_kind == CLOCK_CORRECTION_CENTURY_I16_LINEAR_TABLE:
        if not isinstance(clock, ParsedMoonClock):
            raise ValueError("Moon clock correction requested but missing parsed clock")
        base0 = h.phase_start_jd + segment_index * h.period_days
        base1 = h.phase_start_jd + (segment_index + 1) * h.period_days
        return (float(base0 + clock.eval(base0)), float(base1 + clock.eval(base1)))
    if h.segment_addressing_kind in {SEGMENT_FIXED_DAYS, SEGMENT_AFFINE_PERIOD_PHASE}:
        d = h.segment_days if h.segment_addressing_kind == SEGMENT_FIXED_DAYS else h.period_days
        a = h.phase_start_jd + segment_index * d
        return a, a + d
    raise ValueError(f"unsupported segment_addressing_kind={h.segment_addressing_kind}")


def frame_params_for_segments(opm: OpmFile, clock: ParsedMercuryClock | ParsedMoonClock | None = None) -> np.ndarray:
    h = opm.header
    assert opm.frame_coeffs is not None
    mids = np.asarray([0.5 * sum(segment_bounds(h, i, clock)) for i in range(h.segment_count)], dtype=np.float64)
    if len(mids) == 1:
        tnorm = np.zeros(1, dtype=np.float64)
    else:
        tnorm = proto.normalize_time(mids, mids[0], mids[-1])
    params = np.column_stack([proto.cheb_eval(opm.frame_coeffs[i], tnorm) for i in range(opm.frame_coeffs.shape[0])])
    if h.frame_kind == FRAME_CHEB1_NORMAL_APSIS:
        normals = params[:, :3]
        norms = np.linalg.norm(normals, axis=1)
        if np.any(norms <= 0.0):
            raise ValueError(f"{opm.path}: invalid evaluated normal frame")
        params[:, :3] = normals / norms[:, None]
    return params


def reconstruct_positions(opm: OpmFile, jds: np.ndarray) -> np.ndarray:
    h = opm.header
    jds = np.asarray(jds, dtype=np.float64)
    out = np.full((len(jds), 3), np.nan, dtype=np.float64)
    filled = np.zeros(len(jds), dtype=bool)
    coeffs = opm.qcoeffs.astype(np.float64) * opm.quant_steps[None, None, :]

    clock = mercury_clock(opm) or moon_clock(opm)
    params = frame_params_for_segments(opm, clock) if opm.frame_coeffs is not None else None
    boundary_tol = 1e-9
    for si in range(h.segment_count):
        a, b = segment_bounds(h, si, clock)
        lo = max(a, h.coverage_start_jd)
        hi = min(b, h.coverage_start_jd + h.coverage_span_days)
        if si == h.segment_count - 1:
            mask = (jds >= lo - boundary_tol) & (jds <= hi + boundary_tol) & ~filled
        else:
            mask = (jds >= lo - boundary_tol) & (jds < hi + boundary_tol) & ~filled
        if not np.any(mask):
            continue
        tau = normalize_expanded(jds[mask], a, b, h.expansion)
        if h.model_kind == MODEL_RAW_XYZ_CHEB:
            out[mask] = np.column_stack([proto.cheb_eval(coeffs[si, axis], tau) for axis in range(AXIS_COUNT)])
        elif h.model_kind in {MODEL_FIXED_FRAME_SHAPE, MODEL_MEAN_APSIS_FRAME_SHAPE, MODEL_MEAN_LUNAR_APSIS_FRAME_SHAPE}:
            if opm.shape_x is None or opm.shape_y is None or params is None:
                raise ValueError(f"{opm.path}: orbital-frame model missing model table")
            aligned = np.column_stack([
                proto.cheb_eval(opm.shape_x, tau) + proto.cheb_eval(coeffs[si, 0], tau),
                proto.cheb_eval(opm.shape_y, tau) + proto.cheb_eval(coeffs[si, 1], tau),
                proto.cheb_eval(coeffs[si, 2], tau),
            ])
            out[mask] = proto.unalign_positions_normal(aligned, params[si, :3], float(params[si, 3])) if params.shape[1] == 4 else proto.unalign_positions(aligned, float(params[si, 0]), float(params[si, 1]), float(params[si, 2]))
        else:
            raise ValueError(f"{opm.path}: unsupported model_kind={h.model_kind}")
        filled[mask] = True
    if not np.all(filled):
        missing = jds[~filled]
        raise ValueError(f"{opm.path}: {len(missing)} JD(s) outside OPM coverage; first missing JD={missing[0]:.12f}")
    if np.any(~np.isfinite(out)):
        raise ValueError(f"{opm.path}: non-finite reconstruction")
    return out


def truth_positions(spk: SPK, opm: OpmFile, jds: np.ndarray) -> np.ndarray:
    provider, closeable = truth_position_provider(spk, opm)
    try:
        return provider.position(jds)
    finally:
        close_if_needed(closeable)


def truth_velocities(spk: SPK, opm: OpmFile, jds: np.ndarray) -> np.ndarray:
    provider, closeable = truth_position_provider(spk, opm)
    try:
        if not hasattr(provider, "velocity"):
            raise ValueError(f"{opm.path}: truth provider does not expose velocity")
        return provider.velocity(jds)
    finally:
        close_if_needed(closeable)


def close_if_needed(closeable: object | None) -> None:
    if closeable is not None and hasattr(closeable, "close"):
        closeable.close()


def truth_position_provider(spk: SPK, opm: OpmFile) -> tuple[object, object | None]:
    body = body_name_from_id(opm.header.body_id)
    if opm.header.storage_vector_id == STORAGE_SSB_TO_BODY:
        return BaryProvider(spk, SPK_TARGET_IDS[body]), None
    if opm.header.storage_vector_id == STORAGE_SUN_TO_BODY:
        provider = proto.HelioProvider(SPK_TARGET_IDS[body])
        return provider, provider
    if opm.header.storage_vector_id == STORAGE_EARTH_TO_MOON:
        provider = moon_proto.GeoMoonProvider()
        return provider, provider
    raise ValueError(f"{opm.path}: unsupported storage vector {opm.header.storage_vector_id}")


def cheb_eval_segment_coeffs(coeffs: np.ndarray, tau: np.ndarray) -> np.ndarray:
    degree = coeffs.shape[1] - 1
    vander = np.polynomial.chebyshev.chebvander(tau.reshape(-1), degree)
    vander = vander.reshape((tau.shape[0], tau.shape[1], degree + 1))
    return np.einsum("snd,sd->sn", vander, coeffs)


def cheb_derivative_coeffs(coeffs: np.ndarray) -> np.ndarray:
    return np.polynomial.chebyshev.chebder(coeffs, axis=-1)


def cheb_eval_segment_derivative_coeffs(coeffs: np.ndarray, tau: np.ndarray, scale: np.ndarray) -> np.ndarray:
    return cheb_eval_segment_coeffs(coeffs, tau) * scale[:, None]


def cheb_eval_global_derivative_coeffs(coeffs: np.ndarray, tau: np.ndarray, scale: np.ndarray) -> np.ndarray:
    return np.polynomial.chebyshev.chebval(tau, coeffs) * scale[:, None]
def plane_frames_for_params(params: np.ndarray) -> np.ndarray:
    if params.shape[1] == 4:
        normals = params[:, :3].astype(np.float64, copy=True)
        norms = np.linalg.norm(normals, axis=1)
        if np.any(norms <= 0.0):
            raise ValueError("invalid normal-frame params")
        normals /= norms[:, None]
        nx = normals[:, 0]
        ny = normals[:, 1]
        nz = normals[:, 2]
        frames = np.empty((len(params), 3, 3), dtype=np.float64)
        mask = nz > -1.0 + 1e-12
        inv = np.empty_like(nz)
        inv[mask] = 1.0 / (1.0 + nz[mask])
        frames[mask, 0, 0] = 1.0 - nx[mask] * nx[mask] * inv[mask]
        frames[mask, 1, 0] = -nx[mask] * ny[mask] * inv[mask]
        frames[mask, 2, 0] = -nx[mask]
        frames[mask, 0, 1] = -nx[mask] * ny[mask] * inv[mask]
        frames[mask, 1, 1] = 1.0 - ny[mask] * ny[mask] * inv[mask]
        frames[mask, 2, 1] = -ny[mask]
        frames[mask, :, 2] = normals[mask]
        if np.any(~mask):
            frames[~mask, :, 0] = np.asarray([1.0, 0.0, 0.0])
            frames[~mask, :, 1] = np.asarray([0.0, -1.0, 0.0])
            frames[~mask, :, 2] = normals[~mask]
        return frames
    plane_u = params[:, 0]
    plane_v = params[:, 1]
    den_inv = 1.0 / (1.0 + plane_u * plane_u + plane_v * plane_v)
    frames = np.empty((len(params), 3, 3), dtype=np.float64)
    frames[:, 0, 0] = (1.0 + plane_v * plane_v - plane_u * plane_u) * den_inv
    frames[:, 1, 0] = 2.0 * plane_v * plane_u * den_inv
    frames[:, 2, 0] = -2.0 * plane_u * den_inv
    frames[:, 0, 1] = 2.0 * plane_v * plane_u * den_inv
    frames[:, 1, 1] = (1.0 - plane_v * plane_v + plane_u * plane_u) * den_inv
    frames[:, 2, 1] = 2.0 * plane_v * den_inv
    frames[:, 0, 2] = 2.0 * plane_u * den_inv
    frames[:, 1, 2] = -2.0 * plane_v * den_inv
    frames[:, 2, 2] = (1.0 - plane_u * plane_u - plane_v * plane_v) * den_inv
    return frames


def unalign_positions_batched(aligned: np.ndarray, params: np.ndarray) -> np.ndarray:
    apsis_angle = params[:, -1]
    c = np.cos(apsis_angle)
    s = np.sin(apsis_angle)
    local = np.empty_like(aligned)
    local[:, :, 0] = c[:, None] * aligned[:, :, 0] - s[:, None] * aligned[:, :, 1]
    local[:, :, 1] = s[:, None] * aligned[:, :, 0] + c[:, None] * aligned[:, :, 1]
    local[:, :, 2] = aligned[:, :, 2]
    frames = plane_frames_for_params(params)
    return np.einsum("snj,skj->snk", local, frames)


def segment_chunk_nodes(
    opm: OpmFile,
    start_segment: int,
    stop_segment: int,
    nodes_per_segment: int,
    clock: ParsedMercuryClock | ParsedMoonClock | None,
    *,
    include_endpoints: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    h = opm.header
    coverage_end = h.coverage_start_jd + h.coverage_span_days
    segment_indices: list[int] = []
    bounds: list[tuple[float, float]] = []
    nodes_parts: list[np.ndarray] = []
    for si in range(start_segment, stop_segment):
        a, b = segment_bounds(h, si, clock)
        lo = max(a, h.coverage_start_jd)
        hi = min(b, coverage_end)
        if hi <= lo:
            continue
        segment_indices.append(si)
        bounds.append((a, b))
        nodes = proto.cheb_nodes(lo, hi, nodes_per_segment)
        if include_endpoints:
            nodes = np.unique(np.concatenate([nodes, np.asarray([lo, hi], dtype=np.float64)]))
        nodes_parts.append(nodes)
    if not segment_indices:
        node_count = nodes_per_segment + (2 if include_endpoints else 0)
        return (
            np.empty((0,), dtype=np.int64),
            np.empty((0, node_count), dtype=np.float64),
            np.empty((0,), dtype=np.float64),
            np.empty((0,), dtype=np.float64),
        )
    bound_arr = np.asarray(bounds, dtype=np.float64)
    return (
        np.asarray(segment_indices, dtype=np.int64),
        np.vstack(nodes_parts),
        bound_arr[:, 0],
        bound_arr[:, 1],
    )


def reconstruct_segment_nodes(
    opm: OpmFile,
    segment_indices: np.ndarray,
    tau: np.ndarray,
    coeffs: np.ndarray,
    params: np.ndarray | None,
) -> np.ndarray:
    h = opm.header
    segment_coeffs = coeffs[segment_indices]
    if h.model_kind == MODEL_RAW_XYZ_CHEB:
        return np.stack(
            [cheb_eval_segment_coeffs(segment_coeffs[:, axis, :], tau) for axis in range(AXIS_COUNT)],
            axis=2,
        )
    if h.model_kind in {MODEL_FIXED_FRAME_SHAPE, MODEL_MEAN_APSIS_FRAME_SHAPE, MODEL_MEAN_LUNAR_APSIS_FRAME_SHAPE}:
        if opm.shape_x is None or opm.shape_y is None or params is None:
            raise ValueError(f"{opm.path}: orbital-frame model missing model table")
        aligned = np.empty((len(segment_indices), tau.shape[1], AXIS_COUNT), dtype=np.float64)
        aligned[:, :, 0] = proto.cheb_eval(opm.shape_x, tau) + cheb_eval_segment_coeffs(segment_coeffs[:, 0, :], tau)
        aligned[:, :, 1] = proto.cheb_eval(opm.shape_y, tau) + cheb_eval_segment_coeffs(segment_coeffs[:, 1, :], tau)
        aligned[:, :, 2] = cheb_eval_segment_coeffs(segment_coeffs[:, 2, :], tau)
        return unalign_positions_batched(aligned, params[segment_indices])
    raise ValueError(f"{opm.path}: unsupported model_kind={h.model_kind}")


def reconstruct_segment_node_velocities(
    opm: OpmFile,
    segment_indices: np.ndarray,
    tau: np.ndarray,
    dcoeffs: np.ndarray,
    params: np.ndarray | None,
    scale: np.ndarray,
) -> np.ndarray:
    h = opm.header
    segment_dcoeffs = dcoeffs[segment_indices]
    if h.model_kind == MODEL_RAW_XYZ_CHEB:
        return np.stack(
            [cheb_eval_segment_derivative_coeffs(segment_dcoeffs[:, axis, :], tau, scale) for axis in range(AXIS_COUNT)],
            axis=2,
        )
    if h.model_kind in {MODEL_FIXED_FRAME_SHAPE, MODEL_MEAN_APSIS_FRAME_SHAPE, MODEL_MEAN_LUNAR_APSIS_FRAME_SHAPE}:
        if opm.shape_x is None or opm.shape_y is None or params is None:
            raise ValueError(f"{opm.path}: orbital-frame model missing model table")
        dshape_x = cheb_derivative_coeffs(opm.shape_x)
        dshape_y = cheb_derivative_coeffs(opm.shape_y)
        aligned = np.empty((len(segment_indices), tau.shape[1], AXIS_COUNT), dtype=np.float64)
        aligned[:, :, 0] = cheb_eval_global_derivative_coeffs(dshape_x, tau, scale) + cheb_eval_segment_derivative_coeffs(segment_dcoeffs[:, 0, :], tau, scale)
        aligned[:, :, 1] = cheb_eval_global_derivative_coeffs(dshape_y, tau, scale) + cheb_eval_segment_derivative_coeffs(segment_dcoeffs[:, 1, :], tau, scale)
        aligned[:, :, 2] = cheb_eval_segment_derivative_coeffs(segment_dcoeffs[:, 2, :], tau, scale)
        return unalign_positions_batched(aligned, params[segment_indices])
    raise ValueError(f"{opm.path}: unsupported model_kind={h.model_kind}")


def validate_segment_chunk(
    provider: object,
    opm: OpmFile,
    coeffs: np.ndarray,
    params: np.ndarray | None,
    clock: ParsedMercuryClock | ParsedMoonClock | None,
    start_segment: int,
    stop_segment: int,
    nodes_per_segment: int,
) -> np.ndarray:
    segment_indices, jds, a, b = segment_chunk_nodes(opm, start_segment, stop_segment, nodes_per_segment, clock)
    if len(segment_indices) == 0:
        return np.empty((0,), dtype=np.float64)
    width = b - a
    expanded_a = a - opm.header.expansion * width
    expanded_b = b + opm.header.expansion * width
    tau = (2.0 * jds - expanded_a[:, None] - expanded_b[:, None]) / (expanded_b - expanded_a)[:, None]
    recon = reconstruct_segment_nodes(opm, segment_indices, tau, coeffs, params).reshape((-1, AXIS_COUNT))
    truth = provider.position(jds.reshape(-1))
    return proto.angular_errors_arcsec(truth, recon)


def native_residual_segment_chunk(
    provider: object,
    opm: OpmFile,
    coeffs: np.ndarray,
    dcoeffs: np.ndarray,
    params: np.ndarray | None,
    clock: ParsedMercuryClock | ParsedMoonClock | None,
    start_segment: int,
    stop_segment: int,
    nodes_per_segment: int,
    *,
    include_endpoints: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not hasattr(provider, "velocity"):
        raise ValueError(f"{opm.path}: truth provider does not expose velocity")
    segment_indices, jds, a, b = segment_chunk_nodes(opm, start_segment, stop_segment, nodes_per_segment, clock, include_endpoints=include_endpoints)
    if len(segment_indices) == 0:
        empty = np.empty((0,), dtype=np.float64)
        return empty, empty, empty
    width = b - a
    expanded_a = a - opm.header.expansion * width
    expanded_b = b + opm.header.expansion * width
    scale = 2.0 / (expanded_b - expanded_a)
    tau = (2.0 * jds - expanded_a[:, None] - expanded_b[:, None]) / (expanded_b - expanded_a)[:, None]
    flat_jds = jds.reshape(-1)
    recon_pos = reconstruct_segment_nodes(opm, segment_indices, tau, coeffs, params).reshape((-1, AXIS_COUNT))
    recon_vel = reconstruct_segment_node_velocities(opm, segment_indices, tau, dcoeffs, params, scale).reshape((-1, AXIS_COUNT))
    truth_pos = provider.position(flat_jds)
    truth_vel = provider.velocity(flat_jds)
    pos_err_km = np.linalg.norm(recon_pos - truth_pos, axis=1)
    vel_err_km_day = np.linalg.norm(recon_vel - truth_vel, axis=1)
    return flat_jds, pos_err_km, vel_err_km_day


def validate_one(
    spk: SPK,
    path: Path,
    nodes_per_segment: int,
    *,
    check_crc: bool = True,
    segment_chunk_size: int = 4096,
    progress: bool = False,
    progress_segments: int = 50000,
) -> tuple[str, int, float, float, float, float, str]:
    opm = read_opm(path, check_crc=check_crc)
    h = opm.header
    if segment_chunk_size <= 0:
        raise ValueError("segment_chunk_size must be positive")
    coeffs = opm.qcoeffs.astype(np.float64) * opm.quant_steps[None, None, :]
    clock = mercury_clock(opm) or moon_clock(opm)
    params = frame_params_for_segments(opm, clock) if opm.frame_coeffs is not None else None
    err_parts = []
    provider, closeable = truth_position_provider(spk, opm)
    last_progress = 0
    try:
        for start in range(0, h.segment_count, segment_chunk_size):
            stop = min(start + segment_chunk_size, h.segment_count)
            err = validate_segment_chunk(provider, opm, coeffs, params, clock, start, stop, nodes_per_segment)
            if len(err):
                err_parts.append(err)
            if progress and (stop == h.segment_count or stop - last_progress >= progress_segments):
                print(f"  validated {stop}/{h.segment_count} segments for {path}", flush=True)
                last_progress = stop
    finally:
        close_if_needed(closeable)
    if not err_parts:
        raise ValueError(f"{path}: no validation samples inside coverage")
    err = np.concatenate(err_parts)
    body = body_name_from_id(h.body_id)
    if body == "sun":
        status = "DIAG"
    else:
        status = "PASS" if float(np.max(err)) <= 0.001 else "MISS"
    return (
        body,
        h.segment_count,
        float(np.percentile(err, 50)),
        float(np.percentile(err, 95)),
        float(np.percentile(err, 99)),
        float(np.max(err)),
        status,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate OPM1 coverage files against DE441")
    parser.add_argument("paths", nargs="*", type=Path, help="OPM files or directories; default j2000-opm/*.opm")
    parser.add_argument("--nodes-per-segment", type=int, default=32)
    parser.add_argument("--segment-chunk-size", type=int, default=4096, help="segments to validate per vectorized chunk")
    parser.add_argument("--progress", action="store_true", help="print validation progress for large files")
    parser.add_argument("--progress-segments", type=int, default=50000, help="progress interval in segments")
    parser.add_argument("--de441", type=Path, required=True, help="path to DE441 BSP file")
    parser.add_argument("--no-crc", action="store_true", help="skip CRC64 validation when reading OPM files")
    return parser.parse_args()


def expand_paths(paths: list[Path]) -> list[Path]:
    if not paths:
        paths = [SCRIPT_DIR / "j2000-opm"]
    out: list[Path] = []
    for path in paths:
        if path.is_dir():
            out.extend(sorted(path.glob("*.opm")))
        else:
            out.append(path)
    return out


def main() -> int:
    args = parse_args()
    proto.set_de441_path(args.de441)
    moon_proto.set_de441_path(args.de441)
    paths = expand_paths(args.paths)
    if not paths:
        raise SystemExit("no .opm files found")
    failures = 0
    print("body segments p50_arcsec p95_arcsec p99_arcsec max_arcsec status path")
    with SPK.open(str(args.de441)) as spk:
        for path in paths:
            body, segments, p50, p95, p99, max_err, status = validate_one(
                spk,
                path,
                args.nodes_per_segment,
                check_crc=not args.no_crc,
                segment_chunk_size=args.segment_chunk_size,
                progress=args.progress,
                progress_segments=args.progress_segments,
            )
            if status == "MISS":
                failures += 1
            print(f"{body} {segments} {p50:.9g} {p95:.9g} {p99:.9g} {max_err:.9g} {status} {path}")
    print(f"{('PASS' if failures == 0 else 'FAIL')}: {len(paths) - failures}/{len(paths)} OPM files passed")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
