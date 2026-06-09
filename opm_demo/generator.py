#!/usr/bin/env python3
"""Generate coverage-range OPM1 files for the reference Python demo.

A OPM file describes one body over the Julian-date range stored in its coverage
fields.  Century slicing is only one packaging strategy, not part of the file
format semantics.
"""
from __future__ import annotations

import argparse
import json
import math
import shutil
import struct
import subprocess
import sys
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
from jplephem.spk import SPK

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import opm_demo.packing as pack  # noqa: E402
from opm_demo.body_configs import CONFIGS, DEFAULT_BODY_ORDER, MOON_CENTURY_TABLE, BodyConfig  # noqa: E402
import opm_demo.orbit_model as proto  # noqa: E402
import opm_demo.moon_model as moon_proto  # noqa: E402
from opm_demo.format import HEADER_CRC64_OFFSET, PAYLOAD_CRC64_OFFSET, crc64_ecma  # noqa: E402


JD_J2000 = 2451545.0
CENTURY_DAYS = 36525.0
SOURCE_DE441 = 1
POSITION_KM = 1
TIME_TDB_JD = 1
AXIS_COUNT = 3

MAGIC = b"OPM1"
ENDIAN_TAG = 0x0102
FIXED_HEADER_SIZE = 320
HEADER_MINOR_VERSION = 0
FLAGS = 0

OPM_BODY_IDS = {
    "ssb": 1,
    "sun": 10,
    "mercury": 199,
    "venus": 299,
    "emb": 3,
    "earth": 399,
    "moon": 301,
    "mars": 499,
    "jupiter": 599,
    "saturn": 699,
    "uranus": 799,
    "neptune": 899,
    "pluto": 999,
}

SPK_TARGET_IDS = {
    "sun": 10,
    "mercury": 1,
    "venus": 2,
    "emb": 3,
    "mars": 4,
    "jupiter": 5,
    "saturn": 6,
    "uranus": 7,
    "neptune": 8,
    "pluto": 9,
}

STORAGE_CENTER_TO_BODY = 1
STORAGE_SSB_TO_BODY = 2
STORAGE_SUN_TO_BODY = 3
STORAGE_EARTH_TO_MOON = 4

MODEL_KIND = {
    "raw_xyz_cheb": 1,
    "mean_apsis_frame_shape": 2,
    "mean_lunar_apsis_frame_shape": 3,
    "fixed_frame_shape": 4,
}

CLOCK_KIND = {
    "raw_fixed": 0,
    "global_anomalistic": 2,
    "global_anomalistic_cheb8": 3,
    "global_anomalistic_century_i16": 4,
    "global_inertial_phase": 5,
    "fixed_period": 6,
}

FRAME_NONE = 0
FRAME_CHEB1_PLANE_APSIS = 1
REFERENCE_SHAPE_NONE = 0
REFERENCE_SHAPE_MEAN_XY_CHEB = 1
RESIDUAL_XYZ_DEGREE_MAJOR_EXACT_WIDTH = 1
SEGMENT_FIXED_DAYS = 0
SEGMENT_AFFINE_PERIOD_PHASE = 1
SEGMENT_EXPLICIT_BOUNDARIES_F64 = 2
SEGMENT_FLAG_HAS_EXPLICIT_BOUNDARY_BLOCK = 1 << 0

CLOCK_CORRECTION_NONE = 0
CLOCK_CORRECTION_CHEB_EVENT_TIME_F64 = 1
CLOCK_CORRECTION_CENTURY_I16_LINEAR_TABLE = 2

MERCURY_CLOCK_DEGREE = 8
MERCURY_CLOCK_CONSTANTS_PATH = SCRIPT_DIR.parent / "data" / "opm_mercury_cheb8_clock.json"
MOON_CLOCK_INTERPOLATION_LINEAR = 1
MOON_CLOCK_STATISTIC_MEDIAN = 1
MOON_CLOCK_CONSTANTS_PATH = SCRIPT_DIR.parent / "data" / "opm_moon_century_i16_clock.json"
MOON_CLOCK_TABLE_STRUCT = struct.Struct("<dddIIII")

DESCRIPTOR_MODEL_POLICY_V1 = 100
DESCRIPTOR_ENTRY_STRUCT = struct.Struct("<HHIQ")
MODEL_POLICY_STRUCT = struct.Struct("<12HII")

EVENT_NONE = 0
EVENT_PERIHELION = 1
EVENT_PERIGEE = 2
EVENT_INERTIAL_PHASE_CYCLE = 3
EVENT_FIXED_TIME_BOUNDARY = 4

EVENT_SOURCE_NONE = 0
EVENT_SOURCE_CENTER_TO_BODY = 1
EVENT_SOURCE_SSB_TO_BODY = 2
EVENT_SOURCE_SUN_TO_BODY = 3
EVENT_SOURCE_EARTH_TO_MOON = 4

BOUNDARY_NONE_OR_FIXED_TIME = 0
BOUNDARY_TRUE_DETECTED_EVENTS = 1
BOUNDARY_GLOBAL_MEAN_EVENT_FIT = 2
BOUNDARY_GLOBAL_MEAN_PLUS_CHEB_CORRECTION = 3
BOUNDARY_GLOBAL_MEAN_PLUS_CENTURY_I16_TABLE = 4
BOUNDARY_GLOBAL_INERTIAL_PHASE_FIT = 5
BOUNDARY_EXPLICIT_BOUNDARIES_ONLY = 6

DETECTION_NONE = 0
DETECTION_RADIUS_MIN_GRID_ARGRELEXTREMA = 1
DETECTION_RADIUS_MIN_REFINED = 2
DETECTION_UNWRAPPED_INERTIAL_PHASE_CROSSING = 3

EVENT_CORRECTION_NONE = 0
EVENT_CORRECTION_CHEB_EVENT_TIME = 1
EVENT_CORRECTION_CENTURY_I16_LINEAR = 2

FRAME_POLICY_NONE = 0
FRAME_POLICY_PER_SEGMENT_BEST_PLANE_APSIS = 1
FRAME_POLICY_CHEB1_BEST_PLANE_APSIS = 2
FRAME_POLICY_FIXED_GLOBAL_FRAME = 3

APSIS_POLICY_NONE = 0
APSIS_POLICY_MIN_RADIUS_DIRECTION = 1
APSIS_POLICY_FITTED_EVENT_DIRECTION = 2
APSIS_POLICY_FIXED_REFERENCE_AXIS = 3

SHAPE_POLICY_NONE = 0
SHAPE_POLICY_MEAN_XY = 1
SHAPE_POLICY_MEDIAN_XY = 2
SHAPE_POLICY_MIDRANGE_XY = 3
SHAPE_POLICY_TRIMMED_MEAN_XY = 4

RESIDUAL_POLICY_NONE = 0
RESIDUAL_POLICY_DIRECT_FIT_FROZEN_SHAPE = 1
RESIDUAL_POLICY_COEFF_SUBTRACTION_TRUNCATION = 2

QUANT_POLICY_EXPLICIT_STEPS = 0
WIDTH_POLICY_EXPLICIT_AXIS_DEGREE = 0
PAYLOAD_ORDER_AXIS_DEGREE_SEGMENT = 0

SSB = 0

HEADER_STRUCT = struct.Struct(
    "<"
    "4sHBBIIB3s"      # magic, endian, minor, reserved, sizes, flags
    "BB"              # source ID fields
    "dddd"            # source and coverage JD ranges
    "HH10B"           # body/center u16, compact enum fields
    "I"               # segment count
    "ddddd"           # segment/clock scalar fields
    "fB3s"            # expansion fraction, segment flags, reserved
    "6B"              # degree/count/correction compact fields
    "B7sQ"            # descriptor count, reserved, descriptor offset
    "15Q"             # block offsets/sizes, file size, CRC placeholders
    "58s"             # reserved tail; fixed header remains 320 bytes
)
assert HEADER_STRUCT.size == FIXED_HEADER_SIZE, HEADER_STRUCT.size


@dataclass
class BaryProvider:
    spk: SPK
    target_id: int

    def __post_init__(self) -> None:
        self.segments = sorted(
            [s for s in self.spk.segments if s.center == SSB and s.target == self.target_id],
            key=lambda s: s.start_jd,
        )
        if not self.segments:
            raise RuntimeError(f"No SPK segment center=0 target={self.target_id}")

    def position(self, jd_arr: np.ndarray) -> np.ndarray:
        tdb = np.asarray(jd_arr, dtype=np.float64)
        out = np.full((len(tdb), 3), np.nan, dtype=np.float64)
        for seg in self.segments:
            mask = (tdb >= seg.start_jd) & (tdb <= seg.end_jd)
            if np.any(mask):
                out[mask] = seg.compute(tdb[mask]).T
        if np.any(~np.isfinite(out)):
            bad = int(np.sum(~np.isfinite(out[:, 0])))
            raise RuntimeError(f"Missing SPK coverage for target {self.target_id}: {bad} samples")
        return out


@dataclass(frozen=True)
class PackedBody:
    cfg: BodyConfig
    boundaries: np.ndarray
    quant_steps: np.ndarray
    widths: np.ndarray
    qcoeffs: np.ndarray
    payload: bytes
    model_table: bytes
    clock_table: bytes
    p50: float
    p95: float
    p99: float
    max_err: float

    @property
    def segment_count(self) -> int:
        return int(len(self.boundaries) - 1)


def century_index_from_j2000(jd: float, *, rounding: str = "floor") -> int:
    value = (float(jd) - JD_J2000) / CENTURY_DAYS
    if rounding == "round":
        return int(round(value))
    return int(math.floor(value))


def source_bounds(spk: SPK) -> tuple[float, float]:
    starts = [float(seg.start_jd) for seg in spk.segments]
    ends = [float(seg.end_jd) for seg in spk.segments]
    return min(starts), max(ends)


def fixed_bounds(jd_start: float, jd_end: float, dseg: float) -> list[tuple[float, float]]:
    eps = 1e-8
    i0 = int(math.floor((jd_start - JD_J2000) / dseg))
    i1 = int(math.ceil((jd_end - JD_J2000) / dseg))
    out: list[tuple[float, float]] = []
    for i in range(i0, i1 + 1):
        a = JD_J2000 + i * dseg
        b = a + dseg
        if b <= jd_start + eps or a >= jd_end - eps:
            continue
        out.append((a, b))
    return out


def expanded_bounds(a: float, b: float, expansion: float) -> tuple[float, float]:
    if expansion == 0.0:
        return a, b
    pad = expansion * (b - a)
    return a - pad, b + pad


class RangeSafetyError(RuntimeError):
    """Requested coverage range cannot be safely fit from the source SPK."""


def body_config_for_generation(body: str) -> BodyConfig:
    cfg = replace(
        CONFIGS[body],
        segment_domain_expansion_fraction=float(np.float32(CONFIGS[body].segment_domain_expansion_fraction)),
    )
    if cfg.method == "mean_apsis_frame_shape" and body == "mercury":
        clock = mercury_clock()
        return replace(
            cfg,
            clock=replace(cfg.clock, period_days=clock.period_days, phase_start_jd=clock.phase_start_jd),
        )
    if cfg.method == "mean_lunar_apsis_frame_shape":
        clock = moon_century_clock()
        return replace(
            cfg,
            clock=replace(cfg.clock, period_days=clock.period_days, phase_start_jd=clock.phase_start_jd),
        )
    return cfg


def required_fit_bounds_for_body(cfg: BodyConfig, jd_start: float, jd_end: float) -> tuple[float, float]:
    if cfg.method in {"raw_xyz_cheb", "fixed_frame_shape"}:
        if cfg.segment_days is None:
            raise RangeSafetyError(f"{cfg.body}: fixed-segment method is missing segment_days")
        bounds = fixed_bounds(jd_start, jd_end, cfg.segment_days)
        if not bounds:
            raise RangeSafetyError(f"{cfg.body}: requested range does not overlap any fit segment")
        expanded = [expanded_bounds(a, b, cfg.segment_domain_expansion_fraction) for a, b in bounds]
        return min(a for a, _ in expanded), max(b for _, b in expanded)
    if cfg.method in {"mean_apsis_frame_shape", "mean_lunar_apsis_frame_shape"}:
        period = float(cfg.clock.period_days or 0.0)
        # Orbital/perigee models need source data outside the header coverage to
        # find complete event-to-event segments.  edge_margin_days covers the
        # explicit search margin; the period term covers the boundary segment that
        # overlaps the requested range, plus its expanded Chebyshev domain.
        margin = max(
            float(cfg.edge_margin_days) + period * float(cfg.segment_domain_expansion_fraction),
            period * (1.0 + float(cfg.segment_domain_expansion_fraction)),
        )
        return jd_start - margin, jd_end + margin
    raise RangeSafetyError(f"{cfg.body}: unsupported generation method {cfg.method}")


def validate_requested_range(
    spk: SPK,
    bodies: list[str],
    jd_start: float,
    days: float,
    *,
    range_safety: str,
) -> None:
    if not math.isfinite(jd_start) or not math.isfinite(days):
        raise RangeSafetyError("--jd-start and --days must be finite")
    if days <= 0.0:
        raise RangeSafetyError("--days must be positive")
    if range_safety == "none":
        return
    if range_safety != "strict":
        raise RangeSafetyError(f"unsupported range safety mode: {range_safety}")

    jd_end = jd_start + days
    if not math.isfinite(jd_end):
        raise RangeSafetyError("requested coverage end is not finite")
    source_start, source_end = source_bounds(spk)
    eps = 1e-9
    for body in bodies:
        cfg = body_config_for_generation(body)
        fit_start, fit_end = required_fit_bounds_for_body(cfg, jd_start, jd_end)
        if fit_start < source_start - eps or fit_end > source_end + eps:
            raise RangeSafetyError(
                f"requested range {jd_start:.9f}..{jd_end:.9f} for {body} requires fit samples "
                f"{fit_start:.9f}..{fit_end:.9f} ({cfg.method}), outside source coverage "
                f"{source_start:.9f}..{source_end:.9f}; choose a range farther from the BSP boundary"
            )


def cheb_nodes_expanded(a: float, b: float, n: int, expansion: float) -> np.ndarray:
    fa, fb = expanded_bounds(a, b, expansion)
    return proto.cheb_nodes(fa, fb, n)


def normalize_time_expanded(jd: np.ndarray | float, a: float, b: float, expansion: float) -> np.ndarray:
    fa, fb = expanded_bounds(a, b, expansion)
    return proto.normalize_time(jd, fa, fb)


def summarize_errors(truth: np.ndarray, recon: np.ndarray) -> tuple[float, float, float, float]:
    err = proto.angular_errors_arcsec(truth, recon)
    return (
        float(np.percentile(err, 50)),
        float(np.percentile(err, 95)),
        float(np.percentile(err, 99)),
        float(np.max(err)),
    )


@dataclass(frozen=True)
class MercuryClock:
    degree: int
    period_days: float
    phase_start_jd: float
    domain_start_jd: float
    event_count: int
    coeff: np.ndarray

    @property
    def domain_end_jd(self) -> float:
        return self.phase_start_jd + self.period_days * (self.event_count - 1)

    def normalize(self, jd: np.ndarray | float) -> np.ndarray:
        return 2.0 * (np.asarray(jd, dtype=np.float64) - self.domain_start_jd) / (self.domain_end_jd - self.domain_start_jd) - 1.0

    def boundary(self, indices: np.ndarray | int | float) -> np.ndarray:
        idx = np.asarray(indices, dtype=np.float64)
        base = self.phase_start_jd + self.period_days * idx
        corr = np.polynomial.chebyshev.chebval(self.normalize(base), self.coeff)
        return base + corr

    def bounds(self, jd_start: float, jd_end: float, min_len: float = 10.0) -> list[tuple[float, float]]:
        i0 = int(math.floor((jd_start - self.phase_start_jd) / self.period_days)) - 4
        i1 = int(math.ceil((jd_end - self.phase_start_jd) / self.period_days)) + 4
        inds = np.arange(i0, i1 + 1, dtype=np.int64)
        b = self.boundary(inds)
        out: list[tuple[float, float]] = []
        for a, c in zip(b[:-1], b[1:]):
            if c <= jd_start or a >= jd_end:
                continue
            if c > a + min_len:
                out.append((float(a), float(c)))
        return out

    def to_clock_table(self, file_event_index_start: int) -> bytes:
        return struct.pack(
            "<HHIdd",
            self.degree,
            len(self.coeff),
            0,
            float(file_event_index_start),
            float(self.event_count - 1),
        ) + np.asarray(self.coeff, dtype="<f8").tobytes()


_MERCURY_CLOCK_CACHE: MercuryClock | None = None


def mercury_clock() -> MercuryClock:
    global _MERCURY_CLOCK_CACHE
    if _MERCURY_CLOCK_CACHE is None:
        data = json.loads(MERCURY_CLOCK_CONSTANTS_PATH.read_text(encoding="utf-8"))
        coeff = np.asarray(data["coefficients"], dtype=np.float64)
        if int(data["degree"]) != MERCURY_CLOCK_DEGREE or len(coeff) != MERCURY_CLOCK_DEGREE + 1:
            raise RuntimeError("bad Mercury Cheb8 clock constants")
        _MERCURY_CLOCK_CACHE = MercuryClock(
            MERCURY_CLOCK_DEGREE,
            float(data["period_days"]),
            float(data["phase_start_jd"]),
            float(data["domain_start_jd"]),
            int(data["event_count"]),
            coeff,
        )
    return _MERCURY_CLOCK_CACHE


def opm_quant_steps(degree: int, base_km: float, pattern: str) -> np.ndarray:
    # OPM1 stores quant steps as f32. Quantize and reconstruct with the exact
    # stored values so writer-side validation matches independent readers.
    return pack.degree_quant_steps(degree, base_km, pattern).astype(np.float32).astype(np.float64)


def quantize_coeffs(coeffs: np.ndarray, steps: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    quantized = np.round(coeffs / steps[None, :]).astype(np.int64)
    return quantized, quantized.astype(np.float64) * steps[None, :]


def pack_qcoeffs(qcoeffs: np.ndarray) -> tuple[np.ndarray, bytes]:
    widths = np.vstack([
        pack.exact_axis_degree_widths([qcoeffs[s, axis] for s in range(qcoeffs.shape[0])])
        for axis in range(AXIS_COUNT)
    ]).astype(np.uint8)
    streams = []
    for axis in range(AXIS_COUNT):
        axis_arrays = [qcoeffs[s, axis] for s in range(qcoeffs.shape[0])]
        stream = pack.pack_degree_major(axis_arrays, widths[axis])
        roundtrip = pack.unpack_degree_major(stream, qcoeffs.shape[0], qcoeffs.shape[2] - 1, widths[axis])
        if not all(np.array_equal(a, b) for a, b in zip(axis_arrays, roundtrip)):
            raise RuntimeError(f"payload roundtrip failed for axis {axis}")
        streams.append(stream)
    return widths, b"".join(streams)


def pack_model_table(shape_x: np.ndarray | None, shape_y: np.ndarray | None, frame_coeffs: np.ndarray | None) -> bytes:
    parts: list[bytes] = []
    if shape_x is not None and shape_y is not None:
        parts.append(np.asarray(shape_x, dtype="<f8").tobytes())
        parts.append(np.asarray(shape_y, dtype="<f8").tobytes())
    if frame_coeffs is not None:
        parts.append(np.asarray(frame_coeffs, dtype="<f8").tobytes())
    return b"".join(parts)


def fit_cheb1_frame_model(module, tmids: np.ndarray, values: np.ndarray):
    if len(tmids) == 1:
        tnorm = np.zeros(1, dtype=np.float64)
        model = module.TimeModel(
            name="cheb1",
            coeff_plane_u=np.asarray([values[0, 0], 0.0], dtype=np.float64),
            coeff_plane_v=np.asarray([values[0, 1], 0.0], dtype=np.float64),
            coeff_apsis_angle=np.asarray([values[0, 2], 0.0], dtype=np.float64),
            eval_fn=lambda coeff, t: module.cheb_eval(coeff, t),
        )
    else:
        tnorm = module.normalize_time(tmids, tmids[0], tmids[-1])
        model = module.fit_cheb_model(tnorm, values, 1)
    return tnorm, model


def frame_values_from_segments(segments: list) -> tuple[np.ndarray, np.ndarray]:
    tmids = np.asarray([s.tmid for s in segments], dtype=np.float64)
    values = np.column_stack([
        np.asarray([s.plane_u_best for s in segments], dtype=np.float64),
        np.asarray([s.plane_v_best for s in segments], dtype=np.float64),
        np.unwrap(np.asarray([s.apsis_angle_best for s in segments], dtype=np.float64)),
    ])
    return tmids, values


def fit_raw_sun(provider: BaryProvider, cfg: BodyConfig, jd_start: float, jd_end: float, node_oversample: int) -> PackedBody:
    assert cfg.segment_days is not None
    degree = cfg.residual_degree
    bounds = fixed_bounds(jd_start, jd_end, cfg.segment_days)
    quant_steps = opm_quant_steps(degree, cfg.quant.base_km, cfg.quant.pattern)
    fit_nodes = (degree + 1) * node_oversample
    eval_nodes = max(32, fit_nodes)
    qcoeffs = []
    eval_jds_parts = []
    truth_parts = []
    recon_parts = []
    for a, b in bounds:
        nodes = cheb_nodes_expanded(a, b, fit_nodes, cfg.segment_domain_expansion_fraction)
        tau = normalize_time_expanded(nodes, a, b, cfg.segment_domain_expansion_fraction)
        coeffs = np.vstack([proto.cheb_fit(tau, provider.position(nodes)[:, axis], degree) for axis in range(AXIS_COUNT)])
        quantized, reconstructed_coeffs = quantize_coeffs(coeffs, quant_steps)
        qcoeffs.append(quantized)

        ej = proto.cheb_nodes(max(a, jd_start), min(b, jd_end), eval_nodes)
        etau = normalize_time_expanded(ej, a, b, cfg.segment_domain_expansion_fraction)
        recon = np.column_stack([proto.cheb_eval(reconstructed_coeffs[axis], etau) for axis in range(AXIS_COUNT)])
        eval_jds_parts.append(ej)
        truth_parts.append(provider.position(ej))
        recon_parts.append(recon)
    qarr = np.stack(qcoeffs, axis=0)
    widths, payload = pack_qcoeffs(qarr)
    p50, p95, p99, max_err = summarize_errors(np.vstack(truth_parts), np.vstack(recon_parts))
    boundaries = np.asarray([bounds[0][0]] + [b for _, b in bounds], dtype=np.float64)
    return PackedBody(cfg, boundaries, quant_steps, widths, qarr, payload, b"", b"", p50, p95, p99, max_err)


def fit_fixed_frame_body(provider: BaryProvider, cfg: BodyConfig, jd_start: float, jd_end: float, node_oversample: int) -> PackedBody:
    assert cfg.segment_days is not None and cfg.shape_degree is not None
    bounds = fixed_bounds(jd_start, jd_end, cfg.segment_days)
    max_degree = max(cfg.shape_degree, cfg.residual_degree)
    fit_nodes = (max_degree + 1) * node_oversample

    tmids = []
    best_params = []
    for a, b in bounds:
        nodes = cheb_nodes_expanded(a, b, fit_nodes, cfg.segment_domain_expansion_fraction)
        best_params.append(proto.fit_best_frame_params(provider.position(nodes)))
        tmids.append(0.5 * (a + b))

    tmids_arr = np.asarray(tmids, dtype=np.float64)
    values = np.column_stack([
        np.asarray([frame_values[0] for frame_values in best_params]),
        np.asarray([frame_values[1] for frame_values in best_params]),
        np.unwrap(np.asarray([frame_values[2] for frame_values in best_params])),
    ])
    tnorm, frame_model = fit_cheb1_frame_model(proto, tmids_arr, values)
    params = proto.eval_model(frame_model, tnorm)

    aligned_coeffs = np.zeros((len(bounds), AXIS_COUNT, cfg.shape_degree + 1), dtype=np.float64)
    for si, (a, b) in enumerate(bounds):
        nodes = cheb_nodes_expanded(a, b, fit_nodes, cfg.segment_domain_expansion_fraction)
        tau = normalize_time_expanded(nodes, a, b, cfg.segment_domain_expansion_fraction)
        plane_u, plane_v, apsis_angle = params[si]
        aligned = proto.align_positions(provider.position(nodes), float(plane_u), float(plane_v), float(apsis_angle))
        for axis in range(AXIS_COUNT):
            aligned_coeffs[si, axis] = proto.cheb_fit(tau, aligned[:, axis], cfg.shape_degree)
    shape_x = np.mean(aligned_coeffs[:, 0, :], axis=0)
    shape_y = np.mean(aligned_coeffs[:, 1, :], axis=0)

    quant_steps = opm_quant_steps(cfg.residual_degree, cfg.quant.base_km, cfg.quant.pattern)
    qcoeffs = []
    eval_jds_parts = []
    truth_parts = []
    recon_parts = []
    eval_nodes = max(32, (cfg.residual_degree + 1) * node_oversample)
    for si, (a, b) in enumerate(bounds):
        nodes = cheb_nodes_expanded(a, b, fit_nodes, cfg.segment_domain_expansion_fraction)
        tau = normalize_time_expanded(nodes, a, b, cfg.segment_domain_expansion_fraction)
        plane_u, plane_v, apsis_angle = params[si]
        pos = provider.position(nodes)
        aligned_truth = proto.align_positions(pos, float(plane_u), float(plane_v), float(apsis_angle))
        c = np.vstack([
            proto.cheb_fit(tau, aligned_truth[:, 0] - proto.cheb_eval(shape_x, tau), cfg.residual_degree),
            proto.cheb_fit(tau, aligned_truth[:, 1] - proto.cheb_eval(shape_y, tau), cfg.residual_degree),
            proto.cheb_fit(tau, aligned_truth[:, 2], cfg.residual_degree),
        ])
        quantized, reconstructed_coeffs = quantize_coeffs(c, quant_steps)
        qcoeffs.append(quantized)

        ej = proto.cheb_nodes(max(a, jd_start), min(b, jd_end), eval_nodes)
        etau = normalize_time_expanded(ej, a, b, cfg.segment_domain_expansion_fraction)
        aligned = np.column_stack([
            proto.cheb_eval(shape_x, etau) + proto.cheb_eval(reconstructed_coeffs[0], etau),
            proto.cheb_eval(shape_y, etau) + proto.cheb_eval(reconstructed_coeffs[1], etau),
            proto.cheb_eval(reconstructed_coeffs[2], etau),
        ])
        recon = proto.unalign_positions(aligned, float(plane_u), float(plane_v), float(apsis_angle))
        eval_jds_parts.append(ej)
        truth_parts.append(provider.position(ej))
        recon_parts.append(recon)

    qarr = np.stack(qcoeffs, axis=0)
    widths, payload = pack_qcoeffs(qarr)
    frame_coeffs = np.vstack([frame_model.coeff_plane_u, frame_model.coeff_plane_v, frame_model.coeff_apsis_angle])
    model_table = pack_model_table(shape_x, shape_y, frame_coeffs)
    p50, p95, p99, max_err = summarize_errors(np.vstack(truth_parts), np.vstack(recon_parts))
    boundaries = np.asarray([bounds[0][0]] + [b for _, b in bounds], dtype=np.float64)
    return PackedBody(cfg, boundaries, quant_steps, widths, qarr, payload, model_table, b"", p50, p95, p99, max_err)


def fit_helio_mean_apsis_body(
    cfg: BodyConfig,
    jd_start: float,
    jd_end: float,
    node_oversample: int,
    clock: MercuryClock | None = None,
) -> PackedBody:
    assert cfg.shape_degree is not None
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
        fa, fb = expanded_bounds(a, b, cfg.segment_domain_expansion_fraction)
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

    tmids_arr, values = frame_values_from_segments(segments)
    tnorm, model = fit_cheb1_frame_model(proto, tmids_arr, values)
    params = proto.eval_model(model, tnorm)
    coeffs = np.zeros((len(segments), AXIS_COUNT, cfg.shape_degree + 1), dtype=np.float64)
    for si, seg in enumerate(segments):
        plane_u, plane_v, apsis_angle = params[si]
        tau = normalize_time_expanded(seg.nodes, seg.jd0, seg.jd1, cfg.segment_domain_expansion_fraction)
        aligned = proto.align_positions(seg.pos, float(plane_u), float(plane_v), float(apsis_angle))
        for axis in range(AXIS_COUNT):
            coeffs[si, axis] = proto.cheb_fit(tau, aligned[:, axis], cfg.shape_degree)

    shape = proto.mean_shape_from_coeffs(coeffs[:, :2, :], "mean")
    shape_x = shape[0]
    shape_y = shape[1]
    quant_steps = opm_quant_steps(cfg.residual_degree, cfg.quant.base_km, cfg.quant.pattern)
    qcoeffs = []
    eval_jds_parts = []
    recon_parts = []
    eval_nodes = max(32, (cfg.residual_degree + 1) * node_oversample)
    for si, seg in enumerate(segments):
        plane_u, plane_v, apsis_angle = params[si]
        tau = normalize_time_expanded(seg.nodes, seg.jd0, seg.jd1, cfg.segment_domain_expansion_fraction)
        aligned_truth = proto.align_positions(seg.pos, float(plane_u), float(plane_v), float(apsis_angle))
        c = np.vstack([
            proto.cheb_fit(tau, aligned_truth[:, 0] - proto.cheb_eval(shape_x, tau), cfg.residual_degree),
            proto.cheb_fit(tau, aligned_truth[:, 1] - proto.cheb_eval(shape_y, tau), cfg.residual_degree),
            proto.cheb_fit(tau, aligned_truth[:, 2], cfg.residual_degree),
        ])
        quantized, reconstructed_coeffs = quantize_coeffs(c, quant_steps)
        qcoeffs.append(quantized)

        a = max(seg.jd0, jd_start)
        b = min(seg.jd1, jd_end)
        ej = old_cheb_nodes(a, b, eval_nodes)
        etau = normalize_time_expanded(ej, seg.jd0, seg.jd1, cfg.segment_domain_expansion_fraction)
        aligned = np.column_stack([
            proto.cheb_eval(shape_x, etau) + proto.cheb_eval(reconstructed_coeffs[0], etau),
            proto.cheb_eval(shape_y, etau) + proto.cheb_eval(reconstructed_coeffs[1], etau),
            proto.cheb_eval(reconstructed_coeffs[2], etau),
        ])
        recon_parts.append(proto.unalign_positions(aligned, float(plane_u), float(plane_v), float(apsis_angle)))
        eval_jds_parts.append(ej)

    eval_jds = np.concatenate(eval_jds_parts)
    provider = proto.HelioProvider(SPK_TARGET_IDS[cfg.body])
    try:
        truth = provider.position(eval_jds)
    finally:
        provider.close()

    qarr = np.stack(qcoeffs, axis=0)
    widths, payload = pack_qcoeffs(qarr)
    frame_coeffs = np.vstack([model.coeff_plane_u, model.coeff_plane_v, model.coeff_apsis_angle])
    model_table = pack_model_table(shape_x, shape_y, frame_coeffs)
    p50, p95, p99, max_err = summarize_errors(truth, np.vstack(recon_parts))
    boundaries = np.asarray([segments[0].jd0] + [s.jd1 for s in segments], dtype=np.float64)
    clock_table = b""
    if clock is not None:
        file_event_index_start = int(round((boundaries[0] - clock.phase_start_jd) / clock.period_days))
        clock_table = clock.to_clock_table(file_event_index_start)
    return PackedBody(cfg, boundaries, quant_steps, widths, qarr, payload, model_table, clock_table, p50, p95, p99, max_err)

@dataclass(frozen=True)
class MoonCenturyClock:
    domain_start_jd: float
    century_days: float
    quantum_seconds: float
    period_days: float
    phase_start_jd: float
    values: np.ndarray

    @property
    def quantum_days(self) -> float:
        return self.quantum_seconds / 86400.0

    def eval(self, jd: np.ndarray | float) -> np.ndarray:
        x = (np.asarray(jd, dtype=np.float64) - self.domain_start_jd) / self.century_days
        k = np.floor(x).astype(np.int64)
        u = x - k
        k = np.clip(k, 0, len(self.values) - 2)
        u = np.where((x < 0) | (x > len(self.values) - 1), 0.0, u)
        vals = self.values.astype(np.float64) * self.quantum_days
        return vals[k] + (vals[k + 1] - vals[k]) * u

    def boundary(self, indices: np.ndarray | int | float) -> np.ndarray:
        base = self.phase_start_jd + self.period_days * np.asarray(indices, dtype=np.float64)
        return base + self.eval(base)

    def bounds(self, jd_start: float, jd_end: float, min_len: float = 5.0) -> list[tuple[float, float]]:
        i0 = int(math.floor((jd_start - self.phase_start_jd) / self.period_days)) - 8
        i1 = int(math.ceil((jd_end - self.phase_start_jd) / self.period_days)) + 8
        inds = np.arange(i0, i1 + 1, dtype=np.int64)
        b = self.boundary(inds)
        out: list[tuple[float, float]] = []
        for a, c in zip(b[:-1], b[1:]):
            if c <= jd_start or a >= jd_end:
                continue
            if c > a + min_len:
                out.append((float(a), float(c)))
        return out

    def to_clock_table(self) -> bytes:
        return MOON_CLOCK_TABLE_STRUCT.pack(
            float(self.domain_start_jd),
            float(self.century_days),
            float(self.quantum_seconds),
            int(len(self.values)),
            MOON_CLOCK_INTERPOLATION_LINEAR,
            MOON_CLOCK_STATISTIC_MEDIAN,
            0,
        ) + np.asarray(self.values, dtype="<i2").tobytes()


_MOON_CLOCK_CACHE: MoonCenturyClock | None = None


def moon_century_clock() -> MoonCenturyClock:
    global _MOON_CLOCK_CACHE
    if _MOON_CLOCK_CACHE is None:
        if MOON_CLOCK_CONSTANTS_PATH.exists():
            data = json.loads(MOON_CLOCK_CONSTANTS_PATH.read_text(encoding="utf-8"))
            values = np.asarray(data["table_values"], dtype=np.int16)
            _MOON_CLOCK_CACHE = MoonCenturyClock(
                float(data["domain_start_jd"]),
                float(data["century_days"]),
                float(data["quantum_seconds"]),
                float(data["period_days"]),
                float(data["phase_start_jd"]),
                values,
            )
        else:
            raise FileNotFoundError(f"missing persisted Moon clock table: {MOON_CLOCK_CONSTANTS_PATH}")
        if len(_MOON_CLOCK_CACHE.values) != int(MOON_CENTURY_TABLE["count"]):
            raise RuntimeError("bad Moon century i16 table count")
    return _MOON_CLOCK_CACHE


def select_full_segments(segments: list, jd_start: float, jd_end: float, body: str) -> list:
    selected = [s for s in segments if s.jd1 > jd_start and s.jd0 < jd_end]
    if not selected:
        raise RuntimeError(f"no {body} segments overlap requested coverage")
    if selected[0].jd0 > jd_start or selected[-1].jd1 < jd_end:
        raise RuntimeError(f"{body} segments cover {selected[0].jd0}..{selected[-1].jd1}, wanted {jd_start}..{jd_end}")
    for prev, cur in zip(selected, selected[1:]):
        if abs(prev.jd1 - cur.jd0) > 1e-8:
            raise RuntimeError(f"non-contiguous {body} segments: {prev.jd1} -> {cur.jd0}")
    return selected


def fit_geo_mean_perigee_moon(cfg: BodyConfig, jd_start: float, jd_end: float, node_oversample: int, clock: MoonCenturyClock) -> PackedBody:
    assert cfg.shape_degree is not None
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
        fa, fb = expanded_bounds(a, b, cfg.segment_domain_expansion_fraction)
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

    segments = select_full_segments(segments_all, jd_start, jd_end, cfg.body)
    tmids_arr, values = frame_values_from_segments(segments)
    tnorm, model = fit_cheb1_frame_model(moon_proto, tmids_arr, values)
    params = moon_proto.eval_model(model, tnorm)

    coeffs = np.zeros((len(segments), AXIS_COUNT, cfg.shape_degree + 1), dtype=np.float64)
    for si, seg in enumerate(segments):
        plane_u, plane_v, apsis_angle = params[si]
        tau = normalize_time_expanded(seg.nodes, seg.jd0, seg.jd1, cfg.segment_domain_expansion_fraction)
        aligned = moon_proto.align_positions(seg.pos, float(plane_u), float(plane_v), float(apsis_angle))
        for axis in range(AXIS_COUNT):
            coeffs[si, axis] = moon_proto.cheb_fit(tau, aligned[:, axis], cfg.shape_degree)
    shape_x = np.mean(coeffs[:, 0, :], axis=0)
    shape_y = np.mean(coeffs[:, 1, :], axis=0)

    quant_steps = opm_quant_steps(cfg.residual_degree, cfg.quant.base_km, cfg.quant.pattern)
    qcoeffs = []
    eval_jds_parts = []
    truth_parts = []
    recon_parts = []
    eval_nodes = max(32, (cfg.residual_degree + 1) * node_oversample)
    for si, seg in enumerate(segments):
        plane_u, plane_v, apsis_angle = params[si]
        tau = normalize_time_expanded(seg.nodes, seg.jd0, seg.jd1, cfg.segment_domain_expansion_fraction)
        aligned_truth = moon_proto.align_positions(seg.pos, float(plane_u), float(plane_v), float(apsis_angle))
        c = np.vstack([
            moon_proto.cheb_fit(tau, aligned_truth[:, 0] - moon_proto.cheb_eval(shape_x, tau), cfg.residual_degree),
            moon_proto.cheb_fit(tau, aligned_truth[:, 1] - moon_proto.cheb_eval(shape_y, tau), cfg.residual_degree),
            moon_proto.cheb_fit(tau, aligned_truth[:, 2], cfg.residual_degree),
        ])
        quantized, reconstructed_coeffs = quantize_coeffs(c, quant_steps)
        qcoeffs.append(quantized)

        a = max(seg.jd0, jd_start)
        b = min(seg.jd1, jd_end)
        ej = old_cheb_nodes(a, b, eval_nodes)
        etau = normalize_time_expanded(ej, seg.jd0, seg.jd1, cfg.segment_domain_expansion_fraction)
        aligned = np.column_stack([
            moon_proto.cheb_eval(shape_x, etau) + moon_proto.cheb_eval(reconstructed_coeffs[0], etau),
            moon_proto.cheb_eval(shape_y, etau) + moon_proto.cheb_eval(reconstructed_coeffs[1], etau),
            moon_proto.cheb_eval(reconstructed_coeffs[2], etau),
        ])
        recon_parts.append(moon_proto.unalign_positions(aligned, float(plane_u), float(plane_v), float(apsis_angle)))
        eval_jds_parts.append(ej)
        
    provider = moon_proto.GeoMoonProvider()
    eval_jds = np.concatenate(eval_jds_parts)
    try:
        truth = provider.position(eval_jds)
    finally:
        provider.close()

    qarr = np.stack(qcoeffs, axis=0)
    widths, payload = pack_qcoeffs(qarr)
    frame_coeffs = np.vstack([model.coeff_plane_u, model.coeff_plane_v, model.coeff_apsis_angle])
    model_table = pack_model_table(shape_x, shape_y, frame_coeffs)
    p50, p95, p99, max_err = summarize_errors(truth, np.vstack(recon_parts))
    boundaries = np.asarray([segments[0].jd0] + [s.jd1 for s in segments], dtype=np.float64)
    return PackedBody(cfg, boundaries, quant_steps, widths, qarr, payload, model_table, clock.to_clock_table(), p50, p95, p99, max_err)


def storage_vector_id(cfg: BodyConfig) -> int:
    if cfg.center == "ssb":
        return STORAGE_SSB_TO_BODY
    if cfg.center == "sun":
        return STORAGE_SUN_TO_BODY
    if cfg.center == "earth" and cfg.body == "moon":
        return STORAGE_EARTH_TO_MOON
    return STORAGE_CENTER_TO_BODY


def center_id(cfg: BodyConfig) -> int:
    return OPM_BODY_IDS[cfg.center]


def frame_kind(cfg: BodyConfig) -> int:
    return FRAME_NONE if cfg.method == "raw_xyz_cheb" else FRAME_CHEB1_PLANE_APSIS


def reference_shape_kind(cfg: BodyConfig) -> int:
    return REFERENCE_SHAPE_NONE if cfg.shape_degree is None else REFERENCE_SHAPE_MEAN_XY_CHEB


def segment_addressing_kind(cfg: BodyConfig) -> int:
    if cfg.clock.kind in {"global_anomalistic", "global_anomalistic_cheb8", "global_anomalistic_century_i16", "global_inertial_phase"}:
        return SEGMENT_AFFINE_PERIOD_PHASE
    return SEGMENT_FIXED_DAYS


def segment_flags(cfg: BodyConfig) -> int:
    # Current recommended OPM bodies are all formula-addressed. If a future file
    # truly needs explicit boundaries, set HAS_EXPLICIT_BOUNDARY_BLOCK and place
    # the boundary block after the fixed header/descriptor section.
    return 0


def model_policy_payload(cfg: BodyConfig) -> bytes:
    if cfg.method == "raw_xyz_cheb":
        event_kind = EVENT_NONE
        event_source = EVENT_SOURCE_NONE
        boundary_policy = BOUNDARY_NONE_OR_FIXED_TIME
        detection = DETECTION_NONE
        frame_policy = FRAME_POLICY_NONE
        apsis_policy = APSIS_POLICY_NONE
        shape_policy = SHAPE_POLICY_NONE
        residual_policy = RESIDUAL_POLICY_NONE
    elif cfg.method == "mean_apsis_frame_shape":
        event_kind = EVENT_PERIHELION
        event_source = EVENT_SOURCE_SUN_TO_BODY
        boundary_policy = BOUNDARY_GLOBAL_MEAN_EVENT_FIT
        detection = DETECTION_RADIUS_MIN_GRID_ARGRELEXTREMA
        frame_policy = FRAME_POLICY_CHEB1_BEST_PLANE_APSIS
        apsis_policy = APSIS_POLICY_MIN_RADIUS_DIRECTION
        shape_policy = SHAPE_POLICY_MEAN_XY
        residual_policy = RESIDUAL_POLICY_DIRECT_FIT_FROZEN_SHAPE
    elif cfg.method == "mean_lunar_apsis_frame_shape":
        event_kind = EVENT_PERIGEE
        event_source = EVENT_SOURCE_EARTH_TO_MOON
        boundary_policy = BOUNDARY_GLOBAL_MEAN_PLUS_CENTURY_I16_TABLE
        detection = DETECTION_RADIUS_MIN_GRID_ARGRELEXTREMA
        frame_policy = FRAME_POLICY_CHEB1_BEST_PLANE_APSIS
        apsis_policy = APSIS_POLICY_MIN_RADIUS_DIRECTION
        shape_policy = SHAPE_POLICY_MEAN_XY
        residual_policy = RESIDUAL_POLICY_DIRECT_FIT_FROZEN_SHAPE
    elif cfg.method == "fixed_frame_shape":
        if cfg.clock.kind == "global_inertial_phase":
            event_kind = EVENT_INERTIAL_PHASE_CYCLE
            event_source = EVENT_SOURCE_SUN_TO_BODY
            boundary_policy = BOUNDARY_GLOBAL_INERTIAL_PHASE_FIT
            detection = DETECTION_UNWRAPPED_INERTIAL_PHASE_CROSSING
        elif cfg.clock.kind == "global_anomalistic":
            event_kind = EVENT_PERIHELION
            event_source = EVENT_SOURCE_SUN_TO_BODY
            boundary_policy = BOUNDARY_GLOBAL_MEAN_EVENT_FIT
            detection = DETECTION_RADIUS_MIN_GRID_ARGRELEXTREMA
        else:
            event_kind = EVENT_FIXED_TIME_BOUNDARY
            event_source = EVENT_SOURCE_NONE
            boundary_policy = BOUNDARY_NONE_OR_FIXED_TIME
            detection = DETECTION_NONE
        frame_policy = FRAME_POLICY_CHEB1_BEST_PLANE_APSIS
        apsis_policy = APSIS_POLICY_MIN_RADIUS_DIRECTION
        shape_policy = SHAPE_POLICY_MEAN_XY
        residual_policy = RESIDUAL_POLICY_DIRECT_FIT_FROZEN_SHAPE
    else:
        raise ValueError(f"unsupported model policy for {cfg.method}")

    if cfg.clock.kind == "global_anomalistic_cheb8":
        correction_policy = EVENT_CORRECTION_CHEB_EVENT_TIME
        boundary_policy = BOUNDARY_GLOBAL_MEAN_PLUS_CHEB_CORRECTION
    elif cfg.clock.kind == "global_anomalistic_century_i16":
        correction_policy = EVENT_CORRECTION_CENTURY_I16_LINEAR
        boundary_policy = BOUNDARY_GLOBAL_MEAN_PLUS_CENTURY_I16_TABLE
    else:
        correction_policy = EVENT_CORRECTION_NONE

    return MODEL_POLICY_STRUCT.pack(
        event_kind,
        event_source,
        boundary_policy,
        detection,
        correction_policy,
        frame_policy,
        apsis_policy,
        shape_policy,
        residual_policy,
        QUANT_POLICY_EXPLICIT_STEPS,
        WIDTH_POLICY_EXPLICIT_AXIS_DEGREE,
        PAYLOAD_ORDER_AXIS_DEGREE_SEGMENT,
        0,
        0,
    )


def build_descriptor_section(cfg: BodyConfig, base_offset: int) -> tuple[bytes, int, int]:
    payload = model_policy_payload(cfg)
    table_size = DESCRIPTOR_ENTRY_STRUCT.size
    payload_offset = base_offset + table_size
    entry = DESCRIPTOR_ENTRY_STRUCT.pack(DESCRIPTOR_MODEL_POLICY_V1, 0, len(payload), payload_offset)
    return entry + payload, 1, base_offset


def write_opm_file(path: Path, packed: PackedBody, source_start: float, source_end: float, jd_start: float, days: float) -> int:
    cfg = packed.cfg
    segment_table = b""
    quant_table = np.asarray(packed.quant_steps, dtype="<f4").tobytes()
    width_table = np.asarray(packed.widths, dtype=np.uint8).tobytes()

    cursor = FIXED_HEADER_SIZE
    descriptor_section, descriptor_count, descriptor_table_offset = build_descriptor_section(cfg, cursor)
    cursor += len(descriptor_section)
    segment_table_offset = 0
    segment_table_size = 0
    clock_table_offset = cursor if packed.clock_table else 0
    clock_table_size = len(packed.clock_table)
    cursor += clock_table_size
    quant_table_offset = cursor
    quant_table_size = len(quant_table)
    cursor += quant_table_size
    width_table_offset = cursor
    width_table_size = len(width_table)
    cursor += width_table_size
    model_table_offset = cursor if packed.model_table else 0
    model_table_size = len(packed.model_table)
    cursor += model_table_size
    payload_offset = cursor
    payload_size = len(packed.payload)
    cursor += payload_size
    file_size = cursor

    shape_degree = 255 if cfg.shape_degree is None else int(cfg.shape_degree)
    if cfg.clock.kind == "global_anomalistic_cheb8" and cfg.clock.period_days is not None and cfg.clock.phase_start_jd is not None:
        segment_days = float(cfg.clock.period_days)
        period_days = float(cfg.clock.period_days)
        phase_start_jd = float(cfg.clock.phase_start_jd)
    elif cfg.clock.kind == "global_anomalistic_century_i16" and cfg.clock.period_days is not None and cfg.clock.phase_start_jd is not None:
        segment_days = float(cfg.clock.period_days)
        period_days = float(cfg.clock.period_days)
        file_event_index_start = int(round((packed.boundaries[0] - float(cfg.clock.phase_start_jd)) / period_days))
        phase_start_jd = float(cfg.clock.phase_start_jd) + file_event_index_start * period_days
    else:
        actual_period_days = float((packed.boundaries[-1] - packed.boundaries[0]) / packed.segment_count)
        segment_days = actual_period_days
        period_days = actual_period_days
        phase_start_jd = float(packed.boundaries[0])
    event_step = float(cfg.apsis_step_days if cfg.apsis_step_days else 0.0)
    if cfg.clock.kind == "global_anomalistic_cheb8":
        clock_correction_kind = CLOCK_CORRECTION_CHEB_EVENT_TIME_F64
        clock_correction_degree = MERCURY_CLOCK_DEGREE
    elif cfg.clock.kind == "global_anomalistic_century_i16":
        clock_correction_kind = CLOCK_CORRECTION_CENTURY_I16_LINEAR_TABLE
        clock_correction_degree = 0
    else:
        clock_correction_kind = CLOCK_CORRECTION_NONE
        clock_correction_degree = 0
    header = HEADER_STRUCT.pack(
        MAGIC,
        ENDIAN_TAG,
        HEADER_MINOR_VERSION,
        0,
        FIXED_HEADER_SIZE,
        payload_offset,
        FLAGS,
        b"\0" * 3,
        SOURCE_DE441,
        0,
        float(source_start),
        float(source_end),
        float(jd_start),
        float(days),
        OPM_BODY_IDS[cfg.body],
        center_id(cfg),
        storage_vector_id(cfg),
        POSITION_KM,
        TIME_TDB_JD,
        MODEL_KIND[cfg.method],
        CLOCK_KIND[cfg.clock.kind],
        frame_kind(cfg),
        reference_shape_kind(cfg),
        RESIDUAL_XYZ_DEGREE_MAJOR_EXACT_WIDTH,
        segment_addressing_kind(cfg),
        AXIS_COUNT,
        packed.segment_count,
        segment_days,
        period_days,
        phase_start_jd,
        float(cfg.edge_margin_days),
        event_step,
        float(cfg.segment_domain_expansion_fraction),
        segment_flags(cfg),
        b"\0" * 3,
        int(cfg.residual_degree),
        shape_degree,
        int(cfg.residual_degree + 1),
        int(AXIS_COUNT * (cfg.residual_degree + 1)),
        clock_correction_kind,
        clock_correction_degree,
        descriptor_count,
        b"\0" * 7,
        descriptor_table_offset,
        segment_table_offset,
        segment_table_size,
        clock_table_offset,
        clock_table_size,
        quant_table_offset,
        quant_table_size,
        width_table_offset,
        width_table_size,
        model_table_offset,
        model_table_size,
        payload_offset,
        payload_size,
        file_size,
        0,
        0,
        b"\0" * 58,
    )
    data = bytearray(header + descriptor_section + segment_table + packed.clock_table + quant_table + width_table + packed.model_table + packed.payload)
    if len(data) != file_size:
        raise RuntimeError(f"file size mismatch: header says {file_size}, got {len(data)}")
    payload_crc = crc64_ecma(bytes(data[FIXED_HEADER_SIZE:]))
    struct.pack_into("<Q", data, PAYLOAD_CRC64_OFFSET, payload_crc)
    header_for_crc = bytearray(data[:FIXED_HEADER_SIZE])
    struct.pack_into("<Q", header_for_crc, HEADER_CRC64_OFFSET, 0)
    header_crc = crc64_ecma(bytes(header_for_crc))
    struct.pack_into("<Q", data, HEADER_CRC64_OFFSET, header_crc)
    data_bytes = bytes(data)
    sanity_check_bytes(data_bytes, packed)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data_bytes)
    return file_size


def sanity_check_bytes(data: bytes, packed: PackedBody) -> None:
    fields = HEADER_STRUCT.unpack(data[:FIXED_HEADER_SIZE])
    if fields[0] != MAGIC:
        raise RuntimeError("bad magic after pack")
    if fields[1] != ENDIAN_TAG:
        raise RuntimeError("bad endian tag after pack")
    file_size = fields[56]
    if file_size != len(data):
        raise RuntimeError("bad file_size after pack")
    header_crc = fields[57]
    payload_crc = fields[58]
    if header_crc == 0 or payload_crc == 0:
        raise RuntimeError("CRC64 fields were not written")
    boundaries = packed.boundaries
    if np.any(np.diff(boundaries) <= 0.0):
        raise RuntimeError("segment boundaries are not strictly increasing")
    if packed.widths.shape != (AXIS_COUNT, packed.cfg.residual_degree + 1):
        raise RuntimeError("width table shape mismatch")
    descriptor_count = fields[41]
    descriptor_offset = fields[43]
    if descriptor_count != 1:
        raise RuntimeError("missing model_policy_v1 descriptor")
    kind, _flags, size, payload_offset = DESCRIPTOR_ENTRY_STRUCT.unpack(
        data[descriptor_offset : descriptor_offset + DESCRIPTOR_ENTRY_STRUCT.size]
    )
    if kind != DESCRIPTOR_MODEL_POLICY_V1 or size != MODEL_POLICY_STRUCT.size:
        raise RuntimeError("bad model_policy_v1 descriptor entry")
    if payload_offset + size > len(data):
        raise RuntimeError("model_policy_v1 descriptor points outside file")
    if packed.clock_table:
        if packed.cfg.clock.kind == "global_anomalistic_cheb8":
            if fields[39] != CLOCK_CORRECTION_CHEB_EVENT_TIME_F64 or fields[40] != MERCURY_CLOCK_DEGREE:
                raise RuntimeError("bad Mercury clock correction header fields")
        elif packed.cfg.clock.kind == "global_anomalistic_century_i16":
            if fields[39] != CLOCK_CORRECTION_CENTURY_I16_LINEAR_TABLE:
                raise RuntimeError("bad Moon clock correction header fields")
        else:
            raise RuntimeError("unexpected clock table for body without clock correction")
        clock_offset = fields[46]
        clock_size = fields[47]
        if clock_offset <= 0 or clock_offset + clock_size > len(data):
            raise RuntimeError("clock table points outside file")


def build_body(spk: SPK, body: str, jd_start: float, days: float, node_oversample: int) -> PackedBody:
    cfg = body_config_for_generation(body)
    jd_end = jd_start + days
    if cfg.method == "raw_xyz_cheb":
        return fit_raw_sun(BaryProvider(spk, SPK_TARGET_IDS[body]), cfg, jd_start, jd_end, node_oversample)
    if cfg.method == "fixed_frame_shape":
        return fit_fixed_frame_body(BaryProvider(spk, SPK_TARGET_IDS[body]), cfg, jd_start, jd_end, node_oversample)
    if cfg.method == "mean_apsis_frame_shape":
        clock = mercury_clock() if body == "mercury" else None
        return fit_helio_mean_apsis_body(cfg, jd_start, jd_end, node_oversample, clock)
    if cfg.method == "mean_lunar_apsis_frame_shape":
        return fit_geo_mean_perigee_moon(cfg, jd_start, jd_end, node_oversample, moon_century_clock())
    raise NotImplementedError(f"unsupported first-pass OPM body/method: {body} {cfg.method}")


def first_pass_bodies() -> list[str]:
    return list(DEFAULT_BODY_ORDER)


NATIVE_POLISH_BODIES = {"mercury", "venus", "moon"}
SSB_SUN_ANCHOR_BODIES = {"emb", "mars", "jupiter", "saturn", "uranus", "neptune", "pluto"}
REPO_ROOT = SCRIPT_DIR.parent


def run_pipeline_command(cmd: list[str], *, log_path: Path | None = None) -> str:
    proc = subprocess.run(cmd, cwd=REPO_ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text(proc.stdout, encoding="utf-8")
    if proc.returncode != 0:
        detail = f"; see {log_path}" if log_path is not None else f"\n{proc.stdout}"
        raise RuntimeError(f"command failed ({proc.returncode}): {' '.join(cmd)}{detail}")
    return proc.stdout


POLISH_NODES_PER_SEGMENT = 32
POLISH_P99_SLACK_ABS = 1e-6


def polish_one_body(
    *,
    body: str,
    input_opm: Path,
    output_opm: Path,
    de441: Path,
    sun_opm: Path | None,
    jobs: int,
    nodes_per_segment: int,
    p99_slack_abs: float,
    log_dir: Path,
) -> None:
    output_opm.parent.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{body}.log"
    if body == "sun":
        cmd = [
            sys.executable,
            "tools/optimize_opm_global_tail.py",
            str(input_opm),
            "--de441",
            str(de441),
            "--nodes-per-segment",
            str(nodes_per_segment),
            "--jobs",
            str(jobs),
            "--limit",
            "0",
            "--progress-every",
            "0",
            "--error-metric",
            "km",
            "--output",
            str(output_opm),
        ]
    elif body in NATIVE_POLISH_BODIES:
        cmd = [
            sys.executable,
            "tools/optimize_opm_native_guarded_pmax.py",
            str(input_opm),
            "--de441",
            str(de441),
            "--nodes-per-segment",
            str(nodes_per_segment),
            "--jobs",
            str(jobs),
            "--limit",
            "0",
            "--progress-every",
            "0",
            "--output",
            str(output_opm),
        ]
    elif body in SSB_SUN_ANCHOR_BODIES:
        if sun_opm is None:
            raise RuntimeError(f"{body} polish needs a polished Sun anchor; use --all --polish")
        cmd = [
            sys.executable,
            "tools/optimize_opm_ssb_sun_anchor_pmax.py",
            str(input_opm),
            "--sun-opm",
            str(sun_opm),
            "--de441",
            str(de441),
            "--nodes-per-segment",
            str(nodes_per_segment),
            "--jobs",
            str(jobs),
            "--limit",
            "0",
            "--progress-every",
            "0",
            "--p99-slack-abs",
            str(p99_slack_abs),
            "--output",
            str(output_opm),
        ]
    else:
        raise RuntimeError(f"unsupported polish body: {body}")
    print(f"polishing {body}...", flush=True)
    run_pipeline_command(cmd, log_path=log_path)


def generated_path(body: str, *, output: Path | None, output_root: Path, single_body: bool) -> Path:
    if single_body:
        if output is None:
            raise SystemExit("--output is required with --body")
        return output
    return output_root / f"{body}.opm"


def raw_output_location(output: Path | None, output_root: Path, raw_root: Path | None, *, single_body: bool) -> tuple[Path | None, Path]:
    if raw_root is not None:
        root = raw_root
    elif single_body:
        if output is None:
            raise SystemExit("--output is required with --body")
        root = output.parent / ".raw"
    else:
        root = output_root / ".raw"
    raw_output = root / output.name if single_body and output is not None else None
    return raw_output, root


def validate_outputs(*, de441: Path, target: Path, nodes_per_segment: int, log_path: Path | None = None) -> None:
    print(f"validating {target}...", flush=True)
    run_pipeline_command(
        [
            sys.executable,
            "validate_opm.py",
            "--de441",
            str(de441),
            "--nodes-per-segment",
            str(nodes_per_segment),
            str(target),
        ],
        log_path=log_path,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write OPM1 coverage files")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--body", choices=DEFAULT_BODY_ORDER)
    group.add_argument("--all", action="store_true", help="write all configured OPM bodies")
    parser.add_argument("--jd-start", type=float, default=JD_J2000)
    parser.add_argument("--days", type=float, default=CENTURY_DAYS)
    parser.add_argument("--output", type=Path, help="output .opm path for --body")
    parser.add_argument("--output-root", type=Path, default=Path("out/small/j2000-opm"), help="output directory for --all")
    parser.add_argument("--validate", action="store_true", help="validate written OPM files against DE441 after generation/polish")
    parser.add_argument("--polish", action="store_true", help="run the recommended no-size-increase polish pipeline after raw generation")
    parser.add_argument("--jobs", type=int, default=4, help="worker processes for polish; ignored unless --polish is set")
    parser.add_argument("--de441", type=Path, required=True, help="path to DE441 BSP file")
    return parser.parse_args()


def generate_range(
    *,
    de441: Path,
    jd_start: float,
    days: float,
    bodies: list[str],
    output: Path | None,
    output_root: Path,
    node_oversample: int,
    validate: bool,
    range_safety: str = "strict",
    polish: bool = False,
    polish_jobs: int = 4,
    polish_nodes_per_segment: int = 32,
    polish_p99_slack_abs: float = 1e-6,
    raw_root: Path | None = None,
    sun_opm: Path | None = None,
) -> int:
    proto.set_de441_path(de441)
    moon_proto.set_de441_path(de441)
    single_body = len(bodies) == 1
    if single_body and output is None:
        raise SystemExit("--output is required with --body")
    final_output_root = Path(output_root)
    final_output = output
    raw_output = output
    raw_output_root = final_output_root
    if polish:
        raw_output, raw_output_root = raw_output_location(output, final_output_root, raw_root, single_body=single_body)
        if single_body and raw_output == final_output:
            raise SystemExit("raw intermediate root must differ from final --output when using --body --polish")
        if not single_body and raw_output_root == final_output_root:
            raise SystemExit("raw intermediate root must differ from final --output-root when using --all --polish")
        if any(body in SSB_SUN_ANCHOR_BODIES for body in bodies) and "sun" not in bodies and sun_opm is None:
            raise SystemExit("--body SSB-body --polish needs a polished Sun anchor; use --all --polish")

    with SPK.open(str(de441)) as spk:
        src_start, src_end = source_bounds(spk)
        validate_requested_range(spk, bodies, float(jd_start), float(days), range_safety=range_safety)
        for body in bodies:
            print(f"fitting {body}...", flush=True)
            packed = build_body(spk, body, float(jd_start), float(days), int(node_oversample))
            out = generated_path(body, output=raw_output, output_root=raw_output_root, single_body=single_body)
            size = write_opm_file(out, packed, src_start, src_end, float(jd_start), float(days))
            status = "PASS" if packed.max_err <= 0.001 else "MISS"
            print(
                f"  wrote {out} size={size / 1024.0:.3f} KiB segments={packed.segment_count} "
                f"p99={packed.p99:.6g}\" max={packed.max_err:.6g}\" {status}",
                flush=True,
            )

    final_target = final_output if single_body else final_output_root
    if polish:
        log_dir = (final_output.parent if single_body and final_output is not None else final_output_root) / "logs" / "polish"
        polished_sun = sun_opm
        if "sun" in bodies:
            raw_sun = generated_path("sun", output=raw_output, output_root=raw_output_root, single_body=single_body)
            out_sun = generated_path("sun", output=final_output, output_root=final_output_root, single_body=single_body)
            polish_one_body(
                body="sun",
                input_opm=raw_sun,
                output_opm=out_sun,
                de441=de441,
                sun_opm=None,
                jobs=polish_jobs,
                nodes_per_segment=polish_nodes_per_segment,
                p99_slack_abs=polish_p99_slack_abs,
                log_dir=log_dir,
            )
            polished_sun = out_sun
        for body in bodies:
            if body == "sun":
                continue
            raw_opm = generated_path(body, output=raw_output, output_root=raw_output_root, single_body=single_body)
            out_opm = generated_path(body, output=final_output, output_root=final_output_root, single_body=single_body)
            polish_one_body(
                body=body,
                input_opm=raw_opm,
                output_opm=out_opm,
                de441=de441,
                sun_opm=polished_sun,
                jobs=polish_jobs,
                nodes_per_segment=polish_nodes_per_segment,
                p99_slack_abs=polish_p99_slack_abs,
                log_dir=log_dir,
            )
    if validate:
        validate_outputs(
            de441=de441,
            target=final_target,
            nodes_per_segment=polish_nodes_per_segment if polish else 32,
            log_path=(final_output.parent if single_body and final_output is not None else final_output_root) / "logs" / "validate.log",
        )
    return 0


def main() -> int:
    args = parse_args()
    bodies = first_pass_bodies() if args.all else [args.body]
    try:
        return generate_range(
            de441=args.de441,
            jd_start=float(args.jd_start),
            days=float(args.days),
            bodies=bodies,
            output=args.output,
            output_root=args.output_root,
            node_oversample=3,
            validate=bool(args.validate),
            range_safety="strict",
            polish=bool(args.polish),
            polish_jobs=int(args.jobs),
            polish_nodes_per_segment=POLISH_NODES_PER_SEGMENT,
            polish_p99_slack_abs=POLISH_P99_SLACK_ABS,
            raw_root=None,
            sun_opm=None,
        )
    except RangeSafetyError as exc:
        raise SystemExit(f"range safety check failed: {exc}") from exc


if __name__ == "__main__":
    raise SystemExit(main())
