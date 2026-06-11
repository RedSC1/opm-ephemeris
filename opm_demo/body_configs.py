"""Recommended OPM demo body configurations.

These parameters reproduce the prototype OPM1 coverage files used by the paper
example code. They are reference-demo settings, not production product policy.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

Center = Literal["ssb", "sun", "earth"]
ClockKind = Literal[
    "raw_fixed",
    "global_anomalistic",
    "global_anomalistic_cheb8",
    "global_anomalistic_century_i16",
    "global_inertial_phase",
    "fixed_period",
]
MethodKind = Literal[
    "raw_xyz_cheb",
    "mean_apsis_frame_shape",
    "mean_lunar_apsis_frame_shape",
    "fixed_frame_shape",
]


@dataclass(frozen=True)
class QuantConfig:
    base_km: float
    pattern: str = "flat"


@dataclass(frozen=True)
class ClockConfig:
    kind: ClockKind
    period_days: float | None = None
    phase_start_jd: float | None = None
    correction: str = "none"


@dataclass(frozen=True)
class BodyConfig:
    body: str
    center: Center
    method: MethodKind
    clock: ClockConfig
    residual_degree: int
    shape_degree: int | None
    quant: QuantConfig
    segment_days: float | None = None
    edge_margin_days: float = 0.0
    apsis_step_days: float = 0.5
    segment_domain_expansion_fraction: float = 0.0
    estimated_size_kib: float | None = None
    worst_p99_arcsec: float | None = None
    worst_max_arcsec: float | None = None
    validation_scope: str = "7 epochs"
    notes: str = ""


# DE441-derived global periods/phase starts for the reference demo.
MERCURY_GLOBAL_PERIOD = 87.969349505206
MERCURY_PHASE_START = -3099979.465529
VENUS_GLOBAL_PERIOD = 224.700615924424
VENUS_PHASE_START = -3099946.588707
MARS_GLOBAL_PERIOD = 686.996026060798
MARS_PHASE_START = -3099422.621460
EMB_GLOBAL_PERIOD = 365.256362982910
EMB_PHASE_START = -3100088.351481
MOON_GLOBAL_PERIOD = 27.554538221087
MOON_PHASE_START = -3099992.935766

MOON_CENTURY_TABLE = {
    "domain_start_jd": -3100255.0,
    "century_days": 36525.0,
    "count": 306,
    "quantum_seconds": 60.0,
    "storage": "int16",
    "statistic": "median",
    "interpolation": "linear",
    "table_i16_bytes": 612,
    "charged_overhead_kib": 0.629,
}

CONFIGS: dict[str, BodyConfig] = {
    "sun": BodyConfig(
        body="sun",
        center="ssb",
        method="raw_xyz_cheb",
        clock=ClockConfig("raw_fixed"),
        segment_days=180.0,
        residual_degree=25,
        shape_degree=None,
        quant=QuantConfig(0.01, "flat"),
        estimated_size_kib=18.37,
        validation_scope="diagnostic anchor",
        notes="Shared SSB->Sun anchor; own angular error is diagnostic only.",
    ),
    "mercury": BodyConfig(
        body="mercury",
        center="sun",
        method="mean_apsis_frame_shape",
        clock=ClockConfig(
            "global_anomalistic_cheb8",
            period_days=MERCURY_GLOBAL_PERIOD,
            phase_start_jd=MERCURY_PHASE_START,
            correction="Chebyshev degree 8 event-time correction",
        ),
        residual_degree=24,
        shape_degree=40,
        quant=QuantConfig(0.032, "linear:0.65"),
        edge_margin_days=120.0,
        apsis_step_days=0.5,
        segment_domain_expansion_fraction=0.01,
        estimated_size_kib=33.593,
        worst_p99_arcsec=0.000499332,
        worst_max_arcsec=0.000753993,
        notes="Corrected global clock; f=0.01 promoted for margin.",
    ),
    "venus": BodyConfig(
        body="venus",
        center="sun",
        method="mean_apsis_frame_shape",
        clock=ClockConfig("global_anomalistic", period_days=VENUS_GLOBAL_PERIOD, phase_start_jd=VENUS_PHASE_START),
        residual_degree=24,
        shape_degree=40,
        quant=QuantConfig(0.06, "flat"),
        edge_margin_days=300.0,
        apsis_step_days=1.0,
        segment_domain_expansion_fraction=0.01,
        estimated_size_kib=16.026,
        worst_p99_arcsec=0.000422766,
        worst_max_arcsec=0.000603196,
        notes="f=0.01 promoted after 7-epoch expansion validation.",
    ),
    "emb": BodyConfig(
        body="emb",
        center="ssb",
        method="fixed_frame_shape",
        clock=ClockConfig("global_inertial_phase", period_days=EMB_GLOBAL_PERIOD, phase_start_jd=EMB_PHASE_START),
        segment_days=EMB_GLOBAL_PERIOD,
        residual_degree=28,
        shape_degree=22,
        quant=QuantConfig(0.02, "growth:1.25"),
        segment_domain_expansion_fraction=0.01,
        estimated_size_kib=14.229,
        worst_p99_arcsec=0.000437883,
        worst_max_arcsec=0.000686947,
        notes="f=0.01 selected to suppress segment-boundary velocity spikes.",
    ),
    "mars": BodyConfig(
        body="mars",
        center="ssb",
        method="fixed_frame_shape",
        clock=ClockConfig("global_anomalistic", period_days=MARS_GLOBAL_PERIOD, phase_start_jd=MARS_PHASE_START),
        segment_days=MARS_GLOBAL_PERIOD,
        residual_degree=28,
        shape_degree=22,
        quant=QuantConfig(0.04, "flat"),
        segment_domain_expansion_fraction=0.01,
        estimated_size_kib=7.555,
        worst_p99_arcsec=0.000540820,
        worst_max_arcsec=0.000859381,
        notes="f=0.01 selected to suppress segment-boundary velocity spikes.",
    ),
    "jupiter": BodyConfig(
        body="jupiter",
        center="ssb",
        method="fixed_frame_shape",
        clock=ClockConfig("fixed_period"),
        segment_days=3000.0,
        residual_degree=24,
        shape_degree=16,
        quant=QuantConfig(0.5, "growth:1.25"),
        segment_domain_expansion_fraction=0.0075,
        estimated_size_kib=2.078,
        worst_p99_arcsec=0.000535508,
        worst_max_arcsec=0.000660875,
        notes="f=0.0075 selected to reduce analytic-velocity boundary spikes without increasing angular pmax.",
    ),
    "saturn": BodyConfig(
        body="saturn",
        center="ssb",
        method="fixed_frame_shape",
        clock=ClockConfig("fixed_period"),
        segment_days=3500.0,
        residual_degree=24,
        shape_degree=16,
        quant=QuantConfig(1.0, "growth:1.25"),
        segment_domain_expansion_fraction=0.01,
        estimated_size_kib=1.859,
        worst_p99_arcsec=0.000583687,
        worst_max_arcsec=0.000764458,
        notes="f=0.01 promoted after 7-epoch expansion validation.",
    ),
    "uranus": BodyConfig(
        body="uranus",
        center="ssb",
        method="fixed_frame_shape",
        clock=ClockConfig("fixed_period"),
        segment_days=8000.0,
        residual_degree=30,
        shape_degree=12,
        quant=QuantConfig(1.6, "linear:0.5"),
        segment_domain_expansion_fraction=0.01,
        estimated_size_kib=3.475,
        worst_p99_arcsec=0.000421106,
        worst_max_arcsec=0.000456195,
        notes="f=0.01 selected to suppress segment-boundary velocity spikes.",
    ),
    "neptune": BodyConfig(
        body="neptune",
        center="ssb",
        method="fixed_frame_shape",
        clock=ClockConfig("fixed_period"),
        segment_days=10000.0,
        residual_degree=30,
        shape_degree=12,
        quant=QuantConfig(3.5, "flat"),
        segment_domain_expansion_fraction=0.005,
        estimated_size_kib=2.354,
        worst_p99_arcsec=0.000469321,
        worst_max_arcsec=0.000498496,
        notes="f=0.005 selected to reduce analytic-velocity boundary spikes while preserving angular pmax.",
    ),
    "pluto": BodyConfig(
        body="pluto",
        center="ssb",
        method="fixed_frame_shape",
        clock=ClockConfig("fixed_period"),
        segment_days=10000.0,
        residual_degree=30,
        shape_degree=12,
        quant=QuantConfig(3.5, "growth:1.25"),
        segment_domain_expansion_fraction=0.0075,
        estimated_size_kib=2.576,
        worst_p99_arcsec=0.000473854,
        worst_max_arcsec=0.000527245,
        notes="f=0.0075 selected to reduce analytic-velocity boundary spikes without increasing position pmax.",
    ),
    "moon": BodyConfig(
        body="moon",
        center="earth",
        method="mean_lunar_apsis_frame_shape",
        clock=ClockConfig(
            "global_anomalistic_century_i16",
            period_days=MOON_GLOBAL_PERIOD,
            phase_start_jd=MOON_PHASE_START,
            correction="all-century int16 median linear table",
        ),
        residual_degree=24,
        shape_degree=32,
        quant=QuantConfig(0.00025, "flat"),
        segment_domain_expansion_fraction=0.01,
        estimated_size_kib=206.641,
        worst_p99_arcsec=0.000501518,
        worst_max_arcsec=0.000829396,
        notes="Full 306-entry century table is stored in every Moon file.",
    ),
}

DEFAULT_BODY_ORDER = [
    "sun",
    "mercury",
    "venus",
    "emb",
    "mars",
    "jupiter",
    "saturn",
    "uranus",
    "neptune",
    "pluto",
    "moon",
]


def planet_total_kib(include_sun: bool = True, include_moon: bool = False) -> float:
    names = ["mercury", "venus", "emb", "mars", "jupiter", "saturn", "uranus", "neptune", "pluto"]
    if include_sun:
        names.insert(0, "sun")
    if include_moon:
        names.append("moon")
    return sum(float(CONFIGS[name].estimated_size_kib or 0.0) for name in names)
