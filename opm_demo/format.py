"""OPM1 binary format constants used by the Python demo."""
from __future__ import annotations

import struct

MAGIC = b"OPM1"
ENDIAN_TAG = 0x0102
FIXED_HEADER_SIZE = 320
HEADER_MINOR_VERSION = 0
FLAGS = 0

JD_J2000 = 2451545.0
CENTURY_DAYS = 36525.0
SOURCE_DE441 = 1
POSITION_KM = 1
TIME_TDB_JD = 1
AXIS_COUNT = 3

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

MODEL_RAW_XYZ_CHEB = 1
MODEL_MEAN_APSIS_FRAME_SHAPE = 2
MODEL_MEAN_LUNAR_APSIS_FRAME_SHAPE = 3
MODEL_FIXED_FRAME_SHAPE = 4

CLOCK_RAW_FIXED = 0
CLOCK_GLOBAL_ANOMALISTIC = 2
CLOCK_GLOBAL_ANOMALISTIC_CHEB8 = 3
CLOCK_GLOBAL_ANOMALISTIC_CENTURY_I16 = 4
CLOCK_GLOBAL_INERTIAL_PHASE = 5
CLOCK_FIXED_PERIOD = 6

FRAME_NONE = 0
FRAME_CHEB1_PLANE_APSIS = 1
REFERENCE_SHAPE_NONE = 0
REFERENCE_SHAPE_MEAN_XY_CHEB = 1
RESIDUAL_XYZ_DEGREE_MAJOR_EXACT_WIDTH = 1
SEGMENT_FIXED_DAYS = 0
SEGMENT_AFFINE_PERIOD_PHASE = 1

CLOCK_CORRECTION_NONE = 0
CLOCK_CORRECTION_CHEB_EVENT_TIME_F64 = 1
CLOCK_CORRECTION_CENTURY_I16_LINEAR_TABLE = 2

MERCURY_CLOCK_DEGREE = 8
MOON_CLOCK_INTERPOLATION_LINEAR = 1
MOON_CLOCK_STATISTIC_MEDIAN = 1
MOON_CLOCK_TABLE_STRUCT = struct.Struct("<dddIIII")

DESCRIPTOR_MODEL_POLICY_V1 = 100
DESCRIPTOR_ENTRY_STRUCT = struct.Struct("<HHIQ")
MODEL_POLICY_STRUCT = struct.Struct("<12HII")

HEADER_STRUCT = struct.Struct(
    "<"
    "4sHBBIIB3s"
    "BB"
    "dddd"
    "HH10B"
    "I"
    "ddddd"
    "fB3s"
    "6B"
    "B7sQ"
    "15Q"
    "58s"
)
assert HEADER_STRUCT.size == FIXED_HEADER_SIZE
HEADER_CRC64_OFFSET = 246
PAYLOAD_CRC64_OFFSET = 254
_CRC64_ECMA_POLY = 0x42F0E1EBA9EA3693
_CRC64_ECMA_TABLE: tuple[int, ...] | None = None


def _crc64_ecma_table() -> tuple[int, ...]:
    global _CRC64_ECMA_TABLE
    if _CRC64_ECMA_TABLE is None:
        table = []
        for byte in range(256):
            value = byte << 56
            for _ in range(8):
                if value & (1 << 63):
                    value = ((value << 1) ^ _CRC64_ECMA_POLY) & 0xFFFFFFFFFFFFFFFF
                else:
                    value = (value << 1) & 0xFFFFFFFFFFFFFFFF
            table.append(value)
        _CRC64_ECMA_TABLE = tuple(table)
    return _CRC64_ECMA_TABLE


def crc64_ecma(data: bytes) -> int:
    """Return CRC-64/ECMA-182 for data."""
    crc = 0
    table = _crc64_ecma_table()
    for byte in data:
        crc = table[((crc >> 56) ^ byte) & 0xFF] ^ ((crc << 8) & 0xFFFFFFFFFFFFFFFF)
    return crc


assert crc64_ecma(b"123456789") == 0x6C40DF5F0B497347


def body_name_from_id(body_id: int) -> str:
    for name, value in OPM_BODY_IDS.items():
        if value == body_id:
            return name
    raise ValueError(f"unknown OPM body id {body_id}")
