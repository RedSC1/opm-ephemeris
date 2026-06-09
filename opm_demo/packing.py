"""Bit packing and quantization helpers for the OPM Python demo."""
from __future__ import annotations

import numpy as np


class BitWriter:
    def __init__(self) -> None:
        self.data = bytearray()
        self.acc = 0
        self.nbits = 0

    def write(self, value: int, width: int) -> None:
        if width <= 0:
            return
        self.acc |= (int(value) & ((1 << width) - 1)) << self.nbits
        self.nbits += width
        while self.nbits >= 8:
            self.data.append(self.acc & 0xFF)
            self.acc >>= 8
            self.nbits -= 8

    def finish(self) -> bytes:
        if self.nbits:
            self.data.append(self.acc & 0xFF)
            self.acc = 0
            self.nbits = 0
        return bytes(self.data)


class BitReader:
    def __init__(self, data: bytes) -> None:
        self.data = data
        self.pos = 0
        self.acc = 0
        self.nbits = 0

    def read(self, width: int) -> int:
        while self.nbits < width:
            if self.pos >= len(self.data):
                raise EOFError("bitstream underflow")
            self.acc |= self.data[self.pos] << self.nbits
            self.pos += 1
            self.nbits += 8
        value = self.acc & ((1 << width) - 1)
        self.acc >>= width
        self.nbits -= width
        return value


def zigzag_encode(value: int) -> int:
    return value * 2 if value >= 0 else -value * 2 - 1


def zigzag_decode(value: int) -> int:
    return value // 2 if value % 2 == 0 else -(value // 2) - 1


def zigzag_bit_length(value: int) -> int:
    return max(1, zigzag_encode(value).bit_length())


def pack_degree_major(q_arrays: list[np.ndarray], widths: np.ndarray | list[int]) -> bytes:
    """Pack one axis of segment-major coefficient arrays in degree-major order."""
    width_arr = np.asarray(widths, dtype=int)
    arr = np.vstack([np.asarray(a, dtype=np.int64).ravel() for a in q_arrays])
    if arr.shape[1] != len(width_arr):
        raise ValueError("width table length does not match coefficient count")
    bw = BitWriter()
    for coeff_idx, width in enumerate(width_arr):
        for value in arr[:, coeff_idx]:
            encoded = zigzag_encode(int(value))
            if encoded >= (1 << int(width)):
                raise ValueError(f"value does not fit width {int(width)} at coeff {coeff_idx}: {value}")
            bw.write(encoded, int(width))
    return bw.finish()


def unpack_degree_major(data: bytes, segment_count: int, degree: int, widths: np.ndarray | list[int]) -> list[np.ndarray]:
    width_arr = np.asarray(widths, dtype=int)
    if len(width_arr) != degree + 1:
        raise ValueError("width table length does not match degree")
    br = BitReader(data)
    arr = np.zeros((segment_count, degree + 1), dtype=np.int64)
    for coeff_idx, width in enumerate(width_arr):
        for seg_idx in range(segment_count):
            arr[seg_idx, coeff_idx] = zigzag_decode(br.read(int(width)))
    return [arr[i].copy() for i in range(segment_count)]


def exact_axis_degree_widths(q_arrays: list[np.ndarray]) -> np.ndarray:
    arr = np.vstack([np.asarray(a, dtype=np.int64).ravel() for a in q_arrays])
    return np.asarray([max(zigzag_bit_length(int(v)) for v in arr[:, k]) for k in range(arr.shape[1])], dtype=np.uint8)


def degree_quant_steps(degree: int, base_quant_unit_km: float, pattern: str) -> np.ndarray:
    x = np.arange(degree + 1, dtype=np.float64) / float(degree)
    if pattern == "flat":
        mult = np.ones(degree + 1, dtype=np.float64)
    elif pattern.startswith("growth:"):
        mult = float(pattern.split(":", 1)[1]) ** x
    elif pattern.startswith("linear:"):
        mult = 1.0 + float(pattern.split(":", 1)[1]) * x
    else:
        raise ValueError(f"unknown degree quant pattern: {pattern}")
    return base_quant_unit_km * mult
