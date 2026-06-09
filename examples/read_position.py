#!/usr/bin/env python3
"""Print reconstructed OPM positions for one or more Julian dates."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from opm_demo.validator import read_opm, reconstruct_positions


def main() -> int:
    parser = argparse.ArgumentParser(description="Read a OPM file and print reconstructed positions")
    parser.add_argument("opm", type=Path)
    parser.add_argument("jd", type=float, nargs="+", help="TDB Julian date(s)")
    args = parser.parse_args()
    opm = read_opm(args.opm)
    jds = np.asarray(args.jd, dtype=np.float64)
    xyz = reconstruct_positions(opm, jds)
    for jd, row in zip(jds, xyz):
        print(f"{jd:.9f} {row[0]:.12e} {row[1]:.12e} {row[2]:.12e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
