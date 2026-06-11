# OPM Python demo

Reference Python code for generating, reading, validating, and benchmarking prototype OPM1 files. This repository is paper/demo code, not the production C++ implementation.

## Install

```bash
python3 -m pip install -r requirements.txt
```

You need a local DE404/DE441-style BSP file. The path is always supplied explicitly; the demo does not hardcode any BSP location.

## Generate a range

Generate the J2000 Julian century:

```bash
python3 generate_range.py \
  --de441 /path/to/de441.bsp \
  --all \
  --jd-start 2451545.0 \
  --days 36525 \
  --output-root out/small/j2000-opm
```

Generate one body:

```bash
python3 generate_range.py \
  --de441 /path/to/de441.bsp \
  --body moon \
  --jd-start 2451545.0 \
  --days 36525 \
  --output out/small/j2000-opm/moon.opm
```

Recommended release-style generation uses the 600-year shard used by the paper:

```bash
python3 generate_range.py \
  --de441 /path/to/de441.bsp \
  --all \
  --jd-start 2378495.0 \
  --days 219150 \
  --output-root out/opm600/j1800 \
  --polish \
  --validate
```

## Validate

```bash
python3 validate_opm.py --de441 /path/to/de441.bsp out/small/j2000-opm
```

## Dense Swiss Ephemeris comparison

Reproduce the paper's dense geocentric comparison and SVG plots with explicit local paths for DE441 and Swiss Ephemeris data:

```bash
python3 tools/dense_compare_opm_swiss_geocentric.py \
  --de441 /path/to/de441.bsp \
  --opm-root out/opm600/j1800 \
  --swiss-ephe /path/to/swiss/ephe \
  --nodes-per-segment 512 \
  --plot-dir out/opm600/j1800-plots \
  > out/opm600/j1800-dense-512.txt
```

The script uses `swe.calc()` through `pyswisseph`; the Swiss Ephemeris file path is supplied by the caller and is not hardcoded.

## Read positions

```bash
python3 examples/read_position.py out/opm600/j1800/moon.opm 2451545.0
```

Output columns are:

```text
jd x_km y_km z_km
```

## Coverage model

An OPM file is coverage-range based. The header stores source and coverage JD ranges, body/center IDs, model kinds, segment addressing, and table/payload locations. It does not store a century index. A one-century shard and a 600-year shard are the same OPM file model with different coverage ranges.

## Benchmarks

Run random-JD accuracy/speed checks across an OPM root:

```bash
python3 benchmark_random_jd.py \
  --de441 /path/to/de441.bsp \
  --opm-root out/opm600/c+0000 \
  --samples 10000 \
  --seed 1
```

Benchmark reconstruction throughput against direct BSP reads:

```bash
python3 benchmark_read.py \
  --de441 /path/to/de441.bsp \
  --opm-root out/small/j2000-opm \
  --samples 100000
```

## Active layout

Generated files are ignored by git and are organized under `out/`. Historical body-packed and tuning experiments have been moved under `legacy/`; current workflows should use `generate_range.py`, `validate_opm.py`, and the OPM polish tools under `tools/`.

## Integrity checks

OPM1 files written by this demo include CRC-64/ECMA-182 checksums:

```text
header_crc64   computed over the fixed header with this field zeroed
payload_crc64  computed over all bytes after the fixed header
```

Readers validate them by default; pass `--no-crc` to validation or benchmark scripts to skip this check when desired.

## Included clock constants

`data/opm_mercury_cheb8_clock.json` stores the persisted Mercury event-time Chebyshev correction. `data/opm_moon_century_i16_clock.json` stores the Moon 306-entry int16 century table. Keeping these in `data/` avoids rescanning the full source range for normal demo generation.
