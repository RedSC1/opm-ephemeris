# OPM Ephemeris

Compact Chebyshev ephemeris representation with Python reference tools, validation artifacts, and a technical preprint.

This repository contains paper/demo code for generating, reading, validating, and benchmarking prototype OPM1 files. It is not the production C++ implementation.

## Technical preprint

The technical preprint is archived on Zenodo:

- Zenodo record: <https://zenodo.org/records/20636663>

The PDFs and sources are also included in `docs/`:

- English: [`docs/opm-short-paper-en.pdf`](docs/opm-short-paper-en.pdf) ([source](docs/opm-short-paper-en.md))
- Chinese: [`docs/opm-short-paper-zh.pdf`](docs/opm-short-paper-zh.pdf) ([source](docs/opm-short-paper-zh.md))

## Full-range validation overview

A private full-range generation run over the strict-safe DE441 interval, JD `-3092455.0..7991545.0` (about 30,346 Julian years), produced 51 OPM shards. The final polished `.opm` files occupy about 84 MiB. A 512-node-per-segment OPM-vs-DE441 geocentric angular validation over all shards produced the following overview. Values are in arcseconds; `p50`/`p95`/`p99` are the worst shard-level percentiles across the 51 shards, while `max` is the full-run maximum.

| Body | Samples | worst p50 | worst p95 | worst p99 | max |
|---|---:|---:|---:|---:|---:|
| Sun | 31,602,390 | 0.0000836 | 0.000303 | 0.000396 | 0.000603 |
| Moon | 206,384,055 | 0.000199 | 0.000348 | 0.000391 | 0.000580 |
| Mercury | 64,663,188 | 0.000106 | 0.000337 | 0.000520 | 0.00116 |
| Venus | 25,344,811 | 0.000131 | 0.000649 | 0.00117 | 0.00276 |
| Mars | 8,307,737 | 0.000151 | 0.000635 | 0.00101 | 0.00248 |
| Jupiter | 1,920,210 | 0.000183 | 0.000371 | 0.000445 | 0.000658 |
| Saturn | 1,650,372 | 0.000185 | 0.000364 | 0.000456 | 0.000809 |
| Uranus | 736,719 | 0.000202 | 0.000374 | 0.000432 | 0.000579 |
| Neptune | 594,618 | 0.000215 | 0.000402 | 0.000458 | 0.000575 |
| Pluto | 594,618 | 0.000180 | 0.000357 | 0.000443 | 0.000612 |

The repository does not redistribute the DE441 BSP file used for generation or validation.

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

The script uses `swe.calc()` through `pyswisseph`; the Swiss Ephemeris file path is supplied by the caller and is not hardcoded. For the 1800--2400 interval used by the paper, the corresponding Swiss Ephemeris core files in the local DE441-based installation are `sepl_18.se1` and `semo_18.se1`. In Swiss Ephemeris terminology, `sepl_18.se1` is the planetary file used for Sun through Pluto, while `semo_18.se1` is the Moon file. This does not mean that `sepl_18.se1` stores a separate independent fitted series for every requested output convention; Swiss Ephemeris applies its own internal coordinate conventions and transformations, e.g. deriving barycentric Sun from Earth-related heliocentric/barycentric quantities when needed.

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

## License

Code in this repository is licensed under the Apache License 2.0; see `LICENSE`.

Paper sources, generated PDFs, figures, plots, and generated OPM release artifacts are licensed under Creative Commons Attribution 4.0 International (CC BY 4.0), unless otherwise noted; see `LICENSE-DOCS`.

JPL/NAIF ephemeris data, Swiss Ephemeris, `pyswisseph`, fonts, and other third-party dependencies are governed by their own licenses and terms. They are not redistributed as part of this repository; see `THIRD_PARTY_NOTICES.md`.

## Included clock constants

`data/opm_mercury_cheb8_clock.json` stores the persisted Mercury event-time Chebyshev correction. `data/opm_moon_century_i16_clock.json` stores the Moon 306-entry int16 century table. Keeping these in `data/` avoids rescanning the full source range for normal demo generation.
