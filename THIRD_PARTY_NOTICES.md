# Third-Party Notices

This repository contains OPM demo code, paper sources, generated figures, and generated OPM artifacts. Several external packages and data sources are used to reproduce the results. They are not relicensed by this repository.

## JPL / NAIF ephemeris data

The generation and validation workflows require a local DE404/DE441-style SPK/BSP ephemeris file supplied by the user. JPL/NAIF ephemeris kernels are third-party data and are governed by their own terms and notices. They are not included in this repository.

## Swiss Ephemeris and pyswisseph

Swiss Ephemeris is third-party software and data distributed by Astrodienst under its own licensing terms. The Swiss comparison scripts in this repository require a local Swiss Ephemeris installation and ephemeris data path supplied by the user.

The Python package `pyswisseph` is a binding to the Swiss Ephemeris C library and is governed by its own license and the license terms of Swiss Ephemeris. This repository does not include or redistribute Swiss Ephemeris source code, binaries, or `.se1` ephemeris files.

## Python dependencies

Python dependencies listed in `requirements.txt` are third-party packages governed by their own licenses. Install them from their upstream package indexes or repositories.

## Fonts and PDF tooling

The PDF build scripts use locally installed fonts and TeX/Pandoc tooling. Those tools and fonts are governed by their own licenses and are not distributed as part of this repository.
