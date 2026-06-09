"""Body-packed OPM demo configuration profile.

These overrides keep the century-sliced reference settings compact while giving
single-file body-wide packages a little more worst-case margin over long
coverage ranges.
"""
from __future__ import annotations

from dataclasses import replace

from opm_demo.body_configs import CONFIGS, BodyConfig, QuantConfig


BODY_PACKED_CONFIGS: dict[str, BodyConfig] = dict(CONFIGS)

# Full-range Mercury needs both a little more residual capacity and slightly
# tighter quantization to keep the body-wide max below 0.001 arcsec.
BODY_PACKED_CONFIGS["mercury"] = replace(
    CONFIGS["mercury"],
    residual_degree=26,
    quant=QuantConfig(0.028, "linear:0.65"),
)

# Full-range Mars has a few mid-segment residual peaks with degree 28. Degree 30
# lowers the full-range max error substantially for a small size increase.
BODY_PACKED_CONFIGS["mars"] = replace(CONFIGS["mars"], residual_degree=30)

# Full-range Uranus has sparse long segments; degree 30 gives the body-wide
# residual fit enough margin without changing the century-sliced profile.
BODY_PACKED_CONFIGS["uranus"] = replace(CONFIGS["uranus"], residual_degree=30)
BODY_PACKED_CONFIGS["neptune"] = replace(
    CONFIGS["neptune"],
    residual_degree=30,
    quant=QuantConfig(3.5, "flat"),
    segment_domain_expansion_fraction=0.01,
)

# Seven-century Pluto body-packed files share one longer reference shape; the
# default quantization occasionally lets the validation max cross 0.001 arcsec.
BODY_PACKED_CONFIGS["pluto"] = replace(
    CONFIGS["pluto"],
    residual_degree=28,
    quant=QuantConfig(4.0, "growth:1.25"),
    segment_domain_expansion_fraction=0.01,
)
