# Body-Packed Guarded Rounding Tuning Notes

> **Status: superseded tuning history.** This note records intermediate guarded-rounding experiments. Do not use its package totals as current guidance. The current final selected full-range sizes and metrics are in `docs/final-body-packed-tuning-results.md`, and the current compact size table is in `docs/size-summary.md`.

This document records writer-only tuning candidates using the same comparison format for each body:

- **A. Original file**: current size and validation accuracy.
- **B. Original quant + guarded rounding**: same quantization and size, but with width-safe guarded rounding applied to the tail. This shows what the rounding optimizer can do without saving space.
- **C. Relaxed quant + guarded rounding**: real size-saving candidate. Compare this primarily against **B**, not only against **A**.

All validation figures below use 32 validation nodes per segment against DE441 unless otherwise noted.

## Moon checkpoint

Baseline file:

```text
out/body-packed/full/moon/moon.pef
```

Candidate file:

```text
out/body-packed/tuning/moon/q034-guarded085/moon.pef
```

Candidate settings:

```text
quant base       = 0.00034 km
quant pattern    = flat
guard threshold  = 0.00085 arcsec
objective        = p99-guarded greedy
width constraint = no width increase
optimized segs   = 3,415 / 402,838
```

Important writer note: preserve the baseline PEF `period_days` and `phase_start_jd` exactly when writing Moon candidates. Reusing truncated constants from `body_configs.py` caused an approximately `0.02"` Moon validation error.

### Three-way comparison

| Case | Quant | Guarded? | Size MiB | Width sum | p50 arcsec | p95 arcsec | p99 arcsec | p99.5 arcsec | p99.9 arcsec | max arcsec | Notes |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| A. Original | 0.00025 | No | 62.574 | 1303 | 0.000228298 | 0.000414955 | 0.000500016 | 0.000531905 | 0.000599011 | 0.000888801 | Current full Moon PEF |
| B. Original quant + guarded | 0.00025 | Yes, threshold `0.00085`, width-safe | 62.574 | 1303 | 0.000228299 | 0.000414956 | 0.000500012 | 0.000531902 | 0.000598983 | 0.000848564 | 5 selected segments; no size change |
| C. Relaxed quant + guarded | 0.00034 | Yes, threshold `0.00085`, width-safe | 60.894 | 1268 | 0.000303027 | 0.000548237 | 0.000656208 | 0.000696230 | 0.000773251 | 0.000849998 | Written candidate validates PASS |

### C vs B tradeoff

| Metric | B. Original quant + guarded | C. Relaxed quant + guarded | Change |
|---|---:|---:|---:|
| Size | 62.574 MiB | 60.894 MiB | **-1.681 MiB** |
| p50 | 0.000228299 | 0.000303027 | +32.7% |
| p95 | 0.000414956 | 0.000548237 | +32.1% |
| p99 | 0.000500012 | 0.000656208 | +31.2% |
| p99.5 | 0.000531902 | 0.000696230 | +30.9% |
| p99.9 | 0.000598983 | 0.000773251 | +29.1% |
| max | 0.000848564 | 0.000849998 | +0.17% |

Summary:

```text
Moon q=0.00034 guarded085 saves 1.681 MiB.
The max remains essentially fixed near 0.00085 arcsec, while the overall error distribution worsens by about 30% versus original-quant guarded rounding.
```

### Full-set impact

Current full body-packed set:

```text
96.001 MiB
```

Replacing only Moon with the `q034-guarded085` candidate:

```text
94.320 MiB
```

Remaining gap to 90 MiB:

```text
4.320 MiB
```

## Mercury preliminary observation

Baseline file:

```text
out/body-packed/full/inner/mercury.pef
```

Relaxed candidate tested diagnostically:

```text
out/body-packed/tuning/mercury/d26-q032/mercury.pef
```

### Three-way comparison

| Case | Quant | Guarded? | Size MiB | Width sum | p50 arcsec | p95 arcsec | p99 arcsec | p99.5 arcsec | p99.9 arcsec | max arcsec | Notes |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| A. Original | 0.028, `linear:0.65` | No | 14.456 | 961 | 0.000215143 | 0.000414746 | 0.000510497 | 0.000545710 | 0.000619839 | 0.000950487 | Current full Mercury PEF, PASS |
| B. Original quant + guarded | 0.028, `linear:0.65` | Yes, threshold `0.00085` | 14.456 | 961 | 0.000215146 | 0.000414759 | 0.000510511 | 0.000545718 | 0.000619788 | 0.000846258 | 4 selected segments; no written candidate |
| C. Relaxed quant + guarded | 0.032, `linear:0.65` | Yes, threshold `0.00085` | 14.246 | 947 | 0.000245404 | 0.000472733 | 0.000580963 | 0.000621267 | 0.000704173 | 0.000849913 | Diagnostic only; 129 selected segments; not written yet |

### C vs B tradeoff

| Metric | B. Original quant + guarded | C. Relaxed quant + guarded | Change |
|---|---:|---:|---:|
| Size | 14.456 MiB | 14.246 MiB | **-0.211 MiB** |
| p50 | 0.000215146 | 0.000245404 | +14.1% |
| p95 | 0.000414759 | 0.000472733 | +14.0% |
| p99 | 0.000510511 | 0.000580963 | +13.8% |
| p99.5 | 0.000545718 | 0.000621267 | +13.8% |
| p99.9 | 0.000619788 | 0.000704173 | +13.6% |
| max | 0.000846258 | 0.000849913 | +0.43% |


## Venus / EMB / Mars approximate quant scans

These are **not final candidate rows**. They are quick scans made by taking the currently stored/reconstructed PEF coefficients, requantizing them more coarsely, and validating the result. This is useful for triage, but final candidates should be regenerated from unquantized fits/caches and compared with the A/B/C format above.

### Venus approximate scan

Baseline:

```text
out/body-packed/full/inner/venus.pef
size = 5.725 MiB
p99  = 0.000417384 arcsec
max  = 0.000692822 arcsec
PASS
```

Approximate relaxed quant highlights:

| Quant base | Est. size MiB | Est. save MiB | p99 arcsec | p99.9 arcsec | max arcsec | seg >= 0.001 |
|---:|---:|---:|---:|---:|---:|---:|
| 0.070 | 5.619 | 0.106 | 0.000590128 | 0.000706782 | 0.001026656 | 1 |
| 0.080 | 5.566 | 0.159 | 0.000678033 | 0.000811383 | 0.001106109 | 27 |
| 0.100 | 5.442 | 0.283 | 0.000776338 | 0.000931033 | 0.001343906 | 461 |

Summary: Venus quant relaxation appears to save only about `0.1-0.2 MiB` before tail rescue becomes more involved, so it is unlikely to be a major contributor.

### EMB approximate scan

Baseline:

```text
out/body-packed/full/inner/emb.pef
size = 4.739 MiB
p99  = 0.000455092 arcsec
max  = 0.000787517 arcsec
PASS
```

Approximate relaxed quant highlights using pattern `growth:1.25`:

| Quant base | Est. size MiB | Est. save MiB | p99 arcsec | p99.9 arcsec | max arcsec | seg >= 0.001 |
|---:|---:|---:|---:|---:|---:|---:|
| 0.030 | 4.573 | 0.167 | 0.000469989 | 0.000620081 | 0.000822148 | 0 |
| 0.034 | 4.508 | 0.232 | 0.000470966 | 0.000623673 | 0.000893110 | 0 |
| 0.038 | 4.453 | 0.286 | 0.000472517 | 0.000627184 | 0.000867567 | 0 |
| 0.042 | 4.410 | 0.330 | 0.000496916 | 0.000657148 | 0.000898920 | 0 |

Summary: EMB has a small but clean-looking quant relaxation window, perhaps `0.2-0.3 MiB`, with max still under `0.001` in the approximate scan.

### Mars approximate scan

Baseline:

```text
out/body-packed/full/mars/mars.pef
size = 2.512 MiB
p99  = 0.000358258 arcsec
max  = 0.000652955 arcsec
PASS
```

Approximate relaxed quant highlights using flat quant:

| Quant base | Est. size MiB | Est. save MiB | p99 arcsec | p99.9 arcsec | max arcsec | seg >= 0.001 |
|---:|---:|---:|---:|---:|---:|---:|
| 0.070 | 2.378 | 0.135 | 0.000415191 | 0.000532655 | 0.000749443 | 0 |
| 0.080 | 2.341 | 0.171 | 0.000470887 | 0.000585156 | 0.000934438 | 0 |
| 0.090 | 2.318 | 0.195 | 0.000453963 | 0.000570333 | 0.000785992 | 0 |
| 0.100 | 2.295 | 0.218 | 0.000472036 | 0.000590186 | 0.000809631 | 0 |

Summary: Mars can probably save about `0.15-0.22 MiB` from quant relaxation, but the absolute contribution is small.

## Current direction after triage

Current confirmed or plausible writer-only savings from simple quant relaxation plus guarded rounding are roughly:

```text
Moon       confirmed  1.681 MiB
Mercury    plausible  0.211 MiB
Venus      approximate 0.1-0.2 MiB
EMB        approximate 0.2-0.3 MiB
Mars       approximate 0.15-0.22 MiB
```

Even optimistic totals are only around `2.4-2.6 MiB`, still short of the `4.320 MiB` remaining after the Moon candidate. This suggests simple quant relaxation + guarded rounding is useful but probably insufficient alone; additional levers such as degree changes, segment model changes, or accepting a more aggressive Moon candidate may be needed.
