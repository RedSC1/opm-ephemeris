# Final body-packed PEF tuning results

This records the final tuned body-packed candidates from the full-range PEF tuning pass. Generated files under `out/` are local experimental artifacts and are not source-controlled fixtures.

## Metric policy

- **Sun** is optimized with a **linear km** metric. Its own SSB-to-Sun angular error is diagnostic only because the SSB-to-Sun vector is short and angular error is pathological.
- **SSB-centered planets and EMB** are tuned against the final quantized Sun anchor:

  ```text
  (Body_pef - Sun_pef) vs (Body_DE441 - Sun_DE441)
  ```

  Final Sun anchor:

  ```text
  out/body-packed/tuning/sun/q010-pmax/sun.pef
  ```

- **Mercury / Venus** are stored as Sun-to-body vectors, so ordinary body validation is already the heliocentric metric.
- **Moon** is stored as Earth-to-Moon. The Moon error contribution to Earth-Sun was checked separately and is negligible after the Earth-Moon mass-ratio scale factor.
- Unless noted otherwise, validation uses 32 nodes per segment against DE441.

## Final selected files

| Body | Final candidate | Size MiB | Segments | Storage | Model | Residual degree | Shape degree | Quant steps | Width sum | Axis bits |
|---|---|---:|---:|---|---|---:|---:|---|---:|---|
| Sun | `out/body-packed/tuning/sun/q010-pmax/sun.pef` | 4.874 | 61,666 | SSB->Sun | `raw_xyz_cheb` | 25 | none | flat `0.01 km` | 663 | `(230, 228, 205)` |
| Mercury | `out/body-packed/tuning/mercury/d26-q032-global-tail-full/mercury.pef` | 14.246 | 126,177 | Sun->Mercury | `mean_apsis_frame_shape` | 26 | 40 | `0.032..0.0528 km` (`linear:0.65`) | 947 | `(330, 321, 296)` |
| Venus | `out/body-packed/tuning/venus/q08-global-tail-full/venus.pef` | 5.566 | 49,396 | Sun->Venus | `mean_apsis_frame_shape` | 24 | 40 | flat `0.08 km` | 945 | `(333, 330, 282)` |
| EMB | `out/body-packed/tuning/emb/sun-q010-pmax-auto-revisit/emb.pef` | 4.739 | 30,389 | SSB->EMB | `fixed_frame_shape` | 28 | 22 | `0.02..0.025 km` (`growth:1.25`) | 1,308 | `(474, 471, 363)` |
| Mars | `out/body-packed/tuning/mars/sun-q010-pmax-d30-auto-revisit-slack1e-6/mars.pef` | 2.512 | 16,156 | SSB->Mars | `fixed_frame_shape` | 30 | 22 | flat `0.04 km` | 1,304 | `(469, 468, 367)` |
| Jupiter | `out/body-packed/tuning/jupiter/sun-q010-pmax-auto-revisit-slack1e-6/jupiter.pef` | 0.452 | 3,699 | SSB->Jupiter | `fixed_frame_shape` | 24 | 16 | `0.5..0.625 km` (`growth:1.25`) | 1,024 | `(389, 390, 245)` |
| Saturn | `out/body-packed/tuning/saturn/sun-q010-pmax-auto-revisit-slack1e-6/saturn.pef` | 0.351 | 3,171 | SSB->Saturn | `fixed_frame_shape` | 24 | 16 | `1.0..1.25 km` (`growth:1.25`) | 927 | `(347, 342, 238)` |
| Uranus | `out/body-packed/tuning/uranus/sun-q010-pmax-auto-revisit-slack1e-6/uranus.pef` | 0.143 | 1,386 | SSB->Uranus | `fixed_frame_shape` | 30 | 12 | `1.6..2.4 km` (`linear:0.5`) | 863 | `(340, 342, 181)` |
| Neptune | `out/body-packed/tuning/neptune/sun-q010-pmax-auto-revisit-slack1e-6/neptune.pef` | 0.081 | 1,109 | SSB->Neptune | `fixed_frame_shape` | 30 | 12 | flat `3.5 km` | 604 | `(249, 240, 115)` |
| Pluto | `out/body-packed/tuning/pluto/sun-q010-pmax-auto-revisit-slack1e-6/pluto.pef` | 0.092 | 1,109 | SSB->Pluto | `fixed_frame_shape` | 28 | 12 | `4.0..5.0 km` (`growth:1.25`) | 691 | `(267, 262, 162)` |
| Moon | `out/body-packed/tuning/moon/q034-global-tail-full/moon.pef` | 60.894 | 402,838 | Earth->Moon | `mean_lunar_apsis_frame_shape` | 24 | 32 | flat `0.00034 km` | 1,268 | `(448, 452, 368)` |

Totals:

```text
All final files, including Sun: 93.951 MiB
All final files, excluding Sun: 89.077 MiB
```

## Final ordinary validation metrics

These are direct file validation metrics. For SSB-centered bodies, see the next section for the Sun-anchor composite tuning metric actually used during pmax optimization.

| Body | p50 arcsec | p95 arcsec | p99 arcsec | Max arcsec | Status | Notes |
|---|---:|---:|---:|---:|---|---|
| Sun | 0.005039128 | 0.015928971 | 0.033542784 | 1.058775260 | DIAG | Angular diagnostic only; Sun was optimized in km. |
| Mercury | 0.000282005 | 0.000401767 | 0.000431557 | 0.000554868 | PASS | Sun->Mercury native vector. |
| Venus | 0.000328459 | 0.000461560 | 0.000498806 | 0.000616284 | PASS | Sun->Venus native vector. |
| EMB | 0.000097625 | 0.000295291 | 0.000388635 | 0.000564649 | PASS | File sanity check; final tuning used EMB-Sun composite. |
| Mars | 0.000131254 | 0.000236922 | 0.000279050 | 0.000373265 | PASS | d30 selected; remote validation. |
| Jupiter | 0.000260654 | 0.000374364 | 0.000401563 | 0.000450486 | PASS | Sun-anchor pmax tuned. |
| Saturn | 0.000225223 | 0.000389613 | 0.000449774 | 0.000633259 | PASS | Sun-anchor pmax tuned. |
| Uranus | 0.000277117 | 0.000390323 | 0.000420470 | 0.000478815 | PASS | Sun-anchor pmax tuned. |
| Neptune | 0.000288219 | 0.000409380 | 0.000439448 | 0.000493354 | PASS | Sun-anchor pmax tuned. |
| Pluto | 0.000248005 | 0.000440072 | 0.000503303 | 0.000591930 | PASS | Sun-anchor pmax tuned. |
| Moon | 0.000346058 | 0.000485152 | 0.000518640 | 0.000638824 | PASS | Earth->Moon native vector. |

## Tuning objective metrics

### Sun km metric

Sun pmax tuning used a linear km error metric, not angular error.

| Body | Metric | Initial p50 | Initial p95 | Initial p99 | Initial max | Optimized p50 | Optimized p95 | Optimized p99 | Optimized max | Accepted | Rejected nochange | Rejected budget |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Sun | km | 0.017724974 | 0.032046069 | 0.038482733 | 0.065142974 | 0.021060806 | 0.028944300 | 0.030697959 | 0.037622403 | 61,325 | 341 | 0 |

### Sun-anchor composite arcsec metric

For these rows, the metric is:

```text
(Body_pef - Sun_pef) vs (Body_DE441 - Sun_DE441)
```

| Body | Initial p50 | Initial p95 | Initial p99 | Initial max | Optimized p50 | Optimized p95 | Optimized p99 | Optimized max | Accepted | Rejected nochange | Rejected budget | p99 slack |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| EMB | 0.000091574 | 0.000272898 | 0.000456962 | 0.000788265 | 0.000101496 | 0.000296610 | 0.000388215 | 0.000547023 | 30,386 | 3 | 0 | 1 |
| Mars | 0.000110850 | 0.000269961 | 0.000359679 | 0.000664731 | 0.000132659 | 0.000236969 | 0.000278159 | 0.000368470 | 16,155 | 1 | 0 | 0 |
| Jupiter | 0.000220588 | 0.000411409 | 0.000500723 | 0.000848626 | 0.000260485 | 0.000373933 | 0.000400764 | 0.000450515 | 3,664 | 35 | 0 | 0 |
| Saturn | 0.000204516 | 0.000406845 | 0.000514163 | 0.000904270 | 0.000225251 | 0.000389573 | 0.000449595 | 0.000636241 | 3,065 | 106 | 0 | 20 |
| Uranus | 0.000232251 | 0.000429852 | 0.000523283 | 0.000852620 | 0.000277138 | 0.000390242 | 0.000420517 | 0.000480195 | 1,381 | 5 | 0 | 0 |
| Neptune | 0.000249284 | 0.000459710 | 0.000555372 | 0.000765640 | 0.000288267 | 0.000409281 | 0.000439359 | 0.000492772 | 1,099 | 10 | 0 | 0 |
| Pluto | 0.000223610 | 0.000457737 | 0.000574707 | 0.000837881 | 0.000247918 | 0.000439949 | 0.000503459 | 0.000591399 | 1,093 | 16 | 0 | 0 |

### Native vector pmax tuning metrics

These bodies are not SSB-centered relative to the final user-facing vector, so their ordinary native vector metric is the relevant pmax objective.

| Body | Metric | Initial p50 | Initial p95 | Initial p99 | Initial max | Optimized p50 | Optimized p95 | Optimized p99 | Optimized max | Accepted | Rejected nochange | Rejected budget |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Mercury | Sun->Mercury arcsec | 0.000245305 | 0.000472518 | 0.000581232 | 0.001003249 | 0.000282005 | 0.000401767 | 0.000431557 | 0.000554868 | 125,746 | 431 | 0 |
| Venus | Sun->Venus arcsec | 0.000308416 | 0.000561900 | 0.000678033 | 0.001106109 | 0.000328459 | 0.000461560 | 0.000498806 | 0.000616284 | 49,325 | 71 | 0 |
| Moon | Earth->Moon arcsec | 0.000302182 | 0.000547422 | 0.000658286 | 0.001249797 | 0.000346058 | 0.000485152 | 0.000518640 | 0.000638824 | 400,235 | 2,603 | 0 |

## Mars candidate decision

Mars had three relevant full-range candidates:

| Candidate | Size MiB | Composite p99 arcsec | Composite max arcsec | Ordinary validation p99 arcsec | Ordinary validation max arcsec | Decision |
|---|---:|---:|---:|---:|---:|---|
| q040 baseline + pmax | 2.468 | 0.000428889 | 0.000584690 | 0.000429820 | 0.000590077 | Rejected; saves only about 45 KiB vs d30. |
| q035 + pmax | 2.501 | 0.000427206 | 0.000586567 | 0.000428384 | 0.000596904 | Rejected; little accuracy gain vs q040. |
| d30/q040 + pmax | 2.512 | 0.000278159 | 0.000368470 | 0.000279050 | 0.000373265 | **Selected**; large accuracy win for about 45 KiB. |

## Earth-Sun end-to-end Moon correction check

The stricter geocenter check compared:

```text
A = (EMB_pef - Sun_pef) vs (EMB_DE441 - Sun_DE441)
B = (EMB_pef - alpha*Moon_pef - Sun_pef)
    vs (EMB_DE441 - alpha*Moon_DE441 - Sun_DE441)
```

with:

```text
alpha = 0.012150584395829191
```

Using 32 nodes per EMB segment:

| Metric | EMB-Sun A | Earth-Sun B | Absolute delta |
|---|---:|---:|---:|
| p50 arcsec | 0.000101496 | 0.000101495 | 5.06e-09 |
| p95 arcsec | 0.000296610 | 0.000296613 | 1.38e-08 |
| p99 arcsec | 0.000388215 | 0.000388223 | 1.75e-08 |
| max arcsec | 0.000547023 | 0.000547027 | 2.98e-08 |

Conclusion: Moon's error contribution to Earth-Sun is about `1e-8 arcsec`, so EMB tuning does not need to include the Moon term.

## Tooling notes

- Generic Sun-anchor optimizer: `tools/optimize_ssb_body_with_sun_pmax.py`
- EMB-specific predecessor: `tools/optimize_emb_with_sun_pmax.py`
- Deferred p99-slack revisit was important for EMB, Mars, and Saturn tail cleanup.
- Final SSB-centered outer planet tuning used `--p99-slack-abs 1e-6`.
- EMB final used the tighter default slack and accepted one deferred budget candidate.
