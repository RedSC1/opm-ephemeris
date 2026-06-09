# OPM artifact size summary

This note records the **current canonical size numbers** for the OPM demo artifacts. Generated files under `out/` are local experimental artifacts and are ignored by git, so treat these numbers as run results for the paper/demo rather than source-controlled fixtures.

For final accuracy/tuning details, use `docs/final-body-packed-tuning-results.md` as the source of truth.

## Current final tuned full-range set

These are the final selected body-packed candidates from the full-range tuning pass.

| Body | Final candidate | Segments | Size MiB | Notes |
|---|---|---:|---:|---|
| Sun | `out/body-packed/tuning/sun/q010-pmax/sun.opm` | 61,666 | 4.874 | SSB->Sun anchor; optimized with km metric. |
| Mercury | `out/body-packed/tuning/mercury/d26-q032-global-tail-full/mercury.opm` | 126,177 | 14.246 | Sun->Mercury native vector. |
| Venus | `out/body-packed/tuning/venus/q08-global-tail-full/venus.opm` | 49,396 | 5.566 | Sun->Venus native vector. |
| EMB | `out/body-packed/tuning/emb/sun-q010-pmax-auto-revisit/emb.opm` | 30,389 | 4.739 | Tuned with Sun-anchor composite metric. |
| Mars | `out/body-packed/tuning/mars/sun-q010-pmax-d30-auto-revisit-slack1e-6/mars.opm` | 16,156 | 2.512 | d30 selected for much lower tail. |
| Jupiter | `out/body-packed/tuning/jupiter/sun-q010-pmax-auto-revisit-slack1e-6/jupiter.opm` | 3,699 | 0.452 | Tuned with Sun-anchor composite metric. |
| Saturn | `out/body-packed/tuning/saturn/sun-q010-pmax-auto-revisit-slack1e-6/saturn.opm` | 3,171 | 0.351 | Tuned with Sun-anchor composite metric. |
| Uranus | `out/body-packed/tuning/uranus/sun-q010-pmax-auto-revisit-slack1e-6/uranus.opm` | 1,386 | 0.143 | Tuned with Sun-anchor composite metric. |
| Neptune | `out/body-packed/tuning/neptune/sun-q010-pmax-auto-revisit-slack1e-6/neptune.opm` | 1,109 | 0.081 | Tuned with Sun-anchor composite metric. |
| Pluto | `out/body-packed/tuning/pluto/sun-q010-pmax-auto-revisit-slack1e-6/pluto.opm` | 1,109 | 0.092 | Tuned with Sun-anchor composite metric. |
| Moon | `out/body-packed/tuning/moon/q034-global-tail-full/moon.opm` | 402,838 | 60.894 | Earth->Moon native vector. |

Totals:

```text
All final files, including Sun: 93.951 MiB
All final files, excluding Sun: 89.077 MiB
```

The Moon still dominates the final full-range set:

```text
Moon final size       = 60.894 MiB
Moon share incl. Sun  ≈ 64.8%
Moon share excl. Sun  ≈ 68.4%
```

## Historical pre-final full-source-safe set

The old generator-default/body-packed full-source-safe set was:

```text
out/body-packed/full
```

| Body | Old size MiB | Final size MiB | Change MiB | Notes |
|---|---:|---:|---:|---|
| Sun | 4.874 | 4.874 | 0.000 | Same final Sun path/size after pmax. |
| Mercury | 14.456 | 14.246 | -0.210 | Final d26/q032 global-tail candidate. |
| Venus | 5.725 | 5.566 | -0.159 | Final q08 global-tail candidate. |
| EMB | 4.739 | 4.739 | 0.000 | Same size, pmax tuned. |
| Mars | 2.512 | 2.512 | 0.000 | Same size class, d30 selected for accuracy. |
| Jupiter | 0.452 | 0.452 | 0.000 | Same size, pmax tuned. |
| Saturn | 0.351 | 0.351 | 0.000 | Same size, pmax tuned. |
| Uranus | 0.143 | 0.143 | 0.000 | Same size, pmax tuned. |
| Neptune | 0.081 | 0.081 | 0.000 | Same size, pmax tuned. |
| Pluto | 0.092 | 0.092 | 0.000 | Same size, pmax tuned. |
| Moon | 62.574 | 60.894 | -1.680 | Final q034 global-tail candidate. |

```text
Old full-source-safe total: 96.001 MiB
Final tuned total:         93.951 MiB
Delta:                     -2.050 MiB
```

Keep the old `out/body-packed/full` numbers only as a historical baseline. They are no longer the recommended final package numbers.

## Small and sample artifacts

These older sample totals are useful for layout/overhead intuition, but they are not the current final tuned full-range package.

| Package | Root | Files | Total KiB | Total MiB | Notes |
|---|---|---:|---:|---:|---|
| Small J2000 century | `out/small/j2000-opm` | 11 | 286.490 | 0.280 | One one-century OPM per body. |
| Small 7-century, century-sliced | `out/small/full-opm-7c` | 77 | 2008.890 | 1.962 | Seven century directories × 11 bodies. |
| Body-packed 7-century | `out/body-packed/seven-century` | 11 | 1952.250 | 1.906 | One 7-century OPM per body. |

For that seven-century sample:

```text
small 7-century total       = 2008.890 KiB
body-packed 7-century total = 1952.250 KiB
delta                       = -56.640 KiB (-2.82%)
```

This is only one profile. Century-sliced, body-packed, and future 600-year sharding optimize for different distribution/access patterns.

## Benchmark and validation commands

Representative validation/benchmark commands:

```bash
python3 validate_opm.py \
  --de441 /path/to/de441.bsp \
  out/body-packed/tuning

python3 benchmark_read.py \
  --de441 /path/to/de441.bsp \
  --opm-root out/body-packed/tuning \
  --samples 100000
```

If benchmarking a generated 600-year package, point `--opm-root` at that package root instead.
