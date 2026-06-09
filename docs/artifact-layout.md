# OPM artifact layout

Generated files are kept under `out/` and are ignored by git. The current active layout is range-based OPM generation; older body-packed and tuning artifacts are historical and live under `legacy/`.

## Current active outputs

```text
out/small/j2000-opm/       # one J2000-century OPM per body
out/small/full-opm/        # full source range split by safe full centuries
out/opm600/<shard>/        # recommended 600-year shard package
out/opm600/<shard>/.raw/   # raw intermediate files when --polish is used
out/opm600/<shard>/logs/   # polish and validation logs
```

Use `generate_range.py` directly for arbitrary ranges and 600-year shards:

```bash
python3 generate_range.py \
  --de441 /path/to/de441.bsp \
  --all \
  --jd-start 2451545 \
  --days 219150 \
  --output-root out/opm600/c+0000 \
  --polish \
  --validate
```

`generate_full.py` remains as a convenience orchestrator for safe full-century batches.

## Legacy area

Historical entrypoints, body-packed experiments, and superseded tuning notes are under:

```text
legacy/
legacy/tools/
legacy/docs/
legacy/scripts/
```

Do not use `legacy/` files as current guidance unless explicitly reproducing an old experiment.
