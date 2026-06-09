#!/usr/bin/env python3
"""Verify no-size-increase pmax polishing on seven single-century OPM samples.

The sample pattern uses the DE441 full-coverage boundary centuries, J2000,
and the remaining four centuries rounded from an even spacing across the full
DE441 full-coverage century-index range.  Optimization may shrink the width
table/file size, but must not increase quantization, degree, or payload width.
"""
from __future__ import annotations

import argparse
import csv
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
from jplephem.spk import SPK

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import opm_demo.moon_model as moon_proto  # noqa: E402
import opm_demo.orbit_model as proto  # noqa: E402
from opm_demo.body_configs import DEFAULT_BODY_ORDER  # noqa: E402
from opm_demo.format import CENTURY_DAYS, JD_J2000  # noqa: E402
from opm_demo.generator import source_bounds  # noqa: E402
from opm_demo import validator  # noqa: E402

NATIVE_BODIES = {"mercury", "venus", "moon"}
SSB_BODIES = {"emb", "mars", "jupiter", "saturn", "uranus", "neptune", "pluto"}
PERCENTILE_RE = re.compile(
    r"p50=(?P<p50>[0-9.eE+-]+)\s+p95=(?P<p95>[0-9.eE+-]+)\s+"
    r"p99=(?P<p99>[0-9.eE+-]+).*?max=(?P<max>[0-9.eE+-]+)"
)
ACCEPT_RE = re.compile(r"accepted=(?P<accepted>[0-9,]+); rejected_nochange=(?P<nochange>[0-9,]+); rejected_budget=(?P<budget>[0-9,]+)")


def run(cmd: list[str], *, log_path: Path | None = None) -> str:
    proc = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text(proc.stdout, encoding="utf-8")
    if proc.returncode != 0:
        if log_path is not None:
            raise RuntimeError(f"command failed ({proc.returncode}); see {log_path}: {' '.join(cmd)}")
        raise RuntimeError(f"command failed ({proc.returncode}): {' '.join(cmd)}\n{proc.stdout}")
    return proc.stdout


def century_index(jd: float) -> int:
    return int((jd - JD_J2000) // CENTURY_DAYS)


def century_start(index: int) -> float:
    return JD_J2000 + index * CENTURY_DAYS


def sample_indices(de441: Path) -> list[int]:
    with SPK.open(str(de441)) as spk:
        source_start, source_end = source_bounds(spk)
    # Use whole J2000-aligned centuries fully inside the DE441 coverage.
    # The partial overlap centuries at the raw source boundaries cannot be fit with
    # expanded Chebyshev nodes because they would sample outside the BSP coverage.
    first = int(np.ceil((source_start - JD_J2000) / CENTURY_DAYS))
    last = int(np.floor((source_end - JD_J2000) / CENTURY_DAYS)) - 1
    indices = [int(round(x)) for x in np.linspace(first, last, 7)]
    # Guard the intended pattern explicitly: both full-coverage boundary
    # centuries plus J2000.
    indices[0] = first
    indices[-1] = last
    if 0 not in indices:
        indices[len(indices) // 2] = 0
    return indices


def complete_century_dir(path: Path) -> bool:
    return all((path / f"{body}.opm").exists() and (path / f"{body}.opm").stat().st_size > 0 for body in DEFAULT_BODY_ORDER)


def parse_optimizer_log(text: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for line in text.splitlines():
        if line.startswith("initial global:"):
            m = PERCENTILE_RE.search(line)
            if m:
                for k, v in m.groupdict().items():
                    out[f"initial_{k}"] = v
        elif line.startswith("optimized global:"):
            m = PERCENTILE_RE.search(line)
            if m:
                for k, v in m.groupdict().items():
                    out[f"optimized_{k}"] = v
        elif line.startswith("accepted="):
            m = ACCEPT_RE.search(line)
            if m:
                out.update({k: v.replace(",", "") for k, v in m.groupdict().items()})
    return out


def optimize_body(args: argparse.Namespace, century: str, body: str, input_dir: Path, output_dir: Path, log_dir: Path) -> dict[str, str]:
    input_opm = input_dir / f"{body}.opm"
    output_opm = output_dir / f"{body}.opm"
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / century / f"{body}.log"
    if body == "sun":
        cmd = [
            sys.executable,
            "tools/optimize_opm_global_tail.py",
            str(input_opm),
            "--de441",
            str(args.de441),
            "--nodes-per-segment",
            str(args.nodes_per_segment),
            "--jobs",
            str(args.jobs),
            "--limit",
            "0",
            "--progress-every",
            "0",
            "--error-metric",
            "km",
            "--output",
            str(output_opm),
        ]
    elif body in NATIVE_BODIES:
        cmd = [
            sys.executable,
            "tools/optimize_opm_native_guarded_pmax.py",
            str(input_opm),
            "--de441",
            str(args.de441),
            "--nodes-per-segment",
            str(args.nodes_per_segment),
            "--jobs",
            str(args.jobs),
            "--limit",
            "0",
            "--progress-every",
            "0",
            "--output",
            str(output_opm),
        ]
    elif body in SSB_BODIES:
        cmd = [
            sys.executable,
            "tools/optimize_opm_ssb_sun_anchor_pmax.py",
            str(input_opm),
            "--sun-opm",
            str(output_dir / "sun.opm"),
            "--de441",
            str(args.de441),
            "--nodes-per-segment",
            str(args.nodes_per_segment),
            "--jobs",
            str(args.jobs),
            "--limit",
            "0",
            "--progress-every",
            "0",
            "--p99-slack-abs",
            str(args.p99_slack_abs),
            "--output",
            str(output_opm),
        ]
    else:
        raise ValueError(f"unknown body {body}")
    text = run(cmd, log_path=log_path)
    stats = parse_optimizer_log(text)
    stats.update({"century": century, "body": body, "input": str(input_opm), "output": str(output_opm), "log": str(log_path)})
    return stats


def validate_file(spk: SPK, path: Path, nodes_per_segment: int) -> dict[str, str]:
    body, segments, p50, p95, p99, max_err, status = validator.validate_one(spk, path, nodes_per_segment)
    return {
        "segments": str(segments),
        "validate_p50": f"{p50:.12g}",
        "validate_p95": f"{p95:.12g}",
        "validate_p99": f"{p99:.12g}",
        "validate_max": f"{max_err:.12g}",
        "status": status,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--de441", type=Path, required=True)
    parser.add_argument("--input-root", type=Path, default=Path("out/small/full-opm-7c-boundary"))
    parser.add_argument("--output-root", type=Path, default=Path("out/small/full-opm-7c-boundary-pmax"))
    parser.add_argument("--summary", type=Path, default=Path("docs/century-pmax-polish-verification.md"))
    parser.add_argument("--nodes-per-segment", type=int, default=32)
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--p99-slack-abs", type=float, default=1e-6)
    parser.add_argument("--resume-generate", action="store_true")
    args = parser.parse_args()

    proto.set_de441_path(args.de441)
    moon_proto.set_de441_path(args.de441)

    indices = sample_indices(args.de441)
    log_dir = args.output_root / "logs"
    rows: list[dict[str, str]] = []

    print(f"sample_indices={','.join(f'c{i:+05d}' for i in indices)}", flush=True)
    with SPK.open(str(args.de441)) as spk:
        source_start, source_end = source_bounds(spk)

    for index in indices:
        century = f"c{index:+05d}"
        start = century_start(index)
        days = CENTURY_DAYS
        if start < source_start or start + days > source_end:
            raise RuntimeError(f"sample century {century} is not fully inside DE441 coverage")
        input_dir = args.input_root / century
        output_dir = args.output_root / century
        if not (args.resume_generate and complete_century_dir(input_dir)):
            print(f"generate {century} jd_start={start:.9f} days={days:.9f}", flush=True)
            run([
                sys.executable,
                "generate_range.py",
                "--de441",
                str(args.de441),
                "--all",
                "--jd-start",
                f"{start:.9f}",
                "--days",
                f"{days:.9f}",
                "--output-root",
                str(input_dir),
            ], log_path=log_dir / century / "generate.log")
        else:
            print(f"generate {century}: reuse existing", flush=True)

        # Sun first; SSB bodies use the same-century polished Sun as anchor.
        order = ["sun"] + [b for b in DEFAULT_BODY_ORDER if b != "sun"]
        for body in order:
            print(f"optimize {century} {body}", flush=True)
            row = optimize_body(args, century, body, input_dir, output_dir, log_dir)
            row["size_before_bytes"] = str((input_dir / f"{body}.opm").stat().st_size)
            row["size_after_bytes"] = str((output_dir / f"{body}.opm").stat().st_size)
            row["size_same"] = str(row["size_before_bytes"] == row["size_after_bytes"])
            rows.append(row)

    print("validate before/after", flush=True)
    with SPK.open(str(args.de441)) as spk:
        for row in rows:
            before = validate_file(spk, Path(row["input"]), args.nodes_per_segment)
            after = validate_file(spk, Path(row["output"]), args.nodes_per_segment)
            row.update({f"before_{k}": v for k, v in before.items()})
            row.update({f"after_{k}": v for k, v in after.items()})

    args.output_root.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_root / "century-pmax-polish-results.csv"
    fieldnames = sorted({k for row in rows for k in row})
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    args.summary.parent.mkdir(parents=True, exist_ok=True)
    total_before = sum(int(r["size_before_bytes"]) for r in rows)
    total_after = sum(int(r["size_after_bytes"]) for r in rows)
    pass_count = sum(1 for r in rows if r.get("after_status") in {"PASS", "DIAG"})
    size_same_count = sum(1 for r in rows if r["size_same"] == "True")
    improved_max = sum(1 for r in rows if float(r.get("after_validate_max", "nan")) <= float(r.get("before_validate_max", "nan")))

    lines = [
        "# Seven-century pmax polish verification",
        "",
        "This verifies a writer-side pmax polish pass on seven single-century OPM samples without changing quantization or increasing intended bit widths. Width-table shrinkage is allowed.",
        "",
        "Sample centuries:",
        "",
        "```text",
        ", ".join(f"c{i:+05d}" for i in indices),
        "```",
        "",
        "The sample uses the DE441 full-coverage boundary centuries, J2000, and the remaining centuries rounded from an even spacing across the full DE441 full-coverage century-index range.",
        "",
        "## Summary",
        "",
        f"- Rows: {len(rows)} files = 7 centuries × 11 bodies",
        f"- Output root: `{args.output_root}`",
        f"- CSV: `{csv_path}`",
        f"- Validation pass/diag rows: {pass_count}/{len(rows)}",
        f"- Exact file-size matches: {size_same_count}/{len(rows)}",
        f"- Direct validation max non-increased rows: {improved_max}/{len(rows)}",
        f"- Total size before: {total_before / 1024:.3f} KiB",
        f"- Total size after: {total_after / 1024:.3f} KiB",
        "",
        "## Per-file validation before/after",
        "",
        "| Century | Body | Size before KiB | Size after KiB | Same size | Before p99 | Before max | After p99 | After max | Status | Objective p99 before -> after | Objective max before -> after |",
        "|---|---|---:|---:|---|---:|---:|---:|---:|---|---:|---:|",
    ]
    for r in rows:
        lines.append(
            "| {century} | {body} | {size_before:.3f} | {size_after:.3f} | {same} | {bp99} | {bmax} | {ap99} | {amax} | {status} | {ip99} -> {op99} | {imax} -> {omax} |".format(
                century=r["century"],
                body=r["body"],
                size_before=int(r["size_before_bytes"]) / 1024,
                size_after=int(r["size_after_bytes"]) / 1024,
                same=r["size_same"],
                bp99=r.get("before_validate_p99", ""),
                bmax=r.get("before_validate_max", ""),
                ap99=r.get("after_validate_p99", ""),
                amax=r.get("after_validate_max", ""),
                status=r.get("after_status", ""),
                ip99=r.get("initial_p99", ""),
                op99=r.get("optimized_p99", ""),
                imax=r.get("initial_max", ""),
                omax=r.get("optimized_max", ""),
            )
        )
    lines.append("")
    args.summary.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {args.summary}")
    print(f"wrote {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
