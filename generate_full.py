#!/usr/bin/env python3
"""Generate the OPM demo set across the DE441 source range."""
from __future__ import annotations

import argparse
import concurrent.futures
import math
import subprocess
import sys
from pathlib import Path

from jplephem.spk import SPK

from opm_demo.body_configs import DEFAULT_BODY_ORDER
from opm_demo.format import CENTURY_DAYS, JD_J2000
from opm_demo.generator import source_bounds


def century_index(jd: float) -> int:
    return int((jd - JD_J2000) // CENTURY_DAYS)


def century_start(index: int) -> float:
    return JD_J2000 + index * CENTURY_DAYS


def century_dir(root: Path, index: int) -> Path:
    return root / f"c{index:+05d}"


def expected_files(directory: Path) -> list[Path]:
    return [directory / f"{body}.opm" for body in DEFAULT_BODY_ORDER]


def has_complete_outputs(directory: Path) -> bool:
    return all(path.exists() and path.stat().st_size > 0 for path in expected_files(directory))


def run_command(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)


def generate_one(args: tuple[int, float, float, Path, Path, bool, bool]) -> tuple[int, int, str]:
    index, start_jd, days, output_root, de441_path, resume, validate = args
    directory = century_dir(output_root, index)
    if resume and has_complete_outputs(directory):
        if not validate:
            return index, 0, f"SKIP c{index:+05d}: existing files"
        validation = run_command([
            sys.executable,
            "validate_opm.py",
            "--de441",
            str(de441_path),
            str(directory),
        ])
        if validation.returncode == 0:
            return index, 0, f"SKIP c{index:+05d}: existing files validated"

    directory.mkdir(parents=True, exist_ok=True)
    generation = run_command([
        sys.executable,
        "generate_range.py",
        "--de441",
        str(de441_path),
        "--all",
        "--jd-start",
        f"{start_jd:.9f}",
        "--days",
        f"{days:.9f}",
        "--output-root",
        str(directory),
    ])
    if generation.returncode != 0:
        return index, generation.returncode, generation.stdout

    if validate:
        validation = run_command([
            sys.executable,
            "validate_opm.py",
            "--de441",
            str(de441_path),
            str(directory),
        ])
        if validation.returncode != 0:
            return index, validation.returncode, generation.stdout + validation.stdout
        return index, 0, generation.stdout + validation.stdout
    return index, 0, generation.stdout


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate century-sliced OPM files across DE441 coverage")
    parser.add_argument("--de441", type=Path, required=True, help="path to DE441 BSP file")
    parser.add_argument("--output-root", type=Path, default=Path("out/small/full-opm"))
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--resume", action="store_true", help="skip completed centuries")
    parser.add_argument("--validate", action="store_true", help="validate each century after generation")
    # Generation uses the production default node oversampling from generate_range.py.
    # Keep generate_full.py focused on orchestration rather than exposing writer tuning knobs.
    parser.add_argument("--start-index", type=int, help="first J2000-relative century index to generate")
    parser.add_argument("--end-index", type=int, help="last J2000-relative century index to generate")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    with SPK.open(str(args.de441)) as spk:
        source_start, source_end = source_bounds(spk)

    first_index = int(math.ceil((source_start - JD_J2000) / CENTURY_DAYS)) if args.start_index is None else int(args.start_index)
    last_index = int(math.floor((source_end - JD_J2000) / CENTURY_DAYS)) - 1 if args.end_index is None else int(args.end_index)
    tasks = []
    for index in range(first_index, last_index + 1):
        start = century_start(index)
        end = start + CENTURY_DAYS
        if end <= source_start or start >= source_end:
            continue
        tasks.append((index, start, CENTURY_DAYS, args.output_root, args.de441, args.resume, args.validate))

    print(f"centuries={len(tasks)} range=c{first_index:+05d}..c{last_index:+05d}")
    failures = 0
    if args.jobs <= 1:
        for task in tasks:
            index, code, output = generate_one(task)
            print(output.rstrip())
            if code != 0:
                failures += 1
                print(f"FAIL c{index:+05d}: exit={code}")
    else:
        with concurrent.futures.ProcessPoolExecutor(max_workers=args.jobs) as executor:
            for index, code, output in executor.map(generate_one, tasks):
                print(output.rstrip())
                if code != 0:
                    failures += 1
                    print(f"FAIL c{index:+05d}: exit={code}")

    print(f"{('PASS' if failures == 0 else 'FAIL')}: {len(tasks) - failures}/{len(tasks)} centuries generated")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
