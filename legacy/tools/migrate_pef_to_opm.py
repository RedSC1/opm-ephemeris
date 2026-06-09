#!/usr/bin/env python3
"""Migrate legacy PEF1 files to OPM1 by patching magic and file extension.

This is a byte-level migration for already-generated artifacts whose binary layout
is otherwise unchanged. It rewrites only the first four magic bytes, recomputes the
header CRC64, preserves the payload bytes/CRC, and writes `.opm` outputs.
"""
from __future__ import annotations

import argparse
import shutil
import struct
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from opm_demo.format import FIXED_HEADER_SIZE, HEADER_CRC64_OFFSET, MAGIC, crc64_ecma

LEGACY_MAGIC = b"PEF1"
OPM_SUFFIX = ".opm"
PEF_SUFFIX = ".pef"


def iter_input_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    return sorted(path.rglob(f"*{PEF_SUFFIX}"))


def output_path_for(src: Path, *, input_root: Path, output_root: Path | None, in_place: bool) -> Path:
    if in_place:
        return src.with_suffix(OPM_SUFFIX)
    if output_root is None:
        raise ValueError("output_root is required unless in_place=True")
    if input_root.is_file():
        rel = src.with_suffix(OPM_SUFFIX).name
        return output_root / rel
    rel = src.relative_to(input_root).with_suffix(OPM_SUFFIX)
    return output_root / rel


def convert_bytes(data: bytes, src: Path) -> bytes:
    if len(data) < FIXED_HEADER_SIZE:
        raise ValueError(f"{src}: too small for fixed OPM header")
    found_magic = data[:4]
    if found_magic == MAGIC:
        converted = bytearray(data)
    elif found_magic == LEGACY_MAGIC:
        converted = bytearray(data)
        converted[:4] = MAGIC
    else:
        raise ValueError(f"{src}: expected {LEGACY_MAGIC!r} or {MAGIC!r}, found {found_magic!r}")

    # Recompute header CRC because the magic changed. Payload bytes and payload CRC
    # are unchanged.
    struct.pack_into("<Q", converted, HEADER_CRC64_OFFSET, 0)
    header_crc = crc64_ecma(bytes(converted[:FIXED_HEADER_SIZE]))
    struct.pack_into("<Q", converted, HEADER_CRC64_OFFSET, header_crc)
    return bytes(converted)


def migrate_one(src: Path, dst: Path, *, overwrite: bool, keep: bool) -> None:
    if dst.exists() and not overwrite:
        raise FileExistsError(f"{dst}: exists; pass --overwrite to replace")
    data = src.read_bytes()
    converted = convert_bytes(data, src)
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_bytes(converted)
    if not keep and src != dst:
        src.unlink()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Migrate legacy .pef/PEF1 artifacts to .opm/OPM1")
    parser.add_argument("input", type=Path, help="legacy .pef file or directory containing .pef files")
    parser.add_argument("--output-root", type=Path, help="write converted .opm files under this root")
    parser.add_argument("--in-place", action="store_true", help="write .opm siblings next to source .pef files")
    parser.add_argument("--keep", action="store_true", help="keep source .pef files when using --in-place")
    parser.add_argument("--overwrite", action="store_true", help="overwrite existing .opm outputs")
    parser.add_argument("--copy-non-pef", action="store_true", help="when using --output-root on a directory, copy non-.pef files too")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.in_place and args.output_root is not None:
        raise SystemExit("--in-place and --output-root are mutually exclusive")
    if not args.in_place and args.output_root is None:
        raise SystemExit("pass --output-root or --in-place")

    src_root = args.input
    files = iter_input_files(src_root)
    if not files:
        raise SystemExit(f"no {PEF_SUFFIX} files found under {src_root}")

    if args.copy_non_pef and args.output_root is not None and src_root.is_dir():
        for item in src_root.rglob("*"):
            if not item.is_file() or item.suffix == PEF_SUFFIX:
                continue
            dst = args.output_root / item.relative_to(src_root)
            if dst.exists() and not args.overwrite:
                continue
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(item, dst)

    converted = 0
    for src in files:
        dst = output_path_for(src, input_root=src_root, output_root=args.output_root, in_place=args.in_place)
        migrate_one(src, dst, overwrite=bool(args.overwrite), keep=bool(args.keep or not args.in_place))
        converted += 1
        print(f"{src} -> {dst}")

    print(f"converted {converted} file(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
