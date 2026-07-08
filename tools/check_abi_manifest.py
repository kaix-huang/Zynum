#!/usr/bin/env python3
# Copyright (C) 2026 Zynum contributors
# SPDX-License-Identifier: LGPL-3.0-or-later

"""Check that built Zynum BLAS libraries export every symbol in the ABI manifest."""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from pathlib import Path


def manifest_symbols(path: Path) -> list[str]:
    data = json.loads(path.read_text())
    symbols: list[str] = []
    for section in ("fortran", "cblas"):
        exports = data.get(section, {}).get("exports", [])
        symbols.extend(item["name"] for item in exports)
    return symbols


def run_nm(args: list[str]) -> set[str]:
    proc = subprocess.run(args, check=True, text=True, stdout=subprocess.PIPE)
    symbols: set[str] = set()
    for line in proc.stdout.splitlines():
        parts = line.split()
        if not parts:
            continue
        token = parts[-1]
        if token.endswith(":"):
            continue
        symbols.add(token)
        if token.startswith("_"):
            symbols.add(token[1:])
    return symbols


def dynamic_nm_args(path: Path) -> list[str]:
    if platform.system() == "Darwin":
        return ["nm", "-gU", str(path)]
    return ["nm", "-D", "--defined-only", str(path)]


def check_library(label: str, exported: set[str], expected: list[str]) -> list[str]:
    missing = [symbol for symbol in expected if symbol not in exported]
    if missing:
        preview = ", ".join(missing[:20])
        if len(missing) > 20:
            preview += f", ... ({len(missing)} total)"
        print(f"{label} is missing ABI symbols: {preview}", file=sys.stderr)
    return missing


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--dynamic", type=Path, required=True)
    parser.add_argument("--static", type=Path, required=True)
    args = parser.parse_args()

    expected = manifest_symbols(args.manifest)
    dynamic_symbols = run_nm(dynamic_nm_args(args.dynamic))
    static_symbols = run_nm(["nm", "-g", str(args.static)])

    missing_dynamic = check_library("dynamic library", dynamic_symbols, expected)
    missing_static = check_library("static library", static_symbols, expected)
    if missing_dynamic or missing_static:
        return 1

    print(f"ABI manifest check passed for {len(expected)} symbols.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
