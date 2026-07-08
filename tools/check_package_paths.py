#!/usr/bin/env python3
# Copyright (C) 2026 Zynum contributors
# SPDX-License-Identifier: LGPL-3.0-or-later

"""Validate build.zig.zon package paths and optionally create a source archive."""

from __future__ import annotations

import argparse
import re
import sys
import tarfile
from pathlib import Path

FORBIDDEN_PREFIXES = (
    ".git",
    ".local-docs",
    ".zig-cache",
    "zig-out",
)


def package_paths(zon_path: Path) -> list[str]:
    paths: list[str] = []
    in_paths = False
    for line in zon_path.read_text().splitlines():
        if ".paths" in line and ".{" in line:
            in_paths = True
            continue
        if not in_paths:
            continue
        if "}" in line:
            break
        match = re.search(r'"([^"]+)"', line)
        if match:
            paths.append(match.group(1))
    if not paths:
        raise ValueError("no .paths entries found in build.zig.zon")
    return paths


def validate_path(root: Path, rel: str) -> None:
    if rel.startswith("/") or ".." in Path(rel).parts:
        raise ValueError(f"invalid package path: {rel}")
    if rel.startswith(FORBIDDEN_PREFIXES):
        raise ValueError(f"forbidden package path: {rel}")
    if not (root / rel).exists():
        raise FileNotFoundError(f"package path does not exist: {rel}")


def add_archive_path(tar: tarfile.TarFile, root: Path, rel: str) -> None:
    path = root / rel
    if path.is_dir():
        for child in sorted(path.rglob("*")):
            if child.is_file():
                tar.add(child, arcname=str(child.relative_to(root)))
    else:
        tar.add(path, arcname=rel)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--archive", type=Path)
    args = parser.parse_args()

    root = args.root.resolve()
    paths = package_paths(root / "build.zig.zon")
    for rel in paths:
        validate_path(root, rel)

    if args.archive:
        args.archive.parent.mkdir(parents=True, exist_ok=True)
        with tarfile.open(args.archive, "w:gz") as tar:
            for rel in paths:
                add_archive_path(tar, root, rel)

    print(f"checked {len(paths)} package paths")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
