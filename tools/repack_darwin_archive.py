#!/usr/bin/env python3
# Copyright (C) 2026 Zynum contributors
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Recreate a trusted build archive with Darwin's Mach-O member alignment."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import stat
import subprocess
import tempfile


def member_names(listing: str) -> list[str]:
    """Reject paths and duplicate names before extracting into a flat directory."""
    names = listing.splitlines()
    seen = set()
    for name in names:
        if (
            not name
            or name in {".", ".."}
            or name.startswith(("-", "@"))
            or any(character in name for character in ("/", "\\", "\x00", "\r", "\t"))
        ):
            raise ValueError(f"Unsafe archive member name: {name!r}")
        if name in seen:
            raise ValueError(f"Duplicate archive member name: {name!r}")
        seen.add(name)
    return names


def repack(zig: str, input_archive: Path, output_archive: Path) -> None:
    executable = shutil.which(zig)
    if executable is None:
        raise FileNotFoundError(f"Zig executable not found: {zig}")
    executable = str(Path(executable).resolve())
    source = input_archive.resolve(strict=True)
    destination = output_archive.absolute()
    if not source.is_file():
        raise ValueError(f"Input is not a regular archive: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)

    def ar(*arguments: str, cwd: Path | None = None) -> str:
        return subprocess.run(
            [executable, "ar", *arguments], cwd=cwd, check=True,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        ).stdout

    members = member_names(ar("t", str(source)))
    # Keep the replacement on the destination filesystem. Existing output is
    # untouched until extraction, recreation and ordering checks all succeed.
    with tempfile.TemporaryDirectory(prefix="zynum-darwin-ar-", dir=destination.parent) as temporary:
        workspace = Path(temporary)
        extracted = workspace / "members"
        extracted.mkdir()
        ar("x", str(source), cwd=extracted)
        for name in members:
            member = extracted / name
            mode = member.lstat().st_mode
            if not stat.S_ISREG(mode):
                raise ValueError(f"Extracted archive member is not a regular file: {name!r}")
            # Zig-produced archives can store mode 000. llvm-ar preserves it
            # while extracting, so restore owner access before reading members.
            member.chmod(stat.S_IMODE(mode) | stat.S_IRUSR | stat.S_IWUSR)
        staged = workspace / "repacked.a"
        ar("--format=darwin", "rcs", str(staged), *members, cwd=extracted)
        if member_names(ar("t", str(staged))) != members:
            raise ValueError("Repacked archive member ordering differs from input")
        os.replace(staged, destination)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("zig", help="Zig executable path or command")
    parser.add_argument("input", type=Path, help="Trusted input build archive")
    parser.add_argument("output", type=Path, help="Recreated Darwin archive")
    arguments = parser.parse_args()
    try:
        repack(arguments.zig, arguments.input, arguments.output)
    except subprocess.CalledProcessError as error:
        parser.exit(1, f"Archive command failed: {error.stderr or error}\n")
    except (OSError, ValueError) as error:
        parser.exit(1, f"Cannot repack Darwin archive: {error}\n")


if __name__ == "__main__":
    main()
