#!/usr/bin/env python3
# Copyright (C) 2026 Zynum contributors
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Run six bounded, isolated native lifecycle checks; not inventory enumeration."""
import os
from pathlib import Path
import platform
import subprocess
import sys

CASES = (
    "ordinary-persistent", "persistent-ordinary", "ordinary-only", "persistent-only",
    "restart-ordinary-persistent", "restart-persistent-ordinary",
)


def main():
    if len(sys.argv) != 2:
        raise SystemExit("usage: run_blas_io_lifecycle.py PROBE")
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        raise SystemExit("lifecycle regression requires native macOS AArch64")
    probe = Path(sys.argv[1]).resolve(strict=True)
    env = dict(os.environ, ZYNUM_MAXIMUM_THREADS="2")
    for case in CASES:
        try:
            result = subprocess.run([str(probe), case], env=env, capture_output=True, text=True, timeout=15)
        except subprocess.TimeoutExpired as exc:
            raise SystemExit(f"lifecycle {case} timed out after 15 seconds") from exc
        if result.returncode != 0:
            sys.stderr.write(result.stderr)
            raise SystemExit(f"lifecycle {case} failed with exit {result.returncode}")
        restores = 2 if case.startswith("restart-") else 1
        expected = f"ZYNUM-RUNTIME-LIFECYCLE-V1 case={case} restores={restores} status=ok\n"
        if result.stdout != expected or result.stderr:
            raise SystemExit(f"lifecycle {case} returned an unexpected result")
        sys.stdout.write(result.stdout)


if __name__ == "__main__":
    main()
