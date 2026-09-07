# Copyright (C) 2026 Zynum contributors
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Build-graph configuration tests for mutually exclusive experimental profiles."""

from __future__ import annotations

import itertools
import shutil
import subprocess
import unittest
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
PROFILE_FLAGS = (
    "structured-object-candidates",
    "structured-object-baseline",
    "level1-sve-candidates",
    "level1-fixed-candidates",
    "level2-fixed-candidates",
    "level2-width-candidates",
)
STRUCTURED_CONFLICT = (
    "structured-object-candidates and structured-object-baseline are mutually exclusive"
)
PROFILE_CONFLICT = "experimental profile flags are mutually exclusive"


@unittest.skipUnless(shutil.which("zig"), "Zig is required for build graph checks")
class BuildProfileTests(unittest.TestCase):
    def _configure(self, *arguments: str) -> subprocess.CompletedProcess[str]:
        zig = shutil.which("zig")
        assert zig is not None
        # --help evaluates build.zig without compiling or installing artifacts.
        return subprocess.run(
            [zig, "build", "--help", *arguments],
            cwd=REPOSITORY_ROOT,
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )

    def _assert_configures(self, *arguments: str) -> None:
        result = self._configure(*arguments)
        self.assertEqual(0, result.returncode, result.stdout + result.stderr)
        self.assertIn("install-libraries", result.stdout)

    def test_default_and_explicitly_disabled_profiles(self) -> None:
        self._assert_configures()
        self._assert_configures(*(f"-D{flag}=false" for flag in PROFILE_FLAGS))

    def test_each_single_profile(self) -> None:
        for selected in PROFILE_FLAGS:
            with self.subTest(profile=selected, others="omitted"):
                self._assert_configures(f"-D{selected}=true")
            with self.subTest(profile=selected, others="false"):
                self._assert_configures(
                    *(f"-D{flag}={'true' if flag == selected else 'false'}"
                      for flag in PROFILE_FLAGS)
                )

    def test_all_conflicting_combinations(self) -> None:
        for size in range(2, len(PROFILE_FLAGS) + 1):
            for flags in itertools.combinations(PROFILE_FLAGS, size):
                with self.subTest(profiles=flags):
                    result = self._configure(*(f"-D{flag}=true" for flag in flags))
                    output = result.stdout + result.stderr
                    self.assertNotEqual(0, result.returncode, output)
                    expected = (
                        STRUCTURED_CONFLICT
                        if set(PROFILE_FLAGS[:2]).issubset(flags)
                        else PROFILE_CONFLICT
                    )
                    self.assertIn(expected, output)
                    for flag in flags:
                        if expected == PROFILE_CONFLICT:
                            self.assertIn(f"-D{flag}", output)

    def test_non_profile_controls_remain_composable(self) -> None:
        for selected in (None, *PROFILE_FLAGS):
            with self.subTest(profile=selected):
                arguments = [
                    "-Dtarget=aarch64-macos",
                    "-Dapple-amx=true",
                    "-Dlevel2-compact-triangular-baseline=true",
                ]
                if selected is not None:
                    arguments.append(f"-D{selected}=true")
                self._assert_configures(*arguments)


if __name__ == "__main__":
    unittest.main()
