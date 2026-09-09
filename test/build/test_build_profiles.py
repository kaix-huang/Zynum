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
        source = (REPOSITORY_ROOT / "build.zig").read_text(encoding="utf-8")
        for arguments in (("-Dcpu=baseline",), ("-Ddispatch=dynamic",), ("-Ddispatch=specialized",), ("-Ddispatch=specialized", "-Dcpu=baseline")):
            with self.subTest(dispatch=arguments):
                self._assert_configures(*arguments)
        self._assert_configures("-Doptimize=ReleaseSafe")
        self._assert_configures("-Ddispatch=dynamic", "-Dcpu=baseline")
        self.assertIn("target_query.cpu_features_add = .empty", source)
        self.assertIn("target_query.cpu_features_sub = .empty", source)
        self.assertIn('.default_target = .{ .cpu_model = .baseline }', source)
        self.assertIn('const explicit_cpu = b.user_input_options.contains("cpu")', source)
        self.assertIn('.auto => !explicit_cpu', source)
        self.assertIn('libraries[index].lto = .none', source)
        for runner in (
            "run_bench",
            "run_gemm_sweep",
            "run_gemm_sweep_isolated",
            "run_vector_matrix_sweep",
        ):
            with self.subTest(benchmark=runner):
                self.assertNotIn(f"{runner}.step.dependOn(b.getInstallStep())", source)
                self.assertIn(f"{runner}.addFileArg(lib.getEmittedBin())", source)
        self.assertIn("run_gemm_sweep.step.dependOn(&install_dynamic_lib.step)", source)
        self.assertIn('b.getInstallPath(.prefix, "gemm_sweep.csv")', source)
        # Standalone runs stay narrow while the default install still ships both
        # libraries and the benchmark executables.
        self.assertIn("b.getInstallStep().dependOn(install_static_lib)", source)
        for library in ("dynamic",):
            self.assertIn(
                f"b.getInstallStep().dependOn(&install_{library}_lib.step)", source
            )
        for artifact in ("bench", "gemm_sweep", "vector_matrix_sweep"):
            self.assertIn(f"b.installArtifact({artifact})", source)

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
