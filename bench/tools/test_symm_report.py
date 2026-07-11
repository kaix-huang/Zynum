#!/usr/bin/env python3
# Copyright (C) 2026 Zynum contributors
# SPDX-License-Identifier: LGPL-3.0-or-later

from __future__ import annotations

import csv
import importlib.util
import io
import os
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest import mock

TOOLS_DIR = Path(__file__).resolve().parent
REPO_ROOT = TOOLS_DIR.parents[1]


def load_tool(module_name):
    spec = importlib.util.spec_from_file_location(
        module_name, TOOLS_DIR / f"{module_name}.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


runner = load_tool("run_symm_report")
checker = load_tool("check_symm_report")


def report_row(
    library,
    *,
    routine="dsymm",
    kind="f64",
    shape="tiny",
    m=3,
    n=2,
    side="L",
    uplo="U",
    alpha=(0.75, 0.0),
    beta=(0.25, 0.0),
    metric_min=1,
    metric_median=2,
    metric_max=3,
    status="ok",
    check_status="checked-ok",
):
    order = m if side == "L" else n
    factor = 8 if kind.startswith("c") else 2
    return {
        "level": "level3",
        "routine": routine,
        "kind": kind,
        "library": library,
        "library_path": f"lib{library}.so",
        "shape": shape,
        "m": str(m),
        "n": str(n),
        "side": side,
        "uplo": uplo,
        "alpha_re": str(alpha[0]),
        "alpha_im": str(alpha[1]),
        "beta_re": str(beta[0]),
        "beta_im": str(beta[1]),
        "order": str(order),
        "lda": str(order),
        "ldb": str(m),
        "ldc": str(m),
        "reps": "2",
        "flop_count": str(factor * m * n * order),
        "best_ns": "8",
        "median_ns": "10",
        "p95_ns": "12",
        "max_ns": "12",
        "gflops": str(metric_max),
        "median_gflops": str(metric_median),
        "metric": "gflops",
        "status": status,
        "check_status": check_status,
        "check_max_abs_error": "0",
        "check_max_rel_error": "0",
        "check_samples": str(m * n),
        "check_raw_output": "",
        "process_repeats": "3",
        "successful_repeats": "3" if status == "ok" else "2",
        "metric_min": str(metric_min),
        "metric_median": str(metric_median),
        "metric_max": str(metric_max),
        "metric_samples": f"{metric_min},{metric_median},{metric_max}",
    }


class SymmRunnerTests(unittest.TestCase):
    def test_default_cases_cover_broad_shapes_routines_sides_and_uplos(self):
        args = runner.parse_args(["--csv", os.devnull])
        cases = runner.requested_cases(args)
        self.assertEqual(args.process_repeats, 3)
        self.assertEqual(len(cases), len(runner.DEFAULT_SHAPES) * 6 * 2 * 2)
        self.assertEqual({case.routine.name for case in cases}, set(runner.ROUTINES))
        self.assertEqual({case.shape.name for case in cases}, {
            "square128",
            "tall512x128",
            "wide128x512",
        })
        self.assertEqual({case.side for case in cases}, {"L", "R"})
        self.assertEqual({case.uplo for case in cases}, {"U", "L"})
        for case in cases:
            alpha = runner.parse_scalar(case.alpha)
            beta = runner.parse_scalar(case.beta)
            if case.routine.complex_scalars:
                self.assertNotEqual(alpha[1], 0)
                self.assertNotEqual(beta[1], 0)
            else:
                self.assertEqual(alpha[1], 0)
                self.assertEqual(beta[1], 0)

    def test_explicit_complex_scalars_are_filtered_from_real_routines(self):
        args = runner.parse_args(
            [
                "--csv",
                os.devnull,
                "--routine",
                "ssymm",
                "--routine",
                "zhemm",
                "--shape",
                "tiny:3:2",
                "--side",
                "R",
                "--uplo",
                "L",
                "--alpha",
                "0.5",
                "--alpha",
                "0.5,0.125",
                "--beta",
                "-0.25",
                "--beta",
                "-0.25,0.0625",
            ]
        )
        cases = runner.requested_cases(args)
        real_cases = [case for case in cases if case.routine.name == "ssymm"]
        complex_cases = [case for case in cases if case.routine.name == "zhemm"]
        self.assertEqual(len(real_cases), 1)
        self.assertEqual(len(complex_cases), 4)
        self.assertEqual(real_cases[0].alpha, "0.5")
        self.assertEqual(real_cases[0].beta, "-0.25")

    def test_case_command_forwards_complete_parameters(self):
        args = runner.parse_args(
            ["--csv", os.devnull, "--probe", "probe", "--reps", "7"]
        )
        case = runner.SymmCase(
            runner.ROUTINES["zhemm"],
            runner.Shape("wide", 64, 513),
            "R",
            "L",
            "0.75,-0.125",
            "0.25,0.0625",
        )
        command = runner.case_command(args, "MKL", "libmkl_rt.so", case)
        pairs = dict(zip(command[1::2], command[2::2]))
        self.assertEqual(pairs["--blas"], "libmkl_rt.so")
        self.assertEqual(pairs["--library"], "MKL")
        self.assertEqual(pairs["--routine"], "zhemm")
        self.assertEqual(pairs["--shape"], "wide")
        self.assertEqual(pairs["--m"], "64")
        self.assertEqual(pairs["--n"], "513")
        self.assertEqual(pairs["--side"], "R")
        self.assertEqual(pairs["--uplo"], "L")
        self.assertEqual(pairs["--alpha"], "0.75,-0.125")
        self.assertEqual(pairs["--beta"], "0.25,0.0625")
        self.assertEqual(pairs["--reps"], "7")

    @mock.patch.object(runner.subprocess, "run")
    def test_probe_failure_becomes_parameterized_error_row(self, run):
        run.return_value = subprocess.CompletedProcess(
            ["probe"], 1, stdout="", stderr="missing symbol"
        )
        args = runner.parse_args(["--csv", os.devnull, "--probe", "probe"])
        case = runner.SymmCase(
            runner.ROUTINES["dsymm"],
            runner.Shape("rect", 7, 3),
            "R",
            "U",
            "0.75",
            "0.25",
        )
        row = runner.run_one_process(args, "TestBLAS", "libblas.so", case)
        self.assertEqual(row["status"], "error")
        self.assertEqual(row["routine"], "dsymm")
        self.assertEqual(row["m"], "7")
        self.assertEqual(row["n"], "3")
        self.assertEqual(row["order"], "3")
        self.assertIn("missing symbol", row["check_raw_output"])

    def test_explicit_missing_path_is_checked_without_loading_blas(self):
        self.assertFalse(runner.library_available("/not/a/real/libblas.so"))
        self.assertTrue(runner.library_available("libblas.so"))
        if sys.platform == "darwin":
            self.assertTrue(runner.library_available(runner.DEFAULT_ACCELERATE))


class SymmAggregationTests(unittest.TestCase):
    def test_process_median_uses_probe_median_gflops(self):
        rows = []
        for median_value, best_value in ((2, 20), (4, 5), (3, 7)):
            row = report_row(
                "Zynum", metric_median=median_value, metric_max=best_value
            )
            row["median_gflops"] = str(median_value)
            row["gflops"] = str(best_value)
            rows.append(row)
        aggregate = runner.aggregate_repeats(rows)
        self.assertEqual(aggregate["process_repeats"], 3)
        self.assertEqual(aggregate["successful_repeats"], 3)
        self.assertEqual(aggregate["metric_min"], "2")
        self.assertEqual(aggregate["metric_median"], "3")
        self.assertEqual(aggregate["metric_max"], "4")
        self.assertEqual(aggregate["metric_samples"], "2,4,3")
        self.assertEqual(aggregate["gflops"], "20")

    def test_any_bad_repeat_contaminates_aggregate(self):
        good = report_row("Zynum")
        bad = report_row(
            "Zynum", status="correctness_failed", check_status="correctness_failed"
        )
        bad["check_max_abs_error"] = "4.5"
        bad["check_raw_output"] = "reference tolerance exceeded"
        aggregate = runner.aggregate_repeats([good, bad])
        self.assertEqual(aggregate["successful_repeats"], 1)
        self.assertEqual(aggregate["status"], "correctness_failed")
        self.assertEqual(aggregate["check_status"], "correctness_failed")
        self.assertEqual(aggregate["check_max_abs_error"], "4.5")
        self.assertIn("repeat=2", aggregate["check_raw_output"])


class SymmCheckerTests(unittest.TestCase):
    def run_checker(self, rows, *extra_args):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "symm.csv"
            with path.open("w", newline="") as file:
                writer = csv.DictWriter(file, fieldnames=runner.CSV_FIELDNAMES)
                writer.writeheader()
                writer.writerows(rows)
            stdout = io.StringIO()
            stderr = io.StringIO()
            with redirect_stdout(stdout), redirect_stderr(stderr):
                result = checker.main(
                    [str(path), "--comparator", "Reference", *extra_args]
                )
        return result, stdout.getvalue(), stderr.getvalue()

    def test_default_checker_uses_fresh_process_median(self):
        rows = [
            report_row("Zynum", metric_min=1, metric_median=2, metric_max=10),
            report_row("Reference", metric_min=1, metric_median=4, metric_max=8),
        ]
        median_result, median_stdout, median_stderr = self.run_checker(rows)
        best_result, best_stdout, best_stderr = self.run_checker(
            rows, "--stat", "best"
        )
        self.assertEqual(median_result, 1, median_stderr)
        self.assertIn("stat=median", median_stdout)
        self.assertEqual(best_result, 0, best_stderr)
        self.assertIn("stat=best", best_stdout)

    def test_checker_accepts_separate_negative_complex_scalar_filter(self):
        args = checker.parse_args(
            ["report.csv", "--beta", "-0.25,0.0625"]
        )
        self.assertEqual(args.beta, ["-0.25,0.0625"])

    def test_complete_parameters_keep_groups_separate(self):
        rows = [
            report_row("Zynum", side="L"),
            report_row("Reference", side="L"),
            report_row(
                "Zynum", side="R", uplo="L", alpha=(0.5, 0.0), beta=(0.0, 0.0)
            ),
        ]
        result, stdout, stderr = self.run_checker(rows)
        self.assertEqual(result, 1, stderr)
        self.assertIn("checked=1 passed=1 failed=0 missing=1", stdout)
        self.assertIn("side=R", stdout)
        self.assertIn("uplo=L", stdout)
        self.assertIn("alpha=0.5,0.0", stdout)

    def test_bad_zynum_correctness_is_not_performance_evidence(self):
        rows = [
            report_row(
                "Zynum",
                status="correctness_failed",
                check_status="correctness_failed",
            ),
            report_row("Reference"),
        ]
        result, stdout, stderr = self.run_checker(rows)
        self.assertEqual(result, 2, stdout)
        self.assertIn("not eligible", stderr)


class SymmProbeIntegrationTests(unittest.TestCase):
    @unittest.skipUnless(
        (REPO_ROOT / "zig-out/bin/symm-probe").is_file()
        and (REPO_ROOT / runner.default_zynum_blas()).is_file(),
        "SYMM probe and Zynum shared library have not been built",
    )
    def test_all_six_routines_sides_and_uplos_pass_full_reference_checks(self):
        probe = REPO_ROOT / "zig-out/bin/symm-probe"
        library = REPO_ROOT / runner.default_zynum_blas()
        for routine, spec in runner.ROUTINES.items():
            alpha = runner.COMPLEX_ALPHA if spec.complex_scalars else runner.REAL_ALPHA
            beta = runner.COMPLEX_BETA if spec.complex_scalars else runner.REAL_BETA
            for side in ("L", "R"):
                for uplo in ("U", "L"):
                    with self.subTest(routine=routine, side=side, uplo=uplo):
                        result = subprocess.run(
                            [
                                str(probe),
                                "--blas",
                                str(library),
                                "--library",
                                "Zynum",
                                "--routine",
                                routine,
                                "--shape",
                                "tiny",
                                "--m",
                                "3",
                                "--n",
                                "2",
                                "--side",
                                side,
                                "--uplo",
                                uplo,
                                "--alpha",
                                alpha,
                                "--beta",
                                beta,
                                "--reps",
                                "1",
                            ],
                            cwd=REPO_ROOT,
                            capture_output=True,
                            text=True,
                            check=False,
                        )
                        self.assertEqual(result.returncode, 0, result.stderr)
                        rows = list(csv.DictReader(result.stdout.splitlines()))
                        self.assertEqual(len(rows), 1)
                        self.assertEqual(rows[0]["routine"], routine)
                        self.assertEqual(rows[0]["status"], "ok", rows[0])
                        self.assertEqual(rows[0]["check_status"], "checked-ok")
                        self.assertEqual(rows[0]["check_samples"], "6")


if __name__ == "__main__":
    unittest.main()
