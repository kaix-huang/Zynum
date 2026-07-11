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


runner = load_tool("run_triangular_matrix_report")
checker = load_tool("check_triangular_matrix_report")


def integration_blas():
    if sys.platform == "darwin":
        return Path(runner.DEFAULT_ACCELERATE)
    return REPO_ROOT / runner.default_zynum_blas()


def report_row(
    library,
    *,
    routine="dtrsm",
    family="trsm",
    kind="f64",
    shape="tiny",
    m=3,
    n=2,
    side="L",
    uplo="U",
    trans="N",
    diag="N",
    alpha=(0.75, 0.0),
    metric_min=1,
    metric_median=2,
    metric_max=3,
    status="ok",
    check_status="checked-ok",
):
    order = m if side == "L" else n
    factor = 4 if kind.startswith("c") else 1
    return {
        "level": "level3",
        "routine": routine,
        "family": family,
        "kind": kind,
        "library": library,
        "library_path": f"lib{library}.so",
        "shape": shape,
        "m": str(m),
        "n": str(n),
        "side": side,
        "uplo": uplo,
        "trans": trans,
        "diag": diag,
        "alpha_re": str(alpha[0]),
        "alpha_im": str(alpha[1]),
        "order": str(order),
        "lda": str(order),
        "ldb": str(m),
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


class TriangularMatrixRunnerTests(unittest.TestCase):
    def test_default_cases_cover_all_broad_parameters(self):
        args = runner.parse_args(["--csv", os.devnull])
        cases = runner.requested_cases(args)
        self.assertEqual(args.process_repeats, 3)
        self.assertEqual(len(cases), 480)
        self.assertEqual({case.routine.name for case in cases}, set(runner.ROUTINES))
        self.assertEqual(
            {case.shape.name for case in cases},
            {"square128", "tall512x128", "wide128x512"},
        )
        self.assertEqual({case.side for case in cases}, {"L", "R"})
        self.assertEqual({case.uplo for case in cases}, {"U", "L"})
        self.assertEqual({case.diag for case in cases}, {"N", "U"})
        for case in cases:
            legal = {"N", "T", "C"} if case.routine.complex_scalars else {"N", "T"}
            self.assertIn(case.trans, legal)
            alpha = runner.parse_scalar(case.alpha)
            self.assertEqual(alpha[1] != 0, case.routine.complex_scalars)

    def test_complex_transpose_and_scalar_filters_skip_illegal_real_cases(self):
        args = runner.parse_args(
            [
                "--csv",
                os.devnull,
                "--routine",
                "strmm",
                "--routine",
                "ctrsm",
                "--shape",
                "tiny:3:2",
                "--side",
                "R",
                "--uplo",
                "L",
                "--trans",
                "C",
                "--diag",
                "U",
                "--alpha",
                "0.5,0.125",
            ]
        )
        cases = runner.requested_cases(args)
        self.assertEqual(len(cases), 1)
        self.assertEqual(cases[0].routine.name, "ctrsm")
        self.assertEqual(cases[0].trans, "C")

    def test_case_command_forwards_complete_parameters(self):
        args = runner.parse_args(
            ["--csv", os.devnull, "--probe", "probe", "--reps", "7"]
        )
        case = runner.TriangularMatrixCase(
            runner.ROUTINES["ztrsm"],
            runner.Shape("wide", 64, 513),
            "R",
            "L",
            "C",
            "U",
            "0.75,-0.125",
        )
        command = runner.case_command(args, "MKL", "libmkl_rt.so", case)
        pairs = dict(zip(command[1::2], command[2::2]))
        self.assertEqual(pairs["--blas"], "libmkl_rt.so")
        self.assertEqual(pairs["--routine"], "ztrsm")
        self.assertEqual(pairs["--shape"], "wide")
        self.assertEqual(pairs["--m"], "64")
        self.assertEqual(pairs["--n"], "513")
        self.assertEqual(pairs["--side"], "R")
        self.assertEqual(pairs["--uplo"], "L")
        self.assertEqual(pairs["--trans"], "C")
        self.assertEqual(pairs["--diag"], "U")
        self.assertEqual(pairs["--alpha"], "0.75,-0.125")
        self.assertEqual(pairs["--reps"], "7")

    @mock.patch.object(runner.subprocess, "run")
    def test_probe_failure_becomes_complete_error_row(self, run):
        run.return_value = subprocess.CompletedProcess(
            ["probe"], 1, stdout="", stderr="missing symbol"
        )
        args = runner.parse_args(["--csv", os.devnull, "--probe", "probe"])
        case = runner.TriangularMatrixCase(
            runner.ROUTINES["dtrmm"],
            runner.Shape("rect", 7, 3),
            "R",
            "U",
            "T",
            "N",
            "0.75",
        )
        row = runner.run_one_process(args, "TestBLAS", "libblas.so", case)
        self.assertEqual(row["status"], "error")
        self.assertEqual(row["family"], "trmm")
        self.assertEqual(row["order"], "3")
        self.assertEqual(row["trans"], "T")
        self.assertEqual(row["diag"], "N")
        self.assertIn("missing symbol", row["check_raw_output"])

    def test_explicit_missing_path_is_checked_without_loading_blas(self):
        self.assertFalse(runner.library_available("/not/a/real/libblas.so"))
        self.assertTrue(runner.library_available("libblas.so"))


class TriangularMatrixAggregationTests(unittest.TestCase):
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
        self.assertIn("repeat=2", aggregate["check_raw_output"])


class TriangularMatrixCheckerTests(unittest.TestCase):
    def run_checker(self, rows, *extra_args):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "triangular.csv"
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

    def test_complete_parameters_keep_groups_separate(self):
        rows = [
            report_row("Zynum", trans="N", diag="N"),
            report_row("Reference", trans="N", diag="N"),
            report_row(
                "Zynum",
                routine="ztrmm",
                family="trmm",
                kind="c64",
                side="R",
                uplo="L",
                trans="C",
                diag="U",
                alpha=(0.5, 0.125),
            ),
        ]
        result, stdout, stderr = self.run_checker(rows)
        self.assertEqual(result, 1, stderr)
        self.assertIn("checked=1 passed=1 failed=0 missing=1", stdout)
        self.assertIn("trans=C", stdout)
        self.assertIn("diag=U", stdout)
        self.assertIn("family=trmm", stdout)

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


class TriangularMatrixProbeIntegrationTests(unittest.TestCase):
    @unittest.skipUnless(
        (REPO_ROOT / "zig-out/bin/triangular-matrix-probe").is_file()
        and (sys.platform == "darwin" or integration_blas().is_file()),
        "triangular matrix probe and an integration BLAS are not available",
    )
    def test_all_routines_and_legal_transposes_pass_full_reference_checks(self):
        probe = REPO_ROOT / "zig-out/bin/triangular-matrix-probe"
        library = integration_blas()
        for index, (routine, spec) in enumerate(runner.ROUTINES.items()):
            alpha = runner.COMPLEX_ALPHA if spec.complex_scalars else runner.REAL_ALPHA
            transposes = ("N", "T", "C") if spec.complex_scalars else ("N", "T")
            for trans in transposes:
                side = "L" if index % 2 == 0 else "R"
                uplo = "U" if index % 2 == 0 else "L"
                diag = "N" if trans == "N" else "U"
                with self.subTest(routine=routine, trans=trans):
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
                            "--trans",
                            trans,
                            "--diag",
                            diag,
                            "--alpha",
                            alpha,
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
