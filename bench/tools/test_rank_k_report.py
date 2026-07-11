#!/usr/bin/env python3
# Copyright (C) 2026 Zynum contributors
# SPDX-License-Identifier: LGPL-3.0-or-later

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


runner = load_tool("run_rank_k_report")
checker = load_tool("check_rank_k_report")


def report_row(
    library,
    *,
    routine="dsyrk",
    trans="N",
    alpha=(0.75, 0.0),
    beta=(0.25, 0.0),
    metric_min=1,
    metric_median=2,
    metric_max=3,
    status="ok",
    check_status="checked-ok",
    ldb=None,
):
    rank2k = routine.endswith("2k")
    kind = {"s": "f32", "d": "f64", "c": "c32", "z": "c64"}[routine[0]]
    factor = 8 if kind.startswith("c") else 2
    if rank2k:
        factor *= 2
    lda = "3" if trans == "N" else "2"
    return {
        "level": "level3",
        "routine": routine,
        "kind": kind,
        "library": library,
        "library_path": f"lib{library}.so",
        "shape": "tiny",
        "n": "3",
        "k": "2",
        "uplo": "U",
        "trans": trans,
        "alpha_re": str(alpha[0]),
        "alpha_im": str(alpha[1]),
        "beta_re": str(beta[0]),
        "beta_im": str(beta[1]),
        "lda": lda,
        "ldb": (lda if ldb is None else ldb) if rank2k else "",
        "ldc": "3",
        "reps": "2",
        "flop_count": str(factor * 6 * 2),
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
        "check_samples": "9",
        "check_raw_output": "",
        "process_repeats": "3",
        "successful_repeats": "3" if status == "ok" else "2",
        "metric_min": str(metric_min),
        "metric_median": str(metric_median),
        "metric_max": str(metric_max),
        "metric_samples": f"{metric_min},{metric_median},{metric_max}",
    }


class RankKRunnerTests(unittest.TestCase):
    def test_default_cases_cover_all_legal_transposes_and_uplos(self):
        args = runner.parse_args(["--csv", os.devnull])
        cases = runner.requested_cases(args)
        self.assertEqual(args.process_repeats, 3)
        self.assertEqual(len(cases), len(runner.DEFAULT_SHAPES) * 12 * 2 * 2)
        self.assertEqual(len(cases), 192)
        by_routine = {}
        for case in cases:
            by_routine.setdefault(case.routine.name, set()).add(case.trans)
        self.assertEqual(by_routine["ssyrk"], {"N", "T"})
        self.assertEqual(by_routine["dsyrk"], {"N", "T"})
        self.assertEqual(by_routine["csyrk"], {"N", "T"})
        self.assertEqual(by_routine["zsyrk"], {"N", "T"})
        self.assertEqual(by_routine["cherk"], {"N", "C"})
        self.assertEqual(by_routine["zherk"], {"N", "C"})
        self.assertEqual(by_routine["ssyr2k"], {"N", "T"})
        self.assertEqual(by_routine["dsyr2k"], {"N", "T"})
        self.assertEqual(by_routine["csyr2k"], {"N", "T"})
        self.assertEqual(by_routine["zsyr2k"], {"N", "T"})
        self.assertEqual(by_routine["cher2k"], {"N", "C"})
        self.assertEqual(by_routine["zher2k"], {"N", "C"})
        self.assertEqual({case.uplo for case in cases}, {"U", "L"})

    def test_explicit_complex_scalars_require_complex_syrk_selection(self):
        args = runner.parse_args(
            [
                "--csv",
                os.devnull,
                "--routine",
                "csyrk",
                "--shape",
                "tiny:3:2",
                "--uplo",
                "L",
                "--trans",
                "T",
                "--alpha",
                "0.5,0.125",
                "--beta",
                "-0.25,0.0625",
            ]
        )
        cases = runner.requested_cases(args)
        self.assertEqual(len(cases), 1)
        self.assertEqual(cases[0].alpha, "0.5,0.125")
        self.assertEqual(cases[0].beta, "-0.25,0.0625")

        invalid = runner.parse_args(
            ["--csv", os.devnull, "--routine", "cherk", "--alpha", "1,0.5"]
        )
        with self.assertRaisesRegex(ValueError, "complex alpha"):
            runner.requested_cases(invalid)

    def test_her2k_accepts_complex_alpha_and_requires_real_beta(self):
        valid = runner.parse_args(
            [
                "--csv",
                os.devnull,
                "--routine",
                "zher2k",
                "--shape",
                "tiny:3:2",
                "--alpha",
                "0.5,0.125",
                "--beta",
                "-0.25",
            ]
        )
        cases = runner.requested_cases(valid)
        self.assertEqual(len(cases), 4)
        self.assertEqual({case.trans for case in cases}, {"N", "C"})

        invalid = runner.parse_args(
            [
                "--csv",
                os.devnull,
                "--routine",
                "cher2k",
                "--beta",
                "0.25,0.0625",
            ]
        )
        with self.assertRaisesRegex(ValueError, "complex beta"):
            runner.requested_cases(invalid)

    def test_case_command_forwards_complete_parameters(self):
        args = runner.parse_args(
            ["--csv", os.devnull, "--probe", "probe", "--reps", "7"]
        )
        case = runner.RankKCase(
            runner.ROUTINES["zher2k"],
            runner.Shape("highk", 64, 513),
            "L",
            "C",
            "0.75,0.125",
            "0.25",
        )
        command = runner.case_command(args, "MKL", "libmkl_rt.so", case)
        pairs = dict(zip(command[1::2], command[2::2]))
        self.assertEqual(pairs["--blas"], "libmkl_rt.so")
        self.assertEqual(pairs["--library"], "MKL")
        self.assertEqual(pairs["--routine"], "zher2k")
        self.assertEqual(pairs["--shape"], "highk")
        self.assertEqual(pairs["--n"], "64")
        self.assertEqual(pairs["--k"], "513")
        self.assertEqual(pairs["--uplo"], "L")
        self.assertEqual(pairs["--trans"], "C")
        self.assertEqual(pairs["--alpha"], "0.75,0.125")
        self.assertEqual(pairs["--beta"], "0.25")
        self.assertEqual(pairs["--reps"], "7")

    def test_rank2k_flop_count_is_twice_matching_rank_k(self):
        shape = runner.Shape("tiny", 3, 2)
        syrk = runner.RankKCase(
            runner.ROUTINES["dsyrk"], shape, "U", "N", "1", "0"
        )
        syr2k = runner.RankKCase(
            runner.ROUTINES["dsyr2k"], shape, "U", "N", "1", "0"
        )
        self.assertEqual(runner.flop_count(syr2k), 2 * runner.flop_count(syrk))

    @mock.patch.object(runner.subprocess, "run")
    def test_probe_failure_becomes_parameterized_error_row(self, run):
        run.return_value = subprocess.CompletedProcess(
            ["probe"], 1, stdout="", stderr="missing symbol"
        )
        args = runner.parse_args(["--csv", os.devnull, "--probe", "probe"])
        case = runner.RankKCase(
            runner.ROUTINES["dsyr2k"],
            runner.Shape("rect", 7, 3),
            "U",
            "T",
            "0.75",
            "0.25",
        )
        row = runner.run_one_process(args, "TestBLAS", "libblas.so", case)
        self.assertEqual(row["status"], "error")
        self.assertEqual(row["routine"], "dsyr2k")
        self.assertEqual(row["n"], "7")
        self.assertEqual(row["k"], "3")
        self.assertEqual(row["lda"], "3")
        self.assertEqual(row["ldb"], "3")
        self.assertEqual(row["flop_count"], "336")
        self.assertIn("missing symbol", row["check_raw_output"])


class RankKAggregationTests(unittest.TestCase):
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


class RankKCheckerTests(unittest.TestCase):
    def run_checker(self, rows, *extra_args):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "rank_k.csv"
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
            report_row(
                "Zynum", metric_min=1, metric_median=2, metric_max=10
            ),
            report_row(
                "Reference", metric_min=1, metric_median=4, metric_max=8
            ),
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
            report_row("Zynum", trans="N"),
            report_row("Reference", trans="N"),
            report_row("Zynum", trans="T", alpha=(0.5, 0.0)),
        ]
        result, stdout, stderr = self.run_checker(rows)
        self.assertEqual(result, 1, stderr)
        self.assertIn("checked=1 passed=1 failed=0 missing=1", stdout)
        self.assertIn("trans=T", stdout)
        self.assertIn("alpha=0.5,0.0", stdout)

    def test_rank2k_ldb_keeps_groups_separate(self):
        rows = [
            report_row("Zynum", routine="dsyr2k", ldb="3"),
            report_row("Reference", routine="dsyr2k", ldb="3"),
            report_row("Zynum", routine="dsyr2k", ldb="4"),
        ]
        result, stdout, stderr = self.run_checker(rows)
        self.assertEqual(result, 1, stderr)
        self.assertIn("checked=1 passed=1 failed=0 missing=1", stdout)
        self.assertIn("ldb=4", stdout)

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


class RankKProbeIntegrationTests(unittest.TestCase):
    @unittest.skipUnless(
        (REPO_ROOT / "zig-out/bin/rank-k-probe").is_file()
        and (REPO_ROOT / runner.default_zynum_blas()).is_file(),
        "rank-k probe and Zynum shared library have not been built",
    )
    def test_all_twelve_routines_cover_both_triangles_and_legal_transposes(self):
        probe = REPO_ROOT / "zig-out/bin/rank-k-probe"
        library = REPO_ROOT / runner.default_zynum_blas()
        for routine, spec in runner.ROUTINES.items():
            alpha = "0.75" if spec.alpha_must_be_real else "0.75,0.125"
            beta = "0.25" if spec.beta_must_be_real else "0.25,0.0625"
            for uplo in ("U", "L"):
                for trans in spec.transposes:
                    with self.subTest(routine=routine, uplo=uplo, trans=trans):
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
                                "--n",
                                "3",
                                "--k",
                                "2",
                                "--uplo",
                                uplo,
                                "--trans",
                                trans,
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
                        self.assertEqual(
                            rows[0]["check_status"], "checked-ok", rows[0]
                        )
                        self.assertEqual(
                            rows[0]["ldb"], rows[0]["lda"] if spec.rank2k else ""
                        )


if __name__ == "__main__":
    unittest.main()
