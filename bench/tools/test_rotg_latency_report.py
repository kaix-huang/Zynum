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
        module_name, TOOLS_DIR / (module_name + ".py")
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


runner = load_tool("run_rotg_latency_report")
checker = load_tool("check_rotg_latency_report")


def report_row(
    library,
    routine="drotg",
    input_case="balanced",
    metric_min=8.0,
    metric_median=10.0,
    metric_max=12.0,
    status="ok",
    check_status="checked-ok",
):
    kind = runner.ROUTINES[routine][0]
    is_rotmg = runner.ROUTINES[routine][1]
    return {
        "level": "level1",
        "routine": routine,
        "kind": kind,
        "library": library,
        "library_path": "lib{}.so".format(library),
        "case": input_case,
        "corpus_size": "2",
        "samples": "5",
        "calls_per_sample": "1000",
        "total_calls": "5000",
        "best_ns_per_call": "7",
        "median_ns_per_call": "9",
        "p95_ns_per_call": "11",
        "max_ns_per_call": "12",
        "median_full_ns_per_call": "15",
        "median_harness_ns_per_call": "6",
        "nonpositive_pairs": "0",
        "metric": "ns_per_call",
        "status": status,
        "check_status": check_status,
        "check_max_abs_error": "0",
        "check_max_rel_error": "0",
        "check_samples": "12",
        "expected_flag": (
            format(runner.EXPECTED_FLAGS[input_case], ".17g") if is_rotmg else ""
        ),
        "observed_flag": (
            format(runner.EXPECTED_FLAGS[input_case], ".17g") if is_rotmg else ""
        ),
        "checksum": "12345",
        "check_raw_output": "",
        "process_repeats": "3",
        "successful_repeats": "3" if status == "ok" else "2",
        "metric_min": str(metric_min),
        "metric_median": str(metric_median),
        "metric_max": str(metric_max),
        "metric_samples": "{},{},{}".format(
            metric_min, metric_median, metric_max
        ),
    }


class RotgLatencyRunnerTests(unittest.TestCase):
    def test_default_matrix_covers_all_routines_exponents_and_rotmg_flags(self):
        args = runner.parse_args(["--csv", os.devnull])
        cases = runner.requested_cases(args)
        self.assertEqual(len(cases), 4 * len(runner.ROTG_CASES) + 2 * len(runner.ROTMG_CASES))
        self.assertEqual(args.process_repeats, 3)
        by_routine = {}
        for case in cases:
            by_routine.setdefault(case.routine, set()).add(case.input_case)
        for routine in ("srotg", "drotg", "crotg", "zrotg"):
            self.assertEqual(by_routine[routine], set(runner.ROTG_CASES))
            self.assertIn("tiny_exponent", by_routine[routine])
            self.assertIn("huge_exponent", by_routine[routine])
        for routine in ("srotmg", "drotmg"):
            self.assertEqual(by_routine[routine], set(runner.ROTMG_CASES))
            self.assertEqual(
                {runner.EXPECTED_FLAGS[value] for value in by_routine[routine]},
                {-2.0, -1.0, 0.0, 1.0},
            )

    def test_incompatible_explicit_case_is_rejected(self):
        args = runner.parse_args(
            [
                "--csv",
                os.devnull,
                "--routine",
                "srotg",
                "--case",
                "flag_one_q2_dominant",
            ]
        )
        with self.assertRaisesRegex(ValueError, "not valid"):
            runner.requested_cases(args)

    def test_case_command_forwards_complete_measurement_parameters(self):
        args = runner.parse_args(
            [
                "--csv",
                os.devnull,
                "--probe",
                "probe",
                "--samples",
                "7",
                "--calls-per-sample",
                "1234",
            ]
        )
        case = runner.LatencyCase("drotmg", "flag_zero_q1_dominant")
        command = runner.case_command(args, "MKL", "libmkl_rt.so", case)
        pairs = dict(zip(command[1::2], command[2::2]))
        self.assertEqual(pairs["--blas"], "libmkl_rt.so")
        self.assertEqual(pairs["--library"], "MKL")
        self.assertEqual(pairs["--routine"], "drotmg")
        self.assertEqual(pairs["--case"], "flag_zero_q1_dominant")
        self.assertEqual(pairs["--samples"], "7")
        self.assertEqual(pairs["--calls-per-sample"], "1234")

    def test_probe_flag_matching_is_numeric_not_textual(self):
        args = runner.parse_args(["--csv", os.devnull])
        case = runner.LatencyCase("drotmg", "flag_neg2_zero_p2")
        row = runner.error_row(args, "Zynum", "libzynum.so", case, "")
        row.update(
            {
                "expected_flag": "-2.00000000000000000",
                "level": "level1",
                "routine": "drotmg",
                "kind": "f64",
                "library": "Zynum",
                "library_path": "libzynum.so",
                "case": "flag_neg2_zero_p2",
                "samples": "9",
                "calls_per_sample": "100000",
                "total_calls": "900000",
                "metric": "ns_per_call",
            }
        )
        self.assertEqual(
            runner.probe_row_mismatches(
                args, row, "Zynum", "libzynum.so", case
            ),
            [],
        )

    @mock.patch.object(runner.subprocess, "run")
    def test_probe_failure_becomes_parameterized_error_row(self, run):
        run.return_value = subprocess.CompletedProcess(
            ["probe"], 1, stdout="", stderr="error: MissingSymbol"
        )
        args = runner.parse_args(["--csv", os.devnull, "--probe", "probe"])
        case = runner.LatencyCase("srotmg", "flag_one_q2_dominant")
        row = runner.run_one_process(args, "TestBLAS", "libblas.so", case)
        self.assertEqual(row["status"], "error")
        self.assertEqual(row["routine"], "srotmg")
        self.assertEqual(row["case"], "flag_one_q2_dominant")
        self.assertEqual(row["expected_flag"], "1")
        self.assertIn("MissingSymbol", row["check_raw_output"])


class RotgLatencyAggregationTests(unittest.TestCase):
    def test_process_aggregate_uses_median_of_probe_medians(self):
        rows = []
        for value in (12.0, 8.0, 10.0):
            row = report_row("Zynum")
            row["median_ns_per_call"] = str(value)
            rows.append(row)
        aggregate = runner.aggregate_repeats(rows)
        self.assertEqual(aggregate["process_repeats"], 3)
        self.assertEqual(aggregate["successful_repeats"], 3)
        self.assertEqual(aggregate["metric_min"], "8")
        self.assertEqual(aggregate["metric_median"], "10")
        self.assertEqual(aggregate["metric_max"], "12")
        self.assertEqual(aggregate["metric_samples"], "12,8,10")

    def test_any_bad_repeat_contaminates_aggregate(self):
        good = report_row("Zynum")
        bad = report_row(
            "Zynum",
            status="correctness_failed",
            check_status="correctness_failed",
        )
        bad["check_raw_output"] = "reference tolerance exceeded"
        aggregate = runner.aggregate_repeats([good, bad])
        self.assertEqual(aggregate["successful_repeats"], 1)
        self.assertEqual(aggregate["status"], "correctness_failed")
        self.assertEqual(aggregate["check_status"], "correctness_failed")
        self.assertIn("repeat=2", aggregate["check_raw_output"])


class RotgLatencyCheckerTests(unittest.TestCase):
    def run_checker(self, rows, *extra_args):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "rotg_latency.csv"
            with path.open("w", newline="") as file:
                writer = csv.DictWriter(file, fieldnames=runner.CSV_FIELDNAMES)
                writer.writeheader()
                writer.writerows(rows)
            stdout = io.StringIO()
            stderr = io.StringIO()
            with redirect_stdout(stdout), redirect_stderr(stderr):
                result = checker.main(
                    [str(path), "--comparator", "Reference"] + list(extra_args)
                )
        return result, stdout.getvalue(), stderr.getvalue()

    def test_checker_compares_fresh_process_median_latency(self):
        rows = [
            report_row(
                "Zynum", metric_min=1.0, metric_median=10.0, metric_max=11.0
            ),
            report_row(
                "Reference", metric_min=4.0, metric_median=8.0, metric_max=20.0
            ),
        ]
        result, stdout, stderr = self.run_checker(rows)
        self.assertEqual(result, 1, stderr)
        self.assertIn("stat=median", stdout)
        self.assertIn("FAIL 1.250000", stdout)

    def test_status_and_correctness_are_checked_before_latency(self):
        rows = [
            report_row(
                "Zynum",
                metric_median=1.0,
                status="correctness_failed",
                check_status="correctness_failed",
            ),
            report_row("Reference", metric_median=100.0),
        ]
        result, stdout, stderr = self.run_checker(rows)
        self.assertEqual(result, 2, stdout)
        self.assertIn("not eligible", stderr)
        self.assertNotIn("FAIL", stdout)

    def test_complete_case_key_keeps_groups_separate(self):
        rows = [
            report_row("Zynum", input_case="balanced"),
            report_row("Reference", input_case="balanced"),
            report_row("Zynum", input_case="a_dominant"),
        ]
        result, stdout, stderr = self.run_checker(rows)
        self.assertEqual(result, 1, stderr)
        self.assertIn("checked=1 passed=1 failed=0 missing=1", stdout)
        self.assertIn("case=a_dominant", stdout)


class RotgLatencyProbeIntegrationTests(unittest.TestCase):
    @unittest.skipUnless(
        (REPO_ROOT / "zig-out/bin/rotg-latency-probe").is_file()
        and (REPO_ROOT / runner.default_zynum_blas()).is_file(),
        "ROTG latency probe and Zynum shared library have not been built",
    )
    def test_all_default_corpus_cases_pass_independent_reference_checks(self):
        probe = REPO_ROOT / "zig-out/bin/rotg-latency-probe"
        library = REPO_ROOT / runner.default_zynum_blas()
        args = runner.parse_args(["--csv", os.devnull])
        for case in runner.requested_cases(args):
            with self.subTest(routine=case.routine, input_case=case.input_case):
                result = subprocess.run(
                    [
                        str(probe),
                        "--blas",
                        str(library),
                        "--library",
                        "Zynum",
                        "--routine",
                        case.routine,
                        "--case",
                        case.input_case,
                        "--samples",
                        "1",
                        "--calls-per-sample",
                        "1000",
                    ],
                    cwd=str(REPO_ROOT),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    universal_newlines=True,
                    check=False,
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                rows = list(csv.DictReader(result.stdout.splitlines()))
                self.assertEqual(len(rows), 1)
                self.assertEqual(rows[0]["routine"], case.routine)
                self.assertEqual(rows[0]["case"], case.input_case)
                self.assertEqual(rows[0]["check_status"], "checked-ok", rows[0])
                if case.input_case in runner.EXPECTED_FLAGS:
                    self.assertEqual(
                        float(rows[0]["observed_flag"]),
                        runner.EXPECTED_FLAGS[case.input_case],
                    )


if __name__ == "__main__":
    unittest.main()
