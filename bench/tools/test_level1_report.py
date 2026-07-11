#!/usr/bin/env python3
# Copyright (C) 2026 Zynum contributors
# SPDX-License-Identifier: LGPL-3.0-or-later

import csv
import ctypes
import importlib.util
import io
import os
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest import mock

TOOLS_DIR = Path(__file__).resolve().parent


def load_tool(module_name):
    spec = importlib.util.spec_from_file_location(
        module_name, TOOLS_DIR / f"{module_name}.py"
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {module_name}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


runner = load_tool("run_level1_report")
checker = load_tool("check_level1_report")


def report_row(library, op="sdot", value=2.0, **overrides):
    row = {
        "group": "real_f32",
        "op": op,
        "variant": "default",
        "library": library,
        "library_path": f"lib{library}.so",
        "n": "257",
        "incx": "-2",
        "incy": "3",
        "metric": "rate_gops",
        "status": "ok",
        "rate_gops": str(value),
        "bandwidth_gbps": "",
        "metric_min": str(value),
        "metric_median": str(value),
        "check_status": "sampled-ok",
        "symbol": "cblas_sdot",
        "abi_surface": "cblas",
        "preflight_symbol": "cblas_sdot",
        "preflight_abi_surface": "cblas",
        "capability_status": "supported",
    }
    row.update(overrides)
    return row


class Level1RunnerTests(unittest.TestCase):
    def test_signed_stride_parser_and_legacy_inc_pairs(self):
        self.assertEqual(runner.parse_stride("-2147483648"), -(1 << 31))
        with self.assertRaises(Exception):
            runner.parse_stride("0")

        args = runner.parse_args(["--csv", os.devnull, "--inc", "2"])
        self.assertEqual(args.strides, [2])
        self.assertEqual(args.stride_pairs, [(2, 2)])

    def test_independent_stride_cli_builds_cartesian_pairs(self):
        args = runner.parse_args(
            [
                "--csv",
                os.devnull,
                "--incx",
                "-2",
                "--incx",
                "3",
                "--incy",
                "-1",
                "--incy",
                "4",
            ]
        )
        self.assertEqual(args.stride_pairs, [(-2, -1), (-2, 4), (3, -1), (3, 4)])
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            runner.parse_args(["--csv", os.devnull, "--inc", "2", "--incx", "-2"])

    def test_positive_cases_preserve_full_set_and_negative_cases_are_stable(self):
        self.assertEqual(runner.STABLE_NEGATIVE_OPS, checker.STABLE_NEGATIVE_OPS)
        positive = runner.level1_cases([(2, 2)])
        self.assertEqual(
            {(group, op) for group, op, _, _, _ in positive},
            set(runner.LEVEL1_OPS),
        )

        negative = runner.level1_cases([(-2, 3)])
        negative_ops = {op for _, op, _, _, _ in negative}
        self.assertEqual(negative_ops, runner.STABLE_NEGATIVE_OPS)
        self.assertTrue(
            {"scopy", "sswap", "saxpy", "sdot", "cdotu", "srot", "srotm"}
            <= negative_ops
        )
        self.assertTrue(
            negative_ops.isdisjoint(
                {"saxpby", "caxpby", "sscal", "sasum", "isamax", "snrm2"}
            )
        )

    def test_copy_byte_coverage_can_be_disabled_without_removing_vector_copy(self):
        args = runner.parse_args(
            [
                "--csv",
                os.devnull,
                "--incx",
                "-2",
                "--incy",
                "3",
                "--skip-copy-byte-coverage",
            ]
        )
        self.assertEqual(runner.selected_copy_cases(args), [])
        self.assertIn(
            ("copy", "scopy", "default", -2, 3),
            runner.level1_cases(args.stride_pairs),
        )

    def test_vector_layout_uses_signed_start_and_absolute_span(self):
        positive = runner.VectorLayout(5, 3)
        negative = runner.VectorLayout(5, -3)
        self.assertEqual((positive.span, positive.start), (13, 0))
        self.assertEqual((negative.span, negative.start), (13, 12))
        self.assertEqual([negative.index(i) for i in range(5)], [12, 9, 6, 3, 0])

    def test_guarded_array_allows_active_writes_but_rejects_gap_and_guard_writes(self):
        array, _ = runner.real_array(ctypes.c_float, 4, 1, -2)
        array.set_logical(0, ctypes.c_float(9.0))
        self.assertEqual(array.modified_element_count(True), 0)

        array[1] = ctypes.c_float(7.0)
        self.assertEqual(array.modified_element_count(True), 1)

        guarded, _ = runner.real_array(ctypes.c_float, 4, 1, -2)
        guarded.storage[0] = ctypes.c_float(3.0)
        self.assertEqual(guarded.modified_element_count(True), 1)

    def test_unstable_negative_operation_is_excluded_before_library_loading(self):
        result = runner.check_level1_op("/not/a/library", "snrm2", -2, 3)
        self.assertEqual(result["check_status"], "missing")
        self.assertEqual(result["capability_status"], "excluded-by-policy")

    def test_probe_output_records_actual_symbol_surface(self):
        output = (
            "iters=2 elapsed_ns=3 rate_Gops=4 bandwidth_GBps=5 checksum=6 "
            "symbol=cblas_sdot abi_surface=cblas\n"
        )
        self.assertEqual(
            runner.parse_probe_output(output),
            (4.0, 5.0, "cblas_sdot", "cblas"),
        )

    @mock.patch.object(runner, "run_once")
    @mock.patch.object(runner, "check_level1_op_isolated")
    def test_run_forwards_independent_strides_and_keeps_matching_surface(
        self, check, run_once
    ):
        check.return_value = runner.check_result(
            "sampled-ok",
            0.0,
            symbol="saxpy_",
            abi_surface="fortran",
            memory_status="guarded-ok",
        )
        run_once.return_value = {
            "status": "ok",
            "returncode": 0,
            "rate_gops": 1.0,
            "bandwidth_gbps": 2.0,
            "symbol": "saxpy_",
            "abi_surface": "fortran",
            "raw_output": "ok",
        }
        args = runner.parse_args(
            ["--csv", os.devnull, "--level1-probe", "probe", "--n", "17"]
        )
        row = runner.run_level1_op(
            args, "Test", "libtest.so", "real_f32", "saxpy", "default", -2, 3
        )
        command = run_once.call_args.args[0]
        self.assertEqual(command[command.index("--incx") + 1], "-2")
        self.assertEqual(command[command.index("--incy") + 1], "3")
        self.assertEqual(row["status"], "ok")
        self.assertEqual((row["incx"], row["incy"]), (-2, 3))

    @mock.patch.object(runner, "run_once")
    @mock.patch.object(runner, "check_level1_op_isolated")
    def test_unsupported_symbol_is_missing_and_never_timed(self, check, run_once):
        check.return_value = runner.check_result(
            "missing",
            raw="missing scopy_",
            capability_status="unsupported",
        )
        args = runner.parse_args(
            ["--csv", os.devnull, "--level1-probe", "probe", "--n", "17"]
        )
        row = runner.run_level1_op(
            args, "Test", "libtest.so", "copy", "scopy", "default", -2, 3
        )
        self.assertEqual(row["status"], "missing")
        self.assertEqual(row["capability_status"], "unsupported")
        run_once.assert_not_called()

    @mock.patch.object(runner, "run_once")
    @mock.patch.object(runner, "check_level1_op_isolated")
    def test_surface_mismatch_invalidates_timing(self, check, run_once):
        check.return_value = runner.check_result(
            "sampled-ok",
            0.0,
            symbol="sdot_",
            abi_surface="fortran",
            memory_status="guarded-ok",
        )
        run_once.return_value = {
            "status": "ok",
            "returncode": 0,
            "rate_gops": 9.0,
            "bandwidth_gbps": 9.0,
            "symbol": "cblas_sdot",
            "abi_surface": "cblas",
            "raw_output": "ok",
        }
        args = runner.parse_args(
            ["--csv", os.devnull, "--level1-probe", "probe", "--n", "17"]
        )
        row = runner.run_level1_op(
            args, "Test", "libtest.so", "real_f32", "sdot", "default", -2, 3
        )
        self.assertEqual(row["status"], "surface_mismatch")
        self.assertIsNone(row["rate_gops"])


class Level1CheckerTests(unittest.TestCase):
    def run_checker(self, rows, *extra_args):
        fields = sorted({field for row in rows for field in row})
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "level1.csv"
            with path.open("w", newline="") as csv_file:
                writer = csv.DictWriter(csv_file, fieldnames=fields)
                writer.writeheader()
                writer.writerows(rows)
            stdout = io.StringIO()
            stderr = io.StringIO()
            with redirect_stdout(stdout), redirect_stderr(stderr):
                result = checker.main(
                    [
                        str(path),
                        "--comparator",
                        "Reference",
                        "--stat",
                        "median",
                        *extra_args,
                    ]
                )
        return result, stdout.getvalue(), stderr.getvalue()

    def test_checker_excludes_axpby_from_negative_gate(self):
        rows = [
            report_row("Zynum"),
            report_row("Reference", value=1.0),
            report_row(
                "Zynum",
                op="saxpby",
                value=0.01,
                symbol="saxpby_",
                preflight_symbol="saxpby_",
                abi_surface="fortran",
                preflight_abi_surface="fortran",
            ),
            report_row(
                "Reference",
                op="saxpby",
                value=100.0,
                symbol="saxpby_",
                preflight_symbol="saxpby_",
                abi_surface="fortran",
                preflight_abi_surface="fortran",
            ),
        ]
        result, stdout, stderr = self.run_checker(rows)
        self.assertEqual(result, 0, stderr)
        self.assertIn("checked=1 passed=1", stdout)
        self.assertIn("excluded_negative=2", stdout)

    def test_checker_accepts_legacy_positive_rows_without_surface_columns(self):
        rows = [
            report_row("Zynum", incx="1", incy="1"),
            report_row("Reference", value=1.0, incx="1", incy="1"),
        ]
        for row in rows:
            for field in (
                "symbol",
                "abi_surface",
                "preflight_symbol",
                "preflight_abi_surface",
                "capability_status",
            ):
                row.pop(field)
        result, stdout, stderr = self.run_checker(rows)
        self.assertEqual(result, 0, stderr)
        self.assertIn("checked=1 passed=1", stdout)

    def test_negative_only_ignores_positive_rows_in_same_report(self):
        positive_zynum = report_row("Zynum", value=0.01, incx="1", incy="1")
        positive_reference = report_row("Reference", value=100.0, incx="1", incy="1")
        rows = [
            report_row("Zynum"),
            report_row("Reference", value=1.0),
            positive_zynum,
            positive_reference,
        ]
        result, stdout, stderr = self.run_checker(rows, "--negative-only")
        self.assertEqual(result, 0, stderr)
        self.assertIn("checked=1 passed=1", stdout)

    def test_checker_rejects_negative_row_without_surface_preflight(self):
        row = report_row("Zynum", symbol="", abi_surface="")
        result, _, stderr = self.run_checker([row, report_row("Reference")])
        self.assertEqual(result, 2)
        self.assertIn("lacks a supported capability surface", stderr)

    def test_checker_rejects_nonpositive_timing_metric(self):
        result, _, stderr = self.run_checker(
            [report_row("Zynum", value=0.0), report_row("Reference", value=1.0)]
        )
        self.assertEqual(result, 2)
        self.assertIn("bad metric value", stderr)


@unittest.skipUnless(
    runner.library_available(runner.DEFAULT_ACCELERATE), "Accelerate is unavailable"
)
class Level1AccelerateIntegrationTests(unittest.TestCase):
    def test_representative_mixed_sign_stable_operations(self):
        cases = (
            ("scopy", "default"),
            ("sswap", "default"),
            ("saxpy", "default"),
            ("sdot", "default"),
            ("cdotu", "default"),
            ("csrot", "default"),
            ("srotm", "flag_0"),
        )
        for op, variant in cases:
            with self.subTest(op=op, variant=variant):
                result = runner.check_level1_op(
                    runner.DEFAULT_ACCELERATE, op, -2, 3, variant
                )
                self.assertEqual(result["check_status"], "sampled-ok", result)
                self.assertEqual(result["capability_status"], "supported")
                self.assertEqual(result["check_memory_status"], "guarded-ok")
                self.assertTrue(result["preflight_symbol"])
                self.assertTrue(result["preflight_abi_surface"])


if __name__ == "__main__":
    unittest.main()
