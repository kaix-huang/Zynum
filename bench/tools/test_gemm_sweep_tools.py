#!/usr/bin/env python3
# Copyright (C) 2026 Zynum contributors
# SPDX-License-Identifier: LGPL-3.0-or-later

import csv
import importlib.util
import io
import json
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
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


checker = load_tool("check_gemm_sweep")
runner = load_tool("run_gemm_sweep_isolated")


def gemm_row(
    library,
    trans,
    gflops,
    *,
    kind="sgemm",
    best_ns=1,
    median_ns=1,
    max_ns=1,
):
    return {
        "kind": kind,
        "transa": trans[0],
        "transb": trans[1],
        "shape_index": "0",
        "label": "tiny",
        "m": "2",
        "n": "2",
        "k": "2",
        "library": library,
        "gflops": str(gflops),
        "best_ns": str(best_ns),
        "median_ns": str(median_ns),
        "p95_ns": str(max_ns),
        "max_ns": str(max_ns),
        "reps": "3",
        "process_repeats": "1",
        "check": "checked-ok",
    }


class GemmIsolatedRunnerTests(unittest.TestCase):
    def test_isolated_shape_suffix_distinguishes_same_leading_dimension(self):
        cube = runner.isolated_shape_suffix("128:128:128")
        high_k = runner.isolated_shape_suffix("128:128:4096")
        self.assertNotEqual(cube, high_k)
        self.assertTrue(cube.startswith("_128_"))

    def test_transpose_cli_defaults_to_nn_and_normalizes_explicit_pairs(self):
        default_args = runner.parse_args(["--csv", os.devnull])
        self.assertIsNone(default_args.trans)

        args = runner.parse_args(
            ["--csv", os.devnull, "--trans", "nt", "--trans", "CC"]
        )
        self.assertEqual(args.trans, ["NT", "CC"])

    @mock.patch.object(runner.subprocess, "run")
    def test_worker_command_forwards_each_transpose_pair(self, run):
        args = runner.parse_args(
            [
                "--csv",
                os.devnull,
                "--gemm-sweep",
                "gemm-sweep",
                "--trans",
                "NT",
                "--trans",
                "CC",
            ]
        )
        runner.run_one_process(
            args,
            "Zynum",
            "libzynum.so",
            Path("out.csv"),
            kind="cgemm",
            shapes=["tiny:2:2:2"],
        )
        command = run.call_args.args[0]
        self.assertEqual(
            [
                command[index + 1]
                for index, value in enumerate(command)
                if value == "--trans"
            ],
            ["NT", "CC"],
        )

    def test_metadata_records_effective_transpose_selection(self):
        args = runner.parse_args(
            ["--csv", os.devnull, "--trans", "NT", "--trans", "CC"]
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "gemm.csv"
            with mock.patch.object(runner, "zig_version", return_value="test"), mock.patch.object(
                runner, "git_source_snapshot", return_value={}
            ):
                runner.write_metadata(args, [], ["tiny:2:2:2"], output)
            metadata = json.loads(
                output.with_suffix(output.suffix + ".meta.json").read_text()
            )
        self.assertEqual(metadata["transposes"], ["NT", "CC"])

    def test_repeat_merge_distinguishes_transpose_and_upgrades_legacy_nn(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            first = temp / "first.csv"
            second = temp / "second.csv"
            output = temp / "merged.csv"
            legacy_fields = [
                field
                for field in runner.CSV_FIELDNAMES
                if field not in {"transa", "transb"}
            ]
            with first.open("w", newline="") as csv_file:
                writer = csv.DictWriter(csv_file, fieldnames=legacy_fields)
                writer.writeheader()
                legacy = gemm_row("Zynum", "NN", 4)
                legacy.pop("transa")
                legacy.pop("transb")
                writer.writerow(legacy)
            with second.open("w", newline="") as csv_file:
                writer = csv.DictWriter(csv_file, fieldnames=runner.CSV_FIELDNAMES)
                writer.writeheader()
                writer.writerow(gemm_row("Zynum", "NT", 3))

            runner.best_rows_csv([first, second], output)
            with output.open(newline="") as csv_file:
                rows = list(csv.DictReader(csv_file))

        self.assertEqual(len(rows), 2)
        self.assertEqual(
            {(row["transa"], row["transb"]) for row in rows},
            {("N", "N"), ("N", "T")},
        )


class GemmCheckerTests(unittest.TestCase):
    def run_checker(self, rows, *extra_args, fieldnames=None):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "gemm.csv"
            with path.open("w", newline="") as csv_file:
                writer = csv.DictWriter(
                    csv_file, fieldnames=fieldnames or runner.CSV_FIELDNAMES
                )
                writer.writeheader()
                writer.writerows(rows)
            stdout = io.StringIO()
            stderr = io.StringIO()
            with redirect_stdout(stdout), redirect_stderr(stderr):
                result = checker.main(
                    [str(path), "--comparator", "Reference", *extra_args]
                )
        return result, stdout.getvalue(), stderr.getvalue()

    def test_checker_keeps_transpose_groups_separate(self):
        rows = [
            gemm_row("Zynum", "NN", 2),
            gemm_row("Reference", "NN", 1),
            gemm_row("Zynum", "NT", 2),
        ]
        result, stdout, stderr = self.run_checker(rows)
        self.assertEqual(result, 1, stderr)
        self.assertIn("checked=1 passed=1 failed=0 missing=1", stdout)
        self.assertIn("trans=NT", stdout)

    def test_checker_best_median_and_min_statistics(self):
        rows = [
            gemm_row("Zynum", "NN", 10, best_ns=1, median_ns=8, max_ns=16),
            gemm_row("Reference", "NN", 8, best_ns=2, median_ns=4, max_ns=8),
        ]
        best_result, best_stdout, best_stderr = self.run_checker(rows)
        median_result, median_stdout, median_stderr = self.run_checker(
            rows, "--stat", "median"
        )
        min_result, min_stdout, min_stderr = self.run_checker(
            rows, "--stat", "min"
        )

        self.assertEqual(best_result, 0, best_stderr)
        self.assertIn("passed=1 failed=0", best_stdout)
        self.assertEqual(median_result, 1, median_stderr)
        self.assertIn("stat=median", median_stdout)
        self.assertEqual(min_result, 1, min_stderr)
        self.assertIn("stat=min", min_stdout)

    def test_checker_treats_legacy_rows_as_nn(self):
        rows = [gemm_row("Zynum", "NN", 2), gemm_row("Reference", "NN", 1)]
        legacy_fields = [
            field
            for field in runner.CSV_FIELDNAMES
            if field not in {"transa", "transb"}
        ]
        for row in rows:
            row.pop("transa")
            row.pop("transb")
        result, stdout, stderr = self.run_checker(
            rows, "--trans", "NN", fieldnames=legacy_fields
        )
        self.assertEqual(result, 0, stderr)
        self.assertIn("checked=1 passed=1 failed=0", stdout)

    def test_complex_median_uses_complex_flop_factor(self):
        row = gemm_row("Zynum", "CC", 1, kind="cgemm", median_ns=2)
        self.assertEqual(checker.row_gflops(row, "median"), 32.0)


if __name__ == "__main__":
    unittest.main()
