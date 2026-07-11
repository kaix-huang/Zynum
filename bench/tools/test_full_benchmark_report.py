#!/usr/bin/env python3
# Copyright (C) 2026 Zynum contributors
# SPDX-License-Identifier: LGPL-3.0-or-later

import csv
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

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


report = load_tool("render_full_benchmark_report")


def write_rows(path, rows):
    fields = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", newline="") as file_handle:
        writer = csv.DictWriter(file_handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def library_pair(base, zynum_value, comparator_value, comparator="OpenBLAS"):
    zynum = dict(base, library="Zynum", metric_median=str(zynum_value))
    other = dict(base, library=comparator, metric_median=str(comparator_value))
    return [zynum, other]


class FullBenchmarkReportTest(unittest.TestCase):
    def test_all_schemas_invalid_rows_and_outputs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            input_dir = root / "input"
            output_dir = root / "output"
            input_dir.mkdir()

            level1 = library_pair(
                {
                    "group": "real_f32",
                    "op": "saxpy",
                    "variant": "default",
                    "n": "1024",
                    "incx": "1",
                    "incy": "1",
                    "metric": "rate_gops",
                    "status": "ok",
                    "check_status": "sampled-ok",
                    "rate_gops": "1",
                    "bandwidth_gbps": "",
                },
                12,
                10,
            )
            level1.append(
                dict(
                    level1[1],
                    library="MKL",
                    metric_median="1000",
                    check_status="correctness-failed",
                )
            )
            write_rows(input_dir / "level1_full.csv", level1)

            write_rows(
                input_dir / "scalar_latency_full.csv",
                library_pair(
                    {
                        "level": "level1",
                        "routine": "drotg",
                        "kind": "f64",
                        "case": "ordinary",
                        "corpus_size": "4",
                        "median_ns_per_call": "1",
                        "metric": "ns_per_call",
                        "status": "ok",
                        "check_status": "checked-ok",
                    },
                    5,
                    10,
                    "MKL",
                ),
            )

            write_rows(
                input_dir / "level2_full.csv",
                library_pair(
                    {
                        "level": "level2",
                        "case": "sgemv_n",
                        "kind": "f32",
                        "n": "512",
                        "m": "512",
                        "shape": "sq512",
                        "rate_gops": "1",
                        "metric": "gops",
                        "status": "ok",
                        "check_status": "checked-ok",
                    },
                    8,
                    10,
                    "MKL",
                ),
            )

            gemm_base = {
                "kind": "sgemm",
                "transa": "N",
                "transb": "N",
                "shape_index": "0",
                "label": "sq64",
                "m": "64",
                "n": "64",
                "k": "64",
                "gflops": "1",
                "best_ns": "900",
                "check": "checked-ok",
            }
            write_rows(
                input_dir / "gemm_full.csv",
                [
                    dict(gemm_base, library="zynum-blas", median_ns="1000"),
                    dict(gemm_base, library="AOCL-BLIS", median_ns="2000"),
                ],
            )

            write_rows(
                input_dir / "rank_k_full.csv",
                library_pair(
                    {
                        "level": "level3",
                        "routine": "ssyrk",
                        "kind": "f32",
                        "shape": "n128_k32",
                        "n": "128",
                        "k": "32",
                        "uplo": "U",
                        "trans": "N",
                        "metric": "gflops",
                        "status": "ok",
                        "check_status": "sampled-ok",
                        "gflops": "1",
                    },
                    6,
                    3,
                ),
            )

            write_rows(
                input_dir / "symm_hemm_full.csv",
                library_pair(
                    {
                        "level": "level3",
                        "routine": "ssymm",
                        "kind": "f32",
                        "shape": "square128",
                        "m": "128",
                        "n": "128",
                        "side": "L",
                        "uplo": "U",
                        "beta_re": "0.25",
                        "metric": "gflops",
                        "status": "ok",
                        "check_status": "checked-ok",
                        "gflops": "1",
                    },
                    3,
                    6,
                ),
            )

            write_rows(
                input_dir / "trmm_trsm_full.csv",
                library_pair(
                    {
                        "level": "level3",
                        "routine": "strmm",
                        "family": "trmm",
                        "kind": "f32",
                        "shape": "square128",
                        "m": "128",
                        "n": "128",
                        "side": "L",
                        "uplo": "U",
                        "trans": "N",
                        "diag": "N",
                        "metric": "gflops",
                        "status": "ok",
                        "check_status": "checked-ok",
                        "gflops": "1",
                    },
                    8,
                    8,
                ),
            )
            write_rows(input_dir / "unrelated.csv", [{"name": "not a benchmark"}])

            rendered = report.render_report(
                input_dir, output_dir, ["MKL", "OpenBLAS", "AOCL-BLIS"]
            )
            by_category = {item["id"]: item for item in rendered["categories"]}

            self.assertAlmostEqual(by_category["level1"]["results"][0]["ratio"], 1.2)
            self.assertEqual(by_category["level1"]["rows"]["rejected"], 1)
            self.assertEqual(
                by_category["level1"]["results"][0]["missing_comparators"],
                ["MKL", "AOCL-BLIS"],
            )
            self.assertAlmostEqual(
                by_category["scalar-latency"]["results"][0]["ratio"], 2.0
            )
            self.assertAlmostEqual(by_category["level2"]["results"][0]["ratio"], 0.8)
            self.assertAlmostEqual(by_category["gemm"]["results"][0]["ratio"], 2.0)
            self.assertAlmostEqual(by_category["rank-k"]["results"][0]["ratio"], 2.0)
            self.assertAlmostEqual(by_category["symm-hemm"]["results"][0]["ratio"], 0.5)
            self.assertAlmostEqual(by_category["trmm-trsm"]["results"][0]["ratio"], 1.0)
            self.assertEqual(rendered["files"]["ignored_count"], 1)

            expected = {
                "index.html",
                "summary.csv",
                "summary.json",
                *(report.SVG_NAMES.values()),
            }
            self.assertTrue(expected <= {path.name for path in output_dir.iterdir()})
            summary = json.loads((output_dir / "summary.json").read_text())
            self.assertEqual(summary["schema_version"], 1)
            svg = (output_dir / "level1.svg").read_text()
            self.assertIn("Real performance values", svg)
            self.assertIn("1.0 is the strict ratio gate", svg)
            self.assertIn("OpenBLAS", svg)
            self.assertIn('class="library-bar"', svg)
            self.assertIn('class="library-line"', svg)
            self.assertIn('data-library="Zynum"', svg)
            self.assertIn('data-metric="rate_gops"', svg)
            self.assertIn('data-value="12"', svg)
            self.assertNotIn("1000.000x", svg)

    def test_missing_categories_and_missing_comparator_are_explicit(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            input_dir = root / "input"
            output_dir = root / "output"
            input_dir.mkdir()
            write_rows(
                input_dir / "level1_only.csv",
                [
                    {
                        "group": "real_f64",
                        "op": "ddot",
                        "variant": "default",
                        "library": "Zynum",
                        "n": "4096",
                        "metric": "rate_gops",
                        "metric_median": "4",
                        "rate_gops": "4",
                        "status": "ok",
                        "check_status": "passed",
                    }
                ],
            )

            rendered = report.render_report(input_dir, output_dir)
            by_category = {item["id"]: item for item in rendered["categories"]}
            self.assertEqual(
                by_category["level1"]["results"][0]["status"],
                "missing-comparator",
            )
            self.assertEqual(by_category["level1"]["status"], "missing")
            for category in report.CATEGORY_ORDER[1:]:
                self.assertEqual(by_category[category]["status"], "missing")
                self.assertIn("missing", (output_dir / report.SVG_NAMES[category]).read_text())

            with (output_dir / "summary.csv").open(newline="") as file_handle:
                rows = list(csv.DictReader(file_handle))
            self.assertEqual(len(rows), len(report.CATEGORY_ORDER))
            self.assertEqual(
                next(row for row in rows if row["category"] == "gemm")["status"],
                "missing",
            )

    def test_expected_process_repeats_rejects_incomplete_rows(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            input_dir = root / "input"
            output_dir = root / "output"
            input_dir.mkdir()
            base = {
                "group": "real_f64",
                "op": "ddot",
                "variant": "default",
                "n": "4096",
                "metric": "rate_gops",
                "rate_gops": "4",
                "status": "ok",
                "check_status": "checked-ok",
                "process_repeats": "3",
            }
            write_rows(
                input_dir / "level1_repeats.csv",
                [
                    dict(
                        base,
                        library="Zynum",
                        metric_median="4",
                        successful_repeats="3",
                    ),
                    dict(
                        base,
                        library="OpenBLAS",
                        metric_median="5",
                        successful_repeats="2",
                    ),
                ],
            )

            rendered = report.render_report(
                input_dir, output_dir, ["OpenBLAS"], 3
            )
            level1 = rendered["categories"][0]
            self.assertEqual(level1["rows"]["accepted"], 1)
            self.assertEqual(level1["rows"]["rejected"], 1)
            self.assertEqual(level1["results"][0]["status"], "missing-comparator")

    def test_level1_aggregated_repeats_without_process_column_are_accepted(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            input_dir = root / "input"
            output_dir = root / "output"
            input_dir.mkdir()
            base = {
                "group": "real_f64",
                "op": "ddot",
                "variant": "default",
                "n": "4096",
                "metric": "rate_gops",
                "rate_gops": "4",
                "status": "ok",
                "check_status": "checked-ok",
                "successful_repeats": "3",
            }
            write_rows(
                input_dir / "level1_aggregated_repeats.csv",
                [
                    dict(base, library="Zynum", metric_median="5"),
                    dict(base, library="OpenBLAS", metric_median="4"),
                ],
            )

            rendered = report.render_report(
                input_dir, output_dir, ["OpenBLAS"], 3
            )
            level1 = rendered["categories"][0]
            self.assertEqual(level1["rows"]["accepted"], 2)
            self.assertEqual(level1["rows"]["rejected"], 0)
            self.assertEqual(level1["results"][0]["status"], "passed")


if __name__ == "__main__":
    unittest.main()
