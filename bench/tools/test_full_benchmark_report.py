#!/usr/bin/env python3
# Copyright (C) 2026 Zynum contributors
# SPDX-License-Identifier: LGPL-3.0-or-later

import csv
import errno
import hashlib
import importlib.util
import io
import itertools
import json
import math
import os
import stat
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

TOOLS_DIR = Path(__file__).resolve().parent
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

import report_publication as publication  # noqa: E402


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


def publish_outputs_for_rendering_tests(outputs):
    for output in outputs:
        output.path.parent.mkdir(parents=True, exist_ok=True)
        output.path.write_bytes(output.contents)


def library_pair(base, zynum_value, comparator_value, comparator="OpenBLAS"):
    zynum = dict(base, library="Zynum", metric_median=str(zynum_value))
    other = dict(base, library=comparator, metric_median=str(comparator_value))
    return [zynum, other]


def valid_evidence_row(category):
    shared = {
        "library": "Zynum",
        "metric_median": "4",
        "status": "ok",
        "check_status": "checked-ok",
    }
    rows = {
        "level1": {
            "group": "real_f64",
            "op": "ddot",
            "variant": "default",
            "n": "4096",
            "metric": "rate_gops",
            "rate_gops": "4",
        },
        "scalar-latency": {
            "routine": "drotg",
            "kind": "f64",
            "case": "ordinary",
            "corpus_size": "4",
            "median_ns_per_call": "4",
            "metric": "ns_per_call",
        },
        "level2": {
            "case": "dgemv_n",
            "kind": "f64",
            "n": "512",
            "rate_gops": "4",
            "metric": "gops",
        },
        "gemm": {
            "kind": "dgemm",
            "transa": "N",
            "transb": "N",
            "shape_index": "0",
            "label": "sq64",
            "m": "64",
            "n": "64",
            "k": "64",
            "gflops": "4",
            "median_ns": "4",
            "check": "checked-ok",
        },
        "rank-k": {
            "routine": "dsyrk",
            "kind": "f64",
            "shape": "n128_k32",
            "n": "128",
            "k": "32",
            "uplo": "U",
            "trans": "N",
            "metric": "gflops",
            "gflops": "4",
        },
        "symm-hemm": {
            "routine": "dsymm",
            "kind": "f64",
            "shape": "square128",
            "m": "128",
            "n": "128",
            "side": "L",
            "uplo": "U",
            "beta_re": "0",
            "metric": "gflops",
            "gflops": "4",
        },
        "trmm-trsm": {
            "routine": "dtrmm",
            "family": "trmm",
            "kind": "f64",
            "shape": "square128",
            "m": "128",
            "n": "128",
            "side": "L",
            "uplo": "U",
            "trans": "N",
            "diag": "N",
            "metric": "gflops",
            "gflops": "4",
        },
    }
    row = dict(shared)
    row.update(rows[category])
    if category == "gemm":
        row.pop("metric_median")
        row.pop("status")
        row.pop("check_status")
    return row


class FullBenchmarkReportTest(unittest.TestCase):
    def test_render_svg_extreme_finite_axes_are_bounded_and_deterministic(self):
        for name, value in (
            ("maximum-finite", sys.float_info.max),
            ("minimum-subnormal", math.ulp(0.0)),
        ):
            with self.subTest(name=name):
                result = {
                    "case_id": f"gemm:{name}",
                    "case": name,
                    "metric": "gflops",
                    "zynum_value": value,
                    "fastest_comparator": None,
                    "comparator_value": None,
                    "ratio": None,
                    "status": "missing-comparator",
                    "missing_comparators": [],
                    "source_files": ["extreme.csv"],
                    "libraries": {"Zynum": value},
                }

                first = report.render_svg("gemm", [result])
                second = report.render_svg("gemm", [result])

                self.assertEqual(first, second)
                self.assertLess(len(first), 20_000)
                self.assertNotIn(b"nan", first.lower())
                self.assertNotIn(b"inf", first.lower())

    def test_main_renders_extreme_finite_axes_without_unbounded_output(self):
        with tempfile.TemporaryDirectory(dir=TOOLS_DIR) as temporary:
            root = Path(temporary)
            input_dir = root / "input"
            first_output = root / "first-output"
            second_output = root / "second-output"
            input_dir.mkdir()
            maximum = format(sys.float_info.max, ".17g")
            minimum = format(math.ulp(0.0), ".17g")
            write_rows(
                input_dir / "level1.csv",
                [
                    dict(
                        valid_evidence_row("level1"),
                        metric_median=maximum,
                        rate_gops=maximum,
                    )
                ],
            )
            write_rows(
                input_dir / "rank_k.csv",
                [
                    dict(
                        valid_evidence_row("rank-k"),
                        metric_median=minimum,
                        gflops=minimum,
                    )
                ],
            )

            with mock.patch.object(
                report, "publish_outputs", publish_outputs_for_rendering_tests
            ):
                for output_dir in (first_output, second_output):
                    with redirect_stdout(io.StringIO()) as output:
                        return_code = report.main(
                            [
                                "--input-dir",
                                str(input_dir),
                                "--output-dir",
                                str(output_dir),
                            ]
                        )
                    self.assertEqual(return_code, 0, output.getvalue())
                    for name in ("level1.svg", "rank-k.svg"):
                        svg = (output_dir / name).read_bytes()
                        self.assertLess(len(svg), 20_000)
                        self.assertNotIn(b"nan", svg.lower())
                        self.assertNotIn(b"inf", svg.lower())

            self.assertEqual(
                {
                    path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                    for path in first_output.iterdir()
                },
                {
                    path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                    for path in second_output.iterdir()
                },
            )

    def test_all_schemas_and_outputs(self):
        with tempfile.TemporaryDirectory(dir=TOOLS_DIR) as temporary:
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

            with mock.patch.object(
                report, "publish_outputs", publish_outputs_for_rendering_tests
            ):
                rendered = report.render_report(
                    input_dir, output_dir, ["MKL", "OpenBLAS", "AOCL-BLIS"]
                )
            by_category = {item["id"]: item for item in rendered["categories"]}

            self.assertAlmostEqual(by_category["level1"]["results"][0]["ratio"], 1.2)
            self.assertEqual(by_category["level1"]["rows"]["rejected"], 0)
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
            self.assertEqual(expected, {path.name for path in output_dir.iterdir()})
            self.assertEqual(
                {
                    path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                    for path in output_dir.iterdir()
                },
                {
                    "level1.svg": "098609f39d158c0671af04f1c5327f7748bc77050ba7dbbf9a15c2628ef0478c",
                    "scalar-latency.svg": "5576de4f259221f69c460f580477ce9ae49b300f15b10bee91c95281cb60a693",
                    "level2.svg": "0a229b19d39653bc79506f5bdcfd9d2a40f0be17968a80fe277b739c0ab19cd3",
                    "gemm.svg": "08c0e2c40f36d6b232f3cc40df2711ff8a16e83047f1850397e76a234af88f58",
                    "rank-k.svg": "0b264bd9bcb940ff0c01c4476e3d1021e2b67eb8609789ee1bea6cece913eca8",
                    "symm-hemm.svg": "19a822d042e22a4e9916963870e03d333fb41efd90157c32dfe7c91113473a12",
                    "trmm-trsm.svg": "a4ec77dd134be218168d03c16faecb7c950987972ba7075aa5b78c118b327607",
                    "summary.csv": "5fd8d529279ed356fbb0f444168a6ef91bfb7f154703a7a43981317c2c45efd6",
                    "summary.json": "52a2811d97db4f4f0f7948832517916aa98a6426f21935c54d99b20c37d428d1",
                    "index.html": "98cd72de5853ac38d64dd010a4cf1d119666cc60193ae5b20e24240d98d8d9ff",
                },
            )
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

    def test_materializes_ten_outputs_before_one_ordered_publish_call(self):
        with tempfile.TemporaryDirectory(dir=TOOLS_DIR) as temporary:
            root = Path(temporary)
            input_dir = root / "input"
            output_dir = root / "nested" / "output"
            input_dir.mkdir()
            write_rows(input_dir / "level1.csv", [valid_evidence_row("level1")])
            published = []

            with mock.patch.object(
                report,
                "publish_outputs",
                side_effect=lambda outputs: published.append(tuple(outputs)),
            ) as publish:
                rendered = report.render_report(input_dir, output_dir)

            publish.assert_called_once()
            self.assertEqual(len(published), 1)
            outputs = published[0]
            self.assertEqual(
                [item.path.name for item in outputs],
                [
                    "level1.svg",
                    "scalar-latency.svg",
                    "level2.svg",
                    "gemm.svg",
                    "rank-k.svg",
                    "symm-hemm.svg",
                    "trmm-trsm.svg",
                    "summary.csv",
                    "summary.json",
                    "index.html",
                ],
            )
            self.assertEqual(
                [item.path.parent for item in outputs], [output_dir] * len(outputs)
            )
            self.assertTrue(all(isinstance(item.contents, bytes) for item in outputs))
            self.assertEqual(
                [item.contents for item in outputs[: len(report.CATEGORY_ORDER)]],
                [
                    report.render_svg(category["id"], category["results"])
                    for category in rendered["categories"]
                ],
            )
            self.assertEqual(
                outputs[-3].contents,
                report.render_summary_csv(rendered["categories"]),
            )
            self.assertEqual(
                outputs[-2].contents,
                (json.dumps(rendered, indent=2, sort_keys=True) + "\n").encode("utf-8"),
            )
            self.assertEqual(outputs[-1].contents, report.render_index(rendered))
            self.assertFalse(output_dir.parent.exists())

    def test_last_serializer_failure_precedes_publication_and_parent_creation(self):
        with tempfile.TemporaryDirectory(dir=TOOLS_DIR) as temporary:
            root = Path(temporary)
            input_dir = root / "input"
            output_dir = root / "nested" / "output"
            input_dir.mkdir()
            write_rows(input_dir / "level1.csv", [valid_evidence_row("level1")])

            with (
                mock.patch.object(
                    report,
                    "render_index",
                    side_effect=ValueError("injected final serialization failure"),
                ),
                mock.patch.object(report, "publish_outputs") as publish,
                self.assertRaisesRegex(ValueError, "final serialization failure"),
            ):
                report.render_report(input_dir, output_dir)

            publish.assert_not_called()
            self.assertFalse(output_dir.parent.exists())

    def test_nth_commit_failure_restores_complete_old_ten_file_generation(self):
        with tempfile.TemporaryDirectory(dir=TOOLS_DIR) as temporary:
            root = Path(temporary)
            input_dir = root / "input"
            output_dir = root / "output"
            input_dir.mkdir()
            output_dir.mkdir()
            write_rows(input_dir / "level1.csv", [valid_evidence_row("level1")])
            names = [
                *report.SVG_NAMES.values(),
                "summary.csv",
                "summary.json",
                "index.html",
            ]
            original_generation = {
                name: f"old generation: {name}\n".encode("utf-8") for name in names
            }
            for name, contents in original_generation.items():
                (output_dir / name).write_bytes(contents)

            replace = publication._replace_name
            calls = 0

            def fail_fifth_replace(parent, source, destination):
                nonlocal calls
                calls += 1
                if calls == 5:
                    raise OSError(errno.EIO, "injected fifth publication failure")
                return replace(parent, source, destination)

            with (
                mock.patch.object(
                    publication, "_replace_name", side_effect=fail_fifth_replace
                ),
                self.assertRaises(publication.RollbackIndeterminateError) as caught,
            ):
                report.render_report(input_dir, output_dir)

            self.assertEqual(calls, 5)
            self.assertEqual(
                {
                    name: (output_dir / name).read_bytes()
                    for name in original_generation
                },
                original_generation,
            )
            recovery_paths = caught.exception.recovery_paths
            self.assertTrue(recovery_paths)
            self.assertCountEqual(
                [path.read_bytes() for path in recovery_paths],
                original_generation.values(),
            )

    def test_nth_commit_failure_leaves_no_partial_new_output_tree(self):
        with tempfile.TemporaryDirectory(dir=TOOLS_DIR) as temporary:
            root = Path(temporary)
            input_dir = root / "input"
            output_dir = root / "new" / "output"
            input_dir.mkdir()
            write_rows(input_dir / "level1.csv", [valid_evidence_row("level1")])

            replace = publication._replace_name
            calls = 0

            def fail_fifth_replace(parent, source, destination):
                nonlocal calls
                calls += 1
                if calls == 5:
                    raise OSError(errno.EIO, "injected fifth publication failure")
                return replace(parent, source, destination)

            with (
                mock.patch.object(
                    publication, "_replace_name", side_effect=fail_fifth_replace
                ),
                self.assertRaisesRegex(OSError, "injected fifth publication failure"),
            ):
                report.render_report(input_dir, output_dir)

            self.assertEqual(calls, 5)
            self.assertFalse(output_dir.parent.exists())

    def test_symlink_destination_and_victim_remain_unchanged(self):
        with tempfile.TemporaryDirectory(dir=TOOLS_DIR) as temporary:
            root = Path(temporary)
            input_dir = root / "input"
            output_dir = root / "output"
            input_dir.mkdir()
            output_dir.mkdir()
            write_rows(input_dir / "level1.csv", [valid_evidence_row("level1")])
            victim = root / "victim.txt"
            victim.write_bytes(b"victim must not change\n")
            destination = output_dir / "index.html"
            destination.symlink_to(victim)

            with self.assertRaises(OSError):
                report.render_report(input_dir, output_dir)

            self.assertTrue(destination.is_symlink())
            self.assertEqual(destination.readlink(), victim)
            self.assertEqual(victim.read_bytes(), b"victim must not change\n")

            arena_prefix = ".zynum-cleanup-v2-"
            arenas = [
                path
                for path in output_dir.iterdir()
                if path.name.startswith(arena_prefix)
            ]
            self.assertEqual(len(arenas), 1)
            arena = arenas[0]
            arena_metadata = arena.stat(follow_symlinks=False)
            self.assertTrue(stat.S_ISDIR(arena_metadata.st_mode))
            self.assertEqual(stat.S_IMODE(arena_metadata.st_mode), 0o700)
            self.assertEqual(list(arena.iterdir()), [])
            if hasattr(os, "geteuid"):
                self.assertEqual(arena.name, f"{arena_prefix}{os.geteuid()}")
                self.assertEqual(arena_metadata.st_uid, os.geteuid())
            else:
                self.assertTrue(arena.name.removeprefix(arena_prefix).isdigit())
            self.assertEqual(
                {path.name for path in output_dir.iterdir()},
                {"index.html", arena.name},
            )

    def test_publication_oserror_remains_cli_status_two(self):
        with tempfile.TemporaryDirectory(dir=TOOLS_DIR) as temporary:
            root = Path(temporary)
            input_dir = root / "input"
            output_dir = root / "output"
            input_dir.mkdir()
            write_rows(input_dir / "level1.csv", [valid_evidence_row("level1")])

            with (
                mock.patch.object(
                    report,
                    "publish_outputs",
                    side_effect=OSError(errno.EIO, "injected publication error"),
                ) as publish,
                redirect_stdout(io.StringIO()) as output,
            ):
                return_code = report.main(
                    [
                        "--input-dir",
                        str(input_dir),
                        "--output-dir",
                        str(output_dir),
                    ]
                )

            publish.assert_called_once()
            self.assertEqual(return_code, 2)
            self.assertIn("injected publication error", output.getvalue())
            self.assertFalse(output_dir.exists())

    def test_finite_extreme_median_does_not_overflow(self):
        case_id = "level1:extreme"
        groups = {
            case_id: {
                "case_id": case_id,
                "case": "extreme",
                "metric": "rate_gops",
                "libraries": {
                    "Zynum": [1e308, 1e308],
                    "OpenBLAS": [1e308, 1e308],
                },
                "sources": {"extreme.csv"},
            }
        }

        result = report.aggregate_category("level1", groups)[0]

        self.assertEqual(result["zynum_value"], 1e308)
        self.assertEqual(result["comparator_value"], 1e308)
        self.assertEqual(result["ratio"], 1.0)

    def test_ratio_overflow_and_underflow_fail_without_publishing(self):
        cases = (
            ("overflow", "1e308", "1e-308", False),
            ("underflow", "1e-308", "1e308", True),
        )
        for name, zynum_value, comparator_value, preexisting in cases:
            with (
                self.subTest(name=name),
                tempfile.TemporaryDirectory(dir=TOOLS_DIR) as temporary,
            ):
                root = Path(temporary)
                input_dir = root / "input"
                output_dir = root / "output"
                input_dir.mkdir()
                if preexisting:
                    output_dir.mkdir()
                    (output_dir / "index.html").write_text("old report")
                write_rows(
                    input_dir / "level1.csv",
                    library_pair(
                        {
                            "group": "real_f64",
                            "op": "ddot",
                            "variant": "default",
                            "n": "4096",
                            "metric": "rate_gops",
                            "rate_gops": "1",
                            "status": "ok",
                            "check_status": "checked-ok",
                        },
                        zynum_value,
                        comparator_value,
                    ),
                )

                with redirect_stdout(io.StringIO()) as output:
                    return_code = report.main(
                        [
                            "--input-dir",
                            str(input_dir),
                            "--output-dir",
                            str(output_dir),
                        ]
                    )

                self.assertEqual(return_code, 2)
                self.assertIn("ratio", output.getvalue())
                if preexisting:
                    self.assertEqual(
                        {path.name for path in output_dir.iterdir()}, {"index.html"}
                    )
                    self.assertEqual(
                        (output_dir / "index.html").read_text(), "old report"
                    )
                else:
                    self.assertFalse(output_dir.exists())

    def test_duplicate_semantic_slots_within_and_across_files_are_rejected(self):
        for across_files in (False, True):
            with (
                self.subTest(across_files=across_files),
                tempfile.TemporaryDirectory(dir=TOOLS_DIR) as temporary,
            ):
                root = Path(temporary)
                input_dir = root / "input"
                output_dir = root / "output"
                input_dir.mkdir()
                base = {
                    "group": "real_f64",
                    "op": "ddot",
                    "variant": "default",
                    "library": "Zynum",
                    "n": "4096",
                    "metric": "rate_gops",
                    "metric_median": "4",
                    "rate_gops": "4",
                    "status": "ok",
                    "check_status": "checked-ok",
                }
                if across_files:
                    write_rows(input_dir / "first.csv", [base])
                    write_rows(
                        input_dir / "second.csv",
                        [dict(base, library="zynum-blas")],
                    )
                else:
                    write_rows(input_dir / "rows.csv", [base, dict(base)])

                with self.assertRaisesRegex(ValueError, "duplicate semantic slot"):
                    report.render_report(input_dir, output_dir)

                self.assertFalse(output_dir.exists())

    def test_comparator_ties_are_stable_under_library_shuffle(self):
        libraries = (
            ("Zynum", 2.0),
            ("OpenBLAS", 1.0),
            ("AOCL-BLIS", 1.0),
        )
        for category in ("level1", "scalar-latency"):
            reference = None
            for shuffled in itertools.permutations(libraries):
                case_id = f"{category}:tie"
                groups = {
                    case_id: {
                        "case_id": case_id,
                        "case": "tie",
                        "metric": (
                            "ns_per_call"
                            if category == "scalar-latency"
                            else "rate_gops"
                        ),
                        "libraries": {library: [value] for library, value in shuffled},
                        "sources": {"tie.csv"},
                    }
                }

                result = report.aggregate_category(category, groups)[0]

                self.assertEqual(result["fastest_comparator"], "AOCL-BLIS")
                if reference is None:
                    reference = result
                else:
                    self.assertEqual(result, reference)

    def test_invalid_raw_evidence_fails_cli_and_preserves_old_output(self):
        with tempfile.TemporaryDirectory(dir=TOOLS_DIR) as temporary:
            root = Path(temporary)
            input_dir = root / "input"
            output_dir = root / "output"
            input_dir.mkdir()
            output_dir.mkdir()
            (output_dir / "index.html").write_text("old report")
            (output_dir / "keep.txt").write_text("unrelated")
            write_rows(
                input_dir / "level1.csv",
                [
                    {
                        "group": "real_f64",
                        "op": "ddot",
                        "variant": "default",
                        "library": "Zynum",
                        "n": "4096",
                        "metric": "rate_gops",
                        "metric_median": "nan",
                        "rate_gops": "4",
                        "status": "ok",
                        "check_status": "checked-ok",
                    }
                ],
            )

            with redirect_stdout(io.StringIO()) as output:
                return_code = report.main(
                    [
                        "--input-dir",
                        str(input_dir),
                        "--output-dir",
                        str(output_dir),
                    ]
                )

            self.assertEqual(return_code, 2)
            self.assertIn("invalid level1 evidence", output.getvalue())
            self.assertEqual(
                {path.name for path in output_dir.iterdir()},
                {"index.html", "keep.txt"},
            )
            self.assertEqual((output_dir / "index.html").read_text(), "old report")
            self.assertEqual((output_dir / "keep.txt").read_text(), "unrelated")

    def test_all_supplied_performance_fields_require_positive_finite_values(self):
        fields_by_category = {
            "level1": ("seconds", "rate_gops", "bandwidth_gbps"),
            "scalar-latency": (
                "best_ns_per_call",
                "median_ns_per_call",
                "p95_ns_per_call",
                "max_ns_per_call",
                "median_full_ns_per_call",
                "median_harness_ns_per_call",
            ),
            "level2": ("time_ns", "rate_gops"),
            "gemm": ("gflops", "best_ns", "median_ns", "p95_ns", "max_ns"),
            "rank-k": (
                "best_ns",
                "median_ns",
                "p95_ns",
                "max_ns",
                "gflops",
                "median_gflops",
            ),
            "symm-hemm": (
                "best_ns",
                "median_ns",
                "p95_ns",
                "max_ns",
                "gflops",
                "median_gflops",
            ),
            "trmm-trsm": (
                "best_ns",
                "median_ns",
                "p95_ns",
                "max_ns",
                "gflops",
                "median_gflops",
            ),
        }
        for category, fields in fields_by_category.items():
            for field in fields:
                with (
                    self.subTest(category=category, field=field),
                    tempfile.TemporaryDirectory(dir=TOOLS_DIR) as temporary,
                ):
                    root = Path(temporary)
                    input_dir = root / "input"
                    output_dir = root / "output"
                    input_dir.mkdir()
                    write_rows(
                        input_dir / "evidence.csv",
                        [dict(valid_evidence_row(category), **{field: "nan"})],
                    )

                    with self.assertRaisesRegex(ValueError, field):
                        report.render_report(input_dir, output_dir)

                    self.assertFalse(output_dir.exists())

    def test_invalid_aggregate_samples_and_unused_fields_preserve_old_output(self):
        cases = (
            ("level1", "metric_min", "-1"),
            ("level1", "metric_max", "nan"),
            ("level1", "metric_samples", "1,0,3"),
            ("level1", "metric_samples", "1,nan,3"),
            ("level1", "rate_gops", "nan"),
            ("rank-k", "rate_gops", "nan"),
        )
        for category, field, value in cases:
            with (
                self.subTest(category=category, field=field, value=value),
                tempfile.TemporaryDirectory(dir=TOOLS_DIR) as temporary,
            ):
                root = Path(temporary)
                input_dir = root / "input"
                output_dir = root / "output"
                input_dir.mkdir()
                output_dir.mkdir()
                (output_dir / "index.html").write_text("old report")
                (output_dir / "keep.txt").write_text("unrelated")
                row = dict(valid_evidence_row(category), **{field: value})
                if category == "level1" and field == "rate_gops":
                    row.update(metric="bandwidth_gbps", bandwidth_gbps="4")
                write_rows(input_dir / "evidence.csv", [row])

                with redirect_stdout(io.StringIO()) as output:
                    return_code = report.main(
                        [
                            "--input-dir",
                            str(input_dir),
                            "--output-dir",
                            str(output_dir),
                        ]
                    )

                self.assertEqual(return_code, 2)
                self.assertIn(field, output.getvalue())
                self.assertEqual(
                    {path.name for path in output_dir.iterdir()},
                    {"index.html", "keep.txt"},
                )
                self.assertEqual((output_dir / "index.html").read_text(), "old report")
                self.assertEqual((output_dir / "keep.txt").read_text(), "unrelated")

    def test_invalid_derived_metric_fails_before_publishing(self):
        with tempfile.TemporaryDirectory(dir=TOOLS_DIR) as temporary:
            root = Path(temporary)
            input_dir = root / "input"
            output_dir = root / "output"
            input_dir.mkdir()
            output_dir.mkdir()
            (output_dir / "index.html").write_text("old report")
            (output_dir / "keep.txt").write_text("unrelated")
            row = dict(
                valid_evidence_row("gemm"),
                m="1e308",
                n="1e308",
                k="1e308",
            )
            write_rows(input_dir / "gemm.csv", [row])

            with redirect_stdout(io.StringIO()) as output:
                return_code = report.main(
                    [
                        "--input-dir",
                        str(input_dir),
                        "--output-dir",
                        str(output_dir),
                    ]
                )

            self.assertEqual(return_code, 2)
            self.assertIn("gemm derived metric", output.getvalue())
            self.assertEqual(
                {path.name for path in output_dir.iterdir()},
                {"index.html", "keep.txt"},
            )
            self.assertEqual((output_dir / "index.html").read_text(), "old report")
            self.assertEqual((output_dir / "keep.txt").read_text(), "unrelated")

    def test_legacy_rows_without_aggregate_columns_remain_valid(self):
        with tempfile.TemporaryDirectory(dir=TOOLS_DIR) as temporary:
            root = Path(temporary)
            input_dir = root / "input"
            output_dir = root / "output"
            input_dir.mkdir()
            base = valid_evidence_row("level1")
            base.pop("metric_median")
            write_rows(
                input_dir / "legacy.csv",
                [
                    dict(base, library="Zynum", rate_gops="5"),
                    dict(base, library="OpenBLAS", rate_gops="4"),
                ],
            )

            with mock.patch.object(
                report, "publish_outputs", publish_outputs_for_rendering_tests
            ):
                rendered = report.render_report(input_dir, output_dir)

            level1 = rendered["categories"][0]
            self.assertEqual(level1["rows"]["accepted"], 2)
            self.assertEqual(level1["results"][0]["zynum_value"], 5.0)
            self.assertEqual(level1["results"][0]["comparator_value"], 4.0)

    def test_missing_categories_and_missing_comparator_are_explicit(self):
        with tempfile.TemporaryDirectory(dir=TOOLS_DIR) as temporary:
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

            with mock.patch.object(
                report, "publish_outputs", publish_outputs_for_rendering_tests
            ):
                rendered = report.render_report(input_dir, output_dir)
            by_category = {item["id"]: item for item in rendered["categories"]}
            self.assertEqual(
                by_category["level1"]["results"][0]["status"],
                "missing-comparator",
            )
            self.assertEqual(by_category["level1"]["status"], "missing")
            for category in report.CATEGORY_ORDER[1:]:
                self.assertEqual(by_category[category]["status"], "missing")
                self.assertIn(
                    "missing", (output_dir / report.SVG_NAMES[category]).read_text()
                )

            with (output_dir / "summary.csv").open(newline="") as file_handle:
                rows = list(csv.DictReader(file_handle))
            self.assertEqual(len(rows), len(report.CATEGORY_ORDER))
            self.assertEqual(
                next(row for row in rows if row["category"] == "gemm")["status"],
                "missing",
            )

    def test_expected_process_repeats_fail_on_incomplete_rows(self):
        with tempfile.TemporaryDirectory(dir=TOOLS_DIR) as temporary:
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

            with self.assertRaisesRegex(ValueError, "successful_repeats=2"):
                report.render_report(input_dir, output_dir, ["OpenBLAS"], 3)

            self.assertFalse(output_dir.exists())

    def test_level1_repeat_fields_must_be_paired_without_publishing(self):
        for present_field in ("process_repeats", "successful_repeats"):
            with (
                self.subTest(present_field=present_field),
                tempfile.TemporaryDirectory(dir=TOOLS_DIR) as temporary,
            ):
                root = Path(temporary)
                input_dir = root / "input"
                output_dir = root / "output"
                input_dir.mkdir()
                output_dir.mkdir()
                (output_dir / "index.html").write_text("old report")
                (output_dir / "keep.txt").write_text("unrelated")
                row = dict(valid_evidence_row("level1"), **{present_field: "3"})
                write_rows(input_dir / "level1_repeats.csv", [row])

                with redirect_stdout(io.StringIO()) as output:
                    return_code = report.main(
                        [
                            "--input-dir",
                            str(input_dir),
                            "--output-dir",
                            str(output_dir),
                        ]
                    )

                self.assertEqual(return_code, 2)
                self.assertIn("must either both be present", output.getvalue())
                self.assertEqual(
                    {path.name for path in output_dir.iterdir()},
                    {"index.html", "keep.txt"},
                )
                self.assertEqual((output_dir / "index.html").read_text(), "old report")
                self.assertEqual((output_dir / "keep.txt").read_text(), "unrelated")

    def test_level1_repeat_fields_require_equal_canonical_positive_integers(self):
        invalid = (
            ("mismatch", "3", "2", "successful_repeats=2"),
            ("process-zero", "0", "1", "process_repeats"),
            ("successful-zero", "1", "0", "successful_repeats"),
            ("process-leading-zero", "03", "3", "process_repeats"),
            ("successful-leading-zero", "3", "03", "successful_repeats"),
            ("process-decimal", "3.0", "3", "process_repeats"),
            ("successful-signed", "3", "+3", "successful_repeats"),
        )
        for name, process_repeats, successful_repeats, message in invalid:
            with (
                self.subTest(name=name),
                tempfile.TemporaryDirectory(dir=TOOLS_DIR) as temporary,
            ):
                root = Path(temporary)
                input_dir = root / "input"
                output_dir = root / "output"
                input_dir.mkdir()
                output_dir.mkdir()
                (output_dir / "index.html").write_text("old report")
                row = dict(
                    valid_evidence_row("level1"),
                    process_repeats=process_repeats,
                    successful_repeats=successful_repeats,
                )
                write_rows(input_dir / "level1_repeats.csv", [row])

                with redirect_stdout(io.StringIO()) as output:
                    return_code = report.main(
                        [
                            "--input-dir",
                            str(input_dir),
                            "--output-dir",
                            str(output_dir),
                        ]
                    )

                self.assertEqual(return_code, 2)
                self.assertIn(message, output.getvalue())
                self.assertEqual(
                    {path.name for path in output_dir.iterdir()}, {"index.html"}
                )
                self.assertEqual((output_dir / "index.html").read_text(), "old report")

    def test_level1_equal_repeats_and_empty_legacy_fields_are_accepted(self):
        for name, repeat_fields, expected in (
            ("equal", {"process_repeats": "3", "successful_repeats": "3"}, 3),
            ("empty-legacy", {"process_repeats": "", "successful_repeats": ""}, None),
        ):
            with (
                self.subTest(name=name),
                tempfile.TemporaryDirectory(dir=TOOLS_DIR) as temporary,
            ):
                root = Path(temporary)
                input_dir = root / "input"
                output_dir = root / "output"
                input_dir.mkdir()
                row = dict(valid_evidence_row("level1"), **repeat_fields)
                write_rows(input_dir / "level1_repeats.csv", [row])

                with mock.patch.object(
                    report, "publish_outputs", publish_outputs_for_rendering_tests
                ):
                    rendered = report.render_report(
                        input_dir, output_dir, expected_process_repeats=expected
                    )

                level1 = rendered["categories"][0]
                self.assertEqual(level1["rows"]["accepted"], 1)
                self.assertEqual(level1["rows"]["rejected"], 0)

    def test_non_level1_matching_evidence_cardinality_is_accepted_without_expected_count(
        self,
    ):
        with tempfile.TemporaryDirectory(dir=TOOLS_DIR) as temporary:
            root = Path(temporary)
            input_dir = root / "input"
            output_dir = root / "output"
            input_dir.mkdir()
            base = dict(
                valid_evidence_row("level2"),
                process_repeats="3",
                successful_repeats="3",
                metric_samples="3,4,5",
            )
            write_rows(
                input_dir / "level2.csv",
                [
                    dict(base, library="Zynum", metric_median="4"),
                    dict(base, library="OpenBLAS", metric_median="5"),
                ],
            )

            with mock.patch.object(
                report, "publish_outputs", publish_outputs_for_rendering_tests
            ):
                rendered = report.render_report(input_dir, output_dir)

            level2 = next(
                category
                for category in rendered["categories"]
                if category["id"] == "level2"
            )
            self.assertEqual(level2["rows"]["accepted"], 2)
            self.assertEqual(level2["rows"]["rejected"], 0)

    def test_non_level1_case_ids_exclude_evidence_cardinality_fields(self):
        semantic_changes = {
            "scalar-latency": {"case": "scaled"},
            "level2": {"trans": "T"},
            "gemm": {"transa": "T"},
            "rank-k": {"uplo": "L"},
            "symm-hemm": {"uplo": "L"},
            "trmm-trsm": {"uplo": "L"},
        }
        evidence_changes = {
            "process_repeats": "2",
            "successful_repeats": "2",
            "metric_samples": "3,4",
        }
        for category in report.CATEGORY_ORDER[1:]:
            with self.subTest(category=category):
                reader = report.CASE_READERS[category]
                base = valid_evidence_row(category)
                case_id = reader(base)[0]

                self.assertEqual(reader(dict(base, **evidence_changes))[0], case_id)
                self.assertNotEqual(
                    reader(dict(base, **semantic_changes[category]))[0], case_id
                )

    def test_structured_cross_library_cardinality_mismatch_fails_before_publish(self):
        for category in ("rank-k", "symm-hemm", "trmm-trsm"):
            for interface in ("render_report", "main"):
                with (
                    self.subTest(category=category, interface=interface),
                    tempfile.TemporaryDirectory(dir=TOOLS_DIR) as temporary,
                ):
                    root = Path(temporary)
                    input_dir = root / "input"
                    output_dir = root / "output"
                    input_dir.mkdir()
                    base = valid_evidence_row(category)
                    write_rows(
                        input_dir / f"{category}.csv",
                        [
                            dict(
                                base,
                                library="Zynum",
                                process_repeats="3",
                                successful_repeats="3",
                                metric_samples="3,4,5",
                            ),
                            dict(
                                base,
                                library="OpenBLAS",
                                process_repeats="2",
                                successful_repeats="2",
                                metric_samples="3,4",
                            ),
                        ],
                    )

                    if interface == "render_report":
                        with self.assertRaisesRegex(
                            ValueError, "inconsistent evidence cardinality"
                        ):
                            report.render_report(input_dir, output_dir)
                    else:
                        with redirect_stdout(io.StringIO()) as output:
                            return_code = report.main(
                                [
                                    "--input-dir",
                                    str(input_dir),
                                    "--output-dir",
                                    str(output_dir),
                                ]
                            )
                        self.assertEqual(return_code, 2)
                        self.assertIn(
                            "inconsistent evidence cardinality", output.getvalue()
                        )

                    self.assertFalse(output_dir.exists())

    def test_non_level1_cross_library_repeat_count_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory(dir=TOOLS_DIR) as temporary:
            root = Path(temporary)
            input_dir = root / "input"
            output_dir = root / "output"
            input_dir.mkdir()
            base = dict(valid_evidence_row("level2"), library="Zynum")
            write_rows(
                input_dir / "level2.csv",
                [
                    dict(
                        base,
                        process_repeats="3",
                        successful_repeats="3",
                        metric_samples="3,4,5",
                    ),
                    dict(
                        base,
                        library="OpenBLAS",
                        process_repeats="2",
                        successful_repeats="2",
                        metric_samples="4,5",
                    ),
                ],
            )

            with self.assertRaisesRegex(
                ValueError, "inconsistent evidence cardinality"
            ):
                report.render_report(input_dir, output_dir)

            self.assertFalse(output_dir.exists())

    def test_non_level1_metric_sample_count_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory(dir=TOOLS_DIR) as temporary:
            root = Path(temporary)
            input_dir = root / "input"
            output_dir = root / "output"
            input_dir.mkdir()
            row = dict(
                valid_evidence_row("level2"),
                process_repeats="3",
                successful_repeats="3",
                metric_samples="3,4",
            )
            write_rows(input_dir / "level2.csv", [row])

            with self.assertRaisesRegex(ValueError, "metric_samples count=2"):
                report.render_report(input_dir, output_dir)

            self.assertFalse(output_dir.exists())

    def test_partial_non_level1_cardinality_evidence_is_rejected(self):
        with tempfile.TemporaryDirectory(dir=TOOLS_DIR) as temporary:
            root = Path(temporary)
            input_dir = root / "input"
            output_dir = root / "output"
            input_dir.mkdir()
            row = dict(valid_evidence_row("level2"), process_repeats="3")
            write_rows(input_dir / "level2.csv", [row])

            with self.assertRaisesRegex(
                ValueError, "must all be present when cardinality evidence is declared"
            ):
                report.render_report(input_dir, output_dir)

            self.assertFalse(output_dir.exists())

    def test_non_level1_row_without_cardinality_fields_remains_legacy_compatible(self):
        with tempfile.TemporaryDirectory(dir=TOOLS_DIR) as temporary:
            root = Path(temporary)
            input_dir = root / "input"
            output_dir = root / "output"
            input_dir.mkdir()
            write_rows(input_dir / "level2.csv", [valid_evidence_row("level2")])

            with mock.patch.object(
                report, "publish_outputs", publish_outputs_for_rendering_tests
            ):
                rendered = report.render_report(input_dir, output_dir)

            level2 = next(
                category
                for category in rendered["categories"]
                if category["id"] == "level2"
            )
            self.assertEqual(level2["rows"]["accepted"], 1)
            self.assertEqual(level2["rows"]["rejected"], 0)


if __name__ == "__main__":
    unittest.main()
