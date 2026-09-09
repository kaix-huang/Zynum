#!/usr/bin/env python3
# Copyright (C) 2026 Zynum contributors
# SPDX-License-Identifier: LGPL-3.0-or-later

import csv
import errno
import hashlib
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import plot_gemm_sweep
import plot_level1_report
import plot_level2_report
import report_publication

INVALID_PERFORMANCE_VALUES = (
    "NaN",
    "Infinity",
    "-Infinity",
    "0",
    "-1",
    "not-a-number",
)
MAXIMUM_FINITE_TEXT = str(sys.float_info.max)


def write_rows(path, fieldnames, rows):
    with path.open("w", newline="", encoding="utf-8") as file_handle:
        writer = csv.DictWriter(file_handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def digest(contents):
    return hashlib.sha256(contents).hexdigest()


def write_level1_fixture(path, values=(("Zynum", "9.5"), ("OpenBLAS", "10.25"))):
    fields = (
        "group",
        "op",
        "variant",
        "incx",
        "incy",
        "library",
        "n",
        "copy_bytes",
        "seconds",
        "metric",
        "status",
        "bandwidth_gbps",
        "rate_gops",
    )
    rows = []
    for library, value in values:
        rows.append(
            {
                "group": "real_f64",
                "op": "ddot",
                "variant": "stride2",
                "incx": "2",
                "incy": "2",
                "library": library,
                "n": "4096",
                "copy_bytes": "",
                "seconds": "1",
                "metric": "rate_gops",
                "status": "ok",
                "bandwidth_gbps": "",
                "rate_gops": value,
            }
        )
    write_rows(path, fields, rows)


class ReportPlotterPublicationTest(unittest.TestCase):
    def assert_outputs_are_finite(self, outputs):
        for output in outputs:
            contents = output.contents.lower()
            self.assertNotIn(b"nan", contents)
            self.assertNotIn(b"inf", contents)

    def assert_median_statistic(self, module, csv_path, argv):
        with csv_path.open(newline="") as handle:
            reader = csv.DictReader(handle)
            fields = list(reader.fieldnames)
            rows = list(reader)
        is_gemm = module is plot_gemm_sweep
        field = "median_ns" if is_gemm else "metric_median"
        median_argv = [*argv, "--stat", "median"]

        # Old CSVs without median evidence must never silently fall back to best.
        with (
            mock.patch.object(module, "publish_outputs") as publish,
            self.assertRaisesRegex(ValueError, field + " must be"),
        ):
            module.main(median_argv)
        publish.assert_not_called()

        fields.append(field)
        for row in rows:
            row[field] = "512" if is_gemm else "3.25"
        write_rows(csv_path, fields, rows)
        if is_gemm:
            for kind, expected in (
                ("sgemm", 16),
                ("dgemm", 16),
                ("cgemm", 64),
                ("zgemm", 64),
            ):
                evidence = dict(rows[0], kind=kind, m="16", n="16", k="16")
                self.assertEqual(module.row_gflops(evidence, "median"), expected)
                self.assertEqual(module.row_gflops(evidence), float(evidence["gflops"]))
        else:
            key = "value" if module is plot_level1_report else "rate_gops"
            self.assertTrue(
                all(row[key] == 3.25 for row in module.read_rows(csv_path, "median"))
            )
            self.assertNotEqual(module.read_rows(csv_path)[0][key], 3.25)
        published = []
        with mock.patch.object(module, "publish_outputs", side_effect=published.extend):
            module.main(median_argv)
        self.assertTrue(published)
        for output in published:
            self.assertIn(
                b"Median-time" if is_gemm else b"Statistic: median", output.contents
            )
        self.assert_outputs_are_finite(published)

        for value in (*INVALID_PERFORMANCE_VALUES, ""):
            with self.subTest(stat="median", value=value):
                rows[0][field] = value
                write_rows(csv_path, fields, rows)
                with (
                    mock.patch.object(module, "publish_outputs") as publish,
                    self.assertRaisesRegex(ValueError, field + " must be"),
                ):
                    module.main(median_argv)
                publish.assert_not_called()

    def test_gemm_publishes_one_exact_output_in_one_call(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temporary:
            root = Path(temporary)
            csv_path = root / "gemm.csv"
            output_path = root / "nested" / "gemm.svg"
            fields = (
                "kind",
                "shape_index",
                "label",
                "m",
                "n",
                "k",
                "library",
                "gflops",
                "check",
            )
            rows = []
            for shape_index, label, m, n, k in (
                (7, "skinny", 32, 8, 16),
                (3, "square", 16, 16, 16),
            ):
                rows.extend(
                    [
                        {
                            "kind": "sgemm",
                            "shape_index": shape_index,
                            "label": label,
                            "m": m,
                            "n": n,
                            "k": k,
                            "library": "zynum-blas",
                            "gflops": "12.5",
                            "check": "checked-ok",
                        },
                        {
                            "kind": "sgemm",
                            "shape_index": shape_index,
                            "label": label,
                            "m": m,
                            "n": n,
                            "k": k,
                            "library": "OpenBLAS",
                            "gflops": "10.25",
                            "check": "checked-ok",
                        },
                    ]
                )
            write_rows(csv_path, fields, rows)
            published = []

            with (
                mock.patch.object(
                    plot_gemm_sweep,
                    "publish_outputs",
                    side_effect=lambda outputs: published.append(tuple(outputs)),
                ) as publish,
                mock.patch.object(
                    sys,
                    "argv",
                    ["plot_gemm_sweep.py", str(csv_path), str(output_path)],
                ),
            ):
                self.assertEqual(plot_gemm_sweep.main(), 0)

            publish.assert_called_once()
            self.assertEqual(len(published), 1)
            self.assertEqual([item.path for item in published[0]], [output_path])
            contents = published[0][0].contents
            self.assertIsInstance(contents, bytes)
            self.assertEqual(
                digest(contents),
                "d8c29f308e82a75c43d3b954011a03e7bd2ecbb0049520f2ad531e8b2f54cac9",
            )
            self.assertFalse(output_path.parent.exists())
            self.assert_median_statistic(
                plot_gemm_sweep, csv_path, [str(csv_path), str(output_path)]
            )

    def test_gemm_rejects_non_positive_or_non_finite_checked_evidence(self):
        fields = (
            "kind",
            "shape_index",
            "label",
            "m",
            "n",
            "k",
            "library",
            "gflops",
            "check",
        )
        for value in INVALID_PERFORMANCE_VALUES:
            with (
                self.subTest(value=value),
                tempfile.TemporaryDirectory(dir=Path.cwd()) as temporary,
            ):
                root = Path(temporary)
                csv_path = root / "gemm.csv"
                output_path = root / "gemm.svg"
                old_contents = b"old gemm output\n"
                output_path.write_bytes(old_contents)
                write_rows(
                    csv_path,
                    fields,
                    [
                        {
                            "kind": "sgemm",
                            "shape_index": "0",
                            "label": "square",
                            "m": "16",
                            "n": "16",
                            "k": "16",
                            "library": "zynum-blas",
                            "gflops": value,
                            "check": "checked-ok",
                        }
                    ],
                )

                with (
                    mock.patch.object(plot_gemm_sweep, "publish_outputs") as publish,
                    mock.patch.object(
                        sys,
                        "argv",
                        ["plot_gemm_sweep.py", str(csv_path), str(output_path)],
                    ),
                    self.assertRaisesRegex(ValueError, "gflops must be"),
                ):
                    plot_gemm_sweep.main()

                publish.assert_not_called()
                self.assertEqual(output_path.read_bytes(), old_contents)

    def test_gemm_renders_maximum_finite_evidence_with_finite_coordinates(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temporary:
            root = Path(temporary)
            csv_path = root / "gemm.csv"
            output_path = root / "gemm.svg"
            fields = (
                "kind",
                "shape_index",
                "label",
                "m",
                "n",
                "k",
                "library",
                "gflops",
                "check",
            )
            rows = []
            for shape_index, size in enumerate((16, 32)):
                rows.append(
                    {
                        "kind": "sgemm",
                        "shape_index": str(shape_index),
                        "label": "square",
                        "m": str(size),
                        "n": str(size),
                        "k": str(size),
                        "library": "zynum-blas",
                        "gflops": MAXIMUM_FINITE_TEXT,
                        "check": "checked-ok",
                    }
                )
            write_rows(csv_path, fields, rows)
            published = []

            with (
                mock.patch.object(
                    plot_gemm_sweep,
                    "publish_outputs",
                    side_effect=lambda outputs: published.extend(outputs),
                ) as publish,
                mock.patch.object(
                    sys,
                    "argv",
                    ["plot_gemm_sweep.py", str(csv_path), str(output_path)],
                ),
            ):
                self.assertEqual(plot_gemm_sweep.main(), 0)

            publish.assert_called_once()
            self.assert_outputs_are_finite(published)

    def test_level1_publishes_two_exact_outputs_in_one_ordered_call(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temporary:
            root = Path(temporary)
            csv_path = root / "level1.csv"
            bars_path = root / "nested" / "level1.svg"
            ratio_path = root / "nested" / "ratio.svg"
            write_level1_fixture(csv_path)
            published = []

            with mock.patch.object(
                plot_level1_report,
                "publish_outputs",
                side_effect=lambda outputs: published.append(tuple(outputs)),
            ) as publish:
                plot_level1_report.main(
                    [
                        str(csv_path),
                        "--bars-svg",
                        str(bars_path),
                        "--ratio-svg",
                        str(ratio_path),
                    ]
                )

            publish.assert_called_once()
            self.assertEqual(len(published), 1)
            self.assertEqual(
                [item.path for item in published[0]], [bars_path, ratio_path]
            )
            self.assertEqual(
                [digest(item.contents) for item in published[0]],
                [
                    "7ec6a988f56149dfd64a1b894995981f382af279fdc47183b26320a219bb2d36",
                    "299c6dd765359cf3b49167cfabd14990f04e698feb3a930f224908a1bcefc1a4",
                ],
            )
            self.assertTrue(
                all(isinstance(item.contents, bytes) for item in published[0])
            )
            self.assertFalse(bars_path.parent.exists())
            self.assert_median_statistic(
                plot_level1_report,
                csv_path,
                [
                    str(csv_path),
                    "--bars-svg",
                    str(bars_path),
                    "--ratio-svg",
                    str(ratio_path),
                ],
            )

    def test_level1_rejects_non_positive_or_non_finite_ok_evidence(self):
        for value in INVALID_PERFORMANCE_VALUES:
            with (
                self.subTest(value=value),
                tempfile.TemporaryDirectory(dir=Path.cwd()) as temporary,
            ):
                root = Path(temporary)
                csv_path = root / "level1.csv"
                bars_path = root / "bars.svg"
                ratio_path = root / "ratio.svg"
                old_bars = b"old level1 bars\n"
                old_ratio = b"old level1 ratio\n"
                bars_path.write_bytes(old_bars)
                ratio_path.write_bytes(old_ratio)
                write_level1_fixture(csv_path, (("Zynum", value),))

                with (
                    mock.patch.object(plot_level1_report, "publish_outputs") as publish,
                    self.assertRaisesRegex(ValueError, "rate_gops must be"),
                ):
                    plot_level1_report.main(
                        [
                            str(csv_path),
                            "--bars-svg",
                            str(bars_path),
                            "--ratio-svg",
                            str(ratio_path),
                        ]
                    )

                publish.assert_not_called()
                self.assertEqual(bars_path.read_bytes(), old_bars)
                self.assertEqual(ratio_path.read_bytes(), old_ratio)

    def test_level1_rejects_overflow_and_underflow_ratios_before_render(self):
        for zynum, comparator in (
            ("1e308", "1e-308"),
            ("1e-308", "1e308"),
        ):
            with (
                self.subTest(zynum=zynum, comparator=comparator),
                tempfile.TemporaryDirectory(dir=Path.cwd()) as temporary,
            ):
                root = Path(temporary)
                csv_path = root / "level1.csv"
                bars_path = root / "bars.svg"
                ratio_path = root / "ratio.svg"
                old_bars = b"old level1 bars\n"
                old_ratio = b"old level1 ratio\n"
                bars_path.write_bytes(old_bars)
                ratio_path.write_bytes(old_ratio)
                write_level1_fixture(
                    csv_path,
                    (("Zynum", zynum), ("OpenBLAS", comparator)),
                )

                with (
                    mock.patch.object(plot_level1_report, "render_bars") as bars,
                    mock.patch.object(plot_level1_report, "render_ratio") as ratio,
                    mock.patch.object(plot_level1_report, "publish_outputs") as publish,
                    self.assertRaisesRegex(ValueError, "ratio must be"),
                ):
                    plot_level1_report.main(
                        [
                            str(csv_path),
                            "--bars-svg",
                            str(bars_path),
                            "--ratio-svg",
                            str(ratio_path),
                        ]
                    )

                bars.assert_not_called()
                ratio.assert_not_called()
                publish.assert_not_called()
                self.assertEqual(bars_path.read_bytes(), old_bars)
                self.assertEqual(ratio_path.read_bytes(), old_ratio)

    def test_level1_renders_maximum_finite_evidence_with_finite_coordinates(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temporary:
            root = Path(temporary)
            csv_path = root / "level1.csv"
            bars_path = root / "bars.svg"
            ratio_path = root / "ratio.svg"
            write_level1_fixture(
                csv_path,
                (
                    ("Zynum", MAXIMUM_FINITE_TEXT),
                    ("OpenBLAS", MAXIMUM_FINITE_TEXT),
                ),
            )
            published = []

            with mock.patch.object(
                plot_level1_report,
                "publish_outputs",
                side_effect=lambda outputs: published.extend(outputs),
            ) as publish:
                plot_level1_report.main(
                    [
                        str(csv_path),
                        "--bars-svg",
                        str(bars_path),
                        "--ratio-svg",
                        str(ratio_path),
                    ]
                )

            publish.assert_called_once()
            self.assertEqual(len(published), 2)
            self.assert_outputs_are_finite(published)

    def test_level1_rejects_same_and_canonical_colliding_destinations(self):
        for collision in ("same", "absolute-relative"):
            with (
                self.subTest(collision=collision),
                tempfile.TemporaryDirectory(dir=Path.cwd()) as temporary,
            ):
                root = Path(temporary)
                csv_path = root / "level1.csv"
                output_path = root / "level1.svg"
                write_level1_fixture(csv_path)
                ratio_path = (
                    output_path
                    if collision == "same"
                    else Path(os.path.relpath(output_path, Path.cwd()))
                )

                with self.assertRaisesRegex(ValueError, "duplicate report destination"):
                    plot_level1_report.main(
                        [
                            str(csv_path),
                            "--bars-svg",
                            str(output_path),
                            "--ratio-svg",
                            str(ratio_path),
                        ]
                    )

                self.assertFalse(output_path.exists())
                self.assertEqual({path.name for path in root.iterdir()}, {"level1.csv"})

    def test_level1_second_serializer_failure_precedes_publication(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temporary:
            root = Path(temporary)
            csv_path = root / "level1.csv"
            bars_path = root / "nested" / "bars.svg"
            ratio_path = root / "nested" / "ratio.svg"
            write_level1_fixture(csv_path)

            with (
                mock.patch.object(
                    plot_level1_report,
                    "render_ratio",
                    side_effect=ValueError("injected ratio serialization failure"),
                ),
                mock.patch.object(plot_level1_report, "publish_outputs") as publish,
                self.assertRaisesRegex(ValueError, "ratio serialization failure"),
            ):
                plot_level1_report.main(
                    [
                        str(csv_path),
                        "--bars-svg",
                        str(bars_path),
                        "--ratio-svg",
                        str(ratio_path),
                    ]
                )

            publish.assert_not_called()
            self.assertFalse(bars_path.parent.exists())

    def test_level1_second_commit_failure_rolls_back_both_outputs(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temporary:
            root = Path(temporary)
            csv_path = root / "level1.csv"
            bars_path = root / "bars.svg"
            ratio_path = root / "ratio.svg"
            write_level1_fixture(csv_path)
            bars_path.write_bytes(b"old bars\n")
            ratio_path.write_bytes(b"old ratio\n")
            replace = report_publication._replace_name
            calls = 0

            def fail_second_replace(parent, source, destination):
                nonlocal calls
                calls += 1
                if calls == 2:
                    raise OSError(errno.EIO, "injected second publication failure")
                return replace(parent, source, destination)

            with (
                mock.patch.object(
                    report_publication,
                    "_replace_name",
                    side_effect=fail_second_replace,
                ),
                self.assertRaises(
                    report_publication.RollbackIndeterminateError
                ) as caught,
            ):
                plot_level1_report.main(
                    [
                        str(csv_path),
                        "--bars-svg",
                        str(bars_path),
                        "--ratio-svg",
                        str(ratio_path),
                    ]
                )

            self.assertEqual(calls, 2)
            self.assertEqual(bars_path.read_bytes(), b"old bars\n")
            self.assertEqual(ratio_path.read_bytes(), b"old ratio\n")
            recovery_paths = caught.exception.recovery_paths
            self.assertTrue(recovery_paths)
            self.assertCountEqual(
                [path.read_bytes() for path in recovery_paths],
                [b"old bars\n", b"old ratio\n"],
            )

    def test_level2_publishes_one_exact_output_in_one_call(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temporary:
            root = Path(temporary)
            csv_path = root / "level2.csv"
            output_path = root / "nested" / "level2.svg"
            fields = ("case", "kind", "library", "n", "rate_gops", "status")
            write_rows(
                csv_path,
                fields,
                [
                    {
                        "case": "dgemv_n",
                        "kind": "f64",
                        "library": "Zynum",
                        "n": "512",
                        "rate_gops": "8.5",
                        "status": "ok",
                    },
                    {
                        "case": "dgemv_n",
                        "kind": "f64",
                        "library": "OpenBLAS",
                        "n": "512",
                        "rate_gops": "9.25",
                        "status": "ok",
                    },
                ],
            )
            published = []

            with mock.patch.object(
                plot_level2_report,
                "publish_outputs",
                side_effect=lambda outputs: published.append(tuple(outputs)),
            ) as publish:
                plot_level2_report.main([str(csv_path), "--bars-svg", str(output_path)])

            publish.assert_called_once()
            self.assertEqual(len(published), 1)
            self.assertEqual([item.path for item in published[0]], [output_path])
            contents = published[0][0].contents
            self.assertIsInstance(contents, bytes)
            self.assertEqual(
                digest(contents),
                "031a793a46a6d758afcece7345fe2f5ffb1ae700071b1855f5c100f24961eff0",
            )
            self.assertFalse(output_path.parent.exists())
            self.assert_median_statistic(
                plot_level2_report,
                csv_path,
                [str(csv_path), "--bars-svg", str(output_path)],
            )

    def test_level2_rejects_non_positive_or_non_finite_ok_evidence(self):
        fields = ("case", "kind", "library", "n", "rate_gops", "status")
        for value in INVALID_PERFORMANCE_VALUES:
            with (
                self.subTest(value=value),
                tempfile.TemporaryDirectory(dir=Path.cwd()) as temporary,
            ):
                root = Path(temporary)
                csv_path = root / "level2.csv"
                output_path = root / "level2.svg"
                old_contents = b"old level2 output\n"
                output_path.write_bytes(old_contents)
                write_rows(
                    csv_path,
                    fields,
                    [
                        {
                            "case": "dgemv_n",
                            "kind": "f64",
                            "library": "Zynum",
                            "n": "512",
                            "rate_gops": value,
                            "status": "ok",
                        }
                    ],
                )

                with (
                    mock.patch.object(plot_level2_report, "publish_outputs") as publish,
                    self.assertRaisesRegex(ValueError, "rate_gops must be"),
                ):
                    plot_level2_report.main(
                        [str(csv_path), "--bars-svg", str(output_path)]
                    )

                publish.assert_not_called()
                self.assertEqual(output_path.read_bytes(), old_contents)

    def test_level2_renders_maximum_finite_evidence_with_finite_coordinates(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temporary:
            root = Path(temporary)
            csv_path = root / "level2.csv"
            output_path = root / "level2.svg"
            fields = ("case", "kind", "library", "n", "rate_gops", "status")
            write_rows(
                csv_path,
                fields,
                [
                    {
                        "case": "dgemv_n",
                        "kind": "f64",
                        "library": "Zynum",
                        "n": "512",
                        "rate_gops": MAXIMUM_FINITE_TEXT,
                        "status": "ok",
                    }
                ],
            )
            published = []

            with mock.patch.object(
                plot_level2_report,
                "publish_outputs",
                side_effect=lambda outputs: published.extend(outputs),
            ) as publish:
                plot_level2_report.main([str(csv_path), "--bars-svg", str(output_path)])

            publish.assert_called_once()
            self.assert_outputs_are_finite(published)


if __name__ == "__main__":
    unittest.main()
