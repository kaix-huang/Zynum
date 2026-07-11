#!/usr/bin/env python3
# Copyright (C) 2026 Zynum contributors
# SPDX-License-Identifier: LGPL-3.0-or-later

"""Render a checked, cross-level Zynum benchmark report.

The input directory may contain the CSV files emitted by the Level 1, scalar
ROTG/ROTMG latency, Level 2, GEMM, rank-k, SYMM/HEMM, and TRMM/TRSM runners.
Only correctness-checked rows are eligible.  The report uses the fresh-process
median whenever the runner provides it and compares Zynum with the fastest
non-Zynum library for each logical case.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median
from typing import Callable, Mapping, TypedDict


CATEGORY_ORDER = (
    "level1",
    "scalar-latency",
    "level2",
    "gemm",
    "rank-k",
    "symm-hemm",
    "trmm-trsm",
)

CATEGORY_TITLES = {
    "level1": "Level 1",
    "scalar-latency": "ROTG / ROTMG Scalar Latency",
    "level2": "Level 2",
    "gemm": "GEMM",
    "rank-k": "Rank-K / Rank-2K",
    "symm-hemm": "SYMM / HEMM",
    "trmm-trsm": "TRMM / TRSM",
}

SVG_NAMES = {category: f"{category}.svg" for category in CATEGORY_ORDER}

CHECK_FIELDS = ("check_status", "check", "correctness_status", "correctness")
ACCEPTED_CHECKS = {
    "ok",
    "passed",
    "sampled",
    "checked-ok",
    "sampled-ok",
    "correctness-ok",
    "correctness-passed",
}

SUMMARY_FIELDS = (
    "category",
    "case_id",
    "case",
    "metric",
    "zynum_value",
    "fastest_comparator",
    "comparator_value",
    "ratio",
    "status",
    "missing_comparators",
    "library_values",
    "source_files",
)

LIBRARY_ORDER = (
    "Zynum",
    "MKL",
    "OpenBLAS",
    "AOCL-BLIS",
    "ATLAS",
    "Upstream-BLIS",
)

LIBRARY_COLORS = {
    "Zynum": "#111827",
    "MKL": "#2563eb",
    "OpenBLAS": "#16a34a",
    "AOCL-BLIS": "#d97706",
    "ATLAS": "#7c3aed",
    "Upstream-BLIS": "#db2777",
}

CsvRow = Mapping[str, str | None]
CaseData = tuple[str, str, str, float]


class RawCase(TypedDict):
    case_id: str
    case: str
    metric: str
    libraries: defaultdict[str, list[float]]
    sources: set[str]


class RowStats(TypedDict):
    seen: int
    accepted: int
    rejected: int
    reasons: Counter[str]


class BenchmarkResult(TypedDict):
    case_id: str
    case: str
    metric: str
    zynum_value: float | None
    fastest_comparator: str | None
    comparator_value: float | None
    ratio: float | None
    status: str
    missing_comparators: list[str]
    source_files: list[str]
    libraries: dict[str, float]


class CaseCounts(TypedDict):
    total: int
    passed: int
    failed: int
    missing: int
    comparator_incomplete: int


class ReportRows(TypedDict):
    seen: int
    accepted: int
    rejected: int
    rejection_reasons: dict[str, int]


class CategoryReport(TypedDict):
    id: str
    title: str
    status: str
    svg: str
    files: list[str]
    rows: ReportRows
    cases: CaseCounts
    results: list[BenchmarkResult]


class InputData(TypedDict):
    csv_files: list[str]
    recognized: list[str]
    ignored: list[str]
    category_files: dict[str, set[str]]
    row_stats: dict[str, RowStats]
    groups: dict[str, dict[str, RawCase]]


class FileSummary(TypedDict):
    scanned: int
    recognized: int
    recognized_paths: list[str]
    ignored_count: int
    ignored_paths: list[str]


class FullReport(TypedDict):
    schema_version: int
    ratio: dict[str, str]
    gate: float
    statistic: str
    expected_process_repeats: int | None
    requested_comparators: list[str]
    files: FileSummary
    categories: list[CategoryReport]


class InvalidRow(ValueError):
    pass


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Render checked cross-level benchmark real-performance charts."
    )
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--comparator",
        action="append",
        default=None,
        help="Canonical comparator label. Repeat to require and restrict the external set.",
    )
    parser.add_argument(
        "--expected-process-repeats",
        type=int,
        default=None,
        help="Reject rows that do not record this fresh-process repeat count.",
    )
    return parser.parse_args(argv)


def classify_header(fieldnames):
    fields = set(fieldnames or ())
    if {
        "kind",
        "transa",
        "transb",
        "shape_index",
        "label",
        "m",
        "n",
        "k",
        "library",
        "gflops",
        "check",
    } <= fields:
        return "gemm"
    if {"group", "op", "variant", "library", "n", "metric", "check_status"} <= fields:
        return "level1"
    if {
        "routine",
        "kind",
        "library",
        "case",
        "median_ns_per_call",
        "metric",
        "check_status",
    } <= fields:
        return "scalar-latency"
    if {"case", "kind", "library", "n", "rate_gops", "check_status"} <= fields:
        return "level2"
    if {
        "routine",
        "family",
        "kind",
        "library",
        "shape",
        "m",
        "n",
        "side",
        "trans",
        "diag",
        "check_status",
    } <= fields:
        return "trmm-trsm"
    if {
        "routine",
        "kind",
        "library",
        "shape",
        "m",
        "n",
        "side",
        "beta_re",
        "check_status",
    } <= fields:
        return "symm-hemm"
    if {
        "routine",
        "kind",
        "library",
        "shape",
        "n",
        "k",
        "uplo",
        "trans",
        "check_status",
    } <= fields:
        return "rank-k"
    return None


def text(row: CsvRow, field: str, default: str = "") -> str:
    value = row.get(field)
    return default if value in (None, "") else str(value).strip()


def normalized_library(value: str) -> str:
    name = value.strip()
    folded = re.sub(r"[^a-z0-9]+", "", name.lower())
    if folded in {"zynum", "zynumblas", "libzynum", "libzynumblas"}:
        return "Zynum"
    return name


def checked_row(row: CsvRow, expected_process_repeats: int | None = None) -> None:
    status = text(row, "status")
    if status and status.lower() != "ok":
        raise InvalidRow(f"status={status}")
    checks = [text(row, field) for field in CHECK_FIELDS if field in row]
    checks = [value for value in checks if value]
    if not checks:
        raise InvalidRow("missing correctness status")
    for value in checks:
        if value.lower() not in ACCEPTED_CHECKS:
            raise InvalidRow(f"correctness={value}")
    process_repeats_text = text(row, "process_repeats")
    successful_repeats_text = text(row, "successful_repeats")
    if not process_repeats_text:
        if expected_process_repeats is not None:
            if not successful_repeats_text:
                raise InvalidRow("missing repeat count")
            try:
                successful_repeats = int(successful_repeats_text)
            except ValueError as exc:
                raise InvalidRow("invalid successful_repeats") from exc
            if successful_repeats != expected_process_repeats:
                raise InvalidRow(
                    f"successful_repeats={successful_repeats}, "
                    f"expected={expected_process_repeats}"
                )
        return
    if process_repeats_text:
        try:
            process_repeats = int(process_repeats_text)
        except ValueError as exc:
            raise InvalidRow("invalid process_repeats") from exc
        if process_repeats <= 0:
            raise InvalidRow("non-positive process_repeats")
        if (
            expected_process_repeats is not None
            and process_repeats != expected_process_repeats
        ):
            raise InvalidRow(
                f"process_repeats={process_repeats}, expected={expected_process_repeats}"
            )
        if successful_repeats_text:
            try:
                successful_repeats = int(successful_repeats_text)
            except ValueError as exc:
                raise InvalidRow("invalid successful_repeats") from exc
            if successful_repeats != process_repeats:
                raise InvalidRow(
                    f"successful_repeats={successful_repeats}, process_repeats={process_repeats}"
                )


def positive_float(value: str | None, field: str) -> float:
    if value is None:
        raise InvalidRow(f"missing {field}")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise InvalidRow(f"invalid {field}") from exc
    if not math.isfinite(result) or result <= 0:
        raise InvalidRow(f"non-positive {field}")
    return result


def median_metric(row: CsvRow, fallback_field: str) -> float:
    value = text(row, "metric_median")
    if value:
        return positive_float(value, "metric_median")
    return positive_float(text(row, fallback_field), fallback_field)


def gemm_metric(row: CsvRow) -> float:
    elapsed = text(row, "median_ns")
    if elapsed:
        elapsed_ns = positive_float(elapsed, "median_ns")
        factor = 8.0 if text(row, "kind").lower() in {"cgemm", "zgemm"} else 2.0
        work = (
            factor
            * positive_float(text(row, "m"), "m")
            * positive_float(text(row, "n"), "n")
            * positive_float(text(row, "k"), "k")
        )
        return work / elapsed_ns
    return positive_float(text(row, "gflops"), "gflops")


def joined_key(category: str, fields: list[str]) -> str:
    return category + ":" + "\x1f".join(fields)


def level1_case(row: CsvRow) -> CaseData:
    variant = text(row, "variant", "default")
    incx = text(row, "incx", "1")
    incy = text(row, "incy", "1")
    n = text(row, "n")
    metric = text(row, "metric")
    fields = [
        text(row, "group"),
        text(row, "op"),
        variant,
        incx,
        incy,
        n,
        metric,
    ]
    label = text(row, "op")
    if variant != "default":
        label += f":{variant}"
    label += f" n={n} inc=({incx},{incy})"
    copy_bytes = text(row, "copy_bytes")
    if copy_bytes:
        label += f" bytes={copy_bytes}"
    return joined_key("level1", fields), label, metric, median_metric(
        row, "bandwidth_gbps" if metric == "bandwidth_gbps" else "rate_gops"
    )


def level2_case(row: CsvRow) -> CaseData:
    n = text(row, "n")
    m = text(row, "m", n)
    shape = text(row, "shape", f"sq{n}")
    operation_fields = [
        text(row, field)
        for field in (
            "storage",
            "lda",
            "k",
            "kl",
            "ku",
            "uplo",
            "trans",
            "diag",
            "incx",
            "incy",
        )
    ]
    metric = text(row, "metric", "gops")
    fields = [text(row, "case"), text(row, "kind"), shape, m, n]
    fields.extend(operation_fields)
    fields.append(metric)
    label = f"{text(row, 'case')} {text(row, 'kind')} {shape} {m}x{n}"
    details = [
        f"{name}={text(row, name)}"
        for name in ("storage", "k", "kl", "ku", "uplo", "trans", "diag")
        if text(row, name)
    ]
    if details:
        label += " " + ",".join(details)
    return joined_key("level2", fields), label, metric, median_metric(row, "rate_gops")


def scalar_latency_case(row: CsvRow) -> CaseData:
    fields = [
        text(row, "routine"),
        text(row, "kind"),
        text(row, "case"),
        text(row, "corpus_size"),
        text(row, "expected_flag"),
        text(row, "metric", "ns_per_call"),
    ]
    label = f"{fields[0]} {fields[2]}"
    if fields[4]:
        try:
            flag_label = f"{float(fields[4]):g}"
        except ValueError:
            flag_label = fields[4]
        label += f" flag={flag_label}"
    return (
        joined_key("scalar-latency", fields),
        label,
        text(row, "metric", "ns_per_call"),
        median_metric(row, "median_ns_per_call"),
    )


def gemm_case(row: CsvRow) -> CaseData:
    fields = [
        text(row, "kind"),
        text(row, "transa", "N"),
        text(row, "transb", "N"),
        text(row, "label"),
        text(row, "m"),
        text(row, "n"),
        text(row, "k"),
    ]
    label = (
        f"{fields[0]} {fields[1]}{fields[2]} {fields[3]} "
        f"{fields[4]}x{fields[5]}x{fields[6]}"
    )
    return joined_key("gemm", fields), label, "gflops", gemm_metric(row)


def rank_k_case(row: CsvRow) -> CaseData:
    key_fields = [
        text(row, field)
        for field in (
            "routine",
            "kind",
            "shape",
            "n",
            "k",
            "uplo",
            "trans",
            "alpha_re",
            "alpha_im",
            "beta_re",
            "beta_im",
            "lda",
            "ldb",
            "ldc",
            "reps",
            "process_repeats",
            "metric",
        )
    ]
    label = (
        f"{text(row, 'routine')} {text(row, 'kind')} {text(row, 'shape')} "
        f"n={text(row, 'n')} k={text(row, 'k')} "
        f"{text(row, 'uplo')}/{text(row, 'trans')}"
    )
    return (
        joined_key("rank-k", key_fields),
        label,
        text(row, "metric", "gflops"),
        median_metric(row, "gflops"),
    )


def symm_case(row: CsvRow) -> CaseData:
    key_fields = [
        text(row, field)
        for field in (
            "routine",
            "kind",
            "shape",
            "m",
            "n",
            "side",
            "uplo",
            "alpha_re",
            "alpha_im",
            "beta_re",
            "beta_im",
            "order",
            "lda",
            "ldb",
            "ldc",
            "reps",
            "process_repeats",
            "metric",
        )
    ]
    label = (
        f"{text(row, 'routine')} {text(row, 'kind')} {text(row, 'shape')} "
        f"{text(row, 'm')}x{text(row, 'n')} "
        f"side={text(row, 'side')} uplo={text(row, 'uplo')}"
    )
    return (
        joined_key("symm-hemm", key_fields),
        label,
        text(row, "metric", "gflops"),
        median_metric(row, "gflops"),
    )


def triangular_case(row: CsvRow) -> CaseData:
    key_fields = [
        text(row, field)
        for field in (
            "routine",
            "family",
            "kind",
            "shape",
            "m",
            "n",
            "side",
            "uplo",
            "trans",
            "diag",
            "alpha_re",
            "alpha_im",
            "order",
            "lda",
            "ldb",
            "reps",
            "process_repeats",
            "metric",
        )
    ]
    label = (
        f"{text(row, 'routine')} {text(row, 'kind')} {text(row, 'shape')} "
        f"{text(row, 'm')}x{text(row, 'n')} side={text(row, 'side')} "
        f"{text(row, 'uplo')}/{text(row, 'trans')}/{text(row, 'diag')}"
    )
    return (
        joined_key("trmm-trsm", key_fields),
        label,
        text(row, "metric", "gflops"),
        median_metric(row, "gflops"),
    )


CASE_READERS: dict[str, Callable[[CsvRow], CaseData]] = {
    "level1": level1_case,
    "scalar-latency": scalar_latency_case,
    "level2": level2_case,
    "gemm": gemm_case,
    "rank-k": rank_k_case,
    "symm-hemm": symm_case,
    "trmm-trsm": triangular_case,
}


def read_inputs(
    input_dir: Path, expected_process_repeats: int | None = None
) -> InputData:
    groups: dict[str, dict[str, RawCase]] = {
        category: {} for category in CATEGORY_ORDER
    }
    category_files: dict[str, set[str]] = {
        category: set() for category in CATEGORY_ORDER
    }
    row_stats: dict[str, RowStats] = {
        category: {"seen": 0, "accepted": 0, "rejected": 0, "reasons": Counter()}
        for category in CATEGORY_ORDER
    }
    csv_files = sorted(path for path in input_dir.rglob("*.csv") if path.is_file())
    recognized: list[str] = []
    ignored: list[str] = []

    for path in csv_files:
        relative = path.relative_to(input_dir).as_posix()
        try:
            file_handle = path.open(newline="", encoding="utf-8-sig")
        except OSError:
            ignored.append(relative)
            continue
        with file_handle:
            reader = csv.DictReader(file_handle)
            category = classify_header(reader.fieldnames)
            if category is None:
                ignored.append(relative)
                continue
            recognized.append(relative)
            category_files[category].add(relative)
            reader_fn = CASE_READERS[category]
            for row in reader:
                row_stats[category]["seen"] += 1
                try:
                    checked_row(row, expected_process_repeats)
                    library = normalized_library(text(row, "library"))
                    if not library:
                        raise InvalidRow("missing library")
                    case_id, label, metric, value = reader_fn(row)
                except (InvalidRow, KeyError, TypeError, ValueError) as exc:
                    row_stats[category]["rejected"] += 1
                    reason = str(exc) or exc.__class__.__name__
                    row_stats[category]["reasons"][reason] += 1
                    continue
                row_stats[category]["accepted"] += 1
                case = groups[category].setdefault(
                    case_id,
                    {
                        "case_id": case_id,
                        "case": label,
                        "metric": metric,
                        "libraries": defaultdict(list),
                        "sources": set(),
                    },
                )
                case["libraries"][library].append(value)
                case["sources"].add(relative)

    return {
        "csv_files": [path.relative_to(input_dir).as_posix() for path in csv_files],
        "recognized": recognized,
        "ignored": ignored,
        "category_files": category_files,
        "row_stats": row_stats,
        "groups": groups,
    }


def aggregate_category(
    category: str,
    groups: dict[str, RawCase],
    requested_comparators: list[str] | None = None,
) -> list[BenchmarkResult]:
    results: list[BenchmarkResult] = []
    for case_id in sorted(groups):
        raw = groups[case_id]
        values = {
            library: median(samples)
            for library, samples in raw["libraries"].items()
        }
        zynum = values.get("Zynum")
        comparators = {name: value for name, value in values.items() if name != "Zynum"}
        missing_comparators = []
        if requested_comparators:
            missing_comparators = [
                name for name in requested_comparators if name not in comparators
            ]
            comparators = {
                name: comparators[name]
                for name in requested_comparators
                if name in comparators
            }
        best_name = ""
        best_value = None
        ratio = None
        lower_is_better = category == "scalar-latency"
        if comparators:
            selector = min if lower_is_better else max
            best_name, best_value = selector(
                comparators.items(), key=lambda item: (item[1], item[0])
            )
        if zynum is None:
            status = "missing-zynum"
        elif best_value is None:
            status = "missing-comparator"
        else:
            ratio = best_value / zynum if lower_is_better else zynum / best_value
            status = "passed" if ratio >= 1.0 else "failed"
        results.append(
            {
                "case_id": case_id,
                "case": raw["case"],
                "metric": raw["metric"],
                "zynum_value": zynum,
                "fastest_comparator": best_name or None,
                "comparator_value": best_value,
                "ratio": ratio,
                "status": status,
                "missing_comparators": missing_comparators,
                "source_files": sorted(raw["sources"]),
                "libraries": dict(sorted(values.items())),
            }
        )
    return results


def category_status(results: list[BenchmarkResult]) -> str:
    if not results or not any(result["ratio"] is not None for result in results):
        return "missing"
    statuses = {result["status"] for result in results}
    if "failed" in statuses:
        return "failed"
    if statuses - {"passed"} or any(
        result["missing_comparators"] for result in results
    ):
        return "incomplete"
    return "passed"


def status_counts(results: list[BenchmarkResult]) -> CaseCounts:
    counts: Counter[str] = Counter(result["status"] for result in results)
    return {
        "total": len(results),
        "passed": counts["passed"],
        "failed": counts["failed"],
        "missing": counts["missing-zynum"] + counts["missing-comparator"],
        "comparator_incomplete": sum(
            bool(result["missing_comparators"]) for result in results
        ),
    }


def shortened(value: str, limit: int = 76) -> str:
    if len(value) <= limit:
        return value
    return value[: limit - 3] + "..."


def ratio_tick(exponent: int) -> str:
    value = 2.0**exponent
    if value >= 1000:
        return f"{value:.0f}x"
    if value >= 1:
        return f"{value:g}x"
    if value >= 0.01:
        return f"{value:.3g}x"
    return f"{value:.1e}x"


def render_svg(
    category: str, results: list[BenchmarkResult], output_path: Path
) -> None:
    width = 1500
    label_right = 510
    chart_left = 540
    chart_width = 720
    value_x = 1280
    row_height = 24
    missing_result: BenchmarkResult = {
        "case_id": "",
        "case": "missing",
        "metric": "",
        "zynum_value": None,
        "fastest_comparator": None,
        "comparator_value": None,
        "ratio": None,
        "status": "missing",
        "missing_comparators": [],
        "source_files": [],
        "libraries": {},
    }
    def format_value(value: float) -> str:
        return f"{value:.4g}"

    def tick_values(max_value: float) -> list[float]:
        if max_value <= 0 or not math.isfinite(max_value):
            return [0.0, 1.0]
        raw = max_value / 5.0
        exponent = math.floor(math.log10(raw))
        base = raw / (10.0**exponent)
        if base <= 1:
            step_base = 1.0
        elif base <= 2:
            step_base = 2.0
        elif base <= 5:
            step_base = 5.0
        else:
            step_base = 10.0
        step = step_base * 10.0**exponent
        top = math.ceil(max_value / step) * step
        values: list[float] = []
        value = 0.0
        while value <= top + step * 0.5:
            values.append(value)
            value += step
        return values

    metric_units = {
        "bandwidth_gbps": "GB/s",
        "gflops": "GFLOPS",
        "gops": "GOPS",
        "ns_per_call": "ns/call",
        "rate_gops": "GOPS",
    }

    metric_groups: list[tuple[str, list[BenchmarkResult]]] = []
    grouped: dict[str, list[BenchmarkResult]] = {}
    for result in results:
        grouped.setdefault(result["metric"], []).append(result)
    metric_groups.extend(grouped.items())
    if not metric_groups:
        metric_groups.append(("", [missing_result]))

    def ordered_libraries(result: BenchmarkResult) -> list[str]:
        return sorted(
            result["libraries"],
            key=lambda name: (
                LIBRARY_ORDER.index(name)
                if name in LIBRARY_ORDER
                else len(LIBRARY_ORDER),
                name,
            ),
        )

    counts = status_counts(results)
    status = category_status(results)
    panel_gap = 36
    base_top = 124
    panel_layout: list[tuple[str, list[BenchmarkResult], float, list[float], int, int, int, int]] = []
    cursor = base_top
    for metric, metric_rows in metric_groups:
        values = [
            value
            for result in metric_rows
            for value in result["libraries"].values()
            if value > 0
        ]
        max_value = max(values, default=1.0)
        ticks = tick_values(max_value)
        max_tick = ticks[-1]
        line_top = cursor + 34
        line_bottom = line_top + 140
        bar_top = line_bottom + 58
        panel_bottom = bar_top + row_height * len(metric_rows) + 32
        panel_layout.append(
            (metric, metric_rows, max_tick, ticks, line_top, line_bottom, bar_top, panel_bottom)
        )
        cursor = panel_bottom + panel_gap
    height = max(300, cursor)

    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        """
<style>
  text { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; fill: #111827; }
  .title { font-size: 25px; font-weight: 700; }
  .subtitle { font-size: 13px; fill: #4b5563; }
  .panel-title { font-size: 16px; font-weight: 700; }
  .case { font-size: 11px; fill: #374151; }
  .value { font-size: 11px; font-variant-numeric: tabular-nums; }
  .tick { font-size: 10px; fill: #6b7280; }
  .legend { font-size: 12px; }
  .grid { stroke: #e5e7eb; stroke-width: 1; }
  .library-bar { opacity: 0.9; }
  .library-line { fill: none; stroke-width: 2.2; stroke-linecap: round; stroke-linejoin: round; }
  .library-line[data-library="Zynum"] { stroke-width: 2.6; }
  .library-line-point { stroke: #ffffff; stroke-width: 0.7; }
</style>
""",
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="36" y="40" class="title">{html.escape(CATEGORY_TITLES[category])}: Real performance values</text>',
        '<text x="36" y="64" class="subtitle">Real metric values in native units; 1.0 is the strict ratio gate in the summary. Fresh-process median; lower is better only for ns/call.</text>',
        f'<text x="36" y="84" class="subtitle">status={status}; cases={counts["total"]}; passed={counts["passed"]}; failed={counts["failed"]}; missing={counts["missing"]}; comparator-incomplete={counts["comparator_incomplete"]}</text>',
    ]

    libraries = sorted(
        {name for result in results for name in result["libraries"]},
        key=lambda name: (
            LIBRARY_ORDER.index(name) if name in LIBRARY_ORDER else len(LIBRARY_ORDER),
            name,
        ),
    )
    legend_x = 40
    for library in libraries:
        color = LIBRARY_COLORS.get(library, "#6b7280")
        svg.append(f'<rect x="{legend_x}" y="99" width="12" height="9" fill="{color}"/>')
        svg.append(
            f'<text x="{legend_x + 18}" y="108" class="legend">{html.escape(library)}</text>'
        )
        legend_x += 38 + 7 * len(library)

    for metric, metric_rows, max_tick, ticks, line_top, line_bottom, bar_top, panel_bottom in panel_layout:
        unit = metric_units.get(metric, metric or "value")
        lower_better = metric == "ns_per_call"
        direction = "lower is better" if lower_better else "higher is better"
        panel_title = f"{unit} ({direction})"
        svg.append(
            f'<text x="36" y="{line_top - 15}" class="panel-title" data-metric="{html.escape(metric)}">{html.escape(panel_title)}</text>'
        )
        svg.append(
            f'<rect x="{chart_left}" y="{line_top}" width="{chart_width}" height="{line_bottom - line_top}" fill="#f9fafb" stroke="#e5e7eb"/>'
        )
        line_height = line_bottom - line_top

        def value_y(value: float) -> float:
            return line_bottom - value / max_tick * line_height

        for tick in ticks:
            y = value_y(tick)
            svg.append(
                f'<line x1="{chart_left}" y1="{y:.1f}" x2="{chart_left + chart_width}" y2="{y:.1f}" class="grid"/>'
            )
            svg.append(
                f'<text x="{chart_left - 8}" y="{y + 3:.1f}" text-anchor="end" class="tick">{format_value(tick)}</text>'
            )
        svg.append(
            f'<line x1="{chart_left}" y1="{line_top}" x2="{chart_left}" y2="{line_bottom}" class="grid"/>'
        )
        svg.append(
            f'<line x1="{chart_left}" y1="{line_bottom}" x2="{chart_left + chart_width}" y2="{line_bottom}" class="grid"/>'
        )
        for library in libraries:
            segments: list[list[tuple[float, float]]] = []
            segment: list[tuple[float, float]] = []
            for index, result in enumerate(metric_rows):
                value = result["libraries"].get(library)
                if value is None or value <= 0:
                    if segment:
                        segments.append(segment)
                        segment = []
                    continue
                point_x = chart_left + chart_width * index / max(1, len(metric_rows) - 1)
                segment.append((point_x, value_y(value)))
            if segment:
                segments.append(segment)
            color = LIBRARY_COLORS.get(library, "#6b7280")
            for points in segments:
                point_text = " ".join(f"{x:.1f},{y:.1f}" for x, y in points)
                svg.append(
                    f'<polyline class="library-line" data-library="{html.escape(library)}" data-metric="{html.escape(metric)}" points="{point_text}" stroke="{color}"><title>{html.escape(library)} {html.escape(unit)} values</title></polyline>'
                )
                for point_x, point_y in points:
                    svg.append(
                        f'<circle class="library-line-point" data-library="{html.escape(library)}" cx="{point_x:.1f}" cy="{point_y:.1f}" r="2.3" fill="{color}"/>'
                    )
        for index, label in ((0, "first"), (len(metric_rows) - 1, "last")):
            if index < 0:
                continue
            point_x = chart_left + chart_width * index / max(1, len(metric_rows) - 1)
            svg.append(
                f'<text x="{point_x:.1f}" y="{line_bottom + 14}" text-anchor="middle" class="tick">{label}</text>'
            )
        svg.append(
            f'<text x="{chart_left}" y="{bar_top - 20}" class="tick">All-library grouped bars by case ({html.escape(unit)})</text>'
        )
        for tick in ticks:
            x = chart_left + tick / max_tick * chart_width
            svg.append(
                f'<line x1="{x:.1f}" y1="{bar_top - 10}" x2="{x:.1f}" y2="{panel_bottom - 20}" class="grid"/>'
            )
            svg.append(
                f'<text x="{x:.1f}" y="{bar_top - 13}" text-anchor="middle" class="tick">{format_value(tick)}</text>'
            )
        for index, result in enumerate(metric_rows):
            y = bar_top + index * row_height
            full_label = result["case"]
            svg.append(
                f'<text x="{label_right}" y="{y + 4}" text-anchor="end" class="case">{html.escape(shortened(full_label))}<title>{html.escape(full_label)}</title></text>'
            )
            for library_index, library in enumerate(ordered_libraries(result)):
                value = result["libraries"].get(library)
                if value is None or value <= 0:
                    continue
                x_end = chart_left + value / max_tick * chart_width
                bar_y = y + (library_index - (len(result["libraries"]) - 1) / 2) * 2.8
                color = LIBRARY_COLORS.get(library, "#6b7280")
                tooltip = f"{library}: value={value:.9g} {unit}"
                svg.append(
                    f'<rect class="library-bar" data-library="{html.escape(library)}" data-metric="{html.escape(metric)}" data-value="{value:.9g}" x="{chart_left:.1f}" y="{bar_y - 1.1:.1f}" width="{max(1.5, x_end - chart_left):.1f}" height="2.2" fill="{color}"><title>{html.escape(tooltip)}</title></rect>'
                )
            zynum_value = result["zynum_value"]
            comparator = result["fastest_comparator"]
            comparator_value = result["comparator_value"]
            if zynum_value is None:
                summary = result["status"]
                summary_color = "#6b7280"
            elif comparator is None or comparator_value is None:
                summary = f"Zynum {format_value(zynum_value)} {unit}"
                summary_color = "#6b7280"
            else:
                summary = f"Zynum {format_value(zynum_value)} | {comparator} {format_value(comparator_value)} | {result['ratio']:.3f}x"
                summary_color = "#15803d" if result["ratio"] is not None and result["ratio"] >= 1.0 else "#dc2626"
            svg.append(
                f'<text x="{value_x}" y="{y + 4}" class="value" fill="{summary_color}">{html.escape(summary)}</text>'
            )
    svg.append("</svg>\n")
    output_path.write_text("\n".join(svg), encoding="utf-8")


def csv_value(value: float | str | None) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return format(value, ".17g")
    return str(value)


def write_summary_csv(categories: list[CategoryReport], output_path: Path) -> None:
    with output_path.open("w", newline="", encoding="utf-8") as file_handle:
        writer = csv.DictWriter(file_handle, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        for category in categories:
            if not category["results"]:
                writer.writerow(
                    {
                        "category": category["id"],
                        "case": "missing",
                        "status": "missing",
                    }
                )
                continue
            for result in category["results"]:
                writer.writerow(
                    {
                        "category": category["id"],
                        "case_id": result["case_id"],
                        "case": result["case"],
                        "metric": result["metric"],
                        "zynum_value": csv_value(result["zynum_value"]),
                        "fastest_comparator": result["fastest_comparator"] or "",
                        "comparator_value": csv_value(result["comparator_value"]),
                        "ratio": csv_value(result["ratio"]),
                        "status": result["status"],
                        "missing_comparators": ";".join(
                            result["missing_comparators"]
                        ),
                        "library_values": json.dumps(
                            result["libraries"], sort_keys=True
                        ),
                        "source_files": ";".join(result["source_files"]),
                    }
                )


def html_number(value: float | None) -> str:
    return "" if value is None else f"{value:.6g}"


def render_index(report: FullReport, output_path: Path) -> None:
    sections: list[str] = []
    for category in report["categories"]:
        counts = category["cases"]
        rows = []
        if not category["results"]:
            rows.append('<tr><td colspan="7" class="missing">missing</td></tr>')
        for result in category["results"]:
            library_values = "; ".join(
                f"{name}={value:.6g}"
                for name, value in sorted(
                    result["libraries"].items(),
                    key=lambda item: (
                        LIBRARY_ORDER.index(item[0])
                        if item[0] in LIBRARY_ORDER
                        else len(LIBRARY_ORDER),
                        item[0],
                    ),
                )
            )
            rows.append(
                "<tr>"
                f"<td>{html.escape(result['case'])}</td>"
                f"<td>{html.escape(result['metric'])}</td>"
                f"<td>{html_number(result['zynum_value'])}</td>"
                f"<td>{html.escape(result['fastest_comparator'] or '')}</td>"
                f"<td>{html_number(result['ratio'])}</td>"
                f"<td>{html.escape(library_values)}</td>"
                f"<td class=\"{html.escape(result['status'])}\">{html.escape(result['status'])}</td>"
                "</tr>"
            )
        sections.append(
            f"""
<section>
  <h2>{html.escape(category['title'])}</h2>
  <p class="meta">status={category['status']} | cases={counts['total']} | passed={counts['passed']} | failed={counts['failed']} | missing={counts['missing']} | comparator-incomplete={counts['comparator_incomplete']} | accepted rows={category['rows']['accepted']} | rejected rows={category['rows']['rejected']}</p>
  <p><a href="{category['svg']}">Open full-size SVG</a></p>
  <a href="{category['svg']}"><img src="{category['svg']}" alt="{html.escape(category['title'])} real performance chart"></a>
  <details>
    <summary>Case table ({counts['total']})</summary>
    <table><thead><tr><th>Case</th><th>Metric</th><th>Zynum</th><th>Fastest comparator</th><th>Ratio</th><th>All library medians</th><th>Status</th></tr></thead><tbody>{''.join(rows)}</tbody></table>
  </details>
</section>
"""
        )

    document = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Zynum full benchmark report</title>
<style>
  :root {{ color-scheme: light; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }}
  body {{ margin: 0; color: #111827; background: #f3f4f6; }}
  header, section {{ background: #fff; border-bottom: 1px solid #d1d5db; padding: 24px max(24px, calc((100vw - 1440px) / 2)); }}
  h1, h2 {{ margin: 0 0 10px; }}
  .meta {{ color: #4b5563; font-size: 14px; }}
  img {{ display: block; width: 100%; background: #fff; border: 1px solid #e5e7eb; }}
  table {{ border-collapse: collapse; width: 100%; margin-top: 12px; font-size: 13px; }}
  th, td {{ border-bottom: 1px solid #e5e7eb; padding: 7px; text-align: left; }}
  th {{ position: sticky; top: 0; background: #f9fafb; }}
  .passed {{ color: #15803d; }} .failed {{ color: #dc2626; }} .missing, .missing-zynum, .missing-comparator {{ color: #6b7280; }}
  details {{ margin-top: 18px; }} a {{ color: #1d4ed8; }}
</style>
</head>
<body>
<header>
  <h1>Zynum Full Benchmark Report</h1>
  <p>Charts show real metric values in native units. Throughput ratio = Zynum / fastest comparator; latency ratio = fastest comparator latency / Zynum latency. The 1.0 strict gate is supplementary. Only correctness-checked rows are included.</p>
  <p><a href="summary.csv">summary.csv</a> | <a href="summary.json">summary.json</a></p>
  <p class="meta">CSV files scanned={report['files']['scanned']}; recognized={report['files']['recognized']}; ignored={report['files']['ignored_count']}</p>
</header>
{''.join(sections)}
</body>
</html>
"""
    output_path.write_text(document, encoding="utf-8")


def render_report(
    input_dir: Path,
    output_dir: Path,
    comparators: list[str] | None = None,
    expected_process_repeats: int | None = None,
) -> FullReport:
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    if not input_dir.is_dir():
        raise ValueError(f"input directory does not exist: {input_dir}")
    if expected_process_repeats is not None and expected_process_repeats <= 0:
        raise ValueError("expected process repeats must be positive")
    output_dir.mkdir(parents=True, exist_ok=True)
    requested_comparators = (
        [normalized_library(name) for name in comparators] if comparators else None
    )

    inputs = read_inputs(input_dir, expected_process_repeats)
    categories: list[CategoryReport] = []
    for category in CATEGORY_ORDER:
        results = aggregate_category(
            category, inputs["groups"][category], requested_comparators
        )
        counts = status_counts(results)
        row_stats = inputs["row_stats"][category]
        category_report: CategoryReport = {
            "id": category,
            "title": CATEGORY_TITLES[category],
            "status": category_status(results),
            "svg": SVG_NAMES[category],
            "files": sorted(inputs["category_files"][category]),
            "rows": {
                "seen": row_stats["seen"],
                "accepted": row_stats["accepted"],
                "rejected": row_stats["rejected"],
                "rejection_reasons": dict(sorted(row_stats["reasons"].items())),
            },
            "cases": counts,
            "results": results,
        }
        render_svg(category, results, output_dir / SVG_NAMES[category])
        categories.append(category_report)

    report: FullReport = {
        "schema_version": 1,
        "ratio": {
            "throughput": "Zynum / fastest comparator",
            "latency": "fastest comparator latency / Zynum latency",
        },
        "gate": 1.0,
        "statistic": "fresh-process median",
        "expected_process_repeats": expected_process_repeats,
        "requested_comparators": requested_comparators or [],
        "files": {
            "scanned": len(inputs["csv_files"]),
            "recognized": len(inputs["recognized"]),
            "recognized_paths": inputs["recognized"],
            "ignored_count": len(inputs["ignored"]),
            "ignored_paths": inputs["ignored"],
        },
        "categories": categories,
    }
    write_summary_csv(categories, output_dir / "summary.csv")
    (output_dir / "summary.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    render_index(report, output_dir / "index.html")
    return report


def main(argv=None):
    args = parse_args(argv)
    try:
        report = render_report(
            args.input_dir,
            args.output_dir,
            args.comparator,
            args.expected_process_repeats,
        )
    except (OSError, ValueError) as exc:
        print(f"error: {exc}")
        return 2
    for category in report["categories"]:
        counts = category["cases"]
        print(
            f"{category['id']}: status={category['status']} "
            f"cases={counts['total']} passed={counts['passed']} "
            f"failed={counts['failed']} missing={counts['missing']} "
            f"rejected_rows={category['rows']['rejected']}"
        )
    print(args.output_dir / "index.html")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
