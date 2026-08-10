#!/usr/bin/env python3
# Copyright (C) 2026 Zynum contributors
# SPDX-License-Identifier: LGPL-3.0-or-later

import argparse
import csv
import sys
from collections import defaultdict

from report_comparison import (
    best_higher_row,
    parse_positive_finite,
    positive_finite_ratio,
    validate_optional_metric_evidence,
)

CHECKED_STATUSES = {"sampled-ok", "checked-ok"}
STABLE_NEGATIVE_OPS = frozenset(
    {
        "scopy",
        "dcopy",
        "ccopy",
        "zcopy",
        "sswap",
        "dswap",
        "cswap",
        "zswap",
        "saxpy",
        "daxpy",
        "caxpy",
        "zaxpy",
        "sdot",
        "ddot",
        "sdsdot",
        "dsdot",
        "cdotu",
        "zdotu",
        "cdotc",
        "zdotc",
        "srot",
        "drot",
        "csrot",
        "zdrot",
        "srotm",
        "drotm",
    }
)


def parse_positive_integer(value, field):
    text = str(value)
    if not text.isascii() or not text.isdigit() or text.startswith("0"):
        raise ValueError(f"{field} must be a positive integer, got {value!r}")
    return int(text)


def positive_integer_argument(value):
    try:
        return parse_positive_integer(value, "--expected-process-repeats")
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Check Level 1 report CSV rows against the fastest comparator library."
    )
    parser.add_argument("csv", help="Path to a run_level1_report CSV file.")
    parser.add_argument(
        "--zynum", default="Zynum", help="Library label for Zynum rows."
    )
    parser.add_argument(
        "--comparator",
        action="append",
        default=None,
        help="Comparator library label. May be passed more than once.",
    )
    parser.add_argument(
        "--ratio",
        type=float,
        default=1.0,
        help="Required Zynum/comparator metric ratio. Use 1.0 for strict no-slower-than.",
    )
    parser.add_argument(
        "--stat",
        choices=("best", "median", "min"),
        default="best",
        help="Process-repeat statistic to compare. Older CSV files support only best.",
    )
    parser.add_argument(
        "--expected-process-repeats",
        type=positive_integer_argument,
        help=(
            "Require this process-repeat count on every row. Legacy CSV files "
            "without repeat-count columns are accepted only when this option is absent."
        ),
    )
    parser.add_argument(
        "--group", action="append", help="Restrict to one or more Level 1 groups."
    )
    parser.add_argument(
        "--op", action="append", help="Restrict to one or more operations."
    )
    parser.add_argument(
        "--negative-only",
        action="store_true",
        help="Check only rows where incx or incy is negative.",
    )
    parser.add_argument(
        "--allow-missing-comparators",
        action="store_true",
        help="Skip groups with no requested comparator rows instead of failing.",
    )
    parser.add_argument(
        "--allow-unchecked",
        action="store_true",
        help="Allow rows whose Level 1 correctness check is absent or not checked.",
    )
    parser.add_argument(
        "--worst", type=int, default=20, help="Number of worst rows to print."
    )
    return parser.parse_args(argv)


def row_has_negative_stride(row):
    try:
        return int(row.get("incx") or "1") < 0 or int(row.get("incy") or "1") < 0
    except ValueError:
        return False


def group_key(row):
    return (
        row["group"],
        row["op"],
        row.get("variant") or "default",
        row.get("incx") or "1",
        row.get("incy") or "1",
        row["n"],
        row["metric"],
    )


def row_allowed(args, row):
    if args.group is not None and row["group"] not in args.group:
        return False
    if args.op is not None and row["op"] not in args.op:
        return False
    if args.negative_only and not row_has_negative_stride(row):
        return False
    if row_has_negative_stride(row) and row["op"] not in STABLE_NEGATIVE_OPS:
        return False
    return True


def metric_value(args, row):
    if args.stat != "best":
        field = "metric_median" if args.stat == "median" else "metric_min"
        value = row.get(field)
        if value in (None, ""):
            raise KeyError(field)
        return parse_positive_finite(value, field)
    field = "bandwidth_gbps" if row["metric"] == "bandwidth_gbps" else "rate_gops"
    return parse_positive_finite(row[field], field)


def validate_repeat_evidence(args, row):
    has_process = "process_repeats" in row
    has_successful = "successful_repeats" in row
    if not has_process and not has_successful:
        if args.expected_process_repeats is not None:
            raise ValueError(
                "legacy row has no process-repeat evidence required by "
                "--expected-process-repeats"
            )
        return
    if not has_process or not has_successful:
        raise ValueError(
            "process_repeats and successful_repeats must either both be present "
            "or both be absent for a legacy row"
        )
    process_repeats = parse_positive_integer(row["process_repeats"], "process_repeats")
    successful_repeats = parse_positive_integer(
        row["successful_repeats"], "successful_repeats"
    )
    if successful_repeats != process_repeats:
        raise ValueError(
            "successful_repeats must equal process_repeats, got "
            f"{successful_repeats} != {process_repeats}"
        )
    if (
        args.expected_process_repeats is not None
        and process_repeats != args.expected_process_repeats
    ):
        raise ValueError(
            f"process_repeats is {process_repeats}, expected "
            f"{args.expected_process_repeats}"
        )


def main(argv=None):
    args = parse_args(argv)
    try:
        args.ratio = parse_positive_finite(args.ratio, "--ratio")
    except ValueError as exc:
        print(exc, file=sys.stderr)
        return 2
    comparators = args.comparator or ["Accelerate", "OpenBLAS"]

    groups = defaultdict(dict)
    excluded_negative = 0
    with open(args.csv, newline="") as f:
        for row in csv.DictReader(f):
            if (
                row_has_negative_stride(row)
                and row.get("op") not in STABLE_NEGATIVE_OPS
            ):
                excluded_negative += 1
            try:
                validate_repeat_evidence(args, row)
            except ValueError as exc:
                print(f"bad repeat evidence in row: {exc}: {row}", file=sys.stderr)
                return 2
            if not row_allowed(args, row):
                continue
            if row_has_negative_stride(row):
                required_surface = (
                    row.get("symbol"),
                    row.get("abi_surface"),
                    row.get("preflight_symbol"),
                    row.get("preflight_abi_surface"),
                )
                if row.get("capability_status") != "supported" or any(
                    value in (None, "") for value in required_surface
                ):
                    if row.get("library") != args.zynum:
                        continue
                    print(
                        f"negative-stride row lacks a supported capability surface: {row}",
                        file=sys.stderr,
                    )
                    return 2
                if required_surface[:2] != required_surface[2:]:
                    print(
                        f"negative-stride probe/preflight surface mismatch: {row}",
                        file=sys.stderr,
                    )
                    return 2
            if not args.allow_unchecked:
                if (
                    row.get("check_status") not in CHECKED_STATUSES
                    or row.get("status") != "ok"
                ):
                    if row.get("library") != args.zynum:
                        continue
                    print(
                        f"unchecked Level 1 row is not eligible for comparison: {row}",
                        file=sys.stderr,
                    )
                    return 2
            try:
                validate_optional_metric_evidence(row)
                row["metric_value"] = metric_value(args, row)
                key = group_key(row)
                library = row["library"]
            except (KeyError, ValueError):
                print(f"bad metric value in row: {row}", file=sys.stderr)
                return 2
            if library in groups[key]:
                print(
                    f"duplicate library row for {key!r}: {library}",
                    file=sys.stderr,
                )
                return 2
            groups[key][library] = row

    failures = []
    missing = []
    checked = 0
    for key, by_library in sorted(groups.items()):
        zynum = by_library.get(args.zynum)
        if zynum is None and args.zynum == "Zynum":
            zynum = by_library.get("zynum-blas")
        if zynum is None:
            missing.append((key, args.zynum))
            continue

        comparator_rows = [
            row for name in comparators if (row := by_library.get(name)) is not None
        ]
        if not comparator_rows:
            if not args.allow_missing_comparators:
                missing.append((key, ",".join(comparators)))
            continue

        checked += 1
        best = best_higher_row(comparator_rows, "metric_value")
        try:
            ratio = positive_finite_ratio(zynum["metric_value"], best["metric_value"])
        except ValueError as exc:
            print(f"bad comparison ratio for {key!r}: {exc}", file=sys.stderr)
            return 2
        if ratio < args.ratio:
            failures.append((ratio, key, zynum, best))

    failures.sort(key=lambda item: (item[0], item[1]))
    missing.sort(key=lambda item: (item[0], item[1]))
    print(
        f"checked={checked} passed={checked - len(failures)} "
        f"failed={len(failures)} missing={len(missing)} ratio={args.ratio:.6g} "
        f"stat={args.stat} excluded_negative={excluded_negative}"
    )
    no_checks = checked == 0
    if no_checks:
        print("no matching Zynum/comparator groups were checked", file=sys.stderr)

    for ratio, key, zynum, best in failures[: args.worst]:
        group, op, variant, incx, incy, n, metric = key
        print(
            f"FAIL {ratio:.6f} {group} {op} variant={variant} "
            f"incx={incx} incy={incy} n={n} metric={metric} "
            f"{args.zynum}={zynum['metric_value']:.6f} "
            f"best={best['library']}:{best['metric_value']:.6f}"
        )

    for key, label in missing[: args.worst]:
        group, op, variant, incx, incy, n, metric = key
        print(
            f"MISSING {label} {group} {op} variant={variant} "
            f"incx={incx} incy={incy} n={n} metric={metric}"
        )

    if no_checks:
        return 2
    return 1 if failures or missing else 0


if __name__ == "__main__":
    sys.exit(main())
