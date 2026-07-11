#!/usr/bin/env python3
# Copyright (C) 2026 Zynum contributors
# SPDX-License-Identifier: LGPL-3.0-or-later

import argparse
import csv
import math
import sys
from collections import defaultdict


CHECKED_STATUSES = {"sampled-ok", "checked-ok"}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Validate ROTG/ROTMG report status and correctness, then compare "
            "fresh-process median latency against the fastest requested BLAS."
        )
    )
    parser.add_argument("csv", help="Path to a run_rotg_latency_report CSV file.")
    parser.add_argument("--zynum", default="Zynum")
    parser.add_argument(
        "--comparator",
        action="append",
        default=None,
        help="Comparator library label. May be repeated.",
    )
    parser.add_argument(
        "--ratio",
        type=float,
        default=1.0,
        help="Maximum allowed Zynum/comparator median ns/call ratio.",
    )
    parser.add_argument("--routine", action="append")
    parser.add_argument("--kind", action="append")
    parser.add_argument("--case", action="append")
    parser.add_argument("--allow-missing-comparators", action="store_true")
    parser.add_argument("--worst", type=int, default=20)
    return parser.parse_args(argv)


def group_key(row):
    return (
        row["routine"],
        row["kind"],
        row["case"],
        row["corpus_size"],
        row["samples"],
        row["calls_per_sample"],
        row["process_repeats"],
        row["metric"],
    )


def row_allowed(args, row):
    filters = (
        (args.routine, row["routine"]),
        (args.kind, row["kind"]),
        (args.case, row["case"]),
    )
    return not any(
        selected is not None and value not in selected
        for selected, value in filters
    )


def describe_key(key):
    routine, kind, input_case, corpus_size, samples, calls, repeats, metric = key
    return (
        "{} {} case={} corpus_size={} samples={} calls_per_sample={} "
        "process_repeats={} metric={}".format(
            routine,
            kind,
            input_case,
            corpus_size,
            samples,
            calls,
            repeats,
            metric,
        )
    )


def row_eligible(row):
    if row.get("status") != "ok" or row.get("check_status") not in CHECKED_STATUSES:
        return False
    try:
        repeats = int(row["process_repeats"])
        successful = int(row["successful_repeats"])
        median = float(row["metric_median"])
        return (
            repeats > 0
            and successful == repeats
            and math.isfinite(median)
            and median > 0
            and row.get("metric") == "ns_per_call"
        )
    except (KeyError, TypeError, ValueError):
        return False


def main(argv=None):
    args = parse_args(argv)
    if args.ratio <= 0 or not math.isfinite(args.ratio):
        print("--ratio must be finite and positive", file=sys.stderr)
        return 2
    comparators = args.comparator or ["Accelerate", "OpenBLAS"]
    relevant_libraries = set(comparators + [args.zynum])

    try:
        csv_file = open(args.csv, newline="")
    except OSError as exc:
        print(exc, file=sys.stderr)
        return 2

    selected_rows = []
    with csv_file:
        for row in csv.DictReader(csv_file):
            try:
                allowed = row_allowed(args, row)
            except KeyError:
                print("latency CSV row is missing required parameters: {}".format(row), file=sys.stderr)
                return 2
            if allowed and row.get("library") in relevant_libraries:
                selected_rows.append(row)

    # Eligibility is a separate first phase. No latency value is read for a
    # comparison until every selected Zynum row has passed status/correctness.
    for row in selected_rows:
        if row.get("library") == args.zynum and not row_eligible(row):
            print(
                "unchecked ROTG/ROTMG row is not eligible for comparison: {}".format(row),
                file=sys.stderr,
            )
            return 2

    groups = defaultdict(dict)
    for row in selected_rows:
        if not row_eligible(row):
            continue
        try:
            key = group_key(row)
            row["median_latency"] = float(row["metric_median"])
        except (KeyError, TypeError, ValueError):
            print("bad latency metric or parameter value in row: {}".format(row), file=sys.stderr)
            return 2
        if row["library"] in groups[key]:
            print(
                "duplicate library row for {}: {}".format(
                    describe_key(key), row["library"]
                ),
                file=sys.stderr,
            )
            return 2
        groups[key][row["library"]] = row

    failures = []
    missing = []
    checked_count = 0
    for key, by_library in groups.items():
        zynum = by_library.get(args.zynum)
        if zynum is None:
            missing.append((key, args.zynum))
            continue
        comparator_rows = [
            by_library[name] for name in comparators if name in by_library
        ]
        if not comparator_rows:
            if not args.allow_missing_comparators:
                missing.append((key, ",".join(comparators)))
            continue
        checked_count += 1
        fastest = min(comparator_rows, key=lambda row: row["median_latency"])
        ratio = zynum["median_latency"] / fastest["median_latency"]
        if ratio > args.ratio:
            failures.append((ratio, key, zynum, fastest))

    failures.sort(key=lambda item: item[0], reverse=True)
    print(
        "checked={} passed={} failed={} missing={} ratio={:.6g} stat=median".format(
            checked_count,
            checked_count - len(failures),
            len(failures),
            len(missing),
            args.ratio,
        )
    )
    if checked_count == 0:
        print("no matching Zynum/comparator groups were checked", file=sys.stderr)

    for ratio, key, zynum, fastest in failures[: args.worst]:
        print(
            "FAIL {:.6f} {} {}={:.6f} best={}:{:.6f}".format(
                ratio,
                describe_key(key),
                args.zynum,
                zynum["median_latency"],
                fastest["library"],
                fastest["median_latency"],
            )
        )
    for key, label in missing[: args.worst]:
        print("MISSING {} {}".format(label, describe_key(key)))

    if checked_count == 0:
        return 2
    return 1 if failures or missing else 0


if __name__ == "__main__":
    sys.exit(main())
