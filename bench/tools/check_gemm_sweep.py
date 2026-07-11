#!/usr/bin/env python3
# Copyright (C) 2026 Zynum contributors
# SPDX-License-Identifier: LGPL-3.0-or-later

import argparse
import csv
import sys
from collections import defaultdict

CHECKED_STATUSES = {"sampled-ok", "checked-ok"}


def parse_transpose_spec(value):
    pair = value.upper()
    if len(pair) != 2 or any(trans not in "NTC" for trans in pair):
        raise argparse.ArgumentTypeError(
            f"transpose pair must contain two N/T/C characters, got {value!r}"
        )
    return pair


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Check GEMM sweep CSV rows against the fastest comparator library."
    )
    parser.add_argument("csv", help="Path to a gemm_sweep CSV file.")
    parser.add_argument("--zynum", default="Zynum", help="Library label for Zynum rows.")
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
        help="Required Zynum/comparator GFLOP/s ratio. Use 1.0 for strict no-slower-than.",
    )
    parser.add_argument("--kind", action="append", help="Restrict to one or more GEMM kinds.")
    parser.add_argument("--label", action="append", help="Restrict to one or more shape labels.")
    parser.add_argument(
        "--trans",
        action="append",
        type=parse_transpose_spec,
        help="Restrict to one or more transpose pairs. Legacy rows are treated as NN.",
    )
    parser.add_argument(
        "--stat",
        choices=["best", "median", "min"],
        default="best",
        help="Timing statistic to compare. min uses max_ns (the minimum observed throughput).",
    )
    parser.add_argument(
        "--allow-missing-comparators",
        action="store_true",
        help="Skip groups with no requested comparator rows instead of failing.",
    )
    parser.add_argument(
        "--allow-unchecked",
        action="store_true",
        help="Allow rows whose GEMM correctness check is absent or not checked.",
    )
    parser.add_argument("--worst", type=int, default=20, help="Number of worst rows to print.")
    return parser.parse_args(argv)


def transpose_fields(row):
    return (row.get("transa") or "N").upper(), (row.get("transb") or "N").upper()


def group_key(row):
    return (
        row["kind"],
        *transpose_fields(row),
        row["label"],
        row["m"],
        row["n"],
        row["k"],
    )


def row_allowed(args, row):
    if args.kind is not None and row["kind"] not in args.kind:
        return False
    if args.label is not None and row["label"] not in args.label:
        return False
    if args.trans is not None and "".join(transpose_fields(row)) not in args.trans:
        return False
    return True


def flop_factor(kind):
    return 8.0 if kind in {"cgemm", "zgemm"} else 2.0


def row_gflops(row, stat):
    if stat == "best":
        return float(row["gflops"])

    timing_field = "median_ns" if stat == "median" else "max_ns"
    timing_text = row.get(timing_field) or row.get("best_ns")
    if not timing_text:
        return float(row["gflops"])
    elapsed_ns = int(timing_text)
    if elapsed_ns <= 0:
        raise ValueError(f"non-positive {timing_field}")
    work = (
        flop_factor(row["kind"])
        * int(row["m"])
        * int(row["n"])
        * int(row["k"])
    )
    return work / elapsed_ns


def main(argv=None):
    args = parse_args(argv)
    if args.ratio <= 0:
        print("--ratio must be positive", file=sys.stderr)
        return 2
    comparators = args.comparator or ["Accelerate", "OpenBLAS"]

    groups = defaultdict(dict)
    with open(args.csv, newline="") as f:
        for row in csv.DictReader(f):
            if not row_allowed(args, row):
                continue
            if not args.allow_unchecked and row.get("check") not in CHECKED_STATUSES:
                print(
                    f"unchecked GEMM row is not eligible for comparison: {row}",
                    file=sys.stderr,
                )
                return 2
            try:
                row["gflops_value"] = row_gflops(row, args.stat)
            except (KeyError, TypeError, ValueError) as error:
                print(
                    f"bad {args.stat} GFLOP/s value in row ({error}): {row}",
                    file=sys.stderr,
                )
                return 2
            groups[group_key(row)][row["library"]] = row

    failures = []
    missing = []
    checked = 0
    for key, by_library in groups.items():
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
        best = max(comparator_rows, key=lambda row: row["gflops_value"])
        required = args.ratio * best["gflops_value"]
        ratio = zynum["gflops_value"] / best["gflops_value"] if best["gflops_value"] > 0 else 1.0
        if zynum["gflops_value"] < required:
            failures.append((ratio, key, zynum, best))

    failures.sort(key=lambda item: item[0])
    print(
        f"checked={checked} passed={checked - len(failures)} "
        f"failed={len(failures)} missing={len(missing)} ratio={args.ratio:.6g} stat={args.stat}"
    )
    no_checks = checked == 0
    if no_checks:
        print("no matching Zynum/comparator groups were checked", file=sys.stderr)

    for ratio, key, zynum, best in failures[: args.worst]:
        kind, transa, transb, label, m, n, k = key
        print(
            f"FAIL {ratio:.6f} {kind} trans={transa}{transb} {label} m={m} n={n} k={k} "
            f"{args.zynum}={zynum['gflops_value']:.6f} "
            f"best={best['library']}:{best['gflops_value']:.6f}"
        )

    for key, label in missing[: args.worst]:
        kind, transa, transb, shape, m, n, k = key
        print(
            f"MISSING {label} {kind} trans={transa}{transb} "
            f"{shape} m={m} n={n} k={k}"
        )

    if no_checks:
        return 2
    return 1 if failures or missing else 0


if __name__ == "__main__":
    sys.exit(main())
