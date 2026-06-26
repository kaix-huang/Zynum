#!/usr/bin/env python3
# Copyright (C) 2026 Zynum contributors
# SPDX-License-Identifier: LGPL-3.0-or-later

import argparse
import csv
import sys
from collections import defaultdict


def parse_args():
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
        "--allow-missing-comparators",
        action="store_true",
        help="Skip groups with no requested comparator rows instead of failing.",
    )
    parser.add_argument("--worst", type=int, default=20, help="Number of worst rows to print.")
    return parser.parse_args()


def group_key(row):
    return (row["kind"], row["label"], row["m"], row["n"], row["k"])


def row_allowed(args, row):
    if args.kind is not None and row["kind"] not in args.kind:
        return False
    if args.label is not None and row["label"] not in args.label:
        return False
    return True


def main():
    args = parse_args()
    if args.ratio <= 0:
        print("--ratio must be positive", file=sys.stderr)
        return 2
    comparators = args.comparator or ["Accelerate", "OpenBLAS"]

    groups = defaultdict(dict)
    with open(args.csv, newline="") as f:
        for row in csv.DictReader(f):
            if not row_allowed(args, row):
                continue
            try:
                row["gflops_value"] = float(row["gflops"])
            except (KeyError, ValueError):
                print(f"bad gflops value in row: {row}", file=sys.stderr)
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
        f"failed={len(failures)} missing={len(missing)} ratio={args.ratio:.6g}"
    )

    for ratio, key, zynum, best in failures[: args.worst]:
        kind, label, m, n, k = key
        print(
            f"FAIL {ratio:.6f} {kind} {label} m={m} n={n} k={k} "
            f"{args.zynum}={zynum['gflops_value']:.6f} "
            f"best={best['library']}:{best['gflops_value']:.6f}"
        )

    for key, label in missing[: args.worst]:
        kind, shape, m, n, k = key
        print(f"MISSING {label} {kind} {shape} m={m} n={n} k={k}")

    return 1 if failures or missing else 0


if __name__ == "__main__":
    sys.exit(main())
