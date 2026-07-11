#!/usr/bin/env python3
# Copyright (C) 2026 Zynum contributors
# SPDX-License-Identifier: LGPL-3.0-or-later

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict

CHECKED_STATUSES = {"checked-ok"}


def normalize_negative_scalar_args(argv):
    values = list(sys.argv[1:] if argv is None else argv)
    normalized = []
    index = 0
    while index < len(values):
        value = values[index]
        if value == "--alpha" and index + 1 < len(values):
            scalar = values[index + 1]
            if scalar.startswith("-"):
                try:
                    scalar_pair(scalar)
                except ValueError:
                    pass
                else:
                    normalized.append(f"--alpha={scalar}")
                    index += 2
                    continue
        normalized.append(value)
        index += 1
    return normalized


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Compare checked TRMM/TRSM report rows against the fastest requested "
            "BLAS using fresh-process statistics."
        )
    )
    parser.add_argument(
        "csv", help="Path to a run_triangular_matrix_report CSV file."
    )
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
        help="Required Zynum/comparator GFLOP/s ratio.",
    )
    parser.add_argument(
        "--stat",
        choices=("median", "best", "min"),
        default="median",
        help="Fresh-process statistic to compare. Defaults to median.",
    )
    parser.add_argument("--routine", action="append")
    parser.add_argument("--family", action="append", choices=("trmm", "trsm"))
    parser.add_argument("--kind", action="append")
    parser.add_argument("--shape", action="append")
    parser.add_argument("--m", action="append")
    parser.add_argument("--n", action="append")
    parser.add_argument("--side", action="append", choices=("L", "R"))
    parser.add_argument("--uplo", action="append", choices=("U", "L"))
    parser.add_argument("--trans", action="append", choices=("N", "T", "C"))
    parser.add_argument("--diag", action="append", choices=("N", "U"))
    parser.add_argument(
        "--alpha",
        action="append",
        help="Restrict alpha to RE or RE,IM. May be repeated.",
    )
    parser.add_argument("--allow-missing-comparators", action="store_true")
    parser.add_argument("--allow-unchecked", action="store_true")
    parser.add_argument("--worst", type=int, default=20)
    return parser.parse_args(normalize_negative_scalar_args(argv))


def scalar_pair(value):
    parts = value.split(",")
    if len(parts) not in (1, 2):
        raise ValueError(value)
    return float(parts[0]), float(parts[1]) if len(parts) == 2 else 0.0


def row_scalar(row):
    return float(row["alpha_re"]), float(row["alpha_im"])


def group_key(row):
    return (
        row["routine"],
        row["family"],
        row["kind"],
        row["shape"],
        row["m"],
        row["n"],
        row["side"],
        row["uplo"],
        row["trans"],
        row["diag"],
        row["alpha_re"],
        row["alpha_im"],
        row["order"],
        row["lda"],
        row["ldb"],
        row["reps"],
        row["process_repeats"],
        row["metric"],
    )


def row_allowed(args, row):
    filters = (
        (args.routine, row["routine"]),
        (args.family, row["family"]),
        (args.kind, row["kind"]),
        (args.shape, row["shape"]),
        (args.m, row["m"]),
        (args.n, row["n"]),
        (args.side, row["side"]),
        (args.uplo, row["uplo"]),
        (args.trans, row["trans"]),
        (args.diag, row["diag"]),
    )
    if any(
        selected is not None and value not in selected
        for selected, value in filters
    ):
        return False
    try:
        if args.alpha is not None and row_scalar(row) not in {
            scalar_pair(value) for value in args.alpha
        }:
            return False
    except ValueError:
        return False
    return True


def metric_value(args, row):
    field = {
        "median": "metric_median",
        "best": "metric_max",
        "min": "metric_min",
    }[args.stat]
    value = row.get(field)
    if value in (None, ""):
        raise KeyError(field)
    return float(value)


def describe_key(key):
    (
        routine,
        family,
        kind,
        shape,
        m,
        n,
        side,
        uplo,
        trans,
        diag,
        alpha_re,
        alpha_im,
        order,
        lda,
        ldb,
        reps,
        process_repeats,
        metric,
    ) = key
    return (
        f"{routine} family={family} {kind} shape={shape} m={m} n={n} "
        f"side={side} uplo={uplo} trans={trans} diag={diag} "
        f"alpha={alpha_re},{alpha_im} order={order} lda={lda} ldb={ldb} "
        f"reps={reps} process_repeats={process_repeats} metric={metric}"
    )


def main(argv=None):
    args = parse_args(argv)
    if args.ratio <= 0:
        print("--ratio must be positive", file=sys.stderr)
        return 2
    comparators = args.comparator or ["Accelerate", "OpenBLAS"]

    groups = defaultdict(dict)
    try:
        csv_file = open(args.csv, newline="")
    except OSError as exc:
        print(exc, file=sys.stderr)
        return 2
    with csv_file:
        for row in csv.DictReader(csv_file):
            try:
                allowed = row_allowed(args, row)
            except KeyError:
                print(
                    f"TRMM/TRSM CSV row is missing required parameters: {row}",
                    file=sys.stderr,
                )
                return 2
            if not allowed:
                continue
            checked = (
                row.get("status") == "ok"
                and row.get("check_status") in CHECKED_STATUSES
            )
            if not args.allow_unchecked and not checked:
                if row.get("library") == args.zynum:
                    print(
                        f"unchecked TRMM/TRSM row is not eligible: {row}",
                        file=sys.stderr,
                    )
                    return 2
                continue
            try:
                row["metric_value"] = metric_value(args, row)
                key = group_key(row)
            except (KeyError, ValueError):
                print(
                    f"bad TRMM/TRSM metric or parameter value in row: {row}",
                    file=sys.stderr,
                )
                return 2
            if row["library"] in groups[key]:
                print(
                    f"duplicate library row for {describe_key(key)}: "
                    f"{row['library']}",
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
        best = max(comparator_rows, key=lambda row: row["metric_value"])
        required = args.ratio * best["metric_value"]
        ratio = (
            zynum["metric_value"] / best["metric_value"]
            if best["metric_value"] > 0
            else 1.0
        )
        if zynum["metric_value"] < required:
            failures.append((ratio, key, zynum, best))

    failures.sort(key=lambda item: item[0])
    print(
        f"checked={checked_count} passed={checked_count - len(failures)} "
        f"failed={len(failures)} missing={len(missing)} "
        f"ratio={args.ratio:.6g} stat={args.stat}"
    )
    if checked_count == 0:
        print("no matching Zynum/comparator groups were checked", file=sys.stderr)

    for ratio, key, zynum, best in failures[: args.worst]:
        print(
            f"FAIL {ratio:.6f} {describe_key(key)} "
            f"{args.zynum}={zynum['metric_value']:.6f} "
            f"best={best['library']}:{best['metric_value']:.6f}"
        )
    for key, label in missing[: args.worst]:
        print(f"MISSING {label} {describe_key(key)}")

    if checked_count == 0:
        return 2
    return 1 if failures or missing else 0


if __name__ == "__main__":
    sys.exit(main())
