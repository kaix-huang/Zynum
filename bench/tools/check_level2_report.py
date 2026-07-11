#!/usr/bin/env python3
# Copyright (C) 2026 Zynum contributors
# SPDX-License-Identifier: LGPL-3.0-or-later

import argparse
import csv
import sys
from collections import defaultdict

CHECKED_STATUSES = {"sampled-ok", "checked-ok"}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Check Level 2 report CSV rows against the fastest comparator library."
    )
    parser.add_argument("csv", help="Path to a run_level2_report CSV file.")
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
        help="Required Zynum/comparator GOP/s ratio. Use 1.0 for strict no-slower-than.",
    )
    parser.add_argument(
        "--stat",
        choices=("best", "median", "min"),
        default="best",
        help="Process-repeat statistic to compare. Older CSV files support only best.",
    )
    parser.add_argument(
        "--case", action="append", help="Restrict to one or more Level 2 cases."
    )
    parser.add_argument(
        "--kind", action="append", help="Restrict to one or more scalar kinds."
    )
    parser.add_argument(
        "--shape", action="append", help="Restrict to one or more shape labels."
    )
    parser.add_argument(
        "--m", action="append", help="Restrict to one or more matrix row counts."
    )
    parser.add_argument(
        "--n", action="append", help="Restrict to one or more matrix column counts."
    )
    parser.add_argument(
        "--uplo",
        action="append",
        choices=("U", "L"),
        help="Restrict structured Level 2 rows by upper/lower storage.",
    )
    parser.add_argument(
        "--trans",
        action="append",
        choices=("N", "T", "C"),
        help="Restrict rows by transpose or conjugate-transpose mode.",
    )
    parser.add_argument(
        "--diag",
        action="append",
        choices=("N", "U"),
        help="Restrict triangular rows by non-unit/unit diagonal.",
    )
    parser.add_argument(
        "--incx",
        action="append",
        help="Restrict structured Level 2 rows by x vector increment.",
    )
    parser.add_argument(
        "--incy",
        action="append",
        help="Restrict structured Level 2 rows by y vector increment.",
    )
    parser.add_argument(
        "--storage",
        action="append",
        help="Restrict rows by dense, packed, or banded storage family.",
    )
    parser.add_argument(
        "--lda", action="append", help="Restrict rows by leading dimension."
    )
    parser.add_argument(
        "--k", action="append", help="Restrict symmetric/Hermitian band width."
    )
    parser.add_argument(
        "--kl", action="append", help="Restrict general-band lower bandwidth."
    )
    parser.add_argument(
        "--ku", action="append", help="Restrict general-band upper bandwidth."
    )
    parser.add_argument(
        "--allow-missing-comparators",
        action="store_true",
        help="Skip groups with no requested comparator rows instead of failing.",
    )
    parser.add_argument(
        "--allow-unchecked",
        action="store_true",
        help="Allow rows whose Level 2 correctness check is absent or not checked.",
    )
    parser.add_argument(
        "--worst", type=int, default=20, help="Number of worst rows to print."
    )
    return parser.parse_args(argv)


def shape_fields(row):
    n = row["n"]
    m = row.get("m") or n
    shape = row.get("shape") or f"sq{n}"
    return shape, m, n


def operation_fields(row):
    return (
        row.get("storage") or "",
        row.get("lda") or "",
        row.get("k") or "",
        row.get("kl") or "",
        row.get("ku") or "",
        row.get("uplo") or "",
        row.get("trans") or "",
        row.get("diag") or "",
        row.get("incx") or "",
        row.get("incy") or "",
    )


def group_key(row):
    shape, m, n = shape_fields(row)
    return (
        row["case"],
        row["kind"],
        shape,
        m,
        n,
        *operation_fields(row),
        row["metric"],
    )


def row_allowed(args, row):
    shape, m, n = shape_fields(row)
    storage, lda, k, kl, ku, uplo, trans, diag, incx, incy = operation_fields(row)
    if args.case is not None and row["case"] not in args.case:
        return False
    if args.kind is not None and row["kind"] not in args.kind:
        return False
    if args.shape is not None and shape not in args.shape:
        return False
    if args.m is not None and m not in args.m:
        return False
    if args.n is not None and n not in args.n:
        return False
    if args.storage is not None and storage not in args.storage:
        return False
    if args.lda is not None and lda not in args.lda:
        return False
    if args.k is not None and k not in args.k:
        return False
    if args.kl is not None and kl not in args.kl:
        return False
    if args.ku is not None and ku not in args.ku:
        return False
    if args.uplo is not None and uplo not in args.uplo:
        return False
    if args.trans is not None and trans not in args.trans:
        return False
    if args.diag is not None and diag not in args.diag:
        return False
    if args.incx is not None and incx not in args.incx:
        return False
    if args.incy is not None and incy not in args.incy:
        return False
    return True


def metric_value(args, row):
    if args.stat != "best":
        field = "metric_median" if args.stat == "median" else "metric_min"
        value = row.get(field)
        if value in (None, ""):
            raise KeyError(field)
        return float(value)
    return float(row["rate_gops"])


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
            if not args.allow_unchecked:
                if (
                    row.get("check_status") not in CHECKED_STATUSES
                    or row.get("status") != "ok"
                ):
                    if row.get("library") != args.zynum:
                        continue
                    print(
                        f"unchecked Level 2 row is not eligible for comparison: {row}",
                        file=sys.stderr,
                    )
                    return 2
            try:
                row["metric_value"] = metric_value(args, row)
            except (KeyError, ValueError):
                print(f"bad metric value in row: {row}", file=sys.stderr)
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
        f"checked={checked} passed={checked - len(failures)} "
        f"failed={len(failures)} missing={len(missing)} ratio={args.ratio:.6g} "
        f"stat={args.stat}"
    )
    no_checks = checked == 0
    if no_checks:
        print("no matching Zynum/comparator groups were checked", file=sys.stderr)

    for ratio, key, zynum, best in failures[: args.worst]:
        (
            case,
            kind,
            shape,
            m,
            n,
            storage,
            lda,
            k,
            kl,
            ku,
            uplo,
            trans,
            diag,
            incx,
            incy,
            metric,
        ) = key
        print(
            f"FAIL {ratio:.6f} {case} {kind} shape={shape} m={m} n={n} "
            f"storage={storage or '-'} lda={lda or '-'} k={k or '-'} "
            f"kl={kl or '-'} ku={ku or '-'} "
            f"uplo={uplo or '-'} trans={trans or '-'} diag={diag or '-'} "
            f"incx={incx or '-'} incy={incy or '-'} metric={metric} "
            f"{args.zynum}={zynum['metric_value']:.6f} "
            f"best={best['library']}:{best['metric_value']:.6f}"
        )

    for key, label in missing[: args.worst]:
        (
            case,
            kind,
            shape,
            m,
            n,
            storage,
            lda,
            k,
            kl,
            ku,
            uplo,
            trans,
            diag,
            incx,
            incy,
            metric,
        ) = key
        print(
            f"MISSING {label} {case} {kind} shape={shape} m={m} n={n} "
            f"storage={storage or '-'} lda={lda or '-'} k={k or '-'} "
            f"kl={kl or '-'} ku={ku or '-'} "
            f"uplo={uplo or '-'} trans={trans or '-'} diag={diag or '-'} "
            f"incx={incx or '-'} incy={incy or '-'} metric={metric}"
        )

    if no_checks:
        return 2
    return 1 if failures or missing else 0


if __name__ == "__main__":
    sys.exit(main())
