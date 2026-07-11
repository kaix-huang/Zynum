#!/usr/bin/env python3
# Copyright (C) 2026 Zynum contributors
# SPDX-License-Identifier: LGPL-3.0-or-later

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

DEFAULT_ACCELERATE = (
    "/System/Library/Frameworks/Accelerate.framework/Accelerate"
    if sys.platform == "darwin"
    else "none"
)
DEFAULT_OPENBLAS = (
    "/opt/homebrew/opt/openblas/lib/libopenblas.dylib"
    if sys.platform == "darwin"
    else "none"
)
DEFAULT_SHAPES = (
    "square128:128:128",
    "tall512x128:512:128",
    "wide128x512:128:512",
)
REAL_ALPHA = "0.75"
REAL_BETA = "0.25"
COMPLEX_ALPHA = "0.75,-0.125"
COMPLEX_BETA = "0.25,0.0625"
CHECKED_STATUSES = {"checked-ok"}

PROBE_FIELDNAMES = [
    "level",
    "routine",
    "kind",
    "library",
    "library_path",
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
    "flop_count",
    "best_ns",
    "median_ns",
    "p95_ns",
    "max_ns",
    "gflops",
    "median_gflops",
    "metric",
    "status",
    "check_status",
    "check_max_abs_error",
    "check_max_rel_error",
    "check_samples",
    "check_raw_output",
]

CSV_FIELDNAMES = PROBE_FIELDNAMES + [
    "process_repeats",
    "successful_repeats",
    "metric_min",
    "metric_median",
    "metric_max",
    "metric_samples",
]


@dataclass(frozen=True)
class Shape:
    name: str
    m: int
    n: int


@dataclass(frozen=True)
class RoutineSpec:
    name: str
    kind: str
    complex_scalars: bool


@dataclass(frozen=True)
class SymmCase:
    routine: RoutineSpec
    shape: Shape
    side: str
    uplo: str
    alpha: str
    beta: str


ROUTINES = {
    spec.name: spec
    for spec in (
        RoutineSpec("ssymm", "f32", False),
        RoutineSpec("dsymm", "f64", False),
        RoutineSpec("csymm", "c32", True),
        RoutineSpec("zsymm", "c64", True),
        RoutineSpec("chemm", "c32", True),
        RoutineSpec("zhemm", "c64", True),
    )
}


def default_zynum_blas():
    if sys.platform == "darwin":
        return "zig-out/lib/libzynum_blas.dylib"
    if sys.platform == "win32":
        return "zig-out/bin/zynum_blas.dll"
    return "zig-out/lib/libzynum_blas.so"


def parse_shape_spec(value):
    parts = value.split(":")
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(f"shape must be NAME:M:N, got {value!r}")
    name = parts[0].strip()
    if not name:
        raise argparse.ArgumentTypeError("shape name must not be empty")
    try:
        m, n = (int(part, 10) for part in parts[1:])
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"shape dimensions must be integers, got {value!r}"
        ) from exc
    if m < 1 or n < 1:
        raise argparse.ArgumentTypeError(
            f"shape dimensions must be positive, got {value!r}"
        )
    return Shape(name, m, n)


def parse_scalar(value):
    parts = value.split(",")
    if len(parts) not in (1, 2) or any(not part.strip() for part in parts):
        raise argparse.ArgumentTypeError(
            f"scalar must be RE or RE,IM, got {value!r}"
        )
    try:
        real = float(parts[0])
        imaginary = float(parts[1]) if len(parts) == 2 else 0.0
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"scalar must be RE or RE,IM, got {value!r}"
        ) from exc
    return real, imaginary


def scalar_text(value):
    real, imaginary = parse_scalar(value)
    if imaginary == 0:
        return format(real, ".17g")
    return f"{format(real, '.17g')},{format(imaginary, '.17g')}"


def upper_choice(choices):
    def parse(value):
        result = value.upper()
        if result not in choices:
            raise argparse.ArgumentTypeError(
                f"expected one of {','.join(choices)}, got {value!r}"
            )
        return result

    return parse


def routine_name(value):
    name = value.lower()
    if name not in ROUTINES:
        raise argparse.ArgumentTypeError(
            f"unknown routine {value!r}; choose from {','.join(ROUTINES)}"
        )
    return name


def normalize_negative_scalar_args(argv):
    values = list(sys.argv[1:] if argv is None else argv)
    normalized = []
    index = 0
    while index < len(values):
        value = values[index]
        if value in ("--alpha", "--beta") and index + 1 < len(values):
            scalar = values[index + 1]
            if scalar.startswith("-"):
                try:
                    parse_scalar(scalar)
                except argparse.ArgumentTypeError:
                    pass
                else:
                    normalized.append(f"{value}={scalar}")
                    index += 2
                    continue
        normalized.append(value)
        index += 1
    return normalized


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Run SYMM/HEMM comparator cases with one fresh process per "
            "library/case/repeat and write an aggregate CSV."
        )
    )
    parser.add_argument("--probe", default="zig-out/bin/symm-probe")
    parser.add_argument("--zynum", default=default_zynum_blas())
    parser.add_argument("--accelerate", default=DEFAULT_ACCELERATE)
    parser.add_argument("--openblas", default=DEFAULT_OPENBLAS)
    parser.add_argument("--mkl")
    parser.add_argument("--aocl-blis")
    parser.add_argument("--atlas")
    parser.add_argument(
        "--extra-blas",
        action="append",
        default=[],
        metavar="LABEL=PATH",
        help="Additional drop-in Fortran BLAS comparator. May be repeated.",
    )
    parser.add_argument(
        "--routine",
        action="append",
        type=routine_name,
        default=[],
        help="SYMM or HEMM routine to include. Defaults to all six routines.",
    )
    parser.add_argument(
        "--shape",
        action="append",
        type=parse_shape_spec,
        default=[],
        metavar="NAME:M:N",
        help="Matrix shape. May be repeated.",
    )
    parser.add_argument(
        "--side",
        action="append",
        type=upper_choice(("L", "R")),
        default=[],
        help="Structured operand side. Defaults to L and R.",
    )
    parser.add_argument(
        "--uplo",
        action="append",
        type=upper_choice(("U", "L")),
        default=[],
        help="Stored input triangle. Defaults to U and L.",
    )
    parser.add_argument(
        "--alpha",
        action="append",
        default=[],
        help=(
            "Alpha as RE or RE,IM. May be repeated. Defaults are real for "
            "s/d and complex for c/z routines."
        ),
    )
    parser.add_argument(
        "--beta",
        action="append",
        default=[],
        help=(
            "Beta as RE or RE,IM. May be repeated. Defaults are real for "
            "s/d and complex for c/z routines."
        ),
    )
    parser.add_argument("--reps", type=int, default=5)
    parser.add_argument(
        "--process-repeats",
        type=int,
        default=3,
        help="Independent processes per library and complete SYMM/HEMM case.",
    )
    parser.add_argument("--csv", required=True)
    parser.add_argument("--skip-missing", action="store_true")
    args = parser.parse_args(normalize_negative_scalar_args(argv))
    if args.reps < 1:
        parser.error("--reps must be at least 1")
    if args.process_repeats < 1:
        parser.error("--process-repeats must be at least 1")
    try:
        args.alpha = [scalar_text(value) for value in args.alpha]
        args.beta = [scalar_text(value) for value in args.beta]
    except argparse.ArgumentTypeError as exc:
        parser.error(str(exc))
    return args


def unique_preserving_order(values):
    return list(dict.fromkeys(values))


def requested_shapes(args):
    return args.shape or [parse_shape_spec(value) for value in DEFAULT_SHAPES]


def requested_routines(args):
    names = unique_preserving_order(args.routine or ROUTINES)
    return [ROUTINES[name] for name in names]


def requested_sides(args):
    return unique_preserving_order(args.side or ("L", "R"))


def requested_uplos(args):
    return unique_preserving_order(args.uplo or ("U", "L"))


def routine_scalars(routine, requested, real_default, complex_default, name):
    if not requested:
        return [complex_default if routine.complex_scalars else real_default]
    if routine.complex_scalars:
        return requested
    values = [value for value in requested if parse_scalar(value)[1] == 0]
    if not values:
        raise ValueError(
            f"selected {routine.name} has no real {name}; real routines reject "
            "complex scalars"
        )
    return values


def requested_cases(args):
    cases = []
    for routine in requested_routines(args):
        alphas = routine_scalars(
            routine, args.alpha, REAL_ALPHA, COMPLEX_ALPHA, "alpha"
        )
        betas = routine_scalars(
            routine, args.beta, REAL_BETA, COMPLEX_BETA, "beta"
        )
        for shape in requested_shapes(args):
            for side in requested_sides(args):
                for uplo in requested_uplos(args):
                    for alpha in alphas:
                        for beta in betas:
                            cases.append(
                                SymmCase(routine, shape, side, uplo, alpha, beta)
                            )
    if not cases:
        raise ValueError("the selected filters produce no SYMM/HEMM cases")
    return cases


def append_extra_blas(candidates, items):
    for item in items:
        if "=" not in item:
            raise ValueError(f"--extra-blas must be LABEL=PATH, got {item!r}")
        label, path = (part.strip() for part in item.split("=", 1))
        if not label or not path:
            raise ValueError(f"--extra-blas must be LABEL=PATH, got {item!r}")
        candidates.append((label, path))


def libraries(args):
    result = [("Zynum", args.zynum)]
    candidates = [
        ("Accelerate", args.accelerate),
        ("OpenBLAS", args.openblas),
    ]
    if args.mkl:
        candidates.append(("MKL", args.mkl))
    if args.aocl_blis:
        candidates.append(("AOCL-BLIS", args.aocl_blis))
    if args.atlas:
        candidates.append(("ATLAS", args.atlas))
    append_extra_blas(candidates, args.extra_blas)
    result.extend((label, path) for label, path in candidates if path and path != "none")
    return result


def library_available(path):
    candidate = Path(path)
    if candidate.exists():
        return True
    if (
        sys.platform == "darwin"
        and path.startswith("/System/Library/Frameworks/")
        and ".framework/" in path
    ):
        # System framework images may exist only in the dyld shared cache.
        return True
    # Do not load a BLAS into the controller just to test it. A bare soname is
    # delegated to the worker's dynamic loader; explicit paths must exist.
    return "/" not in path and "\\" not in path


def selected_libraries(args):
    result = []
    for index, (label, path) in enumerate(libraries(args)):
        if library_available(path):
            result.append((label, path))
            continue
        if index == 0 or not args.skip_missing:
            raise ValueError(f"BLAS library is not available: {label}={path}")
        print(f"[symm] skipping missing comparator {label}={path}", file=sys.stderr)
    return result


def case_command(args, library_name, library_path, case):
    return [
        args.probe,
        "--blas",
        library_path,
        "--library",
        library_name,
        "--routine",
        case.routine.name,
        "--shape",
        case.shape.name,
        "--m",
        str(case.shape.m),
        "--n",
        str(case.shape.n),
        "--side",
        case.side,
        "--uplo",
        case.uplo,
        "--alpha",
        case.alpha,
        "--beta",
        case.beta,
        "--reps",
        str(args.reps),
    ]


def structured_order(case):
    return case.shape.m if case.side == "L" else case.shape.n


def flop_count(case):
    factor = 8 if case.routine.complex_scalars else 2
    return factor * case.shape.m * case.shape.n * structured_order(case)


def error_row(args, library_name, library_path, case, detail):
    alpha_re, alpha_im = parse_scalar(case.alpha)
    beta_re, beta_im = parse_scalar(case.beta)
    order = structured_order(case)
    return {
        "level": "level3",
        "routine": case.routine.name,
        "kind": case.routine.kind,
        "library": library_name,
        "library_path": library_path,
        "shape": case.shape.name,
        "m": str(case.shape.m),
        "n": str(case.shape.n),
        "side": case.side,
        "uplo": case.uplo,
        "alpha_re": format(alpha_re, ".17g"),
        "alpha_im": format(alpha_im, ".17g"),
        "beta_re": format(beta_re, ".17g"),
        "beta_im": format(beta_im, ".17g"),
        "order": str(order),
        "lda": str(order),
        "ldb": str(case.shape.m),
        "ldc": str(case.shape.m),
        "reps": str(args.reps),
        "flop_count": str(flop_count(case)),
        "best_ns": "",
        "median_ns": "",
        "p95_ns": "",
        "max_ns": "",
        "gflops": "",
        "median_gflops": "",
        "metric": "gflops",
        "status": "error",
        "check_status": "error",
        "check_max_abs_error": "",
        "check_max_rel_error": "",
        "check_samples": "0",
        "check_raw_output": detail,
    }


def probe_row_matches(args, row, library_name, library_path, case):
    alpha_re, alpha_im = parse_scalar(case.alpha)
    beta_re, beta_im = parse_scalar(case.beta)
    order = structured_order(case)
    expected = {
        "level": "level3",
        "routine": case.routine.name,
        "kind": case.routine.kind,
        "library": library_name,
        "library_path": library_path,
        "shape": case.shape.name,
        "m": str(case.shape.m),
        "n": str(case.shape.n),
        "side": case.side,
        "uplo": case.uplo,
        "order": str(order),
        "lda": str(order),
        "ldb": str(case.shape.m),
        "ldc": str(case.shape.m),
        "reps": str(args.reps),
        "flop_count": str(flop_count(case)),
        "metric": "gflops",
        "check_samples": str(case.shape.m * case.shape.n),
    }
    mismatches = [
        f"{field}={row.get(field)!r} expected {value!r}"
        for field, value in expected.items()
        if row.get(field) != value
    ]
    try:
        scalars_match = (
            float(row["alpha_re"]) == alpha_re
            and float(row["alpha_im"]) == alpha_im
            and float(row["beta_re"]) == beta_re
            and float(row["beta_im"]) == beta_im
        )
    except (KeyError, ValueError):
        scalars_match = False
    if not scalars_match:
        mismatches.append("alpha/beta fields do not match the requested scalars")
    return mismatches


def child_environment():
    env = os.environ.copy()
    env.setdefault("OPENBLAS_DYNAMIC", "0")
    return env


def run_one_process(args, library_name, library_path, case):
    command = case_command(args, library_name, library_path, case)
    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        check=False,
        env=child_environment(),
    )
    if result.returncode != 0:
        detail = f"exit={result.returncode}"
        output = " ".join(
            part.strip() for part in (result.stdout, result.stderr) if part.strip()
        )
        if output:
            detail += f" {output}"
        return error_row(args, library_name, library_path, case, detail)
    rows = list(csv.DictReader(result.stdout.splitlines()))
    if len(rows) != 1:
        return error_row(
            args,
            library_name,
            library_path,
            case,
            f"probe returned {len(rows)} rows",
        )
    row = rows[0]
    missing_fields = [field for field in PROBE_FIELDNAMES if field not in row]
    if missing_fields:
        return error_row(
            args,
            library_name,
            library_path,
            case,
            f"probe row missing fields: {','.join(missing_fields)}",
        )
    mismatches = probe_row_matches(args, row, library_name, library_path, case)
    if mismatches:
        return error_row(
            args,
            library_name,
            library_path,
            case,
            "probe row mismatch: " + "; ".join(mismatches),
        )
    return {field: row.get(field, "") for field in PROBE_FIELDNAMES}


def repeat_row_eligible(row):
    if row.get("status") != "ok" or row.get("check_status") not in CHECKED_STATUSES:
        return False
    try:
        return float(row["median_gflops"]) >= 0 and int(row["median_ns"]) > 0
    except (KeyError, ValueError):
        return False


def failure_status(rows):
    statuses = {row.get("status", "error") for row in rows}
    checks = {row.get("check_status", "error") for row in rows}
    if "error" in statuses or "error" in checks:
        return "error", "error"
    if "missing" in statuses or "missing" in checks:
        return "missing", "missing"
    return "correctness_failed", "correctness_failed"


def aggregate_repeats(rows):
    if not rows:
        raise ValueError("cannot aggregate an empty repeat list")
    eligible = [row for row in rows if repeat_row_eligible(row)]
    base = dict(
        max(eligible, key=lambda row: float(row["gflops"])) if eligible else rows[0]
    )
    values = [float(row["median_gflops"]) for row in eligible]
    base.update(
        {
            "process_repeats": len(rows),
            "successful_repeats": len(eligible),
            "metric_min": format(min(values), ".17g") if values else "",
            "metric_median": format(statistics.median(values), ".17g")
            if values
            else "",
            "metric_max": format(max(values), ".17g") if values else "",
            "metric_samples": ",".join(format(value, ".17g") for value in values),
        }
    )
    errors = []
    details = []
    for repeat, row in enumerate(rows, 1):
        try:
            errors.append(float(row.get("check_max_abs_error") or 0))
        except ValueError:
            pass
        if not repeat_row_eligible(row) or row.get("check_raw_output"):
            detail = (
                f"repeat={repeat}: status={row.get('status', '')} "
                f"check_status={row.get('check_status', '')}"
            )
            if row.get("check_raw_output"):
                detail += f" {row['check_raw_output']}"
            details.append(detail)
    if errors:
        base["check_max_abs_error"] = format(max(errors), ".9g")
    if details:
        base["check_raw_output"] = " | ".join(details)
    if len(eligible) != len(rows):
        base["status"], base["check_status"] = failure_status(rows)
    return base


def command_output(command):
    try:
        result = subprocess.run(
            command, check=True, capture_output=True, text=True
        )
    except Exception:
        return None
    return result.stdout.strip()


def sha256_file(path):
    candidate = Path(path)
    if not candidate.is_file():
        return None
    digest = hashlib.sha256()
    with candidate.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def environment_snapshot():
    names = [
        "ZYNUM_MAXIMUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "OPENBLAS_DYNAMIC",
        "VECLIB_MAXIMUM_THREADS",
        "MKL_NUM_THREADS",
        "MKL_DYNAMIC",
        "OMP_NUM_THREADS",
        "BLIS_NUM_THREADS",
        "AOCL_DYNAMIC",
    ]
    env = child_environment()
    return {name: env.get(name, "unset") for name in names}


def zynum_maximum_threads_detected():
    value = os.environ.get("ZYNUM_MAXIMUM_THREADS")
    if value:
        try:
            parsed = int(value, 10)
            if parsed > 0:
                return min(parsed, max(1, os.cpu_count() or 1))
        except ValueError:
            pass
    return max(1, os.cpu_count() or 1)


def write_metadata(args, output, selected, cases):
    source_status = command_output(["git", "status", "--short"])
    metadata = {
        "generated_at_unix": time.time(),
        "argv": sys.argv,
        "cwd": os.getcwd(),
        "platform": platform.platform(),
        "python_version": sys.version,
        "zig_version": command_output(["zig", "version"]),
        "source": {
            "revision": command_output(["git", "rev-parse", "HEAD"]),
            "branch": command_output(["git", "branch", "--show-current"]),
            "dirty": bool(source_status),
            "status_short": source_status,
        },
        "detected_cpu_count": os.cpu_count(),
        "zynum_maximum_threads": zynum_maximum_threads_detected(),
        "reps": args.reps,
        "process_repeats": args.process_repeats,
        "isolation": "fresh process per library/routine/shape/side/uplo/scalars/repeat",
        "process_metric": "probe median_gflops",
        "correctness_check": (
            "independent scalar reference for every C element before timing; "
            "unstored A triangle and Hermitian diagonal imaginary values are ignored"
        ),
        "case_count_per_library": len(cases),
        "environment": environment_snapshot(),
        "probe": {"path": args.probe, "sha256": sha256_file(args.probe)},
        "libraries": [
            {"name": name, "path": path, "sha256": sha256_file(path)}
            for name, path in selected
        ],
        "shapes": [
            {"name": shape.name, "m": shape.m, "n": shape.n}
            for shape in requested_shapes(args)
        ],
        "routines": [routine.name for routine in requested_routines(args)],
        "sides": requested_sides(args),
        "uplos": requested_uplos(args),
        "alphas": args.alpha or "routine defaults",
        "betas": args.beta or "routine defaults",
    }
    metadata_path = output.with_suffix(output.suffix + ".meta.json")
    with metadata_path.open("w") as file:
        json.dump(metadata, file, indent=2, sort_keys=True)
        file.write("\n")


def run_controller(args):
    probe = Path(args.probe)
    if not probe.is_file():
        raise ValueError(f"SYMM probe is not available: {args.probe}")
    cases = requested_cases(args)
    selected = selected_libraries(args)
    rows = []
    for library_name, library_path in selected:
        for case_index, case in enumerate(cases, 1):
            print(
                f"[symm {library_name}] case={case_index}/{len(cases)} "
                f"{case.routine.name} shape={case.shape.name} "
                f"m={case.shape.m} n={case.shape.n} side={case.side} "
                f"uplo={case.uplo} alpha={case.alpha} beta={case.beta}",
                file=sys.stderr,
                flush=True,
            )
            repeats = [
                run_one_process(args, library_name, library_path, case)
                for _ in range(args.process_repeats)
            ]
            rows.append(aggregate_repeats(repeats))

    output = Path(args.csv)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=CSV_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)
    write_metadata(args, output, selected, cases)


def main(argv=None):
    args = parse_args(argv)
    try:
        run_controller(args)
    except ValueError as exc:
        print(exc, file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
