#!/usr/bin/env python3
# Copyright (C) 2026 Zynum contributors
# SPDX-License-Identifier: LGPL-3.0-or-later

from __future__ import annotations

import argparse
import csv
import io
import os
import platform
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

TOOLS_DIR = Path(__file__).resolve().parent
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from report_schedule import (  # noqa: E402
    SCHEDULE_CHOICES,
    collect_repeats,
    validate_schedule,
    validate_unique_library_labels,
)
from report_comparison import (  # noqa: E402
    parse_positive_finite,
    positive_finite_median,
    validate_optional_metric_evidence,
    validate_performance_fields,
)
import benchmark_artifacts  # noqa: E402
import benchmark_metadata  # noqa: E402
from report_publication import ReportOutput, publish_outputs  # noqa: E402

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
    "n128_k32:128:32",
    "n128_k128:128:128",
    "n128_k512:128:512",
    "n512_k128:512:128",
)
CHECKED_STATUSES = {"sampled-ok", "checked-ok"}

PROBE_FIELDNAMES = [
    "level",
    "routine",
    "kind",
    "library",
    "library_path",
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
    n: int
    k: int


@dataclass(frozen=True)
class RoutineSpec:
    name: str
    kind: str
    transposes: tuple[str, ...]
    alpha_must_be_real: bool
    beta_must_be_real: bool
    rank2k: bool


@dataclass(frozen=True)
class RankKCase:
    routine: RoutineSpec
    shape: Shape
    uplo: str
    trans: str
    alpha: str
    beta: str


ROUTINES = {
    spec.name: spec
    for spec in (
        RoutineSpec("ssyrk", "f32", ("N", "T"), True, True, False),
        RoutineSpec("dsyrk", "f64", ("N", "T"), True, True, False),
        RoutineSpec("csyrk", "c32", ("N", "T"), False, False, False),
        RoutineSpec("zsyrk", "c64", ("N", "T"), False, False, False),
        RoutineSpec("cherk", "c32", ("N", "C"), True, True, False),
        RoutineSpec("zherk", "c64", ("N", "C"), True, True, False),
        RoutineSpec("ssyr2k", "f32", ("N", "T"), True, True, True),
        RoutineSpec("dsyr2k", "f64", ("N", "T"), True, True, True),
        RoutineSpec("csyr2k", "c32", ("N", "T"), False, False, True),
        RoutineSpec("zsyr2k", "c64", ("N", "T"), False, False, True),
        RoutineSpec("cher2k", "c32", ("N", "C"), False, True, True),
        RoutineSpec("zher2k", "c64", ("N", "C"), False, True, True),
    )
}


def default_zynum_blas():
    if sys.platform == "darwin":
        return "zig-out/lib/libzynum_blas.dylib"
    if sys.platform == "win32":
        return "zig-out/bin/zynum_blas.dll"
    return "zig-out/lib/libzynum_blas.so"


def default_executable(path):
    return f"{path}.exe" if sys.platform == "win32" else path


def parse_shape_spec(value):
    parts = value.split(":")
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(f"shape must be NAME:N:K, got {value!r}")
    name = parts[0].strip()
    if not name:
        raise argparse.ArgumentTypeError("shape name must not be empty")
    try:
        n, k = (int(part, 10) for part in parts[1:])
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"shape dimensions must be integers, got {value!r}"
        ) from exc
    if n < 1 or k < 1:
        raise argparse.ArgumentTypeError(
            f"shape dimensions must be positive, got {value!r}"
        )
    return Shape(name, n, k)


def parse_scalar(value):
    parts = value.split(",")
    if len(parts) not in (1, 2) or any(not part.strip() for part in parts):
        raise argparse.ArgumentTypeError(f"scalar must be RE or RE,IM, got {value!r}")
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
            "Run SYRK/HERK/SYR2K/HER2K comparator cases with one fresh process per "
            "library/case/repeat and write an aggregate CSV."
        )
    )
    parser.add_argument(
        "--probe", default=default_executable("zig-out/bin/rank-k-probe")
    )
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
        help="Additional drop-in BLAS comparator. May be repeated.",
    )
    parser.add_argument(
        "--routine",
        action="append",
        type=routine_name,
        default=[],
        help="Rank-k or rank-2k routine to include. Defaults to all 12 routines.",
    )
    parser.add_argument(
        "--shape",
        action="append",
        type=parse_shape_spec,
        default=[],
        metavar="NAME:N:K",
        help="Rank-k shape. May be repeated.",
    )
    parser.add_argument(
        "--uplo",
        action="append",
        type=upper_choice(("U", "L")),
        default=[],
        help="Stored output triangle. Defaults to U and L.",
    )
    parser.add_argument(
        "--trans",
        action="append",
        type=upper_choice(("N", "T", "C")),
        default=[],
        help="Transpose filter. Defaults to every legal mode for each routine.",
    )
    parser.add_argument(
        "--alpha",
        action="append",
        default=[],
        help="Alpha as RE or RE,IM. May be repeated.",
    )
    parser.add_argument(
        "--beta",
        action="append",
        default=[],
        help="Beta as RE or RE,IM. May be repeated.",
    )
    parser.add_argument("--reps", type=int, default=5)
    parser.add_argument(
        "--process-repeats",
        type=int,
        default=3,
        help="Independent processes per library and complete rank-k case.",
    )
    parser.add_argument(
        "--process-schedule",
        choices=SCHEDULE_CHOICES,
        help=(
            "Fresh-process ordering; interleaved uses cyclic Latin rotations and "
            "requires repeats to be a multiple of the selected library count."
        ),
    )
    parser.add_argument(
        "--schedule",
        choices=SCHEDULE_CHOICES,
        dest="legacy_schedule",
        help="Compatibility alias for --process-schedule.",
    )
    parser.add_argument("--csv", required=True)
    parser.add_argument("--skip-missing", action="store_true")
    benchmark_metadata.add_identity_arguments(parser)
    args = parser.parse_args(normalize_negative_scalar_args(argv))
    if (
        args.process_schedule is not None
        and args.legacy_schedule is not None
        and args.process_schedule != args.legacy_schedule
    ):
        parser.error("--process-schedule conflicts with --schedule")
    args.process_schedule = (
        args.process_schedule or args.legacy_schedule or "library-major"
    )
    del args.legacy_schedule
    if args.reps < 1:
        parser.error("--reps must be at least 1")
    if args.process_repeats < 1:
        parser.error("--process-repeats must be at least 1")
    try:
        args.alpha = [scalar_text(value) for value in (args.alpha or ["0.75"])]
        args.beta = [scalar_text(value) for value in (args.beta or ["0.25"])]
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


def requested_uplos(args):
    return unique_preserving_order(args.uplo or ("U", "L"))


def validate_scalars(routines, alphas, betas):
    has_complex_alpha = any(parse_scalar(value)[1] != 0 for value in alphas)
    real_alpha_routines = [
        routine.name for routine in routines if routine.alpha_must_be_real
    ]
    if has_complex_alpha and real_alpha_routines:
        raise ValueError(
            "complex alpha is not valid for the selected routines: "
            f"{','.join(real_alpha_routines)}"
        )
    has_complex_beta = any(parse_scalar(value)[1] != 0 for value in betas)
    real_beta_routines = [
        routine.name for routine in routines if routine.beta_must_be_real
    ]
    if has_complex_beta and real_beta_routines:
        raise ValueError(
            "complex beta is not valid for the selected routines: "
            f"{','.join(real_beta_routines)}"
        )


def requested_cases(args):
    shapes = requested_shapes(args)
    routines = requested_routines(args)
    validate_scalars(routines, args.alpha, args.beta)
    selected_transposes = set(args.trans) if args.trans else None
    cases = []
    for routine in routines:
        transposes = [
            trans
            for trans in routine.transposes
            if selected_transposes is None or trans in selected_transposes
        ]
        for shape in shapes:
            for uplo in requested_uplos(args):
                for trans in transposes:
                    for alpha in args.alpha:
                        for beta in args.beta:
                            cases.append(
                                RankKCase(routine, shape, uplo, trans, alpha, beta)
                            )
    if not cases:
        raise ValueError("the selected routines and transpose filters produce no cases")
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
    result.extend(
        (label, path) for label, path in candidates if path and path != "none"
    )
    return result


def library_available(path):
    if Path(path).exists():
        return True
    if platform_image_path(path):
        return True
    return "/" not in path and "\\" not in path


def platform_image_path(path):
    return sys.platform == "darwin" and path == DEFAULT_ACCELERATE


def library_artifact_request(name, path):
    if name == "Accelerate" and platform_image_path(path) and not Path(path).exists():
        return benchmark_artifacts.ArtifactRequest.platform_image(name, path)
    return benchmark_artifacts.ArtifactRequest.library(name, path)


def selected_libraries(args):
    result = []
    for index, (label, path) in enumerate(libraries(args)):
        if library_available(path):
            result.append((label, path))
            continue
        if index == 0 or not args.skip_missing:
            raise ValueError(f"BLAS library is not available: {label}={path}")
        print(f"[rank-k] skipping missing comparator {label}={path}", file=sys.stderr)
    return result


def case_command(args, library_name, library_path, case, *, probe_path=None):
    return [
        args.probe if probe_path is None else probe_path,
        "--blas",
        library_path,
        "--library",
        library_name,
        "--routine",
        case.routine.name,
        "--shape",
        case.shape.name,
        "--n",
        str(case.shape.n),
        "--k",
        str(case.shape.k),
        "--uplo",
        case.uplo,
        "--trans",
        case.trans,
        "--alpha",
        case.alpha,
        "--beta",
        case.beta,
        "--reps",
        str(args.reps),
    ]


def flop_count(case):
    triangle = case.shape.n * (case.shape.n + 1) // 2
    factor = 8 if case.routine.kind.startswith("c") else 2
    if case.routine.rank2k:
        factor *= 2
    return factor * triangle * case.shape.k


def error_row(args, library_name, library_path, case, detail):
    alpha_re, alpha_im = parse_scalar(case.alpha)
    beta_re, beta_im = parse_scalar(case.beta)
    lda = case.shape.n if case.trans == "N" else case.shape.k
    return {
        "level": "level3",
        "routine": case.routine.name,
        "kind": case.routine.kind,
        "library": library_name,
        "library_path": library_path,
        "shape": case.shape.name,
        "n": str(case.shape.n),
        "k": str(case.shape.k),
        "uplo": case.uplo,
        "trans": case.trans,
        "alpha_re": format(alpha_re, ".17g"),
        "alpha_im": format(alpha_im, ".17g"),
        "beta_re": format(beta_re, ".17g"),
        "beta_im": format(beta_im, ".17g"),
        "lda": str(lda),
        "ldb": str(lda) if case.routine.rank2k else "",
        "ldc": str(case.shape.n),
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
    expected = {
        "routine": case.routine.name,
        "kind": case.routine.kind,
        "library": library_name,
        "library_path": library_path,
        "shape": case.shape.name,
        "n": str(case.shape.n),
        "k": str(case.shape.k),
        "uplo": case.uplo,
        "trans": case.trans,
        "lda": str(case.shape.n if case.trans == "N" else case.shape.k),
        "ldb": str(case.shape.n if case.trans == "N" else case.shape.k)
        if case.routine.rank2k
        else "",
        "ldc": str(case.shape.n),
        "reps": str(args.reps),
        "flop_count": str(flop_count(case)),
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


def run_one_process(
    args,
    library_name,
    library_path,
    case,
    *,
    probe_path,
    public_library_path,
    redact_private_paths,
):
    command = case_command(
        args, library_name, library_path, case, probe_path=probe_path
    )
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False,
            env=child_environment(),
        )
    except OSError as exc:
        return error_row(
            args,
            library_name,
            public_library_path,
            case,
            redact_private_paths(str(exc)),
        )
    if result.returncode != 0:
        detail = f"exit={result.returncode}"
        output = " ".join(
            part.strip() for part in (result.stdout, result.stderr) if part.strip()
        )
        if output:
            detail += f" {output}"
        return error_row(
            args,
            library_name,
            public_library_path,
            case,
            redact_private_paths(detail),
        )
    rows = list(csv.DictReader(result.stdout.splitlines()))
    if len(rows) != 1:
        return error_row(
            args,
            library_name,
            public_library_path,
            case,
            redact_private_paths(f"probe returned {len(rows)} rows"),
        )
    row = rows[0]
    missing_fields = [field for field in PROBE_FIELDNAMES if field not in row]
    if missing_fields:
        return error_row(
            args,
            library_name,
            public_library_path,
            case,
            redact_private_paths(
                f"probe row missing fields: {','.join(missing_fields)}"
            ),
        )
    mismatches = probe_row_matches(args, row, library_name, library_path, case)
    if mismatches:
        return error_row(
            args,
            library_name,
            public_library_path,
            case,
            redact_private_paths("probe row mismatch: " + "; ".join(mismatches)),
        )
    result_row = {field: row.get(field, "") for field in PROBE_FIELDNAMES}
    result_row["library_path"] = public_library_path
    return redact_private_paths(result_row)


def repeat_row_eligible(row):
    if row.get("status") != "ok" or row.get("check_status") not in CHECKED_STATUSES:
        return False
    try:
        parse_positive_finite(row["median_gflops"], "median_gflops")
        return int(row["median_ns"]) > 0
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
    for repeat, row in enumerate(rows, 1):
        if row.get("status") == "ok":
            try:
                validate_performance_fields(
                    row,
                    required=(
                        "best_ns",
                        "median_ns",
                        "p95_ns",
                        "max_ns",
                        "gflops",
                        "median_gflops",
                    ),
                )
            except ValueError as exc:
                raise ValueError(
                    f"invalid rank-k performance evidence in repeat {repeat}: {exc}"
                ) from exc
    eligible = [row for row in rows if repeat_row_eligible(row)]
    base = dict(
        max(eligible, key=lambda row: float(row["gflops"])) if eligible else rows[0]
    )
    values = [float(row["median_gflops"]) for row in eligible]
    summary = {
        "metric_min": format(min(values), ".17g") if values else "",
        "metric_median": (
            format(positive_finite_median(values, "metric_median"), ".17g")
            if values
            else ""
        ),
        "metric_max": format(max(values), ".17g") if values else "",
        "metric_samples": ",".join(format(value, ".17g") for value in values),
    }
    validate_optional_metric_evidence(summary)
    base.update(
        {
            "process_repeats": len(rows),
            "successful_repeats": len(eligible),
            **summary,
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
        result = subprocess.run(command, check=True, capture_output=True, text=True)
    except Exception:
        return None
    return result.stdout.strip()


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


def serialize_metadata(args, selected, cases, benchmark_identity, artifacts):
    source = benchmark_identity["source"]
    probe_record = artifacts.legacy_records("binary")[0]
    metadata = {
        "generated_at_unix": time.time(),
        "argv": sys.argv,
        "cwd": os.getcwd(),
        "platform": platform.platform(),
        "python_version": sys.version,
        "zig_version": command_output(["zig", "version"]),
        "source": benchmark_metadata.legacy_source_snapshot(source),
        "detected_cpu_count": os.cpu_count(),
        "zynum_maximum_threads": zynum_maximum_threads_detected(),
        "reps": args.reps,
        "process_repeats": args.process_repeats,
        "schedule": args.process_schedule,
        "isolation": "fresh process per library/routine/shape/uplo/trans/scalars/repeat",
        "process_metric": "probe median_gflops",
        "correctness_check": (
            "independent full or sampled scalar reference before timing, including "
            "unstored-triangle preservation and Hermitian diagonals"
        ),
        "case_count_per_library": len(cases),
        "environment": environment_snapshot(),
        "probe": {
            "path": probe_record["path"],
            "sha256": probe_record["sha256"],
        },
        "libraries": artifacts.legacy_records("library"),
        "benchmark_identity": benchmark_identity,
        "shapes": [
            {"name": shape.name, "n": shape.n, "k": shape.k}
            for shape in requested_shapes(args)
        ],
        "routines": [routine.name for routine in requested_routines(args)],
        "uplos": requested_uplos(args),
        "transposes": args.trans or "all legal modes per routine",
        "alphas": args.alpha,
        "betas": args.beta,
    }
    return benchmark_metadata.serialize_public_metadata(
        metadata,
        controller="run_rank_k_report.py",
        parameter_keys=(
            "reps",
            "process_repeats",
            "schedule",
            "shapes",
            "routines",
            "uplos",
            "transposes",
            "alphas",
            "betas",
        ),
    )


def serialize_csv(rows):
    file = io.StringIO(newline="")
    writer = csv.DictWriter(file, fieldnames=CSV_FIELDNAMES)
    writer.writeheader()
    writer.writerows(rows)
    return file.getvalue().encode("utf-8")


def run_controller(args):
    probe = Path(args.probe)
    if not probe.is_file():
        raise ValueError(f"rank-k probe is not available: {args.probe}")
    cases = requested_cases(args)
    selected = selected_libraries(args)
    validate_unique_library_labels(selected)
    validate_schedule(len(selected), args.process_repeats, args.process_schedule)

    requests = [
        benchmark_artifacts.ArtifactRequest.binary("rank_k_probe", args.probe),
        *(library_artifact_request(name, path) for name, path in selected),
    ]
    artifacts = benchmark_artifacts.ArtifactSnapshotSet.capture(requests)
    outputs = None
    try:
        frozen_probe = artifacts.for_role("binary")[0]
        frozen_libraries = artifacts.for_role("library")
        benchmark_identity = benchmark_metadata.collect_benchmark_identity_from_frozen(
            args,
            libraries=frozen_libraries,
            binaries=(frozen_probe,),
        )

        def announce(library_index, case_index, repeat_index):
            library_name, _ = selected[library_index]
            case = cases[case_index]
            repeat = (
                ""
                if repeat_index is None
                else f" repeat={repeat_index + 1}/{args.process_repeats}"
            )
            print(
                f"[rank-k {library_name}] case={case_index + 1}/{len(cases)}{repeat} "
                f"{case.routine.name} shape={case.shape.name} "
                f"n={case.shape.n} k={case.shape.k} uplo={case.uplo} "
                f"trans={case.trans} alpha={case.alpha} beta={case.beta}",
                file=sys.stderr,
                flush=True,
            )

        def run_one(library_index, case_index, _repeat_index):
            library_name, public_library_path = selected[library_index]
            return run_one_process(
                args,
                library_name,
                frozen_libraries[library_index].execution_path,
                cases[case_index],
                probe_path=frozen_probe.execution_path,
                public_library_path=public_library_path,
                redact_private_paths=artifacts.redact_private_paths,
            )

        samples = collect_repeats(
            selected,
            cases,
            args.process_repeats,
            args.process_schedule,
            run_one,
            announce,
        )
        samples = artifacts.redact_private_paths(samples)
        rows = []
        for library_index, _ in enumerate(selected):
            for case_index, _ in enumerate(cases):
                rows.append(aggregate_repeats(samples[library_index][case_index]))

        output = Path(args.csv)
        csv_contents = artifacts.redact_private_paths(serialize_csv(rows))
        metadata_contents = artifacts.redact_private_paths(
            serialize_metadata(args, selected, cases, benchmark_identity, artifacts)
        )
        metadata_path = output.with_suffix(output.suffix + ".meta.json")
        outputs = [
            ReportOutput(output, csv_contents),
            ReportOutput(metadata_path, metadata_contents),
        ]
        artifacts.finalize()
    finally:
        artifacts.close()
    publish_outputs(outputs)


def main(argv=None):
    args = parse_args(argv)
    try:
        run_controller(args)
    except (ValueError, benchmark_artifacts.ArtifactSnapshotError) as exc:
        print(exc, file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
