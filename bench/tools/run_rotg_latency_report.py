#!/usr/bin/env python3
# Copyright (C) 2026 Zynum contributors
# SPDX-License-Identifier: LGPL-3.0-or-later

import argparse
import csv
import ctypes
import hashlib
import json
import math
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
CHECKED_STATUSES = {"sampled-ok", "checked-ok"}

ROTG_CASES = (
    "zero",
    "a_zero",
    "b_zero",
    "balanced",
    "a_dominant",
    "b_dominant",
    "tiny_exponent",
    "huge_exponent",
    "mixed_exponent",
)
ROTMG_CASES = (
    "flag_neg2_zero_p2",
    "flag_neg1_negative_d1",
    "flag_neg1_negative_q2",
    "flag_zero_q1_dominant",
    "flag_one_q2_dominant",
    "flag_neg1_tiny_scale",
    "flag_neg1_huge_scale",
)
EXPECTED_FLAGS = {
    "flag_neg2_zero_p2": -2.0,
    "flag_neg1_negative_d1": -1.0,
    "flag_neg1_negative_q2": -1.0,
    "flag_zero_q1_dominant": 0.0,
    "flag_one_q2_dominant": 1.0,
    "flag_neg1_tiny_scale": -1.0,
    "flag_neg1_huge_scale": -1.0,
}
ROUTINES = {
    "srotg": ("f32", False),
    "drotg": ("f64", False),
    "crotg": ("c32", False),
    "zrotg": ("c64", False),
    "srotmg": ("f32", True),
    "drotmg": ("f64", True),
}

PROBE_FIELDNAMES = [
    "level",
    "routine",
    "kind",
    "library",
    "library_path",
    "case",
    "corpus_size",
    "samples",
    "calls_per_sample",
    "total_calls",
    "best_ns_per_call",
    "median_ns_per_call",
    "p95_ns_per_call",
    "max_ns_per_call",
    "median_full_ns_per_call",
    "median_harness_ns_per_call",
    "nonpositive_pairs",
    "metric",
    "status",
    "check_status",
    "check_max_abs_error",
    "check_max_rel_error",
    "check_samples",
    "expected_flag",
    "observed_flag",
    "checksum",
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
class LatencyCase:
    routine: str
    input_case: str


def default_zynum_blas():
    if sys.platform == "darwin":
        return "zig-out/lib/libzynum_blas.dylib"
    if sys.platform == "win32":
        return "zig-out/bin/zynum_blas.dll"
    return "zig-out/lib/libzynum_blas.so"


def routine_name(value):
    result = value.lower()
    if result not in ROUTINES:
        raise argparse.ArgumentTypeError(
            "unknown routine {!r}; choose from {}".format(value, ",".join(ROUTINES))
        )
    return result


def case_name(value):
    result = value.lower()
    if result not in ROTG_CASES and result not in ROTMG_CASES:
        raise argparse.ArgumentTypeError("unknown latency corpus case {!r}".format(value))
    return result


def positive_int(value):
    try:
        result = int(value, 10)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected a positive integer") from exc
    if result < 1:
        raise argparse.ArgumentTypeError("expected a positive integer")
    return result


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Run ROTG/ROTMG scalar-generator latency cases with one fresh process "
            "per library/routine/corpus/repeat and write an aggregate CSV."
        )
    )
    parser.add_argument("--probe", default="zig-out/bin/rotg-latency-probe")
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
    parser.add_argument("--routine", action="append", type=routine_name, default=[])
    parser.add_argument(
        "--case",
        action="append",
        type=case_name,
        default=[],
        help="Corpus case to include. May be repeated.",
    )
    parser.add_argument("--samples", type=positive_int, default=9)
    parser.add_argument("--calls-per-sample", type=positive_int, default=100_000)
    parser.add_argument("--process-repeats", type=positive_int, default=3)
    parser.add_argument("--csv", required=True)
    parser.add_argument("--skip-missing", action="store_true")
    return parser.parse_args(argv)


def unique_preserving_order(values):
    return list(dict.fromkeys(values))


def cases_for_routine(routine):
    return ROTMG_CASES if ROUTINES[routine][1] else ROTG_CASES


def requested_cases(args):
    routines = unique_preserving_order(args.routine or ROUTINES.keys())
    selected_cases = unique_preserving_order(args.case)
    supported_cases = set()
    for routine in routines:
        supported_cases.update(cases_for_routine(routine))
    incompatible = [value for value in selected_cases if value not in supported_cases]
    if incompatible:
        raise ValueError(
            "corpus case(s) {} are not valid for {}".format(
                ",".join(incompatible), ",".join(routines)
            )
        )
    result = []
    for routine in routines:
        compatible = cases_for_routine(routine)
        if selected_cases:
            routine_cases = [value for value in selected_cases if value in compatible]
        else:
            routine_cases = compatible
        result.extend(LatencyCase(routine, value) for value in routine_cases)
    if not result:
        raise ValueError("no latency cases selected")
    return result


def append_extra_blas(candidates, items):
    for item in items:
        if "=" not in item:
            raise ValueError("--extra-blas must be LABEL=PATH, got {!r}".format(item))
        label, path = (part.strip() for part in item.split("=", 1))
        if not label or not path:
            raise ValueError("--extra-blas must be LABEL=PATH, got {!r}".format(item))
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
    try:
        ctypes.CDLL(path)
        return True
    except OSError:
        return False


def selected_libraries(args):
    result = []
    for index, (label, path) in enumerate(libraries(args)):
        if library_available(path):
            result.append((label, path))
            continue
        if index == 0 or not args.skip_missing:
            raise ValueError("BLAS library is not available: {}={}".format(label, path))
        print(
            "[rotg-latency] skipping missing comparator {}={}".format(label, path),
            file=sys.stderr,
        )
    return result


def case_command(args, library_name, library_path, case):
    return [
        args.probe,
        "--blas",
        library_path,
        "--library",
        library_name,
        "--routine",
        case.routine,
        "--case",
        case.input_case,
        "--samples",
        str(args.samples),
        "--calls-per-sample",
        str(args.calls_per_sample),
    ]


def expected_flag_text(case):
    value = EXPECTED_FLAGS.get(case.input_case)
    return "" if value is None else format(value, ".17g")


def error_row(args, library_name, library_path, case, detail):
    return {
        "level": "level1",
        "routine": case.routine,
        "kind": ROUTINES[case.routine][0],
        "library": library_name,
        "library_path": library_path,
        "case": case.input_case,
        "corpus_size": "",
        "samples": str(args.samples),
        "calls_per_sample": str(args.calls_per_sample),
        "total_calls": str(args.samples * args.calls_per_sample),
        "best_ns_per_call": "",
        "median_ns_per_call": "",
        "p95_ns_per_call": "",
        "max_ns_per_call": "",
        "median_full_ns_per_call": "",
        "median_harness_ns_per_call": "",
        "nonpositive_pairs": "",
        "metric": "ns_per_call",
        "status": "error",
        "check_status": "error",
        "check_max_abs_error": "",
        "check_max_rel_error": "",
        "check_samples": "0",
        "expected_flag": expected_flag_text(case),
        "observed_flag": "",
        "checksum": "",
        "check_raw_output": detail,
    }


def probe_row_mismatches(args, row, library_name, library_path, case):
    expected = {
        "level": "level1",
        "routine": case.routine,
        "kind": ROUTINES[case.routine][0],
        "library": library_name,
        "library_path": library_path,
        "case": case.input_case,
        "samples": str(args.samples),
        "calls_per_sample": str(args.calls_per_sample),
        "total_calls": str(args.samples * args.calls_per_sample),
        "metric": "ns_per_call",
    }
    mismatches = [
        "{}={!r} expected {!r}".format(field, row.get(field), value)
        for field, value in expected.items()
        if row.get(field) != value
    ]
    expected_flag = EXPECTED_FLAGS.get(case.input_case)
    if expected_flag is None:
        if row.get("expected_flag") not in (None, ""):
            mismatches.append("expected_flag must be empty for ROTG")
    else:
        try:
            flag_matches = float(row["expected_flag"]) == expected_flag
        except (KeyError, TypeError, ValueError):
            flag_matches = False
        if not flag_matches:
            mismatches.append(
                "expected_flag={!r} expected {}".format(
                    row.get("expected_flag"), expected_flag_text(case)
                )
            )
    return mismatches


def child_environment():
    env = os.environ.copy()
    env.setdefault("OPENBLAS_DYNAMIC", "0")
    return env


def run_one_process(args, library_name, library_path, case):
    command = case_command(args, library_name, library_path, case)
    result = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
        check=False,
        env=child_environment(),
    )
    if result.returncode != 0:
        detail = "exit={}".format(result.returncode)
        output = " ".join(
            part.strip() for part in (result.stdout, result.stderr) if part.strip()
        )
        if output:
            detail += " " + output
        return error_row(args, library_name, library_path, case, detail)
    rows = list(csv.DictReader(result.stdout.splitlines()))
    if len(rows) != 1:
        return error_row(
            args,
            library_name,
            library_path,
            case,
            "probe returned {} rows".format(len(rows)),
        )
    row = rows[0]
    missing_fields = [field for field in PROBE_FIELDNAMES if field not in row]
    if missing_fields:
        return error_row(
            args,
            library_name,
            library_path,
            case,
            "probe row missing fields: {}".format(",".join(missing_fields)),
        )
    mismatches = probe_row_mismatches(
        args, row, library_name, library_path, case
    )
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
        value = float(row["median_ns_per_call"])
        return (
            math.isfinite(value)
            and value > 0
            and int(row["nonpositive_pairs"]) == 0
        )
    except (KeyError, TypeError, ValueError):
        return False


def failure_status(rows):
    statuses = {row.get("status", "error") for row in rows}
    checks = {row.get("check_status", "error") for row in rows}
    if "error" in statuses or "error" in checks:
        return "error", "error"
    if "correctness_failed" in statuses or "correctness_failed" in checks:
        return "correctness_failed", "correctness_failed"
    return "timing_failed", "checked-ok"


def aggregate_repeats(rows):
    if not rows:
        raise ValueError("cannot aggregate an empty repeat list")
    eligible = [row for row in rows if repeat_row_eligible(row)]
    base = dict(
        min(eligible, key=lambda row: float(row["median_ns_per_call"]))
        if eligible
        else rows[0]
    )
    values = [float(row["median_ns_per_call"]) for row in eligible]
    base.update(
        {
            "process_repeats": len(rows),
            "successful_repeats": len(eligible),
            "metric_min": format(min(values), ".17g") if values else "",
            "metric_median": (
                format(statistics.median(values), ".17g") if values else ""
            ),
            "metric_max": format(max(values), ".17g") if values else "",
            "metric_samples": ",".join(format(value, ".17g") for value in values),
        }
    )
    abs_errors = []
    rel_errors = []
    details = []
    for repeat, row in enumerate(rows, 1):
        try:
            abs_errors.append(float(row.get("check_max_abs_error") or 0))
            rel_errors.append(float(row.get("check_max_rel_error") or 0))
        except ValueError:
            pass
        if not repeat_row_eligible(row) or row.get("check_raw_output"):
            detail = "repeat={}: status={} check_status={}".format(
                repeat, row.get("status", ""), row.get("check_status", "")
            )
            if row.get("check_raw_output"):
                detail += " " + row["check_raw_output"]
            details.append(detail)
    if abs_errors:
        base["check_max_abs_error"] = format(max(abs_errors), ".9g")
    if rel_errors:
        base["check_max_rel_error"] = format(max(rel_errors), ".9g")
    if details:
        base["check_raw_output"] = " | ".join(details)
    if len(eligible) != len(rows):
        base["status"], base["check_status"] = failure_status(rows)
    return base


def command_output(command):
    try:
        result = subprocess.run(
            command,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
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
        "samples": args.samples,
        "calls_per_sample": args.calls_per_sample,
        "process_repeats": args.process_repeats,
        "isolation": "fresh process per library/routine/corpus/process repeat",
        "process_metric": "paired-harness-subtracted median ns/call",
        "aggregate_metric": "median of per-process median ns/call",
        "correctness_check": (
            "independent scalar reference over every corpus member before timing; "
            "ROTMG checks outputs, defined PARAM fields, and expected flag"
        ),
        "harness": (
            "alternating AB/BA paired full and baseline batches with identical input "
            "reset, loop, and output-bit consumption; only the BLAS call is omitted"
        ),
        "case_count_per_library": len(cases),
        "environment": environment_snapshot(),
        "probe": {"path": args.probe, "sha256": sha256_file(args.probe)},
        "libraries": [
            {"name": name, "path": path, "sha256": sha256_file(path)}
            for name, path in selected
        ],
        "cases": [
            {"routine": case.routine, "case": case.input_case} for case in cases
        ],
    }
    metadata_path = output.with_suffix(output.suffix + ".meta.json")
    with metadata_path.open("w") as file:
        json.dump(metadata, file, indent=2, sort_keys=True)
        file.write("\n")


def run_controller(args):
    probe = Path(args.probe)
    if not probe.is_file():
        raise ValueError("ROTG latency probe is not available: {}".format(args.probe))
    cases = requested_cases(args)
    selected = selected_libraries(args)
    rows = []
    for library_name, library_path in selected:
        for case_index, case in enumerate(cases, 1):
            print(
                "[rotg-latency {}] case={}/{} {} {}".format(
                    library_name,
                    case_index,
                    len(cases),
                    case.routine,
                    case.input_case,
                ),
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
