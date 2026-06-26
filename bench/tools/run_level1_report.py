#!/usr/bin/env python3
# Copyright (C) 2026 Zynum contributors
# SPDX-License-Identifier: LGPL-3.0-or-later

import argparse
import csv
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

DEFAULT_ACCELERATE = "/System/Library/Frameworks/Accelerate.framework/Accelerate"
DEFAULT_OPENBLAS = "/opt/homebrew/opt/openblas/lib/libopenblas.dylib"

LEVEL1_OPS = [
    ("real_f32", "sscal"),
    ("real_f32", "saxpy"),
    ("real_f32", "sdot"),
    ("real_f32", "sasum"),
    ("real_f32", "snrm2"),
    ("real_f64", "dscal"),
    ("real_f64", "daxpy"),
    ("real_f64", "ddot"),
    ("real_f64", "dasum"),
    ("real_f64", "dnrm2"),
    ("complex_f32", "csscal"),
    ("complex_f32", "cscal"),
    ("complex_f32", "caxpy"),
    ("complex_f32", "caxpby"),
    ("complex_f32", "cdotu"),
    ("complex_f32", "cdotc"),
    ("complex_f32", "scasum"),
    ("complex_f32", "scnrm2"),
    ("complex_f64", "zdscal"),
    ("complex_f64", "zscal"),
    ("complex_f64", "zaxpy"),
    ("complex_f64", "zaxpby"),
    ("complex_f64", "zdotu"),
    ("complex_f64", "zdotc"),
    ("complex_f64", "dzasum"),
    ("complex_f64", "dznrm2"),
]

COPY_OPS = [
    ("copy", "scopy", "s"),
    ("copy", "dcopy", "d"),
    ("copy", "ccopy", "c"),
    ("copy", "zcopy", "z"),
]

LIBS = [
    ("Zynum", "zynum"),
    ("Accelerate", "accelerate"),
    ("OpenBLAS", "openblas"),
]

RESULT_RE = re.compile(
    r"(?:rate_Gops=(?P<gops>[-+0-9.eE]+)\s+)?bandwidth_GBps=(?P<gbps>[-+0-9.eE]+)"
)


def default_zynum_blas():
    if sys.platform == "darwin":
        return "zig-out/lib/libzynum_blas.dylib"
    if sys.platform == "win32":
        return "zig-out/bin/zynum_blas.dll"
    return "zig-out/lib/libzynum_blas.so"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run representative Level 1 fresh-process probes and write a report CSV."
    )
    parser.add_argument(
        "--level1-probe", default="zig-out/perf-report/bin/level1_probe"
    )
    parser.add_argument("--copy-probe", default="zig-out/perf-report/bin/dcopy_probe")
    parser.add_argument("--zynum", default=default_zynum_blas())
    parser.add_argument("--accelerate", default=DEFAULT_ACCELERATE)
    parser.add_argument("--openblas", default=DEFAULT_OPENBLAS)
    parser.add_argument("--n", type=int, default=1024 * 1024)
    parser.add_argument("--seconds", type=int, default=1)
    parser.add_argument("--copy-seconds", type=int, default=1)
    parser.add_argument("--process-repeats", type=int, default=1)
    parser.add_argument("--csv", required=True)
    parser.add_argument("--skip-missing", action="store_true")
    args = parser.parse_args()
    if args.process_repeats < 1:
        parser.error("--process-repeats must be at least 1")
    return args


def sha256_file(path):
    candidate = Path(path)
    if not candidate.is_file():
        return None
    digest = hashlib.sha256()
    with candidate.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def zig_version():
    try:
        result = subprocess.run(
            ["zig", "version"],
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return None
    return result.stdout.strip()


def git_revision():
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return None
    return result.stdout.strip()


def environment_snapshot():
    names = [
        "ZYNUM_MAXIMUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "OPENBLAS_DYNAMIC",
        "VECLIB_MAXIMUM_THREADS",
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "MKL_DYNAMIC",
    ]
    return {name: os.environ.get(name, "unset") for name in names}


def zynum_maximum_threads_detected():
    value = os.environ.get("ZYNUM_MAXIMUM_THREADS")
    if value:
        try:
            parsed = int(value, 10)
            if parsed > 0:
                return parsed
        except ValueError:
            pass
    return max(1, os.cpu_count() or 1)


def libraries(args):
    return [
        ("Zynum", args.zynum),
        ("Accelerate", args.accelerate),
        ("OpenBLAS", args.openblas),
    ]


def parse_probe_output(output):
    match = RESULT_RE.search(output)
    if not match:
        return None, None
    gops = match.group("gops")
    gbps = match.group("gbps")
    return (float(gops) if gops is not None else None, float(gbps))


def run_once(cmd):
    result = subprocess.run(cmd, capture_output=True, text=True)
    output = (result.stdout or "") + (result.stderr or "")
    if result.returncode != 0:
        return {
            "status": "missing" if "MissingSymbol" in output or "MissingCopy" in output else "error",
            "returncode": result.returncode,
            "rate_gops": None,
            "bandwidth_gbps": None,
            "raw_output": output.strip(),
        }
    gops, gbps = parse_probe_output(output)
    if gbps is None:
        return {
            "status": "parse_error",
            "returncode": result.returncode,
            "rate_gops": None,
            "bandwidth_gbps": None,
            "raw_output": output.strip(),
        }
    return {
        "status": "ok",
        "returncode": result.returncode,
        "rate_gops": gops,
        "bandwidth_gbps": gbps,
        "raw_output": output.strip(),
    }


def metric_value(row):
    if row["metric"] == "bandwidth_gbps":
        return row["bandwidth_gbps"]
    return row["rate_gops"]


def choose_best(rows):
    ok_rows = [row for row in rows if row["status"] == "ok" and metric_value(row) is not None]
    if ok_rows:
        return max(ok_rows, key=metric_value)
    return rows[0]


def run_level1_op(args, library_name, library_path, group, op):
    rows = []
    cmd = [
        args.level1_probe,
        "--lib",
        library_path,
        "--op",
        op,
        "--n",
        str(args.n),
        "--seconds",
        str(args.seconds),
    ]
    for repeat in range(args.process_repeats):
        result = run_once(cmd)
        rows.append(
            {
                "group": group,
                "op": op,
                "library": library_name,
                "library_path": library_path,
                "n": args.n,
                "seconds": args.seconds,
                "repeat": repeat,
                "metric": "rate_gops",
                **result,
            }
        )
    return choose_best(rows)


def run_copy_op(args, library_name, library_path, group, op, kind):
    rows = []
    cmd = [
        args.copy_probe,
        "--lib",
        library_path,
        "--kind",
        kind,
        "--n",
        str(args.n),
        "--seconds",
        str(args.copy_seconds),
    ]
    for repeat in range(args.process_repeats):
        result = run_once(cmd)
        rows.append(
            {
                "group": group,
                "op": op,
                "library": library_name,
                "library_path": library_path,
                "n": args.n,
                "seconds": args.copy_seconds,
                "repeat": repeat,
                "metric": "bandwidth_gbps",
                **result,
            }
        )
    return choose_best(rows)


def main():
    args = parse_args()
    if not Path(args.level1_probe).exists():
        print(f"missing --level1-probe {args.level1_probe}", file=sys.stderr)
        return 2
    if not Path(args.copy_probe).exists():
        print(f"missing --copy-probe {args.copy_probe}", file=sys.stderr)
        return 2

    output_path = Path(args.csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for library_name, library_path in libraries(args):
        if args.skip_missing and not Path(library_path).exists():
            continue
        for group, op, kind in COPY_OPS:
            row = run_copy_op(args, library_name, library_path, group, op, kind)
            rows.append(row)
            print(f"{library_name:10s} {op:8s} {row['status']:11s} {metric_value(row)}")
        for group, op in LEVEL1_OPS:
            row = run_level1_op(args, library_name, library_path, group, op)
            rows.append(row)
            print(f"{library_name:10s} {op:8s} {row['status']:11s} {metric_value(row)}")

    fields = [
        "group",
        "op",
        "library",
        "library_path",
        "n",
        "seconds",
        "repeat",
        "metric",
        "status",
        "returncode",
        "rate_gops",
        "bandwidth_gbps",
        "raw_output",
    ]
    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})

    meta = {
        "generated_at_unix": time.time(),
        "isolation": "fresh process per library/op/repeat; best repeat kept",
        "n": args.n,
        "seconds": args.seconds,
        "copy_seconds": args.copy_seconds,
        "process_repeats": args.process_repeats,
        "environment": environment_snapshot(),
        "zynum_maximum_threads_detected": zynum_maximum_threads_detected(),
        "zig_version": zig_version(),
        "git_revision": git_revision(),
        "probes": {
            "level1_probe": args.level1_probe,
            "level1_probe_sha256": sha256_file(args.level1_probe),
            "copy_probe": args.copy_probe,
            "copy_probe_sha256": sha256_file(args.copy_probe),
        },
        "libraries": {
            name: {"path": path, "sha256": sha256_file(path)}
            for name, path in libraries(args)
        },
    }
    with output_path.with_suffix(output_path.suffix + ".meta.json").open("w") as f:
        json.dump(meta, f, indent=2, sort_keys=True)
        f.write("\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
