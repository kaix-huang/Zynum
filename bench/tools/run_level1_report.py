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


class ComplexF32(ctypes.Structure):
    _fields_ = [("re", ctypes.c_float), ("im", ctypes.c_float)]


class ComplexF64(ctypes.Structure):
    _fields_ = [("re", ctypes.c_double), ("im", ctypes.c_double)]


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
    parser.add_argument("--mkl")
    parser.add_argument("--aocl-blis")
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
        "BLIS_NUM_THREADS",
        "AOCL_DYNAMIC",
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
    result = [
        ("Zynum", args.zynum),
        ("Accelerate", args.accelerate),
        ("OpenBLAS", args.openblas),
    ]
    if args.mkl:
        result.append(("MKL", args.mkl))
    if args.aocl_blis:
        result.append(("AOCL-BLIS", args.aocl_blis))
    return result


def library_available(path):
    if Path(path).exists():
        return True
    try:
        ctypes.CDLL(path)
        return True
    except OSError:
        return False


def check_result(status, error=None, raw=""):
    return {
        "check_status": status,
        "check_max_abs_error": "" if error is None else f"{error:.9g}",
        "check_raw_output": raw,
    }


def next_fill(seed):
    seed[0] = (seed[0] * 6364136223846793005 + 1442695040888963407) & (
        (1 << 64) - 1
    )
    return ((seed[0] >> 32) % 1000) / 1000.0 - 0.5


def real_array(ctype, count, seed_value):
    seed = [seed_value]
    values = [ctype(next_fill(seed)).value for _ in range(count)]
    return (ctype * count)(*values), values


def complex_array(complex_type, scalar_type, count, seed_value):
    seed = [seed_value]
    values = [
        complex(scalar_type(next_fill(seed)).value, scalar_type(next_fill(seed)).value)
        for _ in range(count)
    ]
    return (complex_type * count)(*[complex_type(v.real, v.imag) for v in values]), values


def max_real_error(actual, expected):
    return max((abs(float(actual[i]) - expected[i]) for i in range(len(expected))), default=0.0)


def max_complex_error(actual, expected):
    return max(
        (
            abs(complex(float(actual[i].re), float(actual[i].im)) - expected[i])
            for i in range(len(expected))
        ),
        default=0.0,
    )


def complex_values(array):
    return [complex(float(v.re), float(v.im)) for v in array]


def lookup(lib, *names):
    for name in names:
        try:
            return getattr(lib, name)
        except AttributeError:
            pass
    return None


def check_copy_op(library_path, kind):
    try:
        lib = ctypes.CDLL(library_path)
    except OSError as exc:
        return check_result("missing", raw=str(exc))
    spec = {
        "s": ("scopy_", ctypes.sizeof(ctypes.c_float)),
        "d": ("dcopy_", ctypes.sizeof(ctypes.c_double)),
        "c": ("ccopy_", 2 * ctypes.sizeof(ctypes.c_float)),
        "z": ("zcopy_", 2 * ctypes.sizeof(ctypes.c_double)),
    }[kind]
    fn = lookup(lib, spec[0])
    if fn is None:
        return check_result("missing", raw=f"missing {spec[0]}")
    n = 257
    byte_count = n * spec[1]
    x = (ctypes.c_ubyte * byte_count)()
    y = (ctypes.c_ubyte * byte_count)()
    seed = [0x123456789ABCDEF0]
    for i in range(byte_count):
        seed[0] = (seed[0] * 6364136223846793005 + 1442695040888963407) & (
            (1 << 64) - 1
        )
        x[i] = (seed[0] >> 32) & 0xFF
    ni = ctypes.c_int(n)
    inc = ctypes.c_int(1)
    fn(ctypes.byref(ni), x, ctypes.byref(inc), y, ctypes.byref(inc))
    mismatches = sum(1 for i in range(byte_count) if x[i] != y[i])
    if mismatches:
        return check_result("correctness_failed", float(mismatches), f"{mismatches} byte mismatches")
    return check_result("sampled-ok", 0.0)


def call_complex_dot(lib, op, n, x, y, complex_type):
    inc = ctypes.c_int(1)
    ni = ctypes.c_int(n)
    out = complex_type()
    fn = lookup(lib, f"cblas_{op}_sub", f"{op}_sub_")
    if fn is None:
        return None
    if getattr(fn, "__name__", "").startswith("cblas_"):
        fn(ni, x, inc, y, inc, ctypes.byref(out))
    else:
        fn(ctypes.byref(ni), x, ctypes.byref(inc), y, ctypes.byref(inc), ctypes.byref(out))
    return complex(float(out.re), float(out.im))


def check_level1_op(library_path, op):
    try:
        lib = ctypes.CDLL(library_path)
    except OSError as exc:
        return check_result("missing", raw=str(exc))
    n = 257
    ni = ctypes.c_int(n)
    inc = ctypes.c_int(1)
    try:
        if op[0] in "sd" and op not in ("scasum", "dzasum", "scnrm2", "dznrm2"):
            ctype = ctypes.c_float if op[0] == "s" else ctypes.c_double
            tol = 1e-3 if ctype is ctypes.c_float else 1e-10
            x, x0 = real_array(ctype, n, 0x123456789ABCDEF0)
            y, y0 = real_array(ctype, n, 0x0FEDCBA987654321)
            alpha = ctype(0.75)
            if op.endswith("scal"):
                fn = lookup(lib, f"{op}_")
                if fn is None:
                    return check_result("missing", raw=f"missing {op}_")
                fn(ctypes.byref(ni), ctypes.byref(alpha), x, ctypes.byref(inc))
                err = max_real_error(x, [alpha.value * v for v in x0])
            elif op.endswith("axpy"):
                fn = lookup(lib, f"{op}_")
                if fn is None:
                    return check_result("missing", raw=f"missing {op}_")
                fn(ctypes.byref(ni), ctypes.byref(alpha), x, ctypes.byref(inc), y, ctypes.byref(inc))
                err = max_real_error(y, [alpha.value * x0[i] + y0[i] for i in range(n)])
            elif op.endswith("dot"):
                fn = lookup(lib, f"cblas_{op}", f"{op}_")
                if fn is None:
                    return check_result("missing", raw=f"missing cblas_{op}/{op}_")
                fn.restype = ctype
                if getattr(fn, "__name__", "").startswith("cblas_"):
                    got = float(fn(ni, x, inc, y, inc))
                else:
                    got = float(fn(ctypes.byref(ni), x, ctypes.byref(inc), y, ctypes.byref(inc)))
                err = abs(got - sum(x0[i] * y0[i] for i in range(n)))
                tol *= n
            elif op.endswith("asum"):
                fn = lookup(lib, f"cblas_{op}", f"{op}_")
                if fn is None:
                    return check_result("missing", raw=f"missing cblas_{op}/{op}_")
                fn.restype = ctype
                if getattr(fn, "__name__", "").startswith("cblas_"):
                    got = float(fn(ni, x, inc))
                else:
                    got = float(fn(ctypes.byref(ni), x, ctypes.byref(inc)))
                err = abs(got - sum(abs(v) for v in x0))
                tol *= n
            elif op.endswith("nrm2"):
                fn = lookup(lib, f"cblas_{op}", f"{op}_")
                if fn is None:
                    return check_result("missing", raw=f"missing cblas_{op}/{op}_")
                fn.restype = ctype
                if getattr(fn, "__name__", "").startswith("cblas_"):
                    got = float(fn(ni, x, inc))
                else:
                    got = float(fn(ctypes.byref(ni), x, ctypes.byref(inc)))
                err = abs(got - math.sqrt(sum(v * v for v in x0)))
                tol *= n
            else:
                return check_result("error", raw=f"unhandled op {op}")
            return check_result("sampled-ok" if err <= tol else "correctness_failed", err)

        complex_type = ComplexF32 if op[0] in ("c", "s") else ComplexF64
        scalar_type = ctypes.c_float if complex_type is ComplexF32 else ctypes.c_double
        real_tol = 1e-3 if scalar_type is ctypes.c_float else 1e-10
        x, x0 = complex_array(complex_type, scalar_type, n, 0x123456789ABCDEF0)
        y, y0 = complex_array(complex_type, scalar_type, n, 0x0FEDCBA987654321)
        alpha_r = scalar_type(0.75)
        alpha_c = complex_type(0.75, -0.125)
        beta_c = complex_type(0.5, 0.25)
        if op in ("csscal", "zdscal"):
            fn = lookup(lib, f"{op}_")
            if fn is None:
                return check_result("missing", raw=f"missing {op}_")
            fn(ctypes.byref(ni), ctypes.byref(alpha_r), x, ctypes.byref(inc))
            err = max_complex_error(x, [alpha_r.value * v for v in x0])
        elif op in ("cscal", "zscal"):
            fn = lookup(lib, f"{op}_")
            if fn is None:
                return check_result("missing", raw=f"missing {op}_")
            fn(ctypes.byref(ni), ctypes.byref(alpha_c), x, ctypes.byref(inc))
            err = max_complex_error(x, [complex(alpha_c.re, alpha_c.im) * v for v in x0])
        elif op in ("caxpy", "zaxpy"):
            fn = lookup(lib, f"{op}_")
            if fn is None:
                return check_result("missing", raw=f"missing {op}_")
            fn(ctypes.byref(ni), ctypes.byref(alpha_c), x, ctypes.byref(inc), y, ctypes.byref(inc))
            err = max_complex_error(y, [complex(alpha_c.re, alpha_c.im) * x0[i] + y0[i] for i in range(n)])
        elif op in ("caxpby", "zaxpby"):
            fn = lookup(lib, f"{op}_")
            if fn is None:
                return check_result("missing", raw=f"missing {op}_")
            fn(ctypes.byref(ni), ctypes.byref(alpha_c), x, ctypes.byref(inc), ctypes.byref(beta_c), y, ctypes.byref(inc))
            err = max_complex_error(
                y,
                [
                    complex(alpha_c.re, alpha_c.im) * x0[i]
                    + complex(beta_c.re, beta_c.im) * y0[i]
                    for i in range(n)
                ],
            )
        elif op in ("cdotu", "zdotu", "cdotc", "zdotc"):
            got = call_complex_dot(lib, op, n, x, y, complex_type)
            if got is None:
                return check_result("missing", raw=f"missing {op}_sub_/cblas_{op}_sub")
            expected = sum(
                (x0[i].conjugate() if op.endswith("dotc") else x0[i]) * y0[i]
                for i in range(n)
            )
            err = abs(got - expected)
            real_tol *= n
        elif op in ("scasum", "dzasum"):
            fn = lookup(lib, f"cblas_{op}", f"{op}_")
            if fn is None:
                return check_result("missing", raw=f"missing cblas_{op}/{op}_")
            fn.restype = scalar_type
            if getattr(fn, "__name__", "").startswith("cblas_"):
                got = float(fn(ni, x, inc))
            else:
                got = float(fn(ctypes.byref(ni), x, ctypes.byref(inc)))
            err = abs(got - sum(abs(v.real) + abs(v.imag) for v in x0))
            real_tol *= n
        elif op in ("scnrm2", "dznrm2"):
            fn = lookup(lib, f"cblas_{op}", f"{op}_")
            if fn is None:
                return check_result("missing", raw=f"missing cblas_{op}/{op}_")
            fn.restype = scalar_type
            if getattr(fn, "__name__", "").startswith("cblas_"):
                got = float(fn(ni, x, inc))
            else:
                got = float(fn(ctypes.byref(ni), x, ctypes.byref(inc)))
            err = abs(got - math.sqrt(sum(v.real * v.real + v.imag * v.imag for v in x0)))
            real_tol *= n
        else:
            return check_result("error", raw=f"unhandled op {op}")
        return check_result("sampled-ok" if err <= real_tol else "correctness_failed", err)
    except Exception as exc:
        return check_result("error", raw=str(exc))


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


def unchecked_row(args, library_name, library_path, group, op, metric, check):
    status = "missing" if check["check_status"] == "missing" else "correctness_failed"
    if check["check_status"] == "error":
        status = "error"
    return {
        "group": group,
        "op": op,
        "library": library_name,
        "library_path": library_path,
        "n": args.n,
        "seconds": args.seconds,
        "repeat": "",
        "metric": metric,
        "status": status,
        "returncode": "",
        "rate_gops": None,
        "bandwidth_gbps": None,
        "raw_output": check["check_raw_output"],
        **check,
    }


def run_level1_op(args, library_name, library_path, group, op):
    check = check_level1_op(library_path, op)
    if check["check_status"] != "sampled-ok":
        return unchecked_row(args, library_name, library_path, group, op, "rate_gops", check)
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
                **check,
            }
        )
    return choose_best(rows)


def run_copy_op(args, library_name, library_path, group, op, kind):
    check = check_copy_op(library_path, kind)
    if check["check_status"] != "sampled-ok":
        row = unchecked_row(args, library_name, library_path, group, op, "bandwidth_gbps", check)
        row["seconds"] = args.copy_seconds
        return row
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
                **check,
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
        if args.skip_missing and not library_available(library_path):
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
        "check_status",
        "check_max_abs_error",
        "check_raw_output",
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
        "correctness_check": "sampled per library/op before timing",
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
