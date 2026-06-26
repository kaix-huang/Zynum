#!/usr/bin/env python3
# Copyright (C) 2026 Zynum contributors
# SPDX-License-Identifier: LGPL-3.0-or-later

import argparse
import csv
import ctypes
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

DEFAULT_ACCELERATE = "/System/Library/Frameworks/Accelerate.framework/Accelerate"
DEFAULT_OPENBLAS = "/opt/homebrew/opt/openblas/lib/libopenblas.dylib"

LIBS = [
    ("Zynum", "zynum"),
    ("Accelerate", "accelerate"),
    ("OpenBLAS", "openblas"),
]

DEFAULT_N = [128, 256, 512]


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
        description="Run representative Level 2 fresh-process probes and write a report CSV."
    )
    parser.add_argument("--zynum", default=default_zynum_blas())
    parser.add_argument("--accelerate", default=DEFAULT_ACCELERATE)
    parser.add_argument("--openblas", default=DEFAULT_OPENBLAS)
    parser.add_argument("--n", action="append", type=int, default=[])
    parser.add_argument("--reps-small", type=int, default=260)
    parser.add_argument("--reps-large", type=int, default=130)
    parser.add_argument("--csv", required=True)
    parser.add_argument("--skip-missing", action="store_true")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--library-name", help=argparse.SUPPRESS)
    parser.add_argument("--library-path", help=argparse.SUPPRESS)
    parser.add_argument("--worker-n", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--worker-reps", type=int, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.reps_small < 1 or args.reps_large < 1:
        parser.error("repetition counts must be at least 1")
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
            ["zig", "version"], check=True, capture_output=True, text=True
        )
    except Exception:
        return None
    return result.stdout.strip()


def git_revision():
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
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


def next_fill(seed):
    seed[0] = (seed[0] * 6364136223846793005 + 1442695040888963407) & (
        (1 << 64) - 1
    )
    return ((seed[0] >> 32) % 1000) / 1000.0 - 0.5


def real_array(ctype, count, seed_value):
    seed = [seed_value]
    array_type = ctype * count
    out = array_type()
    for index in range(count):
        out[index] = ctype(next_fill(seed))
    return out


def complex_array(complex_type, count, seed_value):
    seed = [seed_value]
    array_type = complex_type * count
    out = array_type()
    for index in range(count):
        out[index] = complex_type(next_fill(seed), next_fill(seed))
    return out


def ptr(array):
    return ctypes.cast(array, ctypes.c_void_p)


def copy_array(dst, src):
    ctypes.memmove(ctypes.addressof(dst), ctypes.addressof(src), ctypes.sizeof(src))


def best_time(call, setup, reps):
    best = None
    for _ in range(reps):
        setup()
        start = time.perf_counter_ns()
        call()
        elapsed = time.perf_counter_ns() - start
        if elapsed > 0 and (best is None or elapsed < best):
            best = elapsed
    return best if best is not None else 0


def emit(rows, case, kind, library_name, n, elapsed_ns, work):
    rate = work / (elapsed_ns / 1e9) / 1e9 if elapsed_ns > 0 else 0.0
    rows.append(
        {
            "level": "level2",
            "case": case,
            "kind": kind,
            "library": library_name,
            "n": n,
            "time_ns": elapsed_ns,
            "rate_gops": f"{rate:.6f}",
            "metric": "gops",
            "status": "ok",
        }
    )


def run_worker(args):
    if not args.library_name or not args.library_path or not args.worker_n:
        raise SystemExit("--worker requires library name/path and n")
    library_name = args.library_name
    n = args.worker_n
    reps = args.worker_reps or 1
    lib = ctypes.CDLL(args.library_path)
    blas_int = ctypes.c_int
    one = blas_int(1)
    ni = blas_int(n)
    trans_n = ctypes.create_string_buffer(b"N")
    trans_t = ctypes.create_string_buffer(b"T")
    uplo_u = ctypes.create_string_buffer(b"U")
    rows = []

    for kind, ctype, prefix in [
        ("f32", ctypes.c_float, "s"),
        ("f64", ctypes.c_double, "d"),
    ]:
        matrix = real_array(ctype, n * n, 0x3141592653589793)
        x = real_array(ctype, n, 0x2718281828459045)
        y0 = real_array(ctype, n, 0x1618033988749895)
        y = real_array(ctype, n, 0x1123581321345589)
        alpha = ctype(0.7)
        beta = ctype(0.3)
        gemv = getattr(lib, prefix + "gemv_")

        def setup_y():
            copy_array(y, y0)

        def run_gemv_n():
            gemv(
                trans_n,
                ctypes.byref(ni),
                ctypes.byref(ni),
                ctypes.byref(alpha),
                ptr(matrix),
                ctypes.byref(ni),
                ptr(x),
                ctypes.byref(one),
                ctypes.byref(beta),
                ptr(y),
                ctypes.byref(one),
            )

        elapsed = best_time(run_gemv_n, setup_y, reps)
        emit(rows, prefix + "gemv_n", kind, library_name, n, elapsed, 2 * n * n)

        def run_gemv_t():
            gemv(
                trans_t,
                ctypes.byref(ni),
                ctypes.byref(ni),
                ctypes.byref(alpha),
                ptr(matrix),
                ctypes.byref(ni),
                ptr(x),
                ctypes.byref(one),
                ctypes.byref(beta),
                ptr(y),
                ctypes.byref(one),
            )

        elapsed = best_time(run_gemv_t, setup_y, reps)
        emit(rows, prefix + "gemv_t", kind, library_name, n, elapsed, 2 * n * n)

        symv = getattr(lib, prefix + "symv_")

        def run_symv():
            symv(
                uplo_u,
                ctypes.byref(ni),
                ctypes.byref(alpha),
                ptr(matrix),
                ctypes.byref(ni),
                ptr(x),
                ctypes.byref(one),
                ctypes.byref(beta),
                ptr(y),
                ctypes.byref(one),
            )

        elapsed = best_time(run_symv, setup_y, reps)
        emit(rows, prefix + "symv", kind, library_name, n, elapsed, 2 * n * n)

        matrix0 = real_array(ctype, n * n, 0x123456789abcdef0)
        target = real_array(ctype, n * n, 0xfeedfacecafebeef)
        gy = real_array(ctype, n, 0x0102030405060708)
        ger = getattr(lib, prefix + "ger_")

        def setup_a():
            copy_array(target, matrix0)

        def run_ger():
            ger(
                ctypes.byref(ni),
                ctypes.byref(ni),
                ctypes.byref(alpha),
                ptr(x),
                ctypes.byref(one),
                ptr(gy),
                ctypes.byref(one),
                ptr(target),
                ctypes.byref(ni),
            )

        elapsed = best_time(run_ger, setup_a, reps)
        emit(rows, prefix + "ger", kind, library_name, n, elapsed, 2 * n * n)

    for kind, complex_type, prefix in [
        ("c32", ComplexF32, "c"),
        ("c64", ComplexF64, "z"),
    ]:
        matrix = complex_array(complex_type, n * n, 0x3141592653589793)
        x = complex_array(complex_type, n, 0x2718281828459045)
        y0 = complex_array(complex_type, n, 0x1618033988749895)
        y = complex_array(complex_type, n, 0x1123581321345589)
        alpha = complex_type(0.7, 0.125)
        beta = complex_type(0.3, -0.0625)
        gemv = getattr(lib, prefix + "gemv_")

        def setup_y():
            copy_array(y, y0)

        def run_gemv_n():
            gemv(
                trans_n,
                ctypes.byref(ni),
                ctypes.byref(ni),
                ctypes.byref(alpha),
                ptr(matrix),
                ctypes.byref(ni),
                ptr(x),
                ctypes.byref(one),
                ctypes.byref(beta),
                ptr(y),
                ctypes.byref(one),
            )

        elapsed = best_time(run_gemv_n, setup_y, reps)
        emit(rows, prefix + "gemv_n", kind, library_name, n, elapsed, 8 * n * n)

        def run_gemv_t():
            gemv(
                trans_t,
                ctypes.byref(ni),
                ctypes.byref(ni),
                ctypes.byref(alpha),
                ptr(matrix),
                ctypes.byref(ni),
                ptr(x),
                ctypes.byref(one),
                ctypes.byref(beta),
                ptr(y),
                ctypes.byref(one),
            )

        elapsed = best_time(run_gemv_t, setup_y, reps)
        emit(rows, prefix + "gemv_t", kind, library_name, n, elapsed, 8 * n * n)

        hemv = getattr(lib, prefix + "hemv_")

        def run_hemv():
            hemv(
                uplo_u,
                ctypes.byref(ni),
                ctypes.byref(alpha),
                ptr(matrix),
                ctypes.byref(ni),
                ptr(x),
                ctypes.byref(one),
                ctypes.byref(beta),
                ptr(y),
                ctypes.byref(one),
            )

        elapsed = best_time(run_hemv, setup_y, reps)
        emit(rows, prefix + "hemv", kind, library_name, n, elapsed, 8 * n * n)

        matrix0 = complex_array(complex_type, n * n, 0x123456789abcdef0)
        target = complex_array(complex_type, n * n, 0xfeedfacecafebeef)
        gy = complex_array(complex_type, n, 0x0102030405060708)

        def setup_a():
            copy_array(target, matrix0)

        geru = getattr(lib, prefix + "geru_")

        def run_geru():
            geru(
                ctypes.byref(ni),
                ctypes.byref(ni),
                ctypes.byref(alpha),
                ptr(x),
                ctypes.byref(one),
                ptr(gy),
                ctypes.byref(one),
                ptr(target),
                ctypes.byref(ni),
            )

        elapsed = best_time(run_geru, setup_a, reps)
        emit(rows, prefix + "geru", kind, library_name, n, elapsed, 8 * n * n)

        gerc = getattr(lib, prefix + "gerc_")

        def run_gerc():
            gerc(
                ctypes.byref(ni),
                ctypes.byref(ni),
                ctypes.byref(alpha),
                ptr(x),
                ctypes.byref(one),
                ptr(gy),
                ctypes.byref(one),
                ptr(target),
                ctypes.byref(ni),
            )

        elapsed = best_time(run_gerc, setup_a, reps)
        emit(rows, prefix + "gerc", kind, library_name, n, elapsed, 8 * n * n)

    writer = csv.DictWriter(
        sys.stdout,
        fieldnames=[
            "level",
            "case",
            "kind",
            "library",
            "n",
            "time_ns",
            "rate_gops",
            "metric",
            "status",
        ],
    )
    writer.writeheader()
    writer.writerows(rows)


def run_one_process(script, library_name, library_path, n, reps):
    cmd = [
        sys.executable,
        str(script),
        "--worker",
        "--csv",
        os.devnull,
        "--library-name",
        library_name,
        "--library-path",
        library_path,
        "--worker-n",
        str(n),
        "--worker-reps",
        str(reps),
    ]
    return subprocess.run(cmd, capture_output=True, text=True, check=False)


def write_metadata(args, output_path, selected_libraries, sizes):
    output = Path(output_path)
    metadata = {
        "generated_at_unix": time.time(),
        "argv": sys.argv,
        "cwd": os.getcwd(),
        "zig_version": zig_version(),
        "git_revision": git_revision(),
        "detected_cpu_count": os.cpu_count(),
        "zynum_maximum_threads": zynum_maximum_threads_detected(),
        "sizes": sizes,
        "reps_small": args.reps_small,
        "reps_large": args.reps_large,
        "environment": environment_snapshot(),
        "libraries": [
            {
                "name": name,
                "path": path,
                "sha256": sha256_file(path),
            }
            for name, path in selected_libraries
        ],
    }
    with output.with_suffix(output.suffix + ".meta.json").open("w") as f:
        json.dump(metadata, f, indent=2, sort_keys=True)
        f.write("\n")


def run_controller(args):
    sizes = args.n or DEFAULT_N
    selected_libraries = libraries(args)
    rows = []
    script = Path(__file__)
    for n in sizes:
        reps = args.reps_small if n <= 256 else args.reps_large
        for library_name, library_path in selected_libraries:
            if args.skip_missing and not Path(library_path).exists():
                continue
            print(
                f"[level2 {library_name}] n={n} reps={reps} path={library_path}",
                file=sys.stderr,
                flush=True,
            )
            result = run_one_process(script, library_name, library_path, n, reps)
            if result.returncode != 0:
                sys.stderr.write(result.stdout)
                sys.stderr.write(result.stderr)
                raise SystemExit(result.returncode)
            for row in csv.DictReader(result.stdout.splitlines()):
                rows.append(row)

    output = Path(args.csv)
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "level",
        "case",
        "kind",
        "library",
        "n",
        "time_ns",
        "rate_gops",
        "metric",
        "status",
    ]
    with output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    write_metadata(args, output, selected_libraries, sizes)


def main():
    args = parse_args()
    if args.worker:
        run_worker(args)
    else:
        run_controller(args)


if __name__ == "__main__":
    main()
