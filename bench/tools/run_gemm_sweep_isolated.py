#!/usr/bin/env python3
# Copyright (C) 2026 Zynum contributors
# SPDX-License-Identifier: LGPL-3.0-or-later

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path


DEFAULT_ACCELERATE = "/System/Library/Frameworks/Accelerate.framework/Accelerate"
DEFAULT_OPENBLAS = "/opt/homebrew/opt/openblas/lib/libopenblas.dylib"
DEFAULT_SHAPES = [
    "sq64:64:64:64",
    "sq96:96:96:96",
    "sq128:128:128:128",
    "sq192:192:192:192",
    "sq256:256:256:256",
    "sq384:384:384:384",
    "sq512:512:512:512",
    "sq768:768:768:768",
    "sq1024:1024:1024:1024",
    "m1024_n64_k1024:1024:64:1024",
    "m2048_n64_k512:2048:64:512",
    "m4096_n32_k256:4096:32:256",
    "m512_n64_k2048:512:64:2048",
    "m64_n1024_k1024:64:1024:1024",
    "m64_n2048_k512:64:2048:512",
    "m32_n4096_k256:32:4096:256",
    "m64_n512_k2048:64:512:2048",
    "m1024_n1024_k64:1024:1024:64",
    "m1024_n1024_k128:1024:1024:128",
    "m1024_n1024_k256:1024:1024:256",
    "m256_n256_k2048:256:256:2048",
    "m128_n128_k4096:128:128:4096",
    "m1536_n256_k256:1536:256:256",
    "m256_n1536_k256:256:1536:256",
    "m512_n256_k768:512:256:768",
    "m256_n512_k768:256:512:768",
    "m768_n512_k256:768:512:256",
    "m512_n768_k256:512:768:256",
]


def parse_args():
    p = argparse.ArgumentParser(
        description="Run gemm-sweep with one BLAS library per fresh OS process and merge the CSV output."
    )
    p.add_argument("--gemm-sweep", default="zig-out/bin/gemm-sweep")
    p.add_argument(
        "--zynum-blas",
        "--zynum",
        "--zig",
        dest="zynum_blas",
        default="zig-out/lib/libzynum_blas.dylib",
    )
    p.add_argument("--accelerate", default=DEFAULT_ACCELERATE)
    p.add_argument("--openblas", default=DEFAULT_OPENBLAS)
    p.add_argument("--mkl")
    p.add_argument("--reps", type=int, default=30)
    p.add_argument(
        "--process-repeats",
        type=int,
        default=1,
        help="Run each fresh-process benchmark this many times and keep the best GF/s row for each kind/shape.",
    )
    p.add_argument("--csv", required=True)
    p.add_argument("--kind", action="append", choices=["sgemm", "dgemm", "cgemm", "zgemm"])
    p.add_argument(
        "--isolate-kind",
        action="store_true",
        help="When no --kind filter is supplied, run each GEMM kind in a separate fresh process per library.",
    )
    p.add_argument(
        "--isolate-shape",
        action="store_true",
        help="Run each shape in a separate fresh process per library/kind group. Uses the default sweep shapes when no --shape is supplied.",
    )
    p.add_argument("--shape", action="append", default=[])
    p.add_argument("--skip-missing", action="store_true")
    args = p.parse_args()
    if args.process_repeats < 1:
        p.error("--process-repeats must be at least 1")
    return args


def parse_shape_spec(spec):
    parts = spec.split(":")
    if len(parts) == 4:
        label = parts[0]
        dims = parts[1:]
    elif len(parts) == 3:
        label = spec
        dims = parts
    else:
        raise ValueError(f"bad shape spec: {spec}")
    m, n, k = (str(int(value)) for value in dims)
    return label, m, n, k


def shape_index_map(shape_specs):
    result = {}
    for index, spec in enumerate(shape_specs):
        result[parse_shape_spec(spec)] = str(index)
    return result


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


def existing_libs(args):
    libs = [("zynum-blas", args.zynum_blas)]
    candidates = [("Accelerate", args.accelerate), ("OpenBLAS", args.openblas)]
    if args.mkl:
        candidates.append(("MKL", args.mkl))
    for name, path in candidates:
        if not path:
            continue
        if args.skip_missing and name != "Accelerate" and not Path(path).exists():
            continue
        libs.append((name, path))
    return libs


def best_rows_csv(inputs, output):
    best = {}
    fieldnames = ["kind", "shape_index", "label", "m", "n", "k", "library", "gflops", "best_ns", "reps"]
    for csv_path in inputs:
        with open(csv_path, newline="") as inp:
            for row in csv.DictReader(inp):
                key = (row["kind"], row["label"], row["m"], row["n"], row["k"])
                try:
                    gflops = float(row["gflops"])
                except ValueError:
                    gflops = 0.0
                old = best.get(key)
                if old is None or gflops > float(old["gflops"]):
                    best[key] = row
    with open(output, "w", newline="") as out:
        writer = csv.DictWriter(out, fieldnames=fieldnames)
        writer.writeheader()
        for row in best.values():
            writer.writerow(row)


def run_one_process(args, name, path, out, kind=None, shapes=None):
    cmd = [
        args.gemm_sweep,
        "--zynum-blas",
        path,
        "--reps",
        str(args.reps),
        "--csv",
        str(out),
    ]
    kinds = [kind] if kind else (args.kind or [])
    for selected_kind in kinds:
        cmd += ["--kind", selected_kind]
    for shape in shapes if shapes is not None else args.shape:
        cmd += ["--shape", shape]

    env = os.environ.copy()
    env.setdefault("ZYNUM_BLAS_GEMM_POOL", "0")
    env.setdefault("OPENBLAS_DYNAMIC", "0")

    print(f"[isolated {name}] {' '.join(cmd)}", file=sys.stderr, flush=True)
    subprocess.run(cmd, check=True, env=env)


def run_one(args, name, path, tmp_dir, kind=None, shapes=None):
    suffix = f"_{kind}" if kind else ""
    if shapes and len(shapes) == 1:
        suffix += "_" + shapes[0].split(":", 1)[0]
    out = tmp_dir / f"{name}{suffix}.csv"

    if args.process_repeats == 1:
        run_one_process(args, name, path, out, kind, shapes)
        return out

    repeat_outputs = []
    for repeat in range(args.process_repeats):
        repeat_out = tmp_dir / f"{name}{suffix}_repeat{repeat + 1}.csv"
        run_one_process(args, name, path, repeat_out, kind, shapes)
        repeat_outputs.append(repeat_out)
    best_rows_csv(repeat_outputs, out)
    return out


def merge(rows_by_lib, output_path, shape_indexes):
    fieldnames = ["kind", "shape_index", "label", "m", "n", "k", "library", "gflops", "best_ns", "reps"]
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for name, csv_path in rows_by_lib:
            with open(csv_path, newline="") as inp:
                for row in csv.DictReader(inp):
                    shape_key = (row["label"], row["m"], row["n"], row["k"])
                    row["shape_index"] = shape_indexes.get(shape_key, row["shape_index"])
                    row["library"] = name
                    writer.writerow(row)


def write_metadata(args, libs, shape_specs, output_path):
    output = Path(output_path)
    env_names = [
        "ZYNUM_BLAS_AMX",
        "ZYNUM_BLAS_GEMM_POOL",
        "ZYNUM_BLAS_GEMM_IO",
        "ZYNUM_BLAS_NUM_THREADS",
        "OPENBLAS_DYNAMIC",
        "OPENBLAS_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "MKL_DYNAMIC",
        "MKL_NUM_THREADS",
        "ZIG_GLOBAL_CACHE_DIR",
    ]
    metadata = {
        "generated_at_unix": time.time(),
        "argv": sys.argv,
        "cwd": os.getcwd(),
        "zig_version": zig_version(),
        "reps": args.reps,
        "process_repeats": args.process_repeats,
        "isolate_kind": args.isolate_kind,
        "isolate_shape": args.isolate_shape,
        "kinds": args.kind,
        "shapes": shape_specs,
        "environment": {name: os.environ.get(name) for name in env_names if os.environ.get(name) is not None},
        "binaries": {
            "gemm_sweep": {
                "path": args.gemm_sweep,
                "sha256": sha256_file(args.gemm_sweep),
            },
            "libraries": [
                {
                    "name": name,
                    "path": path,
                    "sha256": sha256_file(path),
                }
                for name, path in libs
            ],
        },
    }
    with output.with_suffix(output.suffix + ".meta.json").open("w") as f:
        json.dump(metadata, f, indent=2, sort_keys=True)
        f.write("\n")


def main():
    args = parse_args()
    libs = existing_libs(args)
    isolated_kinds = ["sgemm", "dgemm", "cgemm", "zgemm"] if args.isolate_kind and not args.kind else [None]
    shape_specs = args.shape or DEFAULT_SHAPES
    if args.isolate_shape:
        shape_groups = [[shape] for shape in shape_specs]
    else:
        shape_groups = [None]
    with tempfile.TemporaryDirectory(prefix="zynum-blas-gemm-isolated-") as td:
        tmp_dir = Path(td)
        rows_by_lib = []
        for name, path in libs:
            for kind in isolated_kinds:
                for shapes in shape_groups:
                    rows_by_lib.append((name, run_one(args, name, path, tmp_dir, kind, shapes)))
        merge(rows_by_lib, args.csv, shape_index_map(shape_specs))
    write_metadata(args, libs, shape_specs, args.csv)


if __name__ == "__main__":
    main()
