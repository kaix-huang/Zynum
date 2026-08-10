#!/usr/bin/env python3
# Copyright (C) 2026 Zynum contributors
# SPDX-License-Identifier: LGPL-3.0-or-later

import argparse
import csv
import io
import os
import platform
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import benchmark_metadata
from benchmark_artifacts import (
    ArtifactRequest,
    ArtifactSnapshotError,
    ArtifactSnapshotSet,
)
from report_comparison import (
    best_higher_row,
    nearest_rank_percentile,
    parse_positive_finite,
    positive_finite_median,
)
from report_publication import ReportOutput, publish_outputs
from report_schedule import (
    SCHEDULE_CHOICES,
    library_repeat_schedule,
    validate_unique_library_labels,
)

DEFAULT_ACCELERATE = "/System/Library/Frameworks/Accelerate.framework/Accelerate"
DEFAULT_OPENBLAS = "/opt/homebrew/opt/openblas/lib/libopenblas.dylib"
DEFAULT_SHAPES = [
    "m1_n1_k1:1:1:1",
    "m8_n8_k8:8:8:8",
    "m31_n31_k31:31:31:31",
    "m33_n33_k33:33:33:33",
    "sq64:64:64:64",
    "sq96:96:96:96",
    "sq128:128:128:128",
    "sq192:192:192:192",
    "sq256:256:256:256",
    "sq384:384:384:384",
    "sq512:512:512:512",
    "sq768:768:768:768",
    "sq1024:1024:1024:1024",
    "m63_n65_k17:63:65:17",
    "m65_n63_k33:65:63:33",
    "m127_n129_k31:127:129:31",
    "m129_n127_k33:129:127:33",
    "m1_n4096_k256:1:4096:256",
    "m4096_n1_k256:4096:1:256",
    "m1024_n64_k1024:1024:64:1024",
    "m2048_n64_k512:2048:64:512",
    "m4096_n32_k256:4096:32:256",
    "m2048_n17_k257:2048:17:257",
    "m512_n64_k2048:512:64:2048",
    "m64_n1024_k1024:64:1024:1024",
    "m64_n2048_k512:64:2048:512",
    "m32_n4096_k256:32:4096:256",
    "m17_n2048_k257:17:2048:257",
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
    "m384_n640_k96:384:640:96",
    "m640_n384_k96:640:384:96",
]

CSV_FIELDNAMES = [
    "kind",
    "transa",
    "transb",
    "shape_index",
    "label",
    "m",
    "n",
    "k",
    "library",
    "gflops",
    "best_ns",
    "median_ns",
    "p95_ns",
    "max_ns",
    "reps",
    "process_repeats",
    "check",
]


def default_zynum_blas():
    if sys.platform == "darwin":
        return "zig-out/lib/libzynum_blas.dylib"
    if sys.platform == "win32":
        return "zig-out/bin/zynum_blas.dll"
    return "zig-out/lib/libzynum_blas.so"


def default_executable(path):
    return f"{path}.exe" if sys.platform == "win32" else path


def parse_transpose_spec(value):
    pair = value.upper()
    if len(pair) != 2 or any(trans not in "NTC" for trans in pair):
        raise argparse.ArgumentTypeError(
            f"transpose pair must contain two N/T/C characters, got {value!r}"
        )
    return pair


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Run gemm-sweep with one BLAS library per fresh OS process and merge the CSV output."
    )
    p.add_argument("--gemm-sweep", default=default_executable("zig-out/bin/gemm-sweep"))
    p.add_argument(
        "--zynum-blas",
        "--zynum",
        "--zig",
        dest="zynum_blas",
        default=default_zynum_blas(),
    )
    p.add_argument("--accelerate", default=DEFAULT_ACCELERATE)
    p.add_argument("--openblas", default=DEFAULT_OPENBLAS)
    p.add_argument("--mkl")
    p.add_argument("--aocl-blis")
    p.add_argument("--atlas")
    p.add_argument(
        "--extra-blas",
        action="append",
        default=[],
        metavar="LABEL=PATH",
        help="Additional drop-in BLAS comparator. May be passed more than once.",
    )
    p.add_argument("--reps", type=int, default=30)
    p.add_argument(
        "--process-repeats",
        type=int,
        default=1,
        help="Run each fresh-process benchmark this many times and merge per-process timing distributions for each kind/shape.",
    )
    p.add_argument(
        "--process-schedule",
        choices=SCHEDULE_CHOICES,
        default="library-major",
        help=(
            "Fresh-process ordering; interleaved uses cyclic Latin rotations and "
            "requires repeats to be a multiple of the selected library count."
        ),
    )
    p.add_argument("--csv", required=True)
    p.add_argument(
        "--check",
        action="store_true",
        help="Ask gemm-sweep to run correctness checks before timing each row.",
    )
    p.add_argument(
        "--kind", action="append", choices=["sgemm", "dgemm", "cgemm", "zgemm"]
    )
    p.add_argument(
        "--trans",
        action="append",
        type=parse_transpose_spec,
        help="GEMM transpose pair. Repeat for multiple combinations; defaults to NN.",
    )
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
    benchmark_metadata.add_identity_arguments(p)
    args = p.parse_args(argv)
    if args.process_repeats < 1:
        p.error("--process-repeats must be at least 1")
    return args


def library_path_exists(path):
    if Path(path).exists():
        return True
    return sys.platform == "darwin" and path == DEFAULT_ACCELERATE


def library_disabled(path):
    return not path or path == "none"


def library_artifact_request(name, path):
    if Path(path).is_file():
        return ArtifactRequest.library(name, path)
    if name == "Accelerate" and sys.platform == "darwin" and path == DEFAULT_ACCELERATE:
        return ArtifactRequest.platform_image(name, path)
    return ArtifactRequest.library(name, path)


def append_extra_blas(candidates, items):
    for item in items:
        if "=" not in item:
            raise ValueError(f"--extra-blas must be LABEL=PATH, got {item!r}")
        label, path = item.split("=", 1)
        label = label.strip()
        path = path.strip()
        if not label or not path:
            raise ValueError(f"--extra-blas must be LABEL=PATH, got {item!r}")
        candidates.append((label, path))


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


def transpose_fields(row):
    return (
        (row.get("transa") or "N").upper(),
        (row.get("transb") or "N").upper(),
    )


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


def child_environment_snapshot(names):
    env = os.environ.copy()
    env.setdefault("OPENBLAS_DYNAMIC", "0")
    return {name: env.get(name, "unset") for name in names}


def zynum_maximum_threads():
    detected = max(1, os.cpu_count() or 1)
    value = os.environ.get("ZYNUM_MAXIMUM_THREADS")
    if not value:
        return detected
    try:
        parsed = int(value, 10)
    except ValueError:
        return detected
    return parsed if parsed > 0 else detected


def existing_libs(args):
    libs = [("Zynum", args.zynum_blas)]
    candidates = [("Accelerate", args.accelerate), ("OpenBLAS", args.openblas)]
    if args.mkl:
        candidates.append(("MKL", args.mkl))
    if args.aocl_blis:
        candidates.append(("AOCL-BLIS", args.aocl_blis))
    if args.atlas:
        candidates.append(("ATLAS", args.atlas))
    append_extra_blas(candidates, args.extra_blas)
    for name, path in candidates:
        if library_disabled(path):
            continue
        if args.skip_missing and not library_path_exists(path):
            continue
        libs.append((name, path))
    return libs


def gemm_semantic_key(row):
    return (
        row["kind"],
        *transpose_fields(row),
        row["label"],
        row["m"],
        row["n"],
        row["k"],
    )


def expected_process_keys(args, shape_specs, kind=None, shapes=None):
    selected_kinds = (
        [kind]
        if kind
        else (
            args.kind
            or [
                "sgemm",
                "dgemm",
                "cgemm",
                "zgemm",
            ]
        )
    )
    selected_shapes = shapes if shapes is not None else shape_specs
    keys = []
    seen = set()
    for selected_kind in selected_kinds:
        for transpose in args.trans or ["NN"]:
            transa, transb = transpose
            if selected_kind in {"sgemm", "dgemm"} and "C" in (transa, transb):
                continue
            for shape in selected_shapes:
                label, m, n, k = parse_shape_spec(shape)
                key = (selected_kind, transa, transb, label, m, n, k)
                if key in seen:
                    raise ValueError(f"duplicate requested GEMM key {key!r}")
                seen.add(key)
                keys.append(key)
    return keys


def repeat_rows_by_key(csv_path, expected_keys=None):
    rows_by_key = {}
    key_order = []
    try:
        with open(csv_path, newline="") as inp:
            for row in csv.DictReader(inp):
                key = gemm_semantic_key(row)
                if key in rows_by_key:
                    raise ValueError(
                        f"repeat CSV {csv_path} has duplicate GEMM key {key!r}"
                    )
                rows_by_key[key] = row
                key_order.append(key)
    except KeyError as exc:
        raise ValueError(
            f"repeat CSV {csv_path} is missing canonical field {exc.args[0]!r}"
        ) from exc
    if expected_keys is not None:
        expected = set(expected_keys)
        actual = set(rows_by_key)
        if actual != expected:
            missing = sorted(expected - actual)
            extra = sorted(actual - expected)
            raise ValueError(
                f"repeat CSV {csv_path} GEMM key mismatch: "
                f"missing={missing!r} extra={extra!r}"
            )
    return rows_by_key, key_order


def best_rows_csv(inputs, output, expected_keys=None):
    repeats = []
    key_order = None
    first_repeat_keys = None
    for csv_path in inputs:
        rows_by_key, current_order = repeat_rows_by_key(csv_path, expected_keys)
        current_keys = set(rows_by_key)
        if first_repeat_keys is None:
            first_repeat_keys = current_keys
            key_order = current_order
        elif current_keys != first_repeat_keys:
            missing = sorted(first_repeat_keys - current_keys)
            extra = sorted(current_keys - first_repeat_keys)
            raise ValueError(
                f"repeat CSV {csv_path} GEMM key mismatch: "
                f"missing={missing!r} extra={extra!r}"
            )
        repeats.append(rows_by_key)

    merged_rows = []
    for key in key_order or []:
        merged_rows.append(merge_repeat_rows([repeat[key] for repeat in repeats]))
    with open(output, "w", newline="") as out:
        writer = csv.DictWriter(out, fieldnames=CSV_FIELDNAMES, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(merged_rows)


def merged_check_status(rows):
    checks = {row.get("check", "unchecked") for row in rows}
    if checks <= {"checked-ok", "sampled-ok"}:
        return "checked-ok"
    if checks == {"unchecked"}:
        return "unchecked"
    return sorted(checks)[0]


def gemm_timing(row, field):
    value = row.get(field) or row.get("best_ns")
    return parse_positive_finite(value, field)


def format_gemm_evidence(value, field):
    return format(parse_positive_finite(value, field), ".17g")


def validate_gemm_evidence(row):
    parse_positive_finite(row.get("gflops"), "gflops")
    for field in ("best_ns", "median_ns", "p95_ns", "max_ns"):
        gemm_timing(row, field)


def merge_repeat_rows(rows):
    for row in rows:
        validate_gemm_evidence(row)
    base = best_higher_row(rows, "gflops").copy()
    base["transa"], base["transb"] = transpose_fields(base)
    best_values = [gemm_timing(row, "best_ns") for row in rows]
    median_values = [gemm_timing(row, "median_ns") for row in rows]
    p95_values = [gemm_timing(row, "p95_ns") for row in rows]
    max_values = [gemm_timing(row, "max_ns") for row in rows]
    base["best_ns"] = format_gemm_evidence(min(best_values), "best_ns")
    base["median_ns"] = format_gemm_evidence(
        positive_finite_median(median_values, "median_ns"), "median_ns"
    )
    base["p95_ns"] = format_gemm_evidence(
        nearest_rank_percentile(p95_values, 95, "p95_ns"), "p95_ns"
    )
    base["max_ns"] = format_gemm_evidence(max(max_values), "max_ns")
    base["process_repeats"] = str(len(rows))
    base["check"] = merged_check_status(rows)
    validate_gemm_evidence(base)
    return base


def run_one_process(
    args,
    name,
    path,
    out,
    kind=None,
    shapes=None,
    *,
    execution_binary=None,
    execution_library=None,
    artifacts=None,
):
    binary_path = execution_binary or args.gemm_sweep
    library_path = execution_library or path
    cmd = [
        binary_path,
        "--zynum-blas",
        library_path,
        "--reps",
        str(args.reps),
        "--csv",
        str(out),
    ]
    kinds = [kind] if kind else (args.kind or [])
    for selected_kind in kinds:
        cmd += ["--kind", selected_kind]
    for transpose in args.trans or []:
        cmd += ["--trans", transpose]
    for shape in shapes if shapes is not None else args.shape:
        cmd += ["--shape", shape]
    if args.check:
        cmd.append("--check")

    env = os.environ.copy()
    env.setdefault("OPENBLAS_DYNAMIC", "0")

    if execution_binary is not None and cmd[0] != execution_binary:
        raise ValueError("GEMM child command did not use the frozen probe")
    library_arg = cmd[cmd.index("--zynum-blas") + 1]
    if execution_library is not None and library_arg != execution_library:
        raise ValueError("GEMM child command did not use the frozen library")
    public_cmd = [args.gemm_sweep, *cmd[1:]]
    public_cmd[public_cmd.index("--zynum-blas") + 1] = path
    public_cmd[public_cmd.index("--csv") + 1] = args.csv
    if artifacts is not None:
        public_cmd = artifacts.redact_private_paths(public_cmd)
    print(f"[isolated {name}] {' '.join(public_cmd)}", file=sys.stderr, flush=True)
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, env=env)
    except OSError:
        raise ValueError(f"failed to start frozen GEMM probe for {name}") from None
    stdout = result.stdout or ""
    stderr = result.stderr or ""
    private_output = str(out)
    private_output_root = str(Path(out).parent)
    stdout = stdout.replace(private_output, args.csv).replace(
        private_output_root, "<private-benchmark-output-root>"
    )
    stderr = stderr.replace(private_output, args.csv).replace(
        private_output_root, "<private-benchmark-output-root>"
    )
    if artifacts is not None:
        stdout, stderr = artifacts.redact_private_paths((stdout, stderr))
    if stdout:
        print(stdout, end="" if stdout.endswith("\n") else "\n")
    if stderr:
        print(
            stderr,
            end="" if stderr.endswith("\n") else "\n",
            file=sys.stderr,
        )
    if result.returncode != 0:
        detail = (stdout + stderr).strip()
        suffix = f": {detail}" if detail else ""
        raise ValueError(f"GEMM probe for {name} exited {result.returncode}{suffix}")


def intermediate_output_path(
    tmp_dir, library_index, case_index, repeat_index=None, *, merged=False
):
    indexes = {
        "library_index": library_index,
        "case_index": case_index,
    }
    if merged:
        if repeat_index is not None:
            raise ValueError("merged intermediate output cannot have a repeat index")
        marker = "merged"
    else:
        if repeat_index is None:
            raise ValueError("repeat intermediate output requires a repeat index")
        indexes["repeat_index"] = repeat_index
        marker = f"repeat_{repeat_index}"
    for label, value in indexes.items():
        if type(value) is not int or value < 0:
            raise ValueError(f"{label} must be a nonnegative integer")

    private_dir = Path(tmp_dir)
    output = private_dir / (f"library_{library_index}_case_{case_index}_{marker}.csv")
    if output.parent != private_dir:
        raise AssertionError("intermediate output must be a direct child of its root")
    return output


def serialize_csv(rows_by_lib, shape_indexes):
    output_rows = []
    for name, csv_path in rows_by_lib:
        with open(csv_path, newline="") as inp:
            for row in csv.DictReader(inp):
                validate_gemm_evidence(row)
                shape_key = (row["label"], row["m"], row["n"], row["k"])
                row["shape_index"] = shape_indexes.get(shape_key, row["shape_index"])
                row["library"] = name
                row["transa"], row["transb"] = transpose_fields(row)
                row.setdefault("median_ns", row.get("best_ns", ""))
                row.setdefault("p95_ns", row.get("best_ns", ""))
                row.setdefault("max_ns", row.get("best_ns", ""))
                row.setdefault("process_repeats", "1")
                row.setdefault("check", "unchecked")
                output_rows.append(row)
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=CSV_FIELDNAMES, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(output_rows)
    return buffer.getvalue().encode("utf-8")


def serialize_metadata(
    args,
    libs,
    shape_specs,
    benchmark_identity,
    *,
    binary_record=None,
    library_records=None,
):
    env_names = [
        "ZYNUM_MAXIMUM_THREADS",
        "OPENBLAS_DYNAMIC",
        "OPENBLAS_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "MKL_DYNAMIC",
        "MKL_NUM_THREADS",
        "OMP_NUM_THREADS",
        "BLIS_NUM_THREADS",
        "AOCL_DYNAMIC",
        "ZIG_GLOBAL_CACHE_DIR",
    ]
    metadata = {
        "generated_at_unix": time.time(),
        "argv": sys.argv,
        "cwd": os.getcwd(),
        "os": platform.platform(),
        "python_version": sys.version,
        "zig_version": zig_version(),
        "source": benchmark_metadata.legacy_source_snapshot(
            benchmark_identity["source"]
        ),
        "detected_cpu_count": os.cpu_count(),
        "zynum_maximum_threads": zynum_maximum_threads(),
        "reps": args.reps,
        "process_repeats": args.process_repeats,
        "schedule": args.process_schedule,
        "correctness_check": "checked" if args.check else "unchecked",
        "isolate_kind": args.isolate_kind,
        "isolate_shape": args.isolate_shape,
        "kinds": args.kind,
        "transposes": args.trans or ["NN"],
        "shapes": shape_specs,
        "environment": child_environment_snapshot(env_names),
        "binaries": {
            "gemm_sweep": binary_record or {"path": args.gemm_sweep, "sha256": None},
            "libraries": library_records
            or [{"name": name, "path": path, "sha256": None} for name, path in libs],
        },
        "benchmark_identity": benchmark_identity,
    }
    return benchmark_metadata.serialize_public_metadata(
        metadata,
        controller="run_gemm_sweep_isolated.py",
        parameter_keys=(
            "reps",
            "process_repeats",
            "schedule",
            "correctness_check",
            "isolate_kind",
            "isolate_shape",
            "kinds",
            "transposes",
            "shapes",
        ),
    )


def run_controller(args):
    libs = existing_libs(args)
    validate_unique_library_labels(libs)
    isolated_kinds = (
        ["sgemm", "dgemm", "cgemm", "zgemm"]
        if args.isolate_kind and not args.kind
        else [None]
    )
    shape_specs = args.shape or DEFAULT_SHAPES
    if args.isolate_shape:
        shape_groups = [[shape] for shape in shape_specs]
    else:
        shape_groups = [None]
    cases = [
        (
            kind,
            shapes,
            expected_process_keys(args, shape_specs, kind, shapes),
        )
        for kind in isolated_kinds
        for shapes in shape_groups
    ]
    schedule = library_repeat_schedule(
        len(libs),
        args.process_repeats,
        args.process_schedule,
        case_count=len(cases),
    )
    requests = [ArtifactRequest.binary("gemm_sweep", args.gemm_sweep)]
    requests.extend(library_artifact_request(name, path) for name, path in libs)
    artifacts = ArtifactSnapshotSet.capture(requests)
    private_output_root = None
    try:
        frozen_binaries = artifacts.for_role("binary")
        frozen_libraries = artifacts.for_role("library")
        benchmark_identity = benchmark_metadata.collect_benchmark_identity_from_frozen(
            args,
            libraries=frozen_libraries,
            binaries=frozen_binaries,
        )
        with tempfile.TemporaryDirectory(prefix="zynum-blas-gemm-isolated-") as td:
            tmp_dir = Path(td)
            private_output_root = str(tmp_dir)
            repeat_outputs = [[[] for _ in cases] for _ in libs]
            for library_index, case_index, repeat_index in schedule:
                name, public_library = libs[library_index]
                kind, shapes, _ = cases[case_index]
                out = intermediate_output_path(
                    tmp_dir,
                    library_index,
                    case_index,
                    repeat_index,
                )
                run_one_process(
                    args,
                    name,
                    public_library,
                    out,
                    kind,
                    shapes,
                    execution_binary=frozen_binaries[0].execution_path,
                    execution_library=frozen_libraries[library_index].execution_path,
                    artifacts=artifacts,
                )
                repeat_outputs[library_index][case_index].append(out)

            rows_by_lib = []
            for library_index, (name, _) in enumerate(libs):
                for case_index, (_kind, _shapes, expected_keys) in enumerate(cases):
                    outputs = repeat_outputs[library_index][case_index]
                    if len(outputs) == 1:
                        repeat_rows_by_key(outputs[0], expected_keys)
                        out = outputs[0]
                    else:
                        out = intermediate_output_path(
                            tmp_dir, library_index, case_index, merged=True
                        )
                        best_rows_csv(outputs, out, expected_keys)
                    rows_by_lib.append((name, out))
            csv_bytes = serialize_csv(rows_by_lib, shape_index_map(shape_specs))
        binary_records = artifacts.legacy_records("binary")
        library_records = artifacts.legacy_records("library")
        metadata_bytes = serialize_metadata(
            args,
            libs,
            shape_specs,
            benchmark_identity,
            binary_record={
                "path": binary_records[0]["path"],
                "sha256": binary_records[0]["sha256"],
            },
            library_records=library_records,
        )
        csv_bytes, metadata_bytes = artifacts.redact_private_paths(
            (csv_bytes, metadata_bytes)
        )
        artifacts.finalize()
    except ValueError as exc:
        message = artifacts.redact_private_paths(str(exc))
        if private_output_root is not None:
            message = message.replace(
                private_output_root, "<private-benchmark-output-root>"
            )
        raise ValueError(message) from None
    finally:
        artifacts.close()
    output = Path(args.csv)
    publish_outputs(
        [
            ReportOutput(output, csv_bytes),
            ReportOutput(
                output.with_suffix(output.suffix + ".meta.json"), metadata_bytes
            ),
        ]
    )


def main():
    args = parse_args()
    try:
        run_controller(args)
    except (ArtifactSnapshotError, ValueError) as exc:
        print(exc, file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
