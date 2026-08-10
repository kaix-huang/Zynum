#!/usr/bin/env python3
# Copyright (C) 2026 Zynum contributors
# SPDX-License-Identifier: LGPL-3.0-or-later

import argparse
import csv
import ctypes
import io
import json
import math
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import benchmark_metadata
from benchmark_artifacts import (
    ArtifactRequest,
    ArtifactSnapshotError,
    ArtifactSnapshotSet,
)
from report_comparison import (
    positive_finite_median,
    validate_optional_metric_evidence,
    validate_performance_fields,
)
from report_publication import ReportOutput, publish_outputs
from report_schedule import (
    SCHEDULE_CHOICES,
    library_repeat_schedule,
    validate_schedule,
    validate_unique_library_labels,
)

DEFAULT_ACCELERATE = "/System/Library/Frameworks/Accelerate.framework/Accelerate"
DEFAULT_OPENBLAS = "/opt/homebrew/opt/openblas/lib/libopenblas.dylib"

LEVEL1_OPS = [
    ("swap", "sswap"),
    ("swap", "dswap"),
    ("swap", "cswap"),
    ("swap", "zswap"),
    ("index", "isamax"),
    ("index", "idamax"),
    ("index", "icamax"),
    ("index", "izamax"),
    ("real_f32", "sscal"),
    ("real_f32", "saxpy"),
    ("real_f32", "saxpby"),
    ("real_f32", "sdot"),
    ("real_f32", "sasum"),
    ("real_f32", "snrm2"),
    ("real_f32", "srot"),
    ("real_f32", "srotm"),
    ("real_f64", "dscal"),
    ("real_f64", "daxpy"),
    ("real_f64", "daxpby"),
    ("real_f64", "ddot"),
    ("real_f64", "dasum"),
    ("real_f64", "dnrm2"),
    ("real_f64", "drot"),
    ("real_f64", "drotm"),
    ("mixed_dot", "sdsdot"),
    ("mixed_dot", "dsdot"),
    ("complex_f32", "csscal"),
    ("complex_f32", "cscal"),
    ("complex_f32", "caxpy"),
    ("complex_f32", "caxpby"),
    ("complex_f32", "cdotu"),
    ("complex_f32", "cdotc"),
    ("complex_f32", "scasum"),
    ("complex_f32", "scnrm2"),
    ("complex_f32", "csrot"),
    ("complex_f64", "zdscal"),
    ("complex_f64", "zscal"),
    ("complex_f64", "zaxpy"),
    ("complex_f64", "zaxpby"),
    ("complex_f64", "zdotu"),
    ("complex_f64", "zdotc"),
    ("complex_f64", "dzasum"),
    ("complex_f64", "dznrm2"),
    ("complex_f64", "zdrot"),
]

LEVEL1_BANDWIDTH_OPS = {
    "scopy",
    "dcopy",
    "ccopy",
    "zcopy",
    "sswap",
    "dswap",
    "cswap",
    "zswap",
    "isamax",
    "idamax",
    "icamax",
    "izamax",
}

LEVEL1_VARIANTS = {
    "srotm": ("flag_m1", "flag_0", "flag_p1"),
    "drotm": ("flag_m1", "flag_0", "flag_p1"),
}

# Only operations with a portable cross-library negative-stride contract belong
# in the default negative gate. In particular, AXPBY is not admitted merely
# because a library exports a symbol.
NEGATIVE_LEVEL1_OPS = [
    ("copy", "scopy"),
    ("copy", "dcopy"),
    ("copy", "ccopy"),
    ("copy", "zcopy"),
    ("swap", "sswap"),
    ("swap", "dswap"),
    ("swap", "cswap"),
    ("swap", "zswap"),
    ("real_f32", "saxpy"),
    ("real_f32", "sdot"),
    ("real_f32", "srot"),
    ("real_f32", "srotm"),
    ("real_f64", "daxpy"),
    ("real_f64", "ddot"),
    ("real_f64", "drot"),
    ("real_f64", "drotm"),
    ("mixed_dot", "sdsdot"),
    ("mixed_dot", "dsdot"),
    ("complex_f32", "caxpy"),
    ("complex_f32", "cdotu"),
    ("complex_f32", "cdotc"),
    ("complex_f32", "csrot"),
    ("complex_f64", "zaxpy"),
    ("complex_f64", "zdotu"),
    ("complex_f64", "zdotc"),
    ("complex_f64", "zdrot"),
]
STABLE_NEGATIVE_OPS = frozenset(op for _, op in NEGATIVE_LEVEL1_OPS)

COPY_KIND_ORDER = ("s", "d", "c", "z")
COPY_KIND_SPECS = {
    "s": ("scopy", "scopy_", 4),
    "d": ("dcopy", "dcopy_", 8),
    "c": ("ccopy", "ccopy_", 8),
    "z": ("zcopy", "zcopy_", 16),
}
DEFAULT_COPY_BYTE_SIZES = (
    4 * 1024,
    8 * 1024,
    16 * 1024,
    32 * 1024,
    64 * 1024,
    128 * 1024,
    256 * 1024,
    512 * 1024,
    1 * 1024 * 1024,
    2 * 1024 * 1024,
    3 * 1024 * 1024,
    4 * 1024 * 1024,
    5 * 1024 * 1024,
    6 * 1024 * 1024,
    7 * 1024 * 1024,
    8 * 1024 * 1024,
    10 * 1024 * 1024,
    12 * 1024 * 1024,
    14 * 1024 * 1024,
    15 * 1024 * 1024,
    16 * 1024 * 1024,
    24 * 1024 * 1024,
    32 * 1024 * 1024,
)

LIBS = [
    ("Zynum", "zynum"),
    ("Accelerate", "accelerate"),
    ("OpenBLAS", "openblas"),
]

RESULT_RE = re.compile(
    r"(?:rate_Gops=(?P<gops>[-+0-9.eE]+)\s+)?bandwidth_GBps=(?P<gbps>[-+0-9.eE]+)"
)
SURFACE_RE = re.compile(r"\bsymbol=(?P<symbol>\S+)\s+abi_surface=(?P<abi_surface>\S+)")
COPY_CHECK_CACHE = {}
LEVEL1_CHECK_CACHE = {}


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


def default_executable(path):
    return f"{path}.exe" if sys.platform == "win32" else path


def parse_byte_size(value):
    raw = value.strip().lower().replace("_", "")
    match = re.fullmatch(r"([0-9]+)([a-z]*)", raw)
    if match is None:
        raise argparse.ArgumentTypeError(f"invalid byte size: {value}")
    number = int(match.group(1), 10)
    suffix = match.group(2)
    factors = {
        "": 1,
        "b": 1,
        "k": 1024,
        "kb": 1024,
        "kib": 1024,
        "m": 1024 * 1024,
        "mb": 1024 * 1024,
        "mib": 1024 * 1024,
        "g": 1024 * 1024 * 1024,
        "gb": 1024 * 1024 * 1024,
        "gib": 1024 * 1024 * 1024,
    }
    if suffix not in factors:
        raise argparse.ArgumentTypeError(f"invalid byte-size suffix: {value}")
    return number * factors[suffix]


def parse_stride(value):
    try:
        stride = int(value, 10)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid stride: {value}") from exc
    if stride == 0 or stride < -(1 << 31) or stride > (1 << 31) - 1:
        raise argparse.ArgumentTypeError(
            "stride must be a nonzero signed 32-bit BLAS integer"
        )
    return stride


def format_byte_size(value):
    units = (
        (1024 * 1024 * 1024, "GiB"),
        (1024 * 1024, "MiB"),
        (1024, "KiB"),
    )
    for factor, suffix in units:
        if value >= factor and value % factor == 0:
            return f"{value // factor}{suffix}"
    return f"{value}B"


def unique_preserving_order(values):
    seen = set()
    result = []
    for value in values:
        if value not in seen:
            seen.add(value)
            result.append(value)
    return result


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Run representative Level 1 fresh-process probes and write a report CSV."
    )
    parser.add_argument(
        "--level1-probe",
        default=default_executable("zig-out/perf-report/bin/level1_probe"),
    )
    parser.add_argument(
        "--copy-probe",
        default=default_executable("zig-out/perf-report/bin/dcopy_probe"),
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
        help="Additional drop-in BLAS comparator. May be passed more than once.",
    )
    parser.add_argument("--n", type=int, default=1024 * 1024)
    parser.add_argument(
        "--inc",
        dest="strides",
        action="append",
        type=parse_stride,
        help=(
            "Compatibility stride applied to both vectors. May be passed more "
            "than once; defaults to 1."
        ),
    )
    parser.add_argument(
        "--incx",
        dest="incx_values",
        action="append",
        type=parse_stride,
        help="Independent X stride. May be passed more than once.",
    )
    parser.add_argument(
        "--incy",
        dest="incy_values",
        action="append",
        type=parse_stride,
        help="Independent Y stride. May be passed more than once.",
    )
    parser.add_argument("--seconds", type=int, default=1)
    parser.add_argument("--copy-seconds", type=int, default=1)
    parser.add_argument(
        "--copy-byte-size",
        dest="copy_byte_sizes",
        action="append",
        type=parse_byte_size,
        help=(
            "Byte size for a copy case. May be passed more than once; accepts "
            "plain bytes or K/KiB/M/MiB/G/GiB suffixes. Defaults to a cache- and "
            "dispatch-boundary coverage set."
        ),
    )
    parser.add_argument(
        "--copy-only",
        action="store_true",
        help="Run only the copy byte-size coverage cases.",
    )
    parser.add_argument(
        "--skip-copy-byte-coverage",
        action="store_true",
        help="Skip the independent COPY byte-size sweep while retaining vector COPY operations.",
    )
    parser.add_argument(
        "--group", action="append", help="Restrict to one or more Level 1 groups."
    )
    parser.add_argument(
        "--op", action="append", help="Restrict to one or more operations."
    )
    parser.add_argument("--process-repeats", type=int, default=1)
    parser.add_argument(
        "--process-schedule",
        choices=SCHEDULE_CHOICES,
        default="library-major",
        help="Fresh-process library schedule (default: library-major).",
    )
    parser.add_argument("--csv", required=True)
    parser.add_argument("--skip-missing", action="store_true")
    benchmark_metadata.add_identity_arguments(parser)
    args = parser.parse_args(argv)
    if args.process_repeats < 1:
        parser.error("--process-repeats must be at least 1")
    if args.copy_byte_sizes is None:
        args.copy_byte_sizes = list(DEFAULT_COPY_BYTE_SIZES)
    args.copy_byte_sizes = unique_preserving_order(args.copy_byte_sizes)
    if args.strides is not None and (
        args.incx_values is not None or args.incy_values is not None
    ):
        parser.error("--inc cannot be combined with --incx or --incy")
    if args.incx_values is not None or args.incy_values is not None:
        incx_values = unique_preserving_order(args.incx_values or [1])
        incy_values = unique_preserving_order(args.incy_values or [1])
        args.stride_pairs = [
            (incx, incy) for incx in incx_values for incy in incy_values
        ]
        args.strides = None
    else:
        args.strides = unique_preserving_order(args.strides or [1])
        args.stride_pairs = [(stride, stride) for stride in args.strides]
    if not args.copy_byte_sizes:
        parser.error("at least one --copy-byte-size is required")
    for byte_count in args.copy_byte_sizes:
        if byte_count <= 0:
            parser.error("--copy-byte-size must be positive")
        if byte_count % COPY_KIND_SPECS["s"][2] != 0:
            parser.error("--copy-byte-size must be divisible by 4 bytes")
    return args


def copy_cases(byte_sizes):
    cases = []
    for index, byte_count in enumerate(byte_sizes):
        start = index % len(COPY_KIND_ORDER)
        ordered_kinds = COPY_KIND_ORDER[start:] + COPY_KIND_ORDER[:start]
        kind = next(
            (
                candidate
                for candidate in ordered_kinds
                if byte_count % COPY_KIND_SPECS[candidate][2] == 0
            ),
            None,
        )
        if kind is None:
            raise ValueError(
                f"no BLAS copy kind can represent {byte_count} bytes exactly"
            )
        op, _, elem_size = COPY_KIND_SPECS[kind]
        cases.append(
            {
                "group": "copy",
                "op": op,
                "kind": kind,
                "copy_bytes": byte_count,
                "copy_elements": byte_count // elem_size,
            }
        )
    return cases


def selected_copy_cases(args):
    if args.skip_copy_byte_coverage:
        return []
    return [
        case
        for case in copy_cases(args.copy_byte_sizes)
        if case_allowed(args, case["group"], case["op"])
    ]


def level1_cases(stride_pairs):
    cases = []
    for incx, incy in stride_pairs:
        operations = NEGATIVE_LEVEL1_OPS if incx < 0 or incy < 0 else LEVEL1_OPS
        for group, op in operations:
            for variant in LEVEL1_VARIANTS.get(op, ("default",)):
                cases.append((group, op, variant, incx, incy))
    return cases


def case_allowed(args, group, op):
    if args.group is not None and group not in args.group:
        return False
    if args.op is not None and op not in args.op:
        return False
    return True


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
    if args.atlas:
        result.append(("ATLAS", args.atlas))
    for item in args.extra_blas:
        if "=" not in item:
            raise ValueError(f"--extra-blas must be LABEL=PATH, got {item!r}")
        label, path = item.split("=", 1)
        label = label.strip()
        path = path.strip()
        if not label or not path:
            raise ValueError(f"--extra-blas must be LABEL=PATH, got {item!r}")
        result.append((label, path))
    return result


def platform_image_path(path):
    return sys.platform == "darwin" and path == DEFAULT_ACCELERATE


def library_available(path):
    return Path(path).is_file() or platform_image_path(path)


def library_artifact_request(name, path):
    if Path(path).is_file():
        return ArtifactRequest.library(name, path)
    if name == "Accelerate" and platform_image_path(path):
        return ArtifactRequest.platform_image(name, path)
    return ArtifactRequest.library(name, path)


def check_result(
    status,
    error=None,
    raw="",
    *,
    symbol="",
    abi_surface="",
    capability_status=None,
    memory_status="not-run",
):
    if capability_status is None:
        capability_status = "supported" if symbol else "unavailable"
    return {
        "check_status": status,
        "check_max_abs_error": "" if error is None else f"{error:.9g}",
        "check_raw_output": raw,
        "preflight_symbol": symbol,
        "preflight_abi_surface": abi_surface,
        "capability_status": capability_status,
        "check_memory_status": memory_status,
    }


def next_fill(seed):
    seed[0] = (seed[0] * 6364136223846793005 + 1442695040888963407) & ((1 << 64) - 1)
    return ((seed[0] >> 32) % 1000) / 1000.0 - 0.5


class VectorLayout:
    def __init__(self, count, inc):
        if count < 0:
            raise ValueError("vector count must be nonnegative")
        if inc == 0 or inc < -(1 << 31) or inc > (1 << 31) - 1:
            raise ValueError("stride must be a nonzero signed BLAS integer")
        self.count = count
        self.inc = inc
        self.magnitude = abs(inc)
        self.span = 0 if count == 0 else 1 + (count - 1) * self.magnitude
        self.start = self.span - 1 if count and inc < 0 else 0

    def index(self, logical_index):
        if logical_index < 0 or logical_index >= self.count:
            raise IndexError(logical_index)
        offset = logical_index * self.magnitude
        return self.start - offset if self.inc < 0 else self.start + offset

    def is_active(self, physical_index):
        if physical_index < 0 or physical_index >= self.span or self.count == 0:
            return False
        if self.inc < 0:
            if physical_index > self.start:
                return False
            distance = self.start - physical_index
        else:
            distance = physical_index
        return (
            distance % self.magnitude == 0 and distance // self.magnitude < self.count
        )


GUARD_ELEMENTS = 8


class GuardedArray:
    def __init__(self, element_type, count, inc):
        self.element_type = element_type
        self.layout = VectorLayout(count, inc)
        self.element_size = ctypes.sizeof(element_type)
        total = self.layout.span + 2 * GUARD_ELEMENTS
        self.storage = (element_type * total)()
        ctypes.memset(ctypes.addressof(self.storage), 0xA5, ctypes.sizeof(self.storage))
        self.ptr = ctypes.cast(
            ctypes.byref(self.storage, GUARD_ELEMENTS * self.element_size),
            ctypes.POINTER(element_type),
        )
        self.snapshot = None

    @property
    def _as_parameter_(self):
        return self.ptr

    def __getitem__(self, physical_index):
        return self.ptr[physical_index]

    def __setitem__(self, physical_index, value):
        self.ptr[physical_index] = value

    def set_logical(self, logical_index, value):
        self.ptr[self.layout.index(logical_index)] = value

    def capture(self):
        self.snapshot = ctypes.string_at(
            ctypes.addressof(self.storage), ctypes.sizeof(self.storage)
        )

    def modified_element_count(self, active_may_change):
        if self.snapshot is None:
            raise RuntimeError("guarded array snapshot was not captured")
        current = ctypes.string_at(
            ctypes.addressof(self.storage), ctypes.sizeof(self.storage)
        )
        modified = 0
        for allocation_index in range(len(self.storage)):
            in_data = (
                GUARD_ELEMENTS <= allocation_index < GUARD_ELEMENTS + self.layout.span
            )
            physical_index = allocation_index - GUARD_ELEMENTS if in_data else 0
            if active_may_change and in_data and self.layout.is_active(physical_index):
                continue
            start = allocation_index * self.element_size
            end = start + self.element_size
            if current[start:end] != self.snapshot[start:end]:
                modified += 1
        return modified


def vector_storage_len(count, stride):
    return VectorLayout(count, stride).span


def vector_start(count, stride):
    return VectorLayout(count, stride).start


def vector_index(count, stride, logical_index):
    return VectorLayout(count, stride).index(logical_index)


def real_array(ctype, count, seed_value, stride=1):
    seed = [seed_value]
    result = GuardedArray(ctype, count, stride)
    for physical_index in range(result.layout.span):
        result[physical_index] = ctype(next_fill(seed))
    values = [float(result[result.layout.index(i)]) for i in range(count)]
    result.capture()
    return result, values


def complex_array(complex_type, scalar_type, count, seed_value, stride=1):
    seed = [seed_value]
    result = GuardedArray(complex_type, count, stride)
    for physical_index in range(result.layout.span):
        result[physical_index] = complex_type(
            scalar_type(next_fill(seed)).value,
            scalar_type(next_fill(seed)).value,
        )
    values = [
        complex(
            float(result[result.layout.index(i)].re),
            float(result[result.layout.index(i)].im),
        )
        for i in range(count)
    ]
    result.capture()
    return result, values


def real_array_from_logical(ctype, values, stride):
    result = GuardedArray(ctype, len(values), stride)
    for i, value in enumerate(values):
        result.set_logical(i, ctype(value))
    result.capture()
    return result


def max_real_error(actual, expected, stride=1):
    layout = getattr(actual, "layout", VectorLayout(len(expected), stride))
    return max(
        (
            abs(float(actual[layout.index(i)]) - expected[i])
            for i in range(len(expected))
        ),
        default=0.0,
    )


def max_complex_error(actual, expected, stride=1):
    layout = getattr(actual, "layout", VectorLayout(len(expected), stride))
    return max(
        (
            abs(
                complex(
                    float(actual[layout.index(i)].re),
                    float(actual[layout.index(i)].im),
                )
                - expected[i]
            )
            for i in range(len(expected))
        ),
        default=0.0,
    )


def complex_values(array):
    if isinstance(array, GuardedArray):
        return [
            complex(
                float(array[array.layout.index(i)].re),
                float(array[array.layout.index(i)].im),
            )
            for i in range(array.layout.count)
        ]
    return [complex(float(v.re), float(v.im)) for v in array]


def lookup(lib, *names):
    for name in names:
        try:
            return getattr(lib, name)
        except AttributeError:
            pass
    return None


def lookup_surface(lib, candidates):
    for name, abi_surface in candidates:
        try:
            return getattr(lib, name), name, abi_surface
        except AttributeError:
            pass
    return None, "", ""


def level1_symbol_candidates(op):
    if op in ("saxpby", "daxpby"):
        return (
            (f"cblas_{op}", "cblas"),
            (f"catlas_{op}", "catlas"),
            (f"{op}_", "fortran"),
        )
    if op in ("sdot", "ddot", "sdsdot", "dsdot"):
        return ((f"cblas_{op}", "cblas"), (f"{op}_", "fortran"))
    if op in ("cdotu", "zdotu", "cdotc", "zdotc"):
        return ((f"{op}_sub_", "fortran"), (f"cblas_{op}_sub", "cblas"))
    if op in (
        "sasum",
        "dasum",
        "scasum",
        "dzasum",
        "snrm2",
        "dnrm2",
        "scnrm2",
        "dznrm2",
    ):
        return ((f"cblas_{op}", "cblas"), (f"{op}_", "fortran"))
    return ((f"{op}_", "fortran"),)


def finish_level1_check(error, tolerance, symbol, abi_surface, buffers, mutable):
    modified = {}
    for name, buffer in buffers.items():
        count = buffer.modified_element_count(name in mutable)
        if count:
            modified[name] = count
    if modified:
        detail = ", ".join(
            f"{name}={count}" for name, count in sorted(modified.items())
        )
        return check_result(
            "correctness_failed",
            max(float(sum(modified.values())), error),
            raw=f"guard/gap/read-only storage modified: {detail}",
            symbol=symbol,
            abi_surface=abi_surface,
            memory_status="failed",
        )
    return check_result(
        "sampled-ok" if error <= tolerance else "correctness_failed",
        error,
        symbol=symbol,
        abi_surface=abi_surface,
        memory_status="guarded-ok",
    )


def check_copy_op(library_path, kind):
    cache_key = (library_path, kind)
    if cache_key in COPY_CHECK_CACHE:
        return COPY_CHECK_CACHE[cache_key]
    try:
        lib = ctypes.CDLL(library_path)
    except OSError as exc:
        result = check_result("missing", raw=str(exc), capability_status="unavailable")
        COPY_CHECK_CACHE[cache_key] = result
        return result
    _, symbol, elem_size = COPY_KIND_SPECS[kind]
    fn, actual_symbol, abi_surface = lookup_surface(lib, ((symbol, "fortran"),))
    if fn is None:
        result = check_result(
            "missing",
            raw=f"missing {symbol}",
            capability_status="unsupported",
        )
        COPY_CHECK_CACHE[cache_key] = result
        return result
    n = 257
    byte_count = n * elem_size
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
        result = check_result(
            "correctness_failed",
            float(mismatches),
            f"{mismatches} byte mismatches",
            symbol=actual_symbol,
            abi_surface=abi_surface,
        )
        COPY_CHECK_CACHE[cache_key] = result
        return result
    result = check_result(
        "sampled-ok",
        0.0,
        symbol=actual_symbol,
        abi_surface=abi_surface,
        memory_status="unit-span-ok",
    )
    COPY_CHECK_CACHE[cache_key] = result
    return result


def check_worker_result(
    check_type,
    library_path,
    name,
    incx=1,
    incy=1,
    variant="default",
    *,
    controller_script=None,
    public_library_path=None,
    artifacts=None,
):
    script_path = controller_script or __file__
    cmd = [
        sys.executable,
        script_path,
        "--check-worker",
        "--check-type",
        check_type,
        "--check-library",
        library_path,
        "--check-name",
        name,
        "--check-incx",
        str(incx),
        "--check-incy",
        str(incy),
        "--check-variant",
        variant,
    ]
    if controller_script is not None and cmd[1] != controller_script:
        raise ValueError("check worker did not use the frozen controller script")
    library_arg = cmd[cmd.index("--check-library") + 1]
    if library_arg != library_path:
        raise ValueError("check worker did not use the frozen library")
    env = os.environ.copy()
    tool_dir = str(Path(__file__).resolve().parent)
    env["PYTHONPATH"] = os.pathsep.join(
        part for part in (tool_dir, env.get("PYTHONPATH")) if part
    )
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, env=env)
    except OSError:
        return check_result("error", raw="failed to start frozen check worker")
    stdout = result.stdout or ""
    stderr = result.stderr or ""
    if artifacts is not None:
        stdout, stderr = artifacts.redact_private_paths((stdout, stderr))
    output = stdout + stderr
    if result.returncode != 0:
        return check_result(
            "error",
            raw=f"check worker exited {result.returncode}: {output.strip()}",
        )
    try:
        decoded = json.loads(stdout)
    except json.JSONDecodeError as exc:
        return check_result(
            "error", raw=f"invalid check worker output: {exc}: {output.strip()}"
        )
    fields = (
        "check_status",
        "check_max_abs_error",
        "check_raw_output",
        "preflight_symbol",
        "preflight_abi_surface",
        "capability_status",
        "check_memory_status",
    )
    return {field: decoded.get(field, "") for field in fields}


def check_level1_op_isolated(
    library_path,
    op,
    incx=1,
    incy=1,
    variant="default",
    *,
    controller_script=None,
    public_library_path=None,
    artifacts=None,
):
    cache_key = (library_path, op, incx, incy, variant)
    if cache_key not in LEVEL1_CHECK_CACHE:
        LEVEL1_CHECK_CACHE[cache_key] = check_worker_result(
            "level1",
            library_path,
            op,
            incx,
            incy,
            variant,
            controller_script=controller_script,
            public_library_path=public_library_path,
            artifacts=artifacts,
        )
    return LEVEL1_CHECK_CACHE[cache_key]


def check_copy_op_isolated(
    library_path,
    kind,
    *,
    controller_script=None,
    public_library_path=None,
    artifacts=None,
):
    cache_key = (library_path, kind)
    if cache_key not in COPY_CHECK_CACHE:
        COPY_CHECK_CACHE[cache_key] = check_worker_result(
            "copy",
            library_path,
            kind,
            controller_script=controller_script,
            public_library_path=public_library_path,
            artifacts=artifacts,
        )
    return COPY_CHECK_CACHE[cache_key]


def check_level1_op(library_path, op, incx=1, incy=None, variant="default"):
    if incy is None:
        incy = incx
    if (incx < 0 or incy < 0) and op not in STABLE_NEGATIVE_OPS:
        return check_result(
            "missing",
            raw=f"{op} is excluded from the stable negative-stride set",
            capability_status="excluded-by-policy",
        )
    try:
        lib = ctypes.CDLL(library_path)
    except OSError as exc:
        return check_result("missing", raw=str(exc), capability_status="unavailable")

    candidates = level1_symbol_candidates(op)
    fn, symbol, abi_surface = lookup_surface(lib, candidates)
    if fn is None:
        names = "/".join(name for name, _ in candidates)
        return check_result(
            "missing",
            raw=f"missing {names}",
            capability_status="unsupported",
        )

    n = 257
    ni = ctypes.c_int(n)
    incx_arg = ctypes.c_int(incx)
    incy_arg = ctypes.c_int(incy)
    try:
        if op in ("isamax", "idamax", "icamax", "izamax"):
            target = 137
            if op in ("isamax", "idamax"):
                ctype = ctypes.c_float if op == "isamax" else ctypes.c_double
                x, x0 = real_array(ctype, n, 0x123456789ABCDEF0, incx)
                x.set_logical(target, ctype(-9.0))
                x0[target] = -9.0
                expected = max(range(n), key=lambda i: abs(x0[i])) + 1
            else:
                complex_type = ComplexF32 if op == "icamax" else ComplexF64
                scalar_type = ctypes.c_float if op == "icamax" else ctypes.c_double
                x, x0 = complex_array(
                    complex_type, scalar_type, n, 0x123456789ABCDEF0, incx
                )
                x.set_logical(target, complex_type(-9.0, 8.0))
                x0[target] = complex(-9.0, 8.0)
                expected = (
                    max(range(n), key=lambda i: abs(x0[i].real) + abs(x0[i].imag)) + 1
                )
            x.capture()
            fn.restype = ctypes.c_int
            got = int(fn(ctypes.byref(ni), x, ctypes.byref(incx_arg)))
            return finish_level1_check(
                float(abs(got - expected)),
                0.0,
                symbol,
                abi_surface,
                {"x": x},
                set(),
            )

        real_f32_ops = {
            "scopy",
            "sscal",
            "sswap",
            "saxpy",
            "saxpby",
            "sdot",
            "sdsdot",
            "dsdot",
            "sasum",
            "snrm2",
            "srot",
            "srotm",
        }
        real_f64_ops = {
            "dcopy",
            "dscal",
            "dswap",
            "daxpy",
            "daxpby",
            "ddot",
            "dasum",
            "dnrm2",
            "drot",
            "drotm",
        }
        if op in real_f32_ops or op in real_f64_ops:
            ctype = ctypes.c_float if op in real_f32_ops else ctypes.c_double
            tolerance = 1e-3 if ctype is ctypes.c_float else 1e-10
            x, x0 = real_array(ctype, n, 0x123456789ABCDEF0, incx)
            y, y0 = real_array(ctype, n, 0x0FEDCBA987654321, incy)
            alpha = ctype(0.75)
            mutable = set()

            if op in ("scopy", "dcopy"):
                fn(
                    ctypes.byref(ni),
                    x,
                    ctypes.byref(incx_arg),
                    y,
                    ctypes.byref(incy_arg),
                )
                error = max_real_error(y, x0)
                mutable.add("y")
            elif op.endswith("scal"):
                fn(ctypes.byref(ni), ctypes.byref(alpha), x, ctypes.byref(incx_arg))
                error = max_real_error(x, [alpha.value * value for value in x0])
                mutable.add("x")
            elif op.endswith("swap"):
                fn(
                    ctypes.byref(ni),
                    x,
                    ctypes.byref(incx_arg),
                    y,
                    ctypes.byref(incy_arg),
                )
                error = max(max_real_error(x, y0), max_real_error(y, x0))
                mutable.update(("x", "y"))
            elif op.endswith("axpy"):
                fn(
                    ctypes.byref(ni),
                    ctypes.byref(alpha),
                    x,
                    ctypes.byref(incx_arg),
                    y,
                    ctypes.byref(incy_arg),
                )
                error = max_real_error(
                    y, [alpha.value * x0[i] + y0[i] for i in range(n)]
                )
                mutable.add("y")
            elif op.endswith("axpby"):
                beta = ctype(0.5)
                if abi_surface in ("cblas", "catlas"):
                    fn(ni, alpha, x, incx_arg, beta, y, incy_arg)
                else:
                    fn(
                        ctypes.byref(ni),
                        ctypes.byref(alpha),
                        x,
                        ctypes.byref(incx_arg),
                        ctypes.byref(beta),
                        y,
                        ctypes.byref(incy_arg),
                    )
                error = max_real_error(
                    y,
                    [alpha.value * x0[i] + beta.value * y0[i] for i in range(n)],
                )
                mutable.add("y")
            elif op in ("sdot", "ddot", "sdsdot", "dsdot"):
                result_type = (
                    ctypes.c_float if op in ("sdot", "sdsdot") else ctypes.c_double
                )
                fn.restype = result_type
                expected = sum(x0[i] * y0[i] for i in range(n))
                if op == "sdsdot":
                    sb = ctypes.c_float(0.125)
                    expected = ctypes.c_float(sb.value + expected).value
                    if abi_surface == "cblas":
                        got = float(fn(ni, sb, x, incx_arg, y, incy_arg))
                    else:
                        got = float(
                            fn(
                                ctypes.byref(ni),
                                ctypes.byref(sb),
                                x,
                                ctypes.byref(incx_arg),
                                y,
                                ctypes.byref(incy_arg),
                            )
                        )
                    tolerance = 1e-4
                elif abi_surface == "cblas":
                    got = float(fn(ni, x, incx_arg, y, incy_arg))
                else:
                    got = float(
                        fn(
                            ctypes.byref(ni),
                            x,
                            ctypes.byref(incx_arg),
                            y,
                            ctypes.byref(incy_arg),
                        )
                    )
                error = abs(got - expected)
                if op != "sdsdot":
                    tolerance *= n
            elif op.endswith("asum"):
                fn.restype = ctype
                got = (
                    float(fn(ni, x, incx_arg))
                    if abi_surface == "cblas"
                    else float(fn(ctypes.byref(ni), x, ctypes.byref(incx_arg)))
                )
                error = abs(got - sum(abs(value) for value in x0))
                tolerance *= n
            elif op.endswith("nrm2"):
                fn.restype = ctype
                got = (
                    float(fn(ni, x, incx_arg))
                    if abi_surface == "cblas"
                    else float(fn(ctypes.byref(ni), x, ctypes.byref(incx_arg)))
                )
                error = abs(got - math.sqrt(sum(value * value for value in x0)))
                tolerance *= n
            elif op in ("srot", "drot"):
                c = ctype(0.8)
                s = ctype(0.6)
                fn(
                    ctypes.byref(ni),
                    x,
                    ctypes.byref(incx_arg),
                    y,
                    ctypes.byref(incy_arg),
                    ctypes.byref(c),
                    ctypes.byref(s),
                )
                error = max(
                    max_real_error(
                        x, [c.value * x0[i] + s.value * y0[i] for i in range(n)]
                    ),
                    max_real_error(
                        y, [c.value * y0[i] - s.value * x0[i] for i in range(n)]
                    ),
                )
                mutable.update(("x", "y"))
            elif op in ("srotm", "drotm"):
                a = 1.0 / 1024.0
                if variant in ("default", "flag_m1"):
                    param = (ctype * 5)(-1.0, 0.8, -0.6, 0.6, 0.8)
                    expected_x = [0.8 * x0[i] + 0.6 * y0[i] for i in range(n)]
                    expected_y = [-0.6 * x0[i] + 0.8 * y0[i] for i in range(n)]
                elif variant == "flag_0":
                    param = (ctype * 5)(0.0, math.nan, -a, a, math.nan)
                    expected_x = [x0[i] + a * y0[i] for i in range(n)]
                    expected_y = [-a * x0[i] + y0[i] for i in range(n)]
                elif variant == "flag_p1":
                    param = (ctype * 5)(1.0, a, math.nan, math.nan, -a)
                    expected_x = [a * x0[i] + y0[i] for i in range(n)]
                    expected_y = [-x0[i] - a * y0[i] for i in range(n)]
                else:
                    return check_result(
                        "error",
                        raw=f"invalid ROTM variant {variant}",
                        symbol=symbol,
                        abi_surface=abi_surface,
                    )
                fn(
                    ctypes.byref(ni),
                    x,
                    ctypes.byref(incx_arg),
                    y,
                    ctypes.byref(incy_arg),
                    param,
                )
                error = max(
                    max_real_error(x, expected_x), max_real_error(y, expected_y)
                )
                mutable.update(("x", "y"))
            else:
                return check_result(
                    "error",
                    raw=f"unhandled op {op}",
                    symbol=symbol,
                    abi_surface=abi_surface,
                )
            return finish_level1_check(
                error,
                tolerance,
                symbol,
                abi_surface,
                {"x": x, "y": y},
                mutable,
            )

        complex_type = ComplexF32 if op[0] in ("c", "s") else ComplexF64
        scalar_type = ctypes.c_float if complex_type is ComplexF32 else ctypes.c_double
        tolerance = 1e-3 if scalar_type is ctypes.c_float else 1e-10
        x, x0 = complex_array(complex_type, scalar_type, n, 0x123456789ABCDEF0, incx)
        y, y0 = complex_array(complex_type, scalar_type, n, 0x0FEDCBA987654321, incy)
        alpha_r = scalar_type(0.75)
        alpha_c = complex_type(0.75, -0.125)
        beta_c = complex_type(0.5, 0.25)
        mutable = set()
        if op in ("ccopy", "zcopy"):
            fn(
                ctypes.byref(ni),
                x,
                ctypes.byref(incx_arg),
                y,
                ctypes.byref(incy_arg),
            )
            error = max_complex_error(y, x0)
            mutable.add("y")
        elif op in ("csscal", "zdscal"):
            fn(ctypes.byref(ni), ctypes.byref(alpha_r), x, ctypes.byref(incx_arg))
            error = max_complex_error(x, [alpha_r.value * value for value in x0])
            mutable.add("x")
        elif op in ("cscal", "zscal"):
            fn(ctypes.byref(ni), ctypes.byref(alpha_c), x, ctypes.byref(incx_arg))
            alpha_value = complex(alpha_c.re, alpha_c.im)
            error = max_complex_error(x, [alpha_value * value for value in x0])
            mutable.add("x")
        elif op in ("cswap", "zswap"):
            fn(
                ctypes.byref(ni),
                x,
                ctypes.byref(incx_arg),
                y,
                ctypes.byref(incy_arg),
            )
            error = max(max_complex_error(x, y0), max_complex_error(y, x0))
            mutable.update(("x", "y"))
        elif op in ("caxpy", "zaxpy"):
            fn(
                ctypes.byref(ni),
                ctypes.byref(alpha_c),
                x,
                ctypes.byref(incx_arg),
                y,
                ctypes.byref(incy_arg),
            )
            alpha_value = complex(alpha_c.re, alpha_c.im)
            error = max_complex_error(
                y, [alpha_value * x0[i] + y0[i] for i in range(n)]
            )
            mutable.add("y")
        elif op in ("caxpby", "zaxpby"):
            fn(
                ctypes.byref(ni),
                ctypes.byref(alpha_c),
                x,
                ctypes.byref(incx_arg),
                ctypes.byref(beta_c),
                y,
                ctypes.byref(incy_arg),
            )
            alpha_value = complex(alpha_c.re, alpha_c.im)
            beta_value = complex(beta_c.re, beta_c.im)
            error = max_complex_error(
                y,
                [alpha_value * x0[i] + beta_value * y0[i] for i in range(n)],
            )
            mutable.add("y")
        elif op in ("cdotu", "zdotu", "cdotc", "zdotc"):
            out = complex_type()
            if abi_surface == "cblas":
                fn(ni, x, incx_arg, y, incy_arg, ctypes.byref(out))
            else:
                fn(
                    ctypes.byref(ni),
                    x,
                    ctypes.byref(incx_arg),
                    y,
                    ctypes.byref(incy_arg),
                    ctypes.byref(out),
                )
            got = complex(float(out.re), float(out.im))
            expected = sum(
                (x0[i].conjugate() if op.endswith("dotc") else x0[i]) * y0[i]
                for i in range(n)
            )
            error = abs(got - expected)
            tolerance *= n
        elif op in ("scasum", "dzasum"):
            fn.restype = scalar_type
            got = (
                float(fn(ni, x, incx_arg))
                if abi_surface == "cblas"
                else float(fn(ctypes.byref(ni), x, ctypes.byref(incx_arg)))
            )
            error = abs(got - sum(abs(v.real) + abs(v.imag) for v in x0))
            tolerance *= n
        elif op in ("scnrm2", "dznrm2"):
            fn.restype = scalar_type
            got = (
                float(fn(ni, x, incx_arg))
                if abi_surface == "cblas"
                else float(fn(ctypes.byref(ni), x, ctypes.byref(incx_arg)))
            )
            error = abs(
                got - math.sqrt(sum(v.real * v.real + v.imag * v.imag for v in x0))
            )
            tolerance *= n
        elif op in ("csrot", "zdrot"):
            c = scalar_type(0.8)
            s = scalar_type(0.6)
            fn(
                ctypes.byref(ni),
                x,
                ctypes.byref(incx_arg),
                y,
                ctypes.byref(incy_arg),
                ctypes.byref(c),
                ctypes.byref(s),
            )
            error = max(
                max_complex_error(
                    x, [c.value * x0[i] + s.value * y0[i] for i in range(n)]
                ),
                max_complex_error(
                    y, [c.value * y0[i] - s.value * x0[i] for i in range(n)]
                ),
            )
            mutable.update(("x", "y"))
        else:
            return check_result(
                "error",
                raw=f"unhandled op {op}",
                symbol=symbol,
                abi_surface=abi_surface,
            )
        return finish_level1_check(
            error,
            tolerance,
            symbol,
            abi_surface,
            {"x": x, "y": y},
            mutable,
        )
    except Exception as exc:
        return check_result(
            "error", raw=str(exc), symbol=symbol, abi_surface=abi_surface
        )


def parse_probe_output(output):
    match = RESULT_RE.search(output)
    if not match:
        return None, None, "", ""
    gops = match.group("gops")
    gbps = match.group("gbps")
    surface = SURFACE_RE.search(output)
    return (
        float(gops) if gops is not None else None,
        float(gbps),
        surface.group("symbol") if surface else "",
        surface.group("abi_surface") if surface else "",
    )


def run_once(
    cmd,
    *,
    expected_binary=None,
    expected_library=None,
    artifacts=None,
):
    if expected_binary is not None and cmd[0] != expected_binary:
        raise ValueError("probe command did not use the frozen binary")
    if expected_library is not None:
        library_arg = cmd[cmd.index("--lib") + 1]
        if library_arg != expected_library:
            raise ValueError("probe command did not use the frozen library")
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
    except OSError:
        return {
            "status": "error",
            "returncode": "",
            "rate_gops": None,
            "bandwidth_gbps": None,
            "symbol": "",
            "abi_surface": "",
            "raw_output": "failed to start frozen Level 1 probe",
        }
    output = (result.stdout or "") + (result.stderr or "")
    if artifacts is not None:
        output = artifacts.redact_private_paths(output)
    if result.returncode != 0:
        return {
            "status": "missing"
            if "MissingSymbol" in output or "MissingCopy" in output
            else "error",
            "returncode": result.returncode,
            "rate_gops": None,
            "bandwidth_gbps": None,
            "symbol": "",
            "abi_surface": "",
            "raw_output": output.strip(),
        }
    gops, gbps, symbol, abi_surface = parse_probe_output(output)
    if gbps is None:
        return {
            "status": "parse_error",
            "returncode": result.returncode,
            "rate_gops": None,
            "bandwidth_gbps": None,
            "symbol": "",
            "abi_surface": "",
            "raw_output": output.strip(),
        }
    return {
        "status": "ok",
        "returncode": result.returncode,
        "rate_gops": gops,
        "bandwidth_gbps": gbps,
        "symbol": symbol,
        "abi_surface": abi_surface,
        "raw_output": output.strip(),
    }


def metric_value(row):
    if row["metric"] == "bandwidth_gbps":
        return row["bandwidth_gbps"]
    return row["rate_gops"]


def choose_best(rows):
    if not rows:
        raise ValueError("cannot aggregate an empty Level 1 repeat group")
    for repeat, row in enumerate(rows, 1):
        if row is None:
            raise ValueError(f"repeat {repeat} is missing its result")
        status = row.get("status")
        if status != "ok":
            raise ValueError(
                f"repeat {repeat} returned non-ok status {status!r}: "
                f"{row.get('raw_output', '')}"
            )
        metric_field = (
            "bandwidth_gbps" if row["metric"] == "bandwidth_gbps" else "rate_gops"
        )
        other_field = (
            "rate_gops" if metric_field == "bandwidth_gbps" else "bandwidth_gbps"
        )
        validate_performance_fields(
            row,
            required=(metric_field,),
            optional=(other_field,),
        )
    values = [metric_value(row) for row in rows]
    ordered = sorted(values)
    best = dict(max(rows, key=metric_value))
    summary = {
        "metric_min": ordered[0],
        "metric_median": positive_finite_median(values, "metric_median"),
        "metric_max": ordered[-1],
        "metric_samples": ",".join(format(value, ".17g") for value in values),
    }
    validate_optional_metric_evidence(summary)
    best.update(
        {
            "process_repeats": len(rows),
            "successful_repeats": len(rows),
            **summary,
        }
    )
    return best


def unchecked_row(
    args,
    library_name,
    library_path,
    group,
    op,
    metric,
    check,
    n=None,
    seconds=None,
    extra=None,
    variant="default",
    incx=1,
    incy=1,
):
    status = "missing" if check["check_status"] == "missing" else "correctness_failed"
    if check["check_status"] == "error":
        status = "error"
    row = {
        "group": group,
        "op": op,
        "variant": variant,
        "library": library_name,
        "library_path": library_path,
        "n": args.n if n is None else n,
        "incx": incx,
        "incy": incy,
        "seconds": args.seconds if seconds is None else seconds,
        "repeat": "",
        "metric": metric,
        "status": status,
        "returncode": "",
        "rate_gops": None,
        "bandwidth_gbps": None,
        "symbol": check.get("preflight_symbol", ""),
        "abi_surface": check.get("preflight_abi_surface", ""),
        "raw_output": check["check_raw_output"],
        **check,
    }
    if extra is not None:
        row.update(extra)
    return row


def run_level1_op(
    args,
    library_name,
    library_path,
    group,
    op,
    variant,
    incx,
    incy,
    *,
    execution_library_path=None,
    controller_script_path=None,
    artifacts=None,
):
    private_library = execution_library_path or library_path
    metric = "bandwidth_gbps" if op in LEVEL1_BANDWIDTH_OPS else "rate_gops"
    check = check_level1_op_isolated(
        private_library,
        op,
        incx,
        incy,
        variant,
        controller_script=controller_script_path,
        public_library_path=library_path,
        artifacts=artifacts,
    )
    if check["check_status"] != "sampled-ok":
        return unchecked_row(
            args,
            library_name,
            library_path,
            group,
            op,
            metric,
            check,
            variant=variant,
            incx=incx,
            incy=incy,
        )
    rows = []
    cmd = [
        args.level1_probe,
        "--lib",
        private_library,
        "--op",
        op,
        "--variant",
        variant,
        "--incx",
        str(incx),
        "--incy",
        str(incy),
        "--n",
        str(args.n),
        "--seconds",
        str(args.seconds),
    ]
    for repeat in range(args.process_repeats):
        result = run_once(
            cmd,
            expected_binary=args.level1_probe if artifacts is not None else None,
            expected_library=private_library if artifacts is not None else None,
            artifacts=artifacts,
        )
        expected_surface = (
            check.get("preflight_symbol", ""),
            check.get("preflight_abi_surface", ""),
        )
        actual_surface = (result.get("symbol", ""), result.get("abi_surface", ""))
        if result["status"] == "ok" and actual_surface != expected_surface:
            result.update(
                {
                    "status": "surface_mismatch",
                    "rate_gops": None,
                    "bandwidth_gbps": None,
                    "raw_output": (
                        f"{result['raw_output']}\npreflight surface "
                        f"{expected_surface[0]}/{expected_surface[1]} != probe "
                        f"surface {actual_surface[0]}/{actual_surface[1]}"
                    ).strip(),
                }
            )
        rows.append(
            {
                "group": group,
                "op": op,
                "variant": variant,
                "library": library_name,
                "library_path": library_path,
                "n": args.n,
                "incx": incx,
                "incy": incy,
                "seconds": args.seconds,
                "repeat": repeat,
                "metric": metric,
                **result,
                **check,
            }
        )
    return choose_best(rows)


def run_copy_op(
    args,
    library_name,
    library_path,
    case,
    *,
    execution_library_path=None,
    controller_script_path=None,
    artifacts=None,
):
    private_library = execution_library_path or library_path
    group = case["group"]
    op = case["op"]
    kind = case["kind"]
    copy_bytes = case["copy_bytes"]
    copy_elements = case["copy_elements"]
    copy_extra = {
        "copy_bytes": copy_bytes,
        "copy_kind": kind,
        "copy_elements": copy_elements,
    }
    check = check_copy_op_isolated(
        private_library,
        kind,
        controller_script=controller_script_path,
        public_library_path=library_path,
        artifacts=artifacts,
    )
    if check["check_status"] != "sampled-ok":
        return unchecked_row(
            args,
            library_name,
            library_path,
            group,
            op,
            "bandwidth_gbps",
            check,
            n=copy_elements,
            seconds=args.copy_seconds,
            extra=copy_extra,
        )
    rows = []
    cmd = [
        args.copy_probe,
        "--lib",
        private_library,
        "--kind",
        kind,
        "--n",
        str(copy_elements),
        "--seconds",
        str(args.copy_seconds),
    ]
    for repeat in range(args.process_repeats):
        result = run_once(
            cmd,
            expected_binary=args.copy_probe if artifacts is not None else None,
            expected_library=private_library if artifacts is not None else None,
            artifacts=artifacts,
        )
        if not result.get("symbol"):
            result["symbol"] = check.get("preflight_symbol", "")
            result["abi_surface"] = check.get("preflight_abi_surface", "")
        rows.append(
            {
                "group": group,
                "op": op,
                "variant": "default",
                "library": library_name,
                "library_path": library_path,
                "n": copy_elements,
                "incx": 1,
                "incy": 1,
                "seconds": args.copy_seconds,
                "repeat": repeat,
                "metric": "bandwidth_gbps",
                **copy_extra,
                **result,
                **check,
            }
        )
    return choose_best(rows)


def check_worker_main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--check-worker", action="store_true")
    parser.add_argument("--check-type", choices=("copy", "level1"), required=True)
    parser.add_argument("--check-library", required=True)
    parser.add_argument("--check-name", required=True)
    parser.add_argument("--check-inc", type=parse_stride)
    parser.add_argument("--check-incx", type=parse_stride)
    parser.add_argument("--check-incy", type=parse_stride)
    parser.add_argument("--check-variant", default="default")
    args = parser.parse_args()
    if args.check_type == "copy":
        result = check_copy_op(args.check_library, args.check_name)
    else:
        legacy_inc = args.check_inc if args.check_inc is not None else 1
        result = check_level1_op(
            args.check_library,
            args.check_name,
            args.check_incx if args.check_incx is not None else legacy_inc,
            args.check_incy if args.check_incy is not None else legacy_inc,
            args.check_variant,
        )
    json.dump(result, sys.stdout, sort_keys=True)
    sys.stdout.write("\n")
    return 0


def serialize_report(
    args,
    selected_libraries,
    rows,
    identity,
    binary_records,
    library_records,
):
    fields = [
        "group",
        "op",
        "variant",
        "library",
        "library_path",
        "n",
        "incx",
        "incy",
        "copy_bytes",
        "copy_kind",
        "copy_elements",
        "seconds",
        "repeat",
        "metric",
        "process_repeats",
        "successful_repeats",
        "metric_min",
        "metric_median",
        "metric_max",
        "metric_samples",
        "status",
        "returncode",
        "rate_gops",
        "bandwidth_gbps",
        "symbol",
        "abi_surface",
        "preflight_symbol",
        "preflight_abi_surface",
        "capability_status",
        "check_status",
        "check_max_abs_error",
        "check_memory_status",
        "check_raw_output",
        "raw_output",
    ]
    csv_buffer = io.StringIO(newline="")
    writer = csv.DictWriter(csv_buffer, fieldnames=fields)
    writer.writeheader()
    for row in rows:
        writer.writerow({field: row.get(field, "") for field in fields})
    csv_bytes = csv_buffer.getvalue().encode("utf-8")

    binaries_by_name = {record["name"]: record for record in binary_records}
    libraries_by_name = {record["name"]: record for record in library_records}
    meta = {
        "generated_at_unix": time.time(),
        "isolation": (
            "fresh process per library/op/size/repeat; best repeat kept as the "
            "primary metric with min/median/max and ordered samples retained"
        ),
        "n": args.n,
        "seconds": args.seconds,
        "copy_seconds": args.copy_seconds,
        "copy_only": args.copy_only,
        "copy_byte_coverage": not args.skip_copy_byte_coverage,
        "copy_byte_sizes": args.copy_byte_sizes,
        "groups": args.group,
        "ops": args.op,
        "level1_variants": LEVEL1_VARIANTS,
        "level1_strides": args.strides,
        "level1_stride_pairs": [list(pair) for pair in args.stride_pairs],
        "stable_negative_operations": sorted(STABLE_NEGATIVE_OPS),
        "negative_stride_policy": (
            "Only stable_negative_operations are generated when incx or incy "
            "is negative; AXPBY, SCAL, ASUM, IAMAX, and NRM2 are excluded."
        ),
        "copy_case_policy": (
            "Copy rows use the concrete scopy/dcopy/ccopy/zcopy op name; n is "
            "the function element count, and copy_bytes records the measured "
            "byte footprint. The independent byte-size sweep may be disabled "
            "without disabling vector COPY operation coverage."
        ),
        "process_repeats": args.process_repeats,
        "correctness_check": (
            "isolated per library/op/variant/incx/incy before timing with "
            "logical-stride references, guard canaries, and gap checks"
        ),
        "environment": environment_snapshot(),
        "zynum_maximum_threads_detected": zynum_maximum_threads_detected(),
        "zig_version": zig_version(),
        "git_revision": benchmark_metadata.source_git_revision(identity["source"]),
        "probes": {
            "level1_probe": args.level1_probe,
            "level1_probe_sha256": binaries_by_name["level1_probe"]["sha256"],
            "copy_probe": args.copy_probe,
            "copy_probe_sha256": binaries_by_name["copy_probe"]["sha256"],
        },
        "libraries": {
            name: {
                "path": path,
                "sha256": libraries_by_name[name]["sha256"],
            }
            for name, path in selected_libraries
        },
        "benchmark_identity": identity,
    }
    metadata_bytes = benchmark_metadata.serialize_public_metadata(
        meta,
        controller="run_level1_report.py",
        parameter_keys=(
            "n",
            "seconds",
            "copy_seconds",
            "copy_only",
            "copy_byte_coverage",
            "copy_byte_sizes",
            "groups",
            "ops",
            "level1_strides",
            "level1_stride_pairs",
            "process_repeats",
        ),
    )
    return csv_bytes, metadata_bytes


def run_controller(args, selected_libraries, scheduled_cases, schedule):
    requests = [
        ArtifactRequest.binary("level1_probe", args.level1_probe),
        ArtifactRequest.binary("copy_probe", args.copy_probe),
        ArtifactRequest.interpreter_script("run_level1_report", __file__),
    ]
    requests.extend(
        library_artifact_request(name, path) for name, path in selected_libraries
    )
    artifacts = ArtifactSnapshotSet.capture(requests)
    try:
        frozen_binaries = artifacts.for_role("binary")
        frozen_libraries = artifacts.for_role("library")
        identity = benchmark_metadata.collect_benchmark_identity_from_frozen(
            args,
            libraries=frozen_libraries,
            binaries=frozen_binaries,
        )
        repeat_rows = [
            [[None] * args.process_repeats for _ in scheduled_cases]
            for _ in selected_libraries
        ]
        sample_args = argparse.Namespace(**vars(args))
        sample_args.process_repeats = 1
        for library_index, case_index, repeat in schedule:
            library_name, public_library = selected_libraries[library_index]
            case_type, case = scheduled_cases[case_index]
            sample_args.level1_probe = frozen_binaries[0].execution_path
            sample_args.copy_probe = frozen_binaries[1].execution_path
            private_controller = frozen_binaries[2].execution_path
            private_library = frozen_libraries[library_index].execution_path
            try:
                if case_type == "copy":
                    row = run_copy_op(
                        sample_args,
                        library_name,
                        public_library,
                        case,
                        execution_library_path=private_library,
                        controller_script_path=private_controller,
                        artifacts=artifacts,
                    )
                else:
                    group, op, variant, incx, incy = case
                    row = run_level1_op(
                        sample_args,
                        library_name,
                        public_library,
                        group,
                        op,
                        variant,
                        incx,
                        incy,
                        execution_library_path=private_library,
                        controller_script_path=private_controller,
                        artifacts=artifacts,
                    )
            except ValueError as exc:
                raise ValueError(
                    f"invalid Level 1 performance evidence: {exc}"
                ) from None
            if row["status"] == "ok":
                row["repeat"] = repeat
            repeat_rows[library_index][case_index][repeat] = row

        rows = []
        for library_index, (library_name, _library_path) in enumerate(
            selected_libraries
        ):
            for case_index, (case_type, case) in enumerate(scheduled_cases):
                try:
                    row = choose_best(repeat_rows[library_index][case_index])
                except ValueError as exc:
                    raise ValueError(
                        f"invalid Level 1 performance evidence: {exc}"
                    ) from None
                rows.append(row)
                if case_type == "copy":
                    size = format_byte_size(case["copy_bytes"])
                    print(
                        f"{library_name:10s} {case['op']:8s} "
                        f"{size:>6s}/{case['kind']:<1s} "
                        f"{row['status']:11s} {metric_value(row)}"
                    )
                else:
                    group, op, variant, incx, incy = case
                    case_label = op if variant == "default" else f"{op}:{variant}"
                    if incx == incy:
                        if incx != 1:
                            case_label += f":inc{incx}"
                    else:
                        case_label += f":incx{incx}:incy{incy}"
                    print(
                        f"{library_name:10s} {case_label:16s} "
                        f"{row['status']:11s} {metric_value(row)}"
                    )

        rows = artifacts.redact_private_paths(rows)
        binary_records = artifacts.legacy_records("binary")
        library_records = artifacts.legacy_records("library")
        csv_bytes, metadata_bytes = serialize_report(
            args,
            selected_libraries,
            rows,
            identity,
            binary_records,
            library_records,
        )
        csv_bytes, metadata_bytes = artifacts.redact_private_paths(
            (csv_bytes, metadata_bytes)
        )
        artifacts.finalize()
    except ValueError as exc:
        raise ValueError(artifacts.redact_private_paths(str(exc))) from None
    finally:
        artifacts.close()

    output_path = Path(args.csv)
    publish_outputs(
        [
            ReportOutput(output_path, csv_bytes),
            ReportOutput(
                output_path.with_suffix(output_path.suffix + ".meta.json"),
                metadata_bytes,
            ),
        ]
    )

    return 0


def main():
    if "--check-worker" in sys.argv:
        return check_worker_main()

    args = parse_args()
    if not Path(args.level1_probe).exists():
        print(f"missing --level1-probe {args.level1_probe}", file=sys.stderr)
        return 2
    if not Path(args.copy_probe).exists():
        print(f"missing --copy-probe {args.copy_probe}", file=sys.stderr)
        return 2

    try:
        configured_libraries = libraries(args)
    except ValueError as exc:
        print(exc, file=sys.stderr)
        return 2
    selected_libraries = []
    for library_name, library_path in configured_libraries:
        if args.skip_missing and not library_available(library_path):
            if library_name == "Zynum":
                print(
                    f"--skip-missing cannot skip required Zynum library {library_path}",
                    file=sys.stderr,
                )
                return 2
            continue
        selected_libraries.append((library_name, library_path))
    try:
        validate_unique_library_labels(selected_libraries)
        copy_case_list = selected_copy_cases(args)
        level1_ops = [
            (group, op, variant, incx, incy)
            for group, op, variant, incx, incy in level1_cases(args.stride_pairs)
            if case_allowed(args, group, op)
        ]
        scheduled_cases = [("copy", case) for case in copy_case_list]
        if not args.copy_only:
            scheduled_cases.extend(("level1", case) for case in level1_ops)
        validate_schedule(
            len(selected_libraries), args.process_repeats, args.process_schedule
        )
        schedule = (
            library_repeat_schedule(
                len(selected_libraries),
                args.process_repeats,
                args.process_schedule,
                case_count=len(scheduled_cases),
            )
            if scheduled_cases
            else []
        )
        return run_controller(args, selected_libraries, scheduled_cases, schedule)
    except (ArtifactSnapshotError, ValueError) as exc:
        print(exc, file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
