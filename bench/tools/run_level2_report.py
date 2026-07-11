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
from dataclasses import dataclass
from pathlib import Path

DEFAULT_ACCELERATE = "/System/Library/Frameworks/Accelerate.framework/Accelerate"
DEFAULT_OPENBLAS = "/opt/homebrew/opt/openblas/lib/libopenblas.dylib"

LIBS = [
    ("Zynum", "zynum"),
    ("Accelerate", "accelerate"),
    ("OpenBLAS", "openblas"),
]

DEFAULT_N = [128, 256, 512]
CHECKED_STATUSES = {"sampled-ok", "checked-ok"}

TRIANGULAR_OPERATIONS = (
    "strmv",
    "dtrmv",
    "ctrmv",
    "ztrmv",
    "strsv",
    "dtrsv",
    "ctrsv",
    "ztrsv",
)
RANK_UPDATE_OPERATIONS = (
    "ssyr",
    "dsyr",
    "cher",
    "zher",
    "ssyr2",
    "dsyr2",
    "cher2",
    "zher2",
)
BANDED_OPERATIONS = (
    "sgbmv",
    "dgbmv",
    "cgbmv",
    "zgbmv",
    "ssbmv",
    "dsbmv",
    "chbmv",
    "zhbmv",
)
PACKED_STRUCTURED_MV_OPERATIONS = ("sspmv", "dspmv", "chpmv", "zhpmv")
PACKED_TRIANGULAR_OPERATIONS = (
    "stpmv",
    "dtpmv",
    "ctpmv",
    "ztpmv",
    "stpsv",
    "dtpsv",
    "ctpsv",
    "ztpsv",
)
PACKED_MV_OPERATIONS = PACKED_STRUCTURED_MV_OPERATIONS + PACKED_TRIANGULAR_OPERATIONS
PACKED_RANK_OPERATIONS = (
    "sspr",
    "dspr",
    "chpr",
    "zhpr",
    "sspr2",
    "dspr2",
    "chpr2",
    "zhpr2",
)
TRIANGULAR_BANDED_OPERATIONS = (
    "stbmv",
    "dtbmv",
    "ctbmv",
    "ztbmv",
    "stbsv",
    "dtbsv",
    "ctbsv",
    "ztbsv",
)
PACKED_OPERATIONS = PACKED_MV_OPERATIONS + PACKED_RANK_OPERATIONS
COMPACT_BANDED_OPERATIONS = BANDED_OPERATIONS + TRIANGULAR_BANDED_OPERATIONS
OP_EXPANSIONS = {
    "legacy": ("legacy",),
    "trmv": ("strmv", "dtrmv", "ctrmv", "ztrmv"),
    "trsv": ("strsv", "dtrsv", "ctrsv", "ztrsv"),
    "strmv": ("strmv",),
    "dtrmv": ("dtrmv",),
    "ctrmv": ("ctrmv",),
    "ztrmv": ("ztrmv",),
    "strsv": ("strsv",),
    "dtrsv": ("dtrsv",),
    "ctrsv": ("ctrsv",),
    "ztrsv": ("ztrsv",),
    "triangular": TRIANGULAR_OPERATIONS,
    "ssyr": ("ssyr",),
    "dsyr": ("dsyr",),
    "cher": ("cher",),
    "zher": ("zher",),
    "ssyr2": ("ssyr2",),
    "dsyr2": ("dsyr2",),
    "cher2": ("cher2",),
    "zher2": ("zher2",),
    "rank-update": RANK_UPDATE_OPERATIONS,
    **{operation: (operation,) for operation in BANDED_OPERATIONS},
    "banded": BANDED_OPERATIONS,
    **{operation: (operation,) for operation in PACKED_OPERATIONS},
    "spmv": ("sspmv", "dspmv"),
    "hpmv": ("chpmv", "zhpmv"),
    "tpmv": ("stpmv", "dtpmv", "ctpmv", "ztpmv"),
    "tpsv": ("stpsv", "dtpsv", "ctpsv", "ztpsv"),
    "packed-mv": PACKED_MV_OPERATIONS,
    "packed-rank": PACKED_RANK_OPERATIONS,
    **{operation: (operation,) for operation in TRIANGULAR_BANDED_OPERATIONS},
    "tbmv": ("stbmv", "dtbmv", "ctbmv", "ztbmv"),
    "tbsv": ("stbsv", "dtbsv", "ctbsv", "ztbsv"),
    "triangular-banded": TRIANGULAR_BANDED_OPERATIONS,
    "compact": PACKED_OPERATIONS + TRIANGULAR_BANDED_OPERATIONS,
    "all": (
        ("legacy",)
        + TRIANGULAR_OPERATIONS
        + RANK_UPDATE_OPERATIONS
        + BANDED_OPERATIONS
    ),
}

CSV_FIELDNAMES = [
    "level",
    "case",
    "kind",
    "library",
    "n",
    "time_ns",
    "rate_gops",
    "metric",
    "status",
    "check_status",
    "check_max_abs_error",
    "check_raw_output",
    "shape",
    "m",
    "storage",
    "lda",
    "k",
    "kl",
    "ku",
    "uplo",
    "trans",
    "diag",
    "incx",
    "incy",
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
class BandedProfile:
    name: str
    n: int
    bandwidth: int


@dataclass(frozen=True)
class PackedProfile:
    name: str
    n: int


@dataclass(frozen=True)
class TriangularCase:
    case: str
    kind: str
    ctype: type
    uplo: str
    trans: str
    diag: str
    incx: int = 1


@dataclass(frozen=True)
class RankUpdateCase:
    case: str
    kind: str
    value_type: type
    real_type: type
    uplo: str
    rank2: bool
    hermitian: bool
    incx: int = 1
    incy: int = 1


@dataclass(frozen=True)
class BandedCase:
    case: str
    kind: str
    value_type: type
    storage: str
    lda: int
    trans: str = ""
    uplo: str = ""
    k: object = ""
    kl: object = ""
    ku: object = ""
    incx: int = 1
    incy: int = 1


@dataclass(frozen=True)
class PackedStructuredMvCase:
    case: str
    kind: str
    value_type: type
    uplo: str
    hermitian: bool
    storage: str
    incx: int = 1
    incy: int = 1


@dataclass(frozen=True)
class CompactTriangularCase:
    case: str
    kind: str
    value_type: type
    storage: str
    uplo: str
    trans: str
    diag: str
    solve: bool
    lda: object = ""
    k: object = ""
    incx: int = 1


@dataclass(frozen=True)
class PackedRankCase:
    case: str
    kind: str
    value_type: type
    real_type: type
    uplo: str
    rank2: bool
    hermitian: bool
    storage: str
    incx: int = 1
    incy: int = 1


class ComplexF32(ctypes.Structure):
    _fields_ = [("re", ctypes.c_float), ("im", ctypes.c_float)]


class ComplexF64(ctypes.Structure):
    _fields_ = [("re", ctypes.c_double), ("im", ctypes.c_double)]


DEFAULT_BANDED_PROFILES = (
    BandedProfile("n512_bw8", 512, 8),
    BandedProfile("n2048_bw64", 2048, 64),
)
DEFAULT_PACKED_PROFILES = (
    PackedProfile("n128", 128),
    PackedProfile("n512", 512),
    PackedProfile("n2048", 2048),
)
DEFAULT_TRIANGULAR_BANDED_PROFILES = (
    BandedProfile("n128_bw8", 128, 8),
    BandedProfile("n512_bw8", 512, 8),
    BandedProfile("n2048_bw64", 2048, 64),
)


def default_zynum_blas():
    if sys.platform == "darwin":
        return "zig-out/lib/libzynum_blas.dylib"
    if sys.platform == "win32":
        return "zig-out/bin/zynum_blas.dll"
    return "zig-out/lib/libzynum_blas.so"


def parse_shape_spec(spec):
    parts = spec.split(":")
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(f"shape must be NAME:M:N, got {spec!r}")
    name = parts[0].strip()
    if not name:
        raise argparse.ArgumentTypeError("shape name must not be empty")
    try:
        m, n = (int(value, 10) for value in parts[1:])
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"shape dimensions must be integers, got {spec!r}"
        ) from exc
    if m < 1 or n < 1:
        raise argparse.ArgumentTypeError(
            f"shape dimensions must be positive, got {spec!r}"
        )
    return Shape(name, m, n)


def parse_banded_profile_spec(spec):
    parts = spec.split(":")
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(
            f"band profile must be NAME:N:BANDWIDTH, got {spec!r}"
        )
    name = parts[0].strip()
    if not name:
        raise argparse.ArgumentTypeError("band profile name must not be empty")
    try:
        n, bandwidth = (int(value, 10) for value in parts[1:])
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"band profile dimensions must be integers, got {spec!r}"
        ) from exc
    if n < 1 or bandwidth < 0 or bandwidth >= n:
        raise argparse.ArgumentTypeError(
            "band profile requires N >= 1 and 0 <= BANDWIDTH < N, "
            f"got {spec!r}"
        )
    return BandedProfile(name, n, bandwidth)


def parse_packed_profile_spec(spec):
    parts = spec.split(":")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError(
            f"packed profile must be NAME:N, got {spec!r}"
        )
    name = parts[0].strip()
    if not name:
        raise argparse.ArgumentTypeError("packed profile name must not be empty")
    try:
        n = int(parts[1], 10)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"packed profile dimension must be an integer, got {spec!r}"
        ) from exc
    if n < 1:
        raise argparse.ArgumentTypeError(
            f"packed profile dimension must be positive, got {spec!r}"
        )
    return PackedProfile(name, n)


def parse_op(value):
    op = value.strip().lower()
    if op not in OP_EXPANSIONS:
        choices = ", ".join(OP_EXPANSIONS)
        raise argparse.ArgumentTypeError(f"unknown operation {value!r}; choose from {choices}")
    return op


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Run representative Level 2 fresh-process probes and write a report CSV."
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
    parser.add_argument(
        "--n",
        action="append",
        type=int,
        default=[],
        help="Square matrix dimension. May be repeated.",
    )
    parser.add_argument(
        "--shape",
        action="append",
        type=parse_shape_spec,
        default=[],
        metavar="NAME:M:N",
        help="Matrix shape label, row count, and column count. May be repeated.",
    )
    parser.add_argument(
        "--band-profile",
        action="append",
        type=parse_banded_profile_spec,
        default=[],
        metavar="NAME:N:BANDWIDTH",
        help=(
            "Square compact-band profile. Defaults to n512_bw8 and n2048_bw64 "
            "when --op banded is selected. May be repeated."
        ),
    )
    parser.add_argument(
        "--packed-profile",
        action="append",
        type=parse_packed_profile_spec,
        default=[],
        metavar="NAME:N",
        help=(
            "Square packed-storage profile. Defaults to n128, n512, and n2048 "
            "when a packed operation is selected. May be repeated."
        ),
    )
    parser.add_argument(
        "--op",
        action="append",
        type=parse_op,
        default=[],
        help=(
            "Operation selection. Use trmv/trsv for all four scalar kinds, "
            "triangular for all 80 dense triangular cases, legacy for the "
            "historical cases, rank-update for all 16 dense SYR/HER/SYR2/HER2 "
            "cases, banded for GBMV/SBMV/HBMV, packed-mv for SPMV/HPMV and "
            "TPMV/TPSV, packed-rank for SPR/HPR/SPR2/HPR2, "
            "triangular-banded for TBMV/TBSV, or all for the historical set. "
            "May be repeated."
        ),
    )
    parser.add_argument("--reps-small", type=int, default=260)
    parser.add_argument("--reps-large", type=int, default=130)
    parser.add_argument(
        "--process-repeats",
        type=int,
        default=1,
        help="Run each library/shape in this many independent worker processes.",
    )
    parser.add_argument("--csv", required=True)
    parser.add_argument("--skip-missing", action="store_true")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--library-name", help=argparse.SUPPRESS)
    parser.add_argument("--library-path", help=argparse.SUPPRESS)
    parser.add_argument("--worker-shape", help=argparse.SUPPRESS)
    parser.add_argument("--worker-m", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--worker-n", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--worker-bandwidth", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--worker-reps", type=int, help=argparse.SUPPRESS)
    parser.add_argument(
        "--worker-op",
        action="append",
        type=parse_op,
        default=[],
        help=argparse.SUPPRESS,
    )
    args = parser.parse_args(argv)
    if args.reps_small < 1 or args.reps_large < 1:
        parser.error("repetition counts must be at least 1")
    if args.process_repeats < 1:
        parser.error("--process-repeats must be at least 1")
    if any(n < 1 for n in args.n):
        parser.error("--n values must be positive")
    return args


def requested_shapes(args):
    if not args.n and not args.shape:
        return [Shape(f"sq{n}", n, n) for n in DEFAULT_N]
    return [Shape(f"sq{n}", n, n) for n in args.n] + list(args.shape)


def requested_banded_profiles(args):
    return list(args.band_profile or DEFAULT_BANDED_PROFILES)


def requested_packed_profiles(args):
    if args.packed_profile:
        return list(args.packed_profile)
    if args.n or args.shape:
        return [
            PackedProfile(shape.name, shape.n)
            for shape in requested_shapes(args)
            if shape.m == shape.n
        ]
    return list(DEFAULT_PACKED_PROFILES)


def requested_triangular_banded_profiles(args):
    return list(args.band_profile or DEFAULT_TRIANGULAR_BANDED_PROFILES)


def unique_preserving_order(values):
    return list(dict.fromkeys(values))


def expand_operations(selectors):
    selected = selectors or ["legacy"]
    return unique_preserving_order(
        operation for selector in selected for operation in OP_EXPANSIONS[selector]
    )


def requested_operations(args):
    return expand_operations(args.op)


def triangular_cases(operations):
    cases = []
    specs = {
        "strmv": ("f32", ctypes.c_float),
        "dtrmv": ("f64", ctypes.c_double),
        "ctrmv": ("c32", ComplexF32),
        "ztrmv": ("c64", ComplexF64),
        "strsv": ("f32", ctypes.c_float),
        "dtrsv": ("f64", ctypes.c_double),
        "ctrsv": ("c32", ComplexF32),
        "ztrsv": ("c64", ComplexF64),
    }
    for operation in operations:
        if operation not in specs:
            continue
        kind, ctype = specs[operation]
        transposes = ("N", "T", "C") if kind in ("c32", "c64") else ("N", "T")
        for uplo in ("U", "L"):
            for trans in transposes:
                for diag in ("N", "U"):
                    cases.append(
                        TriangularCase(operation, kind, ctype, uplo, trans, diag)
                    )
    return cases


def rank_update_cases(operations):
    cases = []
    specs = {
        "ssyr": ("f32", ctypes.c_float, ctypes.c_float, False, False),
        "dsyr": ("f64", ctypes.c_double, ctypes.c_double, False, False),
        "cher": ("c32", ComplexF32, ctypes.c_float, False, True),
        "zher": ("c64", ComplexF64, ctypes.c_double, False, True),
        "ssyr2": ("f32", ctypes.c_float, ctypes.c_float, True, False),
        "dsyr2": ("f64", ctypes.c_double, ctypes.c_double, True, False),
        "cher2": ("c32", ComplexF32, ctypes.c_float, True, True),
        "zher2": ("c64", ComplexF64, ctypes.c_double, True, True),
    }
    for operation in operations:
        spec = specs.get(operation)
        if spec is None:
            continue
        kind, value_type, real_type, rank2, hermitian = spec
        for uplo in ("U", "L"):
            cases.append(
                RankUpdateCase(
                    operation,
                    kind,
                    value_type,
                    real_type,
                    uplo,
                    rank2,
                    hermitian,
                )
            )
    return cases


def banded_cases(operations, bandwidth):
    cases = []
    specs = {
        "sgbmv": ("f32", ctypes.c_float, ("N", "T")),
        "dgbmv": ("f64", ctypes.c_double, ("N", "T")),
        "cgbmv": ("c32", ComplexF32, ("N", "T", "C")),
        "zgbmv": ("c64", ComplexF64, ("N", "T", "C")),
    }
    structured_specs = {
        "ssbmv": ("f32", ctypes.c_float, "symmetric-band"),
        "dsbmv": ("f64", ctypes.c_double, "symmetric-band"),
        "chbmv": ("c32", ComplexF32, "hermitian-band"),
        "zhbmv": ("c64", ComplexF64, "hermitian-band"),
    }
    for operation in operations:
        general = specs.get(operation)
        if general is not None:
            kind, value_type, transposes = general
            for trans in transposes:
                cases.append(
                    BandedCase(
                        operation,
                        kind,
                        value_type,
                        "general-band",
                        2 * bandwidth + 1,
                        trans=trans,
                        kl=bandwidth,
                        ku=bandwidth,
                    )
                )
            continue
        structured = structured_specs.get(operation)
        if structured is not None:
            kind, value_type, storage = structured
            for uplo in ("U", "L"):
                cases.append(
                    BandedCase(
                        operation,
                        kind,
                        value_type,
                        storage,
                        bandwidth + 1,
                        uplo=uplo,
                        k=bandwidth,
                    )
                )
    return cases


def packed_structured_mv_cases(operations):
    cases = []
    specs = {
        "sspmv": ("f32", ctypes.c_float, False),
        "dspmv": ("f64", ctypes.c_double, False),
        "chpmv": ("c32", ComplexF32, True),
        "zhpmv": ("c64", ComplexF64, True),
    }
    for operation in operations:
        spec = specs.get(operation)
        if spec is None:
            continue
        kind, value_type, hermitian = spec
        storage = "hermitian-packed" if hermitian else "symmetric-packed"
        for uplo in ("U", "L"):
            cases.append(
                PackedStructuredMvCase(
                    operation, kind, value_type, uplo, hermitian, storage
                )
            )
    return cases


def packed_triangular_cases(operations):
    cases = []
    specs = {
        "stpmv": ("f32", ctypes.c_float, False),
        "dtpmv": ("f64", ctypes.c_double, False),
        "ctpmv": ("c32", ComplexF32, False),
        "ztpmv": ("c64", ComplexF64, False),
        "stpsv": ("f32", ctypes.c_float, True),
        "dtpsv": ("f64", ctypes.c_double, True),
        "ctpsv": ("c32", ComplexF32, True),
        "ztpsv": ("c64", ComplexF64, True),
    }
    for operation in operations:
        spec = specs.get(operation)
        if spec is None:
            continue
        kind, value_type, solve = spec
        transposes = ("N", "T", "C") if kind in ("c32", "c64") else ("N", "T")
        for uplo in ("U", "L"):
            for trans in transposes:
                for diag in ("N", "U"):
                    cases.append(
                        CompactTriangularCase(
                            operation,
                            kind,
                            value_type,
                            "triangular-packed",
                            uplo,
                            trans,
                            diag,
                            solve,
                        )
                    )
    return cases


def packed_rank_cases(operations):
    cases = []
    specs = {
        "sspr": ("f32", ctypes.c_float, ctypes.c_float, False, False),
        "dspr": ("f64", ctypes.c_double, ctypes.c_double, False, False),
        "chpr": ("c32", ComplexF32, ctypes.c_float, False, True),
        "zhpr": ("c64", ComplexF64, ctypes.c_double, False, True),
        "sspr2": ("f32", ctypes.c_float, ctypes.c_float, True, False),
        "dspr2": ("f64", ctypes.c_double, ctypes.c_double, True, False),
        "chpr2": ("c32", ComplexF32, ctypes.c_float, True, True),
        "zhpr2": ("c64", ComplexF64, ctypes.c_double, True, True),
    }
    for operation in operations:
        spec = specs.get(operation)
        if spec is None:
            continue
        kind, value_type, real_type, rank2, hermitian = spec
        storage = "hermitian-packed" if hermitian else "symmetric-packed"
        for uplo in ("U", "L"):
            cases.append(
                PackedRankCase(
                    operation,
                    kind,
                    value_type,
                    real_type,
                    uplo,
                    rank2,
                    hermitian,
                    storage,
                )
            )
    return cases


def triangular_banded_cases(operations, bandwidth):
    cases = []
    specs = {
        "stbmv": ("f32", ctypes.c_float, False),
        "dtbmv": ("f64", ctypes.c_double, False),
        "ctbmv": ("c32", ComplexF32, False),
        "ztbmv": ("c64", ComplexF64, False),
        "stbsv": ("f32", ctypes.c_float, True),
        "dtbsv": ("f64", ctypes.c_double, True),
        "ctbsv": ("c32", ComplexF32, True),
        "ztbsv": ("c64", ComplexF64, True),
    }
    for operation in operations:
        spec = specs.get(operation)
        if spec is None:
            continue
        kind, value_type, solve = spec
        transposes = ("N", "T", "C") if kind in ("c32", "c64") else ("N", "T")
        for uplo in ("U", "L"):
            for trans in transposes:
                for diag in ("N", "U"):
                    cases.append(
                        CompactTriangularCase(
                            operation,
                            kind,
                            value_type,
                            "triangular-band",
                            uplo,
                            trans,
                            diag,
                            solve,
                            lda=bandwidth + 1,
                            k=bandwidth,
                        )
                    )
    return cases


def operations_for_shape(operations, shape):
    if shape.m == shape.n:
        return operations
    return [operation for operation in operations if operation == "legacy"]


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


def library_available(path):
    if Path(path).exists():
        return True
    try:
        ctypes.CDLL(path)
        return True
    except OSError:
        return False


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


def as_complex(value):
    return complex(float(value.re), float(value.im))


def write_complex(target, value):
    target.re = value.real
    target.im = value.imag


def max_real_error(actual, expected):
    return max((abs(float(actual[i]) - expected[i]) for i in range(len(expected))), default=0.0)


def max_complex_error(actual, expected):
    return max((abs(as_complex(actual[i]) - expected[i]) for i in range(len(expected))), default=0.0)


def check_result(status, error=0.0, raw=""):
    return {
        "check_status": status,
        "check_max_abs_error": f"{error:.9g}",
        "check_raw_output": raw,
    }


def tolerance(kind, n):
    if kind in ("f32", "c32"):
        return 1e-3 * max(1, n)
    return 1e-10 * max(1, n)


def real_gemv_expected(matrix, x, y0, m, n, lda, alpha, beta, trans):
    out = []
    output_length = m if trans == "N" else n
    inner_length = n if trans == "N" else m
    for output_index in range(output_length):
        total = 0.0
        for inner_index in range(inner_length):
            matrix_index = (
                output_index + inner_index * lda
                if trans == "N"
                else inner_index + output_index * lda
            )
            total += float(matrix[matrix_index]) * float(x[inner_index])
        out.append(float(alpha) * total + float(beta) * float(y0[output_index]))
    return out


def real_symv_expected(matrix, x, y0, n, alpha, beta):
    out = []
    for row in range(n):
        total = 0.0
        for col in range(n):
            a = matrix[row + col * n] if row <= col else matrix[col + row * n]
            total += float(a) * float(x[col])
        out.append(float(alpha) * total + float(beta) * float(y0[row]))
    return out


def real_triangular_mv_expected(matrix, x, n, lda, uplo, trans, diag):
    out = []
    for output_index in range(n):
        total = 0.0
        for input_index in range(n):
            row, col = (
                (output_index, input_index)
                if trans == "N"
                else (input_index, output_index)
            )
            if (uplo == "U" and row > col) or (uplo == "L" and row < col):
                continue
            value = 1.0 if diag == "U" and row == col else float(matrix[row + col * lda])
            total += value * float(x[input_index])
        out.append(total)
    return out


def complex_triangular_mv_expected(matrix, x, n, lda, uplo, trans, diag):
    out = []
    for output_index in range(n):
        total = 0j
        for input_index in range(n):
            row, col = (
                (output_index, input_index)
                if trans == "N"
                else (input_index, output_index)
            )
            if (uplo == "U" and row > col) or (uplo == "L" and row < col):
                continue
            value = (
                1 + 0j
                if diag == "U" and row == col
                else as_complex(matrix[row + col * lda])
            )
            if trans == "C":
                value = value.conjugate()
            total += value * as_complex(x[input_index])
        out.append(total)
    return out


def safe_triangular_matrix(
    value_type, n, uplo, seed_value, complex_values=False
):
    if not complex_values:
        matrix = real_array(value_type, n * n, seed_value)
        off_diagonal_scale = 0.25 / max(1, n - 1)
        for col in range(n):
            for row in range(n):
                index = row + col * n
                if row == col:
                    matrix[index] = value_type(
                        1.5 + 0.25 * abs(float(matrix[index]))
                    )
                elif (uplo == "U" and row < col) or (
                    uplo == "L" and row > col
                ):
                    matrix[index] = value_type(
                        float(matrix[index]) * off_diagonal_scale
                    )
                else:
                    matrix[index] = value_type(
                        2.0 + abs(float(matrix[index]))
                    )
        return matrix

    matrix = complex_array(value_type, n * n, seed_value)
    off_diagonal_scale = 0.25 / max(1, n - 1)
    for col in range(n):
        for row in range(n):
            index = row + col * n
            value = as_complex(matrix[index])
            if row == col:
                matrix[index] = value_type(
                    1.5 + 0.25 * abs(value.real), 0.125 * value.imag
                )
            elif (uplo == "U" and row < col) or (uplo == "L" and row > col):
                scaled = value * off_diagonal_scale
                matrix[index] = value_type(scaled.real, scaled.imag)
            else:
                matrix[index] = value_type(
                    2.0 + abs(value.real), 1.5 + abs(value.imag)
                )
    return matrix


def real_ger_expected(matrix0, x, y, m, n, lda, alpha):
    out = [float(matrix0[i]) for i in range(lda * n)]
    for col in range(n):
        for row in range(m):
            out[row + col * lda] += float(alpha) * float(x[row]) * float(y[col])
    return out


def real_rank_update_expected(matrix0, x, y, n, lda, alpha, uplo):
    out = [float(matrix0[i]) for i in range(lda * n)]
    for col in range(n):
        row_range = range(col + 1) if uplo == "U" else range(col, n)
        for row in row_range:
            update = float(alpha) * float(x[row]) * float(x[col])
            if y is not None:
                update = float(alpha) * (
                    float(x[row]) * float(y[col])
                    + float(y[row]) * float(x[col])
                )
            out[row + col * lda] += update
    return out


def complex_gemv_expected(matrix, x, y0, m, n, lda, alpha, beta, trans):
    alpha = as_complex(alpha)
    beta = as_complex(beta)
    out = []
    output_length = m if trans == "N" else n
    inner_length = n if trans == "N" else m
    for output_index in range(output_length):
        total = 0j
        for inner_index in range(inner_length):
            matrix_index = (
                output_index + inner_index * lda
                if trans == "N"
                else inner_index + output_index * lda
            )
            value = as_complex(matrix[matrix_index])
            if trans == "C":
                value = value.conjugate()
            total += value * as_complex(x[inner_index])
        out.append(alpha * total + beta * as_complex(y0[output_index]))
    return out


def complex_hemv_expected(matrix, x, y0, n, alpha, beta):
    alpha = as_complex(alpha)
    beta = as_complex(beta)
    out = []
    for row in range(n):
        total = 0j
        for col in range(n):
            if row < col:
                a = as_complex(matrix[row + col * n])
            elif row == col:
                a = complex(float(matrix[row + col * n].re), 0.0)
            else:
                a = as_complex(matrix[col + row * n]).conjugate()
            total += a * as_complex(x[col])
        out.append(alpha * total + beta * as_complex(y0[row]))
    return out


def complex_ger_expected(matrix0, x, y, m, n, lda, alpha, conjugate_y):
    alpha = as_complex(alpha)
    out = [as_complex(matrix0[i]) for i in range(lda * n)]
    for col in range(n):
        yv = as_complex(y[col]).conjugate() if conjugate_y else as_complex(y[col])
        for row in range(m):
            out[row + col * lda] += alpha * as_complex(x[row]) * yv
    return out


def complex_rank_update_expected(matrix0, x, y, n, lda, alpha, uplo):
    out = [as_complex(matrix0[i]) for i in range(lda * n)]
    for col in range(n):
        row_range = range(col + 1) if uplo == "U" else range(col, n)
        for row in row_range:
            if y is None:
                update = float(alpha) * as_complex(x[row]) * as_complex(x[col]).conjugate()
            else:
                complex_alpha = as_complex(alpha)
                update = (
                    complex_alpha
                    * as_complex(x[row])
                    * as_complex(y[col]).conjugate()
                    + complex_alpha.conjugate()
                    * as_complex(y[row])
                    * as_complex(x[col]).conjugate()
                )
            value = out[row + col * lda] + update
            out[row + col * lda] = complex(value.real, 0.0) if row == col else value
    return out


def scalar_value(value, complex_values):
    if complex_values:
        return as_complex(value)
    return float(getattr(value, "value", value))


def array_from_values(value_type, values, complex_values):
    out = (value_type * len(values))()
    for index, value in enumerate(values):
        if complex_values:
            out[index] = value_type(value.real, value.imag)
        else:
            out[index] = value_type(value)
    return out


def packed_index(n, uplo, row, col):
    if uplo == "U":
        return col * (col + 1) // 2 + row
    return col * n - col * (col - 1) // 2 + row - col


def packed_structured_value(matrix, n, uplo, row, col, hermitian):
    if uplo == "U":
        direct = row <= col
    else:
        direct = row >= col
    stored_row, stored_col = (row, col) if direct else (col, row)
    value = scalar_value(
        matrix[packed_index(n, uplo, stored_row, stored_col)], hermitian
    )
    if hermitian:
        if row == col:
            return complex(value.real, 0.0)
        if not direct:
            return value.conjugate()
    return value


def packed_structured_mv_expected(
    matrix, x, y0, n, alpha, beta, uplo, hermitian=False
):
    alpha_value = scalar_value(alpha, hermitian)
    beta_value = scalar_value(beta, hermitian)
    out = []
    for row in range(n):
        total = 0j if hermitian else 0.0
        for col in range(n):
            total += packed_structured_value(
                matrix, n, uplo, row, col, hermitian
            ) * scalar_value(x[col], hermitian)
        out.append(
            alpha_value * total + beta_value * scalar_value(y0[row], hermitian)
        )
    return out


def safe_packed_triangular_matrix(
    value_type, n, uplo, seed_value, complex_values=False
):
    count = n * (n + 1) // 2
    matrix = (value_type * count)()
    seed = [seed_value]
    off_diagonal_scale = 0.25 / max(1, n - 1)
    for col in range(n):
        rows = range(col + 1) if uplo == "U" else range(col, n)
        for row in rows:
            index = packed_index(n, uplo, row, col)
            real = next_fill(seed)
            imag = next_fill(seed) if complex_values else 0.0
            if row == col:
                real = 1.5 + 0.25 * abs(real)
                imag *= 0.125
            else:
                real *= off_diagonal_scale
                imag *= off_diagonal_scale
            if complex_values:
                matrix[index] = value_type(real, imag)
            else:
                matrix[index] = value_type(real)
    return matrix


def triangular_packed_value(
    matrix, n, uplo, row, col, diag, complex_values=False
):
    zero = 0j if complex_values else 0.0
    if (uplo == "U" and row > col) or (uplo == "L" and row < col):
        return zero
    if row == col and diag == "U":
        return 1 + 0j if complex_values else 1.0
    return scalar_value(matrix[packed_index(n, uplo, row, col)], complex_values)


def triangular_packed_mv_expected(
    matrix, x, n, uplo, trans, diag, complex_values=False
):
    out = []
    for output_index in range(n):
        total = 0j if complex_values else 0.0
        for input_index in range(n):
            row, col = (
                (output_index, input_index)
                if trans == "N"
                else (input_index, output_index)
            )
            value = triangular_packed_value(
                matrix, n, uplo, row, col, diag, complex_values
            )
            if trans == "C":
                value = value.conjugate()
            total += value * scalar_value(x[input_index], complex_values)
        out.append(total)
    return out


def safe_triangular_band_matrix(
    value_type, n, bandwidth, uplo, seed_value, complex_values=False
):
    lda = bandwidth + 1
    matrix = (
        complex_array(value_type, lda * n, seed_value)
        if complex_values
        else real_array(value_type, lda * n, seed_value)
    )
    seed = [seed_value ^ 0x9E3779B97F4A7C15]
    off_diagonal_scale = 0.25 / max(1, bandwidth)
    for col in range(n):
        row0 = max(0, col - bandwidth) if uplo == "U" else col
        row1 = col + 1 if uplo == "U" else min(n, col + bandwidth + 1)
        for row in range(row0, row1):
            band_row = bandwidth + row - col if uplo == "U" else row - col
            index = band_row + col * lda
            real = next_fill(seed)
            imag = next_fill(seed) if complex_values else 0.0
            if row == col:
                real = 1.5 + 0.25 * abs(real)
                imag *= 0.125
            else:
                real *= off_diagonal_scale
                imag *= off_diagonal_scale
            if complex_values:
                matrix[index] = value_type(real, imag)
            else:
                matrix[index] = value_type(real)
    return matrix


def triangular_band_value(
    matrix,
    n,
    lda,
    bandwidth,
    uplo,
    row,
    col,
    diag,
    complex_values=False,
):
    zero = 0j if complex_values else 0.0
    if abs(row - col) > bandwidth:
        return zero
    if (uplo == "U" and row > col) or (uplo == "L" and row < col):
        return zero
    if row == col and diag == "U":
        return 1 + 0j if complex_values else 1.0
    band_row = bandwidth + row - col if uplo == "U" else row - col
    return scalar_value(matrix[band_row + col * lda], complex_values)


def triangular_band_mv_expected(
    matrix,
    x,
    n,
    lda,
    bandwidth,
    uplo,
    trans,
    diag,
    complex_values=False,
):
    out = []
    for output_index in range(n):
        total = 0j if complex_values else 0.0
        input0 = max(0, output_index - bandwidth)
        input1 = min(n, output_index + bandwidth + 1)
        for input_index in range(input0, input1):
            row, col = (
                (output_index, input_index)
                if trans == "N"
                else (input_index, output_index)
            )
            value = triangular_band_value(
                matrix,
                n,
                lda,
                bandwidth,
                uplo,
                row,
                col,
                diag,
                complex_values,
            )
            if trans == "C":
                value = value.conjugate()
            total += value * scalar_value(x[input_index], complex_values)
        out.append(total)
    return out


def packed_rank_expected(
    matrix0, x, y, n, alpha, uplo, hermitian=False
):
    out = [scalar_value(matrix0[index], hermitian) for index in range(len(matrix0))]
    if hermitian and y is not None:
        alpha_value = as_complex(alpha)
    else:
        alpha_value = float(getattr(alpha, "value", alpha))
    for col in range(n):
        rows = range(col + 1) if uplo == "U" else range(col, n)
        for row in rows:
            x_row = scalar_value(x[row], hermitian)
            x_col = scalar_value(x[col], hermitian)
            if hermitian:
                if y is None:
                    update = alpha_value * x_row * x_col.conjugate()
                else:
                    y_row = scalar_value(y[row], True)
                    y_col = scalar_value(y[col], True)
                    update = (
                        alpha_value * x_row * y_col.conjugate()
                        + alpha_value.conjugate() * y_row * x_col.conjugate()
                    )
            elif y is None:
                update = alpha_value * x_row * x_col
            else:
                update = alpha_value * (
                    x_row * scalar_value(y[col], False)
                    + scalar_value(y[row], False) * x_col
                )
            index = packed_index(n, uplo, row, col)
            value = out[index] + update
            out[index] = (
                complex(value.real, 0.0)
                if hermitian and row == col
                else value
            )
    return out


def general_band_expected(
    matrix,
    x,
    y0,
    m,
    n,
    lda,
    kl,
    ku,
    alpha,
    beta,
    trans,
    complex_values=False,
):
    alpha_value = scalar_value(alpha, complex_values)
    beta_value = scalar_value(beta, complex_values)
    output_length = m if trans == "N" else n
    inner_length = n if trans == "N" else m
    out = []
    for output_index in range(output_length):
        total = 0j if complex_values else 0.0
        for inner_index in range(inner_length):
            row, col = (
                (output_index, inner_index)
                if trans == "N"
                else (inner_index, output_index)
            )
            if row < col - ku or row > col + kl:
                continue
            value = scalar_value(matrix[ku + row - col + col * lda], complex_values)
            if trans == "C":
                value = value.conjugate()
            total += value * scalar_value(x[inner_index], complex_values)
        out.append(
            alpha_value * total
            + beta_value * scalar_value(y0[output_index], complex_values)
        )
    return out


def structured_band_value(matrix, n, lda, k, uplo, row, col, hermitian):
    if abs(row - col) > k:
        return 0j if hermitian else 0.0
    if uplo == "U":
        direct = row <= col
        stored_row, stored_col = (row, col) if direct else (col, row)
        index = k + stored_row - stored_col + stored_col * lda
    else:
        direct = row >= col
        stored_row, stored_col = (row, col) if direct else (col, row)
        index = stored_row - stored_col + stored_col * lda
    value = scalar_value(matrix[index], hermitian)
    if hermitian:
        if row == col:
            return complex(value.real, 0.0)
        if not direct:
            return value.conjugate()
    return value


def structured_band_expected(
    matrix,
    x,
    y0,
    n,
    lda,
    k,
    alpha,
    beta,
    uplo,
    hermitian=False,
):
    alpha_value = scalar_value(alpha, hermitian)
    beta_value = scalar_value(beta, hermitian)
    out = []
    for row in range(n):
        total = 0j if hermitian else 0.0
        for col in range(max(0, row - k), min(n, row + k + 1)):
            value = structured_band_value(
                matrix, n, lda, k, uplo, row, col, hermitian
            )
            total += value * scalar_value(x[col], hermitian)
        out.append(
            alpha_value * total + beta_value * scalar_value(y0[row], hermitian)
        )
    return out


def checked_vector(
    call,
    setup,
    actual,
    expected,
    kind,
    n,
    complex_values=False,
    tolerance_limit=None,
):
    setup()
    call()
    error = max_complex_error(actual, expected) if complex_values else max_real_error(actual, expected)
    limit = tolerance(kind, n) if tolerance_limit is None else tolerance_limit
    if error > limit:
        return check_result("correctness_failed", error, f"max_abs_error={error} tolerance={limit}")
    return check_result("sampled-ok", error)


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


def emit(
    rows,
    case,
    kind,
    library_name,
    shape,
    elapsed_ns,
    work,
    check,
    parameters=None,
):
    rate = work / (elapsed_ns / 1e9) / 1e9 if elapsed_ns > 0 else 0.0
    rows.append(
        {
            "level": "level2",
            "case": case,
            "kind": kind,
            "library": library_name,
            "n": shape.n,
            "time_ns": elapsed_ns,
            "rate_gops": f"{rate:.6f}",
            "metric": "gops",
            "status": "ok"
            if check["check_status"] == "sampled-ok"
            else "correctness_failed",
            **check,
            "shape": shape.name,
            "m": shape.m,
            "storage": getattr(parameters, "storage", ""),
            "lda": getattr(parameters, "lda", ""),
            "k": getattr(parameters, "k", ""),
            "kl": getattr(parameters, "kl", ""),
            "ku": getattr(parameters, "ku", ""),
            "uplo": getattr(parameters, "uplo", ""),
            "trans": getattr(parameters, "trans", ""),
            "diag": getattr(parameters, "diag", ""),
            "incx": getattr(parameters, "incx", ""),
            "incy": getattr(parameters, "incy", ""),
        }
    )


def triangular_tolerance(kind, n):
    size_scale = max(1.0, n / 128.0)
    limits = {
        "f32": 5e-5,
        "f64": 1e-12,
        "c32": 1e-4,
        "c64": 2e-12,
    }
    return limits[kind] * size_scale


def triangular_work(case, n):
    return (4 if case.kind in ("c32", "c64") else 1) * n * n


def run_triangular_cases(lib, library_name, shape, reps, operations):
    if shape.m != shape.n:
        return []
    n = shape.n
    blas_int = ctypes.c_int
    ni = blas_int(n)
    ldai = blas_int(n)
    one = blas_int(1)
    rows = []
    matrices = {}
    initial_vectors = {}

    for case_index, case in enumerate(triangular_cases(operations)):
        complex_values = case.kind in ("c32", "c64")
        matrix_key = (case.kind, case.uplo)
        if matrix_key not in matrices:
            matrices[matrix_key] = safe_triangular_matrix(
                case.ctype,
                n,
                case.uplo,
                0x4D595DF4D0F33173 ^ (case_index << 8),
                complex_values=complex_values,
            )
        if case.kind not in initial_vectors:
            array_factory = complex_array if complex_values else real_array
            initial_vectors[case.kind] = array_factory(
                case.ctype, n, 0x2718281828459045 ^ case_index
            )

        matrix = matrices[matrix_key]
        solution = initial_vectors[case.kind]
        reference = (
            complex_triangular_mv_expected
            if complex_values
            else real_triangular_mv_expected
        )
        product = reference(
            matrix,
            solution,
            n,
            n,
            case.uplo,
            case.trans,
            case.diag,
        )
        if case.case.endswith("trsv"):
            input_values = product
            expected = [scalar_value(value, complex_values) for value in solution]
        else:
            input_values = [scalar_value(value, complex_values) for value in solution]
            expected = product

        initial = array_from_values(case.ctype, input_values, complex_values)
        actual = (case.ctype * n)()
        operation = getattr(lib, case.case + "_")
        uplo = ctypes.create_string_buffer(case.uplo.encode("ascii"))
        trans = ctypes.create_string_buffer(case.trans.encode("ascii"))
        diag = ctypes.create_string_buffer(case.diag.encode("ascii"))

        def setup_x():
            copy_array(actual, initial)

        def run_case():
            operation(
                uplo,
                trans,
                diag,
                ctypes.byref(ni),
                ptr(matrix),
                ctypes.byref(ldai),
                ptr(actual),
                ctypes.byref(one),
            )

        check = checked_vector(
            run_case,
            setup_x,
            actual,
            expected,
            case.kind,
            n,
            complex_values=complex_values,
            tolerance_limit=triangular_tolerance(case.kind, n),
        )
        elapsed = best_time(run_case, setup_x, reps)
        emit(
            rows,
            case.case,
            case.kind,
            library_name,
            shape,
            elapsed,
            triangular_work(case, n),
            check,
            parameters=case,
        )
    return rows


def rank_update_tolerance(kind):
    return 5e-5 if kind in ("f32", "c32") else 1e-12


def rank_update_work(case, n):
    triangle_elements = n * (n + 1) // 2
    if case.kind in ("c32", "c64"):
        return (16 if case.rank2 else 8) * triangle_elements
    return (4 if case.rank2 else 2) * triangle_elements


def run_rank_update_cases(lib, library_name, shape, reps, operations):
    if shape.m != shape.n:
        return []
    n = shape.n
    blas_int = ctypes.c_int
    ni = blas_int(n)
    ldai = blas_int(n)
    one = blas_int(1)
    rows = []

    for case_index, case in enumerate(rank_update_cases(operations)):
        seed = 0x4D595DF4D0F33173 ^ (case_index << 12)
        if case.hermitian:
            matrix0 = complex_array(case.value_type, n * n, seed)
            x = complex_array(case.value_type, n, seed ^ 0x2718281828459045)
            y = (
                complex_array(case.value_type, n, seed ^ 0x1618033988749895)
                if case.rank2
                else None
            )
            alpha = (
                case.value_type(0.7, -0.125)
                if case.rank2
                else case.real_type(0.7)
            )
            expected = complex_rank_update_expected(
                matrix0,
                x,
                y,
                n,
                n,
                alpha if case.rank2 else alpha.value,
                case.uplo,
            )
        else:
            matrix0 = real_array(case.value_type, n * n, seed)
            x = real_array(case.value_type, n, seed ^ 0x2718281828459045)
            y = (
                real_array(case.value_type, n, seed ^ 0x1618033988749895)
                if case.rank2
                else None
            )
            alpha = case.real_type(0.7)
            expected = real_rank_update_expected(
                matrix0, x, y, n, n, alpha.value, case.uplo
            )

        array_type = case.value_type * (n * n)
        target = array_type()
        operation = getattr(lib, case.case + "_")
        uplo = ctypes.create_string_buffer(case.uplo.encode("ascii"))

        def setup_a():
            copy_array(target, matrix0)

        if case.rank2:

            def run_case():
                operation(
                    uplo,
                    ctypes.byref(ni),
                    ctypes.byref(alpha),
                    ptr(x),
                    ctypes.byref(one),
                    ptr(y),
                    ctypes.byref(one),
                    ptr(target),
                    ctypes.byref(ldai),
                )

        else:

            def run_case():
                operation(
                    uplo,
                    ctypes.byref(ni),
                    ctypes.byref(alpha),
                    ptr(x),
                    ctypes.byref(one),
                    ptr(target),
                    ctypes.byref(ldai),
                )

        check = checked_vector(
            run_case,
            setup_a,
            target,
            expected,
            case.kind,
            n,
            complex_values=case.hermitian,
            tolerance_limit=rank_update_tolerance(case.kind),
        )
        elapsed = best_time(run_case, setup_a, reps)
        emit(
            rows,
            case.case,
            case.kind,
            library_name,
            shape,
            elapsed,
            rank_update_work(case, n),
            check,
            parameters=case,
        )
    return rows


def packed_structured_mv_work(case, n):
    return (8 if case.hermitian else 2) * n * n


def run_packed_structured_mv_cases(lib, library_name, shape, reps, operations):
    if shape.m != shape.n:
        return []
    n = shape.n
    packed_count = n * (n + 1) // 2
    blas_int = ctypes.c_int
    ni = blas_int(n)
    one = blas_int(1)
    rows = []

    for case_index, case in enumerate(packed_structured_mv_cases(operations)):
        complex_values = case.hermitian
        seed = 0x4D595DF4D0F33173 ^ (case_index << 12)
        array_factory = complex_array if complex_values else real_array
        matrix = array_factory(case.value_type, packed_count, seed)
        x = array_factory(case.value_type, n, seed ^ 0x2718281828459045)
        y0 = array_factory(case.value_type, n, seed ^ 0x1618033988749895)
        y = array_factory(case.value_type, n, seed ^ 0x1123581321345589)
        if complex_values:
            alpha = case.value_type(0.7, 0.125)
            beta = case.value_type(0.3, -0.0625)
        else:
            alpha = case.value_type(0.7)
            beta = case.value_type(0.3)
        operation = getattr(lib, case.case + "_")
        uplo = ctypes.create_string_buffer(case.uplo.encode("ascii"))

        def setup_y():
            copy_array(y, y0)

        def run_case():
            operation(
                uplo,
                ctypes.byref(ni),
                ctypes.byref(alpha),
                ptr(matrix),
                ptr(x),
                ctypes.byref(one),
                ctypes.byref(beta),
                ptr(y),
                ctypes.byref(one),
            )

        expected = packed_structured_mv_expected(
            matrix,
            x,
            y0,
            n,
            alpha,
            beta,
            case.uplo,
            hermitian=case.hermitian,
        )
        check = checked_vector(
            run_case,
            setup_y,
            y,
            expected,
            case.kind,
            n,
            complex_values=complex_values,
        )
        elapsed = best_time(run_case, setup_y, reps)
        emit(
            rows,
            case.case,
            case.kind,
            library_name,
            shape,
            elapsed,
            packed_structured_mv_work(case, n),
            check,
            parameters=case,
        )
    return rows


def compact_triangular_tolerance(kind, n):
    size_scale = max(1.0, n / 128.0)
    return (1e-4 if kind in ("f32", "c32") else 2e-12) * size_scale


def compact_triangular_work(case, n):
    return (4 if case.kind in ("c32", "c64") else 1) * n * n


def run_packed_triangular_cases(lib, library_name, shape, reps, operations):
    if shape.m != shape.n:
        return []
    n = shape.n
    blas_int = ctypes.c_int
    ni = blas_int(n)
    one = blas_int(1)
    rows = []
    matrices = {}
    solutions = {}
    reference_products = {}

    for case_index, case in enumerate(packed_triangular_cases(operations)):
        complex_values = case.kind in ("c32", "c64")
        matrix_key = (case.kind, case.uplo)
        if matrix_key not in matrices:
            matrices[matrix_key] = safe_packed_triangular_matrix(
                case.value_type,
                n,
                case.uplo,
                0x4D595DF4D0F33173 ^ (case_index << 8),
                complex_values=complex_values,
            )
        if case.kind not in solutions:
            array_factory = complex_array if complex_values else real_array
            solutions[case.kind] = array_factory(
                case.value_type, n, 0x2718281828459045 ^ case_index
            )

        matrix = matrices[matrix_key]
        solution = solutions[case.kind]
        reference_key = (case.kind, case.uplo, case.trans, case.diag)
        if reference_key not in reference_products:
            reference_products[reference_key] = triangular_packed_mv_expected(
                matrix,
                solution,
                n,
                case.uplo,
                case.trans,
                case.diag,
                complex_values=complex_values,
            )
        product = reference_products[reference_key]
        if case.solve:
            input_values = product
            expected = [scalar_value(value, complex_values) for value in solution]
        else:
            input_values = [scalar_value(value, complex_values) for value in solution]
            expected = product

        initial = array_from_values(case.value_type, input_values, complex_values)
        actual = (case.value_type * n)()
        operation = getattr(lib, case.case + "_")
        uplo = ctypes.create_string_buffer(case.uplo.encode("ascii"))
        trans = ctypes.create_string_buffer(case.trans.encode("ascii"))
        diag = ctypes.create_string_buffer(case.diag.encode("ascii"))

        def setup_x():
            copy_array(actual, initial)

        def run_case():
            operation(
                uplo,
                trans,
                diag,
                ctypes.byref(ni),
                ptr(matrix),
                ptr(actual),
                ctypes.byref(one),
            )

        check = checked_vector(
            run_case,
            setup_x,
            actual,
            expected,
            case.kind,
            n,
            complex_values=complex_values,
            tolerance_limit=compact_triangular_tolerance(case.kind, n),
        )
        elapsed = best_time(run_case, setup_x, reps)
        emit(
            rows,
            case.case,
            case.kind,
            library_name,
            shape,
            elapsed,
            compact_triangular_work(case, n),
            check,
            parameters=case,
        )
    return rows


def packed_rank_work(case, n):
    triangle_elements = n * (n + 1) // 2
    if case.kind in ("c32", "c64"):
        return (16 if case.rank2 else 8) * triangle_elements
    return (4 if case.rank2 else 2) * triangle_elements


def run_packed_rank_cases(lib, library_name, shape, reps, operations):
    if shape.m != shape.n:
        return []
    n = shape.n
    packed_count = n * (n + 1) // 2
    blas_int = ctypes.c_int
    ni = blas_int(n)
    one = blas_int(1)
    rows = []

    for case_index, case in enumerate(packed_rank_cases(operations)):
        complex_values = case.hermitian
        seed = 0x4D595DF4D0F33173 ^ (case_index << 12)
        array_factory = complex_array if complex_values else real_array
        matrix0 = array_factory(case.value_type, packed_count, seed)
        target = (case.value_type * packed_count)()
        x = array_factory(case.value_type, n, seed ^ 0x2718281828459045)
        y = (
            array_factory(case.value_type, n, seed ^ 0x1618033988749895)
            if case.rank2
            else None
        )
        if case.hermitian and case.rank2:
            alpha = case.value_type(0.7, -0.125)
        else:
            alpha = case.real_type(0.7)
        expected = packed_rank_expected(
            matrix0,
            x,
            y,
            n,
            alpha,
            case.uplo,
            hermitian=case.hermitian,
        )
        operation = getattr(lib, case.case + "_")
        uplo = ctypes.create_string_buffer(case.uplo.encode("ascii"))

        def setup_a():
            copy_array(target, matrix0)

        if case.rank2:

            def run_case():
                operation(
                    uplo,
                    ctypes.byref(ni),
                    ctypes.byref(alpha),
                    ptr(x),
                    ctypes.byref(one),
                    ptr(y),
                    ctypes.byref(one),
                    ptr(target),
                )

        else:

            def run_case():
                operation(
                    uplo,
                    ctypes.byref(ni),
                    ctypes.byref(alpha),
                    ptr(x),
                    ctypes.byref(one),
                    ptr(target),
                )

        check = checked_vector(
            run_case,
            setup_a,
            target,
            expected,
            case.kind,
            n,
            complex_values=complex_values,
            tolerance_limit=rank_update_tolerance(case.kind),
        )
        elapsed = best_time(run_case, setup_a, reps)
        emit(
            rows,
            case.case,
            case.kind,
            library_name,
            shape,
            elapsed,
            packed_rank_work(case, n),
            check,
            parameters=case,
        )
    return rows


def banded_tolerance(kind, bandwidth):
    terms = 2 * bandwidth + 1
    return (5e-5 if kind in ("f32", "c32") else 1e-12) * max(1, terms)


def general_band_element_count(m, n, kl, ku):
    return sum(
        max(0, min(m - 1, col + kl) - max(0, col - ku) + 1)
        for col in range(n)
    )


def banded_work(case, n):
    if case.storage == "general-band":
        elements = general_band_element_count(n, n, case.kl, case.ku)
    else:
        elements = general_band_element_count(n, n, case.k, case.k)
    return (8 if case.kind in ("c32", "c64") else 2) * elements


def run_banded_cases(lib, library_name, profile, reps, operations):
    n = profile.n
    bandwidth = profile.bandwidth
    shape = Shape(profile.name, n, n)
    blas_int = ctypes.c_int
    ni = blas_int(n)
    bandwidth_i = blas_int(bandwidth)
    one = blas_int(1)
    rows = []

    for case_index, case in enumerate(banded_cases(operations, bandwidth)):
        complex_values = case.kind in ("c32", "c64")
        seed = 0x4D595DF4D0F33173 ^ (case_index << 12)
        array_factory = complex_array if complex_values else real_array
        lda = case.lda
        ldai = blas_int(lda)
        matrix = array_factory(case.value_type, lda * n, seed)
        x = array_factory(case.value_type, n, seed ^ 0x2718281828459045)
        y0 = array_factory(case.value_type, n, seed ^ 0x1618033988749895)
        y = array_factory(case.value_type, n, seed ^ 0x1123581321345589)
        if complex_values:
            alpha = case.value_type(0.7, 0.125)
            beta = case.value_type(0.3, -0.0625)
        else:
            alpha = case.value_type(0.7)
            beta = case.value_type(0.3)
        operation = getattr(lib, case.case + "_")

        def setup_y():
            copy_array(y, y0)

        if case.storage == "general-band":
            trans = ctypes.create_string_buffer(case.trans.encode("ascii"))

            def run_case():
                operation(
                    trans,
                    ctypes.byref(ni),
                    ctypes.byref(ni),
                    ctypes.byref(bandwidth_i),
                    ctypes.byref(bandwidth_i),
                    ctypes.byref(alpha),
                    ptr(matrix),
                    ctypes.byref(ldai),
                    ptr(x),
                    ctypes.byref(one),
                    ctypes.byref(beta),
                    ptr(y),
                    ctypes.byref(one),
                )

            expected = general_band_expected(
                matrix,
                x,
                y0,
                n,
                n,
                lda,
                bandwidth,
                bandwidth,
                alpha,
                beta,
                case.trans,
                complex_values=complex_values,
            )
        else:
            uplo = ctypes.create_string_buffer(case.uplo.encode("ascii"))

            def run_case():
                operation(
                    uplo,
                    ctypes.byref(ni),
                    ctypes.byref(bandwidth_i),
                    ctypes.byref(alpha),
                    ptr(matrix),
                    ctypes.byref(ldai),
                    ptr(x),
                    ctypes.byref(one),
                    ctypes.byref(beta),
                    ptr(y),
                    ctypes.byref(one),
                )

            expected = structured_band_expected(
                matrix,
                x,
                y0,
                n,
                lda,
                bandwidth,
                alpha,
                beta,
                case.uplo,
                hermitian=case.storage == "hermitian-band",
            )

        check = checked_vector(
            run_case,
            setup_y,
            y,
            expected,
            case.kind,
            n,
            complex_values=complex_values,
            tolerance_limit=banded_tolerance(case.kind, bandwidth),
        )
        elapsed = best_time(run_case, setup_y, reps)
        emit(
            rows,
            case.case,
            case.kind,
            library_name,
            shape,
            elapsed,
            banded_work(case, n),
            check,
            parameters=case,
        )
    return rows


def triangular_banded_work(case, n, bandwidth):
    elements = general_band_element_count(n, n, bandwidth, bandwidth)
    return (4 if case.kind in ("c32", "c64") else 1) * elements


def run_triangular_banded_cases(lib, library_name, profile, reps, operations):
    n = profile.n
    bandwidth = profile.bandwidth
    shape = Shape(profile.name, n, n)
    blas_int = ctypes.c_int
    ni = blas_int(n)
    bandwidth_i = blas_int(bandwidth)
    one = blas_int(1)
    rows = []
    matrices = {}
    solutions = {}
    reference_products = {}

    for case_index, case in enumerate(
        triangular_banded_cases(operations, bandwidth)
    ):
        complex_values = case.kind in ("c32", "c64")
        matrix_key = (case.kind, case.uplo)
        if matrix_key not in matrices:
            matrices[matrix_key] = safe_triangular_band_matrix(
                case.value_type,
                n,
                bandwidth,
                case.uplo,
                0x4D595DF4D0F33173 ^ (case_index << 8),
                complex_values=complex_values,
            )
        if case.kind not in solutions:
            array_factory = complex_array if complex_values else real_array
            solutions[case.kind] = array_factory(
                case.value_type, n, 0x2718281828459045 ^ case_index
            )

        matrix = matrices[matrix_key]
        solution = solutions[case.kind]
        reference_key = (case.kind, case.uplo, case.trans, case.diag)
        if reference_key not in reference_products:
            reference_products[reference_key] = triangular_band_mv_expected(
                matrix,
                solution,
                n,
                case.lda,
                bandwidth,
                case.uplo,
                case.trans,
                case.diag,
                complex_values=complex_values,
            )
        product = reference_products[reference_key]
        if case.solve:
            input_values = product
            expected = [scalar_value(value, complex_values) for value in solution]
        else:
            input_values = [scalar_value(value, complex_values) for value in solution]
            expected = product

        initial = array_from_values(case.value_type, input_values, complex_values)
        actual = (case.value_type * n)()
        operation = getattr(lib, case.case + "_")
        uplo = ctypes.create_string_buffer(case.uplo.encode("ascii"))
        trans = ctypes.create_string_buffer(case.trans.encode("ascii"))
        diag = ctypes.create_string_buffer(case.diag.encode("ascii"))
        ldai = blas_int(case.lda)

        def setup_x():
            copy_array(actual, initial)

        def run_case():
            operation(
                uplo,
                trans,
                diag,
                ctypes.byref(ni),
                ctypes.byref(bandwidth_i),
                ptr(matrix),
                ctypes.byref(ldai),
                ptr(actual),
                ctypes.byref(one),
            )

        check = checked_vector(
            run_case,
            setup_x,
            actual,
            expected,
            case.kind,
            n,
            complex_values=complex_values,
            tolerance_limit=compact_triangular_tolerance(case.kind, n),
        )
        elapsed = best_time(run_case, setup_x, reps)
        emit(
            rows,
            case.case,
            case.kind,
            library_name,
            shape,
            elapsed,
            triangular_banded_work(case, n, bandwidth),
            check,
            parameters=case,
        )
    return rows


def run_worker(args):
    if not args.library_name or not args.library_path or args.worker_n is None:
        raise SystemExit("--worker requires library name/path and n")
    library_name = args.library_name
    m = args.worker_m if args.worker_m is not None else args.worker_n
    n = args.worker_n
    if m < 1 or n < 1:
        raise SystemExit("--worker dimensions must be positive")
    shape = Shape(args.worker_shape or f"sq{n}", m, n)
    reps = args.worker_reps or 1
    lib = ctypes.CDLL(args.library_path)
    operations = expand_operations(args.worker_op)
    compact_banded_operations = [
        operation for operation in operations if operation in COMPACT_BANDED_OPERATIONS
    ]
    if compact_banded_operations and args.worker_bandwidth is None:
        raise SystemExit("banded worker operations require --worker-bandwidth")
    if compact_banded_operations and (
        m != n or args.worker_bandwidth < 0 or args.worker_bandwidth >= n
    ):
        raise SystemExit(
            "banded worker operations require m=n and 0 <= bandwidth < n"
        )
    banded_profile = (
        BandedProfile(shape.name, n, args.worker_bandwidth)
        if compact_banded_operations
        else None
    )
    if "legacy" not in operations:
        rows = run_triangular_cases(lib, library_name, shape, reps, operations)
        rows.extend(run_rank_update_cases(lib, library_name, shape, reps, operations))
        rows.extend(
            run_packed_structured_mv_cases(
                lib, library_name, shape, reps, operations
            )
        )
        rows.extend(
            run_packed_triangular_cases(lib, library_name, shape, reps, operations)
        )
        rows.extend(run_packed_rank_cases(lib, library_name, shape, reps, operations))
        if banded_profile is not None:
            rows.extend(
                run_banded_cases(
                    lib, library_name, banded_profile, reps, operations
                )
            )
            rows.extend(
                run_triangular_banded_cases(
                    lib, library_name, banded_profile, reps, operations
                )
            )
        writer = csv.DictWriter(sys.stdout, fieldnames=CSV_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)
        return

    blas_int = ctypes.c_int
    one = blas_int(1)
    mi = blas_int(m)
    ni = blas_int(n)
    lda = m
    ldai = blas_int(lda)
    trans_n = ctypes.create_string_buffer(b"N")
    trans_t = ctypes.create_string_buffer(b"T")
    trans_c = ctypes.create_string_buffer(b"C")
    uplo_u = ctypes.create_string_buffer(b"U")
    rows = []

    for kind, ctype, prefix in [
        ("f32", ctypes.c_float, "s"),
        ("f64", ctypes.c_double, "d"),
    ]:
        matrix = real_array(ctype, lda * n, 0x3141592653589793)
        x_m = real_array(ctype, m, 0x2718281828459045)
        x_n = real_array(ctype, n, 0x2718281828459045)
        y_m0 = real_array(ctype, m, 0x1618033988749895)
        y_m = real_array(ctype, m, 0x1123581321345589)
        y_n0 = real_array(ctype, n, 0x1618033988749895)
        y_n = real_array(ctype, n, 0x1123581321345589)
        alpha = ctype(0.7)
        beta = ctype(0.3)
        gemv = getattr(lib, prefix + "gemv_")

        def setup_y_m():
            copy_array(y_m, y_m0)

        def setup_y_n():
            copy_array(y_n, y_n0)

        def run_gemv_n():
            gemv(
                trans_n,
                ctypes.byref(mi),
                ctypes.byref(ni),
                ctypes.byref(alpha),
                ptr(matrix),
                ctypes.byref(ldai),
                ptr(x_n),
                ctypes.byref(one),
                ctypes.byref(beta),
                ptr(y_m),
                ctypes.byref(one),
            )

        check = checked_vector(
            run_gemv_n,
            setup_y_m,
            y_m,
            real_gemv_expected(
                matrix, x_n, y_m0, m, n, lda, alpha.value, beta.value, "N"
            ),
            kind,
            n,
        )
        elapsed = best_time(run_gemv_n, setup_y_m, reps)
        emit(
            rows,
            prefix + "gemv_n",
            kind,
            library_name,
            shape,
            elapsed,
            2 * m * n,
            check,
        )

        def run_gemv_t():
            gemv(
                trans_t,
                ctypes.byref(mi),
                ctypes.byref(ni),
                ctypes.byref(alpha),
                ptr(matrix),
                ctypes.byref(ldai),
                ptr(x_m),
                ctypes.byref(one),
                ctypes.byref(beta),
                ptr(y_n),
                ctypes.byref(one),
            )

        check = checked_vector(
            run_gemv_t,
            setup_y_n,
            y_n,
            real_gemv_expected(
                matrix, x_m, y_n0, m, n, lda, alpha.value, beta.value, "T"
            ),
            kind,
            m,
        )
        elapsed = best_time(run_gemv_t, setup_y_n, reps)
        emit(
            rows,
            prefix + "gemv_t",
            kind,
            library_name,
            shape,
            elapsed,
            2 * m * n,
            check,
        )

        if m == n:
            symv = getattr(lib, prefix + "symv_")

            def run_symv():
                symv(
                    uplo_u,
                    ctypes.byref(ni),
                    ctypes.byref(alpha),
                    ptr(matrix),
                    ctypes.byref(ldai),
                    ptr(x_m),
                    ctypes.byref(one),
                    ctypes.byref(beta),
                    ptr(y_m),
                    ctypes.byref(one),
                )

            check = checked_vector(
                run_symv,
                setup_y_m,
                y_m,
                real_symv_expected(matrix, x_m, y_m0, n, alpha.value, beta.value),
                kind,
                n,
            )
            elapsed = best_time(run_symv, setup_y_m, reps)
            emit(
                rows,
                prefix + "symv",
                kind,
                library_name,
                shape,
                elapsed,
                2 * n * n,
                check,
            )

        matrix0 = real_array(ctype, lda * n, 0x123456789abcdef0)
        target = real_array(ctype, lda * n, 0xfeedfacecafebeef)
        gy_n = real_array(ctype, n, 0x0102030405060708)
        ger = getattr(lib, prefix + "ger_")

        def setup_a():
            copy_array(target, matrix0)

        def run_ger():
            ger(
                ctypes.byref(mi),
                ctypes.byref(ni),
                ctypes.byref(alpha),
                ptr(x_m),
                ctypes.byref(one),
                ptr(gy_n),
                ctypes.byref(one),
                ptr(target),
                ctypes.byref(ldai),
            )

        check = checked_vector(
            run_ger,
            setup_a,
            target,
            real_ger_expected(matrix0, x_m, gy_n, m, n, lda, alpha.value),
            kind,
            max(m, n),
        )
        elapsed = best_time(run_ger, setup_a, reps)
        emit(
            rows,
            prefix + "ger",
            kind,
            library_name,
            shape,
            elapsed,
            2 * m * n,
            check,
        )

    for kind, complex_type, prefix in [
        ("c32", ComplexF32, "c"),
        ("c64", ComplexF64, "z"),
    ]:
        matrix = complex_array(complex_type, lda * n, 0x3141592653589793)
        x_m = complex_array(complex_type, m, 0x2718281828459045)
        x_n = complex_array(complex_type, n, 0x2718281828459045)
        y_m0 = complex_array(complex_type, m, 0x1618033988749895)
        y_m = complex_array(complex_type, m, 0x1123581321345589)
        y_n0 = complex_array(complex_type, n, 0x1618033988749895)
        y_n = complex_array(complex_type, n, 0x1123581321345589)
        alpha = complex_type(0.7, 0.125)
        beta = complex_type(0.3, -0.0625)
        gemv = getattr(lib, prefix + "gemv_")

        def setup_y_m():
            copy_array(y_m, y_m0)

        def setup_y_n():
            copy_array(y_n, y_n0)

        def run_gemv_n():
            gemv(
                trans_n,
                ctypes.byref(mi),
                ctypes.byref(ni),
                ctypes.byref(alpha),
                ptr(matrix),
                ctypes.byref(ldai),
                ptr(x_n),
                ctypes.byref(one),
                ctypes.byref(beta),
                ptr(y_m),
                ctypes.byref(one),
            )

        check = checked_vector(
            run_gemv_n,
            setup_y_m,
            y_m,
            complex_gemv_expected(matrix, x_n, y_m0, m, n, lda, alpha, beta, "N"),
            kind,
            n,
            complex_values=True,
        )
        elapsed = best_time(run_gemv_n, setup_y_m, reps)
        emit(
            rows,
            prefix + "gemv_n",
            kind,
            library_name,
            shape,
            elapsed,
            8 * m * n,
            check,
        )

        def run_gemv_t():
            gemv(
                trans_t,
                ctypes.byref(mi),
                ctypes.byref(ni),
                ctypes.byref(alpha),
                ptr(matrix),
                ctypes.byref(ldai),
                ptr(x_m),
                ctypes.byref(one),
                ctypes.byref(beta),
                ptr(y_n),
                ctypes.byref(one),
            )

        check = checked_vector(
            run_gemv_t,
            setup_y_n,
            y_n,
            complex_gemv_expected(matrix, x_m, y_n0, m, n, lda, alpha, beta, "T"),
            kind,
            m,
            complex_values=True,
        )
        elapsed = best_time(run_gemv_t, setup_y_n, reps)
        emit(
            rows,
            prefix + "gemv_t",
            kind,
            library_name,
            shape,
            elapsed,
            8 * m * n,
            check,
        )

        def run_gemv_c():
            gemv(
                trans_c,
                ctypes.byref(mi),
                ctypes.byref(ni),
                ctypes.byref(alpha),
                ptr(matrix),
                ctypes.byref(ldai),
                ptr(x_m),
                ctypes.byref(one),
                ctypes.byref(beta),
                ptr(y_n),
                ctypes.byref(one),
            )

        check = checked_vector(
            run_gemv_c,
            setup_y_n,
            y_n,
            complex_gemv_expected(matrix, x_m, y_n0, m, n, lda, alpha, beta, "C"),
            kind,
            m,
            complex_values=True,
        )
        elapsed = best_time(run_gemv_c, setup_y_n, reps)
        emit(
            rows,
            prefix + "gemv_c",
            kind,
            library_name,
            shape,
            elapsed,
            8 * m * n,
            check,
        )

        if m == n:
            hemv = getattr(lib, prefix + "hemv_")

            def run_hemv():
                hemv(
                    uplo_u,
                    ctypes.byref(ni),
                    ctypes.byref(alpha),
                    ptr(matrix),
                    ctypes.byref(ldai),
                    ptr(x_m),
                    ctypes.byref(one),
                    ctypes.byref(beta),
                    ptr(y_m),
                    ctypes.byref(one),
                )

            check = checked_vector(
                run_hemv,
                setup_y_m,
                y_m,
                complex_hemv_expected(matrix, x_m, y_m0, n, alpha, beta),
                kind,
                n,
                complex_values=True,
            )
            elapsed = best_time(run_hemv, setup_y_m, reps)
            emit(
                rows,
                prefix + "hemv",
                kind,
                library_name,
                shape,
                elapsed,
                8 * n * n,
                check,
            )

        matrix0 = complex_array(complex_type, lda * n, 0x123456789abcdef0)
        target = complex_array(complex_type, lda * n, 0xfeedfacecafebeef)
        gy_n = complex_array(complex_type, n, 0x0102030405060708)

        def setup_a():
            copy_array(target, matrix0)

        geru = getattr(lib, prefix + "geru_")

        def run_geru():
            geru(
                ctypes.byref(mi),
                ctypes.byref(ni),
                ctypes.byref(alpha),
                ptr(x_m),
                ctypes.byref(one),
                ptr(gy_n),
                ctypes.byref(one),
                ptr(target),
                ctypes.byref(ldai),
            )

        check = checked_vector(
            run_geru,
            setup_a,
            target,
            complex_ger_expected(matrix0, x_m, gy_n, m, n, lda, alpha, False),
            kind,
            max(m, n),
            complex_values=True,
        )
        elapsed = best_time(run_geru, setup_a, reps)
        emit(
            rows,
            prefix + "geru",
            kind,
            library_name,
            shape,
            elapsed,
            8 * m * n,
            check,
        )

        gerc = getattr(lib, prefix + "gerc_")

        def run_gerc():
            gerc(
                ctypes.byref(mi),
                ctypes.byref(ni),
                ctypes.byref(alpha),
                ptr(x_m),
                ctypes.byref(one),
                ptr(gy_n),
                ctypes.byref(one),
                ptr(target),
                ctypes.byref(ldai),
            )

        check = checked_vector(
            run_gerc,
            setup_a,
            target,
            complex_ger_expected(matrix0, x_m, gy_n, m, n, lda, alpha, True),
            kind,
            max(m, n),
            complex_values=True,
        )
        elapsed = best_time(run_gerc, setup_a, reps)
        emit(
            rows,
            prefix + "gerc",
            kind,
            library_name,
            shape,
            elapsed,
            8 * m * n,
            check,
        )

    rows.extend(run_triangular_cases(lib, library_name, shape, reps, operations))
    rows.extend(run_rank_update_cases(lib, library_name, shape, reps, operations))
    rows.extend(run_packed_structured_mv_cases(lib, library_name, shape, reps, operations))
    rows.extend(run_packed_triangular_cases(lib, library_name, shape, reps, operations))
    rows.extend(run_packed_rank_cases(lib, library_name, shape, reps, operations))
    if banded_profile is not None:
        rows.extend(
            run_banded_cases(lib, library_name, banded_profile, reps, operations)
        )
        rows.extend(
            run_triangular_banded_cases(
                lib, library_name, banded_profile, reps, operations
            )
        )
    writer = csv.DictWriter(
        sys.stdout,
        fieldnames=CSV_FIELDNAMES,
    )
    writer.writeheader()
    writer.writerows(rows)


def run_one_process(
    script, library_name, library_path, shape, reps, operations, bandwidth=None
):
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
        "--worker-shape",
        shape.name,
        "--worker-m",
        str(shape.m),
        "--worker-n",
        str(shape.n),
        "--worker-reps",
        str(reps),
    ]
    for operation in operations:
        cmd.extend(("--worker-op", operation))
    if bandwidth is not None:
        cmd.extend(("--worker-bandwidth", str(bandwidth)))
    return subprocess.run(cmd, capture_output=True, text=True, check=False)


def process_group_key(row):
    n = row["n"]
    m = row.get("m") or n
    shape = row.get("shape") or f"sq{n}"
    return (
        row["case"],
        row["kind"],
        shape,
        m,
        n,
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
        row["metric"],
    )


def repeat_row_eligible(row):
    if row is None:
        return False
    if row.get("status") != "ok" or row.get("check_status") not in CHECKED_STATUSES:
        return False
    try:
        return float(row["rate_gops"]) >= 0.0 and int(row["time_ns"]) > 0
    except (KeyError, ValueError):
        return False


def median(values):
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2 == 1:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2


def repeat_failure_status(rows):
    existing = [row for row in rows if row is not None]
    if len(existing) != len(rows):
        return "error", "error"
    statuses = {row.get("status", "error") for row in existing}
    check_statuses = {row.get("check_status", "error") for row in existing}
    if "error" in statuses or "error" in check_statuses:
        return "error", "error"
    if "missing" in statuses or "missing" in check_statuses:
        return "missing", "missing"
    if "correctness_failed" in statuses or "correctness_failed" in check_statuses:
        return "correctness_failed", "correctness_failed"
    return "error", "error"


def aggregate_repeat_group(rows):
    eligible_rows = [row for row in rows if repeat_row_eligible(row)]
    existing_rows = [row for row in rows if row is not None]
    if not existing_rows:
        raise ValueError("cannot aggregate an empty repeat group")

    if eligible_rows:
        result = dict(max(eligible_rows, key=lambda row: float(row["rate_gops"])))
        values = [float(row["rate_gops"]) for row in eligible_rows]
        result.update(
            {
                "successful_repeats": len(eligible_rows),
                "metric_min": format(min(values), ".17g"),
                "metric_median": format(median(values), ".17g"),
                "metric_max": format(max(values), ".17g"),
                "metric_samples": ",".join(format(value, ".17g") for value in values),
            }
        )
    else:
        result = dict(existing_rows[0])
        result.update(
            {
                "successful_repeats": 0,
                "metric_min": "",
                "metric_median": "",
                "metric_max": "",
                "metric_samples": "",
            }
        )

    result["process_repeats"] = len(rows)
    errors = []
    details = []
    for repeat, row in enumerate(rows):
        if row is None:
            details.append(f"repeat={repeat}: missing result row")
            continue
        try:
            errors.append(float(row["check_max_abs_error"]))
        except (KeyError, ValueError):
            pass
        raw = row.get("check_raw_output", "")
        if not repeat_row_eligible(row) or raw:
            detail = (
                f"repeat={repeat}: status={row.get('status', '')} "
                f"check_status={row.get('check_status', '')}"
            )
            if raw:
                detail += f" {raw}"
            details.append(detail)
    if errors:
        result["check_max_abs_error"] = f"{max(errors):.9g}"
    if details:
        result["check_raw_output"] = " | ".join(details)

    if len(eligible_rows) != len(rows):
        result["status"], result["check_status"] = repeat_failure_status(rows)
    return result


def aggregate_worker_repeats(repeat_rows):
    if not repeat_rows:
        return []
    repeat_maps = []
    key_order = []
    seen = set()
    for rows in repeat_rows:
        current = {}
        for row in rows:
            key = process_group_key(row)
            if key in current:
                raise ValueError(f"duplicate worker row for {key}")
            current[key] = row
            if key not in seen:
                seen.add(key)
                key_order.append(key)
        repeat_maps.append(current)
    return [
        aggregate_repeat_group([current.get(key) for current in repeat_maps])
        for key in key_order
    ]


def write_metadata(
    args,
    output_path,
    selected_libraries,
    shapes,
    packed_profiles=(),
    banded_profiles=(),
):
    output = Path(output_path)
    metadata = {
        "generated_at_unix": time.time(),
        "argv": sys.argv,
        "cwd": os.getcwd(),
        "zig_version": zig_version(),
        "git_revision": git_revision(),
        "detected_cpu_count": os.cpu_count(),
        "zynum_maximum_threads": zynum_maximum_threads_detected(),
        "sizes": unique_preserving_order(
            [shape.n for shape in shapes if shape.m == shape.n]
            + [profile.n for profile in packed_profiles]
            + [profile.n for profile in banded_profiles]
        ),
        "shapes": [
            {"name": shape.name, "m": shape.m, "n": shape.n} for shape in shapes
        ],
        "banded_profiles": [
            {
                "name": profile.name,
                "n": profile.n,
                "bandwidth": profile.bandwidth,
            }
            for profile in banded_profiles
        ],
        "packed_profiles": [
            {"name": profile.name, "n": profile.n}
            for profile in packed_profiles
        ],
        "operations": requested_operations(args),
        "reps_small": args.reps_small,
        "reps_large": args.reps_large,
        "process_repeats": args.process_repeats,
        "isolation": (
            "fresh process per library/shape/repeat; best repeat kept as the "
            "primary metric with min/median/max and ordered samples retained"
        ),
        "correctness_check": "sampled per library/case/shape before timing",
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
    operations = requested_operations(args)
    selected_packed_operations = [
        operation for operation in operations if operation in PACKED_OPERATIONS
    ]
    dense_operations = [
        operation
        for operation in operations
        if operation not in PACKED_OPERATIONS
        and operation not in COMPACT_BANDED_OPERATIONS
    ]
    selected_banded_operations = [
        operation for operation in operations if operation in BANDED_OPERATIONS
    ]
    selected_triangular_banded_operations = [
        operation
        for operation in operations
        if operation in TRIANGULAR_BANDED_OPERATIONS
    ]
    selected_compact_banded_operations = (
        selected_banded_operations + selected_triangular_banded_operations
    )
    shapes = requested_shapes(args) if dense_operations else []
    packed_profiles = (
        requested_packed_profiles(args) if selected_packed_operations else []
    )
    banded_profiles = []
    if selected_banded_operations:
        banded_profiles.extend(requested_banded_profiles(args))
    if selected_triangular_banded_operations:
        banded_profiles.extend(requested_triangular_banded_profiles(args))
    banded_profiles = unique_preserving_order(banded_profiles)
    selected_libraries = libraries(args)
    rows = []
    script = Path(__file__)
    jobs = [
        (shape, operations_for_shape(dense_operations, shape), None)
        for shape in shapes
    ]
    jobs.extend(
        (
            Shape(profile.name, profile.n, profile.n),
            selected_packed_operations,
            None,
        )
        for profile in packed_profiles
    )
    jobs.extend(
        (
            Shape(profile.name, profile.n, profile.n),
            selected_compact_banded_operations,
            profile.bandwidth,
        )
        for profile in banded_profiles
    )
    for shape, shape_operations, bandwidth in jobs:
        if not shape_operations:
            print(
                f"[level2] shape={shape.name} m={shape.m} n={shape.n} "
                "skipping square-only selected operations on a non-square shape",
                file=sys.stderr,
                flush=True,
            )
            continue
        reps = args.reps_small if shape.m * shape.n <= 256 * 256 else args.reps_large
        if shape.m != shape.n:
            print(
                f"[level2] shape={shape.name} m={shape.m} n={shape.n} "
                "skipping square-only SYMV/HEMV",
                file=sys.stderr,
                flush=True,
            )
        for library_name, library_path in selected_libraries:
            if args.skip_missing and not library_available(library_path):
                continue
            repeat_rows = []
            for repeat in range(args.process_repeats):
                print(
                    f"[level2 {library_name}] shape={shape.name} "
                    f"m={shape.m} n={shape.n} reps={reps} "
                    f"bandwidth={bandwidth if bandwidth is not None else '-'} "
                    f"process={repeat + 1}/{args.process_repeats} "
                    f"path={library_path}",
                    file=sys.stderr,
                    flush=True,
                )
                result = run_one_process(
                    script,
                    library_name,
                    library_path,
                    shape,
                    reps,
                    shape_operations,
                    bandwidth=bandwidth,
                )
                if result.returncode != 0:
                    sys.stderr.write(result.stdout)
                    sys.stderr.write(result.stderr)
                    raise SystemExit(result.returncode)
                process_rows = list(csv.DictReader(result.stdout.splitlines()))
                if not process_rows:
                    sys.stderr.write(result.stderr)
                    raise SystemExit(
                        f"worker returned no rows for {library_name} {shape.name} "
                        f"repeat {repeat + 1}"
                    )
                repeat_rows.append(process_rows)
            rows.extend(aggregate_worker_repeats(repeat_rows))

    output = Path(args.csv)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)
    write_metadata(
        args,
        output,
        selected_libraries,
        shapes,
        packed_profiles=packed_profiles,
        banded_profiles=banded_profiles,
    )


def main():
    args = parse_args()
    if args.worker:
        run_worker(args)
    else:
        run_controller(args)


if __name__ == "__main__":
    main()
