#!/usr/bin/env python3
# Copyright (C) 2026 Zynum contributors
# SPDX-License-Identifier: LGPL-3.0-or-later

import csv
import ctypes
import ctypes.util
import importlib.util
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

TOOLS_DIR = Path(__file__).resolve().parent
REPO_ROOT = TOOLS_DIR.parents[1]


def load_tool(module_name):
    spec = importlib.util.spec_from_file_location(
        module_name, TOOLS_DIR / f"{module_name}.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


checker = load_tool("check_level2_report")
runner = load_tool("run_level2_report")


def find_test_blas():
    local_zynum = REPO_ROOT / runner.default_zynum_blas()
    candidates = [
        runner.DEFAULT_ACCELERATE,
        ctypes.util.find_library("blas"),
        str(local_zynum),
        runner.DEFAULT_OPENBLAS,
    ]
    required = [
        "sgemv_",
        "dgemv_",
        "cgemv_",
        "zgemv_",
        "sger_",
        "dger_",
        "cgeru_",
        "cgerc_",
        "zgeru_",
        "zgerc_",
        "strmv_",
        "dtrmv_",
        "ctrmv_",
        "ztrmv_",
        "strsv_",
        "dtrsv_",
        "ctrsv_",
        "ztrsv_",
        "ssyr_",
        "dsyr_",
        "cher_",
        "zher_",
        "ssyr2_",
        "dsyr2_",
        "cher2_",
        "zher2_",
        "sgbmv_",
        "dgbmv_",
        "cgbmv_",
        "zgbmv_",
        "ssbmv_",
        "dsbmv_",
        "chbmv_",
        "zhbmv_",
        "sspmv_",
        "dspmv_",
        "chpmv_",
        "zhpmv_",
        "stpmv_",
        "dtpmv_",
        "ctpmv_",
        "ztpmv_",
        "stpsv_",
        "dtpsv_",
        "ctpsv_",
        "ztpsv_",
        "sspr_",
        "dspr_",
        "chpr_",
        "zhpr_",
        "sspr2_",
        "dspr2_",
        "chpr2_",
        "zhpr2_",
        "stbmv_",
        "dtbmv_",
        "ctbmv_",
        "ztbmv_",
        "stbsv_",
        "dtbsv_",
        "ctbsv_",
        "ztbsv_",
    ]
    for candidate in candidates:
        if not candidate:
            continue
        try:
            library = ctypes.CDLL(candidate)
        except OSError:
            continue
        if all(hasattr(library, symbol) for symbol in required):
            return candidate
    return None


TEST_BLAS = find_test_blas()


def worker_row(
    library,
    rate,
    time_ns,
    *,
    status="ok",
    check_status="sampled-ok",
    check_error="0",
    check_raw="",
    case="sgemv_n",
    kind="f32",
    uplo="",
    trans="",
    diag="",
    incx="",
    incy="",
    storage="",
    lda="",
    k="",
    kl="",
    ku="",
):
    return {
        "level": "level2",
        "case": case,
        "kind": kind,
        "library": library,
        "n": "2",
        "time_ns": str(time_ns),
        "rate_gops": str(rate),
        "metric": "gops",
        "status": status,
        "check_status": check_status,
        "check_max_abs_error": check_error,
        "check_raw_output": check_raw,
        "shape": "rect3x2",
        "m": "3",
        "storage": storage,
        "lda": lda,
        "k": k,
        "kl": kl,
        "ku": ku,
        "uplo": uplo,
        "trans": trans,
        "diag": diag,
        "incx": incx,
        "incy": incy,
    }


class Level2RunnerTests(unittest.TestCase):
    def test_repeated_square_and_rectangular_cli_shapes(self):
        default_args = runner.parse_args(["--csv", os.devnull])
        self.assertEqual(default_args.process_repeats, 1)
        self.assertEqual(runner.requested_operations(default_args), ["legacy"])
        self.assertEqual(
            runner.requested_shapes(default_args),
            [runner.Shape(f"sq{n}", n, n) for n in runner.DEFAULT_N],
        )

        args = runner.parse_args(
            [
                "--csv",
                os.devnull,
                "--n",
                "4",
                "--n",
                "7",
                "--shape",
                "tall:9:3",
                "--shape",
                "wide:3:9",
                "--process-repeats",
                "3",
            ]
        )
        self.assertEqual(args.process_repeats, 3)
        self.assertEqual(
            runner.requested_shapes(args),
            [
                runner.Shape("sq4", 4, 4),
                runner.Shape("sq7", 7, 7),
                runner.Shape("tall", 9, 3),
                runner.Shape("wide", 3, 9),
            ],
        )

        shape_only_args = runner.parse_args(
            ["--csv", os.devnull, "--shape", "only:11:5"]
        )
        self.assertEqual(
            runner.requested_shapes(shape_only_args),
            [runner.Shape("only", 11, 5)],
        )

    def test_triangular_cli_case_expansion(self):
        args = runner.parse_args(
            [
                "--csv",
                os.devnull,
                "--shape",
                "sq128:128:128",
                "--shape",
                "sq512:512:512",
                "--shape",
                "sq2048:2048:2048",
                "--op",
                "trmv",
                "--op",
                "trsv",
            ]
        )
        self.assertEqual(
            runner.requested_shapes(args),
            [
                runner.Shape("sq128", 128, 128),
                runner.Shape("sq512", 512, 512),
                runner.Shape("sq2048", 2048, 2048),
            ],
        )
        operations = runner.requested_operations(args)
        self.assertEqual(operations, list(runner.TRIANGULAR_OPERATIONS))
        cases = runner.triangular_cases(operations)
        self.assertEqual(len(cases), 80)
        self.assertEqual({case.case for case in cases}, set(operations))
        self.assertEqual({case.kind for case in cases}, {"f32", "f64", "c32", "c64"})
        self.assertEqual({case.uplo for case in cases}, {"U", "L"})
        self.assertEqual(
            {case.trans for case in cases if case.kind in ("f32", "f64")},
            {"N", "T"},
        )
        self.assertEqual(
            {case.trans for case in cases if case.kind in ("c32", "c64")},
            {"N", "T", "C"},
        )
        self.assertEqual({case.diag for case in cases}, {"N", "U"})
        self.assertEqual({case.incx for case in cases}, {1})

        trmv = runner.requested_operations(
            runner.parse_args(["--csv", os.devnull, "--op", "trmv"])
        )
        trsv = runner.requested_operations(
            runner.parse_args(["--csv", os.devnull, "--op", "trsv"])
        )
        self.assertEqual(trmv, ["strmv", "dtrmv", "ctrmv", "ztrmv"])
        self.assertEqual(trsv, ["strsv", "dtrsv", "ctrsv", "ztrsv"])

    def test_complex_dense_triangular_reference(self):
        matrix = (runner.ComplexF64 * 4)(
            runner.ComplexF64(1, 2),
            runner.ComplexF64(100, 200),
            runner.ComplexF64(3, 4),
            runner.ComplexF64(5, -1),
        )
        x = (runner.ComplexF64 * 2)(
            runner.ComplexF64(2, -1), runner.ComplexF64(-1, 3)
        )
        self.assertEqual(
            runner.complex_triangular_mv_expected(
                matrix, x, 2, 2, "U", "N", "N"
            ),
            [-11 + 8j, -2 + 16j],
        )
        self.assertEqual(
            runner.complex_triangular_mv_expected(
                matrix, x, 2, 2, "U", "T", "N"
            ),
            [4 + 3j, 8 + 21j],
        )
        self.assertEqual(
            runner.complex_triangular_mv_expected(
                matrix, x, 2, 2, "U", "C", "N"
            ),
            [-5j, -6 + 3j],
        )
        self.assertEqual(
            runner.complex_triangular_mv_expected(
                matrix, x, 2, 2, "U", "N", "U"
            ),
            [-13 + 4j, -1 + 3j],
        )

    def test_complex_dense_triangular_check_rejects_wrong_result(self):
        actual = (runner.ComplexF64 * 2)(
            runner.ComplexF64(0, 0), runner.ComplexF64(0, 0)
        )
        check = runner.checked_vector(
            lambda: None,
            lambda: None,
            actual,
            [1 + 2j, -3 + 4j],
            "c64",
            2,
            complex_values=True,
            tolerance_limit=runner.triangular_tolerance("c64", 2),
        )
        self.assertEqual(check["check_status"], "correctness_failed")
        self.assertGreater(float(check["check_max_abs_error"]), 0.0)

    def test_rank_update_cli_case_expansion_and_official_group_count(self):
        args = runner.parse_args(
            [
                "--csv",
                os.devnull,
                "--shape",
                "sq128:128:128",
                "--shape",
                "sq512:512:512",
                "--shape",
                "sq2048:2048:2048",
                "--op",
                "rank-update",
            ]
        )
        operations = runner.requested_operations(args)
        self.assertEqual(operations, list(runner.RANK_UPDATE_OPERATIONS))
        cases = runner.rank_update_cases(operations)
        self.assertEqual(len(cases), 16)
        self.assertEqual({case.case for case in cases}, set(operations))
        self.assertEqual({case.kind for case in cases}, {"f32", "f64", "c32", "c64"})
        self.assertEqual({case.uplo for case in cases}, {"U", "L"})
        self.assertEqual({case.incx for case in cases}, {1})
        self.assertEqual({case.incy for case in cases}, {1})

        logical_groups = {
            (shape.name, case.case, case.kind, case.uplo, case.incx, case.incy)
            for shape in runner.requested_shapes(args)
            for case in cases
        }
        self.assertEqual(len(logical_groups), 48)

    def test_banded_cli_case_expansion_and_official_group_count(self):
        args = runner.parse_args(["--csv", os.devnull, "--op", "banded"])
        operations = runner.requested_operations(args)
        self.assertEqual(operations, list(runner.BANDED_OPERATIONS))
        self.assertEqual(
            runner.requested_banded_profiles(args),
            list(runner.DEFAULT_BANDED_PROFILES),
        )
        cases = runner.banded_cases(operations, 8)
        self.assertEqual(len(cases), 18)
        self.assertEqual(
            {(case.case, case.trans) for case in cases if case.storage == "general-band"},
            {
                ("sgbmv", "N"),
                ("sgbmv", "T"),
                ("dgbmv", "N"),
                ("dgbmv", "T"),
                ("cgbmv", "N"),
                ("cgbmv", "T"),
                ("cgbmv", "C"),
                ("zgbmv", "N"),
                ("zgbmv", "T"),
                ("zgbmv", "C"),
            },
        )
        self.assertEqual(
            {(case.case, case.uplo) for case in cases if case.storage != "general-band"},
            {
                (case, uplo)
                for case in ("ssbmv", "dsbmv", "chbmv", "zhbmv")
                for uplo in ("U", "L")
            },
        )
        logical_groups = {
            (
                profile.name,
                case.case,
                case.kind,
                case.storage,
                case.lda,
                case.k,
                case.kl,
                case.ku,
                case.uplo,
                case.trans,
                case.incx,
                case.incy,
            )
            for profile in runner.requested_banded_profiles(args)
            for case in runner.banded_cases(operations, profile.bandwidth)
        }
        self.assertEqual(len(logical_groups), 36)

        custom = runner.parse_args(
            [
                "--csv",
                os.devnull,
                "--op",
                "banded",
                "--band-profile",
                "smoke:7:2",
            ]
        )
        self.assertEqual(
            runner.requested_banded_profiles(custom),
            [runner.BandedProfile("smoke", 7, 2)],
        )

    def test_packed_profiles_and_case_expansion(self):
        args = runner.parse_args(["--csv", os.devnull, "--op", "packed-mv"])
        operations = runner.requested_operations(args)
        self.assertEqual(operations, list(runner.PACKED_MV_OPERATIONS))
        self.assertEqual(
            runner.requested_packed_profiles(args),
            list(runner.DEFAULT_PACKED_PROFILES),
        )
        structured = runner.packed_structured_mv_cases(operations)
        triangular = runner.packed_triangular_cases(operations)
        self.assertEqual(len(structured), 8)
        self.assertEqual(len(triangular), 80)
        self.assertEqual({case.uplo for case in structured + triangular}, {"U", "L"})
        self.assertEqual({case.diag for case in triangular}, {"N", "U"})
        self.assertEqual(
            {case.trans for case in triangular if case.kind.startswith("f")},
            {"N", "T"},
        )
        self.assertEqual(
            {case.trans for case in triangular if case.kind.startswith("c")},
            {"N", "T", "C"},
        )
        self.assertEqual({case.incx for case in triangular}, {1})
        self.assertEqual(
            len(runner.requested_packed_profiles(args))
            * (len(structured) + len(triangular)),
            264,
        )

        rank_args = runner.parse_args(
            [
                "--csv",
                os.devnull,
                "--op",
                "packed-rank",
                "--packed-profile",
                "smoke:7",
            ]
        )
        self.assertEqual(
            runner.requested_packed_profiles(rank_args),
            [runner.PackedProfile("smoke", 7)],
        )
        rank_cases = runner.packed_rank_cases(
            runner.requested_operations(rank_args)
        )
        self.assertEqual(len(rank_cases), 16)
        self.assertEqual({case.uplo for case in rank_cases}, {"U", "L"})
        self.assertEqual({case.incx for case in rank_cases}, {1})
        self.assertEqual({case.incy for case in rank_cases}, {1})

    def test_triangular_banded_profiles_and_case_expansion(self):
        args = runner.parse_args(
            ["--csv", os.devnull, "--op", "triangular-banded"]
        )
        operations = runner.requested_operations(args)
        self.assertEqual(operations, list(runner.TRIANGULAR_BANDED_OPERATIONS))
        self.assertEqual(
            runner.requested_triangular_banded_profiles(args),
            list(runner.DEFAULT_TRIANGULAR_BANDED_PROFILES),
        )
        cases = runner.triangular_banded_cases(operations, 8)
        self.assertEqual(len(cases), 80)
        self.assertEqual({case.storage for case in cases}, {"triangular-band"})
        self.assertEqual({case.lda for case in cases}, {9})
        self.assertEqual({case.k for case in cases}, {8})
        self.assertEqual({case.uplo for case in cases}, {"U", "L"})
        self.assertEqual({case.diag for case in cases}, {"N", "U"})
        self.assertEqual(
            {case.trans for case in cases if case.kind.startswith("f")},
            {"N", "T"},
        )
        self.assertEqual(
            {case.trans for case in cases if case.kind.startswith("c")},
            {"N", "T", "C"},
        )
        self.assertEqual({case.incx for case in cases}, {1})

    def test_banded_references_decode_compact_storage(self):
        general = (ctypes.c_double * 9)(777, 1, 2, 3, 4, 5, 6, 7, 777)
        x3 = (ctypes.c_double * 3)(1, 2, 3)
        y3 = (ctypes.c_double * 3)(0, 0, 0)
        self.assertEqual(
            runner.general_band_expected(
                general, x3, y3, 3, 3, 3, 1, 1, 1, 0, "N"
            ),
            [7, 28, 31],
        )
        self.assertEqual(
            runner.general_band_expected(
                general, x3, y3, 3, 3, 3, 1, 1, 1, 0, "T"
            ),
            [5, 26, 33],
        )

        complex_general = (runner.ComplexF64 * 6)(
            runner.ComplexF64(777, 777),
            runner.ComplexF64(1, 1),
            runner.ComplexF64(2, 3),
            runner.ComplexF64(4, 5),
            runner.ComplexF64(6, 7),
            runner.ComplexF64(777, 777),
        )
        complex_x = (runner.ComplexF64 * 2)(
            runner.ComplexF64(1, 0), runner.ComplexF64(0, 1)
        )
        complex_y = (runner.ComplexF64 * 2)(
            runner.ComplexF64(0, 0), runner.ComplexF64(0, 0)
        )
        one = runner.ComplexF64(1, 0)
        zero = runner.ComplexF64(0, 0)
        self.assertEqual(
            runner.general_band_expected(
                complex_general,
                complex_x,
                complex_y,
                2,
                2,
                3,
                1,
                1,
                one,
                zero,
                "C",
                complex_values=True,
            ),
            [4 + 1j, 11 + 1j],
        )

        symmetric_upper = (ctypes.c_double * 6)(777, 1, 2, 3, 4, 5)
        self.assertEqual(
            runner.structured_band_expected(
                symmetric_upper, x3, y3, 3, 2, 1, 1, 0, "U"
            ),
            [5, 20, 23],
        )

        hermitian_upper = (runner.ComplexF64 * 4)(
            runner.ComplexF64(777, 777),
            runner.ComplexF64(1, 99),
            runner.ComplexF64(2, 3),
            runner.ComplexF64(4, 88),
        )
        self.assertEqual(
            runner.structured_band_expected(
                hermitian_upper,
                complex_x,
                complex_y,
                2,
                2,
                1,
                one,
                zero,
                "U",
                hermitian=True,
            ),
            [-2 + 2j, 2 + 1j],
        )

    def test_packed_and_triangular_band_references_decode_storage(self):
        n = 3
        upper = (ctypes.c_double * 6)(1, 2, 3, 4, 5, 6)
        lower = (ctypes.c_double * 6)(1, 2, 4, 3, 5, 6)
        x = (ctypes.c_double * 3)(1, 2, 3)
        y = (ctypes.c_double * 3)(0, 0, 0)
        self.assertEqual(
            runner.packed_structured_mv_expected(
                upper, x, y, n, 1, 0, "U"
            ),
            [17, 23, 32],
        )
        self.assertEqual(
            runner.packed_structured_mv_expected(
                lower, x, y, n, 1, 0, "L"
            ),
            [17, 23, 32],
        )

        hermitian = (runner.ComplexF64 * 3)(
            runner.ComplexF64(1, 99),
            runner.ComplexF64(2, 3),
            runner.ComplexF64(4, 88),
        )
        complex_x = (runner.ComplexF64 * 2)(
            runner.ComplexF64(1, 0), runner.ComplexF64(0, 1)
        )
        complex_y = (runner.ComplexF64 * 2)(
            runner.ComplexF64(0, 0), runner.ComplexF64(0, 0)
        )
        self.assertEqual(
            runner.packed_structured_mv_expected(
                hermitian,
                complex_x,
                complex_y,
                2,
                runner.ComplexF64(1, 0),
                runner.ComplexF64(0, 0),
                "U",
                hermitian=True,
            ),
            [-2 + 2j, 2 + 1j],
        )

        triangular = (ctypes.c_double * 3)(2, 3, 4)
        tx = (ctypes.c_double * 2)(1, 2)
        self.assertEqual(
            runner.triangular_packed_mv_expected(
                triangular, tx, 2, "U", "N", "N"
            ),
            [8, 8],
        )
        self.assertEqual(
            runner.triangular_packed_mv_expected(
                triangular, tx, 2, "U", "T", "N"
            ),
            [2, 11],
        )
        self.assertEqual(
            runner.triangular_packed_mv_expected(
                triangular, tx, 2, "U", "N", "U"
            ),
            [7, 2],
        )

        band = (ctypes.c_double * 6)(777, 2, 3, 4, 5, 6)
        self.assertEqual(
            runner.triangular_band_mv_expected(
                band, x, 3, 2, 1, "U", "N", "N"
            ),
            [8, 23, 18],
        )
        self.assertEqual(
            runner.triangular_band_mv_expected(
                band, x, 3, 2, 1, "U", "T", "N"
            ),
            [2, 11, 28],
        )

        rank_matrix = (ctypes.c_double * 3)(1, 2, 3)
        rank_x = (ctypes.c_double * 2)(1, 2)
        rank_y = (ctypes.c_double * 2)(3, 4)
        self.assertEqual(
            runner.packed_rank_expected(
                rank_matrix, rank_x, None, 2, 2, "U"
            ),
            [3, 6, 11],
        )
        self.assertEqual(
            runner.packed_rank_expected(
                rank_matrix, rank_x, rank_y, 2, 1, "U"
            ),
            [7, 12, 19],
        )

    def test_rank_update_references_preserve_unstored_triangle(self):
        n = 2
        real_matrix = (ctypes.c_double * 4)(1, 777, 2, 3)
        x = (ctypes.c_double * 2)(1, 2)
        y = (ctypes.c_double * 2)(3, 4)
        self.assertEqual(
            runner.real_rank_update_expected(real_matrix, x, None, n, n, 2, "U"),
            [3, 777, 6, 11],
        )
        self.assertEqual(
            runner.real_rank_update_expected(real_matrix, x, y, n, n, 1, "L"),
            [7, 787, 2, 19],
        )

        complex_matrix = (runner.ComplexF64 * 4)(
            runner.ComplexF64(1, 99),
            runner.ComplexF64(777, 777),
            runner.ComplexF64(2, 1),
            runner.ComplexF64(5, 88),
        )
        cx = (runner.ComplexF64 * 2)(
            runner.ComplexF64(1, 1), runner.ComplexF64(2, 0)
        )
        cy = (runner.ComplexF64 * 2)(
            runner.ComplexF64(3, 0), runner.ComplexF64(-1, 1)
        )
        expected = runner.complex_rank_update_expected(
            complex_matrix,
            cx,
            cy,
            n,
            n,
            runner.ComplexF64(1, 0),
            "U",
        )
        self.assertEqual(expected, [7 + 0j, 777 + 777j, 8 - 1j, 1 + 0j])

    def test_triangular_matrix_is_safe_for_solve(self):
        n = 9
        for uplo in ("U", "L"):
            matrix = runner.safe_triangular_matrix(ctypes.c_double, n, uplo, 1234)
            for row in range(n):
                off_diagonal_sum = 0.0
                for col in range(n):
                    if row == col:
                        continue
                    if (uplo == "U" and row < col) or (uplo == "L" and row > col):
                        off_diagonal_sum += abs(float(matrix[row + col * n]))
                self.assertLess(off_diagonal_sum, 1.0)
                self.assertGreater(abs(float(matrix[row + row * n])), off_diagonal_sum)

    @unittest.skipUnless(TEST_BLAS, "no drop-in BLAS library is available")
    def test_triangular_worker_correctness(self):
        n = 7
        result = subprocess.run(
            [
                sys.executable,
                str(TOOLS_DIR / "run_level2_report.py"),
                "--worker",
                "--csv",
                os.devnull,
                "--library-name",
                "TestBLAS",
                "--library-path",
                TEST_BLAS,
                "--worker-shape",
                "sq7",
                "--worker-m",
                str(n),
                "--worker-n",
                str(n),
                "--worker-reps",
                "1",
                "--worker-op",
                "triangular",
            ],
            cwd=REPO_ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        rows = list(csv.DictReader(result.stdout.splitlines()))
        self.assertEqual(len(rows), 80)
        identities = {
            (
                row["case"],
                row["kind"],
                row["uplo"],
                row["trans"],
                row["diag"],
                row["incx"],
            )
            for row in rows
        }
        self.assertEqual(len(identities), 80)
        for row in rows:
            self.assertEqual(row["shape"], "sq7")
            self.assertEqual(row["m"], str(n))
            self.assertEqual(row["n"], str(n))
            self.assertEqual(row["incx"], "1")
            self.assertEqual(row["status"], "ok", row)
            self.assertEqual(row["check_status"], "sampled-ok", row)
            case = next(
                case
                for case in runner.triangular_cases(runner.TRIANGULAR_OPERATIONS)
                if case.case == row["case"]
                and case.uplo == row["uplo"]
                and case.trans == row["trans"]
                and case.diag == row["diag"]
            )
            self.assertAlmostEqual(
                float(row["rate_gops"]),
                runner.triangular_work(case, n) / int(row["time_ns"]),
                places=5,
            )

    @unittest.skipUnless(TEST_BLAS, "no drop-in BLAS library is available")
    def test_rank_update_worker_correctness(self):
        n = 7
        result = subprocess.run(
            [
                sys.executable,
                str(TOOLS_DIR / "run_level2_report.py"),
                "--worker",
                "--csv",
                os.devnull,
                "--library-name",
                "TestBLAS",
                "--library-path",
                TEST_BLAS,
                "--worker-shape",
                "sq7",
                "--worker-m",
                str(n),
                "--worker-n",
                str(n),
                "--worker-reps",
                "1",
                "--worker-op",
                "rank-update",
            ],
            cwd=REPO_ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        rows = list(csv.DictReader(result.stdout.splitlines()))
        self.assertEqual(len(rows), 16)
        identities = {
            (
                row["case"],
                row["kind"],
                row["uplo"],
                row["incx"],
                row["incy"],
            )
            for row in rows
        }
        self.assertEqual(len(identities), 16)
        for row in rows:
            self.assertEqual(row["shape"], "sq7")
            self.assertEqual(row["m"], str(n))
            self.assertEqual(row["n"], str(n))
            self.assertEqual(row["incx"], "1")
            self.assertEqual(row["incy"], "1")
            self.assertEqual(row["status"], "ok", row)
            self.assertEqual(row["check_status"], "sampled-ok", row)
            case = next(
                case
                for case in runner.rank_update_cases(runner.RANK_UPDATE_OPERATIONS)
                if case.case == row["case"] and case.uplo == row["uplo"]
            )
            self.assertAlmostEqual(
                float(row["rate_gops"]),
                runner.rank_update_work(case, n) / int(row["time_ns"]),
                places=5,
            )

    @unittest.skipUnless(TEST_BLAS, "no drop-in BLAS library is available")
    def test_banded_worker_correctness(self):
        n = 7
        bandwidth = 2
        result = subprocess.run(
            [
                sys.executable,
                str(TOOLS_DIR / "run_level2_report.py"),
                "--worker",
                "--csv",
                os.devnull,
                "--library-name",
                "TestBLAS",
                "--library-path",
                TEST_BLAS,
                "--worker-shape",
                "band7_bw2",
                "--worker-m",
                str(n),
                "--worker-n",
                str(n),
                "--worker-bandwidth",
                str(bandwidth),
                "--worker-reps",
                "1",
                "--worker-op",
                "banded",
            ],
            cwd=REPO_ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        rows = list(csv.DictReader(result.stdout.splitlines()))
        self.assertEqual(len(rows), 18)
        identities = {
            (
                row["case"],
                row["kind"],
                row["storage"],
                row["lda"],
                row["k"],
                row["kl"],
                row["ku"],
                row["uplo"],
                row["trans"],
                row["incx"],
                row["incy"],
            )
            for row in rows
        }
        self.assertEqual(len(identities), 18)
        for row in rows:
            self.assertEqual(row["shape"], "band7_bw2")
            self.assertEqual(row["m"], str(n))
            self.assertEqual(row["n"], str(n))
            self.assertEqual(row["incx"], "1")
            self.assertEqual(row["incy"], "1")
            self.assertEqual(
                row["lda"],
                str(2 * bandwidth + 1 if row["storage"] == "general-band" else bandwidth + 1),
            )
            self.assertEqual(row["status"], "ok", row)
            self.assertEqual(row["check_status"], "sampled-ok", row)
            case = next(
                case
                for case in runner.banded_cases(runner.BANDED_OPERATIONS, bandwidth)
                if case.case == row["case"]
                and case.uplo == row["uplo"]
                and case.trans == row["trans"]
            )
            self.assertAlmostEqual(
                float(row["rate_gops"]),
                runner.banded_work(case, n) / int(row["time_ns"]),
                places=5,
            )

    @unittest.skipUnless(TEST_BLAS, "no drop-in BLAS library is available")
    def test_compact_worker_correctness(self):
        n = 5
        bandwidth = 2
        expected_counts = {
            "packed-mv": 88,
            "packed-rank": 16,
            "triangular-banded": 80,
        }
        for operation, expected_count in expected_counts.items():
            with self.subTest(operation=operation):
                command = [
                    sys.executable,
                    str(TOOLS_DIR / "run_level2_report.py"),
                    "--worker",
                    "--csv",
                    os.devnull,
                    "--library-name",
                    "TestBLAS",
                    "--library-path",
                    TEST_BLAS,
                    "--worker-shape",
                    "compact5",
                    "--worker-m",
                    str(n),
                    "--worker-n",
                    str(n),
                    "--worker-reps",
                    "1",
                    "--worker-op",
                    operation,
                ]
                if operation == "triangular-banded":
                    command.extend(("--worker-bandwidth", str(bandwidth)))
                result = subprocess.run(
                    command,
                    cwd=REPO_ROOT,
                    check=False,
                    capture_output=True,
                    text=True,
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                rows = list(csv.DictReader(result.stdout.splitlines()))
                self.assertEqual(len(rows), expected_count)
                identities = {
                    (
                        row["case"],
                        row["kind"],
                        row["storage"],
                        row["lda"],
                        row["k"],
                        row["uplo"],
                        row["trans"],
                        row["diag"],
                        row["incx"],
                        row["incy"],
                    )
                    for row in rows
                }
                self.assertEqual(len(identities), expected_count)
                for row in rows:
                    self.assertEqual(row["shape"], "compact5")
                    self.assertEqual(row["m"], str(n))
                    self.assertEqual(row["n"], str(n))
                    self.assertEqual(row["incx"], "1")
                    self.assertNotEqual(row["storage"], "")
                    self.assertEqual(row["status"], "ok", row)
                    self.assertEqual(row["check_status"], "sampled-ok", row)
                    if operation == "triangular-banded":
                        self.assertEqual(row["lda"], str(bandwidth + 1))
                        self.assertEqual(row["k"], str(bandwidth))
                    if (
                        operation == "packed-rank"
                        or row["case"] in runner.PACKED_STRUCTURED_MV_OPERATIONS
                    ):
                        self.assertEqual(row["incy"], "1")

    @unittest.skipUnless(TEST_BLAS, "no drop-in BLAS library is available")
    def test_rectangular_worker_correctness_and_operation_counts(self):
        for m, n in [(3, 5), (5, 3)]:
            with self.subTest(m=m, n=n):
                self.check_rectangular_worker(m, n)

    def check_rectangular_worker(self, m, n):
        shape = f"rect{m}x{n}"
        result = subprocess.run(
            [
                sys.executable,
                str(TOOLS_DIR / "run_level2_report.py"),
                "--worker",
                "--csv",
                os.devnull,
                "--library-name",
                "TestBLAS",
                "--library-path",
                TEST_BLAS,
                "--worker-shape",
                shape,
                "--worker-m",
                str(m),
                "--worker-n",
                str(n),
                "--worker-reps",
                "1",
            ],
            cwd=REPO_ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        rows = list(csv.DictReader(result.stdout.splitlines()))
        self.assertEqual(len(rows), 16)
        self.assertEqual(
            {row["case"] for row in rows},
            {
                "sgemv_n",
                "sgemv_t",
                "sger",
                "dgemv_n",
                "dgemv_t",
                "dger",
                "cgemv_n",
                "cgemv_t",
                "cgemv_c",
                "cgeru",
                "cgerc",
                "zgemv_n",
                "zgemv_t",
                "zgemv_c",
                "zgeru",
                "zgerc",
            },
        )
        for row in rows:
            self.assertEqual(row["shape"], shape)
            self.assertEqual(row["m"], str(m))
            self.assertEqual(row["n"], str(n))
            self.assertEqual(row["status"], "ok", row)
            self.assertEqual(row["check_status"], "sampled-ok", row)
            work = (8 if row["kind"].startswith("c") else 2) * m * n
            self.assertAlmostEqual(
                float(row["rate_gops"]),
                work / int(row["time_ns"]),
                places=5,
            )

    @unittest.skipUnless(TEST_BLAS, "no drop-in BLAS library is available")
    def test_controller_aggregates_independent_worker_processes(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "level2.csv"
            missing = str(Path(temp_dir) / "missing-blas")
            result = subprocess.run(
                [
                    sys.executable,
                    str(TOOLS_DIR / "run_level2_report.py"),
                    "--zynum",
                    TEST_BLAS,
                    "--accelerate",
                    missing,
                    "--openblas",
                    missing,
                    "--shape",
                    "rect3x2:3:2",
                    "--reps-small",
                    "1",
                    "--reps-large",
                    "1",
                    "--process-repeats",
                    "2",
                    "--skip-missing",
                    "--csv",
                    str(output),
                ],
                cwd=REPO_ROOT,
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            with output.open(newline="") as csv_file:
                rows = list(csv.DictReader(csv_file))
            metadata = json.loads(
                output.with_suffix(output.suffix + ".meta.json").read_text()
            )

        self.assertEqual(len(rows), 16)
        self.assertEqual(metadata["process_repeats"], 2)
        for row in rows:
            self.assertEqual(row["process_repeats"], "2")
            self.assertEqual(row["successful_repeats"], "2")
            samples = [float(value) for value in row["metric_samples"].split(",")]
            self.assertEqual(len(samples), 2)
            self.assertAlmostEqual(float(row["rate_gops"]), max(samples))
            self.assertLessEqual(float(row["metric_min"]), float(row["metric_median"]))
            self.assertLessEqual(float(row["metric_median"]), float(row["metric_max"]))

    @unittest.skipUnless(TEST_BLAS, "no drop-in BLAS library is available")
    def test_rank_update_controller_keeps_fresh_process_statistics(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "level2-rank-update.csv"
            missing = str(Path(temp_dir) / "missing-blas")
            result = subprocess.run(
                [
                    sys.executable,
                    str(TOOLS_DIR / "run_level2_report.py"),
                    "--zynum",
                    TEST_BLAS,
                    "--accelerate",
                    missing,
                    "--openblas",
                    missing,
                    "--shape",
                    "sq3:3:3",
                    "--op",
                    "rank-update",
                    "--reps-small",
                    "1",
                    "--process-repeats",
                    "2",
                    "--skip-missing",
                    "--csv",
                    str(output),
                ],
                cwd=REPO_ROOT,
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            with output.open(newline="") as csv_file:
                rows = list(csv.DictReader(csv_file))
            metadata = json.loads(
                output.with_suffix(output.suffix + ".meta.json").read_text()
            )

        self.assertEqual(len(rows), 16)
        self.assertEqual(metadata["operations"], list(runner.RANK_UPDATE_OPERATIONS))
        for row in rows:
            self.assertEqual(row["process_repeats"], "2")
            self.assertEqual(row["successful_repeats"], "2")
            self.assertEqual(len(row["metric_samples"].split(",")), 2)
            self.assertNotEqual(row["uplo"], "")
            self.assertEqual(row["incx"], "1")
            self.assertEqual(row["incy"], "1")

    @unittest.skipUnless(TEST_BLAS, "no drop-in BLAS library is available")
    def test_banded_controller_keeps_fresh_process_statistics(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "level2-banded.csv"
            missing = str(Path(temp_dir) / "missing-blas")
            result = subprocess.run(
                [
                    sys.executable,
                    str(TOOLS_DIR / "run_level2_report.py"),
                    "--zynum",
                    TEST_BLAS,
                    "--accelerate",
                    missing,
                    "--openblas",
                    missing,
                    "--op",
                    "banded",
                    "--band-profile",
                    "smoke3_bw1:3:1",
                    "--reps-small",
                    "1",
                    "--process-repeats",
                    "2",
                    "--skip-missing",
                    "--csv",
                    str(output),
                ],
                cwd=REPO_ROOT,
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            with output.open(newline="") as csv_file:
                rows = list(csv.DictReader(csv_file))
            metadata = json.loads(
                output.with_suffix(output.suffix + ".meta.json").read_text()
            )

        self.assertEqual(len(rows), 18)
        self.assertEqual(metadata["operations"], list(runner.BANDED_OPERATIONS))
        self.assertEqual(
            metadata["banded_profiles"],
            [{"name": "smoke3_bw1", "n": 3, "bandwidth": 1}],
        )
        for row in rows:
            self.assertEqual(row["process_repeats"], "2")
            self.assertEqual(row["successful_repeats"], "2")
            self.assertEqual(len(row["metric_samples"].split(",")), 2)
            self.assertNotEqual(row["storage"], "")
            self.assertEqual(row["incx"], "1")
            self.assertEqual(row["incy"], "1")

    @unittest.skipUnless(TEST_BLAS, "no drop-in BLAS library is available")
    def test_compact_controller_routes_profiles_and_metadata(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "level2-compact.csv"
            missing = str(Path(temp_dir) / "missing-blas")
            result = subprocess.run(
                [
                    sys.executable,
                    str(TOOLS_DIR / "run_level2_report.py"),
                    "--zynum",
                    TEST_BLAS,
                    "--accelerate",
                    missing,
                    "--openblas",
                    missing,
                    "--op",
                    "sspmv",
                    "--op",
                    "stbmv",
                    "--packed-profile",
                    "packed3:3",
                    "--band-profile",
                    "band3_bw1:3:1",
                    "--reps-small",
                    "1",
                    "--skip-missing",
                    "--csv",
                    str(output),
                ],
                cwd=REPO_ROOT,
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            with output.open(newline="") as csv_file:
                rows = list(csv.DictReader(csv_file))
            metadata = json.loads(
                output.with_suffix(output.suffix + ".meta.json").read_text()
            )

        self.assertEqual(len(rows), 10)
        self.assertEqual(metadata["operations"], ["sspmv", "stbmv"])
        self.assertEqual(
            metadata["packed_profiles"], [{"name": "packed3", "n": 3}]
        )
        self.assertEqual(
            metadata["banded_profiles"],
            [{"name": "band3_bw1", "n": 3, "bandwidth": 1}],
        )
        self.assertEqual({row["shape"] for row in rows}, {"packed3", "band3_bw1"})
        self.assertEqual({row["status"] for row in rows}, {"ok"})
        self.assertEqual({row["check_status"] for row in rows}, {"sampled-ok"})


class Level2AggregationTests(unittest.TestCase):
    def test_best_and_ordered_process_statistics(self):
        rows = runner.aggregate_worker_repeats(
            [
                [worker_row("Zynum", 2, 50)],
                [worker_row("Zynum", 4, 25)],
                [worker_row("Zynum", 3, 33)],
            ]
        )
        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual(row["rate_gops"], "4")
        self.assertEqual(row["time_ns"], "25")
        self.assertEqual(row["process_repeats"], 3)
        self.assertEqual(row["successful_repeats"], 3)
        self.assertEqual(row["metric_min"], "2")
        self.assertEqual(row["metric_median"], "3")
        self.assertEqual(row["metric_max"], "4")
        self.assertEqual(row["metric_samples"], "2,4,3")

    def test_correctness_failure_contaminates_aggregate(self):
        rows = runner.aggregate_worker_repeats(
            [
                [worker_row("Zynum", 2, 50)],
                [
                    worker_row(
                        "Zynum",
                        100,
                        1,
                        status="correctness_failed",
                        check_status="correctness_failed",
                        check_error="4.5",
                        check_raw="bad result",
                    )
                ],
            ]
        )
        row = rows[0]
        self.assertEqual(row["rate_gops"], "2")
        self.assertEqual(row["successful_repeats"], 1)
        self.assertEqual(row["status"], "correctness_failed")
        self.assertEqual(row["check_status"], "correctness_failed")
        self.assertEqual(row["check_max_abs_error"], "4.5")
        self.assertIn("repeat=1", row["check_raw_output"])

    def test_triangular_parameters_are_distinct_repeat_groups(self):
        rows = runner.aggregate_worker_repeats(
            [
                [
                    worker_row(
                        "Zynum",
                        2,
                        50,
                        case="strmv",
                        uplo="U",
                        trans="N",
                        diag="N",
                        incx="1",
                    ),
                    worker_row(
                        "Zynum",
                        3,
                        40,
                        case="strmv",
                        uplo="L",
                        trans="N",
                        diag="N",
                        incx="1",
                    ),
                ]
            ]
        )
        self.assertEqual(len(rows), 2)
        self.assertEqual({row["uplo"] for row in rows}, {"U", "L"})

    def test_rank_update_parameters_are_distinct_repeat_groups(self):
        rows = runner.aggregate_worker_repeats(
            [
                [
                    worker_row(
                        "Zynum",
                        2,
                        50,
                        case="ssyr2",
                        uplo="U",
                        incx="1",
                        incy="1",
                    ),
                    worker_row(
                        "Zynum",
                        3,
                        40,
                        case="ssyr2",
                        uplo="L",
                        incx="1",
                        incy="1",
                    ),
                ]
            ]
        )
        self.assertEqual(len(rows), 2)
        self.assertEqual({row["uplo"] for row in rows}, {"U", "L"})
        self.assertEqual({row["incy"] for row in rows}, {"1"})

    def test_banded_parameters_are_distinct_repeat_groups(self):
        rows = runner.aggregate_worker_repeats(
            [
                [
                    worker_row(
                        "Zynum",
                        2,
                        50,
                        case="sgbmv",
                        storage="general-band",
                        lda="3",
                        kl="1",
                        ku="1",
                        trans="N",
                        incx="1",
                        incy="1",
                    ),
                    worker_row(
                        "Zynum",
                        3,
                        40,
                        case="sgbmv",
                        storage="general-band",
                        lda="5",
                        kl="2",
                        ku="2",
                        trans="N",
                        incx="1",
                        incy="1",
                    ),
                ]
            ]
        )
        self.assertEqual(len(rows), 2)
        self.assertEqual({row["kl"] for row in rows}, {"1", "2"})
        self.assertEqual({row["ku"] for row in rows}, {"1", "2"})
        self.assertEqual({row["lda"] for row in rows}, {"3", "5"})


class Level2CheckerTests(unittest.TestCase):
    def run_checker(self, rows, fieldnames, *extra_args):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "level2.csv"
            with path.open("w", newline="") as output:
                writer = csv.DictWriter(output, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)
            stdout = io.StringIO()
            stderr = io.StringIO()
            with redirect_stdout(stdout), redirect_stderr(stderr):
                result = checker.main(
                    [str(path), "--comparator", "Reference", *extra_args]
                )
        return result, stdout.getvalue(), stderr.getvalue()

    @staticmethod
    def comparison_rows(shape, m, n, **parameters):
        common = {
            "case": "sgemv_n",
            "kind": "f32",
            "shape": shape,
            "m": str(m),
            "n": str(n),
            "metric": "gops",
            "status": "ok",
            "check_status": "sampled-ok",
            **parameters,
        }
        return [
            {**common, "library": "Zynum", "rate_gops": "2.0"},
            {**common, "library": "Reference", "rate_gops": "1.0"},
        ]

    def test_checker_groups_by_shape_m_and_n(self):
        rows = []
        rows.extend(self.comparison_rows("tall", 3, 2))
        rows.extend(self.comparison_rows("tall", 4, 2))
        rows.extend(self.comparison_rows("alias", 3, 2))
        result, stdout, stderr = self.run_checker(
            rows,
            [
                "case",
                "kind",
                "shape",
                "m",
                "n",
                "metric",
                "status",
                "check_status",
                "library",
                "rate_gops",
            ],
        )
        self.assertEqual(result, 0, stderr)
        self.assertIn("checked=3 passed=3 failed=0 missing=0", stdout)

    def test_checker_groups_and_filters_triangular_parameters(self):
        rows = []
        for uplo in ("U", "L"):
            for trans in ("N", "T"):
                for diag in ("N", "U"):
                    rows.extend(
                        self.comparison_rows(
                            "sq8",
                            8,
                            8,
                            case="strmv",
                            uplo=uplo,
                            trans=trans,
                            diag=diag,
                            incx="1",
                        )
                    )
        result, stdout, stderr = self.run_checker(
            rows,
            runner.CSV_FIELDNAMES,
        )
        self.assertEqual(result, 0, stderr)
        self.assertIn("checked=8 passed=8 failed=0 missing=0", stdout)

        result, stdout, stderr = self.run_checker(
            rows,
            runner.CSV_FIELDNAMES,
            "--uplo",
            "L",
            "--trans",
            "T",
            "--diag",
            "U",
            "--incx",
            "1",
        )
        self.assertEqual(result, 0, stderr)
        self.assertIn("checked=1 passed=1 failed=0 missing=0", stdout)

    def test_checker_groups_and_filters_rank_update_parameters(self):
        rows = []
        for case, kind in [
            ("ssyr", "f32"),
            ("dsyr", "f64"),
            ("cher", "c32"),
            ("zher", "c64"),
            ("ssyr2", "f32"),
            ("dsyr2", "f64"),
            ("cher2", "c32"),
            ("zher2", "c64"),
        ]:
            for uplo in ("U", "L"):
                rows.extend(
                    self.comparison_rows(
                        "sq8",
                        8,
                        8,
                        case=case,
                        kind=kind,
                        uplo=uplo,
                        incx="1",
                        incy="1",
                    )
                )
        result, stdout, stderr = self.run_checker(
            rows,
            runner.CSV_FIELDNAMES,
        )
        self.assertEqual(result, 0, stderr)
        self.assertIn("checked=16 passed=16 failed=0 missing=0", stdout)

        result, stdout, stderr = self.run_checker(
            rows,
            runner.CSV_FIELDNAMES,
            "--case",
            "cher2",
            "--uplo",
            "L",
            "--incx",
            "1",
            "--incy",
            "1",
        )
        self.assertEqual(result, 0, stderr)
        self.assertIn("checked=1 passed=1 failed=0 missing=0", stdout)

    def test_checker_groups_and_filters_banded_parameters(self):
        rows = []
        for case in runner.banded_cases(runner.BANDED_OPERATIONS, 8):
            rows.extend(
                self.comparison_rows(
                    "n512_bw8",
                    512,
                    512,
                    case=case.case,
                    kind=case.kind,
                    storage=case.storage,
                    lda=str(case.lda),
                    k=str(case.k),
                    kl=str(case.kl),
                    ku=str(case.ku),
                    uplo=case.uplo,
                    trans=case.trans,
                    incx=str(case.incx),
                    incy=str(case.incy),
                )
            )
        result, stdout, stderr = self.run_checker(
            rows,
            runner.CSV_FIELDNAMES,
        )
        self.assertEqual(result, 0, stderr)
        self.assertIn("checked=18 passed=18 failed=0 missing=0", stdout)

        result, stdout, stderr = self.run_checker(
            rows,
            runner.CSV_FIELDNAMES,
            "--storage",
            "general-band",
            "--lda",
            "17",
            "--kl",
            "8",
            "--ku",
            "8",
            "--trans",
            "C",
            "--incy",
            "1",
        )
        self.assertEqual(result, 0, stderr)
        self.assertIn("checked=2 passed=2 failed=0 missing=0", stdout)

        result, stdout, stderr = self.run_checker(
            rows,
            runner.CSV_FIELDNAMES,
            "--storage",
            "hermitian-band",
            "--k",
            "8",
            "--uplo",
            "L",
        )
        self.assertEqual(result, 0, stderr)
        self.assertIn("checked=2 passed=2 failed=0 missing=0", stdout)

    def test_checker_groups_and_filters_packed_and_triangular_band(self):
        rows = []
        operations = list(runner.PACKED_OPERATIONS)
        compact_cases = []
        compact_cases.extend(runner.packed_structured_mv_cases(operations))
        compact_cases.extend(runner.packed_triangular_cases(operations))
        compact_cases.extend(runner.packed_rank_cases(operations))
        compact_cases.extend(
            runner.triangular_banded_cases(
                runner.TRIANGULAR_BANDED_OPERATIONS, 8
            )
        )
        for case in compact_cases:
            rows.extend(
                self.comparison_rows(
                    "compact8",
                    8,
                    8,
                    case=case.case,
                    kind=case.kind,
                    storage=case.storage,
                    lda=str(getattr(case, "lda", "")),
                    k=str(getattr(case, "k", "")),
                    uplo=case.uplo,
                    trans=getattr(case, "trans", ""),
                    diag=getattr(case, "diag", ""),
                    incx=str(case.incx),
                    incy=str(getattr(case, "incy", "")),
                )
            )
        result, stdout, stderr = self.run_checker(
            rows,
            runner.CSV_FIELDNAMES,
        )
        self.assertEqual(result, 0, stderr)
        self.assertIn("checked=184 passed=184 failed=0 missing=0", stdout)

        result, stdout, stderr = self.run_checker(
            rows,
            runner.CSV_FIELDNAMES,
            "--case",
            "ztpsv",
            "--storage",
            "triangular-packed",
            "--uplo",
            "L",
            "--trans",
            "C",
            "--diag",
            "U",
            "--incx",
            "1",
        )
        self.assertEqual(result, 0, stderr)
        self.assertIn("checked=1 passed=1 failed=0 missing=0", stdout)

        result, stdout, stderr = self.run_checker(
            rows,
            runner.CSV_FIELDNAMES,
            "--case",
            "ctbmv",
            "--storage",
            "triangular-band",
            "--lda",
            "9",
            "--k",
            "8",
            "--uplo",
            "U",
            "--trans",
            "C",
            "--diag",
            "N",
        )
        self.assertEqual(result, 0, stderr)
        self.assertIn("checked=1 passed=1 failed=0 missing=0", stdout)

    def test_checker_median_gate_differs_from_best(self):
        rows = self.comparison_rows("rect", 3, 2)
        rows[0].update(
            {
                "rate_gops": "10",
                "metric_min": "1",
                "metric_median": "2",
                "metric_max": "10",
                "metric_samples": "1,10,2",
            }
        )
        rows[1].update(
            {
                "rate_gops": "8",
                "metric_min": "3",
                "metric_median": "4",
                "metric_max": "8",
                "metric_samples": "3,8,4",
            }
        )
        best_result, best_stdout, best_stderr = self.run_checker(
            rows, runner.CSV_FIELDNAMES
        )
        median_result, median_stdout, median_stderr = self.run_checker(
            rows, runner.CSV_FIELDNAMES, "--stat", "median"
        )
        self.assertEqual(best_result, 0, best_stderr)
        self.assertIn("passed=1 failed=0", best_stdout)
        self.assertEqual(median_result, 1, median_stderr)
        self.assertIn("passed=0 failed=1", median_stdout)
        self.assertIn("stat=median", median_stdout)

    def test_checker_rejects_correctness_polluted_aggregate(self):
        polluted = runner.aggregate_worker_repeats(
            [
                [worker_row("Zynum", 2, 50)],
                [
                    worker_row(
                        "Zynum",
                        100,
                        1,
                        status="correctness_failed",
                        check_status="correctness_failed",
                    )
                ],
            ]
        )[0]
        reference = runner.aggregate_worker_repeats(
            [[worker_row("Reference", 1, 100)], [worker_row("Reference", 1, 100)]]
        )[0]
        result, stdout, stderr = self.run_checker(
            [polluted, reference], runner.CSV_FIELDNAMES
        )
        self.assertEqual(result, 2, stdout)
        self.assertIn("unchecked Level 2 row", stderr)

    def test_checker_accepts_legacy_square_csv(self):
        rows = self.comparison_rows("unused", 8, 8)
        for row in rows:
            row.pop("shape")
            row.pop("m")
        result, stdout, stderr = self.run_checker(
            rows,
            [
                "case",
                "kind",
                "n",
                "metric",
                "status",
                "check_status",
                "library",
                "rate_gops",
            ],
            "--shape",
            "sq8",
            "--m",
            "8",
        )
        self.assertEqual(result, 0, stderr)
        self.assertIn("checked=1 passed=1 failed=0 missing=0", stdout)


if __name__ == "__main__":
    unittest.main()
