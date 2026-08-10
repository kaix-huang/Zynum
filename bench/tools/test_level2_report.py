#!/usr/bin/env python3
# Copyright (C) 2026 Zynum contributors
# SPDX-License-Identifier: LGPL-3.0-or-later

import csv
import ctypes
import ctypes.util
import hashlib
import importlib.util
import io
import json
import os
import stat
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest import mock

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

WINDOWS_TEST_BLAS_WINMODE = 0x00000900
WINDOWS_TEST_BLAS_REQUIRED_SYMBOLS = (
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
)


def _windows_file_snapshot(path):
    try:
        if not path.is_absolute() or path.resolve(strict=True) != path:
            return None
        is_junction = getattr(path, "is_junction", None)
        if path.is_symlink() or (is_junction is not None and is_junction()):
            return None
        path_stat = os.stat(path, follow_symlinks=False)
        attributes = getattr(path_stat, "st_file_attributes", 0)
        if (
            not stat.S_ISREG(path_stat.st_mode)
            or path_stat.st_size <= 0
            or path_stat.st_nlink != 1
            or attributes & 0x00000400
        ):
            return None
        digest = hashlib.sha256()
        with path.open("rb") as source:
            descriptor_before = os.fstat(source.fileno())
            remaining = descriptor_before.st_size
            while remaining:
                chunk = source.read(min(1024 * 1024, remaining))
                if not chunk:
                    return None
                digest.update(chunk)
                remaining -= len(chunk)
            if source.read(1):
                return None
            descriptor_after = os.fstat(source.fileno())
        path_after = os.stat(path, follow_symlinks=False)
    except OSError:
        return None

    def identity(value):
        return (
            value.st_dev,
            value.st_ino,
            value.st_mode,
            value.st_nlink,
            value.st_size,
            value.st_mtime_ns,
            value.st_ctime_ns,
            getattr(value, "st_file_attributes", 0),
        )

    frozen = identity(path_stat)
    if (
        frozen != identity(descriptor_before)
        or frozen != identity(descriptor_after)
        or frozen != identity(path_after)
    ):
        return None
    return (*frozen, digest.hexdigest())


def _windows_test_blas_identity(path):
    return _windows_file_snapshot(path)


def find_test_blas():
    local_zynum = REPO_ROOT / runner.default_zynum_blas()
    if sys.platform == "win32":
        identity = _windows_test_blas_identity(local_zynum)
        if identity is None:
            return None, None
        try:
            library = ctypes.CDLL(str(local_zynum), winmode=WINDOWS_TEST_BLAS_WINMODE)
        except OSError:
            return None, None
        if _windows_test_blas_identity(local_zynum) != identity or not all(
            hasattr(library, symbol) for symbol in WINDOWS_TEST_BLAS_REQUIRED_SYMBOLS
        ):
            return None, None
        return str(local_zynum), library

    candidates = [
        runner.DEFAULT_ACCELERATE,
        ctypes.util.find_library("blas"),
        str(local_zynum),
        runner.DEFAULT_OPENBLAS,
    ]
    for candidate in candidates:
        if not candidate:
            continue
        try:
            library = ctypes.CDLL(candidate)
        except OSError:
            continue
        if all(
            hasattr(library, symbol) for symbol in WINDOWS_TEST_BLAS_REQUIRED_SYMBOLS
        ):
            return candidate, library
    return None, None


TEST_BLAS, _TEST_BLAS_LIBRARY = find_test_blas()
TEST_FILE_BLAS = TEST_BLAS if TEST_BLAS and Path(TEST_BLAS).is_file() else None


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
    def test_library_availability_never_loads_a_live_path(self):
        with mock.patch.object(runner.ctypes, "CDLL") as loader:
            self.assertFalse(
                runner.library_available("Comparator", "lib-not-a-file.so")
            )
        loader.assert_not_called()

    @staticmethod
    def unknown_identity():
        return {
            "source": {
                "revision": None,
                "branch": None,
                "dirty": None,
                "status_short": None,
            }
        }

    @classmethod
    def identity_for_libraries(cls, libraries):
        identity = cls.unknown_identity()
        identity["payload"] = {
            "artifacts": {
                "libraries": [
                    {"name": name, "path": path, "sha256": None}
                    for name, path in libraries
                ]
            }
        }
        return identity

    def run_publication_controller(self, output, *, identity=None, publish=None):
        artifact_dir = output.parent if output.parent.exists() else output.parent.parent
        library_path = artifact_dir / "libzynum-test.so"
        library_path.write_bytes(b"library-a")
        args = runner.parse_args(
            ["--csv", str(output), "--shape", "smoke:2:2", "--op", "legacy"]
        )
        completed = subprocess.CompletedProcess(
            [], 0, stdout="case\nsgemv_n\n", stderr=""
        )
        aggregate = worker_row("Zynum", 2, 3, case="sgemv_n")
        aggregate.update(
            {
                "shape": "smoke",
                "m": "2",
                "process_repeats": "1",
                "successful_repeats": "1",
                "metric_min": "2",
                "metric_median": "2",
                "metric_max": "2",
                "metric_samples": "2",
            }
        )
        with (
            mock.patch.object(
                runner, "libraries", return_value=[("Zynum", str(library_path))]
            ),
            mock.patch.object(runner, "run_one_process", return_value=completed),
            mock.patch.object(
                runner, "aggregate_worker_repeats", return_value=[aggregate]
            ),
            mock.patch.object(runner, "zig_version", return_value=None),
            mock.patch.object(
                runner.benchmark_metadata,
                "collect_benchmark_identity_from_frozen",
                return_value=identity
                or self.identity_for_libraries([("Zynum", str(library_path))]),
            ),
            mock.patch.object(
                runner, "publish_outputs", side_effect=publish
            ) as publisher,
            redirect_stderr(io.StringIO()),
        ):
            runner.run_controller(args)
        return publisher

    def test_controller_publishes_one_ordered_immutable_pair(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            parent = Path(temp_dir).resolve() / "absent"
            output = parent / "level2.csv"

            def assert_prepublication(outputs):
                self.assertFalse(parent.exists())
                self.assertEqual(
                    [item.path for item in outputs],
                    [output, output.with_suffix(".csv.meta.json")],
                )
                self.assertTrue(all(type(item.contents) is bytes for item in outputs))
                self.assertTrue(outputs[0].contents.endswith(b"\r\n"))
                self.assertTrue(outputs[1].contents.endswith(b"\n"))
                self.assertNotIn(b"\r\n", outputs[1].contents)
                self.assertEqual(
                    outputs[0].contents.split(b"\r\n", 1)[0].decode("utf-8").split(","),
                    runner.CSV_FIELDNAMES,
                )
                rows = list(
                    csv.DictReader(io.StringIO(outputs[0].contents.decode("utf-8")))
                )
                self.assertEqual(
                    [(row["library"], row["case"]) for row in rows],
                    [("Zynum", "sgemv_n")],
                )
                self.assertEqual(
                    json.loads(outputs[1].contents)["shapes"],
                    [{"m": 2, "n": 2, "name": "smoke"}],
                )

            publisher = self.run_publication_controller(
                output, publish=assert_prepublication
            )
            publisher.assert_called_once()
            self.assertFalse(parent.exists())

    def test_metadata_serialization_failure_precedes_publication(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir).resolve() / "level2.csv"
            metadata = output.with_suffix(".csv.meta.json")
            output.write_bytes(b"previous csv\n")
            metadata.write_bytes(b"previous metadata\n")
            identity = self.identity_for_libraries([("Zynum", "public")])
            identity["invalid"] = float("nan")
            publish = mock.Mock()

            with self.assertRaises(ValueError):
                self.run_publication_controller(
                    output, identity=identity, publish=publish
                )

            publish.assert_not_called()
            self.assertEqual(output.read_bytes(), b"previous csv\n")
            self.assertEqual(metadata.read_bytes(), b"previous metadata\n")

    def test_publisher_failure_receives_one_batch_and_cannot_split_pair(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir).resolve() / "level2.csv"
            metadata = output.with_suffix(".csv.meta.json")
            output.write_bytes(b"previous csv\n")
            metadata.write_bytes(b"previous metadata\n")
            calls = []

            def fail(outputs):
                calls.append(outputs)
                raise OSError("injected publication failure")

            with self.assertRaisesRegex(OSError, "injected publication failure"):
                self.run_publication_controller(output, publish=fail)

            self.assertEqual(len(calls), 1)
            self.assertEqual(len(calls[0]), 2)
            self.assertEqual(output.read_bytes(), b"previous csv\n")
            self.assertEqual(metadata.read_bytes(), b"previous metadata\n")

    def test_controller_reuses_one_frozen_library_and_script_without_private_leaks(
        self,
    ):
        with tempfile.TemporaryDirectory() as temp_dir:
            library = Path(temp_dir) / "libzynum.so"
            library.write_bytes(b"library-a")
            output = Path(temp_dir) / "level2.csv"
            args = runner.parse_args(
                [
                    "--csv",
                    str(output),
                    "--shape",
                    "smoke:2:2",
                    "--op",
                    "legacy",
                    "--process-repeats",
                    "3",
                ]
            )
            private_paths = []
            identities = []
            capture = runner.benchmark_artifacts.ArtifactSnapshotSet.capture

            def collect_identity(_args, *, libraries=(), binaries=(), **_kwargs):
                identities.append(
                    {
                        "libraries": [item.metadata_record() for item in libraries],
                        "binaries": [item.metadata_record() for item in binaries],
                    }
                )
                identity = self.unknown_identity()
                identity["payload"] = {"artifacts": identities[-1]}
                return identity

            def run_one(script, _name, library_path, *_args, **_kwargs):
                private_paths.extend((str(script), str(library_path)))
                self.assertNotEqual(str(script), str(Path(runner.__file__)))
                self.assertNotEqual(str(library_path), str(library))
                self.assertEqual(
                    Path(script).read_bytes(), Path(runner.__file__).read_bytes()
                )
                self.assertEqual(Path(library_path).read_bytes(), b"library-a")
                if len(private_paths) == 2:
                    library.write_bytes(b"library-b")
                elif len(private_paths) == 4:
                    library.write_bytes(b"library-a")
                row = worker_row(
                    "Zynum",
                    2,
                    3,
                    check_raw="{} {}".format(script, library_path),
                )
                row.update({"shape": "smoke", "m": "2", "n": "2"})
                buffer = io.StringIO(newline="")
                writer = csv.DictWriter(buffer, fieldnames=runner.CSV_FIELDNAMES)
                writer.writeheader()
                writer.writerow(row)
                return subprocess.CompletedProcess([], 0, buffer.getvalue(), "")

            def publish(outputs):
                self.assertEqual(len(private_paths), 6)
                for path in private_paths:
                    self.assertFalse(Path(path).exists())
                    self.assertNotIn(path.encode(), outputs[0].contents)
                    self.assertNotIn(path.encode(), outputs[1].contents)
                csv_rows = list(
                    csv.DictReader(io.StringIO(outputs[0].contents.decode("utf-8")))
                )
                self.assertEqual(len(csv_rows), 1)
                self.assertIn(str(library), csv_rows[0]["check_raw_output"])
                metadata = json.loads(outputs[1].contents)
                expected_sha = hashlib.sha256(b"library-a").hexdigest()
                self.assertNotIn("path", metadata["libraries"][0])
                self.assertEqual(metadata["libraries"][0]["sha256"], expected_sha)
                self.assertEqual(
                    metadata["benchmark_identity"]["payload"]["artifacts"]["libraries"][
                        0
                    ]["sha256"],
                    expected_sha,
                )

            with (
                mock.patch.object(
                    runner, "libraries", return_value=[("Zynum", str(library))]
                ),
                mock.patch.object(
                    runner.benchmark_artifacts.ArtifactSnapshotSet,
                    "capture",
                    wraps=capture,
                ) as capture_snapshot,
                mock.patch.object(runner, "run_one_process", side_effect=run_one),
                mock.patch.object(
                    runner.benchmark_metadata,
                    "collect_benchmark_identity_from_frozen",
                    side_effect=collect_identity,
                ),
                mock.patch.object(runner, "publish_outputs", side_effect=publish),
                mock.patch.object(runner, "zig_version", return_value=None),
                redirect_stderr(io.StringIO()),
            ):
                runner.run_controller(args)

            capture_snapshot.assert_called_once()
            self.assertEqual(len(identities), 1)
            self.assertEqual(library.read_bytes(), b"library-a")

    def test_stage_drift_and_cleanup_failure_never_publish_or_replace_outputs(self):
        failure_modes = ("finalize", "close")
        for failure_mode in failure_modes:
            with self.subTest(failure_mode=failure_mode):
                with tempfile.TemporaryDirectory() as temp_dir:
                    library = Path(temp_dir) / "libzynum.so"
                    library.write_bytes(b"library-a")
                    output = Path(temp_dir) / "level2.csv"
                    metadata = output.with_suffix(".csv.meta.json")
                    output.write_bytes(b"old csv\n")
                    metadata.write_bytes(b"old metadata\n")
                    args = runner.parse_args(
                        [
                            "--csv",
                            str(output),
                            "--shape",
                            "smoke:2:2",
                            "--op",
                            "legacy",
                        ]
                    )
                    private_library = []
                    original_finalize = (
                        runner.benchmark_artifacts.ArtifactSnapshotSet.finalize
                    )
                    original_close = (
                        runner.benchmark_artifacts.ArtifactSnapshotSet.close
                    )

                    def run_one(_script, _name, library_path, *_args, **_kwargs):
                        private_library.append(str(library_path))
                        return subprocess.CompletedProcess([], 0, "case\nsgemv_n\n", "")

                    def drift(snapshot):
                        os.chmod(private_library[0], 0o600)
                        descriptor = os.open(private_library[0], os.O_WRONLY)
                        try:
                            os.pwrite(descriptor, b"B", 0)
                        finally:
                            os.close(descriptor)
                        original_finalize(snapshot)

                    def close_then_fail(snapshot):
                        original_close(snapshot)
                        raise runner.benchmark_artifacts.ArtifactCleanupError(
                            (
                                runner.benchmark_artifacts.CleanupIssue(
                                    "injected_cleanup_failure", None
                                ),
                            )
                        )

                    def capture_in_temp(requests, **limits):
                        return runner.benchmark_artifacts.ArtifactSnapshotSet(
                            requests,
                            private_parent=temp_dir,
                            **limits,
                        )

                    identity = self.identity_for_libraries([("Zynum", str(library))])
                    patches = [
                        mock.patch.object(
                            runner,
                            "libraries",
                            return_value=[("Zynum", str(library))],
                        ),
                        mock.patch.object(
                            runner, "run_one_process", side_effect=run_one
                        ),
                        mock.patch.object(
                            runner, "aggregate_worker_repeats", return_value=[]
                        ),
                        mock.patch.object(
                            runner.benchmark_metadata,
                            "collect_benchmark_identity_from_frozen",
                            return_value=identity,
                        ),
                        mock.patch.object(runner, "publish_outputs"),
                        mock.patch.object(runner, "zig_version", return_value=None),
                        mock.patch.object(
                            runner.benchmark_artifacts.ArtifactSnapshotSet,
                            "capture",
                            side_effect=capture_in_temp,
                        ),
                    ]
                    if failure_mode == "finalize":
                        patches.append(
                            mock.patch.object(
                                runner.benchmark_artifacts.ArtifactSnapshotSet,
                                "finalize",
                                new=drift,
                            )
                        )
                    else:
                        patches.append(
                            mock.patch.object(
                                runner.benchmark_artifacts.ArtifactSnapshotSet,
                                "close",
                                new=close_then_fail,
                            )
                        )
                    with (
                        patches[0] as _,
                        patches[1] as _,
                        patches[2] as _,
                        patches[3] as _,
                        patches[4] as publisher,
                        patches[5] as _,
                        patches[6] as _,
                        patches[7] as _,
                        redirect_stderr(io.StringIO()),
                    ):
                        with self.assertRaises(
                            runner.benchmark_artifacts.ArtifactCleanupError
                        ) as raised:
                            runner.run_controller(args)

                    publisher.assert_not_called()
                    self.assertEqual(output.read_bytes(), b"old csv\n")
                    self.assertEqual(metadata.read_bytes(), b"old metadata\n")
                    self.assertFalse(Path(private_library[0]).exists())
                    if failure_mode == "finalize":
                        self.assertIsInstance(
                            raised.exception.__context__,
                            runner.benchmark_artifacts.ArtifactVerificationError,
                        )
                        retained = [
                            Path(path)
                            for path in raised.exception.recovery_paths
                            if Path(path).name == Path(private_library[0]).name
                            and Path(path).is_file()
                        ]
                        self.assertEqual(len(retained), 1)
                        drifted = retained[0]
                        self.assertTrue(
                            drifted.parent.name.startswith(
                                ".zynum-benchmark-artifact-quarantine-"
                            )
                        )
                        self.assertEqual(drifted.read_bytes(), b"Bibrary-a")
                        self.assertIn(
                            "private_artifact_replaced",
                            {issue.code for issue in raised.exception.issues},
                        )
                        drifted.unlink()
                        drifted.parent.rmdir()
                    else:
                        self.assertFalse(raised.exception.recovery_paths)

    def run_recording_worker(self, selectors, *, m=2, n=2):
        class RecordingLibrary:
            def __init__(self):
                self.calls = []

            def __getattr__(self, name):
                def operation(*_args):
                    self.calls.append(name)

                return operation

        def checked_vector(call, setup, *_args, **_kwargs):
            setup()
            call()
            return {
                "check_status": "sampled-ok",
                "check_max_abs_error": "0",
                "check_raw_output": "",
            }

        def best_time(call, setup, _reps):
            setup()
            call()
            return 1

        library = RecordingLibrary()
        argv = [
            "--worker",
            "--csv",
            os.devnull,
            "--library-name",
            "recording",
            "--library-path",
            "unused",
            "--worker-m",
            str(m),
            "--worker-n",
            str(n),
            "--worker-reps",
            "1",
        ]
        for selector in selectors:
            argv.extend(("--worker-op", selector))
        if any(selector == "all" for selector in selectors):
            argv.extend(("--worker-bandwidth", "0"))
        args = runner.parse_args(argv)
        stdout = io.StringIO()
        with (
            mock.patch.object(runner.ctypes, "CDLL", return_value=library),
            mock.patch.object(runner, "checked_vector", checked_vector),
            mock.patch.object(runner, "best_time", best_time),
            redirect_stdout(stdout),
        ):
            runner.run_worker(args)
        rows = list(csv.DictReader(stdout.getvalue().splitlines()))
        return library.calls, rows

    def test_repeated_square_and_rectangular_cli_shapes(self):
        default_args = runner.parse_args(["--csv", os.devnull])
        self.assertEqual(default_args.process_repeats, 1)
        self.assertEqual(default_args.process_schedule, "library-major")
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

    def test_focused_legacy_backed_selectors_expand_without_legacy(self):
        args = runner.parse_args(
            [
                "--csv",
                os.devnull,
                "--shape",
                "rect:256:128",
                "--op",
                "complex-ger",
            ]
        )
        operations = runner.requested_operations(args)
        self.assertEqual(
            operations,
            ["cgeru", "zgeru", "cgerc", "zgerc"],
        )
        self.assertEqual(
            runner.operations_for_shape(operations, runner.Shape("rect", 256, 128)),
            operations,
        )
        rows = [
            {"case": "sgemv_n"},
            {"case": "cgeru"},
            {"case": "zgerc"},
        ]
        self.assertEqual(
            runner.filter_legacy_rows(rows, operations),
            [{"case": "cgeru"}, {"case": "zgerc"}],
        )

        mixed = operations + ["strmv", "dtrmv"]
        mixed_rows = rows + [{"case": "strmv"}, {"case": "dtrmv"}]
        self.assertEqual(
            runner.filter_legacy_rows(mixed_rows, mixed),
            [
                {"case": "cgeru"},
                {"case": "zgerc"},
                {"case": "strmv"},
                {"case": "dtrmv"},
            ],
        )

    def test_rectangular_shapes_filter_square_only_symmetric_selectors(self):
        rectangular = runner.Shape("rect", 3, 2)
        for selector in (*runner.SYMMETRIC_MV_OPERATIONS, "symmetric-mv"):
            with self.subTest(selector=selector):
                operations = runner.expand_operations([selector])
                self.assertEqual(
                    runner.operations_for_shape(operations, rectangular), []
                )

        mixed = runner.expand_operations(["symmetric-mv", "complex-ger"])
        self.assertEqual(
            runner.operations_for_shape(mixed, rectangular),
            list(runner.COMPLEX_GER_OPERATIONS),
        )
        self.assertEqual(
            runner.operations_for_shape(
                runner.expand_operations(["symmetric-mv", "legacy"]), rectangular
            ),
            ["legacy"],
        )
        self.assertEqual(
            runner.operations_for_shape(runner.expand_operations(["all"]), rectangular),
            ["legacy"],
        )

    def test_rectangular_symmetric_jobs_skip_or_run_only_compatible_union(self):
        for selector in (*runner.SYMMETRIC_MV_OPERATIONS, "symmetric-mv"):
            with self.subTest(selector=selector):
                with tempfile.TemporaryDirectory() as temp_dir:
                    output = Path(temp_dir).resolve() / "level2.csv"
                    library = Path(temp_dir) / "libzynum.so"
                    library.write_bytes(b"library-a")
                    args = runner.parse_args(
                        [
                            "--csv",
                            str(output),
                            "--shape",
                            "rect:3:2",
                            "--op",
                            selector,
                        ]
                    )
                    stderr = io.StringIO()
                    with (
                        mock.patch.object(
                            runner,
                            "libraries",
                            return_value=[("Zynum", str(library))],
                        ),
                        mock.patch.object(
                            runner.benchmark_metadata,
                            "collect_benchmark_identity_from_frozen",
                            return_value=self.unknown_identity(),
                        ),
                        mock.patch.object(runner, "run_one_process") as run_process,
                        redirect_stderr(stderr),
                    ):
                        runner.run_controller(args)
                    run_process.assert_not_called()
                    self.assertIn(
                        "skipping square-only selected operations on a non-square shape",
                        stderr.getvalue(),
                    )
                    with output.open(newline="") as csv_file:
                        self.assertEqual(list(csv.DictReader(csv_file)), [])

        operations = runner.operations_for_shape(
            runner.expand_operations(["symmetric-mv", "complex-ger"]),
            runner.Shape("rect", 3, 2),
        )
        calls, rows = self.run_recording_worker(operations, m=3, n=2)
        self.assertEqual(set(calls), {"cgeru_", "zgeru_", "cgerc_", "zgerc_"})
        self.assertEqual(
            {row["case"] for row in rows}, set(runner.COMPLEX_GER_OPERATIONS)
        )

        rectangular_legacy_symbols = {
            "sgemv_",
            "dgemv_",
            "sger_",
            "dger_",
            "cgemv_",
            "zgemv_",
            "cgeru_",
            "zgeru_",
            "cgerc_",
            "zgerc_",
        }
        rectangular_legacy_cases = {
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
        }
        for selectors in ((), ("legacy",), ("all",)):
            with self.subTest(rectangular_full_selectors=selectors):
                shape_operations = runner.operations_for_shape(
                    runner.expand_operations(list(selectors)),
                    runner.Shape("rect", 3, 2),
                )
                self.assertEqual(shape_operations, ["legacy"])
                calls, rows = self.run_recording_worker(shape_operations, m=3, n=2)
                self.assertEqual(set(calls), rectangular_legacy_symbols)
                self.assertEqual(
                    {row["case"] for row in rows}, rectangular_legacy_cases
                )

    def test_operation_expansion_preserves_order_dedup_and_explicit_full_modes(self):
        self.assertEqual(runner.expand_operations([]), ["legacy"])
        self.assertEqual(runner.expand_operations(["legacy"]), ["legacy"])
        self.assertEqual(
            runner.expand_operations(["all"]),
            list(runner.OP_EXPANSIONS["all"]),
        )
        self.assertEqual(
            runner.expand_operations(
                [
                    "complex-ger",
                    "cgeru",
                    "symmetric-mv",
                    "ssymv",
                    "trmv",
                    "complex-ger",
                ]
            ),
            [
                *runner.COMPLEX_GER_OPERATIONS,
                *runner.SYMMETRIC_MV_OPERATIONS,
                "strmv",
                "dtrmv",
                "ctrmv",
                "ztrmv",
            ],
        )
        self.assertEqual(
            runner.expand_operations(["cgeru", "legacy", "cgeru"]),
            ["cgeru", "legacy"],
        )
        legacy_rows = [
            {"case": "sgemv_n"},
            {"case": "cgeru"},
            {"case": "ssymv"},
        ]
        focused_selectors = (
            *runner.COMPLEX_GER_OPERATIONS,
            *runner.SYMMETRIC_MV_OPERATIONS,
            "complex-ger",
            "symmetric-mv",
        )
        for full_selector in ("legacy", "all"):
            full_expansion = runner.expand_operations([full_selector])
            for focused_selector in focused_selectors:
                with self.subTest(
                    full_selector=full_selector,
                    focused_selector=focused_selector,
                ):
                    focused_expansion = runner.expand_operations([focused_selector])
                    forward = runner.expand_operations(
                        [full_selector, focused_selector, full_selector]
                    )
                    reverse = runner.expand_operations(
                        [focused_selector, full_selector, focused_selector]
                    )
                    self.assertEqual(
                        forward,
                        runner.unique_preserving_order(
                            full_expansion + focused_expansion
                        ),
                    )
                    self.assertEqual(
                        reverse,
                        runner.unique_preserving_order(
                            focused_expansion + full_expansion
                        ),
                    )
                    self.assertEqual(len(forward), len(set(forward)))
                    self.assertEqual(len(reverse), len(set(reverse)))
                    self.assertEqual(
                        runner.filter_legacy_rows(legacy_rows, forward), legacy_rows
                    )
                    self.assertEqual(
                        runner.filter_legacy_rows(legacy_rows, reverse), legacy_rows
                    )
        for selector, expansion in runner.OP_EXPANSIONS.items():
            if selector not in {"legacy", "all"}:
                self.assertNotIn("legacy", expansion, selector)

    def test_workers_invoke_exactly_focused_or_explicit_full_routines(self):
        legacy_symbols = {
            "sgemv_",
            "dgemv_",
            "ssymv_",
            "dsymv_",
            "sger_",
            "dger_",
            "cgemv_",
            "zgemv_",
            "chemv_",
            "zhemv_",
            "cgeru_",
            "zgeru_",
            "cgerc_",
            "zgerc_",
        }
        legacy_cases = {
            "sgemv_n",
            "sgemv_t",
            "ssymv",
            "sger",
            "dgemv_n",
            "dgemv_t",
            "dsymv",
            "dger",
            "cgemv_n",
            "cgemv_t",
            "cgemv_c",
            "chemv",
            "cgeru",
            "cgerc",
            "zgemv_n",
            "zgemv_t",
            "zgemv_c",
            "zhemv",
            "zgeru",
            "zgerc",
        }
        all_operations = (
            runner.TRIANGULAR_OPERATIONS
            + runner.RANK_UPDATE_OPERATIONS
            + runner.BANDED_OPERATIONS
        )
        all_symbols = legacy_symbols | {f"{operation}_" for operation in all_operations}
        all_cases = legacy_cases | set(all_operations)
        focused_expectations = {
            ("cgeru",): {"cgeru_"},
            ("zgeru",): {"zgeru_"},
            ("cgerc",): {"cgerc_"},
            ("zgerc",): {"zgerc_"},
            ("complex-ger",): {"cgeru_", "zgeru_", "cgerc_", "zgerc_"},
            ("ssymv",): {"ssymv_"},
            ("dsymv",): {"dsymv_"},
            ("chemv",): {"chemv_"},
            ("zhemv",): {"zhemv_"},
            ("symmetric-mv",): {"ssymv_", "dsymv_", "chemv_", "zhemv_"},
        }
        expectations = [
            *(
                (selectors, symbols, set(runner.expand_operations(selectors)))
                for selectors, symbols in focused_expectations.items()
            ),
            (("legacy",), legacy_symbols, legacy_cases),
            (("legacy", "cgeru"), legacy_symbols, legacy_cases),
            (("cgeru", "legacy"), legacy_symbols, legacy_cases),
            (("legacy", "complex-ger", "legacy"), legacy_symbols, legacy_cases),
            (("all",), all_symbols, all_cases),
            (("all", "cgeru"), all_symbols, all_cases),
            (("cgeru", "all"), all_symbols, all_cases),
            (("all", "complex-ger", "all"), all_symbols, all_cases),
        ]
        for selectors, expected_symbols, expected_cases in expectations:
            with self.subTest(selectors=selectors):
                calls, rows = self.run_recording_worker(selectors)
                self.assertEqual(set(calls), expected_symbols)
                self.assertEqual({row["case"] for row in rows}, expected_cases)
                self.assertTrue(calls)

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
        x = (runner.ComplexF64 * 2)(runner.ComplexF64(2, -1), runner.ComplexF64(-1, 3))
        self.assertEqual(
            runner.complex_triangular_mv_expected(matrix, x, 2, 2, "U", "N", "N"),
            [-11 + 8j, -2 + 16j],
        )
        self.assertEqual(
            runner.complex_triangular_mv_expected(matrix, x, 2, 2, "U", "T", "N"),
            [4 + 3j, 8 + 21j],
        )
        self.assertEqual(
            runner.complex_triangular_mv_expected(matrix, x, 2, 2, "U", "C", "N"),
            [-5j, -6 + 3j],
        )
        self.assertEqual(
            runner.complex_triangular_mv_expected(matrix, x, 2, 2, "U", "N", "U"),
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
            {
                (case.case, case.trans)
                for case in cases
                if case.storage == "general-band"
            },
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
            {
                (case.case, case.uplo)
                for case in cases
                if case.storage != "general-band"
            },
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
        rank_cases = runner.packed_rank_cases(runner.requested_operations(rank_args))
        self.assertEqual(len(rank_cases), 16)
        self.assertEqual({case.uplo for case in rank_cases}, {"U", "L"})
        self.assertEqual({case.incx for case in rank_cases}, {1})
        self.assertEqual({case.incy for case in rank_cases}, {1})

    def test_triangular_banded_profiles_and_case_expansion(self):
        args = runner.parse_args(["--csv", os.devnull, "--op", "triangular-banded"])
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
            runner.general_band_expected(general, x3, y3, 3, 3, 3, 1, 1, 1, 0, "N"),
            [7, 28, 31],
        )
        self.assertEqual(
            runner.general_band_expected(general, x3, y3, 3, 3, 3, 1, 1, 1, 0, "T"),
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
            runner.packed_structured_mv_expected(upper, x, y, n, 1, 0, "U"),
            [17, 23, 32],
        )
        self.assertEqual(
            runner.packed_structured_mv_expected(lower, x, y, n, 1, 0, "L"),
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
            runner.triangular_packed_mv_expected(triangular, tx, 2, "U", "N", "N"),
            [8, 8],
        )
        self.assertEqual(
            runner.triangular_packed_mv_expected(triangular, tx, 2, "U", "T", "N"),
            [2, 11],
        )
        self.assertEqual(
            runner.triangular_packed_mv_expected(triangular, tx, 2, "U", "N", "U"),
            [7, 2],
        )

        band = (ctypes.c_double * 6)(777, 2, 3, 4, 5, 6)
        self.assertEqual(
            runner.triangular_band_mv_expected(band, x, 3, 2, 1, "U", "N", "N"),
            [8, 23, 18],
        )
        self.assertEqual(
            runner.triangular_band_mv_expected(band, x, 3, 2, 1, "U", "T", "N"),
            [2, 11, 28],
        )

        rank_matrix = (ctypes.c_double * 3)(1, 2, 3)
        rank_x = (ctypes.c_double * 2)(1, 2)
        rank_y = (ctypes.c_double * 2)(3, 4)
        self.assertEqual(
            runner.packed_rank_expected(rank_matrix, rank_x, None, 2, 2, "U"),
            [3, 6, 11],
        )
        self.assertEqual(
            runner.packed_rank_expected(rank_matrix, rank_x, rank_y, 2, 1, "U"),
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
        cx = (runner.ComplexF64 * 2)(runner.ComplexF64(1, 1), runner.ComplexF64(2, 0))
        cy = (runner.ComplexF64 * 2)(runner.ComplexF64(3, 0), runner.ComplexF64(-1, 1))
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
                str(
                    2 * bandwidth + 1
                    if row["storage"] == "general-band"
                    else bandwidth + 1
                ),
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

    def test_library_repeat_schedule_uses_balanced_latin_rotations(self):
        self.assertEqual(
            runner.library_repeat_schedule(3, 3, "interleaved"),
            [
                (0, 0, 0),
                (1, 0, 0),
                (2, 0, 0),
                (1, 0, 1),
                (2, 0, 1),
                (0, 0, 1),
                (2, 0, 2),
                (0, 0, 2),
                (1, 0, 2),
            ],
        )
        self.assertEqual(
            runner.library_repeat_schedule(2, 3, "library-major"),
            [
                (0, 0, 0),
                (0, 0, 1),
                (0, 0, 2),
                (1, 0, 0),
                (1, 0, 1),
                (1, 0, 2),
            ],
        )
        self.assertEqual(
            runner.library_repeat_schedule(1, 7, "interleaved"),
            [(0, 0, repeat) for repeat in range(7)],
        )

    def test_interleaved_repeat_balance_depends_on_selected_library_count(self):
        args = runner.parse_args(
            [
                "--process-repeats",
                "3",
                "--interleave-libraries",
                "--csv",
                "unused.csv",
            ]
        )
        self.assertTrue(args.interleave_libraries)
        self.assertEqual(args.process_schedule, "interleaved")
        self.assertEqual(args.process_repeats, 3)
        with self.assertRaisesRegex(ValueError, "multiple of the 3 selected"):
            runner.library_repeat_schedule(3, 4, "interleaved")

    def test_process_schedule_alias_conflict_and_invalid_value_exit_two(self):
        explicit = runner.parse_args(
            [
                "--csv",
                os.devnull,
                "--process-schedule",
                "interleaved",
                "--interleave-libraries",
            ]
        )
        self.assertEqual(explicit.process_schedule, "interleaved")
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as conflict:
            runner.parse_args(
                [
                    "--csv",
                    os.devnull,
                    "--process-schedule",
                    "library-major",
                    "--interleave-libraries",
                ]
            )
        self.assertEqual(conflict.exception.code, 2)
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as invalid:
            runner.parse_args(["--csv", os.devnull, "--process-schedule", "invalid"])
        self.assertEqual(invalid.exception.code, 2)

    def test_regular_file_precedes_default_macos_accelerate_platform_image(self):
        with (
            mock.patch.object(runner.sys, "platform", "darwin"),
            mock.patch.object(
                runner.Path, "is_file", autospec=True, return_value=True
            ) as is_file,
        ):
            request = runner.library_artifact_request(
                "Accelerate", runner.DEFAULT_ACCELERATE
            )
        self.assertEqual(request.source_kind, "file")
        is_file.assert_called_once()
        self.assertEqual(is_file.call_args.args[0], Path(runner.DEFAULT_ACCELERATE))

    def test_symlink_to_regular_file_remains_file_backed(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            target = Path(temp_dir) / "libaccelerate-target.dylib"
            target.write_bytes(b"accelerate-file")
            link = Path(temp_dir) / "Accelerate"
            link.symlink_to(target)
            with (
                mock.patch.object(runner.sys, "platform", "darwin"),
                mock.patch.object(runner, "DEFAULT_ACCELERATE", str(link)),
            ):
                request = runner.library_artifact_request("Accelerate", str(link))
            self.assertEqual(request.source_kind, "file")
            with runner.benchmark_artifacts.ArtifactSnapshotSet.capture(
                [request]
            ) as snapshot:
                self.assertEqual(
                    snapshot.legacy_records("library"),
                    [
                        {
                            "name": "Accelerate",
                            "path": str(link),
                            "sha256": hashlib.sha256(b"accelerate-file").hexdigest(),
                        }
                    ],
                )
                snapshot.finalize()

    def test_only_missing_default_macos_accelerate_is_a_platform_image(self):
        with (
            mock.patch.object(runner.sys, "platform", "darwin"),
            mock.patch.object(
                runner.Path, "is_file", autospec=True, return_value=False
            ) as is_file,
        ):
            request = runner.library_artifact_request(
                "Accelerate", runner.DEFAULT_ACCELERATE
            )
            self.assertEqual(request.source_kind, "platform_image")
            is_file.assert_called_once()
            with runner.benchmark_artifacts.ArtifactSnapshotSet.capture(
                [request]
            ) as snapshot:
                self.assertEqual(
                    snapshot.legacy_records("library"),
                    [
                        {
                            "name": "Accelerate",
                            "path": runner.DEFAULT_ACCELERATE,
                            "sha256": None,
                        }
                    ],
                )
                snapshot.finalize()

    def test_default_accelerate_file_probe_errors_do_not_fall_back(self):
        with (
            mock.patch.object(runner.sys, "platform", "darwin"),
            mock.patch.object(
                runner.Path,
                "is_file",
                autospec=True,
                side_effect=OSError("probe denied"),
            ),
            self.assertRaisesRegex(OSError, "probe denied"),
        ):
            runner.library_artifact_request("Accelerate", runner.DEFAULT_ACCELERATE)

    def test_non_default_accelerate_requests_fail_closed_as_files(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            default_path = str(Path(temp_dir) / "missing-default" / "Accelerate")
            invalid_requests = (
                ("wrong label", "Other", default_path),
                ("wrong path", "Accelerate", default_path + ".wrong"),
                (
                    "traversal",
                    "Accelerate",
                    str(Path(temp_dir) / "directory" / ".." / "missing-library"),
                ),
                ("bare soname", "Accelerate", "libblas.so"),
            )
            for description, name, path in invalid_requests:
                with (
                    self.subTest(description=description),
                    mock.patch.object(runner.sys, "platform", "darwin"),
                    mock.patch.object(runner, "DEFAULT_ACCELERATE", default_path),
                ):
                    request = runner.library_artifact_request(name, path)
                    self.assertEqual(request.source_kind, "file")
                    with self.assertRaises(
                        runner.benchmark_artifacts.ArtifactCaptureError
                    ):
                        runner.benchmark_artifacts.ArtifactSnapshotSet.capture(
                            [request]
                        )

    def test_controller_serializes_library_major_then_case_major_after_skips(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            zynum = Path(temp_dir) / "zynum"
            reference = Path(temp_dir) / "reference"
            zynum.write_bytes(b"zynum-a")
            reference.write_bytes(b"reference-a")
            args = runner.parse_args(
                [
                    "--csv",
                    str(Path(temp_dir).resolve() / "level2.csv"),
                    "--shape",
                    "smoke:2:2",
                    "--shape",
                    "smoke2:3:3",
                    "--process-repeats",
                    "2",
                    "--process-schedule",
                    "interleaved",
                    "--skip-missing",
                ]
            )
            calls = []

            def run_one(_script, library_name, *_args, **_kwargs):
                case = _args[1].name
                calls.append((library_name, case))
                return subprocess.CompletedProcess(
                    [],
                    0,
                    stdout=f"library,case\n{library_name},{case}\n",
                    stderr="",
                )

            def aggregate(repeat_rows):
                return [dict(repeat_rows[0][0])]

            def collect_identity(_args, libraries=(), **_kwargs):
                return self.identity_for_libraries(
                    [(artifact.name, artifact.path) for artifact in libraries]
                )

            with (
                mock.patch.object(
                    runner,
                    "libraries",
                    return_value=[
                        ("Zynum", str(zynum)),
                        ("Missing", "missing"),
                        ("Reference", str(reference)),
                    ],
                ),
                mock.patch.object(
                    runner,
                    "library_available",
                    side_effect=lambda _name, path: path != "missing",
                ),
                mock.patch.object(runner, "run_one_process", side_effect=run_one),
                mock.patch.object(
                    runner, "aggregate_worker_repeats", side_effect=aggregate
                ),
                mock.patch.object(runner, "zig_version", return_value=None),
                mock.patch.object(
                    runner.benchmark_metadata,
                    "collect_benchmark_identity_from_frozen",
                    side_effect=collect_identity,
                ) as collect_identity,
                redirect_stderr(io.StringIO()),
            ):
                runner.run_controller(args)
            with Path(args.csv).open(newline="") as csv_file:
                rows = list(csv.DictReader(csv_file))
            metadata = json.loads(
                Path(args.csv)
                .with_suffix(Path(args.csv).suffix + ".meta.json")
                .read_text()
            )

        self.assertEqual(
            calls,
            [
                ("Zynum", "smoke"),
                ("Reference", "smoke"),
                ("Reference", "smoke2"),
                ("Zynum", "smoke2"),
                ("Reference", "smoke"),
                ("Zynum", "smoke"),
                ("Zynum", "smoke2"),
                ("Reference", "smoke2"),
            ],
        )
        collect_identity.assert_called_once()
        self.assertEqual(
            [
                (artifact.name, artifact.path)
                for artifact in collect_identity.call_args.kwargs["libraries"]
            ],
            [("Zynum", str(zynum)), ("Reference", str(reference))],
        )
        csv_labels = [row["library"] for row in rows]
        self.assertEqual(
            [(row["library"], row["case"]) for row in rows],
            [
                ("Zynum", "smoke"),
                ("Zynum", "smoke2"),
                ("Reference", "smoke"),
                ("Reference", "smoke2"),
            ],
        )
        identity_labels = [
            library["name"]
            for library in metadata["benchmark_identity"]["payload"]["artifacts"][
                "libraries"
            ]
        ]
        metadata_labels = [library["name"] for library in metadata["libraries"]]
        self.assertEqual(csv_labels, ["Zynum", "Zynum", "Reference", "Reference"])
        self.assertEqual(list(dict.fromkeys(csv_labels)), identity_labels)
        self.assertEqual(identity_labels, metadata_labels)
        self.assertNotIn("Missing", csv_labels + identity_labels + metadata_labels)

    def test_interleaved_balance_uses_post_skip_library_count(self):
        args = runner.parse_args(
            [
                "--csv",
                os.devnull,
                "--process-repeats",
                "3",
                "--process-schedule",
                "interleaved",
                "--skip-missing",
            ]
        )
        with (
            mock.patch.object(
                runner,
                "libraries",
                return_value=[
                    ("Zynum", "zynum"),
                    ("Missing", "missing"),
                    ("Reference", "reference"),
                ],
            ),
            mock.patch.object(
                runner,
                "library_available",
                side_effect=lambda _name, path: path != "missing",
            ),
            mock.patch.object(
                runner.benchmark_metadata,
                "collect_benchmark_identity_from_frozen",
            ) as collect_identity,
            mock.patch.object(
                runner.benchmark_artifacts.ArtifactSnapshotSet, "capture"
            ) as capture_snapshot,
            self.assertRaisesRegex(ValueError, "2 selected libraries"),
        ):
            runner.run_controller(args)
        collect_identity.assert_not_called()
        capture_snapshot.assert_not_called()

    def test_skip_missing_never_skips_zynum(self):
        args = runner.parse_args(["--csv", os.devnull, "--skip-missing"])
        with (
            mock.patch.object(
                runner,
                "libraries",
                return_value=[("Zynum", "missing-zynum"), ("Reference", "reference")],
            ),
            mock.patch.object(
                runner,
                "library_available",
                side_effect=lambda _name, path: path == "reference",
            ),
            mock.patch.object(
                runner.benchmark_metadata,
                "collect_benchmark_identity_from_frozen",
            ) as collect_identity,
            self.assertRaisesRegex(ValueError, "cannot skip required Zynum"),
        ):
            runner.run_controller(args)
        collect_identity.assert_not_called()

    def test_duplicate_library_labels_fail_before_identity_payload_and_publication(
        self,
    ):
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir).resolve() / "level2.csv"
            metadata = output.with_suffix(output.suffix + ".meta.json")
            output.write_text("existing csv\n")
            metadata.write_text("existing metadata\n")
            args = runner.parse_args(["--csv", str(output), "--skip-missing"])
            with (
                mock.patch.object(runner, "parse_args", return_value=args),
                mock.patch.object(
                    runner,
                    "libraries",
                    return_value=[
                        ("Zynum", "zynum"),
                        ("Missing", "missing"),
                        ("zynum-blas", "alias"),
                    ],
                ),
                mock.patch.object(
                    runner,
                    "library_available",
                    side_effect=lambda _name, path: path != "missing",
                ),
                mock.patch.object(
                    runner.benchmark_metadata,
                    "collect_benchmark_identity_from_frozen",
                ) as collect_identity,
                mock.patch.object(
                    runner.benchmark_artifacts.ArtifactSnapshotSet, "capture"
                ) as capture_snapshot,
                mock.patch.object(runner, "run_one_process") as run_payload,
                mock.patch.object(runner, "publish_outputs") as publish_metadata,
                redirect_stderr(io.StringIO()) as stderr,
            ):
                self.assertEqual(runner.main(), 2)

            self.assertIn("duplicate semantic library label", stderr.getvalue())
            collect_identity.assert_not_called()
            capture_snapshot.assert_not_called()
            run_payload.assert_not_called()
            publish_metadata.assert_not_called()
            self.assertEqual(output.read_text(), "existing csv\n")
            self.assertEqual(metadata.read_text(), "existing metadata\n")

    def test_metadata_projects_unknown_git_state_from_identity(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir).resolve() / "level2.csv"
            args = runner.parse_args(["--csv", str(output)])
            identity = self.unknown_identity()
            with mock.patch.object(runner, "zig_version", return_value=None):
                metadata_bytes = runner.serialize_metadata(
                    args,
                    [{"name": "Zynum", "path": "zynum", "sha256": "hash"}],
                    [],
                    identity,
                )
            metadata = json.loads(metadata_bytes)
        self.assertIsNone(metadata["git_revision"])
        self.assertIsNone(metadata["benchmark_identity"]["source"]["dirty"])

    @unittest.skipUnless(TEST_FILE_BLAS, "no file-backed BLAS library is available")
    def test_controller_aggregates_independent_worker_processes(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir).resolve() / "level2.csv"
            missing = str(Path(temp_dir).resolve() / "missing-blas")
            result = subprocess.run(
                [
                    sys.executable,
                    str(TOOLS_DIR / "run_level2_report.py"),
                    "--zynum",
                    TEST_FILE_BLAS,
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

    @unittest.skipUnless(TEST_FILE_BLAS, "no file-backed BLAS library is available")
    def test_rank_update_controller_keeps_fresh_process_statistics(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir).resolve() / "level2-rank-update.csv"
            missing = str(Path(temp_dir).resolve() / "missing-blas")
            result = subprocess.run(
                [
                    sys.executable,
                    str(TOOLS_DIR / "run_level2_report.py"),
                    "--zynum",
                    TEST_FILE_BLAS,
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

    @unittest.skipUnless(TEST_FILE_BLAS, "no file-backed BLAS library is available")
    def test_banded_controller_keeps_fresh_process_statistics(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir).resolve() / "level2-banded.csv"
            missing = str(Path(temp_dir).resolve() / "missing-blas")
            result = subprocess.run(
                [
                    sys.executable,
                    str(TOOLS_DIR / "run_level2_report.py"),
                    "--zynum",
                    TEST_FILE_BLAS,
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

    @unittest.skipUnless(TEST_FILE_BLAS, "no file-backed BLAS library is available")
    def test_compact_controller_routes_profiles_and_metadata(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir).resolve() / "level2-compact.csv"
            missing = str(Path(temp_dir).resolve() / "missing-blas")
            result = subprocess.run(
                [
                    sys.executable,
                    str(TOOLS_DIR / "run_level2_report.py"),
                    "--zynum",
                    TEST_FILE_BLAS,
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
        self.assertEqual(metadata["packed_profiles"], [{"name": "packed3", "n": 3}])
        self.assertEqual(
            metadata["banded_profiles"],
            [{"name": "band3_bw1", "n": 3, "bandwidth": 1}],
        )
        self.assertEqual({row["shape"] for row in rows}, {"packed3", "band3_bw1"})
        self.assertEqual({row["status"] for row in rows}, {"ok"})
        self.assertEqual({row["check_status"] for row in rows}, {"sampled-ok"})


class Level2AggregationTests(unittest.TestCase):
    def test_aggregate_rejects_every_invalid_performance_sample(self):
        for field in ("time_ns", "rate_gops"):
            for value in ("0", "nan", "inf", "-inf", "1e999999"):
                row = worker_row("Zynum", 1, 1)
                row[field] = value
                with self.subTest(field=field, value=value):
                    with self.assertRaises(ValueError):
                        runner.aggregate_worker_repeats([[row]])

    def test_even_process_median_is_safe_at_finite_float_extremes(self):
        for value in (1e308, 5e-324):
            repeats = [[worker_row("Zynum", value, 1)] for _ in range(2)]
            with self.subTest(value=value):
                row = runner.aggregate_worker_repeats(repeats)[0]
                self.assertEqual(float(row["metric_min"]), value)
                self.assertEqual(float(row["metric_median"]), value)
                self.assertEqual(float(row["metric_max"]), value)
                self.assertEqual(
                    [float(sample) for sample in row["metric_samples"].split(",")],
                    [value, value],
                )

    @mock.patch.object(runner, "positive_finite_median", return_value=float("inf"))
    def test_invalid_derived_median_prevents_aggregation(self, _median):
        repeats = [[worker_row("Zynum", 1, 1)] for _ in range(2)]
        with self.assertRaisesRegex(ValueError, "metric_median"):
            runner.aggregate_worker_repeats(repeats)

    @mock.patch.object(
        runner,
        "run_controller",
        side_effect=ValueError("metric_median must be finite and positive"),
    )
    def test_invalid_derived_evidence_returns_two_before_publication(self, _run):
        args = runner.parse_args(["--csv", os.devnull])
        with (
            mock.patch.object(runner, "parse_args", return_value=args),
            redirect_stderr(io.StringIO()),
        ):
            self.assertEqual(runner.main(), 2)

    def test_semantic_group_order_is_stable_under_worker_row_shuffle(self):
        first_repeat = [
            worker_row("Zynum", 2, 50, case="zgemv"),
            worker_row("Zynum", 3, 40, case="agemv"),
        ]
        second_repeat = [
            worker_row("Zynum", 4, 25, case="zgemv"),
            worker_row("Zynum", 5, 20, case="agemv"),
        ]
        canonical = runner.aggregate_worker_repeats([first_repeat, second_repeat])
        shuffled = runner.aggregate_worker_repeats(
            [list(reversed(first_repeat)), list(reversed(second_repeat))]
        )
        self.assertEqual(shuffled, canonical)
        self.assertEqual([row["case"] for row in canonical], ["agemv", "zgemv"])

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
    CARDINALITY_FIELDS = [
        "process_repeats",
        "successful_repeats",
        "metric_samples",
    ]

    def run_checker(self, rows, fieldnames, *extra_args):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir).resolve() / "level2.csv"
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
            "process_repeats": "3",
            "successful_repeats": "3",
            "metric_samples": "1,2,3",
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
                *self.CARDINALITY_FIELDS,
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
            runner.triangular_banded_cases(runner.TRIANGULAR_BANDED_OPERATIONS, 8)
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
                *self.CARDINALITY_FIELDS,
            ],
            "--shape",
            "sq8",
            "--m",
            "8",
        )
        self.assertEqual(result, 0, stderr)
        self.assertIn("checked=1 passed=1 failed=0 missing=0", stdout)

    def test_checker_rejects_invalid_metrics_thresholds_and_ratio_extremes(self):
        fields = [
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
            *self.CARDINALITY_FIELDS,
            "metric_median",
        ]
        for value in ("nan", "inf", "-inf", "0", "-1"):
            rows = self.comparison_rows("rect", 3, 2)
            rows[0]["metric_median"] = value
            rows[1]["metric_median"] = "1"
            with self.subTest(metric=value):
                result, _, _ = self.run_checker(rows, fields, "--stat", "median")
                self.assertEqual(result, 2)
            with self.subTest(threshold=value):
                result, _, _ = self.run_checker(
                    self.comparison_rows("rect", 3, 2),
                    fields[:-1],
                    f"--ratio={value}",
                )
                self.assertEqual(result, 2)
        for candidate, comparator in (("1e308", "1e-308"), ("1e-308", "1e308")):
            rows = self.comparison_rows("rect", 3, 2)
            rows[0]["rate_gops"] = candidate
            rows[1]["rate_gops"] = comparator
            with self.subTest(candidate=candidate):
                result, _, stderr = self.run_checker(rows, fields[:-1])
                self.assertEqual(result, 2)
                self.assertIn("bad comparison ratio", stderr)

    def test_checker_rejects_duplicate_rows_and_is_shuffle_stable(self):
        fields = [
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
            *self.CARDINALITY_FIELDS,
        ]
        base = self.comparison_rows("rect", 3, 2)
        for duplicate in (dict(base[0]), {**base[0], "rate_gops": "9"}):
            with self.subTest(value=duplicate["rate_gops"]):
                result, _, stderr = self.run_checker(
                    [base[0], duplicate, base[1]], fields
                )
                self.assertEqual(result, 2)
                self.assertIn("duplicate library row", stderr)

        rows = self.comparison_rows("z-shape", 3, 2)
        rows.extend(self.comparison_rows("a-shape", 4, 2))
        rows.append({**rows[0], "library": "Second", "rate_gops": "1.0"})
        rows[0]["rate_gops"] = "0.5"
        rows[1]["rate_gops"] = "1.0"
        first = self.run_checker(rows, fields, "--comparator", "Second")
        shuffled = self.run_checker(
            list(reversed(rows)), fields, "--comparator", "Second"
        )
        self.assertEqual(first, shuffled)
        self.assertEqual(first[0], 1)
        self.assertIn("best=Reference:1.000000", first[1])

    def test_checker_binds_matching_cross_library_evidence_cardinality(self):
        rows = self.comparison_rows("rect", 3, 2)

        result, stdout, stderr = self.run_checker(rows, runner.CSV_FIELDNAMES)

        self.assertEqual(result, 0, stderr)
        self.assertIn("checked=1 passed=1 failed=0 missing=0", stdout)

    def test_checker_rejects_cross_library_repeat_count_mismatch(self):
        rows = self.comparison_rows("rect", 3, 2)
        rows[1].update(
            process_repeats="2",
            successful_repeats="2",
            metric_samples="1,2",
        )

        result, _, stderr = self.run_checker(rows, runner.CSV_FIELDNAMES)

        self.assertEqual(result, 2)
        self.assertIn("inconsistent evidence cardinality", stderr)
        self.assertIn("(3, 3, 3)", stderr)
        self.assertIn("(2, 2, 2)", stderr)

    def test_checker_rejects_metric_sample_count_mismatch(self):
        rows = self.comparison_rows("rect", 3, 2)
        rows[1]["metric_samples"] = "1,2"

        result, _, stderr = self.run_checker(rows, runner.CSV_FIELDNAMES)

        self.assertEqual(result, 2)
        self.assertIn("metric_samples count=2, successful_repeats=3", stderr)


if __name__ == "__main__":
    unittest.main()
