#!/usr/bin/env python3
# Copyright (C) 2026 Zynum contributors
# SPDX-License-Identifier: LGPL-3.0-or-later

import csv
import ctypes
import hashlib
import importlib.util
import io
import json
import os
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest import mock

import benchmark_artifacts

TOOLS_DIR = Path(__file__).resolve().parent


def load_tool(module_name):
    spec = importlib.util.spec_from_file_location(
        module_name, TOOLS_DIR / f"{module_name}.py"
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {module_name}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


runner = load_tool("run_level1_report")
checker = load_tool("check_level1_report")
REAL_ARTIFACT_CAPTURE = runner.ArtifactSnapshotSet.capture


class FakeFrozenArtifact:
    def __init__(self, request, index):
        self.name = request.name
        self.path = os.fspath(request.path)
        self.role = request.role
        self.sha256 = None
        self.source_kind = request.source_kind
        self._execution_path = (
            self.path
            if request.source_kind == "platform_image"
            else f"/private-artifacts/{request.role}-{index}"
        )

    @property
    def execution_path(self):
        return self._execution_path


class FakeArtifactSnapshot:
    def __init__(self, requests):
        self.artifacts = tuple(
            FakeFrozenArtifact(request, index) for index, request in enumerate(requests)
        )

    def for_role(self, role):
        return tuple(item for item in self.artifacts if item.role == role)

    def legacy_records(self, role):
        return [
            {"name": item.name, "path": item.path, "sha256": item.sha256}
            for item in self.for_role(role)
        ]

    def redact_private_paths(self, value):
        replacements = [
            (item.execution_path, item.path)
            for item in self.artifacts
            if item.source_kind == "file"
        ]
        if isinstance(value, str):
            for private, public in replacements:
                value = value.replace(private, public)
            return value
        if isinstance(value, bytes):
            for private, public in replacements:
                value = value.replace(private.encode(), public.encode())
            return value
        if isinstance(value, dict):
            return {key: self.redact_private_paths(item) for key, item in value.items()}
        if isinstance(value, list):
            return [self.redact_private_paths(item) for item in value]
        if isinstance(value, tuple):
            return tuple(self.redact_private_paths(item) for item in value)
        return value

    def finalize(self):
        return None

    def close(self):
        return None


def report_row(library, op="sdot", value=2.0, **overrides):
    row = {
        "group": "real_f32",
        "op": op,
        "variant": "default",
        "library": library,
        "library_path": f"lib{library}.so",
        "n": "257",
        "incx": "-2",
        "incy": "3",
        "metric": "rate_gops",
        "status": "ok",
        "rate_gops": str(value),
        "bandwidth_gbps": "",
        "metric_min": str(value),
        "metric_median": str(value),
        "metric_max": str(value),
        "metric_samples": str(value),
        "process_repeats": "2",
        "successful_repeats": "2",
        "check_status": "sampled-ok",
        "symbol": "cblas_sdot",
        "abi_surface": "cblas",
        "preflight_symbol": "cblas_sdot",
        "preflight_abi_surface": "cblas",
        "capability_status": "supported",
    }
    row.update(overrides)
    return row


class Level1RunnerTests(unittest.TestCase):
    def setUp(self):
        self.capture_patcher = mock.patch.object(
            runner.ArtifactSnapshotSet,
            "capture",
            side_effect=lambda requests: FakeArtifactSnapshot(requests),
        )
        self.capture_mock = self.capture_patcher.start()
        self.addCleanup(self.capture_patcher.stop)

    def test_library_availability_never_loads_a_live_path(self):
        with mock.patch.object(runner.ctypes, "CDLL") as loader:
            self.assertFalse(runner.library_available("lib-not-a-file.so"))
        loader.assert_not_called()

    def test_platform_image_exception_is_exact_and_cannot_traverse(self):
        traversal = (
            "/System/Library/Frameworks/Accelerate.framework/"
            "../../../../private/tmp/unhashed.dylib"
        )
        with (
            mock.patch.object(runner.sys, "platform", "darwin"),
            mock.patch.object(runner.Path, "is_file", return_value=False),
        ):
            self.assertFalse(runner.platform_image_path(traversal))
            self.assertEqual(
                runner.library_artifact_request("Accelerate", traversal).source_kind,
                "file",
            )
            self.assertEqual(
                runner.library_artifact_request(
                    "Other", runner.DEFAULT_ACCELERATE
                ).source_kind,
                "file",
            )
            self.assertEqual(
                runner.library_artifact_request(
                    "Accelerate", runner.DEFAULT_ACCELERATE
                ).source_kind,
                "platform_image",
            )

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

    def run_publication_main(self, output, *, identity=None, publish=None):
        args = runner.parse_args(
            ["--csv", str(output), "--skip-copy-byte-coverage", "--op", "sdot"]
        )
        sample = {
            "group": "real_f32",
            "op": "sdot",
            "variant": "default",
            "status": "ok",
            "metric": "rate_gops",
            "library": "Zynum",
            "library_path": "zynum",
            "rate_gops": 2.0,
            "bandwidth_gbps": "",
            "repeat": 0,
        }
        with (
            mock.patch.object(runner, "parse_args", return_value=args),
            mock.patch.object(runner.Path, "exists", return_value=True),
            mock.patch.object(runner, "selected_copy_cases", return_value=[]),
            mock.patch.object(
                runner,
                "level1_cases",
                return_value=[("real_f32", "sdot", "default", 1, 1)],
            ),
            mock.patch.object(runner, "case_allowed", return_value=True),
            mock.patch.object(runner, "libraries", return_value=[("Zynum", "zynum")]),
            mock.patch.object(runner, "run_level1_op", return_value=sample),
            mock.patch.object(runner, "zig_version", return_value=None),
            mock.patch.object(
                runner.benchmark_metadata,
                "collect_benchmark_identity_from_frozen",
                return_value=identity
                or self.identity_for_libraries([("Zynum", "zynum")]),
            ),
            mock.patch.object(
                runner, "publish_outputs", side_effect=publish
            ) as publisher,
            redirect_stdout(io.StringIO()),
            redirect_stderr(io.StringIO()),
        ):
            result = runner.main()
        return result, publisher

    @staticmethod
    def level1_probe_source(score):
        return f"""#!/usr/bin/env python3
import argparse
import sys

parser = argparse.ArgumentParser()
parser.add_argument("--lib", required=True)
parser.add_argument("--op")
parser.add_argument("--variant")
parser.add_argument("--incx")
parser.add_argument("--incy")
parser.add_argument("--n")
parser.add_argument("--seconds")
parser.add_argument("--kind")
args = parser.parse_args()
print("argv=" + " ".join(sys.argv))
print("stderr-argv=" + " ".join(sys.argv), file=sys.stderr)
library = open(args.lib, encoding="utf-8").read().strip()
value = {score} if {score} != 7 else (7 if library == "A" else 99)
print(
    f"iters=1 elapsed_ns=1 rate_Gops={{value}} bandwidth_GBps=8 "
    "checksum=1 symbol=sdot_ abi_surface=fortran"
)
"""

    def test_real_controller_freezes_probes_library_and_controller_identity(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir).resolve()
            level1_probe = root / "level1-probe"
            copy_probe = root / "copy-probe"
            library = root / "libzynum.so"
            output = root / "level1.csv"
            probe_a = self.level1_probe_source(7).encode()
            library_a = b"A\n"
            for probe in (level1_probe, copy_probe):
                probe.write_bytes(probe_a)
                probe.chmod(0o755)
            library.write_bytes(library_a)
            args = runner.parse_args(
                [
                    "--csv",
                    str(output),
                    "--level1-probe",
                    str(level1_probe),
                    "--copy-probe",
                    str(copy_probe),
                    "--zynum",
                    str(library),
                    "--accelerate",
                    "none",
                    "--openblas",
                    "none",
                    "--skip-missing",
                    "--skip-copy-byte-coverage",
                    "--op",
                    "sdot",
                    "--process-repeats",
                    "2",
                ]
            )
            private_paths = []
            child_count = 0
            real_subprocess_run = runner.subprocess.run

            def capture_then_replace(requests):
                artifacts = REAL_ARTIFACT_CAPTURE(requests)
                private_paths.extend(
                    item.execution_path
                    for item in artifacts.artifacts
                    if item.source_kind == "file"
                )
                for index, probe in enumerate((level1_probe, copy_probe)):
                    replacement = root / f"replacement-probe-{index}"
                    replacement.write_text(self.level1_probe_source(123))
                    replacement.chmod(0o755)
                    replacement.replace(probe)
                replacement_library = root / "replacement-library"
                replacement_library.write_text("B\n")
                replacement_library.replace(library)
                return artifacts

            def run_then_restore_original(command, *run_args, **run_kwargs):
                nonlocal child_count
                result = real_subprocess_run(command, *run_args, **run_kwargs)
                if isinstance(command, list) and "--lib" in command:
                    child_count += 1
                    if child_count == 1:
                        for index, probe in enumerate((level1_probe, copy_probe)):
                            restored = root / f"restored-probe-{index}"
                            restored.write_bytes(probe_a)
                            restored.chmod(0o755)
                            restored.replace(probe)
                        restored_library = root / "restored-library"
                        restored_library.write_bytes(library_a)
                        restored_library.replace(library)
                return result

            preflight = runner.check_result(
                "sampled-ok",
                0.0,
                symbol="sdot_",
                abi_surface="fortran",
                memory_status="guarded-ok",
            )
            stdout = io.StringIO()
            stderr = io.StringIO()
            with (
                mock.patch.object(runner, "parse_args", return_value=args),
                mock.patch.object(
                    runner.ArtifactSnapshotSet,
                    "capture",
                    side_effect=capture_then_replace,
                ),
                mock.patch.object(
                    runner, "check_level1_op_isolated", return_value=preflight
                ),
                mock.patch.object(
                    runner.subprocess,
                    "run",
                    side_effect=run_then_restore_original,
                ),
                redirect_stdout(stdout),
                redirect_stderr(stderr),
            ):
                self.assertEqual(runner.main(), 0)

            with output.open(newline="") as csv_file:
                rows = list(csv.DictReader(csv_file))
            metadata_bytes = output.with_suffix(".csv.meta.json").read_bytes()
            metadata = json.loads(metadata_bytes)
            self.assertEqual([row["rate_gops"] for row in rows], ["7.0"])
            self.assertEqual(child_count, 2)
            expected_probe_hash = hashlib.sha256(probe_a).hexdigest()
            expected_library_hash = hashlib.sha256(library_a).hexdigest()
            self.assertEqual(
                metadata["probes"]["level1_probe_sha256"], expected_probe_hash
            )
            self.assertEqual(
                metadata["probes"]["copy_probe_sha256"], expected_probe_hash
            )
            self.assertEqual(
                metadata["libraries"]["Zynum"]["sha256"],
                expected_library_hash,
            )
            identity_artifacts = metadata["benchmark_identity"]["payload"]["artifacts"]
            self.assertEqual(
                [record["name"] for record in identity_artifacts["binaries"]],
                ["level1_probe", "copy_probe", "run_level1_report"],
            )
            self.assertEqual(
                identity_artifacts["binaries"][0]["sha256"], expected_probe_hash
            )
            self.assertEqual(
                identity_artifacts["libraries"][0]["sha256"],
                expected_library_hash,
            )
            published = output.read_bytes() + metadata_bytes
            diagnostics = stdout.getvalue() + stderr.getvalue()
            for private_path in private_paths:
                self.assertNotIn(private_path.encode(), published)
                self.assertNotIn(private_path, diagnostics)
                self.assertFalse(Path(private_path).exists())
            self.assertNotIn(".zynum-benchmark-artifacts-", diagnostics)

    def test_check_worker_uses_frozen_script_and_library_and_redacts_output(self):
        requests = [
            runner.ArtifactRequest.interpreter_script(
                "run_level1_report", "/public/run_level1_report.py"
            ),
            runner.ArtifactRequest.library("Zynum", "/public/libzynum.so"),
        ]
        artifacts = FakeArtifactSnapshot(requests)
        script = artifacts.for_role("binary")[0].execution_path
        library = artifacts.for_role("library")[0].execution_path
        payload = runner.check_result(
            "missing",
            raw=f"cannot load {library}",
            capability_status="unavailable",
        )
        completed = mock.Mock(
            returncode=0,
            stdout=json.dumps(payload) + "\n",
            stderr="",
        )
        with mock.patch.object(runner.subprocess, "run", return_value=completed) as run:
            result = runner.check_worker_result(
                "level1",
                library,
                "sdot",
                controller_script=script,
                artifacts=artifacts,
            )

        command = run.call_args.args[0]
        self.assertEqual(command[1], script)
        self.assertEqual(command[command.index("--check-library") + 1], library)
        self.assertEqual(result["check_raw_output"], "cannot load /public/libzynum.so")
        self.assertNotIn("/private-artifacts/", json.dumps(result))

    def test_stage_drift_before_publication_returns_two_and_preserves_outputs(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir).resolve()
            level1_probe = root / "level1-probe"
            copy_probe = root / "copy-probe"
            library = root / "libzynum.so"
            output = root / "level1.csv"
            metadata = output.with_suffix(".csv.meta.json")
            for probe in (level1_probe, copy_probe):
                probe.write_text("#!/bin/sh\nexit 0\n")
                probe.chmod(0o755)
            library.write_text("A\n")
            output.write_bytes(b"previous csv\n")
            metadata.write_bytes(b"previous metadata\n")
            args = runner.parse_args(
                [
                    "--csv",
                    str(output),
                    "--level1-probe",
                    str(level1_probe),
                    "--copy-probe",
                    str(copy_probe),
                    "--zynum",
                    str(library),
                    "--accelerate",
                    "none",
                    "--openblas",
                    "none",
                    "--skip-missing",
                    "--skip-copy-byte-coverage",
                    "--op",
                    "sdot",
                ]
            )
            captured = []
            drifted_payloads = []
            drifted_paths = []
            cleanup_errors = []

            def capture_real(requests, **limits):
                artifacts = runner.ArtifactSnapshotSet(
                    requests,
                    private_parent=root,
                    **limits,
                )
                captured.append(artifacts)
                return artifacts

            real_close = runner.ArtifactSnapshotSet.close

            def record_cleanup_error(snapshot):
                try:
                    real_close(snapshot)
                except benchmark_artifacts.ArtifactCleanupError as error:
                    cleanup_errors.append(error)
                    raise

            sample = {
                "group": "real_f32",
                "op": "sdot",
                "variant": "default",
                "status": "ok",
                "metric": "rate_gops",
                "library": "Zynum",
                "library_path": str(library),
                "rate_gops": 2.0,
                "bandwidth_gbps": "",
                "repeat": 0,
            }
            real_serialize = runner.serialize_report

            def serialize_then_corrupt(*serialize_args, **serialize_kwargs):
                result = real_serialize(*serialize_args, **serialize_kwargs)
                private_probe = Path(captured[0].for_role("binary")[0].execution_path)
                payload = private_probe.read_bytes()
                private_probe.chmod(0o700)
                drifted = bytes([payload[0] ^ 1]) + payload[1:]
                private_probe.write_bytes(drifted)
                private_probe.chmod(0o500)
                drifted_payloads.append(drifted)
                drifted_paths.append(private_probe)
                return result

            publish = mock.Mock()
            stderr = io.StringIO()
            with (
                mock.patch.object(runner, "parse_args", return_value=args),
                mock.patch.object(
                    runner.ArtifactSnapshotSet,
                    "capture",
                    side_effect=capture_real,
                ),
                mock.patch.object(
                    runner.ArtifactSnapshotSet,
                    "close",
                    new=record_cleanup_error,
                ),
                mock.patch.object(
                    runner.benchmark_metadata,
                    "collect_benchmark_identity_from_frozen",
                    return_value=self.unknown_identity(),
                ),
                mock.patch.object(runner, "run_level1_op", return_value=sample),
                mock.patch.object(
                    runner, "serialize_report", side_effect=serialize_then_corrupt
                ),
                mock.patch.object(runner, "publish_outputs", publish),
                redirect_stdout(io.StringIO()),
                redirect_stderr(stderr),
            ):
                self.assertEqual(runner.main(), 2)

            publish.assert_not_called()
            self.assertEqual(len(cleanup_errors), 1)
            self.assertEqual(cleanup_errors[0].code, "artifact_cleanup_incomplete")
            self.assertEqual(output.read_bytes(), b"previous csv\n")
            self.assertEqual(metadata.read_bytes(), b"previous metadata\n")
            self.assertNotIn(".zynum-benchmark-artifacts-", stderr.getvalue())
            private_probe = drifted_paths[0]
            self.assertFalse(private_probe.exists())
            retained = [
                Path(path)
                for path in cleanup_errors[0].recovery_paths
                if Path(path).name == private_probe.name and Path(path).is_file()
            ]
            self.assertEqual(len(retained), 1)
            drifted = retained[0]
            self.assertTrue(
                drifted.parent.name.startswith(".zynum-benchmark-artifact-quarantine-")
            )
            self.assertEqual(drifted.read_bytes(), drifted_payloads[0])
            self.assertIn(
                "private_artifact_replaced",
                {issue.code for issue in cleanup_errors[0].issues},
            )
            drifted.unlink()
            drifted.parent.rmdir()

    def test_file_backed_bare_soname_fails_closed_before_execution(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir).resolve()
            level1_probe = root / "level1-probe"
            copy_probe = root / "copy-probe"
            output = root / "level1.csv"
            for probe in (level1_probe, copy_probe):
                probe.write_text("#!/bin/sh\nexit 0\n")
                probe.chmod(0o755)
            args = runner.parse_args(
                [
                    "--csv",
                    str(output),
                    "--level1-probe",
                    str(level1_probe),
                    "--copy-probe",
                    str(copy_probe),
                    "--skip-copy-byte-coverage",
                    "--op",
                    "sdot",
                ]
            )
            run_payload = mock.Mock()
            publish = mock.Mock()
            stderr = io.StringIO()
            with (
                mock.patch.object(runner, "parse_args", return_value=args),
                mock.patch.object(
                    runner,
                    "libraries",
                    return_value=[("Zynum", "libzynum.so")],
                ),
                mock.patch.object(
                    runner.ArtifactSnapshotSet,
                    "capture",
                    side_effect=REAL_ARTIFACT_CAPTURE,
                ),
                mock.patch.object(runner, "run_level1_op", run_payload),
                mock.patch.object(runner, "publish_outputs", publish),
                redirect_stderr(stderr),
            ):
                self.assertEqual(runner.main(), 2)

            self.assertIn("file artifact path must be explicit", stderr.getvalue())
            run_payload.assert_not_called()
            publish.assert_not_called()
            self.assertFalse(output.exists())

    def run_scheduled_main(self, schedule, *, repeats=2, configured_libraries=None):
        temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(temp_dir.cleanup)
        output = Path(temp_dir.name).resolve() / "level1.csv"
        if configured_libraries is None:
            configured_libraries = [
                ("Zynum", "zynum"),
                ("Missing", "missing"),
                ("Reference", "reference"),
            ]
        args = runner.parse_args(
            [
                "--csv",
                str(output),
                "--process-repeats",
                str(repeats),
                "--process-schedule",
                schedule,
                "--skip-copy-byte-coverage",
                "--skip-missing",
            ]
        )
        calls = []

        def run_level1(_args, library_name, _path, *_case, **_kwargs):
            calls.append(library_name)
            value = float(len(calls))
            return {
                "status": "ok",
                "metric": "rate_gops",
                "library": library_name,
                "library_path": _path,
                "rate_gops": value,
                "bandwidth_gbps": None,
                "repeat": 0,
            }

        def collect_identity(_args, libraries=(), **_kwargs):
            return self.identity_for_libraries(
                [(item.name, item.path) for item in libraries]
            )

        with (
            mock.patch.object(runner, "parse_args", return_value=args),
            mock.patch.object(runner.Path, "exists", return_value=True),
            mock.patch.object(runner, "selected_copy_cases", return_value=[]),
            mock.patch.object(
                runner,
                "level1_cases",
                return_value=[("real_f32", "sdot", "default", 1, 1)],
            ),
            mock.patch.object(runner, "case_allowed", return_value=True),
            mock.patch.object(
                runner,
                "libraries",
                return_value=configured_libraries,
            ),
            mock.patch.object(
                runner,
                "library_available",
                side_effect=lambda path: path != "missing",
            ),
            mock.patch.object(runner, "run_level1_op", side_effect=run_level1),
            mock.patch.object(runner, "zig_version", return_value=None),
            mock.patch.object(
                runner.benchmark_metadata,
                "collect_benchmark_identity_from_frozen",
                side_effect=collect_identity,
            ) as collect_identity,
            redirect_stdout(io.StringIO()),
            redirect_stderr(io.StringIO()),
        ):
            result = runner.main()

        metadata = json.loads(
            output.with_suffix(output.suffix + ".meta.json").read_text()
        )
        with output.open(newline="") as csv_file:
            rows = list(csv.DictReader(csv_file))
        return result, calls, rows, metadata, collect_identity

    def test_process_schedule_is_the_only_schedule_cli(self):
        with mock.patch.object(runner.sys, "platform", "win32"):
            windows = runner.parse_args(["--csv", os.devnull])
        self.assertEqual(
            windows.level1_probe, "zig-out/perf-report/bin/level1_probe.exe"
        )
        self.assertEqual(windows.copy_probe, "zig-out/perf-report/bin/dcopy_probe.exe")
        self.assertEqual(windows.zynum, "zig-out/bin/zynum_blas.dll")
        default = runner.parse_args(["--csv", os.devnull])
        self.assertEqual(default.process_schedule, "library-major")

        for schedule in runner.SCHEDULE_CHOICES:
            with self.subTest(schedule=schedule):
                args = runner.parse_args(
                    ["--csv", os.devnull, "--process-schedule", schedule]
                )
                self.assertEqual(args.process_schedule, schedule)

        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as legacy:
            runner.parse_args(["--csv", os.devnull, "--interleave-libraries"])
        self.assertEqual(legacy.exception.code, 2)
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as invalid:
            runner.parse_args(["--csv", os.devnull, "--process-schedule", "invalid"])
        self.assertEqual(invalid.exception.code, 2)

    def test_controller_publishes_one_ordered_immutable_pair(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            parent = Path(temp_dir).resolve() / "absent"
            output = parent / "level1.csv"

            def assert_prepublication(outputs):
                self.assertFalse(os.path.exists(parent))
                self.assertEqual(
                    [item.path for item in outputs],
                    [output, output.with_suffix(".csv.meta.json")],
                )
                self.assertTrue(all(type(item.contents) is bytes for item in outputs))
                self.assertTrue(outputs[0].contents.endswith(b"\r\n"))
                self.assertTrue(outputs[1].contents.endswith(b"\n"))
                self.assertNotIn(b"\r\n", outputs[1].contents)
                header = outputs[0].contents.split(b"\r\n", 1)[0]
                self.assertTrue(header.startswith(b"group,op,variant,library,"))
                self.assertTrue(header.endswith(b",check_raw_output,raw_output"))
                rows = list(
                    csv.DictReader(io.StringIO(outputs[0].contents.decode("utf-8")))
                )
                self.assertEqual(
                    [(row["library"], row["op"]) for row in rows], [("Zynum", "sdot")]
                )
                self.assertEqual(json.loads(outputs[1].contents)["process_repeats"], 1)

            result, publisher = self.run_publication_main(
                output, publish=assert_prepublication
            )
            self.assertEqual(result, 0)
            publisher.assert_called_once()
            self.assertFalse(parent.exists())

    def test_metadata_serialization_failure_precedes_publication(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir).resolve() / "level1.csv"
            metadata = output.with_suffix(".csv.meta.json")
            output.write_bytes(b"previous csv\n")
            metadata.write_bytes(b"previous metadata\n")
            identity = self.identity_for_libraries([("Zynum", "zynum")])
            identity["invalid"] = float("nan")
            publish = mock.Mock()

            result, _ = self.run_publication_main(
                output, identity=identity, publish=publish
            )

            self.assertEqual(result, 2)
            publish.assert_not_called()
            self.assertEqual(output.read_bytes(), b"previous csv\n")
            self.assertEqual(metadata.read_bytes(), b"previous metadata\n")

    def test_publisher_failure_receives_one_batch_and_cannot_split_pair(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir).resolve() / "level1.csv"
            metadata = output.with_suffix(".csv.meta.json")
            output.write_bytes(b"previous csv\n")
            metadata.write_bytes(b"previous metadata\n")
            calls = []

            def fail(outputs):
                calls.append(outputs)
                raise OSError("injected publication failure")

            with self.assertRaisesRegex(OSError, "injected publication failure"):
                self.run_publication_main(output, publish=fail)

            self.assertEqual(len(calls), 1)
            self.assertEqual(len(calls[0]), 2)
            self.assertEqual(output.read_bytes(), b"previous csv\n")
            self.assertEqual(metadata.read_bytes(), b"previous metadata\n")

    def test_schedules_use_actual_libraries_and_collect_identity_once(self):
        result, calls, rows, metadata, collect_identity = self.run_scheduled_main(
            "interleaved"
        )
        self.assertEqual(result, 0)
        self.assertEqual(calls, ["Zynum", "Reference", "Reference", "Zynum"])
        self.assertEqual([row["library"] for row in rows], ["Zynum", "Reference"])
        self.assertEqual(
            {
                row["library"]: (
                    row["process_repeats"],
                    row["successful_repeats"],
                )
                for row in rows
            },
            {"Zynum": ("2", "2"), "Reference": ("2", "2")},
        )
        collect_identity.assert_called_once()
        self.assertEqual(
            [
                (item.name, item.path)
                for item in collect_identity.call_args.kwargs["libraries"]
            ],
            [("Zynum", "zynum"), ("Reference", "reference")],
        )
        self.assertIsNone(metadata["git_revision"])
        self.assertIsNone(metadata["benchmark_identity"]["source"]["dirty"])
        csv_labels = [row["library"] for row in rows]
        identity_labels = [
            library["name"]
            for library in metadata["benchmark_identity"]["payload"]["artifacts"][
                "libraries"
            ]
        ]
        metadata_labels = list(metadata["libraries"])
        self.assertEqual(csv_labels, identity_labels)
        self.assertEqual(set(csv_labels), set(metadata_labels))
        self.assertNotIn("Missing", csv_labels)

        result, calls, rows, _, collect_identity = self.run_scheduled_main(
            "library-major"
        )
        self.assertEqual(result, 0)
        self.assertEqual(calls, ["Zynum", "Zynum", "Reference", "Reference"])
        self.assertEqual([row["library"] for row in rows], ["Zynum", "Reference"])
        collect_identity.assert_called_once()

    def test_interleaved_schedule_accepts_any_positive_repeat_count_for_one_library(
        self,
    ):
        result, calls, rows, _, collect_identity = self.run_scheduled_main(
            "interleaved",
            repeats=3,
            configured_libraries=[("Zynum", "zynum")],
        )
        self.assertEqual(result, 0)
        self.assertEqual(calls, ["Zynum", "Zynum", "Zynum"])
        self.assertEqual([row["library"] for row in rows], ["Zynum"])
        collect_identity.assert_called_once()

    def test_skip_missing_never_skips_zynum(self):
        args = runner.parse_args(
            ["--csv", os.devnull, "--skip-missing", "--skip-copy-byte-coverage"]
        )
        with (
            mock.patch.object(runner, "parse_args", return_value=args),
            mock.patch.object(runner.Path, "exists", return_value=True),
            mock.patch.object(
                runner,
                "libraries",
                return_value=[("Zynum", "missing-zynum"), ("Reference", "ref")],
            ),
            mock.patch.object(
                runner, "library_available", side_effect=lambda path: path == "ref"
            ),
            mock.patch.object(
                runner.benchmark_metadata,
                "collect_benchmark_identity_from_frozen",
            ) as collect_identity,
            redirect_stderr(io.StringIO()) as stderr,
        ):
            self.assertEqual(runner.main(), 2)
        self.assertIn("cannot skip required Zynum", stderr.getvalue())
        collect_identity.assert_not_called()

    def test_duplicate_library_labels_fail_before_identity_payload_and_publication(
        self,
    ):
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir).resolve() / "level1.csv"
            metadata = output.with_suffix(output.suffix + ".meta.json")
            output.write_text("existing csv\n")
            metadata.write_text("existing metadata\n")
            args = runner.parse_args(
                [
                    "--csv",
                    str(output),
                    "--skip-copy-byte-coverage",
                    "--skip-missing",
                ]
            )
            with (
                mock.patch.object(runner, "parse_args", return_value=args),
                mock.patch.object(runner.Path, "exists", return_value=True),
                mock.patch.object(
                    runner,
                    "libraries",
                    return_value=[
                        ("Zynum", "zynum"),
                        ("Missing", "missing"),
                        ("libzynum-blas", "alias"),
                    ],
                ),
                mock.patch.object(
                    runner,
                    "library_available",
                    side_effect=lambda path: path != "missing",
                ),
                mock.patch.object(
                    runner.benchmark_metadata,
                    "collect_benchmark_identity_from_frozen",
                ) as collect_identity,
                mock.patch.object(runner, "run_level1_op") as run_payload,
                mock.patch.object(runner, "selected_copy_cases") as select_payload,
                redirect_stderr(io.StringIO()) as stderr,
            ):
                self.assertEqual(runner.main(), 2)

            self.assertIn("duplicate semantic library label", stderr.getvalue())
            collect_identity.assert_not_called()
            run_payload.assert_not_called()
            select_payload.assert_not_called()
            self.assertEqual(output.read_text(), "existing csv\n")
            self.assertEqual(metadata.read_text(), "existing metadata\n")

    def test_aggregate_rejects_every_invalid_performance_sample(self):
        for field in ("rate_gops", "bandwidth_gbps"):
            for value in ("0", "nan", "inf", "-inf", "1e999999"):
                row = {
                    "status": "ok",
                    "metric": "rate_gops",
                    "rate_gops": 1,
                    "bandwidth_gbps": 1,
                }
                row[field] = value
                with self.subTest(field=field, value=value):
                    with self.assertRaises(ValueError):
                        runner.choose_best([row])

    def test_aggregate_rejects_mixed_or_missing_repeat_results(self):
        ok = {
            "status": "ok",
            "metric": "rate_gops",
            "rate_gops": 2.0,
            "bandwidth_gbps": "",
        }
        for failed in (
            None,
            {**ok, "status": "error", "raw_output": "worker failed"},
            {**ok, "status": "missing"},
            {**ok, "status": "parse_error"},
            {**ok, "status": "surface_mismatch"},
        ):
            with self.subTest(failed=failed):
                with self.assertRaises(ValueError):
                    runner.choose_best([ok, failed])

    def test_aggregate_records_complete_repeat_contract_and_sample_order(self):
        rows = [
            {
                "status": "ok",
                "metric": "rate_gops",
                "rate_gops": value,
                "bandwidth_gbps": "",
            }
            for value in (3.0, 1.0, 2.0)
        ]
        aggregate = runner.choose_best(rows)
        self.assertEqual(aggregate["process_repeats"], 3)
        self.assertEqual(aggregate["successful_repeats"], 3)
        self.assertEqual(aggregate["metric_samples"], "3,1,2")
        self.assertEqual(aggregate["rate_gops"], 3.0)

    def test_even_process_median_is_safe_at_finite_float_extremes(self):
        for value in (1e308, 5e-324):
            rows = [
                {
                    "status": "ok",
                    "metric": "rate_gops",
                    "rate_gops": value,
                    "bandwidth_gbps": "",
                }
                for _ in range(2)
            ]
            with self.subTest(value=value):
                aggregate = runner.choose_best(rows)
                self.assertEqual(aggregate["process_repeats"], 2)
                self.assertEqual(aggregate["successful_repeats"], 2)
                self.assertEqual(float(aggregate["metric_min"]), value)
                self.assertEqual(float(aggregate["metric_median"]), value)
                self.assertEqual(float(aggregate["metric_max"]), value)
                self.assertEqual(
                    [
                        float(sample)
                        for sample in aggregate["metric_samples"].split(",")
                    ],
                    [value, value],
                )

    @mock.patch.object(runner, "positive_finite_median", return_value=float("inf"))
    def test_invalid_derived_median_prevents_aggregation(self, _median):
        rows = [
            {
                "status": "ok",
                "metric": "rate_gops",
                "rate_gops": 1,
                "bandwidth_gbps": "",
            }
            for _ in range(2)
        ]
        with self.assertRaisesRegex(ValueError, "metric_median"):
            runner.choose_best(rows)

    def test_invalid_derived_evidence_returns_two_before_publication(self):
        args = runner.parse_args(["--csv", os.devnull])
        with (
            mock.patch.object(runner, "parse_args", return_value=args),
            mock.patch.object(runner.Path, "exists", return_value=True),
            mock.patch.object(runner, "selected_copy_cases", return_value=[]),
            mock.patch.object(
                runner,
                "level1_cases",
                return_value=[("real_f32", "sdot", "default", 1, 1)],
            ),
            mock.patch.object(runner, "case_allowed", return_value=True),
            mock.patch.object(
                runner, "libraries", return_value=[("Zynum", "libzynum.so")]
            ),
            mock.patch.object(
                runner,
                "run_level1_op",
                side_effect=ValueError("metric_median must be finite and positive"),
            ),
            mock.patch.object(
                runner.benchmark_metadata,
                "collect_benchmark_identity_from_frozen",
                return_value=self.unknown_identity(),
            ),
            redirect_stderr(io.StringIO()),
        ):
            self.assertEqual(runner.main(), 2)

    def test_mixed_repeat_failure_preserves_existing_outputs_and_parent_boundary(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir).resolve() / "existing.csv"
            metadata_path = output.with_suffix(output.suffix + ".meta.json")
            output.write_text("old csv\n")
            metadata_path.write_text("old metadata\n")
            args = runner.parse_args(
                [
                    "--csv",
                    str(output),
                    "--process-repeats",
                    "2",
                    "--skip-copy-byte-coverage",
                ]
            )
            results = iter(
                [
                    {
                        "status": "ok",
                        "metric": "rate_gops",
                        "library": "Zynum",
                        "library_path": "zynum",
                        "rate_gops": 1.0,
                        "bandwidth_gbps": "",
                        "repeat": 0,
                    },
                    {
                        "status": "error",
                        "metric": "rate_gops",
                        "library": "Zynum",
                        "library_path": "zynum",
                        "rate_gops": None,
                        "bandwidth_gbps": None,
                        "repeat": 0,
                        "raw_output": "worker failed",
                    },
                ]
            )
            with (
                mock.patch.object(runner, "parse_args", return_value=args),
                mock.patch.object(runner.Path, "exists", return_value=True),
                mock.patch.object(runner, "selected_copy_cases", return_value=[]),
                mock.patch.object(
                    runner,
                    "level1_cases",
                    return_value=[("real_f32", "sdot", "default", 1, 1)],
                ),
                mock.patch.object(runner, "case_allowed", return_value=True),
                mock.patch.object(
                    runner, "libraries", return_value=[("Zynum", "zynum")]
                ),
                mock.patch.object(runner, "run_level1_op", side_effect=results),
                mock.patch.object(
                    runner.benchmark_metadata,
                    "collect_benchmark_identity_from_frozen",
                    return_value=self.unknown_identity(),
                ),
                redirect_stderr(io.StringIO()) as stderr,
            ):
                self.assertEqual(runner.main(), 2)
            self.assertIn("repeat 2 returned non-ok", stderr.getvalue())
            self.assertEqual(output.read_text(), "old csv\n")
            self.assertEqual(metadata_path.read_text(), "old metadata\n")

            absent_parent = Path(temp_dir).resolve() / "not-created"
            args.csv = str(absent_parent / "level1.csv")
            results = iter(
                [
                    {
                        "status": "ok",
                        "metric": "rate_gops",
                        "library": "Zynum",
                        "library_path": "zynum",
                        "rate_gops": 1.0,
                        "bandwidth_gbps": "",
                        "repeat": 0,
                    },
                    {
                        "status": "error",
                        "metric": "rate_gops",
                        "library": "Zynum",
                        "library_path": "zynum",
                        "rate_gops": None,
                        "bandwidth_gbps": None,
                        "repeat": 0,
                    },
                ]
            )
            with (
                mock.patch.object(runner, "parse_args", return_value=args),
                mock.patch.object(runner.Path, "exists", return_value=True),
                mock.patch.object(runner, "selected_copy_cases", return_value=[]),
                mock.patch.object(
                    runner,
                    "level1_cases",
                    return_value=[("real_f32", "sdot", "default", 1, 1)],
                ),
                mock.patch.object(runner, "case_allowed", return_value=True),
                mock.patch.object(
                    runner, "libraries", return_value=[("Zynum", "zynum")]
                ),
                mock.patch.object(runner, "run_level1_op", side_effect=results),
                mock.patch.object(
                    runner.benchmark_metadata,
                    "collect_benchmark_identity_from_frozen",
                    return_value=self.unknown_identity(),
                ),
                redirect_stderr(io.StringIO()),
            ):
                self.assertEqual(runner.main(), 2)
            self.assertFalse(absent_parent.exists())

    def test_signed_stride_parser_and_legacy_inc_pairs(self):
        self.assertEqual(runner.parse_stride("-2147483648"), -(1 << 31))
        with self.assertRaises(Exception):
            runner.parse_stride("0")

        args = runner.parse_args(["--csv", os.devnull, "--inc", "2"])
        self.assertEqual(args.strides, [2])
        self.assertEqual(args.stride_pairs, [(2, 2)])

    def test_independent_stride_cli_builds_cartesian_pairs(self):
        args = runner.parse_args(
            [
                "--csv",
                os.devnull,
                "--incx",
                "-2",
                "--incx",
                "3",
                "--incy",
                "-1",
                "--incy",
                "4",
            ]
        )
        self.assertEqual(args.stride_pairs, [(-2, -1), (-2, 4), (3, -1), (3, 4)])
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            runner.parse_args(["--csv", os.devnull, "--inc", "2", "--incx", "-2"])

    def test_positive_cases_preserve_full_set_and_negative_cases_are_stable(self):
        self.assertEqual(runner.STABLE_NEGATIVE_OPS, checker.STABLE_NEGATIVE_OPS)
        positive = runner.level1_cases([(2, 2)])
        self.assertEqual(
            {(group, op) for group, op, _, _, _ in positive},
            set(runner.LEVEL1_OPS),
        )

        negative = runner.level1_cases([(-2, 3)])
        negative_ops = {op for _, op, _, _, _ in negative}
        self.assertEqual(negative_ops, runner.STABLE_NEGATIVE_OPS)
        self.assertTrue(
            {"scopy", "sswap", "saxpy", "sdot", "cdotu", "srot", "srotm"}
            <= negative_ops
        )
        self.assertTrue(
            negative_ops.isdisjoint(
                {"saxpby", "caxpby", "sscal", "sasum", "isamax", "snrm2"}
            )
        )

    def test_copy_byte_coverage_can_be_disabled_without_removing_vector_copy(self):
        args = runner.parse_args(
            [
                "--csv",
                os.devnull,
                "--incx",
                "-2",
                "--incy",
                "3",
                "--skip-copy-byte-coverage",
            ]
        )
        self.assertEqual(runner.selected_copy_cases(args), [])
        self.assertIn(
            ("copy", "scopy", "default", -2, 3),
            runner.level1_cases(args.stride_pairs),
        )

    def test_vector_layout_uses_signed_start_and_absolute_span(self):
        positive = runner.VectorLayout(5, 3)
        negative = runner.VectorLayout(5, -3)
        self.assertEqual((positive.span, positive.start), (13, 0))
        self.assertEqual((negative.span, negative.start), (13, 12))
        self.assertEqual([negative.index(i) for i in range(5)], [12, 9, 6, 3, 0])

    def test_guarded_array_allows_active_writes_but_rejects_gap_and_guard_writes(self):
        array, _ = runner.real_array(ctypes.c_float, 4, 1, -2)
        array.set_logical(0, ctypes.c_float(9.0))
        self.assertEqual(array.modified_element_count(True), 0)

        array[1] = ctypes.c_float(7.0)
        self.assertEqual(array.modified_element_count(True), 1)

        guarded, _ = runner.real_array(ctypes.c_float, 4, 1, -2)
        guarded.storage[0] = ctypes.c_float(3.0)
        self.assertEqual(guarded.modified_element_count(True), 1)

    def test_unstable_negative_operation_is_excluded_before_library_loading(self):
        result = runner.check_level1_op("/not/a/library", "snrm2", -2, 3)
        self.assertEqual(result["check_status"], "missing")
        self.assertEqual(result["capability_status"], "excluded-by-policy")

    def test_probe_output_records_actual_symbol_surface(self):
        output = (
            "iters=2 elapsed_ns=3 rate_Gops=4 bandwidth_GBps=5 checksum=6 "
            "symbol=cblas_sdot abi_surface=cblas\n"
        )
        self.assertEqual(
            runner.parse_probe_output(output),
            (4.0, 5.0, "cblas_sdot", "cblas"),
        )

    @mock.patch.object(runner, "run_once")
    @mock.patch.object(runner, "check_level1_op_isolated")
    def test_run_forwards_independent_strides_and_keeps_matching_surface(
        self, check, run_once
    ):
        check.return_value = runner.check_result(
            "sampled-ok",
            0.0,
            symbol="saxpy_",
            abi_surface="fortran",
            memory_status="guarded-ok",
        )
        run_once.return_value = {
            "status": "ok",
            "returncode": 0,
            "rate_gops": 1.0,
            "bandwidth_gbps": 2.0,
            "symbol": "saxpy_",
            "abi_surface": "fortran",
            "raw_output": "ok",
        }
        args = runner.parse_args(
            ["--csv", os.devnull, "--level1-probe", "probe", "--n", "17"]
        )
        row = runner.run_level1_op(
            args, "Test", "libtest.so", "real_f32", "saxpy", "default", -2, 3
        )
        command = run_once.call_args.args[0]
        self.assertEqual(command[command.index("--incx") + 1], "-2")
        self.assertEqual(command[command.index("--incy") + 1], "3")
        self.assertEqual(row["status"], "ok")
        self.assertEqual((row["incx"], row["incy"]), (-2, 3))

    @mock.patch.object(runner, "run_once")
    @mock.patch.object(runner, "check_level1_op_isolated")
    def test_unsupported_symbol_is_missing_and_never_timed(self, check, run_once):
        check.return_value = runner.check_result(
            "missing",
            raw="missing scopy_",
            capability_status="unsupported",
        )
        args = runner.parse_args(
            ["--csv", os.devnull, "--level1-probe", "probe", "--n", "17"]
        )
        row = runner.run_level1_op(
            args, "Test", "libtest.so", "copy", "scopy", "default", -2, 3
        )
        self.assertEqual(row["status"], "missing")
        self.assertEqual(row["capability_status"], "unsupported")
        run_once.assert_not_called()

    @mock.patch.object(runner, "run_once")
    @mock.patch.object(runner, "check_level1_op_isolated")
    def test_surface_mismatch_invalidates_timing(self, check, run_once):
        check.return_value = runner.check_result(
            "sampled-ok",
            0.0,
            symbol="sdot_",
            abi_surface="fortran",
            memory_status="guarded-ok",
        )
        run_once.return_value = {
            "status": "ok",
            "returncode": 0,
            "rate_gops": 9.0,
            "bandwidth_gbps": 9.0,
            "symbol": "cblas_sdot",
            "abi_surface": "cblas",
            "raw_output": "ok",
        }
        args = runner.parse_args(
            ["--csv", os.devnull, "--level1-probe", "probe", "--n", "17"]
        )
        with self.assertRaisesRegex(ValueError, "surface_mismatch"):
            runner.run_level1_op(
                args, "Test", "libtest.so", "real_f32", "sdot", "default", -2, 3
            )


class Level1CheckerTests(unittest.TestCase):
    def run_checker(self, rows, *extra_args):
        fields = sorted({field for row in rows for field in row})
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir).resolve() / "level1.csv"
            with path.open("w", newline="") as csv_file:
                writer = csv.DictWriter(csv_file, fieldnames=fields)
                writer.writeheader()
                writer.writerows(rows)
            stdout = io.StringIO()
            stderr = io.StringIO()
            with redirect_stdout(stdout), redirect_stderr(stderr):
                result = checker.main(
                    [
                        str(path),
                        "--comparator",
                        "Reference",
                        "--stat",
                        "median",
                        *extra_args,
                    ]
                )
        return result, stdout.getvalue(), stderr.getvalue()

    def test_checker_excludes_axpby_from_negative_gate(self):
        rows = [
            report_row("Zynum"),
            report_row("Reference", value=1.0),
            report_row(
                "Zynum",
                op="saxpby",
                value=0.01,
                symbol="saxpby_",
                preflight_symbol="saxpby_",
                abi_surface="fortran",
                preflight_abi_surface="fortran",
            ),
            report_row(
                "Reference",
                op="saxpby",
                value=100.0,
                symbol="saxpby_",
                preflight_symbol="saxpby_",
                abi_surface="fortran",
                preflight_abi_surface="fortran",
            ),
        ]
        result, stdout, stderr = self.run_checker(rows)
        self.assertEqual(result, 0, stderr)
        self.assertIn("checked=1 passed=1", stdout)
        self.assertIn("excluded_negative=2", stdout)

    def test_checker_accepts_explicit_legacy_rows_without_repeat_or_surface_columns(
        self,
    ):
        rows = [
            report_row("Zynum", incx="1", incy="1"),
            report_row("Reference", value=1.0, incx="1", incy="1"),
        ]
        for row in rows:
            for field in (
                "symbol",
                "abi_surface",
                "preflight_symbol",
                "preflight_abi_surface",
                "capability_status",
                "process_repeats",
                "successful_repeats",
            ):
                row.pop(field)
        result, stdout, stderr = self.run_checker(rows)
        self.assertEqual(result, 0, stderr)
        self.assertIn("checked=1 passed=1", stdout)

    def test_checker_enforces_complete_repeat_evidence(self):
        valid_rows = [report_row("Zynum"), report_row("Reference", value=1.0)]
        result, stdout, stderr = self.run_checker(
            valid_rows, "--expected-process-repeats", "2"
        )
        self.assertEqual(result, 0, stderr)
        self.assertIn("checked=1 passed=1", stdout)

        invalid_sets = []
        missing_column = [dict(row) for row in valid_rows]
        for row in missing_column:
            row.pop("successful_repeats")
        invalid_sets.append(missing_column)
        invalid_sets.extend(
            [
                [
                    report_row("Zynum", successful_repeats=value),
                    report_row("Reference", value=1.0),
                ]
                for value in ("", "0", "-1", "1.5", "01", "1")
            ]
        )
        for rows in invalid_sets:
            with self.subTest(rows=rows):
                result, _, stderr = self.run_checker(rows)
                self.assertEqual(result, 2)
                self.assertIn("bad repeat evidence", stderr)

        result, _, stderr = self.run_checker(
            valid_rows, "--expected-process-repeats", "3"
        )
        self.assertEqual(result, 2)
        self.assertIn("expected 3", stderr)

    def test_expected_repeats_rejects_legacy_csv_without_repeat_columns(self):
        rows = [report_row("Zynum"), report_row("Reference", value=1.0)]
        for row in rows:
            row.pop("process_repeats")
            row.pop("successful_repeats")
        result, _, stderr = self.run_checker(rows, "--expected-process-repeats", "2")
        self.assertEqual(result, 2)
        self.assertIn("legacy row has no process-repeat evidence", stderr)

    def test_negative_only_ignores_positive_rows_in_same_report(self):
        positive_zynum = report_row("Zynum", value=0.01, incx="1", incy="1")
        positive_reference = report_row("Reference", value=100.0, incx="1", incy="1")
        rows = [
            report_row("Zynum"),
            report_row("Reference", value=1.0),
            positive_zynum,
            positive_reference,
        ]
        result, stdout, stderr = self.run_checker(rows, "--negative-only")
        self.assertEqual(result, 0, stderr)
        self.assertIn("checked=1 passed=1", stdout)

    def test_checker_rejects_negative_row_without_surface_preflight(self):
        row = report_row("Zynum", symbol="", abi_surface="")
        result, _, stderr = self.run_checker([row, report_row("Reference")])
        self.assertEqual(result, 2)
        self.assertIn("lacks a supported capability surface", stderr)

    def test_checker_rejects_nonpositive_timing_metric(self):
        result, _, stderr = self.run_checker(
            [report_row("Zynum", value=0.0), report_row("Reference", value=1.0)]
        )
        self.assertEqual(result, 2)
        self.assertIn("bad metric value", stderr)

    def test_checker_rejects_invalid_metrics_thresholds_and_optional_evidence(self):
        for value in ("nan", "inf", "-inf", "0", "-1"):
            with self.subTest(metric=value):
                result, _, _ = self.run_checker(
                    [
                        report_row("Zynum", metric_median=value),
                        report_row("Reference", value=1.0),
                    ]
                )
                self.assertEqual(result, 2)
            with self.subTest(threshold=value):
                result, _, _ = self.run_checker(
                    [report_row("Zynum"), report_row("Reference")],
                    f"--ratio={value}",
                )
                self.assertEqual(result, 2)
        result, _, stderr = self.run_checker(
            [
                report_row("Zynum", metric_max="nan"),
                report_row("Reference"),
            ]
        )
        self.assertEqual(result, 2)
        self.assertIn("bad metric value", stderr)

    def test_checker_rejects_duplicate_rows_even_when_values_match(self):
        for duplicate in (
            report_row("Zynum"),
            report_row("Zynum", value=9.0),
        ):
            with self.subTest(value=duplicate["metric_median"]):
                result, _, stderr = self.run_checker(
                    [report_row("Zynum"), duplicate, report_row("Reference")]
                )
                self.assertEqual(result, 2)
                self.assertIn("duplicate library row", stderr)

    def test_checker_output_is_stable_under_csv_shuffle_and_comparator_ties(self):
        rows = [
            report_row("Zynum", op="sdot", value=0.5),
            report_row("Reference", op="sdot", value=1.0),
            report_row("Second", op="sdot", value=1.0),
            report_row("Zynum", op="ddot", value=0.25),
            report_row("Reference", op="ddot", value=1.0),
        ]
        first = self.run_checker(rows, "--comparator", "Second")
        shuffled = self.run_checker(list(reversed(rows)), "--comparator", "Second")
        self.assertEqual(first, shuffled)
        self.assertEqual(first[0], 1)
        self.assertIn("best=Reference:1.000000", first[1])


@unittest.skipUnless(
    runner.library_available(runner.DEFAULT_ACCELERATE), "Accelerate is unavailable"
)
class Level1AccelerateIntegrationTests(unittest.TestCase):
    def test_representative_mixed_sign_stable_operations(self):
        cases = (
            ("scopy", "default"),
            ("sswap", "default"),
            ("saxpy", "default"),
            ("sdot", "default"),
            ("cdotu", "default"),
            ("csrot", "default"),
            ("srotm", "flag_0"),
        )
        for op, variant in cases:
            with self.subTest(op=op, variant=variant):
                result = runner.check_level1_op(
                    runner.DEFAULT_ACCELERATE, op, -2, 3, variant
                )
                self.assertEqual(result["check_status"], "sampled-ok", result)
                self.assertEqual(result["capability_status"], "supported")
                self.assertEqual(result["check_memory_status"], "guarded-ok")
                self.assertTrue(result["preflight_symbol"])
                self.assertTrue(result["preflight_abi_surface"])


if __name__ == "__main__":
    unittest.main()
