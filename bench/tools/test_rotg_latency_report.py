#!/usr/bin/env python3
# Copyright (C) 2026 Zynum contributors
# SPDX-License-Identifier: LGPL-3.0-or-later

import csv
import hashlib
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
from unittest import mock


TOOLS_DIR = Path(__file__).resolve().parent
REPO_ROOT = TOOLS_DIR.parents[1]


def write_artifact(directory, name, contents=b"artifact", *, executable=False):
    path = Path(directory) / name
    path.write_bytes(contents)
    if executable:
        path.chmod(0o700)
    return path


def load_tool(module_name):
    spec = importlib.util.spec_from_file_location(
        module_name, TOOLS_DIR / (module_name + ".py")
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


runner = load_tool("run_rotg_latency_report")
checker = load_tool("check_rotg_latency_report")


def report_row(
    library,
    routine="drotg",
    input_case="balanced",
    metric_min=8.0,
    metric_median=10.0,
    metric_max=12.0,
    status="ok",
    check_status="checked-ok",
):
    kind = runner.ROUTINES[routine][0]
    is_rotmg = runner.ROUTINES[routine][1]
    return {
        "level": "level1",
        "routine": routine,
        "kind": kind,
        "library": library,
        "library_path": "lib{}.so".format(library),
        "case": input_case,
        "corpus_size": "2",
        "samples": "5",
        "calls_per_sample": "1000",
        "total_calls": "5000",
        "best_ns_per_call": "7",
        "median_ns_per_call": "9",
        "p95_ns_per_call": "11",
        "max_ns_per_call": "12",
        "median_full_ns_per_call": "15",
        "median_harness_ns_per_call": "6",
        "nonpositive_pairs": "0",
        "metric": "ns_per_call",
        "status": status,
        "check_status": check_status,
        "check_max_abs_error": "0",
        "check_max_rel_error": "0",
        "check_samples": "12",
        "expected_flag": (
            format(runner.EXPECTED_FLAGS[input_case], ".17g") if is_rotmg else ""
        ),
        "observed_flag": (
            format(runner.EXPECTED_FLAGS[input_case], ".17g") if is_rotmg else ""
        ),
        "checksum": "12345",
        "check_raw_output": "",
        "process_repeats": "3",
        "successful_repeats": "3" if status == "ok" else "2",
        "metric_min": str(metric_min),
        "metric_median": str(metric_median),
        "metric_max": str(metric_max),
        "metric_samples": "{},{},{}".format(metric_min, metric_median, metric_max),
    }


class RotgLatencyRunnerTests(unittest.TestCase):
    def test_library_availability_never_loads_a_live_path(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            missing_library = Path(temp_dir) / "lib-not-a-file.so"
            self.assertFalse(
                runner.library_available("Comparator", str(missing_library))
            )

    def minimal_controller_args(self, temp_dir, output):
        probe = write_artifact(temp_dir, "probe", executable=True)
        zynum = write_artifact(temp_dir, "libzynum.so")
        return runner.parse_args(
            [
                "--probe",
                str(probe),
                "--zynum",
                str(zynum),
                "--accelerate",
                "none",
                "--openblas",
                "none",
                "--routine",
                "drotg",
                "--case",
                "balanced",
                "--process-repeats",
                "1",
                "--csv",
                str(output),
            ]
        )

    def test_controller_calls_publisher_once_with_exact_ordered_bytes(self):
        identity = {"schema_version": 2, "source": {}}
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir).resolve() / "rotg.csv"
            args = self.minimal_controller_args(temp_dir, output)
            row = runner.aggregate_repeats([report_row("Zynum")])
            expected_csv = io.StringIO(newline="")
            writer = csv.DictWriter(expected_csv, fieldnames=runner.CSV_FIELDNAMES)
            writer.writeheader()
            writer.writerow(row)
            with (
                mock.patch.object(
                    runner,
                    "selected_libraries",
                    return_value=[("Zynum", args.zynum)],
                ),
                mock.patch.object(
                    runner, "run_one_process", return_value=report_row("Zynum")
                ),
                mock.patch.object(
                    runner.benchmark_metadata,
                    "collect_benchmark_identity_from_frozen",
                    return_value=identity,
                ),
                mock.patch.object(runner, "command_output", return_value="0.16.0"),
                mock.patch.object(runner, "publish_outputs") as publisher,
                redirect_stderr(io.StringIO()),
            ):
                runner.run_controller(args)

            publisher.assert_called_once()
            published = publisher.call_args.args[0]
            self.assertIsInstance(published, list)
            self.assertEqual(
                [item.path for item in published],
                [output, output.with_suffix(".csv.meta.json")],
            )
            self.assertTrue(all(type(item.contents) is bytes for item in published))
            self.assertEqual(
                published[0].contents, expected_csv.getvalue().encode("utf-8")
            )
            self.assertTrue(published[0].contents.endswith(b"\r\n"))
            self.assertTrue(published[1].contents.endswith(b"\n"))
            self.assertNotIn(b"\r\n", published[1].contents)
            metadata = json.loads(published[1].contents)
            self.assertEqual(
                published[1].contents,
                (json.dumps(metadata, indent=2, sort_keys=True) + "\n").encode("utf-8"),
            )
            self.assertEqual(metadata["benchmark_identity"], identity)
            self.assertFalse(output.exists())

    def test_metadata_serialization_failure_precedes_publication(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "rotg.csv"
            metadata = output.with_suffix(".csv.meta.json")
            output.write_bytes(b"old csv\n")
            metadata.write_bytes(b"old metadata\n")
            args = self.minimal_controller_args(temp_dir, output)
            with (
                mock.patch.object(
                    runner,
                    "selected_libraries",
                    return_value=[("Zynum", args.zynum)],
                ),
                mock.patch.object(
                    runner, "run_one_process", return_value=report_row("Zynum")
                ),
                mock.patch.object(
                    runner.benchmark_metadata,
                    "collect_benchmark_identity_from_frozen",
                    return_value={"source": {}, "invalid": float("nan")},
                ),
                mock.patch.object(runner, "command_output", return_value="0.16.0"),
                mock.patch.object(runner, "publish_outputs") as publisher,
                redirect_stderr(io.StringIO()),
            ):
                with self.assertRaisesRegex(ValueError, "not JSON compliant"):
                    runner.run_controller(args)

            publisher.assert_not_called()
            self.assertEqual(output.read_bytes(), b"old csv\n")
            self.assertEqual(metadata.read_bytes(), b"old metadata\n")

    def test_publisher_failure_cannot_split_existing_pair(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "rotg.csv"
            metadata = output.with_suffix(".csv.meta.json")
            output.write_bytes(b"old csv\n")
            metadata.write_bytes(b"old metadata\n")
            args = self.minimal_controller_args(temp_dir, output)
            with (
                mock.patch.object(
                    runner,
                    "selected_libraries",
                    return_value=[("Zynum", args.zynum)],
                ),
                mock.patch.object(
                    runner, "run_one_process", return_value=report_row("Zynum")
                ),
                mock.patch.object(
                    runner.benchmark_metadata,
                    "collect_benchmark_identity_from_frozen",
                    return_value={"source": {}},
                ),
                mock.patch.object(runner, "command_output", return_value="0.16.0"),
                mock.patch.object(
                    runner, "publish_outputs", side_effect=OSError("publish failed")
                ) as publisher,
                redirect_stderr(io.StringIO()),
            ):
                with self.assertRaisesRegex(OSError, "publish failed"):
                    runner.run_controller(args)

            publisher.assert_called_once()
            self.assertEqual(output.read_bytes(), b"old csv\n")
            self.assertEqual(metadata.read_bytes(), b"old metadata\n")

    def test_absent_parent_is_not_created_before_publisher(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir).resolve() / "absent" / "rotg.csv"
            args = self.minimal_controller_args(temp_dir, output)

            def fail_publication(_outputs):
                self.assertFalse(output.parent.exists())
                raise OSError("publish failed")

            with (
                mock.patch.object(
                    runner,
                    "selected_libraries",
                    return_value=[("Zynum", args.zynum)],
                ),
                mock.patch.object(
                    runner, "run_one_process", return_value=report_row("Zynum")
                ),
                mock.patch.object(
                    runner.benchmark_metadata,
                    "collect_benchmark_identity_from_frozen",
                    return_value={"source": {}},
                ),
                mock.patch.object(runner, "command_output", return_value="0.16.0"),
                mock.patch.object(
                    runner, "publish_outputs", side_effect=fail_publication
                ),
                redirect_stderr(io.StringIO()),
            ):
                with self.assertRaisesRegex(OSError, "publish failed"):
                    runner.run_controller(args)

            self.assertFalse(output.parent.exists())

    def test_controller_reuses_one_frozen_probe_and_library_without_private_leaks(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            probe = write_artifact(temp_dir, "probe", b"probe-a", executable=True)
            library = write_artifact(temp_dir, "libzynum.so", b"library-a")
            output = Path(temp_dir) / "rotg.csv"
            args = runner.parse_args(
                [
                    "--probe",
                    str(probe),
                    "--zynum",
                    str(library),
                    "--accelerate",
                    "none",
                    "--openblas",
                    "none",
                    "--routine",
                    "drotg",
                    "--case",
                    "balanced",
                    "--process-repeats",
                    "3",
                    "--csv",
                    str(output),
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
                return {
                    "schema_version": 2,
                    "source": {},
                    "payload": {"artifacts": identities[-1]},
                }

            def run_one(
                _args,
                _library_name,
                library_path,
                _case,
                *,
                probe_path=None,
            ):
                private_paths.extend((str(probe_path), str(library_path)))
                self.assertNotEqual(str(probe_path), str(probe))
                self.assertNotEqual(str(library_path), str(library))
                self.assertEqual(Path(probe_path).read_bytes(), b"probe-a")
                self.assertEqual(Path(library_path).read_bytes(), b"library-a")
                if len(private_paths) == 2:
                    probe.write_bytes(b"probe-b")
                    library.write_bytes(b"library-b")
                elif len(private_paths) == 4:
                    probe.write_bytes(b"probe-a")
                    library.write_bytes(b"library-a")
                row = report_row("Zynum")
                row["library_path"] = str(library_path)
                row["check_raw_output"] = "{} {}".format(probe_path, library_path)
                return row

            def publish(outputs):
                self.assertEqual(len(private_paths), 6)
                for path in private_paths:
                    self.assertFalse(Path(path).exists())
                    self.assertNotIn(path.encode(), outputs[0].contents)
                    self.assertNotIn(path.encode(), outputs[1].contents)
                rows = list(
                    csv.DictReader(io.StringIO(outputs[0].contents.decode("utf-8")))
                )
                self.assertEqual(rows[0]["library_path"], str(library))
                self.assertIn(str(probe), rows[0]["check_raw_output"])
                metadata = json.loads(outputs[1].contents)
                probe_sha = hashlib.sha256(b"probe-a").hexdigest()
                library_sha = hashlib.sha256(b"library-a").hexdigest()
                self.assertNotIn("path", metadata["probe"])
                self.assertEqual(metadata["probe"]["sha256"], probe_sha)
                self.assertNotIn("path", metadata["libraries"][0])
                self.assertEqual(metadata["libraries"][0]["sha256"], library_sha)
                self.assertEqual(
                    metadata["benchmark_identity"]["payload"]["artifacts"]["binaries"][
                        0
                    ]["sha256"],
                    probe_sha,
                )

            with (
                mock.patch.object(
                    runner,
                    "selected_libraries",
                    return_value=[("Zynum", str(library))],
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
                mock.patch.object(runner, "command_output", return_value="0.16.0"),
                redirect_stderr(io.StringIO()),
            ):
                runner.run_controller(args)

            capture_snapshot.assert_called_once()
            self.assertEqual(len(identities), 1)
            self.assertEqual(probe.read_bytes(), b"probe-a")
            self.assertEqual(library.read_bytes(), b"library-a")

    def test_stage_drift_and_cleanup_failure_never_publish_or_replace_outputs(self):
        for failure_mode in ("finalize", "close"):
            with self.subTest(failure_mode=failure_mode):
                with tempfile.TemporaryDirectory() as temp_dir:
                    probe = write_artifact(
                        temp_dir, "probe", b"probe-a", executable=True
                    )
                    library = write_artifact(temp_dir, "libzynum.so", b"library-a")
                    output = Path(temp_dir) / "rotg.csv"
                    metadata = output.with_suffix(".csv.meta.json")
                    output.write_bytes(b"old csv\n")
                    metadata.write_bytes(b"old metadata\n")
                    args = runner.parse_args(
                        [
                            "--probe",
                            str(probe),
                            "--zynum",
                            str(library),
                            "--accelerate",
                            "none",
                            "--openblas",
                            "none",
                            "--routine",
                            "drotg",
                            "--case",
                            "balanced",
                            "--process-repeats",
                            "1",
                            "--csv",
                            str(output),
                        ]
                    )
                    private_probe = []
                    original_finalize = (
                        runner.benchmark_artifacts.ArtifactSnapshotSet.finalize
                    )
                    original_close = (
                        runner.benchmark_artifacts.ArtifactSnapshotSet.close
                    )

                    def run_one(
                        _args,
                        _name,
                        _library_path,
                        _case,
                        *,
                        probe_path=None,
                    ):
                        private_probe.append(str(probe_path))
                        return report_row("Zynum")

                    def drift(snapshot):
                        os.chmod(private_probe[0], 0o600)
                        descriptor = os.open(private_probe[0], os.O_WRONLY)
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

                    patches = [
                        mock.patch.object(
                            runner,
                            "selected_libraries",
                            return_value=[("Zynum", str(library))],
                        ),
                        mock.patch.object(
                            runner, "run_one_process", side_effect=run_one
                        ),
                        mock.patch.object(
                            runner.benchmark_metadata,
                            "collect_benchmark_identity_from_frozen",
                            return_value={"schema_version": 2, "source": {}},
                        ),
                        mock.patch.object(runner, "publish_outputs"),
                        mock.patch.object(
                            runner, "command_output", return_value="0.16.0"
                        ),
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
                        patches[3] as publisher,
                        patches[4] as _,
                        patches[5] as _,
                        patches[6] as _,
                        redirect_stderr(io.StringIO()),
                    ):
                        with self.assertRaises(
                            runner.benchmark_artifacts.ArtifactCleanupError
                        ) as raised:
                            runner.run_controller(args)

                    publisher.assert_not_called()
                    self.assertEqual(output.read_bytes(), b"old csv\n")
                    self.assertEqual(metadata.read_bytes(), b"old metadata\n")
                    self.assertFalse(Path(private_probe[0]).exists())
                    if failure_mode == "finalize":
                        self.assertIsInstance(
                            raised.exception.__context__,
                            runner.benchmark_artifacts.ArtifactVerificationError,
                        )
                        retained = [
                            Path(path)
                            for path in raised.exception.recovery_paths
                            if Path(path).name == Path(private_probe[0]).name
                            and Path(path).is_file()
                        ]
                        self.assertEqual(len(retained), 1)
                        drifted = retained[0]
                        self.assertTrue(
                            drifted.parent.name.startswith(
                                ".zynum-benchmark-artifact-quarantine-"
                            )
                        )
                        self.assertEqual(drifted.read_bytes(), b"Brobe-a")
                        self.assertIn(
                            "private_artifact_replaced",
                            {issue.code for issue in raised.exception.issues},
                        )
                        drifted.unlink()
                        drifted.parent.rmdir()
                    else:
                        self.assertFalse(raised.exception.recovery_paths)

    def test_process_schedule_is_the_only_supported_schedule_option(self):
        with mock.patch.object(runner.sys, "platform", "win32"):
            windows = runner.parse_args(["--csv", os.devnull])
        self.assertEqual(windows.probe, "zig-out/bin/rotg-latency-probe.exe")
        self.assertEqual(windows.zynum, "zig-out/bin/zynum_blas.dll")
        self.assertEqual(
            runner.parse_args(["--csv", os.devnull]).process_schedule,
            "library-major",
        )
        self.assertEqual(
            runner.parse_args(
                ["--csv", os.devnull, "--process-schedule", "interleaved"]
            ).process_schedule,
            "interleaved",
        )
        invalid_argvs = (
            ["--csv", os.devnull, "--process-schedule", "invalid"],
            ["--csv", os.devnull, "--schedule", "interleaved"],
        )
        for argv in invalid_argvs:
            with self.subTest(argv=argv), redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as raised:
                    runner.parse_args(argv)
                self.assertEqual(raised.exception.code, 2)

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
            target = write_artifact(temp_dir, "libaccelerate-target.dylib")
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
                            "sha256": hashlib.sha256(b"artifact").hexdigest(),
                        }
                    ],
                )
                snapshot.finalize()

    def test_only_missing_default_macos_accelerate_is_a_platform_image(self):
        with (
            mock.patch.object(runner.sys, "platform", "darwin"),
            mock.patch.object(runner.benchmark_artifacts.sys, "platform", "darwin"),
            mock.patch.object(
                runner,
                "DEFAULT_ACCELERATE",
                runner.benchmark_artifacts.DEFAULT_ACCELERATE_IMAGE,
            ),
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

    def test_interleaved_single_library_accepts_arbitrary_positive_repeats(self):
        case = runner.LatencyCase("drotg", "balanced")
        order = []
        samples = runner.collect_repeats(
            [("Zynum", "libzynum.so")],
            [case],
            5,
            "interleaved",
            lambda library, case_index, repeat: (
                order.append((library, case_index, repeat)),
                repeat,
            )[1],
            lambda *_: None,
        )
        self.assertEqual(order, [(0, 0, repeat) for repeat in range(5)])
        self.assertEqual(samples, [[[0, 1, 2, 3, 4]]])

    def test_controller_uses_actual_library_count_and_one_cached_identity(self):
        source = {
            "revision": None,
            "branch": None,
            "dirty": None,
            "status_short": None,
            "identity_status": "unreadable",
        }
        identity = {"schema_version": 2, "source": source}
        with tempfile.TemporaryDirectory() as temp_dir:
            probe = write_artifact(temp_dir, "probe", executable=True)
            zynum = write_artifact(temp_dir, "zynum")
            accelerate = write_artifact(temp_dir, "accelerate")
            output = Path(temp_dir).resolve() / "rotg.csv"
            args = runner.parse_args(
                [
                    "--probe",
                    str(probe),
                    "--zynum",
                    str(zynum),
                    "--accelerate",
                    str(accelerate),
                    "--openblas",
                    "missing",
                    "--skip-missing",
                    "--routine",
                    "drotg",
                    "--case",
                    "balanced",
                    "--process-repeats",
                    "2",
                    "--process-schedule",
                    "interleaved",
                    "--csv",
                    str(output),
                ]
            )
            execution_order = []

            def run_one(_args, library_name, _library_path, _case, **_kwargs):
                execution_order.append(library_name)
                return report_row(library_name)

            with (
                mock.patch.object(
                    runner,
                    "library_available",
                    side_effect=lambda _name, path: path != "missing",
                ),
                mock.patch.object(runner, "run_one_process", side_effect=run_one),
                mock.patch.object(
                    runner.benchmark_metadata,
                    "collect_benchmark_identity_from_frozen",
                    return_value=identity,
                ) as collect_identity,
                mock.patch.object(
                    runner, "collect_repeats", wraps=runner.collect_repeats
                ) as collect_samples,
                mock.patch.object(
                    runner, "command_output", return_value="0.16.0"
                ) as command_output,
                redirect_stderr(io.StringIO()),
            ):
                runner.run_controller(args)

            selected = [("Zynum", str(zynum)), ("Accelerate", str(accelerate))]
            collect_identity.assert_called_once()
            self.assertEqual(
                [
                    artifact.path
                    for artifact in collect_identity.call_args.kwargs["libraries"]
                ],
                [str(zynum), str(accelerate)],
            )
            self.assertEqual(
                [
                    artifact.path
                    for artifact in collect_identity.call_args.kwargs["binaries"]
                ],
                [str(probe)],
            )
            self.assertEqual(collect_samples.call_args.args[0], selected)
            self.assertEqual(
                execution_order,
                ["Zynum", "Accelerate", "Accelerate", "Zynum"],
            )
            command_output.assert_called_once_with(["zig", "version"])
            with output.open(newline="") as file:
                self.assertEqual(
                    [row["library"] for row in csv.DictReader(file)],
                    ["Zynum", "Accelerate"],
                )
            metadata = json.loads(
                output.with_suffix(output.suffix + ".meta.json").read_text()
            )
            self.assertEqual(metadata["schedule"], "interleaved")
            self.assertEqual(metadata["benchmark_identity"]["schema_version"], 2)
            self.assertNotIn("status_short", metadata["benchmark_identity"]["source"])
            self.assertEqual(
                metadata["source"],
                {
                    "revision": None,
                    "branch": None,
                    "dirty": None,
                },
            )

    def test_invalid_actual_schedule_prevents_identity_payload_and_publication(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            probe = write_artifact(temp_dir, "probe", executable=True)
            output = Path(temp_dir) / "rotg.csv"
            metadata = output.with_suffix(output.suffix + ".meta.json")
            output.write_text("existing csv\n")
            metadata.write_text("existing metadata\n")
            argv = [
                "--probe",
                str(probe),
                "--zynum",
                "zynum",
                "--accelerate",
                "accelerate",
                "--openblas",
                "missing",
                "--skip-missing",
                "--routine",
                "drotg",
                "--case",
                "balanced",
                "--process-repeats",
                "3",
                "--process-schedule",
                "interleaved",
                "--csv",
                str(output),
            ]
            with (
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
                mock.patch.object(runner, "collect_repeats") as collect_samples,
                mock.patch.object(runner, "run_one_process") as run_payload,
                redirect_stderr(io.StringIO()),
            ):
                self.assertEqual(runner.main(argv), 2)

            collect_identity.assert_not_called()
            capture_snapshot.assert_not_called()
            collect_samples.assert_not_called()
            run_payload.assert_not_called()
            self.assertEqual(output.read_text(), "existing csv\n")
            self.assertEqual(metadata.read_text(), "existing metadata\n")

    def test_duplicate_library_labels_fail_before_identity_payload_and_publication(
        self,
    ):
        with tempfile.TemporaryDirectory() as temp_dir:
            probe = write_artifact(temp_dir, "probe", executable=True)
            output = Path(temp_dir) / "rotg.csv"
            metadata = output.with_suffix(output.suffix + ".meta.json")
            output.write_text("existing csv\n")
            metadata.write_text("existing metadata\n")
            argv = [
                "--probe",
                str(probe),
                "--zynum",
                "zynum",
                "--accelerate",
                "missing",
                "--openblas",
                "none",
                "--extra-blas",
                "libzynum-blas=alias",
                "--skip-missing",
                "--csv",
                str(output),
            ]
            with (
                mock.patch.object(runner, "requested_cases", return_value=[object()]),
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
                self.assertEqual(runner.main(argv), 2)

            self.assertIn("duplicate semantic library label", stderr.getvalue())
            collect_identity.assert_not_called()
            capture_snapshot.assert_not_called()
            run_payload.assert_not_called()
            publish_metadata.assert_not_called()
            self.assertEqual(output.read_text(), "existing csv\n")
            self.assertEqual(metadata.read_text(), "existing metadata\n")

    def test_valid_single_library_controller_collects_identity_once(self):
        identity = {
            "schema_version": 2,
            "source": {
                "revision": None,
                "branch": None,
                "dirty": None,
                "status_short": None,
            },
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            probe = write_artifact(temp_dir, "probe", executable=True)
            zynum = write_artifact(temp_dir, "zynum")
            args = runner.parse_args(
                [
                    "--probe",
                    str(probe),
                    "--zynum",
                    str(zynum),
                    "--accelerate",
                    "none",
                    "--openblas",
                    "none",
                    "--routine",
                    "drotg",
                    "--case",
                    "balanced",
                    "--process-repeats",
                    "5",
                    "--process-schedule",
                    "interleaved",
                    "--csv",
                    str(Path(temp_dir).resolve() / "rotg.csv"),
                ]
            )
            with (
                mock.patch.object(
                    runner,
                    "selected_libraries",
                    return_value=[("Zynum", str(zynum))],
                ),
                mock.patch.object(
                    runner,
                    "run_one_process",
                    side_effect=lambda *_, **__: report_row("Zynum"),
                ) as run_payload,
                mock.patch.object(
                    runner.benchmark_metadata,
                    "collect_benchmark_identity_from_frozen",
                    return_value=identity,
                ) as collect_identity,
                mock.patch.object(runner, "command_output", return_value="0.16.0"),
                redirect_stderr(io.StringIO()),
            ):
                runner.run_controller(args)

            collect_identity.assert_called_once()
            self.assertEqual(
                collect_identity.call_args.kwargs["libraries"][0].path, str(zynum)
            )
            self.assertEqual(
                collect_identity.call_args.kwargs["binaries"][0].path, str(probe)
            )
            self.assertEqual(run_payload.call_count, 5)

    def test_aggregate_rejects_every_invalid_performance_sample(self):
        fields = (
            "best_ns_per_call",
            "median_ns_per_call",
            "p95_ns_per_call",
            "max_ns_per_call",
            "median_full_ns_per_call",
            "median_harness_ns_per_call",
        )
        for field in fields:
            for value in ("0", "nan", "inf", "-inf", "1e999999"):
                row = report_row("Zynum")
                row[field] = value
                with self.subTest(field=field, value=value):
                    with self.assertRaises(ValueError):
                        runner.aggregate_repeats([row])

    def test_default_matrix_covers_all_routines_exponents_and_rotmg_flags(self):
        args = runner.parse_args(["--csv", os.devnull])
        cases = runner.requested_cases(args)
        self.assertEqual(
            len(cases), 4 * len(runner.ROTG_CASES) + 2 * len(runner.ROTMG_CASES)
        )
        self.assertEqual(args.process_repeats, 3)
        by_routine = {}
        for case in cases:
            by_routine.setdefault(case.routine, set()).add(case.input_case)
        for routine in ("srotg", "drotg", "crotg", "zrotg"):
            self.assertEqual(by_routine[routine], set(runner.ROTG_CASES))
            self.assertIn("tiny_exponent", by_routine[routine])
            self.assertIn("huge_exponent", by_routine[routine])
        for routine in ("srotmg", "drotmg"):
            self.assertEqual(by_routine[routine], set(runner.ROTMG_CASES))
            self.assertEqual(
                {runner.EXPECTED_FLAGS[value] for value in by_routine[routine]},
                {-2.0, -1.0, 0.0, 1.0},
            )

    def test_incompatible_explicit_case_is_rejected(self):
        args = runner.parse_args(
            [
                "--csv",
                os.devnull,
                "--routine",
                "srotg",
                "--case",
                "flag_one_q2_dominant",
            ]
        )
        with self.assertRaisesRegex(ValueError, "not valid"):
            runner.requested_cases(args)

    def test_case_command_forwards_complete_measurement_parameters(self):
        args = runner.parse_args(
            [
                "--csv",
                os.devnull,
                "--probe",
                "probe",
                "--samples",
                "7",
                "--calls-per-sample",
                "1234",
            ]
        )
        case = runner.LatencyCase("drotmg", "flag_zero_q1_dominant")
        command = runner.case_command(args, "MKL", "libmkl_rt.so", case)
        pairs = dict(zip(command[1::2], command[2::2]))
        self.assertEqual(pairs["--blas"], "libmkl_rt.so")
        self.assertEqual(pairs["--library"], "MKL")
        self.assertEqual(pairs["--routine"], "drotmg")
        self.assertEqual(pairs["--case"], "flag_zero_q1_dominant")
        self.assertEqual(pairs["--samples"], "7")
        self.assertEqual(pairs["--calls-per-sample"], "1234")

    def test_probe_flag_matching_is_numeric_not_textual(self):
        args = runner.parse_args(["--csv", os.devnull])
        case = runner.LatencyCase("drotmg", "flag_neg2_zero_p2")
        row = runner.error_row(args, "Zynum", "libzynum.so", case, "")
        row.update(
            {
                "expected_flag": "-2.00000000000000000",
                "level": "level1",
                "routine": "drotmg",
                "kind": "f64",
                "library": "Zynum",
                "library_path": "libzynum.so",
                "case": "flag_neg2_zero_p2",
                "samples": "9",
                "calls_per_sample": "100000",
                "total_calls": "900000",
                "metric": "ns_per_call",
            }
        )
        self.assertEqual(
            runner.probe_row_mismatches(args, row, "Zynum", "libzynum.so", case),
            [],
        )

    @mock.patch.object(runner.subprocess, "run")
    def test_probe_failure_becomes_parameterized_error_row(self, run):
        run.return_value = subprocess.CompletedProcess(
            ["probe"], 1, stdout="", stderr="error: MissingSymbol"
        )
        args = runner.parse_args(["--csv", os.devnull, "--probe", "probe"])
        case = runner.LatencyCase("srotmg", "flag_one_q2_dominant")
        row = runner.run_one_process(args, "TestBLAS", "libblas.so", case)
        self.assertEqual(row["status"], "error")
        self.assertEqual(row["routine"], "srotmg")
        self.assertEqual(row["case"], "flag_one_q2_dominant")
        self.assertEqual(row["expected_flag"], "1")
        self.assertIn("MissingSymbol", row["check_raw_output"])


class RotgLatencyAggregationTests(unittest.TestCase):
    def test_process_aggregate_uses_median_of_probe_medians(self):
        rows = []
        for value in (12.0, 8.0, 10.0):
            row = report_row("Zynum")
            row["median_ns_per_call"] = str(value)
            rows.append(row)
        aggregate = runner.aggregate_repeats(rows)
        self.assertEqual(aggregate["process_repeats"], 3)
        self.assertEqual(aggregate["successful_repeats"], 3)
        self.assertEqual(aggregate["metric_min"], "8")
        self.assertEqual(aggregate["metric_median"], "10")
        self.assertEqual(aggregate["metric_max"], "12")
        self.assertEqual(aggregate["metric_samples"], "12,8,10")

    def test_even_process_median_is_safe_at_finite_float_extremes(self):
        for value in (1e308, 5e-324):
            rows = []
            for _ in range(2):
                row = report_row("Zynum")
                row["median_ns_per_call"] = str(value)
                rows.append(row)
            with self.subTest(value=value):
                aggregate = runner.aggregate_repeats(rows)
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
        with self.assertRaisesRegex(ValueError, "metric_median"):
            runner.aggregate_repeats([report_row("Zynum"), report_row("Zynum")])

    @mock.patch.object(
        runner,
        "run_controller",
        side_effect=ValueError("metric_median must be finite and positive"),
    )
    def test_invalid_derived_evidence_returns_two_before_publication(self, _run):
        with redirect_stderr(io.StringIO()):
            self.assertEqual(runner.main(["--csv", os.devnull]), 2)

    def test_any_bad_repeat_contaminates_aggregate(self):
        good = report_row("Zynum")
        bad = report_row(
            "Zynum",
            status="correctness_failed",
            check_status="correctness_failed",
        )
        bad["check_raw_output"] = "reference tolerance exceeded"
        aggregate = runner.aggregate_repeats([good, bad])
        self.assertEqual(aggregate["successful_repeats"], 1)
        self.assertEqual(aggregate["status"], "correctness_failed")
        self.assertEqual(aggregate["check_status"], "correctness_failed")
        self.assertIn("repeat=2", aggregate["check_raw_output"])


class RotgLatencyCheckerTests(unittest.TestCase):
    def run_checker(self, rows, *extra_args):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "rotg_latency.csv"
            with path.open("w", newline="") as file:
                writer = csv.DictWriter(file, fieldnames=runner.CSV_FIELDNAMES)
                writer.writeheader()
                writer.writerows(rows)
            stdout = io.StringIO()
            stderr = io.StringIO()
            with redirect_stdout(stdout), redirect_stderr(stderr):
                result = checker.main(
                    [str(path), "--comparator", "Reference"] + list(extra_args)
                )
        return result, stdout.getvalue(), stderr.getvalue()

    def test_checker_compares_fresh_process_median_latency(self):
        rows = [
            report_row("Zynum", metric_min=1.0, metric_median=10.0, metric_max=11.0),
            report_row("Reference", metric_min=4.0, metric_median=8.0, metric_max=20.0),
        ]
        result, stdout, stderr = self.run_checker(rows)
        self.assertEqual(result, 1, stderr)
        self.assertIn("stat=median", stdout)
        self.assertIn("FAIL 1.250000", stdout)

    def test_status_and_correctness_are_checked_before_latency(self):
        rows = [
            report_row(
                "Zynum",
                metric_median=1.0,
                status="correctness_failed",
                check_status="correctness_failed",
            ),
            report_row("Reference", metric_median=100.0),
        ]
        result, stdout, stderr = self.run_checker(rows)
        self.assertEqual(result, 2, stdout)
        self.assertIn("not eligible", stderr)
        self.assertNotIn("FAIL", stdout)

    def test_complete_case_key_keeps_groups_separate(self):
        rows = [
            report_row("Zynum", input_case="balanced"),
            report_row("Reference", input_case="balanced"),
            report_row("Zynum", input_case="a_dominant"),
        ]
        result, stdout, stderr = self.run_checker(rows)
        self.assertEqual(result, 1, stderr)
        self.assertIn("checked=1 passed=1 failed=0 missing=1", stdout)
        self.assertIn("case=a_dominant", stdout)

    def test_checker_rejects_invalid_metrics_thresholds_and_ratio_extremes(self):
        for value in ("nan", "inf", "-inf", "0", "-1"):
            with self.subTest(metric=value):
                result, _, _ = self.run_checker(
                    [
                        report_row("Zynum", metric_median=value),
                        report_row("Reference"),
                    ]
                )
                self.assertEqual(result, 2)
            with self.subTest(threshold=value):
                result, _, _ = self.run_checker(
                    [report_row("Zynum"), report_row("Reference")],
                    f"--ratio={value}",
                )
                self.assertEqual(result, 2)
        for candidate, comparator in ((1e308, 1e-308), (1e-308, 1e308)):
            result, _, stderr = self.run_checker(
                [
                    report_row("Zynum", metric_median=candidate),
                    report_row("Reference", metric_median=comparator),
                ]
            )
            self.assertEqual(result, 2)
            self.assertIn("comparison ratio", stderr)

    def test_checker_rejects_duplicates_and_is_shuffle_stable(self):
        original = report_row("Zynum", metric_median=2)
        for duplicate in (
            dict(original),
            report_row("Zynum", metric_median=9),
        ):
            with self.subTest(value=duplicate["metric_median"]):
                result, _, stderr = self.run_checker(
                    [original, duplicate, report_row("Reference", metric_median=1)]
                )
                self.assertEqual(result, 2)
                self.assertIn("duplicate library row", stderr)

        rows = [
            original,
            report_row("Reference", metric_median=1),
            report_row("Second", metric_median=1),
            report_row("Zynum", input_case="a_dominant", metric_median=3),
            report_row("Reference", input_case="a_dominant", metric_median=1),
        ]
        first = self.run_checker(rows, "--comparator", "Second")
        shuffled = self.run_checker(list(reversed(rows)), "--comparator", "Second")
        self.assertEqual(first, shuffled)
        self.assertEqual(first[0], 1)
        self.assertIn("best=Reference:1.000000", first[1])


class RotgLatencyProbeIntegrationTests(unittest.TestCase):
    @unittest.skipUnless(
        (REPO_ROOT / "zig-out/bin/rotg-latency-probe").is_file()
        and (REPO_ROOT / runner.default_zynum_blas()).is_file(),
        "ROTG latency probe and Zynum shared library have not been built",
    )
    def test_all_default_corpus_cases_pass_independent_reference_checks(self):
        probe = REPO_ROOT / "zig-out/bin/rotg-latency-probe"
        library = REPO_ROOT / runner.default_zynum_blas()
        args = runner.parse_args(["--csv", os.devnull])
        for case in runner.requested_cases(args):
            with self.subTest(routine=case.routine, input_case=case.input_case):
                result = subprocess.run(
                    [
                        str(probe),
                        "--blas",
                        str(library),
                        "--library",
                        "Zynum",
                        "--routine",
                        case.routine,
                        "--case",
                        case.input_case,
                        "--samples",
                        "1",
                        "--calls-per-sample",
                        "1000",
                    ],
                    cwd=str(REPO_ROOT),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    universal_newlines=True,
                    check=False,
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                rows = list(csv.DictReader(result.stdout.splitlines()))
                self.assertEqual(len(rows), 1)
                self.assertEqual(rows[0]["routine"], case.routine)
                self.assertEqual(rows[0]["case"], case.input_case)
                self.assertEqual(rows[0]["check_status"], "checked-ok", rows[0])
                if case.input_case in runner.EXPECTED_FLAGS:
                    self.assertEqual(
                        float(rows[0]["observed_flag"]),
                        runner.EXPECTED_FLAGS[case.input_case],
                    )


if __name__ == "__main__":
    unittest.main()
