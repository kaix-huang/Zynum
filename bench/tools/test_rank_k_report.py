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


def load_tool(module_name):
    spec = importlib.util.spec_from_file_location(
        module_name, TOOLS_DIR / f"{module_name}.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


runner = load_tool("run_rank_k_report")
checker = load_tool("check_rank_k_report")


def report_row(
    library,
    *,
    routine="dsyrk",
    trans="N",
    alpha=(0.75, 0.0),
    beta=(0.25, 0.0),
    metric_min=1,
    metric_median=2,
    metric_max=3,
    status="ok",
    check_status="checked-ok",
    ldb=None,
):
    rank2k = routine.endswith("2k")
    kind = {"s": "f32", "d": "f64", "c": "c32", "z": "c64"}[routine[0]]
    factor = 8 if kind.startswith("c") else 2
    if rank2k:
        factor *= 2
    lda = "3" if trans == "N" else "2"
    return {
        "level": "level3",
        "routine": routine,
        "kind": kind,
        "library": library,
        "library_path": f"lib{library}.so",
        "shape": "tiny",
        "n": "3",
        "k": "2",
        "uplo": "U",
        "trans": trans,
        "alpha_re": str(alpha[0]),
        "alpha_im": str(alpha[1]),
        "beta_re": str(beta[0]),
        "beta_im": str(beta[1]),
        "lda": lda,
        "ldb": (lda if ldb is None else ldb) if rank2k else "",
        "ldc": "3",
        "reps": "2",
        "flop_count": str(factor * 6 * 2),
        "best_ns": "8",
        "median_ns": "10",
        "p95_ns": "12",
        "max_ns": "12",
        "gflops": str(metric_max),
        "median_gflops": str(metric_median),
        "metric": "gflops",
        "status": status,
        "check_status": check_status,
        "check_max_abs_error": "0",
        "check_max_rel_error": "0",
        "check_samples": "9",
        "check_raw_output": "",
        "process_repeats": "3",
        "successful_repeats": "3" if status == "ok" else "2",
        "metric_min": str(metric_min),
        "metric_median": str(metric_median),
        "metric_max": str(metric_max),
        "metric_samples": f"{metric_min},{metric_median},{metric_max}",
    }


class RankKRunnerTests(unittest.TestCase):
    def minimal_controller_args(self, temp_dir, output):
        probe = Path(temp_dir) / "probe"
        probe.touch()
        probe.chmod(0o755)
        zynum = Path(temp_dir) / "libzynum.so"
        zynum.write_bytes(b"zynum")
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
                "dsyrk",
                "--shape",
                "tiny:3:2",
                "--uplo",
                "U",
                "--trans",
                "N",
                "--process-repeats",
                "1",
                "--csv",
                str(output),
            ]
        )

    def test_controller_calls_publisher_once_with_exact_ordered_bytes(self):
        identity = {"schema_version": 2, "source": {}}
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir).resolve() / "rank-k.csv"
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
            output = Path(temp_dir).resolve() / "rank-k.csv"
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
            output = Path(temp_dir) / "rank-k.csv"
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
            output = Path(temp_dir).resolve() / "absent" / "rank-k.csv"
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

    def test_frozen_artifact_survives_a_b_a_source_replacement_across_repeats(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir).resolve() / "rank-k.csv"
            args = self.minimal_controller_args(temp_dir, output)
            args.process_repeats = 3
            public_library = Path(args.zynum)
            public_library.write_bytes(b"A")
            expected_sha256 = hashlib.sha256(b"A").hexdigest()
            observed_bytes = []
            private_paths = []

            def collect_identity(_args, *, libraries, binaries):
                self.assertEqual(libraries[0].sha256, expected_sha256)
                self.assertEqual(binaries[0].path, args.probe)
                return {
                    "schema_version": 2,
                    "source": {},
                    "frozen_library_sha256": libraries[0].sha256,
                }

            def run_one(
                _args,
                library_name,
                private_library,
                _case,
                *,
                probe_path,
                public_library_path,
                **_kwargs,
            ):
                self.assertNotEqual(private_library, public_library_path)
                self.assertNotEqual(probe_path, args.probe)
                private_paths.append(private_library)
                observed_bytes.append(Path(private_library).read_bytes())
                if len(observed_bytes) == 1:
                    public_library.write_bytes(b"B")
                elif len(observed_bytes) == 2:
                    public_library.write_bytes(b"A")
                row = report_row(library_name)
                row["library_path"] = public_library_path
                return row

            with (
                mock.patch.object(
                    runner.benchmark_metadata,
                    "collect_benchmark_identity_from_frozen",
                    side_effect=collect_identity,
                ),
                mock.patch.object(runner, "run_one_process", side_effect=run_one),
                mock.patch.object(runner, "command_output", return_value="0.16.0"),
                redirect_stderr(io.StringIO()),
            ):
                runner.run_controller(args)

            self.assertEqual(observed_bytes, [b"A", b"A", b"A"])
            self.assertEqual(len(set(private_paths)), 1)
            self.assertFalse(Path(private_paths[0]).exists())
            metadata = json.loads(output.with_suffix(".csv.meta.json").read_bytes())
            self.assertEqual(
                metadata["benchmark_identity"]["frozen_library_sha256"],
                expected_sha256,
            )
            self.assertEqual(metadata["libraries"][0]["sha256"], expected_sha256)

    def test_stage_drift_and_cleanup_failure_preserve_existing_outputs(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "rank-k.csv"
            metadata = output.with_suffix(".csv.meta.json")
            output.write_bytes(b"old csv\n")
            metadata.write_bytes(b"old metadata\n")
            args = self.minimal_controller_args(temp_dir, output)
            Path(args.zynum).write_bytes(b"A")
            private_paths = []

            def run_one(_args, library_name, private_library, _case, **_kwargs):
                private_paths.append(private_library)
                row = report_row(library_name)
                row["library_path"] = args.zynum
                return row

            real_serialize = runner.serialize_metadata

            def serialize_then_drift(*serialize_args):
                contents = real_serialize(*serialize_args)
                staged = Path(private_paths[0])
                staged.chmod(0o600)
                staged.write_bytes(b"B")
                staged.chmod(0o400)
                return contents

            def capture_in_temp(requests, **limits):
                return runner.benchmark_artifacts.ArtifactSnapshotSet(
                    requests,
                    private_parent=temp_dir,
                    **limits,
                )

            with (
                mock.patch.object(
                    runner.benchmark_metadata,
                    "collect_benchmark_identity_from_frozen",
                    return_value={"source": {}},
                ),
                mock.patch.object(runner, "run_one_process", side_effect=run_one),
                mock.patch.object(
                    runner, "serialize_metadata", side_effect=serialize_then_drift
                ),
                mock.patch.object(
                    runner.benchmark_artifacts.ArtifactSnapshotSet,
                    "capture",
                    side_effect=capture_in_temp,
                ),
                mock.patch.object(runner, "command_output", return_value="0.16.0"),
                mock.patch.object(runner, "publish_outputs") as publisher,
                redirect_stderr(io.StringIO()),
            ):
                with self.assertRaises(
                    runner.benchmark_artifacts.ArtifactCleanupError
                ) as raised:
                    runner.run_controller(args)

            self.assertIsInstance(
                raised.exception.__context__,
                runner.benchmark_artifacts.ArtifactVerificationError,
            )
            publisher.assert_not_called()
            self.assertEqual(output.read_bytes(), b"old csv\n")
            self.assertEqual(metadata.read_bytes(), b"old metadata\n")
            self.assertFalse(Path(private_paths[0]).exists())
            retained = [
                Path(path)
                for path in raised.exception.recovery_paths
                if Path(path).name == Path(private_paths[0]).name
                and Path(path).is_file()
            ]
            self.assertEqual(len(retained), 1)
            drifted = retained[0]
            self.assertTrue(
                drifted.parent.name.startswith(".zynum-benchmark-artifact-quarantine-")
            )
            self.assertEqual(drifted.read_bytes(), b"B")
            self.assertIn(
                "private_artifact_replaced",
                {issue.code for issue in raised.exception.issues},
            )
            drifted.unlink()
            drifted.parent.rmdir()

            real_close = runner.benchmark_artifacts.ArtifactSnapshotSet.close

            def close_then_report_failure(snapshot):
                real_close(snapshot)
                raise runner.benchmark_artifacts.ArtifactCleanupError(
                    (runner.benchmark_artifacts.CleanupIssue("synthetic", None),)
                )

            with (
                mock.patch.object(
                    runner.benchmark_metadata,
                    "collect_benchmark_identity_from_frozen",
                    return_value={"source": {}},
                ),
                mock.patch.object(
                    runner, "run_one_process", return_value=report_row("Zynum")
                ),
                mock.patch.object(runner, "command_output", return_value="0.16.0"),
                mock.patch.object(
                    runner.benchmark_artifacts.ArtifactSnapshotSet,
                    "close",
                    autospec=True,
                    side_effect=close_then_report_failure,
                ),
                mock.patch.object(runner, "publish_outputs") as publisher,
                redirect_stderr(io.StringIO()),
            ):
                with self.assertRaises(runner.benchmark_artifacts.ArtifactCleanupError):
                    runner.run_controller(args)
            publisher.assert_not_called()
            self.assertEqual(output.read_bytes(), b"old csv\n")
            self.assertEqual(metadata.read_bytes(), b"old metadata\n")

    def test_capture_is_fail_closed_and_platform_images_are_explicit(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "rank-k.csv"
            metadata = output.with_suffix(".csv.meta.json")
            output.write_bytes(b"old csv\n")
            metadata.write_bytes(b"old metadata\n")
            args = self.minimal_controller_args(temp_dir, output)
            args.zynum = "libzynum.so"
            with (
                mock.patch.object(runner, "parse_args", return_value=args),
                mock.patch.object(runner, "publish_outputs") as publisher,
                redirect_stderr(io.StringIO()),
            ):
                self.assertEqual(runner.main([]), 2)
            publisher.assert_not_called()
            self.assertEqual(output.read_bytes(), b"old csv\n")
            self.assertEqual(metadata.read_bytes(), b"old metadata\n")

        framework = runner.DEFAULT_ACCELERATE
        with (
            mock.patch.object(runner, "platform_image_path", return_value=True),
            mock.patch.object(runner.Path, "exists", return_value=False),
        ):
            request = runner.library_artifact_request("Accelerate", framework)
        self.assertEqual(request.source_kind, "platform_image")
        self.assertEqual(request.path, framework)

        traversal = (
            "/System/Library/Frameworks/Accelerate.framework/"
            "../../../../private/tmp/unhashed.dylib"
        )
        with mock.patch.object(runner.sys, "platform", "darwin"):
            self.assertFalse(runner.platform_image_path(traversal))
            self.assertEqual(
                runner.library_artifact_request("Accelerate", traversal).source_kind,
                "file",
            )
            self.assertEqual(
                runner.library_artifact_request("Other", framework).source_kind,
                "file",
            )

    def test_child_private_path_is_mapped_and_public_path_mutant_is_rejected(self):
        args = runner.parse_args(
            [
                "--csv",
                os.devnull,
                "--probe",
                "public-probe",
                "--reps",
                "2",
                "--routine",
                "dsyrk",
                "--shape",
                "tiny:3:2",
                "--uplo",
                "U",
                "--trans",
                "N",
            ]
        )
        case = runner.requested_cases(args)[0]
        private_probe = "/private/root/probe"
        private_library = "/private/root/library"
        public_library = "/public/libzynum.so"

        def completed_row(library_path):
            row = report_row("Zynum")
            row["library_path"] = library_path
            output = io.StringIO(newline="")
            writer = csv.DictWriter(output, fieldnames=runner.PROBE_FIELDNAMES)
            writer.writeheader()
            writer.writerow({field: row[field] for field in runner.PROBE_FIELDNAMES})
            return subprocess.CompletedProcess(
                [private_probe], 0, stdout=output.getvalue(), stderr=""
            )

        def redact(value):
            return runner.benchmark_artifacts.ArtifactSnapshotSet._redact_value(
                value,
                [
                    (private_probe, args.probe),
                    (private_library, public_library),
                ],
            )

        with mock.patch.object(
            runner.subprocess, "run", return_value=completed_row(private_library)
        ):
            row = runner.run_one_process(
                args,
                "Zynum",
                private_library,
                case,
                probe_path=private_probe,
                public_library_path=public_library,
                redact_private_paths=redact,
            )
        self.assertEqual(row["status"], "ok")
        self.assertEqual(row["library_path"], public_library)

        with mock.patch.object(
            runner.subprocess, "run", return_value=completed_row(public_library)
        ):
            mutant = runner.run_one_process(
                args,
                "Zynum",
                private_library,
                case,
                probe_path=private_probe,
                public_library_path=public_library,
                redact_private_paths=redact,
            )
        self.assertEqual(mutant["status"], "error")
        self.assertNotIn(private_probe, json.dumps(mutant))
        self.assertNotIn(private_library, json.dumps(mutant))

        leaked_failure = subprocess.CompletedProcess(
            [private_probe],
            1,
            stdout=f"stdout={private_probe}",
            stderr=f"stderr={private_library}",
        )
        with mock.patch.object(runner.subprocess, "run", return_value=leaked_failure):
            failure = runner.run_one_process(
                args,
                "Zynum",
                private_library,
                case,
                probe_path=private_probe,
                public_library_path=public_library,
                redact_private_paths=redact,
            )
        failure_json = json.dumps(failure)
        self.assertNotIn(private_probe, failure_json)
        self.assertNotIn(private_library, failure_json)
        self.assertIn(args.probe, failure_json)
        self.assertIn(public_library, failure_json)

    def test_process_schedule_is_canonical_and_legacy_alias_conflicts_exit_two(self):
        with mock.patch.object(runner.sys, "platform", "win32"):
            windows = runner.parse_args(["--csv", os.devnull])
        self.assertEqual(windows.probe, "zig-out/bin/rank-k-probe.exe")
        self.assertEqual(windows.zynum, "zig-out/bin/zynum_blas.dll")
        self.assertEqual(
            runner.parse_args(["--csv", os.devnull]).process_schedule,
            "library-major",
        )
        self.assertEqual(
            runner.parse_args(
                ["--csv", os.devnull, "--schedule", "interleaved"]
            ).process_schedule,
            "interleaved",
        )
        self.assertEqual(
            runner.parse_args(
                [
                    "--csv",
                    os.devnull,
                    "--process-schedule",
                    "interleaved",
                    "--schedule",
                    "interleaved",
                ]
            ).process_schedule,
            "interleaved",
        )
        invalid_argvs = (
            ["--csv", os.devnull, "--process-schedule", "invalid"],
            ["--csv", os.devnull, "--schedule", "invalid"],
            [
                "--csv",
                os.devnull,
                "--process-schedule",
                "library-major",
                "--schedule",
                "interleaved",
            ],
        )
        for argv in invalid_argvs:
            with self.subTest(argv=argv), redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as raised:
                    runner.parse_args(argv)
                self.assertEqual(raised.exception.code, 2)

    def test_interleaved_single_library_accepts_arbitrary_positive_repeats(self):
        case = runner.RankKCase(
            runner.ROUTINES["dsyrk"],
            runner.Shape("tiny", 3, 2),
            "U",
            "N",
            "0.75",
            "0.25",
        )
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
            probe = Path(temp_dir) / "probe"
            probe.touch()
            probe.chmod(0o755)
            zynum = Path(temp_dir) / "libzynum.so"
            accelerate = Path(temp_dir) / "libaccelerate.so"
            zynum.write_bytes(b"zynum")
            accelerate.write_bytes(b"accelerate")
            output = Path(temp_dir).resolve() / "rank-k.csv"
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
                    "dsyrk",
                    "--shape",
                    "tiny:3:2",
                    "--uplo",
                    "U",
                    "--trans",
                    "N",
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
                    side_effect=lambda path: path != "missing",
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
            frozen_call = collect_identity.call_args.kwargs
            self.assertEqual(
                [artifact.name for artifact in frozen_call["libraries"]],
                ["Zynum", "Accelerate"],
            )
            self.assertEqual(frozen_call["binaries"][0].name, "rank_k_probe")
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
            probe = Path(temp_dir) / "probe"
            probe.touch()
            probe.chmod(0o755)
            output = Path(temp_dir) / "rank-k.csv"
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
                "dsyrk",
                "--shape",
                "tiny:3:2",
                "--uplo",
                "U",
                "--trans",
                "N",
                "--process-repeats",
                "3",
                "--schedule",
                "interleaved",
                "--csv",
                str(output),
            ]
            with (
                mock.patch.object(
                    runner,
                    "library_available",
                    side_effect=lambda path: path != "missing",
                ),
                mock.patch.object(
                    runner.benchmark_metadata,
                    "collect_benchmark_identity_from_frozen",
                ) as collect_identity,
                mock.patch.object(runner, "collect_repeats") as collect_samples,
                mock.patch.object(runner, "run_one_process") as run_payload,
                redirect_stderr(io.StringIO()),
            ):
                self.assertEqual(runner.main(argv), 2)

            collect_identity.assert_not_called()
            collect_samples.assert_not_called()
            run_payload.assert_not_called()
            self.assertEqual(output.read_text(), "existing csv\n")
            self.assertEqual(metadata.read_text(), "existing metadata\n")

    def test_duplicate_library_labels_fail_before_identity_payload_and_publication(
        self,
    ):
        with tempfile.TemporaryDirectory() as temp_dir:
            probe = Path(temp_dir) / "probe"
            probe.touch()
            probe.chmod(0o755)
            output = Path(temp_dir) / "rank-k.csv"
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
                "libzynum=alias",
                "--skip-missing",
                "--csv",
                str(output),
            ]
            with (
                mock.patch.object(runner, "requested_cases", return_value=[object()]),
                mock.patch.object(
                    runner,
                    "library_available",
                    side_effect=lambda path: path != "missing",
                ),
                mock.patch.object(
                    runner.benchmark_metadata,
                    "collect_benchmark_identity_from_frozen",
                ) as collect_identity,
                mock.patch.object(runner, "run_one_process") as run_payload,
                mock.patch.object(runner, "publish_outputs") as publish_metadata,
                redirect_stderr(io.StringIO()) as stderr,
            ):
                self.assertEqual(runner.main(argv), 2)

            self.assertIn("duplicate semantic library label", stderr.getvalue())
            collect_identity.assert_not_called()
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
            probe = Path(temp_dir) / "probe"
            probe.touch()
            probe.chmod(0o755)
            library = Path(temp_dir) / "libzynum.so"
            library.write_bytes(b"library")
            library.chmod(0o644)
            args = runner.parse_args(
                [
                    "--probe",
                    str(probe),
                    "--zynum",
                    str(library),
                    "--routine",
                    "dsyrk",
                    "--shape",
                    "tiny:3:2",
                    "--uplo",
                    "U",
                    "--trans",
                    "N",
                    "--process-repeats",
                    "5",
                    "--process-schedule",
                    "interleaved",
                    "--csv",
                    str(Path(temp_dir).resolve() / "rank-k.csv"),
                ]
            )
            with (
                mock.patch.object(
                    runner,
                    "selected_libraries",
                    return_value=[("Zynum", args.zynum)],
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
            frozen_call = collect_identity.call_args.kwargs
            self.assertEqual(frozen_call["libraries"][0].name, "Zynum")
            self.assertEqual(frozen_call["binaries"][0].name, "rank_k_probe")
            self.assertEqual(run_payload.call_count, 5)

    def test_default_cases_cover_all_legal_transposes_and_uplos(self):
        args = runner.parse_args(["--csv", os.devnull])
        cases = runner.requested_cases(args)
        self.assertEqual(args.process_repeats, 3)
        self.assertEqual(len(cases), len(runner.DEFAULT_SHAPES) * 12 * 2 * 2)
        self.assertEqual(len(cases), 192)
        by_routine = {}
        for case in cases:
            by_routine.setdefault(case.routine.name, set()).add(case.trans)
        self.assertEqual(by_routine["ssyrk"], {"N", "T"})
        self.assertEqual(by_routine["dsyrk"], {"N", "T"})
        self.assertEqual(by_routine["csyrk"], {"N", "T"})
        self.assertEqual(by_routine["zsyrk"], {"N", "T"})
        self.assertEqual(by_routine["cherk"], {"N", "C"})
        self.assertEqual(by_routine["zherk"], {"N", "C"})
        self.assertEqual(by_routine["ssyr2k"], {"N", "T"})
        self.assertEqual(by_routine["dsyr2k"], {"N", "T"})
        self.assertEqual(by_routine["csyr2k"], {"N", "T"})
        self.assertEqual(by_routine["zsyr2k"], {"N", "T"})
        self.assertEqual(by_routine["cher2k"], {"N", "C"})
        self.assertEqual(by_routine["zher2k"], {"N", "C"})
        self.assertEqual({case.uplo for case in cases}, {"U", "L"})

    def test_explicit_complex_scalars_require_complex_syrk_selection(self):
        args = runner.parse_args(
            [
                "--csv",
                os.devnull,
                "--routine",
                "csyrk",
                "--shape",
                "tiny:3:2",
                "--uplo",
                "L",
                "--trans",
                "T",
                "--alpha",
                "0.5,0.125",
                "--beta",
                "-0.25,0.0625",
            ]
        )
        cases = runner.requested_cases(args)
        self.assertEqual(len(cases), 1)
        self.assertEqual(cases[0].alpha, "0.5,0.125")
        self.assertEqual(cases[0].beta, "-0.25,0.0625")

        invalid = runner.parse_args(
            ["--csv", os.devnull, "--routine", "cherk", "--alpha", "1,0.5"]
        )
        with self.assertRaisesRegex(ValueError, "complex alpha"):
            runner.requested_cases(invalid)

    def test_her2k_accepts_complex_alpha_and_requires_real_beta(self):
        valid = runner.parse_args(
            [
                "--csv",
                os.devnull,
                "--routine",
                "zher2k",
                "--shape",
                "tiny:3:2",
                "--alpha",
                "0.5,0.125",
                "--beta",
                "-0.25",
            ]
        )
        cases = runner.requested_cases(valid)
        self.assertEqual(len(cases), 4)
        self.assertEqual({case.trans for case in cases}, {"N", "C"})

        invalid = runner.parse_args(
            [
                "--csv",
                os.devnull,
                "--routine",
                "cher2k",
                "--beta",
                "0.25,0.0625",
            ]
        )
        with self.assertRaisesRegex(ValueError, "complex beta"):
            runner.requested_cases(invalid)

    def test_case_command_forwards_complete_parameters(self):
        args = runner.parse_args(
            ["--csv", os.devnull, "--probe", "probe", "--reps", "7"]
        )
        case = runner.RankKCase(
            runner.ROUTINES["zher2k"],
            runner.Shape("highk", 64, 513),
            "L",
            "C",
            "0.75,0.125",
            "0.25",
        )
        command = runner.case_command(args, "MKL", "libmkl_rt.so", case)
        pairs = dict(zip(command[1::2], command[2::2]))
        self.assertEqual(pairs["--blas"], "libmkl_rt.so")
        self.assertEqual(pairs["--library"], "MKL")
        self.assertEqual(pairs["--routine"], "zher2k")
        self.assertEqual(pairs["--shape"], "highk")
        self.assertEqual(pairs["--n"], "64")
        self.assertEqual(pairs["--k"], "513")
        self.assertEqual(pairs["--uplo"], "L")
        self.assertEqual(pairs["--trans"], "C")
        self.assertEqual(pairs["--alpha"], "0.75,0.125")
        self.assertEqual(pairs["--beta"], "0.25")
        self.assertEqual(pairs["--reps"], "7")

    def test_rank2k_flop_count_is_twice_matching_rank_k(self):
        shape = runner.Shape("tiny", 3, 2)
        syrk = runner.RankKCase(runner.ROUTINES["dsyrk"], shape, "U", "N", "1", "0")
        syr2k = runner.RankKCase(runner.ROUTINES["dsyr2k"], shape, "U", "N", "1", "0")
        self.assertEqual(runner.flop_count(syr2k), 2 * runner.flop_count(syrk))

    @mock.patch.object(runner.subprocess, "run")
    def test_probe_failure_becomes_parameterized_error_row(self, run):
        run.return_value = subprocess.CompletedProcess(
            ["probe"], 1, stdout="", stderr="missing symbol"
        )
        args = runner.parse_args(["--csv", os.devnull, "--probe", "probe"])
        case = runner.RankKCase(
            runner.ROUTINES["dsyr2k"],
            runner.Shape("rect", 7, 3),
            "U",
            "T",
            "0.75",
            "0.25",
        )
        row = runner.run_one_process(
            args,
            "TestBLAS",
            "private-libblas.so",
            case,
            probe_path="private-probe",
            public_library_path="libblas.so",
            redact_private_paths=lambda value: value,
        )
        self.assertEqual(row["status"], "error")
        self.assertEqual(row["routine"], "dsyr2k")
        self.assertEqual(row["n"], "7")
        self.assertEqual(row["k"], "3")
        self.assertEqual(row["lda"], "3")
        self.assertEqual(row["ldb"], "3")
        self.assertEqual(row["flop_count"], "336")
        self.assertIn("missing symbol", row["check_raw_output"])


class RankKAggregationTests(unittest.TestCase):
    def test_aggregate_rejects_every_invalid_performance_sample(self):
        fields = (
            "best_ns",
            "median_ns",
            "p95_ns",
            "max_ns",
            "gflops",
            "median_gflops",
        )
        for field in fields:
            for value in ("0", "nan", "inf", "-inf", "1e999999"):
                row = report_row("Zynum")
                row[field] = value
                with self.subTest(field=field, value=value):
                    with self.assertRaises(ValueError):
                        runner.aggregate_repeats([row])

    def test_even_process_median_is_safe_at_finite_float_extremes(self):
        for value in (1e308, 5e-324):
            rows = [
                report_row(
                    "Zynum", metric_min=value, metric_median=value, metric_max=value
                )
                for _ in range(2)
            ]
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

    def test_process_median_uses_probe_median_gflops(self):
        rows = []
        for median_value, best_value in ((2, 20), (4, 5), (3, 7)):
            row = report_row("Zynum", metric_median=median_value, metric_max=best_value)
            row["median_gflops"] = str(median_value)
            row["gflops"] = str(best_value)
            rows.append(row)
        aggregate = runner.aggregate_repeats(rows)
        self.assertEqual(aggregate["process_repeats"], 3)
        self.assertEqual(aggregate["successful_repeats"], 3)
        self.assertEqual(aggregate["metric_min"], "2")
        self.assertEqual(aggregate["metric_median"], "3")
        self.assertEqual(aggregate["metric_max"], "4")
        self.assertEqual(aggregate["metric_samples"], "2,4,3")
        self.assertEqual(aggregate["gflops"], "20")

    def test_any_bad_repeat_contaminates_aggregate(self):
        good = report_row("Zynum")
        bad = report_row(
            "Zynum", status="correctness_failed", check_status="correctness_failed"
        )
        bad["check_max_abs_error"] = "4.5"
        bad["check_raw_output"] = "reference tolerance exceeded"
        aggregate = runner.aggregate_repeats([good, bad])
        self.assertEqual(aggregate["successful_repeats"], 1)
        self.assertEqual(aggregate["status"], "correctness_failed")
        self.assertEqual(aggregate["check_status"], "correctness_failed")
        self.assertEqual(aggregate["check_max_abs_error"], "4.5")
        self.assertIn("repeat=2", aggregate["check_raw_output"])


class RankKCheckerTests(unittest.TestCase):
    def run_checker(self, rows, *extra_args, comparators=("Reference",)):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "rank_k.csv"
            with path.open("w", newline="") as file:
                writer = csv.DictWriter(file, fieldnames=runner.CSV_FIELDNAMES)
                writer.writeheader()
                writer.writerows(rows)
            stdout = io.StringIO()
            stderr = io.StringIO()
            with redirect_stdout(stdout), redirect_stderr(stderr):
                comparator_args = [
                    option
                    for comparator in comparators
                    for option in ("--comparator", comparator)
                ]
                result = checker.main([str(path), *comparator_args, *extra_args])
        return result, stdout.getvalue(), stderr.getvalue()

    def test_default_checker_uses_fresh_process_median(self):
        rows = [
            report_row("Zynum", metric_min=1, metric_median=2, metric_max=10),
            report_row("Reference", metric_min=1, metric_median=4, metric_max=8),
        ]
        median_result, median_stdout, median_stderr = self.run_checker(rows)
        best_result, best_stdout, best_stderr = self.run_checker(rows, "--stat", "best")
        self.assertEqual(median_result, 1, median_stderr)
        self.assertIn("stat=median", median_stdout)
        self.assertEqual(best_result, 0, best_stderr)
        self.assertIn("stat=best", best_stdout)

    def test_complete_parameters_keep_groups_separate(self):
        rows = [
            report_row("Zynum", trans="N"),
            report_row("Reference", trans="N"),
            report_row("Zynum", trans="T", alpha=(0.5, 0.0)),
        ]
        result, stdout, stderr = self.run_checker(rows)
        self.assertEqual(result, 1, stderr)
        self.assertIn("checked=1 passed=1 failed=0 missing=1", stdout)
        self.assertIn("trans=T", stdout)
        self.assertIn("alpha=0.5,0.0", stdout)

    def test_rank2k_ldb_keeps_groups_separate(self):
        rows = [
            report_row("Zynum", routine="dsyr2k", ldb="3"),
            report_row("Reference", routine="dsyr2k", ldb="3"),
            report_row("Zynum", routine="dsyr2k", ldb="4"),
        ]
        result, stdout, stderr = self.run_checker(rows)
        self.assertEqual(result, 1, stderr)
        self.assertIn("checked=1 passed=1 failed=0 missing=1", stdout)
        self.assertIn("ldb=4", stdout)

    def test_bad_zynum_correctness_is_not_performance_evidence(self):
        rows = [
            report_row(
                "Zynum",
                status="correctness_failed",
                check_status="correctness_failed",
            ),
            report_row("Reference"),
        ]
        result, stdout, stderr = self.run_checker(rows)
        self.assertEqual(result, 2, stdout)
        self.assertIn("not eligible", stderr)

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
        original = report_row(
            "Zynum", metric_min=0.5, metric_median=0.5, metric_max=0.5
        )
        for duplicate in (
            dict(original),
            report_row("Zynum", metric_median=9),
        ):
            with self.subTest(value=duplicate["metric_median"]):
                result, _, stderr = self.run_checker(
                    [original, duplicate, report_row("Reference")]
                )
                self.assertEqual(result, 2)
                self.assertIn("duplicate library row", stderr)

        rows = [
            original,
            report_row("Reference", metric_median=1),
            report_row("Second", metric_median=1),
            report_row("Zynum", trans="T", metric_median=0.25),
            report_row("Reference", trans="T", metric_median=1),
        ]
        first = self.run_checker(rows, comparators=("Reference", "Second"))
        shuffled = self.run_checker(
            list(reversed(rows)), comparators=("Reference", "Second")
        )
        reversed_cli = self.run_checker(rows, comparators=("Second", "Reference"))
        paired = self.run_checker(
            rows,
            "--stat",
            "paired-median",
            comparators=("Reference", "Second"),
        )
        paired_reversed = self.run_checker(
            rows,
            "--stat",
            "paired-median",
            comparators=("Second", "Reference"),
        )
        self.assertEqual(first, shuffled)
        self.assertEqual(first, reversed_cli)
        self.assertEqual(paired, paired_reversed)
        self.assertEqual(first[0], 1)
        self.assertIn("best=Reference:1.000000", first[1])
        self.assertIn("best=Reference:1.000000", paired[1])


class RankKProbeIntegrationTests(unittest.TestCase):
    @unittest.skipUnless(
        (REPO_ROOT / "zig-out/bin/rank-k-probe").is_file()
        and (REPO_ROOT / runner.default_zynum_blas()).is_file(),
        "rank-k probe and Zynum shared library have not been built",
    )
    def test_all_twelve_routines_cover_both_triangles_and_legal_transposes(self):
        probe = REPO_ROOT / "zig-out/bin/rank-k-probe"
        library = REPO_ROOT / runner.default_zynum_blas()
        for routine, spec in runner.ROUTINES.items():
            alpha = "0.75" if spec.alpha_must_be_real else "0.75,0.125"
            beta = "0.25" if spec.beta_must_be_real else "0.25,0.0625"
            for uplo in ("U", "L"):
                for trans in spec.transposes:
                    with self.subTest(routine=routine, uplo=uplo, trans=trans):
                        result = subprocess.run(
                            [
                                str(probe),
                                "--blas",
                                str(library),
                                "--library",
                                "Zynum",
                                "--routine",
                                routine,
                                "--shape",
                                "tiny",
                                "--n",
                                "3",
                                "--k",
                                "2",
                                "--uplo",
                                uplo,
                                "--trans",
                                trans,
                                "--alpha",
                                alpha,
                                "--beta",
                                beta,
                                "--reps",
                                "1",
                            ],
                            cwd=REPO_ROOT,
                            capture_output=True,
                            text=True,
                            check=False,
                        )
                        self.assertEqual(result.returncode, 0, result.stderr)
                        rows = list(csv.DictReader(result.stdout.splitlines()))
                        self.assertEqual(len(rows), 1)
                        self.assertEqual(rows[0]["routine"], routine)
                        self.assertEqual(rows[0]["status"], "ok", rows[0])
                        self.assertEqual(rows[0]["check_status"], "checked-ok", rows[0])
                        self.assertEqual(
                            rows[0]["ldb"], rows[0]["lda"] if spec.rank2k else ""
                        )


if __name__ == "__main__":
    unittest.main()
