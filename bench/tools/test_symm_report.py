#!/usr/bin/env python3
# Copyright (C) 2026 Zynum contributors
# SPDX-License-Identifier: LGPL-3.0-or-later

from __future__ import annotations

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


runner = load_tool("run_symm_report")
checker = load_tool("check_symm_report")


def report_row(
    library,
    *,
    routine="dsymm",
    kind="f64",
    shape="tiny",
    m=3,
    n=2,
    side="L",
    uplo="U",
    alpha=(0.75, 0.0),
    beta=(0.25, 0.0),
    metric_min=1,
    metric_median=2,
    metric_max=3,
    status="ok",
    check_status="checked-ok",
):
    order = m if side == "L" else n
    factor = 8 if kind.startswith("c") else 2
    return {
        "level": "level3",
        "routine": routine,
        "kind": kind,
        "library": library,
        "library_path": f"lib{library}.so",
        "shape": shape,
        "m": str(m),
        "n": str(n),
        "side": side,
        "uplo": uplo,
        "alpha_re": str(alpha[0]),
        "alpha_im": str(alpha[1]),
        "beta_re": str(beta[0]),
        "beta_im": str(beta[1]),
        "order": str(order),
        "lda": str(order),
        "ldb": str(m),
        "ldc": str(m),
        "reps": "2",
        "flop_count": str(factor * m * n * order),
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
        "check_samples": str(m * n),
        "check_raw_output": "",
        "process_repeats": "3",
        "successful_repeats": "3" if status == "ok" else "2",
        "metric_min": str(metric_min),
        "metric_median": str(metric_median),
        "metric_max": str(metric_max),
        "metric_samples": f"{metric_min},{metric_median},{metric_max}",
    }


class SymmRunnerTests(unittest.TestCase):
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
                "dsymm",
                "--shape",
                "tiny:3:2",
                "--side",
                "L",
                "--uplo",
                "U",
                "--alpha",
                "0.75",
                "--beta",
                "0.25",
                "--process-repeats",
                "1",
                "--csv",
                str(output),
            ]
        )

    def test_controller_calls_publisher_once_with_exact_ordered_bytes(self):
        identity = {"schema_version": 2, "source": {}}
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "symm.csv"
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
            output = Path(temp_dir) / "symm.csv"
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
            output = Path(temp_dir) / "symm.csv"
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
            output = Path(temp_dir).resolve() / "absent" / "symm.csv"
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
            output = Path(temp_dir).resolve() / "symm.csv"
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

    def test_stage_drift_and_bare_soname_capture_preserve_existing_outputs(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "symm.csv"
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
                "dsymm",
                "--shape",
                "tiny:3:2",
                "--side",
                "L",
                "--uplo",
                "U",
                "--alpha",
                "0.75",
                "--beta",
                "0.25",
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

    def test_default_cases_cover_broad_shapes_routines_sides_and_uplos(self):
        args = runner.parse_args(["--csv", os.devnull])
        cases = runner.requested_cases(args)
        self.assertEqual(args.process_repeats, 3)
        self.assertEqual(len(cases), len(runner.DEFAULT_SHAPES) * 6 * 2 * 2)
        self.assertEqual({case.routine.name for case in cases}, set(runner.ROUTINES))
        self.assertEqual(
            {case.shape.name for case in cases},
            {
                "square128",
                "tall512x128",
                "wide128x512",
            },
        )
        self.assertEqual({case.side for case in cases}, {"L", "R"})
        self.assertEqual({case.uplo for case in cases}, {"U", "L"})
        for case in cases:
            alpha = runner.parse_scalar(case.alpha)
            beta = runner.parse_scalar(case.beta)
            if case.routine.complex_scalars:
                self.assertNotEqual(alpha[1], 0)
                self.assertNotEqual(beta[1], 0)
            else:
                self.assertEqual(alpha[1], 0)
                self.assertEqual(beta[1], 0)

    def test_explicit_complex_scalars_are_filtered_from_real_routines(self):
        args = runner.parse_args(
            [
                "--csv",
                os.devnull,
                "--routine",
                "ssymm",
                "--routine",
                "zhemm",
                "--shape",
                "tiny:3:2",
                "--side",
                "R",
                "--uplo",
                "L",
                "--alpha",
                "0.5",
                "--alpha",
                "0.5,0.125",
                "--beta",
                "-0.25",
                "--beta",
                "-0.25,0.0625",
            ]
        )
        cases = runner.requested_cases(args)
        real_cases = [case for case in cases if case.routine.name == "ssymm"]
        complex_cases = [case for case in cases if case.routine.name == "zhemm"]
        self.assertEqual(len(real_cases), 1)
        self.assertEqual(len(complex_cases), 4)
        self.assertEqual(real_cases[0].alpha, "0.5")
        self.assertEqual(real_cases[0].beta, "-0.25")

    def test_case_command_forwards_complete_parameters(self):
        args = runner.parse_args(
            ["--csv", os.devnull, "--probe", "probe", "--reps", "7"]
        )
        case = runner.SymmCase(
            runner.ROUTINES["zhemm"],
            runner.Shape("wide", 64, 513),
            "R",
            "L",
            "0.75,-0.125",
            "0.25,0.0625",
        )
        command = runner.case_command(args, "MKL", "libmkl_rt.so", case)
        pairs = dict(zip(command[1::2], command[2::2]))
        self.assertEqual(pairs["--blas"], "libmkl_rt.so")
        self.assertEqual(pairs["--library"], "MKL")
        self.assertEqual(pairs["--routine"], "zhemm")
        self.assertEqual(pairs["--shape"], "wide")
        self.assertEqual(pairs["--m"], "64")
        self.assertEqual(pairs["--n"], "513")
        self.assertEqual(pairs["--side"], "R")
        self.assertEqual(pairs["--uplo"], "L")
        self.assertEqual(pairs["--alpha"], "0.75,-0.125")
        self.assertEqual(pairs["--beta"], "0.25,0.0625")
        self.assertEqual(pairs["--reps"], "7")

    @mock.patch.object(runner.subprocess, "run")
    def test_probe_failure_becomes_parameterized_error_row(self, run):
        run.return_value = subprocess.CompletedProcess(
            ["probe"], 1, stdout="", stderr="missing symbol"
        )
        args = runner.parse_args(["--csv", os.devnull, "--probe", "probe"])
        case = runner.SymmCase(
            runner.ROUTINES["dsymm"],
            runner.Shape("rect", 7, 3),
            "R",
            "U",
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
        self.assertEqual(row["routine"], "dsymm")
        self.assertEqual(row["m"], "7")
        self.assertEqual(row["n"], "3")
        self.assertEqual(row["order"], "3")
        self.assertIn("missing symbol", row["check_raw_output"])

    def test_explicit_missing_path_is_checked_without_loading_blas(self):
        self.assertFalse(runner.library_available("/not/a/real/libblas.so"))
        self.assertTrue(runner.library_available("libblas.so"))
        if sys.platform == "darwin":
            self.assertTrue(runner.library_available(runner.DEFAULT_ACCELERATE))

    def test_process_schedule_default_alias_and_conflicts(self):
        with mock.patch.object(runner.sys, "platform", "win32"):
            windows = runner.parse_args(["--csv", os.devnull])
        self.assertEqual(windows.probe, "zig-out/bin/symm-probe.exe")
        self.assertEqual(windows.zynum, "zig-out/bin/zynum_blas.dll")
        default = runner.parse_args(["--csv", os.devnull])
        canonical = runner.parse_args(
            ["--csv", os.devnull, "--process-schedule", "interleaved"]
        )
        alias = runner.parse_args(["--csv", os.devnull, "--schedule", "interleaved"])
        matching = runner.parse_args(
            [
                "--csv",
                os.devnull,
                "--process-schedule",
                "interleaved",
                "--schedule",
                "interleaved",
            ]
        )
        self.assertEqual(default.process_schedule, "library-major")
        self.assertEqual(default.schedule, "library-major")
        self.assertEqual(canonical.process_schedule, "interleaved")
        self.assertEqual(alias.process_schedule, "interleaved")
        self.assertEqual(matching.process_schedule, "interleaved")

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

    def test_interleaved_uses_actual_libraries_and_collects_identity_once(self):
        identity = {
            "schema_version": 1,
            "source": {
                "revision": None,
                "branch": None,
                "dirty": None,
                "status_short": None,
                "identity_status": "unreadable",
            },
        }
        execution = []
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir).resolve()
            probe = temp / "probe"
            zynum = temp / "libzynum.so"
            comparator = temp / "libmkl.so"
            missing = temp / "missing-openblas.so"
            output = temp / "report.csv"
            for path in (probe, zynum, comparator):
                path.touch()
            probe.chmod(0o755)
            args = runner.parse_args(
                [
                    "--probe",
                    str(probe),
                    "--zynum",
                    str(zynum),
                    "--accelerate",
                    "none",
                    "--openblas",
                    str(missing),
                    "--mkl",
                    str(comparator),
                    "--skip-missing",
                    "--routine",
                    "dsymm",
                    "--shape",
                    "tiny:3:2",
                    "--side",
                    "L",
                    "--uplo",
                    "U",
                    "--alpha",
                    "0.75",
                    "--beta",
                    "0.25",
                    "--process-repeats",
                    "2",
                    "--process-schedule",
                    "interleaved",
                    "--csv",
                    str(output),
                ]
            )

            def run_one(_args, library_name, _library_path, _case, **_kwargs):
                execution.append(library_name)
                return report_row(library_name)

            with (
                mock.patch.object(
                    runner.benchmark_metadata,
                    "collect_benchmark_identity_from_frozen",
                    return_value=identity,
                ) as collect_identity,
                mock.patch.object(runner, "run_one_process", side_effect=run_one),
                mock.patch.object(
                    runner, "command_output", return_value="0.16.0"
                ) as command_output,
                redirect_stderr(io.StringIO()),
            ):
                runner.run_controller(args)

            self.assertEqual(execution, ["Zynum", "MKL", "MKL", "Zynum"])
            collect_identity.assert_called_once()
            frozen_call = collect_identity.call_args.kwargs
            self.assertEqual(
                [artifact.name for artifact in frozen_call["libraries"]],
                ["Zynum", "MKL"],
            )
            self.assertEqual(frozen_call["binaries"][0].name, "symm_probe")
            command_output.assert_called_once_with(["zig", "version"])
            with output.with_suffix(".csv.meta.json").open() as file:
                metadata = json.load(file)
            self.assertEqual(metadata["benchmark_identity"]["schema_version"], 1)
            self.assertNotIn("status_short", metadata["benchmark_identity"]["source"])
            self.assertEqual(metadata["source"]["dirty"], None)
            self.assertEqual(metadata["schedule"], "interleaved")

    def test_interleaved_validates_after_skip_missing_and_allows_one_library(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir).resolve()
            probe = temp / "probe"
            zynum = temp / "libzynum.so"
            comparator = temp / "libmkl.so"
            missing = temp / "missing-openblas.so"
            for path in (probe, zynum, comparator):
                path.touch()
            probe.chmod(0o755)

            def arguments(repeats, mkl):
                return runner.parse_args(
                    [
                        "--probe",
                        str(probe),
                        "--zynum",
                        str(zynum),
                        "--accelerate",
                        "none",
                        "--openblas",
                        str(missing),
                        "--mkl",
                        mkl,
                        "--skip-missing",
                        "--routine",
                        "dsymm",
                        "--shape",
                        "tiny:3:2",
                        "--side",
                        "L",
                        "--uplo",
                        "U",
                        "--alpha",
                        "0.75",
                        "--beta",
                        "0.25",
                        "--process-repeats",
                        str(repeats),
                        "--process-schedule",
                        "interleaved",
                        "--csv",
                        str(temp / f"report-{repeats}-{mkl == 'none'}.csv"),
                    ]
                )

            invalid = arguments(3, str(comparator))
            with (
                mock.patch.object(
                    runner.benchmark_metadata,
                    "collect_benchmark_identity_from_frozen",
                ) as collect_identity,
                redirect_stderr(io.StringIO()),
            ):
                with self.assertRaisesRegex(ValueError, "2 selected libraries"):
                    runner.run_controller(invalid)
            collect_identity.assert_not_called()
            self.assertFalse(Path(invalid.csv).exists())

            one_library = arguments(5, "none")
            execution = []

            def run_one(_args, library_name, _library_path, _case, **_kwargs):
                execution.append(library_name)
                return report_row(library_name)

            with (
                mock.patch.object(
                    runner.benchmark_metadata,
                    "collect_benchmark_identity_from_frozen",
                    return_value={"source": {}},
                ),
                mock.patch.object(runner, "run_one_process", side_effect=run_one),
                mock.patch.object(runner, "command_output", return_value="0.16.0"),
                redirect_stderr(io.StringIO()),
            ):
                runner.run_controller(one_library)
            self.assertEqual(execution, ["Zynum"] * 5)

    def test_duplicate_library_labels_fail_before_identity_payload_and_publication(
        self,
    ):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            probe = temp / "probe"
            zynum = temp / "libzynum.so"
            alias = temp / "alias.so"
            missing = temp / "missing.so"
            output = temp / "symm.csv"
            metadata = output.with_suffix(output.suffix + ".meta.json")
            for path in (probe, zynum, alias):
                path.touch()
            probe.chmod(0o755)
            output.write_text("existing csv\n")
            metadata.write_text("existing metadata\n")
            argv = [
                "--probe",
                str(probe),
                "--zynum",
                str(zynum),
                "--accelerate",
                "none",
                "--openblas",
                str(missing),
                "--extra-blas",
                f"zynum-blas={alias}",
                "--skip-missing",
                "--csv",
                str(output),
            ]
            with (
                mock.patch.object(runner, "requested_cases", return_value=[object()]),
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

    def test_skip_missing_never_skips_zynum(self):
        args = runner.parse_args(
            [
                "--csv",
                os.devnull,
                "--zynum",
                "/not/a/real/libzynum.so",
                "--accelerate",
                "none",
                "--openblas",
                "none",
                "--skip-missing",
            ]
        )
        with self.assertRaisesRegex(ValueError, "Zynum"):
            runner.selected_libraries(args)


class SymmAggregationTests(unittest.TestCase):
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


class SymmCheckerTests(unittest.TestCase):
    def run_checker(self, rows, *extra_args):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "symm.csv"
            with path.open("w", newline="") as file:
                writer = csv.DictWriter(file, fieldnames=runner.CSV_FIELDNAMES)
                writer.writeheader()
                writer.writerows(rows)
            stdout = io.StringIO()
            stderr = io.StringIO()
            with redirect_stdout(stdout), redirect_stderr(stderr):
                result = checker.main(
                    [str(path), "--comparator", "Reference", *extra_args]
                )
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

    def test_checker_accepts_separate_negative_complex_scalar_filter(self):
        args = checker.parse_args(["report.csv", "--beta", "-0.25,0.0625"])
        self.assertEqual(args.beta, ["-0.25,0.0625"])

    def test_complete_parameters_keep_groups_separate(self):
        rows = [
            report_row("Zynum", side="L"),
            report_row("Reference", side="L"),
            report_row("Zynum", side="R", uplo="L", alpha=(0.5, 0.0), beta=(0.0, 0.0)),
        ]
        result, stdout, stderr = self.run_checker(rows)
        self.assertEqual(result, 1, stderr)
        self.assertIn("checked=1 passed=1 failed=0 missing=1", stdout)
        self.assertIn("side=R", stdout)
        self.assertIn("uplo=L", stdout)
        self.assertIn("alpha=0.5,0.0", stdout)

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
        original = report_row("Zynum", metric_median=0.5)
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
            report_row("Zynum", side="R", metric_median=0.25),
            report_row("Reference", side="R", metric_median=1),
        ]
        first = self.run_checker(rows, "--comparator", "Second")
        shuffled = self.run_checker(list(reversed(rows)), "--comparator", "Second")
        self.assertEqual(first, shuffled)
        self.assertEqual(first[0], 1)
        self.assertIn("best=Reference:1.000000", first[1])


class SymmProbeIntegrationTests(unittest.TestCase):
    @unittest.skipUnless(
        (REPO_ROOT / "zig-out/bin/symm-probe").is_file()
        and (REPO_ROOT / runner.default_zynum_blas()).is_file(),
        "SYMM probe and Zynum shared library have not been built",
    )
    def test_all_six_routines_sides_and_uplos_pass_full_reference_checks(self):
        probe = REPO_ROOT / "zig-out/bin/symm-probe"
        library = REPO_ROOT / runner.default_zynum_blas()
        for routine, spec in runner.ROUTINES.items():
            alpha = runner.COMPLEX_ALPHA if spec.complex_scalars else runner.REAL_ALPHA
            beta = runner.COMPLEX_BETA if spec.complex_scalars else runner.REAL_BETA
            for side in ("L", "R"):
                for uplo in ("U", "L"):
                    with self.subTest(routine=routine, side=side, uplo=uplo):
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
                                "--m",
                                "3",
                                "--n",
                                "2",
                                "--side",
                                side,
                                "--uplo",
                                uplo,
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
                        self.assertEqual(rows[0]["check_status"], "checked-ok")
                        self.assertEqual(rows[0]["check_samples"], "6")


if __name__ == "__main__":
    unittest.main()
