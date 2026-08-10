#!/usr/bin/env python3
# Copyright (C) 2026 Zynum contributors
# SPDX-License-Identifier: LGPL-3.0-or-later

import csv
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
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


checker = load_tool("check_gemm_sweep")
runner = load_tool("run_gemm_sweep_isolated")
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


def gemm_row(
    library,
    trans,
    gflops,
    *,
    kind="sgemm",
    best_ns=1,
    median_ns=1,
    max_ns=1,
):
    return {
        "kind": kind,
        "transa": trans[0],
        "transb": trans[1],
        "shape_index": "0",
        "label": "tiny",
        "m": "2",
        "n": "2",
        "k": "2",
        "library": library,
        "gflops": str(gflops),
        "best_ns": str(best_ns),
        "median_ns": str(median_ns),
        "p95_ns": str(max_ns),
        "max_ns": str(max_ns),
        "reps": "3",
        "process_repeats": "1",
        "check": "checked-ok",
    }


def benchmark_identity(source=None):
    return {
        "schema_version": "test",
        "source": source
        or {
            "revision": "test-revision",
            "branch": "test-branch",
            "dirty": False,
            "status_short": "",
        },
    }


def write_gemm_csv(path, rows, fieldnames=None):
    with path.open("w", newline="") as csv_file:
        writer = csv.DictWriter(
            csv_file, fieldnames=fieldnames or runner.CSV_FIELDNAMES
        )
        writer.writeheader()
        writer.writerows(rows)


class GemmIsolatedRunnerTests(unittest.TestCase):
    def test_platform_image_exception_is_exact_and_cannot_traverse(self):
        traversal = (
            "/System/Library/Frameworks/Accelerate.framework/"
            "../../../../private/tmp/unhashed.dylib"
        )
        with (
            mock.patch.object(runner.sys, "platform", "darwin"),
            mock.patch.object(runner.Path, "is_file", return_value=False),
        ):
            self.assertFalse(runner.library_path_exists(traversal))
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

    def setUp(self):
        self.capture_patcher = mock.patch.object(
            runner.ArtifactSnapshotSet,
            "capture",
            side_effect=lambda requests: FakeArtifactSnapshot(requests),
        )
        self.capture_mock = self.capture_patcher.start()
        self.addCleanup(self.capture_patcher.stop)

    def run_publication_controller(self, output, *, identity=None, publish=None):
        args = runner.parse_args(
            [
                "--csv",
                str(output),
                "--kind",
                "sgemm",
                "--shape",
                "tiny:2:2:2",
            ]
        )

        def write_repeat(_args, _name, _path, repeat_output, _kind, _shapes, **_kwargs):
            write_gemm_csv(repeat_output, [gemm_row("Zynum", "NN", 4)])

        with (
            mock.patch.object(
                runner, "existing_libs", return_value=[("Zynum", "zynum.so")]
            ),
            mock.patch.object(
                runner.benchmark_metadata,
                "collect_benchmark_identity_from_frozen",
                return_value=identity or benchmark_identity(),
            ),
            mock.patch.object(runner, "run_one_process", side_effect=write_repeat),
            mock.patch.object(runner, "zig_version", return_value=None),
            mock.patch.object(
                runner, "publish_outputs", side_effect=publish
            ) as publisher,
        ):
            runner.run_controller(args)
        return publisher

    @staticmethod
    def gemm_probe_source(score):
        fields = repr(runner.CSV_FIELDNAMES)
        return f"""#!/usr/bin/env python3
import argparse
import csv
import sys

parser = argparse.ArgumentParser()
parser.add_argument("--zynum-blas", required=True)
parser.add_argument("--reps")
parser.add_argument("--csv", required=True)
parser.add_argument("--kind", action="append")
parser.add_argument("--trans", action="append")
parser.add_argument("--shape", action="append")
parser.add_argument("--check", action="store_true")
args = parser.parse_args()
print("argv=" + " ".join(sys.argv))
print("stderr-argv=" + " ".join(sys.argv), file=sys.stderr)
library = open(args.zynum_blas, encoding="utf-8").read().strip()
value = {score} if {score} != 7 else (7 if library == "A" else 99)
row = {{
    "kind": (args.kind or ["sgemm"])[0],
    "transa": "N", "transb": "N", "shape_index": "0",
    "label": "tiny", "m": "2", "n": "2", "k": "2",
    "library": "worker", "gflops": str(value), "best_ns": "1",
    "median_ns": "1", "p95_ns": "1", "max_ns": "1",
    "reps": "1", "process_repeats": "1", "check": "checked-ok",
}}
with open(args.csv, "w", newline="") as output:
    writer = csv.DictWriter(output, fieldnames={fields})
    writer.writeheader()
    writer.writerow(row)
"""

    def test_real_controller_freezes_probe_and_library_and_redacts_private_paths(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir).resolve()
            probe = root / "gemm-probe"
            library = root / "libzynum.so"
            output = root / "report.csv"
            probe_a = self.gemm_probe_source(7).encode()
            library_a = b"A\n"
            probe.write_bytes(probe_a)
            probe.chmod(0o755)
            library.write_bytes(library_a)
            args = runner.parse_args(
                [
                    "--csv",
                    str(output),
                    "--gemm-sweep",
                    str(probe),
                    "--zynum-blas",
                    str(library),
                    "--accelerate",
                    "none",
                    "--openblas",
                    "none",
                    "--kind",
                    "sgemm",
                    "--shape",
                    "tiny:2:2:2",
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
                replacement_probe = root / "replacement-probe"
                replacement_probe.write_text(self.gemm_probe_source(123))
                replacement_probe.chmod(0o755)
                replacement_probe.replace(probe)
                replacement_library = root / "replacement-library"
                replacement_library.write_text("B\n")
                replacement_library.replace(library)
                return artifacts

            def run_then_restore_original(command, *run_args, **run_kwargs):
                nonlocal child_count
                result = real_subprocess_run(command, *run_args, **run_kwargs)
                if isinstance(command, list) and "--zynum-blas" in command:
                    child_count += 1
                    if child_count == 1:
                        restored_probe = root / "restored-probe"
                        restored_probe.write_bytes(probe_a)
                        restored_probe.chmod(0o755)
                        restored_probe.replace(probe)
                        restored_library = root / "restored-library"
                        restored_library.write_bytes(library_a)
                        restored_library.replace(library)
                return result

            stdout = io.StringIO()
            stderr = io.StringIO()
            with (
                mock.patch.object(
                    runner.ArtifactSnapshotSet,
                    "capture",
                    side_effect=capture_then_replace,
                ),
                mock.patch.object(
                    runner.subprocess,
                    "run",
                    side_effect=run_then_restore_original,
                ),
                redirect_stdout(stdout),
                redirect_stderr(stderr),
            ):
                runner.run_controller(args)

            with output.open(newline="") as csv_file:
                rows = list(csv.DictReader(csv_file))
            metadata_bytes = output.with_suffix(".csv.meta.json").read_bytes()
            metadata = json.loads(metadata_bytes)
            self.assertEqual([row["gflops"] for row in rows], ["7"])
            self.assertEqual(child_count, 2)
            expected_probe_hash = hashlib.sha256(probe_a).hexdigest()
            expected_library_hash = hashlib.sha256(library_a).hexdigest()
            self.assertEqual(
                metadata["binaries"]["gemm_sweep"]["sha256"],
                expected_probe_hash,
            )
            self.assertEqual(
                metadata["binaries"]["libraries"][0]["sha256"],
                expected_library_hash,
            )
            identity_artifacts = metadata["benchmark_identity"]["payload"]["artifacts"]
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
            self.assertNotIn("zynum-blas-gemm-isolated-", diagnostics)

    def test_stage_drift_before_publication_returns_two_and_preserves_outputs(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir).resolve()
            probe = root / "gemm-probe"
            library = root / "libzynum.so"
            output = root / "report.csv"
            metadata = output.with_suffix(".csv.meta.json")
            probe.write_text("#!/bin/sh\nexit 0\n")
            probe.chmod(0o755)
            library.write_text("A\n")
            output.write_bytes(b"previous csv\n")
            metadata.write_bytes(b"previous metadata\n")
            args = runner.parse_args(
                [
                    "--csv",
                    str(output),
                    "--gemm-sweep",
                    str(probe),
                    "--zynum-blas",
                    str(library),
                    "--accelerate",
                    "none",
                    "--openblas",
                    "none",
                    "--kind",
                    "sgemm",
                    "--shape",
                    "tiny:2:2:2",
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

            def write_repeat(
                _args, _name, _path, repeat_output, _kind, _shapes, **_kwargs
            ):
                write_gemm_csv(repeat_output, [gemm_row("Zynum", "NN", 4)])

            real_serialize = runner.serialize_metadata

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
                    return_value=benchmark_identity(),
                ),
                mock.patch.object(runner, "run_one_process", side_effect=write_repeat),
                mock.patch.object(
                    runner,
                    "serialize_metadata",
                    side_effect=serialize_then_corrupt,
                ),
                mock.patch.object(runner, "publish_outputs", publish),
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

    def test_repeat_merge_rejects_every_invalid_performance_sample(self):
        fields = ("gflops", "best_ns", "median_ns", "p95_ns", "max_ns")
        for field in fields:
            for value in ("0", "nan", "inf", "-inf", "1e999999"):
                row = gemm_row("Zynum", "NN", 1)
                row[field] = value
                with self.subTest(field=field, value=value):
                    with self.assertRaises(ValueError):
                        runner.merge_repeat_rows([row])

    def test_repeat_merge_preserves_extreme_finite_derived_values(self):
        huge_timing = 10**308
        rows = [
            gemm_row(
                "Zynum",
                "NN",
                1e308,
                best_ns=huge_timing,
                median_ns=huge_timing,
                max_ns=huge_timing,
            )
            for _ in range(2)
        ]
        aggregate = runner.merge_repeat_rows(rows)
        self.assertEqual(float(aggregate["gflops"]), 1e308)
        for field in ("best_ns", "median_ns", "p95_ns", "max_ns"):
            self.assertEqual(float(aggregate[field]), float(huge_timing))

    def test_even_repeat_median_and_nearest_rank_p95(self):
        rows = [
            gemm_row("Zynum", "NN", 1, median_ns=value, max_ns=value)
            for value in (10, 30)
        ]
        aggregate = runner.merge_repeat_rows(rows)
        self.assertEqual(float(aggregate["median_ns"]), 20)
        self.assertEqual(float(aggregate["p95_ns"]), 30)

        fractional = runner.merge_repeat_rows(
            [gemm_row("Zynum", "NN", 1, median_ns=value) for value in (10, 11)]
        )
        self.assertEqual(float(fractional["median_ns"]), 10.5)

    @mock.patch.object(
        runner,
        "run_controller",
        side_effect=ValueError("median_ns must be finite and positive"),
    )
    def test_invalid_derived_evidence_returns_two_before_publication(self, _run):
        args = runner.parse_args(["--csv", os.devnull])
        with (
            mock.patch.object(runner, "parse_args", return_value=args),
            redirect_stderr(io.StringIO()),
        ):
            self.assertEqual(runner.main(), 2)

    def test_intermediate_paths_are_numeric_direct_children(self):
        private_dir = Path("private-root")
        repeat = runner.intermediate_output_path(private_dir, 2, 3, 4)
        merged = runner.intermediate_output_path(private_dir, 2, 3, merged=True)
        self.assertEqual(repeat, private_dir / "library_2_case_3_repeat_4.csv")
        self.assertEqual(merged, private_dir / "library_2_case_3_merged.csv")
        self.assertEqual(repeat.parent, private_dir)
        self.assertEqual(merged.parent, private_dir)

        for indexes in (("2", 3, 4), (2, -1, 4), (2, 3, True)):
            with self.subTest(indexes=indexes), self.assertRaises(ValueError):
                runner.intermediate_output_path(private_dir, *indexes)

    def test_malicious_labels_cannot_escape_private_intermediate_root(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir).resolve()
            private_root = root / "private"
            private_root.mkdir()
            output = root / "public.csv"
            metadata = output.with_suffix(".csv.meta.json")
            output.write_bytes(b"previous csv\n")
            metadata.write_bytes(b"previous metadata\n")

            absolute_label = str(root / "absolute-victim")
            traversal_label = "../traversal-victim"
            sentinels = [
                Path(f"{absolute_label}.csv"),
                root / "traversal-victim.csv",
                root / "shape-victim.csv",
            ]
            for sentinel in sentinels:
                sentinel.write_bytes(b"sentinel\n")

            args = runner.parse_args(
                [
                    "--csv",
                    str(output),
                    "--zynum-blas",
                    "zynum.so",
                    "--accelerate",
                    "none",
                    "--openblas",
                    "none",
                    "--extra-blas",
                    f"{absolute_label}=absolute.so",
                    "--extra-blas",
                    f"{traversal_label}=traversal.so",
                    "--kind",
                    "sgemm",
                    "--shape",
                    f"{root / 'absolute-shape'}:2:2:2",
                    "--shape",
                    "../traversal-shape:3:3:3",
                    "--isolate-shape",
                    "--process-repeats",
                    "2",
                ]
            )
            temporary_directory = mock.MagicMock()
            temporary_directory.__enter__.return_value = str(private_root)
            temporary_directory.__exit__.return_value = False
            probe_paths = []
            merged_paths = []

            def write_invalid_repeat(
                _args, _name, _path, repeat_output, _kind, _shapes, **_kwargs
            ):
                probe_paths.append(repeat_output)
                row = gemm_row("untrusted", "NN", 4)
                row["label"] = "unexpected"
                write_gemm_csv(repeat_output, [row])

            real_best_rows_csv = runner.best_rows_csv

            def record_merged_path(inputs, merged_output, expected_keys):
                merged_paths.append(merged_output)
                return real_best_rows_csv(inputs, merged_output, expected_keys)

            with (
                mock.patch.object(
                    runner.tempfile,
                    "TemporaryDirectory",
                    return_value=temporary_directory,
                ),
                mock.patch.object(
                    runner.benchmark_metadata,
                    "collect_benchmark_identity_from_frozen",
                    return_value=benchmark_identity(),
                ),
                mock.patch.object(
                    runner, "run_one_process", side_effect=write_invalid_repeat
                ),
                mock.patch.object(
                    runner, "best_rows_csv", side_effect=record_merged_path
                ),
                mock.patch.object(runner, "publish_outputs") as publish_outputs,
                self.assertRaisesRegex(ValueError, "GEMM key mismatch"),
            ):
                runner.run_controller(args)

            self.assertEqual(len(probe_paths), 12)
            self.assertEqual(len(merged_paths), 1)
            for path in probe_paths + merged_paths:
                self.assertEqual(path.parent, private_root)
                self.assertRegex(
                    path.name,
                    r"^library_\d+_case_\d+_(?:repeat_\d+|merged)\.csv$",
                )
                for forbidden in (
                    absolute_label,
                    traversal_label,
                    "absolute-shape",
                    "traversal-shape",
                    "sgemm",
                ):
                    self.assertNotIn(forbidden, path.name)
            self.assertTrue(all(path.is_file() for path in private_root.iterdir()))
            publish_outputs.assert_not_called()
            self.assertEqual(output.read_bytes(), b"previous csv\n")
            self.assertEqual(metadata.read_bytes(), b"previous metadata\n")
            for sentinel in sentinels:
                self.assertEqual(sentinel.read_bytes(), b"sentinel\n")

    def test_transpose_cli_defaults_to_nn_and_normalizes_explicit_pairs(self):
        with mock.patch.object(runner.sys, "platform", "win32"):
            windows = runner.parse_args(["--csv", os.devnull])
        self.assertEqual(windows.gemm_sweep, "zig-out/bin/gemm-sweep.exe")
        self.assertEqual(windows.zynum_blas, "zig-out/bin/zynum_blas.dll")
        default_args = runner.parse_args(["--csv", os.devnull])
        self.assertIsNone(default_args.trans)
        self.assertEqual(default_args.process_schedule, "library-major")

        args = runner.parse_args(
            ["--csv", os.devnull, "--trans", "nt", "--trans", "CC"]
        )
        self.assertEqual(args.trans, ["NT", "CC"])

    def test_process_schedule_rejects_unknown_choice(self):
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as raised:
            runner.parse_args(["--csv", os.devnull, "--process-schedule", "randomized"])
        self.assertEqual(raised.exception.code, 2)

    def test_duplicate_library_labels_fail_before_identity_payload_and_publication(
        self,
    ):
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir).resolve() / "gemm.csv"
            metadata = output.with_suffix(output.suffix + ".meta.json")
            output.write_text("existing csv\n")
            metadata.write_text("existing metadata\n")
            args = runner.parse_args(["--csv", str(output)])
            with (
                mock.patch.object(runner, "parse_args", return_value=args),
                mock.patch.object(
                    runner,
                    "existing_libs",
                    return_value=[
                        ("Zynum", "zynum"),
                        (" lib-zynum-blas ", "alias"),
                    ],
                ),
                mock.patch.object(
                    runner.benchmark_metadata,
                    "collect_benchmark_identity_from_frozen",
                ) as collect_identity,
                mock.patch.object(runner, "run_one_process") as run_payload,
                mock.patch.object(runner, "publish_outputs") as publish_metadata,
                redirect_stderr(io.StringIO()) as stderr,
            ):
                self.assertEqual(runner.main(), 2)

            self.assertIn("duplicate semantic library label", stderr.getvalue())
            collect_identity.assert_not_called()
            run_payload.assert_not_called()
            publish_metadata.assert_not_called()
            self.assertEqual(output.read_text(), "existing csv\n")
            self.assertEqual(metadata.read_text(), "existing metadata\n")

    @mock.patch.object(runner.subprocess, "run")
    def test_worker_command_forwards_each_transpose_pair(self, run):
        run.return_value.returncode = 0
        run.return_value.stdout = ""
        run.return_value.stderr = ""
        args = runner.parse_args(
            [
                "--csv",
                os.devnull,
                "--gemm-sweep",
                "gemm-sweep",
                "--trans",
                "NT",
                "--trans",
                "CC",
            ]
        )
        runner.run_one_process(
            args,
            "Zynum",
            "libzynum.so",
            Path("out.csv"),
            kind="cgemm",
            shapes=["tiny:2:2:2"],
        )
        command = run.call_args.args[0]
        self.assertEqual(
            [
                command[index + 1]
                for index, value in enumerate(command)
                if value == "--trans"
            ],
            ["NT", "CC"],
        )

    def test_metadata_records_effective_transpose_selection(self):
        args = runner.parse_args(
            ["--csv", os.devnull, "--trans", "NT", "--trans", "CC"]
        )
        with mock.patch.object(runner, "zig_version", return_value="test"):
            metadata_bytes = runner.serialize_metadata(
                args, [], ["tiny:2:2:2"], benchmark_identity()
            )
        metadata = json.loads(metadata_bytes)
        self.assertEqual(metadata["transposes"], ["NT", "CC"])

    def test_metadata_projects_unknown_git_state_without_claiming_clean(self):
        args = runner.parse_args(["--csv", os.devnull])
        identity = benchmark_identity(
            {
                "revision": None,
                "branch": None,
                "dirty": None,
                "status_short": None,
                "identity_status": "unreadable",
            }
        )
        with mock.patch.object(runner, "zig_version", return_value="test"):
            metadata_bytes = runner.serialize_metadata(args, [], [], identity)
        metadata = json.loads(metadata_bytes)
        self.assertEqual(
            metadata["source"],
            {
                "revision": None,
                "branch": None,
                "dirty": None,
            },
        )
        self.assertIsNone(metadata["benchmark_identity"]["source"]["dirty"])
        self.assertNotIn("status_short", metadata["benchmark_identity"]["source"])

    def test_controller_publishes_one_ordered_immutable_pair(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            parent = Path(temp_dir).resolve() / "absent"
            output = parent / "gemm.csv"

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
                    [(row["library"], row["label"]) for row in rows],
                    [("Zynum", "tiny")],
                )
                self.assertEqual(
                    json.loads(outputs[1].contents)["shapes"], ["tiny:2:2:2"]
                )

            publisher = self.run_publication_controller(
                output, publish=assert_prepublication
            )
            publisher.assert_called_once()
            self.assertFalse(parent.exists())

    def test_metadata_serialization_failure_precedes_publication(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir).resolve() / "gemm.csv"
            metadata = output.with_suffix(".csv.meta.json")
            output.write_bytes(b"previous csv\n")
            metadata.write_bytes(b"previous metadata\n")
            invalid_identity = benchmark_identity()
            invalid_identity["invalid"] = float("nan")
            publish = mock.Mock()

            with self.assertRaises(ValueError):
                self.run_publication_controller(
                    output, identity=invalid_identity, publish=publish
                )

            self.assertEqual(output.read_bytes(), b"previous csv\n")
            self.assertEqual(metadata.read_bytes(), b"previous metadata\n")
            publish.assert_not_called()

    def test_publisher_failure_receives_one_batch_and_cannot_split_pair(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir).resolve() / "gemm.csv"
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

    def test_controller_delegates_symlink_rejection_to_production_publisher(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir).resolve()
            victim = root / "victim.csv"
            victim.write_bytes(b"victim contents\n")
            output = root / "gemm.csv"
            output.symlink_to(victim)
            metadata = output.with_suffix(".csv.meta.json")
            metadata.write_bytes(b"previous metadata\n")

            with self.assertRaises(OSError):
                self.run_publication_controller(output, publish=runner.publish_outputs)

            self.assertTrue(output.is_symlink())
            self.assertEqual(victim.read_bytes(), b"victim contents\n")
            self.assertEqual(metadata.read_bytes(), b"previous metadata\n")

    def test_controller_collects_identity_once_and_honors_schedule_order(self):
        args = runner.parse_args(
            [
                "--csv",
                os.devnull,
                "--process-repeats",
                "2",
                "--process-schedule",
                "interleaved",
            ]
        )
        libraries = [("A", "a.so"), ("B", "b.so")]
        identity = benchmark_identity()
        observed = []

        def record_process(_args, name, _path, _out, _kind, _shapes, **_kwargs):
            observed.append(name)

        with (
            mock.patch.object(runner, "existing_libs", return_value=libraries),
            mock.patch.object(
                runner.benchmark_metadata,
                "collect_benchmark_identity_from_frozen",
                return_value=identity,
            ) as collect_identity,
            mock.patch.object(runner, "run_one_process", side_effect=record_process),
            mock.patch.object(runner, "best_rows_csv"),
            mock.patch.object(runner, "serialize_csv", return_value=b"csv\r\n"),
            mock.patch.object(
                runner, "serialize_metadata", return_value=b"{}\n"
            ) as serialize_metadata,
            mock.patch.object(runner, "publish_outputs") as publish_outputs,
        ):
            runner.run_controller(args)

        self.assertEqual(observed, ["A", "B", "B", "A"])
        collect_identity.assert_called_once()
        self.assertEqual(collect_identity.call_args.args, (args,))
        self.assertEqual(
            [
                (item.name, item.path)
                for item in collect_identity.call_args.kwargs["libraries"]
            ],
            libraries,
        )
        self.assertEqual(
            [
                (item.name, item.path)
                for item in collect_identity.call_args.kwargs["binaries"]
            ],
            [("gemm_sweep", args.gemm_sweep)],
        )
        self.assertIs(serialize_metadata.call_args.args[-1], identity)
        publish_outputs.assert_called_once()

    def test_library_major_schedule_preserves_legacy_order(self):
        args = runner.parse_args(["--csv", os.devnull, "--process-repeats", "2"])
        observed = []
        with (
            mock.patch.object(
                runner, "existing_libs", return_value=[("A", "a"), ("B", "b")]
            ),
            mock.patch.object(
                runner.benchmark_metadata,
                "collect_benchmark_identity_from_frozen",
                return_value=benchmark_identity(),
            ),
            mock.patch.object(
                runner,
                "run_one_process",
                side_effect=lambda _a, name, *_rest, **_kwargs: observed.append(name),
            ),
            mock.patch.object(runner, "best_rows_csv"),
            mock.patch.object(runner, "serialize_csv", return_value=b"csv\r\n"),
            mock.patch.object(runner, "serialize_metadata", return_value=b"{}\n"),
            mock.patch.object(runner, "publish_outputs"),
        ):
            runner.run_controller(args)
        self.assertEqual(observed, ["A", "A", "B", "B"])

    def test_interleaved_validation_uses_actual_post_filter_library_count(self):
        args = runner.parse_args(
            [
                "--csv",
                os.devnull,
                "--zynum-blas",
                "missing-zynum",
                "--accelerate",
                "missing-accelerate",
                "--openblas",
                "openblas",
                "--skip-missing",
                "--process-repeats",
                "3",
                "--process-schedule",
                "interleaved",
            ]
        )
        with (
            mock.patch.object(
                runner,
                "library_path_exists",
                side_effect=lambda path: path == "openblas",
            ),
            mock.patch.object(
                runner.benchmark_metadata,
                "collect_benchmark_identity_from_frozen",
                return_value=benchmark_identity(),
            ) as collect_identity,
            mock.patch.object(runner, "run_one_process"),
            mock.patch.object(runner, "best_rows_csv"),
            mock.patch.object(runner, "serialize_csv", return_value=b"csv\r\n"),
            mock.patch.object(runner, "serialize_metadata", return_value=b"{}\n"),
            mock.patch.object(runner, "publish_outputs"),
        ):
            with self.assertRaisesRegex(ValueError, "2 selected libraries"):
                runner.run_controller(args)
        collect_identity.assert_not_called()

        with (
            mock.patch.object(runner, "library_path_exists", return_value=False),
            mock.patch.object(
                runner.benchmark_metadata,
                "collect_benchmark_identity_from_frozen",
                return_value=benchmark_identity(),
            ),
            mock.patch.object(runner, "run_one_process") as run_process,
            mock.patch.object(runner, "best_rows_csv"),
            mock.patch.object(runner, "serialize_csv", return_value=b"csv\r\n"),
            mock.patch.object(runner, "serialize_metadata", return_value=b"{}\n"),
            mock.patch.object(runner, "publish_outputs"),
        ):
            runner.run_controller(args)
        self.assertEqual(run_process.call_count, 3)
        self.assertTrue(
            all(call.args[1] == "Zynum" for call in run_process.call_args_list)
        )

    def test_interleaved_repeat_conflict_exits_two_before_execution(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir).resolve() / "gemm.csv"
            metadata = output.with_suffix(output.suffix + ".meta.json")
            output.write_text("previous csv\n")
            metadata.write_text("previous metadata\n")
            args = runner.parse_args(
                [
                    "--csv",
                    str(output),
                    "--process-repeats",
                    "3",
                    "--process-schedule",
                    "interleaved",
                ]
            )
            with (
                mock.patch.object(runner, "parse_args", return_value=args),
                mock.patch.object(
                    runner,
                    "existing_libs",
                    return_value=[("A", "a"), ("B", "b")],
                ),
                mock.patch.object(
                    runner.benchmark_metadata,
                    "collect_benchmark_identity_from_frozen",
                ) as collect_identity,
                mock.patch.object(runner, "run_one_process") as run_process,
                mock.patch.object(runner, "serialize_csv") as serialize_csv,
                mock.patch.object(runner, "serialize_metadata") as serialize_metadata,
                mock.patch.object(runner, "publish_outputs") as publish_outputs,
                redirect_stderr(io.StringIO()) as stderr,
            ):
                self.assertEqual(runner.main(), 2)
            collect_identity.assert_not_called()
            run_process.assert_not_called()
            serialize_csv.assert_not_called()
            serialize_metadata.assert_not_called()
            publish_outputs.assert_not_called()
            self.assertEqual(output.read_text(), "previous csv\n")
            self.assertEqual(metadata.read_text(), "previous metadata\n")
            self.assertIn("multiple of the 2 selected libraries", stderr.getvalue())

    def test_invalid_evidence_does_not_overwrite_existing_csv_or_json(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            source = temp / "bad.csv"
            output = temp / "output.csv"
            with source.open("w", newline="") as csv_file:
                writer = csv.DictWriter(csv_file, fieldnames=runner.CSV_FIELDNAMES)
                writer.writeheader()
                writer.writerow(gemm_row("Zynum", "NN", "nan"))
            output.write_text("previous csv\n")
            with self.assertRaises(ValueError):
                runner.best_rows_csv([source], output)
            self.assertEqual(output.read_text(), "previous csv\n")

            args = runner.parse_args(["--csv", str(output)])
            metadata_path = output.with_suffix(output.suffix + ".meta.json")
            metadata_path.write_text("previous json\n")
            invalid_identity = benchmark_identity()
            invalid_identity["invalid"] = float("nan")
            with (
                mock.patch.object(runner, "zig_version", return_value="test"),
                self.assertRaises(ValueError),
            ):
                runner.serialize_metadata(args, [], [], invalid_identity)
            self.assertEqual(metadata_path.read_text(), "previous json\n")

    def test_repeat_merge_rejects_missing_case_without_overwriting_output(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            first = temp / "first.csv"
            second = temp / "second.csv"
            output = temp / "merged.csv"
            case_a = gemm_row("Zynum", "NN", 4)
            case_a["label"] = "A"
            case_b = gemm_row("Zynum", "NN", 3)
            case_b["label"] = "B"
            write_gemm_csv(first, [case_a, case_b])
            write_gemm_csv(second, [dict(case_a)])
            output.write_text("previous csv\n")

            with self.assertRaisesRegex(ValueError, "key mismatch.*missing=.*B"):
                runner.best_rows_csv([first, second], output)
            self.assertEqual(output.read_text(), "previous csv\n")

    def test_controller_rejects_incomplete_or_non_unique_repeat_rows(self):
        def case_row(label, size):
            row = gemm_row("Zynum", "NN", 4)
            row["label"] = label
            row["m"] = str(size)
            row["n"] = str(size)
            row["k"] = str(size)
            return row

        case_a = case_row("A", 2)
        case_b = case_row("B", 3)
        case_extra = case_row("extra", 4)
        scenarios = (
            (1, [case_a], "key mismatch.*missing=.*B"),
            (2, [case_a], "key mismatch.*missing=.*B"),
            (1, [case_a, case_b, case_extra], "key mismatch.*extra=.*extra"),
            (1, [case_a, case_b, case_b], "duplicate GEMM key"),
        )
        for process_repeats, repeat_rows, message in scenarios:
            with self.subTest(
                process_repeats=process_repeats,
                message=message,
            ):
                with tempfile.TemporaryDirectory() as temp_dir:
                    output = Path(temp_dir).resolve() / "gemm.csv"
                    metadata = output.with_suffix(output.suffix + ".meta.json")
                    output.write_text("previous csv\n")
                    metadata.write_text("previous metadata\n")
                    args = runner.parse_args(
                        [
                            "--csv",
                            str(output),
                            "--zynum-blas",
                            "zynum.so",
                            "--accelerate",
                            "none",
                            "--openblas",
                            "none",
                            "--kind",
                            "sgemm",
                            "--shape",
                            "A:2:2:2",
                            "--shape",
                            "B:3:3:3",
                            "--process-repeats",
                            str(process_repeats),
                        ]
                    )

                    def write_partial_repeat(
                        _args,
                        _name,
                        _path,
                        repeat_output,
                        _kind,
                        _shapes,
                        **_kwargs,
                    ):
                        write_gemm_csv(repeat_output, repeat_rows)

                    with (
                        mock.patch.object(
                            runner.benchmark_metadata,
                            "collect_benchmark_identity_from_frozen",
                            return_value=benchmark_identity(),
                        ),
                        mock.patch.object(
                            runner,
                            "run_one_process",
                            side_effect=write_partial_repeat,
                        ),
                        mock.patch.object(runner, "serialize_csv") as serialize_csv,
                        mock.patch.object(
                            runner, "serialize_metadata"
                        ) as serialize_metadata,
                        mock.patch.object(runner, "publish_outputs") as publish_outputs,
                    ):
                        with self.assertRaisesRegex(ValueError, message):
                            runner.run_controller(args)
                    serialize_csv.assert_not_called()
                    serialize_metadata.assert_not_called()
                    publish_outputs.assert_not_called()
                    self.assertEqual(output.read_text(), "previous csv\n")
                    self.assertEqual(metadata.read_text(), "previous metadata\n")

    def test_repeat_merge_rejects_duplicate_and_extra_keys(self):
        def case(label):
            row = gemm_row("Zynum", "NN", 1)
            row["label"] = label
            return row

        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            first = temp / "first.csv"
            second = temp / "second.csv"
            output = temp / "merged.csv"
            write_gemm_csv(first, [case("A")])
            expected_keys = [runner.gemm_semantic_key(case("A"))]

            for rows, message in (
                ([case("A"), case("A")], "duplicate GEMM key"),
                ([case("A"), case("extra")], "key mismatch.*extra=.*extra"),
            ):
                with self.subTest(message=message):
                    write_gemm_csv(second, rows)
                    output.write_text("previous csv\n")
                    with self.assertRaisesRegex(ValueError, message):
                        runner.best_rows_csv([first, second], output, expected_keys)
                    self.assertEqual(output.read_text(), "previous csv\n")

    def test_expected_process_keys_cover_combined_isolation_dimensions(self):
        args = runner.parse_args(
            [
                "--csv",
                os.devnull,
                "--trans",
                "NT",
                "--trans",
                "CC",
            ]
        )
        shapes = ["A:2:3:4", "B:5:6:7"]
        all_keys = runner.expected_process_keys(args, shapes)
        self.assertEqual(len(all_keys), 12)
        self.assertEqual(
            {key[0] for key in all_keys},
            {"sgemm", "dgemm", "cgemm", "zgemm"},
        )
        self.assertEqual({key[1:3] for key in all_keys}, {("N", "T"), ("C", "C")})
        self.assertEqual({key[3] for key in all_keys}, {"A", "B"})

        kind_isolated = runner.expected_process_keys(args, shapes, kind="dgemm")
        self.assertEqual(len(kind_isolated), 2)
        self.assertEqual({key[0] for key in kind_isolated}, {"dgemm"})
        self.assertEqual({key[1:3] for key in kind_isolated}, {("N", "T")})

        shape_isolated = runner.expected_process_keys(args, shapes, shapes=[shapes[1]])
        self.assertEqual(len(shape_isolated), 6)
        self.assertEqual({key[3] for key in shape_isolated}, {"B"})

        combined = runner.expected_process_keys(
            args,
            shapes,
            kind="zgemm",
            shapes=[shapes[0]],
        )
        self.assertEqual(
            combined,
            [
                ("zgemm", "N", "T", "A", "2", "3", "4"),
                ("zgemm", "C", "C", "A", "2", "3", "4"),
            ],
        )

        explicit = runner.parse_args(
            [
                "--csv",
                os.devnull,
                "--kind",
                "sgemm",
                "--kind",
                "zgemm",
            ]
        )
        self.assertEqual(
            {key[0] for key in runner.expected_process_keys(explicit, shapes)},
            {"sgemm", "zgemm"},
        )

    def test_repeat_merge_preserves_first_repeat_order_and_transposes(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            first = temp / "first.csv"
            second = temp / "second.csv"
            output = temp / "merged.csv"
            nn = gemm_row("Zynum", "NN", 4)
            nt = gemm_row("Zynum", "NT", 3)
            write_gemm_csv(first, [nt, nn])
            write_gemm_csv(second, [nn, nt])

            runner.best_rows_csv([first, second], output)
            with output.open(newline="") as csv_file:
                rows = list(csv.DictReader(csv_file))

        self.assertEqual(
            [(row["transa"], row["transb"]) for row in rows],
            [("N", "T"), ("N", "N")],
        )
        self.assertEqual([row["process_repeats"] for row in rows], ["2", "2"])

    def test_repeat_merge_upgrades_matching_legacy_rows_to_nn(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            first = temp / "first.csv"
            second = temp / "second.csv"
            output = temp / "merged.csv"
            legacy_fields = [
                field
                for field in runner.CSV_FIELDNAMES
                if field not in {"transa", "transb"}
            ]
            legacy = gemm_row("Zynum", "NN", 4)
            legacy.pop("transa")
            legacy.pop("transb")
            write_gemm_csv(first, [legacy], legacy_fields)
            write_gemm_csv(second, [legacy], legacy_fields)

            runner.best_rows_csv([first, second], output)
            with output.open(newline="") as csv_file:
                rows = list(csv.DictReader(csv_file))

        self.assertEqual(len(rows), 1)
        self.assertEqual((rows[0]["transa"], rows[0]["transb"]), ("N", "N"))


class GemmCheckerTests(unittest.TestCase):
    def run_checker(
        self,
        rows,
        *extra_args,
        fieldnames=None,
        comparators=("Reference",),
    ):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir).resolve() / "gemm.csv"
            with path.open("w", newline="") as csv_file:
                writer = csv.DictWriter(
                    csv_file, fieldnames=fieldnames or runner.CSV_FIELDNAMES
                )
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

    def test_checker_keeps_transpose_groups_separate(self):
        rows = [
            gemm_row("Zynum", "NN", 2),
            gemm_row("Reference", "NN", 1),
            gemm_row("Zynum", "NT", 2),
        ]
        result, stdout, stderr = self.run_checker(rows)
        self.assertEqual(result, 1, stderr)
        self.assertIn("checked=1 passed=1 failed=0 missing=1", stdout)
        self.assertIn("trans=NT", stdout)

    def test_checker_best_median_and_min_statistics(self):
        rows = [
            gemm_row("Zynum", "NN", 10, best_ns=1, median_ns=8, max_ns=16),
            gemm_row("Reference", "NN", 8, best_ns=2, median_ns=4, max_ns=8),
        ]
        best_result, best_stdout, best_stderr = self.run_checker(rows)
        median_result, median_stdout, median_stderr = self.run_checker(
            rows, "--stat", "median"
        )
        min_result, min_stdout, min_stderr = self.run_checker(rows, "--stat", "min")

        self.assertEqual(best_result, 0, best_stderr)
        self.assertIn("passed=1 failed=0", best_stdout)
        self.assertEqual(median_result, 1, median_stderr)
        self.assertIn("stat=median", median_stdout)
        self.assertEqual(min_result, 1, min_stderr)
        self.assertIn("stat=min", min_stdout)

    def test_checker_treats_legacy_rows_as_nn(self):
        rows = [gemm_row("Zynum", "NN", 2), gemm_row("Reference", "NN", 1)]
        legacy_fields = [
            field
            for field in runner.CSV_FIELDNAMES
            if field not in {"transa", "transb", "process_repeats"}
        ]
        for row in rows:
            row.pop("transa")
            row.pop("transb")
            row.pop("process_repeats")
        result, stdout, stderr = self.run_checker(
            rows, "--trans", "NN", fieldnames=legacy_fields
        )
        self.assertEqual(result, 0, stderr)
        self.assertIn("checked=1 passed=1 failed=0", stdout)

    def test_complex_median_uses_complex_flop_factor(self):
        row = gemm_row("Zynum", "CC", 1, kind="cgemm", median_ns=2)
        self.assertEqual(checker.row_gflops(row, "median"), 32.0)

    def test_checker_accepts_fractional_median_timing(self):
        rows = [
            gemm_row("Zynum", "NN", 2, median_ns=10.5),
            gemm_row("Reference", "NN", 1, median_ns=21.5),
        ]
        result, stdout, stderr = self.run_checker(rows, "--stat", "median")
        self.assertEqual(result, 0, stderr)
        self.assertIn("passed=1 failed=0", stdout)

    def test_checker_rejects_partial_or_inconsistent_process_repeats(self):
        zynum = gemm_row("Zynum", "NN", 2)
        reference = gemm_row("Reference", "NN", 1)
        reference["process_repeats"] = ""
        result, _, stderr = self.run_checker([zynum, reference])
        self.assertEqual(result, 2)
        self.assertIn("partial process_repeats evidence", stderr)

        reference["process_repeats"] = "2"
        result, _, stderr = self.run_checker([zynum, reference])
        self.assertEqual(result, 2)
        self.assertIn("inconsistent process_repeats evidence", stderr)

        for value in ("0", "-1", "1.5", "nan"):
            with self.subTest(value=value):
                zynum["process_repeats"] = value
                result, _, stderr = self.run_checker([zynum, reference])
                self.assertEqual(result, 2)
                self.assertIn("bad process_repeats evidence", stderr)

    def test_checker_rejects_invalid_metrics_thresholds_and_ratio_extremes(self):
        for value in ("nan", "inf", "-inf", "0", "-1"):
            with self.subTest(metric=value):
                rows = [
                    gemm_row("Zynum", "NN", value),
                    gemm_row("Reference", "NN", 1),
                ]
                result, _, _ = self.run_checker(rows)
                self.assertEqual(result, 2)
            with self.subTest(threshold=value):
                result, _, _ = self.run_checker(
                    [gemm_row("Zynum", "NN", 2), gemm_row("Reference", "NN", 1)],
                    f"--ratio={value}",
                )
                self.assertEqual(result, 2)
        for candidate, comparator in (("1e308", "1e-308"), ("1e-308", "1e308")):
            result, _, stderr = self.run_checker(
                [
                    gemm_row("Zynum", "NN", candidate),
                    gemm_row("Reference", "NN", comparator),
                ]
            )
            self.assertEqual(result, 2)
            self.assertIn("bad comparison ratio", stderr)

    def test_checker_rejects_duplicates_and_is_shuffle_stable(self):
        original = gemm_row("Zynum", "NN", 0.5)
        for duplicate in (dict(original), gemm_row("Zynum", "NN", 9)):
            with self.subTest(value=duplicate["gflops"]):
                result, _, stderr = self.run_checker(
                    [original, duplicate, gemm_row("Reference", "NN", 1)]
                )
                self.assertEqual(result, 2)
                self.assertIn("duplicate library row", stderr)

        rows = [
            original,
            gemm_row("Reference", "NN", 1),
            gemm_row("Second", "NN", 1),
            gemm_row("Zynum", "NT", 0.25),
            gemm_row("Reference", "NT", 1),
        ]
        first = self.run_checker(rows, comparators=("Reference", "Second"))
        shuffled = self.run_checker(
            list(reversed(rows)), comparators=("Reference", "Second")
        )
        reversed_cli = self.run_checker(rows, comparators=("Second", "Reference"))
        self.assertEqual(first, shuffled)
        self.assertEqual(first, reversed_cli)
        self.assertEqual(first[0], 1)
        self.assertIn("best=Reference:1.000000", first[1])


if __name__ == "__main__":
    unittest.main()
