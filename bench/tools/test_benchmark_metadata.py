#!/usr/bin/env python3
# Copyright (C) 2026 Zynum contributors
# SPDX-License-Identifier: LGPL-3.0-or-later

import argparse
import hashlib
import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import benchmark_metadata


def clean_git_environment(**updates):
    environment = {
        name: value
        for name, value in os.environ.items()
        if not name.upper().startswith("GIT_")
    }
    environment.update(updates)
    return environment


def run_git(root, *arguments):
    return subprocess.run(
        ["git", "-C", str(root), *arguments],
        check=True,
        capture_output=True,
        env=clean_git_environment(),
    )


def initialize_git_repository(root, *, commit=True):
    run_git(root, "init", "--quiet")
    if commit:
        (root / "tracked.txt").write_text("fixture\n")
        run_git(root, "add", "tracked.txt")
        run_git(
            root,
            "-c",
            "user.name=Benchmark Metadata Test",
            "-c",
            "user.email=benchmark-metadata@example.invalid",
            "commit",
            "--quiet",
            "-m",
            "fixture",
        )


def coverage_document(*stable_ids):
    entries = [
        {
            "level": "level1",
            "stable_id": stable_id,
            "operation": "dot",
            "scalar": "f64",
            "implementation": "generic",
            "specialization": "test",
            "capability": "portable",
            "availability": "implemented",
            "lifecycle": "production",
            "state": "none",
            "evidence": {
                "build": False,
                "native_correctness": False,
                "native_performance": False,
            },
            "evidence_note": "test fixture",
        }
        for stable_id in stable_ids
    ]
    return {
        "schema_version": 1,
        "generator": "test fixture",
        "streaming_vector_bytes": 64,
        "summary": {
            "total": len(entries),
            "implemented": len(entries),
            "experimental": 0,
            "rejected": 0,
            "missing": 0,
            "unsupported": 0,
            "build_tested": 0,
            "native_correctness_tested": 0,
            "native_performance_tested": 0,
        },
        "entries": entries,
    }


def identity_args(*registry_ids, selected_paths=()):
    return argparse.Namespace(
        registry_id=list(registry_ids),
        selected_path=list(selected_paths),
        target_capability=[],
    )


class BenchmarkMetadataTest(unittest.TestCase):
    def test_public_serialization_is_safe_for_all_report_controllers(self):
        private_root = "/Users/private-owner/benchmark-worktree"
        sensitive_untracked = "?? customer-secret-results.csv"
        secret_argument = "--token=super-secret-argv-value"
        digest = "a" * 64
        private_metadata = {
            "argv": [
                private_root + "/bench/tools/controller.py",
                secret_argument,
            ],
            "cwd": private_root,
            "process_repeats": 4,
            "zynum_maximum_threads": 10,
            "benchmark_identity": {
                "schema_version": 2,
                "source": {
                    "repository_root": private_root,
                    "revision": "0123456789abcdef",
                    "branch": "benchmark/public-projection",
                    "dirty": True,
                    "status_short": sensitive_untracked,
                    "snapshot_manifest": private_root + "/source-identity.json",
                    "snapshot_manifest_sha256": digest,
                    "snapshot_tree_sha256": "b" * 64,
                    "identity_status": "git",
                    "cleanliness_status": "known",
                },
                "controller": {
                    "compiler": {
                        "name": "zig",
                        "version": "0.16.0",
                        "executable": private_root + "/tools/zig",
                    },
                    "python": {
                        "version": "3.13.5",
                        "executable": private_root + "/venv/bin/python",
                    },
                    "host": {
                        "machine": "arm64",
                        "detection_source": "/proc/cpuinfo",
                    },
                },
                "payload": {
                    "build": {
                        "requested": {
                            "target_triple": "x86_64-linux-gnu",
                            "cpu": "x86_64_v3",
                            "optimization": "ReleaseFast",
                        },
                        "declaration_status": "complete",
                    },
                    "artifacts": {
                        "libraries": [
                            {
                                "name": "Zynum",
                                "path": private_root + "/lib/libzynum.so",
                                "sha256": digest,
                            }
                        ]
                    },
                },
            },
            "diagnostic_message": "tool failed at " + private_root + "/probe",
        }

        for controller in sorted(benchmark_metadata.PUBLIC_CONTROLLERS):
            with self.subTest(controller=controller):
                serialized = benchmark_metadata.serialize_public_metadata(
                    private_metadata,
                    controller=controller,
                    parameter_keys=("process_repeats",),
                )
                for marker in (
                    private_root.encode(),
                    sensitive_untracked.encode(),
                    secret_argument.encode(),
                ):
                    self.assertNotIn(marker, serialized)

                public = json.loads(serialized)
                self.assertEqual(
                    public["metadata_projection"],
                    {
                        "audience": "public",
                        "private_diagnostics": "excluded",
                        "schema_version": 1,
                    },
                )
                self.assertEqual(public["command"]["controller"], controller)
                self.assertEqual(
                    public["command"]["parameters"], {"process_repeats": 4}
                )
                self.assertEqual(public["zynum_maximum_threads"], 10)
                identity = public["benchmark_identity"]
                self.assertEqual(identity["source"]["revision"], "0123456789abcdef")
                self.assertTrue(identity["source"]["dirty"])
                self.assertEqual(identity["source"]["snapshot_tree_sha256"], "b" * 64)
                self.assertEqual(
                    identity["payload"]["build"]["requested"]["cpu"], "x86_64_v3"
                )
                self.assertEqual(
                    identity["payload"]["artifacts"]["libraries"],
                    [{"name": "Zynum", "sha256": digest}],
                )
                self.assertEqual(
                    identity["controller"]["compiler"]["version"], "0.16.0"
                )

        self.assertEqual(private_metadata["argv"][1], secret_argument)
        self.assertEqual(
            private_metadata["benchmark_identity"]["source"]["status_short"],
            sensitive_untracked,
        )

    def test_preferred_identity_collector_observes_source_exactly_once(self):
        source = {
            "revision": "fixture-revision",
            "branch": "main",
            "dirty": None,
            "status_short": None,
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with (
                mock.patch.object(
                    benchmark_metadata,
                    "source_snapshot",
                    return_value=source,
                ) as observe_source,
                mock.patch.object(benchmark_metadata, "host_snapshot", return_value={}),
                mock.patch.object(
                    benchmark_metadata, "compiler_snapshot", return_value={}
                ),
            ):
                snapshot = benchmark_metadata.collect_benchmark_identity(
                    identity_args(), root=root
                )

        observe_source.assert_called_once_with(root, None)
        self.assertIs(snapshot["source"], source)

    def test_identity_snapshot_is_a_compatibility_delegate(self):
        args = identity_args()
        libraries = [("library", "/tmp/library")]
        binaries = [("binary", "/tmp/binary")]
        expected = object()
        with mock.patch.object(
            benchmark_metadata,
            "collect_benchmark_identity",
            return_value=expected,
        ) as collect:
            actual = benchmark_metadata.identity_snapshot(
                args,
                libraries=libraries,
                binaries=binaries,
                root="/tmp/root",
            )

        self.assertIs(actual, expected)
        collect.assert_called_once_with(args, libraries, binaries, "/tmp/root")

    def test_frozen_identity_matches_legacy_artifact_schema_and_bytes(self):
        source_snapshot = {
            "revision": "fixture-revision",
            "branch": "main",
            "dirty": False,
            "status_short": "",
        }
        coverage_snapshot = {
            "path": "/public/docs/kernel_coverage.json",
            "sha256": None,
            "schema_version": None,
            "generator": None,
            "summary": None,
            "stable_ids": [],
            "authority_status": "missing",
            "authority_diagnostic": "kernel_coverage_missing",
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            library = root / "libfixture.so"
            library.write_bytes(b"library bytes")
            library.chmod(0o644)
            requests = [
                benchmark_metadata.benchmark_artifacts.ArtifactRequest.library(
                    "Fixture", library
                )
            ]
            with benchmark_metadata.benchmark_artifacts.ArtifactSnapshotSet(
                requests
            ) as artifacts:
                with (
                    mock.patch.object(
                        benchmark_metadata,
                        "source_snapshot",
                        return_value=source_snapshot,
                    ),
                    mock.patch.object(
                        benchmark_metadata,
                        "coverage_snapshot",
                        return_value=coverage_snapshot,
                    ),
                    mock.patch.object(
                        benchmark_metadata, "host_snapshot", return_value={}
                    ),
                    mock.patch.object(
                        benchmark_metadata, "compiler_snapshot", return_value={}
                    ),
                ):
                    frozen = benchmark_metadata.collect_benchmark_identity_from_frozen(
                        identity_args(),
                        libraries=artifacts.for_role("library"),
                        root=root,
                    )
                    legacy = benchmark_metadata.collect_benchmark_identity(
                        identity_args(),
                        libraries=[("Fixture", str(library))],
                        root=root,
                    )

                frozen_bytes = json.dumps(
                    frozen, separators=(",", ":"), ensure_ascii=True
                ).encode()
                legacy_bytes = json.dumps(
                    legacy, separators=(",", ":"), ensure_ascii=True
                ).encode()
                self.assertEqual(frozen_bytes, legacy_bytes)
                digest = hashlib.sha256(b"library bytes").hexdigest()
                expected_artifacts = (
                    '{"hash_claim_scope":"content_identity_only; artifact hashes do not '
                    'prove build flags","binaries":[],"libraries":[{"name":"Fixture",'
                    '"path":'
                    + json.dumps(str(library))
                    + ',"sha256":"'
                    + digest
                    + '"}]}'
                ).encode()
                self.assertEqual(
                    json.dumps(
                        frozen["payload"]["artifacts"],
                        separators=(",", ":"),
                        ensure_ascii=True,
                    ).encode(),
                    expected_artifacts,
                )

    def test_frozen_identity_never_live_hashes_or_exposes_private_path(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            library = root / "libfixture.so"
            library.write_bytes(b"A")
            library.chmod(0o644)
            with benchmark_metadata.benchmark_artifacts.ArtifactSnapshotSet(
                [
                    benchmark_metadata.benchmark_artifacts.ArtifactRequest.library(
                        "Fixture", library
                    )
                ]
            ) as artifacts:
                frozen_artifact = artifacts.artifacts[0]
                private_path = frozen_artifact.execution_path
                library.write_bytes(b"B")
                with (
                    mock.patch.object(
                        benchmark_metadata,
                        "sha256_file",
                        side_effect=AssertionError("live hashing is forbidden"),
                    ),
                    mock.patch.object(
                        benchmark_metadata, "source_snapshot", return_value={}
                    ),
                    mock.patch.object(
                        benchmark_metadata,
                        "coverage_snapshot",
                        return_value={
                            "stable_ids": [],
                            "authority_status": "missing",
                            "authority_diagnostic": "kernel_coverage_missing",
                        },
                    ),
                    mock.patch.object(
                        benchmark_metadata, "host_snapshot", return_value={}
                    ),
                    mock.patch.object(
                        benchmark_metadata, "compiler_snapshot", return_value={}
                    ),
                ):
                    identity = (
                        benchmark_metadata.collect_benchmark_identity_from_frozen(
                            identity_args(), libraries=[frozen_artifact], root=root
                        )
                    )

                record = identity["payload"]["artifacts"]["libraries"][0]
                self.assertEqual(record["path"], str(library))
                self.assertEqual(record["sha256"], hashlib.sha256(b"A").hexdigest())
                self.assertNotIn(private_path, json.dumps(identity))

    def test_frozen_identity_rejects_live_tuples_dicts_and_wrong_roles(self):
        for item in (("library", "/tmp/library"), {"name": "library"}):
            with self.subTest(item_type=type(item).__name__):
                with self.assertRaisesRegex(TypeError, "FrozenArtifact"):
                    benchmark_metadata.collect_benchmark_identity_from_frozen(
                        identity_args(), libraries=[item]
                    )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            binary = root / "probe"
            binary.write_bytes(b"binary")
            binary.chmod(0o755)
            with benchmark_metadata.benchmark_artifacts.ArtifactSnapshotSet(
                [
                    benchmark_metadata.benchmark_artifacts.ArtifactRequest.binary(
                        "probe", binary
                    )
                ]
            ) as artifacts:
                with self.assertRaisesRegex(ValueError, "does not match"):
                    benchmark_metadata.collect_benchmark_identity_from_frozen(
                        identity_args(), libraries=artifacts.artifacts, root=root
                    )

    def test_legacy_source_projections_preserve_all_five_identity_states(self):
        cases = (
            ("git", "git-revision", "main", False, ""),
            ("exported", "exported-revision", "snapshot", True, " M file"),
            ("no_git", None, None, None, None),
            ("unavailable", None, None, None, None),
            ("unreadable", None, None, None, None),
        )
        for status, revision, branch, dirty, status_short in cases:
            with self.subTest(status=status):
                source = {
                    "identity_status": status,
                    "revision": revision,
                    "branch": branch,
                    "dirty": dirty,
                    "status_short": status_short,
                    "identity_diagnostic": "not part of the legacy projection",
                }
                self.assertEqual(
                    benchmark_metadata.legacy_source_snapshot(source),
                    {
                        "revision": revision,
                        "branch": branch,
                        "dirty": dirty,
                        "status_short": status_short,
                    },
                )
                self.assertEqual(
                    benchmark_metadata.source_git_revision(source), revision
                )

        self.assertEqual(
            benchmark_metadata.legacy_source_snapshot({}),
            {
                "revision": None,
                "branch": None,
                "dirty": None,
                "status_short": None,
            },
        )
        self.assertIsNone(benchmark_metadata.source_git_revision({}))

    def test_identity_records_validated_selection_and_hashes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "docs").mkdir()
            coverage = root / "docs" / "kernel_coverage.json"
            coverage.write_text(
                json.dumps(coverage_document("level1.dot.f64.test.generic"))
            )
            library = root / "libtest.so"
            library.write_bytes(b"library")
            args = argparse.Namespace(
                registry_id=["level1.dot.f64.test.generic"],
                selected_path=["ddot:n=64=level1.dot.f64.test.generic"],
                target_capability=["test_capability"],
            )
            snapshot = benchmark_metadata.identity_snapshot(
                args,
                libraries=[("test", str(library))],
                root=root,
            )
            self.assertEqual(snapshot["schema_version"], 2)
            self.assertEqual(
                set(snapshot), {"schema_version", "source", "controller", "payload"}
            )
            self.assertEqual(
                set(snapshot["controller"]), {"host", "compiler", "python"}
            )
            self.assertEqual(
                set(snapshot["payload"]),
                {
                    "build",
                    "declared_capabilities",
                    "kernel_selection",
                    "artifacts",
                },
            )
            self.assertEqual(
                snapshot["payload"]["kernel_selection"]["registry_ids"],
                ["level1.dot.f64.test.generic"],
            )
            self.assertEqual(
                snapshot["payload"]["kernel_selection"]["selected_paths"][0]["scope"],
                "ddot:n=64",
            )
            self.assertEqual(
                snapshot["payload"]["declared_capabilities"],
                ["test_capability"],
            )
            self.assertIsNotNone(
                snapshot["payload"]["artifacts"]["libraries"][0]["sha256"]
            )
            self.assertIn(
                "do not prove build flags",
                snapshot["payload"]["artifacts"]["hash_claim_scope"],
            )
            self.assertIsNotNone(
                snapshot["payload"]["kernel_selection"]["coverage_artifact"]["sha256"]
            )
            self.assertNotIn("declared_capabilities", snapshot["controller"]["host"])

    def test_build_declaration_complete_partial_and_unspecified(self):
        complete = argparse.Namespace(
            build_target="aarch64-macos.15.0",
            build_cpu="apple_m4+sme+sme2",
            build_optimize="ReleaseFast",
        )
        partial = argparse.Namespace(
            build_target="x86_64-linux-gnu",
            build_cpu=None,
            build_optimize=None,
        )

        complete_build = benchmark_metadata.build_declaration(complete)
        partial_build = benchmark_metadata.build_declaration(partial)
        unspecified_build = benchmark_metadata.build_declaration(argparse.Namespace())

        self.assertEqual(complete_build["declaration_status"], "complete")
        self.assertEqual(complete_build["missing_fields"], [])
        self.assertEqual(
            complete_build["requested"],
            {
                "target_triple": "aarch64-macos.15.0",
                "cpu": "apple_m4+sme+sme2",
                "optimization": "ReleaseFast",
            },
        )
        self.assertEqual(partial_build["declaration_status"], "partial")
        self.assertEqual(partial_build["missing_fields"], ["cpu", "optimization"])
        self.assertEqual(unspecified_build["declaration_status"], "unspecified")
        self.assertEqual(
            unspecified_build["missing_fields"],
            ["target_triple", "cpu", "optimization"],
        )
        self.assertEqual(complete_build["validation"]["artifact_match"], "not_verified")

    def test_build_declaration_rejects_invalid_namespace_values(self):
        invalid = (
            argparse.Namespace(
                build_target="x86_64 linux",
                build_cpu=None,
                build_optimize=None,
            ),
            argparse.Namespace(
                build_target=None,
                build_cpu="native\npoison",
                build_optimize=None,
            ),
            argparse.Namespace(
                build_target=None,
                build_cpu=None,
                build_optimize="fast",
            ),
        )
        for args in invalid:
            with self.subTest(args=vars(args)):
                with self.assertRaises(ValueError):
                    benchmark_metadata.build_declaration(args)

    def test_identity_cli_accepts_only_valid_build_declarations(self):
        parser = argparse.ArgumentParser()
        benchmark_metadata.add_identity_arguments(parser)
        args = parser.parse_args(
            [
                "--build-target",
                "x86_64-linux-gnu",
                "--build-cpu",
                "x86_64_v3+avx512f",
                "--build-optimize",
                "ReleaseSafe",
            ]
        )
        self.assertEqual(args.build_target, "x86_64-linux-gnu")
        self.assertEqual(args.build_cpu, "x86_64_v3+avx512f")
        self.assertEqual(args.build_optimize, "ReleaseSafe")

        for invalid_args in (
            ["--build-target", ""],
            ["--build-cpu", "native/escape"],
            ["--build-optimize", "fast"],
        ):
            with self.subTest(invalid_args=invalid_args):
                with (
                    mock.patch("sys.stderr"),
                    self.assertRaises(SystemExit) as raised,
                ):
                    parser.parse_args(invalid_args)
                self.assertEqual(raised.exception.code, 2)

    def test_unknown_registry_id_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "docs").mkdir()
            (root / "docs" / "kernel_coverage.json").write_text(
                json.dumps(coverage_document("known"))
            )
            args = argparse.Namespace(
                registry_id=["unknown"], selected_path=[], target_capability=[]
            )
            with self.assertRaisesRegex(ValueError, "absent"):
                benchmark_metadata.identity_snapshot(args, root=root)

    def test_registry_identity_requires_valid_coverage_authority(self):
        cases = {
            "missing": None,
            "malformed": "{not-json",
            "empty": json.dumps(coverage_document()),
        }
        for expected_status, contents in cases.items():
            with self.subTest(expected_status=expected_status):
                with tempfile.TemporaryDirectory() as directory:
                    root = Path(directory)
                    if contents is not None:
                        (root / "docs").mkdir()
                        (root / "docs" / "kernel_coverage.json").write_text(contents)
                    args = argparse.Namespace(
                        registry_id=["claimed-id"],
                        selected_path=[],
                        target_capability=[],
                    )

                    with self.assertRaisesRegex(
                        ValueError, "coverage is {}".format(expected_status)
                    ):
                        benchmark_metadata.identity_snapshot(args, root=root)

    def test_selected_path_identity_requires_valid_coverage_authority(self):
        with tempfile.TemporaryDirectory() as directory:
            args = argparse.Namespace(
                registry_id=[],
                selected_path=["ddot:n=64=claimed-id"],
                target_capability=[],
            )

            with self.assertRaisesRegex(ValueError, "coverage is missing"):
                benchmark_metadata.identity_snapshot(args, root=directory)

    def test_no_registry_identity_does_not_require_coverage_authority(self):
        cases = {
            "missing": None,
            "malformed": "{not-json",
            "empty": json.dumps(coverage_document()),
        }
        for expected_status, contents in cases.items():
            with self.subTest(expected_status=expected_status):
                with tempfile.TemporaryDirectory() as directory:
                    root = Path(directory)
                    if contents is not None:
                        (root / "docs").mkdir()
                        (root / "docs" / "kernel_coverage.json").write_text(contents)
                    args = argparse.Namespace(
                        registry_id=[], selected_path=[], target_capability=[]
                    )

                    snapshot = benchmark_metadata.identity_snapshot(args, root=root)

                    selection = snapshot["payload"]["kernel_selection"]
                    self.assertEqual(selection["registry_ids"], [])
                    self.assertEqual(
                        selection["coverage_artifact"]["authority_status"],
                        expected_status,
                    )
                    if contents is not None:
                        self.assertIsNotNone(selection["coverage_artifact"]["sha256"])

    def test_mixed_invalid_entries_cannot_authorize_valid_row(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "docs").mkdir()
            document = coverage_document("known")
            document["entries"].append({"stable_id": "broken"})
            document["summary"]["total"] = 2
            (root / "docs" / "kernel_coverage.json").write_text(json.dumps(document))

            with self.assertRaisesRegex(ValueError, "coverage is malformed"):
                benchmark_metadata.identity_snapshot(identity_args("known"), root=root)

    def test_duplicate_stable_ids_are_malformed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "docs").mkdir()
            (root / "docs" / "kernel_coverage.json").write_text(
                json.dumps(coverage_document("duplicate", "duplicate"))
            )

            coverage = benchmark_metadata.coverage_snapshot(root)

            self.assertEqual(coverage["authority_status"], "malformed")
            self.assertIn("duplicate stable_id", coverage["authority_detail"])
            with self.assertRaisesRegex(ValueError, "coverage is malformed"):
                benchmark_metadata.identity_snapshot(
                    identity_args("duplicate"), root=root
                )

    def test_wrong_schema_and_entries_type_are_malformed(self):
        documents = {
            "wrong schema": coverage_document("known"),
            "wrong entries type": coverage_document("known"),
        }
        documents["wrong schema"]["schema_version"] = 2
        documents["wrong entries type"]["entries"] = {"stable_id": "known"}
        for label, document in documents.items():
            with self.subTest(label=label):
                with tempfile.TemporaryDirectory() as directory:
                    root = Path(directory)
                    (root / "docs").mkdir()
                    (root / "docs" / "kernel_coverage.json").write_text(
                        json.dumps(document)
                    )

                    coverage = benchmark_metadata.coverage_snapshot(root)

                    self.assertEqual(coverage["authority_status"], "malformed")

    def test_inconsistent_summary_is_malformed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "docs").mkdir()
            document = coverage_document("known")
            document["summary"]["total"] = 0
            (root / "docs" / "kernel_coverage.json").write_text(json.dumps(document))

            coverage = benchmark_metadata.coverage_snapshot(root)

            self.assertEqual(coverage["authority_status"], "malformed")
            self.assertIn("summary.total", coverage["authority_detail"])

    def test_control_capture_failure_is_classified_and_claims_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "docs").mkdir()
            (root / "docs" / "kernel_coverage.json").write_text(
                json.dumps(coverage_document("known"))
            )
            with mock.patch.object(
                benchmark_metadata.repository_snapshot,
                "capture_control_artifact",
                side_effect=benchmark_metadata.repository_snapshot.RepositorySnapshotError(
                    "fixture capture denial"
                ),
            ):
                snapshot = benchmark_metadata.identity_snapshot(
                    identity_args(), root=root
                )
                with self.assertRaisesRegex(ValueError, "coverage is unreadable"):
                    benchmark_metadata.identity_snapshot(
                        identity_args("known"), root=root
                    )

            coverage = snapshot["payload"]["kernel_selection"]["coverage_artifact"]
            self.assertEqual(coverage["authority_status"], "unreadable")
            self.assertEqual(
                coverage["authority_diagnostic"], "kernel_coverage_read_failed"
            )

    def test_coverage_hash_and_parse_use_the_same_frozen_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "docs").mkdir()
            path = root / "docs" / "kernel_coverage.json"
            old_bytes = json.dumps(coverage_document("old-id")).encode()
            path.write_text(json.dumps(coverage_document("new-id")))
            artifact = mock.Mock(
                bytes=old_bytes,
                sha256=hashlib.sha256(old_bytes).hexdigest(),
            )
            with mock.patch.object(
                benchmark_metadata.repository_snapshot,
                "capture_control_artifact",
                return_value=artifact,
            ):
                coverage = benchmark_metadata.coverage_snapshot(root)

            self.assertEqual(coverage["authority_status"], "valid")
            self.assertEqual(coverage["stable_ids"], ["old-id"])
            self.assertEqual(coverage["sha256"], hashlib.sha256(old_bytes).hexdigest())

    def test_coverage_rejects_duplicate_keys_constants_and_non_objects(self):
        document = json.dumps(coverage_document("known"))
        malformed_documents = {
            "duplicate key": document.replace(
                '"schema_version": 1',
                '"schema_version": 1, "schema_version": 1',
                1,
            ),
            "NaN": document[:-1] + ', "nonstandard": NaN}',
            "Infinity": document[:-1] + ', "nonstandard": Infinity}',
            "negative Infinity": document[:-1] + ', "nonstandard": -Infinity}',
            "non-object": "[]",
        }
        for label, contents in malformed_documents.items():
            with self.subTest(label=label), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                (root / "docs").mkdir()
                (root / "docs" / "kernel_coverage.json").write_text(contents)

                coverage = benchmark_metadata.coverage_snapshot(root)

                self.assertEqual(coverage["authority_status"], "malformed")
                self.assertEqual(
                    coverage["authority_diagnostic"],
                    "kernel_coverage_invalid_json",
                )

    def test_coverage_rejects_symlink_special_and_oversized_controls(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            docs = root / "docs"
            docs.mkdir()
            path = docs / "kernel_coverage.json"
            target = root / "target.json"
            target.write_text(json.dumps(coverage_document("known")))
            path.symlink_to(target)
            symlink = benchmark_metadata.coverage_snapshot(root)
            self.assertEqual(symlink["authority_status"], "unreadable")
            self.assertEqual(
                symlink["authority_diagnostic"],
                "kernel_coverage_unsafe_file_type",
            )

            path.unlink()
            os.mkfifo(path)
            special = benchmark_metadata.coverage_snapshot(root)
            self.assertEqual(special["authority_status"], "unreadable")
            self.assertEqual(
                special["authority_diagnostic"],
                "kernel_coverage_unsafe_file_type",
            )

            path.unlink()
            path.write_bytes(b"{}")
            with mock.patch.object(benchmark_metadata, "MAX_CONTROL_ARTIFACT_BYTES", 1):
                oversized = benchmark_metadata.coverage_snapshot(root)
            self.assertEqual(oversized["authority_status"], "unreadable")
            self.assertEqual(
                oversized["authority_diagnostic"], "kernel_coverage_read_failed"
            )

    def test_compiler_snapshot_parses_zig_0_16_object_environment(self):
        environment = """\
.{
    .zig_exe = "/usr/local/bin/zig",
    .lib_dir = "/usr/local/lib/zig",
    .version = "0.16.0",
    .target = "x86_64-linux.6.1...6.1-gnu.2.36",
    .env = .{
        .HOME = "/home/fixture",
        .NO_COLOR = "1",
    },
}
"""

        def output(command):
            return environment.strip() if command == ["zig", "env"] else "0.16.0"

        with (
            mock.patch.object(benchmark_metadata.shutil, "which", return_value="zig"),
            mock.patch.object(benchmark_metadata, "command_output", side_effect=output),
        ):
            snapshot = benchmark_metadata.compiler_snapshot()

        self.assertEqual(snapshot["default_target"], "x86_64-linux.6.1...6.1-gnu.2.36")
        self.assertEqual(snapshot["executable"], "/usr/local/bin/zig")
        self.assertEqual(
            snapshot["default_target_diagnostic"],
            {
                "status": "ok",
                "classification": "zig_env_parsed",
                "format": "zig-object",
            },
        )

    def test_compiler_snapshot_classifies_invalid_environment_output(self):
        def output(command):
            return "not valid zig env" if command == ["zig", "env"] else "0.16.0"

        with (
            mock.patch.object(benchmark_metadata.shutil, "which", return_value="zig"),
            mock.patch.object(benchmark_metadata, "command_output", side_effect=output),
        ):
            snapshot = benchmark_metadata.compiler_snapshot()

        self.assertIsNone(snapshot["default_target"])
        self.assertEqual(
            snapshot["default_target_diagnostic"]["classification"],
            "zig_env_parse_error",
        )

    def test_selected_path_requires_scope_and_id(self):
        with self.assertRaisesRegex(ValueError, "SCOPE=ID"):
            benchmark_metadata.parse_selected_paths(["missing-scope"])

    def test_exported_source_identity_overrides_missing_git_worktree(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            identity = root / "source_identity.json"
            identity_bytes = json.dumps(
                {
                    "revision": "0123456789abcdef",
                    "branch": "fixture-branch",
                    "dirty": True,
                    "status_short": " M src/example.zig",
                    "snapshot_tree_sha256": "tree-digest",
                    "snapshot_created_utc": "2026-07-18T00:00:00Z",
                }
            ).encode()
            identity.write_bytes(identity_bytes)

            with mock.patch.dict(
                os.environ,
                clean_git_environment(GIT_DIR="/redirected/.git"),
                clear=True,
            ):
                snapshot = benchmark_metadata.source_snapshot(root, identity)

            self.assertEqual(snapshot["revision"], "0123456789abcdef")
            self.assertTrue(snapshot["dirty"])
            self.assertEqual(snapshot["identity_status"], "exported")
            self.assertEqual(snapshot["cleanliness_status"], "known")
            self.assertEqual(snapshot["snapshot_tree_sha256"], "tree-digest")
            self.assertEqual(
                snapshot["snapshot_manifest_sha256"],
                hashlib.sha256(identity_bytes).hexdigest(),
            )

    def test_exported_identity_hash_and_parse_use_the_same_frozen_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "source_identity.json"
            old_bytes = json.dumps({"revision": "old-revision"}).encode()
            path.write_text(json.dumps({"revision": "new-revision"}))
            artifact = mock.Mock(
                bytes=old_bytes,
                sha256=hashlib.sha256(old_bytes).hexdigest(),
            )
            with mock.patch.object(
                benchmark_metadata.repository_snapshot,
                "capture_control_artifact",
                return_value=artifact,
            ):
                snapshot = benchmark_metadata.source_snapshot(root, path)

            self.assertEqual(snapshot["revision"], "old-revision")
            self.assertEqual(
                snapshot["snapshot_manifest_sha256"],
                hashlib.sha256(old_bytes).hexdigest(),
            )

    def test_exported_identity_rejects_duplicate_revision(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            identity = root / "source_identity.json"
            identity.write_text(
                '{"revision":"first","revision":"second"}',
            )

            with self.assertRaisesRegex(ValueError, "duplicate object key"):
                benchmark_metadata.source_snapshot(root, identity)

    def test_exported_identity_rejects_constants_and_non_objects(self):
        malformed_documents = {
            "NaN": '{"revision":"valid","branch":NaN}',
            "Infinity": '{"revision":"valid","branch":Infinity}',
            "negative Infinity": '{"revision":"valid","branch":-Infinity}',
            "non-object": "[]",
        }
        for label, contents in malformed_documents.items():
            with self.subTest(label=label), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                identity = root / "source_identity.json"
                identity.write_text(contents)

                with self.assertRaises(ValueError):
                    benchmark_metadata.source_snapshot(root, identity)

    def test_exported_identity_rejects_wrong_field_types(self):
        wrong_json_types = (False, 0, 1.5, [], {})
        for revision in (None, "", *wrong_json_types):
            with self.subTest(field="revision", value=revision):
                with tempfile.TemporaryDirectory() as directory:
                    root = Path(directory)
                    identity = root / "source_identity.json"
                    identity.write_text(json.dumps({"revision": revision}))

                    with self.assertRaisesRegex(ValueError, "non-empty revision"):
                        benchmark_metadata.source_snapshot(root, identity)

        optional_string_fields = (
            "branch",
            "repository_root",
            "status_short",
            "snapshot_tree_sha256",
            "snapshot_created_utc",
        )
        for field in optional_string_fields:
            for value in wrong_json_types:
                with self.subTest(field=field, value=value):
                    with tempfile.TemporaryDirectory() as directory:
                        root = Path(directory)
                        identity = root / "source_identity.json"
                        identity.write_text(
                            json.dumps({"revision": "valid", field: value})
                        )

                        with self.assertRaisesRegex(
                            ValueError, "{} must be null or a string".format(field)
                        ):
                            benchmark_metadata.source_snapshot(root, identity)

        for dirty in (0, 1, 1.5, "", [], {}):
            with self.subTest(field="dirty", value=dirty):
                with tempfile.TemporaryDirectory() as directory:
                    root = Path(directory)
                    identity = root / "source_identity.json"
                    identity.write_text(
                        json.dumps({"revision": "valid", "dirty": dirty})
                    )

                    with self.assertRaisesRegex(
                        ValueError, "dirty must be null or a boolean"
                    ):
                        benchmark_metadata.source_snapshot(root, identity)

    def test_exported_identity_accepts_nulls_and_string_contents(self):
        optional_string_fields = (
            "branch",
            "repository_root",
            "status_short",
            "snapshot_tree_sha256",
            "snapshot_created_utc",
        )
        documents = (
            {
                "revision": "revision",
                "dirty": None,
                **{field: None for field in optional_string_fields},
            },
            {
                "revision": " ",
                "dirty": False,
                **{field: "" for field in optional_string_fields},
            },
        )
        for document in documents:
            with (
                self.subTest(document=document),
                tempfile.TemporaryDirectory() as directory,
            ):
                root = Path(directory)
                identity = root / "source_identity.json"
                identity.write_text(json.dumps(document))

                snapshot = benchmark_metadata.source_snapshot(root, identity)

                self.assertEqual(snapshot["revision"], document["revision"])
                self.assertIs(snapshot["dirty"], document["dirty"])
                for field in optional_string_fields:
                    self.assertEqual(snapshot[field], document[field])

    def test_exported_identity_rejects_symlink_special_and_oversized_controls(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "source_identity.json"
            target = root / "target.json"
            target.write_text(json.dumps({"revision": "target"}))
            path.symlink_to(target)
            with self.assertRaisesRegex(ValueError, "cannot read --source-identity"):
                benchmark_metadata.source_snapshot(root, path)

            path.unlink()
            os.mkfifo(path)
            with self.assertRaisesRegex(ValueError, "cannot read --source-identity"):
                benchmark_metadata.source_snapshot(root, path)

            path.unlink()
            path.write_bytes(b"{}")
            with (
                mock.patch.object(benchmark_metadata, "MAX_CONTROL_ARTIFACT_BYTES", 1),
                self.assertRaisesRegex(ValueError, "cannot read --source-identity"),
            ):
                benchmark_metadata.source_snapshot(root, path)

    def test_no_git_source_identity_never_claims_clean(self):
        with tempfile.TemporaryDirectory() as directory:
            with mock.patch.dict(
                os.environ,
                clean_git_environment(GIT_DIR="/unrelated/repository/.git"),
                clear=True,
            ):
                snapshot = benchmark_metadata.source_snapshot(directory)

        self.assertIsNone(snapshot["revision"])
        self.assertIsNone(snapshot["dirty"])
        self.assertIsNone(snapshot["status_short"])
        self.assertEqual(snapshot["identity_status"], "no_git")
        self.assertEqual(snapshot["cleanliness_status"], "unknown")

    def test_git_command_failure_never_claims_clean(self):
        with tempfile.TemporaryDirectory() as directory:
            with mock.patch.object(
                benchmark_metadata.repository_git,
                "open_repository",
                side_effect=benchmark_metadata.repository_git.RepositoryGitCommandError(
                    "fixture"
                ),
            ):
                snapshot = benchmark_metadata.source_snapshot(directory)

        self.assertIsNone(snapshot["revision"])
        self.assertIsNone(snapshot["dirty"])
        self.assertEqual(snapshot["identity_status"], "unreadable")
        self.assertEqual(snapshot["cleanliness_status"], "unknown")

    def test_valid_git_source_identity_preserves_clean_state(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            initialize_git_repository(root)
            with mock.patch.dict(os.environ, clean_git_environment(), clear=True):
                snapshot = benchmark_metadata.source_snapshot(root)

        self.assertRegex(snapshot["revision"], r"\A[0-9a-f]{40}(?:[0-9a-f]{24})?\Z")
        self.assertFalse(snapshot["dirty"])
        self.assertEqual(snapshot["identity_status"], "git")
        self.assertEqual(snapshot["cleanliness_status"], "known")

    def test_detached_git_source_identity_preserves_revision_without_branch(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            initialize_git_repository(root)
            run_git(root, "checkout", "--detach", "--quiet")
            with mock.patch.dict(os.environ, clean_git_environment(), clear=True):
                snapshot = benchmark_metadata.source_snapshot(root)

        self.assertIsNotNone(snapshot["revision"])
        self.assertIsNone(snapshot["branch"])
        self.assertEqual(snapshot["identity_status"], "git")
        self.assertEqual(snapshot["cleanliness_status"], "known")

    def test_identity_rejects_any_field_change_between_complete_rounds(self):
        git = benchmark_metadata.repository_git

        def result(stdout, returncode=0, stderr=b""):
            return subprocess.CompletedProcess((), returncode, stdout, stderr)

        revision = b"1" * 40 + b"\n"
        branch = b"main\n"
        status = b""
        index = b"100644 " + b"2" * 40 + b" 0\ttracked.txt\0"
        first_round = [
            result(revision),
            result(branch),
            result(status),
            result(index),
        ]
        changed = {
            "revision": result(b"3" * 40 + b"\n"),
            "branch": result(b"other\n"),
            "status": result(b"?? new.txt\n"),
            "index": result(b"100644 " + b"4" * 40 + b" 0\ttracked.txt\0"),
        }
        for offset, (field, replacement) in enumerate(changed.items()):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as directory:
                second_round = list(first_round)
                second_round[offset] = replacement
                repository = git.RepositoryGit(Path(directory), Path("/git"), {})
                with mock.patch.object(
                    git.RepositoryGit,
                    "run",
                    side_effect=first_round + second_round,
                ) as run:
                    with self.assertRaises(git.RepositoryGitSnapshotError):
                        repository.observe_identity(include_index=True)
                self.assertEqual(run.call_count, 8)

    def test_failed_status_with_empty_stdout_is_not_a_clean_identity(self):
        git = benchmark_metadata.repository_git
        responses = [
            subprocess.CompletedProcess((), 0, b"1" * 40 + b"\n", b""),
            subprocess.CompletedProcess((), 0, b"main\n", b""),
            git.RepositoryGitCommandError("status observation", 1),
        ]
        with tempfile.TemporaryDirectory() as directory:
            repository = git.RepositoryGit(Path(directory), Path("/git"), {})
            with mock.patch.object(
                git.RepositoryGit,
                "run",
                side_effect=(responses[0], responses[1], responses[2]),
            ):
                with self.assertRaises(git.RepositoryGitCommandError):
                    repository.observe_identity()

    def test_branch_observation_failure_makes_source_identity_unknown(self):
        repository = mock.Mock()
        repository.observe_identity.side_effect = (
            benchmark_metadata.repository_git.RepositoryGitCommandError(
                "branch observation"
            )
        )
        with tempfile.TemporaryDirectory() as directory:
            with mock.patch.object(
                benchmark_metadata.repository_git,
                "open_repository",
                return_value=repository,
            ):
                snapshot = benchmark_metadata.source_snapshot(directory)

        self.assertIsNone(snapshot["branch"])
        self.assertIsNone(snapshot["dirty"])
        self.assertEqual(snapshot["identity_status"], "unreadable")
        self.assertEqual(snapshot["cleanliness_status"], "unknown")

    def test_git_without_head_is_unavailable(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            initialize_git_repository(root, commit=False)
            with mock.patch.dict(os.environ, clean_git_environment(), clear=True):
                snapshot = benchmark_metadata.source_snapshot(root)

        self.assertIsNone(snapshot["revision"])
        self.assertEqual(snapshot["identity_status"], "unavailable")
        self.assertEqual(snapshot["cleanliness_status"], "unknown")

    def test_missing_git_executable_is_unavailable(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / ".git").mkdir()
            with mock.patch.dict(os.environ, {"PATH": ""}, clear=True):
                snapshot = benchmark_metadata.source_snapshot(root)

        self.assertEqual(snapshot["identity_status"], "unavailable")
        self.assertIsNone(snapshot["revision"])

    def test_ambient_git_redirection_and_config_are_unreadable(self):
        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory)
            requested = parent / "requested"
            redirected = parent / "redirected"
            requested.mkdir()
            redirected.mkdir()
            initialize_git_repository(requested)
            initialize_git_repository(redirected)
            cases = (
                {"GIT_DIR": str(redirected / ".git")},
                {
                    "GIT_CONFIG_COUNT": "1",
                    "GIT_CONFIG_KEY_0": "core.worktree",
                    "GIT_CONFIG_VALUE_0": str(redirected),
                },
            )
            for ambient in cases:
                with self.subTest(ambient=tuple(sorted(ambient))):
                    with mock.patch.dict(
                        os.environ,
                        clean_git_environment(**ambient),
                        clear=True,
                    ):
                        snapshot = benchmark_metadata.source_snapshot(requested)
                    self.assertEqual(snapshot["identity_status"], "unreadable")
                    self.assertIsNone(snapshot["revision"])
                    self.assertEqual(snapshot["cleanliness_status"], "unknown")

    def test_exported_source_identity_requires_revision(self):
        with tempfile.TemporaryDirectory() as directory:
            identity = Path(directory) / "source_identity.json"
            identity.write_text(json.dumps({"branch": "main"}))
            with self.assertRaisesRegex(ValueError, "non-empty revision"):
                benchmark_metadata.source_snapshot(directory, identity)


if __name__ == "__main__":
    unittest.main()
