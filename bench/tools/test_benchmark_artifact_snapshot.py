#!/usr/bin/env python3
# Copyright (C) 2026 Zynum contributors
# SPDX-License-Identifier: LGPL-3.0-or-later

import hashlib
import json
import os
import shutil
import stat
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import benchmark_artifacts


def stat_with_uid(metadata, uid):
    values = list(metadata)
    values[stat.ST_UID] = uid
    return os.stat_result(values)


class BenchmarkArtifactSnapshotTest(unittest.TestCase):
    def make_file(self, root, name="artifact", contents=b"artifact", mode=0o755):
        path = Path(root) / name
        path.write_bytes(contents)
        path.chmod(mode)
        return path

    def test_file_capture_has_public_metadata_and_private_execution_projection(self):
        with tempfile.TemporaryDirectory() as directory:
            source = self.make_file(directory, contents=b"frozen bytes")
            source_identity = (source.stat().st_dev, source.stat().st_ino)
            snapshot = benchmark_artifacts.ArtifactSnapshotSet.capture(
                [benchmark_artifacts.ArtifactRequest.binary("probe", source)],
                private_parent=directory,
            )
            artifact = snapshot.artifacts[0]
            execution_path = Path(artifact.execution_path)
            private_root = execution_path.parent
            try:
                self.assertEqual(
                    artifact.metadata_record(),
                    {
                        "name": "probe",
                        "path": str(source),
                        "sha256": hashlib.sha256(b"frozen bytes").hexdigest(),
                    },
                )
                self.assertEqual(artifact.legacy_record(), artifact.metadata_record())
                self.assertNotEqual(execution_path, source)
                self.assertEqual(execution_path.read_bytes(), b"frozen bytes")
                self.assertEqual(stat.S_IMODE(private_root.stat().st_mode), 0o700)
                self.assertEqual(stat.S_IMODE(execution_path.stat().st_mode), 0o500)
                self.assertNotEqual(
                    (execution_path.stat().st_dev, execution_path.stat().st_ino),
                    source_identity,
                )
                snapshot.finalize()
                self.assertTrue(snapshot.finalized)
            finally:
                snapshot.close()
            self.assertFalse(private_root.exists())
            arena = private_root.parent
            self.assertEqual(arena.name, f".zynum-cleanup-v2-{os.geteuid()}")
            self.assertEqual(list(arena.iterdir()), [])
            self.assertEqual(snapshot.cleanup_status, "complete")

    def test_private_root_mode_is_set_through_held_descriptor(self):
        with tempfile.TemporaryDirectory() as directory:
            source = self.make_file(directory)
            with (
                mock.patch.object(
                    benchmark_artifacts.os,
                    "chmod",
                    side_effect=NotImplementedError("follow_symlinks is unavailable"),
                ),
                mock.patch.object(
                    benchmark_artifacts.os,
                    "fchmod",
                    wraps=benchmark_artifacts.os.fchmod,
                ) as held_descriptor_chmod,
            ):
                with benchmark_artifacts.ArtifactSnapshotSet.capture(
                    [benchmark_artifacts.ArtifactRequest.binary("probe", source)]
                ) as snapshot:
                    private_root = Path(snapshot.artifacts[0].execution_path).parent
                    self.assertEqual(stat.S_IMODE(private_root.stat().st_mode), 0o700)
                    self.assertIn(
                        (snapshot._root_fd, 0o700),
                        [call.args for call in held_descriptor_chmod.call_args_list],
                    )
                    self.assertIsInstance(
                        snapshot._cleanup_directory,
                        benchmark_artifacts.repository_snapshot.CleanupDirectory,
                    )
                    snapshot.finalize()

    def test_library_copy_is_read_only_and_original_is_never_reused(self):
        with tempfile.TemporaryDirectory() as directory:
            source = self.make_file(directory, contents=b"A", mode=0o644)
            with benchmark_artifacts.ArtifactSnapshotSet(
                [benchmark_artifacts.ArtifactRequest.library("library", source)]
            ) as snapshot:
                artifact = snapshot.artifacts[0]
                execution_path = Path(artifact.execution_path)
                self.assertEqual(stat.S_IMODE(execution_path.stat().st_mode), 0o400)
                source.write_bytes(b"B")
                source.unlink()
                self.assertEqual(execution_path.read_bytes(), b"A")
                self.assertEqual(artifact.sha256, hashlib.sha256(b"A").hexdigest())
                snapshot.finalize()

    def test_interpreter_script_is_a_frozen_binary_without_source_execute_bit(self):
        with tempfile.TemporaryDirectory() as directory:
            source = self.make_file(
                directory,
                name="worker.py",
                contents=b"print('frozen')\n",
                mode=0o644,
            )
            with benchmark_artifacts.ArtifactSnapshotSet(
                [
                    benchmark_artifacts.ArtifactRequest.interpreter_script(
                        "worker", source
                    )
                ]
            ) as snapshot:
                artifact = snapshot.artifacts[0]
                execution_path = Path(artifact.execution_path)
                self.assertEqual(artifact.role, "binary")
                self.assertEqual(
                    stat.S_IMODE(execution_path.stat().st_mode),
                    0o500,
                )
                self.assertEqual(execution_path.read_bytes(), source.read_bytes())
                snapshot.finalize()

            frozen_bytes = b"print('capsule')\n"
            frozen_digest = hashlib.sha256(frozen_bytes).hexdigest()
            benchmark_artifacts._set_frozen_source_resolver(
                lambda path: (os.path.realpath(path), frozen_bytes, frozen_digest)
            )
            try:
                with (
                    mock.patch.object(
                        benchmark_artifacts,
                        "_open_resolved_regular",
                        side_effect=AssertionError("live interpreter path opened"),
                    ),
                    benchmark_artifacts.ArtifactSnapshotSet(
                        [
                            benchmark_artifacts.ArtifactRequest.interpreter_script(
                                "worker", source
                            )
                        ]
                    ) as snapshot,
                ):
                    artifact = snapshot.artifacts[0]
                    self.assertEqual(str(source), artifact.path)
                    self.assertEqual(frozen_digest, artifact.sha256)
                    self.assertEqual(
                        frozen_bytes, Path(artifact.execution_path).read_bytes()
                    )
                    snapshot.finalize()
            finally:
                benchmark_artifacts._set_frozen_source_resolver(None)

    def test_explicit_platform_image_is_the_only_null_hash_exception(self):
        request = benchmark_artifacts.ArtifactRequest.platform_image(
            "Accelerate", benchmark_artifacts.DEFAULT_ACCELERATE_IMAGE
        )
        with mock.patch.object(benchmark_artifacts.sys, "platform", "darwin"):
            with benchmark_artifacts.ArtifactSnapshotSet([request]) as snapshot:
                artifact = snapshot.artifacts[0]
                self.assertEqual(
                    artifact.execution_path,
                    benchmark_artifacts.DEFAULT_ACCELERATE_IMAGE,
                )
                self.assertEqual(
                    artifact.metadata_record(),
                    {
                        "name": "Accelerate",
                        "path": benchmark_artifacts.DEFAULT_ACCELERATE_IMAGE,
                        "sha256": None,
                    },
                )
                snapshot.finalize()

            denied = (
                benchmark_artifacts.ArtifactRequest.platform_image(
                    "Other", benchmark_artifacts.DEFAULT_ACCELERATE_IMAGE
                ),
                benchmark_artifacts.ArtifactRequest.platform_image(
                    "Accelerate",
                    "/System/Library/Frameworks/Other.framework/Other",
                ),
                benchmark_artifacts.ArtifactRequest.platform_image(
                    "Accelerate",
                    "/System/Library/Frameworks/Accelerate.framework/"
                    "../../../../private/tmp/unhashed.dylib",
                ),
            )
            for invalid in denied:
                with (
                    self.subTest(invalid=invalid),
                    self.assertRaises(benchmark_artifacts.ArtifactCaptureError),
                ):
                    benchmark_artifacts.ArtifactSnapshotSet([invalid])

        with (
            mock.patch.object(benchmark_artifacts.sys, "platform", "linux"),
            self.assertRaises(benchmark_artifacts.ArtifactCaptureError),
        ):
            benchmark_artifacts.ArtifactSnapshotSet([request])

        with self.assertRaises(benchmark_artifacts.ArtifactCaptureError) as raised:
            benchmark_artifacts.ArtifactSnapshotSet(
                [benchmark_artifacts.ArtifactRequest.library("bad", "libblas.so")]
            )
        self.assertEqual(raised.exception.code, "bare_soname_rejected")

    def test_final_and_parent_symlinks_resolve_with_a_fixed_bound(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            real_parent = root / "real"
            real_parent.mkdir()
            source = self.make_file(real_parent, contents=b"through links", mode=0o644)
            (root / "parent").symlink_to(real_parent, target_is_directory=True)
            (real_parent / "link").symlink_to(source.name)
            requested = root / "parent" / "link"
            with benchmark_artifacts.ArtifactSnapshotSet(
                [benchmark_artifacts.ArtifactRequest.library("library", requested)]
            ) as snapshot:
                self.assertEqual(
                    Path(snapshot.artifacts[0].execution_path).read_bytes(),
                    b"through links",
                )

            loop_a = root / "loop-a"
            loop_b = root / "loop-b"
            loop_a.symlink_to(loop_b.name)
            loop_b.symlink_to(loop_a.name)
            with self.assertRaises(benchmark_artifacts.ArtifactCaptureError) as raised:
                benchmark_artifacts.ArtifactSnapshotSet(
                    [benchmark_artifacts.ArtifactRequest.library("loop", loop_a)],
                    max_symlinks=3,
                )
            self.assertEqual(raised.exception.code, "artifact_symlink_limit")

            parent_failure = root / "parent-fstat-failure"
            parent_failure.mkdir()
            parent_source = self.make_file(
                parent_failure,
                contents=b"parent",
                mode=0o644,
            )
            real_open = benchmark_artifacts.os.open
            real_fstat = benchmark_artifacts.os.fstat
            sentinel = real_open(parent_source, os.O_RDONLY | os.O_CLOEXEC)
            descriptor_count = len(os.listdir("/dev/fd"))
            try:
                for attempt in range(16):
                    opened_parent_descriptors = []

                    def fail_parent_open_fstat(path, *args, **kwargs):
                        descriptor = real_open(path, *args, **kwargs)
                        if path == parent_failure.name:
                            opened_parent_descriptors.append(descriptor)
                        return descriptor

                    def injected_parent_fstat(descriptor):
                        if descriptor in opened_parent_descriptors:
                            raise OSError("injected parent fstat failure")
                        return real_fstat(descriptor)

                    with (
                        self.subTest(parent_fstat_attempt=attempt),
                        mock.patch.object(
                            benchmark_artifacts.os,
                            "open",
                            side_effect=fail_parent_open_fstat,
                        ),
                        mock.patch.object(
                            benchmark_artifacts.os,
                            "fstat",
                            side_effect=injected_parent_fstat,
                        ),
                        self.assertRaises(
                            benchmark_artifacts.ArtifactCaptureError
                        ) as raised,
                    ):
                        benchmark_artifacts._open_resolved_regular(
                            str(parent_source),
                            benchmark_artifacts.DEFAULT_MAX_SYMLINKS,
                        )
                    self.assertEqual(
                        raised.exception.code,
                        "artifact_parent_open_failed",
                    )
                    self.assertEqual(len(opened_parent_descriptors), 1)
                    with self.assertRaises(OSError):
                        real_fstat(opened_parent_descriptors[0])
                    real_fstat(sentinel)
                    self.assertEqual(len(os.listdir("/dev/fd")), descriptor_count)
            finally:
                os.close(sentinel)

            reroot = root / "reroot"
            reroot.symlink_to(source)
            descriptor_count = len(os.listdir("/dev/fd"))
            for attempt in range(16):
                root_opens = 0
                replacement_sentinel = None

                def fail_root_reopen(path, *args, **kwargs):
                    nonlocal root_opens, replacement_sentinel
                    if path == os.sep:
                        root_opens += 1
                        if root_opens == 2:
                            replacement_sentinel = real_open(
                                source,
                                os.O_RDONLY | os.O_CLOEXEC,
                            )
                            raise OSError("injected root reopen failure")
                    return real_open(path, *args, **kwargs)

                with (
                    self.subTest(root_reopen_attempt=attempt),
                    mock.patch.object(
                        benchmark_artifacts.os,
                        "open",
                        side_effect=fail_root_reopen,
                    ),
                    self.assertRaises(
                        benchmark_artifacts.ArtifactCaptureError
                    ) as raised,
                ):
                    benchmark_artifacts._open_resolved_regular(
                        str(reroot),
                        benchmark_artifacts.DEFAULT_MAX_SYMLINKS,
                    )
                self.assertEqual(raised.exception.code, "artifact_root_open_failed")
                self.assertIsNotNone(replacement_sentinel)
                assert replacement_sentinel is not None
                real_fstat(replacement_sentinel)
                os.close(replacement_sentinel)
                self.assertEqual(len(os.listdir("/dev/fd")), descriptor_count)

    def test_special_and_directory_artifacts_fail_without_opening_them(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fifo = root / "fifo"
            os.mkfifo(fifo)
            cases = (fifo, root)
            for path in cases:
                with self.subTest(path=path):
                    with self.assertRaises(
                        benchmark_artifacts.ArtifactCaptureError
                    ) as raised:
                        benchmark_artifacts.ArtifactSnapshotSet(
                            [
                                benchmark_artifacts.ArtifactRequest.library(
                                    "unsafe", path
                                )
                            ]
                        )
                    self.assertEqual(raised.exception.code, "artifact_not_regular")

    def test_parent_symlink_a_to_b_drift_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            parent_a = root / "A"
            parent_b = root / "B"
            parent_a.mkdir()
            parent_b.mkdir()
            self.make_file(parent_a, "library", b"same", 0o644)
            self.make_file(parent_b, "library", b"same", 0o644)
            link = root / "current"
            link.symlink_to(parent_a, target_is_directory=True)
            requested = link / "library"
            real_read = benchmark_artifacts.os.read
            changed = False

            def replace_parent(descriptor, count):
                nonlocal changed
                data = real_read(descriptor, count)
                if data and not changed:
                    changed = True
                    link.unlink()
                    link.symlink_to(parent_b, target_is_directory=True)
                return data

            with (
                mock.patch.object(
                    benchmark_artifacts.os, "read", side_effect=replace_parent
                ),
                self.assertRaises(benchmark_artifacts.ArtifactCaptureError) as raised,
            ):
                benchmark_artifacts.ArtifactSnapshotSet(
                    [benchmark_artifacts.ArtifactRequest.library("library", requested)]
                )
            self.assertEqual(raised.exception.code, "artifact_path_drift")

    def test_source_content_a_to_b_to_a_is_caught_by_pre_post_fstat(self):
        with tempfile.TemporaryDirectory() as directory:
            source = self.make_file(directory, contents=b"A", mode=0o644)
            initial_metadata = source.stat()
            real_read = benchmark_artifacts.os.read
            changed = False

            def mutate_and_restore(descriptor, count):
                nonlocal changed
                if not changed:
                    changed = True
                    with source.open("r+b") as mutable:
                        mutable.write(b"B")
                        mutable.flush()
                        os.fsync(mutable.fileno())
                        mutable.seek(0)
                        mutable.write(b"A")
                        mutable.flush()
                        os.fsync(mutable.fileno())
                    os.utime(
                        source,
                        ns=(
                            initial_metadata.st_atime_ns,
                            initial_metadata.st_mtime_ns + 2_000_000_000,
                        ),
                    )
                return real_read(descriptor, count)

            with (
                mock.patch.object(
                    benchmark_artifacts.os, "read", side_effect=mutate_and_restore
                ),
                self.assertRaises(benchmark_artifacts.ArtifactCaptureError) as raised,
            ):
                benchmark_artifacts.ArtifactSnapshotSet(
                    [benchmark_artifacts.ArtifactRequest.library("library", source)]
                )
            self.assertEqual(raised.exception.code, "artifact_source_drift")

    def test_short_read_and_growth_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            short_source = self.make_file(directory, "short", b"abc", 0o644)
            with (
                mock.patch.object(benchmark_artifacts.os, "read", return_value=b""),
                self.assertRaises(benchmark_artifacts.ArtifactCaptureError) as raised,
            ):
                benchmark_artifacts.ArtifactSnapshotSet(
                    [benchmark_artifacts.ArtifactRequest.library("short", short_source)]
                )
            self.assertEqual(raised.exception.code, "artifact_short_read")

            growth_source = self.make_file(directory, "growth", b"a", 0o644)
            real_read = benchmark_artifacts.os.read
            grew = False

            def grow_after_first_read(descriptor, count):
                nonlocal grew
                data = real_read(descriptor, count)
                if data and not grew:
                    grew = True
                    with growth_source.open("ab") as mutable:
                        mutable.write(b"b")
                        mutable.flush()
                        os.fsync(mutable.fileno())
                return data

            with (
                mock.patch.object(
                    benchmark_artifacts.os, "read", side_effect=grow_after_first_read
                ),
                self.assertRaises(benchmark_artifacts.ArtifactCaptureError) as raised,
            ):
                benchmark_artifacts.ArtifactSnapshotSet(
                    [
                        benchmark_artifacts.ArtifactRequest.library(
                            "growth", growth_source
                        )
                    ]
                )
            self.assertEqual(raised.exception.code, "artifact_growth")

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = self.make_file(root, "capture-race", b"abc", 0o644)
            private_parent = root / "private-parent"
            private_parent.mkdir()
            real_rename = benchmark_artifacts.os.rename
            recreated = []

            def claim_then_recreate(source_leaf, destination_leaf, **kwargs):
                result = real_rename(source_leaf, destination_leaf, **kwargs)
                source_fd = kwargs["src_dir_fd"]
                foreign = os.open(
                    source_leaf,
                    os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | os.O_NOFOLLOW,
                    0o600,
                    dir_fd=source_fd,
                )
                try:
                    os.write(foreign, b"foreign-after-capture-error")
                finally:
                    os.close(foreign)
                recreated.append((source_fd, source_leaf))
                return result

            with (
                mock.patch.object(benchmark_artifacts.os, "read", return_value=b""),
                mock.patch.object(
                    benchmark_artifacts.os,
                    "rename",
                    side_effect=claim_then_recreate,
                ),
                self.assertRaises(benchmark_artifacts.ArtifactCleanupError) as raised,
            ):
                benchmark_artifacts.ArtifactSnapshotSet(
                    [benchmark_artifacts.ArtifactRequest.library("race", source)],
                    private_parent=private_parent,
                )
            self.assertIsInstance(
                raised.exception.__cause__,
                benchmark_artifacts.ArtifactCaptureError,
            )
            self.assertEqual(raised.exception.__cause__.code, "artifact_short_read")
            self.assertEqual(len(recreated), 1)
            arena = private_parent / f".zynum-cleanup-v2-{os.geteuid()}"
            private_roots = list(arena.glob(".zynum-benchmark-artifacts-*"))
            self.assertEqual(len(private_roots), 1)
            recreated_leaf = private_roots[0] / recreated[0][1]
            self.assertEqual(
                recreated_leaf.read_bytes(),
                b"foreign-after-capture-error",
            )
            quarantines = list(arena.glob(".zynum-benchmark-artifact-quarantine-*"))
            self.assertEqual(len(quarantines), 1)
            self.assertEqual(
                (quarantines[0] / recreated[0][1]).read_bytes(),
                b"",
            )
            shutil.rmtree(private_roots[0])

    def test_size_count_and_total_limits_apply_to_unique_sources(self):
        with tempfile.TemporaryDirectory() as directory:
            first = self.make_file(directory, "first", b"12", 0o644)
            second = self.make_file(directory, "second", b"34", 0o644)
            with self.assertRaises(benchmark_artifacts.ArtifactCaptureError) as raised:
                benchmark_artifacts.ArtifactSnapshotSet(
                    [benchmark_artifacts.ArtifactRequest.library("first", first)],
                    max_artifact_bytes=1,
                )
            self.assertEqual(raised.exception.code, "artifact_size_limit")

            with self.assertRaises(benchmark_artifacts.ArtifactCaptureError) as raised:
                benchmark_artifacts.ArtifactSnapshotSet(
                    [
                        benchmark_artifacts.ArtifactRequest.library("first", first),
                        benchmark_artifacts.ArtifactRequest.library("second", second),
                    ],
                    max_artifacts=1,
                )
            self.assertEqual(raised.exception.code, "artifact_count_limit")

            with self.assertRaises(benchmark_artifacts.ArtifactCaptureError) as raised:
                benchmark_artifacts.ArtifactSnapshotSet(
                    [
                        benchmark_artifacts.ArtifactRequest.library("first", first),
                        benchmark_artifacts.ArtifactRequest.library("second", second),
                    ],
                    max_total_bytes=3,
                )
            self.assertEqual(raised.exception.code, "artifact_total_bytes_limit")

            duplicate = benchmark_artifacts.ArtifactRequest.library("first", first)
            with benchmark_artifacts.ArtifactSnapshotSet(
                [duplicate, duplicate], max_artifacts=1, max_total_bytes=2
            ) as snapshot:
                self.assertEqual(len(snapshot.artifacts), 1)

    def test_same_path_multi_role_deduplicates_the_source_read(self):
        with tempfile.TemporaryDirectory() as directory:
            source = self.make_file(directory, contents=b"shared", mode=0o755)
            real_open = benchmark_artifacts._open_resolved_regular
            source_opens = 0

            def count_open(path, limit):
                nonlocal source_opens
                source_opens += 1
                return real_open(path, limit)

            with mock.patch.object(
                benchmark_artifacts,
                "_open_resolved_regular",
                side_effect=count_open,
            ):
                with benchmark_artifacts.ArtifactSnapshotSet(
                    [
                        benchmark_artifacts.ArtifactRequest.binary("probe", source),
                        benchmark_artifacts.ArtifactRequest.library("library", source),
                        benchmark_artifacts.ArtifactRequest.library(
                            "library-alias", source
                        ),
                    ]
                ) as snapshot:
                    binary, library, alias = snapshot.artifacts
                    self.assertEqual(source_opens, 2)  # capture and path-drift check
                    self.assertEqual(binary.sha256, library.sha256)
                    self.assertEqual(library.execution_path, alias.execution_path)
                    self.assertNotEqual(binary.execution_path, library.execution_path)
                    self.assertEqual(
                        stat.S_IMODE(Path(binary.execution_path).stat().st_mode), 0o500
                    )
                    self.assertEqual(
                        stat.S_IMODE(Path(library.execution_path).stat().st_mode), 0o400
                    )

    def test_private_stage_a_to_b_and_restore_is_verified_on_every_lookup(self):
        with tempfile.TemporaryDirectory() as directory:
            source = self.make_file(directory, contents=b"A", mode=0o644)
            snapshot = benchmark_artifacts.ArtifactSnapshotSet(
                [benchmark_artifacts.ArtifactRequest.library("library", source)]
            )
            artifact = snapshot.artifacts[0]
            private = Path(artifact.execution_path)
            original = private.with_name("held-original")
            private.rename(original)
            private.write_bytes(b"B")
            private.chmod(0o400)
            try:
                with self.assertRaises(
                    benchmark_artifacts.ArtifactVerificationError
                ) as raised:
                    _ = artifact.execution_path
                self.assertEqual(raised.exception.code, "private_artifact_drift")
                private.unlink()
                original.rename(private)
                self.assertEqual(Path(artifact.execution_path).read_bytes(), b"A")
                snapshot.finalize()
            finally:
                if original.exists() and not private.exists():
                    original.rename(private)
                snapshot.close()

    def test_private_stage_mode_digest_and_finalize_mutations_fail(self):
        mutations = ("mode", "digest")
        for mutation in mutations:
            with (
                self.subTest(mutation=mutation),
                tempfile.TemporaryDirectory() as directory,
            ):
                source = self.make_file(directory, contents=b"A", mode=0o644)
                snapshot = benchmark_artifacts.ArtifactSnapshotSet(
                    [benchmark_artifacts.ArtifactRequest.library("library", source)]
                )
                artifact = snapshot.artifacts[0]
                private = Path(artifact.execution_path)
                try:
                    if mutation == "mode":
                        private.chmod(0o600)
                    else:
                        private.chmod(0o600)
                        private.write_bytes(b"B")
                        private.chmod(0o400)
                    with self.assertRaises(
                        benchmark_artifacts.ArtifactVerificationError
                    ):
                        snapshot.finalize()
                finally:
                    private.chmod(0o600)
                    private.write_bytes(b"A")
                    private.chmod(0o400)
                    snapshot.close()

        real_fchmod = benchmark_artifacts.os.fchmod
        real_fsync = benchmark_artifacts.os.fsync
        real_fstat = benchmark_artifacts.os.fstat
        real_open = benchmark_artifacts.os.open
        finalize_failures = ("fchmod", "fsync", "writer_fstat", "open", "opened_fstat")
        for failure in finalize_failures:
            for attempt in range(3):
                with (
                    self.subTest(fd_ownership=failure, attempt=attempt),
                    tempfile.TemporaryDirectory() as directory,
                ):
                    root = Path(directory)
                    source = self.make_file(root, "source", b"A", 0o644)
                    private_root = root / "private"
                    private_root.mkdir(mode=0o700)
                    root_fd = real_open(
                        private_root,
                        os.O_RDONLY | os.O_CLOEXEC | os.O_DIRECTORY | os.O_NOFOLLOW,
                    )
                    writer = real_open(
                        "artifact-0000-library",
                        os.O_WRONLY
                        | os.O_CREAT
                        | os.O_EXCL
                        | os.O_CLOEXEC
                        | os.O_NOFOLLOW,
                        0o600,
                        dir_fd=root_fd,
                    )
                    os.write(writer, b"A")
                    reopened = []
                    fstat_calls = 0

                    def injected_fchmod(descriptor, mode):
                        if failure == "fchmod":
                            raise OSError("injected fchmod failure")
                        return real_fchmod(descriptor, mode)

                    def injected_fsync(descriptor):
                        if failure == "fsync":
                            raise OSError("injected fsync failure")
                        return real_fsync(descriptor)

                    def injected_fstat(descriptor):
                        nonlocal fstat_calls
                        fstat_calls += 1
                        if failure == "writer_fstat" and fstat_calls == 1:
                            raise OSError("injected writer fstat failure")
                        if failure == "opened_fstat" and fstat_calls == 2:
                            raise OSError("injected opened fstat failure")
                        return real_fstat(descriptor)

                    def injected_open(*args, **kwargs):
                        if failure == "open":
                            raise OSError("injected open failure")
                        descriptor = real_open(*args, **kwargs)
                        reopened.append(descriptor)
                        return descriptor

                    owner = object.__new__(benchmark_artifacts.ArtifactSnapshotSet)
                    owner._root_fd = root_fd
                    outputs = {"library": writer}
                    try:
                        with (
                            mock.patch.object(
                                benchmark_artifacts.os,
                                "fchmod",
                                side_effect=injected_fchmod,
                            ),
                            mock.patch.object(
                                benchmark_artifacts.os,
                                "fsync",
                                side_effect=injected_fsync,
                            ),
                            mock.patch.object(
                                benchmark_artifacts.os,
                                "fstat",
                                side_effect=injected_fstat,
                            ),
                            mock.patch.object(
                                benchmark_artifacts.os,
                                "open",
                                side_effect=injected_open,
                            ),
                            self.assertRaises(
                                benchmark_artifacts.ArtifactCaptureError
                            ) as raised,
                        ):
                            owner._finish_outputs(
                                outputs,
                                ("library",),
                                str(source),
                                0,
                                str(source),
                                source.stat(),
                                hashlib.sha256(b"A").hexdigest(),
                            )
                        self.assertEqual(
                            raised.exception.code,
                            "private_artifact_finalize_failed",
                        )
                        self.assertEqual(outputs, {})
                        for descriptor in (writer, *reopened):
                            with self.assertRaises(OSError):
                                real_fstat(descriptor)
                    finally:
                        os.close(root_fd)

        with tempfile.TemporaryDirectory() as directory:
            source = self.make_file(directory, contents=b"A", mode=0o644)
            snapshot = benchmark_artifacts.ArtifactSnapshotSet(
                [benchmark_artifacts.ArtifactRequest.library("library", source)]
            )
            artifact = snapshot.artifacts[0]
            try:
                for attempt in range(3):
                    with self.subTest(
                        fd_ownership="verify_opened_fstat", attempt=attempt
                    ):
                        reopened = []
                        fstat_calls = 0

                        def injected_verify_open(*args, **kwargs):
                            descriptor = real_open(*args, **kwargs)
                            reopened.append(descriptor)
                            return descriptor

                        def injected_verify_fstat(descriptor):
                            nonlocal fstat_calls
                            fstat_calls += 1
                            if fstat_calls == 3:
                                raise OSError("injected verification fstat failure")
                            return real_fstat(descriptor)

                        with (
                            mock.patch.object(
                                benchmark_artifacts.os,
                                "open",
                                side_effect=injected_verify_open,
                            ),
                            mock.patch.object(
                                benchmark_artifacts.os,
                                "fstat",
                                side_effect=injected_verify_fstat,
                            ),
                            self.assertRaises(
                                benchmark_artifacts.ArtifactVerificationError
                            ) as raised,
                        ):
                            _ = artifact.execution_path
                        self.assertEqual(
                            raised.exception.code,
                            "private_artifact_unreadable",
                        )
                        self.assertEqual(len(reopened), 1)
                        with self.assertRaises(OSError):
                            real_fstat(reopened[0])
                snapshot.finalize()
            finally:
                snapshot.close()

    def test_source_modes_and_setid_are_rejected(self):
        cases = ((0o666, "artifact_unsafe_mode"), (0o600, "artifact_not_executable"))
        with tempfile.TemporaryDirectory() as directory:
            for mode, expected in cases:
                with self.subTest(mode=oct(mode)):
                    source = self.make_file(directory, str(mode), b"x", mode)
                    request = benchmark_artifacts.ArtifactRequest.binary(
                        "binary", source
                    )
                    with self.assertRaises(
                        benchmark_artifacts.ArtifactCaptureError
                    ) as raised:
                        benchmark_artifacts.ArtifactSnapshotSet([request])
                    self.assertEqual(raised.exception.code, expected)

            setid = self.make_file(directory, "setid", b"x", 0o4755)
            with self.assertRaises(benchmark_artifacts.ArtifactCaptureError) as raised:
                benchmark_artifacts.ArtifactSnapshotSet(
                    [benchmark_artifacts.ArtifactRequest.binary("setid", setid)]
                )
            self.assertEqual(raised.exception.code, "artifact_setid_rejected")

    def test_redaction_removes_private_paths_from_nested_logs_and_json(self):
        with tempfile.TemporaryDirectory() as directory:
            source = self.make_file(directory, contents=b"x", mode=0o644)
            with benchmark_artifacts.ArtifactSnapshotSet(
                [benchmark_artifacts.ArtifactRequest.library("library", source)]
            ) as snapshot:
                artifact = snapshot.artifacts[0]
                private = artifact.execution_path
                payload = {
                    "stdout": "loaded {}".format(private),
                    "argv": [private],
                    private: private.encode(),
                }
                redacted = snapshot.redact_private_paths(payload)
                serialized = json.dumps(
                    redacted,
                    default=lambda value: value.decode("utf-8", errors="strict"),
                )
                self.assertNotIn(private, serialized)
                self.assertIn(str(source), serialized)
                self.assertNotIn(private, repr(artifact))
                self.assertNotIn(private, json.dumps(artifact.metadata_record()))

    def test_cleanup_failure_is_structured_and_never_claims_publication(self):
        real_close = benchmark_artifacts.os.close
        real_fstat = benchmark_artifacts.os.fstat
        real_fsync = benchmark_artifacts.os.fsync

        cleanup = benchmark_artifacts.repository_snapshot
        cleanup_source = Path(cleanup.__file__).read_text(encoding="utf-8")
        self.assertEqual(cleanup_source.count(".close_issue()"), 1)
        self.assertIn("close_issue = arena.close_issue()", cleanup_source)

        with (
            self.subTest(arena_duplicate="failure_preserves_known_identities"),
            tempfile.TemporaryDirectory() as directory,
        ):
            directory_path = Path(directory)
            descriptor = os.open(
                directory,
                os.O_RDONLY | os.O_CLOEXEC | os.O_DIRECTORY | os.O_NOFOLLOW,
            )
            try:
                anchor = cleanup.DirectoryAnchor(descriptor, directory_path)
                arena = cleanup.CleanupArena.open(anchor)
                with (
                    mock.patch.object(
                        cleanup.os,
                        "dup",
                        side_effect=OSError("injected duplicate failure"),
                    ),
                    self.assertRaises(cleanup.CleanupFailure) as raised,
                ):
                    arena.duplicate()
                self.assertEqual(
                    raised.exception.outcome.arena_identity, arena.identity
                )
                self.assertEqual(
                    raised.exception.outcome.recovery_anchor_identity,
                    arena.anchor.identity,
                )
                cleanup.finalize_arena_outcome(
                    arena,
                    cleanup.CleanupOutcome(
                        disposition=cleanup.CleanupDisposition.REMOVED,
                        recovery_paths=(),
                        issues=(),
                    ),
                )
            finally:
                os.close(descriptor)

        def reuse_closed_descriptor(descriptor, path):
            blocker = os.open(path, os.O_RDONLY | os.O_CLOEXEC)
            if blocker != descriptor:
                os.dup2(blocker, descriptor, inheritable=False)
                real_close(blocker)
            return descriptor

        with (
            self.subTest(setup_arena_handoff="success_uses_finalizer"),
            tempfile.TemporaryDirectory() as directory,
        ):
            source = self.make_file(directory, contents=b"A", mode=0o644)
            cleanup = benchmark_artifacts.repository_snapshot
            real_finalize = cleanup.finalize_arena_outcome
            with mock.patch.object(
                cleanup,
                "finalize_arena_outcome",
                wraps=real_finalize,
            ) as finalize_arena:
                snapshot = benchmark_artifacts.ArtifactSnapshotSet(
                    [benchmark_artifacts.ArtifactRequest.library("library", source)],
                    private_parent=directory,
                )
            finalize_arena.assert_called_once()
            self.assertIs(
                finalize_arena.call_args.args[1].disposition,
                cleanup.CleanupDisposition.REMOVED,
            )
            snapshot.close()

        with (
            self.subTest(setup_arena_handoff="failure_uses_finalizer"),
            tempfile.TemporaryDirectory() as directory,
        ):
            source = self.make_file(directory, contents=b"A", mode=0o644)
            cleanup = benchmark_artifacts.repository_snapshot
            real_finalize = cleanup.finalize_arena_outcome

            def fail_root_create(arena, **_kwargs):
                raise cleanup.CleanupFailure(
                    cleanup.CleanupOutcome(
                        disposition=cleanup.CleanupDisposition.RETAINED,
                        recovery_paths=(arena.path,),
                        issues=(
                            cleanup.CleanupIssue("injected_setup_failure", arena.path),
                        ),
                        arena_identity=arena.identity,
                        recovery_anchor_identity=arena.anchor.identity,
                    )
                )

            with (
                mock.patch.object(
                    cleanup.CleanupDirectory,
                    "create",
                    side_effect=fail_root_create,
                ),
                mock.patch.object(
                    cleanup,
                    "finalize_arena_outcome",
                    wraps=real_finalize,
                ) as finalize_arena,
                self.assertRaises(benchmark_artifacts.ArtifactCleanupError),
            ):
                benchmark_artifacts.ArtifactSnapshotSet(
                    [benchmark_artifacts.ArtifactRequest.library("library", source)],
                    private_parent=directory,
                )
            finalize_arena.assert_called_once()
            self.assertIs(
                finalize_arena.call_args.args[1].disposition,
                cleanup.CleanupDisposition.RETAINED,
            )

        with (
            self.subTest(directory_fsync="success"),
            tempfile.TemporaryDirectory() as directory,
        ):
            first_source = self.make_file(directory, "first", contents=b"A", mode=0o644)
            second_source = self.make_file(
                directory, "second", contents=b"B", mode=0o644
            )
            snapshot = benchmark_artifacts.ArtifactSnapshotSet(
                [
                    benchmark_artifacts.ArtifactRequest.library("first", first_source),
                    benchmark_artifacts.ArtifactRequest.library(
                        "second", second_source
                    ),
                ],
                private_parent=directory,
            )
            private_paths = tuple(
                Path(artifact.execution_path) for artifact in snapshot.artifacts
            )
            root_fd = snapshot._root_fd
            arena_fd = snapshot._cleanup_directory.arena.descriptor.fileno()
            private_root = private_paths[0].parent
            arena = private_root.parent
            owned_descriptors = {
                root_fd,
                arena_fd,
                *(copy.descriptor for copy in snapshot._copies.values()),
                *(pending.descriptor for pending in snapshot._pending_copies.values()),
            }
            sync_calls = []
            close_calls = []

            def record_fsync(descriptor):
                sync_calls.append(descriptor)
                return real_fsync(descriptor)

            def record_close(descriptor):
                close_calls.append(descriptor)
                return real_close(descriptor)

            with (
                mock.patch.object(
                    benchmark_artifacts.os,
                    "fsync",
                    side_effect=record_fsync,
                ),
                mock.patch.object(
                    benchmark_artifacts.os,
                    "close",
                    side_effect=record_close,
                ),
            ):
                snapshot.close()
            self.assertEqual(len(sync_calls), 12)
            self.assertEqual(sync_calls.count(root_fd), 3)
            for descriptor in owned_descriptors:
                self.assertIn(descriptor, close_calls)
                with self.assertRaises(OSError):
                    real_fstat(descriptor)
            for private in private_paths:
                self.assertFalse(private.exists())
            self.assertFalse(private_root.exists())
            self.assertEqual(list(arena.iterdir()), [])
            self.assertEqual(snapshot.cleanup_status, "complete")

        fsync_failures = {
            "source": {"private_artifact_claim_source_fsync_failed"},
            "quarantine": {"private_artifact_claim_quarantine_fsync_failed"},
            "both": {
                "private_artifact_claim_quarantine_fsync_failed",
                "private_artifact_claim_source_fsync_failed",
            },
        }
        for failure, expected_issues in fsync_failures.items():
            with (
                self.subTest(directory_fsync=failure),
                tempfile.TemporaryDirectory() as directory,
            ):
                first_source = self.make_file(
                    directory, "first", contents=b"A", mode=0o644
                )
                second_source = self.make_file(
                    directory, "second", contents=b"B", mode=0o644
                )
                snapshot = benchmark_artifacts.ArtifactSnapshotSet(
                    [
                        benchmark_artifacts.ArtifactRequest.library(
                            "first", first_source
                        ),
                        benchmark_artifacts.ArtifactRequest.library(
                            "second", second_source
                        ),
                    ],
                    private_parent=directory,
                )
                private_paths = tuple(
                    Path(artifact.execution_path) for artifact in snapshot.artifacts
                )
                root_fd = snapshot._root_fd
                copies = list(snapshot._copies.values())
                self.assertEqual(len(copies), 2)
                first_copy, second_copy = copies
                owned_descriptors = {
                    root_fd,
                    *(copy.descriptor for copy in snapshot._copies.values()),
                    *(
                        pending.descriptor
                        for pending in snapshot._pending_copies.values()
                    ),
                }
                sync_calls = []
                close_calls = []

                def fail_selected_fsync(descriptor):
                    ordinal = len(sync_calls)
                    sync_calls.append(descriptor)
                    failure_ordinals = {
                        "quarantine": {1},
                        "source": {2},
                        "both": {1, 2},
                    }
                    if ordinal in failure_ordinals[failure]:
                        raise OSError("injected directory fsync failure")
                    return real_fsync(descriptor)

                def record_close(descriptor):
                    close_calls.append(descriptor)
                    return real_close(descriptor)

                try:
                    with (
                        mock.patch.object(
                            benchmark_artifacts.os,
                            "fsync",
                            side_effect=fail_selected_fsync,
                        ),
                        mock.patch.object(
                            benchmark_artifacts.os,
                            "close",
                            side_effect=record_close,
                        ),
                        self.assertRaises(
                            benchmark_artifacts.ArtifactCleanupError
                        ) as raised,
                    ):
                        snapshot.close()
                    error = raised.exception
                    self.assertEqual(error.publication_status, "not_published")
                    self.assertFalse(error.cleanup_complete)
                    claim_fsync_issues = {
                        (issue.code, issue.artifact_id)
                        for issue in error.issues
                        if issue.code.startswith("private_artifact_claim_")
                        and issue.code.endswith("_fsync_failed")
                    }
                    self.assertEqual(
                        claim_fsync_issues,
                        {(issue, first_copy.artifact_id) for issue in expected_issues},
                    )
                    self.assertGreaterEqual(len(sync_calls), 3)
                    for descriptor in owned_descriptors:
                        self.assertIn(descriptor, close_calls)
                        with self.assertRaises(OSError):
                            real_fstat(descriptor)
                    for private in private_paths:
                        self.assertFalse(private.exists())
                    private_root = private_paths[0].parent
                    arena = private_root.parent
                    quarantines = list(
                        arena.glob(".zynum-benchmark-artifact-quarantine-*")
                    )
                    self.assertEqual(len(quarantines), 1)
                    retained = quarantines[0] / first_copy.leaf
                    self.assertEqual(
                        retained.read_bytes(),
                        b"A",
                    )
                    retained_metadata = retained.stat()
                    self.assertEqual(
                        (retained_metadata.st_dev, retained_metadata.st_ino),
                        first_copy.identity,
                    )
                    self.assertFalse((quarantines[0] / second_copy.leaf).exists())
                finally:
                    for residual in Path(directory).glob(".zynum-benchmark-artifact*"):
                        shutil.rmtree(residual)

        for owner_failure, failed_observation in (
            ("initial", 1),
            ("post_configuration", 2),
        ):
            with (
                self.subTest(cleanup_quarantine_owner=owner_failure),
                tempfile.TemporaryDirectory() as directory,
            ):
                first_source = self.make_file(
                    directory, "first", contents=b"A", mode=0o644
                )
                second_source = self.make_file(
                    directory, "second", contents=b"B", mode=0o644
                )
                snapshot = benchmark_artifacts.ArtifactSnapshotSet(
                    [
                        benchmark_artifacts.ArtifactRequest.library(
                            "first", first_source
                        ),
                        benchmark_artifacts.ArtifactRequest.library(
                            "second", second_source
                        ),
                    ],
                    private_parent=directory,
                )
                private_paths = tuple(
                    Path(artifact.execution_path) for artifact in snapshot.artifacts
                )
                root_fd = snapshot._root_fd
                owned_descriptors = {
                    root_fd,
                    *(copy.descriptor for copy in snapshot._copies.values()),
                    *(
                        pending.descriptor
                        for pending in snapshot._pending_copies.values()
                    ),
                }
                real_fstat = benchmark_artifacts.os.fstat
                real_open = benchmark_artifacts.os.open
                real_close = benchmark_artifacts.os.close
                real_unlink = benchmark_artifacts.os.unlink
                real_rmdir = benchmark_artifacts.os.rmdir
                quarantine_descriptor = None
                quarantine_observations = 0
                close_calls = []
                unlink_calls = []
                rmdir_calls = []

                def capture_quarantine_open(path, flags, mode=0o777, *, dir_fd=None):
                    nonlocal quarantine_descriptor
                    descriptor = real_open(path, flags, mode, dir_fd=dir_fd)
                    if ".zynum-benchmark-artifact-quarantine-" in os.fspath(path):
                        quarantine_descriptor = descriptor
                    return descriptor

                def inject_quarantine_owner(descriptor):
                    nonlocal quarantine_observations
                    metadata = real_fstat(descriptor)
                    if descriptor == quarantine_descriptor:
                        quarantine_observations += 1
                        if quarantine_observations == failed_observation:
                            return stat_with_uid(metadata, os.geteuid() + 1)
                    return metadata

                def record_close(descriptor):
                    close_calls.append(descriptor)
                    return real_close(descriptor)

                def record_unlink(path, *args, **kwargs):
                    unlink_calls.append(path)
                    return real_unlink(path, *args, **kwargs)

                def record_rmdir(path, *args, **kwargs):
                    rmdir_calls.append(path)
                    return real_rmdir(path, *args, **kwargs)

                try:
                    with (
                        mock.patch.object(
                            benchmark_artifacts.os,
                            "open",
                            side_effect=capture_quarantine_open,
                        ),
                        mock.patch.object(
                            benchmark_artifacts.os,
                            "fstat",
                            side_effect=inject_quarantine_owner,
                        ),
                        mock.patch.object(
                            benchmark_artifacts.os,
                            "close",
                            side_effect=record_close,
                        ),
                        mock.patch.object(
                            benchmark_artifacts.os,
                            "unlink",
                            side_effect=record_unlink,
                        ),
                        mock.patch.object(
                            benchmark_artifacts.os,
                            "rmdir",
                            side_effect=record_rmdir,
                        ),
                        self.assertRaises(
                            benchmark_artifacts.ArtifactCleanupError
                        ) as raised,
                    ):
                        snapshot.close()
                    error = raised.exception
                    self.assertEqual(error.publication_status, "not_published")
                    self.assertFalse(error.cleanup_complete)
                    self.assertEqual(
                        [(issue.code, issue.artifact_id) for issue in error.issues],
                        [
                            ("cleanup_quarantine_credential_unverified", None),
                            ("private_root_recovery_required", None),
                        ],
                    )
                    self.assertIsNotNone(quarantine_descriptor)
                    assert quarantine_descriptor is not None
                    for descriptor in owned_descriptors | {quarantine_descriptor}:
                        self.assertIn(descriptor, close_calls)
                        with self.assertRaises(OSError):
                            real_fstat(descriptor)
                    self.assertEqual(unlink_calls, [private_paths[1].name])
                    self.assertEqual(private_paths[0].read_bytes(), b"A")
                    self.assertFalse(private_paths[1].exists())
                    private_root = private_paths[0].parent
                    arena = private_root.parent
                    quarantines = list(
                        arena.glob(".zynum-benchmark-artifact-quarantine-*")
                    )
                    self.assertEqual(len(quarantines), 1)
                    self.assertEqual(list(quarantines[0].iterdir()), [])
                    self.assertNotIn(quarantines[0].name, rmdir_calls)
                    self.assertEqual(
                        error.recovery_paths,
                        (str(quarantines[0]), str(private_root)),
                    )
                    self.assertEqual(error.issues[0].recovery_path, str(quarantines[0]))
                    self.assertEqual(error.issues[1].recovery_path, str(private_root))
                finally:
                    for residual in Path(directory).glob(".zynum-benchmark-artifact*"):
                        shutil.rmtree(residual)

        with (
            self.subTest(cleanup_quarantine_setup="open_failed_after_creation"),
            tempfile.TemporaryDirectory() as directory,
        ):
            source = self.make_file(directory, contents=b"A", mode=0o644)
            snapshot = benchmark_artifacts.ArtifactSnapshotSet(
                [benchmark_artifacts.ArtifactRequest.library("library", source)],
                private_parent=directory,
            )
            private = Path(snapshot.artifacts[0].execution_path)
            real_open = benchmark_artifacts.os.open
            real_rmdir = benchmark_artifacts.os.rmdir
            rmdir_calls = []

            def fail_quarantine_open(path, flags, mode=0o777, *, dir_fd=None):
                if ".zynum-benchmark-artifact-quarantine-" in os.fspath(path):
                    raise OSError("injected quarantine open failure")
                return real_open(path, flags, mode, dir_fd=dir_fd)

            def record_rmdir(path, *args, **kwargs):
                rmdir_calls.append(os.fspath(path))
                return real_rmdir(path, *args, **kwargs)

            try:
                with (
                    mock.patch.object(
                        benchmark_artifacts.os,
                        "open",
                        side_effect=fail_quarantine_open,
                    ),
                    mock.patch.object(
                        benchmark_artifacts.os,
                        "rmdir",
                        side_effect=record_rmdir,
                    ),
                    self.assertRaises(
                        benchmark_artifacts.ArtifactCleanupError
                    ) as raised,
                ):
                    snapshot.close()
                error = raised.exception
                arena = private.parent.parent
                quarantines = list(arena.glob(".zynum-benchmark-artifact-quarantine-*"))
                self.assertEqual(len(quarantines), 1)
                quarantine_path = quarantines[0]
                self.assertEqual(
                    [
                        (issue.code, issue.artifact_id, issue.recovery_path)
                        for issue in error.issues
                    ],
                    [
                        (
                            "cleanup_quarantine_create_failed",
                            None,
                            str(quarantine_path),
                        ),
                        (
                            "private_root_recovery_required",
                            None,
                            str(private.parent),
                        ),
                    ],
                )
                self.assertEqual(
                    error.recovery_paths,
                    (str(quarantine_path), str(private.parent)),
                )
                self.assertEqual(error.candidate_paths, (str(private),))
                self.assertNotIn(str(private), error.recovery_paths)
                self.assertEqual(rmdir_calls, [])
                self.assertEqual(private.read_bytes(), b"A")
                self.assertEqual(list(quarantine_path.iterdir()), [])
            finally:
                for residual in Path(directory).glob(".zynum-benchmark-artifact*"):
                    shutil.rmtree(residual)

        with (
            self.subTest(cleanup_outcome="unaddressable_candidate_is_not_recovery"),
            tempfile.TemporaryDirectory() as directory,
        ):
            source = self.make_file(directory, contents=b"A", mode=0o644)
            snapshot = benchmark_artifacts.ArtifactSnapshotSet(
                [benchmark_artifacts.ArtifactRequest.library("library", source)],
                private_parent=directory,
            )
            private = Path(snapshot.artifacts[0].execution_path)
            cleanup = benchmark_artifacts.repository_snapshot
            outcome = cleanup.CleanupOutcome(
                disposition=cleanup.CleanupDisposition.UNADDRESSABLE,
                recovery_paths=(),
                issues=(
                    cleanup.CleanupIssue(
                        "cleanup_arena_binding_unknown",
                        private.parent.parent,
                    ),
                ),
                candidate_paths=(private, private),
                arena_binding=cleanup.ArenaBinding.UNKNOWN,
                public_candidate=cleanup.PublicCandidate.PRESENT,
            )
            with (
                mock.patch.object(
                    cleanup,
                    "claim_and_remove",
                    return_value=outcome,
                ) as shared_cleanup,
                self.assertRaises(benchmark_artifacts.ArtifactCleanupError) as raised,
            ):
                snapshot.close()
            error = raised.exception
            shared_cleanup.assert_called_once()
            self.assertEqual(error.cleanup_status, "unaddressable")
            self.assertEqual(error.candidate_paths, (str(private),))
            self.assertEqual(error.recovery_paths, ())
            self.assertTrue(all(issue.recovery_path is None for issue in error.issues))
            self.assertEqual(private.read_bytes(), b"A")

        with (
            self.subTest(cleanup_outcome="retained_then_unaddressable_clears_recovery"),
            tempfile.TemporaryDirectory() as directory,
        ):
            first_source = self.make_file(directory, "first", contents=b"A", mode=0o644)
            second_source = self.make_file(
                directory, "second", contents=b"B", mode=0o644
            )
            snapshot = benchmark_artifacts.ArtifactSnapshotSet(
                [
                    benchmark_artifacts.ArtifactRequest.library("first", first_source),
                    benchmark_artifacts.ArtifactRequest.library(
                        "second", second_source
                    ),
                ],
                private_parent=directory,
            )
            cleanup = benchmark_artifacts.repository_snapshot
            first_private, second_private = (
                Path(artifact.execution_path) for artifact in snapshot.artifacts
            )
            arena_identity = snapshot._cleanup_directory.arena.identity
            anchor_identity = snapshot._cleanup_directory.arena.anchor.identity
            old_quarantine = first_private.parent.parent / "old-retained-quarantine"
            old_claimed = old_quarantine / first_private.name
            retained = cleanup.CleanupOutcome(
                disposition=cleanup.CleanupDisposition.RETAINED,
                recovery_paths=(old_claimed, old_quarantine),
                issues=(cleanup.CleanupIssue("cleanup_claimed_foreign", old_claimed),),
                arena_identity=arena_identity,
                recovery_anchor_identity=anchor_identity,
            )
            unaddressable = cleanup.CleanupOutcome(
                disposition=cleanup.CleanupDisposition.UNADDRESSABLE,
                recovery_paths=(),
                issues=(
                    cleanup.CleanupIssue(
                        "cleanup_arena_binding_rebound",
                        second_private.parent.parent,
                    ),
                ),
                candidate_paths=(second_private,),
                arena_binding=cleanup.ArenaBinding.REBOUND,
                public_candidate=cleanup.PublicCandidate.PRESENT,
                arena_identity=arena_identity,
                recovery_anchor_identity=anchor_identity,
            )
            real_finalize = cleanup.finalize_arena_outcome
            with (
                mock.patch.object(
                    cleanup,
                    "claim_and_remove",
                    side_effect=(retained, unaddressable),
                ) as shared_cleanup,
                mock.patch.object(
                    cleanup,
                    "finalize_arena_outcome",
                    wraps=real_finalize,
                ) as finalize_arena,
                self.assertRaises(benchmark_artifacts.ArtifactCleanupError) as raised,
            ):
                snapshot.close()
            error = raised.exception
            self.assertEqual(shared_cleanup.call_count, 2)
            finalize_arena.assert_called_once()
            self.assertEqual(error.cleanup_status, "unaddressable")
            self.assertEqual(error.recovery_paths, ())
            self.assertEqual(error.candidate_paths, (str(second_private),))
            self.assertTrue(all(issue.recovery_path is None for issue in error.issues))
            self.assertEqual(first_private.read_bytes(), b"A")
            self.assertEqual(second_private.read_bytes(), b"B")

        with (
            self.subTest(final_binding="retained_root_rebind_after_arena_close"),
            tempfile.TemporaryDirectory() as directory,
        ):
            source = self.make_file(directory, contents=b"A", mode=0o644)
            snapshot = benchmark_artifacts.ArtifactSnapshotSet(
                [benchmark_artifacts.ArtifactRequest.library("library", source)],
                private_parent=directory,
            )
            cleanup = benchmark_artifacts.repository_snapshot
            private = Path(snapshot.artifacts[0].execution_path)
            arena = snapshot._cleanup_directory.arena
            anchor = Path(directory)
            detached_anchor = anchor.with_name(f"{anchor.name}-root-close-detached")
            retained = cleanup.CleanupOutcome(
                disposition=cleanup.CleanupDisposition.RETAINED,
                recovery_paths=(private,),
                issues=(cleanup.CleanupIssue("cleanup_claimed_foreign", private),),
                arena_identity=arena.identity,
                recovery_anchor_identity=arena.anchor.identity,
            )
            real_close_issue = cleanup.CleanupArena.close_issue
            rebound_after_close = False

            def close_then_rebind(candidate):
                nonlocal rebound_after_close
                issue = real_close_issue(candidate)
                if candidate is arena:
                    anchor.rename(detached_anchor)
                    anchor.mkdir(mode=0o700)
                    rebound_after_close = True
                return issue

            try:
                with (
                    mock.patch.object(
                        cleanup,
                        "claim_and_remove",
                        return_value=retained,
                    ),
                    mock.patch.object(
                        cleanup.CleanupArena,
                        "close_issue",
                        new=close_then_rebind,
                    ),
                    self.assertRaises(
                        benchmark_artifacts.ArtifactCleanupError
                    ) as raised,
                ):
                    snapshot.close()
                error = raised.exception
                self.assertTrue(rebound_after_close)
                self.assertEqual(error.cleanup_status, "unaddressable")
                self.assertEqual(error.recovery_paths, ())
                self.assertTrue(
                    all(issue.recovery_path is None for issue in error.issues)
                )
                self.assertIn(
                    "cleanup_arena_binding_rebound",
                    {issue.code for issue in error.issues},
                )
            finally:
                if detached_anchor.exists():
                    shutil.rmtree(detached_anchor)

        with (
            self.subTest(cleanup_quarantine_setup_close="uncertain_after_transfer"),
            tempfile.TemporaryDirectory() as directory,
        ):
            source = self.make_file(directory, contents=b"A", mode=0o644)
            snapshot = benchmark_artifacts.ArtifactSnapshotSet(
                [benchmark_artifacts.ArtifactRequest.library("library", source)],
                private_parent=directory,
            )
            private = Path(snapshot.artifacts[0].execution_path)
            real_open = benchmark_artifacts.os.open
            real_close = benchmark_artifacts.os.close
            real_unlink = benchmark_artifacts.os.unlink
            real_rmdir = benchmark_artifacts.os.rmdir
            quarantine_descriptor = None
            quarantine_close_baseline = None
            reused_descriptor = None
            quarantine_fstats = 0
            close_calls = []
            unlink_calls = []
            rmdir_calls = []

            def capture_quarantine_open(path, flags, mode=0o777, *, dir_fd=None):
                nonlocal quarantine_close_baseline, quarantine_descriptor
                descriptor = real_open(path, flags, mode, dir_fd=dir_fd)
                if ".zynum-benchmark-artifact-quarantine-" in os.fspath(path):
                    quarantine_descriptor = descriptor
                    quarantine_close_baseline = close_calls.count(descriptor)
                return descriptor

            def close_setup_after_transfer(descriptor):
                nonlocal reused_descriptor
                close_calls.append(descriptor)
                if descriptor == quarantine_descriptor and reused_descriptor is None:
                    real_close(descriptor)
                    reused_descriptor = reuse_closed_descriptor(descriptor, source)
                    raise OSError("injected quarantine setup close uncertainty")
                return real_close(descriptor)

            def fail_configured_quarantine_owner(descriptor):
                nonlocal quarantine_fstats
                metadata = real_fstat(descriptor)
                if descriptor == quarantine_descriptor:
                    quarantine_fstats += 1
                    if quarantine_fstats == 2:
                        return stat_with_uid(metadata, os.geteuid() + 1)
                return metadata

            def record_unlink(path, *args, **kwargs):
                unlink_calls.append(path)
                return real_unlink(path, *args, **kwargs)

            def record_rmdir(path, *args, **kwargs):
                rmdir_calls.append(path)
                return real_rmdir(path, *args, **kwargs)

            try:
                with (
                    mock.patch.object(
                        benchmark_artifacts.os,
                        "open",
                        side_effect=capture_quarantine_open,
                    ),
                    mock.patch.object(
                        benchmark_artifacts.os,
                        "close",
                        side_effect=close_setup_after_transfer,
                    ),
                    mock.patch.object(
                        benchmark_artifacts.os,
                        "fstat",
                        side_effect=fail_configured_quarantine_owner,
                    ),
                    mock.patch.object(
                        benchmark_artifacts.os,
                        "unlink",
                        side_effect=record_unlink,
                    ),
                    mock.patch.object(
                        benchmark_artifacts.os,
                        "rmdir",
                        side_effect=record_rmdir,
                    ),
                    self.assertRaises(
                        benchmark_artifacts.ArtifactCleanupError
                    ) as raised,
                ):
                    snapshot.close()
                error = raised.exception
                self.assertEqual(error.publication_status, "not_published")
                self.assertEqual(
                    [issue.code for issue in error.issues],
                    [
                        "cleanup_quarantine_credential_unverified",
                        "cleanup_quarantine_setup_descriptor_close_failed",
                        "private_root_recovery_required",
                    ],
                )
                self.assertIsNotNone(quarantine_descriptor)
                assert quarantine_descriptor is not None
                self.assertIsNotNone(quarantine_close_baseline)
                assert quarantine_close_baseline is not None
                self.assertEqual(
                    close_calls.count(quarantine_descriptor)
                    - quarantine_close_baseline,
                    1,
                )
                self.assertIsNotNone(reused_descriptor)
                assert reused_descriptor is not None
                real_fstat(reused_descriptor)
                quarantines = list(
                    private.parent.parent.glob(".zynum-benchmark-artifact-quarantine-*")
                )
                self.assertEqual(len(quarantines), 1)
                quarantine_path = quarantines[0]
                self.assertEqual(
                    error.recovery_paths,
                    (str(quarantine_path), str(private.parent)),
                )
                self.assertEqual(
                    [issue.recovery_path for issue in error.issues],
                    [str(quarantine_path), None, str(private.parent)],
                )
                self.assertEqual(unlink_calls, [])
                self.assertEqual(rmdir_calls, [])
                self.assertEqual(private.read_bytes(), b"A")
                self.assertEqual(list(quarantine_path.iterdir()), [])
            finally:
                if reused_descriptor is not None:
                    real_close(reused_descriptor)
                for residual in Path(directory).glob(".zynum-benchmark-artifact*"):
                    shutil.rmtree(residual)

        release_point = "finalize"
        with (
            self.subTest(pre_shared_close_uncertainty=release_point),
            tempfile.TemporaryDirectory() as directory,
        ):
            source = self.make_file(directory, contents=b"A", mode=0o644)
            real_dup = benchmark_artifacts.os.dup
            real_read = benchmark_artifacts.os.read
            real_rename = benchmark_artifacts.os.rename
            real_unlink = benchmark_artifacts.os.unlink
            real_rmdir = benchmark_artifacts.os.rmdir
            writer_descriptor = None
            writer_close_baseline = None
            reused_descriptor = None
            close_calls = []
            rename_calls = []
            unlink_calls = []
            rmdir_calls = []

            def capture_writer_dup(descriptor):
                nonlocal writer_close_baseline, writer_descriptor
                writer_descriptor = real_dup(descriptor)
                writer_close_baseline = close_calls.count(writer_descriptor)
                return writer_descriptor

            def selected_read(descriptor, size):
                if release_point == "capture_error":
                    return b""
                return real_read(descriptor, size)

            def fail_writer_close_after_release(descriptor):
                nonlocal reused_descriptor
                close_calls.append(descriptor)
                if descriptor == writer_descriptor and reused_descriptor is None:
                    real_close(descriptor)
                    reused_descriptor = reuse_closed_descriptor(descriptor, source)
                    raise OSError("injected writer descriptor close uncertainty")
                return real_close(descriptor)

            def record_rename(source_name, destination_name, **kwargs):
                rename_calls.append(source_name)
                return real_rename(source_name, destination_name, **kwargs)

            def record_unlink(path, *args, **kwargs):
                unlink_calls.append(path)
                return real_unlink(path, *args, **kwargs)

            def record_rmdir(path, *args, **kwargs):
                rmdir_calls.append(path)
                return real_rmdir(path, *args, **kwargs)

            try:
                with (
                    mock.patch.object(
                        benchmark_artifacts.os,
                        "dup",
                        side_effect=capture_writer_dup,
                    ),
                    mock.patch.object(
                        benchmark_artifacts.os,
                        "read",
                        side_effect=selected_read,
                    ),
                    mock.patch.object(
                        benchmark_artifacts.os,
                        "close",
                        side_effect=fail_writer_close_after_release,
                    ),
                    mock.patch.object(
                        benchmark_artifacts.os,
                        "rename",
                        side_effect=record_rename,
                    ),
                    mock.patch.object(
                        benchmark_artifacts.os,
                        "unlink",
                        side_effect=record_unlink,
                    ),
                    mock.patch.object(
                        benchmark_artifacts.os,
                        "rmdir",
                        side_effect=record_rmdir,
                    ),
                    self.assertRaises(
                        benchmark_artifacts.ArtifactCleanupError
                    ) as raised,
                ):
                    benchmark_artifacts.ArtifactSnapshotSet(
                        [
                            benchmark_artifacts.ArtifactRequest.library(
                                "library", source
                            )
                        ],
                        private_parent=directory,
                    )
                error = raised.exception
                self.assertEqual(error.publication_status, "not_published")
                self.assertIsInstance(
                    error.__cause__, benchmark_artifacts.ArtifactCaptureError
                )
                expected_cause = (
                    "artifact_short_read"
                    if release_point == "capture_error"
                    else "private_artifact_finalize_failed"
                )
                self.assertEqual(error.__cause__.code, expected_cause)
                self.assertIsNotNone(writer_descriptor)
                self.assertIsNotNone(writer_close_baseline)
                assert writer_descriptor is not None
                assert writer_close_baseline is not None
                self.assertEqual(
                    close_calls.count(writer_descriptor) - writer_close_baseline,
                    1,
                )
                self.assertIsNotNone(reused_descriptor)
                assert reused_descriptor is not None
                real_fstat(reused_descriptor)
                arena = Path(directory) / f".zynum-cleanup-v2-{os.geteuid()}"
                roots = list(arena.glob(".zynum-benchmark-artifacts-*"))
                self.assertEqual(len(roots), 1)
                leaf = roots[0] / "artifact-0000-library"
                self.assertEqual(
                    leaf.read_bytes(),
                    b"" if release_point == "capture_error" else b"A",
                )
                self.assertEqual(
                    error.recovery_paths,
                    (str(leaf), str(roots[0])),
                )
                self.assertEqual(rename_calls, [])
                self.assertEqual(unlink_calls, [])
                self.assertEqual(rmdir_calls, [])
                self.assertFalse(
                    list(arena.glob(".zynum-benchmark-artifact-quarantine-*"))
                )
            finally:
                if reused_descriptor is not None:
                    real_close(reused_descriptor)
                for residual in Path(directory).glob(".zynum-benchmark-artifact*"):
                    shutil.rmtree(residual)

        with (
            self.subTest(pre_shared_close_uncertainty="capture_error"),
            tempfile.TemporaryDirectory() as directory,
        ):
            source = self.make_file(directory, contents=b"A", mode=0o644)
            real_dup = benchmark_artifacts.os.dup
            writer_descriptor = None
            writer_close_baseline = None
            reused_descriptor = None
            close_calls = []

            def capture_error_writer_dup(descriptor):
                nonlocal writer_close_baseline, writer_descriptor
                writer_descriptor = real_dup(descriptor)
                writer_close_baseline = close_calls.count(writer_descriptor)
                return writer_descriptor

            def fail_capture_error_writer_close(descriptor):
                nonlocal reused_descriptor
                close_calls.append(descriptor)
                if descriptor == writer_descriptor and reused_descriptor is None:
                    real_close(descriptor)
                    reused_descriptor = reuse_closed_descriptor(descriptor, source)
                    raise OSError("injected capture-error writer close uncertainty")
                return real_close(descriptor)

            try:
                with (
                    mock.patch.object(
                        benchmark_artifacts.os,
                        "dup",
                        side_effect=capture_error_writer_dup,
                    ),
                    mock.patch.object(
                        benchmark_artifacts.os,
                        "read",
                        return_value=b"",
                    ),
                    mock.patch.object(
                        benchmark_artifacts.os,
                        "close",
                        side_effect=fail_capture_error_writer_close,
                    ),
                    mock.patch.object(
                        benchmark_artifacts.repository_snapshot,
                        "claim_and_remove",
                        side_effect=AssertionError(
                            "close-uncertain leaf reached shared cleanup"
                        ),
                    ) as shared_cleanup,
                    self.assertRaises(
                        benchmark_artifacts.ArtifactCleanupError
                    ) as raised,
                ):
                    benchmark_artifacts.ArtifactSnapshotSet(
                        [
                            benchmark_artifacts.ArtifactRequest.library(
                                "library", source
                            )
                        ],
                        private_parent=directory,
                    )
                self.assertEqual(raised.exception.__cause__.code, "artifact_short_read")
                shared_cleanup.assert_not_called()
                self.assertIsNotNone(writer_descriptor)
                self.assertIsNotNone(writer_close_baseline)
                assert writer_descriptor is not None
                assert writer_close_baseline is not None
                self.assertEqual(
                    close_calls.count(writer_descriptor) - writer_close_baseline,
                    1,
                )
                self.assertIsNotNone(reused_descriptor)
                assert reused_descriptor is not None
                real_fstat(reused_descriptor)
                arena = Path(directory) / f".zynum-cleanup-v2-{os.geteuid()}"
                roots = list(arena.glob(".zynum-benchmark-artifacts-*"))
                self.assertEqual(len(roots), 1)
                retained = roots[0] / "artifact-0000-library"
                self.assertEqual(retained.read_bytes(), b"")
                self.assertEqual(
                    raised.exception.recovery_paths,
                    (str(retained), str(roots[0])),
                )
            finally:
                if reused_descriptor is not None:
                    real_close(reused_descriptor)
                for residual in Path(directory).glob(".zynum-benchmark-artifact*"):
                    shutil.rmtree(residual)

        with (
            self.subTest(close_uncertainty="held_copy_before_claim"),
            tempfile.TemporaryDirectory() as directory,
        ):
            first_source = self.make_file(directory, "first", contents=b"A", mode=0o644)
            second_source = self.make_file(
                directory, "second", contents=b"B", mode=0o644
            )
            snapshot = benchmark_artifacts.ArtifactSnapshotSet(
                [
                    benchmark_artifacts.ArtifactRequest.library("first", first_source),
                    benchmark_artifacts.ArtifactRequest.library(
                        "second", second_source
                    ),
                ],
                private_parent=directory,
            )
            private_paths = tuple(
                Path(artifact.execution_path) for artifact in snapshot.artifacts
            )
            first_copy, second_copy = list(snapshot._copies.values())
            close_calls = []
            rename_calls = []
            unlink_calls = []
            reused_descriptor = None
            real_rename = benchmark_artifacts.os.rename
            real_unlink = benchmark_artifacts.os.unlink

            def fail_held_copy_close_after_transfer(descriptor):
                nonlocal reused_descriptor
                close_calls.append(descriptor)
                if descriptor == first_copy.descriptor and reused_descriptor is None:
                    real_close(descriptor)
                    reused_descriptor = reuse_closed_descriptor(
                        descriptor, first_source
                    )
                    raise OSError("injected held-copy close uncertainty")
                return real_close(descriptor)

            def record_rename(source_name, destination_name, **kwargs):
                rename_calls.append(source_name)
                return real_rename(source_name, destination_name, **kwargs)

            def record_unlink(path, *args, **kwargs):
                unlink_calls.append(path)
                return real_unlink(path, *args, **kwargs)

            try:
                with (
                    mock.patch.object(
                        benchmark_artifacts.os,
                        "close",
                        side_effect=fail_held_copy_close_after_transfer,
                    ),
                    mock.patch.object(
                        benchmark_artifacts.os,
                        "rename",
                        side_effect=record_rename,
                    ),
                    mock.patch.object(
                        benchmark_artifacts.os,
                        "unlink",
                        side_effect=record_unlink,
                    ),
                    self.assertRaises(
                        benchmark_artifacts.ArtifactCleanupError
                    ) as raised,
                ):
                    snapshot.close()
                error = raised.exception
                self.assertEqual(error.publication_status, "not_published")
                self.assertIn(
                    ("descriptor_close_failed", first_copy.artifact_id),
                    {(issue.code, issue.artifact_id) for issue in error.issues},
                )
                self.assertEqual(close_calls.count(first_copy.descriptor), 1)
                self.assertIsNotNone(reused_descriptor)
                assert reused_descriptor is not None
                real_fstat(reused_descriptor)
                self.assertNotIn(first_copy.leaf, rename_calls)
                self.assertNotIn(first_copy.leaf, unlink_calls)
                self.assertIn(second_copy.leaf, rename_calls)
                self.assertIn(second_copy.leaf, unlink_calls)
                self.assertEqual(private_paths[0].read_bytes(), b"A")
                self.assertFalse(private_paths[1].exists())
                self.assertEqual(
                    error.recovery_paths,
                    (str(private_paths[0]), str(private_paths[0].parent)),
                )
                self.assertFalse(
                    list(
                        private_paths[0].parent.parent.glob(
                            ".zynum-benchmark-artifact-quarantine-*"
                        )
                    )
                )
            finally:
                if reused_descriptor is not None:
                    real_close(reused_descriptor)
                for residual in Path(directory).glob(".zynum-benchmark-artifact*"):
                    shutil.rmtree(residual)

        with (
            self.subTest(close_uncertainty="claimed_after_transfer"),
            tempfile.TemporaryDirectory() as directory,
        ):
            first_source = self.make_file(directory, "first", contents=b"A", mode=0o644)
            second_source = self.make_file(
                directory, "second", contents=b"B", mode=0o644
            )
            snapshot = benchmark_artifacts.ArtifactSnapshotSet(
                [
                    benchmark_artifacts.ArtifactRequest.library("first", first_source),
                    benchmark_artifacts.ArtifactRequest.library(
                        "second", second_source
                    ),
                ],
                private_parent=directory,
            )
            private_paths = tuple(
                Path(artifact.execution_path) for artifact in snapshot.artifacts
            )
            root_fd = snapshot._root_fd
            copies = list(snapshot._copies.values())
            first_copy, second_copy = copies
            owned_descriptors = {
                root_fd,
                *(copy.descriptor for copy in snapshot._copies.values()),
                *(pending.descriptor for pending in snapshot._pending_copies.values()),
            }
            sync_calls = []
            close_calls = []
            unlink_calls = []
            failed_claim_descriptor = None
            failed_claim_close_baseline = None
            reuse_blocker = None
            real_open = benchmark_artifacts.os.open
            real_unlink = benchmark_artifacts.os.unlink

            def record_fsync(descriptor):
                sync_calls.append(descriptor)
                return real_fsync(descriptor)

            def capture_claimed_open(path, flags, mode=0o777, *, dir_fd=None):
                nonlocal failed_claim_close_baseline, failed_claim_descriptor
                descriptor = real_open(path, flags, mode, dir_fd=dir_fd)
                if path == first_copy.leaf and dir_fd != root_fd:
                    failed_claim_descriptor = descriptor
                    failed_claim_close_baseline = close_calls.count(descriptor)
                return descriptor

            def fail_first_claimed_close_after_transfer(descriptor):
                nonlocal failed_claim_descriptor, reuse_blocker
                close_calls.append(descriptor)
                if descriptor == failed_claim_descriptor and reuse_blocker is None:
                    real_close(descriptor)
                    reuse_blocker = reuse_closed_descriptor(descriptor, first_source)
                    raise OSError("injected claimed descriptor close uncertainty")
                return real_close(descriptor)

            def record_unlink(path, *args, **kwargs):
                unlink_calls.append(path)
                return real_unlink(path, *args, **kwargs)

            try:
                with (
                    mock.patch.object(
                        benchmark_artifacts.os,
                        "fsync",
                        side_effect=record_fsync,
                    ),
                    mock.patch.object(
                        benchmark_artifacts.os,
                        "open",
                        side_effect=capture_claimed_open,
                    ),
                    mock.patch.object(
                        benchmark_artifacts.os,
                        "close",
                        side_effect=fail_first_claimed_close_after_transfer,
                    ),
                    mock.patch.object(
                        benchmark_artifacts.os,
                        "unlink",
                        side_effect=record_unlink,
                    ),
                    self.assertRaises(
                        benchmark_artifacts.ArtifactCleanupError
                    ) as raised,
                ):
                    snapshot.close()
                error = raised.exception
                self.assertEqual(error.publication_status, "not_published")
                self.assertFalse(error.cleanup_complete)
                self.assertIn(
                    ("cleanup_descriptor_close_failed", first_copy.artifact_id),
                    {(issue.code, issue.artifact_id) for issue in error.issues},
                )
                self.assertIn(
                    "cleanup_recovery_required",
                    {issue.code for issue in error.issues},
                )
                self.assertGreaterEqual(len(sync_calls), 3)
                self.assertNotIn(first_copy.leaf, unlink_calls)
                self.assertIn(second_copy.leaf, unlink_calls)
                self.assertIsNotNone(failed_claim_descriptor)
                self.assertIsNotNone(failed_claim_close_baseline)
                assert failed_claim_close_baseline is not None
                self.assertEqual(
                    close_calls.count(failed_claim_descriptor)
                    - failed_claim_close_baseline,
                    1,
                )
                self.assertIsNotNone(reuse_blocker)
                assert reuse_blocker is not None
                real_fstat(reuse_blocker)
                for descriptor in owned_descriptors:
                    if descriptor == failed_claim_descriptor:
                        continue
                    self.assertIn(descriptor, close_calls)
                    with self.assertRaises(OSError):
                        real_fstat(descriptor)
                for private in private_paths:
                    self.assertFalse(private.exists())
                quarantines = list(
                    private_paths[0].parent.parent.glob(
                        ".zynum-benchmark-artifact-quarantine-*"
                    )
                )
                self.assertEqual(len(quarantines), 1)
                retained = quarantines[0] / first_copy.leaf
                self.assertEqual(retained.read_bytes(), b"A")
                retained_metadata = retained.stat()
                self.assertEqual(
                    (retained_metadata.st_dev, retained_metadata.st_ino),
                    first_copy.identity,
                )
                self.assertFalse((quarantines[0] / second_copy.leaf).exists())
            finally:
                if reuse_blocker is not None:
                    real_close(reuse_blocker)
                for residual in Path(directory).glob(".zynum-benchmark-artifact*"):
                    shutil.rmtree(residual)

        with (
            self.subTest(close_uncertainty="quarantine_after_transfer"),
            tempfile.TemporaryDirectory() as directory,
        ):
            first_source = self.make_file(directory, "first", contents=b"A", mode=0o644)
            second_source = self.make_file(
                directory, "second", contents=b"B", mode=0o644
            )
            snapshot = benchmark_artifacts.ArtifactSnapshotSet(
                [
                    benchmark_artifacts.ArtifactRequest.library("first", first_source),
                    benchmark_artifacts.ArtifactRequest.library(
                        "second", second_source
                    ),
                ],
                private_parent=directory,
            )
            private_paths = tuple(
                Path(artifact.execution_path) for artifact in snapshot.artifacts
            )
            private_root = private_paths[0].parent
            root_fd = snapshot._root_fd
            owned_descriptors = {
                root_fd,
                *(copy.descriptor for copy in snapshot._copies.values()),
                *(pending.descriptor for pending in snapshot._pending_copies.values()),
            }
            sync_calls = []
            close_calls = []
            post_failure_stat_calls = []
            post_failure_rmdir_calls = []
            quarantine_close_failed = False
            quarantine_descriptor = None
            quarantine_close_baseline = None
            reuse_blocker = None
            real_open = benchmark_artifacts.os.open
            real_stat = benchmark_artifacts.os.stat
            real_rmdir = benchmark_artifacts.os.rmdir

            def record_fsync(descriptor):
                sync_calls.append(descriptor)
                return real_fsync(descriptor)

            def capture_quarantine_open(path, flags, mode=0o777, *, dir_fd=None):
                nonlocal quarantine_close_baseline, quarantine_descriptor
                descriptor = real_open(path, flags, mode, dir_fd=dir_fd)
                if ".zynum-benchmark-artifact-quarantine-" in os.fspath(path):
                    quarantine_descriptor = descriptor
                    quarantine_close_baseline = close_calls.count(descriptor)
                return descriptor

            def fail_quarantine_close_after_transfer(descriptor):
                nonlocal quarantine_close_failed, reuse_blocker
                close_calls.append(descriptor)
                if descriptor == quarantine_descriptor and not quarantine_close_failed:
                    real_close(descriptor)
                    quarantine_close_failed = True
                    reuse_blocker = reuse_closed_descriptor(descriptor, first_source)
                    raise OSError("injected quarantine descriptor close uncertainty")
                return real_close(descriptor)

            def record_post_failure_stat(path, *args, **kwargs):
                if quarantine_close_failed:
                    post_failure_stat_calls.append(os.fspath(path))
                return real_stat(path, *args, **kwargs)

            def record_post_failure_rmdir(path, *args, **kwargs):
                if quarantine_close_failed:
                    post_failure_rmdir_calls.append(os.fspath(path))
                return real_rmdir(path, *args, **kwargs)

            try:
                with (
                    mock.patch.object(
                        benchmark_artifacts.os,
                        "fsync",
                        side_effect=record_fsync,
                    ),
                    mock.patch.object(
                        benchmark_artifacts.os,
                        "open",
                        side_effect=capture_quarantine_open,
                    ),
                    mock.patch.object(
                        benchmark_artifacts.os,
                        "close",
                        side_effect=fail_quarantine_close_after_transfer,
                    ),
                    mock.patch.object(
                        benchmark_artifacts.os,
                        "stat",
                        side_effect=record_post_failure_stat,
                    ),
                    mock.patch.object(
                        benchmark_artifacts.os,
                        "rmdir",
                        side_effect=record_post_failure_rmdir,
                    ),
                    self.assertRaises(
                        benchmark_artifacts.ArtifactCleanupError
                    ) as raised,
                ):
                    snapshot.close()
                error = raised.exception
                self.assertEqual(error.publication_status, "not_published")
                self.assertFalse(error.cleanup_complete)
                self.assertIn(
                    ("quarantine_descriptor_close_failed", None),
                    {(issue.code, issue.artifact_id) for issue in error.issues},
                )
                self.assertNotIn(
                    "cleanup_quarantine_remove_failed",
                    {issue.code for issue in error.issues},
                )
                self.assertIsNotNone(quarantine_descriptor)
                assert quarantine_descriptor is not None
                self.assertTrue(quarantine_close_failed)
                self.assertIsNotNone(quarantine_close_baseline)
                assert quarantine_close_baseline is not None
                self.assertEqual(
                    close_calls.count(quarantine_descriptor)
                    - quarantine_close_baseline,
                    1,
                )
                self.assertIsNotNone(reuse_blocker)
                assert reuse_blocker is not None
                real_fstat(reuse_blocker)
                for descriptor in owned_descriptors:
                    if descriptor == quarantine_descriptor:
                        continue
                    self.assertIn(descriptor, close_calls)
                for private in private_paths:
                    self.assertFalse(private.exists())
                self.assertTrue(private_root.exists())
                quarantines = list(
                    private_root.parent.glob(".zynum-benchmark-artifact-quarantine-*")
                )
                self.assertEqual(len(quarantines), 1)
                quarantine_path = quarantines[0]
                self.assertEqual(list(quarantine_path.iterdir()), [])
                self.assertEqual(
                    error.recovery_paths,
                    (str(quarantine_path), str(private_root)),
                )
                self.assertNotIn(quarantine_path.name, post_failure_stat_calls)
                self.assertNotIn(quarantine_path.name, post_failure_rmdir_calls)
            finally:
                if reuse_blocker is not None:
                    real_close(reuse_blocker)
                for residual in Path(directory).glob(".zynum-benchmark-artifact*"):
                    shutil.rmtree(residual)

        with (
            self.subTest(close_uncertainty="root_after_transfer"),
            tempfile.TemporaryDirectory() as directory,
        ):
            source = self.make_file(directory, contents=b"A", mode=0o644)
            snapshot = benchmark_artifacts.ArtifactSnapshotSet(
                [benchmark_artifacts.ArtifactRequest.library("library", source)],
                private_parent=directory,
            )
            private = Path(snapshot.artifacts[0].execution_path)
            private_root = private.parent
            root_fd = snapshot._root_fd
            close_calls = []
            post_failure_lstat_calls = []
            post_failure_rmdir_calls = []
            root_close_failed = False
            reuse_blocker = None
            real_lstat = benchmark_artifacts.os.lstat
            real_rmdir = benchmark_artifacts.os.rmdir

            def fail_root_close_after_transfer(descriptor):
                nonlocal root_close_failed, reuse_blocker
                close_calls.append(descriptor)
                if descriptor == root_fd and not root_close_failed:
                    real_close(descriptor)
                    root_close_failed = True
                    reuse_blocker = reuse_closed_descriptor(descriptor, source)
                    raise OSError("injected root descriptor close uncertainty")
                return real_close(descriptor)

            def record_root_post_failure_lstat(path, *args, **kwargs):
                if root_close_failed:
                    post_failure_lstat_calls.append(os.fspath(path))
                return real_lstat(path, *args, **kwargs)

            def record_root_post_failure_rmdir(path, *args, **kwargs):
                if root_close_failed:
                    post_failure_rmdir_calls.append(os.fspath(path))
                return real_rmdir(path, *args, **kwargs)

            try:
                with (
                    mock.patch.object(
                        benchmark_artifacts.os,
                        "close",
                        side_effect=fail_root_close_after_transfer,
                    ),
                    mock.patch.object(
                        benchmark_artifacts.os,
                        "lstat",
                        side_effect=record_root_post_failure_lstat,
                    ),
                    mock.patch.object(
                        benchmark_artifacts.os,
                        "rmdir",
                        side_effect=record_root_post_failure_rmdir,
                    ),
                    self.assertRaises(
                        benchmark_artifacts.ArtifactCleanupError
                    ) as raised,
                ):
                    snapshot.close()
                error = raised.exception
                self.assertEqual(error.publication_status, "not_published")
                self.assertIn(
                    (
                        "cleanup_directory_descriptor_close_uncertain",
                        None,
                        str(private_root),
                    ),
                    {
                        (issue.code, issue.artifact_id, issue.recovery_path)
                        for issue in error.issues
                    },
                )
                self.assertEqual(error.recovery_paths, (str(private_root),))
                self.assertIsNone(snapshot._root_fd)
                self.assertTrue(root_close_failed)
                self.assertEqual(close_calls.count(root_fd), 1)
                self.assertIsNotNone(reuse_blocker)
                assert reuse_blocker is not None
                real_fstat(reuse_blocker)
                self.assertTrue(private_root.is_dir())
                self.assertEqual(list(private_root.iterdir()), [])
                self.assertNotIn(str(private_root), post_failure_lstat_calls)
                self.assertNotIn(str(private_root), post_failure_rmdir_calls)
            finally:
                if reuse_blocker is not None:
                    real_close(reuse_blocker)
                for residual in Path(directory).glob(".zynum-benchmark-artifact*"):
                    shutil.rmtree(residual)

        with (
            self.subTest(final_binding="claim_fsync_failures_then_anchor_rebind"),
            tempfile.TemporaryDirectory() as directory,
        ):
            source = self.make_file(directory, contents=b"A", mode=0o644)
            snapshot = benchmark_artifacts.ArtifactSnapshotSet(
                [benchmark_artifacts.ArtifactRequest.library("library", source)],
                private_parent=directory,
            )
            root_fd = snapshot._root_fd
            anchor = Path(directory)
            detached_anchor = anchor.with_name(f"{anchor.name}-claim-fsync-detached")
            real_open = benchmark_artifacts.os.open
            quarantine_fd = None
            destination_fsync_attempts = 0
            source_fsync_attempts = 0
            rebound = False

            def capture_claim_fsync_quarantine(path, flags, mode=0o777, *, dir_fd=None):
                nonlocal quarantine_fd
                descriptor = real_open(path, flags, mode, dir_fd=dir_fd)
                if ".zynum-benchmark-artifact-quarantine-" in os.fspath(path):
                    quarantine_fd = descriptor
                return descriptor

            def fail_both_claim_fsyncs(descriptor):
                nonlocal destination_fsync_attempts, rebound, source_fsync_attempts
                if descriptor == quarantine_fd:
                    destination_fsync_attempts += 1
                    raise OSError("injected claim destination fsync failure")
                if descriptor == root_fd:
                    source_fsync_attempts += 1
                    anchor.rename(detached_anchor)
                    anchor.mkdir(mode=0o700)
                    rebound = True
                    raise OSError("injected claim source fsync failure")
                return real_fsync(descriptor)

            try:
                with (
                    mock.patch.object(
                        benchmark_artifacts.os,
                        "open",
                        side_effect=capture_claim_fsync_quarantine,
                    ),
                    mock.patch.object(
                        benchmark_artifacts.os,
                        "fsync",
                        side_effect=fail_both_claim_fsyncs,
                    ),
                    self.assertRaises(
                        benchmark_artifacts.ArtifactCleanupError
                    ) as raised,
                ):
                    snapshot.close()
                error = raised.exception
                self.assertTrue(rebound)
                self.assertEqual(destination_fsync_attempts, 1)
                self.assertEqual(source_fsync_attempts, 1)
                self.assertEqual(error.cleanup_status, "unaddressable")
                self.assertEqual(error.recovery_paths, ())
                self.assertEqual(error.candidate_paths, ())
                self.assertTrue(
                    {
                        "private_artifact_claim_quarantine_fsync_failed",
                        "private_artifact_claim_source_fsync_failed",
                        "cleanup_arena_binding_rebound",
                    }
                    <= {issue.code for issue in error.issues}
                )
                self.assertEqual(snapshot.cleanup_status, "unaddressable")
            finally:
                if detached_anchor.exists():
                    shutil.rmtree(detached_anchor)

        with (
            self.subTest(final_binding="verifier_retained_then_anchor_rebind"),
            tempfile.TemporaryDirectory() as directory,
        ):
            source = self.make_file(directory, contents=b"A", mode=0o644)
            snapshot = benchmark_artifacts.ArtifactSnapshotSet(
                [benchmark_artifacts.ArtifactRequest.library("library", source)],
                private_parent=directory,
            )
            cleanup = benchmark_artifacts.repository_snapshot
            anchor = Path(directory)
            detached_anchor = anchor.with_name(f"{anchor.name}-verifier-detached")
            real_verify_claimed = cleanup.CleanupQuarantine.verify_claimed
            verifier_calls = 0

            def retain_as_foreign(_target, _descriptor, _claimed_path):
                nonlocal verifier_calls
                verifier_calls += 1
                return cleanup.ClaimVerification.FOREIGN

            def finish_verifier_then_rebind(quarantine, verifier):
                result = real_verify_claimed(quarantine, verifier)
                self.assertIs(result, cleanup.ClaimVerification.FOREIGN)
                anchor.rename(detached_anchor)
                anchor.mkdir(mode=0o700)
                return result

            try:
                with (
                    mock.patch.object(
                        snapshot,
                        "_verify_cleanup_target",
                        side_effect=retain_as_foreign,
                    ),
                    mock.patch.object(
                        cleanup.CleanupQuarantine,
                        "verify_claimed",
                        new=finish_verifier_then_rebind,
                    ),
                    self.assertRaises(
                        benchmark_artifacts.ArtifactCleanupError
                    ) as raised,
                ):
                    snapshot.close()
                error = raised.exception
                self.assertEqual(verifier_calls, 1)
                self.assertEqual(error.cleanup_status, "unaddressable")
                self.assertEqual(error.recovery_paths, ())
                self.assertEqual(error.candidate_paths, ())
                self.assertIn(
                    "private_artifact_replaced",
                    {issue.code for issue in error.issues},
                )
                self.assertIn(
                    "cleanup_arena_binding_rebound",
                    {issue.code for issue in error.issues},
                )
            finally:
                if detached_anchor.exists():
                    shutil.rmtree(detached_anchor)

        with (
            self.subTest(final_binding="child_teardown_then_anchor_rebind"),
            tempfile.TemporaryDirectory() as directory,
        ):
            source = self.make_file(directory, contents=b"A", mode=0o644)
            snapshot = benchmark_artifacts.ArtifactSnapshotSet(
                [benchmark_artifacts.ArtifactRequest.library("library", source)],
                private_parent=directory,
            )
            anchor = Path(directory)
            detached_anchor = anchor.with_name(f"{anchor.name}-teardown-detached")
            real_rmdir = benchmark_artifacts.os.rmdir
            quarantine_removed = False
            rebound_after_arena_fsync = False

            def record_quarantine_removal(path, *args, **kwargs):
                nonlocal quarantine_removed
                result = real_rmdir(path, *args, **kwargs)
                if os.fspath(path).startswith(".zynum-benchmark-artifact-quarantine-"):
                    quarantine_removed = True
                return result

            def rebind_after_removed_child_fsync(descriptor):
                nonlocal rebound_after_arena_fsync
                result = real_fsync(descriptor)
                if quarantine_removed and not rebound_after_arena_fsync:
                    anchor.rename(detached_anchor)
                    anchor.mkdir(mode=0o700)
                    rebound_after_arena_fsync = True
                return result

            try:
                with (
                    mock.patch.object(
                        benchmark_artifacts.os,
                        "rmdir",
                        side_effect=record_quarantine_removal,
                    ),
                    mock.patch.object(
                        benchmark_artifacts.os,
                        "fsync",
                        side_effect=rebind_after_removed_child_fsync,
                    ),
                    self.assertRaises(
                        benchmark_artifacts.ArtifactCleanupError
                    ) as raised,
                ):
                    snapshot.close()
                error = raised.exception
                self.assertTrue(quarantine_removed)
                self.assertTrue(rebound_after_arena_fsync)
                self.assertEqual(error.cleanup_status, "unaddressable")
                self.assertEqual(error.recovery_paths, ())
                self.assertNotEqual(snapshot.cleanup_status, "complete")
                self.assertIn(
                    "cleanup_arena_binding_rebound",
                    {issue.code for issue in error.issues},
                )
            finally:
                if detached_anchor.exists():
                    shutil.rmtree(detached_anchor)

        with (
            self.subTest(arena_binding="anchor_sandwich_detects_middle_rebind"),
            tempfile.TemporaryDirectory() as directory,
        ):
            source = self.make_file(directory, contents=b"A", mode=0o644)
            snapshot = benchmark_artifacts.ArtifactSnapshotSet(
                [benchmark_artifacts.ArtifactRequest.library("library", source)],
                private_parent=directory,
            )
            cleanup = benchmark_artifacts.repository_snapshot
            arena = snapshot._cleanup_directory.arena
            anchor = Path(directory)
            detached_anchor = anchor.with_name(f"{anchor.name}-sandwich-detached")
            real_stat = benchmark_artifacts.os.stat
            middle_rebound = False

            def rebind_during_arena_stat(path, *args, **kwargs):
                nonlocal middle_rebound
                result = real_stat(path, *args, **kwargs)
                if (
                    not middle_rebound
                    and path == arena.name
                    and kwargs.get("dir_fd") == arena.anchor.descriptor
                    and kwargs.get("follow_symlinks") is False
                ):
                    anchor.rename(detached_anchor)
                    anchor.mkdir(mode=0o700)
                    middle_rebound = True
                return result

            try:
                with mock.patch.object(
                    benchmark_artifacts.os,
                    "stat",
                    side_effect=rebind_during_arena_stat,
                ):
                    binding = arena.binding()
                self.assertTrue(middle_rebound)
                self.assertIs(binding, cleanup.ArenaBinding.REBOUND)
                with self.assertRaises(
                    benchmark_artifacts.ArtifactCleanupError
                ) as raised:
                    snapshot.close()
                self.assertEqual(raised.exception.cleanup_status, "unaddressable")
                self.assertEqual(raised.exception.recovery_paths, ())
            finally:
                if detached_anchor.exists():
                    shutil.rmtree(detached_anchor)

        with (
            self.subTest(cleanup_directory_finish="arena_fsync_after_child_removed"),
            tempfile.TemporaryDirectory() as directory,
        ):
            source = self.make_file(directory, contents=b"A", mode=0o644)
            snapshot = benchmark_artifacts.ArtifactSnapshotSet(
                [benchmark_artifacts.ArtifactRequest.library("library", source)],
                private_parent=directory,
            )
            private = Path(snapshot.artifacts[0].execution_path)
            private_root = private.parent
            arena = private_root.parent
            arena_fd = snapshot._cleanup_directory.arena.descriptor.fileno()
            arena_fsync_failed = False

            def fail_arena_fsync_after_root_removal(descriptor):
                nonlocal arena_fsync_failed
                if descriptor == arena_fd and not private_root.exists():
                    arena_fsync_failed = True
                    raise OSError("injected arena fsync failure after child removal")
                return real_fsync(descriptor)

            with (
                mock.patch.object(
                    benchmark_artifacts.os,
                    "fsync",
                    side_effect=fail_arena_fsync_after_root_removal,
                ),
                self.assertRaises(benchmark_artifacts.ArtifactCleanupError) as raised,
            ):
                snapshot.close()
            error = raised.exception
            self.assertTrue(arena_fsync_failed)
            self.assertIn(
                "cleanup_arena_fsync_failed",
                {issue.code for issue in error.issues},
            )
            self.assertFalse(private_root.exists())
            self.assertEqual(list(arena.iterdir()), [])
            self.assertEqual(error.recovery_paths, ())
            self.assertEqual(error.candidate_paths, ())
            self.assertEqual(error.cleanup_status, "recovery_required")

        with (
            self.subTest(cleanup_arena_binding="canonical_rebind_is_unaddressable"),
            tempfile.TemporaryDirectory() as directory,
        ):
            source = self.make_file(directory, contents=b"A", mode=0o644)
            snapshot = benchmark_artifacts.ArtifactSnapshotSet(
                [benchmark_artifacts.ArtifactRequest.library("library", source)],
                private_parent=directory,
            )
            private = Path(snapshot.artifacts[0].execution_path)
            private_root = private.parent
            canonical_arena = private_root.parent
            detached_arena = Path(directory) / "detached-owned-arena"
            canonical_arena.rename(detached_arena)
            canonical_arena.mkdir(mode=0o700)
            actual_private = detached_arena / private_root.name / private.name
            real_rename = benchmark_artifacts.os.rename
            real_unlink = benchmark_artifacts.os.unlink
            real_rmdir = benchmark_artifacts.os.rmdir
            rename_calls = []
            unlink_calls = []
            rmdir_calls = []

            def record_rebind_rename(source_name, destination_name, **kwargs):
                rename_calls.append((source_name, destination_name))
                return real_rename(source_name, destination_name, **kwargs)

            def record_rebind_unlink(path, *args, **kwargs):
                unlink_calls.append(os.fspath(path))
                return real_unlink(path, *args, **kwargs)

            def record_rebind_rmdir(path, *args, **kwargs):
                rmdir_calls.append(os.fspath(path))
                return real_rmdir(path, *args, **kwargs)

            try:
                with (
                    mock.patch.object(
                        benchmark_artifacts.os,
                        "rename",
                        side_effect=record_rebind_rename,
                    ),
                    mock.patch.object(
                        benchmark_artifacts.os,
                        "unlink",
                        side_effect=record_rebind_unlink,
                    ),
                    mock.patch.object(
                        benchmark_artifacts.os,
                        "rmdir",
                        side_effect=record_rebind_rmdir,
                    ),
                    self.assertRaises(
                        benchmark_artifacts.ArtifactCleanupError
                    ) as raised,
                ):
                    snapshot.close()
                error = raised.exception
                self.assertEqual(error.cleanup_status, "unaddressable")
                self.assertEqual(snapshot.cleanup_status, "unaddressable")
                self.assertIn(
                    "cleanup_arena_binding_rebound",
                    {issue.code for issue in error.issues},
                )
                self.assertEqual(error.recovery_paths, ())
                self.assertEqual(error.candidate_paths, ())
                self.assertTrue(
                    all(issue.recovery_path is None for issue in error.issues)
                )
                self.assertEqual(rename_calls, [])
                self.assertEqual(unlink_calls, [])
                self.assertEqual(rmdir_calls, [])
                self.assertEqual(actual_private.read_bytes(), b"A")
                self.assertEqual(list(canonical_arena.iterdir()), [])
            finally:
                shutil.rmtree(canonical_arena)
                shutil.rmtree(detached_arena)

        with (
            self.subTest(private_root_public_name="foreign_replacement_retained"),
            tempfile.TemporaryDirectory() as directory,
        ):
            source = self.make_file(directory, contents=b"A", mode=0o644)
            snapshot = benchmark_artifacts.ArtifactSnapshotSet(
                [benchmark_artifacts.ArtifactRequest.library("library", source)],
                private_parent=directory,
            )
            private = Path(snapshot.artifacts[0].execution_path)
            private_root = private.parent
            detached_root = Path(directory) / "detached-owned-root"
            private_root.rename(detached_root)
            private_root.mkdir(mode=0o700)
            foreign_marker = private_root / "foreign"
            foreign_marker.write_bytes(b"foreign-root")
            real_rmdir = benchmark_artifacts.os.rmdir
            rmdir_calls = []

            def record_root_rmdir(path, *args, **kwargs):
                rmdir_calls.append(os.fspath(path))
                return real_rmdir(path, *args, **kwargs)

            try:
                with (
                    mock.patch.object(
                        benchmark_artifacts.os,
                        "rmdir",
                        side_effect=record_root_rmdir,
                    ),
                    self.assertRaises(
                        benchmark_artifacts.ArtifactCleanupError
                    ) as raised,
                ):
                    snapshot.close()
                self.assertEqual(snapshot.cleanup_status, "recovery_required")
                self.assertIn(
                    "cleanup_directory_teardown_failed",
                    {issue.code for issue in raised.exception.issues},
                )
                self.assertEqual(foreign_marker.read_bytes(), b"foreign-root")
                self.assertEqual(list(detached_root.iterdir()), [])
                self.assertNotIn(private_root.name, rmdir_calls)
                self.assertNotIn(detached_root.name, rmdir_calls)
            finally:
                shutil.rmtree(private_root)
                shutil.rmtree(detached_root)

        with tempfile.TemporaryDirectory() as directory:
            source = self.make_file(directory, contents=b"A", mode=0o644)
            snapshot = benchmark_artifacts.ArtifactSnapshotSet(
                [benchmark_artifacts.ArtifactRequest.library("library", source)],
                private_parent=directory,
            )
            private = Path(snapshot.artifacts[0].execution_path)
            private_root = private.parent
            held = private.with_name("held")
            copy_identity = next(iter(snapshot._copies.values())).identity
            real_rename = benchmark_artifacts.os.rename
            real_stat = benchmark_artifacts.os.stat
            replaced_after_stat = False

            def replace_after_successful_stat(source_leaf, destination_leaf, **kwargs):
                nonlocal replaced_after_stat
                source_fd = kwargs["src_dir_fd"]
                observed = real_stat(
                    source_leaf,
                    dir_fd=source_fd,
                    follow_symlinks=False,
                )
                self.assertEqual(
                    (observed.st_dev, observed.st_ino),
                    copy_identity,
                )
                real_rename(
                    source_leaf,
                    held.name,
                    src_dir_fd=source_fd,
                    dst_dir_fd=source_fd,
                )
                foreign = os.open(
                    source_leaf,
                    os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | os.O_NOFOLLOW,
                    0o600,
                    dir_fd=source_fd,
                )
                try:
                    os.write(foreign, b"foreign")
                    os.fchmod(foreign, 0o400)
                finally:
                    os.close(foreign)
                replaced_after_stat = True
                return real_rename(source_leaf, destination_leaf, **kwargs)

            try:
                with (
                    mock.patch.object(
                        benchmark_artifacts.os,
                        "rename",
                        side_effect=replace_after_successful_stat,
                    ),
                    self.assertRaises(
                        benchmark_artifacts.ArtifactCleanupError
                    ) as raised,
                ):
                    snapshot.close()
                error = raised.exception
                self.assertEqual(error.publication_status, "not_published")
                self.assertFalse(error.cleanup_complete)
                self.assertTrue(error.issues)
                self.assertNotIn(str(private_root), str(error))
                self.assertTrue(replaced_after_stat)
                self.assertEqual(held.read_bytes(), b"A")
                quarantines = list(
                    private_root.parent.glob(".zynum-benchmark-artifact-quarantine-*")
                )
                self.assertEqual(len(quarantines), 1)
                self.assertEqual(
                    (quarantines[0] / private.name).read_bytes(),
                    b"foreign",
                )
            finally:
                for residual in Path(directory).glob(".zynum-benchmark-artifact*"):
                    shutil.rmtree(residual)

        with tempfile.TemporaryDirectory() as directory:
            source = self.make_file(directory, contents=b"A", mode=0o644)
            snapshot = benchmark_artifacts.ArtifactSnapshotSet(
                [benchmark_artifacts.ArtifactRequest.library("library", source)],
                private_parent=directory,
            )
            private = Path(snapshot.artifacts[0].execution_path)
            private_root = private.parent
            root_fd = snapshot._root_fd
            real_stat = benchmark_artifacts.os.stat
            recreated = False

            def recreate_public_name_after_claim_stat(path, *args, **kwargs):
                nonlocal recreated
                result = real_stat(path, *args, **kwargs)
                if (
                    not recreated
                    and path == private.name
                    and kwargs.get("dir_fd") != root_fd
                    and kwargs.get("follow_symlinks") is False
                ):
                    recreated = True
                    private.write_bytes(b"foreign-after-stat")
                    private.chmod(0o400)
                return result

            try:
                with (
                    mock.patch.object(
                        benchmark_artifacts.os,
                        "stat",
                        side_effect=recreate_public_name_after_claim_stat,
                    ),
                    self.assertRaises(
                        benchmark_artifacts.ArtifactCleanupError
                    ) as raised,
                ):
                    snapshot.close()
                self.assertTrue(recreated)
                self.assertIn(
                    "private_artifact_name_recreated",
                    {issue.code for issue in raised.exception.issues},
                )
                self.assertEqual(private.read_bytes(), b"foreign-after-stat")
                quarantines = list(
                    private_root.parent.glob(".zynum-benchmark-artifact-quarantine-*")
                )
                self.assertEqual(len(quarantines), 1)
                self.assertEqual((quarantines[0] / private.name).read_bytes(), b"A")
            finally:
                shutil.rmtree(private_root)


if __name__ == "__main__":
    unittest.main()
