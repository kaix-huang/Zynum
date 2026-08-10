#!/usr/bin/env python3
# Copyright (C) 2026 Zynum contributors
# SPDX-License-Identifier: LGPL-3.0-or-later

import dataclasses
import errno
import os
import stat
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

TOOLS_DIR = Path(__file__).resolve().parent
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

import report_publication as publication  # noqa: E402
from report_publication import (  # noqa: E402
    ReportOutput,
    RollbackIndeterminateError,
    TransactionCompleteCleanupError,
    publish_outputs,
)


def _cleanup_arena_name():
    if not publication._platform_support_available():
        return ".zynum-cleanup-v2-unsupported"
    return f".zynum-cleanup-v2-{os.geteuid()}"


CLEANUP_ARENA_NAME = _cleanup_arena_name()


class ReportPublicationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(dir=TOOLS_DIR)
        self.root = Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    def sidecars(self):
        return sorted(self.root.rglob(".report-publish-*"))

    def assert_file(self, path, contents, mode=0o644):
        self.assertEqual(path.read_bytes(), contents)
        self.assertEqual(stat.S_IMODE(path.stat().st_mode), mode)

    def cleanup_recovery_paths(self, exception):
        if isinstance(exception, publication.repository_snapshot.CleanupFailure):
            return exception.outcome.recovery_paths
        return exception.recovery_paths

    def cleanup_candidate_paths(self, exception):
        if isinstance(exception, publication.repository_snapshot.CleanupFailure):
            return exception.outcome.candidate_paths
        return exception.candidate_paths

    def cleanup_issue_text(self, exception):
        if isinstance(exception, publication.repository_snapshot.CleanupFailure):
            return "; ".join(
                f"{issue.code}: {issue.error}" for issue in exception.outcome.issues
            )
        return str(exception)

    def filesystem_names_alias(self, first, second):
        capability = self.root / "name-alias-capability"
        capability.mkdir()
        first_path = capability / first
        second_path = capability / second
        first_path.write_bytes(b"first")
        aliases = False
        try:
            descriptor = os.open(
                second_path,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                0o600,
            )
        except FileExistsError:
            aliases = True
        else:
            os.close(descriptor)
            second_path.unlink()
        first_path.unlink()
        capability.rmdir()
        return aliases

    def set_xattr(self, path, name, value):
        descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
        try:
            library = publication.ctypes.CDLL(None, use_errno=True)
            setter = library.fsetxattr
            setter.restype = publication.ctypes.c_int
            setter.argtypes = [
                publication.ctypes.c_int,
                publication.ctypes.c_char_p,
                publication.ctypes.c_void_p,
                publication.ctypes.c_size_t,
                publication.ctypes.c_uint32,
                publication.ctypes.c_int,
            ]
            value_buffer = publication.ctypes.create_string_buffer(value)
            result = setter(descriptor, name, value_buffer, len(value), 0, 0)
            if result != 0:
                error = publication.ctypes.get_errno()
                raise OSError(error, os.strerror(error))
        finally:
            os.close(descriptor)

    def test_report_output_is_frozen_and_single_publication_is_exact(self):
        destination = self.root / "report.json"
        destination.write_bytes(b"old")
        destination.chmod(0o600)
        output = ReportOutput(destination, b"new\x00report\n")

        with self.assertRaises(dataclasses.FrozenInstanceError):
            output.contents = b"changed"
        publish_outputs([output])

        self.assert_file(destination, b"new\x00report\n")
        self.assertEqual(self.sidecars(), [])

    def test_multi_output_order_is_deterministic_and_backups_are_copies(self):
        first = self.root / "z.json"
        second = self.root / "a.csv"
        first.write_bytes(b"old-z")
        second.write_bytes(b"old-a")
        replace_order = []
        backup_inodes = []
        real_replace = publication._replace_name
        real_verify = publication._verify_item_precommit

        def record_replace(parent, source, destination):
            if source.endswith(".stage"):
                replace_order.append(destination)
            return real_replace(parent, source, destination)

        def inspect_backup(item):
            real_verify(item)
            if item.backup is not None:
                backup_inodes.append(
                    (item.original.identity.inode, item.backup.identity.inode)
                )

        with (
            mock.patch.object(publication, "_replace_name", record_replace),
            mock.patch.object(publication, "_verify_item_precommit", inspect_backup),
        ):
            publish_outputs(
                [ReportOutput(first, b"new-z"), ReportOutput(second, b"new-a")]
            )

        self.assertEqual(replace_order, ["a.csv", "z.json"])
        self.assertTrue(backup_inodes)
        self.assertTrue(all(old != backup for old, backup in backup_inodes))
        self.assert_file(first, b"new-z")
        self.assert_file(second, b"new-a")
        self.assertEqual(self.sidecars(), [])

    def test_multiple_parents_and_nested_missing_parents_are_supported(self):
        outputs = [
            ReportOutput(self.root / "one" / "deep" / "a.json", b"a"),
            ReportOutput(self.root / "two" / "b.csv", b"b"),
            ReportOutput(self.root / "one" / "c.md", b"c"),
        ]

        publish_outputs(outputs)

        for output in outputs:
            self.assert_file(output.path, output.contents)
        self.assertEqual(self.sidecars(), [])

    def test_invalid_inputs_are_rejected_before_filesystem_mutation(self):
        missing = self.root / "missing" / "report"
        invalid_cases = (
            [],
            [ReportOutput(Path(), b"value")],
            [ReportOutput(missing, bytearray(b"value"))],
            [ReportOutput(missing, memoryview(b"value"))],
            [ReportOutput(missing, b"one"), ReportOutput(missing, b"two")],
        )
        for outputs in invalid_cases:
            with self.subTest(outputs=outputs):
                with self.assertRaises((TypeError, ValueError)):
                    publish_outputs(outputs)
                self.assertFalse((self.root / "missing").exists())

    def test_existing_hard_link_is_rejected_before_staging(self):
        destination = self.root / "report"
        alias = self.root / "report-alias"
        destination.write_bytes(b"old")
        os.link(destination, alias)

        with self.assertRaisesRegex(OSError, "hard-link topology"):
            publish_outputs([ReportOutput(destination, b"new")])

        self.assert_file(destination, b"old")
        self.assert_file(alias, b"old")
        self.assertEqual(destination.stat().st_ino, alias.stat().st_ino)
        self.assertEqual(self.sidecars(), [])

    def test_existing_ownership_mismatch_is_rejected_before_staging(self):
        destination = self.root / "report"
        destination.write_bytes(b"old")
        actual_uid, actual_gid = publication._current_owner()

        with mock.patch.object(
            publication,
            "_current_owner",
            return_value=(actual_uid + 1, actual_gid),
        ):
            with self.assertRaisesRegex(OSError, "ownership"):
                publish_outputs([ReportOutput(destination, b"new")])

        self.assert_file(destination, b"old")
        self.assertEqual(self.sidecars(), [])

    def test_setgid_parent_rollback_preserves_original_uid_gid(self):
        effective_gid = os.getegid()
        alternate_groups = [group for group in os.getgroups() if group != effective_gid]
        if not alternate_groups:
            self.skipTest("no alternate supplementary group is available")
        parent = self.root / "setgid-parent"
        parent.mkdir()
        os.chown(parent, -1, alternate_groups[0])
        parent.chmod(0o2750)
        inheritance_probe = parent / "inheritance-probe"
        inheritance_probe.write_bytes(b"probe")
        if inheritance_probe.stat().st_gid != alternate_groups[0]:
            inheritance_probe.unlink()
            self.skipTest("filesystem does not apply setgid directory inheritance")
        inheritance_probe.unlink()

        destination = parent / "report"
        destination.write_bytes(b"old")
        os.chown(destination, os.geteuid(), effective_gid)
        original = destination.stat()
        self.assertNotEqual(original.st_gid, parent.stat().st_gid)

        with mock.patch.object(
            publication,
            "_verify_item_published",
            side_effect=OSError("force rollback"),
        ):
            with self.assertRaises(RollbackIndeterminateError) as caught:
                publish_outputs([ReportOutput(destination, b"new")])

        restored = destination.stat()
        self.assertEqual(
            (restored.st_uid, restored.st_gid), (original.st_uid, original.st_gid)
        )
        self.assert_file(destination, b"old")
        backups = list(parent.glob(".report-publish-*.backup"))
        self.assertEqual(len(backups), 1)
        backup = backups[0].stat()
        self.assertEqual(
            (backup.st_uid, backup.st_gid), (original.st_uid, original.st_gid)
        )
        self.assertEqual(backups[0].read_bytes(), b"old")
        self.assertEqual(caught.exception.recovery_paths, tuple(backups))

    def test_backup_fchown_failure_is_precommit_and_reports_prior_recovery(self):
        first = self.root / "a"
        second = self.root / "b"
        first.write_bytes(b"old-a")
        second.write_bytes(b"old-b")
        real_set_ownership = publication._set_backup_ownership
        calls = 0

        def fail_second_fchown(descriptor, uid, gid):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise OSError("injected fchown failure")
            return real_set_ownership(descriptor, uid, gid)

        with mock.patch.object(
            publication,
            "_set_backup_ownership",
            side_effect=fail_second_fchown,
        ):
            with self.assertRaisesRegex(
                RollbackIndeterminateError, "retained recovery paths"
            ) as caught:
                publish_outputs(
                    [ReportOutput(first, b"new-a"), ReportOutput(second, b"new-b")]
                )

        self.assert_file(first, b"old-a")
        self.assert_file(second, b"old-b")
        backups = list(self.root.glob(".report-publish-*.backup"))
        self.assertEqual(len(backups), 1)
        self.assertEqual(backups[0].read_bytes(), b"old-a")
        self.assertEqual(caught.exception.recovery_paths, tuple(backups))

    @unittest.skipUnless(sys.platform == "darwin", "Darwin file flags are unavailable")
    def test_existing_nonzero_file_flags_are_rejected_before_staging(self):
        destination = self.root / "report"
        destination.write_bytes(b"old")
        os.chflags(destination, stat.UF_NODUMP)
        try:
            with self.assertRaisesRegex(OSError, "file flags"):
                publish_outputs([ReportOutput(destination, b"new")])
            self.assert_file(destination, b"old")
            self.assertEqual(self.sidecars(), [])
        finally:
            os.chflags(destination, 0)

    @unittest.skipUnless(sys.platform == "darwin", "Darwin xattrs are unavailable")
    def test_existing_custom_xattr_is_rejected_and_preserved(self):
        destination = self.root / "report"
        destination.write_bytes(b"old")
        custom_name = b"org.zynum.rollback-test"
        custom_value = b"must survive"
        self.set_xattr(destination, custom_name, custom_value)

        with self.assertRaisesRegex(OSError, "extended attributes"):
            publish_outputs([ReportOutput(destination, b"new")])

        self.assert_file(destination, b"old")
        descriptor = os.open(destination, os.O_RDONLY | os.O_NOFOLLOW)
        try:
            observed = dict(publication._record_xattrs(descriptor))
        finally:
            os.close(descriptor)
        self.assertEqual(observed[custom_name], custom_value)
        self.assertEqual(self.sidecars(), [])

    @unittest.skipUnless(sys.platform == "darwin", "Darwin xattrs are unavailable")
    def test_automatic_provenance_xattr_is_exactly_verified_on_backup(self):
        destination = self.root / "report"
        destination.write_bytes(b"old")
        descriptor = os.open(destination, os.O_RDONLY | os.O_NOFOLLOW)
        try:
            original_xattrs = dict(publication._record_xattrs(descriptor))
        finally:
            os.close(descriptor)
        if b"com.apple.provenance" not in original_xattrs:
            self.skipTest("filesystem did not attach automatic provenance")
        real_prepare = publication._prepare_backup

        def verify_backup(item):
            real_prepare(item)
            self.assertEqual(item.backup.xattrs, item.original.xattrs)
            self.assertIn(b"com.apple.provenance", dict(item.backup.xattrs))

        with mock.patch.object(
            publication, "_prepare_backup", side_effect=verify_backup
        ):
            publish_outputs([ReportOutput(destination, b"new")])

        self.assert_file(destination, b"new")
        self.assertEqual(self.sidecars(), [])

    def test_existing_acl_is_rejected_before_staging(self):
        destination = self.root / "report"
        destination.write_bytes(b"old")

        with mock.patch.object(publication, "_descriptor_has_acl", return_value=True):
            with self.assertRaisesRegex(OSError, "ACL"):
                publish_outputs([ReportOutput(destination, b"new")])

        self.assert_file(destination, b"old")
        self.assertEqual(self.sidecars(), [])

    def test_darwin_acl_get_entry_zero_means_an_entry_was_obtained(self):
        class FakeFunction:
            def __init__(self, result):
                self.result = result

            def __call__(self, *_args):
                return self.result

        class FakeLibrary:
            acl_get_fd_np = FakeFunction(1)
            acl_get_entry = FakeFunction(0)
            acl_free = FakeFunction(0)

        with (
            mock.patch.object(publication.sys, "platform", "darwin"),
            mock.patch.object(publication.ctypes, "CDLL", return_value=FakeLibrary()),
        ):
            self.assertTrue(publication._descriptor_has_acl(123))

    def test_case_alias_collision_fails_before_destination_inspection(self):
        if not self.filesystem_names_alias("Report.svg", "report.svg"):
            self.skipTest("filesystem permits case-distinct names")
        parent = self.root / "case-alias"
        parent.mkdir()
        sentinel = parent / "sentinel"
        sentinel.write_bytes(b"unchanged")

        with mock.patch.object(
            publication,
            "_open_regular_name",
            side_effect=AssertionError("destination inspection must not start"),
        ):
            with self.assertRaisesRegex(ValueError, "canonical.*collision"):
                publish_outputs(
                    [
                        ReportOutput(parent / "Report.svg", b"first"),
                        ReportOutput(parent / "report.svg", b"second"),
                    ]
                )

        self.assertEqual(set(parent.iterdir()), {sentinel, parent / CLEANUP_ARENA_NAME})
        self.assertEqual(list((parent / CLEANUP_ARENA_NAME).iterdir()), [])
        self.assertEqual(sentinel.read_bytes(), b"unchanged")

    def test_unicode_normalization_alias_fails_before_destination_inspection(self):
        composed = "\N{LATIN SMALL LETTER E WITH ACUTE}.svg"
        decomposed = "e\N{COMBINING ACUTE ACCENT}.svg"
        if not self.filesystem_names_alias(composed, decomposed):
            self.skipTest("filesystem permits normalization-distinct names")
        parent = self.root / "unicode-alias"
        parent.mkdir()
        sentinel = parent / "sentinel"
        sentinel.write_bytes(b"unchanged")

        with mock.patch.object(
            publication,
            "_open_regular_name",
            side_effect=AssertionError("destination inspection must not start"),
        ):
            with self.assertRaisesRegex(ValueError, "canonical.*collision"):
                publish_outputs(
                    [
                        ReportOutput(parent / composed, b"first"),
                        ReportOutput(parent / decomposed, b"second"),
                    ]
                )

        self.assertEqual(set(parent.iterdir()), {sentinel, parent / CLEANUP_ARENA_NAME})
        self.assertEqual(list((parent / CLEANUP_ARENA_NAME).iterdir()), [])
        self.assertEqual(sentinel.read_bytes(), b"unchanged")

    def test_case_distinct_names_publish_on_case_sensitive_filesystem(self):
        if self.filesystem_names_alias("Report.svg", "report.svg"):
            self.skipTest("filesystem aliases case-distinct names")
        parent = self.root / "case-distinct"

        publish_outputs(
            [
                ReportOutput(parent / "Report.svg", b"first"),
                ReportOutput(parent / "report.svg", b"second"),
            ]
        )

        self.assert_file(parent / "Report.svg", b"first")
        self.assert_file(parent / "report.svg", b"second")
        self.assertEqual(self.sidecars(), [])

    def test_collision_preflight_groups_two_parent_handles_by_identity(self):
        parent_path = self.root / "identity-group"
        parent_path.mkdir()
        first_descriptor = os.open(parent_path, publication._directory_flags())
        second_descriptor = os.open(parent_path, publication._directory_flags())
        first_parent = publication._Parent(parent_path, first_descriptor)
        second_parent = publication._Parent(
            parent_path / ".." / parent_path.name, second_descriptor
        )
        items = [
            publication._PreparedOutput(
                publication._OutputSpec(
                    parent_path / "first", parent_path, "first", b"first"
                ),
                first_parent,
            ),
            publication._PreparedOutput(
                publication._OutputSpec(
                    parent_path / "second", parent_path, "second", b"second"
                ),
                second_parent,
            ),
        ]
        real_create_probe = publication._create_collision_probe
        real_mkdir = publication.os.mkdir
        probe_creation_dir_fds = []

        def record_probe_creation(path, mode=0o777, *, dir_fd=None):
            if isinstance(path, str) and path.endswith(".probe"):
                probe_creation_dir_fds.append(dir_fd)
            return real_mkdir(path, mode, dir_fd=dir_fd)

        try:
            with (
                mock.patch.object(
                    publication,
                    "_create_collision_probe",
                    wraps=real_create_probe,
                ) as create_probe,
                mock.patch.object(
                    publication.os,
                    "mkdir",
                    side_effect=record_probe_creation,
                ),
            ):
                publication._preflight_canonical_collisions(items)
            create_probe.assert_called_once()
        finally:
            os.close(first_descriptor)
            os.close(second_descriptor)
        self.assertEqual(probe_creation_dir_fds, [first_descriptor])
        self.assertEqual(
            list(parent_path.iterdir()), [parent_path / CLEANUP_ARENA_NAME]
        )
        self.assertEqual(list((parent_path / CLEANUP_ARENA_NAME).iterdir()), [])

        for failure in ("probe", "parent"):
            with self.subTest(probe_creation_fsync_failure=failure):
                fsync_parent_path = self.root / f"probe-fsync-{failure}"
                fsync_parent_path.mkdir()
                parent_descriptor = os.open(
                    fsync_parent_path, publication._directory_flags()
                )
                parent = publication._Parent(fsync_parent_path, parent_descriptor)
                token = f"fsync-{failure}"
                probe_path = fsync_parent_path / f".report-publish-{token}.probe"
                events = []
                real_fsync = publication.os.fsync

                def fail_probe_creation_fsync(descriptor):
                    kind = "parent" if descriptor == parent_descriptor else "probe"
                    events.append(kind)
                    if kind == failure:
                        raise OSError(f"injected {kind} creation fsync failure")
                    return real_fsync(descriptor)

                try:
                    with (
                        mock.patch.object(
                            publication.secrets,
                            "token_hex",
                            return_value=token,
                        ),
                        mock.patch.object(
                            publication.os,
                            "fsync",
                            side_effect=fail_probe_creation_fsync,
                        ),
                    ):
                        with self.assertRaisesRegex(
                            RollbackIndeterminateError,
                            f"injected {failure} creation fsync failure",
                        ) as caught:
                            publication._create_collision_probe(parent)
                finally:
                    os.close(parent_descriptor)

                expected_events = (
                    ["probe"] if failure == "probe" else ["probe", "parent"]
                )
                self.assertEqual(events, expected_events)
                self.assertTrue(probe_path.is_dir())
                self.assertEqual(caught.exception.recovery_paths, ())
                self.assertEqual(caught.exception.candidate_paths, (probe_path,))

    def test_probe_cleanup_failure_preserves_probe_without_destination_changes(self):
        parent = self.root / "probe-cleanup"
        parent.mkdir()
        sentinel = parent / "sentinel"
        sentinel.write_bytes(b"unchanged")
        destination = parent / "report"

        with (
            mock.patch.object(
                publication,
                "_remove_probe_file",
                side_effect=OSError("injected probe cleanup failure"),
            ),
            mock.patch.object(
                publication,
                "_open_regular_name",
                side_effect=AssertionError("destination inspection must not start"),
            ),
        ):
            with self.assertRaisesRegex(
                RollbackIndeterminateError, "probe cleanup failed"
            ) as caught:
                publish_outputs([ReportOutput(destination, b"new")])

        self.assertFalse(destination.exists())
        self.assertEqual(sentinel.read_bytes(), b"unchanged")
        probes = list(parent.rglob(".report-publish-*.quarantine/claimed"))
        self.assertEqual(len(probes), 1)
        self.assertEqual(stat.S_IMODE(probes[0].stat().st_mode), 0o700)
        self.assertEqual([path.name for path in probes[0].iterdir()], ["report"])
        self.assertEqual(caught.exception.recovery_paths, tuple(probes))

    def test_probe_cleanup_retention_branches_report_exact_probe_path(self):
        for mode in (
            "leaf",
            "file-fstat",
            "file-close",
            "fsync",
            "close",
            "inspect",
            "rmdir",
            "foreign-swap",
        ):
            with self.subTest(mode=mode):
                parent = self.root / f"probe-cleanup-{mode}"
                parent.mkdir()
                destination = parent / "report"
                parent_descriptor = os.open(parent, publication._directory_flags())
                publication_parent = publication._Parent(parent, parent_descriptor)
                item = publication._PreparedOutput(
                    publication._OutputSpec(
                        destination, parent, destination.name, b"new"
                    ),
                    publication_parent,
                )
                token = f"cleanup-{mode}"
                public_probe = parent / f".report-publish-{token}.probe"
                quarantine = (
                    parent / CLEANUP_ARENA_NAME / f".report-publish-{token}.quarantine"
                )
                claimed = quarantine / "claimed"
                retained = {
                    "leaf": claimed,
                    "file-fstat": claimed,
                    "file-close": claimed,
                    "fsync": public_probe,
                    "close": public_probe,
                    "inspect": quarantine,
                    "rmdir": quarantine,
                    "foreign-swap": claimed,
                }[mode]
                captured_probe = None
                captured_descriptor = None
                leaf_descriptor = None
                reused_descriptor = None
                close_calls = 0
                leaf_fstat_failed = False
                rmdir_calls = 0
                probe_stats = 0
                probe_fsyncs = 0
                real_create = publication._create_collision_probe
                real_create_quarantine = (
                    publication._create_directory_cleanup_quarantine
                )
                real_remove = publication._remove_probe_file
                real_open = publication.os.open
                real_close = publication.os.close
                real_fstat = publication.os.fstat
                real_fsync = publication.os.fsync
                real_stat = publication.os.stat
                real_rmdir = publication.os.rmdir
                detached_probe = parent / f"detached-{token}"

                def capture_probe(probe_parent):
                    nonlocal captured_probe, captured_descriptor
                    probe = real_create(probe_parent)
                    captured_probe = probe
                    captured_descriptor = probe.descriptor
                    return probe

                def capture_leaf_open(path, flags, open_mode=0o777, *, dir_fd=None):
                    nonlocal leaf_descriptor
                    descriptor = real_open(path, flags, open_mode, dir_fd=dir_fd)
                    if path == destination.name and dir_fd == captured_descriptor:
                        leaf_descriptor = descriptor
                    return descriptor

                def maybe_fail_leaf_fstat(descriptor):
                    nonlocal leaf_fstat_failed
                    if (
                        mode == "file-fstat"
                        and descriptor == leaf_descriptor
                        and not leaf_fstat_failed
                    ):
                        leaf_fstat_failed = True
                        raise OSError("injected probe leaf fstat failure")
                    return real_fstat(descriptor)

                def maybe_fail_leaf(probe, file):
                    if mode == "leaf":
                        raise OSError("injected probe leaf cleanup failure")
                    return real_remove(probe, file)

                def maybe_fail_close(descriptor):
                    nonlocal close_calls, reused_descriptor
                    if (
                        mode == "file-close"
                        and descriptor == leaf_descriptor
                        and close_calls == 0
                    ):
                        close_calls += 1
                        real_close(descriptor)
                        reused_descriptor = real_open(os.devnull, os.O_RDONLY)
                        self.assertEqual(reused_descriptor, leaf_descriptor)
                        raise OSError("injected probe leaf close failure")
                    if (
                        mode == "close"
                        and descriptor == captured_descriptor
                        and close_calls == 0
                    ):
                        close_calls += 1
                        self.assertIsNotNone(captured_probe)
                        self.assertTrue(captured_probe.descriptor_owner.close_attempted)
                        real_close(descriptor)
                        raise OSError("injected probe close failure")
                    return real_close(descriptor)

                def maybe_fail_probe_fsync(descriptor):
                    nonlocal probe_fsyncs
                    if descriptor == captured_descriptor:
                        probe_fsyncs += 1
                        if mode == "fsync" and probe_fsyncs == 1:
                            raise OSError("injected probe cleanup fsync failure")
                    return real_fsync(descriptor)

                def maybe_fail_inspection(path, *args, **kwargs):
                    nonlocal probe_stats
                    if path == quarantine.name:
                        probe_stats += 1
                        if mode == "inspect" and probe_stats == 3:
                            raise OSError("injected probe inspection failure")
                    return real_stat(path, *args, **kwargs)

                def maybe_fail_rmdir(path, *, dir_fd=None):
                    nonlocal rmdir_calls
                    if path == quarantine.name:
                        rmdir_calls += 1
                    if mode == "rmdir" and path == quarantine.name:
                        raise OSError("injected probe rmdir failure")
                    return real_rmdir(path, dir_fd=dir_fd)

                def swap_probe_before_shared_claim(source, public_name, suffix):
                    cleanup = real_create_quarantine(source, public_name, suffix)
                    if mode == "foreign-swap":
                        os.rename(public_probe, detached_probe)
                        public_probe.mkdir(mode=0o700)
                        (public_probe / "foreign").write_bytes(b"foreign")
                    return cleanup

                try:
                    with (
                        mock.patch.object(
                            publication.secrets, "token_hex", return_value=token
                        ),
                        mock.patch.object(
                            publication,
                            "_create_collision_probe",
                            side_effect=capture_probe,
                        ),
                        mock.patch.object(
                            publication,
                            "_remove_probe_file",
                            side_effect=maybe_fail_leaf,
                        ),
                        mock.patch.object(
                            publication.os, "open", side_effect=capture_leaf_open
                        ),
                        mock.patch.object(
                            publication.os,
                            "fstat",
                            side_effect=maybe_fail_leaf_fstat,
                        ),
                        mock.patch.object(
                            publication.os,
                            "fsync",
                            side_effect=maybe_fail_probe_fsync,
                        ),
                        mock.patch.object(
                            publication.os, "close", side_effect=maybe_fail_close
                        ),
                        mock.patch.object(
                            publication.os,
                            "stat",
                            side_effect=maybe_fail_inspection,
                        ),
                        mock.patch.object(
                            publication.os, "rmdir", side_effect=maybe_fail_rmdir
                        ),
                        mock.patch.object(
                            publication,
                            "_create_directory_cleanup_quarantine",
                            side_effect=swap_probe_before_shared_claim,
                        ),
                    ):
                        with self.assertRaises(RollbackIndeterminateError) as caught:
                            publication._preflight_canonical_collisions([item])
                finally:
                    real_close(parent_descriptor)
                    if reused_descriptor is not None:
                        real_fstat(reused_descriptor)
                        real_close(reused_descriptor)

                self.assertFalse(destination.exists())
                self.assertTrue(retained.is_dir())
                expected_children = (
                    ["report"]
                    if mode in {"leaf", "file-fstat", "file-close"}
                    else ["foreign"]
                    if mode == "foreign-swap"
                    else []
                )
                self.assertEqual(
                    [path.name for path in retained.iterdir()], expected_children
                )
                if mode in {"fsync", "close"}:
                    self.assertEqual(caught.exception.recovery_paths, ())
                    self.assertEqual(caught.exception.candidate_paths, (retained,))
                else:
                    self.assertEqual(caught.exception.recovery_paths, (retained,))
                if mode == "close":
                    self.assertEqual(close_calls, 1)
                    self.assertEqual(rmdir_calls, 0)
                if mode == "file-close":
                    self.assertEqual(close_calls, 1)
                    self.assertIsNotNone(reused_descriptor)
                if mode == "file-fstat":
                    self.assertTrue(leaf_fstat_failed)
                if mode == "fsync":
                    self.assertEqual(probe_fsyncs, 1)
                if mode == "foreign-swap":
                    self.assertTrue(detached_probe.is_dir())
                    self.assertEqual(list(detached_probe.iterdir()), [])

    def test_symlink_and_special_destinations_are_rejected_without_victim_changes(self):
        victim = self.root / "victim"
        victim.write_bytes(b"victim")
        symlink = self.root / "report"
        symlink.symlink_to(victim)
        with self.assertRaises(OSError):
            publish_outputs([ReportOutput(symlink, b"attack")])
        self.assertEqual(victim.read_bytes(), b"victim")
        self.assertTrue(symlink.is_symlink())

        fifo = self.root / "report.fifo"
        os.mkfifo(fifo)
        with self.assertRaises(OSError):
            publish_outputs([ReportOutput(fifo, b"attack")])
        self.assertTrue(stat.S_ISFIFO(fifo.lstat().st_mode))
        self.assertEqual(self.sidecars(), [])

    def test_symlink_parent_is_rejected_without_victim_changes(self):
        victim = self.root / "victim"
        victim.mkdir()
        link = self.root / "linked"
        link.symlink_to(victim, target_is_directory=True)

        with self.assertRaises(OSError):
            publish_outputs([ReportOutput(link / "report", b"attack")])

        self.assertEqual(list(victim.iterdir()), [])
        self.assertTrue(link.is_symlink())

    def test_unsupported_platform_fails_before_parent_creation(self):
        with mock.patch.object(publication.os, "supports_dir_fd", frozenset()):
            self.assertFalse(publication._platform_support_available())
        with (
            mock.patch.object(publication.os, "name", "nt"),
            mock.patch.object(publication.sys, "platform", "win32"),
        ):
            self.assertFalse(publication._platform_support_available())
        with mock.patch.object(publication.os, "O_NOFOLLOW", None, create=True):
            self.assertFalse(publication._platform_support_available())
        with mock.patch.object(
            publication.os,
            "supports_follow_symlinks",
            frozenset(),
        ):
            self.assertFalse(publication._platform_support_available())

        destination = self.root / "missing" / "report"
        with mock.patch.object(
            publication, "_platform_support_available", return_value=False
        ):
            with self.assertRaises(publication.ReportPublicationError) as caught:
                publish_outputs([ReportOutput(destination, b"value")])
        self.assertEqual(caught.exception.errno, errno.ENOTSUP)
        self.assertIn("POSIX no-follow descriptor-relative APIs", str(caught.exception))
        self.assertFalse(destination.parent.exists())

        with (
            mock.patch.object(
                publication, "_platform_support_available", return_value=False
            ),
            mock.patch.object(
                publication.os,
                "geteuid",
                side_effect=AssertionError("unsupported import touched geteuid"),
                create=True,
            ),
        ):
            self.assertEqual(_cleanup_arena_name(), ".zynum-cleanup-v2-unsupported")

    def test_stage_name_collision_leaves_destination_and_collision_unchanged(self):
        destination = self.root / "report"
        destination.write_bytes(b"old")
        collision = self.root / ".report-publish-collision.stage"
        collision.write_bytes(b"not ours")

        with mock.patch.object(
            publication.secrets, "token_hex", return_value="collision"
        ):
            with self.assertRaisesRegex(OSError, "unique.*stage"):
                publish_outputs([ReportOutput(destination, b"new")])

        self.assert_file(destination, b"old")
        self.assertEqual(collision.read_bytes(), b"not ours")
        self.assertEqual(self.sidecars(), [collision])

    def test_failure_before_replace_rolls_back_and_removes_created_parents(self):
        destination = self.root / "new" / "deep" / "report"
        with mock.patch.object(
            publication,
            "_verify_item_precommit",
            side_effect=OSError("injected precommit failure"),
        ):
            with self.assertRaisesRegex(OSError, "injected precommit"):
                publish_outputs([ReportOutput(destination, b"new")])

        self.assertFalse((self.root / "new").exists())
        self.assertEqual(self.sidecars(), [])

        duplicate_failure_parent = self.root / "dup-failure"
        mkdir_calls = []
        real_mkdir = publication.os.mkdir

        def record_mkdir(path, mode=0o777, *, dir_fd=None):
            mkdir_calls.append((path, dir_fd))
            return real_mkdir(path, mode, dir_fd=dir_fd)

        with (
            mock.patch.object(
                publication,
                "_require_platform_support",
                return_value=None,
            ),
            mock.patch.object(
                publication.os,
                "dup",
                side_effect=OSError("injected parent-anchor duplicate failure"),
            ),
            mock.patch.object(
                publication.os,
                "mkdir",
                side_effect=record_mkdir,
            ),
        ):
            with self.assertRaisesRegex(
                OSError, "injected parent-anchor duplicate failure"
            ):
                publish_outputs(
                    [ReportOutput(duplicate_failure_parent / "report", b"new")]
                )

        self.assertEqual(mkdir_calls, [])
        self.assertFalse(duplicate_failure_parent.exists())
        self.assertEqual(self.sidecars(), [])

    def test_stage_cleanup_claim_preserves_foreign_swap_and_reports_exact_path(self):
        destination = self.root / "report"
        foreign = self.root / "foreign-stage"
        foreign.write_bytes(b"foreign-stage-bytes")

        def swap_stage(parent, name, label):
            if label.startswith("report stage"):
                os.replace(foreign, parent.path / name)

        with (
            mock.patch.object(
                publication,
                "_verify_item_precommit",
                side_effect=OSError("injected precommit failure"),
            ),
            mock.patch.object(
                publication,
                "_before_public_sidecar_claim",
                side_effect=swap_stage,
            ),
        ):
            with self.assertRaisesRegex(
                RollbackIndeterminateError, "foreign bytes"
            ) as caught:
                publish_outputs([ReportOutput(destination, b"owned-stage")])

        self.assertFalse(destination.exists())
        quarantined = list(self.root.rglob(".report-publish-*.quarantine/claimed"))
        self.assertEqual(len(quarantined), 1)
        self.assertEqual(quarantined[0].read_bytes(), b"foreign-stage-bytes")
        self.assertEqual(caught.exception.recovery_paths, tuple(quarantined))
        self.assertTrue(all(path.exists() for path in caught.exception.recovery_paths))

    def test_stage_cleanup_preserves_public_name_reappearing_after_claim(self):
        destination = self.root / "report"

        def recreate_stage(parent, name, label):
            if label.startswith("report stage"):
                (parent.path / name).write_bytes(b"post-claim-foreign")

        with (
            mock.patch.object(
                publication,
                "_verify_item_precommit",
                side_effect=OSError("injected precommit failure"),
            ),
            mock.patch.object(
                publication,
                "_after_public_sidecar_claim",
                side_effect=recreate_stage,
            ),
        ):
            with self.assertRaisesRegex(
                RollbackIndeterminateError, "reappeared after the claim"
            ) as caught:
                publish_outputs([ReportOutput(destination, b"owned-stage")])

        self.assertFalse(destination.exists())
        public_stages = list(self.root.glob(".report-publish-*.stage"))
        self.assertEqual(len(public_stages), 1)
        self.assertEqual(public_stages[0].read_bytes(), b"post-claim-foreign")
        claimed_stages = list(self.root.rglob(".report-publish-*.quarantine/claimed"))
        self.assertEqual(len(claimed_stages), 1)
        self.assertEqual(claimed_stages[0].read_bytes(), b"owned-stage")
        self.assertEqual(
            caught.exception.recovery_paths,
            tuple(claimed_stages),
        )
        self.assertEqual(caught.exception.candidate_paths, tuple(public_stages))

        for mode in ("destination", "source", "both"):
            with self.subTest(claim_fsync_failure=mode):
                parent = self.root / f"stage-claim-fsync-{mode}"
                parent.mkdir()
                destination = parent / "report"
                events = []
                after_claim_called = False
                real_fsync = publication.os.fsync
                claim_started = False
                quarantine_descriptor = None

                real_create = publication._create_rollback_quarantine

                def capture_quarantine(parent_handle, public_name):
                    nonlocal quarantine_descriptor
                    quarantine = real_create(parent_handle, public_name)
                    quarantine_descriptor = quarantine.descriptor.fileno()
                    return quarantine

                def fail_claim_fsync(descriptor):
                    nonlocal claim_started
                    if descriptor == quarantine_descriptor:
                        claim_started = True
                        events.append("destination")
                        if mode in {"destination", "both"}:
                            raise OSError("injected destination claim fsync failure")
                    elif claim_started and descriptor not in {quarantine_descriptor}:
                        events.append("source")
                        if mode in {"source", "both"}:
                            raise OSError("injected source claim fsync failure")
                    return real_fsync(descriptor)

                def record_after_claim(*_args):
                    nonlocal after_claim_called
                    after_claim_called = True

                with (
                    mock.patch.object(
                        publication, "_require_platform_support", return_value=None
                    ),
                    mock.patch.object(
                        publication,
                        "_verify_item_precommit",
                        side_effect=OSError("injected precommit failure"),
                    ),
                    mock.patch.object(
                        publication,
                        "_create_rollback_quarantine",
                        side_effect=capture_quarantine,
                    ),
                    mock.patch.object(
                        publication.os, "fsync", side_effect=fail_claim_fsync
                    ),
                    mock.patch.object(
                        publication,
                        "_after_public_sidecar_claim",
                        side_effect=record_after_claim,
                    ),
                ):
                    with self.assertRaises(RollbackIndeterminateError) as caught:
                        publish_outputs([ReportOutput(destination, b"owned-stage")])

                claimed = list(parent.rglob(".report-publish-*.quarantine/claimed"))
                self.assertEqual(events, ["destination", "source"])
                self.assertFalse(after_claim_called)
                self.assertFalse(destination.exists())
                self.assertEqual(len(claimed), 1)
                self.assertEqual(claimed[0].read_bytes(), b"owned-stage")
                self.assertEqual(caught.exception.recovery_paths, tuple(claimed))
                if mode in {"destination", "both"}:
                    self.assertIn(
                        "destination claim fsync failure", str(caught.exception)
                    )
                if mode in {"source", "both"}:
                    self.assertIn("source claim fsync failure", str(caught.exception))

    def test_stage_cleanup_already_absent_does_not_make_rollback_indeterminate(self):
        destination = self.root / "report"

        def remove_stage(parent, name, label):
            if label.startswith("report stage"):
                os.unlink(name, dir_fd=parent.descriptor)

        with (
            mock.patch.object(
                publication,
                "_verify_item_precommit",
                side_effect=OSError("injected precommit failure"),
            ),
            mock.patch.object(
                publication,
                "_before_public_sidecar_claim",
                side_effect=remove_stage,
            ),
        ):
            with self.assertRaisesRegex(OSError, "injected precommit failure"):
                publish_outputs([ReportOutput(destination, b"owned-stage")])

        self.assertFalse(destination.exists())
        self.assertEqual(self.sidecars(), [])

    def test_claim_file_not_found_interrupted_public_observation_reports_foreign(self):
        destination = self.root / "report"
        stage = self.root / ".report-publish-stage.stage"
        real_stat = publication.os.stat
        real_rename = publication.os.rename
        recreated = False
        interrupted = False

        def recreate_then_report_missing(
            source, destination_name, *, src_dir_fd=None, dst_dir_fd=None
        ):
            nonlocal recreated
            if source != stage.name:
                return real_rename(
                    source,
                    destination_name,
                    src_dir_fd=src_dir_fd,
                    dst_dir_fd=dst_dir_fd,
                )
            os.unlink(source, dir_fd=src_dir_fd)
            descriptor = os.open(
                source,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                0o600,
                dir_fd=src_dir_fd,
            )
            try:
                publication._write_all(descriptor, b"foreign-after-missing-claim")
            finally:
                os.close(descriptor)
            recreated = True
            raise FileNotFoundError("injected claim FileNotFound")

        def interrupt_public_observation(path, *args, **kwargs):
            nonlocal interrupted
            if recreated and not interrupted and path == stage.name:
                interrupted = True
                raise InterruptedError("injected public observation interruption")
            return real_stat(path, *args, **kwargs)

        with (
            mock.patch.object(
                publication, "_require_platform_support", return_value=None
            ),
            mock.patch.object(
                publication.secrets,
                "token_hex",
                side_effect=("probe", "probe-cleanup", "stage", "quarantine"),
            ),
            mock.patch.object(
                publication,
                "_verify_item_precommit",
                side_effect=OSError("injected precommit failure"),
            ),
            mock.patch.object(
                publication.os,
                "rename",
                side_effect=recreate_then_report_missing,
            ),
            mock.patch.object(
                publication.os,
                "stat",
                side_effect=interrupt_public_observation,
            ),
        ):
            with self.assertRaisesRegex(
                RollbackIndeterminateError, "public-name inspection failed"
            ) as caught:
                publish_outputs([ReportOutput(destination, b"owned-stage")])

        self.assertTrue(interrupted)
        self.assertEqual(stage.read_bytes(), b"foreign-after-missing-claim")
        quarantines = list(self.root.rglob(".report-publish-*.quarantine"))
        self.assertEqual(len(quarantines), 1)
        self.assertEqual(
            caught.exception.recovery_paths,
            tuple(quarantines),
        )
        self.assertEqual(caught.exception.candidate_paths, (stage,))
        self.assertIs(
            caught.exception.public_candidate,
            publication.repository_snapshot.PublicCandidate.PRESENT,
        )
        self.assertFalse(destination.exists())

    def test_quarantine_creation_failure_interrupted_public_observation_reports_stage(
        self,
    ):
        destination = self.root / "report"
        stage = self.root / ".report-publish-stage.stage"
        real_stat = publication.os.stat
        creation_failed = False
        interrupted = False

        def fail_quarantine_creation(_parent, _public_name):
            nonlocal creation_failed
            creation_failed = True
            outcome = publication.repository_snapshot.CleanupOutcome(
                publication.repository_snapshot.CleanupDisposition.UNADDRESSABLE,
                (),
                (
                    publication.repository_snapshot.CleanupIssue(
                        "cleanup_quarantine_create_failed",
                        self.root,
                        OSError("injected quarantine creation failure"),
                    ),
                ),
                (stage,),
                publication.repository_snapshot.ArenaBinding.UNKNOWN,
                publication.repository_snapshot.PublicCandidate.UNKNOWN,
            )
            raise publication.repository_snapshot.CleanupFailure(outcome)

        def interrupt_public_observation(path, *args, **kwargs):
            nonlocal interrupted
            if creation_failed and not interrupted and path == stage.name:
                interrupted = True
                raise InterruptedError("injected public observation interruption")
            return real_stat(path, *args, **kwargs)

        with (
            mock.patch.object(
                publication, "_require_platform_support", return_value=None
            ),
            mock.patch.object(
                publication.secrets,
                "token_hex",
                side_effect=("probe", "probe-cleanup", "stage"),
            ),
            mock.patch.object(
                publication,
                "_verify_item_precommit",
                side_effect=OSError("injected precommit failure"),
            ),
            mock.patch.object(
                publication,
                "_create_rollback_quarantine",
                side_effect=fail_quarantine_creation,
            ),
            mock.patch.object(
                publication.os,
                "stat",
                side_effect=interrupt_public_observation,
            ),
        ):
            with self.assertRaisesRegex(
                RollbackIndeterminateError, "quarantine creation failed"
            ) as caught:
                publish_outputs([ReportOutput(destination, b"owned-stage")])

        self.assertFalse(interrupted)
        self.assertEqual(stage.read_bytes(), b"owned-stage")
        self.assertEqual(caught.exception.recovery_paths, ())
        self.assertEqual(caught.exception.candidate_paths, (stage,))
        self.assertIs(
            caught.exception.public_candidate,
            publication.repository_snapshot.PublicCandidate.UNKNOWN,
        )

    def test_post_claim_interrupted_public_observation_reports_recreated_stage(self):
        destination = self.root / "report"
        stage = self.root / ".report-publish-stage.stage"
        real_stat = publication.os.stat
        recreated = False
        interrupted = False

        def recreate_stage(parent, name, label):
            nonlocal recreated
            if label.startswith("report stage"):
                (parent.path / name).write_bytes(b"post-claim-foreign")
                recreated = True

        def interrupt_public_observation(path, *args, **kwargs):
            nonlocal interrupted
            if recreated and not interrupted and path == stage.name:
                interrupted = True
                raise InterruptedError("injected public observation interruption")
            return real_stat(path, *args, **kwargs)

        with (
            mock.patch.object(
                publication, "_require_platform_support", return_value=None
            ),
            mock.patch.object(
                publication.secrets,
                "token_hex",
                side_effect=("probe", "probe-cleanup", "stage", "quarantine"),
            ),
            mock.patch.object(
                publication,
                "_verify_item_precommit",
                side_effect=OSError("injected precommit failure"),
            ),
            mock.patch.object(
                publication,
                "_after_public_sidecar_claim",
                side_effect=recreate_stage,
            ),
            mock.patch.object(
                publication.os,
                "stat",
                side_effect=interrupt_public_observation,
            ),
        ):
            with self.assertRaisesRegex(
                RollbackIndeterminateError, "public-name inspection failed"
            ) as caught:
                publish_outputs([ReportOutput(destination, b"owned-stage")])

        self.assertTrue(interrupted)
        self.assertEqual(stage.read_bytes(), b"post-claim-foreign")
        claimed = list(self.root.rglob(".report-publish-*.quarantine/claimed"))
        self.assertEqual(len(claimed), 1)
        self.assertEqual(claimed[0].read_bytes(), b"owned-stage")
        self.assertEqual(
            caught.exception.recovery_paths,
            tuple(claimed),
        )
        self.assertEqual(caught.exception.candidate_paths, (stage,))

    def test_retained_quarantine_observation_errors_report_claimed_member(self):
        parent = self.root / "retained-quarantine-member"
        parent.mkdir()
        destination = parent / "report"
        foreign = parent / "foreign"
        foreign.write_bytes(b"foreign-claimed")
        quarantine = (
            parent / CLEANUP_ARENA_NAME / ".report-publish-quarantine.quarantine"
        )
        claimed = quarantine / "claimed"
        real_stat = publication.os.stat
        claimed_stats = 0
        interrupted = False
        stage_swapped = False

        def swap_stage(stage_parent, name, label):
            nonlocal stage_swapped
            if label.startswith("report stage"):
                os.replace(foreign, stage_parent.path / name)
                stage_swapped = True

        def interrupt_quarantine_observation(path, *args, **kwargs):
            nonlocal claimed_stats, interrupted
            if stage_swapped and path == "claimed":
                claimed_stats += 1
                if claimed_stats == 2:
                    interrupted = True
                    raise InterruptedError(
                        "injected claimed-member observation interruption"
                    )
            return real_stat(path, *args, **kwargs)

        with (
            mock.patch.object(
                publication, "_require_platform_support", return_value=None
            ),
            mock.patch.object(
                publication.secrets,
                "token_hex",
                side_effect=("probe", "probe-cleanup", "stage", "quarantine"),
            ),
            mock.patch.object(
                publication,
                "_verify_item_precommit",
                side_effect=OSError("injected precommit failure"),
            ),
            mock.patch.object(
                publication,
                "_before_public_sidecar_claim",
                side_effect=swap_stage,
            ),
            mock.patch.object(
                publication.os,
                "stat",
                side_effect=interrupt_quarantine_observation,
            ),
        ):
            with self.assertRaises(RollbackIndeterminateError) as caught:
                publish_outputs([ReportOutput(destination, b"owned-stage")])

        self.assertTrue(interrupted)
        self.assertEqual(claimed.read_bytes(), b"foreign-claimed")
        self.assertEqual(caught.exception.recovery_paths, (claimed,))

        close_parent = self.root / "normal-cleanup-close"
        close_parent.mkdir()
        close_destination = close_parent / "report"
        close_quarantine = (
            close_parent / CLEANUP_ARENA_NAME / ".report-publish-close.quarantine"
        )
        real_create = publication._create_rollback_quarantine
        real_close = publication.os.close
        real_rmdir = publication.os.rmdir
        captured = None
        captured_descriptor = None
        close_calls = 0
        quarantine_rmdirs = 0

        def capture_quarantine(parent_handle, public_name):
            nonlocal captured, captured_descriptor
            captured = real_create(parent_handle, public_name)
            captured_descriptor = captured.descriptor.fileno()
            return captured

        def close_then_fail(descriptor):
            nonlocal close_calls
            if descriptor == captured_descriptor and close_calls == 0:
                close_calls += 1
                self.assertIsNotNone(captured)
                self.assertTrue(captured.descriptor.close_attempted)
                real_close(descriptor)
                raise OSError("injected normal quarantine close failure")
            return real_close(descriptor)

        def record_rmdir(path, *, dir_fd=None):
            nonlocal quarantine_rmdirs
            if path == close_quarantine.name:
                quarantine_rmdirs += 1
            return real_rmdir(path, dir_fd=dir_fd)

        with (
            mock.patch.object(
                publication, "_require_platform_support", return_value=None
            ),
            mock.patch.object(
                publication.secrets,
                "token_hex",
                side_effect=(
                    "probe-close",
                    "probe-cleanup-close",
                    "stage-close",
                    "close",
                ),
            ),
            mock.patch.object(
                publication,
                "_verify_item_precommit",
                side_effect=OSError("injected precommit failure"),
            ),
            mock.patch.object(
                publication,
                "_create_rollback_quarantine",
                side_effect=capture_quarantine,
            ),
            mock.patch.object(publication.os, "close", side_effect=close_then_fail),
            mock.patch.object(publication.os, "rmdir", side_effect=record_rmdir),
        ):
            with self.assertRaisesRegex(
                RollbackIndeterminateError, "quarantine descriptor cleanup failed"
            ) as close_caught:
                publish_outputs([ReportOutput(close_destination, b"owned-stage")])

        self.assertEqual(close_calls, 1)
        self.assertEqual(quarantine_rmdirs, 0)
        self.assertTrue(close_quarantine.is_dir())
        self.assertEqual(list(close_quarantine.iterdir()), [])
        self.assertEqual(close_caught.exception.recovery_paths, (close_quarantine,))

        claimed_close_parent = self.root / "claimed-cleanup-close"
        claimed_close_parent.mkdir()
        claimed_close_destination = claimed_close_parent / "report"
        claimed_close_quarantine = (
            claimed_close_parent
            / CLEANUP_ARENA_NAME
            / ".report-publish-claimed-close.quarantine"
        )
        claimed_close_path = claimed_close_quarantine / "claimed"
        real_create = publication._create_rollback_quarantine
        real_open = publication.os.open
        real_close = publication.os.close
        real_unlink = publication.os.unlink
        real_rmdir = publication.os.rmdir
        quarantine_descriptor = None
        claimed_descriptor = None
        claimed_close_calls = 0
        claimed_unlinks = 0
        quarantine_rmdirs = 0

        def capture_claimed_quarantine(parent_handle, public_name):
            nonlocal quarantine_descriptor
            quarantine_handle = real_create(parent_handle, public_name)
            quarantine_descriptor = quarantine_handle.descriptor.fileno()
            return quarantine_handle

        def capture_claimed_open(path, flags, mode=0o777, *, dir_fd=None):
            nonlocal claimed_descriptor
            descriptor = real_open(path, flags, mode, dir_fd=dir_fd)
            if path == "claimed" and dir_fd == quarantine_descriptor:
                claimed_descriptor = descriptor
            return descriptor

        def close_claimed_then_fail(descriptor):
            nonlocal claimed_close_calls
            if descriptor == claimed_descriptor and claimed_close_calls == 0:
                claimed_close_calls += 1
                real_close(descriptor)
                raise OSError("injected claimed descriptor close failure")
            return real_close(descriptor)

        def record_claimed_unlink(path, *, dir_fd=None):
            nonlocal claimed_unlinks
            if path == "claimed" and dir_fd == quarantine_descriptor:
                claimed_unlinks += 1
            return real_unlink(path, dir_fd=dir_fd)

        def record_claimed_rmdir(path, *, dir_fd=None):
            nonlocal quarantine_rmdirs
            if path == claimed_close_quarantine.name:
                quarantine_rmdirs += 1
            return real_rmdir(path, dir_fd=dir_fd)

        with (
            mock.patch.object(
                publication, "_require_platform_support", return_value=None
            ),
            mock.patch.object(
                publication.secrets,
                "token_hex",
                side_effect=(
                    "probe-claimed-close",
                    "probe-cleanup-claimed-close",
                    "stage-claimed-close",
                    "claimed-close",
                ),
            ),
            mock.patch.object(
                publication,
                "_verify_item_precommit",
                side_effect=OSError("injected precommit failure"),
            ),
            mock.patch.object(
                publication,
                "_create_rollback_quarantine",
                side_effect=capture_claimed_quarantine,
            ),
            mock.patch.object(publication.os, "open", side_effect=capture_claimed_open),
            mock.patch.object(
                publication.os, "close", side_effect=close_claimed_then_fail
            ),
            mock.patch.object(
                publication.os, "unlink", side_effect=record_claimed_unlink
            ),
            mock.patch.object(
                publication.os, "rmdir", side_effect=record_claimed_rmdir
            ),
        ):
            with self.assertRaisesRegex(
                RollbackIndeterminateError, "claimed descriptor cleanup failed"
            ) as claimed_close_caught:
                publish_outputs(
                    [ReportOutput(claimed_close_destination, b"owned-stage")]
                )

        self.assertEqual(claimed_close_calls, 1)
        self.assertEqual(claimed_unlinks, 0)
        self.assertEqual(quarantine_rmdirs, 0)
        self.assertEqual(claimed_close_path.read_bytes(), b"owned-stage")
        self.assertEqual(
            claimed_close_caught.exception.recovery_paths, (claimed_close_path,)
        )

    def test_empty_quarantine_observation_error_reports_quarantine_root(self):
        destination = self.root / "report"
        quarantine = (
            self.root / CLEANUP_ARENA_NAME / ".report-publish-quarantine.quarantine"
        )
        real_stat = publication.os.stat
        real_rmdir = publication.os.rmdir
        root_stats = 0
        interrupted = False

        def fail_quarantine_rmdir(path, *, dir_fd=None):
            if path == quarantine.name:
                raise OSError("injected quarantine rmdir failure")
            return real_rmdir(path, dir_fd=dir_fd)

        def interrupt_root_reobservation(path, *args, **kwargs):
            nonlocal root_stats, interrupted
            if path == quarantine.name:
                root_stats += 1
                if root_stats == 3:
                    interrupted = True
                    raise InterruptedError(
                        "injected quarantine-root reobservation interruption"
                    )
            return real_stat(path, *args, **kwargs)

        with (
            mock.patch.object(
                publication, "_require_platform_support", return_value=None
            ),
            mock.patch.object(
                publication.secrets,
                "token_hex",
                side_effect=("probe", "probe-cleanup", "stage", "quarantine"),
            ),
            mock.patch.object(
                publication,
                "_verify_item_precommit",
                side_effect=OSError("injected precommit failure"),
            ),
            mock.patch.object(
                publication.os, "rmdir", side_effect=fail_quarantine_rmdir
            ),
            mock.patch.object(
                publication.os,
                "stat",
                side_effect=interrupt_root_reobservation,
            ),
        ):
            with self.assertRaisesRegex(
                RollbackIndeterminateError, "quarantine cleanup failed"
            ) as caught:
                publish_outputs([ReportOutput(destination, b"owned-stage")])

        self.assertTrue(interrupted)
        self.assertTrue(quarantine.is_dir())
        self.assertEqual(list(quarantine.iterdir()), [])
        self.assertEqual(caught.exception.recovery_paths, (quarantine,))

    def test_precommit_failure_retains_verified_recovery_backup(self):
        destination = self.root / "report"
        destination.write_bytes(b"old")

        with mock.patch.object(
            publication,
            "_verify_item_precommit",
            side_effect=OSError("injected precommit failure"),
        ):
            with self.assertRaisesRegex(
                RollbackIndeterminateError, "verified recovery creation"
            ) as caught:
                publish_outputs([ReportOutput(destination, b"new")])

        self.assert_file(destination, b"old")
        backups = list(self.root.glob(".report-publish-*.backup"))
        self.assertEqual(len(backups), 1)
        self.assertEqual(backups[0].read_bytes(), b"old")
        self.assertEqual(caught.exception.recovery_paths, tuple(backups))
        self.assertEqual(set(self.sidecars()), set(backups))

    def test_partial_backup_failure_cleans_only_owned_sidecars(self):
        destination = self.root / "report"
        destination.write_bytes(b"old contents")

        def fail_copy(source, destination_descriptor, size):
            self.assertEqual(size, len(b"old contents"))
            publication._write_all(destination_descriptor, b"partial")
            raise OSError("injected backup copy failure")

        with mock.patch.object(publication, "_copy_all", fail_copy):
            with self.assertRaisesRegex(
                RollbackIndeterminateError, "unverified partial"
            ) as caught:
                publish_outputs([ReportOutput(destination, b"new")])

        self.assert_file(destination, b"old contents")
        backups = list(self.root.glob(".report-publish-*.backup"))
        self.assertEqual(len(backups), 1)
        self.assertEqual(backups[0].read_bytes(), b"partial")
        self.assertEqual(caught.exception.recovery_paths, tuple(backups))

        writable_parent = self.root / "writable-original"
        writable_parent.mkdir()
        writable_destination = writable_parent / "report"
        writable_destination.write_bytes(b"old-writable")
        writable_destination.chmod(0o666)
        real_record = publication._record_artifact
        mutated = False

        def mutate_after_writable_mode(descriptor, label):
            nonlocal mutated
            if label == "report recovery backup":
                partial = next(writable_parent.glob(".report-publish-*.backup"))
                partial.write_bytes(b"foreign-in-place-mutation")
                mutated = True
                raise OSError("injected post-mode artifact failure")
            return real_record(descriptor, label)

        with mock.patch.object(
            publication, "_record_artifact", side_effect=mutate_after_writable_mode
        ):
            with self.assertRaisesRegex(
                RollbackIndeterminateError, "without identity-only cleanup"
            ) as writable_caught:
                publish_outputs([ReportOutput(writable_destination, b"new-writable")])

        self.assertTrue(mutated)
        self.assert_file(writable_destination, b"old-writable", mode=0o666)
        writable_backups = list(writable_parent.glob(".report-publish-*.backup"))
        self.assertEqual(len(writable_backups), 1)
        self.assertEqual(writable_backups[0].read_bytes(), b"foreign-in-place-mutation")
        self.assertEqual(
            writable_caught.exception.recovery_paths, tuple(writable_backups)
        )

    def test_partial_backup_cleanup_claim_preserves_foreign_swap(self):
        destination = self.root / "report"
        destination.write_bytes(b"old contents")
        foreign = self.root / "foreign-backup"
        foreign.write_bytes(b"foreign-backup-bytes")

        def fail_after_private_artifact(*_args):
            raise OSError("injected backup ownership failure")

        def swap_partial_backup(parent, name, label):
            if label.startswith("partial report recovery backup"):
                os.replace(foreign, parent.path / name)

        with (
            mock.patch.object(
                publication,
                "_set_backup_ownership",
                side_effect=fail_after_private_artifact,
            ),
            mock.patch.object(
                publication,
                "_before_public_sidecar_claim",
                side_effect=swap_partial_backup,
            ),
        ):
            with self.assertRaisesRegex(
                RollbackIndeterminateError, "foreign bytes"
            ) as caught:
                publish_outputs([ReportOutput(destination, b"new")])

        self.assert_file(destination, b"old contents")
        quarantined = list(self.root.rglob(".report-publish-*.quarantine/claimed"))
        self.assertEqual(len(quarantined), 1)
        self.assertEqual(quarantined[0].read_bytes(), b"foreign-backup-bytes")
        self.assertEqual(caught.exception.recovery_paths, tuple(quarantined))
        self.assertTrue(all(path.exists() for path in caught.exception.recovery_paths))

    def test_hashing_is_bounded_to_the_initial_size_with_extending_reader(self):
        size = 1024 * 1024 + 7
        calls = []

        def extending_reader(descriptor, requested, offset):
            calls.append((descriptor, requested, offset))
            return b"x" * requested

        with mock.patch.object(publication, "_pread", extending_reader):
            observed = publication._digest_descriptor(123, size)

        self.assertEqual(
            calls,
            [(123, 1024 * 1024, 0), (123, 7, 1024 * 1024)],
        )
        self.assertEqual(observed, publication.hashlib.sha256(b"x" * size).hexdigest())

    def test_unidentifiable_created_sidecar_closes_fd_and_is_preserved(self):
        destination = self.root / "report"
        before_fds = len(list(Path("/dev/fd").iterdir()))

        with (
            mock.patch.object(
                publication,
                "_freeze_created_file_identity",
                side_effect=OSError("injected fstat failure"),
            ),
            mock.patch.object(
                publication.secrets, "token_hex", return_value="unidentified"
            ),
        ):
            with self.assertRaisesRegex(
                RollbackIndeterminateError, "identity could not be frozen"
            ) as caught:
                publish_outputs([ReportOutput(destination, b"new")])

        after_fds = len(list(Path("/dev/fd").iterdir()))
        self.assertEqual(after_fds, before_fds)
        self.assertFalse(destination.exists())
        sidecar = self.root / ".report-publish-unidentified.stage"
        self.assertTrue(sidecar.is_file())
        self.assertEqual(sidecar.read_bytes(), b"")
        self.assertEqual(caught.exception.recovery_paths, ())
        self.assertEqual(caught.exception.candidate_paths, (sidecar,))

        prepared_parent = self.root / "prepared-stage-close"
        prepared_parent.mkdir()
        destination = prepared_parent / "report"
        real_open_unique = publication._open_unique
        real_close = publication.os.close
        stage_descriptor = None
        stage_name = None
        reused_descriptor = None
        close_calls = 0

        def capture_prepared_stage(parent_handle, suffix):
            nonlocal stage_descriptor, stage_name
            result = real_open_unique(parent_handle, suffix)
            if suffix == "stage":
                stage_name, stage_descriptor, _identity = result
            return result

        def close_prepared_stage_then_reuse(descriptor):
            nonlocal close_calls, reused_descriptor
            if descriptor == stage_descriptor and close_calls == 0:
                close_calls += 1
                real_close(descriptor)
                reused_descriptor = os.open(os.devnull, os.O_RDONLY)
                self.assertEqual(reused_descriptor, stage_descriptor)
                raise OSError("injected prepared stage close uncertainty")
            return real_close(descriptor)

        try:
            with (
                mock.patch.object(
                    publication, "_require_platform_support", return_value=None
                ),
                mock.patch.object(
                    publication,
                    "_open_unique",
                    side_effect=capture_prepared_stage,
                ),
                mock.patch.object(
                    publication.os,
                    "close",
                    side_effect=close_prepared_stage_then_reuse,
                ),
            ):
                with self.assertRaisesRegex(
                    RollbackIndeterminateError, "pathname cleanup prohibited"
                ) as prepared_caught:
                    publish_outputs([ReportOutput(destination, b"prepared-bytes")])
        finally:
            if reused_descriptor is not None:
                os.fstat(reused_descriptor)
                real_close(reused_descriptor)

        self.assertEqual(close_calls, 1)
        self.assertIsNotNone(stage_name)
        prepared_stage = prepared_parent / stage_name
        self.assertEqual(prepared_stage.read_bytes(), b"prepared-bytes")
        self.assertEqual(prepared_caught.exception.recovery_paths, ())
        self.assertEqual(prepared_caught.exception.candidate_paths, (prepared_stage,))
        self.assertEqual(
            list(prepared_parent.rglob(".report-publish-*.quarantine")), []
        )

    def test_unknown_stage_identity_close_failure_reports_retained_stage(self):
        parent_descriptor = os.open(self.root, publication._directory_flags())
        parent = publication._Parent(self.root, parent_descriptor)
        sidecar = self.root / ".report-publish-close-failure.stage"
        real_close = publication.os.close

        def close_then_fail(descriptor):
            if descriptor != parent_descriptor:
                real_close(descriptor)
                raise OSError("injected close failure")
            return real_close(descriptor)

        try:
            with (
                mock.patch.object(
                    publication,
                    "_freeze_created_file_identity",
                    side_effect=OSError("injected identity failure"),
                ),
                mock.patch.object(
                    publication.secrets,
                    "token_hex",
                    return_value="close-failure",
                ),
                mock.patch.object(publication.os, "close", side_effect=close_then_fail),
            ):
                with self.assertRaises(RollbackIndeterminateError) as caught:
                    publication._open_unique(parent, "stage")
        finally:
            real_close(parent_descriptor)

        self.assertTrue(sidecar.is_file())
        self.assertEqual(sidecar.read_bytes(), b"")
        self.assertEqual(caught.exception.recovery_paths, ())
        self.assertEqual(caught.exception.candidate_paths, (sidecar,))

    def test_unknown_stage_identity_observation_error_reports_retained_stage(self):
        parent_descriptor = os.open(self.root, publication._directory_flags())
        parent = publication._Parent(self.root, parent_descriptor)
        sidecar = self.root / ".report-publish-observation-error.stage"
        real_stat = publication.os.stat
        interrupted = False

        def interrupt_sidecar_observation(path, *args, **kwargs):
            nonlocal interrupted
            if path == sidecar.name:
                interrupted = True
                raise InterruptedError("injected sidecar observation interruption")
            return real_stat(path, *args, **kwargs)

        try:
            with (
                mock.patch.object(
                    publication,
                    "_freeze_created_file_identity",
                    side_effect=OSError("injected identity failure"),
                ),
                mock.patch.object(
                    publication.secrets,
                    "token_hex",
                    return_value="observation-error",
                ),
                mock.patch.object(
                    publication.os,
                    "stat",
                    side_effect=interrupt_sidecar_observation,
                ),
            ):
                with self.assertRaises(RollbackIndeterminateError) as caught:
                    publication._open_unique(parent, "stage")
        finally:
            os.close(parent_descriptor)

        self.assertTrue(interrupted)
        self.assertEqual(sidecar.read_bytes(), b"")
        self.assertEqual(caught.exception.recovery_paths, ())
        self.assertEqual(caught.exception.candidate_paths, (sidecar,))

    def test_outer_rollback_merges_unknown_stage_and_quarantined_foreign_bytes(self):
        first = self.root / "a"
        second = self.root / "b"
        foreign = self.root / "foreign-first-stage"
        foreign.write_bytes(b"foreign-first-stage-bytes")
        real_freeze = publication._freeze_created_file_identity
        freezes = 0

        def fail_second_stage_identity(descriptor):
            nonlocal freezes
            freezes += 1
            if freezes == 2:
                os.replace(
                    foreign,
                    self.root / ".report-publish-first.stage",
                )
                raise OSError("injected second stage identity failure")
            return real_freeze(descriptor)

        with (
            mock.patch.object(
                publication.secrets,
                "token_hex",
                side_effect=(
                    "probe",
                    "probe-cleanup",
                    "first",
                    "second",
                    "rollback",
                ),
            ),
            mock.patch.object(
                publication,
                "_freeze_created_file_identity",
                side_effect=fail_second_stage_identity,
            ),
        ):
            with self.assertRaises(RollbackIndeterminateError) as caught:
                publish_outputs(
                    [
                        ReportOutput(first, b"owned-first"),
                        ReportOutput(second, b"owned-second"),
                    ]
                )

        unknown_stage = self.root / ".report-publish-second.stage"
        claimed = (
            self.root
            / CLEANUP_ARENA_NAME
            / ".report-publish-rollback.quarantine"
            / "claimed"
        )
        self.assertEqual(caught.exception.recovery_paths, (claimed,))
        self.assertEqual(caught.exception.candidate_paths, (unknown_stage,))
        self.assertEqual(unknown_stage.read_bytes(), b"")
        self.assertEqual(claimed.read_bytes(), b"foreign-first-stage-bytes")
        self.assertFalse(first.exists())
        self.assertFalse(second.exists())

    def test_unknown_stage_identity_with_absent_name_preserves_original_error(self):
        parent_descriptor = os.open(self.root, publication._directory_flags())
        parent = publication._Parent(self.root, parent_descriptor)
        sidecar = self.root / ".report-publish-removed-before-error.stage"

        def remove_then_fail(_descriptor):
            sidecar.unlink()
            raise OSError("injected identity failure after removal")

        try:
            with (
                mock.patch.object(
                    publication,
                    "_freeze_created_file_identity",
                    side_effect=remove_then_fail,
                ),
                mock.patch.object(
                    publication.secrets,
                    "token_hex",
                    return_value="removed-before-error",
                ),
            ):
                with self.assertRaisesRegex(
                    OSError, "identity failure after removal"
                ) as caught:
                    publication._open_unique(parent, "stage")
        finally:
            os.close(parent_descriptor)

        self.assertNotIsInstance(caught.exception, RollbackIndeterminateError)
        self.assertFalse(sidecar.exists())

    def test_unknown_early_directory_identity_reports_exact_recovery_path(self):
        cases = (
            ("probe", publication._create_collision_probe),
            (
                "quarantine",
                lambda parent: publication._create_rollback_quarantine(
                    parent, "report"
                ),
            ),
        )
        for suffix, creator in cases:
            with self.subTest(suffix=suffix):
                parent_path = self.root / f"unknown-{suffix}-identity"
                parent_path.mkdir()
                parent_descriptor = os.open(parent_path, publication._directory_flags())
                parent = publication._Parent(parent_path, parent_descriptor)
                token = f"unknown-{suffix}"
                retained_parent = (
                    parent_path
                    if suffix == "probe"
                    else parent_path / CLEANUP_ARENA_NAME
                )
                retained = retained_parent / f".report-publish-{token}.{suffix}"
                real_stat = publication.os.stat
                failed = False

                def fail_first_identity_stat(path, *args, **kwargs):
                    nonlocal failed
                    if path == retained.name and not failed:
                        failed = True
                        raise OSError("injected identity stat failure")
                    return real_stat(path, *args, **kwargs)

                try:
                    with (
                        mock.patch.object(
                            publication.secrets, "token_hex", return_value=token
                        ),
                        mock.patch.object(
                            publication.os,
                            "stat",
                            side_effect=fail_first_identity_stat,
                        ),
                    ):
                        with self.assertRaises(
                            (
                                RollbackIndeterminateError,
                                publication.repository_snapshot.CleanupFailure,
                            )
                        ) as caught:
                            creator(parent)
                finally:
                    os.close(parent_descriptor)

                self.assertTrue(retained.is_dir())
                self.assertEqual(list(retained.iterdir()), [])
                if suffix == "probe":
                    self.assertEqual(self.cleanup_recovery_paths(caught.exception), ())
                    self.assertEqual(
                        self.cleanup_candidate_paths(caught.exception), (retained,)
                    )
                else:
                    self.assertEqual(
                        self.cleanup_recovery_paths(caught.exception), (retained,)
                    )

        parent_path = self.root / "quarantine-setup-close"
        parent_path.mkdir()
        parent_descriptor = os.open(parent_path, publication._directory_flags())
        parent = publication._Parent(parent_path, parent_descriptor)
        token = "setup-close"
        retained = (
            parent_path / CLEANUP_ARENA_NAME / f".report-publish-{token}.quarantine"
        )
        real_open = publication.os.open
        real_close = publication.os.close
        real_fchmod = publication.os.fchmod
        real_rmdir = publication.os.rmdir
        quarantine_descriptor = None
        close_calls = 0
        quarantine_rmdirs = 0

        def capture_quarantine_open(path, flags, mode=0o777, *, dir_fd=None):
            nonlocal quarantine_descriptor
            descriptor = real_open(path, flags, mode, dir_fd=dir_fd)
            if path == retained.name:
                quarantine_descriptor = descriptor
            return descriptor

        def close_setup_after_transfer(descriptor):
            nonlocal close_calls
            if descriptor == quarantine_descriptor:
                close_calls += 1
                real_close(descriptor)
                raise OSError("injected setup quarantine close failure")
            return real_close(descriptor)

        def record_setup_rmdir(path, *, dir_fd=None):
            nonlocal quarantine_rmdirs
            if path == retained.name:
                quarantine_rmdirs += 1
            return real_rmdir(path, dir_fd=dir_fd)

        def fail_quarantine_fchmod(descriptor, mode):
            if descriptor == quarantine_descriptor:
                raise OSError("injected post-fchmod verification failure")
            return real_fchmod(descriptor, mode)

        try:
            with (
                mock.patch.object(publication.secrets, "token_hex", return_value=token),
                mock.patch.object(
                    publication.os,
                    "open",
                    side_effect=capture_quarantine_open,
                ),
                mock.patch.object(
                    publication.os,
                    "close",
                    side_effect=close_setup_after_transfer,
                ),
                mock.patch.object(
                    publication.os,
                    "rmdir",
                    side_effect=record_setup_rmdir,
                ),
                mock.patch.object(
                    publication.os,
                    "fchmod",
                    side_effect=fail_quarantine_fchmod,
                ),
            ):
                with self.assertRaises(
                    publication.repository_snapshot.CleanupFailure
                ) as close_caught:
                    publication._create_rollback_quarantine(parent, "report")
        finally:
            real_close(parent_descriptor)

        self.assertEqual(close_calls, 1)
        self.assertEqual(quarantine_rmdirs, 0)
        self.assertTrue(retained.is_dir())
        self.assertEqual(list(retained.iterdir()), [])
        self.assertIn(
            "cleanup_quarantine_descriptor_close_uncertain",
            self.cleanup_issue_text(close_caught.exception),
        )
        self.assertEqual(
            self.cleanup_recovery_paths(close_caught.exception), (retained,)
        )

        parent_path = self.root / "setup-close-rebind"
        parent_path.mkdir()
        public_probe = parent_path / ".report-publish-rebound.probe"
        public_probe.mkdir(mode=0o700)
        moved_parent = self.root / "setup-close-rebind-detached"
        parent_descriptor = os.open(parent_path, publication._directory_flags())
        source = publication.repository_snapshot.DirectoryAnchor(
            parent_descriptor, parent_path
        )
        real_arena_close_issue = (
            publication.repository_snapshot.CleanupArena.close_issue
        )
        rebound = False

        def close_arena_then_rebind(arena):
            nonlocal rebound
            issue = real_arena_close_issue(arena)
            if not rebound and arena.path.parent == parent_path:
                rebound = True
                os.rename(parent_path, moved_parent)
                parent_path.mkdir()
            return issue

        try:
            with mock.patch.object(
                publication.repository_snapshot.CleanupArena,
                "close_issue",
                side_effect=close_arena_then_rebind,
                autospec=True,
            ):
                with self.assertRaises(
                    publication.repository_snapshot.CleanupFailure
                ) as rebound_caught:
                    publication._create_directory_cleanup_quarantine(
                        source,
                        public_probe.name,
                        ".quarantine",
                    )
        finally:
            os.close(parent_descriptor)

        self.assertTrue(rebound)
        self.assertEqual(
            rebound_caught.exception.outcome.disposition,
            publication.repository_snapshot.CleanupDisposition.UNADDRESSABLE,
        )
        self.assertEqual(self.cleanup_recovery_paths(rebound_caught.exception), ())
        self.assertIn(
            "binding_rebound", self.cleanup_issue_text(rebound_caught.exception)
        )
        self.assertTrue(moved_parent.joinpath(CLEANUP_ARENA_NAME).is_dir())

        for mode in ("mode", "identity"):
            with self.subTest(post_fchmod=mode):
                parent_path = self.root / f"post-fchmod-{mode}"
                parent_path.mkdir()
                parent_descriptor = os.open(parent_path, publication._directory_flags())
                parent = publication._Parent(parent_path, parent_descriptor)
                token = f"post-fchmod-{mode}"
                quarantine = (
                    parent_path
                    / CLEANUP_ARENA_NAME
                    / f".report-publish-{token}.quarantine"
                )
                real_open = publication.os.open
                real_fstat = publication.os.fstat
                quarantine_descriptor = None
                directory_fstats = 0

                def capture_quarantine_open(
                    path, flags, open_mode=0o777, *, dir_fd=None
                ):
                    nonlocal quarantine_descriptor
                    descriptor = real_open(path, flags, open_mode, dir_fd=dir_fd)
                    if path == quarantine.name:
                        quarantine_descriptor = descriptor
                    return descriptor

                def alter_post_fchmod_metadata(descriptor):
                    nonlocal directory_fstats
                    metadata = real_fstat(descriptor)
                    if descriptor == quarantine_descriptor:
                        directory_fstats += 1
                        if directory_fstats == 2:
                            values = list(metadata)
                            if mode == "mode":
                                values[stat.ST_MODE] = (
                                    metadata.st_mode & ~0o7777
                                ) | 0o755
                            else:
                                values[stat.ST_INO] = metadata.st_ino + 1
                            return os.stat_result(values)
                    return metadata

                try:
                    with (
                        mock.patch.object(
                            publication.secrets,
                            "token_hex",
                            return_value=token,
                        ),
                        mock.patch.object(
                            publication.os,
                            "open",
                            side_effect=capture_quarantine_open,
                        ),
                        mock.patch.object(
                            publication.os,
                            "fstat",
                            side_effect=alter_post_fchmod_metadata,
                        ),
                    ):
                        with self.assertRaises(
                            publication.repository_snapshot.CleanupFailure
                        ) as post_caught:
                            publication._create_rollback_quarantine(parent, "report")
                finally:
                    os.close(parent_descriptor)

                self.assertEqual(directory_fstats, 2)
                self.assertTrue(quarantine.is_dir())
                self.assertIn(
                    "configured credential is unsafe",
                    self.cleanup_issue_text(post_caught.exception),
                )
                self.assertEqual(
                    self.cleanup_recovery_paths(post_caught.exception), (quarantine,)
                )

        creators = (
            ("probe", publication._create_collision_probe),
            (
                "quarantine",
                lambda parent: publication._create_rollback_quarantine(
                    parent, "report"
                ),
            ),
        )
        for suffix, creator in creators:
            with self.subTest(pre_fchmod_foreign_owner=suffix):
                parent_path = self.root / f"foreign-owner-{suffix}"
                parent_path.mkdir()
                parent_descriptor = os.open(parent_path, publication._directory_flags())
                parent = publication._Parent(parent_path, parent_descriptor)
                token = f"foreign-owner-{suffix}"
                retained_parent = (
                    parent_path
                    if suffix == "probe"
                    else parent_path / CLEANUP_ARENA_NAME
                )
                retained = retained_parent / f".report-publish-{token}.{suffix}"
                detached = retained_parent / f"detached-{suffix}"
                real_open = publication.os.open
                real_fstat = publication.os.fstat
                real_fchmod = publication.os.fchmod
                real_rmdir = publication.os.rmdir
                captured_descriptor = None
                frozen_metadata = None
                owner_altered = False
                fchmod_calls = 0
                rmdir_calls = 0

                def replace_before_open(path, flags, mode=0o777, *, dir_fd=None):
                    nonlocal captured_descriptor, frozen_metadata
                    if path == retained.name:
                        frozen_metadata = publication.os.stat(
                            path, dir_fd=dir_fd, follow_symlinks=False
                        )
                        publication.os.rename(
                            path,
                            detached.name,
                            src_dir_fd=dir_fd,
                            dst_dir_fd=dir_fd,
                        )
                        publication.os.mkdir(path, 0o700, dir_fd=dir_fd)
                    descriptor = real_open(path, flags, mode, dir_fd=dir_fd)
                    if path == retained.name:
                        captured_descriptor = descriptor
                    return descriptor

                def report_foreign_owner(descriptor):
                    nonlocal owner_altered
                    metadata = real_fstat(descriptor)
                    if descriptor == captured_descriptor and not owner_altered:
                        owner_altered = True
                        self.assertIsNotNone(frozen_metadata)
                        values = list(metadata)
                        values[stat.ST_DEV] = frozen_metadata.st_dev
                        values[stat.ST_INO] = frozen_metadata.st_ino
                        values[stat.ST_UID] = metadata.st_uid + 1
                        return os.stat_result(values)
                    return metadata

                def record_fchmod(descriptor, mode):
                    nonlocal fchmod_calls
                    if descriptor == captured_descriptor:
                        fchmod_calls += 1
                    return real_fchmod(descriptor, mode)

                def record_rmdir(path, *, dir_fd=None):
                    nonlocal rmdir_calls
                    if path == retained.name:
                        rmdir_calls += 1
                    return real_rmdir(path, dir_fd=dir_fd)

                try:
                    with (
                        mock.patch.object(
                            publication.secrets, "token_hex", return_value=token
                        ),
                        mock.patch.object(
                            publication.os,
                            "open",
                            side_effect=replace_before_open,
                        ),
                        mock.patch.object(
                            publication.os,
                            "fstat",
                            side_effect=report_foreign_owner,
                        ),
                        mock.patch.object(
                            publication.os,
                            "fchmod",
                            side_effect=record_fchmod,
                        ),
                        mock.patch.object(
                            publication.os,
                            "rmdir",
                            side_effect=record_rmdir,
                        ),
                    ):
                        with self.assertRaises(
                            (
                                RollbackIndeterminateError,
                                publication.repository_snapshot.CleanupFailure,
                            )
                        ) as owner_caught:
                            creator(parent)
                finally:
                    os.close(parent_descriptor)

                self.assertTrue(owner_altered)
                self.assertEqual(fchmod_calls, 0)
                self.assertEqual(rmdir_calls, 0)
                self.assertTrue(retained.is_dir())
                self.assertTrue(detached.is_dir())
                if suffix == "probe":
                    self.assertEqual(
                        self.cleanup_recovery_paths(owner_caught.exception), ()
                    )
                    self.assertEqual(
                        self.cleanup_candidate_paths(owner_caught.exception),
                        (retained,),
                    )
                else:
                    self.assertEqual(
                        self.cleanup_recovery_paths(owner_caught.exception),
                        (retained,),
                    )

        for mode in ("owner", "mode", "identity"):
            with self.subTest(probe_post_fchmod=mode):
                parent_path = self.root / f"probe-post-fchmod-{mode}"
                parent_path.mkdir()
                parent_descriptor = os.open(parent_path, publication._directory_flags())
                parent = publication._Parent(parent_path, parent_descriptor)
                token = f"probe-post-{mode}"
                probe = parent_path / f".report-publish-{token}.probe"
                real_open = publication.os.open
                real_fstat = publication.os.fstat
                probe_descriptor = None
                directory_fstats = 0

                def capture_probe_metadata_open(
                    path, flags, open_mode=0o777, *, dir_fd=None
                ):
                    nonlocal probe_descriptor
                    descriptor = real_open(path, flags, open_mode, dir_fd=dir_fd)
                    if path == probe.name:
                        probe_descriptor = descriptor
                    return descriptor

                def alter_probe_post_fchmod(descriptor):
                    nonlocal directory_fstats
                    metadata = real_fstat(descriptor)
                    if descriptor == probe_descriptor:
                        directory_fstats += 1
                        if directory_fstats == 2:
                            values = list(metadata)
                            if mode == "owner":
                                values[stat.ST_UID] = metadata.st_uid + 1
                            elif mode == "mode":
                                values[stat.ST_MODE] = (
                                    metadata.st_mode & ~0o7777
                                ) | 0o755
                            else:
                                values[stat.ST_INO] = metadata.st_ino + 1
                            return os.stat_result(values)
                    return metadata

                try:
                    with (
                        mock.patch.object(
                            publication.secrets, "token_hex", return_value=token
                        ),
                        mock.patch.object(
                            publication.os,
                            "open",
                            side_effect=capture_probe_metadata_open,
                        ),
                        mock.patch.object(
                            publication.os,
                            "fstat",
                            side_effect=alter_probe_post_fchmod,
                        ),
                    ):
                        with self.assertRaises(
                            RollbackIndeterminateError
                        ) as probe_caught:
                            publication._create_collision_probe(parent)
                finally:
                    os.close(parent_descriptor)

                self.assertEqual(directory_fstats, 2)
                self.assertTrue(probe.is_dir())
                self.assertEqual(
                    self.cleanup_recovery_paths(probe_caught.exception), ()
                )
                self.assertEqual(
                    self.cleanup_candidate_paths(probe_caught.exception), (probe,)
                )

        for mode in ("reuse", "close-error"):
            with self.subTest(probe_setup_close=mode):
                parent_path = self.root / f"probe-setup-{mode}"
                parent_path.mkdir()
                parent_descriptor = os.open(parent_path, publication._directory_flags())
                parent = publication._Parent(parent_path, parent_descriptor)
                token = f"probe-setup-{mode}"
                probe = parent_path / f".report-publish-{token}.probe"
                real_open = publication.os.open
                real_close = publication.os.close
                real_fstat = publication.os.fstat
                real_rmdir = publication.os.rmdir
                captured_descriptor = None
                reused_descriptor = None
                close_calls = 0
                rmdir_calls = 0
                identity_altered = False

                def capture_probe_open(path, flags, open_mode=0o777, *, dir_fd=None):
                    nonlocal captured_descriptor
                    descriptor = real_open(path, flags, open_mode, dir_fd=dir_fd)
                    if path == probe.name:
                        captured_descriptor = descriptor
                    return descriptor

                def alter_probe_identity(descriptor):
                    nonlocal identity_altered
                    metadata = real_fstat(descriptor)
                    if descriptor == captured_descriptor and not identity_altered:
                        identity_altered = True
                        values = list(metadata)
                        values[stat.ST_INO] = metadata.st_ino + 1
                        return os.stat_result(values)
                    return metadata

                def close_probe_setup(descriptor):
                    nonlocal close_calls, reused_descriptor
                    if descriptor == captured_descriptor:
                        close_calls += 1
                        if close_calls == 1:
                            real_close(descriptor)
                            if mode == "reuse":
                                reused_descriptor = real_open(os.devnull, os.O_RDONLY)
                                self.assertEqual(reused_descriptor, captured_descriptor)
                                return None
                            raise OSError("injected probe setup close failure")
                    return real_close(descriptor)

                def record_probe_rmdir(path, *, dir_fd=None):
                    nonlocal rmdir_calls
                    if path == probe.name:
                        rmdir_calls += 1
                    return real_rmdir(path, dir_fd=dir_fd)

                try:
                    with (
                        mock.patch.object(
                            publication.secrets, "token_hex", return_value=token
                        ),
                        mock.patch.object(
                            publication.os,
                            "open",
                            side_effect=capture_probe_open,
                        ),
                        mock.patch.object(
                            publication.os,
                            "fstat",
                            side_effect=alter_probe_identity,
                        ),
                        mock.patch.object(
                            publication.os,
                            "close",
                            side_effect=close_probe_setup,
                        ),
                        mock.patch.object(
                            publication.os,
                            "rmdir",
                            side_effect=record_probe_rmdir,
                        ),
                    ):
                        with self.assertRaises(
                            RollbackIndeterminateError
                        ) as close_caught:
                            publication._create_collision_probe(parent)
                finally:
                    real_close(parent_descriptor)

                self.assertTrue(identity_altered)
                self.assertEqual(close_calls, 1)
                self.assertEqual(rmdir_calls, 0)
                self.assertTrue(probe.is_dir())
                self.assertEqual(
                    self.cleanup_recovery_paths(close_caught.exception), ()
                )
                self.assertEqual(
                    self.cleanup_candidate_paths(close_caught.exception), (probe,)
                )
                if reused_descriptor is not None:
                    real_fstat(reused_descriptor)
                    real_close(reused_descriptor)

    def test_early_directory_open_and_cleanup_failure_reports_exact_path(self):
        cases = (
            ("probe", publication._create_collision_probe),
            (
                "quarantine",
                lambda parent: publication._create_rollback_quarantine(
                    parent, "report"
                ),
            ),
        )
        for suffix, creator in cases:
            with self.subTest(suffix=suffix):
                parent_path = self.root / f"failed-{suffix}-setup"
                parent_path.mkdir()
                parent_descriptor = os.open(parent_path, publication._directory_flags())
                parent = publication._Parent(parent_path, parent_descriptor)
                token = f"failed-{suffix}"
                retained_parent = (
                    parent_path
                    if suffix == "probe"
                    else parent_path / CLEANUP_ARENA_NAME
                )
                retained = retained_parent / f".report-publish-{token}.{suffix}"
                real_open = publication.os.open
                real_rmdir = publication.os.rmdir

                def fail_created_directory_open(
                    path, flags, mode=0o777, *, dir_fd=None
                ):
                    if path == retained.name:
                        raise OSError("injected directory open failure")
                    return real_open(path, flags, mode, dir_fd=dir_fd)

                def fail_created_directory_cleanup(path, *, dir_fd=None):
                    if path == retained.name:
                        raise OSError("injected directory cleanup failure")
                    return real_rmdir(path, dir_fd=dir_fd)

                try:
                    with (
                        mock.patch.object(
                            publication.secrets, "token_hex", return_value=token
                        ),
                        mock.patch.object(
                            publication.os,
                            "open",
                            side_effect=fail_created_directory_open,
                        ),
                        mock.patch.object(
                            publication.os,
                            "rmdir",
                            side_effect=fail_created_directory_cleanup,
                        ),
                    ):
                        with self.assertRaises(
                            (
                                RollbackIndeterminateError,
                                publication.repository_snapshot.CleanupFailure,
                            )
                        ) as caught:
                            creator(parent)
                finally:
                    os.close(parent_descriptor)

                self.assertTrue(retained.is_dir())
                self.assertEqual(list(retained.iterdir()), [])
                if suffix == "probe":
                    self.assertEqual(self.cleanup_recovery_paths(caught.exception), ())
                    self.assertEqual(
                        self.cleanup_candidate_paths(caught.exception), (retained,)
                    )
                else:
                    self.assertEqual(
                        self.cleanup_recovery_paths(caught.exception), (retained,)
                    )

    def test_early_directory_cleanup_that_removed_name_preserves_open_error(self):
        cases = (
            ("probe", publication._create_collision_probe),
            (
                "quarantine",
                lambda parent: publication._create_rollback_quarantine(
                    parent, "report"
                ),
            ),
        )
        for suffix, creator in cases:
            with self.subTest(suffix=suffix):
                parent_path = self.root / f"removed-{suffix}-setup"
                parent_path.mkdir()
                parent_descriptor = os.open(parent_path, publication._directory_flags())
                parent = publication._Parent(parent_path, parent_descriptor)
                token = f"removed-{suffix}"
                cleaned_parent = (
                    parent_path
                    if suffix == "probe"
                    else parent_path / CLEANUP_ARENA_NAME
                )
                cleaned = cleaned_parent / f".report-publish-{token}.{suffix}"
                real_open = publication.os.open
                real_rmdir = publication.os.rmdir
                rmdir_calls = 0

                def fail_created_directory_open(
                    path, flags, mode=0o777, *, dir_fd=None
                ):
                    if path == cleaned.name:
                        raise OSError("injected directory open failure")
                    return real_open(path, flags, mode, dir_fd=dir_fd)

                def remove_then_fail_cleanup(path, *, dir_fd=None):
                    nonlocal rmdir_calls
                    rmdir_calls += 1
                    result = real_rmdir(path, dir_fd=dir_fd)
                    if path == cleaned.name:
                        raise OSError("injected post-cleanup failure")
                    return result

                try:
                    with (
                        mock.patch.object(
                            publication.secrets, "token_hex", return_value=token
                        ),
                        mock.patch.object(
                            publication.os,
                            "open",
                            side_effect=fail_created_directory_open,
                        ),
                        mock.patch.object(
                            publication.os,
                            "rmdir",
                            side_effect=remove_then_fail_cleanup,
                        ),
                    ):
                        with self.assertRaises(
                            (
                                RollbackIndeterminateError,
                                publication.repository_snapshot.CleanupFailure,
                            )
                        ) as caught:
                            creator(parent)
                finally:
                    os.close(parent_descriptor)

                self.assertEqual(rmdir_calls, 0)
                self.assertTrue(cleaned.is_dir())
                if suffix == "probe":
                    self.assertEqual(self.cleanup_recovery_paths(caught.exception), ())
                    self.assertEqual(
                        self.cleanup_candidate_paths(caught.exception), (cleaned,)
                    )
                else:
                    self.assertEqual(
                        self.cleanup_recovery_paths(caught.exception), (cleaned,)
                    )

    def test_nth_replace_failure_completely_restores_originals(self):
        destinations = [self.root / name for name in ("a", "b", "c")]
        for index, destination in enumerate(destinations):
            destination.write_bytes(f"old-{index}".encode())
        real_replace = publication._replace_name
        stage_replaces = 0

        def fail_second_stage(parent, source, destination):
            nonlocal stage_replaces
            if source.endswith(".stage"):
                stage_replaces += 1
                if stage_replaces == 2:
                    raise OSError("injected second replace failure")
            return real_replace(parent, source, destination)

        with mock.patch.object(publication, "_replace_name", fail_second_stage):
            with self.assertRaisesRegex(
                RollbackIndeterminateError, "retained recovery paths"
            ) as caught:
                publish_outputs(
                    [
                        ReportOutput(destination, f"new-{index}".encode())
                        for index, destination in enumerate(destinations)
                    ]
                )

        for index, destination in enumerate(destinations):
            self.assert_file(destination, f"old-{index}".encode())
        backups = list(self.root.glob(".report-publish-*.backup"))
        self.assertEqual(len(backups), 3)
        self.assertEqual(
            {path.read_bytes() for path in backups},
            {b"old-0", b"old-1", b"old-2"},
        )
        self.assertEqual(set(caught.exception.recovery_paths), set(backups))
        self.assertEqual(set(self.sidecars()), set(backups))

    def test_post_commit_verification_failure_completely_restores_outputs(self):
        old = self.root / "old"
        absent = self.root / "absent"
        old.write_bytes(b"old")
        old.chmod(0o600)

        with mock.patch.object(
            publication,
            "_verify_item_published",
            side_effect=OSError("injected verification failure"),
        ):
            with self.assertRaisesRegex(
                RollbackIndeterminateError, "continued old-data reachability"
            ) as caught:
                publish_outputs(
                    [ReportOutput(old, b"new-old"), ReportOutput(absent, b"new")]
                )

        self.assert_file(old, b"old", 0o600)
        self.assertFalse(absent.exists())
        backups = list(self.root.glob(".report-publish-*.backup"))
        self.assertEqual(len(backups), 1)
        self.assertEqual(backups[0].read_bytes(), b"old")
        self.assertEqual(caught.exception.recovery_paths, tuple(backups))

    def test_commit_directory_fsync_failure_completely_restores_outputs(self):
        old = self.root / "old"
        absent = self.root / "absent"
        old.write_bytes(b"old")
        real_fsync = publication._fsync_directory
        calls = 0

        def fail_once(parent):
            nonlocal calls
            calls += 1
            if calls == 1:
                raise OSError("injected directory fsync failure")
            return real_fsync(parent)

        with mock.patch.object(publication, "_fsync_directory", fail_once):
            with self.assertRaises(RollbackIndeterminateError) as caught:
                publish_outputs(
                    [ReportOutput(old, b"new-old"), ReportOutput(absent, b"new")]
                )

        self.assert_file(old, b"old")
        self.assertFalse(absent.exists())
        backups = list(self.root.glob(".report-publish-*.backup"))
        self.assertEqual(len(backups), 1)
        self.assertEqual(backups[0].read_bytes(), b"old")
        self.assertEqual(caught.exception.recovery_paths, tuple(backups))

    def test_external_replacement_is_preserved_and_backup_is_retained(self):
        destination = self.root / "report"
        destination.write_bytes(b"old")
        real_verify = publication._verify_item_published
        injected = False

        def replace_externally(item, stage):
            nonlocal injected
            if not injected:
                injected = True
                external = self.root / "external"
                external.write_bytes(b"external")
                os.replace(external, destination)
            return real_verify(item, stage)

        with mock.patch.object(
            publication, "_verify_item_published", replace_externally
        ):
            with self.assertRaisesRegex(
                RollbackIndeterminateError, "external replacement claimed"
            ) as caught:
                publish_outputs([ReportOutput(destination, b"new")])

        self.assertFalse(destination.exists())
        quarantined = list(self.root.rglob(".report-publish-*.quarantine/claimed"))
        self.assertEqual(len(quarantined), 1)
        self.assertEqual(quarantined[0].read_bytes(), b"external")
        backups = list(self.root.glob(".report-publish-*.backup"))
        self.assertEqual(len(backups), 1)
        self.assertEqual(backups[0].read_bytes(), b"old")
        expected = tuple(sorted([*quarantined, *backups], key=os.fspath))
        self.assertEqual(caught.exception.recovery_paths, expected)

    def test_absent_destination_external_replacement_reports_quarantine_member(self):
        destination = self.root / "report"
        real_verify = publication._verify_item_published
        injected = False

        def replace_new_output_externally(item, stage):
            nonlocal injected
            if not injected:
                injected = True
                external = self.root / "external"
                external.write_bytes(b"external-absent")
                os.replace(external, destination)
            return real_verify(item, stage)

        with mock.patch.object(
            publication, "_verify_item_published", replace_new_output_externally
        ):
            with self.assertRaisesRegex(
                RollbackIndeterminateError, "external replacement claimed"
            ) as caught:
                publish_outputs([ReportOutput(destination, b"new")])

        self.assertFalse(destination.exists())
        quarantined = list(self.root.rglob(".report-publish-*.quarantine/claimed"))
        self.assertEqual(len(quarantined), 1)
        self.assertEqual(quarantined[0].read_bytes(), b"external-absent")
        self.assertEqual(caught.exception.recovery_paths, tuple(quarantined))
        self.assertEqual(self.sidecars(), [quarantined[0].parent])

    def test_external_file_appearing_immediately_before_restore_survives(self):
        destination = self.root / "report"
        destination.write_bytes(b"old")

        def create_external(_item):
            destination.write_bytes(b"external-race")

        with (
            mock.patch.object(
                publication,
                "_verify_item_published",
                side_effect=OSError("force rollback"),
            ),
            mock.patch.object(publication, "_before_restore_original", create_external),
        ):
            with self.assertRaisesRegex(
                RollbackIndeterminateError, "appeared before restoration"
            ) as caught:
                publish_outputs([ReportOutput(destination, b"new")])

        self.assert_file(destination, b"external-race")
        quarantined = list(self.root.rglob(".report-publish-*.quarantine/claimed"))
        self.assertEqual(len(quarantined), 1)
        self.assertEqual(quarantined[0].read_bytes(), b"new")
        backups = list(self.root.glob(".report-publish-*.backup"))
        self.assertEqual(len(backups), 1)
        self.assertEqual(backups[0].read_bytes(), b"old")
        expected = tuple(
            sorted([*(path.parent for path in quarantined), *backups], key=os.fspath)
        )
        self.assertEqual(caught.exception.recovery_paths, expected)

    def test_external_replacement_at_former_backup_cleanup_keeps_old_recovery(self):
        destination = self.root / "report"
        destination.write_bytes(b"old")

        def replace_restored_destination(_item):
            external = self.root / "external-at-retention"
            external.write_bytes(b"external-retention-race")
            os.replace(external, destination)

        with (
            mock.patch.object(
                publication,
                "_verify_item_published",
                side_effect=OSError("force rollback"),
            ),
            mock.patch.object(
                publication,
                "_before_retain_failure_backup",
                replace_restored_destination,
            ),
        ):
            with self.assertRaises(RollbackIndeterminateError) as caught:
                publish_outputs([ReportOutput(destination, b"new")])

        self.assert_file(destination, b"external-retention-race")
        backups = list(self.root.glob(".report-publish-*.backup"))
        self.assertEqual(len(backups), 1)
        self.assertEqual(backups[0].read_bytes(), b"old")
        self.assertEqual(caught.exception.recovery_paths, tuple(backups))
        self.assertEqual(set(self.sidecars()), set(backups))

    def test_rollback_failure_is_indeterminate_and_keeps_recovery_backup(self):
        destination = self.root / "report"
        destination.write_bytes(b"old")
        real_stat = publication.os.stat
        claimed_stats = 0
        interrupted = False
        rollback_started = False

        def fail_rollback_link(*_args, **_kwargs):
            raise OSError("injected rollback failure")

        def force_rollback(*_args, **_kwargs):
            nonlocal rollback_started
            rollback_started = True
            raise OSError("force rollback")

        def interrupt_claimed_observation(path, *args, **kwargs):
            nonlocal claimed_stats, interrupted
            if rollback_started and path == "claimed":
                claimed_stats += 1
                if claimed_stats == 1:
                    interrupted = True
                    raise InterruptedError(
                        "injected claimed-member rollback inspection interruption"
                    )
            return real_stat(path, *args, **kwargs)

        with (
            mock.patch.object(
                publication, "_require_platform_support", return_value=None
            ),
            mock.patch.object(
                publication,
                "_link_backup_if_absent",
                side_effect=fail_rollback_link,
            ),
            mock.patch.object(
                publication,
                "_verify_item_published",
                side_effect=force_rollback,
            ),
            mock.patch.object(
                publication.os,
                "stat",
                side_effect=interrupt_claimed_observation,
            ),
        ):
            with self.assertRaisesRegex(
                RollbackIndeterminateError, "claimed-name inspection failed"
            ) as caught:
                publish_outputs([ReportOutput(destination, b"new")])

        self.assertTrue(interrupted)
        self.assertIn(
            "injected claimed-member rollback inspection interruption",
            str(caught.exception),
        )
        self.assertFalse(destination.exists())
        quarantined = list(self.root.rglob(".report-publish-*.quarantine/claimed"))
        self.assertEqual(len(quarantined), 1)
        claimed = quarantined[0]
        self.assertEqual(claimed.read_bytes(), b"new")
        self.assertTrue(claimed.parent.is_dir())
        backups = list(self.root.glob(".report-publish-*.backup"))
        self.assertEqual(len(backups), 1)
        self.assertEqual(backups[0].read_bytes(), b"old")
        expected = tuple(sorted([claimed, *backups], key=os.fspath))
        self.assertEqual(caught.exception.recovery_paths, expected)
        self.assertIn(str(claimed), str(caught.exception))
        self.assertNotIn(claimed.parent, caught.exception.recovery_paths)

        for mode in ("destination", "source", "both"):
            with self.subTest(rollback_claim_fsync_failure=mode):
                parent = self.root / f"rollback-claim-fsync-{mode}"
                parent.mkdir()
                destination = parent / "report"
                destination.write_bytes(b"old")
                events = []
                rollback_claim_started = False
                restore_started = False
                real_fsync = publication.os.fsync
                real_create = publication._create_rollback_quarantine
                quarantine_descriptor = None
                parent_descriptor = None

                def capture_quarantine(parent_handle, public_name):
                    nonlocal quarantine_descriptor, parent_descriptor
                    quarantine = real_create(parent_handle, public_name)
                    quarantine_descriptor = quarantine.descriptor.fileno()
                    parent_descriptor = parent_handle.descriptor
                    return quarantine

                def fail_rollback_claim_fsync(descriptor):
                    nonlocal rollback_claim_started
                    if descriptor == quarantine_descriptor:
                        rollback_claim_started = True
                        events.append("destination")
                        if mode in {"destination", "both"}:
                            raise OSError(
                                "injected rollback destination claim fsync failure"
                            )
                    elif rollback_claim_started and descriptor == parent_descriptor:
                        events.append("source")
                        if mode in {"source", "both"}:
                            raise OSError(
                                "injected rollback source claim fsync failure"
                            )
                    return real_fsync(descriptor)

                def record_restore(_item):
                    nonlocal restore_started
                    restore_started = True

                with (
                    mock.patch.object(
                        publication,
                        "_verify_item_published",
                        side_effect=OSError("force rollback"),
                    ),
                    mock.patch.object(
                        publication,
                        "_create_rollback_quarantine",
                        side_effect=capture_quarantine,
                    ),
                    mock.patch.object(
                        publication.os,
                        "fsync",
                        side_effect=fail_rollback_claim_fsync,
                    ),
                    mock.patch.object(
                        publication,
                        "_before_restore_original",
                        side_effect=record_restore,
                    ),
                ):
                    with self.assertRaises(RollbackIndeterminateError) as sync_caught:
                        publish_outputs([ReportOutput(destination, b"new")])

                claimed_paths = list(
                    parent.rglob(".report-publish-*.quarantine/claimed")
                )
                backups = list(parent.glob(".report-publish-*.backup"))
                self.assertEqual(events, ["destination", "source"])
                self.assertFalse(restore_started)
                self.assertFalse(destination.exists())
                self.assertEqual(len(claimed_paths), 1)
                self.assertEqual(claimed_paths[0].read_bytes(), b"new")
                self.assertEqual(len(backups), 1)
                self.assertEqual(backups[0].read_bytes(), b"old")
                expected = tuple(sorted([*claimed_paths, *backups], key=os.fspath))
                self.assertEqual(sync_caught.exception.recovery_paths, expected)
                if mode in {"destination", "both"}:
                    self.assertIn(
                        "rollback destination claim fsync failure",
                        str(sync_caught.exception),
                    )
                if mode in {"source", "both"}:
                    self.assertIn(
                        "rollback source claim fsync failure",
                        str(sync_caught.exception),
                    )

        parent = self.root / "two-item-retained-then-rebound"
        parent.mkdir()
        first = parent / "a"
        second = parent / "b"
        first.write_bytes(b"old-a")
        second.write_bytes(b"old-b")
        detached_parent = self.root / "two-item-retained-detached"
        real_cleanup = publication._cleanup_rollback_quarantine
        rebound = False
        first_outcome = None

        def retain_first_then_rebind(*args, **kwargs):
            nonlocal rebound, first_outcome
            outcome = real_cleanup(*args, **kwargs)
            if not rebound:
                first_outcome = outcome
                self.assertEqual(
                    outcome.disposition,
                    publication.repository_snapshot.CleanupDisposition.RETAINED,
                )
                os.rename(parent, detached_parent)
                parent.mkdir()
                rebound = True
            return outcome

        with (
            mock.patch.object(
                publication,
                "_verify_item_published",
                side_effect=OSError("force two-item rollback"),
            ),
            mock.patch.object(
                publication,
                "_link_backup_if_absent",
                side_effect=OSError("retain first rollback quarantine"),
            ),
            mock.patch.object(
                publication,
                "_cleanup_rollback_quarantine",
                side_effect=retain_first_then_rebind,
            ),
        ):
            with self.assertRaises(RollbackIndeterminateError) as rebound_caught:
                publish_outputs(
                    [ReportOutput(first, b"new-a"), ReportOutput(second, b"new-b")]
                )

        self.assertTrue(rebound)
        self.assertIsNotNone(first_outcome)
        self.assertEqual(rebound_caught.exception.recovery_paths, ())
        self.assertIn(
            "stable recovery anchor was rebound", str(rebound_caught.exception)
        )
        self.assertTrue(list(detached_parent.rglob(".report-publish-*.quarantine")))

    def test_rollback_continues_after_one_external_replacement(self):
        first = self.root / "a"
        second = self.root / "b"
        first.write_bytes(b"old-a")
        second.write_bytes(b"old-b")
        real_verify = publication._verify_item_published
        injected = False

        def replace_one_then_fail(item, stage):
            nonlocal injected
            if not injected:
                injected = True
                external = self.root / "external"
                external.write_bytes(b"external-a")
                os.replace(external, first)
            return real_verify(item, stage)

        with mock.patch.object(
            publication, "_verify_item_published", replace_one_then_fail
        ):
            with self.assertRaises(RollbackIndeterminateError) as caught:
                publish_outputs(
                    [ReportOutput(first, b"new-a"), ReportOutput(second, b"new-b")]
                )

        self.assertFalse(first.exists())
        quarantined = list(self.root.rglob(".report-publish-*.quarantine/claimed"))
        self.assertEqual(len(quarantined), 1)
        self.assertEqual(quarantined[0].read_bytes(), b"external-a")
        self.assert_file(second, b"old-b")
        backups = list(self.root.glob(".report-publish-*.backup"))
        self.assertEqual(len(backups), 2)
        self.assertEqual({path.read_bytes() for path in backups}, {b"old-a", b"old-b"})
        expected = tuple(sorted([*quarantined, *backups], key=os.fspath))
        self.assertEqual(caught.exception.recovery_paths, expected)

    def test_backup_cleanup_failure_keeps_coherent_new_generation_and_recovery(self):
        first = self.root / "a"
        second = self.root / "b"
        first.write_bytes(b"old-a")
        second.write_bytes(b"old-b")
        real_unlink = publication.os.unlink
        failed = False

        def fail_backup_cleanup(name, *, dir_fd=None):
            nonlocal failed
            if name == "claimed" and not failed:
                failed = True
                raise OSError("injected backup cleanup failure")
            return real_unlink(name, dir_fd=dir_fd)

        with (
            mock.patch.object(
                publication, "_require_platform_support", return_value=None
            ),
            mock.patch.object(
                publication.os,
                "unlink",
                side_effect=fail_backup_cleanup,
            ),
        ):
            with self.assertRaisesRegex(
                TransactionCompleteCleanupError, "transaction complete"
            ) as caught:
                publish_outputs(
                    [ReportOutput(first, b"new-a"), ReportOutput(second, b"new-b")]
                )

        self.assert_file(first, b"new-a")
        self.assert_file(second, b"new-b")
        backups = sorted(self.root.glob(".report-publish-*.backup"))
        self.assertEqual(backups, [])
        quarantined = list(self.root.rglob(".report-publish-*.quarantine/claimed"))
        self.assertEqual(len(quarantined), 1)
        self.assertIn(quarantined[0].read_bytes(), {b"old-a", b"old-b"})
        self.assertEqual(caught.exception.recovery_paths, tuple(quarantined))
        self.assertTrue(all(path.exists() for path in caught.exception.recovery_paths))

        first_parent = self.root / "transaction-a-unaddressable"
        second_parent = self.root / "transaction-b-observable"
        first_parent.mkdir()
        second_parent.mkdir()
        first = first_parent / "report"
        second = second_parent / "report"
        first.write_bytes(b"old-first")
        second.write_bytes(b"old-second")
        detached_first_parent = self.root / "transaction-a-detached"
        rebound = False
        second_candidate = None

        def mutate_after_claim(parent_handle, name, label):
            nonlocal rebound, second_candidate
            if (
                label.startswith("transaction-complete")
                and parent_handle.path == first_parent
                and not rebound
            ):
                os.rename(first_parent, detached_first_parent)
                first_parent.mkdir()
                rebound = True
            elif (
                label.startswith("transaction-complete")
                and parent_handle.path == second_parent
                and second_candidate is None
            ):
                second_candidate = second_parent / name
                second_candidate.write_bytes(b"observable-second")

        with (
            mock.patch.object(
                publication,
                "_after_public_sidecar_claim",
                side_effect=mutate_after_claim,
            ),
        ):
            with self.assertRaises(TransactionCompleteCleanupError) as tainted_caught:
                publish_outputs(
                    [
                        ReportOutput(first, b"new-first"),
                        ReportOutput(second, b"new-second"),
                    ]
                )

        self.assertTrue(rebound)
        self.assertIsNotNone(second_candidate)
        self.assertEqual(second_candidate.read_bytes(), b"observable-second")
        second_claimed = list(
            second_parent.rglob(".report-publish-*.quarantine/claimed")
        )
        self.assertEqual(len(second_claimed), 1)
        self.assertEqual(second_claimed[0].read_bytes(), b"old-second")
        self.assertEqual(tainted_caught.exception.recovery_paths, ())
        self.assertNotIn(str(second_claimed[0]), str(tainted_caught.exception))
        self.assertIn(second_candidate, tainted_caught.exception.candidate_paths)
        self.assertIn("cleanup_arena_binding_rebound", str(tainted_caught.exception))

    def test_backup_cleanup_claim_preserves_foreign_swap_and_fsyncs_parent(self):
        destination = self.root / "report"
        destination.write_bytes(b"old")
        foreign = self.root / "foreign-backup"
        foreign.write_bytes(b"foreign-backup-bytes")
        fsynced = []
        real_fsync = publication._fsync_backup_cleanup_parent

        def swap_backup(parent, name, label):
            if label.startswith("transaction-complete"):
                os.replace(foreign, parent.path / name)

        def record_fsync(parent):
            fsynced.append(parent.path)
            return real_fsync(parent)

        with (
            mock.patch.object(
                publication,
                "_before_public_sidecar_claim",
                side_effect=swap_backup,
            ),
            mock.patch.object(
                publication,
                "_fsync_backup_cleanup_parent",
                side_effect=record_fsync,
            ),
        ):
            with self.assertRaisesRegex(
                TransactionCompleteCleanupError, "coherent new generation retained"
            ) as caught:
                publish_outputs([ReportOutput(destination, b"new")])

        self.assert_file(destination, b"new")
        quarantined = list(self.root.rglob(".report-publish-*.quarantine/claimed"))
        self.assertEqual(len(quarantined), 1)
        self.assertEqual(quarantined[0].read_bytes(), b"foreign-backup-bytes")
        self.assertEqual(caught.exception.recovery_paths, tuple(quarantined))
        self.assertTrue(all(path.exists() for path in caught.exception.recovery_paths))
        self.assertEqual(fsynced, [self.root])

    def test_nested_transaction_cleanup_merges_causal_and_item_recovery_paths(self):
        first = self.root / "a"
        second = self.root / "b"
        first.write_bytes(b"old-a")
        second.write_bytes(b"old-b")
        quarantine = (
            self.root / CLEANUP_ARENA_NAME / ".report-publish-nested.quarantine"
        )
        claimed = quarantine / "claimed"
        cleaned = self.root / ".report-publish-cleaned.backup"

        def fail_nested_cleanup(item, _accumulator):
            quarantine.mkdir(mode=0o700, parents=True, exist_ok=True)
            claimed.write_bytes(b"nested-recovery")
            cleaned.write_bytes(b"cleaned-recovery")
            cleaned.unlink()
            parent_metadata = os.fstat(item.parent.descriptor)
            causal_outcome = publication.repository_snapshot.CleanupOutcome(
                publication.repository_snapshot.CleanupDisposition.RETAINED,
                (claimed,),
                (),
                recovery_anchor_identity=(
                    parent_metadata.st_dev,
                    parent_metadata.st_ino,
                ),
            )
            raise TransactionCompleteCleanupError(
                "injected nested cleanup failure",
                recovery_paths=(cleaned, claimed),
                cleanup_outcomes=(causal_outcome,),
            )

        with (
            mock.patch.object(
                publication,
                "_cleanup_backup_after_transaction",
                side_effect=fail_nested_cleanup,
            ),
            mock.patch.object(
                publication,
                "_require_platform_support",
                return_value=None,
            ),
        ):
            with self.assertRaises(TransactionCompleteCleanupError) as caught:
                publish_outputs(
                    [ReportOutput(first, b"new-a"), ReportOutput(second, b"new-b")]
                )

        backups = list(self.root.glob(".report-publish-*.backup"))
        self.assertEqual(len(backups), 2)
        expected = tuple(sorted([*backups, claimed], key=os.fspath))
        self.assertEqual(caught.exception.recovery_paths, expected)
        self.assertNotIn(cleaned, caught.exception.recovery_paths)
        self.assertEqual(
            {path.read_bytes() for path in caught.exception.recovery_paths},
            {b"old-a", b"old-b", b"nested-recovery"},
        )
        self.assert_file(first, b"new-a")
        self.assert_file(second, b"new-b")

    def test_transaction_recovery_backup_presence_is_tristate(self):
        for mode in ("mismatch", "absent"):
            with self.subTest(mode=mode):
                parent = self.root / f"backup-{mode}"
                parent.mkdir()
                destination = parent / "report"
                destination.write_bytes(b"old")
                quarantine = (
                    parent / CLEANUP_ARENA_NAME / ".report-publish-causal.quarantine"
                )
                claimed = quarantine / "claimed"

                def fail_cleanup(item, _accumulator):
                    assert item.backup_name is not None
                    backup = item.parent.path / item.backup_name
                    if mode == "mismatch":
                        foreign = parent / "foreign-backup"
                        foreign.write_bytes(b"foreign-backup")
                        os.replace(foreign, backup)
                    else:
                        os.unlink(item.backup_name, dir_fd=item.parent.descriptor)
                    quarantine.mkdir(mode=0o700, parents=True)
                    claimed.write_bytes(b"causal-recovery")
                    parent_metadata = os.fstat(item.parent.descriptor)
                    causal_outcome = publication.repository_snapshot.CleanupOutcome(
                        publication.repository_snapshot.CleanupDisposition.RETAINED,
                        (claimed,),
                        (),
                        recovery_anchor_identity=(
                            parent_metadata.st_dev,
                            parent_metadata.st_ino,
                        ),
                    )
                    raise TransactionCompleteCleanupError(
                        "injected nested cleanup failure",
                        recovery_paths=(claimed,),
                        cleanup_outcomes=(causal_outcome,),
                    )

                with mock.patch.object(
                    publication,
                    "_cleanup_backup_after_transaction",
                    side_effect=fail_cleanup,
                ):
                    with self.assertRaises(TransactionCompleteCleanupError) as caught:
                        publish_outputs([ReportOutput(destination, b"new")])

                backups = list(parent.glob(".report-publish-*.backup"))
                expected = [claimed]
                if mode == "mismatch":
                    self.assertEqual(len(backups), 1)
                    self.assertEqual(backups[0].read_bytes(), b"foreign-backup")
                    self.assertEqual(caught.exception.candidate_paths, tuple(backups))
                    self.assertIs(
                        caught.exception.public_candidate,
                        publication.repository_snapshot.PublicCandidate.PRESENT,
                    )
                else:
                    self.assertEqual(backups, [])
                    self.assertEqual(caught.exception.candidate_paths, ())
                self.assertEqual(
                    caught.exception.recovery_paths,
                    tuple(sorted(expected, key=os.fspath)),
                )
                self.assert_file(destination, b"new")

    def test_backup_cleanup_verifies_absence_before_parent_fsync(self):
        first = self.root / "a"
        second = self.root / "b"
        first.write_bytes(b"old-a")
        second.write_bytes(b"old-b")
        events = []
        real_unlink = publication.os.unlink
        real_stat = publication.os.stat
        real_fsync = publication._fsync_backup_cleanup_parent
        backup_claim_started = False

        def record_remove(name, *, dir_fd=None):
            nonlocal backup_claim_started
            if name == "claimed":
                backup_claim_started = True
                events.append(("unlink", name))
            return real_unlink(name, dir_fd=dir_fd)

        def record_absent(name, *args, **kwargs):
            nonlocal backup_claim_started
            try:
                return real_stat(name, *args, **kwargs)
            except FileNotFoundError:
                if backup_claim_started and name == "claimed":
                    events.append(("absent", name))
                    backup_claim_started = False
                raise

        def record_fsync(parent):
            events.append(("fsync", os.fspath(parent.path)))
            return real_fsync(parent)

        with (
            mock.patch.object(
                publication, "_require_platform_support", return_value=None
            ),
            mock.patch.object(
                publication.os,
                "unlink",
                side_effect=record_remove,
            ),
            mock.patch.object(publication.os, "stat", side_effect=record_absent),
            mock.patch.object(
                publication,
                "_fsync_backup_cleanup_parent",
                side_effect=record_fsync,
            ),
        ):
            publish_outputs(
                [ReportOutput(first, b"new-a"), ReportOutput(second, b"new-b")]
            )

        self.assertEqual(
            [kind for kind, _value in events],
            [
                "unlink",
                "absent",
                "unlink",
                "absent",
                "fsync",
            ],
        )
        self.assertEqual(self.sidecars(), [])

        for mode, expected_public in (
            ("present", publication.repository_snapshot.PublicCandidate.PRESENT),
            ("unknown", publication.repository_snapshot.PublicCandidate.UNKNOWN),
        ):
            with self.subTest(final_backup_recheck=mode):
                parent = self.root / f"final-backup-recheck-{mode}"
                parent.mkdir()
                destination = parent / "report"
                destination.write_bytes(b"old")
                real_absent = publication._verify_absent
                real_create = publication._create_rollback_quarantine
                real_record = publication.repository_snapshot.CleanupAccumulator.record
                real_stat = publication.os.stat
                candidate = None
                claimed = None
                unknown_recheck = False
                recorded_outcomes = []

                def capture_quarantine(parent_handle, public_name):
                    nonlocal claimed
                    quarantine = real_create(parent_handle, public_name)
                    if parent_handle.path == parent:
                        claimed = quarantine.claimed_path
                    return quarantine

                def inject_final_observation(parent_handle, name, label):
                    nonlocal candidate, unknown_recheck
                    if label.startswith("transaction-complete"):
                        candidate = parent_handle.path / name
                        if mode == "present":
                            candidate.write_bytes(b"reappeared")
                        else:
                            unknown_recheck = True
                    return real_absent(parent_handle, name, label)

                def fail_unknown_observation(path, *args, **kwargs):
                    if (
                        unknown_recheck
                        and candidate is not None
                        and path == candidate.name
                        and kwargs.get("dir_fd") is not None
                    ):
                        raise OSError("injected final backup inspection failure")
                    return real_stat(path, *args, **kwargs)

                def capture_outcome(accumulator, outcome):
                    recorded_outcomes.append(outcome)
                    return real_record(accumulator, outcome)

                with (
                    mock.patch.object(
                        publication, "_require_platform_support", return_value=None
                    ),
                    mock.patch.object(
                        publication,
                        "_create_rollback_quarantine",
                        side_effect=capture_quarantine,
                    ),
                    mock.patch.object(
                        publication,
                        "_verify_absent",
                        side_effect=inject_final_observation,
                    ),
                    mock.patch.object(
                        publication.os,
                        "stat",
                        side_effect=fail_unknown_observation,
                    ),
                    mock.patch.object(
                        publication.repository_snapshot.CleanupAccumulator,
                        "record",
                        autospec=True,
                        side_effect=capture_outcome,
                    ),
                ):
                    with self.assertRaises(
                        TransactionCompleteCleanupError
                    ) as recheck_caught:
                        publish_outputs([ReportOutput(destination, b"new")])

                self.assertIsNotNone(candidate)
                self.assertIsNotNone(claimed)
                self.assertEqual(recheck_caught.exception.recovery_paths, ())
                self.assertEqual(recheck_caught.exception.candidate_paths, (candidate,))
                self.assertIs(
                    recheck_caught.exception.public_candidate, expected_public
                )
                self.assertNotIn(str(claimed), str(recheck_caught.exception))
                self.assertGreaterEqual(len(recorded_outcomes), 2)
                observation_index, observation_outcome = next(
                    (index, outcome)
                    for index, outcome in enumerate(recorded_outcomes)
                    if any(
                        issue.code
                        in {
                            "cleanup_public_name_reappeared",
                            "cleanup_public_absence_uninspectable",
                        }
                        for issue in outcome.issues
                    )
                )
                cleanup_outcome = recorded_outcomes[observation_index - 1]
                self.assertEqual(observation_outcome.recovery_paths, ())
                self.assertEqual(
                    observation_outcome.arena_identity,
                    cleanup_outcome.arena_identity,
                )
                self.assertEqual(
                    observation_outcome.recovery_anchor_identity,
                    cleanup_outcome.recovery_anchor_identity,
                )

    def test_backup_cleanup_fsync_failure_keeps_new_generation_without_rollback(self):
        for mode in ("destination", "source", "both"):
            with self.subTest(claim_fsync_failure=mode):
                parent = self.root / f"backup-claim-fsync-{mode}"
                parent.mkdir()
                destination = parent / "report"
                destination.write_bytes(b"old")
                events = []
                claim_started = False
                after_claim_called = False
                absence_checked = False
                real_fsync = publication.os.fsync
                real_absent = publication._verify_absent
                quarantine_descriptor = None
                parent_descriptor = None
                real_create = publication._create_rollback_quarantine

                def capture_quarantine(parent_handle, public_name):
                    nonlocal quarantine_descriptor, parent_descriptor
                    quarantine = real_create(parent_handle, public_name)
                    quarantine_descriptor = quarantine.descriptor.fileno()
                    parent_descriptor = parent_handle.descriptor
                    return quarantine

                def fail_claim_fsync(descriptor):
                    nonlocal claim_started
                    if descriptor == quarantine_descriptor:
                        claim_started = True
                        events.append("destination")
                        if mode in {"destination", "both"}:
                            raise OSError("injected destination claim fsync failure")
                    elif claim_started and descriptor == parent_descriptor:
                        events.append("source")
                        if mode in {"source", "both"}:
                            raise OSError("injected source claim fsync failure")
                    return real_fsync(descriptor)

                def record_after_claim(*_args):
                    nonlocal after_claim_called
                    after_claim_called = True

                def record_absence(parent_handle, name, label):
                    nonlocal absence_checked
                    if label.startswith("transaction-complete"):
                        absence_checked = True
                    return real_absent(parent_handle, name, label)

                with (
                    mock.patch.object(
                        publication,
                        "_create_rollback_quarantine",
                        side_effect=capture_quarantine,
                    ),
                    mock.patch.object(
                        publication.os, "fsync", side_effect=fail_claim_fsync
                    ),
                    mock.patch.object(
                        publication,
                        "_fsync_backup_cleanup_parent",
                        return_value=None,
                    ),
                    mock.patch.object(
                        publication,
                        "_after_public_sidecar_claim",
                        side_effect=record_after_claim,
                    ),
                    mock.patch.object(
                        publication,
                        "_verify_absent",
                        side_effect=record_absence,
                    ),
                ):
                    with self.assertRaises(TransactionCompleteCleanupError) as caught:
                        publish_outputs([ReportOutput(destination, b"new")])

                claimed = list(parent.rglob(".report-publish-*.quarantine/claimed"))
                self.assertEqual(events, ["destination", "source"])
                self.assertFalse(after_claim_called)
                self.assertFalse(absence_checked)
                self.assert_file(destination, b"new")
                self.assertEqual(len(claimed), 1)
                self.assertEqual(claimed[0].read_bytes(), b"old")
                self.assertEqual(caught.exception.recovery_paths, tuple(claimed))
                cleanup_error = str(caught.exception.__cause__)
                if mode in {"destination", "both"}:
                    self.assertIn("destination claim fsync failure", cleanup_error)
                if mode in {"source", "both"}:
                    self.assertIn("source claim fsync failure", cleanup_error)

    def test_changed_created_parent_is_preserved_and_reported(self):
        created_parent = self.root / "created"
        destination = created_parent / "report"
        real_verify_parent = publication._verify_parent
        injected = False

        def swap_parent(parent, stage):
            nonlocal injected
            if not injected:
                injected = True
                moved = self.root / "moved-owned-parent"
                created_parent.rename(moved)
                created_parent.mkdir()
                (created_parent / "external").write_bytes(b"external")
            return real_verify_parent(parent, stage)

        with mock.patch.object(publication, "_verify_parent", swap_parent):
            with self.assertRaisesRegex(
                RollbackIndeterminateError, "claimed-name inspection failed"
            ) as caught:
                publish_outputs([ReportOutput(destination, b"new")])

        claimed = list(self.root.rglob(".report-publish-*.created/claimed"))
        self.assertEqual(len(claimed), 1)
        self.assertEqual((claimed[0] / "external").read_bytes(), b"external")
        self.assertEqual(caught.exception.recovery_paths, ())
        self.assertTrue(caught.exception.candidate_paths)
        self.assertTrue((self.root / "moved-owned-parent").is_dir())

        rebound_anchor = self.root / "rebound-anchor"
        rebound_anchor.mkdir()
        rebound_destination = rebound_anchor / "created" / "report"
        moved_anchor = self.root / "moved-rebound-anchor"
        rebound_injected = False

        def rebind_stable_anchor(parent, stage):
            nonlocal rebound_injected
            if not rebound_injected:
                rebound_injected = True
                rebound_anchor.rename(moved_anchor)
                rebound_anchor.mkdir()
                (rebound_anchor / "external").write_bytes(b"external")
            return real_verify_parent(parent, stage)

        with mock.patch.object(publication, "_verify_parent", rebind_stable_anchor):
            with self.assertRaisesRegex(
                RollbackIndeterminateError, "stable recovery anchor was rebound"
            ) as rebound_caught:
                publish_outputs([ReportOutput(rebound_destination, b"new")])

        self.assertEqual(rebound_caught.exception.recovery_paths, ())
        self.assertIn(
            rebound_anchor / "created",
            rebound_caught.exception.candidate_paths,
        )
        self.assertTrue((moved_anchor / "created").is_dir())
        self.assertEqual((rebound_anchor / "external").read_bytes(), b"external")


if __name__ == "__main__":
    unittest.main()
