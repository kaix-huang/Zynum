# Copyright (C) 2026 Zynum contributors
# SPDX-License-Identifier: LGPL-3.0-or-later

from __future__ import annotations

import dataclasses
import importlib.util
import inspect
import json
import os
import subprocess
import sys
import tarfile
import tempfile
import typing
import unittest
from pathlib import Path, PureWindowsPath
from unittest import mock

ROOT = Path(__file__).resolve().parents[3]
MODULE_PATH = ROOT / "tools/check_package_paths.py"
SPEC = importlib.util.spec_from_file_location("check_package_paths", MODULE_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("unable to load package checker")
checker = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = checker
SPEC.loader.exec_module(checker)

_TRANSACTION_SEAM_TESTS = frozenset(
    {
        "test_archive_bytes_and_metadata_are_normalized",
        "test_archive_rejects_content_change_during_read",
        "test_archive_rejects_member_replaced_after_enumeration",
        "test_archive_rejects_regular_file_identity_change",
        "test_archive_rejects_same_inode_content_change_with_restored_metadata",
        "test_distribution_license_documents_are_complete_and_archivable",
        "test_real_repository_archive_has_no_ignored_or_private_metadata",
    }
)


def run_git(root: Path, *arguments: str) -> None:
    subprocess.run(("git", *arguments), cwd=root, check=True)


def commit_all(root: Path) -> None:
    run_git(root, "add", "--all")
    run_git(
        root,
        "-c",
        "user.name=Zynum Tests",
        "-c",
        "user.email=tests@zynum.invalid",
        "commit",
        "-qm",
        "fixture",
    )


def cleanup_arena(root: Path) -> Path:
    arena = root / f".zynum-cleanup-v2-{os.geteuid()}"
    if not arena.is_dir():
        raise AssertionError(f"expected canonical cleanup arena under {root.resolve()}")
    return arena


class PackageArchiveTests(unittest.TestCase):
    def setUp(self) -> None:
        self._strict_create_archive = checker.create_archive
        if self._testMethodName in _TRANSACTION_SEAM_TESTS:
            checker.create_archive = checker._test_only_materialize_archive_transaction

    def tearDown(self) -> None:
        checker.create_archive = self._strict_create_archive

    @staticmethod
    def _publishable_fixture(root: Path) -> object:
        payload = root / "payload"
        payload.mkdir()
        (payload / "data.txt").write_text("data\n", encoding="utf-8")
        manifest = root / "build.zig.zon"
        manifest.write_text(
            '.{ .paths = .{"build.zig.zon", "payload"} }\n',
            encoding="utf-8",
        )
        run_git(root, "init", "-q")
        commit_all(root)
        paths = checker.package_paths(manifest, root=root)
        return checker.publishable_package_files(root, paths)

    def test_package_path_forbidden_names_are_matched_by_component(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            allowed = root / ".github/workflows"
            allowed.mkdir(parents=True)
            checker.validate_path(root, ".github/workflows")
            forbidden = root / ".git/hooks"
            forbidden.mkdir(parents=True)
            with self.assertRaisesRegex(ValueError, "forbidden package path"):
                checker.validate_path(root, ".git/hooks")

    def test_git_allowlist_excludes_ignored_files(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            (root / "payload").mkdir()
            (root / ".gitignore").write_text(
                ".DS_Store\n__pycache__/\n*.pyc\n", encoding="utf-8"
            )
            (root / "payload/keep.txt").write_text("keep\n", encoding="utf-8")
            for special in ("space name.txt", "unicode-λ.txt", "line\nbreak.txt"):
                (root / "payload" / special).write_text("keep\n", encoding="utf-8")
            (root / "payload/.DS_Store").write_bytes(b"host")
            (root / "payload/__pycache__").mkdir()
            (root / "payload/__pycache__/cache.pyc").write_bytes(b"host")
            subprocess.run(("git", "init", "-q"), cwd=root, check=True)
            files = checker.package_files(root, ["payload"])
            self.assertEqual(
                sorted(
                    [
                        "payload/keep.txt",
                        "payload/space name.txt",
                        "payload/unicode-λ.txt",
                        "payload/line\nbreak.txt",
                    ]
                ),
                [member.path for member in files],
            )
            subprocess.run(
                ("git", "add", "-f", "payload/.DS_Store"), cwd=root, check=True
            )
            with self.assertRaisesRegex(ValueError, "local metadata"):
                checker.package_files(root, ["payload"])

    def test_git_allowlist_omits_tracked_files_deleted_from_worktree(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            payload = root / "payload"
            payload.mkdir()
            kept = payload / "kept.txt"
            deleted = payload / "deleted.txt"
            kept.write_text("kept", encoding="utf-8")
            deleted.write_text("deleted", encoding="utf-8")
            subprocess.run(("git", "init", "-q"), cwd=root, check=True)
            subprocess.run(("git", "add", "payload"), cwd=root, check=True)
            deleted.unlink()
            files = checker.package_files(root, ["payload"])
            self.assertEqual(["payload/kept.txt"], [item.path for item in files])

    def test_git_allowlist_rejects_deleted_file_reappearing_after_boundary(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            payload = root / "payload"
            payload.mkdir()
            target = payload / "deleted.txt"
            target.write_text("deleted", encoding="utf-8")
            run_git(root, "init", "-q")
            run_git(root, "add", "payload/deleted.txt")
            target.unlink()
            repository = checker.repository_git.open_repository(root)
            self.assertIsNotNone(repository)
            assert repository is not None
            real_ls_files = repository.ls_files

            def list_then_recreate(_: object, paths: object) -> tuple[str, ...]:
                listed = real_ls_files(paths)  # type: ignore[arg-type]
                target.write_text("replacement", encoding="utf-8")
                return listed

            with (
                mock.patch.object(
                    checker.repository_git.RepositoryGit,
                    "ls_files",
                    autospec=True,
                    side_effect=list_then_recreate,
                ),
                self.assertRaisesRegex(ValueError, "appeared after snapshot"),
            ):
                checker.package_files(root, ["payload"], repository=repository)

    def test_git_allowlist_rejects_disappearance_after_presence_boundary(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            payload = root / "payload"
            payload.mkdir()
            target = payload / "target.txt"
            target.write_text("present", encoding="utf-8")
            subprocess.run(("git", "init", "-q"), cwd=root, check=True)

            def disappear(*_: object) -> object:
                target.unlink()
                raise FileNotFoundError(target)

            with mock.patch.object(checker, "_package_member", side_effect=disappear):
                with self.assertRaises(FileNotFoundError):
                    checker.package_files(root, ["payload"])

    def test_git_allowlist_rejects_real_disappearance_after_listing(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            payload = root / "payload"
            payload.mkdir()
            target = payload / "target.txt"
            target.write_text("present", encoding="utf-8")
            run_git(root, "init", "-q")
            repository = checker.repository_git.open_repository(root)
            self.assertIsNotNone(repository)
            assert repository is not None
            real_ls_files = repository.ls_files

            def list_then_delete(_: object, paths: object) -> tuple[str, ...]:
                listed = real_ls_files(paths)  # type: ignore[arg-type]
                target.unlink()
                return listed

            with (
                mock.patch.object(
                    checker.repository_git.RepositoryGit,
                    "ls_files",
                    autospec=True,
                    side_effect=list_then_delete,
                ),
                self.assertRaisesRegex(ValueError, "changed during enumeration"),
            ):
                checker.package_files(root, ["payload"], repository=repository)

    def test_archive_mode_rejects_malformed_duplicate_and_overlapping_paths(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            payload = root / "payload"
            payload.mkdir()
            (payload / "keep.txt").write_text("keep\n", encoding="utf-8")
            cases = (
                (["../payload"], "invalid package path"),
                (["payload//keep.txt"], "invalid package path"),
                ([":(glob)payload/*"], "invalid package path"),
                (["payload/*.txt"], "invalid package path"),
                (["payload", "payload"], "duplicate package path"),
                (["payload", "payload/keep.txt"], "overlapping package paths"),
            )
            for paths, message in cases:
                with self.subTest(paths=paths):
                    with self.assertRaisesRegex(ValueError, message):
                        checker.package_files(root, paths)

    def test_archive_mode_rejects_symlinks_and_nonregular_files(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            payload = root / "payload"
            payload.mkdir()
            external = root / "external.txt"
            external.write_text("external\n", encoding="utf-8")
            link = payload / "link.txt"
            try:
                link.symlink_to(external)
            except OSError as exc:
                self.skipTest(f"symlinks are unavailable: {exc}")
            with self.assertRaisesRegex(ValueError, "symlink"):
                checker.package_files(root, ["payload"])
            link.unlink()

            fifo = payload / "pipe"
            try:
                os.mkfifo(fifo)
            except (AttributeError, OSError) as exc:
                self.skipTest(f"FIFOs are unavailable: {exc}")
            with self.assertRaisesRegex(ValueError, "regular file"):
                checker.package_files(root, ["payload"])

    def test_archive_mode_rejects_forbidden_local_metadata(self) -> None:
        cases = (
            ".DS_Store",
            "__pycache__/cache.pyc",
            ".zig-cache/cache.bin",
        )
        for relative in cases:
            with self.subTest(relative=relative):
                with tempfile.TemporaryDirectory() as name:
                    root = Path(name)
                    target = root / "payload" / relative
                    target.parent.mkdir(parents=True)
                    target.write_bytes(b"local")
                    with self.assertRaisesRegex(ValueError, "local metadata"):
                        checker.package_files(root, ["payload"])

    def test_repository_mode_fails_closed_on_invalid_git_marker(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            (root / "payload").mkdir()
            (root / "payload/data.txt").write_text("data\n", encoding="utf-8")
            (root / ".git").write_text("not a gitfile\n", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "top-level verification"):
                checker.package_files(root, ["payload"])

    def test_repository_mode_rejects_git_top_level_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            (root / "payload").mkdir()
            (root / "payload/data.txt").write_text("data\n", encoding="utf-8")
            (root / ".git").mkdir()
            reported_root = root.parent
            result = subprocess.CompletedProcess(
                args=("git",),
                returncode=0,
                stdout=os.fsencode(reported_root) + b"\n",
                stderr=b"",
            )
            with (
                mock.patch.dict(os.environ, {}, clear=True),
                mock.patch.object(
                    checker.repository_git.RepositoryGit, "run", return_value=result
                ),
                self.assertRaisesRegex(RuntimeError, "exact Git worktree top-level"),
            ):
                checker.package_files(root, ["payload"])

    def test_repository_git_marker_is_bound_before_executable_resolution(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            run_git(root, "init", "-q")
            original = root / ".git-original"

            def replace_marker(_environment: object) -> Path:
                (root / ".git").rename(original)
                (root / ".git").write_text(
                    "gitdir: ../redirected/.git\n", encoding="utf-8"
                )
                return Path("/usr/bin/git")

            with (
                mock.patch.dict(os.environ, {}, clear=True),
                mock.patch.object(
                    checker.repository_git,
                    "_git_executable",
                    side_effect=replace_marker,
                ),
                self.assertRaisesRegex(RuntimeError, "Git marker changed"),
            ):
                checker.repository_git.open_repository(root)

    def test_repository_git_output_and_path_counts_are_bounded(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "path count limit"):
            checker.repository_git._decode_path_list(b"a\0b\0", "fixture", max_paths=1)
        with self.assertRaisesRegex(RuntimeError, "output limit"):
            checker.repository_git._decode_path_list(b"ab\0", "fixture", max_bytes=2)

        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            executable = root / "emit.py"
            executable.write_text(
                f"#!{sys.executable}\nimport os\nos.write(1, b'x' * 4096)\n",
                encoding="utf-8",
            )
            executable.chmod(0o700)
            repository = checker.repository_git.RepositoryGit(root, executable, {})
            with self.assertRaisesRegex(RuntimeError, "output limit"):
                repository.run(
                    (), operation="bounded fixture", check=False, stdout_limit=32
                )

    def test_repository_git_strictly_decodes_one_top_level_path_line(self) -> None:
        invalid = (
            b"",
            b"relative/path\n",
            b"/first\n/second\n",
            b"/path\r\n",
            b"/path\0suffix\n",
            b"\xff\n",
        )
        for stdout in invalid:
            with self.subTest(stdout=stdout), self.assertRaises(RuntimeError):
                checker.repository_git._canonical_top_level(stdout)

    def test_repository_git_accepts_windows_separator_and_case_spellings(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            canonical = Path(name).resolve(strict=True)

            class SimulatedWindowsPath:
                def __init__(self, value: str) -> None:
                    self.value = PureWindowsPath(value)

                def is_absolute(self) -> bool:
                    return self.value.is_absolute()

                def resolve(self, *, strict: bool) -> Path:
                    self.assert_strict(strict)
                    normalized = self.value.as_posix().casefold()
                    if normalized != "c:/source/zynum":
                        raise FileNotFoundError(self.value)
                    return canonical

                @staticmethod
                def assert_strict(strict: bool) -> None:
                    if not strict:
                        raise AssertionError("top-level path was not resolved strictly")

            for reported in (b"C:/Source/Zynum\n", b"c:\\source\\ZYNUM\n"):
                with self.subTest(reported=reported):
                    observed = checker.repository_git._canonical_top_level(
                        reported, path_factory=SimulatedWindowsPath
                    )
                    self.assertTrue(
                        checker.repository_git._same_directory(canonical, observed)
                    )

    def test_repository_mode_rejects_every_ambient_git_name(self) -> None:
        variables = (
            "GIT_CONFIG_COUNT",
            "GIT_CONFIG_PARAMETERS",
            "GIT_CONFIG_GLOBAL",
            "GIT_CONFIG_SYSTEM",
            "GIT_FUTURE",
        )
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            (root / "payload").mkdir()
            (root / "payload/data.txt").write_text("data\n", encoding="utf-8")
            run_git(root, "init", "-q")
            for variable in variables:
                with (
                    self.subTest(variable=variable),
                    mock.patch.dict(os.environ, {variable: "private-value"}),
                ):
                    with self.assertRaisesRegex(RuntimeError, variable) as raised:
                        checker.package_files(root, ["payload"])
                    self.assertNotIn("private-value", str(raised.exception))

    def test_archive_mode_ignores_ambient_git_names_without_invoking_git(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            (root / "payload").mkdir()
            (root / "payload/data.txt").write_text("data\n", encoding="utf-8")
            environment = {
                "GIT_CONFIG_COUNT": "malformed",
                "GIT_CONFIG_PARAMETERS": "private-value",
                "GIT_CONFIG_GLOBAL": "/private/global",
                "GIT_CONFIG_SYSTEM": "/private/system",
                "GIT_FUTURE": "private-value",
            }
            with (
                mock.patch.dict(os.environ, environment),
                mock.patch.object(
                    checker.repository_git.subprocess,
                    "run",
                    side_effect=AssertionError("archive mode invoked Git"),
                ),
            ):
                files = checker.package_files(root, ["payload"])
            self.assertEqual(["payload/data.txt"], [member.path for member in files])

    def test_repository_git_ignores_parent_home_and_global_configuration(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as name:
            base = Path(name)
            root = base / "repository"
            home = base / "parent-home"
            root.mkdir()
            home.mkdir()
            (root / "payload").mkdir()
            (root / "payload/data.txt").write_text("data\n", encoding="utf-8")
            excluded = base / "parent-excludes"
            excluded.write_text("*.txt\n", encoding="utf-8")
            (home / ".gitconfig").write_text(
                f"[core]\n\texcludesFile = {excluded}\n",
                encoding="utf-8",
            )
            run_git(root, "init", "-q")
            with mock.patch.dict(
                os.environ,
                {"HOME": str(home), "XDG_CONFIG_HOME": str(home / "xdg")},
            ):
                files = checker.package_files(root, ["payload"])
            self.assertEqual(["payload/data.txt"], [member.path for member in files])

    def test_repository_git_pins_worktree_excludes_and_fsmonitor_policy(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            base = Path(name)
            root = base / "repository"
            other_worktree = base / "other-worktree"
            root.mkdir()
            other_worktree.mkdir()
            payload = root / "payload"
            payload.mkdir()
            (payload / "keep.txt").write_text("keep\n", encoding="utf-8")
            (payload / "external.txt").write_text("external\n", encoding="utf-8")
            (payload / "info.txt").write_text("info\n", encoding="utf-8")
            external_excludes = base / "external-excludes"
            external_excludes.write_text("payload/external.txt\n", encoding="utf-8")
            run_git(root, "init", "-q")
            run_git(root, "config", "core.worktree", str(other_worktree))
            run_git(root, "config", "core.excludesFile", str(external_excludes))
            run_git(root, "config", "core.fsmonitor", str(base / "must-not-run"))
            run_git(root, "config", "core.untrackedCache", "true")
            (root / ".git/info/exclude").write_text(
                "payload/info.txt\n", encoding="utf-8"
            )

            repository = checker.repository_git.open_repository(root)
            self.assertIsNotNone(repository)
            assert repository is not None
            command = repository._command(("status", "--short"))
            self.assertIn(f"--work-tree={repository.root}", command)
            self.assertIn(f"core.excludesFile={os.devnull}", command)
            self.assertIn("core.fsmonitor=false", command)
            self.assertIn("core.untrackedCache=false", command)
            files = checker.package_files(root, ["payload"], repository=repository)
            self.assertEqual(
                ["payload/external.txt", "payload/keep.txt"],
                [member.path for member in files],
            )

    def test_repository_git_supports_an_exact_linked_worktree(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            base = Path(name)
            primary = base / "primary"
            linked = base / "linked"
            primary.mkdir()
            (primary / "payload").mkdir()
            (primary / "payload/data.txt").write_text("data\n", encoding="utf-8")
            run_git(primary, "init", "-q")
            run_git(primary, "add", "payload/data.txt")
            run_git(
                primary,
                "-c",
                "user.name=Zynum Test",
                "-c",
                "user.email=test.invalid@example.invalid",
                "commit",
                "-qm",
                "fixture",
            )
            run_git(primary, "worktree", "add", "-q", "--detach", str(linked))
            self.assertTrue((linked / ".git").is_file())
            files = checker.package_files(linked, ["payload"])
            self.assertEqual(["payload/data.txt"], [member.path for member in files])

    def test_package_paths_accepts_full_static_zon_syntax(self) -> None:
        source = r"""
            .{
                .name = .fixture,
                // Multiple strings on one line and comments are valid ZON.
                .paths = .{ "alpha", "two\x2fparts", // layout is flexible
                    "unicode-\u{03bb}" },
                .version = "0.0.0",
            }
        """
        with tempfile.TemporaryDirectory() as name:
            zon = Path(name) / "build.zig.zon"
            zon.write_text(source, encoding="utf-8")
            paths = checker.package_paths(zon)
            self.assertEqual(("alpha", "two/parts", "unicode-λ"), tuple(paths))
            self.assertEqual(Path(name).resolve(), paths.root)
            self.assertEqual("alpha", paths[0])
            self.assertEqual(("two/parts", "unicode-λ"), paths[1:])
            self.assertIn("unicode-λ", paths)
            self.assertEqual(3, len(paths))

    def test_package_paths_rejects_manifest_leaf_and_parent_symlinks(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            base = Path(name)
            root = base / "repository"
            external = base / "external"
            root.mkdir()
            external.mkdir()
            external_manifest = external / "build.zig.zon"
            external_manifest.write_text('.{ .paths = .{"payload"} }', encoding="utf-8")
            try:
                (root / "build.zig.zon").symlink_to(external_manifest)
            except OSError as exc:
                self.skipTest(f"symlinks are unavailable: {exc}")
            with self.assertRaisesRegex(ValueError, "safely"):
                checker.package_paths(root / "build.zig.zon", root=root)

            (root / "build.zig.zon").unlink()
            (root / "config").symlink_to(external, target_is_directory=True)
            with self.assertRaisesRegex(ValueError, "canonical root"):
                checker.package_paths(root / "config/build.zig.zon", root=root)

    def test_package_paths_rejects_special_manifest_file(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            manifest = root / "build.zig.zon"
            try:
                os.mkfifo(manifest)
            except (AttributeError, OSError) as exc:
                self.skipTest(f"FIFOs are unavailable: {exc}")
            with self.assertRaisesRegex(ValueError, "regular file"):
                checker.package_paths(manifest, root=root)

    def test_package_paths_rejects_nested_manifest_before_read(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            base = Path(name)
            root = base / "repository"
            external = base / "external"
            config = root / "config"
            config.mkdir(parents=True)
            external.mkdir()
            manifest = config / "build.zig.zon"
            manifest.write_text('.{ .paths = .{"payload"} }', encoding="utf-8")
            (external / "build.zig.zon").write_text(
                '.{ .paths = .{"private"} }', encoding="utf-8"
            )
            with (
                mock.patch.object(
                    checker.repository_snapshot,
                    "_read_regular_file",
                    side_effect=AssertionError("nested manifest was read"),
                ),
                self.assertRaisesRegex(ValueError, "canonical root"),
            ):
                checker.package_paths(manifest, root=root)

    def test_package_paths_rejects_manifest_leaf_replaced_during_parsing(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            manifest = root / "build.zig.zon"
            manifest.write_text('.{ .paths = .{"payload"} }', encoding="utf-8")
            result = subprocess.CompletedProcess(
                args=("zig",),
                returncode=0,
                stdout=b'{"schema_version":1,"zig_version":"0.16.0","paths":["payload"]}\n',
                stderr=b"",
            )

            def replace_leaf(*_: object, **__: object) -> object:
                manifest.rename(root / "original-build.zig.zon")
                manifest.write_text('.{ .paths = .{"private"} }', encoding="utf-8")
                return result

            with (
                mock.patch.object(checker.subprocess, "run", side_effect=replace_leaf),
                self.assertRaisesRegex(ValueError, "changed after parsing"),
            ):
                checker.package_paths(manifest, root=root)

    def test_package_paths_rejects_same_inode_change_during_manifest_read(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            manifest = root / "build.zig.zon"
            manifest.write_text('.{ .paths = .{"payload"} }', encoding="utf-8")
            original = manifest.stat()
            replacement = '.{ .paths = .{"private"} }'
            self.assertEqual(original.st_size, len(replacement.encode("utf-8")))
            real_read = checker.repository_snapshot._read_regular_file
            reads = 0

            def mutate_manifest(
                stream: object, *, capture_bytes: bool, frozen_size: int
            ) -> tuple[str, bytes | None]:
                nonlocal reads
                result = real_read(
                    stream,
                    capture_bytes=capture_bytes,
                    frozen_size=frozen_size,
                )
                reads += 1
                if reads == 1:
                    manifest.write_text(replacement, encoding="utf-8")
                    os.utime(
                        manifest,
                        ns=(original.st_atime_ns, original.st_mtime_ns),
                    )
                return result

            with (
                mock.patch.object(
                    checker.repository_snapshot,
                    "_read_regular_file",
                    side_effect=mutate_manifest,
                ),
                self.assertRaisesRegex(
                    ValueError,
                    r"changed (?:while being read|after snapshot)",
                ),
            ):
                checker.package_paths(manifest, root=root)

    def test_manifest_snapshot_rejects_change_between_parse_and_enumeration(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            (root / "payload").mkdir()
            (root / "payload/data.txt").write_text("data\n", encoding="utf-8")
            manifest = root / "build.zig.zon"
            manifest.write_text(
                '.{ .paths = .{"build.zig.zon", "payload"} } // A\n',
                encoding="utf-8",
            )
            run_git(root, "init", "-q")
            paths = checker.package_paths(manifest, root=root)
            original = manifest.stat()
            manifest.write_text(
                '.{ .paths = .{"build.zig.zon", "payload"} } // B\n',
                encoding="utf-8",
            )
            os.utime(manifest, ns=(original.st_atime_ns, original.st_mtime_ns))
            with self.assertRaisesRegex(ValueError, "changed after parsing"):
                checker.package_files(root, paths)

    def test_package_paths_are_immutable(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            (root / "payload").mkdir()
            (root / "payload/data.txt").write_text("data\n", encoding="utf-8")
            manifest = root / "build.zig.zon"
            manifest.write_text(
                '.{ .paths = .{"build.zig.zon", "payload"} }\n',
                encoding="utf-8",
            )
            run_git(root, "init", "-q")
            paths = checker.package_paths(manifest, root=root)
            with self.assertRaises(TypeError):
                paths[0] = "payload"
            with self.assertRaises(AttributeError):
                paths.append("payload")
            with self.assertRaises(dataclasses.FrozenInstanceError):
                paths.paths = ("payload",)

    def test_package_paths_declares_and_returns_immutable_type_state(self) -> None:
        signature = inspect.signature(checker.package_paths, eval_str=True)
        hints = typing.get_type_hints(checker.package_paths)
        self.assertIs(checker.PackagePaths, signature.return_annotation)
        self.assertIs(checker.PackagePaths, hints["return"])

        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            (root / "payload").mkdir()
            (root / "payload/data.txt").write_text("data\n", encoding="utf-8")
            manifest = root / "build.zig.zon"
            manifest.write_text(
                '.{ .paths = .{"build.zig.zon", "payload"} }\n',
                encoding="utf-8",
            )
            run_git(root, "init", "-q")
            commit_all(root)
            paths = checker.package_paths(manifest, root=root)
            self.assertIs(type(paths), checker.PackagePaths)
            self.assertNotIsInstance(paths, list)
            for mutation in (
                "append",
                "clear",
                "extend",
                "insert",
                "pop",
                "remove",
                "reverse",
                "sort",
            ):
                with self.subTest(mutation=mutation):
                    self.assertFalse(hasattr(paths, mutation))
            snapshot = checker.publishable_package_files(root, paths)
            self.assertIs(type(snapshot), checker.PublishablePackageSnapshot)

    def test_manifest_snapshot_rejects_change_between_enumeration_and_archive(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            (root / "payload").mkdir()
            (root / "payload/data.txt").write_text("data\n", encoding="utf-8")
            manifest = root / "build.zig.zon"
            manifest.write_text(
                '.{ .paths = .{"build.zig.zon", "payload"} } // A\n',
                encoding="utf-8",
            )
            run_git(root, "init", "-q")
            commit_all(root)
            paths = checker.package_paths(manifest, root=root)
            files = checker.publishable_package_files(root, paths)
            original = manifest.stat()
            manifest.write_text(
                '.{ .paths = .{"build.zig.zon", "payload"} } // B\n',
                encoding="utf-8",
            )
            os.utime(manifest, ns=(original.st_atime_ns, original.st_mtime_ns))
            destination = root / "source.tar.gz"
            with self.assertRaisesRegex(
                ValueError, "content changed before archive destination preparation"
            ):
                checker.create_archive(root, files, destination)
            self.assertFalse(destination.exists())

    def test_package_paths_rejects_nonstatic_or_malformed_zon(self) -> None:
        cases = {
            "missing": '.{ .description = ".paths = .{ \\"fake\\" }" }',
            "duplicate-field": '.{ .paths = .{"a"}, .paths = .{"b"} }',
            "duplicate-path": '.{ .paths = .{"a", "a"} }',
            "empty-list": ".{ .paths = .{} }",
            "empty-path": '.{ .paths = .{""} }',
            "dynamic": ".{ .paths = makePaths() }",
            "non-string": ".{ .paths = .{1} }",
            "trailing": '.{ .paths = .{"a"} } trailing',
            "malformed": '.{ .paths = .{"a" ',
        }
        for label, source in cases.items():
            with self.subTest(case=label), tempfile.TemporaryDirectory() as name:
                zon = Path(name) / "build.zig.zon"
                zon.write_text(source, encoding="utf-8")
                with self.assertRaises(ValueError):
                    checker.package_paths(zon)

    def test_package_paths_rejects_oversized_input_before_execution(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            zon = Path(name) / "build.zig.zon"
            zon.write_bytes(b" " * (checker.MAX_ZON_BYTES + 1))
            with (
                mock.patch.object(
                    checker.subprocess,
                    "run",
                    side_effect=AssertionError("oversized input invoked Zig"),
                ),
                self.assertRaisesRegex(ValueError, "input limit"),
            ):
                checker.package_paths(zon)

    def test_package_path_decoder_accepts_compatible_zig_patch_releases(
        self,
    ) -> None:
        for version in ("0.16.0", "0.16.1", "0.16.999"):
            with self.subTest(version=version):
                output = (
                    json.dumps(
                        {
                            "schema_version": 1,
                            "zig_version": version,
                            "paths": ["payload"],
                        },
                        separators=(",", ":"),
                    ).encode()
                    + b"\n"
                )
                self.assertEqual(["payload"], checker._decode_package_paths(output))

    def test_package_path_decoder_rejects_incompatible_or_malformed_zig_versions(
        self,
    ) -> None:
        versions = (
            "0.15.9",
            "0.17.0",
            "1.16.0",
            "0.16",
            "0.16.0.1",
            "0.16.01",
            "0.16.0-dev.1",
            16,
        )
        for version in versions:
            with self.subTest(version=version):
                output = (
                    json.dumps(
                        {
                            "schema_version": 1,
                            "zig_version": version,
                            "paths": ["payload"],
                        },
                        separators=(",", ":"),
                    ).encode()
                    + b"\n"
                )
                with self.assertRaisesRegex(RuntimeError, "incompatible Zig version"):
                    checker._decode_package_paths(output)

    def test_package_paths_rejects_untrusted_helper_output(self) -> None:
        valid = {
            "schema_version": 1,
            "zig_version": "0.16.0",
            "paths": ["payload"],
        }
        invalid_outputs = (
            b"{}",
            b"{}\n\n",
            b"not-json\n",
            json.dumps({**valid, "extra": True}, separators=(",", ":")).encode()
            + b"\n",
            b'{"schema_version":1,"schema_version":1,"zig_version":"0.16.0","paths":["payload"]}\n',
            json.dumps(
                {**valid, "schema_version": True}, separators=(",", ":")
            ).encode()
            + b"\n",
            json.dumps({**valid, "schema_version": 2}, separators=(",", ":")).encode()
            + b"\n",
            json.dumps(
                {**valid, "zig_version": "0.17.0"}, separators=(",", ":")
            ).encode()
            + b"\n",
            json.dumps({**valid, "paths": "payload"}, separators=(",", ":")).encode()
            + b"\n",
            json.dumps({**valid, "paths": []}, separators=(",", ":")).encode() + b"\n",
            json.dumps(
                {**valid, "paths": ["payload", "payload"]}, separators=(",", ":")
            ).encode()
            + b"\n",
            json.dumps({**valid, "paths": [1]}, separators=(",", ":")).encode() + b"\n",
            b'{"schema_version":1,"zig_version":"0.16.0","paths":["\\ud800"]}\n',
        )
        with tempfile.TemporaryDirectory() as name:
            zon = Path(name) / "build.zig.zon"
            zon.write_text('.{ .paths = .{"payload"} }', encoding="utf-8")
            for output in invalid_outputs:
                result = subprocess.CompletedProcess(
                    args=("zig",), returncode=0, stdout=output, stderr=b""
                )
                with (
                    self.subTest(output=output[:60]),
                    mock.patch.object(checker.subprocess, "run", return_value=result),
                    self.assertRaises((RuntimeError, ValueError)),
                ):
                    checker.package_paths(zon)

            failures = (
                subprocess.CompletedProcess(
                    args=("zig",), returncode=1, stdout=b"", stderr=b""
                ),
                subprocess.CompletedProcess(
                    args=("zig",), returncode=0, stdout=b"{}\n", stderr=b"warning"
                ),
            )
            for result in failures:
                with (
                    self.subTest(returncode=result.returncode, stderr=result.stderr),
                    mock.patch.object(checker.subprocess, "run", return_value=result),
                    self.assertRaises(ValueError),
                ):
                    checker.package_paths(zon)

    def test_archive_mode_cannot_create_a_release_archive(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            (root / "payload").mkdir()
            (root / "payload/data.txt").write_text("data\n", encoding="utf-8")
            (root / "build.zig.zon").write_text(
                '.{\n    .paths = .{\n        "payload",\n    },\n}\n',
                encoding="utf-8",
            )
            destination = root / "source.tar.gz"
            with mock.patch.object(
                sys,
                "argv",
                [
                    "check_package_paths.py",
                    "--root",
                    str(root),
                    "--archive",
                    str(destination),
                ],
            ):
                with self.assertRaisesRegex(ValueError, "exact Git worktree"):
                    checker.main()
            self.assertFalse(destination.exists())

    def test_inspection_snapshots_cannot_cross_publication_boundary(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            payload = root / "payload"
            payload.mkdir()
            (payload / "data.txt").write_text("data\n", encoding="utf-8")
            snapshot = checker.package_files(root, ["payload"], repository=None)
            self.assertEqual(["payload/data.txt"], [member.path for member in snapshot])
            destination = root / "not-created" / "source.tar.gz"

            for candidate in (snapshot, dataclasses.replace(snapshot)):
                with (
                    self.subTest(candidate=type(candidate).__name__),
                    self.assertRaisesRegex(TypeError, "PublishablePackageSnapshot"),
                ):
                    checker.create_archive(root, candidate, destination)
                self.assertFalse(destination.parent.exists())
                self.assertFalse(destination.exists())
                self.assertEqual([], list(root.glob(".source.tar.gz.*")))

    def test_publishable_factory_rejects_plain_rewritten_and_wrong_root_paths(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as name:
            base = Path(name)
            root = base / "repository"
            other = base / "other"
            root.mkdir()
            other.mkdir()
            (root / "payload").mkdir()
            (root / "payload/data.txt").write_text("data\n", encoding="utf-8")
            manifest = root / "build.zig.zon"
            manifest.write_text(
                '.{ .paths = .{"build.zig.zon", "payload"} }\n',
                encoding="utf-8",
            )
            run_git(root, "init", "-q")
            paths = checker.package_paths(manifest, root=root)

            with self.assertRaisesRegex(TypeError, "package_paths"):
                checker.publishable_package_files(root, list(paths))
            with self.assertRaisesRegex(TypeError, "parser-only"):
                checker.PackagePaths(list(paths), paths.manifest)
            rewritten = object.__new__(checker.PackagePaths)
            object.__setattr__(rewritten, "root", paths.root)
            object.__setattr__(rewritten, "paths", ("payload",))
            object.__setattr__(rewritten, "manifest", paths.manifest)
            object.__setattr__(rewritten, "_parse_proof", paths._parse_proof)
            with self.assertRaisesRegex(ValueError, "invalid or rebound"):
                checker.publishable_package_files(root, rewritten)
            with self.assertRaisesRegex(ValueError, "invalid or rebound"):
                checker.publishable_package_files(other, paths)

    def test_publishable_factory_requires_parsed_manifest_in_membership(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            (root / "payload").mkdir()
            (root / "payload/data.txt").write_text("data\n", encoding="utf-8")
            manifest = root / "build.zig.zon"
            manifest.write_text('.{ .paths = .{"payload"} }\n', encoding="utf-8")
            run_git(root, "init", "-q")
            commit_all(root)
            paths = checker.package_paths(manifest, root=root)
            with self.assertRaisesRegex(ValueError, "manifest must include"):
                checker.publishable_package_files(root, paths)

            manifest.write_text('.{ .paths = .{"build.zig.zon"} }\n', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "changed after parsing"):
                checker.publishable_package_files(root, paths)

    def test_publication_proof_rejects_forgery_rebinding_and_transplant(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            base = Path(name)
            first_root = base / "first"
            second_root = base / "second"
            first_root.mkdir()
            second_root.mkdir()
            first = self._publishable_fixture(first_root)
            second = self._publishable_fixture(second_root)

            forged_proof = checker.PublishablePackageSnapshot(
                root=first.root,
                paths=first.paths,
                members=first.members,
                tree=first.tree,
                repository=first.repository,
                git_files=first.git_files,
                manifest=first.manifest,
                _publication_proof=object(),
            )

            class ForgedSubtype(checker.PublishablePackageSnapshot):
                pass

            forged_subtype = ForgedSubtype(
                root=first.root,
                paths=first.paths,
                members=first.members,
                tree=first.tree,
                repository=first.repository,
                git_files=first.git_files,
                manifest=first.manifest,
                _publication_proof=first._publication_proof,
            )
            assert first.repository is not None
            rebound_repository = dataclasses.replace(first.repository)
            candidates = (
                forged_proof,
                forged_subtype,
                dataclasses.replace(first, members=tuple(reversed(first.members))),
                dataclasses.replace(first, repository=rebound_repository),
                dataclasses.replace(
                    first, git_files=dataclasses.replace(first.git_files)
                ),
                dataclasses.replace(
                    second,
                    _publication_proof=first._publication_proof,
                ),
            )
            for index, candidate in enumerate(candidates):
                destination = first_root / f"not-created-{index}" / "source.tar.gz"
                with (
                    self.subTest(index=index),
                    self.assertRaises((TypeError, ValueError)),
                ):
                    checker.create_archive(first_root, candidate, destination)
                self.assertFalse(destination.parent.exists())

    def test_publication_proof_rejects_git_membership_drift_before_side_effects(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            snapshot = self._publishable_fixture(root)
            (root / "payload/late.txt").write_text("late\n", encoding="utf-8")
            destination = root / "not-created" / "source.tar.gz"
            with self.assertRaisesRegex(ValueError, "Git file set changed"):
                checker.create_archive(root, snapshot, destination)
            self.assertFalse(destination.parent.exists())
            self.assertFalse(destination.exists())

    def test_archive_cli_rejects_dirty_publication_subject_before_side_effects(
        self,
    ) -> None:
        cases = (
            "untracked-allowlisted",
            "tracked-modified",
            "staged-new",
            "staged-modified",
            "generator-drift",
        )
        for case in cases:
            with self.subTest(case=case), tempfile.TemporaryDirectory() as name:
                base = Path(name)
                root = base / "repository"
                payload = root / "payload"
                payload.mkdir(parents=True)
                data = payload / "data.txt"
                generated = payload / "generated.zig"
                data.write_text("committed\n", encoding="utf-8")
                generated.write_text("// generated\n", encoding="utf-8")
                (root / "build.zig.zon").write_text(
                    '.{ .paths = .{"build.zig.zon", "payload"} }\n',
                    encoding="utf-8",
                )
                run_git(root, "init", "-q")
                commit_all(root)

                if case == "untracked-allowlisted":
                    (payload / "untracked.txt").write_text(
                        "untracked\n", encoding="utf-8"
                    )
                elif case == "tracked-modified":
                    data.write_text("worktree drift\n", encoding="utf-8")
                elif case == "staged-new":
                    (payload / "staged.txt").write_text("staged\n", encoding="utf-8")
                    run_git(root, "add", "payload/staged.txt")
                elif case == "staged-modified":
                    data.write_text("staged drift\n", encoding="utf-8")
                    run_git(root, "add", "payload/data.txt")
                else:
                    generated.write_text("// generator drift\n", encoding="utf-8")

                destination = base / f"not-created-{case}" / "source.tar.gz"
                with (
                    mock.patch.object(
                        sys,
                        "argv",
                        [
                            "check_package_paths.py",
                            "--root",
                            str(root),
                            "--archive",
                            str(destination),
                        ],
                    ),
                    self.assertRaisesRegex(
                        ValueError, "publication paths differ from HEAD"
                    ),
                ):
                    checker.main()
                self.assertFalse(destination.parent.exists())
                self.assertFalse(destination.exists())

    def test_clean_archive_cli_matches_committed_tree_and_summary(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            base = Path(name)
            root = base / "repository"
            payload = root / "payload"
            payload.mkdir(parents=True)
            (payload / "data.txt").write_text("committed data\n", encoding="utf-8")
            (payload / "generated.zig").write_text(
                "// committed generated output\n", encoding="utf-8"
            )
            (root / ".gitignore").write_text("payload/ignored.txt\n", encoding="utf-8")
            (root / "build.zig.zon").write_text(
                '.{ .paths = .{"build.zig.zon", "payload"} }\n',
                encoding="utf-8",
            )
            run_git(root, "init", "-q")
            commit_all(root)
            (payload / "ignored.txt").write_text("ignored\n", encoding="utf-8")

            revision = subprocess.check_output(
                ("git", "rev-parse", "HEAD"), cwd=root, text=True
            ).strip()
            expected_names = subprocess.check_output(
                (
                    "git",
                    "ls-tree",
                    "-r",
                    "--name-only",
                    revision,
                    "--",
                    "build.zig.zon",
                    "payload",
                ),
                cwd=root,
                text=True,
            ).splitlines()
            destination = base / "release" / "source.tar.gz"
            with (
                mock.patch.object(
                    sys,
                    "argv",
                    [
                        "check_package_paths.py",
                        "--root",
                        str(root),
                        "--archive",
                        str(destination),
                    ],
                ),
                mock.patch("builtins.print") as print_mock,
            ):
                self.assertEqual(0, checker.main())

            print_mock.assert_called_once_with(
                f"checked 2 package paths and {len(expected_names)} files"
            )
            with tarfile.open(destination, "r:gz") as archive:
                self.assertEqual(expected_names, archive.getnames())
                for path in expected_names:
                    committed = subprocess.check_output(
                        ("git", "show", f"{revision}:{path}"), cwd=root
                    )
                    extracted = archive.extractfile(path)
                    self.assertIsNotNone(extracted)
                    assert extracted is not None
                    self.assertEqual(committed, extracted.read())

    def test_archive_cli_rejects_tracked_bytes_hidden_from_status(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            base = Path(name)
            root = base / "repository"
            payload = root / "payload"
            payload.mkdir(parents=True)
            data = payload / "data.txt"
            data.write_text("safe\n", encoding="utf-8")
            (root / "build.zig.zon").write_text(
                '.{ .paths = .{"build.zig.zon", "payload"} }\n',
                encoding="utf-8",
            )
            run_git(root, "init", "-q")
            commit_all(root)
            run_git(root, "update-index", "--assume-unchanged", "payload/data.txt")
            data.write_text("evil\n", encoding="utf-8")
            self.assertEqual(
                b"",
                subprocess.check_output(
                    ("git", "status", "--porcelain=v1", "--", "payload"),
                    cwd=root,
                ),
            )

            destination = base / "not-created-hidden-drift" / "source.tar.gz"
            with (
                mock.patch.object(
                    sys,
                    "argv",
                    [
                        "check_package_paths.py",
                        "--root",
                        str(root),
                        "--archive",
                        str(destination),
                    ],
                ),
                self.assertRaisesRegex(ValueError, "content differs from HEAD"),
            ):
                checker.main()
            self.assertFalse(destination.parent.exists())
            self.assertFalse(destination.exists())

    def test_archive_bytes_and_metadata_are_normalized(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            (root / "payload").mkdir()
            regular = root / "payload/data.txt"
            executable = root / "payload/run.sh"
            regular.write_text("data\n", encoding="utf-8")
            executable.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
            executable.chmod(0o755)
            subprocess.run(("git", "init", "-q"), cwd=root, check=True)
            files = checker.package_files(root, ["payload"])
            expected_names = ["payload/data.txt", "payload/run.sh"]
            first = root / "first.tar.gz"
            second = root / "second.tar.gz"
            checker.create_archive(root, files, first)
            checker.create_archive(root, files, second)
            self.assertEqual(first.read_bytes(), second.read_bytes())
            self.assertEqual(b"\0\0\0\0", first.read_bytes()[4:8])
            with tarfile.open(first, "r:gz") as archive:
                members = archive.getmembers()
            self.assertEqual(expected_names, [member.name for member in members])
            self.assertTrue(all(member.isfile() for member in members))
            self.assertTrue(
                all(member.uid == 0 and member.gid == 0 for member in members)
            )
            self.assertTrue(
                all(member.uname == "" and member.gname == "" for member in members)
            )
            self.assertTrue(all(member.mtime == 0 for member in members))
            self.assertEqual([0o644, 0o755], [member.mode for member in members])
            self.assertTrue(
                all("mtime" not in member.pax_headers for member in members)
            )

    def test_publishable_factory_archive_is_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            files = self._publishable_fixture(root)
            first = root / "strict-first.tar.gz"
            second = root / "strict-second.tar.gz"
            checker.create_archive(root, files, first)
            checker.create_archive(root, files, second)
            self.assertEqual(first.read_bytes(), second.read_bytes())
            with tarfile.open(first, "r:gz") as archive:
                self.assertEqual(
                    ["build.zig.zon", "payload/data.txt"],
                    archive.getnames(),
                )
                self.assertEqual(
                    b'.{ .paths = .{"build.zig.zon", "payload"} }\n',
                    archive.extractfile("build.zig.zon").read(),
                )
                self.assertEqual(
                    b"data\n",
                    archive.extractfile("payload/data.txt").read(),
                )

    def test_archive_publication_rolls_back_source_mutation_at_replace(self) -> None:
        for existing in (False, True):
            with self.subTest(existing=existing), tempfile.TemporaryDirectory() as name:
                root = Path(name)
                payload = root / "payload"
                payload.mkdir()
                source = payload / "data.txt"
                source.write_text("safe\n", encoding="utf-8")
                destination = root / "source.tar.gz"
                if existing:
                    destination.write_bytes(b"previous archive\n")
                run_git(root, "init", "-q")
                snapshot = checker.package_files(root, ["payload"])
                previous = destination.read_bytes() if existing else None
                real_replace = os.replace
                replacements = 0

                def replace_then_mutate(
                    src: object, dst: object, **kwargs: object
                ) -> None:
                    nonlocal replacements
                    replacements += 1
                    real_replace(src, dst, **kwargs)  # type: ignore[arg-type]
                    if replacements == 1:
                        source.write_text("evil\n", encoding="utf-8")

                with (
                    mock.patch.object(
                        checker.os, "replace", side_effect=replace_then_mutate
                    ),
                    self.assertRaisesRegex(RuntimeError, "rollback_indeterminate"),
                ):
                    checker._test_only_materialize_archive_transaction(
                        root, snapshot, destination
                    )

                if previous is None:
                    self.assertFalse(destination.exists())
                else:
                    self.assertEqual(previous, destination.read_bytes())
                sidecars = [
                    path
                    for path in root.iterdir()
                    if path.name.startswith(".source.tar.gz.")
                ]
                if previous is None:
                    self.assertEqual([], sidecars)
                else:
                    self.assertEqual(1, len(sidecars))
                    self.assertTrue(sidecars[0].name.endswith(".backup"))
                    self.assertEqual(previous, sidecars[0].read_bytes())

    def test_archive_rejects_prepared_artifact_content_mutation(self) -> None:
        cases = {
            "content-after-record": "digest changed",
            "append-during-pread": "metadata changed while hashing",
            "truncate-during-pread": "early EOF",
            "oversized-pread": "oversized positional read",
        }
        for mutation, message in cases.items():
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as name:
                root = Path(name)
                payload = root / "payload"
                payload.mkdir()
                (payload / "data.txt").write_text("safe\n", encoding="utf-8")
                destination = root / "source.tar.gz"
                destination.write_bytes(b"previous archive\n")
                run_git(root, "init", "-q")
                snapshot = checker.package_files(root, ["payload"])
                real_verify_tree = checker._verify_tree
                real_pread = os.pread
                mutated_reads = 0

                def mutate_after_record(
                    root_path: Path, package: object, stage: str
                ) -> None:
                    real_verify_tree(root_path, package, stage)
                    if (
                        mutation == "content-after-record"
                        and stage == "during archive materialization"
                    ):
                        temporary = next(root.glob(".source.tar.gz.*.tmp"))
                        metadata = temporary.stat()
                        contents = bytearray(temporary.read_bytes())
                        contents[-1] ^= 1
                        temporary.write_bytes(contents)
                        os.chmod(temporary, metadata.st_mode)

                def adversarial_pread(
                    descriptor: int, requested: int, offset: int
                ) -> bytes:
                    nonlocal mutated_reads
                    if mutation == "truncate-during-pread":
                        try:
                            os.ftruncate(descriptor, os.fstat(descriptor).st_size // 2)
                        except OSError:
                            return real_pread(descriptor, requested, offset)
                        mutated_reads += 1
                        return real_pread(descriptor, requested, offset)

                    chunk = real_pread(descriptor, requested, offset)
                    if mutation == "append-during-pread":
                        try:
                            os.pwrite(
                                descriptor,
                                b"x",
                                os.fstat(descriptor).st_size,
                            )
                        except OSError:
                            return chunk
                        mutated_reads += 1
                    elif mutation == "oversized-pread":
                        try:
                            os.pwrite(descriptor, b"", 0)
                        except OSError:
                            return chunk
                        mutated_reads += 1
                        return chunk + b"x"
                    return chunk

                with (
                    mock.patch.object(
                        checker, "_verify_tree", side_effect=mutate_after_record
                    ),
                    mock.patch.object(
                        checker.os, "pread", side_effect=adversarial_pread
                    ),
                    self.assertRaisesRegex(ValueError, message),
                ):
                    checker._test_only_materialize_archive_transaction(
                        root, snapshot, destination
                    )
                self.assertEqual(b"previous archive\n", destination.read_bytes())
                self.assertEqual([], list(root.glob(".source.tar.gz.*")))
                if mutation != "content-after-record":
                    self.assertGreater(mutated_reads, 0)
                    self.assertLessEqual(mutated_reads, 2)

        self._assert_archive_member_reads_are_bounded()

    def _assert_archive_member_reads_are_bounded(self) -> None:
        for growth in ("finite", "endlessly-extending"):
            with self.subTest(growth=growth), tempfile.TemporaryDirectory() as name:
                root = Path(name)
                (root / "payload").mkdir()
                target = root / "payload/data.bin"
                target.write_bytes(b"abcde")
                run_git(root, "init", "-q")
                files = checker.package_files(root, ["payload"])
                destination = root / "source.tar.gz"
                previous = b"previous archive\n"
                destination.write_bytes(previous)
                real_open = checker._open_package_member
                requested_sizes: list[int] = []
                returned_bytes = 0

                class GrowingStream:
                    def __init__(self, stream: object) -> None:
                        self.stream = stream

                    def __enter__(self) -> "GrowingStream":
                        return self

                    def __exit__(self, *_: object) -> None:
                        self.stream.close()  # type: ignore[attr-defined]

                    def fileno(self) -> int:
                        return self.stream.fileno()  # type: ignore[attr-defined,no-any-return]

                    def read(self, size: int = -1) -> bytes:
                        nonlocal returned_bytes
                        requested_sizes.append(size)
                        if growth == "endlessly-extending" and len(requested_sizes) > 1:
                            raise AssertionError(
                                "archive read beyond the frozen package member size"
                            )
                        data = self.stream.read(size)  # type: ignore[attr-defined]
                        returned_bytes += len(data)
                        with target.open("ab") as writer:
                            writer.write(b"growth")
                        return data

                def growing_open(
                    root_path: Path, member: object
                ) -> tuple[object, os.stat_result]:
                    stream, metadata = real_open(root_path, member)
                    return GrowingStream(stream), metadata

                with (
                    mock.patch.object(
                        checker, "_open_package_member", side_effect=growing_open
                    ),
                    self.assertRaisesRegex(
                        ValueError, "content changed after enumeration"
                    ),
                ):
                    checker._test_only_materialize_archive_transaction(
                        root, files, destination
                    )

                self.assertEqual([5], requested_sizes)
                self.assertEqual(5, returned_bytes)
                self.assertEqual(previous, destination.read_bytes())
                self.assertEqual([], list(root.glob(".source.tar.gz.*")))

        invalid_reads = {
            "early-eof": "early EOF",
            "non-byte": "unexpected read result",
            "oversized": "oversized read",
        }
        for invalid_read, message in invalid_reads.items():
            with (
                self.subTest(invalid_read=invalid_read),
                tempfile.TemporaryDirectory() as name,
            ):
                root = Path(name)
                (root / "payload").mkdir()
                (root / "payload/data.bin").write_bytes(b"abcde")
                run_git(root, "init", "-q")
                files = checker.package_files(root, ["payload"])
                destination = root / "source.tar.gz"
                previous = b"previous archive\n"
                destination.write_bytes(previous)
                real_open = checker._open_package_member
                read_calls = 0

                class InvalidStream:
                    def __init__(self, stream: object) -> None:
                        self.stream = stream

                    def __enter__(self) -> "InvalidStream":
                        return self

                    def __exit__(self, *_: object) -> None:
                        self.stream.close()  # type: ignore[attr-defined]

                    def fileno(self) -> int:
                        return self.stream.fileno()  # type: ignore[attr-defined,no-any-return]

                    def read(self, size: int = -1) -> object:
                        nonlocal read_calls
                        read_calls += 1
                        if invalid_read == "early-eof":
                            return b"ab" if read_calls == 1 else b""
                        if invalid_read == "non-byte":
                            return bytearray(b"abcde")
                        return b"abcdef"

                def invalid_open(
                    root_path: Path, member: object
                ) -> tuple[object, os.stat_result]:
                    stream, metadata = real_open(root_path, member)
                    return InvalidStream(stream), metadata

                with (
                    mock.patch.object(
                        checker, "_open_package_member", side_effect=invalid_open
                    ),
                    self.assertRaisesRegex(ValueError, message),
                ):
                    checker._test_only_materialize_archive_transaction(
                        root, files, destination
                    )

                self.assertEqual(2 if invalid_read == "early-eof" else 1, read_calls)
                self.assertEqual(previous, destination.read_bytes())
                self.assertEqual([], list(root.glob(".source.tar.gz.*")))

    def test_archive_rejects_temporary_path_swap_without_deleting_foreign_path(
        self,
    ) -> None:
        cases = {
            "late-path-swap": "unexpected artifact.*recovery material retained",
            "record-failure-swap": "identity is unknown.*unclaimed candidate",
            "fdopen-failure-swap": "stream creation failed after descriptor ownership transfer.*unclaimed candidate",
            "fdopen-claim-race": "stream creation failed after descriptor ownership transfer.*unclaimed candidate",
            "precommit-claim-race": "unexpected artifact.*recovery material retained",
        }
        for failure, message in cases.items():
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as name:
                root = Path(name)
                payload = root / "payload"
                payload.mkdir()
                (payload / "data.txt").write_text("safe\n", encoding="utf-8")
                destination = root / "source.tar.gz"
                destination.write_bytes(b"previous archive\n")
                run_git(root, "init", "-q")
                snapshot = checker.package_files(root, ["payload"])
                real_verify_tree = checker._verify_tree
                real_record = checker._record_artifact
                real_fdopen = os.fdopen
                real_rename = os.rename
                swapped_path: Path | None = None

                def install_foreign_temporary() -> Path:
                    temporary = next(root.glob(".source.tar.gz.*.tmp"))
                    temporary.unlink()
                    temporary.write_bytes(b"foreign path\n")
                    return temporary

                def swap_temporary(
                    root_path: Path, package: object, stage: str
                ) -> None:
                    nonlocal swapped_path
                    real_verify_tree(root_path, package, stage)
                    if failure == "late-path-swap" and stage == (
                        "during archive materialization"
                    ):
                        temporary = next(root.glob(".source.tar.gz.*.tmp"))
                        temporary.rename(temporary.with_suffix(".prepared"))
                        temporary.write_bytes(b"foreign path\n")
                        swapped_path = temporary
                    elif failure == "precommit-claim-race" and stage == (
                        "during archive materialization"
                    ):
                        raise ValueError("precommit validation failed")

                def fail_record(descriptor: int, label: str) -> object:
                    nonlocal swapped_path
                    if failure == "record-failure-swap" and label == (
                        "archive temporary"
                    ):
                        swapped_path = install_foreign_temporary()
                        raise ValueError("record failed")
                    return real_record(descriptor, label)

                def fail_fdopen(
                    descriptor: int, mode: str, *args: object, **kwargs: object
                ) -> object:
                    nonlocal swapped_path
                    if (
                        failure
                        in {
                            "fdopen-failure-swap",
                            "fdopen-claim-race",
                        }
                        and mode == "w+b"
                    ):
                        swapped_path = install_foreign_temporary()
                        raise OSError("fdopen failed")
                    return real_fdopen(descriptor, mode, *args, **kwargs)  # type: ignore[call-overload]

                def replace_during_cleanup_claim(
                    source: object, target: object, **kwargs: object
                ) -> None:
                    nonlocal swapped_path
                    if (
                        failure == "precommit-claim-race"
                        and os.fspath(target) == "claimed"
                        and os.fspath(source).endswith(".tmp")
                    ):
                        public_temporary = root / os.fspath(source)
                        public_temporary.unlink()
                        public_temporary.write_bytes(b"foreign path\n")
                        swapped_path = public_temporary
                    real_rename(source, target, **kwargs)  # type: ignore[arg-type]

                with (
                    mock.patch.object(
                        checker, "_verify_tree", side_effect=swap_temporary
                    ),
                    mock.patch.object(
                        checker, "_record_artifact", side_effect=fail_record
                    ),
                    mock.patch.object(checker.os, "fdopen", side_effect=fail_fdopen),
                    mock.patch.object(
                        checker.os,
                        "rename",
                        side_effect=replace_during_cleanup_claim,
                    ),
                    self.assertRaisesRegex(RuntimeError, message) as raised,
                ):
                    checker._test_only_materialize_archive_transaction(
                        root, snapshot, destination
                    )
                self.assertIsNotNone(swapped_path)
                assert swapped_path is not None
                self.assertEqual(b"previous archive\n", destination.read_bytes())
                if failure in {
                    "record-failure-swap",
                    "fdopen-failure-swap",
                    "fdopen-claim-race",
                }:
                    self.assertEqual(b"foreign path\n", swapped_path.read_bytes())
                    self.assertIn(
                        os.fspath(root.resolve() / swapped_path.name),
                        str(raised.exception),
                    )
                    self.assertIn("unclaimed candidate", str(raised.exception))
                    self.assertNotIn("recovery material", str(raised.exception).lower())
                    self.assertNotIn("recovery path", str(raised.exception).lower())
                    self.assertEqual([], list(cleanup_arena(root).iterdir()))
                else:
                    cleanups = list((cleanup_arena(root)).glob(".*.cleanup"))
                    self.assertEqual(1, len(cleanups))
                    recovery = cleanups[0] / "claimed"
                    self.assertEqual(b"foreign path\n", recovery.read_bytes())
                    self.assertIn(f"{cleanups[0].name}/claimed", str(raised.exception))

        for close_failure in ("fdopen", "stream"):
            with (
                self.subTest(close_failure=close_failure),
                tempfile.TemporaryDirectory() as name,
            ):
                root = Path(name)
                payload = root / "payload"
                payload.mkdir()
                (payload / "data.txt").write_text("safe\n", encoding="utf-8")
                destination = root / "source.tar.gz"
                destination.write_bytes(b"previous archive\n")
                run_git(root, "init", "-q")
                snapshot = checker.package_files(root, ["payload"])
                real_close = os.close
                real_fdopen = os.fdopen
                stream_close_calls = 0
                reused_descriptor: int | None = None

                class CloseFailingStream:
                    def __init__(self, stream: typing.BinaryIO) -> None:
                        self.stream = stream

                    @property
                    def closed(self) -> bool:
                        return self.stream.closed

                    def close(self) -> None:
                        nonlocal reused_descriptor, stream_close_calls
                        stream_close_calls += 1
                        transferred_descriptor = self.stream.fileno()
                        self.stream.close()
                        reused_descriptor = os.open(
                            root / "stream-close-reuse",
                            os.O_CREAT | os.O_EXCL | os.O_WRONLY,
                            0o600,
                        )
                        self_test.assertEqual(transferred_descriptor, reused_descriptor)
                        raise OSError("stream close failed after descriptor close")

                    def __getattr__(self, attribute: str) -> object:
                        return getattr(self.stream, attribute)

                def fail_or_wrap_fdopen(
                    descriptor: int,
                    mode: str,
                    *args: object,
                    **kwargs: object,
                ) -> object:
                    nonlocal reused_descriptor
                    if mode != "w+b":
                        return real_fdopen(descriptor, mode, *args, **kwargs)  # type: ignore[call-overload]
                    if close_failure == "fdopen":
                        real_close(descriptor)
                        reused_descriptor = os.open(
                            root / "fdopen-reuse",
                            os.O_CREAT | os.O_EXCL | os.O_WRONLY,
                            0o600,
                        )
                        self.assertEqual(descriptor, reused_descriptor)
                        raise OSError("fdopen failed")
                    return CloseFailingStream(
                        real_fdopen(descriptor, mode, *args, **kwargs)  # type: ignore[call-overload]
                    )

                self_test = self
                with (
                    mock.patch.object(
                        checker.os, "fdopen", side_effect=fail_or_wrap_fdopen
                    ),
                    mock.patch.object(
                        checker, "_claim_and_remove_public_artifact"
                    ) as cleanup_artifact,
                    self.assertRaisesRegex(
                        RuntimeError,
                        "unclaimed candidate",
                    ) as raised,
                ):
                    checker._test_only_materialize_archive_transaction(
                        root, snapshot, destination
                    )

                temporaries = list(root.glob(".source.tar.gz.*.tmp"))
                self.assertEqual(1, len(temporaries))
                self.assertIn(
                    os.fspath(root.resolve() / temporaries[0].name),
                    str(raised.exception),
                )
                self.assertNotIn("recovery material", str(raised.exception).lower())
                self.assertNotIn("recovery path", str(raised.exception).lower())
                cleanup_artifact.assert_not_called()
                self.assertIsNotNone(reused_descriptor)
                os.fstat(typing.cast(int, reused_descriptor))
                self.assertEqual(
                    0 if close_failure == "fdopen" else 1, stream_close_calls
                )
                real_close(typing.cast(int, reused_descriptor))
                self.assertEqual(b"previous archive\n", destination.read_bytes())

        with (
            self.subTest(close_failure="refresh-fd-reuse"),
            tempfile.TemporaryDirectory() as name,
        ):
            root = Path(name)
            payload = root / "payload"
            payload.mkdir()
            (payload / "data.txt").write_text("safe\n", encoding="utf-8")
            destination = root / "source.tar.gz"
            destination.write_bytes(b"previous archive\n")
            run_git(root, "init", "-q")
            snapshot = checker.package_files(root, ["payload"])
            real_verify_tree = checker._verify_tree
            real_open_name = checker._open_name_no_follow
            real_close = os.close
            refresh_descriptor: int | None = None
            reused_descriptor: int | None = None

            def fail_before_publication(
                root_path: Path, package: object, stage: str
            ) -> None:
                real_verify_tree(root_path, package, stage)
                if stage == "during archive materialization":
                    raise ValueError("prepublication verification failed")

            def capture_refresh_descriptor(
                directory_descriptor: int,
                directory_path: Path,
                public_name: str,
                label: str,
            ) -> object:
                nonlocal refresh_descriptor
                owner = real_open_name(
                    directory_descriptor, directory_path, public_name, label
                )
                if label == "archive temporary":
                    refresh_descriptor = owner.fileno()
                return owner

            def fail_refresh_close_after_reuse(descriptor: int) -> None:
                nonlocal reused_descriptor
                real_close(descriptor)
                if descriptor == refresh_descriptor:
                    reused_descriptor = os.open(
                        root / "refresh-close-reuse",
                        os.O_CREAT | os.O_EXCL | os.O_WRONLY,
                        0o600,
                    )
                    self.assertEqual(descriptor, reused_descriptor)
                    raise OSError("refresh close failed after descriptor reuse")

            with (
                mock.patch.object(
                    checker, "_verify_tree", side_effect=fail_before_publication
                ),
                mock.patch.object(
                    checker,
                    "_open_name_no_follow",
                    side_effect=capture_refresh_descriptor,
                ),
                mock.patch.object(
                    checker.os, "close", side_effect=fail_refresh_close_after_reuse
                ),
                mock.patch.object(
                    checker, "_claim_and_remove_public_artifact"
                ) as cleanup_artifact,
                self.assertRaisesRegex(
                    RuntimeError,
                    "archive temporary refresh descriptor close failed.*unclaimed candidate",
                ) as raised,
            ):
                checker._test_only_materialize_archive_transaction(
                    root, snapshot, destination
                )

            temporaries = list(root.glob(".source.tar.gz.*.tmp"))
            self.assertEqual(1, len(temporaries))
            self.assertIn(
                os.fspath(root.resolve() / temporaries[0].name),
                str(raised.exception),
            )
            self.assertNotIn("recovery material", str(raised.exception).lower())
            self.assertNotIn("recovery path", str(raised.exception).lower())
            cleanup_artifact.assert_not_called()
            self.assertIsNotNone(reused_descriptor)
            os.fstat(typing.cast(int, reused_descriptor))
            real_close(typing.cast(int, reused_descriptor))
            self.assertEqual(b"previous archive\n", destination.read_bytes())

    def test_archive_rejects_symlink_and_special_destination(self) -> None:
        for kind in ("symlink", "directory"):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as name:
                root = Path(name)
                payload = root / "payload"
                payload.mkdir()
                (payload / "data.txt").write_text("safe\n", encoding="utf-8")
                destination = root / "source.tar.gz"
                if kind == "symlink":
                    target = root / "external.tar.gz"
                    target.write_bytes(b"external\n")
                    try:
                        destination.symlink_to(target)
                    except OSError as exc:
                        self.skipTest(f"symlinks are unavailable: {exc}")
                else:
                    destination.mkdir()
                run_git(root, "init", "-q")
                snapshot = checker.package_files(root, ["payload"])
                with self.assertRaisesRegex(ValueError, "safe regular|regular file"):
                    checker._test_only_materialize_archive_transaction(
                        root, snapshot, destination
                    )
                if kind == "symlink":
                    self.assertTrue(destination.is_symlink())
                    self.assertEqual(b"external\n", target.read_bytes())
                else:
                    self.assertTrue(destination.is_dir())

    def test_archive_replace_failure_preserves_existing_destination(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            payload = root / "payload"
            payload.mkdir()
            (payload / "data.txt").write_text("safe\n", encoding="utf-8")
            destination = root / "source.tar.gz"
            destination.write_bytes(b"previous archive\n")
            run_git(root, "init", "-q")
            snapshot = checker.package_files(root, ["payload"])
            real_replace = os.replace

            def fail_publication_replace(
                source: object, target: object, **kwargs: object
            ) -> None:
                if (
                    os.fspath(source).endswith(".tmp")
                    and os.fspath(target) == destination.name
                ):
                    raise OSError("replace failed")
                real_replace(source, target, **kwargs)  # type: ignore[arg-type]

            with (
                mock.patch.object(
                    checker.os, "replace", side_effect=fail_publication_replace
                ),
                self.assertRaisesRegex(RuntimeError, "rollback_indeterminate"),
            ):
                checker._test_only_materialize_archive_transaction(
                    root, snapshot, destination
                )
            self.assertEqual(b"previous archive\n", destination.read_bytes())
            backups = list(root.glob(".source.tar.gz.*.backup"))
            self.assertEqual(1, len(backups))
            self.assertEqual(b"previous archive\n", backups[0].read_bytes())

    def test_archive_rollback_races_and_quarantine_cleanup_fail_closed(self) -> None:
        cleanup_module = checker.repository_snapshot

        with (
            self.subTest(failure="retained-rebind-unaddressable"),
            tempfile.TemporaryDirectory() as name,
        ):
            root = Path(name).resolve()
            recovery = root / ".cleanup" / "claimed"
            stale_final_recovery = root / ".cleanup" / "stale-final"
            first_candidate = root / "first-candidate"
            final_candidate = root / "final-candidate"
            anchor_identity = (41, 42)
            retained = cleanup_module.CleanupOutcome(
                cleanup_module.CleanupDisposition.RETAINED,
                (recovery,),
                (
                    cleanup_module.CleanupIssue(
                        "cleanup_provisional_retained", root / ".cleanup"
                    ),
                ),
                candidate_paths=(first_candidate,),
                public_candidate=cleanup_module.PublicCandidate.PRESENT,
                arena_identity=(51, 52),
                recovery_anchor_identity=anchor_identity,
            )
            rebound = cleanup_module.CleanupOutcome(
                cleanup_module.CleanupDisposition.UNADDRESSABLE,
                (stale_final_recovery,),
                (
                    cleanup_module.CleanupIssue(
                        "cleanup_arena_binding_rebound", root / ".cleanup"
                    ),
                ),
                candidate_paths=(final_candidate,),
                arena_binding=cleanup_module.ArenaBinding.REBOUND,
                public_candidate=cleanup_module.PublicCandidate.UNKNOWN,
                arena_identity=(61, 62),
                recovery_anchor_identity=anchor_identity,
            )
            description = checker._rollback_recovery_description(
                root,
                ".source.tar.gz.backup",
                rebound,
                retained,
            )
            self.assertIn("cleanup namespace is unaddressable", description)
            self.assertIn("cleanup_provisional_retained", description)
            self.assertIn("cleanup_arena_binding_rebound", description)
            self.assertIn(os.fspath(first_candidate), description)
            self.assertIn(os.fspath(final_candidate), description)
            self.assertNotIn(os.fspath(recovery), description)
            self.assertNotIn(os.fspath(stale_final_recovery), description)
            self.assertNotIn("recovery material", description)
            self.assertNotIn(".source.tar.gz.backup", description)

            unknown_identity = cleanup_module.CleanupOutcome(
                cleanup_module.CleanupDisposition.UNADDRESSABLE,
                (),
                (
                    cleanup_module.CleanupIssue(
                        "cleanup_identity_unknown", root / ".cleanup"
                    ),
                ),
                arena_binding=cleanup_module.ArenaBinding.UNKNOWN,
            )
            unknown_description = checker._rollback_recovery_description(
                root,
                ".source.tar.gz.backup",
                unknown_identity,
                retained,
            )
            self.assertIn("cleanup_identity_unknown", unknown_description)
            self.assertNotIn(os.fspath(recovery), unknown_description)
            self.assertNotIn("recovery material", unknown_description)
            self.assertNotIn(".source.tar.gz.backup", unknown_description)

        def create_test_quarantine(
            parent_descriptor: int, parent_path: Path
        ) -> typing.Any:
            return cleanup_module.CleanupQuarantine.create(
                cleanup_module.DirectoryAnchor(parent_descriptor, parent_path),
                "artifact",
                quarantine_prefix=".artifact.",
                quarantine_suffix=".cleanup",
            )

        for failure in (
            "old-restoration-race",
            "quarantine-cleanup",
            "quarantine-identity-replacement",
            "post-restore-replacement",
        ):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as name:
                root = Path(name)
                payload = root / "payload"
                payload.mkdir()
                source = payload / "data.txt"
                source.write_text("safe\n", encoding="utf-8")
                destination = root / "source.tar.gz"
                destination.write_bytes(b"previous archive\n")
                run_git(root, "init", "-q")
                snapshot = checker.package_files(root, ["payload"])
                real_replace = os.replace
                real_link = os.link
                real_rmdir = os.rmdir
                real_verify = checker._verify_artifact_name
                real_close_quarantine = cleanup_module.OwnedDescriptor.close_once
                published = False
                raced = False
                restored_checks = 0

                def mutate_source_after_publish(
                    src: object, dst: object, **kwargs: object
                ) -> None:
                    nonlocal published
                    real_replace(src, dst, **kwargs)  # type: ignore[arg-type]
                    if not published:
                        published = True
                        source.write_text("evil\n", encoding="utf-8")

                def race_old_restoration(
                    src: object,
                    dst: object,
                    *args: object,
                    **kwargs: object,
                ) -> None:
                    nonlocal raced
                    if (
                        failure == "old-restoration-race"
                        and os.fspath(src).endswith(".backup")
                        and os.fspath(dst) == destination.name
                    ):
                        raced = True
                        destination.write_bytes(b"external writer\n")
                    real_link(src, dst, *args, **kwargs)  # type: ignore[arg-type]

                def fail_quarantine_cleanup(
                    path: object, *args: object, **kwargs: object
                ) -> None:
                    if failure == "quarantine-cleanup" and os.fspath(path).endswith(
                        ".quarantine"
                    ):
                        raise PermissionError("quarantine rmdir denied")
                    real_rmdir(path, *args, **kwargs)  # type: ignore[arg-type]

                def replace_quarantine_after_close(owner: typing.Any) -> None:
                    nonlocal raced
                    real_close_quarantine(owner)
                    if (
                        failure == "quarantine-identity-replacement"
                        and owner.recovery_path.name.endswith(".quarantine")
                    ):
                        raced = True
                        quarantine_path = root / owner.recovery_path
                        quarantine_path.rmdir()
                        quarantine_path.mkdir(mode=0o700)

                def replace_after_final_restore_check(
                    directory_descriptor: int,
                    directory_path: Path,
                    entry: str,
                    expected: object,
                    label: str,
                ) -> None:
                    nonlocal raced, restored_checks
                    real_verify(
                        directory_descriptor,
                        directory_path,
                        entry,
                        expected,
                        label,
                    )
                    if label == "restored archive destination":
                        restored_checks += 1
                        if (
                            failure == "post-restore-replacement"
                            and restored_checks == 2
                        ):
                            raced = True
                            destination.unlink()
                            destination.write_bytes(b"external writer\n")

                with (
                    mock.patch.object(
                        checker.os, "replace", side_effect=mutate_source_after_publish
                    ),
                    mock.patch.object(
                        checker.os, "link", side_effect=race_old_restoration
                    ),
                    mock.patch.object(
                        checker.os, "rmdir", side_effect=fail_quarantine_cleanup
                    ),
                    mock.patch.object(
                        checker,
                        "_verify_artifact_name",
                        side_effect=replace_after_final_restore_check,
                    ),
                    mock.patch.object(
                        cleanup_module.OwnedDescriptor,
                        "close_once",
                        replace_quarantine_after_close,
                    ),
                    mock.patch.object(checker, "_require_publication_support"),
                    self.assertRaisesRegex(RuntimeError, "rollback_indeterminate"),
                ):
                    checker._test_only_materialize_archive_transaction(
                        root, snapshot, destination
                    )

                backups = list(root.glob(".source.tar.gz.*.backup"))
                quarantines = list(
                    (cleanup_arena(root)).glob(".source.tar.gz.*.quarantine")
                )
                self.assertEqual(1, len(backups))
                self.assertEqual(b"previous archive\n", backups[0].read_bytes())
                if failure == "old-restoration-race":
                    self.assertEqual(1, len(quarantines))
                    self.assertTrue(raced)
                    self.assertEqual(b"external writer\n", destination.read_bytes())
                    self.assertTrue((quarantines[0] / "published-artifact").is_file())
                elif failure == "quarantine-cleanup":
                    self.assertEqual(1, len(quarantines))
                    self.assertFalse(raced)
                    self.assertEqual(b"previous archive\n", destination.read_bytes())
                    self.assertEqual([], list(quarantines[0].iterdir()))
                elif failure == "quarantine-identity-replacement":
                    self.assertEqual(1, len(quarantines))
                    self.assertTrue(raced)
                    self.assertEqual(b"previous archive\n", destination.read_bytes())
                    self.assertEqual([], list(quarantines[0].iterdir()))
                else:
                    self.assertEqual([], quarantines)
                    self.assertTrue(raced)
                    self.assertEqual(b"external writer\n", destination.read_bytes())

        for setup_failure in ("mode-setting", "fstat", "close"):
            with (
                self.subTest(setup_failure=setup_failure),
                tempfile.TemporaryDirectory() as name,
            ):
                root = Path(name)
                parent_descriptor = os.open(
                    root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
                )
                real_close = os.close
                real_fstat = os.fstat
                real_fchmod = os.fchmod
                close_calls: list[int] = []
                fstat_calls = 0
                fchmod_calls = 0

                def record_close(descriptor: int) -> None:
                    close_calls.append(descriptor)
                    real_close(descriptor)
                    if setup_failure == "close":
                        raise OSError("close failed after ownership transfer")

                def fail_opened_fstat(descriptor: int) -> object:
                    nonlocal fstat_calls
                    fstat_calls += 1
                    if fstat_calls == 7:
                        raise OSError("fstat failed")
                    return real_fstat(descriptor)

                def fail_quarantine_fchmod(descriptor: int, mode: int) -> None:
                    nonlocal fchmod_calls
                    fchmod_calls += 1
                    if fchmod_calls == 2:
                        raise PermissionError("mode setting denied")
                    real_fchmod(descriptor, mode)

                patches: list[typing.Any] = [
                    mock.patch.object(checker.os, "close", side_effect=record_close)
                ]
                if setup_failure in {"mode-setting", "close"}:
                    patches.append(
                        mock.patch.object(
                            checker.os,
                            "fchmod",
                            side_effect=fail_quarantine_fchmod,
                        )
                    )
                else:
                    patches.append(
                        mock.patch.object(
                            checker.os,
                            "fstat",
                            side_effect=fail_opened_fstat,
                        )
                    )
                try:
                    with (
                        patches[0],
                        patches[1],
                        self.assertRaises(cleanup_module.CleanupFailure) as raised,
                    ):
                        create_test_quarantine(parent_descriptor, root.resolve())
                    quarantines = list(
                        (cleanup_arena(root)).glob(".artifact.*.cleanup")
                    )
                    self.assertEqual(2, len(close_calls))
                    self.assertEqual(1, len(quarantines))
                    self.assertIn(
                        quarantines[0].name,
                        " ".join(
                            map(os.fspath, raised.exception.outcome.recovery_paths)
                        ),
                    )
                    quarantines[0].rmdir()
                finally:
                    real_close(parent_descriptor)

        for post_drift in ("owner", "mode"):
            with (
                self.subTest(setup_failure=f"post-{post_drift}-drift"),
                tempfile.TemporaryDirectory() as name,
            ):
                root = Path(name)
                parent_descriptor = os.open(
                    root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
                )
                real_fstat = os.fstat
                real_close = os.close
                fstat_calls = 0
                close_calls: list[int] = []

                def drift_after_mode_setting(descriptor: int) -> object:
                    nonlocal fstat_calls
                    metadata = real_fstat(descriptor)
                    fstat_calls += 1
                    if fstat_calls != 8:
                        return metadata
                    configured_mode = (
                        (metadata.st_mode & ~0o777) | 0o755
                        if post_drift == "mode"
                        else metadata.st_mode
                    )
                    configured_owner = (
                        metadata.st_uid + 1
                        if post_drift == "owner"
                        else metadata.st_uid
                    )
                    return mock.Mock(
                        st_mode=configured_mode,
                        st_dev=metadata.st_dev,
                        st_ino=metadata.st_ino,
                        st_uid=configured_owner,
                    )

                def record_post_drift_close(descriptor: int) -> None:
                    close_calls.append(descriptor)
                    real_close(descriptor)

                try:
                    with (
                        mock.patch.object(
                            checker.os,
                            "fstat",
                            side_effect=drift_after_mode_setting,
                        ),
                        mock.patch.object(
                            checker.os,
                            "close",
                            side_effect=record_post_drift_close,
                        ),
                        mock.patch.object(checker.os, "rmdir") as remove_directory,
                        self.assertRaises(cleanup_module.CleanupFailure) as raised,
                    ):
                        create_test_quarantine(parent_descriptor, root.resolve())
                    quarantines = list(
                        (cleanup_arena(root)).glob(".artifact.*.cleanup")
                    )
                    self.assertEqual(8, fstat_calls)
                    self.assertEqual(2, len(close_calls))
                    remove_directory.assert_not_called()
                    self.assertEqual(1, len(quarantines))
                    self.assertEqual([], list(quarantines[0].iterdir()))
                    issue = raised.exception.outcome.issues[0]
                    self.assertIn(
                        quarantines[0].name,
                        os.fspath(issue.path),
                    )
                    self.assertIsNotNone(issue.error)
                    self.assertIn(
                        "credential is unsafe",
                        str(issue.error),
                    )
                    quarantines[0].rmdir()
                finally:
                    real_close(parent_descriptor)

        with self.subTest(setup_failure="empty-directory-replacement"):
            with tempfile.TemporaryDirectory() as name:
                root = Path(name)
                parent_descriptor = os.open(
                    root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
                )
                real_open = os.open
                real_close = os.close
                real_rmdir = os.rmdir
                real_mkdir = os.mkdir
                replaced = False
                held_quarantine_descriptor: int | None = None
                identities: list[tuple[int, int]] = []

                def replace_before_open(
                    path: object,
                    flags: int,
                    *args: object,
                    **kwargs: object,
                ) -> int:
                    nonlocal held_quarantine_descriptor, replaced
                    if not replaced and os.fspath(path).endswith(".cleanup"):
                        replaced = True
                        quarantine_parent = typing.cast(int, kwargs.get("dir_fd"))
                        held_quarantine_descriptor = real_open(
                            path,
                            os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                            dir_fd=quarantine_parent,
                        )
                        before = os.fstat(held_quarantine_descriptor)
                        identities.append((before.st_dev, before.st_ino))
                        real_rmdir(path, dir_fd=quarantine_parent)  # type: ignore[arg-type]
                        real_mkdir(
                            path,
                            0o700,
                            dir_fd=quarantine_parent,  # type: ignore[arg-type]
                        )
                        after = os.stat(
                            path,
                            dir_fd=quarantine_parent,  # type: ignore[arg-type]
                            follow_symlinks=False,
                        )
                        identities.append((after.st_dev, after.st_ino))
                    return real_open(path, flags, *args, **kwargs)  # type: ignore[arg-type]

                try:
                    with (
                        mock.patch.object(
                            checker.os, "open", side_effect=replace_before_open
                        ),
                        self.assertRaises(cleanup_module.CleanupFailure),
                    ):
                        create_test_quarantine(parent_descriptor, root.resolve())
                    quarantines = list(
                        (cleanup_arena(root)).glob(".artifact.*.cleanup")
                    )
                    self.assertTrue(replaced)
                    self.assertEqual(2, len(identities))
                    self.assertNotEqual(identities[0], identities[1])
                    self.assertEqual(1, len(quarantines))
                    self.assertEqual([], list(quarantines[0].iterdir()))
                    quarantines[0].rmdir()
                finally:
                    try:
                        if held_quarantine_descriptor is not None:
                            real_close(held_quarantine_descriptor)
                    finally:
                        real_close(parent_descriptor)

        with self.subTest(cleanup_failure="normal-final-identity-replacement"):
            with tempfile.TemporaryDirectory() as name:
                root = Path(name)
                public = root / "artifact"
                public.write_bytes(b"owned\n")
                parent_descriptor = os.open(
                    root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
                )
                artifact_descriptor = os.open(
                    public.name,
                    os.O_RDONLY | os.O_NOFOLLOW,
                    dir_fd=parent_descriptor,
                )
                try:
                    expected = checker._record_artifact(
                        artifact_descriptor, "test artifact"
                    )
                finally:
                    os.close(artifact_descriptor)
                real_close_quarantine = cleanup_module.OwnedDescriptor.close_once
                close_calls = 0

                def replace_after_normal_close(owner: typing.Any) -> None:
                    nonlocal close_calls
                    if not owner.recovery_path.name.endswith(".cleanup"):
                        real_close_quarantine(owner)
                        return
                    close_calls += 1
                    real_close_quarantine(owner)
                    quarantine_path = root / owner.recovery_path
                    quarantine_path.rmdir()
                    quarantine_path.mkdir(mode=0o700)

                try:
                    with (
                        mock.patch.object(
                            cleanup_module.OwnedDescriptor,
                            "close_once",
                            replace_after_normal_close,
                        ),
                        self.assertRaisesRegex(
                            RuntimeError, "recovery material retained"
                        ) as raised,
                    ):
                        checker._claim_and_remove_public_artifact(
                            parent_descriptor,
                            root.resolve(),
                            public.name,
                            expected,
                            "test artifact",
                        )
                    quarantines = list(
                        (cleanup_arena(root)).glob(".artifact.*.cleanup")
                    )
                    self.assertEqual(1, close_calls)
                    self.assertFalse(public.exists())
                    self.assertEqual(1, len(quarantines))
                    self.assertEqual([], list(quarantines[0].iterdir()))
                    self.assertIsNotNone(raised.exception.__cause__)
                    self.assertIn("credential changed", str(raised.exception.__cause__))
                    quarantines[0].rmdir()
                finally:
                    os.close(parent_descriptor)

        with self.subTest(cleanup_failure="normal-close-after-transfer"):
            with tempfile.TemporaryDirectory() as name:
                root = Path(name)
                public = root / "artifact"
                public.write_bytes(b"owned\n")
                parent_descriptor = os.open(
                    root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
                )
                artifact_descriptor = os.open(
                    public.name,
                    os.O_RDONLY | os.O_NOFOLLOW,
                    dir_fd=parent_descriptor,
                )
                try:
                    expected = checker._record_artifact(
                        artifact_descriptor, "test artifact"
                    )
                finally:
                    os.close(artifact_descriptor)
                real_close_quarantine = cleanup_module.OwnedDescriptor.close_once
                real_os_close = os.close
                helper_close_calls = 0
                closed_descriptors: list[int] = []
                quarantine_descriptor: int | None = None
                closed_quarantine: typing.Any = None

                def record_descriptor_close(descriptor: int) -> None:
                    closed_descriptors.append(descriptor)
                    real_os_close(descriptor)

                def fail_after_normal_close(owner: typing.Any) -> None:
                    nonlocal helper_close_calls
                    nonlocal quarantine_descriptor, closed_quarantine
                    if not owner.recovery_path.name.endswith(".cleanup"):
                        real_close_quarantine(owner)
                        return
                    helper_close_calls += 1
                    quarantine_descriptor = owner.fileno()
                    closed_quarantine = owner
                    real_close_quarantine(owner)
                    raise OSError("close failure after ownership transfer")

                try:
                    with (
                        mock.patch.object(
                            checker.os,
                            "close",
                            side_effect=record_descriptor_close,
                        ),
                        mock.patch.object(
                            cleanup_module.OwnedDescriptor,
                            "close_once",
                            fail_after_normal_close,
                        ),
                        self.assertRaisesRegex(
                            RuntimeError,
                            "cleanup quarantine setup failed; recovery material retained as",
                        ) as raised,
                    ):
                        checker._claim_and_remove_public_artifact(
                            parent_descriptor,
                            root.resolve(),
                            public.name,
                            expected,
                            "test artifact",
                        )
                    quarantines = list(
                        (cleanup_arena(root)).glob(".artifact.*.cleanup")
                    )
                    self.assertEqual(1, helper_close_calls)
                    self.assertIsNotNone(quarantine_descriptor)
                    self.assertEqual(
                        1,
                        closed_descriptors.count(
                            typing.cast(int, quarantine_descriptor)
                        ),
                    )
                    self.assertTrue(closed_quarantine.close_attempted)
                    self.assertIsNone(closed_quarantine._descriptor)
                    self.assertFalse(public.exists())
                    self.assertEqual(1, len(quarantines))
                    self.assertEqual([], list(quarantines[0].iterdir()))
                    self.assertIn(quarantines[0].name, str(raised.exception))
                    quarantines[0].rmdir()
                finally:
                    os.close(parent_descriptor)

        with self.subTest(cleanup_failure="rollback-close-after-transfer"):
            with tempfile.TemporaryDirectory() as name:
                root = Path(name)
                destination = root / "artifact"
                destination.write_bytes(b"published\n")
                backup = root / "artifact.backup"
                backup.write_bytes(b"previous\n")
                parent_descriptor = os.open(
                    root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
                )
                destination_descriptor = os.open(
                    destination.name,
                    os.O_RDONLY | os.O_NOFOLLOW,
                    dir_fd=parent_descriptor,
                )
                try:
                    prepared = checker._record_artifact(
                        destination_descriptor, "published test artifact"
                    )
                finally:
                    os.close(destination_descriptor)
                backup_descriptor = os.open(
                    backup.name,
                    os.O_RDONLY | os.O_NOFOLLOW,
                    dir_fd=parent_descriptor,
                )
                try:
                    original = checker._record_artifact(
                        backup_descriptor, "rollback backup"
                    )
                finally:
                    os.close(backup_descriptor)
                real_close_quarantine = cleanup_module.OwnedDescriptor.close_once
                real_os_close = os.close
                real_verify = checker._verify_artifact_name
                helper_close_calls = 0
                restored_observations = 0
                closed_descriptors = []
                quarantine_descriptor = None
                closed_quarantine = None

                def record_rollback_descriptor_close(descriptor: int) -> None:
                    closed_descriptors.append(descriptor)
                    real_os_close(descriptor)

                def observe_restored_destination(
                    directory_descriptor: int,
                    directory_path: Path,
                    entry: str,
                    expected: object,
                    label: str,
                ) -> None:
                    nonlocal restored_observations
                    real_verify(
                        directory_descriptor,
                        directory_path,
                        entry,
                        expected,
                        label,
                    )
                    if label == "restored archive destination":
                        restored_observations += 1

                def fail_after_rollback_close(owner: typing.Any) -> None:
                    nonlocal helper_close_calls
                    nonlocal quarantine_descriptor, closed_quarantine
                    if not owner.recovery_path.name.endswith(".quarantine"):
                        real_close_quarantine(owner)
                        return
                    helper_close_calls += 1
                    quarantine_descriptor = owner.fileno()
                    closed_quarantine = owner
                    real_close_quarantine(owner)
                    raise OSError("close failure after ownership transfer")

                try:
                    with (
                        mock.patch.object(
                            checker.os,
                            "close",
                            side_effect=record_rollback_descriptor_close,
                        ),
                        mock.patch.object(
                            checker,
                            "_verify_artifact_name",
                            side_effect=observe_restored_destination,
                        ),
                        mock.patch.object(
                            cleanup_module.OwnedDescriptor,
                            "close_once",
                            fail_after_rollback_close,
                        ),
                        self.assertRaisesRegex(
                            RuntimeError,
                            "rollback_indeterminate: archive quarantine cleanup failed",
                        ) as raised,
                    ):
                        checker._rollback_publication(
                            parent_descriptor,
                            root.resolve(),
                            destination.name,
                            prepared,
                            backup.name,
                            original,
                        )
                    quarantines = list(
                        (cleanup_arena(root)).glob(".artifact.*.quarantine")
                    )
                    self.assertEqual(1, helper_close_calls)
                    self.assertIsNotNone(quarantine_descriptor)
                    self.assertEqual(
                        1,
                        closed_descriptors.count(
                            typing.cast(int, quarantine_descriptor)
                        ),
                    )
                    self.assertTrue(closed_quarantine.close_attempted)
                    self.assertIsNone(closed_quarantine._descriptor)
                    self.assertEqual(1, restored_observations)
                    self.assertEqual(b"previous\n", destination.read_bytes())
                    self.assertEqual(b"previous\n", backup.read_bytes())
                    self.assertEqual(1, len(quarantines))
                    self.assertEqual([], list(quarantines[0].iterdir()))
                    diagnostic = str(raised.exception)
                    self.assertIn(quarantines[0].name, diagnostic)
                    self.assertIn(backup.name, diagnostic)
                    quarantines[0].rmdir()
                finally:
                    os.close(parent_descriptor)

        for issue_code, uncertainty in (
            ("cleanup_arena_fsync_failed", "durability uncertainty"),
            (
                "cleanup_arena_descriptor_close_uncertain",
                "descriptor-close uncertainty",
            ),
        ):
            with (
                self.subTest(cleanup_failure=issue_code),
                tempfile.TemporaryDirectory() as name,
            ):
                root = Path(name).resolve()
                arena = root / f".zynum-cleanup-v2-{os.geteuid()}"
                outcome = cleanup_module.CleanupOutcome(
                    disposition=cleanup_module.CleanupDisposition.RETAINED,
                    recovery_paths=(),
                    issues=(cleanup_module.CleanupIssue(issue_code, arena),),
                )
                with self.assertRaisesRegex(
                    RuntimeError, f"cleanup arena {uncertainty}"
                ) as raised:
                    checker._raise_cleanup_outcome(
                        outcome,
                        root,
                        "artifact",
                        "claimed",
                        "test artifact",
                    )
                direct_diagnostic = str(raised.exception)
                rollback_diagnostic = checker._rollback_recovery_description(
                    root, None, outcome
                )
                for diagnostic in (direct_diagnostic, rollback_diagnostic):
                    self.assertIn(os.fspath(arena), diagnostic)
                    self.assertNotIn("recovery material", diagnostic.lower())
                    self.assertNotIn("recovery path", diagnostic.lower())

        with self.subTest(cleanup_failure="setup-candidate-and-recovery-paths"):
            with tempfile.TemporaryDirectory() as name:
                root = Path(name).resolve()
                public = root / "artifact"
                residue = (
                    root
                    / f".zynum-cleanup-v2-{os.geteuid()}"
                    / ".artifact.setup.cleanup"
                )
                outcome = cleanup_module.CleanupOutcome(
                    disposition=cleanup_module.CleanupDisposition.RETAINED,
                    recovery_paths=(
                        residue.relative_to(root),
                        residue.relative_to(root),
                    ),
                    issues=(
                        cleanup_module.CleanupIssue(
                            "cleanup_quarantine_setup_failed", residue
                        ),
                    ),
                    candidate_paths=(public, public),
                    public_candidate=cleanup_module.PublicCandidate.PRESENT,
                )
                with self.assertRaises(RuntimeError) as raised:
                    checker._raise_cleanup_outcome(
                        outcome,
                        root,
                        public.name,
                        "claimed",
                        "test artifact",
                    )
                direct_diagnostic = str(raised.exception)
                rollback_diagnostic = checker._rollback_recovery_description(
                    root, None, outcome
                )
                for diagnostic in (direct_diagnostic, rollback_diagnostic):
                    self.assertIn(
                        f"recovery material retained as {residue}",
                        diagnostic,
                    )
                    self.assertIn(
                        f"unclaimed candidate was observed present at {public}",
                        diagnostic,
                    )
                    self.assertEqual(1, diagnostic.count(os.fspath(public)))
                    self.assertEqual(1, diagnostic.count(os.fspath(residue)))

        with self.subTest(cleanup_failure="unaddressable-rebound-candidate"):
            with tempfile.TemporaryDirectory() as name:
                root = Path(name).resolve()
                public = root / "artifact"
                arena = root / f".zynum-cleanup-v2-{os.geteuid()}"
                outcome = cleanup_module.CleanupOutcome(
                    disposition=cleanup_module.CleanupDisposition.UNADDRESSABLE,
                    recovery_paths=(),
                    issues=(
                        cleanup_module.CleanupIssue("cleanup_claimed_foreign", public),
                        cleanup_module.CleanupIssue(
                            "cleanup_arena_binding_rebound", arena
                        ),
                    ),
                    candidate_paths=(public,),
                    arena_binding=cleanup_module.ArenaBinding.REBOUND,
                    public_candidate=cleanup_module.PublicCandidate.PRESENT,
                )
                with self.assertRaises(RuntimeError) as raised:
                    checker._raise_cleanup_outcome(
                        outcome,
                        root,
                        public.name,
                        "claimed",
                        "test artifact",
                    )
                direct_diagnostic = str(raised.exception)
                rollback_diagnostic = checker._rollback_recovery_description(
                    root, None, outcome
                )
                for diagnostic in (direct_diagnostic, rollback_diagnostic):
                    self.assertIn("unclaimed candidate", diagnostic)
                    self.assertIn(os.fspath(public), diagnostic)
                    self.assertIn("unaddressable", diagnostic)
                    self.assertIn("arena binding is rebound", diagnostic)
                    self.assertNotIn("recovery material", diagnostic.lower())
                    self.assertNotIn("recovery path", diagnostic.lower())

    def test_archive_external_destination_replacement_is_not_overwritten(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            payload = root / "payload"
            payload.mkdir()
            (payload / "data.txt").write_text("safe\n", encoding="utf-8")
            destination = root / "source.tar.gz"
            destination.write_bytes(b"previous archive\n")
            run_git(root, "init", "-q")
            snapshot = checker.package_files(root, ["payload"])
            real_verify = checker._verify_artifact_name

            def replace_published(
                directory_descriptor: int,
                directory_path: Path,
                name: str,
                expected: object,
                label: str,
            ) -> None:
                if label == "published archive destination":
                    os.unlink(name, dir_fd=directory_descriptor)
                    descriptor = os.open(
                        name,
                        os.O_CREAT | os.O_EXCL | os.O_WRONLY,
                        0o644,
                        dir_fd=directory_descriptor,
                    )
                    with os.fdopen(descriptor, "wb") as stream:
                        stream.write(b"external writer\n")
                real_verify(
                    directory_descriptor,
                    directory_path,
                    name,
                    expected,
                    label,
                )

            with (
                mock.patch.object(
                    checker, "_verify_artifact_name", side_effect=replace_published
                ),
                self.assertRaisesRegex(RuntimeError, "rollback_indeterminate"),
            ):
                checker._test_only_materialize_archive_transaction(
                    root, snapshot, destination
                )
            self.assertFalse(destination.exists())
            backups = list(root.glob(".source.tar.gz.*.backup"))
            quarantines = list(
                (cleanup_arena(root)).glob(".source.tar.gz.*.quarantine")
            )
            self.assertEqual(1, len(backups))
            self.assertEqual(b"previous archive\n", backups[0].read_bytes())
            self.assertEqual(1, len(quarantines))
            self.assertEqual(
                b"external writer\n",
                (quarantines[0] / "published-artifact").read_bytes(),
            )

    def test_archive_published_digest_mismatch_is_quarantined(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            payload = root / "payload"
            payload.mkdir()
            (payload / "data.txt").write_text("safe\n", encoding="utf-8")
            destination = root / "source.tar.gz"
            destination.write_bytes(b"previous archive\n")
            run_git(root, "init", "-q")
            snapshot = checker.package_files(root, ["payload"])
            real_verify = checker._verify_artifact_name

            def corrupt_published(
                directory_descriptor: int,
                directory_path: Path,
                name: str,
                expected: object,
                label: str,
            ) -> None:
                if label == "published archive destination":
                    descriptor = os.open(name, os.O_RDWR, dir_fd=directory_descriptor)
                    with os.fdopen(descriptor, "r+b") as stream:
                        contents = bytearray(stream.read())
                        contents[-1] ^= 1
                        stream.seek(0)
                        stream.write(contents)
                        stream.truncate()
                real_verify(
                    directory_descriptor,
                    directory_path,
                    name,
                    expected,
                    label,
                )

            with (
                mock.patch.object(
                    checker, "_verify_artifact_name", side_effect=corrupt_published
                ),
                self.assertRaisesRegex(RuntimeError, "rollback_indeterminate"),
            ):
                checker._test_only_materialize_archive_transaction(
                    root, snapshot, destination
                )
            self.assertFalse(destination.exists())
            backups = list(root.glob(".source.tar.gz.*.backup"))
            quarantines = list(
                (cleanup_arena(root)).glob(".source.tar.gz.*.quarantine")
            )
            self.assertEqual(1, len(backups))
            self.assertEqual(b"previous archive\n", backups[0].read_bytes())
            self.assertEqual(1, len(quarantines))
            self.assertTrue((quarantines[0] / "published-artifact").is_file())

        for operation in ("cleanup", "rollback"):
            for sync_failure in ("destination", "source", "both"):
                with (
                    self.subTest(operation=operation, sync_failure=sync_failure),
                    tempfile.TemporaryDirectory() as name,
                ):
                    root = Path(name)
                    parent_descriptor = os.open(
                        root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
                    )
                    public = root / "artifact"
                    public.write_bytes(b"published\n")
                    public_descriptor = os.open(
                        public.name,
                        os.O_RDONLY | os.O_NOFOLLOW,
                        dir_fd=parent_descriptor,
                    )
                    try:
                        expected = checker._record_artifact(
                            public_descriptor, "published test artifact"
                        )
                    finally:
                        os.close(public_descriptor)
                    backup_name: str | None = None
                    original: object | None = None
                    if operation == "rollback":
                        backup = root / "artifact.backup"
                        backup.write_bytes(b"previous\n")
                        backup_descriptor = os.open(
                            backup.name,
                            os.O_RDONLY | os.O_NOFOLLOW,
                            dir_fd=parent_descriptor,
                        )
                        try:
                            original = checker._record_artifact(
                                backup_descriptor, "rollback backup"
                            )
                        finally:
                            os.close(backup_descriptor)
                        backup_name = backup.name
                    sync_order: list[str] = []
                    sync_counts = {"destination": 0, "source": 0}

                    def fail_claim_sync(descriptor: int) -> None:
                        role = (
                            "source"
                            if descriptor == parent_descriptor
                            else "destination"
                        )
                        sync_order.append(role)
                        sync_counts[role] += 1
                        is_claim_sync = (
                            sync_counts[role] > 2
                            if role == "destination"
                            else sync_counts[role] > 1
                        )
                        if is_claim_sync and sync_failure in {role, "both"}:
                            raise OSError(f"{role} fsync failed")

                    try:
                        with (
                            mock.patch.object(
                                checker.os, "fsync", side_effect=fail_claim_sync
                            ),
                            mock.patch.object(
                                checker.repository_snapshot.CleanupQuarantine,
                                "verify_claimed",
                            ) as inspect_artifact,
                            mock.patch.object(
                                checker.repository_snapshot.CleanupQuarantine,
                                "remove_verified_claim",
                            ) as remove_artifact,
                            mock.patch.object(checker.os, "rmdir") as remove_directory,
                            self.assertRaisesRegex(
                                RuntimeError,
                                "destination quarantine|source parent",
                            ),
                        ):
                            if operation == "cleanup":
                                checker._claim_and_remove_public_artifact(
                                    parent_descriptor,
                                    root.resolve(),
                                    public.name,
                                    expected,
                                    "test artifact",
                                )
                            else:
                                checker._rollback_publication(
                                    parent_descriptor,
                                    root.resolve(),
                                    public.name,
                                    expected,
                                    backup_name,
                                    original,
                                )
                        self.assertEqual(
                            [
                                "destination",
                                "source",
                                "destination",
                                "destination",
                                "source",
                            ],
                            sync_order,
                        )
                        inspect_artifact.assert_not_called()
                        remove_artifact.assert_not_called()
                        remove_directory.assert_not_called()
                        self.assertFalse(public.exists())
                        suffix = "cleanup" if operation == "cleanup" else "quarantine"
                        claimed_name = (
                            "claimed"
                            if operation == "cleanup"
                            else "published-artifact"
                        )
                        quarantines = list(
                            (cleanup_arena(root)).glob(f".artifact.*.{suffix}")
                        )
                        self.assertEqual(1, len(quarantines))
                        self.assertEqual(
                            b"published\n",
                            (quarantines[0] / claimed_name).read_bytes(),
                        )
                        if operation == "rollback":
                            self.assertEqual(
                                b"previous\n",
                                (root / typing.cast(str, backup_name)).read_bytes(),
                            )
                        (quarantines[0] / claimed_name).unlink()
                        quarantines[0].rmdir()
                    finally:
                        os.close(parent_descriptor)

        for operation in ("cleanup", "rollback"):
            with (
                self.subTest(operation=operation, recovery=True),
                tempfile.TemporaryDirectory() as name,
            ):
                root = Path(name)
                parent_descriptor = os.open(
                    root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
                )
                public = root / "artifact"
                public.write_bytes(b"published\n")
                public_descriptor = os.open(
                    public.name,
                    os.O_RDONLY | os.O_NOFOLLOW,
                    dir_fd=parent_descriptor,
                )
                try:
                    expected = checker._record_artifact(
                        public_descriptor, "published recovery artifact"
                    )
                finally:
                    os.close(public_descriptor)
                try:
                    if operation == "cleanup":
                        checker._claim_and_remove_public_artifact(
                            parent_descriptor,
                            root.resolve(),
                            public.name,
                            expected,
                            "test artifact",
                        )
                        self.assertFalse(public.exists())
                    else:
                        backup = root / "artifact.backup"
                        backup.write_bytes(b"previous\n")
                        backup_descriptor = os.open(
                            backup.name,
                            os.O_RDONLY | os.O_NOFOLLOW,
                            dir_fd=parent_descriptor,
                        )
                        try:
                            original = checker._record_artifact(
                                backup_descriptor, "rollback recovery backup"
                            )
                        finally:
                            os.close(backup_descriptor)
                        with self.assertRaisesRegex(
                            RuntimeError, "old destination was observed restored"
                        ):
                            checker._rollback_publication(
                                parent_descriptor,
                                root.resolve(),
                                public.name,
                                expected,
                                backup.name,
                                original,
                            )
                        self.assertEqual(b"previous\n", public.read_bytes())
                        self.assertEqual(b"previous\n", backup.read_bytes())
                    self.assertEqual(
                        [],
                        list((cleanup_arena(root)).glob(".artifact.*.cleanup")),
                    )
                    self.assertEqual(
                        [],
                        list((cleanup_arena(root)).glob(".artifact.*.quarantine")),
                    )
                finally:
                    os.close(parent_descriptor)

    def test_archive_absent_rollback_preserves_concurrent_replacement(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            payload = root / "payload"
            payload.mkdir()
            source = payload / "data.txt"
            source.write_text("safe\n", encoding="utf-8")
            run_git(root, "init", "-q")
            snapshot = checker.package_files(root, ["payload"])
            destination = root / "source.tar.gz"
            real_replace = os.replace
            real_verify_absent = checker._verify_name_absent
            published = False
            raced = False

            def mutate_source_after_publish(
                src: object, dst: object, **kwargs: object
            ) -> None:
                nonlocal published
                real_replace(src, dst, **kwargs)  # type: ignore[arg-type]
                if not published:
                    published = True
                    source.write_text("evil\n", encoding="utf-8")

            def replace_after_claim(
                directory_descriptor: int, entry: str, label: str
            ) -> None:
                nonlocal raced
                if label == "archive destination during rollback" and not raced:
                    raced = True
                    destination.write_bytes(b"external writer\n")
                real_verify_absent(directory_descriptor, entry, label)

            with (
                mock.patch.object(
                    checker.os, "replace", side_effect=mutate_source_after_publish
                ),
                mock.patch.object(
                    checker, "_verify_name_absent", side_effect=replace_after_claim
                ),
                self.assertRaisesRegex(RuntimeError, "rollback_indeterminate"),
            ):
                checker._test_only_materialize_archive_transaction(
                    root, snapshot, destination
                )
            self.assertTrue(raced)
            self.assertEqual(b"external writer\n", destination.read_bytes())
            quarantines = list(
                (cleanup_arena(root)).glob(".source.tar.gz.*.quarantine")
            )
            self.assertEqual(1, len(quarantines))
            self.assertTrue((quarantines[0] / "published-artifact").is_file())

    def test_archive_success_removes_transaction_sidecars(self) -> None:
        with self.subTest(release_clean="arena-owner-finalizer-only"):
            adapter_source = (ROOT / "tools/check_package_paths.py").read_text(
                encoding="utf-8"
            )
            self.assertNotIn(".close_issue()", adapter_source)
            self.assertIn(
                "repository_snapshot.finalize_arena_outcome(",
                inspect.getsource(checker._audit_cleanup_arena_empty),
            )

        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            payload = root / "payload"
            payload.mkdir()
            (payload / "data.txt").write_text("safe\n", encoding="utf-8")
            destination = root / "source.tar.gz"
            destination.write_bytes(b"previous archive\n")
            run_git(root, "init", "-q")
            snapshot = checker.package_files(root, ["payload"])
            checker._test_only_materialize_archive_transaction(
                root, snapshot, destination
            )
            first = destination.read_bytes()
            checker._test_only_materialize_archive_transaction(
                root, snapshot, destination
            )
            self.assertEqual(first, destination.read_bytes())
            self.assertEqual([], list(root.glob(".source.tar.gz.*")))
            arena = cleanup_arena(root)
            self.assertTrue(arena.is_dir())
            self.assertEqual([], list(arena.iterdir()))

        with self.subTest(release_clean="entry-retained-child"):
            with tempfile.TemporaryDirectory() as name:
                root = Path(name)
                payload = root / "payload"
                payload.mkdir()
                (payload / "data.txt").write_text("safe\n", encoding="utf-8")
                destination = root / "source.tar.gz"
                previous = b"previous archive\n"
                destination.write_bytes(previous)
                run_git(root, "init", "-q")
                snapshot = checker.package_files(root, ["payload"])
                arena = root / f".zynum-cleanup-v2-{os.geteuid()}"
                arena.mkdir(mode=0o700)
                retained = arena / "foreign-child"
                retained.write_bytes(b"foreign cleanup material\n")

                with self.assertRaisesRegex(
                    RuntimeError, "not release-clean.*foreign-child"
                ) as raised:
                    checker._test_only_materialize_archive_transaction(
                        root, snapshot, destination
                    )

                self.assertEqual(previous, destination.read_bytes())
                self.assertEqual(b"foreign cleanup material\n", retained.read_bytes())
                self.assertEqual([], list(root.glob(".source.tar.gz.*")))
                self.assertIn(os.fspath(retained.resolve()), str(raised.exception))

        for lookalike in ("symlink", "unsafe-mode"):
            with (
                self.subTest(release_clean=f"entry-{lookalike}"),
                tempfile.TemporaryDirectory() as name,
            ):
                root = Path(name)
                payload = root / "payload"
                payload.mkdir()
                (payload / "data.txt").write_text("safe\n", encoding="utf-8")
                destination = root / "source.tar.gz"
                previous = b"previous archive\n"
                destination.write_bytes(previous)
                run_git(root, "init", "-q")
                snapshot = checker.package_files(root, ["payload"])
                arena = root / f".zynum-cleanup-v2-{os.geteuid()}"
                if lookalike == "symlink":
                    foreign = root / "foreign-arena"
                    foreign.mkdir(mode=0o700)
                    marker = foreign / "marker"
                    marker.write_bytes(b"foreign\n")
                    arena.symlink_to(foreign, target_is_directory=True)
                else:
                    arena.mkdir(mode=0o700)
                    arena.chmod(0o755)

                with self.assertRaisesRegex(
                    RuntimeError, "cleanup arena validation failed"
                ) as raised:
                    checker._test_only_materialize_archive_transaction(
                        root, snapshot, destination
                    )

                self.assertEqual(previous, destination.read_bytes())
                self.assertEqual([], list(root.glob(".source.tar.gz.*")))
                self.assertIn(os.fspath(arena.absolute()), str(raised.exception))
                if lookalike == "symlink":
                    self.assertEqual(b"foreign\n", marker.read_bytes())

        with self.subTest(release_clean="entry-v1-marker-ignored"):
            with tempfile.TemporaryDirectory() as name:
                root = Path(name)
                payload = root / "payload"
                payload.mkdir()
                (payload / "data.txt").write_text("safe\n", encoding="utf-8")
                destination = root / "source.tar.gz"
                previous = b"previous archive\n"
                destination.write_bytes(previous)
                run_git(root, "init", "-q")
                snapshot = checker.package_files(root, ["payload"])
                reserved = root / ".zynum-cleanup-v1-foreign"
                reserved.mkdir(mode=0o700)
                marker = reserved / "marker"
                marker.write_bytes(b"foreign\n")
                marker_before = marker.stat()
                real_audit = checker._audit_cleanup_arena_empty
                real_scandir = os.scandir

                def audit_without_parent_enumeration(
                    directory_descriptor: int,
                    directory_path: Path,
                    stage: str,
                    **kwargs: object,
                ) -> tuple[int, int]:
                    parent = os.fstat(directory_descriptor)

                    def reject_parent_scandir(target: object) -> object:
                        if type(target) is int:
                            observed = os.fstat(target)
                            if (observed.st_dev, observed.st_ino) == (
                                parent.st_dev,
                                parent.st_ino,
                            ):
                                raise AssertionError(
                                    "package audit enumerated the destination parent"
                                )
                        return real_scandir(target)  # type: ignore[arg-type]

                    with mock.patch.object(
                        checker.os,
                        "scandir",
                        side_effect=reject_parent_scandir,
                    ):
                        return real_audit(
                            directory_descriptor,
                            directory_path,
                            stage,
                            **kwargs,  # type: ignore[arg-type]
                        )

                with mock.patch.object(
                    checker,
                    "_audit_cleanup_arena_empty",
                    side_effect=audit_without_parent_enumeration,
                ):
                    checker._test_only_materialize_archive_transaction(
                        root, snapshot, destination
                    )

                self.assertNotEqual(previous, destination.read_bytes())
                self.assertEqual(b"foreign\n", marker.read_bytes())
                marker_after = marker.stat()
                self.assertEqual(
                    (
                        marker_before.st_dev,
                        marker_before.st_ino,
                        marker_before.st_mode,
                        marker_before.st_size,
                        marker_before.st_mtime_ns,
                    ),
                    (
                        marker_after.st_dev,
                        marker_after.st_ino,
                        marker_after.st_mode,
                        marker_after.st_size,
                        marker_after.st_mtime_ns,
                    ),
                )
                self.assertTrue(reserved.is_dir())
                self.assertEqual([], list(root.glob(".source.tar.gz.*")))
                self.assertEqual([], list(cleanup_arena(root).iterdir()))

        for audit_failure in ("inspection", "fsync"):
            with (
                self.subTest(release_clean=audit_failure),
                tempfile.TemporaryDirectory() as name,
            ):
                root = Path(name).resolve()
                parent_descriptor = os.open(
                    root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
                )
                try:
                    checker._audit_cleanup_arena_empty(
                        parent_descriptor, root, "during test setup"
                    )
                    arena = cleanup_arena(root)
                    patch_target = (
                        "scandir" if audit_failure == "inspection" else "fsync"
                    )
                    with (
                        mock.patch.object(
                            checker.os,
                            patch_target,
                            side_effect=OSError(f"arena {audit_failure} failed"),
                        ),
                        self.assertRaisesRegex(
                            RuntimeError, "not release-clean"
                        ) as raised,
                    ):
                        checker._audit_cleanup_arena_empty(
                            parent_descriptor, root, "during fault injection"
                        )
                    self.assertIn(os.fspath(arena), str(raised.exception))
                    self.assertEqual([], list(arena.iterdir()))
                finally:
                    os.close(parent_descriptor)

        for boundary in ("entry", "exit"):
            with (
                self.subTest(release_clean=f"{boundary}-post-scan-rebind"),
                tempfile.TemporaryDirectory() as name,
            ):
                root = Path(name).resolve()
                parent_descriptor = os.open(
                    root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
                )
                expected_identity: tuple[int, int] | None = None
                try:
                    if boundary == "exit":
                        expected_identity = checker._audit_cleanup_arena_empty(
                            parent_descriptor, root, "during test setup"
                        )
                    arena = (
                        cleanup_arena(root)
                        if expected_identity is not None
                        else root / f".zynum-cleanup-v2-{os.geteuid()}"
                    )
                    real_scandir = os.scandir
                    identities: list[tuple[int, int]] = []

                    class RebindAfterScan:
                        def __init__(self, entries: typing.Any) -> None:
                            self.entries = entries

                        def __enter__(self) -> typing.Any:
                            return self.entries.__enter__()

                        def __exit__(self, *args: object) -> object:
                            result = self.entries.__exit__(*args)
                            before = arena.stat()
                            identities.append((before.st_dev, before.st_ino))
                            arena.rmdir()
                            (root / f"{boundary}-arena-rebind-displacer").mkdir()
                            arena.mkdir(mode=0o700)
                            after = arena.stat()
                            identities.append((after.st_dev, after.st_ino))
                            return result

                    def rebind_after_scan(target: object) -> object:
                        return RebindAfterScan(
                            real_scandir(target)  # type: ignore[arg-type]
                        )

                    options: dict[str, object] = {}
                    if expected_identity is not None:
                        options = {
                            "create": False,
                            "expected_identity": expected_identity,
                        }
                    with (
                        mock.patch.object(
                            checker.os,
                            "scandir",
                            side_effect=rebind_after_scan,
                        ),
                        self.assertRaisesRegex(
                            RuntimeError,
                            "not release-clean.*binding changed after inspection.*rebound",
                        ),
                    ):
                        checker._audit_cleanup_arena_empty(
                            parent_descriptor,
                            root,
                            f"at simulated publication {boundary}",
                            **options,  # type: ignore[arg-type]
                        )
                    self.assertEqual(2, len(identities))
                    self.assertNotEqual(identities[0], identities[1])
                    self.assertEqual([], list(cleanup_arena(root).iterdir()))
                finally:
                    os.close(parent_descriptor)

        for boundary in ("entry", "exit"):
            with (
                self.subTest(release_clean=f"{boundary}-close-time-rebind"),
                tempfile.TemporaryDirectory() as name,
            ):
                root = Path(name).resolve()
                parent_descriptor = os.open(
                    root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
                )
                expected_identity: tuple[int, int] | None = None
                close_calls = 0
                real_close_once = checker.repository_snapshot.OwnedDescriptor.close_once
                try:
                    if boundary == "exit":
                        expected_identity = checker._audit_cleanup_arena_empty(
                            parent_descriptor, root, "during test setup"
                        )
                    arena = (
                        cleanup_arena(root)
                        if expected_identity is not None
                        else root / f".zynum-cleanup-v2-{os.geteuid()}"
                    )
                    identities: list[tuple[int, int]] = []

                    def rebind_during_arena_close(owner: typing.Any) -> None:
                        nonlocal close_calls
                        if owner.recovery_path != arena:
                            real_close_once(owner)
                            return
                        close_calls += 1
                        real_close_once(owner)
                        before = arena.stat()
                        identities.append((before.st_dev, before.st_ino))
                        arena.rmdir()
                        (root / f"{boundary}-close-rebind-displacer").mkdir()
                        arena.mkdir(mode=0o700)
                        after = arena.stat()
                        identities.append((after.st_dev, after.st_ino))

                    options: dict[str, object] = {}
                    if expected_identity is not None:
                        options = {
                            "create": False,
                            "expected_identity": expected_identity,
                        }
                    with (
                        mock.patch.object(
                            checker.repository_snapshot.OwnedDescriptor,
                            "close_once",
                            rebind_during_arena_close,
                        ),
                        self.assertRaisesRegex(
                            RuntimeError,
                            "not release-clean.*binding changed after descriptor close.*rebound",
                        ),
                    ):
                        checker._audit_cleanup_arena_empty(
                            parent_descriptor,
                            root,
                            f"at simulated publication {boundary}",
                            **options,  # type: ignore[arg-type]
                        )
                    self.assertEqual(1, close_calls)
                    self.assertEqual(2, len(identities))
                    self.assertNotEqual(identities[0], identities[1])
                    self.assertEqual([], list(cleanup_arena(root).iterdir()))
                finally:
                    os.close(parent_descriptor)

        with self.subTest(release_clean="descriptor-close-fd-reuse"):
            with tempfile.TemporaryDirectory() as name:
                root = Path(name).resolve()
                parent_descriptor = os.open(
                    root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
                )
                reused_descriptor: int | None = None
                close_calls = 0
                real_close_once = checker.repository_snapshot.OwnedDescriptor.close_once
                real_close = os.close
                try:
                    checker._audit_cleanup_arena_empty(
                        parent_descriptor, root, "during test setup"
                    )
                    arena = cleanup_arena(root)

                    def fail_arena_close_after_reuse(owner: typing.Any) -> None:
                        nonlocal close_calls, reused_descriptor
                        if owner.recovery_path != arena:
                            real_close_once(owner)
                            return
                        close_calls += 1
                        descriptor = owner.fileno()
                        real_close_once(owner)
                        candidate = os.open(
                            root / "arena-close-reuse",
                            os.O_CREAT | os.O_EXCL | os.O_WRONLY,
                            0o600,
                        )
                        if candidate != descriptor:
                            os.dup2(candidate, descriptor)
                            real_close(candidate)
                        reused_descriptor = descriptor
                        raise OSError("arena close failed after descriptor reuse")

                    with (
                        mock.patch.object(
                            checker.repository_snapshot.OwnedDescriptor,
                            "close_once",
                            fail_arena_close_after_reuse,
                        ),
                        self.assertRaisesRegex(
                            RuntimeError, "descriptor close failed"
                        ) as raised,
                    ):
                        checker._audit_cleanup_arena_empty(
                            parent_descriptor, root, "during close fault injection"
                        )
                    self.assertEqual(1, close_calls)
                    self.assertIsNotNone(reused_descriptor)
                    os.fstat(typing.cast(int, reused_descriptor))
                    self.assertIn(os.fspath(arena), str(raised.exception))
                finally:
                    if reused_descriptor is not None:
                        real_close(reused_descriptor)
                    real_close(parent_descriptor)

        with self.subTest(release_clean="exit-retained-child"):
            with tempfile.TemporaryDirectory() as name:
                root = Path(name)
                payload = root / "payload"
                payload.mkdir()
                (payload / "data.txt").write_text("safe\n", encoding="utf-8")
                destination = root / "source.tar.gz"
                destination.write_bytes(b"previous archive\n")
                run_git(root, "init", "-q")
                snapshot = checker.package_files(root, ["payload"])
                real_cleanup_backup = checker._cleanup_backup_after_transaction
                retained: Path | None = None

                def inject_child_after_backup_cleanup(
                    directory_descriptor: int,
                    directory_path: Path,
                    backup_name: str,
                    original: object,
                ) -> None:
                    nonlocal retained
                    real_cleanup_backup(
                        directory_descriptor,
                        directory_path,
                        backup_name,
                        original,
                    )
                    retained = cleanup_arena(root) / "foreign-exit-child"
                    retained.write_bytes(b"foreign cleanup material\n")

                with (
                    mock.patch.object(
                        checker,
                        "_cleanup_backup_after_transaction",
                        side_effect=inject_child_after_backup_cleanup,
                    ),
                    self.assertRaisesRegex(
                        RuntimeError, "not release-clean.*foreign-exit-child"
                    ) as raised,
                ):
                    checker._test_only_materialize_archive_transaction(
                        root, snapshot, destination
                    )

                self.assertIsNotNone(retained)
                retained_path = typing.cast(Path, retained)
                self.assertEqual(
                    b"foreign cleanup material\n", retained_path.read_bytes()
                )
                self.assertIn(os.fspath(retained_path), str(raised.exception))
                self.assertEqual([], list(root.glob(".source.tar.gz.*")))
                with tarfile.open(destination, "r:gz") as archive:
                    self.assertEqual(["payload/data.txt"], archive.getnames())

        with self.subTest(release_clean="exit-arena-rebound"):
            with tempfile.TemporaryDirectory() as name:
                root = Path(name)
                payload = root / "payload"
                payload.mkdir()
                (payload / "data.txt").write_text("safe\n", encoding="utf-8")
                destination = root / "source.tar.gz"
                destination.write_bytes(b"previous archive\n")
                run_git(root, "init", "-q")
                snapshot = checker.package_files(root, ["payload"])
                real_cleanup_backup = checker._cleanup_backup_after_transaction
                real_open = os.open
                real_close = os.close
                held_arena_descriptor: int | None = None
                identities: list[tuple[int, int]] = []

                def replace_arena_after_backup_cleanup(
                    directory_descriptor: int,
                    directory_path: Path,
                    backup_name: str,
                    original: object,
                ) -> None:
                    nonlocal held_arena_descriptor
                    real_cleanup_backup(
                        directory_descriptor,
                        directory_path,
                        backup_name,
                        original,
                    )
                    arena = cleanup_arena(root)
                    if held_arena_descriptor is not None:
                        raise AssertionError("cleanup arena was rebound more than once")
                    held_arena_descriptor = real_open(
                        arena, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
                    )
                    before = os.fstat(held_arena_descriptor)
                    identities.append((before.st_dev, before.st_ino))
                    arena.rmdir()
                    (root / "arena-rebind-displacer").mkdir()
                    arena.mkdir(mode=0o700)
                    after = arena.stat()
                    identities.append((after.st_dev, after.st_ino))

                try:
                    with (
                        mock.patch.object(
                            checker,
                            "_cleanup_backup_after_transaction",
                            side_effect=replace_arena_after_backup_cleanup,
                        ),
                        self.assertRaisesRegex(
                            RuntimeError, "cleanup arena validation failed"
                        ) as raised,
                    ):
                        checker._test_only_materialize_archive_transaction(
                            root, snapshot, destination
                        )

                    self.assertEqual(2, len(identities))
                    self.assertNotEqual(identities[0], identities[1])
                    arena = cleanup_arena(root)
                    self.assertEqual([], list(arena.iterdir()))
                    diagnostic = str(raised.exception)
                    self.assertIn(os.fspath(arena), diagnostic)
                    self.assertIn("arena binding is rebound", diagnostic)
                    self.assertNotIn("recovery material", diagnostic.lower())
                    self.assertEqual([], list(root.glob(".source.tar.gz.*")))
                    with tarfile.open(destination, "r:gz") as archive:
                        self.assertEqual(["payload/data.txt"], archive.getnames())
                finally:
                    if held_arena_descriptor is not None:
                        real_close(held_arena_descriptor)

    def test_archive_parent_swap_fails_closed_at_every_publication_boundary(
        self,
    ) -> None:
        stages = (
            "immediately before archive publication",
            "immediately after archive publication",
            "at archive publication success boundary",
        )
        for attack in ("external-symlink", "replacement-aba"):
            for stage in stages:
                for existing in (False, True):
                    with (
                        self.subTest(attack=attack, stage=stage, existing=existing),
                        tempfile.TemporaryDirectory() as name,
                    ):
                        base = Path(name)
                        root = base / "repository"
                        payload = root / "payload"
                        publish = base / "publish"
                        external = base / "external"
                        payload.mkdir(parents=True)
                        publish.mkdir()
                        external.mkdir()
                        (payload / "data.txt").write_text("safe\n", encoding="utf-8")
                        destination = publish / "source.tar.gz"
                        if existing:
                            destination.write_bytes(b"previous archive\n")
                        run_git(root, "init", "-q")
                        snapshot = checker.package_files(root, ["payload"])
                        previous = destination.read_bytes() if existing else None
                        held = base / "held-publish"
                        real_verify = checker._verify_public_parent
                        attacked = False

                        def swap_parent(
                            parent: Path, descriptor: int, observed_stage: str
                        ) -> None:
                            nonlocal attacked
                            if observed_stage == stage and not attacked:
                                attacked = True
                                publish.rename(held)
                                if attack == "external-symlink":
                                    publish.symlink_to(
                                        external, target_is_directory=True
                                    )
                                else:
                                    publish.mkdir()
                            real_verify(parent, descriptor, observed_stage)

                        with (
                            mock.patch.object(
                                checker,
                                "_verify_public_parent",
                                side_effect=swap_parent,
                            ),
                            self.assertRaisesRegex(
                                RuntimeError,
                                "unaddressable.*arena binding is rebound",
                            ) as raised,
                        ):
                            checker._test_only_materialize_archive_transaction(
                                root, snapshot, destination
                            )

                        self.assertTrue(attacked)
                        held_destination = held / destination.name
                        if stage == stages[0]:
                            if previous is None:
                                self.assertFalse(held_destination.exists())
                            else:
                                self.assertEqual(
                                    previous, held_destination.read_bytes()
                                )
                        else:
                            self.assertTrue(held_destination.is_file())
                            if previous is not None:
                                self.assertNotEqual(
                                    previous, held_destination.read_bytes()
                                )
                            with tarfile.open(held_destination, "r:gz") as archive:
                                self.assertEqual(
                                    ["payload/data.txt"], archive.getnames()
                                )
                        self.assertFalse((external / destination.name).exists())
                        backups = list(held.glob(f".{destination.name}.*.backup"))
                        temporaries = list(held.glob(f".{destination.name}.*.tmp"))
                        if existing:
                            self.assertEqual(1, len(backups))
                            self.assertEqual(previous, backups[0].read_bytes())
                        else:
                            self.assertEqual([], backups)
                        if stage == stages[0]:
                            self.assertEqual(1, len(temporaries))
                            self.assertIn(temporaries[0].name, str(raised.exception))
                            with tarfile.open(temporaries[0], "r:gz") as archive:
                                self.assertEqual(
                                    ["payload/data.txt"], archive.getnames()
                                )
                        else:
                            self.assertEqual([], temporaries)
                        if attack == "replacement-aba":
                            self.assertEqual(
                                [], list(publish.glob(f".{destination.name}.*"))
                            )

    def test_archive_existing_destination_durability_failure_rolls_back(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            payload = root / "payload"
            payload.mkdir()
            (payload / "data.txt").write_text("safe\n", encoding="utf-8")
            destination = root / "source.tar.gz"
            destination.write_bytes(b"previous archive\n")
            run_git(root, "init", "-q")
            snapshot = checker.package_files(root, ["payload"])
            real_fsync = os.fsync
            calls = 0

            def fail_existing_publication_fsync(descriptor: int) -> None:
                nonlocal calls
                calls += 1
                if calls == 7:
                    raise OSError("existing publication fsync failed")
                real_fsync(descriptor)

            with (
                mock.patch.object(
                    checker.os,
                    "fsync",
                    side_effect=fail_existing_publication_fsync,
                ),
                self.assertRaisesRegex(RuntimeError, "rollback_indeterminate"),
            ):
                checker._test_only_materialize_archive_transaction(
                    root, snapshot, destination
                )
            self.assertEqual(b"previous archive\n", destination.read_bytes())
            backups = list(root.glob(".source.tar.gz.*.backup"))
            self.assertEqual(1, len(backups))
            self.assertEqual(b"previous archive\n", backups[0].read_bytes())

    def test_archive_backup_cleanup_failure_keeps_new_archive_and_recovery(
        self,
    ) -> None:
        for failure in (
            "private-unlink",
            "fsync",
            "replacement-before-helper",
            "replacement-before-claim",
            "replacement-after-claim",
        ):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as name:
                root = Path(name)
                payload = root / "payload"
                payload.mkdir()
                (payload / "data.txt").write_text("safe\n", encoding="utf-8")
                destination = root / "source.tar.gz"
                destination.write_bytes(b"previous archive\n")
                run_git(root, "init", "-q")
                snapshot = checker.package_files(root, ["payload"])
                real_unlink = os.unlink
                real_fsync = os.fsync
                real_rename = os.rename
                real_cleanup_backup = checker._cleanup_backup_after_transaction
                fsync_calls = 0
                foreign = b"foreign cleanup writer\n"
                cleanup_public_name: str | None = None

                def fail_backup_unlink(
                    path: object, *args: object, **kwargs: object
                ) -> None:
                    if failure == "private-unlink" and os.fspath(path) == "claimed":
                        raise PermissionError("claimed backup unlink denied")
                    real_unlink(path, *args, **kwargs)  # type: ignore[arg-type]

                def fail_backup_cleanup_fsync(descriptor: int) -> None:
                    nonlocal fsync_calls
                    fsync_calls += 1
                    if failure == "fsync" and fsync_calls == 12:
                        raise OSError("backup cleanup fsync failed")
                    real_fsync(descriptor)

                def replace_backup_at_claim(
                    source: object, target: object, **kwargs: object
                ) -> None:
                    if (
                        failure
                        in {
                            "replacement-before-claim",
                            "replacement-after-claim",
                        }
                        and os.fspath(source).endswith(".backup")
                        and os.fspath(target) == "claimed"
                    ):
                        public_backup = root / os.fspath(source)
                        if failure == "replacement-before-claim":
                            public_backup.unlink()
                            public_backup.write_bytes(foreign)
                            real_rename(source, target, **kwargs)  # type: ignore[arg-type]
                            return
                        real_rename(source, target, **kwargs)  # type: ignore[arg-type]
                        public_backup.write_bytes(foreign)
                        return
                    real_rename(source, target, **kwargs)  # type: ignore[arg-type]

                def replace_backup_before_helper(
                    directory_descriptor: int,
                    directory_path: Path,
                    backup_name: str,
                    original: object,
                ) -> None:
                    nonlocal cleanup_public_name
                    cleanup_public_name = backup_name
                    if failure == "replacement-before-helper":
                        public_backup = root / backup_name
                        public_backup.unlink()
                        public_backup.write_bytes(foreign)
                    real_cleanup_backup(
                        directory_descriptor,
                        directory_path,
                        backup_name,
                        original,
                    )

                messages = {
                    "private-unlink": "transaction complete.*cleanup failed.*recovery material retained",
                    "fsync": "transaction complete.*cleanup failed.*cleanup arena",
                    "replacement-before-helper": "transaction complete.*unexpected artifact.*recovery material retained",
                    "replacement-before-claim": "transaction complete.*unexpected artifact.*recovery material retained",
                    "replacement-after-claim": "transaction complete.*pathname reappeared.*preserved",
                }
                with (
                    mock.patch.object(
                        checker.os, "unlink", side_effect=fail_backup_unlink
                    ),
                    mock.patch.object(
                        checker.os, "fsync", side_effect=fail_backup_cleanup_fsync
                    ),
                    mock.patch.object(
                        checker.os,
                        "rename",
                        side_effect=replace_backup_at_claim,
                    ),
                    mock.patch.object(
                        checker,
                        "_cleanup_backup_after_transaction",
                        side_effect=replace_backup_before_helper,
                    ),
                    mock.patch.object(checker, "_require_publication_support"),
                    self.assertRaisesRegex(RuntimeError, messages[failure]) as raised,
                ):
                    checker._test_only_materialize_archive_transaction(
                        root, snapshot, destination
                    )

                self.assertNotEqual(b"previous archive\n", destination.read_bytes())
                with tarfile.open(destination, "r:gz") as archive:
                    self.assertEqual(["payload/data.txt"], archive.getnames())
                backups = list(root.glob(".source.tar.gz.*.backup"))
                cleanups = list((cleanup_arena(root)).glob(".*.cleanup"))
                if failure == "private-unlink":
                    self.assertEqual([], backups)
                    self.assertEqual(1, len(cleanups))
                    recovery = cleanups[0] / "claimed"
                    self.assertEqual(b"previous archive\n", recovery.read_bytes())
                    self.assertIn(f"{cleanups[0].name}/claimed", str(raised.exception))
                elif failure in {
                    "replacement-before-helper",
                    "replacement-before-claim",
                }:
                    self.assertEqual([], backups)
                    self.assertEqual(1, len(cleanups))
                    recovery = cleanups[0] / "claimed"
                    self.assertEqual(foreign, recovery.read_bytes())
                    self.assertIn(f"{cleanups[0].name}/claimed", str(raised.exception))
                    if failure == "replacement-before-helper":
                        self.assertIsNotNone(cleanup_public_name)
                        self.assertIn(
                            typing.cast(str, cleanup_public_name),
                            str(raised.exception),
                        )
                elif failure == "replacement-after-claim":
                    self.assertEqual(1, len(backups))
                    self.assertEqual(foreign, backups[0].read_bytes())
                    self.assertEqual(1, len(cleanups))
                    recovery = cleanups[0] / "claimed"
                    self.assertEqual(b"previous archive\n", recovery.read_bytes())
                    self.assertIn(f"{cleanups[0].name}/claimed", str(raised.exception))
                else:
                    self.assertEqual([], backups)
                    self.assertEqual([], cleanups)
                    diagnostic = str(raised.exception)
                    self.assertIn("cleanup arena durability uncertainty", diagnostic)
                    self.assertIn(os.fspath(cleanup_arena(root)), diagnostic)
                    self.assertNotIn("recovery material", diagnostic.lower())
                    self.assertNotIn("recovery path", diagnostic.lower())

        for finalization in ("transaction-complete", "failed-rollback"):
            with (
                self.subTest(finalization=finalization),
                tempfile.TemporaryDirectory() as name,
            ):
                root = Path(name)
                payload = root / "payload"
                payload.mkdir()
                source = payload / "data.txt"
                source.write_text("safe\n", encoding="utf-8")
                destination = root / "source.tar.gz"
                destination.write_bytes(b"previous archive\n")
                run_git(root, "init", "-q")
                snapshot = checker.package_files(root, ["payload"])
                real_open_name = checker._open_name_no_follow
                real_close = os.close
                real_replace = os.replace
                original_descriptor: int | None = None
                reused_descriptor: int | None = None
                original_close_calls = 0
                publication_replaced = False

                def capture_original_descriptor(
                    directory_descriptor: int,
                    directory_path: Path,
                    public_name: str,
                    label: str,
                ) -> object:
                    nonlocal original_descriptor
                    owner = real_open_name(
                        directory_descriptor, directory_path, public_name, label
                    )
                    if label == "archive destination" and original_descriptor is None:
                        original_descriptor = owner.fileno()
                    return owner

                def fail_original_close_after_reuse(descriptor: int) -> None:
                    nonlocal original_close_calls, reused_descriptor
                    if descriptor == original_descriptor:
                        original_close_calls += 1
                    real_close(descriptor)
                    if descriptor == original_descriptor and reused_descriptor is None:
                        candidate_descriptor = os.open(
                            root / "original-close-reuse",
                            os.O_CREAT | os.O_EXCL | os.O_WRONLY,
                            0o600,
                        )
                        if candidate_descriptor != descriptor:
                            os.dup2(candidate_descriptor, descriptor)
                            real_close(candidate_descriptor)
                        reused_descriptor = descriptor
                        raise OSError("original close failed after descriptor reuse")

                def mutate_source_after_publication(
                    source_name: object,
                    destination_name: object,
                    **kwargs: object,
                ) -> None:
                    nonlocal publication_replaced
                    real_replace(source_name, destination_name, **kwargs)  # type: ignore[arg-type]
                    if (
                        finalization == "failed-rollback"
                        and not publication_replaced
                        and os.fspath(destination_name) == destination.name
                    ):
                        publication_replaced = True
                        source.write_text("evil\n", encoding="utf-8")

                message = (
                    "archive transaction complete.*original archive destination "
                    "descriptor close failed"
                    if finalization == "transaction-complete"
                    else "rollback_indeterminate.*old destination was observed "
                    "restored.*original archive destination descriptor close failed"
                )
                with (
                    mock.patch.object(
                        checker,
                        "_open_name_no_follow",
                        side_effect=capture_original_descriptor,
                    ),
                    mock.patch.object(
                        checker.os,
                        "replace",
                        side_effect=mutate_source_after_publication,
                    ),
                    mock.patch.object(
                        checker.os,
                        "close",
                        side_effect=fail_original_close_after_reuse,
                    ),
                    mock.patch.object(
                        checker, "_cleanup_backup_after_transaction"
                    ) as cleanup_backup,
                    self.assertRaisesRegex(RuntimeError, message) as raised,
                ):
                    checker._test_only_materialize_archive_transaction(
                        root, snapshot, destination
                    )

                backups = list(root.glob(".source.tar.gz.*.backup"))
                self.assertEqual(1, len(backups))
                self.assertIn(
                    os.fspath(root.resolve() / backups[0].name),
                    str(raised.exception),
                )
                self.assertEqual(1, original_close_calls)
                self.assertIsNotNone(reused_descriptor)
                os.fstat(typing.cast(int, reused_descriptor))
                real_close(typing.cast(int, reused_descriptor))
                cleanup_backup.assert_not_called()
                if finalization == "transaction-complete":
                    self.assertNotEqual(b"previous archive\n", destination.read_bytes())
                else:
                    self.assertTrue(publication_replaced)
                    self.assertEqual(b"previous archive\n", destination.read_bytes())

    def test_archive_sidecar_operations_use_anchored_directories(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            payload = root / "payload"
            payload.mkdir()
            (payload / "data.txt").write_text("safe\n", encoding="utf-8")
            destination = root / "source.tar.gz"
            destination.write_bytes(b"previous archive\n")
            run_git(root, "init", "-q")
            snapshot = checker.package_files(root, ["payload"])
            observed: list[tuple[str, str, int]] = []
            real_open = os.open
            real_link = os.link
            real_replace = os.replace
            real_unlink = os.unlink

            def relevant(path: object) -> bool:
                value = os.fspath(path)
                return value == destination.name or value.startswith(
                    f".{destination.name}."
                )

            def traced_open(
                path: object,
                flags: int,
                mode: int = 0o777,
                *,
                dir_fd: int | None = None,
            ) -> int:
                if relevant(path):
                    self.assertIsNotNone(dir_fd)
                    observed.append(("open", os.fspath(path), dir_fd))  # type: ignore[arg-type]
                return real_open(path, flags, mode, dir_fd=dir_fd)  # type: ignore[arg-type]

            def traced_link(
                source: object,
                destination_path: object,
                *,
                src_dir_fd: int | None = None,
                dst_dir_fd: int | None = None,
                follow_symlinks: bool = True,
            ) -> None:
                self.assertTrue(relevant(source))
                self.assertTrue(relevant(destination_path))
                self.assertEqual(src_dir_fd, dst_dir_fd)
                self.assertIsNotNone(src_dir_fd)
                self.assertFalse(follow_symlinks)
                observed.append(("link", os.fspath(source), src_dir_fd))  # type: ignore[arg-type]
                real_link(
                    source,
                    destination_path,
                    src_dir_fd=src_dir_fd,
                    dst_dir_fd=dst_dir_fd,
                    follow_symlinks=follow_symlinks,
                )

            def traced_replace(
                source: object,
                destination_path: object,
                *,
                src_dir_fd: int | None = None,
                dst_dir_fd: int | None = None,
            ) -> None:
                self.assertTrue(relevant(source))
                self.assertIsNotNone(src_dir_fd)
                self.assertIsNotNone(dst_dir_fd)
                if os.fspath(destination_path) == "claimed":
                    self.assertNotEqual(src_dir_fd, dst_dir_fd)
                else:
                    self.assertTrue(relevant(destination_path))
                    self.assertEqual(src_dir_fd, dst_dir_fd)
                observed.append(("replace", os.fspath(source), src_dir_fd))  # type: ignore[arg-type]
                real_replace(
                    source,
                    destination_path,
                    src_dir_fd=src_dir_fd,
                    dst_dir_fd=dst_dir_fd,
                )

            def traced_unlink(path: object, *, dir_fd: int | None = None) -> None:
                if relevant(path) or os.fspath(path) == "claimed":
                    self.assertIsNotNone(dir_fd)
                    observed.append(("unlink", os.fspath(path), dir_fd))  # type: ignore[arg-type]
                real_unlink(path, dir_fd=dir_fd)  # type: ignore[arg-type]

            with (
                mock.patch.object(checker.os, "open", side_effect=traced_open),
                mock.patch.object(checker.os, "link", side_effect=traced_link),
                mock.patch.object(checker.os, "replace", side_effect=traced_replace),
                mock.patch.object(checker.os, "unlink", side_effect=traced_unlink),
                mock.patch.object(
                    checker.repository_snapshot,
                    "_require_descriptor_relative_io",
                ),
                mock.patch.object(checker, "_require_publication_support"),
            ):
                checker._test_only_materialize_archive_transaction(
                    root, snapshot, destination
                )

            self.assertTrue(
                {"open", "link", "replace", "unlink"}.issubset(
                    {operation for operation, _, _ in observed}
                )
            )
            self.assertEqual(2, len({descriptor for _, _, descriptor in observed}))
            self.assertTrue(
                all(Path(component).name == component for _, component, _ in observed)
            )

    def test_archive_destination_hardlink_replacement_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            payload = root / "payload"
            payload.mkdir()
            (payload / "data.txt").write_text("safe\n", encoding="utf-8")
            destination = root / "source.tar.gz"
            destination.write_bytes(b"previous archive\n")
            external = root / "external.tar.gz"
            external.write_bytes(b"external writer\n")
            run_git(root, "init", "-q")
            snapshot = checker.package_files(root, ["payload"])
            real_verify = checker._verify_public_parent
            attacked = False

            def inject_hardlink(parent: Path, descriptor: int, stage: str) -> None:
                nonlocal attacked
                real_verify(parent, descriptor, stage)
                if stage == "immediately before archive publication":
                    attacked = True
                    destination.unlink()
                    os.link(external, destination)

            with (
                mock.patch.object(
                    checker, "_verify_public_parent", side_effect=inject_hardlink
                ),
                self.assertRaisesRegex(RuntimeError, "rollback_indeterminate"),
            ):
                checker._test_only_materialize_archive_transaction(
                    root, snapshot, destination
                )

            self.assertTrue(attacked)
            self.assertEqual(b"external writer\n", external.read_bytes())
            self.assertEqual(b"external writer\n", destination.read_bytes())
            backups = list(root.glob(".source.tar.gz.*.backup"))
            self.assertEqual(1, len(backups))
            self.assertEqual(b"previous archive\n", backups[0].read_bytes())

    def test_archive_rejects_member_replaced_after_enumeration(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            (root / "payload").mkdir()
            target = root / "payload/data.txt"
            target.write_text("safe\n", encoding="utf-8")
            external = root / "external.txt"
            external.write_text("private\n", encoding="utf-8")
            subprocess.run(("git", "init", "-q"), cwd=root, check=True)
            files = checker.package_files(root, ["payload"])
            target.unlink()
            try:
                target.symlink_to(external)
            except OSError as exc:
                self.skipTest(f"symlinks are unavailable: {exc}")
            destination = root / "source.tar.gz"
            with self.assertRaisesRegex(ValueError, "changed after enumeration"):
                checker.create_archive(root, files, destination)
            self.assertFalse(destination.exists())

    def test_archive_rejects_member_replaced_during_addfile(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            (root / "payload").mkdir()
            target = root / "payload/data.txt"
            target.write_text("safe\n", encoding="utf-8")
            run_git(root, "init", "-q")
            files = checker.package_files(root, ["payload"])
            real_addfile = tarfile.TarFile.addfile

            def add_then_replace(
                archive: tarfile.TarFile,
                tarinfo: tarfile.TarInfo,
                fileobj: object = None,
            ) -> None:
                real_addfile(archive, tarinfo, fileobj)  # type: ignore[arg-type]
                if tarinfo.name == "payload/data.txt":
                    target.rename(root / "original-data.txt")
                    target.write_text("evil\n", encoding="utf-8")

            destination = root / "source.tar.gz"
            with (
                mock.patch.object(
                    checker.tarfile.TarFile,
                    "addfile",
                    autospec=True,
                    side_effect=add_then_replace,
                ),
                self.assertRaisesRegex(ValueError, "archive materialization"),
            ):
                checker._test_only_materialize_archive_transaction(
                    root, files, destination
                )
            self.assertFalse(destination.exists())

    def test_package_enumeration_rejects_parent_replaced_after_git_listing(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as name:
            base = Path(name)
            root = base / "repository"
            external = base / "external"
            root.mkdir()
            external.mkdir()
            payload = root / "payload"
            payload.mkdir()
            (payload / "data.txt").write_text("safe\n", encoding="utf-8")
            (external / "data.txt").write_text("private\n", encoding="utf-8")
            run_git(root, "init", "-q")
            repository = checker.repository_git.open_repository(root)
            self.assertIsNotNone(repository)
            assert repository is not None
            real_ls_files = repository.ls_files
            moved = root / "moved-payload"

            def replace_parent(_: object, paths: object) -> tuple[str, ...]:
                result = real_ls_files(paths)  # type: ignore[arg-type]
                payload.rename(moved)
                payload.symlink_to(external, target_is_directory=True)
                return result

            with (
                mock.patch.object(
                    checker.repository_git.RepositoryGit,
                    "ls_files",
                    autospec=True,
                    side_effect=replace_parent,
                ),
                self.assertRaisesRegex(ValueError, "parent changed|unsafe"),
            ):
                checker.package_files(root, ["payload"], repository=repository)

    def test_package_enumeration_rejects_nested_directory_mixed_snapshot(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            nested = root / "payload/nested"
            nested.mkdir(parents=True)
            (nested / "a.txt").write_text("first\n", encoding="utf-8")
            (nested / "b.txt").write_text("second\n", encoding="utf-8")
            run_git(root, "init", "-q")
            real_package_member = checker._package_member

            def replace_after_first(
                root_descriptor: int,
                rel: str,
                directories: dict[tuple[str, ...], tuple[int, int, int]],
            ) -> object:
                member = real_package_member(root_descriptor, rel, directories)
                if rel == "payload/nested/a.txt":
                    nested.rename(root / "payload/original-nested")
                    nested.mkdir()
                    (nested / "b.txt").write_text("replacement\n", encoding="utf-8")
                return member

            with (
                mock.patch.object(
                    checker, "_package_member", side_effect=replace_after_first
                ),
                self.assertRaisesRegex(ValueError, "parent changed"),
            ):
                checker.package_files(root, ["payload"])

    def test_package_validation_fails_closed_without_openat_support(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            (root / "payload").mkdir()
            with (
                mock.patch.object(checker.os, "supports_dir_fd", set()),
                self.assertRaisesRegex(RuntimeError, "descriptor-relative"),
            ):
                checker.package_files(root, ["payload"], repository=None)

    def test_archive_rejects_parent_directory_replaced_after_enumeration(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as name:
            base = Path(name)
            root = base / "repository"
            external = base / "external"
            root.mkdir()
            external.mkdir()
            payload = root / "payload"
            payload.mkdir()
            (payload / "data.txt").write_text("safe\n", encoding="utf-8")
            (external / "data.txt").write_text("private\n", encoding="utf-8")
            run_git(root, "init", "-q")
            files = checker.package_files(root, ["payload"])
            payload.rename(root / "moved-payload")
            payload.symlink_to(external, target_is_directory=True)
            destination = root / "source.tar.gz"
            with self.assertRaisesRegex(ValueError, "changed after enumeration"):
                checker._test_only_materialize_archive_transaction(
                    root, files, destination
                )
            self.assertFalse(destination.exists())

    def test_archive_rejects_regular_file_identity_change(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            (root / "payload").mkdir()
            target = root / "payload/data.txt"
            target.write_text("first\n", encoding="utf-8")
            subprocess.run(("git", "init", "-q"), cwd=root, check=True)
            files = checker.package_files(root, ["payload"])
            target.unlink()
            target.write_text("second\n", encoding="utf-8")
            destination = root / "source.tar.gz"
            with self.assertRaisesRegex(ValueError, "changed after enumeration"):
                checker.create_archive(root, files, destination)
            self.assertFalse(destination.exists())

    def test_archive_rejects_same_inode_content_change_with_restored_metadata(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            (root / "payload").mkdir()
            target = root / "payload/data.txt"
            target.write_text("safe\n", encoding="utf-8")
            subprocess.run(("git", "init", "-q"), cwd=root, check=True)
            files = checker.package_files(root, ["payload"])
            original = target.stat()
            target.write_text("evil\n", encoding="utf-8")
            os.utime(target, ns=(original.st_atime_ns, original.st_mtime_ns))
            self.assertEqual(files[0].inode, target.stat().st_ino)
            self.assertEqual(files[0].size, target.stat().st_size)
            self.assertEqual(files[0].mtime_ns, target.stat().st_mtime_ns)
            destination = root / "source.tar.gz"
            with self.assertRaisesRegex(
                ValueError, "content changed after enumeration"
            ):
                checker.create_archive(root, files, destination)
            self.assertFalse(destination.exists())

    def test_archive_rejects_content_change_during_read(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            (root / "payload").mkdir()
            target = root / "payload/data.bin"
            original_contents = b"a" * (2 * 1024 * 1024)
            target.write_bytes(original_contents)
            subprocess.run(("git", "init", "-q"), cwd=root, check=True)
            files = checker.package_files(root, ["payload"])
            original_metadata = target.stat()
            real_open = checker._open_package_member

            class MutatingStream:
                def __init__(self, stream: object) -> None:
                    self.stream = stream
                    self.changed = False

                def __enter__(self) -> "MutatingStream":
                    return self

                def __exit__(self, *_: object) -> None:
                    self.stream.close()  # type: ignore[attr-defined]

                def fileno(self) -> int:
                    return self.stream.fileno()  # type: ignore[attr-defined,no-any-return]

                def read(self, size: int = -1) -> bytes:
                    data = self.stream.read(size)  # type: ignore[attr-defined]
                    if data and not self.changed:
                        self.changed = True
                        target.write_bytes(b"b" * len(original_contents))
                        os.utime(
                            target,
                            ns=(
                                original_metadata.st_atime_ns,
                                original_metadata.st_mtime_ns,
                            ),
                        )
                    return data

            def mutating_open(
                root_path: Path, member: object
            ) -> tuple[object, os.stat_result]:
                stream, metadata = real_open(root_path, member)
                return MutatingStream(stream), metadata

            destination = root / "source.tar.gz"
            with mock.patch.object(
                checker, "_open_package_member", side_effect=mutating_open
            ):
                with self.assertRaisesRegex(
                    ValueError, "content changed after enumeration"
                ):
                    checker.create_archive(root, files, destination)
            self.assertFalse(destination.exists())

    def test_git_addition_between_pre_and_post_snapshots_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            payload = root / "payload"
            payload.mkdir()
            (payload / "first.txt").write_text("first\n", encoding="utf-8")
            run_git(root, "init", "-q")
            repository = checker.repository_git.open_repository(root)
            self.assertIsNotNone(repository)
            assert repository is not None
            real_snapshot = repository.snapshot_file_set
            observations = 0

            def add_before_post(
                _: object, paths: object
            ) -> checker.repository_git.RepositoryFileSet:
                nonlocal observations
                observations += 1
                if observations == 2:
                    (payload / "late.txt").write_text("late\n", encoding="utf-8")
                return real_snapshot(paths)  # type: ignore[arg-type]

            with (
                mock.patch.object(
                    checker.repository_git.RepositoryGit,
                    "snapshot_file_set",
                    autospec=True,
                    side_effect=add_before_post,
                ),
                self.assertRaisesRegex(ValueError, "Git file set changed"),
            ):
                checker.package_files(root, ["payload"], repository=repository)

    def test_archive_rejects_untracked_and_newly_staged_additions(self) -> None:
        for staged in (False, True):
            with self.subTest(staged=staged), tempfile.TemporaryDirectory() as name:
                root = Path(name)
                payload = root / "payload"
                payload.mkdir()
                (payload / "first.txt").write_text("first\n", encoding="utf-8")
                run_git(root, "init", "-q")
                snapshot = checker.package_files(root, ["payload"])
                (payload / "late.txt").write_text("late\n", encoding="utf-8")
                if staged:
                    run_git(root, "add", "payload/late.txt")
                destination = root / "source.tar.gz"
                with self.assertRaisesRegex(ValueError, "Git file set changed"):
                    checker._test_only_materialize_archive_transaction(
                        root, snapshot, destination
                    )
                self.assertFalse(destination.exists())

    def test_archive_rejects_disappearance_restore_and_content_replacement(
        self,
    ) -> None:
        for mutation in ("disappear", "restore", "replace-content"):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as name:
                root = Path(name)
                payload = root / "payload"
                payload.mkdir()
                target = payload / "data.txt"
                target.write_text("safe\n", encoding="utf-8")
                run_git(root, "init", "-q")
                snapshot = checker.package_files(root, ["payload"])
                original = target.stat()
                if mutation == "disappear":
                    target.unlink()
                elif mutation == "restore":
                    target.rename(root / "original.txt")
                    target.write_text("safe\n", encoding="utf-8")
                else:
                    target.write_text("evil\n", encoding="utf-8")
                    os.utime(target, ns=(original.st_atime_ns, original.st_mtime_ns))
                destination = root / "source.tar.gz"
                with self.assertRaisesRegex(ValueError, "changed after enumeration"):
                    checker._test_only_materialize_archive_transaction(
                        root, snapshot, destination
                    )
                self.assertFalse(destination.exists())

    def test_archive_mode_rejects_directory_swap_to_external_symlink(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            base = Path(name)
            root = base / "archive"
            external = base / "external"
            payload = root / "payload"
            payload.mkdir(parents=True)
            external.mkdir()
            (payload / "data.txt").write_text("safe\n", encoding="utf-8")
            (external / "data.txt").write_text("private\n", encoding="utf-8")
            snapshot = checker.package_files(root, ["payload"], repository=None)
            payload.rename(root / "original-payload")
            try:
                payload.symlink_to(external, target_is_directory=True)
            except OSError as exc:
                self.skipTest(f"symlinks are unavailable: {exc}")
            destination = root / "source.tar.gz"
            with self.assertRaisesRegex(ValueError, "changed after enumeration"):
                checker._test_only_materialize_archive_transaction(
                    root, snapshot, destination
                )
            self.assertFalse(destination.exists())

    def test_archive_mode_rejects_late_file_in_walked_directory(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            payload = root / "payload"
            payload.mkdir()
            (payload / "first.txt").write_text("first\n", encoding="utf-8")
            snapshot = checker.package_files(root, ["payload"], repository=None)
            (payload / "late.txt").write_text("late\n", encoding="utf-8")
            destination = root / "source.tar.gz"
            with self.assertRaisesRegex(ValueError, "directory entries changed"):
                checker._test_only_materialize_archive_transaction(
                    root, snapshot, destination
                )
            self.assertFalse(destination.exists())

    def test_package_snapshot_is_immutable_and_sequence_compatible(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            payload = root / "payload"
            payload.mkdir()
            (payload / "data.txt").write_text("data\n", encoding="utf-8")
            snapshot = checker.package_files(root, ["payload"], repository=None)
            self.assertIsInstance(snapshot.members, tuple)
            self.assertEqual(["payload/data.txt"], [item.path for item in snapshot])
            with self.assertRaises(dataclasses.FrozenInstanceError):
                snapshot.paths = ("other",)

    def test_shared_snapshot_preserves_raw_symlink_target_without_following(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            payload = root / "payload"
            payload.mkdir()
            external = root / "external.txt"
            external.write_text("private\n", encoding="utf-8")
            link = payload / "link.txt"
            try:
                link.symlink_to("../external.txt")
            except OSError as exc:
                self.skipTest(f"symlinks are unavailable: {exc}")
            with checker.repository_snapshot.SnapshotSession(root) as session:
                walked = session.walk_archive(["payload"])
                snapshot = session.seal()
            symlink = next(node for node in walked if node.path == "payload/link.txt")
            self.assertEqual("symlink", symlink.kind)
            self.assertEqual(b"../external.txt", symlink.raw_symlink_target)
            self.assertNotIn("external.txt", [node.path for node in snapshot.nodes])

    def test_archive_package_admits_selected_root_before_content_read(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            (root / "payload.txt").write_bytes(b"payload")
            limits = checker.repository_snapshot.DirectoryStructureLimits(
                max_directories=1,
                max_entries=0,
                max_depth=1,
                max_total_name_bytes=64,
                max_total_structure_bytes=256,
            )
            reads: list[int] = []
            real_read = checker.repository_snapshot._read_regular_file

            def record_read(
                stream: object, *, capture_bytes: bool, frozen_size: int
            ) -> tuple[str, bytes | None]:
                reads.append(frozen_size)
                return real_read(
                    stream,
                    capture_bytes=capture_bytes,
                    frozen_size=frozen_size,
                )

            with (
                mock.patch.object(
                    checker.repository_snapshot,
                    "DEFAULT_DIRECTORY_STRUCTURE_LIMITS",
                    limits,
                ),
                mock.patch.object(
                    checker.repository_snapshot,
                    "_read_regular_file",
                    side_effect=record_read,
                ),
                self.assertRaisesRegex(ValueError, "max_entries"),
            ):
                checker.package_files(root, ["payload.txt"], repository=None)
            self.assertEqual([], reads)

    def test_archive_root_snapshot_rejects_root_and_nested_entry_changes(self) -> None:
        for location, mutation in (
            ("root", "add"),
            ("nested", "add"),
            ("root", "delete"),
            ("nested", "delete"),
        ):
            with (
                self.subTest(location=location, mutation=mutation),
                tempfile.TemporaryDirectory() as name,
            ):
                root = Path(name)
                payload = root / "payload"
                payload.mkdir()
                target = (
                    root / "root.txt" if location == "root" else payload / "nested.txt"
                )
                target.write_text("present\n", encoding="utf-8")
                with checker.repository_snapshot.SnapshotSession(root) as session:
                    session.walk_archive_root(())
                    snapshot = session.seal()
                if mutation == "add":
                    parent = root if location == "root" else payload
                    (parent / "late.txt").write_text("late\n", encoding="utf-8")
                else:
                    target.unlink()
                with checker.repository_snapshot.SnapshotSession(root) as session:
                    with self.assertRaisesRegex(
                        ValueError, "directory entries changed|member changed"
                    ):
                        session.verify(snapshot)

    def test_archive_root_snapshot_ignores_nested_excluded_component_churn(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            payload = root / "payload"
            excluded = payload / "private"
            excluded.mkdir(parents=True)
            (payload / "included.txt").write_text("included\n", encoding="utf-8")
            private_file = excluded / "secret.txt"
            private_file.write_text("secret\n", encoding="utf-8")
            with checker.repository_snapshot.SnapshotSession(root) as session:
                walked = session.walk_archive_root({"private"})
                snapshot = session.seal()
            self.assertNotIn("payload/private", [node.path for node in walked])
            payload_directory = next(
                directory
                for directory in snapshot.directories
                if directory.path == "payload"
            )
            self.assertEqual(frozenset({"private"}), payload_directory.excluded_parts)

            private_file.unlink()
            excluded.rmdir()
            excluded.mkdir()
            (excluded / "replacement.txt").write_text("replacement\n", encoding="utf-8")
            with checker.repository_snapshot.SnapshotSession(root) as session:
                session.verify(snapshot)

    def test_archive_root_snapshot_rejects_top_level_directory_symlink_swap(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as name:
            base = Path(name)
            root = base / "repository"
            external = base / "external"
            payload = root / "payload"
            payload.mkdir(parents=True)
            external.mkdir()
            (payload / "data.txt").write_text("safe\n", encoding="utf-8")
            (external / "data.txt").write_text("private\n", encoding="utf-8")
            with checker.repository_snapshot.SnapshotSession(root) as session:
                session.walk_archive_root(())
                snapshot = session.seal()
            payload.rename(base / "original-payload")
            try:
                payload.symlink_to(external, target_is_directory=True)
            except OSError as exc:
                self.skipTest(f"symlinks are unavailable: {exc}")
            with checker.repository_snapshot.SnapshotSession(root) as session:
                with self.assertRaisesRegex(ValueError, "directory changed"):
                    session.verify(snapshot)

    @unittest.skipIf(os.name == "nt", "byte filenames are POSIX-specific")
    def test_archive_mode_rejects_non_utf8_directory_entry(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            payload = root / "payload"
            payload.mkdir()
            try:
                descriptor = os.open(
                    os.fsencode(payload) + b"/\xff", os.O_CREAT | os.O_WRONLY
                )
            except OSError as exc:
                self.skipTest(f"non-UTF-8 filenames are unavailable: {exc}")
            os.close(descriptor)
            with self.assertRaisesRegex(ValueError, "valid UTF-8"):
                checker.package_files(root, ["payload"], repository=None)

    def test_real_repository_archive_has_no_ignored_or_private_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            archive_path = Path(name) / "source.tar.gz"
            paths = checker.package_paths(ROOT / "build.zig.zon")
            self.assertIn("bench/tools", paths)
            self.assertIn(".github/workflows", paths)
            self.assertNotIn(".local-docs", paths)
            files = checker.package_files(ROOT, paths)
            checker.create_archive(ROOT, files, archive_path)
            with tarfile.open(archive_path, "r:gz") as archive:
                members = archive.getmembers()
            names = [member.name for member in members]
            self.assertFalse(any(".DS_Store" in name for name in names))
            self.assertFalse(
                any("__pycache__" in name or name.endswith(".pyc") for name in names)
            )
            self.assertTrue(
                all(member.uid == 0 and member.gid == 0 for member in members)
            )
            self.assertTrue(
                all(member.uname == "" and member.gname == "" for member in members)
            )
            self.assertTrue(all(member.mtime == 0 for member in members))
            for helper in (
                "benchmark_metadata.py",
                "report_comparison.py",
                "report_schedule.py",
            ):
                self.assertIn(f"bench/tools/{helper}", names)
            self.assertFalse(
                any(
                    name == ".local-docs" or name.startswith(".local-docs/")
                    for name in names
                )
            )

            extracted = Path(name) / "extracted"
            with tarfile.open(archive_path, "r:gz") as archive:
                archive.extractall(extracted, filter="data")
            extracted_paths = checker.package_paths(extracted / "build.zig.zon")
            self.assertEqual(len(paths), len(extracted_paths))
            extracted_files = checker.package_files(extracted, extracted_paths)
            self.assertEqual(names, [member.path for member in extracted_files])
            expected_output = (
                f"checked {len(paths)} package paths and {len(names)} files "
                "(archive mode; repository omissions not checked)\n"
            )

            unrelated = Path(name) / "unrelated-worktree"
            unrelated.mkdir()
            (unrelated / ".gitignore").write_text("*\n", encoding="utf-8")
            nested = unrelated / "nested-source-archive"
            with tarfile.open(archive_path, "r:gz") as archive:
                archive.extractall(nested, filter="data")

            import_smoke_entries = (
                "check_rank_k_report.py",
                "check_symm_report.py",
                "check_triangular_matrix_report.py",
                "run_gemm_sweep_isolated.py",
                "run_level1_report.py",
                "run_level2_report.py",
                "run_rank_k_report.py",
                "run_rotg_latency_report.py",
                "run_symm_report.py",
                "run_triangular_matrix_report.py",
            )

            def checker_command(root: Path) -> tuple[str, ...]:
                return (
                    sys.executable,
                    str(root / "tools/check_package_paths.py"),
                    "--root",
                    str(root),
                )

            process_cases = [
                (
                    "initialize unrelated worktree",
                    ("git", "init", "-q"),
                    unrelated,
                    None,
                ),
                (
                    "check extracted archive",
                    checker_command(extracted),
                    extracted,
                    expected_output,
                ),
                (
                    "check nested archive",
                    checker_command(nested),
                    nested,
                    expected_output,
                ),
                *(
                    (
                        f"import smoke {entry}",
                        (sys.executable, f"bench/tools/{entry}", "--help"),
                        extracted,
                        None,
                    )
                    for entry in import_smoke_entries
                ),
            ]
            process_environment = {
                **os.environ,
                "PYTHONDONTWRITEBYTECODE": "1",
            }
            process_environment = {
                name: value
                for name, value in process_environment.items()
                if not name.upper().startswith("GIT_")
            }
            for label, command, process_root, expected_stdout in process_cases:
                with self.subTest(process=label):
                    result = subprocess.run(
                        command,
                        cwd=process_root,
                        env=process_environment,
                        stdin=subprocess.DEVNULL,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        text=True,
                        check=False,
                    )
                    self.assertEqual(0, result.returncode, result.stderr)
                    if expected_stdout is not None:
                        self.assertEqual(expected_stdout, result.stdout)

            extracted_checker_path = extracted / "tools/check_build_inventory.py"
            extracted_spec = importlib.util.spec_from_file_location(
                "extracted_build_inventory_checker", extracted_checker_path
            )
            self.assertIsNotNone(extracted_spec)
            self.assertIsNotNone(extracted_spec.loader if extracted_spec else None)
            extracted_checker = importlib.util.module_from_spec(extracted_spec)
            assert extracted_spec and extracted_spec.loader
            extracted_spec.loader.exec_module(extracted_checker)
            self.assertEqual(
                [],
                extracted_checker.validate(
                    extracted, extracted / "tools/build_inventory.json"
                ),
            )
            with mock.patch.object(
                sys,
                "argv",
                ["check_package_paths.py", "--root", str(ROOT)],
            ):
                self.assertEqual(0, checker.main())

    def test_distribution_license_documents_are_complete_and_archivable(self) -> None:
        license_names = ("LICENSE", "COPYING", "COPYING.LESSER")
        declared_paths = checker.package_paths(ROOT / "build.zig.zon")
        for name in license_names:
            self.assertIn(name, declared_paths)

        self.assertEqual(
            (ROOT / "LICENSE").read_bytes(),
            (ROOT / "COPYING.LESSER").read_bytes(),
        )
        with tempfile.TemporaryDirectory() as name:
            archive_path = Path(name) / "licenses.tar.gz"
            files = checker.package_files(ROOT, license_names, repository=None)
            checker.create_archive(ROOT, files, archive_path)
            with tarfile.open(archive_path, "r:gz") as archive:
                self.assertEqual(sorted(license_names), archive.getnames())
                for license_name in license_names:
                    extracted = archive.extractfile(license_name)
                    self.assertIsNotNone(extracted)
                    assert extracted is not None
                    self.assertEqual(
                        (ROOT / license_name).read_bytes(),
                        extracted.read(),
                    )


if __name__ == "__main__":
    unittest.main()
