#!/usr/bin/env python3
# Copyright (C) 2026 Zynum contributors
# SPDX-License-Identifier: LGPL-3.0-or-later

"""Hermetic, exact-root Git access shared by repository tooling."""

from __future__ import annotations

import hashlib
import os
import re
import shutil
import stat
import subprocess
import sys
import threading
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Protocol


class RepositoryGitError(RuntimeError):
    """Git policy or execution could not prove the requested repository fact."""


class RepositoryGitUnavailable(RepositoryGitError):
    """A repository marker exists, but no usable Git executable is available."""


class RepositoryGitCommandError(RepositoryGitError):
    """A hermetic Git command failed or produced an invalid response."""

    def __init__(self, operation: str, returncode: int | None = None) -> None:
        super().__init__(f"Git {operation} failed")
        self.operation = operation
        self.returncode = returncode


class RepositoryGitSnapshotError(RepositoryGitError, ValueError):
    """Consecutive Git observations did not describe one stable state."""


_INHERITED_ENVIRONMENT = frozenset(
    {
        "PATH",
        "TMPDIR",
        "TMP",
        "TEMP",
        # Windows needs these to create a child process and resolve executables.
        "SYSTEMROOT",
        "WINDIR",
        "COMSPEC",
        "PATHEXT",
    }
)
_FIXED_ENVIRONMENT = {
    "LANG": "C",
    "LANGUAGE": "C",
    "LC_ALL": "C",
    "GIT_CONFIG_NOSYSTEM": "1",
    "GIT_CONFIG_GLOBAL": os.devnull,
    "GIT_CONFIG_SYSTEM": os.devnull,
    "GIT_OPTIONAL_LOCKS": "0",
    "GIT_TERMINAL_PROMPT": "0",
}

MAX_GIT_MARKER_BYTES = 64 * 1024
MAX_GIT_SMALL_OUTPUT_BYTES = 1024 * 1024
MAX_GIT_PATH_OUTPUT_BYTES = 256 * 1024 * 1024
MAX_GIT_PATHS = 1_000_000
MAX_GIT_STDERR_BYTES = 1024 * 1024


class _PathFactory(Protocol):
    def __call__(self, value: str) -> Path: ...


def strict_root(root: Path) -> Path:
    """Return a strictly resolved, non-symlink directory root."""
    try:
        supplied_metadata = root.lstat()
        canonical = root.resolve(strict=True)
        metadata = canonical.lstat()
    except OSError as exc:
        raise RepositoryGitError(
            "unable to resolve the repository root strictly"
        ) from exc
    if (
        stat.S_ISLNK(supplied_metadata.st_mode)
        or stat.S_ISLNK(metadata.st_mode)
        or not stat.S_ISDIR(metadata.st_mode)
    ):
        raise RepositoryGitError("repository root must be a non-symlink directory")
    return canonical


@dataclass(frozen=True, slots=True)
class RepositoryGitMarkerIdentity:
    kind: str
    device: int
    inode: int
    mode: int
    size: int
    mtime_ns: int
    ctime_ns: int
    sha256: str | None


def _marker_identity(
    metadata: os.stat_result, *, kind: str, sha256: str | None
) -> RepositoryGitMarkerIdentity:
    directory = kind == "directory"
    return RepositoryGitMarkerIdentity(
        kind=kind,
        device=metadata.st_dev,
        inode=metadata.st_ino,
        mode=stat.S_IFMT(metadata.st_mode) if directory else metadata.st_mode,
        size=0 if directory else metadata.st_size,
        mtime_ns=0 if directory else metadata.st_mtime_ns,
        ctime_ns=0 if directory else metadata.st_ctime_ns,
        sha256=sha256,
    )


def _capture_git_marker(root: Path) -> RepositoryGitMarkerIdentity | None:
    marker = root / ".git"
    try:
        metadata = marker.lstat()
    except FileNotFoundError:
        return None
    except OSError as exc:
        raise RepositoryGitError("unable to inspect the repository Git marker") from exc
    if stat.S_ISLNK(metadata.st_mode) or not (
        stat.S_ISDIR(metadata.st_mode) or stat.S_ISREG(metadata.st_mode)
    ):
        raise RepositoryGitError("repository Git marker has an unsafe filesystem type")
    if stat.S_ISDIR(metadata.st_mode):
        return _marker_identity(metadata, kind="directory", sha256=None)
    if metadata.st_size > MAX_GIT_MARKER_BYTES:
        raise RepositoryGitError("repository Git marker exceeds its size limit")

    flags = os.O_RDONLY
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = -1
    try:
        descriptor = os.open(marker, flags)
        opened = os.fstat(descriptor)
        initial = _marker_identity(metadata, kind="file", sha256=None)
        opened_identity = _marker_identity(opened, kind="file", sha256=None)
        if initial != opened_identity or not stat.S_ISREG(opened.st_mode):
            raise RepositoryGitSnapshotError(
                "repository Git marker changed while being opened"
            )
        contents = bytearray()
        remaining = opened.st_size
        while remaining:
            chunk = os.read(descriptor, min(64 * 1024, remaining))
            if not chunk:
                raise RepositoryGitSnapshotError(
                    "repository Git marker reached early EOF"
                )
            contents.extend(chunk)
            remaining -= len(chunk)
        if os.read(descriptor, 1):
            raise RepositoryGitSnapshotError("repository Git marker grew while read")
        final = os.fstat(descriptor)
        if _marker_identity(final, kind="file", sha256=None) != initial:
            raise RepositoryGitSnapshotError(
                "repository Git marker changed while being read"
            )
    except RepositoryGitError:
        raise
    except OSError as exc:
        raise RepositoryGitError("unable to inspect the repository Git marker") from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    return _marker_identity(
        metadata,
        kind="file",
        sha256=hashlib.sha256(contents).hexdigest(),
    )


def _git_marker(root: Path) -> bool:
    return _capture_git_marker(root) is not None


def _verify_git_marker(
    root: Path, expected: RepositoryGitMarkerIdentity | None
) -> None:
    if expected is None:
        return
    if _capture_git_marker(root) != expected:
        raise RepositoryGitSnapshotError(
            "repository Git marker changed during observation"
        )


class _BoundedPipeCapture:
    def __init__(self, limit: int) -> None:
        self.limit = limit
        self.chunks: list[bytes] = []
        self.overflow = False
        self.error: OSError | None = None

    def read(self, descriptor: int) -> None:
        total = 0
        try:
            while True:
                chunk = os.read(
                    descriptor,
                    min(64 * 1024, self.limit + 1 - total),
                )
                if not chunk:
                    return
                self.chunks.append(chunk)
                total += len(chunk)
                if total > self.limit:
                    self.overflow = True
                    return
        except OSError as exc:
            self.error = exc
        finally:
            os.close(descriptor)

    def contents(self) -> bytes:
        return b"".join(self.chunks)


def _ambient_git_names(environment: Mapping[str, str]) -> tuple[str, ...]:
    return tuple(
        sorted(name for name in environment if name.upper().startswith("GIT_"))
    )


def _child_environment(environment: Mapping[str, str]) -> dict[str, str]:
    child = {
        name: value
        for name, value in environment.items()
        if name.upper() in _INHERITED_ENVIRONMENT
    }
    child.setdefault("PATH", os.defpath)
    child.update(_FIXED_ENVIRONMENT)
    return child


def _git_executable(environment: Mapping[str, str]) -> Path:
    discovered = shutil.which("git", path=environment.get("PATH", os.defpath))
    if discovered is None:
        raise RepositoryGitUnavailable("Git is unavailable for a root containing .git")
    try:
        executable = Path(discovered).resolve(strict=True)
        metadata = executable.stat()
    except OSError as exc:
        raise RepositoryGitUnavailable(
            "Git is unavailable for a root containing .git"
        ) from exc
    if not stat.S_ISREG(metadata.st_mode) or not os.access(executable, os.X_OK):
        raise RepositoryGitUnavailable("Git is unavailable for a root containing .git")
    return executable


def _decode_top_level(stdout: bytes) -> str:
    """Decode exactly one absolute path line from ``git rev-parse``."""
    if (
        not stdout
        or not stdout.endswith(b"\n")
        or stdout.count(b"\n") != 1
        or b"\r" in stdout
        or b"\0" in stdout
    ):
        raise RepositoryGitCommandError("top-level verification framing")
    try:
        decoded = stdout[:-1].decode(sys.getfilesystemencoding(), errors="strict")
    except UnicodeDecodeError as exc:
        raise RepositoryGitCommandError("top-level verification encoding") from exc
    if not decoded:
        raise RepositoryGitCommandError("top-level verification framing")
    return decoded


def _canonical_top_level(stdout: bytes, *, path_factory: _PathFactory = Path) -> Path:
    """Resolve Git's native path spelling to a canonical directory.

    ``Path`` deliberately supplies the host platform's path rules here. On
    Windows that accepts both Git's forward slashes and native backslashes and
    resolves case using the filesystem rather than a POSIX lexical comparison.
    """
    candidate = path_factory(_decode_top_level(stdout))
    if not candidate.is_absolute():
        raise RepositoryGitCommandError("top-level verification path")
    try:
        resolved = candidate.resolve(strict=True)
        return strict_root(resolved)
    except (OSError, RepositoryGitError) as exc:
        raise RepositoryGitCommandError("top-level verification path") from exc


def _same_directory(left: Path, right: Path) -> bool:
    """Compare canonical directories with the host's identity semantics."""
    try:
        return os.path.samefile(left, right)
    except OSError as exc:
        raise RepositoryGitCommandError("top-level verification identity") from exc


def _decode_path_list(
    stdout: bytes,
    operation: str,
    *,
    max_bytes: int = MAX_GIT_PATH_OUTPUT_BYTES,
    max_paths: int = MAX_GIT_PATHS,
) -> tuple[str, ...]:
    if len(stdout) > max_bytes:
        raise RepositoryGitCommandError(f"{operation} output limit")
    if stdout and not stdout.endswith(b"\0"):
        raise RepositoryGitCommandError(f"{operation} framing")
    if stdout.count(b"\0") > max_paths:
        raise RepositoryGitCommandError(f"{operation} path count limit")
    raw_paths = stdout.split(b"\0")[:-1] if stdout else ()
    if any(not raw for raw in raw_paths):
        raise RepositoryGitCommandError(f"{operation} framing")

    decoded: list[str] = []
    seen: set[str] = set()
    for raw in raw_paths:
        try:
            path = raw.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise RepositoryGitCommandError(f"{operation} encoding") from exc
        pure = PurePosixPath(path)
        if (
            pure.is_absolute()
            or not pure.parts
            or pure.as_posix() != path
            or any(part in {"", ".", ".."} for part in pure.parts)
        ):
            raise RepositoryGitCommandError(f"{operation} path safety")
        if path in seen:
            raise RepositoryGitCommandError(f"{operation} uniqueness")
        seen.add(path)
        decoded.append(path)
    return tuple(sorted(decoded))


@dataclass(frozen=True, slots=True)
class RepositoryFileSet:
    """One stable Git view of selected repository paths."""

    listed: tuple[str, ...]
    deleted: tuple[str, ...]
    present: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class RepositoryTreeEntry:
    """One regular-file subject from a committed Git tree."""

    path: str
    mode: int
    object_id: str


@dataclass(frozen=True, slots=True)
class RepositoryPublicationSnapshot:
    """One stable, clean publication subject bound to a HEAD commit tree."""

    revision: str
    entries: tuple[RepositoryTreeEntry, ...]


@dataclass(frozen=True, slots=True)
class RepositoryGitIdentity:
    """One immutable, coherently sampled Git identity."""

    revision: str | None
    branch: str | None
    detached: bool
    status_bytes: bytes
    status_lines: tuple[str, ...]
    status_sha256: str
    index_sha256: str | None


def _decode_one_line(stdout: bytes, operation: str, *, encoding: str) -> str:
    if (
        not stdout
        or not stdout.endswith(b"\n")
        or stdout.count(b"\n") != 1
        or b"\r" in stdout
        or b"\0" in stdout
    ):
        raise RepositoryGitCommandError(f"{operation} framing")
    try:
        value = stdout[:-1].decode(encoding, errors="strict")
    except UnicodeDecodeError as exc:
        raise RepositoryGitCommandError(f"{operation} encoding") from exc
    if not value:
        raise RepositoryGitCommandError(f"{operation} framing")
    return value


def _decode_revision(stdout: bytes) -> str:
    revision = _decode_one_line(stdout, "revision observation", encoding="ascii")
    if re.fullmatch(r"(?:[0-9a-f]{40}|[0-9a-f]{64})", revision) is None:
        raise RepositoryGitCommandError("revision observation framing")
    return revision


def _decode_status(stdout: bytes) -> tuple[str, ...]:
    if len(stdout) > MAX_GIT_PATH_OUTPUT_BYTES:
        raise RepositoryGitCommandError("status observation output limit")
    if b"\0" in stdout or b"\r" in stdout or (stdout and not stdout.endswith(b"\n")):
        raise RepositoryGitCommandError("status observation framing")
    try:
        text = stdout.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise RepositoryGitCommandError("status observation encoding") from exc
    lines = tuple(text.splitlines())
    if len(lines) > MAX_GIT_PATHS:
        raise RepositoryGitCommandError("status observation path count limit")
    if any(len(line) < 3 or line[2] != " " for line in lines):
        raise RepositoryGitCommandError("status observation framing")
    return lines


def _validate_index(stdout: bytes) -> None:
    if len(stdout) > MAX_GIT_PATH_OUTPUT_BYTES:
        raise RepositoryGitCommandError("index observation output limit")
    if stdout and not stdout.endswith(b"\0"):
        raise RepositoryGitCommandError("index observation framing")
    entries = stdout.split(b"\0")[:-1] if stdout else ()
    if len(entries) > MAX_GIT_PATHS:
        raise RepositoryGitCommandError("index observation path count limit")
    if any(not entry for entry in entries):
        raise RepositoryGitCommandError("index observation framing")
    for entry in entries:
        header, separator, raw_path = entry.partition(b"\t")
        fields = header.split(b" ")
        if (
            not separator
            or not raw_path
            or len(fields) != 3
            or re.fullmatch(rb"[0-7]{6}", fields[0]) is None
            or re.fullmatch(rb"(?:[0-9a-f]{40}|[0-9a-f]{64})", fields[1]) is None
            or re.fullmatch(rb"[0-3]", fields[2]) is None
        ):
            raise RepositoryGitCommandError("index observation framing")
        try:
            raw_path.decode("utf-8", errors="strict")
        except UnicodeDecodeError as exc:
            raise RepositoryGitCommandError("index observation encoding") from exc


def _decode_tree_entries(
    stdout: bytes, operation: str
) -> tuple[RepositoryTreeEntry, ...]:
    if len(stdout) > MAX_GIT_PATH_OUTPUT_BYTES:
        raise RepositoryGitCommandError(f"{operation} output limit")
    if stdout and not stdout.endswith(b"\0"):
        raise RepositoryGitCommandError(f"{operation} framing")
    records = stdout.split(b"\0")[:-1] if stdout else ()
    if len(records) > MAX_GIT_PATHS:
        raise RepositoryGitCommandError(f"{operation} path count limit")

    entries: list[RepositoryTreeEntry] = []
    seen: set[str] = set()
    for record in records:
        header, separator, raw_path = record.partition(b"\t")
        fields = header.split(b" ")
        if (
            not separator
            or not raw_path
            or len(fields) != 3
            or fields[1] != b"blob"
            or fields[0] not in {b"100644", b"100755"}
            or re.fullmatch(rb"(?:[0-9a-f]{40}|[0-9a-f]{64})", fields[2]) is None
        ):
            raise RepositoryGitCommandError(f"{operation} regular-file tree")
        decoded = _decode_path_list(raw_path + b"\0", operation)
        if len(decoded) != 1 or decoded[0] in seen:
            raise RepositoryGitCommandError(f"{operation} uniqueness")
        path = decoded[0]
        seen.add(path)
        entries.append(
            RepositoryTreeEntry(
                path=path,
                mode=int(fields[0], 8),
                object_id=fields[2].decode("ascii"),
            )
        )
    return tuple(sorted(entries, key=lambda entry: entry.path))


@dataclass(frozen=True)
class RepositoryGit:
    """A verified worktree and its one-time-resolved hermetic Git boundary."""

    root: Path
    executable: Path
    _environment: Mapping[str, str] = field(repr=False, compare=False)
    _marker: RepositoryGitMarkerIdentity | None = field(
        default=None, repr=False, compare=False
    )

    def _command(self, arguments: Sequence[str]) -> tuple[str, ...]:
        return (
            str(self.executable),
            "--no-pager",
            "--literal-pathspecs",
            "-C",
            str(self.root),
            f"--work-tree={self.root}",
            "-c",
            f"core.excludesFile={os.devnull}",
            "-c",
            "core.fsmonitor=false",
            "-c",
            "core.untrackedCache=false",
            *arguments,
        )

    def run(
        self,
        arguments: Sequence[str],
        *,
        operation: str,
        check: bool = True,
        allow_stderr_on_failure: bool = False,
        stdout_limit: int = MAX_GIT_SMALL_OUTPUT_BYTES,
    ) -> subprocess.CompletedProcess[bytes]:
        """Run Git with fixed arguments/environment and closed standard input."""
        if type(stdout_limit) is not int or stdout_limit < 0:
            raise TypeError("stdout_limit must be a non-negative integer")
        _verify_git_marker(self.root, self._marker)
        stdout_read = stdout_write = stderr_read = stderr_write = -1
        try:
            stdout_read, stdout_write = os.pipe()
            stderr_read, stderr_write = os.pipe()
            stdout_capture = _BoundedPipeCapture(stdout_limit)
            stderr_capture = _BoundedPipeCapture(MAX_GIT_STDERR_BYTES)
            stdout_thread = threading.Thread(
                target=stdout_capture.read,
                args=(stdout_read,),
                daemon=True,
            )
            stderr_thread = threading.Thread(
                target=stderr_capture.read,
                args=(stderr_read,),
                daemon=True,
            )
            stdout_thread.start()
            stderr_thread.start()
            stdout_read = stderr_read = -1
            result = subprocess.run(
                self._command(arguments),
                cwd=self.root,
                env=dict(self._environment),
                stdin=subprocess.DEVNULL,
                stdout=stdout_write,
                stderr=stderr_write,
                check=False,
            )
        except OSError as exc:
            raise RepositoryGitCommandError(operation) from exc
        finally:
            for descriptor in (stdout_write, stderr_write):
                if descriptor >= 0:
                    os.close(descriptor)
            if "stdout_thread" in locals():
                stdout_thread.join()
                stderr_thread.join()
            for descriptor in (stdout_read, stderr_read):
                if descriptor >= 0:
                    os.close(descriptor)
        if stdout_capture.error is not None or stderr_capture.error is not None:
            raise RepositoryGitCommandError(operation)
        if stdout_capture.overflow or stderr_capture.overflow:
            raise RepositoryGitCommandError(f"{operation} output limit")
        captured = subprocess.CompletedProcess(
            result.args,
            result.returncode,
            stdout_capture.contents(),
            stderr_capture.contents(),
        )
        _verify_git_marker(self.root, self._marker)
        if captured.stderr and not (
            allow_stderr_on_failure and not check and result.returncode != 0
        ):
            raise RepositoryGitCommandError(operation, result.returncode)
        if check and result.returncode != 0:
            raise RepositoryGitCommandError(operation, result.returncode)
        return captured

    def ls_files(self, paths: Sequence[str] = ()) -> tuple[str, ...]:
        """Return sorted, unique, safe UTF-8 repository-relative paths."""
        result = self.run(
            (
                "ls-files",
                "--full-name",
                "--cached",
                "--others",
                "--exclude-standard",
                "-z",
                "--",
                *paths,
            ),
            operation="file enumeration",
            stdout_limit=MAX_GIT_PATH_OUTPUT_BYTES,
        )
        return _decode_path_list(result.stdout, "file enumeration")

    def deleted_files(self, paths: Sequence[str] = ()) -> tuple[str, ...]:
        """Freeze tracked paths already absent before file enumeration."""
        result = self.run(
            (
                "diff-files",
                "--name-only",
                "--diff-filter=D",
                "--no-renames",
                "-z",
                "--",
                *paths,
            ),
            operation="deleted-file snapshot",
            stdout_limit=MAX_GIT_PATH_OUTPUT_BYTES,
        )
        return _decode_path_list(result.stdout, "deleted-file snapshot")

    def _observe_identity_round(self, *, include_index: bool) -> RepositoryGitIdentity:
        revision_result = self.run(
            ("rev-parse", "--verify", "--quiet", "HEAD"),
            operation="revision observation",
            check=False,
            allow_stderr_on_failure=True,
        )
        if revision_result.returncode == 0:
            if revision_result.stderr:
                raise RepositoryGitCommandError(
                    "revision observation", revision_result.returncode
                )
            revision = _decode_revision(revision_result.stdout)
        elif (
            revision_result.returncode == 1
            and not revision_result.stdout
            and not revision_result.stderr
        ):
            revision = None
        else:
            raise RepositoryGitCommandError(
                "revision observation", revision_result.returncode
            )

        branch_result = self.run(
            ("symbolic-ref", "--quiet", "--short", "HEAD"),
            operation="branch observation",
            check=False,
            allow_stderr_on_failure=True,
        )
        if branch_result.returncode == 0:
            if branch_result.stderr:
                raise RepositoryGitCommandError(
                    "branch observation", branch_result.returncode
                )
            branch = _decode_one_line(
                branch_result.stdout, "branch observation", encoding="utf-8"
            )
            detached = False
        elif (
            branch_result.returncode == 1
            and not branch_result.stdout
            and not branch_result.stderr
        ):
            branch = None
            detached = True
        else:
            raise RepositoryGitCommandError(
                "branch observation", branch_result.returncode
            )
        if revision is None and detached:
            raise RepositoryGitCommandError("HEAD observation coherence")

        status_result = self.run(
            ("status", "--porcelain=v1", "--untracked-files=all"),
            operation="status observation",
            stdout_limit=MAX_GIT_PATH_OUTPUT_BYTES,
        )
        status_lines = _decode_status(status_result.stdout)

        index_sha256 = None
        if include_index:
            index_result = self.run(
                ("ls-files", "--stage", "-z"),
                operation="index observation",
                stdout_limit=MAX_GIT_PATH_OUTPUT_BYTES,
            )
            _validate_index(index_result.stdout)
            index_sha256 = hashlib.sha256(index_result.stdout).hexdigest()

        return RepositoryGitIdentity(
            revision=revision,
            branch=branch,
            detached=detached,
            status_bytes=status_result.stdout,
            status_lines=status_lines,
            status_sha256=hashlib.sha256(status_result.stdout).hexdigest(),
            index_sha256=index_sha256,
        )

    def observe_identity(self, include_index: bool = False) -> RepositoryGitIdentity:
        """Return two identical complete consecutive Git identity observations."""
        if type(include_index) is not bool:
            raise TypeError("include_index must be a boolean")
        first = self._observe_identity_round(include_index=include_index)
        second = self._observe_identity_round(include_index=include_index)
        if first != second:
            raise RepositoryGitSnapshotError(
                "Git identity changed between consecutive observations"
            )
        return first

    def _publication_round(self, paths: Sequence[str]) -> RepositoryPublicationSnapshot:
        revision_result = self.run(
            ("rev-parse", "--verify", "HEAD^{commit}"),
            operation="publication revision observation",
            check=False,
            allow_stderr_on_failure=True,
        )
        if revision_result.returncode != 0:
            raise RepositoryGitSnapshotError(
                "source archive publication requires a committed HEAD"
            )
        if revision_result.stderr:
            raise RepositoryGitCommandError(
                "publication revision observation", revision_result.returncode
            )
        revision = _decode_revision(revision_result.stdout)

        status = self.run(
            (
                "status",
                "--porcelain=v1",
                "-z",
                "--untracked-files=all",
                "--",
                *paths,
            ),
            operation="publication status observation",
            stdout_limit=MAX_GIT_PATH_OUTPUT_BYTES,
        )
        if status.stdout:
            raise RepositoryGitSnapshotError(
                "source archive publication paths differ from HEAD: tracked, "
                "staged, or untracked non-ignored content is present"
            )

        tree = self.run(
            (
                "ls-tree",
                "-r",
                "-z",
                "--full-tree",
                revision,
                "--",
                *paths,
            ),
            operation="publication tree observation",
            stdout_limit=MAX_GIT_PATH_OUTPUT_BYTES,
        )
        return RepositoryPublicationSnapshot(
            revision=revision,
            entries=_decode_tree_entries(tree.stdout, "publication tree observation"),
        )

    def snapshot_publication(
        self, paths: Sequence[str]
    ) -> RepositoryPublicationSnapshot:
        """Bind clean selected paths to two identical HEAD tree observations.

        Unlike :meth:`snapshot_file_set`, this publication authority never
        admits untracked worktree files or index content that differs from
        ``HEAD``.
        """
        first = self._publication_round(paths)
        second = self._publication_round(paths)
        if first != second:
            raise RepositoryGitSnapshotError(
                "source archive publication state changed between observations"
            )
        return first

    def snapshot_file_set(self, paths: Sequence[str] = ()) -> RepositoryFileSet:
        """Return two identical consecutive listed/deleted observations.

        Sampling deletions before listings preserves a clear absence boundary:
        a deleted path restored during listing necessarily changes the second
        pair and cannot be accepted as an ordinary present member.
        """
        try:
            first_deleted = self.deleted_files(paths)
            first_listed = self.ls_files(paths)
            second_deleted = self.deleted_files(paths)
            second_listed = self.ls_files(paths)
        except RepositoryGitError:
            raise
        except OSError as exc:
            raise RepositoryGitSnapshotError(
                "Git file set changed during enumeration; repository parent "
                "changed or is unsafe; a tracked-deleted member may have "
                "appeared after snapshot"
            ) from exc
        if (
            first_listed != second_listed
            or first_deleted != second_deleted
            or not set(first_deleted).issubset(first_listed)
        ):
            raise RepositoryGitSnapshotError(
                "Git file set changed during enumeration; repository parent "
                "changed or is unsafe; a tracked-deleted member may have "
                "appeared after snapshot"
            )
        return RepositoryFileSet(
            listed=first_listed,
            deleted=first_deleted,
            present=tuple(
                path for path in first_listed if path not in set(first_deleted)
            ),
        )


def open_repository(
    root: Path, *, environ: Mapping[str, str] | None = None
) -> RepositoryGit | None:
    """Open an exact worktree, or return ``None`` for a marker-free archive.

    Marker-free roots never inspect Git-related environment and never resolve or
    execute Git. A root containing a safe marker rejects every ambient ``GIT_*``
    name before constructing a hermetic child environment.
    """
    canonical = strict_root(root)
    marker = _capture_git_marker(canonical)
    if marker is None:
        return None

    source_environment = os.environ if environ is None else environ
    redirected = _ambient_git_names(source_environment)
    if redirected:
        raise RepositoryGitError(
            "ambient Git environment prevents exact-root verification: "
            + ", ".join(redirected)
        )

    executable = _git_executable(source_environment)
    _verify_git_marker(canonical, marker)
    repository = RepositoryGit(
        canonical,
        executable,
        _child_environment(source_environment),
        _marker=marker,
    )
    result = repository.run(
        ("rev-parse", "--show-toplevel"),
        operation="top-level verification",
    )
    reported_root = _canonical_top_level(result.stdout)
    if not _same_directory(canonical, reported_root):
        raise RepositoryGitError(
            "supplied root is not the exact Git worktree top-level"
        )
    return repository
