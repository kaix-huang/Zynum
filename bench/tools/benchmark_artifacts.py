#!/usr/bin/env python3
# Copyright (C) 2026 Zynum contributors
# SPDX-License-Identifier: LGPL-3.0-or-later

"""Frozen benchmark executable and library artifacts.

The public path recorded in benchmark metadata is deliberately different from
the private path used for execution.  Direct file artifacts are copied once
from a held regular-file descriptor into a mode-0700 temporary directory; the
copy and its SHA-256 are produced by the same bounded read.  Platform loader
images are the only explicit non-file exception and consequently have no hash.
"""

from __future__ import annotations

import hashlib
import importlib.util
import os
import re
import stat
import sys
import tempfile
import threading
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal


_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
_REPOSITORY_SNAPSHOT_PATH = _REPOSITORY_ROOT / "tools" / "repository_snapshot.py"
_REPOSITORY_SNAPSHOT_SPEC = importlib.util.spec_from_file_location(
    "_zynum_benchmark_artifact_repository_snapshot",
    _REPOSITORY_SNAPSHOT_PATH,
)
if _REPOSITORY_SNAPSHOT_SPEC is None or _REPOSITORY_SNAPSHOT_SPEC.loader is None:
    raise RuntimeError("cannot load the repository snapshot cleanup owner")
repository_snapshot = importlib.util.module_from_spec(_REPOSITORY_SNAPSHOT_SPEC)
sys.modules[_REPOSITORY_SNAPSHOT_SPEC.name] = repository_snapshot
_REPOSITORY_SNAPSHOT_SPEC.loader.exec_module(repository_snapshot)


ArtifactRole = Literal["binary", "library"]
ArtifactSourceKind = Literal["file", "platform_image"]
CleanupStatus = Literal[
    "open",
    "complete",
    "recovery_required",
    "unaddressable",
]

DEFAULT_MAX_ARTIFACT_BYTES = 2 * 1024 * 1024 * 1024
DEFAULT_MAX_ARTIFACTS = 64
DEFAULT_MAX_TOTAL_BYTES = 4 * 1024 * 1024 * 1024
DEFAULT_MAX_SYMLINKS = 40
_COPY_CHUNK_BYTES = 1024 * 1024
_PRIVATE_MODES = {"binary": 0o500, "library": 0o400}
DEFAULT_ACCELERATE_IMAGE = "/System/Library/Frameworks/Accelerate.framework/Accelerate"
_FROZEN_SOURCE_RESOLVER: Any = None


def _set_frozen_source_resolver(resolver: Any) -> None:
    global _FROZEN_SOURCE_RESOLVER
    if resolver is not None and not callable(resolver):
        raise TypeError("frozen source resolver must be callable")
    _FROZEN_SOURCE_RESOLVER = resolver


def _resolved_interpreter_source(public_path: str) -> tuple[bytes, str] | None:
    resolver = _FROZEN_SOURCE_RESOLVER
    if resolver is None:
        return None
    resolved = resolver(public_path)
    if (
        type(resolved) is not tuple
        or len(resolved) != 3
        or not isinstance(resolved[0], str)
        or type(resolved[1]) is not bytes
        or not isinstance(resolved[2], str)
        or resolved[0] != os.path.realpath(public_path)
        or not re.fullmatch(r"[0-9a-f]{64}", resolved[2])
        or hashlib.sha256(resolved[1]).hexdigest() != resolved[2]
    ):
        raise ArtifactCaptureError(
            "frozen_source_resolution_invalid",
            "frozen interpreter source resolution is noncanonical",
            public_path=public_path,
        )
    return resolved[1], resolved[2]


class ArtifactSnapshotError(RuntimeError):
    """Structured fail-closed benchmark artifact error."""

    publication_status = "not_published"

    def __init__(
        self,
        code: str,
        message: str,
        *,
        public_path: str | None = None,
        artifact_id: str | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.public_path = public_path
        self.artifact_id = artifact_id


class ArtifactCaptureError(ArtifactSnapshotError):
    """A requested artifact could not be frozen safely."""


class ArtifactVerificationError(ArtifactSnapshotError):
    """A private frozen artifact no longer matches its capture."""


@dataclass(frozen=True, slots=True)
class CleanupIssue:
    """A cleanup failure and any exact pathname retained for recovery."""

    code: str
    artifact_id: str | None
    recovery_path: str | None = None


class ArtifactCleanupError(ArtifactSnapshotError):
    """One or more owned private artifacts could not be proved removed."""

    def __init__(
        self,
        issues: tuple[CleanupIssue, ...],
        *,
        candidate_paths: tuple[str, ...] = (),
        cleanup_status: CleanupStatus = "recovery_required",
    ) -> None:
        super().__init__(
            "artifact_cleanup_incomplete",
            "private benchmark artifact cleanup was incomplete; no publication is claimed",
        )
        if cleanup_status == "unaddressable":
            issues = tuple(
                CleanupIssue(issue.code, issue.artifact_id) for issue in issues
            )
        self.issues = issues
        self.candidate_paths = candidate_paths
        self.cleanup_status = cleanup_status
        self.recovery_paths = tuple(
            dict.fromkeys(
                issue.recovery_path
                for issue in issues
                if issue.recovery_path is not None
            )
        )
        self.cleanup_complete = False


@dataclass(frozen=True, slots=True)
class ArtifactRequest:
    """One public benchmark artifact request.

    ``source_kind='file'`` is the default and requires an explicit filesystem
    path.  A loader-resolved image (for example a platform framework image) must
    opt in through :meth:`platform_image`; it is recorded with ``sha256=None``.
    """

    name: str
    path: str | os.PathLike[str]
    role: ArtifactRole
    source_kind: ArtifactSourceKind = "file"
    require_source_executable: bool = True

    @classmethod
    def binary(cls, name: str, path: str | os.PathLike[str]) -> ArtifactRequest:
        return cls(name=name, path=path, role="binary")

    @classmethod
    def library(cls, name: str, path: str | os.PathLike[str]) -> ArtifactRequest:
        return cls(
            name=name,
            path=path,
            role="library",
            require_source_executable=False,
        )

    @classmethod
    def interpreter_script(
        cls, name: str, path: str | os.PathLike[str]
    ) -> ArtifactRequest:
        """Capture a script executed through an already-selected interpreter."""

        return cls(
            name=name,
            path=path,
            role="binary",
            require_source_executable=False,
        )

    @classmethod
    def platform_image(cls, name: str, image: str) -> ArtifactRequest:
        return cls(
            name=name,
            path=image,
            role="library",
            source_kind="platform_image",
            require_source_executable=False,
        )


@dataclass(frozen=True, slots=True, repr=False)
class FrozenArtifact:
    """Immutable public projection backed by an owned snapshot set."""

    name: str
    path: str
    role: ArtifactRole
    sha256: str | None
    source_kind: ArtifactSourceKind
    size: int | None
    _owner: ArtifactSnapshotSet
    _copy_key: tuple[str, ArtifactRole] | None

    def __repr__(self) -> str:
        return (
            "FrozenArtifact(name={!r}, path={!r}, role={!r}, sha256={!r}, "
            "source_kind={!r}, size={!r})"
        ).format(
            self.name,
            self.path,
            self.role,
            self.sha256,
            self.source_kind,
            self.size,
        )

    @property
    def execution_path(self) -> str:
        """Return a freshly verified execution path.

        File-backed artifacts return only the private frozen path.  An explicit
        platform image returns its public loader image name/path.
        """

        return self._owner.execution_path(self)

    def lookup_execution_path(self) -> str:
        """Method spelling of :attr:`execution_path` for controller adapters."""

        return self.execution_path

    def metadata_record(self) -> dict[str, str | None]:
        """Return the stable public ``name/path/sha256`` metadata projection."""

        return self._owner.metadata_record(self)

    def legacy_record(self) -> dict[str, str | None]:
        """Return the same record used by legacy family metadata."""

        return self.metadata_record()


@dataclass(slots=True)
class _PrivateCopy:
    artifact_id: str
    role: ArtifactRole
    public_path: str
    leaf: str
    descriptor: int
    identity: tuple[int, int]
    source_identity: tuple[int, int]
    size: int
    mode: int
    sha256: str


@dataclass(slots=True)
class _PendingPrivateCopy:
    artifact_id: str
    public_path: str
    leaf: str
    descriptor: int
    identity: tuple[int, int] | None


@dataclass(frozen=True, slots=True)
class _CleanupTarget:
    artifact_id: str
    leaf: str
    descriptor: int
    identity: tuple[int, int]
    size: int
    mode: int
    sha256: str


@dataclass(frozen=True, slots=True)
class _OpenedSource:
    descriptor: int
    metadata: os.stat_result
    path_fingerprint: tuple[tuple[Any, ...], ...]


def _stat_identity(metadata: os.stat_result) -> tuple[int, int]:
    return metadata.st_dev, metadata.st_ino


def _source_fingerprint(metadata: os.stat_result) -> tuple[int, ...]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_mode,
        metadata.st_nlink,
        metadata.st_uid,
        metadata.st_gid,
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
        getattr(metadata, "st_flags", 0),
    )


def _normalize_components(
    base: list[str], target: str, remaining: list[str]
) -> list[str]:
    if os.path.isabs(target):
        combined: list[str] = []
    else:
        combined = list(base)
    combined.extend(target.split(os.sep))
    combined.extend(remaining)
    normalized: list[str] = []
    for component in combined:
        if not component or component == ".":
            continue
        if component == "..":
            if normalized:
                normalized.pop()
            continue
        normalized.append(component)
    return normalized


def _path_has_directory(path: str) -> bool:
    separators = (os.sep,) if os.altsep is None else (os.sep, os.altsep)
    return os.path.isabs(path) or any(separator in path for separator in separators)


def _open_resolved_regular(path: str, max_symlinks: int) -> _OpenedSource:
    """Open ``path`` without following a component behind our descriptor walk."""

    if not _path_has_directory(path):
        raise ArtifactCaptureError(
            "bare_soname_rejected",
            "file artifact path must be explicit; use platform_image for a loader image",
            public_path=path,
        )
    absolute = os.path.abspath(path)
    pending = deque(_normalize_components([], absolute, []))
    if not pending:
        raise ArtifactCaptureError(
            "artifact_path_not_file",
            "artifact path does not name a file",
            public_path=path,
        )

    root_flags = os.O_RDONLY | os.O_CLOEXEC | os.O_DIRECTORY
    try:
        directory_fd = os.open(os.sep, root_flags)
    except OSError:
        raise ArtifactCaptureError(
            "artifact_root_open_failed",
            "cannot anchor artifact path traversal",
            public_path=path,
        ) from None

    resolved: list[str] = []
    trace: list[tuple[Any, ...]] = []
    followed = 0
    try:
        while pending:
            component = pending.popleft()
            try:
                before = os.stat(component, dir_fd=directory_fd, follow_symlinks=False)
            except OSError:
                raise ArtifactCaptureError(
                    "artifact_path_unreadable",
                    "artifact path cannot be inspected safely",
                    public_path=path,
                ) from None

            if stat.S_ISLNK(before.st_mode):
                followed += 1
                if followed > max_symlinks:
                    raise ArtifactCaptureError(
                        "artifact_symlink_limit",
                        "artifact path exceeds the symlink-resolution limit",
                        public_path=path,
                    )
                try:
                    target = os.readlink(component, dir_fd=directory_fd)
                    after = os.stat(
                        component, dir_fd=directory_fd, follow_symlinks=False
                    )
                except OSError:
                    raise ArtifactCaptureError(
                        "artifact_symlink_drift",
                        "artifact symlink changed during resolution",
                        public_path=path,
                    ) from None
                if _source_fingerprint(before) != _source_fingerprint(after):
                    raise ArtifactCaptureError(
                        "artifact_symlink_drift",
                        "artifact symlink changed during resolution",
                        public_path=path,
                    )
                trace.append(
                    (
                        "symlink",
                        *(_source_fingerprint(before)),
                        target,
                    )
                )
                replacement = _normalize_components(
                    resolved,
                    target,
                    list(pending),
                )
                pending = deque(replacement)
                resolved = []
                replacement_fd: int | None = None
                try:
                    replacement_fd = os.open(os.sep, root_flags)
                except OSError:
                    raise ArtifactCaptureError(
                        "artifact_root_open_failed",
                        "cannot re-anchor artifact path traversal",
                        public_path=path,
                    ) from None
                old_directory_fd = directory_fd
                directory_fd = replacement_fd
                replacement_fd = None
                os.close(old_directory_fd)
                continue

            is_leaf = not pending
            if is_leaf:
                if not stat.S_ISREG(before.st_mode):
                    raise ArtifactCaptureError(
                        "artifact_not_regular",
                        "artifact leaf must resolve to one regular file",
                        public_path=path,
                    )
                try:
                    descriptor = os.open(
                        component,
                        os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW,
                        dir_fd=directory_fd,
                    )
                except OSError:
                    raise ArtifactCaptureError(
                        "artifact_open_failed",
                        "artifact leaf cannot be opened safely",
                        public_path=path,
                    ) from None
                try:
                    opened = os.fstat(descriptor)
                except OSError:
                    os.close(descriptor)
                    raise ArtifactCaptureError(
                        "artifact_stat_failed",
                        "artifact descriptor cannot be inspected",
                        public_path=path,
                    ) from None
                if (
                    not stat.S_ISREG(opened.st_mode)
                    or _stat_identity(opened) != _stat_identity(before)
                    or stat.S_IFMT(opened.st_mode) != stat.S_IFMT(before.st_mode)
                ):
                    os.close(descriptor)
                    raise ArtifactCaptureError(
                        "artifact_not_regular",
                        "artifact leaf must resolve to one regular file",
                        public_path=path,
                    )
                trace.append(("leaf", opened.st_dev, opened.st_ino, opened.st_mode))
                return _OpenedSource(descriptor, opened, tuple(trace))

            if not stat.S_ISDIR(before.st_mode):
                raise ArtifactCaptureError(
                    "artifact_parent_not_directory",
                    "artifact path contains a non-directory parent",
                    public_path=path,
                )
            next_fd: int | None = None
            try:
                try:
                    next_fd = os.open(
                        component,
                        os.O_RDONLY | os.O_CLOEXEC | os.O_DIRECTORY | os.O_NOFOLLOW,
                        dir_fd=directory_fd,
                    )
                    opened_parent = os.fstat(next_fd)
                except OSError:
                    raise ArtifactCaptureError(
                        "artifact_parent_open_failed",
                        "artifact parent cannot be opened safely",
                        public_path=path,
                    ) from None
                if not stat.S_ISDIR(opened_parent.st_mode) or _stat_identity(
                    opened_parent
                ) != _stat_identity(before):
                    raise ArtifactCaptureError(
                        "artifact_parent_drift",
                        "artifact parent changed during traversal",
                        public_path=path,
                    )
                trace.append(
                    (
                        "directory",
                        opened_parent.st_dev,
                        opened_parent.st_ino,
                        stat.S_IMODE(opened_parent.st_mode),
                    )
                )
                resolved.append(component)
                old_directory_fd = directory_fd
                directory_fd = next_fd
                next_fd = None
                os.close(old_directory_fd)
            finally:
                if next_fd is not None:
                    os.close(next_fd)
    finally:
        os.close(directory_fd)

    raise AssertionError("unreachable artifact path traversal")


class ArtifactSnapshotSet:
    """Context-managed set of frozen artifacts and their two projections."""

    publication_status = "not_published"

    def __init__(
        self,
        requests: tuple[ArtifactRequest, ...] | list[ArtifactRequest],
        *,
        max_artifact_bytes: int = DEFAULT_MAX_ARTIFACT_BYTES,
        max_artifacts: int = DEFAULT_MAX_ARTIFACTS,
        max_total_bytes: int = DEFAULT_MAX_TOTAL_BYTES,
        max_symlinks: int = DEFAULT_MAX_SYMLINKS,
        private_parent: str | os.PathLike[str] | None = None,
    ) -> None:
        self._lock = threading.RLock()
        self._root_path: str | None = None
        self._root_fd: int | None = None
        self._root_identity: tuple[int, int] | None = None
        self._cleanup_directory: repository_snapshot.CleanupDirectory | None = None
        self._recovery_anchor_owner: repository_snapshot.OwnedDescriptor | None = None
        self._initial_cleanup_issues: list[CleanupIssue] = []
        self._cleanup_accumulator = repository_snapshot.CleanupAccumulator()
        self._cleanup_unaddressable = False
        self._copies: dict[tuple[str, ArtifactRole], _PrivateCopy] = {}
        self._pending_copies: dict[str, _PendingPrivateCopy] = {}
        self._uncertain_private_leaves: dict[str, str] = {}
        self._artifacts: tuple[FrozenArtifact, ...] = ()
        self._closed = False
        self._finalized = False
        self._cleanup_status: CleanupStatus = "open"
        self._limits = self._validate_limits(
            max_artifact_bytes,
            max_artifacts,
            max_total_bytes,
            max_symlinks,
        )
        try:
            self._capture(tuple(requests), private_parent)
        except BaseException as capture_error:
            issues = self._cleanup()
            self._closed = True
            if issues:
                raise self._cleanup_error(issues) from capture_error
            raise

    @classmethod
    def capture(
        cls,
        requests: tuple[ArtifactRequest, ...] | list[ArtifactRequest],
        **limits: Any,
    ) -> ArtifactSnapshotSet:
        """Capture requests immediately and return their context owner."""

        return cls(requests, **limits)

    @staticmethod
    def _validate_limits(
        max_artifact_bytes: int,
        max_artifacts: int,
        max_total_bytes: int,
        max_symlinks: int,
    ) -> tuple[int, int, int, int]:
        values = (
            max_artifact_bytes,
            max_artifacts,
            max_total_bytes,
            max_symlinks,
        )
        if any(type(value) is not int or value <= 0 for value in values):
            raise ValueError("artifact snapshot limits must be positive integers")
        return values

    def __enter__(self) -> ArtifactSnapshotSet:
        with self._lock:
            if self._closed:
                raise ArtifactSnapshotError(
                    "snapshot_closed", "artifact snapshot set is already closed"
                )
            return self

    def __exit__(self, _type: Any, _value: Any, _traceback: Any) -> None:
        self.close()

    @property
    def artifacts(self) -> tuple[FrozenArtifact, ...]:
        return self._artifacts

    @property
    def closed(self) -> bool:
        return self._closed

    @property
    def finalized(self) -> bool:
        return self._finalized

    @property
    def cleanup_status(self) -> CleanupStatus:
        """Report whether cleanup completed or retained explicit material."""

        return self._cleanup_status

    def for_role(self, role: ArtifactRole) -> tuple[FrozenArtifact, ...]:
        self._validate_role(role)
        return tuple(artifact for artifact in self._artifacts if artifact.role == role)

    def metadata_records(self, role: ArtifactRole) -> list[dict[str, str | None]]:
        """Return public metadata records in request order."""

        return [artifact.metadata_record() for artifact in self.for_role(role)]

    def legacy_records(self, role: ArtifactRole) -> list[dict[str, str | None]]:
        """Return the exact same records for retained family metadata."""

        return self.metadata_records(role)

    def metadata_record(self, artifact: FrozenArtifact) -> dict[str, str | None]:
        with self._lock:
            self._require_owned_open(artifact)
            if artifact._copy_key is not None:
                self._verify_copy(self._copies[artifact._copy_key])
            return {
                "name": artifact.name,
                "path": artifact.path,
                "sha256": artifact.sha256,
            }

    def execution_path(self, artifact: FrozenArtifact) -> str:
        with self._lock:
            self._require_owned_open(artifact)
            if artifact.source_kind == "platform_image":
                return artifact.path
            if artifact._copy_key is None:
                raise ArtifactVerificationError(
                    "artifact_copy_missing",
                    "file artifact has no private execution copy",
                    public_path=artifact.path,
                )
            copy = self._copies[artifact._copy_key]
            self._verify_copy(copy)
            if self._root_path is None:
                raise ArtifactVerificationError(
                    "artifact_root_missing",
                    "private artifact root is unavailable",
                    public_path=artifact.path,
                    artifact_id=copy.artifact_id,
                )
            return os.path.join(self._root_path, copy.leaf)

    def finalize(self) -> None:
        """Verify every frozen file immediately before controller finalization."""

        with self._lock:
            self._require_open()
            for copy in self._copies.values():
                self._verify_copy(copy)
            self._finalized = True

    def redact_private_paths(self, value: Any) -> Any:
        """Recursively replace generated private paths with their public paths."""

        with self._lock:
            replacements = []
            if self._root_path is not None:
                for artifact in self._artifacts:
                    if artifact._copy_key is None:
                        continue
                    copy = self._copies.get(artifact._copy_key)
                    if copy is None:
                        continue
                    replacements.append(
                        (os.path.join(self._root_path, copy.leaf), artifact.path)
                    )
                replacements.append((self._root_path, "<private-artifact-root>"))
            replacements.sort(key=lambda item: len(item[0]), reverse=True)
        return self._redact_value(value, replacements)

    def close(self) -> None:
        """Close descriptors and remove files through the shared cleanup owner."""

        with self._lock:
            if self._closed:
                return
            issues = self._cleanup()
            self._closed = True
            if issues:
                raise self._cleanup_error(issues)

    def _cleanup_error(self, issues: list[CleanupIssue]) -> ArtifactCleanupError:
        aggregate = self._cleanup_accumulator.snapshot()
        return ArtifactCleanupError(
            tuple(issues),
            candidate_paths=tuple(
                os.fspath(path) for path in aggregate.candidate_paths
            ),
            cleanup_status=self._cleanup_status,
        )

    @staticmethod
    def _redact_value(value: Any, replacements: list[tuple[str, str]]) -> Any:
        if isinstance(value, str):
            for private, public in replacements:
                value = value.replace(private, public)
            return value
        if isinstance(value, bytes):
            for private, public in replacements:
                value = value.replace(os.fsencode(private), os.fsencode(public))
            return value
        if isinstance(value, dict):
            return {
                ArtifactSnapshotSet._redact_value(
                    key, replacements
                ): ArtifactSnapshotSet._redact_value(item, replacements)
                for key, item in value.items()
            }
        if isinstance(value, list):
            return [
                ArtifactSnapshotSet._redact_value(item, replacements) for item in value
            ]
        if isinstance(value, tuple):
            return tuple(
                ArtifactSnapshotSet._redact_value(item, replacements) for item in value
            )
        return value

    @staticmethod
    def _validate_role(role: str) -> None:
        if role not in _PRIVATE_MODES:
            raise ValueError("artifact role must be 'binary' or 'library'")

    @staticmethod
    def _validated_request(request: ArtifactRequest) -> tuple[str, str]:
        if not isinstance(request, ArtifactRequest):
            raise TypeError("artifact requests must be ArtifactRequest instances")
        ArtifactSnapshotSet._validate_role(request.role)
        if request.source_kind not in ("file", "platform_image"):
            raise ValueError("artifact source_kind must be 'file' or 'platform_image'")
        if type(request.require_source_executable) is not bool:
            raise TypeError("require_source_executable must be a boolean")
        if not isinstance(request.name, str) or not request.name.strip():
            raise ValueError("artifact name must be a non-empty string")
        try:
            public_path = os.fspath(request.path)
        except TypeError:
            raise TypeError("artifact path must be string-like") from None
        if not isinstance(public_path, str):
            raise TypeError("artifact path must decode to a string")
        if not public_path or "\0" in public_path:
            raise ValueError("artifact path must be a non-empty NUL-free string")
        if request.source_kind == "platform_image" and request.role != "library":
            raise ValueError("platform_image is permitted only for library artifacts")
        if request.source_kind == "platform_image" and (
            sys.platform != "darwin"
            or request.name != "Accelerate"
            or public_path != DEFAULT_ACCELERATE_IMAGE
        ):
            raise ArtifactCaptureError(
                "platform_image_not_allowed",
                "platform_image is restricted to the approved loader-managed image",
                public_path=public_path,
            )
        return request.name, public_path

    def _capture(
        self,
        requests: tuple[ArtifactRequest, ...],
        private_parent: str | os.PathLike[str] | None,
    ) -> None:
        max_artifact_bytes, max_artifacts, max_total_bytes, _ = self._limits
        normalized: list[tuple[ArtifactRequest, str, str]] = []
        seen_requests: set[tuple[str, str, str, str, bool]] = set()
        file_groups: dict[str, list[tuple[ArtifactRequest, str, str]]] = {}
        resolved_groups: dict[str, tuple[bytes, str]] = {}
        for request in requests:
            name, public_path = self._validated_request(request)
            request_key = (
                name,
                public_path,
                request.role,
                request.source_kind,
                request.require_source_executable,
            )
            if request_key in seen_requests:
                continue
            seen_requests.add(request_key)
            entry = request, name, public_path
            normalized.append(entry)
            if request.source_kind == "file":
                group_key = os.path.normcase(os.path.abspath(public_path))
                file_groups.setdefault(group_key, []).append(entry)

        for group_key, entries in file_groups.items():
            if all(
                entry[0].role == "binary"
                and entry[0].require_source_executable is False
                for entry in entries
            ):
                resolved = _resolved_interpreter_source(entries[0][2])
                if resolved is not None:
                    resolved_groups[group_key] = resolved

        if len(file_groups) > max_artifacts:
            raise ArtifactCaptureError(
                "artifact_count_limit",
                "artifact set exceeds the unique file-count limit",
            )
        if file_groups:
            if os.name != "posix":
                raise ArtifactCaptureError(
                    "artifact_platform_unsupported",
                    "direct benchmark artifact snapshots require POSIX descriptor APIs",
                )
            self._create_private_root(private_parent)

        total_bytes = 0
        copies_by_group: dict[tuple[str, ArtifactRole], _PrivateCopy] = {}
        for group_index, (group_key, entries) in enumerate(file_groups.items()):
            representative_path = entries[0][2]
            roles = tuple(dict.fromkeys(entry[0].role for entry in entries))
            require_executable = any(
                entry[0].role == "binary" and entry[0].require_source_executable
                for entry in entries
            )
            resolved = resolved_groups.get(group_key)
            if resolved is None:
                copies, source_size = self._capture_file_group(
                    representative_path,
                    group_key,
                    roles,
                    group_index,
                    max_artifact_bytes,
                    max_total_bytes - total_bytes,
                    require_executable,
                )
            else:
                copies, source_size = self._capture_frozen_source_group(
                    representative_path,
                    group_key,
                    roles,
                    group_index,
                    max_artifact_bytes,
                    max_total_bytes - total_bytes,
                    resolved,
                )
            total_bytes += source_size
            for role, copy in copies.items():
                key = group_key, role
                self._copies[key] = copy
                copies_by_group[key] = copy

        artifacts: list[FrozenArtifact] = []
        for request, name, public_path in normalized:
            if request.source_kind == "platform_image":
                artifacts.append(
                    FrozenArtifact(
                        name=name,
                        path=public_path,
                        role=request.role,
                        sha256=None,
                        source_kind="platform_image",
                        size=None,
                        _owner=self,
                        _copy_key=None,
                    )
                )
                continue
            group_key = os.path.normcase(os.path.abspath(public_path))
            copy_key = group_key, request.role
            copy = copies_by_group[copy_key]
            artifacts.append(
                FrozenArtifact(
                    name=name,
                    path=public_path,
                    role=request.role,
                    sha256=copy.sha256,
                    source_kind="file",
                    size=copy.size,
                    _owner=self,
                    _copy_key=copy_key,
                )
            )
        self._artifacts = tuple(artifacts)

    def _create_private_root(
        self, private_parent: str | os.PathLike[str] | None
    ) -> None:
        parent_path = Path(
            tempfile.gettempdir()
            if private_parent is None
            else os.fspath(private_parent)
        ).absolute()
        parent_owner: repository_snapshot.OwnedDescriptor | None = None
        arena: repository_snapshot.CleanupArena | None = None
        try:
            parent_descriptor = os.open(
                parent_path,
                os.O_RDONLY | os.O_CLOEXEC | os.O_DIRECTORY | os.O_NOFOLLOW,
            )
            parent_owner = repository_snapshot.OwnedDescriptor.take(
                parent_descriptor,
                parent_path,
            )
            parent = repository_snapshot.DirectoryAnchor(
                parent_owner.fileno(),
                parent_path,
            )
            arena = repository_snapshot.CleanupArena.open(parent)
            directory = repository_snapshot.CleanupDirectory.create(
                arena,
                prefix=".zynum-benchmark-artifacts-",
                suffix="",
            )
            self._cleanup_directory = directory
            self._recovery_anchor_owner = parent_owner
            parent_owner = None
            self._root_path = os.fspath(directory.path)
            self._root_fd = directory.fileno()
            self._root_identity = directory.identity
        except repository_snapshot.CleanupFailure as error:
            outcome = error.outcome
            if arena is not None:
                outcome = repository_snapshot.finalize_arena_outcome(arena, outcome)
                arena = None
            self._record_shared_outcome(outcome)
            self._initial_cleanup_issues.extend(self._adapt_shared_outcome(outcome))
            raise ArtifactCaptureError(
                "private_root_create_failed",
                "cannot create the private benchmark artifact root",
            ) from None
        except OSError:
            raise ArtifactCaptureError(
                "private_root_create_failed",
                "cannot create the private benchmark artifact root",
            ) from None
        finally:
            if arena is not None:
                outcome = repository_snapshot.finalize_arena_outcome(
                    arena,
                    repository_snapshot.CleanupOutcome(
                        disposition=repository_snapshot.CleanupDisposition.REMOVED,
                        recovery_paths=(),
                        issues=(),
                        arena_identity=arena.identity,
                        recovery_anchor_identity=arena.anchor.identity,
                    ),
                )
                self._record_shared_outcome(outcome)
                self._initial_cleanup_issues.extend(self._adapt_shared_outcome(outcome))
            if parent_owner is not None:
                try:
                    parent_owner.close_once()
                except OSError:
                    self._initial_cleanup_issues.append(
                        CleanupIssue(
                            "private_parent_descriptor_close_failed",
                            None,
                        )
                    )
        if self._initial_cleanup_issues:
            raise ArtifactCaptureError(
                "private_root_create_failed",
                "cannot create the private benchmark artifact root",
            )

    def _close_private_descriptor_once(
        self,
        descriptor: int,
        leaf: str,
        artifact_id: str,
    ) -> bool:
        root_path = getattr(self, "_root_path", None)
        recovery_path = Path(root_path) / leaf if root_path is not None else Path(leaf)
        owner = repository_snapshot.OwnedDescriptor.take(
            descriptor,
            recovery_path,
        )
        try:
            owner.close_once()
        except OSError:
            uncertain = getattr(self, "_uncertain_private_leaves", None)
            if uncertain is None:
                uncertain = {}
                self._uncertain_private_leaves = uncertain
            uncertain.setdefault(leaf, artifact_id)
            return False
        return True

    def _capture_file_group(
        self,
        public_path: str,
        group_key: str,
        roles: tuple[ArtifactRole, ...],
        group_index: int,
        max_artifact_bytes: int,
        remaining_total_bytes: int,
        require_executable: bool,
    ) -> tuple[dict[ArtifactRole, _PrivateCopy], int]:
        max_symlinks = self._limits[3]
        opened = _open_resolved_regular(public_path, max_symlinks)
        source_fd = opened.descriptor
        initial = opened.metadata
        try:
            self._validate_source(
                initial,
                public_path,
                require_executable=require_executable,
            )
            if initial.st_size > max_artifact_bytes:
                raise ArtifactCaptureError(
                    "artifact_size_limit",
                    "artifact exceeds the per-file byte limit",
                    public_path=public_path,
                )
            if initial.st_size > remaining_total_bytes:
                raise ArtifactCaptureError(
                    "artifact_total_bytes_limit",
                    "artifact set exceeds the cumulative byte limit",
                    public_path=public_path,
                )
            outputs = self._create_output_descriptors(roles, group_index, public_path)
            digest = hashlib.sha256()
            remaining = initial.st_size
            try:
                while remaining:
                    try:
                        chunk = os.read(source_fd, min(_COPY_CHUNK_BYTES, remaining))
                    except OSError:
                        raise ArtifactCaptureError(
                            "artifact_read_failed",
                            "artifact source read failed",
                            public_path=public_path,
                        ) from None
                    if not chunk:
                        raise ArtifactCaptureError(
                            "artifact_short_read",
                            "artifact source ended before its initial size",
                            public_path=public_path,
                        )
                    digest.update(chunk)
                    for descriptor in outputs.values():
                        self._write_all(descriptor, chunk, public_path)
                    remaining -= len(chunk)
                try:
                    extra = os.read(source_fd, 1)
                except OSError:
                    raise ArtifactCaptureError(
                        "artifact_read_failed",
                        "artifact source final bound check failed",
                        public_path=public_path,
                    ) from None
                if extra:
                    raise ArtifactCaptureError(
                        "artifact_growth",
                        "artifact source grew beyond its initial size",
                        public_path=public_path,
                    )
                try:
                    final_source = os.fstat(source_fd)
                except OSError:
                    raise ArtifactCaptureError(
                        "artifact_post_stat_failed",
                        "artifact source post-read stat failed",
                        public_path=public_path,
                    ) from None
                if _source_fingerprint(initial) != _source_fingerprint(final_source):
                    raise ArtifactCaptureError(
                        "artifact_source_drift",
                        "artifact source changed while it was captured",
                        public_path=public_path,
                    )
                path_check = _open_resolved_regular(public_path, max_symlinks)
                try:
                    if (
                        path_check.path_fingerprint != opened.path_fingerprint
                        or _stat_identity(path_check.metadata)
                        != _stat_identity(initial)
                    ):
                        raise ArtifactCaptureError(
                            "artifact_path_drift",
                            "artifact path changed while it was captured",
                            public_path=public_path,
                        )
                finally:
                    os.close(path_check.descriptor)
                hexdigest = digest.hexdigest()
                copies = self._finish_outputs(
                    outputs,
                    roles,
                    group_key,
                    group_index,
                    public_path,
                    initial,
                    hexdigest,
                )
            except BaseException:
                while outputs:
                    role, descriptor = outputs.popitem()
                    self._close_private_descriptor_once(
                        descriptor,
                        self._leaf(group_index, role),
                        "artifact-{:04d}-{}".format(group_index, role),
                    )
                raise
        finally:
            os.close(source_fd)
        return copies, initial.st_size

    def _capture_frozen_source_group(
        self,
        public_path: str,
        group_key: str,
        roles: tuple[ArtifactRole, ...],
        group_index: int,
        max_artifact_bytes: int,
        remaining_total_bytes: int,
        resolved: tuple[bytes, str],
    ) -> tuple[dict[ArtifactRole, _PrivateCopy], int]:
        contents, hexdigest = resolved
        size = len(contents)
        if size > max_artifact_bytes:
            raise ArtifactCaptureError(
                "artifact_size_limit",
                "artifact exceeds the per-file byte limit",
                public_path=public_path,
            )
        if size > remaining_total_bytes:
            raise ArtifactCaptureError(
                "artifact_total_bytes_limit",
                "artifact set exceeds the cumulative byte limit",
                public_path=public_path,
            )
        outputs = self._create_output_descriptors(roles, group_index, public_path)
        try:
            for descriptor in outputs.values():
                self._write_all(descriptor, contents, public_path)
            source = os.stat_result(
                (
                    stat.S_IFREG | 0o400,
                    -1,
                    -1,
                    1,
                    os.getuid(),
                    os.getgid(),
                    size,
                    0,
                    0,
                    0,
                )
            )
            copies = self._finish_outputs(
                outputs,
                roles,
                group_key,
                group_index,
                public_path,
                source,
                hexdigest,
            )
        except BaseException:
            while outputs:
                role, descriptor = outputs.popitem()
                self._close_private_descriptor_once(
                    descriptor,
                    self._leaf(group_index, role),
                    "artifact-{:04d}-{}".format(group_index, role),
                )
            raise
        return copies, size

    @staticmethod
    def _validate_source(
        metadata: os.stat_result,
        public_path: str,
        *,
        require_executable: bool,
    ) -> None:
        if not stat.S_ISREG(metadata.st_mode):
            raise ArtifactCaptureError(
                "artifact_not_regular",
                "artifact source must be a regular file",
                public_path=public_path,
            )
        mode = stat.S_IMODE(metadata.st_mode)
        if mode & (stat.S_ISUID | stat.S_ISGID):
            raise ArtifactCaptureError(
                "artifact_setid_rejected",
                "set-id benchmark artifacts are not permitted",
                public_path=public_path,
            )
        if mode & 0o022:
            raise ArtifactCaptureError(
                "artifact_unsafe_mode",
                "group/world-writable benchmark artifacts are not permitted",
                public_path=public_path,
            )
        if require_executable and not mode & 0o111:
            raise ArtifactCaptureError(
                "artifact_not_executable",
                "binary artifact source has no execute permission",
                public_path=public_path,
            )
        if not mode & 0o444:
            raise ArtifactCaptureError(
                "artifact_not_readable",
                "artifact source has no read permission",
                public_path=public_path,
            )

    def _create_output_descriptors(
        self,
        roles: tuple[ArtifactRole, ...],
        group_index: int,
        public_path: str,
    ) -> dict[ArtifactRole, int]:
        if self._root_fd is None:
            raise ArtifactCaptureError(
                "private_root_missing",
                "private benchmark artifact root is unavailable",
                public_path=public_path,
            )
        outputs: dict[ArtifactRole, int] = {}
        for role in roles:
            leaf = self._leaf(group_index, role)
            writer: int | None = None
            try:
                owned_descriptor = os.open(
                    leaf,
                    os.O_RDWR | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | os.O_NOFOLLOW,
                    0o600,
                    dir_fd=self._root_fd,
                )
                pending = _PendingPrivateCopy(
                    artifact_id="artifact-{:04d}-{}".format(group_index, role),
                    public_path=public_path,
                    leaf=leaf,
                    descriptor=owned_descriptor,
                    identity=None,
                )
                self._pending_copies[leaf] = pending
                pending.identity = _stat_identity(os.fstat(owned_descriptor))
                writer = os.dup(owned_descriptor)
                outputs[role] = writer
                writer = None
            except OSError:
                if writer is not None:
                    descriptor_to_close = writer
                    writer = None
                    self._close_private_descriptor_once(
                        descriptor_to_close,
                        leaf,
                        pending.artifact_id,
                    )
                while outputs:
                    output_role, descriptor = outputs.popitem()
                    self._close_private_descriptor_once(
                        descriptor,
                        self._leaf(group_index, output_role),
                        "artifact-{:04d}-{}".format(group_index, output_role),
                    )
                raise ArtifactCaptureError(
                    "private_artifact_create_failed",
                    "cannot create a private benchmark artifact",
                    public_path=public_path,
                ) from None
        return outputs

    @staticmethod
    def _write_all(descriptor: int, contents: bytes, public_path: str) -> None:
        offset = 0
        while offset < len(contents):
            try:
                written = os.write(descriptor, contents[offset:])
            except OSError:
                raise ArtifactCaptureError(
                    "private_artifact_write_failed",
                    "private benchmark artifact write failed",
                    public_path=public_path,
                ) from None
            if written <= 0:
                raise ArtifactCaptureError(
                    "private_artifact_short_write",
                    "private benchmark artifact write made no progress",
                    public_path=public_path,
                )
            offset += written

    def _finish_outputs(
        self,
        outputs: dict[ArtifactRole, int],
        roles: tuple[ArtifactRole, ...],
        group_key: str,
        group_index: int,
        public_path: str,
        source: os.stat_result,
        hexdigest: str,
    ) -> dict[ArtifactRole, _PrivateCopy]:
        if self._root_fd is None:
            raise ArtifactCaptureError(
                "private_root_missing",
                "private benchmark artifact root is unavailable",
                public_path=public_path,
            )
        finished: dict[ArtifactRole, _PrivateCopy] = {}
        try:
            for role in roles:
                writer = outputs.pop(role)
                mode = _PRIVATE_MODES[role]
                leaf = self._leaf(group_index, role)
                artifact_id = "artifact-{:04d}-{}".format(group_index, role)
                writer_failed = False
                try:
                    os.fchmod(writer, mode)
                    os.fsync(writer)
                    staged = os.fstat(writer)
                except OSError:
                    writer_failed = True
                finally:
                    descriptor_to_close = writer
                    writer = -1
                    if not self._close_private_descriptor_once(
                        descriptor_to_close,
                        leaf,
                        artifact_id,
                    ):
                        writer_failed = True
                if writer_failed:
                    raise ArtifactCaptureError(
                        "private_artifact_finalize_failed",
                        "private benchmark artifact finalization failed",
                        public_path=public_path,
                    ) from None

                descriptor: int | None = None
                try:
                    try:
                        descriptor = os.open(
                            leaf,
                            os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW,
                            dir_fd=self._root_fd,
                        )
                        opened = os.fstat(descriptor)
                    except OSError:
                        raise ArtifactCaptureError(
                            "private_artifact_finalize_failed",
                            "private benchmark artifact finalization failed",
                            public_path=public_path,
                        ) from None
                    copy = _PrivateCopy(
                        artifact_id=artifact_id,
                        role=role,
                        public_path=public_path,
                        leaf=leaf,
                        descriptor=descriptor,
                        identity=_stat_identity(opened),
                        source_identity=_stat_identity(source),
                        size=source.st_size,
                        mode=mode,
                        sha256=hexdigest,
                    )
                    if (
                        _stat_identity(staged) != copy.identity
                        or copy.identity == copy.source_identity
                    ):
                        raise ArtifactCaptureError(
                            "private_artifact_identity_invalid",
                            "private artifact is not one independent stable inode",
                            public_path=public_path,
                            artifact_id=artifact_id,
                        )
                    self._verify_copy(copy, capture=True)
                    finished[role] = copy
                    # Ownership moves to ``finished`` only after verification.
                    descriptor = None
                finally:
                    if descriptor is not None:
                        descriptor_to_close = descriptor
                        descriptor = None
                        self._close_private_descriptor_once(
                            descriptor_to_close,
                            leaf,
                            artifact_id,
                        )
        except BaseException:
            while finished:
                _finished_role, copy = finished.popitem()
                self._close_private_descriptor_once(
                    copy.descriptor,
                    copy.leaf,
                    copy.artifact_id,
                )
            raise
        return finished

    @staticmethod
    def _leaf(group_index: int, role: ArtifactRole) -> str:
        return "artifact-{:04d}-{}".format(group_index, role)

    def _verify_root(self, public_path: str, artifact_id: str) -> None:
        if (
            self._root_path is None
            or self._root_fd is None
            or self._root_identity is None
        ):
            raise ArtifactVerificationError(
                "private_root_missing",
                "private benchmark artifact root is unavailable",
                public_path=public_path,
                artifact_id=artifact_id,
            )
        try:
            path_metadata = os.lstat(self._root_path)
            opened = os.fstat(self._root_fd)
        except OSError:
            raise ArtifactVerificationError(
                "private_root_unreadable",
                "private benchmark artifact root cannot be verified",
                public_path=public_path,
                artifact_id=artifact_id,
            ) from None
        if (
            not stat.S_ISDIR(path_metadata.st_mode)
            or _stat_identity(path_metadata) != self._root_identity
            or _stat_identity(opened) != self._root_identity
            or stat.S_IMODE(path_metadata.st_mode) != 0o700
            or stat.S_IMODE(opened.st_mode) != 0o700
        ):
            raise ArtifactVerificationError(
                "private_root_drift",
                "private benchmark artifact root changed after capture",
                public_path=public_path,
                artifact_id=artifact_id,
            )

    def _verify_copy(self, copy: _PrivateCopy, *, capture: bool = False) -> None:
        self._verify_root(copy.public_path, copy.artifact_id)
        assert self._root_fd is not None
        path_fd: int | None = None
        try:
            try:
                path_metadata = os.stat(
                    copy.leaf, dir_fd=self._root_fd, follow_symlinks=False
                )
                descriptor_metadata = os.fstat(copy.descriptor)
                path_fd = os.open(
                    copy.leaf,
                    os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW,
                    dir_fd=self._root_fd,
                )
                opened = os.fstat(path_fd)
            except OSError:
                raise ArtifactVerificationError(
                    "private_artifact_unreadable",
                    "private benchmark artifact cannot be verified",
                    public_path=copy.public_path,
                    artifact_id=copy.artifact_id,
                ) from None
            if (
                not stat.S_ISREG(opened.st_mode)
                or _stat_identity(path_metadata) != copy.identity
                or _stat_identity(descriptor_metadata) != copy.identity
                or _stat_identity(opened) != copy.identity
                or copy.identity == copy.source_identity
                or path_metadata.st_size != copy.size
                or descriptor_metadata.st_size != copy.size
                or opened.st_size != copy.size
                or stat.S_IMODE(path_metadata.st_mode) != copy.mode
                or stat.S_IMODE(descriptor_metadata.st_mode) != copy.mode
                or stat.S_IMODE(opened.st_mode) != copy.mode
            ):
                raise ArtifactVerificationError(
                    "private_artifact_drift",
                    "private benchmark artifact identity, size, or mode changed",
                    public_path=copy.public_path,
                    artifact_id=copy.artifact_id,
                )
            before = _source_fingerprint(opened)
            digest = hashlib.sha256()
            remaining = copy.size
            while remaining:
                try:
                    chunk = os.read(path_fd, min(_COPY_CHUNK_BYTES, remaining))
                except OSError:
                    raise ArtifactVerificationError(
                        "private_artifact_read_failed",
                        "private benchmark artifact verification read failed",
                        public_path=copy.public_path,
                        artifact_id=copy.artifact_id,
                    ) from None
                if not chunk:
                    raise ArtifactVerificationError(
                        "private_artifact_short_read",
                        "private benchmark artifact ended before its frozen size",
                        public_path=copy.public_path,
                        artifact_id=copy.artifact_id,
                    )
                digest.update(chunk)
                remaining -= len(chunk)
            try:
                extra = os.read(path_fd, 1)
                after = os.fstat(path_fd)
            except OSError:
                raise ArtifactVerificationError(
                    "private_artifact_read_failed",
                    "private benchmark artifact final verification failed",
                    public_path=copy.public_path,
                    artifact_id=copy.artifact_id,
                ) from None
            if extra:
                raise ArtifactVerificationError(
                    "private_artifact_growth",
                    "private benchmark artifact grew beyond its frozen size",
                    public_path=copy.public_path,
                    artifact_id=copy.artifact_id,
                )
            if before != _source_fingerprint(after):
                raise ArtifactVerificationError(
                    "private_artifact_drift",
                    "private benchmark artifact changed during verification",
                    public_path=copy.public_path,
                    artifact_id=copy.artifact_id,
                )
            if digest.hexdigest() != copy.sha256:
                raise ArtifactVerificationError(
                    "private_artifact_digest_mismatch",
                    "private benchmark artifact digest no longer matches capture",
                    public_path=copy.public_path,
                    artifact_id=copy.artifact_id,
                )
        finally:
            if path_fd is not None:
                descriptor_to_close = path_fd
                path_fd = None
                if not self._close_private_descriptor_once(
                    descriptor_to_close,
                    copy.leaf,
                    copy.artifact_id,
                ):
                    raise ArtifactVerificationError(
                        "private_artifact_descriptor_close_failed",
                        "private benchmark artifact descriptor close was uncertain",
                        public_path=copy.public_path,
                        artifact_id=copy.artifact_id,
                    ) from None
        if capture:
            return

    def _require_open(self) -> None:
        if self._closed:
            raise ArtifactSnapshotError(
                "snapshot_closed", "artifact snapshot set is already closed"
            )

    def _require_owned_open(self, artifact: FrozenArtifact) -> None:
        self._require_open()
        if artifact._owner is not self or not any(
            artifact is owned for owned in self._artifacts
        ):
            raise ArtifactSnapshotError(
                "artifact_not_owned",
                "frozen artifact is not owned by this snapshot set",
            )

    @staticmethod
    def _pending_cleanup_target(
        pending: _PendingPrivateCopy,
    ) -> _CleanupTarget | None:
        try:
            before = os.fstat(pending.descriptor)
            if not stat.S_ISREG(before.st_mode):
                return None
            identity = _stat_identity(before)
            if pending.identity is not None and identity != pending.identity:
                return None
            digest = hashlib.sha256()
            offset = 0
            while offset < before.st_size:
                chunk = os.pread(
                    pending.descriptor,
                    min(_COPY_CHUNK_BYTES, before.st_size - offset),
                    offset,
                )
                if not chunk:
                    return None
                digest.update(chunk)
                offset += len(chunk)
            if os.pread(pending.descriptor, 1, before.st_size):
                return None
            after = os.fstat(pending.descriptor)
        except OSError:
            return None
        if _source_fingerprint(before) != _source_fingerprint(after):
            return None
        return _CleanupTarget(
            artifact_id=pending.artifact_id,
            leaf=pending.leaf,
            descriptor=pending.descriptor,
            identity=identity,
            size=before.st_size,
            mode=stat.S_IMODE(before.st_mode),
            sha256=digest.hexdigest(),
        )

    @staticmethod
    def _verify_cleanup_target(
        target: _CleanupTarget,
        descriptor: int,
        claimed_path: os.stat_result,
    ) -> repository_snapshot.ClaimVerification:
        try:
            before = os.fstat(descriptor)
            digest = hashlib.sha256()
            offset = 0
            while offset < target.size:
                chunk = os.pread(
                    descriptor,
                    min(_COPY_CHUNK_BYTES, target.size - offset),
                    offset,
                )
                if not chunk:
                    break
                digest.update(chunk)
                offset += len(chunk)
            extra = os.pread(descriptor, 1, target.size)
            after = os.fstat(descriptor)
        except OSError:
            return repository_snapshot.ClaimVerification.UNKNOWN
        matches = (
            stat.S_ISREG(claimed_path.st_mode)
            and stat.S_ISREG(before.st_mode)
            and _stat_identity(claimed_path) == target.identity
            and _stat_identity(before) == target.identity
            and _stat_identity(after) == target.identity
            and claimed_path.st_size == target.size
            and before.st_size == target.size
            and after.st_size == target.size
            and stat.S_IMODE(claimed_path.st_mode) == target.mode
            and stat.S_IMODE(before.st_mode) == target.mode
            and stat.S_IMODE(after.st_mode) == target.mode
            and offset == target.size
            and not extra
            and _source_fingerprint(before) == _source_fingerprint(after)
            and digest.hexdigest() == target.sha256
        )
        return (
            repository_snapshot.ClaimVerification.MATCH
            if matches
            else repository_snapshot.ClaimVerification.FOREIGN
        )

    @staticmethod
    def _adapt_shared_issues(
        shared_issues: tuple[repository_snapshot.CleanupIssue, ...],
        recovery_paths: tuple[Path, ...],
    ) -> list[CleanupIssue]:
        remaining_paths = [os.fspath(path) for path in recovery_paths]
        issues: list[CleanupIssue] = []
        for shared_issue in shared_issues:
            diagnostic_path = os.fspath(shared_issue.path)
            recovery_path = None
            if diagnostic_path in remaining_paths:
                recovery_path = diagnostic_path
                remaining_paths.remove(diagnostic_path)
            issues.append(CleanupIssue(shared_issue.code, None, recovery_path))
        issues.extend(
            CleanupIssue("cleanup_recovery_required", None, path)
            for path in remaining_paths
        )
        return issues

    @classmethod
    def _adapt_shared_outcome(
        cls,
        outcome: repository_snapshot.CleanupOutcome,
    ) -> list[CleanupIssue]:
        return cls._adapt_shared_issues(outcome.issues, outcome.recovery_paths)

    def _record_shared_outcome(
        self,
        outcome: repository_snapshot.CleanupOutcome,
    ) -> None:
        self._cleanup_accumulator.record(outcome)
        self._cleanup_unaddressable = self._cleanup_accumulator.snapshot().unaddressable

    @staticmethod
    def _adapt_cleanup_outcome(
        target: _CleanupTarget,
        outcome: repository_snapshot.CleanupOutcome,
    ) -> list[CleanupIssue]:
        setup_failed = any(
            issue.code == "cleanup_quarantine_setup_failed" for issue in outcome.issues
        )
        code_map = {
            "cleanup_quarantine_create_failed": "cleanup_quarantine_create_failed",
            "cleanup_quarantine_setup_failed": (
                "cleanup_quarantine_credential_unverified"
            ),
            "cleanup_claim_failed": "private_artifact_claim_failed",
            "cleanup_claim_destination_fsync_failed": (
                "private_artifact_claim_quarantine_fsync_failed"
            ),
            "cleanup_claim_source_fsync_failed": (
                "private_artifact_claim_source_fsync_failed"
            ),
            "cleanup_public_absence_uninspectable": (
                "private_artifact_name_uninspectable"
            ),
            "cleanup_public_name_reappeared": "private_artifact_name_recreated",
            "cleanup_claimed_uninspectable": "private_artifact_uninspectable",
            "cleanup_claimed_foreign": "private_artifact_replaced",
            "cleanup_claimed_unknown": "private_artifact_uninspectable",
            "cleanup_claimed_descriptor_close_uncertain": (
                "cleanup_descriptor_close_failed"
            ),
            "cleanup_claim_remove_failed": "private_artifact_remove_failed",
            "cleanup_quarantine_fsync_failed": "cleanup_quarantine_remove_failed",
            "cleanup_quarantine_teardown_failed": "cleanup_quarantine_remove_failed",
        }
        recovery_paths = [os.fspath(path) for path in outcome.recovery_paths]
        remaining_paths = list(recovery_paths)
        issues: list[CleanupIssue] = []
        directory_issue_codes = {
            "cleanup_arena_binding_absent",
            "cleanup_arena_binding_rebound",
            "cleanup_arena_binding_unknown",
            "cleanup_arena_descriptor_close_uncertain",
            "cleanup_arena_fsync_failed",
            "cleanup_quarantine_create_failed",
            "cleanup_quarantine_setup_failed",
            "cleanup_quarantine_descriptor_close_uncertain",
            "cleanup_quarantine_fsync_failed",
            "cleanup_quarantine_teardown_failed",
        }
        for shared_issue in outcome.issues:
            shared_code = shared_issue.code
            code = shared_code
            if code == "cleanup_quarantine_setup_failed":
                code = (
                    "cleanup_quarantine_credential_unverified"
                    if "credential" in str(shared_issue.error)
                    else "cleanup_quarantine_create_failed"
                )
            elif code == "cleanup_quarantine_descriptor_close_uncertain":
                code = (
                    "cleanup_quarantine_setup_descriptor_close_failed"
                    if setup_failed
                    else "quarantine_descriptor_close_failed"
                )
            else:
                code = code_map.get(code, code)
            diagnostic_path = os.fspath(shared_issue.path)
            recovery_path = None
            if diagnostic_path in remaining_paths:
                recovery_path = diagnostic_path
                remaining_paths.remove(diagnostic_path)
            artifact_id = (
                None if shared_code in directory_issue_codes else target.artifact_id
            )
            issues.append(CleanupIssue(code, artifact_id, recovery_path))
        issues.extend(
            CleanupIssue("cleanup_recovery_required", target.artifact_id, path)
            for path in remaining_paths
        )
        return issues

    def _cleanup_target(
        self,
        target: _CleanupTarget,
        source: repository_snapshot.DirectoryAnchor,
        arena: repository_snapshot.CleanupArena,
    ) -> tuple[list[CleanupIssue], bool]:
        try:
            outcome = repository_snapshot.claim_and_remove(
                source,
                target.leaf,
                lambda descriptor, claimed_path: self._verify_cleanup_target(
                    target, descriptor, claimed_path
                ),
                quarantine_prefix=".zynum-benchmark-artifact-quarantine-",
                quarantine_suffix="",
                claimed_name=target.leaf,
                arena=arena,
            )
        except repository_snapshot.CleanupFailure as error:
            outcome = error.outcome
        self._record_shared_outcome(outcome)
        issues = self._adapt_cleanup_outcome(target, outcome)
        return issues, outcome.disposition in {
            repository_snapshot.CleanupDisposition.ABSENT,
            repository_snapshot.CleanupDisposition.REMOVED,
        }

    def _cleanup(self) -> list[CleanupIssue]:
        issues = list(self._initial_cleanup_issues)
        self._initial_cleanup_issues.clear()
        root_fd = self._root_fd
        cleanup_directory = self._cleanup_directory
        if cleanup_directory is not None:
            arena_binding = cleanup_directory.arena.binding()
            if arena_binding is not repository_snapshot.ArenaBinding.BOUND:
                binding_outcome = repository_snapshot.CleanupOutcome(
                    disposition=repository_snapshot.CleanupDisposition.UNADDRESSABLE,
                    recovery_paths=(),
                    issues=(
                        repository_snapshot.CleanupIssue(
                            f"cleanup_arena_binding_{arena_binding.name.lower()}",
                            cleanup_directory.arena.path,
                        ),
                    ),
                    arena_binding=arena_binding,
                    arena_identity=cleanup_directory.arena.identity,
                    recovery_anchor_identity=(cleanup_directory.arena.anchor.identity),
                )
                self._record_shared_outcome(binding_outcome)
                issues.extend(self._adapt_shared_outcome(binding_outcome))
        targets: list[_CleanupTarget] = []
        claimed_leaves: set[str] = set()
        for copy in self._copies.values():
            targets.append(
                _CleanupTarget(
                    artifact_id=copy.artifact_id,
                    leaf=copy.leaf,
                    descriptor=copy.descriptor,
                    identity=copy.identity,
                    size=copy.size,
                    mode=copy.mode,
                    sha256=copy.sha256,
                )
            )
            claimed_leaves.add(copy.leaf)
        for pending in self._pending_copies.values():
            if pending.leaf in claimed_leaves:
                continue
            target = self._pending_cleanup_target(pending)
            if target is None:
                issues.append(
                    CleanupIssue("private_artifact_uninspectable", pending.artifact_id)
                )
                continue
            targets.append(target)
            claimed_leaves.add(pending.leaf)

        uncertain_leaves: set[str] = set(self._uncertain_private_leaves)
        for leaf, artifact_id in self._uncertain_private_leaves.items():
            recovery_path = (
                os.path.join(self._root_path, leaf)
                if self._root_path is not None and not self._cleanup_unaddressable
                else None
            )
            issues.append(
                CleanupIssue("descriptor_close_failed", artifact_id, recovery_path)
            )
        closed_descriptors: set[int] = set()
        descriptor_owners = [
            (copy.descriptor, copy.artifact_id, copy.leaf)
            for copy in self._copies.values()
        ]
        descriptor_owners.extend(
            (pending.descriptor, pending.artifact_id, pending.leaf)
            for pending in self._pending_copies.values()
        )
        leaves_by_descriptor: dict[int, set[str]] = {}
        for descriptor, _artifact_id, leaf in descriptor_owners:
            leaves_by_descriptor.setdefault(descriptor, set()).add(leaf)
        for descriptor, artifact_id, leaf in descriptor_owners:
            if descriptor in closed_descriptors:
                continue
            closed_descriptors.add(descriptor)
            try:
                os.close(descriptor)
            except OSError:
                uncertain_leaves.update(leaves_by_descriptor[descriptor])
                recovery_path = (
                    os.path.join(self._root_path, leaf)
                    if self._root_path is not None and not self._cleanup_unaddressable
                    else None
                )
                issues.append(
                    CleanupIssue("descriptor_close_failed", artifact_id, recovery_path)
                )
        self._copies.clear()
        self._pending_copies.clear()
        self._uncertain_private_leaves.clear()

        root_teardown_permitted = (
            self._root_identity is not None and not self._cleanup_unaddressable
        )
        if targets:
            if self._cleanup_unaddressable:
                root_teardown_permitted = False
            elif (
                root_fd is None or self._root_path is None or cleanup_directory is None
            ):
                root_teardown_permitted = False
                issues.extend(
                    CleanupIssue("private_root_missing", target.artifact_id)
                    for target in targets
                    if target.leaf not in uncertain_leaves
                )
            else:
                source = repository_snapshot.DirectoryAnchor(
                    root_fd,
                    Path(self._root_path),
                )
                for target in targets:
                    if target.leaf in uncertain_leaves:
                        root_teardown_permitted = False
                        continue
                    target_issues, completed = self._cleanup_target(
                        target,
                        source,
                        cleanup_directory.arena,
                    )
                    issues.extend(target_issues)
                    root_teardown_permitted = root_teardown_permitted and completed

        self._root_fd = None
        self._cleanup_directory = None
        if cleanup_directory is not None and root_teardown_permitted:
            directory_outcome = cleanup_directory.finish_empty()
            self._record_shared_outcome(directory_outcome)
            issues.extend(self._adapt_shared_outcome(directory_outcome))
        elif cleanup_directory is not None:
            root_issues: list[repository_snapshot.CleanupIssue] = []
            try:
                cleanup_directory.descriptor.close_once()
            except OSError as error:
                root_issues.append(
                    repository_snapshot.CleanupIssue(
                        "root_descriptor_close_failed",
                        cleanup_directory.path,
                        error,
                    )
                )
            recovery_paths: tuple[Path, ...] = ()
            if not self._cleanup_unaddressable:
                recovery_paths = (cleanup_directory.path,)
                if not root_issues:
                    root_issues.append(
                        repository_snapshot.CleanupIssue(
                            "private_root_recovery_required",
                            cleanup_directory.path,
                        )
                    )
            root_outcome = repository_snapshot.finalize_arena_outcome(
                cleanup_directory.arena,
                repository_snapshot.CleanupOutcome(
                    disposition=(
                        repository_snapshot.CleanupDisposition.UNADDRESSABLE
                        if self._cleanup_unaddressable
                        else repository_snapshot.CleanupDisposition.RETAINED
                    ),
                    recovery_paths=recovery_paths,
                    issues=tuple(root_issues),
                    arena_identity=cleanup_directory.arena.identity,
                    recovery_anchor_identity=cleanup_directory.arena.anchor.identity,
                ),
            )
            self._record_shared_outcome(root_outcome)
            issues.extend(self._adapt_shared_outcome(root_outcome))
        elif root_fd is not None:
            try:
                os.close(root_fd)
            except OSError:
                issues.append(
                    CleanupIssue(
                        "root_descriptor_close_failed",
                        None,
                        None if self._cleanup_unaddressable else self._root_path,
                    )
                )
        anchor_owner = self._recovery_anchor_owner
        self._recovery_anchor_owner = None
        if anchor_owner is not None:
            try:
                anchor_owner.close_once()
            except OSError:
                issues.append(
                    CleanupIssue("recovery_anchor_descriptor_close_failed", None)
                )
        if (
            not self._cleanup_unaddressable
            and self._root_path is not None
            and self._root_identity is None
            and not any(issue.recovery_path == self._root_path for issue in issues)
        ):
            issues.append(
                CleanupIssue(
                    "private_root_credential_unverified",
                    None,
                    self._root_path,
                )
            )
        aggregate = self._cleanup_accumulator.snapshot()
        self._cleanup_unaddressable = aggregate.unaddressable
        self._cleanup_status = (
            "unaddressable"
            if self._cleanup_unaddressable
            else "recovery_required"
            if issues
            else "complete"
        )
        if self._cleanup_unaddressable:
            issues = [CleanupIssue(issue.code, issue.artifact_id) for issue in issues]
        return issues


__all__ = [
    "ArtifactCaptureError",
    "ArtifactCleanupError",
    "ArtifactRequest",
    "ArtifactSnapshotError",
    "ArtifactSnapshotSet",
    "ArtifactVerificationError",
    "CleanupIssue",
    "CleanupStatus",
    "FrozenArtifact",
]
