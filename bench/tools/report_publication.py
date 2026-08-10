#!/usr/bin/env python3
# Copyright (C) 2026 Zynum contributors
# SPDX-License-Identifier: LGPL-3.0-or-later

"""Fail-closed publication for one coherent generation of benchmark reports.

The implementation narrows every filesystem operation after traversal to a
single name relative to a held directory descriptor.  It deliberately does
not claim portable compare-and-swap semantics, crash atomicity, or atomic
multi-file visibility: readers may observe the deterministic replace sequence.
Any failed transaction that created a verified backup intentionally retains it
and reports its path; portable filesystems cannot prove that a restored public
name remains attached while deleting the last independent recovery name.
"""

from __future__ import annotations

import dataclasses
import ctypes
import errno
import hashlib
import importlib.util
import os
import secrets
import stat
import sys
from collections.abc import Sequence
from pathlib import Path


_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
_REPOSITORY_SNAPSHOT_PATH = _REPOSITORY_ROOT / "tools" / "repository_snapshot.py"
_REPOSITORY_SNAPSHOT_SPEC = importlib.util.spec_from_file_location(
    "_zynum_report_repository_snapshot", _REPOSITORY_SNAPSHOT_PATH
)
if _REPOSITORY_SNAPSHOT_SPEC is None or _REPOSITORY_SNAPSHOT_SPEC.loader is None:
    raise RuntimeError("unable to load the repository snapshot cleanup policy")
repository_snapshot = importlib.util.module_from_spec(_REPOSITORY_SNAPSHOT_SPEC)
sys.modules[_REPOSITORY_SNAPSHOT_SPEC.name] = repository_snapshot
_REPOSITORY_SNAPSHOT_SPEC.loader.exec_module(repository_snapshot)


@dataclasses.dataclass(frozen=True, slots=True)
class ReportOutput:
    """A fully materialized report destination and its immutable bytes."""

    path: Path
    contents: bytes


class ReportPublicationError(OSError):
    """Base class for publication failures with filesystem-style semantics."""


class RollbackIndeterminateError(ReportPublicationError):
    """Publication failed with intentionally retained or uncertain recovery."""

    def __init__(
        self,
        *args: object,
        recovery_paths: Sequence[Path] = (),
        candidate_paths: Sequence[Path] = (),
        public_candidate=repository_snapshot.PublicCandidate.ABSENT,
    ) -> None:
        super().__init__(*args)
        self.recovery_paths = tuple(recovery_paths)
        self.candidate_paths = tuple(candidate_paths)
        self.public_candidate = public_candidate


class TransactionCompleteCleanupError(ReportPublicationError):
    """The new generation is coherent but recovery-material cleanup failed."""

    def __init__(
        self,
        *args: object,
        recovery_paths: Sequence[Path] = (),
        candidate_paths: Sequence[Path] = (),
        public_candidate=repository_snapshot.PublicCandidate.ABSENT,
        cleanup_outcomes: Sequence[repository_snapshot.CleanupOutcome] = (),
    ) -> None:
        super().__init__(*args)
        self.recovery_paths = tuple(recovery_paths)
        self.candidate_paths = tuple(candidate_paths)
        self.public_candidate = public_candidate
        self.cleanup_outcomes = tuple(cleanup_outcomes)


@dataclasses.dataclass(frozen=True, slots=True)
class _FileIdentity:
    device: int
    inode: int


@dataclasses.dataclass(frozen=True, slots=True)
class _Artifact:
    identity: _FileIdentity
    size: int
    mode: int
    uid: int
    gid: int
    sha256: str
    xattrs: tuple[tuple[bytes, bytes], ...]


@dataclasses.dataclass(frozen=True, slots=True)
class _OutputSpec:
    path: Path
    parent_path: Path
    name: str
    contents: bytes


@dataclasses.dataclass(frozen=True, slots=True)
class _Parent:
    path: Path
    descriptor: int


@dataclasses.dataclass(slots=True)
class _CreatedDirectory:
    parent_owner: repository_snapshot.OwnedDescriptor
    name: str
    path: Path
    identity: _FileIdentity | None

    @property
    def parent_descriptor(self) -> int:
        return self.parent_owner.fileno()


@dataclasses.dataclass(slots=True)
class _ProbeFile:
    name: str
    identity: _FileIdentity | None = None
    cleanup_permitted: bool = False


@dataclasses.dataclass(slots=True)
class _CollisionProbe:
    parent: _Parent
    name: str
    descriptor_owner: repository_snapshot.OwnedDescriptor
    identity: _FileIdentity
    owner_uid: int
    files: list[_ProbeFile]

    @property
    def descriptor(self) -> int:
        return self.descriptor_owner.fileno()

    @property
    def path(self) -> Path:
        return self.parent.path / self.name


@dataclasses.dataclass(slots=True)
class _PreparedOutput:
    spec: _OutputSpec
    parent: _Parent
    original_descriptor: int | None = None
    original: _Artifact | None = None
    stage_name: str | None = None
    stage_identity: _FileIdentity | None = None
    stage: _Artifact | None = None
    stage_cleanup_prohibited: bool = False
    backup_name: str | None = None
    backup_identity: _FileIdentity | None = None
    backup_cleanup_artifact: _Artifact | None = None
    backup_cleanup_prohibited: bool = False
    backup: _Artifact | None = None
    committed: bool = False
    commit_indeterminate: bool = False
    retain_backup: bool = False
    retained_quarantine_paths: list[Path] = dataclasses.field(default_factory=list)
    candidate_paths: list[Path] = dataclasses.field(default_factory=list)
    public_candidates: list[repository_snapshot.PublicCandidate] = dataclasses.field(
        default_factory=list
    )


def _platform_support_available() -> bool:
    """Return whether fail-closed report publication APIs are available."""

    if os.name != "posix" or sys.platform not in {"darwin", "linux"}:
        return False
    if os.O_NOFOLLOW is None:
        return False
    if os.open not in os.supports_dir_fd:
        return False
    if os.stat not in os.supports_dir_fd:
        return False
    if os.mkdir not in os.supports_dir_fd:
        return False
    if os.unlink not in os.supports_dir_fd:
        return False
    if os.rmdir not in os.supports_dir_fd:
        return False
    if os.rename not in os.supports_dir_fd:
        return False
    if os.stat not in os.supports_follow_symlinks:
        return False
    return True


def _require_platform_support() -> None:
    if not _platform_support_available():
        raise ReportPublicationError(
            errno.ENOTSUP,
            "report publication requires POSIX no-follow descriptor-relative APIs",
        )


def _component_name(name: str, label: str) -> str:
    if (
        not name
        or name in {".", ".."}
        or "\x00" in name
        or os.sep in name
        or (os.altsep is not None and os.altsep in name)
    ):
        raise ValueError(f"{label} must be a nonempty single path component")
    return name


def _normalize_outputs(outputs: Sequence[ReportOutput]) -> tuple[_OutputSpec, ...]:
    materialized = tuple(outputs)
    if not materialized:
        raise ValueError("report publication requires at least one destination")

    normalized: list[_OutputSpec] = []
    seen: set[str] = set()
    for index, output in enumerate(materialized):
        if not isinstance(output, ReportOutput):
            raise TypeError(f"report output {index} must be a ReportOutput")
        if not isinstance(output.path, Path):
            raise TypeError(f"report output {index} destination must be a Path")
        if not isinstance(output.contents, bytes):
            raise TypeError(f"report output {index} contents must be immutable bytes")

        original_path = output.path
        name = _component_name(original_path.name, "report destination")
        for part in (
            original_path.parts[1:]
            if original_path.is_absolute()
            else original_path.parts
        ):
            _component_name(part, "report destination path")
        path = (
            original_path if original_path.is_absolute() else Path.cwd() / original_path
        )
        key = os.fspath(path)
        if key in seen:
            raise ValueError(f"duplicate report destination: {path}")
        seen.add(key)
        normalized.append(
            _OutputSpec(
                path=path,
                parent_path=path.parent,
                name=name,
                contents=bytes(output.contents),
            )
        )
    return tuple(sorted(normalized, key=lambda output: os.fspath(output.path)))


def _directory_flags() -> int:
    return os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW


def _file_flags() -> int:
    return os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK


def _identity(metadata: os.stat_result) -> _FileIdentity:
    return _FileIdentity(metadata.st_dev, metadata.st_ino)


def _same_identity(
    metadata: os.stat_result, expected: _FileIdentity, *, directory: bool = False
) -> bool:
    expected_type = stat.S_ISDIR if directory else stat.S_ISREG
    return expected_type(metadata.st_mode) and _identity(metadata) == expected


def _open_parent(path: Path, created: list[_CreatedDirectory]) -> _Parent:
    if not path.is_absolute():
        raise ValueError("report destination parent must be absolute")
    descriptor = os.open(path.anchor, _directory_flags())
    current = Path(path.anchor)
    try:
        for part in path.parts[1:]:
            part = _component_name(part, "report destination parent")
            current /= part
            try:
                child = os.open(part, _directory_flags(), dir_fd=descriptor)
            except FileNotFoundError:
                reserved_parent = repository_snapshot.OwnedDescriptor.take(
                    os.dup(descriptor), current.parent
                )
                parent_transferred = False
                try:
                    try:
                        os.mkdir(part, 0o755, dir_fd=descriptor)
                    except FileExistsError:
                        reserved_parent.close_once()
                        child = os.open(part, _directory_flags(), dir_fd=descriptor)
                    else:
                        record = _CreatedDirectory(reserved_parent, part, current, None)
                        created.append(record)
                        parent_transferred = True
                        metadata = os.stat(
                            part, dir_fd=descriptor, follow_symlinks=False
                        )
                        if not stat.S_ISDIR(metadata.st_mode):
                            raise ReportPublicationError(
                                errno.ENOTDIR,
                                f"created report parent is not a directory: {current}",
                            )
                        record.identity = _identity(metadata)
                        child = os.open(part, _directory_flags(), dir_fd=descriptor)
                        if not _same_identity(
                            os.fstat(child), record.identity, directory=True
                        ):
                            os.close(child)
                            raise ReportPublicationError(
                                errno.ESTALE,
                                "created report parent changed during traversal: "
                                f"{current}",
                            )
                finally:
                    if not parent_transferred and not reserved_parent.close_attempted:
                        reserved_parent.close_once()
            except OSError as exc:
                raise ReportPublicationError(
                    exc.errno or errno.EINVAL,
                    f"report destination parent is not a no-follow directory: {current}",
                ) from exc
            try:
                os.close(descriptor)
            except OSError:
                pass
            descriptor = child
        return _Parent(path, descriptor)
    except Exception:
        os.close(descriptor)
        raise


def _open_existing_parent(path: Path) -> int:
    descriptor = os.open(path.anchor, _directory_flags())
    try:
        for part in path.parts[1:]:
            child = os.open(
                _component_name(part, "report destination parent"),
                _directory_flags(),
                dir_fd=descriptor,
            )
            os.close(descriptor)
            descriptor = child
        return descriptor
    except Exception:
        os.close(descriptor)
        raise


def _verify_parent(parent: _Parent, stage: str) -> None:
    try:
        observed_descriptor = _open_existing_parent(parent.path)
    except OSError as exc:
        raise ReportPublicationError(
            exc.errno or errno.ESTALE,
            f"report destination parent changed {stage}: {parent.path}",
        ) from exc
    try:
        expected = os.fstat(parent.descriptor)
        observed = os.fstat(observed_descriptor)
    finally:
        os.close(observed_descriptor)
    if not (
        stat.S_ISDIR(expected.st_mode)
        and stat.S_ISDIR(observed.st_mode)
        and _identity(expected) == _identity(observed)
    ):
        raise ReportPublicationError(
            errno.ESTALE, f"report destination parent changed {stage}: {parent.path}"
        )


def _pread(descriptor: int, size: int, offset: int) -> bytes:
    return os.pread(descriptor, size, offset)


def _digest_descriptor(descriptor: int, size: int) -> str:
    digest = hashlib.sha256()
    offset = 0
    while offset < size:
        requested = min(1024 * 1024, size - offset)
        chunk = _pread(descriptor, requested, offset)
        if not chunk:
            raise ReportPublicationError(
                errno.EIO, "regular file ended before its frozen size while hashing"
            )
        if len(chunk) > requested:
            raise ReportPublicationError(
                errno.EIO, "regular file read exceeded its frozen hash bound"
            )
        digest.update(chunk)
        offset += len(chunk)
    return digest.hexdigest()


def _record_artifact(descriptor: int, label: str) -> _Artifact:
    before = os.fstat(descriptor)
    if not stat.S_ISREG(before.st_mode):
        raise ReportPublicationError(errno.EINVAL, f"{label} is not a regular file")
    digest = _digest_descriptor(descriptor, before.st_size)
    after = os.fstat(descriptor)
    stable_fields = (
        "st_dev",
        "st_ino",
        "st_size",
        "st_mode",
        "st_mtime_ns",
        "st_ctime_ns",
    )
    if any(getattr(before, field) != getattr(after, field) for field in stable_fields):
        raise ReportPublicationError(errno.ESTALE, f"{label} changed while inspected")
    return _Artifact(
        identity=_identity(after),
        size=after.st_size,
        mode=stat.S_IMODE(after.st_mode),
        uid=after.st_uid,
        gid=after.st_gid,
        sha256=digest,
        xattrs=_record_xattrs(descriptor),
    )


def _record_xattrs(descriptor: int) -> tuple[tuple[bytes, bytes], ...]:
    if sys.platform not in {"darwin", "linux"}:
        raise ReportPublicationError(
            errno.ENOTSUP,
            "cannot inspect report extended attributes",
        )
    library = ctypes.CDLL(None, use_errno=True)
    try:
        flistxattr = library.flistxattr
        fgetxattr = library.fgetxattr
    except AttributeError as exc:
        raise ReportPublicationError(
            errno.ENOTSUP,
            "cannot inspect report extended attributes",
        ) from exc
    flistxattr.restype = ctypes.c_ssize_t
    fgetxattr.restype = ctypes.c_ssize_t
    if sys.platform == "darwin":
        flistxattr.argtypes = [
            ctypes.c_int,
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_int,
        ]
        fgetxattr.argtypes = [
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_uint32,
            ctypes.c_int,
        ]
        size = flistxattr(descriptor, None, 0, 0)
    else:
        flistxattr.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_size_t]
        fgetxattr.argtypes = [
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_void_p,
            ctypes.c_size_t,
        ]
        size = flistxattr(descriptor, None, 0)
    if size < 0:
        error = ctypes.get_errno()
        raise ReportPublicationError(
            error or errno.EIO,
            "cannot inspect report extended attributes",
        )
    if size == 0:
        return ()
    if size > 1024 * 1024:
        raise ReportPublicationError(
            errno.E2BIG, "report extended-attribute name list is too large"
        )
    names_buffer = ctypes.create_string_buffer(size)
    if sys.platform == "darwin":
        observed_size = flistxattr(descriptor, names_buffer, size, 0)
    else:
        observed_size = flistxattr(descriptor, names_buffer, size)
    if observed_size != size:
        error = ctypes.get_errno()
        raise ReportPublicationError(
            error or errno.ESTALE,
            "report extended attributes changed while inspected",
        )

    result: list[tuple[bytes, bytes]] = []
    for name in sorted(filter(None, names_buffer.raw.split(b"\0"))):
        if sys.platform == "darwin":
            value_size = fgetxattr(descriptor, name, None, 0, 0, 0)
        else:
            value_size = fgetxattr(descriptor, name, None, 0)
        if value_size < 0 or value_size > 16 * 1024 * 1024:
            error = ctypes.get_errno()
            raise ReportPublicationError(
                error or errno.E2BIG,
                f"cannot inspect report extended attribute {name!r}",
            )
        value_buffer = ctypes.create_string_buffer(value_size)
        if sys.platform == "darwin":
            observed_value_size = fgetxattr(
                descriptor, name, value_buffer, value_size, 0, 0
            )
        else:
            observed_value_size = fgetxattr(descriptor, name, value_buffer, value_size)
        if observed_value_size != value_size:
            error = ctypes.get_errno()
            raise ReportPublicationError(
                error or errno.ESTALE,
                f"report extended attribute {name!r} changed while inspected",
            )
        result.append((name, value_buffer.raw[:value_size]))
    return tuple(result)


def _descriptor_has_acl(descriptor: int) -> bool:
    if sys.platform == "linux":
        # Linux POSIX ACLs are exposed by flistxattr as system.posix_acl_*.
        return False
    if sys.platform != "darwin":
        raise ReportPublicationError(
            errno.ENOTSUP, "cannot prove that existing report metadata has no ACL"
        )
    library = ctypes.CDLL(None, use_errno=True)
    try:
        acl_get_fd_np = library.acl_get_fd_np
        acl_get_entry = library.acl_get_entry
        acl_free = library.acl_free
    except AttributeError as exc:
        raise ReportPublicationError(
            errno.ENOTSUP, "cannot inspect existing report ACL"
        ) from exc
    acl_get_fd_np.argtypes = [ctypes.c_int, ctypes.c_int]
    acl_get_fd_np.restype = ctypes.c_void_p
    acl_get_entry.argtypes = [
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_void_p),
    ]
    acl_get_entry.restype = ctypes.c_int
    acl_free.argtypes = [ctypes.c_void_p]
    acl_free.restype = ctypes.c_int

    acl = acl_get_fd_np(descriptor, 0x00000100)
    if not acl:
        error = ctypes.get_errno()
        if error == errno.ENOENT:
            return False
        raise ReportPublicationError(
            error or errno.EIO, "cannot inspect existing report ACL"
        )
    try:
        entry = ctypes.c_void_p()
        result = acl_get_entry(acl, 0, ctypes.byref(entry))
        # Darwin's acl_get_entry(3) returns 0 for a retrieved entry and -1
        # otherwise; this differs from the POSIX.1e convention used on Linux.
        if result == 0:
            return True
        error = ctypes.get_errno()
        raise ReportPublicationError(
            error or errno.EIO, "cannot inspect existing report ACL entries"
        )
    finally:
        acl_free(acl)


def _descriptor_nonrestorable_flags(descriptor: int, metadata: os.stat_result) -> int:
    flags = getattr(metadata, "st_flags", None)
    if flags is not None:
        return flags
    if sys.platform != "linux":
        raise ReportPublicationError(
            errno.ENOTSUP, "cannot inspect existing report file flags"
        )
    import array
    import fcntl

    observed = array.array("I", [0])
    try:
        fcntl.ioctl(descriptor, 0x80086601, observed, True)
    except OSError as exc:
        if exc.errno in {errno.ENOTTY, errno.EOPNOTSUPP}:
            return 0
        raise ReportPublicationError(
            exc.errno or errno.EIO, "cannot inspect existing report file flags"
        ) from exc
    # Structural flags such as EXTENTS are recreated by the filesystem.  These
    # are the user-modifiable flags whose semantics a byte copy cannot restore.
    return observed[0] & 0x000780FF


def _current_owner() -> tuple[int, int]:
    return os.geteuid(), os.getegid()


def _validate_restorable_metadata(
    descriptor: int, artifact: _Artifact, label: str
) -> None:
    metadata = os.fstat(descriptor)
    if not stat.S_ISREG(metadata.st_mode):
        raise ReportPublicationError(errno.EINVAL, f"{label} is not a regular file")
    if metadata.st_nlink != 1:
        raise ReportPublicationError(
            errno.ENOTSUP,
            f"{label} has hard-link topology that report rollback cannot restore",
        )
    expected_uid, expected_gid = _current_owner()
    if metadata.st_uid != expected_uid or metadata.st_gid != expected_gid:
        raise ReportPublicationError(
            errno.ENOTSUP,
            f"{label} ownership cannot be exactly restored",
        )
    if _descriptor_nonrestorable_flags(descriptor, metadata):
        raise ReportPublicationError(
            errno.ENOTSUP, f"{label} has file flags that rollback cannot restore"
        )
    allowed_xattrs = {b"com.apple.provenance"} if sys.platform == "darwin" else set()
    unsupported_xattrs = [
        name for name, _value in artifact.xattrs if name not in allowed_xattrs
    ]
    if unsupported_xattrs:
        raise ReportPublicationError(
            errno.ENOTSUP,
            f"{label} has extended attributes that rollback cannot restore: "
            + ", ".join(repr(name) for name in unsupported_xattrs),
        )
    if _descriptor_has_acl(descriptor):
        raise ReportPublicationError(
            errno.ENOTSUP, f"{label} has an ACL that rollback cannot restore"
        )


def _open_regular_name(parent: _Parent, name: str, label: str) -> int:
    try:
        descriptor = os.open(
            _component_name(name, label), _file_flags(), dir_fd=parent.descriptor
        )
    except FileNotFoundError:
        raise
    except OSError as exc:
        raise ReportPublicationError(
            exc.errno or errno.EINVAL, f"{label} is not a safe regular file"
        ) from exc
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise ReportPublicationError(errno.EINVAL, f"{label} is not a regular file")
    except Exception:
        os.close(descriptor)
        raise
    return descriptor


def _verify_absent(parent: _Parent, name: str, label: str) -> None:
    try:
        os.stat(
            _component_name(name, label),
            dir_fd=parent.descriptor,
            follow_symlinks=False,
        )
    except FileNotFoundError:
        return
    raise ReportPublicationError(errno.EEXIST, f"{label} appeared before publication")


def _verify_artifact_name(
    parent: _Parent, name: str, expected: _Artifact, label: str
) -> None:
    descriptor = _open_regular_name(parent, name, label)
    try:
        observed = _record_artifact(descriptor, label)
    finally:
        os.close(descriptor)
    if observed != expected:
        raise ReportPublicationError(
            errno.ESTALE, f"{label} identity, mode, or digest changed"
        )


def _verify_restorable_artifact_name(
    parent: _Parent, name: str, expected: _Artifact, label: str
) -> None:
    descriptor = _open_regular_name(parent, name, label)
    try:
        observed = _record_artifact(descriptor, label)
        _validate_restorable_metadata(descriptor, observed, label)
    finally:
        os.close(descriptor)
    if observed != expected:
        raise ReportPublicationError(
            errno.ESTALE, f"{label} identity, mode, or digest changed"
        )


def _name_names_identity(parent: _Parent, name: str, expected: _FileIdentity) -> bool:
    try:
        metadata = os.stat(
            _component_name(name, "owned artifact"),
            dir_fd=parent.descriptor,
            follow_symlinks=False,
        )
    except (FileNotFoundError, OSError, ValueError):
        return False
    return _same_identity(metadata, expected)


def _name_names_artifact(parent: _Parent, name: str, expected: _Artifact) -> bool:
    try:
        _verify_artifact_name(parent, name, expected, "owned report artifact")
    except (FileNotFoundError, OSError, ValueError):
        return False
    return True


def _freeze_created_file_identity(descriptor: int) -> _FileIdentity:
    metadata = os.fstat(descriptor)
    if not stat.S_ISREG(metadata.st_mode):
        raise ReportPublicationError(
            errno.EINVAL, "created report sidecar is not a regular file"
        )
    return _identity(metadata)


def _observed_public_candidate(
    parent: _Parent,
    name: str,
    path: Path,
    candidate_paths: list[Path],
) -> repository_snapshot.PublicCandidate:
    try:
        os.stat(
            _component_name(name, "report cleanup candidate"),
            dir_fd=parent.descriptor,
            follow_symlinks=False,
        )
    except FileNotFoundError:
        return repository_snapshot.PublicCandidate.ABSENT
    except (OSError, ValueError):
        _record_recovery_path(candidate_paths, path)
        return repository_snapshot.PublicCandidate.UNKNOWN
    _record_recovery_path(candidate_paths, path)
    return repository_snapshot.PublicCandidate.PRESENT


def _observe_recovery_candidate(
    parent: _Parent,
    name: str,
    path: Path,
    recovery_paths: list[Path],
) -> tuple[bool, Exception | None]:
    """Record present or uninspectable recovery material, never proven absence."""

    try:
        os.stat(
            _component_name(name, "report recovery material"),
            dir_fd=parent.descriptor,
            follow_symlinks=False,
        )
    except FileNotFoundError:
        return False, None
    except (OSError, ValueError) as exc:
        _record_recovery_path(recovery_paths, path)
        return True, exc
    _record_recovery_path(recovery_paths, path)
    return True, None


def _open_unique(parent: _Parent, suffix: str) -> tuple[str, int, _FileIdentity]:
    flags = os.O_RDWR | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW
    for _ in range(128):
        name = _component_name(
            f".report-publish-{secrets.token_hex(16)}.{suffix}",
            "report publication sidecar",
        )
        try:
            descriptor = os.open(name, flags, 0o600, dir_fd=parent.descriptor)
        except FileExistsError:
            continue
        try:
            identity = _freeze_created_file_identity(descriptor)
        except Exception as exc:
            try:
                os.close(descriptor)
            except OSError:
                pass
            candidate_paths: list[Path] = []
            public_candidate = _observed_public_candidate(
                parent, name, parent.path / name, candidate_paths
            )
            if candidate_paths:
                raise RollbackIndeterminateError(
                    errno.EIO,
                    "rollback_indeterminate: created report sidecar identity could not "
                    "be frozen; public path is only an unproved cleanup candidate: "
                    f"{parent.path / name}",
                    candidate_paths=candidate_paths,
                    public_candidate=public_candidate,
                ) from exc
            raise
        return name, descriptor, identity
    raise ReportPublicationError(
        errno.EEXIST, f"unable to allocate unique report publication {suffix}"
    )


def _create_collision_probe(parent: _Parent) -> _CollisionProbe:
    flags = os.O_RDONLY | os.O_CLOEXEC | os.O_DIRECTORY | os.O_NOFOLLOW
    for _ in range(128):
        name = _component_name(
            f".report-publish-{secrets.token_hex(16)}.probe",
            "report collision probe",
        )
        try:
            os.mkdir(name, 0o700, dir_fd=parent.descriptor)
        except FileExistsError:
            continue
        path = parent.path / name
        owner: repository_snapshot.OwnedDescriptor | None = None
        try:
            observed = os.stat(name, dir_fd=parent.descriptor, follow_symlinks=False)
            descriptor = os.open(name, flags, dir_fd=parent.descriptor)
            owner = repository_snapshot.OwnedDescriptor.take(descriptor, path)
            opened = os.fstat(owner.fileno())
            identity = _identity(observed)
            effective_uid = os.geteuid()
            if (
                not stat.S_ISDIR(observed.st_mode)
                or not stat.S_ISDIR(opened.st_mode)
                or not _same_identity(opened, identity, directory=True)
                or observed.st_dev != os.fstat(parent.descriptor).st_dev
                or observed.st_uid != effective_uid
                or opened.st_uid != effective_uid
                or stat.S_IMODE(observed.st_mode) & 0o077
                or stat.S_IMODE(opened.st_mode) & 0o077
            ):
                raise ReportPublicationError(
                    errno.EPERM, "report collision probe initial credential is unsafe"
                )
            os.fchmod(owner.fileno(), 0o700)
            configured_path = os.stat(
                name, dir_fd=parent.descriptor, follow_symlinks=False
            )
            configured_fd = os.fstat(owner.fileno())
            if (
                not _same_identity(configured_path, identity, directory=True)
                or not _same_identity(configured_fd, identity, directory=True)
                or configured_path.st_uid != effective_uid
                or configured_fd.st_uid != effective_uid
                or stat.S_IMODE(configured_path.st_mode) != 0o700
                or stat.S_IMODE(configured_fd.st_mode) != 0o700
            ):
                raise ReportPublicationError(
                    errno.EPERM,
                    "report collision probe configured credential is unsafe",
                )
            os.fsync(owner.fileno())
            os.fsync(parent.descriptor)
            return _CollisionProbe(parent, name, owner, identity, effective_uid, [])
        except BaseException as exc:
            close_error: OSError | None = None
            if owner is not None:
                try:
                    owner.close_once()
                except OSError as close_exc:
                    close_error = close_exc
            candidate_paths: list[Path] = []
            public_candidate = _observed_public_candidate(
                parent, name, path, candidate_paths
            )
            if candidate_paths:
                close_detail = (
                    f"; descriptor close became uncertain: {close_error}"
                    if close_error is not None
                    else ""
                )
                raise RollbackIndeterminateError(
                    errno.EIO,
                    "rollback_indeterminate: report collision probe setup failed "
                    f"({exc}){close_detail}; public path is only an unproved "
                    f"cleanup candidate: {path}",
                    candidate_paths=candidate_paths,
                    public_candidate=public_candidate,
                ) from (close_error or exc)
            raise
    raise ReportPublicationError(
        errno.EEXIST, "unable to allocate unique report collision probe"
    )


def _create_probe_file(probe: _CollisionProbe, name: str) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW
    descriptor = os.open(
        _component_name(name, "report collision probe leaf"),
        flags,
        0o600,
        dir_fd=probe.descriptor,
    )
    path = probe.path / name
    owner = repository_snapshot.OwnedDescriptor.take(descriptor, path)
    file = _ProbeFile(name)
    probe.files.append(file)
    try:
        metadata = os.fstat(owner.fileno())
        if not stat.S_ISREG(metadata.st_mode):
            raise ReportPublicationError(
                errno.EINVAL, "created report collision probe leaf is not regular"
            )
        file.identity = _identity(metadata)
    except BaseException:
        try:
            owner.close_once()
        except OSError as close_exc:
            raise ReportPublicationError(
                errno.EIO,
                f"report collision probe leaf descriptor close became uncertain: {path}",
            ) from close_exc
        raise
    try:
        owner.close_once()
    except OSError as exc:
        raise ReportPublicationError(
            errno.EIO,
            f"report collision probe leaf descriptor close became uncertain: {path}",
        ) from exc
    file.cleanup_permitted = True


def _remove_probe_file(probe: _CollisionProbe, file: _ProbeFile) -> None:
    """Remove a leaf from the held random mode-0700 collision-probe directory."""

    if not file.cleanup_permitted or file.identity is None:
        raise ReportPublicationError(
            errno.EIO,
            f"report collision probe leaf identity or close is uncertain; "
            f"preserved as {probe.path / file.name}",
        )
    metadata = os.stat(file.name, dir_fd=probe.descriptor, follow_symlinks=False)
    if not _same_identity(metadata, file.identity):
        raise ReportPublicationError(
            errno.ESTALE,
            f"report collision probe leaf changed; preserved as "
            f"{probe.path / file.name}",
        )
    os.unlink(file.name, dir_fd=probe.descriptor)


def _cleanup_collision_probe(
    probe: _CollisionProbe,
) -> tuple[
    list[str],
    tuple[Path, ...],
    tuple[Path, ...],
    repository_snapshot.PublicCandidate,
]:
    issues: list[str] = []
    recovery_paths: list[Path] = []
    candidate_paths: list[Path] = []
    public_candidates: list[repository_snapshot.PublicCandidate] = []
    for file in reversed(probe.files):
        try:
            _remove_probe_file(probe, file)
        except Exception as exc:
            issues.append(f"probe leaf cleanup failed for {file.name}: {exc}")

    fsync_failed = False
    try:
        os.fsync(probe.descriptor)
    except OSError as exc:
        fsync_failed = True
        issues.append(f"probe directory fsync failed for {probe.path}: {exc}")

    try:
        probe.descriptor_owner.close_once()
    except OSError as exc:
        issues.append(f"probe descriptor cleanup failed for {probe.path}: {exc}")
        public_candidates.append(
            _observed_public_candidate(
                probe.parent, probe.name, probe.path, candidate_paths
            )
        )
        return (
            issues,
            tuple(recovery_paths),
            tuple(candidate_paths),
            _aggregate_public_candidate(public_candidates),
        )
    if fsync_failed:
        public_candidates.append(
            _observed_public_candidate(
                probe.parent, probe.name, probe.path, candidate_paths
            )
        )
        return (
            issues,
            tuple(recovery_paths),
            tuple(candidate_paths),
            _aggregate_public_candidate(public_candidates),
        )

    source = _directory_anchor(probe.parent)
    try:
        quarantine = _create_directory_cleanup_quarantine(
            source, probe.name, ".quarantine"
        )
    except repository_snapshot.CleanupFailure as exc:
        _record_cleanup_outcome(
            "collision probe cleanup",
            None,
            exc.outcome,
            issues,
            recovery_paths,
            candidate_paths,
            public_candidates,
        )
        return (
            issues,
            tuple(recovery_paths),
            tuple(candidate_paths),
            _aggregate_public_candidate(public_candidates),
        )

    def verify_empty_claim(
        descriptor: int, metadata: os.stat_result
    ) -> repository_snapshot.ClaimVerification:
        try:
            if (
                not _same_identity(metadata, probe.identity, directory=True)
                or metadata.st_uid != probe.owner_uid
                or stat.S_IMODE(metadata.st_mode) != 0o700
            ):
                return repository_snapshot.ClaimVerification.FOREIGN
            if os.listdir(descriptor):
                return repository_snapshot.ClaimVerification.FOREIGN
        except OSError:
            return repository_snapshot.ClaimVerification.UNKNOWN
        return repository_snapshot.ClaimVerification.MATCH

    try:
        quarantine.claim()
        if quarantine.phase is not repository_snapshot.CleanupPhase.ABSENT:
            verification = quarantine.verify_claimed(verify_empty_claim)
            if verification is repository_snapshot.ClaimVerification.MATCH:
                quarantine.remove_verified_claim()
    except repository_snapshot.CleanupFailure:
        pass
    outcome = quarantine.finish(expect_public_absent=True)
    _record_cleanup_outcome(
        "collision probe cleanup",
        quarantine,
        outcome,
        issues,
        recovery_paths,
        candidate_paths,
        public_candidates,
    )
    return (
        issues,
        tuple(recovery_paths),
        tuple(candidate_paths),
        _aggregate_public_candidate(public_candidates),
    )


def _preflight_canonical_collisions(items: Sequence[_PreparedOutput]) -> None:
    groups: dict[_FileIdentity, list[_PreparedOutput]] = {}
    for item in items:
        metadata = os.fstat(item.parent.descriptor)
        if not stat.S_ISDIR(metadata.st_mode):
            raise ReportPublicationError(
                errno.ENOTDIR,
                f"report destination parent is not a directory: {item.parent.path}",
            )
        groups.setdefault(_identity(metadata), []).append(item)

    for identity in sorted(groups, key=lambda value: (value.device, value.inode)):
        group = groups[identity]
        probe = _create_collision_probe(group[0].parent)
        failure: Exception | None = None
        try:
            for item in group:
                try:
                    _create_probe_file(probe, item.spec.name)
                except FileExistsError as exc:
                    raise ValueError(
                        "duplicate or canonical report destination collision in "
                        f"{item.parent.path}: {item.spec.name!r}"
                    ) from exc
        except Exception as exc:
            failure = exc

        issues, recovery_paths, candidate_paths, public_candidate = (
            _cleanup_collision_probe(probe)
        )
        if issues:
            if recovery_paths or candidate_paths:
                candidate_detail = (
                    "; unproved candidate paths: "
                    + ", ".join(os.fspath(path) for path in candidate_paths)
                    if candidate_paths
                    else ""
                )
                raise RollbackIndeterminateError(
                    errno.EIO,
                    "rollback_indeterminate: report collision probe cleanup failed; "
                    f"recovery material preserved as {probe.path}: "
                    + "; ".join(issues)
                    + candidate_detail,
                    recovery_paths=recovery_paths,
                    candidate_paths=candidate_paths,
                    public_candidate=public_candidate,
                ) from failure
            if failure is None:
                raise ReportPublicationError(
                    errno.EIO,
                    "report collision probe cleanup failed: " + "; ".join(issues),
                )
        if failure is not None:
            raise failure


def _write_all(descriptor: int, contents: bytes) -> None:
    offset = 0
    while offset < len(contents):
        written = os.write(descriptor, contents[offset:])
        if written == 0:
            raise ReportPublicationError(errno.EIO, "short report sidecar write")
        offset += written


def _copy_all(source: int, destination: int, size: int) -> None:
    offset = 0
    while offset < size:
        requested = min(1024 * 1024, size - offset)
        chunk = _pread(source, requested, offset)
        if not chunk:
            raise ReportPublicationError(
                errno.EIO, "original report ended before its frozen backup size"
            )
        if len(chunk) > requested:
            raise ReportPublicationError(
                errno.EIO, "original report read exceeded its frozen backup bound"
            )
        _write_all(destination, chunk)
        offset += len(chunk)


def _fsync_file(descriptor: int) -> None:
    os.fsync(descriptor)


def _fsync_directory(parent: _Parent) -> None:
    os.fsync(parent.descriptor)


def _set_backup_ownership(descriptor: int, uid: int, gid: int) -> None:
    os.fchown(descriptor, uid, gid)


def _prepare_stage(item: _PreparedOutput) -> None:
    name, descriptor, identity = _open_unique(item.parent, "stage")
    item.stage_name = name
    item.stage_identity = identity
    owner = repository_snapshot.OwnedDescriptor.take(
        descriptor, item.parent.path / name
    )
    try:
        _write_all(owner.fileno(), item.spec.contents)
        _fsync_file(owner.fileno())
        os.fchmod(owner.fileno(), 0o644)
        _fsync_file(owner.fileno())
        item.stage = _record_artifact(owner.fileno(), "prepared report stage")
    finally:
        try:
            owner.close_once()
        except OSError as exc:
            item.stage_cleanup_prohibited = True
            public_candidate = _observed_public_candidate(
                item.parent,
                name,
                item.parent.path / name,
                item.candidate_paths,
            )
            item.public_candidates.append(public_candidate)
            raise RollbackIndeterminateError(
                errno.EIO,
                "rollback_indeterminate: prepared report stage descriptor close "
                "became uncertain; public stage is only an unproved cleanup "
                f"candidate: {item.parent.path / name}",
                candidate_paths=tuple(item.candidate_paths),
                public_candidate=public_candidate,
            ) from exc


def _prepare_backup(item: _PreparedOutput) -> None:
    assert item.original_descriptor is not None
    assert item.original is not None
    _verify_restorable_artifact_name(
        item.parent, item.spec.name, item.original, "original report destination"
    )
    name, descriptor, identity = _open_unique(item.parent, "backup")
    item.backup_name = name
    item.backup_identity = identity
    owner = repository_snapshot.OwnedDescriptor.take(
        descriptor, item.parent.path / name
    )
    try:
        os.fchmod(owner.fileno(), 0o600)
        _copy_all(item.original_descriptor, owner.fileno(), item.original.size)
        _fsync_file(owner.fileno())
        private_backup = _record_artifact(
            owner.fileno(), "private report recovery backup"
        )
        if (
            private_backup.identity == item.original.identity
            or private_backup.size != item.original.size
            or private_backup.mode != 0o600
            or private_backup.sha256 != item.original.sha256
            or private_backup.xattrs != item.original.xattrs
        ):
            raise ReportPublicationError(
                errno.EIO,
                "private report recovery backup is not an independent exact byte copy",
            )
        item.backup_cleanup_artifact = private_backup
        _set_backup_ownership(owner.fileno(), item.original.uid, item.original.gid)
        item.backup_cleanup_artifact = _record_artifact(
            owner.fileno(), "private report recovery backup after ownership"
        )
        owned_private_backup = item.backup_cleanup_artifact
        if (
            owned_private_backup.size != item.original.size
            or owned_private_backup.mode != 0o600
            or owned_private_backup.uid != item.original.uid
            or owned_private_backup.gid != item.original.gid
            or owned_private_backup.sha256 != item.original.sha256
            or owned_private_backup.xattrs != item.original.xattrs
        ):
            raise ReportPublicationError(
                errno.EIO,
                "private report recovery backup ownership transition changed bytes",
            )
        item.backup_cleanup_prohibited = True
        item.backup_cleanup_artifact = None
        os.fchmod(owner.fileno(), item.original.mode)
        _fsync_file(owner.fileno())
        backup = _record_artifact(owner.fileno(), "report recovery backup")
        if (
            backup.identity == item.original.identity
            or backup.size != item.original.size
            or backup.mode != item.original.mode
            or backup.uid != item.original.uid
            or backup.gid != item.original.gid
            or backup.sha256 != item.original.sha256
            or backup.xattrs != item.original.xattrs
        ):
            raise ReportPublicationError(
                errno.EIO, "report recovery backup is not an independent exact copy"
            )
        item.backup = backup
        item.backup_cleanup_prohibited = False
    finally:
        try:
            owner.close_once()
        except OSError as exc:
            item.backup_cleanup_prohibited = True
            item.backup_cleanup_artifact = None
            _record_recovery_path(
                item.retained_quarantine_paths, item.parent.path / name
            )
            raise RollbackIndeterminateError(
                errno.EIO,
                "rollback_indeterminate: report recovery backup descriptor close "
                f"became uncertain; backup retained as {item.parent.path / name}",
                recovery_paths=(item.parent.path / name,),
            ) from exc


def _replace_name(parent: _Parent, source: str, destination: str) -> None:
    os.replace(
        _component_name(source, "report replace source"),
        _component_name(destination, "report replace destination"),
        src_dir_fd=parent.descriptor,
        dst_dir_fd=parent.descriptor,
    )


def _verify_item_precommit(item: _PreparedOutput) -> None:
    assert item.stage_name is not None and item.stage is not None
    _verify_artifact_name(item.parent, item.stage_name, item.stage, "report stage")
    if item.original is None:
        _verify_absent(item.parent, item.spec.name, "report destination")
    else:
        assert item.backup_name is not None and item.backup is not None
        _verify_restorable_artifact_name(
            item.parent, item.spec.name, item.original, "original report destination"
        )
        _verify_artifact_name(
            item.parent, item.backup_name, item.backup, "report recovery backup"
        )


def _verify_item_published(item: _PreparedOutput, stage: str) -> None:
    assert item.stage is not None
    _verify_artifact_name(
        item.parent, item.spec.name, item.stage, f"published report destination {stage}"
    )
    if item.backup is not None:
        assert item.backup_name is not None
        _verify_artifact_name(
            item.parent,
            item.backup_name,
            item.backup,
            f"report recovery backup {stage}",
        )


def _unique_parents(items: Sequence[_PreparedOutput]) -> tuple[_Parent, ...]:
    parents = {os.fspath(item.parent.path): item.parent for item in items}
    return tuple(parents[key] for key in sorted(parents))


def _directory_anchor(parent: _Parent):
    return repository_snapshot.DirectoryAnchor(parent.descriptor, parent.path)


def _create_directory_cleanup_quarantine(
    source: repository_snapshot.DirectoryAnchor,
    public_name: str,
    suffix: str,
):
    public_path = source.path / public_name
    try:
        arena = repository_snapshot.CleanupArena.open(source)
    except repository_snapshot.CleanupFailure as exc:
        outcome = exc.outcome
        issues = list(outcome.issues)
        candidate_paths = list(outcome.candidate_paths)
        try:
            os.stat(
                public_name,
                dir_fd=source.descriptor,
                follow_symlinks=False,
            )
        except FileNotFoundError:
            public_candidate = repository_snapshot.PublicCandidate.ABSENT
        except OSError as inspect_exc:
            issues.append(
                repository_snapshot.CleanupIssue(
                    "cleanup_public_absence_uninspectable",
                    public_path,
                    inspect_exc,
                )
            )
            candidate_paths.append(public_path)
            public_candidate = repository_snapshot.PublicCandidate.UNKNOWN
        else:
            candidate_paths.append(public_path)
            public_candidate = repository_snapshot.PublicCandidate.PRESENT
        raise repository_snapshot.CleanupFailure(
            repository_snapshot.CleanupOutcome(
                outcome.disposition,
                outcome.recovery_paths,
                tuple(issues),
                tuple(sorted(set(candidate_paths), key=os.fspath)),
                outcome.arena_binding,
                public_candidate,
                outcome.arena_identity,
                outcome.recovery_anchor_identity,
            )
        ) from exc
    creation_error: BaseException | None = None
    quarantine = None
    try:
        quarantine = repository_snapshot.CleanupQuarantine.create(
            source,
            public_name,
            quarantine_prefix=".report-publish-",
            quarantine_suffix=suffix,
            arena=arena,
            claim_kind=repository_snapshot.ClaimKind.DIRECTORY,
        )
    except BaseException as exc:
        creation_error = exc
    if creation_error is not None:
        if isinstance(creation_error, repository_snapshot.CleanupFailure):
            outcome = creation_error.outcome
        else:
            candidate_path = source.path / public_name
            outcome = repository_snapshot.CleanupOutcome(
                repository_snapshot.CleanupDisposition.UNADDRESSABLE,
                (),
                (
                    repository_snapshot.CleanupIssue(
                        "cleanup_directory_quarantine_create_failed",
                        candidate_path,
                        creation_error,
                    ),
                ),
                (candidate_path,),
                repository_snapshot.ArenaBinding.UNKNOWN,
                repository_snapshot.PublicCandidate.UNKNOWN,
            )
        final_outcome = repository_snapshot.finalize_arena_outcome(arena, outcome)
        raise repository_snapshot.CleanupFailure(final_outcome) from creation_error
    assert quarantine is not None
    handoff_outcome = repository_snapshot.finalize_arena_outcome(
        arena,
        repository_snapshot.CleanupOutcome(
            repository_snapshot.CleanupDisposition.REMOVED,
            (),
            (),
        ),
    )
    if (
        handoff_outcome.disposition
        is not repository_snapshot.CleanupDisposition.REMOVED
    ):
        outcome = quarantine.finish(expect_public_absent=False)
        unaddressable = (
            handoff_outcome.disposition
            is repository_snapshot.CleanupDisposition.UNADDRESSABLE
            or outcome.disposition
            is repository_snapshot.CleanupDisposition.UNADDRESSABLE
        )
        raise repository_snapshot.CleanupFailure(
            repository_snapshot.CleanupOutcome(
                (
                    repository_snapshot.CleanupDisposition.UNADDRESSABLE
                    if unaddressable
                    else repository_snapshot.CleanupDisposition.RETAINED
                ),
                () if unaddressable else outcome.recovery_paths,
                (*handoff_outcome.issues, *outcome.issues),
                tuple(
                    sorted(
                        {
                            *handoff_outcome.candidate_paths,
                            *outcome.candidate_paths,
                        },
                        key=os.fspath,
                    )
                ),
                (
                    handoff_outcome.arena_binding
                    if handoff_outcome.arena_binding
                    is not repository_snapshot.ArenaBinding.BOUND
                    else outcome.arena_binding
                ),
                _aggregate_public_candidate(
                    (
                        handoff_outcome.public_candidate,
                        outcome.public_candidate,
                    )
                ),
                outcome.arena_identity or handoff_outcome.arena_identity,
                (
                    outcome.recovery_anchor_identity
                    or handoff_outcome.recovery_anchor_identity
                ),
            )
        )
    return quarantine


def _create_rollback_quarantine(parent: _Parent, public_name: str):
    return repository_snapshot.CleanupQuarantine.create(
        _directory_anchor(parent),
        public_name,
        quarantine_prefix=".report-publish-",
        quarantine_suffix=".quarantine",
    )


def _normalized_cleanup_recovery_paths(quarantine, outcome) -> tuple[Path, ...]:
    paths = set(outcome.recovery_paths)
    if quarantine is not None and quarantine.claimed_path in paths:
        paths.discard(quarantine.quarantine_path)
    return tuple(sorted(paths, key=os.fspath))


def _record_cleanup_outcome(
    label: str,
    quarantine,
    outcome,
    issues: list[str],
    recovery_paths: list[Path],
    candidate_paths: list[Path] | None = None,
    public_candidates: list[repository_snapshot.PublicCandidate] | None = None,
    accumulator: repository_snapshot.CleanupAccumulator | None = None,
) -> None:
    messages = {
        "cleanup_claim_destination_fsync_failed": "claimed-name directory fsync failed",
        "cleanup_claim_source_fsync_failed": "public-parent claim fsync failed",
        "cleanup_public_name_reappeared": "public name reappeared after the claim",
        "cleanup_public_absence_uninspectable": "public-name inspection failed",
        "cleanup_claimed_foreign": "public name contained foreign bytes",
        "cleanup_claimed_unknown": "claimed-name inspection failed",
        "cleanup_claimed_descriptor_close_uncertain": (
            "claimed descriptor cleanup failed"
        ),
        "cleanup_quarantine_descriptor_close_uncertain": (
            "quarantine descriptor cleanup failed"
        ),
        "cleanup_quarantine_teardown_failed": "quarantine cleanup failed",
        "cleanup_recovery_anchor_rebound": "stable recovery anchor was rebound",
        "cleanup_recovery_anchor_uninspectable": (
            "stable recovery anchor could not be inspected"
        ),
        "cleanup_recovery_anchor_not_rename_protected": (
            "recovery anchor is not rename-protected"
        ),
    }
    for issue in outcome.issues:
        detail = f": {issue.error}" if issue.error is not None else ""
        issues.append(
            f"{label} {messages.get(issue.code, issue.code)} at {issue.path}{detail}"
        )
    normalized_paths = set(_normalized_cleanup_recovery_paths(quarantine, outcome))
    if accumulator is not None:
        accumulator.record(
            repository_snapshot.CleanupOutcome(
                outcome.disposition,
                tuple(normalized_paths),
                outcome.issues,
                outcome.candidate_paths,
                outcome.arena_binding,
                outcome.public_candidate,
                outcome.arena_identity,
                outcome.recovery_anchor_identity,
            )
        )
    if quarantine is not None and quarantine.claimed_path in recovery_paths:
        normalized_paths.discard(quarantine.quarantine_path)
    if accumulator is None:
        for path in sorted(normalized_paths, key=os.fspath):
            _record_recovery_path(recovery_paths, path)
    if candidate_paths is not None:
        for path in outcome.candidate_paths:
            _record_recovery_path(candidate_paths, path)
            issues.append(
                f"{label} candidate path is not proved recovery material: {path}"
            )
    if public_candidates is not None:
        public_candidates.append(outcome.public_candidate)
    if quarantine is not None and quarantine.claimed_path in recovery_paths:
        while quarantine.quarantine_path in recovery_paths:
            recovery_paths.remove(quarantine.quarantine_path)


def _claim_destination(quarantine, destination: str) -> None:
    if destination != quarantine.public_name:
        raise ValueError("rollback claim source changed")
    quarantine.claim()


def _before_restore_original(_item: _PreparedOutput) -> None:
    """Narrow deterministic race-injection boundary used by rollback tests."""


def _before_retain_failure_backup(_item: _PreparedOutput) -> None:
    """Race-injection boundary at the former failure-path backup cleanup."""


def _link_backup_if_absent(item: _PreparedOutput) -> None:
    assert item.backup_name is not None
    os.link(
        item.backup_name,
        item.spec.name,
        src_dir_fd=item.parent.descriptor,
        dst_dir_fd=item.parent.descriptor,
        follow_symlinks=False,
    )


def _record_recovery_path(paths: list[Path], path: Path) -> None:
    if path not in paths:
        paths.append(path)


def _aggregate_public_candidate(
    values: Sequence[repository_snapshot.PublicCandidate],
) -> repository_snapshot.PublicCandidate:
    if repository_snapshot.PublicCandidate.UNKNOWN in values:
        return repository_snapshot.PublicCandidate.UNKNOWN
    if repository_snapshot.PublicCandidate.PRESENT in values:
        return repository_snapshot.PublicCandidate.PRESENT
    return repository_snapshot.PublicCandidate.ABSENT


def _cleanup_rollback_quarantine(
    quarantine,
    issues: list[str],
    recovery_paths: list[Path],
    candidate_paths: list[Path],
    public_candidates: list[repository_snapshot.PublicCandidate],
    accumulator: repository_snapshot.CleanupAccumulator | None = None,
    *,
    expect_public_absent: bool,
):
    outcome = quarantine.finish(expect_public_absent=expect_public_absent)
    _record_cleanup_outcome(
        "report cleanup",
        quarantine,
        outcome,
        issues,
        recovery_paths,
        candidate_paths,
        public_candidates,
        accumulator,
    )
    return outcome


def _before_public_sidecar_claim(_parent: _Parent, _name: str, _label: str) -> None:
    """Narrow race-injection boundary immediately before a public-name claim."""


def _after_public_sidecar_claim(_parent: _Parent, _name: str, _label: str) -> None:
    """Narrow race-injection boundary after a successful public-name claim."""


def _claim_and_remove_public_sidecar(
    parent: _Parent,
    name: str,
    identity: _FileIdentity,
    label: str,
    issues: list[str],
    recovery_paths: list[Path],
    candidate_paths: list[Path],
    public_candidates: list[repository_snapshot.PublicCandidate],
    accumulator: repository_snapshot.CleanupAccumulator | None = None,
    *,
    artifact: _Artifact | None = None,
    final_outcomes: list[repository_snapshot.CleanupOutcome] | None = None,
) -> bool | None:
    """Claim a public sidecar into a private quarantine before inspecting it."""

    try:
        quarantine = _create_rollback_quarantine(parent, name)
    except repository_snapshot.CleanupFailure as exc:
        issues.append(f"{label} quarantine creation failed: {exc}")
        _record_cleanup_outcome(
            label,
            None,
            exc.outcome,
            issues,
            recovery_paths,
            candidate_paths,
            public_candidates,
            accumulator,
        )
        if final_outcomes is not None:
            final_outcomes.append(exc.outcome)
        return False

    try:
        _before_public_sidecar_claim(parent, name, label)
        _claim_destination(quarantine, name)
    except repository_snapshot.CleanupFailure:
        outcome = _cleanup_rollback_quarantine(
            quarantine,
            issues,
            recovery_paths,
            candidate_paths,
            public_candidates,
            accumulator,
            expect_public_absent=True,
        )
        if final_outcomes is not None:
            final_outcomes.append(outcome)
        return None

    if quarantine.phase is repository_snapshot.CleanupPhase.ABSENT:
        outcome = _cleanup_rollback_quarantine(
            quarantine,
            issues,
            recovery_paths,
            candidate_paths,
            public_candidates,
            accumulator,
            expect_public_absent=True,
        )
        if final_outcomes is not None:
            final_outcomes.append(outcome)
        return False

    try:
        _after_public_sidecar_claim(parent, name, label)
    except Exception as exc:
        issues.append(f"{label} post-claim observation hook failed: {exc}")

    def verify_claimed(descriptor, _metadata):
        if artifact is not None:
            try:
                observed = _record_artifact(descriptor, label)
            except Exception:
                return repository_snapshot.ClaimVerification.UNKNOWN
            return (
                repository_snapshot.ClaimVerification.MATCH
                if observed == artifact
                else repository_snapshot.ClaimVerification.FOREIGN
            )
        observed = os.fstat(descriptor)
        return (
            repository_snapshot.ClaimVerification.MATCH
            if _same_identity(observed, identity)
            else repository_snapshot.ClaimVerification.FOREIGN
        )

    try:
        verification = quarantine.verify_claimed(verify_claimed)
        if verification is repository_snapshot.ClaimVerification.MATCH:
            quarantine.remove_verified_claim()
    except repository_snapshot.CleanupFailure:
        pass

    outcome = _cleanup_rollback_quarantine(
        quarantine,
        issues,
        recovery_paths,
        candidate_paths,
        public_candidates,
        accumulator,
        expect_public_absent=True,
    )
    if final_outcomes is not None:
        final_outcomes.append(outcome)
    if outcome.disposition is repository_snapshot.CleanupDisposition.REMOVED:
        return True
    if outcome.disposition is repository_snapshot.CleanupDisposition.ABSENT:
        return False
    return None


def _rollback_committed_item(
    item: _PreparedOutput,
    issues: list[str],
    accumulator: repository_snapshot.CleanupAccumulator,
) -> None:
    assert item.stage is not None
    try:
        quarantine = _create_rollback_quarantine(item.parent, item.spec.name)
    except repository_snapshot.CleanupFailure as exc:
        _record_cleanup_outcome(
            f"rollback destination for {item.spec.path}",
            None,
            exc.outcome,
            issues,
            item.retained_quarantine_paths,
            item.candidate_paths,
            item.public_candidates,
            accumulator,
        )
        item.retain_backup = item.backup is not None
        issues.append(
            f"rollback quarantine creation failed for {item.spec.path}: {exc}"
        )
        return

    try:
        _claim_destination(quarantine, item.spec.name)
    except repository_snapshot.CleanupFailure:
        item.retain_backup = item.backup is not None
        _cleanup_rollback_quarantine(
            quarantine,
            issues,
            item.retained_quarantine_paths,
            item.candidate_paths,
            item.public_candidates,
            accumulator,
            expect_public_absent=False,
        )
        return

    if quarantine.phase is repository_snapshot.CleanupPhase.ABSENT:
        item.retain_backup = item.backup is not None
        issues.append(f"destination claim indeterminate for {item.spec.path}: absent")
        _cleanup_rollback_quarantine(
            quarantine,
            issues,
            item.retained_quarantine_paths,
            item.candidate_paths,
            item.public_candidates,
            accumulator,
            expect_public_absent=False,
        )
        return

    def verify_stage(descriptor, _metadata):
        try:
            observed = _record_artifact(descriptor, "quarantined prepared report")
        except Exception:
            return repository_snapshot.ClaimVerification.UNKNOWN
        return (
            repository_snapshot.ClaimVerification.MATCH
            if observed == item.stage
            else repository_snapshot.ClaimVerification.FOREIGN
        )

    try:
        verification = quarantine.verify_claimed(verify_stage)
    except repository_snapshot.CleanupFailure:
        verification = repository_snapshot.ClaimVerification.UNKNOWN

    if verification is not repository_snapshot.ClaimVerification.MATCH:
        item.retain_backup = item.backup is not None
        issues.append(
            f"external replacement claimed from {item.spec.path} and preserved; "
            f"claimed bytes preserved as {quarantine.claimed_path}"
        )
        _cleanup_rollback_quarantine(
            quarantine,
            issues,
            item.retained_quarantine_paths,
            item.candidate_paths,
            item.public_candidates,
            accumulator,
            expect_public_absent=False,
        )
        return

    try:
        if item.original is None:
            try:
                _verify_absent(item.parent, item.spec.name, "rollback destination")
            except Exception as exc:
                issues.append(
                    f"external destination appeared at {item.spec.path} and was preserved: {exc}"
                )
            quarantine.remove_verified_claim()
        else:
            assert item.backup_name is not None and item.backup is not None
            _verify_artifact_name(
                item.parent,
                item.backup_name,
                item.backup,
                "report rollback backup",
            )
            _before_restore_original(item)
            try:
                _link_backup_if_absent(item)
            except FileExistsError as exc:
                item.retain_backup = True
                issues.append(
                    f"external destination appeared before restoration at {item.spec.path} "
                    "and was preserved; prepared bytes retained as "
                    f"{quarantine.claimed_path}; recovery backup "
                    f"retained as {item.parent.path / item.backup_name}: {exc}"
                )
                try:
                    _fsync_directory(item.parent)
                except Exception as fsync_exc:
                    issues.append(
                        f"rollback recovery fsync failed for {item.spec.path}: {fsync_exc}"
                    )
                return
            _verify_artifact_name(
                item.parent,
                item.spec.name,
                item.backup,
                "restored report destination",
            )
            quarantine.remove_verified_claim(expect_public_absent=False)
            _verify_artifact_name(
                item.parent,
                item.spec.name,
                item.backup,
                "restored report destination before recovery retention",
            )
            _before_retain_failure_backup(item)
            _verify_artifact_name(
                item.parent,
                item.spec.name,
                item.backup,
                "restored report destination at recovery-retention boundary",
            )
            item.retain_backup = True
            issues.append(
                f"public namespace restoration was attempted for {item.spec.path}; "
                "continued old-data reachability cannot be proved, so recovery backup "
                f"was retained as {item.parent.path / item.backup_name}"
            )
        item.committed = False
    except Exception as exc:
        item.retain_backup = item.backup is not None
        issues.append(
            f"rollback failed for {item.spec.path}: {exc}; quarantined bytes: "
            f"{quarantine.claimed_path}"
        )
    finally:
        _cleanup_rollback_quarantine(
            quarantine,
            issues,
            item.retained_quarantine_paths,
            item.candidate_paths,
            item.public_candidates,
            accumulator,
            expect_public_absent=item.original is None,
        )

    try:
        _fsync_directory(item.parent)
    except Exception as exc:
        issues.append(f"rollback directory fsync failed for {item.spec.path}: {exc}")


def _rollback_item(
    item: _PreparedOutput,
    issues: list[str],
    accumulator: repository_snapshot.CleanupAccumulator,
) -> None:
    assert item.stage_name is not None
    assert item.stage_identity is not None
    if item.stage_cleanup_prohibited:
        stage_path = item.parent.path / item.stage_name
        item.retain_backup = item.backup_name is not None
        if item.backup_name is not None:
            _record_recovery_path(
                item.retained_quarantine_paths,
                item.parent.path / item.backup_name,
            )
        issues.append(
            f"stage descriptor close was uncertain for {item.spec.path}; "
            "pathname cleanup prohibited and public stage is only an unproved "
            f"cleanup candidate: {stage_path}"
        )
        return
    if item.commit_indeterminate:
        item.retain_backup = item.backup is not None
        issues.append(f"replace boundary indeterminate for {item.spec.path}")

    if item.committed:
        _rollback_committed_item(item, issues, accumulator)
        return

    cleanup_completed = _claim_and_remove_public_sidecar(
        item.parent,
        item.stage_name,
        item.stage_identity,
        f"report stage for {item.spec.path}",
        issues,
        item.retained_quarantine_paths,
        item.candidate_paths,
        item.public_candidates,
        accumulator,
        artifact=item.stage,
    )
    if cleanup_completed is None:
        item.retain_backup = item.backup is not None
        return
    try:
        _fsync_directory(item.parent)
    except Exception as exc:
        issues.append(
            f"stage-cleanup directory fsync failed for {item.spec.path}: {exc}"
        )

    if item.backup_name is None or item.backup_identity is None:
        return
    if item.original is None:
        issues.append(f"unexpected recovery backup for {item.spec.path}; preserved")
        return
    if item.backup is not None:
        item.retain_backup = True
        destination_state = (
            "original destination was still observed"
            if _name_names_artifact(item.parent, item.spec.name, item.original)
            else "public destination changed or vanished"
        )
        issues.append(
            f"transaction failed after verified recovery creation for {item.spec.path}; "
            f"{destination_state}, and recovery backup was retained as "
            f"{item.parent.path / item.backup_name}"
        )
        return
    if item.backup_cleanup_prohibited or item.backup_cleanup_artifact is None:
        item.retain_backup = True
        backup_path = item.parent.path / item.backup_name
        _record_recovery_path(item.retained_quarantine_paths, backup_path)
        issues.append(
            f"unverified partial report recovery backup for {item.spec.path} "
            f"was retained without identity-only cleanup as {backup_path}"
        )
        return
    if not _name_names_artifact(item.parent, item.spec.name, item.original):
        item.retain_backup = True
        issues.append(
            f"external destination change preserved at {item.spec.path}; "
            f"recovery backup retained as {item.parent.path / item.backup_name}"
        )
        return
    if item.retain_backup:
        return
    cleanup_completed = _claim_and_remove_public_sidecar(
        item.parent,
        item.backup_name,
        item.backup_identity,
        f"partial report recovery backup for {item.spec.path}",
        issues,
        item.retained_quarantine_paths,
        item.candidate_paths,
        item.public_candidates,
        accumulator,
        artifact=item.backup_cleanup_artifact,
    )
    if cleanup_completed is None:
        item.retain_backup = item.backup is not None
        return
    try:
        _fsync_directory(item.parent)
    except Exception as exc:
        issues.append(
            f"partial-backup cleanup fsync failed for {item.spec.path}: {exc}"
        )


def _verify_and_empty_created_branch(
    descriptor: int,
    top: _CreatedDirectory,
    branch: Sequence[_CreatedDirectory],
) -> repository_snapshot.ClaimVerification:
    if top.identity is None:
        return repository_snapshot.ClaimVerification.UNKNOWN
    by_path = {directory.path: directory for directory in branch}
    child_records: dict[Path, list[_CreatedDirectory]] = {}
    for directory in branch:
        if directory.path == top.path:
            continue
        parent_record = by_path.get(directory.path.parent)
        if parent_record is None:
            return repository_snapshot.ClaimVerification.UNKNOWN
        child_records.setdefault(parent_record.path, []).append(directory)
    arena_name = f".zynum-cleanup-v2-{os.geteuid()}"

    def verify_tree(directory_descriptor: int, record: _CreatedDirectory) -> None:
        metadata = os.fstat(directory_descriptor)
        if record.identity is None or not _same_identity(
            metadata, record.identity, directory=True
        ):
            raise ReportPublicationError(
                errno.ESTALE, f"created-parent identity changed for {record.path}"
            )
        children = child_records.get(record.path, [])
        expected_names = {child.name for child in children}
        observed_names = set(os.listdir(directory_descriptor))
        allowed_names = expected_names | {arena_name}
        if not expected_names.issubset(observed_names) or not observed_names.issubset(
            allowed_names
        ):
            raise ReportPublicationError(
                errno.ENOTEMPTY,
                f"created-parent contents changed for {record.path}",
            )
        if arena_name in observed_names:
            arena_descriptor = os.open(
                arena_name,
                _directory_flags(),
                dir_fd=directory_descriptor,
            )
            arena_owner = repository_snapshot.OwnedDescriptor.take(
                arena_descriptor, record.path / arena_name
            )
            try:
                arena_metadata = os.fstat(arena_owner.fileno())
                if (
                    not stat.S_ISDIR(arena_metadata.st_mode)
                    or arena_metadata.st_uid != os.geteuid()
                    or stat.S_IMODE(arena_metadata.st_mode) != 0o700
                    or os.listdir(arena_owner.fileno())
                ):
                    raise ReportPublicationError(
                        errno.ENOTEMPTY,
                        f"created-parent cleanup arena changed for {record.path}",
                    )
            finally:
                arena_owner.close_once()
        for child in children:
            child_descriptor = os.open(
                child.name,
                _directory_flags(),
                dir_fd=directory_descriptor,
            )
            child_owner = repository_snapshot.OwnedDescriptor.take(
                child_descriptor, child.path
            )
            try:
                verify_tree(child_owner.fileno(), child)
            finally:
                child_owner.close_once()

    def empty_tree(directory_descriptor: int, record: _CreatedDirectory) -> None:
        for child in reversed(child_records.get(record.path, [])):
            child_descriptor = os.open(
                child.name,
                _directory_flags(),
                dir_fd=directory_descriptor,
            )
            child_owner = repository_snapshot.OwnedDescriptor.take(
                child_descriptor, child.path
            )
            empty_tree(child_owner.fileno(), child)
            child_owner.close_once()
            os.rmdir(child.name, dir_fd=directory_descriptor)
        try:
            os.stat(
                arena_name,
                dir_fd=directory_descriptor,
                follow_symlinks=False,
            )
        except FileNotFoundError:
            pass
        else:
            os.rmdir(arena_name, dir_fd=directory_descriptor)

    try:
        verify_tree(descriptor, top)
        empty_tree(descriptor, top)
    except Exception:
        return repository_snapshot.ClaimVerification.UNKNOWN
    return repository_snapshot.ClaimVerification.MATCH


def _cleanup_created_branch(
    top: _CreatedDirectory,
    branch: Sequence[_CreatedDirectory],
    issues: list[str],
    recovery_paths: list[Path],
    candidate_paths: list[Path],
    public_candidates: list[repository_snapshot.PublicCandidate],
    accumulator: repository_snapshot.CleanupAccumulator,
) -> None:
    source = repository_snapshot.DirectoryAnchor(top.parent_descriptor, top.path.parent)
    try:
        quarantine = _create_directory_cleanup_quarantine(source, top.name, ".created")
    except repository_snapshot.CleanupFailure as exc:
        _record_cleanup_outcome(
            f"created-parent cleanup for {top.path}",
            None,
            exc.outcome,
            issues,
            recovery_paths,
            candidate_paths,
            public_candidates,
            accumulator,
        )
        return

    try:
        quarantine.claim()
        if quarantine.phase is not repository_snapshot.CleanupPhase.ABSENT:
            verification = quarantine.verify_claimed(
                lambda descriptor, _metadata: _verify_and_empty_created_branch(
                    descriptor, top, branch
                )
            )
            if verification is repository_snapshot.ClaimVerification.MATCH:
                quarantine.remove_verified_claim()
    except repository_snapshot.CleanupFailure:
        pass
    outcome = quarantine.finish(expect_public_absent=True)
    _record_cleanup_outcome(
        f"created-parent cleanup for {top.path}",
        quarantine,
        outcome,
        issues,
        recovery_paths,
        candidate_paths,
        public_candidates,
        accumulator,
    )


def _cleanup_created_directories(
    created: Sequence[_CreatedDirectory],
    issues: list[str],
    recovery_paths: list[Path],
    candidate_paths: list[Path],
    public_candidates: list[repository_snapshot.PublicCandidate],
    accumulator: repository_snapshot.CleanupAccumulator,
) -> None:
    created_paths = {directory.path for directory in created}
    tops = [
        directory for directory in created if directory.path.parent not in created_paths
    ]
    for top in reversed(tops):
        branch = [
            directory
            for directory in created
            if directory.path == top.path or top.path in directory.path.parents
        ]
        _cleanup_created_branch(
            top,
            branch,
            issues,
            recovery_paths,
            candidate_paths,
            public_candidates,
            accumulator,
        )


def _rollback_transaction(
    items: Sequence[_PreparedOutput], created: Sequence[_CreatedDirectory]
) -> tuple[
    list[str],
    list[Path],
    list[Path],
    repository_snapshot.PublicCandidate,
    bool,
]:
    issues: list[str] = []
    recovery_paths: list[Path] = []
    candidate_paths: list[Path] = []
    public_candidates: list[repository_snapshot.PublicCandidate] = []
    accumulator = repository_snapshot.CleanupAccumulator()
    for item in reversed(items):
        if item.stage_name is not None and item.stage_identity is not None:
            _rollback_item(item, issues, accumulator)
    _cleanup_created_directories(
        created,
        issues,
        recovery_paths,
        candidate_paths,
        public_candidates,
        accumulator,
    )
    aggregate = accumulator.snapshot()
    return (
        issues,
        (
            []
            if aggregate.unaddressable
            else [*recovery_paths, *aggregate.recovery_paths]
        ),
        [*candidate_paths, *aggregate.candidate_paths],
        _aggregate_public_candidate((*public_candidates, aggregate.public_candidate)),
        aggregate.unaddressable,
    )


def _name_is_proven_absent(parent: _Parent, name: str) -> bool:
    """Return true only for descriptor-relative, no-follow proven absence."""

    recovery_paths: list[Path] = []
    observed, _ = _observe_recovery_candidate(
        parent, name, parent.path / name, recovery_paths
    )
    return not observed


def _retained_recovery_paths(
    items: Sequence[_PreparedOutput],
) -> tuple[Path, ...]:
    paths: set[Path] = set()
    for item in items:
        paths.update(item.retained_quarantine_paths)
        if (
            item.backup_name is not None
            and item.backup is not None
            and not _name_is_proven_absent(item.parent, item.backup_name)
        ):
            paths.add(item.parent.path / item.backup_name)
    return tuple(sorted(paths, key=os.fspath))


def _merged_candidate_paths(
    items: Sequence[_PreparedOutput], *causes: BaseException
) -> tuple[Path, ...]:
    paths = {path for item in items for path in item.candidate_paths}
    for cause in causes:
        paths.update(
            path
            for path in getattr(cause, "candidate_paths", ())
            if isinstance(path, Path)
        )
    return tuple(sorted(paths, key=os.fspath))


def _merged_public_candidate(
    items: Sequence[_PreparedOutput], *causes: BaseException
) -> repository_snapshot.PublicCandidate:
    values = [candidate for item in items for candidate in item.public_candidates]
    values.extend(
        candidate
        for cause in causes
        if (
            candidate := getattr(
                cause,
                "public_candidate",
                repository_snapshot.PublicCandidate.ABSENT,
            )
        )
        is not repository_snapshot.PublicCandidate.ABSENT
    )
    return _aggregate_public_candidate(values)


def _recovery_path_may_still_exist(path: Path) -> bool:
    """Re-observe a retained path without following intermediate symlinks.

    Proven absence excludes a stale recovery path.  Any other traversal or
    inspection failure remains indeterminate and therefore keeps the path in
    the recovery inventory.
    """

    if not path.is_absolute() or len(path.parts) < 2:
        return True
    try:
        descriptor = os.open(path.anchor, _directory_flags())
    except OSError:
        return True
    try:
        for component in path.parts[1:-1]:
            try:
                child = os.open(
                    _component_name(component, "report recovery path"),
                    _directory_flags(),
                    dir_fd=descriptor,
                )
            except FileNotFoundError:
                return False
            except (OSError, ValueError):
                return True
            try:
                os.close(descriptor)
            except OSError:
                pass
            descriptor = child
        try:
            os.stat(
                _component_name(path.parts[-1], "report recovery path"),
                dir_fd=descriptor,
                follow_symlinks=False,
            )
        except FileNotFoundError:
            return False
        except (OSError, ValueError):
            return True
        return True
    finally:
        try:
            os.close(descriptor)
        except OSError:
            pass


def _merged_recovery_paths(
    items: Sequence[_PreparedOutput], *causes: BaseException
) -> tuple[Path, ...]:
    """Merge item and causal recovery inventories in deterministic path order."""

    candidates = set(_retained_recovery_paths(items))
    for cause in causes:
        for path in getattr(cause, "recovery_paths", ()):
            if isinstance(path, Path):
                candidates.add(path)
    return tuple(
        sorted(
            (path for path in candidates if _recovery_path_may_still_exist(path)),
            key=os.fspath,
        )
    )


def _record_structured_cleanup_causes(
    error: BaseException,
    accumulator: repository_snapshot.CleanupAccumulator,
) -> None:
    pending = [error]
    seen: set[int] = set()
    while pending:
        current = pending.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))
        if isinstance(current, repository_snapshot.CleanupFailure):
            accumulator.record(current.outcome)
        else:
            outcome = getattr(current, "outcome", None)
            if isinstance(outcome, repository_snapshot.CleanupOutcome):
                accumulator.record(outcome)
        for structured in getattr(current, "cleanup_outcomes", ()):
            if isinstance(structured, repository_snapshot.CleanupOutcome):
                accumulator.record(structured)
        if current.__cause__ is not None:
            pending.append(current.__cause__)
        if current.__context__ is not None:
            pending.append(current.__context__)


def _audit_transaction_backup(
    item: _PreparedOutput,
    accumulator: repository_snapshot.CleanupAccumulator,
) -> None:
    assert item.backup_name is not None and item.backup is not None
    backup_path = item.parent.path / item.backup_name
    source = _directory_anchor(item.parent)
    try:
        anchor = repository_snapshot.StableRecoveryAnchor.open(source)
    except repository_snapshot.CleanupFailure as exc:
        outcome = exc.outcome
        accumulator.record(
            repository_snapshot.CleanupOutcome(
                repository_snapshot.CleanupDisposition.UNADDRESSABLE,
                (),
                outcome.issues,
                tuple(sorted({*outcome.candidate_paths, backup_path}, key=os.fspath)),
                outcome.arena_binding,
                repository_snapshot.PublicCandidate.UNKNOWN,
                outcome.arena_identity,
                outcome.recovery_anchor_identity,
            )
        )
        return

    def record_audit(
        disposition: repository_snapshot.CleanupDisposition,
        recovery_paths: tuple[Path, ...] = (),
        issues: tuple[repository_snapshot.CleanupIssue, ...] = (),
        candidate_paths: tuple[Path, ...] = (),
        public_candidate: repository_snapshot.PublicCandidate = (
            repository_snapshot.PublicCandidate.ABSENT
        ),
    ) -> None:
        final_binding = anchor.binding()
        if final_binding is not repository_snapshot.ArenaBinding.BOUND:
            disposition = repository_snapshot.CleanupDisposition.UNADDRESSABLE
            recovery_paths = ()
            issues = (
                *issues,
                repository_snapshot.CleanupIssue(
                    f"cleanup_recovery_anchor_{final_binding.name.lower()}",
                    item.parent.path,
                ),
            )
            candidate_paths = tuple(
                sorted({*candidate_paths, backup_path}, key=os.fspath)
            )
            public_candidate = repository_snapshot.PublicCandidate.UNKNOWN
        accumulator.record(
            repository_snapshot.CleanupOutcome(
                disposition,
                recovery_paths,
                issues,
                candidate_paths,
                final_binding,
                public_candidate,
                None,
                anchor.identity,
            )
        )

    def record_candidate(
        public_candidate: repository_snapshot.PublicCandidate,
        code: str,
        error: BaseException | None = None,
    ) -> None:
        record_audit(
            repository_snapshot.CleanupDisposition.REMOVED,
            issues=(repository_snapshot.CleanupIssue(code, backup_path, error),),
            candidate_paths=(backup_path,),
            public_candidate=public_candidate,
        )

    try:
        observed = os.stat(
            item.backup_name,
            dir_fd=item.parent.descriptor,
            follow_symlinks=False,
        )
    except FileNotFoundError:
        record_audit(repository_snapshot.CleanupDisposition.ABSENT)
        return
    except (OSError, ValueError) as exc:
        record_candidate(
            repository_snapshot.PublicCandidate.UNKNOWN,
            "report_backup_audit_uninspectable",
            exc,
        )
        return

    if (
        not stat.S_ISREG(observed.st_mode)
        or _identity(observed) != item.backup.identity
    ):
        record_candidate(
            repository_snapshot.PublicCandidate.PRESENT,
            "report_backup_audit_foreign",
        )
        return

    try:
        descriptor = os.open(
            item.backup_name,
            _file_flags(),
            dir_fd=item.parent.descriptor,
        )
    except FileNotFoundError:
        record_audit(repository_snapshot.CleanupDisposition.ABSENT)
        return
    except (OSError, ValueError) as exc:
        record_candidate(
            repository_snapshot.PublicCandidate.UNKNOWN,
            "report_backup_audit_open_failed",
            exc,
        )
        return

    audit_error: BaseException | None = None
    artifact: _Artifact | None = None
    try:
        artifact = _record_artifact(descriptor, "transaction-complete backup audit")
    except BaseException as exc:
        audit_error = exc
    try:
        os.close(descriptor)
    except OSError as exc:
        audit_error = audit_error or exc
    if audit_error is not None:
        record_candidate(
            repository_snapshot.PublicCandidate.UNKNOWN,
            "report_backup_audit_uninspectable",
            audit_error,
        )
        return
    if artifact != item.backup:
        record_candidate(
            repository_snapshot.PublicCandidate.PRESENT,
            "report_backup_audit_foreign",
        )
        return
    record_audit(
        repository_snapshot.CleanupDisposition.RETAINED,
        recovery_paths=(backup_path,),
    )


def _cleanup_backup_after_transaction(
    item: _PreparedOutput,
    accumulator: repository_snapshot.CleanupAccumulator,
) -> None:
    assert item.backup_name is not None and item.backup is not None
    issues: list[str] = []
    recovery_paths = item.retained_quarantine_paths
    cleanup_outcomes: list[repository_snapshot.CleanupOutcome] = []
    cleanup_completed = _claim_and_remove_public_sidecar(
        item.parent,
        item.backup_name,
        item.backup.identity,
        "transaction-complete report recovery backup",
        issues,
        recovery_paths,
        item.candidate_paths,
        item.public_candidates,
        accumulator,
        artifact=item.backup,
        final_outcomes=cleanup_outcomes,
    )
    if cleanup_completed is not None:
        try:
            _verify_absent(
                item.parent,
                item.backup_name,
                "transaction-complete report recovery backup",
            )
        except Exception as exc:
            final_outcome = cleanup_outcomes[-1]
            candidate_path = item.parent.path / item.backup_name
            public_candidate = (
                repository_snapshot.PublicCandidate.PRESENT
                if isinstance(exc, ReportPublicationError) and exc.errno == errno.EEXIST
                else repository_snapshot.PublicCandidate.UNKNOWN
            )
            observation_outcome = repository_snapshot.CleanupOutcome(
                final_outcome.disposition,
                (),
                (
                    repository_snapshot.CleanupIssue(
                        (
                            "cleanup_public_name_reappeared"
                            if public_candidate
                            is repository_snapshot.PublicCandidate.PRESENT
                            else "cleanup_public_absence_uninspectable"
                        ),
                        candidate_path,
                        exc,
                    ),
                ),
                (candidate_path,),
                final_outcome.arena_binding,
                public_candidate,
                final_outcome.arena_identity,
                final_outcome.recovery_anchor_identity,
            )
            accumulator.record(observation_outcome)
            item.candidate_paths.append(candidate_path)
            item.public_candidates.append(public_candidate)
            issues.append(
                "transaction-complete report recovery backup public-name absence "
                f"could not be verified: {exc}"
            )
    if not issues:
        return
    aggregate = accumulator.snapshot()
    ordered_paths = aggregate.recovery_paths
    ordered_candidates = aggregate.candidate_paths
    raise TransactionCompleteCleanupError(
        errno.EIO,
        "report transaction complete but backup cleanup became indeterminate; "
        "coherent new generation retained; " + "; ".join(issues),
        recovery_paths=ordered_paths,
        candidate_paths=ordered_candidates,
        public_candidate=aggregate.public_candidate,
    )


def _fsync_backup_cleanup_parent(parent: _Parent) -> None:
    _fsync_directory(parent)


def _close_resources(
    items: Sequence[_PreparedOutput],
    parents: Sequence[_Parent],
    created: Sequence[_CreatedDirectory],
) -> None:
    for item in items:
        if item.original_descriptor is not None:
            try:
                os.close(item.original_descriptor)
            except OSError:
                pass
            item.original_descriptor = None
    for parent in parents:
        try:
            os.close(parent.descriptor)
        except OSError:
            pass
    for directory in created:
        try:
            directory.parent_owner.close_once()
        except OSError:
            pass


def publish_outputs(outputs: Sequence[ReportOutput]) -> None:
    """Publish a deterministic report generation or fail with guarded recovery.

    Inputs are completely materialized and validated before any filesystem
    mutation.  Successful return means every destination has the supplied
    bytes at mode ``0644`` and no publication sidecars remain.
    """

    specs = _normalize_outputs(outputs)
    _require_platform_support()
    created: list[_CreatedDirectory] = []
    parents_by_path: dict[str, _Parent] = {}
    items: list[_PreparedOutput] = []
    transaction_complete = False
    transaction_cleanup_accumulator = repository_snapshot.CleanupAccumulator()
    try:
        for spec in specs:
            parent_key = os.fspath(spec.parent_path)
            parent = parents_by_path.get(parent_key)
            if parent is None:
                parent = _open_parent(spec.parent_path, created)
                parents_by_path[parent_key] = parent
            items.append(_PreparedOutput(spec=spec, parent=parent))

        _preflight_canonical_collisions(items)

        for item in items:
            try:
                descriptor = _open_regular_name(
                    item.parent, item.spec.name, "report destination"
                )
            except FileNotFoundError:
                _verify_absent(item.parent, item.spec.name, "report destination")
            else:
                item.original_descriptor = descriptor
                item.original = _record_artifact(descriptor, "report destination")
                _validate_restorable_metadata(
                    descriptor, item.original, "report destination"
                )

        for item in items:
            _prepare_stage(item)
        for item in items:
            if item.original is not None:
                _prepare_backup(item)

        parents = _unique_parents(items)
        for parent in parents:
            _verify_parent(parent, "immediately before report commit")
        for item in items:
            _verify_item_precommit(item)

        for item in items:
            assert item.stage_name is not None and item.stage is not None
            try:
                _replace_name(item.parent, item.stage_name, item.spec.name)
            except Exception:
                if _name_names_artifact(item.parent, item.spec.name, item.stage):
                    item.committed = True
                elif not _name_names_artifact(item.parent, item.stage_name, item.stage):
                    item.commit_indeterminate = True
                    item.retain_backup = item.backup is not None
                raise
            else:
                item.committed = True

        for parent in parents:
            _verify_parent(parent, "immediately after report commit")
            _fsync_directory(parent)
        for item in items:
            _verify_item_published(item, "after commit")
        for parent in parents:
            _verify_parent(parent, "at report transaction-complete boundary")
        for item in items:
            _verify_item_published(item, "at transaction-complete boundary")
        transaction_complete = True

        backups = tuple(item for item in items if item.backup is not None)
        for item in backups:
            assert item.backup_name is not None and item.backup is not None
            _verify_artifact_name(
                item.parent,
                item.backup_name,
                item.backup,
                "transaction-complete report recovery backup",
            )
        cleanup_parents: dict[_FileIdentity, _Parent] = {}
        cleanup_errors: list[BaseException] = []
        for item in backups:
            cleanup_parents[_identity(os.fstat(item.parent.descriptor))] = item.parent
            try:
                _cleanup_backup_after_transaction(item, transaction_cleanup_accumulator)
            except Exception as cleanup_exc:
                _record_structured_cleanup_causes(
                    cleanup_exc, transaction_cleanup_accumulator
                )
                cleanup_errors.append(cleanup_exc)
        fsync_issues: list[str] = []
        for identity in sorted(
            cleanup_parents, key=lambda value: (value.device, value.inode)
        ):
            parent = cleanup_parents[identity]
            try:
                _fsync_backup_cleanup_parent(parent)
            except Exception as fsync_exc:
                fsync_issues.append(f"{parent.path}: {fsync_exc}")
        if cleanup_errors or fsync_issues:
            for item in backups:
                _audit_transaction_backup(item, transaction_cleanup_accumulator)
            aggregate = transaction_cleanup_accumulator.snapshot()
            recovery_paths = aggregate.recovery_paths
            candidate_paths = aggregate.candidate_paths
            detail = (
                "; recovery material retained: "
                + ", ".join(os.fspath(path) for path in recovery_paths)
                if recovery_paths
                else ""
            )
            candidate_detail = (
                "; unproved candidate paths: "
                + ", ".join(os.fspath(path) for path in candidate_paths)
                if candidate_paths
                else ""
            )
            issue_detail = (
                "; structured cleanup issues: "
                + " | ".join(
                    f"{issue.code} at {issue.path}"
                    + (f": {issue.error}" if issue.error is not None else "")
                    for issue in aggregate.issues
                )
                if aggregate.issues
                else ""
            )
            cleanup_detail = (
                "; cleanup failures: "
                + " | ".join(str(error) for error in cleanup_errors)
                if cleanup_errors
                else ""
            )
            fsync_detail = (
                "; backup-cleanup directory fsync failed: " + ", ".join(fsync_issues)
                if fsync_issues
                else ""
            )
            cause = cleanup_errors[0] if cleanup_errors else None
            error = TransactionCompleteCleanupError(
                errno.EIO,
                "report transaction complete but backup cleanup failed; "
                "coherent new generation retained"
                + detail
                + candidate_detail
                + issue_detail
                + cleanup_detail
                + fsync_detail,
                recovery_paths=recovery_paths,
                candidate_paths=candidate_paths,
                public_candidate=aggregate.public_candidate,
            )
            if cause is not None:
                raise error from cause
            raise error
    except Exception as exc:
        if transaction_complete:
            if isinstance(exc, TransactionCompleteCleanupError):
                raise
            _record_structured_cleanup_causes(exc, transaction_cleanup_accumulator)
            for item in backups:
                _audit_transaction_backup(item, transaction_cleanup_accumulator)
            aggregate = transaction_cleanup_accumulator.snapshot()
            recovery_paths = aggregate.recovery_paths
            candidate_paths = aggregate.candidate_paths
            detail = (
                "; recovery material retained: "
                + ", ".join(os.fspath(path) for path in recovery_paths)
                if recovery_paths
                else ""
            )
            candidate_detail = (
                "; unproved candidate paths: "
                + ", ".join(os.fspath(path) for path in candidate_paths)
                if candidate_paths
                else ""
            )
            issue_detail = (
                "; structured cleanup issues: "
                + " | ".join(
                    f"{issue.code} at {issue.path}"
                    + (f": {issue.error}" if issue.error is not None else "")
                    for issue in aggregate.issues
                )
                if aggregate.issues
                else ""
            )
            raise TransactionCompleteCleanupError(
                errno.EIO,
                "report transaction complete but backup cleanup failed; "
                "coherent new generation retained"
                + detail
                + candidate_detail
                + issue_detail,
                recovery_paths=recovery_paths,
                candidate_paths=candidate_paths,
                public_candidate=aggregate.public_candidate,
            ) from exc
        (
            issues,
            rollback_recovery_paths,
            rollback_candidate_paths,
            rollback_public_candidate,
            rollback_unaddressable,
        ) = _rollback_transaction(items, created)
        if issues:
            recovery_paths = (
                ()
                if rollback_unaddressable
                else tuple(
                    sorted(
                        {
                            *_merged_recovery_paths(items, exc),
                            *(
                                path
                                for path in rollback_recovery_paths
                                if _recovery_path_may_still_exist(path)
                            ),
                        },
                        key=os.fspath,
                    )
                )
            )
            recovery_detail = (
                "; retained recovery paths: "
                + ", ".join(os.fspath(path) for path in recovery_paths)
                if recovery_paths
                else ""
            )
            candidate_paths = tuple(
                sorted(
                    {
                        *_merged_candidate_paths(items, exc),
                        *rollback_candidate_paths,
                    },
                    key=os.fspath,
                )
            )
            candidate_detail = (
                "; unproved candidate paths: "
                + ", ".join(os.fspath(path) for path in candidate_paths)
                if candidate_paths
                else ""
            )
            public_candidate = _aggregate_public_candidate(
                (
                    _merged_public_candidate(items, exc),
                    rollback_public_candidate,
                )
            )
            raise RollbackIndeterminateError(
                errno.EIO,
                "rollback_indeterminate: "
                + "; ".join(issues)
                + recovery_detail
                + candidate_detail,
                recovery_paths=recovery_paths,
                candidate_paths=candidate_paths,
                public_candidate=public_candidate,
            ) from exc
        raise
    finally:
        _close_resources(items, tuple(parents_by_path.values()), created)


__all__ = [
    "ReportOutput",
    "ReportPublicationError",
    "RollbackIndeterminateError",
    "TransactionCompleteCleanupError",
    "publish_outputs",
]
