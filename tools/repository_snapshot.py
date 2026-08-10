#!/usr/bin/env python3
# Copyright (C) 2026 Zynum contributors
# SPDX-License-Identifier: LGPL-3.0-or-later

"""Race-resistant repository tree snapshots.

All traversal in this module is descriptor-relative and refuses to follow
symlinks.  Callers provide strict repository-relative POSIX paths and receive
immutable facts that can be verified again immediately before publishing an
artifact.
"""

from __future__ import annotations

import hashlib
import enum
import os
import secrets
import stat
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import BinaryIO, Callable


class RepositorySnapshotError(ValueError):
    """The requested filesystem fact could not be proved safely."""


@dataclass(frozen=True, slots=True)
class DirectoryAnchor:
    """One borrowed directory descriptor plus its diagnostic pathname."""

    descriptor: int
    path: Path

    def __post_init__(self) -> None:
        if type(self.descriptor) is not int or self.descriptor < 0:
            raise ValueError("directory anchor descriptor must be non-negative")
        if not isinstance(self.path, Path):
            raise TypeError("directory anchor path must be pathlib.Path")


@dataclass(frozen=True, slots=True)
class StableRecoveryAnchor:
    """A held directory whose child names resist in-scope credential changes."""

    descriptor: int
    path: Path
    identity: tuple[int, int]
    device: int

    @classmethod
    def open(cls, source: DirectoryAnchor) -> "StableRecoveryAnchor":
        try:
            held = os.fstat(source.descriptor)
            named = os.stat(source.path, follow_symlinks=False)
        except OSError as exc:
            issue = CleanupIssue(
                "cleanup_recovery_anchor_uninspectable", source.path, exc
            )
            raise CleanupFailure(
                _cleanup_outcome(
                    CleanupDisposition.UNADDRESSABLE,
                    (issue,),
                    (),
                    arena_binding=ArenaBinding.UNKNOWN,
                )
            ) from exc
        identity = (held.st_dev, held.st_ino)
        if (
            not stat.S_ISDIR(held.st_mode)
            or not stat.S_ISDIR(named.st_mode)
            or (named.st_dev, named.st_ino) != identity
        ):
            issue = CleanupIssue("cleanup_recovery_anchor_rebound", source.path)
            raise CleanupFailure(
                _cleanup_outcome(
                    CleanupDisposition.UNADDRESSABLE,
                    (issue,),
                    (),
                    arena_binding=ArenaBinding.REBOUND,
                    recovery_anchor_identity=identity,
                )
            )
        effective_uid = os.geteuid()
        mode = stat.S_IMODE(held.st_mode)
        sticky_owner_protected = bool(held.st_mode & stat.S_ISVTX) and held.st_uid in {
            effective_uid,
            0,
        }
        private_owner_protected = held.st_uid == effective_uid and not (
            mode & (stat.S_IWGRP | stat.S_IWOTH)
        )
        if not (sticky_owner_protected or private_owner_protected):
            issue = CleanupIssue(
                "cleanup_recovery_anchor_not_rename_protected", source.path
            )
            raise CleanupFailure(
                _cleanup_outcome(
                    CleanupDisposition.UNADDRESSABLE,
                    (issue,),
                    (),
                    arena_binding=ArenaBinding.UNKNOWN,
                    recovery_anchor_identity=identity,
                )
            )
        return cls(source.descriptor, source.path, identity, held.st_dev)

    def directory_anchor(self) -> DirectoryAnchor:
        return DirectoryAnchor(self.descriptor, self.path)

    def binding(self) -> ArenaBinding:
        try:
            named = os.stat(self.path, follow_symlinks=False)
        except FileNotFoundError:
            return ArenaBinding.ABSENT
        except OSError:
            return ArenaBinding.UNKNOWN
        if (
            not stat.S_ISDIR(named.st_mode)
            or (named.st_dev, named.st_ino) != self.identity
        ):
            return ArenaBinding.REBOUND
        return ArenaBinding.BOUND


class CleanupPhase(enum.Enum):
    READY = enum.auto()
    CLAIMED_UNSYNCED = enum.auto()
    CLAIMED_DURABLE = enum.auto()
    VERIFIED = enum.auto()
    CLAIM_REMOVED = enum.auto()
    CLOSED = enum.auto()
    ABSENT = enum.auto()
    REMOVED = enum.auto()
    RETAINED = enum.auto()
    UNADDRESSABLE = enum.auto()


class ClaimVerification(enum.Enum):
    MATCH = enum.auto()
    FOREIGN = enum.auto()
    UNKNOWN = enum.auto()


class ClaimKind(enum.Enum):
    REGULAR_FILE = enum.auto()
    DIRECTORY = enum.auto()


class ArenaBinding(enum.Enum):
    BOUND = enum.auto()
    ABSENT = enum.auto()
    REBOUND = enum.auto()
    UNKNOWN = enum.auto()


class PublicCandidate(enum.Enum):
    ABSENT = enum.auto()
    PRESENT = enum.auto()
    UNKNOWN = enum.auto()


class CleanupDisposition(enum.Enum):
    ABSENT = enum.auto()
    REMOVED = enum.auto()
    RETAINED = enum.auto()
    UNADDRESSABLE = enum.auto()


@dataclass(frozen=True, slots=True)
class CleanupIssue:
    code: str
    path: Path
    error: BaseException | None = field(default=None, compare=False, repr=False)


@dataclass(frozen=True, slots=True)
class CleanupOutcome:
    disposition: CleanupDisposition
    recovery_paths: tuple[Path, ...]
    issues: tuple[CleanupIssue, ...]
    candidate_paths: tuple[Path, ...] = ()
    arena_binding: ArenaBinding = ArenaBinding.BOUND
    public_candidate: PublicCandidate = PublicCandidate.ABSENT
    arena_identity: tuple[int, int] | None = None
    recovery_anchor_identity: tuple[int, int] | None = None


class CleanupFailure(RuntimeError):
    """A private cleanup could not safely remove all recovery material."""

    def __init__(self, outcome: CleanupOutcome) -> None:
        super().__init__(
            "; ".join(issue.code for issue in outcome.issues)
            or "repository cleanup failed closed"
        )
        self.outcome = outcome


@dataclass(frozen=True, slots=True)
class CleanupAggregate:
    recovery_paths: tuple[Path, ...]
    candidate_paths: tuple[Path, ...]
    issues: tuple[CleanupIssue, ...]
    unaddressable: bool
    tainted_anchor_identities: tuple[tuple[int, int], ...]
    public_candidate: PublicCandidate


@dataclass(slots=True)
class CleanupAccumulator:
    """Monotonically aggregate outcomes without reviving stale pathnames."""

    _issues: list[CleanupIssue] = field(default_factory=list)
    _candidate_paths: list[Path] = field(default_factory=list)
    _recovery_by_anchor: dict[tuple[int, int] | None, set[Path]] = field(
        default_factory=dict
    )
    _tainted_anchors: set[tuple[int, int]] = field(default_factory=set)
    _globally_unaddressable: bool = False
    _public_candidate: PublicCandidate = PublicCandidate.ABSENT

    @staticmethod
    def _anchor_key(outcome: CleanupOutcome) -> tuple[int, int] | None:
        return outcome.recovery_anchor_identity or outcome.arena_identity

    def record(self, outcome: CleanupOutcome) -> None:
        if not isinstance(outcome, CleanupOutcome):
            raise TypeError("cleanup accumulator requires CleanupOutcome values")
        self._issues.extend(outcome.issues)
        self._candidate_paths.extend(outcome.candidate_paths)
        if outcome.public_candidate is PublicCandidate.UNKNOWN:
            self._public_candidate = PublicCandidate.UNKNOWN
        elif (
            outcome.public_candidate is PublicCandidate.PRESENT
            and self._public_candidate is PublicCandidate.ABSENT
        ):
            self._public_candidate = PublicCandidate.PRESENT

        key = self._anchor_key(outcome)
        unaddressable = (
            outcome.disposition is CleanupDisposition.UNADDRESSABLE
            or outcome.arena_binding is not ArenaBinding.BOUND
        )
        if unaddressable:
            if key is None:
                self._globally_unaddressable = True
                self._recovery_by_anchor.clear()
            else:
                self._tainted_anchors.add(key)
                self._recovery_by_anchor.pop(key, None)
                self._recovery_by_anchor.pop(None, None)
            return
        if self._globally_unaddressable or key in self._tainted_anchors:
            return
        if key is None and self._tainted_anchors:
            return
        self._recovery_by_anchor.setdefault(key, set()).update(outcome.recovery_paths)

    def snapshot(self) -> CleanupAggregate:
        recovery_paths = (
            ()
            if self._globally_unaddressable or self._tainted_anchors
            else _cleanup_paths(
                path
                for key, paths in self._recovery_by_anchor.items()
                if key is None or key not in self._tainted_anchors
                for path in paths
            )
        )
        return CleanupAggregate(
            recovery_paths=recovery_paths,
            candidate_paths=_cleanup_paths(self._candidate_paths),
            issues=tuple(self._issues),
            unaddressable=(self._globally_unaddressable or bool(self._tainted_anchors)),
            tainted_anchor_identities=tuple(sorted(self._tainted_anchors)),
            public_candidate=self._public_candidate,
        )


@dataclass(slots=True)
class OwnedDescriptor:
    """Single-owner descriptor whose close is attempted at most once."""

    _descriptor: int | None
    recovery_path: Path
    _close_attempted: bool = False
    _close_uncertain: bool = False

    @classmethod
    def take(cls, descriptor: int, recovery_path: Path) -> "OwnedDescriptor":
        if type(descriptor) is not int or descriptor < 0:
            raise ValueError("owned descriptor must be non-negative")
        return cls(descriptor, recovery_path)

    @property
    def close_uncertain(self) -> bool:
        return self._close_uncertain

    @property
    def close_attempted(self) -> bool:
        return self._close_attempted

    def fileno(self) -> int:
        if self._descriptor is None or self._close_attempted:
            raise RuntimeError("owned descriptor is no longer borrowable")
        return self._descriptor

    def transfer(self) -> int:
        descriptor = self.fileno()
        self._descriptor = None
        return descriptor

    def close_once(self) -> None:
        if self._close_attempted:
            raise RuntimeError("owned descriptor close was already attempted")
        if self._descriptor is None:
            raise RuntimeError("owned descriptor ownership was transferred")
        descriptor = self._descriptor
        self._descriptor = None
        self._close_attempted = True
        try:
            os.close(descriptor)
        except OSError:
            self._close_uncertain = True
            raise


def _cleanup_paths(paths: Iterable[Path]) -> tuple[Path, ...]:
    return tuple(sorted(set(paths), key=os.fspath))


def _cleanup_outcome(
    disposition: CleanupDisposition,
    issues: Iterable[CleanupIssue],
    recovery_paths: Iterable[Path],
    *,
    candidate_paths: Iterable[Path] = (),
    arena_binding: ArenaBinding = ArenaBinding.BOUND,
    public_candidate: PublicCandidate = PublicCandidate.ABSENT,
    arena_identity: tuple[int, int] | None = None,
    recovery_anchor_identity: tuple[int, int] | None = None,
) -> CleanupOutcome:
    return CleanupOutcome(
        disposition=disposition,
        recovery_paths=_cleanup_paths(recovery_paths),
        issues=tuple(issues),
        candidate_paths=_cleanup_paths(candidate_paths),
        arena_binding=arena_binding,
        public_candidate=public_candidate,
        arena_identity=arena_identity,
        recovery_anchor_identity=recovery_anchor_identity,
    )


def _cleanup_component(name: str, label: str) -> str:
    if (
        type(name) is not str
        or not name
        or name in {".", ".."}
        or "/" in name
        or "\\" in name
        or "\0" in name
    ):
        raise ValueError(f"{label} must be one safe path component")
    return name


def _preclaim_outcome(
    outcome: CleanupOutcome,
    source: DirectoryAnchor,
    public_name: str,
) -> CleanupOutcome:
    """Conservatively retain an unclaimed public target in setup failures."""
    public_path = source.path / public_name
    issues = list(outcome.issues)
    candidate_paths = list(outcome.candidate_paths)
    public_candidate = PublicCandidate.ABSENT
    try:
        os.stat(
            public_name,
            dir_fd=source.descriptor,
            follow_symlinks=False,
        )
    except FileNotFoundError:
        pass
    except OSError as exc:
        issues.append(
            CleanupIssue("cleanup_public_absence_uninspectable", public_path, exc)
        )
        candidate_paths.append(public_path)
        public_candidate = PublicCandidate.UNKNOWN
    else:
        candidate_paths.append(public_path)
        public_candidate = PublicCandidate.PRESENT
    return _cleanup_outcome(
        outcome.disposition,
        issues,
        outcome.recovery_paths,
        candidate_paths=candidate_paths,
        arena_binding=outcome.arena_binding,
        public_candidate=public_candidate,
        arena_identity=outcome.arena_identity,
        recovery_anchor_identity=outcome.recovery_anchor_identity,
    )


@dataclass(slots=True)
class CleanupArena:
    """A bounded credential-private namespace retained under a public parent."""

    anchor: StableRecoveryAnchor
    name: str
    path: Path
    descriptor: OwnedDescriptor
    identity: tuple[int, int]
    owner_uid: int

    @classmethod
    def open(
        cls,
        source: DirectoryAnchor | StableRecoveryAnchor,
        name: str | None = None,
        *,
        create: bool = True,
        expected_identity: tuple[int, int] | None = None,
    ) -> "CleanupArena":
        if os.name != "posix":
            source_path = source.path
            path = source_path / ".zynum-cleanup-v2-unsupported"
            issue = CleanupIssue("cleanup_posix_semantics_unavailable", path)
            raise CleanupFailure(
                _cleanup_outcome(
                    CleanupDisposition.UNADDRESSABLE,
                    (issue,),
                    (),
                    arena_binding=ArenaBinding.UNKNOWN,
                )
            )
        anchor = (
            source
            if isinstance(source, StableRecoveryAnchor)
            else StableRecoveryAnchor.open(source)
        )
        source = anchor.directory_anchor()
        if name is None:
            name = f".zynum-cleanup-v2-{os.geteuid()}"
        name = _cleanup_component(name, "cleanup arena name")
        path = source.path / name
        try:
            source_metadata = os.fstat(source.descriptor)
        except OSError as exc:
            issue = CleanupIssue(
                "cleanup_source_anchor_uninspectable", source.path, exc
            )
            raise CleanupFailure(
                _cleanup_outcome(
                    CleanupDisposition.UNADDRESSABLE,
                    (issue,),
                    (),
                    arena_binding=ArenaBinding.UNKNOWN,
                    recovery_anchor_identity=anchor.identity,
                )
            ) from exc
        if not stat.S_ISDIR(source_metadata.st_mode):
            issue = CleanupIssue("cleanup_source_anchor_not_directory", source.path)
            raise CleanupFailure(
                _cleanup_outcome(
                    CleanupDisposition.UNADDRESSABLE,
                    (issue,),
                    (),
                    arena_binding=ArenaBinding.UNKNOWN,
                    recovery_anchor_identity=anchor.identity,
                )
            )

        created_here = False
        if create:
            try:
                os.mkdir(name, 0o700, dir_fd=source.descriptor)
                created_here = True
            except FileExistsError:
                pass
            except OSError as exc:
                issue = CleanupIssue("cleanup_arena_create_failed", path, exc)
                raise CleanupFailure(
                    _cleanup_outcome(
                        CleanupDisposition.UNADDRESSABLE,
                        (issue,),
                        (),
                        arena_binding=ArenaBinding.UNKNOWN,
                        recovery_anchor_identity=anchor.identity,
                    )
                ) from exc

        owned: OwnedDescriptor | None = None
        identity: tuple[int, int] | None = None
        try:
            observed = os.stat(name, dir_fd=source.descriptor, follow_symlinks=False)
            descriptor = os.open(
                name,
                os.O_RDONLY | os.O_CLOEXEC | os.O_DIRECTORY | os.O_NOFOLLOW,
                dir_fd=source.descriptor,
            )
            owned = OwnedDescriptor.take(descriptor, path)
            opened = os.fstat(owned.fileno())
            effective_uid = os.geteuid()
            identity = (observed.st_dev, observed.st_ino)
            if expected_identity is not None and identity != expected_identity:
                raise OSError("cleanup arena identity differs from expected binding")
            if (
                not stat.S_ISDIR(observed.st_mode)
                or not stat.S_ISDIR(opened.st_mode)
                or (opened.st_dev, opened.st_ino) != identity
                or observed.st_dev != source_metadata.st_dev
                or observed.st_uid != effective_uid
                or opened.st_uid != effective_uid
                or stat.S_IMODE(observed.st_mode) & 0o077
                or stat.S_IMODE(opened.st_mode) & 0o077
            ):
                raise OSError("cleanup arena initial credential is unsafe")
            if created_here:
                os.fchmod(owned.fileno(), 0o700)
            configured_path = os.stat(
                name,
                dir_fd=source.descriptor,
                follow_symlinks=False,
            )
            configured_fd = os.fstat(owned.fileno())
            if (
                not stat.S_ISDIR(configured_path.st_mode)
                or not stat.S_ISDIR(configured_fd.st_mode)
                or (configured_path.st_dev, configured_path.st_ino) != identity
                or (configured_fd.st_dev, configured_fd.st_ino) != identity
                or configured_path.st_uid != effective_uid
                or configured_fd.st_uid != effective_uid
                or stat.S_IMODE(configured_path.st_mode) != 0o700
                or stat.S_IMODE(configured_fd.st_mode) != 0o700
            ):
                raise OSError("cleanup arena configured credential is unsafe")
            if created_here:
                os.fsync(owned.fileno())
                os.fsync(source.descriptor)
            final_binding = os.stat(
                name,
                dir_fd=source.descriptor,
                follow_symlinks=False,
            )
            if (
                not stat.S_ISDIR(final_binding.st_mode)
                or (final_binding.st_dev, final_binding.st_ino) != identity
                or final_binding.st_uid != effective_uid
                or stat.S_IMODE(final_binding.st_mode) != 0o700
            ):
                raise OSError("cleanup arena binding changed before return")
            anchor_binding = anchor.binding()
            if anchor_binding is not ArenaBinding.BOUND:
                raise OSError(
                    "cleanup recovery anchor binding changed before arena return"
                )
            return cls(anchor, name, path, owned, identity, effective_uid)
        except BaseException as exc:
            if isinstance(exc, FileNotFoundError) and not create:
                binding = ArenaBinding.ABSENT
                code = "cleanup_arena_binding_absent"
            elif expected_identity is not None:
                binding = ArenaBinding.REBOUND
                code = "cleanup_arena_binding_rebound"
            else:
                binding = ArenaBinding.UNKNOWN
                code = "cleanup_arena_setup_failed"
            issues = [CleanupIssue(code, path, exc)]
            if owned is not None:
                try:
                    owned.close_once()
                except OSError as close_exc:
                    issues.append(
                        CleanupIssue(
                            "cleanup_arena_descriptor_close_uncertain",
                            path,
                            close_exc,
                        )
                    )
            raise CleanupFailure(
                _cleanup_outcome(
                    CleanupDisposition.UNADDRESSABLE,
                    issues,
                    (),
                    arena_binding=binding,
                    arena_identity=identity,
                    recovery_anchor_identity=anchor.identity,
                )
            ) from exc

    def close_issue(self) -> CleanupIssue | None:
        try:
            self.descriptor.close_once()
        except OSError as exc:
            return CleanupIssue(
                "cleanup_arena_descriptor_close_uncertain", self.path, exc
            )
        return None

    def binding(self) -> ArenaBinding:
        before = self.anchor.binding()
        if before is not ArenaBinding.BOUND:
            return before
        try:
            observed = os.stat(
                self.name,
                dir_fd=self.anchor.descriptor,
                follow_symlinks=False,
            )
        except FileNotFoundError:
            return ArenaBinding.ABSENT
        except OSError:
            return ArenaBinding.UNKNOWN
        if (
            not stat.S_ISDIR(observed.st_mode)
            or (observed.st_dev, observed.st_ino) != self.identity
            or observed.st_uid != self.owner_uid
            or stat.S_IMODE(observed.st_mode) != 0o700
        ):
            return ArenaBinding.REBOUND
        return self.anchor.binding()

    def binding_issue(self) -> CleanupIssue | None:
        binding = self.binding()
        if binding is ArenaBinding.BOUND:
            return None
        return CleanupIssue(
            f"cleanup_arena_binding_{binding.name.lower()}",
            self.path,
        )

    def duplicate(self) -> "CleanupArena":
        """Return an independently owned descriptor for the same held arena."""
        try:
            descriptor = os.dup(self.descriptor.fileno())
        except OSError as exc:
            issue = CleanupIssue(
                "cleanup_arena_descriptor_duplicate_failed", self.path, exc
            )
            binding = self.binding()
            raise CleanupFailure(
                _cleanup_outcome(
                    (
                        CleanupDisposition.RETAINED
                        if binding is ArenaBinding.BOUND
                        else CleanupDisposition.UNADDRESSABLE
                    ),
                    (issue,),
                    (),
                    arena_binding=binding,
                    arena_identity=self.identity,
                    recovery_anchor_identity=self.anchor.identity,
                )
            ) from exc
        return CleanupArena(
            anchor=self.anchor,
            name=self.name,
            path=self.path,
            descriptor=OwnedDescriptor.take(descriptor, self.path),
            identity=self.identity,
            owner_uid=self.owner_uid,
        )


def finalize_arena_outcome(
    arena: CleanupArena,
    outcome: CleanupOutcome,
) -> CleanupOutcome:
    """Close one arena owner and prove addressability at the return boundary."""
    issues = list(outcome.issues)
    close_issue = arena.close_issue()
    if close_issue is not None:
        issues.append(close_issue)
    final_binding = arena.binding()
    if final_binding is not ArenaBinding.BOUND:
        code = f"cleanup_arena_binding_{final_binding.name.lower()}"
        if all(issue.code != code for issue in issues):
            issues.append(CleanupIssue(code, arena.path))
    binding = (
        outcome.arena_binding
        if outcome.arena_binding is not ArenaBinding.BOUND
        else final_binding
    )
    if (
        outcome.disposition is CleanupDisposition.UNADDRESSABLE
        or binding is not ArenaBinding.BOUND
        or final_binding is not ArenaBinding.BOUND
    ):
        disposition = CleanupDisposition.UNADDRESSABLE
        recovery_paths: Iterable[Path] = ()
    else:
        disposition = (
            CleanupDisposition.RETAINED
            if close_issue is not None
            else outcome.disposition
        )
        recovery_paths = outcome.recovery_paths
    return _cleanup_outcome(
        disposition,
        issues,
        recovery_paths,
        candidate_paths=outcome.candidate_paths,
        arena_binding=binding,
        public_candidate=outcome.public_candidate,
        arena_identity=outcome.arena_identity or arena.identity,
        recovery_anchor_identity=(
            outcome.recovery_anchor_identity or arena.anchor.identity
        ),
    )


@dataclass(slots=True)
class CleanupDirectory:
    """One random operation directory owned inside a held cleanup arena."""

    arena: CleanupArena
    name: str
    path: Path
    descriptor: OwnedDescriptor
    identity: tuple[int, int]
    owner_uid: int

    @classmethod
    def create(
        cls,
        arena: CleanupArena,
        *,
        prefix: str,
        suffix: str,
        token_bytes: int = 16,
    ) -> "CleanupDirectory":
        if not isinstance(arena, CleanupArena):
            raise TypeError("cleanup directory arena must be CleanupArena")
        if type(prefix) is not str or type(suffix) is not str:
            raise TypeError("cleanup directory prefix and suffix must be strings")
        if type(token_bytes) is not int or not 1 <= token_bytes <= 64:
            raise ValueError("cleanup directory token_bytes must be between 1 and 64")
        _cleanup_component(
            f"{prefix}{'0' * (token_bytes * 2)}{suffix}",
            "cleanup directory name",
        )
        arena = arena.duplicate()
        binding = arena.binding()
        if binding is not ArenaBinding.BOUND:
            binding_issue = CleanupIssue(
                f"cleanup_arena_binding_{binding.name.lower()}", arena.path
            )
            raise CleanupFailure(
                finalize_arena_outcome(
                    arena,
                    _cleanup_outcome(
                        CleanupDisposition.UNADDRESSABLE,
                        (binding_issue,),
                        (),
                        arena_binding=binding,
                    ),
                )
            )
        try:
            arena_metadata = os.fstat(arena.descriptor.fileno())
        except OSError as exc:
            issues = [CleanupIssue("cleanup_arena_uninspectable", arena.path, exc)]
            raise CleanupFailure(
                finalize_arena_outcome(
                    arena,
                    _cleanup_outcome(CleanupDisposition.RETAINED, issues, ()),
                )
            ) from exc
        effective_uid = os.geteuid()
        flags = os.O_RDONLY | os.O_CLOEXEC | os.O_DIRECTORY | os.O_NOFOLLOW
        for _ in range(128):
            name = f"{prefix}{secrets.token_hex(token_bytes)}{suffix}"
            try:
                os.mkdir(name, 0o700, dir_fd=arena.descriptor.fileno())
            except FileExistsError:
                continue
            except OSError as exc:
                issue = CleanupIssue(
                    "cleanup_directory_create_failed", arena.path / name, exc
                )
                raise CleanupFailure(
                    finalize_arena_outcome(
                        arena,
                        _cleanup_outcome(
                            CleanupDisposition.RETAINED,
                            (issue,),
                            (),
                        ),
                    )
                ) from exc
            path = arena.path / name
            owned: OwnedDescriptor | None = None
            try:
                created = os.stat(
                    name,
                    dir_fd=arena.descriptor.fileno(),
                    follow_symlinks=False,
                )
                descriptor = os.open(name, flags, dir_fd=arena.descriptor.fileno())
                owned = OwnedDescriptor.take(descriptor, path)
                opened = os.fstat(owned.fileno())
                identity = (created.st_dev, created.st_ino)
                if (
                    not stat.S_ISDIR(created.st_mode)
                    or not stat.S_ISDIR(opened.st_mode)
                    or (opened.st_dev, opened.st_ino) != identity
                    or created.st_dev != arena_metadata.st_dev
                    or created.st_uid != effective_uid
                    or opened.st_uid != effective_uid
                    or stat.S_IMODE(created.st_mode) & 0o077
                    or stat.S_IMODE(opened.st_mode) & 0o077
                ):
                    raise OSError("cleanup directory initial credential is unsafe")
                os.fchmod(owned.fileno(), 0o700)
                configured_path = os.stat(
                    name,
                    dir_fd=arena.descriptor.fileno(),
                    follow_symlinks=False,
                )
                configured_fd = os.fstat(owned.fileno())
                if (
                    not stat.S_ISDIR(configured_path.st_mode)
                    or not stat.S_ISDIR(configured_fd.st_mode)
                    or (configured_path.st_dev, configured_path.st_ino) != identity
                    or (configured_fd.st_dev, configured_fd.st_ino) != identity
                    or configured_path.st_uid != effective_uid
                    or configured_fd.st_uid != effective_uid
                    or stat.S_IMODE(configured_path.st_mode) != 0o700
                    or stat.S_IMODE(configured_fd.st_mode) != 0o700
                ):
                    raise OSError("cleanup directory configured credential is unsafe")
                os.fsync(arena.descriptor.fileno())
                binding = arena.binding()
                if binding is not ArenaBinding.BOUND:
                    issues = [
                        CleanupIssue(
                            f"cleanup_arena_binding_{binding.name.lower()}", arena.path
                        )
                    ]
                    try:
                        owned.close_once()
                    except OSError as exc:
                        issues.append(
                            CleanupIssue(
                                "cleanup_directory_descriptor_close_uncertain",
                                path,
                                exc,
                            )
                        )
                    else:
                        try:
                            os.rmdir(name, dir_fd=arena.descriptor.fileno())
                            os.fsync(arena.descriptor.fileno())
                        except OSError as exc:
                            issues.append(
                                CleanupIssue(
                                    "cleanup_directory_teardown_failed", path, exc
                                )
                            )
                    raise CleanupFailure(
                        finalize_arena_outcome(
                            arena,
                            _cleanup_outcome(
                                CleanupDisposition.UNADDRESSABLE,
                                issues,
                                (),
                                arena_binding=binding,
                            ),
                        )
                    )
                return cls(arena, name, path, owned, identity, effective_uid)
            except CleanupFailure:
                raise
            except BaseException as exc:
                issues = [CleanupIssue("cleanup_directory_setup_failed", path, exc)]
                if owned is not None:
                    try:
                        owned.close_once()
                    except OSError as close_exc:
                        issues.append(
                            CleanupIssue(
                                "cleanup_directory_descriptor_close_uncertain",
                                path,
                                close_exc,
                            )
                        )
                raise CleanupFailure(
                    finalize_arena_outcome(
                        arena,
                        _cleanup_outcome(
                            CleanupDisposition.RETAINED,
                            issues,
                            (path,),
                        ),
                    )
                ) from exc
        issues = [CleanupIssue("cleanup_directory_name_exhausted", arena.path)]
        raise CleanupFailure(
            finalize_arena_outcome(
                arena,
                _cleanup_outcome(CleanupDisposition.RETAINED, issues, ()),
            )
        )

    def fileno(self) -> int:
        return self.descriptor.fileno()

    def finish_empty(self) -> CleanupOutcome:
        issues: list[CleanupIssue] = []
        recovery_paths: list[Path] = []
        binding = self.arena.binding()
        if binding is not ArenaBinding.BOUND:
            issues.append(
                CleanupIssue(
                    f"cleanup_arena_binding_{binding.name.lower()}", self.arena.path
                )
            )
            try:
                self.descriptor.close_once()
            except OSError as exc:
                issues.append(
                    CleanupIssue(
                        "cleanup_directory_descriptor_close_uncertain", self.path, exc
                    )
                )
            return finalize_arena_outcome(
                self.arena,
                _cleanup_outcome(
                    CleanupDisposition.UNADDRESSABLE,
                    issues,
                    (),
                    arena_binding=binding,
                ),
            )
        teardown_permitted = True
        try:
            os.fsync(self.descriptor.fileno())
        except OSError as exc:
            teardown_permitted = False
            issues.append(
                CleanupIssue("cleanup_directory_fsync_failed", self.path, exc)
            )
            recovery_paths.append(self.path)
        try:
            self.descriptor.close_once()
        except OSError as exc:
            teardown_permitted = False
            issues.append(
                CleanupIssue(
                    "cleanup_directory_descriptor_close_uncertain", self.path, exc
                )
            )
            recovery_paths.append(self.path)
        if teardown_permitted:
            binding = self.arena.binding()
            if binding is not ArenaBinding.BOUND:
                issues.append(
                    CleanupIssue(
                        f"cleanup_arena_binding_{binding.name.lower()}", self.arena.path
                    )
                )
                return finalize_arena_outcome(
                    self.arena,
                    _cleanup_outcome(
                        CleanupDisposition.UNADDRESSABLE,
                        issues,
                        (),
                        arena_binding=binding,
                    ),
                )
            try:
                observed = os.stat(
                    self.name,
                    dir_fd=self.arena.descriptor.fileno(),
                    follow_symlinks=False,
                )
                if (
                    not stat.S_ISDIR(observed.st_mode)
                    or (observed.st_dev, observed.st_ino) != self.identity
                    or observed.st_uid != self.owner_uid
                    or stat.S_IMODE(observed.st_mode) != 0o700
                ):
                    raise OSError("cleanup directory credential changed")
                os.rmdir(self.name, dir_fd=self.arena.descriptor.fileno())
                try:
                    os.stat(
                        self.name,
                        dir_fd=self.arena.descriptor.fileno(),
                        follow_symlinks=False,
                    )
                except FileNotFoundError:
                    pass
                else:
                    raise OSError("cleanup directory remained after teardown")
            except OSError as exc:
                issues.append(
                    CleanupIssue("cleanup_directory_teardown_failed", self.path, exc)
                )
                recovery_paths.append(self.path)
            else:
                binding = self.arena.binding()
                if binding is not ArenaBinding.BOUND:
                    issues.append(
                        CleanupIssue(
                            f"cleanup_arena_binding_{binding.name.lower()}",
                            self.arena.path,
                        )
                    )
                    return finalize_arena_outcome(
                        self.arena,
                        _cleanup_outcome(
                            CleanupDisposition.UNADDRESSABLE,
                            issues,
                            (),
                            arena_binding=binding,
                        ),
                    )
                try:
                    os.fsync(self.arena.descriptor.fileno())
                except OSError as exc:
                    issues.append(
                        CleanupIssue("cleanup_arena_fsync_failed", self.arena.path, exc)
                    )
        if issues:
            outcome = _cleanup_outcome(
                CleanupDisposition.RETAINED, issues, recovery_paths
            )
        else:
            outcome = _cleanup_outcome(CleanupDisposition.REMOVED, (), ())
        return finalize_arena_outcome(self.arena, outcome)


@dataclass(slots=True)
class CleanupQuarantine:
    """Descriptor-held claim-first cleanup under one anchored parent.

    The quarantine is private only against actors that cannot actually mutate a
    mode-0700 directory owned by the effective UID. Same-credential, root,
    equivalent-capability, ACL-authorized, and continuing private-directory
    writers remain outside this portable contract; this is not pathname CAS.
    """

    source: DirectoryAnchor
    arena: CleanupArena
    public_name: str
    quarantine_name: str
    quarantine_path: Path
    claimed_name: str
    descriptor: OwnedDescriptor
    identity: tuple[int, int]
    owner_uid: int
    claim_kind: ClaimKind = ClaimKind.REGULAR_FILE
    phase: CleanupPhase = CleanupPhase.READY
    _issues: list[CleanupIssue] = field(default_factory=list)
    _recovery_paths: list[Path] = field(default_factory=list)
    _candidate_paths: list[Path] = field(default_factory=list)
    _arena_binding: ArenaBinding = ArenaBinding.BOUND
    _public_candidate: PublicCandidate = PublicCandidate.ABSENT

    @property
    def claimed_path(self) -> Path:
        return self.quarantine_path / self.claimed_name

    @property
    def public_path(self) -> Path:
        return self.source.path / self.public_name

    @classmethod
    def create(
        cls,
        source: DirectoryAnchor,
        public_name: str,
        *,
        quarantine_prefix: str,
        quarantine_suffix: str,
        token_bytes: int = 16,
        claimed_name: str = "claimed",
        arena_name: str | None = None,
        arena: CleanupArena | None = None,
        claim_kind: ClaimKind = ClaimKind.REGULAR_FILE,
    ) -> "CleanupQuarantine":
        public_name = _cleanup_component(public_name, "public cleanup name")
        if public_name.startswith(".zynum-cleanup-v"):
            raise ValueError("public cleanup name uses the reserved arena prefix")
        claimed_name = _cleanup_component(claimed_name, "claimed cleanup name")
        if type(quarantine_prefix) is not str or type(quarantine_suffix) is not str:
            raise TypeError("quarantine prefix and suffix must be strings")
        if type(token_bytes) is not int or not 1 <= token_bytes <= 64:
            raise ValueError("quarantine token_bytes must be between 1 and 64")
        _cleanup_component(
            f"{quarantine_prefix}{'0' * (token_bytes * 2)}{quarantine_suffix}",
            "quarantine name",
        )
        if os.name != "posix":
            public_path = source.path / public_name
            issue = CleanupIssue("cleanup_posix_semantics_unavailable", public_path)
            raise CleanupFailure(
                _cleanup_outcome(
                    CleanupDisposition.UNADDRESSABLE,
                    (issue,),
                    (),
                    candidate_paths=(public_path,),
                    arena_binding=ArenaBinding.UNKNOWN,
                    public_candidate=PublicCandidate.UNKNOWN,
                )
            )

        if not isinstance(claim_kind, ClaimKind):
            raise TypeError("cleanup claim_kind must be ClaimKind")
        if arena is not None and not isinstance(arena, CleanupArena):
            raise TypeError("cleanup quarantine arena must be CleanupArena")
        try:
            arena = (
                CleanupArena.open(source, arena_name)
                if arena is None
                else arena.duplicate()
            )
        except CleanupFailure as exc:
            raise CleanupFailure(
                _preclaim_outcome(exc.outcome, source, public_name)
            ) from exc
        try:
            arena_metadata = os.fstat(arena.descriptor.fileno())
            source_metadata = os.fstat(source.descriptor)
        except OSError as exc:
            issues = [CleanupIssue("cleanup_anchor_uninspectable", arena.path, exc)]
            outcome = finalize_arena_outcome(
                arena,
                _cleanup_outcome(CleanupDisposition.RETAINED, issues, ()),
            )
            raise CleanupFailure(
                _preclaim_outcome(outcome, source, public_name)
            ) from exc
        if arena_metadata.st_dev != source_metadata.st_dev:
            issues = [
                CleanupIssue(
                    "cleanup_arena_source_device_mismatch",
                    arena.path,
                )
            ]
            outcome = finalize_arena_outcome(
                arena,
                _cleanup_outcome(CleanupDisposition.RETAINED, issues, ()),
            )
            raise CleanupFailure(_preclaim_outcome(outcome, source, public_name))
        binding = arena.binding()
        if binding is not ArenaBinding.BOUND:
            issues = [
                CleanupIssue(
                    f"cleanup_arena_binding_{binding.name.lower()}", arena.path
                )
            ]
            outcome = finalize_arena_outcome(
                arena,
                _cleanup_outcome(
                    CleanupDisposition.UNADDRESSABLE,
                    issues,
                    (),
                    arena_binding=binding,
                ),
            )
            raise CleanupFailure(_preclaim_outcome(outcome, source, public_name))

        directory_flags = os.O_RDONLY | os.O_CLOEXEC | os.O_DIRECTORY | os.O_NOFOLLOW
        effective_uid = os.geteuid()
        for _ in range(128):
            quarantine_name = (
                f"{quarantine_prefix}{secrets.token_hex(token_bytes)}"
                f"{quarantine_suffix}"
            )
            try:
                _cleanup_component(quarantine_name, "quarantine name")
            except ValueError as exc:
                issue = CleanupIssue(
                    "cleanup_quarantine_name_invalid", source.path / public_name, exc
                )
                outcome = finalize_arena_outcome(
                    arena,
                    _cleanup_outcome(
                        CleanupDisposition.RETAINED,
                        (issue,),
                        (),
                    ),
                )
                raise CleanupFailure(
                    _preclaim_outcome(outcome, source, public_name)
                ) from exc
            try:
                os.mkdir(
                    quarantine_name,
                    0o700,
                    dir_fd=arena.descriptor.fileno(),
                )
            except FileExistsError:
                continue
            except OSError as exc:
                path = arena.path / quarantine_name
                issue = CleanupIssue("cleanup_quarantine_create_failed", path, exc)
                outcome = finalize_arena_outcome(
                    arena,
                    _cleanup_outcome(
                        CleanupDisposition.RETAINED,
                        (issue,),
                        (),
                    ),
                )
                raise CleanupFailure(
                    _preclaim_outcome(outcome, source, public_name)
                ) from exc

            path = arena.path / quarantine_name
            owned: OwnedDescriptor | None = None
            try:
                created = os.stat(
                    quarantine_name,
                    dir_fd=arena.descriptor.fileno(),
                    follow_symlinks=False,
                )
                frozen_identity = (created.st_dev, created.st_ino)
                descriptor = os.open(
                    quarantine_name,
                    directory_flags,
                    dir_fd=arena.descriptor.fileno(),
                )
                owned = OwnedDescriptor.take(descriptor, path)
                opened = os.fstat(owned.fileno())
                if (
                    not stat.S_ISDIR(created.st_mode)
                    or not stat.S_ISDIR(opened.st_mode)
                    or (opened.st_dev, opened.st_ino) != frozen_identity
                    or created.st_dev != arena_metadata.st_dev
                    or created.st_uid != effective_uid
                    or opened.st_uid != effective_uid
                    or stat.S_IMODE(created.st_mode) & 0o077
                    or stat.S_IMODE(opened.st_mode) & 0o077
                ):
                    raise OSError("cleanup quarantine initial credential is unsafe")
                os.fchmod(owned.fileno(), 0o700)
                configured_path = os.stat(
                    quarantine_name,
                    dir_fd=arena.descriptor.fileno(),
                    follow_symlinks=False,
                )
                configured_fd = os.fstat(owned.fileno())
                if (
                    not stat.S_ISDIR(configured_path.st_mode)
                    or not stat.S_ISDIR(configured_fd.st_mode)
                    or (configured_path.st_dev, configured_path.st_ino)
                    != frozen_identity
                    or (configured_fd.st_dev, configured_fd.st_ino) != frozen_identity
                    or configured_path.st_uid != effective_uid
                    or configured_fd.st_uid != effective_uid
                    or stat.S_IMODE(configured_path.st_mode) != 0o700
                    or stat.S_IMODE(configured_fd.st_mode) != 0o700
                ):
                    raise OSError("cleanup quarantine configured credential is unsafe")
                os.fsync(arena.descriptor.fileno())
                binding = arena.binding()
                if binding is not ArenaBinding.BOUND:
                    issues = [
                        CleanupIssue(
                            f"cleanup_arena_binding_{binding.name.lower()}", arena.path
                        )
                    ]
                    try:
                        owned.close_once()
                    except OSError as exc:
                        issues.append(
                            CleanupIssue(
                                "cleanup_quarantine_descriptor_close_uncertain",
                                path,
                                exc,
                            )
                        )
                    else:
                        try:
                            os.rmdir(
                                quarantine_name,
                                dir_fd=arena.descriptor.fileno(),
                            )
                            os.fsync(arena.descriptor.fileno())
                        except OSError as exc:
                            issues.append(
                                CleanupIssue(
                                    "cleanup_quarantine_teardown_failed", path, exc
                                )
                            )
                    outcome = finalize_arena_outcome(
                        arena,
                        _cleanup_outcome(
                            CleanupDisposition.UNADDRESSABLE,
                            issues,
                            (),
                            arena_binding=binding,
                        ),
                    )
                    raise CleanupFailure(
                        _preclaim_outcome(outcome, source, public_name)
                    )
                return cls(
                    source=source,
                    arena=arena,
                    public_name=public_name,
                    quarantine_name=quarantine_name,
                    quarantine_path=path,
                    claimed_name=claimed_name,
                    descriptor=owned,
                    identity=frozen_identity,
                    owner_uid=effective_uid,
                    claim_kind=claim_kind,
                )
            except CleanupFailure:
                raise
            except BaseException as exc:
                issues = [CleanupIssue("cleanup_quarantine_setup_failed", path, exc)]
                if owned is not None:
                    try:
                        owned.close_once()
                    except OSError as close_exc:
                        issues.append(
                            CleanupIssue(
                                "cleanup_quarantine_descriptor_close_uncertain",
                                path,
                                close_exc,
                            )
                        )
                outcome = finalize_arena_outcome(
                    arena,
                    _cleanup_outcome(
                        CleanupDisposition.RETAINED,
                        issues,
                        (path,),
                    ),
                )
                raise CleanupFailure(
                    _preclaim_outcome(outcome, source, public_name)
                ) from exc
        issue = CleanupIssue(
            "cleanup_quarantine_name_exhausted", source.path / public_name
        )
        issues = [issue]
        outcome = finalize_arena_outcome(
            arena,
            _cleanup_outcome(CleanupDisposition.RETAINED, issues, ()),
        )
        raise CleanupFailure(_preclaim_outcome(outcome, source, public_name))

    def _retain(
        self,
        code: str,
        path: Path,
        error: BaseException | None = None,
        *,
        recovery_paths: Iterable[Path] = (),
    ) -> CleanupFailure:
        self.phase = CleanupPhase.RETAINED
        self._issues.append(CleanupIssue(code, path, error))
        self._recovery_paths.extend(recovery_paths)
        return CleanupFailure(self._outcome(CleanupDisposition.RETAINED))

    def _retain_public_candidate(
        self,
        code: str,
        state: PublicCandidate,
        error: BaseException | None = None,
        *,
        recovery_paths: Iterable[Path] = (),
    ) -> CleanupFailure:
        self._candidate_paths.append(self.public_path)
        if (
            self._public_candidate is not PublicCandidate.UNKNOWN
            or state is PublicCandidate.UNKNOWN
        ):
            self._public_candidate = state
        return self._retain(
            code,
            self.public_path,
            error,
            recovery_paths=recovery_paths,
        )

    def _retain_preclaim(
        self,
        code: str,
        error: BaseException | None = None,
    ) -> CleanupFailure:
        binding = self.arena.binding()
        self.phase = (
            CleanupPhase.RETAINED
            if binding is ArenaBinding.BOUND
            else CleanupPhase.UNADDRESSABLE
        )
        self._issues.append(CleanupIssue(code, self.public_path, error))
        if binding is ArenaBinding.BOUND:
            self._recovery_paths.append(self.quarantine_path)
        else:
            self._issues.append(
                CleanupIssue(
                    f"cleanup_arena_binding_{binding.name.lower()}", self.arena.path
                )
            )
        disposition = (
            CleanupDisposition.RETAINED
            if binding is ArenaBinding.BOUND
            else CleanupDisposition.UNADDRESSABLE
        )
        outcome = _preclaim_outcome(
            _cleanup_outcome(
                disposition,
                self._issues,
                self._recovery_paths,
                arena_binding=binding,
                arena_identity=self.arena.identity,
                recovery_anchor_identity=self.arena.anchor.identity,
            ),
            self.source,
            self.public_name,
        )
        self._candidate_paths.extend(outcome.candidate_paths)
        self._arena_binding = outcome.arena_binding
        self._public_candidate = outcome.public_candidate
        return CleanupFailure(outcome)

    def _outcome(self, disposition: CleanupDisposition) -> CleanupOutcome:
        return _cleanup_outcome(
            disposition,
            self._issues,
            self._recovery_paths,
            candidate_paths=self._candidate_paths,
            arena_binding=self._arena_binding,
            public_candidate=self._public_candidate,
            arena_identity=self.arena.identity,
            recovery_anchor_identity=self.arena.anchor.identity,
        )

    def _with_arena_close(self, outcome: CleanupOutcome) -> CleanupOutcome:
        finalized = finalize_arena_outcome(self.arena, outcome)
        self._issues[:] = finalized.issues
        self._arena_binding = finalized.arena_binding
        if finalized.disposition is CleanupDisposition.UNADDRESSABLE:
            self.phase = CleanupPhase.UNADDRESSABLE
        elif finalized.disposition is CleanupDisposition.RETAINED:
            self.phase = CleanupPhase.RETAINED
        return finalized

    def _public_is_absent(self) -> bool:
        try:
            os.stat(
                self.public_name,
                dir_fd=self.source.descriptor,
                follow_symlinks=False,
            )
        except FileNotFoundError:
            return True
        return False

    def claim(self) -> None:
        if self.phase is not CleanupPhase.READY:
            raise RuntimeError("cleanup claim requires READY phase")
        binding = self.arena.binding()
        if binding is not ArenaBinding.BOUND:
            self.phase = CleanupPhase.UNADDRESSABLE
            issue = CleanupIssue(
                f"cleanup_arena_binding_{binding.name.lower()}", self.arena.path
            )
            self._issues.append(issue)
            outcome = _cleanup_outcome(
                CleanupDisposition.UNADDRESSABLE,
                self._issues,
                (),
                arena_binding=binding,
                arena_identity=self.arena.identity,
                recovery_anchor_identity=self.arena.anchor.identity,
            )
            outcome = _preclaim_outcome(outcome, self.source, self.public_name)
            self._candidate_paths.extend(outcome.candidate_paths)
            self._arena_binding = outcome.arena_binding
            self._public_candidate = outcome.public_candidate
            raise CleanupFailure(outcome)
        try:
            os.rename(
                self.public_name,
                self.claimed_name,
                src_dir_fd=self.source.descriptor,
                dst_dir_fd=self.descriptor.fileno(),
            )
        except FileNotFoundError as exc:
            try:
                absent = self._public_is_absent()
            except OSError as observe_exc:
                raise self._retain_preclaim(
                    "cleanup_public_absence_uninspectable", observe_exc
                ) from exc
            if not absent:
                raise self._retain_preclaim("cleanup_public_name_reappeared") from exc
            self.phase = CleanupPhase.ABSENT
            return
        except OSError as exc:
            raise self._retain_preclaim("cleanup_claim_failed", exc) from exc

        self.phase = CleanupPhase.CLAIMED_UNSYNCED
        failures: list[CleanupIssue] = []
        try:
            os.fsync(self.descriptor.fileno())
        except OSError as exc:
            failures.append(
                CleanupIssue(
                    "cleanup_claim_destination_fsync_failed", self.claimed_path, exc
                )
            )
        try:
            os.fsync(self.source.descriptor)
        except OSError as exc:
            failures.append(
                CleanupIssue("cleanup_claim_source_fsync_failed", self.source.path, exc)
            )
        binding = self.arena.binding()
        if binding is not ArenaBinding.BOUND:
            self.phase = CleanupPhase.UNADDRESSABLE
            self._arena_binding = binding
            self._issues.extend(failures)
            self._issues.append(
                CleanupIssue(
                    f"cleanup_arena_binding_{binding.name.lower()}", self.arena.path
                )
            )
            raise CleanupFailure(
                _cleanup_outcome(
                    CleanupDisposition.UNADDRESSABLE,
                    self._issues,
                    (),
                    arena_binding=binding,
                    arena_identity=self.arena.identity,
                    recovery_anchor_identity=self.arena.anchor.identity,
                )
            )
        if failures:
            self.phase = CleanupPhase.RETAINED
            self._issues.extend(failures)
            self._recovery_paths.extend((self.claimed_path, self.quarantine_path))
            raise CleanupFailure(self._outcome(CleanupDisposition.RETAINED))
        self.phase = CleanupPhase.CLAIMED_DURABLE

    def verify_claimed(
        self,
        verifier: Callable[[int, os.stat_result], ClaimVerification],
    ) -> ClaimVerification:
        if self.phase is not CleanupPhase.CLAIMED_DURABLE:
            raise RuntimeError("claimed verification requires durable claim")
        try:
            if not self._public_is_absent():
                raise self._retain_public_candidate(
                    "cleanup_public_name_reappeared",
                    PublicCandidate.PRESENT,
                    recovery_paths=(self.claimed_path,),
                )
        except CleanupFailure:
            raise
        except OSError as exc:
            raise self._retain_public_candidate(
                "cleanup_public_absence_uninspectable",
                PublicCandidate.UNKNOWN,
                exc,
                recovery_paths=(self.claimed_path,),
            ) from exc

        claimed_owner: OwnedDescriptor | None = None
        verification = ClaimVerification.UNKNOWN
        try:
            claimed_path_metadata = os.stat(
                self.claimed_name,
                dir_fd=self.descriptor.fileno(),
                follow_symlinks=False,
            )
            claimed_flags = os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW
            if self.claim_kind is ClaimKind.REGULAR_FILE:
                claimed_flags |= os.O_NONBLOCK
            else:
                claimed_flags |= os.O_DIRECTORY
            claimed_fd = os.open(
                self.claimed_name,
                claimed_flags,
                dir_fd=self.descriptor.fileno(),
            )
            claimed_owner = OwnedDescriptor.take(claimed_fd, self.claimed_path)
            opened = os.fstat(claimed_owner.fileno())
            expected_kind_matches = (
                stat.S_ISREG(claimed_path_metadata.st_mode)
                and stat.S_ISREG(opened.st_mode)
                if self.claim_kind is ClaimKind.REGULAR_FILE
                else stat.S_ISDIR(claimed_path_metadata.st_mode)
                and stat.S_ISDIR(opened.st_mode)
            )
            if not expected_kind_matches or (
                claimed_path_metadata.st_dev,
                claimed_path_metadata.st_ino,
            ) != (opened.st_dev, opened.st_ino):
                verification = ClaimVerification.FOREIGN
            else:
                verification = verifier(claimed_owner.fileno(), claimed_path_metadata)
                if not isinstance(verification, ClaimVerification):
                    verification = ClaimVerification.UNKNOWN
        except BaseException as exc:
            if isinstance(exc, CleanupFailure):
                raise
            self._issues.append(
                CleanupIssue("cleanup_claimed_uninspectable", self.claimed_path, exc)
            )
            verification = ClaimVerification.UNKNOWN
        finally:
            if claimed_owner is not None:
                try:
                    claimed_owner.close_once()
                except OSError as exc:
                    raise self._retain(
                        "cleanup_claimed_descriptor_close_uncertain",
                        self.claimed_path,
                        exc,
                        recovery_paths=(self.claimed_path, self.quarantine_path),
                    ) from exc

        if verification is ClaimVerification.MATCH:
            self.phase = CleanupPhase.VERIFIED
        else:
            self.phase = CleanupPhase.RETAINED
            code = (
                "cleanup_claimed_foreign"
                if verification is ClaimVerification.FOREIGN
                else "cleanup_claimed_unknown"
            )
            self._issues.append(CleanupIssue(code, self.claimed_path))
            self._recovery_paths.extend((self.claimed_path, self.quarantine_path))
        return verification

    def remove_verified_claim(self, *, expect_public_absent: bool = True) -> None:
        if self.phase is not CleanupPhase.VERIFIED:
            raise RuntimeError("cleanup removal requires verified claim")
        binding = self.arena.binding()
        if binding is not ArenaBinding.BOUND:
            self.phase = CleanupPhase.UNADDRESSABLE
            self._arena_binding = binding
            self._issues.append(
                CleanupIssue(
                    f"cleanup_arena_binding_{binding.name.lower()}", self.arena.path
                )
            )
            raise CleanupFailure(
                _cleanup_outcome(
                    CleanupDisposition.UNADDRESSABLE,
                    self._issues,
                    (),
                    arena_binding=binding,
                    arena_identity=self.arena.identity,
                    recovery_anchor_identity=self.arena.anchor.identity,
                )
            )
        try:
            if expect_public_absent and not self._public_is_absent():
                raise self._retain_public_candidate(
                    "cleanup_public_name_reappeared",
                    PublicCandidate.PRESENT,
                    recovery_paths=(self.claimed_path,),
                )
            if self.claim_kind is ClaimKind.REGULAR_FILE:
                os.unlink(self.claimed_name, dir_fd=self.descriptor.fileno())
            else:
                os.rmdir(self.claimed_name, dir_fd=self.descriptor.fileno())
            try:
                os.stat(
                    self.claimed_name,
                    dir_fd=self.descriptor.fileno(),
                    follow_symlinks=False,
                )
            except FileNotFoundError:
                pass
            else:
                raise OSError("claimed cleanup name remained after removal")
        except CleanupFailure:
            raise
        except OSError as exc:
            raise self._retain(
                "cleanup_claim_remove_failed",
                self.claimed_path,
                exc,
                recovery_paths=(self.claimed_path, self.quarantine_path),
            ) from exc
        self.phase = CleanupPhase.CLAIM_REMOVED

    def finish(self, *, expect_public_absent: bool) -> CleanupOutcome:
        if self.phase is CleanupPhase.UNADDRESSABLE:
            try:
                self.descriptor.close_once()
            except OSError as exc:
                self._issues.append(
                    CleanupIssue(
                        "cleanup_quarantine_descriptor_close_uncertain",
                        self.quarantine_path,
                        exc,
                    )
                )
            return self._with_arena_close(
                _cleanup_outcome(
                    CleanupDisposition.UNADDRESSABLE,
                    self._issues,
                    (),
                    candidate_paths=self._candidate_paths,
                    arena_binding=self._arena_binding,
                    public_candidate=self._public_candidate,
                )
            )
        terminal_disposition = (
            CleanupDisposition.ABSENT
            if self.phase is CleanupPhase.ABSENT
            else CleanupDisposition.REMOVED
        )
        teardown_permitted = self.phase in {
            CleanupPhase.ABSENT,
            CleanupPhase.CLAIM_REMOVED,
        }
        if self.phase not in {
            CleanupPhase.ABSENT,
            CleanupPhase.CLAIM_REMOVED,
            CleanupPhase.RETAINED,
        }:
            self.phase = CleanupPhase.RETAINED
            self._issues.append(
                CleanupIssue("cleanup_finished_incomplete", self.quarantine_path)
            )
            self._recovery_paths.append(self.quarantine_path)

        if teardown_permitted:
            try:
                os.fsync(self.descriptor.fileno())
            except OSError as exc:
                teardown_permitted = False
                self.phase = CleanupPhase.RETAINED
                self._issues.append(
                    CleanupIssue(
                        "cleanup_quarantine_fsync_failed", self.quarantine_path, exc
                    )
                )
                self._recovery_paths.append(self.quarantine_path)

        try:
            self.descriptor.close_once()
        except OSError as exc:
            self.phase = CleanupPhase.RETAINED
            self._issues.append(
                CleanupIssue(
                    "cleanup_quarantine_descriptor_close_uncertain",
                    self.quarantine_path,
                    exc,
                )
            )
            self._recovery_paths.append(self.quarantine_path)
            return self._with_arena_close(self._outcome(CleanupDisposition.RETAINED))

        if self.phase is CleanupPhase.RETAINED or not teardown_permitted:
            return self._with_arena_close(self._outcome(CleanupDisposition.RETAINED))
        self.phase = CleanupPhase.CLOSED

        binding = self.arena.binding()
        if binding is not ArenaBinding.BOUND:
            self.phase = CleanupPhase.UNADDRESSABLE
            self._issues.append(
                CleanupIssue(
                    f"cleanup_arena_binding_{binding.name.lower()}", self.arena.path
                )
            )
            return self._with_arena_close(
                _cleanup_outcome(
                    CleanupDisposition.UNADDRESSABLE,
                    self._issues,
                    (),
                    arena_binding=binding,
                )
            )

        try:
            observed = os.stat(
                self.quarantine_name,
                dir_fd=self.arena.descriptor.fileno(),
                follow_symlinks=False,
            )
            if (
                not stat.S_ISDIR(observed.st_mode)
                or (observed.st_dev, observed.st_ino) != self.identity
                or observed.st_uid != self.owner_uid
                or stat.S_IMODE(observed.st_mode) != 0o700
            ):
                raise OSError("cleanup quarantine credential changed before teardown")
            os.rmdir(
                self.quarantine_name,
                dir_fd=self.arena.descriptor.fileno(),
            )
            try:
                os.stat(
                    self.quarantine_name,
                    dir_fd=self.arena.descriptor.fileno(),
                    follow_symlinks=False,
                )
            except FileNotFoundError:
                pass
            else:
                raise OSError("cleanup quarantine remained after teardown")
        except OSError as exc:
            self.phase = CleanupPhase.RETAINED
            self._issues.append(
                CleanupIssue(
                    "cleanup_quarantine_teardown_failed", self.quarantine_path, exc
                )
            )
            self._recovery_paths.append(self.quarantine_path)
            return self._with_arena_close(self._outcome(CleanupDisposition.RETAINED))
        try:
            os.fsync(self.arena.descriptor.fileno())
        except OSError as exc:
            self.phase = CleanupPhase.RETAINED
            self._issues.append(
                CleanupIssue("cleanup_arena_fsync_failed", self.arena.path, exc)
            )
            return self._with_arena_close(self._outcome(CleanupDisposition.RETAINED))

        if expect_public_absent:
            try:
                absent = self._public_is_absent()
            except OSError as exc:
                self.phase = CleanupPhase.RETAINED
                self._issues.append(
                    CleanupIssue(
                        "cleanup_public_absence_uninspectable", self.public_path, exc
                    )
                )
                self._candidate_paths.append(self.public_path)
                self._public_candidate = PublicCandidate.UNKNOWN
                return self._with_arena_close(
                    self._outcome(CleanupDisposition.RETAINED)
                )
            if not absent:
                self.phase = CleanupPhase.RETAINED
                self._issues.append(
                    CleanupIssue("cleanup_public_name_reappeared", self.public_path)
                )
                self._candidate_paths.append(self.public_path)
                self._public_candidate = PublicCandidate.PRESENT
                return self._with_arena_close(
                    self._outcome(CleanupDisposition.RETAINED)
                )

        self.phase = (
            CleanupPhase.ABSENT
            if terminal_disposition is CleanupDisposition.ABSENT
            else CleanupPhase.REMOVED
        )
        return self._with_arena_close(self._outcome(terminal_disposition))


def claim_and_remove(
    source: DirectoryAnchor,
    public_name: str,
    verifier: Callable[[int, os.stat_result], ClaimVerification],
    *,
    quarantine_prefix: str,
    quarantine_suffix: str,
    token_bytes: int = 16,
    claimed_name: str = "claimed",
    expect_public_absent: bool = True,
    arena_name: str | None = None,
    arena: CleanupArena | None = None,
    claim_kind: ClaimKind = ClaimKind.REGULAR_FILE,
) -> CleanupOutcome:
    quarantine = CleanupQuarantine.create(
        source,
        public_name,
        quarantine_prefix=quarantine_prefix,
        quarantine_suffix=quarantine_suffix,
        token_bytes=token_bytes,
        claimed_name=claimed_name,
        arena_name=arena_name,
        arena=arena,
        claim_kind=claim_kind,
    )
    try:
        quarantine.claim()
        if quarantine.phase is CleanupPhase.ABSENT:
            return quarantine.finish(expect_public_absent=expect_public_absent)
        verification = quarantine.verify_claimed(verifier)
        if verification is ClaimVerification.MATCH:
            quarantine.remove_verified_claim()
    except CleanupFailure:
        pass
    return quarantine.finish(expect_public_absent=expect_public_absent)


@dataclass(frozen=True, slots=True)
class DirectoryIdentity:
    device: int
    inode: int
    mode: int


@dataclass(frozen=True, slots=True)
class NodeIdentity:
    device: int
    inode: int
    size: int
    mode: int
    mtime_ns: int
    ctime_ns: int


@dataclass(frozen=True, slots=True)
class FrozenDirectory:
    path: str
    identity: DirectoryIdentity
    children: tuple[str, ...] | None = None
    excluded_parts: frozenset[str] = frozenset()
    structure_children: tuple[tuple[str, str], ...] | None = None
    structure_identity: NodeIdentity | None = None


@dataclass(frozen=True, slots=True)
class DirectoryStructureLimits:
    max_directories: int
    max_entries: int
    max_depth: int
    max_total_name_bytes: int
    max_total_structure_bytes: int


DEFAULT_DIRECTORY_STRUCTURE_LIMITS = DirectoryStructureLimits(
    max_directories=100_000,
    max_entries=1_000_000,
    max_depth=128,
    max_total_name_bytes=64 * 1024 * 1024,
    max_total_structure_bytes=256 * 1024 * 1024,
)


@dataclass(frozen=True, slots=True)
class RegularFileLimits:
    max_file_bytes: int
    max_round_bytes: int
    max_cached_file_bytes: int
    max_cached_total_bytes: int


DEFAULT_REGULAR_FILE_LIMITS = RegularFileLimits(
    max_file_bytes=512 * 1024 * 1024,
    max_round_bytes=4 * 1024 * 1024 * 1024,
    max_cached_file_bytes=64 * 1024 * 1024,
    max_cached_total_bytes=256 * 1024 * 1024,
)


@dataclass(slots=True)
class _RegularFileRound:
    limits: RegularFileLimits
    read_bytes: int = 0
    cached_bytes: int = 0

    def admit(self, size: int, *, capture_bytes: bool, limit: int) -> None:
        if size < 0:
            raise RepositorySnapshotError("regular file has an invalid frozen size")
        if size > limit:
            raise RepositorySnapshotError("regular file exceeds its input limit")
        if self.read_bytes + size > self.limits.max_round_bytes:
            raise RepositorySnapshotError(
                "regular-file round exceeds its cumulative input limit"
            )
        if capture_bytes:
            if size > self.limits.max_cached_file_bytes:
                raise RepositorySnapshotError(
                    "regular file exceeds its cache input limit"
                )
            if self.cached_bytes + size > self.limits.max_cached_total_bytes:
                raise RepositorySnapshotError(
                    "regular-file round exceeds its cumulative cache limit"
                )
        self.read_bytes += size
        if capture_bytes:
            self.cached_bytes += size


@dataclass(slots=True)
class _DirectoryStructureRound:
    limits: DirectoryStructureLimits
    directories: int = 0
    entries: int = 0
    total_name_bytes: int = 0
    total_structure_bytes: int = 0

    def enter_directory(self, depth: int) -> None:
        if depth > self.limits.max_depth:
            raise RepositorySnapshotError(
                "repository directory structure exceeds max_depth"
            )
        self.directories += 1
        if self.directories > self.limits.max_directories:
            raise RepositorySnapshotError(
                "repository directory structure exceeds max_directories"
            )

    def admit_entry(self, encoded_name: bytes, encoded_path: bytes) -> None:
        self.entries += 1
        if self.entries > self.limits.max_entries:
            raise RepositorySnapshotError(
                "repository directory structure exceeds max_entries"
            )
        self.total_name_bytes += len(encoded_name)
        if self.total_name_bytes > self.limits.max_total_name_bytes:
            raise RepositorySnapshotError(
                "repository directory structure exceeds max_total_name_bytes"
            )
        self.total_structure_bytes += len(encoded_path) + 16
        if self.total_structure_bytes > self.limits.max_total_structure_bytes:
            raise RepositorySnapshotError(
                "repository directory structure exceeds max_total_structure_bytes"
            )


@dataclass(frozen=True, slots=True)
class FrozenDirectoryStructure:
    directories: tuple[FrozenDirectory, ...]
    excluded_parts: frozenset[str]
    limits: DirectoryStructureLimits


@dataclass(frozen=True, slots=True)
class AbsenceProof:
    path: str
    missing_prefix: str


@dataclass(frozen=True, slots=True)
class FrozenNode:
    path: str
    kind: str
    identity: NodeIdentity | DirectoryIdentity
    parents: tuple[FrozenDirectory, ...]
    sha256: str | None = None
    bytes: bytes | None = None
    symlink_target: bytes | None = None

    @property
    def content_sha256(self) -> str:
        if self.sha256 is None:
            raise AttributeError("non-regular nodes have no content digest")
        return self.sha256

    @property
    def content(self) -> bytes | None:
        return self.bytes

    @property
    def raw_symlink_target(self) -> bytes | None:
        return self.symlink_target

    @property
    def device(self) -> int:
        return self.identity.device

    @property
    def inode(self) -> int:
        return self.identity.inode

    @property
    def mode(self) -> int:
        return self.identity.mode

    @property
    def size(self) -> int:
        if not isinstance(self.identity, NodeIdentity):
            return 0
        return self.identity.size

    @property
    def mtime_ns(self) -> int:
        if not isinstance(self.identity, NodeIdentity):
            return 0
        return self.identity.mtime_ns

    @property
    def directory_identities(self) -> tuple[tuple[int, int, int], ...]:
        return tuple(
            (parent.identity.device, parent.identity.inode, parent.identity.mode)
            for parent in self.parents
        )


@dataclass(frozen=True, slots=True)
class TreeSnapshot:
    root: Path
    root_identity: DirectoryIdentity
    nodes: tuple[FrozenNode, ...]
    directories: tuple[FrozenDirectory, ...]
    absences: tuple[AbsenceProof, ...]
    structure: FrozenDirectoryStructure | None = None

    def node(self, path: str) -> FrozenNode:
        relative_path(path)
        for node in self.nodes:
            if node.path == path:
                return node
        raise KeyError(path)

    @property
    def regular_nodes(self) -> tuple[FrozenNode, ...]:
        return tuple(node for node in self.nodes if node.kind == "regular")


def relative_path(path: str) -> tuple[str, ...]:
    """Validate and split one strict, relative POSIX path."""
    if type(path) is not str or not path or "\0" in path or "\\" in path:
        raise RepositorySnapshotError(f"invalid repository-relative path: {path!r}")
    pure = PurePosixPath(path)
    parts = pure.parts
    if (
        pure.is_absolute()
        or pure.as_posix() != path
        or not parts
        or any(part in {"", ".", ".."} for part in parts)
    ):
        raise RepositorySnapshotError(f"invalid repository-relative path: {path!r}")
    try:
        path.encode("utf-8", errors="strict")
    except UnicodeEncodeError as exc:
        raise RepositorySnapshotError(
            f"repository-relative path is not valid UTF-8: {path!r}"
        ) from exc
    return parts


def _require_descriptor_relative_io() -> None:
    required_dir_fd = {os.open, os.stat, os.readlink}
    if (
        not hasattr(os, "O_NOFOLLOW")
        or not hasattr(os, "O_DIRECTORY")
        or not required_dir_fd.issubset(os.supports_dir_fd)
        or os.listdir not in os.supports_fd
        or os.scandir not in os.supports_fd
    ):
        raise RuntimeError(
            "repository snapshots require descriptor-relative no-follow filesystem I/O"
        )


def _directory_identity(metadata: os.stat_result) -> DirectoryIdentity:
    return DirectoryIdentity(metadata.st_dev, metadata.st_ino, metadata.st_mode)


def _node_identity(metadata: os.stat_result) -> NodeIdentity:
    return NodeIdentity(
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_size,
        metadata.st_mode,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
    )


def _directory_flags() -> int:
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    return flags


def _regular_flags() -> int:
    flags = os.O_RDONLY | os.O_NOFOLLOW
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    if hasattr(os, "O_NONBLOCK"):
        flags |= os.O_NONBLOCK
    return flags


def _read_regular_file(
    stream: BinaryIO, *, capture_bytes: bool, frozen_size: int
) -> tuple[str, bytes | None]:
    digest = hashlib.sha256()
    contents = bytearray() if capture_bytes else None
    remaining = frozen_size
    while remaining:
        chunk = stream.read(min(1024 * 1024, remaining))
        if not isinstance(chunk, bytes):
            raise RepositorySnapshotError("regular file returned non-byte input")
        if not chunk:
            raise RepositorySnapshotError("regular file ended before its frozen size")
        if len(chunk) > remaining:
            raise RepositorySnapshotError("regular file exceeded its frozen size")
        digest.update(chunk)
        if contents is not None:
            contents.extend(chunk)
        remaining -= len(chunk)
    growth = stream.read(1)
    if not isinstance(growth, bytes):
        raise RepositorySnapshotError("regular file returned non-byte input")
    if growth:
        raise RepositorySnapshotError("regular file grew beyond its frozen size")
    return digest.hexdigest(), bytes(contents) if contents is not None else None


class SnapshotSession:
    """Capture and verify one repository root through an anchored descriptor."""

    def __init__(
        self,
        root: Path,
        *,
        regular_file_limits: RegularFileLimits | None = None,
        directory_structure_limits: DirectoryStructureLimits | None = None,
    ) -> None:
        _require_descriptor_relative_io()
        self._regular_file_limits = self._validate_regular_file_limits(
            DEFAULT_REGULAR_FILE_LIMITS
            if regular_file_limits is None
            else regular_file_limits
        )
        self._directory_structure_limits = self._validate_directory_structure_limits(
            DEFAULT_DIRECTORY_STRUCTURE_LIMITS
            if directory_structure_limits is None
            else directory_structure_limits
        )
        self.root = root
        try:
            self._root_fd = os.open(root, _directory_flags())
            metadata = os.fstat(self._root_fd)
        except OSError as exc:
            raise RepositorySnapshotError(
                "unable to open repository root safely"
            ) from exc
        if not stat.S_ISDIR(metadata.st_mode):
            os.close(self._root_fd)
            raise RepositorySnapshotError("repository root must be a directory")
        self._root_identity = _directory_identity(metadata)
        self._nodes: dict[str, FrozenNode] = {}
        self._directories: dict[str, FrozenDirectory] = {
            "": FrozenDirectory("", self._root_identity)
        }
        self._absences: dict[str, AbsenceProof] = {}
        self._structure: FrozenDirectoryStructure | None = None
        self._snapshot: TreeSnapshot | None = None
        self._capture_round = _RegularFileRound(self._regular_file_limits)
        self._closed = False

    def __enter__(self) -> SnapshotSession:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def close(self) -> None:
        if not self._closed:
            os.close(self._root_fd)
            self._closed = True

    def _ensure_open(self) -> None:
        if self._closed:
            raise RepositorySnapshotError("snapshot session is closed")

    def _ensure_capturing(self) -> None:
        self._ensure_open()
        if self._snapshot is not None:
            raise RepositorySnapshotError("snapshot session is already sealed")

    @staticmethod
    def _validate_regular_file_limits(limits: RegularFileLimits) -> RegularFileLimits:
        if not isinstance(limits, RegularFileLimits):
            raise RepositorySnapshotError(
                "regular file limits must be a RegularFileLimits value"
            )
        values = (
            limits.max_file_bytes,
            limits.max_round_bytes,
            limits.max_cached_file_bytes,
            limits.max_cached_total_bytes,
        )
        if any(type(value) is not int or value < 0 for value in values):
            raise RepositorySnapshotError(
                "regular file limits must be non-negative integers"
            )
        if limits.max_cached_file_bytes > limits.max_file_bytes:
            raise RepositorySnapshotError(
                "cached-file limit must not exceed the regular-file limit"
            )
        if limits.max_cached_total_bytes > limits.max_round_bytes:
            raise RepositorySnapshotError(
                "cached-total limit must not exceed the regular-file round limit"
            )
        return limits

    @staticmethod
    def _validate_directory_structure_limits(
        limits: DirectoryStructureLimits,
    ) -> DirectoryStructureLimits:
        if not isinstance(limits, DirectoryStructureLimits):
            raise RepositorySnapshotError(
                "directory structure limits must be a DirectoryStructureLimits value"
            )
        values = (
            limits.max_directories,
            limits.max_entries,
            limits.max_depth,
            limits.max_total_name_bytes,
            limits.max_total_structure_bytes,
        )
        if any(type(value) is not int or value < 0 for value in values):
            raise RepositorySnapshotError(
                "directory structure limits must be non-negative integers"
            )
        if limits.max_directories == 0:
            raise RepositorySnapshotError(
                "directory structure limit must permit the repository root"
            )
        return limits

    @classmethod
    def _structure_limits(
        cls,
        *,
        max_directories: int,
        max_entries: int,
        max_depth: int,
        max_total_name_bytes: int,
        max_total_structure_bytes: int,
    ) -> DirectoryStructureLimits:
        return cls._validate_directory_structure_limits(
            DirectoryStructureLimits(
                max_directories,
                max_entries,
                max_depth,
                max_total_name_bytes,
                max_total_structure_bytes,
            )
        )

    def _merge_directory(self, directory: FrozenDirectory) -> None:
        previous = self._directories.get(directory.path)
        if previous is not None and previous.identity != directory.identity:
            raise RepositorySnapshotError(
                f"repository directory changed during snapshot: {directory.path or '.'}"
            )
        if previous is not None and previous.children is not None:
            if directory.children is not None and (
                previous.children != directory.children
                or previous.excluded_parts != directory.excluded_parts
            ):
                raise RepositorySnapshotError(
                    f"repository directory entries changed during snapshot: "
                    f"{directory.path or '.'}"
                )
            return
        self._directories[directory.path] = directory

    @staticmethod
    def _entry_kind(metadata: os.stat_result) -> str:
        if stat.S_ISREG(metadata.st_mode):
            return "regular"
        if stat.S_ISDIR(metadata.st_mode):
            return "directory"
        if stat.S_ISLNK(metadata.st_mode):
            return "symlink"
        return "special"

    def _open_parent(self, path: str) -> tuple[int, tuple[FrozenDirectory, ...]]:
        parts = relative_path(path)
        current = os.dup(self._root_fd)
        parents = [FrozenDirectory("", _directory_identity(os.fstat(current)))]
        if parents[0].identity != self._root_identity:
            os.close(current)
            raise RepositorySnapshotError("repository root changed during snapshot")
        self._merge_directory(parents[0])
        prefix: list[str] = []
        try:
            for part in parts[:-1]:
                try:
                    child = os.open(part, _directory_flags(), dir_fd=current)
                except OSError as exc:
                    raise RepositorySnapshotError(
                        f"repository member parent changed or is unsafe: {path}"
                    ) from exc
                os.close(current)
                current = child
                metadata = os.fstat(current)
                if not stat.S_ISDIR(metadata.st_mode):
                    raise RepositorySnapshotError(
                        f"repository member parent is not a directory: {path}"
                    )
                prefix.append(part)
                directory = FrozenDirectory(
                    "/".join(prefix), _directory_identity(metadata)
                )
                self._merge_directory(directory)
                parents.append(directory)
            return current, tuple(parents)
        except Exception:
            os.close(current)
            raise

    def _verify_parent_chain(
        self, path: str, expected: tuple[FrozenDirectory, ...]
    ) -> None:
        descriptor, observed = self._open_parent(path)
        try:
            if observed != expected:
                raise RepositorySnapshotError(
                    f"repository member parent changed during snapshot: {path}"
                )
        finally:
            os.close(descriptor)

    def _capture_node(
        self,
        path: str,
        *,
        include_bytes: bool,
        limit: int | None,
        regular_round: _RegularFileRound | None = None,
        expected_kind: str | None = None,
        structure_round: _DirectoryStructureRound | None = None,
        structure_depth: int | None = None,
    ) -> FrozenNode:
        if limit is not None and (type(limit) is not int or limit < 0):
            raise RepositorySnapshotError(
                "regular file caller limit must be a non-negative integer"
            )
        previous = self._nodes.get(path)
        if previous is not None and (not include_bytes or previous.bytes is not None):
            if structure_round is not None and previous.kind == "directory":
                if structure_depth is None:
                    raise AssertionError("directory admission requires a depth")
                structure_round.enter_directory(structure_depth)
            effective_limit = self._regular_file_limits.max_file_bytes
            if limit is not None:
                effective_limit = min(effective_limit, limit)
            if previous.kind == "regular" and previous.size > effective_limit:
                raise RepositorySnapshotError("regular file exceeds its input limit")
            return previous
        round_budget = self._capture_round if regular_round is None else regular_round
        parent_fd, parents = self._open_parent(path)
        name = relative_path(path)[-1]
        descriptor = -1
        try:
            try:
                metadata = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
            except FileNotFoundError:
                raise
            except OSError as exc:
                raise RepositorySnapshotError(
                    f"unable to inspect repository member safely: {path}"
                ) from exc

            observed_kind = self._entry_kind(metadata)
            if expected_kind is not None and observed_kind != expected_kind:
                raise RepositorySnapshotError(
                    f"repository member kind changed during snapshot: {path}"
                )

            if structure_round is not None and observed_kind == "directory":
                if structure_depth is None:
                    raise AssertionError("directory admission requires a depth")
                structure_round.enter_directory(structure_depth)

            if stat.S_ISREG(metadata.st_mode):
                try:
                    descriptor = os.open(name, _regular_flags(), dir_fd=parent_fd)
                except OSError as exc:
                    raise RepositorySnapshotError(
                        f"repository member changed or is unsafe: {path}"
                    ) from exc
                opened = os.fstat(descriptor)
                identity = _node_identity(opened)
                if not stat.S_ISREG(opened.st_mode) or identity != _node_identity(
                    metadata
                ):
                    raise RepositorySnapshotError(
                        f"repository member changed during snapshot: {path}"
                    )
                effective_limit = self._regular_file_limits.max_file_bytes
                if limit is not None:
                    effective_limit = min(effective_limit, limit)
                round_budget.admit(
                    identity.size,
                    capture_bytes=include_bytes,
                    limit=effective_limit,
                )
                with os.fdopen(descriptor, "rb", closefd=True) as stream:
                    descriptor = -1
                    digest, contents = _read_regular_file(
                        stream,
                        capture_bytes=include_bytes,
                        frozen_size=identity.size,
                    )
                    if _node_identity(os.fstat(stream.fileno())) != identity:
                        raise RepositorySnapshotError(
                            f"repository member changed while being read: {path}"
                        )
                    try:
                        rebound = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
                    except OSError as exc:
                        raise RepositorySnapshotError(
                            f"repository member path changed while being read: {path}"
                        ) from exc
                    if _node_identity(rebound) != identity:
                        raise RepositorySnapshotError(
                            f"repository member path changed while being read: {path}"
                        )
                    if _directory_identity(os.fstat(parent_fd)) != parents[-1].identity:
                        raise RepositorySnapshotError(
                            f"repository member parent changed while being read: {path}"
                        )
                    self._verify_parent_chain(path, parents)
                node = FrozenNode(path, "regular", identity, parents, digest, contents)
            elif stat.S_ISDIR(metadata.st_mode):
                try:
                    descriptor = os.open(name, _directory_flags(), dir_fd=parent_fd)
                except OSError as exc:
                    raise RepositorySnapshotError(
                        f"repository directory changed or is unsafe: {path}"
                    ) from exc
                opened = os.fstat(descriptor)
                identity = _directory_identity(opened)
                if identity != _directory_identity(metadata):
                    raise RepositorySnapshotError(
                        f"repository directory changed during snapshot: {path}"
                    )
                node = FrozenNode(path, "directory", identity, parents)
                self._merge_directory(FrozenDirectory(path, identity))
            elif stat.S_ISLNK(metadata.st_mode):
                identity = _node_identity(metadata)
                try:
                    target = os.readlink(os.fsencode(name), dir_fd=parent_fd)
                    verified = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
                except OSError as exc:
                    raise RepositorySnapshotError(
                        f"repository symlink changed during snapshot: {path}"
                    ) from exc
                if not isinstance(target, bytes):
                    target = os.fsencode(target)
                if _node_identity(verified) != identity:
                    raise RepositorySnapshotError(
                        f"repository symlink changed during snapshot: {path}"
                    )
                node = FrozenNode(
                    path,
                    "symlink",
                    identity,
                    parents,
                    symlink_target=target,
                )
            else:
                node = FrozenNode(path, "special", _node_identity(metadata), parents)

            previous = self._nodes.get(path)
            if previous is not None:
                comparable_previous = FrozenNode(
                    previous.path,
                    previous.kind,
                    previous.identity,
                    previous.parents,
                    previous.sha256,
                    None,
                    previous.symlink_target,
                )
                comparable_node = FrozenNode(
                    node.path,
                    node.kind,
                    node.identity,
                    node.parents,
                    node.sha256,
                    None,
                    node.symlink_target,
                )
                if comparable_previous != comparable_node:
                    raise RepositorySnapshotError(
                        f"repository member changed during snapshot: {path}"
                    )
                if previous.bytes is not None and node.bytes is None:
                    node = previous
            self._nodes[path] = node
            return node
        finally:
            if descriptor >= 0:
                os.close(descriptor)
            os.close(parent_fd)

    def capture_paths(
        self,
        paths: Iterable[str],
        *,
        include_bytes: bool = False,
        limit: int | None = None,
    ) -> tuple[FrozenNode, ...]:
        self._ensure_capturing()
        captured = []
        for path in paths:
            captured.append(
                self._capture_node(path, include_bytes=include_bytes, limit=limit)
            )
        return tuple(captured)

    @staticmethod
    def _safe_child_names(
        descriptor: int,
        path: str,
        *,
        excluded_parts: frozenset[str],
        structure_round: _DirectoryStructureRound,
        depth: int,
        admit_directory: bool = True,
    ) -> tuple[str, ...]:
        if admit_directory:
            structure_round.enter_directory(depth)
        names: list[str] = []
        try:
            with os.scandir(descriptor) as iterator:
                for entry in iterator:
                    name = entry.name
                    if (
                        type(name) is not str
                        or not name
                        or name in {".", ".."}
                        or "/" in name
                        or "\0" in name
                    ):
                        raise RepositorySnapshotError(
                            f"repository directory returned an unsafe child name: {path}"
                        )
                    try:
                        encoded_name = name.encode("utf-8", errors="strict")
                    except UnicodeEncodeError as exc:
                        raise RepositorySnapshotError(
                            f"repository child name is not valid UTF-8: {path}"
                        ) from exc
                    if name in excluded_parts:
                        continue
                    child_path = f"{path}/{name}" if path not in {"", "."} else name
                    structure_round.admit_entry(
                        encoded_name,
                        child_path.encode("utf-8"),
                    )
                    names.append(name)
        except OSError as exc:
            raise RepositorySnapshotError(
                f"unable to enumerate repository directory: {path}"
            ) from exc
        if len(names) != len(set(names)):
            raise RepositorySnapshotError(
                f"repository directory returned duplicate child names: {path}"
            )
        return tuple(sorted(names))

    @staticmethod
    def _excluded_parts(parts: Iterable[str]) -> frozenset[str]:
        excluded: set[str] = set()
        for part in parts:
            if type(part) is not str:
                raise RepositorySnapshotError(
                    "archive exclusions must be path-part strings"
                )
            try:
                validated = relative_path(part)
            except RepositorySnapshotError as exc:
                raise RepositorySnapshotError(
                    f"invalid archive exclusion path part: {part!r}"
                ) from exc
            if len(validated) != 1:
                raise RepositorySnapshotError(
                    f"archive exclusion must be one path part: {part!r}"
                )
            excluded.add(part)
        return frozenset(excluded)

    def _observe_directory_structure(
        self,
        excluded_parts: frozenset[str],
        limits: DirectoryStructureLimits,
    ) -> FrozenDirectoryStructure:
        directories: list[FrozenDirectory] = []
        structure_round = _DirectoryStructureRound(limits)

        def visit(
            descriptor: int,
            path: str,
            depth: int,
            *,
            directory_admitted: bool = False,
        ) -> None:
            if not directory_admitted:
                structure_round.enter_directory(depth)
            initial_metadata = os.fstat(descriptor)
            if not stat.S_ISDIR(initial_metadata.st_mode):
                raise RepositorySnapshotError(
                    "repository directory structure contains an unsafe directory"
                )
            initial_identity = _node_identity(initial_metadata)
            entries: list[tuple[str, str, DirectoryIdentity | None]] = []
            try:
                with os.scandir(descriptor) as iterator:
                    for entry in iterator:
                        name = entry.name
                        if (
                            type(name) is not str
                            or not name
                            or name in {".", ".."}
                            or "/" in name
                            or "\0" in name
                        ):
                            raise RepositorySnapshotError(
                                "repository directory structure returned an unsafe name"
                            )
                        try:
                            encoded_name = name.encode("utf-8", errors="strict")
                        except UnicodeEncodeError as exc:
                            raise RepositorySnapshotError(
                                "repository directory structure contains a non-UTF-8 name"
                            ) from exc
                        if name in excluded_parts:
                            continue
                        child_path = f"{path}/{name}" if path else name
                        structure_round.admit_entry(
                            encoded_name,
                            child_path.encode("utf-8"),
                        )
                        metadata = os.stat(
                            name, dir_fd=descriptor, follow_symlinks=False
                        )
                        kind = self._entry_kind(metadata)
                        child_identity = (
                            _directory_identity(metadata)
                            if kind == "directory"
                            else None
                        )
                        entries.append((name, kind, child_identity))
            except OSError as exc:
                raise RepositorySnapshotError(
                    "unable to enumerate repository directory structure safely"
                ) from exc

            if _node_identity(os.fstat(descriptor)) != initial_identity:
                raise RepositorySnapshotError(
                    "repository directory structure changed during enumeration"
                )
            entries.sort(key=lambda item: item[0])
            if len(entries) != len({name for name, _, _ in entries}):
                raise RepositorySnapshotError(
                    "repository directory structure returned duplicate names"
                )
            directories.append(
                FrozenDirectory(
                    path,
                    _directory_identity(initial_metadata),
                    excluded_parts=excluded_parts,
                    structure_children=tuple((name, kind) for name, kind, _ in entries),
                    structure_identity=initial_identity,
                )
            )

            for name, kind, expected_identity in entries:
                if kind != "directory":
                    continue
                if depth >= limits.max_depth:
                    raise RepositorySnapshotError(
                        "repository directory structure exceeds max_depth"
                    )
                child = -1
                try:
                    structure_round.enter_directory(depth + 1)
                    child = os.open(name, _directory_flags(), dir_fd=descriptor)
                    if _directory_identity(os.fstat(child)) != expected_identity:
                        raise RepositorySnapshotError(
                            "repository directory structure changed during traversal"
                        )
                    child_path = f"{path}/{name}" if path else name
                    visit(
                        child,
                        child_path,
                        depth + 1,
                        directory_admitted=True,
                    )
                except OSError as exc:
                    raise RepositorySnapshotError(
                        "repository directory structure changed during traversal"
                    ) from exc
                finally:
                    if child >= 0:
                        os.close(child)

        try:
            visit(self._root_fd, "", 0)
        except RepositorySnapshotError:
            raise
        except OSError as exc:
            raise RepositorySnapshotError(
                "unable to freeze repository directory structure safely"
            ) from exc
        return FrozenDirectoryStructure(
            tuple(sorted(directories, key=lambda item: (item.path != "", item.path))),
            excluded_parts,
            limits,
        )

    def freeze_directory_structure(
        self,
        excluded_parts: Iterable[str] = (),
        *,
        max_directories: int | None = None,
        max_entries: int | None = None,
        max_depth: int | None = None,
        max_total_name_bytes: int | None = None,
        max_total_structure_bytes: int | None = None,
    ) -> FrozenDirectoryStructure:
        """Freeze bounded child-name and node-kind topology below the root.

        Excluded path parts are removed before accounting and are never
        traversed.  All other names remain internal snapshot facts; failures do
        not expose them in exception text.
        """
        self._ensure_capturing()
        if self._structure is not None:
            raise RepositorySnapshotError(
                "repository directory structure is already frozen"
            )
        excluded = self._excluded_parts(excluded_parts)
        supplied = (
            max_directories,
            max_entries,
            max_depth,
            max_total_name_bytes,
        )
        if (
            all(value is None for value in supplied)
            and max_total_structure_bytes is None
        ):
            limits = self._directory_structure_limits
        elif any(value is None for value in supplied):
            raise RepositorySnapshotError(
                "directory structure limits must be supplied together"
            )
        else:
            assert max_directories is not None
            assert max_entries is not None
            assert max_depth is not None
            assert max_total_name_bytes is not None
            limits = self._structure_limits(
                max_directories=max_directories,
                max_entries=max_entries,
                max_depth=max_depth,
                max_total_name_bytes=max_total_name_bytes,
                max_total_structure_bytes=(
                    self._directory_structure_limits.max_total_structure_bytes
                    if max_total_structure_bytes is None
                    else max_total_structure_bytes
                ),
            )
        try:
            structure = self._observe_directory_structure(excluded, limits)
        except RepositorySnapshotError as exc:
            raise RepositorySnapshotError(
                f"repository directory structure snapshot failed: {exc}"
            ) from exc
        self._structure = structure
        return structure

    def _walk_archive_paths(
        self,
        paths: Iterable[str],
        excluded_parts: frozenset[str],
        *,
        regular_round: _RegularFileRound | None = None,
        frozen_structure: FrozenDirectoryStructure | None = None,
    ) -> tuple[FrozenNode, ...]:
        walked: list[FrozenNode] = []
        structure_round = _DirectoryStructureRound(self._directory_structure_limits)
        frozen_directories = (
            {}
            if frozen_structure is None
            else {
                directory.path: directory for directory in frozen_structure.directories
            }
        )
        frozen_kinds: dict[str, str] = {}
        if frozen_structure is not None:
            for directory in frozen_structure.directories:
                for name, kind in directory.structure_children or ():
                    child_path = f"{directory.path}/{name}" if directory.path else name
                    frozen_kinds[child_path] = kind

        def visit(path: str) -> None:
            if (
                len(relative_path(path))
                > self._directory_structure_limits.max_depth + 1
            ):
                raise RepositorySnapshotError(
                    "repository directory structure exceeds max_depth"
                )
            node = self._capture_node(
                path,
                include_bytes=False,
                limit=None,
                regular_round=regular_round,
                expected_kind=frozen_kinds.get(path),
                structure_round=(
                    None if frozen_structure is not None else structure_round
                ),
                structure_depth=len(relative_path(path)),
            )
            walked.append(node)
            if node.kind != "directory":
                return
            parent_fd, _ = self._open_parent(path)
            descriptor = -1
            try:
                descriptor = os.open(
                    relative_path(path)[-1], _directory_flags(), dir_fd=parent_fd
                )
                if _directory_identity(os.fstat(descriptor)) != node.identity:
                    raise RepositorySnapshotError(
                        f"repository directory changed during enumeration: {path}"
                    )
                if frozen_structure is None:
                    names = self._safe_child_names(
                        descriptor,
                        path,
                        excluded_parts=excluded_parts,
                        structure_round=structure_round,
                        depth=len(relative_path(path)),
                        admit_directory=False,
                    )
                else:
                    frozen_directory = frozen_directories.get(path)
                    if frozen_directory is None:
                        raise RepositorySnapshotError(
                            f"repository directory changed during archive enumeration: {path}"
                        )
                    names = tuple(
                        name for name, _ in frozen_directory.structure_children or ()
                    )
                self._merge_directory(
                    FrozenDirectory(path, node.identity, names, excluded_parts)
                )
                if _directory_identity(os.fstat(descriptor)) != node.identity:
                    raise RepositorySnapshotError(
                        f"repository directory changed during enumeration: {path}"
                    )
            except OSError as exc:
                raise RepositorySnapshotError(
                    f"repository directory changed or is unsafe: {path}"
                ) from exc
            finally:
                if descriptor >= 0:
                    os.close(descriptor)
                os.close(parent_fd)
            for name in names:
                visit(f"{path}/{name}")

        for path in paths:
            if frozen_structure is None:
                parts = relative_path(path)
                structure_round.admit_entry(
                    parts[-1].encode("utf-8"),
                    path.encode("utf-8"),
                )
            visit(path)
        return tuple(walked)

    def walk_archive(self, paths: Iterable[str]) -> tuple[FrozenNode, ...]:
        """Capture complete subtrees, including raw symlinks and special nodes."""
        self._ensure_capturing()
        return self._walk_archive_paths(paths, frozenset())

    def walk_archive_root(
        self, excluded_parts: Iterable[str] = ()
    ) -> tuple[FrozenNode, ...]:
        """Capture the whole root, excluding matching parts at every depth.

        Each visited directory freezes the child-name set after applying the
        same immutable exclusion rule.  Churn confined to excluded components
        is therefore irrelevant, while every included root or nested entry is
        covered by ``seal`` and ``verify``.
        """
        self._ensure_capturing()
        excluded = self._excluded_parts(excluded_parts)
        if self._structure is None:
            self.freeze_directory_structure(excluded)
        assert self._structure is not None
        structure = self._structure
        if structure.excluded_parts != excluded:
            raise RepositorySnapshotError(
                "archive exclusions differ from the frozen directory structure"
            )
        identity = _directory_identity(os.fstat(self._root_fd))
        if identity != self._root_identity:
            raise RepositorySnapshotError(
                "repository root changed during archive enumeration"
            )
        root_structure = next(
            directory for directory in structure.directories if directory.path == ""
        )
        names = tuple(name for name, _ in root_structure.structure_children or ())
        self._merge_directory(FrozenDirectory("", identity, names, excluded))
        if _directory_identity(os.fstat(self._root_fd)) != identity:
            raise RepositorySnapshotError(
                "repository root changed during archive enumeration"
            )
        walked = self._walk_archive_paths(
            names,
            excluded,
            frozen_structure=structure,
        )
        # Archive snapshots retain bounded per-directory child sets.  The
        # whole-tree structure is an admission proof, not a second observation
        # whose directory timestamps would reject churn confined to excluded
        # components.
        self._structure = None
        return walked

    def prove_absent(self, path: str) -> AbsenceProof:
        self._ensure_capturing()
        parts = relative_path(path)
        current = os.dup(self._root_fd)
        prefix: list[str] = []
        try:
            self._merge_directory(
                FrozenDirectory("", _directory_identity(os.fstat(current)))
            )
            for index, part in enumerate(parts):
                is_leaf = index == len(parts) - 1
                try:
                    metadata = os.stat(part, dir_fd=current, follow_symlinks=False)
                except FileNotFoundError:
                    prefix.append(part)
                    proof = AbsenceProof(path, "/".join(prefix))
                    self._absences[path] = proof
                    return proof
                except OSError as exc:
                    raise RepositorySnapshotError(
                        f"unable to prove repository path absent safely: {path}"
                    ) from exc
                prefix.append(part)
                if is_leaf:
                    raise RepositorySnapshotError(
                        f"repository path appeared after absence boundary: {path}"
                    )
                if not stat.S_ISDIR(metadata.st_mode):
                    raise RepositorySnapshotError(
                        f"repository absence parent is unsafe: {path}"
                    )
                try:
                    child = os.open(part, _directory_flags(), dir_fd=current)
                except OSError as exc:
                    raise RepositorySnapshotError(
                        f"repository absence parent changed or is unsafe: {path}"
                    ) from exc
                os.close(current)
                current = child
                self._merge_directory(
                    FrozenDirectory(
                        "/".join(prefix), _directory_identity(os.fstat(current))
                    )
                )
        finally:
            os.close(current)
        raise AssertionError("unreachable")

    def _snapshot_value(self) -> TreeSnapshot:
        return TreeSnapshot(
            self.root,
            self._root_identity,
            tuple(self._nodes[path] for path in sorted(self._nodes)),
            tuple(
                self._directories[path]
                for path in sorted(
                    self._directories, key=lambda item: (item != "", item)
                )
            ),
            tuple(self._absences[path] for path in sorted(self._absences)),
            self._structure,
        )

    def seal(self) -> TreeSnapshot:
        self._ensure_capturing()
        candidate = self._snapshot_value()
        self.verify(candidate)
        self._snapshot = candidate
        return candidate

    def _verify_directory(
        self,
        directory: FrozenDirectory,
        structure_round: _DirectoryStructureRound,
    ) -> None:
        if directory.path == "":
            identity = _directory_identity(os.fstat(self._root_fd))
            descriptor = os.dup(self._root_fd)
        else:
            parent_fd, _ = self._open_parent(directory.path)
            try:
                descriptor = os.open(
                    relative_path(directory.path)[-1],
                    _directory_flags(),
                    dir_fd=parent_fd,
                )
            except OSError as exc:
                os.close(parent_fd)
                raise RepositorySnapshotError(
                    f"repository directory changed after snapshot: {directory.path}"
                ) from exc
            os.close(parent_fd)
            identity = _directory_identity(os.fstat(descriptor))
        try:
            if identity != directory.identity:
                raise RepositorySnapshotError(
                    f"repository directory changed after snapshot: "
                    f"{directory.path or '.'}"
                )
            if directory.children is not None:
                names = self._safe_child_names(
                    descriptor,
                    directory.path or ".",
                    excluded_parts=directory.excluded_parts,
                    structure_round=structure_round,
                    depth=len(PurePosixPath(directory.path).parts),
                )
                if names != directory.children:
                    raise RepositorySnapshotError(
                        f"repository directory entries changed after snapshot: "
                        f"{directory.path or '.'}"
                    )
        finally:
            os.close(descriptor)

    def _observe_node(
        self, expected: FrozenNode, regular_round: _RegularFileRound
    ) -> FrozenNode:
        return self._capture_node(
            expected.path,
            include_bytes=expected.bytes is not None,
            limit=expected.size,
            regular_round=regular_round,
        )

    def _verify_absence(self, proof: AbsenceProof) -> None:
        parts = relative_path(proof.path)
        missing_parts = relative_path(proof.missing_prefix)
        current = os.dup(self._root_fd)
        try:
            prefix: list[str] = []
            for index, part in enumerate(parts):
                try:
                    metadata = os.stat(part, dir_fd=current, follow_symlinks=False)
                except FileNotFoundError:
                    prefix.append(part)
                    if tuple(prefix) != missing_parts:
                        raise RepositorySnapshotError(
                            f"repository absence boundary changed after snapshot: {proof.path}"
                        ) from None
                    return
                prefix.append(part)
                if index == len(parts) - 1:
                    raise RepositorySnapshotError(
                        f"repository path appeared after snapshot: {proof.path}"
                    )
                if not stat.S_ISDIR(metadata.st_mode):
                    raise RepositorySnapshotError(
                        f"repository absence parent changed after snapshot: {proof.path}"
                    )
                child = os.open(part, _directory_flags(), dir_fd=current)
                os.close(current)
                current = child
        except OSError as exc:
            raise RepositorySnapshotError(
                f"unable to verify repository absence safely: {proof.path}"
            ) from exc
        finally:
            os.close(current)

    def verify(self, snapshot: TreeSnapshot | None = None) -> None:
        self._ensure_open()
        expected = self._snapshot if snapshot is None else snapshot
        if expected is None:
            raise RepositorySnapshotError("snapshot session is not sealed")
        if expected.root_identity != self._root_identity:
            raise RepositorySnapshotError("repository root identity changed")
        if _directory_identity(os.fstat(self._root_fd)) != expected.root_identity:
            raise RepositorySnapshotError("repository root changed after snapshot")

        if expected.structure is not None:
            try:
                observed_structure = self._observe_directory_structure(
                    expected.structure.excluded_parts,
                    expected.structure.limits,
                )
            except RepositorySnapshotError as exc:
                raise RepositorySnapshotError(
                    "repository directory changed or directory entries changed after snapshot"
                ) from exc
            if observed_structure != expected.structure:
                raise RepositorySnapshotError(
                    "repository directory changed or directory entries changed after snapshot"
                )

        saved_nodes = self._nodes
        saved_directories = self._directories
        saved_absences = self._absences
        saved_snapshot = self._snapshot
        verification_round = _RegularFileRound(self._regular_file_limits)
        verification_structure_round = _DirectoryStructureRound(
            self._directory_structure_limits
        )
        try:
            self._nodes = {}
            self._directories = {"": FrozenDirectory("", self._root_identity)}
            self._absences = {}
            self._snapshot = None
            for directory in expected.directories:
                self._verify_directory(directory, verification_structure_round)
            for node in expected.nodes:
                try:
                    observed = self._observe_node(node, verification_round)
                except FileNotFoundError as exc:
                    raise RepositorySnapshotError(
                        f"repository member changed after snapshot: {node.path}"
                    ) from exc
                if observed != node:
                    raise RepositorySnapshotError(
                        f"repository member content changed after snapshot: {node.path}"
                    )
            for proof in expected.absences:
                self._verify_absence(proof)
        finally:
            self._nodes = saved_nodes
            self._directories = saved_directories
            self._absences = saved_absences
            self._snapshot = saved_snapshot

    def open_verified_regular(self, node: FrozenNode) -> BinaryIO:
        """Open, hash, and rewind the exact regular node on one descriptor."""
        self._ensure_open()
        if node.kind != "regular" or not isinstance(node.identity, NodeIdentity):
            raise RepositorySnapshotError(
                f"repository member is not regular: {node.path}"
            )
        parent_fd, parents = self._open_parent(node.path)
        descriptor = -1
        try:
            descriptor = os.open(
                relative_path(node.path)[-1], _regular_flags(), dir_fd=parent_fd
            )
            identity = _node_identity(os.fstat(descriptor))
            if identity != node.identity or parents != node.parents:
                raise RepositorySnapshotError(
                    f"repository member changed after snapshot: {node.path}"
                )
            regular_round = _RegularFileRound(self._regular_file_limits)
            regular_round.admit(
                identity.size,
                capture_bytes=node.bytes is not None,
                limit=min(self._regular_file_limits.max_file_bytes, node.size),
            )
            stream = os.fdopen(descriptor, "rb", closefd=True)
            descriptor = -1
            try:
                digest, contents = _read_regular_file(
                    stream,
                    capture_bytes=node.bytes is not None,
                    frozen_size=node.size,
                )
                if digest != node.sha256 or (
                    node.bytes is not None and contents != node.bytes
                ):
                    raise RepositorySnapshotError(
                        f"repository member content changed after snapshot: {node.path}"
                    )
                if _node_identity(os.fstat(stream.fileno())) != node.identity:
                    raise RepositorySnapshotError(
                        f"repository member changed while being read: {node.path}"
                    )
                try:
                    rebound = os.stat(
                        relative_path(node.path)[-1],
                        dir_fd=parent_fd,
                        follow_symlinks=False,
                    )
                except OSError as exc:
                    raise RepositorySnapshotError(
                        f"repository member path changed while being read: {node.path}"
                    ) from exc
                if _node_identity(rebound) != node.identity:
                    raise RepositorySnapshotError(
                        f"repository member path changed while being read: {node.path}"
                    )
                if _directory_identity(os.fstat(parent_fd)) != parents[-1].identity:
                    raise RepositorySnapshotError(
                        f"repository member parent changed while being read: {node.path}"
                    )
                self._verify_parent_chain(node.path, parents)
                stream.seek(0)
                return stream
            except Exception:
                stream.close()
                raise
        except OSError as exc:
            raise RepositorySnapshotError(
                f"repository member changed or is unsafe: {node.path}"
            ) from exc
        finally:
            if descriptor >= 0:
                os.close(descriptor)
            os.close(parent_fd)


def regular_nodes(nodes: Iterable[FrozenNode]) -> Iterator[FrozenNode]:
    return (node for node in nodes if node.kind == "regular")


def capture_control_artifact(root: Path, rel: str, *, max_bytes: int) -> FrozenNode:
    """Freeze one bounded regular control artifact and its exact input bytes.

    The path is resolved beneath an anchored repository descriptor without
    following symlinks.  The bytes, digest, and final identity are all observed
    through the same bounded regular-file descriptor, followed by an anchored
    path-identity check before the frozen fact is returned.
    """
    relative_path(rel)
    if type(max_bytes) is not int or max_bytes < 0:
        raise RepositorySnapshotError("control artifact limit must be non-negative")
    try:
        with SnapshotSession(root) as session:
            node = session.capture_paths((rel,), include_bytes=True, limit=max_bytes)[0]
            if node.kind != "regular" or node.bytes is None or node.sha256 is None:
                raise RepositorySnapshotError(
                    f"control artifact must be a regular file: {rel}"
                )
            return session.seal().node(rel)
    except FileNotFoundError as exc:
        raise RepositorySnapshotError(f"control artifact is missing: {rel}") from exc
