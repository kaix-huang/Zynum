#!/usr/bin/env python3
# Copyright (C) 2026 Zynum contributors
# SPDX-License-Identifier: LGPL-3.0-or-later

"""Validate build.zig.zon package paths and optionally create a source archive."""

from __future__ import annotations

import argparse
import contextvars
import dataclasses
import enum
import gzip
import hashlib
import importlib.util
import json
import os
import secrets
import shutil
import stat
import subprocess
import sys
import tarfile
import tempfile
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import BinaryIO, overload


def _load_sibling(name: str, module_name: str) -> object:
    path = Path(__file__).with_name(name)
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load {name}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


repository_snapshot = _load_sibling(
    "repository_snapshot.py", "_zynum_repository_snapshot"
)
repository_git = _load_sibling("repository_git.py", "_zynum_repository_git")

FORBIDDEN_PATH_PARTS = frozenset((".git", ".local-docs", ".zig-cache", "zig-out"))
FORBIDDEN_MEMBER_NAMES = frozenset((".DS_Store",))
FORBIDDEN_MEMBER_PARTS = frozenset(("__pycache__",))
FORBIDDEN_MEMBER_SUFFIXES = frozenset((".pyc", ".pyo"))
MAX_ZON_BYTES = 1024 * 1024
MAX_PACKAGE_PATHS = 4096
MAX_PACKAGE_PATH_BYTES = 4096
MAX_TOTAL_PACKAGE_PATH_BYTES = 1024 * 1024
PACKAGE_PATH_SCHEMA_VERSION = 1
# Add a series only after its parser APIs and emitted schema have been audited.
PACKAGE_PATH_COMPATIBLE_ZIG_SERIES = frozenset({("0", "16")})
MAX_PACKAGE_PATH_OUTPUT_BYTES = 7 * 1024 * 1024
_AUTO_REPOSITORY = object()

PackageMember = repository_snapshot.FrozenNode


_PACKAGE_PATHS_FACTORY = object()
_PUBLISHABLE_SNAPSHOT_FACTORY = object()
_PACKAGE_PARSE_CONTEXT: contextvars.ContextVar[
    tuple[Path, PackageMember, bool] | None
] = contextvars.ContextVar("zynum_package_parse_context", default=None)


@dataclasses.dataclass(frozen=True, slots=True, init=False)
class _PackagePathsProof:
    """Module-private binding issued only by the canonical manifest parser."""

    authority: object
    root: Path
    paths: tuple[str, ...]
    manifest: PackageMember

    def __init__(
        self,
        authority: object,
        root: Path,
        paths: tuple[str, ...],
        manifest: PackageMember,
    ) -> None:
        if authority is not _PACKAGE_PATHS_FACTORY:
            raise TypeError("package path proof is factory-only")
        object.__setattr__(self, "authority", authority)
        object.__setattr__(self, "root", root)
        object.__setattr__(self, "paths", paths)
        object.__setattr__(self, "manifest", manifest)


@dataclasses.dataclass(frozen=True, slots=True, init=False)
class PackagePaths(Sequence[str]):
    """Immutable canonical-root paths plus their exact parsed manifest proof."""

    root: Path
    paths: tuple[str, ...]
    manifest: PackageMember
    _parse_proof: _PackagePathsProof = dataclasses.field(repr=False, compare=False)

    def __init__(self, paths: list[str], manifest: PackageMember) -> None:
        context = _PACKAGE_PARSE_CONTEXT.get()
        if context is None or context[1] is not manifest or not context[2]:
            raise TypeError("PackagePaths construction is parser-only")
        root = context[0]
        frozen_paths = tuple(paths)
        proof = _PackagePathsProof(_PACKAGE_PATHS_FACTORY, root, frozen_paths, manifest)
        object.__setattr__(self, "root", root)
        object.__setattr__(self, "paths", frozen_paths)
        object.__setattr__(self, "manifest", manifest)
        object.__setattr__(self, "_parse_proof", proof)
        _PACKAGE_PARSE_CONTEXT.set(None)

    @overload
    def __getitem__(self, index: int) -> str: ...

    @overload
    def __getitem__(self, index: slice) -> tuple[str, ...]: ...

    def __getitem__(self, index: int | slice) -> str | tuple[str, ...]:
        return self.paths[index]

    def __iter__(self) -> Iterator[str]:
        return iter(self.paths)

    def __len__(self) -> int:
        return len(self.paths)


@dataclasses.dataclass(frozen=True, slots=True)
class PackageSnapshot(Sequence[PackageMember]):
    """Immutable package membership plus its filesystem and Git proofs."""

    root: Path
    paths: tuple[str, ...]
    members: tuple[PackageMember, ...]
    tree: repository_snapshot.TreeSnapshot
    repository: repository_git.RepositoryGit | None
    git_files: repository_git.RepositoryFileSet | None
    manifest: PackageMember | None

    @overload
    def __getitem__(self, index: int) -> PackageMember: ...

    @overload
    def __getitem__(self, index: slice) -> tuple[PackageMember, ...]: ...

    def __getitem__(
        self, index: int | slice
    ) -> PackageMember | tuple[PackageMember, ...]:
        return self.members[index]

    def __iter__(self) -> Iterator[PackageMember]:
        return iter(self.members)

    def __len__(self) -> int:
        return len(self.members)


@dataclasses.dataclass(frozen=True, slots=True, init=False)
class _PublishableSnapshotProof:
    """Identity binding for one factory-created publication snapshot."""

    authority: object
    parsed_paths: PackagePaths
    root: Path
    manifest: PackageMember
    members: tuple[PackageMember, ...]
    tree: repository_snapshot.TreeSnapshot
    repository: repository_git.RepositoryGit
    git_files: repository_git.RepositoryFileSet
    publication: repository_git.RepositoryPublicationSnapshot

    def __init__(
        self,
        authority: object,
        *,
        parsed_paths: PackagePaths,
        root: Path,
        manifest: PackageMember,
        members: tuple[PackageMember, ...],
        tree: repository_snapshot.TreeSnapshot,
        repository: repository_git.RepositoryGit,
        git_files: repository_git.RepositoryFileSet,
        publication: repository_git.RepositoryPublicationSnapshot,
    ) -> None:
        if authority is not _PUBLISHABLE_SNAPSHOT_FACTORY:
            raise TypeError("publishable package proof is factory-only")
        object.__setattr__(self, "authority", authority)
        object.__setattr__(self, "parsed_paths", parsed_paths)
        object.__setattr__(self, "root", root)
        object.__setattr__(self, "manifest", manifest)
        object.__setattr__(self, "members", members)
        object.__setattr__(self, "tree", tree)
        object.__setattr__(self, "repository", repository)
        object.__setattr__(self, "git_files", git_files)
        object.__setattr__(self, "publication", publication)


@dataclasses.dataclass(frozen=True, slots=True)
class PublishablePackageSnapshot(PackageSnapshot):
    """Factory-created package snapshot authorized for archive publication."""

    _publication_proof: _PublishableSnapshotProof = dataclasses.field(
        repr=False, compare=False
    )


def package_paths(zon_path: Path, *, root: Path | None = None) -> PackagePaths:
    supplied_root = zon_path.parent if root is None else root
    candidate = zon_path if zon_path.is_absolute() else supplied_root / zon_path
    try:
        lexical_relative = Path(
            os.path.relpath(candidate.absolute(), supplied_root.absolute())
        ).as_posix()
    except (OSError, ValueError) as exc:
        raise ValueError("build.zig.zon must be inside the package root") from exc
    package_root = repository_git.strict_root(supplied_root)
    source, manifest = _read_stable_package_file(
        package_root,
        lexical_relative,
        limit=MAX_ZON_BYTES,
        description="build.zig.zon",
    )

    zig = shutil.which("zig", path=os.environ.get("PATH", os.defpath))
    if zig is None:
        raise RuntimeError("Zig 0.16.x is required to parse build.zig.zon")
    try:
        zig_executable = Path(zig).resolve(strict=True)
        helper = (
            Path(__file__).with_name("parse_package_paths.zig").resolve(strict=True)
        )
    except OSError as exc:
        raise RuntimeError("unable to resolve the package-path parser") from exc

    with tempfile.TemporaryDirectory(prefix="zynum-package-paths-") as cache_name:
        cache_root = Path(cache_name)
        local_cache = cache_root / "local"
        global_cache = cache_root / "global"
        local_cache.mkdir()
        global_cache.mkdir()
        try:
            result = subprocess.run(
                (
                    str(zig_executable),
                    "run",
                    "--cache-dir",
                    str(local_cache),
                    "--global-cache-dir",
                    str(global_cache),
                    str(helper),
                ),
                cwd=helper.parent,
                env=_zig_environment(os.environ),
                input=source,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
        except OSError as exc:
            raise RuntimeError("unable to execute the package-path parser") from exc
    if result.returncode != 0 or result.stderr:
        raise ValueError("build.zig.zon is not a valid static package manifest")
    paths = _decode_package_paths(result.stdout)
    _verify_manifest_snapshot(package_root, manifest)
    return PackagePaths(paths, manifest)


def _zig_environment(environment: Mapping[str, str]) -> dict[str, str]:
    allowed = {"PATH", "TMPDIR", "TMP", "TEMP", "SYSTEMROOT", "WINDIR"}
    child = {
        name: value for name, value in environment.items() if name.upper() in allowed
    }
    child.setdefault("PATH", os.defpath)
    child.update({"LANG": "C", "LC_ALL": "C"})
    return child


def _decode_package_paths(stdout: bytes) -> list[str]:
    if (
        len(stdout) > MAX_PACKAGE_PATH_OUTPUT_BYTES
        or not stdout.endswith(b"\n")
        or stdout.count(b"\n") != 1
    ):
        raise RuntimeError("package-path parser returned invalid output framing")
    try:
        text = stdout[:-1].decode("utf-8")
        value, end = json.JSONDecoder(object_pairs_hook=_unique_json_object).raw_decode(
            text
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise RuntimeError("package-path parser returned invalid JSON") from exc
    if end != len(text) or not isinstance(value, dict):
        raise RuntimeError("package-path parser returned invalid JSON framing")
    if set(value) != {"schema_version", "zig_version", "paths"}:
        raise RuntimeError("package-path parser returned unexpected fields")
    if (
        type(value["schema_version"]) is not int
        or value["schema_version"] != PACKAGE_PATH_SCHEMA_VERSION
        or type(value["paths"]) is not list
    ):
        raise RuntimeError("package-path parser returned an incompatible schema")
    if not _compatible_package_path_zig_version(value["zig_version"]):
        raise RuntimeError("package-path parser used an incompatible Zig version")

    paths = value["paths"]
    if not 0 < len(paths) <= MAX_PACKAGE_PATHS:
        raise ValueError("build.zig.zon has an invalid package path count")
    total_bytes = 0
    seen: set[str] = set()
    for path in paths:
        if type(path) is not str:
            raise RuntimeError("package-path parser returned a non-string path")
        try:
            encoded = path.encode("utf-8")
        except UnicodeEncodeError as exc:
            raise RuntimeError("package-path parser returned invalid Unicode") from exc
        if not encoded or len(encoded) > MAX_PACKAGE_PATH_BYTES:
            raise ValueError("build.zig.zon has an invalid package path length")
        if path in seen:
            raise ValueError("build.zig.zon has a duplicate package path")
        seen.add(path)
        total_bytes += len(encoded)
        if total_bytes > MAX_TOTAL_PACKAGE_PATH_BYTES:
            raise ValueError("build.zig.zon package paths exceed the total size limit")
    return paths


def _compatible_package_path_zig_version(value: object) -> bool:
    """Accept stable patch releases from explicitly compatible Zig series."""
    if type(value) is not str:
        return False
    components = value.split(".")
    if len(components) != 3 or any(
        not component
        or not component.isascii()
        or not component.isdecimal()
        or (len(component) > 1 and component.startswith("0"))
        for component in components
    ):
        return False
    return tuple(components[:2]) in PACKAGE_PATH_COMPATIBLE_ZIG_SERIES


def _unique_json_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    value: dict[str, object] = {}
    for name, member in pairs:
        if name in value:
            raise ValueError(f"duplicate JSON field: {name}")
        value[name] = member
    return value


def _relative_parts(rel: str) -> tuple[str, ...]:
    if (
        type(rel) is not str
        or rel.startswith(":")
        or any(character in rel for character in "*?[")
    ):
        raise ValueError(f"invalid package path: {rel}")
    try:
        return repository_snapshot.relative_path(rel)
    except repository_snapshot.RepositorySnapshotError as exc:
        raise ValueError(f"invalid package path: {rel}") from exc


def _validate_member_name(rel: str) -> None:
    parts = _relative_parts(rel)
    if any(part in FORBIDDEN_PATH_PARTS for part in parts):
        raise ValueError(f"package member contains local metadata: {rel}")
    relative = PurePosixPath(rel)
    if (
        relative.name in FORBIDDEN_MEMBER_NAMES
        or any(part in FORBIDDEN_MEMBER_PARTS for part in parts)
        or relative.suffix in FORBIDDEN_MEMBER_SUFFIXES
    ):
        raise ValueError(f"package member contains local metadata: {rel}")


def validate_path(root: Path, rel: str) -> None:
    parts = _relative_parts(rel)
    if any(part in FORBIDDEN_PATH_PARTS for part in parts):
        raise ValueError(f"forbidden package path: {rel}")
    try:
        with repository_snapshot.SnapshotSession(root) as session:
            node = session.capture_paths((rel,))[0]
            if node.kind not in {"regular", "directory"}:
                raise ValueError(
                    f"package path must be a regular file or directory: {rel}"
                )
            session.seal()
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"package path does not exist: {rel}") from exc
    except repository_snapshot.RepositorySnapshotError as exc:
        raise ValueError(
            f"package path must not traverse a symlink or unsafe component: {rel}"
        ) from exc


def _validate_package_paths(paths: Sequence[str]) -> None:
    seen: dict[tuple[str, ...], str] = {}
    for rel in paths:
        parts = _relative_parts(rel)
        if parts in seen:
            raise ValueError(f"duplicate package path: {rel}")
        for other_parts, other in seen.items():
            common = min(len(parts), len(other_parts))
            if parts[:common] == other_parts[:common]:
                raise ValueError(f"overlapping package paths: {other} and {rel}")
        seen[parts] = rel


def _repository_path_is_selected(rel: str, paths: Sequence[str]) -> bool:
    member_parts = PurePosixPath(rel).parts
    return any(
        member_parts[: len(PurePosixPath(path).parts)] == PurePosixPath(path).parts
        for path in paths
    )


def _validate_repository_files(
    paths: Sequence[str], file_set: repository_git.RepositoryFileSet
) -> None:
    for rel in file_set.listed:
        _validate_member_name(rel)
        if not _repository_path_is_selected(rel, paths):
            raise ValueError(f"Git returned a file outside package paths: {rel}")


def _package_member(
    session: repository_snapshot.SnapshotSession,
    rel: str,
    _directories: object = None,
) -> PackageMember:
    """Compatibility seam; all actual opening remains in repository_snapshot."""
    node = session.capture_paths((rel,))[0]
    if node.kind != "regular":
        raise ValueError(f"package member must be a regular file: {rel}")
    return node


def _capture_manifest_in_session(
    session: repository_snapshot.SnapshotSession, manifest: PackageMember | None
) -> PackageMember | None:
    if manifest is None:
        return None
    observed = session.capture_paths(
        (manifest.path,), include_bytes=True, limit=MAX_ZON_BYTES
    )[0]
    if observed != manifest:
        raise ValueError("build.zig.zon changed after parsing")
    return observed


def _validate_parsed_package_paths(root: Path, paths: PackagePaths) -> None:
    if type(paths) is not PackagePaths:
        raise TypeError("publishable package paths must come from package_paths()")
    proof = paths._parse_proof
    if (
        type(proof) is not _PackagePathsProof
        or proof.authority is not _PACKAGE_PATHS_FACTORY
        or proof.root is not paths.root
        or proof.paths is not paths.paths
        or proof.manifest is not paths.manifest
        or paths.root != root
        or paths.manifest.path != "build.zig.zon"
        or paths.manifest.bytes is None
    ):
        raise ValueError("package path parse proof is invalid or rebound")


def package_files(
    root: Path,
    paths: Sequence[str],
    *,
    repository: repository_git.RepositoryGit | None | object = _AUTO_REPOSITORY,
) -> PackageSnapshot:
    root = repository_git.strict_root(root)
    manifest: PackageMember | None = None
    if isinstance(paths, PackagePaths):
        _validate_parsed_package_paths(root, paths)
        manifest = paths.manifest
    _validate_package_paths(paths)
    if repository is _AUTO_REPOSITORY:
        repository = repository_git.open_repository(root)
    if repository is not None and not isinstance(
        repository, repository_git.RepositoryGit
    ):
        raise TypeError("repository must be a RepositoryGit handle or None")
    if repository is not None and repository.root != root:
        raise RuntimeError("package enumeration root does not match the repository")

    git_before: repository_git.RepositoryFileSet | None = None
    if repository is not None:
        git_before = repository.snapshot_file_set(paths)
        _validate_repository_files(paths, git_before)

    try:
        with repository_snapshot.SnapshotSession(root) as session:
            members: list[PackageMember] = []
            if repository is not None:
                _capture_manifest_in_session(session, manifest)
                selected_nodes = session.capture_paths(paths)
                for node in selected_nodes:
                    if node.kind not in {"regular", "directory"}:
                        kind = (
                            "symlink" if node.kind == "symlink" else "non-regular file"
                        )
                        raise ValueError(
                            f"package path must not be a {kind}: {node.path}"
                        )
                assert git_before is not None
                selected = set(git_before.present)
                for node in selected_nodes:
                    if node.kind == "regular" and node.path not in selected:
                        raise ValueError(
                            f"package file is excluded by repository policy: {node.path}"
                        )
                for rel in git_before.deleted:
                    session.prove_absent(rel)
                for rel in git_before.present:
                    members.append(_package_member(session, rel, None))
            else:
                walked = session.walk_archive(paths)
                selected_by_path = {node.path: node for node in walked}
                selected_nodes = tuple(selected_by_path[path] for path in paths)
                for node in selected_nodes:
                    if node.kind not in {"regular", "directory"}:
                        kind = (
                            "symlink" if node.kind == "symlink" else "non-regular file"
                        )
                        raise ValueError(
                            f"package path must not be a {kind}: {node.path}"
                        )
                captured_manifest = _capture_manifest_in_session(session, manifest)
                for node in walked:
                    _validate_member_name(node.path)
                    if node.kind == "symlink":
                        raise ValueError(
                            f"package member must not be a symlink: {node.path}"
                        )
                    if node.kind == "special":
                        raise ValueError(
                            f"package member must be a regular file: {node.path}"
                        )
                members.extend(
                    captured_manifest
                    if captured_manifest is not None
                    and node.path == captured_manifest.path
                    else node
                    for node in walked
                    if node.kind == "regular"
                )

            members = sorted(
                {member.path: member for member in members}.values(),
                key=lambda item: item.path,
            )
            tree = session.seal()
    except repository_snapshot.RepositorySnapshotError as exc:
        detail = str(exc)
        if "directory changed" in detail:
            detail = f"package member parent changed: {detail}"
        raise ValueError(
            f"package member changed during enumeration: {detail}"
        ) from exc

    if repository is not None:
        git_after = repository.snapshot_file_set(paths)
        if git_after != git_before:
            raise ValueError("Git file set changed during enumeration")

    observed_manifest = (
        next((member for member in members if member.path == manifest.path), None)
        if manifest is not None
        else None
    )
    if manifest is not None and observed_manifest != manifest:
        raise ValueError(
            "package manifest must include the exact bytes used to parse paths"
        )

    frozen_paths = paths.paths if type(paths) is PackagePaths else tuple(paths)
    return PackageSnapshot(
        root=root,
        paths=frozen_paths,
        members=tuple(members),
        tree=tree,
        repository=repository,
        git_files=git_before,
        manifest=manifest,
    )


def _member_git_object_id(root: Path, member: PackageMember, revision: str) -> str:
    algorithm = "sha1" if len(revision) == 40 else "sha256"
    object_digest = hashlib.new(algorithm, usedforsecurity=False)
    object_digest.update(f"blob {member.size}\0".encode("ascii"))
    content_digest = hashlib.sha256()
    try:
        source, _ = _open_package_member(root, member)
    except repository_snapshot.RepositorySnapshotError as exc:
        raise ValueError(
            f"package member changed while matching HEAD: {member.path}"
        ) from exc
    with source:
        remaining = member.size
        while remaining:
            chunk = source.read(min(1024 * 1024, remaining))
            if not isinstance(chunk, bytes) or not chunk:
                raise ValueError(
                    f"package member changed while matching HEAD: {member.path}"
                )
            remaining -= len(chunk)
            object_digest.update(chunk)
            content_digest.update(chunk)
        if source.read(1) or not _metadata_matches_member(
            os.fstat(source.fileno()), member
        ):
            raise ValueError(
                f"package member changed while matching HEAD: {member.path}"
            )
    if content_digest.hexdigest() != member.content_sha256:
        raise ValueError(f"package member content differs from HEAD: {member.path}")
    return object_digest.hexdigest()


def _validate_committed_publication_subject(
    root: Path,
    inspection: PackageSnapshot,
    publication: repository_git.RepositoryPublicationSnapshot,
) -> None:
    entries = {entry.path: entry for entry in publication.entries}
    if tuple(entries) != tuple(member.path for member in inspection.members):
        raise ValueError("package membership differs from the committed HEAD tree")
    for member in inspection.members:
        entry = entries[member.path]
        expected_mode = 0o100755 if member.mode & 0o111 else 0o100644
        if entry.mode != expected_mode:
            raise ValueError(f"package member mode differs from HEAD: {member.path}")
        if _member_git_object_id(root, member, publication.revision) != entry.object_id:
            raise ValueError(f"package member content differs from HEAD: {member.path}")


def publishable_package_files(
    root: Path, parsed_paths: PackagePaths
) -> PublishablePackageSnapshot:
    """Freeze clean HEAD subjects; inspection membership grants no authority."""
    root = repository_git.strict_root(root)
    _validate_parsed_package_paths(root, parsed_paths)
    repository = repository_git.open_repository(root)
    if repository is None:
        raise ValueError(
            "source archive creation requires the exact Git worktree top-level"
        )
    _verify_manifest_snapshot(root, parsed_paths.manifest)
    if "build.zig.zon" not in parsed_paths:
        raise ValueError("package manifest must include the exact parsed build.zig.zon")
    publication_before = repository.snapshot_publication(parsed_paths)
    inspection = package_files(root, parsed_paths, repository=repository)
    if inspection.git_files is None or inspection.manifest is None:
        raise ValueError("publishable package snapshot is missing provenance")
    publication_after = repository.snapshot_publication(parsed_paths)
    if publication_after != publication_before:
        raise ValueError("committed publication subject changed during enumeration")
    _validate_committed_publication_subject(root, inspection, publication_before)
    proof = _PublishableSnapshotProof(
        _PUBLISHABLE_SNAPSHOT_FACTORY,
        parsed_paths=parsed_paths,
        root=inspection.root,
        manifest=inspection.manifest,
        members=inspection.members,
        tree=inspection.tree,
        repository=repository,
        git_files=inspection.git_files,
        publication=publication_before,
    )
    snapshot = PublishablePackageSnapshot(
        root=inspection.root,
        paths=inspection.paths,
        members=inspection.members,
        tree=inspection.tree,
        repository=repository,
        git_files=inspection.git_files,
        manifest=inspection.manifest,
        _publication_proof=proof,
    )
    _validate_publication_authority(root, snapshot)
    return snapshot


def _read_stable_package_file(
    root: Path,
    rel: str,
    *,
    limit: int,
    description: str,
) -> tuple[bytes, PackageMember]:
    if description == "build.zig.zon" and rel != "build.zig.zon":
        raise ValueError(
            "package paths may only be parsed from the canonical root build.zig.zon"
        )
    try:
        with repository_snapshot.SnapshotSession(root) as session:
            node = session.capture_paths((rel,), include_bytes=True, limit=limit)[0]
            if node.kind != "regular":
                if node.kind == "symlink":
                    raise ValueError(f"unable to read {description} safely")
                raise ValueError(f"{description} must be a regular file")
            snapshot = session.seal()
            session.verify(snapshot)
    except FileNotFoundError as exc:
        raise ValueError(f"unable to read {description} safely") from exc
    except repository_snapshot.RepositorySnapshotError as exc:
        message = str(exc)
        if "input limit" in message:
            raise ValueError(f"{description} exceeds the parser input limit") from exc
        if "while being read" in message:
            raise ValueError(f"{description} changed while being read") from exc
        if "directory changed" in message:
            raise ValueError(f"{description} parent changed or is unsafe") from exc
        raise ValueError(f"unable to read {description} safely: {message}") from exc
    assert node.bytes is not None
    if description == "build.zig.zon":
        _PACKAGE_PARSE_CONTEXT.set((root, node, False))
    return node.bytes, node


def _verify_manifest_snapshot(root: Path, manifest: PackageMember) -> None:
    try:
        with repository_snapshot.SnapshotSession(root) as session:
            observed = session.capture_paths(
                (manifest.path,), include_bytes=True, limit=MAX_ZON_BYTES
            )[0]
            session.seal()
    except (FileNotFoundError, repository_snapshot.RepositorySnapshotError) as exc:
        raise ValueError("build.zig.zon changed after parsing") from exc
    if observed != manifest:
        raise ValueError("build.zig.zon changed after parsing")
    context = _PACKAGE_PARSE_CONTEXT.get()
    if (
        context is not None
        and context[0] == root
        and context[1] is manifest
        and not context[2]
    ):
        _PACKAGE_PARSE_CONTEXT.set((root, manifest, True))


class _SnapshotStream:
    def __init__(
        self, stream: BinaryIO, session: repository_snapshot.SnapshotSession
    ) -> None:
        self._stream = stream
        self._session = session

    def __enter__(self) -> _SnapshotStream:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def __getattr__(self, name: str) -> object:
        return getattr(self._stream, name)

    def read(self, size: int = -1) -> bytes:
        return self._stream.read(size)

    def fileno(self) -> int:
        return self._stream.fileno()

    def close(self) -> None:
        try:
            self._stream.close()
        finally:
            self._session.close()


def _open_package_member(
    root: Path, member: PackageMember
) -> tuple[BinaryIO, os.stat_result]:
    session = repository_snapshot.SnapshotSession(root)
    try:
        stream = session.open_verified_regular(member)
        metadata = os.fstat(stream.fileno())
    except Exception:
        session.close()
        raise
    return _SnapshotStream(stream, session), metadata  # type: ignore[return-value]


def normalized_tar_info(metadata: os.stat_result, rel: str) -> tarfile.TarInfo:
    info = tarfile.TarInfo(rel)
    info.size = metadata.st_size
    info.mode = 0o755 if metadata.st_mode & 0o111 else 0o644
    info.uid = 0
    info.gid = 0
    info.uname = ""
    info.gname = ""
    info.mtime = 0
    info.type = tarfile.REGTYPE
    info.pax_headers = {}
    return info


def _metadata_matches_member(metadata: os.stat_result, member: PackageMember) -> bool:
    identity = member.identity
    return (
        isinstance(identity, repository_snapshot.NodeIdentity)
        and metadata.st_dev == identity.device
        and metadata.st_ino == identity.inode
        and metadata.st_size == identity.size
        and metadata.st_mode == identity.mode
        and metadata.st_mtime_ns == identity.mtime_ns
        and metadata.st_ctime_ns == identity.ctime_ns
    )


def _verify_tree(root: Path, snapshot: PackageSnapshot, stage: str) -> None:
    try:
        with repository_snapshot.SnapshotSession(root) as session:
            session.verify(snapshot.tree)
    except repository_snapshot.RepositorySnapshotError as exc:
        detail = str(exc)
        if "content changed" in detail:
            raise ValueError(
                f"package member content changed {stage}: {detail}"
            ) from exc
        raise ValueError(f"package member changed {stage}: {detail}") from exc


def _verify_inspection_snapshot(
    root: Path, snapshot: PackageSnapshot, stage: str
) -> None:
    _verify_tree(root, snapshot, stage)
    if snapshot.repository is not None:
        observed = snapshot.repository.snapshot_file_set(snapshot.paths)
        if observed != snapshot.git_files:
            raise ValueError(f"Git file set changed {stage}")


def _validate_publication_authority(
    root: Path, snapshot: PackageSnapshot
) -> PublishablePackageSnapshot:
    if type(snapshot) is not PublishablePackageSnapshot:
        raise TypeError(
            "create_archive requires a PublishablePackageSnapshot from "
            "publishable_package_files()"
        )
    proof = snapshot._publication_proof
    if (
        type(proof) is not _PublishableSnapshotProof
        or proof.authority is not _PUBLISHABLE_SNAPSHOT_FACTORY
        or proof.root is not snapshot.root
        or proof.members is not snapshot.members
        or proof.tree is not snapshot.tree
        or proof.repository is not snapshot.repository
        or proof.git_files is not snapshot.git_files
        or proof.manifest is not snapshot.manifest
        or type(proof.publication) is not repository_git.RepositoryPublicationSnapshot
        or snapshot.root != root
        or snapshot.repository is None
        or snapshot.git_files is None
        or snapshot.manifest is None
    ):
        raise ValueError("publishable package snapshot proof is invalid or rebound")
    parsed_paths = proof.parsed_paths
    _validate_parsed_package_paths(root, parsed_paths)
    if (
        snapshot.paths is not parsed_paths.paths
        or snapshot.manifest is not parsed_paths.manifest
        or snapshot.repository.root != root
        or snapshot.tree.root != root
        or tuple(member.path for member in snapshot.members)
        != snapshot.git_files.present
        or tuple(member.path for member in snapshot.members)
        != tuple(entry.path for entry in proof.publication.entries)
        or "build.zig.zon" not in snapshot.paths
        or "build.zig.zon" not in snapshot.git_files.present
    ):
        raise ValueError("publishable package snapshot provenance is inconsistent")
    try:
        manifest_member = next(
            member for member in snapshot.members if member.path == "build.zig.zon"
        )
        tree_manifest = snapshot.tree.node("build.zig.zon")
    except (KeyError, StopIteration) as exc:
        raise ValueError("publishable package snapshot omits build.zig.zon") from exc
    if (
        manifest_member != snapshot.manifest
        or tree_manifest != snapshot.manifest
        or snapshot.manifest.bytes is None
    ):
        raise ValueError(
            "publishable package manifest differs from the parsed frozen manifest"
        )
    return snapshot


def _verify_package_snapshot(root: Path, snapshot: PackageSnapshot, stage: str) -> None:
    publishable = _validate_publication_authority(root, snapshot)
    _verify_inspection_snapshot(root, publishable, stage)
    observed_publication = publishable.repository.snapshot_publication(
        publishable.paths
    )
    if observed_publication != publishable._publication_proof.publication:
        raise ValueError(f"committed publication subject changed {stage}")
    _verify_manifest_snapshot(root, publishable.manifest)


def _copy_frozen_package_member(
    source: BinaryIO,
    contents: BinaryIO,
    member: PackageMember,
) -> str:
    digest = hashlib.sha256()
    remaining = member.size
    while remaining:
        requested = min(1024 * 1024, remaining)
        chunk = source.read(requested)
        if not isinstance(chunk, bytes):
            raise ValueError(
                f"package member returned an unexpected read result: {member.path}"
            )
        if not chunk:
            raise ValueError(
                f"package member reached early EOF while being read: {member.path}"
            )
        if len(chunk) > requested:
            raise ValueError(
                f"package member returned an oversized read: {member.path}"
            )
        digest.update(chunk)
        contents.write(chunk)
        remaining -= len(chunk)
    return digest.hexdigest()


@dataclasses.dataclass(frozen=True, slots=True)
class _Artifact:
    device: int
    inode: int
    size: int
    mode: int
    sha256: str


def _sha256_descriptor(descriptor: int, size: int, label: str) -> str:
    digest = hashlib.sha256()
    offset = 0
    while offset < size:
        requested = min(1024 * 1024, size - offset)
        chunk = os.pread(descriptor, requested, offset)
        if not isinstance(chunk, bytes):
            raise ValueError(f"{label} returned an unexpected positional read")
        if not chunk:
            raise ValueError(f"{label} reached early EOF while hashing")
        if len(chunk) > requested:
            raise ValueError(f"{label} returned an oversized positional read")
        digest.update(chunk)
        offset += len(chunk)
    return digest.hexdigest()


def _stable_artifact_metadata(metadata: os.stat_result) -> tuple[int, ...]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_size,
        metadata.st_mode,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
    )


def _record_artifact(descriptor: int, label: str) -> _Artifact:
    metadata = os.fstat(descriptor)
    if not stat.S_ISREG(metadata.st_mode):
        raise ValueError(f"{label} is not a regular file")
    digest = _sha256_descriptor(descriptor, metadata.st_size, label)
    observed = os.fstat(descriptor)
    if _stable_artifact_metadata(observed) != _stable_artifact_metadata(metadata):
        raise ValueError(f"{label} metadata changed while hashing")
    return _Artifact(
        device=metadata.st_dev,
        inode=metadata.st_ino,
        size=metadata.st_size,
        mode=metadata.st_mode,
        sha256=digest,
    )


def _same_inode(metadata: os.stat_result, artifact: _Artifact) -> bool:
    return (
        stat.S_ISREG(metadata.st_mode)
        and metadata.st_dev == artifact.device
        and metadata.st_ino == artifact.inode
    )


def _require_publication_support() -> None:
    required_dir_fd = (os.open, os.stat, os.link, os.unlink, os.mkdir, os.rmdir)
    if (
        os.name != "posix"
        or not hasattr(os, "O_DIRECTORY")
        or not hasattr(os, "O_NOFOLLOW")
        or any(function not in os.supports_dir_fd for function in required_dir_fd)
        or os.stat not in os.supports_follow_symlinks
        or os.link not in os.supports_follow_symlinks
    ):
        raise RuntimeError(
            "archive publication requires POSIX no-follow directory-relative APIs"
        )


def _component_name(name: str, label: str) -> str:
    if not name or name in {".", ".."} or os.sep in name:
        raise ValueError(f"{label} must be a single path component")
    if os.altsep is not None and os.altsep in name:
        raise ValueError(f"{label} must be a single path component")
    return name


def _open_directory_no_follow(path: Path, label: str) -> int:
    if not path.is_absolute():
        raise ValueError(f"{label} must be absolute")
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    try:
        descriptor = os.open(path.anchor, flags)
    except OSError as exc:
        raise ValueError(f"{label} is not a safe directory") from exc
    try:
        for part in path.parts[1:]:
            part = _component_name(part, label)
            try:
                child = os.open(part, flags, dir_fd=descriptor)
            except OSError as exc:
                raise ValueError(f"{label} is not a no-follow directory") from exc
            os.close(descriptor)
            descriptor = child
    except Exception:
        os.close(descriptor)
        raise
    return descriptor


def _verify_public_parent(parent: Path, directory_descriptor: int, stage: str) -> None:
    try:
        observed_descriptor = _open_directory_no_follow(
            parent, "archive destination parent"
        )
    except ValueError as exc:
        raise ValueError(f"archive destination parent changed {stage}") from exc
    try:
        expected = os.fstat(directory_descriptor)
        observed = os.fstat(observed_descriptor)
    finally:
        os.close(observed_descriptor)
    if (
        not stat.S_ISDIR(expected.st_mode)
        or not stat.S_ISDIR(observed.st_mode)
        or expected.st_dev != observed.st_dev
        or expected.st_ino != observed.st_ino
    ):
        raise ValueError(f"archive destination parent changed {stage}")


class _PackageDescriptorCloseUncertain(RuntimeError):
    """A transferred package descriptor was closed once with uncertain result."""

    def __init__(self, label: str, recovery_path: Path) -> None:
        self.recovery_path = recovery_path
        super().__init__(
            f"{label} descriptor close failed; recovery material retained as "
            f"{recovery_path}"
        )


def _close_package_descriptor(
    owner: repository_snapshot.OwnedDescriptor, label: str
) -> None:
    try:
        owner.close_once()
    except OSError as exc:
        raise _PackageDescriptorCloseUncertain(label, owner.recovery_path) from exc


def _open_name_no_follow(
    directory_descriptor: int,
    directory_path: Path,
    name: str,
    label: str,
) -> repository_snapshot.OwnedDescriptor:
    name = _component_name(name, label)
    recovery_path = directory_path / name
    try:
        descriptor = os.open(
            name, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=directory_descriptor
        )
    except FileNotFoundError:
        raise
    except OSError as exc:
        raise ValueError(f"{label} is not a safe regular file") from exc
    owner = repository_snapshot.OwnedDescriptor.take(descriptor, recovery_path)
    try:
        if not stat.S_ISREG(os.fstat(owner.fileno()).st_mode):
            raise ValueError(f"{label} is not a regular file")
    except Exception as exc:
        try:
            _close_package_descriptor(owner, label)
        except _PackageDescriptorCloseUncertain as close_exc:
            raise close_exc from exc
        raise
    return owner


def _verify_artifact_name(
    directory_descriptor: int,
    directory_path: Path,
    name: str,
    expected: _Artifact,
    label: str,
) -> None:
    owner = _open_name_no_follow(directory_descriptor, directory_path, name, label)
    try:
        observed = _record_artifact(owner.fileno(), label)
    except Exception as exc:
        try:
            _close_package_descriptor(owner, label)
        except _PackageDescriptorCloseUncertain as close_exc:
            raise close_exc from exc
        raise
    _close_package_descriptor(owner, label)
    if observed != expected:
        raise ValueError(f"{label} identity or digest changed")


def _name_names_inode(
    directory_descriptor: int, directory_path: Path, name: str, expected: _Artifact
) -> bool:
    try:
        owner = _open_name_no_follow(directory_descriptor, directory_path, name, name)
    except (FileNotFoundError, ValueError):
        return False
    try:
        matches = _same_inode(os.fstat(owner.fileno()), expected)
    except Exception as exc:
        try:
            _close_package_descriptor(owner, name)
        except _PackageDescriptorCloseUncertain as close_exc:
            raise close_exc from exc
        raise
    _close_package_descriptor(owner, name)
    return matches


def _verify_name_absent(directory_descriptor: int, name: str, label: str) -> None:
    name = _component_name(name, label)
    try:
        os.stat(name, dir_fd=directory_descriptor, follow_symlinks=False)
    except FileNotFoundError:
        return
    raise ValueError(f"{label} appeared before publication")


def _create_destination_backup(directory_descriptor: int, destination_name: str) -> str:
    for _ in range(128):
        candidate = _component_name(
            f".{destination_name}.{secrets.token_hex(16)}.backup",
            "archive destination backup",
        )
        try:
            os.link(
                destination_name,
                candidate,
                src_dir_fd=directory_descriptor,
                dst_dir_fd=directory_descriptor,
                follow_symlinks=False,
            )
        except FileExistsError:
            continue
        return candidate
    raise RuntimeError("unable to allocate a unique archive destination backup")


def _replace_name(directory_descriptor: int, source: str, destination: str) -> None:
    try:
        os.replace(
            _component_name(source, "archive replace source"),
            _component_name(destination, "archive replace destination"),
            src_dir_fd=directory_descriptor,
            dst_dir_fd=directory_descriptor,
        )
    except TypeError as exc:
        raise RuntimeError(
            "archive publication requires directory-relative replace"
        ) from exc


def _cleanup_anchor(
    directory_descriptor: int, directory_path: Path
) -> repository_snapshot.DirectoryAnchor:
    if not directory_path.is_absolute():
        raise ValueError("archive cleanup directory path must be absolute")
    return repository_snapshot.DirectoryAnchor(directory_descriptor, directory_path)


def _audit_cleanup_arena_empty(
    directory_descriptor: int,
    directory_path: Path,
    stage: str,
    *,
    create: bool = True,
    expected_identity: tuple[int, int] | None = None,
) -> tuple[int, int]:
    anchor = _cleanup_anchor(directory_descriptor, directory_path)
    try:
        arena = repository_snapshot.CleanupArena.open(
            anchor,
            create=create,
            expected_identity=expected_identity,
        )
    except repository_snapshot.CleanupFailure as exc:
        recovery_paths = _absolute_cleanup_recovery_paths(exc.outcome, directory_path)
        diagnostics = _absolute_cleanup_paths(
            (issue.path for issue in exc.outcome.issues), directory_path
        )
        details = (
            "diagnostic paths: " + "; ".join(os.fspath(path) for path in diagnostics)
            if diagnostics
            else f"diagnostic path: {directory_path}"
        )
        if recovery_paths:
            details += "; recovery material retained as " + "; ".join(
                os.fspath(path) for path in recovery_paths
            )
        candidate = _cleanup_candidate_description((exc.outcome,), directory_path)
        if candidate is not None:
            details += f"; {candidate}"
        addressability = _cleanup_addressability_description((exc.outcome,))
        if addressability is not None:
            details += f"; {addressability}"
        cause = _cleanup_cause(exc.outcome) or exc
        raise RuntimeError(
            f"archive cleanup arena validation failed {stage}; {details}"
        ) from cause

    audit_error: BaseException | None = None
    inspection_binding = repository_snapshot.ArenaBinding.BOUND
    try:
        os.fsync(arena.descriptor.fileno())
        with os.scandir(arena.descriptor.fileno()) as entries:
            first = next(entries, None)
        if first is not None:
            audit_error = RuntimeError(
                "archive cleanup arena contains retained material at "
                f"{arena.path / first.name}"
            )
        inspection_binding = arena.binding()
        if inspection_binding is not repository_snapshot.ArenaBinding.BOUND:
            binding_error = (
                "archive cleanup arena binding changed after inspection: "
                f"{inspection_binding.name.lower()} at {arena.path}"
            )
            preceding = f"{audit_error}; " if audit_error is not None else ""
            audit_error = RuntimeError(f"{preceding}{binding_error}")
    except BaseException as exc:
        audit_error = exc

    outcome = repository_snapshot.finalize_arena_outcome(
        arena,
        repository_snapshot.CleanupOutcome(
            (
                repository_snapshot.CleanupDisposition.ABSENT
                if inspection_binding is repository_snapshot.ArenaBinding.BOUND
                else repository_snapshot.CleanupDisposition.UNADDRESSABLE
            ),
            (),
            (),
            arena_binding=inspection_binding,
        ),
    )
    close_issue = next(
        (
            issue
            for issue in outcome.issues
            if issue.code == "cleanup_arena_descriptor_close_uncertain"
        ),
        None,
    )
    if close_issue is not None:
        message = (
            f"archive cleanup arena descriptor close failed {stage}; cleanup arena "
            f"retained as {arena.path}"
        )
        if audit_error is not None:
            message += f"; preceding arena audit failed: {audit_error}"
        if outcome.arena_binding is not repository_snapshot.ArenaBinding.BOUND:
            message += (
                "; cleanup arena binding changed after descriptor close: "
                f"{outcome.arena_binding.name.lower()} at {arena.path}"
            )
        cause = close_issue.error or audit_error
        if cause is None:
            raise RuntimeError(message)
        raise RuntimeError(message) from cause
    if (
        outcome.disposition is repository_snapshot.CleanupDisposition.UNADDRESSABLE
        or outcome.arena_binding is not repository_snapshot.ArenaBinding.BOUND
    ):
        preceding = f"{audit_error}; " if audit_error is not None else ""
        raise RuntimeError(
            f"archive cleanup arena is not release-clean {stage}: {preceding}"
            "cleanup arena binding changed after descriptor close: "
            f"{outcome.arena_binding.name.lower()} at {arena.path}"
        )
    if audit_error is not None:
        raise RuntimeError(
            f"archive cleanup arena is not release-clean {stage}: {audit_error}; "
            f"cleanup arena retained as {arena.path}"
        ) from audit_error
    if (
        outcome.disposition is not repository_snapshot.CleanupDisposition.ABSENT
        or outcome.arena_identity is None
    ):
        raise RuntimeError(
            f"archive cleanup arena is not release-clean {stage}: finalized cleanup "
            f"disposition is {outcome.disposition.name.lower()} at {arena.path}"
        )
    return outcome.arena_identity


def _artifact_cleanup_verifier(
    expected: _Artifact, label: str
) -> Callable[[int, os.stat_result], repository_snapshot.ClaimVerification]:
    def verify(
        descriptor: int, _: os.stat_result
    ) -> repository_snapshot.ClaimVerification:
        observed = _record_artifact(descriptor, label)
        if observed == expected:
            return repository_snapshot.ClaimVerification.MATCH
        return repository_snapshot.ClaimVerification.FOREIGN

    return verify


def _cleanup_issue_codes(
    outcome: repository_snapshot.CleanupOutcome,
) -> frozenset[str]:
    return frozenset(issue.code for issue in outcome.issues)


def _cleanup_cause(
    outcome: repository_snapshot.CleanupOutcome,
) -> BaseException | None:
    return next(
        (issue.error for issue in outcome.issues if issue.error is not None), None
    )


def _cleanup_issue_path(
    outcome: repository_snapshot.CleanupOutcome, *codes: str
) -> str | None:
    expected = frozenset(codes)
    return next(
        (os.fspath(issue.path) for issue in outcome.issues if issue.code in expected),
        None,
    )


def _cleanup_arena_uncertainty_description(
    outcome: repository_snapshot.CleanupOutcome,
) -> str | None:
    codes = _cleanup_issue_codes(outcome)
    arena_codes = {
        "cleanup_arena_fsync_failed",
        "cleanup_arena_descriptor_close_uncertain",
    }
    if not codes & arena_codes:
        return None
    if arena_codes <= codes:
        uncertainty = "durability and descriptor-close uncertainty"
    elif "cleanup_arena_fsync_failed" in codes:
        uncertainty = "durability uncertainty"
    else:
        uncertainty = "descriptor-close uncertainty"
    paths = sorted(
        {issue.path for issue in outcome.issues if issue.code in arena_codes}
    )
    if not paths:
        return f"cleanup arena {uncertainty}"
    return f"cleanup arena {uncertainty} at " + "; ".join(
        os.fspath(path) for path in paths
    )


def _absolute_cleanup_paths(
    paths: Iterable[Path], directory_path: Path
) -> tuple[Path, ...]:
    if not directory_path.is_absolute():
        raise ValueError("archive cleanup directory path must be absolute")
    return tuple(
        sorted(
            {path if path.is_absolute() else directory_path / path for path in paths},
            key=os.fspath,
        )
    )


def _absolute_cleanup_recovery_paths(
    outcome: repository_snapshot.CleanupOutcome,
    directory_path: Path,
) -> tuple[Path, ...]:
    return _absolute_cleanup_paths(outcome.recovery_paths, directory_path)


def _cleanup_candidate_description(
    outcomes: Iterable[repository_snapshot.CleanupOutcome],
    directory_path: Path,
) -> str | None:
    outcomes = tuple(outcomes)
    candidates = _absolute_cleanup_paths(
        (path for outcome in outcomes for path in outcome.candidate_paths),
        directory_path,
    )
    if not candidates:
        return None
    states = {outcome.public_candidate for outcome in outcomes}
    if repository_snapshot.PublicCandidate.UNKNOWN in states:
        observation = "presence is uncertain"
    elif repository_snapshot.PublicCandidate.PRESENT in states:
        observation = "was observed present"
    else:
        observation = "requires inspection"
    return f"unclaimed candidate {observation} at " + "; ".join(
        os.fspath(path) for path in candidates
    )


def _cleanup_addressability_description(
    outcomes: Iterable[repository_snapshot.CleanupOutcome],
) -> str | None:
    outcomes = tuple(outcomes)
    bindings = {
        outcome.arena_binding
        for outcome in outcomes
        if outcome.arena_binding is not repository_snapshot.ArenaBinding.BOUND
    }
    unaddressable = any(
        outcome.disposition is repository_snapshot.CleanupDisposition.UNADDRESSABLE
        for outcome in outcomes
    )
    if not unaddressable and not bindings:
        return None
    status = "unaddressable" if unaddressable else "tainted"
    binding = ", ".join(sorted(item.name.lower() for item in bindings)) or "bound"
    return f"cleanup namespace is {status}; arena binding is {binding}"


def _raise_cleanup_outcome(
    outcome: repository_snapshot.CleanupOutcome,
    directory_path: Path,
    public_name: str,
    claimed_name: str,
    label: str,
) -> None:
    public_path = directory_path / public_name
    codes = _cleanup_issue_codes(outcome)
    cause = _cleanup_cause(outcome)
    recovery_material = "; ".join(
        os.fspath(path)
        for path in _absolute_cleanup_recovery_paths(outcome, directory_path)
    )
    arena_uncertainty = _cleanup_arena_uncertainty_description(outcome)
    candidate = _cleanup_candidate_description((outcome,), directory_path)
    addressability = _cleanup_addressability_description((outcome,))
    if outcome.disposition is repository_snapshot.CleanupDisposition.UNADDRESSABLE:
        retained = (
            f"; recovery material retained as {recovery_material}"
            if recovery_material
            else ""
        )
        message = f"{label} cleanup failed closed{retained}"
    elif "cleanup_public_name_reappeared" in codes:
        recovery = (
            f"; recovery material retained as {recovery_material}"
            if recovery_material
            else ""
        )
        message = (
            f"{label} cleanup public pathname reappeared and was preserved as "
            f"{public_path}{recovery}"
        )
    elif codes & {
        "cleanup_claimed_foreign",
        "cleanup_claimed_unknown",
        "cleanup_claimed_uninspectable",
        "cleanup_claimed_descriptor_close_uncertain",
    }:
        recovery = recovery_material or os.fspath(public_path)
        message = (
            f"{label} cleanup claimed an unexpected artifact; recovery material "
            f"retained as {recovery}"
        )
    elif codes & {
        "cleanup_claim_destination_fsync_failed",
        "cleanup_claim_source_fsync_failed",
    }:
        destination_failed = "cleanup_claim_destination_fsync_failed" in codes
        source_failed = "cleanup_claim_source_fsync_failed" in codes
        if destination_failed and source_failed:
            failure = "destination quarantine and source parent fsyncs failed"
        elif destination_failed:
            failure = "destination quarantine fsync failed"
        else:
            failure = "source parent fsync failed"
        recovery = recovery_material or os.fspath(public_path)
        message = (
            f"{label} cleanup claim durability failed: {failure}; recovery material "
            f"retained as {recovery}"
        )
    elif "cleanup_source_fsync_failed" in codes:
        retained = (
            f"; recovery material retained as {recovery_material}"
            if recovery_material
            else ""
        )
        message = (
            f"{label} cleanup fsync failed; public pathname was observed absent "
            f"as {public_path}{retained}"
        )
    elif arena_uncertainty is not None:
        retained = (
            f"; recovery material retained as {recovery_material}"
            if recovery_material
            else ""
        )
        message = f"{label} cleanup failed; {arena_uncertainty}{retained}"
    elif codes & {
        "cleanup_claim_failed",
        "cleanup_public_absence_uninspectable",
    }:
        retained = (
            f"; recovery material retained as {recovery_material}"
            if recovery_material
            else ""
        )
        message = f"{label} cleanup claim failed{retained}"
    elif codes & {
        "cleanup_arena_create_failed",
        "cleanup_arena_setup_failed",
        "cleanup_arena_descriptor_close_uncertain",
        "cleanup_arena_source_device_mismatch",
    }:
        if recovery_material:
            message = (
                f"{label} cleanup arena setup failed; recovery material retained "
                f"as {recovery_material}"
            )
        else:
            arena = _cleanup_issue_path(
                outcome,
                "cleanup_arena_create_failed",
                "cleanup_arena_setup_failed",
                "cleanup_arena_descriptor_close_uncertain",
                "cleanup_arena_source_device_mismatch",
            ) or os.fspath(public_path)
            message = f"{label} cleanup arena setup failed; diagnostic path: {arena}"
    elif codes & {
        "cleanup_quarantine_create_failed",
        "cleanup_quarantine_setup_failed",
        "cleanup_quarantine_descriptor_close_uncertain",
        "cleanup_quarantine_name_exhausted",
    }:
        if recovery_material:
            message = (
                f"{label} cleanup quarantine setup failed; recovery material "
                f"retained as {recovery_material}"
            )
        else:
            residue = _cleanup_issue_path(
                outcome,
                "cleanup_quarantine_create_failed",
                "cleanup_quarantine_setup_failed",
                "cleanup_quarantine_descriptor_close_uncertain",
                "cleanup_quarantine_name_exhausted",
            ) or os.fspath(public_path)
            message = (
                f"{label} cleanup quarantine setup failed; quarantine residue at "
                f"{residue}"
            )
    elif addressability is not None:
        message = f"{label} cleanup failed closed"
    else:
        residue = (
            f"recovery material retained as {recovery_material}"
            if recovery_material
            else f"cleanup quarantine residue retained as {public_path}"
        )
        message = f"{label} cleanup failed; {residue}"
    if candidate is not None:
        message += f"; {candidate}"
    if addressability is not None:
        message += f"; {addressability}"
    if cause is None:
        raise RuntimeError(message)
    raise RuntimeError(message) from cause


def _claim_and_remove_public_artifact(
    directory_descriptor: int,
    directory_path: Path,
    public_name: str,
    expected: _Artifact,
    label: str,
) -> None:
    claimed_name = "claimed"
    try:
        outcome = repository_snapshot.claim_and_remove(
            _cleanup_anchor(directory_descriptor, directory_path),
            public_name,
            _artifact_cleanup_verifier(expected, f"claimed {label}"),
            quarantine_prefix=f".{public_name}.",
            quarantine_suffix=".cleanup",
            claimed_name=claimed_name,
        )
    except repository_snapshot.CleanupFailure as exc:
        _raise_cleanup_outcome(
            exc.outcome, directory_path, public_name, claimed_name, label
        )
    if outcome.disposition is repository_snapshot.CleanupDisposition.REMOVED:
        return
    if outcome.disposition is repository_snapshot.CleanupDisposition.ABSENT:
        raise FileNotFoundError(public_name)
    _raise_cleanup_outcome(outcome, directory_path, public_name, claimed_name, label)


def _refresh_owned_public_artifact(
    directory_descriptor: int,
    directory_path: Path,
    public_name: str,
    ownership: _Artifact,
    label: str,
) -> _Artifact:
    owner = _open_name_no_follow(
        directory_descriptor, directory_path, public_name, label
    )
    try:
        if not _same_inode(os.fstat(owner.fileno()), ownership):
            raise ValueError(f"{label} pathname no longer names the owned inode")
        observed = _record_artifact(owner.fileno(), label)
    except Exception as exc:
        try:
            _close_package_descriptor(owner, label)
        except _PackageDescriptorCloseUncertain as close_exc:
            raise close_exc from exc
        raise
    _close_package_descriptor(owner, label)
    return observed


def _rollback_recovery_description(
    directory_path: Path,
    backup_name: str | None,
    outcome: repository_snapshot.CleanupOutcome,
    *additional_outcomes: repository_snapshot.CleanupOutcome,
) -> str:
    outcomes = (outcome, *additional_outcomes)
    accumulator = repository_snapshot.CleanupAccumulator()
    for observed in outcomes:
        accumulator.record(observed)
    aggregate = accumulator.snapshot()
    details: list[str] = []
    recovery_paths = _absolute_cleanup_paths(
        aggregate.recovery_paths,
        directory_path,
    )
    if recovery_paths:
        details.append(
            "recovery material retained as "
            + "; ".join(os.fspath(path) for path in recovery_paths)
        )
    arena_uncertainties = sorted(
        {
            uncertainty
            for observed in outcomes
            if (uncertainty := _cleanup_arena_uncertainty_description(observed))
            is not None
        }
    )
    details.extend(arena_uncertainties)
    candidate_paths = _absolute_cleanup_paths(
        aggregate.candidate_paths,
        directory_path,
    )
    if candidate_paths:
        if aggregate.public_candidate is repository_snapshot.PublicCandidate.UNKNOWN:
            observation = "presence is uncertain"
        elif aggregate.public_candidate is repository_snapshot.PublicCandidate.PRESENT:
            observation = "was observed present"
        else:
            observation = "requires inspection"
        details.append(
            f"unclaimed candidate {observation} at "
            + "; ".join(os.fspath(path) for path in candidate_paths)
        )
    addressability = _cleanup_addressability_description(outcomes)
    if addressability is not None:
        details.append(addressability)
    if aggregate.unaddressable:
        issues = sorted(
            {
                (
                    issue.code,
                    issue.path
                    if issue.path.is_absolute()
                    else directory_path / issue.path,
                )
                for issue in aggregate.issues
            },
            key=lambda item: (item[0], os.fspath(item[1])),
        )
        if issues:
            details.append(
                "cleanup issues: "
                + "; ".join(f"{code} at {path}" for code, path in issues)
            )
    elif backup_name is not None:
        details.append(f"prior destination retained as {directory_path / backup_name}")
    if not details:
        details.append("cleanup completed without retained recovery paths")
    return "; ".join(details)


def _rollback_finish_and_raise(
    quarantine: repository_snapshot.CleanupQuarantine,
    directory_path: Path,
    backup_name: str | None,
    message: str,
    cause: BaseException | None = None,
    *,
    expect_public_absent: bool,
    prior_outcome: repository_snapshot.CleanupOutcome | None = None,
) -> None:
    outcome = quarantine.finish(expect_public_absent=expect_public_absent)
    additional = () if prior_outcome is None else (prior_outcome,)
    recovery = _rollback_recovery_description(
        directory_path, backup_name, outcome, *additional
    )
    error = RuntimeError(f"rollback_indeterminate: {message}; {recovery}")
    if cause is None:
        raise error
    raise error from cause


def _rollback_cleanup_failure(
    quarantine: repository_snapshot.CleanupQuarantine,
    directory_path: Path,
    backup_name: str | None,
    outcome: repository_snapshot.CleanupOutcome,
) -> None:
    codes = _cleanup_issue_codes(outcome)
    recovery = _rollback_recovery_description(directory_path, backup_name, outcome)
    if "cleanup_source_fsync_failed" in codes:
        detail = "archive quarantine cleanup durability failed after teardown"
    else:
        detail = "archive quarantine cleanup failed"
    cause = _cleanup_cause(outcome)
    error = RuntimeError(f"rollback_indeterminate: {detail}; {recovery}")
    if cause is None:
        raise error
    raise error from cause


def _rollback_publication(
    directory_descriptor: int,
    directory_path: Path,
    destination_name: str,
    prepared: _Artifact,
    backup_name: str | None,
    original: _Artifact | None,
) -> None:
    claimed_name = "published-artifact"
    try:
        quarantine = repository_snapshot.CleanupQuarantine.create(
            _cleanup_anchor(directory_descriptor, directory_path),
            destination_name,
            quarantine_prefix=f".{destination_name}.",
            quarantine_suffix=".quarantine",
            claimed_name=claimed_name,
        )
    except repository_snapshot.CleanupFailure as exc:
        recovery = _rollback_recovery_description(
            directory_path, backup_name, exc.outcome
        )
        cause = _cleanup_cause(exc.outcome) or exc
        raise RuntimeError(
            "rollback_indeterminate: unable to create archive rollback "
            f"quarantine; {recovery}"
        ) from cause

    try:
        quarantine.claim()
    except repository_snapshot.CleanupFailure as exc:
        codes = _cleanup_issue_codes(exc.outcome)
        if codes & {
            "cleanup_claim_destination_fsync_failed",
            "cleanup_claim_source_fsync_failed",
        }:
            destination_failed = "cleanup_claim_destination_fsync_failed" in codes
            source_failed = "cleanup_claim_source_fsync_failed" in codes
            if destination_failed and source_failed:
                detail = "destination quarantine and source parent fsyncs failed"
            elif destination_failed:
                detail = "destination quarantine fsync failed"
            else:
                detail = "source parent fsync failed"
            message = f"archive rollback claim durability failed: {detail}"
        else:
            message = "unable to claim the archive destination"
        _rollback_finish_and_raise(
            quarantine,
            directory_path,
            backup_name,
            message,
            _cleanup_cause(exc.outcome) or exc,
            expect_public_absent=False,
            prior_outcome=exc.outcome,
        )
    if quarantine.phase is repository_snapshot.CleanupPhase.ABSENT:
        _rollback_finish_and_raise(
            quarantine,
            directory_path,
            backup_name,
            "archive destination was absent before rollback claim",
            expect_public_absent=original is None,
        )

    try:
        verification = quarantine.verify_claimed(
            _artifact_cleanup_verifier(prepared, "claimed archive destination")
        )
    except repository_snapshot.CleanupFailure as exc:
        _rollback_finish_and_raise(
            quarantine,
            directory_path,
            backup_name,
            "claimed destination could not be verified as the prepared archive",
            _cleanup_cause(exc.outcome) or exc,
            expect_public_absent=False,
        )
    if verification is not repository_snapshot.ClaimVerification.MATCH:
        _rollback_finish_and_raise(
            quarantine,
            directory_path,
            backup_name,
            "claimed destination is not the prepared archive",
            expect_public_absent=False,
        )

    if original is None:
        try:
            _verify_name_absent(
                directory_descriptor,
                destination_name,
                "archive destination during rollback",
            )
        except Exception as exc:
            _rollback_finish_and_raise(
                quarantine,
                directory_path,
                backup_name,
                "archive destination reappeared while rolling back an originally "
                "absent destination",
                exc,
                expect_public_absent=False,
            )
    else:
        if backup_name is None:
            _rollback_finish_and_raise(
                quarantine,
                directory_path,
                backup_name,
                "prior destination backup is unavailable",
                expect_public_absent=True,
            )
        try:
            _verify_artifact_name(
                directory_descriptor,
                directory_path,
                backup_name,
                original,
                "archive destination rollback backup",
            )
        except Exception as exc:
            _rollback_finish_and_raise(
                quarantine,
                directory_path,
                backup_name,
                "prior destination backup changed before restoration",
                exc,
                expect_public_absent=True,
            )
        try:
            os.link(
                backup_name,
                destination_name,
                src_dir_fd=directory_descriptor,
                dst_dir_fd=directory_descriptor,
                follow_symlinks=False,
            )
        except FileExistsError as exc:
            _rollback_finish_and_raise(
                quarantine,
                directory_path,
                backup_name,
                "archive destination reappeared before old destination restoration",
                exc,
                expect_public_absent=False,
            )
        except Exception as exc:
            _rollback_finish_and_raise(
                quarantine,
                directory_path,
                backup_name,
                "old archive destination restoration failed",
                exc,
                expect_public_absent=True,
            )
        try:
            _verify_artifact_name(
                directory_descriptor,
                directory_path,
                destination_name,
                original,
                "restored archive destination",
            )
        except Exception as exc:
            _rollback_finish_and_raise(
                quarantine,
                directory_path,
                backup_name,
                "restored archive destination changed",
                exc,
                expect_public_absent=False,
            )

    try:
        quarantine.remove_verified_claim(expect_public_absent=original is None)
    except repository_snapshot.CleanupFailure:
        outcome = quarantine.finish(expect_public_absent=original is None)
        _rollback_cleanup_failure(quarantine, directory_path, backup_name, outcome)

    if original is None:
        try:
            _verify_name_absent(
                directory_descriptor,
                destination_name,
                "archive destination after rollback",
            )
        except Exception as exc:
            _rollback_finish_and_raise(
                quarantine,
                directory_path,
                backup_name,
                "archive destination reappeared after rollback",
                exc,
                expect_public_absent=False,
            )

    outcome = quarantine.finish(expect_public_absent=original is None)
    if outcome.disposition is not repository_snapshot.CleanupDisposition.REMOVED:
        _rollback_cleanup_failure(quarantine, directory_path, backup_name, outcome)

    if original is None:
        raise RuntimeError(
            "rollback_indeterminate: failed archive publication was removed and "
            "the originally absent destination was observed absent"
        )

    assert backup_name is not None
    try:
        _verify_artifact_name(
            directory_descriptor,
            directory_path,
            destination_name,
            original,
            "restored archive destination",
        )
        _verify_artifact_name(
            directory_descriptor,
            directory_path,
            backup_name,
            original,
            "archive destination rollback backup",
        )
    except Exception as exc:
        raise RuntimeError(
            "rollback_indeterminate: restored namespace changed; prior "
            f"destination retained as {directory_path / backup_name}"
        ) from exc
    raise RuntimeError(
        "rollback_indeterminate: old destination was observed restored; prior "
        f"destination retained as {directory_path / backup_name}"
    )


def _cleanup_backup_after_transaction(
    directory_descriptor: int,
    directory_path: Path,
    backup_name: str,
    original: _Artifact,
) -> None:
    label = "archive destination backup"
    try:
        _claim_and_remove_public_artifact(
            directory_descriptor,
            directory_path,
            backup_name,
            original,
            label,
        )
    except Exception as exc:
        raise RuntimeError(
            "archive transaction complete but backup cleanup failed for public "
            f"backup {directory_path / backup_name}: {exc}"
        ) from exc


class _PublicationState(enum.Enum):
    MATERIALIZING = enum.auto()
    PREPARED = enum.auto()
    COMMITTED = enum.auto()
    PUBLISHED_SOURCE_VERIFIED = enum.auto()
    TRANSACTION_COMPLETE = enum.auto()


class _ArchiveStreamOwner:
    """Attempt archive stream close once and remember an uncertain failure."""

    def __init__(self, stream: BinaryIO, recovery_path: Path) -> None:
        self.stream = stream
        self.recovery_path = recovery_path
        self.close_attempted = False
        self.close_uncertain = False

    def close_once(self) -> None:
        if self.close_attempted:
            return
        self.close_attempted = True
        try:
            self.stream.close()
        except OSError as exc:
            self.close_uncertain = True
            raise _PackageDescriptorCloseUncertain(
                "archive temporary stream", self.recovery_path
            ) from exc

    def __enter__(self) -> BinaryIO:
        return self.stream

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: object,
    ) -> None:
        self.close_once()


def _temporary_candidate_diagnostic(detail: str, path: Path) -> str:
    return f"{detail}; unclaimed candidate presence is uncertain at {path}"


def _create_temporary_archive(
    directory_descriptor: int, directory_path: Path, destination_name: str
) -> tuple[str, BinaryIO, _Artifact]:
    flags = os.O_RDWR | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW
    for _ in range(128):
        name = _component_name(
            f".{destination_name}.{secrets.token_hex(16)}.tmp",
            "archive temporary",
        )
        try:
            descriptor = os.open(name, flags, 0o600, dir_fd=directory_descriptor)
        except FileExistsError:
            continue
        recovery_path = directory_path / name
        owner = repository_snapshot.OwnedDescriptor.take(descriptor, recovery_path)
        try:
            created = _record_artifact(owner.fileno(), "archive temporary")
        except Exception as exc:
            try:
                _close_package_descriptor(owner, "archive temporary")
            except _PackageDescriptorCloseUncertain as close_exc:
                raise RuntimeError(
                    _temporary_candidate_diagnostic(
                        "archive temporary identity and descriptor close are uncertain",
                        recovery_path,
                    )
                ) from close_exc
            raise RuntimeError(
                _temporary_candidate_diagnostic(
                    "archive temporary identity is unknown", recovery_path
                )
            ) from exc
        descriptor = owner.transfer()
        try:
            stream = os.fdopen(descriptor, "w+b")
        except Exception as exc:
            raise RuntimeError(
                _temporary_candidate_diagnostic(
                    "archive temporary stream creation failed after descriptor "
                    "ownership transfer",
                    recovery_path,
                )
            ) from exc
        return name, stream, created
    raise RuntimeError("unable to allocate a unique archive temporary")


def _materialize_archive_transaction(
    root: Path,
    files: PackageSnapshot,
    destination: Path,
    *,
    verify_snapshot: Callable[[Path, PackageSnapshot, str], None],
) -> None:
    """Private archive transaction; callers supply the source-state verifier."""
    root = repository_git.strict_root(root)
    if not isinstance(files, PackageSnapshot):
        raise TypeError("create_archive requires an immutable PackageSnapshot")
    if files.root != root:
        raise ValueError("package snapshot root does not match archive root")

    _require_publication_support()
    verifier = verify_snapshot
    verifier(root, files, "after enumeration")
    destination_name = _component_name(destination.name, "archive destination")
    lexical_parent = Path(os.path.abspath(os.fspath(destination.parent)))
    lexical_parent.mkdir(parents=True, exist_ok=True)
    try:
        directory_path = lexical_parent.resolve(strict=True)
    except OSError as exc:
        raise ValueError("archive destination parent is not stable") from exc
    directory_descriptor = _open_directory_no_follow(
        directory_path, "archive destination parent"
    )
    state = _PublicationState.MATERIALIZING
    try:
        entry_arena_identity = _audit_cleanup_arena_empty(
            directory_descriptor,
            directory_path,
            "at archive publication entry",
        )
        temporary_name, raw, created = _create_temporary_archive(
            directory_descriptor, directory_path, destination_name
        )
        temporary_path = directory_path / temporary_name
        raw_owner = _ArchiveStreamOwner(raw, temporary_path)
        prepared: _Artifact | None = None
        original: _Artifact | None = None
        original_owner: repository_snapshot.OwnedDescriptor | None = None
        backup_name: str | None = None
        transaction_error: Exception | None = None
        try:
            try:
                try:
                    original_owner = _open_name_no_follow(
                        directory_descriptor,
                        directory_path,
                        destination_name,
                        "archive destination",
                    )
                except FileNotFoundError:
                    _verify_name_absent(
                        directory_descriptor,
                        destination_name,
                        "archive destination",
                    )
                else:
                    original = _record_artifact(
                        original_owner.fileno(), "archive destination"
                    )

                with raw_owner as raw:
                    with gzip.GzipFile(
                        filename="", mode="wb", compresslevel=9, fileobj=raw, mtime=0
                    ) as compressed:
                        with tarfile.open(
                            fileobj=compressed, mode="w", format=tarfile.PAX_FORMAT
                        ) as archive:
                            for member in files:
                                try:
                                    source, metadata = _open_package_member(
                                        root, member
                                    )
                                except (
                                    repository_snapshot.RepositorySnapshotError
                                ) as exc:
                                    raise ValueError(
                                        "package member changed after enumeration: "
                                        f"{member.path}"
                                    ) from exc
                                with source:
                                    with tempfile.SpooledTemporaryFile(
                                        max_size=8 * 1024 * 1024, mode="w+b"
                                    ) as contents:
                                        content_sha256 = _copy_frozen_package_member(
                                            source, contents, member
                                        )
                                        if (
                                            content_sha256 != member.content_sha256
                                            or not _metadata_matches_member(
                                                os.fstat(source.fileno()), member
                                            )
                                        ):
                                            raise ValueError(
                                                "package member content changed after "
                                                f"enumeration: {member.path}"
                                            )
                                        contents.seek(0)
                                        archive.addfile(
                                            normalized_tar_info(metadata, member.path),
                                            contents,
                                        )
                    raw.flush()
                    os.fsync(raw.fileno())
                    os.fchmod(raw.fileno(), 0o644)
                    os.fsync(raw.fileno())
                    prepared = _record_artifact(raw.fileno(), "prepared archive")

                state = _PublicationState.PREPARED
                verifier(root, files, "during archive materialization")
                _verify_artifact_name(
                    directory_descriptor,
                    directory_path,
                    temporary_name,
                    prepared,
                    "prepared archive pathname",
                )

                if original is None:
                    _verify_name_absent(
                        directory_descriptor,
                        destination_name,
                        "archive destination",
                    )
                else:
                    backup_name = _create_destination_backup(
                        directory_descriptor, destination_name
                    )
                    _verify_artifact_name(
                        directory_descriptor,
                        directory_path,
                        backup_name,
                        original,
                        "archive destination backup",
                    )
                    _verify_artifact_name(
                        directory_descriptor,
                        directory_path,
                        destination_name,
                        original,
                        "archive destination",
                    )
                    os.fsync(directory_descriptor)

                _verify_artifact_name(
                    directory_descriptor,
                    directory_path,
                    temporary_name,
                    prepared,
                    "prepared archive pathname",
                )
                if original is None:
                    _verify_name_absent(
                        directory_descriptor,
                        destination_name,
                        "archive destination",
                    )
                else:
                    assert backup_name is not None
                    _verify_artifact_name(
                        directory_descriptor,
                        directory_path,
                        backup_name,
                        original,
                        "archive destination backup",
                    )
                    _verify_artifact_name(
                        directory_descriptor,
                        directory_path,
                        destination_name,
                        original,
                        "archive destination",
                    )
                verifier(root, files, "immediately before archive publication")
                _verify_public_parent(
                    directory_path,
                    directory_descriptor,
                    "immediately before archive publication",
                )
                _verify_artifact_name(
                    directory_descriptor,
                    directory_path,
                    temporary_name,
                    prepared,
                    "prepared archive pathname",
                )
                if original is None:
                    _verify_name_absent(
                        directory_descriptor,
                        destination_name,
                        "archive destination",
                    )
                else:
                    assert backup_name is not None
                    _verify_artifact_name(
                        directory_descriptor,
                        directory_path,
                        backup_name,
                        original,
                        "archive destination backup",
                    )
                    _verify_artifact_name(
                        directory_descriptor,
                        directory_path,
                        destination_name,
                        original,
                        "archive destination",
                    )
                try:
                    _replace_name(
                        directory_descriptor, temporary_name, destination_name
                    )
                except Exception:
                    if _name_names_inode(
                        directory_descriptor,
                        directory_path,
                        destination_name,
                        prepared,
                    ):
                        state = _PublicationState.COMMITTED
                    elif not _name_names_inode(
                        directory_descriptor,
                        directory_path,
                        temporary_name,
                        prepared,
                    ):
                        recovery = (
                            "; prior destination preserved as "
                            f"{directory_path / backup_name}"
                            if backup_name is not None
                            else ""
                        )
                        raise RuntimeError(
                            "rollback_indeterminate: archive replace boundary was "
                            f"externally changed{recovery}"
                        )
                    raise
                else:
                    state = _PublicationState.COMMITTED

                _verify_public_parent(
                    directory_path,
                    directory_descriptor,
                    "immediately after archive publication",
                )
                os.fsync(directory_descriptor)
                _verify_artifact_name(
                    directory_descriptor,
                    directory_path,
                    destination_name,
                    prepared,
                    "published archive destination",
                )
                verifier(root, files, "after archive publication")
                if backup_name is not None:
                    assert original is not None
                    _verify_artifact_name(
                        directory_descriptor,
                        directory_path,
                        backup_name,
                        original,
                        "archive destination backup",
                    )
                state = _PublicationState.PUBLISHED_SOURCE_VERIFIED
                verifier(root, files, "at archive publication success boundary")
                _verify_public_parent(
                    directory_path,
                    directory_descriptor,
                    "at archive publication success boundary",
                )
                state = _PublicationState.TRANSACTION_COMPLETE
            except Exception as exc:
                if not raw_owner.close_attempted:
                    try:
                        raw_owner.close_once()
                    except _PackageDescriptorCloseUncertain:
                        raise
                if raw_owner.close_uncertain:
                    raise RuntimeError(
                        _temporary_candidate_diagnostic(
                            "archive temporary stream descriptor close failed",
                            temporary_path,
                        )
                    ) from exc
                if isinstance(exc, _PackageDescriptorCloseUncertain):
                    detail = (
                        _temporary_candidate_diagnostic(
                            str(exc).split("; recovery material retained as", 1)[0],
                            temporary_path,
                        )
                        if exc.recovery_path == temporary_path
                        else str(exc)
                    )
                    prior = (
                        "; prior destination retained as "
                        f"{directory_path / backup_name}"
                        if backup_name is not None
                        else ""
                    )
                    raise RuntimeError(f"{detail}{prior}") from exc
                if state in {
                    _PublicationState.COMMITTED,
                    _PublicationState.PUBLISHED_SOURCE_VERIFIED,
                }:
                    try:
                        _rollback_publication(
                            directory_descriptor,
                            directory_path,
                            destination_name,
                            prepared,
                            backup_name,
                            original,
                        )
                    except Exception as rollback_exc:
                        raise rollback_exc from exc
                else:
                    cleanup_error: Exception | None = None
                    recovery_error: Exception | None = None
                    if backup_name is not None and original is not None:
                        recovery_error = RuntimeError(
                            "rollback_indeterminate: archive transaction failed after "
                            "creating recovery material; prior destination retained "
                            f"as {directory_path / backup_name}"
                        )
                    try:
                        ownership = prepared if prepared is not None else created
                        try:
                            cleanup_expected = _refresh_owned_public_artifact(
                                directory_descriptor,
                                directory_path,
                                temporary_name,
                                ownership,
                                "archive temporary",
                            )
                        except _PackageDescriptorCloseUncertain as close_exc:
                            raise RuntimeError(
                                _temporary_candidate_diagnostic(
                                    "archive temporary refresh descriptor close failed",
                                    temporary_path,
                                )
                            ) from close_exc
                        except Exception:
                            cleanup_expected = ownership
                        _claim_and_remove_public_artifact(
                            directory_descriptor,
                            directory_path,
                            temporary_name,
                            cleanup_expected,
                            "archive temporary",
                        )
                    except FileNotFoundError:
                        pass
                    except Exception as cleanup_exc:
                        cleanup_error = cleanup_exc
                    if cleanup_error is not None:
                        recovery = (
                            "; prior destination retained as "
                            f"{directory_path / backup_name}"
                            if backup_name is not None
                            else ""
                        )
                        raise RuntimeError(
                            "archive transaction cleanup failed: "
                            f"{cleanup_error}{recovery}"
                        ) from cleanup_error
                    if recovery_error is not None:
                        raise recovery_error from exc
                raise
        except Exception as exc:
            transaction_error = exc

        close_errors: list[_PackageDescriptorCloseUncertain] = []
        if not raw_owner.close_attempted:
            try:
                raw_owner.close_once()
            except _PackageDescriptorCloseUncertain as close_exc:
                close_errors.append(close_exc)
        if original_owner is not None:
            try:
                _close_package_descriptor(
                    original_owner, "original archive destination"
                )
            except _PackageDescriptorCloseUncertain as close_exc:
                close_errors.append(close_exc)

        if close_errors:
            details = "; ".join(
                _temporary_candidate_diagnostic(
                    str(error).split("; recovery material retained as", 1)[0],
                    temporary_path,
                )
                if error.recovery_path == temporary_path
                else str(error)
                for error in close_errors
            )
            if backup_name is not None:
                details += (
                    f"; prior destination retained as {directory_path / backup_name}"
                )
            elif original is not None:
                details += f"; prior destination remains at {directory_path / destination_name}"
            if transaction_error is not None:
                raise RuntimeError(
                    f"{transaction_error}; {details}"
                ) from transaction_error
            boundary = (
                "archive transaction complete"
                if state is _PublicationState.TRANSACTION_COMPLETE
                else "archive transaction finalization failed"
            )
            raise RuntimeError(f"{boundary}; {details}") from close_errors[0]

        if transaction_error is not None:
            raise transaction_error

        if backup_name is not None:
            assert original is not None
            _cleanup_backup_after_transaction(
                directory_descriptor, directory_path, backup_name, original
            )
        _audit_cleanup_arena_empty(
            directory_descriptor,
            directory_path,
            "at archive publication exit",
            create=False,
            expected_identity=entry_arena_identity,
        )
    finally:
        try:
            os.close(directory_descriptor)
        except OSError:
            if state is not _PublicationState.TRANSACTION_COMPLETE:
                raise


def _test_only_materialize_archive_transaction(
    root: Path, files: PackageSnapshot, destination: Path
) -> None:
    """Exercise transaction mechanics without granting publication authority."""
    _materialize_archive_transaction(
        root,
        files,
        destination,
        verify_snapshot=_verify_inspection_snapshot,
    )


def create_archive(
    root: Path, files: PublishablePackageSnapshot, destination: Path
) -> None:
    root = repository_git.strict_root(root)
    _verify_package_snapshot(root, files, "before archive destination preparation")
    _materialize_archive_transaction(
        root,
        files,
        destination,
        verify_snapshot=_verify_package_snapshot,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--archive", type=Path)
    args = parser.parse_args()

    root = repository_git.strict_root(args.root)
    paths = package_paths(root / "build.zig.zon", root=root)
    repository: repository_git.RepositoryGit | None
    if args.archive:
        files = publishable_package_files(root, paths)
        repository = files.repository
        create_archive(root, files, args.archive)
    else:
        repository = repository_git.open_repository(root)
        files = package_files(root, paths, repository=repository)

    message = f"checked {len(paths)} package paths and {len(files)} files"
    if repository is None:
        message += " (archive mode; repository omissions not checked)"
    print(message)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
