#!/usr/bin/env python3
# Copyright (C) 2026 Zynum contributors
# SPDX-License-Identifier: LGPL-3.0-or-later

"""Fail-closed, dependency-free parity checks for Darwin ABI artifacts.

This module intentionally does not use ``otool``, ``nm``, ``ar``, ``codesign``,
or the ABI baseline observer.  It parses the bytes that are ultimately being
attested and assigns every byte to a proof atom.  The four accepted sources of
fresh-build volatility are:

* a dynamic library's ``LC_UUID`` payload;
* its unique real ``N_OSO`` cache path and mtime;
* the resulting linker-generated ad-hoc CodeDirectory; and
* a static object's unique DWARF-v4 global-cache ``b/<32hex>`` directory.

All other bytes and structural facts are invariant.  Unsupported encodings and
layouts raise :class:`ParityError`; callers must not turn that into a pass.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import re
import struct
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping, Sequence


MAX_ARTIFACT_BYTES = 512 * 1024 * 1024
MAX_LOAD_COMMANDS = 4096
MAX_SECTIONS = 65536
MAX_SYMBOLS = 4_000_000
MAX_ARCHIVE_MEMBERS = 65536
MAX_ATOM_CLAIMS = 262_144

MH_MAGIC_64_BYTES = b"\xcf\xfa\xed\xfe"
CPU_TYPE_ARM64 = 0x0100000C
MH_OBJECT = 0x1
MH_DYLIB = 0x6

LC_SEGMENT_64 = 0x19
LC_SYMTAB = 0x2
LC_DYSYMTAB = 0xB
LC_LOAD_DYLIB = 0xC
LC_ID_DYLIB = 0xD
LC_LOAD_DYLINKER = 0xE
LC_PREBOUND_DYLIB = 0x10
LC_LOAD_WEAK_DYLIB = 0x80000018
LC_REEXPORT_DYLIB = 0x8000001F
LC_LAZY_LOAD_DYLIB = 0x20
LC_LOAD_UPWARD_DYLIB = 0x80000023
LC_UUID = 0x1B
LC_CODE_SIGNATURE = 0x1D
LC_DYLD_INFO_ONLY = 0x80000022
LC_FUNCTION_STARTS = 0x26
LC_SOURCE_VERSION = 0x2A
LC_DATA_IN_CODE = 0x29
LC_LINKER_OPTIMIZATION_HINT = 0x2E
LC_BUILD_VERSION = 0x32
LC_RPATH = 0x8000001C

S_ZEROFILL = 0x1
S_THREAD_LOCAL_ZEROFILL = 0x12
ZEROFILL_TYPES = frozenset((S_ZEROFILL, S_THREAD_LOCAL_ZEROFILL))

N_STAB = 0xE0
N_PEXT = 0x10
N_TYPE = 0x0E
N_EXT = 0x01
N_OSO = 0x66
N_TYPE_NAMES = {
    0x0: "undefined",
    0x2: "absolute",
    0xA: "indirect",
    0xC: "prebound-undefined",
    0xE: "section",
}

CSMAGIC_EMBEDDED_SIGNATURE = 0xFADE0CC0
CSMAGIC_CODEDIRECTORY = 0xFADE0C02
CSSLOT_CODEDIRECTORY = 0
CS_ADHOC = 0x2
CS_LINKER_SIGNED = 0x20000
CS_HASHTYPE_SHA256 = 2

AR_MAGIC = b"!<arch>\n"
AR_HEADER_SIZE = 60

CACHE_OBJECT_RE = re.compile(
    rb"^/(?:[^/\x00]+/)*local-cache/o/([0-9a-f]{32})/([^/\x00]+\.o)$"
)
GLOBAL_CACHE_DIR_RE = re.compile(rb"^/(?:[^/\x00]+/)*global-cache/b/([0-9a-f]{32})$")


class ParityError(ValueError):
    """An artifact cannot participate in a fail-closed parity decision."""

    def __init__(self, message: str, *, locator: str | None = None) -> None:
        self.locator = locator
        super().__init__(f"{locator}: {message}" if locator else message)


def sha256_bytes(data: bytes) -> str:
    """Return the lowercase SHA-256 digest of *data*."""

    return hashlib.sha256(data).hexdigest()


def canonical_json_bytes(value: Any) -> bytes:
    """Serialize a JSON value in the form used for canonical digests."""

    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")


@dataclasses.dataclass(frozen=True)
class ByteAtom:
    """A complete, non-overlapping classification of an artifact byte range."""

    locator: str
    role: str
    offset: int
    size: int
    payload: bytes
    depends_on: tuple[str, ...] = ()
    note: str | None = None
    canonical_payload: bytes | None = None

    def __post_init__(self) -> None:
        if self.role not in {
            "invariant",
            "allowed_payload",
            "derived_field",
            "derived_padding",
        }:
            raise ParityError(f"unknown atom role {self.role!r}", locator=self.locator)
        if self.offset < 0 or self.size < 0 or len(self.payload) != self.size:
            raise ParityError("invalid atom range", locator=self.locator)
        if self.canonical_payload is not None and self.role != "derived_field":
            raise ParityError(
                "only derived fields may carry a canonical payload",
                locator=self.locator,
            )

    def json(self, *, include_payload: bool = False) -> dict[str, Any]:
        result: dict[str, Any] = {
            "locator": self.locator,
            "role": self.role,
            "offset": self.offset,
            "size": self.size,
            "sha256": sha256_bytes(self.payload),
        }
        if self.depends_on:
            result["depends_on"] = list(self.depends_on)
        if self.note is not None:
            result["note"] = self.note
        if self.canonical_payload is not None:
            result["canonical_payload_hex"] = self.canonical_payload.hex()
            result["canonical_payload_sha256"] = sha256_bytes(self.canonical_payload)
        if include_payload:
            result["payload_hex"] = self.payload.hex()
        return result


@dataclasses.dataclass(frozen=True)
class ProvenanceLocator:
    """A filesystem fact that a caller may bind without this module doing I/O."""

    kind: str
    source_locator: str
    path: str
    path_offset: int
    path_size: int
    expected_basename: str
    expected_mtime_seconds: int | None = None

    def json(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@dataclasses.dataclass(frozen=True)
class SymbolRecord:
    index: int
    name: str
    base_type: str
    section: int
    external: bool
    private_external: bool
    stab: int
    desc: int
    value: int
    type_code: int

    def comparison_key(self) -> tuple[Any, ...]:
        normalized_name = self.name
        if self.type_code == N_OSO:
            match = CACHE_OBJECT_RE.fullmatch(self.name.encode("utf-8"))
            if match is not None:
                normalized_name = f"<local-cache>/o/<digest>/{_text(match.group(2), 'symbol.N_OSO.basename')}"
        return (
            normalized_name,
            self.base_type,
            self.section,
            self.external,
            self.private_external,
            self.stab,
            self.desc,
            self.type_code,
        )

    def json(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@dataclasses.dataclass(frozen=True)
class Section:
    index: int
    segment: str
    name: str
    addr: int
    size: int
    offset: int
    align: int
    reloff: int
    nreloc: int
    flags: int
    reserved1: int
    reserved2: int
    reserved3: int

    @property
    def is_zerofill(self) -> bool:
        return (self.flags & 0xFF) in ZEROFILL_TYPES

    def json(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@dataclasses.dataclass(frozen=True)
class Segment:
    index: int
    name: str
    vmaddr: int
    vmsize: int
    fileoff: int
    filesize: int
    maxprot: int
    initprot: int
    flags: int
    sections: tuple[Section, ...]

    def json(self) -> dict[str, Any]:
        return {
            **{
                field.name: getattr(self, field.name)
                for field in dataclasses.fields(self)
                if field.name != "sections"
            },
            "sections": [section.json() for section in self.sections],
        }


@dataclasses.dataclass(frozen=True)
class CodeDirectory:
    offset: int
    size: int
    version: int
    flags: int
    identifier: str
    code_limit: int
    code_slots: int
    page_size: int
    hash_type: int
    hash_size: int
    cdhash: str

    def json(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@dataclasses.dataclass(frozen=True)
class DwarfDirectory:
    unit_index: int
    directory_index: int
    path: str
    offset: int
    size: int

    def json(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@dataclasses.dataclass(frozen=True)
class DwarfFile:
    unit_index: int
    file_index: int
    name: str
    directory_index: int
    mtime: int
    length: int

    def json(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@dataclasses.dataclass(frozen=True)
class MachOArtifact:
    raw: bytes
    label: str
    filetype: int
    flags: int
    segments: tuple[Segment, ...]
    symbols: tuple[SymbolRecord, ...]
    structured_axes: Mapping[str, Any]
    atoms: tuple[ByteAtom, ...]
    provenance: tuple[ProvenanceLocator, ...]
    code_directory: CodeDirectory | None
    dwarf_directories: tuple[DwarfDirectory, ...] = ()
    dwarf_files: tuple[DwarfFile, ...] = ()

    @property
    def raw_sha256(self) -> str:
        return sha256_bytes(self.raw)

    @property
    def canonical_digest(self) -> str:
        return canonical_digest(self.atoms)


@dataclasses.dataclass(frozen=True)
class RanlibEntry:
    index: int
    symbol: str
    string_offset: int
    member_offset: int

    def json(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@dataclasses.dataclass(frozen=True)
class ArchiveMember:
    index: int
    name: str
    header_offset: int
    data_offset: int
    stored_size: int
    object: MachOArtifact | None

    def json(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "name": self.name,
            "header_offset": self.header_offset,
            "data_offset": self.data_offset,
            "stored_size": self.stored_size,
            "object_sha256": self.object.raw_sha256
            if self.object is not None
            else None,
        }


@dataclasses.dataclass(frozen=True)
class ArchiveArtifact:
    raw: bytes
    label: str
    members: tuple[ArchiveMember, ...]
    ranlib: tuple[RanlibEntry, ...]
    structured_axes: Mapping[str, Any]
    symbols: tuple[SymbolRecord, ...]
    atoms: tuple[ByteAtom, ...]
    provenance: tuple[ProvenanceLocator, ...]

    @property
    def raw_sha256(self) -> str:
        return sha256_bytes(self.raw)

    @property
    def canonical_digest(self) -> str:
        return canonical_digest(self.atoms)


@dataclasses.dataclass(frozen=True)
class Comparison:
    """JSON-ready evidence for a pairwise artifact parity decision."""

    artifact_kind: str
    left_label: str
    right_label: str
    left_raw_sha256: str
    right_raw_sha256: str
    left_size: int
    right_size: int
    structured_axes: Mapping[str, Any]
    symbol_axes: Mapping[str, Any]
    allowed_edit_atoms: tuple[Mapping[str, Any], ...]
    derived_fields: tuple[Mapping[str, Any], ...]
    left_canonical_digest: str
    right_canonical_digest: str
    uncovered_left_bytes: tuple[Mapping[str, int], ...]
    uncovered_right_bytes: tuple[Mapping[str, int], ...]
    unpaired_atoms: tuple[str, ...]
    invalid_derived_fields: tuple[str, ...]
    provenance_locators: Mapping[str, Any]
    verdict: str
    failures: tuple[str, ...]

    def json(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    def json_bytes(self) -> bytes:
        return canonical_json_bytes(self.json()) + b"\n"


_AtomMetadata = tuple[str, str, tuple[str, ...], str | None, bytes | None]


@dataclasses.dataclass(slots=True)
class _AtomInterval:
    """One node in the builder's AVL tree of assigned half-open ranges."""

    start: int
    end: int
    value: _AtomMetadata
    left: _AtomInterval | None = None
    right: _AtomInterval | None = None
    height: int = 1


def _interval_height(node: _AtomInterval | None) -> int:
    return 0 if node is None else node.height


def _update_interval_height(node: _AtomInterval) -> None:
    node.height = 1 + max(_interval_height(node.left), _interval_height(node.right))


def _rotate_interval_left(node: _AtomInterval) -> _AtomInterval:
    replacement = node.right
    assert replacement is not None
    node.right = replacement.left
    replacement.left = node
    _update_interval_height(node)
    _update_interval_height(replacement)
    return replacement


def _rotate_interval_right(node: _AtomInterval) -> _AtomInterval:
    replacement = node.left
    assert replacement is not None
    node.left = replacement.right
    replacement.right = node
    _update_interval_height(node)
    _update_interval_height(replacement)
    return replacement


def _balance_interval(node: _AtomInterval) -> _AtomInterval:
    _update_interval_height(node)
    balance = _interval_height(node.left) - _interval_height(node.right)
    if balance > 1:
        assert node.left is not None
        if _interval_height(node.left.left) < _interval_height(node.left.right):
            node.left = _rotate_interval_left(node.left)
        return _rotate_interval_right(node)
    if balance < -1:
        assert node.right is not None
        if _interval_height(node.right.right) < _interval_height(node.right.left):
            node.right = _rotate_interval_right(node.right)
        return _rotate_interval_left(node)
    return node


def _insert_interval(
    root: _AtomInterval | None, interval: _AtomInterval
) -> _AtomInterval:
    if root is None:
        return interval
    if interval.start < root.start:
        root.left = _insert_interval(root.left, interval)
    elif interval.start > root.start:
        root.right = _insert_interval(root.right, interval)
    else:
        raise AssertionError(f"duplicate interval start {interval.start}")
    return _balance_interval(root)


def _remove_interval(root: _AtomInterval | None, start: int) -> _AtomInterval | None:
    if root is None:
        raise AssertionError(f"missing interval start {start}")
    if start < root.start:
        root.left = _remove_interval(root.left, start)
    elif start > root.start:
        root.right = _remove_interval(root.right, start)
    elif root.left is None:
        return root.right
    elif root.right is None:
        return root.left
    else:
        successor = root.right
        while successor.left is not None:
            successor = successor.left
        root.start = successor.start
        root.end = successor.end
        root.value = successor.value
        root.right = _remove_interval(root.right, successor.start)
    return _balance_interval(root)


def _lower_bound_interval(
    root: _AtomInterval | None, start: int
) -> _AtomInterval | None:
    result = None
    while root is not None:
        if root.start >= start:
            result = root
            root = root.left
        else:
            root = root.right
    return result


def _predecessor_interval(
    root: _AtomInterval | None, start: int
) -> _AtomInterval | None:
    result = None
    while root is not None:
        if root.start < start:
            result = root
            root = root.right
        else:
            root = root.left
    return result


def _successor_interval(root: _AtomInterval | None, start: int) -> _AtomInterval | None:
    result = None
    while root is not None:
        if root.start > start:
            result = root
            root = root.left
        else:
            root = root.right
    return result


def _ordered_intervals(root: _AtomInterval | None) -> Iterable[_AtomInterval]:
    stack: list[_AtomInterval] = []
    while root is not None or stack:
        while root is not None:
            stack.append(root)
            root = root.left
        root = stack.pop()
        yield root
        root = root.right


class _AtomBuilder:
    """Assign byte ranges using sparse, deterministically balanced intervals."""

    def __init__(self, data: bytes, prefix: str) -> None:
        self.data = data
        self.prefix = prefix
        self._root: _AtomInterval | None = None
        self._claim_count = 0

    def _overlaps(self, start: int, end: int) -> list[tuple[int, int, _AtomMetadata]]:
        first = _lower_bound_interval(self._root, start)
        predecessor = _predecessor_interval(self._root, start)
        if predecessor is not None and predecessor.end > start:
            first = predecessor

        result: list[tuple[int, int, _AtomMetadata]] = []
        current = first
        while current is not None and current.start < end:
            if current.end > start:
                result.append((current.start, current.end, current.value))
            current = _successor_interval(self._root, current.start)
        return result

    def _put(self, start: int, end: int, value: _AtomMetadata) -> None:
        predecessor = _predecessor_interval(self._root, start)
        if (
            predecessor is not None
            and predecessor.end == start
            and predecessor.value == value
        ):
            start = predecessor.start
            self._root = _remove_interval(self._root, predecessor.start)

        successor = _lower_bound_interval(self._root, start)
        if (
            successor is not None
            and successor.start == end
            and successor.value == value
        ):
            end = successor.end
            self._root = _remove_interval(self._root, successor.start)

        self._root = _insert_interval(
            self._root, _AtomInterval(start=start, end=end, value=value)
        )

    def add(
        self,
        start: int,
        size: int,
        locator: str,
        role: str = "invariant",
        *,
        depends_on: Sequence[str] = (),
        note: str | None = None,
        canonical_payload: bytes | None = None,
        replace: bool = False,
    ) -> None:
        _require_range(len(self.data), start, size, locator)
        if size == 0:
            return
        full = f"{self.prefix}.{locator}" if self.prefix else locator
        value = (full, role, tuple(depends_on), note, canonical_payload)
        end = start + size
        overlaps = self._overlaps(start, end)
        if not replace and overlaps:
            prior_start, _, prior_value = overlaps[0]
            conflict = max(start, prior_start)
            raise ParityError(
                f"byte {conflict} is already assigned to {prior_value[0]}",
                locator=full,
            )
        if replace:
            for prior_start, _, prior_value in overlaps:
                if prior_value[1] != "invariant":
                    conflict = max(start, prior_start)
                    raise ParityError(
                        f"cannot replace {prior_value[1]} atom at byte {conflict}",
                        locator=full,
                    )
        if self._claim_count >= MAX_ATOM_CLAIMS:
            raise ParityError(
                f"non-empty atom claim limit {MAX_ATOM_CLAIMS} exceeded",
                locator=full,
            )

        for prior_start, prior_end, prior_value in overlaps:
            self._root = _remove_interval(self._root, prior_start)
            if prior_start < start:
                self._root = _insert_interval(
                    self._root,
                    _AtomInterval(prior_start, start, prior_value),
                )
            if prior_end > end:
                self._root = _insert_interval(
                    self._root,
                    _AtomInterval(end, prior_end, prior_value),
                )
        self._put(start, end, value)
        self._claim_count += 1

    def finish(self) -> tuple[ByteAtom, ...]:
        intervals = tuple(_ordered_intervals(self._root))
        uncovered: list[dict[str, int]] = []
        cursor = 0
        for interval in intervals:
            if cursor < interval.start:
                uncovered.append({"offset": cursor, "size": interval.start - cursor})
            cursor = interval.end
        if cursor < len(self.data):
            uncovered.append({"offset": cursor, "size": len(self.data) - cursor})
        if uncovered:
            raise ParityError(
                f"uncovered byte ranges: {tuple(uncovered)}", locator=self.prefix
            )
        atoms: list[ByteAtom] = []
        occurrences: defaultdict[str, int] = defaultdict(int)
        for interval in intervals:
            start = interval.start
            end = interval.end
            value = interval.value
            base_locator, role, dependencies, note, canonical_payload = value
            occurrence = occurrences[base_locator]
            occurrences[base_locator] += 1
            locator = (
                base_locator if occurrence == 0 else f"{base_locator}#{occurrence}"
            )
            atoms.append(
                ByteAtom(
                    locator,
                    role,
                    start,
                    end - start,
                    self.data[start:end],
                    dependencies,
                    note,
                    canonical_payload,
                )
            )
        locators = [atom.locator for atom in atoms]
        if len(locators) != len(set(locators)):
            raise ParityError("atom locators are not unique", locator=self.prefix)
        return tuple(atoms)


def _uncovered_ranges(cover: Sequence[object | None]) -> tuple[dict[str, int], ...]:
    result: list[dict[str, int]] = []
    index = 0
    while index < len(cover):
        if cover[index] is not None:
            index += 1
            continue
        end = index + 1
        while end < len(cover) and cover[end] is None:
            end += 1
        result.append({"offset": index, "size": end - index})
        index = end
    return tuple(result)


def _require_range(total: int, offset: int, size: int, locator: str) -> None:
    if offset < 0 or size < 0 or offset > total or size > total - offset:
        raise ParityError(
            f"range [{offset}, {offset + size}) exceeds {total} bytes", locator=locator
        )


def _read_source(
    source: bytes | bytearray | memoryview | os.PathLike[str] | str, label: str | None
) -> tuple[bytes, str]:
    if isinstance(source, (bytes, bytearray, memoryview)):
        data = bytes(source)
        effective_label = label or "<bytes>"
    else:
        path = Path(source)
        try:
            size = path.stat().st_size
        except OSError as exc:
            raise ParityError(f"cannot stat artifact: {exc}") from exc
        if size > MAX_ARTIFACT_BYTES:
            raise ParityError(f"artifact exceeds {MAX_ARTIFACT_BYTES} bytes")
        try:
            data = path.read_bytes()
        except OSError as exc:
            raise ParityError(f"cannot read artifact: {exc}") from exc
        effective_label = label or str(path)
    if len(data) > MAX_ARTIFACT_BYTES:
        raise ParityError(f"artifact exceeds {MAX_ARTIFACT_BYTES} bytes")
    return data, effective_label


def _cstring(data: bytes, start: int, limit: int, locator: str) -> tuple[bytes, int]:
    _require_range(len(data), start, 0, locator)
    if limit < start or limit > len(data):
        raise ParityError("invalid string limit", locator=locator)
    end = data.find(b"\0", start, limit)
    if end < 0:
        raise ParityError("unterminated string", locator=locator)
    return data[start:end], end + 1


def _text(value: bytes, locator: str) -> str:
    try:
        return value.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ParityError("string is not UTF-8", locator=locator) from exc


def _fixed_name(value: bytes, locator: str) -> str:
    nul = value.find(b"\0")
    if nul >= 0:
        if any(value[nul:]):
            raise ParityError(
                "nonzero bytes after fixed-name terminator", locator=locator
            )
        value = value[:nul]
    return _text(value, locator)


def _uleb(data: bytes, offset: int, limit: int, locator: str) -> tuple[int, int]:
    value = 0
    shift = 0
    start = offset
    while offset < limit and offset - start < 10:
        byte = data[offset]
        offset += 1
        value |= (byte & 0x7F) << shift
        if not byte & 0x80:
            if offset - start > 1 and byte == 0:
                raise ParityError("non-minimal ULEB128", locator=locator)
            return value, offset
        shift += 7
    raise ParityError("truncated or oversized ULEB128", locator=locator)


def _version(value: int) -> str:
    return f"{value >> 16}.{(value >> 8) & 0xFF}.{value & 0xFF}"


def _align_up(value: int, alignment: int) -> int:
    if alignment <= 0 or alignment & (alignment - 1):
        raise ParityError(f"invalid power-of-two alignment {alignment}")
    return (value + alignment - 1) & -alignment


def _check_disjoint(
    ranges: Iterable[tuple[int, int, str]], *, total: int, context: str
) -> None:
    material = sorted((start, size, name) for start, size, name in ranges if size)
    previous_end = 0
    previous_name = ""
    for start, size, name in material:
        _require_range(total, start, size, name)
        if start < previous_end:
            raise ParityError(f"overlaps {previous_name}", locator=name)
        previous_end = start + size
        previous_name = name


def parse_code_directory(
    data: bytes, offset: int, size: int, *, builder: _AtomBuilder | None = None
) -> CodeDirectory:
    """Parse and cryptographically verify one linker-generated ad-hoc signature."""

    locator = "macho.code_signature"
    _require_range(len(data), offset, size, locator)
    if size < 20:
        raise ParityError("signature is too small", locator=locator)
    magic, super_length, count = struct.unpack_from(">III", data, offset)
    if magic != CSMAGIC_EMBEDDED_SIGNATURE or super_length != size or count != 1:
        raise ParityError("requires a single-slot embedded signature", locator=locator)
    slot_type, blob_offset = struct.unpack_from(">II", data, offset + 12)
    if slot_type != CSSLOT_CODEDIRECTORY or blob_offset != 20:
        raise ParityError("requires exactly one primary CodeDirectory", locator=locator)
    cd = offset + blob_offset
    if size < blob_offset + 88:
        raise ParityError("CodeDirectory header is truncated", locator=locator)
    fields = struct.unpack_from(">IIIIIIIII4BIIIIQQQQ", data, cd)
    (
        cd_magic,
        cd_length,
        version,
        flags,
        hash_offset,
        ident_offset,
        special_slots,
        code_slots,
        code_limit_32,
        hash_size,
        hash_type,
        platform,
        page_exp,
        spare2,
        scatter_offset,
        team_offset,
        spare3,
        code_limit_64,
        exec_seg_base,
        exec_seg_limit,
        exec_seg_flags,
    ) = fields
    if cd_magic != CSMAGIC_CODEDIRECTORY or cd_length != size - blob_offset:
        raise ParityError("invalid CodeDirectory length or magic", locator=locator)
    if version != 0x20400:
        raise ParityError(
            f"unsupported CodeDirectory version 0x{version:x}", locator=locator
        )
    if flags != CS_ADHOC | CS_LINKER_SIGNED:
        raise ParityError(
            f"signature flags 0x{flags:x} are not ad-hoc linker-signed", locator=locator
        )
    if special_slots != 0 or scatter_offset != 0 or team_offset != 0:
        raise ParityError(
            "special, scatter, team, entitlement, or requirement slots are forbidden",
            locator=locator,
        )
    if spare2 != 0 or spare3 != 0 or platform != 0:
        raise ParityError("nonzero reserved CodeDirectory field", locator=locator)
    if hash_type != CS_HASHTYPE_SHA256 or hash_size != hashlib.sha256().digest_size:
        raise ParityError(
            "only full SHA-256 CodeDirectories are supported", locator=locator
        )
    if page_exp > 30:
        raise ParityError("invalid CodeDirectory page exponent", locator=locator)
    page_size = 1 << page_exp
    if code_limit_32 == 0xFFFFFFFF:
        code_limit = code_limit_64
        if code_limit <= 0xFFFFFFFF:
            raise ParityError(
                "non-canonical 64-bit CodeDirectory code limit", locator=locator
            )
    else:
        code_limit = code_limit_32
        if code_limit_64 != 0:
            raise ParityError(
                "unexpected 64-bit CodeDirectory code limit", locator=locator
            )
    if code_limit != offset:
        raise ParityError(
            "CodeDirectory code limit does not equal signature offset", locator=locator
        )
    expected_slots = (code_limit + page_size - 1) // page_size
    if code_slots != expected_slots:
        raise ParityError(
            "CodeDirectory code-slot count is not derived from code limit",
            locator=locator,
        )
    if ident_offset < 88 or ident_offset >= cd_length:
        raise ParityError("invalid CodeDirectory identifier offset", locator=locator)
    ident_bytes, ident_end = _cstring(
        data, cd + ident_offset, cd + cd_length, f"{locator}.identifier"
    )
    identifier = _text(ident_bytes, f"{locator}.identifier")
    if not identifier:
        raise ParityError("empty CodeDirectory identifier", locator=locator)
    expected_hash_offset = ident_end - cd
    if hash_offset != expected_hash_offset:
        raise ParityError(
            "CodeDirectory hash offset is not immediately after identifier",
            locator=locator,
        )
    if hash_offset + code_slots * hash_size != cd_length:
        raise ParityError(
            "CodeDirectory hash array does not exactly fill the blob", locator=locator
        )
    for index in range(code_slots):
        page_start = index * page_size
        page_end = min(page_start + page_size, code_limit)
        expected = hashlib.sha256(data[page_start:page_end]).digest()
        actual_start = cd + hash_offset + index * hash_size
        actual = data[actual_start : actual_start + hash_size]
        if actual != expected:
            raise ParityError(
                f"page hash {index} does not match signed bytes", locator=locator
            )
    if builder is not None:
        dynamic_dependencies = (
            "macho.load[LC_UUID].uuid",
            "macho.symtab.n_oso.mtime",
            "macho.symtab.n_oso.path",
        )
        # Preserve every non-derived CodeDirectory byte as an invariant.  Only
        # the enclosing lengths, code limit/slot count, and verified page hashes
        # are consequences of the three accepted dynamic payloads.
        builder.add(offset, size, "code_signature", replace=True)
        builder.add(
            offset + 4,
            4,
            "code_signature.superblob_length",
            "derived_field",
            depends_on=dynamic_dependencies,
            replace=True,
        )
        builder.add(
            cd + 4,
            4,
            "code_signature.directory_length",
            "derived_field",
            depends_on=dynamic_dependencies,
            replace=True,
        )
        builder.add(
            cd + 28,
            8,
            "code_signature.code_slots_and_limit",
            "derived_field",
            depends_on=dynamic_dependencies,
            replace=True,
        )
        builder.add(
            cd + hash_offset,
            code_slots * hash_size,
            "code_signature.page_hashes",
            "derived_field",
            depends_on=dynamic_dependencies,
            replace=True,
        )
    return CodeDirectory(
        offset=offset,
        size=size,
        version=version,
        flags=flags,
        identifier=identifier,
        code_limit=code_limit,
        code_slots=code_slots,
        page_size=page_size,
        hash_type=hash_type,
        hash_size=hash_size,
        cdhash=sha256_bytes(data[cd : cd + cd_length]),
    )


def parse_dwarf_v4_debug_line(
    section_data: bytes,
    *,
    file_offset: int = 0,
    builder: _AtomBuilder | None = None,
    locator_prefix: str = "dwarf.debug_line",
    allowed_dependency_locator: str = "dwarf.global_cache_directory",
) -> tuple[tuple[DwarfDirectory, ...], tuple[DwarfFile, ...], ProvenanceLocator]:
    """Parse DWARF32 v4 line tables and locate the unique cache ``builtin.zig``."""

    directories: list[DwarfDirectory] = []
    files: list[DwarfFile] = []
    offset = 0
    unit_index = 0
    while offset < len(section_data):
        unit_start = offset
        if len(section_data) - offset < 10:
            raise ParityError(
                "truncated line-table unit header", locator=locator_prefix
            )
        total_length = struct.unpack_from("<I", section_data, offset)[0]
        if total_length == 0xFFFFFFFF:
            raise ParityError(
                "DWARF64 line tables are unsupported", locator=locator_prefix
            )
        unit_end = offset + 4 + total_length
        _require_range(len(section_data), offset, 4 + total_length, locator_prefix)
        version = struct.unpack_from("<H", section_data, offset + 4)[0]
        if version != 4:
            raise ParityError(
                f"unsupported DWARF line-table version {version}",
                locator=locator_prefix,
            )
        prologue_length = struct.unpack_from("<I", section_data, offset + 6)[0]
        prologue_start = offset + 10
        prologue_end = prologue_start + prologue_length
        if prologue_end > unit_end or prologue_length < 6:
            raise ParityError(
                "invalid DWARF line-table prologue length", locator=locator_prefix
            )
        cursor = prologue_start
        min_inst, max_ops, default_stmt, _line_base, line_range, opcode_base = (
            struct.unpack_from("<BBBbBB", section_data, cursor)
        )
        cursor += 6
        if (
            min_inst == 0
            or max_ops == 0
            or default_stmt not in (0, 1)
            or line_range == 0
            or opcode_base == 0
        ):
            raise ParityError(
                "invalid DWARF v4 line-table scalar prologue", locator=locator_prefix
            )
        if cursor + opcode_base - 1 > prologue_end:
            raise ParityError(
                "truncated standard opcode lengths", locator=locator_prefix
            )
        cursor += opcode_base - 1
        directory_index = 1
        while True:
            value, next_cursor = _cstring(
                section_data, cursor, prologue_end, f"{locator_prefix}.directory"
            )
            if not value:
                cursor = next_cursor
                break
            directories.append(
                DwarfDirectory(
                    unit_index,
                    directory_index,
                    _text(value, f"{locator_prefix}.directory[{directory_index}]"),
                    file_offset + cursor,
                    len(value),
                )
            )
            directory_index += 1
            cursor = next_cursor
        file_index = 1
        while True:
            name, next_cursor = _cstring(
                section_data, cursor, prologue_end, f"{locator_prefix}.file"
            )
            cursor = next_cursor
            if not name:
                break
            dir_index, cursor = _uleb(
                section_data, cursor, prologue_end, f"{locator_prefix}.file.dir"
            )
            mtime, cursor = _uleb(
                section_data, cursor, prologue_end, f"{locator_prefix}.file.mtime"
            )
            length, cursor = _uleb(
                section_data, cursor, prologue_end, f"{locator_prefix}.file.length"
            )
            if dir_index >= directory_index:
                raise ParityError(
                    "file references an absent directory", locator=locator_prefix
                )
            files.append(
                DwarfFile(
                    unit_index,
                    file_index,
                    _text(name, f"{locator_prefix}.file[{file_index}]"),
                    dir_index,
                    mtime,
                    length,
                )
            )
            file_index += 1
        if cursor != prologue_end:
            raise ParityError(
                "DWARF file table does not exactly fill the prologue",
                locator=locator_prefix,
            )
        if builder is not None:
            base = file_offset + unit_start
            builder.add(
                base,
                4,
                f"{locator_prefix}.unit[{unit_index}].total_length",
                "derived_field",
                depends_on=(allowed_dependency_locator,),
                replace=True,
            )
            builder.add(
                base + 6,
                4,
                f"{locator_prefix}.unit[{unit_index}].prologue_length",
                "derived_field",
                depends_on=(allowed_dependency_locator,),
                replace=True,
            )
        offset = unit_end
        unit_index += 1
    eligible = [
        item
        for item in directories
        if GLOBAL_CACHE_DIR_RE.fullmatch(item.path.encode("utf-8"))
    ]
    if len(eligible) != 1:
        raise ParityError(
            f"expected one global-cache b/<32hex> directory, found {len(eligible)}",
            locator=locator_prefix,
        )
    cache_dir = eligible[0]
    builtin_files = [
        item
        for item in files
        if item.unit_index == cache_dir.unit_index
        and item.directory_index == cache_dir.directory_index
        and item.name == "builtin.zig"
    ]
    if len(builtin_files) != 1:
        raise ParityError(
            "global-cache directory must be referenced by exactly one builtin.zig",
            locator=locator_prefix,
        )
    if builder is not None:
        builder.add(
            cache_dir.offset,
            cache_dir.size,
            "dwarf.global_cache_directory",
            "allowed_payload",
            replace=True,
            note="unique DWARF-v4 global-cache b/<32hex> directory",
        )
    builtin_path = f"{cache_dir.path}/builtin.zig"
    provenance = ProvenanceLocator(
        kind="static_builtin",
        source_locator=f"{locator_prefix}.unit[{cache_dir.unit_index}].directory[{cache_dir.directory_index}]",
        path=builtin_path,
        path_offset=cache_dir.offset,
        path_size=cache_dir.size,
        expected_basename="builtin.zig",
    )
    return tuple(directories), tuple(files), provenance


def _parse_load_string(
    data: bytes, command_offset: int, command_size: int, field_offset: int, locator: str
) -> str:
    if field_offset < 8 or field_offset >= command_size:
        raise ParityError("invalid load-command string offset", locator=locator)
    value, end = _cstring(
        data, command_offset + field_offset, command_offset + command_size, locator
    )
    if any(data[end : command_offset + command_size]):
        raise ParityError("nonzero load-command string padding", locator=locator)
    return _text(value, locator)


def _parse_macho_bytes(
    data: bytes, label: str, expected_filetype: int, prefix: str
) -> MachOArtifact:
    if len(data) < 32:
        raise ParityError("Mach-O header is truncated", locator=prefix)
    if data[:4] != MH_MAGIC_64_BYTES:
        if data[:4] in {
            b"\xfe\xed\xfa\xcf",
            b"\xce\xfa\xed\xfe",
            b"\xfe\xed\xfa\xce",
            b"\xca\xfe\xba\xbe",
            b"\xbe\xba\xfe\xca",
            b"\xca\xfe\xba\xbf",
            b"\xbf\xba\xfe\xca",
        }:
            raise ParityError(
                "fat, 32-bit, or big-endian Mach-O is unsupported", locator=prefix
            )
        raise ParityError("not a little-endian 64-bit Mach-O", locator=prefix)
    magic, cpu_type, cpu_subtype, filetype, ncmds, sizeofcmds, flags, reserved = (
        struct.unpack_from("<IiiIIIII", data)
    )
    if (
        magic != 0xFEEDFACF
        or cpu_type != CPU_TYPE_ARM64
        or filetype != expected_filetype
    ):
        raise ParityError(
            "requires a thin arm64 Mach-O of the requested file type", locator=prefix
        )
    if reserved != 0 or ncmds > MAX_LOAD_COMMANDS or sizeofcmds > len(data) - 32:
        raise ParityError("invalid Mach-O header fields", locator=prefix)
    load_end = 32 + sizeofcmds
    builder = _AtomBuilder(data, prefix)
    builder.add(0, 32, "header")
    if load_end < len(data):
        builder.add(load_end, len(data) - load_end, "content")

    segments: list[Segment] = []
    sections: list[Section] = []
    load_axes: list[dict[str, Any]] = []
    symtab: tuple[int, int, int, int] | None = None
    dysymtab: tuple[int, ...] | None = None
    signature_range: tuple[int, int] | None = None
    build_versions: list[dict[str, Any]] = []
    install_names: list[dict[str, Any]] = []
    dependencies: list[dict[str, Any]] = []
    rpaths: list[str] = []
    linkedit_ranges: list[tuple[int, int, str]] = []
    command_offset = 32
    static_dependency = f"{prefix}.dwarf.global_cache_directory"
    seen_singletons: Counter[int] = Counter()
    command_names = {
        LC_SEGMENT_64: "LC_SEGMENT_64",
        LC_SYMTAB: "LC_SYMTAB",
        LC_DYSYMTAB: "LC_DYSYMTAB",
        LC_LOAD_DYLIB: "LC_LOAD_DYLIB",
        LC_ID_DYLIB: "LC_ID_DYLIB",
        LC_LOAD_DYLINKER: "LC_LOAD_DYLINKER",
        LC_PREBOUND_DYLIB: "LC_PREBOUND_DYLIB",
        LC_LOAD_WEAK_DYLIB: "LC_LOAD_WEAK_DYLIB",
        LC_REEXPORT_DYLIB: "LC_REEXPORT_DYLIB",
        LC_LAZY_LOAD_DYLIB: "LC_LAZY_LOAD_DYLIB",
        LC_LOAD_UPWARD_DYLIB: "LC_LOAD_UPWARD_DYLIB",
        LC_UUID: "LC_UUID",
        LC_CODE_SIGNATURE: "LC_CODE_SIGNATURE",
        LC_DYLD_INFO_ONLY: "LC_DYLD_INFO_ONLY",
        LC_FUNCTION_STARTS: "LC_FUNCTION_STARTS",
        LC_SOURCE_VERSION: "LC_SOURCE_VERSION",
        LC_DATA_IN_CODE: "LC_DATA_IN_CODE",
        LC_LINKER_OPTIMIZATION_HINT: "LC_LINKER_OPTIMIZATION_HINT",
        LC_BUILD_VERSION: "LC_BUILD_VERSION",
        LC_RPATH: "LC_RPATH",
    }
    for command_index in range(ncmds):
        if command_offset + 8 > load_end:
            raise ParityError("truncated load command", locator=prefix)
        command, command_size = struct.unpack_from("<II", data, command_offset)
        name = command_names.get(command)
        if name is None:
            raise ParityError(
                f"unsupported load command 0x{command:x}",
                locator=f"{prefix}.load[{command_index}]",
            )
        if (
            command_size < 8
            or command_size % 8
            or command_offset + command_size > load_end
        ):
            raise ParityError(
                "invalid load-command size", locator=f"{prefix}.load[{command_index}]"
            )
        locator = f"load[{command_index}:{name}]"
        builder.add(command_offset, command_size, locator)
        axis: dict[str, Any] = {
            "index": command_index,
            "command": name,
            "size": command_size,
        }
        seen_singletons[command] += 1
        if command == LC_SEGMENT_64:
            if command_size < 72:
                raise ParityError("truncated segment command", locator=locator)
            values = struct.unpack_from("<II16sQQQQiiII", data, command_offset)
            (
                _,
                _,
                segname_raw,
                vmaddr,
                vmsize,
                fileoff,
                filesize,
                maxprot,
                initprot,
                nsects,
                segflags,
            ) = values
            if nsects > MAX_SECTIONS or command_size != 72 + 80 * nsects:
                raise ParityError(
                    "segment section count does not match command size", locator=locator
                )
            segname = _fixed_name(segname_raw, f"{locator}.segname")
            _require_range(len(data), fileoff, filesize, f"{locator}.file_range")
            segment_sections: list[Section] = []
            for local_index in range(nsects):
                section_offset = command_offset + 72 + local_index * 80
                section_values = struct.unpack_from(
                    "<16s16sQQIIIIIIII", data, section_offset
                )
                (
                    sectname_raw,
                    section_seg_raw,
                    addr,
                    section_size,
                    file_section_offset,
                    align,
                    reloff,
                    nreloc,
                    section_flags,
                    reserved1,
                    reserved2,
                    reserved3,
                ) = section_values
                sectname = _fixed_name(
                    sectname_raw, f"{locator}.section[{local_index}].name"
                )
                section_seg = _fixed_name(
                    section_seg_raw, f"{locator}.section[{local_index}].segment"
                )
                if section_seg != segname and not (
                    expected_filetype == MH_OBJECT and segname == ""
                ):
                    raise ParityError(
                        "section segment name disagrees with enclosing segment",
                        locator=locator,
                    )
                section = Section(
                    len(sections) + 1,
                    section_seg,
                    sectname,
                    addr,
                    section_size,
                    file_section_offset,
                    align,
                    reloff,
                    nreloc,
                    section_flags,
                    reserved1,
                    reserved2,
                    reserved3,
                )
                if align > 63:
                    raise ParityError(
                        "invalid section alignment exponent", locator=locator
                    )
                if not section.is_zerofill:
                    _require_range(
                        len(data),
                        file_section_offset,
                        section_size,
                        f"{locator}.{sectname}.data",
                    )
                    if expected_filetype != MH_OBJECT and not (
                        fileoff <= file_section_offset
                        and file_section_offset + section_size <= fileoff + filesize
                    ):
                        raise ParityError(
                            "section is outside its file-backed segment",
                            locator=locator,
                        )
                if nreloc:
                    _require_range(
                        len(data),
                        reloff,
                        nreloc * 8,
                        f"{locator}.{sectname}.relocations",
                    )
                sections.append(section)
                segment_sections.append(section)
            segment = Segment(
                len(segments),
                segname,
                vmaddr,
                vmsize,
                fileoff,
                filesize,
                maxprot,
                initprot,
                segflags,
                tuple(segment_sections),
            )
            segments.append(segment)
            axis.update(segment.json())
            if expected_filetype == MH_OBJECT:
                atom_dependencies = (static_dependency,)
                builder.add(
                    command_offset + 32,
                    8,
                    f"{locator}.derived_vmsize",
                    "derived_field",
                    depends_on=atom_dependencies,
                    replace=True,
                )
                builder.add(
                    command_offset + 48,
                    8,
                    f"{locator}.derived_filesize",
                    "derived_field",
                    depends_on=atom_dependencies,
                    replace=True,
                )
                segment_debug_lines = [
                    section
                    for section in segment_sections
                    if section.name == "__debug_line" and section.segment == "__DWARF"
                ]
                if len(segment_debug_lines) > 1:
                    raise ParityError(
                        "multiple __DWARF,__debug_line sections in one object segment",
                        locator=locator,
                    )
                debug_line_end = (
                    segment_debug_lines[0].addr + segment_debug_lines[0].size
                    if segment_debug_lines
                    else None
                )
                for local_index, section in enumerate(segment_sections):
                    section_base = command_offset + 72 + local_index * 80
                    if section.name == "__debug_line" and section.segment == "__DWARF":
                        builder.add(
                            section_base + 40,
                            8,
                            f"{locator}.section[{local_index}].debug_line_size",
                            "derived_field",
                            depends_on=atom_dependencies,
                            replace=True,
                        )
                    if (
                        debug_line_end is not None
                        and section is not segment_debug_lines[0]
                        and section.addr >= debug_line_end
                    ):
                        if not section.is_zerofill:
                            raise ParityError(
                                "only zero-fill sections may follow __debug_line in object address layout",
                                locator=locator,
                            )
                        builder.add(
                            section_base + 32,
                            8,
                            f"{locator}.section[{local_index}].addr",
                            "derived_field",
                            depends_on=atom_dependencies,
                            replace=True,
                        )
                    if section.reloff:
                        builder.add(
                            section_base + 56,
                            4,
                            f"{locator}.section[{local_index}].reloff",
                            "derived_field",
                            depends_on=atom_dependencies,
                            replace=True,
                        )
            elif segname == "__LINKEDIT":
                atom_dependencies = (
                    "macho.load[LC_UUID].uuid",
                    "macho.symtab.n_oso.mtime",
                    "macho.symtab.n_oso.path",
                )
                builder.add(
                    command_offset + 32,
                    8,
                    f"{locator}.derived_vmsize",
                    "derived_field",
                    depends_on=atom_dependencies,
                    replace=True,
                )
                builder.add(
                    command_offset + 48,
                    8,
                    f"{locator}.derived_filesize",
                    "derived_field",
                    depends_on=atom_dependencies,
                    replace=True,
                )
        elif command == LC_SYMTAB:
            if command_size != 24 or symtab is not None:
                raise ParityError("invalid or duplicate LC_SYMTAB", locator=locator)
            symtab = struct.unpack_from("<IIII", data, command_offset + 8)
            axis.update(dict(zip(("symoff", "nsyms", "stroff", "strsize"), symtab)))
            if expected_filetype == MH_OBJECT:
                builder.add(
                    command_offset + 8,
                    16,
                    f"{locator}.layout",
                    "derived_field",
                    depends_on=(static_dependency,),
                    replace=True,
                )
            else:
                builder.add(
                    command_offset + 20,
                    4,
                    f"{locator}.strsize",
                    "derived_field",
                    depends_on=("macho.symtab.n_oso.path",),
                    replace=True,
                )
        elif command == LC_DYSYMTAB:
            if command_size != 80 or dysymtab is not None:
                raise ParityError("invalid or duplicate LC_DYSYMTAB", locator=locator)
            dysymtab = struct.unpack_from("<18I", data, command_offset + 8)
            axis["fields"] = list(dysymtab)
            if expected_filetype == MH_OBJECT:
                # Every nonzero file offset in this command is layout-derived.
                for field_index in (6, 8, 10, 12, 14, 16):
                    if dysymtab[field_index]:
                        builder.add(
                            command_offset + 8 + field_index * 4,
                            4,
                            f"{locator}.offset[{field_index}]",
                            "derived_field",
                            depends_on=(static_dependency,),
                            replace=True,
                        )
        elif command in {
            LC_ID_DYLIB,
            LC_LOAD_DYLIB,
            LC_LOAD_WEAK_DYLIB,
            LC_REEXPORT_DYLIB,
            LC_LAZY_LOAD_DYLIB,
            LC_LOAD_UPWARD_DYLIB,
        }:
            if command_size < 24:
                raise ParityError("truncated dylib command", locator=locator)
            string_offset, timestamp, current, compatibility = struct.unpack_from(
                "<IIII", data, command_offset + 8
            )
            dylib_name = _parse_load_string(
                data, command_offset, command_size, string_offset, f"{locator}.name"
            )
            value = {
                "name": dylib_name,
                "timestamp": timestamp,
                "current_version": _version(current),
                "compatibility_version": _version(compatibility),
                "kind": name,
            }
            (install_names if command == LC_ID_DYLIB else dependencies).append(value)
            axis.update(value)
        elif command in {LC_LOAD_DYLINKER, LC_RPATH}:
            if command_size < 12:
                raise ParityError("truncated path command", locator=locator)
            string_offset = struct.unpack_from("<I", data, command_offset + 8)[0]
            path = _parse_load_string(
                data, command_offset, command_size, string_offset, f"{locator}.path"
            )
            axis["path"] = path
            if command == LC_RPATH:
                rpaths.append(path)
        elif command == LC_UUID:
            if command_size != 24:
                raise ParityError("invalid LC_UUID size", locator=locator)
            builder.add(
                command_offset + 8,
                16,
                "load[LC_UUID].uuid",
                "allowed_payload",
                replace=True,
                note="fresh-build UUID",
            )
            axis["uuid"] = data[command_offset + 8 : command_offset + 24].hex()
        elif command == LC_BUILD_VERSION:
            if command_size < 24:
                raise ParityError("truncated LC_BUILD_VERSION", locator=locator)
            platform, minos, sdk, ntools = struct.unpack_from(
                "<IIII", data, command_offset + 8
            )
            if command_size != 24 + ntools * 8:
                raise ParityError(
                    "LC_BUILD_VERSION tool count mismatch", locator=locator
                )
            tools = [
                struct.unpack_from("<II", data, command_offset + 24 + index * 8)
                for index in range(ntools)
            ]
            value = {
                "platform": platform,
                "minimum_platform": _version(minos),
                "sdk": _version(sdk),
                "tools": [list(item) for item in tools],
            }
            build_versions.append(value)
            axis.update(value)
        elif command == LC_DYLD_INFO_ONLY:
            if command_size != 48:
                raise ParityError("invalid LC_DYLD_INFO_ONLY size", locator=locator)
            values = struct.unpack_from("<10I", data, command_offset + 8)
            labels = ("rebase", "bind", "weak_bind", "lazy_bind", "export")
            for index, range_name in enumerate(labels):
                range_offset, range_size = values[index * 2 : index * 2 + 2]
                if range_size:
                    linkedit_ranges.append(
                        (range_offset, range_size, f"{locator}.{range_name}")
                    )
            axis["ranges"] = dict(
                zip(
                    labels,
                    (list(values[index : index + 2]) for index in range(0, 10, 2)),
                )
            )
        elif command in {
            LC_CODE_SIGNATURE,
            LC_FUNCTION_STARTS,
            LC_DATA_IN_CODE,
            LC_LINKER_OPTIMIZATION_HINT,
        }:
            if command_size != 16:
                raise ParityError("invalid linkedit-data command size", locator=locator)
            dataoff, datasize = struct.unpack_from("<II", data, command_offset + 8)
            axis.update({"dataoff": dataoff, "datasize": datasize})
            if command == LC_CODE_SIGNATURE:
                if signature_range is not None or not datasize:
                    raise ParityError(
                        "missing or duplicate code signature", locator=locator
                    )
                signature_range = (dataoff, datasize)
                builder.add(
                    command_offset + 8,
                    8,
                    f"{locator}.layout",
                    "derived_field",
                    depends_on=(
                        "macho.load[LC_UUID].uuid",
                        "macho.symtab.n_oso.mtime",
                        "macho.symtab.n_oso.path",
                    ),
                    replace=True,
                )
            elif datasize:
                linkedit_ranges.append((dataoff, datasize, locator))
                if expected_filetype == MH_OBJECT:
                    builder.add(
                        command_offset + 8,
                        8,
                        f"{locator}.layout",
                        "derived_field",
                        depends_on=(static_dependency,),
                        replace=True,
                    )
        elif command == LC_SOURCE_VERSION:
            if command_size != 16:
                raise ParityError("invalid LC_SOURCE_VERSION size", locator=locator)
            axis["version"] = struct.unpack_from("<Q", data, command_offset + 8)[0]
        elif command == LC_PREBOUND_DYLIB:
            raise ParityError("prebound dylib layout is unsupported", locator=locator)
        else:
            raise AssertionError(command)
        load_axes.append(axis)
        command_offset += command_size
    if command_offset != load_end:
        raise ParityError(
            "load commands do not exactly fill sizeofcmds", locator=prefix
        )
    if (
        seen_singletons[LC_SYMTAB] != 1
        or seen_singletons[LC_DYSYMTAB] != 1
        or seen_singletons[LC_BUILD_VERSION] != 1
    ):
        raise ParityError(
            "requires exactly one symtab, dysymtab, and build-version command",
            locator=prefix,
        )
    if symtab is None or dysymtab is None:
        raise AssertionError("singleton validation failed")

    section_data_ranges = [
        (
            section.offset,
            section.size,
            f"section[{section.index}]:{section.segment},{section.name}",
        )
        for section in sections
        if not section.is_zerofill and section.size
    ]
    segment_file_ranges = [
        (segment.fileoff, segment.filesize, f"segment[{segment.index}]:{segment.name}")
        for segment in segments
        if segment.filesize
    ]
    _check_disjoint(segment_file_ranges, total=len(data), context=f"{prefix}.segments")
    _check_disjoint(section_data_ranges, total=len(data), context=f"{prefix}.sections")
    relocation_ranges = [
        (
            section.reloff,
            section.nreloc * 8,
            f"relocations[{section.index}]:{section.segment},{section.name}",
        )
        for section in sections
        if section.nreloc
    ]
    _check_disjoint(relocation_ranges, total=len(data), context=f"{prefix}.relocations")

    symoff, nsyms, stroff, strsize = symtab
    if nsyms > MAX_SYMBOLS:
        raise ParityError("symbol table is unreasonably large", locator=prefix)
    _require_range(len(data), symoff, nsyms * 16, f"{prefix}.symtab")
    _require_range(len(data), stroff, strsize, f"{prefix}.strtab")
    if strsize == 0 or data[stroff] != 0:
        raise ParityError("Mach-O string table must begin with NUL", locator=prefix)
    all_table_ranges = linkedit_ranges + [
        (symoff, nsyms * 16, "symtab"),
        (stroff, strsize, "strtab"),
    ]
    if signature_range is not None:
        all_table_ranges.append((*signature_range, "code_signature"))
    _check_disjoint(all_table_ranges, total=len(data), context=f"{prefix}.tables")
    _check_disjoint(
        section_data_ranges + relocation_ranges + all_table_ranges,
        total=len(data),
        context=f"{prefix}.concrete_file_regions",
    )

    layout_shifted_section_indices: set[int] = set()
    object_debug_lines: list[Section] = []
    if expected_filetype == MH_OBJECT:
        if len(segments) != 1:
            raise ParityError(
                f"MH_OBJECT requires exactly one segment, found {len(segments)}",
                locator=prefix,
            )
        object_segment = segments[0]
        object_debug_lines = [
            section
            for section in sections
            if section.segment == "__DWARF" and section.name == "__debug_line"
        ]
        if len(object_debug_lines) != 1:
            raise ParityError(
                f"expected one __DWARF,__debug_line section, found {len(object_debug_lines)}",
                locator=prefix,
            )
        debug_line_for_layout = object_debug_lines[0]
        file_sections = [
            section for section in sections if not section.is_zerofill and section.size
        ]
        if not file_sections:
            raise ParityError("MH_OBJECT has no file-backed sections", locator=prefix)
        whole_file_segment = (
            object_segment.fileoff == 0
            and object_segment.filesize == len(data)
            and object_segment.vmaddr == 0
            and object_segment.vmsize == len(data)
        )
        if not whole_file_segment:
            if (
                min(section.offset for section in file_sections)
                != object_segment.fileoff
            ):
                raise ParityError(
                    "object segment file offset is not the first section offset",
                    locator=prefix,
                )
            file_end = max(section.offset + section.size for section in file_sections)
            if object_segment.fileoff + object_segment.filesize != file_end:
                raise ParityError(
                    "object segment file size is not derived from its sections",
                    locator=prefix,
                )
            for section in file_sections:
                expected_offset = object_segment.fileoff + (
                    section.addr - object_segment.vmaddr
                )
                if section.offset != expected_offset:
                    raise ParityError(
                        "object section file offset is not derived from its address",
                        locator=f"{prefix}.section[{section.index}]",
                    )
            virtual_end = max(section.addr + section.size for section in sections)
            if object_segment.vmaddr + object_segment.vmsize != virtual_end:
                raise ParityError(
                    "object segment virtual size is not derived from its sections",
                    locator=prefix,
                )
        debug_line_end = debug_line_for_layout.addr + debug_line_for_layout.size
        tail_sections = sorted(
            (
                section
                for section in sections
                if section is not debug_line_for_layout
                and section.addr >= debug_line_end
            ),
            key=lambda section: (section.addr, section.index),
        )
        cursor = debug_line_end
        for section in tail_sections:
            if not section.is_zerofill:
                raise ParityError(
                    "non-zero-fill section follows __debug_line in object address layout",
                    locator=f"{prefix}.section[{section.index}]",
                )
            expected_addr = _align_up(cursor, 1 << section.align)
            if section.addr != expected_addr:
                raise ParityError(
                    "post-__debug_line section address is not precisely alignment-derived",
                    locator=f"{prefix}.section[{section.index}]",
                )
            layout_shifted_section_indices.add(section.index)
            cursor = section.addr + section.size

    symbols: list[SymbolRecord] = []
    oso_candidates: list[tuple[SymbolRecord, int, bytes]] = []
    for index in range(nsyms):
        entry_offset = symoff + index * 16
        n_strx, n_type, n_sect, n_desc, n_value = struct.unpack_from(
            "<IBBHQ", data, entry_offset
        )
        if n_strx >= strsize:
            raise ParityError(
                "symbol string index is outside the string table",
                locator=f"{prefix}.symbol[{index}]",
            )
        if n_strx:
            name_bytes, _ = _cstring(
                data,
                stroff + n_strx,
                stroff + strsize,
                f"{prefix}.symbol[{index}].name",
            )
        else:
            name_bytes = b""
        symbol = SymbolRecord(
            index,
            _text(name_bytes, f"{prefix}.symbol[{index}].name"),
            "stab"
            if n_type & N_STAB
            else N_TYPE_NAMES.get(n_type & N_TYPE, f"unknown-0x{n_type & N_TYPE:x}"),
            n_sect,
            bool(n_type & N_EXT),
            bool(n_type & N_PEXT),
            n_type & N_STAB,
            n_desc,
            n_value,
            n_type,
        )
        if not n_type & N_STAB and (n_type & N_TYPE) not in N_TYPE_NAMES:
            raise ParityError(
                "unknown nlist_64 base type", locator=f"{prefix}.symbol[{index}]"
            )
        if n_sect > len(sections):
            raise ParityError(
                "nlist_64 section ordinal is out of range",
                locator=f"{prefix}.symbol[{index}]",
            )
        symbols.append(symbol)
        if (
            expected_filetype == MH_OBJECT
            and not n_type & N_STAB
            and (n_type & N_TYPE) == 0xE
            and n_sect in layout_shifted_section_indices
        ):
            target = sections[n_sect - 1]
            if not target.addr <= n_value < target.addr + target.size:
                raise ParityError(
                    "layout-derived nlist value is outside its zero-fill section",
                    locator=f"{prefix}.symbol[{index}]",
                )
            builder.add(
                entry_offset + 8,
                8,
                f"symtab.symbol[{index}].layout_value",
                "derived_field",
                depends_on=(static_dependency,),
                canonical_payload=struct.pack("<Q", n_value - target.addr),
                replace=True,
            )
        if expected_filetype == MH_DYLIB:
            builder.add(
                entry_offset,
                4,
                f"symtab.symbol[{index}].n_strx",
                "derived_field",
                depends_on=("macho.symtab.n_oso.path",),
                replace=True,
            )
        if n_type == N_OSO and name_bytes:
            oso_candidates.append((symbol, n_strx, name_bytes))
        if expected_filetype == MH_OBJECT and n_type == N_OSO:
            raise ParityError(
                "archive object unexpectedly contains N_OSO",
                locator=f"{prefix}.symbol[{index}]",
            )

    ilocal, nlocal, iextdef, nextdef, iundef, nundef = dysymtab[:6]
    if (
        ilocal,
        ilocal + nlocal,
        iextdef,
        iextdef + nextdef,
        iundef,
        iundef + nundef,
    ) != (
        0,
        iextdef,
        iextdef,
        iundef,
        iundef,
        nsyms,
    ):
        raise ParityError(
            "dysymtab symbol partitions are not contiguous and complete", locator=prefix
        )

    provenance: list[ProvenanceLocator] = []
    if expected_filetype == MH_DYLIB:
        if len(oso_candidates) != 1:
            raise ParityError(
                f"expected one real N_OSO entry, found {len(oso_candidates)}",
                locator=prefix,
            )
        oso_symbol, oso_strx, oso_path = oso_candidates[0]
        match = CACHE_OBJECT_RE.fullmatch(oso_path)
        if match is None:
            raise ParityError(
                "N_OSO path is not a local-cache o/<32hex>/<object>.o path",
                locator=prefix,
            )
        object_basename = _text(match.group(2), f"{prefix}.n_oso.basename")
        path_offset = stroff + oso_strx
        builder.add(
            path_offset,
            len(oso_path),
            "symtab.n_oso.path",
            "allowed_payload",
            replace=True,
            note="unique real N_OSO local-cache object path",
        )
        builder.add(
            symoff + oso_symbol.index * 16 + 8,
            8,
            "symtab.n_oso.mtime",
            "allowed_payload",
            replace=True,
            note="N_OSO source-object mtime",
        )
        provenance.append(
            ProvenanceLocator(
                kind="dynamic_cache_object",
                source_locator=f"{prefix}.symtab.symbol[{oso_symbol.index}].N_OSO",
                path=_text(oso_path, f"{prefix}.n_oso.path"),
                path_offset=path_offset,
                path_size=len(oso_path),
                expected_basename=object_basename,
                expected_mtime_seconds=oso_symbol.value,
            )
        )
        if (
            seen_singletons[LC_UUID] != 1
            or seen_singletons[LC_CODE_SIGNATURE] != 1
            or seen_singletons[LC_ID_DYLIB] != 1
        ):
            raise ParityError(
                "dynamic library requires one UUID, code signature, and install name",
                locator=prefix,
            )
    else:
        if (
            signature_range is not None
            or seen_singletons[LC_UUID]
            or seen_singletons[LC_ID_DYLIB]
        ):
            raise ParityError(
                "MH_OBJECT contains dynamic-only commands", locator=prefix
            )

    dwarf_directories: tuple[DwarfDirectory, ...] = ()
    dwarf_files: tuple[DwarfFile, ...] = ()
    if expected_filetype == MH_OBJECT:
        debug_line = object_debug_lines[0]
        dwarf_directories, dwarf_files, dwarf_provenance = parse_dwarf_v4_debug_line(
            data[debug_line.offset : debug_line.offset + debug_line.size],
            file_offset=debug_line.offset,
            builder=builder,
            locator_prefix="dwarf.debug_line",
            allowed_dependency_locator=static_dependency,
        )
        provenance.append(dwarf_provenance)
        for section in sections:
            if not section.nreloc:
                continue
            for relocation_index in range(section.nreloc):
                relocation_offset = section.reloff + relocation_index * 8
                relocation_address, relocation_word = struct.unpack_from(
                    "<iI", data, relocation_offset
                )
                if relocation_address < 0:
                    raise ParityError(
                        "scattered or negative Mach-O relocation is unsupported",
                        locator=f"{prefix}.relocation[{section.index},{relocation_index}]",
                    )
                relocation_symbol = relocation_word & 0xFFFFFF
                relocation_pcrel = (relocation_word >> 24) & 1
                relocation_length = (relocation_word >> 25) & 3
                relocation_external = (relocation_word >> 27) & 1
                relocation_type = (relocation_word >> 28) & 0xF
                if (
                    not relocation_external
                    and relocation_symbol in layout_shifted_section_indices
                ):
                    if (
                        relocation_pcrel
                        or relocation_length != 3
                        or relocation_type != 0
                    ):
                        raise ParityError(
                            "unsupported relocation form targeting a layout-derived section",
                            locator=f"{prefix}.relocation[{section.index},{relocation_index}]",
                        )
                    if relocation_address > section.size - 8:
                        raise ParityError(
                            "layout-derived relocation addend is outside its source section",
                            locator=f"{prefix}.relocation[{section.index},{relocation_index}]",
                        )
                    target = sections[relocation_symbol - 1]
                    addend_offset = section.offset + relocation_address
                    addend = struct.unpack_from("<Q", data, addend_offset)[0]
                    if not target.addr <= addend < target.addr + target.size:
                        raise ParityError(
                            "local relocation addend is outside its target section",
                            locator=f"{prefix}.relocation[{section.index},{relocation_index}]",
                        )
                    builder.add(
                        addend_offset,
                        8,
                        f"relocation[{section.index},{relocation_index}].layout_addend",
                        "derived_field",
                        depends_on=(static_dependency,),
                        canonical_payload=struct.pack("<Q", addend - target.addr),
                        replace=True,
                    )
                if section is debug_line:
                    builder.add(
                        relocation_offset,
                        4,
                        f"relocation[{section.index},{relocation_index}].address",
                        "derived_field",
                        depends_on=(static_dependency,),
                        replace=True,
                    )

    code_directory: CodeDirectory | None = None
    if expected_filetype == MH_DYLIB:
        assert signature_range is not None
        signature_offset, signature_size = signature_range
        if signature_offset + signature_size != len(data):
            raise ParityError(
                "code signature must be the final file region", locator=prefix
            )
        code_directory = parse_code_directory(
            data, signature_offset, signature_size, builder=builder
        )
        if (
            not install_names
            or code_directory.identifier != PurePosixPath(install_names[0]["name"]).name
        ):
            raise ParityError(
                "CodeDirectory identifier does not match install-name basename",
                locator=prefix,
            )
        # Signature alignment is linker-derived padding, not a blanket string-table mask.
        string_end = stroff + strsize
        if string_end > signature_offset:
            raise ParityError("string table overlaps code signature", locator=prefix)
        if string_end < signature_offset:
            padding = data[string_end:signature_offset]
            if any(padding):
                raise ParityError(
                    "nonzero padding before code signature", locator=prefix
                )
            builder.add(
                string_end,
                len(padding),
                "code_signature.alignment_padding",
                "derived_padding",
                depends_on=("macho.symtab.n_oso.path",),
                replace=True,
            )

    axes = {
        "format": "Mach-O",
        "endianness": "little",
        "bits": 64,
        "architecture": "arm64",
        "cpu_subtype": cpu_subtype,
        "filetype": "MH_DYLIB" if expected_filetype == MH_DYLIB else "MH_OBJECT",
        "flags": flags,
        "load_commands": load_axes,
        "segments": [segment.json() for segment in segments],
        "build_versions": build_versions,
        "install_name": install_names,
        "dependencies": dependencies,
        "rpaths": rpaths,
        "symbol_partitions": {
            "local": [ilocal, nlocal],
            "external_defined": [iextdef, nextdef],
            "undefined": [iundef, nundef],
        },
        "code_directory": code_directory.json() if code_directory else None,
        "dwarf_directories": [item.json() for item in dwarf_directories],
        "dwarf_files": [item.json() for item in dwarf_files],
    }
    return MachOArtifact(
        raw=data,
        label=label,
        filetype=filetype,
        flags=flags,
        segments=tuple(segments),
        symbols=tuple(symbols),
        structured_axes=axes,
        atoms=builder.finish(),
        provenance=tuple(provenance),
        code_directory=code_directory,
        dwarf_directories=dwarf_directories,
        dwarf_files=dwarf_files,
    )


def parse_dynamic_artifact(
    source: bytes | bytearray | memoryview | os.PathLike[str] | str,
    *,
    label: str | None = None,
) -> MachOArtifact:
    """Parse a thin little-endian arm64 ``MH_DYLIB`` and prove its layout."""

    data, effective_label = _read_source(source, label)
    return _parse_macho_bytes(data, effective_label, MH_DYLIB, "macho")


def parse_macho_object(
    data: bytes, *, label: str = "<object>", locator_prefix: str = "macho_object"
) -> MachOArtifact:
    """Parse one arm64 ``MH_OBJECT`` member; useful to offline fixture tests."""

    return _parse_macho_bytes(bytes(data), label, MH_OBJECT, locator_prefix)


def _parse_ascii_int(value: bytes, base: int, locator: str) -> int:
    stripped = value.rstrip(b" ")
    if not stripped or any(
        byte not in (b"0123456789" if base == 10 else b"01234567") for byte in stripped
    ):
        raise ParityError("invalid archive numeric field", locator=locator)
    try:
        return int(stripped, base)
    except ValueError as exc:
        raise ParityError("invalid archive numeric field", locator=locator) from exc


def parse_ranlib(
    data: bytes, member_offsets: Mapping[int, str], *, base_offset: int = 0
) -> tuple[RanlibEntry, ...]:
    """Parse a Darwin little-endian ``__.SYMDEF`` ranlib body."""

    locator = "archive.ranlib"
    if len(data) < 8:
        raise ParityError("ranlib body is truncated", locator=locator)
    table_size = struct.unpack_from("<I", data, 0)[0]
    if table_size % 8 or table_size > len(data) - 8:
        raise ParityError("invalid ranlib table byte count", locator=locator)
    strings_size_offset = 4 + table_size
    strings_size = struct.unpack_from("<I", data, strings_size_offset)[0]
    strings_offset = strings_size_offset + 4
    if strings_offset + strings_size != len(data):
        raise ParityError(
            "ranlib string table does not exactly fill member", locator=locator
        )
    strings = data[strings_offset:]
    boundaries: dict[int, bytes] = {}
    cursor = 0
    while cursor < len(strings):
        value, end = _cstring(strings, cursor, len(strings), f"{locator}.strings")
        if not value:
            if any(strings[cursor:]):
                raise ParityError(
                    "nonzero bytes after ranlib string terminator", locator=locator
                )
            break
        boundaries[cursor] = value
        cursor = end
    entries: list[RanlibEntry] = []
    for index in range(table_size // 8):
        string_offset, member_offset = struct.unpack_from("<II", data, 4 + index * 8)
        if string_offset not in boundaries:
            raise ParityError(
                "ranlib string offset is not at a string boundary",
                locator=f"{locator}[{index}]",
            )
        if member_offset not in member_offsets:
            raise ParityError(
                "ranlib member offset does not name an archive header",
                locator=f"{locator}[{index}]",
            )
        entries.append(
            RanlibEntry(
                index,
                _text(boundaries[string_offset], f"{locator}[{index}].symbol"),
                string_offset,
                member_offset,
            )
        )
    return tuple(entries)


def parse_static_archive(
    source: bytes | bytearray | memoryview | os.PathLike[str] | str,
    *,
    label: str | None = None,
) -> ArchiveArtifact:
    """Parse a Darwin archive containing ``__.SYMDEF`` then arm64 objects."""

    data, effective_label = _read_source(source, label)
    if not data.startswith(AR_MAGIC):
        raise ParityError("not a Darwin ar archive")
    builder = _AtomBuilder(data, "archive")
    builder.add(0, len(AR_MAGIC), "magic")
    raw_members: list[dict[str, Any]] = []
    offset = len(AR_MAGIC)
    while offset < len(data):
        if len(raw_members) >= MAX_ARCHIVE_MEMBERS:
            raise ParityError("too many archive members")
        _require_range(len(data), offset, AR_HEADER_SIZE, "archive.member.header")
        header = data[offset : offset + AR_HEADER_SIZE]
        if header[58:60] != b"`\n":
            raise ParityError(
                "invalid archive member trailer",
                locator=f"archive.member[{len(raw_members)}]",
            )
        name_field = header[:16]
        date = _parse_ascii_int(header[16:28], 10, "archive.member.date")
        uid = _parse_ascii_int(header[28:34], 10, "archive.member.uid")
        gid = _parse_ascii_int(header[34:40], 10, "archive.member.gid")
        mode = _parse_ascii_int(header[40:48], 8, "archive.member.mode")
        stored_size = _parse_ascii_int(header[48:58], 10, "archive.member.size")
        payload_offset = offset + AR_HEADER_SIZE
        _require_range(len(data), payload_offset, stored_size, "archive.member.payload")
        extended = name_field.rstrip(b" ")
        if not extended.startswith(b"#1/"):
            raise ParityError(
                "only Darwin #1/<length> member names are supported",
                locator=f"archive.member[{len(raw_members)}]",
            )
        name_length = _parse_ascii_int(
            extended[3:], 10, "archive.member.extended_name_length"
        )
        if name_length == 0 or name_length > stored_size:
            raise ParityError(
                "invalid extended archive name length",
                locator=f"archive.member[{len(raw_members)}]",
            )
        name_storage = data[payload_offset : payload_offset + name_length]
        logical_name_bytes = name_storage.rstrip(b"\0")
        if not logical_name_bytes or any(name_storage[len(logical_name_bytes) :]):
            raise ParityError(
                "invalid Darwin extended member name padding",
                locator=f"archive.member[{len(raw_members)}]",
            )
        member_name = _text(
            logical_name_bytes, f"archive.member[{len(raw_members)}].name"
        )
        body_offset = payload_offset + name_length
        body_size = stored_size - name_length
        raw_members.append(
            {
                "index": len(raw_members),
                "name": member_name,
                "header_offset": offset,
                "payload_offset": payload_offset,
                "name_length": name_length,
                "body_offset": body_offset,
                "body_size": body_size,
                "stored_size": stored_size,
                "date": date,
                "uid": uid,
                "gid": gid,
                "mode": mode,
            }
        )
        member_index = len(raw_members) - 1
        builder.add(offset, AR_HEADER_SIZE, f"member[{member_index}].header")
        builder.add(
            payload_offset, name_length, f"member[{member_index}].extended_name"
        )
        if body_size:
            builder.add(body_offset, body_size, f"member[{member_index}].body")
        offset = payload_offset + stored_size
        if stored_size & 1:
            _require_range(
                len(data), offset, 1, f"archive.member[{member_index}].padding"
            )
            if data[offset : offset + 1] != b"\n":
                raise ParityError(
                    "archive odd-member padding must be newline",
                    locator=f"archive.member[{member_index}]",
                )
            builder.add(
                offset,
                1,
                f"member[{member_index}].padding",
                "derived_padding",
                depends_on=("dwarf.global_cache_directory",),
            )
            offset += 1
    if (
        offset != len(data)
        or len(raw_members) < 2
        or raw_members[0]["name"] != "__.SYMDEF"
    ):
        raise ParityError("archive must contain __.SYMDEF followed by object members")
    if any(item["name"] == "__.SYMDEF" for item in raw_members[1:]):
        raise ParityError("duplicate __.SYMDEF member")
    member_offset_names = {
        item["header_offset"]: item["name"] for item in raw_members[1:]
    }
    ranlib_body = data[
        raw_members[0]["body_offset"] : raw_members[0]["body_offset"]
        + raw_members[0]["body_size"]
    ]
    ranlib = parse_ranlib(
        ranlib_body, member_offset_names, base_offset=raw_members[0]["body_offset"]
    )
    # Ranlib member offsets are derived from preceding member sizes; all other index bytes are invariant.
    for entry in ranlib:
        absolute = raw_members[0]["body_offset"] + 4 + entry.index * 8 + 4
        builder.add(
            absolute,
            4,
            f"ranlib[{entry.index}].member_offset",
            "derived_field",
            depends_on=("dwarf.global_cache_directory",),
            replace=True,
        )

    members: list[ArchiveMember] = []
    all_symbols: list[SymbolRecord] = []
    provenance: list[ProvenanceLocator] = []
    object_names: set[str] = set()
    for raw in raw_members:
        member_object: MachOArtifact | None = None
        if raw["index"] > 0:
            if raw["name"] in object_names:
                raise ParityError(
                    "duplicate archive object member name",
                    locator=f"archive.member[{raw['index']}]",
                )
            object_names.add(raw["name"])
            object_bytes = data[
                raw["body_offset"] : raw["body_offset"] + raw["body_size"]
            ]
            member_object = parse_macho_object(
                object_bytes,
                label=f"{effective_label}({raw['name']})",
                locator_prefix=f"object[{raw['index'] - 1}:{raw['name']}]",
            )
            # Replace the opaque object range with its rebased proof atoms.
            for atom in member_object.atoms:
                local_locator = f"member[{raw['index']}].{atom.locator}"
                builder.add(
                    raw["body_offset"] + atom.offset,
                    atom.size,
                    local_locator,
                    atom.role,
                    depends_on=atom.depends_on,
                    note=atom.note,
                    canonical_payload=atom.canonical_payload,
                    replace=True,
                )
            # The ar member's decimal size is precisely derived from its reconstructed object and extended name.
            builder.add(
                raw["header_offset"] + 48,
                10,
                f"member[{raw['index']}].stored_size",
                "derived_field",
                depends_on=("dwarf.global_cache_directory",),
                replace=True,
            )
            all_symbols.extend(member_object.symbols)
            for item in member_object.provenance:
                provenance.append(
                    dataclasses.replace(
                        item,
                        source_locator=f"archive.member[{raw['index']}].{item.source_locator}",
                        path_offset=raw["body_offset"] + item.path_offset,
                    )
                )
        members.append(
            ArchiveMember(
                raw["index"],
                raw["name"],
                raw["header_offset"],
                raw["body_offset"],
                raw["stored_size"],
                member_object,
            )
        )
    if len(provenance) != 1:
        raise ParityError(
            f"archive must expose exactly one cache builtin path, found {len(provenance)}"
        )

    external_defined = [
        symbol.name
        for symbol in all_symbols
        if symbol.external
        and not symbol.stab
        and symbol.base_type not in {"undefined", "prebound-undefined"}
    ]
    ranlib_symbols = [entry.symbol for entry in ranlib]
    if Counter(ranlib_symbols) != Counter(external_defined):
        raise ParityError(
            "ranlib index is not the multiset of externally defined object symbols"
        )
    member_by_offset = {item["header_offset"]: item["name"] for item in raw_members}
    expected_object_for_symbol: dict[str, Counter[str]] = defaultdict(Counter)
    for member in members[1:]:
        assert member.object is not None
        for symbol in member.object.symbols:
            if (
                symbol.external
                and not symbol.stab
                and symbol.base_type not in {"undefined", "prebound-undefined"}
            ):
                expected_object_for_symbol[symbol.name][member.name] += 1
    actual_object_for_symbol: dict[str, Counter[str]] = defaultdict(Counter)
    for entry in ranlib:
        actual_object_for_symbol[entry.symbol][
            member_by_offset[entry.member_offset]
        ] += 1
    if actual_object_for_symbol != expected_object_for_symbol:
        raise ParityError("ranlib entries point at the wrong object member")

    axes = {
        "format": "Darwin ar",
        "member_order": [member.name for member in members],
        "members": [member.json() for member in members],
        "normalized_metadata": [
            {
                "name": raw["name"],
                "date": raw["date"],
                "uid": raw["uid"],
                "gid": raw["gid"],
                "mode": raw["mode"],
            }
            for raw in raw_members
        ],
        "index": [entry.json() for entry in ranlib],
        "objects": [
            member.object.structured_axes
            for member in members
            if member.object is not None
        ],
    }
    archive_atoms = list(builder.finish())
    existing_atom_locators = {atom.locator for atom in archive_atoms}
    for raw in raw_members:
        padding_locator = f"archive.member[{raw['index']}].padding"
        if padding_locator not in existing_atom_locators:
            archive_atoms.append(
                ByteAtom(
                    padding_locator,
                    "derived_padding",
                    raw["payload_offset"] + raw["stored_size"],
                    0,
                    b"",
                    ("dwarf.global_cache_directory",),
                    "zero-length archive member padding",
                )
            )
    return ArchiveArtifact(
        raw=data,
        label=effective_label,
        members=tuple(members),
        ranlib=ranlib,
        structured_axes=axes,
        symbols=tuple(all_symbols),
        atoms=tuple(archive_atoms),
        provenance=tuple(provenance),
    )


def canonicalize_atoms(atoms: Sequence[ByteAtom]) -> bytes:
    """Return the stable logical serialization used for canonical digests."""

    values: list[dict[str, Any]] = []
    for atom in sorted(atoms, key=lambda item: item.locator):
        value: dict[str, Any] = {"locator": atom.locator, "role": atom.role}
        if atom.role == "invariant":
            value["payload_sha256"] = sha256_bytes(atom.payload)
            value["size"] = atom.size
        else:
            value["canonical_token"] = atom.locator
            value["depends_on"] = list(atom.depends_on)
            if atom.canonical_payload is not None:
                value["canonical_payload_hex"] = atom.canonical_payload.hex()
        values.append(value)
    return canonical_json_bytes(values)


def canonical_digest(atoms: Sequence[ByteAtom]) -> str:
    return sha256_bytes(canonicalize_atoms(atoms))


def canonicalize_dynamic_artifact(artifact: MachOArtifact) -> bytes:
    if artifact.filetype != MH_DYLIB:
        raise ParityError("not a dynamic artifact")
    return canonicalize_atoms(artifact.atoms)


def canonicalize_static_archive(artifact: ArchiveArtifact) -> bytes:
    return canonicalize_atoms(artifact.atoms)


def _symbol_axes(
    left: Sequence[SymbolRecord], right: Sequence[SymbolRecord]
) -> dict[str, Any]:
    def entry(key: tuple[Any, ...], count: int) -> dict[str, Any]:
        name, base_type, section, external, private_external, stab, desc, type_code = (
            key
        )
        return {
            "name": name,
            "base_type": base_type,
            "section": section,
            "external": external,
            "private_external": private_external,
            "stab": stab,
            "desc": desc,
            "type_code": type_code,
            "count": count,
        }

    left_keys = [item.comparison_key() for item in left]
    right_keys = [item.comparison_key() for item in right]
    left_counter = Counter(left_keys)
    right_counter = Counter(right_keys)
    return {
        "order_equal": left_keys == right_keys,
        "multiset_equal": left_counter == right_counter,
        "left_count": len(left),
        "right_count": len(right),
        "left_only": [
            entry(key, count)
            for key, count in sorted(
                (left_counter - right_counter).items(), key=lambda item: repr(item[0])
            )
        ],
        "right_only": [
            entry(key, count)
            for key, count in sorted(
                (right_counter - left_counter).items(), key=lambda item: repr(item[0])
            )
        ],
    }


def _normalized_axes(value: Any) -> Any:
    """Remove fields which are independently proven by allowed/derived atoms."""

    if isinstance(value, list):
        return [_normalized_axes(item) for item in value]
    if isinstance(value, tuple):
        return [_normalized_axes(item) for item in value]
    if not isinstance(value, dict):
        return value
    result: dict[str, Any] = {}
    for key, item in value.items():
        if key in {
            "uuid",
            "code_directory",
            "symoff",
            "stroff",
            "strsize",
            "dataoff",
            "datasize",
            "addr",
            "fileoff",
            "filesize",
            "vmsize",
            "reloff",
            "member_offset",
            "header_offset",
            "data_offset",
            "stored_size",
            "object_sha256",
            "index",
            "ranges",
            "fields",
            "size",
        }:
            continue
        if key == "dwarf_directories":
            result[key] = [
                {
                    inner_key: inner_value
                    for inner_key, inner_value in directory.items()
                    if inner_key not in {"path", "offset", "size"}
                }
                for directory in item
            ]
            continue
        result[key] = _normalized_axes(item)
    return result


def _compare_parsed(
    artifact_kind: str,
    left: MachOArtifact | ArchiveArtifact,
    right: MachOArtifact | ArchiveArtifact,
) -> Comparison:
    failures: list[str] = []
    left_atoms = {atom.locator: atom for atom in left.atoms}
    right_atoms = {atom.locator: atom for atom in right.atoms}
    unpaired = tuple(sorted(set(left_atoms) ^ set(right_atoms)))
    if unpaired:
        failures.append(f"{len(unpaired)} atom locators are unpaired")
    allowed_edits: list[dict[str, Any]] = []
    derived_fields: list[dict[str, Any]] = []
    invalid_derived: list[str] = []

    def dependency_changed(dependency: str) -> bool:
        candidates = [dependency]
        candidates.extend(
            locator
            for locator, atom in left_atoms.items()
            if atom.role == "allowed_payload" and locator.endswith(dependency)
        )
        return any(
            candidate in left_atoms
            and candidate in right_atoms
            and left_atoms[candidate].payload != right_atoms[candidate].payload
            for candidate in candidates
        )

    def byte_difference_count(left_payload: bytes, right_payload: bytes) -> int:
        shared = sum(
            left_byte != right_byte
            for left_byte, right_byte in zip(left_payload, right_payload)
        )
        return shared + abs(len(left_payload) - len(right_payload))

    for locator in sorted(set(left_atoms) & set(right_atoms)):
        left_atom = left_atoms[locator]
        right_atom = right_atoms[locator]
        if (
            left_atom.role != right_atom.role
            or left_atom.depends_on != right_atom.depends_on
        ):
            failures.append(f"atom contract differs at {locator}")
            continue
        equal = left_atom.payload == right_atom.payload
        if left_atom.role == "invariant" and not equal:
            failures.append(f"invariant atom differs: {locator}")
        elif left_atom.role == "allowed_payload":
            allowed_edits.append(
                {
                    "locator": locator,
                    "left": left_atom.json(include_payload=False),
                    "right": right_atom.json(include_payload=False),
                    "changed": not equal,
                    "byte_difference_count": byte_difference_count(
                        left_atom.payload, right_atom.payload
                    ),
                    "size_delta": right_atom.size - left_atom.size,
                }
            )
        elif left_atom.role in {"derived_field", "derived_padding"}:
            changed_dependencies = [
                dependency
                for dependency in left_atom.depends_on
                if dependency_changed(dependency)
            ]
            canonical_payload_equal = (
                left_atom.canonical_payload == right_atom.canonical_payload
            )
            valid_change = canonical_payload_equal and (
                equal or bool(changed_dependencies)
            )
            if not valid_change:
                invalid_derived.append(locator)
                if not canonical_payload_equal:
                    failures.append(
                        f"derived atom canonical payload differs: {locator}"
                    )
                else:
                    failures.append(
                        f"derived atom changed without a changed allowed base: {locator}"
                    )
            derived_fields.append(
                {
                    "locator": locator,
                    "role": left_atom.role,
                    "left": left_atom.json(),
                    "right": right_atom.json(),
                    "changed": not equal,
                    "byte_difference_count": byte_difference_count(
                        left_atom.payload, right_atom.payload
                    ),
                    "size_delta": right_atom.size - left_atom.size,
                    "changed_dependencies": changed_dependencies,
                    "canonical_payload_equal": canonical_payload_equal,
                    "valid": valid_change,
                }
            )
    left_axes = _normalized_axes(dict(left.structured_axes))
    right_axes = _normalized_axes(dict(right.structured_axes))
    axes_equal = left_axes == right_axes
    if not axes_equal:
        failures.append("structured axes differ")
    symbols = _symbol_axes(left.symbols, right.symbols)
    if not symbols["order_equal"]:
        failures.append("symbol order differs")
    if not symbols["multiset_equal"]:
        failures.append("symbol multiset differs")
    left_digest = left.canonical_digest
    right_digest = right.canonical_digest
    if left_digest != right_digest:
        failures.append("canonical digests differ")
    left_provenance = [item.json() for item in left.provenance]
    right_provenance = [item.json() for item in right.provenance]
    # Allowed paths may differ; their stable kind/basename relationship may not.
    left_provenance_shape = [
        (item.kind, item.expected_basename) for item in left.provenance
    ]
    right_provenance_shape = [
        (item.kind, item.expected_basename) for item in right.provenance
    ]
    if left_provenance_shape != right_provenance_shape:
        failures.append("provenance locator kind or basename differs")
    failures = list(dict.fromkeys(failures))
    return Comparison(
        artifact_kind=artifact_kind,
        left_label=left.label,
        right_label=right.label,
        left_raw_sha256=left.raw_sha256,
        right_raw_sha256=right.raw_sha256,
        left_size=len(left.raw),
        right_size=len(right.raw),
        structured_axes={"equal": axes_equal, "left": left_axes, "right": right_axes},
        symbol_axes=symbols,
        allowed_edit_atoms=tuple(allowed_edits),
        derived_fields=tuple(derived_fields),
        left_canonical_digest=left_digest,
        right_canonical_digest=right_digest,
        uncovered_left_bytes=(),
        uncovered_right_bytes=(),
        unpaired_atoms=unpaired,
        invalid_derived_fields=tuple(invalid_derived),
        provenance_locators={"left": left_provenance, "right": right_provenance},
        verdict="pass" if not failures else "fail",
        failures=tuple(failures),
    )


def compare_dynamic_artifact_pair(
    left: bytes | bytearray | memoryview | os.PathLike[str] | str | MachOArtifact,
    right: bytes | bytearray | memoryview | os.PathLike[str] | str | MachOArtifact,
    *,
    left_label: str | None = None,
    right_label: str | None = None,
) -> Comparison:
    """Parse, canonicalize, and compare two dynamic artifacts."""

    left_artifact = (
        left
        if isinstance(left, MachOArtifact)
        else parse_dynamic_artifact(left, label=left_label)
    )
    right_artifact = (
        right
        if isinstance(right, MachOArtifact)
        else parse_dynamic_artifact(right, label=right_label)
    )
    if left_artifact.filetype != MH_DYLIB or right_artifact.filetype != MH_DYLIB:
        raise ParityError("dynamic comparison received a non-MH_DYLIB artifact")
    return _compare_parsed("dynamic", left_artifact, right_artifact)


def compare_static_archive_pair(
    left: bytes | bytearray | memoryview | os.PathLike[str] | str | ArchiveArtifact,
    right: bytes | bytearray | memoryview | os.PathLike[str] | str | ArchiveArtifact,
    *,
    left_label: str | None = None,
    right_label: str | None = None,
) -> Comparison:
    """Parse, canonicalize, and compare two static Darwin archives."""

    left_artifact = (
        left
        if isinstance(left, ArchiveArtifact)
        else parse_static_archive(left, label=left_label)
    )
    right_artifact = (
        right
        if isinstance(right, ArchiveArtifact)
        else parse_static_archive(right, label=right_label)
    )
    return _compare_parsed("static", left_artifact, right_artifact)


# Short aliases kept intentional for callers that do not use the “pair” wording.
compare_dynamic_artifacts = compare_dynamic_artifact_pair
compare_static_archives = compare_static_archive_pair


def bind_provenance(
    locator: ProvenanceLocator,
    *,
    exists: bool,
    mtime_seconds: int | None = None,
    sha256: str | None = None,
) -> dict[str, Any]:
    """Validate caller-supplied filesystem facts without touching the filesystem.

    A CLI can stat and hash ``locator.path`` and pass the observations here.  Dynamic
    N_OSO mtimes are checked exactly; static builtin hashes are deliberately reported
    rather than hard-coded because the policy layer owns the expected digest.
    """

    failures: list[str] = []
    if not exists:
        failures.append("path does not exist")
    if locator.expected_mtime_seconds is not None:
        if mtime_seconds is None:
            failures.append("mtime was not supplied")
        elif mtime_seconds != locator.expected_mtime_seconds:
            failures.append("mtime does not match N_OSO n_value")
    if sha256 is not None and not re.fullmatch(r"[0-9a-f]{64}", sha256):
        failures.append("sha256 is not 64 lowercase hexadecimal characters")
    return {
        "locator": locator.json(),
        "exists": exists,
        "mtime_seconds": mtime_seconds,
        "sha256": sha256,
        "verdict": "pass" if not failures else "fail",
        "failures": failures,
    }


__all__ = [
    "ArchiveArtifact",
    "ArchiveMember",
    "ByteAtom",
    "CodeDirectory",
    "Comparison",
    "DwarfDirectory",
    "DwarfFile",
    "MachOArtifact",
    "ParityError",
    "ProvenanceLocator",
    "RanlibEntry",
    "Section",
    "Segment",
    "SymbolRecord",
    "bind_provenance",
    "canonical_digest",
    "canonicalize_atoms",
    "canonicalize_dynamic_artifact",
    "canonicalize_static_archive",
    "compare_dynamic_artifact_pair",
    "compare_dynamic_artifacts",
    "compare_static_archive_pair",
    "compare_static_archives",
    "parse_code_directory",
    "parse_dynamic_artifact",
    "parse_dwarf_v4_debug_line",
    "parse_macho_object",
    "parse_ranlib",
    "parse_static_archive",
    "sha256_bytes",
]
