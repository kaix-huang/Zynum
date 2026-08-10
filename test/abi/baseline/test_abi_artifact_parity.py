# Copyright (C) 2026 Zynum contributors
# SPDX-License-Identifier: LGPL-3.0-or-later

"""Offline fixtures for the byte parser and fresh-artifact verifier."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import io
import json
import os
import random
import struct
import sys
import tempfile
import tracemalloc
import types
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[3]


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


core = load_module(
    "test_abi_artifact_parity_core", ROOT / "tools/abi_artifact_parity.py"
)
verifier = load_module(
    "test_verify_abi_artifact_parity", ROOT / "tools/verify_abi_artifact_parity.py"
)
artifacts = verifier.load_benchmark_artifacts_module(ROOT)


def align(value: int, boundary: int) -> int:
    return (value + boundary - 1) & -boundary


def fixed(value: str) -> bytes:
    raw = value.encode()
    assert len(raw) <= 16
    return raw.ljust(16, b"\0")


def load_string(command: int, value: str, *, dylib: bool) -> bytes:
    raw = value.encode() + b"\0"
    header_size = 24 if dylib else 12
    size = align(header_size + len(raw), 8)
    if dylib:
        header = struct.pack("<IIIIII", command, size, header_size, 0, 0x10000, 0x10000)
    else:
        header = struct.pack("<III", command, size, header_size)
    return header + raw + b"\0" * (size - header_size - len(raw))


def code_directory(
    signed: bytes, identifier: str, *, flags: int = 0x20002, slot_type: int = 0
) -> bytes:
    page_exp = 12
    page_size = 1 << page_exp
    slots = (len(signed) + page_size - 1) // page_size
    ident = identifier.encode() + b"\0"
    hash_offset = 88 + len(ident)
    hashes = b"".join(
        hashlib.sha256(
            signed[index * page_size : min((index + 1) * page_size, len(signed))]
        ).digest()
        for index in range(slots)
    )
    length = hash_offset + len(hashes)
    directory = struct.pack(
        ">IIIIIIIII4BIIIIQQQQ",
        0xFADE0C02,
        length,
        0x20400,
        flags,
        hash_offset,
        88,
        0,
        slots,
        len(signed),
        32,
        2,
        0,
        page_exp,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
    )
    return (
        struct.pack(">IIIII", 0xFADE0CC0, 20 + length, 1, slot_type, 20)
        + directory
        + ident
        + hashes
    )


def dynamic_fixture(
    *,
    cache_prefix: str = "/fixture-a",
    digest: str = "1" * 32,
    object_name: str = "zynum.o",
    mtime: int = 100,
    uuid: bytes = bytes(range(16)),
    minos: int = 0x000D0000,
    sdk: int = 0x001A0400,
    install_name: str = "@rpath/libzynum_blas.dylib",
    dependency: str = "/usr/lib/libSystem.B.dylib",
    rpath: str = "@loader_path",
    extra_oso: bool = False,
    signature_flags: int = 0x20002,
    signature_slot: int = 0,
) -> bytes:
    oso = f"{cache_prefix}/local-cache/o/{digest}/{object_name}"
    strings = b"\0" + oso.encode() + b"\0"
    string_offsets = [1]
    if extra_oso:
        other = f"{cache_prefix}/local-cache/o/{'2' * 32}/other.o"
        string_offsets.append(len(strings))
        strings += other.encode() + b"\0"
    nsyms = len(string_offsets)
    commands: list[bytes] = []
    # Offsets are filled after the fixed-size command array has been assembled.
    commands.extend(
        [
            b"SYMTAB",
            struct.pack("<II18I", 0xB, 80, 0, nsyms, nsyms, 0, nsyms, 0, *([0] * 12)),
            struct.pack("<IIIIII", 0x32, 24, 1, minos, sdk, 0),
            load_string(0xD, install_name, dylib=True),
            load_string(0xC, dependency, dylib=True),
            load_string(0x8000001C, rpath, dylib=False),
            struct.pack("<II16s", 0x1B, 24, uuid),
            b"CODESIG",
        ]
    )
    sizeofcmds = sum(
        24 if item == b"SYMTAB" else 16 if item == b"CODESIG" else len(item)
        for item in commands
    )
    symoff = 32 + sizeofcmds
    stroff = symoff + nsyms * 16
    signature_offset = stroff + len(strings)
    signature_size = 20 + 88 + len(Path(install_name).name.encode()) + 1 + 32
    real_commands: list[bytes] = []
    for item in commands:
        if item == b"SYMTAB":
            real_commands.append(
                struct.pack("<IIIIII", 0x2, 24, symoff, nsyms, stroff, len(strings))
            )
        elif item == b"CODESIG":
            real_commands.append(
                struct.pack("<IIII", 0x1D, 16, signature_offset, signature_size)
            )
        else:
            real_commands.append(item)
    header = struct.pack(
        "<IiiIIIII", 0xFEEDFACF, 0x0100000C, 0, 6, len(real_commands), sizeofcmds, 0, 0
    )
    symbols = b"".join(
        struct.pack("<IBBHQ", offset, 0x66, 0, 0, mtime + index)
        for index, offset in enumerate(string_offsets)
    )
    signed = header + b"".join(real_commands) + symbols + strings
    signature = code_directory(
        signed, Path(install_name).name, flags=signature_flags, slot_type=signature_slot
    )
    assert len(signature) == signature_size
    return signed + signature


def uleb(value: int) -> bytes:
    result = bytearray()
    while True:
        byte = value & 0x7F
        value >>= 7
        result.append(byte | (0x80 if value else 0))
        if not value:
            return bytes(result)


def dwarf_line(cache_prefix: str, digest: str, *, program: bytes = b"\0") -> bytes:
    directory = f"{cache_prefix}/global-cache/b/{digest}".encode()
    scalar = struct.pack("<BBBbBB", 1, 1, 1, -5, 14, 2) + b"\0"
    directories = directory + b"\0\0"
    files = b"builtin.zig\0" + uleb(1) + uleb(0) + uleb(0) + b"\0"
    prologue = scalar + directories + files
    body = struct.pack("<HI", 4, len(prologue)) + prologue + program
    return struct.pack("<I", len(body)) + body


def object_fixture(
    *,
    cache_prefix: str = "/fixture-a",
    digest: str = "3" * 32,
    program: bytes = b"\0",
    local_type: int = 0x0E,
    external_type: int = 0x0F,
) -> bytes:
    dwarf = dwarf_line(cache_prefix, digest, program=program)
    strings = b"\0_hidden\0_foo\0"
    nsyms = 2
    sizeofcmds = 152 + 24 + 80 + 24
    section_offset = 32 + sizeofcmds
    symoff = section_offset + len(dwarf)
    stroff = symoff + nsyms * 16
    total = stroff + len(strings)
    segment = struct.pack(
        "<II16sQQQQiiII",
        0x19,
        152,
        fixed(""),
        0,
        total,
        0,
        total,
        7,
        7,
        1,
        0,
    ) + struct.pack(
        "<16s16sQQIIIIIIII",
        fixed("__debug_line"),
        fixed("__DWARF"),
        0,
        len(dwarf),
        section_offset,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
    )
    symtab = struct.pack("<IIIIII", 0x2, 24, symoff, nsyms, stroff, len(strings))
    dysymtab = struct.pack("<II18I", 0xB, 80, 0, 1, 1, 1, 2, 0, *([0] * 12))
    build = struct.pack("<IIIIII", 0x32, 24, 1, 0x000D0000, 0x001A0400, 0)
    header = struct.pack("<IiiIIIII", 0xFEEDFACF, 0x0100000C, 0, 1, 4, sizeofcmds, 0, 0)
    symbols = struct.pack("<IBBHQ", 1, local_type, 1, 0, 0) + struct.pack(
        "<IBBHQ", 9, external_type, 1, 0, 0
    )
    return header + segment + symtab + dysymtab + build + dwarf + symbols + strings


def layout_shifted_object_fixture(
    *,
    cache_prefix: str,
    digest: str = "4" * 32,
    relocation_type: int = 0,
    relocation_relative_offset: int = 3,
    tail32_symbol_relative_offset: int = 2,
    tail_symbol_out_of_range: bool = False,
) -> bytes:
    """Build a DWARF object whose two aligned zero-fill tails move with DWARF."""

    dwarf = dwarf_line(cache_prefix, digest)
    strings = b"\0_data\0_tail16\0_foo\0"
    nsyms = 3
    section_count = 4
    segment_size = 72 + 80 * section_count
    sizeofcmds = segment_size + 24 + 80 + 24
    data_offset = 32 + sizeofcmds
    debug_offset = data_offset + 8
    relocation_offset = debug_offset + len(dwarf)
    symoff = relocation_offset + 8
    stroff = symoff + nsyms * 16
    total = stroff + len(strings)

    debug_addr = 8
    tail16_addr = align(debug_addr + len(dwarf), 16)
    tail32_addr = align(tail16_addr + 5, 32)
    addend = tail32_addr + relocation_relative_offset
    tail16_symbol_value = tail16_addr + 1
    tail32_symbol_value = tail32_addr + (
        99 if tail_symbol_out_of_range else tail32_symbol_relative_offset
    )

    sections = [
        struct.pack(
            "<16s16sQQIIIIIIII",
            fixed("__data"),
            fixed("__DATA"),
            0,
            8,
            data_offset,
            3,
            relocation_offset,
            1,
            0,
            0,
            0,
            0,
        ),
        struct.pack(
            "<16s16sQQIIIIIIII",
            fixed("__debug_line"),
            fixed("__DWARF"),
            debug_addr,
            len(dwarf),
            debug_offset,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
        ),
        struct.pack(
            "<16s16sQQIIIIIIII",
            fixed("__tail16"),
            fixed("__DATA"),
            tail16_addr,
            5,
            0,
            4,
            0,
            0,
            core.S_ZEROFILL,
            0,
            0,
            0,
        ),
        struct.pack(
            "<16s16sQQIIIIIIII",
            fixed("__tail32"),
            fixed("__DATA"),
            tail32_addr,
            7,
            0,
            5,
            0,
            0,
            core.S_ZEROFILL,
            0,
            0,
            0,
        ),
    ]
    segment = struct.pack(
        "<II16sQQQQiiII",
        core.LC_SEGMENT_64,
        segment_size,
        fixed(""),
        0,
        total,
        0,
        total,
        7,
        7,
        section_count,
        0,
    ) + b"".join(sections)
    symtab = struct.pack(
        "<IIIIII", core.LC_SYMTAB, 24, symoff, nsyms, stroff, len(strings)
    )
    dysymtab = struct.pack(
        "<II18I", core.LC_DYSYMTAB, 80, 0, 2, 2, 1, 3, 0, *([0] * 12)
    )
    build = struct.pack(
        "<IIIIII", core.LC_BUILD_VERSION, 24, 1, 0x000D0000, 0x001A0400, 0
    )
    header = struct.pack(
        "<IiiIIIII",
        0xFEEDFACF,
        core.CPU_TYPE_ARM64,
        0,
        core.MH_OBJECT,
        4,
        sizeofcmds,
        0,
        0,
    )
    symbols = b"".join(
        (
            struct.pack("<IBBHQ", 1, 0x0E, 1, 0, 0),
            struct.pack("<IBBHQ", 7, 0x0E, 3, 0, tail16_symbol_value),
            struct.pack("<IBBHQ", 15, 0x0F, 4, 0, tail32_symbol_value),
        )
    )
    relocation_word = 4 | (3 << 25) | (relocation_type << 28)
    relocation = struct.pack("<iI", 0, relocation_word)
    return (
        header
        + segment
        + symtab
        + dysymtab
        + build
        + struct.pack("<Q", addend)
        + dwarf
        + relocation
        + symbols
        + strings
    )


def ar_header(
    name: str,
    body: bytes,
    *,
    date: int = 0,
    uid: int = 0,
    gid: int = 0,
    mode: int = 0o100644,
) -> bytes:
    name_raw = name.encode()
    stored_size = len(name_raw) + len(body)
    fields = (
        f"#1/{len(name_raw)}".encode().ljust(16),
        str(date).encode().ljust(12),
        str(uid).encode().ljust(6),
        str(gid).encode().ljust(6),
        f"{mode:o}".encode().ljust(8),
        str(stored_size).encode().ljust(10),
        b"`\n",
    )
    payload = b"".join(fields) + name_raw + body
    return payload + (b"\n" if stored_size & 1 else b"")


def archive_fixture(
    *,
    cache_prefix: str = "/fixture-a",
    digest: str = "3" * 32,
    program: bytes = b"\0",
    date: int = 0,
) -> bytes:
    obj = object_fixture(cache_prefix=cache_prefix, digest=digest, program=program)
    symbol = b"_foo\0"
    empty_ranlib = (
        struct.pack("<I", 8)
        + struct.pack("<II", 0, 0)
        + struct.pack("<I", len(symbol))
        + symbol
    )
    first = ar_header("__.SYMDEF", empty_ranlib, date=date)
    object_offset = 8 + len(first)
    ranlib = (
        struct.pack("<I", 8)
        + struct.pack("<II", 0, object_offset)
        + struct.pack("<I", len(symbol))
        + symbol
    )
    first = ar_header("__.SYMDEF", ranlib, date=date)
    assert object_offset == 8 + len(first)
    return b"!<arch>\n" + first + ar_header("zcu.o", obj, date=date)


def layout_shifted_archive_fixture(**kwargs) -> bytes:
    obj = layout_shifted_object_fixture(**kwargs)
    symbol = b"_foo\0"
    empty_ranlib = (
        struct.pack("<I", 8)
        + struct.pack("<II", 0, 0)
        + struct.pack("<I", len(symbol))
        + symbol
    )
    first = ar_header("__.SYMDEF", empty_ranlib)
    object_offset = 8 + len(first)
    ranlib = (
        struct.pack("<I", 8)
        + struct.pack("<II", 0, object_offset)
        + struct.pack("<I", len(symbol))
        + symbol
    )
    first = ar_header("__.SYMDEF", ranlib)
    assert object_offset == 8 + len(first)
    return b"!<arch>\n" + first + ar_header("zcu.o", obj)


def comparison_fixtures(**changes):
    left_dynamic = dynamic_fixture()
    right_dynamic = dynamic_fixture(
        cache_prefix=changes.get("dynamic_prefix", "/longer-fixture-b"),
        mtime=changes.get("mtime", 200),
        uuid=changes.get("uuid", bytes(reversed(range(16)))),
        minos=changes.get("minos", 0x000D0000),
        sdk=changes.get("sdk", 0x001A0400),
        install_name=changes.get("install_name", "@rpath/libzynum_blas.dylib"),
        dependency=changes.get("dependency", "/usr/lib/libSystem.B.dylib"),
        rpath=changes.get("rpath", "@loader_path"),
    )
    left_static = archive_fixture()
    right_static = archive_fixture(
        cache_prefix=changes.get("static_prefix", "/longer-fixture-b")
    )
    return left_dynamic, right_dynamic, left_static, right_static


class DenseAtomOracle:
    """Small independent bytewise model for differential interval tests."""

    def __init__(self, data: bytes, prefix: str) -> None:
        self.data = data
        self.prefix = prefix
        self.cover = [None] * len(data)

    def add(
        self,
        start: int,
        size: int,
        locator: str,
        role: str = "invariant",
        *,
        depends_on=(),
        note=None,
        canonical_payload=None,
        replace: bool = False,
    ) -> None:
        if (
            start < 0
            or size < 0
            or start > len(self.data)
            or size > len(self.data) - start
        ):
            raise core.ParityError(
                f"range [{start}, {start + size}) exceeds {len(self.data)} bytes",
                locator=locator,
            )
        if size == 0:
            return
        full = f"{self.prefix}.{locator}" if self.prefix else locator
        value = (full, role, tuple(depends_on), note, canonical_payload)
        for index in range(start, start + size):
            prior = self.cover[index]
            if prior is not None and not replace:
                raise core.ParityError(
                    f"byte {index} is already assigned to {prior[0]}", locator=full
                )
            if prior is not None and replace and prior[1] != "invariant":
                raise core.ParityError(
                    f"cannot replace {prior[1]} atom at byte {index}", locator=full
                )
        self.cover[start : start + size] = [value] * size

    def finish(self):
        gaps = core._uncovered_ranges(self.cover)
        if gaps:
            raise core.ParityError(
                f"uncovered byte ranges: {gaps}", locator=self.prefix
            )
        result = []
        occurrences = {}
        start = 0
        while start < len(self.data):
            value = self.cover[start]
            end = start + 1
            while end < len(self.data) and self.cover[end] == value:
                end += 1
            base_locator, role, dependencies, note, canonical_payload = value
            occurrence = occurrences.get(base_locator, 0)
            occurrences[base_locator] = occurrence + 1
            locator = (
                base_locator if occurrence == 0 else f"{base_locator}#{occurrence}"
            )
            result.append(
                core.ByteAtom(
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
            start = end
        return tuple(result)


class AtomBuilderTests(unittest.TestCase):
    def test_dense_oracle_differential(self) -> None:
        rng = random.Random(0xA81A2026)
        for case in range(40):
            data = bytes(range(64))
            sparse = core._AtomBuilder(data, f"case[{case}]")
            dense = DenseAtomOracle(data, f"case[{case}]")
            for claim in range(100):
                start = rng.randrange(len(data) + 1)
                size = rng.randrange(len(data) - start + 1)
                role = rng.choice(
                    ("invariant", "invariant", "allowed_payload", "derived_field")
                )
                kwargs = {
                    "depends_on": tuple(
                        f"dependency[{index}]" for index in range(rng.randrange(3))
                    ),
                    "note": rng.choice((None, "note-a", "note-b")),
                    "canonical_payload": (
                        bytes((claim & 0xFF,)) if role == "derived_field" else None
                    ),
                    "replace": bool(rng.randrange(2)),
                }
                locator = f"claim[{rng.randrange(12)}]"
                outcomes = []
                for builder in (sparse, dense):
                    try:
                        builder.add(start, size, locator, role, **kwargs)
                    except core.ParityError as error:
                        outcomes.append(str(error))
                    else:
                        outcomes.append(None)
                self.assertEqual(outcomes[0], outcomes[1], (case, claim, kwargs))

            results = []
            for builder in (sparse, dense):
                try:
                    results.append(builder.finish())
                except core.ParityError as error:
                    results.append(str(error))
            self.assertEqual(results[0], results[1], case)

    def test_overlap_reports_the_leftmost_conflict_and_is_atomic(self) -> None:
        builder = core._AtomBuilder(b"abcdef", "fixture")
        builder.add(2, 3, "first")
        with self.assertRaises(core.ParityError) as caught:
            builder.add(0, 4, "second")
        self.assertEqual(
            str(caught.exception),
            "fixture.second: byte 2 is already assigned to fixture.first",
        )
        builder.add(0, 2, "prefix")
        builder.add(5, 1, "suffix")
        self.assertEqual(
            [(atom.locator, atom.offset, atom.size) for atom in builder.finish()],
            [
                ("fixture.prefix", 0, 2),
                ("fixture.first", 2, 3),
                ("fixture.suffix", 5, 1),
            ],
        )

    def test_replace_covers_gaps_and_invariants(self) -> None:
        builder = core._AtomBuilder(b"abcdefgh", "fixture")
        builder.add(0, 2, "base")
        builder.add(4, 2, "base")
        builder.add(1, 4, "replacement", "allowed_payload", replace=True)
        builder.add(6, 2, "tail")
        atoms = builder.finish()
        self.assertEqual(
            [(atom.locator, atom.role, atom.offset, atom.size) for atom in atoms],
            [
                ("fixture.base", "invariant", 0, 1),
                ("fixture.replacement", "allowed_payload", 1, 4),
                ("fixture.base#1", "invariant", 5, 1),
                ("fixture.tail", "invariant", 6, 2),
            ],
        )

    def test_replace_rejects_first_noninvariant_without_partial_write(self) -> None:
        builder = core._AtomBuilder(b"abcdef", "fixture")
        builder.add(0, 2, "head")
        builder.add(4, 2, "volatile", "allowed_payload")
        with self.assertRaises(core.ParityError) as caught:
            builder.add(1, 4, "replacement", replace=True)
        self.assertEqual(
            str(caught.exception),
            "fixture.replacement: cannot replace allowed_payload atom at byte 4",
        )
        builder.add(2, 2, "gap")
        self.assertEqual(
            [atom.locator for atom in builder.finish()],
            ["fixture.head", "fixture.gap", "fixture.volatile"],
        )

    def test_gap_diagnostics_remain_ordered(self) -> None:
        builder = core._AtomBuilder(b"0123456789", "fixture")
        builder.add(3, 1, "middle")
        builder.add(8, 1, "end")
        with self.assertRaises(core.ParityError) as caught:
            builder.finish()
        self.assertEqual(
            str(caught.exception),
            "fixture: uncovered byte ranges: ({'offset': 0, 'size': 3}, "
            "{'offset': 4, 'size': 4}, {'offset': 9, 'size': 1})",
        )

    def test_merge_uses_complete_metadata_and_preserves_locator_suffixes(self) -> None:
        builder = core._AtomBuilder(b"abcdefgh", "fixture")
        common = {
            "role": "derived_field",
            "depends_on": ("source",),
            "note": "same",
            "canonical_payload": b"canonical",
        }
        builder.add(0, 1, "field", **common)
        builder.add(1, 1, "field", **common)
        builder.add(2, 1, "field", **{**common, "note": "different"})
        builder.add(3, 1, "separator")
        builder.add(4, 1, "field", **common)
        builder.add(5, 1, "field", **common)
        builder.add(6, 1, "separator")
        builder.add(7, 1, "field", **common)
        atoms = builder.finish()
        self.assertEqual(
            [(atom.locator, atom.offset, atom.size, atom.note) for atom in atoms],
            [
                ("fixture.field", 0, 2, "same"),
                ("fixture.field#1", 2, 1, "different"),
                ("fixture.separator", 3, 1, None),
                ("fixture.field#2", 4, 2, "same"),
                ("fixture.separator#1", 6, 1, None),
                ("fixture.field#3", 7, 1, "same"),
            ],
        )
        self.assertEqual(atoms[0].canonical_payload, b"canonical")

    def test_zero_size_is_a_noop_and_nonempty_claims_are_capped(self) -> None:
        with mock.patch.object(core, "MAX_ATOM_CLAIMS", 2):
            builder = core._AtomBuilder(b"abc", "fixture")
            builder.add(3, 0, "zero")
            builder.add(0, 1, "first")
            with self.assertRaises(core.ParityError):
                builder.add(0, 1, "overlap")
            builder.add(1, 1, "second")
            with self.assertRaises(core.ParityError) as caught:
                builder.add(2, 1, "third")
        self.assertEqual(
            str(caught.exception),
            "fixture.third: non-empty atom claim limit 2 exceeded",
        )

    def test_adversarial_reverse_microintervals_keep_logarithmic_height(self) -> None:
        count = 16_384
        builder = core._AtomBuilder(bytes(count), "fixture")
        for index in reversed(range(count)):
            builder.add(index, 1, f"byte[{index}]")
        self.assertIsNotNone(builder._root)
        self.assertLessEqual(builder._root.height, 2 * count.bit_length())
        atoms = builder.finish()
        self.assertEqual(len(atoms), count)
        self.assertEqual((atoms[0].offset, atoms[-1].offset), (0, count - 1))

    def test_logical_512_mib_input_uses_interval_scaled_memory(self) -> None:
        class LogicalBytes:
            def __len__(self) -> int:
                return core.MAX_ARTIFACT_BYTES

        tracemalloc.start()
        try:
            baseline, _ = tracemalloc.get_traced_memory()
            builder = core._AtomBuilder(LogicalBytes(), "logical")
            for index in range(4_096):
                offset = (index * 104_729) % core.MAX_ARTIFACT_BYTES
                builder.add(offset, 1, f"byte[{index}]")
            current, peak = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()
        self.assertLess(current - baseline, 8 * 1024 * 1024)
        self.assertLess(peak - baseline, 12 * 1024 * 1024)
        self.assertEqual(builder._claim_count, 4_096)

    def test_existing_canonical_and_report_hashes_are_stable(self) -> None:
        dynamic = core.parse_dynamic_artifact(dynamic_fixture())
        static = core.parse_static_archive(archive_fixture())
        comparison = core.compare_dynamic_artifact_pair(
            dynamic_fixture(),
            dynamic_fixture(
                cache_prefix="/longer-fixture-b",
                mtime=200,
                uuid=bytes(reversed(range(16))),
            ),
        )
        self.assertEqual(
            dynamic.canonical_digest,
            "a08957b07974d1891b692bbd450d98e57f06a6be10264f856220d4a7aa818f04",
        )
        self.assertEqual(
            static.canonical_digest,
            "e9a909166d2bd61f27cf9710ab4d2627a1967b58c5494c779576f77b90082cd0",
        )
        self.assertEqual(
            hashlib.sha256(comparison.json_bytes()).hexdigest(),
            "31677a7776a18a8b9c8a7ac84272174a176b0085eda56fba4debddb4395a829b",
        )


class CoreFixtureTests(unittest.TestCase):
    def assert_parity_failure(
        self, left: bytes, right: bytes, *, dynamic: bool, contains: str
    ) -> None:
        compare = (
            core.compare_dynamic_artifact_pair
            if dynamic
            else core.compare_static_archive_pair
        )
        result = compare(left, right)
        self.assertEqual(result.verdict, "fail")
        self.assertTrue(
            any(contains in item for item in result.failures), result.failures
        )

    def assert_parse_failure(self, operation, value: bytes, contains: str) -> None:
        with self.assertRaises(core.ParityError) as caught:
            operation(value)
        self.assertIn(contains, str(caught.exception))

    def test_raw_identical_and_all_four_volatility_classes(self) -> None:
        dynamic = dynamic_fixture()
        static = archive_fixture()
        self.assertEqual(
            core.compare_dynamic_artifact_pair(dynamic, dynamic).verdict, "pass"
        )
        self.assertEqual(
            core.compare_static_archive_pair(static, static).verdict, "pass"
        )
        left_dynamic, right_dynamic, left_static, right_static = comparison_fixtures()
        dynamic_result = core.compare_dynamic_artifact_pair(left_dynamic, right_dynamic)
        static_result = core.compare_static_archive_pair(left_static, right_static)
        self.assertEqual(dynamic_result.verdict, "pass", dynamic_result.failures)
        self.assertEqual(static_result.verdict, "pass", static_result.failures)
        self.assertNotEqual(
            dynamic_result.left_raw_sha256, dynamic_result.right_raw_sha256
        )
        self.assertNotEqual(
            static_result.left_raw_sha256, static_result.right_raw_sha256
        )

    def test_equal_length_uuid_noso_mtime_and_signature(self) -> None:
        left = dynamic_fixture(cache_prefix="/fixture-a", mtime=1, uuid=b"a" * 16)
        right = dynamic_fixture(cache_prefix="/fixture-b", mtime=2, uuid=b"b" * 16)
        result = core.compare_dynamic_artifact_pair(left, right)
        self.assertEqual(result.verdict, "pass", result.failures)
        self.assertTrue(any(item["changed"] for item in result.allowed_edit_atoms))

    def test_equal_and_variable_length_dwarf_offsets(self) -> None:
        equal_left = archive_fixture(cache_prefix="/fixture-a")
        equal_right = archive_fixture(cache_prefix="/fixture-b")
        variable = archive_fixture(cache_prefix="/much/longer/fixture-cache-root")
        self.assertEqual(
            core.compare_static_archive_pair(equal_left, equal_right).verdict, "pass"
        )
        result = core.compare_static_archive_pair(equal_left, variable)
        self.assertEqual(result.verdict, "pass", result.failures)
        self.assertTrue(any(item["changed"] for item in result.derived_fields))

    def test_variable_dwarf_rebases_aligned_tails_relocation_and_nlists(self) -> None:
        left = layout_shifted_archive_fixture(cache_prefix="/fixture-a")
        right = layout_shifted_archive_fixture(
            cache_prefix=(
                "/a/much/longer/fixture-cache-root-that-shifts-two-alignments"
            )
        )
        result = core.compare_static_archive_pair(left, right)
        self.assertEqual(result.verdict, "pass", result.failures)
        self.assertEqual(result.left_canonical_digest, result.right_canonical_digest)
        self.assertEqual(result.uncovered_left_bytes, ())
        self.assertEqual(result.uncovered_right_bytes, ())
        self.assertEqual(result.unpaired_atoms, ())
        self.assertEqual(result.invalid_derived_fields, ())
        self.assertEqual(result.failures, ())

        prefix = "archive.member[1].object[0:zcu.o]."
        expected_changed_locators = {
            prefix + "load[0:LC_SEGMENT_64].section[2].addr",
            prefix + "load[0:LC_SEGMENT_64].section[3].addr",
            prefix + "relocation[1,0].layout_addend",
            prefix + "symtab.symbol[1].layout_value",
            prefix + "symtab.symbol[2].layout_value",
        }
        derived_by_locator = {item["locator"]: item for item in result.derived_fields}
        self.assertLessEqual(expected_changed_locators, set(derived_by_locator))
        for locator in expected_changed_locators:
            with self.subTest(locator=locator):
                self.assertTrue(derived_by_locator[locator]["changed"])
                self.assertTrue(derived_by_locator[locator]["valid"])
                self.assertEqual(derived_by_locator[locator]["left"]["size"], 8)
                self.assertEqual(derived_by_locator[locator]["right"]["size"], 8)

    def test_layout_shifted_nlist_relative_offset_change_fails(self) -> None:
        short_root = "/fixture-a"
        long_root = "/a/much/longer/fixture-cache-root-that-shifts-two-alignments"
        left = layout_shifted_archive_fixture(cache_prefix=short_root)
        correctly_rebased = layout_shifted_archive_fixture(cache_prefix=long_root)
        wrong_relative_offset = layout_shifted_archive_fixture(
            cache_prefix=long_root, tail32_symbol_relative_offset=3
        )

        good = core.compare_static_archive_pair(left, correctly_rebased)
        self.assertEqual(good.verdict, "pass", good.failures)
        self.assertEqual(good.left_canonical_digest, good.right_canonical_digest)
        self.assertEqual(good.invalid_derived_fields, ())
        self.assertEqual(good.failures, ())

        # Both artifacts are structurally valid: only the in-range relative
        # nlist offset differs from the frozen +2 contract.
        core.parse_static_archive(wrong_relative_offset)
        bad = core.compare_static_archive_pair(left, wrong_relative_offset)
        locator = "archive.member[1].object[0:zcu.o].symtab.symbol[2].layout_value"
        self.assertEqual(bad.verdict, "fail")
        self.assertNotEqual(bad.left_canonical_digest, bad.right_canonical_digest)
        self.assertEqual(bad.invalid_derived_fields, (locator,))
        self.assertEqual(
            bad.failures,
            (
                f"derived atom canonical payload differs: {locator}",
                "canonical digests differ",
            ),
        )
        evidence = next(
            item for item in bad.derived_fields if item["locator"] == locator
        )
        self.assertFalse(evidence["canonical_payload_equal"])
        self.assertFalse(evidence["valid"])
        self.assertEqual(evidence["left"]["canonical_payload_hex"], "0200000000000000")
        self.assertEqual(evidence["right"]["canonical_payload_hex"], "0300000000000000")

    def test_layout_shifted_relocation_relative_offset_change_fails(self) -> None:
        short_root = "/fixture-a"
        long_root = "/a/much/longer/fixture-cache-root-that-shifts-two-alignments"
        left = layout_shifted_archive_fixture(cache_prefix=short_root)
        correctly_rebased = layout_shifted_archive_fixture(cache_prefix=long_root)
        wrong_relative_offset = layout_shifted_archive_fixture(
            cache_prefix=long_root, relocation_relative_offset=4
        )

        good = core.compare_static_archive_pair(left, correctly_rebased)
        self.assertEqual(good.verdict, "pass", good.failures)
        self.assertEqual(good.left_canonical_digest, good.right_canonical_digest)
        self.assertEqual(good.invalid_derived_fields, ())
        self.assertEqual(good.failures, ())

        # Both artifacts are structurally valid: only the in-range ARM64 local
        # unsigned 8-byte addend offset differs from the frozen +3 contract.
        core.parse_static_archive(wrong_relative_offset)
        bad = core.compare_static_archive_pair(left, wrong_relative_offset)
        locator = "archive.member[1].object[0:zcu.o].relocation[1,0].layout_addend"
        self.assertEqual(bad.verdict, "fail")
        self.assertNotEqual(bad.left_canonical_digest, bad.right_canonical_digest)
        self.assertEqual(bad.invalid_derived_fields, (locator,))
        self.assertEqual(
            bad.failures,
            (
                f"derived atom canonical payload differs: {locator}",
                "canonical digests differ",
            ),
        )
        evidence = next(
            item for item in bad.derived_fields if item["locator"] == locator
        )
        self.assertFalse(evidence["canonical_payload_equal"])
        self.assertFalse(evidence["valid"])
        self.assertEqual(evidence["left"]["canonical_payload_hex"], "0300000000000000")
        self.assertEqual(evidence["right"]["canonical_payload_hex"], "0400000000000000")

    def test_layout_shifted_relocation_and_nlist_validation_is_fail_closed(
        self,
    ) -> None:
        cases = (
            (
                {"relocation_type": 1},
                "object[0:zcu.o].relocation[1,0]: unsupported relocation form",
            ),
            (
                {"tail_symbol_out_of_range": True},
                "object[0:zcu.o].symbol[2]: layout-derived nlist value is outside",
            ),
        )
        for changes, expected in cases:
            with self.subTest(changes=changes):
                self.assert_parse_failure(
                    core.parse_static_archive,
                    layout_shifted_archive_fixture(
                        cache_prefix="/fixture-a", **changes
                    ),
                    expected,
                )

    def test_platform_install_dependency_and_rpath_are_invariant(self) -> None:
        left, _, _, _ = comparison_fixtures()
        cases = {
            "minimum": dynamic_fixture(minos=0x000E0000),
            "sdk": dynamic_fixture(sdk=0x001A0500),
            "install": dynamic_fixture(install_name="@rpath/libother.dylib"),
            "dependency": dynamic_fixture(dependency="/usr/lib/libobjc.A.dylib"),
            "rpath": dynamic_fixture(rpath="@executable_path"),
        }
        for label, right in cases.items():
            with self.subTest(label=label):
                self.assert_parity_failure(
                    left, right, dynamic=True, contains="invariant atom differs"
                )

    def test_uuid_command_fields_are_not_allowed(self) -> None:
        value = bytearray(dynamic_fixture())
        command_offset = value.find(struct.pack("<I", 0x1B), 32)
        struct.pack_into("<I", value, command_offset + 4, 16)
        self.assert_parse_failure(
            core.parse_dynamic_artifact, bytes(value), "invalid LC_UUID size"
        )

    def test_extra_and_bad_noso_are_rejected(self) -> None:
        self.assert_parse_failure(
            core.parse_dynamic_artifact,
            dynamic_fixture(extra_oso=True),
            "expected one real N_OSO",
        )
        value = dynamic_fixture(cache_prefix="relative")
        self.assert_parse_failure(
            core.parse_dynamic_artifact, value, "N_OSO path is not"
        )

    def test_signature_bad_hash_flags_and_slot_are_rejected(self) -> None:
        bad_hash = bytearray(dynamic_fixture())
        bad_hash[-1] ^= 1
        cases = [
            (bytes(bad_hash), "page hash"),
            (dynamic_fixture(signature_flags=2), "not ad-hoc linker-signed"),
            (dynamic_fixture(signature_slot=1), "one primary CodeDirectory"),
        ]
        for value, message in cases:
            with self.subTest(message=message):
                self.assert_parse_failure(core.parse_dynamic_artifact, value, message)

    def test_archive_metadata_order_size_padding_and_index_are_rejected(self) -> None:
        base = archive_fixture()
        metadata = archive_fixture(date=1)
        self.assert_parity_failure(
            base, metadata, dynamic=False, contains="invariant atom differs"
        )
        for program_size in range(1, 5):
            padded = archive_fixture(program=b"\0" * program_size)
            padding = bytearray(padded)
            first_size = int(padding[8 + 48 : 8 + 58].rstrip())
            second_header = 8 + 60 + first_size + (first_size & 1)
            second_size = int(padding[second_header + 48 : second_header + 58].rstrip())
            if second_size & 1:
                break
        else:
            self.fail("fixture builder did not produce odd archive member size")
        padding_offset = second_header + 60 + second_size
        padding[padding_offset] = 0
        self.assert_parse_failure(
            core.parse_static_archive, bytes(padding), "padding must be newline"
        )
        bad_size = bytearray(base)
        bad_size[8 + 48 : 8 + 58] = b"9999999999"
        self.assert_parse_failure(core.parse_static_archive, bytes(bad_size), "exceeds")
        bad_index = bytearray(base)
        ranlib_member_offset = 8 + 60 + len("__.SYMDEF") + 8
        struct.pack_into("<I", bad_index, ranlib_member_offset, 1)
        self.assert_parse_failure(
            core.parse_static_archive, bytes(bad_index), "member offset"
        )
        # Swapping members necessarily violates the mandatory first __.SYMDEF rule.
        self.assert_parse_failure(
            core.parse_static_archive,
            b"!<arch>\n" + base[base.find(b"#1/5", 8) :],
            "__.SYMDEF",
        )

    def test_other_dwarf_line_bytes_are_invariant(self) -> None:
        left = archive_fixture(program=b"\0")
        right = archive_fixture(program=b"\1")
        self.assert_parity_failure(
            left, right, dynamic=False, contains="invariant atom differs"
        )

    def test_symbol_visibility_and_type_are_rejected(self) -> None:
        left = archive_fixture()
        visibility = b"!<arch>\n" + ar_header(
            "__.SYMDEF", struct.pack("<I", 0) + struct.pack("<I", 0)
        )
        # Direct object parse gives stable symbol locators without needing to rebuild ranlib.
        original = object_fixture()
        for label, changed in (
            ("visibility", object_fixture(external_type=0x0E)),
            ("type", object_fixture(external_type=0x05)),
        ):
            with self.subTest(label=label):
                if label == "visibility":
                    self.assertNotEqual(
                        core.parse_macho_object(original).symbols,
                        core.parse_macho_object(changed).symbols,
                    )
                else:
                    self.assert_parse_failure(
                        core.parse_macho_object, changed, "unknown nlist_64 base type"
                    )
        self.assertTrue(left.startswith(b"!<arch>\n"))
        self.assertTrue(visibility.startswith(b"!<arch>\n"))

    def test_fat_32_big_truncated_and_unknown_are_rejected(self) -> None:
        cases = [
            (b"\xca\xfe\xba\xbe" + b"\0" * 28, "fat, 32-bit, or big-endian"),
            (b"\xce\xfa\xed\xfe" + b"\0" * 28, "fat, 32-bit, or big-endian"),
            (b"\xfe\xed\xfa\xcf" + b"\0" * 28, "fat, 32-bit, or big-endian"),
            (b"\xcf\xfa\xed", "truncated"),
            (b"NOPE" + b"\0" * 28, "not a little-endian"),
        ]
        for value, message in cases:
            with self.subTest(message=message):
                self.assert_parse_failure(core.parse_dynamic_artifact, value, message)


def policy_fixture() -> dict:
    return {
        "schema_version": 3,
        "artifact_build_configuration": {
            "configuration_id": "fixture-v1",
            "command_template": [
                "zig",
                "build",
                "install",
                "--prefix",
                "<temporary-install-prefix>",
                "--cache-dir",
                "<isolated-local-cache>",
                "--global-cache-dir",
                "<isolated-global-cache>",
            ],
            "resolved": {"zig_version": "0.16.0", "artifact_sdk": "26.4"},
            "resolved_build_inputs": {
                "generated_builtin_zig_sha256": {
                    "dynamic_library": "a" * 64,
                    "static_library": "b" * 64,
                }
            },
        },
    }


def control_fixture():
    policy = policy_fixture()
    observation = {
        "schema": copy.deepcopy(verifier.OBSERVATION_SCHEMA),
        "artifact_build_configuration": copy.deepcopy(
            policy["artifact_build_configuration"]
        ),
        "sources": {
            "declarations": [
                {"exported_name": "foo", "visibility": "default"},
                {"exported_name": "hidden", "visibility": "hidden"},
            ]
        },
    }
    receipt = {
        "schema": verifier.RECEIPT_SCHEMA,
        "configuration": copy.deepcopy(policy["artifact_build_configuration"]),
    }
    return policy, observation, receipt


class VerifierUnitTests(unittest.TestCase):
    def test_missing_fresh_builtin_failure_omits_sensitive_cache_path(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fresh_root = Path(temporary) / "fresh-private"
            global_cache = fresh_root / "global-cache"
            global_cache.mkdir(parents=True)
            missing = global_cache / "b" / ("3" * 32) / "builtin.zig"
            locator = types.SimpleNamespace(
                path=str(missing),
                kind="static_builtin",
                source_locator="fixture.DWARF",
            )
            redactor = verifier.SensitivePathRedactor()
            redactor.add_root(fresh_root, "<fresh-root>")
            with self.assertRaises(verifier.VerificationError) as caught:
                verifier._bind_builtin(
                    locator,
                    cache_root=global_cache,
                    expected_sha256="0" * 64,
                    label="fresh.dynamic.builtin",
                )
            report = verifier.failure_report(caught.exception, redactor=redactor)
            self.assertEqual(report["failures"][0]["code"], "provenance_path_missing")
            serialized = json.dumps(report)
            self.assertNotIn(str(fresh_root), serialized)
            self.assertEqual(
                report["failures"][0]["message"],
                "artifact-referenced provenance path cannot be resolved",
            )

    def test_fresh_root_creation_failure_has_stable_public_error(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            blocked = Path(temporary) / "private-parent"
            blocked.write_bytes(b"not a directory")
            fresh_root = blocked / "fresh-root"
            redactor = verifier.SensitivePathRedactor()
            redactor.add_root(fresh_root, "<fresh-root>")
            with self.assertRaises(verifier.VerificationError) as caught:
                verifier.prepare_fresh_roots(fresh_root)
            report = verifier.failure_report(caught.exception, redactor=redactor)
            failure = report["failures"][0]
            self.assertEqual(failure["code"], "fresh_root_create_failed")
            self.assertEqual(failure["locator"], "fresh_root")
            self.assertNotIn(str(fresh_root), json.dumps(report))
            self.assertEqual(
                failure["message"],
                "fresh root or one of its isolated children cannot be created",
            )

    def test_cache_object_oserror_is_generic_and_redacted(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            prefix = Path(temporary) / "private-build"
            local_cache = prefix / "local-cache"
            global_cache = prefix / "global-cache"
            local_cache.mkdir(parents=True)
            global_cache.mkdir(parents=True)
            dynamic = core.parse_dynamic_artifact(
                dynamic_fixture(cache_prefix=str(prefix))
            )
            static = core.parse_static_archive(
                archive_fixture(cache_prefix=str(prefix))
            )
            redactor = verifier.SensitivePathRedactor()
            redactor.add_root(local_cache, "<frozen-local-cache>")
            redactor.add_root(global_cache, "<frozen-global-cache>")
            with self.assertRaises(verifier.VerificationError) as caught:
                verifier.bind_artifact_provenance(
                    core,
                    dynamic,
                    static,
                    local_cache=local_cache,
                    global_cache=global_cache,
                    expected_dynamic_sha="0" * 64,
                    expected_static_sha="0" * 64,
                    label="frozen",
                )
            report = verifier.failure_report(caught.exception, redactor=redactor)
            self.assertEqual(report["failures"][0]["code"], "provenance_path_missing")
            serialized = json.dumps(report)
            self.assertNotIn(str(local_cache), serialized)
            self.assertNotIn(str(global_cache), serialized)

    def test_receipt_roots_are_registered_without_entering_failure_json(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            replacements = {
                "<temporary-install-prefix>": root / "private-install",
                "<isolated-local-cache>": root / "private-local-cache",
                "<isolated-global-cache>": root / "private-global-cache",
            }
            for path in replacements.values():
                path.mkdir()
            configuration = policy_fixture()["artifact_build_configuration"]
            argv = [
                str(replacements[item]) if item in replacements else item
                for item in configuration["command_template"]
            ]
            receipt = {
                "configuration": configuration,
                "execution": {"argv": argv},
            }
            redactor = verifier.SensitivePathRedactor()
            verifier.receipt_cache_roots(receipt, redactor=redactor)
            report = verifier.failure_report(
                RuntimeError(" ".join(str(path) for path in replacements.values())),
                redactor=redactor,
            )
            serialized = json.dumps(report)
            for path in replacements.values():
                self.assertNotIn(str(path), serialized)
            for placeholder in (
                "<frozen-install-root>",
                "<frozen-local-cache>",
                "<frozen-global-cache>",
            ):
                self.assertIn(placeholder, serialized)

    def test_snapshot_private_path_is_redacted_from_message_and_locator(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            public = Path(temporary) / "public"
            public.write_bytes(b"A")
            with artifacts.ArtifactSnapshotSet.capture(
                [artifacts.ArtifactRequest.library("library", public)],
                private_parent=temporary,
            ) as snapshot_set:
                redactor = verifier.SensitivePathRedactor()
                private = redactor.add_snapshot_artifact(snapshot_set.artifacts[0])
                exc = verifier.VerificationError(
                    "snapshot_private_fixture",
                    f"private failure at {private}",
                    locator=private,
                )
                report = verifier.failure_report(exc, redactor=redactor)
                serialized = json.dumps(report)
                self.assertNotIn(str(Path(private).parent), serialized)
                self.assertIn("<private-artifact-root>", serialized)
                self.assertEqual(
                    report["failures"][0]["code"], "snapshot_private_fixture"
                )

    def test_main_redacts_unexpected_exception_before_output(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fresh_root = Path(temporary) / "generated-fresh"
            args = types.SimpleNamespace(
                fresh_root=fresh_root,
                output=Path(temporary) / "report.json",
            )
            written: list[dict] = []

            def capture_report(_path, report, *, redactor):
                written.append(redactor.sanitize_value(report))

            with (
                mock.patch.object(verifier, "parse_args", return_value=args),
                mock.patch.object(
                    verifier,
                    "verify",
                    side_effect=RuntimeError(f"unexpected {fresh_root}/install"),
                ),
                mock.patch.object(verifier, "write_report", side_effect=capture_report),
            ):
                self.assertEqual(verifier.main([]), 1)
            self.assertEqual(
                written[0]["failures"][0]["code"], "verification_exception"
            )
            serialized = json.dumps(written[0])
            self.assertNotIn(str(fresh_root), serialized)
            self.assertIn("<fresh-root>/install", serialized)

    def test_report_write_stderr_is_redacted(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fresh_root = Path(temporary) / "generated-fresh"
            args = types.SimpleNamespace(
                fresh_root=fresh_root,
                output=Path(temporary) / "report.json",
            )
            write_error = verifier.VerificationError(
                "report_write_failed",
                f"could not publish {fresh_root}/private-report",
                locator=str(fresh_root),
            )
            stderr = io.StringIO()
            with (
                mock.patch.object(verifier, "parse_args", return_value=args),
                mock.patch.object(
                    verifier,
                    "verify",
                    return_value={"schema": {"name": "fixture", "version": 1}},
                ),
                mock.patch.object(verifier, "write_report", side_effect=write_error),
                mock.patch.object(sys, "stderr", stderr),
            ):
                self.assertEqual(verifier.main([]), 1)
            emitted = stderr.getvalue()
            self.assertNotIn(str(fresh_root), emitted)
            self.assertIn("<fresh-root>", emitted)
            self.assertEqual(
                json.loads(emitted)["failures"][0]["code"], "report_write_failed"
            )

    def test_write_report_applies_final_sensitive_path_sanitizer(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fresh_root = root / "generated-fresh"
            output = root / "report.json"
            redactor = verifier.SensitivePathRedactor()
            redactor.add_root(fresh_root, "<fresh-root>")
            verifier.write_report(
                output,
                {
                    "schema": {"name": "fixture", "version": 1},
                    "detail": f"failed at {fresh_root}/install/private",
                },
                redactor=redactor,
            )
            raw = output.read_text(encoding="ascii")
            self.assertNotIn(str(fresh_root), raw)
            self.assertEqual(
                json.loads(raw)["detail"],
                "failed at <fresh-root>/install/private",
            )

    def test_exact_regular_reader_rejects_unsafe_or_unstable_files(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            regular = root / "regular"
            regular.write_bytes(b"abc")
            captured = verifier.read_exact_regular_file(
                regular, max_bytes=3, locator="fixture"
            )
            self.assertEqual(captured.data, b"abc")
            self.assertEqual(captured.sha256, hashlib.sha256(b"abc").hexdigest())

            symlink = root / "symlink"
            symlink.symlink_to(regular)
            fifo = root / "fifo"
            os.mkfifo(fifo)
            for label, path, limit, code in (
                ("symlink", symlink, 3, "file_not_regular"),
                ("special", fifo, 3, "file_not_regular"),
                ("oversize", regular, 2, "file_size_out_of_range"),
            ):
                with self.subTest(label=label):
                    with self.assertRaises(verifier.VerificationError) as caught:
                        verifier.read_exact_regular_file(
                            path, max_bytes=limit, locator="fixture"
                        )
                    self.assertEqual(caught.exception.code, code)

            with mock.patch.object(verifier.os, "read", return_value=b""):
                with self.assertRaises(verifier.VerificationError) as caught:
                    verifier.read_exact_regular_file(
                        regular, max_bytes=3, locator="fixture"
                    )
            self.assertEqual(caught.exception.code, "file_short_read")

            with mock.patch.object(verifier.os, "read", side_effect=(b"abc", b"x")):
                with self.assertRaises(verifier.VerificationError) as caught:
                    verifier.read_exact_regular_file(
                        regular, max_bytes=3, locator="fixture"
                    )
            self.assertEqual(caught.exception.code, "file_grew_during_read")

            replacement = root / "replacement"
            replacement.write_bytes(b"xyz")
            actual_read = os.read
            replaced = False

            def replace_after_read(descriptor: int, size: int) -> bytes:
                nonlocal replaced
                value = actual_read(descriptor, size)
                if not replaced:
                    os.replace(replacement, regular)
                    replaced = True
                return value

            with mock.patch.object(verifier.os, "read", side_effect=replace_after_read):
                with self.assertRaises(verifier.VerificationError) as caught:
                    verifier.read_exact_regular_file(
                        regular, max_bytes=3, locator="fixture"
                    )
            self.assertEqual(caught.exception.code, "file_changed_during_read")

    def test_json_reader_rejects_duplicate_nonobject_and_nonfinite_values(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "input.json"
            for label, raw, code in (
                ("duplicate", b'{"x":1,"x":2}', "duplicate_json_key"),
                ("nonobject", b"[]", "invalid_json_type"),
                ("nan", b'{"x":NaN}', "nonfinite_json_number"),
                ("positive_infinity", b'{"x":Infinity}', "nonfinite_json_number"),
                ("negative_infinity", b'{"x":-Infinity}', "nonfinite_json_number"),
            ):
                path.write_bytes(raw)
                with self.subTest(label=label):
                    with self.assertRaises(verifier.VerificationError) as caught:
                        verifier.load_json(path, locator="fixture")
                    self.assertEqual(caught.exception.code, code)

    def test_frozen_and_fresh_snapshot_pairs_remain_bound_to_a(self) -> None:
        for label in ("frozen", "fresh"):
            with self.subTest(label=label), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                dynamic_path = root / "libdynamic"
                static_path = root / "libstatic"
                dynamic_path.write_bytes(b"dynamic-A")
                static_path.write_bytes(b"static-A")
                with artifacts.ArtifactSnapshotSet.capture(
                    [
                        artifacts.ArtifactRequest.library(
                            f"{label}-dynamic", dynamic_path
                        ),
                        artifacts.ArtifactRequest.library(
                            f"{label}-static", static_path
                        ),
                    ],
                    private_parent=root,
                ) as snapshot_set:
                    dynamic_snapshot, static_snapshot = snapshot_set.artifacts
                    dynamic_path.write_bytes(b"dynamic-B")
                    static_path.write_bytes(b"static-B")
                    self.assertEqual(
                        verifier.snapshot_artifact_bytes(
                            dynamic_snapshot, locator=f"{label}.dynamic"
                        ).data,
                        b"dynamic-A",
                    )
                    self.assertEqual(
                        verifier.snapshot_artifact_bytes(
                            static_snapshot, locator=f"{label}.static"
                        ).data,
                        b"static-A",
                    )
                    snapshot_set.finalize()

    def test_validate_frozen_artifact_uses_captured_metadata(self) -> None:
        policy, observation, receipt = control_fixture()
        del policy
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "libdynamic"
            path.write_bytes(b"A")
            digest = hashlib.sha256(b"A").hexdigest()
            receipt["artifacts"] = {
                "dynamic_library": {"path": str(path), "size": 1, "sha256": digest}
            }
            observation["artifacts"] = {
                "dynamic": {"status": "observed", "sha256": digest}
            }
            with artifacts.ArtifactSnapshotSet.capture(
                [artifacts.ArtifactRequest.library("dynamic", path)],
                private_parent=temporary,
            ) as snapshot_set:
                path.write_bytes(b"B")
                self.assertEqual(
                    verifier.validate_frozen_artifact(
                        snapshot_set.artifacts[0],
                        kind="dynamic",
                        observation=observation,
                        receipt=receipt,
                    ),
                    {"size": 1, "sha256": digest},
                )

    def test_system_tools_receive_only_reverified_private_paths(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            public_dynamic = root / "public-dynamic"
            public_static = root / "public-static"
            public_dynamic.write_bytes(dynamic_fixture())
            public_static.write_bytes(archive_fixture())
            with artifacts.ArtifactSnapshotSet.capture(
                [
                    artifacts.ArtifactRequest.library("dynamic", public_dynamic),
                    artifacts.ArtifactRequest.library("static", public_static),
                ],
                private_parent=root,
            ) as snapshot_set:
                dynamic_snapshot, static_snapshot = snapshot_set.artifacts
                dynamic = core.parse_dynamic_artifact(
                    verifier.snapshot_artifact_bytes(
                        dynamic_snapshot, locator="dynamic"
                    ).data
                )
                static = core.parse_static_archive(
                    verifier.snapshot_artifact_bytes(
                        static_snapshot, locator="static"
                    ).data
                )
                seen: list[str] = []

                def runner(argv, **_kwargs):
                    seen.append(argv[-1])
                    self.assertNotIn(
                        argv[-1], {str(public_dynamic), str(public_static)}
                    )
                    return verifier.CommandResult(
                        tuple(argv), 0, b"", b"", False, False, None, 1
                    )

                with self.assertRaises(verifier.VerificationError):
                    verifier.system_cross_checks(
                        dynamic_snapshot,
                        static_snapshot,
                        dynamic,
                        static,
                        root=ROOT,
                        timeout=1,
                        runner=runner,
                    )
                self.assertEqual(len(seen), 2)
                self.assertTrue(all(str(root) in item for item in seen))

                def failing_runner(argv, **_kwargs):
                    return verifier.CommandResult(
                        tuple(argv),
                        7,
                        b"",
                        (argv[-1] + ": failed").encode(),
                        False,
                        False,
                        None,
                        1,
                    )

                with self.assertRaises(verifier.VerificationError) as caught:
                    verifier.system_cross_checks(
                        dynamic_snapshot,
                        static_snapshot,
                        dynamic,
                        static,
                        root=ROOT,
                        timeout=1,
                        runner=failing_runner,
                    )
                private_root = str(Path(dynamic_snapshot.execution_path).parent)
                self.assertNotIn(private_root, str(caught.exception))
                self.assertNotIn(
                    private_root, json.dumps(verifier.failure_report(caught.exception))
                )

    def test_finalize_cleanup_drift_and_private_path_redaction_fail_closed(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            public = Path(temporary) / "public"
            public.write_bytes(b"A")
            snapshot_set = artifacts.ArtifactSnapshotSet.capture(
                [artifacts.ArtifactRequest.library("library", public)],
                private_parent=temporary,
            )
            private = Path(snapshot_set.artifacts[0].execution_path)
            replacement = private.with_name("replacement")
            replacement.write_bytes(b"B")
            replacement.chmod(0o400)
            os.replace(replacement, private)
            with self.assertRaises(artifacts.ArtifactVerificationError):
                snapshot_set.finalize()
            with self.assertRaises(artifacts.ArtifactCleanupError) as caught:
                snapshot_set.close()
            safe = verifier._snapshot_failure(caught.exception)
            self.assertNotIn(str(private), str(safe))
            self.assertNotIn(
                str(private.parent), json.dumps(verifier.failure_report(safe))
            )

    def test_provenance_object_and_builtins_use_exact_reader(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            prefix = Path(temporary)
            local_cache = prefix / "local-cache"
            global_cache = prefix / "global-cache"
            object_path = local_cache / "o" / ("1" * 32) / "zynum.o"
            object_path.parent.mkdir(parents=True)
            object_path.write_bytes(object_fixture(cache_prefix=str(prefix)))
            os.utime(object_path, ns=(100_000_000_000, 100_000_000_000))
            builtin = global_cache / "b" / ("3" * 32) / "builtin.zig"
            builtin.parent.mkdir(parents=True)
            builtin.write_bytes(b"builtin")
            dynamic = core.parse_dynamic_artifact(
                dynamic_fixture(cache_prefix=str(prefix))
            )
            static = core.parse_static_archive(
                archive_fixture(cache_prefix=str(prefix))
            )
            with mock.patch.object(
                verifier,
                "read_exact_regular_file",
                wraps=verifier.read_exact_regular_file,
            ) as exact_read:
                result = verifier.bind_artifact_provenance(
                    core,
                    dynamic,
                    static,
                    local_cache=local_cache,
                    global_cache=global_cache,
                    expected_dynamic_sha=hashlib.sha256(b"builtin").hexdigest(),
                    expected_static_sha=hashlib.sha256(b"builtin").hexdigest(),
                    label="fixture",
                )
            self.assertEqual(result["verdict"], "pass")
            self.assertEqual(exact_read.call_count, 3)

    def test_control_documents_require_exact_configuration(self) -> None:
        policy, observation, receipt = control_fixture()
        self.assertEqual(
            verifier.validate_control_documents(policy, observation, receipt)[
                "configuration_id"
            ],
            "fixture-v1",
        )
        for label, mutation in (
            ("config", lambda: receipt["configuration"].update(configuration_id="old")),
            ("schema", lambda: observation["schema"].update(version=2)),
            (
                "static_old_hash",
                lambda: receipt["configuration"]["resolved_build_inputs"][
                    "generated_builtin_zig_sha256"
                ].update(static_library="c" * 64),
            ),
        ):
            policy, observation, receipt = control_fixture()
            mutation()
            with self.subTest(label=label):
                with self.assertRaises(verifier.VerificationError) as caught:
                    verifier.validate_control_documents(policy, observation, receipt)
                self.assertIn(
                    caught.exception.code,
                    {"receipt_configuration_mismatch", "observation_schema_mismatch"},
                )

    def test_fresh_root_nonempty_and_placeholder_validation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "fresh"
            root.mkdir()
            (root / "owned-by-someone-else").write_text("x")
            with self.assertRaises(verifier.VerificationError) as caught:
                verifier.prepare_fresh_roots(root)
            self.assertEqual(caught.exception.code, "fresh_root_not_empty")
        configuration = policy_fixture()["artifact_build_configuration"]
        roots = {
            placeholder: Path("fixture") / directory
            for placeholder, directory in verifier.PLACEHOLDERS.items()
        }
        command, template = verifier.materialize_build_command(configuration, roots)
        self.assertEqual(template, configuration["command_template"])
        self.assertFalse(any(argument in verifier.PLACEHOLDERS for argument in command))
        bad = copy.deepcopy(configuration)
        bad["command_template"].remove("<isolated-global-cache>")
        with self.assertRaises(verifier.VerificationError) as caught:
            verifier.materialize_build_command(bad, {})
        self.assertEqual(caught.exception.code, "invalid_command_placeholders")

    def test_build_failure_timeout_truncation_and_cleanup_are_distinct(self) -> None:
        base = dict(
            argv=("tool",),
            returncode=0,
            stdout=b"",
            stderr=b"",
            timed_out=False,
            truncated=False,
            cleanup_failure=None,
            duration_ms=1,
        )
        cases = {
            "command_failed": {"returncode": 7},
            "command_timed_out": {"timed_out": True},
            "command_output_truncated": {"truncated": True},
            "process_cleanup_failed": {"cleanup_failure": "survived"},
        }
        for code, changes in cases.items():
            with self.subTest(code=code):
                result = verifier.CommandResult(**{**base, **changes})
                with self.assertRaises(verifier.VerificationError) as caught:
                    verifier.require_command(result, locator="fixture")
                self.assertEqual(caught.exception.code, code)

    def test_run_bounded_marks_real_truncation(self) -> None:
        result = verifier.run_bounded(
            (sys.executable, "-c", "import os; os.write(1, b'x' * 4096)"),
            cwd=ROOT,
            timeout=5,
            max_output=32,
        )
        self.assertTrue(result.truncated)
        self.assertLessEqual(len(result.stdout), 32)

    def test_source_names_and_exact_underscore_accounting(self) -> None:
        _, observation, _ = control_fixture()
        public, hidden = verifier.source_names_from_observation(observation)
        artifact = core.parse_macho_object(object_fixture())
        result = verifier.source_accounting(artifact.symbols, public, hidden)
        self.assertEqual(result["verdict"], "pass")
        self.assertEqual(result["public"]["matched"], ["foo"])
        self.assertEqual(result["hidden"]["matched"], ["hidden"])
        self.assertIsNone(verifier._normalize_exact_symbol("__foo", {"foo"}))

    def test_three_symbol_axes_order_multiset_and_types(self) -> None:
        left = core.parse_macho_object(object_fixture())
        right = core.parse_macho_object(object_fixture(cache_prefix="/longer-root"))
        result = verifier.compare_symbol_axes(
            verifier.symbol_axis(left.symbols), verifier.symbol_axis(right.symbols)
        )
        self.assertEqual(result["verdict"], "pass")
        altered = core.parse_macho_object(
            object_fixture(local_type=0x0F, external_type=0x0E)
        )
        with self.assertRaises(verifier.VerificationError) as caught:
            verifier.compare_symbol_axes(
                verifier.symbol_axis(left.symbols),
                verifier.symbol_axis(altered.symbols),
            )
        self.assertEqual(caught.exception.code, "symbol_axis_mismatch")

    def test_cache_presence_without_artifact_reference_cannot_bind(self) -> None:
        dynamic = core.parse_dynamic_artifact(
            dynamic_fixture(cache_prefix="/elsewhere")
        )
        static = core.parse_static_archive(archive_fixture(cache_prefix="/elsewhere"))
        with tempfile.TemporaryDirectory() as temporary:
            cache = Path(temporary)
            (cache / "some" / "global-cache" / "b" / ("3" * 32)).mkdir(parents=True)
            (
                cache / "some" / "global-cache" / "b" / ("3" * 32) / "builtin.zig"
            ).write_bytes(b"present")
            with self.assertRaises(verifier.VerificationError) as caught:
                verifier.bind_artifact_provenance(
                    core,
                    dynamic,
                    static,
                    local_cache=cache / "local-cache",
                    global_cache=cache / "global-cache",
                    expected_dynamic_sha="a" * 64,
                    expected_static_sha="b" * 64,
                    label="fixture",
                )
            self.assertIn(
                caught.exception.code,
                {"provenance_path_missing", "provenance_outside_cache"},
            )

    def test_otool_parser_and_contradiction(self) -> None:
        loads = """fixture:\nLoad command 0\n          cmd LC_BUILD_VERSION\n      cmdsize 24\n     platform MACOS\n        minos 13.0\n          sdk 26.4\n       ntools 0\nLoad command 1\n          cmd LC_ID_DYLIB\n      cmdsize 56\n         name @rpath/libzynum_blas.dylib (offset 24)\nLoad command 2\n          cmd LC_LOAD_DYLIB\n      cmdsize 56\n         name /usr/lib/libSystem.B.dylib (offset 24)\nLoad command 3\n          cmd LC_RPATH\n      cmdsize 32\n         path @loader_path (offset 12)\n"""
        libraries = """fixture:\n\t@rpath/libzynum_blas.dylib (compatibility version 1.0.0, current version 1.0.0)\n\t/usr/lib/libSystem.B.dylib (compatibility version 1.0.0, current version 1.0.0)\n"""
        parsed = verifier.parse_otool(loads, libraries)
        self.assertEqual(parsed["sdk"], "26.4")
        with self.assertRaises(verifier.VerificationError) as caught:
            verifier.parse_otool(loads, libraries.replace("libSystem", "libobjc"))
        self.assertEqual(caught.exception.code, "otool_libraries_mismatch")

    def test_nm_ar_and_codesign_parsers_reject_unknown_or_bad_values(self) -> None:
        nm = "00000000 (__TEXT,__text) external _foo\n         (undefined) external _bar\n00000000 (__TEXT,__text) non-external _local\n"
        axes = verifier.parse_nm_axes(nm)
        self.assertEqual([sum(value.values()) for value in axes.values()], [1, 1, 1])
        ar = "rw-r--r-- 0/0 123 Jan 1 00:00 1970 zcu.o\n"
        self.assertEqual(verifier.parse_ar_tv(ar)[0]["name"], "zcu.o")
        self.assertEqual(verifier.parse_ar_tv("not ar output"), [])

    def test_failure_report_is_canonical_and_machine_readable(self) -> None:
        report = verifier.failure_report(
            verifier.VerificationError("fixture", "bad", locator="x")
        )
        self.assertEqual(report["verdict"], "fail")
        self.assertEqual(report["failures"][0]["code"], "fixture")
        self.assertEqual(json.loads(verifier.canonical_json_bytes(report)), report)

    def test_usage_error_is_two(self) -> None:
        with mock.patch.object(sys, "stderr"):
            with self.assertRaises(SystemExit) as caught:
                verifier.parse_args([])
        self.assertEqual(caught.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
