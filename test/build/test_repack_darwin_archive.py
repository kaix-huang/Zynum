# Copyright (C) 2026 Zynum contributors
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Focused safety and Mach-O alignment checks for archive recreation."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import shutil
import stat
import subprocess
import tempfile
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("repack_darwin_archive", ROOT / "tools/repack_darwin_archive.py")
assert SPEC is not None and SPEC.loader is not None
repacker = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(repacker)


class RepackTests(unittest.TestCase):
    def test_member_validation(self) -> None:
        self.assertEqual(["second.o", "first member.o"], repacker.member_names("second.o\nfirst member.o\n"))
        self.assertEqual([], repacker.member_names(""))
        for listing in ("../x.o\n", "/x.o\n", "x/y.o\n", "x\\y.o\n", "-x.o\n", "@args\n", ".\n", "..\n", "\n", "a.o\na.o\n", "a\tb.o\n"):
            with self.subTest(listing=listing), self.assertRaises(ValueError):
                repacker.member_names(listing)

    def test_restores_owner_permissions_and_preserves_order(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "input.a"
            output = root / "output.a"
            source.write_bytes(b"input")
            output.write_bytes(b"old")
            names = ["z.o", "a member.o"]
            calls = []

            def run(command, **kwargs):
                calls.append(command)
                self.assertNotIn("shell", kwargs)
                if command[2] == "x":
                    for name in names:
                        member = kwargs["cwd"] / name
                        member.write_bytes(name.encode())
                        member.chmod(0)
                if command[2] == "--format=darwin":
                    self.assertEqual("rcs", command[3])
                    self.assertEqual(names, command[5:])
                    self.assertFalse(Path(command[4]).exists())
                    for name in names:
                        mode = (kwargs["cwd"] / name).stat().st_mode
                        self.assertEqual(stat.S_IRUSR | stat.S_IWUSR, stat.S_IMODE(mode))
                    Path(command[4]).write_bytes(b"darwin")
                return subprocess.CompletedProcess(command, 0, "\n".join(names) + "\n", "")

            with mock.patch.object(repacker.shutil, "which", return_value="/tool/zig"), mock.patch.object(repacker.subprocess, "run", side_effect=run):
                repacker.repack("zig", source, output)
            self.assertEqual(b"darwin", output.read_bytes())
            self.assertEqual(["t", "x", "--format=darwin", "t"], [call[2] for call in calls])
            self.assertEqual(["input.a", "output.a"], sorted(path.name for path in root.iterdir()))

    def test_failed_recreation_preserves_existing_output(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "input.a"
            output = Path(temporary) / "output.a"
            source.write_bytes(b"input")
            output.write_bytes(b"old")

            def run(command, **kwargs):
                if command[2] == "--format=darwin":
                    raise subprocess.CalledProcessError(1, command, stderr="creation failed")
                return subprocess.CompletedProcess(command, 0, "", "")

            with mock.patch.object(repacker.shutil, "which", return_value="/tool/zig"), mock.patch.object(repacker.subprocess, "run", side_effect=run):
                with self.assertRaises(subprocess.CalledProcessError):
                    repacker.repack("zig", source, output)
            self.assertEqual(b"old", output.read_bytes())

    @unittest.skipUnless(shutil.which("zig"), "Zig is required for the archive smoke check")
    def test_real_macho_members_are_eight_byte_aligned(self) -> None:
        zig = shutil.which("zig")
        assert zig is not None
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "tiny.c"
            source.write_text("int zynum_archive_smoke(void) { return 7; }\n")
            obj = root / "tiny.o"
            subprocess.run([zig, "cc", "-target", "aarch64-macos", "-c", str(source), "-o", str(obj)], check=True, capture_output=True)
            archive = root / "input.a"
            subprocess.run([zig, "ar", "--format=gnu", "rcs", str(archive), str(obj)], check=True, capture_output=True)
            output = root / "output.a"
            repacker.repack(zig, archive, output)
            data = output.read_bytes()
            self.assertEqual(b"!<arch>\n", data[:8])
            offset = 8
            object_count = 0
            while offset < len(data):
                header = data[offset:offset + 60]
                self.assertEqual(b"`\n", header[58:60])
                size = int(header[48:58])
                start = offset + 60
                name = header[:16].decode().strip()
                if name.startswith("#1/"):
                    start += int(name[3:])
                if data[start:start + 4] == b"\xcf\xfa\xed\xfe":
                    self.assertEqual(0, start % 8)
                    self.assertEqual(obj.read_bytes(), data[start:start + obj.stat().st_size])
                    object_count += 1
                offset += 60 + size + size % 2
            self.assertEqual(1, object_count)


if __name__ == "__main__":
    unittest.main()
