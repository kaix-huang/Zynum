# Copyright (C) 2026 Zynum contributors
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Contract tests for the public test inventory validator."""

from __future__ import annotations

import ast
import contextlib
import copy
import ctypes
import dataclasses
import errno
import hashlib
import importlib.util
import io
import json
import os
import re
import shlex
import shutil
import signal
import stat
import subprocess
import sys
import tempfile
import types
import unittest
from collections import Counter
from pathlib import Path
from typing import Any
from unittest import mock

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
CHECKER_PATH = REPOSITORY_ROOT / "tools/check_test_inventory.py"
SPEC = importlib.util.spec_from_file_location("check_test_inventory", CHECKER_PATH)
assert SPEC and SPEC.loader
CHECKER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CHECKER)


class TestInventoryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.inventory = json.loads(
            (REPOSITORY_ROOT / "tools/test_inventory.json").read_text(encoding="utf-8")
        )
        files = {
            "build.zig",
            ".github/workflows/ci.yml",
            ".github/workflows/release.yml",
            "tools/build_inventory.json",
            "tools/check_build_inventory.py",
            "tools/check_test_inventory.py",
            "tools/repository_git.py",
            "tools/repository_snapshot.py",
            "tools/test_inventory.json",
            "tools/test_inventory_runner.zig",
        }
        files.update(row["path"] for row in cls.inventory["zig_test_files"])
        files.update(row["path"] for row in cls.inventory["python_test_modules"])
        build_inventory = json.loads(
            (REPOSITORY_ROOT / "tools/build_inventory.json").read_text(encoding="utf-8")
        )
        files.update(
            row["path"]
            for row in build_inventory["repository_file_classifications"]
            if row["kind"] in {"python-source", "zig-source"}
        )
        cls.fixture_files = sorted(files)

    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(prefix="test-inventory-fixture-")
        self.root = Path(self.temporary.name)
        for relative in self.fixture_files:
            source = REPOSITORY_ROOT / relative
            destination = self.root / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
        self.inventory_path = self.root / "tools/test_inventory.json"
        self.synthetic_module_names: list[str] = []
        self.synthetic_module_stack = contextlib.ExitStack()
        self.synthetic_source_directory = tempfile.TemporaryDirectory(
            prefix="test-inventory-synthetic-modules-"
        )

    def tearDown(self) -> None:
        self.synthetic_module_stack.close()
        self.synthetic_source_directory.cleanup()
        self.temporary.cleanup()

    def _synthetic_module_binding(self, values: dict[str, Any]) -> Any:
        name = f"_zynum_test_inventory_fixture_{len(self.synthetic_module_names)}"
        source_path = Path(self.synthetic_source_directory.name) / f"{name}.py"
        source_path.write_bytes(b"")
        reviewed = CHECKER._PythonReviewedSourceModule(
            f"synthetic/{name}.py",
            name,
            source_path,
            b"",
            hashlib.sha256(b"").hexdigest(),
        )
        registry = self.synthetic_module_stack.enter_context(
            CHECKER._registered_python_tooling_modules((reviewed,))
        )
        binding = registry[0]
        binding.namespace.update(values)
        self.synthetic_module_names.append(name)
        return binding

    def _write(self, inventory: dict[str, Any]) -> None:
        self.inventory_path.write_text(
            json.dumps(inventory, indent=2) + "\n", encoding="utf-8"
        )

    def _set_fixture_runner_digest_slots(
        self, *, current: str | None = None, next_digest: str | None = None
    ) -> None:
        runner = self.root / "tools/test_inventory_runner.zig"
        text = runner.read_text(encoding="utf-8")
        current_value = (
            CHECKER.CURRENT_TEST_INVENTORY_SHA256 if current is None else current
        )
        next_value = "null" if next_digest is None else f'"{next_digest}"'
        text, current_count = re.subn(
            r'^const CURRENT_TEST_INVENTORY_SHA256: \[\]const u8 = "[0-9a-f]{64}";$',
            f'const CURRENT_TEST_INVENTORY_SHA256: []const u8 = "{current_value}";',
            text,
            count=1,
            flags=re.MULTILINE,
        )
        text, next_count = re.subn(
            r"^const NEXT_TEST_INVENTORY_SHA256: \?\[\]const u8 = "
            r'(?:null|"[0-9a-f]{64}");$',
            f"const NEXT_TEST_INVENTORY_SHA256: ?[]const u8 = {next_value};",
            text,
            count=1,
            flags=re.MULTILINE,
        )
        self.assertEqual((1, 1), (current_count, next_count))
        runner.write_text(text, encoding="utf-8")

    @contextlib.contextmanager
    def _reviewed_digest_slots(
        self,
        *,
        inventory_current: str | None = None,
        inventory_next: str | None = None,
        native_current: str | None = None,
        native_next: str | None = None,
    ) -> Any:
        runner = self.root / "tools/test_inventory_runner.zig"
        original = runner.read_bytes()
        effective_inventory_current = (
            CHECKER.CURRENT_TEST_INVENTORY_SHA256
            if inventory_current is None
            else inventory_current
        )
        effective_native_current = (
            CHECKER.CURRENT_NATIVE_PROJECTION_SHA256
            if native_current is None
            else native_current
        )
        self._set_fixture_runner_digest_slots(
            current=effective_inventory_current, next_digest=inventory_next
        )
        try:
            with (
                mock.patch.object(
                    CHECKER,
                    "CURRENT_TEST_INVENTORY_SHA256",
                    effective_inventory_current,
                ),
                mock.patch.object(
                    CHECKER, "NEXT_TEST_INVENTORY_SHA256", inventory_next
                ),
                mock.patch.object(
                    CHECKER,
                    "CURRENT_NATIVE_PROJECTION_SHA256",
                    effective_native_current,
                ),
                mock.patch.object(
                    CHECKER, "NEXT_NATIVE_PROJECTION_SHA256", native_next
                ),
            ):
                yield
        finally:
            runner.write_bytes(original)

    def _runner_digest_slots(self, runner: Path) -> tuple[str, str | None]:
        text = runner.read_text(encoding="utf-8")
        current_matches = re.findall(
            r'^const CURRENT_TEST_INVENTORY_SHA256: \[\]const u8 = "([0-9a-f]{64})";$',
            text,
            flags=re.MULTILINE,
        )
        next_matches = re.findall(
            r"^const NEXT_TEST_INVENTORY_SHA256: \?\[\]const u8 = "
            r'(null|"[0-9a-f]{64}");$',
            text,
            flags=re.MULTILINE,
        )
        self.assertEqual(1, len(current_matches))
        self.assertEqual(1, len(next_matches))
        runner_next = None if next_matches[0] == "null" else next_matches[0].strip('"')
        return current_matches[0], runner_next

    def _assert_reviewed_fixture_generation(
        self, inventory_path: Path, runner: Path
    ) -> str:
        current = CHECKER.CURRENT_TEST_INVENTORY_SHA256
        next_digest = CHECKER.NEXT_TEST_INVENTORY_SHA256
        self.assertRegex(current, r"\A[0-9a-f]{64}\Z")
        reviewed = {current}
        if next_digest is not None:
            self.assertRegex(next_digest, r"\A[0-9a-f]{64}\Z")
            self.assertNotEqual(current, next_digest)
            reviewed.add(next_digest)
        self.assertEqual((current, next_digest), self._runner_digest_slots(runner))
        generation = hashlib.sha256(inventory_path.read_bytes()).hexdigest()
        self.assertIn(generation, reviewed)
        return generation

    @contextlib.contextmanager
    def _fixture_generation_as_current(
        self,
        *,
        inventory_next: str | None = None,
        native_current: str | None = None,
        native_next: str | None = None,
    ) -> Any:
        generation = self._assert_reviewed_fixture_generation(
            self.inventory_path, self.root / "tools/test_inventory_runner.zig"
        )
        with self._reviewed_digest_slots(
            inventory_current=generation,
            inventory_next=inventory_next,
            native_current=native_current,
            native_next=native_next,
        ):
            yield generation

    def _errors(self) -> str:
        inventory_digest = hashlib.sha256(self.inventory_path.read_bytes()).hexdigest()
        if inventory_digest == CHECKER.CURRENT_TEST_INVENTORY_SHA256:
            errors = CHECKER.validate(
                self.root, self.inventory_path, structure_only=True
            )
        else:
            with self._reviewed_digest_slots(inventory_next=inventory_digest):
                errors = CHECKER.validate(
                    self.root, self.inventory_path, structure_only=True
                )
        self.assertTrue(errors, "mutation unexpectedly validated")
        return "\n".join(errors)

    @contextlib.contextmanager
    def _review_current_python_tooling_sources(self) -> Any:
        def refreshed_manifest(
            manifest: tuple[tuple[str, str], ...],
        ) -> tuple[tuple[str, str], ...]:
            return tuple(
                (
                    path,
                    hashlib.sha256((self.root / path).read_bytes()).hexdigest(),
                )
                for path, _ in manifest
            )

        reviewed = refreshed_manifest(CHECKER._PYTHON_TOOLING_REVIEWED_SOURCE_SHA256)
        execution = refreshed_manifest(CHECKER._PYTHON_TOOLING_EXECUTION_SOURCE_SHA256)
        manifest_digest = hashlib.sha256(
            json.dumps(
                (execution, CHECKER._PYTHON_TOOLING_EXECUTION_MODULES),
                separators=(",", ":"),
                ensure_ascii=True,
            ).encode("ascii")
        ).hexdigest()
        with (
            mock.patch.object(
                CHECKER, "_PYTHON_TOOLING_REVIEWED_SOURCE_SHA256", reviewed
            ),
            mock.patch.object(
                CHECKER, "_PYTHON_TOOLING_EXECUTION_SOURCE_SHA256", execution
            ),
            mock.patch.object(
                CHECKER,
                "_PYTHON_TOOLING_EXECUTION_MANIFEST_SHA256",
                manifest_digest,
            ),
        ):
            yield

    def _append_python_test_declaration_drift(self) -> tuple[str, str]:
        path = "test/build/test_test_inventory.py"
        class_name = "RefreshDriftFixtureTests"
        method_name = "test_added_after_inventory_snapshot"
        source = self.root / path
        source.write_text(
            source.read_text(encoding="utf-8")
            + (
                f"\n\nclass {class_name}(unittest.TestCase):\n"
                f"    def {method_name}(self) -> None:\n"
                "        self.assertTrue(True)\n"
            ),
            encoding="utf-8",
        )
        return path, f"{class_name}.{method_name}"

    def _native_refresh_protocol(
        self, *, pending: bool, candidate_ordinal: int = 0
    ) -> tuple[str, Path]:
        def encode(tag: str, value: str) -> str:
            payload = value.encode("utf-8")
            return f"{tag}:{len(payload)}:{payload.hex()}"

        expectation_state = CHECKER.PENDING_STATE if pending else CHECKER.FROZEN_STATE
        candidate_rows = [
            row
            for row in self.inventory["test_mode_rows"]
            if row["expectation_state"] == expectation_state
            and row["root_id"].startswith("zig-root:")
        ]
        selected_row = candidate_rows[candidate_ordinal]
        reference_row = (
            selected_row
            if selected_row["expected_test_set_id"] is not None
            else next(
                row
                for row in self.inventory["test_mode_rows"]
                if row["root_id"] == selected_row["root_id"]
                and row["optimize_mode_id"] == selected_row["optimize_mode_id"]
                and row["expected_test_set_id"] is not None
            )
        )
        reference_set = next(
            row
            for row in self.inventory["expected_test_sets"]
            if row["id"] == reference_row["expected_test_set_id"]
        )
        protocol_lines = [
            "ZYNUM-TEST-INVENTORY-V2",
            f"mode:{selected_row['expected_actual_module_optimize']}",
            encode("root", selected_row["root_id"]),
            encode("class", selected_row["enumeration_class_id"]),
            f"count:{len(reference_set['tests'])}",
            *(
                f"test:{ordinal}:" + encode("test", test["name"]).partition(":")[2]
                for ordinal, test in enumerate(reference_set["tests"])
            ),
        ]
        protocol = self.root / f"refresh-protocol-{candidate_ordinal}.log"
        protocol.write_text("\n".join(protocol_lines) + "\n", encoding="utf-8")
        return selected_row["environment_id"], protocol

    def _refresh_arguments(self, environment_id: str, protocol: Path) -> list[str]:
        return [
            "--root",
            str(self.root),
            "--inventory",
            str(self.inventory_path),
            "--structure-only",
            "--refresh-from-protocol",
            "--protocol-log",
            f"{environment_id}={protocol}",
        ]

    def _source_current_fixture_inventory(self) -> dict[str, Any]:
        environment_id, protocol = self._native_refresh_protocol(pending=False)
        with self._fixture_generation_as_current() as generation:
            try:
                candidate = CHECKER.refresh_from_protocol(
                    self.root,
                    self.inventory_path,
                    [(environment_id, protocol)],
                )
            except CHECKER.InventoryError as exc:
                digest_match = re.search(
                    r"reviewed whole-file test inventory mismatch: observed "
                    r"sha256=([0-9a-f]{64})",
                    str(exc),
                )
                if digest_match is None:
                    raise
                with self._reviewed_digest_slots(
                    inventory_current=generation,
                    inventory_next=digest_match.group(1),
                ):
                    candidate = CHECKER.refresh_from_protocol(
                        self.root,
                        self.inventory_path,
                        [(environment_id, protocol)],
                    )
        self.inventory_path.write_bytes(candidate.bytes)
        return candidate.inventory

    def _freeze_pending_native_rows(
        self,
        inventory: dict[str, Any],
        *,
        row_ids: set[str] | None = None,
    ) -> dict[str, Any]:
        candidate = copy.deepcopy(inventory)
        reference_sets: dict[tuple[str, str], str] = {}
        for row in candidate["test_mode_rows"]:
            if (
                row["root_id"].startswith("zig-root:")
                and row["expectation_state"] == CHECKER.FROZEN_STATE
            ):
                reference_sets.setdefault(
                    (row["root_id"], row["optimize_mode_id"]),
                    row["expected_test_set_id"],
                )
        bindings_by_row = {
            row["row_id"]: row for row in candidate["native_observation_bindings"]
        }
        for row in candidate["test_mode_rows"]:
            if (
                row["expectation_state"] != CHECKER.PENDING_STATE
                or not row["root_id"].startswith("zig-root:")
                or (row_ids is not None and row["id"] not in row_ids)
            ):
                continue
            set_id = reference_sets[(row["root_id"], row["optimize_mode_id"])]
            row["expected_test_set_id"] = set_id
            row["expectation_state"] = CHECKER.FROZEN_STATE
            bindings_by_row[row["id"]] = CHECKER._native_observation_binding(
                row, set_id
            )
        candidate["native_observation_bindings"] = sorted(
            bindings_by_row.values(), key=lambda row: row["id"]
        )
        CHECKER._refresh_native_gaps(candidate)
        candidate["strict_summary"] = CHECKER._section_summary(candidate)
        return candidate

    def _rename_frozen_native_test_coherently(
        self, inventory: dict[str, Any]
    ) -> dict[str, Any]:
        candidate = copy.deepcopy(inventory)
        binding = candidate["native_observation_bindings"][0]
        old_set_id = binding["expected_test_set_id"]
        expected_set = next(
            row for row in candidate["expected_test_sets"] if row["id"] == old_set_id
        )
        names = [row["name"] for row in expected_set["tests"]]
        names[0] = "reviewed-projection-forgery." + names[0]
        tests = CHECKER._expected_test_rows(expected_set["root_id"], names)
        expected_set["tests"] = tests
        expected_set["count"] = len(tests)
        expected_set["digest"] = CHECKER._fact_digest(tests)
        expected_set["id"] = CHECKER._content_set_id(expected_set["root_id"], tests)
        new_set_id = expected_set["id"]
        rows_by_id = {row["id"]: row for row in candidate["test_mode_rows"]}
        for row in candidate["test_mode_rows"]:
            if row["expected_test_set_id"] == old_set_id:
                row["expected_test_set_id"] = new_set_id
        candidate["native_observation_bindings"] = sorted(
            (
                CHECKER._native_observation_binding(
                    rows_by_id[row["row_id"]], new_set_id
                )
                if row["expected_test_set_id"] == old_set_id
                else row
                for row in candidate["native_observation_bindings"]
            ),
            key=lambda row: row["id"],
        )
        candidate["expected_test_sets"].sort(key=lambda row: row["id"])
        candidate["strict_summary"] = CHECKER._section_summary(candidate)
        return candidate

    def test_positive_repository_validation(self) -> None:
        repository_inventory = REPOSITORY_ROOT / "tools/test_inventory.json"
        repository_runner = REPOSITORY_ROOT / "tools/test_inventory_runner.zig"
        live_generation = self._assert_reviewed_fixture_generation(
            repository_inventory, repository_runner
        )
        fixture_generation = self._assert_reviewed_fixture_generation(
            self.inventory_path, self.root / "tools/test_inventory_runner.zig"
        )
        self.assertEqual(live_generation, fixture_generation)
        environment = __import__("os").environ
        ambient_git_pager = environment.pop("GIT_PAGER", None)
        try:
            self.assertEqual(
                [],
                CHECKER.validate(
                    REPOSITORY_ROOT,
                    REPOSITORY_ROOT / "tools/test_inventory.json",
                    structure_only=True,
                ),
            )
        finally:
            if ambient_git_pager is not None:
                environment["GIT_PAGER"] = ambient_git_pager

        with self._reviewed_digest_slots(
            inventory_current=fixture_generation, inventory_next=None
        ):
            self.assertEqual(
                [],
                CHECKER.validate(
                    self.root,
                    self.inventory_path,
                    structure_only=True,
                    require_current_only=True,
                ),
            )
        open_next = CHECKER.CURRENT_TEST_INVENTORY_SHA256
        if open_next == fixture_generation:
            open_next = CHECKER.NEXT_TEST_INVENTORY_SHA256
        if open_next is None or open_next == fixture_generation:
            open_next = "0" * 64 if fixture_generation != "0" * 64 else "1" * 64
        with self._reviewed_digest_slots(
            inventory_current=fixture_generation, inventory_next=open_next
        ):
            self.assertEqual(
                [],
                CHECKER.validate(
                    self.root,
                    self.inventory_path,
                    structure_only=True,
                ),
            )

        environment["GIT_PAGER"] = "less"
        try:
            ambient_commands = (
                [
                    "--root",
                    str(REPOSITORY_ROOT),
                    "--structure-only",
                ],
                [
                    "--root",
                    str(REPOSITORY_ROOT),
                    "--refresh-from-protocol",
                    "--protocol-log",
                    "env:aarch64-linux-gnu-baseline=missing-protocol.log",
                ],
            )
            for arguments in ambient_commands:
                with self.subTest(ambient_git_pager=arguments[-1]):
                    stderr = io.StringIO()
                    with contextlib.redirect_stderr(stderr):
                        result = CHECKER.main(arguments)
                    output = stderr.getvalue()
                    self.assertEqual(1, result)
                    self.assertNotIn("Traceback", output)
                    self.assertEqual(1, len(output.splitlines()))
                    self.assertIn("test inventory error:", output)
                    self.assertIn("GIT_PAGER", output)
        finally:
            if ambient_git_pager is None:
                environment.pop("GIT_PAGER", None)
            else:
                environment["GIT_PAGER"] = ambient_git_pager
        self.assertEqual(312, len(self.inventory["test_mode_rows"]))
        self.assertEqual(42, len(self.inventory["expected_test_sets"]))
        self.assertEqual(123, len(self.inventory["native_observation_bindings"]))
        self.assertEqual(123, CHECKER._matrix_incomplete_count(self.inventory))

    def test_fixture_positive_validation(self) -> None:
        self.assertEqual(
            [],
            CHECKER.validate(self.root, self.inventory_path, structure_only=True),
        )
        workflow_commands = {
            row["workflow_observation_id"]: row["command_template"]
            for row in self.inventory["workflow_mode_bindings"]
        }
        self.assertEqual(
            {
                "workflow-launch:.github/workflows/ci.yml:target-tests:test-debug-target": "zig build test ${{ matrix.target_args }} -Dtest-optimize=Debug -Dhost-tool-smoke=false --summary failures",
                "workflow-launch:.github/workflows/ci.yml:target-tests:test-releasesafe-target": "zig build --release=safe test ${{ matrix.target_args }} -Dtest-optimize=ReleaseSafe -Dhost-tool-smoke=false --summary failures",
                "workflow-launch:.github/workflows/ci.yml:target-tests:test-releasefast-target": "zig build --release=fast test ${{ matrix.target_args }} -Dtest-optimize=ReleaseFast -Dhost-tool-smoke=false --summary failures",
                "workflow-launch:.github/workflows/release.yml:artifacts:test": "zig build test ${{ matrix.target_args }} -Dtest-optimize=ReleaseSafe -Dhost-tool-smoke=false --summary failures",
            },
            workflow_commands,
        )
        build_inventory_root = next(
            row
            for row in self.inventory["test_roots"]
            if row["id"] == "python-root:build-inventory-direct"
        )
        self.assertIsNone(build_inventory_root["aggregate_step_observation_id"])
        self.assertIs(build_inventory_root["matrix_applicable"], True)
        windows_build_inventory_rows = [
            row
            for row in self.inventory["test_mode_rows"]
            if row["root_id"] == "python-root:build-inventory-direct"
            and row["environment_id"] == "env:x86-64-windows-gnu-baseline"
        ]
        self.assertEqual(3, len(windows_build_inventory_rows))
        self.assertEqual(
            {"mode:Debug", "mode:ReleaseSafe", "mode:ReleaseFast"},
            {row["optimize_mode_id"] for row in windows_build_inventory_rows},
        )
        for row in windows_build_inventory_rows:
            self.assertEqual("inapplicable", row["disposition"])
            self.assertEqual("predicate:host-tool-smoke-disabled", row["predicate_id"])
            self.assertIsNone(row["command_template"])
            self.assertIsNone(row["expected_test_set_id"])
        weakened_windows = copy.deepcopy(self.inventory)
        weakened_row = next(
            row
            for row in weakened_windows["test_mode_rows"]
            if row["id"] == windows_build_inventory_rows[0]["id"]
        )
        weakened_row.update(
            {
                "disposition": "execute",
                "predicate_id": "predicate:always",
                "command_template": "python3 -B test/build/test_build_inventory.py",
                "expected_test_set_id": next(
                    row["id"]
                    for row in weakened_windows["expected_test_sets"]
                    if row["root_id"] == "python-root:build-inventory-direct"
                ),
            }
        )
        weakened_windows["strict_summary"] = CHECKER._section_summary(weakened_windows)
        self._write(weakened_windows)
        self.assertIn("immutable matrix fields changed", self._errors())
        self._write(copy.deepcopy(self.inventory))
        for weakened_flag in ("true", "${{ matrix.host_tool_smoke }}"):
            with self.subTest(workflow_host_tool_flag=weakened_flag):
                inventory = copy.deepcopy(self.inventory)
                inventory["workflow_mode_bindings"][0]["command_template"] = inventory[
                    "workflow_mode_bindings"
                ][0]["command_template"].replace("false", weakened_flag)
                inventory["strict_summary"] = CHECKER._section_summary(inventory)
                self._write(inventory)
                self.assertIn("workflow_mode_bindings", self._errors())
        self._write(copy.deepcopy(self.inventory))
        original = self.inventory_path.read_bytes()
        whitespace = original + b" "
        self.inventory_path.write_bytes(whitespace)
        errors = CHECKER.validate(self.root, self.inventory_path, structure_only=True)
        whitespace_digest = hashlib.sha256(whitespace).hexdigest()
        self.assertEqual(1, len(errors))
        self.assertIn(f"observed sha256={whitespace_digest}", errors[0])
        with self._reviewed_digest_slots(inventory_next=whitespace_digest):
            self.assertEqual(
                [],
                CHECKER.validate(self.root, self.inventory_path, structure_only=True),
            )

        reordered = copy.deepcopy(self.inventory)
        reordered = {key: reordered[key] for key in reversed(tuple(reordered))}
        reordered_bytes = CHECKER._canonical_inventory_bytes(reordered)
        self.inventory_path.write_bytes(reordered_bytes)
        reordered_digest = hashlib.sha256(reordered_bytes).hexdigest()
        errors = CHECKER.validate(self.root, self.inventory_path, structure_only=True)
        self.assertEqual(1, len(errors))
        self.assertIn(f"observed sha256={reordered_digest}", errors[0])

        same_length = original.replace(
            b'"schema_version": 3', b'"schema_version": 4', 1
        )
        self.assertEqual(len(original), len(same_length))
        self.inventory_path.write_bytes(same_length)
        same_length_digest = hashlib.sha256(same_length).hexdigest()
        errors = CHECKER.validate(self.root, self.inventory_path, structure_only=True)
        self.assertEqual(1, len(errors))
        self.assertIn(f"observed sha256={same_length_digest}", errors[0])

        exact_limit = original + b" " * (CHECKER.MAX_INVENTORY_BYTES - len(original))
        self.assertEqual(CHECKER.MAX_INVENTORY_BYTES, len(exact_limit))
        self.inventory_path.write_bytes(exact_limit)
        exact_limit_digest = hashlib.sha256(exact_limit).hexdigest()
        with self._reviewed_digest_slots(inventory_next=exact_limit_digest):
            self.assertEqual(
                [],
                CHECKER.validate(self.root, self.inventory_path, structure_only=True),
            )
        self.inventory_path.write_bytes(exact_limit + b" ")
        self.assertIn("exceeds", self._errors())
        self.inventory_path.write_bytes(original)

    def test_native_compiler_sets_are_frozen_and_content_deduplicated(self) -> None:
        rows = self.inventory["test_mode_rows"]
        rows_by_id = {row["id"]: row for row in rows}
        bindings = self.inventory["native_observation_bindings"]
        self.assertEqual(123, CHECKER._matrix_incomplete_count(self.inventory))
        self.assertEqual(bindings, sorted(bindings, key=lambda row: row["id"]))
        self.assertEqual(123, len(bindings))
        self.assertEqual(123, len({row["row_id"] for row in bindings}))
        for binding in bindings:
            with self.subTest(binding=binding["row_id"]):
                row = rows_by_id[binding["row_id"]]
                self.assertTrue(row["root_id"].startswith("zig-root:"))
                self.assertEqual(CHECKER.FROZEN_STATE, row["expectation_state"])
                self.assertEqual(
                    CHECKER._native_observation_binding(
                        row, row["expected_test_set_id"]
                    ),
                    binding,
                )
        self.assertEqual(
            60,
            sum(
                row["environment_id"] == "env:aarch64-macos-baseline"
                and row["root_id"].startswith("zig-root:")
                and row["expectation_state"] == CHECKER.FROZEN_STATE
                for row in rows
            ),
        )
        self.assertEqual(
            63,
            sum(
                row["environment_id"] == "env:x86-64-linux-gnu-baseline"
                and row["root_id"].startswith("zig-root:")
                and row["expectation_state"] == CHECKER.FROZEN_STATE
                for row in rows
            ),
        )
        pending_gaps = {
            gap["id"]: len(gap["subject_ids"])
            for gap in self.inventory["known_gaps"]
            if gap["kind"] == "native-test-enumeration-required"
        }
        self.assertEqual(
            {
                "gap:native-test-enumeration:env-aarch64-linux-gnu-baseline": 60,
                "gap:native-test-enumeration:env-x86-64-windows-gnu-baseline": 63,
            },
            pending_gaps,
        )
        gemm = {
            row["optimize_mode_id"]: row["expected_test_set_id"]
            for row in rows
            if row["environment_id"] == "env:aarch64-macos-baseline"
            and row["root_id"] == "zig-root:gemm-registry-tests"
        }
        self.assertNotEqual(gemm["mode:Debug"], gemm["mode:ReleaseSafe"])
        self.assertEqual(gemm["mode:ReleaseSafe"], gemm["mode:ReleaseFast"])

    def test_duplicate_json_key_and_unknown_key_fail(self) -> None:
        original = self.inventory_path.read_text(encoding="utf-8")
        self.inventory_path.write_text(
            '{"schema_version":0,' + original.lstrip()[1:], encoding="utf-8"
        )
        self.assertIn("duplicate JSON object key", self._errors())

        inventory = copy.deepcopy(self.inventory)
        inventory["unknown"] = True
        self._write(inventory)
        self.assertIn("top-level keys", self._errors())

    def test_missing_duplicate_and_folded_logical_roots_fail(self) -> None:
        inventory = copy.deepcopy(self.inventory)
        inventory["test_roots"].pop()
        self._write(inventory)
        self.assertIn("test_roots", self._errors())

        inventory = copy.deepcopy(self.inventory)
        inventory["test_roots"].append(copy.deepcopy(inventory["test_roots"][0]))
        self._write(inventory)
        self.assertIn("duplicate id", self._errors())

        inventory = copy.deepcopy(self.inventory)
        variants = [
            row
            for row in inventory["test_roots"]
            if row.get("physical_path") == "test/api/public_surface_contract_test.zig"
        ]
        self.assertEqual(2, len(variants))
        inventory["test_roots"].remove(variants[1])
        self._write(inventory)
        self.assertIn("test_roots", self._errors())

        self._write(copy.deepcopy(self.inventory))
        build_path = self.root / "tools/build_inventory.json"
        build_inventory = json.loads(build_path.read_text(encoding="utf-8"))
        aggregate = next(
            row
            for row in build_inventory["build_observations"]
            if row["id"] == CHECKER.AGGREGATE_STEP_ID
        )
        aggregate["direct_dependencies"] = [
            dependency
            for dependency in aggregate["direct_dependencies"]
            if dependency["id"] != "launch:build.zig:build:run_modern_tests"
        ]
        build_path.write_text(
            json.dumps(build_inventory, indent=2) + "\n", encoding="utf-8"
        )
        self.assertIn("aggregate edge", self._errors())

        baseline_build_inventory = json.loads(
            (REPOSITORY_ROOT / "tools/build_inventory.json").read_text(encoding="utf-8")
        )
        for dependency in CHECKER.HOST_TOOL_SMOKE_DIRECT_DEPENDENCIES:
            with self.subTest(host_tool_dependency_removed=dependency["id"]):
                build_inventory = copy.deepcopy(baseline_build_inventory)
                observations = {
                    row["id"]: row for row in build_inventory["build_observations"]
                }
                host_step = observations[CHECKER.HOST_TOOL_SMOKE_STEP_ID]
                host_step["direct_dependencies"] = [
                    edge
                    for edge in host_step["direct_dependencies"]
                    if edge["id"] != dependency["id"]
                ]
                build_path.write_text(
                    json.dumps(build_inventory, indent=2) + "\n", encoding="utf-8"
                )
                self.assertIn("host-tool smoke aggregate closure", self._errors())

        for mutation in ("extra-host-dependency", "inverted-condition", "bypass"):
            with self.subTest(host_tool_path_mutation=mutation):
                build_inventory = copy.deepcopy(baseline_build_inventory)
                observations = {
                    row["id"]: row for row in build_inventory["build_observations"]
                }
                host_step = observations[CHECKER.HOST_TOOL_SMOKE_STEP_ID]
                aggregate = observations[CHECKER.AGGREGATE_STEP_ID]
                if mutation == "extra-host-dependency":
                    host_step["direct_dependencies"].append(
                        {
                            "id": "launch:build.zig:build:build_inventory_tests",
                            "condition": "always",
                        }
                    )
                    expected = "host-tool smoke aggregate closure"
                elif mutation == "inverted-condition":
                    next(
                        edge
                        for edge in aggregate["direct_dependencies"]
                        if edge["id"] == CHECKER.HOST_TOOL_SMOKE_STEP_ID
                    )["condition"] = "host-tool-smoke is false"
                    expected = "host-tool smoke canonical aggregate edge"
                else:
                    aggregate["direct_dependencies"].append(
                        {
                            "id": CHECKER.PYTHON_TOOLING_STEP_ID,
                            "condition": "always",
                        }
                    )
                    expected = "bypasses the unique host-tool smoke path"
                build_path.write_text(
                    json.dumps(build_inventory, indent=2) + "\n", encoding="utf-8"
                )
                self.assertIn(expected, self._errors())

        build_inventory = json.loads(
            (REPOSITORY_ROOT / "tools/build_inventory.json").read_text(encoding="utf-8")
        )
        selector = next(
            row
            for row in build_inventory["build_observations"]
            if row["id"] == "step:build.zig:build:test-public-api-contract"
        )
        selector["direct_dependencies"].pop()
        build_path.write_text(
            json.dumps(build_inventory, indent=2) + "\n", encoding="utf-8"
        )
        self.assertIn("test_roots", self._errors())

    def test_zig_declaration_add_delete_and_reorder_fail_sanity(self) -> None:
        path = self.root / "test/api/zynum_test.zig"
        original = path.read_text(encoding="utf-8")
        path.write_text(original + '\ntest "fixture added" {}\n', encoding="utf-8")
        self.assertIn("zig_test_files", self._errors())

        path.write_text(
            original.replace(
                'test "modern typed gemm API"',
                'test "modern typed gemm API removed"',
                1,
            ),
            encoding="utf-8",
        )
        self.assertIn("zig_test_files", self._errors())

        first = "top-level package exposes Zynum BLAS namespace"
        second = "modern typed gemm API"
        changed = (
            original.replace(first, "fixture-swap", 1)
            .replace(second, first, 1)
            .replace("fixture-swap", second, 1)
        )
        path.write_text(changed, encoding="utf-8")
        self.assertIn("zig_test_files", self._errors())

    def test_expected_set_empty_count_digest_and_reachability_fail(self) -> None:
        inventory = copy.deepcopy(self.inventory)
        target = next(
            row
            for row in inventory["expected_test_sets"]
            if row["root_id"] == "zig-root:modern-tests"
        )
        target["tests"] = []
        target["count"] = 0
        target["digest"] = CHECKER._fact_digest([])
        target["id"] = CHECKER._content_set_id(target["root_id"], [])
        self._write(inventory)
        self.assertIn("nonempty", self._errors())

        inventory = copy.deepcopy(self.inventory)
        target = next(
            row
            for row in inventory["expected_test_sets"]
            if row["root_id"] == "zig-root:modern-tests"
        )
        target["count"] += 1
        self._write(inventory)
        self.assertIn("count/digest", self._errors())

        self._write(copy.deepcopy(self.inventory))
        source = self.root / "src/blas/triangular_parallel_test.zig"
        text = source.read_text(encoding="utf-8")
        source.write_text(
            text.replace(
                '@import("core/matrix_matrix/triangular_right_parallel.zig")',
                '@import("core/matrix_matrix/triangular.zig")',
                1,
            ),
            encoding="utf-8",
        )
        self.assertIn("reaching declaration", self._errors())

    def test_python_unregistered_and_dynamic_discovery_fail(self) -> None:
        added = self.root / "test/new/test_added.py"
        added.parent.mkdir(parents=True)
        added.write_text(
            "import unittest\nclass Added(unittest.TestCase):\n    def test_added(self): pass\n",
            encoding="utf-8",
        )
        self.assertIn("19 Python test candidates", self._errors())

        added.unlink()
        source = self.root / "test/build/test_test_inventory.py"
        source.write_text(
            source.read_text(encoding="utf-8")
            + "\ndef load_tests(*args): return args\n",
            encoding="utf-8",
        )
        self.assertIn("dynamic unittest discovery", self._errors())

    def test_predicate_ast_digest_is_version_stable(self) -> None:
        empty_call = ast.parse("probe()", mode="eval").body
        mutated_call = ast.parse("probe(flag=True)", mode="eval").body
        full_field_dump = (
            "Call(func=Name(id='probe', ctx=Load()), args=[], keywords=[])"
        )
        self.assertEqual(
            hashlib.sha256(full_field_dump.encode("utf-8")).hexdigest(),
            CHECKER._predicate_ast_sha256(empty_call),
        )
        self.assertNotEqual(
            CHECKER._predicate_ast_sha256(empty_call),
            CHECKER._predicate_ast_sha256(mutated_call),
        )

    def test_predicate_x86_and_mode_row_mutations_fail(self) -> None:
        platform_predicate_id = (
            "python-skip-predicate:report-publication-platform-unavailable"
        )
        artifact_predicate_id = (
            "python-skip-predicate:artifact-snapshot-platform-unavailable"
        )
        platform_skip_entries = [
            entry
            for entry in self.inventory["python_skip_contracts"][0]["entries"]
            if entry["predicate_id"] == platform_predicate_id
        ]
        self.assertEqual(60, len(platform_skip_entries))
        artifact_skip_entries = [
            entry
            for entry in self.inventory["python_skip_contracts"][0]["entries"]
            if entry["predicate_id"] == artifact_predicate_id
        ]
        self.assertEqual(33, len(artifact_skip_entries))
        self.assertEqual(0x00000900, CHECKER.WINDOWS_PYTHON_TOOLING_BLAS_WINMODE)

        tooling_root = next(
            row
            for row in self.inventory["test_roots"]
            if row["id"] == CHECKER.PYTHON_TOOLING_ROOT_ID
        )
        admitted_sources = CHECKER._PYTHON_TOOLING_REVIEWED_SOURCE_SHA256
        self.assertIs(type(admitted_sources), tuple)
        self.assertEqual(14, len(admitted_sources))
        self.assertEqual(
            tuple(tooling_root["module_paths"]),
            tuple(path for path, _ in admitted_sources),
        )
        self.assertEqual(
            len(admitted_sources), len({path for path, _ in admitted_sources})
        )
        for path, digest in admitted_sources:
            self.assertRegex(digest, r"\A[0-9a-f]{64}\Z")
            self.assertEqual(
                digest,
                hashlib.sha256((self.root / path).read_bytes()).hexdigest(),
            )

        execution_sources = CHECKER._PYTHON_TOOLING_EXECUTION_SOURCE_SHA256
        execution_instances = CHECKER._PYTHON_TOOLING_EXECUTION_MODULES
        self.assertIs(type(execution_sources), tuple)
        self.assertIs(type(execution_instances), tuple)
        self.assertEqual((39, 41), (len(execution_sources), len(execution_instances)))
        self.assertEqual(admitted_sources, execution_sources[:14])
        self.assertEqual(
            set(path for path, _ in execution_sources),
            set(path for _, path in execution_instances),
        )
        self.assertEqual(41, len({name for name, _ in execution_instances}))
        for path, digest in execution_sources:
            self.assertRegex(digest, r"\A[0-9a-f]{64}\Z")
            self.assertEqual(
                digest,
                hashlib.sha256((self.root / path).read_bytes()).hexdigest(),
            )

        execution_context = CHECKER.BUILD_CHECKER._make_discovery_context(
            self.root, self.inventory_path
        )
        execution_closure = CHECKER._freeze_python_tooling_execution_closure(
            self.root, execution_context, tooling_root["module_paths"]
        )
        self.assertEqual(39, len(execution_closure.sources))
        self.assertEqual(41, len(execution_closure.module_paths))
        self.assertEqual(execution_context.root, execution_closure.root)
        for path, source in execution_closure.sources.items():
            expected_source_path = execution_context.root / path
            self.assertEqual(expected_source_path, source.source_path)
            self.assertEqual(
                expected_source_path, source.source_path.resolve(strict=True)
            )
        execution_capsule = CHECKER._python_tooling_execution_capsule(execution_closure)
        capsule_root, capsule_sources, capsule_modules = (
            CHECKER._decode_python_tooling_execution_capsule(execution_capsule)
        )
        self.assertEqual(str(execution_closure.root), capsule_root)
        self.assertEqual(39, len(capsule_sources))
        self.assertEqual(41, len(capsule_modules))
        if os.name == "posix":
            capsule_probe = CHECKER._python_tooling_posix_capsule_probe(
                execution_capsule
            )
            self.assertEqual(0, capsule_probe.returncode, capsule_probe.stderr)
            self.assertIn("zynum-capsule-target-ok|__main__|", capsule_probe.stdout)
            self.assertIn(
                str(execution_closure.root / "bench/tools/report_schedule.py"),
                capsule_probe.stdout,
            )
            nested_probe = CHECKER._python_tooling_posix_capsule_probe(
                execution_capsule, nested=True
            )
            self.assertEqual(0, nested_probe.returncode, nested_probe.stderr)
            self.assertEqual("zynum-capsule-nested-ok\n", nested_probe.stdout)
        with tempfile.TemporaryDirectory(prefix="capsule-windows-transport-") as temp:
            controller = Path(temp) / "run_level2_report.py"
            controller_bytes = b"# frozen controller fixture\n"
            controller.write_bytes(controller_bytes)
            frozen_source = types.SimpleNamespace(
                inventory_path="bench/tools/run_level2_report.py",
                source_path=controller,
                source_bytes=controller_bytes,
                source_sha256=hashlib.sha256(controller_bytes).hexdigest(),
            )
            fake_closure = types.SimpleNamespace(
                sources={frozen_source.inventory_path: frozen_source},
                module_paths={
                    "run_level2_report": frozen_source.inventory_path,
                },
                verify=mock.Mock(),
            )
            launcher = CHECKER._PythonToolingCapsuleLauncher(
                fake_closure, b"reviewed capsule"
            )
            bypass_args = ["not-python", str(controller)]

            def rejected_passthrough(*_args, **_kwargs):
                self.fail("executable override reached trusted subprocess passthrough")

            bypass_patch = mock.patch.object(
                CHECKER,
                "_PYTHON_TOOLING_TRUSTED_SUBPROCESS_RUN",
                rejected_passthrough,
            )
            bypass_patch.start()
            try:
                with self.assertRaisesRegex(
                    CHECKER.InventoryError, "transport is checker-owned"
                ):
                    launcher.run(bypass_args, executable=sys.executable)
            finally:
                bypass_patch.stop()
            embedded_source = CHECKER._PYTHON_TOOLING_BOOTSTRAP_SOURCE
            self.assertLess(
                embedded_source.index("if 'executable' in kwargs"),
                embedded_source.index("if os.path.realpath(argv[0])"),
            )
            relative_args = [sys.executable, controller.name, "--help"]

            def timeout_run(command, **_kwargs):
                raise subprocess.TimeoutExpired(command, 0.01)

            timeout_patch = mock.patch.object(
                CHECKER, "_PYTHON_TOOLING_TRUSTED_SUBPROCESS_RUN", timeout_run
            )
            timeout_patch.start()
            try:
                with self.assertRaises(subprocess.TimeoutExpired) as timeout_context:
                    launcher.run(relative_args, cwd=temp, timeout=0.01)
            finally:
                timeout_patch.stop()
            self.assertIs(relative_args, timeout_context.exception.cmd)
            for owned_keyword in (
                "close_fds",
                "creationflags",
                "executable",
                "pass_fds",
                "preexec_fn",
                "startupinfo",
            ):
                with self.assertRaisesRegex(
                    CHECKER.InventoryError, "transport is checker-owned"
                ):
                    launcher.run(
                        relative_args,
                        cwd=temp,
                        **{owned_keyword: None},
                    )
            inherited_calls = []
            launched = {}

            class StartupInfo:
                lpAttributeList = None

            def windows_run(command, **kwargs):
                launched["command"] = command
                launched["kwargs"] = kwargs
                return subprocess.CompletedProcess(command, 0, "ok", "")

            original_args = [sys.executable, str(controller), "--help"]
            windows_run_patch = mock.patch.object(
                CHECKER, "_PYTHON_TOOLING_TRUSTED_SUBPROCESS_RUN", windows_run
            )
            windows_name = mock.patch.object(CHECKER.os, "name", "nt")
            windows_inheritable = mock.patch.object(
                CHECKER.os,
                "set_handle_inheritable",
                side_effect=lambda handle, value: inherited_calls.append(
                    (handle, value)
                ),
                create=True,
            )
            windows_startup = mock.patch.object(
                CHECKER.subprocess, "STARTUPINFO", StartupInfo, create=True
            )
            fake_msvcrt = types.ModuleType("msvcrt")
            fake_msvcrt.get_osfhandle = lambda _fd: 919
            windows_runtime = mock.patch.object(
                CHECKER,
                "_python_tooling_windows_handle_runtime",
                return_value=fake_msvcrt,
            )
            windows_name.start()
            windows_inheritable.start()
            windows_startup.start()
            windows_runtime.start()
            windows_run_patch.start()
            try:
                windows_result = launcher.run(
                    original_args, capture_output=True, text=True
                )
            finally:
                windows_run_patch.stop()
                windows_runtime.stop()
                windows_startup.stop()
                windows_inheritable.stop()
                windows_name.stop()
            self.assertIs(original_args, windows_result.args)
            self.assertEqual([(919, True), (919, False)], inherited_calls)
            self.assertEqual("919", launched["command"][6])
            self.assertTrue(launched["kwargs"]["close_fds"])
            self.assertEqual(
                {"handle_list": [919]},
                launched["kwargs"]["startupinfo"].lpAttributeList,
            )
            unsupported_name = mock.patch.object(CHECKER.os, "name", "unsupported")
            unsupported_name.start()
            try:
                with self.assertRaisesRegex(
                    CHECKER.InventoryError, "capsule transport is unsupported"
                ):
                    launcher.run(original_args)
            finally:
                unsupported_name.stop()
            trusted_run = CHECKER.subprocess.run
            benchmark_module = types.SimpleNamespace(_FROZEN_SOURCE_RESOLVER=None)
            fake_closure.instances = {
                "benchmark_artifacts": types.SimpleNamespace(module=benchmark_module)
            }

            def rejected_install(resolver):
                benchmark_module._FROZEN_SOURCE_RESOLVER = resolver
                if resolver is not None:
                    raise RuntimeError("install injection")

            benchmark_module._set_frozen_source_resolver = rejected_install
            with self.assertRaisesRegex(
                CHECKER.InventoryError, "resolver install failed"
            ):
                with CHECKER._python_tooling_capsule_runtime(
                    fake_closure, b"reviewed capsule"
                ):
                    self.fail("failed resolver install entered the runtime")
            self.assertIs(CHECKER.subprocess.run, trusted_run)
            self.assertIsNone(benchmark_module._FROZEN_SOURCE_RESOLVER)

            def rejected_restore(resolver):
                if resolver is None:
                    raise RuntimeError("restore injection")
                benchmark_module._FROZEN_SOURCE_RESOLVER = resolver

            benchmark_module._set_frozen_source_resolver = rejected_restore
            with self.assertRaisesRegex(
                CHECKER.InventoryError, "resolver restore failed"
            ):
                with CHECKER._python_tooling_capsule_runtime(
                    fake_closure, b"reviewed capsule"
                ):
                    self.assertIsNot(CHECKER.subprocess.run, trusted_run)
            self.assertIs(CHECKER.subprocess.run, trusted_run)
        for description, mutant in (
            ("truncate", execution_capsule[:-1]),
            (
                "corrupt",
                execution_capsule[:-33]
                + bytes([execution_capsule[-33] ^ 1])
                + execution_capsule[-32:],
            ),
            ("duplicate", execution_capsule + execution_capsule[-32:]),
        ):
            with (
                self.subTest(python_tooling_capsule=description),
                self.assertRaises(CHECKER.InventoryError),
            ):
                CHECKER._decode_python_tooling_execution_capsule(mutant)

        for raw_sink_source in (
            b"import os\nos.popen('python')\n",
            b"import os\nos.posix_spawn('python', (), {})\n",
            b"import os\nos.posix_spawnp('python', (), {})\n",
            b"import os\nos.fork()\n",
            b"import os\nos.forkpty()\n",
            b"import os\nos.startfile('tool.py')\n",
            b"from os import popen as launch\nlaunch('python')\n",
            b"import os\nlaunch = os.posix_spawn\nlaunch('python', (), {})\n",
        ):
            raw_sink = CHECKER._PythonFrozenSource(
                "synthetic/raw_sink.py",
                Path("synthetic/raw_sink.py"),
                raw_sink_source,
                hashlib.sha256(raw_sink_source).hexdigest(),
                (1, 1, len(raw_sink_source), 1, 1),
                stat.S_IFREG | 0o444,
            )
            with self.assertRaisesRegex(CHECKER.InventoryError, "raw subprocess site"):
                CHECKER._python_tooling_subprocess_source_audit(raw_sink)

        def assert_execution_preflight_denied(
            expected_error: str,
            *,
            context: Any = execution_context,
        ) -> None:
            with (
                mock.patch.object(CHECKER.ast, "parse") as rejected_parse,
                mock.patch.object(
                    CHECKER, "_python_windows_blas_source_audit"
                ) as rejected_audit,
                mock.patch("builtins.compile") as rejected_compile,
                mock.patch("builtins.exec") as rejected_exec,
                self.assertRaisesRegex(CHECKER.InventoryError, expected_error),
            ):
                CHECKER._freeze_python_tooling_execution_closure(
                    self.root, context, tooling_root["module_paths"]
                )
            rejected_parse.assert_not_called()
            rejected_audit.assert_not_called()
            rejected_compile.assert_not_called()
            rejected_exec.assert_not_called()

        for path, _ in execution_sources:
            node = execution_context.public_files.node(path)
            self.assertTrue(node.bytes)
            mutated_bytes = bytes([node.bytes[0] ^ 1]) + node.bytes[1:]
            mutated_node = dataclasses.replace(
                node,
                bytes=mutated_bytes,
                sha256=hashlib.sha256(mutated_bytes).hexdigest(),
            )
            node_index = dict(execution_context.public_files.node_index)
            node_index[path] = mutated_node
            nodes = tuple(
                mutated_node if candidate.path == path else candidate
                for candidate in execution_context.public_files.nodes
            )
            mutated_universe = execution_context.public_files._replace(
                nodes=nodes, node_index=node_index
            )
            mutated_context = execution_context._replace(public_files=mutated_universe)
            with self.subTest(python_tooling_execution_byte_admission=path):
                assert_execution_preflight_denied(
                    "closure source is not admitted", context=mutated_context
                )

        execution_source_mutants = (
            execution_sources[:-1],
            (*execution_sources, ("bench/tools/extra.py", "0" * 64)),
            (*execution_sources[:-1], execution_sources[0]),
            (execution_sources[1], execution_sources[0], *execution_sources[2:]),
            ((execution_sources[0][0], "0" * 64), *execution_sources[1:]),
            (
                (execution_sources[0][0], execution_sources[0][1].upper()),
                *execution_sources[1:],
            ),
            ((execution_sources[0][0], "0" * 63), *execution_sources[1:]),
        )
        for mutant in execution_source_mutants:
            with mock.patch.object(
                CHECKER, "_PYTHON_TOOLING_EXECUTION_SOURCE_SHA256", mutant
            ):
                assert_execution_preflight_denied("execution")

        execution_instance_mutants = (
            execution_instances[:-1],
            (*execution_instances, ("extra", execution_instances[0][1])),
            (*execution_instances[:-1], execution_instances[0]),
            (execution_instances[1], execution_instances[0], *execution_instances[2:]),
        )
        for mutant in execution_instance_mutants:
            with mock.patch.object(
                CHECKER, "_PYTHON_TOOLING_EXECUTION_MODULES", mutant
            ):
                assert_execution_preflight_denied("execution")

        helper_path = self.root / "bench/tools/benchmark_artifacts.py"
        helper_bytes = helper_path.read_bytes()
        try:
            helper_path.write_bytes(bytes([helper_bytes[0] ^ 1]) + helper_bytes[1:])
            with self.assertRaisesRegex(
                CHECKER.InventoryError, "closure source changed after freeze"
            ):
                execution_closure.live_recheck()
        finally:
            helper_path.write_bytes(helper_bytes)

        execution_context = CHECKER.BUILD_CHECKER._make_discovery_context(
            self.root, self.inventory_path
        )
        execution_closure = CHECKER._freeze_python_tooling_execution_closure(
            self.root, execution_context, tooling_root["module_paths"]
        )
        held_helper_path = helper_path.with_name("benchmark_artifacts.held.py")
        helper_path.rename(held_helper_path)
        helper_path.write_bytes(helper_bytes)
        try:
            with self.assertRaisesRegex(
                CHECKER.InventoryError, "closure source changed after freeze"
            ):
                execution_closure.live_recheck()
        finally:
            helper_path.unlink()
            held_helper_path.rename(helper_path)

        execution_context = CHECKER.BUILD_CHECKER._make_discovery_context(
            self.root, self.inventory_path
        )
        execution_closure = CHECKER._freeze_python_tooling_execution_closure(
            self.root, execution_context, tooling_root["module_paths"]
        )
        tools_directory = helper_path.parent
        held_tools_directory = tools_directory.with_name("tools.held")
        tools_directory.rename(held_tools_directory)
        tools_directory.mkdir()
        helper_path.write_bytes(helper_bytes)
        try:
            with self.assertRaisesRegex(
                CHECKER.InventoryError, "closure source physical path changed"
            ):
                execution_closure.live_recheck()
        finally:
            helper_path.unlink()
            tools_directory.rmdir()
            held_tools_directory.rename(tools_directory)

        execution_context = CHECKER.BUILD_CHECKER._make_discovery_context(
            self.root, self.inventory_path
        )
        execution_closure = CHECKER._freeze_python_tooling_execution_closure(
            self.root, execution_context, tooling_root["module_paths"]
        )
        with CHECKER._python_tooling_execution_imports(execution_closure):
            _, _, _, closure_reviewed = CHECKER._python_tooling_source_skip_review(
                self.root,
                tooling_root["module_paths"],
                tooling_root["discovery_start"],
                tooling_root["discovery_pattern"],
                _closure=execution_closure,
            )
            CHECKER._load_python_tooling_closure_dependencies(
                execution_closure,
                {reviewed.module_name for reviewed in closure_reviewed},
            )
            with CHECKER._registered_python_tooling_modules(closure_reviewed):
                execution_closure.verify(require_complete=True)
                alias_name = "_zynum_benchmark_artifact_repository_snapshot"
                alias_instance = execution_closure.instances[alias_name]
                original_loader = alias_instance.module.__loader__
                alias_instance.module.__loader__ = object()
                try:
                    with self.assertRaisesRegex(
                        CHECKER.InventoryError, "frozen module identity changed"
                    ):
                        execution_closure.verify(require_complete=True)
                finally:
                    alias_instance.module.__loader__ = original_loader
                sys.modules[alias_name] = types.ModuleType(alias_name)
                try:
                    with self.assertRaisesRegex(
                        CHECKER.InventoryError, "frozen module identity changed"
                    ):
                        execution_closure.verify(require_complete=True)
                finally:
                    sys.modules[alias_name] = alias_instance.module
                with self.assertRaisesRegex(
                    CHECKER.InventoryError, "outside the frozen manifest"
                ):
                    importlib.util.spec_from_file_location(
                        "benchmark_artifacts",
                        self.root / "bench/tools/probe_level2_case.py",
                    )
                with tempfile.TemporaryDirectory(
                    prefix="test-inventory-external-spec-"
                ) as external_directory:
                    external_path = Path(external_directory) / "external_module.py"
                    external_path.write_bytes(b"EXTERNAL_SENTINEL = True\n")
                    external_spec = importlib.util.spec_from_file_location(
                        "_zynum_external_spec_positive", external_path
                    )
                    self.assertIsNotNone(external_spec)
                    self.assertEqual(str(external_path), external_spec.origin)

                    external_root_copy = (
                        Path(external_directory) / "benchmark_artifacts.py"
                    )
                    external_root_copy.write_bytes(
                        execution_closure.sources[
                            "bench/tools/benchmark_artifacts.py"
                        ].source_bytes
                    )
                    with (
                        mock.patch("builtins.exec") as rejected_exec,
                        self.assertRaisesRegex(
                            CHECKER.InventoryError, "outside the frozen manifest"
                        ),
                    ):
                        importlib.util.spec_from_file_location(
                            "benchmark_artifacts", external_root_copy
                        )
                    rejected_exec.assert_not_called()

                    lexical_repo_alias = (
                        self.root / "bench/tools/post_preflight_external_alias.py"
                    )
                    trusted_realpath = CHECKER.os.path.realpath
                    with (
                        mock.patch.object(
                            CHECKER.os.path,
                            "realpath",
                            return_value=str(external_path),
                        ),
                        self.assertRaisesRegex(
                            CHECKER.InventoryError, "outside the frozen manifest"
                        ),
                    ):
                        importlib.util.spec_from_file_location(
                            "_zynum_repo_lexical_external_resolved",
                            lexical_repo_alias,
                        )
                    with (
                        mock.patch.object(
                            CHECKER.os.path,
                            "realpath",
                            return_value=str(
                                execution_closure.sources[
                                    "bench/tools/benchmark_artifacts.py"
                                ].source_path
                            ),
                        ),
                        self.assertRaisesRegex(
                            CHECKER.InventoryError, "outside the frozen manifest"
                        ),
                    ):
                        importlib.util.spec_from_file_location(
                            "benchmark_artifacts", external_path
                        )
                    self.assertIs(CHECKER.os.path.realpath, trusted_realpath)

        def review_admitted_sources(module_paths: list[str] | None = None) -> Any:
            return CHECKER._python_tooling_source_skip_review(
                self.root,
                tooling_root["module_paths"] if module_paths is None else module_paths,
                tooling_root["discovery_start"],
                tooling_root["discovery_pattern"],
            )

        def assert_preflight_denies_without_execution(
            expected_error: str,
            module_paths: list[str] | None = None,
        ) -> None:
            with (
                mock.patch.object(CHECKER.ast, "parse") as rejected_parse,
                mock.patch.object(
                    CHECKER, "_python_windows_blas_source_audit"
                ) as rejected_audit,
                mock.patch("builtins.compile") as rejected_compile,
                mock.patch("builtins.exec") as rejected_exec,
                self.assertRaisesRegex(CHECKER.InventoryError, expected_error),
            ):
                review_admitted_sources(module_paths)
            rejected_parse.assert_not_called()
            rejected_audit.assert_not_called()
            rejected_compile.assert_not_called()
            rejected_exec.assert_not_called()

        for path, _ in admitted_sources:
            source_path = self.root / path
            original_bytes = source_path.read_bytes()
            replacement = b" " if original_bytes[-1:] != b" " else b"\n"
            with self.subTest(python_tooling_source_byte_admission=path):
                try:
                    source_path.write_bytes(original_bytes[:-1] + replacement)
                    assert_preflight_denies_without_execution("exact bytes")
                finally:
                    source_path.write_bytes(original_bytes)

        admitted_paths = list(tooling_root["module_paths"])
        module_path_mutants = (
            ("missing", admitted_paths[:-1]),
            ("extra", [*admitted_paths, "bench/tools/test_extra.py"]),
            ("duplicate", [*admitted_paths[:-1], admitted_paths[0]]),
            ("reorder", [admitted_paths[1], admitted_paths[0], *admitted_paths[2:]]),
        )
        for description, module_path_mutant in module_path_mutants:
            with self.subTest(python_tooling_inventory_admission=description):
                assert_preflight_denies_without_execution(
                    "module order", module_path_mutant
                )

        admission_mutants = (
            ("missing", admitted_sources[:-1]),
            (
                "extra",
                (
                    *admitted_sources,
                    ("bench/tools/test_extra.py", "0" * 64),
                ),
            ),
            ("duplicate", (*admitted_sources[:-1], admitted_sources[0])),
            (
                "reorder",
                (admitted_sources[1], admitted_sources[0], *admitted_sources[2:]),
            ),
            (
                "wrong",
                (
                    (admitted_sources[0][0], "0" * 64),
                    *admitted_sources[1:],
                ),
            ),
            (
                "uppercase",
                (
                    (admitted_sources[0][0], admitted_sources[0][1].upper()),
                    *admitted_sources[1:],
                ),
            ),
            (
                "malformed",
                (
                    (admitted_sources[0][0], "0" * 63),
                    *admitted_sources[1:],
                ),
            ),
        )
        for description, admission_mutant in admission_mutants:
            with (
                self.subTest(python_tooling_digest_admission=description),
                mock.patch.object(
                    CHECKER,
                    "_PYTHON_TOOLING_REVIEWED_SOURCE_SHA256",
                    admission_mutant,
                ),
            ):
                assert_preflight_denies_without_execution("Python tooling")

        runtime_fixture = self.root / "bench/tools/test_exact_source_runtime_fixture.py"
        frozen_runtime_source = b"FROZEN_SOURCE_EXECUTED = True\n"
        replacement_runtime_source = b"REPLACEMENT_SOURCE_EXECUTED = True\n"
        runtime_fixture.write_bytes(frozen_runtime_source)
        frozen_runtime_review = CHECKER._reviewed_python_source_module(
            self.root,
            "bench/tools/test_exact_source_runtime_fixture.py",
            "bench/tools",
            "test_*.py",
        )
        runtime_fixture.write_bytes(replacement_runtime_source)
        executed_source_markers: list[tuple[bool, bool]] = []
        trusted_exec = exec

        def record_frozen_exec(code: Any, namespace: dict[str, Any]) -> None:
            trusted_exec(code, namespace)
            executed_source_markers.append(
                (
                    namespace.get("FROZEN_SOURCE_EXECUTED") is True,
                    "REPLACEMENT_SOURCE_EXECUTED" in namespace,
                )
            )

        try:
            with (
                mock.patch("builtins.exec", side_effect=record_frozen_exec),
                self.assertRaisesRegex(
                    CHECKER.InventoryError,
                    "source changed between review and module execution",
                ),
            ):
                with CHECKER._registered_python_tooling_modules(
                    (frozen_runtime_review,)
                ):
                    self.fail("replaced Python tooling source must not be yielded")
        finally:
            runtime_fixture.unlink(missing_ok=True)
        self.assertEqual([(True, False)], executed_source_markers)

        cli_build_output = self.root / "zig-cache/o/windows/zynum_blas.dll"
        cli_installed_output = self.root / CHECKER.WINDOWS_PYTHON_TOOLING_BLAS_PATH
        parsed_windows_cli = CHECKER._parser().parse_args(
            [
                "--run-python-tooling-root",
                CHECKER.PYTHON_TOOLING_ROOT_ID,
                "--windows-zynum-blas-build-output",
                str(cli_build_output),
                "--windows-zynum-blas-installed-output",
                str(cli_installed_output),
            ]
        )
        self.assertEqual(
            cli_build_output,
            parsed_windows_cli.windows_zynum_blas_build_output,
        )
        self.assertEqual(
            cli_installed_output,
            parsed_windows_cli.windows_zynum_blas_installed_output,
        )
        with (
            contextlib.redirect_stderr(io.StringIO()),
            self.assertRaises(SystemExit) as obsolete_option,
        ):
            CHECKER._parser().parse_args(
                [
                    "--windows-python-tooling-emitted-dll",
                    str(cli_build_output),
                ]
            )
        self.assertEqual(2, obsolete_option.exception.code)
        with contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(
                2,
                CHECKER.main(
                    [
                        "--windows-zynum-blas-build-output",
                        str(cli_build_output),
                    ]
                ),
            )
            self.assertEqual(
                2,
                CHECKER.main(
                    [
                        "--windows-zynum-blas-installed-output",
                        str(cli_installed_output),
                    ]
                ),
            )
        with (
            mock.patch.object(CHECKER.sys, "platform", "win32"),
            contextlib.redirect_stderr(io.StringIO()),
        ):
            for lone_option, lone_path in (
                ("--windows-zynum-blas-build-output", cli_build_output),
                ("--windows-zynum-blas-installed-output", cli_installed_output),
            ):
                self.assertEqual(
                    2,
                    CHECKER.main(
                        [
                            "--run-python-tooling-root",
                            CHECKER.PYTHON_TOOLING_ROOT_ID,
                            lone_option,
                            str(lone_path),
                        ]
                    ),
                )
            self.assertEqual(
                2,
                CHECKER.main(
                    [
                        "--run-python-tooling-root",
                        CHECKER.PYTHON_TOOLING_ROOT_ID,
                        "--windows-zynum-blas-build-output",
                        str(cli_build_output),
                        "--windows-zynum-blas-installed-output",
                        str(cli_installed_output),
                        "--structure-only",
                    ]
                ),
            )
            self.assertEqual(
                2,
                CHECKER.main(
                    [
                        "--run-python-tooling-root",
                        "python-root:noncanonical",
                        "--windows-zynum-blas-build-output",
                        str(cli_build_output),
                        "--windows-zynum-blas-installed-output",
                        str(cli_installed_output),
                    ]
                ),
            )

        with (
            mock.patch.object(CHECKER.sys, "platform", "linux"),
            contextlib.redirect_stderr(io.StringIO()),
        ):
            self.assertEqual(
                2,
                CHECKER.main(
                    [
                        "--run-python-tooling-root",
                        CHECKER.PYTHON_TOOLING_ROOT_ID,
                        "--windows-zynum-blas-build-output",
                        str(cli_build_output),
                        "--windows-zynum-blas-installed-output",
                        str(cli_installed_output),
                    ]
                ),
            )

        for missing_capability in (
            "O_NOFOLLOW",
            "supports_dir_fd",
            "supports_follow_symlinks",
            "open",
            "stat",
            "mkdir",
            "unlink",
            "rmdir",
            "rename",
        ):
            with (
                self.subTest(missing_report_capability=missing_capability),
                mock.patch.object(CHECKER.os, "name", "posix"),
                mock.patch.object(CHECKER.os, missing_capability, None),
                mock.patch.object(CHECKER.sys, "platform", "linux"),
            ):
                self.assertTrue(CHECKER._report_publication_platform_unavailable())

        artifact_capabilities = (
            "O_CLOEXEC",
            "O_DIRECTORY",
            "O_NOFOLLOW",
            "fchmod",
            "fstat",
            "geteuid",
            "open",
            "unlink",
        )
        for missing_capability in artifact_capabilities:
            artifact_os = mock.Mock(
                spec_set=[
                    "name",
                    *(
                        capability
                        for capability in artifact_capabilities
                        if capability != missing_capability
                    ),
                ]
            )
            artifact_os.name = "posix"
            self.assertFalse(hasattr(artifact_os, missing_capability))
            with (
                self.subTest(missing_artifact_capability=missing_capability),
                mock.patch.object(CHECKER, "os", artifact_os),
                mock.patch.object(CHECKER.sys, "platform", "linux"),
            ):
                self.assertTrue(CHECKER._artifact_snapshot_platform_unavailable())

        real_temporary_directory = tempfile.TemporaryDirectory
        with real_temporary_directory(
            prefix="test-inventory-provenance-probe-self-test-"
        ) as raw_probe_parent:
            probe_parent = Path(raw_probe_parent)
            default_root = probe_parent / "default-temp"
            probe_root = probe_parent / "reviewed-source-parent"
            default_root.mkdir()
            probe_root.mkdir()
            temporary_directory_roots: list[Path | None] = []

            def tracked_temporary_directory(*args: Any, **kwargs: Any) -> Any:
                directory = kwargs.get("dir")
                temporary_directory_roots.append(
                    None if directory is None else Path(directory)
                )
                return real_temporary_directory(*args, **kwargs)

            class DarwinFlistxattr:
                def __init__(self, names: tuple[bytes, ...], *, error: bool = False):
                    self.names = names
                    self.error = error
                    self.calls: list[tuple[Any, ...]] = []
                    self.restype: Any = None
                    self.argtypes: Any = None

                def __call__(self, *args: Any) -> int:
                    self.calls.append(args)
                    if self.error:
                        return -1
                    encoded = b"".join(name + b"\0" for name in self.names)
                    if args[1] is None:
                        return len(encoded)
                    ctypes.memmove(args[1], encoded, len(encoded))
                    return len(encoded)

            class DarwinLibrary:
                def __init__(self, flistxattr: DarwinFlistxattr) -> None:
                    self.flistxattr = flistxattr

            darwin_os = types.SimpleNamespace(
                O_CLOEXEC=os.O_CLOEXEC,
                O_CREAT=os.O_CREAT,
                O_EXCL=os.O_EXCL,
                O_NOFOLLOW=os.O_NOFOLLOW,
                O_RDONLY=os.O_RDONLY,
                O_WRONLY=os.O_WRONLY,
                chown=lambda *_args: None,
                close=os.close,
                getegid=os.getegid,
                getgroups=lambda: [os.getegid() + 1],
                open=os.open,
                strerror=os.strerror,
            )
            self.assertFalse(hasattr(darwin_os, "listxattr"))

            provenance_predicate_id = (
                "python-skip-predicate:no-automatic-provenance-xattr"
            )
            predicate_cases = (
                ((), False, True),
                ((b"com.apple.provenance",), False, False),
                ((), True, None),
            )
            for names, probe_error, expected in predicate_cases:
                flistxattr = DarwinFlistxattr(names, error=probe_error)
                temporary_directory_roots.clear()
                with (
                    self.subTest(
                        darwin_descriptor_xattrs=names,
                        darwin_descriptor_error=probe_error,
                    ),
                    mock.patch.object(CHECKER, "os", darwin_os),
                    mock.patch.object(CHECKER.sys, "platform", "darwin"),
                    mock.patch.object(
                        CHECKER.tempfile,
                        "TemporaryDirectory",
                        side_effect=tracked_temporary_directory,
                    ),
                    mock.patch.object(CHECKER.tempfile, "tempdir", str(default_root)),
                    mock.patch.object(
                        CHECKER.ctypes,
                        "CDLL",
                        return_value=DarwinLibrary(flistxattr),
                    ),
                    mock.patch.object(CHECKER.ctypes, "get_errno", return_value=5),
                ):
                    if probe_error:
                        with self.assertRaisesRegex(
                            CHECKER.InventoryError,
                            "cannot inspect descriptor xattrs",
                        ):
                            CHECKER._dynamic_python_skip_predicates(probe_root)
                    else:
                        predicates = CHECKER._dynamic_python_skip_predicates(probe_root)
                        self.assertIs(predicates[provenance_predicate_id], expected)
                self.assertEqual(
                    2 if probe_error else 4,
                    len(temporary_directory_roots),
                )
                self.assertEqual({probe_root}, set(temporary_directory_roots))
                self.assertEqual([], list(default_root.iterdir()))
                self.assertTrue(flistxattr.calls)
                self.assertTrue(
                    all(isinstance(call[0], int) for call in flistxattr.calls)
                )

        available_runner = mock.Mock(DEFAULT_ACCELERATE=object())
        available_runner.library_available.return_value = True
        available_runner.default_zynum_blas.return_value = "zig-out/lib/libzynum.so"
        available_modules = {
            "test_level1_report.py": {"runner": available_runner},
            "test_level2_report.py": {
                "TEST_BLAS": object(),
                "TEST_FILE_BLAS": object(),
            },
            "test_rank_k_report.py": {
                "REPO_ROOT": self.root,
                "runner": available_runner,
            },
            "test_rotg_latency_report.py": {
                "REPO_ROOT": self.root,
                "runner": available_runner,
            },
            "test_symm_report.py": {
                "REPO_ROOT": self.root,
                "runner": available_runner,
            },
            "test_triangular_matrix_report.py": {
                "REPO_ROOT": self.root,
                "integration_blas": lambda: self.root / "zig-out/lib/libzynum.so",
            },
        }
        reviewed_publication_source = (
            self.root / "bench/tools/test_report_publication.py"
        )
        publication_binding = types.SimpleNamespace(
            reviewed=types.SimpleNamespace(
                inventory_path="bench/tools/test_report_publication.py",
                source_path=reviewed_publication_source,
            )
        )
        dynamic_predicates = {
            predicate_id: False
            for predicate_id in CHECKER.REPORT_PUBLICATION_DYNAMIC_SKIP_PREDICATE_IDS
        }
        with (
            mock.patch.object(CHECKER.sys, "platform", "darwin"),
            mock.patch.object(
                CHECKER,
                "_report_publication_platform_unavailable",
                return_value=False,
            ),
            mock.patch.object(
                CHECKER,
                "_artifact_snapshot_platform_unavailable",
                return_value=False,
            ),
            mock.patch.object(CHECKER, "_verify_python_source_module_registry"),
            mock.patch.object(CHECKER, "_verify_python_source_module_binding"),
            mock.patch.object(
                CHECKER,
                "_dynamic_python_skip_predicates",
                return_value=dynamic_predicates,
            ) as dynamic_probe,
            mock.patch.object(
                CHECKER,
                "_required_python_tooling_module",
                side_effect=lambda modules, name: available_modules[name],
            ),
            mock.patch.object(Path, "is_file", autospec=True, return_value=True),
        ):
            CHECKER._python_skip_predicates((publication_binding,))
        dynamic_probe.assert_called_once_with(reviewed_publication_source.parent)

        windows_os = mock.Mock(spec_set=["name"])
        windows_os.name = "nt"
        self.assertFalse(hasattr(windows_os, "O_NOFOLLOW"))
        windows_fixture_paths = {
            self.root / relative_path
            for relative_path in CHECKER.WINDOWS_PYTHON_TOOLING_FIXTURE_PATHS
        }
        windows_library = self.root / "zig-out/lib/libzynum.so"
        available_paths = {*windows_fixture_paths, windows_library}

        def selectively_available(path: Path) -> bool:
            return Path(path) in available_paths

        with (
            mock.patch.object(CHECKER, "os", windows_os),
            mock.patch.object(CHECKER.sys, "platform", "win32"),
            mock.patch.object(
                CHECKER,
                "_dynamic_python_skip_predicates",
                side_effect=AssertionError("publication probes must not run"),
            ) as dynamic_publication_probes,
            mock.patch.object(
                CHECKER,
                "_verify_python_source_module_registry",
            ),
            mock.patch.object(
                CHECKER,
                "_required_python_tooling_module",
                side_effect=lambda modules, name: available_modules[name],
            ),
            mock.patch.object(
                Path, "is_file", autospec=True, side_effect=selectively_available
            ) as probed_files,
        ):
            windows_like_predicates = CHECKER._python_skip_predicates(())
        dynamic_publication_probes.assert_not_called()
        probed_paths = [call.args[0] for call in probed_files.call_args_list]
        self.assertFalse(any(path.suffix == ".exe" for path in probed_paths))
        self.assertTrue(any(path.name == "rank-k-probe" for path in probed_paths))
        self.assertTrue(
            any(path.name == "triangular-matrix-probe" for path in probed_paths)
        )
        runtime_probe_predicate_ids = (
            "python-skip-predicate:rank-k-artifacts-unavailable",
            "python-skip-predicate:rotg-latency-artifacts-unavailable",
            "python-skip-predicate:symm-artifacts-unavailable",
            "python-skip-predicate:triangular-matrix-artifacts-unavailable",
        )
        self.assertTrue(
            all(
                windows_like_predicates[predicate_id]
                for predicate_id in runtime_probe_predicate_ids
            )
        )
        available_paths.update({path.with_suffix("") for path in windows_fixture_paths})
        with (
            mock.patch.object(CHECKER, "os", windows_os),
            mock.patch.object(CHECKER.sys, "platform", "win32"),
            mock.patch.object(
                CHECKER,
                "_dynamic_python_skip_predicates",
                side_effect=AssertionError("publication probes must not run"),
            ),
            mock.patch.object(CHECKER, "_verify_python_source_module_registry"),
            mock.patch.object(
                CHECKER,
                "_required_python_tooling_module",
                side_effect=lambda modules, name: available_modules[name],
            ),
            mock.patch.object(
                Path, "is_file", autospec=True, side_effect=selectively_available
            ),
        ):
            real_probe_predicates = CHECKER._python_skip_predicates(())
        self.assertTrue(
            all(
                real_probe_predicates[predicate_id] is False
                for predicate_id in runtime_probe_predicate_ids
            )
        )
        self.assertIs(windows_like_predicates[platform_predicate_id], True)
        self.assertIs(windows_like_predicates[artifact_predicate_id], True)
        self.assertTrue(
            all(
                windows_like_predicates[predicate_id] is False
                for predicate_id in (
                    CHECKER.REPORT_PUBLICATION_SUBORDINATE_SKIP_PREDICATE_IDS
                )
            )
        )
        windows_skip_pairs = frozenset(
            (CHECKER._unittest_runtime_id(entry["test"]), entry["reason"])
            for entry in self.inventory["python_skip_contracts"][0]["entries"]
            if windows_like_predicates[entry["predicate_id"]]
        )
        self.assertEqual(
            frozenset(
                (CHECKER._unittest_runtime_id(entry["test"]), entry["reason"])
                for entry in self.inventory["python_skip_contracts"][0]["entries"]
                if entry["predicate_id"]
                in {
                    artifact_predicate_id,
                    platform_predicate_id,
                    *runtime_probe_predicate_ids,
                }
            ),
            windows_skip_pairs,
        )
        self.assertEqual(97, len(windows_skip_pairs))
        tooling_set = next(
            row
            for row in self.inventory["expected_test_sets"]
            if row["root_id"] == CHECKER.PYTHON_TOOLING_ROOT_ID
        )
        self.assertEqual(465, tooling_set["count"])

        fixture_paths = tuple(
            self.root / relative_path
            for relative_path in CHECKER.WINDOWS_PYTHON_TOOLING_FIXTURE_PATHS
        )
        fixture_paths[0].parent.mkdir(parents=True, exist_ok=True)
        for fixture_path in fixture_paths[:-1]:
            fixture_path.write_bytes(b"windows tooling applicability fixture")
        with self.assertRaisesRegex(
            CHECKER.InventoryError,
            "Windows Python tooling executable fixture",
        ):
            CHECKER._require_windows_python_tooling_fixtures(self.root)
        fixture_paths[-1].write_bytes(b"")
        with self.assertRaisesRegex(CHECKER.InventoryError, "fixture is empty"):
            CHECKER._require_windows_python_tooling_fixtures(self.root)
        fixture_paths[-1].write_bytes(b"windows tooling applicability fixture")
        fixture_paths[0].unlink()
        fixture_paths[0].symlink_to(fixture_paths[1])
        with self.assertRaisesRegex(
            CHECKER.InventoryError,
            "Windows Python tooling executable fixture",
        ):
            CHECKER._require_windows_python_tooling_fixtures(self.root)
        fixture_paths[0].unlink()
        fixture_paths[0].write_bytes(b"windows tooling applicability fixture")
        CHECKER._require_windows_python_tooling_fixtures(self.root)

        windows_library = self.root / "zig-out/bin/zynum_blas.dll"
        emitted_library = self.root / "zig-cache/o/windows/zynum_blas.dll"
        emitted_library.parent.mkdir(parents=True)
        alternate_library = self.root / "alternate/zynum_blas.dll"
        alternate_library.parent.mkdir()
        alternate_library.write_bytes(b"alternate Windows library fixture")
        hardlink_library = self.root / "alternate/zynum_blas-hardlink.dll"
        valid_windows_bytes = b"installed Windows library fixture"

        class WindowsFunction:
            def __init__(self, callback: Any) -> None:
                self.callback = callback
                self.argtypes: list[Any] | None = None
                self.restype: Any = None

            def __call__(self, *args: Any) -> Any:
                return self.callback(*args)

        class WindowsKernel32:
            def __init__(self) -> None:
                self.next_file_handle = 0x1000
                self.module_handle = 0x5000
                self.forwarder_handle = 0x6000
                self.module_path = windows_library.resolve()
                self.open_files: dict[int, tuple[Any, Path]] = {}
                self.reparse_paths: set[Path] = set()
                self.missing_symbol: str | None = None
                self.forwarded_symbol: str | None = None
                self.export_addresses = {
                    symbol: 0x100000 + ordinal * 0x20
                    for ordinal, symbol in enumerate(
                        CHECKER.WINDOWS_PYTHON_TOOLING_BLAS_REQUIRED_SYMBOLS, 1
                    )
                }
                self.CreateFileW = WindowsFunction(self.create_file)
                self.CloseHandle = WindowsFunction(self.close_handle)
                self.GetFileInformationByHandle = WindowsFunction(
                    self.get_file_information
                )
                self.GetFileInformationByHandleEx = WindowsFunction(
                    self.get_file_information_ex
                )
                self.GetFileSizeEx = WindowsFunction(self.get_file_size)
                self.SetFilePointerEx = WindowsFunction(self.set_file_pointer)
                self.ReadFile = WindowsFunction(self.read_file)
                self.GetModuleFileNameW = WindowsFunction(self.get_module_filename)
                self.GetProcAddress = WindowsFunction(self.get_proc_address)
                self.GetModuleHandleExW = WindowsFunction(self.get_module_handle_ex)

            def create_file(
                self,
                path: str,
                access: int,
                sharing: int,
                security: Any,
                disposition: int,
                flags: int,
                template: Any,
            ) -> int:
                self.assert_api_shape(
                    access == 0x80000000
                    and sharing == 1
                    and security is None
                    and disposition == 3
                    and flags == 0x00200080
                    and template is None
                )
                resolved = Path(path).resolve(strict=True)
                source = resolved.open("rb")
                handle = self.next_file_handle
                self.next_file_handle += 1
                self.open_files[handle] = (source, resolved)
                return handle

            def assert_api_shape(self, condition: bool) -> None:
                if not condition:
                    raise AssertionError("noncanonical Windows API call")

            def close_handle(self, handle: int) -> int:
                entry = self.open_files.pop(handle, None)
                if entry is None:
                    return 0
                entry[0].close()
                return 1

            def file_entry(self, handle: int) -> tuple[Any, Path, os.stat_result]:
                source, path = self.open_files[handle]
                return source, path, os.fstat(source.fileno())

            def get_file_information(self, handle: int, output: Any) -> int:
                _, path, observed = self.file_entry(handle)
                value = ctypes.cast(
                    output,
                    ctypes.POINTER(CHECKER._WindowsByHandleFileInformation),
                ).contents
                size = observed.st_size
                value.file_attributes = (
                    0x00000400 if path in self.reparse_paths else 0x00000080
                )
                value.volume_serial_number = observed.st_dev & 0xFFFFFFFF
                value.file_size_high = (size >> 32) & 0xFFFFFFFF
                value.file_size_low = size & 0xFFFFFFFF
                value.number_of_links = observed.st_nlink
                value.file_index_high = (observed.st_ino >> 32) & 0xFFFFFFFF
                value.file_index_low = observed.st_ino & 0xFFFFFFFF
                return 1

            def get_file_information_ex(
                self, handle: int, info_class: int, output: Any, size: int
            ) -> int:
                self.assert_api_shape(
                    info_class == 18
                    and size == ctypes.sizeof(CHECKER._WindowsFileIdInfo)
                )
                _, _, observed = self.file_entry(handle)
                value = ctypes.cast(
                    output, ctypes.POINTER(CHECKER._WindowsFileIdInfo)
                ).contents
                value.volume_serial_number = observed.st_dev or 1
                identifier = int(observed.st_ino or 1).to_bytes(16, "little")
                for index, byte in enumerate(identifier):
                    value.file_id.identifier[index] = byte
                return 1

            def get_file_size(self, handle: int, output: Any) -> int:
                _, _, observed = self.file_entry(handle)
                ctypes.cast(
                    output, ctypes.POINTER(ctypes.c_int64)
                ).contents.value = observed.st_size
                return 1

            def set_file_pointer(
                self, handle: int, distance: Any, new_position: Any, origin: int
            ) -> int:
                self.assert_api_shape(new_position is None and origin == 0)
                source, _, _ = self.file_entry(handle)
                offset = distance.value if hasattr(distance, "value") else distance
                source.seek(offset)
                return 1

            def read_file(
                self,
                handle: int,
                buffer: Any,
                requested: int,
                received: Any,
                overlapped: Any,
            ) -> int:
                self.assert_api_shape(overlapped is None and requested > 0)
                source, _, _ = self.file_entry(handle)
                chunk = source.read(requested)
                if chunk:
                    ctypes.memmove(buffer, chunk, len(chunk))
                ctypes.cast(
                    received, ctypes.POINTER(ctypes.c_uint32)
                ).contents.value = len(chunk)
                return 1

            def get_module_filename(
                self, handle: int, buffer: Any, capacity: int
            ) -> int:
                self.assert_api_shape(handle == self.module_handle)
                value = str(self.module_path)
                self.assert_api_shape(len(value) < capacity - 1)
                buffer.value = value
                return len(value)

            def get_proc_address(self, handle: int, symbol_bytes: bytes) -> int:
                self.assert_api_shape(handle == self.module_handle)
                symbol = symbol_bytes.decode("ascii")
                if symbol == self.missing_symbol:
                    return 0
                return self.export_addresses.get(symbol, 0)

            def get_module_handle_ex(
                self, flags: int, address: Any, output: Any
            ) -> int:
                self.assert_api_shape(flags == 0x00000006)
                address_value = address.value if hasattr(address, "value") else address
                owner = self.module_handle
                if (
                    self.forwarded_symbol is not None
                    and address_value == (self.export_addresses[self.forwarded_symbol])
                ):
                    owner = self.forwarder_handle
                ctypes.cast(
                    output, ctypes.POINTER(ctypes.c_void_p)
                ).contents.value = owner
                return 1

        windows_contracts: list[Any] = []
        active_cdll_instances: list[Any] = []
        windows_identity_events: list[str] = []

        def execute_windows_suite(
            suite: unittest.TestSuite,
            contract: Any,
            trusted: Any,
        ) -> Any:
            del suite, trusted
            windows_contracts.append(contract)
            self.assertTrue(callable(contract.runtime_integrity_callback))
            contract.runtime_integrity_callback()
            level2_sources = [
                binding.source_module
                for binding in contract.discovered_test_bindings
                if binding.source_module is not None
                and binding.source_module.name == "test_level2_report"
            ]
            self.assertTrue(level2_sources)
            self.assertTrue(
                all(
                    type(source.namespace.get("_TEST_BLAS_LIBRARY"))
                    is type(active_cdll_instances[0])
                    and source.namespace["_TEST_BLAS_LIBRARY"]
                    is active_cdll_instances[-1]
                    for source in level2_sources
                )
            )
            skips = (
                contract.required_decorator_skips
                | contract.permitted_dynamic_skips
                | contract.platform_skips
            )
            return CHECKER._PythonToolingOutcome(
                contract.discovered_count, True, skips, 0, 0, 0, 0
            )

        inventory_digest = hashlib.sha256(self.inventory_path.read_bytes()).hexdigest()
        original_registration = CHECKER._registered_python_tooling_modules
        windows_probes: dict[str, Any] = {}

        def reset_windows_libraries(contents: bytes = valid_windows_bytes) -> None:
            for path in (hardlink_library, windows_library, emitted_library):
                if path.is_symlink() or path.exists():
                    path.unlink()
            windows_library.write_bytes(contents)
            emitted_library.write_bytes(contents)

        def good_cdll_type(kernel32: WindowsKernel32) -> type[Any]:
            class WindowsZynumLibrary:
                def __init__(
                    self,
                    path: str,
                    *,
                    winmode: int | None = None,
                    **kwargs: Any,
                ) -> None:
                    self.assert_loader_shape(path, winmode, kwargs)
                    self._handle = kernel32.module_handle
                    windows_identity_events.append("cdll-load")
                    active_cdll_instances.append(self)

                def assert_loader_shape(
                    self, path: str, winmode: int | None, kwargs: dict[str, Any]
                ) -> None:
                    if (
                        Path(path).resolve(strict=True) != windows_library.resolve()
                        or winmode != 0x00000900
                        or kwargs
                    ):
                        raise AssertionError("noncanonical Windows CDLL load")

                def __getattr__(self, name: str) -> object:
                    if name not in CHECKER.WINDOWS_PYTHON_TOOLING_BLAS_REQUIRED_SYMBOLS:
                        raise AttributeError(name)
                    if name == kernel32.missing_symbol:
                        raise AttributeError(name)
                    return object()

            return WindowsZynumLibrary

        def run_windows_tooling(
            *,
            kernel32: WindowsKernel32 | None = None,
            cdll_type: type[Any] | None = None,
            emitted: Path | None = emitted_library,
            installed: Path | None = windows_library,
            execute_suite: Any = execute_windows_suite,
            registration: Any = None,
        ) -> Any:
            kernel32 = kernel32 or WindowsKernel32()
            active_cdll_instances.clear()
            windows_identity_events.clear()
            cdll_type = cdll_type or good_cdll_type(kernel32)
            find_probe = mock.Mock(
                side_effect=AssertionError("Windows must not use loader discovery")
            )
            suite_probe = mock.Mock(side_effect=execute_suite)

            class WindowsKernelLoader:
                def __new__(
                    cls, name: str, *, use_last_error: bool = False
                ) -> WindowsKernel32:
                    if name != "kernel32" or use_last_error is not True:
                        raise AssertionError("noncanonical Windows kernel load")
                    return kernel32

            windows_probes.clear()
            windows_probes.update(
                find_library=find_probe,
                execute_suite=suite_probe,
                kernel32=kernel32,
                cdll_type=cdll_type,
            )
            try:
                with contextlib.ExitStack() as stack:
                    stack.enter_context(
                        self._reviewed_digest_slots(inventory_current=inventory_digest)
                    )
                    stack.enter_context(
                        mock.patch.object(CHECKER.sys, "platform", "win32")
                    )
                    stack.enter_context(
                        mock.patch.object(
                            CHECKER.os, "getcwd", return_value=str(self.root)
                        )
                    )
                    stack.enter_context(
                        mock.patch.object(
                            CHECKER,
                            "_report_publication_platform_unavailable",
                            return_value=True,
                        )
                    )
                    stack.enter_context(
                        mock.patch.object(
                            CHECKER,
                            "_artifact_snapshot_platform_unavailable",
                            return_value=True,
                        )
                    )
                    stack.enter_context(
                        mock.patch("ctypes.util.find_library", new=find_probe)
                    )
                    stack.enter_context(mock.patch("ctypes.CDLL", new=cdll_type))
                    stack.enter_context(
                        mock.patch.object(
                            ctypes,
                            "WinDLL",
                            new=WindowsKernelLoader,
                            create=True,
                        )
                    )
                    stack.enter_context(
                        mock.patch.object(
                            CHECKER,
                            "_execute_python_tooling_suite",
                            new=suite_probe,
                        )
                    )
                    selected_registration = registration or original_registration

                    @contextlib.contextmanager
                    def observed_registration(reviewed_modules: Any) -> Any:
                        windows_identity_events.append("module-registration")
                        with selected_registration(reviewed_modules) as registry:
                            yield registry

                    stack.enter_context(
                        mock.patch.object(
                            CHECKER,
                            "_registered_python_tooling_modules",
                            new=observed_registration,
                        )
                    )
                    return CHECKER._run_python_tooling_root(
                        self.root,
                        self.inventory_path,
                        CHECKER.PYTHON_TOOLING_ROOT_ID,
                        emitted,
                        (
                            installed.resolve(strict=True)
                            if installed is not None
                            else None
                        ),
                    )
            finally:
                self.assertEqual({}, kernel32.open_files)

        reset_windows_libraries()
        windows_summary = run_windows_tooling()
        self.assertEqual(1, len(windows_contracts))
        windows_probes["find_library"].assert_not_called()
        self.assertEqual(2, len(active_cdll_instances))
        self.assertEqual(
            ["cdll-load", "module-registration", "cdll-load"],
            windows_identity_events,
        )
        windows_contract = windows_contracts[0]
        non_platform_pairs = (
            windows_contract.required_decorator_skips
            | windows_contract.permitted_dynamic_skips
        )
        non_platform_predicate_counts = Counter(
            entry["predicate_id"]
            for entry in self.inventory["python_skip_contracts"][0]["entries"]
            if (
                CHECKER._unittest_runtime_id(entry["test"]),
                entry["reason"],
            )
            in non_platform_pairs
        )
        self.assertEqual(
            Counter(
                {
                    "python-skip-predicate:accelerate-unavailable": 1,
                    "python-skip-predicate:rank-k-artifacts-unavailable": 1,
                    "python-skip-predicate:rotg-latency-artifacts-unavailable": 1,
                    "python-skip-predicate:symm-artifacts-unavailable": 1,
                    "python-skip-predicate:triangular-matrix-artifacts-unavailable": 1,
                }
            ),
            non_platform_predicate_counts,
        )
        self.assertEqual(
            0,
            sum(
                non_platform_predicate_counts[predicate_id]
                for predicate_id in (
                    "python-skip-predicate:drop-in-blas-unavailable",
                    "python-skip-predicate:file-backed-blas-unavailable",
                )
            ),
        )
        self.assertEqual(5, windows_summary.dynamic_skips)
        self.assertEqual(33, windows_summary.artifact_platform_skips)
        self.assertEqual(60, windows_summary.publication_platform_skips)
        self.assertEqual(93, windows_summary.platform_skips)
        self.assertEqual(98, len(windows_summary.outcome.skips))

        def registration_with_globals(values: dict[str, Any]) -> Any:
            @contextlib.contextmanager
            def registered(reviewed_modules: Any) -> Any:
                with original_registration(reviewed_modules) as registry:
                    level2 = next(
                        binding
                        for binding in registry
                        if binding.name == "test_level2_report"
                    )
                    level2.namespace.update(values)
                    yield registry

            return registered

        for description, globals_mutation in (
            (
                "alternate-globals",
                {
                    "TEST_BLAS": str(alternate_library.resolve()),
                    "TEST_FILE_BLAS": str(alternate_library.resolve()),
                },
            ),
            (
                "mismatched-globals",
                {"TEST_FILE_BLAS": str(alternate_library.resolve())},
            ),
        ):
            reset_windows_libraries()
            with (
                self.subTest(windows_blas_identity=description),
                self.assertRaises(CHECKER.InventoryError),
            ):
                run_windows_tooling(
                    registration=registration_with_globals(globals_mutation),
                )
            windows_probes["find_library"].assert_not_called()
            windows_probes["execute_suite"].assert_not_called()

        reset_windows_libraries()
        windows_library.unlink()
        windows_library.symlink_to(alternate_library)
        with (
            self.subTest(windows_blas_identity="symlink"),
            self.assertRaises(CHECKER.InventoryError),
        ):
            run_windows_tooling()
        windows_probes["find_library"].assert_not_called()

        reset_windows_libraries(b"")
        with (
            self.subTest(windows_blas_identity="empty"),
            self.assertRaises(CHECKER.InventoryError),
        ):
            run_windows_tooling()

        reset_windows_libraries()
        os.link(windows_library, hardlink_library)
        with (
            self.subTest(windows_blas_identity="hardlink"),
            self.assertRaisesRegex(CHECKER.InventoryError, "unique-link"),
        ):
            run_windows_tooling()

        reset_windows_libraries()
        emitted_library.write_bytes(b"different build-emitted DLL")
        with (
            self.subTest(windows_blas_identity="alternate-emitted-digest"),
            self.assertRaisesRegex(CHECKER.InventoryError, "emitted and installed"),
        ):
            run_windows_tooling()

        reset_windows_libraries()
        with (
            self.subTest(windows_blas_identity="explicit-installed-path-required"),
            self.assertRaisesRegex(CHECKER.InventoryError, "explicit.*installed"),
        ):
            run_windows_tooling(installed=None)

        reset_windows_libraries()
        with (
            self.subTest(windows_blas_identity="alternate-installed-path"),
            self.assertRaisesRegex(CHECKER.InventoryError, "path is noncanonical"),
        ):
            run_windows_tooling(installed=alternate_library)

        reset_windows_libraries()
        reparse_kernel = WindowsKernel32()
        reparse_kernel.reparse_paths.add(windows_library.resolve())
        with (
            self.subTest(windows_blas_identity="reparse"),
            self.assertRaisesRegex(CHECKER.InventoryError, "identity is noncanonical"),
        ):
            run_windows_tooling(kernel32=reparse_kernel)

        def invalid_cdll_type(
            kernel32: WindowsKernel32, handle: Any, *, proxy: bool = False
        ) -> type[Any]:
            class InvalidWindowsLibrary:
                def __new__(cls, *args: Any, **kwargs: Any) -> Any:
                    if proxy:
                        return types.SimpleNamespace(_handle=handle)
                    return super().__new__(cls)

                def __init__(self, *args: Any, **kwargs: Any) -> None:
                    del args, kwargs
                    self._handle = handle

                def __getattr__(self, name: str) -> object:
                    if name in CHECKER.WINDOWS_PYTHON_TOOLING_BLAS_REQUIRED_SYMBOLS:
                        return object()
                    raise AttributeError(name)

            return InvalidWindowsLibrary

        for description, handle, proxy in (
            ("fake-handle", object(), False),
            ("proxy-loader", 0x5000, True),
            ("zero-handle", 0, False),
        ):
            reset_windows_libraries()
            kernel32 = WindowsKernel32()
            with (
                self.subTest(windows_blas_identity=description),
                self.assertRaises(CHECKER.InventoryError),
            ):
                run_windows_tooling(
                    kernel32=kernel32,
                    cdll_type=invalid_cdll_type(kernel32, handle, proxy=proxy),
                )

        reset_windows_libraries()
        alternate_path_kernel = WindowsKernel32()
        alternate_path_kernel.module_path = alternate_library.resolve()
        with (
            self.subTest(windows_blas_identity="alternate-module-path"),
            self.assertRaisesRegex(CHECKER.InventoryError, "module path differs"),
        ):
            run_windows_tooling(kernel32=alternate_path_kernel)

        reset_windows_libraries()
        forwarder_kernel = WindowsKernel32()
        forwarder_kernel.forwarded_symbol = "ztbsv_"
        with (
            self.subTest(windows_blas_identity="forwarder-owner"),
            self.assertRaisesRegex(CHECKER.InventoryError, "foreign owner"),
        ):
            run_windows_tooling(kernel32=forwarder_kernel)

        reset_windows_libraries()
        missing_symbol_kernel = WindowsKernel32()
        missing_symbol_kernel.missing_symbol = "ztbsv_"
        with (
            self.subTest(windows_blas_identity="missing-required-symbol"),
            self.assertRaises(CHECKER.InventoryError),
        ):
            run_windows_tooling(kernel32=missing_symbol_kernel)

        def mutate_after_suite(
            suite: unittest.TestSuite,
            contract: Any,
            trusted: Any,
        ) -> Any:
            outcome = execute_windows_suite(suite, contract, trusted)
            level2 = next(
                binding.source_module
                for binding in contract.discovered_test_bindings
                if binding.source_module is not None
                and binding.source_module.name == "test_level2_report"
            )
            level2.namespace["TEST_FILE_BLAS"] = str(alternate_library.resolve())
            return outcome

        reset_windows_libraries()
        with (
            self.subTest(windows_blas_identity="unrestored-post-suite-mutation"),
            self.assertRaisesRegex(CHECKER.InventoryError, "changed during the suite"),
        ):
            run_windows_tooling(
                execute_suite=mutate_after_suite,
            )
        windows_probes["find_library"].assert_not_called()
        windows_probes["execute_suite"].assert_called_once()

        def mutate_handle_after_suite(
            suite: unittest.TestSuite, contract: Any, trusted: Any
        ) -> Any:
            outcome = execute_windows_suite(suite, contract, trusted)
            active_cdll_instances[0]._handle = 0x7777
            return outcome

        reset_windows_libraries()
        with (
            self.subTest(windows_blas_identity="module-handle-mutation"),
            self.assertRaisesRegex(CHECKER.InventoryError, "changed"),
        ):
            run_windows_tooling(execute_suite=mutate_handle_after_suite)

        def mutate_export_after_suite(
            suite: unittest.TestSuite, contract: Any, trusted: Any
        ) -> Any:
            outcome = execute_windows_suite(suite, contract, trusted)
            windows_probes["kernel32"].export_addresses["ztbsv_"] += 1
            return outcome

        reset_windows_libraries()
        with (
            self.subTest(windows_blas_identity="export-address-mutation"),
            self.assertRaisesRegex(CHECKER.InventoryError, "changed"),
        ):
            run_windows_tooling(execute_suite=mutate_export_after_suite)

        def replace_installed_after_suite(
            suite: unittest.TestSuite, contract: Any, trusted: Any
        ) -> Any:
            outcome = execute_windows_suite(suite, contract, trusted)
            replacement = windows_library.with_suffix(".replacement")
            replacement.write_bytes(b"replacement Windows DLL")
            os.replace(replacement, windows_library)
            return outcome

        reset_windows_libraries()
        with (
            self.subTest(windows_blas_identity="installed-replacement"),
            self.assertRaises(CHECKER.InventoryError),
        ):
            run_windows_tooling(execute_suite=replace_installed_after_suite)

        source_path = self.root / "bench/tools/test_level2_report.py"
        original_source = source_path.read_text(encoding="utf-8")
        for description, injection in (
            (
                "protected-global",
                '\ndef mutate_guarded_global():\n    global TEST_BLAS\n    TEST_BLAS = "proxy"\n',
            ),
            (
                "handle",
                "\ndef mutate_guarded_handle():\n    _TEST_BLAS_LIBRARY._handle = 0\n",
            ),
            (
                "loader",
                "\ndef mutate_guarded_loader():\n    ctypes.CDLL = object\n",
            ),
        ):
            with self.subTest(windows_blas_source_mutation=description):
                source_path.write_text(original_source + injection, encoding="utf-8")
                reviewed = CHECKER._reviewed_python_source_module(
                    self.root,
                    "bench/tools/test_level2_report.py",
                    "bench/tools",
                    "test_*.py",
                )
                tree = ast.parse(
                    reviewed.source_bytes.decode("utf-8"),
                    filename=reviewed.inventory_path,
                )
                with self.assertRaisesRegex(CHECKER.InventoryError, "Windows BLAS"):
                    CHECKER._python_windows_blas_source_audit(tree, reviewed)
        source_path.write_text(original_source, encoding="utf-8")

        cross_module_path = self.root / "bench/tools/test_level1_report.py"
        original_cross_module_source = cross_module_path.read_text(encoding="utf-8")
        for description, injection in (
            (
                "module-attribute-restore",
                "\nimport test_level2_report as guarded_level2\n"
                "def mutate_restore_guarded_global():\n"
                "    previous = guarded_level2.TEST_BLAS\n"
                '    guarded_level2.TEST_BLAS = "proxy"\n'
                "    guarded_level2.TEST_BLAS = previous\n",
            ),
            (
                "module-dict-restore",
                "\nimport test_level2_report as guarded_level2\n"
                "def mutate_restore_guarded_dict():\n"
                '    previous = guarded_level2.__dict__["TEST_FILE_BLAS"]\n'
                '    guarded_level2.__dict__["TEST_FILE_BLAS"] = "proxy"\n'
                '    guarded_level2.__dict__["TEST_FILE_BLAS"] = previous\n',
            ),
            (
                "vars-setattr-handle-restore",
                "\nfrom test_level2_report import _TEST_BLAS_LIBRARY as guarded_library\n"
                "def mutate_restore_guarded_handle():\n"
                "    previous = guarded_library._handle\n"
                '    setattr(guarded_library, "_handle", 0)\n'
                '    object.__setattr__(guarded_library, "_handle", previous)\n',
            ),
            (
                "ctypes-alias-restore",
                "\nimport ctypes as guarded_ctypes\n"
                "def mutate_restore_guarded_loader():\n"
                "    previous = guarded_ctypes.CDLL\n"
                "    guarded_ctypes.CDLL = object\n"
                "    guarded_ctypes.CDLL = previous\n",
            ),
            (
                "reflection-restore",
                "\nimport importlib\n"
                "def mutate_restore_reflected_global():\n"
                '    guarded = importlib.import_module("test_level2_report")\n'
                '    previous = vars(guarded)["TEST_BLAS"]\n'
                '    vars(guarded)["TEST_BLAS"] = "proxy"\n'
                '    vars(guarded)["TEST_BLAS"] = previous\n',
            ),
            (
                "dict-update-restore",
                "\nimport ctypes as guarded_ctypes\n"
                "from unittest import mock as guarded_mock\n"
                "def mutate_restore_loader_dict():\n"
                "    previous = guarded_ctypes.CDLL\n"
                "    with guarded_mock.patch.dict(\n"
                '        guarded_ctypes.__dict__, {"CDLL": object}\n'
                "    ):\n"
                "        pass\n"
                "    guarded_ctypes.CDLL = previous\n",
            ),
            (
                "default-setattr-test-body-restore",
                "\nimport test_level2_report as guarded_level2\n"
                "def test_mutate_restore_guarded_global(mutate=setattr):\n"
                "    previous = guarded_level2.TEST_BLAS\n"
                '    mutate(guarded_level2, "TEST_BLAS", "proxy")\n'
                '    mutate(guarded_level2, "TEST_BLAS", previous)\n',
            ),
            (
                "default-loader-add-cleanup-restore",
                "\nimport ctypes as guarded_ctypes\n"
                "def test_mutate_restore_guarded_loader(\n"
                "    self, mutate=setattr, original=guarded_ctypes.CDLL\n"
                "):\n"
                '    mutate(guarded_ctypes, "CDLL", object)\n'
                "    self.addCleanup(\n"
                '        mutate, guarded_ctypes, "CDLL", original\n'
                "    )\n",
            ),
            (
                "tuple-subscript-mutate-restore",
                "\nimport test_level2_report as guarded_level2\n"
                "def test_tuple_subscript_mutate_restore():\n"
                "    bundle = (setattr, guarded_level2)\n"
                "    mutate = bundle[0]\n"
                "    target = bundle[1]\n"
                "    previous = target.TEST_BLAS\n"
                '    mutate(target, "TEST_BLAS", "proxy")\n'
                '    mutate(target, "TEST_BLAS", previous)\n',
            ),
            (
                "returned-dict-subscript-mutate-restore",
                "\nimport test_level2_report as guarded_level2\n"
                "def guarded_mutation_bundle():\n"
                '    return {"mutate": setattr, "target": guarded_level2}\n'
                "def test_returned_dict_subscript_mutate_restore():\n"
                "    bundle = guarded_mutation_bundle()\n"
                '    previous = bundle["target"].TEST_BLAS\n'
                '    bundle["mutate"](\n'
                '        bundle["target"], "TEST_BLAS", "proxy"\n'
                "    )\n"
                '    bundle["mutate"](\n'
                '        bundle["target"], "TEST_BLAS", previous\n'
                "    )\n",
            ),
            (
                "starred-args-mutate-restore",
                "\nimport test_level2_report as guarded_level2\n"
                "def apply_guarded_args(*args):\n"
                "    mutate = args[0]\n"
                "    target = args[1]\n"
                "    mutate(target, args[2], args[3])\n"
                "def test_starred_args_mutate_restore():\n"
                "    previous = guarded_level2.TEST_BLAS\n"
                "    mutation = (\n"
                '        setattr, guarded_level2, "TEST_BLAS", "proxy"\n'
                "    )\n"
                "    restoration = (\n"
                '        setattr, guarded_level2, "TEST_BLAS", previous\n'
                "    )\n"
                "    apply_guarded_args(*mutation)\n"
                "    apply_guarded_args(*restoration)\n",
            ),
            (
                "expanded-kwargs-list-mutate-restore",
                "\nimport test_level2_report as guarded_level2\n"
                "def apply_guarded_kwargs(**kwargs):\n"
                '    kwargs["mutate"](\n'
                '        kwargs["target"], kwargs["name"], kwargs["value"]\n'
                "    )\n"
                "def test_expanded_kwargs_list_mutate_restore():\n"
                "    bundle = [setattr, guarded_level2]\n"
                "    previous = guarded_level2.TEST_BLAS\n"
                "    mutation = {\n"
                '        "mutate": bundle[0],\n'
                '        "target": bundle[1],\n'
                '        "name": "TEST_BLAS",\n'
                '        "value": "proxy",\n'
                "    }\n"
                "    restoration = {\n"
                '        "mutate": bundle[0],\n'
                '        "target": bundle[1],\n'
                '        "name": "TEST_BLAS",\n'
                '        "value": previous,\n'
                "    }\n"
                "    apply_guarded_kwargs(**mutation)\n"
                "    apply_guarded_kwargs(**restoration)\n",
            ),
            (
                "comprehension-projection",
                "\nimport test_level2_report as guarded_level2\n"
                "def guarded_comprehension_projection():\n"
                "    bundle = [\n"
                "        item for item in (setattr, guarded_level2)\n"
                "    ]\n"
                "    previous = bundle[1].TEST_BLAS\n"
                '    bundle[0](bundle[1], "TEST_BLAS", "proxy")\n'
                '    bundle[0](bundle[1], "TEST_BLAS", previous)\n',
            ),
            (
                "object-attribute",
                "\nimport test_level2_report as guarded_level2\n"
                "class GuardedMutationHolder:\n"
                "    pass\n"
                "def guarded_object_attribute():\n"
                "    holder = GuardedMutationHolder()\n"
                "    holder.mutate = setattr\n"
                "    holder.target = guarded_level2\n"
                "    previous = holder.target.TEST_BLAS\n"
                '    holder.mutate(holder.target, "TEST_BLAS", "proxy")\n'
                '    holder.mutate(holder.target, "TEST_BLAS", previous)\n',
            ),
            (
                "staticmethod-return",
                "\nimport test_level2_report as guarded_level2\n"
                "class GuardedStaticFactory:\n"
                "    @staticmethod\n"
                "    def capability():\n"
                "        return setattr\n"
                "def guarded_staticmethod_return():\n"
                "    previous = guarded_level2.TEST_BLAS\n"
                "    mutate = GuardedStaticFactory.capability()\n"
                '    mutate(guarded_level2, "TEST_BLAS", "proxy")\n'
                '    mutate(guarded_level2, "TEST_BLAS", previous)\n',
            ),
            (
                "generator-yield",
                "\nimport test_level2_report as guarded_level2\n"
                "def guarded_generator_values():\n"
                "    yield setattr\n"
                "    yield guarded_level2\n"
                "def guarded_generator_restore():\n"
                "    bundle = tuple(guarded_generator_values())\n"
                "    previous = bundle[1].TEST_BLAS\n"
                '    bundle[0](bundle[1], "TEST_BLAS", "proxy")\n'
                '    bundle[0](bundle[1], "TEST_BLAS", previous)\n',
            ),
            (
                "lambda-parameters",
                "\nimport test_level2_report as guarded_level2\n"
                "def guarded_lambda_restore():\n"
                "    previous = guarded_level2.TEST_BLAS\n"
                "    apply = lambda mutate, target, value: mutate(\n"
                '        target, "TEST_BLAS", value\n'
                "    )\n"
                '    apply(setattr, guarded_level2, "proxy")\n'
                "    apply(setattr, guarded_level2, previous)\n",
            ),
            (
                "functools-partial",
                "\nimport functools\n"
                "import test_level2_report as guarded_level2\n"
                "def guarded_partial_restore():\n"
                "    previous = guarded_level2.TEST_BLAS\n"
                "    mutate = functools.partial(\n"
                '        setattr, guarded_level2, "TEST_BLAS"\n'
                "    )\n"
                '    mutate("proxy")\n'
                "    mutate(previous)\n",
            ),
            (
                "operator-setitem",
                "\nimport operator\n"
                "import test_level2_report as guarded_level2\n"
                "def guarded_operator_restore():\n"
                "    previous = guarded_level2.TEST_BLAS\n"
                "    operator.setitem(\n"
                '        guarded_level2.__dict__, "TEST_BLAS", "proxy"\n'
                "    )\n"
                "    operator.setitem(\n"
                '        guarded_level2.__dict__, "TEST_BLAS", previous\n'
                "    )\n",
            ),
            (
                "with-binding",
                "\nimport contextlib\n"
                "import test_level2_report as guarded_level2\n"
                "def guarded_with_binding():\n"
                "    with contextlib.nullcontext(\n"
                "        (setattr, guarded_level2)\n"
                "    ) as bundle:\n"
                "        previous = bundle[1].TEST_BLAS\n"
                '        bundle[0](bundle[1], "TEST_BLAS", "proxy")\n'
                '        bundle[0](bundle[1], "TEST_BLAS", previous)\n',
            ),
            (
                "decorator-capability",
                "\nimport functools\n"
                "import test_level2_report as guarded_level2\n"
                "@functools.partial(\n"
                '    setattr, guarded_level2, "TEST_BLAS", "proxy"\n'
                ")\n"
                "def guarded_decorator_capability():\n"
                "    pass\n",
            ),
            (
                "for-binding",
                "\nimport test_level2_report as guarded_level2\n"
                "def guarded_for_binding():\n"
                "    previous = guarded_level2.TEST_BLAS\n"
                "    for mutate, target in [(setattr, guarded_level2)]:\n"
                '        mutate(target, "TEST_BLAS", "proxy")\n'
                '        mutate(target, "TEST_BLAS", previous)\n',
            ),
            (
                "match-pattern-binding",
                "\nimport test_level2_report as guarded_level2\n"
                "def guarded_match_binding():\n"
                "    previous = guarded_level2.TEST_BLAS\n"
                "    match (setattr, guarded_level2):\n"
                "        case (mutate, target):\n"
                '            mutate(target, "TEST_BLAS", "proxy")\n'
                '            mutate(target, "TEST_BLAS", previous)\n',
            ),
            (
                "dynamic-attribute-name",
                "\nimport test_level2_report as guarded_level2\n"
                'def guarded_dynamic_name(name="TEST_BLAS"):\n'
                "    return getattr(guarded_level2, name)\n",
            ),
            (
                "deferred-cleanup-combination",
                "\nimport test_level2_report as guarded_level2\n"
                "def guarded_deferred_cleanup(self):\n"
                "    previous = guarded_level2.TEST_BLAS\n"
                "    cleanup = (\n"
                '        setattr, guarded_level2, "TEST_BLAS", previous\n'
                "    )\n"
                '    setattr(guarded_level2, "TEST_BLAS", "proxy")\n'
                "    self.addCleanup(*cleanup)\n",
            ),
            (
                "returned-nested-callables",
                "\nimport test_level2_report as guarded_level2\n"
                "def guarded_mutation_factory():\n"
                "    def acquire_mutation():\n"
                "        return setattr\n"
                "    return acquire_mutation\n"
                "def guarded_target_factory():\n"
                "    def acquire_target():\n"
                "        return guarded_level2\n"
                "    return acquire_target\n"
                "def guarded_returned_callable_restore():\n"
                "    mutation_callback = guarded_mutation_factory()\n"
                "    target_callback = guarded_target_factory()\n"
                "    mutate = mutation_callback()\n"
                "    target = target_callback()\n"
                "    previous = target.TEST_BLAS\n"
                '    mutate(target, "TEST_BLAS", "proxy")\n'
                '    mutate(target, "TEST_BLAS", previous)\n',
            ),
            (
                "returned-callable-container",
                "\nimport test_level2_report as guarded_level2\n"
                "def guarded_container_factory():\n"
                "    def acquire_mutation():\n"
                "        return setattr\n"
                "    def acquire_target():\n"
                "        return guarded_level2\n"
                '    return {"callbacks": (acquire_mutation, acquire_target)}\n'
                "def guarded_container_callable_restore():\n"
                "    callbacks = guarded_container_factory()\n"
                '    mutate = callbacks["callbacks"][0]()\n'
                '    target = callbacks["callbacks"][1]()\n'
                "    previous = target.TEST_BLAS\n"
                '    mutate(target, "TEST_BLAS", "proxy")\n'
                '    mutate(target, "TEST_BLAS", previous)\n',
            ),
            (
                "returned-method-and-lambda-callables",
                "\nimport test_level2_report as guarded_level2\n"
                "class GuardedCallableFactory:\n"
                "    @staticmethod\n"
                "    def mutation():\n"
                "        def acquire():\n"
                "            return setattr\n"
                "        return acquire\n"
                "    def target(self):\n"
                "        return lambda: guarded_level2\n"
                "def guarded_method_callable_restore():\n"
                "    factory = GuardedCallableFactory()\n"
                "    mutation_callback = GuardedCallableFactory.mutation()\n"
                "    target_callback = factory.target()\n"
                "    mutate = mutation_callback()\n"
                "    target = target_callback()\n"
                "    previous = target.TEST_BLAS\n"
                '    mutate(target, "TEST_BLAS", "proxy")\n'
                '    mutate(target, "TEST_BLAS", previous)\n',
            ),
            (
                "class-body-execution",
                "\nimport test_level2_report as guarded_level2\n"
                "class GuardedClassBodyMutation:\n"
                "    mutate = setattr\n"
                "    target = guarded_level2\n"
                "    previous = target.TEST_BLAS\n"
                '    mutate(target, "TEST_BLAS", "proxy")\n'
                '    mutate(target, "TEST_BLAS", previous)\n',
            ),
            (
                "class-definition-authority",
                "\nguarded_authority = eval\n"
                "class GuardedAuthorityBase(guarded_authority, "
                "metaclass=guarded_authority):\n"
                "    pass\n",
            ),
            (
                "function-annotation-authority",
                "\nguarded_authority = eval\n"
                "def guarded_annotation[TypeParameter: guarded_authority](\n"
                "    value: guarded_authority,\n"
                ") -> guarded_authority:\n"
                "    return value\n",
            ),
            (
                "match-class-authority",
                "\ndef guarded_match_authority(value):\n"
                "    match value:\n"
                "        case object(__globals__=namespace):\n"
                "            return namespace\n",
            ),
            (
                "static-reflection-authority",
                "\ndef guarded_reflection_authority():\n"
                '    return getattr(object, "__subclasses__")\n',
            ),
            (
                "annotated-assignment-authority",
                "\nguarded_authority = eval\nguarded_annotation: guarded_authority\n",
            ),
            (
                "type-alias-authority",
                "\nguarded_authority = eval\n"
                "type GuardedAlias[TypeParameter: guarded_authority] = "
                "guarded_authority\n",
            ),
            (
                "returned-callable-expression-forms",
                "\nimport test_level2_report as guarded_level2\n"
                "from external import passthrough as guarded_passthrough\n"
                "def guarded_expression_mutation_factory():\n"
                "    def acquire():\n"
                "        return setattr\n"
                "    return acquire\n"
                "def guarded_expression_target_factory():\n"
                "    def acquire():\n"
                "        return guarded_level2\n"
                "    return acquire\n"
                "async def guarded_expression_callable_restore():\n"
                "    mutation_callback = await guarded_passthrough(\n"
                "        None or guarded_expression_mutation_factory()\n"
                "    )\n"
                "    target_callback = await guarded_passthrough(\n"
                "        None or guarded_expression_target_factory()\n"
                "    )\n"
                "    mutate = mutation_callback()\n"
                "    target = target_callback()\n"
                "    previous = target.TEST_BLAS\n"
                '    mutate(target, "TEST_BLAS", "proxy")\n'
                '    mutate(target, "TEST_BLAS", previous)\n',
            ),
            (
                "returned-callable-defaults",
                "\nimport test_level2_report as guarded_level2\n"
                "def guarded_default_mutation_factory():\n"
                "    def acquire():\n"
                "        return setattr\n"
                "    return acquire\n"
                "def guarded_default_target_factory():\n"
                "    def acquire():\n"
                "        return guarded_level2\n"
                "    return acquire\n"
                "def guarded_default_callable_restore(\n"
                "    mutation_callback=guarded_default_mutation_factory(),\n"
                "    target_callback=guarded_default_target_factory(),\n"
                "):\n"
                "    mutate = mutation_callback()\n"
                "    target = target_callback()\n"
                "    previous = target.TEST_BLAS\n"
                '    mutate(target, "TEST_BLAS", "proxy")\n'
                '    mutate(target, "TEST_BLAS", previous)\n',
            ),
            (
                "returned-callable-comprehension-match",
                "\nimport test_level2_report as guarded_level2\n"
                "def guarded_match_mutation_factory():\n"
                "    def acquire():\n"
                "        return setattr\n"
                "    return acquire\n"
                "def guarded_match_target_factory():\n"
                "    def acquire():\n"
                "        return guarded_level2\n"
                "    return acquire\n"
                "def guarded_comprehension_match_restore():\n"
                "    callbacks = [item for item in (\n"
                "        guarded_match_mutation_factory(),\n"
                "        guarded_match_target_factory(),\n"
                "    )]\n"
                "    match callbacks:\n"
                "        case (mutation_callback, target_callback):\n"
                "            mutate = mutation_callback()\n"
                "            target = target_callback()\n"
                "            previous = target.TEST_BLAS\n"
                '            mutate(target, "TEST_BLAS", "proxy")\n'
                '            mutate(target, "TEST_BLAS", previous)\n',
            ),
            (
                "returned-callable-binop-generator",
                "\nimport test_level2_report as guarded_level2\n"
                "from external import passthrough as guarded_passthrough\n"
                "def guarded_binary_mutation_factory():\n"
                "    def acquire():\n"
                "        return setattr\n"
                "    return acquire\n"
                "def guarded_binary_target_factory():\n"
                "    def acquire():\n"
                "        return guarded_level2\n"
                "    return acquire\n"
                "def guarded_binary_generator_restore():\n"
                "    callbacks = guarded_passthrough(\n"
                "        item for item in (\n"
                "            guarded_binary_mutation_factory()\n"
                "            + guarded_binary_target_factory(),\n"
                "        )\n"
                "    )\n"
                "    mutation_callback, target_callback = callbacks\n"
                "    mutate = mutation_callback()\n"
                "    target = target_callback()\n"
                "    previous = target.TEST_BLAS\n"
                '    mutate(target, "TEST_BLAS", "proxy")\n'
                '    mutate(target, "TEST_BLAS", previous)\n',
            ),
            (
                "returned-callable-for-binding",
                "\ndef guarded_for_callable_factory():\n"
                "    def callback():\n"
                "        return setattr\n"
                "    return callback\n"
                "def guarded_for_callable_escape():\n"
                "    for callback in (guarded_for_callable_factory(),):\n"
                "        acquired = callback()\n"
                "        acquired(acquired)\n",
            ),
            (
                "returned-callable-with-binding",
                "\nfrom external import passthrough as guarded_passthrough\n"
                "def guarded_with_callable_factory():\n"
                "    def callback():\n"
                "        return setattr\n"
                "    return callback\n"
                "def guarded_with_callable_escape():\n"
                "    with guarded_passthrough(\n"
                "        guarded_with_callable_factory()\n"
                "    ) as callback:\n"
                "        acquired = callback()\n"
                "        acquired(acquired)\n",
            ),
            (
                "returned-callable-indirect-target",
                "\ndef guarded_indirect_factory():\n"
                "    def callback():\n"
                "        return setattr\n"
                "    return callback\n"
                "class GuardedIndirectHolder:\n"
                "    pass\n"
                "guarded_holder = GuardedIndirectHolder()\n"
                "guarded_holder.callback = guarded_indirect_factory()\n",
            ),
            (
                "positional-match-class-authority",
                "\ndef guarded_positional_match_authority(value):\n"
                "    match value:\n"
                "        case object(namespace):\n"
                "            return namespace\n",
            ),
            (
                "traceback-closure-authority",
                "\ndef guarded_traceback_closure_authority(error, function):\n"
                "    return (\n"
                "        error.__traceback__.tb_frame\n"
                "        or function.__closure__[0].cell_contents\n"
                "    )\n",
            ),
            (
                "builtins-getattr-authority",
                "\nimport builtins\n"
                "guarded_runtime_callable = builtins.getattr(\n"
                '    builtins, "eval"\n'
                ")\n",
            ),
            (
                "builtins-vars-authority",
                "\nimport builtins\n"
                "guarded_runtime_namespace = builtins.vars(builtins)\n",
            ),
            (
                "augmented-callable-binding",
                "\ndef guarded_augmented_factory():\n"
                "    def callback():\n"
                "        return setattr\n"
                "    return callback\n"
                "def guarded_augmented_escape():\n"
                "    callbacks = ()\n"
                "    callbacks += (guarded_augmented_factory(),)\n"
                "    acquired = callbacks[0]()\n"
                "    acquired(acquired)\n",
            ),
            (
                "authority-wildcard-import",
                "\nfrom builtins import *\n",
            ),
            (
                "dict-key-value-callable-lookup",
                "\ndef guarded_dict_callable_factory():\n"
                "    def callback():\n"
                "        return setattr\n"
                "    return callback\n"
                "def guarded_dict_callable_escape():\n"
                "    callbacks = {\n"
                "        guarded_dict_callable_factory(): "
                "guarded_dict_callable_factory()\n"
                "    }\n"
                '    callback = callbacks.get("callback")\n'
                "    acquired = callback()\n"
                "    acquired(acquired)\n",
            ),
            (
                "dangerous-class-indirect-escape",
                "\nclass GuardedDangerousClass:\n"
                "    def expose(self):\n"
                "        return setattr\n"
                "def guarded_dangerous_class_factory():\n"
                "    return GuardedDangerousClass\n"
                "class GuardedClassHolder:\n"
                "    pass\n"
                "guarded_class_holder = GuardedClassHolder()\n"
                "guarded_class_holder.kind = "
                "guarded_dangerous_class_factory()\n",
            ),
            (
                "iteration-protocol-callable",
                "\nclass GuardedIterationCarrier:\n"
                "    def __iter__(self):\n"
                "        return self\n"
                "    def __next__(self):\n"
                "        def acquire():\n"
                "            return setattr\n"
                "        return acquire\n"
                "def guarded_iteration_protocol_escape():\n"
                "    for callback in GuardedIterationCarrier():\n"
                "        acquired = callback()\n"
                "        acquired(acquired)\n",
            ),
            (
                "context-protocol-callable",
                "\nclass GuardedContextCarrier:\n"
                "    def __enter__(self):\n"
                "        def acquire():\n"
                "            return setattr\n"
                "        return acquire\n"
                "    def __exit__(self, *args):\n"
                "        return False\n"
                "def guarded_context_protocol_escape():\n"
                "    with GuardedContextCarrier() as callback:\n"
                "        acquired = callback()\n"
                "        acquired(acquired)\n",
            ),
            (
                "await-protocol-callable",
                "\nclass GuardedAwaitCarrier:\n"
                "    def __await__(self):\n"
                "        def acquire():\n"
                "            return setattr\n"
                "        return acquire\n"
                "async def guarded_await_protocol_escape():\n"
                "    callback = await GuardedAwaitCarrier()\n"
                "    acquired = callback()\n"
                "    acquired(acquired)\n",
            ),
            (
                "expression-protocol-callable",
                "\nclass GuardedExpressionCarrier:\n"
                "    def __add__(self, other):\n"
                "        def acquire():\n"
                "            return setattr\n"
                "        return acquire\n"
                "    __eq__ = __add__\n"
                "    __getitem__ = __add__\n"
                "    __format__ = __add__\n"
                "def guarded_expression_protocol_escape():\n"
                "    carrier = GuardedExpressionCarrier()\n"
                "    callbacks = (\n"
                "        carrier + carrier,\n"
                "        carrier == carrier,\n"
                "        carrier[0],\n"
                '        f"{carrier}",\n'
                "    )\n"
                "    acquired = callbacks[0]()\n"
                "    acquired(acquired)\n",
            ),
            (
                "equivalent-authority-attribute",
                "\nclass GuardedAuthorityTool:\n"
                "    pass\n"
                "def guarded_equivalent_authority(tool):\n"
                "    return tool.get_objects\n",
            ),
            (
                "bound-callable-self-authority",
                "\ndef guarded_bound_self_authority():\n"
                '    return getattr(len, "__self__")\n',
            ),
            (
                "constructor-protocol-callable",
                "\nclass GuardedConstructorCarrier:\n"
                "    def __new__(cls, callback):\n"
                "        return callback\n"
                "def guarded_constructor_escape():\n"
                "    acquired = GuardedConstructorCarrier(setattr)\n"
                "    acquired(acquired)\n",
            ),
            (
                "constructor-init-provenance",
                "\nimport test_level2_report as guarded_level2\n"
                "class GuardedInitCarrier:\n"
                "    def __init__(self, mutate, target):\n"
                '        mutate(target, "TEST_BLAS", "proxy")\n'
                "def guarded_init_escape():\n"
                "    GuardedInitCarrier(setattr, guarded_level2)\n",
            ),
            (
                "callable-instance-class-symbol",
                "\nclass GuardedCallableInstance:\n"
                "    def __call__(self):\n"
                "        return setattr\n"
                "def guarded_class_symbol_factory():\n"
                "    return GuardedCallableInstance\n"
                "def guarded_callable_instance_escape():\n"
                "    kind = guarded_class_symbol_factory()\n"
                "    instance = kind()\n"
                "    acquired = instance()\n"
                "    acquired(acquired)\n",
            ),
            (
                "unary-protocol-callable",
                "\nclass GuardedUnaryCarrier:\n"
                "    def __neg__(self):\n"
                "        def callback():\n"
                "            return setattr\n"
                "        return callback\n"
                "def guarded_unary_escape():\n"
                "    callback = -GuardedUnaryCarrier()\n"
                "    acquired = callback()\n"
                "    acquired(acquired)\n",
            ),
            (
                "descriptor-protocol-callable",
                "\nclass GuardedDescriptor:\n"
                "    def __get__(self, instance, owner):\n"
                "        def callback():\n"
                "            return setattr\n"
                "        return callback\n"
                "class GuardedDescriptorHolder:\n"
                "    field = GuardedDescriptor()\n"
                "def guarded_descriptor_escape():\n"
                "    callback = GuardedDescriptorHolder().field\n"
                "    acquired = callback()\n"
                "    acquired(acquired)\n",
            ),
            (
                "two-stage-iteration-protocol",
                "\nclass GuardedIterator:\n"
                "    def __next__(self):\n"
                "        def callback():\n"
                "            return setattr\n"
                "        return callback\n"
                "class GuardedIterable:\n"
                "    def __iter__(self):\n"
                "        return GuardedIterator()\n"
                "def guarded_iteration_chain_escape():\n"
                "    for callback in GuardedIterable():\n"
                "        acquired = callback()\n"
                "        acquired(acquired)\n",
            ),
            (
                "two-stage-await-protocol",
                "\nclass GuardedAwaitIterator:\n"
                "    def __next__(self):\n"
                "        def callback():\n"
                "            return setattr\n"
                "        return callback\n"
                "class GuardedAwaitable:\n"
                "    def __await__(self):\n"
                "        return GuardedAwaitIterator()\n"
                "async def guarded_await_chain_escape():\n"
                "    callback = await GuardedAwaitable()\n"
                "    acquired = callback()\n"
                "    acquired(acquired)\n",
            ),
            (
                "runtime-authority-module-import",
                "\nimport gc\n"
                "def guarded_runtime_module_import():\n"
                "    return gc.get_objects()\n",
            ),
            (
                "noncanonical-runtime-authority-use",
                "\nimport sys\n"
                "def guarded_noncanonical_sys_use():\n"
                "    return sys._getframe()\n",
            ),
            (
                "noncanonical-importlib-use",
                "\nimport importlib.util\n"
                "def guarded_noncanonical_importlib_use():\n"
                "    return importlib.util.module_from_spec(None)\n",
            ),
            (
                "explicit-bound-dunder-call",
                "\ndef guarded_dunder_factory():\n"
                "    return setattr\n"
                "def guarded_explicit_dunder_call():\n"
                "    acquired = guarded_dunder_factory.__call__()\n"
                "    acquired(acquired)\n",
            ),
            (
                "bound-method-dunder-call",
                "\nclass GuardedBoundMethodCarrier:\n"
                "    def method(self):\n"
                "        return setattr\n"
                "def guarded_bound_method_dunder_call():\n"
                "    bound = GuardedBoundMethodCarrier().method\n"
                "    acquired = bound.__call__()\n"
                "    acquired(acquired)\n",
            ),
            (
                "inherited-unary-protocol",
                "\nclass GuardedUnaryBase:\n"
                "    def __neg__(self):\n"
                "        def callback():\n"
                "            return setattr\n"
                "        return callback\n"
                "class GuardedUnaryChild(GuardedUnaryBase):\n"
                "    pass\n"
                "def guarded_inherited_unary_escape():\n"
                "    callback = -GuardedUnaryChild()\n"
                "    acquired = callback()\n"
                "    acquired(acquired)\n",
            ),
            (
                "inherited-property-protocol",
                "\nclass GuardedPropertyBase:\n"
                "    @property\n"
                "    def field(self):\n"
                "        def callback():\n"
                "            return setattr\n"
                "        return callback\n"
                "class GuardedPropertyChild(GuardedPropertyBase):\n"
                "    pass\n"
                "def guarded_inherited_property_escape():\n"
                "    callback = GuardedPropertyChild().field\n"
                "    acquired = callback()\n"
                "    acquired(acquired)\n",
            ),
            (
                "legacy-getitem-iteration",
                "\nclass GuardedLegacyIterable:\n"
                "    def __getitem__(self, index):\n"
                "        def callback():\n"
                "            return setattr\n"
                "        return callback\n"
                "def guarded_legacy_iteration_escape():\n"
                "    for callback in GuardedLegacyIterable():\n"
                "        acquired = callback()\n"
                "        acquired(acquired)\n",
            ),
            (
                "builtin-next-adapter",
                "\nclass GuardedNextIterator:\n"
                "    def __next__(self):\n"
                "        return setattr\n"
                "def guarded_builtin_next_escape():\n"
                "    acquired = next(GuardedNextIterator())\n"
                "    acquired(acquired)\n",
            ),
            (
                "await-return-channel",
                "\nclass GuardedReturnAwaitable:\n"
                "    def __await__(self):\n"
                "        if False:\n"
                "            yield None\n"
                "        return setattr\n"
                "async def guarded_await_return_escape():\n"
                "    acquired = await GuardedReturnAwaitable()\n"
                "    acquired(acquired)\n",
            ),
            (
                "authority-import-alias",
                "\nimport sys as guarded_runtime\n"
                "def guarded_authority_import_alias():\n"
                "    return guarded_runtime.argv\n",
            ),
            (
                "transitive-authority-import",
                "\nfrom package import sys as guarded_runtime\n",
            ),
            (
                "decorator-aware-method-offsets",
                "\nclass GuardedMethodOffsets:\n"
                "    @staticmethod\n"
                "    def static_apply(callback):\n"
                "        callback(callback)\n"
                "    @classmethod\n"
                "    def class_apply(owner, callback):\n"
                "        callback(callback)\n"
                "    def instance_apply(receiver, callback):\n"
                "        callback(callback)\n"
                "def guarded_method_offset_escape():\n"
                "    GuardedMethodOffsets.static_apply(setattr)\n"
                "    GuardedMethodOffsets.class_apply(setattr)\n"
                "    GuardedMethodOffsets.instance_apply(\n"
                "        GuardedMethodOffsets(), setattr\n"
                "    )\n",
            ),
            (
                "container-mutator-writeback",
                "\ndef guarded_mutator_factory():\n"
                "    def callback():\n"
                "        return setattr\n"
                "    return callback\n"
                "def guarded_container_mutator_escape():\n"
                "    callbacks = []\n"
                "    callbacks.append(guarded_mutator_factory())\n"
                "    acquired = callbacks[0]()\n"
                "    acquired(acquired)\n",
            ),
            (
                "nonlocal-rooted-escape",
                "\ndef guarded_nonlocal_outer():\n"
                "    capability = None\n"
                "    def guarded_nonlocal_inner():\n"
                "        nonlocal capability\n"
                "        capability = setattr\n",
            ),
            (
                "global-rooted-escape",
                "\nimport test_level2_report as guarded_level2\n"
                "def guarded_global_escape():\n"
                "    global guarded_global_target\n"
                "    guarded_global_target = guarded_level2\n",
            ),
            (
                "attribute-base-protocol",
                "\nclass GuardedOuterBase:\n"
                "    class Base:\n"
                "        def __neg__(self):\n"
                "            return setattr\n"
                "class GuardedAttributeBaseChild(GuardedOuterBase.Base):\n"
                "    pass\n"
                "def guarded_attribute_base_escape():\n"
                "    acquired = -GuardedAttributeBaseChild()\n"
                "    acquired(acquired)\n",
            ),
            (
                "dynamic-class-base",
                "\ndef guarded_choose_base():\n"
                "    return object\n"
                "class GuardedDynamicBase(guarded_choose_base()):\n"
                "    pass\n",
            ),
            (
                "dynamic-metaclass",
                "\nclass GuardedDynamicMetaclass(metaclass=type):\n    pass\n",
            ),
            (
                "aliased-staticmethod-decorator",
                "\nguarded_static = staticmethod\n"
                "class GuardedAliasedStaticMethod:\n"
                "    @guarded_static\n"
                "    def apply(callback):\n"
                "        callback(callback)\n"
                "def guarded_aliased_staticmethod_escape():\n"
                "    GuardedAliasedStaticMethod.apply(setattr)\n",
            ),
            (
                "unknown-dangerous-decorator",
                "\ndef guarded_decorator(function):\n"
                "    return function\n"
                "class GuardedUnknownDecorator:\n"
                "    @guarded_decorator\n"
                "    def field(self):\n"
                "        return setattr\n",
            ),
            (
                "iter-callable-sentinel",
                "\ndef guarded_iter_producer():\n"
                "    return setattr\n"
                "def guarded_iter_sentinel_escape():\n"
                "    acquired = next(iter(guarded_iter_producer, None))\n"
                "    acquired(acquired)\n",
            ),
            (
                "builtin-iter-getitem-fallback",
                "\nclass GuardedBuiltinLegacySequence:\n"
                "    def __getitem__(self, index):\n"
                "        return setattr\n"
                "def guarded_builtin_legacy_escape():\n"
                "    acquired = next(iter(GuardedBuiltinLegacySequence()))\n"
                "    acquired(acquired)\n",
            ),
            (
                "noncanonical-import-shape",
                "\nimport collections\n",
            ),
            (
                "global-function-definition-binding",
                "\ndef guarded_outer_definition():\n"
                "    global guarded_escaped_definition\n"
                "    def guarded_escaped_definition():\n"
                "        return setattr\n",
            ),
            (
                "global-class-definition-binding",
                "\ndef guarded_outer_class_definition():\n"
                "    global GuardedEscapedClassDefinition\n"
                "    class GuardedEscapedClassDefinition:\n"
                "        pass\n",
            ),
            (
                "global-import-binding",
                "\ndef guarded_outer_import():\n    global sys\n    import sys\n",
            ),
            (
                "global-match-capture-binding",
                "\ndef guarded_match_global():\n"
                "    global guarded_match_escape\n"
                "    match setattr:\n"
                "        case guarded_match_escape:\n"
                "            pass\n",
            ),
            (
                "temporary-container-mutator",
                "\ndef guarded_container_factory():\n"
                "    return []\n"
                "def guarded_temporary_mutator():\n"
                "    guarded_container_factory().append(setattr)\n",
            ),
            (
                "meta-execution-eval",
                '\ndef guarded_meta_execution():\n    return eval("setattr")\n',
            ),
            (
                "meta-alias-exec-restore",
                "\ndef guarded_safe_run(source):\n"
                "    return source\n"
                "def guarded_exec_alias_restore():\n"
                "    run = guarded_safe_run\n"
                "    original = run\n"
                "    run = exec\n"
                '    run("pass")\n'
                "    run = original\n",
            ),
            (
                "meta-alias-eval",
                "\ndef guarded_eval_alias():\n"
                "    run = eval\n"
                '    return run("1 + 1")\n',
            ),
            (
                "meta-from-builtins-eval",
                "\nfrom builtins import eval as guarded_run\n"
                "def guarded_builtins_eval_alias():\n"
                '    return guarded_run("1 + 1")\n',
            ),
            (
                "meta-builtins-exec",
                "\nimport builtins\n"
                "def guarded_builtins_exec():\n"
                '    builtins.exec("pass")\n',
            ),
            (
                "meta-partial-eval",
                "\nimport functools\n"
                "def guarded_partial_eval():\n"
                '    return functools.partial(eval, "1 + 1")()\n',
            ),
            (
                "meta-gc-alias",
                "\nimport gc\n"
                "def guarded_gc_alias():\n"
                "    run = gc.get_objects\n"
                "    return run()\n",
            ),
            (
                "meta-frame-alias",
                "\nfrom sys import _getframe as guarded_frame\n"
                "def guarded_frame_alias():\n"
                "    run = guarded_frame\n"
                "    return run()\n",
            ),
            (
                "meta-sys-modules-alias",
                "\nfrom sys import modules as guarded_modules\n"
                "def guarded_modules_alias(name):\n"
                "    return guarded_modules[name]\n",
            ),
            (
                "meta-canonical-import-alias",
                "\nfrom importlib import import_module as guarded_lookup\n"
                "def guarded_import_alias(name):\n"
                "    return guarded_lookup(name)\n",
            ),
            (
                "meta-builtins-reflection-alias",
                "\nfrom builtins import getattr as guarded_lookup\n"
                "def guarded_reflection_alias(owner, name):\n"
                "    return guarded_lookup(owner, name)\n",
            ),
            (
                "meta-implicit-runtime-namespace",
                "\ndef guarded_implicit_namespace():\n    return __builtins__\n",
            ),
            (
                "meta-function-runtime-namespace",
                "\ndef guarded_namespace_subject():\n"
                "    pass\n"
                "def guarded_function_namespace():\n"
                "    return guarded_namespace_subject.__builtins__\n",
            ),
            (
                "meta-builtins-dict-namespace",
                "\nimport builtins\n"
                "def guarded_builtins_namespace():\n"
                "    return builtins.__dict__\n",
            ),
            (
                "meta-dynamic-namespace-key",
                "\ndef guarded_namespace_key(name):\n    return __builtins__[name]\n",
            ),
            (
                "meta-lookup-return-provenance",
                "\nimport builtins\n"
                "def guarded_lookup_return(name):\n"
                "    return builtins.__dict__.get(name)\n",
            ),
        ):
            with self.subTest(windows_blas_cross_module_mutation=description):
                cross_module_path.write_text(
                    original_cross_module_source + injection,
                    encoding="utf-8",
                )
                reviewed = CHECKER._reviewed_python_source_module(
                    self.root,
                    "bench/tools/test_level1_report.py",
                    "bench/tools",
                    "test_*.py",
                )
                tree = ast.parse(
                    reviewed.source_bytes.decode("utf-8"),
                    filename=reviewed.inventory_path,
                )
                with self.assertRaisesRegex(CHECKER.InventoryError, "Windows BLAS"):
                    CHECKER._python_windows_blas_source_audit(tree, reviewed)
        cross_module_path.write_text(original_cross_module_source, encoding="utf-8")

        inventory = copy.deepcopy(self.inventory)
        inventory["test_mode_rows"][0]["predicate_id"] = "predicate:unknown"
        self._write(inventory)
        self.assertIn("immutable matrix fields", self._errors())

        inventory = copy.deepcopy(self.inventory)
        row = next(
            row
            for row in inventory["test_mode_rows"]
            if row["expectation_state"] == CHECKER.FROZEN_STATE
            and row["root_id"].startswith("zig-root:")
        )
        row["optimize_mode_id"] = "mode:ReleaseFast"
        self._write(inventory)
        self.assertIn("immutable matrix fields", self._errors())

        for field, value in (
            ("cpu", "native"),
            ("cpu", "x86_64_v3"),
            ("cpu", "x86_64_v4"),
            ("cpu", "baseline+avx2"),
            ("resolved_cpu_model", "native"),
            ("cpu_feature_policy", "exact-model-features"),
        ):
            with self.subTest(environment_cpu_field=field, value=value):
                inventory = copy.deepcopy(self.inventory)
                inventory["environment_profiles"][0][field] = value
                self._write(inventory)
                self.assertIn("environment_profiles", self._errors())

    def test_python_root_commands_are_directly_executable(self) -> None:
        original_bytes = self.inventory_path.read_bytes()
        environment_id, protocol = self._native_refresh_protocol(pending=False)
        stdout = io.StringIO()
        stderr = io.StringIO()
        with (
            mock.patch.dict(os.environ),
            mock.patch.object(CHECKER, "_publish_inventory_atomic") as publish,
            contextlib.redirect_stdout(stdout),
            contextlib.redirect_stderr(stderr),
        ):
            os.environ.pop("GIT_PAGER", None)
            os.environ.pop("PAGER", None)
            result = CHECKER.main(self._refresh_arguments(environment_id, protocol))
        self.assertEqual(original_bytes, self.inventory_path.read_bytes())
        original_digest = hashlib.sha256(original_bytes).hexdigest()
        if original_digest != CHECKER.CURRENT_TEST_INVENTORY_SHA256:
            self.assertEqual(CHECKER.NEXT_TEST_INVENTORY_SHA256, original_digest)
        self.assertEqual(0, result, stderr.getvalue())
        publish.assert_called_once()
        candidate_bytes = publish.call_args.args[1]
        self.assertEqual(original_bytes, candidate_bytes)
        candidate_digest = hashlib.sha256(candidate_bytes).hexdigest()

        source_current = self._source_current_fixture_inventory()
        self.assertEqual(
            candidate_digest,
            hashlib.sha256(
                CHECKER._canonical_inventory_bytes(source_current)
            ).hexdigest(),
        )
        roots = {
            row["id"]: row
            for row in source_current["test_roots"]
            if row["language"] == "python"
        }
        self.assertEqual(
            {
                "python-root:abi-artifact-parity-direct",
                "python-root:abi-baseline-discovery",
                "python-root:benchmark-tools-discovery",
                "python-root:build-inventory-direct",
                "python-root:test-inventory-direct",
            },
            set(roots),
        )
        self.assertFalse(
            any(
                gap["kind"].startswith("python-")
                for gap in source_current["known_gaps"]
            )
        )
        commands = {
            row["root_id"]: row["command_template"]
            for row in source_current["test_mode_rows"]
            if row["disposition"] == "execute" and row["root_id"] in roots
        }
        self.assertEqual(
            'python3 -B -m unittest discover -s test/abi/baseline -p "test_*.py"',
            commands["python-root:abi-baseline-discovery"],
        )
        self.assertEqual(
            "python3 -B test/build/test_build_inventory.py",
            commands["python-root:build-inventory-direct"],
        )
        self.assertEqual(
            'python3 -B -m unittest discover -s bench/tools -p "test_*.py"',
            commands["python-root:benchmark-tools-discovery"],
        )
        self.assertEqual(
            "python3 -B test/build/test_test_inventory.py",
            commands["python-root:test-inventory-direct"],
        )

        tooling_root = roots[CHECKER.PYTHON_TOOLING_ROOT_ID]
        self.assertEqual(
            [CHECKER.PYTHON_TOOLING_LAUNCH_ID],
            tooling_root["launch_observation_ids"],
        )
        self.assertEqual(
            CHECKER.AGGREGATE_STEP_ID,
            tooling_root["aggregate_step_observation_id"],
        )
        self.assertIs(tooling_root["matrix_applicable"], False)
        tooling_modules = [
            row
            for row in source_current["python_test_modules"]
            if CHECKER.PYTHON_TOOLING_ROOT_ID in row["root_ids"]
        ]
        self.assertEqual(14, len(tooling_modules))
        self.assertTrue(
            all(
                row["launch_observation_ids"] == [CHECKER.PYTHON_TOOLING_LAUNCH_ID]
                for row in tooling_modules
            )
        )
        tooling_set = next(
            row
            for row in source_current["expected_test_sets"]
            if row["root_id"] == CHECKER.PYTHON_TOOLING_ROOT_ID
        )
        prior_tooling_set = next(
            row
            for row in self.inventory["expected_test_sets"]
            if row["root_id"] == CHECKER.PYTHON_TOOLING_ROOT_ID
        )
        self.assertEqual(465, tooling_set["count"])
        self.assertEqual(prior_tooling_set, tooling_set)

        real_wrapped_binding_counts: list[int] = []
        real_suites: list[unittest.TestSuite] = []
        real_suite_contracts: list[Any] = []
        execute_python_tooling_suite = CHECKER._execute_python_tooling_suite

        def flattened_runtime_ids(item: Any) -> list[str]:
            if isinstance(item, unittest.TestCase):
                return [item.id()]
            identities: list[str] = []
            for child in item:
                identities.extend(flattened_runtime_ids(child))
            return identities

        legacy_module_names = {
            name for name, _ in CHECKER._PYTHON_TOOLING_EXECUTION_MODULES
        }
        legacy_tools = self.root / tooling_root["discovery_start"]
        try:
            with mock.patch.object(sys, "path", [str(legacy_tools), *sys.path]):
                legacy_suite = unittest.TestLoader().discover(
                    str(legacy_tools), pattern=tooling_root["discovery_pattern"]
                )
            legacy_runtime_ids = flattened_runtime_ids(legacy_suite)
        finally:
            for name in legacy_module_names:
                sys.modules.pop(name, None)
        self.assertEqual(465, len(legacy_runtime_ids))

        def execute_stub(
            suite: unittest.TestSuite,
            contract: Any,
            trusted: Any,
        ) -> Any:
            real_suites.append(suite)
            real_suite_contracts.append(contract)
            real_wrapped_binding_counts.append(
                sum(
                    binding.descriptor_wrapped_present is True
                    for binding in contract.discovered_test_bindings
                )
            )
            tampered_runtime_order = (
                "__tampered_runtime_order__",
                *contract.runtime_order[1:],
            )
            with self.assertRaisesRegex(
                CHECKER.InventoryError, "runtime order changed"
            ):
                execute_python_tooling_suite(
                    suite,
                    contract._replace(
                        runtime_integrity_callback=None,
                        runtime_order=tampered_runtime_order,
                    ),
                    trusted,
                )
            skips = (
                contract.required_decorator_skips
                | contract.permitted_dynamic_skips
                | contract.platform_skips
            )
            return CHECKER._PythonToolingOutcome(
                contract.discovered_count, True, skips, 0, 0, 0, 0
            )

        with (
            mock.patch.object(
                unittest.TestLoader,
                "discover",
                side_effect=AssertionError("live discovery is forbidden"),
            ) as rejected_live_discovery,
            mock.patch.object(
                CHECKER, "_execute_python_tooling_suite", side_effect=execute_stub
            ) as stubbed_execution,
        ):
            real_binding_summary = CHECKER._run_python_tooling_root(
                REPOSITORY_ROOT,
                REPOSITORY_ROOT / "tools/test_inventory.json",
                CHECKER.PYTHON_TOOLING_ROOT_ID,
            )
        stubbed_execution.assert_called_once()
        rejected_live_discovery.assert_not_called()
        self.assertEqual(465, real_binding_summary.discovered)
        self.assertEqual(465, real_binding_summary.outcome.executed)
        self.assertEqual(1, len(real_wrapped_binding_counts))
        self.assertEqual(1, len(real_suites))
        self.assertEqual(1, len(real_suite_contracts))
        projected_runtime_ids = real_suite_contracts[0].runtime_order
        self.assertEqual(465, len(projected_runtime_ids))
        self.assertEqual(tuple(legacy_runtime_ids), projected_runtime_ids)
        self.assertEqual(
            projected_runtime_ids, tuple(flattened_runtime_ids(real_suites[0]))
        )
        self.assertEqual(
            CHECKER._PYTHON_TOOLING_RUNTIME_ORDER_SHA256,
            CHECKER._python_tooling_runtime_order_digest(projected_runtime_ids),
        )
        self.assertGreater(real_wrapped_binding_counts[0], 0)
        self.assertEqual(0, real_binding_summary.artifact_platform_skips)
        self.assertEqual(0, real_binding_summary.publication_platform_skips)
        self.assertEqual(0, real_binding_summary.platform_skips)
        self.assertEqual(42, len(source_current["expected_test_sets"]))
        self.assertEqual(312, len(source_current["test_mode_rows"]))
        self.assertEqual(123, len(source_current["native_observation_bindings"]))
        self.assertEqual(123, CHECKER._matrix_incomplete_count(source_current))
        self.assertEqual(
            CHECKER.CURRENT_NATIVE_PROJECTION_SHA256,
            CHECKER._native_projection_digest(source_current),
        )
        source_current_bytes = self.inventory_path.read_bytes()
        publication_platform_predicate_id = (
            "python-skip-predicate:report-publication-platform-unavailable"
        )
        artifact_platform_predicate_id = (
            "python-skip-predicate:artifact-snapshot-platform-unavailable"
        )
        publication_platform_skip_entries = [
            entry
            for entry in source_current["python_skip_contracts"][0]["entries"]
            if entry["predicate_id"] == publication_platform_predicate_id
        ]
        artifact_platform_skip_entries = [
            entry
            for entry in source_current["python_skip_contracts"][0]["entries"]
            if entry["predicate_id"] == artifact_platform_predicate_id
        ]
        self.assertEqual(60, len(publication_platform_skip_entries))
        self.assertEqual(33, len(artifact_platform_skip_entries))
        self.assertEqual(
            {"POSIX report publication APIs are unavailable"},
            {entry["reason"] for entry in publication_platform_skip_entries},
        )
        self.assertEqual(
            {"POSIX artifact snapshot APIs are unavailable"},
            {entry["reason"] for entry in artifact_platform_skip_entries},
        )
        windows_skip_pairs = frozenset(
            (CHECKER._unittest_runtime_id(entry["test"]), entry["reason"])
            for entry in (
                *artifact_platform_skip_entries,
                *publication_platform_skip_entries,
            )
        )
        self.assertEqual(93, len(windows_skip_pairs))
        for mutation in ("reason", "predicate"):
            with self.subTest(report_publication_contract_mutation=mutation):
                candidate = copy.deepcopy(source_current)
                entry = next(
                    item
                    for item in candidate["python_skip_contracts"][0]["entries"]
                    if item["predicate_id"] == publication_platform_predicate_id
                )
                if mutation == "reason":
                    entry["reason"] = "forged platform reason"
                else:
                    entry["predicate_id"] = "python-skip-predicate:not-darwin"
                candidate["python_skip_contracts"][0]["digest"] = CHECKER._fact_digest(
                    candidate["python_skip_contracts"][0]["entries"]
                )
                self._write(candidate)
                self.assertTrue(self._errors())
        self.inventory_path.write_bytes(source_current_bytes)

        for mutation, expected_error in (
            ("unsorted", "strictly sorted"),
            ("duplicate", "duplicates"),
            ("unknown-predicate", "noncanonical"),
            ("wrong-kind", "noncanonical skip kind"),
            ("wrong-predicate-digest", "noncanonical skip kind"),
            ("bad-digest", "count/digest mismatch"),
        ):
            with self.subTest(python_skip_contract=mutation):
                candidate = copy.deepcopy(source_current)
                contract = candidate["python_skip_contracts"][0]
                if mutation == "unsorted":
                    contract["entries"] = list(reversed(contract["entries"]))
                    contract["digest"] = CHECKER._fact_digest(contract["entries"])
                elif mutation == "duplicate":
                    contract["entries"].append(copy.deepcopy(contract["entries"][0]))
                    contract["count"] += 1
                    contract["digest"] = CHECKER._fact_digest(contract["entries"])
                elif mutation == "unknown-predicate":
                    contract["entries"][0]["predicate_id"] = (
                        "python-skip-predicate:unknown"
                    )
                    contract["digest"] = CHECKER._fact_digest(contract["entries"])
                elif mutation == "wrong-kind":
                    contract["entries"][0]["skip_kind"] = "unittest.skipIf"
                    contract["digest"] = CHECKER._fact_digest(contract["entries"])
                elif mutation == "wrong-predicate-digest":
                    contract["entries"][0]["predicate_ast_sha256"] = "0" * 64
                    contract["digest"] = CHECKER._fact_digest(contract["entries"])
                else:
                    contract["digest"] = "0" * 64
                self._write(candidate)
                self.assertIn(expected_error, self._errors())
        self.inventory_path.write_bytes(source_current_bytes)

        decorator_source = self.root / "bench/tools/test_level1_report.py"
        decorator_text = decorator_source.read_text(encoding="utf-8")
        decorator_block = (
            "@unittest.skipUnless(\n"
            "    runner.library_available(runner.DEFAULT_ACCELERATE), "
            '"Accelerate is unavailable"\n'
            ")"
        )
        self.assertEqual(1, decorator_text.count(decorator_block))
        for mutation, replacement, expected_error in (
            (
                "unconditional",
                '@unittest.skip("Accelerate is unavailable")',
                "unconditional unittest.skip",
            ),
            (
                "wrong-kind",
                "@unittest.skipIf(\n"
                "    runner.library_available(runner.DEFAULT_ACCELERATE), "
                '"Accelerate is unavailable"\n'
                ")",
                "skip identity/reason/kind/predicate",
            ),
            (
                "constant-skip-unless",
                '@unittest.skipUnless(False, "Accelerate is unavailable")',
                "skip identity/reason/kind/predicate",
            ),
            (
                "constant-skip-if",
                '@unittest.skipIf(True, "Accelerate is unavailable")',
                "skip identity/reason/kind/predicate",
            ),
        ):
            with self.subTest(python_skip_source_predicate=mutation):
                decorator_source.write_text(
                    decorator_text.replace(decorator_block, replacement, 1),
                    encoding="utf-8",
                )
                try:
                    with self._review_current_python_tooling_sources():
                        self.assertIn(expected_error, self._errors())
                finally:
                    decorator_source.write_text(decorator_text, encoding="utf-8")

        alias_mutations = (
            (
                "additive-returned-alias",
                decorator_text.replace(
                    decorator_block,
                    'force_skip = unittest.skip("Accelerate is unavailable")\n'
                    "@force_skip\n" + decorator_block,
                    1,
                ),
            ),
            (
                "import-alias",
                decorator_text.replace(
                    "import unittest\n",
                    "import unittest\nfrom unittest import skip as force_skip\n",
                    1,
                ).replace(
                    decorator_block,
                    '@force_skip("Accelerate is unavailable")\n' + decorator_block,
                    1,
                ),
            ),
            (
                "assigned-attribute-alias",
                decorator_text.replace(
                    decorator_block,
                    "force_skip = unittest.skip\n"
                    '@force_skip("Accelerate is unavailable")\n' + decorator_block,
                    1,
                ),
            ),
        )
        for mutation, mutated_text in alias_mutations:
            with self.subTest(python_skip_source_alias=mutation):
                decorator_source.write_text(mutated_text, encoding="utf-8")
                try:
                    with self._review_current_python_tooling_sources():
                        expected_error = (
                            "Python Windows BLAS noncanonical import shape"
                            if mutation == "import-alias"
                            else "unittest skip aliases"
                        )
                        self.assertIn(expected_error, self._errors())
                finally:
                    decorator_source.write_text(decorator_text, encoding="utf-8")

        decorator_source.write_text(
            decorator_text.replace(
                decorator_block,
                "def unreviewed_decorator(test):\n"
                "    return test\n\n"
                "@unreviewed_decorator\n" + decorator_block,
                1,
            ),
            encoding="utf-8",
        )
        try:
            with self._review_current_python_tooling_sources():
                self.assertIn("unreviewed decorator", self._errors())
        finally:
            decorator_source.write_text(decorator_text, encoding="utf-8")

        publication_source = self.root / "bench/tools/test_report_publication.py"
        publication_text = publication_source.read_text(encoding="utf-8")
        self.assertNotIn(
            "POSIX report publication APIs are unavailable", publication_text
        )

        dynamic_source = self.root / "bench/tools/test_report_publication.py"
        dynamic_text = dynamic_source.read_text(encoding="utf-8")
        dynamic_call = 'self.skipTest("filesystem permits case-distinct names")'
        self.assertEqual(1, dynamic_text.count(dynamic_call))
        dynamic_mutations = (
            (
                "attribute-escape",
                "self.hidden_skip = self.skipTest",
            ),
            (
                "container-callback",
                "callbacks = [self.skipTest]\n"
                '            callbacks[0]("filesystem permits case-distinct names")',
            ),
            (
                "dunder-getattribute",
                'self.__getattribute__("skipTest")('
                '"filesystem permits case-distinct names")',
            ),
            (
                "computed-getattr",
                'getattr(self, "skip" + "Test")('
                '"filesystem permits case-distinct names")',
            ),
            (
                "getattr-callback-subscript",
                "attribute_getters = [getattr]\n"
                '            attribute_getters[0](self, "skip" + "Test")('
                '"filesystem permits case-distinct names")',
            ),
            (
                "self-alias",
                "skip_owner = self\n"
                "            skip_owner.skipTest("
                '"filesystem permits case-distinct names")',
            ),
            (
                "subscript-indirect",
                "skip_owners = [self]\n"
                "            skip_owners[0].skipTest("
                '"filesystem permits case-distinct names")',
            ),
            (
                "testcase-class-dict",
                'unittest.TestCase.__dict__["skip" + "Test"]('
                'self, "filesystem permits case-distinct names")',
            ),
            (
                "self-class-dict",
                'self.__class__.__dict__["skip" + "Test"]('
                'self, "filesystem permits case-distinct names")',
            ),
            (
                "attrgetter",
                'operator.attrgetter("skip" + "Test")(self)('
                '"filesystem permits case-distinct names")',
            ),
            (
                "vars-class-dict",
                'vars(self.__class__)["skip" + "Test"]('
                'self, "filesystem permits case-distinct names")',
            ),
            (
                "mro-class-dict",
                'self.__class__.__mro__[1].__dict__["skip" + "Test"]('
                'self, "filesystem permits case-distinct names")',
            ),
            (
                "nested-helper",
                "def hidden():\n"
                "                return self.skipTest("
                '"filesystem permits case-distinct names")',
            ),
            (
                "raise-skiptest",
                'raise unittest.SkipTest("filesystem permits case-distinct names")',
            ),
            (
                "lambda-closure",
                "hidden = lambda: self.skipTest("
                '"filesystem permits case-distinct names")',
            ),
            (
                "comprehension",
                "hidden = [self.skipTest("
                '"filesystem permits case-distinct names") for _ in [0]]',
            ),
            (
                "walrus",
                '(hidden := self.skipTest)("filesystem permits case-distinct names")',
            ),
            (
                "computed-object-getattribute-same-line",
                'object.__getattribute__(self, "".join(["skip", "Test"]))('
                '"filesystem permits case-distinct names")',
            ),
            (
                "chained-object-getattribute",
                "object.__getattribute__.__call__("
                'self, "".join(["skip", "Test"]))('
                '"filesystem permits case-distinct names")',
            ),
            (
                "builtins-getattr",
                'builtins.getattr(self, "".join(["skip", "Test"]))('
                '"filesystem permits case-distinct names")',
            ),
            (
                "builtins-getattr-alias",
                "hidden_getattr = builtins.getattr",
            ),
            (
                "type-second-argument",
                'type("Forged", (unittest.TestCase,), {})',
            ),
            (
                "type-third-argument",
                'type("Forged", (), {"hidden": self})',
            ),
            (
                "setattr-third-argument",
                'setattr(object(), "hidden", self)',
            ),
            (
                "interprocedural-function",
                "def reflected(receiver):\n"
                "                return getattr("
                'receiver, "".join(["skip", "Test"]))\n'
                "            reflected(self)("
                '"filesystem permits case-distinct names")',
            ),
            (
                "interprocedural-lambda",
                "reflected = lambda receiver: getattr("
                'receiver, "".join(["skip", "Test"]))\n'
                "            reflected(self)("
                '"filesystem permits case-distinct names")',
            ),
            (
                "interprocedural-returned-object",
                "def reflected(receiver):\n"
                "                return receiver\n"
                "            getattr("
                'reflected(self), "".join(["skip", "Test"]))('
                '"filesystem permits case-distinct names")',
            ),
            (
                "interprocedural-returned-callable",
                "reflected = lambda receiver: getattr("
                'receiver, "".join(["skip", "Test"]))\n'
                "            factory = lambda: reflected\n"
                "            factory()(self)("
                '"filesystem permits case-distinct names")',
            ),
            (
                "interprocedural-helper-alias",
                "def reflected(receiver):\n"
                "                return getattr("
                'receiver, "".join(["skip", "Test"]))\n'
                "            reflected_alias = reflected\n"
                "            reflected_alias(self)("
                '"filesystem permits case-distinct names")',
            ),
            (
                "interprocedural-starred-varargs",
                "def reflected(*receivers):\n"
                "                return getattr("
                'receivers[0], "".join(["skip", "Test"]))\n'
                "            reflected(*(self,))("
                '"filesystem permits case-distinct names")',
            ),
            (
                "interprocedural-expanded-kwargs",
                "def reflected(**receivers):\n"
                "                return getattr("
                'receivers["target"], "".join(["skip", "Test"]))\n'
                '            reflected(**{"target": self})('
                '"filesystem permits case-distinct names")',
            ),
            (
                "interprocedural-nonroot-bound-receiver",
                "class Reflector:\n"
                "                def reflected(helper_self, receiver):\n"
                "                    return getattr("
                'receiver, "".join(["skip", "Test"]))\n'
                "            Reflector().reflected(self)("
                '"filesystem permits case-distinct names")',
            ),
            (
                "interprocedural-same-method-name",
                "class DangerousReflector:\n"
                "                def reflected(helper_self, receiver):\n"
                "                    return getattr("
                'receiver, "".join(["skip", "Test"]))\n'
                "            class SafeReflector:\n"
                "                def reflected(helper_self, receiver):\n"
                "                    return receiver\n"
                "            DangerousReflector().reflected(self)("
                '"filesystem permits case-distinct names")',
            ),
            (
                "interprocedural-lexical-shadow",
                "def reflected(receiver):\n"
                "                return receiver\n"
                "            def wrapper():\n"
                "                def reflected(receiver):\n"
                "                    return getattr("
                'receiver, "".join(["skip", "Test"]))\n'
                "                def decoy():\n"
                "                    def reflected(receiver):\n"
                "                        return receiver\n"
                "                    return reflected\n"
                "                return reflected(self)\n"
                "            wrapper()("
                '"filesystem permits case-distinct names")',
            ),
            (
                "interprocedural-factory-result-alias",
                "def reflected(receiver):\n"
                "                return getattr("
                'receiver, "".join(["skip", "Test"]))\n'
                "            def factory():\n"
                "                return reflected\n"
                "            reflected_alias = factory()\n"
                "            reflected_alias(self)("
                '"filesystem permits case-distinct names")',
            ),
            (
                "interprocedural-class-qualified-unbound",
                "class Reflector:\n"
                "                def reflected(receiver):\n"
                "                    return getattr("
                'receiver, "".join(["skip", "Test"]))\n'
                "            Reflector.reflected(self)("
                '"filesystem permits case-distinct names")',
            ),
            (
                "interprocedural-function-local-testcase",
                "TestBase = unittest.TestCase\n"
                "            class LocalTestCase(TestBase):\n"
                "                pass\n"
                "            getattr("
                'LocalTestCase, "".join(["skip", "Test"]))('
                '"filesystem permits case-distinct names")',
            ),
            (
                "interprocedural-staticmethod-alias",
                "static_alias = staticmethod\n"
                "            class Reflector:\n"
                "                @static_alias\n"
                "                def reflected(receiver):\n"
                "                    return getattr("
                'receiver, "".join(["skip", "Test"]))\n'
                "            Reflector().reflected(self)("
                '"filesystem permits case-distinct names")',
            ),
            (
                "interprocedural-shadowed-classmethod",
                "classmethod = staticmethod\n"
                "            class Reflector:\n"
                "                @classmethod\n"
                "                def reflected(receiver):\n"
                "                    return getattr("
                'receiver, "".join(["skip", "Test"]))\n'
                "            Reflector.reflected(self)("
                '"filesystem permits case-distinct names")',
            ),
            (
                "interprocedural-conflicting-decorator-rebinding",
                "decorator_alias = classmethod\n"
                "            decorator_alias = staticmethod\n"
                "            class Reflector:\n"
                "                @decorator_alias\n"
                "                def reflected(receiver):\n"
                "                    return getattr("
                'receiver, "".join(["skip", "Test"]))\n'
                "            Reflector.reflected(self)("
                '"filesystem permits case-distinct names")',
            ),
            (
                "interprocedural-custom-decorator-rebinding",
                "decorator_alias = staticmethod\n"
                "            decorator_alias = lambda function: function\n"
                "            class Reflector:\n"
                "                @decorator_alias\n"
                "                def reflected(receiver):\n"
                "                    return getattr("
                'receiver, "".join(["skip", "Test"]))\n'
                "            Reflector().reflected(self)("
                '"filesystem permits case-distinct names")',
            ),
            (
                "interprocedural-container-callable",
                "def reflected(receiver):\n"
                "                return getattr("
                'receiver, "".join(["skip", "Test"]))\n'
                "            def factory():\n"
                "                return (reflected,)\n"
                "            factory()[0](self)("
                '"filesystem permits case-distinct names")',
            ),
            (
                "interprocedural-unresolved-attribute-root",
                "external_helper.reflect(self)("
                '"filesystem permits case-distinct names")',
            ),
            (
                "rooted-subscript-assignment",
                'GLOBAL["receiver"] = self',
            ),
            (
                "rooted-nonlocal-assignment",
                "holder = None\n"
                "            def store():\n"
                "                nonlocal holder\n"
                "                holder = self\n"
                "            store()",
            ),
            (
                "rooted-global-augmented-assignment",
                "global GLOBAL\n            GLOBAL += [self]",
            ),
            (
                "rooted-attribute-augmented-assignment",
                "self.holder += [self]",
            ),
            (
                "rooted-subscript-for-assignment",
                "for GLOBAL[0] in (self,):\n                pass",
            ),
            (
                "result-addskip",
                "self._outcome.result.addSkip("
                'self, "filesystem permits case-distinct names")',
            ),
        )
        windows_blas_dynamic_rejections = {
            **dict.fromkeys(
                (
                    "dunder-getattribute",
                    "chained-object-getattribute",
                ),
                "Python Windows BLAS meta-execution capability was invoked or escaped",
            ),
            **dict.fromkeys(
                (
                    "computed-getattr",
                    "computed-object-getattribute-same-line",
                    "builtins-getattr",
                    "interprocedural-function",
                    "interprocedural-lambda",
                    "interprocedural-returned-object",
                    "interprocedural-returned-callable",
                    "interprocedural-helper-alias",
                    "interprocedural-starred-varargs",
                    "interprocedural-expanded-kwargs",
                    "interprocedural-nonroot-bound-receiver",
                    "interprocedural-same-method-name",
                    "interprocedural-lexical-shadow",
                    "interprocedural-factory-result-alias",
                    "interprocedural-class-qualified-unbound",
                    "interprocedural-function-local-testcase",
                    "interprocedural-staticmethod-alias",
                    "interprocedural-shadowed-classmethod",
                    "interprocedural-conflicting-decorator-rebinding",
                    "interprocedural-custom-decorator-rebinding",
                    "interprocedural-container-callable",
                ),
                "Python Windows BLAS dynamic attribute recovery is forbidden",
            ),
            **dict.fromkeys(
                (
                    "getattr-callback-subscript",
                    "builtins-getattr-alias",
                ),
                "Python Windows BLAS meta-execution capability escaped static review",
            ),
            **dict.fromkeys(
                (
                    "self-class-dict",
                    "vars-class-dict",
                    "mro-class-dict",
                ),
                "Python Windows BLAS dynamic runtime namespace lookup is forbidden",
            ),
            "setattr-third-argument": (
                "Python Windows BLAS mutation capability escaped static review"
            ),
            **dict.fromkeys(
                (
                    "rooted-nonlocal-assignment",
                    "rooted-global-augmented-assignment",
                ),
                "Python Windows BLAS identity escaped through a global or nonlocal "
                "binding",
            ),
        }
        for mutation, additive_source in dynamic_mutations:
            with self.subTest(python_skip_capability_escape=mutation):
                dynamic_source.write_text(
                    dynamic_text.replace(
                        dynamic_call,
                        f"{additive_source}\n            {dynamic_call}",
                        1,
                    ),
                    encoding="utf-8",
                )
                try:
                    expected_error = windows_blas_dynamic_rejections.get(
                        mutation,
                        "noncanonical skipTest capability access",
                    )
                    with self._review_current_python_tooling_sources():
                        self.assertIn(
                            expected_error,
                            self._errors(),
                        )
                finally:
                    dynamic_source.write_text(dynamic_text, encoding="utf-8")

        class_header = "class ReportPublicationTests(unittest.TestCase):"
        whole_module_mutations = (
            (
                "module-alias",
                "hidden_skip = unittest.TestCase.skipTest\n\n" + class_header,
            ),
            (
                "class-alias",
                class_header + "\n    hidden_skip = unittest.TestCase.skipTest",
            ),
            (
                "testcase-hook-definition",
                class_header + "\n    def _callTestMethod(self, method):\n"
                "        return None",
            ),
            (
                "testcase-hook-assignment",
                class_header + "\n    run = lambda self, result=None: result",
            ),
        )
        self.assertEqual(1, dynamic_text.count(class_header))
        for mutation, replacement in whole_module_mutations:
            with self.subTest(python_skip_whole_module_escape=mutation):
                dynamic_source.write_text(
                    dynamic_text.replace(class_header, replacement, 1),
                    encoding="utf-8",
                )
                try:
                    expected_error = (
                        "noncanonical execution hook"
                        if mutation.startswith("testcase-hook-")
                        else "noncanonical skipTest capability access"
                    )
                    with self._review_current_python_tooling_sources():
                        self.assertIn(
                            expected_error,
                            self._errors(),
                        )
                finally:
                    dynamic_source.write_text(dynamic_text, encoding="utf-8")

        for accepted_scope, source in (
            (
                "ordinary-helper-self",
                "class Helper:\n"
                "    def reflected(self):\n"
                '        return getattr(self, "ordinary")\n'
                "Helper().reflected()\n",
            ),
            (
                "local-testcase-parameter-shadow",
                "from unittest import TestCase\n"
                "def reflected(TestCase):\n"
                '    return getattr(TestCase, "ordinary")\n'
                "reflected(object())\n",
            ),
            (
                "separate-scopes-same-class-name",
                "from unittest import TestCase\n"
                "def root_scope():\n"
                "    class SameName(TestCase):\n"
                "        pass\n"
                "    return SameName\n"
                "def ordinary_scope():\n"
                "    class SameName:\n"
                "        pass\n"
                '    return getattr(SameName, "ordinary")\n'
                "ordinary_scope()\n",
            ),
            (
                "local-unittest-module-parameter-shadow",
                "import unittest\n"
                "def reflected(unittest):\n"
                '    return getattr(unittest.TestCase, "ordinary")\n'
                "reflected(object())\n",
            ),
            (
                "unknown-decorator-without-root",
                "def preserve(function):\n"
                "    return function\n"
                "class Helper:\n"
                "    @preserve\n"
                "    def reflected(receiver):\n"
                '        return getattr(receiver, "ordinary")\n'
                "Helper().reflected(object())\n",
            ),
            (
                "local-augmented-assignment-without-root",
                "def increment():\n"
                "    count = 0\n"
                "    count += 1\n"
                "    return count\n"
                "increment()\n",
            ),
            (
                "single-proven-staticmethod-alias",
                "decorator_alias = staticmethod\n"
                "class Helper:\n"
                "    @decorator_alias\n"
                "    def reflected(receiver):\n"
                '        return getattr(receiver, "ordinary")\n'
                "Helper().reflected(object())\n",
            ),
        ):
            with self.subTest(python_skip_scope_positive=accepted_scope):
                CHECKER._python_skip_capability_audit(CHECKER.ast.parse(source), set())

        runtime_id = "runtime_authorization_case.RuntimeAuthorizationCase.runTest"
        runtime_reason = "reviewed runtime reason"

        def reviewed_runtime_site() -> None:
            pass

        runtime_test = mock.Mock(spec=unittest.TestCase)
        runtime_test.id.return_value = runtime_id
        stolen_test = mock.Mock(spec=unittest.TestCase)
        stolen_test.id.return_value = "mutable-id"
        authorization = CHECKER._PythonDynamicSkipAuthorization(
            runtime_test,
            runtime_id,
            runtime_reason,
            reviewed_runtime_site.__code__,
            reviewed_runtime_site.__code__.co_firstlineno,
        )
        runtime_contract = CHECKER._PythonToolingSuiteContract(
            2,
            frozenset(),
            frozenset({(runtime_id, runtime_reason)}),
            (authorization,),
            frozenset(),
            (),
            (
                CHECKER._PythonTestBinding(
                    runtime_test,
                    runtime_id,
                    unittest.TestCase,
                    "runTest",
                    reviewed_runtime_site,
                    reviewed_runtime_site,
                    reviewed_runtime_site.__code__,
                    (),
                ),
                CHECKER._PythonTestBinding(
                    stolen_test,
                    "canonical-stolen-test",
                    unittest.TestCase,
                    "runTest",
                    reviewed_runtime_site,
                    reviewed_runtime_site,
                    reviewed_runtime_site.__code__,
                    (),
                ),
            ),
        )
        trusted_runtime = CHECKER._capture_python_unittest_runtime_primitives()
        authorizer = CHECKER._PythonSkipRuntimeAuthorizer(
            runtime_contract, trusted_runtime
        )
        authorized_frame = mock.Mock(
            f_code=reviewed_runtime_site.__code__,
            f_lineno=reviewed_runtime_site.__code__.co_firstlineno,
        )
        with self.assertRaises(unittest.SkipTest) as authorized_skip:
            authorizer.skip_test(runtime_test, runtime_reason, authorized_frame)
        ticket_reason = str(authorized_skip.exception)
        runtime_result = authorizer.result_class()(io.StringIO(), True, 1)
        runtime_result.startTestRun()
        runtime_result.startTest(runtime_test)
        runtime_test.id.return_value = "forged-mutable-id"
        with self.assertRaisesRegex(
            CHECKER.InventoryError,
            "unauthorized skip ticket",
        ):
            runtime_result.addSkip(stolen_test, ticket_reason)
        runtime_result.addSkip(runtime_test, ticket_reason)
        runtime_result.stopTest(runtime_test)
        runtime_result.stopTestRun()
        self.assertEqual([(runtime_test, runtime_reason)], runtime_result.skipped)
        self.assertEqual(
            CHECKER._PythonToolingOutcome(
                executed=1,
                successful=True,
                skips=frozenset({(runtime_id, runtime_reason)}),
                failures=0,
                errors=0,
                expected_failures=0,
                unexpected_successes=0,
            ),
            authorizer.outcome_ledger.outcome(runtime_result),
        )
        for callback_name in (
            "addError",
            "addFailure",
            "addSubTest",
            "addExpectedFailure",
        ):
            with self.subTest(python_inventory_error_callback=callback_name):
                callback_error = CHECKER.InventoryError(
                    f"reviewed {callback_name} InventoryError"
                )
                callback_authorizer = CHECKER._PythonSkipRuntimeAuthorizer(
                    runtime_contract, trusted_runtime
                )
                callback_result = callback_authorizer.result_class()(
                    io.StringIO(), True, 1
                )
                callback_result.startTestRun()
                callback_result.startTest(runtime_test)
                callback_args = (
                    (
                        runtime_test,
                        object(),
                        (type(callback_error), callback_error, None),
                    )
                    if callback_name == "addSubTest"
                    else (
                        runtime_test,
                        (type(callback_error), callback_error, None),
                    )
                )
                with self.assertRaises(CHECKER.InventoryError) as raised:
                    getattr(callback_result, callback_name)(*callback_args)
                self.assertIs(callback_error, raised.exception)
                callback_result.stopTest(runtime_test)
                callback_result.stopTestRun()
        fixture_callback_error = CHECKER.InventoryError(
            "reviewed fixture addError InventoryError"
        )
        fixture_callback_authorizer = CHECKER._PythonSkipRuntimeAuthorizer(
            runtime_contract, trusted_runtime
        )
        fixture_callback_result = fixture_callback_authorizer.result_class()(
            io.StringIO(), True, 1
        )
        fixture_callback_result.startTestRun()
        with self.assertRaises(CHECKER.InventoryError) as raised_fixture_callback:
            fixture_callback_result.addError(
                object(),
                (
                    type(fixture_callback_error),
                    fixture_callback_error,
                    None,
                ),
            )
        self.assertIs(fixture_callback_error, raised_fixture_callback.exception)
        fixture_callback_result.stopTestRun()
        authorizer.require_all_tickets_consumed()
        for mutation, reason, frame in (
            ("direct-skiptest", runtime_reason, authorized_frame),
            (
                "helper-code",
                runtime_reason,
                mock.Mock(
                    f_code=(lambda: None).__code__,
                    f_lineno=reviewed_runtime_site.__code__.co_firstlineno,
                ),
            ),
        ):
            with self.subTest(python_skip_runtime_escape=mutation):
                if mutation == "direct-skiptest":
                    with self.assertRaisesRegex(
                        CHECKER.InventoryError,
                        "unauthorized skipped outcome",
                    ):
                        runtime_result.addSkip(runtime_test, reason)
                else:
                    with self.assertRaisesRegex(
                        CHECKER.InventoryError,
                        "unauthorized dynamic skip",
                    ):
                        authorizer.skip_test(runtime_test, reason, frame)
        with self.assertRaisesRegex(
            CHECKER.InventoryError,
            "unauthorized skip ticket",
        ):
            runtime_result.addSkip(runtime_test, ticket_reason)

        suppressed_authorizer = CHECKER._PythonSkipRuntimeAuthorizer(
            runtime_contract, trusted_runtime
        )
        with self.assertRaises(unittest.SkipTest) as suppressed_skip:
            suppressed_authorizer.skip_test(
                runtime_test, runtime_reason, authorized_frame
            )
        suppressed_result = suppressed_authorizer.result_class()(io.StringIO(), True, 1)
        suppressed_result.addSkip = lambda test, reason: None
        suppressed_result.addSkip(runtime_test, str(suppressed_skip.exception))
        del suppressed_result.addSkip
        with self.assertRaisesRegex(
            CHECKER.InventoryError,
            "unconsumed dynamic skip tickets",
        ):
            suppressed_authorizer.require_all_tickets_consumed()

        frozen_frame_authorizer = CHECKER._PythonSkipRuntimeAuthorizer(
            runtime_contract, trusted_runtime
        )
        forged_frame = mock.Mock(
            f_code=reviewed_runtime_site.__code__,
            f_lineno=reviewed_runtime_site.__code__.co_firstlineno,
        )

        def frozen_frame_wrapper() -> None:
            frozen_frame_authorizer.skip_test(
                runtime_test, runtime_reason, trusted_runtime.getframe(1)
            )

        with (
            mock.patch.object(CHECKER.sys, "_getframe", return_value=forged_frame),
            self.assertRaisesRegex(
                CHECKER.InventoryError,
                "unauthorized dynamic skip",
            ),
        ):
            frozen_frame_wrapper()
        CHECKER._verify_python_unittest_runtime_primitives(
            trusted_runtime, trusted_runtime.test_case_skip_test
        )
        for primitive, replacement in (
            ("TextTestRunner", object()),
            ("TestLoader", object()),
        ):
            with (
                self.subTest(python_unittest_runtime_primitive=primitive),
                mock.patch.object(CHECKER.unittest, primitive, replacement),
                self.assertRaisesRegex(
                    CHECKER.InventoryError,
                    "trusted unittest runtime primitive",
                ),
            ):
                CHECKER._verify_python_unittest_runtime_primitives(
                    trusted_runtime, trusted_runtime.test_case_skip_test
                )

        def forged_test_call(test: Any, result: Any = None) -> None:
            result.addSuccess(test)

        with (
            mock.patch.object(
                CHECKER.unittest.TestCase,
                "__call__",
                forged_test_call,
            ),
            self.assertRaisesRegex(
                CHECKER.InventoryError,
                "trusted unittest runtime primitive",
            ),
        ):
            CHECKER._verify_python_unittest_runtime_primitives(
                trusted_runtime, trusted_runtime.test_case_skip_test
            )

        class ForgedRunCase(unittest.TestCase):
            def run(self, result: Any = None) -> Any:
                result.addSuccess(self)
                return result

        class ForgedCallCase(unittest.TestCase):
            def __call__(self, result: Any = None) -> Any:
                result.addSuccess(self)
                return result

        class ForgedGetattributeCase(unittest.TestCase):
            def __getattribute__(self, name: str) -> Any:
                if name == "run":
                    return lambda result=None: result.addSuccess(self)
                return super().__getattribute__(name)

        class NoOpCallTestMethodCase(unittest.TestCase):
            def _callTestMethod(self, method: Any) -> None:
                return None

        class NoOpCallSetUpCase(unittest.TestCase):
            def _callSetUp(self) -> None:
                return None

        class NoOpCallTearDownCase(unittest.TestCase):
            def _callTearDown(self) -> None:
                return None

        class NoOpCallCleanupCase(unittest.TestCase):
            def _callCleanup(self, function: Any, *args: Any, **kwargs: Any) -> None:
                return None

        class ShadowedMethodCase(unittest.TestCase):
            def declared_failure(self) -> None:
                self.fail("declared failure must execute")

        async_body_ran = [False]

        class AsyncMethodCase(unittest.TestCase):
            async def declared_async(self) -> None:
                async_body_ran[0] = True

        generator_body_ran = [False]

        class GeneratorMethodCase(unittest.TestCase):
            def declared_generator(self) -> Any:
                generator_body_ran[0] = True
                yield None

        class ForgedSuite(unittest.TestSuite):
            pass

        for dispatch, forged_suite in (
            ("test-run", unittest.TestSuite([ForgedRunCase()])),
            ("test-call", unittest.TestSuite([ForgedCallCase()])),
            (
                "test-getattribute-run",
                unittest.TestSuite([ForgedGetattributeCase()]),
            ),
            (
                "test-call-method",
                unittest.TestSuite([NoOpCallTestMethodCase()]),
            ),
            ("test-setup", unittest.TestSuite([NoOpCallSetUpCase()])),
            ("test-teardown", unittest.TestSuite([NoOpCallTearDownCase()])),
            ("test-cleanup", unittest.TestSuite([NoOpCallCleanupCase()])),
            ("custom-suite", ForgedSuite()),
        ):
            with (
                self.subTest(python_unittest_dispatch=dispatch),
                self.assertRaisesRegex(
                    CHECKER.InventoryError,
                    "noncanonical (dispatch|suite)",
                ),
            ):
                CHECKER._flatten_unittest_suite(forged_suite, trusted_runtime)

        shadowed_method = ShadowedMethodCase("declared_failure")
        shadowed_method.__dict__["declared_failure"] = lambda: None
        with self.assertRaisesRegex(
            CHECKER.InventoryError,
            "noncanonical dispatch",
        ):
            CHECKER._flatten_unittest_suite(
                unittest.TestSuite([shadowed_method]), trusted_runtime
            )

        for method_shape, test, body_ran in (
            (
                "async",
                AsyncMethodCase("declared_async"),
                async_body_ran,
            ),
            (
                "generator",
                GeneratorMethodCase("declared_generator"),
                generator_body_ran,
            ),
        ):
            with (
                self.subTest(python_unittest_method_shape=method_shape),
                self.assertRaisesRegex(
                    CHECKER.InventoryError,
                    "noncanonical test method",
                ),
            ):
                CHECKER._flatten_unittest_suite(
                    unittest.TestSuite([test]), trusted_runtime
                )
            self.assertFalse(body_ran[0])

        hidden_body_ran = [False]

        class ForgedIdentityCase(unittest.TestCase):
            def __init__(self) -> None:
                super().__init__("declared_failure")
                self._testMethodName = "hidden_pass"

            def id(self) -> str:
                return "forged.expected.identity.declared_failure"

            def declared_failure(self) -> None:
                self.fail("declared failure must execute")

            def hidden_pass(self) -> None:
                hidden_body_ran[0] = True

        with self.assertRaisesRegex(
            CHECKER.InventoryError,
            "noncanonical dispatch",
        ):
            CHECKER._flatten_unittest_suite(
                unittest.TestSuite([ForgedIdentityCase()]), trusted_runtime
            )
        self.assertFalse(hidden_body_ran[0])

        def frozen_runtime_binding(test: unittest.TestCase) -> Any:
            method_name = object.__getattribute__(test, "_testMethodName")
            descriptor, bound_method, code = CHECKER._verify_python_test_case_dispatch(
                test, trusted_runtime, method_name
            )
            test_class = type(test)
            return CHECKER._PythonTestBinding(
                test,
                CHECKER._canonical_python_test_id(test_class, method_name),
                test_class,
                method_name,
                descriptor,
                bound_method,
                code,
                CHECKER._freeze_python_fixture_bindings(test, test_class, None),
            )

        def execute_frozen_suite(
            frozen_suite: unittest.TestSuite,
            tests: tuple[unittest.TestCase, ...],
            decorator_skips: frozenset[tuple[str, str]] = frozenset(),
        ) -> Any:
            bindings = tuple(frozen_runtime_binding(test) for test in tests)
            contract = CHECKER._PythonToolingSuiteContract(
                len(bindings),
                decorator_skips,
                frozenset(),
                (),
                frozenset(),
                (),
                bindings,
            )
            with contextlib.redirect_stderr(io.StringIO()):
                return CHECKER._execute_python_tooling_suite(
                    frozen_suite, contract, trusted_runtime
                )

        registry_source = self.root / "bench/tools/test_registry_fixture.py"
        registry_source_bytes = (
            "import functools\n"
            "import unittest\n\n"
            "def preserve(function):\n"
            "    @functools.wraps(function)\n"
            "    def wrapped(self):\n"
            "        return function(self)\n"
            "    return wrapped\n\n"
            "class WrappedRegistryCase(unittest.TestCase):\n"
            "    @preserve\n"
            "    def test_wrapped(self):\n"
            "        pass\n"
        ).encode("utf-8")
        registry_source.write_bytes(registry_source_bytes)
        reviewed_registry_module = CHECKER._PythonReviewedSourceModule(
            "bench/tools/test_registry_fixture.py",
            "test_registry_fixture",
            registry_source,
            registry_source_bytes,
            hashlib.sha256(registry_source_bytes).hexdigest(),
        )
        try:
            with CHECKER._registered_python_tooling_modules(
                (reviewed_registry_module,)
            ) as registry:
                source_module = registry[0]
                wrapped_class = source_module.namespace["WrappedRegistryCase"]
                wrapped_test = wrapped_class("test_wrapped")
                descriptor, bound, code = CHECKER._verify_python_test_case_dispatch(
                    wrapped_test, trusted_runtime, "test_wrapped"
                )
                wrapped_binding = CHECKER._PythonTestBinding(
                    test=wrapped_test,
                    runtime_id="test_registry_fixture.WrappedRegistryCase.test_wrapped",
                    test_class=wrapped_class,
                    method_name="test_wrapped",
                    method_descriptor=descriptor,
                    bound_method=bound,
                    code=code,
                    fixtures=CHECKER._freeze_python_fixture_bindings(
                        wrapped_test, wrapped_class, source_module
                    ),
                    source_module=source_module,
                    descriptor_name=descriptor.__name__,
                    descriptor_qualname=descriptor.__qualname__,
                    descriptor_module=descriptor.__module__,
                    descriptor_wrapped_present=True,
                    descriptor_wrapped=descriptor.__wrapped__,
                )
                CHECKER._verify_python_test_bindings(
                    (wrapped_binding,), trusted_runtime
                )

                class SourceBindingSubclass(CHECKER._PythonSourceModuleBinding):
                    __slots__ = ()

                class ReviewedSourceSubclass(CHECKER._PythonReviewedSourceModule):
                    __slots__ = ()

                class SourceBytesSubclass(bytes):
                    pass

                class SourceNamespaceSubclass(dict[str, Any]):
                    pass

                class SourceNameSubclass(str):
                    pass

                oversized_source = b"x" * (CHECKER.MAX_INVENTORY_BYTES + 1)
                forged_source_bindings = (
                    (
                        "binding-type",
                        SourceBindingSubclass(*source_module),
                    ),
                    (
                        "reviewed-type",
                        source_module._replace(
                            reviewed=ReviewedSourceSubclass(*source_module.reviewed)
                        ),
                    ),
                    (
                        "name-type",
                        source_module._replace(
                            name=SourceNameSubclass(source_module.name)
                        ),
                    ),
                    (
                        "source-bytes-type",
                        source_module._replace(
                            reviewed=source_module.reviewed._replace(
                                source_bytes=SourceBytesSubclass(
                                    source_module.reviewed.source_bytes
                                )
                            )
                        ),
                    ),
                    (
                        "source-bytes-size",
                        source_module._replace(
                            reviewed=source_module.reviewed._replace(
                                source_bytes=oversized_source,
                                source_sha256=hashlib.sha256(
                                    oversized_source
                                ).hexdigest(),
                            )
                        ),
                    ),
                    (
                        "namespace-type",
                        source_module._replace(
                            namespace=SourceNamespaceSubclass(source_module.namespace)
                        ),
                    ),
                )
                for guard_name, forged_binding in forged_source_bindings:
                    with (
                        self.subTest(python_registry_runtime_guard=guard_name),
                        self.assertRaisesRegex(
                            CHECKER.InventoryError,
                            "source module binding changed",
                        ),
                    ):
                        CHECKER._verify_python_source_module_binding(forged_binding)

                equal_reviewed_source = CHECKER._PythonReviewedSourceModule(
                    *source_module.reviewed
                )
                self.assertEqual(source_module.reviewed, equal_reviewed_source)
                self.assertIsNot(source_module.reviewed, equal_reviewed_source)
                equal_reviewed_binding = source_module._replace(
                    reviewed=equal_reviewed_source
                )
                CHECKER._verify_python_source_module_binding(equal_reviewed_binding)
                with mock.patch.object(
                    CHECKER.hashlib, "sha256", wraps=CHECKER.hashlib.sha256
                ) as repeated_source_hash:
                    CHECKER._verify_python_source_module_binding(source_module)
                    CHECKER._verify_python_source_module_registry(
                        (source_module,), (source_module.reviewed,)
                    )
                repeated_source_hash.assert_not_called()
                with self.assertRaisesRegex(
                    CHECKER.InventoryError,
                    "reviewed input identity changed",
                ):
                    CHECKER._verify_python_source_module_registry(
                        (equal_reviewed_binding,),
                        (source_module.reviewed,),
                    )
                with self.assertRaisesRegex(
                    CHECKER.InventoryError,
                    "source module registry changed",
                ):
                    CHECKER._verify_python_source_module_registry(
                        [source_module]  # type: ignore[arg-type]
                    )
                for metadata_name, replacement in (
                    ("__name__", "forged_registry_name"),
                    ("__file__", str(registry_source.with_name("forged.py"))),
                    ("__spec__", object()),
                    ("__loader__", object()),
                ):
                    with (
                        self.subTest(python_registry_metadata=metadata_name),
                        mock.patch.object(
                            source_module.module, metadata_name, replacement
                        ),
                        self.assertRaisesRegex(
                            CHECKER.InventoryError, "source module binding changed"
                        ),
                    ):
                        CHECKER._verify_python_source_module_binding(source_module)
                with (
                    mock.patch.dict(
                        "sys.modules",
                        {source_module.name: types.SimpleNamespace()},
                        clear=False,
                    ),
                    self.assertRaisesRegex(
                        CHECKER.InventoryError, "source module binding changed"
                    ),
                ):
                    CHECKER._verify_python_source_module_binding(source_module)
                with (
                    mock.patch.dict(
                        source_module.namespace,
                        {"WrappedRegistryCase": object()},
                        clear=False,
                    ),
                    self.assertRaisesRegex(
                        CHECKER.InventoryError, "class registry binding changed"
                    ),
                ):
                    CHECKER._verify_python_test_bindings(
                        (wrapped_binding,), trusted_runtime
                    )
                with (
                    mock.patch.object(descriptor, "__wrapped__", lambda: None),
                    self.assertRaisesRegex(
                        CHECKER.InventoryError, "test method binding changed"
                    ),
                ):
                    CHECKER._verify_python_test_bindings(
                        (wrapped_binding,), trusted_runtime
                    )

            invalid_reviewed_modules = (
                ReviewedSourceSubclass(*reviewed_registry_module),
                reviewed_registry_module._replace(
                    source_bytes=SourceBytesSubclass(
                        reviewed_registry_module.source_bytes
                    )
                ),
                reviewed_registry_module._replace(
                    source_bytes=oversized_source,
                    source_sha256=hashlib.sha256(oversized_source).hexdigest(),
                ),
                reviewed_registry_module._replace(source_sha256="0" * 64),
            )
            for invalid_reviewed in invalid_reviewed_modules:
                with (
                    self.subTest(
                        python_registry_reviewed_guard=type(invalid_reviewed).__name__
                    ),
                    mock.patch("builtins.compile") as rejected_compile,
                    self.assertRaisesRegex(
                        CHECKER.InventoryError,
                        "reviewed source module changed",
                    ),
                ):
                    with CHECKER._registered_python_tooling_modules(
                        (invalid_reviewed,)
                    ):
                        self.fail("unreviewed source must not execute")
                rejected_compile.assert_not_called()

            wrong_filename_code = compile(b"", "forged-registry.py", "exec")
            for guard_name, compiled in (
                ("code-type", object()),
                ("code-filename", wrong_filename_code),
            ):
                with (
                    self.subTest(python_registry_compile_guard=guard_name),
                    mock.patch("builtins.compile", return_value=compiled),
                    self.assertRaisesRegex(
                        CHECKER.InventoryError,
                        "compiled code binding changed",
                    ),
                ):
                    with CHECKER._registered_python_tooling_modules(
                        (reviewed_registry_module,)
                    ):
                        self.fail("unverified compiled code must not execute")
            replacing_source_bytes = (
                "import sys\n"
                "import types\n"
                "sys.modules[__name__] = types.ModuleType(__name__)\n"
            ).encode("utf-8")
            registry_source.write_bytes(replacing_source_bytes)
            replacing_review = CHECKER._PythonReviewedSourceModule(
                "bench/tools/test_registry_fixture.py",
                "test_registry_fixture",
                registry_source,
                replacing_source_bytes,
                hashlib.sha256(replacing_source_bytes).hexdigest(),
            )
            with self.assertRaisesRegex(
                CHECKER.InventoryError, "source module binding changed"
            ):
                with CHECKER._registered_python_tooling_modules((replacing_review,)):
                    self.fail("replaced source module must not be yielded")
        finally:
            registry_source.unlink(missing_ok=True)

        class CanonicalOutcomeCase(unittest.TestCase):
            def declared_success(self) -> None:
                pass

            def declared_failure(self) -> None:
                self.fail("reviewed failure")

            def declared_error(self) -> None:
                raise RuntimeError("reviewed error")

            def declared_subtest_failure(self) -> None:
                with self.subTest(reviewed=True):
                    self.fail("reviewed subtest failure")

            def declared_subtest_error(self) -> None:
                with self.subTest(reviewed=True):
                    raise RuntimeError("reviewed subtest error")

            @unittest.expectedFailure
            def declared_expected_failure(self) -> None:
                self.fail("reviewed expected failure")

            @unittest.expectedFailure
            def declared_unexpected_success(self) -> None:
                pass

        for outcome_name, expected_counts in (
            ("declared_success", (True, 0, 0, 0, 0)),
            ("declared_failure", (False, 1, 0, 0, 0)),
            ("declared_error", (False, 0, 1, 0, 0)),
            ("declared_subtest_failure", (False, 1, 0, 0, 0)),
            ("declared_subtest_error", (False, 0, 1, 0, 0)),
            ("declared_expected_failure", (False, 0, 0, 1, 0)),
            ("declared_unexpected_success", (False, 0, 0, 0, 1)),
        ):
            with self.subTest(python_outcome_ledger=outcome_name):
                outcome_test = CanonicalOutcomeCase(outcome_name)
                self.assertEqual(
                    CHECKER._PythonToolingOutcome(
                        executed=1,
                        successful=expected_counts[0],
                        skips=frozenset(),
                        failures=expected_counts[1],
                        errors=expected_counts[2],
                        expected_failures=expected_counts[3],
                        unexpected_successes=expected_counts[4],
                    ),
                    execute_frozen_suite(
                        unittest.TestSuite([outcome_test]), (outcome_test,)
                    ),
                )

        decorator_reason = "reviewed decorator skip"

        class DecoratorSkipOutcomeCase(unittest.TestCase):
            @unittest.skip(decorator_reason)
            def declared_skip(self) -> None:
                self.fail("decorator-skipped body must not execute")

        decorator_skip_test = DecoratorSkipOutcomeCase("declared_skip")
        decorator_skip_id = CHECKER._canonical_python_test_id(
            DecoratorSkipOutcomeCase, "declared_skip"
        )
        self.assertEqual(
            CHECKER._PythonToolingOutcome(
                executed=1,
                successful=True,
                skips=frozenset({(decorator_skip_id, decorator_reason)}),
                failures=0,
                errors=0,
                expected_failures=0,
                unexpected_successes=0,
            ),
            execute_frozen_suite(
                unittest.TestSuite([decorator_skip_test]),
                (decorator_skip_test,),
                frozenset({(decorator_skip_id, decorator_reason)}),
            ),
        )

        platform_applicability_events: list[str] = []

        class PlatformApplicabilityCase(unittest.TestCase):
            def setUp(self) -> None:
                platform_applicability_events.append("setup")

            @unittest.skipUnless(False, "subordinate capability skip")
            def declared_platform_test(self) -> None:
                platform_applicability_events.append("body")

        platform_test = PlatformApplicabilityCase("declared_platform_test")
        platform_binding = frozen_runtime_binding(platform_test)
        platform_reason = "reviewed inventory platform applicability"
        platform_pair = (platform_binding.runtime_id, platform_reason)
        platform_contract = CHECKER._PythonToolingSuiteContract(
            1,
            frozenset(),
            frozenset(),
            (),
            frozenset({platform_pair}),
            (
                CHECKER._PythonPlatformSkipAuthorization(
                    platform_test,
                    platform_binding.runtime_id,
                    platform_reason,
                    "python-skip-predicate:artifact-snapshot-platform-unavailable",
                ),
            ),
            (platform_binding,),
        )
        with contextlib.redirect_stderr(io.StringIO()):
            platform_outcome = CHECKER._execute_python_tooling_suite(
                unittest.TestSuite([platform_test]),
                platform_contract,
                trusted_runtime,
            )
        self.assertEqual([], platform_applicability_events)
        self.assertEqual(frozenset({platform_pair}), platform_outcome.skips)
        self.assertEqual(1, platform_outcome.executed)

        inventory_errors = {
            phase: CHECKER.InventoryError(f"reviewed {phase} InventoryError")
            for phase in ("body", "setup", "teardown", "cleanup", "fixture")
        }

        class BodyInventoryErrorCase(unittest.TestCase):
            def declared_error(self) -> None:
                raise inventory_errors["body"]

        class SetupInventoryErrorCase(unittest.TestCase):
            def setUp(self) -> None:
                raise inventory_errors["setup"]

            def declared_error(self) -> None:
                pass

        class TeardownInventoryErrorCase(unittest.TestCase):
            def tearDown(self) -> None:
                raise inventory_errors["teardown"]

            def declared_error(self) -> None:
                pass

        class CleanupInventoryErrorCase(unittest.TestCase):
            def declared_error(self) -> None:
                def fail_cleanup() -> None:
                    raise inventory_errors["cleanup"]

                self.addCleanup(fail_cleanup)

        class FixtureInventoryErrorCase(unittest.TestCase):
            @classmethod
            def setUpClass(cls) -> None:
                raise inventory_errors["fixture"]

            def declared_error(self) -> None:
                pass

        for phase, case_type in (
            ("body", BodyInventoryErrorCase),
            ("setup", SetupInventoryErrorCase),
            ("teardown", TeardownInventoryErrorCase),
            ("cleanup", CleanupInventoryErrorCase),
            ("fixture", FixtureInventoryErrorCase),
        ):
            with self.subTest(python_inventory_error_phase=phase):
                inventory_error_test = case_type("declared_error")
                with self.assertRaises(CHECKER.InventoryError) as raised:
                    execute_frozen_suite(
                        unittest.TestSuite([inventory_error_test]),
                        (inventory_error_test,),
                    )
                self.assertIs(inventory_errors[phase], raised.exception)

        class FixtureErrorOutcomeCase(unittest.TestCase):
            @classmethod
            def setUpClass(cls) -> None:
                raise RuntimeError("reviewed fixture error")

            def declared_error(self) -> None:
                pass

        fixture_error_test = FixtureErrorOutcomeCase("declared_error")
        self.assertEqual(
            CHECKER._PythonToolingOutcome(
                executed=0,
                successful=False,
                skips=frozenset(),
                failures=0,
                errors=1,
                expected_failures=0,
                unexpected_successes=0,
            ),
            execute_frozen_suite(
                unittest.TestSuite([fixture_error_test]), (fixture_error_test,)
            ),
        )

        class ClearedOutcomeContainerCase(unittest.TestCase):
            def declared_mutation(self) -> None:
                result = self._outcome.result
                self.addCleanup(result.failures.clear)
                self.fail("populate the frozen failures container")

        class ReplacedOutcomeContainerCase(unittest.TestCase):
            def declared_mutation(self) -> None:
                self._outcome.result.errors = []

        class ForgedOutcomeCallbackCase(unittest.TestCase):
            def declared_mutation(self) -> None:
                self._outcome.result.addSuccess(self)

        class ForgedOutcomeContainerCase(unittest.TestCase):
            def declared_mutation(self) -> None:
                self._outcome.result.skipped.append((self, "forged skip"))

        class ForgedTestsRunCase(unittest.TestCase):
            def declared_mutation(self) -> None:
                self._outcome.result.testsRun = 0

        class ForgedSuiteStateCase(unittest.TestCase):
            def declared_entered(self) -> None:
                self._outcome.result._testRunEntered = False

            def declared_stop(self) -> None:
                self._outcome.result.shouldStop = True

            def declared_previous_class(self) -> None:
                self._outcome.result._previousTestClass = object

            def declared_module_failure(self) -> None:
                self._outcome.result._moduleSetUpFailed = True

        for mutation, case_type, method_name in (
            ("clear-container", ClearedOutcomeContainerCase, "declared_mutation"),
            (
                "replace-container",
                ReplacedOutcomeContainerCase,
                "declared_mutation",
            ),
            ("forge-container", ForgedOutcomeContainerCase, "declared_mutation"),
            ("forge-callback", ForgedOutcomeCallbackCase, "declared_mutation"),
            ("forge-tests-run", ForgedTestsRunCase, "declared_mutation"),
            ("suite-entered", ForgedSuiteStateCase, "declared_entered"),
            ("suite-stop", ForgedSuiteStateCase, "declared_stop"),
            (
                "suite-previous-class",
                ForgedSuiteStateCase,
                "declared_previous_class",
            ),
            (
                "suite-module-failure",
                ForgedSuiteStateCase,
                "declared_module_failure",
            ),
        ):
            with self.subTest(python_outcome_mutation=mutation):
                mutation_test = case_type(method_name)
                with self.assertRaises(CHECKER.InventoryError):
                    execute_frozen_suite(
                        unittest.TestSuite([mutation_test]), (mutation_test,)
                    )

        class AddErrorClassReplacementCase(unittest.TestCase):
            def declared_mutation(self) -> None:
                result = self._outcome.result
                type(result).addError = trusted_runtime.result_add_error

        add_error_class_replacement_test = AddErrorClassReplacementCase(
            "declared_mutation"
        )
        with self.assertRaisesRegex(
            CHECKER.InventoryError,
            "tampered result object",
        ):
            execute_frozen_suite(
                unittest.TestSuite([add_error_class_replacement_test]),
                (add_error_class_replacement_test,),
            )

        class AddErrorInstanceShadowCase(unittest.TestCase):
            def setUp(self) -> None:
                self._outcome.result.addError = trusted_runtime.result_add_error

            def declared_mutation(self) -> None:
                pass

        add_error_instance_shadow_test = AddErrorInstanceShadowCase("declared_mutation")
        with self.assertRaisesRegex(
            CHECKER.InventoryError,
            "tampered result object",
        ):
            execute_frozen_suite(
                unittest.TestSuite([add_error_instance_shadow_test]),
                (add_error_instance_shadow_test,),
            )

        fixture_mutation_body_ran = [False]

        class FixtureMutationCase(unittest.TestCase):
            def setUp(self) -> None:
                self.id = lambda: "forged.after-fixture"
                type(self).setUp = lambda fixture_self: None

            def declared_failure(self) -> None:
                fixture_mutation_body_ran[0] = True

        fixture_mutation_test = FixtureMutationCase("declared_failure")
        with self.assertRaises(CHECKER.InventoryError):
            execute_frozen_suite(
                unittest.TestSuite([fixture_mutation_test]),
                (fixture_mutation_test,),
            )
        self.assertFalse(fixture_mutation_body_ran[0])

        setup_exception_events: list[str] = []

        class SetupExceptionMutationCase(unittest.TestCase):
            def setUp(self) -> None:
                self.addCleanup(lambda: setup_exception_events.append("cleanup"))
                type(self).setUp = lambda fixture_self: None
                setup_exception_events.append("setup-enter")
                raise RuntimeError("setup failure after lifecycle mutation")

            def declared_failure(self) -> None:
                setup_exception_events.append("body")

        setup_exception_test = SetupExceptionMutationCase("declared_failure")
        with self.assertRaises(CHECKER.InventoryError):
            execute_frozen_suite(
                unittest.TestSuite([setup_exception_test]),
                (setup_exception_test,),
            )
        self.assertEqual(["setup-enter"], setup_exception_events)

        body_exception_events: list[str] = []

        class BodyExceptionMutationCase(unittest.TestCase):
            def setUp(self) -> None:
                self.addCleanup(lambda: body_exception_events.append("cleanup"))

            def declared_failure(self) -> None:
                self._callCleanup = lambda function, *args, **kwargs: None
                body_exception_events.append("body-enter")
                self.fail("body failure after internal hook mutation")

        body_exception_test = BodyExceptionMutationCase("declared_failure")
        with self.assertRaises(CHECKER.InventoryError):
            execute_frozen_suite(
                unittest.TestSuite([body_exception_test]),
                (body_exception_test,),
            )
        self.assertEqual(["body-enter"], body_exception_events)

        cleanup_return_events: list[str] = []

        class CleanupReturnMutationCase(unittest.TestCase):
            def declared_failure(self) -> None:
                self.addCleanup(lambda: cleanup_return_events.append("later-cleanup"))

                def mutate_cleanup_hook() -> None:
                    self._callCleanup = lambda function, *args, **kwargs: None
                    cleanup_return_events.append("mutating-cleanup")

                self.addCleanup(mutate_cleanup_hook)
                cleanup_return_events.append("body")

        cleanup_return_test = CleanupReturnMutationCase("declared_failure")
        with self.assertRaises(CHECKER.InventoryError):
            execute_frozen_suite(
                unittest.TestSuite([cleanup_return_test]),
                (cleanup_return_test,),
            )
        self.assertEqual(["body", "mutating-cleanup"], cleanup_return_events)

        cleanup_exception_events: list[str] = []

        class CleanupExceptionMutationCase(unittest.TestCase):
            def declared_failure(self) -> None:
                self.addCleanup(
                    lambda: cleanup_exception_events.append("later-cleanup")
                )

                def mutate_cleanup_hook() -> None:
                    self._callCleanup = lambda function, *args, **kwargs: None
                    cleanup_exception_events.append("mutating-cleanup")
                    raise RuntimeError("cleanup failure after hook mutation")

                self.addCleanup(mutate_cleanup_hook)
                cleanup_exception_events.append("body")

        cleanup_exception_test = CleanupExceptionMutationCase("declared_failure")
        with self.assertRaises(CHECKER.InventoryError):
            execute_frozen_suite(
                unittest.TestSuite([cleanup_exception_test]),
                (cleanup_exception_test,),
            )
        self.assertEqual(["body", "mutating-cleanup"], cleanup_exception_events)

        class AsyncFixtureCase(unittest.TestCase):
            async def setUp(self) -> None:
                pass

            def declared_failure(self) -> None:
                pass

        class GeneratorFixtureCase(unittest.TestCase):
            def tearDown(self) -> Any:
                yield None

            def declared_failure(self) -> None:
                pass

        for fixture_shape, case_type in (
            ("async-setup", AsyncFixtureCase),
            ("generator-teardown", GeneratorFixtureCase),
        ):
            with (
                self.subTest(python_unittest_fixture_shape=fixture_shape),
                self.assertRaisesRegex(
                    CHECKER.InventoryError,
                    "noncanonical synchronous callable",
                ),
            ):
                frozen_runtime_binding(case_type("declared_failure"))

        duplicate_events: list[str] = []
        duplicate_suite: unittest.TestSuite

        class DuplicateChildCase(unittest.TestCase):
            @classmethod
            def setUpClass(cls) -> None:
                duplicate_suite._tests[1] = duplicate_suite._tests[0]

            def declared_a(self) -> None:
                duplicate_events.append("a")

            def declared_b(self) -> None:
                duplicate_events.append("b")

        duplicate_a = DuplicateChildCase("declared_a")
        duplicate_b = DuplicateChildCase("declared_b")
        duplicate_suite = unittest.TestSuite([duplicate_a, duplicate_b])
        with self.assertRaisesRegex(
            CHECKER.InventoryError,
            "duplicate or unknown test object",
        ):
            execute_frozen_suite(
                duplicate_suite,
                (duplicate_a, duplicate_b),
            )
        self.assertEqual(["a"], duplicate_events)

        class_fixture_events: list[str] = []

        class NextFixtureCase(unittest.TestCase):
            @classmethod
            def setUpClass(cls) -> None:
                class_fixture_events.append("next-setup")

            def declared_next(self) -> None:
                class_fixture_events.append("next-body")

        next_setup_descriptor = NextFixtureCase.__dict__["setUpClass"]

        class PreviousFixtureCase(unittest.TestCase):
            @classmethod
            def tearDownClass(cls) -> None:
                def forged_next_setup(next_cls: type[Any]) -> None:
                    NextFixtureCase.setUpClass = next_setup_descriptor
                    class_fixture_events.append("forged-next-setup")

                NextFixtureCase.setUpClass = classmethod(forged_next_setup)

            def declared_previous(self) -> None:
                class_fixture_events.append("previous-body")

        previous_fixture_test = PreviousFixtureCase("declared_previous")
        next_fixture_test = NextFixtureCase("declared_next")
        with self.assertRaises(CHECKER.InventoryError):
            execute_frozen_suite(
                unittest.TestSuite([previous_fixture_test, next_fixture_test]),
                (previous_fixture_test, next_fixture_test),
            )
        self.assertEqual(["previous-body"], class_fixture_events)

        class_setup_error_events: list[str] = []

        class ClassSetupErrorMutationCase(unittest.TestCase):
            @classmethod
            def setUpClass(cls) -> None:
                cls.setUpClass = classmethod(lambda setup_cls: None)
                class_setup_error_events.append("class-setup")
                raise RuntimeError("class setup failure after binding mutation")

            def declared_failure(self) -> None:
                class_setup_error_events.append("body")

        class_setup_error_test = ClassSetupErrorMutationCase("declared_failure")
        with self.assertRaises(CHECKER.InventoryError):
            execute_frozen_suite(
                unittest.TestSuite([class_setup_error_test]),
                (class_setup_error_test,),
            )
        self.assertEqual(["class-setup"], class_setup_error_events)

        previous_teardown_events: list[str] = []

        class PreviousTeardownMutationCase(unittest.TestCase):
            @classmethod
            def tearDownClass(cls) -> None:
                cls.tearDownClass = classmethod(lambda teardown_cls: None)
                previous_teardown_events.append("previous-teardown")

            def declared_previous(self) -> None:
                previous_teardown_events.append("previous-body")

        class FollowingTeardownCase(unittest.TestCase):
            def declared_following(self) -> None:
                previous_teardown_events.append("following-body")

        previous_teardown_test = PreviousTeardownMutationCase("declared_previous")
        following_teardown_test = FollowingTeardownCase("declared_following")
        with self.assertRaises(CHECKER.InventoryError):
            execute_frozen_suite(
                unittest.TestSuite([previous_teardown_test, following_teardown_test]),
                (previous_teardown_test, following_teardown_test),
            )
        self.assertEqual(
            ["previous-body", "previous-teardown"],
            previous_teardown_events,
        )

        module_fixture_events: list[str] = []

        def next_module_setup() -> None:
            module_fixture_events.append("next-module-setup")

        next_module_owner: dict[str, Any] = {}

        def previous_module_teardown() -> None:
            def forged_next_setup() -> None:
                del next_module_owner["setUpModule"]
                module_fixture_events.append("forged-next-module-setup")

            next_module_owner["setUpModule"] = forged_next_setup
            module_fixture_events.append("previous-module-teardown")

        previous_module_owner = {"tearDownModule": previous_module_teardown}
        previous_module_binding = self._synthetic_module_binding(previous_module_owner)
        previous_module_owner = previous_module_binding.namespace
        next_module_binding = self._synthetic_module_binding(next_module_owner)
        next_module_owner = next_module_binding.namespace
        previous_module_fixture = CHECKER._PythonFixtureBinding(
            "module",
            previous_module_binding,
            "tearDownModule",
            True,
            previous_module_teardown,
            previous_module_teardown,
            previous_module_teardown.__code__,
        )
        next_module_fixture = CHECKER._PythonFixtureBinding(
            "module",
            next_module_binding,
            "setUpModule",
            False,
            None,
            None,
            None,
        )
        with self.assertRaises(CHECKER.InventoryError):
            CHECKER._verify_python_module_fixture_transition(
                (previous_module_fixture,),
                (next_module_fixture,),
                previous_module_teardown,
            )
        self.assertEqual(["previous-module-teardown"], module_fixture_events)

        present_module_owner = {"setUpModule": next_module_setup}
        present_module_binding = self._synthetic_module_binding(present_module_owner)
        present_module_owner = present_module_binding.namespace
        present_module_fixture = CHECKER._PythonFixtureBinding(
            "module",
            present_module_binding,
            "setUpModule",
            True,
            next_module_setup,
            next_module_setup,
            next_module_setup.__code__,
        )

        def present_none_method(test_self: unittest.TestCase) -> None:
            pass

        present_none_module = self._synthetic_module_binding({"setUpModule": None})
        PresentNoneModuleFixtureCase = type(
            "PresentNoneModuleFixtureCase",
            (unittest.TestCase,),
            {
                "__module__": present_none_module.name,
                "declared_present_none": present_none_method,
            },
        )
        present_none_module.namespace["PresentNoneModuleFixtureCase"] = (
            PresentNoneModuleFixtureCase
        )
        present_none_test = PresentNoneModuleFixtureCase("declared_present_none")
        present_none_fixtures = CHECKER._freeze_python_fixture_bindings(
            present_none_test,
            PresentNoneModuleFixtureCase,
            present_none_module,
        )
        frozen_present_none = next(
            fixture
            for fixture in present_none_fixtures
            if fixture.kind == "module" and fixture.name == "setUpModule"
        )
        self.assertTrue(frozen_present_none.present)
        self.assertIsNone(frozen_present_none.descriptor)
        CHECKER._verify_python_fixture_binding(frozen_present_none)

        def replacement_module_setup() -> None:
            module_fixture_events.append("replacement-module-setup")

        def replace_present_module_setup() -> None:
            present_module_owner["setUpModule"] = replacement_module_setup

        with self.assertRaises(CHECKER.InventoryError):
            CHECKER._verify_python_module_fixture_transition(
                (),
                (present_module_fixture,),
                replace_present_module_setup,
            )
        present_module_owner["setUpModule"] = next_module_setup
        present_module_owner["setUpModule"] = None
        with self.assertRaises(CHECKER.InventoryError):
            CHECKER._verify_python_fixture_binding(present_module_fixture)
        present_module_owner["setUpModule"] = next_module_setup
        del present_module_owner["setUpModule"]
        with self.assertRaises(CHECKER.InventoryError):
            CHECKER._verify_python_fixture_binding(present_module_fixture)

        absent_teardown_owner: dict[str, Any] = {}
        absent_teardown_binding = self._synthetic_module_binding(absent_teardown_owner)
        absent_teardown_owner = absent_teardown_binding.namespace
        absent_teardown_fixture = CHECKER._PythonFixtureBinding(
            "module",
            absent_teardown_binding,
            "tearDownModule",
            False,
            None,
            None,
            None,
        )

        def injected_module_teardown() -> None:
            module_fixture_events.append("injected-module-teardown")

        absent_teardown_owner["tearDownModule"] = injected_module_teardown
        with self.assertRaises(CHECKER.InventoryError):
            CHECKER._verify_python_fixture_binding(absent_teardown_fixture)
        absent_teardown_owner["tearDownModule"] = None
        with self.assertRaises(CHECKER.InventoryError):
            CHECKER._verify_python_fixture_binding(absent_teardown_fixture)

        module_setup_error_events: list[str] = []

        def module_setup_error() -> None:
            module_setup_error_owner["setUpModule"] = replacement_module_setup
            module_setup_error_events.append("module-setup")
            raise RuntimeError("module setup failure after binding mutation")

        module_setup_error_owner = {"setUpModule": module_setup_error}
        module_setup_error_binding = self._synthetic_module_binding(
            module_setup_error_owner
        )
        module_setup_error_owner = module_setup_error_binding.namespace
        module_setup_error_fixture = CHECKER._PythonFixtureBinding(
            "module",
            module_setup_error_binding,
            "setUpModule",
            True,
            module_setup_error,
            module_setup_error,
            module_setup_error.__code__,
        )
        with self.assertRaises(CHECKER.InventoryError):
            CHECKER._invoke_python_fixture_helper(
                module_setup_error,
                lambda: CHECKER._verify_python_fixture_binding(
                    module_setup_error_fixture
                ),
            )
        self.assertEqual(["module-setup"], module_setup_error_events)

        adjacent_module_events: list[str] = []
        adjacent_absent_owner: dict[str, Any] = {}
        adjacent_absent_binding = self._synthetic_module_binding(adjacent_absent_owner)
        adjacent_absent_owner = adjacent_absent_binding.namespace
        adjacent_absent_fixture = CHECKER._PythonFixtureBinding(
            "module",
            adjacent_absent_binding,
            "setUpModule",
            False,
            None,
            None,
            None,
        )

        def forged_absent_module_setup() -> None:
            del adjacent_absent_owner["setUpModule"]
            adjacent_module_events.append("forged-absent-module-setup")

        def previous_class_injects_absent_setup() -> None:
            adjacent_absent_owner["setUpModule"] = forged_absent_module_setup
            adjacent_module_events.append("previous-class-teardown-absent")

        CHECKER._invoke_python_fixture_helper(
            previous_class_injects_absent_setup, lambda: None
        )
        with self.assertRaises(CHECKER.InventoryError):
            CHECKER._invoke_python_fixture_helper(
                forged_absent_module_setup,
                lambda: CHECKER._verify_python_fixture_binding(adjacent_absent_fixture),
            )
        self.assertEqual(["previous-class-teardown-absent"], adjacent_module_events)

        adjacent_present_owner = {"setUpModule": next_module_setup}
        adjacent_present_binding = self._synthetic_module_binding(
            adjacent_present_owner
        )
        adjacent_present_owner = adjacent_present_binding.namespace
        adjacent_present_fixture = CHECKER._PythonFixtureBinding(
            "module",
            adjacent_present_binding,
            "setUpModule",
            True,
            next_module_setup,
            next_module_setup,
            next_module_setup.__code__,
        )

        def forged_present_module_setup() -> None:
            adjacent_present_owner["setUpModule"] = next_module_setup
            adjacent_module_events.append("forged-present-module-setup")

        def previous_class_replaces_present_setup() -> None:
            adjacent_present_owner["setUpModule"] = forged_present_module_setup
            adjacent_module_events.append("previous-class-teardown-present")

        CHECKER._invoke_python_fixture_helper(
            previous_class_replaces_present_setup, lambda: None
        )
        with self.assertRaises(CHECKER.InventoryError):
            CHECKER._invoke_python_fixture_helper(
                forged_present_module_setup,
                lambda: CHECKER._verify_python_fixture_binding(
                    adjacent_present_fixture
                ),
            )
        self.assertEqual(
            [
                "previous-class-teardown-absent",
                "previous-class-teardown-present",
            ],
            adjacent_module_events,
        )

        adjacent_class_events: list[str] = []

        class AdjacentClassSetupCase(unittest.TestCase):
            @classmethod
            def setUpClass(cls) -> None:
                adjacent_class_events.append("original-class-setup")

        adjacent_class_descriptor = AdjacentClassSetupCase.__dict__["setUpClass"]
        adjacent_class_bound = AdjacentClassSetupCase.setUpClass
        adjacent_class_fixture = CHECKER._PythonFixtureBinding(
            "class",
            AdjacentClassSetupCase,
            "setUpClass",
            True,
            adjacent_class_descriptor,
            adjacent_class_bound,
            adjacent_class_descriptor.__func__.__code__,
        )

        def forged_adjacent_class_setup(cls: type[Any]) -> None:
            AdjacentClassSetupCase.setUpClass = adjacent_class_descriptor
            adjacent_class_events.append("forged-class-setup")

        def module_setup_injects_class_wrapper() -> None:
            AdjacentClassSetupCase.setUpClass = classmethod(forged_adjacent_class_setup)
            adjacent_class_events.append("module-setup")

        CHECKER._invoke_python_fixture_helper(
            module_setup_injects_class_wrapper, lambda: None
        )
        with self.assertRaises(CHECKER.InventoryError):
            CHECKER._invoke_python_fixture_helper(
                AdjacentClassSetupCase.setUpClass,
                lambda: CHECKER._verify_python_fixture_binding(adjacent_class_fixture),
            )
        self.assertEqual(["module-setup"], adjacent_class_events)

        forged_loader = trusted_runtime.loader_type()
        forged_loader.suiteClass = ForgedSuite
        with self.assertRaisesRegex(
            CHECKER.InventoryError,
            "suiteClass is noncanonical",
        ):
            CHECKER._verify_python_test_loader_instance(forged_loader, trusted_runtime)
        with (
            mock.patch.object(
                CHECKER.unittest.TestLoader,
                "suiteClass",
                ForgedSuite,
            ),
            self.assertRaisesRegex(
                CHECKER.InventoryError,
                "trusted unittest runtime primitive",
            ),
        ):
            CHECKER._verify_python_unittest_runtime_primitives(
                trusted_runtime, trusted_runtime.test_case_skip_test
            )
        result_class = authorizer.result_class()
        authorized_methods = {
            name: getattr(result_class, name)
            for name in CHECKER._PYTHON_RESULT_CALLBACK_NAMES
        }
        with self.assertRaisesRegex(
            CHECKER.InventoryError,
            "tampered result object",
        ):
            CHECKER._verify_python_tooling_result_integrity(
                mock.Mock(),
                result_class,
                authorized_methods,
                authorizer.outcome_ledger,
            )
        empty_result_contract = CHECKER._PythonToolingSuiteContract(
            0, frozenset(), frozenset(), (), frozenset(), (), ()
        )
        for callback_name in CHECKER._PYTHON_RESULT_CALLBACK_NAMES:
            with self.subTest(python_result_class_callback=callback_name):
                callback_authorizer = CHECKER._PythonSkipRuntimeAuthorizer(
                    empty_result_contract, trusted_runtime
                )
                callback_result_class = callback_authorizer.result_class()
                callback_methods = {
                    name: getattr(callback_result_class, name)
                    for name in CHECKER._PYTHON_RESULT_CALLBACK_NAMES
                }
                callback_result = callback_result_class(io.StringIO(), True, 1)
                setattr(callback_result_class, callback_name, lambda *args: None)
                with self.assertRaisesRegex(
                    CHECKER.InventoryError,
                    "tampered result object",
                ):
                    CHECKER._verify_python_tooling_result_integrity(
                        callback_result,
                        callback_result_class,
                        callback_methods,
                        callback_authorizer.outcome_ledger,
                    )
            with self.subTest(python_result_instance_callback=callback_name):
                callback_authorizer = CHECKER._PythonSkipRuntimeAuthorizer(
                    empty_result_contract, trusted_runtime
                )
                callback_result_class = callback_authorizer.result_class()
                callback_methods = {
                    name: getattr(callback_result_class, name)
                    for name in CHECKER._PYTHON_RESULT_CALLBACK_NAMES
                }
                callback_result = callback_result_class(io.StringIO(), True, 1)
                setattr(callback_result, callback_name, lambda *args: None)
                with self.assertRaisesRegex(
                    CHECKER.InventoryError,
                    "tampered result object",
                ):
                    CHECKER._verify_python_tooling_result_integrity(
                        callback_result,
                        callback_result_class,
                        callback_methods,
                        callback_authorizer.outcome_ledger,
                    )

        source_current_bytes = self.inventory_path.read_bytes()
        source_current_digest = hashlib.sha256(source_current_bytes).hexdigest()
        expected_count = tooling_set["count"]
        suite = real_suites[0]
        suite_contract = real_suite_contracts[0]
        extra_skip_pairs = (
            suite_contract.required_decorator_skips
            | suite_contract.permitted_dynamic_skips
        )

        skipped_test = mock.Mock()
        predicate_false_skip = next(
            entry
            for entry in source_current["python_skip_contracts"][0]["entries"]
            if (
                CHECKER._unittest_runtime_id(entry["test"]),
                entry["reason"],
            )
            not in extra_skip_pairs
            and entry["predicate_id"]
            != "python-skip-predicate:report-publication-platform-unavailable"
        )
        skipped_test.id.return_value = CHECKER._unittest_runtime_id(
            predicate_false_skip["test"]
        )
        unexpected_skip_result = mock.Mock(
            testsRun=expected_count,
            skipped=[(skipped_test, predicate_false_skip["reason"])],
            expectedFailures=[],
            unexpectedSuccesses=[],
        )
        unexpected_skip_result.wasSuccessful.return_value = True
        with (
            self._reviewed_digest_slots(inventory_current=source_current_digest),
            mock.patch.object(
                CHECKER.unittest.TestLoader,
                "discover",
                return_value=suite,
            ),
            mock.patch.object(
                CHECKER,
                "_execute_python_tooling_suite",
                return_value=(
                    CHECKER._PythonToolingOutcome(
                        expected_count,
                        True,
                        frozenset(
                            {
                                (
                                    skipped_test.id(),
                                    predicate_false_skip["reason"],
                                )
                            },
                        ),
                        0,
                        0,
                        0,
                        0,
                    )
                ),
            ),
            mock.patch.object(CHECKER.os, "getcwd", return_value=str(self.root)),
            mock.patch.object(
                CHECKER,
                "_python_tooling_suite_contract",
                return_value=suite_contract,
            ),
            contextlib.redirect_stderr(io.StringIO()) as skip_stderr,
        ):
            result = CHECKER.main(
                [
                    "--root",
                    str(self.root),
                    "--inventory",
                    str(self.inventory_path),
                    "--run-python-tooling-root",
                    CHECKER.PYTHON_TOOLING_ROOT_ID,
                ]
            )
        self.assertEqual(1, result)
        self.assertIn("unexpected_skips", skip_stderr.getvalue())
        self.assertIn(predicate_false_skip["reason"], skip_stderr.getvalue())
        self.assertEqual(expected_count, unexpected_skip_result.testsRun)

        missing_dynamic_pair = (
            skipped_test.id(),
            predicate_false_skip["reason"],
        )
        missing_dynamic_contract = suite_contract._replace(
            permitted_dynamic_skips=frozenset({missing_dynamic_pair})
        )
        with (
            self._reviewed_digest_slots(inventory_current=source_current_digest),
            mock.patch.object(
                CHECKER.unittest.TestLoader,
                "discover",
                return_value=suite,
            ),
            mock.patch.object(
                CHECKER,
                "_execute_python_tooling_suite",
                return_value=CHECKER._PythonToolingOutcome(
                    expected_count, True, frozenset(), 0, 0, 0, 0
                ),
            ),
            mock.patch.object(CHECKER.os, "getcwd", return_value=str(self.root)),
            mock.patch.object(
                CHECKER,
                "_python_tooling_suite_contract",
                return_value=missing_dynamic_contract,
            ),
            contextlib.redirect_stderr(io.StringIO()) as missing_dynamic_stderr,
        ):
            result = CHECKER.main(
                [
                    "--root",
                    str(self.root),
                    "--inventory",
                    str(self.inventory_path),
                    "--run-python-tooling-root",
                    CHECKER.PYTHON_TOOLING_ROOT_ID,
                ]
            )
        self.assertEqual(1, result)
        self.assertIn("missing_dynamic_skips", missing_dynamic_stderr.getvalue())

        for outcome_name, expected_error in (
            ("expectedFailures", "expected failures"),
            ("unexpectedSuccesses", "unexpected successes"),
        ):
            with self.subTest(rejected_unittest_outcome=outcome_name):
                with (
                    self._reviewed_digest_slots(
                        inventory_current=source_current_digest
                    ),
                    mock.patch.object(
                        CHECKER.unittest.TestLoader,
                        "discover",
                        return_value=suite,
                    ),
                    mock.patch.object(
                        CHECKER,
                        "_execute_python_tooling_suite",
                        side_effect=CHECKER.InventoryError(expected_error),
                    ),
                    mock.patch.object(
                        CHECKER.os, "getcwd", return_value=str(self.root)
                    ),
                    mock.patch.object(
                        CHECKER,
                        "_python_tooling_suite_contract",
                        return_value=suite_contract,
                    ),
                    contextlib.redirect_stderr(io.StringIO()) as outcome_stderr,
                ):
                    result = CHECKER.main(
                        [
                            "--root",
                            str(self.root),
                            "--inventory",
                            str(self.inventory_path),
                            "--run-python-tooling-root",
                            CHECKER.PYTHON_TOOLING_ROOT_ID,
                        ]
                    )
                self.assertEqual(1, result)
                self.assertIn(expected_error, outcome_stderr.getvalue())

        base_cli = [
            "--root",
            str(self.root),
            "--inventory",
            str(self.inventory_path),
            "--run-python-tooling-root",
            CHECKER.PYTHON_TOOLING_ROOT_ID,
        ]
        incompatible_options = (
            ["--structure-only"],
            ["--require-current-only"],
            ["--refresh-from-protocol"],
            ["--protocol-log", "env:aarch64-macos-baseline=unused.log"],
        )
        for option in incompatible_options:
            with (
                self.subTest(incompatible_option=option[0]),
                mock.patch.object(CHECKER, "_run_python_tooling_root") as run_root,
                contextlib.redirect_stderr(io.StringIO()) as incompatible_stderr,
            ):
                result = CHECKER.main([*base_cli, *option])
                self.assertEqual(2, result)
                run_root.assert_not_called()
                self.assertIn("incompatible", incompatible_stderr.getvalue())

        admission_cases: list[tuple[str, Path, str]] = []
        symlink_inventory = self.root / "tools/test_inventory-symlink.json"
        symlink_inventory.symlink_to(self.inventory_path.name)
        admission_cases.append(("symlink", symlink_inventory, "cannot read"))
        fifo_inventory = self.root / "tools/test_inventory-fifo.json"
        os.mkfifo(fifo_inventory)
        admission_cases.append(("fifo", fifo_inventory, "not a regular file"))
        oversized_inventory = self.root / "tools/test_inventory-oversized.json"
        oversized_inventory.write_bytes(b"x" * (CHECKER.MAX_INVENTORY_BYTES + 1))
        admission_cases.append(("oversized", oversized_inventory, "exceeds"))
        wrong_digest_inventory = self.root / "tools/test_inventory-wrong-digest.json"
        wrong_digest_inventory.write_bytes(source_current_bytes + b" ")
        admission_cases.append(
            ("wrong-digest", wrong_digest_inventory, "reviewed whole-file")
        )
        try:
            for label, inventory_path, expected_error in admission_cases:
                with (
                    self.subTest(inventory_admission=label),
                    self._reviewed_digest_slots(
                        inventory_current=source_current_digest
                    ),
                    mock.patch.object(
                        CHECKER.unittest.TestLoader, "discover"
                    ) as rejected_discovery,
                    contextlib.redirect_stderr(io.StringIO()) as admission_stderr,
                ):
                    result = CHECKER.main(
                        [
                            "--root",
                            str(self.root),
                            "--inventory",
                            str(inventory_path),
                            "--run-python-tooling-root",
                            CHECKER.PYTHON_TOOLING_ROOT_ID,
                        ]
                    )
                    self.assertEqual(1, result)
                    rejected_discovery.assert_not_called()
                    self.assertIn(expected_error, admission_stderr.getvalue())
        finally:
            symlink_inventory.unlink(missing_ok=True)
            fifo_inventory.unlink(missing_ok=True)
            oversized_inventory.unlink(missing_ok=True)
            wrong_digest_inventory.unlink(missing_ok=True)

        duplicate_inventory = self.root / "tools/test_inventory-duplicate.json"
        duplicate_bytes = source_current_bytes.replace(
            b'{\n  "schema_id":',
            b'{\n  "schema_id": "duplicate",\n  "schema_id":',
            1,
        )
        duplicate_inventory.write_bytes(duplicate_bytes)
        try:
            with (
                self._reviewed_digest_slots(
                    inventory_next=hashlib.sha256(duplicate_bytes).hexdigest()
                ),
                mock.patch.object(
                    CHECKER, "_validate_inventory_data"
                ) as rejected_validation,
                contextlib.redirect_stderr(io.StringIO()) as duplicate_stderr,
            ):
                result = CHECKER.main(
                    [
                        "--root",
                        str(self.root),
                        "--inventory",
                        str(duplicate_inventory),
                        "--run-python-tooling-root",
                        CHECKER.PYTHON_TOOLING_ROOT_ID,
                    ]
                )
            self.assertEqual(1, result)
            rejected_validation.assert_not_called()
            self.assertIn("duplicate JSON object key", duplicate_stderr.getvalue())
        finally:
            duplicate_inventory.unlink(missing_ok=True)

        with (
            self._reviewed_digest_slots(inventory_current=source_current_digest),
            mock.patch.object(CHECKER, "_validate_inventory_data", return_value=[]),
            mock.patch.object(
                CHECKER.unittest.TestLoader, "discover"
            ) as wrong_root_discovery,
            contextlib.redirect_stderr(io.StringIO()) as wrong_root_stderr,
        ):
            result = CHECKER.main(
                [
                    "--root",
                    str(self.root),
                    "--inventory",
                    str(self.inventory_path),
                    "--run-python-tooling-root",
                    "python-root:wrong",
                ]
            )
        self.assertEqual(1, result)
        wrong_root_discovery.assert_not_called()
        self.assertIn("resolve uniquely", wrong_root_stderr.getvalue())

        zero_count_inventory = copy.deepcopy(source_current)
        next(
            row
            for row in zero_count_inventory["expected_test_sets"]
            if row["root_id"] == CHECKER.PYTHON_TOOLING_ROOT_ID
        )["count"] = 0
        zero_count_bytes = CHECKER._canonical_inventory_bytes(zero_count_inventory)
        self.inventory_path.write_bytes(zero_count_bytes)
        with (
            self._reviewed_digest_slots(
                inventory_next=hashlib.sha256(zero_count_bytes).hexdigest()
            ),
            mock.patch.object(CHECKER, "_validate_inventory_data", return_value=[]),
            mock.patch.object(
                CHECKER.unittest.TestLoader, "discover"
            ) as zero_count_discovery,
            contextlib.redirect_stderr(io.StringIO()) as zero_count_stderr,
        ):
            result = CHECKER.main(base_cli)
        self.assertEqual(1, result)
        zero_count_discovery.assert_not_called()
        self.assertIn("expected count must be positive", zero_count_stderr.getvalue())
        self.inventory_path.write_bytes(source_current_bytes)

        escaped_inventory = copy.deepcopy(source_current)
        next(
            row
            for row in escaped_inventory["test_roots"]
            if row["id"] == CHECKER.PYTHON_TOOLING_ROOT_ID
        )["discovery_start"] = "../bench/tools"
        escaped_bytes = CHECKER._canonical_inventory_bytes(escaped_inventory)
        self.inventory_path.write_bytes(escaped_bytes)
        with (
            self._reviewed_digest_slots(
                inventory_next=hashlib.sha256(escaped_bytes).hexdigest()
            ),
            mock.patch.object(CHECKER, "_validate_inventory_data", return_value=[]),
            mock.patch.object(
                CHECKER.unittest.TestLoader, "discover"
            ) as escaped_discovery,
            contextlib.redirect_stderr(io.StringIO()) as escaped_stderr,
        ):
            result = CHECKER.main(base_cli)
        self.assertEqual(1, result)
        escaped_discovery.assert_not_called()
        self.assertIn("escapes the repository", escaped_stderr.getvalue())
        self.inventory_path.write_bytes(source_current_bytes)

        with (
            self._reviewed_digest_slots(inventory_current=source_current_digest),
            mock.patch.object(CHECKER, "_validate_inventory_data", return_value=[]),
            mock.patch.object(
                CHECKER.unittest.TestLoader, "discover"
            ) as wrong_cwd_discovery,
            contextlib.redirect_stderr(io.StringIO()) as wrong_cwd_stderr,
        ):
            result = CHECKER.main(base_cli)
        self.assertEqual(1, result)
        wrong_cwd_discovery.assert_not_called()
        self.assertIn(
            "requires the repository root as cwd", wrong_cwd_stderr.getvalue()
        )

        execution_cases = (
            (
                "discovery-drift",
                expected_count - 1,
                CHECKER._PythonToolingOutcome(
                    expected_count - 1, True, frozenset(), 0, 0, 0, 0
                ),
            ),
            (
                "executed-drift",
                expected_count,
                CHECKER._PythonToolingOutcome(
                    expected_count - 1, True, frozenset(), 0, 0, 0, 0
                ),
            ),
            (
                "result-failure",
                expected_count,
                CHECKER._PythonToolingOutcome(
                    expected_count, False, frozenset(), 0, 0, 0, 0
                ),
            ),
            (
                "failure-count",
                expected_count,
                CHECKER._PythonToolingOutcome(
                    expected_count, True, frozenset(), 1, 0, 0, 0
                ),
            ),
            (
                "error-count",
                expected_count,
                CHECKER._PythonToolingOutcome(
                    expected_count, True, frozenset(), 0, 1, 0, 0
                ),
            ),
            (
                "expected-failure-count",
                expected_count,
                CHECKER._PythonToolingOutcome(
                    expected_count, True, frozenset(), 0, 0, 1, 0
                ),
            ),
            (
                "unexpected-success-count",
                expected_count,
                CHECKER._PythonToolingOutcome(
                    expected_count, True, frozenset(), 0, 0, 0, 1
                ),
            ),
        )
        for label, discovered, execution_outcome in execution_cases:
            case_suite = mock.Mock()
            case_suite.countTestCases.return_value = discovered
            with (
                self.subTest(python_tooling_execution=label),
                self._reviewed_digest_slots(inventory_current=source_current_digest),
                mock.patch.object(CHECKER, "_validate_inventory_data", return_value=[]),
                mock.patch.object(
                    CHECKER.unittest.TestLoader,
                    "discover",
                    return_value=case_suite,
                ),
                mock.patch.object(
                    CHECKER,
                    "_execute_python_tooling_suite",
                    return_value=execution_outcome,
                ),
                mock.patch.object(
                    CHECKER,
                    "_python_tooling_suite_contract",
                    return_value=CHECKER._PythonToolingSuiteContract(
                        discovered,
                        frozenset(),
                        frozenset(),
                        (),
                        frozenset(),
                        (),
                        (),
                    ),
                ),
                mock.patch.object(CHECKER.os, "getcwd", return_value=str(self.root)),
                contextlib.redirect_stderr(io.StringIO()) as execution_stderr,
            ):
                result = CHECKER.main(base_cli)
            self.assertEqual(1, result)
            self.assertIn("test contract failed", execution_stderr.getvalue())

        root_mutations = (
            ("launch-missing", "launch_observation_ids", []),
            (
                "launch-wrong",
                "launch_observation_ids",
                ["launch:wrong-python-tooling-tests"],
            ),
            ("aggregate-missing", "aggregate_step_observation_id", None),
            ("matrix-enabled", "matrix_applicable", True),
        )
        for label, field, value in root_mutations:
            with self.subTest(python_tooling_root=label):
                inventory = copy.deepcopy(source_current)
                root_row = next(
                    row
                    for row in inventory["test_roots"]
                    if row["id"] == CHECKER.PYTHON_TOOLING_ROOT_ID
                )
                root_row[field] = value
                self._write(inventory)
                self.assertIn("test_roots", self._errors())

        inventory = copy.deepcopy(source_current)
        module = next(
            row
            for row in inventory["python_test_modules"]
            if CHECKER.PYTHON_TOOLING_ROOT_ID in row["root_ids"]
        )
        module["launch_observation_ids"] = ["launch:wrong-python-tooling-tests"]
        self._write(inventory)
        self.assertIn("python_test_modules", self._errors())

        self._write(copy.deepcopy(source_current))
        build_path = self.root / "tools/build_inventory.json"
        original_build_bytes = build_path.read_bytes()
        try:
            build_inventory = json.loads(original_build_bytes)
            tooling_step = next(
                row
                for row in build_inventory["build_observations"]
                if row["id"] == CHECKER.PYTHON_TOOLING_STEP_ID
            )
            tooling_step["direct_dependencies"] = []
            build_path.write_text(
                json.dumps(build_inventory, indent=2) + "\n", encoding="utf-8"
            )
            self.assertIn("Python tooling focused-step closure drifted", self._errors())
        finally:
            build_path.write_bytes(original_build_bytes)

        inventory = copy.deepcopy(source_current)
        structured_skip = next(
            row
            for row in inventory["test_mode_rows"]
            if row["root_id"] == "zig-root:structured-object-tests"
            and row["disposition"] == "structured-skip"
        )
        structured_skip["disposition"] = "execute"
        self._write(inventory)
        self.assertIn("immutable matrix fields", self._errors())

        inventory = copy.deepcopy(source_current)
        inventory["test_mode_rows"].pop()
        self._write(inventory)
        self.assertIn("cover the matrix exactly", self._errors())

    def test_evidence_slot_ids_are_unique_and_row_bound(self) -> None:
        inventory = copy.deepcopy(self.inventory)
        first, second = inventory["test_mode_rows"][:2]
        second["evidence_slot_id"] = first["evidence_slot_id"]
        self._write(inventory)
        self.assertIn("duplicate evidence slot IDs", self._errors())

        inventory = copy.deepcopy(self.inventory)
        inventory["test_mode_rows"][0]["evidence_slot_id"] = (
            "evidence-slot:not-this-row"
        )
        self._write(inventory)
        self.assertIn("stable row join key", self._errors())

    def test_enumeration_class_set_foreign_key_and_state_mutations_fail(self) -> None:
        inventory = copy.deepcopy(self.inventory)
        row = next(
            row
            for row in inventory["test_mode_rows"]
            if row["expectation_state"] == CHECKER.FROZEN_STATE
            and row["root_id"].startswith("zig-root:")
        )
        row["enumeration_class_id"] = "enumeration-class:python-static"
        self._write(inventory)
        self.assertIn("enumeration class", self._errors())

        inventory = copy.deepcopy(self.inventory)
        frozen = next(
            row
            for row in inventory["test_mode_rows"]
            if row["expectation_state"] == CHECKER.FROZEN_STATE
            and row["root_id"].startswith("zig-root:")
        )
        inventory["native_observation_bindings"] = [
            binding
            for binding in inventory["native_observation_bindings"]
            if binding["row_id"] != frozen["id"]
        ]
        inventory["strict_summary"] = CHECKER._section_summary(inventory)
        self._write(inventory)
        self.assertIn("exact native observation binding", self._errors())
        invalid_whole_digest = hashlib.sha256(
            self.inventory_path.read_bytes()
        ).hexdigest()
        with (
            self._reviewed_digest_slots(inventory_next=invalid_whole_digest),
            self.assertRaisesRegex(
                CHECKER.InventoryError,
                "existing inventory is invalid:.*exact native observation binding",
            ),
        ):
            CHECKER.refresh_from_protocol(
                self.root,
                self.inventory_path,
                [(frozen["environment_id"], self.root / "unused-protocol.log")],
            )

        inventory = copy.deepcopy(self.inventory)
        pending = next(
            row
            for row in inventory["test_mode_rows"]
            if row["expectation_state"] == CHECKER.PENDING_STATE
        )
        frozen_same_root_mode = next(
            row
            for row in inventory["test_mode_rows"]
            if row["root_id"] == pending["root_id"]
            and row["optimize_mode_id"] == pending["optimize_mode_id"]
            and row["expectation_state"] == CHECKER.FROZEN_STATE
        )
        pending["expectation_state"] = CHECKER.FROZEN_STATE
        pending["expected_test_set_id"] = frozen_same_root_mode["expected_test_set_id"]
        inventory["strict_summary"] = CHECKER._section_summary(inventory)
        self._write(inventory)
        self.assertIn("exact native observation binding", self._errors())

        inventory = copy.deepcopy(self.inventory)
        row = next(
            row
            for row in inventory["test_mode_rows"]
            if row["expectation_state"] == CHECKER.FROZEN_STATE
            and row["root_id"].startswith("zig-root:")
        )
        row["expected_test_set_id"] = "set:missing"
        self._write(inventory)
        self.assertIn("foreign key", self._errors())

        inventory = copy.deepcopy(self.inventory)
        row = next(
            row
            for row in inventory["test_mode_rows"]
            if row["expectation_state"] == CHECKER.PENDING_STATE
        )
        row["expectation_state"] = CHECKER.FROZEN_STATE
        self._write(inventory)
        self.assertIn("frozen row", self._errors())

        inventory = copy.deepcopy(self.inventory)
        row = next(
            row
            for row in inventory["test_mode_rows"]
            if row["expectation_state"] == "not-applicable"
        )
        row["expectation_state"] = CHECKER.PENDING_STATE
        self._write(inventory)
        self.assertIn("requires-native state", self._errors())

    def test_expected_set_identity_name_and_order_mutations_fail(self) -> None:
        inventory = copy.deepcopy(self.inventory)
        target = next(
            row
            for row in inventory["expected_test_sets"]
            if row["root_id"] == "zig-root:modern-tests"
        )
        target["id"] = "set:wrong"
        self._write(inventory)
        self.assertIn("content-bound", self._errors())

        inventory = copy.deepcopy(self.inventory)
        target = next(
            row
            for row in inventory["expected_test_sets"]
            if row["root_id"] == "zig-root:modern-tests"
        )
        old_id = target["id"]
        target["tests"][0]["name"] += " changed"
        target["digest"] = CHECKER._fact_digest(target["tests"])
        target["id"] = CHECKER._content_set_id(target["root_id"], target["tests"])
        for row in inventory["test_mode_rows"]:
            if row["expected_test_set_id"] == old_id:
                row["expected_test_set_id"] = target["id"]
        self._write(inventory)
        self.assertIn("no reaching declaration", self._errors())

        inventory = copy.deepcopy(self.inventory)
        target = next(
            row
            for row in inventory["expected_test_sets"]
            if row["root_id"] == "zig-root:modern-tests"
        )
        target["tests"][0], target["tests"][1] = (
            target["tests"][1],
            target["tests"][0],
        )
        target["digest"] = CHECKER._fact_digest(target["tests"])
        self._write(inventory)
        self.assertIn("IDs/order", self._errors())

    def test_pending_gap_closure_and_default_cli_fail_closed(self) -> None:
        inventory = copy.deepcopy(self.inventory)
        native_gap = next(
            gap
            for gap in inventory["known_gaps"]
            if gap["kind"] == "native-test-enumeration-required"
        )
        native_gap["subject_ids"].pop()
        self._write(inventory)
        self.assertIn("known_gaps", self._errors())

        default = subprocess.run(
            [
                sys.executable,
                str(CHECKER_PATH),
                "--root",
                str(REPOSITORY_ROOT),
            ],
            check=False,
            capture_output=True,
            text=True,
            env={
                key: value
                for key, value in __import__("os").environ.items()
                if key != "GIT_PAGER"
            },
        )
        self.assertEqual(1, default.returncode)
        self.assertIn("matrix incomplete: 123 rows", default.stderr)

        structure = subprocess.run(
            [
                sys.executable,
                str(CHECKER_PATH),
                "--root",
                str(REPOSITORY_ROOT),
                "--structure-only",
            ],
            check=False,
            capture_output=True,
            text=True,
            env={
                key: value
                for key, value in __import__("os").environ.items()
                if key != "GIT_PAGER"
            },
        )
        self.assertEqual(0, structure.returncode, structure.stderr)
        self.assertIn("matrix incomplete: 123 rows", structure.stdout)

    def test_explicit_test_optimize_and_mode_mismatch_fail(self) -> None:
        workflow = self.root / ".github/workflows/ci.yml"
        text = workflow.read_text(encoding="utf-8")
        workflow.write_text(
            text.replace("-Dtest-optimize=Debug ", "", 1), encoding="utf-8"
        )
        self.assertIn("explicit test optimize", self._errors())

        workflow.write_text(
            text.replace("-Dtest-optimize=ReleaseFast ", "", 1), encoding="utf-8"
        )
        self.assertIn("explicit test optimize", self._errors())

        workflow.write_text(text, encoding="utf-8")
        inventory = copy.deepcopy(self.inventory)
        row = next(
            row
            for row in inventory["test_mode_rows"]
            if row["expected_actual_module_optimize"] == "Debug"
        )
        row["expected_actual_module_optimize"] = "ReleaseFast"
        self._write(inventory)
        self.assertIn("immutable matrix fields", self._errors())

    def test_contributor_commands_pair_test_and_artifact_optimize_modes(self) -> None:
        def commands(path: Path) -> list[list[str]]:
            text = path.read_text(encoding="utf-8")
            return [
                shlex.split(command)
                for command in re.findall(
                    r"(?:^|`)(zig build [^`\n]+)(?:$|`)", text, re.M
                )
                if "test" in shlex.split(command)
            ]

        documents = (
            REPOSITORY_ROOT / "CONTRIBUTING.md",
            REPOSITORY_ROOT / "docs/contributors/README.md",
        )
        expected_modes = (
            ("Debug", None),
            ("ReleaseSafe", "--release=safe"),
            ("ReleaseFast", "--release=fast"),
        )
        for document in documents:
            document_commands = commands(document)
            self.assertTrue(document_commands, f"no test commands found in {document}")
            for command in document_commands:
                if "-Dcpu=baseline" in command:
                    self.assertEqual(
                        1,
                        sum(token.startswith("-Dtest-optimize=") for token in command),
                        f"baseline test command lacks one explicit test mode: {command}",
                    )
            for test_mode, release_option in expected_modes:
                with self.subTest(document=document, test_mode=test_mode):
                    matches = [
                        command
                        for command in document_commands
                        if "-Dcpu=baseline" in command
                        and f"-Dtest-optimize={test_mode}" in command
                        and (
                            release_option in command
                            if release_option is not None
                            else not any(
                                token.startswith("--release") for token in command
                            )
                        )
                    ]
                    self.assertTrue(
                        matches,
                        f"{document} lacks the {test_mode} contributor test command",
                    )

        performance_commands = [
            command
            for command in commands(REPOSITORY_ROOT / "CONTRIBUTING.md")
            if "-Dtarget=aarch64-macos" in command
        ]
        self.assertEqual(1, len(performance_commands))
        self.assertIn("-Dcpu=baseline", performance_commands[0])
        self.assertIn("--release=fast", performance_commands[0])
        self.assertIn("-Dtest-optimize=ReleaseFast", performance_commands[0])

        direct_checker_blocks: list[tuple[Path, str]] = []
        for document in sorted((REPOSITORY_ROOT / "docs").rglob("*.md")):
            for block in re.findall(
                r"```(?:sh|bash|shell)\n(.*?)```",
                document.read_text(encoding="utf-8"),
                re.DOTALL,
            ):
                if (
                    "tools/check_test_inventory.py" in block
                    or "zig build test-test-inventory" in block
                ):
                    direct_checker_blocks.append((document, block))
        self.assertTrue(direct_checker_blocks)
        self.assertIn(
            REPOSITORY_ROOT / "docs/contributors/README.md",
            {document for document, _ in direct_checker_blocks},
        )
        for document, block in direct_checker_blocks:
            with self.subTest(sanitized_direct_checker_document=document):
                self.assertIn(
                    'env -i HOME="$HOME" PATH="$PATH"',
                    block,
                    f"public checker block inherits Git control variables: {document}",
                )

    def test_runner_protocol_and_isolated_object_mutations_fail(self) -> None:
        def encode(tag: str, value: str) -> str:
            payload = value.encode("utf-8")
            return f"{tag}:{len(payload)}:{payload.hex()}"

        pending_row = next(
            row
            for row in self.inventory["test_mode_rows"]
            if row["expectation_state"] == CHECKER.PENDING_STATE
        )
        reference_row = next(
            row
            for row in self.inventory["test_mode_rows"]
            if row["root_id"] == pending_row["root_id"]
            and row["optimize_mode_id"] == pending_row["optimize_mode_id"]
            and row["expected_test_set_id"] is not None
        )
        reference_set = next(
            row
            for row in self.inventory["expected_test_sets"]
            if row["id"] == reference_row["expected_test_set_id"]
        )
        protocol_lines = [
            "ZYNUM-TEST-INVENTORY-V2",
            f"mode:{pending_row['expected_actual_module_optimize']}",
            encode("root", pending_row["root_id"]),
            encode("class", pending_row["enumeration_class_id"]),
            f"count:{len(reference_set['tests'])}",
            *(
                f"test:{ordinal}:" + encode("test", test["name"]).partition(":")[2]
                for ordinal, test in enumerate(reference_set["tests"])
            ),
        ]
        protocol = self.root / "protocol.log"
        protocol.write_text("\n".join(protocol_lines) + "\n", encoding="utf-8")

        original_inventory_bytes = self.inventory_path.read_bytes()
        fixture_generation = self._assert_reviewed_fixture_generation(
            self.inventory_path, self.root / "tools/test_inventory_runner.zig"
        )
        with self._reviewed_digest_slots(
            inventory_current=fixture_generation, inventory_next=None
        ):
            with self.assertRaisesRegex(
                CHECKER.InventoryError, "reviewed whole-file test inventory mismatch"
            ) as whole_file_rejection:
                CHECKER.refresh_from_protocol(
                    self.root,
                    self.inventory_path,
                    [(pending_row["environment_id"], protocol)],
                )
        whole_match = re.search(
            r"observed sha256=([0-9a-f]{64})", str(whole_file_rejection.exception)
        )
        self.assertIsNotNone(whole_match)
        candidate_digest = whole_match.group(1)
        self.assertEqual(original_inventory_bytes, self.inventory_path.read_bytes())

        with self._reviewed_digest_slots(
            inventory_current=fixture_generation, inventory_next=candidate_digest
        ):
            with self.assertRaisesRegex(
                CHECKER.InventoryError, "reviewed native projection mismatch"
            ) as native_rejection:
                CHECKER.refresh_from_protocol(
                    self.root,
                    self.inventory_path,
                    [(pending_row["environment_id"], protocol)],
                )
        native_match = re.search(
            r"observed sha256=([0-9a-f]{64})", str(native_rejection.exception)
        )
        self.assertIsNotNone(native_match)
        native_candidate_digest = native_match.group(1)
        self.assertEqual(original_inventory_bytes, self.inventory_path.read_bytes())

        stdout = io.StringIO()
        stderr = io.StringIO()
        with (
            self._reviewed_digest_slots(
                inventory_current=fixture_generation,
                inventory_next=candidate_digest,
                native_next=native_candidate_digest,
            ),
            mock.patch.dict(os.environ),
            contextlib.redirect_stdout(stdout),
            contextlib.redirect_stderr(stderr),
        ):
            os.environ.pop("GIT_PAGER", None)
            os.environ.pop("PAGER", None)
            result = CHECKER.main(
                self._refresh_arguments(pending_row["environment_id"], protocol)
            )
        self.assertEqual(0, result, stderr.getvalue())
        refreshed_bytes = self.inventory_path.read_bytes()
        self.assertEqual(candidate_digest, hashlib.sha256(refreshed_bytes).hexdigest())
        self.assertLessEqual(len(refreshed_bytes), CHECKER.MAX_INVENTORY_BYTES)
        refreshed = json.loads(refreshed_bytes)
        refreshed_row = next(
            row for row in refreshed["test_mode_rows"] if row["id"] == pending_row["id"]
        )
        self.assertEqual(CHECKER.FROZEN_STATE, refreshed_row["expectation_state"])
        refreshed_set = next(
            row
            for row in refreshed["expected_test_sets"]
            if row["id"] == refreshed_row["expected_test_set_id"]
        )
        self.assertEqual(
            [row["name"] for row in reference_set["tests"]],
            [row["name"] for row in refreshed_set["tests"]],
        )
        refreshed_binding = next(
            row
            for row in refreshed["native_observation_bindings"]
            if row["row_id"] == pending_row["id"]
        )
        self.assertEqual(
            CHECKER._native_observation_binding(
                refreshed_row, refreshed_row["expected_test_set_id"]
            ),
            refreshed_binding,
        )
        self.inventory_path.write_bytes(original_inventory_bytes)

        second_protocol = self.root / "protocol-second.log"
        for label, second_lines, ordered_paths in (
            ("same", protocol_lines, (protocol, second_protocol)),
            (
                "different",
                [
                    *protocol_lines[: -len(reference_set["tests"])],
                    *(
                        f"test:{ordinal}:"
                        + encode(
                            "test",
                            test["name"] + (" changed" if ordinal == 0 else ""),
                        ).partition(":")[2]
                        for ordinal, test in enumerate(reference_set["tests"])
                    ),
                ],
                (protocol, second_protocol),
            ),
            ("different-reversed", protocol_lines, (second_protocol, protocol)),
        ):
            with self.subTest(global_duplicate=label):
                protocol.write_text("\n".join(protocol_lines) + "\n", encoding="utf-8")
                second_protocol.write_text(
                    "\n".join(second_lines) + "\n", encoding="utf-8"
                )
                with self.assertRaisesRegex(
                    CHECKER.InventoryError,
                    "duplicate compiler enumeration protocol observation",
                ):
                    CHECKER.refresh_from_protocol(
                        self.root,
                        self.inventory_path,
                        [
                            (pending_row["environment_id"], path)
                            for path in ordered_paths
                        ],
                    )

        bounded_protocol_cases = (
            (
                "test-count",
                [
                    *protocol_lines[:4],
                    f"count:{CHECKER.MAX_PROTOCOL_TESTS_PER_BLOCK + 1}",
                ],
                "test count exceeds limit",
            ),
            (
                "test-count-digits",
                [*protocol_lines[:4], "count:" + "9" * 100],
                "test count exceeds limit",
            ),
            (
                "value-bytes",
                [
                    *protocol_lines[:4],
                    "count:1",
                    "test:0:"
                    + encode(
                        "test", "x" * (CHECKER.MAX_PROTOCOL_VALUE_BYTES + 1)
                    ).partition(":")[2],
                ],
                "payload exceeds byte limit",
            ),
            (
                "line-bytes",
                ["x" * (CHECKER.MAX_PROTOCOL_LINE_BYTES + 1)],
                "line exceeds limit",
            ),
        )
        for label, case_lines, expected in bounded_protocol_cases:
            with self.subTest(protocol_bound=label):
                protocol.write_text("\n".join(case_lines) + "\n", encoding="utf-8")
                with self.assertRaisesRegex(CHECKER.InventoryError, expected):
                    CHECKER._parse_protocol_log(protocol)

        too_many_blocks: list[str] = []
        for ordinal in range(CHECKER.MAX_PROTOCOL_BLOCKS + 1):
            root_id = f"zig-root:bounded-{ordinal}"
            too_many_blocks.extend(
                (
                    "ZYNUM-TEST-INVENTORY-V2",
                    f"mode:{CHECKER.MODES[ordinal % len(CHECKER.MODES)]}",
                    encode("root", root_id),
                    encode("class", pending_row["enumeration_class_id"]),
                    "count:1",
                    "test:0:" + encode("test", f"bounded-{ordinal}").partition(":")[2],
                )
            )
        protocol.write_text("\n".join(too_many_blocks) + "\n", encoding="utf-8")
        with self.assertRaisesRegex(
            CHECKER.InventoryError, "block count exceeds limit"
        ):
            CHECKER._parse_protocol_log(protocol)

        protocol.write_bytes(b"x" * (CHECKER.MAX_PROTOCOL_BYTES + 1))
        with self.assertRaisesRegex(CHECKER.InventoryError, "exceeds .* bytes"):
            CHECKER._parse_protocol_log(protocol)

        base_protocol = ("\n".join(protocol_lines) + "\n").encode("utf-8")
        per_log_bytes = CHECKER.MAX_PROTOCOL_BYTES // 2 + 1024
        noise = b"noise\n" * (
            (per_log_bytes - len(base_protocol) + len(b"noise\n") - 1)
            // len(b"noise\n")
        )
        protocol.write_bytes(noise + base_protocol)
        second_protocol.write_bytes(noise + base_protocol)
        with self.assertRaisesRegex(CHECKER.InventoryError, "cumulative byte limit"):
            CHECKER.refresh_from_protocol(
                self.root,
                self.inventory_path,
                [
                    (pending_row["environment_id"], protocol),
                    (pending_row["environment_id"], second_protocol),
                ],
            )

        protocol.unlink()
        protocol.symlink_to(second_protocol)
        with self.assertRaisesRegex(
            CHECKER.InventoryError, "cannot read compiler enumeration protocol"
        ):
            CHECKER._parse_protocol_log(protocol)
        protocol.unlink()

        fifo = self.root / "protocol.fifo"
        os.mkfifo(fifo)
        previous_handler = signal.getsignal(signal.SIGALRM)

        def timeout_handler(_signum: int, _frame: Any) -> None:
            raise TimeoutError("FIFO admission blocked")

        signal.signal(signal.SIGALRM, timeout_handler)
        signal.setitimer(signal.ITIMER_REAL, 1.0)
        try:
            with self.assertRaisesRegex(CHECKER.InventoryError, "not a regular file"):
                CHECKER._parse_protocol_log(fifo)
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0.0)
            signal.signal(signal.SIGALRM, previous_handler)
            fifo.unlink()

        with self.assertRaisesRegex(CHECKER.InventoryError, "not a regular file"):
            CHECKER._parse_protocol_log(Path("/dev/null"))

        for label, mutated_lines, expected in (
            (
                "v1",
                ["ZYNUM-TEST-INVENTORY-V1", *protocol_lines[1:]],
                "unsupported compiler enumeration protocol V1",
            ),
            (
                "missing-class",
                [*protocol_lines[:3], *protocol_lines[4:]],
                "missing protocol class",
            ),
        ):
            with self.subTest(protocol=label):
                protocol.write_text("\n".join(mutated_lines) + "\n", encoding="utf-8")
                with self.assertRaisesRegex(CHECKER.InventoryError, expected):
                    CHECKER._parse_protocol_log(protocol)

        protocol_lines[3] = encode("class", "enumeration-class:x86-64-linux-gnu-elf")
        protocol.write_text("\n".join(protocol_lines) + "\n", encoding="utf-8")
        with self.assertRaisesRegex(
            CHECKER.InventoryError, "protocol enumeration class mismatch"
        ):
            CHECKER.refresh_from_protocol(
                self.root,
                self.inventory_path,
                [(pending_row["environment_id"], protocol)],
            )

        vector_source = self.root / "digest_vectors.zig"
        vector_binary = self.root / "test-inventory-digest-vectors"
        vector_source.write_text(
            'test "Ω \\"quote\\" \\\\ backslash\\ncontrol\\x01" {}\n',
            encoding="utf-8",
        )
        vector_name = 'digest_vectors.test.Ω "quote" \\ backslash\ncontrol\x01'
        vector_root_id = "zig-root:header-smoke-tests"
        vector_inventory = copy.deepcopy(self.inventory)
        host_architecture = {
            "aarch64": "aarch64",
            "arm64": "aarch64",
            "amd64": "x86_64",
            "x86_64": "x86_64",
        }.get(os.uname().machine.lower())
        host_os = {"darwin": "macos", "linux": "linux"}.get(sys.platform)
        self.assertIsNotNone(
            host_architecture,
            f"unsupported native test architecture: {os.uname().machine}",
        )
        self.assertIsNotNone(host_os, f"unsupported native test OS: {sys.platform}")
        vector_profiles = [
            profile
            for profile in vector_inventory["environment_profiles"]
            if profile["architecture"] == host_architecture
            and profile["os"] == host_os
            and profile["cpu"] == "baseline"
            and profile["host_tool_smoke"] is True
        ]
        self.assertEqual(1, len(vector_profiles), vector_profiles)
        vector_profile = vector_profiles[0]
        vector_environment_id = vector_profile["id"]
        vector_rows = [
            row
            for row in vector_inventory["test_mode_rows"]
            if row["environment_id"] == vector_environment_id
            and row["root_id"] == vector_root_id
            and row["optimize_mode_id"] == "mode:Debug"
        ]
        self.assertEqual(1, len(vector_rows), vector_rows)
        vector_row = vector_rows[0]
        vector_row_id = vector_row["id"]
        vector_class_id = vector_row["enumeration_class_id"]
        vector_tests = CHECKER._expected_test_rows(vector_root_id, [vector_name])
        self.assertEqual(
            "92c72baecf7cfa3ef98ee5ee6afe86b7de8ac197f08766248c408028625be0a2",
            CHECKER._fact_digest(vector_tests),
        )
        vector_set_id = CHECKER._content_set_id(vector_root_id, vector_tests)
        vector_set = {
            "id": vector_set_id,
            "root_id": vector_root_id,
            "tests": vector_tests,
            "count": len(vector_tests),
            "digest": CHECKER._fact_digest(vector_tests),
            "enumeration_source": CHECKER.ZIG_ENUMERATION_SOURCE,
        }
        vector_inventory["expected_test_sets"].append(vector_set)
        vector_inventory["expected_test_sets"].sort(key=lambda row: row["id"])
        vector_row["expectation_state"] = CHECKER.FROZEN_STATE
        vector_row["expected_test_set_id"] = vector_set_id
        vector_inventory["native_observation_bindings"] = [
            binding
            for binding in vector_inventory["native_observation_bindings"]
            if binding["row_id"] != vector_row_id
        ]
        vector_inventory["native_observation_bindings"].append(
            CHECKER._native_observation_binding(vector_row, vector_set_id)
        )
        vector_inventory["native_observation_bindings"].sort(key=lambda row: row["id"])
        CHECKER._refresh_native_gaps(vector_inventory)
        self.assertFalse(
            any(
                vector_row_id in gap["subject_ids"]
                for gap in vector_inventory["known_gaps"]
                if gap["kind"] == "native-test-enumeration-required"
            )
        )
        vector_inventory["strict_summary"] = CHECKER._section_summary(vector_inventory)
        self._write(vector_inventory)
        vector_bytes = self.inventory_path.read_bytes()
        exact_limit_bytes = vector_bytes + b" " * (
            CHECKER.MAX_INVENTORY_BYTES - len(vector_bytes)
        )
        vector_digest = hashlib.sha256(vector_bytes).hexdigest()
        exact_limit_digest = hashlib.sha256(exact_limit_bytes).hexdigest()
        vector_runner = self.root / "tools/test_inventory_vector_runner.zig"
        shutil.copy2(REPOSITORY_ROOT / "tools/test_inventory_runner.zig", vector_runner)
        vector_runner_text = vector_runner.read_text(encoding="utf-8")
        vector_runner_text, vector_current_count = re.subn(
            r'^const CURRENT_TEST_INVENTORY_SHA256: \[\]const u8 = "[0-9a-f]{64}";$',
            f'const CURRENT_TEST_INVENTORY_SHA256: []const u8 = "{vector_digest}";',
            vector_runner_text,
            count=1,
            flags=re.MULTILINE,
        )
        vector_runner_text, vector_next_count = re.subn(
            r"^const NEXT_TEST_INVENTORY_SHA256: \?\[\]const u8 = "
            r'(?:null|"[0-9a-f]{64}");$',
            f'const NEXT_TEST_INVENTORY_SHA256: ?[]const u8 = "{exact_limit_digest}";',
            vector_runner_text,
            count=1,
            flags=re.MULTILINE,
        )
        self.assertEqual((1, 1), (vector_current_count, vector_next_count))
        mutation_marker = (
            "    if (read_count != frozen_size) return "
            "error.InventoryTruncatedDuringRead;\n"
        )
        mutation_injection = r"""
    if (std.mem.indexOf(u8, path, "snapshot-") != null) {
        if (std.mem.indexOf(u8, path, "snapshot-rebind-") != null) {
            const release_path = try std.fmt.allocPrint(allocator, "{s}.release", .{path});
            defer allocator.free(release_path);
            std.debug.print("MUTATION-READY\n", .{});
            while (true) {
                const release_file = Io.Dir.cwd().openFile(runner_io, release_path, .{
                    .follow_symlinks = false,
                }) catch |err| switch (err) {
                    error.FileNotFound => continue,
                    else => return err,
                };
                release_file.close(runner_io);
                break;
            }
        } else {
            const mutation_file = try Io.Dir.cwd().openFile(runner_io, path, .{
                .mode = .read_write,
                .follow_symlinks = false,
            });
            defer mutation_file.close(runner_io);
            if (std.mem.endsWith(u8, path, "snapshot-truncate.json")) {
                try mutation_file.setLength(runner_io, 0);
            } else if (std.mem.endsWith(u8, path, "snapshot-growth.json")) {
                try mutation_file.setLength(runner_io, before.size + 1);
            } else if (std.mem.endsWith(u8, path, "snapshot-in-place.json")) {
                try mutation_file.writePositionalAll(runner_io, "X", 0);
            } else if (std.mem.endsWith(u8, path, "snapshot-aba.json")) {
                try mutation_file.setLength(runner_io, 0);
                try mutation_file.writePositionalAll(runner_io, bytes, 0);
            }
        }
    }
"""
        self.assertEqual(1, vector_runner_text.count(mutation_marker))
        vector_runner.write_text(
            vector_runner_text.replace(
                mutation_marker,
                mutation_marker + mutation_injection,
                1,
            ),
            encoding="utf-8",
        )

        compile_vector = subprocess.run(
            [
                "zig",
                "test",
                str(vector_source),
                "--name",
                "digest_vectors",
                "-O",
                "Debug",
                "-target",
                vector_profile["target"],
                "-mcpu",
                "baseline",
                "--test-runner",
                str(vector_runner),
                "--test-no-exec",
                f"-femit-bin={vector_binary}",
            ],
            check=False,
            capture_output=True,
            text=True,
            cwd=self.root,
            timeout=120.0,
        )
        self.assertEqual(0, compile_vector.returncode, compile_vector.stderr)

        def run_vector_inventory(
            inventory_path: Path = self.inventory_path,
        ) -> subprocess.CompletedProcess[str]:
            return subprocess.run(
                [
                    "./test-inventory-digest-vectors",
                    str(inventory_path),
                    "--inventory-environment",
                    vector_environment_id,
                    "--inventory-root",
                    vector_root_id,
                    "--inventory-mode",
                    "Debug",
                    "--inventory-class",
                    vector_class_id,
                ],
                check=False,
                capture_output=True,
                text=True,
                cwd=self.root,
                timeout=5.0,
            )

        try:
            self._write(vector_inventory)
            vector_result = run_vector_inventory()
            self.assertEqual(0, vector_result.returncode, vector_result.stderr)
            self.assertIn("ZYNUM-TEST-INVENTORY-V2", vector_result.stdout)

            for mutation in ("truncate", "growth", "in-place", "aba"):
                with self.subTest(stable_snapshot_mutation=mutation):
                    mutation_inventory = self.inventory_path.with_name(
                        f"snapshot-{mutation}.json"
                    )
                    mutation_inventory.write_bytes(vector_bytes)
                    try:
                        mutation_result = run_vector_inventory(mutation_inventory)
                        self.assertNotEqual(0, mutation_result.returncode)
                        self.assertNotIn(
                            "ZYNUM-TEST-INVENTORY-V2", mutation_result.stdout
                        )
                    finally:
                        mutation_inventory.unlink(missing_ok=True)

            self.assertEqual(CHECKER.MAX_INVENTORY_BYTES, len(exact_limit_bytes))
            self.inventory_path.write_bytes(exact_limit_bytes)
            vector_result = run_vector_inventory()
            self.assertEqual(0, vector_result.returncode, vector_result.stderr)
            self.assertIn("ZYNUM-TEST-INVENTORY-V2", vector_result.stdout)

            self.inventory_path.write_bytes(exact_limit_bytes + b" ")
            vector_result = run_vector_inventory()
            self.assertNotEqual(0, vector_result.returncode)
            self.assertNotIn("ZYNUM-TEST-INVENTORY-V2", vector_result.stdout)

            for replacement_kind, replacement_bytes in (
                ("same-size", vector_bytes),
                ("changed-size", vector_bytes + b" "),
            ):
                with self.subTest(pathname_rebind=replacement_kind):
                    race_inventory = self.inventory_path.with_name(
                        f"snapshot-rebind-{replacement_kind}.json"
                    )
                    race_replacement = race_inventory.with_suffix(".replacement")
                    race_release = Path(f"{race_inventory}.release")
                    race_inventory.write_bytes(vector_bytes)
                    race_replacement.write_bytes(replacement_bytes)
                    race_process = subprocess.Popen(
                        [
                            "./test-inventory-digest-vectors",
                            str(race_inventory),
                            "--inventory-environment",
                            vector_environment_id,
                            "--inventory-root",
                            vector_root_id,
                            "--inventory-mode",
                            "Debug",
                            "--inventory-class",
                            vector_class_id,
                        ],
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        text=True,
                        cwd=self.root,
                    )
                    try:
                        self.assertIsNotNone(race_process.stderr)
                        self.assertEqual(
                            "MUTATION-READY\n", race_process.stderr.readline()
                        )
                        os.replace(race_replacement, race_inventory)
                        race_release.write_bytes(b"release")
                        race_stdout, _race_stderr = race_process.communicate(
                            timeout=5.0
                        )
                        self.assertNotEqual(0, race_process.returncode)
                        self.assertNotIn("ZYNUM-TEST-INVENTORY-V2", race_stdout)
                    finally:
                        race_inventory.unlink(missing_ok=True)
                        race_replacement.unlink(missing_ok=True)
                        race_release.unlink(missing_ok=True)

            self._write(vector_inventory)
            symlink_inventory = self.inventory_path.with_name(
                "test_inventory-symlink.json"
            )
            symlink_inventory.symlink_to(self.inventory_path.name)
            try:
                vector_result = run_vector_inventory(symlink_inventory)
                self.assertNotEqual(0, vector_result.returncode)
                self.assertNotIn("ZYNUM-TEST-INVENTORY-V2", vector_result.stdout)
            finally:
                symlink_inventory.unlink(missing_ok=True)

            with self.subTest(real_runner_path="fifo-without-writer"):
                fifo_inventory = self.inventory_path.with_name("test_inventory.fifo")
                try:
                    os.mkfifo(fifo_inventory)
                except (AttributeError, NotImplementedError):
                    self.skipTest("FIFO creation is unsupported on this platform")
                except OSError as error:
                    unsupported = {
                        getattr(errno, name)
                        for name in ("ENOSYS", "ENOTSUP", "EOPNOTSUPP")
                        if hasattr(errno, name)
                    }
                    if error.errno in unsupported:
                        self.skipTest("FIFO creation is unsupported on this filesystem")
                    raise
                try:
                    vector_result = run_vector_inventory(fifo_inventory)
                    self.assertNotEqual(0, vector_result.returncode)
                    self.assertNotIn("ZYNUM-TEST-INVENTORY-V2", vector_result.stdout)
                finally:
                    fifo_inventory.unlink(missing_ok=True)

            for mutation in (
                "row-and-evidence-slot",
                "test-and-linked-identities",
                "set-and-linked-binding",
                "binding-identity-and-digest",
            ):
                with self.subTest(real_runner_mutation=mutation):
                    candidate = copy.deepcopy(vector_inventory)
                    row = next(
                        item
                        for item in candidate["test_mode_rows"]
                        if item["id"] == vector_row_id
                    )
                    expected_set = next(
                        item
                        for item in candidate["expected_test_sets"]
                        if item["id"] == vector_set_id
                    )
                    candidate["native_observation_bindings"] = [
                        item
                        for item in candidate["native_observation_bindings"]
                        if item["row_id"] != vector_row_id
                    ]
                    if mutation == "row-and-evidence-slot":
                        row["id"] = "row:forged"
                        row["evidence_slot_id"] = "evidence-slot:forged"
                    elif mutation == "test-and-linked-identities":
                        expected_set["tests"][0]["id"] = "test:forged:0"
                        expected_set["digest"] = CHECKER._fact_digest(
                            expected_set["tests"]
                        )
                        expected_set["id"] = CHECKER._content_set_id(
                            vector_root_id, expected_set["tests"]
                        )
                        row["expected_test_set_id"] = expected_set["id"]
                    elif mutation == "set-and-linked-binding":
                        expected_set["digest"] = "0" * 64
                        expected_set["id"] = f"set:{vector_root_id}:{'0' * 64}"
                        row["expected_test_set_id"] = expected_set["id"]
                    candidate_binding = CHECKER._native_observation_binding(
                        row, row["expected_test_set_id"]
                    )
                    if mutation == "binding-identity-and-digest":
                        candidate_binding["digest"] = "0" * 64
                        candidate_binding["id"] = "native-observation:" + "0" * 64
                    candidate["native_observation_bindings"].append(candidate_binding)
                    self.assertEqual(1, expected_set["count"])
                    self.assertEqual([vector_name], [expected_set["tests"][0]["name"]])
                    self._write(candidate)
                    vector_result = run_vector_inventory()
                    self.assertNotEqual(0, vector_result.returncode)
                    self.assertNotIn("ZYNUM-TEST-INVENTORY-V2", vector_result.stdout)
        finally:
            self._write(copy.deepcopy(self.inventory))
            vector_binary.unlink(missing_ok=True)
            vector_source.unlink(missing_ok=True)
            vector_runner.unlink(missing_ok=True)

        runner = self.root / "tools/test_inventory_runner.zig"
        runner_text = runner.read_text(encoding="utf-8")
        validation_line = (
            "    const validation = try validateInventory(allocator, arguments);\n"
        )
        runner.write_text(
            runner_text.replace(validation_line, "", 1).replace(
                "    try emitProtocol(writer, arguments);\n",
                "    try emitProtocol(writer, arguments);\n" + validation_line,
                1,
            ),
            encoding="utf-8",
        )
        self.assertIn("validation must precede", self._errors())

        shutil.copy2(REPOSITORY_ROOT / "tools/test_inventory_runner.zig", runner)
        runner.write_text(
            runner.read_text(encoding="utf-8").replace(
                "@tagName(builtin.mode)", '"Debug"'
            ),
            encoding="utf-8",
        )
        self.assertIn("protocol identity", self._errors())

        shutil.copy2(REPOSITORY_ROOT / "tools/test_inventory_runner.zig", runner)
        runner.write_text(
            re.sub(
                r"^\s*\.follow_symlinks = false,\n",
                "",
                runner.read_text(encoding="utf-8"),
                flags=re.MULTILINE,
            ),
            encoding="utf-8",
        )
        self.assertIn("reparse-point", self._errors())

        shutil.copy2(REPOSITORY_ROOT / "tools/test_inventory_runner.zig", runner)
        runner.write_text(
            runner.read_text(encoding="utf-8").replace(
                "inventoryMetadataStable(before, admitted_path)",
                "inventoryMetadataStable(before, before)",
                1,
            ),
            encoding="utf-8",
        )
        self.assertIn("pathname rebinding", self._errors())

        shutil.copy2(REPOSITORY_ROOT / "tools/test_inventory_runner.zig", runner)
        duplicated_runner, duplicate_slot_count = re.subn(
            r"^(const NEXT_TEST_INVENTORY_SHA256: \?\[\]const u8 = "
            r'(?:null|"[0-9a-f]{64}");)$',
            r"\1\nconst CURRENT_TEST_INVENTORY_SHA256: []const u8 = "
            f'"{CHECKER.CURRENT_TEST_INVENTORY_SHA256}";',
            runner.read_text(encoding="utf-8"),
            count=1,
            flags=re.MULTILINE,
        )
        self.assertEqual(1, duplicate_slot_count)
        runner.write_text(duplicated_runner, encoding="utf-8")
        self.assertIn("unique strict format", self._errors())

        shutil.copy2(REPOSITORY_ROOT / "tools/test_inventory_runner.zig", runner)
        runner.write_text(
            runner.read_text(encoding="utf-8").replace(
                "    try validateInventoryDigest(bytes);\n", "", 1
            ),
            encoding="utf-8",
        )
        self.assertIn("must precede JSON parsing", self._errors())

        shutil.copy2(REPOSITORY_ROOT / "tools/test_inventory_runner.zig", runner)
        build_inventory_path = self.root / "tools/build_inventory.json"
        build_inventory = json.loads(build_inventory_path.read_text(encoding="utf-8"))
        isolated = next(
            row
            for row in build_inventory["build_observations"]
            if row["id"] == "compile:build.zig:build:stride2_isolated_test_library"
        )
        isolated["optimize_source"] = "optimize"
        build_inventory_path.write_text(
            json.dumps(build_inventory, indent=2) + "\n", encoding="utf-8"
        )
        self.assertIn("production optimize", self._errors())

    def test_reviewed_native_projection_rejects_coherent_forgery(self) -> None:
        baseline = self._source_current_fixture_inventory()
        projection = CHECKER._native_projection(baseline)
        self.assertEqual(CHECKER.NATIVE_PROJECTION_SCHEMA_ID, projection["schema_id"])
        self.assertEqual(
            CHECKER.NATIVE_PROJECTION_SCHEMA_VERSION, projection["schema_version"]
        )
        self.assertEqual(246, len(projection["native_execution_rows"]))
        self.assertEqual(123, len(projection["native_observation_bindings"]))
        current_digest = CHECKER._native_projection_digest(baseline)
        self.assertEqual(CHECKER.CURRENT_NATIVE_PROJECTION_SHA256, current_digest)
        self.assertIsNone(CHECKER.NEXT_NATIVE_PROJECTION_SHA256)
        self.assertEqual(
            [],
            CHECKER.validate(self.root, self.inventory_path, structure_only=True),
        )

        all_frozen = self._freeze_pending_native_rows(baseline)
        self.assertEqual(0, CHECKER._matrix_incomplete_count(all_frozen))
        self.assertEqual(246, len(all_frozen["native_observation_bindings"]))
        all_frozen_digest = CHECKER._native_projection_digest(all_frozen)
        self.assertNotEqual(current_digest, all_frozen_digest)
        self._write(all_frozen)
        all_frozen_whole_digest = hashlib.sha256(
            self.inventory_path.read_bytes()
        ).hexdigest()
        with self._reviewed_digest_slots(inventory_next=all_frozen_whole_digest):
            errors = CHECKER.validate(
                self.root, self.inventory_path, structure_only=True
            )
        self.assertEqual(1, len(errors))
        self.assertIn("reviewed native projection mismatch:", errors[0])
        self.assertIn(f"observed sha256={all_frozen_digest}", errors[0])

        renamed = self._rename_frozen_native_test_coherently(baseline)
        renamed_digest = CHECKER._native_projection_digest(renamed)
        self.assertNotEqual(current_digest, renamed_digest)
        self._write(renamed)
        renamed_whole_digest = hashlib.sha256(
            self.inventory_path.read_bytes()
        ).hexdigest()
        with self._reviewed_digest_slots(inventory_next=renamed_whole_digest):
            errors = CHECKER.validate(
                self.root, self.inventory_path, structure_only=True
            )
        self.assertEqual(1, len(errors))
        self.assertIn("reviewed native projection mismatch:", errors[0])
        self.assertIn(f"observed sha256={renamed_digest}", errors[0])

    def test_reviewed_native_projection_blocks_refresh_and_direct_publication(
        self,
    ) -> None:
        baseline = self._source_current_fixture_inventory()
        environment_id, protocol = self._native_refresh_protocol(pending=False)
        forged = self._freeze_pending_native_rows(baseline)
        self._write(forged)
        forged_bytes = self.inventory_path.read_bytes()
        forged_whole_digest = hashlib.sha256(forged_bytes).hexdigest()
        stderr = io.StringIO()
        with (
            mock.patch.dict(os.environ),
            mock.patch.object(CHECKER, "_publish_inventory_atomic") as publish,
            contextlib.redirect_stderr(stderr),
        ):
            os.environ.pop("GIT_PAGER", None)
            os.environ.pop("PAGER", None)
            result = CHECKER.main(self._refresh_arguments(environment_id, protocol))
        self.assertEqual(1, result)
        publish.assert_not_called()
        self.assertEqual(forged_bytes, self.inventory_path.read_bytes())
        self.assertIn("reviewed whole-file test inventory mismatch:", stderr.getvalue())

        stderr = io.StringIO()
        with (
            self._reviewed_digest_slots(inventory_next=forged_whole_digest),
            mock.patch.dict(os.environ),
            mock.patch.object(CHECKER, "_publish_inventory_atomic") as publish,
            contextlib.redirect_stderr(stderr),
        ):
            os.environ.pop("GIT_PAGER", None)
            os.environ.pop("PAGER", None)
            result = CHECKER.main(self._refresh_arguments(environment_id, protocol))
        self.assertEqual(1, result)
        publish.assert_not_called()
        self.assertEqual(forged_bytes, self.inventory_path.read_bytes())
        self.assertIn("reviewed native projection mismatch:", stderr.getvalue())

        self._write(copy.deepcopy(baseline))
        snapshot = CHECKER._read_regular_stable_snapshot(
            self.inventory_path, CHECKER.MAX_INVENTORY_BYTES, "test inventory"
        )
        candidate_bytes = CHECKER._canonical_inventory_bytes(forged)
        temporary_pattern = f".{self.inventory_path.name}.*.tmp"
        self.assertEqual([], list(self.inventory_path.parent.glob(temporary_pattern)))
        with mock.patch.object(CHECKER.os, "open") as open_file:
            with self.assertRaisesRegex(
                CHECKER.InventoryError,
                "reviewed whole-file test inventory mismatch:",
            ):
                CHECKER._publish_inventory_atomic(
                    self.inventory_path, candidate_bytes, snapshot
                )
        open_file.assert_not_called()
        self.assertEqual([], list(self.inventory_path.parent.glob(temporary_pattern)))

        with (
            mock.patch.object(
                CHECKER, "NEXT_TEST_INVENTORY_SHA256", forged_whole_digest
            ),
            mock.patch.object(CHECKER.os, "open") as open_file,
        ):
            with self.assertRaisesRegex(
                CHECKER.InventoryError, "reviewed native projection mismatch:"
            ):
                CHECKER._publish_inventory_atomic(
                    self.inventory_path, candidate_bytes, snapshot
                )
        open_file.assert_not_called()

        valid_bytes = self.inventory_path.read_bytes()
        valid_digest = hashlib.sha256(valid_bytes).hexdigest()
        valid_native_digest = CHECKER._native_projection_digest(baseline)
        snapshot = CHECKER._read_regular_stable_snapshot(
            self.inventory_path, CHECKER.MAX_INVENTORY_BYTES, "test inventory"
        )
        token = "f" * 24
        foreign_name = f".{self.inventory_path.name}.{token}.tmp"
        foreign_path = self.inventory_path.with_name(foreign_name)
        owned_path = self.inventory_path.with_name(f"{foreign_name}.owned")
        cleanup_arena = self.inventory_path.parent / (
            f".zynum-cleanup-v2-{os.geteuid()}"
        )
        foreign_quarantine = cleanup_arena / (
            f"{foreign_name}.{token}.cleanup-quarantine"
        )
        foreign_recovery = foreign_quarantine / "claimed"
        actual_close = CHECKER.os.close
        swapped_temporary = False

        def swap_temporary_after_close(descriptor: int) -> None:
            nonlocal swapped_temporary
            actual_close(descriptor)
            if not swapped_temporary:
                swapped_temporary = True
                foreign_path.replace(owned_path)
                foreign_path.write_bytes(b"foreign")

        try:
            with (
                self._reviewed_digest_slots(
                    inventory_current=valid_digest,
                    native_current=valid_native_digest,
                ),
                mock.patch.object(CHECKER.secrets, "token_hex", return_value=token),
                mock.patch.object(
                    CHECKER.os, "close", side_effect=swap_temporary_after_close
                ),
                self.assertRaisesRegex(
                    CHECKER.InventoryError,
                    "inventory publication temporary changed before replace",
                ) as publication_failure,
            ):
                CHECKER._publish_inventory_atomic(
                    self.inventory_path, valid_bytes, snapshot
                )
            self.assertEqual(valid_bytes, self.inventory_path.read_bytes())
            self.assertFalse(foreign_path.exists())
            self.assertEqual(b"foreign", foreign_recovery.read_bytes())
            self.assertEqual(valid_bytes, owned_path.read_bytes())
            publication_message = str(publication_failure.exception)
            self.assertIn("exact recovery paths:", publication_message)
            self.assertIn(os.fspath(foreign_recovery), publication_message)
            self.assertIn("public_candidate=absent", publication_message)
            self.assertIn("candidate paths: none", publication_message)
            self.assertNotIn(os.fspath(owned_path), publication_message)
        finally:
            foreign_path.unlink(missing_ok=True)
            owned_path.unlink(missing_ok=True)
            shutil.rmtree(foreign_quarantine, ignore_errors=True)

    def test_reviewed_native_projection_two_slot_migration(self) -> None:
        baseline = self._source_current_fixture_inventory()
        old_bytes = self.inventory_path.read_bytes()
        baseline_generation = hashlib.sha256(old_bytes).hexdigest()
        pending_rows = [
            row
            for row in baseline["test_mode_rows"]
            if row["expectation_state"] == CHECKER.PENDING_STATE
            and row["root_id"].startswith("zig-root:")
        ]
        first_candidate = self._freeze_pending_native_rows(
            baseline, row_ids={pending_rows[0]["id"]}
        )
        first_digest = CHECKER._native_projection_digest(first_candidate)
        self.assertNotEqual(CHECKER.CURRENT_NATIVE_PROJECTION_SHA256, first_digest)
        environment_id, protocol = self._native_refresh_protocol(
            pending=True, candidate_ordinal=0
        )

        stderr = io.StringIO()
        with (
            self._reviewed_digest_slots(
                inventory_current=baseline_generation, inventory_next=None
            ),
            mock.patch.dict(os.environ),
            mock.patch.object(CHECKER, "_publish_inventory_atomic") as publish,
            contextlib.redirect_stderr(stderr),
        ):
            os.environ.pop("GIT_PAGER", None)
            os.environ.pop("PAGER", None)
            result = CHECKER.main(self._refresh_arguments(environment_id, protocol))
        self.assertEqual(1, result)
        publish.assert_not_called()
        self.assertEqual(old_bytes, self.inventory_path.read_bytes())
        first_whole_match = re.search(
            r"observed sha256=([0-9a-f]{64})", stderr.getvalue()
        )
        self.assertIsNotNone(first_whole_match)
        first_whole_digest = first_whole_match.group(1)

        stderr = io.StringIO()
        with (
            self._reviewed_digest_slots(
                inventory_current=baseline_generation,
                inventory_next=first_whole_digest,
            ),
            mock.patch.dict(os.environ),
            mock.patch.object(CHECKER, "_publish_inventory_atomic") as publish,
            contextlib.redirect_stderr(stderr),
        ):
            os.environ.pop("GIT_PAGER", None)
            os.environ.pop("PAGER", None)
            result = CHECKER.main(self._refresh_arguments(environment_id, protocol))
        self.assertEqual(1, result)
        publish.assert_not_called()
        self.assertIn(f"observed sha256={first_digest}", stderr.getvalue())

        stdout = io.StringIO()
        stderr = io.StringIO()
        with (
            self._reviewed_digest_slots(
                inventory_current=baseline_generation,
                inventory_next=first_whole_digest,
                native_next=first_digest,
            ),
            mock.patch.dict(os.environ),
            contextlib.redirect_stdout(stdout),
            contextlib.redirect_stderr(stderr),
        ):
            os.environ.pop("GIT_PAGER", None)
            os.environ.pop("PAGER", None)
            result = CHECKER.main(self._refresh_arguments(environment_id, protocol))
        self.assertEqual(0, result, stderr.getvalue())
        promoted_bytes = self.inventory_path.read_bytes()
        promoted_whole_digest = hashlib.sha256(promoted_bytes).hexdigest()
        self.assertEqual(first_whole_digest, promoted_whole_digest)
        promoted = json.loads(promoted_bytes)
        self.assertEqual(first_digest, CHECKER._native_projection_digest(promoted))

        with self._reviewed_digest_slots(
            inventory_current=promoted_whole_digest,
            native_current=first_digest,
        ):
            self.assertEqual(
                [],
                CHECKER.validate(
                    self.root,
                    self.inventory_path,
                    structure_only=True,
                    require_current_only=True,
                ),
            )
            self.inventory_path.write_bytes(old_bytes)
            old_errors = CHECKER.validate(
                self.root,
                self.inventory_path,
                structure_only=True,
                require_current_only=True,
            )
            self.assertTrue(old_errors)
            self.assertTrue(all("reviewed whole-file" in error for error in old_errors))

        self.inventory_path.write_bytes(old_bytes)
        second_candidate = self._freeze_pending_native_rows(
            baseline, row_ids={pending_rows[1]["id"]}
        )
        second_digest = CHECKER._native_projection_digest(second_candidate)
        self.assertNotEqual(first_digest, second_digest)
        second_environment_id, second_protocol = self._native_refresh_protocol(
            pending=True, candidate_ordinal=1
        )
        with (
            self._reviewed_digest_slots(
                inventory_current=baseline_generation,
                inventory_next=first_whole_digest,
                native_next=first_digest,
            ),
            mock.patch.object(CHECKER, "_publish_inventory_atomic") as publish,
            mock.patch.dict(os.environ),
            contextlib.redirect_stderr(io.StringIO()) as second_stderr,
        ):
            os.environ.pop("GIT_PAGER", None)
            os.environ.pop("PAGER", None)
            result = CHECKER.main(
                self._refresh_arguments(second_environment_id, second_protocol)
            )
        self.assertEqual(1, result)
        publish.assert_not_called()
        self.assertEqual(old_bytes, self.inventory_path.read_bytes())
        self.assertIn("reviewed whole-file", second_stderr.getvalue())
        self.assertNotIn(
            f"observed sha256={first_whole_digest}", second_stderr.getvalue()
        )

        malformed_policies = (
            ("0" * 63, None),
            ("A" * 64, None),
            ("g" * 64, None),
            (baseline_generation, "0" * 63),
            (baseline_generation, "A" * 64),
            (
                baseline_generation,
                baseline_generation,
            ),
        )
        for current, next_digest in malformed_policies:
            with (
                self.subTest(current=current, next_digest=next_digest),
                mock.patch.object(CHECKER, "CURRENT_TEST_INVENTORY_SHA256", current),
                mock.patch.object(CHECKER, "NEXT_TEST_INVENTORY_SHA256", next_digest),
            ):
                errors = CHECKER.validate(
                    self.root, self.inventory_path, structure_only=True
                )
                self.assertTrue(errors)
                self.assertTrue(all("reviewed whole-file" in error for error in errors))

        self.inventory_path.write_bytes(old_bytes)
        with self._reviewed_digest_slots(
            inventory_current=baseline_generation,
            inventory_next=first_whole_digest,
        ):
            errors = CHECKER.validate(
                self.root,
                self.inventory_path,
                structure_only=True,
                require_current_only=True,
            )
        self.assertTrue(errors)
        self.assertTrue(all("current-only policy" in error for error in errors))

        self.inventory_path.write_bytes(promoted_bytes)

    def test_refresh_rebases_stale_python_source_facts(self) -> None:
        native_digest_before = CHECKER._native_projection_digest(self.inventory)
        self.assertEqual(CHECKER.CURRENT_NATIVE_PROJECTION_SHA256, native_digest_before)
        fixture_generation = self._assert_reviewed_fixture_generation(
            self.inventory_path, self.root / "tools/test_inventory_runner.zig"
        )
        path, declaration_name = self._append_python_test_declaration_drift()
        environment_id, protocol = self._native_refresh_protocol(pending=False)
        recorded_python_module = next(
            row for row in self.inventory["python_test_modules"] if row["path"] == path
        )
        discovered_python_module = next(
            row
            for row in CHECKER.discover(self.root, self.inventory_path)[
                "python_test_modules"
            ]
            if row["path"] == path
        )
        self.assertNotEqual(recorded_python_module, discovered_python_module)
        self.assertTrue(
            CHECKER.validate(self.root, self.inventory_path, structure_only=True),
            "ordinary validation unexpectedly accepted stale Python source facts",
        )
        old_native_bindings = {
            row["row_id"]: copy.deepcopy(row)
            for row in self.inventory["native_observation_bindings"]
        }
        old_rows = {
            row["id"]: copy.deepcopy(row)
            for row in self.inventory["test_mode_rows"]
            if row["id"] in old_native_bindings
        }
        old_sets = {
            row["id"]: copy.deepcopy(row)
            for row in self.inventory["expected_test_sets"]
            if row["id"] in {item["expected_test_set_id"] for item in old_rows.values()}
        }
        stdout = io.StringIO()
        stderr = io.StringIO()
        with (
            self._reviewed_digest_slots(
                inventory_current=fixture_generation, inventory_next=None
            ),
            mock.patch.dict(os.environ),
            mock.patch.object(CHECKER, "_publish_inventory_atomic") as publish,
            contextlib.redirect_stdout(stdout),
            contextlib.redirect_stderr(stderr),
        ):
            os.environ.pop("GIT_PAGER", None)
            os.environ.pop("PAGER", None)
            result = CHECKER.main(self._refresh_arguments(environment_id, protocol))
        self.assertEqual(1, result)
        publish.assert_not_called()
        candidate_match = re.search(
            r"observed sha256=([0-9a-f]{64})", stderr.getvalue()
        )
        self.assertIsNotNone(candidate_match)
        python_only_candidate_digest = candidate_match.group(1)

        stdout = io.StringIO()
        stderr = io.StringIO()
        with (
            self._reviewed_digest_slots(
                inventory_current=fixture_generation,
                inventory_next=python_only_candidate_digest,
            ),
            mock.patch.dict(os.environ),
            contextlib.redirect_stdout(stdout),
            contextlib.redirect_stderr(stderr),
        ):
            os.environ.pop("GIT_PAGER", None)
            os.environ.pop("PAGER", None)
            result = CHECKER.main(self._refresh_arguments(environment_id, protocol))
        self.assertEqual(0, result, stderr.getvalue())

        refreshed_bytes = self.inventory_path.read_bytes()
        self.assertEqual(
            python_only_candidate_digest,
            hashlib.sha256(refreshed_bytes).hexdigest(),
        )
        refreshed = json.loads(refreshed_bytes)
        self.assertEqual(
            native_digest_before, CHECKER._native_projection_digest(refreshed)
        )
        with self._reviewed_digest_slots(
            inventory_current=fixture_generation,
            inventory_next=python_only_candidate_digest,
        ):
            self.assertEqual(
                [],
                CHECKER.validate(
                    self.root,
                    self.inventory_path,
                    structure_only=True,
                ),
            )
        python_module = next(
            row for row in refreshed["python_test_modules"] if row["path"] == path
        )
        declaration = python_module["declarations"][-1]
        self.assertEqual(declaration_name, declaration["name"])
        self.assertEqual(len(python_module["declarations"]) - 1, declaration["ordinal"])
        self.assertEqual(
            f"python-decl:{path}:{declaration['ordinal']}", declaration["id"]
        )

        python_root_id = "python-root:test-inventory-direct"
        python_sets = [
            row
            for row in refreshed["expected_test_sets"]
            if row["root_id"] == python_root_id
        ]
        self.assertEqual(1, len(python_sets))
        python_set = python_sets[0]
        self.assertEqual(
            CHECKER._fact_digest(python_set["tests"]), python_set["digest"]
        )
        self.assertEqual(
            CHECKER._content_set_id(python_root_id, python_set["tests"]),
            python_set["id"],
        )
        self.assertIn(
            f"{path}::{declaration_name}",
            [row["name"] for row in python_set["tests"]],
        )
        python_rows = [
            row
            for row in refreshed["test_mode_rows"]
            if row["root_id"] == python_root_id and row["disposition"] == "execute"
        ]
        self.assertTrue(python_rows)
        self.assertTrue(
            all(
                row["expected_test_set_id"] == python_set["id"]
                and row["expectation_state"] == CHECKER.FROZEN_STATE
                for row in python_rows
            )
        )
        self.assertEqual(
            CHECKER._section_summary(refreshed), refreshed["strict_summary"]
        )

        refreshed_bindings = {
            row["row_id"]: row for row in refreshed["native_observation_bindings"]
        }
        refreshed_rows = {row["id"]: row for row in refreshed["test_mode_rows"]}
        refreshed_sets = {row["id"]: row for row in refreshed["expected_test_sets"]}
        for row_id, binding in old_native_bindings.items():
            with self.subTest(preserved_native_row=row_id):
                self.assertEqual(binding, refreshed_bindings[row_id])
                self.assertEqual(old_rows[row_id], refreshed_rows[row_id])
        for set_id, expected_set in old_sets.items():
            with self.subTest(preserved_native_set=set_id):
                self.assertEqual(expected_set, refreshed_sets[set_id])

    def test_refresh_rejects_forged_existing_native_binding_without_publication(
        self,
    ) -> None:
        self._append_python_test_declaration_drift()
        environment_id, protocol = self._native_refresh_protocol(pending=False)
        baseline = copy.deepcopy(self.inventory)
        binding = baseline["native_observation_bindings"][0]
        native_row = next(
            row for row in baseline["test_mode_rows"] if row["id"] == binding["row_id"]
        )
        native_set = next(
            row
            for row in baseline["expected_test_sets"]
            if row["id"] == binding["expected_test_set_id"]
        )
        mutations = (
            (
                "binding-id-and-digest",
                lambda inventory: inventory["native_observation_bindings"][0].update(
                    {"id": "native-observation:" + "0" * 64, "digest": "0" * 64}
                ),
            ),
            (
                "native-set-digest",
                lambda inventory: next(
                    row
                    for row in inventory["expected_test_sets"]
                    if row["id"] == native_set["id"]
                ).update({"digest": "0" * 64}),
            ),
            (
                "native-row-set-foreign-key",
                lambda inventory: next(
                    row
                    for row in inventory["test_mode_rows"]
                    if row["id"] == native_row["id"]
                ).update({"expected_test_set_id": "set:forged"}),
            ),
        )
        for label, mutate in mutations:
            with self.subTest(forged_native_fact=label):
                inventory = copy.deepcopy(baseline)
                mutate(inventory)
                inventory["strict_summary"] = CHECKER._section_summary(inventory)
                self._write(inventory)
                original_bytes = self.inventory_path.read_bytes()
                stdout = io.StringIO()
                stderr = io.StringIO()
                with (
                    mock.patch.dict(os.environ),
                    mock.patch.object(CHECKER, "_publish_inventory_atomic") as publish,
                    contextlib.redirect_stdout(stdout),
                    contextlib.redirect_stderr(stderr),
                ):
                    os.environ.pop("GIT_PAGER", None)
                    os.environ.pop("PAGER", None)
                    result = CHECKER.main(
                        self._refresh_arguments(environment_id, protocol)
                    )
                self.assertEqual(1, result)
                publish.assert_not_called()
                self.assertEqual(original_bytes, self.inventory_path.read_bytes())
                self.assertNotIn("Traceback", stderr.getvalue())

    def test_factory_missing_duplicate_and_wrong_mapping_fail(self) -> None:
        build_path = self.root / "tools/build_inventory.json"

        build_inventory = json.loads(build_path.read_text(encoding="utf-8"))
        factory = next(
            row
            for row in build_inventory["build_observations"]
            if row.get("artifact_role") == CHECKER.FACTORY_ROLE
        )
        factory.pop("artifact_role")
        build_path.write_text(
            json.dumps(build_inventory, indent=2) + "\n", encoding="utf-8"
        )
        self.assertIn("exactly one", self._errors())

        build_inventory = json.loads(
            (REPOSITORY_ROOT / "tools/build_inventory.json").read_text(encoding="utf-8")
        )
        factory = next(
            row
            for row in build_inventory["build_observations"]
            if row.get("artifact_role") == CHECKER.FACTORY_ROLE
        )
        duplicate = copy.deepcopy(factory)
        duplicate["id"] = "compile:fixture:duplicate-inventory-factory"
        build_inventory["build_observations"].append(duplicate)
        build_path.write_text(
            json.dumps(build_inventory, indent=2) + "\n", encoding="utf-8"
        )
        self.assertIn("exactly one", self._errors())

        build_inventory = json.loads(
            (REPOSITORY_ROOT / "tools/build_inventory.json").read_text(encoding="utf-8")
        )
        factory = next(
            row
            for row in build_inventory["build_observations"]
            if row.get("artifact_role") == CHECKER.FACTORY_ROLE
        )
        factory["expansion_cases"][0]["root_id"] = "zig-root:wrong"
        build_path.write_text(
            json.dumps(build_inventory, indent=2) + "\n", encoding="utf-8"
        )
        self.assertIn("map Zig roots exactly", self._errors())

    def test_factory_optional_projection_and_fail_closure_mutations_fail(self) -> None:
        build_path = self.root / "tools/build_inventory.json"
        baseline = json.loads(
            (REPOSITORY_ROOT / "tools/build_inventory.json").read_text(encoding="utf-8")
        )
        mutations = (
            ("projection", "optional enumeration projection"),
            ("system-abi-null", "optional enumeration projection"),
            ("closure", "factory step wiring"),
            ("compile-checker", "must not inherit the POSIX structure checker"),
            ("ordinary-checker", "factory step wiring"),
            ("run-checker", "factory step wiring"),
            ("windows-checker", "Windows native compatibility field"),
            ("windows-run", "Windows native compatibility field"),
        )
        for mutation, expected in mutations:
            with self.subTest(mutation=mutation):
                build_inventory = copy.deepcopy(baseline)
                observations = {
                    row["id"]: row for row in build_inventory["build_observations"]
                }
                if mutation == "projection":
                    factory = observations[CHECKER.FACTORY_COMPILE_ID]
                    factory["enumeration_class_projection"]["fallback"] = (
                        "enumeration-class:aarch64-macos-system-macho"
                    )
                elif mutation == "system-abi-null":
                    factory = observations[CHECKER.FACTORY_COMPILE_ID]
                    system_mapping = next(
                        row
                        for row in factory["enumeration_class_projection"]["mappings"]
                        if row["enumeration_class_id"]
                        == "enumeration-class:aarch64-macos-system-macho"
                    )
                    system_mapping["abi"] = None
                elif mutation == "closure":
                    step = observations["step:build.zig:build:test-inventory-link"]
                    step["direct_dependencies"].pop()
                elif mutation == "compile-checker":
                    factory = observations[CHECKER.FACTORY_COMPILE_ID]
                    factory["structure_checker_dependency"] = (
                        CHECKER.BUILD_CHECKER.TEST_INVENTORY_STRUCTURE_CHECK_ID
                    )
                elif mutation == "ordinary-checker":
                    step = observations["step:build.zig:build:test-inventory-link"]
                    step["direct_dependencies"].pop(0)
                elif mutation == "run-checker":
                    step = observations["step:build.zig:build:test-inventory"]
                    step["direct_dependencies"].pop(0)
                elif mutation == "windows-checker":
                    step = observations[
                        CHECKER.BUILD_CHECKER.TEST_INVENTORY_WINDOWS_NATIVE_LINK_STEP_ID
                    ]
                    step["structure_checker_dependency"] = (
                        CHECKER.BUILD_CHECKER.TEST_INVENTORY_STRUCTURE_CHECK_ID
                    )
                else:
                    step = observations[
                        CHECKER.BUILD_CHECKER.TEST_INVENTORY_WINDOWS_NATIVE_LINK_STEP_ID
                    ]
                    step["test_body_execution"] = True
                build_path.write_text(
                    json.dumps(build_inventory, indent=2) + "\n", encoding="utf-8"
                )
                self.assertIn(expected, self._errors())

    def test_privacy_matrix_row_contract_and_digest_mutations_fail(self) -> None:
        forbidden = (
            ("resolved_command", "secret"),
            ("result", "pass"),
            ("hostname", "private"),
            ("timestamp", "2026-01-01"),
            ("run_id", "one"),
            ("log", ".local-docs/private.log"),
        )
        for key, value in forbidden:
            with self.subTest(key=key):
                inventory = copy.deepcopy(self.inventory)
                inventory["strict_summary"][key] = value
                self._write(inventory)
                self.assertIn("privacy", self._errors())

        inventory = copy.deepcopy(self.inventory)
        inventory["matrix_row_contract"]["required_row_ids"].pop()
        self._write(inventory)
        self.assertIn("matrix_row_contract", self._errors())

        inventory = copy.deepcopy(self.inventory)
        contract = inventory["matrix_row_contract"]
        contract["required_row_ids"].append(contract["required_row_ids"][0])
        self._write(inventory)
        self.assertIn("duplicate row IDs", self._errors())

        inventory = copy.deepcopy(self.inventory)
        inventory["expected_test_sets"][0]["digest"] = "0" * 64
        self._write(inventory)
        self.assertIn("digest", self._errors())

        self._write(copy.deepcopy(self.inventory))
        original_bytes = self.inventory_path.read_bytes()
        private_candidate = copy.deepcopy(self.inventory)
        private_candidate["strict_summary"]["log"] = "/Users/private/evidence.log"
        candidate_errors = CHECKER._validate_inventory_data(
            self.root,
            self.inventory_path,
            private_candidate,
            structure_only=True,
        )
        self.assertTrue(any("privacy" in error for error in candidate_errors))
        with (
            mock.patch.object(
                CHECKER,
                "refresh_from_protocol",
                side_effect=CHECKER.InventoryError("refreshed inventory is invalid"),
            ),
            mock.patch.object(CHECKER, "_publish_inventory_atomic") as publish,
        ):
            stderr = io.StringIO()
            with contextlib.redirect_stderr(stderr):
                result = CHECKER.main(
                    [
                        "--root",
                        str(self.root),
                        "--inventory",
                        str(self.inventory_path),
                        "--refresh-from-protocol",
                        "--protocol-log",
                        "env:aarch64-linux-gnu-baseline=unused.log",
                    ]
                )
            self.assertEqual(1, result)
            publish.assert_not_called()
            self.assertEqual(original_bytes, self.inventory_path.read_bytes())

        candidate_bytes = CHECKER._canonical_inventory_bytes(
            copy.deepcopy(self.inventory)
        )
        cleanup_arena = self.inventory_path.parent / (
            f".zynum-cleanup-v2-{os.geteuid()}"
        )
        for failure_point in ("write", "fsync", "replace"):
            with self.subTest(atomic_publication=failure_point):
                self.inventory_path.write_bytes(original_bytes)
                snapshot = CHECKER._read_regular_stable_snapshot(
                    self.inventory_path,
                    CHECKER.MAX_INVENTORY_BYTES,
                    "test inventory",
                )
                with mock.patch.object(
                    CHECKER.os,
                    failure_point,
                    side_effect=OSError(f"injected {failure_point} failure"),
                ):
                    with self.assertRaisesRegex(
                        CHECKER.InventoryError, "cannot publish refreshed inventory"
                    ) as publication_failure:
                        CHECKER._publish_inventory_atomic(
                            self.inventory_path, candidate_bytes, snapshot
                        )
                self.assertEqual(original_bytes, self.inventory_path.read_bytes())
                temporary_paths = list(
                    self.inventory_path.parent.glob(
                        f".{self.inventory_path.name}.*.tmp"
                    )
                )
                if failure_point == "fsync":
                    self.assertEqual(1, len(temporary_paths))
                    self.assertEqual(candidate_bytes, temporary_paths[0].read_bytes())
                    self.assertIn(
                        temporary_paths[0].name,
                        str(publication_failure.exception),
                    )
                    self.assertIn(
                        os.fspath(cleanup_arena),
                        str(publication_failure.exception),
                    )
                    self.assertIn(
                        "cleanup_quarantine_setup_failed",
                        str(publication_failure.exception),
                    )
                    self.assertIn(
                        "public_candidate=present",
                        str(publication_failure.exception),
                    )
                    self.assertIn(
                        f"candidate paths: {temporary_paths[0]}",
                        str(publication_failure.exception),
                    )
                    self.assertNotIn(
                        f"exact recovery paths: {temporary_paths[0]};",
                        str(publication_failure.exception),
                    )
                    temporary_paths[0].unlink()
                else:
                    self.assertEqual([], temporary_paths)

        for close_fails in (False, True):
            with self.subTest(
                atomic_publication="temporary-fstat",
                close_fails=close_fails,
            ):
                self.inventory_path.write_bytes(original_bytes)
                snapshot = CHECKER._read_regular_stable_snapshot(
                    self.inventory_path,
                    CHECKER.MAX_INVENTORY_BYTES,
                    "test inventory",
                )
                fstat_token = ("d" if close_fails else "c") * 24
                temporary_name = f".{self.inventory_path.name}.{fstat_token}.tmp"
                candidate_path = self.inventory_path.with_name(temporary_name)
                actual_fstat = CHECKER.os.fstat
                actual_close = CHECKER.os.close
                temporary_descriptor: int | None = None
                temporary_close_attempts = 0

                def fail_first_temporary_fstat(
                    descriptor: int,
                ) -> os.stat_result:
                    nonlocal temporary_descriptor
                    if temporary_descriptor is None:
                        temporary_descriptor = descriptor
                        raise OSError("injected temporary fstat failure")
                    return actual_fstat(descriptor)

                def close_temporary_once(descriptor: int) -> None:
                    nonlocal temporary_close_attempts
                    if descriptor == temporary_descriptor:
                        temporary_close_attempts += 1
                        actual_close(descriptor)
                        if close_fails:
                            raise OSError("injected temporary close failure")
                        return
                    actual_close(descriptor)

                try:
                    with (
                        mock.patch.object(
                            CHECKER.secrets,
                            "token_hex",
                            return_value=fstat_token,
                        ),
                        mock.patch.object(
                            CHECKER.os,
                            "fstat",
                            side_effect=fail_first_temporary_fstat,
                        ),
                        mock.patch.object(
                            CHECKER.os,
                            "close",
                            side_effect=close_temporary_once,
                        ),
                        mock.patch.object(CHECKER.os, "replace") as publish,
                        self.assertRaisesRegex(
                            CHECKER.InventoryError,
                            "temporary identity could not be established",
                        ) as publication_failure,
                    ):
                        CHECKER._publish_inventory_atomic(
                            self.inventory_path, candidate_bytes, snapshot
                        )
                    publish.assert_not_called()
                    self.assertEqual(1, temporary_close_attempts)
                    self.assertEqual(original_bytes, self.inventory_path.read_bytes())
                    self.assertTrue(candidate_path.exists())
                    self.assertIn(
                        f"public temporary candidate path retained as {candidate_path}",
                        str(publication_failure.exception),
                    )
                    if close_fails:
                        self.assertIn(
                            "temporary descriptor close failed: injected temporary "
                            "close failure",
                            str(publication_failure.exception),
                        )
                    else:
                        self.assertNotIn(
                            "temporary descriptor close failed",
                            str(publication_failure.exception),
                        )
                finally:
                    candidate_path.unlink(missing_ok=True)

        self.inventory_path.write_bytes(original_bytes)
        snapshot = CHECKER._read_regular_stable_snapshot(
            self.inventory_path, CHECKER.MAX_INVENTORY_BYTES, "test inventory"
        )
        close_token = "e" * 24
        close_temporary_name = f".{self.inventory_path.name}.{close_token}.tmp"
        close_candidate_path = self.inventory_path.with_name(close_temporary_name)
        actual_fstat = CHECKER.os.fstat
        actual_close = CHECKER.os.close
        owned_temporary_descriptor: int | None = None
        owned_temporary_close_attempts = 0

        def capture_owned_temporary(descriptor: int) -> os.stat_result:
            nonlocal owned_temporary_descriptor
            metadata = actual_fstat(descriptor)
            if owned_temporary_descriptor is None and stat.S_ISREG(metadata.st_mode):
                owned_temporary_descriptor = descriptor
            return metadata

        def fail_owned_temporary_close(descriptor: int) -> None:
            nonlocal owned_temporary_close_attempts
            if descriptor == owned_temporary_descriptor:
                owned_temporary_close_attempts += 1
                actual_close(descriptor)
                raise OSError("injected owned temporary close failure")
            actual_close(descriptor)

        try:
            with (
                mock.patch.object(
                    CHECKER.secrets,
                    "token_hex",
                    return_value=close_token,
                ),
                mock.patch.object(
                    CHECKER.os,
                    "fstat",
                    side_effect=capture_owned_temporary,
                ),
                mock.patch.object(
                    CHECKER.os,
                    "close",
                    side_effect=fail_owned_temporary_close,
                ),
                mock.patch.object(
                    CHECKER, "_claim_and_remove_inventory_temporary"
                ) as claim_temporary,
                mock.patch.object(
                    CHECKER, "_publication_capability_error", return_value=None
                ),
                mock.patch.object(CHECKER.os, "unlink") as unlink_temporary,
                mock.patch.object(CHECKER.os, "rmdir") as remove_temporary,
                mock.patch.object(CHECKER.os, "replace") as publish,
                self.assertRaisesRegex(
                    CHECKER.InventoryError,
                    "owned temporary close failure",
                ) as close_failure,
            ):
                CHECKER._publish_inventory_atomic(
                    self.inventory_path, candidate_bytes, snapshot
                )
            self.assertEqual(1, owned_temporary_close_attempts)
            claim_temporary.assert_not_called()
            unlink_temporary.assert_not_called()
            remove_temporary.assert_not_called()
            publish.assert_not_called()
            self.assertEqual(original_bytes, self.inventory_path.read_bytes())
            self.assertEqual(candidate_bytes, close_candidate_path.read_bytes())
            self.assertIn(
                f"public temporary candidate path retained as {close_candidate_path}",
                str(close_failure.exception),
            )
        finally:
            close_candidate_path.unlink(missing_ok=True)

        self.inventory_path.write_bytes(original_bytes)
        snapshot = CHECKER._read_regular_stable_snapshot(
            self.inventory_path, CHECKER.MAX_INVENTORY_BYTES, "test inventory"
        )
        collision_path = self.inventory_path.with_name(
            f".{self.inventory_path.name}.collision.tmp"
        )
        collision_bytes = b"foreign temporary file"
        collision_path.write_bytes(collision_bytes)
        try:
            with mock.patch.object(
                CHECKER.secrets, "token_hex", return_value="collision"
            ):
                with self.assertRaisesRegex(
                    CHECKER.InventoryError, "unique inventory temporary file"
                ):
                    CHECKER._publish_inventory_atomic(
                        self.inventory_path, candidate_bytes, snapshot
                    )
            self.assertEqual(original_bytes, self.inventory_path.read_bytes())
            self.assertEqual(collision_bytes, collision_path.read_bytes())
        finally:
            collision_path.unlink(missing_ok=True)

        shared_name = f".{self.inventory_path.name}.shared-cleanup.tmp"
        shared_path = self.inventory_path.with_name(shared_name)
        shared_path.write_bytes(candidate_bytes)
        shared_metadata = shared_path.stat()
        shared_identity = (shared_metadata.st_dev, shared_metadata.st_ino)
        directory_descriptor = os.open(
            self.inventory_path.parent,
            os.O_RDONLY | os.O_CLOEXEC | os.O_DIRECTORY | os.O_NOFOLLOW,
        )
        normal_token = "5" * 24
        prior_cleanup_entries = (
            set(cleanup_arena.iterdir()) if cleanup_arena.is_dir() else set()
        )
        normal_quarantine = cleanup_arena / (
            f"{shared_name}.{normal_token}.cleanup-quarantine"
        )
        try:
            with mock.patch.object(
                CHECKER.REPOSITORY_SNAPSHOT.secrets,
                "token_hex",
                return_value=normal_token,
            ):
                CHECKER._claim_and_remove_inventory_temporary(
                    directory_descriptor,
                    shared_name,
                    shared_identity,
                    candidate_bytes,
                    directory_path=self.inventory_path.parent,
                )
            self.assertFalse(shared_path.exists())
            self.assertFalse(normal_quarantine.exists())
            self.assertTrue(cleanup_arena.is_dir())
            self.assertEqual(prior_cleanup_entries, set(cleanup_arena.iterdir()))
        finally:
            os.close(directory_descriptor)
            shared_path.unlink(missing_ok=True)
            shutil.rmtree(normal_quarantine, ignore_errors=True)

        preclaim_name = f".{self.inventory_path.name}.shared-preclaim.tmp"
        preclaim_path = self.inventory_path.with_name(preclaim_name)
        preclaim_path.write_bytes(candidate_bytes)
        preclaim_metadata = preclaim_path.stat()
        preclaim_identity = (preclaim_metadata.st_dev, preclaim_metadata.st_ino)
        preclaim_token = "8" * 24
        preclaim_quarantine = cleanup_arena / (
            f"{preclaim_name}.{preclaim_token}.cleanup-quarantine"
        )
        directory_descriptor = os.open(
            self.inventory_path.parent,
            os.O_RDONLY | os.O_CLOEXEC | os.O_DIRECTORY | os.O_NOFOLLOW,
        )
        actual_rename = CHECKER.REPOSITORY_SNAPSHOT.os.rename

        def fail_preclaim_rename(
            source: str,
            destination: str,
            *args: Any,
            **kwargs: Any,
        ) -> None:
            if source == preclaim_name:
                raise OSError("injected preclaim rename failure")
            actual_rename(source, destination, *args, **kwargs)

        try:
            with (
                mock.patch.object(
                    CHECKER.REPOSITORY_SNAPSHOT.secrets,
                    "token_hex",
                    return_value=preclaim_token,
                ),
                mock.patch.object(
                    CHECKER.REPOSITORY_SNAPSHOT.os,
                    "rename",
                    side_effect=fail_preclaim_rename,
                ),
                self.assertRaisesRegex(
                    CHECKER.InventoryError,
                    "cleanup failed closed",
                ) as preclaim_failure,
            ):
                CHECKER._claim_and_remove_inventory_temporary(
                    directory_descriptor,
                    preclaim_name,
                    preclaim_identity,
                    candidate_bytes,
                    directory_path=self.inventory_path.parent,
                )
            self.assertEqual(candidate_bytes, preclaim_path.read_bytes())
            self.assertTrue(preclaim_quarantine.is_dir())
            preclaim_message = str(preclaim_failure.exception)
            self.assertIn("disposition=retained", preclaim_message)
            self.assertIn("arena_binding=bound", preclaim_message)
            self.assertIn("public_candidate=present", preclaim_message)
            self.assertIn(
                f"exact recovery paths: {preclaim_quarantine}", preclaim_message
            )
            self.assertIn(f"candidate paths: {preclaim_path}", preclaim_message)
            self.assertNotIn(
                f"exact recovery paths: {preclaim_path};", preclaim_message
            )
            self.assertIn("cleanup_claim_failed", preclaim_message)
        finally:
            os.close(directory_descriptor)
            preclaim_path.unlink(missing_ok=True)
            shutil.rmtree(preclaim_quarantine, ignore_errors=True)

        unprotected_name = f".{self.inventory_path.name}.shared-unprotected.tmp"
        unprotected_path = self.inventory_path.with_name(unprotected_name)
        unprotected_path.write_bytes(candidate_bytes)
        unprotected_metadata = unprotected_path.stat()
        unprotected_identity = (
            unprotected_metadata.st_dev,
            unprotected_metadata.st_ino,
        )
        parent_mode = stat.S_IMODE(self.inventory_path.parent.stat().st_mode)
        directory_descriptor = -1
        try:
            self.inventory_path.parent.chmod(0o777)
            directory_descriptor = os.open(
                self.inventory_path.parent,
                os.O_RDONLY | os.O_CLOEXEC | os.O_DIRECTORY | os.O_NOFOLLOW,
            )
            with self.assertRaisesRegex(
                CHECKER.InventoryError,
                "cleanup failed closed",
            ) as unprotected_failure:
                CHECKER._claim_and_remove_inventory_temporary(
                    directory_descriptor,
                    unprotected_name,
                    unprotected_identity,
                    candidate_bytes,
                    directory_path=self.inventory_path.parent,
                )
            self.assertEqual(candidate_bytes, unprotected_path.read_bytes())
            unprotected_message = str(unprotected_failure.exception)
            self.assertIn("disposition=unaddressable", unprotected_message)
            self.assertIn("public_candidate=present", unprotected_message)
            self.assertIn("exact recovery paths: none", unprotected_message)
            self.assertIn(f"candidate paths: {unprotected_path}", unprotected_message)
            self.assertIn(
                "cleanup_recovery_anchor_not_rename_protected",
                unprotected_message,
            )
        finally:
            if directory_descriptor >= 0:
                os.close(directory_descriptor)
            self.inventory_path.parent.chmod(parent_mode)
            unprotected_path.unlink(missing_ok=True)

        rebound_name = f".{self.inventory_path.name}.shared-rebound.tmp"
        rebound_path = self.inventory_path.with_name(rebound_name)
        rebound_path.write_bytes(candidate_bytes)
        rebound_metadata = rebound_path.stat()
        rebound_identity = (rebound_metadata.st_dev, rebound_metadata.st_ino)
        displaced_arena = cleanup_arena.with_name(f"{cleanup_arena.name}.displaced")
        shutil.rmtree(displaced_arena, ignore_errors=True)
        directory_descriptor = os.open(
            self.inventory_path.parent,
            os.O_RDONLY | os.O_CLOEXEC | os.O_DIRECTORY | os.O_NOFOLLOW,
        )
        actual_arena_open = CHECKER.REPOSITORY_SNAPSHOT.CleanupArena.open

        def open_then_rebind_arena(*args: Any, **kwargs: Any) -> Any:
            arena = actual_arena_open(*args, **kwargs)
            os.rename(cleanup_arena, displaced_arena)
            cleanup_arena.mkdir(mode=0o700)
            return arena

        try:
            with (
                mock.patch.object(
                    CHECKER.REPOSITORY_SNAPSHOT.CleanupArena,
                    "open",
                    side_effect=open_then_rebind_arena,
                ),
                self.assertRaisesRegex(
                    CHECKER.InventoryError,
                    "cleanup failed closed",
                ) as rebound_failure,
            ):
                CHECKER._claim_and_remove_inventory_temporary(
                    directory_descriptor,
                    rebound_name,
                    rebound_identity,
                    candidate_bytes,
                    directory_path=self.inventory_path.parent,
                )
            self.assertEqual(candidate_bytes, rebound_path.read_bytes())
            rebound_message = str(rebound_failure.exception)
            self.assertIn("disposition=unaddressable", rebound_message)
            self.assertIn("arena_binding=rebound", rebound_message)
            self.assertIn("public_candidate=present", rebound_message)
            self.assertIn("exact recovery paths: none", rebound_message)
            self.assertIn(f"candidate paths: {rebound_path}", rebound_message)
            self.assertIn("cleanup_arena_binding_rebound", rebound_message)
        finally:
            os.close(directory_descriptor)
            rebound_path.unlink(missing_ok=True)
            if displaced_arena.is_dir():
                shutil.rmtree(cleanup_arena, ignore_errors=True)
                os.rename(displaced_arena, cleanup_arena)

        foreign_name = f".{self.inventory_path.name}.shared-foreign.tmp"
        foreign_path = self.inventory_path.with_name(foreign_name)
        foreign_bytes = b"foreign shared cleanup artifact"
        foreign_path.write_bytes(foreign_bytes)
        foreign_metadata = foreign_path.stat()
        foreign_identity = (foreign_metadata.st_dev, foreign_metadata.st_ino)
        foreign_token = "6" * 24
        foreign_quarantine = cleanup_arena / (
            f"{foreign_name}.{foreign_token}.cleanup-quarantine"
        )
        foreign_recovery = foreign_quarantine / "claimed"
        directory_descriptor = os.open(
            self.inventory_path.parent,
            os.O_RDONLY | os.O_CLOEXEC | os.O_DIRECTORY | os.O_NOFOLLOW,
        )
        try:
            with (
                mock.patch.object(
                    CHECKER.REPOSITORY_SNAPSHOT.secrets,
                    "token_hex",
                    return_value=foreign_token,
                ),
                self.assertRaisesRegex(
                    CHECKER.InventoryError,
                    "cleanup failed closed",
                ) as foreign_failure,
            ):
                CHECKER._claim_and_remove_inventory_temporary(
                    directory_descriptor,
                    foreign_name,
                    foreign_identity,
                    candidate_bytes,
                    directory_path=self.inventory_path.parent,
                )
            self.assertFalse(foreign_path.exists())
            self.assertEqual(foreign_bytes, foreign_recovery.read_bytes())
            self.assertIn(
                os.fspath(foreign_recovery),
                str(foreign_failure.exception),
            )
            self.assertIn("public_candidate=absent", str(foreign_failure.exception))
            self.assertIn("candidate paths: none", str(foreign_failure.exception))
        finally:
            os.close(directory_descriptor)
            foreign_path.unlink(missing_ok=True)
            shutil.rmtree(foreign_quarantine, ignore_errors=True)

        swapped_name = f".{self.inventory_path.name}.shared-swapped.tmp"
        swapped_path = self.inventory_path.with_name(swapped_name)
        displaced_candidate = self.inventory_path.with_name(
            f"{swapped_name}.displaced-candidate"
        )
        swapped_path.write_bytes(candidate_bytes)
        swapped_metadata = swapped_path.stat()
        expected_swapped_identity = (
            swapped_metadata.st_dev,
            swapped_metadata.st_ino,
        )
        os.rename(swapped_path, displaced_candidate)
        swapped_foreign_bytes = b"foreign artifact swapped before shared claim"
        swapped_path.write_bytes(swapped_foreign_bytes)
        swapped_token = "9" * 24
        swapped_quarantine = cleanup_arena / (
            f"{swapped_name}.{swapped_token}.cleanup-quarantine"
        )
        swapped_recovery = swapped_quarantine / "claimed"
        directory_descriptor = os.open(
            self.inventory_path.parent,
            os.O_RDONLY | os.O_CLOEXEC | os.O_DIRECTORY | os.O_NOFOLLOW,
        )
        try:
            with (
                mock.patch.object(
                    CHECKER.REPOSITORY_SNAPSHOT.secrets,
                    "token_hex",
                    return_value=swapped_token,
                ),
                self.assertRaisesRegex(
                    CHECKER.InventoryError,
                    "cleanup failed closed",
                ) as swapped_failure,
            ):
                CHECKER._claim_and_remove_inventory_temporary(
                    directory_descriptor,
                    swapped_name,
                    expected_swapped_identity,
                    candidate_bytes,
                    directory_path=self.inventory_path.parent,
                )
            self.assertFalse(swapped_path.exists())
            self.assertEqual(candidate_bytes, displaced_candidate.read_bytes())
            self.assertEqual(swapped_foreign_bytes, swapped_recovery.read_bytes())
            swapped_message = str(swapped_failure.exception)
            self.assertIn("exact recovery paths:", swapped_message)
            self.assertIn(os.fspath(swapped_recovery), swapped_message)
            self.assertIn("public_candidate=absent", swapped_message)
            self.assertIn("candidate paths: none", swapped_message)
            self.assertNotIn(os.fspath(displaced_candidate), swapped_message)
        finally:
            os.close(directory_descriptor)
            swapped_path.unlink(missing_ok=True)
            displaced_candidate.unlink(missing_ok=True)
            shutil.rmtree(swapped_quarantine, ignore_errors=True)

        uncertain_name = f".{self.inventory_path.name}.shared-close.tmp"
        uncertain_path = self.inventory_path.with_name(uncertain_name)
        uncertain_path.write_bytes(candidate_bytes)
        uncertain_metadata = uncertain_path.stat()
        uncertain_identity = (uncertain_metadata.st_dev, uncertain_metadata.st_ino)
        uncertain_token = "7" * 24
        uncertain_quarantine = cleanup_arena / (
            f"{uncertain_name}.{uncertain_token}.cleanup-quarantine"
        )
        directory_descriptor = os.open(
            self.inventory_path.parent,
            os.O_RDONLY | os.O_CLOEXEC | os.O_DIRECTORY | os.O_NOFOLLOW,
        )
        actual_close_once = CHECKER.REPOSITORY_SNAPSHOT.OwnedDescriptor.close_once
        quarantine_close_attempts = 0

        def fail_shared_quarantine_close(owner: Any) -> None:
            nonlocal quarantine_close_attempts
            if owner.recovery_path == uncertain_quarantine:
                quarantine_close_attempts += 1
                actual_close_once(owner)
                raise OSError("injected shared quarantine close uncertainty")
            actual_close_once(owner)

        try:
            with (
                mock.patch.object(
                    CHECKER.REPOSITORY_SNAPSHOT.secrets,
                    "token_hex",
                    return_value=uncertain_token,
                ),
                mock.patch.object(
                    CHECKER.REPOSITORY_SNAPSHOT.OwnedDescriptor,
                    "close_once",
                    autospec=True,
                    side_effect=fail_shared_quarantine_close,
                ),
                mock.patch.object(
                    CHECKER.REPOSITORY_SNAPSHOT.os, "rmdir"
                ) as remove_quarantine,
                self.assertRaisesRegex(
                    CHECKER.InventoryError,
                    "cleanup_quarantine_descriptor_close_uncertain",
                ) as uncertain_failure,
            ):
                CHECKER._claim_and_remove_inventory_temporary(
                    directory_descriptor,
                    uncertain_name,
                    uncertain_identity,
                    candidate_bytes,
                    directory_path=self.inventory_path.parent,
                )
            self.assertEqual(1, quarantine_close_attempts)
            remove_quarantine.assert_not_called()
            self.assertFalse(uncertain_path.exists())
            self.assertTrue(uncertain_quarantine.is_dir())
            self.assertEqual([], list(uncertain_quarantine.iterdir()))
            self.assertIn(
                os.fspath(uncertain_quarantine),
                str(uncertain_failure.exception),
            )
        finally:
            os.close(directory_descriptor)
            uncertain_path.unlink(missing_ok=True)
            shutil.rmtree(uncertain_quarantine, ignore_errors=True)
        self.inventory_path.write_bytes(original_bytes)
        snapshot = CHECKER._read_regular_stable_snapshot(
            self.inventory_path, CHECKER.MAX_INVENTORY_BYTES, "test inventory"
        )
        racing_bytes = original_bytes + b" "
        original_reader = CHECKER._read_regular_stable_snapshot

        def race_reader(*args: Any, **kwargs: Any) -> Any:
            if len(args) >= 3 and args[2] == "inventory publication target":
                self.inventory_path.write_bytes(racing_bytes)
            return original_reader(*args, **kwargs)

        with mock.patch.object(
            CHECKER, "_read_regular_stable_snapshot", side_effect=race_reader
        ):
            with self.assertRaisesRegex(CHECKER.InventoryError, "lost-update"):
                CHECKER._publish_inventory_atomic(
                    self.inventory_path, candidate_bytes, snapshot
                )
        self.assertEqual(racing_bytes, self.inventory_path.read_bytes())

        self.inventory_path.write_bytes(original_bytes)
        snapshot = CHECKER._read_regular_stable_snapshot(
            self.inventory_path, CHECKER.MAX_INVENTORY_BYTES, "test inventory"
        )
        with self.assertRaisesRegex(
            CHECKER.InventoryError, "candidate inventory exceeds"
        ):
            CHECKER._publish_inventory_atomic(
                self.inventory_path,
                b"x" * (CHECKER.MAX_INVENTORY_BYTES + 1),
                snapshot,
            )
        self.assertEqual(original_bytes, self.inventory_path.read_bytes())

        original_fsync = CHECKER.os.fsync
        for failure_point in ("directory-fsync", "directory-close"):
            with self.subTest(indeterminate_publication=failure_point):
                self.inventory_path.write_bytes(original_bytes)
                snapshot = CHECKER._read_regular_stable_snapshot(
                    self.inventory_path,
                    CHECKER.MAX_INVENTORY_BYTES,
                    "test inventory",
                )
                if failure_point == "directory-fsync":
                    fsync_calls = 0

                    def fail_directory_fsync(descriptor: int) -> None:
                        nonlocal fsync_calls
                        fsync_calls += 1
                        if fsync_calls == 2:
                            raise OSError("injected directory fsync failure")
                        original_fsync(descriptor)

                    publication_patch = mock.patch.object(
                        CHECKER.os, "fsync", side_effect=fail_directory_fsync
                    )
                else:
                    original_close = CHECKER.os.close

                    def fail_directory_close(descriptor: int) -> None:
                        is_directory = stat.S_ISDIR(os.fstat(descriptor).st_mode)
                        original_close(descriptor)
                        if is_directory:
                            raise OSError("injected directory close failure")

                    publication_patch = mock.patch.object(
                        CHECKER.os, "close", side_effect=fail_directory_close
                    )
                with publication_patch:
                    with self.assertRaisesRegex(
                        CHECKER.InventoryPublicationIndeterminate,
                        "candidate installed but durability uncertain",
                    ):
                        CHECKER._publish_inventory_atomic(
                            self.inventory_path, candidate_bytes, snapshot
                        )
                self.assertEqual(candidate_bytes, self.inventory_path.read_bytes())

        self.inventory_path.write_bytes(original_bytes)
        snapshot = CHECKER._read_regular_stable_snapshot(
            self.inventory_path, CHECKER.MAX_INVENTORY_BYTES, "test inventory"
        )
        mocked_candidate = CHECKER.RefreshedInventoryCandidate(
            copy.deepcopy(self.inventory),
            candidate_bytes,
            snapshot,
            CHECKER._matrix_incomplete_count(self.inventory),
        )
        with (
            mock.patch.object(
                CHECKER, "refresh_from_protocol", return_value=mocked_candidate
            ),
            mock.patch.object(
                CHECKER,
                "_publish_inventory_atomic",
                side_effect=CHECKER.InventoryPublicationIndeterminate(
                    "candidate installed but durability uncertain: injected"
                ),
            ),
        ):
            stderr = io.StringIO()
            with contextlib.redirect_stderr(stderr):
                result = CHECKER.main(
                    [
                        "--root",
                        str(self.root),
                        "--inventory",
                        str(self.inventory_path),
                        "--refresh-from-protocol",
                        "--protocol-log",
                        "env:aarch64-linux-gnu-baseline=unused.log",
                    ]
                )
        self.assertEqual(3, result)
        self.assertIn("candidate installed but durability uncertain", stderr.getvalue())
        self.assertNotIn("old", stderr.getvalue().lower())

        depth_999_bytes = b'{"x":' * 999 + b"0" + b"}" * 999
        self.inventory_path.write_bytes(depth_999_bytes)
        depth_digest = hashlib.sha256(depth_999_bytes).hexdigest()
        ambient_git_pager = os.environ.pop("GIT_PAGER", None)
        try:
            stderr = io.StringIO()
            with (
                self._reviewed_digest_slots(inventory_next=depth_digest),
                contextlib.redirect_stderr(stderr),
            ):
                result = CHECKER.main(
                    [
                        "--root",
                        str(self.root),
                        "--inventory",
                        str(self.inventory_path),
                        "--structure-only",
                    ]
                )
        finally:
            if ambient_git_pager is not None:
                os.environ["GIT_PAGER"] = ambient_git_pager
        output = stderr.getvalue()
        self.assertEqual(1, result)
        self.assertEqual(1, len(output.splitlines()))
        self.assertIn("maximum depth", output)
        self.assertNotIn("Traceback", output)
        self.assertEqual(depth_999_bytes, self.inventory_path.read_bytes())


if __name__ == "__main__":
    unittest.main()
