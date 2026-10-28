# Copyright (C) 2026 Zynum contributors
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Contract and mutation tests for the build inventory validator."""

from __future__ import annotations

import contextlib
import copy
import errno
import hashlib
import importlib.util
import io
import json
import re
import os
import platform
import shutil
import stat
import subprocess
import sys
import tarfile
import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest import mock

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
CHECKER_PATH = REPOSITORY_ROOT / "tools/check_build_inventory.py"
SPEC = importlib.util.spec_from_file_location("check_build_inventory", CHECKER_PATH)
assert SPEC and SPEC.loader
CHECKER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CHECKER)



def _with_source_reviewed_execution_compile(inventory: dict[str, Any]) -> dict[str, Any]:
    """Stage the exact new source-only row for old-ledger migration fixtures.

    No test sets, observations of execution, pins or files are published here.
    The real-source graph/factory tests separately bind this Compile identity.
    """
    identifier = "compile:build.zig:build:execution_contract_tests"
    if not any(row["id"] == identifier for row in inventory["build_observations"]):
        inventory["build_observations"].append({"id": identifier, **CHECKER._new_test_inventory_observation(identifier, inventory)})
    return inventory


def _build_body_insertion_offset(text: str, relative: str = "build.zig") -> int:
    """Mutate the real construction body, before its reviewed final return."""
    projected = CHECKER._reviewed_configure_projection(text)
    build = CHECKER._zig_build_context(projected, relative)
    if projected == text:
        return build.body_end
    match = CHECKER.re.search(r"\breturn\b", CHECKER._code_mask(text)[build.body_start:build.body_end])
    assert match is not None
    return build.body_start + match.start()


def _prior_component_observation(test, row, width, count):
    """Validate the new finite SYEV branch before checking the old packages."""
    if width != "lp64":
        test.assertNotIn("lapacke-syev=true", row["source_variants"])
        return row
    variants = row["source_variants"]
    root = "src/lapacke_symmetric_composed.zig"
    test.assertEqual(count + 2, len(variants))
    test.assertEqual(root, variants["lapacke-syev=true"])
    prior = {key.removeprefix("lapacke-syev=false,"): value
             for key, value in variants.items() if key.startswith("lapacke-syev=false,")}
    test.assertEqual(count + 1, len(prior))
    test.assertEqual({"lapacke-syev=true"} | {"lapacke-syev=false," + key for key in prior}, set(variants))
    test.assertEqual(set(prior.values()) | {root}, set(row["root_source"]))
    return {**row, "source_variants": prior,
            "root_source": [path for path in row["root_source"] if path != root]}


class NativeConfigureSeamSourceContractTests(unittest.TestCase):
    """Reviewed native graph structure, never inferred native test success."""
    @classmethod
    def setUpClass(cls) -> None:
        cls.source = (REPOSITORY_ROOT / "build.zig").read_text(encoding="utf-8")

    def rejects(self, changed: str) -> None:
        self.assertNotEqual(self.source, changed, "mutation must hit a real source anchor")
        with self.assertRaisesRegex(CHECKER.InventoryError, "reviewed native configure seam changed"):
            CHECKER._reviewed_configure_projection(changed)

    def helper_mutation(self, helper: str, old: str, new: str) -> str:
        masked = CHECKER._code_mask(self.source)
        match = re.search(r"\bfn\s+" + helper + r"\s*\(", masked)
        self.assertIsNotNone(match)
        opening = masked.find("{", match.end())
        closing = CHECKER._matching_brace(masked, opening)
        body = self.source[match.start():closing + 1]
        self.assertEqual(1, body.count(old), (helper, old))
        return self.source[:match.start()] + body.replace(old, new, 1) + self.source[closing + 1:]

    def test_current_projection_is_offset_preserving_and_idempotent(self) -> None:
        projected = CHECKER._reviewed_configure_projection(self.source)
        self.assertEqual(len(self.source), len(projected))
        self.assertEqual([i for i, ch in enumerate(self.source) if ch == "\n"],
                         [i for i, ch in enumerate(projected) if ch == "\n"])
        self.assertEqual(projected, CHECKER._reviewed_configure_projection(projected))
        self.assertIn('b.path("src/numerics/execution/root.zig")', projected)
        self.assertNotIn("configureIntegrated", CHECKER._code_mask(projected))

    def test_profile_must_select_one_owner_graph_before_reference_construction(self) -> None:
        for old, new in (
            ("execution_profile == .integrated_native) return configureIntegrated", "execution_profile == .reference) return configureIntegrated"),
            ("orelse .reference;", "orelse .integrated_native;"),
            ("reference_compat != (execution_profile == .reference)", "false"),
            ('if (b.modules.contains("zynum-core"))', "if (false)"),
            (".lapacke_binding = lapacke_binding_mod,", ".lapacke_binding = undefined,"),
            ("lapacke_binding: ?*std.Build.Module = null,", "lapacke_binding: *std.Build.Module,"),
        ):
            with self.subTest(old=old): self.rejects(self.source.replace(old, new, 1))

    def test_no_forwarded_second_configure_or_escaped_graph_alias(self) -> None:
        for body in ("", "_ = configure(b); _ = configure(b);", "if (true) { _ = configure(b); }",
                     "const f = configure; _ = f(b);", "_ = configure(other);"):
            self.rejects(self.source.replace("_ = configure(b);", body, 1))
        self.rejects(self.source + "\nconst EscapedGraph = ImplementationGraph;\n")
        self.rejects(self.source + "\nconst escaped = configureIntegrated;\n")

    def test_native_cannot_clone_core_execution_resources_or_provider(self) -> None:
        for helper, old, new in (
            ("configureIntegrated", 'core.addImport("zynum-execution", execution);', 'core.addImport("zynum-execution", b.createModule(.{}));'),
            ("configureIntegrated", 'execution.addImport("zynum-numerics-resources", resources);', 'execution.addImport("zynum-numerics-resources", b.createModule(.{}));'),
            ("configureIntegrated", 'resources.addImport("heteris-provider-c", provider_c);', 'resources.addImport("heteris-provider-c", translateProvider(b, target, optimize, heteris_include));'),
            ("addNativeFacade", 'native.addImport("zynum-core", core);', 'native.addImport("zynum-core", b.createModule(.{}));'),
            ("addPlatformLibrary", 'main.addImport("zynum-core", graph.core);', 'main.addImport("zynum-core", b.createModule(.{}));'),
            ("addPlatformLibrary", 'upstream.addImport("zynum-core", graph.core);', 'upstream.addImport("zynum-core", b.createModule(.{}));'),
        ):
            with self.subTest(helper=helper, old=old): self.rejects(self.helper_mutation(helper, old, new))

    def test_native_default_legacy_and_second_archive_leaks_are_rejected(self) -> None:
        for old, new in (
            ('b.path("src/numerics/execution/root.zig")', 'b.path("src/numerics/execution/default.zig")'),
            ('options.addOption(bool, "reference_compat", false);', 'options.addOption(bool, "reference_compat", true);'),
            ('const library = addPlatformLibrary', 'const hidden_owner = b.createModule(.{}); _ = hidden_owner; const library = addPlatformLibrary'),
        ):
            self.rejects(self.helper_mutation("configureIntegrated", old, new))
        self.rejects(self.helper_mutation("addPlatformLibrary", 'main.addImport("zynum-core", graph.core);',
            'main.addImport("zynum-core", graph.core); main.linkLibrary(b.addLibrary(.{ .name = "second-core", .root_module = graph.core }));'))

    def test_sdk_conflict_and_archive_gates_cannot_be_disabled(self) -> None:
        self.rejects(self.helper_mutation("headerIncludePath", "normalized_option != null and normalized_inherited != null", "false"))
        self.rejects(self.helper_mutation("normalizeHeaderPath", "!std.fs.path.isAbsolute(path)", "false"))
        self.rejects(self.helper_mutation("configureIntegrated", 'if (!std.fs.path.isAbsolute(archive))', 'if (false)'))


    def test_source_only_native_templates_do_not_create_protocol_evidence(self) -> None:
        templates = CHECKER.INJECTION_OBSERVATION_TEMPLATES
        translated = templates["compile:build.zig:translateProvider:translated"]
        self.assertEqual(["include/zynum_native.h"], translated["root_source"])
        self.assertEqual("zynum_native.zig", translated["output_name"])
        self.assertEqual(["zynum_native.zig"], translated["produced_outputs"])
        self.assertEqual(["heteris.h", "heteris_runtime_provider.h"], translated["external_include_dependencies"])
        for suffix in ("math_tests", "c_tests"):
            row = templates["compile:build.zig:configureIntegrated:" + suffix]
            self.assertFalse(row["native_inventory_evidence"])
            self.assertIn("explicit absolute heteris-runtime-archive", row["condition"])
        for name in ("execution-profile", "reference-compat"):
            self.assertIn("option:build.zig:configure:" + name, templates)
        self.assertEqual("option:build.zig:build:poison-reference",
                         next(row["source_observation"] for row in CHECKER.INJECTION_OPTION_SURFACES
                              if row["name"] == "poison-reference"))
        with self.assertRaisesRegex(CHECKER.InventoryError, "no reviewed template"):
            CHECKER._new_test_inventory_observation("compile:build.zig:unreviewed:math_tests", {})
        self.rejects(self.helper_mutation("configureIntegrated",
            'options.addOption(bool, "poison_reference", b.option(bool, "poison-reference", "Test-only poison for actual legacy reference imports") orelse false);',
            'options.addOption(bool, "poison_reference", false);'))



class ConfigureSeamSourceContractTests(unittest.TestCase):
    """Exact source composition, never a new native-execution observation."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.current_source = (REPOSITORY_ROOT / "build.zig").read_text(encoding="utf-8")
        cls.graph_type = "pub const ImplementationGraph = struct {\n" + "".join(
            f"    {field}: *std.Build.Module,\n" for field in (
                "core", "execution", "resources", "options", "lapacke_binding", "mkl_sparse_binding")
        ) + "};"
        cls.thin_build = "pub fn build(b: *std.Build) void {\n    _ = configure(b);\n}"
        cls.config_header = "pub fn configure(b: *std.Build) ImplementationGraph {"
        cls.return_block = "return .{\n" + "".join(
            f"        .{field} = {owner},\n" for field, owner in (
                ("core", "zynum_core_mod"), ("execution", "execution_mod"),
                ("resources", "numerics_resources_mod"), ("options", "zynum_build_options_mod"),
                ("lapacke_binding", "lapacke_binding_mod"), ("mkl_sparse_binding", "mkl_sparse_binding_mod"))
        ) + "    };"
        # Keep the old finite six-owner seam's negative coverage. The current
        # native seam is checked independently below; this fixture wraps the
        # same reviewed reference body, with no imported execution observation.
        projected = CHECKER._reviewed_configure_projection(cls.current_source)
        context = CHECKER._zig_build_context(projected, "build.zig")
        signature = re.search(r"\bpub\s+fn\s+build\s*\(", CHECKER._code_mask(projected))
        if signature is None:
            raise AssertionError("reference projection has no build signature")
        cls.source = (projected[:signature.start()] + cls.graph_type + "\n" + cls.thin_build + "\n" +
                      cls.config_header + projected[context.body_start:context.body_end] +
                      cls.return_block + "\n}" + projected[context.body_end + 1:])
        # Independent exact inverse of the old wrapper/type/return. No private
        # baseline checkout or copied mathematical implementation is used.
        cls.legacy = cls.source
        for old, new in ((cls.graph_type, ""), (cls.thin_build, ""),
                         (cls.config_header, "pub fn build(b: *std.Build) void {"),
                         (cls.return_block, "")):
            if cls.legacy.count(old) != 1:
                raise AssertionError("reviewed configure fixture anchor changed: " + old[:65])
            cls.legacy = cls.legacy.replace(old, new, 1)

    def test_exact_seam_preserves_body_locations_and_complete_owner_graph(self) -> None:
        projected = CHECKER._reviewed_configure_projection(self.source)
        self.assertEqual(len(self.source), len(projected))
        self.assertEqual([i for i, ch in enumerate(self.source) if ch == "\n"],
                         [i for i, ch in enumerate(projected) if ch == "\n"])
        start = self.source.index(self.config_header) + len(self.config_header)
        end = self.source.index(self.return_block)
        self.assertEqual(self.source[start:end], projected[start:end])
        self.assertEqual(CHECKER._compact_zig_contract(self.legacy), CHECKER._compact_zig_contract(projected))
        self.assertEqual(projected, CHECKER._reviewed_configure_projection(projected))
        self.assertEqual(CHECKER._numerics_resources_build_contract(self.legacy),
                         CHECKER._numerics_resources_build_contract(self.source))
        self.assertEqual(CHECKER._runtime_lifecycle_build_span(self.source),
                         CHECKER._runtime_lifecycle_build_span(projected))
        context = CHECKER._zig_build_context(self.source, "build.zig")
        self.assertEqual("b", context.receiver)
        self.assertEqual(self.source.index(self.config_header) + len(self.config_header), context.body_start)
        self.assertEqual(end, _build_body_insertion_offset(self.source))
        self.assertEqual(CHECKER._zig_build_context(self.legacy, "build.zig").body_end,
                         _build_body_insertion_offset(self.legacy))

    def test_thin_build_rejects_extra_forwarded_conditional_or_missing_calls(self) -> None:
        for body in ("", "_ = configure(b); _ = configure(b);", "if (true) { _ = configure(b); }",
                     "const forward = configure; _ = forward(b);", "_ = another(b);",
                     "_ = configure(b); b.installArtifact(unreviewed);", "return configure(b);",
                     "_ = configure(other);"):
            mutated = self.source.replace(self.thin_build, "pub fn build(b: *std.Build) void { " + body + " }")
            with self.subTest(body=body), self.assertRaisesRegex(CHECKER.InventoryError, "reviewed configure seam changed"):
                CHECKER._reviewed_configure_projection(mutated)
        for suffix in ("\nconst escaped = configure;", "\nfn elsewhere(b: *std.Build) void { _ = configure(b); }",
                       "\nfn recursion(b: *std.Build) void { build(b); }", "\n" + self.thin_build):
            with self.subTest(suffix=suffix), self.assertRaisesRegex(CHECKER.InventoryError, "reviewed configure seam changed"):
                CHECKER._reviewed_configure_projection(self.source + suffix)

    def test_graph_type_signature_and_all_six_return_bindings_are_exact(self) -> None:
        for old, new in (("resources: *std.Build.Module,", "resources: *const std.Build.Module,"),
                         ("options: *std.Build.Module,", ""),
                         (self.config_header, "fn configure(b: *std.Build) ImplementationGraph {"),
                         (self.config_header, "pub fn configure(other: *std.Build) ImplementationGraph {"),
                         (self.graph_type, self.graph_type + "\nconst GraphAlias = ImplementationGraph;")):
            with self.subTest(mutation=old), self.assertRaisesRegex(CHECKER.InventoryError, "reviewed configure seam changed"):
                CHECKER._reviewed_configure_projection(self.source.replace(old, new, 1))
        for field, owner in CHECKER.CONFIGURE_MODULE_RETURNS:
            for replacement in ("test_module", "b.createModule(.{})", "undefined", owner + ", .extra = " + owner):
                with self.subTest(field=field, replacement=replacement), self.assertRaisesRegex(CHECKER.InventoryError, "final return"):
                    CHECKER._reviewed_configure_projection(self.source.replace(
                        f".{field} = {owner},", f".{field} = {replacement},", 1))

    def test_return_is_last_unconditional_and_each_owner_is_unshadowed_const(self) -> None:
        for replacement in ("if (true) { " + self.return_block + " }", self.return_block + " b.installArtifact(extra);",
                            self.return_block + "\n" + self.return_block, "return other;"):
            with self.subTest(replacement=replacement[:70]), self.assertRaisesRegex(CHECKER.InventoryError, "reviewed configure seam changed"):
                CHECKER._reviewed_configure_projection(self.source.replace(self.return_block, replacement, 1))
        for _, owner in CHECKER.CONFIGURE_MODULE_RETURNS:
            for old, new in ((f"const {owner} =", f"var {owner} ="),
                             (self.return_block, f"{{ const {owner} = undefined; }}\n    " + self.return_block)):
                with self.subTest(owner=owner, mutation=new[:65]), self.assertRaisesRegex(CHECKER.InventoryError, "production owner"):
                    CHECKER._reviewed_configure_projection(self.source.replace(old, new, 1))

    def test_same_source_owner_and_options_clones_hit_graph_guard(self) -> None:
        for path in ("src/zynum.zig", "src/numerics/execution/default.zig", "src/numerics/resources.zig"):
            statement = f'const extra = b.createModule(.{{ .root_source_file = b.path("{path}"), .target = target, .optimize = optimize }});\n    '
            changed = self.source.replace(self.return_block, statement + self.return_block, 1)
            with self.subTest(path=path), self.assertRaisesRegex(CHECKER.InventoryError, "unreviewed same-source owner"):
                CHECKER._numerics_resources_graph_facts(changed)
        changed = self.source.replace(self.return_block, "const extra = zynum_build_options.createModule();\n    " + self.return_block, 1)
        with self.assertRaisesRegex(CHECKER.InventoryError, "Options Module must have one owner"):
            CHECKER._numerics_resources_graph_facts(changed)

    def test_unprojected_body_keeps_receiver_and_owner_edge_rejections(self) -> None:
        changed = self.source.replace(self.return_block, "forward(b);\n    " + self.return_block, 1)
        with self.assertRaisesRegex(CHECKER.InventoryError, "escapes the analyzed build body"):
            CHECKER._zig_build_context(changed, "build.zig")
        edge = 'lapacke_binding_mod.addImport("zynum-core", zynum_core_mod);'
        with self.assertRaisesRegex(CHECKER.InventoryError, "shared core provider changed"):
            CHECKER._numerics_resources_graph_facts(self.source.replace(edge, edge.replace("zynum_core_mod", "zynum_core_test_mod"), 1))
        guard = "!uses_facade_owner and official_tests != numerics_contract_tests"
        self.assertIn(guard, self.source)
        with self.assertRaisesRegex(CHECKER.InventoryError, "owner/source/target/attachment graph drifted"):
            CHECKER._numerics_resources_build_contract(self.source.replace(guard, "true", 1))

    def test_call_digest_binds_raw_source_including_wrapper_trivia(self) -> None:
        # Use normal discovery on a finite source fixture to exercise raw-source
        # hashing; this is parser coverage, not a production graph/readiness proof.
        fields = CHECKER.CONFIGURE_MODULE_RETURNS
        body = "\n".join(f"    const {owner} = b.createModule(.{{}});" for _, owner in fields)
        source = self.graph_type + "\n" + self.thin_build + "\n" + self.config_header + "\n" + body + '\n    const gate = b.step("test", "fixture");\n    ' + self.return_block + "\n}\n"
        with tempfile.TemporaryDirectory(prefix="configure-digest-") as directory:
            root = Path(directory)
            path = root / "build.zig"
            path.write_text(source)
            observed, = CHECKER._discover_build_root(root, "build.zig")
            changed = source.replace("_ = configure(b);", "// reviewed trivia remains source-bound\n    _ = configure(b);")
            path.write_text(changed)
            updated, = CHECKER._discover_build_root(root, "build.zig")
            self.assertEqual(observed["id"], updated["id"])
            self.assertNotEqual(observed["source_digest"], updated["source_digest"])
            # Hashing the projection instead would produce a different digest.
            projected = CHECKER._reviewed_configure_projection(source)
            context = CHECKER._zig_build_context(projected, "build.zig")
            position, call = CHECKER._zig_member_calls(projected, context, "b", "step")[0]
            expected = CHECKER._call_semantics(projected, position, call, "b.step", "gate", None,
                                               build_context=context, source_text=source)
            self.assertEqual(expected, observed["source_digest"])
            self.assertNotEqual(expected, CHECKER._call_semantics(projected, position, call, "b.step", "gate", None, build_context=context))

    def test_stage7_classification_adds_only_the_two_explicit_paths(self) -> None:
        inventory = json.loads((REPOSITORY_ROOT / "tools/build_inventory.json").read_text())
        new_paths = ('src/interop/lapacke_access.zig', 'test/lapack/interop/raw_access_test.zig')
        inventory["repository_file_classifications"] = [
            row for row in inventory["repository_file_classifications"] if row["path"] not in new_paths
        ]
        before = {row["path"]: row for row in inventory["repository_file_classifications"]}
        CHECKER._apply_reviewed_build_inventory_migrations(_with_source_reviewed_execution_compile(inventory))
        after = {row["path"]: row for row in inventory["repository_file_classifications"]}
        new = {"src/interop/lapacke_access.zig": "library-source", "test/lapack/interop/raw_access_test.zig": "test-infrastructure"}
        self.assertEqual(set(new) - set(before), set(after) - set(before))
        self.assertEqual(before, {path: after[path] for path in before})
        for path, owner in new.items():
            self.assertEqual({"path": path, "kind": "zig-source", "class": "non-generated-source", "owner": owner}, after[path])


class PosvInventoryContractTests(unittest.TestCase):
    def test_exact_finite_outputs_sources_and_default_off_selection(self) -> None:
        inventory = json.loads((REPOSITORY_ROOT / "tools/build_inventory.json").read_text())
        self.assertEqual(11, len(CHECKER.POSV_FINITE_SETS))
        self.assertEqual(44, len(CHECKER.POSV_OUTPUTS))
        self.assertEqual(51, len(CHECKER.POSV_CLASSIFICATIONS))
        targets = {row["path"]: row for row in inventory["generator_targets"]}
        classifications = {row["path"]: row for row in inventory["repository_file_classifications"]}
        for path in CHECKER.POSV_OUTPUTS:
            self.assertEqual("tools/generate_checked_abi.py", targets[path]["anchor"]["file"])
            self.assertEqual("subset_outputs", targets[path]["anchor"]["enclosing_function"])
        for row in CHECKER.POSV_CLASSIFICATIONS:
            self.assertEqual(row, classifications[row["path"]])
        for path, expected in CHECKER.POSV_SOURCE_SHA256.items():
            self.assertEqual(expected, hashlib.sha256((REPOSITORY_ROOT / path).read_bytes()).hexdigest())
        self.assertEqual("false", CHECKER.POSV_OPTION_SURFACE["default"])
        self.assertEqual("bool", CHECKER.POSV_OPTION_SURFACE["type"])
        self.assertIsNone(CHECKER.NEXT_SOURCE_PROJECTION_SHA256)

    def test_posv_root_borrows_existing_binding_and_selector_mutations_fail_closed(self) -> None:
        source = (REPOSITORY_ROOT / "build.zig").read_text()
        facts = CHECKER._numerics_resources_build_contract(source)
        self.assertEqual("test/lapack/interop/posv_adapter_test.zig", facts["root_sources"]["lapacke_posv_tests"])
        self.assertEqual("lapacke_binding_test_mod", facts["imports"]["lapacke_posv_tests.root_module"]["numerical-owner"])
        self.assertEqual(CHECKER.POSV_SELECTION_CONTRACT_SHA256, facts["posv_selection_contract_sha256"])
        for changed in (
            source.replace('"lapacke-posv", "Include generated four-scalar POSV high/work in finite LP64/ILP64 and unified LP64 packages") orelse false',
                           '"lapacke-posv", "Include generated four-scalar POSV high/work in finite LP64/ILP64 and unified LP64 packages") orelse true', 1),
            source.replace('"{s}-posv-ordinary-lp64-v1"', '"{s}-unreviewed-posv-ordinary-lp64-v1"', 1),
        ):
            self.assertNotEqual(source, changed)
            with self.assertRaisesRegex(CHECKER.InventoryError, "selector|selection"):
                CHECKER._numerics_resources_graph_facts(changed)


class TrtrsInventoryContractTests(unittest.TestCase):
    def test_closed_outputs_preserve_existing_roots_and_finite_runtime_scope(self) -> None:
        inventory = json.loads((REPOSITORY_ROOT / "tools/build_inventory.json").read_text())
        self.assertEqual(24, len(CHECKER.TRTRS_FINITE_SETS))
        self.assertEqual(96, len(CHECKER.TRTRS_OUTPUTS))
        self.assertEqual(102, len(CHECKER.TRTRS_CLASSIFICATIONS))
        targets = {row["path"]: row for row in inventory["generator_targets"]}
        classifications = {row["path"]: row for row in inventory["repository_file_classifications"]}
        for path in CHECKER.TRTRS_OUTPUTS:
            self.assertEqual("tools/generate_checked_abi.py", targets[path]["anchor"]["file"])
            self.assertEqual("subset_outputs", targets[path]["anchor"]["enclosing_function"])
        for row in CHECKER.TRTRS_CLASSIFICATIONS:
            self.assertEqual(row, classifications[row["path"]])
        for path, expected in CHECKER.TRTRS_SOURCE_SHA256.items():
            self.assertEqual(expected, hashlib.sha256((REPOSITORY_ROOT / path).read_bytes()).hexdigest())
        self.assertEqual("false", CHECKER.TRTRS_OPTION_SURFACE["default"])
        source = (REPOSITORY_ROOT / "build.zig").read_text()
        facts = CHECKER._numerics_resources_build_contract(source)
        self.assertEqual("lapacke_binding_test_mod", facts["imports"]["lapack_tests.root_module"]["numerical-owner"])
        self.assertEqual(128, len(facts["module_source_variants"]["lapacke_lp64_interface"]))
        observations = {row["id"]: row for row in inventory["build_observations"]}
        for width, count in (("lp64", 128), ("ilp64", 16)):
            row = observations["compile:build.zig:build:lapacke_" + width + "_static"]
            row = _prior_component_observation(self, row, width, count)
            self.assertEqual(count + 1, len(row["source_variants"]))
            self.assertEqual("src/lapacke_composed.zig", row["source_variants"]["lapacke-potri=true"])
            base_variants = {key.removeprefix("lapacke-potri=false,"): value
                             for key, value in row["source_variants"].items()
                             if key.startswith("lapacke-potri=false,")}
            self.assertEqual(count, len(base_variants))
            row = {**row, "source_variants": base_variants,
                   "root_source": [path for path in row["root_source"] if path != "src/lapacke_composed.zig"]}
            self.assertEqual(sorted(row["root_source"]), sorted(row["source_variants"].values()))
            self.assertTrue(all(",lapacke-trtrs=" in key for key in row["source_variants"]))
        self.assertNotIn("lapacke_trtrs_tests", facts["root_owners"])
        self.assertIsNone(CHECKER.NEXT_SOURCE_PROJECTION_SHA256)

    def test_default_type_name_and_both_width_compositions_reject_mutation(self) -> None:
        source = (REPOSITORY_ROOT / "build.zig").read_text()
        statement = '    const lapacke_trtrs = b.option(bool, "lapacke-trtrs", "Include generated four-scalar triangular solve high/work in finite LP64/ILP64 and unified LP64 packages") orelse false;'
        self.assertEqual(1, source.count(statement))
        changes = [source.replace(statement, statement.replace("orelse false", "orelse true"), 1),
                   source.replace(statement, statement.replace("b.option(bool", "b.option(usize"), 1),
                   source.replace(statement, statement.replace('"lapacke-trtrs"', '"unreviewed-trtrs"'), 1)]
        for width in ("lp64", "ilp64"):
            before = '"{s}-trtrs-ordinary-' + width + '-v1"'
            self.assertEqual(1, source.count(before))
            changes.append(source.replace(before, before.replace("-trtrs-", "-unreviewed-trtrs-"), 1))
        for changed in changes:
            self.assertNotEqual(source, changed)
            with self.assertRaisesRegex(CHECKER.InventoryError, "selector|selection"):
                CHECKER._numerics_resources_graph_facts(changed)


class CheckedCholeskyInventoryContractTests(unittest.TestCase):
    def test_only_thirteen_explicit_frontend_files_are_added(self) -> None:
        inventory = json.loads((REPOSITORY_ROOT / "tools/build_inventory.json").read_text())
        paths = {row["path"] for row in CHECKER.CHECKED_CHOLESKY_CLASSIFICATIONS}
        self.assertEqual(13, len(paths))
        inventory["repository_file_classifications"] = [row for row in inventory["repository_file_classifications"] if row["path"] not in paths]
        before = {row["path"]: row for row in inventory["repository_file_classifications"]}
        CHECKER._apply_reviewed_build_inventory_migrations(_with_source_reviewed_execution_compile(inventory))
        after = {row["path"]: row for row in inventory["repository_file_classifications"]}
        self.assertEqual(paths, set(after) - set(before))
        self.assertEqual(before, {path: after[path] for path in before})
        for row in CHECKER.CHECKED_CHOLESKY_CLASSIFICATIONS:
            self.assertEqual(row, after[row["path"]])

    def test_generated_shared_wire_and_optional_install_contract_are_explicit(self) -> None:
        outputs = CHECKER.CHECKED_CHOLESKY_OUTPUTS
        self.assertEqual(5, len(outputs))
        self.assertEqual(1, outputs.count("include/zynum/checked/complex.h"))
        for path, expected in CHECKER.CHECKED_CHOLESKY_SOURCE_SHA256.items():
            self.assertEqual(expected, hashlib.sha256((REPOSITORY_ROOT/path).read_bytes()).hexdigest())
        shared = next(row for row in CHECKER.CHECKED_CHOLESKY_DERIVED_CANDIDATES if row["path"].endswith("complex.h"))
        self.assertIn("tools/generate_checked_sparse.py", shared["authoritative_inputs"])
        self.assertIn("tools/checked_complex_wire.py", shared["authoritative_inputs"])
        self.assertEqual("false", CHECKER.CHECKED_CHOLESKY_OPTION_SURFACE["default"])
        self.assertEqual("bool", CHECKER.CHECKED_CHOLESKY_OPTION_SURFACE["type"])
        self.assertIs(True, CHECKER.CHECKED_CHOLESKY_OBSERVATION_TEMPLATES["compile:build.zig:build:csp_static"]["bundle_compiler_rt"])
        self.assertEqual(1, (REPOSITORY_ROOT/"build.zig").read_text().count("csp_static.bundle_compiler_rt = true;"))
        for name in ("complex.h", "cholesky.h", "cholesky_manifest.json"):
            row = CHECKER.CHECKED_CHOLESKY_OBSERVATION_TEMPLATES["install:build.zig:build:include/zynum/checked/"+name]
            self.assertEqual(name != "complex.h", "checked-complex-cholesky=true" in row["condition"])

    def test_pkgconfig_cholesky_marker_remains_source_bound(self) -> None:
        source = (REPOSITORY_ROOT/"build.zig").read_text()
        CHECKER._operator_phase_agreement_build_contract(source)
        marker = 'if (checked_complex_cholesky) "checked_complex_cholesky=true\\n"'
        self.assertIn(marker, source)
        changed = source.replace(marker, 'if (true) "checked_complex_cholesky=true\\n"', 1)
        with self.assertRaisesRegex(CHECKER.InventoryError, "package metadata selection changed"):
            CHECKER._operator_phase_agreement_build_contract(changed)


class NumericsResourcesSourceContractTests(unittest.TestCase):
    """Bounded real-source graph checks; no compiler or inventory publication."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.source = (REPOSITORY_ROOT / "build.zig").read_text(encoding="utf-8")
        cls.resource = (REPOSITORY_ROOT / CHECKER.NUMERICS_RESOURCES_PATH).read_bytes()

    def test_resources_bind_all_owners_host_and_canonical_roots(self) -> None:
        facts = CHECKER._numerics_resources_build_contract(self.source)
        self.assertEqual(11, len(facts["edges"]))
        self.assertEqual(42, len(facts["core_edges"]))
        self.assertEqual(129, len(facts["declarations"]))
        self.assertEqual(28, len(facts["root_sources"]))
        self.assertEqual("src/blas/triangular_band_solve_test.zig", facts["root_sources"]["triangular_band_solve_tests"])
        host = next(edge for edge in facts["edges"] if edge["owner"] == "kernel_coverage_mod")
        self.assertEqual("host_numerics_resources_mod", host["provider"])
        self.assertTrue(all(edge["provider"] == "numerics_resources_mod" for edge in facts["edges"] if edge != host and edge["owner"] != "runtime_lifecycle_execution_mod"))
        CHECKER._numerics_resources_build_contract(self.source.replace(
            "const numerics_resources_mod", "// harmless trivia\n    const numerics_resources_mod", 1))

    def test_lp64_gesvd_selector_preserves_eight_sources_and_one_owner(self) -> None:
        facts = CHECKER._numerics_resources_build_contract(self.source)
        prefix = "interface_subset/mkl2026.lp64/"
        base = "potrf-potrs-getrf-getrs-gesv-stev-gelsd-heev-hegv"
        qr = "-geqrf-ormqr-unmqr-orgqr-ungqr"
        tail = "-ordinary-lp64-v1/exports.zig"
        expected = {
            "base": prefix + base + tail,
            "qr": prefix + base + qr + tail,
            "complex-gelsd-base": prefix + base + "-complex-gelsd" + tail,
            "complex-gelsd-qr": prefix + base + qr + "-complex-gelsd" + tail,
            "gesvd-base": prefix + base + "-gesvd" + tail,
            "gesvd-qr": prefix + base + qr + "-gesvd" + tail,
            "complex-gelsd-gesvd-base": prefix + base + "-complex-gelsd-gesvd" + tail,
            "complex-gelsd-gesvd-qr": prefix + base + qr + "-complex-gelsd-gesvd" + tail,
        }
        expected.update({
            key + "-posv": path.replace("-ordinary-lp64-v1/", "-posv-ordinary-lp64-v1/")
            for key, path in tuple(expected.items())
        })
        self.assertEqual(16, len(expected))
        expected.update({
            key + "-trtrs": path.replace("-ordinary-lp64-v1/", "-trtrs-ordinary-lp64-v1/")
            for key, path in tuple(expected.items())
        })
        self.assertEqual(32, len(expected))
        expected.update({
            key + "-trtri": path.replace("-ordinary-lp64-v1/", "-trtri-ordinary-lp64-v1/")
            for key, path in tuple(expected.items())
        })
        self.assertEqual(64, len(expected))
        expected.update({
            key + "-getri": path.replace("-ordinary-lp64-v1/", "-getri-ordinary-lp64-v1/")
            for key, path in tuple(expected.items())
        })
        self.assertEqual(128, len(expected))
        for owner in ("lapacke_lp64_interface", "abplas_lp64_lapacke"):
            self.assertEqual(expected, facts["module_source_variants"][owner])
            self.assertEqual("lapacke_binding_mod", facts["imports"][owner]["lapacke-interface-owner"])
        self.assertEqual(facts["module_source_variants"]["lapacke_lp64_interface"],
                         facts["module_source_variants"]["abplas_lp64_lapacke"])

    def test_lp64_optional_selector_mutations_fail_closed(self) -> None:
        for symbol, flag in (("lapacke_qr", "lapacke-qr"),
                             ("lapacke_complex_gelsd", "lapacke-complex-gelsd"),
                             ("lapacke_svd", "lapacke-svd"),
                             ("lapacke_posv", "lapacke-posv")):
            start = self.source.index("const " + symbol + " =")
            end = self.source.index(";", start) + 1
            declaration = self.source[start:end]
            self.assertIn("orelse false;", declaration)
            self.assertIn('b.option(bool, "' + flag + '"', declaration)
            mutations = (declaration.replace("orelse false;", "orelse true;"),
                         declaration.replace("b.option(bool,", "b.option(u32,"),
                         declaration.replace('"' + flag + '"', '"unreviewed-selector"'),
                         declaration + "\n" + declaration,
                         "")
            for mutation in mutations:
                with self.subTest(selector=symbol, mutation=mutation):
                    with self.assertRaises(CHECKER.InventoryError):
                        CHECKER._numerics_resources_build_contract(
                            self.source[:start] + mutation + self.source[end:])
        before = 'abplas_lp64_lapacke.addImport("lapacke-interface-owner", lapacke_binding_mod);'
        self.assertEqual(1, self.source.count(before))
        with self.assertRaises(CHECKER.InventoryError):
            CHECKER._numerics_resources_build_contract(self.source.replace(
                before, before.replace("lapacke_binding_mod", "lapacke_services_mod"), 1))
        selected = "-complex-gelsd-gesvd-ordinary-lp64-v1"
        self.assertIn(selected, self.source)
        with self.assertRaises(CHECKER.InventoryError):
            CHECKER._numerics_resources_build_contract(self.source.replace(
                selected, "-complex-gelsd-ordinary-lp64-v1", 1))

    def test_optional_operator_core_checked_owner_and_generated_source_fail_closed(self) -> None:
        replacements = (
            ('operator_exports_mod.addImport("zynum-core", zynum_core_mod)',
             'operator_exports_mod.addImport("zynum-core", zynum_core_test_mod)'),
            ('operator_combined_mod.addImport("checked-interface-owner", csp)',
             'operator_combined_mod.addImport("checked-interface-owner", checked_abi_mod)'),
            ('operator_generate.addOutputFileArg("operator_v1.zig")',
             'operator_generate.addOutputFileArg("unreviewed.zig")'),
        )
        for before, after in replacements:
            with self.subTest(mutation=before):
                self.assertEqual(1, self.source.count(before))
                with self.assertRaises(CHECKER.InventoryError):
                    CHECKER._numerics_resources_build_contract(self.source.replace(before, after, 1))

    def test_resources_missing_swapped_duplicate_and_extra_edges_fail(self) -> None:
        for owner in CHECKER.NUMERICS_RESOURCES_OWNERS:
            provider = {
                "kernel_coverage_mod": "host_numerics_resources_mod",
                "runtime_lifecycle_execution_mod": "runtime_lifecycle_resources_mod",
            }.get(owner, "numerics_resources_mod")
            edge = f'{owner}.addImport("zynum-numerics-resources", {provider});'
            self.assertEqual(1, len(CHECKER.re.findall(r"(?<![A-Za-z0-9_])" + CHECKER.re.escape(edge), self.source)))
            for change in ("", edge.replace(provider, "unreviewed_resource_mod"), edge + "\n" + edge):
                with self.subTest(owner=owner, mutation=change):
                    with self.assertRaisesRegex(CHECKER.InventoryError, "numerics resources"):
                        CHECKER._numerics_resources_build_contract(self.source.replace(edge, change, 1))
        for edge in (
            'abplas_lp64_root.addImport("zynum-core", zynum_core_mod);',
            'abplas_lp64_root.addImport("lapacke-interface-owner", lapacke_binding_mod);',
            'abplas_lp64_root.addImport("zynum-lapacke-exports", abplas_lp64_lapacke_controls_selected);',
            'abplas_lp64_root.addImport("zynum-checked-sparse-exports", csp);',
            'abplas_lp64_lapacke.addImport("lapacke-interface-owner", lapacke_binding_mod);',
        ):
            self.assertEqual(1, self.source.count(edge))
            for change in ("", edge + "\n" + edge, edge.replace(edge.split(", ")[1][:-2], "unreviewed_owner")):
                with self.subTest(combined_edge=edge, mutation=change):
                    with self.assertRaises(CHECKER.InventoryError):
                        CHECKER._numerics_resources_build_contract(self.source.replace(edge, change, 1))
        with self.assertRaisesRegex(CHECKER.InventoryError, "numerics resources"):
            CHECKER._numerics_resources_build_contract(self.source + '\nconst another = numerics_resources_mod;\n')

    def test_resources_owner_source_target_scope_and_host_identity_fail(self) -> None:
        substitutions = (
            ('b.path("src/numerics/resources.zig")', 'b.path("src/numerics/storage.zig")'),
            ('.target = b.graph.host,', '.target = target,'),
            ('const host_numerics_resources_mod = if (resources_host_matches_target)', 'const host_numerics_resources_mod = if (true)'),
            ('target.result.ofmt == b.graph.host.result.ofmt', 'true'),
            ('target.result.dynamic_linker.eql(b.graph.host.result.dynamic_linker)', 'true'),
            ('const official_tests = inventory_case.logical_tests orelse continue;', 'const official_tests = dynamic_tests;'),
            ('zynum_core_test_mod.addImport("zynum-numerics-resources", numerics_resources_mod);', 'if (false) zynum_core_test_mod.addImport("zynum-numerics-resources", numerics_resources_mod);'),
        )
        for before, after in substitutions:
            with self.subTest(mutation=before):
                self.assertIn(before, self.source)
                with self.assertRaisesRegex(CHECKER.InventoryError, "numerics resources"):
                    CHECKER._numerics_resources_build_contract(self.source.replace(before, after, 1))
        for name in ("numerics_resources_mod", "zynum_test_mod", "kernel_coverage_mod", "triangular_band_solve_test_mod", "dynamic_tests"):
            start = self.source.index(f"const {name} =")
            tail = self.source[start:]
            before = ".target = b.graph.host," if name == "kernel_coverage_mod" else ".target = target,"
            self.assertIn(before, tail)
            changed = self.source[:start] + tail.replace(before, ".target = unreviewed_target,", 1)
            with self.subTest(target_owner=name):
                with self.assertRaisesRegex(CHECKER.InventoryError, "numerics resources"):
                    CHECKER._numerics_resources_build_contract(changed)

    def test_resources_frozen_bytes_missing_and_drift_fail(self) -> None:
        with tempfile.TemporaryDirectory(prefix="resources-source-contract-") as directory:
            root = Path(directory)
            (root / "build.zig").write_text("pub fn build() void {}\n")
            source = root / CHECKER.NUMERICS_RESOURCES_PATH
            source.parent.mkdir(parents=True)
            source.write_bytes(self.resource)
            for path in (*CHECKER.EVD_SOURCE_SHA256, *CHECKER.CONTROLS_SOURCE_SHA256, *CHECKER.SYGV_SOURCE_SHA256, *CHECKER.SYEV_SOURCE_SHA256, *CHECKER.CONTOUR_SOURCE_SHA256, *CHECKER.FEAST_SOURCE_SHA256, *CHECKER.FFTW_F32_SOURCE_SHA256, *CHECKER.POTRI_OWNER_SOURCE_SHA256, *CHECKER.SHARED_FACADE_SOURCE_SHA256, *CHECKER.EXECUTION_SOURCE_SHA256, *CHECKER.EXECUTION_OBJECT_SOURCE_SHA256, *CHECKER.COMMON_CORE_SOURCE_SHA256, *CHECKER.MANAGED_LAPACKE_SOURCE_SHA256, *CHECKER.MANAGED_SPARSE_SOURCE_SHA256, *CHECKER.UNIFIED_BLAS_SOURCE_SHA256, *CHECKER.OPERATOR_PHASE_SOURCE_SHA256, *CHECKER.GVD_SOURCE_SHA256, *CHECKER.GELS_SOURCE_SHA256, *CHECKER.GELSS_SOURCE_SHA256, *CHECKER.INJECTION_SOURCE_SHA256):
                destination = root / path
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes((REPOSITORY_ROOT / path).read_bytes())
            context = CHECKER._make_discovery_context(root)
            CHECKER._numerics_resources_source_contract(context)
            source.write_bytes(self.resource + b"\n// drift\n")
            CHECKER._numerics_resources_source_contract(context)
            with self.assertRaisesRegex(CHECKER.InventoryError, "source bytes drifted"):
                CHECKER._numerics_resources_source_contract(CHECKER._make_discovery_context(root))
            source.unlink()
            with self.assertRaises((CHECKER.InventoryError, CHECKER.repository_snapshot.RepositorySnapshotError)):
                CHECKER._numerics_resources_source_contract(CHECKER._make_discovery_context(root))

    def test_resources_classification_uses_reviewed_source_migration(self) -> None:
        inventory = json.loads((REPOSITORY_ROOT / "tools/build_inventory.json").read_text())
        CHECKER._apply_reviewed_build_inventory_migrations(_with_source_reviewed_execution_compile(inventory))
        row = next(x for x in inventory["repository_file_classifications"] if x["path"] == CHECKER.NUMERICS_RESOURCES_PATH)
        self.assertEqual({"path": CHECKER.NUMERICS_RESOURCES_PATH, "kind": "zig-source",
                          "class": "non-generated-source", "owner": "library-source"}, row)

    def test_resources_attachment_keeps_existing_blas_object_exclusions(self) -> None:
        observations = CHECKER._discover_build_root(REPOSITORY_ROOT, "build.zig")
        CHECKER._annotate_test_inventory_factory(self.source, copy.deepcopy(observations))
        changes = (
            ("!uses_facade_owner and official_tests != numerics_contract_tests", "true"),
            ("if (compact_triangular_isolated_test_library) |library| official_tests.root_module.addObject(library);", ""),
            ('official_tests.root_module.addImport("zynum-numerics-resources", numerics_resources_mod);', ""),
        )
        for before, after in changes:
            with self.subTest(mutation=before):
                with self.assertRaisesRegex(CHECKER.InventoryError, "BLAS object attachments"):
                    CHECKER._annotate_test_inventory_factory(self.source.replace(before, after, 1), copy.deepcopy(observations))


class SharedFacadeSourceContractTests(unittest.TestCase):
    """Exact module owners, not a generic Build.Module resolver."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.source = (REPOSITORY_ROOT / "build.zig").read_text()
        cls.graph = CHECKER._numerics_resources_build_contract(cls.source)

    def test_production_test_core_and_all_facades_are_distinct_owned_nodes(self) -> None:
        self.assertEqual("src/zynum.zig", self.graph["module_sources"]["zynum_core_mod"])
        self.assertEqual("src/zynum.zig", self.graph["module_sources"]["zynum_core_test_mod"])
        for owner, provider in CHECKER.SHARED_FACADE_OWNERS.items():
            self.assertEqual({"zynum-core": provider}, self.graph["imports"][owner])
        root = self.graph["root_owners"]["modern_tests"]
        self.assertEqual({"zynum": "zynum_test_mod", "zynum-blas": "zynum_blas_test_mod",
                          "zynum-lapack": "zynum_lapack_test_mod", "zynum-sparse": "zynum_sparse_test_mod",
                          "zynum-numerics-resources": "numerics_resources_mod"}, self.graph["imports"][root])
        self.assertNotIn("zynum-build-options", self.graph["generated_imports"].get(root, ()))

    def test_potri_components_borrow_width_correct_bases_and_existing_core(self) -> None:
        choices = self.graph["module_borrow_variants"]
        self.assertEqual({"lapacke_lp64_selected", "lapacke_ilp64_selected", "abplas_lp64_lapacke_selected",
                          "lapacke_lp64_before_syev", "lapacke_lp64_syev_selected",
                          "abplas_lp64_lapacke_before_syev", "abplas_lp64_lapacke_syev_selected",
                          "operator_binding_owner.?"}, set(choices))
        self.assertEqual({"contour-interface=zynum_v1": "operator_binding_mod"}, choices["operator_binding_owner.?"])
        for prefix, previous in (("lapacke_lp64", "lapacke_lp64_selected"), ("abplas_lp64_lapacke", "abplas_lp64_lapacke_selected")):
            self.assertEqual({"post-POTRI": previous}, choices[prefix + "_before_syev"])
            self.assertEqual({"lapacke-syev=false": prefix + "_before_syev", "lapacke-syev=true": prefix + "_syev_composed_root"}, choices[prefix + "_syev_selected"])
            self.assertEqual("src/lapacke_symmetric_composed.zig", self.graph["module_sources"][prefix + "_syev_composed_root"])
            self.assertEqual("zynum_core_mod", self.graph["imports"][prefix + "_syev_composed_root"]["zynum-core"])
            self.assertEqual({"lapacke-interface-owner": "lapacke_binding_mod"}, self.graph["imports"][prefix + "_syev_component"])
        for selected, base, composed, component, width in (
            ("lapacke_lp64_selected", "lapacke_lp64_interface", "lapacke_lp64_composed_root", "lapacke_lp64_potri_component", "lp64"),
            ("lapacke_ilp64_selected", "lapacke_ilp64_interface", "lapacke_ilp64_composed_root", "lapacke_ilp64_potri_component", "ilp64"),
            ("abplas_lp64_lapacke_selected", "abplas_lp64_lapacke", "abplas_lapacke_composed_root", "abplas_lapacke_potri_component", "lp64"),
        ):
            with self.subTest(selected=selected):
                self.assertEqual({"lapacke-potri=false": base, "lapacke-potri=true": composed}, choices[selected])
                self.assertEqual("src/lapacke_composed.zig", self.graph["module_sources"][composed])
                self.assertEqual(f"interface_components/mkl2026.{width}/potri-v1/exports.zig", self.graph["module_sources"][component])
                self.assertEqual({"zynum-core": "zynum_core_mod", "lapacke-interface-owner": "lapacke_binding_mod",
                                  "zynum-lapacke-base-exports": base, "zynum-lapacke-component-exports": component}, self.graph["imports"][composed])
                self.assertEqual({"lapacke-interface-owner": "lapacke_binding_mod"}, self.graph["imports"][component])
        self.assertEqual(16, len(self.graph["module_source_variants"]["lapacke_ilp64_interface"]))
        self.assertEqual(15, len(self.graph["module_borrow_edges"]))
        self.assertNotIn("lapacke_lp64_composed_root", self.graph["root_owners"].values())

    def test_potri_default_core_binding_and_borrow_consumers_fail_closed(self) -> None:
        mutations = (
            ('"Compose generated four-scalar Cholesky inverse high/work over the same LAPACKE owner") orelse false',
             '"Compose generated four-scalar Cholesky inverse high/work over the same LAPACKE owner") orelse true'),
            ('lapacke_lp64_composed_root.addImport("zynum-core", zynum_core_mod);',
             'lapacke_lp64_composed_root.addImport("zynum-core", zynum_core_test_mod);'),
            ('lapacke_ilp64_composed_root.addImport("lapacke-interface-owner", lapacke_binding_mod);',
             'lapacke_ilp64_composed_root.addImport("lapacke-interface-owner", lapacke_binding_test_mod);'),
            ('abplas_lp64_lapacke_selected = abplas_lapacke_composed_root;',
             'abplas_lp64_lapacke_selected = lapacke_lp64_composed_root;'),
            ('lapacke_lp64_selected = lapacke_lp64_composed_root;',
             'lapacke_lp64_selected = lapacke_ilp64_composed_root;'),
            ('.name = "zynum_lapacke_ilp64", .linkage = .static, .root_module = lapacke_ilp64_selected',
             '.name = "zynum_lapacke_ilp64", .linkage = .static, .root_module = lapacke_lp64_selected'),
        )
        for before, after in mutations:
            with self.subTest(mutation=before):
                self.assertEqual(1, self.source.count(before))
                with self.assertRaises(CHECKER.InventoryError):
                    CHECKER._numerics_resources_build_contract(self.source.replace(before, after, 1))
        for escaped in ("lapacke_lp64_selected", "lapacke_ilp64_selected", "abplas_lp64_lapacke_selected"):
            with self.subTest(escaped=escaped), self.assertRaises(CHECKER.InventoryError):
                CHECKER._numerics_resources_build_contract(self.source + "\nconst escaped_potri_owner = " + escaped + ";\n")

    def test_potri_borrow_metadata_cannot_authorize_wrong_existing_modules(self) -> None:
        for selected in CHECKER.POTRI_MODULE_BORROWS:
            bad = copy.deepcopy(CHECKER.POTRI_MODULE_BORROWS)
            bad[selected]["lapacke-potri=true"] = "zynum_core_mod"
            with self.subTest(selected=selected), mock.patch.object(CHECKER, "POTRI_MODULE_BORROWS", bad):
                with self.assertRaises(CHECKER.InventoryError):
                    CHECKER._numerics_resources_build_contract(self.source)

    def test_missing_swapped_duplicate_and_extra_core_edges_fail(self) -> None:
        for owner, provider in CHECKER.SHARED_FACADE_OWNERS.items():
            edge = f'{owner}.addImport("zynum-core", {provider});'
            wrong = "zynum_core_mod" if provider.endswith("_test_mod") else "zynum_core_test_mod"
            for replacement in ("", edge.replace(provider, wrong), edge + "\n" + edge):
                with self.subTest(owner=owner, replacement=replacement):
                    with self.assertRaisesRegex(CHECKER.InventoryError, "numerics resources"):
                        CHECKER._numerics_resources_build_contract(self.source.replace(edge, replacement, 1))
        for extra in ('const copied_core = zynum_core_mod;', 'unreviewed.addImport("zynum-core", zynum_core_test_mod);'):
            with self.subTest(extra=extra), self.assertRaises(CHECKER.InventoryError):
                CHECKER._numerics_resources_build_contract(self.source + "\n" + extra)

    def test_core_mode_target_and_facade_options_objects_are_bound(self) -> None:
        for name in ("zynum_core_mod", "zynum_core_test_mod"):
            offset = self.source.index(f"const {name} =")
            for old, new in ((".target = target", ".target = b.graph.host"),
                             (".optimize = " + ("test_optimize" if "test" in name else "optimize"), ".optimize = .Debug")):
                text = self.source[:offset] + self.source[offset:].replace(old, new, 1)
                with self.subTest(owner=name, mutation=old), self.assertRaises(CHECKER.InventoryError):
                    CHECKER._numerics_resources_build_contract(text)
        for extra in ('zynum_test_mod.addOptions("zynum-build-options", zynum_build_options);',
                      'zynum_mod.addObject(library);', 'zynum_core_test_mod.addObject(library);'):
            with self.subTest(extra=extra), self.assertRaises(CHECKER.InventoryError):
                CHECKER._numerics_resources_build_contract(self.source + "\n" + extra)

    def test_modern_inline_dependency_and_official_option_exclusion_are_bound(self) -> None:
        changes = (
            ('.{ .name = "zynum-sparse", .module = zynum_sparse_test_mod },', ''),
            ('.{ .name = "zynum-lapack", .module = zynum_lapack_test_mod },', '.{ .name = "zynum-lapack", .module = zynum_lapack_mod },'),
            ('if (official_tests != modern_tests and official_tests != execution_contract_tests)', 'if (true)'),
            ('official_tests == fortran_tests or', ''),
        )
        for old, new in changes:
            self.assertIn(old, self.source)
            with self.subTest(mutation=old), self.assertRaises(CHECKER.InventoryError):
                CHECKER._numerics_resources_build_contract(self.source.replace(old, new, 1))

    def test_facade_source_bytes_and_classification_are_explicit(self) -> None:
        inventory = json.loads((REPOSITORY_ROOT / "tools/build_inventory.json").read_text())
        CHECKER._apply_reviewed_build_inventory_migrations(_with_source_reviewed_execution_compile(inventory))
        rows = {row["path"]: row for row in inventory["repository_file_classifications"]}
        with tempfile.TemporaryDirectory(prefix="shared-facade-source-") as directory:
            root = Path(directory)
            (root / "build.zig").write_text("pub fn build() void {}\n")
            for path in (CHECKER.NUMERICS_RESOURCES_PATH, *CHECKER.EVD_SOURCE_SHA256, *CHECKER.CONTROLS_SOURCE_SHA256, *CHECKER.SYGV_SOURCE_SHA256, *CHECKER.SYEV_SOURCE_SHA256, *CHECKER.CONTOUR_SOURCE_SHA256, *CHECKER.FEAST_SOURCE_SHA256, *CHECKER.FFTW_F32_SOURCE_SHA256, *CHECKER.POTRI_OWNER_SOURCE_SHA256, *CHECKER.SHARED_FACADE_SOURCE_SHA256, *CHECKER.EXECUTION_SOURCE_SHA256, *CHECKER.EXECUTION_OBJECT_SOURCE_SHA256, *CHECKER.COMMON_CORE_SOURCE_SHA256, *CHECKER.MANAGED_LAPACKE_SOURCE_SHA256, *CHECKER.MANAGED_SPARSE_SOURCE_SHA256, *CHECKER.UNIFIED_BLAS_SOURCE_SHA256, *CHECKER.OPERATOR_PHASE_SOURCE_SHA256, *CHECKER.GVD_SOURCE_SHA256, *CHECKER.GELS_SOURCE_SHA256, *CHECKER.GELSS_SOURCE_SHA256, *CHECKER.INJECTION_SOURCE_SHA256):
                dst = root / path
                dst.parent.mkdir(parents=True, exist_ok=True)
                dst.write_bytes((REPOSITORY_ROOT / path).read_bytes())
            CHECKER._numerics_resources_source_contract(CHECKER._make_discovery_context(root))
            for path in CHECKER.SHARED_FACADE_SOURCE_SHA256:
                self.assertEqual({"path": path, "kind": "zig-source", "class": "non-generated-source", "owner": "library-source"}, rows[path])
                original = (root / path).read_bytes()
                for replacement in (None, original + b"\nvar extra_state: u8 = 0;\n"):
                    if replacement is None:
                        (root / path).unlink()
                    else:
                        (root / path).write_bytes(replacement)
                    with self.subTest(path=path, missing=replacement is None), self.assertRaises((CHECKER.InventoryError, CHECKER.repository_snapshot.RepositorySnapshotError)):
                        CHECKER._numerics_resources_source_contract(CHECKER._make_discovery_context(root))
                (root / path).write_bytes(original)

    def _with_module_statement(self, statement: str) -> str:
        # Insert into the real receiver's lexical scope, not an invalid global
        # expression. The official receiver exists only in its canonical loop.
        if "official_tests.root_module" in statement:
            marker = "const official_tests = inventory_case.logical_tests orelse continue;"
            self.assertEqual(3, self.source.count(marker))
            # The first binding is the canonical attachment loop; the other two
            # belong to the enumerator and native-feature launch loops.
            return self.source.replace(marker, marker + "\n        " + statement, 1)
        body_end = _build_body_insertion_offset(self.source)
        return self.source[:body_end] + "    " + statement + "\n" + self.source[body_end:]

    def test_canonical_named_import_overwrite_is_rejected(self) -> None:
        for statement in ('modern_tests.root_module.addImport("zynum", zynum_mod);',
                          'lapack_tests.root_module.addImport("zynum-lapack", zynum_lapack_mod);',
                          'official_tests.root_module.addImport("zynum", zynum_mod);'):
            with self.subTest(statement=statement), self.assertRaises(CHECKER.InventoryError):
                CHECKER._numerics_resources_build_contract(self._with_module_statement(statement))

    def test_canonical_module_property_table_and_alias_escape_are_rejected(self) -> None:
        statements = (
            'modern_tests.root_module.root_source_file = b.path("src/zynum.zig");',
            'modern_tests.root_module.import_table.put(b.allocator, "zynum", zynum_mod) catch unreachable;',
            'const borrowed_root = modern_tests.root_module; borrowed_root.addImport("zynum", zynum_mod);',
            'const borrowed_internal = triangular_band_solve_test_mod; _ = borrowed_internal;',
            'const borrowed_official = official_tests.root_module; borrowed_official.addImport("zynum", zynum_mod);',
            'header_smoke_mod.root_source_file = b.path("src/zynum.zig");',
            'zynum_test_mod.import_table.put(b.allocator, "zynum-core", unreviewed_mod) catch unreachable;',
            'zynum_mod.root_source_file = b.path("src/zynum.zig");',
            'const borrowed_facade = zynum_blas_mod; borrowed_facade.root_source_file = b.path("src/zynum.zig");',
        )
        for statement in statements:
            with self.subTest(statement=statement), self.assertRaisesRegex(CHECKER.InventoryError, "canonical module receiver escapes"):
                CHECKER._numerics_resources_build_contract(self._with_module_statement(statement))

    def test_new_link_templates_bind_core_owner_and_mode(self) -> None:
        for owner, providers in (("zynum_core_mod", CHECKER.SHARED_CORE_PROVIDERS),
                                  ("zynum_core_test_mod", CHECKER.SHARED_CORE_TEST_PROVIDERS)):
            for provider in providers:
                row = CHECKER._new_test_inventory_observation(f"link:build.zig:build:{owner}<-{provider}", {})
                self.assertEqual(owner, row["consumer"])
                self.assertEqual(provider, row["provider"])
                self.assertIn("dynamic_dispatch" if provider.startswith("kernel_") else "x86_64", row["condition"])
        for owner in CHECKER.SHARED_FACADE_OWNERS:
            with self.subTest(owner=owner), self.assertRaises(CHECKER.InventoryError):
                CHECKER._new_test_inventory_observation(f"link:build.zig:build:{owner}<-kernel_libraries", {})


class CommonCoreContextSourceContractTests(unittest.TestCase):
    """411 source-only owner graph; no profile branch or native execution claims."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.source = (REPOSITORY_ROOT / "build.zig").read_text()
        cls.graph = CHECKER._numerics_resources_build_contract(cls.source)

    def reject(self, old: str, new: str, message: str = "(numerics resources|common Core|shared execution)") -> None:
        self.assertIn(old, self.source)
        self.assertNotEqual(old, new)
        with self.assertRaisesRegex(CHECKER.InventoryError, message):
            CHECKER._numerics_resources_build_contract(self.source.replace(old, new, 1))

    def test_core_abi_checked_context_and_combined_owners_are_exact(self) -> None:
        for owner, core in CHECKER.SHARED_ADAPTER_CORE_OWNERS.items():
            self.assertEqual(core, self.graph["imports"][owner]["zynum-core"])
        for owner, bindings in CHECKER.SHARED_ADAPTER_IMPORTS.items():
            for alias, provider in bindings.items():
                self.assertEqual(provider, self.graph["imports"][owner][alias])
        for owner, core in (("checked_abi_mod", "zynum_core_mod"), ("checked_abi_test_mod", "zynum_core_test_mod")):
            for alias in ("zynum-execution", "zynum-numerics-resources"):
                self.assertEqual(self.graph["imports"][core][alias], self.graph["imports"][owner][alias])
        self.assertEqual({"zynum-numerics-resources": "numerics_resources_mod"},
                         self.graph["imports"]["execution_contract_tests.root_module"])
        self.assertEqual(6, len(self.graph["module_source_variants"]["blas_compat_mod"]))
        self.assertNotIn("blas_compat_mod", self.graph["module_sources"])
        self.assertEqual(69, len(self.graph["adapter_edges"]))

    def test_adapter_edges_reject_missing_duplicate_and_wrong_owner(self) -> None:
        # These finite boundaries validate their complete installation/import
        # contract before the common graph walk. Each wrong edge must still
        # raise its exact earlier diagnostic; unrelated errors cannot pass.
        earlier_contract = {'lapacke_lp64_interface': 'finite LAPACKE installation/owner/default-handler contract drifted',
 'lapacke_ilp64_interface': 'finite LAPACKE installation/owner/default-handler contract drifted',
 'sparse_lp64_interface': 'finite Sparse installation/shared-owner contract drifted',
 'sparse_ilp64_interface': 'finite Sparse installation/shared-owner contract drifted',
 'operator_exports_mod': 'Contour build/owner contract changed: package',
 'operator_combined_mod': 'Contour build/owner contract changed: package',
 'abplas_standard_sparse_exports': 'Contour build/owner contract changed: package',
 'sparse_combined_mod': 'Contour build/owner contract changed: package',
 'blas_combined_mod': 'Contour build/owner contract changed: package',
 'abplas_lapacke_composed_root': 'POTRI unified LAPACKE owner/binding selection changed',
 'abplas_lapacke_potri_component': 'POTRI unified LAPACKE owner/binding selection changed',
 'abplas_lp64_lapacke': 'POTRI unified LAPACKE owner/binding selection changed',
 'fftw_f32_composed_root': 'FFTW f32 generator/Core/binding/installation composition changed',
 'fftw_f32_exports_mod': 'FFTW f32 generator/Core/binding/installation composition changed',
 'lapacke_ilp64_composed_root': 'finite LAPACKE installation/owner/default-handler contract drifted',
 'lapacke_ilp64_potri_component': 'finite LAPACKE installation/owner/default-handler contract drifted',
 'lapacke_lp64_composed_root': 'finite LAPACKE installation/owner/default-handler contract drifted',
 'lapacke_lp64_potri_component': 'finite LAPACKE installation/owner/default-handler contract drifted'}
        earlier_edge_contract = {('abplas_lp64_root', 'lapacke-interface-owner'): 'POTRI unified LAPACKE owner/binding selection changed',
 ('abplas_lp64_root', 'zynum-lapacke-exports'): 'POTRI unified LAPACKE owner/binding selection changed',
 ('abplas_lp64_root', 'zynum-checked-sparse-exports'): 'POTRI unified LAPACKE owner/binding selection '
                                                       'changed'}
        earlier_contract.update({
            "feast_exports_mod": "FEAST generator/Core/binding/installation composition changed",
            "feast_combined_mod": "FEAST generator/Core/binding/installation composition changed",
            "contour_exports_mod": "Contour build/owner contract changed: package",
            "contour_combined_mod": "Contour build/owner contract changed: package",
            "lapacke_lp64_syev_component": "finite LAPACKE installation/owner/default-handler contract drifted",
            "lapacke_lp64_syev_composed_root": "finite LAPACKE installation/owner/default-handler contract drifted",
            "abplas_lp64_lapacke_syev_component": "POTRI unified LAPACKE owner/binding selection changed",
            "abplas_lp64_lapacke_syev_composed_root": "POTRI unified LAPACKE owner/binding selection changed",
        })
        for owner, bindings in CHECKER.SHARED_ADAPTER_IMPORTS.items():
            for alias, provider in bindings.items():
                edge = f'{owner}.addImport("{alias}", {provider});'
                for replacement in ("", edge + "\n" + edge, edge.replace(provider, "zynum_core_mod")):
                    with self.subTest(owner=owner, alias=alias, replacement=replacement):
                        self.reject(edge, replacement, earlier_edge_contract.get((owner, alias), earlier_contract.get(owner, "common Core adapter")))

    def test_checked_dependencies_cannot_split_from_their_core(self) -> None:
        for owner in ("checked_abi_mod", "checked_abi_test_mod"):
            core = "zynum_core_mod" if owner == "checked_abi_mod" else "zynum_core_test_mod"
            execution = "execution_mod" if owner == "checked_abi_mod" else "execution_test_mod"
            for alias, provider, wrong in (("zynum-core", core, "zynum_core_test_mod" if core == "zynum_core_mod" else "zynum_core_mod"),
                    ("zynum-execution", execution, "execution_test_mod" if execution == "execution_mod" else "execution_mod"),
                    ("zynum-numerics-resources", "numerics_resources_mod", "host_numerics_resources_mod"),
                    ("zynum-build-options", "zynum_build_options_mod", "runtime_lifecycle_options_mod")):
                edge = f'{owner}.addImport("{alias}", {provider});'
                with self.subTest(owner=owner, alias=alias):
                    self.reject(edge, edge.replace(provider, wrong))
        self.reject('checked_abi_mod.addImport("zynum-build-options", zynum_build_options_mod);',
                    'checked_abi_mod.addImport("zynum-build-options", zynum_build_options.createModule());')

    def test_context_four_resource_edges_reject_missing_duplicate_and_cross_target(self) -> None:
        for owner, provider in (("execution_mod", "numerics_resources_mod"), ("execution_test_mod", "numerics_resources_mod"),
                ("execution_contract_tests.root_module", "numerics_resources_mod"), ("runtime_lifecycle_execution_mod", "runtime_lifecycle_resources_mod")):
            edge = f'{owner}.addImport("zynum-numerics-resources", {provider});'
            for replacement in ("", edge + "\n" + edge, edge.replace(provider, "host_numerics_resources_mod")):
                with self.subTest(owner=owner, replacement=replacement):
                    self.reject(edge, replacement, "numerics resources")

    def test_lifecycle_full_target_and_adapter_target_declarations_are_bound(self) -> None:
        for name in ("blas_abi_mod", "blas_abi_test_mod", "checked_abi_mod", "checked_abi_test_mod", "combined_interfaces_mod"):
            start = self.source.index(f"const {name} =")
            changed = self.source[:start] + self.source[start:].replace(".target = target,", ".target = b.graph.host,", 1)
            with self.subTest(owner=name), self.assertRaisesRegex(CHECKER.InventoryError, "owner/source/target/attachment graph drifted"):
                CHECKER._numerics_resources_build_contract(changed)
        for part in ("runtime_lifecycle_target.result.ofmt == target.result.ofmt",
                     "runtime_lifecycle_target.result.dynamic_linker.eql(target.result.dynamic_linker)",
                     "runtime_lifecycle_target.result.ofmt == b.graph.host.result.ofmt",
                     "runtime_lifecycle_target.result.dynamic_linker.eql(b.graph.host.result.dynamic_linker)"):
            self.reject(part, "true")
        self.reject("std.Target.Query.fromTarget(&runtime_lifecycle_target.result)", "std.Target.Query.fromTarget(&target.result)")
        self.reject(".target = runtime_lifecycle_target,\n        });\n    runtime_lifecycle_execution_mod.addImport", ".target = target,\n        });\n    runtime_lifecycle_execution_mod.addImport")

    def test_six_profile_variants_bind_predicate_paths_and_cardinality(self) -> None:
        for old, new in (("b.path(switch (selected_profile)", "b.path(switch (different_profile)"),
                ('.default => "src/blas/compat.zig",', '.unknown => "src/blas/compat.zig",'),
                ('.default => "src/blas/compat.zig",', '.default => "src/blas/compat.zig",\n .seventh => "src/blas/new.zig",'),
                ('"src/blas/compat_level1_fixed_candidates.zig"', '"src/blas/wrong.zig"'),
                ('"src/blas/compat_level1_fixed_candidates.zig"', '"src/blas/compat_level1_sve_candidates.zig"')):
            with self.subTest(mutation=old):
                self.reject(old, new)

    def test_adapter_receivers_cannot_escape_mutate_or_gain_direct_objects(self) -> None:
        for extra in ('const escaped = blas_abi_mod;', 'checked_abi_test_mod.root_source_file = b.path("src/zynum.zig");',
                'blas_abi_test_mod.import_table.put(b.allocator, "zynum-core", zynum_core_mod) catch unreachable;',
                'combined_interfaces_mod.addImport("zynum-checked-exports", checked_abi_test_mod);',
                'cblas_compat_test_mod.addObject(stride2_isolated_test_library.?);'):
            with self.subTest(extra=extra), self.assertRaises(CHECKER.InventoryError):
                CHECKER._numerics_resources_build_contract(self.source + "\n" + extra)

    def test_combined_product_is_optional_uninstalled_and_keeps_original_barriers(self) -> None:
        step = CHECKER.CHECKED_ABI_OBSERVATION_TEMPLATES["step:build.zig:build:build-combined-interfaces"]
        artifact = CHECKER.CHECKED_ABI_OBSERVATION_TEMPLATES["compile:build.zig:build:combined_interfaces_lib"]
        self.assertEqual([], artifact["install_destinations"])
        self.assertEqual("not-member", step["aggregate_test_membership"])
        self.assertEqual([{"id": "compile:build.zig:build:combined_interfaces_lib", "condition": "always"}], step["direct_dependencies"])
        request = "_ = combined_interfaces_lib.getEmittedBin();"
        self.assertEqual(1, self.source.count(request))
        facts = CHECKER._numerics_resources_build_contract(self.source)
        self.assertEqual([{"owner": "combined_interfaces_lib", "source": "combined_interfaces_lib.getEmittedBin()", "context": ["pubfnbuild(b:*std.Build)void{", "_="]}],
                         [row for row in facts["owner_calls"] if row["owner"] == "combined_interfaces_lib"])
        CHECKER._checked_abi_build_span(self.source)
        for label, replacement in (("missing", ""), ("duplicate", request + "\n    " + request),
                                   ("wrong Compile", "_ = checked_dynamic_lib.getEmittedBin();")):
            changed = self.source.replace(request, replacement, 1)
            with self.subTest(output_request=label, contract="material"), self.assertRaisesRegex(CHECKER.InventoryError, "checked ABI material build contract drifted"):
                CHECKER._checked_abi_build_span(changed)
            with self.subTest(output_request=label, contract="owner graph"), self.assertRaisesRegex(CHECKER.InventoryError, "numerics resources owner/source/target/attachment graph drifted"):
                CHECKER._numerics_resources_build_contract(changed)
        for old, new in (("build_combined_interfaces_step.dependOn(&combined_interfaces_lib.step);", ""),
                ('combined_interfaces_mod.addImport("zynum-checked-exports", checked_abi_mod);', 'combined_interfaces_mod.addImport("zynum-checked-exports", checked_abi_test_mod);'),
                ('const combined_interfaces_lib = b.addLibrary(.{', 'const combined_interfaces_lib = b.addLibrary(.{ .use_llvm = false,')):
            with self.subTest(change=old), self.assertRaisesRegex(CHECKER.InventoryError, "checked ABI material build contract"):
                CHECKER._checked_abi_build_span(self.source.replace(old, new, 1))
        with self.assertRaisesRegex(CHECKER.InventoryError, "binding escapes reviewed block"):
            CHECKER._checked_abi_build_span(self.source + "\nb.installArtifact(combined_interfaces_lib);\n")

    def test_new_common_owner_bytes_are_part_of_frozen_source_contract(self) -> None:
        context = CHECKER._make_discovery_context(REPOSITORY_ROOT)
        original = CHECKER._frozen_regular_bytes
        CHECKER._numerics_resources_source_contract(context)
        for path in ("src/blas/abi/root.zig", "src/interfaces.zig", "src/numerics/execution/context.zig", "src/numerics/execution/context_scratch_test.zig", "src/sparse/core/product.zig", "src/numerics/dense.zig", "src/sparse/operations.zig"):
            def read(active: Any, candidate: str, purpose: str) -> bytes:
                data = original(active, candidate, purpose)
                return data + b"\n// drift\n" if candidate == path else data
            with self.subTest(path=path), mock.patch.object(CHECKER, "_frozen_regular_bytes", side_effect=read):
                with self.assertRaisesRegex(CHECKER.InventoryError, "shared facade source bytes drifted"):
                    CHECKER._numerics_resources_source_contract(context)


class ManagedLapackeSourceContractTests(unittest.TestCase):
    """Typed internal adapters borrow exact Core owners; no standard ABI claim."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.source = (REPOSITORY_ROOT / "build.zig").read_text()
        cls.graph = CHECKER._numerics_resources_build_contract(cls.source)

    def _inside_build(self, statement: str) -> str:
        body_end = _build_body_insertion_offset(self.source)
        return self.source[:body_end] + "    " + statement + "\n" + self.source[body_end:]

    def test_binding_owners_borrow_corresponding_core_and_only_lapack_gains_imports(self) -> None:
        for owner, core, optimize in (("lapacke_binding_mod", "zynum_core_mod", "optimize"),
                                     ("lapacke_binding_test_mod", "zynum_core_test_mod", "test_optimize")):
            self.assertEqual("src/interop/lapacke_binding.zig", self.graph["module_sources"][owner])
            self.assertEqual({"zynum-core": core}, self.graph["imports"][owner])
            declaration, = self.graph["declarations"][owner]
            self.assertEqual(f'const{owner}=b.createModule(.{{.root_source_file=b.path("src/interop/lapacke_binding.zig"),.target=target,.optimize={optimize},.link_libc=true,}});', declaration["source"])
            self.assertNotIn(owner, self.graph["generated_imports"])
            self.assertNotIn("numerical-owner", self.graph["imports"][core])
        self.assertEqual({"general-operator-tests": "general_operator_test_mod", "fftw-interface-owner": "fftw_binding_test_mod",
                          "zynum-lapack": "zynum_lapack_test_mod", "numerical-owner": "lapacke_binding_test_mod",
                          "zynum-core": "zynum_core_test_mod", "zynum-execution": "execution_test_mod",
                          "zynum-numerics-resources": "numerics_resources_mod"},
                         self.graph["imports"][self.graph["root_owners"]["lapack_tests"]])
        self.assertEqual({"zynum-execution": "execution_test_mod", "zynum-numerics-resources": "numerics_resources_mod"},
                         self.graph["imports"][self.graph["root_owners"]["numerics_contract_tests"]])
        binding_consumers = {owner for owner, imports in self.graph["imports"].items()
                             if "lapacke_binding_test_mod" in imports.values()}
        self.assertEqual({"lapack_tests.root_module", "lapacke_posv_tests.root_module", "general_operator_test_mod"}, binding_consumers)

    def test_binding_core_edges_reject_missing_duplicate_and_cross_mode_provider(self) -> None:
        for owner, provider, wrong in (("lapacke_binding_mod", "zynum_core_mod", "zynum_core_test_mod"),
                                      ("lapacke_binding_test_mod", "zynum_core_test_mod", "zynum_core_mod")):
            edge = f'{owner}.addImport("zynum-core", {provider});'
            self.assertEqual(1, self.source.count(edge))
            for replacement, message in (("", "shared facade import changed"),
                    (edge + "\n" + edge, "shared facade import changed"),
                    (edge.replace(provider, wrong), "shared core provider changed")):
                with self.subTest(owner=owner, replacement=replacement), self.assertRaisesRegex(CHECKER.InventoryError, message):
                    CHECKER._numerics_resources_build_contract(self.source.replace(edge, replacement, 1))

    def test_binding_declaration_source_target_and_mode_are_exact(self) -> None:
        for owner in ("lapacke_binding_mod", "lapacke_binding_test_mod"):
            start = self.source.index(f"const {owner} =")
            end = self.source.index("});", start) + 3
            block = self.source[start:end]
            for old, new in ((".target = target,", ".target = b.graph.host,"),
                    (".optimize = " + ("test_optimize" if owner.endswith("_test_mod") else "optimize") + ",", ".optimize = .Debug,"),
                    ('b.path("src/interop/lapacke_binding.zig")', 'b.path("src/zynum.zig")')):
                self.assertIn(old, block)
                changed = self.source[:start] + block.replace(old, new, 1) + self.source[end:]
                message = "interface binding/configuration root ownership changed" if old.startswith("b.path") else "owner/source/target/attachment graph drifted"
                with self.subTest(owner=owner, mutation=old), self.assertRaisesRegex(CHECKER.InventoryError, message):
                    CHECKER._numerics_resources_build_contract(changed)

    def test_lapack_inline_binding_edges_reject_missing_wrong_and_duplicate(self) -> None:
        start = self.source.index("const lapack_tests =")
        end = self.source.index("const run_lapack_tests", start)
        block = self.source[start:end]
        for alias, provider, wrong in (("numerical-owner", "lapacke_binding_test_mod", "lapacke_binding_mod"),
                ("zynum-core", "zynum_core_test_mod", "zynum_core_mod"),
                ("zynum-execution", "execution_test_mod", "execution_mod")):
            entry = f'.{{ .name = "{alias}", .module = {provider} }},'
            self.assertEqual(1, block.count(entry))
            for replacement, message in (("", "owner/source/target/attachment graph drifted"),
                    (entry.replace(provider, wrong), "owner/source/target/attachment graph drifted"),
                    (entry + "\n" + entry, "duplicate named dependency")):
                changed = self.source[:start] + block.replace(entry, replacement, 1) + self.source[end:]
                with self.subTest(alias=alias, replacement=replacement), self.assertRaisesRegex(CHECKER.InventoryError, message):
                    CHECKER._numerics_resources_build_contract(changed)

    def test_binding_receivers_cannot_escape_split_state_or_reverse_core_dependency(self) -> None:
        statements = (
            ('const escaped_binding = lapacke_binding_mod;', "canonical module receiver escapes"),
            ('lapacke_binding_test_mod.root_source_file = b.path("src/zynum.zig");', "canonical module receiver escapes"),
            ('lapacke_binding_test_mod.import_table.put(b.allocator, "zynum-core", zynum_lapack_test_mod) catch unreachable;', "canonical module receiver escapes"),
            ('lapacke_binding_mod.addObject(library);', "owner/source/target/attachment graph drifted"),
            ('lapacke_binding_test_mod.addOptions("zynum-build-options", zynum_build_options);', "owner/source/target/attachment graph drifted"),
            ('lapacke_binding_mod.addImport("zynum-numerics-resources", numerics_resources_mod);', "has an unreviewed import owner"),
            ('lapacke_binding_mod.addImport("zynum-execution", execution_mod);', "shared execution has an unreviewed import owner"),
            ('lapack_tests.root_module.addImport("numerical-owner", lapacke_binding_mod);', "^common Core adapter has an unreviewed import owner: numerical-owner$"),
            ('zynum_core_mod.addImport("numerical-owner", lapacke_binding_mod);', "^common Core adapter has an unreviewed import owner: numerical-owner$"),
        )
        for statement, message in statements:
            with self.subTest(statement=statement), self.assertRaisesRegex(CHECKER.InventoryError, message):
                CHECKER._numerics_resources_build_contract(self._inside_build(statement))

    def test_managed_binding_core_primitive_and_test_bytes_are_bound(self) -> None:
        context = CHECKER._make_discovery_context(REPOSITORY_ROOT)
        original = CHECKER._frozen_regular_bytes
        CHECKER._numerics_resources_source_contract(context)
        for path in CHECKER.MANAGED_LAPACKE_SOURCE_SHA256:
            # Includes the seven Core aliases and all adapter-owned numerical
            # dependencies; a reverse Core import cannot silently enter closure.
            suffix = b'\nconst wrong = @import("interop/lapacke_binding.zig");\n' if path == "src/zynum.zig" else b"\n// source drift\n"
            def read(active: Any, candidate: str, purpose: str) -> bytes:
                data = original(active, candidate, purpose)
                return data + suffix if candidate == path else data
            with self.subTest(path=path), mock.patch.object(CHECKER, "_frozen_regular_bytes", side_effect=read):
                with self.assertRaisesRegex(CHECKER.InventoryError, "shared facade source bytes drifted: " + CHECKER.re.escape(path)):
                    CHECKER._numerics_resources_source_contract(context)

    def test_exact_new_source_classifications_preserve_all_existing_owners(self) -> None:
        inventory = json.loads((REPOSITORY_ROOT / "tools/build_inventory.json").read_text())
        before = {row["path"]: copy.deepcopy(row) for row in inventory["repository_file_classifications"]}
        CHECKER._apply_reviewed_build_inventory_migrations(_with_source_reviewed_execution_compile(inventory))
        after = {row["path"]: row for row in inventory["repository_file_classifications"]}
        expected = {
            "src/interop/lapacke_access.zig": "library-source",
            "src/interop/lapacke_getrf.zig": "library-source",
            "test/lapack/interop/getrf_adapter_test.zig": "test-infrastructure",
            "src/interop/lapacke_getrs.zig": "library-source",
            "test/lapack/interop/getrs_adapter_test.zig": "test-infrastructure",
            "test/lapack/interop/raw_access_test.zig": "test-infrastructure",
            "src/interop/lapacke_binding.zig": "library-source",
            "src/interop/lapacke_common.zig": "library-source",
            "src/interop/lapacke_services.zig": "library-source",
            "test/lapack/interop/shared_services_test.zig": "test-infrastructure",
            "src/interop/lapacke_potrf.zig": "library-source",
            "src/interop/lapacke_potrs.zig": "library-source",
            "src/lapack/core/triangular_substitute.zig": "library-source",
            "test/lapack/interop/managed_adapter_test.zig": "test-infrastructure",
            "test/lapack/interop/managed_support.zig": "test-infrastructure",
            "test/lapack/interop/potrf_adapter_test.zig": "test-infrastructure",
            "test/lapack/interop/potrs_adapter_test.zig": "test-infrastructure",
            "test/lapack/interop/potrs_support.zig": "test-infrastructure",
        }
        expected["src/interop/lapacke_diagnostics.zig"] = "library-source"
        expected["src/interop/lapacke_gesv.zig"] = "library-source"
        expected["test/lapack/interop/gesv_adapter_test.zig"] = "test-infrastructure"
        expected["src/interop/lapacke_default_errors.c"] = "library-source"
        expected.update({"src/interop/lapacke_stev.zig": "library-source", "src/lapack/core/tridiagonal_eigen.zig": "library-source", "test/lapack/interop/stev_adapter_test.zig": "test-infrastructure", "test/lapack/tridiagonal_eigen_test.zig": "test-infrastructure", "tools/link_darwin_interface.py": "build-composition"})
        expected.update({
            "src/interop/lapacke_gelsd.zig": "library-source",
            "src/interop/lapacke_gelsd_column.zig": "library-source",
            "src/interop/lapacke_gelsd_policy.zig": "library-source",
            "src/interop/lapacke_gelsd_query.zig": "library-source",
            "src/interop/lapacke_gelsd_workspace.zig": "library-source",
            "src/lapack/core/bidiagonal.zig": "library-source",
            "src/lapack/core/bidiagonal_svd.zig": "library-source",
            "src/lapack/core/minimum_norm.zig": "library-source",
            "src/lapack/core/rank_least_squares.zig": "library-source",
            "src/lapack/core/svd.zig": "library-source",
            "test/lapack/bidiagonal_test.zig": "test-infrastructure",
            "test/lapack/interop/gelsd_policy_test.zig": "test-infrastructure",
            "src/abplas_lp64.zig": "library-source",
            "src/lapack/core/complex_svd.zig": "library-source",
            "src/lapack/core/complex_minimum_norm.zig": "library-source",
            "src/lapack/core/complex_rank_least_squares.zig": "library-source",
            "test/lapack/complex_svd_test.zig": "test-infrastructure",
            "test/lapack/complex_rank_least_squares_test.zig": "test-infrastructure",
            "test/lapack/complex_svd_facade_test.zig": "test-infrastructure",
            "src/interop/lapacke_complex_gelsd.zig": "library-source",
            "src/interop/lapacke_complex_gelsd_column.zig": "library-source",
            "src/interop/lapacke_complex_gelsd_policy.zig": "library-source",
            "src/interop/lapacke_complex_gelsd_query.zig": "library-source",
            "src/interop/lapacke_complex_gelsd_workspace.zig": "library-source",
            "test/lapack/interop/complex_gelsd_workspace_test.zig": "test-infrastructure",
            "test/lapack/interop/complex_gelsd_query_test.zig": "test-infrastructure",
            "test/lapack/interop/complex_gelsd_column_test.zig": "test-infrastructure",
            "test/lapack/interop/complex_gelsd_adapter_test.zig": "test-infrastructure",
            "test/lapack/interop/complex_gelsd_shared_projection_test.zig": "test-infrastructure",
            "tools/interface_templates/complex_gelsd_lp64.zig.in": "developer-tooling",
        })
        for width in ("lp64", "ilp64"):
            for stem in ("potrf-potrs-getrf-getrs-ordinary-", "potrf-potrs-getrf-getrs-gesv-ordinary-", "potrf-potrs-getrf-getrs-gesv-stev-ordinary-", "potrf-potrs-getrf-getrs-gesv-stev-gelsd-ordinary-"):
                prefix = f"interface_subset/mkl2026.{width}/{stem}{width}-v1/"
                for name in ("subset.h", "exports.zig", "abi_manifest.json", "exports.def"):
                    expected[prefix + name] = "abi-compatibility"
        for subset in (
            "complex-gelsd-ordinary-lp64-v1",
            "potrf-potrs-getrf-getrs-gesv-stev-gelsd-heev-hegv-complex-gelsd-ordinary-lp64-v1",
            "potrf-potrs-getrf-getrs-gesv-stev-gelsd-heev-hegv-geqrf-ormqr-unmqr-orgqr-ungqr-complex-gelsd-ordinary-lp64-v1",
        ):
            for name in ("subset.h", "exports.zig", "abi_manifest.json", "exports.def"):
                expected[f"interface_subset/mkl2026.lp64/{subset}/{name}"] = "abi-compatibility"
        expected.update({
            "src/lapack/core/orthonormal_completion.zig": "library-source",
            "src/interop/lapacke_gesvd.zig": "library-source",
            "src/interop/lapacke_gesvd_policy.zig": "library-source",
            "test/lapack/orthonormal_completion_test.zig": "test-infrastructure",
            "test/lapack/interop/gesvd_adapter_test.zig": "test-infrastructure",
            "test/lapack/interop/gesvd_raw_boundary_test.zig": "test-infrastructure",
            "tools/interface_templates/gesvd_lp64.zig.in": "developer-tooling",
        })
        svd_sets = ["gesvd-ordinary-lp64-v1"]
        for qr in (False, True):
            for complex_gelsd in (False, True):
                svd_sets.append("potrf-potrs-getrf-getrs-gesv-stev-gelsd-heev-hegv"
                    + ("-geqrf-ormqr-unmqr-orgqr-ungqr" if qr else "")
                    + ("-complex-gelsd" if complex_gelsd else "")
                    + "-gesvd-ordinary-lp64-v1")
        self.assertEqual(5, len(svd_sets))
        for subset in svd_sets:
            for name in ("subset.h", "exports.zig", "abi_manifest.json", "exports.def"):
                expected[f"interface_subset/mkl2026.lp64/{subset}/{name}"] = "abi-compatibility"
        self.assertEqual(set(expected), {row["path"] for row in CHECKER.MANAGED_LAPACKE_CLASSIFICATIONS})
        sparse_paths = {row["path"] for row in CHECKER.MANAGED_SPARSE_CLASSIFICATIONS}
        self.assertEqual((set(expected) | sparse_paths | {"tools/standard_abi_bindings.py"}) - before.keys(), after.keys() - before.keys())
        self.assertEqual(before, {path: after[path] for path in before})
        for path, owner in expected.items():
            generated = path.startswith("interface_subset/")
            kind = ("python-source" if path.endswith(".py") else "json-data" if path.endswith(".json") else
                    "other-public-file" if path.endswith((".def", ".in")) else
                    "foreign-language-source" if path.endswith((".c", ".h")) else "zig-source")
            classification = "designated-reproducible-contract-reference" if generated else "non-generated-source"
            self.assertEqual({"path": path, "kind": kind, "class": classification, "owner": owner}, after[path])


class ManagedSparseSourceContractTests(unittest.TestCase):
    """Typed controls retain the existing Core, Context and object ownership."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.source = (REPOSITORY_ROOT / "build.zig").read_text()
        cls.graph = CHECKER._numerics_resources_build_contract(cls.source)

    def test_sparse_direct_imports_converge_with_facade_and_core_resources(self) -> None:
        owner = self.graph["root_owners"]["sparse_tests"]
        self.assertEqual("sparse_tests.root_module", owner)
        self.assertEqual({"checked-sparse-tests": "checked_sparse_test_mod", "zynum-sparse": "zynum_sparse_test_mod", "sparse-interface-owner": "mkl_sparse_binding_test_mod", "zynum-core": "zynum_core_test_mod",
                          "zynum-execution": "execution_test_mod", "zynum-numerics-resources": "numerics_resources_mod"},
                         self.graph["imports"][owner])
        self.assertEqual({"zynum-core": "zynum_core_test_mod"}, self.graph["imports"]["zynum_sparse_test_mod"])
        self.assertEqual("execution_test_mod", self.graph["imports"]["zynum_core_test_mod"]["zynum-execution"])
        for resource_owner in (owner, "zynum_core_test_mod", "execution_test_mod"):
            self.assertEqual("numerics_resources_mod", self.graph["imports"][resource_owner]["zynum-numerics-resources"])
        self.assertEqual("src/facades/sparse.zig", self.graph["module_sources"]["zynum_sparse_test_mod"])
        self.assertNotIn("numerical-owner", self.graph["imports"][owner])
        self.assertEqual({"zynum-core": "zynum_core_mod"}, self.graph["imports"]["lapacke_binding_mod"])
        self.assertEqual({"zynum-core": "zynum_core_test_mod"}, self.graph["imports"]["lapacke_binding_test_mod"])

    def test_sparse_inline_core_and_execution_missing_wrong_or_duplicate_are_rejected(self) -> None:
        start = self.source.index("const sparse_tests =")
        end = self.source.index("const run_sparse_tests", start)
        block = self.source[start:end]
        for alias, provider, wrong in (("zynum-core", "zynum_core_test_mod", "zynum_core_mod"),
                                      ("zynum-execution", "execution_test_mod", "execution_mod")):
            entry = f'.{{ .name = "{alias}", .module = {provider} }},'
            self.assertEqual(1, block.count(entry))
            for replacement, message in (("", "owner/source/target/attachment graph drifted"),
                    (entry.replace(provider, wrong), "owner/source/target/attachment graph drifted"),
                    (entry + "\n" + entry, "duplicate named dependency")):
                changed = self.source[:start] + block.replace(entry, replacement, 1) + self.source[end:]
                with self.subTest(alias=alias, replacement=replacement), self.assertRaisesRegex(CHECKER.InventoryError, message):
                    CHECKER._numerics_resources_build_contract(changed)

    def test_sparse_owner_rebinding_resource_split_and_object_attachments_are_rejected(self) -> None:
        body_end = _build_body_insertion_offset(self.source)
        for statement in (
            'sparse_tests.root_module.addImport("zynum-core", zynum_core_mod);',
            'sparse_tests.root_module.addImport("zynum-execution", execution_mod);',
            'sparse_tests.root_module.addImport("zynum-numerics-resources", host_numerics_resources_mod);',
            'sparse_tests.root_module.addObject(library);',
            'zynum_sparse_test_mod.addObject(library);',
        ):
            changed = self.source[:body_end] + "    " + statement + "\n" + self.source[body_end:]
            with self.subTest(statement=statement), self.assertRaises(CHECKER.InventoryError):
                CHECKER._numerics_resources_build_contract(changed)

    def test_sparse_source_pins_agree_with_all_shared_core_maps_and_reject_drift(self) -> None:
        maps = (CHECKER.SHARED_FACADE_SOURCE_SHA256, CHECKER.EXECUTION_SOURCE_SHA256,
                CHECKER.EXECUTION_OBJECT_SOURCE_SHA256, CHECKER.COMMON_CORE_SOURCE_SHA256,
                CHECKER.MANAGED_LAPACKE_SOURCE_SHA256, CHECKER.MANAGED_SPARSE_SOURCE_SHA256)
        seen = {}
        for mapping in maps:
            for path, digest in mapping.items():
                if path in seen:
                    self.assertEqual(seen[path], digest, path)
                seen[path] = digest
        context = CHECKER._make_discovery_context(REPOSITORY_ROOT)
        original = CHECKER._frozen_regular_bytes
        CHECKER._numerics_resources_source_contract(context)
        self.assertEqual({"src/sparse/object.zig", "src/zynum.zig", "test/sparse/object_test.zig",
                          "test/sparse/context_object_test.zig", "test/sparse/sparse_test.zig",
                          "src/interop/mkl_sparse.zig", "src/interop/mkl_sparse_binding.zig", "src/interop/sparse_handle.zig",
                          "test/sparse/interop/handle_test.zig", "test/sparse/interop/mkl_sparse_adapter_test.zig",
                          "src/interop/mkl_sparse_access.zig", "src/interop/mkl_sparse_raw.zig",
                          "test/sparse/interop/raw_lifecycle_test.zig", "test/sparse/interop/mkl_sparse_mv_test.zig",
                          "test/sparse/interop/mkl_sparse_mm_test.zig",
                          "src/interop/mkl_sparse_solve.zig", "src/interop/mkl_sparse_solve_raw.zig",
                          "src/sparse/core/triangular_complex.zig"},
                         set(CHECKER.MANAGED_SPARSE_SOURCE_SHA256))
        for path in CHECKER.MANAGED_SPARSE_SOURCE_SHA256:
            def read(active: Any, candidate: str, purpose: str) -> bytes:
                data = original(active, candidate, purpose)
                return data + b"\n// source drift\n" if candidate == path else data
            with self.subTest(path=path), mock.patch.object(CHECKER, "_frozen_regular_bytes", side_effect=read):
                with self.assertRaisesRegex(CHECKER.InventoryError, "shared facade source bytes drifted: " + CHECKER.re.escape(path)):
                    CHECKER._numerics_resources_source_contract(context)

    def test_sparse_classifications_preserve_existing_and_lapacke_rows(self) -> None:
        inventory = json.loads((REPOSITORY_ROOT / "tools/build_inventory.json").read_text())
        new_paths = ('src/interop/mkl_sparse_access.zig', 'src/interop/mkl_sparse_raw.zig', 'test/sparse/interop/raw_lifecycle_test.zig', 'test/sparse/interop/mkl_sparse_mv_test.zig', 'test/sparse/interop/mkl_sparse_mm_test.zig')
        inventory["repository_file_classifications"] = [
            row for row in inventory["repository_file_classifications"] if row["path"] not in new_paths
        ]
        before = {row["path"]: copy.deepcopy(row) for row in inventory["repository_file_classifications"]}
        CHECKER._apply_reviewed_build_inventory_migrations(_with_source_reviewed_execution_compile(inventory))
        after = {row["path"]: row for row in inventory["repository_file_classifications"]}
        expected = {"src/interop/mkl_sparse_access.zig": "library-source", "src/interop/mkl_sparse_raw.zig": "library-source",
                    "test/sparse/interop/raw_lifecycle_test.zig": "test-infrastructure",
                    "test/sparse/interop/mkl_sparse_mv_test.zig": "test-infrastructure",
                    "test/sparse/interop/mkl_sparse_mm_test.zig": "test-infrastructure",
                    "src/interop/mkl_sparse.zig": "library-source", "src/interop/mkl_sparse_binding.zig": "library-source",
                    "src/interop/sparse_handle.zig": "library-source", "test/sparse/interop/handle_test.zig": "test-infrastructure",
                    "test/sparse/interop/mkl_sparse_adapter_test.zig": "test-infrastructure", "src/sparse/object.zig": "library-source", "test/sparse/object_test.zig": "test-infrastructure",
                    "test/sparse/context_object_test.zig": "test-infrastructure"}
        self.assertEqual(set(expected), {row["path"] for row in CHECKER.MANAGED_SPARSE_CLASSIFICATIONS})
        lapacke = {row["path"]: row for row in CHECKER.MANAGED_LAPACKE_CLASSIFICATIONS}
        self.assertEqual((set(expected) | lapacke.keys() | {"tools/standard_abi_bindings.py"}) - before.keys(), after.keys() - before.keys())
        self.assertEqual(before, {path: after[path] for path in before})
        self.assertEqual(lapacke, {path: after[path] for path in lapacke})
        for path, owner in expected.items():
            self.assertEqual({"path": path, "kind": "zig-source", "class": "non-generated-source", "owner": owner}, after[path])


class InterfaceServicesSourceContractTests(unittest.TestCase):
    """Bounded typed-owner contracts; no C exports or native evidence inferred."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.source = (REPOSITORY_ROOT / "build.zig").read_text()
        cls.graph = CHECKER._numerics_resources_build_contract(cls.source)

    def inside_build(self, statement: str) -> str:
        body_end = _build_body_insertion_offset(self.source)
        return self.source[:body_end] + statement + "\n" + self.source[body_end:]

    def test_sparse_binding_owners_borrow_core_and_inherit_common_services(self) -> None:
        for owner, core, execution, mode in (("mkl_sparse_binding_mod", "zynum_core_mod", "execution_mod", "optimize"),
                ("mkl_sparse_binding_test_mod", "zynum_core_test_mod", "execution_test_mod", "test_optimize")):
            self.assertEqual({"zynum-core": core}, self.graph["imports"][owner])
            self.assertEqual("src/interop/mkl_sparse_binding.zig", self.graph["module_sources"][owner])
            self.assertEqual(execution, self.graph["imports"][core]["zynum-execution"])
            self.assertEqual("numerics_resources_mod", self.graph["imports"][core]["zynum-numerics-resources"])
            self.assertEqual("numerics_resources_mod", self.graph["imports"][execution]["zynum-numerics-resources"])
            self.assertNotIn(owner, self.graph["generated_imports"])
            self.assertEqual(f'const{owner}=b.createModule(.{{.root_source_file=b.path("src/interop/mkl_sparse_binding.zig"),.target=target,.optimize={mode},.link_libc=true,}});',
                             self.graph["declarations"][owner][0]["source"])
        consumers = {owner for owner, imports in self.graph["imports"].items() if "mkl_sparse_binding_test_mod" in imports.values()}
        self.assertEqual({"sparse_tests.root_module"}, consumers)
        self.assertEqual("mkl_sparse_binding_test_mod", self.graph["imports"]["sparse_tests.root_module"]["sparse-interface-owner"])

    def test_sparse_binding_missing_duplicate_or_cross_mode_edges_fail(self) -> None:
        for owner, core, wrong in (("mkl_sparse_binding_mod", "zynum_core_mod", "zynum_core_test_mod"),
                ("mkl_sparse_binding_test_mod", "zynum_core_test_mod", "zynum_core_mod")):
            edge = f'{owner}.addImport("zynum-core", {core});'
            for replacement in ("", edge + "\n" + edge, edge.replace(core, wrong)):
                with self.subTest(owner=owner, mutation=replacement), self.assertRaises(CHECKER.InventoryError):
                    CHECKER._numerics_resources_build_contract(self.source.replace(edge, replacement, 1))
        entry = '.{ .name = "sparse-interface-owner", .module = mkl_sparse_binding_test_mod },'
        self.assertEqual(1, self.source.count(entry))
        for replacement in ("", entry + "\n" + entry, entry.replace("mkl_sparse_binding_test_mod", "mkl_sparse_binding_mod")):
            with self.subTest(entry=replacement), self.assertRaises(CHECKER.InventoryError):
                CHECKER._numerics_resources_build_contract(self.source.replace(entry, replacement, 1))

    def test_sparse_binding_source_target_mode_and_extra_state_edges_fail(self) -> None:
        for owner in ("mkl_sparse_binding_mod", "mkl_sparse_binding_test_mod"):
            start = self.source.index("const " + owner + " =")
            mode = "test_optimize" if "test" in owner else "optimize"
            for old, new in (('.root_source_file = b.path("src/interop/mkl_sparse_binding.zig")', '.root_source_file = b.path("src/zynum.zig")'),
                    (".target = target", ".target = b.graph.host"), (".optimize = " + mode, ".optimize = .debug")):
                changed = self.source[:start] + self.source[start:].replace(old, new, 1)
                with self.subTest(owner=owner, mutation=old), self.assertRaises(CHECKER.InventoryError):
                    CHECKER._numerics_resources_build_contract(changed)
            for suffix in ('addImport("zynum-numerics-resources", numerics_resources_mod);',
                    'addImport("zynum-execution", execution_mod);', 'addOptions("zynum-build-options", zynum_build_options);',
                    'addObject(library);'):
                with self.subTest(owner=owner, extra=suffix), self.assertRaises(CHECKER.InventoryError):
                    CHECKER._numerics_resources_build_contract(self.inside_build(owner + "." + suffix))
        for statement in ('const escaped_sparse_binding = mkl_sparse_binding_mod;',
                'zynum_core_test_mod.addImport("sparse-interface-owner", mkl_sparse_binding_test_mod);',
                'lapack_tests.root_module.addImport("sparse-interface-owner", mkl_sparse_binding_test_mod);',
                'mkl_sparse_binding_test_mod.import_table.put(b.allocator, "zynum-core", zynum_core_mod) catch unreachable;'):
            with self.subTest(statement=statement), self.assertRaises(CHECKER.InventoryError):
                CHECKER._numerics_resources_build_contract(self.inside_build(statement))

    def test_disconnected_binding_and_configuration_roots_are_rejected(self) -> None:
        for path in CHECKER.INTERFACE_BINDING_ROOT_COUNTS:
            statement = f'const another_owner = b.createModule(.{{ .root_source_file = b.path("{path}"), .target = target, .optimize = test_optimize }});'
            with self.subTest(path=path), self.assertRaisesRegex(CHECKER.InventoryError, "interface binding/configuration root ownership changed"):
                CHECKER._numerics_resources_build_contract(self.inside_build(statement))

    def test_relative_imports_cannot_copy_configuration_shell_or_reverse_core(self) -> None:
        for target, owner in CHECKER.INTERFACE_PRIVATE_IMPORT_OWNERS.items():
            CHECKER._interface_private_source_imports(owner, f'const ok = @import("{Path(target).name}");')
            for source in ("test/lapack/other.zig", "src/blas/unpinned.zig", "src/interop/another.zig"):
                relative = os.path.relpath(target, Path(source).parent)
                with self.subTest(source=source, target=target), self.assertRaisesRegex(CHECKER.InventoryError, "interface private source has another owner"):
                    CHECKER._interface_private_source_imports(source, f'const wrong = @import("{relative}");')
                CHECKER._interface_private_source_imports(source, f'// @import("{relative}")\nconst string = "@import(ignored)";')
        for source in ("src/zynum.zig", "src/blas.zig", "src/lapack.zig", "src/sparse.zig", "src/lapack/unchanged.zig", "src/sparse/unchanged.zig", "src/numerics/unchanged.zig", "src/blas/unchanged.zig"):
            for target in ("src/interop/lapacke_common.zig", "src/interop/mkl_sparse_binding.zig"):
                relative = os.path.relpath(target, Path(source).parent)
                with self.subTest(source=source, target=target), self.assertRaisesRegex(CHECKER.InventoryError, "common numerical source imports interface layer"):
                    CHECKER._interface_private_source_imports(source, f'const wrong = @import("{relative}");')

    def test_source_gate_applies_inward_import_policy_beyond_pinned_files(self) -> None:
        context = CHECKER._make_discovery_context(REPOSITORY_ROOT)
        original = CHECKER._frozen_regular_bytes
        CHECKER._numerics_resources_source_contract(context)
        for path in ("src/blas/types.zig", "src/blas.zig", "src/lapack.zig", "src/sparse.zig"):
            self.assertIn(path, context.public_files.path_set)
            for target, message in (("src/interop/lapacke_services.zig", "interface private source has another owner"),
                                    ("src/interop/lapacke_common.zig", "common numerical source imports interface layer"),
                                    ("src/interop/mkl_sparse_binding.zig", "common numerical source imports interface layer")):
                relative = os.path.relpath(target, Path(path).parent)
                def read(active: Any, candidate: str, purpose: str) -> bytes:
                    data = original(active, candidate, purpose)
                    return data + f'\nconst forbidden = @import("{relative}");\n'.encode() if candidate == path else data
                with self.subTest(path=path, target=target), mock.patch.object(CHECKER, "_frozen_regular_bytes", side_effect=read):
                    with self.assertRaisesRegex(CHECKER.InventoryError, message):
                        CHECKER._numerics_resources_source_contract(context)


class SharedExecutionSourceContractTests(unittest.TestCase):
    """Source-only identity and fail-closed owner graph; no native evidence."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.source = (REPOSITORY_ROOT / "build.zig").read_text()
        cls.graph = CHECKER._numerics_resources_build_contract(cls.source)

    def test_execution_owner_options_and_canonical_isolation_are_exact(self) -> None:
        self.assertEqual(12, len(self.graph["execution_edges"]))
        self.assertEqual(14, len(self.graph["option_edges"]))
        self.assertEqual(25, len(CHECKER.REQUIRED_TEST_INVENTORY_FACTORY_CASES))
        self.assertEqual("src/numerics/execution/default.zig", self.graph["module_sources"]["execution_mod"])
        self.assertEqual("src/numerics/execution/default.zig", self.graph["module_sources"]["execution_test_mod"])
        self.assertIn("if(optimize==test_optimize)execution_modelse", self.graph["declarations"]["execution_test_mod"][0]["source"])
        owner = self.graph["root_owners"]["execution_contract_tests"]
        self.assertEqual("src/numerics/execution/owner.zig", self.graph["root_sources"]["execution_contract_tests"])
        self.assertEqual({"zynum-numerics-resources": "numerics_resources_mod"}, self.graph["imports"].get(owner, {}))
        self.assertEqual([], self.graph["generated_imports"].get(owner, []))
        for root in CHECKER.EXECUTION_DIRECT_TEST_ROOTS:
            receiver = self.graph["root_owners"][root]
            self.assertEqual("execution_test_mod", self.graph["imports"][receiver][CHECKER.EXECUTION_ALIAS])
        self.assertEqual("execution_test_mod", self.graph["imports"][self.graph["root_owners"]["blas_module_tests"]][CHECKER.EXECUTION_ALIAS])
        self.assertEqual("runtime_lifecycle_execution_mod", self.graph["imports"]["runtime_lifecycle_probe.root_module"][CHECKER.EXECUTION_ALIAS])
        for root, alias, compat in (
            ("fortran_tests", "zynum_blas_fortran_compat", "fortran_compat_test_mod"),
            ("cblas_tests", "zynum_blas_cblas_compat", "cblas_compat_test_mod"),
        ):
            with self.subTest(root=root):
                owner = self.graph["root_owners"][root]
                self.assertEqual(root + ".root_module", owner)
                self.assertEqual("execution_test_mod", CHECKER.EXECUTION_IMPORT_OWNERS[owner])
                self.assertEqual("execution_test_mod", self.graph["imports"][owner][CHECKER.EXECUTION_ALIAS])
                self.assertEqual(compat, self.graph["imports"][owner][alias])
                self.assertEqual("blas_abi_test_mod", self.graph["imports"][compat]["zynum-blas-abi"])
                self.assertEqual("zynum_core_test_mod", self.graph["imports"]["blas_abi_test_mod"]["zynum-core"])
                self.assertEqual(self.graph["imports"][owner][CHECKER.EXECUTION_ALIAS],
                                 self.graph["imports"]["zynum_core_test_mod"][CHECKER.EXECUTION_ALIAS])

    def test_execution_edges_missing_swapped_duplicate_and_extra_are_rejected(self) -> None:
        for owner, provider in CHECKER.EXECUTION_IMPORT_OWNERS.items():
            edge = f'{owner}.addImport("zynum-execution", {provider});'
            self.assertEqual(1, len(CHECKER.re.findall(r"(?<![A-Za-z0-9_])" + CHECKER.re.escape(edge), self.source)))
            wrong = "execution_mod" if provider != "execution_mod" else "execution_test_mod"
            for replacement in ("", edge.replace(provider, wrong), edge + "\n" + edge):
                with self.subTest(owner=owner, replacement=replacement), self.assertRaisesRegex(CHECKER.InventoryError, "shared execution"):
                    CHECKER._numerics_resources_build_contract(self.source.replace(edge, replacement, 1))
        with self.assertRaisesRegex(CHECKER.InventoryError, "unreviewed import owner"):
            CHECKER._numerics_resources_build_contract(self.source + '\nunknown.addImport("zynum-execution", execution_test_mod);\n')

    def test_execution_mode_options_instance_and_owner_exclusions_are_bound(self) -> None:
        replacements = (
            ('if (optimize == test_optimize) execution_mod else', 'if (true) execution_mod else'),
            ('const zynum_build_options_mod = zynum_build_options.createModule();', 'const zynum_build_options_mod = other_options.createModule();'),
            ('execution_mod.addImport("zynum-build-options", zynum_build_options_mod);', 'execution_mod.addImport("zynum-build-options", zynum_build_options.createModule());'),
            ('execution_test_mod.addImport("zynum-build-options", zynum_build_options_mod);', 'execution_test_mod.addOptions("zynum-build-options", zynum_build_options);'),
            ('official_tests != modern_tests and official_tests != execution_contract_tests', 'official_tests != modern_tests'),
            ('if (official_tests != execution_contract_tests) official_tests.root_module.addImport', 'official_tests.root_module.addImport'),
            (' and official_tests != execution_contract_tests) {\n            for (kernel_test_libraries)', ') {\n            for (kernel_test_libraries)'),
            ('official_tests == packed_parallel_tests or', ''),
            ('.{ .name = "zynum-execution", .module = execution_test_mod },', '.{ .name = "zynum-execution", .module = execution_mod },'),
        )
        for old, new in replacements:
            self.assertIn(old, self.source)
            with self.subTest(change=old), self.assertRaises(CHECKER.InventoryError):
                CHECKER._numerics_resources_build_contract(self.source.replace(old, new, 1))
        for owner in ("execution_mod", "execution_test_mod", "runtime_lifecycle_execution_mod", "execution_contract_tests"):
            start = self.source.index(f"const {owner} =")
            before = ".target = runtime_lifecycle_target" if owner.startswith("runtime_") else ".target = target"
            for old, new in ((before, ".target = b.graph.host"), (".link_libc = true", ".link_libc = false")):
                changed = self.source[:start] + self.source[start:].replace(old, new, 1)
                self.assertNotEqual(self.source, changed)
                with self.subTest(owner=owner, change=old), self.assertRaises(CHECKER.InventoryError):
                    CHECKER._numerics_resources_build_contract(changed)

    def test_execution_receivers_cannot_escape_or_overwrite(self) -> None:
        statements = (
            'const copied_execution = execution_test_mod; copied_execution.root_source_file = b.path("src/zynum.zig");',
            'execution_test_mod.import_table.put(b.allocator, "zynum-build-options", other_options) catch unreachable;',
            'execution_mod.root_source_file = b.path("src/zynum.zig");',
            'const copied_compat = blas_compat_mod; copied_compat.addImport("zynum-execution", execution_test_mod);',
            'cblas_compat_mod.import_table.put(b.allocator, "zynum-execution", execution_test_mod) catch unreachable;',
            'execution_test_mod.addImport("zynum-build-options", zynum_build_options_mod);',
            'execution_contract_tests.root_module.addImport("zynum-execution", execution_test_mod);',
            'execution_contract_tests.root_module.addImport("zynum-numerics-resources", numerics_resources_mod);',
            'execution_contract_tests.root_module.addOptions("zynum-build-options", zynum_build_options);',
            'execution_contract_tests.root_module.addObject(stride2_isolated_test_library.?);',
            'const copied_compile = execution_contract_tests; copied_compile.root_module.addImport("zynum-execution", execution_test_mod);',
            'const copied_blas_compile = blas_module_tests; copied_blas_compile.root_module.root_source_file = b.path("src/zynum.zig");',
            'blas_module_tests.root_module.addImport("zynum-execution", execution_mod);',
        )
        body_end = _build_body_insertion_offset(self.source)
        for statement in statements:
            text = self.source[:body_end] + statement + "\n" + self.source[body_end:]
            with self.subTest(statement=statement), self.assertRaises(CHECKER.InventoryError):
                CHECKER._numerics_resources_build_contract(text)

    def test_execution_auxiliary_objects_bind_exact_consumers_and_target(self) -> None:
        path = "src/blas/task_runtime_host_object_root.zig"
        auxiliary = (REPOSITORY_ROOT / path).read_text()
        self.assertEqual(
            'comptime{_=@import("reference_cache_hook.zig");_=@import("kernels/isolated/task_runtime_host.zig");}',
            CHECKER._compact_zig_contract(auxiliary),
        )
        context = CHECKER._make_discovery_context(REPOSITORY_ROOT)
        original_read = CHECKER._frozen_regular_bytes
        direct = '    _ = @import("reference_cache_hook.zig");'
        for replacement in ("", '    if (@import("builtin").is_test) _ = @import("reference_cache_hook.zig");'):
            changed = auxiliary.replace(direct, replacement, 1).encode()
            def read_source(active, relative, label):
                return changed if relative == path else original_read(active, relative, label)
            with self.subTest(activation=replacement), mock.patch.object(
                CHECKER, "_frozen_regular_bytes", side_effect=read_source
            ), self.assertRaises(CHECKER.InventoryError):
                CHECKER._numerics_resources_source_contract(context)
        self.assertEqual(set(CHECKER.EXECUTION_OBJECT_CONSUMERS), set(self.graph['linked_module_entries']))
        for root, receiver in CHECKER.EXECUTION_OBJECT_CONSUMERS.items():
            row, = self.graph['linked_module_entries'][root]
            self.assertEqual('task_runtime_host_test_library.root_module', row['module'])
            self.assertEqual('src/blas/task_runtime_host_object_root.zig', row['path'])
            self.assertEqual(['pubfnbuild(b:*std.Build)void{', 'if(target.result.cpu.arch==.x86_64){'], row['context'])
            edge = f'{receiver}.addObject(task_runtime_host_test_library);'
            self.assertEqual(1, self.source.count(edge))
            for replacement in ('', edge.replace('task_runtime_host_test_library', 'stride2_isolated_test_library.?')):
                with self.subTest(root=root, replacement=replacement), self.assertRaisesRegex(CHECKER.InventoryError, 'shared execution auxiliary object attachment'):
                    CHECKER._numerics_resources_build_contract(self.source.replace(edge, replacement, 1))
        edge = 'task_runtime_host_test_library.root_module.addImport("zynum-execution", execution_test_mod);'
        for replacement in ('', edge.replace('execution_test_mod', 'execution_mod')):
            with self.subTest(import_change=replacement), self.assertRaisesRegex(CHECKER.InventoryError, 'shared execution'):
                CHECKER._numerics_resources_build_contract(self.source.replace(edge, replacement, 1))

    def test_execution_source_bytes_and_classification_are_explicit(self) -> None:
        inventory = json.loads((REPOSITORY_ROOT / "tools/build_inventory.json").read_text())
        previous_rows = copy.deepcopy(inventory["repository_file_classifications"])
        CHECKER._apply_reviewed_build_inventory_migrations(_with_source_reviewed_execution_compile(inventory))
        rows = {row["path"]: row for row in inventory["repository_file_classifications"]}
        for previous in previous_rows:
            self.assertEqual(previous, rows[previous["path"]])
        paths = (CHECKER.NUMERICS_RESOURCES_PATH, *CHECKER.EVD_SOURCE_SHA256, *CHECKER.CONTROLS_SOURCE_SHA256, *CHECKER.SYGV_SOURCE_SHA256, *CHECKER.SYEV_SOURCE_SHA256, *CHECKER.CONTOUR_SOURCE_SHA256, *CHECKER.FEAST_SOURCE_SHA256, *CHECKER.FFTW_F32_SOURCE_SHA256, *CHECKER.POTRI_OWNER_SOURCE_SHA256, *CHECKER.SHARED_FACADE_SOURCE_SHA256, *CHECKER.EXECUTION_SOURCE_SHA256, *CHECKER.EXECUTION_OBJECT_SOURCE_SHA256, *CHECKER.COMMON_CORE_SOURCE_SHA256, *CHECKER.MANAGED_LAPACKE_SOURCE_SHA256, *CHECKER.MANAGED_SPARSE_SOURCE_SHA256, *CHECKER.UNIFIED_BLAS_SOURCE_SHA256, *CHECKER.OPERATOR_PHASE_SOURCE_SHA256, *CHECKER.GVD_SOURCE_SHA256, *CHECKER.GELS_SOURCE_SHA256, *CHECKER.GELSS_SOURCE_SHA256, *CHECKER.INJECTION_SOURCE_SHA256)
        with tempfile.TemporaryDirectory(prefix="shared-execution-source-") as directory:
            root = Path(directory)
            (root / "build.zig").write_text("pub fn build() void {}\n")
            for path in paths:
                dst = root / path
                dst.parent.mkdir(parents=True, exist_ok=True)
                dst.write_bytes((REPOSITORY_ROOT / path).read_bytes())
            CHECKER._numerics_resources_source_contract(CHECKER._make_discovery_context(root))
            for path in CHECKER.EXECUTION_SOURCE_SHA256:
                expected_owner = "public-api" if path == "src/blas/runtime.zig" else "library-source"
                self.assertEqual(expected_owner, rows[path]["owner"])
                data = (root / path).read_bytes()
                for replacement in (None, data + b"\n// source drift\n"):
                    if replacement is None:
                        (root / path).unlink()
                    else:
                        (root / path).write_bytes(replacement)
                    with self.subTest(path=path, missing=replacement is None), self.assertRaises((CHECKER.InventoryError, CHECKER.repository_snapshot.RepositorySnapshotError)):
                        CHECKER._numerics_resources_source_contract(CHECKER._make_discovery_context(root))
                (root / path).write_bytes(data)

    def test_execution_launch_barriers_dynamic_guards_and_aggregates_are_bound(self) -> None:
        observed = CHECKER._discover_build_root(REPOSITORY_ROOT, "build.zig")
        factory = next(row for row in observed if row["id"] == CHECKER.TEST_INVENTORY_FACTORY_COMPILE_ID)
        self.assertEqual(25, factory["expansion_case_count"])
        self.assertIn({"root_id": "zig-root:execution-contract-tests", "logical_compile_observation_id": "compile:build.zig:build:execution_contract_tests", "predicate_id": "predicate:always"}, factory["expansion_cases"])
        for old in ('run_execution_contract_tests.step.dependOn(test_inventory_step);',):
            with self.assertRaisesRegex(CHECKER.InventoryError, "body launch barrier"):
                CHECKER._annotate_test_inventory_factory(self.source.replace(old, '', 1), copy.deepcopy(observed))
        for old in ('run_dynamic_execution_contract_tests.step.dependOn(test_dynamic_step);',
                    'run_dynamic_execution_contract_tests.step.dependOn(&test_inventory_structure_check.step);',
                    'run_dynamic_execution_contract_tests.step.dependOn(&native_execution_check.step);',
                    'test_step.dependOn(&run_dynamic_execution_contract_tests.step);',
                    '        test_step.dependOn(&run_execution_contract_tests.step);'):
            with self.subTest(missing=old), self.assertRaisesRegex(CHECKER.InventoryError, '(dynamic correctness|default correctness aggregate)'):
                CHECKER._annotate_multiversion_contract(self.source.replace(old, '', 1), copy.deepcopy(observed))
        for edge in ('execution_contract_test_step.dependOn(&run_execution_contract_tests.step);',
                     'lapack_test_step.dependOn(execution_contract_test_step);',
                     'sparse_test_step.dependOn(execution_contract_test_step);'):
            with self.subTest(missing_focused_edge=edge), self.assertRaisesRegex(CHECKER.InventoryError, 'shared execution focused step'):
                CHECKER._numerics_resources_build_contract(self.source.replace(edge, '', 1))
        compile_row = CHECKER._new_test_inventory_observation('compile:build.zig:build:execution_contract_tests', {})
        self.assertEqual(['src/numerics/execution/owner.zig'], compile_row['root_source'])
        self.assertEqual('test-optimize', compile_row['optimize_source'])
        for module in ('lapack', 'sparse'):
            row = CHECKER._reviewed_observation_refresh_fields(f'step:build.zig:build:test-{module}')
            self.assertIn({'id': 'step:build.zig:build:test-execution-contract', 'condition': 'always'}, row['direct_dependencies'])


class WindowsBenchmarkPolicyTests(unittest.TestCase):
    """Pure policy checks; these do not claim POSIX inventory validation."""

    def _updated_contract(self) -> dict[str, Any]:
        inventory = json.loads(
            (REPOSITORY_ROOT / "tools/build_inventory.json").read_text(encoding="utf-8")
        )
        identifiers = (
            set(CHECKER.WINDOWS_BENCHMARK_PROBE_COMPILE_SOURCES)
            | CHECKER.DEFAULT_EXECUTABLE_INSTALL_IDS
        )
        for item in inventory["build_observations"]:
            if item["id"] in identifiers:
                item.update(CHECKER._reviewed_observation_refresh_fields(item["id"]))
        errors: list[str] = []
        CHECKER._validate_level2_width_and_windows_artifact_contract(inventory, errors)
        self.assertEqual([], errors)
        return inventory

    def test_windows_probe_fixture_substitution_is_rejected(self) -> None:
        inventory = self._updated_contract()
        identifier = "compile:build.zig:build:rank_k_probe"
        item = next(x for x in inventory["build_observations"] if x["id"] == identifier)
        self.assertEqual(["bench/rank_k_probe.zig"], item["root_source"])
        item["root_source_by_target"]["windows"] = CHECKER.WINDOWS_PYTHON_TOOLING_FIXTURE_PATH
        errors: list[str] = []
        CHECKER._validate_level2_width_and_windows_artifact_contract(inventory, errors)
        self.assertIn(
            f"{identifier}: Windows benchmark probe root_source_by_target changed", errors
        )

    def test_windows_default_install_exclusion_is_rejected(self) -> None:
        inventory = self._updated_contract()
        identifier = "install:build.zig:build:gemm_sweep"
        item = next(x for x in inventory["build_observations"] if x["id"] == identifier)
        item["condition"] = "requested target OS is not Windows and install step is reached"
        errors: list[str] = []
        CHECKER._validate_level2_width_and_windows_artifact_contract(inventory, errors)
        self.assertTrue(any(identifier in error and "condition" in error for error in errors))




class RuntimeLifecycleSourceContractTests(unittest.TestCase):
    """Bounded source/metadata checks; never execute the signal-owning worker."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.source = (REPOSITORY_ROOT / "build.zig").read_text(encoding="utf-8")
        cls.support = {path: (REPOSITORY_ROOT / path).read_bytes()
                       for path in CHECKER.RUNTIME_LIFECYCLE_SOURCE_SHA256}

    def _projection(self) -> dict[str, Any]:
        build = [{"id": identifier, **copy.deepcopy(template)}
                 for identifier, template in CHECKER.RUNTIME_LIFECYCLE_OBSERVATION_TEMPLATES.items()]
        build.append({"id": CHECKER.TEST_INVENTORY_AGGREGATE_STEP_ID,
                      "direct_dependencies": [{"id": CHECKER.RUNTIME_LIFECYCLE_STEP_ID,
                                               "condition": CHECKER.RUNTIME_LIFECYCLE_CONDITION}]})
        return {"build_observations": build,
                "python_launches": [{"id": CHECKER.RUNTIME_LIFECYCLE_PYTHON_LAUNCH_ID,
                                     "anchor": {"file": "test/runtime/run_blas_io_lifecycle.py"},
                                     **copy.deepcopy(CHECKER.RUNTIME_LIFECYCLE_PYTHON_TEMPLATE)}]}

    def test_runtime_lifecycle_exact_source_and_graph_contract(self) -> None:
        start, end = CHECKER._runtime_lifecycle_build_span(self.source)
        self.assertIn('b.path("test/runtime/blas_io_lifecycle.c")', self.source[start:end])
        CHECKER._runtime_lifecycle_build_span(self.source[:start] + "// harmless comment\n" + self.source[start:])
        with mock.patch.object(CHECKER, "_frozen_regular_bytes", side_effect=lambda active, path, subject: self.support[path]):
            CHECKER._runtime_lifecycle_source_contract(None)
        errors: list[str] = []
        CHECKER._validate_runtime_lifecycle_contract(self._projection(), errors)
        self.assertEqual([], errors)
        controller = self.support["test/runtime/run_blas_io_lifecycle.py"].decode()
        CHECKER._runtime_lifecycle_python_tree_contract(CHECKER.ast.parse(controller))
        controller_mutations = (
            ("ordinary-persistent", "-c"),
            ('CASES = (', 'CASES = ("-c",'),
            ("for case in CASES:", "for case in sys.argv[2:]:"),
            ("[str(probe), case]", "[case, str(probe)]"),
            ("Path(sys.argv[1])", "Path(os.environ[\"PROBE\"])"),
            ("    env = dict(os.environ", "    subprocess.run([str(probe), '-c'])\n    env = dict(os.environ"),
        )
        for before, after in controller_mutations:
            with self.subTest(controller_ast_mutation=before):
                self.assertIn(before, controller)
                with self.assertRaisesRegex(CHECKER.InventoryError, "complete controller AST"):
                    CHECKER._runtime_lifecycle_python_tree_contract(CHECKER.ast.parse(controller.replace(before, after, 1)))
        for path in self.support:
            self.assertTrue(CHECKER._requires_cached_bytes(path, None))
        for identifier in CHECKER.RUNTIME_LIFECYCLE_OBSERVATION_TEMPLATES:
            self.assertEqual(CHECKER.RUNTIME_LIFECYCLE_OBSERVATION_TEMPLATES[identifier],
                             CHECKER._new_test_inventory_observation(identifier, {}))
        self.assertEqual(CHECKER.RUNTIME_LIFECYCLE_PYTHON_TEMPLATE,
                         CHECKER._new_test_inventory_python_launch(CHECKER.RUNTIME_LIFECYCLE_PYTHON_LAUNCH_ID))
        with self.assertRaisesRegex(CHECKER.InventoryError, "no reviewed template"):
            CHECKER._new_test_inventory_python_launch(CHECKER.RUNTIME_LIFECYCLE_PYTHON_LAUNCH_ID.replace(":1", ":2"))
        with self.assertRaisesRegex(CHECKER.InventoryError, "no reviewed template"):
            CHECKER._new_test_inventory_observation(CHECKER.RUNTIME_LIFECYCLE_COMPILE_ID + "_other", {})
        migrated = json.loads((REPOSITORY_ROOT / "tools/build_inventory.json").read_text(encoding="utf-8"))
        projection = self._projection()
        for section in ("build_observations", "python_launches"):
            existing_ids = {row["id"] for row in migrated[section]}
            migrated[section].extend(row for row in projection[section] if row["id"] not in existing_ids)
        for section in ("build_observations", "python_launches"):
            for row in migrated[section]:
                row.update(CHECKER._reviewed_observation_refresh_fields(row["id"]))
        CHECKER._apply_reviewed_build_inventory_migrations(_with_source_reviewed_execution_compile(migrated))
        CHECKER._apply_reviewed_build_inventory_migrations(_with_source_reviewed_execution_compile(migrated))
        migration_errors: list[str] = []
        CHECKER._validate_runtime_lifecycle_contract(migrated, migration_errors)
        self.assertEqual([], migration_errors)
        self.assertEqual(25, len(CHECKER.REQUIRED_TEST_INVENTORY_FACTORY_CASES))
        self.assertNotIn(CHECKER.RUNTIME_LIFECYCLE_COMPILE_ID,
                         {row[1] for row in CHECKER.REQUIRED_TEST_INVENTORY_FACTORY_CASES})

    def test_runtime_lifecycle_material_build_mutations_fail_closed(self) -> None:
        start, end = CHECKER._runtime_lifecycle_build_span(self.source)
        block = self.source[start:end]
        mutations = (
            ('.cpu_arch = .aarch64', '.cpu_arch = .x86_64'),
            ('.os_tag = .macos', '.os_tag = .linux'),
            ('.abi = .none', '.abi = .gnu'),
            ('.cpu_model = .baseline', '.cpu_model = .native'),
            ('.optimize = test_optimize', '.optimize = optimize'),
            ('.link_libc = true', '.link_libc = false'),
            ('"dynamic_dispatch", false', '"dynamic_dispatch", true'),
            ('"thread_limit", 2', '"thread_limit", 3'),
            ('b.path("src/blas/runtime_lifecycle_probe.zig")', 'b.path("src/zynum.zig")'),
            ('b.path("test/runtime/blas_io_lifecycle.c")', 'b.path("test/runtime/other.c")'),
            ('"-Werror"', '"-Wno-error"'),
            ('runtime_lifecycle_options_mod);', 'zynum_build_options_mod);'),
            ('runtime_lifecycle_probe.getEmittedBin()', 'lib.getEmittedBin()'),
            ('b.graph.host.result.os.tag == .macos', 'true'),
            ('target.result.ofmt == .macho', 'true'),
            ('&native_execution_check.step', '&test_inventory_structure_check.step'),
            ('&test_inventory_structure_check.step', '&native_execution_check.step'),
            ('test_runtime_lifecycle_step.dependOn(&unsupported_runtime_lifecycle.step);', ''),
            ('b.addFail("test-runtime-lifecycle requires', 'b.addFail("unsupported lifecycle'),
        )
        for before, after in mutations:
            with self.subTest(build_mutation=before):
                expected_count = 2 if before in {".optimize = test_optimize", ".link_libc = true", "runtime_lifecycle_options_mod);"} else 1
                self.assertEqual(expected_count, block.count(before))
                positions = [match.start() for match in CHECKER.re.finditer(CHECKER.re.escape(before), block)]
                for position in positions:
                    changed_block = block[:position] + after + block[position + len(before):]
                    changed = self.source[:start] + changed_block + self.source[end:]
                    with self.assertRaisesRegex(CHECKER.InventoryError, "runtime lifecycle"):
                        CHECKER._runtime_lifecycle_build_span(changed)
        aggregate = 'if (runtime_lifecycle_native_supported) test_step.dependOn(test_runtime_lifecycle_step);'
        for changed in (
            self.source.replace(aggregate, 'test_step.dependOn(test_runtime_lifecycle_step);'),
            self.source.replace(aggregate, aggregate + '\n' + aggregate),
            self.source + '\nruntime_lifecycle_probe.root_module.addObject(lib);\n',
            self.source + '\nb.installArtifact(runtime_lifecycle_probe);\n',
            self.source + '\nrun_runtime_lifecycle_tests.setCwd(b.path("elsewhere"));\n',
        ):
            with self.subTest(extra_binding=changed[-100:]):
                with self.assertRaisesRegex(CHECKER.InventoryError, "runtime lifecycle"):
                    CHECKER._runtime_lifecycle_build_span(changed)

    def test_runtime_lifecycle_unsupported_failure_is_named_step_guard(self) -> None:
        value = self._projection()
        rows = {row["id"]: row for row in value["build_observations"]}
        failure = rows[CHECKER.RUNTIME_LIFECYCLE_FAILURE_ID]
        self.assertEqual("conditional-named-step-guard", failure["aggregate_test_membership"])
        self.assertEqual("not(runtime_lifecycle_native_supported)", failure["aggregate_condition"])
        self.assertEqual(
            "fail-closed dependency for explicit test-runtime-lifecycle requests on unsupported targets",
            failure["orphan_reason"],
        )
        self.assertFalse(failure["intentional_orphan"])
        self.assertFalse(failure["native_inventory_evidence"])
        self.assertEqual([], failure["direct_dependencies"])
        self.assertEqual(
            [{"id": CHECKER.RUNTIME_LIFECYCLE_STEP_ID,
              "condition": "runtime_lifecycle_native_supported"}],
            rows[CHECKER.TEST_INVENTORY_AGGREGATE_STEP_ID]["direct_dependencies"],
        )
        self.assertIn(
            {"id": CHECKER.RUNTIME_LIFECYCLE_FAILURE_ID,
             "condition": "not (runtime_lifecycle_native_supported)"},
            rows[CHECKER.RUNTIME_LIFECYCLE_STEP_ID]["direct_dependencies"],
        )
        for changes in (
            {"aggregate_test_membership": "conditional-member"},
            {"aggregate_condition": "runtime_lifecycle_native_supported"},
            {"aggregate_test_membership": "conditional-member",
             "aggregate_condition": "runtime_lifecycle_native_supported"},
        ):
            with self.subTest(unsupported_failure_drift=changes):
                changed = copy.deepcopy(value)
                next(row for row in changed["build_observations"]
                     if row["id"] == CHECKER.RUNTIME_LIFECYCLE_FAILURE_ID).update(changes)
                errors: list[str] = []
                CHECKER._validate_runtime_lifecycle_contract(changed, errors)
                self.assertEqual(
                    {f"{CHECKER.RUNTIME_LIFECYCLE_FAILURE_ID}: runtime lifecycle {field} changed"
                     for field in changes},
                    set(errors),
                )

    def test_runtime_lifecycle_support_and_projection_mutations_fail_closed(self) -> None:
        script = "test/runtime/run_blas_io_lifecycle.py"
        source_mutations = (
            (script, b'timeout=15', b'timeout=30'),
            (script, b'for case in CASES:', b'for case in CASES[:1]:'),
            (script, b'ZYNUM_MAXIMUM_THREADS="2"', b'ZYNUM_MAXIMUM_THREADS="3"'),
            (script, b'result.stdout != expected or result.stderr', b'False'),
            (script, b'[str(probe), case]', b'["other", case]'),
            (script, b'platform.machine() != "arm64"', b'False'),
            (script, b'"ordinary-persistent",', b'"other",'),
            ("test/runtime/blas_io_lifecycle.c", b'a->sa_flags != b->sa_flags', b'0'),
            ("test/runtime/blas_io_lifecycle.c", b'if (work)', b'if (0)'),
            ("src/blas/runtime_lifecycle_probe.zig", b'if (!accepted) return 2;', b''),
            ("src/blas/runtime_lifecycle_probe.zig", b'if (!work.helper_observed.load(.acquire)) return 4;', b''),
        )
        for path, before, after in source_mutations:
            with self.subTest(support_mutation=(path, before)):
                self.assertIn(before, self.support[path])
                changed = {**self.support, path: self.support[path].replace(before, after, 1)}
                with mock.patch.object(CHECKER, "_frozen_regular_bytes", side_effect=lambda active, rel, subject: changed[rel]):
                    with self.assertRaisesRegex(CHECKER.InventoryError, "runtime lifecycle support"):
                        CHECKER._runtime_lifecycle_source_contract(None)
        for mutation in ("missing-compile", "duplicate-child", "timeout", "guards", "kernel-objects", "native-evidence", "aggregate"):
            with self.subTest(projection_mutation=mutation):
                value = self._projection()
                rows = {row["id"]: row for row in value["build_observations"]}
                if mutation == "missing-compile":
                    value["build_observations"].remove(rows[CHECKER.RUNTIME_LIFECYCLE_COMPILE_ID])
                elif mutation == "duplicate-child":
                    value["python_launches"].append(copy.deepcopy(value["python_launches"][0]))
                elif mutation == "timeout":
                    value["python_launches"][0]["timeout_seconds"] = None
                elif mutation == "guards":
                    rows[CHECKER.RUNTIME_LIFECYCLE_LAUNCH_ID]["guard_dependencies"].pop()
                elif mutation == "kernel-objects":
                    rows[CHECKER.RUNTIME_LIFECYCLE_COMPILE_ID]["kernel_objects"] = ["kernel_test_libraries"]
                elif mutation == "native-evidence":
                    rows[CHECKER.RUNTIME_LIFECYCLE_COMPILE_ID]["native_inventory_evidence"] = True
                else:
                    rows[CHECKER.TEST_INVENTORY_AGGREGATE_STEP_ID]["direct_dependencies"][0]["condition"] = "always"
                errors: list[str] = []
                CHECKER._validate_runtime_lifecycle_contract(value, errors)
                self.assertTrue(errors)
                self.assertTrue(all("runtime lifecycle" in error for error in errors))


class CheckedAbiSourceContractTests(unittest.TestCase):
    """The real checked graph has one closed, narrowly reviewed source contract."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.source = (REPOSITORY_ROOT / "build.zig").read_text(encoding="utf-8")

    def test_checked_exact_graph_and_run_receiver_projection(self) -> None:
        start, end = CHECKER._checked_abi_build_span(self.source)
        self.assertIn("const install_checked_abi", self.source[start:end])
        self.assertEqual(1, len(CHECKER._checked_abi_receiver_uses(self.source)))
        CHECKER._zig_build_context(self.source, "build.zig")
        decorated = self.source[:start] + "// harmless graph comment\n" + self.source[start:]
        CHECKER._checked_abi_build_span(decorated)

    def test_checked_material_build_mutations_fail_closed(self) -> None:
        mutations = (
            ('"checked-abi", "Include', '"checked-api", "Include'),
            ('"checked-abi", "Include the experimental checked ABI in default installation") orelse false', '"checked-abi", "Include the experimental checked ABI in default installation") orelse true'),
            ('checked_dynamic_lib.dll_export_fns = false', 'checked_dynamic_lib.dll_export_fns = true'),
            ('checked_test_lib.dll_export_fns = false', 'checked_test_lib.dll_export_fns = true'),
            ('b.path("src/checked/exports.def")', 'b.path("src/blas/exports.def")'),
            ('b.path("src/checked.zig")', 'b.path("src/zynum.zig")'),
            ('.root_module = checked_abi_mod', '.root_module = zynum_mod'),
            ('.root_module = checked_abi_test_mod', '.root_module = checked_abi_mod'),
            ('const checked_abi_test_mod = b.createModule(.{\n        .root_source_file = b.path("src/checked.zig"),\n        .target = target,\n        .optimize = test_optimize', 'const checked_abi_test_mod = b.createModule(.{\n        .root_source_file = b.path("src/checked.zig"),\n        .target = target,\n        .optimize = optimize'),
            ('"-std=c11", "-Wall", "-Wextra", "-Werror", "-pedantic"', '"-std=c11", "-Wall", "-Wextra", "-pedantic"'),
            ('"-std=c++20"', '"-std=c++17"'),
            ('b.path("test/headers/checked_c_smoke.c")', 'b.path("test/headers/other.c")'),
            ('if (install_checked_abi) b.getInstallStep().dependOn(install_checked_abi_step);', 'b.getInstallStep().dependOn(install_checked_abi_step);'),
            ('.custom = "lib/static"', '.custom = "lib"'),
            ('check_checked_abi_step.dependOn(&check_checked_abi_generated.step);', ''),
            ('check_checked_abi_step.dependOn(&checked_abi_exports.step);', ''),
            ('checked_abi_exports.addArg("--static");', 'checked_abi_exports.addArg("--skip-static");'),
            ('run_checked_c_smoke.step.dependOn(checked_native_guard);', ''),
            ('run_checked_cpp_smoke.setCwd(checked_test_lib.getEmittedBinDirectory());', ''),
            ('checked_c_smoke.root_module.addRPath(checked_test_lib.getEmittedBinDirectory());', ''),
            ('checked_required_cpu_features.removeFeature(@backingInt(std.Target.aarch64.Feature.store_pair_suppress));', 'checked_required_cpu_features.removeFeature(@backingInt(std.Target.aarch64.Feature.sve));'),
            ('else\n        &native_execution_check.step;\n    const run_checked_c_smoke', 'else\n        &check_checked_abi_generated.step;\n    const run_checked_c_smoke'),
            ('"-fcheck=all"', '"-fcheck=none"'),
            ('build_checked_fortran_smoke.step.dependOn(checked_native_guard);', ''),
            ('target.result.os.version_range.semver.min', 'b.graph.host.result.os.version_range.semver.min'),
            ('addOutputDirectoryArg2("checked-fortran-modules", .{ .prefix = "-J" })', 'addOutputDirectoryArg2("checked-fortran-modules", .{ .prefix = "-I" })'),
            ('.prefix = "-Wl,-rpath,", .make_absolute = true', '.prefix = "-Wl,-rpath,", .make_absolute = false'),
            ('run_checked_fortran_smoke.addFileArg(checked_fortran_exe);', 'run_checked_fortran_smoke.addArg("unreviewed-executable");'),
            ('std.Build.Step.Run.create(b, "run checked Fortran consumer")', 'std.Build.Step.Run.create(b, "different consumer")'),
        )
        start, end = CHECKER._checked_abi_build_span(self.source)
        block = self.source[start:end]
        for before, after in mutations:
            with self.subTest(before=before):
                self.assertIn(before, block)
                changed = self.source[:start] + block.replace(before, after, 1) + self.source[end:]
                with self.assertRaisesRegex(CHECKER.InventoryError, "checked ABI"):
                    CHECKER._checked_abi_build_span(changed)

    def test_checked_owned_binding_and_extra_run_create_rejected(self) -> None:
        _, end = CHECKER._checked_abi_build_span(self.source)
        escaped = self.source[:end] + "\nconst escaped = checked_abi_mod;\n" + self.source[end:]
        with self.assertRaisesRegex(CHECKER.InventoryError, "binding escapes"):
            CHECKER._checked_abi_build_span(escaped)
        extra = self.source[:end] + '\nconst another = std.Build.Step.Run.create(b, "extra");\n' + self.source[end:]
        with self.assertRaisesRegex(CHECKER.InventoryError, "exactly one"):
            CHECKER._checked_abi_receiver_uses(extra)

    def test_checked_generator_schema_and_single_owner_sources_are_frozen(self) -> None:
        with tempfile.TemporaryDirectory(prefix="checked-abi-closure-") as temporary:
            root = Path(temporary)
            for path in ("build.zig", "build.zig.zon", *CHECKER.CHECKED_ABI_SOURCE_SHA256):
                destination = root / path
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(REPOSITORY_ROOT / path, destination)
            CHECKER._checked_abi_source_contract(CHECKER._make_discovery_context(root))
            for path in CHECKER.CHECKED_ABI_SOURCE_SHA256:
                destination = root / path
                before = destination.read_bytes()
                destination.write_bytes(before + b"\n// changed ownership or schema\n")
                with self.subTest(path=path), self.assertRaisesRegex(CHECKER.InventoryError, "checked ABI source/ownership/schema"):
                    CHECKER._checked_abi_source_contract(CHECKER._make_discovery_context(root))
                destination.write_bytes(before)

    def test_checked_formatter_and_selection_source_mutations_fail_closed(self) -> None:
        support = {path: (REPOSITORY_ROOT / path).read_bytes()
                   for path in CHECKER.CHECKED_ABI_SOURCE_SHA256}
        path = "tools/generate_checked_abi.py"
        mutations = (
            (b'[zig, "fmt", "--stdin"]', b'[zig, "--stdin", "fmt"]'),
            (b'[zig, "fmt", "--stdin"]', b'[zig, "run", "--stdin"]'),
            (b'selected = select_profile(profile)', b'selected = select_profile()'),
            (b'outputs(args.zig, args.profile)', b'outputs(args.zig)'),
            (b"[zig, 'fmt', '--stdin']", b"[zig, 'run', '--stdin']"),
            (b'plan = select_binding_set(profile, binding_set, target_contract)',
             b'plan = select_binding_set(profile, binding_set, "unreviewed-target")'),
            (b'    plan = select_binding_set(profile, binding_set, target_contract)',
             b'    subprocess.run([zig, "version"])\n    plan = select_binding_set(profile, binding_set, target_contract)'),
            (b'    selected = select_profile(profile)',
             b'    subprocess.run([zig, "version"])\n    selected = select_profile(profile)'),
        )
        for before, after in mutations:
            with self.subTest(mutation=before):
                self.assertIn(before, support[path])
                changed = {**support, path: support[path].replace(before, after, 1)}
                with mock.patch.object(CHECKER, "_frozen_regular_bytes",
                                       side_effect=lambda active, rel, subject: changed[rel]):
                    with self.assertRaisesRegex(CHECKER.InventoryError, "checked ABI source/ownership/schema"):
                        CHECKER._checked_abi_source_contract(None)
        standard_path = "tools/standard_abi_bindings.py"
        changed = {**support, standard_path: support[standard_path].replace(
            b"GENERATION_ACCEPTANCES = MappingProxyType({",
            b"GENERATION_ACCEPTANCES = MappingProxyType({'unreviewed': {}, ", 1)}
        self.assertNotEqual(support[standard_path], changed[standard_path])
        with mock.patch.object(CHECKER, "_frozen_regular_bytes",
                               side_effect=lambda active, rel, subject: changed[rel]):
            with self.assertRaisesRegex(CHECKER.InventoryError, "checked ABI source/ownership/schema"):
                CHECKER._checked_abi_source_contract(None)
        identifier = "python-launch:tools/generate_checked_abi.py:_render_subset:subprocess.run:1"
        self.assertEqual({"owner": "abi-compatibility",
                          "argv_shape": ["<explicit Zig formatter>", "fmt", "--stdin"],
                          "launch_class": "finite-standard-abi-generated-source-formatter",
                          "detail_status": "process-lifecycle-out-of-scope",
                          "compile_for": "host", "execute_on": "host", "cwd_shape": "repository-root"},
                         CHECKER._new_test_inventory_python_launch(identifier))
        for unknown in (identifier.replace(":1", ":2"), identifier.replace(":_render_subset:", ":unreviewed_outputs:")):
            with self.subTest(unknown_formatter=unknown), self.assertRaisesRegex(CHECKER.InventoryError, "no reviewed"):
                CHECKER._new_test_inventory_python_launch(unknown)

    def test_checked_exact_classification_and_output_sets(self) -> None:
        paths = [row["path"] for row in CHECKER.CHECKED_ABI_CLASSIFICATIONS]
        self.assertEqual(32, len(paths))
        self.assertEqual(32, len(set(paths)))
        self.assertEqual({"class": "non-generated-source", "kind": "python-source",
                          "owner": "developer-tooling", "path": "tools/standard_abi_bindings.py"},
                         next(row for row in CHECKER.CHECKED_ABI_CLASSIFICATIONS
                              if row["path"] == "tools/standard_abi_bindings.py"))
        self.assertIn("tools/standard_abi_bindings.py", CHECKER.CHECKED_ABI_SOURCE_SHA256)
        self.assertTrue(CHECKER._requires_cached_bytes("tools/standard_abi_bindings.py", None))
        self.assertEqual(6, len(CHECKER.CHECKED_ABI_OUTPUTS))
        generated = {row["path"] for row in CHECKER.CHECKED_ABI_CLASSIFICATIONS if row["class"] == "designated-reproducible-contract-reference"}
        # This historical checked-classification registry includes the two
        # Sparse product sets. The later triangular solves have their own
        # classifications; they are not additional checked-ABI outputs.
        sparse_products = {
            f"interface_subset/mkl2026.{width}/sparse-csr-csc-mv-mm-ordinary-{width}-v1/{name}"
            for width in ("lp64", "ilp64")
            for name in ("subset.h", "exports.zig", "abi_manifest.json", "exports.def")
        }
        self.assertEqual(8, len(sparse_products))
        self.assertLessEqual(sparse_products, {row["path"] for row in CHECKER.FINITE_SPARSE_DERIVED_CANDIDATES})
        self.assertEqual(set(CHECKER.CHECKED_ABI_OUTPUTS) | sparse_products, generated)
        self.assertEqual(set(CHECKER.CHECKED_ABI_OUTPUTS), {row["path"] for row in CHECKER.CHECKED_ABI_DERIVED_CANDIDATES})
        expected_inputs = ["tools/numerical_operation_schema.py", "tools/checked_abi_profile.py",
                           "tools/checked_abi_schema.py", "tools/generate_checked_abi.py",
                           "src/checked/implementation.zig", "build.zig.zon"]
        for row in CHECKER.CHECKED_ABI_DERIVED_CANDIDATES:
            self.assertEqual(expected_inputs, row["authoritative_inputs"])
        for path in expected_inputs[:-1]:
            self.assertIn(path, CHECKER.CHECKED_ABI_SOURCE_SHA256)
            self.assertTrue(CHECKER._requires_cached_bytes(path, None))
        migrated = json.loads((REPOSITORY_ROOT / "tools/build_inventory.json").read_text())
        CHECKER._apply_reviewed_build_inventory_migrations(_with_source_reviewed_execution_compile(migrated))
        self.assertEqual(CHECKER.REQUIRED_SECTION_FACT_DIGESTS["derived_candidates"],
                         CHECKER._json_fact_digest(migrated["derived_candidates"]))
        for row in migrated["derived_candidates"]:
            if row["path"] not in CHECKER.CHECKED_ABI_OUTPUTS:
                continue
            for missing in expected_inputs:
                with self.subTest(output=row["path"], missing_input=missing):
                    changed = copy.deepcopy(migrated["derived_candidates"])
                    target = next(item for item in changed if item["id"] == row["id"])
                    target["authoritative_inputs"].remove(missing)
                    self.assertNotEqual(CHECKER.REQUIRED_SECTION_FACT_DIGESTS["derived_candidates"],
                                        CHECKER._json_fact_digest(changed))
        self.assertIn("src/checked.zig", paths)
        package = CHECKER.CHECKED_ABI_PACKAGE_CANDIDATE
        self.assertEqual("pkgconfig/zynum_checked.pc", package["path"])
        self.assertEqual("non-generated-source", package["class"])
        self.assertNotIn(package["path"], generated)
        self.assertIn("checked-abi is true", package["install_disposition"])

    def test_checked_workflow_projection_binds_material_changes(self) -> None:
        with tempfile.TemporaryDirectory(prefix="checked-workflow-closure-") as temporary:
            root = Path(temporary)
            paths = ("build.zig", "build.zig.zon", ".github/workflows/ci.yml", ".github/workflows/release.yml")
            for path in paths:
                destination = root / path
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(REPOSITORY_ROOT / path, destination)
            observed = CHECKER._discover_workflow_launches(root)
            by_id = {row["id"]: row for row in observed}
            prefix = "workflow-launch:.github/workflows/ci.yml:target-tests:"
            self.assertEqual("runner.os != 'Windows'", by_id[prefix + "run-checked-c-and-c-consumers"]["condition"])
            self.assertEqual("matrix.full_smoke", by_id[prefix + "run-checked-fortran-consumer"]["condition"])
            self.assertFalse(by_id[prefix + "run-checked-fortran-consumer"]["native_inventory_evidence"])
            drift = by_id["workflow-launch:.github/workflows/ci.yml:source-checks:check-generated-files-are-up-to-date"]
            self.assertIn("src/checked/exports.def", drift["argv_shape"][0])
            baseline = {row["id"]: row["source_digest"] for row in observed}
            ci = root / ".github/workflows/ci.yml"
            before = ci.read_text()
            start = before.index("      - name: Run checked C and C++ consumers")
            end = before.index("      - name: Check ABI symbols", start)
            block = before[start:end]
            mutations = (
                ("if: runner.os != 'Windows'", "if: always()"),
                ("if: matrix.full_smoke", "if: always()"),
                ("GITHUB_ENV=/dev/null", "GITHUB_ENV=/tmp/unchecked"),
                ("PATH: ${{ steps.trusted-posix-tools.outputs.path }}", "PATH: /tmp/unchecked"),
                ("-Dtest-optimize=safe", "-Dtest-optimize=fast"),
                ('-Dchecked-gfortran="${{ steps.trusted-posix-tools.outputs.gfortran }}"', '-Dchecked-gfortran="gfortran"'),
            )
            for original, replacement in mutations:
                with self.subTest(original=original):
                    self.assertIn(original, block)
                    ci.write_text(before[:start] + block.replace(original, replacement, 1) + before[end:])
                    changed = {row["id"]: row["source_digest"] for row in CHECKER._discover_workflow_launches(root)}
                    self.assertNotEqual(baseline, changed)
            ci.write_text(before)
            release = root / ".github/workflows/release.yml"
            release_before = release.read_text()
            for original, replacement in (("-Dchecked-abi=true", "-Dchecked-abi=false"), ("zig build check-checked-abi", "zig build build-checked-abi")):
                with self.subTest(original=original):
                    self.assertIn(original, release_before)
                    release.write_text(release_before.replace(original, replacement, 1))
                    changed = {row["id"]: row["source_digest"] for row in CHECKER._discover_workflow_launches(root)}
                    self.assertNotEqual(baseline, changed)
            release.write_text(release_before)

    def test_checked_native_projection_preserves_existing_guard_rejection(self) -> None:
        start, end = CHECKER._checked_abi_build_span(self.source)
        native = self.source[:start] + self.source[end:]
        inventory = json.loads((REPOSITORY_ROOT / "tools/build_inventory.json").read_text())
        CHECKER._annotate_native_feature_test_contract(native, copy.deepcopy(inventory["build_observations"]))
        mutations = (
            ('else if (!b.graph.host.result.cpu.features.isSuperSetOf(target.result.cpu.features))', 'else if (false)'),
            ('native_execution_check.addThirdPartyEnabledArgQemu(.{ .enabled = "--native-executor=qemu" });', ''),
            ('target.result.abi == b.graph.host.result.abi and', 'true and'),
        )
        for before, after in mutations:
            self.assertIn(before, native)
            with self.subTest(before=before), self.assertRaises(CHECKER.InventoryError):
                CHECKER._annotate_native_feature_test_contract(native.replace(before, after, 1), copy.deepcopy(inventory["build_observations"]))

class CheckedAbiProfileGenerationTests(unittest.TestCase):
    """Exercise checked input selection without compiler or native ABI claims."""

    @classmethod
    def setUpClass(cls) -> None:
        # The normal generator uses sibling imports. Keep that search-path change
        # local to this import, with no private fixture or generated baseline.
        with mock.patch.object(sys, "path", [str(REPOSITORY_ROOT / "tools"), *sys.path]), \
             mock.patch.object(sys, "dont_write_bytecode", True):
            import checked_abi_schema
            import generate_checked_abi
            import numerical_operation_schema
        cls.schema = checked_abi_schema
        cls.generator = generate_checked_abi
        cls.core = numerical_operation_schema

    def test_default_and_explicit_checked_inputs_preserve_consumer_contract(self):
        selected = self.schema.select_profile()
        self.assertEqual(self.schema.select_profile("checked-v1"), selected)
        for name in self.schema.LEGACY_FIELDS:
            self.assertEqual(getattr(self.schema, name), selected[name])
            self.assertIs(type(getattr(self.schema, name)), type(selected[name]))
        self.assertEqual(8, len(selected["FUNCTIONS"]))
        self.assertEqual(self.schema.CHECKED_V1_CONTRACT_SHA256,
                         hashlib.sha256(json.dumps(selected, separators=(",", ":")).encode()).hexdigest())
        # A formatter-shaped mock checks its argv only; it is not a Zig build,
        # formatter execution, or native symbol/layout observation.
        with mock.patch.object(self.generator.subprocess, "run") as formatter:
            formatter.side_effect = lambda *args, **kwargs: mock.Mock(stdout=kwargs["input"])
            default = self.generator.outputs("formatter-must-not-run")
            explicit = self.generator.outputs("formatter-must-not-run", "checked-v1")
            self.assertEqual(default, explicit)
            self.assertEqual(set(CHECKER.CHECKED_ABI_OUTPUTS), set(default))
            self.assertEqual(4, formatter.call_count)  # Two Zig files per selection.
            for call in formatter.call_args_list:
                self.assertEqual((["formatter-must-not-run", "fmt", "--stdin"],), call.args)
                self.assertEqual({"input", "text", "capture_output", "check"}, set(call.kwargs))
                self.assertIsInstance(call.kwargs["input"], str)
                for flag in ("text", "capture_output", "check"):
                    self.assertIs(True, call.kwargs[flag])
        for path, text in default.items():
            if not path.endswith(".zig"):
                self.assertEqual((REPOSITORY_ROOT / path).read_text(), text)

    def assert_early_rejection(self, profile_id='checked-v1', profile=None, message=None):
        registry = dict(self.schema.PROFILES)
        if profile is not None:
            registry[profile_id] = profile
        with mock.patch.object(self.schema, 'PROFILES', registry), \
             mock.patch.object(self.generator.subprocess, 'run') as formatter, \
             mock.patch.object(Path, 'write_text') as write, \
             mock.patch.object(Path, 'mkdir') as mkdir:
            with self.assertRaises(self.schema.ProfileError) as error:
                self.generator.outputs('formatter-must-not-run', profile_id)
            if message:
                self.assertIn(message, str(error.exception))
            formatter.assert_not_called()
            write.assert_not_called()
            mkdir.assert_not_called()
            with mock.patch.object(sys, 'argv', ['generator', '--profile', str(profile_id), '--root', 'must-not-exist']), \
                 contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as exit:
                    self.generator.main()
                self.assertEqual(2, exit.exception.code)
            formatter.assert_not_called()
            write.assert_not_called()
            mkdir.assert_not_called()

    def test_common_math_and_checked_wire_inputs_are_separate_and_consumed(self):
        self.assertEqual({'cholesky.factor', 'cholesky.solve', 'lu.factor', 'lu.solve', 'lu.driver',
                          'tridiagonal.eigen', 'least_squares.rank_revealing', 'hermitian.eigen',
                          'hermitian.generalized_eigen', 'symmetric.eigen', 'symmetric.generalized_eigen', 'sparse.mv', 'sparse.mm',
                          'qr.factor', 'qr.apply_real', 'qr.apply_complex',
                          'qr.generate_real', 'qr.generate_complex',
                          'least_squares.rank_revealing_complex', 'svd.decompose',
                          'cholesky.driver', 'triangular.solve', 'triangular.inverse', 'lu.inverse', 'cholesky.inverse',
                          'sparse.trsv', 'sparse.trsm'}, set(self.core.OPERATIONS))
        self.assertTrue(all(set(op) == {'scalar_types', 'index_types', 'operand_roles', 'contract'} for op in self.core.OPERATIONS.values()))
        changed = copy.deepcopy(self.core.OPERATIONS)
        changed['cholesky.factor']['scalar_types'] = ('f64',)
        with mock.patch.object(self.schema, 'OPERATIONS', changed):
            self.assert_early_rejection(message='Unsupported scalar binding')
        changed = copy.deepcopy(self.core.OPERATIONS)
        changed['cholesky.factor']['operand_roles'] = ('triangle', 'matrix')
        with mock.patch.object(self.schema, 'OPERATIONS', changed):
            self.assert_early_rejection(message='operand roles mismatch')

    def test_unknown_and_unready_profiles_reject_before_formatter_or_writes(self):
        for name in ('unknown-vendor', 'mkl2026.lp64', 'mkl2026.ilp64'):
            with self.subTest(profile=name):
                self.assert_early_rejection(name)
        profile = copy.deepcopy(self.schema.PROFILES['checked-v1'])
        profile['generation_ready'] = False
        self.assert_early_rejection(profile=profile, message='not ready')
        profile['generation_ready'] = True
        profile['unresolved_generation_inputs'] = ('missing reviewed layout',)
        self.assert_early_rejection(profile=profile, message='not ready')

    def test_profile_identity_and_incomplete_input_are_not_fallbacks(self):
        profile = copy.deepcopy(self.schema.PROFILES['checked-v1'])
        profile['id'] = 'other-profile'
        self.assert_early_rejection(profile=profile, message='selection/identity mismatch')
        profile = copy.deepcopy(self.schema.PROFILES['checked-v1'])
        del profile['STATUS']
        self.assert_early_rejection(profile=profile, message='incomplete')
        profile = copy.deepcopy(self.schema.PROFILES['checked-v1'])
        profile['renderer'] = 'lapack-standard'
        self.assert_early_rejection(profile=profile, message='No reviewed renderer')

    def test_binding_ids_are_opaque_profile_field_controls_membership(self):
        expected = self.schema.select_profile()
        profile = copy.deepcopy(self.schema.PROFILES['checked-v1'])
        for i, binding in enumerate(profile['bindings']):
            binding['id'] = 'mkl2026.lp64:misleading-' + str(i)
        with mock.patch.dict(self.schema.PROFILES, {'checked-v1': profile}):
            self.assertEqual(expected, self.schema.select_profile())
        profile['bindings'][0]['id'] = 'checked-v1:apparently-selected'
        profile['bindings'][0]['profile'] = 'mkl2026.lp64'
        self.assert_early_rejection(profile=profile, message='different selected profile')

    def test_operation_adapter_scalar_index_and_symbol_mismatches_reject(self):
        changes = ((0, 'operation', 'cholesky.solve'), (0, 'operation', 'unimplemented.operation'),
                   (0, 'adapter', 'solve'), (0, 'scalar', 'complex_f32'),
                   (2, 'index', 'i32'), (2, 'symbol', 'csr_spmv_f64_u32'))
        for offset, field, value in changes:
            with self.subTest(field=field, value=value):
                profile = copy.deepcopy(self.schema.PROFILES['checked-v1'])
                profile['bindings'][offset][field] = value
                self.assert_early_rejection(profile=profile)

    def test_status_layout_and_call_argument_drift_reject_before_formatter(self):
        for mutation in ('status', 'layout', 'role', 'call-name', 'type-order', 'wire-name'):
            with self.subTest(mutation=mutation):
                profile = copy.deepcopy(self.schema.PROFILES['checked-v1'])
                if mutation == 'status':
                    profile['STATUS']['NUMERICAL_FAILURE'] = 99
                elif mutation == 'layout':
                    profile['STRUCTS']['result'][2] = ('info', 'i32')
                elif mutation == 'role':
                    profile['adapters']['factor']['arguments'] = (('triangle', 'i32', 'triangle'), ('a', 'dense', 'matrix'))
                elif mutation == 'call-name':
                    profile['adapters']['factor']['arguments'] = (('a', 'dense', 'matrix'), ('wrong', 'i32', 'triangle'))
                elif mutation == 'type-order':
                    profile['adapters']['spmv']['type_arguments'] = ('index', 'scalar')
                else:
                    profile['adapters']['factor']['arguments'] = (('a', 'const_dense', 'matrix'), ('triangle', 'i32', 'triangle'))
                self.assert_early_rejection(profile=profile)

    def test_duplicate_or_missing_bindings_never_shrink_or_expand_exports(self):
        for mutation in ('duplicate-id', 'duplicate-symbol', 'remove', 'add'):
            with self.subTest(mutation=mutation):
                profile = copy.deepcopy(self.schema.PROFILES['checked-v1'])
                if mutation == 'duplicate-id':
                    profile['bindings'][1]['id'] = profile['bindings'][0]['id']
                elif mutation == 'duplicate-symbol':
                    profile['bindings'][1] = {**profile['bindings'][0], 'id': 'another-id'}
                elif mutation == 'remove':
                    profile['bindings'].pop()
                else:
                    profile['bindings'].append(copy.deepcopy(profile['bindings'][0]))
                self.assert_early_rejection(profile=profile)

    def test_adapter_entry_hash_is_a_binding_guard_not_a_runtime_claim(self):
        profile = copy.deepcopy(self.schema.PROFILES['checked-v1'])
        profile['adapter_source']['sha256'] = '0' * 64
        self.assert_early_rejection(profile=profile, message='source binding')
        with mock.patch.object(Path, 'read_bytes', return_value=b'changed checked adapter'):
            self.assert_early_rejection(message='entry source drift')

    def test_ready_bit_cannot_turn_checked_profile_into_standard_exports(self):
        profile = copy.deepcopy(self.schema.PROFILES['checked-v1'])
        profile['id'] = 'mkl2026.lp64'
        for binding in profile['bindings']:
            binding['profile'] = 'mkl2026.lp64'
        self.assert_early_rejection('mkl2026.lp64', profile, 'No reviewed renderer')
        self.assertEqual({'checked-v1'}, {name for name, p in self.schema.PROFILES.items() if p['generation_ready']})


class BuildInventoryTests(unittest.TestCase):
    maxDiff = 2000

    @classmethod
    def setUpClass(cls) -> None:
        cls.inventory = json.loads(
            (REPOSITORY_ROOT / "tools/build_inventory.json").read_text(encoding="utf-8")
        )
        files = {"tools/build_inventory.json"}
        files.update({
            "tools/check_build_inventory.py",
            "tools/repository_git.py",
            "tools/repository_snapshot.py",
            "test/headers/compat_headers.h",
        })
        files.update(row["path"] for row in CHECKER.CHECKED_ABI_CLASSIFICATIONS)
        files.update(row["path"] for row in CHECKER.ADDITIVE_SPARSE_CLASSIFICATIONS)
        files.update(CHECKER.ADDITIVE_SPARSE_GENERATOR_PINS)
        files.update(row["path"] for row in CHECKER.CHECKED_CHOLESKY_CLASSIFICATIONS)
        files.update(CHECKER.CHECKED_CHOLESKY_SOURCE_SHA256)
        files.update(CHECKER.OPERATOR_PHASE_SOURCE_SHA256)
        files.update(CHECKER.FFTW_F32_SOURCE_SHA256)
        files.update(CHECKER.FEAST_SOURCE_SHA256)
        files.update(CHECKER.CONTOUR_SOURCE_SHA256)
        files.update(CHECKER.SYEV_SOURCE_SHA256)
        files.update(CHECKER.SYGV_SOURCE_SHA256)
        files.update(CHECKER.CONTROLS_SOURCE_SHA256)
        files.update(CHECKER.EVD_SOURCE_SHA256)
        files.update(CHECKER.GVD_SOURCE_SHA256)
        files.update(CHECKER.GELS_SOURCE_SHA256)
        files.update(CHECKER.GELSS_SOURCE_SHA256)
        files.update(row["path"] for row in CHECKER.CONTROLS_COMPONENT_CLASSIFICATIONS)
        files.update(row["path"] for row in CHECKER.EVD_COMPONENT_CLASSIFICATIONS)
        files.update(row["path"] for row in CHECKER.GVD_COMPONENT_CLASSIFICATIONS)
        files.update(row["path"] for row in CHECKER.GELS_COMPONENT_CLASSIFICATIONS)
        files.update(row["path"] for row in CHECKER.GELSS_COMPONENT_CLASSIFICATIONS)
        files.update(row["path"] for row in CHECKER.SYGV_COMPONENT_CLASSIFICATIONS)
        files.update(row["path"] for row in CHECKER.SYEV_COMPONENT_CLASSIFICATIONS)
        files.update(CHECKER.POSV_SOURCE_SHA256)
        files.update(row["path"] for row in CHECKER.POSV_CLASSIFICATIONS)
        files.update(CHECKER.TRTRS_SOURCE_SHA256)
        files.update(row["path"] for row in CHECKER.TRTRS_CLASSIFICATIONS)
        files.update(CHECKER.TRTRI_SOURCE_SHA256)
        files.update(row["path"] for row in CHECKER.TRTRI_CLASSIFICATIONS)
        files.update(CHECKER.GETRI_SOURCE_SHA256)
        files.update(row["path"] for row in CHECKER.GETRI_CLASSIFICATIONS)
        files.update(row["path"] for row in CHECKER.POTRI_COMPONENT_CLASSIFICATIONS)
        files.update(CHECKER.UNIFIED_BLAS_SOURCE_SHA256)
        files.update(CHECKER.RUNTIME_LIFECYCLE_SOURCE_SHA256)
        files.add(CHECKER.NUMERICS_RESOURCES_PATH)
        files.update(CHECKER.SHARED_FACADE_SOURCE_SHA256)
        files.update(CHECKER.EXECUTION_SOURCE_SHA256)
        files.update(CHECKER.INJECTION_SOURCE_SHA256)
        files.update(CHECKER.COMMON_CORE_SOURCE_SHA256)
        files.update(CHECKER.MANAGED_LAPACKE_SOURCE_SHA256)
        files.update(CHECKER.MANAGED_SPARSE_SOURCE_SHA256)
        files.update(CHECKER.EXECUTION_OBJECT_SOURCE_SHA256)
        files.update({
            "src/abplas_operator_lp64.zig",
            "src/abplas_sparse_lp64.zig",
            "src/interop/operator_binding.zig",
            "tools/fftw_profile.json",
            "tools/generate_operator_interface.py",
            "tools/interface_templates/fftw_exports.zig.in",
        })
        files.add("src/lapack/core/cholesky_contract.zig")
        files.add(CHECKER.DEPENDABOT_CONFIG_PATH)
        files.add(CHECKER.LEVEL2_WIDTH_STUB_ROOT_PATH)
        files.update(item["path"] for item in cls.inventory.get("build_manifests", []))
        for section in (
            "build_observations",
            "python_launches",
            "workflow_launches",
            "generator_targets",
        ):
            files.update(item["anchor"]["file"] for item in cls.inventory[section])
        files.update(
            item["path"]
            for item in cls.inventory["derived_candidates"]
            if (REPOSITORY_ROOT / item["path"]).is_file()
        )
        cls.fixture_files = sorted(files)

    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(prefix="build-inventory-fixture-")
        self.root = Path(self.temporary.name)
        for relative in self.fixture_files:
            source = REPOSITORY_ROOT / relative
            destination = self.root / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
        self.inventory_path = self.root / "tools/build_inventory.json"

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _inventory(self) -> dict[str, Any]:
        return json.loads(self.inventory_path.read_text(encoding="utf-8"))

    def _write_inventory(self, inventory: dict[str, Any]) -> None:
        self.inventory_path.write_text(
            json.dumps(inventory, indent=2) + "\n", encoding="utf-8"
        )

    def _assert_binary_license_archive(self, archive_path: Path) -> None:
        expected = {
            name: (REPOSITORY_ROOT / name).read_bytes()
            for name in ("LICENSE", "COPYING")
        }
        with tarfile.open(archive_path, "r:gz") as archive:
            names = archive.getnames()
            self.assertNotIn("COPYING.LESSER", names)
            for name, expected_bytes in expected.items():
                self.assertEqual(1, names.count(name))
                member = archive.getmember(name)
                self.assertTrue(member.isfile())
                extracted = archive.extractfile(member)
                self.assertIsNotNone(extracted)
                assert extracted is not None
                self.assertEqual(expected_bytes, extracted.read())

    def _append_to_build(self, statement: str, relative: str = "build.zig") -> None:
        path = self.root / relative
        text = path.read_text(encoding="utf-8")
        body_end = _build_body_insertion_offset(text, relative)
        path.write_text(text[:body_end] + f"    {statement}\n" + text[body_end:], encoding="utf-8")

    def _assert_error_contains(self, expected: str) -> None:
        errors = CHECKER.validate(self.root, self.inventory_path)
        self.assertTrue(errors, "the mutation unexpectedly validated")
        self.assertIn(expected, "\n".join(errors))

    def _as_curated_svg_asset(self, inventory: dict[str, Any]) -> dict[str, Any]:
        # The public tree intentionally has no benchmark charts without reviewed
        # source data. Replace a required fixture with an empty synthetic SVG so
        # the test covers the SVG policy without asserting benchmark results.
        source_candidate_id = "derived:pkgconfig/zynum_blas.pc"
        candidates = {item["id"]: item for item in inventory["derived_candidates"]}
        self.assertIn(source_candidate_id, candidates)
        candidate = candidates[source_candidate_id]
        source_path = candidate["path"]
        source_file = self.root / source_path
        self.assertTrue(source_file.is_file())
        source_file.unlink()

        fixture_path = "docs/assets/benchmarks/schema-validation.svg"
        fixture_file = self.root / fixture_path
        fixture_file.parent.mkdir(parents=True, exist_ok=True)
        fixture_file.write_text(
            '<svg xmlns="http://www.w3.org/2000/svg"/>\n', encoding="utf-8"
        )

        candidate.clear()
        candidate.update(
            {
                "id": f"derived:{fixture_path}",
                "path": fixture_path,
                "class": "curated-documentation-asset",
                "owner": "documentation-maintainers",
                "tracking_status": "tracked",
                "public_safe_provenance": "empty synthetic SVG test fixture",
                "claim_scope": "schema validation only; no benchmark results",
                "review_date": "2026-08-10",
                "freshness_criteria": "replace when the curated-asset schema changes",
                "replacement_criteria": "retain only while this schema test requires it",
                "deterministic_regeneration_claim": False,
                "raw_inputs_disposition": "no benchmark inputs; generated in the test fixture",
            }
        )

        classifications = {
            item["path"]: item for item in inventory["repository_file_classifications"]
        }
        self.assertIn(source_path, classifications)
        classification = classifications[source_path]
        classification.clear()
        classification.update(
            {
                "path": fixture_path,
                "kind": "visual-asset",
                "class": candidate["class"],
                "owner": candidate["owner"],
            }
        )
        return candidate

    def test_positive_validation_and_cli(self) -> None:
        self.assertEqual([], CHECKER.validate(self.root, self.inventory_path))
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            result = CHECKER.main(
                ["--root", str(self.root), "--inventory", str(self.inventory_path)]
            )
        self.assertEqual(0, result)
        self.assertIn("build inventory valid", stdout.getvalue())

    def test_malformed_inventory_shapes_fail_without_traceback(self) -> None:
        mutations: tuple[tuple[Any, str], ...] = (
            ([], "inventory root must be an object"),
            (
                {"schema_version": 2},
                "build_roots must exactly match independently observed safe build roots",
            ),
        )
        for mutation, expected in mutations:
            with self.subTest(expected=expected):
                self._write_inventory(mutation)
                self._assert_error_contains(expected)

        inventory = copy.deepcopy(self.inventory)
        inventory["python_launches"][0] = "not-an-object"
        self._write_inventory(inventory)
        self._assert_error_contains("python_launches: every entry must be an object")

        inventory = copy.deepcopy(self.inventory)
        del inventory["python_launches"][0]["id"]
        self._write_inventory(inventory)
        self._assert_error_contains(
            "python_launches: every entry requires a non-empty id"
        )

        inventory = copy.deepcopy(self.inventory)
        inventory["derived_candidates"][0]["id"] = []
        self._write_inventory(inventory)
        self._assert_error_contains(
            "derived_candidates: every entry requires a non-empty id"
        )

        inventory = copy.deepcopy(self.inventory)
        inventory["derived_candidates"][0]["path"] = []
        self._write_inventory(inventory)
        self._assert_error_contains(
            "derived_candidates: every entry requires a non-empty string path"
        )

    def test_inventory_json_rejects_duplicate_keys_and_constants(self) -> None:
        original = self.inventory_path.read_text(encoding="utf-8")
        self.inventory_path.write_text(
            '{"schema_version":0,' + original.lstrip()[1:], encoding="utf-8"
        )
        self._assert_error_contains("duplicate JSON object key 'schema_version'")
        self.inventory_path.write_text(
            '{"extra":NaN,' + original.lstrip()[1:], encoding="utf-8"
        )
        self._assert_error_contains("non-standard JSON constant 'NaN'")

    def test_inventory_top_level_keys_and_scope_are_exact(self) -> None:
        inventory = self._inventory()
        inventory["fabricated_source_facts"] = {"removed": True}
        self._write_inventory(inventory)
        self._assert_error_contains("top-level keys must match the schema exactly")

        inventory = self._inventory()
        inventory.pop("fabricated_source_facts")
        del inventory["scope"]
        self._write_inventory(inventory)
        errors = "\n".join(CHECKER.validate(self.root, self.inventory_path))
        self.assertIn("top-level keys must match the schema exactly", errors)
        self.assertIn("scope must match the schema contract exactly", errors)

    def test_deeply_nested_inventory_json_fails_without_traceback(self) -> None:
        self.inventory_path.write_text(
            "[" * 500_000 + "0" + "]" * 500_000, encoding="utf-8"
        )
        errors = CHECKER.validate(self.root, self.inventory_path)
        self.assertTrue(errors)
        self.assertTrue(
            errors[0].startswith(("cannot read inventory", "inventory root must")),
            errors,
        )

    def test_unlisted_project_option_fails_closed(self) -> None:
        self._append_to_build(
            'const fixture_option = b.option(bool, "fixture-option", "fixture");'
        )
        self._assert_error_contains("option:build.zig:build:fixture-option")

    def test_unlisted_named_step_fails_closed(self) -> None:
        self._append_to_build('const fixture_step = b.step("fixture-step", "fixture");')
        self._assert_error_contains("step:build.zig:build:fixture-step")

    def test_unlisted_compile_artifact_fails_closed(self) -> None:
        self._append_to_build(
            'const fixture_executable = b.addExecutable(.{ .name = "fixture", .root_module = b.createModule(.{}) });'
        )
        self._assert_error_contains("compile:build.zig:build:fixture_executable")

    def test_missing_isolated_object_inventory_entry_fails(self) -> None:
        inventory = self._inventory()
        inventory["build_observations"] = [
            item
            for item in inventory["build_observations"]
            if item["id"] != "compile:build.zig:build:stride2_isolated_library"
        ]
        self._write_inventory(inventory)
        self._assert_error_contains("expected exactly 8 isolated libraries")

    def test_unlisted_conditional_link_edge_fails_closed(self) -> None:
        discovered = {
            item["id"]: item
            for item in CHECKER._discover_build_root(self.root, "build.zig")
        }
        self.assertIn(CHECKER.LEVEL2_WIDTH_DEFAULT_ARTIFACT_LINK_ID, discovered)
        self.assertIn(CHECKER.LEVEL2_WIDTH_ENABLED_ARTIFACT_LINK_ID, discovered)
        self.assertNotEqual(
            CHECKER.LEVEL2_WIDTH_DEFAULT_ARTIFACT_LINK_ID,
            CHECKER.LEVEL2_WIDTH_ENABLED_ARTIFACT_LINK_ID,
        )
        self.assertEqual(
            "level2_width_isolated_library",
            CHECKER._new_test_inventory_observation(
                CHECKER.LEVEL2_WIDTH_ENABLED_ARTIFACT_LINK_ID, self._inventory()
            )["provider"],
        )

        path = self.root / "build.zig"
        original = path.read_text(encoding="utf-8")
        mutations = (
            (
                "if (target_has_avx512f and level2_width_selected)",
                "if (target_has_avx512f)",
            ),
            (
                "if (level2_width_enabled_artifact_probe_mod) |probe_mod| {\n"
                "        probe_mod.addObject(level2_width_isolated_library.?);\n"
                "    }",
                "if (level2_width_enabled_artifact_probe_mod) |probe_mod| {\n"
                "        probe_mod.addObject(stride2_isolated_library.?);\n"
                "    }",
            ),
        )
        baseline = discovered[CHECKER.LEVEL2_WIDTH_ENABLED_ARTIFACT_LINK_ID]
        for before, after in mutations:
            with self.subTest(enabled_artifact_mutation=before):
                self.assertIn(before, original)
                path.write_text(original.replace(before, after, 1), encoding="utf-8")
                mutated = {
                    item["id"]: item
                    for item in CHECKER._discover_build_root(self.root, "build.zig")
                }
                if "stride2_isolated_library" in after:
                    self.assertNotIn(
                        CHECKER.LEVEL2_WIDTH_ENABLED_ARTIFACT_LINK_ID, mutated
                    )
                else:
                    self.assertNotEqual(
                        baseline["source_digest"],
                        mutated[CHECKER.LEVEL2_WIDTH_ENABLED_ARTIFACT_LINK_ID][
                            "source_digest"
                        ],
                    )
        path.write_text(original, encoding="utf-8")
        self._append_to_build("zynum_mod.linkLibrary(structured_isolated_library.?);")
        self._assert_error_contains(
            "numerics resources owner/source/target/attachment graph drifted"
        )

    def test_unlisted_build_launch_fails_closed(self) -> None:
        self._append_to_build(
            'const fixture_launch = b.addSystemCommand(&.{ "python3", "-V" });'
        )
        self._assert_error_contains("launch:build.zig:build:fixture_launch")

    def _mutate_build_and_rehash(self, original: str, replacement: str) -> None:
        path = self.root / "build.zig"
        text = path.read_text(encoding="utf-8")
        self.assertIn(original, text)
        path.write_text(text.replace(original, replacement, 1), encoding="utf-8")
        inventory = self._inventory()
        inventory["build_root_digests"]["build.zig"] = hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
        self._write_inventory(inventory)

    def test_inventory_factory_case_table_mutations_fail_closed(self) -> None:
        mutations = (
            (
                '.{ .root_id = "zig-root:modern-tests", .logical_tests = modern_tests },',
                '.{ .root_id = "zig-root:modern/tests", .logical_tests = modern_tests },',
                "non-empty Windows-safe suffixes",
            ),
            (
                '.{ .root_id = "zig-root:blas-module-tests", .logical_tests = blas_module_tests },',
                '.{ .root_id = "zig-root:unknown-tests", .logical_tests = blas_module_tests },',
                "root ids must be unique and canonically sorted",
            ),
            (
                '.{ .root_id = "zig-root:blas-public-surface-contract-tests", .logical_tests = blas_public_surface_contract_tests },',
                '.{ .root_id = "zig-root:blas-module-tests", .logical_tests = blas_public_surface_contract_tests },',
                "root ids must be unique and canonically sorted",
            ),
            (
                '.{ .root_id = "zig-root:blas-public-surface-contract-tests", .logical_tests = blas_public_surface_contract_tests },',
                '.{ .root_id = "zig-root:blas-public-surface-contract-tests", .logical_tests = blas_module_tests },',
                "logical compile observations must be unique",
            ),
            (
                "const inventory_cases = [25]InventoryCase{",
                "const inventory_cases = [20]InventoryCase{",
                "must declare exactly 25 expansion cases",
            ),
            (
                "const structured_object_tests = if (target.result.cpu.arch == .x86_64)",
                "const structured_object_tests = if (target.result.cpu.arch == .aarch64)",
                "cannot normalize predicate for structured_object_tests",
            ),
            (
                "if (!uses_facade_owner and official_tests != numerics_contract_tests and official_tests != execution_contract_tests)",
                "if (official_tests != lapack_tests)",
                "BLAS object attachments must preserve shared facade inheritance and numerical root exclusion",
            ),
            (
                "if (!uses_facade_owner and official_tests != numerics_contract_tests and official_tests != execution_contract_tests)",
                "if (official_tests != sparse_tests)",
                "BLAS object attachments must preserve shared facade inheritance and numerical root exclusion",
            ),
            (
                "if (!uses_facade_owner and official_tests != numerics_contract_tests and official_tests != execution_contract_tests)",
                "if (official_tests != lapack_tests and official_tests != sparse_tests)",
                "BLAS object attachments must preserve shared facade inheritance and numerical root exclusion",
            ),
            (
                "if (!uses_facade_owner and official_tests != numerics_contract_tests and official_tests != execution_contract_tests)",
                "if (true)",
                "BLAS object attachments must preserve shared facade inheritance and numerical root exclusion",
            ),
        )
        baseline_build = (self.root / "build.zig").read_text(encoding="utf-8")
        baseline_inventory = self.inventory_path.read_text(encoding="utf-8")
        for original, replacement, expected in mutations:
            with self.subTest(expected=expected):
                (self.root / "build.zig").write_text(baseline_build, encoding="utf-8")
                self.inventory_path.write_text(baseline_inventory, encoding="utf-8")
                self._mutate_build_and_rehash(original, replacement)
                self._assert_error_contains(expected)

    def test_inventory_enumeration_projection_mutations_fail_closed(self) -> None:
        mutations = (
            (
                "const target = b.resolveTargetQuery(target_query);",
                "const target = b.resolveTargetQuery(target_query); target_query.cpu_model = .baseline;",
                "query provenance and canonical baseline resolved features",
            ),
            (
                "explicit_cpu and !dynamic_dispatch and target_query.cpu_model",
                "target_query.cpu_model",
                "query provenance and canonical baseline resolved features",
            ),
            (
                "var target_query = b.standardTargetOptionsQueryOnly(.{ .default_target = .{ .cpu_model = .baseline } });",
                "const target_query = b.standardTargetOptions(.{}).query;",
                "query provenance and canonical baseline resolved features",
            ),
            (
                "target_query.cpu_model == .baseline",
                "target_query.cpu_model == .native",
                "query provenance and canonical baseline resolved features",
            ),
            (
                "target_query.cpu_features_add.isEmpty()",
                "!target_query.cpu_features_add.isEmpty()",
                "query provenance and canonical baseline resolved features",
            ),
            (
                "target_query.cpu_features_sub.isEmpty()",
                "!target_query.cpu_features_sub.isEmpty()",
                "query provenance and canonical baseline resolved features",
            ),
            (
                "std.Target.Cpu.baseline(target.result.cpu.arch, target.result.os)",
                "std.Target.Cpu.baseline(target.result.cpu.arch, .linux)",
                "query provenance and canonical baseline resolved features",
            ),
            (
                "target.result.cpu.features.eql(expected_baseline_cpu.features)",
                "target.result.cpu.features.eql(target.result.cpu.model.features)",
                "query provenance and canonical baseline resolved features",
            ),
            (
                "&std.Target.aarch64.cpu.apple_m1",
                "&std.Target.aarch64.cpu.apple_m2",
                "exact four baseline mappings",
            ),
            (
                "target.result.os.tag == .macos and target.result.abi == .none and target.result.ofmt == .macho",
                "target.result.os.tag == .macos and target.result.ofmt == .macho",
                "exact four baseline mappings",
            ),
            (
                '"env:aarch64-linux-gnu-baseline"',
                '"env:x86-64-linux-gnu-baseline"',
                "exact four baseline mappings",
            ),
            (
                """    else
        null;""",
                """    else
        @panic(\"unsupported target\");""",
                "null fallback",
            ),
            (
                "if (inventory_profile) |resolved_inventory_profile| {",
                "if (true) {",
                "known target branch is missing",
            ),
            (
                "test_inventory_link_step.dependOn(&unsupported_test_inventory_target.step);",
                "test_inventory_link_step.dependOn(test_inventory_step);",
                "shared failure dependency",
            ),
        )
        baseline_build = (self.root / "build.zig").read_text(encoding="utf-8")
        baseline_inventory = self.inventory_path.read_text(encoding="utf-8")
        for original, replacement, expected in mutations:
            with self.subTest(expected=expected):
                (self.root / "build.zig").write_text(baseline_build, encoding="utf-8")
                self.inventory_path.write_text(baseline_inventory, encoding="utf-8")
                self._mutate_build_and_rehash(original, replacement)
                self._assert_error_contains(expected)

    @unittest.skipUnless(shutil.which("zig"), "Zig is required for build graph checks")
    def test_unknown_target_configures_and_inventory_steps_fail_closed(self) -> None:
        zig = shutil.which("zig")
        assert zig is not None
        configurations = (
            ("aarch64-macos", None),
            ("x86_64-linux-gnu", None),
            ("aarch64-linux-gnu", None),
            ("x86_64-windows-gnu", None),
            ("x86_64-macos", None),
            ("aarch64-macos", "baseline"),
            ("x86_64-linux-gnu", "baseline"),
            ("aarch64-linux-gnu", "baseline"),
            ("x86_64-windows-gnu", "baseline"),
            ("aarch64-macos-gnu", "baseline"),
            ("x86_64-linux-gnu", "x86_64_v3"),
            ("x86_64-linux-gnu", "x86_64_v4"),
        )
        for target, cpu in configurations:
            with self.subTest(target=target, cpu=cpu):
                command = [zig, "build", "--help", f"-Dtarget={target}"]
                if cpu is not None:
                    command.append(f"-Dcpu={cpu}")
                result = subprocess.run(
                    command,
                    cwd=self.root,
                    capture_output=True,
                    text=True,
                    timeout=30,
                    check=False,
                )
                self.assertEqual(0, result.returncode, result.stderr)

        failing_profiles = (
            ("test-inventory-link", "x86_64-macos", "baseline"),
            ("test-inventory", "x86_64-macos", "baseline"),
            ("test-inventory-link", "x86_64-linux-gnu", "x86_64_v3"),
            ("test-inventory", "x86_64-linux-gnu", "x86_64_v4"),
            ("test-inventory-link", "aarch64-macos-gnu", "baseline"),
            ("test-inventory", "aarch64-macos-gnu", "baseline"),
        )
        for step, target, cpu in failing_profiles:
            with self.subTest(step=step, target=target, cpu=cpu):
                result = subprocess.run(
                    [
                        zig,
                        "build",
                        step,
                        f"-Dtarget={target}",
                        f"-Dcpu={cpu}",
                    ],
                    cwd=self.root,
                    capture_output=True,
                    text=True,
                    timeout=30,
                    check=False,
                )
                self.assertNotEqual(0, result.returncode)
                self.assertIn(
                    CHECKER.TEST_INVENTORY_UNSUPPORTED_TARGET_MESSAGE,
                    result.stdout + result.stderr,
                )

    @unittest.skipUnless(shutil.which("zig"), "Zig is required for build graph checks")
    def test_native_feature_step_rejects_non_native_evidence_profiles(self) -> None:
        zig = shutil.which("zig")
        assert zig is not None
        machine = platform.machine().lower()
        if machine in {"aarch64", "arm64"}:
            cross_target = "x86_64-linux-gnu"
            cross_cpu = "x86_64_v3"
        else:
            cross_target = "aarch64-linux-gnu"
            cross_cpu = "generic"
        profiles = [
            (
                "baseline",
                ["-Dcpu=baseline"],
                "test-native-feature requires an explicit non-baseline CPU profile",
            ),
            (
                "cross-or-emulated",
                [f"-Dtarget={cross_target}", f"-Dcpu={cross_cpu}"],
                "test-native-feature requires target arch/os/abi/ofmt to match the build host exactly",
            ),
            (
                "external-executor",
                ["-Dcpu=native", "-fqemu"],
                "native correctness requires direct host execution",
            ),
        ]
        if sys.platform == "darwin" and machine in {"aarch64", "arm64"}:
            profiles.append(
                (
                    "unsupported-feature",
                    ["-Dtarget=aarch64-macos", "-Dcpu=apple_m4+sve"],
                    "test-native-feature requested CPU features are not supported by the build host",
                )
            )

        for profile, options, expected in profiles:
            with self.subTest(profile=profile):
                result = subprocess.run(
                    [
                        zig,
                        "build",
                        "test-native-feature",
                        *options,
                        "--summary",
                        "failures",
                    ],
                    cwd=self.root,
                    capture_output=True,
                    text=True,
                    timeout=60,
                    check=False,
                )
                self.assertNotEqual(0, result.returncode)
                self.assertIn(expected, result.stdout + result.stderr)

    def test_inventory_factory_loop_mutations_fail_closed(self) -> None:
        mutations = (
            (
                'b.fmt("inventory-zig-root-{s}", .{inventory_case.root_id[root_id_prefix.len..]})',
                'b.fmt("inventory-{s}", .{inventory_case.root_id})',
                "physical artifact name contract",
            ),
            (
                "            const inventory_tests = b.addTest(.{\n"
                '                .name = b.fmt("inventory-zig-root-{s}", .{inventory_case.root_id[root_id_prefix.len..]}),\n',
                "            const ignored_physical_name = "
                'b.fmt("inventory-zig-root-{s}", .{inventory_case.root_id[root_id_prefix.len..]});\n'
                "            const inventory_tests = b.addTest(.{\n"
                "                .name = inventory_case.root_id,\n",
                "physical artifact name contract",
            ),
            (
                'const root_id_prefix = "zig-root:";',
                "// root ID prefix removed",
                "physical artifact name contract",
            ),
            (
                'const root_id_prefix = "zig-root:";',
                'const root_id_prefix = "zig-root-";',
                "physical artifact name contract",
            ),
            (
                "            if (!std.mem.startsWith(u8, inventory_case.root_id, root_id_prefix)) {\n"
                '                @panic("test inventory root ID must start with zig-root:");\n'
                "            }\n",
                "",
                "physical artifact name contract",
            ),
            (
                "std.mem.startsWith(u8, inventory_case.root_id, root_id_prefix)",
                "std.mem.endsWith(u8, inventory_case.root_id, root_id_prefix)",
                "physical artifact name contract",
            ),
            (
                "            if (target.result.os.tag == .windows and target.result.ofmt == .coff) {\n"
                "                inventory_tests.root_module.strip = true;\n"
                "            }\n",
                "",
                "Windows COFF inventory compile artifacts must strip debug information",
            ),
            (
                "inventory_tests.root_module.strip = true;",
                "inventory_tests.root_module.strip = false;",
                "Windows COFF inventory compile artifacts must strip debug information",
            ),
            (
                ".root_module = official_tests.root_module,",
                ".root_module = blas_module_tests.root_module,",
                "test inventory factory loop is missing required relation",
            ),
            (
                '.path = b.path("tools/test_inventory_runner.zig"),',
                '.path = b.path("tools/other_runner.zig"),',
                "source field test_runner changed",
            ),
            (
                ".mode = .simple,",
                ".mode = .server,",
                "test inventory factory loop is missing required relation",
            ),
            (
                '"--inventory-root",',
                '"--wrong-root-argument",',
                "argument vector must bind root, mode, environment, and enumeration class exactly",
            ),
            (
                '"--inventory-environment",',
                '"--wrong-environment-argument",',
                "argument vector must bind root, mode, environment, and enumeration class exactly",
            ),
            (
                "const run_inventory_tests = b.addRunArtifact(inventory_tests);",
                "const run_inventory_tests = b.addRunArtifact(official_tests);",
                "test inventory factory loop is missing required relation",
            ),
            (
                "test_inventory_link_step.dependOn(&inventory_tests.step);",
                "test_inventory_link_step.dependOn(&official_tests.step);",
                "test inventory factory loop is missing required relation",
            ),
            (
                "run_inventory_tests.step.dependOn(&test_inventory_structure_check.step);",
                "inventory_tests.step.dependOn(&test_inventory_structure_check.step);",
                "shared Compile node must not inherit the POSIX structure checker",
            ),
            (
                "test_inventory_link_step.dependOn(&test_inventory_structure_check.step);",
                "// ordinary link checker removed",
                "POSIX structure checker relation changed",
            ),
            (
                "test_inventory_step.dependOn(&test_inventory_structure_check.step);",
                "// run-step checker removed",
                "POSIX structure checker relation changed",
            ),
            (
                "run_inventory_tests.step.dependOn(&test_inventory_structure_check.step);",
                "// run-artifact checker removed",
                "POSIX structure checker relation changed",
            ),
            (
                "if (native_canonical_windows_inventory_link) {",
                "if (target.result.os.tag == .windows) {",
                "exactly one canonical guard per inventory case",
            ),
            (
                "test_inventory_link_windows_native_smoke_step.dependOn(&inventory_tests.step);",
                "test_inventory_link_windows_native_smoke_step.dependOn(&inventory_tests.step);\n"
                "                test_inventory_link_windows_native_smoke_step.dependOn(&test_inventory_structure_check.step);",
                "exact guard must depend only on each inventory case Compile node",
            ),
            (
                "test_inventory_link_windows_native_smoke_step.dependOn(&inventory_tests.step);",
                "test_inventory_link_windows_native_smoke_step.dependOn(&inventory_tests.step);\n"
                "                test_inventory_link_windows_native_smoke_step.dependOn(&run_inventory_tests.step);",
                "exact guard must depend only on each inventory case Compile node",
            ),
            (
                "b.graph.host.result.os.tag == target.result.os.tag;",
                "b.graph.host.result.os.tag == .windows or target.result.os.tag == .windows;",
                "Windows native inventory link smoke guard must preserve the exact host and canonical target contract",
            ),
            (
                "test_inventory_step.dependOn(&run_inventory_tests.step);",
                "test_inventory_step.dependOn(&inventory_tests.step);",
                "test inventory factory loop is missing required relation",
            ),
            (
                "test_step.dependOn(test_inventory_step);",
                "test_step.dependOn(test_inventory_link_step);",
                "canonical test aggregate must depend on the test inventory run step",
            ),
            (
                "run_modern_tests.step.dependOn(test_inventory_step);",
                "run_modern_tests.step.dependOn(test_inventory_link_step);",
                "official Zig body launch barrier is missing required relation",
            ),
            (
                "if (run_structured_object_tests) |run| run.step.dependOn(test_inventory_step);",
                "if (run_structured_object_tests) |run| run.step.dependOn(test_inventory_link_step);",
                "official Zig body launch barrier is missing required relation",
            ),
            (
                "const python_tooling_tests = b.addSystemCommand(&.{",
                "const renamed_python_tooling_tests = b.addSystemCommand(&.{",
                "Python tooling test observations are incomplete",
            ),
            (
                '"test-python-tooling",',
                '"renamed-test-python-tooling",',
                "Python tooling test observations are incomplete",
            ),
            (
                'python_tooling_tests.addFileArg2(b.path("tools/check_test_inventory.py"), .{});',
                'python_tooling_tests.addFileArg2(b.path("tools/unknown_checker.py"), .{});',
                "exact explicit checker argv contract",
            ),
            (
                'python_tooling_tests.addArg("--root");',
                'python_tooling_tests.addArg("--inventory");',
                "exact explicit checker argv contract",
            ),
            (
                'python_tooling_tests.addArg("--inventory");',
                'python_tooling_tests.addArg("--missing-inventory");',
                "exact explicit checker argv contract",
            ),
            (
                'python_tooling_tests.addArg("python-root:benchmark-tools-discovery");',
                "python_tooling_tests.addArg(python_tooling_root_id);",
                "exact explicit checker argv contract",
            ),
            (
                'python_tooling_tests.addFileArg2(b.path("tools/test_inventory.json"), .{});',
                'python_tooling_tests.addFileArg2(b.path("tools/unknown_inventory.json"), .{});',
                "exact explicit checker argv contract",
            ),
            (
                'python_tooling_tests.addFileArg2(b.path("tools/check_test_inventory.py"), .{});',
                'python_tooling_tests.addArg("-c");',
                "exact explicit checker argv contract",
            ),
            (
                'python_tooling_tests.setCwd(b.path("."));',
                'python_tooling_tests.setCwd(b.path("bench"));',
                "incorrect repository working directory",
            ),
            (
                'python_tooling_tests.removeEnvironmentVariable("LESS");',
                'python_tooling_tests.removeEnvironmentVariable("LESS");\n'
                '    python_tooling_tests.removeEnvironmentVariable("EXTRA");',
                "exact reviewed Python environment variables",
            ),
            (
                'python_tooling_tests.removeEnvironmentVariable("LESS");',
                'python_tooling_tests.removeEnvironmentVariable("LESS_MISSING");',
                "exact reviewed Python environment variables",
            ),
            (
                "python_tooling_tests.step.dependOn(&test_inventory_structure_check.step);",
                "python_tooling_tests.step.dependOn(&test_inventory_security_tests.step);",
                "exact structure and Windows artifact dependency closure",
            ),
            (
                'python_tooling_tests.addArg("--windows-zynum-blas-build-output");',
                'python_tooling_tests.addArg("--windows-zynum-blas-emitted-output");',
                "exact ordered Windows artifact argv and dependency contract",
            ),
            (
                'python_tooling_tests.addArg("--windows-zynum-blas-installed-output");',
                'python_tooling_tests.addArg("--windows-zynum-blas-install-output");',
                "exact ordered Windows artifact argv and dependency contract",
            ),
            (
                "python_tooling_tests.step.dependOn(&install_triangular_matrix_probe.step);",
                "python_tooling_tests.step.dependOn(&install_static_lib.step);",
                "exact structure and Windows artifact dependency closure",
            ),
            (
                "python_tooling_tests.step.dependOn(&install_dynamic_lib.step);",
                "python_tooling_test_step.dependOn(&install_dynamic_lib.step);",
                "exact structure and Windows artifact dependency closure",
            ),
            *(
                (
                    dependency,
                    "// disconnected direct dependency",
                    "exact structure and Windows artifact dependency closure",
                )
                for dependency in (
                    "python_tooling_tests.step.dependOn(&test_inventory_structure_check.step);",
                    "python_tooling_tests.step.dependOn(&install_dynamic_lib.step);",
                    "python_tooling_tests.step.dependOn(&install_rank_k_probe.step);",
                    "python_tooling_tests.step.dependOn(&install_rotg_latency_probe.step);",
                    "python_tooling_tests.step.dependOn(&install_symm_probe.step);",
                    "python_tooling_tests.step.dependOn(&install_triangular_matrix_probe.step);",
                )
            ),
            (
                'build_profile_tests.addFileArg2(b.path("test/build/test_build_profiles.py"), .{});',
                'build_profile_tests.addFileArg2(b.path("test/build/test_build_inventory.py"), .{});',
                "build profile launch must preserve its exact reviewed argv",
            ),
            (
                "build_profile_test_step.dependOn(&build_profile_tests.step);",
                "build_profile_test_step.dependOn(&build_inventory_tests.step);",
                "build profile named step must close over exactly its launch",
            ),
            (
                "if (host_tool_smoke) test_step.dependOn(host_tool_smoke_test_step);",
                "if (host_tool_smoke) test_step.dependOn(host_tool_smoke_test_step);\n"
                "    test_step.dependOn(build_profile_test_step);",
                "default correctness aggregate must preserve exclusive dynamic and canonical specialized branches",
            ),
            (
                "python_tooling_test_step.dependOn(&python_tooling_tests.step);",
                "python_tooling_test_step.dependOn(&test_inventory_security_tests.step);",
                "named step must close over exactly its launch",
            ),
            (
                "if (host_tool_smoke) test_step.dependOn(host_tool_smoke_test_step);",
                "if (!host_tool_smoke) test_step.dependOn(host_tool_smoke_test_step);",
                "default correctness aggregate must preserve exclusive dynamic and canonical specialized branches",
            ),
            (
                "if (host_tool_smoke) test_step.dependOn(host_tool_smoke_test_step);",
                "if (host_tool_smoke) test_step.dependOn(python_tooling_test_step);",
                "default correctness aggregate must preserve exclusive dynamic and canonical specialized branches",
            ),
            (
                "if (host_tool_smoke) test_step.dependOn(host_tool_smoke_test_step);",
                "if (host_tool_smoke) test_step.dependOn(host_tool_smoke_test_step);\n"
                "    test_step.dependOn(python_tooling_test_step);",
                "default correctness aggregate must preserve exclusive dynamic and canonical specialized branches",
            ),
            (
                "if (host_tool_smoke) test_step.dependOn(host_tool_smoke_test_step);",
                "if (host_tool_smoke) test_step.dependOn(host_tool_smoke_test_step);\n"
                "    test_step.dependOn(&build_inventory_tests.step);",
                "default correctness aggregate must preserve exclusive dynamic and canonical specialized branches",
            ),
            *(
                (
                    dependency,
                    "// removed reviewed host-tool dependency",
                    "must preserve its exact seven direct dependencies",
                )
                for dependency in (
                    "host_tool_smoke_test_step.dependOn(python_tooling_test_step);",
                    "host_tool_smoke_test_step.dependOn(&abi_manifest_smoke_test.step);",
                    "host_tool_smoke_test_step.dependOn(&c_header_smoke_test.step);",
                    "host_tool_smoke_test_step.dependOn(&cpp_header_smoke_test.step);",
                    "host_tool_smoke_test_step.dependOn(&fortran_module_smoke_test.step);",
                    "host_tool_smoke_test_step.dependOn(abi_baseline_observer_test_step);",
                    "host_tool_smoke_test_step.dependOn(build_profile_test_step);",
                )
            ),
            (
                "host_tool_smoke_test_step.dependOn(abi_baseline_observer_test_step);",
                "host_tool_smoke_test_step.dependOn(abi_baseline_observer_test_step);\n"
                "    host_tool_smoke_test_step.dependOn(&build_inventory_tests.step);",
                "must preserve its exact seven direct dependencies",
            ),
        )
        baseline_build = (self.root / "build.zig").read_text(encoding="utf-8")
        baseline_inventory = self.inventory_path.read_text(encoding="utf-8")
        for original, replacement, expected in mutations:
            with self.subTest(expected=expected):
                (self.root / "build.zig").write_text(baseline_build, encoding="utf-8")
                self.inventory_path.write_text(baseline_inventory, encoding="utf-8")
                self._mutate_build_and_rehash(original, replacement)
                self._assert_error_contains(expected)

    def test_native_feature_step_is_guarded_and_separate_from_inventory_run(
        self,
    ) -> None:
        build_source = (REPOSITORY_ROOT / "build.zig").read_text(encoding="utf-8")
        feature_start = build_source.index("const test_native_feature_step = b.step(")
        official_barrier_start = build_source.index(
            "run_blas_module_tests.step.dependOn(test_inventory_step);",
            feature_start,
        )
        feature_source = build_source[feature_start:official_barrier_start]
        default_test_start = build_source.index(
            'const test_step = b.step("test", "Run correctness tests");'
        )
        default_test_end = build_source.index(
            "const bench = b.addExecutable", default_test_start
        )
        default_test_source = build_source[default_test_start:default_test_end]

        required_feature_fragments = (
            '"test-native-feature"',
            (
                '"Run correctness-only tests for an explicit host-supported '
                'non-baseline CPU profile; not inventory evidence"'
            ),
            "test_native_feature_step.dependOn(&test_inventory_structure_check.step);",
            "const explicit_non_baseline_cpu_profile = switch (target_query.cpu_model)",
            ".native, .explicit => true,",
            ".baseline, .determined_by_arch_os => false,",
            "target.result.cpu.arch == b.graph.host.result.cpu.arch",
            "target.result.os.tag == b.graph.host.result.os.tag",
            "target.result.abi == b.graph.host.result.abi",
            "target.result.ofmt == b.graph.host.result.ofmt",
            *(
                'native_execution_check.addThirdPartyEnabledArg' + name
                + '(.{ .enabled = "--native-executor=' + name.lower() + '" });'
                for name in ("Qemu", "Rosetta", "Wine", "Darling", "Wasmtime")
            ),
            "b.graph.host.result.cpu.features.isSuperSetOf(target.result.cpu.features)",
            "for (inventory_cases) |inventory_case|",
            "const official_tests = inventory_case.logical_tests orelse continue;",
            "const run_native_feature_tests = b.addRunArtifact(official_tests);",
            "run_native_feature_tests.step.dependOn(native_feature_profile_guard);",
            "run_native_feature_tests.step.dependOn(&native_execution_check.step);",
            "test_native_feature_step.dependOn(&run_native_feature_tests.step);",
        )
        for fragment in required_feature_fragments:
            with self.subTest(required_fragment=fragment):
                self.assertIn(fragment, feature_source)

        for forbidden_fragment in (
            "test_inventory_runner",
            "run_inventory_tests",
            "test_inventory_step",
            "--inventory-root",
        ):
            with self.subTest(forbidden_fragment=forbidden_fragment):
                self.assertNotIn(forbidden_fragment, feature_source)
        self.assertIn("test_step.dependOn(test_inventory_step);", default_test_source)
        self.assertNotIn("test_native_feature_step", default_test_source)

    def test_native_feature_checker_contract_mutations_fail_closed(self) -> None:
        mutations = (
            (
                ".native, .explicit => true,",
                ".native => true,\n        .explicit => false,",
                "native feature source guard contract changed",
            ),
            (
                "target.result.abi == b.graph.host.result.abi and",
                "target.result.abi != b.graph.host.result.abi and",
                "native feature source guard contract changed",
            ),
            (
                'native_execution_check.addThirdPartyEnabledArgQemu(.{ .enabled = "--native-executor=qemu" });',
                '// QEMU toggle guard omitted',
                "native feature source guard contract changed",
            ),
            (
                "else if (!b.graph.host.result.cpu.features.isSuperSetOf(target.result.cpu.features))",
                "else if (!target.result.cpu.features.isSuperSetOf(b.graph.host.result.cpu.features))",
                "native feature profile guard order or success dependency changed",
            ),
            (
                "const run_native_feature_tests = b.addRunArtifact(official_tests);",
                "const run_native_feature_tests = b.addRunArtifact(inventory_tests);",
                "native feature expansion loop relation changed",
            ),
            (
                "run_native_feature_tests.step.dependOn(native_feature_profile_guard);",
                "run_native_feature_tests.step.dependOn(test_inventory_step);",
                "native feature expansion loop relation changed",
            ),
            (
                "test_native_feature_step.dependOn(&test_inventory_structure_check.step);",
                "test_native_feature_step.dependOn(&test_inventory_security_tests.step);",
                "native feature source guard contract changed",
            ),
        )
        baseline_build = (self.root / "build.zig").read_text(encoding="utf-8")
        baseline_inventory = self.inventory_path.read_text(encoding="utf-8")
        for original, replacement, expected in mutations:
            with self.subTest(expected=expected, original=original):
                (self.root / "build.zig").write_text(baseline_build, encoding="utf-8")
                self.inventory_path.write_text(baseline_inventory, encoding="utf-8")
                self._mutate_build_and_rehash(original, replacement)
                self._assert_error_contains(expected)

    def test_inventory_factory_recorded_contract_mutations_fail_closed(self) -> None:
        mutations = (
            (
                CHECKER.TEST_INVENTORY_FACTORY_COMPILE_ID,
                "artifact_role",
                "wrong-role",
                "enumerator factory role",
            ),
            (
                CHECKER.TEST_INVENTORY_FACTORY_COMPILE_ID,
                "expansion_cases_digest",
                "0" * 64,
                "canonical expansion digest",
            ),
            (
                CHECKER.TEST_INVENTORY_FACTORY_COMPILE_ID,
                "enumeration_class_projection",
                {},
                "optional enumeration class projection",
            ),
            (
                CHECKER.TEST_INVENTORY_FACTORY_COMPILE_ID,
                "structure_checker_dependency",
                CHECKER.TEST_INVENTORY_STRUCTURE_CHECK_ID,
                "must not inherit the POSIX structure checker",
            ),
            (
                CHECKER.TEST_INVENTORY_FACTORY_COMPILE_ID,
                "output_name",
                "inventory-{root-id}",
                "safe physical artifact pattern",
            ),
            (
                CHECKER.TEST_INVENTORY_FACTORY_COMPILE_ID,
                "produced_outputs",
                ["inventory-{root-id}", "inventory-{root-id}.exe"],
                "safe physical artifact patterns",
            ),
            (
                CHECKER.TEST_INVENTORY_FACTORY_LAUNCH_ID,
                "source_factory",
                "compile:build.zig:build:modern_tests",
                "wrong source factory",
            ),
            (
                CHECKER.TEST_INVENTORY_FACTORY_LAUNCH_ID,
                "argument_contract",
                {},
                "ordered argument contract",
            ),
            (
                CHECKER.TEST_INVENTORY_FACTORY_LAUNCH_ID,
                "argv_shape",
                [],
                "argv shape does not match the ordered argument contract",
            ),
            (
                CHECKER.TEST_INVENTORY_LINK_STEP_ID,
                "direct_dependencies",
                [],
                "factory dependency closure is incomplete",
            ),
            (
                CHECKER.TEST_INVENTORY_RUN_STEP_ID,
                "step_role",
                "wrong-role",
                "incorrect test inventory step role",
            ),
            (
                CHECKER.TEST_INVENTORY_RUN_STEP_ID,
                "official_body_launch_barrier",
                {},
                "official body launch barrier contract is incomplete",
            ),
            (
                CHECKER.TEST_INVENTORY_WINDOWS_NATIVE_LINK_STEP_ID,
                "structure_checker_dependency",
                CHECKER.TEST_INVENTORY_STRUCTURE_CHECK_ID,
                "compatibility-only contract",
            ),
            (
                CHECKER.TEST_INVENTORY_WINDOWS_NATIVE_LINK_STEP_ID,
                "test_body_execution",
                True,
                "compatibility-only contract",
            ),
        )
        baseline_inventory = self.inventory_path.read_text(encoding="utf-8")
        for identifier, field, value, expected in mutations:
            with self.subTest(identifier=identifier, field=field):
                self.inventory_path.write_text(baseline_inventory, encoding="utf-8")
                inventory = self._inventory()
                observation = next(
                    item
                    for item in inventory["build_observations"]
                    if item["id"] == identifier
                )
                observation[field] = value
                self._write_inventory(inventory)
                self._assert_error_contains(expected)

        python_tooling_observations = [
            {
                "id": CHECKER.PYTHON_TOOLING_LAUNCH_ID,
                **CHECKER._python_tooling_launch_template(),
            },
            {
                "id": CHECKER.PYTHON_TOOLING_STEP_ID,
                **CHECKER._python_tooling_step_template(),
            },
            {
                "id": CHECKER.HOST_TOOL_SMOKE_STEP_ID,
                **CHECKER._host_tool_smoke_step_template(),
            },
            {
                "id": CHECKER.BUILD_INVENTORY_STEP_ID,
                **CHECKER._build_inventory_step_template(),
            },
            {
                "id": CHECKER.TEST_INVENTORY_AGGREGATE_STEP_ID,
                "direct_dependencies": [
                    {
                        "id": CHECKER.HOST_TOOL_SMOKE_STEP_ID,
                        "condition": "host-tool-smoke is true",
                    }
                ],
            },
        ]
        python_tooling_errors: list[str] = []
        CHECKER._validate_python_tooling_test_contract(
            python_tooling_observations, python_tooling_errors
        )
        self.assertEqual([], python_tooling_errors)
        missing_python_tooling_launch = [
            copy.deepcopy(observation)
            for observation in python_tooling_observations
            if observation["id"] != CHECKER.PYTHON_TOOLING_LAUNCH_ID
        ]
        missing_errors: list[str] = []
        CHECKER._validate_python_tooling_test_contract(
            missing_python_tooling_launch, missing_errors
        )
        self.assertIn("recorded argv_shape changed", "\n".join(missing_errors))
        python_tooling_mutations = (
            (
                CHECKER.PYTHON_TOOLING_LAUNCH_ID,
                "id",
                "launch:build.zig:build:renamed_python_tooling_tests",
                "recorded argv_shape changed",
            ),
            (
                CHECKER.PYTHON_TOOLING_LAUNCH_ID,
                "argv_shape",
                [],
                "recorded argv_shape changed",
            ),
            (
                CHECKER.PYTHON_TOOLING_LAUNCH_ID,
                "cwd_shape",
                "bench-directory",
                "recorded cwd_shape changed",
            ),
            (
                CHECKER.PYTHON_TOOLING_LAUNCH_ID,
                "runner_contract",
                {},
                "legacy inline runner contract is forbidden",
            ),
            (
                CHECKER.PYTHON_TOOLING_LAUNCH_ID,
                "argument_contract",
                {},
                "recorded argument_contract changed",
            ),
            (
                CHECKER.PYTHON_TOOLING_LAUNCH_ID,
                "checker_script",
                {},
                "recorded checker_script changed",
            ),
            (
                CHECKER.PYTHON_TOOLING_LAUNCH_ID,
                "inventory_file",
                {},
                "recorded inventory_file changed",
            ),
            (
                CHECKER.PYTHON_TOOLING_LAUNCH_ID,
                "test_inventory_barrier",
                {},
                "recorded test_inventory_barrier changed",
            ),
            (
                CHECKER.PYTHON_TOOLING_STEP_ID,
                "direct_dependencies",
                [],
                "recorded direct_dependencies changed",
            ),
            (
                CHECKER.PYTHON_TOOLING_STEP_ID,
                "closure_contract",
                {"launch_count": 2},
                "recorded closure_contract changed",
            ),
            (
                CHECKER.HOST_TOOL_SMOKE_STEP_ID,
                "direct_dependencies",
                [],
                "reviewed host-tool aggregate contract",
            ),
            (
                CHECKER.BUILD_INVENTORY_STEP_ID,
                "aggregate_test_membership",
                "member",
                "reviewed standalone inventory contract",
            ),
            (
                CHECKER.TEST_INVENTORY_AGGREGATE_STEP_ID,
                "direct_dependencies",
                [],
                "must record exactly one conditional host-tool smoke dependency",
            ),
        )
        for identifier, field, value, expected in python_tooling_mutations:
            with self.subTest(
                python_tooling_identifier=identifier,
                python_tooling_field=field,
            ):
                mutated = copy.deepcopy(python_tooling_observations)
                observation = next(item for item in mutated if item["id"] == identifier)
                observation[field] = value
                errors: list[str] = []
                CHECKER._validate_python_tooling_test_contract(mutated, errors)
                self.assertIn(expected, "\n".join(errors))

    def test_source_refresh_is_deterministic_and_preserves_factory_ids(self) -> None:
        original_bytes = self.inventory_path.read_bytes()
        original_inventory = self._inventory()
        for public_source_path in (CHECKER_PATH, Path(__file__).resolve()):
            with self.subTest(
                public_source=public_source_path.relative_to(REPOSITORY_ROOT)
            ):
                self.assertNotRegex(
                    public_source_path.read_text(encoding="utf-8"),
                    r"(?i)freeze[0-9]+",
                )
        original_projection_digest = CHECKER._source_projection_digest(
            original_inventory
        )
        self.assertRegex(CHECKER.CURRENT_SOURCE_PROJECTION_SHA256, r"\A[0-9a-f]{64}\Z")
        reviewed_projection_digests = {CHECKER.CURRENT_SOURCE_PROJECTION_SHA256}
        if CHECKER.NEXT_SOURCE_PROJECTION_SHA256 is not None:
            self.assertRegex(CHECKER.NEXT_SOURCE_PROJECTION_SHA256, r"\A[0-9a-f]{64}\Z")
            self.assertNotEqual(
                CHECKER.CURRENT_SOURCE_PROJECTION_SHA256,
                CHECKER.NEXT_SOURCE_PROJECTION_SHA256,
            )
            reviewed_projection_digests.add(CHECKER.NEXT_SOURCE_PROJECTION_SHA256)
        self.assertIn(original_projection_digest, reviewed_projection_digests)
        self.assertIsNone(CHECKER._reviewed_source_projection_error(original_inventory))
        migration_current = (
            "0" * 64 if original_projection_digest != "0" * 64 else "1" * 64
        )
        with (
            self.subTest(source_projection_policy="current-only"),
            mock.patch.object(
                CHECKER,
                "CURRENT_SOURCE_PROJECTION_SHA256",
                original_projection_digest,
            ),
            mock.patch.object(CHECKER, "NEXT_SOURCE_PROJECTION_SHA256", None),
        ):
            self.assertIsNone(
                CHECKER._reviewed_source_projection_error(original_inventory)
            )
        with (
            self.subTest(source_projection_policy="next-window"),
            mock.patch.object(
                CHECKER, "CURRENT_SOURCE_PROJECTION_SHA256", migration_current
            ),
            mock.patch.object(
                CHECKER,
                "NEXT_SOURCE_PROJECTION_SHA256",
                original_projection_digest,
            ),
        ):
            self.assertIsNone(
                CHECKER._reviewed_source_projection_error(original_inventory)
            )
        refresh_owned_sections = {
            "build_observations",
            "python_launches",
            "workflow_launches",
            "generator_targets",
        }
        self.assertTrue(refresh_owned_sections <= set(CHECKER.SOURCE_PROJECTION_FIELDS))
        self.assertTrue(
            refresh_owned_sections.isdisjoint(CHECKER.REQUIRED_SECTION_FACT_DIGESTS)
        )
        self.assertEqual(
            {
                "option_surfaces",
                "repository_file_classifications",
                "derived_candidates",
                "current_gaps",
            },
            set(CHECKER.REQUIRED_SECTION_FACT_DIGESTS),
        )
        projection = CHECKER._source_projection(original_inventory)
        for section in refresh_owned_sections:
            self.assertIs(projection[section], original_inventory[section])
        self.assertEqual(
            CHECKER._canonical_inventory_bytes(original_inventory),
            CHECKER._canonical_inventory_bytes(copy.deepcopy(original_inventory)),
        )
        security_launch = CHECKER._new_test_inventory_observation(
            "launch:build.zig:build:test_inventory_security_tests",
            original_inventory,
        )
        structure_launch = CHECKER._new_test_inventory_observation(
            "launch:build.zig:build:test_inventory_structure_check",
            original_inventory,
        )
        python_tooling_launch = CHECKER._new_test_inventory_observation(
            CHECKER.PYTHON_TOOLING_LAUNCH_ID,
            original_inventory,
        )
        self.assertEqual("test-infrastructure", security_launch["owner"])
        self.assertEqual("host", security_launch["compile_for"])
        self.assertEqual("host", security_launch["execute_on"])
        self.assertIsNone(security_launch["source_artifact"])
        self.assertEqual(
            ["python3", "-B", "test/build/test_test_inventory.py"],
            security_launch["argv_shape"],
        )
        self.assertEqual(
            [
                "python3",
                "-B",
                "tools/check_test_inventory.py",
                "--root",
                ".",
                "--structure-only",
            ],
            structure_launch["argv_shape"],
        )
        self.assertEqual(
            [
                "python3",
                "-B",
                CHECKER.PYTHON_TOOLING_CHECKER_PATH,
                "--root",
                ".",
                "--inventory",
                CHECKER.PYTHON_TOOLING_INVENTORY_PATH,
                "--run-python-tooling-root",
                CHECKER.PYTHON_TOOLING_ROOT_ID,
            ],
            python_tooling_launch["argv_shape"],
        )
        self.assertEqual("repository-root", python_tooling_launch["cwd_shape"])
        self.assertEqual(
            CHECKER.PYTHON_TOOLING_CHECKER_PATH,
            python_tooling_launch["checker_script"]["path"],
        )
        self.assertEqual(
            CHECKER.PYTHON_TOOLING_INVENTORY_PATH,
            python_tooling_launch["inventory_file"]["path"],
        )
        self.assertEqual(
            CHECKER.PYTHON_TOOLING_STRUCTURE_BARRIER_ID,
            python_tooling_launch["test_inventory_barrier"]["dependency_step_id"],
        )
        security_step = CHECKER._new_test_inventory_observation(
            "step:build.zig:build:test-test-inventory",
            original_inventory,
        )
        python_tooling_step = CHECKER._new_test_inventory_observation(
            CHECKER.PYTHON_TOOLING_STEP_ID,
            original_inventory,
        )
        host_tool_smoke_step = CHECKER._new_test_inventory_observation(
            CHECKER.HOST_TOOL_SMOKE_STEP_ID,
            original_inventory,
        )
        build_inventory_step = CHECKER._new_test_inventory_observation(
            CHECKER.BUILD_INVENTORY_STEP_ID,
            original_inventory,
        )
        self.assertEqual("focused-validation", security_step["step_role"])
        self.assertEqual("not-member", security_step["aggregate_test_membership"])
        self.assertFalse(security_step["intentional_orphan"])
        self.assertEqual(
            [
                {
                    "id": "launch:build.zig:build:test_inventory_security_tests",
                    "condition": "always",
                }
            ],
            security_step["direct_dependencies"],
        )
        self.assertEqual(
            [
                {
                    "id": CHECKER.PYTHON_TOOLING_LAUNCH_ID,
                    "condition": "always",
                }
            ],
            python_tooling_step["direct_dependencies"],
        )
        self.assertEqual(1, python_tooling_step["closure_contract"]["launch_count"])
        self.assertEqual(
            list(CHECKER.HOST_TOOL_SMOKE_DIRECT_DEPENDENCIES),
            host_tool_smoke_step["direct_dependencies"],
        )
        self.assertEqual(
            "conditional-member",
            host_tool_smoke_step["aggregate_test_membership"],
        )
        self.assertEqual(
            "not-member", build_inventory_step["aggregate_test_membership"]
        )
        self.assertEqual(
            "explicit named step only", build_inventory_step["aggregate_condition"]
        )
        probe_compile = CHECKER._new_test_inventory_observation(
            CHECKER.LEVEL2_WIDTH_DEFAULT_ARTIFACT_COMPILE_ID,
            original_inventory,
        )
        probe_launch = CHECKER._new_test_inventory_observation(
            CHECKER.LEVEL2_WIDTH_DEFAULT_ARTIFACT_LAUNCH_ID,
            original_inventory,
        )
        probe_step = CHECKER._new_test_inventory_observation(
            CHECKER.LEVEL2_WIDTH_DEFAULT_ARTIFACT_STEP_ID,
            original_inventory,
        )
        probe_link = CHECKER._new_test_inventory_observation(
            CHECKER.LEVEL2_WIDTH_DEFAULT_ARTIFACT_LINK_ID,
            original_inventory,
        )
        self.assertEqual("executable", probe_compile["artifact_kind"])
        self.assertEqual("test-optimize", probe_compile["optimize_source"])
        self.assertEqual(
            CHECKER.LEVEL2_WIDTH_DEFAULT_ARTIFACT_COMPILE_ID,
            probe_launch["source_artifact"],
        )
        self.assertNotIn("test_inventory_barrier", probe_launch)
        self.assertEqual(
            [
                {
                    "id": CHECKER.LEVEL2_WIDTH_DEFAULT_ARTIFACT_LAUNCH_ID,
                    "condition": CHECKER.LEVEL2_WIDTH_DEFAULT_ARTIFACT_CONDITION,
                }
            ],
            probe_step["direct_dependencies"],
        )
        self.assertEqual("conditional-member", probe_step["aggregate_test_membership"])
        self.assertEqual(
            "level2_width_default_artifact_probe_mod", probe_link["consumer"]
        )
        self.assertEqual("level2_width_isolated_library", probe_link["provider"])
        for observation in (probe_compile, probe_step, probe_link):
            self.assertEqual(
                CHECKER.LEVEL2_WIDTH_DEFAULT_ARTIFACT_CONDITION,
                observation["condition"]
                if "condition" in observation
                else observation["aggregate_condition"],
            )
        enabled_compile = CHECKER._new_test_inventory_observation(
            CHECKER.LEVEL2_WIDTH_ENABLED_ARTIFACT_COMPILE_ID,
            original_inventory,
        )
        enabled_launch = CHECKER._new_test_inventory_observation(
            CHECKER.LEVEL2_WIDTH_ENABLED_ARTIFACT_LAUNCH_ID,
            original_inventory,
        )
        enabled_build_step = CHECKER._new_test_inventory_observation(
            CHECKER.LEVEL2_WIDTH_ENABLED_ARTIFACT_BUILD_STEP_ID,
            original_inventory,
        )
        enabled_run_step = CHECKER._new_test_inventory_observation(
            CHECKER.LEVEL2_WIDTH_ENABLED_ARTIFACT_RUN_STEP_ID,
            original_inventory,
        )
        enabled_link = CHECKER._new_test_inventory_observation(
            CHECKER.LEVEL2_WIDTH_ENABLED_ARTIFACT_LINK_ID,
            original_inventory,
        )
        self.assertEqual("optimize", enabled_compile["optimize_source"])
        self.assertEqual(
            CHECKER.LEVEL2_WIDTH_ARTIFACT_CONTRACT_PATH,
            enabled_compile["probe_contract_source"],
        )
        self.assertEqual(
            CHECKER.LEVEL2_WIDTH_ENABLED_ARTIFACT_COMPILE_ID,
            enabled_launch["source_artifact"],
        )
        self.assertEqual(
            "level2_width_enabled_artifact_probe_mod", enabled_link["consumer"]
        )
        self.assertEqual("level2_width_isolated_library", enabled_link["provider"])
        for step in (enabled_build_step, enabled_run_step):
            self.assertEqual(2, len(step["direct_dependencies"]))
            self.assertEqual(
                CHECKER.LEVEL2_WIDTH_ENABLED_ARTIFACT_UNSUPPORTED_STEP_ID,
                step["direct_dependencies"][1]["id"],
            )
            self.assertEqual(
                CHECKER.LEVEL2_WIDTH_ENABLED_ARTIFACT_UNSUPPORTED_CONDITION,
                step["direct_dependencies"][1]["condition"],
            )
        install_step = CHECKER._new_test_inventory_observation(
            CHECKER.INSTALL_LIBRARIES_STEP_ID, original_inventory
        )
        self.assertEqual(
            [
                {"id": CHECKER.INSTALL_DYNAMIC_LIBRARY_ID, "condition": "always"},
                {"id": CHECKER.INSTALL_STATIC_LIBRARY_ID, "condition": "target.result.ofmt != .macho"},
                {"id": "install:build.zig:build:libzynum_blas.a", "condition": "target.result.ofmt == .macho"},
            ],
            install_step["direct_dependencies"],
        )
        for (
            identifier,
            runtime_source,
        ) in CHECKER.WINDOWS_BENCHMARK_PROBE_COMPILE_SOURCES.items():
            runtime_fields = CHECKER._reviewed_observation_refresh_fields(identifier)
            self.assertEqual(
                {
                    "windows": runtime_source,
                    "non-windows": runtime_source,
                },
                runtime_fields["root_source_by_target"],
            )
            self.assertEqual(
                "benchmark-probe-runtime-evidence",
                runtime_fields["evidence_role_by_target"]["windows"],
            )
        race_launch = CHECKER._new_test_inventory_python_launch(
            CHECKER.TEST_INVENTORY_RUNNER_RACE_PYTHON_LAUNCH_ID
        )
        self.assertEqual("temporary fixture repository", race_launch["cwd_shape"])
        self.assertEqual(
            "test-inventory-runner-fixture-execute", race_launch["launch_class"]
        )
        self.assertEqual(
            "./test-inventory-digest-vectors",
            race_launch["argv_shape"][0],
        )
        workflow_ids = (
            "workflow-launch:.github/workflows/ci.yml:source-checks:bind-trusted-posix-toolchain-paths",
            "workflow-launch:.github/workflows/ci.yml:source-checks:check-build-inventory",
            "workflow-launch:.github/workflows/ci.yml:source-checks:check-test-inventory-structure",
            "workflow-launch:.github/workflows/ci.yml:build-inventory-security:run-build-inventory-security-suite",
            "workflow-launch:.github/workflows/ci.yml:ci-gate:require-every-ci-gate-to-succeed",
            "workflow-launch:.github/workflows/ci.yml:test-inventory-security:run-test-inventory-security-suite",
            "workflow-launch:.github/workflows/ci.yml:target-tests:build-windows-python-tooling-executable-fixtures-and-libraries",
            "workflow-launch:.github/workflows/ci.yml:target-tests:check-windows-library-layout-and-tooling-fixture-boundary",
            "workflow-launch:.github/workflows/ci.yml:target-tests:run-windows-dll-abi-and-cblas-l1-l3-compatibility-smoke-not-inventory-evidence",
            "workflow-launch:.github/workflows/ci.yml:target-tests:bind-trusted-windows-toolchain-paths",
            "workflow-launch:.github/workflows/ci.yml:target-tests:bind-trusted-posix-toolchain-paths-and-require-fortran-for-full-smoke",
            "workflow-launch:.github/workflows/ci.yml:target-tests:run-host-tool-smoke-once",
            "workflow-launch:.github/workflows/ci.yml:target-tests:link-test-inventory-for-debug-target-posix-structure-gated",
            "workflow-launch:.github/workflows/ci.yml:target-tests:link-test-inventory-for-releasesafe-target-posix-structure-gated",
            "workflow-launch:.github/workflows/ci.yml:target-tests:link-test-inventory-for-releasefast-target-posix-structure-gated",
            "workflow-launch:.github/workflows/ci.yml:target-tests:windows-native-compile-link-smoke-for-debug-compatibility-only-not-inventory-evidence",
            "workflow-launch:.github/workflows/ci.yml:target-tests:windows-native-compile-link-smoke-for-releasesafe-compatibility-only-not-inventory-evidence",
            "workflow-launch:.github/workflows/ci.yml:target-tests:windows-native-compile-link-smoke-for-releasefast-compatibility-only-not-inventory-evidence",
            "workflow-launch:.github/workflows/ci.yml:capability-builds:compile-enabled-level-2-width-production-artifact-probe",
            "workflow-launch:.github/workflows/ci.yml:capability-builds:bind-trusted-posix-toolchain-paths",
            "workflow-launch:.github/workflows/release.yml:build-inventory-security:bind-trusted-posix-toolchain-paths",
            "workflow-launch:.github/workflows/release.yml:build-inventory-security:require-current-only-build-inventory-policy",
            "workflow-launch:.github/workflows/release.yml:build-inventory-security:require-current-only-test-inventory-policy",
            "workflow-launch:.github/workflows/release.yml:build-inventory-security:run-build-inventory-security-suite",
            "workflow-launch:.github/workflows/release.yml:test-inventory-security:require-current-only-test-inventory-policy",
            "workflow-launch:.github/workflows/release.yml:test-inventory-security:run-test-inventory-security-suite",
            "workflow-launch:.github/workflows/release.yml:test-inventory-security:bind-trusted-posix-toolchain-paths",
            "workflow-launch:.github/workflows/release.yml:artifacts:require-current-only-build-inventory-policy",
            "workflow-launch:.github/workflows/release.yml:artifacts:require-current-only-test-inventory-policy",
            "workflow-launch:.github/workflows/release.yml:artifacts:provision-fresh-publication-workspace",
            "workflow-launch:.github/workflows/release.yml:artifacts:verify-publication-workspace",
            "workflow-launch:.github/workflows/release.yml:artifacts:bind-trusted-posix-toolchain-paths-and-require-fortran-on-linux",
            "workflow-launch:.github/workflows/release.yml:artifacts:run-host-tool-smoke-once",
            "workflow-launch:.github/workflows/release.yml:artifacts:test",
        )
        binding_templates = {
            "workflow-launch:.github/workflows/ci.yml:source-checks:bind-trusted-posix-toolchain-paths": {
                "condition": "always",
                "shell": "/bin/bash --noprofile --norc -p -eo pipefail {0}",
                "evidence_role": "pre-repository source-check POSIX Zig/Python/PATH binding for GitHub environment-file PATH-poisoning mitigation; not immutable-tool-object, inventory, correctness, attestation, or performance evidence",
                "job_timeout_minutes": 120,
            },
            "workflow-launch:.github/workflows/ci.yml:target-tests:bind-trusted-windows-toolchain-paths": {
                "condition": "runner.os == 'Windows'",
                "shell": "C:\\Windows\\System32\\WindowsPowerShell\\v1.0\\powershell.exe -NoLogo -NoProfile -NonInteractive -Command \"$ErrorActionPreference = 'Stop'; $env:GITHUB_ENV = 'NUL'; $env:GITHUB_PATH = 'NUL'; & ([scriptblock]::Create([IO.File]::ReadAllText('{0}'))); if (Test-Path -LiteralPath variable:\\LASTEXITCODE) {{ exit $LASTEXITCODE }}\"",
                "evidence_role": "pre-repository Windows Zig/Python/PATH binding for GitHub environment-file PATH-poisoning mitigation; not immutable-tool-object, inventory, correctness, attestation, or performance evidence",
                "job_timeout_minutes": 300,
            },
            "workflow-launch:.github/workflows/ci.yml:target-tests:bind-trusted-posix-toolchain-paths-and-require-fortran-for-full-smoke": {
                "condition": "runner.os != 'Windows'",
                "shell": "/bin/bash --noprofile --norc -p -eo pipefail {0}",
                "evidence_role": "pre-repository POSIX Zig/Python/PATH binding with fail-closed Linux full-smoke Fortran prerequisite for GitHub environment-file PATH-poisoning mitigation; not immutable-tool-object, inventory, correctness, attestation, or performance evidence",
                "job_timeout_minutes": 300,
            },
            "workflow-launch:.github/workflows/release.yml:artifacts:bind-trusted-posix-toolchain-paths-and-require-fortran-on-linux": {
                "condition": "always",
                "shell": "/bin/bash --noprofile --norc -p -eo pipefail {0}",
                "evidence_role": "pre-repository release POSIX Zig/Python/PATH binding with fail-closed Linux Fortran prerequisite for GitHub environment-file PATH-poisoning mitigation; not immutable-tool-object, inventory, correctness, attestation, or performance evidence",
                "job_timeout_minutes": 180,
            },
            "workflow-launch:.github/workflows/ci.yml:capability-builds:bind-trusted-posix-toolchain-paths": {
                "condition": "always",
                "shell": "/bin/bash --noprofile --norc -p -eo pipefail {0}",
                "evidence_role": "pre-repository capability-build POSIX Zig/Python/PATH binding for GitHub environment-file PATH-poisoning mitigation; not immutable-tool-object, inventory, correctness, attestation, or performance evidence",
                "job_timeout_minutes": 45,
            },
            "workflow-launch:.github/workflows/release.yml:build-inventory-security:bind-trusted-posix-toolchain-paths": {
                "condition": "always",
                "shell": "/bin/bash --noprofile --norc -p -eo pipefail {0}",
                "evidence_role": "pre-repository release build-inventory-security POSIX Zig/Python/PATH binding for GitHub environment-file PATH-poisoning mitigation; not immutable-tool-object, inventory, correctness, attestation, or performance evidence",
                "job_timeout_minutes": 240,
            },
            "workflow-launch:.github/workflows/release.yml:test-inventory-security:bind-trusted-posix-toolchain-paths": {
                "condition": "always",
                "shell": "/bin/bash --noprofile --norc -p -eo pipefail {0}",
                "evidence_role": "pre-repository release test-inventory-security POSIX Zig/Python/PATH binding for GitHub environment-file PATH-poisoning mitigation; not immutable-tool-object, inventory, correctness, attestation, or performance evidence",
                "job_timeout_minutes": 120,
            },
        }
        for identifier, expected_fields in binding_templates.items():
            with self.subTest(reviewed_binding_template=identifier):
                reviewed_fields = CHECKER.REVIEWED_NEW_WORKFLOW_LAUNCH_FIELDS[
                    identifier
                ]
                self.assertEqual(
                    expected_fields,
                    {key: reviewed_fields[key] for key in expected_fields},
                )
        expected_workflow_template = {
            "owner": "release-validation",
            "launch_class": "workflow",
            "detail_status": "process-lifecycle-out-of-scope",
            "compile_for": "host",
            "execute_on": "workflow-runner",
            "cwd_shape": "workflow checkout",
        }
        for identifier in workflow_ids:
            with self.subTest(reviewed_workflow_template=identifier):
                template = CHECKER._new_test_inventory_workflow_launch(identifier)
                expected = dict(expected_workflow_template)
                expected.update(
                    CHECKER.REVIEWED_NEW_WORKFLOW_LAUNCH_FIELDS.get(identifier, {})
                )
                self.assertEqual(expected, template)

        publication_security_ids = {
            "workflow-launch:.github/workflows/release.yml:artifacts:provision-fresh-publication-workspace",
            "workflow-launch:.github/workflows/release.yml:artifacts:verify-publication-workspace",
        }
        with mock.patch.object(
            CHECKER, "_reviewed_source_projection_error", return_value=None
        ):
            publication_candidate = CHECKER._prepare_refreshed_source_candidate(
                self.root, self.inventory_path
            )
        candidate_workflows = {
            item["id"]: item
            for item in publication_candidate.inventory["workflow_launches"]
        }
        self.assertTrue(publication_security_ids <= set(candidate_workflows))
        for identifier in publication_security_ids:
            with self.subTest(publication_security_candidate=identifier):
                self.assertEqual(
                    expected_workflow_template,
                    {
                        key: candidate_workflows[identifier][key]
                        for key in expected_workflow_template
                    },
                )
        candidate_observations = {
            item["id"]: item
            for item in publication_candidate.inventory["build_observations"]
        }
        self.assertEqual(
            "zig-out/lib/zynum_blas.lib",
            candidate_observations["compile:build.zig:build:lib"][
                "install_destinations_by_target"
            ]["windows"]["import_library"],
        )
        self.assertEqual(
            "zig-out/lib/static/zynum_blas.lib",
            candidate_observations["compile:build.zig:build:static_lib"][
                "install_destinations_by_target"
            ]["windows"]["primary"],
        )
        for identifier in CHECKER.DEFAULT_EXECUTABLE_INSTALL_IDS:
            self.assertEqual(
                "install step is reached",
                candidate_observations[identifier]["condition"],
            )
        for identifier in CHECKER.WINDOWS_BENCHMARK_PROBE_COMPILE_SOURCES:
            self.assertEqual(
                "benchmark-probe-runtime-evidence",
                candidate_observations[identifier]["evidence_role_by_target"][
                    "windows"
                ],
            )
        candidate_paths = {
            item["path"]
            for item in publication_candidate.inventory[
                "repository_file_classifications"
            ]
        }
        self.assertTrue(
            CHECKER.NEW_REVIEWED_TEST_INFRASTRUCTURE_CLASSIFICATIONS <= candidate_paths
        )
        self.assertTrue(
            {
                "gap:windows-library-install-collision",
                "gap:windows-default-install-executables",
            }.isdisjoint(
                item["id"] for item in publication_candidate.inventory["current_gaps"]
            )
        )

        artifact_contract_mutations = []
        collision = copy.deepcopy(publication_candidate.inventory)
        collision_observations = {
            item["id"]: item for item in collision["build_observations"]
        }
        collision_observations["compile:build.zig:build:static_lib"][
            "install_destinations_by_target"
        ]["windows"]["primary"] = "zig-out/lib/zynum_blas.lib"
        artifact_contract_mutations.append(
            (collision, "Windows dynamic, import, and static library destinations")
        )
        fixture_runtime = copy.deepcopy(publication_candidate.inventory)
        fixture_observations = {
            item["id"]: item for item in fixture_runtime["build_observations"]
        }
        fixture_observations["compile:build.zig:build:rank_k_probe"][
            "evidence_role_by_target"
        ]["windows"] = "python-tooling-fixture-only"
        artifact_contract_mutations.append(
            (fixture_runtime, "Windows benchmark probe evidence_role_by_target changed")
        )
        enabled_link = copy.deepcopy(publication_candidate.inventory)
        enabled_observations = {
            item["id"]: item for item in enabled_link["build_observations"]
        }
        enabled_observations[CHECKER.LEVEL2_WIDTH_ENABLED_ARTIFACT_LINK_ID][
            "provider"
        ] = "level2_width_isolated_test_library"
        artifact_contract_mutations.append((enabled_link, "required provider changed"))
        enabled_condition = copy.deepcopy(publication_candidate.inventory)
        enabled_condition_observations = {
            item["id"]: item for item in enabled_condition["build_observations"]
        }
        enabled_condition_observations[
            CHECKER.LEVEL2_WIDTH_ENABLED_ARTIFACT_BUILD_STEP_ID
        ]["direct_dependencies"][1]["condition"] = "always"
        artifact_contract_mutations.append(
            (enabled_condition, "required direct_dependencies changed")
        )
        for mutated_inventory, expected_error in artifact_contract_mutations:
            with self.subTest(artifact_contract_mutation=expected_error):
                mutation_errors: list[str] = []
                CHECKER._validate_level2_width_and_windows_artifact_contract(
                    mutated_inventory, mutation_errors
                )
                self.assertIn(expected_error, "\n".join(mutation_errors))

        stub_path = self.root / CHECKER.LEVEL2_WIDTH_STUB_ROOT_PATH
        stub_source = stub_path.read_text(encoding="utf-8")
        stub_mutations = (
            (
                "enabled-object-import",
                stub_source.replace(
                    'const abi = @import("kernels/isolated/x86_64_level2_width_abi.zig");',
                    'const abi = @import("kernels/isolated/x86_64_level2_width_abi.zig");\n'
                    'const enabled_object = @import("kernels/isolated/x86_64_level2_width_object.zig");',
                    1,
                ),
            ),
            (
                "fixed-simd-import",
                stub_source.replace(
                    'const abi = @import("kernels/isolated/x86_64_level2_width_abi.zig");',
                    'const abi = @import("kernels/isolated/x86_64_level2_width_abi.zig");\n'
                    'const fixed_simd = @import("kernels/shared/matrix_vector/fixed_simd.zig");',
                    1,
                ),
            ),
            (
                "enabled-object-forwarder",
                stub_source.replace(
                    'const abi = @import("kernels/isolated/x86_64_level2_width_abi.zig");',
                    'const abi = @import("kernels/isolated/x86_64_level2_width_abi.zig");\n'
                    'const enabled_object = @import("kernels/isolated/x86_64_level2_width_object.zig");',
                    1,
                ).replace(
                    "fn execute(_: *abi.Request) callconv(.c) u8 {\n    return 0;\n}",
                    "fn execute(request: *abi.Request) callconv(.c) u8 {\n"
                    "    return enabled_object.execute(request);\n"
                    "}",
                    1,
                ),
            ),
            (
                "request-body-access",
                stub_source.replace(
                    "fn execute(_: *abi.Request) callconv(.c) u8 {\n    return 0;\n}",
                    "fn execute(request: *abi.Request) callconv(.c) u8 {\n"
                    "    return request.operation;\n"
                    "}",
                    1,
                ),
            ),
        )
        for label, mutation in stub_mutations:
            with self.subTest(level2_width_stub_mutation=label):
                self.assertNotEqual(stub_source, mutation)
                stub_path.write_text(mutation, encoding="utf-8")
                self._assert_error_contains(
                    "Level 2 width disabled object root must remain the exact ABI-only byte-zero rejector"
                )
        stub_path.write_text(stub_source, encoding="utf-8")
        self.assertEqual(original_bytes, self.inventory_path.read_bytes())
        for subject, call in (
            (
                "build observation",
                lambda: CHECKER._new_test_inventory_observation(
                    "step:build.zig:build:unreviewed", original_inventory
                ),
            ),
            (
                "Python launch",
                lambda: CHECKER._new_test_inventory_python_launch(
                    "python-launch:unreviewed"
                ),
            ),
            (
                "workflow launch",
                lambda: CHECKER._new_test_inventory_workflow_launch(
                    "workflow-launch:unreviewed"
                ),
            ),
        ):
            with (
                self.subTest(unreviewed_template=subject),
                self.assertRaises(CHECKER.InventoryError),
            ):
                call()
        with (
            contextlib.redirect_stderr(io.StringIO()),
            self.assertRaises(SystemExit) as invalid_arguments,
        ):
            CHECKER.main(["--not-a-build-inventory-option"])
        self.assertEqual(2, invalid_arguments.exception.code)
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            current_only_status = CHECKER.main(
                [
                    "--root",
                    str(self.root),
                    "--inventory",
                    str(self.inventory_path),
                    "--require-current-only",
                ]
            )
        if CHECKER.NEXT_SOURCE_PROJECTION_SHA256 is None:
            self.assertEqual(0, current_only_status)
            self.assertNotIn("current-only policy", stderr.getvalue())
        else:
            self.assertEqual(1, current_only_status)
            self.assertEqual(
                "build inventory error: current-only policy requires "
                "NEXT_SOURCE_PROJECTION_SHA256 to be empty\n",
                stderr.getvalue(),
            )
        self.assertEqual(original_bytes, self.inventory_path.read_bytes())
        reordered = {
            key: original_inventory[key] for key in reversed(original_inventory)
        }
        self.assertEqual(
            CHECKER._source_projection_digest(original_inventory),
            CHECKER._source_projection_digest(reordered),
        )

        before_temporaries = set(self.inventory_path.parent.glob(".*.tmp"))
        stderr = io.StringIO()
        frozen_observations = {
            section: copy.deepcopy(original_inventory[section])
            for section in (
                "build_observations",
                "python_launches",
                "workflow_launches",
                "generator_targets",
            )
        }
        frozen_observations["workflow_launches"][0]["source_digest"] = "f" * 64
        with (
            mock.patch.object(CHECKER, "discover", return_value=frozen_observations),
            contextlib.redirect_stderr(stderr),
        ):
            drift_status = CHECKER.main(
                [
                    "--root",
                    str(self.root),
                    "--inventory",
                    str(self.inventory_path),
                    "--refresh-source-derived",
                ]
            )
        self.assertEqual(1, drift_status)
        self.assertRegex(
            stderr.getvalue(),
            r"unreviewed source projection observed sha256=[0-9a-f]{64}",
        )
        self.assertEqual(original_bytes, self.inventory_path.read_bytes())
        self.assertEqual(
            before_temporaries, set(self.inventory_path.parent.glob(".*.tmp"))
        )

        candidate_inventory = copy.deepcopy(original_inventory)
        candidate_inventory["workflow_source_digests"] = dict(
            candidate_inventory["workflow_source_digests"]
        )
        first_workflow = sorted(candidate_inventory["workflow_source_digests"])[0]
        candidate_inventory["workflow_source_digests"][first_workflow] = "1" * 64
        candidate_digest = CHECKER._source_projection_digest(candidate_inventory)
        second_candidate = copy.deepcopy(candidate_inventory)
        second_candidate["workflow_source_digests"][first_workflow] = "2" * 64
        second_digest = CHECKER._source_projection_digest(second_candidate)
        self.assertNotEqual(candidate_digest, second_digest)
        with (
            self.subTest(source_projection_policy="next-window-second-candidate"),
            mock.patch.object(
                CHECKER, "NEXT_SOURCE_PROJECTION_SHA256", candidate_digest
            ),
        ):
            self.assertIsNone(
                CHECKER._reviewed_source_projection_error(candidate_inventory)
            )
            self.assertIn(
                second_digest,
                CHECKER._reviewed_source_projection_error(second_candidate) or "",
            )
            candidate_bytes = CHECKER._canonical_inventory_bytes(candidate_inventory)
            expected_snapshot = CHECKER._read_regular_stable_snapshot(
                self.inventory_path,
                CHECKER.SOURCE_REFRESH_MAX_BYTES,
                "test inventory",
            )
            prepared = CHECKER.RefreshedSourceCandidate(
                inventory=candidate_inventory,
                bytes=candidate_bytes,
                expected_snapshot=expected_snapshot,
                projection_sha256=candidate_digest,
            )
            with (
                mock.patch.object(
                    CHECKER,
                    "_prepare_refreshed_source_candidate",
                    return_value=prepared,
                ),
                mock.patch.object(CHECKER, "validate", return_value=[]),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                self.assertEqual(
                    0,
                    CHECKER.main(
                        [
                            "--root",
                            str(self.root),
                            "--inventory",
                            str(self.inventory_path),
                            "--refresh-source-derived",
                        ]
                    ),
                )
            self.assertEqual(candidate_bytes, self.inventory_path.read_bytes())
        with (
            self.subTest(source_projection_policy="promotion"),
            mock.patch.object(
                CHECKER, "CURRENT_SOURCE_PROJECTION_SHA256", candidate_digest
            ),
            mock.patch.object(CHECKER, "NEXT_SOURCE_PROJECTION_SHA256", None),
        ):
            self.assertIsNone(
                CHECKER._reviewed_source_projection_error(candidate_inventory)
            )

        self.inventory_path.write_bytes(original_bytes)
        direct_snapshot = CHECKER._read_regular_stable_snapshot(
            self.inventory_path,
            CHECKER.SOURCE_REFRESH_MAX_BYTES,
            "test inventory",
        )
        with (
            mock.patch.object(CHECKER.os, "open") as publication_open,
            self.assertRaisesRegex(CHECKER.InventoryError, candidate_digest),
        ):
            CHECKER._publish_inventory_atomic(
                self.inventory_path, candidate_bytes, direct_snapshot
            )
        publication_open.assert_not_called()
        invalid_policies = (
            "A" * 64,
            "0" * 63,
            CHECKER.CURRENT_SOURCE_PROJECTION_SHA256,
        )
        for invalid_policy in invalid_policies:
            with (
                self.subTest(invalid_policy=invalid_policy),
                mock.patch.object(
                    CHECKER, "NEXT_SOURCE_PROJECTION_SHA256", invalid_policy
                ),
                mock.patch.object(CHECKER.os, "fsync") as fsync,
                contextlib.redirect_stderr(io.StringIO()),
            ):
                self.assertEqual(
                    1,
                    CHECKER.main(
                        [
                            "--root",
                            str(self.root),
                            "--inventory",
                            str(self.inventory_path),
                            "--refresh-source-derived",
                        ]
                    ),
                )
                fsync.assert_not_called()
                self.assertEqual(original_bytes, self.inventory_path.read_bytes())

        for label, payload, cap_patch, expected in (
            (
                "oversize",
                b" " * (CHECKER.SOURCE_REFRESH_MAX_BYTES + 1),
                contextlib.nullcontext(),
                "exceeds",
            ),
            (
                "depth",
                ("[" * 130 + "0" + "]" * 130).encode(),
                contextlib.nullcontext(),
                "depth",
            ),
            (
                "nodes",
                original_bytes,
                mock.patch.object(CHECKER, "SOURCE_REFRESH_MAX_JSON_NODES", 4),
                "nodes",
            ),
            (
                "collection",
                original_bytes,
                mock.patch.object(CHECKER, "SOURCE_REFRESH_MAX_COLLECTION_ITEMS", 1),
                "collection",
            ),
        ):
            with self.subTest(bound=label):
                self.inventory_path.write_bytes(payload)
                with (
                    cap_patch,
                    self.assertRaisesRegex(CHECKER.InventoryError, expected),
                ):
                    CHECKER._prepare_refreshed_source_candidate(
                        self.root, self.inventory_path
                    )
                self.assertEqual(payload, self.inventory_path.read_bytes())
        self.inventory_path.write_bytes(original_bytes)

        alternate = self.inventory_path.with_name("build_inventory.real.json")
        self.inventory_path.replace(alternate)
        self.inventory_path.symlink_to(alternate.name)
        try:
            with self.assertRaisesRegex(CHECKER.InventoryError, "cannot read"):
                CHECKER._prepare_refreshed_source_candidate(
                    self.root, self.inventory_path
                )
        finally:
            self.inventory_path.unlink()
            alternate.replace(self.inventory_path)
        fifo = self.inventory_path.with_name("build_inventory.fifo")
        os.mkfifo(fifo)
        try:
            with self.assertRaisesRegex(CHECKER.InventoryError, "not a regular file"):
                CHECKER._read_regular_stable_snapshot(
                    fifo, CHECKER.SOURCE_REFRESH_MAX_BYTES, "FIFO inventory"
                )
        finally:
            fifo.unlink()

        def publish_current() -> None:
            snapshot = CHECKER._read_regular_stable_snapshot(
                self.inventory_path,
                CHECKER.SOURCE_REFRESH_MAX_BYTES,
                "test inventory",
            )
            CHECKER._publish_inventory_atomic(
                self.inventory_path, original_bytes, snapshot
            )

        publish_current()
        self.assertEqual(original_bytes, self.inventory_path.read_bytes())

        actual_read_snapshot = CHECKER._read_regular_stable_snapshot
        mutated_bytes = bytearray(original_bytes)
        mutated_bytes[-2] = ord(" ")
        mutated_bytes = bytes(mutated_bytes)

        def mutate_same_inode(
            path: Any, limit: int, subject: str, **kwargs: Any
        ) -> Any:
            if subject == "inventory publication target":
                self.inventory_path.write_bytes(mutated_bytes)
            return actual_read_snapshot(path, limit, subject, **kwargs)

        with (
            mock.patch.object(
                CHECKER, "_read_regular_stable_snapshot", side_effect=mutate_same_inode
            ),
            self.assertRaisesRegex(CHECKER.InventoryError, "changed since refresh"),
        ):
            publish_current()
        self.assertEqual(mutated_bytes, self.inventory_path.read_bytes())
        self.inventory_path.write_bytes(original_bytes)

        replacement = self.inventory_path.with_name("replacement.json")

        def replace_live_path(
            path: Any, limit: int, subject: str, **kwargs: Any
        ) -> Any:
            if subject == "inventory publication target":
                replacement.write_bytes(mutated_bytes)
                os.replace(replacement, self.inventory_path)
            return actual_read_snapshot(path, limit, subject, **kwargs)

        with (
            mock.patch.object(
                CHECKER, "_read_regular_stable_snapshot", side_effect=replace_live_path
            ),
            self.assertRaisesRegex(CHECKER.InventoryError, "changed since refresh"),
        ):
            publish_current()
        self.assertEqual(mutated_bytes, self.inventory_path.read_bytes())
        self.inventory_path.write_bytes(original_bytes)

        temporary_glob = f".{self.inventory_path.name}.*.tmp"
        cleanup_arena_path = self.inventory_path.parent / (
            f".zynum-cleanup-v2-{os.geteuid()}"
        )
        cleanup_arena_identity: tuple[int, int] | None = None

        def assert_cleanup_arena_empty() -> None:
            nonlocal cleanup_arena_identity
            metadata = cleanup_arena_path.lstat()
            self.assertTrue(stat.S_ISDIR(metadata.st_mode))
            self.assertEqual(os.geteuid(), metadata.st_uid)
            self.assertEqual(0o700, stat.S_IMODE(metadata.st_mode))
            observed_identity = (metadata.st_dev, metadata.st_ino)
            if cleanup_arena_identity is None:
                cleanup_arena_identity = observed_identity
            else:
                self.assertEqual(cleanup_arena_identity, observed_identity)
            self.assertEqual([], list(cleanup_arena_path.iterdir()))

        def assert_publish_failure(
            patcher: contextlib.AbstractContextManager[Any], expected: str
        ) -> None:
            before = self.inventory_path.read_bytes()
            with patcher, self.assertRaisesRegex(CHECKER.InventoryError, expected):
                publish_current()
            self.assertEqual(before, self.inventory_path.read_bytes())
            self.assertEqual([], list(self.inventory_path.parent.glob(temporary_glob)))
            assert_cleanup_arena_empty()

        actual_write = CHECKER.os.write
        partial = True

        def short_write(descriptor: int, payload: bytes) -> int:
            nonlocal partial
            if partial and len(payload) > 1:
                partial = False
                return actual_write(descriptor, payload[: len(payload) // 2])
            return actual_write(descriptor, payload)

        with mock.patch.object(CHECKER.os, "write", side_effect=short_write):
            publish_current()
        self.assertEqual(original_bytes, self.inventory_path.read_bytes())
        assert_publish_failure(
            mock.patch.object(
                CHECKER.os,
                "write",
                side_effect=OSError(errno.ENOSPC, "fixture ENOSPC"),
            ),
            "cannot publish",
        )

        actual_fsync = CHECKER.os.fsync
        fsync_calls = 0

        def fail_temp_fsync(descriptor: int) -> None:
            nonlocal fsync_calls
            fsync_calls += 1
            if fsync_calls == 1:
                raise OSError(errno.ENOSPC, "fixture fsync")
            actual_fsync(descriptor)

        assert_publish_failure(
            mock.patch.object(CHECKER.os, "fsync", side_effect=fail_temp_fsync),
            "cannot publish",
        )
        snapshot = CHECKER._read_regular_stable_snapshot(
            self.inventory_path,
            CHECKER.SOURCE_REFRESH_MAX_BYTES,
            "test inventory",
        )
        actual_close = CHECKER.os.close
        close_token = "6" * 24
        close_name = f".{self.inventory_path.name}.{close_token}.tmp"
        close_path = self.inventory_path.with_name(close_name)
        close_owned_path = close_path.with_name(close_name + ".owned")
        temporary_close_attempts = 0
        reused_descriptor = -1

        def fail_temporary_close(descriptor: int) -> None:
            nonlocal temporary_close_attempts, reused_descriptor
            if temporary_close_attempts == 0:
                temporary_close_attempts += 1
                actual_close(descriptor)
                os.replace(close_path, close_owned_path)
                close_path.write_bytes(b"foreign-close")
                reused_descriptor = os.open(os.devnull, os.O_RDONLY)
                self.assertEqual(descriptor, reused_descriptor)
                raise OSError(errno.EIO, "fixture temporary close")
            actual_close(descriptor)

        try:
            with (
                mock.patch.object(
                    CHECKER.secrets, "token_hex", return_value=close_token
                ),
                mock.patch.object(
                    CHECKER.os, "close", side_effect=fail_temporary_close
                ),
                mock.patch.object(
                    CHECKER.os, "rename", wraps=CHECKER.os.rename
                ) as close_rename_mock,
                mock.patch.object(
                    CHECKER.os, "unlink", wraps=CHECKER.os.unlink
                ) as close_unlink_mock,
                mock.patch.object(
                    CHECKER.os, "rmdir", wraps=CHECKER.os.rmdir
                ) as close_rmdir_mock,
                mock.patch.object(
                    CHECKER, "_publication_capability_error", return_value=None
                ),
                self.assertRaises(CHECKER.InventoryError) as close_error,
            ):
                CHECKER._publish_inventory_atomic(
                    self.inventory_path, original_bytes, snapshot
                )
            self.assertEqual(1, temporary_close_attempts)
            self.assertIn("descriptor close failed", str(close_error.exception))
            self.assertIn("state is unknown", str(close_error.exception))
            self.assertIn(
                f"unclaimed candidate was observed present at {close_path}",
                str(close_error.exception),
            )
            self.assertNotIn("recovery material retained", str(close_error.exception))
            close_rename_mock.assert_not_called()
            close_unlink_mock.assert_not_called()
            close_rmdir_mock.assert_not_called()
            os.fstat(reused_descriptor)
            self.assertEqual(b"foreign-close", close_path.read_bytes())
            self.assertEqual(original_bytes, close_owned_path.read_bytes())
            self.assertEqual(original_bytes, self.inventory_path.read_bytes())
            assert_cleanup_arena_empty()
        finally:
            if reused_descriptor >= 0:
                actual_close(reused_descriptor)
            close_path.unlink(missing_ok=True)
            close_owned_path.unlink(missing_ok=True)

        unknown_identity_token = "d" * 24
        unknown_identity_name = (
            f".{self.inventory_path.name}.{unknown_identity_token}.tmp"
        )
        unknown_identity_path = self.inventory_path.with_name(unknown_identity_name)
        actual_fstat = CHECKER.os.fstat
        failed_fstat = False

        def fail_temporary_fstat(descriptor: int) -> Any:
            nonlocal failed_fstat
            if not failed_fstat:
                failed_fstat = True
                raise OSError(errno.EIO, "fixture temporary fstat")
            return actual_fstat(descriptor)

        with (
            mock.patch.object(
                CHECKER.secrets, "token_hex", return_value=unknown_identity_token
            ),
            mock.patch.object(CHECKER.os, "fstat", side_effect=fail_temporary_fstat),
            self.assertRaises(CHECKER.InventoryError) as unknown_identity_error,
        ):
            CHECKER._publish_inventory_atomic(
                self.inventory_path, original_bytes, snapshot
            )
        self.assertIn(
            "temporary identity is unknown", str(unknown_identity_error.exception)
        )
        self.assertIn(
            f"unclaimed candidate was observed present at {unknown_identity_path}",
            str(unknown_identity_error.exception),
        )
        self.assertNotIn(
            "recovery material retained", str(unknown_identity_error.exception)
        )
        self.assertTrue(unknown_identity_path.exists())
        self.assertEqual(b"", unknown_identity_path.read_bytes())
        self.assertEqual(original_bytes, self.inventory_path.read_bytes())
        self.assertEqual(
            [],
            list(
                self.inventory_path.parent.glob(
                    f".{unknown_identity_name}.*.quarantine"
                )
            ),
        )
        unknown_identity_path.unlink()
        assert_cleanup_arena_empty()

        unknown_close_token = "c" * 24
        unknown_close_name = f".{self.inventory_path.name}.{unknown_close_token}.tmp"
        unknown_close_path = self.inventory_path.with_name(unknown_close_name)
        failed_fstat = False
        unknown_descriptor = -1
        unknown_close_attempts = 0
        actual_candidate_stat = CHECKER.os.stat

        def fail_fstat_and_record_descriptor(descriptor: int) -> Any:
            nonlocal failed_fstat, unknown_descriptor
            if not failed_fstat:
                failed_fstat = True
                unknown_descriptor = descriptor
                raise OSError(errno.EIO, "fixture temporary fstat")
            return actual_fstat(descriptor)

        def fail_unknown_temporary_close(descriptor: int) -> None:
            nonlocal unknown_close_attempts
            if descriptor == unknown_descriptor:
                unknown_close_attempts += 1
                raise OSError(errno.EIO, "fixture unknown temporary close")
            actual_close(descriptor)

        def fail_unknown_candidate_stat(path: Any, *args: Any, **kwargs: Any) -> Any:
            if (
                path == unknown_close_name
                and kwargs.get("dir_fd") is not None
                and kwargs.get("follow_symlinks") is False
            ):
                raise OSError(errno.EIO, "fixture candidate observation")
            return actual_candidate_stat(path, *args, **kwargs)

        try:
            with (
                mock.patch.object(
                    CHECKER.secrets, "token_hex", return_value=unknown_close_token
                ),
                mock.patch.object(
                    CHECKER.os,
                    "fstat",
                    side_effect=fail_fstat_and_record_descriptor,
                ),
                mock.patch.object(
                    CHECKER.os,
                    "close",
                    side_effect=fail_unknown_temporary_close,
                ),
                mock.patch.object(
                    CHECKER.os,
                    "stat",
                    side_effect=fail_unknown_candidate_stat,
                ),
                mock.patch.object(
                    CHECKER, "_publication_capability_error", return_value=None
                ),
                self.assertRaises(CHECKER.InventoryError) as unknown_close_error,
            ):
                CHECKER._publish_inventory_atomic(
                    self.inventory_path, original_bytes, snapshot
                )
            self.assertEqual(1, unknown_close_attempts)
            self.assertIn(
                "exactly one close attempt", str(unknown_close_error.exception)
            )
            self.assertIn("state is unknown", str(unknown_close_error.exception))
            self.assertIn(
                f"unclaimed candidate presence is uncertain at {unknown_close_path}",
                str(unknown_close_error.exception),
            )
            self.assertNotIn(
                "recovery material retained", str(unknown_close_error.exception)
            )
            self.assertTrue(unknown_close_path.exists())
            self.assertEqual(original_bytes, self.inventory_path.read_bytes())
            self.assertEqual(
                [],
                list(
                    self.inventory_path.parent.glob(
                        f".{unknown_close_name}.*.quarantine"
                    )
                ),
            )
        finally:
            if unknown_descriptor >= 0:
                actual_close(unknown_descriptor)
            unknown_close_path.unlink(missing_ok=True)
        assert_cleanup_arena_empty()

        actual_unlink = CHECKER.os.unlink
        for claim_failure_mode in ("quarantine", "source", "both"):
            with self.subTest(claim_failure_mode=claim_failure_mode):
                claim_token = {
                    "quarantine": "a" * 24,
                    "source": "b" * 24,
                    "both": "9" * 24,
                }[claim_failure_mode]
                claim_name = f".{self.inventory_path.name}.{claim_token}.tmp"
                claim_path = self.inventory_path.with_name(claim_name)
                claim_fsync_attempts: list[str] = []
                claim_source_metadata = self.inventory_path.parent.stat()
                claim_arena_metadata = cleanup_arena_path.stat()

                def fail_claim_fsync(descriptor: int) -> None:
                    metadata = actual_fstat(descriptor)
                    descriptor_identity = (metadata.st_dev, metadata.st_ino)
                    if descriptor_identity == (
                        claim_arena_metadata.st_dev,
                        claim_arena_metadata.st_ino,
                    ):
                        actual_fsync(descriptor)
                        return
                    label = (
                        "source"
                        if descriptor_identity
                        == (
                            claim_source_metadata.st_dev,
                            claim_source_metadata.st_ino,
                        )
                        else "quarantine"
                    )
                    claim_fsync_attempts.append(label)
                    if claim_failure_mode in (label, "both"):
                        raise OSError(errno.EIO, f"fixture {label} claim fsync")
                    actual_fsync(descriptor)

                with (
                    mock.patch.object(
                        CHECKER.secrets, "token_hex", return_value=claim_token
                    ),
                    mock.patch.object(
                        CHECKER.os,
                        "write",
                        side_effect=OSError(errno.ENOSPC, "fixture claim write"),
                    ),
                    mock.patch.object(
                        CHECKER.os, "fsync", side_effect=fail_claim_fsync
                    ),
                    mock.patch.object(
                        CHECKER, "_publication_capability_error", return_value=None
                    ),
                    mock.patch.object(
                        CHECKER.os, "unlink", wraps=actual_unlink
                    ) as claim_unlink_mock,
                    self.assertRaises(CHECKER.InventoryError) as claim_error,
                ):
                    CHECKER._publish_inventory_atomic(
                        self.inventory_path, original_bytes, snapshot
                    )
                self.assertEqual(["quarantine", "source"], claim_fsync_attempts)
                self.assertIn(
                    "cleanup claim persistence failed", str(claim_error.exception)
                )
                if claim_failure_mode in ("quarantine", "both"):
                    self.assertIn(
                        "quarantine directory fsync failed",
                        str(claim_error.exception),
                    )
                else:
                    self.assertNotIn(
                        "quarantine directory fsync failed",
                        str(claim_error.exception),
                    )
                if claim_failure_mode in ("source", "both"):
                    self.assertIn(
                        "source directory fsync failed", str(claim_error.exception)
                    )
                else:
                    self.assertNotIn(
                        "source directory fsync failed", str(claim_error.exception)
                    )
                claim_unlink_mock.assert_not_called()
                self.assertFalse(claim_path.exists())
                self.assertEqual(original_bytes, self.inventory_path.read_bytes())
                claim_recoveries = list(
                    cleanup_arena_path.glob(f".{claim_name}.*.quarantine/claimed")
                )
                self.assertEqual(1, len(claim_recoveries))
                self.assertIn(str(claim_recoveries[0]), str(claim_error.exception))
                self.assertEqual(b"", claim_recoveries[0].read_bytes())
                claim_recoveries[0].unlink()
                claim_recoveries[0].parent.rmdir()
                assert_cleanup_arena_empty()
                self.assertEqual(
                    [], list(self.inventory_path.parent.glob(temporary_glob))
                )

        setup_token = "6" * 24
        setup_name = f".{self.inventory_path.name}.{setup_token}.tmp"
        setup_path = self.inventory_path.with_name(setup_name)
        setup_quarantine_path = cleanup_arena_path / (
            f".{setup_name}.{setup_token}.quarantine"
        )
        setup_cleanup_started = False
        setup_source_metadata = self.inventory_path.parent.stat()
        setup_arena_metadata = cleanup_arena_path.stat()

        def fail_setup_write(_descriptor: int, _payload: bytes) -> int:
            nonlocal setup_cleanup_started
            setup_cleanup_started = True
            raise OSError(errno.ENOSPC, "fixture setup write")

        def mutate_setup_quarantine_owner(descriptor: int) -> os.stat_result:
            metadata = actual_fstat(descriptor)
            descriptor_identity = (metadata.st_dev, metadata.st_ino)
            if not setup_cleanup_started or descriptor_identity in {
                (setup_source_metadata.st_dev, setup_source_metadata.st_ino),
                (setup_arena_metadata.st_dev, setup_arena_metadata.st_ino),
            }:
                return metadata
            if stat.S_ISDIR(metadata.st_mode):
                fields = list(metadata)
                fields[4] = os.geteuid() + 1
                return os.stat_result(fields)
            return metadata

        try:
            with (
                mock.patch.object(
                    CHECKER.secrets, "token_hex", return_value=setup_token
                ),
                mock.patch.object(CHECKER.os, "write", side_effect=fail_setup_write),
                mock.patch.object(
                    CHECKER.os,
                    "fstat",
                    side_effect=mutate_setup_quarantine_owner,
                ),
                mock.patch.object(
                    CHECKER, "_publication_capability_error", return_value=None
                ),
                self.assertRaises(CHECKER.InventoryError) as setup_error,
            ):
                publish_current()
            setup_diagnostic = str(setup_error.exception)
            self.assertTrue(setup_path.is_file())
            self.assertTrue(setup_quarantine_path.is_dir())
            self.assertEqual(1, setup_diagnostic.count(str(setup_quarantine_path)))
            self.assertEqual(1, setup_diagnostic.count(str(setup_path)))
            self.assertLess(
                setup_diagnostic.index(
                    f"recovery material retained as {setup_quarantine_path}"
                ),
                setup_diagnostic.index(
                    f"unclaimed candidate was observed present at {setup_path}"
                ),
            )
        finally:
            setup_path.unlink(missing_ok=True)
            if setup_quarantine_path.exists():
                setup_quarantine_path.rmdir()
        assert_cleanup_arena_empty()

        post_rmdir_token = "5" * 24
        post_rmdir_name = f".{self.inventory_path.name}.{post_rmdir_token}.tmp"
        post_rmdir_quarantine_path = cleanup_arena_path / (
            f".{post_rmdir_name}.{post_rmdir_token}.quarantine"
        )
        post_rmdir_arena_metadata = cleanup_arena_path.stat()
        post_rmdir_arena_fsyncs = 0

        def fail_post_rmdir_arena_fsync(descriptor: int) -> None:
            nonlocal post_rmdir_arena_fsyncs
            metadata = actual_fstat(descriptor)
            if (metadata.st_dev, metadata.st_ino) == (
                post_rmdir_arena_metadata.st_dev,
                post_rmdir_arena_metadata.st_ino,
            ):
                post_rmdir_arena_fsyncs += 1
                if post_rmdir_arena_fsyncs == 2:
                    raise OSError(errno.EIO, "fixture post-rmdir arena fsync")
            actual_fsync(descriptor)

        with (
            mock.patch.object(
                CHECKER.secrets, "token_hex", return_value=post_rmdir_token
            ),
            mock.patch.object(
                CHECKER.os,
                "write",
                side_effect=OSError(errno.ENOSPC, "fixture post-rmdir write"),
            ),
            mock.patch.object(
                CHECKER.os, "fsync", side_effect=fail_post_rmdir_arena_fsync
            ),
            mock.patch.object(
                CHECKER, "_publication_capability_error", return_value=None
            ),
            self.assertRaises(CHECKER.InventoryError) as post_rmdir_error,
        ):
            publish_current()
        self.assertEqual(2, post_rmdir_arena_fsyncs)
        self.assertFalse(post_rmdir_quarantine_path.exists())
        self.assertNotIn(
            str(post_rmdir_quarantine_path), str(post_rmdir_error.exception)
        )
        self.assertNotIn("recovery material retained", str(post_rmdir_error.exception))
        assert_cleanup_arena_empty()

        rename_protection_token = "4" * 24
        rename_protection_name = (
            f".{self.inventory_path.name}.{rename_protection_token}.tmp"
        )
        rename_protection_path = self.inventory_path.with_name(rename_protection_name)
        source_mode = stat.S_IMODE(self.inventory_path.parent.stat().st_mode)
        unsafe_source_mode = (source_mode | stat.S_IWGRP) & ~stat.S_ISVTX
        self.inventory_path.parent.chmod(unsafe_source_mode)
        try:
            with (
                mock.patch.object(
                    CHECKER.secrets,
                    "token_hex",
                    return_value=rename_protection_token,
                ),
                mock.patch.object(
                    CHECKER.os,
                    "write",
                    side_effect=OSError(
                        errno.ENOSPC, "fixture rename-protection write"
                    ),
                ),
                mock.patch.object(
                    CHECKER, "_publication_capability_error", return_value=None
                ),
                self.assertRaises(CHECKER.InventoryError) as rename_protection_error,
            ):
                publish_current()
            rename_protection_diagnostic = str(rename_protection_error.exception)
            self.assertIn(
                "cleanup_recovery_anchor_not_rename_protected",
                rename_protection_diagnostic,
            )
            self.assertIn(
                f"unclaimed candidate was observed present at {rename_protection_path}",
                rename_protection_diagnostic,
            )
            self.assertIn(
                "cleanup namespace is unaddressable; arena binding is unknown",
                rename_protection_diagnostic,
            )
            self.assertTrue(rename_protection_path.is_file())
        finally:
            self.inventory_path.parent.chmod(source_mode)
            rename_protection_path.unlink(missing_ok=True)
        assert_cleanup_arena_empty()

        rebind_token = "3" * 24
        rebind_name = f".{self.inventory_path.name}.{rebind_token}.tmp"
        rebind_path = self.inventory_path.with_name(rebind_name)
        actual_stat = CHECKER.os.stat
        rebound_observations = 0

        def report_rebound_source(path: Any, *args: Any, **kwargs: Any) -> Any:
            nonlocal rebound_observations
            metadata = actual_stat(path, *args, **kwargs)
            if (
                kwargs.get("dir_fd") is None
                and isinstance(path, (str, bytes, os.PathLike))
                and os.fspath(path) == os.fspath(self.inventory_path.parent)
            ):
                rebound_observations += 1
                fields = list(metadata)
                fields[1] += 1
                return os.stat_result(fields)
            return metadata

        try:
            with (
                mock.patch.object(
                    CHECKER.secrets, "token_hex", return_value=rebind_token
                ),
                mock.patch.object(
                    CHECKER.os,
                    "write",
                    side_effect=OSError(errno.ENOSPC, "fixture rebind write"),
                ),
                mock.patch.object(
                    CHECKER.os, "stat", side_effect=report_rebound_source
                ),
                mock.patch.object(
                    CHECKER, "_publication_capability_error", return_value=None
                ),
                self.assertRaises(CHECKER.InventoryError) as rebind_error,
            ):
                publish_current()
            rebind_diagnostic = str(rebind_error.exception)
            self.assertEqual(1, rebound_observations)
            self.assertIn("cleanup_recovery_anchor_rebound", rebind_diagnostic)
            candidate_diagnostic = (
                f"unclaimed candidate was observed present at {rebind_path}"
            )
            namespace_diagnostic = (
                "cleanup namespace is unaddressable; arena binding is rebound"
            )
            self.assertIn(candidate_diagnostic, rebind_diagnostic)
            self.assertIn(namespace_diagnostic, rebind_diagnostic)
            self.assertLess(
                rebind_diagnostic.index(candidate_diagnostic),
                rebind_diagnostic.index(namespace_diagnostic),
            )
            self.assertTrue(rebind_path.is_file())
        finally:
            rebind_path.unlink(missing_ok=True)
        assert_cleanup_arena_empty()

        actual_fchmod = CHECKER.os.fchmod
        actual_rmdir = CHECKER.os.rmdir
        effective_uid = CHECKER.os.geteuid()
        foreign_uid = effective_uid + 1
        for owner_failure_phase, owner_fstat_call, expected_owner_error in (
            (
                "initial",
                1,
                "initial credential is unsafe",
            ),
            ("post-fchmod", 2, "configured credential is unsafe"),
        ):
            with self.subTest(quarantine_owner_failure=owner_failure_phase):
                owner_token = {
                    "initial": "8" * 32,
                    "post-fchmod": "7" * 32,
                }[owner_failure_phase]
                owner_temporary_name = f"owner-{owner_failure_phase}.tmp"
                owner_quarantine_name = (
                    f".{owner_temporary_name}.{owner_token}.quarantine"
                )
                owner_quarantine_path = cleanup_arena_path / owner_quarantine_name
                owner_fstat_calls = 0
                owner_close_descriptors: list[int] = []
                owner_parent_descriptor = os.open(
                    self.inventory_path.parent, os.O_RDONLY
                )
                arena_metadata = cleanup_arena_path.stat()

                def mutate_quarantine_owner(descriptor: int) -> os.stat_result:
                    nonlocal owner_fstat_calls
                    metadata = actual_fstat(descriptor)
                    if descriptor == owner_parent_descriptor or (
                        metadata.st_dev,
                        metadata.st_ino,
                    ) == (arena_metadata.st_dev, arena_metadata.st_ino):
                        return metadata
                    owner_fstat_calls += 1
                    if owner_fstat_calls == owner_fstat_call:
                        fields = list(metadata)
                        fields[4] = foreign_uid
                        return os.stat_result(fields)
                    return metadata

                def close_quarantine_once(descriptor: int) -> None:
                    owner_close_descriptors.append(descriptor)
                    actual_close(descriptor)

                try:
                    with (
                        mock.patch.object(
                            CHECKER.secrets, "token_hex", return_value=owner_token
                        ),
                        mock.patch.object(
                            CHECKER.os,
                            "fstat",
                            side_effect=mutate_quarantine_owner,
                        ),
                        mock.patch.object(
                            CHECKER.os,
                            "fchmod",
                            wraps=actual_fchmod,
                        ) as owner_fchmod_mock,
                        mock.patch.object(
                            CHECKER.os,
                            "close",
                            side_effect=close_quarantine_once,
                        ),
                        mock.patch.object(
                            CHECKER.os, "unlink", wraps=actual_unlink
                        ) as owner_unlink_mock,
                        mock.patch.object(
                            CHECKER.os, "rmdir", wraps=actual_rmdir
                        ) as owner_rmdir_mock,
                        self.assertRaises(
                            CHECKER.repository_snapshot.CleanupFailure
                        ) as owner_error,
                    ):
                        CHECKER.repository_snapshot.CleanupQuarantine.create(
                            CHECKER.repository_snapshot.DirectoryAnchor(
                                owner_parent_descriptor,
                                self.inventory_path.parent,
                            ),
                            owner_temporary_name,
                            quarantine_prefix=f".{owner_temporary_name}.",
                            quarantine_suffix=".quarantine",
                        )
                    self.assertEqual(owner_fstat_call, owner_fstat_calls)
                    self.assertEqual(
                        0 if owner_failure_phase == "initial" else 1,
                        owner_fchmod_mock.call_count,
                    )
                    self.assertEqual(2, len(owner_close_descriptors))
                    self.assertEqual(
                        len(owner_close_descriptors),
                        len(set(owner_close_descriptors)),
                    )
                    self.assertIn(
                        expected_owner_error,
                        str(owner_error.exception.outcome.issues[0].error),
                    )
                    self.assertIn(
                        owner_quarantine_path,
                        owner_error.exception.outcome.recovery_paths,
                    )
                    owner_unlink_mock.assert_not_called()
                    owner_rmdir_mock.assert_not_called()
                    self.assertTrue(owner_quarantine_path.is_dir())
                    self.assertEqual(original_bytes, self.inventory_path.read_bytes())
                finally:
                    actual_close(owner_parent_descriptor)
                    if owner_quarantine_path.exists():
                        owner_quarantine_path.rmdir()
                assert_cleanup_arena_empty()

        assert_publish_failure(
            mock.patch.object(
                CHECKER.os, "replace", side_effect=OSError(errno.EIO, "fixture replace")
            ),
            "cannot publish",
        )

        token = "f" * 24
        foreign_name = f".{self.inventory_path.name}.{token}.tmp"
        owned_name = self.inventory_path.with_name(foreign_name + ".owned")
        wrote_foreign = False

        def swap_foreign_then_fail(descriptor: int, _payload: bytes) -> int:
            nonlocal wrote_foreign
            if not wrote_foreign:
                wrote_foreign = True
                os.replace(self.inventory_path.with_name(foreign_name), owned_name)
                self.inventory_path.with_name(foreign_name).write_bytes(b"foreign")
            raise OSError(errno.ENOSPC, "fixture foreign temp")

        with (
            mock.patch.object(CHECKER.secrets, "token_hex", return_value=token),
            mock.patch.object(CHECKER.os, "write", side_effect=swap_foreign_then_fail),
            self.assertRaises(CHECKER.InventoryError) as foreign_error,
        ):
            publish_current()
        foreign_recoveries = list(
            cleanup_arena_path.glob(f".{foreign_name}.*.quarantine/claimed")
        )
        self.assertEqual(1, len(foreign_recoveries))
        self.assertIn("claimed unexpected bytes", str(foreign_error.exception))
        self.assertIn(str(foreign_recoveries[0]), str(foreign_error.exception))
        self.assertEqual(b"foreign", foreign_recoveries[0].read_bytes())
        self.assertFalse(self.inventory_path.with_name(foreign_name).exists())
        foreign_recoveries[0].unlink()
        foreign_recoveries[0].parent.rmdir()
        owned_name.unlink(missing_ok=True)
        assert_cleanup_arena_empty()

        cleanup_race_token = "e" * 24
        cleanup_race_name = f".{self.inventory_path.name}.{cleanup_race_token}.tmp"
        cleanup_race_path = self.inventory_path.with_name(cleanup_race_name)
        cleanup_race_replacement = self.inventory_path.with_name(
            cleanup_race_name + ".replacement"
        )
        actual_cleanup_claim = CHECKER.repository_snapshot.CleanupQuarantine.claim
        replaced_before_claim = False

        def replace_before_claim(
            quarantine: Any,
        ) -> None:
            nonlocal replaced_before_claim
            if not replaced_before_claim:
                replaced_before_claim = True
                cleanup_race_replacement.write_bytes(b"cleanup-race-foreign")
                os.replace(cleanup_race_replacement, cleanup_race_path)
            actual_cleanup_claim(quarantine)

        with (
            mock.patch.object(
                CHECKER.secrets, "token_hex", return_value=cleanup_race_token
            ),
            mock.patch.object(
                CHECKER.os,
                "write",
                side_effect=OSError(errno.ENOSPC, "fixture cleanup race"),
            ),
            mock.patch.object(
                CHECKER.repository_snapshot.CleanupQuarantine,
                "claim",
                autospec=True,
                side_effect=replace_before_claim,
            ),
            self.assertRaises(CHECKER.InventoryError) as cleanup_race_error,
        ):
            publish_current()
        self.assertTrue(replaced_before_claim)
        self.assertEqual(original_bytes, self.inventory_path.read_bytes())
        cleanup_race_recoveries = list(
            cleanup_arena_path.glob(f".{cleanup_race_name}.*.quarantine/claimed")
        )
        self.assertEqual(1, len(cleanup_race_recoveries))
        self.assertIn(
            str(cleanup_race_recoveries[0]), str(cleanup_race_error.exception)
        )
        self.assertEqual(
            b"cleanup-race-foreign", cleanup_race_recoveries[0].read_bytes()
        )
        cleanup_race_recoveries[0].unlink()
        cleanup_race_recoveries[0].parent.rmdir()
        assert_cleanup_arena_empty()
        self.assertFalse(cleanup_race_path.exists())
        self.assertFalse(cleanup_race_replacement.exists())
        self.assertEqual([], list(self.inventory_path.parent.glob(temporary_glob)))
        self.assertEqual(
            [],
            list(cleanup_arena_path.glob(f".{cleanup_race_name}.*.quarantine")),
        )

        tools_directory = self.inventory_path.parent
        moved_tools_directory = self.root / "tools-before-parent-swap"
        foreign_tools_directory = self.root / "tools-foreign-parent"
        snapshot = CHECKER._read_regular_stable_snapshot(
            self.inventory_path,
            CHECKER.SOURCE_REFRESH_MAX_BYTES,
            "test inventory",
        )
        actual_replace = CHECKER.os.replace
        swapped_parent = False

        def swap_parent_then_replace(
            source: str,
            destination: str,
            *,
            src_dir_fd: int,
            dst_dir_fd: int,
        ) -> None:
            nonlocal swapped_parent
            if not swapped_parent:
                swapped_parent = True
                os.rename(tools_directory, moved_tools_directory)
                tools_directory.mkdir()
                (tools_directory / self.inventory_path.name).write_bytes(
                    b"foreign-parent"
                )
            actual_replace(
                source,
                destination,
                src_dir_fd=src_dir_fd,
                dst_dir_fd=dst_dir_fd,
            )

        try:
            with (
                mock.patch.object(
                    CHECKER, "NEXT_SOURCE_PROJECTION_SHA256", candidate_digest
                ),
                mock.patch.object(
                    CHECKER.os, "replace", side_effect=swap_parent_then_replace
                ),
            ):
                CHECKER._publish_inventory_atomic(
                    self.inventory_path, candidate_bytes, snapshot
                )
            self.assertEqual(
                candidate_bytes,
                (moved_tools_directory / self.inventory_path.name).read_bytes(),
            )
            self.assertEqual(b"foreign-parent", self.inventory_path.read_bytes())
        finally:
            if tools_directory.exists():
                os.rename(tools_directory, foreign_tools_directory)
            if moved_tools_directory.exists():
                os.rename(moved_tools_directory, tools_directory)
            shutil.rmtree(foreign_tools_directory, ignore_errors=True)
        self.inventory_path.write_bytes(original_bytes)

        fsync_calls = 0

        def fail_directory_fsync(descriptor: int) -> None:
            nonlocal fsync_calls
            fsync_calls += 1
            if fsync_calls == 2:
                raise OSError(errno.EIO, "fixture directory fsync")
            actual_fsync(descriptor)

        snapshot = CHECKER._read_regular_stable_snapshot(
            self.inventory_path,
            CHECKER.SOURCE_REFRESH_MAX_BYTES,
            "test inventory",
        )
        indeterminate_candidate = CHECKER.RefreshedSourceCandidate(
            inventory=candidate_inventory,
            bytes=candidate_bytes,
            expected_snapshot=snapshot,
            projection_sha256=candidate_digest,
        )
        with (
            mock.patch.object(
                CHECKER, "NEXT_SOURCE_PROJECTION_SHA256", candidate_digest
            ),
            mock.patch.object(CHECKER.os, "fsync", side_effect=fail_directory_fsync),
            mock.patch.object(
                CHECKER,
                "_prepare_refreshed_source_candidate",
                return_value=indeterminate_candidate,
            ),
            contextlib.redirect_stderr(io.StringIO()),
        ):
            self.assertEqual(
                3,
                CHECKER.main(
                    [
                        "--root",
                        str(self.root),
                        "--inventory",
                        str(self.inventory_path),
                        "--refresh-source-derived",
                    ]
                ),
            )
        self.assertEqual(candidate_bytes, self.inventory_path.read_bytes())
        self.inventory_path.write_bytes(original_bytes)

        snapshot = CHECKER._read_regular_stable_snapshot(
            self.inventory_path,
            CHECKER.SOURCE_REFRESH_MAX_BYTES,
            "test inventory",
        )
        indeterminate_candidate = indeterminate_candidate._replace(
            expected_snapshot=snapshot
        )
        close_calls = 0

        def fail_directory_close(descriptor: int) -> None:
            nonlocal close_calls
            close_calls += 1
            actual_close(descriptor)
            if close_calls == 4:
                raise OSError(errno.EIO, "fixture directory close")

        with (
            mock.patch.object(
                CHECKER, "NEXT_SOURCE_PROJECTION_SHA256", candidate_digest
            ),
            mock.patch.object(CHECKER.os, "close", side_effect=fail_directory_close),
            mock.patch.object(
                CHECKER,
                "_prepare_refreshed_source_candidate",
                return_value=indeterminate_candidate,
            ),
            contextlib.redirect_stderr(io.StringIO()),
        ):
            self.assertEqual(
                3,
                CHECKER.main(
                    [
                        "--root",
                        str(self.root),
                        "--inventory",
                        str(self.inventory_path),
                        "--refresh-source-derived",
                    ]
                ),
            )
        self.assertEqual(candidate_bytes, self.inventory_path.read_bytes())
        self.inventory_path.write_bytes(original_bytes)

        inventory = original_inventory
        self.assertEqual(3, inventory["schema_version"])
        self.assertEqual("zynum-build-inventory-v3", inventory["schema_id"])
        factory = next(
            item
            for item in inventory["build_observations"]
            if item["id"] == CHECKER.TEST_INVENTORY_FACTORY_COMPILE_ID
        )
        self.assertEqual(25, factory["expansion_case_count"])
        self.assertEqual(
            [
                logical_id
                for _, logical_id, _ in CHECKER.REQUIRED_TEST_INVENTORY_FACTORY_CASES
            ],
            [
                case["logical_compile_observation_id"]
                for case in factory["expansion_cases"]
            ],
        )

    def test_unlisted_python_launch_fails_closed(self) -> None:
        path = self.root / "tools/check_abi_manifest.py"
        with path.open("a", encoding="utf-8") as output:
            output.write(
                "\ndef fixture_launch():\n    return subprocess.run(['python3', '-V'])\n"
            )
        self._assert_error_contains(
            "python-launch:tools/check_abi_manifest.py:fixture_launch:subprocess.run:1"
        )

    def test_benchmark_runner_git_identity_boundary_keeps_real_launches(self) -> None:
        runner_functions = {
            "bench/tools/run_gemm_sweep_isolated.py": {
                "run_one_process",
                "zig_version",
            },
            "bench/tools/run_level1_report.py": {
                "check_worker_result",
                "run_once",
                "zig_version",
            },
            "bench/tools/run_level2_report.py": {"run_one_process", "zig_version"},
            "bench/tools/run_rank_k_report.py": {
                "command_output",
                "run_one_process",
            },
            "bench/tools/run_rotg_latency_report.py": {
                "command_output",
                "run_one_process",
            },
            "bench/tools/run_symm_report.py": {
                "command_output",
                "run_one_process",
            },
            "bench/tools/run_triangular_matrix_report.py": {
                "command_output",
                "run_one_process",
            },
        }
        runner_paths = set(runner_functions)
        inventory_launches = self.inventory["python_launches"]
        self.assertEqual(114, len(inventory_launches))
        self.assertEqual(
            {"python-launch:tools/generate_fftw_f32.py:render:subprocess.run:1":
                "fftw-f32-generated-source-formatter"},
            {item["id"]: item["launch_class"] for item in inventory_launches
             if item["anchor"]["file"] == "tools/generate_fftw_f32.py"},
        )
        expected_module_launches = {
            "python-launch:bench/modules/build.py:run:subprocess.run:1":
                "benchmark-source-bound-host-build-and-provenance",
            "python-launch:bench/modules/run_foundation_comparison.py:capture:subprocess.run:1":
                "benchmark-host-and-repository-provenance",
            "python-launch:bench/modules/run_foundation_comparison.py:main:subprocess.run:1":
                "benchmark-source-bound-worker-transport",
            "python-launch:tools/check_checked_abi.py:run:subprocess.run:1":
                "checked-abi-artifact-inspection",
            "python-launch:tools/generate_checked_abi.py:outputs:subprocess.run:1":
                "checked-abi-generated-source-formatter",
            "python-launch:tools/generate_checked_abi.py:_render_subset:subprocess.run:1":
                "finite-standard-abi-generated-source-formatter",
        }
        self.assertEqual(
            expected_module_launches,
            {
                item["id"]: item["launch_class"]
                for item in inventory_launches
                if item["anchor"]["file"] in {
                    "bench/modules/build.py",
                    "bench/modules/run_foundation_comparison.py",
                    "tools/check_checked_abi.py",
                    "tools/generate_checked_abi.py",
                }
            },
        )
        self.assertEqual(
            {
                "python-launch:tools/generate_multiversion.py:formatted:subprocess.run:1",
                "python-launch:tools/repack_darwin_archive.py:ar:subprocess.run:1",
                "python-launch:test/build/test_repack_darwin_archive.py:test_real_macho_members_are_eight_byte_aligned:subprocess.run:1",
                "python-launch:test/build/test_repack_darwin_archive.py:test_real_macho_members_are_eight_byte_aligned:subprocess.run:2",
            },
            {item["id"] for item in inventory_launches if item["anchor"]["file"] in {
                "tools/generate_multiversion.py", "tools/repack_darwin_archive.py",
                "test/build/test_repack_darwin_archive.py",
            }},
        )
        self.assertNotIn("python_launches", CHECKER.REQUIRED_SECTION_FACT_DIGESTS)
        self.assertEqual(
            inventory_launches,
            CHECKER._source_projection(self.inventory)["python_launches"],
        )

        expected_runner_ids = {
            f"python-launch:{path}:{function}:subprocess.run:1"
            for path, functions in runner_functions.items()
            for function in functions
        }
        actual_runner_ids = {
            item["id"]
            for item in inventory_launches
            if item["anchor"]["file"] in runner_paths
        }
        self.assertEqual(15, len(expected_runner_ids))
        self.assertEqual(expected_runner_ids, actual_runner_ids)
        self.assertEqual(
            {
                "python-launch:bench/tools/run_windows_benchmark.py:command_text:subprocess.run:1",
                "python-launch:bench/tools/run_windows_benchmark.py:run_probe_process:subprocess.run:1",
            },
            {
                item["id"] for item in inventory_launches
                if item["anchor"]["file"] == "bench/tools/run_windows_benchmark.py"
            },
        )

        expected_test_inventory_runner_launches = {
            CHECKER.TEST_INVENTORY_RUNNER_COMPILE_PYTHON_LAUNCH_ID: (
                "test_runner_protocol_and_isolated_object_mutations_fail",
                "subprocess.run",
                "test-inventory-runner-fixture-compile",
                [
                    "zig",
                    "test",
                    "<fixture>/digest_vectors.zig",
                    "--name",
                    "digest_vectors",
                    "-O",
                    "debug",
                    "-target",
                    "<host-native-baseline-target>",
                    "-mcpu",
                    "baseline",
                    "--test-runner",
                    "<fixture>/tools/test_inventory_vector_runner.zig",
                    "--test-no-exec",
                    "-femit-bin=<fixture>/test-inventory-digest-vectors",
                ],
            ),
            CHECKER.TEST_INVENTORY_RUNNER_EXECUTE_PYTHON_LAUNCH_ID: (
                "run_vector_inventory",
                "subprocess.run",
                "test-inventory-runner-fixture-execute",
                [
                    "./test-inventory-digest-vectors",
                    "<fixture-inventory-path>",
                    "--inventory-environment",
                    "<host-native-inventory-environment>",
                    "--inventory-root",
                    "zig-root:header-smoke-tests",
                    "--inventory-mode",
                    "debug",
                    "--inventory-class",
                    "<host-native-enumeration-class>",
                ],
            ),
            CHECKER.TEST_INVENTORY_RUNNER_RACE_PYTHON_LAUNCH_ID: (
                "test_runner_protocol_and_isolated_object_mutations_fail",
                "subprocess.Popen",
                "test-inventory-runner-fixture-execute",
                [
                    "./test-inventory-digest-vectors",
                    "<fixture-inventory-path>",
                    "--inventory-environment",
                    "<host-native-inventory-environment>",
                    "--inventory-root",
                    "zig-root:header-smoke-tests",
                    "--inventory-mode",
                    "debug",
                    "--inventory-class",
                    "<host-native-enumeration-class>",
                ],
            ),
        }
        launches_by_id = {item["id"]: item for item in inventory_launches}
        for identifier, (
            source_function,
            source_symbol,
            launch_class,
            argv_shape,
        ) in expected_test_inventory_runner_launches.items():
            with self.subTest(test_inventory_runner_launch=identifier):
                launch = launches_by_id[identifier]
                self.assertEqual(
                    source_function, launch["anchor"]["enclosing_function"]
                )
                self.assertEqual(source_symbol, launch["anchor"]["symbol"])
                self.assertEqual(1, launch["anchor"]["ordinal"])
                self.assertEqual("temporary fixture repository", launch["cwd_shape"])
                self.assertEqual(launch_class, launch["launch_class"])
                self.assertEqual(argv_shape, launch["argv_shape"])

        for path in sorted(runner_paths):
            tree = CHECKER.ast.parse(
                (self.root / path).read_text(encoding="utf-8"), filename=path
            )
            functions = {
                node.name
                for node in CHECKER.ast.walk(tree)
                if isinstance(
                    node, (CHECKER.ast.FunctionDef, CHECKER.ast.AsyncFunctionDef)
                )
            }
            self.assertNotIn("git_revision", functions, path)
            self.assertFalse(
                any(
                    isinstance(node, CHECKER.ast.Name) and node.id == "repository_git"
                    for node in CHECKER.ast.walk(tree)
                ),
                path,
            )
            frozen_identity_calls = [
                node
                for node in CHECKER.ast.walk(tree)
                if isinstance(node, CHECKER.ast.Call)
                and isinstance(node.func, CHECKER.ast.Attribute)
                and node.func.attr == "collect_benchmark_identity_from_frozen"
                and isinstance(node.func.value, CHECKER.ast.Name)
                and node.func.value.id == "benchmark_metadata"
            ]
            live_identity_calls = [
                node
                for node in CHECKER.ast.walk(tree)
                if isinstance(node, CHECKER.ast.Call)
                and isinstance(node.func, CHECKER.ast.Attribute)
                and node.func.attr == "collect_benchmark_identity"
                and isinstance(node.func.value, CHECKER.ast.Name)
                and node.func.value.id == "benchmark_metadata"
            ]
            self.assertEqual(1, len(frozen_identity_calls), path)
            self.assertEqual(0, len(live_identity_calls), path)

        shared_boundary_paths = {
            "bench/tools/benchmark_metadata.py",
            "tools/repository_git.py",
        }
        self.assertEqual(
            {
                "python-launch:bench/tools/benchmark_metadata.py:command_output:subprocess.run:1",
                "python-launch:tools/repository_git.py:run:subprocess.run:1",
            },
            {
                item["id"]
                for item in inventory_launches
                if item["anchor"]["file"] in shared_boundary_paths
            },
        )
        metadata_tree = CHECKER.ast.parse(
            (self.root / "bench/tools/benchmark_metadata.py").read_text(
                encoding="utf-8"
            )
        )
        metadata_calls = [
            node.func
            for node in CHECKER.ast.walk(metadata_tree)
            if isinstance(node, CHECKER.ast.Call)
            and isinstance(node.func, CHECKER.ast.Attribute)
        ]
        self.assertTrue(
            any(
                call.attr == "open_repository"
                and isinstance(call.value, CHECKER.ast.Name)
                and call.value.id == "repository_git"
                for call in metadata_calls
            )
        )
        self.assertTrue(
            any(
                call.attr == "observe_identity"
                and isinstance(call.value, CHECKER.ast.Name)
                and call.value.id == "repository"
                for call in metadata_calls
            )
        )

        repository_git_tree = CHECKER.ast.parse(
            (self.root / "tools/repository_git.py").read_text(encoding="utf-8")
        )
        repository_git_process_calls = [
            node
            for node in CHECKER.ast.walk(repository_git_tree)
            if isinstance(node, CHECKER.ast.Call)
            and isinstance(node.func, CHECKER.ast.Attribute)
            and node.func.attr == "run"
            and isinstance(node.func.value, CHECKER.ast.Name)
            and node.func.value.id == "subprocess"
        ]
        self.assertEqual(1, len(repository_git_process_calls))

    def test_unlisted_workflow_run_block_fails_closed(self) -> None:
        path = self.root / ".github/workflows/ci.yml"
        text = path.read_text(encoding="utf-8")
        insertion = (
            "\n      - name: Inventory fixture launch\n        run: python3 -V\n"
        )
        path.write_text(
            text.replace(
                "\n  build-inventory-security:",
                insertion + "\n  build-inventory-security:",
                1,
            ),
            encoding="utf-8",
        )
        self._assert_error_contains(
            "workflow-launch:.github/workflows/ci.yml:source-checks:inventory-fixture-launch"
        )

    def test_unlisted_quoted_workflow_run_key_fails_closed(self) -> None:
        path = self.root / ".github/workflows/ci.yml"
        text = path.read_text(encoding="utf-8")
        insertion = (
            "\n      - name: Quoted run key fixture\n        'run': python3 -V\n"
        )
        path.write_text(
            text.replace(
                "\n  build-inventory-security:",
                insertion + "\n  build-inventory-security:",
                1,
            ),
            encoding="utf-8",
        )
        self._assert_error_contains(
            "workflow-launch:.github/workflows/ci.yml:source-checks:quoted-run-key-fixture"
        )

    def test_unlisted_inline_workflow_run_step_fails_closed(self) -> None:
        path = self.root / ".github/workflows/ci.yml"
        text = path.read_text(encoding="utf-8")
        insertion = "\n      - run: python3 -V\n"
        path.write_text(
            text.replace(
                "\n  build-inventory-security:",
                insertion + "\n  build-inventory-security:",
                1,
            ),
            encoding="utf-8",
        )
        errors = "\n".join(CHECKER.validate(self.root, self.inventory_path))
        self.assertIn(
            "workflow-launch:.github/workflows/ci.yml:source-checks:step-", errors
        )
        self.assertIn("unlisted source occurrence", errors)

    def test_unlisted_flow_mapping_workflow_run_step_fails_closed(self) -> None:
        path = self.root / ".github/workflows/ci.yml"
        text = path.read_text(encoding="utf-8")
        insertion = "\n      - { run: python3 -V }\n"
        path.write_text(
            text.replace(
                "\n  build-inventory-security:",
                insertion + "\n  build-inventory-security:",
                1,
            ),
            encoding="utf-8",
        )
        errors = "\n".join(CHECKER.validate(self.root, self.inventory_path))
        self.assertIn(
            "workflow-launch:.github/workflows/ci.yml:source-checks:step-", errors
        )
        self.assertIn("unlisted source occurrence", errors)

    def test_unsupported_flow_run_shape_fails_closed(self) -> None:
        path = self.root / ".github/workflows/ci.yml"
        text = path.read_text(encoding="utf-8")
        insertion = "\n      - { run: [python3, -V] }\n"
        path.write_text(
            text.replace(
                "\n  build-inventory-security:",
                insertion + "\n  build-inventory-security:",
                1,
            ),
            encoding="utf-8",
        )
        errors = "\n".join(CHECKER.validate(self.root, self.inventory_path))
        self.assertIn("source discovery failed closed", errors)
        self.assertIn("workflow run must be a scalar string", errors)

    def test_quoted_job_key_preserves_launch_inventory(self) -> None:
        path = self.root / ".github/workflows/ci.yml"
        text = path.read_text(encoding="utf-8")
        before = {
            item["id"]: (item["category"], item["anchor"], item["call"])
            for item in CHECKER._discover_workflow_launches(self.root)
        }
        path.write_text(
            text.replace("\n  source-checks:\n", "\n  'source-checks':\n", 1),
            encoding="utf-8",
        )
        after = {
            item["id"]: (item["category"], item["anchor"], item["call"])
            for item in CHECKER._discover_workflow_launches(self.root)
        }
        self.assertEqual(before, after)
        self._assert_error_contains(
            "workflow_source_digests must exactly match every normalized workflow run step"
        )

    def test_arbitrary_workflow_indentation_discovers_run(self) -> None:
        path = self.root / ".github/workflows/indent-fixture.yml"
        path.write_text(
            "name: Indent fixture\n"
            "jobs: # ordinary block comment\n"
            "    fixture-job:\n"
            "        runs-on: ubuntu-latest\n"
            "        steps:\n"
            "            - run: python3 -V\n",
            encoding="utf-8",
        )
        self._assert_error_contains(
            "workflow-launch:.github/workflows/indent-fixture.yml:fixture-job:step-1"
        )

    def test_spaced_sequence_indicator_alignment_discovers_run(self) -> None:
        path = self.root / ".github/workflows/spaced-step-fixture.yml"
        path.write_text(
            "name: Spaced step fixture\n"
            "jobs:\n"
            "  fixture-job:\n"
            "    runs-on: ubuntu-latest\n"
            "    steps:\n"
            "      -   name: Spaced launch\n"
            "          run: python3 -V\n",
            encoding="utf-8",
        )
        self._assert_error_contains(
            "workflow-launch:.github/workflows/spaced-step-fixture.yml:fixture-job:spaced-launch"
        )

    def test_indentless_workflow_step_sequence_discovers_run(self) -> None:
        path = self.root / ".github/workflows/indentless-step-fixture.yml"
        path.write_text(
            "name: Indentless step fixture\n"
            "jobs:\n"
            "  fixture-job:\n"
            "    runs-on: ubuntu-latest\n"
            "    steps:\n"
            "    - run: python3 -V\n",
            encoding="utf-8",
        )
        self._assert_error_contains(
            "workflow-launch:.github/workflows/indentless-step-fixture.yml:fixture-job:step-1"
        )

    def test_workflow_steps_mapping_fails_closed(self) -> None:
        path = self.root / ".github/workflows/steps-mapping-fixture.yml"
        path.write_text(
            "name: Steps mapping fixture\n"
            "jobs:\n"
            "  fixture-job:\n"
            "    runs-on: ubuntu-latest\n"
            "    steps:\n"
            "      run: python3 -V\n",
            encoding="utf-8",
        )
        errors = "\n".join(CHECKER.validate(self.root, self.inventory_path))
        self.assertIn("source discovery failed closed", errors)
        self.assertIn("steps must use a block sequence", errors)

    def test_commented_flow_workflow_step_discovers_run(self) -> None:
        path = self.root / ".github/workflows/commented-flow-fixture.yml"
        path.write_text(
            "name: Commented flow fixture\n"
            "jobs:\n"
            "  fixture-job:\n"
            "    runs-on: ubuntu-latest\n"
            "    steps:\n"
            "      - { name: Flow launch, run: python3 -V } # ordinary comment\n",
            encoding="utf-8",
        )
        self._assert_error_contains(
            "workflow-launch:.github/workflows/commented-flow-fixture.yml:fixture-job:flow-launch"
        )

    def test_commented_sequence_indicator_discovers_run(self) -> None:
        path = self.root / ".github/workflows/commented-sequence-fixture.yml"
        path.write_text(
            "name: Commented sequence fixture\n"
            "jobs:\n"
            "  fixture-job:\n"
            "    runs-on: ubuntu-latest\n"
            "    steps:\n"
            "      - # ordinary comment\n"
            "        run: python3 -V\n",
            encoding="utf-8",
        )
        self._assert_error_contains(
            "workflow-launch:.github/workflows/commented-sequence-fixture.yml:fixture-job:step-1"
        )

    def test_workflow_block_scalar_hash_payload_changes_digest(self) -> None:
        path = self.root / ".github/workflows/block-digest-fixture.yml"
        source = (
            "name: Block digest fixture\n"
            "jobs:\n"
            "  fixture-job:\n"
            "    runs-on: ubuntu-latest\n"
            "    steps:\n"
            "      - name: Shell block\n"
            "        run: |\n"
            "          cat <<'EOF'\n"
            "          # payload A\n"
            "          EOF\n"
        )
        path.write_text(source, encoding="utf-8")
        launch_id = (
            "workflow-launch:.github/workflows/block-digest-fixture.yml:"
            "fixture-job:shell-block"
        )
        before = {
            item["id"]: item["source_digest"]
            for item in CHECKER._discover_workflow_launches(self.root)
        }[launch_id]
        path.write_text(source.replace("# payload A", "# payload B"), encoding="utf-8")
        after = {
            item["id"]: item["source_digest"]
            for item in CHECKER._discover_workflow_launches(self.root)
        }[launch_id]
        self.assertNotEqual(before, after)

    def test_workflow_block_scalar_relative_indent_changes_digest(self) -> None:
        path = self.root / ".github/workflows/block-indent-fixture.yml"
        source = (
            "name: Block indent fixture\n"
            "jobs:\n"
            "  fixture-job:\n"
            "    runs-on: ubuntu-latest\n"
            "    steps:\n"
            "      - name: Shell block\n"
            "        run: |2\n"
            "          cat <<'EOF'\n"
            "          payload\n"
            "          EOF\n"
        )
        path.write_text(source, encoding="utf-8")
        launch_id = (
            "workflow-launch:.github/workflows/block-indent-fixture.yml:"
            "fixture-job:shell-block"
        )
        before = {
            item["id"]: item["source_digest"]
            for item in CHECKER._discover_workflow_launches(self.root)
        }[launch_id]
        path.write_text(
            source.replace("          payload", "            payload"), encoding="utf-8"
        )
        after = {
            item["id"]: item["source_digest"]
            for item in CHECKER._discover_workflow_launches(self.root)
        }[launch_id]
        self.assertNotEqual(before, after)

    def test_generic_workflow_block_scalars_are_consumed_as_content(self) -> None:
        path = self.root / ".github/workflows/generic-block-fixture.yml"
        path.write_text(
            "name: |\n"
            "  X: one\n"
            "  X: two\n"
            "jobs:\n"
            "  fixture:\n"
            "    name: |\n"
            "      X: one\n"
            "      X: two\n"
            "    steps:\n"
            "      - run: echo visible\n",
            encoding="utf-8",
        )
        self._assert_error_contains(
            "workflow-launch:.github/workflows/generic-block-fixture.yml:fixture:step-1"
        )

    def test_invalid_workflow_block_scalar_indicators_fail_closed(self) -> None:
        path = self.root / ".github/workflows/invalid-block-fixture.yml"
        template = (
            "name: Invalid block fixture\n"
            "jobs:\n"
            "  fixture-job:\n"
            "    runs-on: ubuntu-latest\n"
            "    steps:\n"
            "      - run: {indicator}\n"
            "          payload\n"
        )
        for indicator in ("|0", "|10", "|++"):
            with self.subTest(indicator=indicator):
                path.write_text(template.format(indicator=indicator), encoding="utf-8")
                with self.assertRaisesRegex(
                    CHECKER.InventoryError,
                    "invalid workflow block scalar indicator",
                ):
                    CHECKER._discover_workflow_launches(self.root)

    def test_invalid_non_run_block_scalar_indicators_fail_closed(self) -> None:
        path = self.root / ".github/workflows/invalid-non-run-block-fixture.yml"
        fixtures = (
            "name: |0\njobs:\n  fixture:\n    steps:\n      - run: echo hidden\n",
            "jobs:\n  fixture:\n    steps:\n      - name: |0\n        run: echo hidden\n",
            "jobs:\n  fixture:\n    steps:\n      - shell: >10\n        run: echo hidden\n",
        )
        for source in fixtures:
            with self.subTest(source=source):
                path.write_text(source, encoding="utf-8")
                with self.assertRaisesRegex(
                    CHECKER.InventoryError,
                    "invalid workflow block scalar indicator",
                ):
                    CHECKER._discover_workflow_launches(self.root)

    def test_insufficient_explicit_block_scalar_indent_fails_closed(self) -> None:
        path = self.root / ".github/workflows/short-block-indent-fixture.yml"
        path.write_text(
            "name: Short block indent fixture\n"
            "jobs:\n"
            "  fixture-job:\n"
            "    runs-on: ubuntu-latest\n"
            "    steps:\n"
            "      - run: |2\n"
            "         payload\n",
            encoding="utf-8",
        )
        with self.assertRaisesRegex(
            CHECKER.InventoryError,
            "inconsistent workflow block scalar indentation",
        ):
            CHECKER._discover_workflow_launches(self.root)

    def test_unsupported_explicit_keys_tags_and_merge_fail_closed(self) -> None:
        path = self.root / ".github/workflows/unsupported-yaml-fixture.yml"
        fixtures = {
            "explicit": (
                "name: Explicit key fixture\n"
                "? jobs\n"
                ":\n"
                "  fixture-job:\n"
                "    steps:\n"
                "      - run: python3 -V\n"
            ),
            "tag": (
                "name: Tagged key fixture\n"
                "!!str jobs:\n"
                "  fixture-job:\n"
                "    steps:\n"
                "      - run: python3 -V\n"
            ),
            "merge": (
                "name: Merge fixture\njobs:\n  fixture-job:\n    <<: *job-template\n"
            ),
        }
        for label, source in fixtures.items():
            with self.subTest(label=label):
                path.write_text(source, encoding="utf-8")
                with self.assertRaises(CHECKER.InventoryError):
                    CHECKER._discover_workflow_launches(self.root)

    def test_workflow_anchor_alias_and_bom_fail_closed(self) -> None:
        path = self.root / ".github/workflows/unsupported-key-fixture.yml"
        fixtures = {
            "anchor": (
                "&jobs-key jobs:\n  fixture-job:\n    steps:\n      - run: python3 -V\n"
            ),
            "alias": (
                "key: &jobs-key jobs\n"
                "*jobs-key:\n"
                "  fixture-job:\n"
                "    steps:\n"
                "      - run: python3 -V\n"
            ),
            "bom": (
                "\ufeffjobs:\n  fixture-job:\n    steps:\n      - run: python3 -V\n"
            ),
        }
        for label, source in fixtures.items():
            with self.subTest(label=label):
                path.write_text(source, encoding="utf-8")
                with self.assertRaises(CHECKER.InventoryError):
                    CHECKER._discover_workflow_launches(self.root)

    def test_invalid_utf8_workflow_fails_closed(self) -> None:
        path = self.root / ".github/workflows/invalid-utf8-fixture.yml"
        path.write_bytes(b"jobs:\n  fixture:\n    steps:\n      - run: \xff\n")
        with self.assertRaisesRegex(
            CHECKER.InventoryError, "workflow is not valid UTF-8"
        ):
            CHECKER._discover_workflow_launches(self.root)

    def test_separated_duplicate_jobs_mapping_fails_closed(self) -> None:
        path = self.root / ".github/workflows/duplicate-jobs-fixture.yml"
        path.write_text(
            "jobs:\n"
            "  first-job:\n"
            "    steps:\n"
            "      - run: printf first\n"
            "name: separator\n"
            "jobs:\n"
            "  second-job:\n"
            "    steps:\n"
            "      - run: printf second\n",
            encoding="utf-8",
        )
        with self.assertRaisesRegex(CHECKER.InventoryError, "duplicate jobs mapping"):
            CHECKER._discover_workflow_launches(self.root)

    def test_root_flow_workflow_mapping_fails_closed(self) -> None:
        path = self.root / ".github/workflows/root-flow-fixture.yml"
        path.write_text(
            "{ name: fixture, jobs: { fixture-job: { steps: [ { run: python3 -V } ] } } }\n",
            encoding="utf-8",
        )
        with self.assertRaisesRegex(
            CHECKER.InventoryError,
            "unsupported workflow mapping syntax",
        ):
            CHECKER._discover_workflow_launches(self.root)

    def test_document_start_with_inline_root_fails_closed(self) -> None:
        path = self.root / ".github/workflows/document-root-fixture.yml"
        for source in (
            "--- {jobs: {j: {steps: [{run: echo hidden}]}}}\n",
            "--- &root {jobs: {j: {steps: [{run: echo hidden}]}}}\n",
        ):
            with self.subTest(source=source):
                path.write_text(source, encoding="utf-8")
                with self.assertRaisesRegex(
                    CHECKER.InventoryError, "document markers are unsupported"
                ):
                    CHECKER._discover_workflow_launches(self.root)

    def test_duplicate_workflow_job_key_fails_closed(self) -> None:
        path = self.root / ".github/workflows/duplicate-job-fixture.yml"
        path.write_text(
            "jobs:\n"
            "  repeated:\n"
            "    steps:\n"
            "      - name: ghost\n"
            "        run: echo ghost\n"
            "  other:\n"
            "    steps:\n"
            "      - run: echo other\n"
            "  repeated:\n"
            "    steps:\n"
            "      - name: live\n"
            "        run: echo live\n",
            encoding="utf-8",
        )
        with self.assertRaisesRegex(
            CHECKER.InventoryError, "duplicate workflow job key"
        ):
            CHECKER._discover_workflow_launches(self.root)

    def test_invalid_workflow_job_identifiers_fail_closed(self) -> None:
        path = self.root / ".github/workflows/invalid-job-id-fixture.yml"
        for job_id in ('""', '"bad job"', "123", "bad.job"):
            with self.subTest(job_id=job_id):
                path.write_text(
                    f"jobs:\n  {job_id}:\n    steps:\n      - run: echo hidden\n",
                    encoding="utf-8",
                )
                with self.assertRaisesRegex(
                    CHECKER.InventoryError,
                    "(?:invalid GitHub Actions job identifier|mapping keys must be scalar strings)",
                ):
                    CHECKER._discover_workflow_launches(self.root)

    def test_duplicate_workflow_launch_identity_fails_closed(self) -> None:
        path = self.root / ".github/workflows/duplicate-launch-fixture.yml"
        path.write_text(
            "jobs:\n"
            "  fixture:\n"
            "    steps:\n"
            "      - name: repeated\n"
            "        run: echo first\n"
            "      - name: repeated\n"
            "        run: echo second\n",
            encoding="utf-8",
        )
        with self.assertRaisesRegex(
            CHECKER.InventoryError, "duplicate workflow launch identity"
        ):
            CHECKER._discover_workflow_launches(self.root)

    def test_block_scalar_workflow_step_name_fails_closed(self) -> None:
        path = self.root / ".github/workflows/block-name-fixture.yml"
        path.write_text(
            "jobs:\n"
            "  fixture:\n"
            "    steps:\n"
            "      - name: |\n"
            "          hidden name\n"
            "        run: echo visible\n",
            encoding="utf-8",
        )
        with self.assertRaisesRegex(
            CHECKER.InventoryError, "block scalar workflow step names are unsupported"
        ):
            CHECKER._discover_workflow_launches(self.root)

    def test_workflow_top_level_scalar_fails_closed(self) -> None:
        path = self.root / ".github/workflows/root-scalar-fixture.yml"
        path.write_text(
            "garbage\njobs:\n  fixture:\n    steps:\n      - run: echo hidden\n",
            encoding="utf-8",
        )
        with self.assertRaisesRegex(
            CHECKER.InventoryError, "workflow top level must use a mapping"
        ):
            CHECKER._discover_workflow_launches(self.root)

    def test_duplicate_workflow_step_property_fails_closed(self) -> None:
        path = self.root / ".github/workflows/duplicate-step-key-fixture.yml"
        path.write_text(
            "jobs:\n"
            "  fixture:\n"
            "    steps:\n"
            "      - shell: bash\n"
            "        shell: sh\n"
            "        run: echo hidden\n",
            encoding="utf-8",
        )
        with self.assertRaisesRegex(
            CHECKER.InventoryError, "duplicate workflow step key 'shell'"
        ):
            CHECKER._discover_workflow_launches(self.root)

    def test_escaped_workflow_step_name_fails_closed(self) -> None:
        path = self.root / ".github/workflows/escaped-name-fixture.yml"
        for name in ('"Te\\u0073t"', '"bad\\q"'):
            with self.subTest(name=name):
                path.write_text(
                    "jobs:\n"
                    "  fixture:\n"
                    "    steps:\n"
                    f"      - name: {name}\n"
                    "        run: echo visible\n",
                    encoding="utf-8",
                )
                with self.assertRaisesRegex(
                    CHECKER.InventoryError,
                    "escaped double-quoted workflow .* is unsupported",
                ):
                    CHECKER._discover_workflow_launches(self.root)

    def test_unterminated_top_level_workflow_scalar_fails_closed(self) -> None:
        path = self.root / ".github/workflows/unterminated-root-fixture.yml"
        path.write_text(
            'name: "unterminated\n'
            "jobs:\n"
            "  fixture:\n"
            "    steps:\n"
            "      - run: echo hidden\n",
            encoding="utf-8",
        )
        with self.assertRaisesRegex(
            CHECKER.InventoryError, "multiline quoted workflow mapping value"
        ):
            CHECKER._discover_workflow_launches(self.root)

    def test_invalid_plain_top_level_workflow_scalar_fails_closed(self) -> None:
        path = self.root / ".github/workflows/invalid-root-value-fixture.yml"
        for value in ("@bad", "bad: value"):
            with self.subTest(value=value):
                path.write_text(
                    f"name: {value}\n"
                    "jobs:\n"
                    "  fixture:\n"
                    "    steps:\n"
                    "      - run: echo hidden\n",
                    encoding="utf-8",
                )
                with self.assertRaisesRegex(
                    CHECKER.InventoryError,
                    "invalid plain workflow mapping value scalar",
                ):
                    CHECKER._discover_workflow_launches(self.root)

    def test_malformed_quoted_workflow_scalars_fail_closed(self) -> None:
        path = self.root / ".github/workflows/multiple-quoted-fixture.yml"
        fixtures = (
            'name: "bad" "more"\njobs:\n  fixture:\n    steps:\n      - run: echo hidden\n',
            'jobs:\n  fixture:\n    steps:\n      - name: "bad" "more"\n        run: echo hidden\n',
            "jobs:\n  fixture:\n    steps:\n      - name: 'bad' 'more'\n        run: echo hidden\n",
        )
        for source in fixtures:
            with self.subTest(source=source):
                path.write_text(source, encoding="utf-8")
                with self.assertRaisesRegex(
                    CHECKER.InventoryError, "multiple quoted workflow"
                ):
                    CHECKER._discover_workflow_launches(self.root)

    def test_multiline_plain_workflow_step_names_fail_closed(self) -> None:
        path = self.root / ".github/workflows/multiline-name-fixture.yml"
        continuations = (
            "          world\n",
            "\n          world\n",
        )
        for continuation in continuations:
            with self.subTest(continuation=continuation):
                path.write_text(
                    "jobs:\n"
                    "  fixture:\n"
                    "    steps:\n"
                    "      - name: hello\n"
                    + continuation
                    + "        run: echo visible\n",
                    encoding="utf-8",
                )
                with self.assertRaisesRegex(
                    CHECKER.InventoryError,
                    "multiline workflow step names are unsupported",
                ):
                    CHECKER._discover_workflow_launches(self.root)

    def test_malformed_and_duplicate_nested_flow_values_fail_closed(self) -> None:
        path = self.root / ".github/workflows/invalid-flow-value-fixture.yml"
        fixtures = (
            "name: [unterminated\njobs:\n  fixture:\n    steps:\n      - run: echo hidden\n",
            "jobs:\n  fixture:\n    steps:\n      - env: { X: one, X: two }\n        run: echo $X\n",
            "jobs:\n  fixture:\n    steps:\n      - env: { X: one\n        run: echo $X\n",
        )
        for source in fixtures:
            with self.subTest(source=source):
                path.write_text(source, encoding="utf-8")
                with self.assertRaises(CHECKER.InventoryError):
                    CHECKER._discover_workflow_launches(self.root)

    def test_trailing_comma_in_workflow_flow_collections_is_supported(self) -> None:
        path = self.root / ".github/workflows/trailing-flow-comma-fixture.yml"
        path.write_text(
            "on: [push,]\n"
            "jobs:\n"
            "  fixture:\n"
            "    steps:\n"
            "      - { env: { X: one, }, run: echo $X, }\n",
            encoding="utf-8",
        )
        self._assert_error_contains(
            "workflow-launch:.github/workflows/trailing-flow-comma-fixture.yml:fixture:step-1"
        )

    def test_compact_sequence_block_scalar_is_consumed_as_content(self) -> None:
        path = self.root / ".github/workflows/compact-block-fixture.yml"
        path.write_text(
            "jobs:\n"
            "  fixture:\n"
            "    strategy:\n"
            "      matrix:\n"
            "        include:\n"
            "          - command: |\n"
            "              X: one\n"
            "              X: two\n"
            "    steps:\n"
            "      - run: ${{ matrix.command }}\n",
            encoding="utf-8",
        )
        self._assert_error_contains(
            "workflow-launch:.github/workflows/compact-block-fixture.yml:fixture:step-1"
        )

    def test_compact_sequence_mapping_values_are_validated(self) -> None:
        path = self.root / ".github/workflows/compact-value-fixture.yml"
        entries = (
            "{ X: one, X: two }",
            "&cmd echo",
            '"bad" "more"',
        )
        for value in entries:
            with self.subTest(value=value):
                path.write_text(
                    "items:\n"
                    f"  - command: {value}\n"
                    "jobs:\n"
                    "  fixture:\n"
                    "    steps:\n"
                    "      - run: echo hidden\n",
                    encoding="utf-8",
                )
                with self.assertRaises(CHECKER.InventoryError):
                    CHECKER._discover_workflow_launches(self.root)

    def test_excessive_workflow_flow_nesting_fails_closed(self) -> None:
        path = self.root / ".github/workflows/deep-flow-fixture.yml"
        path.write_text(
            "name: " + "[" * 80 + "x" + "]" * 80 + "\n"
            "jobs:\n"
            "  fixture:\n"
            "    steps:\n"
            "      - run: echo hidden\n",
            encoding="utf-8",
        )
        with self.assertRaisesRegex(
            CHECKER.InventoryError, "flow nesting exceeds the supported limit"
        ):
            CHECKER._discover_workflow_launches(self.root)

    def test_workflow_symlink_source_fails_closed(self) -> None:
        external = self.root / "external-workflow.yml"
        external.write_text(
            "jobs:\n  fixture:\n    steps:\n      - run: echo hidden\n",
            encoding="utf-8",
        )
        linked = self.root / ".github/workflows/linked.yml"
        try:
            linked.symlink_to(external)
        except OSError as exc:
            self.skipTest(f"symlinks are unavailable: {exc}")
        with self.assertRaisesRegex(CHECKER.InventoryError, "must not use symlinks"):
            CHECKER._discover_workflow_launches(self.root)

    def test_nested_duplicate_workflow_mapping_key_fails_closed(self) -> None:
        path = self.root / ".github/workflows/duplicate-env-fixture.yml"
        path.write_text(
            "jobs:\n"
            "  fixture:\n"
            "    steps:\n"
            "      - env:\n"
            "          X: one\n"
            "          X: two\n"
            "        run: echo hidden\n",
            encoding="utf-8",
        )
        with self.assertRaisesRegex(
            CHECKER.InventoryError, "duplicate workflow mapping key 'X'"
        ):
            CHECKER._discover_workflow_launches(self.root)

    def test_matrix_include_duplicate_keys_fail_closed(self) -> None:
        path = self.root / ".github/workflows/duplicate-matrix-fixture.yml"
        fixtures = (
            "          - command: echo one\n            command: echo two\n",
            "          - { command: echo one, command: echo two }\n",
        )
        for include in fixtures:
            with self.subTest(include=include):
                path.write_text(
                    "jobs:\n"
                    "  fixture:\n"
                    "    strategy:\n"
                    "      matrix:\n"
                    "        include:\n" + include + "    steps:\n"
                    "      - run: ${{ matrix.command }}\n",
                    encoding="utf-8",
                )
                with self.assertRaisesRegex(
                    CHECKER.InventoryError, "duplicate workflow .*key 'command'"
                ):
                    CHECKER._discover_workflow_launches(self.root)

    def test_workflow_anchor_and_alias_values_fail_closed(self) -> None:
        path = self.root / ".github/workflows/value-alias-fixture.yml"
        path.write_text(
            "row: &row\n"
            "  command: echo one\n"
            "jobs:\n"
            "  fixture:\n"
            "    strategy:\n"
            "      matrix:\n"
            "        include:\n"
            "          - *row\n"
            "    steps:\n"
            "      - run: ${{ matrix.command }}\n",
            encoding="utf-8",
        )
        with self.assertRaisesRegex(
            CHECKER.InventoryError, "anchors, aliases, and tags are unsupported"
        ):
            CHECKER._discover_workflow_launches(self.root)

    def test_empty_or_null_workflow_steps_fail_closed(self) -> None:
        path = self.root / ".github/workflows/empty-steps-fixture.yml"
        fixtures = (
            "jobs:\n  fixture:\n    steps:\n",
            "jobs:\n  fixture:\n    steps: # comment only\n    timeout-minutes: 1\n",
        )
        for source in fixtures:
            with self.subTest(source=source):
                path.write_text(source, encoding="utf-8")
                with self.assertRaisesRegex(
                    CHECKER.InventoryError, "steps must use a non-empty block sequence"
                ):
                    CHECKER._discover_workflow_launches(self.root)

    def test_flow_mapping_parentheses_do_not_hide_run(self) -> None:
        path = self.root / ".github/workflows/flow-parentheses-fixture.yml"
        path.write_text(
            "jobs:\n"
            "  fixture:\n"
            "    steps:\n"
            "      - { name: wrapper(foo, run: echo visible) }\n",
            encoding="utf-8",
        )
        launches = CHECKER._discover_workflow_launches(self.root)
        self.assertIn(
            "workflow-launch:.github/workflows/flow-parentheses-fixture.yml:fixture:wrapper-foo",
            {item["id"] for item in launches},
        )

    def test_workflow_expression_inputs_change_launch_digest(self) -> None:
        path = self.root / ".github/workflows/matrix-digest-fixture.yml"
        source = (
            "jobs:\n"
            "  fixture:\n"
            "    strategy:\n"
            "      matrix:\n"
            "        include:\n"
            "          - command: echo A\n"
            "    steps:\n"
            "      - run: ${{ matrix.command }}\n"
        )
        launch_id = (
            "workflow-launch:.github/workflows/matrix-digest-fixture.yml:fixture:step-1"
        )
        path.write_text(source, encoding="utf-8")
        before = {
            item["id"]: item["source_digest"]
            for item in CHECKER._discover_workflow_launches(self.root)
        }[launch_id]
        path.write_text(source.replace("echo A", "echo B"), encoding="utf-8")
        after = {
            item["id"]: item["source_digest"]
            for item in CHECKER._discover_workflow_launches(self.root)
        }[launch_id]
        self.assertNotEqual(before, after)

        ci_source = (self.root / ".github/workflows/ci.yml").read_text(encoding="utf-8")
        release_source = (self.root / ".github/workflows/release.yml").read_text(
            encoding="utf-8"
        )
        discovered_workflow_ids = {
            item["id"] for item in CHECKER._discover_workflow_launches(self.root)
        }
        self.assertEqual(62, len(discovered_workflow_ids - {launch_id}))
        self.assertEqual(
            {
                "workflow-launch:.github/workflows/ci.yml:target-tests:run-checked-c-and-c-consumers",
                "workflow-launch:.github/workflows/ci.yml:target-tests:run-checked-fortran-consumer",
            },
            {
                identifier for identifier in discovered_workflow_ids
                if ":run-checked-" in identifier
            },
        )
        expected_binding_ids = {
            "workflow-launch:.github/workflows/ci.yml:source-checks:bind-trusted-posix-toolchain-paths",
            "workflow-launch:.github/workflows/ci.yml:target-tests:bind-trusted-windows-toolchain-paths",
            "workflow-launch:.github/workflows/ci.yml:target-tests:bind-trusted-posix-toolchain-paths-and-require-fortran-for-full-smoke",
            "workflow-launch:.github/workflows/ci.yml:capability-builds:bind-trusted-posix-toolchain-paths",
            "workflow-launch:.github/workflows/release.yml:build-inventory-security:bind-trusted-posix-toolchain-paths",
            "workflow-launch:.github/workflows/release.yml:test-inventory-security:bind-trusted-posix-toolchain-paths",
            "workflow-launch:.github/workflows/release.yml:artifacts:bind-trusted-posix-toolchain-paths-and-require-fortran-on-linux",
        }
        self.assertEqual(
            expected_binding_ids,
            {
                identifier
                for identifier in discovered_workflow_ids
                if ":bind-trusted-" in identifier
            },
        )
        action_lines = [
            line
            for workflow_source in (ci_source, release_source)
            for line in workflow_source.splitlines()
            if line.lstrip().startswith("- uses:")
        ]
        self.assertEqual(23, len(action_lines))
        for line in action_lines:
            with self.subTest(action_pin=line):
                self.assertRegex(
                    line,
                    (
                        r"^\s+- uses: [A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+@"
                        r"[0-9a-f]{40} # v[0-9]+\.[0-9]+\.[0-9]+$"
                    ),
                )
        output_ref = "${{ steps.publication_workspace.outputs.path }}"

        def job_block(source: str, job: str) -> str:
            marker = f"  {job}:\n"
            start = source.index(marker)
            end = len(source)
            for line in source[start + len(marker) :].splitlines(keepends=True):
                if line.startswith("  ") and not line.startswith("    "):
                    end = source.index(line, start + len(marker))
                    break
            return source[start:end]

        def assert_job_timeout(source: str, job: str, expected: int) -> None:
            lines = job_block(source, job).splitlines()
            self.assertEqual(
                [f"    timeout-minutes: {expected}"],
                [line for line in lines if line.startswith("    timeout-minutes: ")],
            )

        assert_job_timeout(ci_source, "source-checks", 120)
        assert_job_timeout(ci_source, "build-inventory-security", 30)
        assert_job_timeout(ci_source, "test-inventory-security", 30)
        assert_job_timeout(ci_source, "target-tests", 300)
        assert_job_timeout(ci_source, "capability-builds", 45)
        assert_job_timeout(ci_source, "ci-gate", 5)
        assert_job_timeout(release_source, "build-inventory-security", 240)
        assert_job_timeout(release_source, "test-inventory-security", 120)
        assert_job_timeout(release_source, "artifacts", 180)

        def assert_ci_build_security_matrix(source: str) -> None:
            block = job_block(source, "build-inventory-security")
            self.assertIn(
                "    name: Build inventory security / ${{ matrix.os }}\n", block
            )
            self.assertIn("    runs-on: ${{ matrix.os }}\n", block)
            self.assertIn(
                "    strategy:\n"
                "      fail-fast: false\n"
                "      matrix:\n"
                "        os: [macos-15, ubuntu-24.04]\n",
                block,
            )

        assert_ci_build_security_matrix(ci_source)

        def mutate_job(source: str, job: str, before: str, after: str) -> str:
            original = job_block(source, job)
            mutated = original.replace(before, after, 1)
            self.assertNotEqual(original, mutated)
            return source.replace(original, mutated, 1)

        for source, job, expected, legacy in (
            (ci_source, "build-inventory-security", 30, 15),
            (ci_source, "test-inventory-security", 30, 15),
            (ci_source, "target-tests", 300, 90),
            (ci_source, "capability-builds", 45, 60),
            (ci_source, "ci-gate", 5, 1),
            (release_source, "build-inventory-security", 240, 30),
            (release_source, "test-inventory-security", 120, 90),
        ):
            with self.subTest(timeout_mutant=(job, legacy)):
                mutant = mutate_job(
                    source,
                    job,
                    f"    timeout-minutes: {expected}\n",
                    f"    timeout-minutes: {legacy}\n",
                )
                with self.assertRaises(AssertionError):
                    assert_job_timeout(mutant, job, expected)
        for before, after in (
            ("Build inventory security / ${{ matrix.os }}", "Build inventory security"),
            ("runs-on: ${{ matrix.os }}", "runs-on: ubuntu-24.04"),
            ("fail-fast: false", "fail-fast: true"),
            ("os: [macos-15, ubuntu-24.04]", "os: [ubuntu-24.04]"),
        ):
            with self.subTest(ci_build_security_matrix_mutant=before):
                mutant = mutate_job(
                    ci_source, "build-inventory-security", before, after
                )
                with self.assertRaises(AssertionError):
                    assert_ci_build_security_matrix(mutant)

        def named_step(source: str, name: str) -> str:
            marker = f"      - name: {name}\n"
            start = source.index(marker)
            lines = source[start:].splitlines()
            step_lines = [lines[0]]
            for line in lines[1:]:
                if line.startswith("      - ") or (
                    line.startswith("  ") and not line.startswith("   ")
                ):
                    break
                step_lines.append(line)
            return "\n".join(step_lines).rstrip()

        trusted_posix_shell = (
            "/usr/bin/env BASH_ENV=/dev/null ENV=/dev/null "
            "GITHUB_ENV=/dev/null GITHUB_PATH=/dev/null "
            "/bin/bash --noprofile --norc -p -eo pipefail {0}"
        )
        trusted_posix_env = (
            "        env:\n"
            "          PATH: ${{ steps.trusted-posix-tools.outputs.path }}\n"
            "          BASH_ENV: /dev/null\n"
            "          ENV: /dev/null\n"
        )
        trusted_windows_shell = r'''C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe -NoLogo -NoProfile -NonInteractive -Command "$ErrorActionPreference = 'Stop'; $env:GITHUB_ENV = 'NUL'; $env:GITHUB_PATH = 'NUL'; & ([scriptblock]::Create([IO.File]::ReadAllText('{0}'))); if (Test-Path -LiteralPath variable:\LASTEXITCODE) {{ exit $LASTEXITCODE }}"'''
        trusted_windows_env = (
            "        env:\n"
            "          TRUSTED_PATH: ${{ steps.trusted-windows-tools.outputs.path }}\n"
            "          TRUSTED_ZIG: ${{ steps.trusted-windows-tools.outputs.zig }}\n"
            "          TRUSTED_PYTHON: ${{ steps.trusted-windows-tools.outputs.python }}\n"
        )
        terminal_singleton_shell = (
            "/usr/bin/env BASH_ENV=/dev/null ENV=/dev/null "
            "GITHUB_ENV=/dev/null GITHUB_PATH=/dev/null "
            "/bin/bash --noprofile --norc -p -eo pipefail {0}"
        )
        terminal_singleton_env = (
            "        env:\n          BASH_ENV: /dev/null\n          ENV: /dev/null\n"
        )
        ordinary_posix_binding_name = "Bind trusted POSIX toolchain paths"
        windows_binding_name = "Bind trusted Windows toolchain paths"
        ci_posix_binding_name = (
            "Bind trusted POSIX toolchain paths and require Fortran for full smoke"
        )
        release_posix_binding_name = (
            "Bind trusted POSIX toolchain paths and require Fortran on Linux"
        )

        def assert_posix_binding(
            step: str, *, name: str, condition: str | None, require_fortran: str
        ) -> None:
            header = f"      - name: {name}\n        id: trusted-posix-tools\n"
            if condition is not None:
                header += f"        if: {condition}\n"
            header += (
                "        shell: /bin/bash --noprofile --norc -p -eo pipefail {0}\n"
                "        env:\n"
                "          BASH_ENV: /dev/null\n"
                "          ENV: /dev/null\n"
                "        run: |\n"
                "          set -euo pipefail\n"
            )
            if require_fortran:
                header = header.replace(
                    "          ENV: /dev/null\n        run: |\n",
                    "          ENV: /dev/null\n"
                    f"          REQUIRE_FORTRAN: {require_fortran}\n"
                    "        run: |\n",
                )
            self.assertTrue(step.startswith(header))
            for fragment in (
                '[[ -z "$value" || "$value" == *$\'\\r\'* || "$value" == *$\'\\n\'* ]]',
                '[[ "$workspace" != /* || ! -d "$workspace" ]]',
                'candidate="$(PATH="$trusted_path" command -v "$name")"',
                '[[ "$candidate" != /* || ! -f "$candidate" || ! -x "$candidate" ]]',
                '"$workspace"|"$workspace"/*)',
                'zig_version="$("$trusted_zig" version)"',
                '[[ "$zig_version" != 0.17.0 ]]',
                '"$trusted_python" -I -B -c',
                'printf \'zig=%s\\n\' "$trusted_zig" >> "$GITHUB_OUTPUT"',
                'printf \'python=%s\\n\' "$trusted_python" >> "$GITHUB_OUTPUT"',
                'printf \'path=%s\\n\' "$trusted_path" >> "$GITHUB_OUTPUT"',
            ):
                self.assertIn(fragment, step)
            if require_fortran:
                for fragment in (
                    'trusted_gfortran="$(resolve_trusted_tool gfortran)"',
                    '"$trusted_gfortran" --version >/dev/null',
                    'printf \'gfortran=%s\\n\' "$trusted_gfortran" >> "$GITHUB_OUTPUT"',
                ):
                    self.assertIn(fragment, step)
            else:
                self.assertIn("printf 'gfortran=\\n' >> \"$GITHUB_OUTPUT\"", step)
            self.assertEqual(4, step.count('>> "$GITHUB_OUTPUT"'))
            self.assertNotIn("GITHUB_ENV", step)
            self.assertNotIn("GITHUB_PATH", step)
            self.assertNotIn("trusted_shell", step)
            self.assertNotIn("RUNNER_TEMP", step)
            self.assertNotIn("apt", step)
            self.assertNotIn("sudo", step)
            self.assertNotIn("continue-on-error", step)

        def assert_trusted_tool_bindings(ci: str, release: str) -> None:
            ci_jobs = {
                job: job_block(ci, job)
                for job in ("source-checks", "target-tests", "capability-builds")
            }
            release_jobs = {
                job: job_block(release, job)
                for job in (
                    "build-inventory-security",
                    "test-inventory-security",
                    "artifacts",
                )
            }
            target_gate = ci_jobs["target-tests"]
            release_gate = release_jobs["artifacts"]
            windows_binding = named_step(target_gate, windows_binding_name)

            self.assertTrue(
                windows_binding.startswith(
                    f"      - name: {windows_binding_name}\n"
                    "        id: trusted-windows-tools\n"
                    "        if: runner.os == 'Windows'\n"
                    f"        shell: {trusted_windows_shell}\n"
                    "        run: |\n"
                    "          Set-StrictMode -Version Latest\n"
                    "          $ErrorActionPreference = 'Stop'\n"
                )
            )
            for fragment in (
                '$value.Contains("`r") -or $value.Contains("`n")',
                "Get-Command -Name $name -CommandType Application -All -ErrorAction Stop",
                "function Test-DriveRootedAbsolute([string] $path)",
                "[IO.Path]::IsPathRooted($path) -and $path -match '^[A-Za-z]:[\\\\/]'",
                "Test-DriveRootedAbsolute $candidate",
                "$item -isnot [IO.FileInfo]",
                "[IO.File]::Exists($candidate)",
                "$candidate.Equals($workspace, [StringComparison]::OrdinalIgnoreCase)",
                "$candidate.StartsWith($workspacePrefix, [StringComparison]::OrdinalIgnoreCase)",
                "$zigVersion[0] -cne '0.17.0'",
                "& $trustedPython -I -B --version 2>&1",
                '"zig=$trustedZig"',
                '"python=$trustedPython"',
                '"path=$trustedPath"',
            ):
                self.assertIn(fragment, windows_binding)
            self.assertNotIn("& $trustedPython -I -B -c", windows_binding)
            self.assertIn(
                "$utf8NoBom = New-Object Text.UTF8Encoding($false)", windows_binding
            )
            self.assertIn(
                "[IO.File]::AppendAllLines($env:GITHUB_OUTPUT", windows_binding
            )
            self.assertEqual(1, windows_binding.count('"zig=$trustedZig"'))
            self.assertEqual(1, windows_binding.count('"python=$trustedPython"'))
            self.assertEqual(1, windows_binding.count('"path=$trustedPath"'))
            self.assertNotIn("gfortran=", windows_binding)
            self.assertEqual(1, windows_binding.count("$env:GITHUB_ENV = 'NUL'"))
            self.assertEqual(1, windows_binding.count("$env:GITHUB_PATH = 'NUL'"))
            self.assertLess(
                windows_binding.index("$env:GITHUB_ENV = 'NUL'"),
                windows_binding.index("[IO.File]::ReadAllText('{0}')"),
            )
            self.assertLess(
                windows_binding.index("$env:GITHUB_PATH = 'NUL'"),
                windows_binding.index("[IO.File]::ReadAllText('{0}')"),
            )

            posix_binders = (
                (ci_jobs["source-checks"], ordinary_posix_binding_name, None, ""),
                (
                    target_gate,
                    ci_posix_binding_name,
                    "runner.os != 'Windows'",
                    "${{ matrix.full_smoke && runner.os == 'Linux' }}",
                ),
                (ci_jobs["capability-builds"], ordinary_posix_binding_name, None, ""),
                (
                    release_jobs["build-inventory-security"],
                    ordinary_posix_binding_name,
                    None,
                    "",
                ),
                (
                    release_jobs["test-inventory-security"],
                    ordinary_posix_binding_name,
                    None,
                    "",
                ),
                (
                    release_gate,
                    release_posix_binding_name,
                    None,
                    "${{ runner.os == 'Linux' }}",
                ),
            )
            for gate, name, condition, require_fortran in posix_binders:
                self.assertEqual(1, gate.count(f"      - name: {name}\n"))
                assert_posix_binding(
                    named_step(gate, name),
                    name=name,
                    condition=condition,
                    require_fortran=require_fortran,
                )
            self.assertEqual(
                1, target_gate.count(f"      - name: {windows_binding_name}\n")
            )

            for gate in (*ci_jobs.values(), *release_jobs.values()):
                self.assertNotIn("continue-on-error", gate)
                self.assertNotIn("apt-get", gate)
                self.assertNotIn("sudo", gate)

            protected_posix_jobs = (
                (
                    ci_jobs["source-checks"],
                    ordinary_posix_binding_name,
                    (
                        "Check formatting",
                        "Check Python tools compile",
                        "Check build inventory",
                        "Check test inventory structure",
                        "Check package paths",
                        "Regenerate compatibility headers and kernel coverage",
                        "Check generated files are up to date",
                    ),
                ),
                (
                    target_gate,
                    ci_posix_binding_name,
                    (
                        "Run host tool smoke once",
                        "Test Debug target",
                        "Test ReleaseSafe target",
                        "Test ReleaseFast target",
                        "Link test inventory for Debug target (POSIX structure-gated)",
                        "Link test inventory for ReleaseSafe target (POSIX structure-gated)",
                        "Link test inventory for ReleaseFast target (POSIX structure-gated)",
                        "Build install artifacts",
                        "Run checked C and C++ consumers",
                        "Run checked Fortran consumer",
                        "Check ABI symbols",
                        "Check install metadata",
                        "Run Zig example",
                        "Run CBLAS example",
                        "Run Fortran example",
                        "Run benchmark executable smoke tests",
                        "Run probe smoke tests",
                        "Run Python benchmark tool smoke tests",
                    ),
                ),
                (
                    ci_jobs["capability-builds"],
                    ordinary_posix_binding_name,
                    (
                        "Build target capability",
                        "Compile enabled Level 2 width production-artifact probe",
                    ),
                ),
                (
                    release_jobs["build-inventory-security"],
                    ordinary_posix_binding_name,
                    (
                        "Require current-only build inventory policy",
                        "Require current-only test inventory policy",
                        "Run build inventory security suite",
                    ),
                ),
                (
                    release_jobs["test-inventory-security"],
                    ordinary_posix_binding_name,
                    (
                        "Require current-only test inventory policy",
                        "Run test inventory security suite",
                    ),
                ),
                (
                    release_gate,
                    release_posix_binding_name,
                    (
                        "Regenerate compatibility files",
                        "Check generated files are up to date",
                        "Require current-only build inventory policy",
                        "Require current-only test inventory policy",
                        "Run host tool smoke once",
                        "Test",
                        "Build install tree",
                        "Check ABI manifest",
                        "Require clean committed checkout before archiving",
                        "Provision fresh publication workspace",
                        "Pack source archive",
                        "Pack artifact",
                        "Verify publication workspace",
                    ),
                ),
            )
            self.assertEqual(
                45, sum(len(names) for _, _, names in protected_posix_jobs)
            )
            for gate, binding_name, names in protected_posix_jobs:
                self.assertEqual(
                    len(names),
                    gate.count(f"        shell: {trusted_posix_shell}\n"),
                )
                self.assertEqual(len(names), gate.count(trusted_posix_env))
                binding = named_step(gate, binding_name)
                self.assertLess(
                    gate.index(binding),
                    gate.index(f"        shell: {trusted_posix_shell}\n"),
                )
                for name in names:
                    step = named_step(gate, name)
                    self.assertIn(f"        shell: {trusted_posix_shell}\n", step)
                    self.assertIn(trusted_posix_env, step)
                    self.assertEqual(1, step.count("GITHUB_ENV=/dev/null"))
                    self.assertEqual(1, step.count("GITHUB_PATH=/dev/null"))
                    self.assertLess(
                        step.index("GITHUB_ENV=/dev/null"), step.index("/bin/bash")
                    )
                    self.assertLess(
                        step.index("GITHUB_PATH=/dev/null"), step.index("/bin/bash")
                    )

            trusted_windows_steps = (
                "Build Windows Python tooling executable fixtures and libraries",
                "Check Windows library layout and tooling fixture boundary",
                "Run Windows DLL ABI and CBLAS L1-L3 compatibility smoke (not inventory evidence)",
                "Windows native compile/link smoke for Debug (compatibility only; not inventory evidence)",
                "Windows native compile/link smoke for ReleaseSafe (compatibility only; not inventory evidence)",
                "Windows native compile/link smoke for ReleaseFast (compatibility only; not inventory evidence)",
            )
            windows_steps = {
                name: named_step(target_gate, name) for name in trusted_windows_steps
            }
            for step in windows_steps.values():
                self.assertIn(f"        shell: {trusted_windows_shell}\n", step)
                self.assertIn(trusted_windows_env, step)
                self.assertIn(
                    "        run: |\n"
                    "          Set-StrictMode -Version Latest\n"
                    "          $env:PATH = $env:TRUSTED_PATH\n",
                    step,
                )
                self.assertNotIn("Get-Command", step)
                self.assertEqual(1, step.count("$env:GITHUB_ENV = 'NUL'"))
                self.assertEqual(1, step.count("$env:GITHUB_PATH = 'NUL'"))
                self.assertLess(
                    step.index("$env:GITHUB_ENV = 'NUL'"),
                    step.index("[IO.File]::ReadAllText('{0}')"),
                )
                self.assertLess(
                    step.index("$env:GITHUB_PATH = 'NUL'"),
                    step.index("[IO.File]::ReadAllText('{0}')"),
                )
            self.assertIn(
                "& $env:TRUSTED_ZIG build install-libraries",
                windows_steps[trusted_windows_steps[0]],
            )
            self.assertEqual(
                2,
                windows_steps[trusted_windows_steps[1]].count(
                    "@(& $env:TRUSTED_ZIG ar t"
                ),
            )
            self.assertIn(
                "$smokeScript | & $env:TRUSTED_PYTHON -I -B - $completionNonce",
                windows_steps[trusted_windows_steps[2]],
            )
            for name in trusted_windows_steps[3:]:
                self.assertIn("& $env:TRUSTED_ZIG build -j1", windows_steps[name])

            fortran_step = named_step(target_gate, "Run Fortran example")
            self.assertIn(
                "        if: matrix.full_smoke && runner.os == 'Linux'\n",
                fortran_step,
            )
            self.assertEqual(2, fortran_step.count("          gfortran "))

            expected_channel_counts = (
                (ci_jobs["source-checks"], 7),
                (target_gate, 25),
                (ci_jobs["capability-builds"], 2),
                (release_jobs["build-inventory-security"], 3),
                (release_jobs["test-inventory-security"], 2),
                (release_gate, 13),
            )
            for gate, expected_count in expected_channel_counts:
                self.assertEqual(expected_count, gate.count("GITHUB_ENV"))
                self.assertEqual(expected_count, gate.count("GITHUB_PATH"))

        assert_trusted_tool_bindings(ci_source, release_source)

        build_ci_command = (
            '|\n'
            '          python3 -B test/build/test_build_inventory.py \\\n'
            '            BuildInventoryTests.test_positive_validation_and_cli \\\n'
            '            BuildInventoryTests.test_unlisted_named_step_fails_closed \\\n'
            '            BuildInventoryTests.test_unlisted_conditional_link_edge_fails_closed \\\n'
            '            BuildInventoryTests.test_unlisted_workflow_run_block_fails_closed \\\n'
            '            BuildInventoryTests.test_test_inventory_public_files_are_strictly_classified'
        )
        test_ci_command = (
            '|\n'
            '          python3 -B test/build/test_test_inventory.py \\\n'
            '            TestInventoryTests.test_positive_repository_validation \\\n'
            '            TestInventoryTests.test_duplicate_json_key_and_unknown_key_fail \\\n'
            '            TestInventoryTests.test_missing_duplicate_and_folded_logical_roots_fail \\\n'
            '            TestInventoryTests.test_native_compiler_sets_are_frozen_and_content_deduplicated'
        )

        terminal_singletons = (
            (
                "build-inventory-security",
                "Run build inventory security suite",
                build_ci_command,
                "terminal singleton build-inventory security run with GitHub command-file channels masked before repository execution and action post hooks",
            ),
            (
                "test-inventory-security",
                "Run test inventory security suite",
                test_ci_command,
                "terminal singleton test-inventory security run with GitHub command-file channels masked before repository execution and action post hooks",
            ),
            (
                "feature-compile",
                "Run matrix gate",
                "${{ matrix.command }}",
                "terminal singleton native-feature correctness or explicit build-only matrix run with GitHub command-file channels masked before repository execution and action post hooks",
            ),
        )

        def assert_terminal_singletons(source: str) -> None:
            for job, name, command, evidence_role in terminal_singletons:
                gate = job_block(source, job)
                step = named_step(gate, name)
                self.assertEqual(
                    f"      - name: {name}\n"
                    f"        shell: {terminal_singleton_shell}\n"
                    f"{terminal_singleton_env}"
                    f"        run: {command}",
                    step,
                )
                self.assertTrue(gate.rstrip().endswith(step))
                self.assertNotIn("TRUSTED_PATH", step)
                self.assertEqual(1, step.count("GITHUB_ENV=/dev/null"))
                self.assertEqual(1, step.count("GITHUB_PATH=/dev/null"))
                self.assertLess(
                    step.index("GITHUB_ENV=/dev/null"), step.index("/bin/bash")
                )
                self.assertLess(
                    step.index("GITHUB_PATH=/dev/null"), step.index("/bin/bash")
                )
                launch_id = (
                    f"workflow-launch:.github/workflows/ci.yml:{job}:"
                    + name.lower().replace(" ", "-")
                )
                reviewed = CHECKER.REVIEWED_NEW_WORKFLOW_LAUNCH_FIELDS[launch_id]
                self.assertEqual("always", reviewed["condition"])
                self.assertEqual(terminal_singleton_shell, reviewed["shell"])
                self.assertEqual(evidence_role, reviewed["evidence_role"])
                self.assertEqual(1, gate.count("GITHUB_ENV"))
                self.assertEqual(1, gate.count("GITHUB_PATH"))

        assert_terminal_singletons(ci_source)
        self.assertEqual(
            48,
            ci_source.count(f"        shell: {trusted_posix_shell}\n")
            + release_source.count(f"        shell: {trusted_posix_shell}\n"),
        )

        target_test_commands = {
            "Test Debug target": (
                "zig build test ${{ matrix.target_args }} "
                "-Dtest-optimize=debug -Dhost-tool-smoke=false --summary failures"
            ),
            "Test ReleaseSafe target": (
                "zig build --release=safe test ${{ matrix.target_args }} "
                "-Dtest-optimize=safe -Dhost-tool-smoke=false "
                "--summary failures"
            ),
            "Test ReleaseFast target": (
                "zig build --release=fast test ${{ matrix.target_args }} "
                "-Dtest-optimize=fast -Dhost-tool-smoke=false "
                "--summary failures"
            ),
        }
        host_tool_step = (
            "      - name: Run host tool smoke once\n"
            "        if: matrix.zig_gate == 'inventory-certified'\n"
            "        timeout-minutes: 60\n"
            f"        shell: {trusted_posix_shell}\n"
            f"{trusted_posix_env}"
            "        run: zig build test-host-tool-smoke "
            "${{ matrix.target_args }} --summary failures"
        )
        target_link_commands = {
            "Link test inventory for Debug target (POSIX structure-gated)": (
                "zig build test-inventory-link ${{ matrix.target_args }} "
                "-Dtest-optimize=debug --summary failures"
            ),
            "Link test inventory for ReleaseSafe target (POSIX structure-gated)": (
                "zig build --release=safe test-inventory-link "
                "${{ matrix.target_args }} -Dtest-optimize=safe "
                "--summary failures"
            ),
            "Link test inventory for ReleaseFast target (POSIX structure-gated)": (
                "zig build --release=fast test-inventory-link "
                "${{ matrix.target_args }} -Dtest-optimize=fast "
                "--summary failures"
            ),
        }
        windows_link_commands = {
            "Windows native compile/link smoke for Debug (compatibility only; not inventory evidence)": (
                "& $env:TRUSTED_ZIG build -j1 test-inventory-link-windows-native-smoke "
                "${{ matrix.target_args }} -Dtest-optimize=debug --summary failures"
            ),
            "Windows native compile/link smoke for ReleaseSafe (compatibility only; not inventory evidence)": (
                "& $env:TRUSTED_ZIG build -j1 --release=safe test-inventory-link-windows-native-smoke "
                "${{ matrix.target_args }} -Dtest-optimize=safe --summary failures"
            ),
            "Windows native compile/link smoke for ReleaseFast (compatibility only; not inventory evidence)": (
                "& $env:TRUSTED_ZIG build -j1 --release=fast test-inventory-link-windows-native-smoke "
                "${{ matrix.target_args }} -Dtest-optimize=fast --summary failures"
            ),
        }

        def assert_target_test_watchdogs(source: str) -> None:
            target_gate = job_block(source, "target-tests")
            self.assertNotIn("host_tool_smoke:", target_gate)
            self.assertNotIn("matrix.host_tool_smoke", target_gate)
            self.assertNotIn("continue-on-error", target_gate)
            linux_x86_row = target_gate.split("- name: Linux / x86_64 baseline", 1)[1].split("- name:", 1)[0]
            self.assertIn("zig_gate: link-only", linux_x86_row)
            self.assertIn("target_args: -Dtarget=x86_64-linux-gnu -Dcpu=baseline", linux_x86_row)
            self.assertIn(
                "          - name: Linux / ARM64 baseline\n"
                "            os: ubuntu-24.04-arm\n"
                "            cache_target: linux-arm64-baseline\n"
                "            zig_gate: link-only\n"
                "            target_args: -Dtarget=aarch64-linux-gnu -Dcpu=baseline\n",
                target_gate,
            )
            self.assertIn(
                "          - name: Windows / x86_64 native compile/link compatibility smoke\n"
                "            os: windows-2025\n"
                "            cache_target: windows-x86_64-baseline\n"
                "            zig_gate: windows-native-compile-link-smoke\n"
                "            target_args: -Dtarget=x86_64-windows-gnu -Dcpu=baseline\n"
                "            install: false\n",
                target_gate,
            )
            self.assertEqual(
                1, target_gate.count("      - name: Run host tool smoke once")
            )
            self.assertEqual(1, target_gate.count("zig build test-host-tool-smoke"))
            self.assertEqual(
                host_tool_step, named_step(target_gate, "Run host tool smoke once")
            )
            for step_name, command in target_test_commands.items():
                step = named_step(target_gate, step_name)
                self.assertEqual(
                    f"      - name: {step_name}\n"
                    "        if: matrix.zig_gate == 'inventory-certified'\n"
                    f"        timeout-minutes: {60 if step_name == 'Test Debug target' else 120}\n"
                    f"        shell: {trusted_posix_shell}\n"
                    f"{trusted_posix_env}"
                    f"        run: {command}",
                    step,
                )
            for step_name, command in target_link_commands.items():
                step = named_step(target_gate, step_name)
                self.assertEqual(
                    f"      - name: {step_name}\n"
                    "        if: matrix.zig_gate == 'link-only'\n"
                    "        timeout-minutes: 60\n"
                    f"        shell: {trusted_posix_shell}\n"
                    f"{trusted_posix_env}"
                    f"        run: {command}",
                    step,
                )
                self.assertNotIn("host-tool-smoke", step)
            for step_name, command in windows_link_commands.items():
                step = named_step(target_gate, step_name)
                self.assertEqual(
                    f"      - name: {step_name}\n"
                    "        if: matrix.zig_gate == 'windows-native-compile-link-smoke'\n"
                    "        timeout-minutes: 60\n"
                    f"        shell: {trusted_windows_shell}\n"
                    f"{trusted_windows_env}"
                    "        run: |\n"
                    "          Set-StrictMode -Version Latest\n"
                    "          $env:PATH = $env:TRUSTED_PATH\n"
                    "          $ErrorActionPreference = 'Stop'\n"
                    f"          {command}",
                    step,
                )
                self.assertNotIn("test-inventory-link ${{", step)

        assert_target_test_watchdogs(ci_source)
        windows_debug_step = named_step(
            job_block(ci_source, "target-tests"), next(iter(windows_link_commands))
        )
        windows_debug_timeout_mutant = windows_debug_step.replace(
            "        timeout-minutes: 60\n",
            "        timeout-minutes: 61\n",
            1,
        )
        self.assertNotEqual(windows_debug_step, windows_debug_timeout_mutant)
        target_test_mutants = (
            mutate_job(
                ci_source,
                "target-tests",
                "            install: true\n",
                "            host_tool_smoke: true\n            install: true\n",
            ),
            mutate_job(
                ci_source,
                "target-tests",
                "        timeout-minutes: 60\n",
                "        timeout-minutes: 59\n",
            ),
            mutate_job(
                ci_source,
                "target-tests",
                "-Dhost-tool-smoke=false --summary failures\n",
                "-Dhost-tool-smoke=${{ matrix.host_tool_smoke }} --summary failures\n",
            ),
            mutate_job(
                ci_source,
                "target-tests",
                host_tool_step + "\n",
                host_tool_step + "\n\n" + host_tool_step + "\n",
            ),
            mutate_job(
                ci_source,
                "target-tests",
                "-Dtest-optimize=debug --summary failures\n",
                "-Dtest-optimize=debug -Dhost-tool-smoke=false --summary failures\n",
            ),
            mutate_job(
                ci_source,
                "target-tests",
                "            zig_gate: windows-native-compile-link-smoke\n",
                "            zig_gate: link-only\n",
            ),
            mutate_job(
                ci_source,
                "target-tests",
                "        if: matrix.zig_gate == 'windows-native-compile-link-smoke'\n",
                "        if: matrix.zig_gate == 'link-only'\n",
            ),
            mutate_job(
                ci_source,
                "target-tests",
                "test-inventory-link-windows-native-smoke ${{ matrix.target_args }}",
                "test-inventory-link ${{ matrix.target_args }}",
            ),
            mutate_job(
                ci_source,
                "target-tests",
                "& $env:TRUSTED_ZIG build -j1 test-inventory-link-windows-native-smoke",
                "& $env:TRUSTED_ZIG build test-inventory-link-windows-native-smoke",
            ),
            mutate_job(
                ci_source,
                "target-tests",
                "& $env:TRUSTED_ZIG build -j1 --release=safe test-inventory-link-windows-native-smoke",
                "& $env:TRUSTED_ZIG build -j2 --release=safe test-inventory-link-windows-native-smoke",
            ),
            mutate_job(
                ci_source,
                "target-tests",
                "& $env:TRUSTED_ZIG build -j1 --release=fast test-inventory-link-windows-native-smoke",
                "& $env:TRUSTED_ZIG build --release=fast test-inventory-link-windows-native-smoke",
            ),
            mutate_job(
                ci_source,
                "target-tests",
                windows_debug_step,
                windows_debug_timeout_mutant,
            ),
            mutate_job(
                ci_source,
                "target-tests",
                "      - name: Windows native compile/link smoke for Debug (compatibility only; not inventory evidence)\n",
                "      - name: Windows native compile/link smoke for Debug (compatibility only; not inventory evidence)\n"
                "        continue-on-error: true\n",
            ),
        )
        for mutant in target_test_mutants:
            with self.assertRaises(AssertionError):
                assert_target_test_watchdogs(mutant)

        target_gate = job_block(ci_source, "target-tests")
        target_posix_binding = named_step(target_gate, ci_posix_binding_name)
        first_posix_step = named_step(target_gate, "Run host tool smoke once")
        late_target_binding = target_gate.replace(
            target_posix_binding + "\n\n", "", 1
        ).replace(
            first_posix_step,
            first_posix_step + "\n\n" + target_posix_binding,
            1,
        )
        self.assertNotEqual(target_gate, late_target_binding)
        binding_mutants = (
            ci_source.replace(target_posix_binding + "\n\n", "", 1),
            ci_source.replace(target_gate, late_target_binding, 1),
            mutate_job(
                ci_source,
                "source-checks",
                "/bin/bash --noprofile --norc -p -eo pipefail {0}",
                "/bin/bash --noprofile --norc -eo pipefail {0}",
            ),
            mutate_job(
                ci_source,
                "source-checks",
                f"        shell: {trusted_posix_shell}\n",
                "        shell: bash\n",
            ),
            mutate_job(
                ci_source,
                "source-checks",
                "          BASH_ENV: /dev/null\n",
                "",
            ),
            mutate_job(
                release_source,
                "artifacts",
                "          ENV: /dev/null\n",
                "",
            ),
            mutate_job(
                ci_source,
                "source-checks",
                "          PATH: ${{ steps.trusted-posix-tools.outputs.path }}\n",
                "",
            ),
            mutate_job(
                ci_source,
                "source-checks",
                "          PATH: ${{ steps.trusted-posix-tools.outputs.path }}\n",
                "          PATH: ${{ env.PATH }}\n",
            ),
            mutate_job(
                ci_source,
                "source-checks",
                " GITHUB_ENV=/dev/null",
                "",
            ),
            mutate_job(
                ci_source,
                "source-checks",
                "GITHUB_PATH=/dev/null",
                "GITHUB_PATH=/tmp/zynum-path",
            ),
            mutate_job(
                ci_source,
                "source-checks",
                "GITHUB_ENV=/dev/null",
                "GITHUB_ENV=/tmp/zynum-env",
            ),
            mutate_job(
                ci_source,
                "source-checks",
                '>> "$GITHUB_OUTPUT"',
                '>> "$GITHUB_ENV"',
            ),
            mutate_job(
                ci_source,
                "source-checks",
                'zig_version="$("$trusted_zig" version)"',
                "zig_version=0.17.0",
            ),
            mutate_job(
                ci_source,
                "target-tests",
                f"        shell: {trusted_windows_shell}\n",
                "        shell: pwsh\n",
            ),
            mutate_job(
                ci_source,
                "target-tests",
                "{{ exit $LASTEXITCODE }}",
                "{ exit $LASTEXITCODE }",
            ),
            mutate_job(
                ci_source,
                "target-tests",
                "$env:GITHUB_ENV = 'NUL'; ",
                "",
            ),
            mutate_job(
                ci_source,
                "target-tests",
                "$env:GITHUB_PATH = 'NUL'; ",
                "",
            ),
            mutate_job(
                ci_source,
                "target-tests",
                "$env:GITHUB_ENV = 'NUL'",
                "$env:GITHUB_ENV = 'NULL'",
            ),
            mutate_job(
                ci_source,
                "target-tests",
                "$env:GITHUB_PATH = 'NUL'",
                "$env:GITHUB_PATH = 'C:\\Temp\\zynum-path'",
            ),
            mutate_job(
                ci_source,
                "target-tests",
                "$env:GITHUB_ENV = 'NUL'; $env:GITHUB_PATH = 'NUL'; & ([scriptblock]::Create([IO.File]::ReadAllText('{0}')))",
                "& ([scriptblock]::Create([IO.File]::ReadAllText('{0}'))); $env:GITHUB_ENV = 'NUL'; $env:GITHUB_PATH = 'NUL'",
            ),
            mutate_job(
                ci_source,
                "target-tests",
                "[IO.File]::AppendAllLines($env:GITHUB_OUTPUT",
                "[IO.File]::AppendAllLines($env:GITHUB_ENV",
            ),
            mutate_job(
                ci_source,
                "target-tests",
                "$utf8NoBom = New-Object Text.UTF8Encoding($false)",
                "$utf8NoBom = New-Object Text.UTF8Encoding($true)",
            ),
            mutate_job(
                ci_source,
                "target-tests",
                "& $env:TRUSTED_ZIG build install-libraries",
                "zig build install-libraries",
            ),
            mutate_job(
                ci_source,
                "target-tests",
                "$smokeScript | & $env:TRUSTED_PYTHON -I -B - $completionNonce",
                "$smokeScript | python -I -B - $completionNonce",
            ),
            mutate_job(
                ci_source,
                "target-tests",
                "$env:PATH = $env:TRUSTED_PATH\n",
                "",
            ),
            mutate_job(
                release_source,
                "artifacts",
                'trusted_gfortran="$(resolve_trusted_tool gfortran)"',
                'trusted_gfortran="$(command -v gfortran)"',
            ),
        )
        for index, mutant in enumerate(binding_mutants):
            with self.subTest(trusted_binding_mutant=index):
                with self.assertRaises((AssertionError, ValueError)):
                    if mutant.startswith(
                        "# Copyright (C) 2026 Zynum contributors\n# SPDX-License-Identifier: LGPL-3.0-or-later\n\nname: Release"
                    ):
                        assert_trusted_tool_bindings(ci_source, mutant)
                    else:
                        assert_trusted_tool_bindings(mutant, release_source)

        singleton_mutants = (
            mutate_job(
                ci_source,
                "build-inventory-security",
                " GITHUB_ENV=/dev/null",
                "",
            ),
            mutate_job(
                ci_source,
                "test-inventory-security",
                " GITHUB_PATH=/dev/null",
                "",
            ),
            mutate_job(
                ci_source,
                "feature-compile",
                "GITHUB_ENV=/dev/null",
                "GITHUB_ENV=/tmp/zynum-env",
            ),
            mutate_job(
                ci_source,
                "feature-compile",
                "GITHUB_PATH=/dev/null",
                "GITHUB_PATH=/tmp/zynum-path",
            ),
            mutate_job(
                ci_source,
                "build-inventory-security",
                f"        shell: {terminal_singleton_shell}\n",
                "        shell: bash\n",
            ),
            mutate_job(
                ci_source,
                "test-inventory-security",
                "--norc -p -eo pipefail",
                "--norc -eo pipefail",
            ),
            mutate_job(
                ci_source,
                "build-inventory-security",
                "          BASH_ENV: /dev/null\n",
                "",
            ),
            mutate_job(
                ci_source,
                "test-inventory-security",
                "          ENV: /dev/null\n",
                "",
            ),
            mutate_job(
                ci_source,
                "feature-compile",
                "        run: ${{ matrix.command }}\n",
                "        run: ${{ matrix.command }}\n\n"
                "      - uses: actions/cache@caa296126883cff596d87d8935842f9db880ef25 # v5.1.0\n",
            ),
        )
        for index, mutant in enumerate(singleton_mutants):
            with self.subTest(terminal_singleton_mutant=index):
                with self.assertRaises((AssertionError, ValueError)):
                    assert_terminal_singletons(mutant)

        build_suite_command = "python3 -B test/build/test_build_inventory.py"
        test_suite_command = "python3 -B test/build/test_test_inventory.py"

        def assert_security_suite_split(
            source: str, prerequisite_job: str, *, release: bool
        ) -> None:
            prerequisite = job_block(source, prerequisite_job)
            build_gate = job_block(source, "build-inventory-security")
            test_gate = job_block(source, "test-inventory-security")
            build_step = named_step(build_gate, "Run build inventory security suite")
            test_step = named_step(test_gate, "Run test inventory security suite")
            suite_protection = (
                f"        shell: {trusted_posix_shell}\n{trusted_posix_env}"
                if release
                else f"        shell: {terminal_singleton_shell}\n{terminal_singleton_env}"
            )
            self.assertEqual(
                f"      - name: Run build inventory security suite\n"
                f"{suite_protection}"
                f"        run: {build_suite_command if release else build_ci_command}",
                build_step,
            )
            self.assertEqual(
                f"      - name: Run test inventory security suite\n"
                f"{suite_protection}"
                f"        run: {test_suite_command if release else test_ci_command}",
                test_step,
            )
            self.assertNotIn(test_suite_command, build_gate)
            self.assertNotIn(build_suite_command, test_gate)
            self.assertNotIn(build_suite_command, prerequisite)
            self.assertNotIn(test_suite_command, prerequisite)
            self.assertEqual(1, source.count(build_suite_command))
            self.assertEqual(1, source.count(test_suite_command))
            for gate in (build_gate, test_gate):
                checkout_steps = [
                    line
                    for line in gate.splitlines()
                    if line.startswith("      - uses: actions/checkout@")
                ]
                setup_zig_steps = [
                    line
                    for line in gate.splitlines()
                    if line.startswith("      - uses: mlugg/setup-zig@")
                ]
                self.assertEqual(1, len(checkout_steps))
                self.assertEqual(1, len(setup_zig_steps))
                self.assertRegex(
                    checkout_steps[0],
                    (
                        r"^      - uses: actions/checkout@[0-9a-f]{40} "
                        r"# v5\.[0-9]+\.[0-9]+$"
                    ),
                )
                self.assertRegex(
                    setup_zig_steps[0],
                    (
                        r"^      - uses: mlugg/setup-zig@[0-9a-f]{40} "
                        r"# v[0-9]+\.[0-9]+\.[0-9]+$"
                    ),
                )
            if release:
                matrix = "        os: [macos-15, ubuntu-24.04]"
                self.assertEqual(1, build_gate.count(matrix))
                self.assertEqual(1, test_gate.count(matrix))
                for gate in (build_gate, test_gate):
                    self.assertNotIn("actions/upload-artifact@", gate)
                    self.assertNotIn("actions/download-artifact@", gate)
                build_policy = named_step(
                    build_gate, "Require current-only build inventory policy"
                )
                build_test_policy = named_step(
                    build_gate, "Require current-only test inventory policy"
                )
                test_policy = named_step(
                    test_gate, "Require current-only test inventory policy"
                )
                self.assertLess(
                    build_gate.index(build_policy), build_gate.index(build_step)
                )
                self.assertLess(
                    build_gate.index(build_test_policy), build_gate.index(build_step)
                )
                self.assertLess(
                    test_gate.index(test_policy), test_gate.index(test_step)
                )

        assert_security_suite_split(ci_source, "source-checks", release=False)
        assert_security_suite_split(release_source, "artifacts", release=True)

        required_ci_gates = "    needs: source-checks"
        for job in ("target-tests", "capability-builds", "feature-compile"):
            with self.subTest(required_ci_gate=job):
                self.assertEqual(1, job_block(ci_source, job).count(required_ci_gates))
        for job in ("build-inventory-security", "test-inventory-security"):
            self.assertEqual(
                1, job_block(ci_source, job).count("    needs: source-checks")
            )
        self.assertEqual(
            1,
            job_block(release_source, "artifacts").count(
                "    needs: [build-inventory-security, test-inventory-security]"
            ),
        )

        required_ci_gate_dependencies = (
            ("source-checks", "SOURCE_CHECKS_RESULT"),
            ("build-inventory-security", "BUILD_INVENTORY_SECURITY_RESULT"),
            ("test-inventory-security", "TEST_INVENTORY_SECURITY_RESULT"),
            ("target-tests", "TARGET_TESTS_RESULT"),
            ("capability-builds", "CAPABILITY_BUILDS_RESULT"),
            ("feature-compile", "FEATURE_COMPILE_RESULT"),
        )

        def assert_aggregate_ci_gate(source: str) -> None:
            gate = job_block(source, "ci-gate")
            self.assertEqual(1, gate.count("    if: ${{ always() }}"))
            for dependency, variable in required_ci_gate_dependencies:
                self.assertEqual(1, gate.count(f"      - {dependency}"))
                self.assertEqual(1, gate.count(f"needs.{dependency}.result"))
                self.assertEqual(2, gate.count(variable))
                self.assertEqual(1, gate.count(f'            "${variable}"'))
            self.assertNotIn("uses:", gate)
            step = named_step(gate, "Require every CI gate to succeed")
            self.assertIn("        shell: bash", step)
            self.assertIn("          set -euo pipefail", step)
            self.assertIn('          for result in "${results[@]}"; do', step)
            self.assertIn('            if [[ "$result" != "success" ]]; then', step)
            self.assertIn("              exit 1", step)

        assert_aggregate_ci_gate(ci_source)
        aggregate_mutants = (
            mutate_job(
                ci_source,
                "ci-gate",
                "    if: ${{ always() }}\n",
                "    if: ${{ success() }}\n",
            ),
            mutate_job(
                ci_source,
                "ci-gate",
                "      - test-inventory-security\n",
                "",
            ),
            mutate_job(
                ci_source,
                "ci-gate",
                "          TEST_INVENTORY_SECURITY_RESULT: "
                "${{ needs.test-inventory-security.result }}\n",
                "",
            ),
            mutate_job(
                ci_source,
                "ci-gate",
                '            "$TEST_INVENTORY_SECURITY_RESULT"\n',
                "",
            ),
            mutate_job(
                ci_source,
                "ci-gate",
                "              exit 1\n",
                "              exit 0\n",
            ),
        )
        for mutant in aggregate_mutants:
            self.assertNotEqual(ci_source, mutant)
            with self.assertRaises(AssertionError):
                assert_aggregate_ci_gate(mutant)

        for source, prerequisite_job, release in (
            (ci_source, "source-checks", False),
            (release_source, "artifacts", True),
        ):
            merged_suite = (
                "        run: |\n"
                f"          {build_suite_command}\n"
                f"          {test_suite_command}"
            )
            mutant = mutate_job(
                source,
                "build-inventory-security",
                f"        run: {build_suite_command if release else build_ci_command}",
                merged_suite,
            )
            with (
                self.subTest(serial_suite_mutant=prerequisite_job),
                self.assertRaises(AssertionError),
            ):
                assert_security_suite_split(mutant, prerequisite_job, release=release)

        provision = named_step(release_source, "Provision fresh publication workspace")
        source_package = named_step(release_source, "Pack source archive")
        binary_package = named_step(release_source, "Pack artifact")
        verifier = named_step(release_source, "Verify publication workspace")
        upload_prefix = "      - uses: actions/upload-artifact@"
        upload_start = release_source.index(upload_prefix)
        upload_marker = release_source[
            upload_start : release_source.index("\n", upload_start) + 1
        ]
        self.assertRegex(
            upload_marker,
            r"^      - uses: actions/upload-artifact@[0-9a-f]{40} # v6\.[0-9]+\.[0-9]+\n$",
        )
        upload = release_source[upload_start:]

        self.assertNotIn("dist/", release_source)
        self.assertLess(
            release_source.index(provision), release_source.index(source_package)
        )
        self.assertLess(
            release_source.index(source_package), release_source.index(binary_package)
        )
        self.assertLess(
            release_source.index(binary_package), release_source.index(verifier)
        )
        self.assertLess(release_source.index(verifier), upload_start)
        self.assertIn(
            'workspace="$(mktemp -d "${RUNNER_TEMP}/zynum-publication.XXXXXXXX")"',
            provision,
        )
        self.assertIn(
            'find "$workspace" -mindepth 1 -maxdepth 1 -print -quit', provision
        )
        self.assertIn(
            'printf \'path=%s\\n\' "$workspace" >> "$GITHUB_OUTPUT"', provision
        )

        self.assertIn(
            f'--archive "{output_ref}/zynum-source-${{{{ matrix.artifact }}}}.tar.gz"',
            source_package,
        )
        self.assertIn(f'workspace="{output_ref}"', binary_package)
        self.assertIn(f'workspace="{output_ref}"', verifier)
        publication_artifacts = (
            "${{ matrix.artifact }}.tar.gz",
            "${{ matrix.artifact }}.tar.gz.sha256",
            "zynum-source-${{ matrix.artifact }}.tar.gz",
            "zynum-source-${{ matrix.artifact }}.tar.gz.sha256",
        )
        checksum_commands = (
            'shasum -a 256 "${{ matrix.artifact }}.tar.gz" > '
            '"${{ matrix.artifact }}.tar.gz.sha256"',
            'shasum -a 256 "zynum-source-${{ matrix.artifact }}.tar.gz" > '
            '"zynum-source-${{ matrix.artifact }}.tar.gz.sha256"',
        )
        self.assertIn('cd "$workspace"', binary_package)
        observed_checksum_commands = tuple(
            line.strip()
            for line in binary_package.splitlines()
            if "shasum -a 256" in line
        )
        self.assertEqual(checksum_commands, observed_checksum_commands)
        self.assertLess(
            binary_package.index('cd "$workspace"'),
            binary_package.index(checksum_commands[0]),
        )
        self.assertFalse(
            any("$workspace/" in command for command in observed_checksum_commands)
        )
        self.assertIn(
            '-C "$GITHUB_WORKSPACE" LICENSE COPYING',
            binary_package,
        )
        self.assertIn("licenses=(LICENSE COPYING)", binary_package)
        self.assertIn('members="$(tar -tzf "$binary_archive")"', binary_package)
        self.assertIn('grep -Fxc "$license"', binary_package)
        self.assertIn('test "$count" -eq 1', binary_package)
        self.assertIn(
            'cmp - "$GITHUB_WORKSPACE/$license"',
            binary_package,
        )
        package_steps = source_package + binary_package
        for artifact in publication_artifacts:
            with self.subTest(publication_artifact=artifact):
                packaged_path = (
                    f'"{artifact}"'
                    if artifact.endswith(".sha256")
                    else (
                        f"{output_ref}/{artifact}"
                        if artifact.startswith("zynum-source-")
                        else f"$workspace/{artifact}"
                    )
                )
                self.assertIn(packaged_path, package_steps)
                self.assertIn(f'"{artifact}"', verifier)
                self.assertIn(f"{output_ref}/{artifact}", upload)

        self.assertEqual(4, upload.count(f"{output_ref}/"))
        upload_path_source = upload.split("          path: |\n", 1)[1]
        self.assertEqual(
            tuple(f"{output_ref}/{artifact}" for artifact in publication_artifacts),
            tuple(
                line.strip() for line in upload_path_source.splitlines() if line.strip()
            ),
        )
        self.assertIn('arena="$workspace/.zynum-cleanup-v2-$(id -u)"', verifier)
        self.assertNotIn(".zynum-cleanup-v1-", verifier)
        self.assertNotIn("dist/", verifier)
        self.assertNotIn("rm ", verifier)
        verifier_find_lines = [
            line.strip() for line in verifier.splitlines() if "find " in line
        ]
        self.assertEqual(2, len(verifier_find_lines))
        self.assertTrue(
            all(
                'find "$arena" ' in line or 'find "$workspace" ' in line
                for line in verifier_find_lines
            ),
            verifier_find_lines,
        )
        self.assertIn(
            'test "$(find "$workspace" -mindepth 1 -maxdepth 1 | wc -l '
            "| tr -d ' ')\" -eq 5",
            verifier,
        )
        self.assertLess(
            verifier.index('find "$arena" '),
            verifier.index("          for artifact in"),
        )
        self.assertLess(
            verifier.index("          for artifact in"),
            verifier.index('test "$(find "$workspace" '),
        )

        legacy_checkout_text = "dist/.zynum-cleanup-v1-$(id -u)/legacy-checkout-mutant"
        provision_start = release_source.index(
            "      - name: Provision fresh publication workspace\n"
        )
        checkout_prefix = "      - uses: actions/checkout@"
        checkout_start = release_source.index(checkout_prefix)
        checkout_line = release_source[
            checkout_start : release_source.index("\n", checkout_start) + 1
        ]
        self.assertRegex(
            checkout_line,
            r"^      - uses: actions/checkout@[0-9a-f]{40} # v5\.[0-9]+\.[0-9]+\n$",
        )
        checkout_mutant = (
            release_source[:provision_start].replace(
                checkout_line,
                checkout_line
                + f"      # stale checkout text: {legacy_checkout_text}\n",
                1,
            )
            + release_source[provision_start:]
        )
        mutant_provision_start = checkout_mutant.index(
            "      - name: Provision fresh publication workspace\n"
        )
        self.assertIn(legacy_checkout_text, checkout_mutant[:mutant_provision_start])
        self.assertEqual(
            release_source[provision_start:],
            checkout_mutant[mutant_provision_start:],
        )
        self.assertNotIn(legacy_checkout_text, checkout_mutant[mutant_provision_start:])

    def test_binary_release_archive_requires_exact_license_documents(self) -> None:
        def write_archive(path: Path, payloads: dict[str, bytes]) -> None:
            with tarfile.open(path, "w:gz") as archive:
                for name, payload in payloads.items():
                    member = tarfile.TarInfo(name)
                    member.size = len(payload)
                    archive.addfile(member, io.BytesIO(payload))

        expected = {
            name: (REPOSITORY_ROOT / name).read_bytes()
            for name in ("LICENSE", "COPYING")
        }
        legacy = self.root / "legacy-binary.tar.gz"
        write_archive(legacy, {"LICENSE": expected["LICENSE"]})
        with self.assertRaises(AssertionError):
            self._assert_binary_license_archive(legacy)

        corrupted = self.root / "corrupted-binary.tar.gz"
        write_archive(corrupted, {**expected, "COPYING": b"not the GPL\n"})
        with self.assertRaises(AssertionError):
            self._assert_binary_license_archive(corrupted)

        redundant = self.root / "redundant-binary.tar.gz"
        write_archive(redundant, {**expected, "COPYING.LESSER": expected["LICENSE"]})
        with self.assertRaises(AssertionError):
            self._assert_binary_license_archive(redundant)

        complete = self.root / "complete-binary.tar.gz"
        write_archive(complete, expected)
        self._assert_binary_license_archive(complete)

    def test_workflow_property_nesting_changes_launch_digest(self) -> None:
        path = self.root / ".github/workflows/property-indent-fixture.yml"
        nested = (
            "jobs:\n"
            "  fixture:\n"
            "    steps:\n"
            "      - env:\n"
            "          X: y\n"
            "          shell: bash\n"
            "        run: printf '<%s>' \"$shell\"\n"
        )
        launch_id = "workflow-launch:.github/workflows/property-indent-fixture.yml:fixture:step-1"
        path.write_text(nested, encoding="utf-8")
        before = {
            item["id"]: item["source_digest"]
            for item in CHECKER._discover_workflow_launches(self.root)
        }[launch_id]
        path.write_text(
            nested.replace("          shell", "        shell"), encoding="utf-8"
        )
        after = {
            item["id"]: item["source_digest"]
            for item in CHECKER._discover_workflow_launches(self.root)
        }[launch_id]
        self.assertNotEqual(before, after)

    def test_multiline_plain_workflow_run_fails_closed(self) -> None:
        path = self.root / ".github/workflows/plain-multiline-fixture.yml"
        path.write_text(
            "jobs:\n"
            "  fixture-job:\n"
            "    steps:\n"
            "      - run: printf '<%s>' 'foo\n"
            "\n"
            "          bar'\n",
            encoding="utf-8",
        )
        with self.assertRaisesRegex(
            CHECKER.InventoryError,
            "multiline plain workflow run values are unsupported",
        ):
            CHECKER._discover_workflow_launches(self.root)

    def test_duplicate_workflow_steps_mapping_fails_closed(self) -> None:
        path = self.root / ".github/workflows/duplicate-steps-fixture.yml"
        path.write_text(
            "name: Duplicate steps fixture\n"
            "jobs:\n"
            "  fixture-job:\n"
            "    runs-on: ubuntu-latest\n"
            "    steps:\n"
            "      - uses: actions/checkout@v4\n"
            "    steps:\n"
            "      - run: python3 -V\n",
            encoding="utf-8",
        )
        with self.assertRaisesRegex(
            CHECKER.InventoryError,
            "duplicate (?:steps mapping|workflow mapping key 'steps')",
        ):
            CHECKER._discover_workflow_launches(self.root)

    def test_quoted_workflow_run_whitespace_changes_digest(self) -> None:
        path = self.root / ".github/workflows/quoted-space-fixture.yml"
        source = (
            "name: Quoted space fixture\n"
            "jobs:\n"
            "  fixture-job:\n"
            "    runs-on: ubuntu-latest\n"
            "    steps:\n"
            "      - run: \"printf 'a  b'\"\n"
        )
        path.write_text(source, encoding="utf-8")
        launch_id = (
            "workflow-launch:.github/workflows/quoted-space-fixture.yml:"
            "fixture-job:step-1"
        )
        before = {
            item["id"]: item["source_digest"]
            for item in CHECKER._discover_workflow_launches(self.root)
        }[launch_id]
        path.write_text(source.replace("a  b", "a b"), encoding="utf-8")
        after = {
            item["id"]: item["source_digest"]
            for item in CHECKER._discover_workflow_launches(self.root)
        }[launch_id]
        self.assertNotEqual(before, after)

    def test_non_string_workflow_run_scalars_fail_closed(self) -> None:
        path = self.root / ".github/workflows/non-string-run-fixture.yml"
        template = (
            "name: Non-string run fixture\n"
            "jobs:\n"
            "  fixture-job:\n"
            "    runs-on: ubuntu-latest\n"
            "    steps:\n"
            "      - run: {value}\n"
        )
        for value in (
            "true",
            "42",
            "3.5",
            ".5",
            "+.5",
            "-.5",
            "01",
            "00",
            "2026-07-18",
        ):
            with self.subTest(value=value):
                path.write_text(template.format(value=value), encoding="utf-8")
                with self.assertRaisesRegex(
                    CHECKER.InventoryError, "must be a scalar string"
                ):
                    CHECKER._discover_workflow_launches(self.root)

    def test_yaml_1_2_string_like_run_scalars_are_supported(self) -> None:
        path = self.root / ".github/workflows/yaml-1-2-string-fixture.yml"
        for value in ("yes", "no", "on", "off"):
            with self.subTest(value=value):
                path.write_text(
                    f"jobs:\n  fixture:\n    steps:\n      - run: {value}\n",
                    encoding="utf-8",
                )
                launch_ids = {
                    item["id"]
                    for item in CHECKER._discover_workflow_launches(self.root)
                }
                self.assertIn(
                    "workflow-launch:.github/workflows/yaml-1-2-string-fixture.yml:fixture:step-1",
                    launch_ids,
                )

    def test_indented_comments_after_inline_run_are_supported(self) -> None:
        path = self.root / ".github/workflows/inline-run-comment-fixture.yml"
        path.write_text(
            "jobs:\n"
            "  fixture:\n"
            "    steps:\n"
            "      - run: echo first\n"
            "                    # valid indented comment\n"
            "      - run: echo second\n",
            encoding="utf-8",
        )
        launch_ids = {
            item["id"] for item in CHECKER._discover_workflow_launches(self.root)
        }
        self.assertIn(
            "workflow-launch:.github/workflows/inline-run-comment-fixture.yml:fixture:step-2",
            launch_ids,
        )

    def test_invalid_plain_workflow_run_scalars_fail_closed(self) -> None:
        path = self.root / ".github/workflows/invalid-plain-run-fixture.yml"
        template = "jobs:\n  fixture:\n    steps:\n      - run: {value}\n"
        for value in (
            "echo: hidden",
            "- echo hidden",
            "@echo hidden",
            "`echo hidden`",
            "]bad",
            "}bad",
            "%bad",
            ",bad",
        ):
            with self.subTest(value=value):
                path.write_text(template.format(value=value), encoding="utf-8")
                with self.assertRaisesRegex(
                    CHECKER.InventoryError, "invalid plain workflow .* scalar"
                ):
                    CHECKER._discover_workflow_launches(self.root)

    def test_tab_separated_plain_scalar_indicators_fail_closed(self) -> None:
        path = self.root / ".github/workflows/tab-indicator-fixture.yml"
        for value in ("-\techo hidden", "?\techo hidden"):
            with self.subTest(value=value):
                path.write_text(
                    f"jobs:\n  fixture:\n    steps:\n      - run: {value}\n",
                    encoding="utf-8",
                )
                with self.assertRaisesRegex(
                    CHECKER.InventoryError, "invalid plain workflow .* scalar"
                ):
                    CHECKER._discover_workflow_launches(self.root)

    def test_workflow_mapping_colon_requires_separation(self) -> None:
        path = self.root / ".github/workflows/compact-colon-fixture.yml"
        for step in ("- run:echo hidden", "- {run:echo hidden}"):
            with self.subTest(step=step):
                path.write_text(
                    f"jobs:\n  fixture:\n    steps:\n      {step}\n",
                    encoding="utf-8",
                )
                with self.assertRaises(CHECKER.InventoryError):
                    CHECKER._discover_workflow_launches(self.root)

        path.write_text(
            'jobs:\n  fixture:\n    steps:\n      - {"run":"echo visible"}\n',
            encoding="utf-8",
        )
        self._assert_error_contains(
            "workflow-launch:.github/workflows/compact-colon-fixture.yml:fixture:step-1"
        )

    def test_flow_indicators_inside_plain_flow_scalars_fail_closed(self) -> None:
        path = self.root / ".github/workflows/flow-indicator-fixture.yml"
        for run in ("echo[hidden]", "echo{hidden}", "echo[hidden,there]", "foo:"):
            with self.subTest(run=run):
                path.write_text(
                    f"jobs:\n  fixture:\n    steps:\n      - {{ run: {run} }}\n",
                    encoding="utf-8",
                )
                with self.assertRaisesRegex(
                    CHECKER.InventoryError,
                    "flow indicators are invalid in plain workflow",
                ):
                    CHECKER._discover_workflow_launches(self.root)

    def test_flow_indicators_are_supported_in_block_plain_scalars(self) -> None:
        path = self.root / ".github/workflows/block-flow-indicator-fixture.yml"
        for run in (
            "echo [hidden]",
            "echo {hidden}",
            "echo a,b",
            "echo ${{ matrix.command }}",
        ):
            with self.subTest(run=run):
                path.write_text(
                    f"jobs:\n  fixture:\n    steps:\n      - run: {run}\n",
                    encoding="utf-8",
                )
                launch_ids = {
                    item["id"]
                    for item in CHECKER._discover_workflow_launches(self.root)
                }
                self.assertIn(
                    "workflow-launch:.github/workflows/block-flow-indicator-fixture.yml:"
                    "fixture:step-1",
                    launch_ids,
                )

    def test_block_plain_scalar_cannot_end_with_mapping_colon(self) -> None:
        path = self.root / ".github/workflows/trailing-colon-fixture.yml"
        path.write_text(
            "jobs:\n  fixture:\n    steps:\n      - run: echo:\n",
            encoding="utf-8",
        )
        with self.assertRaisesRegex(
            CHECKER.InventoryError, "invalid plain workflow .* scalar"
        ):
            CHECKER._discover_workflow_launches(self.root)

    def test_workflow_step_cannot_combine_run_and_uses(self) -> None:
        path = self.root / ".github/workflows/run-uses-fixture.yml"
        properties = (
            "      - run: echo hidden\n        uses: actions/checkout@v4\n",
            "      - uses: actions/checkout@v4\n        run: echo hidden\n",
        )
        for step in properties:
            with self.subTest(step=step):
                path.write_text(
                    "jobs:\n  fixture:\n    steps:\n" + step,
                    encoding="utf-8",
                )
                with self.assertRaisesRegex(
                    CHECKER.InventoryError, "cannot contain both run and uses"
                ):
                    CHECKER._discover_workflow_launches(self.root)

    def test_inconsistent_workflow_step_property_indent_fails_closed(self) -> None:
        path = self.root / ".github/workflows/property-indent-fixture.yml"
        path.write_text(
            "jobs:\n"
            "  fixture:\n"
            "    steps:\n"
            "      - name: visible\n"
            "       run: echo hidden\n",
            encoding="utf-8",
        )
        with self.assertRaisesRegex(
            CHECKER.InventoryError, "inconsistent workflow step property indentation"
        ):
            CHECKER._discover_workflow_launches(self.root)

    def test_invalid_generic_step_scalar_continuations_fail_closed(self) -> None:
        path = self.root / ".github/workflows/generic-continuation-fixture.yml"
        for value in ("bash", '"bash"', "{ A: B }"):
            with self.subTest(value=value):
                path.write_text(
                    "jobs:\n"
                    "  fixture:\n"
                    "    steps:\n"
                    f"      - shell: {value}\n"
                    "          run: echo hidden\n",
                    encoding="utf-8",
                )
                with self.assertRaisesRegex(
                    CHECKER.InventoryError,
                    "invalid continuation of workflow step scalar",
                ):
                    CHECKER._discover_workflow_launches(self.root)

    def test_supported_generic_step_nested_values_do_not_hide_peer_run(self) -> None:
        path = self.root / ".github/workflows/generic-nested-fixture.yml"
        path.write_text(
            "jobs:\n"
            "  fixture:\n"
            "    steps:\n"
            "      - shell: bash\n"
            "          login\n"
            "        env:\n"
            "          run: nested-value\n"
            "        run: echo visible\n",
            encoding="utf-8",
        )
        self._assert_error_contains(
            "workflow-launch:.github/workflows/generic-nested-fixture.yml:fixture:step-1"
        )

    def test_nested_jobs_mapping_does_not_create_launches(self) -> None:
        path = self.root / ".github/workflows/nested-jobs-fixture.yml"
        path.write_text(
            "metadata:\n  jobs:\n    fake:\n      steps:\n        - run: echo hidden\n",
            encoding="utf-8",
        )
        launches = CHECKER._discover_workflow_launches(self.root)
        self.assertFalse(
            any(
                item["anchor"]["file"] == path.relative_to(self.root).as_posix()
                for item in launches
            )
        )

    def test_duplicate_nested_jobs_mapping_fails_closed(self) -> None:
        path = self.root / ".github/workflows/duplicate-nested-jobs-fixture.yml"
        path.write_text(
            "metadata:\n  jobs:\n    first: one\n  jobs:\n    second: two\n",
            encoding="utf-8",
        )
        with self.assertRaisesRegex(
            CHECKER.InventoryError, "duplicate workflow mapping key 'jobs'"
        ):
            CHECKER._discover_workflow_launches(self.root)

    def test_non_string_workflow_mapping_keys_fail_closed(self) -> None:
        path = self.root / ".github/workflows/non-string-key-fixture.yml"
        path.write_text(
            "jobs:\n"
            "  fixture:\n"
            "    strategy:\n"
            "      matrix:\n"
            "        1: first\n"
            "        01: second\n"
            "    steps:\n"
            "      - run: echo hidden\n",
            encoding="utf-8",
        )
        with self.assertRaisesRegex(
            CHECKER.InventoryError, "mapping keys must be scalar strings"
        ):
            CHECKER._discover_workflow_launches(self.root)

    def test_invalid_workflow_job_scalar_continuation_fails_closed(self) -> None:
        path = self.root / ".github/workflows/job-continuation-fixture.yml"
        path.write_text(
            "jobs:\n"
            "  fixture:\n"
            "    runs-on: ubuntu-latest\n"
            "      steps:\n"
            "        - run: echo hidden\n",
            encoding="utf-8",
        )
        with self.assertRaisesRegex(
            CHECKER.InventoryError,
            "invalid continuation of workflow job scalar",
        ):
            CHECKER._discover_workflow_launches(self.root)

    def test_invalid_workflow_root_scalar_continuation_fails_closed(self) -> None:
        path = self.root / ".github/workflows/root-continuation-fixture.yml"
        path.write_text(
            "name: hello\n"
            "  jobs:\n"
            "    fixture:\n"
            "      steps:\n"
            "        - run: echo hidden\n",
            encoding="utf-8",
        )
        with self.assertRaisesRegex(
            CHECKER.InventoryError,
            "invalid continuation of workflow root scalar",
        ):
            CHECKER._discover_workflow_launches(self.root)

    def test_forbidden_yaml_control_characters_fail_closed(self) -> None:
        path = self.root / ".github/workflows/control-character-fixture.yml"
        for control in ("\x00", "\x01", "\x07"):
            with self.subTest(codepoint=ord(control)):
                path.write_text(
                    "jobs:\n"
                    "  fixture:\n"
                    "    steps:\n"
                    f"      - run: echo {control}hidden\n",
                    encoding="utf-8",
                )
                with self.assertRaisesRegex(
                    CHECKER.InventoryError, "forbidden YAML control character"
                ):
                    CHECKER._discover_workflow_launches(self.root)

    def test_terminal_newline_changes_workflow_block_digest(self) -> None:
        path = self.root / ".github/workflows/block-newline-fixture.yml"
        source = (
            "name: Block newline fixture\n"
            "jobs:\n"
            "  fixture-job:\n"
            "    runs-on: ubuntu-latest\n"
            "    steps:\n"
            "      - run: |+\n"
            "          printf payload"
        )
        launch_id = (
            "workflow-launch:.github/workflows/block-newline-fixture.yml:"
            "fixture-job:step-1"
        )
        path.write_text(source + "\n", encoding="utf-8")
        with_newline = {
            item["id"]: item["source_digest"]
            for item in CHECKER._discover_workflow_launches(self.root)
        }[launch_id]
        path.write_text(source, encoding="utf-8")
        without_newline = {
            item["id"]: item["source_digest"]
            for item in CHECKER._discover_workflow_launches(self.root)
        }[launch_id]
        self.assertNotEqual(with_newline, without_newline)

    def test_tab_after_block_content_indent_is_preserved(self) -> None:
        path = self.root / ".github/workflows/block-tab-fixture.yml"
        path.write_text(
            "name: Block tab fixture\n"
            "jobs:\n"
            "  fixture-job:\n"
            "    runs-on: ubuntu-latest\n"
            "    steps:\n"
            "      - run: |2\n"
            "          \tpayload\n",
            encoding="utf-8",
        )
        launch_ids = {
            item["id"] for item in CHECKER._discover_workflow_launches(self.root)
        }
        self.assertIn(
            "workflow-launch:.github/workflows/block-tab-fixture.yml:fixture-job:step-1",
            launch_ids,
        )

    def test_overindented_leading_block_blank_fails_closed(self) -> None:
        path = self.root / ".github/workflows/block-blank-indent-fixture.yml"
        path.write_text(
            "name: Block blank indent fixture\n"
            "jobs:\n"
            "  fixture-job:\n"
            "    runs-on: ubuntu-latest\n"
            "    steps:\n"
            "      - run: |\n"
            "            \n"
            "          payload\n",
            encoding="utf-8",
        )
        with self.assertRaisesRegex(
            CHECKER.InventoryError,
            "inconsistent workflow block scalar indentation",
        ):
            CHECKER._discover_workflow_launches(self.root)

    def test_workflow_flow_scalar_preserves_quoted_hash(self) -> None:
        path = self.root / ".github/workflows/quoted-hash-fixture.yml"
        path.write_text(
            "name: Quoted hash fixture\n"
            "jobs:\n"
            "  fixture-job:\n"
            "    runs-on: ubuntu-latest\n"
            "    steps:\n"
            "      - { name: Hash launch, run: 'printf #payload' } # comment\n",
            encoding="utf-8",
        )
        self._assert_error_contains(
            "workflow-launch:.github/workflows/quoted-hash-fixture.yml:fixture-job:hash-launch"
        )

    def test_escaped_workflow_run_key_fails_closed(self) -> None:
        path = self.root / ".github/workflows/ci.yml"
        text = path.read_text(encoding="utf-8")
        insertion = (
            '\n      - name: Escaped run fixture\n        "\\\\x72un": python3 -V\n'
        )
        path.write_text(
            text.replace("\n  target-tests:", insertion + "\n  target-tests:", 1),
            encoding="utf-8",
        )
        errors = "\n".join(CHECKER.validate(self.root, self.inventory_path))
        self.assertIn("source discovery failed closed", errors)
        self.assertIn("escaped workflow mapping keys are unsupported", errors)

    def test_escaped_flow_workflow_run_key_fails_closed(self) -> None:
        path = self.root / ".github/workflows/ci.yml"
        text = path.read_text(encoding="utf-8")
        insertion = (
            '\n      - { name: Escaped flow run fixture, "\\\\u0072un": python3 -V }\n'
        )
        path.write_text(
            text.replace("\n  target-tests:", insertion + "\n  target-tests:", 1),
            encoding="utf-8",
        )
        errors = "\n".join(CHECKER.validate(self.root, self.inventory_path))
        self.assertIn("source discovery failed closed", errors)
        self.assertIn("escaped workflow mapping keys are unsupported", errors)

    def test_class_namespace_process_alias_fails_closed(self) -> None:
        path = self.root / "tools/check_abi_manifest.py"
        with path.open("a", encoding="utf-8") as output:
            output.write(
                "\nclass FixtureRunner:\n"
                "    launch = subprocess.run\n\n"
                "def fixture_class_launch():\n"
                "    return FixtureRunner.launch(['python3', '-V'])\n"
            )
        errors = "\n".join(CHECKER.validate(self.root, self.inventory_path))
        self.assertIn("source discovery failed closed", errors)
        self.assertIn("process aliases in class namespaces are unsupported", errors)

    def test_class_shadow_controls_method_default_resolution(self) -> None:
        path = self.root / "tools/check_abi_manifest.py"
        with path.open("a", encoding="utf-8") as output:
            output.write(
                "\nclass FakeProcessFixture:\n"
                "    @staticmethod\n"
                "    def run(arguments):\n"
                "        return arguments\n\n"
                "class ShadowRunner:\n"
                "    subprocess = FakeProcessFixture()\n\n"
                "    @staticmethod\n"
                "    def hidden_shadow(command=subprocess.run):\n"
                "        return command(['python3', '-V'])\n"
            )
        self.assertNotIn(
            "hidden_shadow",
            "\n".join(
                item["id"] for item in CHECKER._discover_python_launches(self.root)
            ),
        )

    def test_match_process_alias_capture_fails_closed(self) -> None:
        path = self.root / "tools/check_abi_manifest.py"
        with path.open("a", encoding="utf-8") as output:
            output.write(
                "\nclass MatchRunner:\n"
                "    match subprocess.run:\n"
                "        case launch:\n"
                "            pass\n\n"
                "def fixture_match_launch():\n"
                "    return MatchRunner.launch(['python3', '-V'])\n"
            )
        errors = "\n".join(CHECKER.validate(self.root, self.inventory_path))
        self.assertIn("source discovery failed closed", errors)
        self.assertIn(
            "process aliases in match capture bindings are unsupported", errors
        )

    def test_repository_local_process_alias_in_class_fails_closed(self) -> None:
        source = self.root / "tools/fixture_process_source.py"
        source.write_text(
            "import subprocess\nlaunch = subprocess.run\n", encoding="utf-8"
        )
        consumer = self.root / "tools/fixture_process_consumer.py"
        consumer.write_text(
            "from tools.fixture_process_source import launch\n\n"
            "class Runner:\n"
            "    run = launch\n\n"
            "def fixture_local_alias_launch():\n"
            "    return Runner.run(['python3', '-V'])\n",
            encoding="utf-8",
        )
        errors = "\n".join(CHECKER.validate(self.root, self.inventory_path))
        self.assertIn("source discovery failed closed", errors)
        self.assertIn("process aliases in class namespaces are unsupported", errors)

    def test_repository_local_imported_process_alias_discovers_launch(self) -> None:
        source = self.root / "tools/fixture_process_source.py"
        source.write_text(
            "import subprocess\nlaunch = subprocess.run\n", encoding="utf-8"
        )
        consumer = self.root / "tools/fixture_process_consumer.py"
        consumer.write_text(
            "from tools.fixture_process_source import launch\n\n"
            "def fixture_local_alias_launch():\n"
            "    return launch(['python3', '-V'])\n",
            encoding="utf-8",
        )
        self._assert_error_contains(
            "python-launch:tools/fixture_process_consumer.py:fixture_local_alias_launch:subprocess.run:1"
        )

    def test_repository_local_module_alias_discovers_launch(self) -> None:
        source = self.root / "tools/fixture_process_source.py"
        source.write_text(
            "import subprocess\nlaunch = subprocess.run\n", encoding="utf-8"
        )
        consumer = self.root / "tools/fixture_process_consumer.py"
        consumer.write_text(
            "import tools.fixture_process_source as process_source\n\n"
            "def fixture_local_module_launch():\n"
            "    return process_source.launch(['python3', '-V'])\n",
            encoding="utf-8",
        )
        self._assert_error_contains(
            "python-launch:tools/fixture_process_consumer.py:fixture_local_module_launch:subprocess.run:1"
        )

    def test_relative_package_submodule_process_alias_fails_closed(self) -> None:
        package = self.root / "fixturepkg"
        package.mkdir()
        (package / "__init__.py").write_text("", encoding="utf-8")
        (package / "source.py").write_text(
            "import subprocess\nlaunch = subprocess.run\n", encoding="utf-8"
        )
        (package / "consumer.py").write_text(
            "from . import source\n\n"
            "class Runner:\n"
            "    launch = source.launch\n\n"
            "def hidden():\n"
            "    return Runner.launch(['python3', '-V'])\n",
            encoding="utf-8",
        )
        errors = "\n".join(CHECKER.validate(self.root, self.inventory_path))
        self.assertIn("source discovery failed closed", errors)
        self.assertIn("process aliases in class namespaces are unsupported", errors)

    def test_absolute_package_submodule_process_alias_fails_closed(self) -> None:
        package = self.root / "fixturepkg"
        package.mkdir()
        (package / "__init__.py").write_text("", encoding="utf-8")
        (package / "source.py").write_text(
            "import subprocess\nlaunch = subprocess.run\n", encoding="utf-8"
        )
        (package / "consumer.py").write_text(
            "from fixturepkg import source\n\n"
            "class Runner:\n"
            "    launch = source.launch\n\n"
            "def hidden():\n"
            "    return Runner.launch(['python3', '-V'])\n",
            encoding="utf-8",
        )
        errors = "\n".join(CHECKER.validate(self.root, self.inventory_path))
        self.assertIn("source discovery failed closed", errors)
        self.assertIn("process aliases in class namespaces are unsupported", errors)

    def test_literal_dynamic_process_import_fails_closed_in_class(self) -> None:
        path = self.root / "tools/check_abi_manifest.py"
        with path.open("a", encoding="utf-8") as output:
            output.write(
                "\nimport importlib\n"
                "proc_fixture = importlib.import_module('subprocess')\n\n"
                "class DynamicRunner:\n"
                "    launch = proc_fixture.run\n\n"
                "def hidden_dynamic():\n"
                "    return DynamicRunner.launch(['python3', '-V'])\n"
            )
        errors = "\n".join(CHECKER.validate(self.root, self.inventory_path))
        self.assertIn("source discovery failed closed", errors)
        self.assertIn("process aliases in class namespaces are unsupported", errors)

    def test_builtin_literal_dynamic_process_import_fails_closed_in_class(
        self,
    ) -> None:
        path = self.root / "tools/check_abi_manifest.py"
        with path.open("a", encoding="utf-8") as output:
            output.write(
                "\nproc_fixture = __import__('subprocess')\n\n"
                "class DynamicRunner:\n"
                "    launch = proc_fixture.run\n\n"
                "def hidden_dynamic():\n"
                "    return DynamicRunner.launch(['python3', '-V'])\n"
            )
        errors = "\n".join(CHECKER.validate(self.root, self.inventory_path))
        self.assertIn("source discovery failed closed", errors)
        self.assertIn("process aliases in class namespaces are unsupported", errors)

    def test_nonliteral_dynamic_module_import_fails_closed(self) -> None:
        path = self.root / "tools/check_abi_manifest.py"
        with path.open("a", encoding="utf-8") as output:
            output.write(
                "\nimport importlib\n"
                "module_name = 'subprocess'\n"
                "proc_fixture = importlib.import_module(module_name)\n"
            )
        errors = "\n".join(CHECKER.validate(self.root, self.inventory_path))
        self.assertIn("source discovery failed closed", errors)
        self.assertIn("dynamic module imports require a literal module name", errors)

    def test_direct_importlib_process_call_is_discovered(self) -> None:
        path = self.root / "tools/check_abi_manifest.py"
        with path.open("a", encoding="utf-8") as output:
            output.write(
                "\nimport importlib\n\n"
                "def direct_dynamic_launch():\n"
                "    return importlib.import_module('subprocess').run(['python3', '-V'])\n"
            )
        self._assert_error_contains(
            "python-launch:tools/check_abi_manifest.py:direct_dynamic_launch:subprocess.run:1"
        )

    def test_direct_importlib_alias_process_call_is_discovered(self) -> None:
        path = self.root / "tools/check_abi_manifest.py"
        with path.open("a", encoding="utf-8") as output:
            output.write(
                "\nfrom importlib import import_module as load_module\n\n"
                "def direct_dynamic_alias_launch():\n"
                "    return load_module('subprocess').run(['python3', '-V'])\n"
            )
        self._assert_error_contains(
            "python-launch:tools/check_abi_manifest.py:direct_dynamic_alias_launch:subprocess.run:1"
        )

    def test_direct_builtin_dynamic_process_call_is_discovered(self) -> None:
        path = self.root / "tools/check_abi_manifest.py"
        with path.open("a", encoding="utf-8") as output:
            output.write(
                "\ndef direct_builtin_dynamic_launch():\n"
                "    return __import__('subprocess').run(['python3', '-V'])\n"
            )
        self._assert_error_contains(
            "python-launch:tools/check_abi_manifest.py:direct_builtin_dynamic_launch:subprocess.run:1"
        )

    def test_assigned_importlib_helper_process_call_is_discovered(self) -> None:
        path = self.root / "tools/check_abi_manifest.py"
        with path.open("a", encoding="utf-8") as output:
            output.write(
                "\nimport importlib\n"
                "load_module = importlib.import_module\n\n"
                "def assigned_dynamic_launch():\n"
                "    return load_module('subprocess').run(['python3', '-V'])\n"
            )
        self._assert_error_contains(
            "python-launch:tools/check_abi_manifest.py:assigned_dynamic_launch:subprocess.run:1"
        )

    def test_assigned_builtin_import_helper_process_call_is_discovered(self) -> None:
        path = self.root / "tools/check_abi_manifest.py"
        with path.open("a", encoding="utf-8") as output:
            output.write(
                "\nload_module = __import__\n\n"
                "def assigned_builtin_dynamic_launch():\n"
                "    return load_module('subprocess').run(['python3', '-V'])\n"
            )
        self._assert_error_contains(
            "python-launch:tools/check_abi_manifest.py:assigned_builtin_dynamic_launch:subprocess.run:1"
        )

    def test_defaulted_import_helper_process_call_is_discovered(self) -> None:
        path = self.root / "tools/check_abi_manifest.py"
        with path.open("a", encoding="utf-8") as output:
            output.write(
                "\nimport importlib\n\n"
                "def defaulted_dynamic_launch(load_module=importlib.import_module):\n"
                "    return load_module('subprocess').run(['python3', '-V'])\n"
            )
        self._assert_error_contains(
            "python-launch:tools/check_abi_manifest.py:defaulted_dynamic_launch:subprocess.run:1"
        )

    def test_repository_local_import_helper_process_call_is_discovered(self) -> None:
        source = self.root / "tools/fixture_import_helper_source.py"
        source.write_text(
            "from importlib import import_module\nload_module = import_module\n",
            encoding="utf-8",
        )
        consumer = self.root / "tools/fixture_import_helper_consumer.py"
        consumer.write_text(
            "from tools.fixture_import_helper_source import load_module\n\n"
            "def local_helper_dynamic_launch():\n"
            "    return load_module('subprocess').run(['python3', '-V'])\n",
            encoding="utf-8",
        )
        self._assert_error_contains(
            "python-launch:tools/fixture_import_helper_consumer.py:local_helper_dynamic_launch:subprocess.run:1"
        )

    def test_assigned_importlib_namespace_process_call_is_discovered(self) -> None:
        path = self.root / "tools/check_abi_manifest.py"
        with path.open("a", encoding="utf-8") as output:
            output.write(
                "\nimport importlib\n"
                "loader = importlib\n\n"
                "def assigned_namespace_launch():\n"
                "    return loader.import_module('subprocess').run(['python3', '-V'])\n"
            )
        self._assert_error_contains(
            "python-launch:tools/check_abi_manifest.py:assigned_namespace_launch:subprocess.run:1"
        )

    def test_defaulted_importlib_namespace_process_call_is_discovered(self) -> None:
        path = self.root / "tools/check_abi_manifest.py"
        with path.open("a", encoding="utf-8") as output:
            output.write(
                "\nimport importlib\n\n"
                "def defaulted_namespace_launch(loader=importlib):\n"
                "    return loader.import_module('subprocess').run(['python3', '-V'])\n"
            )
        self._assert_error_contains(
            "python-launch:tools/check_abi_manifest.py:defaulted_namespace_launch:subprocess.run:1"
        )

    def test_repository_local_importlib_namespace_call_is_discovered(self) -> None:
        source = self.root / "tools/fixture_import_namespace_source.py"
        source.write_text("import importlib\n", encoding="utf-8")
        consumer = self.root / "tools/fixture_import_namespace_consumer.py"
        consumer.write_text(
            "from tools.fixture_import_namespace_source import importlib as loader\n\n"
            "def local_namespace_launch():\n"
            "    return loader.import_module('subprocess').run(['python3', '-V'])\n",
            encoding="utf-8",
        )
        self._assert_error_contains(
            "python-launch:tools/fixture_import_namespace_consumer.py:local_namespace_launch:subprocess.run:1"
        )

    def test_rebinding_module_alias_invalidates_qualified_process_facts(
        self,
    ) -> None:
        source = self.root / "tools/fixture_process_source.py"
        source.write_text(
            "import subprocess\nlaunch = subprocess.run\n", encoding="utf-8"
        )
        consumer = self.root / "tools/fixture_process_consumer.py"
        consumer.write_text(
            "import tools.fixture_process_source as process_source\n\n"
            "class Fake:\n"
            "    @staticmethod\n"
            "    def launch(arguments):\n"
            "        return arguments\n\n"
            "process_source = Fake()\n\n"
            "def hidden():\n"
            "    return process_source.launch(['python3', '-V'])\n",
            encoding="utf-8",
        )
        launches = CHECKER._discover_python_launches(self.root)
        self.assertFalse(
            any(
                item["anchor"]["file"] == consumer.relative_to(self.root).as_posix()
                for item in launches
            )
        )

    def test_class_rebinding_invalidates_qualified_default_process_facts(
        self,
    ) -> None:
        source = self.root / "tools/fixture_process_source.py"
        source.write_text(
            "import subprocess\nlaunch = subprocess.run\n", encoding="utf-8"
        )
        consumer = self.root / "tools/fixture_process_consumer.py"
        consumer.write_text(
            "import tools.fixture_process_source as process_source\n\n"
            "class Fake:\n"
            "    @staticmethod\n"
            "    def launch(arguments):\n"
            "        return arguments\n\n"
            "class ShadowRunner:\n"
            "    process_source = Fake()\n\n"
            "    @staticmethod\n"
            "    def hidden(command=process_source.launch):\n"
            "        return command(['python3', '-V'])\n",
            encoding="utf-8",
        )
        launches = CHECKER._discover_python_launches(self.root)
        self.assertFalse(
            any(
                item["anchor"]["file"] == consumer.relative_to(self.root).as_posix()
                for item in launches
            )
        )

    def test_extended_process_launch_apis_are_discovered(self) -> None:
        path = self.root / "tools/check_abi_manifest.py"
        with path.open("a", encoding="utf-8") as output:
            output.write(
                "\nimport asyncio\n"
                "import os\n\n"
                "def extended_process_api_fixture():\n"
                "    os.posix_spawn('/bin/true', ['/bin/true'], {})\n"
                "    subprocess.getoutput('true')\n"
                "    asyncio.get_running_loop().subprocess_exec(None, '/bin/true')\n"
            )
        errors = "\n".join(CHECKER.validate(self.root, self.inventory_path))
        for call in (
            "os.posix_spawn",
            "subprocess.getoutput",
            "asyncio.loop.subprocess_exec",
        ):
            with self.subTest(call=call):
                self.assertIn(f":extended_process_api_fixture:{call}:1", errors)

    def test_event_loop_factory_aliases_are_discovered(self) -> None:
        path = self.root / "tools/fixture_event_loop_alias.py"
        path.write_text(
            "import asyncio\n\n"
            "async def assigned_loop():\n"
            "    loop = asyncio.get_running_loop()\n"
            "    return await loop.subprocess_exec(None, '/bin/true')\n\n"
            "async def defaulted_loop(loop=asyncio.get_event_loop()):\n"
            "    return await loop.subprocess_shell(None, 'true')\n",
            encoding="utf-8",
        )
        errors = "\n".join(CHECKER.validate(self.root, self.inventory_path))
        self.assertIn(":assigned_loop:asyncio.loop.subprocess_exec:1", errors)
        self.assertIn(":defaulted_loop:asyncio.loop.subprocess_shell:1", errors)

    def test_nested_process_callable_attributes_are_discovered(self) -> None:
        path = self.root / "tools/fixture_nested_callable.py"
        path.write_text(
            "import subprocess\n\n"
            "def nested_callable():\n"
            "    return subprocess.run.__call__.__call__(['python3', '-V'])\n",
            encoding="utf-8",
        )
        self._assert_error_contains(
            "python-launch:tools/fixture_nested_callable.py:nested_callable:subprocess.run:1"
        )

    def test_process_module_dict_callable_is_discovered(self) -> None:
        path = self.root / "tools/fixture_process_subscript.py"
        path.write_text(
            "import subprocess\n"
            "runner = subprocess.__dict__['run']\n"
            "runner(['python3', '-V'])\n",
            encoding="utf-8",
        )
        self._assert_error_contains(
            "python-launch:tools/fixture_process_subscript.py:module:subprocess.run:1"
        )

    def test_defaulted_process_module_dict_callable_is_discovered(self) -> None:
        path = self.root / "tools/fixture_process_subscript_default.py"
        path.write_text(
            "import subprocess\n\n"
            "def launch(runner=subprocess.__dict__['run']):\n"
            "    return runner(['python3', '-V'])\n",
            encoding="utf-8",
        )
        self._assert_error_contains(
            "python-launch:tools/fixture_process_subscript_default.py:launch:subprocess.run:1"
        )

    def test_process_module_dict_mutation_fails_closed(self) -> None:
        path = self.root / "tools/fixture_process_subscript_mutation.py"
        path.write_text(
            "import subprocess\nsubprocess.__dict__['run'] = lambda command: command\n",
            encoding="utf-8",
        )
        errors = "\n".join(CHECKER.validate(self.root, self.inventory_path))
        self.assertIn("source discovery failed closed", errors)
        self.assertIn("mutating tracked process alias subscripts", errors)

    def test_globals_process_module_lookup_is_discovered(self) -> None:
        path = self.root / "tools/fixture_process_globals.py"
        path.write_text(
            "import subprocess\n\n"
            "def launch():\n"
            "    return globals()['subprocess'].run(['python3', '-V'])\n",
            encoding="utf-8",
        )
        self._assert_error_contains(
            "python-launch:tools/fixture_process_globals.py:launch:subprocess.run:1"
        )

    def test_structured_process_default_fails_closed(self) -> None:
        path = self.root / "tools/fixture_structured_default.py"
        path.write_text(
            "import subprocess\n\n"
            "def structured_default(runners=(subprocess.run,)):\n"
            "    return runners[0](['python3', '-V'])\n",
            encoding="utf-8",
        )
        errors = "\n".join(CHECKER.validate(self.root, self.inventory_path))
        self.assertIn("source discovery failed closed", errors)
        self.assertIn(
            "structured process aliases in parameter defaults are unsupported", errors
        )

    def test_builtins_namespace_dynamic_import_is_discovered(self) -> None:
        path = self.root / "tools/fixture_builtins_import.py"
        path.write_text(
            "import builtins\n\n"
            "def builtins_import():\n"
            "    return builtins.__import__('subprocess').run(['python3', '-V'])\n",
            encoding="utf-8",
        )
        self._assert_error_contains(
            "python-launch:tools/fixture_builtins_import.py:builtins_import:subprocess.run:1"
        )

    def test_process_launching_subclass_is_discovered(self) -> None:
        path = self.root / "tools/fixture_process_subclass.py"
        path.write_text(
            "import subprocess\n\n"
            "class Child(subprocess.Popen):\n"
            "    pass\n\n"
            "def subclass_launch():\n"
            "    return Child(['python3', '-V'])\n",
            encoding="utf-8",
        )
        self._assert_error_contains(
            "python-launch:tools/fixture_process_subclass.py:subclass_launch:subprocess.Popen:1"
        )

    def test_conditional_process_launching_subclass_fails_closed(self) -> None:
        path = self.root / "tools/fixture_conditional_process_subclass.py"
        path.write_text(
            "import subprocess\n\n"
            "class Child(subprocess.Popen if __debug__ else object):\n"
            "    pass\n",
            encoding="utf-8",
        )
        errors = "\n".join(CHECKER.validate(self.root, self.inventory_path))
        self.assertIn("source discovery failed closed", errors)
        self.assertIn("class bases must be statically resolvable", errors)

    def test_process_pool_launch_apis_are_discovered(self) -> None:
        path = self.root / "tools/fixture_process_pools.py"
        path.write_text(
            "import multiprocessing.pool\n"
            "from concurrent.futures import ProcessPoolExecutor\n\n"
            "def pool_launches():\n"
            "    multiprocessing.pool.Pool()\n"
            "    executor = ProcessPoolExecutor()\n"
            "    executor.submit(abs, -1)\n"
            "    executor.map(abs, [-1])\n",
            encoding="utf-8",
        )
        errors = "\n".join(CHECKER.validate(self.root, self.inventory_path))
        for call in (
            "multiprocessing.pool.Pool",
            "concurrent.futures.process_pool.submit",
            "concurrent.futures.process_pool.map",
        ):
            with self.subTest(call=call):
                self.assertIn(f":pool_launches:{call}:1", errors)

    def test_manager_and_process_pool_submodule_imports_are_discovered(self) -> None:
        path = self.root / "tools/fixture_process_submodule_imports.py"
        path.write_text(
            "import multiprocessing\n"
            "from multiprocessing.pool import Pool\n"
            "from multiprocessing.managers import SyncManager\n"
            "from concurrent.futures.process import ProcessPoolExecutor\n\n"
            "def launches():\n"
            "    multiprocessing.Manager()\n"
            "    Pool()\n"
            "    manager = SyncManager()\n"
            "    manager.start()\n"
            "    executor = ProcessPoolExecutor()\n"
            "    executor.submit(abs, -1)\n",
            encoding="utf-8",
        )
        errors = "\n".join(CHECKER.validate(self.root, self.inventory_path))
        for call in (
            "multiprocessing.Manager",
            "multiprocessing.pool.Pool",
            "multiprocessing.manager.start",
            "concurrent.futures.process_pool.submit",
        ):
            with self.subTest(call=call):
                self.assertIn(f":launches:{call}:1", errors)

    def test_public_process_submodule_entries_are_discovered(self) -> None:
        path = self.root / "tools/fixture_public_process_submodules.py"
        path.write_text(
            "import asyncio.subprocess\n"
            "from asyncio.subprocess import create_subprocess_exec\n"
            "from multiprocessing.context import Process\n\n"
            "async def asyncio_launches():\n"
            "    await create_subprocess_exec('/bin/true')\n"
            "    await asyncio.subprocess.create_subprocess_exec('/bin/true')\n\n"
            "def multiprocessing_launch():\n"
            "    process = Process(target=print)\n"
            "    process.start()\n",
            encoding="utf-8",
        )
        errors = "\n".join(CHECKER.validate(self.root, self.inventory_path))
        self.assertIn(":asyncio_launches:asyncio.create_subprocess_exec:1", errors)
        self.assertIn(":asyncio_launches:asyncio.create_subprocess_exec:2", errors)
        self.assertIn(":multiprocessing_launch:multiprocessing.process.start:1", errors)

    def test_asyncio_event_loop_factory_entries_are_discovered(self) -> None:
        path = self.root / "tools/fixture_asyncio_loop_factories.py"
        path.write_text(
            "import asyncio\n\n"
            "async def launches():\n"
            "    new_loop = asyncio.new_event_loop()\n"
            "    new_loop.subprocess_exec(asyncio.SubprocessProtocol, '/bin/true')\n"
            "    policy_loop = asyncio.get_event_loop_policy().get_event_loop()\n"
            "    policy_loop.subprocess_exec(asyncio.SubprocessProtocol, '/bin/true')\n"
            "    runner_loop = asyncio.Runner().get_loop()\n"
            "    runner_loop.subprocess_exec(asyncio.SubprocessProtocol, '/bin/true')\n",
            encoding="utf-8",
        )
        errors = "\n".join(CHECKER.validate(self.root, self.inventory_path))
        for ordinal in range(1, 4):
            with self.subTest(ordinal=ordinal):
                self.assertIn(
                    f":launches:asyncio.loop.subprocess_exec:{ordinal}", errors
                )

    def test_concrete_asyncio_event_loop_entry_is_discovered(self) -> None:
        path = self.root / "tools/fixture_concrete_asyncio_loop.py"
        path.write_text(
            "import asyncio\n\n"
            "async def launch():\n"
            "    loop = asyncio.SelectorEventLoop()\n"
            "    loop.subprocess_exec(asyncio.SubprocessProtocol, '/bin/true')\n",
            encoding="utf-8",
        )
        errors = "\n".join(CHECKER.validate(self.root, self.inventory_path))
        self.assertIn(":launch:asyncio.loop.subprocess_exec:1", errors)

        path.write_text(
            "import asyncio\n\n"
            "async def launch():\n"
            "    loop = asyncio.EventLoop()\n"
            "    loop.subprocess_exec(asyncio.SubprocessProtocol, '/bin/true')\n",
            encoding="utf-8",
        )
        errors = "\n".join(CHECKER.validate(self.root, self.inventory_path))
        self.assertIn(":launch:asyncio.loop.subprocess_exec:1", errors)

    def test_concrete_multiprocessing_context_entries_are_discovered(self) -> None:
        path = self.root / "tools/fixture_concrete_process_contexts.py"
        path.write_text(
            "from multiprocessing.context import SpawnContext, SpawnProcess\n\n"
            "def launches():\n"
            "    direct = SpawnProcess(target=print)\n"
            "    direct.start()\n"
            "    contextual = SpawnContext().Process(target=print)\n"
            "    contextual.start()\n",
            encoding="utf-8",
        )
        errors = "\n".join(CHECKER.validate(self.root, self.inventory_path))
        self.assertIn(":launches:multiprocessing.process.start:1", errors)
        self.assertIn(":launches:multiprocessing.process.start:2", errors)

    def test_shared_memory_manager_start_is_discovered(self) -> None:
        path = self.root / "tools/fixture_shared_memory_manager.py"
        path.write_text(
            "from multiprocessing.managers import SharedMemoryManager\n\n"
            "def launch():\n"
            "    manager = SharedMemoryManager()\n"
            "    manager.start()\n",
            encoding="utf-8",
        )
        errors = "\n".join(CHECKER.validate(self.root, self.inventory_path))
        self.assertIn(":launch:multiprocessing.manager.start:1", errors)

    def test_process_namespace_lookup_methods_are_discovered(self) -> None:
        path = self.root / "tools/fixture_process_namespace_methods.py"
        path.write_text(
            "import subprocess\n\n"
            "def launches():\n"
            "    subprocess.__dict__.get('run', print)(['/bin/true'])\n"
            "    subprocess.__getattribute__('run')(['/bin/true'])\n"
            "    globals().get('subprocess', None).run(['/bin/true'])\n"
            "    globals().__getitem__('subprocess').run(['/bin/true'])\n"
            "    namespace = globals()\n"
            "    namespace['subprocess'].run(['/bin/true'])\n"
            "    get_namespace = globals\n"
            "    get_namespace().get('subprocess').run(['/bin/true'])\n"
            "    from builtins import globals as builtin_globals\n"
            "    builtin_globals()['subprocess'].run(['/bin/true'])\n"
            "    import builtins\n"
            "    builtins.globals()['subprocess'].run(['/bin/true'])\n"
            "    locals()['subprocess'].run(['/bin/true'])\n"
            "    vars()['subprocess'].run(['/bin/true'])\n"
            "    globals().__getattribute__('get')('subprocess').run(['/bin/true'])\n"
            "    get_run = subprocess.__dict__.get\n"
            "    get_run('run')(['/bin/true'])\n",
            encoding="utf-8",
        )
        errors = "\n".join(CHECKER.validate(self.root, self.inventory_path))
        for ordinal in range(1, 13):
            with self.subTest(ordinal=ordinal):
                self.assertIn(f":launches:subprocess.run:{ordinal}", errors)

    def test_process_namespace_update_fails_closed(self) -> None:
        path = self.root / "tools/fixture_process_namespace_mutation.py"
        path.write_text(
            "import subprocess\n"
            "update = subprocess.__dict__.update\n"
            "update(run=lambda command: command)\n"
            "subprocess.run(['/bin/true'])\n",
            encoding="utf-8",
        )
        errors = "\n".join(CHECKER.validate(self.root, self.inventory_path))
        self.assertIn("source discovery failed closed", errors)
        self.assertIn("mutating tracked process namespaces", errors)

    def test_globals_namespace_mutation_fails_closed(self) -> None:
        path = self.root / "tools/fixture_globals_namespace_mutation.py"
        path.write_text(
            "import subprocess\n"
            "globals().__setitem__('subprocess', object())\n"
            "subprocess.run(['/bin/true'])\n",
            encoding="utf-8",
        )
        errors = "\n".join(CHECKER.validate(self.root, self.inventory_path))
        self.assertIn("source discovery failed closed", errors)
        self.assertIn("mutating tracked process namespaces", errors)

    def test_process_namespace_accessor_arity_is_checked(self) -> None:
        path = self.root / "tools/fixture_process_namespace_bad_arity.py"
        path.write_text(
            "import subprocess\n"
            "subprocess.__getattribute__('run', print)(['/bin/true'])\n",
            encoding="utf-8",
        )
        errors = "\n".join(CHECKER.validate(self.root, self.inventory_path))
        self.assertIn("source discovery failed closed", errors)
        self.assertNotIn(":subprocess.run:", errors)

    def test_dynamic_process_namespace_lookup_fails_closed(self) -> None:
        path = self.root / "tools/fixture_dynamic_process_namespace.py"
        path.write_text(
            "import subprocess\n"
            "key = 'subprocess'\n"
            "globals()[key].run(['/bin/true'])\n"
            "globals().get(key).run(['/bin/true'])\n",
            encoding="utf-8",
        )
        errors = "\n".join(CHECKER.validate(self.root, self.inventory_path))
        self.assertIn("source discovery failed closed", errors)
        self.assertIn("dynamic process namespace lookups", errors)

    def test_globals_getattribute_is_not_treated_as_mapping_lookup(self) -> None:
        path = self.root / "tools/fixture_globals_getattribute.py"
        path.write_text(
            "import subprocess\n"
            "globals().__getattribute__('subprocess').run(['/bin/true'])\n",
            encoding="utf-8",
        )
        launches = CHECKER._discover_python_launches(self.root)
        self.assertFalse(
            any(
                item["anchor"]["file"] == path.relative_to(self.root).as_posix()
                for item in launches
            )
        )

    def test_dunder_builtins_process_entries_are_discovered(self) -> None:
        path = self.root / "tools/fixture_dunder_builtins.py"
        path.write_text(
            "def launches():\n"
            "    __builtins__.__import__('subprocess').run(['/bin/true'])\n"
            "    __builtins__.globals()['subprocess'].run(['/bin/true'])\n"
            "    __builtins__['__import__']('subprocess').run(['/bin/true'])\n"
            "    __builtins__['globals']()['subprocess'].run(['/bin/true'])\n",
            encoding="utf-8",
        )
        errors = "\n".join(CHECKER.validate(self.root, self.inventory_path))
        for ordinal in range(1, 5):
            with self.subTest(ordinal=ordinal):
                self.assertIn(f":launches:subprocess.run:{ordinal}", errors)

    def test_imported_builtins_dictionary_process_entry_is_discovered(self) -> None:
        path = self.root / "tools/fixture_builtins_dictionary.py"
        path.write_text(
            "from builtins import __dict__ as namespace\n\n"
            "def launch():\n"
            "    namespace['__import__']('subprocess').run(['/bin/true'])\n",
            encoding="utf-8",
        )
        errors = "\n".join(CHECKER.validate(self.root, self.inventory_path))
        self.assertIn(":launch:subprocess.run:1", errors)

    def test_dunder_builtins_dictionary_transport_is_discovered(self) -> None:
        path = self.root / "tools/fixture_dunder_builtins_dictionary.py"
        path.write_text(
            "def launch():\n"
            "    namespace = __builtins__.__dict__\n"
            "    namespace['__import__']('subprocess').run(['/bin/true'])\n",
            encoding="utf-8",
        )
        errors = "\n".join(CHECKER.validate(self.root, self.inventory_path))
        self.assertIn(":launch:subprocess.run:1", errors)

    def test_module_dictionary_getitem_process_entries_are_discovered(self) -> None:
        path = self.root / "tools/fixture_module_dictionary_getitem.py"
        path.write_text(
            "import builtins\n"
            "import subprocess\n\n"
            "def launches():\n"
            "    subprocess.__dict__.__getitem__('run')(['/bin/true'])\n"
            "    builtins.__dict__.__getitem__('__import__')('subprocess').run(\n"
            "        ['/bin/true']\n"
            "    )\n",
            encoding="utf-8",
        )
        errors = "\n".join(CHECKER.validate(self.root, self.inventory_path))
        self.assertIn(":launches:subprocess.run:1", errors)
        self.assertIn(":launches:subprocess.run:2", errors)

    def test_dunder_builtins_accessors_are_discovered(self) -> None:
        path = self.root / "tools/fixture_dunder_builtins_accessors.py"
        path.write_text(
            "def launches():\n"
            "    __builtins__.get('__import__')('subprocess').run(['/bin/true'])\n"
            "    __builtins__.__getattribute__('__import__')(\n"
            "        'subprocess'\n"
            "    ).run(['/bin/true'])\n",
            encoding="utf-8",
        )
        errors = "\n".join(CHECKER.validate(self.root, self.inventory_path))
        self.assertIn(":launches:subprocess.run:1", errors)
        self.assertIn(":launches:subprocess.run:2", errors)

    def test_unmodeled_namespace_copy_fails_closed(self) -> None:
        path = self.root / "tools/fixture_namespace_copy.py"
        for expression in (
            "subprocess.__dict__.copy()['run'](['/bin/true'])",
            "globals().copy()['subprocess'].run(['/bin/true'])",
        ):
            with self.subTest(expression=expression):
                path.write_text(f"import subprocess\n{expression}\n", encoding="utf-8")
                errors = "\n".join(CHECKER.validate(self.root, self.inventory_path))
                self.assertIn("source discovery failed closed", errors)
                self.assertIn("unsupported tracked process namespace", errors)

    def test_sys_modules_process_entry_is_discovered(self) -> None:
        path = self.root / "tools/fixture_sys_modules.py"
        path.write_text(
            "import subprocess\n"
            "import sys\n\n"
            "def launch():\n"
            "    sys.modules['subprocess'].run(['/bin/true'])\n",
            encoding="utf-8",
        )
        errors = "\n".join(CHECKER.validate(self.root, self.inventory_path))
        self.assertIn(":launch:subprocess.run:1", errors)

    def test_dynamic_sys_modules_lookup_fails_closed(self) -> None:
        canonical_root = CHECKER._canonical_root(self.root)
        reviewed_path = canonical_root / "tools/check_test_inventory.py"
        reviewed_source = (
            "import sys\n\n"
            "def _verify_python_source_module_binding(\n"
            "    binding,\n"
            ") -> None:\n"
            "    reviewed = binding.reviewed\n"
            "    module = binding.module\n"
            "    if (\n"
            "        module is None\n"
            "        or sys.modules.get(binding.name) is not module\n"
            "        or reviewed is None\n"
            "    ):\n"
            "        raise RuntimeError\n"
        )
        reviewed_expression = "sys.modules.get(binding.name) is not module"
        self.assertEqual(1, reviewed_source.count(reviewed_expression))

        def discover_source(source: str, source_path: Path = reviewed_path) -> None:
            CHECKER._discover_python_launches(
                self.root,
                _python_files_override=[source_path],
                _trees_override={
                    source_path: CHECKER.ast.parse(
                        source,
                        filename=source_path.relative_to(canonical_root).as_posix(),
                    )
                },
            )

        discover_source(reviewed_source)
        function_body_marker = ") -> None:\n    reviewed = binding.reviewed\n"
        self.assertEqual(1, reviewed_source.count(function_body_marker))
        lookup_alias_source = reviewed_source.replace(
            function_body_marker,
            function_body_marker + "    modules = sys.modules\n",
            1,
        ).replace(reviewed_expression, "modules.get(binding.name) is not module", 1)
        wrong_file_source = (
            "import sys\n\n"
            "def _verify_python_source_module_binding(binding):\n"
            "    module = None\n"
            "    if True or sys.modules.get(binding.name) is not module:\n"
            "        return\n"
        )
        mutants = (
            (
                "result call",
                reviewed_source.replace(
                    reviewed_expression,
                    "sys.modules.get(binding.name).run() is not module",
                    1,
                ),
                reviewed_path,
            ),
            (
                "assignment alias",
                reviewed_source.replace(
                    reviewed_expression,
                    "(resolved := sys.modules.get(binding.name)) is not module",
                    1,
                ),
                reviewed_path,
            ),
            (
                "return",
                reviewed_source.replace(
                    function_body_marker,
                    ") -> None:\n"
                    "    return sys.modules.get(binding.name)\n"
                    "    reviewed = binding.reviewed\n",
                    1,
                ),
                reviewed_path,
            ),
            (
                "yield",
                reviewed_source.replace(
                    function_body_marker,
                    ") -> None:\n"
                    "    yield sys.modules.get(binding.name)\n"
                    "    reviewed = binding.reviewed\n",
                    1,
                ),
                reviewed_path,
            ),
            (
                "argument",
                reviewed_source.replace(
                    reviewed_expression,
                    "bool(sys.modules.get(binding.name))",
                    1,
                ),
                reviewed_path,
            ),
            (
                "container",
                reviewed_source.replace(
                    reviewed_expression,
                    "[sys.modules.get(binding.name)]",
                    1,
                ),
                reviewed_path,
            ),
            (
                "attribute",
                reviewed_source.replace(
                    reviewed_expression,
                    "sys.modules.get(binding.name).value is not module",
                    1,
                ),
                reviewed_path,
            ),
            (
                "subscript",
                reviewed_source.replace(
                    reviewed_expression,
                    "sys.modules.get(binding.name)[0] is not module",
                    1,
                ),
                reviewed_path,
            ),
            (
                "chained comparison",
                reviewed_source.replace(
                    reviewed_expression,
                    "sys.modules.get(binding.name) is not module is not reviewed",
                    1,
                ),
                reviewed_path,
            ),
            (
                "equality",
                reviewed_source.replace(
                    reviewed_expression, "sys.modules.get(binding.name) == module", 1
                ),
                reviewed_path,
            ),
            (
                "inequality",
                reviewed_source.replace(
                    reviewed_expression, "sys.modules.get(binding.name) != module", 1
                ),
                reviewed_path,
            ),
            (
                "identity",
                reviewed_source.replace(
                    reviewed_expression, "sys.modules.get(binding.name) is module", 1
                ),
                reviewed_path,
            ),
            (
                "wrong key",
                reviewed_source.replace(
                    reviewed_expression,
                    "sys.modules.get(binding.file) is not module",
                    1,
                ),
                reviewed_path,
            ),
            (
                "wrong expected value",
                reviewed_source.replace(
                    reviewed_expression,
                    "sys.modules.get(binding.name) is not reviewed",
                    1,
                ),
                reviewed_path,
            ),
            (
                "wrong file",
                wrong_file_source,
                canonical_root / "tools/fixture_reviewed_module_binding.py",
            ),
            (
                "wrong function",
                reviewed_source.replace(
                    "def _verify_python_source_module_binding(",
                    "def _verify_python_source_module_binding_alt(",
                    1,
                ),
                reviewed_path,
            ),
            ("namespace alias", lookup_alias_source, reviewed_path),
            (
                "arity",
                reviewed_source.replace(
                    reviewed_expression,
                    "sys.modules.get(binding.name, None) is not module",
                    1,
                ),
                reviewed_path,
            ),
            (
                "keyword",
                reviewed_source.replace(
                    reviewed_expression,
                    "sys.modules.get(key=binding.name) is not module",
                    1,
                ),
                reviewed_path,
            ),
        )
        for label, source, source_path in mutants:
            with self.subTest(reviewed_module_binding_lookup_mutant=label):
                with self.assertRaises(CHECKER.InventoryError):
                    discover_source(source, source_path)

        path = self.root / "tools/fixture_dynamic_sys_modules.py"
        path.write_text(
            "import subprocess\n"
            "import sys\n"
            "key = 'subprocess'\n"
            "sys.modules[key].run(['/bin/true'])\n",
            encoding="utf-8",
        )
        errors = "\n".join(CHECKER.validate(self.root, self.inventory_path))
        self.assertIn("source discovery failed closed", errors)
        self.assertIn("dynamic process namespace lookups", errors)

    def test_dynamic_code_aliases_fail_closed(self) -> None:
        canonical_root = CHECKER._canonical_root(self.root)
        reviewed_path = canonical_root / "tools/check_test_inventory.py"
        loader_source = (
            "@contextlib.contextmanager\n"
            "def _registered_python_tooling_modules(reviewed_modules):\n"
            "    registry = tuple(())\n"
            "    for binding in registry:\n"
            "        code = compile(\n"
            "            binding.reviewed.source_bytes,\n"
            "            binding.file,\n"
            '            "exec",\n'
            "            dont_inherit=True,\n"
            "        )\n"
            "        if type(code) is not types.CodeType or code.co_filename != binding.file:\n"
            '            raise InventoryError("Python tooling compiled code binding changed")\n'
            "        exec(code, binding.namespace)\n"
            "        _verify_python_source_module_registry(registry, reviewed_modules)\n"
            "    yield registry\n"
        )
        caller_source = (
            "def _run_python_tooling_root(\n"
            "    root, tooling_root, discovery_start, discovery_pattern,\n"
            "):\n"
            "    try:\n"
            "        _, _, dynamic_sites, reviewed_modules = _python_tooling_source_skip_review(\n"
            '            root, tooling_root["module_paths"], discovery_start, discovery_pattern\n'
            "        )\n"
            "        with _registered_python_tooling_modules(reviewed_modules) as module_registry:\n"
            "            return module_registry, dynamic_sites\n"
            "    except InventoryError:\n"
            "        raise\n"
        )
        reviewed_source = (REPOSITORY_ROOT / "tools/check_test_inventory.py").read_text(
            encoding="utf-8"
        )

        def discover_source(source: str, source_path: Path = reviewed_path) -> None:
            CHECKER._discover_python_launches(
                self.root,
                _python_files_override=[source_path],
                _trees_override={
                    source_path: CHECKER.ast.parse(
                        source,
                        filename=source_path.relative_to(canonical_root).as_posix(),
                    )
                },
            )

        discover_source(reviewed_source)
        compile_source = (
            "                code = compile(\n"
            "                    binding.reviewed.source_bytes,\n"
            "                    binding.file,\n"
            '                    "exec",\n'
            "                    dont_inherit=True,\n"
            "                )\n"
        )
        guard_source = (
            "                if type(code) is not types.CodeType or code.co_filename != binding.file:\n"
            '                    raise InventoryError("Python tooling compiled code binding changed")\n'
        )
        exec_source = "                exec(code, binding.namespace)\n"
        verify_source = "            _verify_python_source_module_registry(registry, reviewed_modules)\n"
        pre_verify_source = (
            "        registry = tuple(bindings)\n"
            "        _verify_python_source_module_registry(registry, reviewed_modules)\n"
            "        for binding in registry:\n"
        )
        snapshot_source = (
            "            observed = _read_regular_stable_snapshot(\n"
            "                binding.reviewed.source_path,\n"
            "                MAX_INVENTORY_BYTES,\n"
            '                f"Python tooling runtime source {binding.reviewed.inventory_path}",\n'
            "            )\n"
        )
        digest_source = (
            "            if observed.sha256 != binding.reviewed.source_sha256:\n"
            "                raise InventoryError(\n"
            '                    "Python tooling source changed between review and module execution"\n'
            "                )\n"
        )
        for fragment in (
            compile_source,
            guard_source,
            exec_source,
            verify_source,
            pre_verify_source,
            snapshot_source,
            digest_source,
        ):
            self.assertEqual(1, reviewed_source.count(fragment))
        nested_loader = "def loader_owner():\n" + "".join(
            ("    " + line if line.strip() else line)
            for line in loader_source.splitlines(keepends=True)
        )
        class_loader = "class LoaderOwner:\n" + "".join(
            ("    " + line if line.strip() else line)
            for line in loader_source.splitlines(keepends=True)
        )
        binding_verifier_start = reviewed_source.index(
            "def _verify_python_source_module_binding("
        )
        registry_verifier_start = reviewed_source.index(
            "def _verify_python_source_module_registry("
        )
        loader_start = reviewed_source.index(
            "@contextlib.contextmanager\ndef _registered_python_tooling_modules("
        )
        binding_verifier_pass = (
            reviewed_source[:binding_verifier_start]
            + "def _verify_python_source_module_binding(binding):\n    pass\n\n"
            + reviewed_source[registry_verifier_start:]
        )
        registry_verifier_pass = (
            reviewed_source[:registry_verifier_start]
            + "def _verify_python_source_module_registry(registry, reviewed_modules=None):\n    pass\n\n"
            + reviewed_source[loader_start:]
        )
        exec_mutants = (
            (
                "wrong file",
                reviewed_source,
                canonical_root / "tools/fixture_reviewed_exec.py",
            ),
            (
                "wrong function",
                reviewed_source.replace(
                    "def _registered_python_tooling_modules(",
                    "def _registered_python_tooling_modules_alt(",
                    1,
                ),
                reviewed_path,
            ),
            ("binding verifier pass", binding_verifier_pass, reviewed_path),
            ("registry verifier pass", registry_verifier_pass, reviewed_path),
            (
                "source hash guard weakened",
                reviewed_source.replace(
                    "or hashlib.sha256(reviewed.source_bytes).hexdigest() != reviewed.source_sha256",
                    "or False",
                    1,
                ),
                reviewed_path,
            ),
            (
                "namespace guard weakened",
                reviewed_source.replace(
                    "or type(namespace) is not dict", "or False", 1
                ),
                reviewed_path,
            ),
            (
                "reviewed identity guard weakened",
                reviewed_source.replace("binding.reviewed is not reviewed", "False", 1),
                reviewed_path,
            ),
            (
                "nested function",
                "import contextlib\nimport types\n\n"
                + nested_loader
                + "\n"
                + caller_source,
                reviewed_path,
            ),
            (
                "class function",
                "import contextlib\nimport types\n\n"
                + class_loader
                + "\n"
                + caller_source,
                reviewed_path,
            ),
            (
                "exec alias",
                reviewed_source.replace(
                    "    registry = tuple(bindings)\n",
                    "    registry = tuple(bindings)\n    run_exec = exec\n",
                    1,
                ).replace(
                    exec_source, "            run_exec(code, binding.namespace)\n", 1
                ),
                reviewed_path,
            ),
            (
                "wrong code",
                reviewed_source.replace(
                    exec_source, "            exec(other, binding.namespace)\n", 1
                ),
                reviewed_path,
            ),
            (
                "wrong source",
                reviewed_source.replace(
                    compile_source,
                    compile_source.replace(
                        "binding.reviewed.source_bytes", "binding.source_bytes", 1
                    ),
                    1,
                ),
                reviewed_path,
            ),
            (
                "wrong file",
                reviewed_source.replace(
                    "                binding.file,\n",
                    "                binding.reviewed.source_path,\n",
                    1,
                ),
                reviewed_path,
            ),
            (
                "wrong mode",
                reviewed_source.replace(
                    '                "exec",\n', '                "eval",\n', 1
                ),
                reviewed_path,
            ),
            (
                "extra compile keyword",
                reviewed_source.replace(
                    "                dont_inherit=True,\n",
                    "                dont_inherit=True,\n                optimize=0,\n",
                    1,
                ),
                reviewed_path,
            ),
            (
                "wrong dont inherit",
                reviewed_source.replace("dont_inherit=True", "dont_inherit=False", 1),
                reviewed_path,
            ),
            (
                "missing dont inherit",
                reviewed_source.replace("                dont_inherit=True,\n", "", 1),
                reviewed_path,
            ),
            (
                "wrong compile keyword",
                reviewed_source.replace("dont_inherit=True", "flags=True", 1),
                reviewed_path,
            ),
            (
                "missing pre verify",
                reviewed_source.replace(
                    pre_verify_source,
                    "        registry = tuple(bindings)\n"
                    "        for binding in registry:\n",
                    1,
                ),
                reviewed_path,
            ),
            (
                "wrong pre verify",
                reviewed_source.replace(
                    pre_verify_source,
                    "        registry = tuple(bindings)\n"
                    "        _verify_python_source_module_registry(registry)\n"
                    "        for binding in registry:\n",
                    1,
                ),
                reviewed_path,
            ),
            (
                "wrong registry proof",
                reviewed_source.replace(
                    pre_verify_source,
                    pre_verify_source.replace(
                        "registry = tuple(bindings)",
                        "registry = tuple(reviewed_modules)",
                        1,
                    ),
                    1,
                ),
                reviewed_path,
            ),
            (
                "nonadjacent guard",
                reviewed_source.replace(
                    guard_source, guard_source + "                pass\n", 1
                ),
                reviewed_path,
            ),
            (
                "code rebound",
                reviewed_source.replace(
                    guard_source, guard_source + "                code = other\n", 1
                ),
                reviewed_path,
            ),
            (
                "wrong code type guard",
                reviewed_source.replace("types.CodeType", "types.FunctionType", 1),
                reviewed_path,
            ),
            (
                "missing filename guard",
                reviewed_source.replace(
                    "type(code) is not types.CodeType or code.co_filename != binding.file",
                    "type(code) is not types.CodeType",
                    1,
                ),
                reviewed_path,
            ),
            (
                "wrong filename guard",
                reviewed_source.replace(
                    "code.co_filename != binding.file",
                    "code.co_filename != binding.reviewed.source_path",
                    1,
                ),
                reviewed_path,
            ),
            (
                "wrong guard error",
                reviewed_source.replace(
                    "Python tooling compiled code binding changed",
                    "compiled code changed",
                    1,
                ),
                reviewed_path,
            ),
            (
                "wrong namespace",
                reviewed_source.replace(
                    exec_source, "            exec(code, binding.module)\n", 1
                ),
                reviewed_path,
            ),
            (
                "third locals",
                reviewed_source.replace(
                    exec_source,
                    "            exec(code, binding.namespace, {})\n",
                    1,
                ),
                reviewed_path,
            ),
            (
                "return",
                reviewed_source.replace(
                    exec_source,
                    "            return exec(code, binding.namespace)\n",
                    1,
                ),
                reviewed_path,
            ),
            (
                "assignment",
                reviewed_source.replace(
                    exec_source,
                    "            result = exec(code, binding.namespace)\n",
                    1,
                ),
                reviewed_path,
            ),
            (
                "lambda",
                reviewed_source.replace(
                    exec_source,
                    "            (lambda: exec(code, binding.namespace))\n",
                    1,
                ),
                reviewed_path,
            ),
            (
                "container",
                reviewed_source.replace(
                    exec_source,
                    "            [exec(code, binding.namespace)]\n",
                    1,
                ),
                reviewed_path,
            ),
            (
                "argument",
                reviewed_source.replace(
                    exec_source,
                    "            consume(exec(code, binding.namespace))\n",
                    1,
                ),
                reviewed_path,
            ),
            (
                "missing post verify",
                reviewed_source.replace(verify_source, "            pass\n", 1),
                reviewed_path,
            ),
            (
                "wrong post verify",
                reviewed_source.replace(
                    verify_source,
                    "            _verify_python_source_module_registry(registry)\n",
                    1,
                ),
                reviewed_path,
            ),
            (
                "missing snapshot",
                reviewed_source.replace(
                    snapshot_source, "            observed = None\n", 1
                ),
                reviewed_path,
            ),
            (
                "wrong snapshot source",
                reviewed_source.replace(
                    snapshot_source,
                    snapshot_source.replace(
                        "binding.reviewed.source_path", "binding.file", 1
                    ),
                    1,
                ),
                reviewed_path,
            ),
            (
                "wrong snapshot limit",
                reviewed_source.replace(
                    snapshot_source,
                    snapshot_source.replace("MAX_INVENTORY_BYTES", "MAX_LINE_BYTES", 1),
                    1,
                ),
                reviewed_path,
            ),
            (
                "wrong snapshot label",
                reviewed_source.replace(
                    snapshot_source,
                    snapshot_source.replace(
                        "Python tooling runtime source ", "Python tooling source ", 1
                    ),
                    1,
                ),
                reviewed_path,
            ),
            (
                "snapshot call alias",
                reviewed_source.replace(
                    "    registry = tuple(bindings)\n",
                    "    registry = tuple(bindings)\n"
                    "    read_snapshot = _read_regular_stable_snapshot\n",
                    1,
                ).replace(
                    "            observed = _read_regular_stable_snapshot(\n",
                    "            observed = read_snapshot(\n",
                    1,
                ),
                reviewed_path,
            ),
            (
                "missing digest check",
                reviewed_source.replace(digest_source, "            pass\n", 1),
                reviewed_path,
            ),
            (
                "wrong digest left operand",
                reviewed_source.replace(
                    digest_source,
                    digest_source.replace("observed.sha256", "observed.bytes", 1),
                    1,
                ),
                reviewed_path,
            ),
            (
                "wrong digest right operand",
                reviewed_source.replace(
                    digest_source,
                    digest_source.replace(
                        "binding.reviewed.source_sha256", "binding.file", 1
                    ),
                    1,
                ),
                reviewed_path,
            ),
            (
                "wrong digest operator",
                reviewed_source.replace(
                    digest_source,
                    digest_source.replace(" != ", " == ", 1),
                    1,
                ),
                reviewed_path,
            ),
            (
                "wrong digest error",
                reviewed_source.replace(
                    digest_source,
                    digest_source.replace(
                        "Python tooling source changed between review and module execution",
                        "Python tooling source changed",
                        1,
                    ),
                    1,
                ),
                reviewed_path,
            ),
            (
                "digest error swallowed",
                reviewed_source.replace(
                    digest_source,
                    "            if observed.sha256 != binding.reviewed.source_sha256:\n"
                    "                try:\n"
                    "                    raise InventoryError(\n"
                    '                        "Python tooling source changed between review and module execution"\n'
                    "                    )\n"
                    "                except InventoryError:\n"
                    "                    pass\n",
                    1,
                ),
                reviewed_path,
            ),
            (
                "extra compile",
                reviewed_source.replace(
                    compile_source, compile_source + compile_source, 1
                ),
                reviewed_path,
            ),
            (
                "extra exec",
                reviewed_source.replace(exec_source, exec_source + exec_source, 1),
                reviewed_path,
            ),
            (
                "caller provenance disconnected",
                reviewed_source.replace(
                    "_, _, dynamic_sites, reviewed_modules = _python_tooling_source_skip_review(",
                    "_, _, dynamic_sites, reviewed_input = _python_tooling_source_skip_review(",
                    1,
                ).replace(
                    "_registered_python_tooling_modules(reviewed_modules)",
                    "_registered_python_tooling_modules(reviewed_input)",
                    1,
                ),
                reviewed_path,
            ),
            (
                "eval",
                reviewed_source.replace(
                    exec_source, "            eval(code, binding.namespace)\n", 1
                ),
                reviewed_path,
            ),
        )
        for label, source, source_path in exec_mutants:
            with self.subTest(reviewed_source_exec_mutant=label):
                self.assertTrue(
                    source != reviewed_source or source_path != reviewed_path,
                    label,
                )
                with self.assertRaises(CHECKER.InventoryError):
                    discover_source(source, source_path)

        path = self.root / "tools/fixture_dynamic_code_alias.py"
        for source in (
            "from builtins import eval as evaluate\n"
            "evaluate(\"__import__('subprocess').run(['/bin/true'])\")\n",
            "evaluate = eval\nevaluate('1 + 1')\n",
            "__builtins__.eval('1 + 1')\n",
        ):
            with self.subTest(source=source):
                path.write_text(source, encoding="utf-8")
                errors = "\n".join(CHECKER.validate(self.root, self.inventory_path))
                self.assertIn("source discovery failed closed", errors)
                self.assertIn("dynamic Python code execution", errors)

    def test_deleted_module_builtin_shadows_restore_builtin_lookup(self) -> None:
        path = self.root / "tools/fixture_deleted_builtin_shadow.py"
        path.write_text(
            "globals = lambda: {}\n"
            "del globals\n"
            "__import__ = lambda name: None\n"
            "del __import__\n"
            "import subprocess\n"
            "globals()['subprocess'].run(['/bin/true'])\n"
            "__import__('subprocess').run(['/bin/true'])\n",
            encoding="utf-8",
        )
        errors = "\n".join(CHECKER.validate(self.root, self.inventory_path))
        self.assertIn(":module:subprocess.run:1", errors)
        self.assertIn(":module:subprocess.run:2", errors)

    def test_repository_local_module_shadows_process_module(self) -> None:
        fixture_dir = self.root / "tools/local_shadow_fixture"
        fixture_dir.mkdir()
        local_module = fixture_dir / "subprocess.py"
        local_module.write_text(
            "def run(command):\n    return command\n", encoding="utf-8"
        )
        consumer = fixture_dir / "consumer.py"
        consumer.write_text(
            "import subprocess\n"
            "from subprocess import run\n"
            "subprocess.run(['/bin/true'])\n"
            "run(['/bin/true'])\n",
            encoding="utf-8",
        )
        errors = "\n".join(CHECKER.validate(self.root, self.inventory_path))
        self.assertIn("source discovery failed closed", errors)
        self.assertIn(
            "local modules shadowing tracked standard-library modules", errors
        )

    def test_function_local_import_rejects_process_module_shadow_ambiguity(
        self,
    ) -> None:
        fixture_dir = self.root / "tools/function_local_shadow_fixture"
        fixture_dir.mkdir()
        (fixture_dir / "subprocess.py").write_text(
            "def run(command):\n    return command\n", encoding="utf-8"
        )
        (fixture_dir / "consumer.py").write_text(
            "def launch():\n"
            "    import subprocess\n"
            "    from subprocess import run\n"
            "    subprocess.run(['/bin/true'])\n"
            "    run(['/bin/true'])\n",
            encoding="utf-8",
        )
        errors = "\n".join(CHECKER.validate(self.root, self.inventory_path))
        self.assertIn("source discovery failed closed", errors)
        self.assertIn(
            "local modules shadowing tracked standard-library modules", errors
        )

    def test_local_module_shadowing_dynamic_process_import_fails_closed(self) -> None:
        local_module = self.root / "subprocess.py"
        local_module.write_text(
            "def run(command):\n    return command\n", encoding="utf-8"
        )
        consumer = self.root / "tools/fixture_dynamic_local_subprocess.py"
        consumer.write_text(
            "import importlib\n"
            "importlib.import_module('subprocess').run(['/bin/true'])\n",
            encoding="utf-8",
        )
        errors = "\n".join(CHECKER.validate(self.root, self.inventory_path))
        self.assertIn("source discovery failed closed", errors)
        self.assertIn("local modules shadowing", errors)

    def test_builtin_import_fromlist_process_module_is_discovered(self) -> None:
        path = self.root / "tools/fixture_builtin_fromlist.py"
        path.write_text(
            "def launch():\n"
            "    executor = __import__(\n"
            "        'concurrent.futures',\n"
            "        fromlist=['ProcessPoolExecutor'],\n"
            "    ).ProcessPoolExecutor()\n"
            "    executor.submit(abs, -1)\n",
            encoding="utf-8",
        )
        self._assert_error_contains(
            "python-launch:tools/fixture_builtin_fromlist.py:launch:"
            "concurrent.futures.process_pool.submit:1"
        )

    def test_builtin_import_nonzero_level_fails_closed(self) -> None:
        package = self.root / "tools/relative_import_fixture"
        nested = package / "nested"
        nested.mkdir(parents=True)
        (package / "__init__.py").write_text("", encoding="utf-8")
        (nested / "__init__.py").write_text("", encoding="utf-8")
        (package / "subprocess.py").write_text(
            "def run(command):\n    return command\n", encoding="utf-8"
        )
        (nested / "consumer.py").write_text(
            "__import__(\n"
            "    'subprocess', globals(), locals(), ['run'], 2\n"
            ").run(['/bin/true'])\n",
            encoding="utf-8",
        )
        errors = "\n".join(CHECKER.validate(self.root, self.inventory_path))
        self.assertIn("source discovery failed closed", errors)
        self.assertNotIn(":subprocess.run:", errors)

    def test_equivalent_builtin_import_helpers_are_discovered(self) -> None:
        path = self.root / "tools/fixture_equivalent_builtin_imports.py"
        path.write_text(
            "import importlib\n\n"
            "async def launches():\n"
            "    importlib.__import__('subprocess').run(['/bin/true'])\n"
            "    await __builtins__.__import__(\n"
            "        'asyncio.subprocess'\n"
            "    ).subprocess.create_subprocess_exec('/bin/true')\n",
            encoding="utf-8",
        )
        errors = "\n".join(CHECKER.validate(self.root, self.inventory_path))
        self.assertIn(":launches:subprocess.run:1", errors)
        self.assertIn(":launches:asyncio.create_subprocess_exec:1", errors)

    @unittest.skipUnless(
        sys.version_info >= (3, 12), "requires Python 3.12+ PEP 695 syntax"
    )
    def test_pep_695_bindings_shadow_process_modules(self) -> None:
        module_alias = self.root / "tools/fixture_pep_695_module_alias.py"
        module_alias.write_text(
            "import subprocess\n"
            "type subprocess = int\n\n"
            "def module_alias():\n"
            "    subprocess.run(['/bin/true'])\n",
            encoding="utf-8",
        )
        type_parameter = self.root / "tools/fixture_pep_695_type_parameter.py"
        type_parameter.write_text(
            "import subprocess\n\n"
            "def type_parameter[subprocess]():\n"
            "    subprocess.run(['/bin/true'])\n",
            encoding="utf-8",
        )
        class_parameter = self.root / "tools/fixture_pep_695_class_parameter.py"
        class_parameter.write_text(
            "import subprocess\n"
            "class Fixture[subprocess]:\n"
            "    result = subprocess.run(['/bin/true'])\n",
            encoding="utf-8",
        )
        alias_parameter = self.root / "tools/fixture_pep_695_alias_parameter.py"
        alias_parameter.write_text(
            "import subprocess\n"
            "type Alias[subprocess] = subprocess.run(['/bin/true'])\n"
            "value = Alias.__value__\n",
            encoding="utf-8",
        )
        definition_scope = self.root / "tools/fixture_pep_695_definition_scope.py"
        definition_scope.write_text(
            "import subprocess\n\n"
            "class Fixture[subprocess](subprocess.run(['/bin/true'])):\n"
            "    pass\n\n"
            "def fixture[subprocess]() -> subprocess.run(['/bin/true']):\n"
            "    pass\n",
            encoding="utf-8",
        )
        fixture_paths = {
            module_alias.relative_to(self.root).as_posix(),
            type_parameter.relative_to(self.root).as_posix(),
            class_parameter.relative_to(self.root).as_posix(),
            alias_parameter.relative_to(self.root).as_posix(),
            definition_scope.relative_to(self.root).as_posix(),
        }
        self.assertFalse(
            any(
                item["anchor"]["file"] in fixture_paths
                for item in CHECKER._discover_python_launches(self.root)
            )
        )

    @unittest.skipUnless(
        sys.version_info >= (3, 12), "requires Python 3.12+ PEP 695 syntax"
    )
    def test_pep_695_annotations_do_not_bind_runtime_parameters(self) -> None:
        path = self.root / "tools/fixture_pep_695_annotation_scope.py"
        path.write_text(
            "import subprocess\n\n"
            "def return_annotation[T](subprocess: int) "
            "-> subprocess.run(['/bin/true']):\n"
            "    pass\n\n"
            "def parameter_annotation[T](subprocess: "
            "subprocess.run(['/bin/true'])):\n"
            "    pass\n"
            "values = (return_annotation.__annotations__, "
            "parameter_annotation.__annotations__)\n",
            encoding="utf-8",
        )
        errors = "\n".join(CHECKER.validate(self.root, self.inventory_path))
        self.assertIn(
            "python-launch:tools/fixture_pep_695_annotation_scope.py:module:subprocess.run:1",
            errors,
        )
        self.assertIn(
            "python-launch:tools/fixture_pep_695_annotation_scope.py:module:subprocess.run:2",
            errors,
        )

    def test_pty_and_multiprocessing_launches_are_discovered(self) -> None:
        path = self.root / "tools/fixture_process_modules.py"
        path.write_text(
            "import multiprocessing\n"
            "import pty\n\n"
            "def standard_process_modules():\n"
            "    pty.fork()\n"
            "    process_factory = multiprocessing.Process\n"
            "    process = process_factory(target=print)\n"
            "    process.start()\n",
            encoding="utf-8",
        )
        errors = "\n".join(CHECKER.validate(self.root, self.inventory_path))
        self.assertIn(":standard_process_modules:pty.fork:1", errors)
        self.assertIn(
            ":standard_process_modules:multiprocessing.process.start:1", errors
        )

    def test_posix_process_entries_use_os_canonical_names(self) -> None:
        path = self.root / "tools/fixture_posix_process_entries.py"
        path.write_text(
            "import posix\n"
            "from posix import fork\n\n"
            "def launches():\n"
            "    posix.fork()\n"
            "    fork()\n",
            encoding="utf-8",
        )
        errors = "\n".join(CHECKER.validate(self.root, self.inventory_path))
        self.assertIn(":launches:os.fork:1", errors)
        self.assertIn(":launches:os.fork:2", errors)

    def test_process_import_prefixes_and_builtin_helper_are_discovered(self) -> None:
        path = self.root / "tools/fixture_import_prefixes.py"
        path.write_text(
            "import asyncio.subprocess\n"
            "import importlib.util\n"
            "from builtins import __import__ as load\n\n"
            "def prefixed_import_fixture():\n"
            "    importlib.import_module('subprocess').run(['python3', '-V'])\n"
            "    asyncio.create_subprocess_exec('python3', '-V')\n"
            "    load('subprocess').run(['python3', '-V'])\n",
            encoding="utf-8",
        )
        errors = "\n".join(CHECKER.validate(self.root, self.inventory_path))
        self.assertIn(":prefixed_import_fixture:subprocess.run:1", errors)
        self.assertIn(":prefixed_import_fixture:subprocess.run:2", errors)
        self.assertIn(
            ":prefixed_import_fixture:asyncio.create_subprocess_exec:1", errors
        )

    def test_literal_dynamic_non_process_import_fails_closed(self) -> None:
        path = self.root / "tools/fixture_dynamic_local.py"
        path.write_text(
            "import importlib\n"
            "module = importlib.import_module('tools.check_abi_manifest')\n"
            "module.launch(['python3', '-V'])\n",
            encoding="utf-8",
        )
        errors = "\n".join(CHECKER.validate(self.root, self.inventory_path))
        self.assertIn("source discovery failed closed", errors)
        self.assertIn("dynamic imports of non-process modules are unsupported", errors)

    def test_function_static_local_bindings_prevent_false_launches(self) -> None:
        path = self.root / "tools/fixture_static_locals.py"
        path.write_text(
            "import subprocess\n\n"
            "class Fake:\n"
            "    @staticmethod\n"
            "    def run(arguments):\n"
            "        return arguments\n\n"
            "def later_assignment():\n"
            "    subprocess.run(['python3', '-V'])\n"
            "    subprocess = Fake\n\n"
            "def nested_definition():\n"
            "    subprocess.run(['python3', '-V'])\n"
            "    def subprocess():\n"
            "        return None\n\n"
            "def nested_class_definition():\n"
            "    subprocess.run(['python3', '-V'])\n"
            "    class subprocess:\n"
            "        pass\n\n"
            "def exception_binding():\n"
            "    try:\n"
            "        pass\n"
            "    except Exception as subprocess:\n"
            "        pass\n"
            "    subprocess.run(['python3', '-V'])\n",
            encoding="utf-8",
        )
        self.assertFalse(
            any(
                item["anchor"]["file"] == path.relative_to(self.root).as_posix()
                for item in CHECKER._discover_python_launches(self.root)
            )
        )

    def test_ambiguous_closure_process_binding_fails_closed(self) -> None:
        path = self.root / "tools/fixture_ambiguous_closure.py"
        path.write_text(
            "import subprocess\n\n"
            "def outer():\n"
            "    launch = subprocess.run\n"
            "    def inner():\n"
            "        return launch(['python3', '-V'])\n"
            "    launch = lambda arguments: arguments\n"
            "    return inner\n",
            encoding="utf-8",
        )
        errors = "\n".join(CHECKER.validate(self.root, self.inventory_path))
        self.assertIn("source discovery failed closed", errors)
        self.assertIn("process aliases with multiple enclosing bindings", errors)

    def test_ambiguous_module_process_binding_fails_closed(self) -> None:
        path = self.root / "tools/fixture_ambiguous_module.py"
        path.write_text(
            "import subprocess\n\n"
            "def hidden():\n"
            "    return subprocess.run(['python3', '-V'])\n\n"
            "subprocess = object()\n",
            encoding="utf-8",
        )
        errors = "\n".join(CHECKER.validate(self.root, self.inventory_path))
        self.assertIn("source discovery failed closed", errors)
        self.assertIn("process aliases with multiple enclosing bindings", errors)

    def test_global_and_nonlocal_process_bindings_fail_closed(self) -> None:
        path = self.root / "tools/fixture_scope_directives.py"
        path.write_text(
            "import subprocess\n\n"
            "def global_fixture():\n"
            "    global subprocess\n"
            "    subprocess.run(['python3', '-V'])\n\n"
            "def outer():\n"
            "    launch = subprocess.run\n"
            "    def inner():\n"
            "        nonlocal launch\n"
            "        return launch(['python3', '-V'])\n"
            "    return inner\n",
            encoding="utf-8",
        )
        errors = "\n".join(CHECKER.validate(self.root, self.inventory_path))
        self.assertIn("source discovery failed closed", errors)
        self.assertIn("global and nonlocal process aliases are unsupported", errors)

    def test_assignment_rhs_uses_pre_binding_aliases(self) -> None:
        path = self.root / "tools/fixture_assignment_rhs.py"
        path.write_text(
            "import subprocess\nsubprocess = subprocess.run(['python3', '-V'])\n",
            encoding="utf-8",
        )
        self._assert_error_contains(
            "python-launch:tools/fixture_assignment_rhs.py:module:subprocess.run:1"
        )

    def test_lambda_and_definition_time_process_calls_are_discovered(self) -> None:
        path = self.root / "tools/fixture_definition_time.py"
        path.write_text(
            "import subprocess\n"
            "launch = lambda runner=subprocess.run: runner(['python3', '-V'])\n\n"
            "def definition_time(value=subprocess.run(['python3', '-V'])):\n"
            "    return value\n",
            encoding="utf-8",
        )
        errors = "\n".join(CHECKER.validate(self.root, self.inventory_path))
        self.assertIn(
            "python-launch:tools/fixture_definition_time.py:module:subprocess.run:1",
            errors,
        )
        self.assertIn(
            "python-launch:tools/fixture_definition_time.py:module:subprocess.run:2",
            errors,
        )

    def test_lambda_parameter_shadow_prevents_false_launch(self) -> None:
        path = self.root / "tools/fixture_lambda_shadow.py"
        path.write_text(
            "class Fake:\n"
            "    @staticmethod\n"
            "    def run(arguments):\n"
            "        return arguments\n\n"
            "hidden = lambda subprocess=Fake: subprocess.run(['python3', '-V'])\n",
            encoding="utf-8",
        )
        self.assertFalse(
            any(
                item["anchor"]["file"] == path.relative_to(self.root).as_posix()
                for item in CHECKER._discover_python_launches(self.root)
            )
        )

    def test_process_api_attribute_rebinding_fails_closed(self) -> None:
        path = self.root / "tools/fixture_attribute_rebind.py"
        path.write_text(
            "import subprocess\n"
            "subprocess.run = lambda arguments: arguments\n"
            "subprocess.run(['python3', '-V'])\n",
            encoding="utf-8",
        )
        errors = "\n".join(CHECKER.validate(self.root, self.inventory_path))
        self.assertIn("source discovery failed closed", errors)
        self.assertIn(
            "mutating tracked process alias attributes is unsupported", errors
        )

        canonical_root = CHECKER._canonical_root(self.root)
        reviewed_path = canonical_root / "tools/check_test_inventory.py"
        reviewed_source = reviewed_path.read_text(encoding="utf-8")
        assignment_literal = '    "subprocess.run=capsule_run\\n"\n'
        self.assertEqual(1, reviewed_source.count(assignment_literal))
        bootstrap_assignment_mutants = (
            ("wrong attribute", '    "subprocess.Popen=capsule_run\\n"\n'),
            ("wrong target", '    "process.run=capsule_run\\n"\n'),
            ("wrong function", '    "subprocess.run=trusted_run\\n"\n'),
            ("wrong literal", '    "subprocess.run = capsule_run\\n"\n'),
            ("wrong context", '    "if True: subprocess.run=capsule_run\\n"\n'),
            (
                "additional assignment",
                '    "subprocess.run=capsule_run\\n"\n'
                '    "subprocess.run=capsule_run\\n"\n',
            ),
        )
        for label, replacement in bootstrap_assignment_mutants:
            with self.subTest(bootstrap_process_sink_mutant=label):
                mutated = reviewed_source.replace(assignment_literal, replacement, 1)
                with self.assertRaises(CHECKER.InventoryError):
                    CHECKER._discover_python_launches(
                        canonical_root,
                        _python_files_override=[reviewed_path],
                        _trees_override={
                            reviewed_path: CHECKER.ast.parse(
                                mutated, filename="tools/check_test_inventory.py"
                            )
                        },
                    )

        launcher_executable_guard = (
            '        if "executable" in kwargs:\n'
            '            raise InventoryError("Python tooling capsule transport is checker-owned")\n'
        )
        launcher_passthrough = (
            "        if os.path.realpath(argv[0]) != os.path.realpath(sys.executable):\n"
            "            return _PYTHON_TOOLING_TRUSTED_SUBPROCESS_RUN(args, **kwargs)\n"
        )
        bootstrap_executable_guard = "    \" if 'executable' in kwargs: raise OSError('capsule transport is checker-owned')\\n\"\n"
        bootstrap_passthrough = '    " if os.path.realpath(argv[0])!=os.path.realpath(sys.executable): return trusted_run(args,**kwargs)\\n"\n'
        for fragment in (
            launcher_executable_guard,
            launcher_passthrough,
            bootstrap_executable_guard,
            bootstrap_passthrough,
            CHECKER.REVIEWED_TEST_INVENTORY_BOOTSTRAP_SHA256,
        ):
            self.assertEqual(1, reviewed_source.count(fragment))
        ordering_and_pin_mutants = (
            (
                "non-Python argv0 executable override bypass in physical launcher",
                reviewed_source.replace(launcher_executable_guard, "", 1).replace(
                    launcher_passthrough,
                    launcher_passthrough + launcher_executable_guard,
                    1,
                ),
            ),
            (
                "non-Python argv0 executable override bypass in bootstrap",
                reviewed_source.replace(bootstrap_executable_guard, "", 1).replace(
                    bootstrap_passthrough,
                    bootstrap_passthrough + bootstrap_executable_guard,
                    1,
                ),
            ),
            (
                "old bootstrap pin",
                reviewed_source.replace(
                    CHECKER.REVIEWED_TEST_INVENTORY_BOOTSTRAP_SHA256,
                    "b5c75747b0dd7b3d30d26e8e0887bcaf530a165ec3fb8462a5b437e53c7057d2",
                    1,
                ),
            ),
        )
        for label, mutated in ordering_and_pin_mutants:
            with self.subTest(bootstrap_ordering_or_pin_mutant=label):
                with self.assertRaises(CHECKER.InventoryError):
                    CHECKER._discover_python_launches(
                        canonical_root,
                        _python_files_override=[reviewed_path],
                        _trees_override={
                            reviewed_path: CHECKER.ast.parse(
                                mutated, filename="tools/check_test_inventory.py"
                            )
                        },
                    )

    def test_named_workflow_step_command_drift_fails(self) -> None:
        path = self.root / ".github/workflows/ci.yml"
        text = path.read_text(encoding="utf-8")
        trusted_windows_shell = r'''C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe -NoLogo -NoProfile -NonInteractive -Command "$ErrorActionPreference = 'Stop'; $env:GITHUB_ENV = 'NUL'; $env:GITHUB_PATH = 'NUL'; & ([scriptblock]::Create([IO.File]::ReadAllText('{0}'))); if (Test-Path -LiteralPath variable:\LASTEXITCODE) {{ exit $LASTEXITCODE }}"'''
        trusted_windows_env = (
            "        env:\n"
            "          TRUSTED_PATH: ${{ steps.trusted-windows-tools.outputs.path }}\n"
            "          TRUSTED_ZIG: ${{ steps.trusted-windows-tools.outputs.zig }}\n"
            "          TRUSTED_PYTHON: ${{ steps.trusted-windows-tools.outputs.python }}\n"
        )
        trusted_posix_shell = (
            "/usr/bin/env BASH_ENV=/dev/null ENV=/dev/null "
            "GITHUB_ENV=/dev/null GITHUB_PATH=/dev/null "
            "/bin/bash --noprofile --norc -p -eo pipefail {0}"
        )
        trusted_posix_env = (
            "        env:\n"
            "          PATH: ${{ steps.trusted-posix-tools.outputs.path }}\n"
            "          BASH_ENV: /dev/null\n"
            "          ENV: /dev/null\n"
        )

        def named_step(source: str, name: str) -> str:
            marker = f"      - name: {name}\n"
            start = source.index(marker)
            lines = source[start:].splitlines()
            step_lines = [lines[0]]
            for line in lines[1:]:
                if line.startswith("      - ") or (
                    line.startswith("  ") and not line.startswith("   ")
                ):
                    break
                step_lines.append(line)
            return "\n".join(step_lines).rstrip()

        checkout = (
            "      - uses: actions/checkout@fbc6f3992d24b796d5a048ff273f7fcc4a7b6c09 # v5.1.0\n"
            "        with:\n"
            "          persist-credentials: false\n"
        )
        setup_zig = (
            "      - uses: mlugg/setup-zig@d1434d08867e3ee9daa34448df10607b98908d29 # v2.2.1\n"
            "        with:\n"
            "          version: 0.17.0\n"
            "          cache-key: ${{ matrix.cache_target }}\n"
            "          cache-size-limit: 8192\n"
        )
        expected_source_checker = (
            "      - name: Check build inventory\n"
            f"        shell: {trusted_posix_shell}\n"
            f"{trusted_posix_env}"
            "        run: python3 -B tools/check_build_inventory.py --root ."
        )
        source_checker = named_step(text, "Check build inventory")
        self.assertEqual(expected_source_checker, source_checker)
        expected_windows_build = (
            "      - name: Build Windows Python tooling executable fixtures and libraries\n"
            "        if: runner.os == 'Windows' && matrix.cache_target == 'windows-x86_64-baseline'\n"
            f"        shell: {trusted_windows_shell}\n"
            f"{trusted_windows_env}"
            "        run: |\n"
            "          Set-StrictMode -Version Latest\n"
            "          $env:PATH = $env:TRUSTED_PATH\n"
            "          $ErrorActionPreference = 'Stop'\n"
            "          $dynamicLibrary = 'zig-out/bin/zynum_blas.dll'\n"
            "          if (Test-Path -LiteralPath $dynamicLibrary) {\n"
            '            Write-Error "canonical Windows DLL already exists before the reviewed build: $dynamicLibrary"\n'
            "            exit 1\n"
            "          }\n"
            "          & $env:TRUSTED_ZIG build install-libraries build-rank-k-probe build-rotg-latency-probe build-symm-probe build-triangular-matrix-probe ${{ matrix.target_args }} --release=fast --summary failures\n"
            "          if ($LASTEXITCODE -ne 0) {\n"
            "            exit $LASTEXITCODE\n"
            "          }"
        )
        windows_binding_name = "Bind trusted Windows toolchain paths"
        posix_binding_name = (
            "Bind trusted POSIX toolchain paths and require Fortran for full smoke"
        )
        windows_build_name = (
            "Build Windows Python tooling executable fixtures and libraries"
        )
        windows_layout_name = (
            "Check Windows library layout and tooling fixture boundary"
        )
        windows_gate_name = (
            "Run Windows DLL ABI and CBLAS L1-L3 compatibility smoke "
            "(not inventory evidence)"
        )
        windows_binding = named_step(text, windows_binding_name)
        posix_binding = named_step(text, posix_binding_name)
        windows_build = named_step(text, windows_build_name)
        windows_layout = named_step(text, windows_layout_name)
        windows_gate = named_step(text, windows_gate_name)
        expected_host_tool_step = (
            "      - name: Run host tool smoke once\n"
            "        if: matrix.zig_gate == 'inventory-certified'\n"
            "        timeout-minutes: 60\n"
            f"        shell: {trusted_posix_shell}\n"
            f"{trusted_posix_env}"
            "        run: zig build test-host-tool-smoke "
            "${{ matrix.target_args }} --summary failures"
        )
        host_tool_step = named_step(text, "Run host tool smoke once")
        self.assertEqual(expected_windows_build, windows_build)
        self.assertEqual(expected_host_tool_step, host_tool_step)
        self.assertTrue(
            windows_binding.startswith(
                "      - name: Bind trusted Windows toolchain paths\n"
                "        id: trusted-windows-tools\n"
                "        if: runner.os == 'Windows'\n"
                f"        shell: {trusted_windows_shell}\n"
                "        run: |\n"
            )
        )
        self.assertTrue(
            posix_binding.startswith(
                "      - name: Bind trusted POSIX toolchain paths and require Fortran for full smoke\n"
                "        id: trusted-posix-tools\n"
                "        if: runner.os != 'Windows'\n"
                "        shell: /bin/bash --noprofile --norc -p -eo pipefail {0}\n"
                "        env:\n"
                "          BASH_ENV: /dev/null\n"
                "          ENV: /dev/null\n"
                "          REQUIRE_FORTRAN: ${{ matrix.full_smoke && runner.os == 'Linux' }}\n"
                "        run: |\n"
            )
        )
        checkout_index = text.index(checkout, text.index("  target-tests:\n"))
        setup_index = text.index(setup_zig, checkout_index + len(checkout))
        windows_binding_index = text.index(
            windows_binding, setup_index + len(setup_zig)
        )
        posix_binding_index = text.index(
            posix_binding, windows_binding_index + len(windows_binding)
        )
        build_index = text.index(
            windows_build, posix_binding_index + len(posix_binding)
        )
        layout_index = text.index(windows_layout, build_index + len(windows_build))
        gate_index = text.index(windows_gate, layout_index + len(windows_layout))
        host_tool_index = text.index(host_tool_step, gate_index + len(windows_gate))
        self.assertEqual(checkout_index + len(checkout) + 1, setup_index)
        self.assertEqual(setup_index + len(setup_zig) + 1, windows_binding_index)
        self.assertEqual(
            windows_binding_index + len(windows_binding) + 2,
            posix_binding_index,
        )
        self.assertEqual(
            posix_binding_index + len(posix_binding) + 2,
            build_index,
        )
        self.assertEqual(build_index + len(windows_build) + 2, layout_index)
        self.assertEqual(layout_index + len(windows_layout) + 2, gate_index)
        self.assertEqual(gate_index + len(windows_gate) + 2, host_tool_index)
        windows_gate_marker = f"      - name: {windows_gate_name}\n"
        self.assertTrue(
            windows_gate.startswith(
                windows_gate_marker + "        if: runner.os == 'Windows' && "
                "matrix.cache_target == 'windows-x86_64-baseline'\n"
                f"        shell: {trusted_windows_shell}\n"
                "        timeout-minutes: 15\n"
                f"{trusted_windows_env}"
                "        run: |\n"
                "          Set-StrictMode -Version Latest\n"
                "          $env:PATH = $env:TRUSTED_PATH\n"
                "          $ErrorActionPreference = 'Stop'\n"
            )
        )
        self.assertNotIn("tools/check_test_inventory.py", windows_gate)
        self.assertNotIn("tools/test_inventory.json", windows_gate)
        self.assertNotIn("test-python-tooling", windows_gate)
        self.assertNotIn("--run-python-tooling-root", windows_gate)
        self.assertNotIn("unittest.defaultTestLoader.discover", windows_gate)
        self.assertNotIn('"discovered": 465', windows_gate)
        self.assertNotIn('"skipped": 98', windows_gate)
        self.assertNotIn("$testScript", windows_gate)
        self.assertNotIn("$preflightScript", windows_gate)
        for exact_completion_contract in (
            "Set-StrictMode -Version Latest",
            "$completionNonce = [guid]::NewGuid().ToString('N')",
            "$smokeOutput = @($smokeScript | & $env:TRUSTED_PYTHON -I -B - $completionNonce)",
            "$smokeExitCode = $LASTEXITCODE",
            "if ($null -eq $smokeExitCode -or $smokeExitCode -ne 0)",
            "if ($smokeOutput.Count -ne 1)",
            "if ($smokeOutput[0] -cne $expectedCompletion)",
            "if len(sys.argv) != 2:",
            "completion_nonce = sys.argv[1]",
            're.fullmatch(r"[0-9a-f]{32}", completion_nonce, flags=re.ASCII)',
            '"contract": "zynum-windows-dll-cblas-smoke"',
            '"version": 1',
            '"nonce": completion_nonce',
            "json.dumps(",
            "sort_keys=True",
            'separators=(",", ":")',
            "ensure_ascii=True",
        ):
            self.assertEqual(1, windows_gate.count(exact_completion_contract))
        self.assertEqual(2, windows_gate.count("$ErrorActionPreference = 'Stop'"))
        self.assertEqual(2, windows_gate.count("flush=True"))
        self.assertEqual(1, windows_gate.count('$expectedCompletion = \'{"case_ids"'))
        self.assertNotIn("-I -B -c", windows_gate)
        for ffi_contract in (
            "cblas_daxpy.argtypes = [",
            "cblas_daxpy.restype = None",
            "cblas_dgemv.argtypes = [",
            "cblas_dgemv.restype = None",
            "cblas_dgemm.argtypes = [",
            "cblas_dgemm.restype = None",
        ):
            self.assertEqual(1, windows_gate.count(ffi_contract))
        for case_id in (
            "cblas-daxpy-l1",
            "cblas-dgemv-l2",
            "cblas-dgemm-l3",
        ):
            self.assertEqual(3, windows_gate.count(f'"{case_id}"'))
        for exact_result in (
            "return list(y), [6.0, 1.0, 0.0]",
            "return list(y), [5.0, 11.0]",
            "return list(matrix_c), [58.0, 64.0, 139.0, 154.0]",
            "all(math.isfinite(value) for value in observed)",
            "and observed == expected",
            '"executed": 3',
            '"passed": 3',
            '"failed": 0',
            '"skipped": 0',
            "for case_id, runner in cases:",
            "observed_case_ids != expected_case_ids",
            "len(case_results) != len(expected_case_ids)",
            "summary != expected_summary",
        ):
            self.assertIn(exact_result, windows_gate)
        source_checker_index = text.index(
            source_checker, text.index("  source-checks:\n")
        )
        self.assertLess(source_checker_index, text.index("  target-tests:\n"))
        self.assertTrue(
            windows_layout.startswith(
                "      - name: Check Windows library layout and tooling fixture boundary\n"
                "        if: runner.os == 'Windows' && matrix.cache_target == 'windows-x86_64-baseline'\n"
                f"        shell: {trusted_windows_shell}\n"
                f"{trusted_windows_env}"
                "        run: |\n"
                "          Set-StrictMode -Version Latest\n"
                "          $env:PATH = $env:TRUSTED_PATH\n"
                "          $ErrorActionPreference = 'Stop'\n"
            )
        )
        for exact_layout_contract in (
            "$dynamicLibrary = 'zig-out/bin/zynum_blas.dll'",
            "$importLibrary = 'zig-out/lib/zynum_blas.lib'",
            "$staticLibrary = 'zig-out/lib/static/zynum_blas.lib'",
            "Sort-Object -Unique).Count -ne 3",
            "$importMembers = @(& $env:TRUSTED_ZIG ar t $importLibrary)",
            "$staticMembers = @(& $env:TRUSTED_ZIG ar t $staticLibrary)",
            "'bench-zynum-blas.exe'",
            "'gemm-sweep.exe'",
            "'vector-matrix-sweep.exe'",
            "'level1-probe.exe'",
            "'dcopy-probe.exe'",
            "Get-ChildItem -LiteralPath (Split-Path -Parent $dynamicLibrary) -Filter '*.dll' -File -Force",
        ):
            self.assertIn(exact_layout_contract, windows_layout)
        expected_capability_command = (
            "      - name: Compile enabled Level 2 width production-artifact probe\n"
            "        if: matrix.cache_target == 'x86_64-v4'\n"
            f"        shell: {trusted_posix_shell}\n"
            f"{trusted_posix_env}"
            "        run: >-\n"
            "          zig build build-level2-width-enabled-artifact\n"
            "          ${{ matrix.target_args }}\n"
            "          -Dlevel2-width-candidates=true\n"
            "          --release=fast --summary failures"
        )
        capability_command = named_step(
            text, "Compile enabled Level 2 width production-artifact probe"
        )
        self.assertEqual(expected_capability_command, capability_command)
        build_id = (
            "workflow-launch:.github/workflows/ci.yml:target-tests:"
            "build-windows-python-tooling-executable-fixtures-and-libraries"
        )
        layout_id = (
            "workflow-launch:.github/workflows/ci.yml:target-tests:"
            "check-windows-library-layout-and-tooling-fixture-boundary"
        )
        gate_id = (
            "workflow-launch:.github/workflows/ci.yml:target-tests:"
            "run-windows-dll-abi-and-cblas-l1-l3-compatibility-smoke-not-inventory-evidence"
        )
        host_tool_id = (
            "workflow-launch:.github/workflows/ci.yml:target-tests:"
            "run-host-tool-smoke-once"
        )
        launches = CHECKER._discover_workflow_launches(self.root)
        build = next(item for item in launches if item["id"] == build_id)
        layout = next(item for item in launches if item["id"] == layout_id)
        gate = next(item for item in launches if item["id"] == gate_id)
        host_tool = next(item for item in launches if item["id"] == host_tool_id)
        self.assertEqual("workflow-launch", build["category"])
        self.assertEqual("run", build["call"])
        self.assertEqual(
            {
                "file": ".github/workflows/ci.yml",
                "enclosing_function": "target-tests",
                "symbol": "Build Windows Python tooling executable fixtures and libraries",
                "ordinal": 5,
            },
            build["anchor"],
        )
        self.assertEqual("workflow-launch", layout["category"])
        self.assertEqual("run", layout["call"])
        self.assertEqual(
            {
                "file": ".github/workflows/ci.yml",
                "enclosing_function": "target-tests",
                "symbol": "Check Windows library layout and tooling fixture boundary",
                "ordinal": 6,
            },
            layout["anchor"],
        )
        self.assertEqual("workflow-launch", gate["category"])
        self.assertEqual("run", gate["call"])
        self.assertEqual(
            {
                "file": ".github/workflows/ci.yml",
                "enclosing_function": "target-tests",
                "symbol": windows_gate_name,
                "ordinal": 7,
            },
            gate["anchor"],
        )
        self.assertEqual("workflow-launch", host_tool["category"])
        self.assertEqual("run", host_tool["call"])
        self.assertEqual(
            {
                "file": ".github/workflows/ci.yml",
                "enclosing_function": "target-tests",
                "symbol": "Run host tool smoke once",
                "ordinal": 8,
            },
            host_tool["anchor"],
        )
        source_checker_id = "workflow-launch:.github/workflows/ci.yml:source-checks:check-build-inventory"
        capability_id = (
            "workflow-launch:.github/workflows/ci.yml:capability-builds:"
            "compile-enabled-level-2-width-production-artifact-probe"
        )
        discovered_ids = {item["id"] for item in launches}
        self.assertTrue(
            {
                source_checker_id,
                build_id,
                layout_id,
                gate_id,
                host_tool_id,
                capability_id,
            }
            <= discovered_ids
        )
        for identifier in (
            source_checker_id,
            build_id,
            layout_id,
            gate_id,
            host_tool_id,
            capability_id,
        ):
            self.assertEqual(
                "workflow",
                CHECKER._new_test_inventory_workflow_launch(identifier)["launch_class"],
            )

        original = "          zig build generate-headers --summary failures"
        self.assertIn(original, text)
        path.write_text(
            text.replace(original, "          echo fixture-no-op", 1),
            encoding="utf-8",
        )
        self._assert_error_contains(
            "workflow_source_digests must exactly match every normalized workflow run step"
        )

        mutations = (
            (
                "runner.os == 'Windows' && matrix.cache_target == 'windows-x86_64-baseline'",
                "runner.os == 'Windows'",
                build_id,
            ),
            (" build-rank-k-probe", "", build_id),
            (
                "& $env:TRUSTED_ZIG build install-libraries",
                "& $env:TRUSTED_ZIG build",
                build_id,
            ),
            (
                "$importLibrary = 'zig-out/lib/zynum_blas.lib'",
                "$importLibrary = 'zig-out/lib/static/zynum_blas.lib'",
                layout_id,
            ),
            (
                "$staticLibrary = 'zig-out/lib/static/zynum_blas.lib'",
                "$staticLibrary = 'zig-out/lib/zynum_blas.lib'",
                layout_id,
            ),
            (
                "& $env:TRUSTED_ZIG ar t $importLibrary",
                "& $env:TRUSTED_ZIG ar t $staticLibrary",
                layout_id,
            ),
            ("-Filter '*.dll' -File -Force", "-Filter '*.dll' -File", layout_id),
            ("          $env:PYTHONHOME = $null\n", "", gate_id),
            ("          Set-StrictMode -Version Latest\n", "", gate_id),
            ("          $ErrorActionPreference = 'Stop'\n", "", gate_id),
            (
                "$completionNonce = [guid]::NewGuid().ToString('N')",
                "$completionNonce = '00000000000000000000000000000000'",
                gate_id,
            ),
            ("if len(sys.argv) != 2:", "if len(sys.argv) < 2:", gate_id),
            (
                're.fullmatch(r"[0-9a-f]{32}", completion_nonce, flags=re.ASCII)',
                're.fullmatch(r".*", completion_nonce)',
                gate_id,
            ),
            (
                "repository_root = Path.cwd().resolve(strict=True)",
                "repository_root = Path.cwd().resolve()",
                gate_id,
            ),
            (
                "& stat.FILE_ATTRIBUTE_REPARSE_POINT",
                "& 0",
                gate_id,
            ),
            (
                "or requested_stat.st_nlink != 1",
                "or requested_stat.st_nlink < 1",
                gate_id,
            ),
            ("winmode=0x00000900", "winmode=0", gate_id),
            ("kernel32.GetModuleFileNameW", "kernel32.GetModuleHandleW", gate_id),
            (
                "len(export_names) != 311 or len(set(export_names)) != 311",
                "len(export_names) < 1",
                gate_id,
            ),
            (
                "if not get_proc_address(library._handle, encoded_name)",
                "if False",
                gate_id,
            ),
            (
                'symbol_pattern = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$", flags=re.ASCII)',
                'symbol_pattern = re.compile(r".*")',
                gate_id,
            ),
            ('or "\\x00" in name', 'or "" in name', gate_id),
            (
                "or any(ord(character) < 0x20 or ord(character) == 0x7F for character in name)",
                "or False",
                gate_id,
            ),
            (
                "if len(set(encoded_export_names)) != 311:",
                "if len(encoded_export_names) != 311:",
                gate_id,
            ),
            (
                '              "cblas-daxpy-l1",\n              "cblas-dgemv-l2",\n',
                '              "cblas-dgemv-l2",\n              "cblas-daxpy-l1",\n',
                gate_id,
            ),
            (
                '              ("cblas-dgemm-l3", run_cblas_dgemm),',
                '              ("cblas-dgemm-l4", run_cblas_dgemm),',
                gate_id,
            ),
            (
                "return list(y), [6.0, 1.0, 0.0]",
                "return list(y), [6.0, 1.0, 1.0]",
                gate_id,
            ),
            (
                "return list(y), [5.0, 11.0]",
                "return list(y), [5.0, 10.0]",
                gate_id,
            ),
            (
                "return list(matrix_c), [58.0, 64.0, 139.0, 154.0]",
                "return list(matrix_c), [58.0, 64.0, 139.0, 153.0]",
                gate_id,
            ),
            (
                "all(math.isfinite(value) for value in observed)",
                "all(True for value in observed)",
                gate_id,
            ),
            (
                "and observed == expected",
                "and True",
                gate_id,
            ),
            (
                "for case_id, runner in cases:",
                "for case_id, runner in cases[:2]:",
                gate_id,
            ),
            ('"executed": 3', '"executed": 2', gate_id),
            (
                "$smokeOutput = @($smokeScript | & $env:TRUSTED_PYTHON -I -B - $completionNonce)",
                "$smokeOutput = @($smokeScript | & $env:TRUSTED_PYTHON -B - $completionNonce)",
                gate_id,
            ),
            ("$smokeExitCode = $LASTEXITCODE", "$smokeExitCode = 0", gate_id),
            (
                "$null -eq $smokeExitCode -or $smokeExitCode -ne 0",
                "$smokeExitCode -ne 0",
                gate_id,
            ),
            ("$smokeOutput.Count -ne 1", "$smokeOutput.Count -lt 1", gate_id),
            (
                "$smokeOutput[0] -cne $expectedCompletion",
                "$smokeOutput[0] -notlike '*passed*'",
                gate_id,
            ),
            ("sort_keys=True", "sort_keys=False", gate_id),
            ('separators=(",", ":")', 'separators=(", ", ": ")', gate_id),
            ("ensure_ascii=True", "ensure_ascii=False", gate_id),
            (
                '"nonce": completion_nonce',
                '"nonce": "accepted-without-binding"',
                gate_id,
            ),
            (
                "              flush=True,\n          )\n          '@",
                "          )\n          '@",
                gate_id,
            ),
            (
                "zig build test-host-tool-smoke ${{ matrix.target_args }}",
                "zig build test ${{ matrix.target_args }}",
                host_tool_id,
            ),
            (
                "          -Dlevel2-width-candidates=true\n",
                "",
                capability_id,
            ),
        )
        launch_baselines = {
            item["id"]: item for item in (build, layout, gate, host_tool)
        }
        launch_baselines[capability_id] = next(
            item for item in launches if item["id"] == capability_id
        )
        step_baselines = {
            build_id: windows_build,
            layout_id: windows_layout,
            gate_id: windows_gate,
            host_tool_id: host_tool_step,
            capability_id: capability_command,
        }
        for before, after, identifier in mutations:
            with self.subTest(windows_gate_mutation=before):
                baseline_step = step_baselines[identifier]
                self.assertIn(before, baseline_step)
                mutated_step = baseline_step.replace(before, after, 1)
                self.assertNotEqual(baseline_step, mutated_step)
                path.write_text(
                    text.replace(baseline_step, mutated_step, 1), encoding="utf-8"
                )
                mutated_launches = CHECKER._discover_workflow_launches(self.root)
                mutated = next(
                    item for item in mutated_launches if item["id"] == identifier
                )
                self.assertNotEqual(
                    launch_baselines[identifier]["source_digest"],
                    mutated["source_digest"],
                )
        path.write_text(text.replace(source_checker, "", 1), encoding="utf-8")
        self.assertNotIn(
            source_checker_id,
            {item["id"] for item in CHECKER._discover_workflow_launches(self.root)},
        )
        self._assert_error_contains(
            "workflow_source_digests must exactly match every normalized workflow run step"
        )

    def test_multiline_workflow_command_drift_fails(self) -> None:
        path = self.root / ".github/workflows/ci.yml"
        text = path.read_text(encoding="utf-8")
        original = 'status="$(git status --porcelain --untracked-files=all -- include/zynum/blas include/zynum/checked src/checked/abi_types.zig src/checked/exports.zig src/checked/exports.def docs/kernel_coverage.json)"'
        self.assertIn(original, text)
        path.write_text(text.replace(original, 'status=""', 1), encoding="utf-8")
        self._assert_error_contains(
            "workflow_source_digests must exactly match every normalized workflow run step"
        )

    def test_workflow_source_digest_cannot_be_deleted(self) -> None:
        inventory = self._inventory()
        inventory["workflow_source_digests"].pop(
            "workflow-launch:.github/workflows/ci.yml:source-checks:check-generated-files-are-up-to-date"
        )
        self._write_inventory(inventory)
        self._assert_error_contains(
            "workflow_source_digests must exactly match every normalized workflow run step"
        )

    def test_unlisted_generated_target_fails_closed(self) -> None:
        path = self.root / "tools/generate_compat_headers.zig"
        text = path.read_text(encoding="utf-8")
        insertion = '    try writeGeneratedFile(allocator, io, root, "include/zynum/blas/fixture.h", "fixture");\n\n'
        path.write_text(
            text.replace(
                "    var stdout_buffer", insertion + "    var stdout_buffer", 1
            ),
            encoding="utf-8",
        )
        self._assert_error_contains("generated-target:include/zynum/blas/fixture.h")

    def test_bad_derived_class_and_owner_fail(self) -> None:
        inventory = self._inventory()
        mutated = copy.deepcopy(inventory)
        mutated["derived_candidates"][0]["class"] = "ordinary-generated-file"
        mutated["derived_candidates"][1]["owner"] = "unknown-owner"
        self._write_inventory(mutated)
        errors = "\n".join(CHECKER.validate(self.root, self.inventory_path))
        self.assertIn("invalid derived class", errors)
        self.assertIn("invalid owner", errors)

    def test_nonexistent_additive_derived_candidate_fails(self) -> None:
        inventory = self._inventory()
        inventory["derived_candidates"].append(
            {
                "id": "derived:does/not/exist.fake",
                "path": "does/not/exist.fake",
                "class": "non-generated-source",
                "owner": "build-composition",
                "tracking_status": "tracked",
                "discovery_basis": "fixture",
            }
        )
        self._write_inventory(inventory)
        self._assert_error_contains(
            "derived candidate path must belong to the public universe"
        )

    def test_additive_existing_derived_candidate_requires_review(self) -> None:
        inventory = self._inventory()
        inventory["derived_candidates"].append(
            {
                "id": "derived:build.zig",
                "path": "build.zig",
                "class": "non-generated-source",
                "owner": "build-composition",
                "tracking_status": "tracked",
                "discovery_basis": "fabricated fixture provenance",
            }
        )
        self._write_inventory(inventory)
        self._assert_error_contains("derived_candidates reviewed fact set changed")

    def test_wrong_option_default_and_surface_fail(self) -> None:
        inventory = self._inventory()
        apple_amx_option = next(
            item
            for item in inventory["option_surfaces"]
            if item["id"] == "option-surface:build.zig:apple-amx"
        )
        self.assertEqual(
            (
                "bool",
                "false",
                "Enable the experimental private Apple AMX ISA on a validated AArch64 macOS deployment",
            ),
            (
                apple_amx_option["type"],
                apple_amx_option["default"],
                apple_amx_option["description"],
            ),
        )
        self.assertEqual(
            "option:build.zig:build:apple-amx",
            apple_amx_option["source_observation"],
        )
        self.assertIn("explicit opt-in", apple_amx_option["role"])
        self.assertIn("AArch64 macOS", apple_amx_option["conflict"])

        option = next(
            item
            for item in inventory["option_surfaces"]
            if item["name"] == "test-optimize"
        )
        option["default"] = "debug"
        inventory["option_surfaces"] = [
            item
            for item in inventory["option_surfaces"]
            if item["id"] != "option-surface:build.zig:apple-amx"
        ]
        self._write_inventory(inventory)
        errors = "\n".join(CHECKER.validate(self.root, self.inventory_path))
        self.assertIn(
            "build.zig option surfaces must contain exactly 43 standard/project surfaces",
            errors,
        )
        self.assertIn("default does not match source", errors)

    def test_root_release_and_example_optimize_surfaces(self) -> None:
        inventory = self._inventory()
        releases = {
            item["build_root"]: item
            for item in inventory["option_surfaces"]
            if item["name"] == "release"
        }
        self.assertEqual({"build.zig"}, set(releases))
        self.assertEqual(
            ("bool", "false"),
            (releases["build.zig"]["type"], releases["build.zig"]["default"]),
        )
        self.assertIn("preferred fast", releases["build.zig"]["resolution_note"])
        example = next(
            item
            for item in inventory["option_surfaces"]
            if item["id"] == "option-surface:examples/zig/build.zig:optimize"
        )
        self.assertEqual(
            ("std.lang.Optimize", "debug"), (example["type"], example["default"])
        )
        self.assertEqual(
            ["debug", "safe", "fast", "small"],
            example["value_domain"],
        )

    def test_isolated_production_and_test_optimization_sources(self) -> None:
        inventory = self._inventory()
        artifacts = {
            item["anchor"]["symbol"]: item
            for item in inventory["build_observations"]
            if item.get("isolated_library")
        }
        families = ("stride2", "compact_triangular", "level2_width", "structured")
        for family in families:
            self.assertEqual(
                "optimize", artifacts[f"{family}_isolated_library"]["optimize_source"]
            )
            self.assertEqual(
                "test-optimize",
                artifacts[f"{family}_isolated_test_library"]["optimize_source"],
            )
        artifacts["structured_isolated_test_library"]["optimize_source"] = "optimize"
        self._write_inventory(inventory)
        self._assert_error_contains("incorrect isolated optimize_source")

    def test_no_reviewed_step_is_an_intentional_orphan(self) -> None:
        inventory = self._inventory()
        orphan_ids = {
            item["id"]
            for item in inventory["build_observations"]
            if item.get("category") == "step" and item["intentional_orphan"]
        }
        self.assertEqual(set(), orphan_ids)
        generator = next(
            item
            for item in inventory["build_observations"]
            if item["id"] == "step:build.zig:build:generate-headers"
        )
        self.assertEqual("standalone-generator", generator["step_role"])
        generator["intentional_orphan"] = True
        self._write_inventory(inventory)
        self._assert_error_contains("no reviewed build step is an intentional orphan")

    def test_missing_compile_root_and_output_fail(self) -> None:
        inventory = self._inventory()
        artifact = next(
            item
            for item in inventory["build_observations"]
            if item["id"] == "compile:build.zig:build:bench"
        )
        artifact["root_source"] = []
        artifact["produced_outputs"] = []
        self._write_inventory(inventory)
        errors = "\n".join(CHECKER.validate(self.root, self.inventory_path))
        self.assertIn("root_source must be non-empty", errors)
        self.assertIn("produced_outputs must be non-empty", errors)

    def test_wrong_link_provider_fails(self) -> None:
        inventory = self._inventory()
        edge = next(
            item
            for item in inventory["build_observations"]
            if item.get("category") == "link"
        )
        edge["provider"] = "wrong_provider"
        self._write_inventory(inventory)
        self._assert_error_contains("wrong link provider")

    def test_anonymous_install_fails(self) -> None:
        inventory = self._inventory()
        install = next(
            item
            for item in inventory["build_observations"]
            if item.get("category") == "install"
        )
        install["id"] = "install:build.zig:build:anonymous"
        self._write_inventory(inventory)
        self._assert_error_contains("anonymous install identity is forbidden")

    def test_missing_svg_provenance_fails(self) -> None:
        inventory = self._inventory()
        candidate = self._as_curated_svg_asset(inventory)
        del candidate["claim_scope"]
        self._write_inventory(inventory)
        self._assert_error_contains("missing curated provenance field claim_scope")

    def test_generated_artifact_gate_ids_must_resolve(self) -> None:
        inventory = self._inventory()
        artifact = next(
            item
            for item in inventory["derived_candidates"]
            if item["id"] == "derived:include/zynum/blas/blas.h"
        )
        artifact["deterministic_drift_gate_ids"] = ["workflow-launch:missing-fixture"]
        artifact["consumer_gate_ids"] = []
        self._write_inventory(inventory)
        errors = "\n".join(CHECKER.validate(self.root, self.inventory_path))
        self.assertIn("unknown deterministic drift gate id", errors)
        self.assertIn(
            "tracked reproducible artifact requires a consumer gate id", errors
        )

    def test_curated_review_date_must_be_an_iso_calendar_date(self) -> None:
        inventory = self._inventory()
        candidate = self._as_curated_svg_asset(inventory)
        candidate["review_date"] = "2026-02-31"
        self._write_inventory(inventory)
        self._assert_error_contains("review_date must be an ISO calendar date")

    def test_python_import_aliases_are_discovered(self) -> None:
        path = self.root / "tools/check_abi_manifest.py"
        with path.open("a", encoding="utf-8") as output:
            output.write(
                "\nimport subprocess as sp\n"
                "from subprocess import run\n"
                "def alias_fixture():\n"
                "    sp.run(['python3', '-V'])\n"
                "    run(['python3', '-V'])\n"
            )
        errors = "\n".join(CHECKER.validate(self.root, self.inventory_path))
        self.assertIn(
            "python-launch:tools/check_abi_manifest.py:alias_fixture:subprocess.run:1",
            errors,
        )
        self.assertIn(
            "python-launch:tools/check_abi_manifest.py:alias_fixture:subprocess.run:2",
            errors,
        )

    def test_function_scoped_subprocess_module_alias_is_discovered(self) -> None:
        path = self.root / "tools/check_abi_manifest.py"
        with path.open("a", encoding="utf-8") as output:
            output.write(
                "\ndef scoped_module_alias_fixture():\n"
                "    import subprocess as sp\n"
                "    return sp.run(['python3', '-V'])\n"
            )
        self._assert_error_contains(
            "python-launch:tools/check_abi_manifest.py:scoped_module_alias_fixture:subprocess.run:1"
        )

    def test_alias_launch_before_later_shadow_is_discovered(self) -> None:
        path = self.root / "tools/check_abi_manifest.py"
        with path.open("a", encoding="utf-8") as output:
            output.write(
                "\ndef alias_before_shadow_fixture():\n"
                "    import subprocess as sp\n"
                "    sp.run(['python3', '-V'])\n"
                "    sp = None\n"
            )
        self._assert_error_contains(
            "python-launch:tools/check_abi_manifest.py:alias_before_shadow_fixture:subprocess.run:1"
        )

    def test_module_late_bound_subprocess_callable_is_discovered(self) -> None:
        path = self.root / "tools/check_abi_manifest.py"
        with path.open("a", encoding="utf-8") as output:
            output.write(
                "\ndef module_late_binding_fixture():\n"
                "    return late_runner(['python3', '-V'])\n"
                "from subprocess import run as late_runner\n"
            )
        self._assert_error_contains(
            "python-launch:tools/check_abi_manifest.py:module_late_binding_fixture:subprocess.run:1"
        )

    def test_nested_closure_late_bound_subprocess_callable_is_discovered(self) -> None:
        path = self.root / "tools/check_abi_manifest.py"
        with path.open("a", encoding="utf-8") as output:
            output.write(
                "\ndef outer_late_binding_fixture():\n"
                "    def nested_late_binding_fixture():\n"
                "        return closure_runner(['python3', '-V'])\n"
                "    from subprocess import run as closure_runner\n"
                "    return nested_late_binding_fixture\n"
            )
        self._assert_error_contains(
            "python-launch:tools/check_abi_manifest.py:nested_late_binding_fixture:subprocess.run:1"
        )

    def test_subprocess_callable_default_parameter_is_discovered(self) -> None:
        path = self.root / "tools/check_abi_manifest.py"
        with path.open("a", encoding="utf-8") as output:
            output.write(
                "\ndef default_parameter_fixture(runner=subprocess.run):\n"
                "    return runner(['python3', '-V'])\n"
            )
        self._assert_error_contains(
            "python-launch:tools/check_abi_manifest.py:default_parameter_fixture:subprocess.run:1"
        )

    def test_function_scoped_subprocess_callable_import_is_discovered(self) -> None:
        path = self.root / "tools/check_abi_manifest.py"
        with path.open("a", encoding="utf-8") as output:
            output.write(
                "\ndef scoped_callable_import_fixture():\n"
                "    from subprocess import run\n"
                "    return run(['python3', '-V'])\n"
            )
        self._assert_error_contains(
            "python-launch:tools/check_abi_manifest.py:scoped_callable_import_fixture:subprocess.run:1"
        )

    def test_simple_subprocess_callable_assignment_is_discovered(self) -> None:
        path = self.root / "tools/check_abi_manifest.py"
        with path.open("a", encoding="utf-8") as output:
            output.write(
                "\ndef callable_assignment_fixture():\n"
                "    runner = subprocess.run\n"
                "    return runner(['python3', '-V'])\n"
            )
        self._assert_error_contains(
            "python-launch:tools/check_abi_manifest.py:callable_assignment_fixture:subprocess.run:1"
        )

    def test_chained_subprocess_module_assignments_are_discovered(self) -> None:
        path = self.root / "tools/check_abi_manifest.py"
        with path.open("a", encoding="utf-8") as output:
            output.write(
                "\ndef chained_module_alias_fixture():\n"
                "    proc = subprocess\n"
                "    p2 = proc\n"
                "    return p2.run(['python3', '-V'])\n"
            )
        self._assert_error_contains(
            "python-launch:tools/check_abi_manifest.py:chained_module_alias_fixture:subprocess.run:1"
        )

    def test_destructured_subprocess_callable_assignments_are_discovered(self) -> None:
        path = self.root / "tools/check_abi_manifest.py"
        with path.open("a", encoding="utf-8") as output:
            output.write(
                "\ndef tuple_alias_fixture():\n"
                "    runner, unused = subprocess.run, None\n"
                "    return runner(['python3', '-V'])\n"
                "\ndef list_alias_fixture():\n"
                "    [runner] = [subprocess.run]\n"
                "    return runner(['python3', '-V'])\n"
            )
        errors = "\n".join(CHECKER.validate(self.root, self.inventory_path))
        self.assertIn(
            "python-launch:tools/check_abi_manifest.py:tuple_alias_fixture:subprocess.run:1",
            errors,
        )
        self.assertIn(
            "python-launch:tools/check_abi_manifest.py:list_alias_fixture:subprocess.run:1",
            errors,
        )

    def test_unsupported_process_alias_expressions_fail_closed(self) -> None:
        path = self.root / "tools/check_abi_manifest.py"
        baseline = path.read_text(encoding="utf-8")
        fixtures = (
            "\ndef conditional_alias_fixture(flag):\n"
            "    runner = subprocess.run if flag else None\n"
            "    return runner(['python3', '-V'])\n",
            "\ndef named_expression_fixture():\n"
            "    (runner := subprocess.run)\n"
            "    return runner(['python3', '-V'])\n",
            "\nfrom subprocess import *\n",
        )
        for fixture in fixtures:
            with self.subTest(fixture=fixture.splitlines()[1]):
                path.write_text(baseline + fixture, encoding="utf-8")
                self._assert_error_contains("source discovery failed closed")
        path.write_text(baseline, encoding="utf-8")

    def test_control_flow_dependent_process_alias_shadow_fails_closed(self) -> None:
        path = self.root / "tools/check_abi_manifest.py"
        with path.open("a", encoding="utf-8") as output:
            output.write(
                "\ndef conditional_shadow_fixture():\n"
                "    runner = subprocess.run\n"
                "    if False:\n"
                "        runner = print\n"
                "    return runner(['python3', '-V'])\n"
            )
        errors = "\n".join(CHECKER.validate(self.root, self.inventory_path))
        self.assertIn("source discovery failed closed", errors)
        self.assertIn(
            "control-flow-dependent process alias assignment is unsupported", errors
        )

    def test_zig_comments_and_strings_do_not_create_calls(self) -> None:
        inventory = self._inventory()
        baseline_observations = {
            item["id"]: item
            for item in CHECKER.discover(self.root, inventory)["build_observations"]
        }
        checked_fortran_id = "launch:build.zig:build:run_checked_fortran_smoke"
        checked_fortran = baseline_observations[checked_fortran_id]
        self.assertEqual("std.Build.Step.Run.create", checked_fortran["call"])
        self.assertEqual(
            CHECKER.CHECKED_ABI_BUILD_CONTRACT_SHA256,
            checked_fortran["source_digest"],
        )
        self.assertFalse(checked_fortran["native_inventory_evidence"])
        checked_fortran_contract = next(
            item for item in inventory["build_observations"]
            if item["id"] == checked_fortran_id
        )
        self.assertEqual("generated-executable", checked_fortran_contract["command_shape"])
        self.assertEqual(
            ["<generated checked Fortran executable>"],
            checked_fortran_contract["argv_shape"],
        )
        self.assertEqual(
            "supplemental-foreign-language-consumer",
            checked_fortran_contract["launch_class"],
        )
        self._append_to_build(
            'const harmless = "b.option(bool, \\"not-an-option\\", \\"fixture\\")"; // b.step("not-a-step", "fixture")'
        )
        observed = {
            item["id"]: item
            for item in CHECKER.discover(self.root, inventory)["build_observations"]
        }
        self.assertEqual(set(baseline_observations), set(observed))
        # Run.create is bound to the exact checked block. Appending text outside
        # that block preserves this launch while the full build-file pin rejects.
        self.assertEqual(checked_fortran, observed[checked_fortran_id])
        errors = CHECKER.validate(self.root, self.inventory_path)
        expected_digest_errors = {
            "build_root_digests: full source changed for build.zig",
            *(
                "build_observations: source field source_digest changed for "
                + item["id"]
                for item in inventory["build_observations"]
                if item["anchor"]["file"] == "build.zig"
                and item["anchor"]["enclosing_function"] == "build"
                and "source_digest" in item
                and item["call"] != "implicit"
                and item["id"] != checked_fortran_id
            ),
        }
        self.assertEqual(
            expected_digest_errors,
            set(errors),
        )

    def test_zig_build_receiver_parameter_and_safe_alias_cover_all_call_kinds(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory(prefix="build-receiver-fixture-") as directory:
            root = Path(directory)
            (root / "build.zig").write_text(
                'const std = @import("std");\n'
                "pub fn build(builder: *std.Build) void {\n"
                "    const alias = builder;\n"
                '    const enabled = alias.option(bool, "enabled", "fixture");\n'
                "    const target = builder.standardTargetOptions(.{});\n"
                "    const optimize = alias.standardOptimizeOption(.{});\n"
                "    const module = alias.createModule(.{});\n"
                '    const library = alias.addLibrary(.{ .name = "lib", .root_module = module });\n'
                '    const executable = alias.addExecutable(.{ .name = "exe", .root_module = module });\n'
                '    const tests = alias.addTest(.{ .name = "tests", .root_module = module });\n'
                "    const run = alias.addRunArtifact(executable);\n"
                '    const tool = alias.addSystemCommand(&.{"tool"});\n'
                '    const fixture_step = alias.step("fixture", "fixture");\n'
                "    alias.installArtifact(executable);\n"
                "    const install = alias.addInstallArtifact(executable, .{});\n"
                '    alias.installFile("source", "destination");\n'
                "    executable.linkLibrary(library);\n"
                "    _ = .{ enabled, target, optimize, tests, run, tool, fixture_step, install };\n"
                "}\n",
                encoding="utf-8",
            )
            observations = CHECKER._discover_build_root(root, "build.zig")
            option_semantics = CHECKER._discover_option_surface_semantics(root)
        self.assertEqual(
            {
                "option",
                "step",
                "compile",
                "launch",
                "install",
                "link",
            },
            {item["category"] for item in observations},
        )
        self.assertEqual(
            {
                "b.option",
                "b.step",
                "b.addLibrary",
                "b.addExecutable",
                "b.addTest",
                "b.addRunArtifact",
                "b.addSystemCommand",
                "b.installArtifact",
                "b.addInstallArtifact",
                "b.installFile",
                "linkLibrary",
                "b.standardTargetOptions",
                "b.standardOptimizeOption",
            },
            {item["call"] for item in observations},
        )
        self.assertEqual(
            {
                "type": "bool",
                "default": "unset",
                "description": "fixture",
            },
            option_semantics[("build.zig", "enabled")],
        )

    def test_build_root_reuses_one_zig_context_for_all_observations(self) -> None:
        source = (
            'const std = @import("std");\n'
            "pub fn build(b: *std.Build) void {\n"
            "    const module = b.createModule(.{});\n"
            '    const library = b.addLibrary(.{ .name = "lib", .root_module = module });\n'
            '    const executable = b.addExecutable(.{ .name = "exe", .root_module = module });\n'
            '    const tests = b.addTest(.{ .name = "tests", .root_module = module });\n'
            "    _ = .{ library, executable, tests };\n"
            "}\n"
        )
        with self.subTest(tooling_contract="absent"):
            with tempfile.TemporaryDirectory(
                prefix="build-context-count-"
            ) as directory:
                root = Path(directory)
                (root / "build.zig").write_text(source, encoding="utf-8")
                with mock.patch.object(
                    CHECKER,
                    "_zig_build_context",
                    wraps=CHECKER._zig_build_context,
                ) as build_context:
                    observations = CHECKER._discover_build_root(root, "build.zig")
            self.assertEqual(3, len(observations))
            build_context.assert_called_once()

        tooling_source_sentinels = (
            "const python_tooling_tests = undefined;",
            "const python_tooling_test_step = undefined;",
        )
        for declaration in tooling_source_sentinels:
            with self.subTest(tooling_contract="source-only", declaration=declaration):
                with self.assertRaisesRegex(
                    CHECKER.InventoryError,
                    "Python tooling test observations are incomplete",
                ):
                    CHECKER._annotate_python_tooling_tests(declaration, [])

        required_ids = (
            CHECKER.PYTHON_TOOLING_LAUNCH_ID,
            CHECKER.PYTHON_TOOLING_STEP_ID,
            CHECKER.PYTHON_TOOLING_STRUCTURE_BARRIER_ID,
            CHECKER.TEST_INVENTORY_AGGREGATE_STEP_ID,
        )
        tooling_source = "const python_tooling_tests = undefined;"
        for missing_id in required_ids:
            with self.subTest(tooling_contract="missing-one", missing_id=missing_id):
                partial_observations = [
                    {"id": identifier}
                    for identifier in required_ids
                    if identifier != missing_id
                ]
                with self.assertRaisesRegex(
                    CHECKER.InventoryError,
                    "Python tooling test observations are incomplete",
                ):
                    CHECKER._annotate_python_tooling_tests(
                        tooling_source, partial_observations
                    )

    def test_reused_zig_context_preserves_complete_build_observations(self) -> None:
        source = (
            'const std = @import("std");\n'
            "pub fn build(b: *std.Build) void {\n"
            "    const module = b.createModule(.{});\n"
            '    const library = b.addLibrary(.{ .name = "lib", .root_module = module });\n'
            '    const executable = b.addExecutable(.{ .name = "exe", .root_module = module });\n'
            '    const tests = b.addTest(.{ .name = "tests", .root_module = module });\n'
            "    _ = .{ library, executable, tests };\n"
            "}\n"
        )
        with tempfile.TemporaryDirectory(
            prefix="build-context-equivalence-"
        ) as directory:
            root = Path(directory)
            (root / "build.zig").write_text(source, encoding="utf-8")
            observations = CHECKER._discover_build_root(root, "build.zig")
        self.assertEqual(
            {
                "compile:build.zig:build:library": (
                    "2a3d2002046c0ef2fefed79f275f6a050a1a7688d51b91c7c6693dc9f5b60f62"
                ),
                "compile:build.zig:build:executable": (
                    "aaf85c9ae0cb1be87943823504b022d063c1ebfc26c6b02b807ad7e06494d13e"
                ),
                "compile:build.zig:build:tests": (
                    "adf89d215e6fccc13a955538f08134980d0e23b55573a473eea3b3b2c6a09c67"
                ),
            },
            {item["id"]: item["source_digest"] for item in observations},
        )

    def test_zig_build_receiver_shadow_and_ambiguous_alias_fail_closed(self) -> None:
        fixtures = (
            "pub fn build(builder: *std.Build) void { const alias = builder; { const alias = other; } }",
            "pub fn build(builder: *std.Build) void { const alias = if (flag) builder else other; _ = alias; }",
            'pub fn build(builder: *std.Build) void { _ = other.step("x", "x"); }',
            "pub fn build(first: *std.Build, second: *std.Build) void { _ = first; _ = second; }",
        )
        for source in fixtures:
            with (
                self.subTest(source=source),
                tempfile.TemporaryDirectory(
                    prefix="ambiguous-build-receiver-"
                ) as directory,
            ):
                root = Path(directory)
                (root / "build.zig").write_text(
                    'const std = @import("std");\n' + source + "\n",
                    encoding="utf-8",
                )
                with self.assertRaisesRegex(
                    CHECKER.InventoryError,
                    "shadowed or rebound|ambiguous|unsupported|escapes",
                ):
                    CHECKER._discover_build_root(root, "build.zig")

    def test_zig_build_receiver_escape_contexts_fail_closed(self) -> None:
        fixtures = {
            "helper-argument": "register(builder);",
            "nested-wrapper": "register(wrap(builder));",
            "alias-escape": "const alias = builder; register(alias);",
            "stored": "const stored = .{builder}; _ = stored;",
            "returned": "return builder;",
            "captured": (
                'const Holder = struct { fn hidden() void { _ = builder.step("hidden", "hidden"); } }; '
                "_ = Holder;"
            ),
        }
        prefix = (
            'const std = @import("std");\n'
            "fn register(_: anytype) void {}\n"
            "fn wrap(value: anytype) @TypeOf(value) { return value; }\n"
        )
        for name, body in fixtures.items():
            with (
                self.subTest(name=name),
                tempfile.TemporaryDirectory(
                    prefix="escaping-build-receiver-"
                ) as directory,
            ):
                root = Path(directory)
                (root / "build.zig").write_text(
                    prefix
                    + "pub fn build(builder: *std.Build) void {\n"
                    + body
                    + "\n}\n",
                    encoding="utf-8",
                )
                with self.assertRaisesRegex(
                    CHECKER.InventoryError,
                    "Zig build receiver .* escapes the analyzed build body",
                ):
                    CHECKER._discover_build_root(root, "build.zig")

    def test_zig_build_receiver_nonreceiver_arguments_and_direct_calls_are_safe(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory(prefix="direct-build-receiver-") as directory:
            root = Path(directory)
            (root / "build.zig").write_text(
                'const std = @import("std");\n'
                "fn register(_: anytype) void {}\n"
                "pub fn build(builder: *std.Build) void {\n"
                '    const harmless = "register(builder); builder.addSystemCommand";\n'
                '    // register(builder); builder.step("hidden", "hidden");\n'
                "    const target = harmless.len;\n"
                "    register(target);\n"
                '    const direct = builder.addSystemCommand(&.{"visible"});\n'
                '    const direct_step = builder.step("visible", "visible");\n'
                "    _ = .{ direct, direct_step };\n"
                "}\n",
                encoding="utf-8",
            )
            observations = CHECKER._discover_build_root(root, "build.zig")
        self.assertEqual(
            {
                "launch:build.zig:build:direct",
                "step:build.zig:build:visible",
            },
            {item["id"] for item in observations},
        )

    def test_build_receiver_helper_escape_mutation_fails_source_discovery(
        self,
    ) -> None:
        path = self.root / "build.zig"
        original = path.read_text(encoding="utf-8")
        for before, after in (
            ("libraries[index].lto = .none", "libraries[index].lto = .full"),
            ('\n        options.addOption(bool, "dynamic_dispatch", false);', '\n        options.addOption(bool, "dynamic_dispatch", true);'),
        ):
            with self.subTest(multiversion_factory=before):
                self.assertEqual(1, original.count(before))
                path.write_text(original.replace(before, after, 1), encoding="utf-8")
                self._assert_error_contains("multiversion build factory target/options/non-LTO contract drifted")
                path.write_text(original, encoding="utf-8")
        for module in ("lapack", "sparse"):
            for dependency in ("test_dynamic_step", "&test_inventory_structure_check.step", "&native_execution_check.step"):
                before = f"run_dynamic_{module}_tests.step.dependOn({dependency});"
                with self.subTest(dynamic_foundation_guard=before):
                    self.assertIn(before, original)
                    path.write_text(original.replace(before, "", 1), encoding="utf-8")
                    self._assert_error_contains("dynamic correctness two-process/generator/native guard contract drifted")
                    path.write_text(original, encoding="utf-8")
        self._append_to_build("register(b);")
        errors = "\n".join(CHECKER.validate(self.root, self.inventory_path))
        self.assertIn("source discovery failed closed", errors)
        self.assertIn("Zig build receiver 'b' escapes the analyzed build body", errors)

    def test_x86_compile_guard_mutation_hits_root_digest(self) -> None:
        path = self.root / "build.zig"
        text = path.read_text(encoding="utf-8")
        original = (
            "const stride2_isolated_library = if "
            "(target.result.cpu.arch == .x86_64) b.addObject(.{"
        )
        replacement = original.replace(".x86_64", ".aarch64")
        self.assertIn(original, text)
        path.write_text(
            text.replace(original, replacement, 1),
            encoding="utf-8",
        )
        self._assert_error_contains(
            "build_root_digests: full source changed for build.zig"
        )

    def test_compile_guard_mutation_survives_rehashed_root_digest(self) -> None:
        path = self.root / "build.zig"
        text = path.read_text(encoding="utf-8")
        original = (
            "const stride2_isolated_library = if "
            "(target.result.cpu.arch == .x86_64) b.addObject(.{"
        )
        replacement = original.replace(".x86_64", ".aarch64")
        self.assertIn(original, text)
        path.write_text(
            text.replace(original, replacement, 1),
            encoding="utf-8",
        )
        inventory = self._inventory()
        inventory["build_root_digests"]["build.zig"] = hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
        self._write_inventory(inventory)
        self._assert_error_contains(
            "build_observations: source field source_digest changed"
        )

    def test_link_guard_mutation_hits_root_digest(self) -> None:
        path = self.root / "build.zig"
        text = path.read_text(encoding="utf-8")
        original = "if (stride2_isolated_library) |library| {"
        self.assertIn(original, text)
        path.write_text(
            text.replace(
                original, "if (compact_triangular_isolated_library) |library| {", 1
            ),
            encoding="utf-8",
        )
        self._assert_error_contains(
            "build_root_digests: full source changed for build.zig"
        )

    def test_link_guard_mutation_survives_rehashed_root_digest(self) -> None:
        path = self.root / "build.zig"
        text = path.read_text(encoding="utf-8")
        original = "if (stride2_isolated_library) |library| {"
        self.assertIn(original, text)
        path.write_text(
            text.replace(
                original, "if (compact_triangular_isolated_library) |library| {", 1
            ),
            encoding="utf-8",
        )
        inventory = self._inventory()
        inventory["build_root_digests"]["build.zig"] = hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
        self._write_inventory(inventory)
        self._assert_error_contains(
            "numerics resources owner/source/target/attachment graph drifted"
        )

    def test_nested_link_guard_mutation_survives_rehashed_root_digest(self) -> None:
        path = self.root / "build.zig"
        text = path.read_text(encoding="utf-8")
        original = "if (stride2_isolated_library) |library| {"
        self.assertIn(original, text)
        path.write_text(
            text.replace(
                original,
                "if (stride2_isolated_library) |library| if (host_tool_smoke) {",
                1,
            ),
            encoding="utf-8",
        )
        inventory = self._inventory()
        inventory["build_root_digests"]["build.zig"] = hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
        self._write_inventory(inventory)
        self._assert_error_contains(
            "numerics resources owner/source/target/attachment graph drifted"
        )

    def test_inline_link_guard_mutation_survives_rehashed_root_digest(self) -> None:
        path = self.root / "build.zig"
        text = path.read_text(encoding="utf-8")
        original = "        zynum_core_mod.addObject(library);"
        replacement = (
            "        if (host_tool_smoke) zynum_core_mod.addObject(library);"
        )
        self.assertIn(original, text)
        path.write_text(text.replace(original, replacement, 1), encoding="utf-8")
        inventory = self._inventory()
        inventory["build_root_digests"]["build.zig"] = hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
        self._write_inventory(inventory)
        self._assert_error_contains(
            "numerics resources owner/source/target/attachment graph drifted"
        )

    def test_conditional_link_guard_digest_cannot_be_deleted(self) -> None:
        inventory = self._inventory()
        inventory["conditional_link_guard_digests"].pop(
            "link:build.zig:build:zynum_core_mod<-stride2_isolated_library"
        )
        self._write_inventory(inventory)
        self._assert_error_contains(
            "conditional_link_guard_digests must exactly match every conditional link edge"
        )

    def test_aggregate_guard_mutation_survives_rehashed_root_digest(self) -> None:
        path = self.root / "build.zig"
        text = path.read_text(encoding="utf-8")
        original = "if (host_tool_smoke) test_step.dependOn(host_tool_smoke_test_step);"
        self.assertIn(original, text)
        path.write_text(
            text.replace(
                original,
                "if (!host_tool_smoke) test_step.dependOn(host_tool_smoke_test_step);",
                1,
            ),
            encoding="utf-8",
        )
        inventory = self._inventory()
        inventory["build_root_digests"]["build.zig"] = hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
        self._write_inventory(inventory)
        self._assert_error_contains(
            "default correctness aggregate must preserve exclusive dynamic and canonical specialized branches"
        )

    def test_compat_install_guard_mutation_hits_root_digest(self) -> None:
        path = self.root / "build.zig"
        text = path.read_text(encoding="utf-8")
        self.assertIn("if (install_compat_headers) {", text)
        path.write_text(
            text.replace(
                "if (install_compat_headers) {", "if (!install_compat_headers) {", 1
            ),
            encoding="utf-8",
        )
        self._assert_error_contains(
            "build_root_digests: full source changed for build.zig"
        )

    def test_selector_root_and_aggregate_guard_mutations_hit_root_digest(self) -> None:
        path = self.root / "build.zig"
        baseline = path.read_text(encoding="utf-8")
        mutations = (
            (
                "src/blas/structured_object_stub_root.zig",
                "src/blas/structured_object_root.zig",
            ),
            (
                "if (host_tool_smoke) test_step.dependOn(host_tool_smoke_test_step);",
                "if (!host_tool_smoke) test_step.dependOn(host_tool_smoke_test_step);",
            ),
            ("if (run_structured_object_tests) |run|", "if (null) |run|"),
        )
        for original, replacement in mutations:
            with self.subTest(original=original):
                self.assertIn(original, baseline)
                path.write_text(
                    baseline.replace(original, replacement, 1), encoding="utf-8"
                )
                self._assert_error_contains(
                    "build_root_digests: full source changed for build.zig"
                )
        path.write_text(baseline, encoding="utf-8")

    def test_implicit_install_dependency_partition(self) -> None:
        inventory = self._inventory()
        steps = {
            item["id"]: item
            for item in inventory["build_observations"]
            if item.get("category") == "step"
        }
        root_dependencies = steps["step:build.zig:build:install"]["direct_dependencies"]
        root_ids = {item["id"] for item in root_dependencies}
        self.assertEqual(14, len(root_ids))
        checked_step_id = "step:build.zig:build:install-checked-abi"
        self.assertEqual(
            [{"id": checked_step_id, "condition": "checked-abi is true"}],
            [item for item in root_dependencies if item["id"] == checked_step_id],
        )
        checked_step = steps[checked_step_id]
        self.assertEqual("packaging", checked_step["step_role"])
        self.assertEqual("not-member", checked_step["aggregate_test_membership"])
        checked_dependencies = checked_step["direct_dependencies"]
        self.assertEqual(7, len(checked_dependencies))
        self.assertEqual(
            {
                "install:build.zig:build:checked_dynamic_install": "always",
                "install:build.zig:build:lib/static": "target.result.ofmt != .macho",
                "install:build.zig:build:libzynum_checked.a": "target.result.ofmt == .macho",
                "install:build.zig:build:include/zynum/checked/checked.h": "always",
                "install:build.zig:build:include/zynum/checked/checked.f90": "always",
                "install:build.zig:build:include/zynum/checked/abi_manifest.json": "always",
                "install:build.zig:build:pkgconfig/zynum_checked.pc": "always",
            },
            {item["id"]: item["condition"] for item in checked_dependencies},
        )
        self.assertTrue(root_ids.isdisjoint(item["id"] for item in checked_dependencies))
        self.assertNotIn("install:build.zig:build:install_rank_k_probe", root_ids)
        self.assertEqual(
            [{"id": "install:examples/zig/build.zig:build:exe", "condition": "always"}],
            steps["step:examples/zig/build.zig:build:install"]["direct_dependencies"],
        )
        self.assertEqual(
            "requested target architecture is x86_64",
            steps["step:build.zig:build:test-structured-object"]["direct_dependencies"][
                0
            ]["condition"],
        )

    def test_library_target_output_maps_and_windows_gaps(self) -> None:
        inventory = self._inventory()
        libraries = [
            item
            for item in inventory["build_observations"]
            if item.get("artifact_kind") == "library"
        ]
        self.assertTrue(
            all(
                set(item["produced_outputs_by_target"]) == {"elf", "macho", "windows"}
                for item in libraries
            )
        )
        dynamic = next(
            item for item in libraries if item["id"] == "compile:build.zig:build:lib"
        )
        self.assertEqual(
            "zynum_blas.lib",
            dynamic["produced_outputs_by_target"]["windows"]["import_library"],
        )
        static = next(
            item
            for item in libraries
            if item["id"] == "compile:build.zig:build:static_lib"
        )
        self.assertEqual(
            "zig-out/lib/static/zynum_blas.lib",
            static["install_destinations_by_target"]["windows"]["primary"],
        )
        objects = [item for item in inventory["build_observations"] if item.get("artifact_kind") == "object"]
        self.assertEqual(13, len(objects))
        self.assertEqual(8, sum(bool(item.get("isolated_library")) for item in objects))
        self.assertEqual(
            {"compile:build.zig:build:task_runtime_host_test_library", "compile:build.zig:buildKernelTiers:libraries",
             "compile:build.zig:build:lapacke_lp64_defaults", "compile:build.zig:build:lapacke_ilp64_defaults",
             "compile:build.zig:build:abplas_lp64_defaults"},
            {item["id"] for item in objects if not item.get("isolated_library")},
        )
        defaults = [item for item in objects if item["id"] in {
            "compile:build.zig:build:lapacke_lp64_defaults", "compile:build.zig:build:lapacke_ilp64_defaults",
            "compile:build.zig:build:abplas_lp64_defaults"}]
        self.assertEqual(3, len(defaults))
        for item in defaults:
            self.assertEqual(["src/interop/lapacke_default_errors.c"], item["root_source"])
            self.assertEqual([], item["install_destinations"])
        for item in objects:
            self.assertEqual("b.addObject", item["call"])
            if item.get("isolated_library"):
                self.assertEqual([], item["install_destinations"])
                self.assertTrue(
                    item["produced_outputs_by_target"]["windows"]["primary"].endswith(
                        ".obj"
                    )
                )
        gap_ids = {item["id"] for item in inventory["current_gaps"]}
        self.assertTrue(
            {
                "gap:windows-library-install-collision",
                "gap:windows-default-install-executables",
            }.isdisjoint(gap_ids)
        )
        observations = {item["id"]: item for item in inventory["build_observations"]}
        self.assertEqual(
            [
                "zig-out/bin/zynum_blas.dll",
                "zig-out/lib/zynum_blas.lib",
            ],
            observations[CHECKER.INSTALL_DYNAMIC_LIBRARY_ID]["destination"][-2:],
        )
        self.assertIn(
            "zig-out/lib/static/zynum_blas.lib",
            observations[CHECKER.INSTALL_STATIC_LIBRARY_ID]["destination"],
        )
        for identifier in CHECKER.DEFAULT_EXECUTABLE_INSTALL_IDS:
            self.assertEqual(
                "install step is reached",
                observations[identifier]["condition"],
            )

    def test_example_optimize_forwarding_gap_is_closed(self) -> None:
        inventory = self._inventory()
        self.assertNotIn("gap:example-optimize-forwarding", {item["id"] for item in inventory["current_gaps"]})
        option = next(item for item in inventory["option_surfaces"] if item["id"] == "option-surface:build.zig:optimize")
        self.assertEqual("standard release selection", option["default"])
        self.assertEqual("option:build.zig:build:optimize-override", option["source_observation"])

    def test_every_reviewed_gap_is_mandatory(self) -> None:
        baseline = self._inventory()
        for gap in baseline["current_gaps"]:
            with self.subTest(gap=gap["id"]):
                inventory = copy.deepcopy(baseline)
                inventory["current_gaps"] = [
                    item
                    for item in inventory["current_gaps"]
                    if item["id"] != gap["id"]
                ]
                self._write_inventory(inventory)
                self._assert_error_contains(
                    "current gap ids must match the complete reviewed set"
                )

    def test_required_derived_candidate_cannot_be_deleted(self) -> None:
        inventory = self._inventory()
        inventory["derived_candidates"] = [
            item
            for item in inventory["derived_candidates"]
            if item["id"] != "derived:tools/abi_baseline_observation.json"
        ]
        self._write_inventory(inventory)
        self._assert_error_contains("missing required reviewed derived details")

    def test_derived_source_class_owner_and_unique_path_are_enforced(self) -> None:
        inventory = self._inventory()
        generated = next(
            item
            for item in inventory["derived_candidates"]
            if item["path"].endswith("blas.h")
        )
        generated["class"] = "non-generated-source"
        generated["owner"] = "package-metadata"
        duplicate = copy.deepcopy(inventory["derived_candidates"][-1])
        duplicate["id"] = "derived:duplicate-fixture"
        inventory["derived_candidates"].append(duplicate)
        self._write_inventory(inventory)
        errors = "\n".join(CHECKER.validate(self.root, self.inventory_path))
        self.assertIn(
            "generated target must map to a designated reproducible candidate", errors
        )
        self.assertIn("owner must match its derived candidate", errors)
        self.assertIn("duplicate path", errors)

    def test_reviewed_derived_facts_cannot_be_coordinately_rewritten(
        self,
    ) -> None:
        inventory = self._inventory()
        kernel_coverage = next(
            item
            for item in inventory["derived_candidates"]
            if item["id"] == "derived:docs/kernel_coverage.json"
        )
        kernel_coverage.update(
            {
                "tracking_status": "current-untracked-gap",
                "package_closure_gap": "fabricated package closure gap",
            }
        )
        generated = next(
            item
            for item in inventory["derived_candidates"]
            if item["id"] == "derived:include/zynum/blas/blas.h"
        )
        generated["owner"] = "release-validation"
        generated["deterministic_drift_gate_ids"] = [
            "workflow-launch:.github/workflows/ci.yml:target-tests:test-debug-target"
        ]
        generated["consumer_gate_ids"] = ["launch:build.zig:build:run_modern_tests"]
        target = next(
            item
            for item in inventory["generator_targets"]
            if item["id"] == "generated-target:include/zynum/blas/blas.h"
        )
        target["owner"] = "release-validation"
        self._write_inventory(inventory)
        errors = "\n".join(CHECKER.validate(self.root, self.inventory_path))
        self.assertIn(
            "derived:docs/kernel_coverage.json: reviewed fact set changed", errors
        )
        self.assertIn(
            "derived:include/zynum/blas/blas.h: reviewed fact set changed", errors
        )

    def test_closed_kernel_coverage_gap_cannot_be_restored(self) -> None:
        inventory = self._inventory()
        inventory["current_gaps"].append(
            {
                "id": "gap:kernel-coverage-untracked",
                "classification": "stale untracked classification",
                "owner": "kernel-coverage",
                "observed_result": "tracked generated reference treated as untracked",
                "status": "closed",
            }
        )
        self._write_inventory(inventory)
        self._assert_error_contains(
            "current gap ids must match the complete reviewed set"
        )

    def test_abi_parity_false_gap_cannot_be_restored(self) -> None:
        inventory = self._inventory()
        inventory["current_gaps"].append(
            {
                "id": "gap:abi-parity-not-aggregate",
                "classification": "false dedicated-only claim",
                "owner": "test-infrastructure",
                "observed_result": "ignores aggregate unittest discovery",
                "status": "not a real gap",
            }
        )
        self._write_inventory(inventory)
        self._assert_error_contains(
            "current_gaps reviewed fact set changed"
        )

    def test_abi_parity_aggregate_coverage_truth_is_mandatory(self) -> None:
        inventory = self._inventory()
        parity = next(
            item
            for item in inventory["build_observations"]
            if item["id"] == "step:build.zig:build:test-abi-artifact-parity-verifier"
        )
        self.assertEqual(
            "conditional-test-coverage", parity["aggregate_test_membership"]
        )
        self.assertFalse(parity["intentional_orphan"])
        parity["aggregate_test_membership"] = "not-member"
        parity["aggregate_condition"] = "not-applicable"
        parity["intentional_orphan"] = True
        parity["orphan_reason"] = "only the dedicated step runs this test"
        parity["step_role"] = "known-orphan-gap"
        self._write_inventory(inventory)
        errors = "\n".join(CHECKER.validate(self.root, self.inventory_path))
        self.assertIn(
            "ABI parity step must distinguish missing direct step edge from aggregate test coverage",
            errors,
        )

    def test_owner_vocabulary_is_exact(self) -> None:
        inventory = self._inventory()
        inventory["owner_vocabulary"].append("unrecognized-owner")
        self._write_inventory(inventory)
        self._assert_error_contains(
            "owner_vocabulary must match the schema vocabulary exactly"
        )

    def test_duplicate_semantic_id_fails(self) -> None:
        inventory = self._inventory()
        inventory["workflow_launches"].append(
            copy.deepcopy(inventory["workflow_launches"][0])
        )
        self._write_inventory(inventory)
        self._assert_error_contains("duplicate id")

    def test_build_observation_source_fields_are_mandatory(self) -> None:
        inventory = self._inventory()
        observation = next(
            item
            for item in inventory["build_observations"]
            if item["id"] == "step:build.zig:build:test"
        )
        observation.pop("source_digest")
        observation["anchor"]["symbol"] = "fabricated_step"
        self._write_inventory(inventory)
        errors = "\n".join(CHECKER.validate(self.root, self.inventory_path))
        self.assertIn("source field anchor changed", errors)
        self.assertIn("source field source_digest changed", errors)

    def test_build_observation_reviewed_facts_cannot_be_reclassified(self) -> None:
        inventory = self._inventory()
        observation = next(
            item
            for item in inventory["build_observations"]
            if item["id"] == "step:build.zig:build:test"
        )
        observation["owner"] = "release-validation"
        observation["direct_dependencies"] = []
        observation["aggregate_test_membership"] = "intentional-orphan"
        observation["aggregate_condition"] = "never"
        observation["intentional_orphan"] = True
        self._write_inventory(inventory)
        errors = "\n".join(CHECKER.validate(self.root, self.inventory_path))
        self.assertRegex(
            errors,
            r"unreviewed source projection observed sha256=[0-9a-f]{64}",
        )
        self.assertIn(
            "canonical test aggregate must inventory every native test root",
            errors,
        )
        self.assertIn(
            "step:build.zig:build:test: no reviewed build step is an intentional orphan",
            errors,
        )

    def test_option_and_launch_reviewed_facts_cannot_be_reclassified(self) -> None:
        baseline = self._inventory()
        mutations = (
            ("option_surfaces", "consumers", ["fabricated"]),
            ("python_launches", "execute_on", "fabricated planet"),
            ("workflow_launches", "owner", "build-composition"),
        )
        for section, field, value in mutations:
            with self.subTest(section=section):
                inventory = copy.deepcopy(baseline)
                inventory[section][0][field] = value
                self._write_inventory(inventory)
                errors = "\n".join(CHECKER.validate(self.root, self.inventory_path))
                if section == "option_surfaces":
                    self.assertIn("option_surfaces reviewed fact set changed", errors)
                else:
                    self.assertRegex(
                        errors,
                        r"unreviewed source projection observed sha256=[0-9a-f]{64}",
                    )

    def test_schema_v3_build_roots_and_manifests_are_discovered_independently(
        self,
    ) -> None:
        nested = self.root / "nested/build.zig"
        nested.parent.mkdir()
        nested.write_text("pub fn build(_: anytype) void {}\n", encoding="utf-8")
        manifest = nested.with_name("build.zig.zon")
        manifest.write_text(".{}\n", encoding="utf-8")
        context = CHECKER._make_discovery_context(self.root)
        self.assertIn("nested/build.zig", context.build_roots)
        row = next(
            item
            for item in context.build_manifests
            if item["path"] == "nested/build.zig.zon"
        )
        self.assertEqual(
            {
                "id": "build-manifest:nested/build.zig.zon",
                "path": "nested/build.zig.zon",
                "build_root": "nested/build.zig",
                "content_sha256": hashlib.sha256(b".{}\n").hexdigest(),
            },
            row,
        )
        self._assert_error_contains(
            "build_roots must exactly match independently observed safe build roots"
        )

        manifest.unlink()
        nested.unlink()
        hidden = self.root / ".hidden/build.zig"
        hidden.parent.mkdir()
        hidden.write_text("pub fn build(_: anytype) void {}\n", encoding="utf-8")
        self.assertIn(".hidden/build.zig", CHECKER._discover_build_roots(self.root))
        self._assert_error_contains(
            "build_roots must exactly match independently observed safe build roots"
        )

        hidden.unlink()
        excluded = self.root / "zig-out/nested/build.zig"
        excluded.parent.mkdir(parents=True)
        excluded.write_text("ignored\n", encoding="utf-8")
        self.assertNotIn(
            "zig-out/nested/build.zig", CHECKER._discover_build_roots(self.root)
        )

        with tempfile.TemporaryDirectory(prefix="manifest-free-root-") as directory:
            root = Path(directory)
            (root / "build.zig").write_text(
                "pub fn build() void {}\n", encoding="utf-8"
            )
            context = CHECKER._make_discovery_context(root)
            self.assertEqual(("build.zig",), context.build_roots)
            self.assertEqual((), context.build_manifests)

    def test_schema_v3_manifest_ledger_shape_order_and_digest_are_exact(self) -> None:
        inventory = self._inventory()
        manifests = inventory["build_manifests"]
        self.assertEqual(manifests, sorted(manifests, key=lambda row: row["id"]))
        self.assertTrue(manifests)

        mutations = []
        for field, value in (
            ("id", "build-manifest:fabricated"),
            ("path", "fabricated/build.zig.zon"),
            ("build_root", "fabricated/build.zig"),
            ("content_sha256", "0" * 64),
        ):
            changed = copy.deepcopy(inventory)
            changed["build_manifests"][0][field] = value
            mutations.append(changed)
        duplicate = copy.deepcopy(inventory)
        duplicate["build_manifests"].append(
            copy.deepcopy(duplicate["build_manifests"][0])
        )
        mutations.append(duplicate)
        reordered = copy.deepcopy(inventory)
        reordered["build_manifests"].reverse()
        mutations.append(reordered)
        extra = copy.deepcopy(inventory)
        extra["build_manifests"][0]["extra"] = "not allowed"
        mutations.append(extra)
        missing = copy.deepcopy(inventory)
        missing["build_manifests"].pop()
        mutations.append(missing)

        for changed in mutations:
            with self.subTest(manifests=changed["build_manifests"]):
                self._write_inventory(changed)
                errors = CHECKER.validate(self.root, self.inventory_path)
                self.assertTrue(errors)
                self.assertIn("build_manifests", "\n".join(errors))

    def test_orphan_build_manifest_and_symlink_root_fail_closed(self) -> None:
        manifest = self.root / "nested/build.zig.zon"
        manifest.parent.mkdir()
        manifest.write_text(".{}\n", encoding="utf-8")
        with self.assertRaisesRegex(CHECKER.InventoryError, "no safe sibling"):
            CHECKER._discover_build_roots(self.root)
        manifest.unlink()

        target = self.root / "target.zig"
        target.write_text("pub fn build(_: anytype) void {}\n", encoding="utf-8")
        linked = self.root / "nested/build.zig"
        try:
            linked.symlink_to(target)
        except OSError as exc:
            self.skipTest(f"symlinks are unavailable: {exc}")
        with self.assertRaisesRegex(CHECKER.InventoryError, "non-symlink regular"):
            CHECKER._discover_build_roots(self.root)

        linked.unlink()
        nested_build = self.root / "nested/build.zig"
        nested_build.write_text("pub fn build() void {}\n", encoding="utf-8")
        manifest_target = self.root / "manifest-target.zon"
        manifest_target.write_text(".{}\n", encoding="utf-8")
        linked_manifest = self.root / "nested/build.zig.zon"
        try:
            linked_manifest.symlink_to(manifest_target)
        except OSError as exc:
            self.skipTest(f"symlinks are unavailable: {exc}")
        with self.assertRaisesRegex(CHECKER.InventoryError, "non-symlink regular"):
            CHECKER._make_discovery_context(self.root)
        linked_manifest.unlink()

        if hasattr(CHECKER.os, "mkfifo"):
            CHECKER.os.mkfifo(linked_manifest)
            with self.assertRaisesRegex(CHECKER.InventoryError, "non-symlink regular"):
                CHECKER._make_discovery_context(self.root)

    def test_python_sources_reuse_complete_public_file_universe(self) -> None:
        with tempfile.TemporaryDirectory(prefix="python-public-universe-") as directory:
            root = Path(directory)
            sources = {
                ".github/scripts/check.py": "import subprocess\nsubprocess.run(['one'])\n",
                "maintenance/tools/check.py": "import subprocess\nsubprocess.run(['two'])\n",
            }
            for rel, source in sources.items():
                path = root / rel
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(source, encoding="utf-8")

            launches = CHECKER._discover_python_launches(root)
            files = {item["anchor"]["file"] for item in launches}
            self.assertEqual(set(sources), files)

            target = root / "target.txt"
            target.write_text("not Python\n", encoding="utf-8")
            linked = root / "linked.py"
            try:
                linked.symlink_to(target)
            except OSError as exc:
                self.skipTest(f"symlinks are unavailable: {exc}")
            with self.assertRaisesRegex(CHECKER.InventoryError, "non-symlink regular"):
                CHECKER._discover_python_launches(root)

    def test_discovery_context_uses_exact_git_root_and_one_enumeration(self) -> None:
        with tempfile.TemporaryDirectory(prefix="inventory-git-context-") as directory:
            root = Path(directory)
            result = subprocess.run(
                ["git", "init", "-q", str(root)], capture_output=True, check=False
            )
            self.assertEqual(0, result.returncode, result.stderr)
            (root / ".git/info/exclude").write_text(
                "local-planning/\n", encoding="utf-8"
            )
            (root / ".gitignore").write_text(
                "ignored/\n.github/workflows/ignored.yml\n"
                "docs/assets/benchmarks/ignored.svg\npkgconfig/ignored.pc\n",
                encoding="utf-8",
            )
            (root / "tracked.py").write_text("pass\n", encoding="utf-8")
            result = subprocess.run(
                ["git", "-C", str(root), "add", ".gitignore", "tracked.py"],
                capture_output=True,
                check=False,
            )
            self.assertEqual(0, result.returncode, result.stderr)
            (root / "untracked.py").write_text(
                "import subprocess\nsubprocess.run(['visible'])\n", encoding="utf-8"
            )
            ignored = root / "ignored"
            ignored.mkdir()
            (ignored / "build.zig").write_text("ignored\n", encoding="utf-8")
            (ignored / "hidden.py").write_text(
                "import subprocess\nsubprocess.run(['hidden'])\n", encoding="utf-8"
            )
            local_planning = root / "local-planning"
            local_planning.mkdir()
            (local_planning / "build.zig").write_text(
                "pub fn build(_: anytype) void {}\n", encoding="utf-8"
            )
            (local_planning / "hidden.py").write_text(
                "import subprocess\nsubprocess.run(['local-only'])\n",
                encoding="utf-8",
            )
            visible_build = root / "untracked/build.zig"
            visible_build.parent.mkdir()
            visible_build.write_text(
                "pub fn build(_: anytype) void {}\n", encoding="utf-8"
            )
            visible_manifest = visible_build.with_name("build.zig.zon")
            visible_manifest.write_text(".{}\n", encoding="utf-8")
            visible_workflow = root / ".github/workflows/visible.yml"
            visible_workflow.parent.mkdir(parents=True)
            visible_workflow.write_text(
                "jobs:\n  visible:\n    runs-on: ubuntu-latest\n"
                "    steps:\n      - name: Visible\n        run: echo visible\n",
                encoding="utf-8",
            )
            (visible_workflow.parent / "ignored.yml").write_text(
                "jobs:\n  ignored:\n    steps:\n      - run: false\n", encoding="utf-8"
            )
            ignored_svg = root / "docs/assets/benchmarks/ignored.svg"
            ignored_svg.parent.mkdir(parents=True)
            ignored_svg.write_text("<svg/>\n", encoding="utf-8")
            ignored_pc = root / "pkgconfig/ignored.pc"
            ignored_pc.parent.mkdir(parents=True)
            ignored_pc.write_text("Name: ignored\n", encoding="utf-8")
            original = CHECKER.repository_git.RepositoryGit.ls_files
            enumerations = 0

            def counted(repository, paths=()):
                nonlocal enumerations
                enumerations += 1
                return original(repository, paths)

            CHECKER.repository_git.RepositoryGit.ls_files = counted
            try:
                context = CHECKER._make_discovery_context(root)
                CHECKER._discover_build_roots(root, context)
                launches = CHECKER._discover_python_launches(root, context)
                workflows = CHECKER._discover_workflow_launches(root, context)
                classifications, complete = (
                    CHECKER._discover_repository_file_classifications(root, context)
                )
            finally:
                CHECKER.repository_git.RepositoryGit.ls_files = original
            self.assertEqual("git-checkout", context.public_files.mode)
            self.assertEqual(6, enumerations)
            self.assertTrue(complete)
            self.assertIn("untracked.py", context.public_files.path_set)
            self.assertNotIn("ignored/hidden.py", context.public_files.path_set)
            self.assertNotIn("local-planning/hidden.py", context.public_files.path_set)
            self.assertNotIn("local-planning/build.zig", context.build_roots)
            self.assertIn("untracked/build.zig", context.build_roots)
            self.assertEqual(
                [
                    {
                        "id": "build-manifest:untracked/build.zig.zon",
                        "path": "untracked/build.zig.zon",
                        "build_root": "untracked/build.zig",
                        "content_sha256": hashlib.sha256(b".{}\n").hexdigest(),
                    }
                ],
                list(context.build_manifests),
            )
            self.assertNotIn(
                ".github/workflows/ignored.yml", context.public_files.path_set
            )
            self.assertNotIn(
                "docs/assets/benchmarks/ignored.svg", context.public_files.path_set
            )
            self.assertNotIn("pkgconfig/ignored.pc", context.public_files.path_set)
            self.assertEqual(
                {"untracked.py"}, {item["anchor"]["file"] for item in launches}
            )
            self.assertEqual(
                {".github/workflows/visible.yml"},
                {item["anchor"]["file"] for item in workflows},
            )
            self.assertNotIn(
                "ignored/build.zig", {item["path"] for item in classifications}
            )
            self.assertNotIn(
                "local-planning/build.zig",
                {item["path"] for item in classifications},
            )

        with tempfile.TemporaryDirectory(
            prefix="inventory-nested-", dir=REPOSITORY_ROOT
        ) as directory:
            nested = Path(directory)
            (nested / "archive.py").write_text("pass\n", encoding="utf-8")
            archive_build = nested / "untracked/build.zig"
            archive_build.parent.mkdir()
            archive_build.write_text(
                "pub fn build(_: anytype) void {}\n", encoding="utf-8"
            )
            archive_build.with_name("build.zig.zon").write_text(
                ".{}\n", encoding="utf-8"
            )
            universe = CHECKER._make_public_file_universe(nested)
            self.assertEqual("archive", universe.mode)
            archive_context = CHECKER._make_discovery_context(nested)
            self.assertEqual(context.build_roots, archive_context.build_roots)
            self.assertEqual(context.build_manifests, archive_context.build_manifests)

    def test_public_universe_git_failures_and_archive_symlink_are_fail_closed(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory(prefix="inventory-broken-git-") as directory:
            root = Path(directory)
            (root / ".git").write_text("not a git directory\n", encoding="utf-8")
            with self.assertRaisesRegex(
                CHECKER.InventoryError, "top-level verification failed"
            ):
                CHECKER._make_public_file_universe(root)

        with tempfile.TemporaryDirectory(prefix="inventory-symlink-dir-") as directory:
            root = Path(directory)
            target = root / "target"
            target.mkdir()
            (target / "child.py").write_text("pass\n", encoding="utf-8")
            linked = root / "linked"
            try:
                linked.symlink_to(target, target_is_directory=True)
            except OSError as exc:
                self.skipTest(f"symlinks are unavailable: {exc}")
            universe = CHECKER._make_public_file_universe(root)
            self.assertIn("linked", universe.path_set)
            self.assertNotIn("linked/child.py", universe.path_set)
            classifications, _ = CHECKER._discover_repository_file_classifications(
                root,
                CHECKER.DiscoveryContext(
                    universe.root,
                    universe,
                    *CHECKER._build_roots_from_universe(universe),
                ),
            )
            linked_row = next(
                item for item in classifications if item["path"] == "linked"
            )
            self.assertEqual("symbolic-link", linked_row["kind"])

    def test_public_universe_rejects_git_environment_redirection(self) -> None:
        with tempfile.TemporaryDirectory(prefix="inventory-git-redirect-") as directory:
            base = Path(directory)
            repository = base / "repository"
            archive = base / "archive"
            repository.mkdir()
            archive.mkdir()
            result = subprocess.run(
                ["git", "init", "-q", str(repository)],
                capture_output=True,
                check=False,
            )
            self.assertEqual(0, result.returncode, result.stderr)
            (archive / "present.py").write_text("pass\n", encoding="utf-8")

            names = ("GIT_DIR", "GIT_WORK_TREE")
            previous = {name: CHECKER.os.environ.get(name) for name in names}
            CHECKER.os.environ.update(
                {
                    "GIT_DIR": str(repository / ".git"),
                    "GIT_WORK_TREE": str(archive),
                }
            )
            try:
                universe = CHECKER._make_public_file_universe(archive)
                self.assertEqual("archive", universe.mode)
                self.assertEqual(("present.py",), universe.paths)
                with self.assertRaisesRegex(
                    CHECKER.InventoryError, "ambient Git environment"
                ):
                    CHECKER._make_public_file_universe(repository)
            finally:
                for name, value in previous.items():
                    if value is None:
                        CHECKER.os.environ.pop(name, None)
                    else:
                        CHECKER.os.environ[name] = value

    def test_public_universe_rejects_symlink_git_marker(self) -> None:
        with tempfile.TemporaryDirectory(prefix="inventory-git-marker-") as directory:
            root = Path(directory)
            target = root / "marker-target"
            target.mkdir()
            try:
                (root / ".git").symlink_to(target, target_is_directory=True)
            except OSError as exc:
                self.skipTest(f"symlinks are unavailable: {exc}")
            with self.assertRaisesRegex(CHECKER.InventoryError, "unsafe filesystem"):
                CHECKER._make_public_file_universe(root)

    def test_git_public_enumeration_omits_deleted_cached_paths(self) -> None:
        with tempfile.TemporaryDirectory(
            prefix="inventory-git-malformed-"
        ) as directory:
            root = Path(directory)
            result = subprocess.run(
                ["git", "init", "-q", str(root)], capture_output=True, check=False
            )
            self.assertEqual(0, result.returncode, result.stderr)
            present = root / "present.py"
            present.write_text("pass\n", encoding="utf-8")
            deleted = root / "deleted.py"
            deleted.write_text("pass\n", encoding="utf-8")
            result = subprocess.run(
                ["git", "-C", str(root), "add", "present.py", "deleted.py"],
                capture_output=True,
                check=False,
            )
            self.assertEqual(0, result.returncode, result.stderr)
            deleted.unlink()
            universe = CHECKER._make_public_file_universe(root)
            self.assertIn("present.py", universe.path_set)
            self.assertNotIn("deleted.py", universe.path_set)

    def test_git_snapshot_rejects_disappearance_and_addition_after_capture(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory(prefix="inventory-git-race-") as directory:
            root = Path(directory)
            original = root / "original.py"
            original.write_text("pass\n", encoding="utf-8")
            before = CHECKER.repository_git.RepositoryFileSet(
                listed=("original.py",), deleted=(), present=("original.py",)
            )
            after = CHECKER.repository_git.RepositoryFileSet(
                listed=("late.py",), deleted=(), present=("late.py",)
            )
            repository = mock.Mock(root=root)

            def snapshot() -> Any:
                if repository.snapshot_file_set.call_count == 1:
                    return before
                original.unlink()
                (root / "late.py").write_text("pass\n", encoding="utf-8")
                return after

            repository.snapshot_file_set.side_effect = snapshot
            with (
                mock.patch.object(
                    CHECKER.repository_git,
                    "open_repository",
                    return_value=repository,
                ),
                self.assertRaisesRegex(CHECKER.InventoryError, "Git file set changed"),
            ):
                CHECKER._make_public_file_universe(root)

    def test_git_snapshot_rejects_same_inode_content_and_aba_changes(self) -> None:
        for mutation in ("content", "aba"):
            with (
                self.subTest(mutation=mutation),
                tempfile.TemporaryDirectory(prefix="inventory-git-aba-") as directory,
            ):
                root = Path(directory)
                source = root / "source.py"
                baseline = b"value = 'safe'\n"
                source.write_bytes(baseline)
                initial_metadata = source.stat()
                inode = initial_metadata.st_ino
                file_set = CHECKER.repository_git.RepositoryFileSet(
                    listed=("source.py",), deleted=(), present=("source.py",)
                )
                repository = mock.Mock(root=root)

                def snapshot() -> Any:
                    if repository.snapshot_file_set.call_count == 2:
                        source.write_bytes(b"value = 'evil'\n")
                        if mutation == "aba":
                            source.write_bytes(baseline)
                        os.utime(
                            source,
                            ns=(
                                initial_metadata.st_atime_ns,
                                initial_metadata.st_mtime_ns + 2_000_000_000,
                            ),
                            follow_symlinks=False,
                        )
                    return file_set

                repository.snapshot_file_set.side_effect = snapshot
                with (
                    mock.patch.object(
                        CHECKER.repository_git,
                        "open_repository",
                        return_value=repository,
                    ),
                    self.assertRaisesRegex(
                        CHECKER.InventoryError, "repository snapshot failed closed"
                    ),
                ):
                    CHECKER._make_public_file_universe(root)
                self.assertEqual(inode, source.stat().st_ino)

    def test_archive_snapshot_rejects_directory_symlink_swap_and_late_child(
        self,
    ) -> None:
        for mutation in ("symlink-swap", "late-child"):
            with (
                self.subTest(mutation=mutation),
                tempfile.TemporaryDirectory(
                    prefix="inventory-archive-race-"
                ) as directory,
            ):
                base = Path(directory)
                root = base / "root"
                payload = root / "payload"
                payload.mkdir(parents=True)
                (payload / "build.zig").write_text(
                    "pub fn build() void {}\n", encoding="utf-8"
                )
                external = base / "external"
                external.mkdir()
                (external / "build.zig").write_text("private\n", encoding="utf-8")
                real_capture = CHECKER._capture_cached_node
                mutated = False

                def mutate_then_capture(session: Any, path: str) -> Any:
                    nonlocal mutated
                    if not mutated:
                        mutated = True
                        if mutation == "symlink-swap":
                            payload.rename(base / "held-payload")
                            payload.symlink_to(external, target_is_directory=True)
                        else:
                            (payload / "late.txt").write_text(
                                "late\n", encoding="utf-8"
                            )
                    return real_capture(session, path)

                with (
                    mock.patch.object(
                        CHECKER,
                        "_capture_cached_node",
                        side_effect=mutate_then_capture,
                    ),
                    self.assertRaisesRegex(
                        CHECKER.InventoryError, "repository snapshot failed closed"
                    ),
                ):
                    CHECKER._make_public_file_universe(root)

    def test_sealed_context_is_the_only_downstream_source_of_bytes(self) -> None:
        inventory = self._inventory()
        context = CHECKER._make_discovery_context(self.root, self.inventory_path)
        baseline = CHECKER.discover(self.root, inventory, context)
        baseline_level1 = CHECKER._discover_level1_payload_source_bindings(
            self.root, context
        )
        baseline_classifications = CHECKER._discover_repository_file_classifications(
            self.root, context
        )

        (self.root / "build.zig").write_text("changed after seal\n", encoding="utf-8")
        (self.root / "tools/check_package_paths.py").write_text(
            "changed after seal\n", encoding="utf-8"
        )
        (self.root / ".github/workflows/ci.yml").write_text(
            "changed after seal\n", encoding="utf-8"
        )
        forbidden = AssertionError("downstream live filesystem access")
        with (
            mock.patch.object(Path, "read_text", side_effect=forbidden),
            mock.patch.object(Path, "read_bytes", side_effect=forbidden),
            mock.patch.object(Path, "lstat", side_effect=forbidden),
            mock.patch.object(Path, "stat", side_effect=forbidden),
            mock.patch.object(Path, "is_symlink", side_effect=forbidden),
            mock.patch.object(
                CHECKER.repository_git.RepositoryGit,
                "snapshot_file_set",
                side_effect=forbidden,
            ),
        ):
            self.assertEqual(baseline, CHECKER.discover(self.root, inventory, context))
            self.assertEqual(
                baseline_level1,
                CHECKER._discover_level1_payload_source_bindings(self.root, context),
            )
            self.assertEqual(
                baseline_classifications,
                CHECKER._discover_repository_file_classifications(self.root, context),
            )
        with self.assertRaises(TypeError):
            context.public_files.node_index["fabricated"] = context.public_files.nodes[
                0
            ]

    def test_level1_payload_binding_requires_frozen_execution_keywords(self) -> None:
        path = self.root / "bench/tools/run_level1_report.py"
        original = path.read_text(encoding="utf-8")
        mutations = (
            (
                "expected_binary=args.level1_probe if artifacts is not None else None",
                "expected_binary=args.copy_probe if artifacts is not None else None",
                "expected_binary binding is not exact",
            ),
            (
                "expected_library=private_library if artifacts is not None else None,\n"
                "            artifacts=artifacts,\n",
                "expected_library=private_library if artifacts is not None else None,\n",
                "frozen execution keywords are not exact",
            ),
        )
        for old, new, expected in mutations:
            with self.subTest(expected=expected):
                self.assertIn(old, original)
                path.write_text(original.replace(old, new, 1), encoding="utf-8")
                context = CHECKER._make_discovery_context(
                    self.root,
                    self.inventory_path,
                )
                with self.assertRaisesRegex(CHECKER.InventoryError, expected):
                    CHECKER._discover_level1_payload_source_bindings(
                        self.root,
                        context,
                    )
                path.write_text(original, encoding="utf-8")

    def test_archive_paths_preserve_legal_unicode_and_reject_unsafe_spelling(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory(prefix="inventory-paths-") as directory:
            root = Path(directory)
            legal = "quoted ' path\n合法-δ.py"
            (root / legal).write_text("pass\n", encoding="utf-8")
            if hasattr(os, "mkfifo"):
                os.mkfifo(root / "special")
            universe = CHECKER._make_public_file_universe(root)
            self.assertIn(legal, universe.path_set)
            self.assertEqual("regular", universe.node(legal).kind)
            if hasattr(os, "mkfifo"):
                self.assertEqual("special", universe.node("special").kind)

        if os.name != "nt":
            with tempfile.TemporaryDirectory(
                prefix="inventory-unsafe-path-"
            ) as directory:
                root = Path(directory)
                (root / "bad\\name.py").write_text("pass\n", encoding="utf-8")
                with self.assertRaisesRegex(CHECKER.InventoryError, "invalid"):
                    CHECKER._make_public_file_universe(root)

    def test_custom_inventory_and_snapshot_cache_limits_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory(prefix="inventory-custom-") as directory:
            base = Path(directory)
            root = base / "root"
            inventory = root / "inputs/custom.json"
            inventory.parent.mkdir(parents=True)
            inventory.write_text("{}\n", encoding="utf-8")
            context = CHECKER._make_discovery_context(root, inventory)
            self.assertEqual(b"{}\n", context.inventory_node.bytes)
            with self.assertRaisesRegex(CHECKER.InventoryError, "within"):
                CHECKER._make_discovery_context(root, base / "outside.json")

            target = root / "target.json"
            target.write_text("{}\n", encoding="utf-8")
            inventory.unlink()
            try:
                inventory.symlink_to(target)
            except OSError as exc:
                self.skipTest(f"symlinks are unavailable: {exc}")
            with self.assertRaisesRegex(CHECKER.InventoryError, "frozen regular"):
                CHECKER._make_discovery_context(root, inventory)

        with tempfile.TemporaryDirectory(prefix="inventory-cache-") as directory:
            root = Path(directory)
            (root / "one.py").write_bytes(b"12345")
            with (
                mock.patch.object(CHECKER, "SNAPSHOT_MAX_CACHED_FILE_BYTES", 4),
                self.assertRaisesRegex(CHECKER.InventoryError, "input limit"),
            ):
                CHECKER._make_public_file_universe(root)

            (root / "one.py").write_bytes(b"1234")
            (root / "two.py").write_bytes(b"5678")
            with (
                mock.patch.object(CHECKER, "SNAPSHOT_MAX_CACHED_FILE_BYTES", 4),
                mock.patch.object(CHECKER, "SNAPSHOT_MAX_CACHED_TOTAL_BYTES", 7),
                self.assertRaisesRegex(CHECKER.InventoryError, "cumulative"),
            ):
                CHECKER._make_public_file_universe(root)

    def test_control_artifact_capture_is_bounded_regular_and_byte_exact(self) -> None:
        snapshot = CHECKER.repository_snapshot
        with tempfile.TemporaryDirectory(prefix="control-artifact-") as directory:
            root = Path(directory)
            control = root / "control.json"
            payload = b'{"value":"exact"}\n'
            control.write_bytes(payload)
            node = snapshot.capture_control_artifact(
                root, "control.json", max_bytes=len(payload)
            )
            self.assertEqual("regular", node.kind)
            self.assertEqual(payload, node.bytes)
            self.assertEqual(hashlib.sha256(payload).hexdigest(), node.sha256)

            with self.assertRaisesRegex(snapshot.RepositorySnapshotError, "limit"):
                snapshot.capture_control_artifact(
                    root, "control.json", max_bytes=len(payload) - 1
                )
            with self.assertRaisesRegex(snapshot.RepositorySnapshotError, "missing"):
                snapshot.capture_control_artifact(root, "missing.json", max_bytes=1024)

            target = root / "target.json"
            target.write_bytes(payload)
            control.unlink()
            try:
                control.symlink_to(target)
            except OSError as exc:
                self.skipTest(f"symlinks are unavailable: {exc}")
            with self.assertRaisesRegex(
                snapshot.RepositorySnapshotError, "regular file"
            ):
                snapshot.capture_control_artifact(root, "control.json", max_bytes=1024)

            control.unlink()
            if hasattr(os, "mkfifo"):
                os.mkfifo(control)
                with self.assertRaisesRegex(
                    snapshot.RepositorySnapshotError, "regular file"
                ):
                    snapshot.capture_control_artifact(
                        root, "control.json", max_bytes=1024
                    )

    def test_control_artifact_rejects_swap_and_read_time_change(self) -> None:
        snapshot = CHECKER.repository_snapshot
        for mutation in ("content", "symlink-swap"):
            with (
                self.subTest(mutation=mutation),
                tempfile.TemporaryDirectory(prefix="control-race-") as directory,
            ):
                root = Path(directory)
                control = root / "control.json"
                original = b'{"safe":true}\n'
                control.write_bytes(original)
                real_read = snapshot._read_regular_file

                def mutate_after_read(*args: Any, **kwargs: Any) -> Any:
                    result = real_read(*args, **kwargs)
                    if mutation == "content":
                        control.write_bytes(b'{"evil":true}\n')
                    else:
                        held = root / "held.json"
                        control.rename(held)
                        control.symlink_to(held)
                    return result

                with (
                    mock.patch.object(
                        snapshot,
                        "_read_regular_file",
                        side_effect=mutate_after_read,
                    ),
                    self.assertRaisesRegex(snapshot.RepositorySnapshotError, "changed"),
                ):
                    snapshot.capture_control_artifact(
                        root, "control.json", max_bytes=1024
                    )

    def test_control_artifact_reanchors_parent_chain_before_return(self) -> None:
        snapshot = CHECKER.repository_snapshot
        with tempfile.TemporaryDirectory(prefix="control-parent-race-") as directory:
            root = Path(directory)
            parent = root / "metadata"
            parent.mkdir()
            control = parent / "control.json"
            control.write_bytes(b'{"safe":true}\n')
            detached = root / "detached"
            real_read = snapshot._read_regular_file
            mutated = False

            def replace_parent_after_read(*args: Any, **kwargs: Any) -> Any:
                nonlocal mutated
                result = real_read(*args, **kwargs)
                if not mutated:
                    mutated = True
                    parent.rename(detached)
                    parent.mkdir()
                    (parent / "control.json").write_bytes(b'{"evil":true}\n')
                return result

            with (
                mock.patch.object(
                    snapshot,
                    "_read_regular_file",
                    side_effect=replace_parent_after_read,
                ),
                self.assertRaisesRegex(snapshot.RepositorySnapshotError, "changed"),
            ):
                snapshot.capture_control_artifact(
                    root, "metadata/control.json", max_bytes=1024
                )

    def test_snapshot_reverification_respects_the_frozen_size_bound(self) -> None:
        snapshot = CHECKER.repository_snapshot
        with tempfile.TemporaryDirectory(prefix="snapshot-growth-") as directory:
            root = Path(directory)
            source = root / "source.py"
            source.write_bytes(b"x")
            with snapshot.SnapshotSession(root) as session:
                session.capture_paths(("source.py",), include_bytes=True, limit=1)
                source.write_bytes(b"y" * (2 * 1024 * 1024))
                with self.assertRaisesRegex(
                    snapshot.RepositorySnapshotError, "input limit"
                ):
                    session.seal()

    def test_regular_file_limits_validate_exact_non_boolean_relationships(self) -> None:
        snapshot = CHECKER.repository_snapshot
        with tempfile.TemporaryDirectory(prefix="snapshot-limit-shapes-") as directory:
            root = Path(directory)
            invalid = (
                snapshot.RegularFileLimits(True, 1, 1, 1),
                snapshot.RegularFileLimits(-1, 1, 0, 0),
                snapshot.RegularFileLimits(1, 1, 2, 1),
                snapshot.RegularFileLimits(2, 1, 1, 2),
            )
            for limits in invalid:
                with (
                    self.subTest(limits=limits),
                    self.assertRaises(snapshot.RepositorySnapshotError),
                ):
                    snapshot.SnapshotSession(root, regular_file_limits=limits)

    def test_regular_file_reader_is_exact_and_never_chases_eof(self) -> None:
        snapshot = CHECKER.repository_snapshot

        class RecordingStream(io.BytesIO):
            def __init__(self, value: bytes) -> None:
                super().__init__(value)
                self.requests: list[int] = []

            def read(self, size: int = -1) -> bytes:
                self.requests.append(size)
                return super().read(size)

        payload = b"x" * (1024 * 1024 + 3)
        exact = RecordingStream(payload)
        digest, contents = snapshot._read_regular_file(
            exact, capture_bytes=True, frozen_size=len(payload)
        )
        self.assertEqual([1024 * 1024, 3, 1], exact.requests)
        self.assertEqual(payload, contents)
        self.assertEqual(hashlib.sha256(payload).hexdigest(), digest)

        with self.assertRaisesRegex(snapshot.RepositorySnapshotError, "ended"):
            snapshot._read_regular_file(
                io.BytesIO(b"xy"), capture_bytes=False, frozen_size=3
            )
        with self.assertRaisesRegex(snapshot.RepositorySnapshotError, "grew"):
            snapshot._read_regular_file(
                io.BytesIO(b"xyz!"), capture_bytes=False, frozen_size=3
            )

        class EndlessStream:
            def __init__(self) -> None:
                self.requests: list[int] = []

            def read(self, size: int = -1) -> bytes:
                self.requests.append(size)
                return b"z" * size

        endless = EndlessStream()
        with self.assertRaisesRegex(snapshot.RepositorySnapshotError, "grew"):
            snapshot._read_regular_file(endless, capture_bytes=False, frozen_size=4)
        self.assertEqual([4, 1], endless.requests)

    def test_regular_file_admission_precedes_reads_and_none_cannot_bypass(self) -> None:
        snapshot = CHECKER.repository_snapshot
        limits = snapshot.RegularFileLimits(4, 8, 4, 8)
        with tempfile.TemporaryDirectory(prefix="snapshot-admission-") as directory:
            root = Path(directory)
            (root / "huge.bin").write_bytes(b"12345")
            with (
                snapshot.SnapshotSession(root, regular_file_limits=limits) as session,
                mock.patch.object(
                    snapshot,
                    "_read_regular_file",
                    side_effect=AssertionError("oversized file was read"),
                ),
                self.assertRaisesRegex(snapshot.RepositorySnapshotError, "input limit"),
            ):
                session.capture_paths(("huge.bin",), limit=None)

    def test_regular_file_round_and_cache_boundaries_reset_per_operation(self) -> None:
        snapshot = CHECKER.repository_snapshot
        limits = snapshot.RegularFileLimits(3, 6, 3, 5)
        with tempfile.TemporaryDirectory(prefix="snapshot-rounds-") as directory:
            root = Path(directory)
            (root / "one.py").write_bytes(b"123")
            (root / "two.py").write_bytes(b"456")
            with snapshot.SnapshotSession(root, regular_file_limits=limits) as session:
                nodes = session.capture_paths(("one.py", "two.py"))
                tree = session.seal()
                session.verify(tree)
                with session.open_verified_regular(nodes[0]) as stream:
                    self.assertEqual(b"123", stream.read())

            (root / "three.py").write_bytes(b"78")
            with (
                snapshot.SnapshotSession(root, regular_file_limits=limits) as session,
                self.assertRaisesRegex(snapshot.RepositorySnapshotError, "cumulative"),
            ):
                session.capture_paths(("one.py", "two.py", "three.py"))

            with snapshot.SnapshotSession(root, regular_file_limits=limits) as session:
                first = session.capture_paths(("one.py",), include_bytes=True)[0]
                repeated = session.capture_paths(("one.py",), include_bytes=True)[0]
                self.assertIs(first, repeated)
                with self.assertRaisesRegex(
                    snapshot.RepositorySnapshotError, "input limit"
                ):
                    session.capture_paths(("one.py",), include_bytes=True, limit=2)
                session.capture_paths(("three.py",), include_bytes=True)

            (root / "three.py").write_bytes(b"789")
            with (
                snapshot.SnapshotSession(root, regular_file_limits=limits) as session,
                self.assertRaisesRegex(snapshot.RepositorySnapshotError, "cache"),
            ):
                session.capture_paths(("one.py",), include_bytes=True)
                session.capture_paths(("three.py",), include_bytes=True)

    def test_regular_file_limit_diagnostic_escapes_relative_path_before_read(self) -> None:
        snapshot = CHECKER.repository_snapshot
        limits = snapshot.RegularFileLimits(4, 8, 4, 8)
        with tempfile.TemporaryDirectory(prefix="snapshot-limit-diagnostic-") as directory:
            root = Path(directory)
            (root / "nested").mkdir()
            relative = "nested/input\n\x1b\u202e.bin"
            (root / relative).write_bytes(b"12345")
            with (
                snapshot.SnapshotSession(root, regular_file_limits=limits) as session,
                mock.patch.object(
                    snapshot,
                    "_read_regular_file",
                    side_effect=AssertionError("oversized file was read"),
                ),
                self.assertRaises(snapshot.RepositorySnapshotError) as raised,
            ):
                session.capture_paths((relative,), limit=None)
            message = str(raised.exception)
            self.assertEqual(
                "regular file exceeds its input limit: "
                f"path={relative!a}, size_bytes=5, limit_bytes=4",
                message,
            )
            self.assertTrue(message.isascii())
            self.assertNotIn("\n", message)
            self.assertNotIn("\x1b", message)
            self.assertNotIn(str(root), message)

    def test_cached_regular_file_limit_diagnostic_uses_stricter_caller_bound(self) -> None:
        snapshot = CHECKER.repository_snapshot
        with tempfile.TemporaryDirectory(prefix="snapshot-cached-diagnostic-") as directory:
            root = Path(directory)
            (root / "cached.py").write_bytes(b"1234")
            with snapshot.SnapshotSession(root) as session:
                session.capture_paths(("cached.py",), include_bytes=True)
                with (
                    mock.patch.object(
                        snapshot,
                        "_read_regular_file",
                        side_effect=AssertionError("cached oversized file was reread"),
                    ),
                    self.assertRaises(snapshot.RepositorySnapshotError) as raised,
                ):
                    session.capture_paths(("cached.py",), include_bytes=True, limit=3)
                self.assertEqual(
                    "regular file exceeds its input limit: "
                    "path='cached.py', size_bytes=4, limit_bytes=3",
                    str(raised.exception),
                )

    def test_regular_file_round_limit_diagnostics_report_member_usage_and_bound(self) -> None:
        snapshot = CHECKER.repository_snapshot
        cases = (
            (snapshot.RegularFileLimits(3, 5, 3, 5), False, True,
             "regular-file round exceeds its cumulative input limit: "
             "path='two.py', size_bytes=3, used_bytes=3, limit_bytes=5"),
            (snapshot.RegularFileLimits(6, 12, 2, 4), True, False,
             "regular file exceeds its cache input limit: "
             "path='two.py', size_bytes=3, limit_bytes=2"),
            (snapshot.RegularFileLimits(3, 8, 3, 5), True, True,
             "regular-file round exceeds its cumulative cache limit: "
             "path='two.py', size_bytes=3, used_bytes=3, limit_bytes=5"),
        )
        for limits, include_bytes, capture_first, expected in cases:
            with (
                self.subTest(expected=expected),
                tempfile.TemporaryDirectory(prefix="snapshot-round-diagnostic-") as directory,
            ):
                root = Path(directory)
                (root / "one.py").write_bytes(b"123")
                (root / "two.py").write_bytes(b"456")
                with snapshot.SnapshotSession(root, regular_file_limits=limits) as session:
                    if capture_first:
                        session.capture_paths(("one.py",), include_bytes=include_bytes)
                    with (
                        mock.patch.object(
                            snapshot,
                            "_read_regular_file",
                            side_effect=AssertionError("over-budget file was read"),
                        ),
                        self.assertRaises(snapshot.RepositorySnapshotError) as raised,
                    ):
                        session.capture_paths(("two.py",), include_bytes=include_bytes)
                    self.assertEqual(expected, str(raised.exception))
                    self.assertEqual(3 if capture_first else 0, session._capture_round.read_bytes)
                    self.assertEqual(
                        3 if capture_first and include_bytes else 0,
                        session._capture_round.cached_bytes,
                    )

    def test_archive_and_repository_growth_fail_without_reading_to_moving_eof(
        self,
    ) -> None:
        snapshot = CHECKER.repository_snapshot
        file_set = CHECKER.repository_git.RepositoryFileSet(
            listed=("source.bin",), deleted=(), present=("source.bin",)
        )
        for mode in ("archive", "repository"):
            with (
                self.subTest(mode=mode),
                tempfile.TemporaryDirectory(
                    prefix="snapshot-growth-mode-"
                ) as directory,
            ):
                root = Path(directory)
                source = root / "source.bin"
                source.write_bytes(b"safe")
                real_read = snapshot._read_regular_file
                mutated = False

                def grow_before_read(*args: Any, **kwargs: Any) -> Any:
                    nonlocal mutated
                    if not mutated:
                        mutated = True
                        with source.open("ab") as output:
                            output.write(b"!")
                    return real_read(*args, **kwargs)

                repository = None
                if mode == "repository":
                    repository = mock.Mock(root=root)
                    repository.snapshot_file_set.return_value = file_set
                with (
                    mock.patch.object(
                        CHECKER.repository_git,
                        "open_repository",
                        return_value=repository,
                    ),
                    mock.patch.object(
                        snapshot, "_read_regular_file", side_effect=grow_before_read
                    ),
                    self.assertRaisesRegex(CHECKER.InventoryError, "grew"),
                ):
                    CHECKER._make_public_file_universe(root)

    def test_cached_inventory_path_is_charged_once_and_static_limits_apply(
        self,
    ) -> None:
        payload = b"value = 1\n"
        with tempfile.TemporaryDirectory(prefix="inventory-cache-once-") as directory:
            root = Path(directory)
            inventory = root / "inventory.py"
            inventory.write_bytes(payload)
            with (
                mock.patch.object(
                    CHECKER, "SNAPSHOT_MAX_REGULAR_FILE_BYTES", len(payload)
                ),
                mock.patch.object(
                    CHECKER, "SNAPSHOT_MAX_REGULAR_ROUND_BYTES", len(payload) * 4
                ),
                mock.patch.object(
                    CHECKER, "SNAPSHOT_MAX_CACHED_FILE_BYTES", len(payload)
                ),
                mock.patch.object(
                    CHECKER, "SNAPSHOT_MAX_CACHED_TOTAL_BYTES", len(payload)
                ),
            ):
                context = CHECKER._make_discovery_context(root, inventory)
            self.assertEqual(payload, context.inventory_node.bytes)

            (root / "large.bin").write_bytes(b"x" * (len(payload) + 1))
            with (
                mock.patch.object(
                    CHECKER, "SNAPSHOT_MAX_REGULAR_FILE_BYTES", len(payload)
                ),
                mock.patch.object(
                    CHECKER, "SNAPSHOT_MAX_REGULAR_ROUND_BYTES", len(payload) * 4
                ),
                mock.patch.object(
                    CHECKER, "SNAPSHOT_MAX_CACHED_FILE_BYTES", len(payload)
                ),
                mock.patch.object(
                    CHECKER, "SNAPSHOT_MAX_CACHED_TOTAL_BYTES", len(payload)
                ),
                self.assertRaisesRegex(CHECKER.InventoryError, "input limit"),
            ):
                CHECKER._make_public_file_universe(root, inventory)

    def test_replacement_before_verification_is_rejected(self) -> None:
        snapshot = CHECKER.repository_snapshot
        with tempfile.TemporaryDirectory(
            prefix="snapshot-replace-verify-"
        ) as directory:
            root = Path(directory)
            source = root / "source.py"
            source.write_bytes(b"safe")
            with snapshot.SnapshotSession(root) as session:
                session.capture_paths(("source.py",), include_bytes=True)
                held = root / "held.py"
                source.rename(held)
                source.write_bytes(b"evil")
                with self.assertRaisesRegex(
                    snapshot.RepositorySnapshotError, "changed"
                ):
                    session.seal()

    def test_directory_structure_freeze_is_bounded_and_kind_sensitive(self) -> None:
        snapshot = CHECKER.repository_snapshot
        for mutation in ("entry", "entry-aba", "directory", "symlink"):
            with (
                self.subTest(mutation=mutation),
                tempfile.TemporaryDirectory(prefix="structure-freeze-") as directory,
            ):
                root = Path(directory)
                ignored = root / "ignored-secret-name"
                ignored.write_text("one\n", encoding="utf-8")
                excluded = root / ".git"
                excluded.mkdir()
                with snapshot.SnapshotSession(root) as session:
                    session.freeze_directory_structure(
                        {".git"},
                        max_directories=8,
                        max_entries=8,
                        max_depth=4,
                        max_total_name_bytes=256,
                    )
                    sealed = session.seal()
                    ignored.write_text("content-only change\n", encoding="utf-8")
                    (excluded / "allowed-churn").write_text(
                        "excluded\n", encoding="utf-8"
                    )
                    session.verify(sealed)
                    root_metadata = root.stat()

                    if mutation in {"entry", "entry-aba"}:
                        transient = root / "another-secret-name"
                        transient.write_text("new\n", encoding="utf-8")
                        if mutation == "entry-aba":
                            transient.unlink()
                    else:
                        ignored.unlink()
                        if mutation == "directory":
                            ignored.mkdir()
                        else:
                            try:
                                ignored.symlink_to(excluded, target_is_directory=True)
                            except OSError as exc:
                                self.skipTest(f"symlinks are unavailable: {exc}")
                    os.utime(
                        root,
                        ns=(
                            root_metadata.st_atime_ns,
                            root_metadata.st_mtime_ns + 2_000_000_000,
                        ),
                        follow_symlinks=False,
                    )
                    with self.assertRaises(snapshot.RepositorySnapshotError) as raised:
                        session.verify(sealed)
                    self.assertNotIn("secret-name", str(raised.exception))

        limits = (
            ("max_directories", 1, 8, 4, 1024, 4096),
            ("max_entries", 8, 1, 4, 1024, 4096),
            ("max_depth", 8, 8, 0, 1024, 4096),
            ("max_total_name_bytes", 8, 8, 4, 2, 4096),
            ("max_total_structure_bytes", 8, 8, 4, 1024, 1),
        )
        for (
            expected,
            max_directories,
            max_entries,
            max_depth,
            max_names,
            max_structure,
        ) in limits:
            with (
                self.subTest(limit=expected),
                tempfile.TemporaryDirectory(prefix="structure-limits-") as directory,
            ):
                root = Path(directory)
                (root / "one").mkdir()
                (root / "two").write_text("2", encoding="utf-8")
                with (
                    snapshot.SnapshotSession(root) as session,
                    self.assertRaisesRegex(snapshot.RepositorySnapshotError, expected),
                ):
                    session.freeze_directory_structure(
                        max_directories=max_directories,
                        max_entries=max_entries,
                        max_depth=max_depth,
                        max_total_name_bytes=max_names,
                        max_total_structure_bytes=max_structure,
                    )

    def test_inventory_structure_limits_reject_before_any_content_read(self) -> None:
        file_set_type = CHECKER.repository_git.RepositoryFileSet
        for mode in ("archive", "repository"):
            with (
                self.subTest(mode=mode),
                tempfile.TemporaryDirectory(
                    prefix=f"inventory-structure-admission-{mode}-"
                ) as directory,
            ):
                root = Path(directory)
                (root / "a").write_bytes(b"a")
                (root / "b").write_bytes(b"b")
                repository = None
                if mode == "repository":
                    repository = mock.Mock(root=root)
                    repository.snapshot_file_set.return_value = file_set_type(
                        listed=("a", "b"),
                        deleted=(),
                        present=("a", "b"),
                    )
                reads: list[int] = []
                real_read = CHECKER.repository_snapshot._read_regular_file

                def record_read(
                    stream: Any, *, capture_bytes: bool, frozen_size: int
                ) -> Any:
                    reads.append(frozen_size)
                    return real_read(
                        stream,
                        capture_bytes=capture_bytes,
                        frozen_size=frozen_size,
                    )

                with (
                    mock.patch.object(
                        CHECKER.repository_git,
                        "open_repository",
                        return_value=repository,
                    ),
                    mock.patch.object(CHECKER, "SNAPSHOT_MAX_ENTRIES", 1),
                    mock.patch.object(
                        CHECKER.repository_snapshot,
                        "_read_regular_file",
                        side_effect=record_read,
                    ),
                    self.assertRaisesRegex(CHECKER.InventoryError, "max_entries"),
                ):
                    CHECKER._make_public_file_universe(root)
                self.assertEqual([], reads)

    def test_selected_archive_walk_applies_session_structure_limits(self) -> None:
        snapshot = CHECKER.repository_snapshot
        with tempfile.TemporaryDirectory(
            prefix="selected-archive-structure-admission-"
        ) as directory:
            root = Path(directory)
            payload = root / "payload"
            payload.mkdir()
            (payload / "a").write_bytes(b"a")
            (payload / "b").write_bytes(b"b")
            limits = snapshot.DirectoryStructureLimits(
                max_directories=8,
                max_entries=1,
                max_depth=8,
                max_total_name_bytes=64,
                max_total_structure_bytes=256,
            )
            with (
                snapshot.SnapshotSession(
                    root,
                    directory_structure_limits=limits,
                ) as session,
                self.assertRaisesRegex(snapshot.RepositorySnapshotError, "max_entries"),
            ):
                session.walk_archive(("payload",))

            leaf = root / "leaf"
            leaf.write_bytes(b"leaf")
            no_entries = snapshot.DirectoryStructureLimits(
                max_directories=8,
                max_entries=0,
                max_depth=8,
                max_total_name_bytes=64,
                max_total_structure_bytes=256,
            )
            reads: list[int] = []
            real_read = snapshot._read_regular_file

            def record_read(
                stream: Any, *, capture_bytes: bool, frozen_size: int
            ) -> Any:
                reads.append(frozen_size)
                return real_read(
                    stream,
                    capture_bytes=capture_bytes,
                    frozen_size=frozen_size,
                )

            with (
                snapshot.SnapshotSession(
                    root,
                    directory_structure_limits=no_entries,
                ) as session,
                mock.patch.object(
                    snapshot,
                    "_read_regular_file",
                    side_effect=record_read,
                ),
                self.assertRaisesRegex(snapshot.RepositorySnapshotError, "max_entries"),
            ):
                session.walk_archive(("leaf",))
            self.assertEqual([], reads)

            one_entry = snapshot.DirectoryStructureLimits(
                max_directories=no_entries.max_directories,
                max_entries=1,
                max_depth=no_entries.max_depth,
                max_total_name_bytes=no_entries.max_total_name_bytes,
                max_total_structure_bytes=no_entries.max_total_structure_bytes,
            )
            captures = 0
            real_capture = snapshot.SnapshotSession._capture_node

            def count_capture(session: Any, path: str, **kwargs: Any) -> Any:
                nonlocal captures
                captures += 1
                return real_capture(session, path, **kwargs)

            with (
                snapshot.SnapshotSession(
                    root,
                    directory_structure_limits=one_entry,
                ) as session,
                mock.patch.object(
                    snapshot.SnapshotSession,
                    "_capture_node",
                    count_capture,
                ),
                self.assertRaisesRegex(snapshot.RepositorySnapshotError, "max_entries"),
            ):
                session.walk_archive(("leaf", "leaf"))
            self.assertEqual(1, captures)

            first_directory = root / "d1"
            second_directory = root / "d2"
            first_directory.mkdir()
            second_directory.mkdir()
            one_directory = snapshot.DirectoryStructureLimits(
                max_directories=1,
                max_entries=8,
                max_depth=8,
                max_total_name_bytes=64,
                max_total_structure_bytes=256,
            )
            opened: list[str] = []
            real_open = snapshot.os.open

            def record_directory_open(path: Any, *args: Any, **kwargs: Any) -> Any:
                if path in {"d1", "d2"}:
                    opened.append(path)
                return real_open(path, *args, **kwargs)

            with snapshot.SnapshotSession(
                root,
                directory_structure_limits=one_directory,
            ) as session:
                with (
                    mock.patch.object(
                        snapshot.os,
                        "open",
                        side_effect=record_directory_open,
                    ),
                    self.assertRaisesRegex(
                        snapshot.RepositorySnapshotError,
                        "max_directories",
                    ),
                ):
                    session.walk_archive(("d1", "d2"))
            self.assertIn("d1", opened)
            self.assertNotIn("d2", opened)

    def test_structure_directory_and_kind_admission_precede_open_and_read(self) -> None:
        snapshot = CHECKER.repository_snapshot
        with tempfile.TemporaryDirectory(
            prefix="structure-admission-order-"
        ) as directory:
            root = Path(directory)
            (root / "child").mkdir()
            root_only = snapshot.DirectoryStructureLimits(
                max_directories=1,
                max_entries=8,
                max_depth=8,
                max_total_name_bytes=64,
                max_total_structure_bytes=256,
            )
            child_opens: list[str] = []
            real_open = snapshot.os.open

            def record_open(path: Any, *args: Any, **kwargs: Any) -> Any:
                if path == "child":
                    child_opens.append(path)
                return real_open(path, *args, **kwargs)

            with (
                snapshot.SnapshotSession(
                    root,
                    directory_structure_limits=root_only,
                ) as session,
                mock.patch.object(snapshot.os, "open", side_effect=record_open),
                self.assertRaisesRegex(
                    snapshot.RepositorySnapshotError,
                    "max_directories",
                ),
            ):
                session.freeze_directory_structure()
            self.assertEqual([], child_opens)

        with tempfile.TemporaryDirectory(
            prefix="structure-kind-before-read-"
        ) as directory:
            root = Path(directory)
            leaf = root / "leaf"
            leaf.mkdir()
            reads: list[int] = []
            real_read = snapshot._read_regular_file

            def record_read(
                stream: Any, *, capture_bytes: bool, frozen_size: int
            ) -> Any:
                reads.append(frozen_size)
                return real_read(
                    stream,
                    capture_bytes=capture_bytes,
                    frozen_size=frozen_size,
                )

            with snapshot.SnapshotSession(root) as session:
                session.freeze_directory_structure()
                leaf.rmdir()
                leaf.write_bytes(b"payload")
                with (
                    mock.patch.object(
                        snapshot,
                        "_read_regular_file",
                        side_effect=record_read,
                    ),
                    self.assertRaisesRegex(
                        snapshot.RepositorySnapshotError,
                        "kind changed",
                    ),
                ):
                    session.walk_archive_root()
            self.assertEqual([], reads)

    def test_directory_structure_iteration_stops_at_the_first_excess_entry(
        self,
    ) -> None:
        snapshot = CHECKER.repository_snapshot

        class Entry:
            def __init__(self, name: str) -> None:
                self.name = name

        class Scandir:
            def __init__(self, names: list[str]) -> None:
                self.names = iter(names)
                self.consumed = 0

            def __enter__(self) -> Any:
                return self

            def __exit__(self, *_: Any) -> None:
                return None

            def __iter__(self) -> Any:
                return self

            def __next__(self) -> Entry:
                name = next(self.names)
                self.consumed += 1
                return Entry(name)

        with tempfile.TemporaryDirectory(
            prefix="structure-stream-admission-"
        ) as directory:
            root = Path(directory)
            names = [f"entry-{index}" for index in range(5)]
            for name in names:
                (root / name).write_bytes(b"")
            iterator = Scandir(names)
            limits = snapshot.DirectoryStructureLimits(
                max_directories=1,
                max_entries=1,
                max_depth=1,
                max_total_name_bytes=1024,
                max_total_structure_bytes=4096,
            )
            with snapshot.SnapshotSession(
                root,
                directory_structure_limits=limits,
            ) as session:
                with (
                    mock.patch.object(snapshot.os, "scandir", return_value=iterator),
                    self.assertRaisesRegex(
                        snapshot.RepositorySnapshotError,
                        "max_entries",
                    ),
                ):
                    session.freeze_directory_structure()
            self.assertEqual(2, iterator.consumed)

    def test_git_inventory_fixed_point_order_and_adjacent_mutations(self) -> None:
        file_set_type = CHECKER.repository_git.RepositoryFileSet

        def file_set(*present: str, deleted: tuple[str, ...] = ()) -> Any:
            listed = tuple(sorted((*present, *deleted)))
            return file_set_type(
                listed=listed,
                deleted=tuple(sorted(deleted)),
                present=tuple(sorted(present)),
            )

        with tempfile.TemporaryDirectory(prefix="inventory-fixed-point-") as directory:
            root = Path(directory)
            (root / "original.py").write_text("pass\n", encoding="utf-8")
            (root / "ignored-secret-name").write_text("private\n", encoding="utf-8")
            repository = mock.Mock(root=root)
            repository.snapshot_file_set.return_value = file_set("original.py")
            events: list[str] = []
            real_seal = CHECKER.repository_snapshot.SnapshotSession.seal
            real_verify = CHECKER.repository_snapshot.SnapshotSession.verify
            real_freeze = (
                CHECKER.repository_snapshot.SnapshotSession.freeze_directory_structure
            )

            def record_snapshot() -> Any:
                events.append(f"G{repository.snapshot_file_set.call_count - 1}")
                return file_set("original.py")

            def record_seal(session: Any) -> Any:
                result = real_seal(session)
                events.append("F0")
                return result

            def record_freeze(session: Any, *args: Any, **kwargs: Any) -> Any:
                result = real_freeze(session, *args, **kwargs)
                events.append("S")
                return result

            def record_verify(session: Any, tree: Any = None) -> Any:
                result = real_verify(session, tree)
                if session._snapshot is not None:
                    events.append("F1")
                return result

            repository.snapshot_file_set.side_effect = record_snapshot
            with (
                mock.patch.object(
                    CHECKER.repository_git,
                    "open_repository",
                    return_value=repository,
                ),
                mock.patch.object(
                    CHECKER.repository_snapshot.SnapshotSession,
                    "seal",
                    record_seal,
                ),
                mock.patch.object(
                    CHECKER.repository_snapshot.SnapshotSession,
                    "freeze_directory_structure",
                    record_freeze,
                ),
                mock.patch.object(
                    CHECKER.repository_snapshot.SnapshotSession,
                    "verify",
                    record_verify,
                ),
            ):
                universe = CHECKER._make_public_file_universe(root)
            self.assertEqual(["G0", "S", "F0", "G1", "F1", "G2"], events)
            self.assertEqual(("original.py",), universe.paths)

        scenarios = (
            "late-after-g0",
            "stage-after-f0",
            "kind-before-f0",
            "symlink-before-f0",
            "delete-after-g1",
            "restore-after-g1",
            "late-after-f1",
        )
        for scenario in scenarios:
            with (
                self.subTest(scenario=scenario),
                tempfile.TemporaryDirectory(
                    prefix="inventory-fixed-point-race-"
                ) as directory,
            ):
                root = Path(directory)
                original = root / "original.py"
                original.write_text("pass\n", encoding="utf-8")
                deleted = root / "deleted.py"
                staged = root / "ignored.py"
                if scenario == "restore-after-g1":
                    initial = file_set("original.py", deleted=("deleted.py",))
                else:
                    initial = file_set("original.py")
                if scenario == "stage-after-f0":
                    staged.write_text("ignored before staging\n", encoding="utf-8")

                repository = mock.Mock(root=root)
                real_seal = CHECKER.repository_snapshot.SnapshotSession.seal
                real_freeze = CHECKER.repository_snapshot.SnapshotSession.freeze_directory_structure
                real_verify = CHECKER.repository_snapshot.SnapshotSession.verify
                staged_now = False
                late_now = False

                def snapshot() -> Any:
                    nonlocal late_now, staged_now
                    call = repository.snapshot_file_set.call_count
                    if late_now:
                        result = file_set("original.py", "late.py")
                    elif staged_now:
                        result = file_set("original.py", "ignored.py")
                    else:
                        result = initial
                    if call == 1 and scenario == "late-after-g0":
                        (root / "late.py").write_text("late\n", encoding="utf-8")
                        late_now = True
                    elif call == 2 and scenario == "delete-after-g1":
                        original.unlink()
                    elif call == 2 and scenario == "restore-after-g1":
                        deleted.write_text("restored\n", encoding="utf-8")
                    return result

                def freeze_then_mutate(session: Any, *args: Any, **kwargs: Any) -> Any:
                    result = real_freeze(session, *args, **kwargs)
                    if scenario in {"kind-before-f0", "symlink-before-f0"}:
                        original.unlink()
                        if scenario == "kind-before-f0":
                            original.mkdir()
                        else:
                            target = root / "target.py"
                            target.write_text("target\n", encoding="utf-8")
                            original.symlink_to(target)
                    return result

                def seal_then_mutate(session: Any) -> Any:
                    nonlocal staged_now
                    result = real_seal(session)
                    if scenario == "stage-after-f0":
                        staged_now = True
                    return result

                def verify_then_mutate(session: Any, tree: Any = None) -> Any:
                    result = real_verify(session, tree)
                    if session._snapshot is not None and scenario == "late-after-f1":
                        (root / "late.py").write_text("late\n", encoding="utf-8")
                        repository.snapshot_file_set.side_effect = lambda: file_set(
                            "original.py", "late.py"
                        )
                    return result

                repository.snapshot_file_set.side_effect = snapshot
                with (
                    mock.patch.object(
                        CHECKER.repository_git,
                        "open_repository",
                        return_value=repository,
                    ),
                    mock.patch.object(
                        CHECKER.repository_snapshot.SnapshotSession,
                        "freeze_directory_structure",
                        freeze_then_mutate,
                    ),
                    mock.patch.object(
                        CHECKER.repository_snapshot.SnapshotSession,
                        "seal",
                        seal_then_mutate,
                    ),
                    mock.patch.object(
                        CHECKER.repository_snapshot.SnapshotSession,
                        "verify",
                        verify_then_mutate,
                    ),
                    self.assertRaises(CHECKER.InventoryError),
                ):
                    CHECKER._make_public_file_universe(root)

    def _launches_for_source(self, source: str) -> list[dict[str, Any]]:
        with tempfile.TemporaryDirectory(prefix="python-launch-fixture-") as directory:
            root = Path(directory)
            path = root / "fixture.py"
            path.write_text(source, encoding="utf-8")
            return CHECKER._discover_python_launches(root)

    def _load_checker_mutant(self, original: str, replacement: str) -> Any:
        source = CHECKER_PATH.read_text(encoding="utf-8")
        self.assertEqual(1, source.count(original))
        mutated = source.replace(original, replacement, 1)
        name = (
            "check_build_inventory_mutant_"
            + hashlib.sha256(mutated.encode("utf-8")).hexdigest()[:16]
        )
        with tempfile.TemporaryDirectory(prefix="build-inventory-mutant-") as directory:
            path = Path(directory) / "check_build_inventory.py"
            path.write_text(mutated, encoding="utf-8")
            for dependency in ("repository_git.py", "repository_snapshot.py"):
                shutil.copy2(
                    CHECKER_PATH.with_name(dependency), path.with_name(dependency)
                )
            specification = importlib.util.spec_from_file_location(name, path)
            assert specification is not None and specification.loader is not None
            module = importlib.util.module_from_spec(specification)
            specification.loader.exec_module(module)
            return module

    def _launches_for_files_with_checker(
        self, checker: Any, files: dict[str, str]
    ) -> list[dict[str, Any]]:
        with tempfile.TemporaryDirectory(prefix="python-alias-mutant-") as directory:
            root = Path(directory)
            for relative, source in files.items():
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(source, encoding="utf-8")
            return checker._discover_python_launches(root)

    def test_alias_timeline_supports_out_of_order_cutoff_queries(self) -> None:
        source = (
            "import subprocess\n"
            "def nested():\n"
            " return late_runner(['nested'])\n"
            "early_runner=subprocess.run\n"
            "early_runner(['early'])\n"
            "early_runner=None\n"
            "late_runner=subprocess.run\n"
            "nested()\n"
        )
        first = self._launches_for_source(source)
        second = self._launches_for_source(source)
        self.assertEqual(first, second)
        self.assertEqual(
            [
                "python-launch:fixture.py:module:subprocess.run:1",
                "python-launch:fixture.py:nested:subprocess.run:1",
            ],
            [item["id"] for item in first],
        )
        self.assertTrue(
            all(
                CHECKER.re.fullmatch(r"[0-9a-f]{64}", item["call_semantics_digest"])
                for item in first
            )
        )

    def test_alias_timeline_defers_late_declaration_error_until_crossed(self) -> None:
        source = (
            "import subprocess as process\n"
            "process.run(['early'])\n"
            "if condition:\n"
            " process=object()\n"
            "process.run(['late'])\n"
        )
        with self.assertRaisesRegex(
            CHECKER.InventoryError,
            r"fixture\.py:4: control-flow-dependent process alias assignment",
        ):
            self._launches_for_source(source)

    def test_alias_timeline_does_not_transfer_future_error_before_cutoff(self) -> None:
        early_only = (
            "import subprocess as process\n"
            "process.run(['early'])\n"
            "if condition:\n"
            " process=object()\n"
        )
        self.assertEqual(1, len(self._launches_for_source(early_only)))
        mutant = self._load_checker_mutant(
            "cutoff = bisect.bisect_left(timeline.positions, position)",
            "cutoff = bisect.bisect_left(timeline.positions, position) + 1",
        )
        with self.assertRaisesRegex(
            mutant.InventoryError,
            r"fixture\.py:4: control-flow-dependent process alias assignment",
        ):
            self._launches_for_files_with_checker(mutant, {"fixture.py": early_only})

    def test_alias_timeline_tracks_qualified_child_removal_in_deltas(self) -> None:
        provider = "import subprocess\nlaunch=subprocess.run\n"
        consumers = (
            (
                "delete",
                "import provider as module\n"
                "def nested():\n"
                " return module.launch(['nested'])\n"
                "del module\n"
                "module.launch(['removed'])\n"
                "import provider as module\n"
                "nested()\n",
            ),
            (
                "rebind",
                "import provider as module\n"
                "def nested():\n"
                " return module.launch(['nested'])\n"
                "module=object()\n"
                "module.launch(['removed'])\n"
                "import provider as module\n"
                "nested()\n",
            ),
        )
        pop_mutant = self._load_checker_mutant(
            """            def pop(self, key: str, *default: str) -> str:
                self.remember(key)
                return super().pop(key, *default)
""",
            """            def pop(self, key: str, *default: str) -> str:
                return super().pop(key, *default)
""",
        )
        for label, consumer in consumers:
            files = {"provider.py": provider, "consumer.py": consumer}
            with self.subTest(label=label):
                baseline = self._launches_for_files_with_checker(CHECKER, files)
                self.assertEqual(1, len(baseline))
                mutated = self._launches_for_files_with_checker(pop_mutant, files)
                self.assertEqual(2, len(mutated))

    def test_alias_timeline_adapter_overlay_requires_strictly_later_position(
        self,
    ) -> None:
        owner_source = "def run_command(argv):\n return argv\n"
        consumer_source = self._reviewed_loader_source()

        def equal_position_launches(checker: Any) -> list[dict[str, Any]]:
            with tempfile.TemporaryDirectory(
                prefix="adapter-equal-position-"
            ) as directory:
                root = Path(directory).resolve()
                owner = root / "tools/observe_abi_baseline.py"
                owner.parent.mkdir(parents=True)
                owner.write_text(owner_source, encoding="utf-8")
                consumer = root / "test/abi/baseline/test_observe_abi_baseline.py"
                consumer.parent.mkdir(parents=True)
                consumer.write_text(consumer_source, encoding="utf-8")
                trees = {
                    owner: checker.ast.parse(owner_source),
                    consumer: checker.ast.parse(consumer_source),
                }
                calls = [
                    node
                    for node in checker.ast.walk(trees[consumer])
                    if isinstance(node, checker.ast.Call)
                ]
                exec_call = next(
                    node
                    for node in calls
                    if isinstance(node.func, checker.ast.Attribute)
                    and node.func.attr == "exec_module"
                )
                adapter_call = next(
                    node
                    for node in calls
                    if isinstance(node.func, checker.ast.Attribute)
                    and node.func.attr == "run_command"
                )
                adapter_call.lineno = exec_call.lineno
                adapter_call.col_offset = exec_call.col_offset
                return checker._discover_python_launches(
                    root,
                    _python_files_override=[owner, consumer],
                    _trees_override=trees,
                )

        baseline = equal_position_launches(CHECKER)
        self.assertFalse(any(item["anchor"].get("origin_kind") for item in baseline))
        mutant = self._load_checker_mutant(
            "position is None or position > reviewed_adapter_exec_position",
            "position is None or position >= reviewed_adapter_exec_position",
        )
        mutated = equal_position_launches(mutant)
        self.assertEqual(
            1,
            sum(bool(item["anchor"].get("origin_kind")) for item in mutated),
        )

    def test_alias_timeline_return_copy_isolation_and_unit_identity(self) -> None:
        launches = self._launches_for_source(
            "import subprocess as process\n"
            "[process.run(['hidden']) for process in ()]\n"
            "process.run(['visible'])\n"
        )
        self.assertEqual(
            ["python-launch:fixture.py:module:subprocess.run:1"],
            [item["id"] for item in launches],
        )

        with tempfile.TemporaryDirectory(prefix="alias-unit-identity-") as directory:
            root = Path(directory)
            (root / "tracked.py").write_text(
                "import subprocess\nsubprocess.run(['tracked'])\n",
                encoding="utf-8",
            )
            (root / "shadowed.py").write_text(
                "subprocess=object()\nsubprocess.run(['shadowed'])\n",
                encoding="utf-8",
            )
            unit_launches = CHECKER._discover_python_launches(root)
        self.assertEqual(
            ["python-launch:tracked.py:module:subprocess.run:1"],
            [item["id"] for item in unit_launches],
        )

    def test_alias_timeline_transfers_three_hundred_declarations_once(self) -> None:
        source = "import subprocess\nrunner0=subprocess.run\n"
        source += "".join(
            f"runner{index}=runner{index - 1}\n" for index in range(1, 300)
        )
        source += "".join("runner299(['tool'])\n" for _ in range(300))
        with mock.patch.object(
            CHECKER,
            "_set_python_alias",
            wraps=CHECKER._set_python_alias,
        ) as set_alias:
            launches = self._launches_for_source(source)
        self.assertEqual(300, len(launches))
        self.assertLessEqual(set_alias.call_count, 3_000)
        self.assertTrue(
            all(
                CHECKER.re.fullmatch(r"[0-9a-f]{64}", item["call_semantics_digest"])
                for item in launches
            )
        )

    def test_python_call_semantics_bind_local_launch_behavior_not_formatting(
        self,
    ) -> None:
        baseline = (
            "import subprocess\n"
            "def launch():\n"
            "    command = ['tool', 'one']\n"
            "    return subprocess.run(command, cwd='.', env={'X': '1'})\n"
        )
        formatted = (
            "import subprocess\n\n"
            "# formatting and comments are not semantics\n"
            "def launch( ):\n"
            "    command=['tool','one']\n"
            "    return subprocess.run( command, cwd = '.', env = {'X':'1'} )\n"
        )
        first = self._launches_for_source(baseline)[0]["call_semantics_digest"]
        second = self._launches_for_source(formatted)[0]["call_semantics_digest"]
        self.assertEqual(first, second)
        for old, new in (
            ("'one'", "'two'"),
            ("cwd='.'", "cwd='nested'"),
            ("'X': '1'", "'X': '2'"),
            ("subprocess.run(command", "subprocess.run(command + ['extra']"),
            ("env={'X': '1'}", "env={'X': '1'}, text=True"),
        ):
            with self.subTest(change=(old, new)):
                changed = self._launches_for_source(baseline.replace(old, new))[0]
                self.assertNotEqual(first, changed["call_semantics_digest"])

    def test_every_python_launch_requires_semantics_digest(self) -> None:
        inventory = self._inventory()
        inventory["python_launches"][0].pop("call_semantics_digest")
        self._write_inventory(inventory)
        self._assert_error_contains("missing call_semantics_digest")

    def test_source_discovery_recursion_error_fails_closed(self) -> None:
        original = CHECKER.discover

        def recursive_failure(*_args, **_kwargs):
            raise RecursionError("fixture recursion bound")

        CHECKER.discover = recursive_failure
        try:
            self._assert_error_contains("source discovery failed closed")
        finally:
            CHECKER.discover = original

    def test_embedded_python_carrier_ids_are_stable_and_recursive(self) -> None:
        launches = self._launches_for_source(
            "import subprocess,sys\n"
            "def launch():\n"
            "    subprocess.run((sys.executable, '-c', \"import subprocess as sp; popen = sp.Popen; popen(['one'])\"))\n"
            "    subprocess.run((sys.executable, '-c', \"import subprocess,sys; subprocess.run([sys.executable,'-c',\\\"import subprocess;subprocess.Popen(['two'])\\\"])\"))\n"
        )
        embedded = [item for item in launches if item["anchor"].get("origin_kind")]
        self.assertEqual(3, len(embedded))
        self.assertEqual(
            {1, 2}, {item["anchor"]["embedded_depth"] for item in embedded}
        )
        self.assertTrue(
            all(item["anchor"]["carrier_semantics_digests"] for item in embedded)
        )
        self.assertTrue(
            all(
                len(item["anchor"]["carrier_ordinals"])
                == item["anchor"]["embedded_depth"]
                and item["anchor"]["carrier_ordinal"]
                == item["anchor"]["carrier_ordinals"][-1]
                for item in embedded
            )
        )
        self.assertEqual(
            {"argv-arg0"},
            {item["anchor"]["carrier_shapes"][0] for item in embedded},
        )

    def test_embedded_python_carriers_cover_reviewed_api_shapes(self) -> None:
        code = "import subprocess; subprocess.Popen(['embedded'])"
        launches = self._launches_for_source(
            "import asyncio,os,pty,subprocess,sys as runtime\n"
            "async def launch():\n"
            "    interpreter = runtime.executable\n"
            f"    subprocess.call([interpreter,'-c',{code!r}])\n"
            f"    subprocess.check_output([runtime.executable,'-I','-X','dev','-c',{code!r}])\n"
            f"    subprocess.run(['placeholder','-c',{code!r}], executable=runtime.executable)\n"
            f"    pty.spawn((runtime.executable,'-c',{code!r}))\n"
            f"    os.execv(runtime.executable,(runtime.executable,'-c',{code!r}))\n"
            f"    os.spawnv(os.P_NOWAIT,runtime.executable,(runtime.executable,'-c',{code!r}))\n"
            f"    os.execl(runtime.executable,runtime.executable,'-c',{code!r})\n"
            f"    await asyncio.create_subprocess_exec(runtime.executable,'-c',{code!r})\n"
            "    loop = asyncio.get_running_loop()\n"
            f"    await loop.subprocess_exec(lambda:None,runtime.executable,'-c',{code!r})\n"
        )
        embedded = [item for item in launches if item["anchor"].get("origin_kind")]
        self.assertEqual(9, len(embedded))
        self.assertEqual(
            {
                "argv-arg0",
                "path+argv",
                "mode+path+argv",
                "path+positional-argv",
                "asyncio-varargs",
                "loop-protocol+varargs",
            },
            {item["anchor"]["carrier_shapes"][0] for item in embedded},
        )

    def test_direct_windows_python_interpreter_names_are_carriers(self) -> None:
        code = "import subprocess; subprocess.run(['embedded'])"
        launches = self._launches_for_source(
            "import subprocess\n"
            f"subprocess.run(('python.exe','-c',{code!r}))\n"
            f"subprocess.run(('pythonw3.14.exe','-c',{code!r}))\n"
        )
        embedded = [item for item in launches if item["anchor"].get("origin_kind")]
        self.assertEqual(2, len(embedded))

    def test_embedded_python_reuses_local_module_export_analysis(self) -> None:
        with tempfile.TemporaryDirectory(prefix="embedded-local-export-") as directory:
            root = Path(directory)
            (root / "helper.py").write_text(
                "import subprocess\nlaunch = subprocess.run\n", encoding="utf-8"
            )
            (root / "carrier.py").write_text(
                "import subprocess, sys\n"
                'code = \'from helper import launch\\nlaunch(["python3", "-V"])\'\n'
                "subprocess.run((sys.executable, '-c', code))\n",
                encoding="utf-8",
            )

            launches = CHECKER._discover_python_launches(root)

        embedded = [item for item in launches if item["anchor"].get("origin_kind")]
        self.assertEqual(1, len(embedded))
        self.assertEqual("subprocess.run", embedded[0]["anchor"]["symbol"])
        self.assertEqual("carrier.py", embedded[0]["anchor"]["file"])
        self.assertEqual("module", embedded[0]["anchor"]["embedded_scope"])

    def test_generic_python_argv_helper_is_not_a_carrier(self) -> None:
        launches = self._launches_for_source(
            "import sys\ndef invoke(argv): pass\n"
            "def launch(): invoke((sys.executable,'-c','import subprocess;subprocess.run([\"hidden\"])'))\n"
        )
        self.assertFalse(any(item["anchor"].get("origin_kind") for item in launches))

    def test_carrier_literal_resolution_is_ordered_and_control_safe(self) -> None:
        accepted = self._launches_for_source(
            "import subprocess,sys\n"
            "def f(): code='import sub' + 'process;subprocess.run([\\\"ok\\\"])'; "
            "subprocess.run((sys.executable,'-c',code))\n"
        )
        self.assertEqual(
            1,
            sum(bool(item["anchor"].get("origin_kind")) for item in accepted),
        )
        rejected = (
            "import subprocess,sys\ndef f():\n    subprocess.run((sys.executable,'-c',code))\n    code='pass'\n",
            "import subprocess,sys\ndef f(flag):\n    if flag: code='pass'\n    subprocess.run((sys.executable,'-c',code))\n",
            "import subprocess,sys\ndef f():\n    a=b\n    b=a\n    subprocess.run((sys.executable,'-c',a))\n",
        )
        for source in rejected:
            with self.subTest(source=source):
                with self.assertRaisesRegex(CHECKER.InventoryError, "static string"):
                    self._launches_for_source(source)

    def test_carrier_literals_use_runtime_scopes_and_assignment_time_values(
        self,
    ) -> None:
        embedded_code = "import subprocess;subprocess.run(['embedded'])"
        accepted = (
            "import subprocess,sys\n"
            f"code={embedded_code!r}\n"
            "class C:\n"
            f"    local={embedded_code!r}\n"
            "    subprocess.run((sys.executable,'-c',local))\n"
            "factory=lambda value=subprocess.run((sys.executable,'-c',code)): value\n"
        )
        launches = self._launches_for_source(accepted)
        self.assertEqual(
            2,
            sum(bool(item["anchor"].get("origin_kind")) for item in launches),
        )

        rejected = (
            "import subprocess,sys\ncode='pass'\n(lambda code: subprocess.run((sys.executable,'-c',code)))('pass')\n",
            "import subprocess,sys\n(lambda code='pass': subprocess.run((sys.executable,'-c',code)))()\n",
            "import subprocess,sys\ndef f(values):\n code='pass'\n for code in values: pass\n subprocess.run((sys.executable,'-c',code))\n",
            "import subprocess,sys\ndef f(resource):\n code='pass'\n with resource as code: pass\n subprocess.run((sys.executable,'-c',code))\n",
            "import subprocess,sys\ndef f(values):\n code='pass'\n return [subprocess.run((sys.executable,'-c',code)) for code in values]\n",
            "import subprocess,sys\ndef f():\n a=b\n b='pass'\n subprocess.run((sys.executable,'-c',a))\n",
            "import subprocess,sys\ncode='pass'\nclass C:\n for code in ('pass',): pass\n subprocess.run((sys.executable,'-c',code))\n",
        )
        for source in rejected:
            with self.subTest(source=source):
                with self.assertRaisesRegex(CHECKER.InventoryError, "static string"):
                    self._launches_for_source(source)

    def test_nested_class_carriers_skip_every_enclosing_class_namespace(
        self,
    ) -> None:
        embedded_code = "import subprocess;subprocess.run(['embedded'])"
        accepted = (
            "import subprocess,sys\n"
            "def dynamic_code(): return input()\n"
            f"code={embedded_code!r}\n"
            "class Outer:\n"
            " code=dynamic_code()\n"
            " class Inner:\n"
            "  subprocess.run((sys.executable,'-c',code))\n"
            "def owner():\n"
            f" code={embedded_code!r}\n"
            " class Outer:\n"
            "  code=dynamic_code()\n"
            "  class Inner:\n"
            "   subprocess.run((sys.executable,'-c',code))\n"
        )
        launches = self._launches_for_source(accepted)
        embedded = [item for item in launches if item["anchor"].get("origin_kind")]
        self.assertEqual(2, len(embedded))

        rejected = (
            "import subprocess,sys\n"
            "def dynamic_code(): return input()\n"
            "code=dynamic_code()\n"
            "class Outer:\n"
            f" code={embedded_code!r}\n"
            " class Inner:\n"
            "  subprocess.run((sys.executable,'-c',code))\n",
            "import subprocess,sys\n"
            "def owner(code):\n"
            " class Outer:\n"
            f"  code={embedded_code!r}\n"
            "  class Inner:\n"
            "   subprocess.run((sys.executable,'-c',code))\n",
        )
        for source in rejected:
            with self.subTest(source=source):
                with self.assertRaisesRegex(CHECKER.InventoryError, "static string"):
                    self._launches_for_source(source)

    def test_class_future_bindings_preserve_inherited_carrier_fact(self) -> None:
        embedded_code = "import subprocess;subprocess.run(['embedded'])"
        source = (
            "import subprocess,sys\n"
            f"code={embedded_code!r}\n"
            "class C:\n"
            " subprocess.run((sys.executable,'-c',code))\n"
            " code='pass'\n"
            " code='pass'\n"
        )
        launches = self._launches_for_source(source)
        self.assertEqual(
            1,
            sum(bool(item["anchor"].get("origin_kind")) for item in launches),
        )

        function_source = (
            "import subprocess,sys\n"
            f"code={embedded_code!r}\n"
            "def f():\n"
            " subprocess.run((sys.executable,'-c',code))\n"
            " code='pass'\n"
        )
        with self.assertRaisesRegex(CHECKER.InventoryError, "static string"):
            self._launches_for_source(function_source)

    def test_carrier_literals_reject_every_non_direct_rebinding_kind(self) -> None:
        sources = (
            "import subprocess,sys\ndef f():\n code='pass'\n (code := 'pass')\n subprocess.run((sys.executable,'-c',code))\n",
            "import subprocess,sys\ndef f():\n code='pass'\n del code\n subprocess.run((sys.executable,'-c',code))\n",
            "import subprocess,sys\ndef f():\n code='pass'\n import json as code\n subprocess.run((sys.executable,'-c',code))\n",
            "import subprocess,sys\ndef f():\n code='pass'\n def code(): pass\n subprocess.run((sys.executable,'-c',code))\n",
            "import subprocess,sys\ndef f():\n code='pass'\n class code: pass\n subprocess.run((sys.executable,'-c',code))\n",
            "import subprocess,sys\ndef f():\n code='pass'\n try: raise ValueError\n except ValueError as code: pass\n subprocess.run((sys.executable,'-c',code))\n",
            "import subprocess,sys\ndef f(value):\n code='pass'\n match value:\n  case code: pass\n subprocess.run((sys.executable,'-c',code))\n",
        )
        for source in sources:
            with self.subTest(source=source):
                with self.assertRaisesRegex(CHECKER.InventoryError, "static string"):
                    self._launches_for_source(source)

    def test_outer_carrier_semantics_bind_embedded_launch_digest(self) -> None:
        source = (
            "import subprocess,sys\ndef f():\n"
            "    subprocess.run((sys.executable,'-c','import subprocess;subprocess.run([\\\"x\\\"])'),cwd='one',env={'A':'1'})\n"
        )
        formatted = (
            "import subprocess, sys\n\ndef f( ):\n"
            "    # formatting is not semantic\n"
            "    subprocess.run( (sys.executable, '-c', 'import subprocess;subprocess.run([\\\"x\\\"])'), cwd = 'one', env = {'A': '1'} )\n"
        )

        def embedded_digest(text: str) -> str:
            return next(
                item["call_semantics_digest"]
                for item in self._launches_for_source(text)
                if item["anchor"].get("origin_kind")
            )

        baseline = embedded_digest(source)
        self.assertEqual(baseline, embedded_digest(formatted))
        self.assertNotEqual(
            baseline, embedded_digest(source.replace("cwd='one'", "cwd='two'"))
        )
        self.assertNotEqual(
            baseline, embedded_digest(source.replace("'A':'1'", "'A':'2'"))
        )

    def test_reviewed_observer_adapter_definition_binds_carrier(self) -> None:
        with tempfile.TemporaryDirectory(prefix="reviewed-adapter-") as directory:
            root = Path(directory)
            adapter = root / "tools/observe_abi_baseline.py"
            adapter.parent.mkdir(parents=True)
            adapter.write_text(
                "def run_command(argv):\n    return argv\n", encoding="utf-8"
            )
            consumer = root / "test/abi/baseline/test_observe_abi_baseline.py"
            consumer.parent.mkdir(parents=True)
            consumer.write_text(
                "import importlib.util,sys\nfrom pathlib import Path\n"
                "ROOT=Path(__file__).resolve().parents[3]\n"
                "MODULE_PATH=ROOT/'tools/observe_abi_baseline.py'\n"
                "SPEC=importlib.util.spec_from_file_location('observe_abi_baseline',MODULE_PATH)\n"
                "observer=importlib.util.module_from_spec(SPEC)\n"
                "sys.modules[SPEC.name]=observer\n"
                "SPEC.loader.exec_module(observer)\n"
                "observer.run_command((sys.executable,'-c','import subprocess;subprocess.run([\\\"x\\\"])'))\n",
                encoding="utf-8",
            )
            (root / "direct.py").write_text(
                "import sys\nimport tools.observe_abi_baseline as observer\n"
                "observer.run_command((sys.executable,'-c','import subprocess;subprocess.run([\\\"x\\\"])'))\n",
                encoding="utf-8",
            )
            (root / "alias.py").write_text(
                "import sys\nimport tools.observe_abi_baseline as observer\n"
                "launch=observer.run_command\n"
                "launch((sys.executable,'-c','import subprocess;subprocess.run([\\\"x\\\"])'))\n",
                encoding="utf-8",
            )
            (root / "from_import.py").write_text(
                "import sys\nfrom tools.observe_abi_baseline import run_command\n"
                "run_command((sys.executable,'-c','import subprocess;subprocess.run([\\\"x\\\"])'))\n",
                encoding="utf-8",
            )
            (root / "bridge.py").write_text(
                "from tools.observe_abi_baseline import run_command\n",
                encoding="utf-8",
            )
            (root / "reexport.py").write_text(
                "import sys\nfrom bridge import run_command\n"
                "run_command((sys.executable,'-c','import subprocess;subprocess.run([\\\"x\\\"])'))\n",
                encoding="utf-8",
            )
            first_launches = [
                item
                for item in CHECKER._discover_python_launches(root)
                if item["anchor"].get("origin_kind")
            ]
            self.assertEqual(5, len(first_launches))
            self.assertEqual(
                {"reviewed-adapter-argv0"},
                {item["anchor"]["carrier_shapes"][0] for item in first_launches},
            )
            first_digests = {
                item["anchor"]["adapter_definition_digest"] for item in first_launches
            }
            self.assertEqual(1, len(first_digests))
            adapter.write_text(
                "def run_command(argv):\n    checked=tuple(argv)\n    return checked\n",
                encoding="utf-8",
            )
            second_digests = {
                item["anchor"]["adapter_definition_digest"]
                for item in CHECKER._discover_python_launches(root)
                if item["anchor"].get("origin_kind")
            }
            self.assertEqual(1, len(second_digests))
            self.assertNotEqual(first_digests, second_digests)

    def test_reviewed_observer_dynamic_popen_is_exactly_bound(self) -> None:
        real_launches = CHECKER._discover_python_launches(REPOSITORY_ROOT)
        reviewed_ids = {
            "python-launch:tools/observe_abi_baseline.py:run_command:subprocess.Popen:1",
            "python-launch:tools/observe_abi_baseline.py:main:subprocess.Popen:1",
        }
        reviewed = [item for item in real_launches if item["id"] in reviewed_ids]
        self.assertEqual(reviewed_ids, {item["id"] for item in reviewed})
        self.assertEqual({"subprocess.Popen"}, {item["call"] for item in reviewed})
        self.assertFalse(any("carrier_semantics_digest" in item for item in reviewed))
        bounded = next(
            item
            for item in real_launches
            if item["id"]
            == "python-launch:tools/verify_abi_artifact_parity.py:run_bounded:subprocess.Popen:1"
        )
        self.assertNotIn("carrier_semantics_digest", bounded)

        loader = self._reviewed_loader_source()
        real_owner = (REPOSITORY_ROOT / "tools/observe_abi_baseline.py").read_text(
            encoding="utf-8"
        )
        changed_owners = (
            real_owner.replace(
                "        process = subprocess.Popen(\n",
                "        subprocess.Popen(launch_argv)\n"
                "        process = subprocess.Popen(\n",
                1,
            ),
            real_owner.replace(
                '"--internal-windows-supervisor"',
                '"--changed-windows-supervisor"',
                1,
            ),
            real_owner.replace(
                'raw_args[:1] == ["--internal-windows-supervisor"]',
                'raw_args[:1] == ["--changed-windows-supervisor"]',
                1,
            ),
        )
        for owner in changed_owners:
            with self.subTest(owner=owner):
                with self.assertRaisesRegex(
                    CHECKER.InventoryError, "reviewed Windows supervisor"
                ):
                    self._reviewed_adapter_fixture_launches(owner, loader)

        with tempfile.TemporaryDirectory(prefix="copied-dynamic-popen-") as directory:
            root = Path(directory)
            (root / "copied.py").write_text(
                "import subprocess\ndef run_command(argv):\n return subprocess.Popen(argv)\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(CHECKER.InventoryError, "proof is dynamic"):
                CHECKER._discover_python_launches(root)

        verifier_source = (
            REPOSITORY_ROOT / "tools/verify_abi_artifact_parity.py"
        ).read_text(encoding="utf-8")
        for changed in (
            verifier_source.replace(
                "        process = subprocess.Popen(\n",
                "        subprocess.Popen(list(argv))\n"
                "        process = subprocess.Popen(\n",
                1,
            ),
            verifier_source.replace(
                "            start_new_session=True,",
                "            start_new_session=False,",
                1,
            ),
        ):
            with self.subTest(verifier_change=changed):
                with tempfile.TemporaryDirectory(
                    prefix="changed-bounded-popen-"
                ) as directory:
                    root = Path(directory)
                    verifier = root / "tools/verify_abi_artifact_parity.py"
                    verifier.parent.mkdir(parents=True)
                    verifier.write_text(changed, encoding="utf-8")
                    with self.assertRaisesRegex(
                        CHECKER.InventoryError, "reviewed run_bounded Popen"
                    ):
                        CHECKER._discover_python_launches(root)

    def _reviewed_adapter_fixture_launches(
        self, owner_source: str, consumer_source: str
    ) -> list[dict[str, Any]]:
        with tempfile.TemporaryDirectory(prefix="reviewed-loader-flow-") as directory:
            root = Path(directory)
            owner = root / "tools/observe_abi_baseline.py"
            owner.parent.mkdir(parents=True)
            owner.write_text(owner_source, encoding="utf-8")
            consumer = root / "test/abi/baseline/test_observe_abi_baseline.py"
            consumer.parent.mkdir(parents=True)
            consumer.write_text(consumer_source, encoding="utf-8")
            return CHECKER._discover_python_launches(root)

    @staticmethod
    def _reviewed_loader_source(
        *,
        root_name: str = "ROOT",
        path_name: str = "MODULE_PATH",
        spec_name: str = "SPEC",
        module_name: str = "observer",
        between_root_and_path: str = "",
        between_path_and_spec: str = "",
        between_spec_and_module: str = "",
        between_module_and_register: str = "",
        register: bool = True,
        execute: bool = True,
        call_before_exec: bool = False,
        exec_before_register: bool = False,
        guard_message: str | None = None,
    ) -> str:
        lines = [
            "import importlib.util,sys",
            "from pathlib import Path",
            f"{root_name}=Path(__file__).resolve().parents[3]",
        ]
        if between_root_and_path:
            lines.append(between_root_and_path)
        lines.append(f"{path_name}={root_name}/'tools/observe_abi_baseline.py'")
        if between_path_and_spec:
            lines.append(between_path_and_spec)
        lines.append(
            f"{spec_name}=importlib.util.spec_from_file_location('observe_abi_baseline',{path_name})"
        )
        if guard_message is not None:
            lines.extend(
                (
                    f"if {spec_name} is None or {spec_name}.loader is None:",
                    f" raise RuntimeError({guard_message!r})",
                )
            )
        if between_spec_and_module:
            lines.append(between_spec_and_module)
        lines.append(f"{module_name}=importlib.util.module_from_spec({spec_name})")
        if between_module_and_register:
            lines.append(between_module_and_register)
        call = (
            f"{module_name}.run_command((sys.executable,'-c',"
            "'import subprocess;subprocess.run([\"x\"])'))"
        )
        registration = f"sys.modules[{spec_name}.name]={module_name}"
        execution = f"{spec_name}.loader.exec_module({module_name})"
        if call_before_exec:
            lines.append(call)
        if exec_before_register:
            if execute:
                lines.append(execution)
            if register:
                lines.append(registration)
        else:
            if register:
                lines.append(registration)
            if execute:
                lines.append(execution)
        if not call_before_exec:
            lines.append(call)
        return "\n".join(lines) + "\n"

    def test_reviewed_loader_flow_is_ordered_and_name_independent(self) -> None:
        owner = "def run_command(argv):\n return argv\n"
        launches = self._reviewed_adapter_fixture_launches(
            owner,
            self._reviewed_loader_source(
                root_name="repository_base",
                path_name="adapter_file",
                spec_name="module_specification",
                module_name="loaded_adapter",
            ),
        )
        embedded = [item for item in launches if item["anchor"].get("origin_kind")]
        self.assertEqual(1, len(embedded))
        self.assertEqual(
            "reviewed-adapter-argv0", embedded[0]["anchor"]["carrier_shapes"][0]
        )

        for label, source in (
            (
                "call-before-exec",
                self._reviewed_loader_source(call_before_exec=True),
            ),
            (
                "exec-before-register",
                self._reviewed_loader_source(exec_before_register=True),
            ),
            ("missing-register", self._reviewed_loader_source(register=False)),
            ("missing-exec", self._reviewed_loader_source(execute=False)),
        ):
            with self.subTest(label=label):
                with self.assertRaisesRegex(
                    CHECKER.InventoryError,
                    "loader (?:chain|alias|variables|attributes)",
                ):
                    self._reviewed_adapter_fixture_launches(owner, source)

    def test_alias_timeline_applies_adapter_overlay_only_after_exec(self) -> None:
        owner = "def run_command(argv):\n return argv\n"
        after = self._reviewed_adapter_fixture_launches(
            owner, self._reviewed_loader_source()
        )
        self.assertEqual(
            1,
            sum(bool(item["anchor"].get("origin_kind")) for item in after),
        )
        with self.assertRaisesRegex(
            CHECKER.InventoryError,
            "loader (?:chain|alias|variables|attributes)",
        ):
            self._reviewed_adapter_fixture_launches(
                owner,
                self._reviewed_loader_source(call_before_exec=True),
            )

    def test_reviewed_loader_ignores_unrelated_nested_namespace_bindings(self) -> None:
        names = (
            "ROOT",
            "MODULE_PATH",
            "SPEC",
            "observer",
            "sys",
            "Path",
            "run_command",
        )
        owner = (
            "def run_command(argv):\n return argv\n"
            "def unrelated(run_command):\n"
            " run_command=None\n"
            " class Local:\n"
            "  run_command=None\n"
            " return [run_command for run_command in ()]\n"
            "class OwnerLocal:\n run_command=None\n"
            "OWNER_LOCAL=[run_command for run_command in ()]\n"
        )
        parameters = ",".join(names)
        local_assignments = "\n".join(f" {name}=None" for name in names)
        class_assignments = "\n".join(f" {name}=None" for name in names)
        comprehensions = "\n".join(
            f"UNRELATED_{index}=[{name} for {name} in ()]"
            for index, name in enumerate(names)
        )
        consumer = self._reviewed_loader_source() + (
            f"def unrelated({parameters}):\n"
            f"{local_assignments}\n"
            " return None\n"
            "class LocalNames:\n"
            f"{class_assignments}\n"
            "def declares_only():\n global ROOT,SPEC\n return None\n"
            f"{comprehensions}\n"
        )
        launches = self._reviewed_adapter_fixture_launches(owner, consumer)
        embedded = [item for item in launches if item["anchor"].get("origin_kind")]
        self.assertEqual(1, len(embedded))
        self.assertEqual(
            "reviewed-adapter-argv0", embedded[0]["anchor"]["carrier_shapes"][0]
        )
        self.assertRegex(
            embedded[0]["anchor"]["adapter_definition_digest"], r"^[0-9a-f]{64}$"
        )

    def test_reviewed_loader_rejects_module_and_global_namespace_writes(self) -> None:
        owner = "def run_command(argv):\n return argv\n"
        mutations = (
            "if condition:\n SPEC=object()\n",
            "for SPEC in ():\n pass\n",
            "with context() as SPEC:\n pass\n",
            "match value:\n case SPEC:\n  pass\n",
            "def mutate():\n global SPEC\n SPEC=object()\n",
            "def mutate():\n global SPEC\n del SPEC\n",
        )
        for mutation in mutations:
            with self.subTest(mutation=mutation):
                with self.assertRaisesRegex(CHECKER.InventoryError, "rebound"):
                    self._reviewed_adapter_fixture_launches(
                        owner, self._reviewed_loader_source() + mutation
                    )

    def test_reviewed_loader_variables_cannot_be_rebound_or_escape(self) -> None:
        owner = "def run_command(argv):\n return argv\n"
        mutations = (
            self._reviewed_loader_source(between_root_and_path="ROOT=object()"),
            self._reviewed_loader_source(between_path_and_spec="MODULE_PATH=object()"),
            self._reviewed_loader_source(between_spec_and_module="SPEC=object()"),
            self._reviewed_loader_source(
                between_module_and_register="observer=object()"
            ),
            self._reviewed_loader_source(between_spec_and_module="escaped=SPEC"),
            self._reviewed_loader_source(
                between_module_and_register="escaped=observer"
            ),
            self._reviewed_loader_source(
                between_spec_and_module="if condition:\n SPEC=object()"
            ),
        )
        for source in mutations:
            with self.subTest(source=source):
                with self.assertRaisesRegex(
                    CHECKER.InventoryError,
                    "rebound|alias escapes|guard is unsupported|chain contains",
                ):
                    self._reviewed_adapter_fixture_launches(owner, source)

    def test_reviewed_adapter_owner_must_be_unique_direct_and_unconditional(
        self,
    ) -> None:
        consumer = self._reviewed_loader_source()
        owners = (
            "if True:\n def run_command(argv):\n  return argv\n",
            "def outer():\n def run_command(argv):\n  return argv\n",
            "def run_command(argv):\n return argv\nrun_command=object()\n",
            "def run_command(argv):\n return argv\ndef run_command(argv):\n return argv\n",
        )
        for owner in owners:
            with self.subTest(owner=owner):
                with self.assertRaisesRegex(
                    CHECKER.InventoryError,
                    "direct unconditional run_command|adapter is rebound",
                ):
                    self._reviewed_adapter_fixture_launches(owner, consumer)

    def test_reviewed_loader_resolution_digest_binds_the_ordered_chain(self) -> None:
        owner = "def run_command(argv):\n return argv\n"

        def digest(message: str) -> str:
            launches = self._reviewed_adapter_fixture_launches(
                owner, self._reviewed_loader_source(guard_message=message)
            )
            values = {
                item["anchor"]["adapter_definition_digest"]
                for item in launches
                if item["anchor"].get("origin_kind")
            }
            self.assertEqual(1, len(values))
            return values.pop()

        self.assertNotEqual(digest("unable to load one"), digest("unable to load two"))

    def test_generic_or_fake_adapter_is_not_a_carrier(self) -> None:
        with tempfile.TemporaryDirectory(prefix="generic-adapter-") as directory:
            root = Path(directory)
            (root / "generic.py").write_text(
                "import sys\ndef run_command(argv): return argv\n"
                "run_command((sys.executable,'-c','import subprocess;subprocess.run([\\\"x\\\"])'))\n",
                encoding="utf-8",
            )
            launches = CHECKER._discover_python_launches(root)
        self.assertFalse(any(item["anchor"].get("origin_kind") for item in launches))

    def test_reviewed_observer_adapter_shadow_and_escape_fail_closed(self) -> None:
        for body in (
            "run_command=lambda argv: argv\nrun_command(('python','-cpass'))\n",
            "def sink(value): return value\nsink(run_command)\n",
        ):
            with self.subTest(body=body):
                with tempfile.TemporaryDirectory(prefix="adapter-shadow-") as directory:
                    root = Path(directory)
                    adapter = root / "tools/observe_abi_baseline.py"
                    adapter.parent.mkdir(parents=True)
                    adapter.write_text(
                        "def run_command(argv):\n return argv\n", encoding="utf-8"
                    )
                    (root / "consumer.py").write_text(
                        "from tools.observe_abi_baseline import run_command\n" + body,
                        encoding="utf-8",
                    )
                    with self.assertRaisesRegex(
                        CHECKER.InventoryError,
                        "reviewed adapter alias|process alias escapes",
                    ):
                        CHECKER._discover_python_launches(root)

    def test_python_option_stops_and_shell_carriers_are_explicit(self) -> None:
        launches = self._launches_for_source(
            "import subprocess,sys\ndef f():\n"
            "    subprocess.run((sys.executable,'-m','module','-c','ignored'))\n"
            "    subprocess.run((sys.executable,'script.py','-c','ignored'))\n"
            "    subprocess.run((sys.executable,'--','-c','ignored'))\n"
        )
        self.assertFalse(any(item["anchor"].get("origin_kind") for item in launches))
        shell_sources = (
            "import os\nos.system('python3 -c pass')\n",
            "import os\ndef f(command): os.system(command)\n",
            "import subprocess\nsubprocess.getoutput('python -I -c pass')\n",
        )
        for source in shell_sources:
            with self.subTest(source=source):
                with self.assertRaisesRegex(
                    CHECKER.InventoryError, "shell Python -c carriers"
                ):
                    self._launches_for_source(source)

    def test_attached_python_c_and_complete_option_boundaries(self) -> None:
        code = "import subprocess;subprocess.run(['embedded'])"
        launches = self._launches_for_source(
            "import subprocess,sys\nfrom pathlib import Path\n"
            "TOOLS_DIR=Path(__file__).resolve().parent\n"
            f"subprocess.run((sys.executable,{'-c' + code!r}))\n"
            f"subprocess.run(('PYTHONW3.14.EXE',{'-c' + code!r}))\n"
            "subprocess.run((sys.executable,'-c',''))\n"
            "subprocess.run((sys.executable,'script.py',dynamic_tail))\n"
            "subprocess.run((sys.executable,__file__,dynamic_tail))\n"
            "subprocess.run((sys.executable,str(Path(__file__).resolve()),dynamic_tail))\n"
            "subprocess.run((sys.executable,str(TOOLS_DIR/'worker.py'),dynamic_tail))\n"
        )
        self.assertEqual(
            2,
            sum(bool(item["anchor"].get("origin_kind")) for item in launches),
        )

        ambiguous = (
            "import subprocess,sys\ndef f(option): subprocess.run((sys.executable,option))\n",
            "import subprocess,sys\ndef f(options): subprocess.run((sys.executable,*options))\n",
            f"import subprocess,sys\ndef f(options): subprocess.run((sys.executable,{'-c' + code!r}),**options)\n",
        )
        for source in ambiguous:
            with self.subTest(source=source):
                with self.assertRaisesRegex(
                    CHECKER.InventoryError,
                    "option/marker is dynamic|expansion is ambiguous|keyword expansion is ambiguous",
                ):
                    self._launches_for_source(source)

    def test_cpython_314_cluster_dfa_and_terminal_boundaries(self) -> None:
        code = "import subprocess;subprocess.run(['embedded'])"
        source = "import subprocess,sys\n"
        for option in ("-Ec", "-Ic", "-qc", "-OOc"):
            source += f"subprocess.run((sys.executable,{option!r},{code!r}))\n"
        source += (
            f"subprocess.run((sys.executable,{'-V-c' + code!r}))\n"
            f"subprocess.run((sys.executable,{'-hc' + code!r}))\n"
            "subprocess.run((sys.executable,'-Wignorec','ignored'))\n"
            "subprocess.run((sys.executable,'-Xdevc','ignored'))\n"
            f"subprocess.run((sys.executable,'--check-hash-based-pycs','always','-c',{code!r}))\n"
            "subprocess.run((sys.executable,'script.py',*dynamic_tail))\n"
            "subprocess.run((sys.executable,'--',*dynamic_tail))\n"
            "subprocess.run((sys.executable,'-m','module',*dynamic_tail))\n"
        )
        launches = self._launches_for_source(source)
        self.assertEqual(
            5,
            sum(bool(item["anchor"].get("origin_kind")) for item in launches),
        )

        rejected = (
            "import subprocess,sys\nsubprocess.run((sys.executable,'-J','value'))\n",
            "import subprocess,sys\nsubprocess.run((sys.executable,'--check-hash-based-pycs','sometimes','-c','pass'))\n",
            "import subprocess,sys\ndef f(option): subprocess.run((sys.executable,option,'script.py'))\n",
            "import subprocess\nsubprocess.run(('env','python3','-Ec','pass'))\n",
            "import subprocess\nsubprocess.run(('bash','-c','python3 -Ec pass'))\n",
        )
        for rejected_source in rejected:
            with self.subTest(source=rejected_source):
                with self.assertRaises(CHECKER.InventoryError):
                    self._launches_for_source(rejected_source)

    def test_carrier_call_shape_rejects_expansion_before_identity(self) -> None:
        ordinary = self._launches_for_source(
            "import subprocess\ndef f(command): subprocess.run(command)\n"
        )
        self.assertEqual(1, len(ordinary))
        self.assertNotIn("carrier_semantics_digest", ordinary[0])

        rejected = (
            "import subprocess\ndef f(command,options): subprocess.run(command,**options)\n",
            "import subprocess\ndef f(parts): subprocess.run(*parts)\n",
            "import subprocess,sys\ndef f(options): subprocess.run(('placeholder','-c','pass'),executable=sys.executable,**options)\n",
        )
        for source in rejected:
            with self.subTest(source=source):
                with self.assertRaisesRegex(
                    CHECKER.InventoryError, "expansion is ambiguous"
                ):
                    self._launches_for_source(source)

    def test_carrier_facts_freeze_assignments_and_stable_closures(self) -> None:
        code = "import subprocess;subprocess.run(['embedded'])"
        source = (
            "import subprocess,sys\n"
            f"MODULE_CODE={code!r}\n"
            "def module_user():\n"
            " subprocess.run((sys.executable,'-c',MODULE_CODE))\n"
            "def outer():\n"
            f" closure_code={code!r}\n"
            " def inner(): subprocess.run((sys.executable,'-c',closure_code))\n"
            " return inner\n"
            "def frozen():\n"
            f" a={code!r}\n"
            " b=a\n"
            " a='pass'\n"
            " subprocess.run((sys.executable,'-c',b))\n"
        )
        launches = self._launches_for_source(source)
        self.assertEqual(
            3,
            sum(bool(item["anchor"].get("origin_kind")) for item in launches),
        )

        future = (
            "import subprocess,sys\ndef f():\n a=b\n b='pass'\n"
            " subprocess.run((sys.executable,'-c',a))\n"
        )
        with self.assertRaisesRegex(CHECKER.InventoryError, "static string"):
            self._launches_for_source(future)

    def test_script_stop_requires_physical_canonical_builtins_and_path(self) -> None:
        accepted = self._launches_for_source(
            "import pathlib,subprocess,sys\nfrom pathlib import Path as CanonicalPath\n"
            "ONE=CanonicalPath(__file__).resolve().parent/'worker.py'\n"
            "TWO=pathlib.Path(__file__).resolve().parents[0]/'worker.py'\n"
            "subprocess.run((sys.executable,__file__,dynamic_tail))\n"
            "subprocess.run((sys.executable,str(ONE),dynamic_tail))\n"
            "subprocess.run((sys.executable,str(TWO),dynamic_tail))\n"
        )
        self.assertEqual(3, len(accepted))
        self.assertFalse(any(item["anchor"].get("origin_kind") for item in accepted))

        rejected = (
            "import subprocess,sys\ndef f(__file__): subprocess.run((sys.executable,__file__,dynamic_tail))\n",
            "import subprocess,sys\ndef f(values):\n script='worker.py'\n for script in values: pass\n subprocess.run((sys.executable,script,dynamic_tail))\n",
            "import subprocess,sys\ndef f(str):\n from pathlib import Path\n subprocess.run((sys.executable,str(Path(__file__).resolve()),dynamic_tail))\n",
            "import subprocess,sys\nfrom pathlib import Path\nPath=object\nsubprocess.run((sys.executable,str(Path(__file__).resolve()),dynamic_tail))\n",
            "import subprocess,sys\ndef mutate():\n global __file__\n __file__='-cpass'\nsubprocess.run((sys.executable,__file__,dynamic_tail))\n",
            "import subprocess,sys\nfrom pathlib import Path\ndef mutate():\n global str\n str=lambda value:value\nsubprocess.run((sys.executable,str(Path(__file__).resolve()),dynamic_tail))\n",
            "import subprocess,sys\nfrom pathlib import Path\ndef mutate():\n global Path\n Path=lambda value:value\nsubprocess.run((sys.executable,str(Path(__file__).resolve()),dynamic_tail))\n",
            "import subprocess,sys\nfrom pathlib import Path\ndef mutate():\n global Path\n del Path\nsubprocess.run((sys.executable,str(Path(__file__).resolve()),dynamic_tail))\n",
            "import subprocess,sys\nfrom pathlib import Path\n[subprocess.run((sys.executable,str(Path(__file__).resolve()),tail)) for Path in values]\n",
            "import subprocess,sys\nfrom pathlib import Path\n[subprocess.run((sys.executable,str(Path(__file__).resolve()),tail)) for str in values]\n",
            "import subprocess,sys\n[subprocess.run((sys.executable,__file__,tail)) for __file__ in values]\n",
        )
        for source in rejected:
            with self.subTest(source=source):
                with self.assertRaises(CHECKER.InventoryError):
                    self._launches_for_source(source)

        carrier = self._launches_for_source(
            "import subprocess,sys\nsubprocess.run((sys.executable,'-c','pass'))\n"
        )
        self.assertEqual(1, len(carrier))
        self.assertRegex(carrier[0]["carrier_semantics_digest"], r"^[0-9a-f]{64}$")
        self.assertEqual(
            "python-carrier-semantics-v3",
            CHECKER.PYTHON_CARRIER_SEMANTICS_VERSION,
        )

    def test_carrier_binding_history_is_built_once_per_scope(self) -> None:
        tree = CHECKER.ast.parse(
            "import subprocess,sys\ndef many():\n"
            " code='pass'\n"
            + "".join(
                " subprocess.run((sys.executable,'-c',code))\n" for _ in range(300)
            )
        )
        parents: dict[Any, Any] = {}
        for node in CHECKER.ast.walk(tree):
            for child in CHECKER.ast.iter_child_nodes(node):
                parents[child] = node
        function = next(
            node
            for node in CHECKER.ast.walk(tree)
            if isinstance(node, CHECKER.ast.FunctionDef)
        )
        calls = [
            node
            for node in CHECKER.ast.walk(function)
            if isinstance(node, CHECKER.ast.Call)
            and isinstance(node.func, CHECKER.ast.Attribute)
            and node.func.attr == "run"
        ]
        original_walk = CHECKER.ast.walk
        original_freeze = CHECKER._freeze_carrier_expression
        scope_walks = 0
        freezes = 0

        def counted_walk(node):
            nonlocal scope_walks
            if node is function:
                scope_walks += 1
            return original_walk(node)

        def counted_freeze(*args, **kwargs):
            nonlocal freezes
            freezes += 1
            return original_freeze(*args, **kwargs)

        cache: dict[Any, Any] = {}
        CHECKER.ast.walk = counted_walk
        CHECKER._freeze_carrier_expression = counted_freeze
        try:
            for call in calls:
                facts = CHECKER._point_python_assignments(call, parents, cache)
                self.assertEqual(
                    "pass", CHECKER._literal_python_string(call.args[0].elts[2], facts)
                )
        finally:
            CHECKER.ast.walk = original_walk
            CHECKER._freeze_carrier_expression = original_freeze
        self.assertEqual(0, scope_walks)
        self.assertEqual(1, freezes)
        self.assertIn(("carrier-scope-index", id(tree)), cache)
        self.assertTrue(
            any(
                key[0] == "carrier-fact" and key[1] == id(function)
                for key in cache
                if isinstance(key, tuple)
            )
        )

    def test_carrier_parent_histories_and_path_imports_are_indexed_once(self) -> None:
        siblings = "".join(
            f"def sibling_{index}():\n"
            " subprocess.run((sys.executable,str(Path(__file__).resolve()),dynamic_tail))\n"
            for index in range(120)
        )
        deep_lines = ["def deep_0():"]
        for depth in range(1, 61):
            deep_lines.append(f"{' ' * depth}def deep_{depth}():")
        deep_lines.append(
            f"{' ' * 61}subprocess.run((sys.executable,str(Path(__file__).resolve()),dynamic_tail))"
        )
        source = (
            "import subprocess,sys\nfrom pathlib import Path\n"
            + siblings
            + "\n".join(deep_lines)
            + "\n"
        )
        tree = CHECKER.ast.parse(source)
        parents: dict[Any, Any] = {}
        nodes = list(CHECKER.ast.walk(tree))
        for node in nodes:
            for child in CHECKER.ast.iter_child_nodes(node):
                parents[child] = node
        scopes = [
            tree,
            *(node for node in nodes if isinstance(node, CHECKER.ast.FunctionDef)),
        ]
        calls = [
            node
            for node in nodes
            if isinstance(node, CHECKER.ast.Call)
            and isinstance(node.func, CHECKER.ast.Attribute)
            and node.func.attr == "run"
        ]
        original_walk = CHECKER.ast.walk
        scope_ids = {id(scope) for scope in scopes}
        scope_walk_calls = 0
        walk_yields = 0

        def counted_walk(node):
            nonlocal scope_walk_calls, walk_yields
            if id(node) in scope_ids:
                scope_walk_calls += 1
            iterator = original_walk(node)

            def tracked():
                nonlocal walk_yields
                for candidate in iterator:
                    walk_yields += 1
                    yield candidate

            return tracked()

        cache: dict[Any, Any] = {}
        CHECKER.ast.walk = counted_walk
        try:
            for call in calls:
                self.assertIsNone(
                    CHECKER._extract_python_carrier(
                        call,
                        "subprocess.run",
                        {"subprocess": "subprocess", "sys": "sys"},
                        parents,
                        location="fixture.py:carrier",
                        assignment_cache=cache,
                    )
                )
        finally:
            CHECKER.ast.walk = original_walk
        self.assertEqual(0, scope_walk_calls)
        self.assertLessEqual(walk_yields, 80 * len(calls))
        self.assertEqual(
            1,
            sum(
                key[0] == "carrier-scope-index"
                for key in cache
                if isinstance(key, tuple)
            ),
        )
        self.assertEqual(
            len(scopes),
            sum(key[0] == "carrier-scope" for key in cache if isinstance(key, tuple)),
        )
        self.assertFalse(
            any(
                key[0] in {"carrier-history", "carrier-stable"}
                for key in cache
                if isinstance(key, tuple)
            )
        )
        self.assertTrue(
            any(key[0] == "carrier-fact" for key in cache if isinstance(key, tuple))
        )

        class NoEventIteration(tuple):
            def __iter__(self):
                raise AssertionError("call-stage code iterated the full event ledger")

        for key, value in list(cache.items()):
            if isinstance(key, tuple) and key[0] == "carrier-scope":
                cache[key] = value._replace(events=NoEventIteration(value.events))
        for call in calls:
            self.assertIsNone(
                CHECKER._extract_python_carrier(
                    call,
                    "subprocess.run",
                    {"subprocess": "subprocess", "sys": "sys"},
                    parents,
                    location="fixture.py:carrier",
                    assignment_cache=cache,
                )
            )

    def test_carrier_argv0_and_environment_expansions_fail_closed(self) -> None:
        fixtures = (
            (
                "import asyncio\nasync def f(parts): await asyncio.create_subprocess_exec(*parts)\n",
                "asyncio.create_subprocess_exec",
            ),
            (
                "import asyncio\nasync def f(factory,parts):\n"
                " loop=asyncio.get_event_loop()\n"
                " await loop.subprocess_exec(factory,*parts)\n",
                "asyncio.loop.subprocess_exec",
            ),
            ("import os\ndef f(parts): os.execl('python3',*parts)\n", "os.execl"),
            ("import os\ndef f(parts): os.execle('python3',*parts)\n", "os.execle"),
            (
                "import os\ndef f(parts): os.spawnle(0,'python3',*parts)\n",
                "os.spawnle",
            ),
            (
                "import subprocess\ndef f(parts): subprocess.run([*parts])\n",
                "subprocess.run",
            ),
            (
                "import subprocess,sys\ndef f(parts): subprocess.run([*parts],executable=sys.executable)\n",
                "subprocess.run",
            ),
        )
        for source, dotted in fixtures:
            with self.subTest(dotted=dotted, source=source):
                tree = CHECKER.ast.parse(source)
                parents: dict[Any, Any] = {}
                nodes = list(CHECKER.ast.walk(tree))
                for node in nodes:
                    for child in CHECKER.ast.iter_child_nodes(node):
                        parents[child] = node
                process_call = next(
                    node
                    for node in nodes
                    if isinstance(node, CHECKER.ast.Call)
                    and (
                        isinstance(node.func, CHECKER.ast.Attribute)
                        and node.func.attr
                        in {
                            "create_subprocess_exec",
                            "subprocess_exec",
                            "execl",
                            "execle",
                            "spawnle",
                            "run",
                        }
                    )
                )
                with self.assertRaisesRegex(CHECKER.InventoryError, "ambiguous"):
                    CHECKER._extract_python_carrier(
                        process_call,
                        dotted,
                        {"sys": "sys"},
                        parents,
                        location="fixture.py:carrier",
                    )
                with self.assertRaisesRegex(CHECKER.InventoryError, "ambiguous"):
                    self._launches_for_source(source)

    def test_dynamic_executable_uses_full_cpython_cli_scan(self) -> None:
        for option in ("-cpass", "-Ecpass", "-OOcpass"):
            source = (
                "import subprocess\n"
                f"def f(executable): subprocess.run([executable,{option!r}])\n"
            )
            with self.subTest(option=option):
                tree = CHECKER.ast.parse(source)
                parents: dict[Any, Any] = {}
                nodes = list(CHECKER.ast.walk(tree))
                for node in nodes:
                    for child in CHECKER.ast.iter_child_nodes(node):
                        parents[child] = node
                process_call = next(
                    node
                    for node in nodes
                    if isinstance(node, CHECKER.ast.Call)
                    and isinstance(node.func, CHECKER.ast.Attribute)
                    and node.func.attr == "run"
                )
                with self.assertRaisesRegex(
                    CHECKER.InventoryError, "executable is dynamic"
                ):
                    CHECKER._extract_python_carrier(
                        process_call,
                        "subprocess.run",
                        {},
                        parents,
                        location="fixture.py:carrier",
                    )
                with self.assertRaisesRegex(
                    CHECKER.InventoryError, "executable is dynamic"
                ):
                    self._launches_for_source(source)

    def test_only_plain_subprocess_run_keeps_the_dynamic_command_exception(
        self,
    ) -> None:
        fixtures = (
            (
                "import subprocess\ndef f(command,executable): subprocess.run(command,executable=executable)\n",
                "subprocess.run",
                "run",
            ),
            (
                "import subprocess\ndef f(command): subprocess.Popen(command)\n",
                "subprocess.Popen",
                "Popen",
            ),
            (
                "import subprocess\ndef f(command): subprocess.call(command)\n",
                "subprocess.call",
                "call",
            ),
            (
                "import subprocess\ndef f(command): subprocess.check_call(command)\n",
                "subprocess.check_call",
                "check_call",
            ),
            (
                "import subprocess\ndef f(command): subprocess.check_output(command)\n",
                "subprocess.check_output",
                "check_output",
            ),
            (
                "import os\ndef f(path,argv): os.execv(path,argv)\n",
                "os.execv",
                "execv",
            ),
            (
                "import os\ndef f(mode,path,argv): os.spawnv(mode,path,argv)\n",
                "os.spawnv",
                "spawnv",
            ),
            (
                "import pty\ndef f(argv): pty.spawn(argv)\n",
                "pty.spawn",
                "spawn",
            ),
        )
        for source, dotted, attribute in fixtures:
            with self.subTest(dotted=dotted):
                tree = CHECKER.ast.parse(source)
                parents: dict[Any, Any] = {}
                nodes = list(CHECKER.ast.walk(tree))
                for node in nodes:
                    for child in CHECKER.ast.iter_child_nodes(node):
                        parents[child] = node
                process_call = next(
                    node
                    for node in nodes
                    if isinstance(node, CHECKER.ast.Call)
                    and isinstance(node.func, CHECKER.ast.Attribute)
                    and node.func.attr == attribute
                )
                with self.assertRaisesRegex(CHECKER.InventoryError, "proof is dynamic"):
                    CHECKER._extract_python_carrier(
                        process_call,
                        dotted,
                        {},
                        parents,
                        location="fixture.py:carrier",
                    )
                with self.assertRaisesRegex(CHECKER.InventoryError, "proof is dynamic"):
                    self._launches_for_source(source)

        accepted = self._launches_for_source(
            "import subprocess\ndef f(command,executable,tail):\n"
            " subprocess.run(command)\n"
            " subprocess.run(['placeholder','script.py',*tail],executable=executable)\n"
        )
        self.assertEqual(2, len(accepted))
        self.assertFalse(any(item["anchor"].get("origin_kind") for item in accepted))

    def test_shell_runtime_interpreter_expansions_fail_closed(self) -> None:
        fixtures = (
            (
                "import subprocess\nsubprocess.run('$PYTHON -c pass',shell=True)\n",
                "subprocess.run",
            ),
            ("import os\nos.system('%PYTHON% -c pass')\n", "os.system"),
            (
                "import os\nos.system(\"pwsh -Command '$env:PYTHON -c pass'\")\n",
                "os.system",
            ),
            ("import os\nos.system('py* -c pass')\n", "os.system"),
            (
                "import os\nos.system('alias p=python3; p -c pass')\n",
                "os.system",
            ),
            (
                "import os\nos.system(\"pwsh -Command & ('py'+'thon3') -c pass\")\n",
                "os.system",
            ),
        )
        for source, dotted in fixtures:
            with self.subTest(source=source):
                tree = CHECKER.ast.parse(source)
                parents: dict[Any, Any] = {}
                nodes = list(CHECKER.ast.walk(tree))
                for node in nodes:
                    for child in CHECKER.ast.iter_child_nodes(node):
                        parents[child] = node
                process_call = next(
                    node
                    for node in nodes
                    if isinstance(node, CHECKER.ast.Call)
                    and isinstance(node.func, CHECKER.ast.Attribute)
                )
                with self.assertRaisesRegex(
                    CHECKER.InventoryError, "shell Python -c carriers"
                ):
                    CHECKER._extract_python_carrier(
                        process_call,
                        dotted,
                        {},
                        parents,
                        location="fixture.py:carrier",
                    )
                with self.assertRaisesRegex(
                    CHECKER.InventoryError, "shell Python -c carriers"
                ):
                    self._launches_for_source(source)

    def test_shell_and_indirect_attached_python_c_are_fail_closed(self) -> None:
        sources = (
            "import os\nos.system('PyThOnW3.12.ExE -cpass')\n",
            "import subprocess\nsubprocess.run(('env','pythonw.exe','-cpass'))\n",
            "import subprocess\nsubprocess.run(('bash','-cpython3 -cpass'))\n",
            "import subprocess\nsubprocess.run(('cmd.exe','/cPython.exe -cpass'))\n",
            "import subprocess\nsubprocess.run(('pwsh','-cPythonW3.exe -cpass'))\n",
            "import subprocess\ndef f(parts): subprocess.run(('xcrun',*parts))\n",
        )
        for source in sources:
            with self.subTest(source=source):
                with self.assertRaisesRegex(
                    CHECKER.InventoryError,
                    "shell Python -c carriers|indirect Python -c carrier|dynamic or ambiguous",
                ):
                    self._launches_for_source(source)

    def test_env_wrapper_consumes_option_values_before_python_identity(self) -> None:
        carriers = (
            "('env','-u','python3','python3','-c','pass')",
            "('env','--unset','python3','python3','-cpass')",
            "('env','--unset=python3','NAME=value','--','python3','-Ec','pass')",
            "('env','-P','/bin','python3','-c','pass')",
        )
        for argv in carriers:
            with (
                self.subTest(argv=argv),
                self.assertRaisesRegex(
                    CHECKER.InventoryError, "indirect Python -c carrier"
                ),
            ):
                self._launches_for_source(
                    f"import subprocess\nsubprocess.run({argv})\n"
                )

        non_carriers = (
            "('env','-u','python3','tool','-c','pass')",
            "('xcrun','--find','python3')",
            "('conda','install','python3','-c','pass')",
            "('uv','python','find','python3','-c','pass')",
        )
        for argv in non_carriers:
            with self.subTest(argv=argv):
                launches = self._launches_for_source(
                    f"import subprocess\nsubprocess.run({argv})\n"
                )
                self.assertEqual(1, len(launches))
                self.assertNotIn("carrier_semantics_digest", launches[0])

    def test_static_argv_concatenation_discovers_nested_carriers_and_digests(
        self,
    ) -> None:
        inner = "import subprocess; subprocess.run(['embedded'])"
        source = (
            "import subprocess\n"
            "PREFIX=['python3']\n"
            f"SUFFIX=['-c',{inner!r}]\n"
            "def launch():\n"
            " return subprocess.run(PREFIX + SUFFIX)\n"
        )
        launches = self._launches_for_source(source)
        outer = next(item for item in launches if not item["anchor"].get("origin_kind"))
        embedded = next(item for item in launches if item["anchor"].get("origin_kind"))
        self.assertRegex(outer["call_semantics_digest"], r"^[0-9a-f]{64}$")
        self.assertRegex(outer["carrier_semantics_digest"], r"^[0-9a-f]{64}$")
        self.assertEqual(
            [outer["carrier_semantics_digest"]],
            embedded["anchor"]["carrier_semantics_digests"],
        )
        self.assertEqual(["argv-arg0"], embedded["anchor"]["carrier_shapes"])

        literal = self._launches_for_source(
            f"import subprocess\nsubprocess.run(['python3'] + ['-c',{inner!r}])\n"
        )
        self.assertEqual(
            1, sum(bool(item["anchor"].get("origin_kind")) for item in literal)
        )
        tuple_literal = self._launches_for_source(
            f"import subprocess\nsubprocess.run(('python3',) + ('-c',{inner!r}))\n"
        )
        self.assertEqual(
            1,
            sum(bool(item["anchor"].get("origin_kind")) for item in tuple_literal),
        )

        changed = self._launches_for_source(source.replace("embedded", "mutated"))
        changed_outer = next(
            item for item in changed if not item["anchor"].get("origin_kind")
        )
        self.assertEqual(
            outer["call_semantics_digest"], changed_outer["call_semantics_digest"]
        )
        self.assertNotEqual(
            outer["carrier_semantics_digest"],
            changed_outer["carrier_semantics_digest"],
        )

    def test_static_argv_concatenation_mutants_fail_closed(self) -> None:
        sources = (
            "import subprocess\ndef f(tail): subprocess.run(['python3'] + tail)\n",
            "import subprocess\ndef f(head): subprocess.run(head + ['python3','-c','pass'])\n",
            "import subprocess\nsubprocess.run(['python3'] + ('-c','pass'))\n",
            "import subprocess\ndef f(tail): subprocess.run(['python3'] + [*tail])\n",
        )
        for source in sources:
            with (
                self.subTest(source=source),
                self.assertRaisesRegex(
                    CHECKER.InventoryError,
                    "dynamic or ambiguous|mixes list and tuple|expansion is ambiguous",
                ),
            ):
                self._launches_for_source(source)

        ordinary_dynamic_argv = self._launches_for_source(
            "import subprocess\ndef launch(tail): subprocess.run(['tool', *tail])\n"
        )
        self.assertEqual(1, len(ordinary_dynamic_argv))
        self.assertNotIn("carrier_semantics_digest", ordinary_dynamic_argv[0])

        with (
            mock.patch.object(CHECKER, "STATIC_PYTHON_ARGV_MAX_ITEMS", 1),
            self.assertRaisesRegex(CHECKER.InventoryError, "item limit"),
        ):
            self._launches_for_source(
                "import subprocess\nsubprocess.run(['python3'] + ['-c','pass'])\n"
            )
        with (
            mock.patch.object(CHECKER, "STATIC_PYTHON_ARGV_MAX_DEPTH", 1),
            self.assertRaisesRegex(CHECKER.InventoryError, "depth limit"),
        ):
            self._launches_for_source(
                "import subprocess\nsubprocess.run((['python3'] + ['-c']) + ['pass'])\n"
            )

    def test_nice_wrapper_transparently_discovers_nested_python_carriers(self) -> None:
        inner = "import subprocess; subprocess.run(['embedded'])"
        argv_forms = (
            f"('nice','python3','-c',{inner!r})",
            f"('nice','-n','5','python3','-c',{inner!r})",
            f"('nice','--adjustment=-2','--','python3','-c',{inner!r})",
            f"('nice','--','nice','-n','1','python3','-c',{inner!r})",
        )
        carrier_digests: list[str] = []
        for argv in argv_forms:
            with self.subTest(argv=argv):
                launches = self._launches_for_source(
                    f"import subprocess\nsubprocess.run({argv})\n"
                )
                outer = next(
                    item for item in launches if not item["anchor"].get("origin_kind")
                )
                embedded = next(
                    item for item in launches if item["anchor"].get("origin_kind")
                )
                carrier_digests.append(outer["carrier_semantics_digest"])
                self.assertEqual(
                    ["transparent-nice+argv-arg0"],
                    embedded["anchor"]["carrier_shapes"],
                )
        self.assertEqual(len(argv_forms), len(set(carrier_digests)))

    def test_nice_and_unsupported_static_wrapper_tails_fail_closed(self) -> None:
        sources = (
            "import subprocess\nsubprocess.run(('nice',))\n",
            "import subprocess\nsubprocess.run(('nice','--'))\n",
            "import subprocess\nsubprocess.run(('nice','-n'))\n",
            "import subprocess\nsubprocess.run(('nice','-n','2','--'))\n",
            "import subprocess\nsubprocess.run(('nice','-n','invalid','python3','-c','pass'))\n",
            "import subprocess\nsubprocess.run(('nice','--adjustment=','python3','-c','pass'))\n",
            "import subprocess\nsubprocess.run(('nice','-x','python3','-c','pass'))\n",
            "import subprocess\ndef f(value): subprocess.run(('nice','-n',value,'python3','-c','pass'))\n",
            "import subprocess\ndef f(command): subprocess.run(('nice',command,'-c','pass'))\n",
            "import subprocess\nsubprocess.run(('nice','tool','python3','-c','pass'))\n",
            "import subprocess\nsubprocess.run(('unknown-tool','python3','-c','pass'))\n",
        )
        for source in sources:
            with (
                self.subTest(source=source),
                self.assertRaisesRegex(
                    CHECKER.InventoryError,
                    "nice|unsupported wrapper/tool tail",
                ),
            ):
                self._launches_for_source(source)

        reviewed_script_stops = self._launches_for_source(
            "import subprocess\n"
            "subprocess.run(('sh','script.sh','python3','-c','pass'))\n"
            "subprocess.run(('nice','sh','script.sh','python3','-c','pass'))\n"
        )
        self.assertEqual(2, len(reviewed_script_stops))
        self.assertFalse(
            any("carrier_semantics_digest" in item for item in reviewed_script_stops)
        )

    def test_unsupported_tail_detects_proven_and_windows_python_launchers(
        self,
    ) -> None:
        sources = (
            "import subprocess,sys\nsubprocess.run(('tool',sys.executable,'-c','pass'))\n",
            "import subprocess,sys\nPY=sys.executable\nALIAS=PY\nsubprocess.run(('tool',ALIAS,'-c','pass'))\n",
            "import subprocess,sys as runtime\nPY=runtime.executable\nsubprocess.run(('tool',PY,'-c','pass'))\n",
            "import subprocess\ndef launch(command): subprocess.run((command,'python3','-c','pass'))\n",
            "import subprocess,sys\ndef launch(token): subprocess.run(('tool',token,sys.executable,'-c','pass'))\n",
            "import subprocess\nsubprocess.run(('tool','py','-cpass'))\n",
            "import subprocess\nsubprocess.run(('tool','pyw.exe','-cpass'))\n",
            "import subprocess\nsubprocess.run(('tool','pymanager','exec','-cpass'))\n",
            "import subprocess\nsubprocess.run(('tool','pywmanager.exe','exec','-cpass'))\n",
            "import subprocess\nWIN='pymanager'\nALIAS=WIN\nsubprocess.run(('tool',ALIAS,'exec','-cpass'))\n",
            "import subprocess,sys\nPY=sys.executable\nsubprocess.run(('nice','tool',PY,'-c','pass'))\n",
            "import subprocess\nsubprocess.run(('nice','nice','tool','py','-cpass'))\n",
        )
        for source in sources:
            with (
                self.subTest(source=source),
                self.assertRaisesRegex(
                    CHECKER.InventoryError,
                    "proven Python launcher.*unsupported wrapper/tool tail",
                ),
            ):
                self._launches_for_source(source)

        ordinary = self._launches_for_source(
            "import subprocess\n"
            "subprocess.run(('tool','other','-c','pass'))\n"
            "def launch(dynamic): subprocess.run((dynamic,'other','-c','pass'))\n"
        )
        self.assertEqual(2, len(ordinary))
        self.assertFalse(any("carrier_semantics_digest" in item for item in ordinary))

    def test_shell_tail_detects_windows_and_manager_python_launchers(self) -> None:
        shell_sources = (
            "import os\nos.system('tool py -cpass')\n",
            "import os\nos.system('tool pyw.exe -cpass')\n",
            "import os\nos.system('tool pymanager exec -cpass')\n",
            "import subprocess\nsubprocess.run(('sh','-c','tool pywmanager exec -cpass'))\n",
        )
        for source in shell_sources:
            with (
                self.subTest(source=source),
                self.assertRaisesRegex(CHECKER.InventoryError, "Python -c carrier"),
            ):
                self._launches_for_source(source)

        script_stop = self._launches_for_source(
            "import subprocess\n"
            "subprocess.run(('sh','script.sh','pymanager','exec','-cpass'))\n"
        )
        self.assertEqual(1, len(script_stop))
        self.assertNotIn("carrier_semantics_digest", script_stop[0])

    def test_nice_accepts_proven_interpreter_alias_as_transparent_carrier(self) -> None:
        inner = "import subprocess; subprocess.run(['embedded'])"
        launches = self._launches_for_source(
            "import subprocess,sys\n"
            "PY=sys.executable\n"
            "ALIAS=PY\n"
            f"subprocess.run(('nice','--',ALIAS,'-c',{inner!r}))\n"
        )
        embedded = [item for item in launches if item["anchor"].get("origin_kind")]
        self.assertEqual(1, len(embedded))
        self.assertEqual(
            ["transparent-nice+argv-arg0"],
            embedded[0]["anchor"]["carrier_shapes"],
        )

    def test_carrier_fact_freeze_rejects_exponential_alias_expansion_early(
        self,
    ) -> None:
        lines = ["import subprocess", "A0=('python3',)"]
        lines.extend(f"A{index}=A{index - 1}+A{index - 1}" for index in range(1, 20))
        lines.append("subprocess.run(A19)")
        source = "\n".join(lines) + "\n"
        real_deepcopy = CHECKER.copy.deepcopy
        largest_copy = 0

        def guarded_deepcopy(value, memo=None):
            nonlocal largest_copy
            if isinstance(value, CHECKER.ast.AST):
                largest_copy = max(
                    largest_copy, sum(1 for _ in CHECKER.ast.walk(value))
                )
            return real_deepcopy(value, memo)

        with (
            mock.patch.object(CHECKER, "CARRIER_FREEZE_MAX_NODES", 128),
            mock.patch.object(CHECKER.copy, "deepcopy", side_effect=guarded_deepcopy),
            self.assertRaisesRegex(
                CHECKER.InventoryError,
                "frozen carrier expression exceeds (?:node|item) limit",
            ),
        ):
            self._launches_for_source(source)
        self.assertLessEqual(largest_copy, 128)
        with self.assertRaisesRegex(
            CHECKER.InventoryError,
            "frozen carrier expression exceeds (?:node|item) limit",
        ):
            self._launches_for_source(source)

    def test_irrelevant_large_assignment_skips_carrier_fact_freezing(self) -> None:
        large_dict = "{" + ",".join(f"{index}:{index}" for index in range(4_100)) + "}"
        ordinary_source = (
            f"def inspect():\n observation={large_dict}\n return ordinary()\n"
        )
        with mock.patch.object(
            CHECKER,
            "_point_python_assignments",
            wraps=CHECKER._point_python_assignments,
        ) as assignments:
            self.assertEqual([], self._launches_for_source(ordinary_source))
        assignments.assert_not_called()

        process_source = (
            "import subprocess\n"
            "def inspect():\n"
            f" observation={large_dict}\n"
            " return subprocess.run(['tool'])\n"
        )
        launches = self._launches_for_source(process_source)
        self.assertEqual(1, len(launches))
        self.assertNotIn("carrier_semantics_digest", launches[0])

    def test_many_unrelated_assignments_freeze_no_carrier_facts(self) -> None:
        source = (
            "import subprocess\n"
            + "".join(f"UNRELATED_{index}=({index},)\n" for index in range(300))
            + "subprocess.run(['tool'])\n"
        )
        with mock.patch.object(
            CHECKER,
            "_freeze_carrier_expression",
            wraps=CHECKER._freeze_carrier_expression,
        ) as freeze:
            launches = self._launches_for_source(source)
        self.assertEqual(1, len(launches))
        freeze.assert_not_called()

    def test_superseded_large_assignment_is_not_frozen(self) -> None:
        large_dict = "{" + ",".join(f"{index}:{index}" for index in range(4_100)) + "}"
        source = (
            "import subprocess\n"
            f"ARGV={large_dict}\n"
            "ARGV=['tool']\n"
            "subprocess.run(ARGV)\n"
        )
        launches = self._launches_for_source(source)
        self.assertEqual(1, len(launches))
        self.assertNotIn("carrier_semantics_digest", launches[0])

    def test_referenced_large_assignment_fails_before_deepcopy(self) -> None:
        large_dict = "{" + ",".join(f"{index}:{index}" for index in range(4_100)) + "}"
        source = (
            "import subprocess\n"
            "def inspect():\n"
            f" observation={large_dict}\n"
            " argv=observation\n"
            " return subprocess.run(argv)\n"
        )
        tree = CHECKER.ast.parse(source)
        observation = next(
            node
            for node in CHECKER.ast.walk(tree)
            if isinstance(node, CHECKER.ast.Assign)
            and isinstance(node.targets[0], CHECKER.ast.Name)
            and node.targets[0].id == "observation"
        )
        self.assertGreater(
            sum(1 for _ in CHECKER.ast.walk(observation.value)),
            CHECKER.CARRIER_FREEZE_MAX_NODES,
        )

        real_deepcopy = CHECKER.copy.deepcopy
        largest_copy = 0

        def guarded_deepcopy(value, memo=None):
            nonlocal largest_copy
            if isinstance(value, CHECKER.ast.AST):
                largest_copy = max(
                    largest_copy, sum(1 for _ in CHECKER.ast.walk(value))
                )
            return real_deepcopy(value, memo)

        with (
            mock.patch.object(CHECKER.copy, "deepcopy", side_effect=guarded_deepcopy),
            self.assertRaisesRegex(
                CHECKER.InventoryError,
                "frozen carrier expression exceeds (?:node|item) limit",
            ),
        ):
            self._launches_for_source(source)
        self.assertLessEqual(largest_copy, CHECKER.CARRIER_FREEZE_MAX_NODES)

        env_source = (
            f"import subprocess\nHUGE={large_dict}\nsubprocess.run(['tool'],env=HUGE)\n"
        )
        with self.assertRaisesRegex(
            CHECKER.InventoryError,
            "frozen carrier expression exceeds (?:node|item) limit",
        ):
            self._launches_for_source(env_source)

    def test_carrier_fact_freeze_budgets_and_cycles_fail_closed(self) -> None:
        cases = (
            (
                "CARRIER_FREEZE_MAX_ITEMS",
                1,
                "import subprocess\nCMD=['python3','-c']\nsubprocess.run(CMD)\n",
                "item limit",
            ),
            (
                "CARRIER_FREEZE_MAX_BYTES",
                4,
                "import subprocess\nCMD=['python3']\nsubprocess.run(CMD)\n",
                "byte limit",
            ),
            (
                "CARRIER_FREEZE_MAX_DEPTH",
                1,
                "import subprocess\nCMD=[['python3']]\nsubprocess.run(CMD)\n",
                "depth limit",
            ),
        )
        for constant, limit, source, error in cases:
            with (
                self.subTest(constant=constant),
                mock.patch.object(CHECKER, constant, limit),
                self.assertRaisesRegex(CHECKER.InventoryError, error),
            ):
                self._launches_for_source(source)

        cyclic = CHECKER.ast.List(elts=[], ctx=CHECKER.ast.Load())
        cyclic.elts.append(cyclic)
        with self.assertRaisesRegex(
            CHECKER.InventoryError, "dependency graph is cyclic"
        ):
            CHECKER._carrier_loaded_names_bounded(cyclic)
        with self.assertRaisesRegex(CHECKER.InventoryError, "is cyclic"):
            CHECKER._freeze_carrier_expression(cyclic, {}, frozenset(), set())

        at_limit: CHECKER.ast.expr = CHECKER.ast.Constant(value="python3")
        for _ in range(CHECKER.STATIC_PYTHON_ARGV_MAX_DEPTH):
            at_limit = CHECKER.ast.List(elts=[at_limit], ctx=CHECKER.ast.Load())
        CHECKER._freeze_carrier_expression(at_limit, {}, frozenset(), set())
        beyond_limit = CHECKER.ast.List(elts=[at_limit], ctx=CHECKER.ast.Load())
        with self.assertRaisesRegex(CHECKER.InventoryError, "depth limit"):
            CHECKER._freeze_carrier_expression(beyond_limit, {}, frozenset(), set())

    def test_carrier_fact_freeze_keeps_accepted_digests_deterministic(self) -> None:
        inner = "import subprocess; subprocess.run(['embedded'])"
        source = (
            "import subprocess\n"
            "HEAD=['python3']\n"
            f"TAIL=['-c',{inner!r}]\n"
            "ARGV=HEAD+TAIL\n"
            "ALIAS=ARGV\n"
            "subprocess.run(ALIAS)\n"
        )
        first = self._launches_for_source(source)
        second = self._launches_for_source(source)
        self.assertEqual(first, second)
        self.assertEqual(CHECKER.STATIC_PYTHON_ARGV_MAX_ITEMS, 4_096)
        self.assertEqual(CHECKER.STATIC_PYTHON_ARGV_MAX_DEPTH, 32)
        self.assertEqual(
            CHECKER.CARRIER_FREEZE_MAX_ITEMS,
            CHECKER.STATIC_PYTHON_ARGV_MAX_ITEMS,
        )
        self.assertEqual(
            CHECKER.CARRIER_FREEZE_MAX_DEPTH,
            CHECKER.STATIC_PYTHON_ARGV_MAX_DEPTH,
        )

    def test_ten_thousand_alias_chain_is_iterative_and_digest_stable(self) -> None:
        inner = "import subprocess; subprocess.run(['embedded'])"
        lines = [
            "import subprocess",
            f"A0=['python3','-c',{inner!r}]",
        ]
        lines.extend(f"A{index}=A{index - 1}" for index in range(1, 10_001))
        lines.append("subprocess.run(A10000)")
        source = "\n".join(lines) + "\n"
        tree = CHECKER.ast.parse(source)
        parents: dict[Any, Any] = {}
        nodes = list(CHECKER.ast.walk(tree))
        for node in nodes:
            for child in CHECKER.ast.iter_child_nodes(node):
                parents[child] = node
        call = next(
            node
            for node in nodes
            if isinstance(node, CHECKER.ast.Call)
            and isinstance(node.func, CHECKER.ast.Attribute)
            and node.func.attr == "run"
        )
        carriers = [
            CHECKER._extract_python_carrier(
                call,
                "subprocess.run",
                {"subprocess": "subprocess"},
                parents,
                location="fixture.py:carrier",
                assignment_cache={},
                physical_file=False,
            )
            for _ in range(2)
        ]
        first, second = carriers
        self.assertIsNotNone(first)
        self.assertIsNotNone(second)
        assert first is not None
        assert second is not None
        self.assertEqual(inner, first.code)
        self.assertEqual(
            first.semantics_digest,
            second.semantics_digest,
        )

    def test_type_alias_ast_node_is_optional_and_source_frozen(self) -> None:
        checker_source = CHECKER_PATH.read_text(encoding="utf-8")
        checker_tree = CHECKER.ast.parse(checker_source)
        direct_references = [
            node
            for node in CHECKER.ast.walk(checker_tree)
            if isinstance(node, CHECKER.ast.Attribute)
            and isinstance(node.value, CHECKER.ast.Name)
            and node.value.id == "ast"
            and node.attr == "TypeAlias"
        ]
        self.assertEqual([], direct_references)
        sentinels = [
            node
            for node in CHECKER.ast.walk(checker_tree)
            if isinstance(node, CHECKER.ast.Call)
            and isinstance(node.func, CHECKER.ast.Name)
            and node.func.id == "getattr"
            and len(node.args) >= 2
            and isinstance(node.args[0], CHECKER.ast.Name)
            and node.args[0].id == "ast"
            and isinstance(node.args[1], CHECKER.ast.Constant)
            and node.args[1].value == "TypeAlias"
        ]
        self.assertEqual(1, len(sentinels))

        missing = object()
        original = getattr(CHECKER.ast, "TypeAlias", missing)
        if original is not missing:
            delattr(CHECKER.ast, "TypeAlias")
        try:
            specification = importlib.util.spec_from_file_location(
                "check_build_inventory_without_type_alias", CHECKER_PATH
            )
            assert specification and specification.loader
            compatible = importlib.util.module_from_spec(specification)
            specification.loader.exec_module(compatible)
        finally:
            if original is not missing:
                setattr(CHECKER.ast, "TypeAlias", original)
        self.assertEqual((), compatible.AST_TYPE_ALIAS_TYPES)

        source = "import subprocess\nsubprocess.run(('python3','-c','pass'))\n"
        with tempfile.TemporaryDirectory(prefix="type-alias-compat-") as directory:
            root = Path(directory)
            (root / "fixture.py").write_text(source, encoding="utf-8")
            current = CHECKER._discover_python_launches(root)
            compatibility = compatible._discover_python_launches(root)
        self.assertEqual(current, compatibility)

    def test_python_ast_canonicalizer_is_version_neutral(self) -> None:
        self.assertEqual("python-call-semantics-v2", CHECKER.PYTHON_SEMANTICS_VERSION)
        self.assertEqual(
            "python-carrier-semantics-v3",
            CHECKER.PYTHON_CARRIER_SEMANTICS_VERSION,
        )

        checker_tree = CHECKER.ast.parse(CHECKER_PATH.read_text(encoding="utf-8"))
        canonicalizers = [
            node
            for node in checker_tree.body
            if isinstance(node, CHECKER.ast.FunctionDef)
            and node.name
            in {
                "_canonical_python_ast",
                "_canonical_python_ast_value",
            }
        ]
        self.assertEqual(2, len(canonicalizers))
        direct_dump_references = [
            node
            for canonicalizer in canonicalizers
            for node in CHECKER.ast.walk(canonicalizer)
            if isinstance(node, CHECKER.ast.Attribute)
            and isinstance(node.value, CHECKER.ast.Name)
            and node.value.id == "ast"
            and node.attr == "dump"
        ]
        self.assertEqual([], direct_dump_references)

        function = CHECKER.ast.parse("def fixture():\n    call()\n").body[0]
        expected = (
            "FunctionDef(name='fixture',args=arguments(posonlyargs=[],args=[],"
            "vararg=None,kwonlyargs=[],kw_defaults=[],kwarg=None,defaults=[]),"
            "body=[Expr(value=Call(func=Name(id='call',ctx=Load()),args=[],"
            "keywords=[]))],decorator_list=[],returns=None,type_comment=None)"
        )
        self.assertEqual(expected, CHECKER._canonical_python_ast(function))
        with mock.patch.object(
            CHECKER.ast,
            "dump",
            side_effect=AssertionError("canonicalizer must not call ast.dump"),
        ):
            self.assertEqual(expected, CHECKER._canonical_python_ast(function))

        absent_type_params = copy.deepcopy(function)
        absent_type_params._fields = tuple(
            name for name in absent_type_params._fields if name != "type_params"
        )
        if hasattr(absent_type_params, "type_params"):
            delattr(absent_type_params, "type_params")
        empty_type_params = copy.deepcopy(function)
        empty_type_params._fields = (
            *(name for name in empty_type_params._fields if name != "type_params"),
            "type_params",
        )
        empty_type_params.type_params = []
        self.assertEqual(
            CHECKER._canonical_python_ast(absent_type_params),
            CHECKER._canonical_python_ast(empty_type_params),
        )

        constant_none = CHECKER.ast.Constant(value=None)
        self.assertEqual(
            "Constant(value=None,kind=None)",
            CHECKER._canonical_python_ast(constant_none),
        )
        keyword_none = CHECKER.ast.parse("call(**values)").body[0].value.keywords[0]
        self.assertEqual(
            "keyword(arg=None,value=Name(id='values',ctx=Load()))",
            CHECKER._canonical_python_ast(keyword_none),
        )

        if CHECKER.AST_TRY_STAR_TYPES:
            regular_try = CHECKER.ast.parse(
                "try:\n    pass\nexcept Exception:\n    pass\n"
            ).body[0]
            starred_try = CHECKER.ast.parse(
                "try:\n    pass\nexcept* Exception:\n    pass\n"
            ).body[0]
            self.assertNotEqual(
                CHECKER._canonical_python_ast(regular_try),
                CHECKER._canonical_python_ast(starred_try),
            )

        if hasattr(CHECKER.ast, "TypeVar"):
            call_semantics = []
            for source in (
                "def generic():\n    call()\n",
                "def generic[T]():\n    call()\n",
            ):
                tree = CHECKER.ast.parse(source)
                nodes = list(CHECKER.ast.walk(tree))
                parents = {
                    child: node
                    for node in nodes
                    for child in CHECKER.ast.iter_child_nodes(node)
                }
                call = next(
                    node for node in nodes if isinstance(node, CHECKER.ast.Call)
                )
                call_semantics.append(
                    CHECKER._python_call_semantics_digest(
                        call,
                        "fixture.call",
                        parents,
                    )
                )
            self.assertNotEqual(*call_semantics)

    def test_try_star_ast_node_is_optional_on_python_310(self) -> None:
        checker_source = CHECKER_PATH.read_text(encoding="utf-8")
        checker_tree = CHECKER.ast.parse(checker_source)
        direct_references = [
            node
            for node in CHECKER.ast.walk(checker_tree)
            if isinstance(node, CHECKER.ast.Attribute)
            and isinstance(node.value, CHECKER.ast.Name)
            and node.value.id == "ast"
            and node.attr == "TryStar"
        ]
        self.assertEqual([], direct_references)
        sentinels = [
            node
            for node in CHECKER.ast.walk(checker_tree)
            if isinstance(node, CHECKER.ast.Call)
            and isinstance(node.func, CHECKER.ast.Name)
            and node.func.id == "getattr"
            and len(node.args) >= 2
            and isinstance(node.args[0], CHECKER.ast.Name)
            and node.args[0].id == "ast"
            and isinstance(node.args[1], CHECKER.ast.Constant)
            and node.args[1].value == "TryStar"
        ]
        self.assertEqual(1, len(sentinels))

        missing = object()
        original = getattr(CHECKER.ast, "TryStar", missing)
        if original is not missing:
            delattr(CHECKER.ast, "TryStar")
        try:
            specification = importlib.util.spec_from_file_location(
                "check_build_inventory_without_try_star", CHECKER_PATH
            )
            assert specification and specification.loader
            compatible = importlib.util.module_from_spec(specification)
            specification.loader.exec_module(compatible)
            self.assertEqual((), compatible.AST_TRY_STAR_TYPES)

            source = (
                "import subprocess\n"
                "try:\n"
                "    subprocess.run(('python3','-c','pass'))\n"
                "except OSError:\n"
                "    pass\n"
            )
            tree = compatible.ast.parse(source)
            nodes = list(compatible.ast.walk(tree))
            parents = {
                child: node
                for node in nodes
                for child in compatible.ast.iter_child_nodes(node)
            }
            call = next(
                node
                for node in nodes
                if isinstance(node, compatible.ast.Call)
                and isinstance(node.func, compatible.ast.Attribute)
                and node.func.attr == "run"
            )
            control_path = compatible._python_ancestor_control_path(call, parents)
            self.assertEqual(1, len(control_path))
            self.assertTrue(control_path[0].startswith("Try("), control_path)

            controlled_alias = (
                "import subprocess as process\n"
                "process.run(['early'])\n"
                "try:\n"
                "    process = object()\n"
                "except Exception:\n"
                "    pass\n"
                "process.run(['late'])\n"
            )
            with tempfile.TemporaryDirectory(prefix="try-star-compat-") as directory:
                root = Path(directory)
                (root / "fixture.py").write_text(controlled_alias, encoding="utf-8")
                with self.assertRaisesRegex(
                    compatible.InventoryError,
                    "control-flow-dependent process alias assignment is unsupported",
                ):
                    compatible._discover_python_launches(root)
        finally:
            if original is not missing:
                setattr(CHECKER.ast, "TryStar", original)

    def test_wrapper_dynamic_ambiguous_and_missing_values_fail_closed(self) -> None:
        sources = (
            "import subprocess\nsubprocess.run(('env','-u'))\n",
            "import subprocess\ndef f(name): subprocess.run(('env','-u',name,'python3','-c','pass'))\n",
            "import subprocess\ndef f(command): subprocess.run(('env',command,'-c','pass'))\n",
            "import subprocess\nsubprocess.run(('uv','run','--python'))\n",
            "import subprocess\nsubprocess.run(('conda','run','--name'))\n",
            "import subprocess\nsubprocess.run(('xcrun','--sdk'))\n",
        )
        for source in sources:
            with (
                self.subTest(source=source),
                self.assertRaisesRegex(
                    CHECKER.InventoryError,
                    "requires a value|dynamic or ambiguous",
                ),
            ):
                self._launches_for_source(source)

    def test_windows_python_launcher_layer_precedes_cpython_cli_scan(self) -> None:
        carriers = (
            "import subprocess\nsubprocess.run(('py','-V:PythonCore/3.14','-c','pass'))\n",
            "import subprocess\nsubprocess.run(('py','-v:3.14t','-cpass'))\n",
            "import subprocess\nsubprocess.run(('py','exec','-V:3.14','-cpass'))\n",
            "import subprocess\nsubprocess.run(('pymanager','exec','-c','pass'))\n",
            "import subprocess\nsubprocess.run(('pyw','-cpass'))\n",
            "import subprocess\nsubprocess.run(('py','-3.14t-arm64','-cpass'))\n",
        )
        for source in carriers:
            with self.subTest(source=source):
                with self.assertRaisesRegex(
                    CHECKER.InventoryError, "indirect Python -c carrier"
                ):
                    self._launches_for_source(source)

        terminal = self._launches_for_source(
            "import subprocess\n"
            "subprocess.run(('py','--list','-c','pass'))\n"
            "subprocess.run(('py','help','-c','pass'))\n"
            "subprocess.run(('pymanager','list','-c','pass'))\n"
            "subprocess.run(('pywmanager','install','3.14'))\n"
        )
        self.assertEqual(4, len(terminal))
        self.assertFalse(any("carrier_semantics_digest" in item for item in terminal))

        malformed = (
            "import subprocess\nsubprocess.run(('py','-V:','-cpass'))\n",
            "import subprocess\nsubprocess.run(('pymanager','unknown','-cpass'))\n",
            "import subprocess\ndef f(command): subprocess.run(('pymanager',command,'-cpass'))\n",
        )
        for source in malformed:
            with self.subTest(source=source):
                with self.assertRaisesRegex(CHECKER.InventoryError, "Windows Python"):
                    self._launches_for_source(source)

        direct = self._launches_for_source(
            "import subprocess\nsubprocess.run(('python3.14t.exe','-cpass'))\n"
        )
        self.assertEqual(1, len(direct))
        self.assertIn("carrier_semantics_digest", direct[0])
        with self.assertRaisesRegex(CHECKER.InventoryError, "shell Python -c carriers"):
            self._launches_for_source("import os\nos.system('py -V:3.14 -cpass')\n")

    def test_embedded_scope_tokens_prevent_duplicate_sibling_ids(self) -> None:
        code = (
            "import subprocess\n"
            "if True:\n"
            " def nested(): subprocess.run(['one'])\n"
            "else:\n"
            " def nested(): subprocess.run(['two'])\n"
        )
        launches = self._launches_for_source(
            "import subprocess,sys\ndef f():\n"
            f"    subprocess.run((sys.executable,'-c',{code!r}))\n"
        )
        embedded = [item for item in launches if item["anchor"].get("origin_kind")]
        self.assertEqual(2, len(embedded))
        self.assertEqual(2, len({item["id"] for item in embedded}))
        self.assertEqual(
            {"nested-1", "nested-2"},
            {item["anchor"]["embedded_scope"] for item in embedded},
        )

    def test_embedded_python_carriers_fail_closed(self) -> None:
        fixtures = (
            (
                "import subprocess,sys\ndef f(code): subprocess.run((sys.executable,'-c',code))\n",
                "static string",
            ),
            (
                "import subprocess,sys\ndef f(): subprocess.run((sys.executable,'-c','not python !!!'))\n",
                "syntax error",
            ),
            (
                "import subprocess,sys\ndef f(): subprocess.run((sys.executable,'-c','pass'),shell=True)\n",
                "shell Python -c carriers",
            ),
            (
                "import subprocess,sys\ndef f(parts): subprocess.run((sys.executable,*parts,'-c','pass'))\n",
                "expansion is ambiguous",
            ),
            (
                "import subprocess,sys\ndef f(options): subprocess.run((sys.executable,'-c','pass'),**options)\n",
                "keyword expansion is ambiguous",
            ),
            (
                "import subprocess\ndef f(): subprocess.run(('/usr/bin/env','python3','-c','pass'))\n",
                "indirect Python -c carrier",
            ),
            (
                "import subprocess\ndef f(interpreter): subprocess.run(('/usr/bin/env',interpreter,'-c','pass'))\n",
                "dynamic or ambiguous",
            ),
            (
                "import subprocess\ndef f(): subprocess.run(('/usr/bin/env','-S','python3 -I -c pass'))\n",
                "indirect Python -c carrier",
            ),
            (
                "import subprocess\ndef f(): subprocess.run(('/usr/bin/env','--split-string=python3 -c pass'))\n",
                "indirect Python -c carrier",
            ),
            (
                "import subprocess\ndef f(): subprocess.run(('bash','-c','python3 -c pass'))\n",
                "indirect Python -c carrier",
            ),
            (
                "import subprocess\ndef f(): subprocess.run(('cmd.exe','/c','python.exe -c pass'))\n",
                "indirect Python -c carrier",
            ),
            (
                "import subprocess\ndef f(): subprocess.run(('pwsh','-Command','python3 -c pass'))\n",
                "indirect Python -c carrier",
            ),
            (
                "import subprocess,sys\ndef f(): subprocess.run((sys.executable,'-c',\"exec('pass')\"))\n",
                "dynamic Python code execution",
            ),
        )
        for source, expected in fixtures:
            with self.subTest(source=source, expected=expected):
                with self.assertRaisesRegex(CHECKER.InventoryError, expected):
                    self._launches_for_source(source)

        original_limit = CHECKER.EMBEDDED_PYTHON_MAX_BYTES
        CHECKER.EMBEDDED_PYTHON_MAX_BYTES = 8
        try:
            with self.assertRaisesRegex(CHECKER.InventoryError, "byte limit"):
                self._launches_for_source(
                    "import subprocess,sys\ndef f(): subprocess.run((sys.executable,'-c','print(123456789)'))\n"
                )
        finally:
            CHECKER.EMBEDDED_PYTHON_MAX_BYTES = original_limit

        original_depth = CHECKER.EMBEDDED_PYTHON_MAX_DEPTH
        CHECKER.EMBEDDED_PYTHON_MAX_DEPTH = 0
        try:
            with self.assertRaisesRegex(CHECKER.InventoryError, "depth limit"):
                self._launches_for_source(
                    "import subprocess,sys\ndef f(): subprocess.run((sys.executable,'-c','pass'))\n"
                )
        finally:
            CHECKER.EMBEDDED_PYTHON_MAX_DEPTH = original_depth

        original_nodes = CHECKER.EMBEDDED_PYTHON_MAX_NODES
        CHECKER.EMBEDDED_PYTHON_MAX_NODES = 1
        try:
            with self.assertRaisesRegex(CHECKER.InventoryError, "node limit"):
                self._launches_for_source(
                    "import subprocess,sys\ndef f(): subprocess.run((sys.executable,'-c','pass'))\n"
                )
        finally:
            CHECKER.EMBEDDED_PYTHON_MAX_NODES = original_nodes

        original_units = CHECKER.EMBEDDED_PYTHON_MAX_UNITS
        CHECKER.EMBEDDED_PYTHON_MAX_UNITS = 0
        try:
            with self.assertRaisesRegex(CHECKER.InventoryError, "unit limit"):
                self._launches_for_source(
                    "import subprocess,sys\ndef f(): subprocess.run((sys.executable,'-c','pass'))\n"
                )
        finally:
            CHECKER.EMBEDDED_PYTHON_MAX_UNITS = original_units

    def test_git_config_c_is_not_a_python_carrier(self) -> None:
        launches = self._launches_for_source(
            "import subprocess\n"
            "def f(): subprocess.run(('git','-c','core.quotepath=false','status'))\n"
        )
        self.assertEqual(1, len(launches))
        self.assertFalse(launches[0]["anchor"].get("origin_kind"))

    def test_new_public_files_of_varied_shapes_are_unclassified(self) -> None:
        paths = ("new.zig", "new.json", "tool", ".hidden/new.zig")
        for rel in paths:
            path = self.root / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("fixture\n", encoding="utf-8")
        errors = "\n".join(CHECKER.validate(self.root, self.inventory_path))
        for rel in paths:
            self.assertIn(rel, errors)
        self.assertIn("archive public files are not all classified", errors)

    def test_marker_free_archive_may_omit_absent_derived_detail(self) -> None:
        self.assertFalse((self.root / ".git").exists())
        omitted = "include/zynum/blas/blas.h"
        (self.root / omitted).unlink()
        inventory = self._inventory()
        inventory["derived_candidates"] = [
            item for item in inventory["derived_candidates"] if item["path"] != omitted
        ]
        self._write_inventory(inventory)
        self.assertEqual("archive", CHECKER._make_public_file_universe(self.root).mode)
        self.assertEqual([], CHECKER.validate(self.root, self.inventory_path))

    def test_marker_free_archive_requires_detail_for_present_member(self) -> None:
        self.assertFalse((self.root / ".git").exists())
        present = "include/zynum/blas/blas.h"
        self.assertTrue((self.root / present).is_file())
        inventory = self._inventory()
        inventory["derived_candidates"] = [
            item for item in inventory["derived_candidates"] if item["path"] != present
        ]
        self._write_inventory(inventory)
        errors = "\n".join(CHECKER.validate(self.root, self.inventory_path))
        self.assertIn("missing required reviewed derived details", errors)
        self.assertIn("requires derived_candidates detail", errors)

    def test_repository_mode_keeps_exact_derived_candidate_existence(self) -> None:
        omitted = "include/zynum/blas/blas.h"
        (self.root / omitted).unlink()
        context = CHECKER._make_discovery_context(self.root, self.inventory_path)
        observed, _ = CHECKER._discover_repository_file_classifications(
            self.root, context
        )
        with mock.patch.object(
            CHECKER,
            "_discover_repository_file_classifications",
            return_value=(observed, True),
        ):
            errors = "\n".join(CHECKER.validate(self.root, self.inventory_path))
        self.assertIn(
            "derived candidate path must belong to the public universe", errors
        )

    def test_test_inventory_public_files_are_strictly_classified(self) -> None:
        rows = {
            item["path"]: item
            for item in self.inventory["repository_file_classifications"]
        }
        expected = {
            "src/blas/task_runtime_host_object_root.zig": "zig-source",
            "test/build/level2_width_artifact_probe_contract.zig": "zig-source",
            "test/build/level2_width_default_artifact_probe.zig": "zig-source",
            "test/build/level2_width_enabled_artifact_probe.zig": "zig-source",
            "test/build/test_build_profiles.py": "python-source",
            "src/blas/runtime_lifecycle_probe.zig": "zig-source",
            "test/runtime/blas_io_lifecycle.c": "foreign-language-source",
            "test/runtime/run_blas_io_lifecycle.py": "python-source",
            "test/build/test_test_inventory.py": "python-source",
            "test/build/windows_python_tooling_probe_fixture.zig": "zig-source",
            "tools/check_test_inventory.py": "python-source",
            "tools/test_inventory.json": "json-data",
            "tools/test_inventory_runner.zig": "zig-source",
        }
        self.assertEqual(len(self.inventory["repository_file_classifications"]), len(rows))
        for path, kind in expected.items():
            with self.subTest(path=path):
                self.assertEqual(kind, rows[path]["kind"])
                self.assertEqual("non-generated-source", rows[path]["class"])
                self.assertEqual("test-infrastructure", rows[path]["owner"])
        for path in (
            "src/lapack/core/complex_cholesky.zig",
            "src/numerics/complex.zig",
            "src/numerics/pivot.zig",
            "src/lapack/core/lu.zig",
            "src/lapack/complex_cholesky.zig",
            "src/lapack/complex_lu.zig",
            "src/lapack/complex_support.zig",
            "src/lapack/factorization.zig",
            "src/lapack/least_squares.zig",
            "src/lapack/svd.zig",
            "src/lapack/symmetric_eigen.zig",
            "src/sparse/cg.zig",
            "src/sparse/scalar.zig",
            "src/sparse/triangular.zig",
            "test/lapack/complex_test.zig",
            "test/lapack/least_squares_test.zig",
            "test/lapack/svd_test.zig",
            "test/lapack/symmetric_eigen_test.zig",
            "test/sparse/cg_test.zig",
            "test/sparse/complex_test.zig",
            "test/sparse/triangular_test.zig",
        ):
            with self.subTest(path=path):
                self.assertEqual("zig-source", rows[path]["kind"])
                self.assertEqual("non-generated-source", rows[path]["class"])
                self.assertEqual(
                    "test-infrastructure" if path.startswith("test/") else "library-source",
                    rows[path]["owner"],
                )
        self.assertNotIn("COPYING.LESSER", rows)
        for path in ("COPYING", "LICENSE"):
            with self.subTest(path=path):
                self.assertEqual("legal-governance", rows[path]["kind"])
                self.assertEqual("non-generated-source", rows[path]["class"])
                self.assertEqual("project-governance", rows[path]["owner"])
        dependabot = rows[".github/dependabot.yml"]
        self.assertEqual("configuration-metadata", dependabot["kind"])
        self.assertEqual("non-generated-source", dependabot["class"])
        self.assertEqual("workflow-maintainers", dependabot["owner"])
        section_policy = rows["src/blas/kernels/isolated/object_format_sections.zig"]
        self.assertEqual("zig-source", section_policy["kind"])
        self.assertEqual("non-generated-source", section_policy["class"])
        self.assertEqual("library-source", section_policy["owner"])
        self.assertEqual(
            CHECKER.REQUIRED_SECTION_FACT_DIGESTS["repository_file_classifications"],
            CHECKER._json_fact_digest(
                self.inventory["repository_file_classifications"]
            ),
        )
        dependabot_path = self.root / CHECKER.DEPENDABOT_CONFIG_PATH
        dependabot_bytes = dependabot_path.read_bytes()
        self.assertEqual(
            CHECKER.REVIEWED_DEPENDABOT_CONFIG_SHA256,
            hashlib.sha256(dependabot_bytes).hexdigest(),
        )
        dependabot_path.write_bytes(
            dependabot_bytes.replace(b'"github-actions"', b'"npm"', 1)
        )
        self._assert_error_contains("reviewed Dependabot configuration changed")

    def test_reviewed_classification_digest_class_owner_and_detail_bind(self) -> None:
        inventory = self._inventory()
        row = next(
            item
            for item in inventory["repository_file_classifications"]
            if item["path"] == "include/zynum/blas/blas.h"
        )
        row["owner"] = "documentation-maintainers"
        inventory["repository_file_classifications_digest"] = CHECKER._json_fact_digest(
            inventory["repository_file_classifications"]
        )
        self._write_inventory(inventory)
        errors = "\n".join(CHECKER.validate(self.root, self.inventory_path))
        self.assertIn(
            "repository_file_classifications reviewed fact set changed", errors
        )
        self.assertIn("detail owner mismatch", errors)

        inventory = copy.deepcopy(self.inventory)
        detail = next(
            item
            for item in inventory["derived_candidates"]
            if item["path"] == "include/zynum/blas/blas.h"
        )
        detail["class"] = "non-generated-source"
        self._write_inventory(inventory)
        self._assert_error_contains("class must match whole-ledger classification")

    def test_payload_controller_identity_is_fail_closed(self) -> None:
        inventory = self._inventory()
        controllers = {
            item["id"]: item
            for item in inventory["python_launches"]
            if item["id"] in CHECKER.PAYLOAD_CONTROLLER_LINKS
        }
        ids = list(CHECKER.PAYLOAD_CONTROLLER_LINKS)
        self.assertEqual(6, len(ids))
        self.assertEqual(
            7,
            sum(
                len(bindings) for bindings in CHECKER.PAYLOAD_CONTROLLER_LINKS.values()
            ),
        )
        level1_id = (
            "python-launch:bench/tools/run_level1_report.py:run_once:subprocess.run:1"
        )
        self.assertEqual(
            (
                "compile:build.zig:build:level1_probe",
                "compile:build.zig:build:dcopy_probe",
            ),
            tuple(
                binding.payload_artifact_id
                for binding in CHECKER.PAYLOAD_CONTROLLER_LINKS[level1_id]
            ),
        )
        controllers[ids[0]]["compile_for"] = "host"
        controllers[level1_id]["payload_bindings"].pop()
        controllers[ids[2]]["payload_bindings"][0]["payload_artifact_id"] = (
            "compile:build.zig:build:lib"
        )
        controllers[ids[3]]["payload_bindings"][0]["execution_transport"] = (
            "nonexistent-runner"
        )
        controllers[ids[4]]["payload_bindings"][0]["compatibility_requirement"] = (
            "unchecked"
        )
        payload = next(
            item
            for item in inventory["build_observations"]
            if item["id"]
            == CHECKER.PAYLOAD_CONTROLLER_LINKS[ids[5]][0].payload_artifact_id
        )
        payload["artifact_kind"] = "library"
        self._write_inventory(inventory)
        errors = "\n".join(CHECKER.validate(self.root, self.inventory_path))
        self.assertIn("compile_for must be requested-target", errors)
        self.assertIn("payload_bindings must exactly bind", errors)
        self.assertIn("must be an executable compile observation", errors)

    def test_level1_payload_source_selection_and_gap_are_fail_closed(self) -> None:
        source_path = self.root / "bench/tools/run_level1_report.py"
        source = source_path.read_text(encoding="utf-8")
        source_path.write_text(
            source.replace(
                '        args.copy_probe,\n        "--lib",',
                '        args.level1_probe,\n        "--lib",',
                1,
            ),
            encoding="utf-8",
        )
        errors = "\n".join(CHECKER.validate(self.root, self.inventory_path))
        self.assertIn("run_copy_op:run_once:1 payload selector does not match", errors)

        source_path.write_text(source, encoding="utf-8")
        inventory = self._inventory()
        gap = next(
            item
            for item in inventory["current_gaps"]
            if item["id"] == "gap:cross-target-benchmark-payload-execution"
        )
        gap["controller_ids"].pop()
        gap["controller_count"] -= 1
        gap["observed_result"] = (
            "no emulator or remote runner is wired for a hard-coded subset"
        )
        self._write_inventory(inventory)
        errors = "\n".join(CHECKER.validate(self.root, self.inventory_path))
        self.assertIn("must bind the exact reviewed controller/payload set", errors)
        self.assertIn("does not match the mechanical controller/payload counts", errors)


class FiniteStandardBindingGenerationTests(unittest.TestCase):
    """Finite source contracts; synthetic local acceptance is never ABI evidence."""

    PROFILE = 'mkl2026.lp64'
    SET = 'potrf-potrs-ordinary-lp64-v1'
    LU_SET = 'potrf-potrs-getrf-getrs-ordinary-lp64-v1'
    TARGET = 'apple-arm64-default-c-v1'

    @classmethod
    def setUpClass(cls) -> None:
        with mock.patch.object(sys, 'path', [str(REPOSITORY_ROOT / 'tools'), *sys.path]), \
             mock.patch.object(sys, 'dont_write_bytecode', True):
            import checked_abi_schema
            import generate_checked_abi
            import standard_abi_bindings
        cls.schema = checked_abi_schema
        cls.generator = generate_checked_abi
        cls.standard = standard_abi_bindings
        cls.production_acceptances = dict(standard_abi_bindings.GENERATION_ACCEPTANCES)
        for module in (cls.schema, cls.generator, cls.standard):
            if Path(module.__file__).resolve().parent != (REPOSITORY_ROOT / 'tools').resolve():
                raise AssertionError('finite tests must import this repository tooling')

    @contextlib.contextmanager
    def fixture(self, profile=PROFILE, set_id=SET):
        """Two independent synthetic source files, exact in-memory test acceptance.

        No historical workspace, production pin rewrite or approval loader is
        used. The temporary source projection deliberately is not library code.
        Every patch and directory is reverted when this context exits.
        """
        self.assertEqual(self.production_acceptances, dict(self.standard.GENERATION_ACCEPTANCES))
        with tempfile.TemporaryDirectory() as directory, contextlib.ExitStack() as stack:
            root = Path(directory)
            sources = {'src/fixture_factor.zig': b'// test fixture factor source\n',
                       'src/fixture_services.zig': b'// test fixture services source\n'}
            for relative, content in sources.items():
                destination = root / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(content)
            relative, digest = self.standard.RESERVED_BLAS_MANIFEST
            destination = root / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes((REPOSITORY_ROOT / relative).read_bytes())
            self.assertEqual(digest, hashlib.sha256(destination.read_bytes()).hexdigest())
            dependencies = tuple(sorted((path, hashlib.sha256(data).hexdigest()) for path, data in sources.items()))
            envelope = hashlib.sha256(json.dumps(dependencies, separators=(',', ':')).encode()).hexdigest()
            stack.enter_context(mock.patch.object(self.schema, '_binding_source_root', return_value=root))
            stack.enter_context(mock.patch.object(self.standard, 'SOURCE_DEPENDENCIES', dependencies))
            stack.enter_context(mock.patch.object(self.standard, 'SOURCE_ENVELOPE_SHA256', envelope))
            stack.enter_context(mock.patch.object(self.standard, 'SOURCE_SNAPSHOT_SHA256', envelope))
            plan = self.schema._binding_candidate(profile, set_id, self.TARGET)
            record = {'id': 'unit-test-only-local-source-acceptance', 'profile': profile, 'set': set_id,
                      'target': self.TARGET, 'scope': 'finite-callable-generation-only',
                      'plan_sha256': self.schema._binding_digest(plan), 'source_envelope_sha256': envelope,
                      'review_sha256': hashlib.sha256(b'unit fixture; no native or review evidence').hexdigest()}
            stack.enter_context(mock.patch.object(self.standard, 'GENERATION_ACCEPTANCES',
                                                  {(profile, set_id, self.TARGET): record}))
            yield root, record
        self.assertEqual(self.production_acceptances, dict(self.standard.GENERATION_ACCEPTANCES))

    @contextlib.contextmanager
    def changed_binding(self, index=0, set_id=SET, **updates):
        bindings = dict(self.standard.BINDINGS)
        identity = self.standard.SETS[set_id]['bindings'][index]
        bindings[identity] = {**bindings[identity], **updates}
        with mock.patch.object(self.standard, 'BINDINGS', bindings):
            yield

    def select(self, profile=PROFILE, set_id=SET, target=TARGET):
        return self.schema.select_binding_set(profile, set_id, target)

    def cli(self, arguments):
        with mock.patch.object(sys, 'argv', ['generate_checked_abi.py', *arguments]):
            self.generator.main()

    def fake_formatter(self, argv, **kwargs):
        self.assertEqual(['fixture-formatter', 'fmt', '--stdin'], argv)
        self.assertEqual({'input', 'text', 'capture_output', 'check'}, set(kwargs))
        self.assertIs(kwargs['text'], True)
        self.assertIs(kwargs['capture_output'], True)
        self.assertIs(kwargs['check'], True)
        return mock.Mock(stdout=kwargs['input'])

    def test_production_acceptance_is_exact_finite_sets_with_shared_diagnostic_sources(self):
        expected = {('mkl2026.' + width,
                     stem + width + '-v1', self.TARGET)
                    for stem in ('potrf-potrs-getrf-getrs-ordinary-',
                                 'potrf-potrs-getrf-getrs-gesv-ordinary-',
                                 'potrf-potrs-getrf-getrs-gesv-stev-ordinary-',
                                 'potrf-potrs-getrf-getrs-gesv-stev-gelsd-ordinary-')
                    for width in ('lp64', 'ilp64')}
        expected |= {
            ('mkl2026.lp64', 'hermitian-eigen-ordinary-lp64-v1', self.TARGET),
            ('mkl2026.lp64', 'potrf-potrs-getrf-getrs-gesv-stev-gelsd-heev-hegv-ordinary-lp64-v1', self.TARGET),
        }
        expected.update({("mkl2026.lp64", name, self.TARGET) for name in (
            "geqrf-ormqr-unmqr-ordinary-lp64-v1",
            "potrf-potrs-getrf-getrs-gesv-stev-gelsd-heev-hegv-geqrf-ormqr-unmqr-ordinary-lp64-v1",
            "orgqr-ungqr-ordinary-lp64-v1",
            "geqrf-ormqr-unmqr-orgqr-ungqr-ordinary-lp64-v1",
            "potrf-potrs-getrf-getrs-gesv-stev-gelsd-heev-hegv-geqrf-ormqr-unmqr-orgqr-ungqr-ordinary-lp64-v1",
            'gesvd-ordinary-lp64-v1',
            'potrf-potrs-getrf-getrs-gesv-stev-gelsd-heev-hegv-gesvd-ordinary-lp64-v1',
            'potrf-potrs-getrf-getrs-gesv-stev-gelsd-heev-hegv-complex-gelsd-gesvd-ordinary-lp64-v1',
            'potrf-potrs-getrf-getrs-gesv-stev-gelsd-heev-hegv-geqrf-ormqr-unmqr-orgqr-ungqr-gesvd-ordinary-lp64-v1',
            'potrf-potrs-getrf-getrs-gesv-stev-gelsd-heev-hegv-geqrf-ormqr-unmqr-orgqr-ungqr-complex-gelsd-gesvd-ordinary-lp64-v1',
            "complex-gelsd-ordinary-lp64-v1",
            "potrf-potrs-getrf-getrs-gesv-stev-gelsd-heev-hegv-complex-gelsd-ordinary-lp64-v1",
            "potrf-potrs-getrf-getrs-gesv-stev-gelsd-heev-hegv-geqrf-ormqr-unmqr-orgqr-ungqr-complex-gelsd-ordinary-lp64-v1",
        )})
        def original_symbol_count(set_id):
            count = 8 if set_id == 'hermitian-eigen-ordinary-lp64-v1' else 56 if '-heev-hegv-' in set_id else 48 if '-gelsd-' in set_id else 44 if '-stev-' in set_id else 40 if '-gesv-' in set_id else 32
            count = {
                "geqrf-ormqr-unmqr-ordinary-lp64-v1": 16,
                "potrf-potrs-getrf-getrs-gesv-stev-gelsd-heev-hegv-geqrf-ormqr-unmqr-ordinary-lp64-v1": 72,
                "orgqr-ungqr-ordinary-lp64-v1": 8,
                "geqrf-ormqr-unmqr-orgqr-ungqr-ordinary-lp64-v1": 24,
                "potrf-potrs-getrf-getrs-gesv-stev-gelsd-heev-hegv-geqrf-ormqr-unmqr-orgqr-ungqr-ordinary-lp64-v1": 80,
                'gesvd-ordinary-lp64-v1': 8,
                'potrf-potrs-getrf-getrs-gesv-stev-gelsd-heev-hegv-gesvd-ordinary-lp64-v1': 64,
                'potrf-potrs-getrf-getrs-gesv-stev-gelsd-heev-hegv-complex-gelsd-gesvd-ordinary-lp64-v1': 68,
                'potrf-potrs-getrf-getrs-gesv-stev-gelsd-heev-hegv-geqrf-ormqr-unmqr-orgqr-ungqr-gesvd-ordinary-lp64-v1': 88,
                'potrf-potrs-getrf-getrs-gesv-stev-gelsd-heev-hegv-geqrf-ormqr-unmqr-orgqr-ungqr-complex-gelsd-gesvd-ordinary-lp64-v1': 92,
                "complex-gelsd-ordinary-lp64-v1": 4,
                "potrf-potrs-getrf-getrs-gesv-stev-gelsd-heev-hegv-complex-gelsd-ordinary-lp64-v1": 60,
                "potrf-potrs-getrf-getrs-gesv-stev-gelsd-heev-hegv-geqrf-ormqr-unmqr-orgqr-ungqr-complex-gelsd-ordinary-lp64-v1": 84,
            }.get(set_id, count)
            return count

        # These independently stated install combinations extend the original
        # 23 sets. Never derive the expected whitelist from production SETS or
        # GENERATION_ACCEPTANCES: an unexpected extra acceptance must fail.
        install_bases = {
            ('mkl2026.lp64', 'potrf-potrs-ordinary-lp64-v1'): 16,
            ('mkl2026.ilp64', 'potrf-potrs-ordinary-ilp64-v1'): 16,
            ('mkl2026.ilp64', 'potrf-potrs-getrf-getrs-gesv-stev-gelsd-ordinary-ilp64-v1'): 48,
        }
        for qr in (False, True):
            for complex_gelsd in (False, True):
                for svd in (False, True):
                    name = ('potrf-potrs-getrf-getrs-gesv-stev-gelsd-heev-hegv'
                            + ('-geqrf-ormqr-unmqr-orgqr-ungqr' if qr else '')
                            + ('-complex-gelsd' if complex_gelsd else '')
                            + ('-gesvd' if svd else '') + '-ordinary-lp64-v1')
                    install_bases[('mkl2026.lp64', name)] = 56 + 24 * qr + 4 * complex_gelsd + 8 * svd
        self.assertEqual(11, len(install_bases))
        extra_counts = {}
        for (profile, name), count in install_bases.items():
            extra_counts[(profile, name.replace('-ordinary-', '-posv-ordinary-'), self.TARGET)] = count + 8
        self.assertEqual(11, len(extra_counts))
        expected.update(extra_counts)
        self.assertEqual(34, len(expected))
        solve_bases = {**{(profile, name, self.TARGET): count
                          for (profile, name), count in install_bases.items()}, **extra_counts}
        solve_counts = {(profile, name.replace('-ordinary-', '-trtrs-ordinary-'), target): count + 8
                        for (profile, name, target), count in solve_bases.items()}
        solve_counts.update({('mkl2026.' + width, 'trtrs-ordinary-' + width + '-v1', self.TARGET): 8
                             for width in ('lp64', 'ilp64')})
        self.assertEqual(24, len(solve_counts))
        extra_counts.update(solve_counts)
        expected.update(solve_counts)
        self.assertEqual(58, len(expected))
        inverse_base = set(expected)
        inverse_sources = {}
        # Reuse the original independent symbol-count fixture, never a plan's
        # actual count, for every original base plus its eight inverse symbols.
        for profile, name, target in sorted(inverse_base):
            old_count = extra_counts.get((profile, name, target))
            if old_count is None:
                old_count = original_symbol_count(name)
            inverse_sources[(profile, name, target)] = old_count
        inverse_counts = {(profile, name.replace('-ordinary-', '-trtri-ordinary-'), target): count + 8
                          for (profile, name, target), count in inverse_sources.items()}
        inverse_counts.update({('mkl2026.' + width, 'trtri-ordinary-' + width + '-v1', self.TARGET): 8
                               for width in ('lp64', 'ilp64')})
        self.assertEqual(60, len(inverse_counts))
        extra_counts.update(inverse_counts)
        expected.update(inverse_counts)
        self.assertEqual(118, len(expected))
        lu_inverse_sources = {(profile, name, target): extra_counts.get(
            (profile, name, target), original_symbol_count(name))
            for profile, name, target in expected}
        lu_inverse_counts = {(profile, name.replace('-ordinary-', '-getri-ordinary-'), target): count + 8
                             for (profile, name, target), count in lu_inverse_sources.items()}
        lu_inverse_counts.update({('mkl2026.' + width, 'getri-ordinary-' + width + '-v1', self.TARGET): 8
                                  for width in ('lp64', 'ilp64')})
        self.assertEqual(120, len(lu_inverse_counts))
        extra_counts.update(lu_inverse_counts)
        expected.update(lu_inverse_counts)
        self.assertEqual(238, len(expected))
        self.assertEqual(expected, set(self.standard.GENERATION_ACCEPTANCES))
        for profile, set_id, target in sorted(expected):
            plan = self.select(profile, set_id, target)
            count = 8 if set_id == 'hermitian-eigen-ordinary-lp64-v1' else 56 if '-heev-hegv-' in set_id else 48 if '-gelsd-' in set_id else 44 if '-stev-' in set_id else 40 if '-gesv-' in set_id else 32
            count = {
                "geqrf-ormqr-unmqr-ordinary-lp64-v1": 16,
                "potrf-potrs-getrf-getrs-gesv-stev-gelsd-heev-hegv-geqrf-ormqr-unmqr-ordinary-lp64-v1": 72,
                "orgqr-ungqr-ordinary-lp64-v1": 8,
                "geqrf-ormqr-unmqr-orgqr-ungqr-ordinary-lp64-v1": 24,
                "potrf-potrs-getrf-getrs-gesv-stev-gelsd-heev-hegv-geqrf-ormqr-unmqr-orgqr-ungqr-ordinary-lp64-v1": 80,
                'gesvd-ordinary-lp64-v1': 8,
                'potrf-potrs-getrf-getrs-gesv-stev-gelsd-heev-hegv-gesvd-ordinary-lp64-v1': 64,
                'potrf-potrs-getrf-getrs-gesv-stev-gelsd-heev-hegv-complex-gelsd-gesvd-ordinary-lp64-v1': 68,
                'potrf-potrs-getrf-getrs-gesv-stev-gelsd-heev-hegv-geqrf-ormqr-unmqr-orgqr-ungqr-gesvd-ordinary-lp64-v1': 88,
                'potrf-potrs-getrf-getrs-gesv-stev-gelsd-heev-hegv-geqrf-ormqr-unmqr-orgqr-ungqr-complex-gelsd-gesvd-ordinary-lp64-v1': 92,
                "complex-gelsd-ordinary-lp64-v1": 4,
                "potrf-potrs-getrf-getrs-gesv-stev-gelsd-heev-hegv-complex-gelsd-ordinary-lp64-v1": 60,
                "potrf-potrs-getrf-getrs-gesv-stev-gelsd-heev-hegv-geqrf-ormqr-unmqr-orgqr-ungqr-complex-gelsd-ordinary-lp64-v1": 84,
            }.get(set_id, count)
            count = extra_counts.get((profile, set_id, target), count)
            self.assertEqual(count, len(plan['symbols']))
            if '-getri-ordinary-' in set_id:
                records = [b for b in plan['bindings'] if b['operation'] == 'lu.inverse']
                self.assertEqual(8, len(records))
                self.assertEqual({(s, level) for s in ('f32', 'f64', 'complex_f32', 'complex_f64')
                                  for level in ('high', 'work')}, {(b['scalar'], b['level']) for b in records})
                integer = 'int' if profile == 'mkl2026.lp64' else 'long long int'
                for binding in records:
                    names = ['matrix_layout', 'n', 'a', 'lda', 'ipiv']
                    if binding['level'] == 'work':
                        names.extend(['work', 'lwork'])
                    params = binding['wire_parameters']
                    self.assertEqual(names, [p['name'] for p in params])
                    self.assertFalse(params[2]['pointee_const'])
                    self.assertTrue(params[4]['pointee_const'])
                    self.assertEqual(integer, params[4]['c_type'])
                    self.assertEqual(integer, binding['wire_return']['c_type'])
                    self.assertEqual('raw_getri_profile_defaults', binding['recipe'])
                    if binding['level'] == 'work':
                        self.assertFalse(params[5]['pointee_const'])
                        self.assertEqual(params[2]['c_type'], params[5]['c_type'])
                        self.assertEqual(integer, params[6]['c_type'])
            if set_id == 'hermitian-eigen-ordinary-lp64-v1' or '-heev-hegv-' in set_id:
                records = [b for b in plan['bindings'] if b['operation'].startswith('hermitian.')]
                self.assertEqual(8, len(records))
                self.assertEqual({'complex_f32', 'complex_f64'}, {b['scalar'] for b in records})
                self.assertEqual({'hermitian.eigen', 'hermitian.generalized_eigen'}, {b['operation'] for b in records})
                self.assertEqual({'high', 'work'}, {b['level'] for b in records})
            if '-gelsd-' in set_id and not set_id.startswith('complex-gelsd-'):
                records = [b for b in plan['bindings'] if b['operation'] == 'least_squares.rank_revealing']
                self.assertEqual(4, len(records))
                for row in records:
                    names = ['matrix_layout', 'm', 'n', 'nrhs', 'a', 'lda', 'b', 'ldb', 's', 'rcond', 'rank']
                    if row['level'] == 'work':
                        names.extend(['work', 'lwork', 'iwork'])
                    self.assertEqual(names, [p['name'] for p in row['wire_parameters']])
                    self.assertIn(row['scalar'], ('f32', 'f64'))
            if '-stev-' in set_id:
                records = [b for b in plan['bindings'] if b['operation'] == 'tridiagonal.eigen']
                self.assertEqual(4, len(records))
                for row in records:
                    names = ['matrix_layout', 'jobz', 'n', 'd', 'e', 'z', 'ldz']
                    if row['level'] == 'work':
                        names.append('work')
                    self.assertEqual(names, [p['name'] for p in row['wire_parameters']])
                    self.assertIn(row['scalar'], ('f32', 'f64'))
            self.assertFalse(plan['full_profile_ready'])
            self.assertEqual('finite-callable-generation-only', plan['generation_acceptance']['scope'])
            dependencies = dict(plan['source_dependencies'])
            for path in ('src/interop/lapacke_services.zig', 'src/interop/lapacke_diagnostics.zig',
                         'src/interop/lapacke_default_errors.c'):
                self.assertEqual(hashlib.sha256((REPOSITORY_ROOT / path).read_bytes()).hexdigest(), dependencies[path])
            self.assertIn('per-call-profile-default-diagnostics-v1', plan['semantic_contracts'])
            with self.assertRaisesRegex(self.schema.ProfileError, 'not ready'):
                self.schema.select_profile(profile)
        with self.assertRaisesRegex(self.schema.ProfileError, 'No callable generation acceptance'):
            self.select(self.PROFILE, self.SET)

    def test_complex_gelsd_component_types_and_three_query_destinations(self):
        plan = self.select(self.PROFILE, 'complex-gelsd-ordinary-lp64-v1')
        self.assertEqual(['LAPACKE_cgelsd', 'LAPACKE_cgelsd_work', 'LAPACKE_zgelsd', 'LAPACKE_zgelsd_work'], list(plan['symbols']))
        for record in plan['bindings']:
            real = 'float' if record['scalar'] == 'complex_f32' else 'double'
            complex_type = 'MKL_Complex8' if record['scalar'] == 'complex_f32' else 'MKL_Complex16'
            fields = {p['name']: p for p in record['wire_parameters']}
            self.assertEqual(real, fields['s']['c_type'])
            self.assertEqual(real, fields['rcond']['c_type'])
            self.assertEqual(complex_type, fields['a']['c_type'])
            self.assertEqual(complex_type, fields['b']['c_type'])
            self.assertEqual('int', fields['rank']['c_type'])
            if record['level'] == 'work':
                self.assertEqual(['work', 'lwork', 'rwork', 'iwork'], [p['name'] for p in record['wire_parameters']][-4:])
                self.assertEqual(real, fields['rwork']['c_type'])
                self.assertEqual(complex_type, fields['work']['c_type'])
                self.assertEqual('int', fields['iwork']['c_type'])
            self.assertTrue(all(not p['pointee_const'] for p in record['wire_parameters']))
        self.assertFalse(plan['full_profile_ready'])

    def test_complex_gelsd_type_or_workspace_reordering_fails_before_render(self):
        selected = 'complex-gelsd-ordinary-lp64-v1'
        identity = self.standard.SETS[selected]['bindings'][1]
        original = self.standard.BINDINGS[identity]
        for field in ('s', 'rcond', 'rwork', 'work', 'iwork'):
            parameters = tuple((name, 'scalar' if typ != 'scalar' else 'real', role, depth, const) if name == field else (name, typ, role, depth, const)
                               for name, typ, role, depth, const in original['parameters'])
            with self.subTest(field=field), self.changed_binding(index=1, set_id=selected, parameters=parameters), \
                 mock.patch.object(self.generator.subprocess, 'run') as formatter:
                with self.assertRaisesRegex(self.schema.ProfileError, 'Parameter order/role/type/pointer/const mismatch'):
                    self.generator.subset_outputs(self.PROFILE, selected, self.TARGET, 'fixture-formatter')
                formatter.assert_not_called()
        parameters = (*original['parameters'][:-2], original['parameters'][-1], original['parameters'][-2])
        with self.changed_binding(index=1, set_id=selected, parameters=parameters), \
             mock.patch.object(self.generator.subprocess, 'run') as formatter:
            with self.assertRaises(self.schema.ProfileError):
                self.generator.subset_outputs(self.PROFILE, selected, self.TARGET, 'fixture-formatter')
            formatter.assert_not_called()

    def test_gesv_signature_preserves_mutable_factor_and_pivots_without_transpose(self):
        for width in ('lp64', 'ilp64'):
            records = [self.standard.BINDINGS['gesv-' + width + '-' + str(i)] for i in range(8)]
            self.assertEqual({'LAPACKE_' + scalar + 'gesv' + suffix
                              for scalar in 'sdcz' for suffix in ('', '_work')},
                             {record['symbol'] for record in records})
            for record in records:
                self.assertEqual(['matrix_layout', 'n', 'nrhs', 'a', 'lda', 'ipiv', 'b', 'ldb'],
                                 [parameter[0] for parameter in record['parameters']])
                for parameter in record['parameters']:
                    if parameter[0] in ('a', 'ipiv', 'b'):
                        self.assertEqual((1, False), parameter[3:])
            registry = {key: dict(value) for key, value in self.standard.BINDINGS.items()}
            key = 'gesv-' + width + '-0'
            parameters = list(registry[key]['parameters'])
            parameters[5] = (*parameters[5][:4], True)
            registry[key]['parameters'] = tuple(parameters)
            with mock.patch.object(self.standard, 'BINDINGS', registry), \
                 mock.patch.object(self.generator.subprocess, 'run') as formatter:
                with self.assertRaises(self.schema.ProfileError):
                    self.generator.subset_outputs('mkl2026.' + width,
                        'potrf-potrs-getrf-getrs-gesv-ordinary-' + width + '-v1',
                        self.TARGET, 'fixture-formatter')
                formatter.assert_not_called()

    def test_empty_acceptance_rejects_all_standard_generation_before_side_effects(self):
        self.assertEqual(self.production_acceptances, dict(self.standard.GENERATION_ACCEPTANCES))
        with mock.patch.object(self.standard, 'GENERATION_ACCEPTANCES', {}), \
             mock.patch.object(self.generator.subprocess, 'run', side_effect=AssertionError('formatter reached')) as formatter, \
             mock.patch.object(Path, 'mkdir', side_effect=AssertionError('mkdir reached')) as mkdir, \
             mock.patch.object(Path, 'write_text', side_effect=AssertionError('write reached')) as write_text, \
             mock.patch.object(Path, 'write_bytes', side_effect=AssertionError('write reached')) as write_bytes, \
             mock.patch.dict(os.environ, {'ZYNUM_ACCEPT_BINDINGS': '*', 'ZYNUM_FORCE_ABI': '1'}):
            for profile, set_id in ((self.PROFILE, self.SET), ('mkl2026.ilp64', 'potrf-potrs-ordinary-ilp64-v1'),
                                    (self.PROFILE, self.LU_SET), ('mkl2026.ilp64', 'potrf-potrs-getrf-getrs-ordinary-ilp64-v1'),
                                    (self.PROFILE, 'potrf-potrs-getrf-getrs-gesv-ordinary-lp64-v1'),
                                    ('mkl2026.ilp64', 'potrf-potrs-getrf-getrs-gesv-ordinary-ilp64-v1')):
                with self.subTest(profile=profile):
                    with self.assertRaisesRegex(self.schema.ProfileError, 'No callable generation acceptance'):
                        self.generator.subset_outputs(profile, set_id, self.TARGET, 'fixture-formatter')
                    for check in ([], ['--check']):
                        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
                            self.cli(['--profile', profile, '--binding-set', set_id, '--target-contract', self.TARGET, *check])
                        self.assertEqual(2, error.exception.code)
                    with self.assertRaisesRegex(self.schema.ProfileError, 'not ready'):
                        self.schema.select_profile(profile)
            for observer in (formatter, mkdir, write_text, write_bytes):
                observer.assert_not_called()

    def test_cli_closed_selection_and_checked_dispatch_preserved(self):
        cases = [(['--binding-set', self.SET], 'both --binding-set and --target-contract'),
                 (['--target-contract', self.TARGET], 'both --binding-set and --target-contract'),
                 (['--force'], 'unrecognized arguments'),
                 (['--profile', self.PROFILE], 'not ready'),
                 (['--profile', self.PROFILE, '--binding-set', self.SET, '--target-contract', 'unreviewed'], 'Unreviewed target')]
        for arguments, message in cases:
            with self.subTest(arguments=arguments), contextlib.redirect_stderr(io.StringIO()) as error_text, \
                 mock.patch.object(self.generator.subprocess, 'run', side_effect=AssertionError('formatter reached')) as formatter, \
                 self.assertRaises(SystemExit) as error:
                self.cli(arguments)
            self.assertEqual(2, error.exception.code)
            self.assertIn(message, error_text.getvalue())
            formatter.assert_not_called()
        for arguments in ([], ['--profile', 'checked-v1']):
            with mock.patch.object(self.generator, 'outputs', return_value={}) as checked, \
                 mock.patch.object(self.generator, 'subset_outputs', side_effect=AssertionError('wrong branch')) as finite, \
                 contextlib.redirect_stdout(io.StringIO()) as output:
                self.cli(arguments)
            checked.assert_called_once_with('zig', 'checked-v1')
            finite.assert_not_called()
            self.assertEqual('Generated 6 checked ABI outputs, 8 functions\n', output.getvalue())

    def test_local_exact_plan_is_immutable_with_independent_width_and_const_roles(self):
        for profile, set_id, integer, wire, internal in (
                (self.PROFILE, self.SET, 'int', 'c_int', 'i32'),
                ('mkl2026.ilp64', 'potrf-potrs-ordinary-ilp64-v1', 'long long int', 'c_longlong', 'i64')):
            with self.subTest(profile=profile), self.fixture(profile, set_id):
                plan = self.select(profile, set_id)
                self.assertEqual(16, len(plan['bindings']))
                self.assertEqual({'c_type': integer, 'zig_type': wire, 'internal_type': internal}, plan['bindings'][0]['wire_return'])
                self.assertFalse(plan['full_profile_ready'])
                self.assertEqual('pending', plan['runtime_abi_status'])
                self.assertEqual('pending', plan['build_owner_identity_status'])
                for binding in plan['bindings']:
                    parameters = binding['wire_parameters']
                    self.assertEqual(('int', 'c_int', 'i32'), tuple(parameters[0][k] for k in ('c_type', 'zig_type', 'internal_type')))
                    self.assertEqual('char', parameters[1]['c_type'])
                    if binding['operation'] == 'cholesky.solve':
                        self.assertEqual(('matrix_layout', 'uplo', 'n', 'nrhs', 'a', 'lda', 'b', 'ldb'), tuple(p['name'] for p in parameters))
                        self.assertIs(parameters[4]['pointee_const'], True)
                        self.assertIs(parameters[6]['pointee_const'], False)
                with self.assertRaises(TypeError):
                    plan['profile'] = 'other'
                with self.assertRaises(TypeError):
                    plan['bindings'][0]['wire_parameters'][0]['c_type'] = 'long long int'
                with self.assertRaises(TypeError):
                    plan['owner_imports'][0]['instantiate'] = True

    def test_membership_opaque_ids_duplicates_and_symbol_collisions_rejected(self):
        for args, message in (((self.PROFILE, 'missing', self.TARGET), 'Unknown binding set'),
                              (('checked-v1', self.SET, self.TARGET), 'Unknown standard profile'),
                              (('mkl2026.ilp64', self.SET, self.TARGET), 'different profile')):
            with self.subTest(args=args), self.assertRaisesRegex(self.schema.ProfileError, message):
                self.select(*args)
        identities = self.standard.SETS[self.SET]['bindings']
        for values, message in (((), 'Empty'), (('missing', *identities[1:]), 'Unknown/empty'),
                                ((identities[0], *identities[:-1]), 'Duplicate')):
            sets = {**self.standard.SETS, self.SET: {**self.standard.SETS[self.SET], 'bindings': values}}
            with mock.patch.object(self.standard, 'SETS', sets), self.assertRaisesRegex(self.schema.ProfileError, message):
                self.select()
        with self.changed_binding(1, symbol=self.standard.BINDINGS[identities[0]]['symbol']), \
             self.assertRaisesRegex(self.schema.ProfileError, 'Symbol collision'):
            self.select()
        with self.changed_binding(symbol=self.schema.PREFIX + self.schema.FUNCTIONS[0][0]), \
             self.assertRaisesRegex(self.schema.ProfileError, 'existing API/support'):
            self.select()
        renamed = 'opaque-id-without-width'
        bindings = dict(self.standard.BINDINGS)
        bindings[renamed] = {**bindings.pop(identities[0]), 'id': renamed}
        sets = {**self.standard.SETS, self.SET: {**self.standard.SETS[self.SET], 'bindings': (renamed, *identities[1:])}}
        with mock.patch.object(self.standard, 'BINDINGS', bindings), mock.patch.object(self.standard, 'SETS', sets):
            with self.assertRaisesRegex(self.schema.ProfileError, 'No callable generation acceptance'):
                self.select()
            with self.fixture():
                self.assertEqual(self.PROFILE, self.select()['bindings'][0]['profile'])

    def test_parameters_owner_target_and_self_authorization_mutations_rejected(self):
        original = self.standard.BINDINGS[self.standard.SETS[self.SET]['bindings'][2]]['parameters']
        variants = (original[::-1],
                    tuple((n, t, r, d, False) if n == 'a' else (n, t, r, d, c) for n, t, r, d, c in original),
                    tuple((n, t, r, 2, c) if n == 'a' else (n, t, r, d, c) for n, t, r, d, c in original),
                    tuple((n, t, 'matrix', d, c) if n == 'a' else (n, t, r, d, c) for n, t, r, d, c in original),
                    tuple((n, t, r, True, c) if n == 'a' else (n, t, r, d, c) for n, t, r, d, c in original))
        for parameters in variants:
            with self.subTest(parameters=parameters), self.changed_binding(2, parameters=parameters), \
                 self.assertRaisesRegex(self.schema.ProfileError, 'Parameter order/role/type/pointer/const'):
                self.select()
        for updates, message in (({'owner': '../new-owner.zig'}, 'call recipe'),
                                 ({'recipe': 'arbitrary_source()'}, 'call recipe'),
                                 ({'integer_family': 'explicit64'}, 'Explicit64'),
                                 ({'declaration': ('invented.h',)}, 'Declaration'),
                                 ({'semantic_contracts': ()}, 'Semantic/raw-fault/hook'),
                                 ({'generation_acceptance': True}, 'self-authorize')):
            with self.subTest(updates=updates), self.changed_binding(**updates), \
                 self.assertRaisesRegex(self.schema.ProfileError, message):
                self.select()
        with mock.patch.object(self.standard, 'TARGET', {**self.standard.TARGET, 'arch': 'x86_64'}), \
             self.assertRaisesRegex(self.schema.ProfileError, 'representation or visibility'):
            self.select()
        with mock.patch.object(self.standard, 'OWNER', {**self.standard.OWNER, 'instantiate': True}), \
             self.assertRaisesRegex(self.schema.ProfileError, 'Owner recipe drift'):
            self.select()
        profiles = {**self.schema.PROFILES, self.PROFILE: {**self.schema.PROFILES[self.PROFILE], 'generation_ready': True}}
        with mock.patch.object(self.schema, 'PROFILES', profiles), self.assertRaisesRegex(self.schema.ProfileError, 'must not enable'):
            self.select()

    def test_acceptance_and_source_changes_fail_closed_and_fixture_is_restored(self):
        with self.fixture() as (root, accepted):
            key = (self.PROFILE, self.SET, self.TARGET)
            for update, message in (({'profile': 'mkl2026.ilp64'}, 'selection mismatch'),
                                    ({'scope': 'full-profile-ready'}, 'scope'),
                                    ({'plan_sha256': '0' * 64}, 'complete selected plan'),
                                    ({'source_envelope_sha256': '0' * 64}, 'complete selected plan'),
                                    ({'review_sha256': ''}, 'review digest'), ({'id': '*'}, 'identity')):
                with self.subTest(update=update), mock.patch.object(self.standard, 'GENERATION_ACCEPTANCES', {key: {**accepted, **update}}), \
                     self.assertRaisesRegex(self.schema.ProfileError, message):
                    self.select()
            with mock.patch.object(self.standard, 'SOURCE_DEPENDENCIES', self.standard.SOURCE_DEPENDENCIES[:-1]), \
                 self.assertRaisesRegex(self.schema.ProfileError, 'Source envelope changed'):
                self.select()
            with mock.patch.object(self.standard, 'EXTERNAL_MODULE_CONTRACTS', (('root', 'replacement-owner'),)), \
                 self.assertRaisesRegex(self.schema.ProfileError, 'complete selected plan'):
                self.select()
            victim = root / 'src/fixture_factor.zig'
            victim.write_bytes(b'intentional local fixture drift')
            with self.assertRaisesRegex(self.schema.ProfileError, 'dependency missing or changed'):
                self.select()
            victim.unlink()
            with self.assertRaisesRegex(self.schema.ProfileError, 'dependency missing or changed'):
                self.select()
        with self.assertRaisesRegex(RuntimeError, 'fixture unwinding'):
            with self.fixture():
                raise RuntimeError('fixture unwinding')
        self.assertEqual(self.production_acceptances, dict(self.standard.GENERATION_ACCEPTANCES))

    def test_renderer_two_widths_have_exact_declarations_exports_and_limits(self):
        scalars = (('s', 'float'), ('d', 'double'), ('c', 'lapack_complex_float'), ('z', 'lapack_complex_double'))
        for profile, set_id, integer in ((self.PROFILE, self.SET, 'int'),
                ('mkl2026.ilp64', 'potrf-potrs-ordinary-ilp64-v1', 'long long int')):
            with self.subTest(profile=profile), self.fixture(profile, set_id), \
                 mock.patch.object(self.generator.subprocess, 'run', side_effect=self.fake_formatter) as formatter:
                result = self.generator.subset_outputs(profile, set_id, self.TARGET, 'fixture-formatter')
            formatter.assert_called_once()
            prefix = f'interface_subset/{profile}/{set_id}/'
            self.assertEqual({prefix + name for name in ('subset.h', 'exports.zig', 'abi_manifest.json', 'exports.def')}, set(result))
            header = result[prefix + 'subset.h']
            expected = []
            for letter, c_type in scalars:
                for suffix in ('', '_work'):
                    expected.append(f'{integer} LAPACKE_{letter}potrf{suffix}(int matrix_layout, char uplo, {integer} n, {c_type}* a, {integer} lda);')
                    expected.append(f'{integer} LAPACKE_{letter}potrs{suffix}(int matrix_layout, char uplo, {integer} n, {integer} nrhs, const {c_type}* a, {integer} lda, {c_type}* b, {integer} ldb);')
            self.assertEqual(sorted(expected), sorted(line for line in header.splitlines() if ' LAPACKE_' in line))
            names = {line.split(' LAPACKE_', 1)[1].split('(', 1)[0] for line in expected}
            source = result[prefix + 'exports.zig']
            self.assertEqual(16, source.count('@export('))
            for name in names:
                self.assertIn('.name = "LAPACKE_' + name + '"', source)
            self.assertIn('@import("lapacke-interface-owner")', source)
            self.assertNotIn('wire-faults', source)
            manifest = json.loads(result[prefix + 'abi_manifest.json'])
            for flag in ('public_api_ready', 'full_vendor_profile_ready', 'standard_xerbla_compatibility'):
                self.assertIs(manifest[flag], False)
            self.assertEqual('pending', manifest['this_generated_artifact_runtime_status'])
            self.assertEqual('mkl_types.h', manifest['type_declaration_reference']['path'])
            self.assertEqual({'LAPACKE_' + name for name in names},
                             {line.strip() for line in result[prefix + 'exports.def'].splitlines() if line.startswith('    LAPACKE_')})

    def test_renderer_bad_plan_and_formatter_failure_leave_no_output(self):
        with self.fixture(), self.changed_binding(recipe='arbitrary_source()'), \
             mock.patch.object(self.generator.subprocess, 'run', side_effect=AssertionError('formatter reached')) as formatter, \
             self.assertRaisesRegex(self.schema.ProfileError, 'call recipe'):
            self.generator.subset_outputs(self.PROFILE, self.SET, self.TARGET, 'fixture-formatter')
        formatter.assert_not_called()
        with self.fixture() as (root, _), \
             mock.patch.object(self.generator.subprocess, 'run', side_effect=subprocess.CalledProcessError(1, ['fixture-formatter'])):
            output = root / 'must-not-exist'
            with self.assertRaises(subprocess.CalledProcessError):
                self.cli(['--profile', self.PROFILE, '--binding-set', self.SET, '--target-contract', self.TARGET,
                          '--zig', 'fixture-formatter', '--root', str(output)])
            self.assertFalse(output.exists())

    def test_local_cli_writes_four_files_and_readonly_check_detects_drift(self):
        with self.fixture() as (root, _), mock.patch.object(self.generator.subprocess, 'run', side_effect=self.fake_formatter), \
             contextlib.redirect_stdout(io.StringIO()):
            output = root / 'outputs'
            arguments = ['--profile', self.PROFILE, '--binding-set', self.SET, '--target-contract', self.TARGET,
                         '--zig', 'fixture-formatter', '--root', str(output)]
            self.cli(arguments)
            files = sorted(p for p in output.rglob('*') if p.is_file())
            self.assertEqual(4, len(files))
            before = {str(p): p.read_bytes() for p in files}
            self.cli([*arguments, '--check'])
            self.assertEqual(before, {str(p): p.read_bytes() for p in files})
            header = next(p for p in files if p.name == 'subset.h')
            header.write_text('intentional fixture output drift')
            with self.assertRaisesRegex(SystemExit, 'Generated drift'):
                self.cli([*arguments, '--check'])
            self.assertEqual('intentional fixture output drift', header.read_text())
        self.assertEqual(self.production_acceptances, dict(self.standard.GENERATION_ACCEPTANCES))

    def test_lu_combined_sets_preserve_old_members_and_exact_integer_pointer_roles(self):
        for width, integer, wire, internal in (('lp64', 'int', 'c_int', 'i32'),
                                              ('ilp64', 'long long int', 'c_longlong', 'i64')):
            profile = 'mkl2026.' + width
            old = 'potrf-potrs-ordinary-' + width + '-v1'
            new = 'potrf-potrs-getrf-getrs-ordinary-' + width + '-v1'
            self.assertEqual(self.standard.SETS[old]['bindings'], self.standard.SETS[new]['bindings'][:16])
            with self.fixture(profile, new):
                plan = self.select(profile, new)
            self.assertEqual(32, len(plan['bindings']))
            self.assertEqual(32, len(set(plan['symbols'])))
            self.assertEqual({'lu.factor', 'lu.solve'}, {r['operation'] for r in plan['bindings'][16:]})
            self.assertFalse(plan['full_profile_ready'])
            self.assertEqual('pending', plan['runtime_abi_status'])
            for row in plan['bindings'][16:]:
                parameters = {p['name']: p for p in row['wire_parameters']}
                solve = row['operation'] == 'lu.solve'
                names = ('matrix_layout', 'trans', 'n', 'nrhs', 'a', 'lda', 'ipiv', 'b', 'ldb') if solve else (
                    'matrix_layout', 'm', 'n', 'a', 'lda', 'ipiv')
                self.assertEqual(names, tuple(parameters))
                self.assertEqual('raw_getrs_profile_defaults' if solve else 'raw_getrf_profile_defaults', row['recipe'])
                self.assertEqual((integer, wire, internal), tuple(row['wire_return'].values()))
                self.assertEqual(('int', 'c_int', 'i32'), tuple(parameters['matrix_layout'][k] for k in ('c_type', 'zig_type', 'internal_type')))
                for name in ('ipiv', 'n', 'lda'):
                    self.assertEqual((integer, wire, internal), tuple(parameters[name][k] for k in ('c_type', 'zig_type', 'internal_type')))
                self.assertEqual(1, parameters['ipiv']['pointer_depth'])
                self.assertIs(parameters['ipiv']['pointee_const'], solve)
                self.assertIs(parameters['a']['pointee_const'], solve)
                if solve:
                    self.assertEqual(('char', 'i8', 'u8-bitcast'), tuple(parameters['trans'][k] for k in ('c_type', 'zig_type', 'internal_type')))
                    self.assertIs(parameters['b']['pointee_const'], False)

    def test_lu_parameter_recipe_and_explicit64_mutations_fail_before_formatter(self):
        # GETRF high is member16; GETRS high is member18 in the closed append-only set.
        for index in (16, 18):
            row = self.standard.BINDINGS[self.standard.SETS[self.LU_SET]['bindings'][index]]
            parameters = row['parameters']
            mutations = [parameters[::-1]]
            for name, field, value in (('ipiv', 1, 'c_int'), ('ipiv', 3, 0), ('ipiv', 3, True),
                                       ('ipiv', 4, not (index == 18)), ('ipiv', 2, 'matrix'),
                                       ('matrix_layout', 1, 'lapack_int')):
                mutations.append(tuple(tuple(value if n == name and i == field else part for i, part in enumerate(p))
                                       for p in parameters for n in (p[0],)))
            if index == 18:
                mutations.append(tuple((n, 'lapack_int', r, d, c) if n == 'trans' else (n, t, r, d, c)
                                       for n, t, r, d, c in parameters))
            cases = [({'parameters': variant}, 'Parameter order/role/type/pointer/const') for variant in mutations]
            cases += [({'recipe': 'raw_factor_profile_defaults' if index == 16 else 'raw_solve_profile_defaults'}, 'call recipe'),
                      ({'recipe': 'raw_getrs_profile_defaults' if index == 16 else 'raw_getrf_profile_defaults'}, 'call recipe'),
                      ({'integer_family': 'explicit64'}, 'Explicit64'),
                      ({'owner': '../second-owner.zig'}, 'call recipe'),
                      ({'semantic_contracts': self.standard.SEMANTIC_CONTRACTS}, 'Semantic/raw-fault/hook'),
                      ({'operands': row['operands'][:-1]}, 'operand coverage'),
                      ({'declaration': self.standard.BINDINGS[self.standard.SETS[self.SET]['bindings'][0]]['declaration']}, 'Declaration')]
            for update, message in cases:
                with self.subTest(index=index, update=update), self.fixture(self.PROFILE, self.LU_SET) as (root, _), \
                     self.changed_binding(index, set_id=self.LU_SET, **update), \
                     mock.patch.object(self.generator.subprocess, 'run', side_effect=AssertionError('formatter reached')) as formatter:
                    destination = root / 'must-not-exist'
                    with contextlib.redirect_stderr(io.StringIO()) as errors, self.assertRaises(SystemExit) as result:
                        self.cli(['--profile', self.PROFILE, '--binding-set', self.LU_SET,
                                  '--target-contract', self.TARGET, '--root', str(destination)])
                    self.assertEqual(2, result.exception.code)
                    self.assertIn(message, errors.getvalue())
                    formatter.assert_not_called()
                    self.assertFalse(destination.exists())
        for operation in ('lu.factor', 'lu.solve'):
            changed = {**self.schema.OPERATIONS, operation: {**self.schema.OPERATIONS[operation], 'index_types': ()}}
            with self.fixture(self.PROFILE, self.LU_SET), mock.patch.object(self.schema, 'OPERATIONS', changed), \
                 mock.patch.object(self.generator.subprocess, 'run', side_effect=AssertionError('formatter reached')) as formatter, \
                 self.assertRaisesRegex(self.schema.ProfileError, 'Unsupported pivot index binding'):
                self.generator.subset_outputs(self.PROFILE, self.LU_SET, self.TARGET, 'fixture-formatter')
            formatter.assert_not_called()

    def test_lu_membership_and_opaque_id_cannot_hide_missing_or_cross_width_bindings(self):
        ids = self.standard.SETS[self.LU_SET]['bindings']
        other = self.standard.SETS['potrf-potrs-getrf-getrs-ordinary-ilp64-v1']['bindings']
        for values, message in ((ids[:-1], 'exact ordinary binding count'),
                                ((*ids[:16], *ids[18:20], *ids[16:18], *ids[20:]), 'reordered finite'),
                                ((*ids[:16], *other[16:]), 'different selected profile'),
                                ((*ids, *other), 'Symbol collision')):
            changed = {**self.standard.SETS, self.LU_SET: {**self.standard.SETS[self.LU_SET], 'bindings': values}}
            with mock.patch.object(self.standard, 'SETS', changed), self.assertRaisesRegex(self.schema.ProfileError, message):
                self.select(self.PROFILE, self.LU_SET)
        renamed = 'opaque-lu-record-not-a-profile'
        bindings = dict(self.standard.BINDINGS)
        bindings[renamed] = {**bindings.pop(ids[18]), 'id': renamed}
        sets = {**self.standard.SETS, self.LU_SET: {**self.standard.SETS[self.LU_SET],
                                                'bindings': (*ids[:18], renamed, *ids[19:])}}
        with mock.patch.object(self.standard, 'BINDINGS', bindings), mock.patch.object(self.standard, 'SETS', sets):
            with self.assertRaisesRegex(self.schema.ProfileError, 'Acceptance does not bind'):
                self.select(self.PROFILE, self.LU_SET)
            with self.fixture(self.PROFILE, self.LU_SET):
                self.assertEqual(self.PROFILE, self.select(self.PROFILE, self.LU_SET)['bindings'][18]['profile'])
            bindings[renamed] = {**bindings[renamed], 'profile': 'mkl2026.ilp64'}
            with self.assertRaisesRegex(self.schema.ProfileError, 'different selected profile'):
                self.select(self.PROFILE, self.LU_SET)

    def test_lu_renderer_exact_official_parameter_shapes_and_same_owner_templates(self):
        import re
        for width, integer in (('lp64', 'int'), ('ilp64', 'long long int')):
            profile, set_id = 'mkl2026.' + width, 'potrf-potrs-getrf-getrs-ordinary-' + width + '-v1'
            with self.fixture(profile, set_id), mock.patch.object(self.generator.subprocess, 'run', side_effect=self.fake_formatter) as formatter:
                result = self.generator.subset_outputs(profile, set_id, self.TARGET, 'fixture-formatter')
            formatter.assert_called_once()
            prefix = f'interface_subset/{profile}/{set_id}/'
            header, source = result[prefix + 'subset.h'], result[prefix + 'exports.zig']
            expected = []
            # Independent official C signature shapes; no schema parameter-table reuse.
            for letter, scalar in (('s', 'float'), ('d', 'double'), ('c', 'lapack_complex_float'), ('z', 'lapack_complex_double')):
                for suffix in ('', '_work'):
                    expected += [f'{integer} LAPACKE_{letter}getrf{suffix}(int matrix_layout, {integer} m, {integer} n, {scalar}* a, {integer} lda, {integer}* ipiv);',
                                 f'{integer} LAPACKE_{letter}getrs{suffix}(int matrix_layout, char trans, {integer} n, {integer} nrhs, const {scalar}* a, {integer} lda, const {integer}* ipiv, {scalar}* b, {integer} ldb);']
            actual = [line for line in header.splitlines() if re.search(r'LAPACKE_[sdcz]getr[fs]', line)]
            self.assertEqual(sorted(expected), sorted(actual))
            self.assertEqual(32, source.count('@export('))
            self.assertEqual(1, source.count('@import("lapacke-interface-owner")'))
            self.assertEqual(1, source.count('binding.Shared.getrfRawManagedWithDefaults(true, T, InternalInt, level,'))
            self.assertEqual(1, source.count('binding.Shared.getrsRawManagedWithDefaults(true, T, InternalInt, level,'))
            for forbidden in ('@alignCast', 'Context.init', 'context.begin', 'setCallbacks(', '@import("../', 'pub fn main'):
                self.assertNotIn(forbidden, source)
            template = self.generator._LU_ZIG_TEMPLATE
            factor = template[template.index('fn LuFactor('):template.index('fn LuSolve(')]
            solve = template[template.index('fn LuSolve('):]
            common = ['.layout = @as(c_int, @intCast(normalized_layout)),',
                      '.n = @as(InternalInt, @intCast(n)),', '.lda = @as(InternalInt, @intCast(lda)),']
            args = lambda body: [line.strip() for line in body.splitlines() if line.strip().startswith('.')]
            self.assertEqual([common[0], '.m = @as(InternalInt, @intCast(m)),', common[1], '.values = a,', common[2],
                              '.pivots = if (ipiv) |pointer| @as([*]align(1) InternalInt, @ptrCast(pointer)) else null,'], args(factor))
            self.assertEqual([common[0], '.trans = @bitCast(trans),', common[1], '.nrhs = @as(InternalInt, @intCast(nrhs)),',
                              '.a = a,', common[2], '.pivots = if (ipiv) |pointer| @as([*]align(1) const InternalInt, @ptrCast(pointer)) else null,',
                              '.b = b,', '.ldb = @as(InternalInt, @intCast(ldb)),'], args(solve))
            for body in (factor, solve):
                self.assertIn('}, profileDefaults()) catch |err| terminal(@errorName(err));\n            return finish(result);', body)
                self.assertNotIn('for (', body)
                self.assertNotIn('while (', body)
            self.assertIn('ipiv: ?[*]align(1) WireInt', factor)
            self.assertIn('ipiv: ?[*]align(1) const WireInt', solve)
            self.assertIn('@alignOf(WireInt) != @alignOf(InternalInt)', template)
            manifest = json.loads(result[prefix + 'abi_manifest.json'])
            self.assertEqual(integer, manifest['integer_model']['pivot_element_c_type'])
            self.assertEqual('char', manifest['integer_model']['transpose_c_type'])
            self.assertFalse(manifest['full_vendor_profile_ready'])
            self.assertFalse(manifest['standard_xerbla_compatibility'])
            self.assertEqual('pending', manifest['this_generated_artifact_runtime_status'])

    def test_lu_cli_counts_and_failed_formatter_do_not_publish_partial_outputs(self):
        with self.fixture(self.PROFILE, self.LU_SET) as (root, _):
            destination = root / 'outputs'
            arguments = ['--profile', self.PROFILE, '--binding-set', self.LU_SET, '--target-contract', self.TARGET,
                         '--root', str(destination), '--zig', 'fixture-formatter']
            with mock.patch.object(self.generator.subprocess, 'run', side_effect=subprocess.CalledProcessError(1, ['fixture-formatter'])), \
                 self.assertRaises(subprocess.CalledProcessError):
                self.cli(arguments)
            self.assertFalse(destination.exists())
            with mock.patch.object(self.generator.subprocess, 'run', side_effect=self.fake_formatter) as formatter, \
                 contextlib.redirect_stdout(io.StringIO()) as output:
                self.cli(arguments)
                self.cli([*arguments, '--check'])
            self.assertEqual(2, formatter.call_count)
            self.assertEqual(['Generated 4 finite interface outputs, 32 functions; full profile not ready',
                              'Checked 4 finite interface outputs, 32 functions; full profile not ready'], output.getvalue().splitlines())
            self.assertEqual(4, sum(path.is_file() for path in destination.rglob('*')))

    def test_lu_source_dependencies_include_both_engines_and_actual_shared_owner_bytes(self):
        # Source identity only: this is neither compiler closure nor ABI evidence.
        dependencies = dict(self.standard.SOURCE_DEPENDENCIES)
        for path in ('src/interop/lapacke_getrf.zig', 'src/interop/lapacke_getrs.zig',
                     'src/interop/lapacke_binding.zig', 'src/interop/lapacke_services.zig',
                     'src/lapack/core/lu.zig', 'src/numerics/pivot.zig', 'src/zynum.zig',
                     'src/numerics/execution/default.zig', 'src/numerics/execution/context.zig',
                     'src/numerics/resources.zig', 'src/sparse/core/product.zig'):
            self.assertIn(path, dependencies)
        for path, expected in dependencies.items():
            self.assertEqual(expected, hashlib.sha256((REPOSITORY_ROOT / path).read_bytes()).hexdigest(), path)
        self.assertEqual(self.production_acceptances, dict(self.standard.GENERATION_ACCEPTANCES))
        for profile in ('mkl2026.lp64', 'mkl2026.ilp64'):
            self.assertIs(self.schema.PROFILES[profile]['generation_ready'], False)

    def test_checked_six_frozen_files_and_shared_cleanup_template_contract(self):
        expected = {
            'include/zynum/checked/checked.h': '6d6ca6c9259d3b56612c1ed741c1b9deab3d3881f4f7147c1675107cc1a94c54',
            'include/zynum/checked/checked.f90': '3c5dd455a1f748556c31c7cc380ee0d43f2af6bd4b7fa80977a0055b4734c7c5',
            'include/zynum/checked/abi_manifest.json': '96355c7acf43925023771e33c20af42838a6d9b7662556f8dd1b333a69104755',
            'src/checked/abi_types.zig': 'd80b207134bc498e83f8bafa35b3024d810a81c07e46897a1218fa6e2b7adec7',
            'src/checked/exports.zig': '4e3ec3065541011d925088cd86be0ba9e1448756eba407aec361fd5e765d48b3',
            'src/checked/exports.def': '90a5a4f2d7fb60c4999fc5e259698b754aaece465040152f1c17e934d68fb1a3',
        }
        self.assertEqual(set(CHECKER.CHECKED_ABI_OUTPUTS), set(expected))
        for relative, digest in expected.items():
            self.assertEqual(digest, hashlib.sha256((REPOSITORY_ROOT / relative).read_bytes()).hexdigest(), relative)
        selected = self.schema.select_profile()
        self.assertEqual(selected, self.schema.select_profile('checked-v1'))
        for name in self.schema.LEGACY_FIELDS:
            self.assertEqual(selected[name], getattr(self.schema, name))
        template = self.generator._SUBSET_ZIG_TEMPLATE
        terminal = template[template.index('fn terminal('):template.index('fn finish(')]
        self.assertIn('std.c.abort();', terminal)
        self.assertNotIn('observeReturned', terminal)
        self.assertEqual(3, template.count('binding.Provider.Execution.getContext()'))
        self.assertNotIn('Context.init', template)
        self.assertNotIn('setCallbacks(', template)
        self.assertEqual(2, template.count('}, profileDefaults()) catch'))
        self.assertNotIn('allocator', template)
        self.assertIn('scratch defers and Operation.end have already completed', terminal)



class SparseInterfaceGenerationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with mock.patch.object(sys, 'path', [str(REPOSITORY_ROOT / 'tools'), *sys.path]), mock.patch.object(sys, 'dont_write_bytecode', True):
            import checked_abi_schema
            import standard_sparse_bindings
            import standard_abi_bindings
            import generate_sparse_abi
            import generate_checked_abi
        cls.schema = checked_abi_schema
        cls.sparse = standard_sparse_bindings
        cls.generator = generate_checked_abi
        cls.variants = tuple((width, solved, 'sparse-csr-csc-mv-mm-' + ('trsv-trsm-' if solved else '') + 'ordinary-' + width + '-v1') for width in ('lp64', 'ilp64') for solved in (False, True))

    def test_exact_acceptances_remain_finite(self):
        self.assertEqual({('mkl2026.' + width, selected, self.sparse.TARGET_ID) for width, _, selected in self.variants}, set(self.sparse.GENERATION_ACCEPTANCES))
        for width, solved, selected in self.variants:
            plan = self.schema.select_binding_set('mkl2026.' + width, selected, self.sparse.TARGET_ID)
            self.assertEqual(33 if solved else 25, len(plan['symbols']))
            self.assertEqual('finite-callable-generation-only', plan['generation_acceptance']['scope'])
            self.assertFalse(plan['full_profile_ready'])

    def test_changed_acceptance_plan_is_rejected(self):
        key = ('mkl2026.lp64', 'sparse-csr-csc-mv-mm-ordinary-lp64-v1', self.sparse.TARGET_ID)
        for width, _, selected in self.variants:
            key = ('mkl2026.' + width, selected, self.sparse.TARGET_ID)
            records = dict(self.sparse.GENERATION_ACCEPTANCES)
            records[key] = {**records[key], 'plan_sha256': '0' * 64}
            with mock.patch.object(self.sparse, 'GENERATION_ACCEPTANCES', records):
                with self.assertRaisesRegex(self.schema.ProfileError, 'complete selected plan'):
                    self.schema.select_binding_set(*key)

    def test_exact_both_widths_and_shared_owner(self):
        expected = {'mkl_sparse_destroy'} | {'mkl_sparse_' + scalar + '_' + operation
            for scalar in 'sdcz' for operation in ('create_csr', 'create_csc', 'export_csr', 'export_csc', 'mv', 'mm')}
        for width, solved, selected in self.variants:
            plan = self.schema._binding_candidate('mkl2026.' + width, selected, self.sparse.TARGET_ID)
            solve_symbols = {'mkl_sparse_' + scalar + '_' + operation for scalar in 'sdcz' for operation in ('trsv', 'trsm')} if solved else set()
            self.assertEqual(expected | solve_symbols, set(plan['symbols']))
            self.assertEqual(4 if width == 'lp64' else 8, plan['integer_model'][3])
            if solved:
                self.assertIn('ordinary-' + width + '-csr-csc-triangular-diagonal-vector-panel-solve-v1', plan['semantic_contracts'])
            self.assertFalse(plan['full_profile_ready'])
            self.assertEqual('mkl_sparse_binding_mod', plan['owner_imports'][0]['module_local'])
            self.assertFalse(plan['owner_imports'][0]['instantiate'])

    def test_no_generation_or_formatter_before_acceptance(self):
        with mock.patch.object(self.sparse, 'GENERATION_ACCEPTANCES', {}), mock.patch.object(self.generator.subprocess, 'run', side_effect=AssertionError('unexpected formatter')) as run:
            for width, _, selected in self.variants:
                with self.assertRaisesRegex(self.schema.ProfileError, 'No callable generation acceptance'):
                    self.generator.subset_outputs('mkl2026.' + width, selected, self.sparse.TARGET_ID)
            run.assert_not_called()

    def test_wrong_profile_target_and_unknown_set_are_rejected(self):
        for profile, selected, target in [('mkl2026.ilp64', 'sparse-csr-csc-mv-mm-ordinary-lp64-v1', self.sparse.TARGET_ID),
                ('mkl2026.lp64', 'sparse-unknown', self.sparse.TARGET_ID),
                ('mkl2026.lp64', 'sparse-csr-csc-mv-mm-ordinary-lp64-v1', 'unknown-target')]:
            with self.assertRaises(self.schema.ProfileError):
                self.schema._binding_candidate(profile, selected, target)

    def test_changed_owner_or_source_envelope_is_rejected(self):
        args = ('mkl2026.lp64', 'sparse-csr-csc-mv-mm-ordinary-lp64-v1', self.sparse.TARGET_ID)
        with mock.patch.object(self.sparse, 'OWNER', {**self.sparse.OWNER, 'instantiate': True}):
            with self.assertRaisesRegex(self.schema.ProfileError, 'drift'):
                self.schema._binding_candidate(*args)
        with mock.patch.object(self.sparse, 'SOURCE_DEPENDENCIES', self.sparse.SOURCE_DEPENDENCIES[:-1]):
            with self.assertRaisesRegex(self.schema.ProfileError, 'source envelope drift'):
                self.schema._binding_candidate(*args)

        changed = copy.deepcopy(self.schema.OPERATIONS)
        changed['sparse.mm']['operand_roles'] = ('matrix',)
        with mock.patch.object(self.schema, 'OPERATIONS', changed):
            with self.assertRaisesRegex(self.schema.ProfileError, 'operation domain mismatch'):
                self.schema._binding_candidate(*args)

    def test_existing_lapack_selections_remain_accepted(self):
        for width in ('lp64', 'ilp64'):
            plan = self.schema.select_binding_set('mkl2026.' + width,
                'potrf-potrs-getrf-getrs-ordinary-' + width + '-v1', self.sparse.TARGET_ID)
            self.assertEqual(32, len(plan['symbols']))




class OperatorPhaseInventorySourceContractTests(unittest.TestCase):
    def test_optional_phase_selector_guard_generator_and_metadata_are_exact(self) -> None:
        source = (REPOSITORY_ROOT / "build.zig").read_text()
        CHECKER._numerics_resources_build_contract(source)
        replacements = (
            ('const operator_phase_agreement = b.option(bool,', 'const operator_phase_agreement = b.option(u32,'),
            ('"operator-phase-agreement", "Include additional generated synchronous distributed phase agreement entries") orelse false;', '"operator-phase-agreement", "Include additional generated synchronous distributed phase agreement entries") orelse true;'),
            ('if (operator_phase_agreement and operator_interface == .none)', 'if (operator_phase_agreement and operator_interface == .zynum_v1)'),
            ('if (operator_phase_agreement and operator_interface == .none) {\n        std.log.err("operator-phase-agreement requires -Doperator-interface=zynum_v1", .{});\n        b.invalid_user_input = true;\n    }', 'if (operator_phase_agreement and operator_interface == .none) {\n        std.log.err("operator-phase-agreement requires -Doperator-interface=zynum_v1", .{});\n        b.invalid_user_input = false;\n    }'),
            ('operator_generate.addArg("--with-phase-agreement");', ''),
            ('operator_generate.addFileInput(b.path("src/numerics/synchronization.zig"));', ''),
            ('if (operator_phase_agreement) "operator_phase_agreement=true\\n" else ""', '"operator_phase_agreement=true\\n"'),
        )
        for before, after in replacements:
            with self.subTest(mutation=before):
                self.assertEqual(1, source.count(before))
                with self.assertRaises(CHECKER.InventoryError):
                    CHECKER._numerics_resources_build_contract(source.replace(before, after, 1))
        with self.assertRaisesRegex(CHECKER.InventoryError, "phase selector escapes"):
            CHECKER._numerics_resources_build_contract(source + "\nconst borrowed_phase = operator_phase_agreement;\n")

    def test_phase_sources_and_finite_manifests_remain_frozen_and_classified(self) -> None:
        context = CHECKER._make_discovery_context(REPOSITORY_ROOT)
        CHECKER._numerics_resources_source_contract(context)
        original = CHECKER._frozen_regular_bytes
        self.assertEqual(27, len([p for p in CHECKER.OPERATOR_PHASE_SOURCE_SHA256 if p.startswith("interface_subset/")]))
        for path in CHECKER.OPERATOR_PHASE_SOURCE_SHA256:
            self.assertEqual(CHECKER.OPERATOR_PHASE_SOURCE_SHA256[path], hashlib.sha256((REPOSITORY_ROOT / path).read_bytes()).hexdigest())
            def drift(active: Any, rel: str, description: str, changed: str = path) -> bytes:
                value = original(active, rel, description)
                return value + b"\n" if rel == changed else value
            with self.subTest(path=path), mock.patch.object(CHECKER, "_frozen_regular_bytes", side_effect=drift):
                with self.assertRaisesRegex(CHECKER.InventoryError, "source bytes drifted"):
                    CHECKER._numerics_resources_source_contract(context)
        inventory = json.loads((REPOSITORY_ROOT / "tools/build_inventory.json").read_text())
        CHECKER._apply_reviewed_build_inventory_migrations(inventory)
        classifications = {r["path"]: r for r in inventory["repository_file_classifications"]}
        for path, owner in (("src/numerics/synchronization.zig", "library-source"), ("src/operator_phase_candidate_test.zig", "test-infrastructure")):
            self.assertEqual({"path": path, "kind": "zig-source", "owner": owner, "class": "non-generated-source"}, classifications[path])


class TrtriInventoryContractTests(unittest.TestCase):
    def test_closed_outputs_preserve_existing_roots_and_finite_runtime_scope(self) -> None:
        inventory = json.loads((REPOSITORY_ROOT / "tools/build_inventory.json").read_text())
        self.assertEqual(60, len(CHECKER.TRTRI_FINITE_SETS))
        self.assertEqual(240, len(CHECKER.TRTRI_OUTPUTS))
        self.assertEqual(247, len(CHECKER.TRTRI_CLASSIFICATIONS))
        targets = {row["path"]: row for row in inventory["generator_targets"]}
        classifications = {row["path"]: row for row in inventory["repository_file_classifications"]}
        for path in CHECKER.TRTRI_OUTPUTS:
            self.assertEqual("tools/generate_checked_abi.py", targets[path]["anchor"]["file"])
            self.assertEqual("subset_outputs", targets[path]["anchor"]["enclosing_function"])
        for row in CHECKER.TRTRI_CLASSIFICATIONS:
            self.assertEqual(row, classifications[row["path"]])
        for path, expected in CHECKER.TRTRI_SOURCE_SHA256.items():
            self.assertEqual(expected, hashlib.sha256((REPOSITORY_ROOT / path).read_bytes()).hexdigest())
        self.assertEqual("false", CHECKER.TRTRI_OPTION_SURFACE["default"])
        source = (REPOSITORY_ROOT / "build.zig").read_text()
        facts = CHECKER._numerics_resources_build_contract(source)
        self.assertEqual("lapacke_binding_test_mod", facts["imports"]["lapack_tests.root_module"]["numerical-owner"])
        self.assertEqual(128, len(facts["module_source_variants"]["lapacke_lp64_interface"]))
        observations = {row["id"]: row for row in inventory["build_observations"]}
        for width, count in (("lp64", 128), ("ilp64", 16)):
            row = observations["compile:build.zig:build:lapacke_" + width + "_static"]
            row = _prior_component_observation(self, row, width, count)
            self.assertEqual(count + 1, len(row["source_variants"]))
            self.assertEqual("src/lapacke_composed.zig", row["source_variants"]["lapacke-potri=true"])
            base_variants = {key.removeprefix("lapacke-potri=false,"): value
                             for key, value in row["source_variants"].items()
                             if key.startswith("lapacke-potri=false,")}
            self.assertEqual(count, len(base_variants))
            row = {**row, "source_variants": base_variants,
                   "root_source": [path for path in row["root_source"] if path != "src/lapacke_composed.zig"]}
            self.assertEqual(sorted(row["root_source"]), sorted(row["source_variants"].values()))
            self.assertTrue(all(",lapacke-trtri=" in key for key in row["source_variants"]))
        self.assertNotIn("lapacke_trtri_tests", facts["root_owners"])
        self.assertIsNone(CHECKER.NEXT_SOURCE_PROJECTION_SHA256)

    def test_default_type_name_and_both_width_compositions_reject_mutation(self) -> None:
        source = (REPOSITORY_ROOT / "build.zig").read_text()
        statement = '    const lapacke_trtri = b.option(bool, "lapacke-trtri", "Include generated four-scalar triangular inverse high/work in finite LP64/ILP64 and unified LP64 packages") orelse false;'
        self.assertEqual(1, source.count(statement))
        changes = [source.replace(statement, statement.replace("orelse false", "orelse true"), 1),
                   source.replace(statement, statement.replace("b.option(bool", "b.option(usize"), 1),
                   source.replace(statement, statement.replace('"lapacke-trtri"', '"unreviewed-trtri"'), 1)]
        for width in ("lp64", "ilp64"):
            before = '"{s}-trtri-ordinary-' + width + '-v1"'
            self.assertEqual(1, source.count(before))
            changes.append(source.replace(before, before.replace("-trtri-", "-unreviewed-trtri-"), 1))
        for changed in changes:
            self.assertNotEqual(source, changed)
            with self.assertRaisesRegex(CHECKER.InventoryError, "selector|selection"):
                CHECKER._numerics_resources_graph_facts(changed)


class GetriInventoryContractTests(unittest.TestCase):
    def test_closed_outputs_preserve_existing_roots_and_finite_runtime_scope(self) -> None:
        inventory = json.loads((REPOSITORY_ROOT / "tools/build_inventory.json").read_text())
        self.assertEqual(120, len(CHECKER.GETRI_FINITE_SETS))
        self.assertEqual(480, len(CHECKER.GETRI_OUTPUTS))
        self.assertEqual(487, len(CHECKER.GETRI_CLASSIFICATIONS))
        targets = {row["path"]: row for row in inventory["generator_targets"]}
        classifications = {row["path"]: row for row in inventory["repository_file_classifications"]}
        for path in CHECKER.GETRI_OUTPUTS:
            self.assertEqual("tools/generate_checked_abi.py", targets[path]["anchor"]["file"])
            self.assertEqual("subset_outputs", targets[path]["anchor"]["enclosing_function"])
        for row in CHECKER.GETRI_CLASSIFICATIONS:
            self.assertEqual(row, classifications[row["path"]])
        for path, expected in CHECKER.GETRI_SOURCE_SHA256.items():
            self.assertEqual(expected, hashlib.sha256((REPOSITORY_ROOT / path).read_bytes()).hexdigest())
        self.assertEqual("false", CHECKER.GETRI_OPTION_SURFACE["default"])
        source = (REPOSITORY_ROOT / "build.zig").read_text()
        facts = CHECKER._numerics_resources_build_contract(source)
        self.assertEqual("lapacke_binding_test_mod", facts["imports"]["lapack_tests.root_module"]["numerical-owner"])
        self.assertEqual(128, len(facts["module_source_variants"]["lapacke_lp64_interface"]))
        observations = {row["id"]: row for row in inventory["build_observations"]}
        for width, count in (("lp64", 128), ("ilp64", 16)):
            row = observations["compile:build.zig:build:lapacke_" + width + "_static"]
            row = _prior_component_observation(self, row, width, count)
            self.assertEqual(count + 1, len(row["source_variants"]))
            self.assertEqual("src/lapacke_composed.zig", row["source_variants"]["lapacke-potri=true"])
            base_variants = {key.removeprefix("lapacke-potri=false,"): value
                             for key, value in row["source_variants"].items()
                             if key.startswith("lapacke-potri=false,")}
            self.assertEqual(count, len(base_variants))
            row = {**row, "source_variants": base_variants,
                   "root_source": [path for path in row["root_source"] if path != "src/lapacke_composed.zig"]}
            self.assertEqual(sorted(row["root_source"]), sorted(row["source_variants"].values()))
            self.assertTrue(all(key.startswith("lapacke-getri=") for key in row["source_variants"]))
        self.assertNotIn("lapacke_getri_tests", facts["root_owners"])
        self.assertIsNone(CHECKER.NEXT_SOURCE_PROJECTION_SHA256)

    def test_default_type_name_and_both_width_compositions_reject_mutation(self) -> None:
        source = (REPOSITORY_ROOT / "build.zig").read_text()
        statement = '    const lapacke_getri = b.option(bool, "lapacke-getri", "Include generated four-scalar LU inverse high/work/query in finite LP64/ILP64 and unified LP64 packages") orelse false;'
        self.assertEqual(1, source.count(statement))
        changes = [source.replace(statement, statement.replace("orelse false", "orelse true"), 1),
                   source.replace(statement, statement.replace("b.option(bool", "b.option(usize"), 1),
                   source.replace(statement, statement.replace('"lapacke-getri"', '"unreviewed-getri"'), 1)]
        for width in ("lp64", "ilp64"):
            before = '"{s}-getri-ordinary-' + width + '-v1"'
            self.assertEqual(1, source.count(before))
            changes.append(source.replace(before, before.replace("-getri-", "-unreviewed-getri-"), 1))
        for changed in changes:
            self.assertNotEqual(source, changed)
            with self.assertRaisesRegex(CHECKER.InventoryError, "selector|selection"):
                CHECKER._numerics_resources_graph_facts(changed)


class PotriComponentInventoryContractTests(unittest.TestCase):
    def test_exact_component_outputs_are_owned_and_route_to_component_generator(self) -> None:
        inventory = json.loads((REPOSITORY_ROOT / "tools/build_inventory.json").read_text())
        expected = {f"interface_components/mkl2026.{width}/potri-v1/{name}"
                    for width in ("lp64", "ilp64")
                    for name in ("lapacke.h", "lapacke_potri.h", "exports.zig", "exports.def", "abi_manifest.json")}
        self.assertEqual(expected, set(CHECKER.POTRI_COMPONENT_OUTPUTS))
        self.assertEqual(20, len(CHECKER.POTRI_COMPONENT_CLASSIFICATIONS))
        targets = {row["path"]: row for row in inventory["generator_targets"]}
        candidates = {row["id"]: row for row in inventory["derived_candidates"]}
        classifications = {row["path"]: row for row in inventory["repository_file_classifications"]}
        for row in CHECKER.POTRI_COMPONENT_CLASSIFICATIONS:
            self.assertEqual(row, classifications[row["path"]])
        for row in CHECKER.POTRI_COMPONENT_DERIVED_CANDIDATES:
            self.assertEqual(row, candidates[row["id"]])
            self.assertIn("tools/generate_lapacke_component.py", row["authoritative_inputs"])
            self.assertEqual("tools/generate_lapacke_component.py", targets[row["path"]]["anchor"]["file"])
            self.assertEqual("outputs", targets[row["path"]]["anchor"]["enclosing_function"])
        self.assertEqual("false", CHECKER.POTRI_COMPONENT_OPTION_SURFACE["default"])
        self.assertIsNone(CHECKER.NEXT_SOURCE_PROJECTION_SHA256)

    def test_component_selection_preserves_every_prior_base_and_replaces_only_outer_root(self) -> None:
        inventory = json.loads((REPOSITORY_ROOT / "tools/build_inventory.json").read_text())
        rows = {row["id"]: row for row in inventory["build_observations"]}
        graph = CHECKER._numerics_resources_build_contract((REPOSITORY_ROOT / "build.zig").read_text())
        for width, count in (("lp64", 128), ("ilp64", 16)):
            identifier = "compile:build.zig:build:lapacke_" + width + "_static"
            prior = graph["module_source_variants"]["lapacke_" + width + "_interface"]
            self.assertEqual(count, len(prior))
            self.assertEqual(count, len(set(prior.values())))
            observation = _prior_component_observation(self, rows[identifier], width, count)
            variants = observation["source_variants"]
            false_variants = {key.removeprefix("lapacke-potri=false,"): value
                              for key, value in variants.items()
                              if key.startswith("lapacke-potri=false,")}
            self.assertEqual(count, len(false_variants))
            self.assertEqual(set(prior.values()), set(false_variants.values()))
            axes = ("lapacke-getri", "lapacke-trtri", "lapacke-trtrs", "lapacke-posv")
            if width == "lp64":
                axes += ("lapacke-svd", "lapacke-complex-gelsd", "lapacke-qr")
            combinations = set()
            for key in false_variants:
                parts = key.split(",")
                choices = dict(part.split("=") for part in parts)
                self.assertEqual(len(axes), len(parts))
                self.assertEqual(set(axes), set(choices))
                self.assertTrue(all(value in ("false", "true") for value in choices.values()))
                combinations.add(tuple(choices[axis] for axis in axes))
            self.assertEqual(2 ** len(axes), len(combinations))
            self.assertEqual(count + 1, len(variants))
            self.assertEqual("src/lapacke_composed.zig", variants["lapacke-potri=true"])
            self.assertEqual(sorted(set(variants.values())), observation["root_source"])
        self.assertEqual(8, len(graph["module_borrow_variants"]))
        self.assertEqual(CHECKER.POTRI_MODULE_BORROWS,
                         {key: graph["module_borrow_variants"][key] for key in CHECKER.POTRI_MODULE_BORROWS})



class NativeSpectralHeaderSourceContractTests(unittest.TestCase):
    """Nested public header inputs cannot disappear behind the umbrella root."""
    @classmethod
    def setUpClass(cls) -> None:
        cls.source = (REPOSITORY_ROOT / "build.zig").read_text()

    def test_exact_two_header_inputs_single_root_and_install_edge(self) -> None:
        headers = ("include/zynum_native.h", "include/zynum_native_spectral.h")
        self.assertEqual(headers, CHECKER.NATIVE_PROVIDER_TRANSLATION_INPUTS)
        template = CHECKER.INJECTION_OBSERVATION_TEMPLATES["compile:build.zig:translateProvider:translated"]
        self.assertEqual([headers[0]], template["root_source"])
        self.assertEqual({path: hashlib.sha256((REPOSITORY_ROOT / path).read_bytes()).hexdigest()
                          for path in headers}, template["input_sha256"])
        self.assertEqual(["heteris.h", "heteris_runtime_provider.h"], template["external_include_dependencies"])
        identifier = "install:build.zig:addPlatformLibrary:include/zynum_native_spectral.h"
        row = CHECKER.INJECTION_OBSERVATION_TEMPLATES[identifier]
        self.assertEqual([headers[1]], row["source"])
        self.assertEqual(["zig-out/include/zynum_native_spectral.h"], row["destination"])
        step = CHECKER.INJECTION_OBSERVATION_TEMPLATES["step:build.zig:addPlatformLibrary:platform"]
        self.assertEqual(1, sum(r["id"] == identifier for r in step["direct_dependencies"]))
        classifications = {r["path"]: r for r in CHECKER.INJECTION_SOURCE_CLASSIFICATIONS}
        for path in (headers[1], "integrations/heteris/src/native_spectral_exports.zig",
                     "src/numerics/checked_spectral.zig", "src/numerics/spectral_native.zig"):
            self.assertEqual(hashlib.sha256((REPOSITORY_ROOT / path).read_bytes()).hexdigest(),
                             CHECKER.INJECTION_SOURCE_SHA256[path])
            self.assertEqual("public-api" if path.endswith(".h") else "library-source", classifications[path]["owner"])
            self.assertEqual("non-generated-source", classifications[path]["class"])

    def test_missing_mutated_nested_header_rejects_original_frozen_source_guard(self) -> None:
        tables = (CHECKER.GELSS_SOURCE_SHA256, CHECKER.GELS_SOURCE_SHA256, CHECKER.GVD_SOURCE_SHA256,
                  CHECKER.EVD_SOURCE_SHA256, CHECKER.CONTROLS_SOURCE_SHA256, CHECKER.SYGV_SOURCE_SHA256,
                  CHECKER.SYEV_SOURCE_SHA256, CHECKER.CONTOUR_SOURCE_SHA256, CHECKER.FEAST_SOURCE_SHA256,
                  CHECKER.FFTW_F32_SOURCE_SHA256, CHECKER.POTRI_OWNER_SOURCE_SHA256,
                  CHECKER.SHARED_FACADE_SOURCE_SHA256, CHECKER.EXECUTION_SOURCE_SHA256,
                  CHECKER.EXECUTION_OBJECT_SOURCE_SHA256, CHECKER.COMMON_CORE_SOURCE_SHA256,
                  CHECKER.MANAGED_LAPACKE_SOURCE_SHA256, CHECKER.MANAGED_SPARSE_SOURCE_SHA256,
                  CHECKER.UNIFIED_BLAS_SOURCE_SHA256, CHECKER.OPERATOR_PHASE_SOURCE_SHA256,
                  CHECKER.INJECTION_SOURCE_SHA256)
        with tempfile.TemporaryDirectory(prefix="spectral-header-source-") as directory:
            root = Path(directory)
            (root / "build.zig").write_text("pub fn build() void {}\n")
            paths = {CHECKER.NUMERICS_RESOURCES_PATH} | {path for table in tables for path in table}
            for path in paths:
                target = root / path
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes((REPOSITORY_ROOT / path).read_bytes())
            context = CHECKER._make_discovery_context(root)
            CHECKER._numerics_resources_source_contract(context)
            header = root / "include/zynum_native_spectral.h"
            original = header.read_bytes()
            header.write_bytes(original + b"\n/* unreviewed nested input */\n")
            with self.assertRaisesRegex(CHECKER.InventoryError, "native injection source-only bytes drifted: include/zynum_native_spectral.h"):
                CHECKER._numerics_resources_source_contract(CHECKER._make_discovery_context(root))
            header.unlink()
            with self.assertRaises((CHECKER.InventoryError, CHECKER.repository_snapshot.RepositorySnapshotError)):
                CHECKER._numerics_resources_source_contract(CHECKER._make_discovery_context(root))
            header.write_bytes(original)
            CHECKER._numerics_resources_source_contract(CHECKER._make_discovery_context(root))

    def test_wrong_root_and_duplicate_translation_reject_original_helper_guard(self) -> None:
        changes = (
            ('.root_source_file = b.path("include/zynum_native.h"),',
             '.root_source_file = b.path("include/zynum_native_spectral.h"),'),
            ('const provider_c = translateProvider(b, target, optimize, heteris_include);',
             'const provider_c = translateProvider(b, target, optimize, heteris_include);\n    const duplicate_c = translateProvider(b, target, optimize, heteris_include);'),
        )
        for old, new in changes:
            self.assertIn(old, self.source)
            # The same declaration also exists in the reference branch.
            # Bind this mutation to the integrated helper whose seam is tested.
            start = self.source.index("fn configureIntegrated(") if old.startswith("const provider_c") else 0
            position = self.source.index(old, start)
            changed = self.source[:position] + new + self.source[position + len(old):]
            with self.subTest(anchor=old), self.assertRaisesRegex(CHECKER.InventoryError, "reviewed native configure seam changed"):
                CHECKER._reviewed_configure_projection(changed)

    def test_false_header_install_source_destination_and_missing_edge_reject(self) -> None:
        call = 'b.installFile("include/zynum_native_spectral.h", "include/zynum_native_spectral.h");'
        self.assertEqual(1, self.source.count(call))
        for replacement in (
            'b.installFile("include/zynum_native.h", "include/zynum_native_spectral.h");',
            'b.installFile("include/zynum_native_spectral.h", "include/unreviewed.h");',
            '',
        ):
            with self.subTest(replacement=replacement), self.assertRaisesRegex(CHECKER.InventoryError, "reviewed native configure seam changed"):
                CHECKER._reviewed_configure_projection(self.source.replace(call, replacement, 1))


if __name__ == "__main__":
    unittest.main()
