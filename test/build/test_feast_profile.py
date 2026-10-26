"""Finite FEAST declarations and shared-owner graph fail-closed regressions."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
import generate_feast as generator
import check_build_inventory as inventory


class FeastProfileTests(unittest.TestCase):
    def test_generation_is_declarations_with_explicit_profile_limits(self):
        profile_path = ROOT / "tools/feast_lp64_profile.json"
        profile = json.loads(profile_path.read_text())
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "generated"
            generator.generate(profile, output, profile_path)
            manifest = json.loads((output / "manifest.json").read_text())
            self.assertEqual(generator.validate_manifest(manifest), tuple(generator.SYMBOLS))
            self.assertFalse(manifest["runtime_exported"])
            self.assertFalse(manifest["full_feast_compatibility"])
            self.assertIn("zero-height contour", manifest["pending_modes"])
            self.assertEqual(set(manifest["generated_sha256"]), {"feast_lp64.h", "feast_exports.zig", "FEASTConfig.cmake", "FEASTConfigVersion.cmake"})
            forwards = (output / "feast_exports.zig").read_text()
            self.assertNotIn("Context.init", forwards)
            self.assertNotIn("while", forwards)

    def test_bad_profiles_and_manifests_are_rejected_before_generation(self):
        profile_path = ROOT / "tools/feast_lp64_profile.json"
        profile = json.loads(profile_path.read_text())
        faults = [("schema", "wrong"), ("integer_bits", True), ("integer_bits", 64), ("complex_f64_bytes", 8),
                  ("full_feast_compatibility", True), ("full_feast_compatibility", 0),
                  ("symbols", profile["symbols"] + ["zfeast_hcsrgv_"]), ("target", "unreviewed")]
        with tempfile.TemporaryDirectory() as directory:
            for index, (key, value) in enumerate(faults):
                changed = copy.deepcopy(profile)
                changed[key] = value
                output = Path(directory) / str(index)
                with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                    generator.generate(changed, output, profile_path)
                self.assertFalse(output.exists())
            output = Path(directory) / "valid"
            generator.generate(profile, output, profile_path)
            manifest = json.loads((output / "manifest.json").read_text())
            for key, value in [("schema", "wrong"), ("runtime_exported", True), ("runtime_exported", 0),
                               ("generator_sha256", "unbound"), ("generated_sha256", {}), ("symbols", ["feastinit_"])]:
                changed = copy.deepcopy(manifest)
                changed[key] = value
                with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                    generator.validate_manifest(changed)

    def test_selected_core_default_and_finite_generated_paths_are_bound(self):
        source = (ROOT / "build.zig").read_text()
        facts = inventory._feast_build_contract(source)
        self.assertEqual(facts["option_default"], "none")
        self.assertEqual(facts["Core_owner"], "zynum_core_mod")
        self.assertEqual(len(facts["receiver_uses"]), 5)
        changes = [
            ('over the unified Core") orelse .none;', 'over the unified Core") orelse .feast4_lp64_abplas;'),
            ('feast_binding_mod.addImport("zynum-core", zynum_core_mod);', 'feast_binding_mod.addImport("zynum-core", zynum_core_test_mod);'),
            ('feast_combined_mod.addImport("zynum-abplas-base-exports", abplas_lp64_pre_feast_root);', 'feast_combined_mod.addImport("zynum-abplas-base-exports", abplas_lp64_root);'),
            ('feast_directory.path(b, "feast_exports.zig")', 'feast_directory.path(b, "other.zig")'),
            ('abplas_lp64_shared.addArg("--feast-manifest");', 'abplas_lp64_shared.addArg("--operator-manifest");'),
        ]
        for before, after in changes:
            self.assertIn(before, source)
            with self.subTest(change=before), self.assertRaises(inventory.InventoryError):
                inventory._feast_build_contract(source.replace(before, after, 1))

    def test_numerical_layer_cannot_import_the_interface(self):
        with self.assertRaises(inventory.InventoryError):
            inventory._interface_private_source_imports("src/numerics/invalid.zig", 'const x=@import("../interop/feast.zig");')
        inventory._interface_private_source_imports("src/interop/feast_test.zig", 'const x=@import("feast.zig");')

    def test_duplicate_core_binding_outside_the_frontend_is_rejected(self):
        source = (ROOT / "build.zig").read_text()
        inventory._numerics_resources_graph_facts(source)
        marker = "    const test_native_feature_step = b.step("
        self.assertEqual(source.count(marker), 1)
        extra = '    const cloned_feast = b.createModule(.{ .root_source_file = b.path("src/interop/feast_binding.zig"), .target = target, .optimize = optimize, .link_libc = true });\n    cloned_feast.addImport("zynum-core", zynum_core_mod);\n'
        with self.assertRaises(inventory.InventoryError):
            inventory._numerics_resources_graph_facts(source.replace(marker, extra + marker, 1))


if __name__ == "__main__":
    unittest.main()
