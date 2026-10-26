"""Optional neutral Contour generation and source-owned contract rejection."""
from pathlib import Path
import copy
import importlib.util
import json
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
import generate_contour_interface as contour

class ContourProfile(unittest.TestCase):
    def test_optional_wire_only_and_entry_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            contour.generate(root / "wire")
            wire = json.loads((root / "wire/manifest.json").read_text())
            self.assertEqual(wire["exported_solver_entries"], [])
            self.assertFalse((root / "wire/contour_v1.zig").exists())
            contour.generate(root / "entries", with_entries=True)
            manifest = json.loads((root / "entries/manifest.json").read_text())
            self.assertEqual(len(contour.validate_manifest(manifest)), 4)
            self.assertEqual(manifest["phase_stages"]["contour_control"], 50)
            self.assertEqual(manifest["phase_stages"]["contour_normalization"], 56)
            self.assertFalse(manifest["installed_interface_qualified"])
            self.assertFalse(manifest["ABP_integrated"])
            self.assertEqual((root / "wire/contour_v1_types.zig").read_bytes(),
                             (root / "entries/contour_v1_types.zig").read_bytes())

    def test_link_manifest_rejects_unreviewed_variants(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            contour.generate(root, with_entries=True)
            good = json.loads((root / "manifest.json").read_text())
            cases = []
            for key, value in [("target", "unreviewed"), ("pointer_bits", 32),
                               ("profile", "zynum-v2"), ("rational_result_bytes", 80),
                               ("base_operator_profile", "other"), ("phase_agreement", False)]:
                bad = copy.deepcopy(good)
                bad[key] = value
                cases.append(bad)
            for replacement in [[], list(reversed(good["symbols"])), good["symbols"] + [good["symbols"][0]]]:
                bad = copy.deepcopy(good)
                bad["symbols"] = replacement
                cases.append(bad)
            bad = copy.deepcopy(good)
            bad["layouts"]["f32"]["options_bytes"] += 8
            cases.append(bad)
            bad = copy.deepcopy(good)
            bad["phase_stages"]["contour_control"] = 49
            cases.append(bad)
            for bad in cases:
                with self.subTest(bad=bad):
                    with self.assertRaises(ValueError):
                        contour.validate_manifest(bad)

    def test_reviewed_build_contract_rejects_scope_and_owner_mutations(self):
        path = ROOT / "tools/check_build_inventory.py"
        spec = importlib.util.spec_from_file_location("contour_build_contract_test", path)
        inventory = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = inventory
        spec.loader.exec_module(inventory)
        source = (ROOT / "build.zig").read_text()
        inventory._contour_build_contract(source)
        mutations = [
            source.replace('"contour-interface"', '"other-contour-interface"'),
            source.replace('contour_interface == .zynum_v1 and operator_binding_owner != null',
                           'contour_interface == .none and operator_binding_owner != null'),
            source.replace('operator_binding_owner = operator_binding_mod;',
                           'operator_binding_owner = fftw_binding_mod;'),
            source.replace('contour_exports_mod.addImport("zynum-core", zynum_core_mod)',
                           'contour_exports_mod.addImport("zynum-core", zynum_core_test_mod)'),
        ]
        for mutation in mutations:
            self.assertNotEqual(mutation, source)
            with self.assertRaises(inventory.InventoryError):
                inventory._contour_build_contract(mutation)

if __name__ == "__main__":
    unittest.main()
