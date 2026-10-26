# Copyright (C) 2026 Zynum contributors
# SPDX-License-Identifier: LGPL-3.0-or-later
import copy
import hashlib
import json
from pathlib import Path
import re
import sys
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'tools'))
import generate_lapacke_component as generator
import lapacke_components as components


class SygvComponentTests(unittest.TestCase):
    def test_original_lp64_declarations_and_four_forwarders_have_exact_arity(self):
        output = generator.outputs('mkl2026.lp64', 'sygv', components.TARGET)
        manifest = json.loads(output['abi_manifest.json'])
        wanted = ['LAPACKE_' + tag + 'sygv' + suffix for tag in 'sd' for suffix in ('', '_work')]
        self.assertEqual(manifest['plan']['symbols'], wanted)
        self.assertEqual(re.findall(r'\.name = "(LAPACKE_\w+)"', output['exports.zig']), wanted)
        self.assertEqual(re.findall(r'lapack_int (LAPACKE_\w+)\(', output['lapacke_sygv.h']), wanted)
        self.assertEqual([v['parameter_count'] for v in manifest['plan']['bindings']], [10, 12, 10, 12])
        for row in manifest['plan']['bindings']:
            self.assertEqual(hashlib.sha256(row['declaration'].encode()).hexdigest(), row['original_declaration_sha256'])
            self.assertEqual(row['operation'], 'symmetric.generalized_eigen')
        self.assertEqual(generator.outputs('mkl2026.lp64', 'sygv', components.TARGET), output)
        self.assertFalse(manifest['plan']['full_profile_ready'])
        self.assertEqual(manifest['plan']['runtime_abi_status'], 'pending')

    def test_unreviewed_width_profile_component_and_target_fail_before_formatter(self):
        with mock.patch.object(generator.subprocess, 'run', side_effect=AssertionError('formatter reached')):
            for args in [('mkl2026.ilp64', 'sygv', components.TARGET),
                         ('reference', 'sygv', components.TARGET),
                         ('mkl2026.lp64', 'sygvd', components.TARGET),
                         ('mkl2026.lp64', 'sygv', 'linux')]:
                with self.subTest(args=args), self.assertRaises(ValueError):
                    generator.outputs(*args)

    def test_forged_owner_signature_generator_and_runtime_claims_are_rejected(self):
        original = components.manifest_for('mkl2026.lp64', 'sygv', components.TARGET)
        for field in ['owner', 'arity', 'signature', 'generator', 'runtime']:
            value = copy.deepcopy(original)
            if field == 'owner':
                value['plan']['owner_imports'][0]['instantiate'] = True
            elif field == 'arity':
                value['plan']['bindings'][1]['parameter_count'] = 8
            elif field == 'signature':
                value['plan']['bindings'][0]['declaration'] += ' forged'
            elif field == 'generator':
                value['plan']['generator_inputs']['tools/interface_templates/symmetric_generalized_component_lp64.zig.in'] = '0' * 64
            else:
                value['plan']['runtime_abi_status'] = 'qualified'
            with self.subTest(field=field), self.assertRaises(ValueError):
                components.validate_component(value, 'mkl2026.lp64')

    def test_combined_potri_sygv_keeps_existing_base_and_rejects_duplicates_order_and_width(self):
        name = 'potrf-potrs-getrf-getrs-ordinary-lp64-v1'
        base = json.loads((ROOT / 'interface_subset/mkl2026.lp64' / name / 'abi_manifest.json').read_text())
        potri = components.manifest_for('mkl2026.lp64', 'potri', components.TARGET)
        sygv = components.manifest_for('mkl2026.lp64', 'sygv', components.TARGET)
        syev = components.manifest_for('mkl2026.lp64', 'syev', components.TARGET)
        composed = components.compose(base, [potri, syev, sygv])
        unwrapped, extra = components.unwrap(composed)
        self.assertEqual(unwrapped, base)
        self.assertEqual(len(extra), 16)
        self.assertEqual(composed['plan']['symbols'], base['plan']['symbols'] + extra)
        for items in [[sygv, potri], [sygv, sygv], [potri, sygv, syev], [potri, syev, sygv, potri]]:
            with self.subTest(components=len(items)), self.assertRaises(ValueError):
                components.compose(base, items)
        template = (ROOT / 'tools/interface_templates/symmetric_generalized_component_lp64.zig.in').read_text()
        self.assertEqual(template.count('binding.Shared.generalizedSymmetricRawManagedWithDefaults('), 1)
        self.assertNotIn('native_options', template)
        for token in ('Context.init', 'std.heap', 'thread_pool', 'solveWithContract'):
            self.assertNotIn(token, template)
        root = (ROOT / 'src/lapacke_symmetric_generalized_composed.zig').read_text()
        self.assertIn('generalized.OwnerIdentity != Core', root)
        self.assertIn('WireInt != generalized.WireInt', root)



    def test_build_selection_and_shared_owner_mutations_fail_closed(self):
        import check_build_inventory as inventory
        text = (ROOT / 'build.zig').read_text()
        original = inventory._numerics_resources_graph_facts(text)
        self.assertEqual(inventory._json_fact_digest(original), inventory.NUMERICS_RESOURCES_GRAPH_SHA256)
        replacements = [
            ('"lapacke-sygv", "Compose generated LP64 generalized real symmetric eigenvalue high/work entries over the same LAPACKE owner") orelse false',
             '"lapacke-sygv", "Compose generated LP64 generalized real symmetric eigenvalue high/work entries over the same LAPACKE owner") orelse true'),
            ('lapacke_lp64_sygv_composed_root.addImport("zynum-core", zynum_core_mod)',
             'lapacke_lp64_sygv_composed_root.addImport("zynum-core", zynum_core_test_mod)'),
            ('abplas_lp64_lapacke_sygv_component.addImport("lapacke-interface-owner", lapacke_binding_mod)',
             'abplas_lp64_lapacke_sygv_component.addImport("lapacke-interface-owner", lapacke_binding_test_mod)'),
            ('const lapacke_lp64_before_sygv = lapacke_lp64_syev_selected;',
             'const lapacke_lp64_before_sygv = lapacke_lp64_interface;'),
        ]
        for old, new in replacements:
            self.assertEqual(text.count(old), 1)
            with self.subTest(mutation=new), self.assertRaises(inventory.InventoryError):
                inventory._numerics_resources_graph_facts(text.replace(old, new))


if __name__ == '__main__':
    unittest.main()
