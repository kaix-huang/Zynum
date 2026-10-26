# Copyright (C) 2026 Zynum contributors
# SPDX-License-Identifier: LGPL-3.0-or-later
"""GELS declaration arity, work/query ABI and exact borrowed owner guards."""
import copy
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'tools'))
import generate_lapacke_component as generator
import lapacke_components as components


class GelsComponentTests(unittest.TestCase):
    def test_original_eight_signatures_keep_four_scalar_workspace_arities(self):
        output = generator.outputs('mkl2026.lp64', 'gels', components.TARGET)
        manifest = json.loads(output['abi_manifest.json'])
        wanted = ['LAPACKE_' + tag + name + suffix
                  for tag, name in [('s', 'gels'), ('d', 'gels'), ('c', 'gels'), ('z', 'gels')]
                  for suffix in ['', '_work']]
        self.assertEqual(manifest['plan']['symbols'], wanted)
        self.assertEqual(re.findall(r'\.name = "(LAPACKE_\w+)"', output['exports.zig']), wanted)
        self.assertEqual(re.findall(r'lapack_int (LAPACKE_\w+)\(', output['lapacke_gels.h']), wanted)
        self.assertEqual([r['parameter_count'] for r in manifest['plan']['bindings']], [9, 11, 9, 11, 9, 11, 9, 11])
        for row in manifest['plan']['bindings']:
            self.assertEqual(hashlib.sha256(row['declaration'].encode()).hexdigest(), row['original_declaration_sha256'])
        self.assertFalse(manifest['plan']['full_profile_ready'])
        self.assertEqual(manifest['plan']['runtime_abi_status'], 'pending')
        self.assertEqual(output, generator.outputs('mkl2026.lp64', 'gels', components.TARGET))

    def test_unreviewed_profiles_components_and_targets_fail_before_formatter(self):
        with mock.patch.object(generator.subprocess, 'run', side_effect=AssertionError('formatter reached')):
            for args in [('mkl2026.ilp64', 'gels', components.TARGET),
                         ('reference', 'gels', components.TARGET),
                         ('mkl2026.lp64', 'evx', components.TARGET),
                         ('mkl2026.lp64', 'gels', 'linux')]:
                with self.subTest(args=args), self.assertRaises(ValueError):
                    generator.outputs(*args)

    def test_forged_owner_signature_workspace_arity_and_runtime_claim_fail(self):
        original = components.manifest_for('mkl2026.lp64', 'gels', components.TARGET)
        for field in ['owner', 'arity', 'signature', 'generator', 'runtime']:
            changed = copy.deepcopy(original)
            if field == 'owner':
                changed['plan']['owner_imports'][0]['instantiate'] = True
            elif field == 'arity':
                changed['plan']['bindings'][-1]['parameter_count'] = 10
            elif field == 'signature':
                changed['plan']['bindings'][-1]['declaration'] = changed['plan']['bindings'][-1]['declaration'].replace('lapack_int lwork', 'lapack_int* lwork')
            elif field == 'generator':
                changed['plan']['generator_inputs']['tools/interface_templates/gels_component_lp64.zig.in'] = '0' * 64
            else:
                changed['plan']['runtime_abi_status'] = 'qualified'
            with self.subTest(field=field), self.assertRaises(ValueError):
                components.validate_component(changed, 'mkl2026.lp64')

    def test_ordered_composition_keeps_all_existing_entries_and_rejects_bad_owners(self):
        base = json.loads((ROOT / 'interface_subset/mkl2026.lp64/getri-ordinary-lp64-v1/abi_manifest.json').read_text())
        items = [components.manifest_for('mkl2026.lp64', n, components.TARGET)
                 for n in ['controls', 'evd', 'gels', 'gvd', 'potri', 'syev', 'sygv']]
        result = components.compose(base, items)
        unwrapped, additions = components.unwrap(result)
        self.assertEqual(base, unwrapped)
        self.assertEqual(len(additions), 42)
        self.assertEqual(result['plan']['symbols'], base['plan']['symbols'] + additions)
        for bad in [[items[2], items[0]], [items[2], items[2]],
                    [components.manifest_for('mkl2026.ilp64', 'controls', components.TARGET), items[2]]]:
            with self.subTest(count=len(bad)), self.assertRaises(ValueError):
                components.compose(base, bad)
        text = (ROOT / 'tools/interface_templates/gels_component_lp64.zig.in').read_text()
        self.assertEqual(text.count('binding.Shared.gelsRawManagedWithDefaults('), 1)
        for token in ['Context.init', 'std.heap', 'thread_pool', 'native_options', 'runScratchAdmitted']:
            self.assertNotIn(token, text)

    @unittest.skipUnless(shutil.which('zig'), 'Zig is required for composer identity guards')
    def test_actual_composer_rejects_different_Core_configuration_owner_and_width(self):
        with tempfile.TemporaryDirectory(prefix='gels-composer-guards-') as directory:
            root = Path(directory)
            for name in ['core', 'other_core']:
                (root / (name + '.zig')).write_text('// Distinct module identity.\n')
            for name in ['binding_a', 'binding_b']:
                (root / (name + '.zig')).write_text('pub const Provider = struct { pub const OwnerIdentity = @import("core"); };\n')
            (root / 'base.zig').write_text('pub const WireInt = c_int;\n')
            cases = [(False, False, False, None),
                     (True, False, False, 'borrow the configured Core'),
                     (False, True, False, 'existing interface configuration owner'),
                     (False, False, True, 'integer profiles differ')]
            for i, (wrong_core, wrong_binding, wrong_width, expected) in enumerate(cases):
                with self.subTest(case=i):
                    (root / 'gels.zig').write_text(
                        'pub const OwnerIdentity = @import("' + ('other_core' if wrong_core else 'core') + '");\n'
                        'pub const BindingOwner = @import("' + ('binding_b' if wrong_binding else 'binding_a') + '");\n'
                        'pub const WireInt = ' + ('c_longlong' if wrong_width else 'c_int') + ';\n')
                    argv = ['zig', 'build-obj', '-ODebug', '-target', 'aarch64-macos', '-mcpu', 'baseline',
                            '--dep', 'zynum-core=core', '--dep', 'lapacke-interface-owner=binding_a',
                            '--dep', 'zynum-lapacke-base-exports=base', '--dep', 'zynum-lapacke-gels-exports=gels',
                            '-Mroot=' + str(ROOT/'src/lapacke_gels_composed.zig'),
                            '--dep', 'core', '--dep', 'other_core', '--dep', 'binding_a', '--dep', 'binding_b', '-Mgels=' + str(root/'gels.zig'),
                            '--dep', 'core', '-Mbinding_a=' + str(root/'binding_a.zig'),
                            '--dep', 'core', '-Mbinding_b=' + str(root/'binding_b.zig'),
                            '-Mcore=' + str(root/'core.zig'), '-Mother_core=' + str(root/'other_core.zig'), '-Mbase=' + str(root/'base.zig'),
                            '--cache-dir', str(root/('cache' + str(i))), '-femit-bin=' + str(root/('out' + str(i) + '.o'))]
                    actual = subprocess.run(argv, capture_output=True, text=True, cwd=root)
                    if expected is None:
                        self.assertEqual(actual.returncode, 0, actual.stderr)
                    else:
                        self.assertNotEqual(actual.returncode, 0)
                        self.assertIn(expected, actual.stderr)



    def test_build_selector_binding_and_selected_module_mutations_fail_closed(self):
        import check_build_inventory as inventory
        text = (ROOT / 'build.zig').read_text()
        self.assertEqual(inventory._json_fact_digest(inventory._numerics_resources_graph_facts(text)), inventory.NUMERICS_RESOURCES_GRAPH_SHA256)
        replacements = [
            ('"lapacke-gels", "Compose generated LP64 four-scalar full-rank least-squares high/work/query entries over the existing LAPACKE owner") orelse false',
             '"lapacke-gels", "Compose generated LP64 four-scalar full-rank least-squares high/work/query entries over the existing LAPACKE owner") orelse true'),
            ('lapacke_lp64_gels_composed_root.addImport("zynum-core", zynum_core_mod)',
             'lapacke_lp64_gels_composed_root.addImport("zynum-core", zynum_core_test_mod)'),
            ('abplas_lp64_lapacke_gels_component.addImport("lapacke-interface-owner", lapacke_binding_mod)',
             'abplas_lp64_lapacke_gels_component.addImport("lapacke-interface-owner", lapacke_binding_test_mod)'),
            ('var lapacke_lp64_gels_selected = lapacke_lp64_gvd_selected;',
             'var lapacke_lp64_gels_selected = lapacke_lp64_interface;')]
        for old, new in replacements:
            self.assertEqual(text.count(old), 1)
            with self.subTest(mutation=new), self.assertRaises(inventory.InventoryError):
                inventory._numerics_resources_graph_facts(text.replace(old, new))


if __name__ == '__main__':
    unittest.main(verbosity=2)
