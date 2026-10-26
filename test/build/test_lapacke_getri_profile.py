# Copyright (C) 2026 Zynum contributors
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Finite GETRI declaration/recipe closure and rejection before side effects."""
from pathlib import Path
import copy
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'tools'))
import checked_abi_schema as schema
import standard_abi_bindings as standard
import generate_checked_abi as generator


class GetriProfileTests(unittest.TestCase):
    def test_exact_four_scalar_two_width_five_and_seven_parameter_contract_and_120_sets(self):
        selections = [(name, row) for name, row in standard.SETS.items() if 'getri-ordinary-' in name]
        self.assertEqual(120, len(selections))
        self.assertEqual(118, len(standard._GETRI_BASE_SETS))
        self.assertEqual(58, len(standard._TRTRI_BASE_SETS))
        self.assertEqual(238, len(standard.GENERATION_ACCEPTANCES))
        for name, row in selections:
            with self.subTest(name=name):
                plan = schema.select_binding_set(row['profile'], name, standard.TARGET_ID)
                records = [b for b in plan['bindings'] if b['operation'] == 'lu.inverse']
                self.assertEqual(8, len(records))
                self.assertEqual({(s, level) for s in ('f32', 'f64', 'complex_f32', 'complex_f64')
                    for level in ('high', 'work')}, {(b['scalar'], b['level']) for b in records})
                integer = 'int' if row['profile'] == 'mkl2026.lp64' else 'long long int'
                for binding in records:
                    params = binding['wire_parameters']
                    expected = ('matrix_layout', 'n', 'a', 'lda', 'ipiv')
                    if binding['level'] == 'work':
                        expected += ('work', 'lwork')
                    self.assertEqual(expected, tuple(p['name'] for p in params))
                    self.assertEqual(integer, binding['wire_return']['c_type'])
                    self.assertEqual('int', params[0]['c_type'])
                    self.assertEqual((False, True), tuple(p['pointee_const'] for p in params[:5] if p['pointer_depth']))
                    self.assertEqual(integer, params[4]['c_type'])
                    if binding['level'] == 'work':
                        self.assertFalse(params[5]['pointee_const'])
                        self.assertEqual(integer, params[6]['c_type'])
                    self.assertEqual('raw_getri_profile_defaults', binding['recipe'])
                self.assertEqual('pending', plan['runtime_abi_status'])
                self.assertFalse(plan['full_profile_ready'])
                self.assertFalse(plan['owner_imports'][0]['instantiate'])

    def test_wrong_width_target_partial_and_repeated_addon_reject_before_formatter(self):
        selection = 'getri-ordinary-lp64-v1'
        cases = [('mkl2026.ilp64', selection, standard.TARGET_ID), ('mkl2026.lp64', selection, 'linux-x86_64'),
            ('mkl2026.lp64', 'potrf-getrs-getri-ordinary-lp64-v1', standard.TARGET_ID),
            ('mkl2026.lp64', 'getri-getri-ordinary-lp64-v1', standard.TARGET_ID)]
        for profile, name, target in cases:
            with self.subTest(name=name), tempfile.TemporaryDirectory() as temp:
                output = Path(temp) / 'uncreated'
                result = subprocess.run([sys.executable, '-B', str(ROOT / 'tools/generate_checked_abi.py'),
                    '--profile', profile, '--binding-set', name, '--target-contract', target,
                    '--root', str(output), '--zig', '/never-invoke'], capture_output=True, text=True)
                self.assertNotEqual(0, result.returncode)
                self.assertIn('error:', result.stderr)
                self.assertNotIn('FileNotFoundError', result.stderr)
                self.assertFalse(output.exists())

    def test_pointer_const_order_work_type_recipe_and_self_authorization_mutations_reject(self):
        key = 'getri-lp64-d-work'
        original = dict(standard.BINDINGS[key])
        for change in ('const-a', 'mutable-pivot', 'const-work', 'order', 'lwork-type',
            'declaration', 'recipe', 'semantics', 'runtime'):
            row = copy.deepcopy(original)
            params = list(row['parameters'])
            if change in ('const-a', 'mutable-pivot', 'const-work'):
                index = {'const-a': 2, 'mutable-pivot': 4, 'const-work': 5}[change]
                params[index] = (*params[index][:4], not params[index][4])
                row['parameters'] = tuple(params)
            elif change == 'order':
                params[4], params[5] = params[5], params[4]
                row['parameters'] = tuple(params)
            elif change == 'lwork-type':
                params[6] = ('lwork', 'c_int', 'workspace_length', 0, False)
                row['parameters'] = tuple(params)
            elif change == 'declaration':
                row['declaration'] = (*row['declaration'][:-1], '0' * 64)
            elif change == 'recipe':
                row['recipe'] = 'raw_getrs_profile_defaults'
            elif change == 'semantics':
                row['semantic_contracts'] = standard.LU_SEMANTIC_CONTRACTS
            else:
                row['runtime_evidence'] = ('invented',)
            with self.subTest(change=change), patch.object(standard, 'BINDINGS', {**standard.BINDINGS, key: row}), patch.object(generator.subprocess, 'run') as formatter:
                with self.assertRaises(schema.ProfileError):
                    generator.subset_outputs('mkl2026.lp64', 'getri-ordinary-lp64-v1', standard.TARGET_ID, '/never-invoke')
                formatter.assert_not_called()

    def test_changed_actual_engine_or_numerical_kernel_reject_before_formatter(self):
        for name in ('src/interop/lapacke_getri.zig', 'src/numerics/lu_inverse.zig'):
            with self.subTest(path=name), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                for relative, _ in standard.SOURCE_DEPENDENCIES:
                    output = root / relative
                    output.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(ROOT / relative, output)
                relative = standard.RESERVED_BLAS_MANIFEST[0]
                (root / relative).parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(ROOT / relative, root / relative)
                (root / name).write_bytes((root / name).read_bytes() + b'\n')
                with patch.object(schema, '_binding_source_root', return_value=root), patch.object(generator.subprocess, 'run') as formatter:
                    with self.assertRaisesRegex(schema.ProfileError, 'Accepted source dependency missing or changed: ' + name):
                        generator.subset_outputs('mkl2026.ilp64', 'getri-ordinary-ilp64-v1', standard.TARGET_ID, '/never-invoke')
                    formatter.assert_not_called()

    def test_combined_linker_rejects_standalone_unknown_or_changed_export_lists(self):
        name = 'potrf-potrs-getrf-getrs-gesv-stev-gelsd-heev-hegv-posv-trtrs-trtri-getri-ordinary-lp64-v1'
        manifest = ROOT / 'interface_subset/mkl2026.lp64' / name / 'abi_manifest.json'
        for change in ('standalone', 'unknown', 'missing', 'extra', 'duplicate'):
            value = json.loads(manifest.read_text())
            if change == 'standalone':
                value['plan']['set'] = 'getri-ordinary-lp64-v1'
            elif change == 'unknown':
                value['plan']['set'] = 'unreviewed-getri-ordinary-lp64-v1'
            elif change == 'missing':
                value['plan']['symbols'].remove('LAPACKE_dgetri')
            elif change == 'extra':
                value['plan']['symbols'].append('LAPACKE_unreviewed')
            else:
                value['plan']['symbols'].remove('LAPACKE_dgetri')
                value['plan']['symbols'].append('LAPACKE_sgetri')
            with self.subTest(change=change), tempfile.TemporaryDirectory() as temp:
                path = Path(temp) / 'manifest.json'
                path.write_text(json.dumps(value))
                output = Path(temp) / 'uncreated.dylib'
                result = subprocess.run([sys.executable, '-B', str(ROOT / 'tools/link_darwin_interface.py'),
                    '/never-read.a', str(ROOT / 'include/zynum/checked/sparse_manifest.json'), str(output),
                    '--macos-min', '27.0.1', '--legacy-manifest', str(ROOT / 'include/zynum/checked/abi_manifest.json'),
                    '--lapacke-manifest', str(path)], capture_output=True, text=True)
                self.assertNotEqual(0, result.returncode)
                self.assertIn('unsupported combined LP64 LAPACKE contract' if change in ('standalone', 'unknown')
                    else 'invalid combined LP64 LAPACKE export list', result.stderr)
                self.assertFalse(output.exists())

    def test_forwarding_and_default_off_selection_have_no_numerical_or_resource_ownership(self):
        template = (ROOT / 'tools/interface_templates/getri.zig.in').read_text()
        self.assertEqual(1, template.count('binding.Shared.getriRawManagedWithDefaults('))
        self.assertEqual(1, template.count('binding.Provider.Execution.getContext()'))
        self.assertEqual(2, template.count('pub fn call('))
        for forbidden in ('@import(', 'Context.init', 'allocator', 'invertNonsingular', 'thread_pool'):
            self.assertNotIn(forbidden, template)
        text = (ROOT / 'build.zig').read_text()
        self.assertIn('const lapacke_getri = b.option(bool, "lapacke-getri"', text)
        self.assertIn('LU inverse high/work/query in finite LP64/ILP64 and unified LP64 packages") orelse false;', text)
        for width in ('lp64', 'ilp64'):
            self.assertIn('const lapacke_' + width + '_set = if (lapacke_getri)', text)
            self.assertIn('else\n        lapacke_' + width + '_before_getri_set;', text)
        self.assertEqual(120, sum('getri-ordinary-' in name for name in standard.SETS))


if __name__ == '__main__':
    unittest.main()
