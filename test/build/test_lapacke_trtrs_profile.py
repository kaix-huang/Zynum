# Copyright (C) 2026 Zynum contributors
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Finite TRTRS wire contract, source authorization and early rejection."""
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


class TrtrsProfileTests(unittest.TestCase):
    def test_exact_four_scalar_two_width_ten_parameter_contract_and_24_sets(self):
        selections = [(name, row) for name, row in standard.SETS.items() if 'trtrs-ordinary-' in name]
        self.assertEqual(24, len(selections))
        for name, row in selections:
            with self.subTest(name=name):
                plan = schema.select_binding_set(row['profile'], name, standard.TARGET_ID)
                bindings = [value for value in plan['bindings'] if value['operation'] == 'triangular.solve']
                self.assertEqual(8, len(bindings))
                self.assertEqual({(scalar, level) for scalar in ('f32', 'f64', 'complex_f32', 'complex_f64') for level in ('high', 'work')}, {(value['scalar'], value['level']) for value in bindings})
                integer = 'int' if row['profile'] == 'mkl2026.lp64' else 'long long int'
                for value in bindings:
                    parameters = value['wire_parameters']
                    self.assertEqual(('matrix_layout', 'uplo', 'trans', 'diag', 'n', 'nrhs', 'a', 'lda', 'b', 'ldb'), tuple(item['name'] for item in parameters))
                    self.assertEqual(integer, value['wire_return']['c_type'])
                    self.assertEqual(('char', 'char', 'char'), tuple(parameters[i]['c_type'] for i in (1, 2, 3)))
                    self.assertEqual((integer,) * 4, tuple(parameters[i]['c_type'] for i in (4, 5, 7, 9)))
                    self.assertEqual((True, False), tuple(item['pointee_const'] for item in parameters if item['pointer_depth']))
                    self.assertEqual('raw_trtrs_profile_defaults', value['recipe'])
                self.assertEqual('pending', plan['runtime_abi_status'])
                self.assertFalse(plan['full_profile_ready'])
                self.assertFalse(plan['owner_imports'][0]['instantiate'])
                with self.assertRaises(TypeError):
                    plan['full_profile_ready'] = True

    def test_unknown_cli_target_width_and_partial_combination_fail_before_formatter(self):
        good = 'trtrs-ordinary-lp64-v1'
        cases = [('mkl2026.ilp64', good, standard.TARGET_ID), ('mkl2026.lp64', good, 'linux-x86_64'), ('mkl2026.lp64', 'potrf-potrs-getrf-getrs-trtrs-ordinary-lp64-v1', standard.TARGET_ID), ('mkl2026.lp64', 'unreviewed-trtrs-ordinary-lp64-v1', standard.TARGET_ID)]
        for profile, selection, target in cases:
            with self.subTest(selection=selection), tempfile.TemporaryDirectory() as temp:
                output = Path(temp) / 'uncreated'
                result = subprocess.run([sys.executable, '-B', str(ROOT / 'tools/generate_checked_abi.py'), '--profile', profile, '--binding-set', selection, '--target-contract', target, '--root', str(output), '--zig', '/never-invoke'], capture_output=True, text=True)
                self.assertNotEqual(0, result.returncode)
                self.assertIn('error:', result.stderr)
                self.assertNotIn('FileNotFoundError', result.stderr)
                self.assertFalse(output.exists())

    def test_parameter_const_order_decl_recipe_semantics_and_evidence_cannot_authorize_themselves(self):
        key = 'trtrs-lp64-d-high'
        original = dict(standard.BINDINGS[key])
        for change in ('const-a', 'const-b', 'order', 'diag-type', 'declaration', 'recipe', 'semantics', 'info', 'runtime'):
            row = copy.deepcopy(original)
            parameters = list(row['parameters'])
            if change in ('const-a', 'const-b'):
                index = 6 if change == 'const-a' else 8
                parameters[index] = (*parameters[index][:4], change == 'const-b')
                row['parameters'] = tuple(parameters)
            elif change == 'order':
                parameters[2], parameters[3] = parameters[3], parameters[2]
                row['parameters'] = tuple(parameters)
            elif change == 'diag-type':
                parameters[3] = ('diag', 'lapack_int', 'diagonal_kind', 0, False)
                row['parameters'] = tuple(parameters)
            elif change == 'declaration':
                row['declaration'] = (*row['declaration'][:-1], '0' * 64)
            elif change == 'recipe':
                row['recipe'] = 'raw_getrs_profile_defaults'
            elif change == 'semantics':
                row['semantic_contracts'] = standard.SEMANTIC_CONTRACTS
            elif change == 'info':
                row['return'] = ('c_int', 'info', 0, False)
            else:
                row['runtime_evidence'] = ('invented',)
            with self.subTest(change=change), patch.object(standard, 'BINDINGS', {**standard.BINDINGS, key: row}), patch.object(generator.subprocess, 'run') as formatter:
                with self.assertRaises(schema.ProfileError):
                    generator.subset_outputs('mkl2026.lp64', 'trtrs-ordinary-lp64-v1', standard.TARGET_ID, '/never-invoke')
                formatter.assert_not_called()

    def test_changed_actual_trtrs_engine_rejects_before_formatter(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for name, _ in standard.SOURCE_DEPENDENCIES:
                output = root / name
                output.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(ROOT / name, output)
            name = standard.RESERVED_BLAS_MANIFEST[0]
            (root / name).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / name, root / name)
            changed = root / 'src/interop/lapacke_trtrs.zig'
            changed.write_bytes(changed.read_bytes() + b'\n')
            with patch.object(schema, '_binding_source_root', return_value=root), patch.object(generator.subprocess, 'run') as formatter:
                with self.assertRaisesRegex(schema.ProfileError, 'Accepted source dependency missing or changed: src/interop/lapacke_trtrs.zig'):
                    generator.subset_outputs('mkl2026.ilp64', 'trtrs-ordinary-ilp64-v1', standard.TARGET_ID, '/never-invoke')
                formatter.assert_not_called()

    def test_combined_linker_rejects_standalone_unknown_and_changed_trtrs_exports_before_link(self):
        name = 'potrf-potrs-getrf-getrs-gesv-stev-gelsd-heev-hegv-posv-trtrs-ordinary-lp64-v1'
        path = ROOT / 'interface_subset/mkl2026.lp64' / name / 'abi_manifest.json'
        for change in ('standalone', 'unknown', 'missing', 'extra', 'duplicate'):
            value = json.loads(path.read_text())
            if change == 'standalone':
                value['plan']['set'] = 'trtrs-ordinary-lp64-v1'
            elif change == 'unknown':
                value['plan']['set'] = 'unreviewed-posv-trtrs-ordinary-lp64-v1'
            elif change == 'missing':
                value['plan']['symbols'].remove('LAPACKE_dtrtrs')
            elif change == 'extra':
                value['plan']['symbols'].append('LAPACKE_unreviewed')
            else:
                value['plan']['symbols'].remove('LAPACKE_dtrtrs')
                value['plan']['symbols'].append('LAPACKE_strtrs')
            with self.subTest(change=change), tempfile.TemporaryDirectory() as temp:
                manifest = Path(temp) / 'manifest.json'
                manifest.write_text(json.dumps(value))
                output = Path(temp) / 'uncreated.dylib'
                result = subprocess.run([sys.executable, '-B', str(ROOT / 'tools/link_darwin_interface.py'), '/never-read.a', str(ROOT / 'include/zynum/checked/sparse_manifest.json'), str(output), '--macos-min', '27.0.1', '--legacy-manifest', str(ROOT / 'include/zynum/checked/abi_manifest.json'), '--lapacke-manifest', str(manifest)], capture_output=True, text=True)
                self.assertNotEqual(0, result.returncode)
                self.assertIn('unsupported combined LP64 LAPACKE contract' if change in ('standalone', 'unknown') else 'invalid combined LP64 LAPACKE export list', result.stderr)
                self.assertFalse(output.exists())

    def test_current_generated_files_and_thin_default_context_forwarding_are_exact(self):
        template = (ROOT / 'tools/interface_templates/trtrs.zig.in').read_text()
        self.assertEqual(1, template.count('binding.Shared.trtrsRawManagedWithDefaults('))
        self.assertEqual(1, template.count('binding.Provider.Execution.getContext()'))
        self.assertNotIn('@import(', template)
        self.assertNotIn('Context.init', template)
        for name, row in standard.SETS.items():
            if 'trtrs-ordinary-' not in name:
                continue
            with self.subTest(selection=name):
                outputs = generator.subset_outputs(row['profile'], name, standard.TARGET_ID)
                for path, content in outputs.items():
                    self.assertEqual((ROOT / path).read_text(), content, path)
                manifest = json.loads(next(content for path, content in outputs.items() if path.endswith('/abi_manifest.json')))
                self.assertFalse(manifest['trtrs_numerical_scope']['abplas_adoption'])
                self.assertFalse(manifest['trtrs_numerical_scope']['vendor_nonfinite_payload_rounding_equivalence'])


if __name__ == '__main__':
    unittest.main()
