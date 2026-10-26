# Copyright (C) 2026 Zynum contributors
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Finite POSV declarations, exact source authorization and early rejection."""
from pathlib import Path
import copy
import json
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'tools'))
import checked_abi_schema as schema
import standard_abi_bindings as standard
import generate_checked_abi as generator


class PosvProfileTests(unittest.TestCase):
    def test_exact_four_scalar_two_width_wire_and_eleven_selections(self):
        selections = [(key, row) for key, row in standard.SETS.items() if '-posv-ordinary-' in key]
        self.assertEqual(11, len(selections))
        for name, row in selections:
            with self.subTest(name=name):
                plan = schema.select_binding_set(row['profile'], name, standard.TARGET_ID)
                bindings = [b for b in plan['bindings'] if b['operation'] == 'cholesky.driver']
                self.assertEqual(8, len(bindings))
                width = 32 if row['profile'] == 'mkl2026.lp64' else 64
                self.assertEqual({('f32','high'),('f32','work'),('f64','high'),('f64','work'),('complex_f32','high'),('complex_f32','work'),('complex_f64','high'),('complex_f64','work')}, {(b['scalar'],b['level']) for b in bindings})
                for binding in bindings:
                    self.assertEqual(('matrix_layout','uplo','n','nrhs','a','lda','b','ldb'), tuple(p['name'] for p in binding['wire_parameters']))
                    self.assertEqual('int' if width == 32 else 'long long int', binding['wire_return']['c_type'])
                    self.assertEqual('int', binding['wire_parameters'][0]['c_type'])
                    self.assertEqual((False,False), tuple(p['pointee_const'] for p in binding['wire_parameters'] if p['pointer_depth']))
                    self.assertEqual('raw_posv_profile_defaults', binding['recipe'])
                self.assertFalse(plan['full_profile_ready'])
                self.assertEqual('pending', plan['runtime_abi_status'])
                with self.assertRaises(TypeError): plan['full_profile_ready'] = True

    def test_unsupported_cli_selection_never_invokes_formatter_or_creates_output(self):
        good='potrf-potrs-posv-ordinary-lp64-v1'
        invalid=[('mkl2026.ilp64',good,standard.TARGET_ID), ('mkl2026.lp64',good,'linux-x86_64'), ('mkl2026.lp64','posv-ordinary-lp64-v1',standard.TARGET_ID), ('mkl2026.lp64','geqrf-ormqr-unmqr-posv-ordinary-lp64-v1',standard.TARGET_ID)]
        for profile,selection,target in invalid:
            with self.subTest(selection=selection,profile=profile),tempfile.TemporaryDirectory() as temp:
                output=Path(temp)/'uncreated'
                result=subprocess.run([sys.executable,'-B',str(ROOT/'tools/generate_checked_abi.py'),'--profile',profile,'--binding-set',selection,'--target-contract',target,'--root',str(output),'--zig','/never-invoke'],capture_output=True,text=True)
                self.assertNotEqual(0,result.returncode)
                self.assertIn('error:',result.stderr)
                self.assertNotIn('FileNotFoundError',result.stderr)
                self.assertFalse(output.exists())

        # The combined linker also rejects an unknown set or changed export
        # membership before invoking clang or touching the requested output.
        manifest_path=ROOT/'interface_subset/mkl2026.lp64/potrf-potrs-getrf-getrs-gesv-stev-gelsd-heev-hegv-posv-ordinary-lp64-v1/abi_manifest.json'
        for change in ('unknown','missing','extra'):
            value=json.loads(manifest_path.read_text())
            if change=='unknown': value['plan']['set']='unreviewed-posv-ordinary-lp64-v1'
            elif change=='missing': value['plan']['symbols'].remove('LAPACKE_dposv')
            else: value['plan']['symbols'].append('LAPACKE_unreviewed')
            with self.subTest(linker_change=change),tempfile.TemporaryDirectory() as temp:
                bad=Path(temp)/'manifest.json';bad.write_text(json.dumps(value));output=Path(temp)/'uncreated.dylib'
                result=subprocess.run([sys.executable,'-B',str(ROOT/'tools/link_darwin_interface.py'),'/never-read.a',str(ROOT/'include/zynum/checked/sparse_manifest.json'),str(output),'--macos-min','27.0.1','--legacy-manifest',str(ROOT/'include/zynum/checked/abi_manifest.json'),'--lapacke-manifest',str(bad)],capture_output=True,text=True)
                self.assertNotEqual(0,result.returncode)
                self.assertIn('unsupported combined LP64 LAPACKE contract' if change=='unknown' else 'invalid combined LP64 LAPACKE export list',result.stderr)
                self.assertFalse(output.exists())

    def test_changed_order_mutability_or_signature_is_not_self_authorizing(self):
        name='potrf-potrs-posv-ordinary-lp64-v1'
        key='posv-lp64-d-high'
        original=dict(standard.BINDINGS[key])
        for change in ('const','order','declaration','recipe','semantics','info'):
            row=copy.deepcopy(original)
            if change=='const':
                params=list(row['parameters']);params[4]=(*params[4][0:4],True);row['parameters']=tuple(params)
            elif change=='order': row['parameters']=(row['parameters'][1],row['parameters'][0],*row['parameters'][2:])
            elif change=='declaration': row['declaration']=(*row['declaration'][:-1],'0'*64)
            elif change=='recipe': row['recipe']='raw_solve_profile_defaults'
            elif change=='semantics': row['semantic_contracts']=standard.SEMANTIC_CONTRACTS
            elif change=='info': row['return']=('c_int','info',0,False)
            with self.subTest(change=change),patch.object(standard,'BINDINGS',{**standard.BINDINGS,key:row}),patch.object(generator.subprocess,'run') as formatter:
                with self.assertRaises(schema.ProfileError): generator.subset_outputs('mkl2026.lp64',name,standard.TARGET_ID,'/never-invoke')
                formatter.assert_not_called()

    def test_actual_source_drift_rejects_before_formatter(self):
        name='potrf-potrs-posv-ordinary-ilp64-v1'
        with tempfile.TemporaryDirectory() as temp,patch.object(schema,'_binding_source_root',return_value=Path(temp)),patch.object(generator.subprocess,'run') as formatter:
            with self.assertRaises(schema.ProfileError): generator.subset_outputs('mkl2026.ilp64',name,standard.TARGET_ID,'/never-invoke')
            formatter.assert_not_called()

    def test_current_posv_generated_files_are_exact(self):
        for name,row in standard.SETS.items():
            if '-posv-ordinary-' not in name: continue
            with self.subTest(name=name):
                generated=generator.subset_outputs(row['profile'],name,standard.TARGET_ID)
                for path,content in generated.items(): self.assertEqual((ROOT/path).read_text(),content,path)
                manifest=json.loads(next(v for k,v in generated.items() if k.endswith('/abi_manifest.json')))
                self.assertFalse(manifest['posv_numerical_scope']['abplas_adoption'])


if __name__=='__main__': unittest.main()
