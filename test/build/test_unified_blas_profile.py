# Copyright (C) 2026 Zynum contributors
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Reject forged joint BLAS declarations before invoking the linker."""
import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]


class UnifiedBlasProfileTests(unittest.TestCase):
    def test_forged_names_widths_counts_and_owners_never_create_output(self):
        original = json.loads((ROOT/'include/zynum/blas/abi_manifest.json').read_text())
        mutations = []
        for group in ('cblas','fortran'):
            for field, value in [('export_count',0),('source_paths',['src/fake-owner.zig'])]:
                forged = copy.deepcopy(original)
                forged[group][field] = value
                mutations.append((group+'-'+field,forged))
            for action in ('missing','extra','renamed','duplicate','return-width','argument-width'):
                forged = copy.deepcopy(original)
                rows = forged[group]['exports']
                if action=='missing': rows.pop()
                elif action=='extra': rows.append(copy.deepcopy(rows[0]))
                elif action=='renamed': rows[0]['name']='cblas_fake' if group=='cblas' else 'fake_'
                elif action=='duplicate': rows[0]['name']=rows[1]['name']
                elif action=='return-width': rows[0]['return_zig']='i64'
                elif action=='argument-width': rows[0]['args'][0]['zig_type']='*const i64'
                mutations.append((group+'-'+action,forged))
        for field,value in [('schema',2),('blas_integer_abi','ILP64')]:
            forged = copy.deepcopy(original)
            forged[field] = value
            mutations.append((field,forged))
        with tempfile.TemporaryDirectory() as directory:
            temp = Path(directory)
            manifest = temp/'blas.json'
            output = temp/'uncreated/lib.dylib'
            for label,forged in mutations:
                with self.subTest(label=label):
                    manifest.write_text(json.dumps(forged))
                    command = [sys.executable,str(ROOT/'tools/link_darwin_interface.py'),
                               str(temp/'never-read.a'),str(ROOT/'include/zynum/checked/sparse_manifest.json'),
                               str(output),'--macos-min','14.0.0',
                               '--legacy-manifest',str(ROOT/'include/zynum/checked/abi_manifest.json'),
                               '--lapacke-manifest',str(ROOT/'interface_subset/mkl2026.lp64/potrf-potrs-getrf-getrs-gesv-stev-gelsd-heev-hegv-ordinary-lp64-v1/abi_manifest.json'),
                               '--blas-manifest',str(manifest)]
                    result = subprocess.run(command,capture_output=True,text=True)
                    self.assertNotEqual(0,result.returncode)
                    self.assertIn('unsupported existing LP64 BLAS declaration contract',result.stderr)
                    self.assertFalse(output.parent.exists())

    def test_standalone_interface_cannot_claim_joint_BLAS_owner(self):
        with tempfile.TemporaryDirectory() as directory:
            temp = Path(directory)
            output = temp/'uncreated/lib.dylib'
            command = [sys.executable,str(ROOT/'tools/link_darwin_interface.py'),
                       str(temp/'never-read.a'),
                       str(ROOT/'interface_subset/mkl2026.lp64/potrf-potrs-getrf-getrs-gesv-stev-gelsd-heev-hegv-ordinary-lp64-v1/abi_manifest.json'),
                       str(output),'--macos-min','14.0.0',
                       '--blas-manifest',str(ROOT/'include/zynum/blas/abi_manifest.json')]
            result = subprocess.run(command,capture_output=True,text=True)
            self.assertNotEqual(0,result.returncode)
            self.assertIn('BLAS requires the unified one-owner LP64 package',result.stderr)
            self.assertFalse(output.parent.exists())


if __name__=='__main__':
    unittest.main()
