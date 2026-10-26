# Copyright (C) 2026 Zynum contributors
# SPDX-License-Identifier: LGPL-3.0-or-later
import copy
import json
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'tools'))
import generate_lapacke_component as generator
import lapacke_components as components
import standard_abi_bindings as standard

class PotriComponentTests(unittest.TestCase):
    def test_both_profiles_render_exact_original_five_argument_declarations_and_eight_forwarders(self):
        wanted = ['LAPACKE_' + tag + 'potri' + suffix for tag in 'sdcz' for suffix in ('', '_work')]
        for profile in components.PROFILES:
            output = generator.outputs(profile, 'potri', components.TARGET)
            self.assertEqual(set(output), {'exports.zig', 'lapacke_potri.h', 'lapacke.h', 'abi_manifest.json', 'exports.def'})
            manifest = json.loads(output['abi_manifest.json'])
            self.assertEqual(manifest['plan']['symbols'], wanted)
            self.assertEqual(re.findall(r'\.name = "(LAPACKE_\w+)"', output['exports.zig']), wanted)
            self.assertEqual(re.findall(r'lapack_int (LAPACKE_\w+)\(', output['lapacke_potri.h']), wanted)
            self.assertTrue(all(row['parameter_count'] == 5 for row in manifest['plan']['bindings']))
            self.assertNotIn('lwork', output['lapacke_potri.h'])
            self.assertFalse(manifest['plan']['full_profile_ready'])
            self.assertEqual(generator.outputs(profile, 'potri', components.TARGET), output)

    def test_invalid_component_profile_and_target_are_rejected_before_formatter(self):
        with mock.patch.object(generator.subprocess, 'run', side_effect=AssertionError('formatter reached')):
            for args in [('dfti', 'potri', components.TARGET), ('mkl2026.lp64', 'getri', components.TARGET),
                         ('mkl2026.lp64', 'potri', 'linux'), (None, 'potri', components.TARGET)]:
                with self.subTest(args=args), self.assertRaises(ValueError):
                    generator.outputs(*args)

    def test_forged_component_owners_signatures_targets_or_runtime_claims_are_rejected(self):
        original = components.manifest_for('mkl2026.lp64', 'potri', components.TARGET)
        for change in ('owner', 'signature', 'target', 'runtime', 'symbols'):
            bad = copy.deepcopy(original)
            if change == 'owner': bad['plan']['owner_imports'][0]['instantiate'] = True
            elif change == 'signature': bad['plan']['bindings'][0]['parameter_count'] = 7
            elif change == 'target': bad['plan']['target']['pointer_bits'] = 32
            elif change == 'runtime': bad['plan']['runtime_abi_status'] = 'qualified'
            else: bad['plan']['symbols'].append('LAPACKE_sgetri')
            with self.subTest(change=change), self.assertRaises(ValueError):
                components.validate_component(bad, 'mkl2026.lp64')

    def test_every_existing_accepted_base_can_compose_without_copying_its_bindings(self):
        count = 0
        for profile, name, target in standard.GENERATION_ACCEPTANCES:
            path = ROOT / 'interface_subset' / profile / name / 'abi_manifest.json'
            base = json.loads(path.read_text())
            component = components.manifest_for(profile, 'potri', target)
            composed = components.compose(base, [component])
            unwrapped, extra = components.unwrap(composed)
            self.assertEqual(unwrapped, base)
            self.assertEqual(extra, component['plan']['symbols'])
            self.assertEqual(composed['plan']['symbols'], base['plan']['symbols'] + extra)
            count += 1
        self.assertEqual(count, 238)

    def test_duplicate_cross_width_and_forged_base_compositions_are_rejected(self):
        name = 'potrf-potrs-getrf-getrs-ordinary-lp64-v1'
        base = json.loads((ROOT / 'interface_subset/mkl2026.lp64' / name / 'abi_manifest.json').read_text())
        component = components.manifest_for('mkl2026.lp64', 'potri', components.TARGET)
        wide = components.manifest_for('mkl2026.ilp64', 'potri', components.TARGET)
        for items in ([], [component, component], [wide]):
            with self.subTest(items=len(items)), self.assertRaises(ValueError):
                components.compose(base, items)
        bad = copy.deepcopy(base)
        bad['plan']['symbols'].append('LAPACKE_snotreal')
        with self.assertRaises(ValueError): components.compose(bad, [component])
        bad = components.compose(base, [component])
        bad['plan']['symbols'].pop()
        with self.assertRaises(ValueError): components.unwrap(bad)

    def test_cli_rejects_unreviewed_generation_and_forged_composition_before_output(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            output = root / 'generated'
            result = subprocess.run([sys.executable, '-B', str(ROOT / 'tools/generate_lapacke_component.py'),
                '--profile', 'mkl2026.lp64', '--component', 'potri', '--target-contract', 'wrong', '--output', str(output)], capture_output=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertFalse(output.exists())
            base = root / 'base.json'; base.write_text(json.dumps({'schema': 'unknown'}))
            component = root / 'component.json'; component.write_text(json.dumps(components.manifest_for('mkl2026.lp64', 'potri', components.TARGET)))
            result = subprocess.run([sys.executable, '-B', str(ROOT / 'tools/generate_lapacke_component.py'),
                '--compose-base', str(base), '--component-manifest', str(component), '--output-manifest', str(output / 'manifest.json')], capture_output=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertFalse(output.exists())

    def test_forwarding_and_composition_root_create_no_numerical_or_resource_owners(self):
        template = (ROOT / 'tools/interface_templates/lapacke_component.zig.in').read_text()
        self.assertEqual(template.count('binding.Shared.potriRawManagedWithDefaults('), 1)
        self.assertEqual(template.count('binding.Provider.Execution.getContext()'), 1)
        for token in ('Context.init', 'std.heap', 'invertNonsingular', 'thread_pool'):
            self.assertNotIn(token, template)
        root = (ROOT / 'src/lapacke_composed.zig').read_text()
        self.assertIn('extra.OwnerIdentity != Core', root)
        self.assertIn('WireInt != extra.WireInt', root)
        self.assertIn('component_check', (ROOT / 'build.zig').read_text())

if __name__ == '__main__':
    unittest.main()
