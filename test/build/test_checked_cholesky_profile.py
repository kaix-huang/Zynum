# Copyright (C) 2026 Zynum contributors
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Reject altered additive contracts before invoking tools or writing files."""
import copy
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'tools'))
from checked_cholesky_profile import candidate, validate, PROFILE, TARGET
from generate_checked_cholesky import render


class CholeskyProfileTests(unittest.TestCase):
    def test_changed_wire_or_binding_rejects_before_formatter(self):
        plan = candidate()
        variants = []
        for key, value in [('pointer_bits', 32), ('pointer_bits', True),
                           ('profile', 'mkl2026.ilp64'), ('matrix_layout', 'row-major'),
                           ('status_extensions', {'INVALID_HERMITIAN_DIAGONAL': 14})]:
            p = copy.deepcopy(plan); p[key] = value; variants.append(p)
        for change in ('missing', 'duplicate', 'name', 'scalar', 'parameters', 'layout'):
            p = copy.deepcopy(plan)
            if change == 'missing': p['bindings'].pop()
            elif change == 'duplicate': p['bindings'][0] = copy.deepcopy(p['bindings'][1])
            elif change == 'name': p['symbols'][0] = 'zynum_checked_unreviewed'
            elif change == 'scalar': p['bindings'][0]['scalar'] = 'f64'
            elif change == 'parameters': p['bindings'][0]['parameters'] = ()
            elif change == 'layout': p['complex_layouts']['complex_f64']['alignment'] = 16
            variants.append(p)
        for p in variants:
            with self.subTest(plan=p), patch('generate_checked_cholesky.subprocess.run') as formatter:
                with self.assertRaises(ValueError): render(p, '/never-invoke')
                formatter.assert_not_called()
        self.assertEqual(plan, validate(plan))

    def test_unknown_cli_selection_does_not_create_output(self):
        for profile, target in [('unknown', TARGET), (PROFILE, 'unsupported')]:
            with tempfile.TemporaryDirectory() as directory:
                output = Path(directory)/'uncreated'
                result = subprocess.run([sys.executable, str(ROOT/'tools/generate_checked_cholesky.py'),
                    '--profile', profile, '--target-contract', target, '--root', str(output),
                    '--zig', '/never-invoke'], capture_output=True, text=True)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn('Unknown checked Cholesky', result.stderr)
                self.assertFalse(output.exists())

    def test_bridge_drift_rejects_before_formatter(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); bridge=root/'src/checked/implementation.zig'
            bridge.parent.mkdir(parents=True); bridge.write_text('// unreviewed bridge\n')
            with patch('generate_checked_cholesky.ROOT', root), patch('generate_checked_cholesky.subprocess.run') as formatter:
                with self.assertRaisesRegex(ValueError, 'admission source drift'):
                    render(candidate(), '/never-invoke')
                formatter.assert_not_called()


if __name__ == '__main__':
    unittest.main()
