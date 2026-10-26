# Copyright (C) 2026 Zynum contributors
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Reject malformed additive profiles before any external formatter executes."""
import copy
import tempfile
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'tools'))
from checked_sparse_profile import candidate, PROFILE_ID, SET_ID, TARGET_ID
from checked_sparse_schema import validate_candidate, SparseProfileError
from generate_checked_sparse import render_candidate


def fresh():
    return candidate(PROFILE_ID, SET_ID, TARGET_ID)


class CheckedSparseCandidateTests(unittest.TestCase):
    def test_catalog_is_not_shared_mutable_state(self):
        first = fresh()
        first['complex_layouts']['complex_f64']['size'] = 1
        first['bindings'][0]['symbol'] = 'corrupt'
        self.assertEqual(validate_candidate(fresh())['complex_layouts']['complex_f64']['size'], 16)

    def test_validation_returns_independent_copy(self):
        value = fresh()
        validated = validate_candidate(value)
        validated['bindings'][0]['symbol'] = 'corrupt'
        self.assertNotEqual(value['bindings'][0]['symbol'], 'corrupt')

    def test_invalid_selection(self):
        selection = [PROFILE_ID, SET_ID, TARGET_ID]
        for index in range(3):
            bad = selection.copy()
            bad[index] = 'unknown'
            with self.assertRaises(ValueError):
                candidate(*bad)

    def test_invalid_contract_never_runs_formatter(self):
        mutations = [
            lambda p: p.update(generation_accepted=True),
            lambda p: p.update(pointer_bits=32),
            lambda p: p['flags'].update(ADJOINT=3),
            lambda p: p['shared_abi'].update(status_owner='vendor'),
            lambda p: p['complex_layouts']['complex_f64']['offsets'].update(re=False),
            lambda p: p['complex_layouts']['complex_f32'].update(size=8.0),
            lambda p: p['bindings'][0].update(offset='i64'),
            lambda p: p['bindings'][0].update(ordinal='i64'),
            lambda p: p['bindings'][0].update(adapter='spmv'),
            lambda p: p['bindings'][0].update(type_arguments=('i32', 'f32', 'i32')),
            lambda p: p['bindings'].pop(),
            lambda p: p['bindings'][1].update(symbol=p['bindings'][0]['symbol']),
        ]
        for index, mutate in enumerate(mutations):
            with self.subTest(index=index):
                value = fresh()
                mutate(value)
                with patch('generate_checked_sparse.subprocess.run') as formatter:
                    with self.assertRaises(SparseProfileError):
                        render_candidate(value, 'must-not-execute')
                    formatter.assert_not_called()

    def test_no_missing_or_extra_type_combinations(self):
        value = validate_candidate(fresh())
        actual = {(r['scalar'], r['offset'], r['ordinal'], r['operation']) for r in value['bindings']}
        expected = {(s, o, i, op) for s in ('f32', 'f64', 'complex_f32', 'complex_f64')
                    for o in ('i32', 'u32', 'i64', 'u64')
                    for i in ('i32', 'u32', 'i64', 'u64')
                    for op in ('sparse.mv', 'sparse.mm')}
        self.assertEqual(actual, expected)
        self.assertEqual(len(value['symbols']), 128)


class ReviewedSparseSourceTests(unittest.TestCase):
    def test_current_reviewed_sources(self):
        from checked_sparse_sources import verify_sources
        sources = verify_sources()['sha256']
        self.assertEqual(len(sources), 21)
        self.assertIn('tools/checked_complex_wire.py', sources)
        self.assertIn('src/numerics/operator.zig', sources)
        self.assertIn('src/sparse/operator.zig', sources)

    def test_missing_and_changed_sources_reject_before_formatter(self):
        from checked_sparse_sources import ROOT, REVIEWED_SOURCES, verify_sources
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for relative in REVIEWED_SOURCES:
                target = root / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes((ROOT / relative).read_bytes())
            for relative in REVIEWED_SOURCES:
                target = root / relative
                original = target.read_bytes()
                for changed in (False, True):
                    with self.subTest(source=relative, changed=changed):
                        if changed:
                            target.write_bytes(original + b'\n// drift\n')
                        else:
                            target.unlink()
                        with patch('generate_checked_sparse.verify_sources', side_effect=lambda: verify_sources(root)):
                            with patch('generate_checked_sparse.subprocess.run') as formatter:
                                with self.assertRaisesRegex(SparseProfileError, 'changed or missing'):
                                    render_candidate(fresh(), 'must-not-execute')
                                formatter.assert_not_called()
                        target.write_bytes(original)


class SparseGeneratorCommandTests(unittest.TestCase):
    def test_unknown_selection_leaves_output_root_untouched(self):
        import subprocess
        script = Path(__file__).resolve().parents[2] / 'tools/generate_checked_sparse.py'
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / 'output'
            result = subprocess.run([sys.executable, str(script), '--profile', 'unknown',
                                     '--binding-set', SET_ID, '--target-contract', TARGET_ID,
                                     '--zig', '/must-not-execute', '--root', str(root)],
                                    capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('Unknown checked Sparse', result.stderr)
            self.assertFalse(root.exists())

    def test_check_reports_all_drift_without_writes(self):
        import contextlib
        import io
        from generate_checked_sparse import main
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'existing').write_text('preserve')
            with patch('generate_checked_sparse.render_candidate', return_value={'a': 'one', 'b': 'two'}):
                error = io.StringIO()
                with contextlib.redirect_stderr(error), self.assertRaises(SystemExit):
                    main(['--profile', PROFILE_ID, '--binding-set', SET_ID,
                          '--target-contract', TARGET_ID, '--root', str(root), '--check'])
            self.assertIn('a, b', error.getvalue())
            self.assertEqual([f.name for f in root.iterdir()], ['existing'])
            self.assertEqual((root / 'existing').read_text(), 'preserve')


if __name__ == '__main__':
    unittest.main()
