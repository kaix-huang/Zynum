#!/usr/bin/env python3
# Copyright (C) 2026 Zynum contributors
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Offline schema/policy regressions using synthetic durations; no worker execution."""
import copy
import csv
import importlib.util
import io
from pathlib import Path
import unittest

def load_tool():
    spec = importlib.util.spec_from_file_location('module_controller', Path(__file__).resolve().parents[1] / 'modules/run_foundation_comparison.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


c = load_tool()


def dense(backend='zynum', kind='spotrf', uplo='U', changed=0):
    case = dict(kind=kind, m=32, n=32, nrhs=1 if kind.endswith('potrs') else 0, uplo=uplo)
    row = dict(backend={'zynum': 'zynum-checked', 'openblas': 'openblas-lapacke-work-lp64', 'accelerate': 'accelerate-new-lp64'}[backend],
               **case, lda=32, ldb=32, batch_calls=100000, samples=30,
               median_ns=1, p95_ns=1, best_ns=1, scaled_residual=0,
               check='numerics-padding-ok', factor_unused_triangle_modified_max=changed,
               batch_target_met='true', min_batch_ns=100000, sample_ns=';'.join(['1'] * 30))
    return case, row


def output(row, fields, header=False):
    stream = io.StringIO()
    writer = csv.writer(stream)
    if header:
        writer.writerow(fields)
    writer.writerow(row[k] for k in fields)
    return stream.getvalue().encode()


def checked(backend='zynum', kind='spotrf', changed=0):
    case, row = dense(backend, kind, changed=changed)
    return case, c.validate(output(row, c.DENSE), 'lapack', backend, case)


class ControllerTests(unittest.TestCase):
    def test_exact_case_counts_and_repeat_totals(self):
        self.assertEqual([len(c.cases(m, s)) for m, s in [('lapack','small'), ('lapack','full'), ('sparse','small'), ('sparse','full')]], [28,74,72,144])
        self.assertEqual(28*3*6 + 72*2*4, 1080)
        self.assertEqual(74*3*6 + 144*2*4, 2484)

    def test_dense_round_trip(self):
        _, row = checked()
        self.assertEqual(len(row['sample_ns']), 30)
        self.assertEqual(row['factor_unused_triangle_modified_max'], 0)

    def test_fail_closed_corrupted_dense(self):
        case, good = dense()
        for field, value in [('check','checked-ok'), ('samples',29), ('median_ns',2),
                             ('sample_ns','1;2'), ('best_ns','nan'), ('scaled_residual',101),
                             ('batch_target_met','false'), ('min_batch_ns',99999), ('kind','dpotrf')]:
            with self.subTest(field=field):
                row = dict(good, **{field:value})
                with self.assertRaises(ValueError):
                    c.validate(output(row, c.DENSE), 'lapack', 'zynum', case)
        with self.assertRaises(ValueError):
            c.validate(output(good, c.DENSE) * 2, 'lapack', 'zynum', case)

    def test_storage_exception_is_narrow(self):
        checked('accelerate', changed=10)
        checked('accelerate', 'spotrs', changed=10)
        for backend, uplo, changed in [('zynum','U',1), ('openblas','U',1),
                                       ('accelerate','L',1), ('accelerate','U',497), ('accelerate','U',-1)]:
            case, row = dense(backend, uplo=uplo, changed=changed)
            with self.assertRaises(ValueError):
                c.validate(output(row, c.DENSE), 'lapack', backend, case)

    def test_primary_falls_back_for_timed_deviation_only(self):
        records, selected = [], {'lapack': []}
        for index, kind in enumerate(('spotrf', 'spotrs')):
            selected['lapack'].append(dense(kind=kind)[0])
            for backend, duration in [('zynum',10), ('openblas',8), ('accelerate',2)]:
                _, row = checked(backend, kind, changed=1 if backend == 'accelerate' else 0)
                row['median_ns'] = duration
                for repeat in range(6):
                    records.append(dict(module='lapack', case_index=index, backend=backend, repeat=repeat, result=copy.deepcopy(row)))
        rows, summary = c.aggregate(records, selected)
        primary = summary['lapack:primary-fastest-comparator-with-no-observed-timed-storage-violation']
        self.assertEqual(primary['selected_comparator_by_case'], {'lapack-000':'openblas', 'lapack-001':'accelerate'})
        self.assertEqual(primary['cases'], 2)
        self.assertTrue(primary['full_selected_case_set'])
        self.assertEqual(summary['lapack:numerical-output-only:fastest-comparator']['geometric_mean_speedup'], .2)
        self.assertEqual(sum(row['timed_factor_storage_deviation'] for row in rows), 1)
        self.assertEqual(sum(row['untimed_factor_setup_storage_deviation'] for row in rows), 1)
        with self.assertRaises(ValueError):
            c.aggregate(records[:-1], selected)

    def test_sparse_schema_and_actual_index_layout(self):
        case = c.cases('sparse','small')[0]
        row = dict(case, backend='accelerate', nnz=1000,
                   setup_kind='index_alloc_convert_64offset_32column', setup_ns=1,
                   execution_kind='public_SparseMultiply_borrowed_adapter', samples=30,
                   batch_calls=100000, min_batch_ns=100000, batch_target_met='true',
                   index_layout='i64_offsets_i32_columns', best_ns=1, median_ns=1,
                   p95_ns=1, max_ns=1, gflops=1, check='checked-ok', max_absolute_error=0,
                   scaled_residual=0, padding='checked-ok', checksum=1, sample_ns=';'.join(['1']*30))
        c.validate(output(row, c.SPARSE, True), 'sparse', 'accelerate', case)
        row['index_layout'] = 'i32_offsets_i32_columns'
        with self.assertRaises(ValueError):
            c.validate(output(row, c.SPARSE, True), 'sparse', 'accelerate', case)


if __name__ == '__main__':
    unittest.main()
