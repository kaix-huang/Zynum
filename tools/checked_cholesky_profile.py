# Copyright (C) 2026 Zynum contributors
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Closed additive Cholesky declaration contract; execution evidence is separate."""
import copy
from checked_complex_wire import COMPLEX_LAYOUTS
from numerical_operation_schema import OPERATIONS

PROFILE = 'checked-cholesky-v1'
TARGET = 'apple-arm64-default-c-v1'


def candidate(profile=PROFILE, target=TARGET):
    if (profile, target) != (PROFILE, TARGET):
        raise ValueError('Unknown checked Cholesky profile or target')
    bindings = []
    for scalar in ('complex_f32', 'complex_f64'):
        for adapter, operation in (('factor', 'cholesky.factor'), ('solve', 'cholesky.solve')):
            if scalar not in OPERATIONS[operation]['scalar_types']:
                raise ValueError('Cholesky scalar exceeds numerical domain')
            parameters = (('a', 'dense', 'matrix'), ('triangle', 'i32', 'triangle')) if adapter == 'factor' else (
                ('a', 'const_dense', 'factor'), ('b', 'dense', 'rhs'), ('triangle', 'i32', 'triangle'))
            if tuple(row[2] for row in parameters) != OPERATIONS[operation]['operand_roles']:
                raise ValueError('Cholesky operand role drift')
            bindings.append(dict(symbol='zynum_checked_cholesky_' + adapter + '_' + scalar,
                                 operation=operation, adapter=adapter, scalar=scalar,
                                 parameters=parameters, result='result'))
    return copy.deepcopy(dict(schema='zynum-checked-cholesky-v1', profile=PROFILE, target=TARGET,
        pointer_bits=64, shared_abi='checked-v1', complex_layouts=COMPLEX_LAYOUTS,
        status_extensions={'INVALID_HERMITIAN_DIAGONAL': 15},
        bindings=bindings, symbols=[row['symbol'] for row in bindings],
        matrix_layout='column-major', buffer_length_unit='scalar-elements',
        ownership='caller-owned-factor-and-rhs; shared-default-Context-per-call',
        qualification='declarations-only; numerical and runtime ABI evidence is separate'))


def validate(value):
    # Exact closed catalog, including integer representation (bool != integer).
    import json
    expected = candidate()
    if not isinstance(value, dict) or json.dumps(value, sort_keys=True) != json.dumps(expected, sort_keys=True):
        raise ValueError('Unsupported checked Cholesky declaration contract')
    return copy.deepcopy(value)
