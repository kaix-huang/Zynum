# Copyright (C) 2026 Zynum contributors
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Closed validation of the additive Sparse declaration candidate.

This validates wire descriptions only. It does not authorize public generation,
claim runtime ABI support, or replace the existing checked-v1 selector.
"""
import copy
import itertools


class SparseProfileError(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise SparseProfileError(message)


def validate_candidate(value):
    require(isinstance(value, dict), 'Sparse candidate must be an object')
    require(set(value) == {'profile', 'set', 'target', 'pointer_bits', 'shared_abi',
                          'complex_layouts', 'flags', 'bindings', 'symbols',
                          'generation_accepted', 'runtime_abi_status', 'source_scope'}, 'Sparse candidate fields')
    require((value['profile'], value['set'], value['target']) ==
            ('checked-sparse-v1', 'csr-products-independent-indices-v1', 'apple-arm64-default-c-v1'),
            'Sparse selection mismatch')
    require(type(value['pointer_bits']) is int and value['pointer_bits'] == 64, 'Sparse pointer model')
    require(value['generation_accepted'] is False and value['runtime_abi_status'] == 'pending' and
            value['source_scope'] == 'candidate catalog only; dependency envelope not yet bound', 'Candidate cannot claim acceptance')
    require(value['shared_abi'] == {'header': 'zynum/checked/checked.h', 'zig_source': 'src/checked/abi_types.zig',
                                  'types': ('result', 'const_buffer', 'buffer', 'csr', 'const_dense', 'dense'),
                                  'status_owner': 'checked-v1'}, 'Shared ABI ownership mismatch')
    require(value['flags'] == {'NORMAL': 0, 'TRANSPOSED': 1, 'ADJOINT': 2} and
            all(type(v) is int for v in value['flags'].values()), 'Sparse transform wire values')
    require(value['complex_layouts'] == {
        'complex_f32': {'component': 'float', 'size': 8, 'alignment': 4, 'offsets': {'re': 0, 'im': 4}},
        'complex_f64': {'component': 'double', 'size': 16, 'alignment': 8, 'offsets': {'re': 0, 'im': 8}},
    }, 'Complex wire layout mismatch')
    for layout in value['complex_layouts'].values():
        require(all(type(n) is int for n in (layout['size'], layout['alignment'],
                                            *layout['offsets'].values())),
                'Complex layout requires integer byte counts')
    scalars = ('f32', 'f64', 'complex_f32', 'complex_f64')
    indices = ('i32', 'u32', 'i64', 'u64')
    wire_scalars = ('float', 'double', 'zynum_checked_complex_f32', 'zynum_checked_complex_f64')
    core_scalars = ('f32', 'f64', 'core.sparse.ComplexF32', 'core.sparse.ComplexF64')
    rows = value['bindings']
    require(isinstance(rows, list) and len(rows) == 128, 'Sparse binding count')
    keys = {'symbol', 'operation', 'adapter', 'scalar', 'offset', 'ordinal',
            'scalar_c_type', 'scalar_zig_type', 'type_arguments', 'parameters', 'result'}
    symbols = []
    for row, (scalar, offset, ordinal, operation) in zip(rows, itertools.product(scalars, indices, indices, ('sparse.mv', 'sparse.mm'))):
        require(isinstance(row, dict) and set(row) == keys, 'Sparse binding fields')
        require((row['scalar'], row['offset'], row['ordinal'], row['operation']) ==
                (scalar, offset, ordinal, operation), 'Sparse coverage/type order')
        matrix = operation == 'sparse.mm'
        stem = 'spmm' if matrix else 'spmv'
        symbol = f'zynum_checked_csr_{stem}_{scalar}_{offset}_{ordinal}'
        require(row['symbol'] == symbol, 'Sparse symbol/type mismatch')
        require(row['adapter'] == ('spmmWithIndexTypes' if matrix else 'spmvWithIndexTypes'), 'Sparse adapter mismatch')
        require((row['scalar_c_type'], row['scalar_zig_type']) ==
                (wire_scalars[scalars.index(scalar)], core_scalars[scalars.index(scalar)]), 'Sparse scalar wire/core mismatch')
        require(row['type_arguments'] == (scalar, offset, ordinal), 'Sparse type argument order')
        expected = (('a', 'csr', 'matrix'), ('operation', 'i32', 'operation'),
                    ('alpha', scalar, 'product_scale'), ('x', 'const_dense' if matrix else 'const_buffer', 'input'),
                    ('beta', scalar, 'result_scale'), ('y', 'dense' if matrix else 'buffer', 'result'))
        require(row['parameters'] == expected and row['result'] == 'result', 'Sparse parameter/result contract')
        symbols.append(symbol)
    require(tuple(symbols) == value['symbols'] and len(set(symbols)) == 128, 'Sparse symbol inventory')
    return copy.deepcopy(value)
