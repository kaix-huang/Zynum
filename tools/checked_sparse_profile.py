# Copyright (C) 2026 Zynum contributors
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Additive checked Sparse wire catalog; no generation acceptance or runtime claim."""
import copy
from numerical_operation_schema import OPERATIONS

PROFILE_ID = 'checked-sparse-v1'
SET_ID = 'csr-products-independent-indices-v1'
TARGET_ID = 'apple-arm64-default-c-v1'
INDEX_TYPES = ('i32', 'u32', 'i64', 'u64')
SCALARS = {
    'f32': ('float', 'f32'),
    'f64': ('double', 'f64'),
    'complex_f32': ('zynum_checked_complex_f32', 'core.sparse.ComplexF32'),
    'complex_f64': ('zynum_checked_complex_f64', 'core.sparse.ComplexF64'),
}
# Explicit target wire layouts; shared descriptors remain checked-v1 aliases.
COMPLEX_LAYOUTS = {
    'complex_f32': {'component': 'float', 'size': 8, 'alignment': 4, 'offsets': {'re': 0, 'im': 4}},
    'complex_f64': {'component': 'double', 'size': 16, 'alignment': 8, 'offsets': {'re': 0, 'im': 8}},
}
FLAGS = {'NORMAL': 0, 'TRANSPOSED': 1, 'ADJOINT': 2}
SHARED_ABI = {'header': 'zynum/checked/checked.h', 'zig_source': 'src/checked/abi_types.zig',
              'types': ('result', 'const_buffer', 'buffer', 'csr', 'const_dense', 'dense'),
              'status_owner': 'checked-v1'}


def candidate(profile_id, set_id, target_id):
    if (profile_id, set_id, target_id) != (PROFILE_ID, SET_ID, TARGET_ID):
        raise ValueError('Unknown checked Sparse profile, set or target')
    bindings = []
    for scalar, (c_type, zig_type) in SCALARS.items():
        for offset in INDEX_TYPES:
            for ordinal in INDEX_TYPES:
                for stem, operation, adapter, input_type, output_type in (
                    ('spmv', 'sparse.mv', 'spmvWithIndexTypes', 'const_buffer', 'buffer'),
                    ('spmm', 'sparse.mm', 'spmmWithIndexTypes', 'const_dense', 'dense'),
                ):
                    domain = OPERATIONS[operation]
                    if scalar not in domain['scalar_types'] or offset not in domain['index_types'] or ordinal not in domain['index_types']:
                        raise ValueError('Checked Sparse binding exceeds the numerical type domain')
                    names = ('a', 'operation', 'alpha', 'x', 'beta', 'y')
                    wire = ('csr', 'i32', scalar, input_type, scalar, output_type)
                    roles = ('matrix', 'operation', 'product_scale', 'input', 'result_scale', 'result')
                    if tuple(domain['operand_roles']) != roles:
                        raise ValueError('Checked Sparse operand role drift')
                    bindings.append({'symbol': f'zynum_checked_csr_{stem}_{scalar}_{offset}_{ordinal}',
                                     'operation': operation, 'adapter': adapter, 'scalar': scalar,
                                     'offset': offset, 'ordinal': ordinal, 'scalar_c_type': c_type,
                                     'scalar_zig_type': zig_type,
                                     'type_arguments': (scalar, offset, ordinal),
                                     'parameters': tuple(zip(names, wire, roles)), 'result': 'result'})
    if len(bindings) != 128 or len({row['symbol'] for row in bindings}) != 128:
        raise ValueError('Checked Sparse binding coverage is incomplete')
    return copy.deepcopy({'profile': profile_id, 'set': set_id, 'target': target_id, 'pointer_bits': 64,
            'shared_abi': SHARED_ABI, 'complex_layouts': COMPLEX_LAYOUTS, 'flags': FLAGS,
            'bindings': bindings, 'symbols': tuple(row['symbol'] for row in bindings),
            'generation_accepted': False, 'runtime_abi_status': 'pending',
            'source_scope': 'candidate catalog only; dependency envelope not yet bound'})
