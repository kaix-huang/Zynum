# Copyright (C) 2026 Zynum contributors
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Checked-v1 wire representation and existing-adapter bindings only."""
ABI_REVISION = 1
POINTER_BITS = 64
# Linker-provided data exports are recorded separately from the eight C APIs.
OBJECT_SUPPORT_SYMBOLS = {'macho': ['__dso_handle', '_mh_dylib_header'], 'elf': [], 'coff': []}
# Explicit target data model; generator host ctypes/layout is not authoritative.
PRIMITIVE_LAYOUT = {'i32': (4, 4), 'i64': (8, 8), 'ptr': (8, 8), 'const_ptr': (8, 8)}
PREFIX = 'zynum_checked_'
STATUS = {
    'OK': 0, 'INVALID_FLAG': 1, 'INVALID_DIMENSION': 2,
    'INVALID_LEADING_DIMENSION': 3, 'BUFFER_TOO_SMALL': 4,
    'NULL_POINTER': 5, 'MISALIGNED_POINTER': 6, 'SIZE_OVERFLOW': 7,
    'DIMENSION_MISMATCH': 8, 'ALIASING_NOT_ALLOWED': 9,
    'INVALID_SPARSE_STRUCTURE': 10, 'NONFINITE_INPUT': 11,
    'NUMERICAL_FAILURE': 12, 'INVALID_LENGTH': 13, 'INTERNAL_ERROR': 14,
}
FLAGS = {'LOWER': 1, 'UPPER': 2, 'NORMAL': 0, 'TRANSPOSED': 1}
# Field order is ABI. Pointer type also encodes constness for C and Zig.
STRUCTS = {
    'result': [('status', 'i32'), ('reserved', 'i32'), ('info', 'i64')],
    'buffer': [('data', 'ptr'), ('length', 'i64')],
    'const_buffer': [('data', 'const_ptr'), ('length', 'i64')],
    'dense': [('storage', 'buffer'), ('rows', 'i64'), ('cols', 'i64'), ('ld', 'i64')],
    'const_dense': [('storage', 'const_buffer'), ('rows', 'i64'), ('cols', 'i64'), ('ld', 'i64')],
    'csr': [('row_offsets', 'const_buffer'), ('column_indices', 'const_buffer'),
            ('values', 'const_buffer'), ('rows', 'i64'), ('cols', 'i64')],
}

# Reviewed bridge to the existing adapter file, not a numerical dependency or
# native ABI proof. Roles refer to numerical_operation_schema, while argument
# names, layouts, flags/statuses and call spelling belong to this profile.
ADAPTER_SOURCE = {'path': 'src/checked/implementation.zig', 'sha256': 'bf05e8adfac3a678ab2db7a2b5fa8588a8ebba105ce358d7e8c27e6d238090c5'}
ADAPTERS = {
    'factor': {
        'operation': 'cholesky.factor', 'symbol_stem': 'cholesky_factor',
        'scalar_types': ('f32', 'f64'), 'index_types': (),
        'type_arguments': ('scalar',),
        'arguments': (('a', 'dense', 'matrix'), ('triangle', 'i32', 'triangle')),
    },
    'solve': {
        'operation': 'cholesky.solve', 'symbol_stem': 'cholesky_solve',
        'scalar_types': ('f32', 'f64'), 'index_types': (),
        'type_arguments': ('scalar',),
        'arguments': (('a', 'const_dense', 'factor'), ('b', 'dense', 'rhs'), ('triangle', 'i32', 'triangle')),
    },
    'spmv': {
        'operation': 'sparse.mv', 'symbol_stem': 'csr_spmv',
        'scalar_types': ('f32', 'f64'), 'index_types': ('u32', 'u64'),
        'type_arguments': ('scalar', 'index'),
        'arguments': (('a', 'csr', 'matrix'), ('operation', 'i32', 'operation'),
                      ('alpha', 'scalar', 'product_scale'), ('x', 'const_buffer', 'input'),
                      ('beta', 'scalar', 'result_scale'), ('y', 'buffer', 'result')),
    },
}

# Binding IDs are opaque. Membership comes only from the explicit profile field.
BINDINGS = []
for scalar in ('f32', 'f64'):
    for adapter in ('factor', 'solve', 'spmv'):
        indices = ADAPTERS[adapter]['index_types'] or (None,)
        for index in indices:
            suffix = scalar + ('_' + index if index else '')
            BINDINGS.append({
                'id': 'binding-' + str(len(BINDINGS) + 1),
                'profile': 'checked-v1',
                'operation': ADAPTERS[adapter]['operation'],
                'adapter': adapter,
                'scalar': scalar,
                'index': index,
                'symbol': ADAPTERS[adapter]['symbol_stem'] + '_' + suffix,
            })

CHECKED_PROFILE = {
    'id': 'checked-v1',
    'generation_ready': True,
    'readiness_scope': 'Only generation of the existing reviewed eight checked contracts; no standard API/ABI compatibility or new execution evidence.',
    'unresolved_generation_inputs': (),
    'renderer': 'checked-v1',
    'OBJECT_SUPPORT_SYMBOLS': OBJECT_SUPPORT_SYMBOLS,
    'ABI_REVISION': ABI_REVISION,
    'POINTER_BITS': POINTER_BITS,
    'PRIMITIVE_LAYOUT': PRIMITIVE_LAYOUT,
    'PREFIX': PREFIX,
    'STATUS': STATUS,
    'FLAGS': FLAGS,
    'STRUCTS': STRUCTS,
    'adapter_source': ADAPTER_SOURCE,
    'adapters': ADAPTERS,
    'bindings': BINDINGS,
}
