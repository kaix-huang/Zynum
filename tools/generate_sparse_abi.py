# Copyright (C) 2026 Zynum contributors
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Render accepted finite Sparse plans; no numerical implementation or owner state."""
import json
import hashlib
from pathlib import Path
import subprocess
from checked_abi_schema import select_binding_set, _binding_plain, ProfileError


def outputs(profile, binding_set, target_contract, zig='zig'):
    plan = select_binding_set(profile, binding_set, target_contract)
    return _render(plan, zig)


def _render(plan, zig):
    """Pure private renderer used by bounded pre-acceptance validation fixtures."""
    wide = plan['profile'] == 'mkl2026.ilp64'
    width = 'ilp64' if wide else 'lp64'
    guard = 'ZYNUM_FINITE_SPARSE_' + width.upper() + '_V1_H'
    header = ['// Copyright (C) 2026 Zynum contributors',
        '// SPDX-License-Identifier: LGPL-3.0-or-later',
        '/* Generated finite compatibility declarations, not a complete vendor SDK. */',
        '#ifndef ' + guard, '#define ' + guard, '#include <stddef.h>',
        '#if !defined(__APPLE__) || !defined(__aarch64__)',
        '#error "This finite declaration contract requires Apple arm64"', '#endif',
        '#if ' + ('!defined(MKL_ILP64)' if wide else 'defined(MKL_ILP64)'),
        '#error "MKL_ILP64 must match the selected finite Sparse package"', '#endif',
        '/* If also using vendor headers, include mkl_spblas.h before this header. */',
        '#if !defined(_MKL_TYPES_H_)', '#ifndef MKL_Complex8',
        'typedef struct _MKL_Complex8 { float real, imag; } zynum_sparse_complex_float;',
        '#define MKL_Complex8 zynum_sparse_complex_float', '#endif', '#ifndef MKL_Complex16',
        'typedef struct _MKL_Complex16 { double real, imag; } zynum_sparse_complex_double;',
        '#define MKL_Complex16 zynum_sparse_complex_double', '#endif', '#ifndef MKL_INT',
        '#define MKL_INT ' + ('long long int' if wide else 'int'), '#endif', '#endif',
        '#if !defined(_MKL_SPBLAS_H_)']
    for typ, values in plan['enums'].items():
        header.append('typedef enum { ' + ', '.join(name + ' = ' + str(value) for name, value in values) + ' } ' + typ + ';')
    header += ['struct sparse_matrix;', 'typedef struct sparse_matrix *sparse_matrix_t;',
               'struct matrix_descr { sparse_matrix_type_t type; sparse_fill_mode_t mode; sparse_diag_type_t diag; };',
               '#endif', '#ifdef __cplusplus', '#include <type_traits>',
               '#define ZYNUM_SPARSE_ASSERT static_assert', '#define ZYNUM_SPARSE_ALIGN alignof',
               '#else', '#define ZYNUM_SPARSE_ASSERT _Static_assert', '#define ZYNUM_SPARSE_ALIGN _Alignof', '#endif']
    def check(expr, label):
        header.append('ZYNUM_SPARSE_ASSERT(' + expr + ', "' + label + '");')
    check('sizeof(MKL_INT) == ' + str(8 if wide else 4) + ' && ZYNUM_SPARSE_ALIGN(MKL_INT) == ' + str(8 if wide else 4) + ' && ((MKL_INT)-1) < 0', 'signed integer layout')
    check('sizeof(sparse_matrix_t) == 8 && ZYNUM_SPARSE_ALIGN(sparse_matrix_t) == 8', 'handle layout')
    for typ in plan['enums']:
        check('sizeof(' + typ + ') == 4 && ZYNUM_SPARSE_ALIGN(' + typ + ') == 4', 'enum layout')
        header += ['#ifdef __cplusplus', 'ZYNUM_SPARSE_ASSERT((std::is_same<std::underlying_type<' + typ + '>::type, unsigned int>::value), "enum representation");',
                   '#else', 'ZYNUM_SPARSE_ASSERT(_Generic((' + typ + ')0, unsigned int: 1, default: 0), "enum representation");', '#endif']
    check('sizeof(struct matrix_descr) == 12 && ZYNUM_SPARSE_ALIGN(struct matrix_descr) == 4 && offsetof(struct matrix_descr, type) == 0 && offsetof(struct matrix_descr, mode) == 4 && offsetof(struct matrix_descr, diag) == 8', 'descriptor layout')
    for typ, size, align, imag in [('MKL_Complex8', 8, 4, 4), ('MKL_Complex16', 16, 8, 8)]:
        check('sizeof(' + typ + ') == ' + str(size) + ' && ZYNUM_SPARSE_ALIGN(' + typ + ') == ' + str(align) + ' && offsetof(' + typ + ', real) == 0 && offsetof(' + typ + ', imag) == ' + str(imag), 'default complex layout')
        scalar = 'float' if typ == 'MKL_Complex8' else 'double'
        for field in ('real', 'imag'):
            header += ['#ifdef __cplusplus',
                'ZYNUM_SPARSE_ASSERT((std::is_same<decltype(((' + typ + '*)nullptr)->' + field + '), ' + scalar + '>::value), "default complex scalar type");',
                '#else', 'ZYNUM_SPARSE_ASSERT(_Generic(((' + typ + '*)0)->' + field + ', ' + scalar + ': 1, default: 0), "default complex scalar type");', '#endif']
    header += ['#undef ZYNUM_SPARSE_ASSERT', '#undef ZYNUM_SPARSE_ALIGN', '#ifdef __cplusplus', 'extern "C" {', '#endif']
    exports = []
    for row in plan['bindings']:
        params = [('const ' if const else '') + typ + ' ' + '*' * depth + name
                  for typ, name, depth, const in row['parameters']]
        header.append(row['return'] + ' ' + row['symbol'] + '(' + ', '.join(params) + ');')
        typ = {'ComplexF32': 'C', 'ComplexF64': 'Z'}.get(row['zig_scalar'], row['zig_scalar'])
        if row['recipe'] in ('createBorrowed', 'exportBorrowed'):
            axis = {'csr': '.row', 'csc': '.column'}[row['format']]
            entry = 'Entry(' + typ + ', ' + axis + ').' + ('create' if row['recipe'] == 'createBorrowed' else 'exportView')
        elif row['recipe'] == 'matrixVectorMultiply':
            entry = 'Multiply(' + typ + ').call'
        elif row['recipe'] == 'mm':
            entry = 'MatrixMultiply(' + typ + ').call'
        elif row['recipe'] in ('solveVector','solvePanel'):
            entry = ('TriangularVector' if row['recipe']=='solveVector' else 'TriangularPanel') + '(' + typ + ').call'
        elif row['recipe'] == 'destroy':
            entry = 'destroy'
        else:
            raise ProfileError('Unreviewed Sparse renderer recipe')
        exports.append('    @export(&' + entry + ', .{ .name = "' + row['symbol'] + '" });')
    header += ['#ifdef __cplusplus', '}', '#endif', '#endif']
    source = (Path(__file__).parent / 'interface_templates/sparse_exports.zig.in').read_text()
    if any(r['operation'] in ('sparse.trsv','sparse.trsm') for r in plan['bindings']):
        source += (Path(__file__).parent / 'interface_templates/sparse_solve_exports.zig.in').read_text()
    source = source.replace('@@WIRE_INT@@', 'c_longlong' if wide else 'c_int').replace('@@INTERNAL_INT@@', 'i64' if wide else 'i32').replace('@@EXPORTS@@', '\n'.join(exports))
    source = subprocess.run([zig, 'fmt', '--stdin'], input=source, text=True, capture_output=True, check=True).stdout
    manifest = {'schema': 'zynum-finite-interface-subset-v1', 'plan': _binding_plain(plan),
                'scope': 'finite Sparse interface; no full vendor readiness or error precedence equivalence',
                'header_include_order': 'standalone or vendor mkl_spblas.h first',
                'renderer_sources': {name: hashlib.sha256((Path(__file__).parent / name).read_bytes()).hexdigest()
                                     for name in ('generate_checked_abi.py', 'generate_sparse_abi.py', 'interface_templates/sparse_exports.zig.in')},
                'library': 'zynum_sparse_' + width}
    if any(r['operation'] in ('sparse.trsv','sparse.trsm') for r in plan['bindings']):
        manifest['renderer_sources']['interface_templates/sparse_solve_exports.zig.in'] = hashlib.sha256((Path(__file__).parent / 'interface_templates/sparse_solve_exports.zig.in').read_bytes()).hexdigest()
    h, z, m, d = plan['output_paths']
    return {h: '\n'.join(header) + '\n', z: source, m: json.dumps(manifest, indent=2) + '\n',
            d: '; Copyright (C) 2026 Zynum contributors\n; SPDX-License-Identifier: LGPL-3.0-or-later\n' +
               'LIBRARY zynum_sparse_' + width + '\nEXPORTS\n' + ''.join('    ' + name + '\n' for name in plan['symbols'])}
