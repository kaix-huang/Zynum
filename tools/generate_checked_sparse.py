# Copyright (C) 2026 Zynum contributors
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Generate the experimental additive Sparse candidate; not a released ABI."""
import argparse
import json
from pathlib import Path
import sys
sys.dont_write_bytecode = True
import subprocess
from checked_sparse_schema import validate_candidate
from checked_sparse_sources import verify_sources
from checked_complex_wire import header as complex_header


def render_candidate(candidate, zig):
    plan = validate_candidate(candidate)  # reject before formatter side effects
    verify_sources()  # source identity must pass before formatter or output writes
    c = ['/* Generated candidate; not a released ABI. */', '#ifndef ZYNUM_CHECKED_SPARSE_H',
         '#define ZYNUM_CHECKED_SPARSE_H', '#include <zynum/checked/checked.h>',
         '#include <zynum/checked/complex.h>',
         '#ifdef __cplusplus', 'extern "C" {', '#define ZCS_ASSERT static_assert', '#define ZCS_ALIGNOF alignof',
         '#else', '#define ZCS_ASSERT _Static_assert', '#define ZCS_ALIGNOF _Alignof', '#endif']
    z = ['// Generated candidate; not a released ABI.', 'const abi = @import("abi_types.zig");',
         'const impl = @import("implementation.zig");', 'const core = @import("zynum-core");']
    for name, layout in plan['complex_layouts'].items():
        typ = 'zynum_checked_' + name
        c += [f'ZCS_ASSERT(sizeof({typ}) == {layout["size"]}, "{name} size");',
              f'ZCS_ASSERT(ZCS_ALIGNOF({typ}) == {layout["alignment"]}, "{name} alignment");']
        for field, offset in layout['offsets'].items():
            c.append(f'ZCS_ASSERT(offsetof({typ}, {field}) == {offset}, "{name}.{field}");')
        native = 'core.sparse.ComplexF32' if name == 'complex_f32' else 'core.sparse.ComplexF64'
        z += [f'comptime {{ if (@sizeOf({native}) != {layout["size"]} or @alignOf({native}) != {layout["alignment"]}) @compileError("{name} layout");',
              f'if (@offsetOf({native}, "re") != 0 or @offsetOf({native}, "im") != {layout["offsets"]["im"]}) @compileError("{name} fields"); }}']
    for flag, value in plan['flags'].items():
        c.append(f'#define ZYNUM_CHECKED_SPARSE_{flag} INT32_C({value})')
    for row in plan['bindings']:
        c_types = {'i32': 'int32_t', row['scalar']: row['scalar_c_type']}
        z_types = {'i32': 'i32', row['scalar']: row['scalar_zig_type']}
        ca, za = [], []
        for name, typ, _ in row['parameters']:
            ca.append(c_types.get(typ, 'zynum_checked_' + typ) + ' ' + name)
            za.append(name + ': ' + z_types.get(typ, 'abi.' + typ))
        c.append('zynum_checked_result ' + row['symbol'] + '(' + ', '.join(ca) + ');')
        args = [row['scalar_zig_type'], row['offset'], row['ordinal'], *(v[0] for v in row['parameters'])]
        z += ['pub export fn ' + row['symbol'] + '(' + ', '.join(za) + ') callconv(.c) abi.result {',
              'return impl.' + row['adapter'] + '(' + ', '.join(args) + ');', '}']
    c += ['#undef ZCS_ASSERT', '#undef ZCS_ALIGNOF', '#ifdef __cplusplus', '}', '#endif', '#endif']
    source = subprocess.run([zig, 'fmt', '--stdin'], input='\n'.join(z)+'\n', text=True,
                            capture_output=True, check=True).stdout
    return {'include/zynum/checked/complex.h': complex_header(),
            'include/zynum/checked/sparse.h': '\n'.join(c)+'\n',
            'src/checked/sparse_exports.zig': source,
            'src/checked/sparse_exports.def': 'LIBRARY zynum_checked_sparse\nEXPORTS\n'+''.join('    '+s+'\n' for s in plan['symbols']),
            'include/zynum/checked/sparse_manifest.json': json.dumps({'plan': plan, 'runtime_abi_status': 'pending'}, indent=2)+'\n'}


def main(argv=None):
    from checked_sparse_profile import candidate
    from checked_sparse_schema import SparseProfileError
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile', required=True)
    parser.add_argument('--binding-set', required=True)
    parser.add_argument('--target-contract', required=True)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--zig', default='zig')
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args(argv)
    try:
        plan = candidate(args.profile, args.binding_set, args.target_contract)
        generated = render_candidate(plan, args.zig)
    except (ValueError, SparseProfileError) as error:
        parser.error(str(error))
    # Validate the entire output set before performing any writes.
    if args.check:
        drift = [name for name, content in generated.items()
                 if not (args.root / name).is_file() or (args.root / name).read_text() != content]
        if drift:
            parser.error('Generated Sparse drift: ' + ', '.join(drift))
    else:
        for name, content in generated.items():
            target = args.root / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content)
    print(('Checked' if args.check else 'Generated') +
          ' 5 experimental Sparse outputs, 128 additive entries; release acceptance pending')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
