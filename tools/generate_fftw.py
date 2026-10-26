# Copyright (C) 2026 Zynum contributors
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Generate a selected external FFTW declaration profile, never numerical state.

The finite ABPLaS profile is declaration-only until its runtime binding passes
installation, numerical, lifetime, concurrent-use and application gates.
"""
import argparse
import hashlib
import json
import re
from pathlib import Path

PROFILE = 'fftw3.f64.abplas-v1'
REVIEWED_SYMBOLS = ('fftw_plan_dft_1d', 'fftw_plan_dft_r2c_1d', 'fftw_plan_dft_r2c_3d', 'fftw_plan_dft_c2r_3d', 'fftw_plan_r2r_1d', 'fftw_execute', 'fftw_destroy_plan', 'fftw_alloc_complex', 'fftw_free', 'fftw_alignment_of', 'fftw_execute_dft', 'fftw_execute_dft_r2c', 'fftw_execute_dft_c2r', 'fftw_execute_r2r', 'fftw_plan_dft_2d', 'fftw_plan_dft_3d', 'fftw_plan_dft', 'fftw_plan_dft_r2c_2d', 'fftw_plan_dft_r2c', 'fftw_plan_dft_c2r_1d', 'fftw_plan_dft_c2r_2d', 'fftw_plan_dft_c2r', 'fftw_plan_r2r_2d', 'fftw_plan_r2r_3d', 'fftw_plan_r2r', 'fftw_plan_many_dft', 'fftw_plan_guru_dft', 'fftw_plan_guru64_dft', 'fftw_plan_many_dft_r2c', 'fftw_plan_many_dft_c2r', 'fftw_plan_many_r2r', 'fftw_plan_guru_dft_r2c', 'fftw_plan_guru_dft_c2r', 'fftw_plan_guru_r2r', 'fftw_plan_guru64_dft_r2c', 'fftw_plan_guru64_dft_c2r', 'fftw_plan_guru64_r2r', 'fftw_plan_guru_split_dft', 'fftw_plan_guru_split_dft_r2c', 'fftw_plan_guru_split_dft_c2r', 'fftw_plan_guru64_split_dft', 'fftw_plan_guru64_split_dft_r2c', 'fftw_plan_guru64_split_dft_c2r', 'fftw_execute_split_dft', 'fftw_execute_split_dft_r2c', 'fftw_execute_split_dft_c2r', 'fftw_malloc', 'fftw_alloc_real')


def render(profile, source_hashes, template):
    if profile['profile'] != PROFILE or profile['target'] != 'apple-arm64-default-c-v1':
        raise ValueError('unsupported external profile or target')
    functions = profile['functions']
    symbols = [f['symbol'] for f in functions]
    if tuple(symbols) != REVIEWED_SYMBOLS:
        raise ValueError('profile must contain the exact reviewed forty-eight finite FFTW functions')
    h = [
        '// Copyright (C) 2026 Zynum contributors',
        '// SPDX-License-Identifier: LGPL-3.0-or-later',
        '/* Generated finite FFTW declarations; runtime integration is pending. */',
        '/* Include standalone, or AFTER the original fftw3.h if using both. */',
        '#ifndef ZYNUM_FFTW3_F64_ABPLAS_V1_H',
        '#define ZYNUM_FFTW3_F64_ABPLAS_V1_H',
        '#include <stddef.h>',
        '#if !defined(__APPLE__) || !defined(__aarch64__)',
        '#error "Selected FFTW ABI profile requires Apple arm64"',
        '#endif',
        '#ifndef FFTW3_H',
        '#if !defined(FFTW_NO_Complex) && defined(_Complex_I) && defined(complex) && defined(I)',
        'typedef double _Complex fftw_complex;',
        '#else',
        'typedef double fftw_complex[2];',
        '#endif',
        'typedef struct fftw_plan_s *fftw_plan;',
        'struct fftw_iodim_do_not_use_me { int n, is, os; };',
        'typedef struct fftw_iodim_do_not_use_me fftw_iodim;',
        'struct fftw_iodim64_do_not_use_me { ptrdiff_t n, is, os; };',
        'typedef struct fftw_iodim64_do_not_use_me fftw_iodim64;',
        'enum fftw_r2r_kind_do_not_use_me {',
        ',\n'.join('    FFTW_' + n + ' = ' + str(v) for n, v in profile['r2r_kinds'].items()),
        '};',
        'typedef enum fftw_r2r_kind_do_not_use_me fftw_r2r_kind;',
        '#endif',
    ]
    for name, value in profile['constants'].items():
        suffix = '' if name in ('FFTW_FORWARD', 'FFTW_BACKWARD') else 'U'
        h += ['#ifndef ' + name, '#define ' + name + ' (' + str(value) + suffix + ')', '#endif']
    h += [
        '#ifdef __cplusplus',
        '#define ZYNUM_FFTW_ASSERT static_assert',
        '#define ZYNUM_FFTW_ALIGN alignof',
        '#else',
        '#define ZYNUM_FFTW_ASSERT _Static_assert',
        '#define ZYNUM_FFTW_ALIGN _Alignof',
        '#endif',
        'ZYNUM_FFTW_ASSERT(sizeof(int) == 4 && sizeof(unsigned) == 4, "FFTW integer ABI");',
        'ZYNUM_FFTW_ASSERT(sizeof(size_t) == 8 && sizeof(fftw_plan) == 8, "FFTW pointer ABI");',
        'ZYNUM_FFTW_ASSERT(sizeof(double) == 8 && sizeof(fftw_complex) == 16 && ZYNUM_FFTW_ALIGN(fftw_complex) == 8, "FFTW complex ABI");',
        'ZYNUM_FFTW_ASSERT(sizeof(fftw_r2r_kind) == 4 && ZYNUM_FFTW_ALIGN(fftw_r2r_kind) == 4, "FFTW enum ABI");',
        'ZYNUM_FFTW_ASSERT(sizeof(fftw_iodim) == 12 && ZYNUM_FFTW_ALIGN(fftw_iodim) == 4 && offsetof(fftw_iodim, is) == 4 && offsetof(fftw_iodim, os) == 8, "FFTW guru ABI");',
        'ZYNUM_FFTW_ASSERT(sizeof(fftw_iodim64) == 24 && ZYNUM_FFTW_ALIGN(fftw_iodim64) == 8 && offsetof(fftw_iodim64, is) == 8 && offsetof(fftw_iodim64, os) == 16, "FFTW guru64 ABI");',
        '#undef ZYNUM_FFTW_ASSERT',
        '#undef ZYNUM_FFTW_ALIGN',
        '#ifdef __cplusplus',
        'extern "C" {',
        '#endif',
    ]
    h += [f['returns'] + ' ' + f['symbol'] + '(' + f['parameters'] + ');' for f in functions]
    h += ['#ifdef __cplusplus', '}', '#endif', '#endif']
    manifest = {
        'schema': 'zynum-external-interface-declarations-v1',
        'profile': profile['profile'], 'target': profile['target'],
        'symbols': symbols, 'source_hashes': source_hashes,
        'full_fftw_compatibility': False,
        'runtime_exported': False,
        'status': 'declarations-only-runtime-pending',
        'header_include_order': 'standalone or original fftw3.h first',
        'internal_owner': 'existing Zynum Core; no owner generated by this script',
    }
    names = re.findall(r'export fn (fftw_[a-zA-Z0-9_]+)\(', template)
    if sorted(names) != sorted(symbols):
        raise ValueError('runtime export template differs from selected profile')
    return {
        'exports.zig': template,
        'fftw3_subset.h': '\n'.join(h) + '\n',
        'exports.txt': ''.join(n + '\n' for n in symbols),
        'abi_manifest.json': json.dumps(manifest, indent=2) + '\n',
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile', choices=[PROFILE], default=PROFILE)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    inputs = [Path(__file__), Path(__file__).with_name('fftw_profile.json'), Path(__file__).parent / 'interface_templates/fftw_exports.zig.in']
    profile = json.loads(inputs[1].read_text())
    if args.profile != profile['profile']:
        raise ValueError('requested profile differs from source')
    outputs = render(profile, {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs}, inputs[2].read_text())
    if args.check:
        stale = [n for n, content in outputs.items()
                 if not (args.output / n).is_file() or (args.output / n).read_text() != content]
        if stale:
            parser.exit(1, 'generated interface drift: ' + ', '.join(stale) + '\n')
    else:
        args.output.mkdir(parents=True, exist_ok=True)
        for name, content in outputs.items():
            (args.output / name).write_text(content)


if __name__ == '__main__':
    main()
