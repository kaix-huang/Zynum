# Copyright (C) 2026 Zynum contributors
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Generate declaration/forwarding components without duplicating base profiles."""
import argparse
import json
from pathlib import Path
import subprocess
from lapacke_components import ROOT, TARGET, PROFILES, COMPONENTS, manifest_for, compose

def outputs(profile, component, target, zig='zig'):
    manifest = manifest_for(profile, component, target)
    template_name = {'potri': 'lapacke_component.zig.in', 'syev': 'symmetric_component_lp64.zig.in',
                     'sygv': 'symmetric_generalized_component_lp64.zig.in',
                     'controls': 'lapacke_controls.zig.in', 'evd': 'evd_component_lp64.zig.in',
                     'gvd': 'gvd_component_lp64.zig.in', 'gels': 'gels_component_lp64.zig.in', 'gelss': 'gelss_component_lp64.zig.in'}[component]
    template = (ROOT / 'tools/interface_templates' / template_name).read_text()
    for token in ('Context.init', 'std.heap', 'invertNonsingular', 'thread_pool'):
        if token in template:
            raise ValueError('Forwarding template must not own resources or mathematics')
    wide = profile.endswith('.ilp64')
    code = template.replace('@WIRE@', 'c_longlong' if wide else 'c_int').replace('@INTERNAL@', 'i64' if wide else 'i32').replace('@BYTES@', '8' if wide else '4')
    ctype = {'f32': 'f32', 'f64': 'f64', 'complex_f32': 'C', 'complex_f64': 'Z'}
    if component == 'controls':
        exports = ['    @export(&' + v['configuration_forward'] + ', .{ .name = "' + v['symbol'] + '" });'
                   for v in manifest['plan']['bindings']]
    else:
        frontend = {'potri': 'CholeskyInverse', 'syev': 'SymmetricStandard',
                    'sygv': 'GeneralizedSymmetricStandard', 'evd': 'EvdStandard', 'gvd': 'GvdStandard', 'gels': 'GelsStandard', 'gelss': 'GelssStandard'}[component]
        exports = ['    @export(&' + frontend + '(' + ctype[v['scalar']] + ', .' + v['level'] + ').call, .{ .name = "' + v['symbol'] + '" });' for v in manifest['plan']['bindings']]
    code = code.replace('@EXPORTS@', '\n'.join(exports))
    code = subprocess.run([zig, 'fmt', '--stdin'], input=code, text=True, capture_output=True, check=True).stdout
    guard = 'ZYNUM_LAPACKE_' + component.upper() + '_' + ('ILP64' if wide else 'LP64') + '_COMPONENT_V1_H'
    header = ['/* Generated forwarding component; include standalone or original MKL first. */',
        '#ifndef ' + guard, '#define ' + guard, '#include "lapacke_base.h"',
        '#ifdef __cplusplus', 'extern "C" {', '#endif',
        *[v['declaration'] for v in manifest['plan']['bindings']],
        '#ifdef __cplusplus', '}', '#endif', '#endif', '']
    return {'exports.zig': code, 'lapacke_' + component + '.h': '\n'.join(header),
        'lapacke.h': '#pragma once\n#include "lapacke_base.h"\n#include "lapacke_' + component + '.h"\n',
        'abi_manifest.json': json.dumps(manifest, indent=2) + '\n',
        'exports.def': 'EXPORTS\n' + ''.join('    ' + n + '\n' for n in manifest['plan']['symbols'])}

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile', choices=PROFILES)
    parser.add_argument('--component', choices=COMPONENTS)
    parser.add_argument('--target-contract')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--zig', default='zig')
    parser.add_argument('--check', action='store_true')
    parser.add_argument('--compose-base', type=Path)
    parser.add_argument('--component-manifest', type=Path, action='append')
    parser.add_argument('--output-manifest', type=Path)
    args = parser.parse_args()
    try:
        if args.compose_base is not None:
            if not args.component_manifest or args.output_manifest is None or args.profile or args.component or args.output or args.check or args.target_contract:
                raise ValueError('Composition requires only base/component manifests and one output manifest')
            value = compose(json.loads(args.compose_base.read_text()), [json.loads(p.read_text()) for p in args.component_manifest])
            args.output_manifest.parent.mkdir(parents=True, exist_ok=True)
            args.output_manifest.write_text(json.dumps(value, indent=2) + '\n')
            return
        if not args.profile or not args.component or args.target_contract != TARGET or args.output is None or args.component_manifest or args.output_manifest:
            raise ValueError('Generation requires an explicit reviewed profile/component/target/output')
        generated = outputs(args.profile, args.component, args.target_contract, args.zig)
        if args.check:
            for n, text in generated.items():
                if not (args.output / n).is_file() or (args.output / n).read_text() != text:
                    raise ValueError('Generated component drift: ' + n)
        else:
            args.output.mkdir(parents=True, exist_ok=True)
            for n, text in generated.items():
                (args.output / n).write_text(text)
        print(json.dumps({'component': args.component, 'profile': args.profile,
                          'functions': len(manifest_for(args.profile, args.component, args.target_contract)['plan']['bindings']),
                          'outputs': 5, 'runtime': 'pending'}))
    except (ValueError, KeyError) as error:
        parser.error(str(error))

if __name__ == '__main__':
    main()
