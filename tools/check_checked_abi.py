#!/usr/bin/env python3
# Copyright (C) 2026 Zynum contributors
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Check the separate checked library; never merge its symbols into BLAS parity."""
import argparse
import json
from pathlib import Path
import re
import subprocess


def run(argv):
    return subprocess.run(argv, check=True, text=True, capture_output=True, timeout=60).stdout


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', required=True, type=Path)
    p.add_argument('--dynamic', required=True, type=Path)
    p.add_argument('--static', type=Path)
    p.add_argument('--format', required=True, choices=('macho', 'elf', 'coff'))
    p.add_argument('--nm', default='nm')
    p.add_argument('--readobj', default='llvm-readobj')
    args = p.parse_args()
    manifest = json.loads(args.manifest.read_text())
    if manifest.get('schema_id') != 'zynum-checked-abi-v1' or manifest.get('export_count') != 8:
        raise SystemExit('Unsupported checked ABI manifest')
    expected = {e['name'] for e in manifest['exports']}
    if len(expected) != 8 or not all(x.startswith('zynum_checked_') for x in expected):
        raise SystemExit('Invalid checked ABI declaration set')
    if args.format == 'coff':
        text = run([args.readobj, '--coff-exports', str(args.dynamic)])
        actual = set(re.findall(r'Export \{[\s\S]*?Name: ([^\n]+)', text))
    else:
        flags = ['-gU'] if args.format == 'macho' else ['-D', '--defined-only', '--extern-only']
        text = run([args.nm, *flags, str(args.dynamic)])
        actual = {line.split()[-1] for line in text.splitlines() if len(line.split()) >= 2}
        if args.format == 'macho':
            actual = {name.removeprefix('_') for name in actual}
    support = set(manifest['object_support_symbols'][args.format])
    if actual != expected | support:
        raise SystemExit('Dynamic exports mismatch: missing=' + repr(sorted((expected | support) - actual)) + ' extra=' + repr(sorted(actual - expected - support)))
    if args.format == 'macho':
        types = {line.split()[-1].removeprefix('_'): line.split()[-2] for line in text.splitlines() if len(line.split()) >= 3}
        if any(types[name] != 'T' for name in expected) or any(types[name] != 'D' for name in support):
            raise SystemExit('Mach-O application/support symbol kinds differ')
    if args.static:
        # Archive members also contain private/runtime symbols; only the dynamic
        # export surface is an exact allowlist. Every API must exist in static.
        flags = ['-gU'] if args.format == 'macho' else ['-g', '--defined-only']
        text = run([args.nm, *flags, str(args.static)])
        names = {line.split()[-1] for line in text.splitlines() if len(line.split()) >= 2}
        if args.format == 'macho':
            names = {name.removeprefix('_') for name in names}
        if not expected <= names:
            raise SystemExit('Static archive missing checked APIs: ' + repr(sorted(expected - names)))
    print('Checked ABI: eight API exports, ' + str(len(support)) + ' explicitly recorded support-data exports' + ('; all eight static definitions' if args.static else ''))


if __name__ == '__main__':
    main()
