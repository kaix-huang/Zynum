#!/usr/bin/env python3
# Copyright (C) 2026 Zynum contributors
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Check installed joint BLAS/LAPACKE/FFT calls and resource lifetimes."""
import argparse
import hashlib
import itertools
import json
import os
from pathlib import Path
import re
import shlex
import subprocess


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prefix', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    prefix = args.prefix.resolve()
    source = Path(__file__).resolve().parent.parent / 'test/headers/unified_blas_smoke.c'
    manifest = json.loads((prefix / 'include/zynum/blas/abi_manifest.json').read_text())
    names = {row['name'] for family in ('cblas','fortran') for row in manifest[family]['exports']}
    if manifest.get('blas_integer_abi') != 'LP64' or len(names) != 311:
        parser.error('requires the installed existing 311-entry LP64 BLAS interface')
    shared = prefix / 'lib/libzynum_abplas_lp64.dylib'
    static = prefix / 'lib/libzynum_abplas_lp64.a'
    symbols = subprocess.check_output(['nm','-gU',str(shared)], text=True)
    observed = {line.split()[-1].removeprefix('_') for line in symbols.splitlines() if line.strip()}
    if not names <= observed:
        parser.error('unified library omits declared BLAS exports')
    if {n for n in observed if n.startswith('cblas_')} != {n for n in names if n.startswith('cblas_')}:
        parser.error('unified CBLAS public symbols differ from the installed declarations')
    contexts = {}
    for lib in (static,shared):
        all_symbols = subprocess.check_output(['nm','-a',str(lib)], text=True)
        count = len(re.findall(r' [dDbB] _[^ ]*default_context$', all_symbols, re.MULTILINE))
        if count != 1:
            parser.error('unified artifact requires one actual default Context')
        contexts[str(lib)] = dict(sha256=sha(lib), default_Context_count=count)
    env = dict(os.environ, PKG_CONFIG_PATH=str(prefix/'lib/pkgconfig'),
               PKG_CONFIG_LIBDIR=str(prefix/'lib/pkgconfig'))
    cflags = shlex.split(subprocess.check_output(['pkg-config','--cflags','zynum_abplas_lp64'], env=env, text=True))
    libs = shlex.split(subprocess.check_output(['pkg-config','--libs','zynum_abplas_lp64'], env=env, text=True))
    if '-lzynum_abplas_lp64' not in libs or any(x.startswith('-l') and x != '-lzynum_abplas_lp64' for x in libs):
        parser.error('package flags must select just the unified numerical library')
    args.output.mkdir(parents=True,exist_ok=True)
    rows = []
    for language,kind in itertools.product(('c','cpp'),('a','dylib')):
        label = language+'-'+kind
        executable = args.output / label
        command = ['clang' if language=='c' else 'clang++', '-x', 'c' if language=='c' else 'c++',
                   '-std=c11' if language=='c' else '-std=c++20', '-Wall','-Wextra','-Werror',
                   *cflags, str(source)]
        command += ['-x','none',str(static)] if kind=='a' else libs
        command += ['-Wl,-rpath,'+str(prefix/'lib'),'-o',str(executable)]
        result = subprocess.run(command, env=env, capture_output=True, text=True)
        compile_log = args.output / (label+'-compile.log')
        compile_log.write_text(result.stdout+result.stderr)
        if result.returncode:
            raise RuntimeError(compile_log.read_text())
        dependencies = subprocess.check_output(['otool','-L',str(executable)],text=True)
        if kind=='a' and 'libzynum_' in dependencies:
            raise RuntimeError('static client unexpectedly imports a Zynum shared library')
        for threads in (1,4):
            run_env = dict(env, ZYNUM_MAXIMUM_THREADS=str(threads),
                           DYLD_LIBRARY_PATH=str(prefix/'lib'), DYLD_PRINT_LIBRARIES='1')
            result = subprocess.run([str(executable)],env=run_env,capture_output=True,text=True)
            run_log = args.output / (label+'-'+str(threads)+'-run.log')
            run_log.write_text(result.stdout+result.stderr)
            if result.returncode or result.stdout.strip() != 'unified BLAS numerical, boundaries and cross-module lifetimes passed':
                raise RuntimeError(run_log.read_text())
            images = [line for line in result.stderr.splitlines() if 'libzynum_' in line]
            if kind=='dylib' and (len(images)!=1 or str(shared) not in images[0]):
                raise RuntimeError('shared client did not load the selected single numerical library')
            if kind=='a' and images:
                raise RuntimeError('static client loaded a Zynum shared library')
            rows.append(dict(language=language,linkage=kind,threads=threads,command=command,
                             dependencies=dependencies,actual_Zynum_images=images,
                             run_log_sha256=sha(run_log)))
            print(label,threads,'numerical/boundary/lifetime checks passed',flush=True)
    receipt = dict(status='passed-installed-unified-BLAS-C-Cpp-static-shared-two-thread-settings',
                   prefix=str(prefix),actual_runs=len(rows),runs=rows,artifacts=contexts,
                   existing_BLAS_exports=311,unified_shared_exports=len(observed),
                   source_sha256=sha(source),performance_measured=False,ABP_compiled=False)
    (args.output/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')


if __name__=='__main__':
    main()
