#!/usr/bin/env python3
# Copyright (C) 2026 Zynum contributors
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Build the macOS module comparator package; never execute benchmark workers."""
import argparse
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import platform
import re
import shlex
import shutil
import subprocess
import sys
import tarfile
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = Path(__file__).resolve().parent
FILES = ('build.py', 'run_foundation_comparison.py', 'lapack_sweep.zig',
         'sparse_sweep.zig', 'accelerate_sparse_adapter.c',
         'openblas_header_guard.c', 'accelerate_header_guard.c', 'README.md')
ENV = dict(PATH='/usr/bin:/bin:/usr/sbin:/sbin', LANG='C', LC_ALL='C',
           PYTHONDONTWRITEBYTECODE='1', ZERO_AR_DATE='1',
           HOMEBREW_NO_AUTO_UPDATE='1', HOMEBREW_NO_ANALYTICS='1')
if 'HOME' in os.environ:
    ENV['HOME'] = os.environ['HOME']  # Preserve Homebrew's required discovery context.


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def identity(path, framework=False):
    path = Path(path).absolute()
    record = dict(path=str(path), resolved_path=str(path.resolve()))
    if path.is_file():
        record.update(sha256=digest(path.read_bytes()), bytes=path.stat().st_size)
    else:
        require(framework and sys.platform == 'darwin' and str(path).startswith('/System/Library/Frameworks/'), f'missing artifact: {path}')
        record.update(sha256=None, hash_status='unavailable: OS framework may reside in dyld shared cache')
    return record


def executable(request):
    found = shutil.which(request)
    require(found, f'installed executable not found: {request}')
    return str(Path(found).resolve())


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')


def archive_sources(destination, snapshot):
    # Stable archive content/metadata; binaries need not be bitwise reproducible.
    with destination.open('wb') as handle:
        with gzip.GzipFile(filename='', mode='wb', fileobj=handle, mtime=0) as zipped:
            with tarfile.open(fileobj=zipped, mode='w', format=tarfile.PAX_FORMAT) as archive:
                for name, data in sorted(snapshot.items()):
                    info = tarfile.TarInfo(name)
                    info.size, info.mode, info.mtime = len(data), 0o644, 0
                    archive.addfile(info, io.BytesIO(data))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=lambda s: Path(s).absolute())
    parser.add_argument('--zig', default='zig', help='installed Zig executable; must match repository pin')
    parser.add_argument('--clang', help='installed Clang; default xcrun --find clang')
    parser.add_argument('--sdk', type=lambda s: Path(s).absolute(), help='macOS SDK; default xcrun discovery')
    parser.add_argument('--openblas-prefix', type=lambda s: Path(s).absolute(), help='installed LP64 OpenBLAS prefix; default brew --prefix openblas')
    parser.add_argument('--accelerate', type=lambda s: Path(s).absolute(), default=Path('/System/Library/Frameworks/Accelerate.framework/Accelerate'))
    args = parser.parse_args()
    require(sys.platform == 'darwin', 'This package currently requires macOS Accelerate; no silent substitute backend.')
    require(not args.output.exists(), 'output exists; choose a new directory')
    require((ROOT / 'bench/tools/report_schedule.py').is_file(), 'place this package at bench/modules or another two-level directory within the repository')
    require(not args.output.is_relative_to(PACKAGE / 'src'), 'invalid output directory')
    args.output.mkdir(parents=True)
    commands = []
    manifest = dict(schema_version=1, status='building', repository=str(ROOT),
                    started_utc=datetime.now(timezone.utc).isoformat(),
                    invocation_argv=[sys.executable, *sys.argv], environment=ENV, commands=commands,
                    python=dict(version=sys.version, executable=identity(sys.executable)),
                    claim='Reproducible build recipe and frozen build-source evidence; no benchmark execution, clean-HEAD claim or bitwise-binary reproducibility claim.')

    def run(argv, name, cwd=ROOT):
        try:
            result = subprocess.run(list(map(str, argv)), cwd=cwd, env=ENV,
                                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=300)
            output, returncode = result.stdout, result.returncode
        except subprocess.TimeoutExpired as error:
            output, returncode = error.stdout or b'', 'timeout'
        except OSError as error:
            output, returncode = str(error).encode(), 'spawn-error'
        log = args.output / (name + '.log')
        log.write_bytes(output)
        entry = dict(argv=list(map(str, argv)), cwd=str(cwd), exit_code=returncode,
                     argv_sha256=digest(json.dumps(list(map(str, argv)), separators=(',', ':')).encode()),
                     output_log=identity(log))
        commands.append(entry)
        require(returncode == 0, f'{name} failed; inspect {log}')
        return output.decode().strip()

    try:
        xcrun = executable('xcrun')
        zig = executable(args.zig)
        clang = executable(args.clang or run([xcrun, '--find', 'clang'], 'find-clang'))
        sdk = args.sdk or Path(run([xcrun, '--sdk', 'macosx', '--show-sdk-path'], 'find-sdk'))
        require((sdk / 'System/Library/Frameworks/Accelerate.framework').is_dir(), 'SDK lacks Accelerate')
        prefix = args.openblas_prefix or Path(run([executable('brew'), '--prefix', 'openblas'], 'find-openblas'))
        require((prefix / 'include/lapacke.h').is_file(), 'OpenBLAS LAPACKE headers missing')
        library = prefix / 'lib/libopenblas.dylib'
        zig_version = run([zig, 'version'], 'zig-version')
        pin = re.search(r'\.minimum_zig_version\s*=\s*"([^"]+)"', (ROOT / 'build.zig.zon').read_text())
        require(pin and zig_version == pin[1], f'use the repository Zig pin ({pin[1] if pin else "not found"}); got {zig_version}')
        manifest.update(zig_version=zig_version, clang_version=run([clang, '--version'], 'clang-version'),
                        compilers={'zig': identity(zig), 'clang': identity(clang), 'xcrun': identity(xcrun)},
                        sdk=dict(path=str(sdk.resolve()), version=run([xcrun, '--sdk', str(sdk), '--show-sdk-version'], 'sdk-version')),
                        platform=dict(uname=platform.uname()._asdict(), os_release=run(['sw_vers'], 'os-release'),
                                      cpu=run(['sysctl', '-n', 'machdep.cpu.brand_string'], 'cpu')),
                        git_head=run(['git', 'rev-parse', 'HEAD'], 'git-head'),
                        git_status=run(['git', 'status', '--porcelain=v1', '--untracked-files=all'], 'git-status'),
                        libraries={'openblas': identity(library), 'accelerate': identity(args.accelerate, True)})
        zig_env = run([zig, 'env'], 'zig-env')
        manifest['zig_environment_zon'] = zig_env
        std_match = re.search(r'^\s*\.std_dir = ("(?:[^"\\]|\\.)*"),$', zig_env, re.MULTILINE)
        require(std_match, 'cannot identify Zig standard library in zig env output')
        # Header dependencies below cover Clang resources; fingerprint the Zig
        # standard library too, but do not redistribute installed toolchains.
        std_root = Path(json.loads(std_match[1]))
        std_hashes = {str(p.relative_to(std_root)): digest(p.read_bytes()) for p in sorted(std_root.rglob('*.zig'))}
        manifest['zig_stdlib'] = dict(path=str(std_root), files=len(std_hashes),
                                     source_sha256=std_hashes,
                                     tree_sha256=digest(json.dumps(std_hashes, sort_keys=True).encode()))
        paths = set((ROOT / 'src').rglob('*.zig'))
        paths.update(ROOT / name for name in ('build.zig.zon', 'bench/dynamic_library.zig', 'bench/tools/report_schedule.py', 'bench/tools/test_module_foundation.py'))
        paths.update(PACKAGE / name for name in FILES)
        snapshot = {str(p.relative_to(ROOT)): p.read_bytes() for p in sorted(paths)}
        manifest['source_sha256'] = {name: digest(data) for name, data in snapshot.items()}
        # Build the bytes we archive, not a live checkout that another task may edit.
        source_tree = args.output / 'source'
        for name, data in snapshot.items():
            target = source_tree / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
        source_archive = args.output / 'build-sources.tar.gz'
        archive_sources(source_archive, snapshot)
        manifest['source_archive'] = dict(path=str(source_archive), sha256=digest(source_archive.read_bytes()),
                                         members=len(snapshot), format='deterministic gzip tar; repository-relative regular files')
        package = source_tree / PACKAGE.relative_to(ROOT)
        common = [clang, '-std=c11', '-O3', '-Wall', '-Wextra', '-Werror', '-isysroot', str(sdk)]
        compile_inputs = []
        for stem in ('openblas_header_guard', 'accelerate_header_guard'):
            flags = ['-I' + str(prefix / 'include')] if stem.startswith('openblas') else []
            compile_inputs.append((stem, flags, package / (stem + '.c')))
        compile_inputs.append(('accelerate_sparse_adapter', [], package / 'accelerate_sparse_adapter.c'))
        dependencies = []
        for stem, flags, source in compile_inputs:
            depfile = args.output / (stem + '.d')
            run(common + flags + ['-M', '-MT', 'dependencies', '-MF', str(depfile), str(source)], stem + '-dependencies', source_tree)
            dependencies.append(depfile)

        def dependency_paths():
            paths = set()
            for depfile in dependencies:
                text = depfile.read_text().replace('\\\n', ' ')
                for name in shlex.split(text.split(':', 1)[1]):
                    path = Path(name).resolve()
                    if not path.is_relative_to(source_tree):
                        paths.add(path)
            return paths

        header_hashes = {str(p): identity(p)['sha256'] for p in sorted(dependency_paths())}
        manifest['header_sha256'] = header_hashes
        stubs = set((sdk / 'System/Library/Frameworks/Accelerate.framework').rglob('*.tbd'))
        stubs.add(sdk / 'usr/lib/libSystem.tbd')
        stub_hashes = {str(p.resolve()): identity(p)['sha256'] for p in sorted(stubs)}
        manifest['sdk_link_stub_sha256'] = stub_hashes
        for stem, flags, source in compile_inputs:
            depfile = args.output / (stem + '.d')
            if stem == 'accelerate_sparse_adapter':
                link = ['-dynamiclib', '-framework', 'Accelerate']
                target = args.output / (stem + '.dylib')
            else:
                link = ['-c']
                target = args.output / (stem + '.o')
            run(common + flags + ['-MD', '-MF', str(depfile)] + link + [str(source), '-o', str(target)], 'build-' + stem, source_tree)
            if stem != 'accelerate_sparse_adapter':
                symbols = run(['nm', '-u', str(target)], stem + '-symbols')
                for scalar in 'sd':
                    for operation in ('potrf', 'potrs', 'getrf', 'getrs'):
                        symbol = '_LAPACKE_' + scalar + operation + '_work' if stem.startswith('openblas') else '_' + scalar + operation + '$NEWLAPACK'
                        require(symbol in symbols.split(), 'header guard did not bind required symbol: ' + symbol)
        require({str(p) for p in dependency_paths()} == set(header_hashes), 'C header dependency set changed while compiling')
        for module in ('lapack', 'sparse'):
            run([zig, 'build-exe', '-Ofast', '-mcpu=baseline', '--dep', 'zynum-' + module,
                 '--dep', 'benchmark-dynamic-library', '-Mroot=' + str(package / (module + '_sweep.zig')),
                 '-Ofast', '-mcpu=baseline', '-Mzynum-' + module + '=' + str(source_tree / ('src/' + module + '.zig')),
                 '-Ofast', '-Mbenchmark-dynamic-library=' + str(source_tree / 'bench/dynamic_library.zig'), '-lc',
                 '--cache-dir', str(args.output / 'cache'), '--global-cache-dir', str(args.output / 'global-cache'),
                 '-femit-bin=' + str(args.output / (module + '-sweep'))], 'build-' + module, source_tree)
        binaries = ('lapack-sweep', 'sparse-sweep', 'accelerate_sparse_adapter.dylib')
        manifest['binary_sha256'] = {name: identity(args.output / name)['sha256'] for name in binaries}
        manifest['binary_paths'] = {name: str(args.output / name) for name in binaries}
        for name in binaries:
            linkage = run(['otool', '-L', str(args.output / name)], name + '-linkage')
            if name != 'accelerate_sparse_adapter.dylib':
                require('Accelerate.framework' not in linkage and 'openblas' not in linkage.lower(), 'worker directly links competitor')
        for name, data in snapshot.items():
            require((source_tree / name).read_bytes() == data, 'build snapshot changed: ' + name)
            require((ROOT / name).read_bytes() == data, 'checkout source changed during build; rebuild: ' + name)
        for path, expected in dict(header_hashes, **stub_hashes).items():
            require(identity(path)['sha256'] == expected, 'header changed during build')
        for name, path in [('openblas', library), ('accelerate', args.accelerate)]:
            require(identity(path, name == 'accelerate') == manifest['libraries'][name], 'library changed during build')
        require({str(p.relative_to(std_root)): digest(p.read_bytes()) for p in sorted(std_root.rglob('*.zig'))} == std_hashes, 'Zig standard library changed during build')
        for name, path in [('zig', zig), ('clang', clang), ('xcrun', xcrun)]:
            require(identity(path) == manifest['compilers'][name], 'compiler changed during build')
        manifest.update(status='complete', completed_utc=datetime.now(timezone.utc).isoformat(),
                        measured=False, binary_policy='ReleaseFast, CPU baseline, checked module APIs; no benchmark worker executed by build driver')
        write_json(args.output / 'build-manifest.json', manifest)
        print(args.output / 'build-manifest.json')
    except BaseException as error:
        manifest.update(status='failed', error=f'{type(error).__name__}: {error}')
        write_json(args.output / 'build-failure.json', manifest)
        raise


if __name__ == '__main__':
    main()
