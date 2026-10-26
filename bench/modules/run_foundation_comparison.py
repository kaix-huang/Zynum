#!/usr/bin/env python3
# Copyright (C) 2026 Zynum contributors
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Serialized fresh-process foundation comparisons bound to an explicit build manifest.

Small suite: LAPACK 28 cases; Sparse 72. Full: LAPACK 74; Sparse 144 (adds pad).
--limit-cases truncates each selected module. Each case/backend has its own process:
LAPACK 6 interleaved repeats, Sparse 4; every process retains 30 inner samples.
An output directory appears only after all checks pass; failures retain diagnostics
in a sibling *.incomplete-* directory, without publishing aggregate.csv.
"""
import argparse
import csv
import hashlib
import importlib.util
import io
import json
import math
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import tempfile
import tarfile
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[2]
DENSE = 'backend,kind,m,n,nrhs,uplo,lda,ldb,batch_calls,samples,median_ns,p95_ns,best_ns,scaled_residual,check,factor_unused_triangle_modified_max,batch_target_met,min_batch_ns,sample_ns'.split(',')
SPARSE = 'backend,scalar,index_bits,rows,columns,pattern,transpose,rhs,pad,nnz,setup_kind,setup_ns,execution_kind,samples,batch_calls,min_batch_ns,batch_target_met,index_layout,best_ns,median_ns,p95_ns,max_ns,gflops,check,max_absolute_error,scaled_residual,padding,checksum,sample_ns'.split(',')
ENV = dict(PATH='/usr/bin:/bin:/usr/sbin:/sbin', LANG='C', LC_ALL='C', ZYNUM_MAXIMUM_THREADS='1', OPENBLAS_NUM_THREADS='1', OPENBLAS_DYNAMIC='0', VECLIB_MAXIMUM_THREADS='1', OMP_NUM_THREADS='1', OMP_DYNAMIC='FALSE', MKL_NUM_THREADS='1', MKL_DYNAMIC='FALSE', BLIS_NUM_THREADS='1')
BACKENDS = {'lapack': ['zynum', 'openblas', 'accelerate'], 'sparse': ['zynum', 'accelerate']}
REPEATS = {'lapack': 6, 'sparse': 4}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def capture(*command):
    return subprocess.run(command, cwd=ROOT, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=15).stdout.decode().strip()


def identity(path, framework=False):
    path = Path(path).absolute()
    result = {'path': str(path), 'resolved_path': str(path.resolve())}
    if path.is_file():
        with path.open('rb') as handle:
            digest = hashlib.file_digest(handle, 'sha256').hexdigest()
        result.update(sha256=digest, bytes=path.stat().st_size)
    else:
        require(framework and sys.platform == 'darwin' and str(path).startswith('/System/Library/Frameworks/'), f'missing artifact: {path}')
        result.update(sha256=None, hash_status='unavailable: OS framework may reside in dyld shared cache')
    return result


def source_state():
    names = capture('git', 'ls-files', '-z', '--cached', '--others', '--exclude-standard').split('\0')
    paths = {ROOT / name for name in names if name and Path(name).suffix in {'.zig', '.zon', '.py', '.c', '.h', '.json'}}
    paths.update(Path(__file__).parent / name for name in ('lapack_sweep.zig', 'sparse_sweep.zig', 'accelerate_sparse_adapter.c', 'build.py', 'openblas_header_guard.c', 'accelerate_header_guard.c', 'README.md', Path(__file__).name))
    return {'head': capture('git', 'rev-parse', 'HEAD'), 'status_porcelain': capture('git', 'status', '--porcelain=v1', '--untracked-files=all'),
            'source_sha256': {str(p.relative_to(ROOT)): identity(p)['sha256'] for p in sorted(paths) if p.is_file()}}


def cases(module, suite):
    result = []
    if module == 'lapack':
        squares = [32, 128] if suite == 'small' else [16, 32, 64, 128, 256]
        solve_sizes = [128] if suite == 'small' else [32, 128, 256]
        rectangles = [(128, 64), (64, 128)] + ([(256, 64), (64, 256)] if suite == 'full' else [])
        for scalar in 'sd':
            for n in squares:
                result += [dict(kind=scalar + 'potrf', m=n, n=n, nrhs=0, uplo=u) for u in 'LU']
                result.append(dict(kind=scalar + 'getrf', m=n, n=n, nrhs=0, uplo='N'))
            result += [dict(kind=scalar + 'getrf', m=m, n=n, nrhs=0, uplo='N') for m, n in rectangles]
            for n in solve_sizes:
                for rhs in (1, 8):
                    result += [dict(kind=scalar + 'potrs', m=n, n=n, nrhs=rhs, uplo=u) for u in 'LU']
                    result.append(dict(kind=scalar + 'getrs', m=n, n=n, nrhs=rhs, uplo='N'))
    else:
        for scalar in ('f32', 'f64'):
            for m, n in ((256, 256), (4096, 4096), (8192, 2048)):
                for pattern in ('banded', 'uniform16', 'irregular'):
                    for trans in 'NT':
                        for rhs in (1, 8):
                            for pad in ((0,) if suite == 'small' else (0, 1)):
                                result.append(dict(scalar=scalar, rows=m, columns=n, pattern=pattern, transpose=trans, rhs=rhs, index_bits=32, pad=pad))
    return result


def command(args, module, backend, case):
    cmd = [str(args.lapack_worker if module == 'lapack' else args.sparse_worker), '--backend', backend]
    if module == 'lapack':
        for key in ('kind', 'm', 'n', 'nrhs'):
            cmd += ['--' + key, str(case[key])]
        if case['uplo'] != 'N':
            cmd += ['--uplo', case['uplo']]
        if backend != 'zynum':
            cmd += ['--library', str(getattr(args, backend))]
    else:
        for key in ('scalar', 'rows', 'columns', 'pattern', 'transpose', 'rhs', 'index_bits'):
            cmd += ['--index' if key == 'index_bits' else '--' + key, str(case[key])]
        if case['pad']:
            cmd += ['--pad']
        if backend == 'accelerate':
            cmd += ['--adapter', str(args.sparse_adapter)]
    return cmd


def validate(output, module, backend, case):
    rows = list(csv.reader(io.StringIO(output.decode('utf-8')), strict=True))
    fields = DENSE if module == 'lapack' else SPARSE
    require(len(rows) == (1 if module == 'lapack' else 2), 'unexpected CSV row count')
    if module == 'sparse':
        require(rows[0] == fields, 'sparse schema mismatch')
    require(len(rows[-1]) == len(fields), 'CSV field count mismatch')
    row = dict(zip(fields, rows[-1]))
    for key, expected in case.items():
        require(row[key] == str(expected), f'case mismatch: {key}')
    label = {'zynum': 'zynum-checked', 'openblas': 'openblas-lapacke-work-lp64', 'accelerate': 'accelerate-new-lp64'}[backend] if module == 'lapack' else backend
    require(row['backend'] == label and row['check'] == ('numerics-padding-ok' if module == 'lapack' else 'checked-ok'), 'backend/check mismatch')
    numeric = ['best_ns', 'median_ns', 'p95_ns', 'scaled_residual']
    if module == 'lapack':
        require(int(row['lda']) == case['m'] and int(row['ldb']) == case['n'], 'leading dimension mismatch')
        changed = int(row['factor_unused_triangle_modified_max'])
        require(0 <= changed <= case['n'] * (case['n'] - 1) // 2, 'invalid unused-triangle modification count')
        require(changed == 0 or (backend == 'accelerate' and case['uplo'] == 'U' and case['kind'][1:] in ('potrf', 'potrs')), 'unexpected unused-triangle modification')
        row['factor_unused_triangle_modified_max'] = changed
    else:
        require(row['padding'] == 'checked-ok', 'padding check failed')
        require(row['setup_kind'] == ('index_alloc_convert_and_csr_validate' if backend == 'zynum' else 'index_alloc_convert_64offset_32column'), 'setup scheme mismatch')
        require(row['execution_kind'] == ('checked_public_api' if backend == 'zynum' else 'public_SparseMultiply_borrowed_adapter'), 'execution scheme mismatch')
        require(row['index_layout'] == ('i32_offsets_i32_columns' if backend == 'zynum' else 'i64_offsets_i32_columns'), 'index scheme mismatch')
        require(int(row['nnz']) > 0 and int(row['setup_ns']) >= 0, 'invalid setup/nnz')
        numeric += ['max_ns', 'gflops', 'max_absolute_error', 'checksum']
    for key in numeric:
        row[key] = float(row[key])
        require(math.isfinite(row[key]), f'nonfinite {key}')
    row['sample_ns'] = [float(value) for value in row['sample_ns'].split(';')]
    samples = sorted(row['sample_ns'])
    require(int(row['samples']) == len(samples) == 30 and all(math.isfinite(v) and v > 0 for v in samples), 'invalid samples')
    require(1 <= int(row['batch_calls']) <= 1 << 20 and int(row['min_batch_ns']) > 0, 'invalid batch')
    require(row['batch_target_met'] == str(int(row['min_batch_ns']) >= 100000).lower(), 'batch target mismatch')
    tolerance = 0.00011 if module == 'sparse' else 0.0000011
    for key, value in [('best_ns', samples[0]), ('median_ns', statistics.median(samples)), ('p95_ns', samples[28])] + ([('max_ns', samples[-1])] if module == 'sparse' else []):
        require(row[key] > 0 and math.isclose(row[key], value, abs_tol=tolerance, rel_tol=1e-12), f'inconsistent {key}')
    require(math.isclose(int(row['min_batch_ns']), samples[0] * int(row['batch_calls']), abs_tol=tolerance * int(row['batch_calls']) + 1, rel_tol=1e-12), 'batch/sample mismatch')
    limit = 100 if module == 'lapack' else (2e-5 if case['scalar'] == 'f32' else 2e-12)
    require(0 <= row['scaled_residual'] <= limit, 'residual threshold failed')
    if module == 'sparse':
        require(row['max_absolute_error'] >= 0 and row['gflops'] > 0, 'invalid error/throughput')
    return row


def ratio_summary(ratios, **details):
    return dict(cases=len(ratios), zynum_strict_wins=sum(x > 1 for x in ratios), ties=sum(x == 1 for x in ratios), geometric_mean_speedup=math.exp(statistics.mean(map(math.log, ratios))) if ratios else None, **details)


def aggregate(records, selected):
    rows, summary = [], {}
    for module, suite in selected.items():
        timings, timed_deviation, setup_deviation = {}, {}, {}
        for index, case in enumerate(suite):
            nnz = set()
            for backend in BACKENDS[module]:
                group = [r['result'] for r in records if (r['module'], r['case_index'], r['backend']) == (module, index, backend)]
                require(len(group) == REPEATS[module], 'missing process repetitions')
                medians = [r['median_ns'] for r in group]
                timings[index, backend] = statistics.median(medians)
                if module == 'sparse':
                    nnz.update(int(r['nnz']) for r in group)
                changes = [r['factor_unused_triangle_modified_max'] for r in group] if module == 'lapack' else []
                changed = max(changes, default=0)
                timed_deviation[index, backend] = changed > 0 and case.get('kind', '')[1:] == 'potrf'
                setup_deviation[index, backend] = changed > 0 and case.get('kind', '')[1:] == 'potrs'
                rows.append(dict(module=module, case_id=f'{module}-{index:03d}', backend=backend, case_json=json.dumps(case, sort_keys=True, separators=(',', ':')), process_repeats=len(group), inner_samples=30 * len(group), median_ns=statistics.median(medians), median_process_p95_ns=statistics.median(r['p95_ns'] for r in group), min_process_best_ns=min(r['best_ns'] for r in group), max_scaled_residual=max(r['scaled_residual'] for r in group), factor_unused_triangle_modified_max=changed if changes else '', process_factor_unused_triangle_modified_max=';'.join(map(str, changes)), timed_factor_storage_deviation=timed_deviation[index, backend], untimed_factor_setup_storage_deviation=setup_deviation[index, backend], all_batch_target_met=all(r['batch_target_met'] == 'true' for r in group), process_median_ns=';'.join(map(str, medians))))
            require(module != 'sparse' or len(nnz) == 1, 'cross-backend fixture nnz mismatch')
        for competitor in BACKENDS[module][1:]:
            ratios = [timings[i, competitor] / timings[i, 'zynum'] for i in range(len(suite))]
            conforming = [i for i in range(len(suite)) if not timed_deviation[i, competitor]]
            summary[module + ':numerical-output-only:' + competitor] = ratio_summary(ratios, cases_without_observed_timed_storage_violation=len(conforming), cases_with_timed_storage_deviation=len(suite) - len(conforming), cases_with_untimed_factor_setup_storage_deviation=sum(setup_deviation[i, competitor] for i in range(len(suite))))
            summary[module + ':no-observed-timed-storage-violation-subset:' + competitor] = ratio_summary([ratios[i] for i in conforming], case_ids=[f'{module}-{i:03d}' for i in conforming], full_selected_case_set=len(conforming) == len(suite))
        best_ratios = [min(timings[i, competitor] for competitor in BACKENDS[module][1:]) / timings[i, 'zynum'] for i in range(len(suite))]
        summary[module + ':numerical-output-only:fastest-comparator'] = ratio_summary(best_ratios)
        primary = {}
        for i in range(len(suite)):
            eligible = [b for b in BACKENDS[module][1:] if not timed_deviation[i, b]]
            require(eligible, f'no comparator without observed timed storage violation: {module}-{i:03d}')
            primary[i] = min(eligible, key=lambda b: timings[i, b])
        summary[module + ':primary-fastest-comparator-with-no-observed-timed-storage-violation'] = ratio_summary([timings[i, primary[i]] / timings[i, 'zynum'] for i in range(len(suite))], full_selected_case_set=True, selected_comparator_by_case={f'{module}-{i:03d}': b for i, b in primary.items()}, selected_backend_case_counts={b: list(primary.values()).count(b) for b in BACKENDS[module][1:]})
    return rows, summary


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ('build-manifest', 'lapack-worker', 'sparse-worker', 'sparse-adapter', 'openblas', 'accelerate', 'output'):
        parser.add_argument('--' + flag, required=True, type=lambda s: Path(s).absolute())
    parser.add_argument('--verify-only', action='store_true', help='verify build/source/library bindings without executing workers')
    parser.add_argument('--module', choices=('all', 'lapack', 'sparse'), default='all')
    parser.add_argument('--suite', choices=('small', 'full'), default='small')
    parser.add_argument('--limit-cases', type=int, help='positive limit per selected module; smoke results stay labeled limited')
    args = parser.parse_args()
    require(args.limit_cases is None or args.limit_cases > 0, '--limit-cases must be positive')
    require(not args.output.exists(), 'output already exists; choose a new directory')
    selected = {m: cases(m, args.suite)[:args.limit_cases] for m in BACKENDS if args.module in ('all', m)}
    schedule_path = ROOT / 'bench/tools/report_schedule.py'
    spec = importlib.util.spec_from_file_location('foundation_report_schedule', schedule_path)
    scheduler = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(scheduler)
    artifact_names = ['accelerate'] + (['lapack_worker', 'openblas'] if 'lapack' in selected else []) + (['sparse_worker', 'sparse_adapter'] if 'sparse' in selected else [])
    artifacts = {name: identity(getattr(args, name), name == 'accelerate') for name in artifact_names}
    build_manifest_path = args.build_manifest
    build_manifest = json.loads(build_manifest_path.read_text())
    require(build_manifest.get('schema_version') == 1 and build_manifest.get('status') == 'complete', 'unsupported or incomplete build manifest')
    require(build_manifest['repository'] == str(ROOT), 'build manifest belongs to a different checkout; rebuild here')
    for name in ('openblas', 'accelerate'):
        if name in artifacts:
            require(artifacts[name] == build_manifest['libraries'][name], 'comparator library differs from recorded build')
    built_paths = {'lapack-sweep': args.lapack_worker, 'sparse-sweep': args.sparse_worker, 'accelerate_sparse_adapter.dylib': args.sparse_adapter}
    require(set(build_manifest['binary_sha256']) == set(built_paths), 'build manifest must bind both workers and sparse adapter')
    for name, digest in build_manifest['binary_sha256'].items():
        require(identity(built_paths[name])['sha256'] == digest, 'binary differs from recorded build')
    build_manifest_identity = identity(build_manifest_path)
    source_archive_path = Path(build_manifest['source_archive']['path'])
    source_archive_identity = identity(source_archive_path)
    require(source_archive_identity['sha256'] == build_manifest['source_archive']['sha256'], 'source archive differs from recorded build')
    with tarfile.open(source_archive_path, 'r:gz') as archive:
        archived_sources = {}
        for member in archive:
            require(member.isfile() and member.name not in archived_sources, 'unexpected or duplicate source archive member')
            archived_sources[member.name] = hashlib.sha256(archive.extractfile(member).read()).hexdigest()
        require(archived_sources == build_manifest['source_sha256'], 'source archive contents differ from recorded build')
    for path, digest in build_manifest['source_sha256'].items():
        require(not Path(path).is_absolute() and '..' not in Path(path).parts, 'source manifest path must stay repository relative')
        require(identity(ROOT / path)['sha256'] == digest, 'source differs from recorded build')
    require(str(Path(__file__).relative_to(ROOT)) in build_manifest['source_sha256'], 'controller missing from build source manifest')
    require(str(schedule_path.relative_to(ROOT)) in build_manifest['source_sha256'], 'shared scheduler missing from build source manifest')
    if args.verify_only:
        print(json.dumps({'status': 'build-bindings-verified', 'build_manifest': str(build_manifest_path), 'workers_executed': False}, sort_keys=True))
        return
    source = source_state()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=args.output.name + '.incomplete-', dir=args.output.parent))
    (stage / 'raw').mkdir()
    (stage / 'source-snapshot.tar.gz').write_bytes(source_archive_path.read_bytes())
    require(identity(stage / 'source-snapshot.tar.gz')['sha256'] == source_archive_identity['sha256'], 'source archive changed while copying')
    (stage / 'controller.py').write_bytes(Path(__file__).read_bytes())
    (stage / 'report_schedule.py').write_bytes(schedule_path.read_bytes())
    metadata = dict(status='running', started_utc=datetime.now(timezone.utc).isoformat(), controller_argv=sys.argv, repository=str(ROOT), source=source, dirty=bool(source['status_porcelain']), artifacts=artifacts, schedule_source=identity(schedule_path), environment=ENV, thread_policy='Requested caps=1 for every backend; Zynum foundation currently serial. Caps are configuration, not measured thread counts.', operating_system=dict(uname=platform.uname()._asdict(), platform=platform.platform(), cpu_count=os.cpu_count()), suite=args.suite, limited=args.limit_cases is not None, cases=selected, process_repeats=REPEATS, inner_samples=30, timeout_seconds=300, scheduling='shared library_repeat_schedule interleaved; strictly serialized', build_identity_note='The explicit build manifest binds source archive, compiler/build commands, headers, libraries and binary hashes; source and Git state are recorded honestly, without a clean-HEAD claim.', ratio_definition='competitor median-of-process-medians / Zynum median-of-process-medians; >1 is a strict Zynum win; each case has equal log weight')
    records = []
    try:
        metadata['comparison_policy'] = dict(primary='Per case, fastest comparator with no observed timed storage violation; every selected case must have an eligible comparator. Upper Accelerate POTRF with changed unused elements is excluded; OpenBLAS remains eligible. Limited suites remain smoke runs.', secondary='Numerical-output-only summaries retain every comparator and all selected cases. Per-backend storage-checked subsets are separately labeled and never replace the full-case primary comparison.', lapack_check='numerics-padding-ok validates selected numerical output, factor diagonals and padding; it does not claim complete LAPACK storage conformance.', storage_deviation='Observed Accelerate upper POTRF can modify its unused lower triangle, contrary to LAPACK storage conventions. factor_unused_triangle_modified_max records the maximum count across factorization checks, including untimed POTRS factor setup. POTRS setup deviations do not exclude timed solves: complete factors/pivots remain immutable during solves. Zynum and all other operations retain strict unused-triangle preservation.', storage_references=['https://www.netlib.org/lapack/lug/node121.html', 'https://www.netlib.org/lapack/lug/node122.html'], equivalence_limit='No numerical or storage summary establishes full interface equivalence. All primary eligibility decisions use observed timed-operation storage behavior; untimed setup deviations remain disclosed.')
        metadata['build_manifest'] = build_manifest
        metadata['build_manifest_identity'] = build_manifest_identity
        metadata['reproduction_files'] = {name: identity(stage / name)['sha256'] for name in ('source-snapshot.tar.gz', 'controller.py', 'report_schedule.py')}
        metadata['python'] = {'version': sys.version, 'executable': sys.executable}
        if sys.platform == 'darwin':
            metadata['operating_system'].update(cpu_brand=capture('sysctl', '-n', 'machdep.cpu.brand_string'), hardware_model=capture('sysctl', '-n', 'hw.model'), os_release=capture('sw_vers'))
        write_json(stage / 'metadata.pending.json', metadata)
        with (stage / 'runs.jsonl').open('w') as journal:
            for module, suite in selected.items():
                for backend_index, case_index, repeat in scheduler.library_repeat_schedule(len(BACKENDS[module]), REPEATS[module], 'interleaved', len(suite)):
                    backend, case = BACKENDS[module][backend_index], suite[case_index]
                    cmd = command(args, module, backend, case)
                    stem = f'{module}-{case_index:03d}-{backend}-{repeat:02d}'
                    record = dict(module=module, case_index=case_index, backend=backend, repeat=repeat, command=cmd, cwd=str(ROOT), environment=ENV, stdout=f'raw/{stem}.csv', stderr=f'raw/{stem}.stderr')
                    try:
                        process = subprocess.run(cmd, cwd=ROOT, env=ENV, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=300)
                        stdout, stderr, code = process.stdout, process.stderr, process.returncode
                    except subprocess.TimeoutExpired as exc:
                        stdout, stderr, code = exc.stdout or b'', exc.stderr or b'', 'timeout'
                    except OSError as exc:
                        stdout, stderr, code = b'', str(exc).encode(), 'spawn-error'
                    (stage / record['stdout']).write_bytes(stdout)
                    (stage / record['stderr']).write_bytes(stderr)
                    record['returncode'] = code
                    record.update(stdout_sha256=hashlib.sha256(stdout).hexdigest(), stderr_sha256=hashlib.sha256(stderr).hexdigest())
                    journal.write(json.dumps(record, allow_nan=False) + '\n')
                    journal.flush()
                    require(code == 0, f'worker failed: {stem}; returncode={code}')
                    record['result'] = validate(stdout, module, backend, case)
                    records.append(record)
                    print(f'[{len(records)}] {stem} checked', file=sys.stderr, flush=True)
        require(source_state() == source, 'source or Git state changed during measurement')
        for path, digest in build_manifest['source_sha256'].items():
            require(identity(ROOT / path)['sha256'] == digest, 'build source changed during measurement')
        require(identity(build_manifest_path) == build_manifest_identity, 'build manifest changed during measurement')
        require(identity(source_archive_path) == source_archive_identity, 'source archive changed during measurement')
        require({name: identity(getattr(args, name), name == 'accelerate') for name in artifact_names} == artifacts, 'worker/library identity changed during measurement')
        rows, summary = aggregate(records, selected)
        metadata.update(status='complete', completed_utc=datetime.now(timezone.utc).isoformat(), case_counts={m: len(c) for m, c in selected.items()}, aggregate_rows=len(rows), process_samples=len(records), inner_timing_samples=30 * len(records), summary=summary, runs=records)
        with (stage / 'aggregate.csv').open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        metadata['aggregate_sha256'] = identity(stage / 'aggregate.csv')['sha256']
        write_json(stage / 'metadata.json', metadata)
        (stage / 'metadata.pending.json').unlink()
        require(not args.output.exists(), 'output appeared during run')
        stage.rename(args.output)
        print(args.output)
    except BaseException as exc:
        metadata.update(status='failed', error=f'{type(exc).__name__}: {exc}', validated_process_samples=len(records), runs=records)
        write_json(stage / 'failure.json', metadata)
        print(f'No result published; diagnostics retained at {stage}', file=sys.stderr)
        raise


if __name__ == '__main__':
    main()
