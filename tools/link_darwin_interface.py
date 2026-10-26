#!/usr/bin/env python3
# Copyright (C) 2026 Zynum contributors
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Link a finite Mach-O interface with its reviewed public symbol set."""
import argparse
import json
import re
import os
from pathlib import Path
import subprocess
import tempfile
import sys
sys.dont_write_bytecode = True

def unpack_lapacke(value, parser):
    if value.get('schema') == 'zynum-lapacke-composition-v1':
        from lapacke_components import unwrap
        try:
            return unwrap(value)
        except ValueError as error:
            parser.error(str(error))
    return value, []

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('archive', type=Path)
    parser.add_argument('manifest', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--macos-min', required=True)
    parser.add_argument('--legacy-manifest', type=Path)
    parser.add_argument('--lapacke-manifest', type=Path)
    parser.add_argument('--fft-manifest', type=Path)
    parser.add_argument('--fft-f32-manifest', type=Path)
    parser.add_argument('--operator-manifest', type=Path)
    parser.add_argument('--sparse-manifest', type=Path)
    parser.add_argument('--blas-manifest', type=Path)
    parser.add_argument('--checked-cholesky-manifest', type=Path)
    parser.add_argument('--feast-manifest', type=Path)
    parser.add_argument('--contour-manifest', type=Path)
    args = parser.parse_args()
    if args.lapacke_manifest is not None and args.legacy_manifest is None:
        parser.error('combined LAPACKE requires the shared checked ABI manifest')
    if args.checked_cholesky_manifest is not None and args.legacy_manifest is None:
        parser.error('checked Cholesky requires the shared checked ABI manifest')
    if not re.fullmatch(r'[0-9]+\.[0-9]+\.[0-9]+', args.macos_min):
        parser.error('macos-min must be the configured major.minor.patch')
    manifest = json.loads(args.manifest.read_text())
    if args.legacy_manifest is None:
        manifest, extra_lapacke_symbols = unpack_lapacke(manifest, parser)
        if manifest.get('schema') != 'zynum-finite-interface-subset-v1':
            parser.error('unsupported interface manifest schema')
        plan = manifest['plan']
        if plan.get('profile') not in ('mkl2026.lp64', 'mkl2026.ilp64'):
            parser.error('unsupported LAPACKE integer profile')
        target = plan['target']
        if target['arch'] != 'aarch64' or target['os'] != 'macos':
            parser.error('requires the reviewed Apple arm64 representation')
        names = plan['symbols']
        if not names or len(names) != len(set(names)) or any(not isinstance(n, str) or not re.fullmatch(r'LAPACKE_[A-Za-z0-9_]+', n) for n in names):
            parser.error('invalid finite LAPACKE export list')
        exports = ['_' + n for n in names + extra_lapacke_symbols] + ['_LAPACKE_xerbla', '_xerbla_']
    else:
        plan = manifest.get('plan', {})
        if (plan.get('profile'), plan.get('set'), plan.get('target'), plan.get('pointer_bits')) != ('checked-sparse-v1', 'csr-products-independent-indices-v1', 'apple-arm64-default-c-v1', 64):
            parser.error('unsupported checked Sparse manifest contract')
        legacy = json.loads(args.legacy_manifest.read_text())
        if (legacy.get('schema_id'), legacy.get('abi_revision'), legacy.get('target_pointer_bits')) != ('zynum-checked-abi-v1', 1, 64):
            parser.error('unsupported shared checked ABI contract')
        names = plan['symbols'] + [row['name'] for row in legacy['exports']]
        if len(plan['symbols']) != 128 or legacy.get('export_count') != 8 or len(legacy['exports']) != 8:
            parser.error('unexpected checked Sparse export coverage')
        if len(names) != len(set(names)) or any(not isinstance(n, str) or not re.fullmatch(r'zynum_checked_[A-Za-z0-9_]+', n) for n in names):
            parser.error('invalid checked Sparse export list')
        exports = ['_' + n for n in names]
        if args.lapacke_manifest is not None:
            lapacke = json.loads(args.lapacke_manifest.read_text())
            lapacke, extra_lapacke_symbols = unpack_lapacke(lapacke, parser)
            selection = lapacke.get('plan', {})
            target = selection.get('target', {})
            supported_sets = {
                'potrf-potrs-getrf-getrs-gesv-stev-gelsd-ordinary-lp64-v1': False,
                'potrf-potrs-getrf-getrs-gesv-stev-gelsd-heev-hegv-ordinary-lp64-v1': True,
                'potrf-potrs-getrf-getrs-gesv-stev-gelsd-heev-hegv-geqrf-ormqr-unmqr-ordinary-lp64-v1': True,
                'potrf-potrs-getrf-getrs-gesv-stev-gelsd-heev-hegv-geqrf-ormqr-unmqr-orgqr-ungqr-ordinary-lp64-v1': True,
                'potrf-potrs-getrf-getrs-gesv-stev-gelsd-heev-hegv-complex-gelsd-ordinary-lp64-v1': True,
                'potrf-potrs-getrf-getrs-gesv-stev-gelsd-heev-hegv-geqrf-ormqr-unmqr-orgqr-ungqr-complex-gelsd-ordinary-lp64-v1': True,
                'potrf-potrs-getrf-getrs-gesv-stev-gelsd-heev-hegv-gesvd-ordinary-lp64-v1': True,
                'potrf-potrs-getrf-getrs-gesv-stev-gelsd-heev-hegv-complex-gelsd-gesvd-ordinary-lp64-v1': True,
                'potrf-potrs-getrf-getrs-gesv-stev-gelsd-heev-hegv-geqrf-ormqr-unmqr-orgqr-ungqr-gesvd-ordinary-lp64-v1': True,
                'potrf-potrs-getrf-getrs-gesv-stev-gelsd-heev-hegv-geqrf-ormqr-unmqr-orgqr-ungqr-complex-gelsd-gesvd-ordinary-lp64-v1': True,
            }
            # Eight explicit installable POSV combinations; no arbitrary
            # declaration set or smaller standalone subset enters this bundle.
            supported_sets.update({
                'potrf-potrs-getrf-getrs-gesv-stev-gelsd-heev-hegv'
                + ('-geqrf-ormqr-unmqr-orgqr-ungqr' if qr else '')
                + ('-complex-gelsd' if complex_gelsd else '')
                + ('-gesvd' if svd else '') + '-posv-ordinary-lp64-v1': True
                for qr in (False, True) for complex_gelsd in (False, True) for svd in (False, True)
            })
            # Sixteen exact install selections extend the existing bundle;
            # standalone and historical partial declarations stay separate.
            supported_sets.update({
                'potrf-potrs-getrf-getrs-gesv-stev-gelsd-heev-hegv'
                + ('-geqrf-ormqr-unmqr-orgqr-ungqr' if qr else '')
                + ('-complex-gelsd' if complex_gelsd else '')
                + ('-gesvd' if svd else '')
                + ('-posv' if posv else '') + '-trtrs-ordinary-lp64-v1': True
                for qr in (False, True) for complex_gelsd in (False, True)
                for svd in (False, True) for posv in (False, True)
            })
            supported_sets.update({base.replace('-ordinary-', '-trtri-ordinary-'): hermitian for base, hermitian in tuple(supported_sets.items())})
            supported_sets.update({base.replace('-ordinary-', '-getri-ordinary-'): hermitian for base, hermitian in tuple(supported_sets.items())})
            if selection.get('set') not in supported_sets or lapacke.get('schema') != 'zynum-finite-interface-subset-v1' or (
                selection.get('profile'), target.get('id'),
                target.get('arch'), target.get('os'), target.get('pointer_bits')
            ) != ('mkl2026.lp64', 'apple-arm64-default-c-v1', 'aarch64', 'macos', 64):
                parser.error('unsupported combined LP64 LAPACKE contract')
            expected_lapacke = {
                'LAPACKE_' + scalar + routine + suffix
                for scalar in 'sdcz' for routine in ('potrf', 'potrs', 'getrf', 'getrs', 'gesv')
                for suffix in ('', '_work')
            } | {
                'LAPACKE_' + scalar + routine + suffix
                for scalar in 'sd' for routine in ('stev', 'gelsd') for suffix in ('', '_work')
            }
            if supported_sets[selection['set']]:
                expected_lapacke |= {
                    'LAPACKE_' + scalar + routine + suffix
                    for scalar in 'cz' for routine in ('heev', 'hegv') for suffix in ('', '_work')
                }
            base_set = selection['set'].replace('-getri-', '-').replace('-trtri-', '-').replace('-trtrs-', '-').replace('-posv-', '-').replace('-gesvd-', '-').replace('-complex-gelsd-', '-')
            if base_set in ('potrf-potrs-getrf-getrs-gesv-stev-gelsd-heev-hegv-geqrf-ormqr-unmqr-ordinary-lp64-v1','potrf-potrs-getrf-getrs-gesv-stev-gelsd-heev-hegv-geqrf-ormqr-unmqr-orgqr-ungqr-ordinary-lp64-v1'):
                expected_lapacke |= {'LAPACKE_'+scalar+'geqrf'+suffix for scalar in 'sdcz' for suffix in ('','_work')}
                expected_lapacke |= {'LAPACKE_'+scalar+routine+suffix for scalar,routine in [('s','ormqr'),('d','ormqr'),('c','unmqr'),('z','unmqr')] for suffix in ('','_work')}
            if base_set == 'potrf-potrs-getrf-getrs-gesv-stev-gelsd-heev-hegv-geqrf-ormqr-unmqr-orgqr-ungqr-ordinary-lp64-v1':
                expected_lapacke |= {'LAPACKE_'+scalar+routine+suffix for scalar,routine in [('s','orgqr'),('d','orgqr'),('c','ungqr'),('z','ungqr')] for suffix in ('','_work')}
            if '-complex-gelsd-' in selection['set']:
                expected_lapacke |= {'LAPACKE_'+scalar+'gelsd'+suffix for scalar in 'cz' for suffix in ('','_work')}
            if '-gesvd-' in selection['set']:
                expected_lapacke |= {'LAPACKE_'+scalar+'gesvd'+suffix for scalar in 'sdcz' for suffix in ('','_work')}
            if '-posv-' in selection['set']:
                expected_lapacke |= {'LAPACKE_'+scalar+'posv'+suffix for scalar in 'sdcz' for suffix in ('','_work')}
            if '-trtrs-' in selection['set']:
                expected_lapacke |= {'LAPACKE_'+scalar+'trtrs'+suffix for scalar in 'sdcz' for suffix in ('','_work')}
            if '-trtri-' in selection['set']:
                expected_lapacke |= {'LAPACKE_'+scalar+'trtri'+suffix for scalar in 'sdcz' for suffix in ('','_work')}
            if '-getri-' in selection['set']:
                expected_lapacke |= {'LAPACKE_'+scalar+'getri'+suffix for scalar in 'sdcz' for suffix in ('','_work')}
            lapacke_names = selection.get('symbols', [])
            if (not isinstance(lapacke_names, list) or
                any(not isinstance(n, str) for n in lapacke_names) or
                len(lapacke_names) != len(expected_lapacke) or set(lapacke_names) != expected_lapacke):
                parser.error('invalid combined LP64 LAPACKE export list')
            exports += ['_' + n for n in lapacke_names + extra_lapacke_symbols] + ['_LAPACKE_xerbla', '_xerbla_']
    if args.fft_manifest is not None:
        if args.lapacke_manifest is None:
            parser.error('FFTW requires the unified one-owner LP64 package')
        fft = json.loads(args.fft_manifest.read_text())
        if (fft.get('schema'), fft.get('profile'), fft.get('target')) != (
            'zynum-external-interface-declarations-v1', 'fftw3.f64.abplas-v1', 'apple-arm64-default-c-v1'
        ):
            parser.error('unsupported finite FFTW interface contract')
        expected = {'fftw_plan_dft_1d', 'fftw_plan_dft_r2c_1d', 'fftw_plan_dft_r2c_3d',
                    'fftw_plan_dft_c2r_3d', 'fftw_plan_r2r_1d', 'fftw_execute',
                    'fftw_destroy_plan', 'fftw_alloc_complex', 'fftw_free',
                    'fftw_alignment_of', 'fftw_execute_dft', 'fftw_execute_dft_r2c',
                    'fftw_execute_dft_c2r', 'fftw_execute_r2r',
                    'fftw_plan_dft_2d', 'fftw_plan_dft_3d', 'fftw_plan_dft', 'fftw_plan_dft_r2c_2d', 'fftw_plan_dft_r2c', 'fftw_plan_dft_c2r_1d', 'fftw_plan_dft_c2r_2d', 'fftw_plan_dft_c2r', 'fftw_plan_r2r_2d', 'fftw_plan_r2r_3d', 'fftw_plan_r2r', 'fftw_plan_many_dft', 'fftw_plan_guru_dft', 'fftw_plan_guru64_dft', 'fftw_plan_many_dft_r2c', 'fftw_plan_many_dft_c2r', 'fftw_plan_many_r2r', 'fftw_plan_guru_dft_r2c', 'fftw_plan_guru_dft_c2r', 'fftw_plan_guru_r2r', 'fftw_plan_guru64_dft_r2c', 'fftw_plan_guru64_dft_c2r', 'fftw_plan_guru64_r2r', 'fftw_plan_guru_split_dft', 'fftw_plan_guru_split_dft_r2c', 'fftw_plan_guru_split_dft_c2r', 'fftw_plan_guru64_split_dft', 'fftw_plan_guru64_split_dft_r2c', 'fftw_plan_guru64_split_dft_c2r', 'fftw_execute_split_dft', 'fftw_execute_split_dft_r2c', 'fftw_execute_split_dft_c2r', 'fftw_malloc', 'fftw_alloc_real'}
        names = fft.get('symbols', [])
        if len(names) != len(expected) or set(names) != expected:
            parser.error('invalid finite FFTW public symbol set')
        exports += ['_' + name for name in names]
    if args.fft_f32_manifest is not None:
        if args.fft_manifest is None:
            parser.error('single-precision FFTW requires the selected f64 FFTW base profile')
        single = json.loads(args.fft_f32_manifest.read_text())
        if (single.get('schema'), single.get('profile'), single.get('target'), single.get('requires_profiles')) != (
            'zynum-external-interface-declarations-v1', 'fftw3.f32.abplas-v1', 'apple-arm64-default-c-v1', ['fftw3.f64.abplas-v1']
        ):
            parser.error('unsupported single-precision FFTW companion contract')
        from generate_fftw import REVIEWED_SYMBOLS
        expected_single = {name.replace('fftw_', 'fftwf_', 1) for name in REVIEWED_SYMBOLS}
        single_names = single.get('symbols', [])
        if len(single_names) != 48 or set(single_names) != expected_single:
            parser.error('invalid single-precision FFTW public symbol set')
        exports += ['_' + name for name in single_names]
    if args.operator_manifest is not None:
        if args.lapacke_manifest is None:
            parser.error('operator interface requires the unified one-owner LP64 package')
        operator = json.loads(args.operator_manifest.read_text())
        if (operator.get('schema'), operator.get('profile'), operator.get('target'), operator.get('pointer_bits')) != ('zynum-operator-interface-v1', 'zynum-v1', 'apple-arm64-default-c-v1', 64):
            parser.error('unsupported operator interface contract')
        expected = {
            'zynum_operator_' + routine + '_' + scalar + suffix
            for scalar in ('f32', 'f64', 'complex_f32', 'complex_f64')
            for routine, suffix in (('chebyshev', ''), ('chebyshev_step', ''), ('lanczos', ''), ('csr_spmv', '_i64_i32_admitted'), ('csr_spmm', '_i64_i32_admitted'))
        }
        bicgstab = operator.get('bicgstab', False)
        if type(bicgstab) is not bool:
            parser.error('invalid operator BiCGStab selection')
        if bicgstab:
            if type(operator.get('linear_result_bytes')) is not int or operator['linear_result_bytes'] != 48:
                parser.error('invalid operator linear result layout')
            expected.update('zynum_operator_bicgstab_' + scalar
                for scalar in ('f32', 'f64', 'complex_f32', 'complex_f64'))
        rational_filter = operator.get('rational_filter', False)
        if type(rational_filter) is not bool:
            parser.error('invalid operator rational filter selection')
        if rational_filter:
            if (not bicgstab or type(operator.get('rational_result_bytes')) is not int or
                operator['rational_result_bytes'] != 72):
                parser.error('invalid operator rational filter layout or dependency')
            expected.update('zynum_operator_rational_filter_' + scalar
                for scalar in ('complex_f32', 'complex_f64'))
        projection = operator.get('subspace_projection', False)
        if type(projection) is not bool:
            parser.error('invalid operator subspace projection selection')
        if projection:
            if (type(operator.get('block_reduction_bytes')) is not int or operator['block_reduction_bytes'] != 16 or
                type(operator.get('panel_bytes')) is not int or operator['panel_bytes'] != 48):
                parser.error('invalid operator subspace projection layout')
            expected.update('zynum_operator_subspace_project_' + scalar
                for scalar in ('f32', 'f64', 'complex_f32', 'complex_f64'))
        series_set = operator.get('chebyshev_series_set', False)
        if type(series_set) is not bool:
            parser.error('invalid operator series-set selection')
        if series_set:
            if type(operator.get('series_count_integer_bits')) is not int or operator['series_count_integer_bits'] != 64:
                parser.error('invalid operator series-set count representation')
            expected.update('zynum_operator_chebyshev_series_set_' + scalar
                for scalar in ('f32', 'f64', 'complex_f32', 'complex_f64'))
        normalized_filter = operator.get('normalized_chebyshev_filter', False)
        if type(normalized_filter) is not bool:
            parser.error('invalid normalized operator filter selection')
        if normalized_filter:
            if type(operator.get('panel_bytes')) is not int or operator['panel_bytes'] != 48:
                parser.error('invalid normalized operator panel layout')
            expected.update('zynum_operator_' + routine + '_' + scalar
                for scalar in ('f32', 'f64', 'complex_f32', 'complex_f64')
                for routine in ('normalized_filter', 'normalized_filter_panel'))
        admitted_fft = operator.get('fftw_admitted', False)
        phase_agreement = operator.get('phase_agreement', False)
        if type(phase_agreement) is not bool:
            parser.error('invalid operator phase agreement selection')
        if phase_agreement:
            from generate_operator_interface import phase_stages
            stages = operator.get('phase_stages', {})
            if (type(operator.get('phase_point_bytes')) is not int or operator['phase_point_bytes'] != 24 or
                type(operator.get('agreement_bytes')) is not int or operator['agreement_bytes'] != 16 or
                not isinstance(stages, dict) or any(type(value) is not int for value in stages.values()) or
                stages != {name: value for name, value in phase_stages() if value <= (49 if rational_filter else 44 if bicgstab else 37)}):
                parser.error('invalid canonical operator phase agreement layout or stages')
            expected.update('zynum_operator_' + routine + '_agreed_' + scalar
                for scalar in ('f32', 'f64', 'complex_f32', 'complex_f64')
                for routine in ('chebyshev_step', 'chebyshev', 'lanczos', 'lanczos_chain'))
        if phase_agreement and rational_filter:
            expected.update('zynum_operator_rational_filter_agreed_' + scalar
                for scalar in ('complex_f32', 'complex_f64'))
        if phase_agreement and bicgstab:
            expected.update('zynum_operator_bicgstab_agreed_' + scalar
                for scalar in ('f32', 'f64', 'complex_f32', 'complex_f64'))
        if phase_agreement and projection:
            expected.update('zynum_operator_subspace_project_agreed_' + scalar
                for scalar in ('f32', 'f64', 'complex_f32', 'complex_f64'))
        if phase_agreement and series_set:
            expected.update('zynum_operator_chebyshev_series_set_agreed_' + scalar
                for scalar in ('f32', 'f64', 'complex_f32', 'complex_f64'))
        if phase_agreement and normalized_filter:
            expected.update('zynum_operator_' + routine + '_agreed_' + scalar
                for scalar in ('f32', 'f64', 'complex_f32', 'complex_f64')
                for routine in ('normalized_filter', 'normalized_filter_panel'))
        if operator.get('lanczos_chain') is not True:
            parser.error('operator interface requires the reviewed full-chain declarations')
        expected.update('zynum_operator_lanczos_chain_' + scalar for scalar in ('f32', 'f64', 'complex_f32', 'complex_f64'))
        if type(admitted_fft) is not bool:
            parser.error('invalid operator FFT forwarding declaration')
        if admitted_fft:
            if args.fft_manifest is None:
                parser.error('operator FFT forwarding requires the FFTW profile')
            expected.add('zynum_operator_fftw_execute_admitted')
        names = operator.get('symbols', [])
        if not isinstance(names, list) or len(names) != len(expected) or any(not isinstance(n, str) for n in names) or set(names) != expected:
            parser.error('invalid operator public symbol set')
        exports += ['_' + name for name in names]
    if args.sparse_manifest is not None:
        if args.lapacke_manifest is None:
            parser.error('standard Sparse requires the unified one-owner LP64 package')
        sparse = json.loads(args.sparse_manifest.read_text())
        selection = sparse.get('plan', {})
        target = selection.get('target', {})
        if (sparse.get('schema'), selection.get('profile'), selection.get('set'), target.get('id'), target.get('arch'), target.get('os'), target.get('pointer_bits')) != ('zynum-finite-interface-subset-v1', 'mkl2026.lp64', 'sparse-csr-csc-mv-mm-trsv-trsm-ordinary-lp64-v1', 'apple-arm64-default-c-v1', 'aarch64', 'macos', 64):
            parser.error('unsupported standard Sparse contract')
        expected = {'mkl_sparse_'+scalar+'_'+routine+'_'+fmt for scalar in 'sdcz' for routine in ['create','export'] for fmt in ['csr','csc']} | {'mkl_sparse_'+scalar+'_'+routine for scalar in 'sdcz' for routine in ['mv','mm','trsv','trsm']} | {'mkl_sparse_destroy'}
        names = selection.get('symbols', [])
        if not isinstance(names,list) or len(names)!=33 or set(names)!=expected:
            parser.error('invalid standard Sparse export list')
        exports += ['_'+name for name in names]
    if args.blas_manifest is not None:
        if args.lapacke_manifest is None:
            parser.error('BLAS requires the unified one-owner LP64 package')
        blas = json.loads(args.blas_manifest.read_text())
        repository = Path(__file__).resolve().parent.parent
        reviewed = json.loads((repository / 'include/zynum/blas/abi_manifest.json').read_text())
        if blas != reviewed or blas.get('schema') != 1 or blas.get('blas_integer_abi') != 'LP64':
            parser.error('unsupported existing LP64 BLAS declaration contract')
        blas_exports = []
        for section, count, prefix in (('fortran', 161, ''), ('cblas', 150, 'cblas_')):
            group = blas.get(section, {})
            rows = group.get('exports', [])
            names = [row.get('name') for row in rows if isinstance(row, dict)]
            source = (repository / 'src/blas/abi' / (section + '.zig')).read_text()
            expected = {name for name in re.findall(r'^pub export fn ([A-Za-z_][A-Za-z0-9_]*)\(', source, re.MULTILINE)
                        if name.startswith(prefix)}
            if (group.get('export_count') != count or len(names) != count or
                    len(set(names)) != count or set(names) != expected):
                parser.error('invalid existing LP64 BLAS export list')
            blas_exports += ['_' + name for name in names]
        # LP64 BLAS supplies the strong implementation of this already shared
        # LAPACK error symbol; the default LAPACKE object defines it weakly.
        if set(blas_exports).intersection(exports) != {'_xerbla_'}:
            parser.error('unexpected BLAS/interface symbol collision')
        exports += [name for name in blas_exports if name != '_xerbla_']
    if args.checked_cholesky_manifest is not None:
        from checked_cholesky_profile import validate
        try:
            cholesky = validate(json.loads(args.checked_cholesky_manifest.read_text()))
        except ValueError as error:
            parser.error(str(error))
        exports += ['_' + name for name in cholesky['symbols']]
    if args.feast_manifest is not None:
        if args.lapacke_manifest is None or args.legacy_manifest is None:
            parser.error('FEAST requires the unified one-owner LP64 package')
        from generate_feast import validate_manifest as validate_feast_manifest
        try:
            feast_symbols = validate_feast_manifest(json.loads(args.feast_manifest.read_text()))
        except (ValueError, TypeError) as error:
            parser.error(str(error))
        exports += ['_' + name for name in feast_symbols]
    if args.contour_manifest is not None:
        if args.operator_manifest is None or args.lapacke_manifest is None or args.legacy_manifest is None:
            parser.error('Contour requires the unified one-owner LP64 and operator-v1 package')
        if operator.get('phase_agreement') is not True:
            parser.error('Contour requires the agreed operator-v1 profile')
        from generate_contour_interface import validate_manifest as validate_contour_manifest
        try:
            contour_symbols = validate_contour_manifest(json.loads(args.contour_manifest.read_text()))
        except (ValueError, TypeError) as error:
            parser.error(str(error))
        exports += ['_' + name for name in contour_symbols]
    if len(exports) != len(set(exports)):
        parser.error('interface symbol collision')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.zynum-darwin-link-', dir=args.output.parent) as directory:
        symbols = Path(directory) / 'exports.txt'
        symbols.write_text('\n'.join(exports) + '\n')
        temporary = Path(directory) / args.output.name
        subprocess.run(['xcrun', 'clang', '-arch', 'arm64', '-dynamiclib',
                        '-mmacosx-version-min=' + args.macos_min,
                        '-Wl,-force_load,' + str(args.archive.resolve()),
                        '-Wl,-exported_symbols_list,' + str(symbols),
                        '-Wl,-install_name,@rpath/' + args.output.name,
                        '-o', str(temporary)], check=True)
        observed = subprocess.check_output(['xcrun', 'nm', '-gU', str(temporary)], text=True)
        actual = sorted(line.split()[-1] for line in observed.splitlines() if line.strip())
        if actual != sorted(exports):
            raise RuntimeError('linked export surface differs from finite interface manifest')
        os.replace(temporary, args.output)

if __name__ == '__main__':
    main()
