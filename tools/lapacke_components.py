# Copyright (C) 2026 Zynum contributors
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Source-owned finite components composed over an existing LAPACKE owner."""
import hashlib
import json
from pathlib import Path
import standard_abi_bindings as standard

ROOT = Path(__file__).resolve().parents[1]
TARGET = 'apple-arm64-default-c-v1'
PROFILES = ('mkl2026.lp64', 'mkl2026.ilp64')
COMPONENTS = ('controls', 'evd', 'gels', 'gelss', 'gvd', 'potri', 'syev', 'sygv')
DECLARATIONS = {
    'LAPACKE_sgelss': (8489, 'cb80aa9069901b706790b74535320e6ffb0f5119e44941b2a1a4b8854feae669'),
    'LAPACKE_sgelss_work': (8495, 'b7a3cb81b65b4d24001e7bd30ef4d5c27fd447332d0b3e5de627c897b07d51cb'),
    'LAPACKE_dgelss': (4990, '591f00fde823512812f6e2b06f91f5233338b87c2b50d3b0b06300e4ab967f9f'),
    'LAPACKE_dgelss_work': (4996, 'f18befaa0671fba2bed9e7397b34f26e43452b2fd1aa5dfe619f5123c22217eb'),
    'LAPACKE_cgelss': (878, '4917360693fe5ace2a16d6c0e0054b19cd6733ce0ffe00b9a4f99a1a31521a47'),
    'LAPACKE_cgelss_work': (885, 'eb3d2088dd007065dd8b4956d155d6ca0f0a6266dc10dc22bb27b117f6dc1cf1'),
    'LAPACKE_zgelss': (11890, 'ed6f6c7c2319b38d5d6d22a120a31a2e26a93cac325041150125e99038e27ca9'),
    'LAPACKE_zgelss_work': (11897, 'a4a7257d4c807ca45c9a2525c80205964793a6b83fa9d3aae1db38f4d0c1e14c'),

    'LAPACKE_cgels': (849, '2afbd2e36dee691ad52cbe514f429dddb436ef36b7fa8271747b821013f22234'),
    'LAPACKE_cgels_work': (855, '0064145b3882a0150cc3144503ae4c2afe343955d6dd16e9f36eda2eff9cfb5e'),
    'LAPACKE_dgels': (4966, '6942880cd283e826b0dfab0de4ea06ece1c6605bbfe4bc6b50a6cd4bd1f1a0f6'),
    'LAPACKE_dgels_work': (4971, '857d84c7cde9d924663728c6532bd4fe0a3b7ea1ce38d9cf92658a93c7c26da0'),
    'LAPACKE_sgels': (8465, 'd4fd31eb9be27c4c4fa5561e8c41736db736e4dd6a9ac33cbeec6cdcb2ecb738'),
    'LAPACKE_sgels_work': (8470, '2d155605fa47b3781c46481526ee7d62fa2f5560ac6be267ff1847bfe080c11d'),
    'LAPACKE_zgels': (11861, 'a38e099eb761459f45785e5b830ba9930f1cd89aa9fe3212d15677e1f68c3ad0'),
    'LAPACKE_zgels_work': (11867, '2e309655cd228e06717d87feba2aeb39221e2f78e07cb2d05e2dc450c7151494'),

    'LAPACKE_set_nancheck': (17274, '67c36c9ac0cbb16206193811c3659e9e62753b17080114e7a51844c2fb9f95ad'),
    'LAPACKE_get_nancheck': (17277, '7bd4826548a9816b53d505bd85a82103a1db6034a06a5d323852d03cca728be4'),
    'LAPACKE_spotri': (9932, '5cffa0047d2265c205270c08f6add20ba22e4b63452252be90ebf77a097ff4b0'),
    'LAPACKE_spotri_work': (9936, 'b6943ebf41a49ec74a6a20a982e9a06cf77d0e745577c88f7484417bd3a69722'),
    'LAPACKE_dpotri': (6448, '9b6907e63a073cdbd2e0fd175733337f1fee231d22b1cceac1ae055822ecaf90'),
    'LAPACKE_dpotri_work': (6452, '96da23eaa257475caf258f4107f72aa6754121ca432980e2619190cd2ec03ab1'),
    'LAPACKE_cpotri': (3021, 'a1e46754310a4e06cede2ab211f786769183e0bc92f991b0c9f4aa4095003878'),
    'LAPACKE_cpotri_work': (3025, '95892772dd7ff84a15338203b2e22871960bf811759ace2bb38468784b32cce2'),
    'LAPACKE_zpotri': (14059, '3c279a9c73c3f856fa791e77dd7b879bc7f8335e98f4e539cd9ad4b9b4f8c481'),
    'LAPACKE_zpotri_work': (14063, 'ecbba6dc7d5e6b1e5ac80da6a8a0cf18a86e56fa8e046380eacaa043058385c8'),
    'LAPACKE_ssygv': (10630, 'a0153ab29bf39b144827cedd63c97ecc93c72a2bb057a139c382ae2ca5b3c744'),
    'LAPACKE_ssygv_work': (10635, 'cf9a5911e7cb455fbdc4a030adb78699fdc169a0e8b3a159122f8f89704823e4'),
    'LAPACKE_dsygv': (7181, 'e749cc9ad199abebcd14a83fcdaf6dffa9ee5143292c109fe98f82dab84f96e1'),
    'LAPACKE_dsygv_work': (7186, '2c54d5d3fdf44d4567ae7e23670f73280a5bbc8ca8b5506a4e3a32acccd0d00b'),
    'LAPACKE_ssyev': (10568, 'de567f3eb96051dd82b4d7ba39671c4a72ab8858a79980bb933846b968e99389'),
    'LAPACKE_ssyev_work': (10572, '219622ffb4fd4b899023614694f8bc2bfce81782638b42bc74272db24f3b6a51'),
    'LAPACKE_dsyev': (7119, 'c9f96070a7cfb4f0808c7090ac57a665970ffd3eebebd73fbd3ef6280dad9e0e'),
    'LAPACKE_dsyev_work': (7123, '1adc7f857318602f309cd3d2d84858c64fd17976ad0d4f6565ce92da4212040a'),
    'LAPACKE_ssyevd': (10577, '9b74ea10ea2ab1037ec43173118e5f3024091673445ebe17623a94442cdc4d3d'),
    'LAPACKE_ssyevd_work': (10581, 'b89fdd2dca86711cecbdd9fad92b739d3824305d6238b5377ca740a7bac14dc4'),
    'LAPACKE_dsyevd': (7128, '255693f78673b07d42e284abceb9365d66709aec7f73157d300ea710a5914fcb'),
    'LAPACKE_dsyevd_work': (7132, 'c7795ace6412a8b39e5ec17266c826fd71c7a484408814246d89098b2a2d0f1c'),
    'LAPACKE_cheevd': (1907, '078acd38430d5e417eac7a943f67e11886f6c55a3b64778eeca3ca2b6406590b'),
    'LAPACKE_cheevd_work': (1911, '61cb8b7a266d8ac0a895d194be5371e5b24a7983d0a49ec2e9260a5aad737ff7'),
    'LAPACKE_zheevd': (12932, '6b006a89ae5c27c5c1226d45a45717f948accd4004eb64aa87db2b7e57dcc031'),
    'LAPACKE_zheevd_work': (12936, 'e87b2a289b729339d4d59a351265f8eb707c1b6f69e3d12353fa35f0d1c48d10'),
    'LAPACKE_ssygvd': (10641, '52a061a9b52df7f92bfbe2b9534044e6c169fe63a3f04655dd27e1f9ec2cf07f'),
    'LAPACKE_ssygvd_work': (10646, '5727ee8c205fa207dd149e51dd62c5c80e527107377d907a43c0c235d95f1535'),
    'LAPACKE_dsygvd': (7192, '9c7e664639d25f9c54a3f1dc5af0041decd10a37a97a490513547a741f29e5f7'),
    'LAPACKE_dsygvd_work': (7197, 'cd8183ac4263593cdecc4316a45a6a24312cc6faabcf04c085f7463e783ce86d'),
    'LAPACKE_chegvd': (1984, '20907ff9e2f2a5ea74c9d882bb4418c208b9ae2275b7d733e49c3e33ed116c8b'),
    'LAPACKE_chegvd_work': (1990, '1d4676d43c0d603853a028fb15120a365f202ea78ea6fd402a928fde79e91b9b'),
    'LAPACKE_zhegvd': (13009, '923be0ba83b6cc2efa60e34f4aca6b5a2fca36a9cd393d4816addd0f1b113e9d'),
    'LAPACKE_zhegvd_work': (13015, 'f5a9b2e5e2c8b28321b73768624f222fe7da98793b88ec32829bd596934c29bf'),
}

def require(condition, message):
    if not condition:
        raise ValueError(message)

def digest(value):
    from checked_abi_schema import _binding_digest
    return _binding_digest(value)

def input_hashes(component='potri'):
    names = ('tools/lapacke_components.py', 'tools/generate_lapacke_component.py',
             'tools/interface_templates/lapacke_component.zig.in', 'tools/standard_abi_bindings.py',
             'src/lapacke_composed.zig')
    if component == 'controls':
        names = ('tools/lapacke_components.py', 'tools/generate_lapacke_component.py',
                 'tools/interface_templates/lapacke_controls.zig.in',
                 'tools/standard_abi_bindings.py', 'src/lapacke_controls_composed.zig')
    if component == 'syev':
        names += ('tools/interface_templates/symmetric_component_lp64.zig.in',
                  'src/lapacke_symmetric_composed.zig')
    if component == 'sygv':
        names += ('tools/interface_templates/symmetric_generalized_component_lp64.zig.in',
                  'src/lapacke_symmetric_generalized_composed.zig')
    if component in ('evd', 'gvd', 'gels', 'gelss'):
        names = ('tools/lapacke_components.py', 'tools/generate_lapacke_component.py',
                 'tools/interface_templates/' + component + '_component_lp64.zig.in',
                 'tools/standard_abi_bindings.py', 'src/lapacke_' + component + '_composed.zig')
    return {n: hashlib.sha256((ROOT / n).read_bytes()).hexdigest() for n in names}

def select_component(profile, component, target):
    require(profile in PROFILES and component in COMPONENTS and target == TARGET,
            'Unreviewed LAPACKE component/profile/target')
    require(component not in ('syev', 'sygv', 'evd', 'gvd', 'gels', 'gelss') or profile == 'mkl2026.lp64',
            component.upper() + ' component currently requires the LP64 declaration contract')
    dependencies = tuple(standard.SOURCE_DEPENDENCIES)
    require(all(hashlib.sha256((ROOT / n).read_bytes()).hexdigest() == h for n, h in dependencies),
            'Reviewed component owner source changed')
    required = ({'src/interop/lapacke_potri.zig', 'src/numerics/cholesky_inverse.zig'}
                if component == 'potri' else
                {'src/interop/lapacke_symmetric.zig', 'src/lapack/hermitian_managed.zig',
                 'src/lapack/core/hermitian_tridiagonal.zig', 'src/lapack/core/tridiagonal_eigen.zig'})
    if component == 'sygv':
        required = {'src/interop/lapacke_symmetric_generalized.zig', 'src/lapack/hermitian_managed.zig',
                    'src/lapack/core/hermitian_eigen.zig', 'src/lapack/core/hermitian_tridiagonal.zig',
                    'src/lapack/core/tridiagonal_eigen.zig'}
    if component == 'controls':
        required = {'src/interop/lapacke_binding.zig', 'src/interop/lapacke_services.zig'}
    if component in ('evd', 'gvd'):
        required = {'src/interop/lapacke_evd.zig', 'src/interop/lapacke_evd_policy.zig',
                    'src/lapack/hermitian_managed.zig', 'src/lapack/core/hermitian_eigen.zig',
                    'src/lapack/core/hermitian_tridiagonal.zig', 'src/lapack/core/tridiagonal_eigen.zig'}
    if component == 'gels':
        required = {'src/interop/lapacke_gels.zig', 'src/lapack/full_rank_least_squares.zig',
                    'src/lapack/lq.zig', 'src/interop/lapacke_binding.zig'}
    if component == 'gelss':
        required = {'src/interop/lapacke_gelss.zig', 'src/lapack/svd_least_squares.zig',
                    'src/lapack/core/complex_bidiagonal.zig', 'src/lapack/core/complex_bidiagonal_svd.zig',
                    'src/lapack/core/bidiagonal_svd.zig', 'src/interop/lapacke_binding.zig'}
    require(required <= {n for n, _ in dependencies}, 'Component owner envelope omits its implementation')
    rows = []
    if component == 'controls':
        # Interface configuration is deliberately not a mathematical operation
        # or a new execution owner. C int stays 32-bit in both index profiles.
        for symbol, declaration, forwarding, parameters in (
            ('LAPACKE_set_nancheck', 'void LAPACKE_set_nancheck( int flag );', 'setNancheck', 1),
            ('LAPACKE_get_nancheck', 'int LAPACKE_get_nancheck( void );', 'getNancheck', 0),
        ):
            line, original_hash = DECLARATIONS[symbol]
            require(hashlib.sha256(declaration.encode()).hexdigest() == original_hash,
                    'Original LAPACKE control declaration changed')
            rows.append({'symbol': symbol, 'kind': 'existing-interface-configuration',
                'declaration': declaration, 'parameter_count': parameters,
                'configuration_forward': forwarding, 'control_integer_type': 'c_int',
                'control_integer_bits': 32, 'original_header_line': line,
                'original_declaration_sha256': original_hash})
    scalars = (('s', 'f32', 'float'), ('d', 'f64', 'double'))
    if component == 'controls':
        scalars = ()
    if component in ('potri', 'evd', 'gvd', 'gels', 'gelss'):
        scalars += (('c', 'complex_f32', 'lapack_complex_float'), ('z', 'complex_f64', 'lapack_complex_double'))
    for tag, scalar, ctype in scalars:
        for level, suffix in (('high', ''), ('work', '_work')):
            routine = ('syevd' if tag in 'sd' else 'heevd') if component == 'evd' else component
            if component == 'gvd':
                routine = 'sygvd' if tag in 'sd' else 'hegvd'
            symbol = 'LAPACKE_' + tag + routine + suffix
            if component == 'potri':
                declaration = f'lapack_int {symbol}( int matrix_layout, char uplo, lapack_int n, {ctype}* a, lapack_int lda );'
            elif component in ('evd', 'gvd'):
                real_type = 'float' if tag in 'sc' else 'double'
                work = ''
                if level == 'work':
                    work = f', {ctype}* work, lapack_int lwork'
                    if tag in 'cz':
                        work += f', {real_type}* rwork, lapack_int lrwork'
                    work += ', lapack_int* iwork, lapack_int liwork'
                problem = 'lapack_int itype, ' if component == 'gvd' else ''
                b_operand = f'{ctype}* b, lapack_int ldb, ' if component == 'gvd' else ''
                declaration = f'lapack_int {symbol}( int matrix_layout, {problem}char jobz, char uplo, lapack_int n, {ctype}* a, lapack_int lda, {b_operand}{real_type}* w{work} );'
            elif component == 'gelss':
                real_type = 'float' if tag in 'sc' else 'double'
                work = f', {ctype}* work, lapack_int lwork' if level == 'work' else ''
                if level == 'work' and tag in 'cz':
                    work += f', {real_type}* rwork'
                declaration = f'lapack_int {symbol}( int matrix_layout, lapack_int m, lapack_int n, lapack_int nrhs, {ctype}* a, lapack_int lda, {ctype}* b, lapack_int ldb, {real_type}* s, {real_type} rcond, lapack_int* rank{work} );'
            elif component == 'gels':
                work = f', {ctype}* work, lapack_int lwork' if level == 'work' else ''
                declaration = f'lapack_int {symbol}( int matrix_layout, char trans, lapack_int m, lapack_int n, lapack_int nrhs, {ctype}* a, lapack_int lda, {ctype}* b, lapack_int ldb{work} );'
            elif component == 'sygv':
                work = f', {ctype}* work, lapack_int lwork' if level == 'work' else ''
                declaration = f'lapack_int {symbol}( int matrix_layout, lapack_int itype, char jobz, char uplo, lapack_int n, {ctype}* a, lapack_int lda, {ctype}* b, lapack_int ldb, {ctype}* w{work} );'
            else:
                work = f', {ctype}* work, lapack_int lwork' if level == 'work' else ''
                declaration = f'lapack_int {symbol}( int matrix_layout, char jobz, char uplo, lapack_int n, {ctype}* a, lapack_int lda, {ctype}* w{work} );'
            line, original_hash = DECLARATIONS[symbol]
            require(hashlib.sha256(declaration.encode()).hexdigest() == original_hash,
                    'Original LAPACKE component declaration changed')
            rows.append({'symbol': symbol, 'scalar': scalar, 'level': level,
                'operation': ('symmetric.generalized_eigen' if tag in 'sd' else 'hermitian.generalized_eigen') if component == 'gvd' else ('symmetric.eigen' if tag in 'sd' else 'hermitian.eigen') if component == 'evd' else {'potri': 'cholesky.inverse', 'syev': 'symmetric.eigen', 'sygv': 'symmetric.generalized_eigen', 'gels': 'least_squares.full_rank', 'gelss': 'least_squares.svd'}[component],
                'recipe': 'raw_' + component + '_profile_defaults',
                'declaration': declaration, 'parameter_count': ((7 if level == 'high' else 13 if tag in 'cz' else 11) + (3 if component == 'gvd' else 0)) if component in ('evd', 'gvd') else (11 if level == 'high' else 14 if tag in 'cz' else 13) if component == 'gelss' else (9 if level == 'high' else 11) if component == 'gels' else (5 if component == 'potri' else (10 if level == 'high' else 12) if component == 'sygv' else (7 if level == 'high' else 9)),
                'pointee_const': False,
                'original_header_line': line, 'original_declaration_sha256': original_hash})
    return {'schema': 'standard-lapacke-component-plan-v1', 'component': component,
        'profile': profile, 'target': dict(standard.TARGET), 'bindings': rows,
        'symbols': [v['symbol'] for v in rows], 'owner_imports': [dict(standard.OWNER)],
        'required_base_exports': ['LAPACKE_xerbla', 'xerbla_'],
        'source_dependencies': dependencies, 'source_envelope_sha256': standard.SOURCE_ENVELOPE_SHA256,
        'declaration_headers': standard.DECLARATION_HEADERS,
        'generator_inputs': input_hashes(component), 'runtime_abi_status': 'pending',
        'generated_owner_or_numerics': False, 'full_profile_ready': False}

def manifest_for(profile, component, target):
    return {'schema': 'zynum-lapacke-component-v1', 'plan': select_component(profile, component, target),
            'composition_required': True, 'public_header': 'lapacke_' + component + '.h',
            'base_header': 'lapacke_base.h', 'status': 'finite-generation-only-runtime-pending'}

def validate_component(value, profile):
    require(isinstance(value, dict), 'Invalid component manifest')
    plan = value.get('plan', {})
    expected = manifest_for(profile, plan.get('component'), plan.get('target', {}).get('id'))
    require(digest(value) == digest(expected), 'Component manifest differs from its source-owned plan')
    return expected

def compose(base, components):
    from checked_abi_schema import select_binding_set
    require(isinstance(base, dict) and base.get('schema') == 'zynum-finite-interface-subset-v1',
            'Composition requires an existing finite LAPACKE base')
    plan = base.get('plan', {})
    require(plan.get('profile') in PROFILES, 'Unsupported composition profile')
    expected = select_binding_set(plan['profile'], plan.get('set'), plan.get('target', {}).get('id'))
    require(digest(plan) == digest(expected), 'Base manifest differs from its accepted finite plan')
    require(isinstance(components, list) and 0 < len(components) <= len(COMPONENTS), 'Invalid component list')
    checked = [validate_component(v, plan['profile']) for v in components]
    names = [v['plan']['component'] for v in checked]
    require(names == sorted(set(names)), 'Duplicate or unordered components')
    symbols = list(plan['symbols']) + [n for v in checked for n in v['plan']['symbols']]
    require(len(symbols) == len(set(symbols)), 'Overlapping base/component exports')
    return {'schema': 'zynum-lapacke-composition-v1', 'base': base, 'components': checked,
            'plan': {**plan, 'set': plan['set'] + '+' + '+'.join(names), 'symbols': symbols,
                     'composition_base_set': plan['set']},
            'one_existing_owner': True, 'full_profile_ready': False,
            'status': 'finite-generation-only-runtime-pending'}

def unwrap(value):
    if value.get('schema') != 'zynum-lapacke-composition-v1':
        return value, []
    expected = compose(value.get('base'), value.get('components'))
    require(digest(value) == digest(expected), 'Forged LAPACKE composition manifest')
    return value['base'], [n for c in value['components'] for n in c['plan']['symbols']]
