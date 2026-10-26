# Copyright (C) 2026 Zynum contributors
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Select reviewed profile inputs; retain the existing checked-schema interface."""
import copy
import hashlib
import json
from pathlib import Path

from numerical_operation_schema import OPERATIONS
from checked_abi_profile import CHECKED_PROFILE

DEFAULT_PROFILE = 'checked-v1'
LEGACY_FIELDS = ('OBJECT_SUPPORT_SYMBOLS', 'ABI_REVISION', 'POINTER_BITS', 'PRIMITIVE_LAYOUT',
                 'PREFIX', 'STATUS', 'FLAGS', 'STRUCTS', 'FUNCTIONS')
# Ordered legacy schema projection pins the current eight-symbol wire/call
# contract. It is not a signature over the full numerical dependency closure.
CHECKED_V1_CONTRACT_SHA256 = 'cbed92aa6ec5c85f9f352a1b4e2c02e319accaa14d4699e3223ea3574a535829'
PROFILES = {
    'checked-v1': CHECKED_PROFILE,
    'mkl2026.lp64': {'id': 'mkl2026.lp64', 'generation_ready': False,
                    'unresolved_generation_inputs': ('No implemented/reviewed standard adapter or API/ABI evidence; existing profile prototype has zero callable bindings.',)},
    'mkl2026.ilp64': {'id': 'mkl2026.ilp64', 'generation_ready': False,
                     'unresolved_generation_inputs': ('No implemented/reviewed standard adapter or API/ABI evidence; existing profile prototype has zero callable bindings.',)},
}


class ProfileError(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise ProfileError(message)


def exact_keys(value, expected, context):
    require(isinstance(value, dict) and set(value) == set(expected), context + ': incomplete or unknown fields')


def select_profile(profile_id=DEFAULT_PROFILE):
    """Validate actual selected inputs before any formatter or artifact write."""
    require(isinstance(profile_id, str) and profile_id in PROFILES, 'Unknown interface profile: ' + repr(profile_id))
    profile = copy.deepcopy(PROFILES[profile_id])
    require(profile.get('id') == profile_id, 'Profile selection/identity mismatch')
    require(profile.get('generation_ready') is True and profile.get('unresolved_generation_inputs') == (),
            'Profile is not ready for generation: ' + profile_id)
    exact_keys(profile, ('id', 'generation_ready', 'readiness_scope', 'unresolved_generation_inputs', 'renderer',
                         *LEGACY_FIELDS[:-1], 'adapter_source', 'adapters', 'bindings'), 'profile')
    require(profile_id == DEFAULT_PROFILE and profile['renderer'] == 'checked-v1', 'No reviewed renderer for selected profile')
    exact_keys(profile['adapter_source'], ('path', 'sha256'), 'adapter source')
    require(profile['adapter_source'] == {
        'path': 'src/checked/implementation.zig',
        'sha256': 'bf05e8adfac3a678ab2db7a2b5fa8588a8ebba105ce358d7e8c27e6d238090c5',
    }, 'Unreviewed checked adapter source binding')
    source = Path(__file__).resolve().parents[1] / profile['adapter_source']['path']
    require(source.is_file() and hashlib.sha256(source.read_bytes()).hexdigest() == profile['adapter_source']['sha256'],
            'Checked adapter entry source drift; this pin is not full numerical/ABI evidence')
    require(set(profile['adapters']) == {'factor', 'solve', 'spmv'}, 'Unreviewed checked adapter set')
    require(isinstance(profile['bindings'], list) and len(profile['bindings']) == 8, 'Checked-v1 requires eight exact bindings')
    functions = []
    identities, symbols = set(), set()
    for binding in profile['bindings']:
        exact_keys(binding, ('id', 'profile', 'operation', 'adapter', 'scalar', 'index', 'symbol'), 'binding')
        require(isinstance(binding['id'], str) and binding['id'] and binding['id'] not in identities, 'Missing/duplicate binding identity')
        identities.add(binding['id'])
        # Never infer profile membership from opaque IDs or symbol prefixes.
        require(binding['profile'] == profile_id, 'Binding belongs to a different selected profile')
        require(binding['operation'] in OPERATIONS and binding['adapter'] in profile['adapters'], 'Unknown operation or adapter')
        operation = OPERATIONS[binding['operation']]
        adapter = profile['adapters'][binding['adapter']]
        exact_keys(adapter, ('operation', 'symbol_stem', 'scalar_types', 'index_types', 'type_arguments', 'arguments'), 'adapter')
        require(binding['operation'] == adapter['operation'], 'Operation/adapter binding mismatch')
        require(binding['scalar'] in operation['scalar_types'] and binding['scalar'] in adapter['scalar_types'], 'Unsupported scalar binding')
        index = binding['index']
        require(index in operation['index_types'] and index in adapter['index_types'] if adapter['index_types'] else index is None,
                'Unsupported index binding')
        require(tuple(role for _, _, role in adapter['arguments']) == operation['operand_roles'], 'Operation/adapter operand roles mismatch')
        expected_types = ('scalar', 'index') if adapter['index_types'] else ('scalar',)
        require(adapter['type_arguments'] == expected_types, 'Adapter type argument order mismatch')
        suffix = binding['scalar'] + ('_' + index if index else '')
        require(binding['symbol'] == adapter['symbol_stem'] + '_' + suffix, 'Symbol/type binding mismatch')
        require(binding['symbol'] not in symbols, 'Duplicate export symbol')
        symbols.add(binding['symbol'])
        arguments = [(name, binding['scalar'] if typ == 'scalar' else typ) for name, typ, _ in adapter['arguments']]
        types = [binding[key] for key in adapter['type_arguments']]
        body = binding['adapter'] + '(' + ', '.join([*types, *(name for name, _ in arguments)]) + ')'
        functions.append((binding['symbol'], arguments, body))
    selected = {key: copy.deepcopy(profile[key]) for key in LEGACY_FIELDS[:-1]}
    selected['FUNCTIONS'] = functions
    digest = hashlib.sha256(json.dumps(selected, separators=(',', ':')).encode()).hexdigest()
    require(digest == CHECKED_V1_CONTRACT_SHA256, 'Incomplete or altered checked-v1 wire/call contract')
    return selected


# Existing consumers continue to import the same nine constants and tuple/list
# structures. The generator selects again at each call, using actual profile data.
OBJECT_SUPPORT_SYMBOLS, ABI_REVISION, POINTER_BITS, PRIMITIVE_LAYOUT, PREFIX, STATUS, FLAGS, STRUCTS, FUNCTIONS = select_profile().values()


# Standard finite-set selection is additive. It does not change select_profile,
# its nine constants, or the checked-v1 renderer and six output files.
def _binding_plain(value):
    from collections.abc import Mapping
    if isinstance(value, Mapping):
        return {key: _binding_plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_binding_plain(item) for item in value]
    return value


def _binding_digest(value):
    return hashlib.sha256(json.dumps(_binding_plain(value), sort_keys=True,
                                     separators=(',', ':')).encode()).hexdigest()


def _binding_freeze(value):
    from collections.abc import Mapping
    from types import MappingProxyType
    if isinstance(value, Mapping):
        return MappingProxyType({key: _binding_freeze(item) for key, item in value.items()})
    if isinstance(value, (tuple, list)):
        return tuple(_binding_freeze(item) for item in value)
    return value


def _binding_source_root():
    return Path(__file__).resolve().parents[1]


def _binding_routine_contract(operation):
    """Independent closed validation shapes, not a data-table passthrough."""
    layout = (('matrix_layout', 'c_int', 'layout', 0, False),)
    triangle = layout + (('uplo', 'c_char', 'triangle', 0, False),
                         ('n', 'lapack_int', 'order', 0, False))
    contracts = {
        'svd.decompose': ('gesvd', (('matrix_layout', 'c_int', 'layout', 0, False), ('jobu', 'c_char', 'left_job', 0, False), ('jobvt', 'c_char', 'right_job', 0, False), ('m', 'lapack_int', 'row_count', 0, False), ('n', 'lapack_int', 'column_count', 0, False), ('a', 'scalar', 'matrix', 1, False), ('lda', 'lapack_int', 'matrix_stride', 0, False), ('s', 'real', 'singular_values', 1, False), ('u', 'scalar', 'left_vectors', 1, False), ('ldu', 'lapack_int', 'left_stride', 0, False), ('vt', 'scalar', 'right_vectors', 1, False), ('ldvt', 'lapack_int', 'right_stride', 0, False)), (('matrix', ('a', 'm', 'n', 'lda', 'matrix_layout')), ('singular_values', ('s', 'm', 'n')), ('left_vectors', ('u', 'm', 'n', 'ldu', 'jobu', 'matrix_layout')), ('right_vectors', ('vt', 'm', 'n', 'ldvt', 'jobvt', 'matrix_layout')), ('jobs', ('jobu', 'jobvt'))), 'raw_gesvd_profile_defaults'),
        **{'qr.generate_real': ('orgqr', (('matrix_layout', 'c_int', 'layout', 0, False), ('m', 'lapack_int', 'row_count', 0, False), ('n', 'lapack_int', 'column_count', 0, False), ('k', 'lapack_int', 'reflector_count', 0, False), ('a', 'scalar', 'matrix', 1, False), ('lda', 'lapack_int', 'matrix_stride', 0, False), ('tau', 'scalar', 'reflectors', 1, True)), (('matrix', ('a', 'm', 'n', 'lda', 'matrix_layout')), ('reflectors', ('tau', 'k'))), 'raw_qr_generate_profile_defaults'), 'qr.generate_complex': ('ungqr', (('matrix_layout', 'c_int', 'layout', 0, False), ('m', 'lapack_int', 'row_count', 0, False), ('n', 'lapack_int', 'column_count', 0, False), ('k', 'lapack_int', 'reflector_count', 0, False), ('a', 'scalar', 'matrix', 1, False), ('lda', 'lapack_int', 'matrix_stride', 0, False), ('tau', 'scalar', 'reflectors', 1, True)), (('matrix', ('a', 'm', 'n', 'lda', 'matrix_layout')), ('reflectors', ('tau', 'k'))), 'raw_qr_generate_profile_defaults')},
        **{'qr.factor': ('geqrf', (('matrix_layout', 'c_int', 'layout', 0, False), ('m', 'lapack_int', 'row_count', 0, False), ('n', 'lapack_int', 'column_count', 0, False), ('a', 'scalar', 'matrix', 1, False), ('lda', 'lapack_int', 'matrix_stride', 0, False), ('tau', 'scalar', 'reflectors', 1, False)), (('matrix', ('a', 'm', 'n', 'lda', 'matrix_layout')), ('reflectors', ('tau', 'm', 'n'))), 'raw_qr_factor_profile_defaults'), 'qr.apply_real': ('ormqr', (('matrix_layout', 'c_int', 'layout', 0, False), ('side', 'c_char', 'side', 0, False), ('trans', 'c_char', 'operation', 0, False), ('m', 'lapack_int', 'row_count', 0, False), ('n', 'lapack_int', 'column_count', 0, False), ('k', 'lapack_int', 'reflector_count', 0, False), ('a', 'scalar', 'factor', 1, True), ('lda', 'lapack_int', 'factor_stride', 0, False), ('tau', 'scalar', 'reflectors', 1, True), ('c', 'scalar', 'target', 1, False), ('ldc', 'lapack_int', 'target_stride', 0, False)), (('factor', ('a', 'm', 'n', 'k', 'lda', 'side', 'matrix_layout')), ('reflectors', ('tau', 'k')), ('target', ('c', 'm', 'n', 'ldc', 'matrix_layout')), ('side', ('side',)), ('operation', ('trans',))), 'raw_qr_apply_profile_defaults'), 'qr.apply_complex': ('unmqr', (('matrix_layout', 'c_int', 'layout', 0, False), ('side', 'c_char', 'side', 0, False), ('trans', 'c_char', 'operation', 0, False), ('m', 'lapack_int', 'row_count', 0, False), ('n', 'lapack_int', 'column_count', 0, False), ('k', 'lapack_int', 'reflector_count', 0, False), ('a', 'scalar', 'factor', 1, True), ('lda', 'lapack_int', 'factor_stride', 0, False), ('tau', 'scalar', 'reflectors', 1, True), ('c', 'scalar', 'target', 1, False), ('ldc', 'lapack_int', 'target_stride', 0, False)), (('factor', ('a', 'm', 'n', 'k', 'lda', 'side', 'matrix_layout')), ('reflectors', ('tau', 'k')), ('target', ('c', 'm', 'n', 'ldc', 'matrix_layout')), ('side', ('side',)), ('operation', ('trans',))), 'raw_qr_apply_profile_defaults')},
        'hermitian.eigen': ('heev', layout + (('jobz', 'c_char', 'job', 0, False), ('uplo', 'c_char', 'triangle', 0, False), ('n', 'lapack_int', 'order', 0, False), ('a', 'scalar', 'matrix', 1, False), ('lda', 'lapack_int', 'matrix_stride', 0, False), ('w', 'real', 'eigenvalues', 1, False)), (('matrix', ('a', 'n', 'lda', 'matrix_layout')), ('eigenvalues', ('w', 'n')), ('job', ('jobz',)), ('triangle', ('uplo',))), 'raw_hermitian_standard_profile_defaults'),
        'hermitian.generalized_eigen': ('hegv', layout + (('itype', 'lapack_int', 'problem', 0, False), ('jobz', 'c_char', 'job', 0, False), ('uplo', 'c_char', 'triangle', 0, False), ('n', 'lapack_int', 'order', 0, False), ('a', 'scalar', 'matrix', 1, False), ('lda', 'lapack_int', 'matrix_stride', 0, False), ('b', 'scalar', 'metric', 1, False), ('ldb', 'lapack_int', 'metric_stride', 0, False), ('w', 'real', 'eigenvalues', 1, False)), (('matrix', ('a', 'n', 'lda', 'matrix_layout')), ('metric', ('b', 'n', 'ldb', 'matrix_layout')), ('eigenvalues', ('w', 'n')), ('job', ('jobz',)), ('triangle', ('uplo',)), ('problem', ('itype',))), 'raw_hermitian_generalized_profile_defaults'),
        'least_squares.rank_revealing': ('gelsd', (('matrix_layout', 'c_int', 'layout', 0, False), ('m', 'lapack_int', 'row_count', 0, False), ('n', 'lapack_int', 'column_count', 0, False), ('nrhs', 'lapack_int', 'rhs_count', 0, False), ('a', 'scalar', 'matrix', 1, False), ('lda', 'lapack_int', 'matrix_stride', 0, False), ('b', 'scalar', 'rhs', 1, False), ('ldb', 'lapack_int', 'rhs_stride', 0, False), ('s', 'scalar', 'singular_values', 1, False), ('rcond', 'scalar', 'rank_cutoff', 0, False), ('rank', 'lapack_int', 'rank', 1, False)), (('matrix', ('a', 'm', 'n', 'lda', 'matrix_layout')), ('rhs', ('b', 'm', 'n', 'nrhs', 'ldb', 'matrix_layout')), ('singular_values', ('s', 'm', 'n')), ('rank', ('rank', 'rcond'))), 'raw_gelsd_profile_defaults'),
        'least_squares.rank_revealing_complex': ('gelsd', layout + (('m', 'lapack_int', 'row_count', 0, False), ('n', 'lapack_int', 'column_count', 0, False), ('nrhs', 'lapack_int', 'rhs_count', 0, False), ('a', 'scalar', 'matrix', 1, False), ('lda', 'lapack_int', 'matrix_stride', 0, False), ('b', 'scalar', 'rhs', 1, False), ('ldb', 'lapack_int', 'rhs_stride', 0, False), ('s', 'real', 'singular_values', 1, False), ('rcond', 'real', 'rank_cutoff', 0, False), ('rank', 'lapack_int', 'rank', 1, False)), (('matrix', ('a', 'm', 'n', 'lda', 'matrix_layout')), ('rhs', ('b', 'm', 'n', 'nrhs', 'ldb', 'matrix_layout')), ('singular_values', ('s', 'm', 'n')), ('rank', ('rank', 'rcond'))), 'raw_complex_gelsd_profile_defaults'),
        'tridiagonal.eigen': ('stev', layout + (('jobz', 'c_char', 'job', 0, False), ('n', 'lapack_int', 'order', 0, False), ('d', 'scalar', 'diagonal', 1, False), ('e', 'scalar', 'offdiagonal', 1, False), ('z', 'scalar', 'vectors', 1, False), ('ldz', 'lapack_int', 'vector_stride', 0, False)), (('diagonal', ('d', 'n')), ('offdiagonal', ('e', 'n')), ('vectors', ('z', 'n', 'ldz', 'matrix_layout')), ('job', ('jobz',))), 'raw_stev_profile_defaults'),
        'cholesky.factor': ('potrf', triangle + (
            ('a', 'scalar', 'matrix', 1, False), ('lda', 'lapack_int', 'matrix_stride', 0, False)),
            (('matrix', ('a', 'n', 'lda', 'matrix_layout')), ('triangle', ('uplo',))), 'raw_factor_profile_defaults'),
        'lu.inverse': ('getri', (('matrix_layout', 'c_int', 'layout', 0, False), ('n', 'lapack_int', 'order', 0, False), ('a', 'scalar', 'matrix', 1, False), ('lda', 'lapack_int', 'matrix_stride', 0, False), ('ipiv', 'lapack_int', 'pivots', 1, True)), (('matrix', ('a', 'n', 'lda', 'matrix_layout')), ('pivots', ('ipiv', 'n'))), 'raw_getri_profile_defaults'),
        'triangular.inverse': ('trtri', (('matrix_layout', 'c_int', 'layout', 0, False),
                ('uplo', 'c_char', 'triangle', 0, False),
                ('diag', 'c_char', 'diagonal_kind', 0, False),
                ('n', 'lapack_int', 'order', 0, False),
                ('a', 'scalar', 'matrix', 1, False),
                ('lda', 'lapack_int', 'matrix_stride', 0, False)), (('matrix', ('a', 'n', 'lda', 'matrix_layout')),
         ('triangle', ('uplo',)), ('diagonal_kind', ('diag',))), 'raw_trtri_profile_defaults'),
        'triangular.solve': ('trtrs', (('matrix_layout', 'c_int', 'layout', 0, False),
                ('uplo', 'c_char', 'triangle', 0, False),
                ('trans', 'c_char', 'operation', 0, False),
                ('diag', 'c_char', 'diagonal_kind', 0, False),
                ('n', 'lapack_int', 'order', 0, False),
                ('nrhs', 'lapack_int', 'rhs_count', 0, False),
                ('a', 'scalar', 'factor', 1, True),
                ('lda', 'lapack_int', 'factor_stride', 0, False),
                ('b', 'scalar', 'rhs', 1, False),
                ('ldb', 'lapack_int', 'rhs_stride', 0, False)), (('factor', ('a', 'n', 'lda', 'matrix_layout')),
         ('rhs', ('b', 'n', 'nrhs', 'ldb', 'matrix_layout')),
         ('triangle', ('uplo',)), ('operation', ('trans',)),
         ('diagonal_kind', ('diag',))), 'raw_trtrs_profile_defaults'),
        'cholesky.driver': ('posv', triangle + (
            ('nrhs', 'lapack_int', 'rhs_count', 0, False), ('a', 'scalar', 'matrix', 1, False),
            ('lda', 'lapack_int', 'matrix_stride', 0, False), ('b', 'scalar', 'rhs', 1, False),
            ('ldb', 'lapack_int', 'rhs_stride', 0, False)),
            (('matrix', ('a', 'n', 'lda', 'matrix_layout')), ('rhs', ('b', 'n', 'nrhs', 'ldb', 'matrix_layout')),
             ('triangle', ('uplo',))), 'raw_posv_profile_defaults'),
        'cholesky.solve': ('potrs', triangle + (
            ('nrhs', 'lapack_int', 'rhs_count', 0, False), ('a', 'scalar', 'factor', 1, True),
            ('lda', 'lapack_int', 'factor_stride', 0, False), ('b', 'scalar', 'rhs', 1, False),
            ('ldb', 'lapack_int', 'rhs_stride', 0, False)),
            (('factor', ('a', 'n', 'lda', 'matrix_layout')), ('rhs', ('b', 'n', 'nrhs', 'ldb', 'matrix_layout')),
             ('triangle', ('uplo',))), 'raw_solve_profile_defaults'),
        'lu.driver': ('gesv', layout + (('n', 'lapack_int', 'order', 0, False), ('nrhs', 'lapack_int', 'rhs_count', 0, False), ('a', 'scalar', 'matrix', 1, False), ('lda', 'lapack_int', 'matrix_stride', 0, False), ('ipiv', 'lapack_int', 'pivots', 1, False), ('b', 'scalar', 'rhs', 1, False), ('ldb', 'lapack_int', 'rhs_stride', 0, False)), (('matrix', ('a', 'n', 'lda', 'matrix_layout')), ('pivots', ('ipiv', 'n')), ('rhs', ('b', 'n', 'nrhs', 'ldb', 'matrix_layout'))), 'raw_gesv_profile_defaults'),
        'lu.factor': ('getrf', layout + (
            ('m', 'lapack_int', 'row_count', 0, False), ('n', 'lapack_int', 'column_count', 0, False),
            ('a', 'scalar', 'matrix', 1, False), ('lda', 'lapack_int', 'matrix_stride', 0, False),
            ('ipiv', 'lapack_int', 'pivots', 1, False)),
            (('matrix', ('a', 'm', 'n', 'lda', 'matrix_layout')), ('pivots', ('ipiv', 'm', 'n'))), 'raw_getrf_profile_defaults'),
        'lu.solve': ('getrs', layout + (
            ('trans', 'c_char', 'operation', 0, False), ('n', 'lapack_int', 'order', 0, False),
            ('nrhs', 'lapack_int', 'rhs_count', 0, False), ('a', 'scalar', 'factor', 1, True),
            ('lda', 'lapack_int', 'factor_stride', 0, False), ('ipiv', 'lapack_int', 'pivots', 1, True),
            ('b', 'scalar', 'rhs', 1, False), ('ldb', 'lapack_int', 'rhs_stride', 0, False)),
            (('factor', ('a', 'n', 'lda', 'matrix_layout')), ('pivots', ('ipiv', 'n')),
             ('rhs', ('b', 'n', 'nrhs', 'ldb', 'matrix_layout')), ('operation', ('trans',))), 'raw_getrs_profile_defaults'),
    }
    require(operation in contracts, 'Unreviewed operation')
    return contracts[operation]


def _binding_candidate(profile_id, set_id, target_contract):
    """Validate a description; this private intermediate grants no permission."""
    if isinstance(set_id, str) and set_id.startswith('sparse-'):
        import standard_sparse_bindings as sparse
        return sparse.candidate(profile_id, set_id, target_contract)
    import standard_abi_bindings as standard
    require(profile_id in ('mkl2026.lp64', 'mkl2026.ilp64'), 'Unknown standard profile')
    require(PROFILES[profile_id]['id'] == profile_id and
            PROFILES[profile_id]['generation_ready'] is False,
            'Finite selection must not enable a full standard profile')
    set_shapes = {
        'potrf-potrs-ordinary-lp64-v1': ('mkl2026.lp64', False),
        'potrf-potrs-ordinary-ilp64-v1': ('mkl2026.ilp64', False),
        'potrf-potrs-getrf-getrs-ordinary-lp64-v1': ('mkl2026.lp64', True),
        'potrf-potrs-getrf-getrs-ordinary-ilp64-v1': ('mkl2026.ilp64', True),
    }
    set_shapes.update({'potrf-potrs-getrf-getrs-gesv-ordinary-' + width + '-v1': ('mkl2026.' + width, True) for width in ('lp64', 'ilp64')})
    set_shapes.update({'potrf-potrs-getrf-getrs-gesv-stev-ordinary-' + width + '-v1': ('mkl2026.' + width, True) for width in ('lp64', 'ilp64')})
    set_shapes.update({'potrf-potrs-getrf-getrs-gesv-stev-gelsd-ordinary-' + width + '-v1': ('mkl2026.' + width, True) for width in ('lp64', 'ilp64')})
    posv_bases = (
        'potrf-potrs-ordinary-lp64-v1', 'potrf-potrs-ordinary-ilp64-v1',
        'potrf-potrs-getrf-getrs-gesv-stev-gelsd-ordinary-ilp64-v1',
        *('potrf-potrs-getrf-getrs-gesv-stev-gelsd-heev-hegv'
          + ('-geqrf-ormqr-unmqr-orgqr-ungqr' if qr else '')
          + ('-complex-gelsd' if complex_gelsd else '')
          + ('-gesvd' if svd else '') + '-ordinary-lp64-v1'
          for qr in (False, True) for complex_gelsd in (False, True) for svd in (False, True)),
    )
    posv_sets = {base.replace('-ordinary-', '-posv-ordinary-'): base for base in posv_bases}
    standalone_getri_sets = {'getri-ordinary-' + width + '-v1': 'mkl2026.' + width for width in ('lp64', 'ilp64')}
    standalone_getri = set_id in standalone_getri_sets
    with_getri = standalone_getri or (isinstance(set_id, str) and '-getri-ordinary-' in set_id)
    before_getri_set_id = set_id.replace('-getri-ordinary-', '-ordinary-') if with_getri and not standalone_getri else set_id
    standalone_trtri_sets = {'trtri-ordinary-' + width + '-v1': 'mkl2026.' + width for width in ('lp64', 'ilp64')}
    standalone_trtri = before_getri_set_id in standalone_trtri_sets
    with_trtri = standalone_trtri or (isinstance(before_getri_set_id, str) and '-trtri-ordinary-' in before_getri_set_id)
    before_trtri_set_id = before_getri_set_id.replace('-trtri-ordinary-', '-ordinary-') if with_trtri and not standalone_trtri else before_getri_set_id
    trtrs_bases = (*posv_bases, *posv_sets)
    trtrs_sets = {base.replace('-ordinary-', '-trtrs-ordinary-'): base for base in trtrs_bases}
    standalone_trtrs_sets = {'trtrs-ordinary-' + width + '-v1': 'mkl2026.' + width for width in ('lp64', 'ilp64')}
    standalone_trtrs = before_trtri_set_id in standalone_trtrs_sets
    with_trtrs = standalone_trtrs or before_trtri_set_id in trtrs_sets
    before_trtrs_set_id = trtrs_sets.get(before_trtri_set_id, before_trtri_set_id)
    with_posv = before_trtrs_set_id in posv_sets
    ordinary_set_id = posv_sets.get(before_trtrs_set_id, before_trtrs_set_id)
    gesvd_sets = {'gesvd-ordinary-lp64-v1': 'gesvd-ordinary-lp64-v1', 'potrf-potrs-getrf-getrs-gesv-stev-gelsd-heev-hegv-gesvd-ordinary-lp64-v1': 'potrf-potrs-getrf-getrs-gesv-stev-gelsd-heev-hegv-ordinary-lp64-v1', 'potrf-potrs-getrf-getrs-gesv-stev-gelsd-heev-hegv-complex-gelsd-gesvd-ordinary-lp64-v1': 'potrf-potrs-getrf-getrs-gesv-stev-gelsd-heev-hegv-complex-gelsd-ordinary-lp64-v1', 'potrf-potrs-getrf-getrs-gesv-stev-gelsd-heev-hegv-geqrf-ormqr-unmqr-orgqr-ungqr-gesvd-ordinary-lp64-v1': 'potrf-potrs-getrf-getrs-gesv-stev-gelsd-heev-hegv-geqrf-ormqr-unmqr-orgqr-ungqr-ordinary-lp64-v1', 'potrf-potrs-getrf-getrs-gesv-stev-gelsd-heev-hegv-geqrf-ormqr-unmqr-orgqr-ungqr-complex-gelsd-gesvd-ordinary-lp64-v1': 'potrf-potrs-getrf-getrs-gesv-stev-gelsd-heev-hegv-geqrf-ormqr-unmqr-orgqr-ungqr-complex-gelsd-ordinary-lp64-v1'}
    with_gesvd = ordinary_set_id in gesvd_sets
    standalone_gesvd = ordinary_set_id == 'gesvd-ordinary-lp64-v1'
    selected_base_set_id = gesvd_sets.get(ordinary_set_id, ordinary_set_id)
    complex_sets = (
        'complex-gelsd-ordinary-lp64-v1',
        'potrf-potrs-getrf-getrs-gesv-stev-gelsd-heev-hegv-complex-gelsd-ordinary-lp64-v1',
        'potrf-potrs-getrf-getrs-gesv-stev-gelsd-heev-hegv-geqrf-ormqr-unmqr-orgqr-ungqr-complex-gelsd-ordinary-lp64-v1',
    )
    with_complex_gelsd = selected_base_set_id in complex_sets
    standalone_complex_gelsd = selected_base_set_id == complex_sets[0]
    base_set_id = selected_base_set_id.replace('-complex-gelsd-', '-') if with_complex_gelsd and not standalone_complex_gelsd else selected_base_set_id
    only_q_generate = base_set_id == 'orgqr-ungqr-ordinary-lp64-v1'
    standalone_q_generate = base_set_id in ('orgqr-ungqr-ordinary-lp64-v1','geqrf-ormqr-unmqr-orgqr-ungqr-ordinary-lp64-v1')
    with_q_generate = standalone_q_generate or base_set_id == 'potrf-potrs-getrf-getrs-gesv-stev-gelsd-heev-hegv-geqrf-ormqr-unmqr-orgqr-ungqr-ordinary-lp64-v1'
    set_shapes.update({name:('mkl2026.lp64',not standalone_q_generate) for name in ('orgqr-ungqr-ordinary-lp64-v1','geqrf-ormqr-unmqr-orgqr-ungqr-ordinary-lp64-v1','potrf-potrs-getrf-getrs-gesv-stev-gelsd-heev-hegv-geqrf-ormqr-unmqr-orgqr-ungqr-ordinary-lp64-v1')})
    standalone_qr = standalone_q_generate or base_set_id == 'geqrf-ormqr-unmqr-ordinary-lp64-v1'
    with_qr = with_q_generate or standalone_qr or base_set_id == 'potrf-potrs-getrf-getrs-gesv-stev-gelsd-heev-hegv-geqrf-ormqr-unmqr-ordinary-lp64-v1'
    set_shapes.update({name:('mkl2026.lp64',not standalone_qr) for name in ('geqrf-ormqr-unmqr-ordinary-lp64-v1','potrf-potrs-getrf-getrs-gesv-stev-gelsd-heev-hegv-geqrf-ormqr-unmqr-ordinary-lp64-v1')})
    standalone_hermitian = base_set_id == 'hermitian-eigen-ordinary-lp64-v1'
    with_hermitian = standalone_hermitian or (with_qr and not standalone_qr) or base_set_id == 'potrf-potrs-getrf-getrs-gesv-stev-gelsd-heev-hegv-ordinary-lp64-v1'
    set_shapes.update({name: ('mkl2026.lp64', not standalone_hermitian) for name in ('hermitian-eigen-ordinary-lp64-v1', 'potrf-potrs-getrf-getrs-gesv-stev-gelsd-heev-hegv-ordinary-lp64-v1')})
    with_gelsd = isinstance(set_id, str) and '-gelsd-' in base_set_id and not standalone_complex_gelsd
    with_stev = isinstance(set_id, str) and '-stev-' in set_id
    with_gesv = isinstance(set_id, str) and '-gesv-' in set_id
    set_shapes.update({name: ('mkl2026.lp64', name != complex_sets[0]) for name in complex_sets})
    set_shapes.update({name: ('mkl2026.lp64', name != 'gesvd-ordinary-lp64-v1') for name in gesvd_sets})
    set_shapes.update({name: set_shapes[base] for name, base in posv_sets.items()})
    set_shapes.update({name: set_shapes[base] for name, base in trtrs_sets.items()})
    set_shapes.update({name: (profile, False) for name, profile in standalone_trtrs_sets.items()})
    trtri_sets = {base.replace('-ordinary-', '-trtri-ordinary-'): base for base in tuple(set_shapes)
                  if base not in ('potrf-potrs-ordinary-lp64-v1', 'potrf-potrs-ordinary-ilp64-v1')}
    set_shapes.update({name: set_shapes[base] for name, base in trtri_sets.items()})
    set_shapes.update({name: (profile, False) for name, profile in standalone_trtri_sets.items()})
    getri_sets = {base.replace('-ordinary-', '-getri-ordinary-'): base for base in tuple(set_shapes)
                  if base not in ('potrf-potrs-ordinary-lp64-v1', 'potrf-potrs-ordinary-ilp64-v1')}
    set_shapes.update({name: set_shapes[base] for name, base in getri_sets.items()})
    set_shapes.update({name: (profile, False) for name, profile in standalone_getri_sets.items()})
    require(isinstance(set_id, str) and set_id in set_shapes and set_id in standard.SETS, 'Unknown binding set')
    expected_profile, with_lu = set_shapes[set_id]
    require(profile_id == expected_profile, 'Set belongs to a different profile')
    groups = (('cholesky.factor', 'cholesky.solve'),) + ((('lu.factor', 'lu.solve'),) if with_lu else ())
    if with_gesv:
        groups += (('lu.driver',),)
    if with_stev:
        groups += (('tridiagonal.eigen',),)
    if with_gelsd:
        groups += (('least_squares.rank_revealing',),)
    if standalone_hermitian or standalone_qr:
        groups = ()
    if with_hermitian:
        groups += (('hermitian.eigen', 'hermitian.generalized_eigen'),)
    if with_qr and not only_q_generate: groups += (('qr.factor','qr.apply_real','qr.apply_complex'),)
    if with_q_generate: groups += (('qr.generate_real','qr.generate_complex'),)
    if standalone_complex_gelsd: groups = ()
    if with_complex_gelsd: groups += (('least_squares.rank_revealing_complex',),)
    if standalone_gesvd: groups = ()
    if with_gesvd: groups += (('svd.decompose',),)
    if with_posv: groups += (('cholesky.driver',),)
    if standalone_trtrs: groups = ()
    if with_trtrs: groups += (('triangular.solve',),)
    if standalone_trtri: groups = ()
    if with_trtri: groups += (('triangular.inverse',),)
    if standalone_getri: groups = ()
    if with_getri: groups += (('lu.inverse',),)
    operations = tuple(operation for group in groups for operation in group)
    require(isinstance(target_contract, str) and target_contract == 'apple-arm64-default-c-v1',
            'Unreviewed target contract')
    require(_binding_digest(standard.TARGET) ==
            'c563c7620caac27cb99e0d9a4462fc5014309754f55b0b9a0ac13c2932e8803c',
            'Target representation or visibility drift')
    require(_binding_digest(standard.OWNER) ==
            '5cc1afffbdc9b0b360581582e39b72e6549054a074742199de9bfd9d5349fd51',
            'Owner recipe drift; require original configured binding')
    selected = dict(standard.SETS[set_id])
    exact_keys(selected, ('profile', 'target', 'bindings'), 'binding set')
    require(selected['profile'] == profile_id, 'Set belongs to a different profile')
    require(selected['target'] == target_contract, 'Set target mismatch')
    identities = selected['bindings']
    require(isinstance(identities, tuple) and identities, 'Empty or invalid binding list')
    require(all(isinstance(item, str) and item and item in standard.BINDINGS for item in identities),
            'Unknown/empty binding identity')
    require(len(identities) == len(set(identities)), 'Duplicate binding identity')
    records = [dict(standard.BINDINGS[item]) for item in identities]
    require(all(isinstance(row.get('symbol'), str) for row in records), 'Missing symbol')
    symbols = tuple(row['symbol'] for row in records)
    require(len(symbols) == len(set(symbols)), 'Symbol collision in selected bindings')
    require(len(records) == (8 if with_getri else 0) + (0 if standalone_getri else ((8 if with_trtri else 0) + (0 if standalone_trtri else ((8 if with_trtrs else 0) + (8 if with_posv else 0) + (0 if standalone_trtrs else 8 if standalone_gesvd else 92 if with_gesvd and with_complex_gelsd and with_q_generate else 88 if with_gesvd and with_q_generate else 68 if with_gesvd and with_complex_gelsd else 64 if with_gesvd else 4 if standalone_complex_gelsd else 84 if with_complex_gelsd and with_q_generate else 60 if with_complex_gelsd else 8 if only_q_generate else 24 if standalone_q_generate else 80 if with_q_generate else 16 if standalone_qr else 72 if with_qr else 8 if standalone_hermitian else 56 if with_hermitian else 48 if with_gelsd else 44 if with_stev else 40 if with_gesv else 32 if with_lu else 16))))), 'This finite set requires its exact ordinary binding count')
    # Only actual ABI names are reserved, not wildcard prefixes. The manifest is
    # content-bound; a later combined-product graph still needs its own review.
    reserved_path, reserved_hash = standard.RESERVED_BLAS_MANIFEST
    require(reserved_path == 'include/zynum/blas/abi_manifest.json', 'Reserved manifest path drift')
    reserved_file = _binding_source_root() / reserved_path
    require(reserved_file.is_file() and hashlib.sha256(reserved_file.read_bytes()).hexdigest() == reserved_hash,
            'Reserved BLAS manifest drift')
    reserved_manifest = json.loads(reserved_file.read_text())
    reserved = {item['name'] for section in ('fortran', 'cblas')
                for item in reserved_manifest[section]['exports']}
    reserved.update(PREFIX + name for name, _, _ in FUNCTIONS)
    reserved.update(OBJECT_SUPPORT_SYMBOLS['macho'])
    require(not reserved.intersection(symbols), 'Symbol collision with existing API/support exports')
    integer = ('int', 'c_int', 'i32', 32) if profile_id == 'mkl2026.lp64' else ('long long int', 'c_longlong', 'i64', 64)
    scalar_types = {'f32': ('s', 'float', 'f32'), 'f64': ('d', 'double', 'f64'),
                    'complex_f32': ('c', 'MKL_Complex8', 'ComplexF32'),
                    'complex_f64': ('z', 'MKL_Complex16', 'ComplexF64')}
    coverage, bindings = [], []
    for identity, row in zip(identities, records):
        exact_keys(row, ('id', 'profile', 'operation', 'symbol', 'scalar', 'integer_family', 'level',
                         'return', 'parameters', 'operands', 'owner', 'recipe', 'declaration',
                         'semantic_contracts', 'generation_acceptance', 'runtime_evidence'), 'standard binding')
        require(row['id'] == identity, 'Binding identity/key mismatch')
        require(row['profile'] == profile_id, 'Binding belongs to a different selected profile')
        require(row['operation'] in operations and row['operation'] in OPERATIONS,
                'Unreviewed operation')
        operation = OPERATIONS[row['operation']]
        scalar, level = row['scalar'], row['level']
        require(scalar in scalar_types and scalar in operation['scalar_types'], 'Unreviewed scalar')
        if row['operation'] in ('lu.inverse', 'triangular.inverse', 'triangular.solve', 'cholesky.driver', 'lu.factor', 'lu.solve', 'lu.driver', 'least_squares.rank_revealing', 'least_squares.rank_revealing_complex', 'svd.decompose'):
            require(integer[2] in operation['index_types'], 'Unsupported pivot index binding')
        require(row['integer_family'] == 'ordinary', 'Explicit64 is unavailable on this target')
        require(level in ('high', 'work'), 'Unreviewed LAPACKE level')
        stem, expected_parameters, operands, recipe = _binding_routine_contract(row['operation'])
        if row['operation'] == 'svd.decompose':
            if level == 'high':
                expected_parameters += (('superb', 'real', 'superdiagonal', 1, False),)
            else:
                expected_parameters += (('work', 'scalar', 'workspace', 1, False), ('lwork', 'lapack_int', 'workspace_length', 0, False))
                if scalar in ('complex_f32', 'complex_f64'):
                    expected_parameters += (('rwork', 'real', 'real_workspace', 1, False),)
        if row['operation'] == 'lu.inverse' and level == 'work':
            expected_parameters += (('work', 'scalar', 'workspace', 1, False), ('lwork', 'lapack_int', 'workspace_length', 0, False))
        if row['operation'] == 'least_squares.rank_revealing' and level == 'work':
            expected_parameters += (('work', 'scalar', 'workspace', 1, False), ('lwork', 'lapack_int', 'workspace_length', 0, False), ('iwork', 'lapack_int', 'integer_workspace', 1, False))
        if row['operation'] == 'least_squares.rank_revealing_complex' and level == 'work':
            expected_parameters += (('work', 'scalar', 'workspace', 1, False), ('lwork', 'lapack_int', 'workspace_length', 0, False), ('rwork', 'real', 'real_workspace', 1, False), ('iwork', 'lapack_int', 'integer_workspace', 1, False))
        if row['operation'] == 'tridiagonal.eigen' and level == 'work':
            expected_parameters += (('work', 'scalar', 'workspace', 1, False),)
        if row['operation'].startswith('qr.') and level == 'work':
            expected_parameters += (('work','scalar','workspace',1,False),('lwork','lapack_int','workspace_length',0,False))
        if row['operation'].startswith('hermitian.') and level == 'work':
            expected_parameters += (('work', 'scalar', 'workspace', 1, False), ('lwork', 'lapack_int', 'workspace_length', 0, False), ('rwork', 'real', 'real_workspace', 1, False))
        require(row['symbol'] == 'LAPACKE_' + scalar_types[scalar][0] + stem + ('_work' if level == 'work' else ''),
                'Symbol/operation/scalar/level mismatch')
        require(row['return'] == ('lapack_int', 'info', 0, False) and
                type(row['return'][2]) is int and type(row['return'][3]) is bool, 'INFO representation drift')
        require(row['parameters'] == expected_parameters and
                all(type(parameter[3]) is int and type(parameter[4]) is bool for parameter in row['parameters']),
                'Parameter order/role/type/pointer/const mismatch')
        require(row['operands'] == operands and tuple(role for role, _ in operands) == operation['operand_roles'],
                'Mathematical operand coverage mismatch')
        require(row['owner'] == 'lapacke-interface-owner' and
                row['recipe'] == recipe, 'Unreviewed call recipe')
        declaration = ('mkl_lapacke.h', standard.HEADER_SHA256, *standard.DECLARATIONS[row['symbol']])
        require(row['declaration'] == declaration, 'Declaration binding mismatch')
        semantics = standard.GETRI_SEMANTIC_CONTRACTS if row['operation'] == 'lu.inverse' else standard.TRTRI_SEMANTIC_CONTRACTS if row['operation'] == 'triangular.inverse' else standard.TRTRS_SEMANTIC_CONTRACTS if row['operation'] == 'triangular.solve' else standard.POSV_SEMANTIC_CONTRACTS if row['operation'] == 'cholesky.driver' else standard.GESVD_SEMANTIC_CONTRACTS if row['operation'] == 'svd.decompose' else standard.COMPLEX_GELSD_SEMANTIC_CONTRACTS if row['operation'] == 'least_squares.rank_revealing_complex' else standard.QR_SEMANTIC_CONTRACTS if row['operation'].startswith('qr.') else standard.HERMITIAN_SEMANTIC_CONTRACTS if row['operation'].startswith('hermitian.') else standard.GELSD_SEMANTIC_CONTRACTS if row['operation'] == 'least_squares.rank_revealing' else standard.STEV_SEMANTIC_CONTRACTS if row['operation'] == 'tridiagonal.eigen' else standard.GESV_SEMANTIC_CONTRACTS if row['operation'] == 'lu.driver' else standard.LU_SEMANTIC_CONTRACTS if row['operation'] in ('lu.factor', 'lu.solve') else standard.SEMANTIC_CONTRACTS
        require(row['semantic_contracts'] == semantics,
                'Semantic/raw-fault/hook contract mismatch')
        require(row['generation_acceptance'] is None and row['runtime_evidence'] == (),
                'A binding cannot self-authorize or invent runtime evidence')
        wire_parameters = []
        for name, typ, role, depth, const in expected_parameters:
            representation = (('float', 'f32', 'f32') if scalar in ('f32', 'complex_f32') else ('double', 'f64', 'f64')) if typ == 'real' else integer[:3] if typ == 'lapack_int' else (
                ('int', 'c_int', 'i32') if typ == 'c_int' else (
                    ('char', 'i8', 'u8-bitcast') if typ == 'c_char' else
                    (scalar_types[scalar][1], scalar_types[scalar][2], scalar_types[scalar][2])))
            wire_parameters.append({'name': name, 'role': role, 'c_type': representation[0],
                                    'zig_type': representation[1], 'internal_type': representation[2],
                                    'pointer_depth': depth, 'pointee_const': const})
        bindings.append({**row, 'wire_parameters': wire_parameters,
                         'wire_return': {'c_type': integer[0], 'zig_type': integer[1], 'internal_type': integer[2]},
                         'value_preserving_integer_casts': True})
        coverage.append((scalar, row['operation'], level))
    expected_coverage = [(scalar, operation, level) for group in groups for scalar in scalar_types
                         for operation in group if scalar in OPERATIONS[operation]['scalar_types'] for level in ('high', 'work')]
    require(coverage == expected_coverage, 'Incomplete or reordered finite binding coverage')
    dependencies = standard.SOURCE_DEPENDENCIES
    require(isinstance(dependencies, tuple) and dependencies and
            all(isinstance(row, tuple) and len(row) == 2 and isinstance(row[0], str) and
                row[0].startswith('src/') and '..' not in row[0].split('/') and
                isinstance(row[1], str) and len(row[1]) == 64 and
                all(char in '0123456789abcdef' for char in row[1]) for row in dependencies),
            'Malformed source envelope')
    require(tuple(sorted(dependencies)) == dependencies and
            len({path for path, _ in dependencies}) == len(dependencies), 'Duplicate/unordered source dependency')
    require(hashlib.sha256(json.dumps(dependencies, separators=(',', ':')).encode()).hexdigest() ==
            standard.SOURCE_ENVELOPE_SHA256, 'Source envelope changed or dependency omitted')
    if with_gesvd:
        gesvd_records = tuple(row for row in records if row['operation'] == 'svd.decompose')
        require(_binding_digest(tuple((row['symbol'], row['declaration'], row['parameters']) for row in gesvd_records)) ==
                standard.GESVD_SIGNATURE_INVENTORY_SHA256, 'GESVD original declaration/type inventory mismatch')
    complex_signature_inventory = _binding_digest(tuple((row['symbol'], row['declaration'], row['parameters']) for row in records)) if with_complex_gelsd or with_gesvd or with_posv or with_trtrs or with_trtri or with_getri else None
    return {'schema': 'standard-finite-binding-plan-v1', 'profile': profile_id, 'set': set_id,
            'target': dict(standard.TARGET), 'bindings': bindings, 'symbols': symbols,
            'owner_imports': (dict(standard.OWNER),),
            'semantic_contracts': tuple(dict.fromkeys(item for row in records for item in row['semantic_contracts'])) if with_complex_gelsd or with_gesvd or with_posv or with_trtrs or with_trtri or with_getri else (*standard.QR_SEMANTIC_CONTRACTS, *(standard.HERMITIAN_SEMANTIC_CONTRACTS + standard.COMBINED_SEMANTIC_CONTRACTS + standard.GESV_SEMANTIC_CONTRACTS + standard.STEV_SEMANTIC_CONTRACTS + standard.GELSD_SEMANTIC_CONTRACTS if not standalone_qr else ())) if with_qr else standard.HERMITIAN_SEMANTIC_CONTRACTS if standalone_hermitian else (*standard.HERMITIAN_SEMANTIC_CONTRACTS, *standard.COMBINED_SEMANTIC_CONTRACTS, *standard.GESV_SEMANTIC_CONTRACTS, *standard.STEV_SEMANTIC_CONTRACTS, *standard.GELSD_SEMANTIC_CONTRACTS) if with_hermitian else (*standard.COMBINED_SEMANTIC_CONTRACTS, *standard.GESV_SEMANTIC_CONTRACTS, *standard.STEV_SEMANTIC_CONTRACTS, *standard.GELSD_SEMANTIC_CONTRACTS) if with_gelsd else (*standard.COMBINED_SEMANTIC_CONTRACTS, *standard.GESV_SEMANTIC_CONTRACTS, *standard.STEV_SEMANTIC_CONTRACTS) if with_stev else (*standard.COMBINED_SEMANTIC_CONTRACTS, *standard.GESV_SEMANTIC_CONTRACTS) if with_gesv else standard.COMBINED_SEMANTIC_CONTRACTS if with_lu else standard.SEMANTIC_CONTRACTS,
            'source_dependencies': dependencies, 'source_envelope_sha256': standard.SOURCE_ENVELOPE_SHA256,
            'source_snapshot_sha256': standard.SOURCE_SNAPSHOT_SHA256,
            'external_module_contracts': standard.EXTERNAL_MODULE_CONTRACTS,
            'source_scope': 'literal-import-envelope-with-unverified-external-build-obligations',
            'declaration_inventory_sha256': complex_signature_inventory if with_complex_gelsd or with_gesvd or with_posv or with_trtrs or with_trtri or with_getri else standard.Q_GENERATE_SIGNATURE_INVENTORY_SHA256 if with_q_generate else standard.QR_SIGNATURE_INVENTORY_SHA256 if with_qr else standard.HERMITIAN_SIGNATURE_INVENTORY_SHA256 if with_hermitian else standard.GELSD_SIGNATURE_INVENTORY_SHA256 if with_gelsd else standard.STEV_SIGNATURE_INVENTORY_SHA256 if with_stev else standard.DRIVER_SIGNATURE_INVENTORY_SHA256 if with_gesv else standard.COMBINED_SIGNATURE_INVENTORY_SHA256 if with_lu else standard.SIGNATURE_INVENTORY_SHA256,
            'declaration_headers': standard.DECLARATION_HEADERS,
            'reserved_manifest': standard.RESERVED_BLAS_MANIFEST,
            'output_paths': tuple('interface_subset/' + profile_id + '/' + set_id + '/' + name
                                  for name in ('subset.h', 'exports.zig', 'abi_manifest.json', 'exports.def')),
            'runtime_abi_status': 'pending', 'build_owner_identity_status': 'pending',
            'full_profile_ready': False}


def select_binding_set(profile_id, set_id, target_contract):
    """Return an immutable exact accepted plan, or fail before any write/formatter.

    There is no acceptance argument or registration API. Only an exact entry in
    the source-owned acceptance table can authorize finite callable output.
    An accepted finite plan would not imply a ready complete vendor profile.
    """
    if isinstance(set_id, str) and set_id.startswith('sparse-'):
        import standard_sparse_bindings as standard
    else:
        import standard_abi_bindings as standard
    candidate = _binding_candidate(profile_id, set_id, target_contract)
    acceptance = standard.GENERATION_ACCEPTANCES.get((profile_id, set_id, target_contract))
    require(acceptance is not None, 'No callable generation acceptance for selected finite set')
    acceptance = dict(acceptance)
    exact_keys(acceptance, ('id', 'profile', 'set', 'target', 'scope', 'plan_sha256',
                            'source_envelope_sha256', 'review_sha256'), 'generation acceptance')
    require(isinstance(acceptance['id'], str) and acceptance['id'] and '*' not in acceptance['id'],
            'Invalid acceptance identity')
    require((acceptance['profile'], acceptance['set'], acceptance['target']) ==
            (profile_id, set_id, target_contract), 'Acceptance selection mismatch')
    require(acceptance['scope'] == 'finite-callable-generation-only', 'Invalid acceptance scope')
    require(acceptance['plan_sha256'] == _binding_digest(candidate) and
            acceptance['source_envelope_sha256'] == candidate['source_envelope_sha256'],
            'Acceptance does not bind the complete selected plan and source envelope')
    review = acceptance['review_sha256']
    require(isinstance(review, str) and len(review) == 64 and all(c in '0123456789abcdef' for c in review),
            'Missing acceptance review digest')
    source_root = _binding_source_root()
    for path, expected in candidate['source_dependencies']:
        source = source_root / path
        require(source.is_file() and hashlib.sha256(source.read_bytes()).hexdigest() == expected,
                'Accepted source dependency missing or changed: ' + path)
    return _binding_freeze({**candidate, 'generation_acceptance': acceptance})
