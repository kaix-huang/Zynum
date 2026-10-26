# Copyright (C) 2026 Zynum contributors
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Shared mathematical identities and type domains, without interface state.

IDs reuse the reviewed interface-profile schema. This small selection describes
the existing own-operation type domains; it supplies no ABI layouts, symbols,
resource managers, algorithm bodies or new numerical execution evidence.
"""

REAL_AND_COMPLEX = ('f32', 'f64', 'complex_f32', 'complex_f64')
OPERATIONS = {
    'svd.decompose': {
        'scalar_types': REAL_AND_COMPLEX,
        'index_types': ('i32',),
        'operand_roles': ('matrix', 'singular_values', 'left_vectors', 'right_vectors', 'jobs'),
        'contract': 'Compute finite rectangular singular values and optional full/thin orthonormal left/right singular vectors over borrowed native storage.',
    },
    'least_squares.rank_revealing_complex': {
        'scalar_types': ('complex_f32', 'complex_f64'),
        'index_types': ('i32', 'i64'),
        'operand_roles': ('matrix', 'rhs', 'singular_values', 'rank'),
        'contract': 'Compute finite complex minimum-norm least-squares solutions, real singular values and numerical rank over borrowed native storage.',
    },
    'least_squares.rank_revealing': {
        'scalar_types': ('f32', 'f64'),
        'index_types': ('i32', 'i64'),
        'operand_roles': ('matrix', 'rhs', 'singular_values', 'rank'),
        'contract': 'Compute minimum-norm real least-squares solutions, singular values and numerical rank for rectangular matrices.',
    },
    'tridiagonal.eigen': {
        'scalar_types': ('f32', 'f64'),
        'index_types': (),
        'operand_roles': ('diagonal', 'offdiagonal', 'vectors', 'job'),
        'contract': 'Compute sorted eigenvalues and optional orthonormal eigenvectors of a real symmetric tridiagonal matrix.',
    },
    'triangular.inverse': {
        'scalar_types': REAL_AND_COMPLEX,
        'index_types': ('i32', 'i64'),
        'operand_roles': ('matrix', 'triangle', 'diagonal_kind'),
        'contract': 'Invert a borrowed triangular matrix in place; skip unit diagonal storage, preflight exact nonunit singularity and borrow existing wide workspace for whole-column computation.',
    },
    'triangular.solve': {
        'scalar_types': REAL_AND_COMPLEX,
        'index_types': ('i32', 'i64'),
        'operand_roles': ('factor', 'rhs', 'triangle', 'operation', 'diagonal_kind'),
        'contract': 'Solve op(A) X = B from a borrowed triangular matrix, excluding implicit unit diagonals and rejecting exact nonunit singularity before RHS mutation.',
    },
    'cholesky.driver': {
        'scalar_types': REAL_AND_COMPLEX,
        'index_types': ('i32', 'i64'),
        'operand_roles': ('matrix', 'rhs', 'triangle'),
        'contract': 'Factor a positive-definite matrix and solve A X = B on successful factorization, retaining the caller-owned factor.',
    },
    'cholesky.inverse': {
        'scalar_types': REAL_AND_COMPLEX,
        'index_types': ('i32', 'i64'),
        'operand_roles': ('matrix', 'triangle'),
        'contract': 'Invert a borrowed Cholesky factor through the shared wide triangular inverse and Hermitian Gram product, publishing only the selected triangle.',
    },
    'cholesky.factor': {
        'scalar_types': REAL_AND_COMPLEX,
        'index_types': (),
        'operand_roles': ('matrix', 'triangle'),
        'contract': 'Factor the selected triangle of a positive-definite matrix.',
    },
    'cholesky.solve': {
        'scalar_types': REAL_AND_COMPLEX,
        'index_types': (),
        'operand_roles': ('factor', 'rhs', 'triangle'),
        'contract': 'Solve A X = B from a successful Cholesky factor.',
    },
    'lu.inverse': {
        'scalar_types': REAL_AND_COMPLEX,
        'index_types': ('i32', 'i64'),
        'operand_roles': ('matrix', 'pivots'),
        'contract': 'Invert borrowed combined LU factors with only the first n-1 validated pivots; retain wide output until full publication and borrow the existing workspace lifecycle.',
    },
    'lu.factor': {
        'scalar_types': REAL_AND_COMPLEX,
        'index_types': ('i32', 'i64'),
        'operand_roles': ('matrix', 'pivots'),
        'contract': 'Factor a rectangular general matrix with partial pivoting.',
    },
    'lu.driver': {
        'scalar_types': REAL_AND_COMPLEX,
        'index_types': ('i32', 'i64'),
        'operand_roles': ('matrix', 'pivots', 'rhs'),
        'contract': 'Factor A with partial pivoting and solve A X = B unless singular.',
    },
    'lu.solve': {
        'scalar_types': REAL_AND_COMPLEX,
        'index_types': ('i32', 'i64'),
        'operand_roles': ('factor', 'pivots', 'rhs', 'operation'),
        'contract': 'Solve op(A) X = B from LU factors and pivot steps.',
    },
    'sparse.mm': {
        'scalar_types': REAL_AND_COMPLEX,
        'index_types': ('i32', 'i64', 'u32', 'u64'),
        'operand_roles': ('matrix', 'operation', 'product_scale', 'input', 'result_scale', 'result'),
        'contract': 'Compute C = alpha op(A) B + beta C over sparse and dense panels.',
    },
    'sparse.mv': {
        'scalar_types': REAL_AND_COMPLEX,
        'index_types': ('i32', 'i64', 'u32', 'u64'),
        'operand_roles': ('matrix', 'operation', 'product_scale', 'input', 'result_scale', 'result'),
        'contract': 'Compute y = alpha op(A) x + beta y over sparse storage.',
    },
}

# Pure mathematical domains; no interface state or execution owner.
OPERATIONS.update({'symmetric.generalized_eigen': {
    'scalar_types': ('f32', 'f64'), 'index_types': (),
    'operand_roles': ('matrix', 'metric', 'eigenvalues', 'job', 'triangle', 'problem'),
    'contract': 'All real eigenvalues and optional metric-normalized vectors for symmetric-definite problems A*x=lambda*B*x, A*B*x=lambda*x, or B*A*x=lambda*x.',
}})

OPERATIONS.update({'symmetric.eigen': {
    'scalar_types': ('f32', 'f64'), 'index_types': (),
    'operand_roles': ('matrix', 'eigenvalues', 'job', 'triangle'),
    'contract': 'All sorted real eigenvalues and optional orthonormal vectors of a real symmetric matrix.',
}})

OPERATIONS.update({
    'hermitian.eigen': {
        'scalar_types': ('complex_f32', 'complex_f64'), 'index_types': (),
        'operand_roles': ('matrix', 'eigenvalues', 'job', 'triangle'),
        'contract': 'All sorted real eigenvalues and optional orthonormal vectors of a finite complex Hermitian matrix.',
    },
    'hermitian.generalized_eigen': {
        'scalar_types': ('complex_f32', 'complex_f64'), 'index_types': (),
        'operand_roles': ('matrix', 'metric', 'eigenvalues', 'job', 'triangle', 'problem'),
        'contract': 'All real eigenvalues and optionally B/B-inverse normalized vectors of three finite Hermitian-definite problems with positive definite metric.',
    },
})

OPERATIONS.update({'qr.factor': {'scalar_types': ('f32', 'f64', 'complex_f32', 'complex_f64'), 'index_types': (), 'operand_roles': ('matrix', 'reflectors'), 'contract': 'Compact Householder QR and Q application over borrowed owner storage.'}, 'qr.apply_real': {'scalar_types': ('f32', 'f64'), 'index_types': (), 'operand_roles': ('factor', 'reflectors', 'target', 'side', 'operation'), 'contract': 'Compact Householder QR and Q application over borrowed owner storage.'}, 'qr.apply_complex': {'scalar_types': ('complex_f32', 'complex_f64'), 'index_types': (), 'operand_roles': ('factor', 'reflectors', 'target', 'side', 'operation'), 'contract': 'Compact Householder QR and Q application over borrowed owner storage.'}})

OPERATIONS.update({'qr.generate_real': {'scalar_types': ('f32', 'f64'), 'index_types': (), 'operand_roles': ('matrix', 'reflectors'), 'contract': 'Expand compact Householder reflectors into explicit orthonormal columns in borrowed owner storage.'}, 'qr.generate_complex': {'scalar_types': ('complex_f32', 'complex_f64'), 'index_types': (), 'operand_roles': ('matrix', 'reflectors'), 'contract': 'Expand compact Householder reflectors into explicit orthonormal columns in borrowed owner storage.'}})

OPERATIONS.update({'sparse.trsv': {'scalar_types': ('f32', 'f64', 'complex_f32', 'complex_f64'), 'index_types': ('i32', 'i64'), 'operand_roles': ('matrix', 'operation', 'product_scale', 'input', 'result'), 'contract': 'Borrowed handle triangular/diagonal solve; shared Context staging and own IEEE arithmetic.'}, 'sparse.trsm': {'scalar_types': ('f32', 'f64', 'complex_f32', 'complex_f64'), 'index_types': ('i32', 'i64'), 'operand_roles': ('matrix', 'operation', 'product_scale', 'input', 'result'), 'contract': 'Borrowed handle triangular/diagonal solve; shared Context staging and own IEEE arithmetic.'}})

# Full-rank QR/LQ mathematics; independent of any external interface spelling.
OPERATIONS.update({'least_squares.full_rank': {
    'scalar_types': REAL_AND_COMPLEX, 'index_types': ('i32',),
    'operand_roles': ('matrix', 'rhs', 'operation'),
    'contract': 'Solve full-rank rectangular least-squares or minimum-norm systems for A or its adjoint, retaining QR/LQ factors and transformed residual rows; report exact triangular singularity.',
}})

# Managed SVD least squares; no external layout, overwrite or workspace policy.
OPERATIONS.update({'least_squares.svd': {
    'scalar_types': REAL_AND_COMPLEX, 'index_types': ('i32',),
    'operand_roles': ('matrix', 'rhs', 'solution', 'singular_values', 'right_vectors', 'residual', 'relative_cutoff', 'absolute_cutoff'),
    'contract': 'Solve rectangular minimum-norm least-squares systems from immutable inputs with explicit nonnegative relative and absolute singular cutoffs; publish solutions, descending singular values and right vectors transactionally, with residual coordinates for full-column-rank tall systems.',
}})
