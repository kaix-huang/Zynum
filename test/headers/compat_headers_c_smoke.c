// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later

#include <zynum/blas/blas.h>
#include <zynum/blas/cblas.h>

void zynum_blas_c_header_smoke(void) {
    zynum_blas_int n = 1;
    zynum_blas_complex_float alpha = {1.0f, 0.0f};
    zynum_blas_complex_float x[1] = {{1.0f, 0.0f}};
    zynum_blas_complex_float y[1] = {{0.0f, 0.0f}};
    zynum_blas_complex_float out = {0.0f, 0.0f};

    cblas_caxpy(n, &alpha, x, 1, y, 1);
    cblas_cdotu_sub(n, x, 1, y, 1, &out);
    caxpy_(&n, &alpha, x, &n, y, &n);
    cdotu_sub_(&n, x, &n, y, &n, &out);
}

#if defined(__APPLE__) && defined(__aarch64__)
/* The reviewed LAPACKE target permits caller-defined C complex types. */
#include <complex.h>
#define lapack_complex_float float _Complex
#define lapack_complex_double double _Complex
#include "../../interface_subset/mkl2026.lp64/potrf-potrs-getrf-getrs-ordinary-lp64-v1/subset.h"

#define CHECK_COMPLEX_FACTOR(tag, type, suffix) \
    _Static_assert(_Generic(&LAPACKE_##tag##potrf##suffix, \
        lapack_int (*)(int, char, lapack_int, type *, lapack_int): 1, \
        default: 0), "LAPACKE factor must honor the public complex type")
#define CHECK_COMPLEX_SOLVE(tag, type, suffix) \
    _Static_assert(_Generic(&LAPACKE_##tag##potrs##suffix, \
        lapack_int (*)(int, char, lapack_int, lapack_int, const type *, \
                      lapack_int, type *, lapack_int): 1, \
        default: 0), "LAPACKE solve must honor the public complex type")
CHECK_COMPLEX_FACTOR(c, lapack_complex_float, );
CHECK_COMPLEX_FACTOR(c, lapack_complex_float, _work);
CHECK_COMPLEX_FACTOR(z, lapack_complex_double, );
CHECK_COMPLEX_FACTOR(z, lapack_complex_double, _work);
CHECK_COMPLEX_SOLVE(c, lapack_complex_float, );
CHECK_COMPLEX_SOLVE(c, lapack_complex_float, _work);
CHECK_COMPLEX_SOLVE(z, lapack_complex_double, );
CHECK_COMPLEX_SOLVE(z, lapack_complex_double, _work);
#undef CHECK_COMPLEX_FACTOR
#undef CHECK_COMPLEX_SOLVE
#endif
