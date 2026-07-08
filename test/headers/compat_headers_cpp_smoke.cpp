// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later

#include <zynum/blas/blas.h>
#include <zynum/blas/cblas.h>

static_assert(CblasRowMajor == 101, "CBLAS enum value drifted");
static_assert(sizeof(zynum_blas_int) == 4, "Zynum BLAS currently uses LP64 BLAS integers");

void zynum_blas_cpp_header_smoke() {
    zynum_blas_int n = 1;
    zynum_blas_complex_double alpha{1.0, 0.0};
    zynum_blas_complex_double x[1]{{1.0, 0.0}};
    zynum_blas_complex_double y[1]{{0.0, 0.0}};
    zynum_blas_complex_double out{0.0, 0.0};

    cblas_zaxpy(n, &alpha, x, 1, y, 1);
    cblas_zdotc_sub(n, x, 1, y, 1, &out);
    zaxpy_(&n, &alpha, x, &n, y, &n);
    zdotc_sub_(&n, x, &n, y, &n, &out);
}
