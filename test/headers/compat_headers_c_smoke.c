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
