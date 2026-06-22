// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later

#include <stdio.h>

#include <zynum/blas/cblas.h>

int main(void) {
    const int m = 2;
    const int n = 2;
    const int k = 3;

    /* This CBLAS example uses row-major storage. */
    const double a[] = {
        1.0, 2.0, 3.0,
        4.0, 5.0, 6.0,
    };
    const double b[] = {
        7.0,  8.0,
        9.0, 10.0,
        11.0, 12.0,
    };
    double c[4] = {0.0, 0.0, 0.0, 0.0};

    cblas_dgemm(
        CblasRowMajor,
        CblasNoTrans,
        CblasNoTrans,
        m,
        n,
        k,
        1.0,
        a,
        k,
        b,
        n,
        0.0,
        c,
        n
    );

    puts("C = A x B");
    for (int row = 0; row < m; row += 1) {
        for (int col = 0; col < n; col += 1) {
            printf("%8.1f", c[row * n + col]);
        }
        putchar('\n');
    }

    return 0;
}
