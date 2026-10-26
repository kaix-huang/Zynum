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

#if defined(__APPLE__) && defined(__aarch64__)
/* Keep std::complex usable through the vendor's public type overrides. */
#include <complex>
#include <type_traits>
#define lapack_complex_float std::complex<float>
#define lapack_complex_double std::complex<double>
#include "../../interface_subset/mkl2026.lp64/potrf-potrs-getrf-getrs-ordinary-lp64-v1/subset.h"

#define CHECK_COMPLEX_FACTOR(tag, type, suffix) \
    static_assert(std::is_same_v<decltype(&LAPACKE_##tag##potrf##suffix), \
        lapack_int (*)(int, char, lapack_int, type *, lapack_int)>, \
        "LAPACKE factor must honor the public complex type")
#define CHECK_COMPLEX_SOLVE(tag, type, suffix) \
    static_assert(std::is_same_v<decltype(&LAPACKE_##tag##potrs##suffix), \
        lapack_int (*)(int, char, lapack_int, lapack_int, const type *, \
                      lapack_int, type *, lapack_int)>, \
        "LAPACKE solve must honor the public complex type")
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

/* Also exercise the MKL aliases; use a second finite header so the first
 * public-alias checks above continue to compile independently. */
#undef MKL_Complex8
#undef MKL_Complex16
#define MKL_Complex8 std::complex<float>
#define MKL_Complex16 std::complex<double>
#include "../../interface_subset/mkl2026.lp64/potrf-potrs-getrf-getrs-gesv-ordinary-lp64-v1/subset.h"
static_assert(std::is_same_v<MKL_Complex8, std::complex<float>>,
              "MKL complex8 alias must remain usable");
static_assert(std::is_same_v<MKL_Complex16, std::complex<double>>,
              "MKL complex16 alias must remain usable");
static_assert(std::is_same_v<decltype(&LAPACKE_zgesv),
              lapack_int (*)(int, lapack_int, lapack_int, std::complex<double> *,
                            lapack_int, lapack_int *, std::complex<double> *, lapack_int)>,
              "LAPACKE driver must honor the MKL complex type");
#endif
