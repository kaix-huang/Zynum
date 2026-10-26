// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later

/* Finite experimental declarations; not a complete MKL SDK. */
#ifndef ZYNUM_LAPACKE_POTRF_POTRS_GETRF_GETRS_GESV_STEV_GELSD_POSV_TRTRI_ILP64_V1_H
#define ZYNUM_LAPACKE_POTRF_POTRS_GETRF_GETRS_GESV_STEV_GELSD_POSV_TRTRI_ILP64_V1_H
#include <stddef.h>
#include <limits.h>
/* Standalone default types; preserve an already included MKL type header. */
#if !defined(_MKL_TYPES_H_)
#ifndef MKL_Complex8
typedef struct _MKL_Complex8 { float real, imag; } zynum_lapack_complex_float;
#define MKL_Complex8 zynum_lapack_complex_float
#endif
#ifndef MKL_Complex16
typedef struct _MKL_Complex16 { double real, imag; } zynum_lapack_complex_double;
#define MKL_Complex16 zynum_lapack_complex_double
#endif
#ifndef MKL_INT
#ifdef MKL_ILP64
#define MKL_INT long long int
#else
#define MKL_INT int
#endif
#endif
#endif
#ifndef lapack_int
#define lapack_int MKL_INT
#endif
#ifndef lapack_logical
#define lapack_logical lapack_int
#endif
#ifndef lapack_complex_float
#define lapack_complex_float MKL_Complex8
#endif
#ifndef lapack_complex_double
#define lapack_complex_double MKL_Complex16
#endif
#ifndef LAPACK_ROW_MAJOR
#define LAPACK_ROW_MAJOR 101
#endif
#ifndef LAPACK_COL_MAJOR
#define LAPACK_COL_MAJOR 102
#endif
#if !defined(__APPLE__) || !defined(__aarch64__)
#error "This finite declaration contract requires Apple arm64"
#endif
#if !defined(MKL_ILP64)
#error "MKL_ILP64 must agree with this finite declaration profile"
#endif
#ifdef __cplusplus
#include <complex>
#include <type_traits>
#include <utility>
#ifndef ZYNUM_LAPACKE_CPP_COMPLEX_LAYOUT_V1
#define ZYNUM_LAPACKE_CPP_COMPLEX_LAYOUT_V1
namespace zynum_lapacke_detail_v1 {
template<class T, class R, class = void>
struct ComplexFields : std::is_same<T, std::complex<R> > {};
template<class T, class R>
struct ComplexFields<T, R, decltype((void)std::declval<T&>().real, (void)std::declval<T&>().imag, void())>
  : std::integral_constant<bool, std::is_standard_layout<T>::value &&
      std::is_same<decltype(std::declval<T&>().real), R>::value &&
      std::is_same<decltype(std::declval<T&>().imag), R>::value &&
      offsetof(T, real) == 0 && offsetof(T, imag) == sizeof(R)> {};
}
#endif
#define ZYNUM_SUBSET_ASSERT static_assert
#define ZYNUM_SUBSET_ALIGNOF alignof
extern "C" {
#else
#define ZYNUM_SUBSET_ASSERT _Static_assert
#define ZYNUM_SUBSET_ALIGNOF _Alignof
#endif
ZYNUM_SUBSET_ASSERT(CHAR_BIT == 8 && sizeof(void*) == 8, "pointer/byte representation");
ZYNUM_SUBSET_ASSERT(sizeof(int) == 4 && ((char)-1) < 0, "layout/char representation");
ZYNUM_SUBSET_ASSERT(sizeof(MKL_INT) == 8, "ordinary integer profile");
ZYNUM_SUBSET_ASSERT(sizeof(long long int) == 8, "ILP64 C integer representation");
ZYNUM_SUBSET_ASSERT(sizeof(MKL_Complex8) == 8 && ZYNUM_SUBSET_ALIGNOF(MKL_Complex8) == 4, "complex8 representation");
#ifdef __cplusplus
ZYNUM_SUBSET_ASSERT((zynum_lapacke_detail_v1::ComplexFields<MKL_Complex8, float>::value), "complex8 fields");
#else
ZYNUM_SUBSET_ASSERT(offsetof(MKL_Complex8, real) == 0 && offsetof(MKL_Complex8, imag) == 4, "complex8 fields");
#endif
ZYNUM_SUBSET_ASSERT(sizeof(MKL_Complex16) == 16 && ZYNUM_SUBSET_ALIGNOF(MKL_Complex16) == 8, "complex16 representation");
#ifdef __cplusplus
ZYNUM_SUBSET_ASSERT((zynum_lapacke_detail_v1::ComplexFields<MKL_Complex16, double>::value), "complex16 fields");
#else
ZYNUM_SUBSET_ASSERT(offsetof(MKL_Complex16, real) == 0 && offsetof(MKL_Complex16, imag) == 8, "complex16 fields");
#endif
ZYNUM_SUBSET_ASSERT(sizeof(lapack_complex_float) == 8 && ZYNUM_SUBSET_ALIGNOF(lapack_complex_float) >= 4, "public complex8 representation");
ZYNUM_SUBSET_ASSERT(sizeof(lapack_complex_double) == 16 && ZYNUM_SUBSET_ALIGNOF(lapack_complex_double) >= 8, "public complex16 representation");
long long int LAPACKE_spotrf(int matrix_layout, char uplo, long long int n, float* a, long long int lda);
long long int LAPACKE_spotrf_work(int matrix_layout, char uplo, long long int n, float* a, long long int lda);
long long int LAPACKE_spotrs(int matrix_layout, char uplo, long long int n, long long int nrhs, const float* a, long long int lda, float* b, long long int ldb);
long long int LAPACKE_spotrs_work(int matrix_layout, char uplo, long long int n, long long int nrhs, const float* a, long long int lda, float* b, long long int ldb);
long long int LAPACKE_dpotrf(int matrix_layout, char uplo, long long int n, double* a, long long int lda);
long long int LAPACKE_dpotrf_work(int matrix_layout, char uplo, long long int n, double* a, long long int lda);
long long int LAPACKE_dpotrs(int matrix_layout, char uplo, long long int n, long long int nrhs, const double* a, long long int lda, double* b, long long int ldb);
long long int LAPACKE_dpotrs_work(int matrix_layout, char uplo, long long int n, long long int nrhs, const double* a, long long int lda, double* b, long long int ldb);
long long int LAPACKE_cpotrf(int matrix_layout, char uplo, long long int n, lapack_complex_float* a, long long int lda);
long long int LAPACKE_cpotrf_work(int matrix_layout, char uplo, long long int n, lapack_complex_float* a, long long int lda);
long long int LAPACKE_cpotrs(int matrix_layout, char uplo, long long int n, long long int nrhs, const lapack_complex_float* a, long long int lda, lapack_complex_float* b, long long int ldb);
long long int LAPACKE_cpotrs_work(int matrix_layout, char uplo, long long int n, long long int nrhs, const lapack_complex_float* a, long long int lda, lapack_complex_float* b, long long int ldb);
long long int LAPACKE_zpotrf(int matrix_layout, char uplo, long long int n, lapack_complex_double* a, long long int lda);
long long int LAPACKE_zpotrf_work(int matrix_layout, char uplo, long long int n, lapack_complex_double* a, long long int lda);
long long int LAPACKE_zpotrs(int matrix_layout, char uplo, long long int n, long long int nrhs, const lapack_complex_double* a, long long int lda, lapack_complex_double* b, long long int ldb);
long long int LAPACKE_zpotrs_work(int matrix_layout, char uplo, long long int n, long long int nrhs, const lapack_complex_double* a, long long int lda, lapack_complex_double* b, long long int ldb);
long long int LAPACKE_sgetrf(int matrix_layout, long long int m, long long int n, float* a, long long int lda, long long int* ipiv);
long long int LAPACKE_sgetrf_work(int matrix_layout, long long int m, long long int n, float* a, long long int lda, long long int* ipiv);
long long int LAPACKE_sgetrs(int matrix_layout, char trans, long long int n, long long int nrhs, const float* a, long long int lda, const long long int* ipiv, float* b, long long int ldb);
long long int LAPACKE_sgetrs_work(int matrix_layout, char trans, long long int n, long long int nrhs, const float* a, long long int lda, const long long int* ipiv, float* b, long long int ldb);
long long int LAPACKE_dgetrf(int matrix_layout, long long int m, long long int n, double* a, long long int lda, long long int* ipiv);
long long int LAPACKE_dgetrf_work(int matrix_layout, long long int m, long long int n, double* a, long long int lda, long long int* ipiv);
long long int LAPACKE_dgetrs(int matrix_layout, char trans, long long int n, long long int nrhs, const double* a, long long int lda, const long long int* ipiv, double* b, long long int ldb);
long long int LAPACKE_dgetrs_work(int matrix_layout, char trans, long long int n, long long int nrhs, const double* a, long long int lda, const long long int* ipiv, double* b, long long int ldb);
long long int LAPACKE_cgetrf(int matrix_layout, long long int m, long long int n, lapack_complex_float* a, long long int lda, long long int* ipiv);
long long int LAPACKE_cgetrf_work(int matrix_layout, long long int m, long long int n, lapack_complex_float* a, long long int lda, long long int* ipiv);
long long int LAPACKE_cgetrs(int matrix_layout, char trans, long long int n, long long int nrhs, const lapack_complex_float* a, long long int lda, const long long int* ipiv, lapack_complex_float* b, long long int ldb);
long long int LAPACKE_cgetrs_work(int matrix_layout, char trans, long long int n, long long int nrhs, const lapack_complex_float* a, long long int lda, const long long int* ipiv, lapack_complex_float* b, long long int ldb);
long long int LAPACKE_zgetrf(int matrix_layout, long long int m, long long int n, lapack_complex_double* a, long long int lda, long long int* ipiv);
long long int LAPACKE_zgetrf_work(int matrix_layout, long long int m, long long int n, lapack_complex_double* a, long long int lda, long long int* ipiv);
long long int LAPACKE_zgetrs(int matrix_layout, char trans, long long int n, long long int nrhs, const lapack_complex_double* a, long long int lda, const long long int* ipiv, lapack_complex_double* b, long long int ldb);
long long int LAPACKE_zgetrs_work(int matrix_layout, char trans, long long int n, long long int nrhs, const lapack_complex_double* a, long long int lda, const long long int* ipiv, lapack_complex_double* b, long long int ldb);
long long int LAPACKE_sgesv(int matrix_layout, long long int n, long long int nrhs, float* a, long long int lda, long long int* ipiv, float* b, long long int ldb);
long long int LAPACKE_sgesv_work(int matrix_layout, long long int n, long long int nrhs, float* a, long long int lda, long long int* ipiv, float* b, long long int ldb);
long long int LAPACKE_dgesv(int matrix_layout, long long int n, long long int nrhs, double* a, long long int lda, long long int* ipiv, double* b, long long int ldb);
long long int LAPACKE_dgesv_work(int matrix_layout, long long int n, long long int nrhs, double* a, long long int lda, long long int* ipiv, double* b, long long int ldb);
long long int LAPACKE_cgesv(int matrix_layout, long long int n, long long int nrhs, lapack_complex_float* a, long long int lda, long long int* ipiv, lapack_complex_float* b, long long int ldb);
long long int LAPACKE_cgesv_work(int matrix_layout, long long int n, long long int nrhs, lapack_complex_float* a, long long int lda, long long int* ipiv, lapack_complex_float* b, long long int ldb);
long long int LAPACKE_zgesv(int matrix_layout, long long int n, long long int nrhs, lapack_complex_double* a, long long int lda, long long int* ipiv, lapack_complex_double* b, long long int ldb);
long long int LAPACKE_zgesv_work(int matrix_layout, long long int n, long long int nrhs, lapack_complex_double* a, long long int lda, long long int* ipiv, lapack_complex_double* b, long long int ldb);
long long int LAPACKE_sstev(int matrix_layout, char jobz, long long int n, float* d, float* e, float* z, long long int ldz);
long long int LAPACKE_sstev_work(int matrix_layout, char jobz, long long int n, float* d, float* e, float* z, long long int ldz, float* work);
long long int LAPACKE_dstev(int matrix_layout, char jobz, long long int n, double* d, double* e, double* z, long long int ldz);
long long int LAPACKE_dstev_work(int matrix_layout, char jobz, long long int n, double* d, double* e, double* z, long long int ldz, double* work);
long long int LAPACKE_sgelsd(int matrix_layout, long long int m, long long int n, long long int nrhs, float* a, long long int lda, float* b, long long int ldb, float* s, float rcond, long long int* rank);
long long int LAPACKE_sgelsd_work(int matrix_layout, long long int m, long long int n, long long int nrhs, float* a, long long int lda, float* b, long long int ldb, float* s, float rcond, long long int* rank, float* work, long long int lwork, long long int* iwork);
long long int LAPACKE_dgelsd(int matrix_layout, long long int m, long long int n, long long int nrhs, double* a, long long int lda, double* b, long long int ldb, double* s, double rcond, long long int* rank);
long long int LAPACKE_dgelsd_work(int matrix_layout, long long int m, long long int n, long long int nrhs, double* a, long long int lda, double* b, long long int ldb, double* s, double rcond, long long int* rank, double* work, long long int lwork, long long int* iwork);
long long int LAPACKE_sposv(int matrix_layout, char uplo, long long int n, long long int nrhs, float* a, long long int lda, float* b, long long int ldb);
long long int LAPACKE_sposv_work(int matrix_layout, char uplo, long long int n, long long int nrhs, float* a, long long int lda, float* b, long long int ldb);
long long int LAPACKE_dposv(int matrix_layout, char uplo, long long int n, long long int nrhs, double* a, long long int lda, double* b, long long int ldb);
long long int LAPACKE_dposv_work(int matrix_layout, char uplo, long long int n, long long int nrhs, double* a, long long int lda, double* b, long long int ldb);
long long int LAPACKE_cposv(int matrix_layout, char uplo, long long int n, long long int nrhs, lapack_complex_float* a, long long int lda, lapack_complex_float* b, long long int ldb);
long long int LAPACKE_cposv_work(int matrix_layout, char uplo, long long int n, long long int nrhs, lapack_complex_float* a, long long int lda, lapack_complex_float* b, long long int ldb);
long long int LAPACKE_zposv(int matrix_layout, char uplo, long long int n, long long int nrhs, lapack_complex_double* a, long long int lda, lapack_complex_double* b, long long int ldb);
long long int LAPACKE_zposv_work(int matrix_layout, char uplo, long long int n, long long int nrhs, lapack_complex_double* a, long long int lda, lapack_complex_double* b, long long int ldb);
long long int LAPACKE_strtri(int matrix_layout, char uplo, char diag, long long int n, float* a, long long int lda);
long long int LAPACKE_strtri_work(int matrix_layout, char uplo, char diag, long long int n, float* a, long long int lda);
long long int LAPACKE_dtrtri(int matrix_layout, char uplo, char diag, long long int n, double* a, long long int lda);
long long int LAPACKE_dtrtri_work(int matrix_layout, char uplo, char diag, long long int n, double* a, long long int lda);
long long int LAPACKE_ctrtri(int matrix_layout, char uplo, char diag, long long int n, lapack_complex_float* a, long long int lda);
long long int LAPACKE_ctrtri_work(int matrix_layout, char uplo, char diag, long long int n, lapack_complex_float* a, long long int lda);
long long int LAPACKE_ztrtri(int matrix_layout, char uplo, char diag, long long int n, lapack_complex_double* a, long long int lda);
long long int LAPACKE_ztrtri_work(int matrix_layout, char uplo, char diag, long long int n, lapack_complex_double* a, long long int lda);
#ifdef __cplusplus
}
#endif
#undef ZYNUM_SUBSET_ASSERT
#undef ZYNUM_SUBSET_ALIGNOF
#endif
