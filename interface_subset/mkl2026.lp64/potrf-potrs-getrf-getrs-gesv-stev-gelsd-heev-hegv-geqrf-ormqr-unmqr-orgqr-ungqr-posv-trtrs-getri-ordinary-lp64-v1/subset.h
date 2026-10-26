// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later

/* Finite experimental declarations; not a complete MKL SDK. */
#ifndef ZYNUM_LAPACKE_POTRF_POTRS_GETRF_GETRS_GESV_STEV_GELSD_HEEV_HEGV_GEQRF_ORMQR_UNMQR_POSV_TRTRS_GETRI_LP64_V1_H
#define ZYNUM_LAPACKE_POTRF_POTRS_GETRF_GETRS_GESV_STEV_GELSD_HEEV_HEGV_GEQRF_ORMQR_UNMQR_POSV_TRTRS_GETRI_LP64_V1_H
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
#if defined(MKL_ILP64)
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
ZYNUM_SUBSET_ASSERT(sizeof(MKL_INT) == 4, "ordinary integer profile");
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
int LAPACKE_spotrf(int matrix_layout, char uplo, int n, float* a, int lda);
int LAPACKE_spotrf_work(int matrix_layout, char uplo, int n, float* a, int lda);
int LAPACKE_spotrs(int matrix_layout, char uplo, int n, int nrhs, const float* a, int lda, float* b, int ldb);
int LAPACKE_spotrs_work(int matrix_layout, char uplo, int n, int nrhs, const float* a, int lda, float* b, int ldb);
int LAPACKE_dpotrf(int matrix_layout, char uplo, int n, double* a, int lda);
int LAPACKE_dpotrf_work(int matrix_layout, char uplo, int n, double* a, int lda);
int LAPACKE_dpotrs(int matrix_layout, char uplo, int n, int nrhs, const double* a, int lda, double* b, int ldb);
int LAPACKE_dpotrs_work(int matrix_layout, char uplo, int n, int nrhs, const double* a, int lda, double* b, int ldb);
int LAPACKE_cpotrf(int matrix_layout, char uplo, int n, lapack_complex_float* a, int lda);
int LAPACKE_cpotrf_work(int matrix_layout, char uplo, int n, lapack_complex_float* a, int lda);
int LAPACKE_cpotrs(int matrix_layout, char uplo, int n, int nrhs, const lapack_complex_float* a, int lda, lapack_complex_float* b, int ldb);
int LAPACKE_cpotrs_work(int matrix_layout, char uplo, int n, int nrhs, const lapack_complex_float* a, int lda, lapack_complex_float* b, int ldb);
int LAPACKE_zpotrf(int matrix_layout, char uplo, int n, lapack_complex_double* a, int lda);
int LAPACKE_zpotrf_work(int matrix_layout, char uplo, int n, lapack_complex_double* a, int lda);
int LAPACKE_zpotrs(int matrix_layout, char uplo, int n, int nrhs, const lapack_complex_double* a, int lda, lapack_complex_double* b, int ldb);
int LAPACKE_zpotrs_work(int matrix_layout, char uplo, int n, int nrhs, const lapack_complex_double* a, int lda, lapack_complex_double* b, int ldb);
int LAPACKE_sgetrf(int matrix_layout, int m, int n, float* a, int lda, int* ipiv);
int LAPACKE_sgetrf_work(int matrix_layout, int m, int n, float* a, int lda, int* ipiv);
int LAPACKE_sgetrs(int matrix_layout, char trans, int n, int nrhs, const float* a, int lda, const int* ipiv, float* b, int ldb);
int LAPACKE_sgetrs_work(int matrix_layout, char trans, int n, int nrhs, const float* a, int lda, const int* ipiv, float* b, int ldb);
int LAPACKE_dgetrf(int matrix_layout, int m, int n, double* a, int lda, int* ipiv);
int LAPACKE_dgetrf_work(int matrix_layout, int m, int n, double* a, int lda, int* ipiv);
int LAPACKE_dgetrs(int matrix_layout, char trans, int n, int nrhs, const double* a, int lda, const int* ipiv, double* b, int ldb);
int LAPACKE_dgetrs_work(int matrix_layout, char trans, int n, int nrhs, const double* a, int lda, const int* ipiv, double* b, int ldb);
int LAPACKE_cgetrf(int matrix_layout, int m, int n, lapack_complex_float* a, int lda, int* ipiv);
int LAPACKE_cgetrf_work(int matrix_layout, int m, int n, lapack_complex_float* a, int lda, int* ipiv);
int LAPACKE_cgetrs(int matrix_layout, char trans, int n, int nrhs, const lapack_complex_float* a, int lda, const int* ipiv, lapack_complex_float* b, int ldb);
int LAPACKE_cgetrs_work(int matrix_layout, char trans, int n, int nrhs, const lapack_complex_float* a, int lda, const int* ipiv, lapack_complex_float* b, int ldb);
int LAPACKE_zgetrf(int matrix_layout, int m, int n, lapack_complex_double* a, int lda, int* ipiv);
int LAPACKE_zgetrf_work(int matrix_layout, int m, int n, lapack_complex_double* a, int lda, int* ipiv);
int LAPACKE_zgetrs(int matrix_layout, char trans, int n, int nrhs, const lapack_complex_double* a, int lda, const int* ipiv, lapack_complex_double* b, int ldb);
int LAPACKE_zgetrs_work(int matrix_layout, char trans, int n, int nrhs, const lapack_complex_double* a, int lda, const int* ipiv, lapack_complex_double* b, int ldb);
int LAPACKE_sgesv(int matrix_layout, int n, int nrhs, float* a, int lda, int* ipiv, float* b, int ldb);
int LAPACKE_sgesv_work(int matrix_layout, int n, int nrhs, float* a, int lda, int* ipiv, float* b, int ldb);
int LAPACKE_dgesv(int matrix_layout, int n, int nrhs, double* a, int lda, int* ipiv, double* b, int ldb);
int LAPACKE_dgesv_work(int matrix_layout, int n, int nrhs, double* a, int lda, int* ipiv, double* b, int ldb);
int LAPACKE_cgesv(int matrix_layout, int n, int nrhs, lapack_complex_float* a, int lda, int* ipiv, lapack_complex_float* b, int ldb);
int LAPACKE_cgesv_work(int matrix_layout, int n, int nrhs, lapack_complex_float* a, int lda, int* ipiv, lapack_complex_float* b, int ldb);
int LAPACKE_zgesv(int matrix_layout, int n, int nrhs, lapack_complex_double* a, int lda, int* ipiv, lapack_complex_double* b, int ldb);
int LAPACKE_zgesv_work(int matrix_layout, int n, int nrhs, lapack_complex_double* a, int lda, int* ipiv, lapack_complex_double* b, int ldb);
int LAPACKE_sstev(int matrix_layout, char jobz, int n, float* d, float* e, float* z, int ldz);
int LAPACKE_sstev_work(int matrix_layout, char jobz, int n, float* d, float* e, float* z, int ldz, float* work);
int LAPACKE_dstev(int matrix_layout, char jobz, int n, double* d, double* e, double* z, int ldz);
int LAPACKE_dstev_work(int matrix_layout, char jobz, int n, double* d, double* e, double* z, int ldz, double* work);
int LAPACKE_sgelsd(int matrix_layout, int m, int n, int nrhs, float* a, int lda, float* b, int ldb, float* s, float rcond, int* rank);
int LAPACKE_sgelsd_work(int matrix_layout, int m, int n, int nrhs, float* a, int lda, float* b, int ldb, float* s, float rcond, int* rank, float* work, int lwork, int* iwork);
int LAPACKE_dgelsd(int matrix_layout, int m, int n, int nrhs, double* a, int lda, double* b, int ldb, double* s, double rcond, int* rank);
int LAPACKE_dgelsd_work(int matrix_layout, int m, int n, int nrhs, double* a, int lda, double* b, int ldb, double* s, double rcond, int* rank, double* work, int lwork, int* iwork);
int LAPACKE_cheev(int matrix_layout, char jobz, char uplo, int n, lapack_complex_float* a, int lda, float* w);
int LAPACKE_cheev_work(int matrix_layout, char jobz, char uplo, int n, lapack_complex_float* a, int lda, float* w, lapack_complex_float* work, int lwork, float* rwork);
int LAPACKE_chegv(int matrix_layout, int itype, char jobz, char uplo, int n, lapack_complex_float* a, int lda, lapack_complex_float* b, int ldb, float* w);
int LAPACKE_chegv_work(int matrix_layout, int itype, char jobz, char uplo, int n, lapack_complex_float* a, int lda, lapack_complex_float* b, int ldb, float* w, lapack_complex_float* work, int lwork, float* rwork);
int LAPACKE_zheev(int matrix_layout, char jobz, char uplo, int n, lapack_complex_double* a, int lda, double* w);
int LAPACKE_zheev_work(int matrix_layout, char jobz, char uplo, int n, lapack_complex_double* a, int lda, double* w, lapack_complex_double* work, int lwork, double* rwork);
int LAPACKE_zhegv(int matrix_layout, int itype, char jobz, char uplo, int n, lapack_complex_double* a, int lda, lapack_complex_double* b, int ldb, double* w);
int LAPACKE_zhegv_work(int matrix_layout, int itype, char jobz, char uplo, int n, lapack_complex_double* a, int lda, lapack_complex_double* b, int ldb, double* w, lapack_complex_double* work, int lwork, double* rwork);
int LAPACKE_sgeqrf(int matrix_layout, int m, int n, float* a, int lda, float* tau);
int LAPACKE_sgeqrf_work(int matrix_layout, int m, int n, float* a, int lda, float* tau, float* work, int lwork);
int LAPACKE_sormqr(int matrix_layout, char side, char trans, int m, int n, int k, const float* a, int lda, const float* tau, float* c, int ldc);
int LAPACKE_sormqr_work(int matrix_layout, char side, char trans, int m, int n, int k, const float* a, int lda, const float* tau, float* c, int ldc, float* work, int lwork);
int LAPACKE_dgeqrf(int matrix_layout, int m, int n, double* a, int lda, double* tau);
int LAPACKE_dgeqrf_work(int matrix_layout, int m, int n, double* a, int lda, double* tau, double* work, int lwork);
int LAPACKE_dormqr(int matrix_layout, char side, char trans, int m, int n, int k, const double* a, int lda, const double* tau, double* c, int ldc);
int LAPACKE_dormqr_work(int matrix_layout, char side, char trans, int m, int n, int k, const double* a, int lda, const double* tau, double* c, int ldc, double* work, int lwork);
int LAPACKE_cgeqrf(int matrix_layout, int m, int n, lapack_complex_float* a, int lda, lapack_complex_float* tau);
int LAPACKE_cgeqrf_work(int matrix_layout, int m, int n, lapack_complex_float* a, int lda, lapack_complex_float* tau, lapack_complex_float* work, int lwork);
int LAPACKE_cunmqr(int matrix_layout, char side, char trans, int m, int n, int k, const lapack_complex_float* a, int lda, const lapack_complex_float* tau, lapack_complex_float* c, int ldc);
int LAPACKE_cunmqr_work(int matrix_layout, char side, char trans, int m, int n, int k, const lapack_complex_float* a, int lda, const lapack_complex_float* tau, lapack_complex_float* c, int ldc, lapack_complex_float* work, int lwork);
int LAPACKE_zgeqrf(int matrix_layout, int m, int n, lapack_complex_double* a, int lda, lapack_complex_double* tau);
int LAPACKE_zgeqrf_work(int matrix_layout, int m, int n, lapack_complex_double* a, int lda, lapack_complex_double* tau, lapack_complex_double* work, int lwork);
int LAPACKE_zunmqr(int matrix_layout, char side, char trans, int m, int n, int k, const lapack_complex_double* a, int lda, const lapack_complex_double* tau, lapack_complex_double* c, int ldc);
int LAPACKE_zunmqr_work(int matrix_layout, char side, char trans, int m, int n, int k, const lapack_complex_double* a, int lda, const lapack_complex_double* tau, lapack_complex_double* c, int ldc, lapack_complex_double* work, int lwork);
int LAPACKE_sorgqr(int matrix_layout, int m, int n, int k, float* a, int lda, const float* tau);
int LAPACKE_sorgqr_work(int matrix_layout, int m, int n, int k, float* a, int lda, const float* tau, float* work, int lwork);
int LAPACKE_dorgqr(int matrix_layout, int m, int n, int k, double* a, int lda, const double* tau);
int LAPACKE_dorgqr_work(int matrix_layout, int m, int n, int k, double* a, int lda, const double* tau, double* work, int lwork);
int LAPACKE_cungqr(int matrix_layout, int m, int n, int k, lapack_complex_float* a, int lda, const lapack_complex_float* tau);
int LAPACKE_cungqr_work(int matrix_layout, int m, int n, int k, lapack_complex_float* a, int lda, const lapack_complex_float* tau, lapack_complex_float* work, int lwork);
int LAPACKE_zungqr(int matrix_layout, int m, int n, int k, lapack_complex_double* a, int lda, const lapack_complex_double* tau);
int LAPACKE_zungqr_work(int matrix_layout, int m, int n, int k, lapack_complex_double* a, int lda, const lapack_complex_double* tau, lapack_complex_double* work, int lwork);
int LAPACKE_sposv(int matrix_layout, char uplo, int n, int nrhs, float* a, int lda, float* b, int ldb);
int LAPACKE_sposv_work(int matrix_layout, char uplo, int n, int nrhs, float* a, int lda, float* b, int ldb);
int LAPACKE_dposv(int matrix_layout, char uplo, int n, int nrhs, double* a, int lda, double* b, int ldb);
int LAPACKE_dposv_work(int matrix_layout, char uplo, int n, int nrhs, double* a, int lda, double* b, int ldb);
int LAPACKE_cposv(int matrix_layout, char uplo, int n, int nrhs, lapack_complex_float* a, int lda, lapack_complex_float* b, int ldb);
int LAPACKE_cposv_work(int matrix_layout, char uplo, int n, int nrhs, lapack_complex_float* a, int lda, lapack_complex_float* b, int ldb);
int LAPACKE_zposv(int matrix_layout, char uplo, int n, int nrhs, lapack_complex_double* a, int lda, lapack_complex_double* b, int ldb);
int LAPACKE_zposv_work(int matrix_layout, char uplo, int n, int nrhs, lapack_complex_double* a, int lda, lapack_complex_double* b, int ldb);
int LAPACKE_strtrs(int matrix_layout, char uplo, char trans, char diag, int n, int nrhs, const float* a, int lda, float* b, int ldb);
int LAPACKE_strtrs_work(int matrix_layout, char uplo, char trans, char diag, int n, int nrhs, const float* a, int lda, float* b, int ldb);
int LAPACKE_dtrtrs(int matrix_layout, char uplo, char trans, char diag, int n, int nrhs, const double* a, int lda, double* b, int ldb);
int LAPACKE_dtrtrs_work(int matrix_layout, char uplo, char trans, char diag, int n, int nrhs, const double* a, int lda, double* b, int ldb);
int LAPACKE_ctrtrs(int matrix_layout, char uplo, char trans, char diag, int n, int nrhs, const lapack_complex_float* a, int lda, lapack_complex_float* b, int ldb);
int LAPACKE_ctrtrs_work(int matrix_layout, char uplo, char trans, char diag, int n, int nrhs, const lapack_complex_float* a, int lda, lapack_complex_float* b, int ldb);
int LAPACKE_ztrtrs(int matrix_layout, char uplo, char trans, char diag, int n, int nrhs, const lapack_complex_double* a, int lda, lapack_complex_double* b, int ldb);
int LAPACKE_ztrtrs_work(int matrix_layout, char uplo, char trans, char diag, int n, int nrhs, const lapack_complex_double* a, int lda, lapack_complex_double* b, int ldb);
int LAPACKE_sgetri(int matrix_layout, int n, float* a, int lda, const int* ipiv);
int LAPACKE_sgetri_work(int matrix_layout, int n, float* a, int lda, const int* ipiv, float* work, int lwork);
int LAPACKE_dgetri(int matrix_layout, int n, double* a, int lda, const int* ipiv);
int LAPACKE_dgetri_work(int matrix_layout, int n, double* a, int lda, const int* ipiv, double* work, int lwork);
int LAPACKE_cgetri(int matrix_layout, int n, lapack_complex_float* a, int lda, const int* ipiv);
int LAPACKE_cgetri_work(int matrix_layout, int n, lapack_complex_float* a, int lda, const int* ipiv, lapack_complex_float* work, int lwork);
int LAPACKE_zgetri(int matrix_layout, int n, lapack_complex_double* a, int lda, const int* ipiv);
int LAPACKE_zgetri_work(int matrix_layout, int n, lapack_complex_double* a, int lda, const int* ipiv, lapack_complex_double* work, int lwork);
#ifdef __cplusplus
}
#endif
#undef ZYNUM_SUBSET_ASSERT
#undef ZYNUM_SUBSET_ALIGNOF
#endif
