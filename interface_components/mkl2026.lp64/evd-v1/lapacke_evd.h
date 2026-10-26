/* Generated forwarding component; include standalone or original MKL first. */
#ifndef ZYNUM_LAPACKE_EVD_LP64_COMPONENT_V1_H
#define ZYNUM_LAPACKE_EVD_LP64_COMPONENT_V1_H
#include "lapacke_base.h"
#ifdef __cplusplus
extern "C" {
#endif
lapack_int LAPACKE_ssyevd( int matrix_layout, char jobz, char uplo, lapack_int n, float* a, lapack_int lda, float* w );
lapack_int LAPACKE_ssyevd_work( int matrix_layout, char jobz, char uplo, lapack_int n, float* a, lapack_int lda, float* w, float* work, lapack_int lwork, lapack_int* iwork, lapack_int liwork );
lapack_int LAPACKE_dsyevd( int matrix_layout, char jobz, char uplo, lapack_int n, double* a, lapack_int lda, double* w );
lapack_int LAPACKE_dsyevd_work( int matrix_layout, char jobz, char uplo, lapack_int n, double* a, lapack_int lda, double* w, double* work, lapack_int lwork, lapack_int* iwork, lapack_int liwork );
lapack_int LAPACKE_cheevd( int matrix_layout, char jobz, char uplo, lapack_int n, lapack_complex_float* a, lapack_int lda, float* w );
lapack_int LAPACKE_cheevd_work( int matrix_layout, char jobz, char uplo, lapack_int n, lapack_complex_float* a, lapack_int lda, float* w, lapack_complex_float* work, lapack_int lwork, float* rwork, lapack_int lrwork, lapack_int* iwork, lapack_int liwork );
lapack_int LAPACKE_zheevd( int matrix_layout, char jobz, char uplo, lapack_int n, lapack_complex_double* a, lapack_int lda, double* w );
lapack_int LAPACKE_zheevd_work( int matrix_layout, char jobz, char uplo, lapack_int n, lapack_complex_double* a, lapack_int lda, double* w, lapack_complex_double* work, lapack_int lwork, double* rwork, lapack_int lrwork, lapack_int* iwork, lapack_int liwork );
#ifdef __cplusplus
}
#endif
#endif
