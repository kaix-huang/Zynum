/* Generated forwarding component; include standalone or original MKL first. */
#ifndef ZYNUM_LAPACKE_GVD_LP64_COMPONENT_V1_H
#define ZYNUM_LAPACKE_GVD_LP64_COMPONENT_V1_H
#include "lapacke_base.h"
#ifdef __cplusplus
extern "C" {
#endif
lapack_int LAPACKE_ssygvd( int matrix_layout, lapack_int itype, char jobz, char uplo, lapack_int n, float* a, lapack_int lda, float* b, lapack_int ldb, float* w );
lapack_int LAPACKE_ssygvd_work( int matrix_layout, lapack_int itype, char jobz, char uplo, lapack_int n, float* a, lapack_int lda, float* b, lapack_int ldb, float* w, float* work, lapack_int lwork, lapack_int* iwork, lapack_int liwork );
lapack_int LAPACKE_dsygvd( int matrix_layout, lapack_int itype, char jobz, char uplo, lapack_int n, double* a, lapack_int lda, double* b, lapack_int ldb, double* w );
lapack_int LAPACKE_dsygvd_work( int matrix_layout, lapack_int itype, char jobz, char uplo, lapack_int n, double* a, lapack_int lda, double* b, lapack_int ldb, double* w, double* work, lapack_int lwork, lapack_int* iwork, lapack_int liwork );
lapack_int LAPACKE_chegvd( int matrix_layout, lapack_int itype, char jobz, char uplo, lapack_int n, lapack_complex_float* a, lapack_int lda, lapack_complex_float* b, lapack_int ldb, float* w );
lapack_int LAPACKE_chegvd_work( int matrix_layout, lapack_int itype, char jobz, char uplo, lapack_int n, lapack_complex_float* a, lapack_int lda, lapack_complex_float* b, lapack_int ldb, float* w, lapack_complex_float* work, lapack_int lwork, float* rwork, lapack_int lrwork, lapack_int* iwork, lapack_int liwork );
lapack_int LAPACKE_zhegvd( int matrix_layout, lapack_int itype, char jobz, char uplo, lapack_int n, lapack_complex_double* a, lapack_int lda, lapack_complex_double* b, lapack_int ldb, double* w );
lapack_int LAPACKE_zhegvd_work( int matrix_layout, lapack_int itype, char jobz, char uplo, lapack_int n, lapack_complex_double* a, lapack_int lda, lapack_complex_double* b, lapack_int ldb, double* w, lapack_complex_double* work, lapack_int lwork, double* rwork, lapack_int lrwork, lapack_int* iwork, lapack_int liwork );
#ifdef __cplusplus
}
#endif
#endif
