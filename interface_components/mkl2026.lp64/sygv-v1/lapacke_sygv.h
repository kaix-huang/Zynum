/* Generated forwarding component; include standalone or original MKL first. */
#ifndef ZYNUM_LAPACKE_SYGV_LP64_COMPONENT_V1_H
#define ZYNUM_LAPACKE_SYGV_LP64_COMPONENT_V1_H
#include "lapacke_base.h"
#ifdef __cplusplus
extern "C" {
#endif
lapack_int LAPACKE_ssygv( int matrix_layout, lapack_int itype, char jobz, char uplo, lapack_int n, float* a, lapack_int lda, float* b, lapack_int ldb, float* w );
lapack_int LAPACKE_ssygv_work( int matrix_layout, lapack_int itype, char jobz, char uplo, lapack_int n, float* a, lapack_int lda, float* b, lapack_int ldb, float* w, float* work, lapack_int lwork );
lapack_int LAPACKE_dsygv( int matrix_layout, lapack_int itype, char jobz, char uplo, lapack_int n, double* a, lapack_int lda, double* b, lapack_int ldb, double* w );
lapack_int LAPACKE_dsygv_work( int matrix_layout, lapack_int itype, char jobz, char uplo, lapack_int n, double* a, lapack_int lda, double* b, lapack_int ldb, double* w, double* work, lapack_int lwork );
#ifdef __cplusplus
}
#endif
#endif
