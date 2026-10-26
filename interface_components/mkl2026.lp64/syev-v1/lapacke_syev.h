/* Generated forwarding component; include standalone or original MKL first. */
#ifndef ZYNUM_LAPACKE_SYEV_LP64_COMPONENT_V1_H
#define ZYNUM_LAPACKE_SYEV_LP64_COMPONENT_V1_H
#include "lapacke_base.h"
#ifdef __cplusplus
extern "C" {
#endif
lapack_int LAPACKE_ssyev( int matrix_layout, char jobz, char uplo, lapack_int n, float* a, lapack_int lda, float* w );
lapack_int LAPACKE_ssyev_work( int matrix_layout, char jobz, char uplo, lapack_int n, float* a, lapack_int lda, float* w, float* work, lapack_int lwork );
lapack_int LAPACKE_dsyev( int matrix_layout, char jobz, char uplo, lapack_int n, double* a, lapack_int lda, double* w );
lapack_int LAPACKE_dsyev_work( int matrix_layout, char jobz, char uplo, lapack_int n, double* a, lapack_int lda, double* w, double* work, lapack_int lwork );
#ifdef __cplusplus
}
#endif
#endif
