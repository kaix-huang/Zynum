/* Generated forwarding component; include standalone or original MKL first. */
#ifndef ZYNUM_LAPACKE_POTRI_ILP64_COMPONENT_V1_H
#define ZYNUM_LAPACKE_POTRI_ILP64_COMPONENT_V1_H
#include "lapacke_base.h"
#ifdef __cplusplus
extern "C" {
#endif
lapack_int LAPACKE_spotri( int matrix_layout, char uplo, lapack_int n, float* a, lapack_int lda );
lapack_int LAPACKE_spotri_work( int matrix_layout, char uplo, lapack_int n, float* a, lapack_int lda );
lapack_int LAPACKE_dpotri( int matrix_layout, char uplo, lapack_int n, double* a, lapack_int lda );
lapack_int LAPACKE_dpotri_work( int matrix_layout, char uplo, lapack_int n, double* a, lapack_int lda );
lapack_int LAPACKE_cpotri( int matrix_layout, char uplo, lapack_int n, lapack_complex_float* a, lapack_int lda );
lapack_int LAPACKE_cpotri_work( int matrix_layout, char uplo, lapack_int n, lapack_complex_float* a, lapack_int lda );
lapack_int LAPACKE_zpotri( int matrix_layout, char uplo, lapack_int n, lapack_complex_double* a, lapack_int lda );
lapack_int LAPACKE_zpotri_work( int matrix_layout, char uplo, lapack_int n, lapack_complex_double* a, lapack_int lda );
#ifdef __cplusplus
}
#endif
#endif
