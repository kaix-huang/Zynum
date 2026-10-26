/* Generated forwarding component; include standalone or original MKL first. */
#ifndef ZYNUM_LAPACKE_GELSS_LP64_COMPONENT_V1_H
#define ZYNUM_LAPACKE_GELSS_LP64_COMPONENT_V1_H
#include "lapacke_base.h"
#ifdef __cplusplus
extern "C" {
#endif
lapack_int LAPACKE_sgelss( int matrix_layout, lapack_int m, lapack_int n, lapack_int nrhs, float* a, lapack_int lda, float* b, lapack_int ldb, float* s, float rcond, lapack_int* rank );
lapack_int LAPACKE_sgelss_work( int matrix_layout, lapack_int m, lapack_int n, lapack_int nrhs, float* a, lapack_int lda, float* b, lapack_int ldb, float* s, float rcond, lapack_int* rank, float* work, lapack_int lwork );
lapack_int LAPACKE_dgelss( int matrix_layout, lapack_int m, lapack_int n, lapack_int nrhs, double* a, lapack_int lda, double* b, lapack_int ldb, double* s, double rcond, lapack_int* rank );
lapack_int LAPACKE_dgelss_work( int matrix_layout, lapack_int m, lapack_int n, lapack_int nrhs, double* a, lapack_int lda, double* b, lapack_int ldb, double* s, double rcond, lapack_int* rank, double* work, lapack_int lwork );
lapack_int LAPACKE_cgelss( int matrix_layout, lapack_int m, lapack_int n, lapack_int nrhs, lapack_complex_float* a, lapack_int lda, lapack_complex_float* b, lapack_int ldb, float* s, float rcond, lapack_int* rank );
lapack_int LAPACKE_cgelss_work( int matrix_layout, lapack_int m, lapack_int n, lapack_int nrhs, lapack_complex_float* a, lapack_int lda, lapack_complex_float* b, lapack_int ldb, float* s, float rcond, lapack_int* rank, lapack_complex_float* work, lapack_int lwork, float* rwork );
lapack_int LAPACKE_zgelss( int matrix_layout, lapack_int m, lapack_int n, lapack_int nrhs, lapack_complex_double* a, lapack_int lda, lapack_complex_double* b, lapack_int ldb, double* s, double rcond, lapack_int* rank );
lapack_int LAPACKE_zgelss_work( int matrix_layout, lapack_int m, lapack_int n, lapack_int nrhs, lapack_complex_double* a, lapack_int lda, lapack_complex_double* b, lapack_int ldb, double* s, double rcond, lapack_int* rank, lapack_complex_double* work, lapack_int lwork, double* rwork );
#ifdef __cplusplus
}
#endif
#endif
