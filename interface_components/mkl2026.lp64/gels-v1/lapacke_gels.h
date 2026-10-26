/* Generated forwarding component; include standalone or original MKL first. */
#ifndef ZYNUM_LAPACKE_GELS_LP64_COMPONENT_V1_H
#define ZYNUM_LAPACKE_GELS_LP64_COMPONENT_V1_H
#include "lapacke_base.h"
#ifdef __cplusplus
extern "C" {
#endif
lapack_int LAPACKE_sgels( int matrix_layout, char trans, lapack_int m, lapack_int n, lapack_int nrhs, float* a, lapack_int lda, float* b, lapack_int ldb );
lapack_int LAPACKE_sgels_work( int matrix_layout, char trans, lapack_int m, lapack_int n, lapack_int nrhs, float* a, lapack_int lda, float* b, lapack_int ldb, float* work, lapack_int lwork );
lapack_int LAPACKE_dgels( int matrix_layout, char trans, lapack_int m, lapack_int n, lapack_int nrhs, double* a, lapack_int lda, double* b, lapack_int ldb );
lapack_int LAPACKE_dgels_work( int matrix_layout, char trans, lapack_int m, lapack_int n, lapack_int nrhs, double* a, lapack_int lda, double* b, lapack_int ldb, double* work, lapack_int lwork );
lapack_int LAPACKE_cgels( int matrix_layout, char trans, lapack_int m, lapack_int n, lapack_int nrhs, lapack_complex_float* a, lapack_int lda, lapack_complex_float* b, lapack_int ldb );
lapack_int LAPACKE_cgels_work( int matrix_layout, char trans, lapack_int m, lapack_int n, lapack_int nrhs, lapack_complex_float* a, lapack_int lda, lapack_complex_float* b, lapack_int ldb, lapack_complex_float* work, lapack_int lwork );
lapack_int LAPACKE_zgels( int matrix_layout, char trans, lapack_int m, lapack_int n, lapack_int nrhs, lapack_complex_double* a, lapack_int lda, lapack_complex_double* b, lapack_int ldb );
lapack_int LAPACKE_zgels_work( int matrix_layout, char trans, lapack_int m, lapack_int n, lapack_int nrhs, lapack_complex_double* a, lapack_int lda, lapack_complex_double* b, lapack_int ldb, lapack_complex_double* work, lapack_int lwork );
#ifdef __cplusplus
}
#endif
#endif
