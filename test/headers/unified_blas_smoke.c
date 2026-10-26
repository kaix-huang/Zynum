// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
#include <cblas.h>
#include <zynum/blas/blas.h>
#include <lapacke.h>
#include <zynum/checked/sparse.h>
#include <zynum/fft/fftw3_subset.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK(x) do { if (!(x)) { fprintf(stderr, "failed line %d\n", __LINE__); return 1; } } while (0)
#define NEAR(x,y) (fabs((x)-(y)) < 1e-11)

static int lapack_errors = 0;
static lapack_int lapack_error_info = 0;
#ifdef __cplusplus
extern "C"
#endif
void LAPACKE_xerbla(const char *name, lapack_int info) {
    (void)name;
    ++lapack_errors;
    lapack_error_info = info;
}

int main(void) {
    double a[6] = {1,2,3,4,5,6}, b[6] = {2,1,0,-1,3,2};
    double c[6] = {0,0,91,0,0,92};
    cblas_dgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,2,2,3,1,a,3,b,2,0,c,3);
    CHECK(NEAR(c[0],11) && NEAR(c[1],5) && NEAR(c[3],26) && NEAR(c[4],11));
    CHECK(c[2] == 91 && c[5] == 92);
    double ac[6] = {1,4,2,5,3,6}, bc[6] = {2,0,3,1,-1,2}, cc[6] = {0,0,93,0,0,94};
    cblas_dgemm(CblasColMajor,CblasNoTrans,CblasNoTrans,2,2,3,1,ac,2,bc,3,0,cc,3);
    CHECK(NEAR(cc[0],11) && NEAR(cc[1],26) && NEAR(cc[3],5) && NEAR(cc[4],11));
    CHECK(cc[2] == 93 && cc[5] == 94);
    double x[2] = {2,3}, y[5] = {0,95,0,96,0};
    cblas_dgemv(CblasColMajor,CblasTrans,2,3,1,ac,2,x,1,0,y,2);
    CHECK(NEAR(y[0],14) && NEAR(y[2],19) && NEAR(y[4],24));
    CHECK(y[1] == 95 && y[3] == 96);

    zynum_blas_complex_double z[6] = {{1,2},{-1,1},{2,-1},{3,.5},{.25,-2},{-2,3}};
    zynum_blas_complex_double gram[4], one = {1,0}, zero = {0,0};
    memset(gram,0,sizeof(gram));
    cblas_zgemm(CblasRowMajor,CblasConjTrans,CblasNoTrans,2,2,3,&one,z,2,z,2,&zero,gram,2);
    CHECK(NEAR(gram[0].real,14.0625) && NEAR(gram[0].imag,0));
    CHECK(NEAR(gram[1].real,0) && NEAR(gram[1].imag,3.75));
    CHECK(NEAR(gram[2].real,0) && NEAR(gram[2].imag,-3.75));
    CHECK(NEAR(gram[3].real,24.25) && NEAR(gram[3].imag,0));
    lapack_complex_double factor[4];
    memcpy(factor,gram,sizeof(factor));
    CHECK(LAPACKE_zpotrf(LAPACK_ROW_MAJOR,'U',2,factor,2) == 0);
    CHECK(NEAR(factor[0].real,3.75) && NEAR(factor[1].real,0) && NEAR(factor[1].imag,1));
    CHECK(NEAR(factor[3].real,sqrt(23.25)));
    lapack_complex_double saved_factor[4];
    memcpy(saved_factor,factor,sizeof(factor));
    CHECK(LAPACKE_zpotrf(999,'U',2,factor,2) == -1);
    CHECK(lapack_errors == 1 && lapack_error_info == -1);
    CHECK(memcmp(saved_factor,factor,sizeof(factor)) == 0);

    const char trans = 'N';
    const zynum_blas_int m = 2, n = 2, k = 3, lda = 2, ldb = 3, ldc = 2;
    const double alpha = 1, beta = 0;
    double f[4] = {0,0,0,0};
    dgemm_(&trans,&trans,&m,&n,&k,&alpha,ac,&lda,bc,&ldb,&beta,f,&ldc);
    CHECK(NEAR(f[0],11) && NEAR(f[1],26) && NEAR(f[2],5) && NEAR(f[3],11));
    float sx[2] = {1,2}, sy[2] = {3,4};
    cblas_saxpy(2,2,sx,1,sy,1);
    CHECK(sy[0] == 5 && sy[1] == 8);
    zynum_blas_complex_float cx[1] = {{1,2}}, cy[1] = {{3,-1}}, ca = {2,-1};
    cblas_caxpy(1,&ca,cx,1,cy,1);
    CHECK(cy[0].real == 7 && cy[0].imag == 2);
    zynum_blas_complex_double zy[1] = {{3,-1}}, za = {2,-1}, zx[1] = {{1,2}};
    cblas_zaxpy(1,&za,zx,1,zy,1);
    CHECK(zy[0].real == 7 && zy[0].imag == 2);

    double saved[6];
    memcpy(saved,c,sizeof(c));
    cblas_dgemm((CBLAS_ORDER)999,CblasNoTrans,CblasNoTrans,2,2,3,1,a,3,b,2,0,c,3);
    CHECK(memcmp(saved,c,sizeof(c)) == 0);
    cblas_dgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,-1,2,3,1,a,3,b,2,0,c,3);
    CHECK(memcmp(saved,c,sizeof(c)) == 0);

    // Exercise the real BLAS thread-pool path, retaining an FFT plan across
    // drain/reset. This is a correctness/lifetime test, without timing.
    const int rows = 256, columns = 256, inner = 512;
    double *large_a = (double*)malloc((size_t)rows*inner*sizeof(double));
    double *large_b = (double*)malloc((size_t)inner*columns*sizeof(double));
    double *large_c = (double*)malloc((size_t)rows*columns*sizeof(double));
    CHECK(large_a && large_b && large_c);
    for (size_t i=0; i<(size_t)rows*inner; ++i) large_a[i]=1;
    for (size_t i=0; i<(size_t)inner*columns; ++i) large_b[i]=1;
    double di[4] = {1,.5,.25,0}, dout[4];
    fftw_plan plan = fftw_plan_r2r_1d(4,di,dout,FFTW_REDFT00,FFTW_ESTIMATE);
    CHECK(plan);
    cblas_dgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,rows,columns,inner,1,large_a,inner,large_b,columns,0,large_c,columns);
    for (size_t i=0; i<(size_t)rows*columns; ++i) CHECK(large_c[i] == inner);
    zynum_blas_shutdown();
    fftw_execute(plan);
    CHECK(NEAR(dout[0],2.5) && NEAR(dout[1],1.25) && NEAR(dout[2],.25) && NEAR(dout[3],.5));
    double solve_a[4] = {3,1,1,2}, solve_b[2] = {9,8};
    lapack_int pivots[2];
    CHECK(LAPACKE_dgesv(LAPACK_COL_MAJOR,2,1,solve_a,2,pivots,solve_b,2) == 0);
    CHECK(NEAR(solve_b[0],2) && NEAR(solve_b[1],3));
    cblas_dgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,rows,columns,inner,1,large_a,inner,large_b,columns,0,large_c,columns);
    for (size_t i=0; i<(size_t)rows*columns; ++i) CHECK(large_c[i] == inner);
    fftw_destroy_plan(plan);
    zynum_blas_shutdown_();
    free(large_a); free(large_b); free(large_c);
    puts("unified BLAS numerical, boundaries and cross-module lifetimes passed");
    return 0;
}
