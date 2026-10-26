// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
#include <zynum/checked/checked.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
static unsigned checks=0;
#define CHECK(x) do {++checks;if(!(x)){fprintf(stderr,"failed %s:%d: %s\n",__FILE__,__LINE__,#x);exit(1);}}while(0)
#define OK(expr) do {zynum_checked_result r_=(expr);CHECK(r_.status==ZYNUM_CHECKED_OK && r_.reserved==0 && r_.info==0);}while(0)
#define FAIL(expr,code) do {zynum_checked_result r_=(expr);CHECK(r_.status==(code) && r_.reserved==0 && r_.info==0);}while(0)
#define NEAR(x,y) CHECK(fabs((double)(x)-(double)(y))<=TOL*(1+fabs((double)(y))))
#define CAT_(a,b) a##b
#define CAT(a,b) CAT_(a,b)
#define T float
#define TOL 0.00001
#define FN(x) CAT(f32_,x)
#define FACTOR zynum_checked_cholesky_factor_f32
#define SOLVE zynum_checked_cholesky_solve_f32
#include "checked_dense_cases.h"
#undef FN
#define FN(x) CAT(f32_u32_,x)
#define I uint32_t
#define SPMV zynum_checked_csr_spmv_f32_u32
#include "checked_sparse_cases.h"
#undef FN
#undef I
#undef SPMV
#define FN(x) CAT(f32_u64_,x)
#define I uint64_t
#define SPMV zynum_checked_csr_spmv_f32_u64
#include "checked_sparse_cases.h"
#undef FN
#undef I
#undef SPMV
#undef FACTOR
#undef SOLVE
#undef T
#undef TOL
#define T double
#define TOL 0.000000000001
#define FN(x) CAT(f64_,x)
#define FACTOR zynum_checked_cholesky_factor_f64
#define SOLVE zynum_checked_cholesky_solve_f64
#include "checked_dense_cases.h"
#undef FN
#define FN(x) CAT(f64_u32_,x)
#define I uint32_t
#define SPMV zynum_checked_csr_spmv_f64_u32
#include "checked_sparse_cases.h"
#undef FN
#undef I
#undef SPMV
#define FN(x) CAT(f64_u64_,x)
#define I uint64_t
#define SPMV zynum_checked_csr_spmv_f64_u64
#include "checked_sparse_cases.h"
int main(void) {
    f32_dense();f64_dense();f32_u32_sparse();f32_u64_sparse();f64_u32_sparse();f64_u64_sparse();
#ifdef __cplusplus
    const char *language="C++";
#else
    const char *language="C";
#endif
    printf("%s checked ABI: %u assertions passed; all 8 exports called\n",language,checks);
    return 0;
}
