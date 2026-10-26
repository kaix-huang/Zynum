// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
#ifndef ZYNUM_CHECKED_CHOLESKY_QUALIFICATION_CASES_H
#define ZYNUM_CHECKED_CHOLESKY_QUALIFICATION_CASES_H

/* Installed-header consumer only. The oracle uses scalar long-double complex
 * arithmetic, never BLAS, LAPACK, or the factorization under test. Before any
 * result is observed, the n=3 fixture budgets are fixed at 64*n*epsilon for
 * reconstruction/backward residual and 128*n*epsilon for forward error. These
 * modest, well-conditioned fixtures do not qualify ill-conditioned accuracy.
 * A scenario is one named setup/contract; assertions are counted separately.
 */
#include <float.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum cc_kind { CC_S, CC_D, CC_C, CC_Z, CC_CHECKED_C, CC_CHECKED_Z, CC_KINDS };
enum { CC_N = 3, CC_RHS = 2, CC_LDA = 5, CC_CAP = 20 };
typedef struct { long double re, im; } cc_z;
typedef struct {
    float s[CC_CAP];
    double d[CC_CAP];
    lapack_complex_float c[CC_CAP];
    lapack_complex_double z[CC_CAP];
    zynum_checked_complex_f32 checked_c[CC_CAP];
    zynum_checked_complex_f64 checked_z[CC_CAP];
} cc_buffer;
typedef struct { long double factor, residual, forward; } cc_metrics;
static unsigned long long cc_assertions, cc_scenarios, cc_start_assertions;
static char cc_case_name[192];
static int cc_is_numerical;
static cc_metrics cc_current, cc_maximum[CC_KINDS];

static void cc_check(int ok, const char *expression, int line) {
    ++cc_assertions;
    if (!ok) {
        fprintf(stderr, "FAIL case=%s line=%d assertion=%s\n", cc_case_name, line, expression);
        exit(1);
    }
}
#define CC_CHECK(expr) cc_check(!!(expr), #expr, __LINE__)
static const char *cc_kind_name(enum cc_kind k) {
    static const char *names[] = {"s", "d", "c", "z", "checked-complex-f32", "checked-complex-f64"};
    return names[k];
}
static int cc_complex(enum cc_kind k) { return k >= CC_C; }
static int cc_single(enum cc_kind k) { return k == CC_S || k == CC_C || k == CC_CHECKED_C; }
static long double cc_epsilon(enum cc_kind k) { return cc_single(k) ? FLT_EPSILON : DBL_EPSILON; }
static cc_z cc_make(long double re, long double im) { cc_z z = {re, im}; return z; }
static cc_z cc_add(cc_z a, cc_z b) { return cc_make(a.re+b.re, a.im+b.im); }
static cc_z cc_sub(cc_z a, cc_z b) { return cc_make(a.re-b.re, a.im-b.im); }
static cc_z cc_conj(cc_z a) { return cc_make(a.re, -a.im); }
static cc_z cc_mul(cc_z a, cc_z b) { return cc_make(a.re*b.re-a.im*b.im, a.re*b.im+a.im*b.re); }
static long double cc_abs(cc_z z) { return fabsl(z.re)+fabsl(z.im); }
static long double cc_max(long double a, long double b) { return a>b ? a : b; }
static size_t cc_size(enum cc_kind k) {
    switch (k) {
    case CC_S: return sizeof(float);
    case CC_D: return sizeof(double);
    case CC_C: return sizeof(lapack_complex_float);
    case CC_Z: return sizeof(lapack_complex_double);
    case CC_CHECKED_C: return sizeof(zynum_checked_complex_f32);
    case CC_CHECKED_Z: return sizeof(zynum_checked_complex_f64);
    default: abort();
    }
}
static void *cc_data(cc_buffer *b, enum cc_kind k) {
    switch (k) {
    case CC_S: return b->s;
    case CC_D: return b->d;
    case CC_C: return b->c;
    case CC_Z: return b->z;
    case CC_CHECKED_C: return b->checked_c;
    case CC_CHECKED_Z: return b->checked_z;
    default: abort();
    }
}
static void cc_store(cc_buffer *b, enum cc_kind k, int i, cc_z x) {
    switch (k) {
    case CC_S: b->s[i]=(float)x.re; break;
    case CC_D: b->d[i]=(double)x.re; break;
    case CC_C: b->c[i].real=(float)x.re; b->c[i].imag=(float)x.im; break;
    case CC_Z: b->z[i].real=(double)x.re; b->z[i].imag=(double)x.im; break;
    case CC_CHECKED_C: b->checked_c[i].re=(float)x.re; b->checked_c[i].im=(float)x.im; break;
    case CC_CHECKED_Z: b->checked_z[i].re=(double)x.re; b->checked_z[i].im=(double)x.im; break;
    default: abort();
    }
}
static cc_z cc_load(const cc_buffer *b, enum cc_kind k, int i) {
    switch (k) {
    case CC_S: return cc_make(b->s[i],0);
    case CC_D: return cc_make(b->d[i],0);
    case CC_C: return cc_make(b->c[i].real,b->c[i].imag);
    case CC_Z: return cc_make(b->z[i].real,b->z[i].imag);
    case CC_CHECKED_C: return cc_make(b->checked_c[i].re,b->checked_c[i].im);
    case CC_CHECKED_Z: return cc_make(b->checked_z[i].re,b->checked_z[i].im);
    default: abort();
    }
}
static void cc_copy(cc_buffer *to, cc_buffer *from, enum cc_kind k) {
    memcpy(cc_data(to,k),cc_data(from,k),CC_CAP*cc_size(k));
}
static void cc_unchanged(cc_buffer *a, cc_buffer *saved, enum cc_kind k) {
    CC_CHECK(memcmp(cc_data(a,k),cc_data(saved,k),CC_CAP*cc_size(k))==0);
}
static int cc_pos(int layout, int row, int col, int ld) {
    return layout==LAPACK_COL_MAJOR ? row+ld*col : ld*row+col;
}
static int cc_selected(int upper, int row, int col) { return upper ? row<=col : row>=col; }
static void cc_begin(const char *family, enum cc_kind k, int layout, int upper, const char *contract) {
    ++cc_scenarios;
    cc_start_assertions=cc_assertions;
    cc_is_numerical=0;
    cc_current.factor=cc_current.residual=cc_current.forward=0;
    snprintf(cc_case_name,sizeof cc_case_name,"%s/%s/%s/%c/%s",family,cc_kind_name(k),
             layout==LAPACK_ROW_MAJOR ? "row" : "column",upper ? 'U':'L',contract);
}
static void cc_end(enum cc_kind k) {
    printf("case=%s assertions=%llu",cc_case_name,cc_assertions-cc_start_assertions);
    if (cc_is_numerical) {
        printf(" factor_scaled=%.9Le residual_scaled=%.9Le forward_scaled=%.9Le",
               cc_current.factor,cc_current.residual,cc_current.forward);
        cc_maximum[k].factor=cc_max(cc_maximum[k].factor,cc_current.factor);
        cc_maximum[k].residual=cc_max(cc_maximum[k].residual,cc_current.residual);
        cc_maximum[k].forward=cc_max(cc_maximum[k].forward,cc_current.forward);
    }
    putchar('\n');
}

/* L has positive real diagonal and genuinely nonreal off-diagonal entries.
 * A=L*L^H is formed independently; real suites use the same real parts. */
static void cc_oracle(enum cc_kind k, cc_z a[CC_N][CC_N]) {
    cc_z l[CC_N][CC_N];
    for (int i=0;i<CC_N;++i) for (int j=0;j<CC_N;++j) l[i][j]=cc_make(0,0);
    l[0][0]=cc_make(2,0); l[1][1]=cc_make(3,0); l[2][2]=cc_make(4,0);
    l[1][0]=cc_make(1,cc_complex(k)?1:0);
    l[2][0]=cc_make(-1,cc_complex(k)?0.5L:0);
    l[2][1]=cc_make(2,cc_complex(k)?-1:0);
    for (int i=0;i<CC_N;++i) for (int j=0;j<CC_N;++j) {
        a[i][j]=cc_make(0,0);
        for (int p=0;p<CC_N;++p) a[i][j]=cc_add(a[i][j],cc_mul(l[i][p],cc_conj(l[j][p])));
    }
}
static void cc_seed_factor(cc_buffer *a, enum cc_kind k, int layout, int upper) {
    cc_z original[CC_N][CC_N]; cc_oracle(k,original);
    for (int i=0;i<CC_CAP;++i) cc_store(a,k,i,cc_make(-777.25L,119.5L));
    for (int i=0;i<CC_N;++i) for (int j=0;j<CC_N;++j) {
        cc_z x=cc_selected(upper,i,j) ? original[i][j] : cc_make(NAN,NAN);
        if (i==j && cc_complex(k)) x.im=-0.0L; /* Both signs of real diagonal zero are legal. */
        cc_store(a,k,cc_pos(layout,i,j,CC_LDA),x);
    }
}
static void cc_preserve_unused(cc_buffer *a, cc_buffer *before, enum cc_kind k,
                               int layout, int rows, int cols, int ld, int triangle, int upper) {
    unsigned char *now=(unsigned char *)cc_data(a,k), *old=(unsigned char *)cc_data(before,k);
    for (int p=0;p<CC_CAP;++p) {
        int used=0;
        for (int i=0;i<rows;++i) for (int j=0;j<cols;++j)
            if (p==cc_pos(layout,i,j,ld) && (!triangle || cc_selected(upper,i,j))) used=1;
        if (!used) CC_CHECK(memcmp(now+p*cc_size(k),old+p*cc_size(k),cc_size(k))==0);
    }
}
static long double cc_reconstruction(cc_buffer *factor, enum cc_kind k, int layout, int upper) {
    cc_z original[CC_N][CC_N]; cc_oracle(k,original);
    long double error=0, scale=0;
    for (int i=0;i<CC_N;++i) for (int j=0;j<CC_N;++j) {
        cc_z sum=cc_make(0,0);
        for (int p=0;p<=i && p<=j;++p) {
            cc_z li=cc_load(factor,k,cc_pos(layout,upper?p:i,upper?i:p,CC_LDA));
            cc_z lj=cc_load(factor,k,cc_pos(layout,upper?p:j,upper?j:p,CC_LDA));
            CC_CHECK(isfinite(li.re) && isfinite(li.im) && isfinite(lj.re) && isfinite(lj.im));
            if (upper) { li=cc_conj(li); lj=cc_conj(lj); }
            sum=cc_add(sum,cc_mul(li,cc_conj(lj)));
        }
        error=cc_max(error,cc_abs(cc_sub(sum,original[i][j])));
        scale=cc_max(scale,cc_abs(original[i][j]));
    }
    return error/(1+scale);
}
static void cc_rhs(cc_buffer *b, enum cc_kind k, int layout, int ld, int pass,
                   cc_z exact[CC_N][CC_RHS], cc_z input[CC_N][CC_RHS]) {
    cc_z a[CC_N][CC_N]; cc_oracle(k,a);
    for (int p=0;p<CC_CAP;++p) cc_store(b,k,p,cc_make(-991.25L,7.5L));
    for (int i=0;i<CC_N;++i) for (int j=0;j<CC_RHS;++j)
        exact[i][j]=cc_make(pass ? (i+1)*(j+1)+0.5L : (i+1)*(j?-0.5L:1),
                            cc_complex(k) ? (pass ? 0.5L*(j+1)-0.25L*i : 0.25L*(i-j)) : 0);
    for (int i=0;i<CC_N;++i) for (int j=0;j<CC_RHS;++j) {
        cc_z sum=cc_make(0,0);
        for (int p=0;p<CC_N;++p) sum=cc_add(sum,cc_mul(a[i][p],exact[p][j]));
        cc_store(b,k,cc_pos(layout,i,j,ld),sum);
        input[i][j]=cc_load(b,k,cc_pos(layout,i,j,ld));
    }
}
static void cc_solution_metrics(cc_buffer *b, enum cc_kind k, int layout, int ld,
                                cc_z exact[CC_N][CC_RHS], cc_z input[CC_N][CC_RHS]) {
    cc_z a[CC_N][CC_N]; cc_oracle(k,a);
    long double anorm=0, forward_error=0, exact_norm=0;
    for (int i=0;i<CC_N;++i) {
        long double row=0;
        for (int p=0;p<CC_N;++p) row+=cc_abs(a[i][p]);
        anorm=cc_max(anorm,row);
    }
    for (int j=0;j<CC_RHS;++j) {
        long double xnorm=0, bnorm=0, rnorm=0;
        for (int i=0;i<CC_N;++i) {
            cc_z x=cc_load(b,k,cc_pos(layout,i,j,ld));
            CC_CHECK(isfinite(x.re) && isfinite(x.im));
            xnorm=cc_max(xnorm,cc_abs(x)); bnorm=cc_max(bnorm,cc_abs(input[i][j]));
            forward_error=cc_max(forward_error,cc_abs(cc_sub(x,exact[i][j])));
            exact_norm=cc_max(exact_norm,cc_abs(exact[i][j]));
            cc_z product=cc_make(0,0);
            for (int p=0;p<CC_N;++p)
                product=cc_add(product,cc_mul(a[i][p],cc_load(b,k,cc_pos(layout,p,j,ld))));
            rnorm=cc_max(rnorm,cc_abs(cc_sub(product,input[i][j])));
        }
        CC_CHECK(anorm*xnorm+bnorm>0);
        cc_current.residual=cc_max(cc_current.residual,rnorm/(anorm*xnorm+bnorm));
    }
    cc_current.forward=cc_max(cc_current.forward,forward_error/(1+exact_norm));
}

static int cc_potrf(enum cc_kind k, int work, int layout, char uplo, int n, void *a, int lda) {
#define CC_RF(prefix,type) (work ? LAPACKE_##prefix##potrf_work(layout,uplo,n,(type *)a,lda) : LAPACKE_##prefix##potrf(layout,uplo,n,(type *)a,lda))
    switch (k) {
    case CC_S: return CC_RF(s,float);
    case CC_D: return CC_RF(d,double);
    case CC_C: return CC_RF(c,lapack_complex_float);
    case CC_Z: return CC_RF(z,lapack_complex_double);
    default: abort();
    }
#undef CC_RF
}
static int cc_potrs(enum cc_kind k, int work, int layout, char uplo, int n, int nrhs,
                     const void *a, int lda, void *b, int ldb) {
#define CC_RS(prefix,type) (work ? LAPACKE_##prefix##potrs_work(layout,uplo,n,nrhs,(const type *)a,lda,(type *)b,ldb) : LAPACKE_##prefix##potrs(layout,uplo,n,nrhs,(const type *)a,lda,(type *)b,ldb))
    switch (k) {
    case CC_S: return CC_RS(s,float);
    case CC_D: return CC_RS(d,double);
    case CC_C: return CC_RS(c,lapack_complex_float);
    case CC_Z: return CC_RS(z,lapack_complex_double);
    default: abort();
    }
#undef CC_RS
}
static zynum_checked_dense cc_dense(void *data, int64_t length, int64_t rows, int64_t cols, int64_t ld) {
    zynum_checked_dense d={{data,length},rows,cols,ld}; return d;
}
static zynum_checked_const_dense cc_const_dense(const void *data, int64_t length, int64_t rows, int64_t cols, int64_t ld) {
    zynum_checked_const_dense d={{data,length},rows,cols,ld}; return d;
}
static zynum_checked_result cc_factor(enum cc_kind k, zynum_checked_dense a, int32_t triangle) {
    switch (k) {
    case CC_S: return zynum_checked_cholesky_factor_f32(a,triangle);
    case CC_D: return zynum_checked_cholesky_factor_f64(a,triangle);
    case CC_CHECKED_C: return zynum_checked_cholesky_factor_complex_f32(a,triangle);
    case CC_CHECKED_Z: return zynum_checked_cholesky_factor_complex_f64(a,triangle);
    default: abort();
    }
}
static zynum_checked_result cc_solve(enum cc_kind k, zynum_checked_const_dense a, zynum_checked_dense b, int32_t triangle) {
    switch (k) {
    case CC_S: return zynum_checked_cholesky_solve_f32(a,b,triangle);
    case CC_D: return zynum_checked_cholesky_solve_f64(a,b,triangle);
    case CC_CHECKED_C: return zynum_checked_cholesky_solve_complex_f32(a,b,triangle);
    case CC_CHECKED_Z: return zynum_checked_cholesky_solve_complex_f64(a,b,triangle);
    default: abort();
    }
}
static void cc_result(zynum_checked_result r, int32_t status, int64_t info) {
    CC_CHECK(r.status==status); CC_CHECK(r.reserved==0); CC_CHECK(r.info==info);
}
static void cc_numerical(enum cc_kind k, int checked, int layout, int upper, int work) {
    cc_buffer a, before, factor, b, before_b;
    cc_z exact[CC_N][CC_RHS], input[CC_N][CC_RHS];
    const char uplo=upper?'U':'L';
    const int32_t triangle=upper?ZYNUM_CHECKED_UPPER:ZYNUM_CHECKED_LOWER;
    const int ldb=layout==LAPACK_COL_MAJOR ? 5 : 4;
    /* Checked lengths are exact physical spans in scalar/complex elements;
     * the larger backing arrays also expose writes beyond those spans. */
    const int factor_elements=(CC_N-1)*CC_LDA+CC_N;
    const int rhs_elements=(CC_RHS-1)*ldb+CC_N;
    cc_begin(checked?"checked":"LAPACKE",k,layout,upper,checked?"factor-reuse":work?"work-factor-reuse":"high-factor-reuse");
    cc_seed_factor(&a,k,layout,upper); cc_copy(&before,&a,k);
    if (checked) cc_result(cc_factor(k,cc_dense(cc_data(&a,k),factor_elements,CC_N,CC_N,CC_LDA),triangle),ZYNUM_CHECKED_OK,0);
    else CC_CHECK(cc_potrf(k,work,layout,uplo,CC_N,cc_data(&a,k),CC_LDA)==0);
    cc_preserve_unused(&a,&before,k,layout,CC_N,CC_N,CC_LDA,1,upper);
    cc_current.factor=cc_reconstruction(&a,k,layout,upper);
    cc_copy(&factor,&a,k);
    for (int pass=0;pass<2;++pass) {
        cc_rhs(&b,k,layout,ldb,pass,exact,input); cc_copy(&before_b,&b,k);
        if (checked) cc_result(cc_solve(k,cc_const_dense(cc_data(&a,k),factor_elements,CC_N,CC_N,CC_LDA),
                                        cc_dense(cc_data(&b,k),rhs_elements,CC_N,CC_RHS,ldb),triangle),ZYNUM_CHECKED_OK,0);
        else CC_CHECK(cc_potrs(k,work,layout,uplo,CC_N,CC_RHS,cc_data(&a,k),CC_LDA,cc_data(&b,k),ldb)==0);
        cc_unchanged(&a,&factor,k);
        cc_preserve_unused(&b,&before_b,k,layout,CC_N,CC_RHS,ldb,0,upper);
        cc_solution_metrics(&b,k,layout,ldb,exact,input);
    }
    cc_is_numerical=1;
    CC_CHECK(isfinite(cc_current.factor) && cc_current.factor<=64*CC_N*cc_epsilon(k));
    CC_CHECK(isfinite(cc_current.residual) && cc_current.residual<=64*CC_N*cc_epsilon(k));
    CC_CHECK(isfinite(cc_current.forward) && cc_current.forward<=128*CC_N*cc_epsilon(k));
    cc_end(k);
}

static void cc_ordinary_boundaries(enum cc_kind k, int layout, int upper, int work) {
    cc_buffer a, b, before_a, before_b;
    cc_z exact[CC_N][CC_RHS], input[CC_N][CC_RHS];
    const char uplo=upper?'U':'L';
    const char *entry=work?"work-boundaries":"high-boundaries";
    cc_begin("LAPACKE",k,layout,upper,entry);
    cc_seed_factor(&a,k,layout,upper); cc_rhs(&b,k,layout,5,0,exact,input);
    cc_copy(&before_a,&a,k); cc_copy(&before_b,&b,k);
    CC_CHECK(cc_potrf(k,work,-7,uplo,CC_N,cc_data(&a,k),CC_LDA)==-1);
    CC_CHECK(cc_potrs(k,work,-7,uplo,CC_N,CC_RHS,cc_data(&a,k),CC_LDA,cc_data(&b,k),5)==-1);
    /* High entry scans only selected triangles. Opposite NaNs are intentional.
     * No selected-NaN assertion depends on the process NaN-check setting. */
    CC_CHECK(cc_potrf(k,work,layout,'X',CC_N,cc_data(&a,k),CC_LDA)==-2);
    CC_CHECK(cc_potrs(k,work,layout,'X',CC_N,CC_RHS,cc_data(&a,k),CC_LDA,cc_data(&b,k),5)==-2);
    cc_unchanged(&a,&before_a,k); cc_unchanged(&b,&before_b,k);
    CC_CHECK(cc_potrf(k,work,layout,uplo,0,NULL,1)==0);
    CC_CHECK(cc_potrs(k,work,layout,uplo,0,0,NULL,1,NULL,1)==0);
    cc_end(k);
    cc_begin("LAPACKE",k,layout,upper,work?"work-leading-minor":"high-leading-minor");
    /* Leading 2x2 block [[4,4],[4,1]] has its first nonpositive pivot at 2. */
    cc_seed_factor(&a,k,layout,upper);
    cc_store(&a,k,cc_pos(layout,0,0,CC_LDA),cc_make(4,0));
    cc_store(&a,k,cc_pos(layout,1,1,CC_LDA),cc_make(1,0));
    cc_store(&a,k,cc_pos(layout,upper?0:1,upper?1:0,CC_LDA),cc_make(4,0));
    cc_copy(&before_a,&a,k);
    CC_CHECK(cc_potrf(k,work,layout,uplo,CC_N,cc_data(&a,k),CC_LDA)==2);
    CC_CHECK(cc_load(&a,k,cc_pos(layout,0,0,CC_LDA)).re==2);
    cc_preserve_unused(&a,&before_a,k,layout,CC_N,CC_N,CC_LDA,1,upper);
    cc_end(k);
}

static void cc_reject_factor(enum cc_kind k, int upper, const char *name,
                              zynum_checked_dense d, int32_t flag, int32_t expected, cc_buffer *a) {
    cc_buffer before;
    cc_begin("checked-boundary",k,LAPACK_COL_MAJOR,upper,name);
    cc_copy(&before,a,k);
    cc_result(cc_factor(k,d,flag),expected,0); cc_unchanged(a,&before,k); cc_end(k);
}
static void cc_reject_solve(enum cc_kind k, int upper, const char *name,
                            zynum_checked_const_dense f, zynum_checked_dense d, int32_t flag,
                            int32_t expected, cc_buffer *a, cc_buffer *b) {
    cc_buffer before_a, before_b;
    cc_begin("checked-boundary",k,LAPACK_COL_MAJOR,upper,name);
    cc_copy(&before_a,a,k); cc_copy(&before_b,b,k);
    cc_result(cc_solve(k,f,d,flag),expected,0);
    cc_unchanged(a,&before_a,k); cc_unchanged(b,&before_b,k); cc_end(k);
}
static void cc_checked_boundaries(enum cc_kind k, int upper) {
    cc_buffer a, b, good_factor, before_a, before_b;
    cc_z exact[CC_N][CC_RHS], input[CC_N][CC_RHS];
    const int32_t flag=upper?ZYNUM_CHECKED_UPPER:ZYNUM_CHECKED_LOWER;
    const int diag=cc_pos(LAPACK_COL_MAJOR,1,1,CC_LDA);
    const int offdiag=cc_pos(LAPACK_COL_MAJOR,upper?0:1,upper?1:0,CC_LDA);
    zynum_checked_dense d, bad, rhs;
    zynum_checked_const_dense f;
    cc_seed_factor(&a,k,LAPACK_COL_MAJOR,upper);
    d=cc_dense(cc_data(&a,k),CC_CAP,CC_N,CC_N,CC_LDA);
    cc_reject_factor(k,upper,"invalid-flag",d,99,ZYNUM_CHECKED_INVALID_FLAG,&a);
#define CC_BAD_FACTOR(label,field,value,status) do { bad=d; bad.field=(value); cc_reject_factor(k,upper,label,bad,flag,status,&a); } while (0)
    CC_BAD_FACTOR("negative-order",rows,-1,ZYNUM_CHECKED_INVALID_DIMENSION);
    CC_BAD_FACTOR("rectangular",cols,2,ZYNUM_CHECKED_DIMENSION_MISMATCH);
    CC_BAD_FACTOR("short-leading-dimension",ld,2,ZYNUM_CHECKED_INVALID_LEADING_DIMENSION);
    CC_BAD_FACTOR("negative-leading-dimension",ld,-1,ZYNUM_CHECKED_INVALID_LEADING_DIMENSION);
    CC_BAD_FACTOR("oversized-leading-dimension",ld,INT64_MAX,ZYNUM_CHECKED_INVALID_LEADING_DIMENSION);
    CC_BAD_FACTOR("short-element-capacity",storage.length,12,ZYNUM_CHECKED_BUFFER_TOO_SMALL);
    CC_BAD_FACTOR("negative-element-capacity",storage.length,-1,ZYNUM_CHECKED_INVALID_LENGTH);
    CC_BAD_FACTOR("byte-count-overflow",storage.length,INT64_MAX,ZYNUM_CHECKED_SIZE_OVERFLOW);
    CC_BAD_FACTOR("null-storage",storage.data,NULL,ZYNUM_CHECKED_NULL_POINTER);
    CC_BAD_FACTOR("misaligned-storage",storage.data,(void *)((unsigned char *)cc_data(&a,k)+1),ZYNUM_CHECKED_MISALIGNED_POINTER);
#undef CC_BAD_FACTOR
    cc_store(&a,k,diag,cc_make(11,0.25L));
    cc_reject_factor(k,upper,"nonreal-diagonal",d,flag,ZYNUM_CHECKED_CHOLESKY_INVALID_HERMITIAN_DIAGONAL,&a);
    cc_seed_factor(&a,k,LAPACK_COL_MAJOR,upper); cc_store(&a,k,offdiag,cc_make(NAN,0));
    cc_reject_factor(k,upper,"selected-real-nan",d,flag,ZYNUM_CHECKED_NONFINITE_INPUT,&a);
    cc_seed_factor(&a,k,LAPACK_COL_MAJOR,upper); cc_store(&a,k,offdiag,cc_make(1,INFINITY));
    cc_reject_factor(k,upper,"selected-imag-infinity",d,flag,ZYNUM_CHECKED_NONFINITE_INPUT,&a);
    cc_begin("checked-boundary",k,LAPACK_COL_MAJOR,upper,"leading-minor-info");
    cc_seed_factor(&a,k,LAPACK_COL_MAJOR,upper);
    cc_store(&a,k,0,cc_make(4,0)); cc_store(&a,k,diag,cc_make(1,0)); cc_store(&a,k,offdiag,cc_make(4,0));
    cc_copy(&before_a,&a,k);
    cc_result(cc_factor(k,d,flag),ZYNUM_CHECKED_NUMERICAL_FAILURE,2);
    CC_CHECK(cc_load(&a,k,0).re==2);
    cc_preserve_unused(&a,&before_a,k,LAPACK_COL_MAJOR,CC_N,CC_N,CC_LDA,1,upper); cc_end(k);

    cc_begin("checked-boundary",k,LAPACK_COL_MAJOR,upper,"solve-boundary-setup");
    cc_seed_factor(&a,k,LAPACK_COL_MAJOR,upper);
    cc_result(cc_factor(k,d,flag),ZYNUM_CHECKED_OK,0); cc_copy(&good_factor,&a,k); cc_end(k);
    cc_rhs(&b,k,LAPACK_COL_MAJOR,5,0,exact,input);
    f=cc_const_dense(cc_data(&a,k),CC_CAP,CC_N,CC_N,CC_LDA);
    rhs=cc_dense(cc_data(&b,k),CC_CAP,CC_N,CC_RHS,5);
    cc_reject_solve(k,upper,"solve-invalid-flag",f,rhs,99,ZYNUM_CHECKED_INVALID_FLAG,&a,&b);
    bad=cc_dense(cc_data(&a,k),CC_CAP,CC_N,CC_RHS,5);
    cc_reject_solve(k,upper,"exact-alias",f,bad,flag,ZYNUM_CHECKED_ALIASING_NOT_ALLOWED,&a,&b);
    bad.storage.data=(void *)((unsigned char *)cc_data(&a,k)+cc_size(k)); bad.storage.length=CC_CAP-1;
    cc_reject_solve(k,upper,"partial-alias",f,bad,flag,ZYNUM_CHECKED_ALIASING_NOT_ALLOWED,&a,&b);
    bad=rhs; bad.rows=2;
    cc_reject_solve(k,upper,"rhs-row-mismatch",f,bad,flag,ZYNUM_CHECKED_DIMENSION_MISMATCH,&a,&b);
    bad=rhs; bad.storage.length=7;
    cc_reject_solve(k,upper,"rhs-short-capacity",f,bad,flag,ZYNUM_CHECKED_BUFFER_TOO_SMALL,&a,&b);
    bad=rhs; bad.ld=2;
    cc_reject_solve(k,upper,"rhs-short-leading-dimension",f,bad,flag,ZYNUM_CHECKED_INVALID_LEADING_DIMENSION,&a,&b);
    cc_store(&a,k,diag,cc_make(3,0.25L));
    cc_reject_solve(k,upper,"solve-nonreal-diagonal",f,rhs,flag,ZYNUM_CHECKED_CHOLESKY_INVALID_HERMITIAN_DIAGONAL,&a,&b);
    cc_copy(&a,&good_factor,k); cc_store(&a,k,offdiag,cc_make(1,NAN));
    cc_reject_solve(k,upper,"solve-selected-imag-nan",f,rhs,flag,ZYNUM_CHECKED_NONFINITE_INPUT,&a,&b);
    cc_copy(&a,&good_factor,k); cc_store(&b,k,0,cc_make(INFINITY,0));
    cc_reject_solve(k,upper,"rhs-nonfinite",f,rhs,flag,ZYNUM_CHECKED_NONFINITE_INPUT,&a,&b);
    cc_rhs(&b,k,LAPACK_COL_MAJOR,5,0,exact,input); cc_copy(&a,&good_factor,k); cc_store(&a,k,diag,cc_make(0,0));
    cc_begin("checked-boundary",k,LAPACK_COL_MAJOR,upper,"solve-nonpositive-diagonal-info");
    cc_copy(&before_a,&a,k); cc_copy(&before_b,&b,k);
    cc_result(cc_solve(k,f,rhs,flag),ZYNUM_CHECKED_NUMERICAL_FAILURE,2);
    cc_unchanged(&a,&before_a,k); cc_unchanged(&b,&before_b,k); cc_end(k);
    cc_copy(&a,&good_factor,k);
    cc_begin("checked-boundary",k,LAPACK_COL_MAJOR,upper,"empty-null-and-zero-rhs");
    cc_result(cc_factor(k,cc_dense(NULL,0,0,0,0),flag),ZYNUM_CHECKED_OK,0);
    cc_result(cc_solve(k,cc_const_dense(NULL,0,0,0,0),cc_dense(NULL,0,0,0,0),flag),ZYNUM_CHECKED_OK,0);
    cc_copy(&before_a,&a,k);
    cc_result(cc_solve(k,f,cc_dense(NULL,0,CC_N,0,0),flag),ZYNUM_CHECKED_OK,0);
    cc_unchanged(&a,&before_a,k); cc_end(k);
    cc_reject_factor(k,upper,"empty-negative-leading-dimension",cc_dense(NULL,0,0,0,-1),flag,ZYNUM_CHECKED_INVALID_LEADING_DIMENSION,&a);
}

static int cc_run(const char *language, const char *include_order) {
    printf("Cholesky installed qualification language=%s include_order=%s n=%d nrhs=%d solves_per_factor=2\n",language,include_order,CC_N,CC_RHS);
    printf("fixed_budgets reconstruction/residual=64*n*epsilon forward=128*n*epsilon; complex_norm=abs(re)+abs(im)\n");
    cc_begin("headers",CC_CHECKED_Z,LAPACK_COL_MAJOR,0,"shared-types-and-status");
    CC_CHECK(sizeof(lapack_int)==4);
    CC_CHECK(sizeof(zynum_checked_complex_f32)==8 && sizeof(zynum_checked_complex_f64)==16);
    CC_CHECK(ZYNUM_CHECKED_CHOLESKY_INVALID_HERMITIAN_DIAGONAL==15);
    cc_end(CC_CHECKED_Z);
    for (int scalar=CC_S;scalar<=CC_Z;++scalar) for (int row=0;row<2;++row)
        for (int upper=0;upper<2;++upper) for (int work=0;work<2;++work) {
            enum cc_kind k=(enum cc_kind)scalar;
            int layout=row?LAPACK_ROW_MAJOR:LAPACK_COL_MAJOR;
            cc_numerical(k,0,layout,upper,work);
            cc_ordinary_boundaries(k,layout,upper,work);
        }
    const enum cc_kind checked_kinds[]={CC_S,CC_D,CC_CHECKED_C,CC_CHECKED_Z};
    for (unsigned i=0;i<sizeof checked_kinds/sizeof checked_kinds[0];++i)
        for (int upper=0;upper<2;++upper) cc_numerical(checked_kinds[i],1,LAPACK_COL_MAJOR,upper,0);
    for (int scalar=CC_CHECKED_C;scalar<=CC_CHECKED_Z;++scalar)
        for (int upper=0;upper<2;++upper) cc_checked_boundaries((enum cc_kind)scalar,upper);
    for (int scalar=0;scalar<CC_KINDS;++scalar)
        printf("maximum scalar=%s factor_scaled=%.9Le residual_scaled=%.9Le forward_scaled=%.9Le\n",
               cc_kind_name((enum cc_kind)scalar),cc_maximum[scalar].factor,cc_maximum[scalar].residual,cc_maximum[scalar].forward);
    printf("PASS language=%s scenarios=%llu assertions=%llu ordinary_numeric_scenarios=32 checked_numeric_scenarios=8; assertion_count_is_not_scenario_count\n",
           language,cc_scenarios,cc_assertions);
    return 0;
}
#undef CC_CHECK
#endif
