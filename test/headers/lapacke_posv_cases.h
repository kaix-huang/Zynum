/* Copyright (C) 2026 Zynum contributors
 * SPDX-License-Identifier: LGPL-3.0-or-later */
#ifndef ZYNUM_POSV_CASES_H
#define ZYNUM_POSV_CASES_H
#include <lapacke.h>
#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
static int hooks=0;
static lapack_int last_info=0;
static char last_name[32];
void LAPACKE_xerbla(const char* name,lapack_int info) { ++hooks; last_info=info; snprintf(last_name,sizeof(last_name),"%s",name); }
#ifndef ZYNUM_POSV_LINKED_BLAS_XERBLA
void xerbla_(const char* name,const lapack_int* info) { ++hooks; last_info=*info; snprintf(last_name,sizeof(last_name),"%.5s",name); }
#endif
#ifdef __cplusplus
}
#endif
static size_t assertions=0,scenarios=0;
static void insist(int yes,const char* text,int line) { ++assertions; if(!yes){fprintf(stderr,"POSV assertion line %d: %s\n",line,text);exit(1);} }
#define REQUIRE(x) insist((x),#x,__LINE__)
typedef struct { double r,i; } Num;
typedef struct { float s[400]; double d[400]; MKL_Complex8 c[400]; MKL_Complex16 z[400]; } Store;
static Num num(double r,double i) { Num x={r,i};return x; }
static Num add(Num a,Num b) {return num(a.r+b.r,a.i+b.i);}
static Num product(Num a,Num b) {return num(a.r*b.r-a.i*b.i,a.r*b.i+a.i*b.r);}
static Num conjugate(Num a) {return num(a.r,-a.i);}
static void* storage(Store* s,int kind) {switch(kind){case 0:return s->s;case 1:return s->d;case 2:return s->c;default:return s->z;}}
static size_t element_size(int kind) {switch(kind){case 0:return sizeof(float);case 1:return sizeof(double);case 2:return sizeof(MKL_Complex8);default:return sizeof(MKL_Complex16);}}
static Num get(const Store* s,int kind,int index) {switch(kind){case 0:return num(s->s[index],0);case 1:return num(s->d[index],0);case 2:return num(s->c[index].real,s->c[index].imag);default:return num(s->z[index].real,s->z[index].imag);}}
static void put(Store* s,int kind,int index,Num value) {switch(kind){case 0:s->s[index]=(float)value.r;break;case 1:s->d[index]=value.r;break;case 2:s->c[index].real=(float)value.r;s->c[index].imag=(float)value.i;break;default:s->z[index].real=value.r;s->z[index].imag=value.i;}}
static int pos(int layout,int i,int j,int ld){return layout==101?i*ld+j:i+j*ld;}
static lapack_int invoke(int kind,int work,int layout,char uplo,lapack_int n,lapack_int nrhs,void* a,lapack_int lda,void* b,lapack_int ldb) {
    switch(kind){
#define ZYNUM_POSV_CALL(K,T,P) case K: return (work?LAPACKE_##P##posv_work:LAPACKE_##P##posv)(layout,uplo,n,nrhs,(T*)a,lda,(T*)b,ldb)
    ZYNUM_POSV_CALL(0,float,s); ZYNUM_POSV_CALL(1,double,d); ZYNUM_POSV_CALL(2,MKL_Complex8,c); ZYNUM_POSV_CALL(3,MKL_Complex16,z);
#undef ZYNUM_POSV_CALL
    } return -9999;
}
static lapack_int reuse(int kind,int layout,char uplo,lapack_int n,lapack_int nrhs,void* a,lapack_int lda,void* b,lapack_int ldb) {
    switch(kind){
#define ZYNUM_POTRS_CALL(K,T,P) case K: return LAPACKE_##P##potrs_work(layout,uplo,n,nrhs,(const T*)a,lda,(T*)b,ldb)
    ZYNUM_POTRS_CALL(0,float,s); ZYNUM_POTRS_CALL(1,double,d); ZYNUM_POTRS_CALL(2,MKL_Complex8,c); ZYNUM_POTRS_CALL(3,MKL_Complex16,z);
#undef ZYNUM_POTRS_CALL
    } return -9999;
}
static void close_num(Num actual,Num expected,double tolerance) { REQUIRE(isfinite(actual.r)&&isfinite(actual.i)); REQUIRE(hypot(actual.r-expected.r,actual.i-expected.i)<=tolerance); }
static void numeric(int kind,int work,int layout,int upper,int n,int nrhs) {
    Store a,b,original_a,original_b,factor;
    Num matrix[289],lower[289],solution[51];
    int lda=n+2,ldb=layout==101?nrhs+2:n+2;
    int ai[400]={0},bi[400]={0};
    const double imag=kind>=2?1:0,eps=(kind==0||kind==2)?FLT_EPSILON:DBL_EPSILON;
    const Num sentinel=num(-777,37*imag);
    ++scenarios;
    for(int i=0;i<400;++i){put(&a,kind,i,sentinel);put(&b,kind,i,sentinel);}
    for(int i=0;i<n;++i)for(int j=0;j<n;++j)lower[i*n+j]=i<j?num(0,0):i==j?num(2+i*0.1,0):num(((i+2*j)%7-3)*0.1,((2*i+j)%5-2)*0.12*imag);
    for(int i=0;i<n;++i)for(int j=0;j<n;++j){
        Num v=num(0,0);for(int k=0;k<n;++k)v=add(v,product(lower[i*n+k],conjugate(lower[j*n+k])));
        matrix[i*n+j]=v;
        if(upper?i<=j:i>=j){put(&a,kind,pos(layout,i,j,lda),v);ai[pos(layout,i,j,lda)]=1;}
    }
    for(int i=0;i<n;++i)for(int j=0;j<nrhs;++j)solution[i*nrhs+j]=num(0.4+i*0.3-j*0.2,(i*0.2+j*0.15)*imag);
    for(int i=0;i<n;++i)for(int j=0;j<nrhs;++j){
        Num v=num(0,0);for(int k=0;k<n;++k)v=add(v,product(matrix[i*n+k],solution[k*nrhs+j]));
        put(&b,kind,pos(layout,i,j,ldb),v);bi[pos(layout,i,j,ldb)]=1;
    }
    original_a=a;original_b=b;
    int old_hooks=hooks;
    REQUIRE(invoke(kind,work,layout,upper?'U':'L',n,nrhs,storage(&a,kind),lda,storage(&b,kind),ldb)==0);
    REQUIRE(hooks==old_hooks);
    for(int i=0;i<n;++i)for(int j=0;j<nrhs;++j)close_num(get(&b,kind,pos(layout,i,j,ldb)),solution[i*nrhs+j],128*n*eps*fmax(1,hypot(solution[i*nrhs+j].r,solution[i*nrhs+j].i)));
    double residual=0,scale=0,reconstruction=0,anorm=0;
    for(int i=0;i<n;++i)for(int j=0;j<nrhs;++j){
        Num v=num(0,0);double local=0;
        for(int k=0;k<n;++k){Num av=matrix[i*n+k],x=get(&b,kind,pos(layout,k,j,ldb));v=add(v,product(av,x));local+=hypot(av.r,av.i)*hypot(x.r,x.i);}
        Num rhs=get(&original_b,kind,pos(layout,i,j,ldb));residual=fmax(residual,hypot(v.r-rhs.r,v.i-rhs.i));scale=fmax(scale,local+hypot(rhs.r,rhs.i));
    }
    REQUIRE(residual<=64*n*eps*fmax(1,scale));
    for(int i=0;i<n;++i)for(int j=0;j<n;++j){
        Num v=num(0,0);for(int k=0;k<=i&&k<=j;++k){Num left=get(&a,kind,pos(layout,upper?k:i,upper?i:k,lda));Num right=get(&a,kind,pos(layout,upper?k:j,upper?j:k,lda));v=add(v,upper?product(conjugate(left),right):product(left,conjugate(right)));}
        reconstruction=fmax(reconstruction,hypot(v.r-matrix[i*n+j].r,v.i-matrix[i*n+j].i));anorm=fmax(anorm,hypot(matrix[i*n+j].r,matrix[i*n+j].i));
    }
    REQUIRE(reconstruction<=64*n*eps*fmax(1,anorm));
    for(int i=0;i<400;++i){
        if(!ai[i])REQUIRE(memcmp((char*)storage(&a,kind)+i*element_size(kind),(char*)storage(&original_a,kind)+i*element_size(kind),element_size(kind))==0);
        if(!bi[i])REQUIRE(memcmp((char*)storage(&b,kind)+i*element_size(kind),(char*)storage(&original_b,kind)+i*element_size(kind),element_size(kind))==0);
    }
    factor=a;
    for(int batch=0;batch<2;++batch){
        for(int i=0;i<n;++i){Num v=num(0,0);for(int k=0;k<n;++k)v=add(v,product(matrix[i*n+k],num(k+batch+1,0.25*batch*imag)));put(&b,kind,pos(layout,i,0,ldb),v);}
        REQUIRE(reuse(kind,layout,upper?'U':'L',n,1,storage(&a,kind),lda,storage(&b,kind),ldb)==0);
        for(int i=0;i<n;++i)close_num(get(&b,kind,pos(layout,i,0,ldb)),num(i+batch+1,0.25*batch*imag),128*n*eps*(i+batch+1));
        REQUIRE(memcmp(storage(&a,kind),storage(&factor,kind),400*element_size(kind))==0);
    }
}
static void boundaries(void) {
    for(int kind=0;kind<4;++kind)for(int layout=101;layout<=102;++layout)for(int upper=0;upper<2;++upper)for(int work=0;work<2;++work){
        for(int nrhs=0;nrhs<=1;++nrhs)for(int fail=0;fail<3;++fail){
            Store a,b,before;for(int i=0;i<400;++i){put(&a,kind,i,num(0,0));put(&b,kind,i,num(13,-2));}
            for(int i=0;i<3;++i)put(&a,kind,pos(layout,i,i,3),num(i==fail?-1:4,0));before=b;++scenarios;
            REQUIRE(invoke(kind,work,layout,upper?'U':'L',3,nrhs,storage(&a,kind),3,storage(&b,kind),3)==fail+1);
            REQUIRE(memcmp(storage(&b,kind),storage(&before,kind),400*element_size(kind))==0);
        }
        Store a,b,before;put(&a,kind,0,num(4,kind>=2?7:0));put(&b,kind,0,num(8,kind>=2?4:0));++scenarios;
        REQUIRE(invoke(kind,work,layout,upper?'U':'L',1,1,storage(&a,kind),1,storage(&b,kind),1)==0);
        close_num(get(&b,kind,0),num(2,kind>=2?1:0),1e-6);
        put(&a,kind,0,num(4,kind>=2?11:0));
        REQUIRE(invoke(kind,work,layout,upper?'U':'L',1,0,storage(&a,kind),1,NULL,layout==101?0:1)==0);
        close_num(get(&a,kind,0),num(2,0),1e-6);
        put(&a,kind,0,num(NAN,0));put(&b,kind,0,num(NAN,0));before=a;int old=hooks;
        REQUIRE(invoke(kind,0,layout,upper?'U':'L',1,1,storage(&a,kind),1,storage(&b,kind),1)==-5);
        REQUIRE(hooks==old);REQUIRE(memcmp(storage(&a,kind),storage(&before,kind),element_size(kind))==0);
        put(&a,kind,0,num(-1,0));REQUIRE(invoke(kind,0,layout,upper?'U':'L',1,1,storage(&a,kind),1,storage(&b,kind),1)==-7);
        put(&a,kind,0,num(4,0));put(&b,kind,0,num(8,0));
        REQUIRE(invoke(kind,0,layout,upper?'U':'L',1,1,storage(&a,kind),1,storage(&b,kind),1)==0);
    }
    double a=4,b=8;
    REQUIRE(LAPACKE_dposv(0,'L',1,1,&a,1,&b,1)==-1);REQUIRE(last_info==-1);REQUIRE(strcmp(last_name,"LAPACKE_dposv")==0);
    REQUIRE(LAPACKE_dposv_work(102,'?',1,1,&a,1,&b,1)==-2);
#ifndef ZYNUM_POSV_LINKED_BLAS_XERBLA
    REQUIRE(last_info==1);REQUIRE(strcmp(last_name,"DPOSV")==0);
#endif
    REQUIRE(LAPACKE_dposv_work(101,'?',2,3,&a,1,&b,0)==-6);REQUIRE(last_info==-6);REQUIRE(strcmp(last_name,"LAPACKE_dposv_work")==0);
#ifdef MKL_ILP64
    lapack_int large=(lapack_int)INT32_MAX+17;
    REQUIRE(sizeof(lapack_int)==8);
    REQUIRE(LAPACKE_dposv_work(102,'L',-large,0,NULL,1,NULL,1)==-3);
    REQUIRE(LAPACKE_dposv_work(102,'L',0,large,NULL,1,NULL,large)==0);
#else
    REQUIRE(sizeof(lapack_int)==4);
#endif
}
static int run_posv_cases(void) {
    const int sizes[]={1,3,9,17},rhs[]={1,3};
    for(int kind=0;kind<4;++kind)for(int layout=101;layout<=102;++layout)for(int upper=0;upper<2;++upper)for(int work=0;work<2;++work)for(int n=0;n<4;++n)for(int r=0;r<2;++r)numeric(kind,work,layout,upper,sizes[n],rhs[r]);
    boundaries();
    printf("{\"status\":\"passed\",\"integer_bytes\":%zu,\"numeric_cases\":256,\"scenarios\":%zu,\"assertions\":%zu}\n",sizeof(lapack_int),scenarios,assertions);
    return 0;
}
#endif
