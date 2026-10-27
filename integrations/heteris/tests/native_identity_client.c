/* SPDX-License-Identifier: LGPL-3.0-or-later */
#define ZYNUM_NATIVE_IDENTITY_TESTING 1
#include <zynum_native.h>
const void *zynum_client_core_identity(void){return zynum_native_core_identity_v1();}
const void *zynum_client_execution_identity(void){return zynum_native_execution_identity_v1();}
zynum_native_status zynum_client_gemm(void *context,uint64_t m,uint64_t n,uint64_t k,double alpha,const double *a,uint64_t al,uint64_t lda,const double *b,uint64_t bl,uint64_t ldb,double beta,double *c,uint64_t cl,uint64_t ldc,zynum_native_report_v1 *report){return zynum_native_gemm_f64_nn_v1(context,m,n,k,alpha,a,al,lda,b,bl,ldb,beta,c,cl,ldc,report);}
