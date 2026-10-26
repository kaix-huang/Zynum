/* SPDX-License-Identifier: LGPL-3.0-or-later */
#include "zynum_standalone.h"
#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <string.h>

static heteris_view view(heteris_storage s, unsigned flags) {
    heteris_view v;
    memset(&v,0,sizeof v);
    v.abi_version=HETERIS_ABI_VERSION; v.struct_size=sizeof v;
    v.dtype=HETERIS_F64; v.rank=1; v.flags=flags; v.storage=s;
    v.storage_bytes=32; v.shape[0]=4; v.strides[0]=8;
    return v;
}
int main(void) {
    zynum_resources r;
    heteris_storage xs,ys,zs;
    void *xp,*yp,*zp;
    uint64_t bytes;
    assert(zynum_resources_create_naive(&r)==HETERIS_OK);
    assert(r.allocate(r.user_data,32,64,&xs)==HETERIS_OK);
    assert(r.allocate(r.user_data,32,64,&ys)==HETERIS_OK);
    assert(r.allocate(r.user_data,32,64,&zs)==HETERIS_OK);
    assert(r.map(r.user_data,xs,&xp,&bytes)==HETERIS_OK && bytes==32);
    assert(r.map(r.user_data,ys,&yp,&bytes)==HETERIS_OK && bytes==32);
    assert(r.map(r.user_data,zs,&zp,&bytes)==HETERIS_OK && bytes==32);
    double *x=(double*)xp,*y=(double*)yp;
    for(int i=0;i<4;i++) { x[i]=i+1; y[i]=10; }
    heteris_view xv=view(xs,HETERIS_VIEW_READ),yv=view(ys,HETERIS_VIEW_READ|HETERIS_VIEW_WRITE);
    assert(zynum_axpy_f64(&r,&xv,&yv,2)==HETERIS_OK);
    double result=0;
    assert(zynum_dot_f64(&r,&xv,&yv,&result)==HETERIS_OK);
    assert(result==160);
    assert(zynum_axpy_f64(&r,&xv,&yv,INFINITY)==HETERIS_INVALID_ARGUMENT);
    for(int i=0;i<4;i++) assert(y[i]==12+2*i);
    /* Plain-C ABI exercise of real and by-value complex GEMM scalars. */
    heteris_view a=view(xs,HETERIS_VIEW_READ),b=view(ys,HETERIS_VIEW_READ),d=view(zs,HETERIS_VIEW_WRITE);
    a.rank=b.rank=d.rank=2;
    a.shape[0]=b.shape[0]=d.shape[0]=2;
    a.shape[1]=b.shape[1]=d.shape[1]=2;
    a.strides[0]=b.strides[0]=d.strides[0]=16;
    a.strides[1]=b.strides[1]=d.strides[1]=8;
    assert(zynum_gemm_f64(&r,ZYNUM_OP_NONE,ZYNUM_OP_NONE,&a,&b,&d,1,0)==HETERIS_OK);
    const double *z=(const double *)zp;
    assert(z[0]==44 && z[1]==50 && z[2]==100 && z[3]==114);
    a.dtype=b.dtype=d.dtype=HETERIS_COMPLEX128;
    a.shape[0]=a.shape[1]=b.shape[0]=b.shape[1]=d.shape[0]=d.shape[1]=1;
    const zynum_complex128 alpha={0,1}, beta={0,0};
    assert(zynum_gemm_c128(&r,ZYNUM_OP_CONJUGATE_TRANSPOSE,ZYNUM_OP_NONE,&a,&b,&d,alpha,beta)==HETERIS_OK);
    assert(z[0]==10 && z[1]==40); /* i*conj(1+2i)*(12+14i) */
    d.flags|=HETERIS_VIEW_READ;
    const zynum_complex128 alpha2={1,-2}, beta2={0.5,-1}, invalid={0,INFINITY};
    assert(zynum_gemm_c128(&r,ZYNUM_OP_CONJUGATE_TRANSPOSE,ZYNUM_OP_NONE,&a,&b,&d,alpha2,beta2)==HETERIS_OK);
    assert(z[0]==65 && z[1]==-80);
    assert(zynum_gemm_c128(&r,ZYNUM_OP_CONJUGATE_TRANSPOSE,ZYNUM_OP_NONE,&a,&b,&d,alpha2,invalid)==HETERIS_INVALID_ARGUMENT);
    assert(z[0]==65 && z[1]==-80);
    assert(zynum_resources_destroy_naive(&r)==HETERIS_BUSY);
    assert(r.release(r.user_data,xs)==HETERIS_OK);
    assert(r.release(r.user_data,ys)==HETERIS_OK);
    assert(r.release(r.user_data,zs)==HETERIS_OK);
    assert(zynum_resources_destroy_naive(&r)==HETERIS_OK);
    puts("Zynum standalone C ABI: axpy/dot/real+complex GEMM and explicit naive provider passed");
    return 0;
}
