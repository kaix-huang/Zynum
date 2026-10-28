// SPDX-License-Identifier: LGPL-3.0-or-later
#ifdef ZYNUM_NATIVE_SPECTRAL_FIRST
#include <zynum_native_spectral.h>
#include <zynum_native.h>
#else
#include <zynum_native.h>
#include <zynum_native_spectral.h>
#endif
#include <zynum_native_spectral.h>
#include <zynum_native.h>
#include <cassert>
#include <cmath>
#include <cstring>
#include <type_traits>
#include <cstddef>
#include <new>
static_assert(sizeof(zynum_native_report_v1)==72);
using Gemm=zynum_native_status(*)(void*,uint64_t,uint64_t,uint64_t,double,const double*,uint64_t,uint64_t,const double*,uint64_t,uint64_t,double,double*,uint64_t,uint64_t,zynum_native_report_v1*);
static_assert(std::is_same_v<decltype(&zynum_native_gemm_f64_nn_v1),Gemm>);
static_assert(std::is_same_v<decltype(&zynum_native_gemm_f64_tn_v1),Gemm>);
using Laplacian=zynum_native_status(*)(void*,uint64_t,uint64_t,uint64_t,const double*,uint64_t,double*,uint64_t,double,double,double,zynum_native_report_v1*);
static_assert(std::is_same_v<decltype(&zynum_native_laplacian_periodic_3d_f64_v1),Laplacian>);

// Both include orders and repeated inclusion are selectable in this same C++20
// fixture. These are source assertions until actual consumer compilation runs.
static_assert(sizeof(zynum_native_spectral_report_v1)==80);
static_assert(offsetof(zynum_native_spectral_report_v1,order)==40);
static_assert(offsetof(zynum_native_spectral_report_v1,reserved)==64);
using SpectralCallback=int32_t(*)(void*,void*,uint64_t,uint64_t,const double*,uint64_t,uint64_t,double*,uint64_t,uint64_t);
static_assert(std::is_same_v<zynum_native_spectral_apply_f64_v1,SpectralCallback>);
using SpectralSeries=zynum_native_spectral_status(*)(void*,SpectralCallback,void*,uint64_t,uint64_t,uint64_t,const double*,uint64_t,uint64_t,const double*,uint64_t,uint64_t,double,double,double*,uint64_t,uint64_t,double*,uint64_t,void*,uint64_t,zynum_native_spectral_report_v1*);
using SpectralMoments=zynum_native_spectral_status(*)(void*,SpectralCallback,void*,uint64_t,uint64_t,uint64_t,const double*,uint64_t,uint64_t,uint64_t,double,double,double*,uint64_t,uint64_t,double*,uint64_t,void*,uint64_t,zynum_native_spectral_report_v1*);
static_assert(std::is_same_v<decltype(&zynum_native_chebyshev_series_f64_v1),SpectralSeries>);
static_assert(std::is_same_v<decltype(&zynum_native_chebyshev_moments_f64_v1),SpectralMoments>);
static_assert(std::is_same_v<decltype(&zynum_native_spectral_guard_query_v1),zynum_native_spectral_status(*)(uint64_t*,uint64_t*)>);
static_assert(std::is_same_v<decltype(&zynum_native_spectral_guard_init_v1),zynum_native_spectral_status(*)(void*,uint64_t)>);
static_assert(std::is_same_v<decltype(&zynum_native_spectral_workspace_elements_v1),zynum_native_spectral_status(*)(uint32_t,uint64_t,uint64_t,uint64_t,uint64_t*)>);
struct CppSpectralState { double factor=2; unsigned calls=0; zynum_native_report_v1 child{}; };
extern "C" {
static int32_t cpp_spectral_apply(void *raw,void *context,uint64_t n,uint64_t w,const double *x,uint64_t xb,uint64_t ldx,double *y,uint64_t yb,uint64_t ldy) {
 try {
  auto &state=*static_cast<CppSpectralState*>(raw);++state.calls;
  assert(ldx==n&&ldy==n&&xb==n*w&&yb==n*w);
  if(n==0||w==0){assert(x==nullptr&&y==nullptr&&xb==0&&yb==0);return 0;}
  // One scalar scaling GEMM per compact column, through the same real Context.
  for(uint64_t j=0;j<w;++j){
   const auto status=zynum_native_gemm_f64_nn_v1(context,n,1,1,1,x+j*n,n,n,&state.factor,1,1,0,y+j*n,n,n,&state.child);
   if(status!=ZYNUM_NATIVE_OK)return status;
  }
  return 0;
 } catch(...) { return -27001; }
}
}
static void cpp_spectral_cases(void *context,const heteris_runtime_provider_api_v1 &api,heteris_rp_token provider,double *x,double *coefficients,double *y) {
 uint64_t bytes=0,alignment=0,elements=0;
 assert(zynum_native_spectral_guard_query_v1(&bytes,&alignment)==ZYNUM_NATIVE_SPECTRAL_OK);
 assert(bytes>0&&alignment>0);
 heteris_rp_scratch_v1 guard{},scratch{};
 assert(api.scratch_allocate(provider,bytes+16,alignment,&guard)==HETERIS_OK);
 assert(zynum_native_spectral_workspace_elements_v1(ZYNUM_NATIVE_SPECTRAL_MOMENTS,2,2,3,&elements)==ZYNUM_NATIVE_SPECTRAL_OK);
 assert(elements==18);
 assert(api.scratch_allocate(provider,(elements+2)*sizeof(double),alignof(double),&scratch)==HETERIS_OK);
 // Raw aligned provider storage backs the opaque Zig object. No C++ atomic is
 // constructed, cast or accessed. Only bytes outside its queried prefix are
 // inspected after init; the storage is stable through every returning call.
 auto *guard_tail=::new (guard.data) unsigned char[bytes+16];
 std::memset(guard_tail,0xa5,static_cast<size_t>(bytes+16));
 auto *work=::new (scratch.data) double[elements+2];
 for(uint64_t i=0;i<elements+2;++i)work[i]=991;
 assert(zynum_native_spectral_guard_init_v1(guard.data,bytes+16)==ZYNUM_NATIVE_SPECTRAL_OK);
 for(uint64_t i=bytes;i<bytes+16;++i)assert(guard_tail[i]==0xa5);
 x[0]=1;x[1]=2;x[2]=x[3]=NAN;x[4]=3;x[5]=4;
 coefficients[0]=1;coefficients[1]=2;coefficients[2]=3;
 for(unsigned i=0;i<16;++i)y[i]=773;
 CppSpectralState state;zynum_native_spectral_report_v1 report{};
 assert(zynum_native_chebyshev_series_f64_v1(context,cpp_spectral_apply,&state,sizeof state,2,2,x,6,4,coefficients,3,3,0,4,y,6,4,work,elements+2,guard.data,bytes+16,&report)==ZYNUM_NATIVE_SPECTRAL_OK);
 assert(y[0]==-2&&y[1]==-4&&y[4]==-6&&y[5]==-8&&y[2]==773&&y[3]==773&&y[6]==773);
 assert(state.calls==2&&report.callbacks_returned==2&&report.phase==ZYNUM_NATIVE_SPECTRAL_COMPLETE&&report.output_state==ZYNUM_NATIVE_OUTPUT_COMPLETE);
 assert(report.admission_attempted==1&&report.admission_succeeded==1&&report.admission_status==ZYNUM_NATIVE_OK&&report.callback_returned==1&&report.callback_status==0);
 state.calls=0;
 assert(zynum_native_chebyshev_moments_f64_v1(context,cpp_spectral_apply,&state,sizeof state,2,2,x,6,4,3,0,4,y,7,4,work,elements+2,guard.data,bytes+16,&report)==ZYNUM_NATIVE_SPECTRAL_OK);
 assert(y[0]==5&&y[1]==0&&y[2]==-5&&y[4]==25&&y[5]==0&&y[6]==-25&&y[3]==773&&y[7]==773);
 assert(state.calls==2&&report.order_valid==1&&report.order==2&&report.callback_order==2&&report.reserved[0]==0&&report.reserved[1]==0);
 assert(work[elements]==991&&work[elements+1]==991);
 for(uint64_t i=bytes;i<bytes+16;++i)assert(guard_tail[i]==0xa5);
 assert(api.scratch_release(scratch.lease)==HETERIS_OK);
 assert(api.scratch_release(guard.lease)==HETERIS_OK);
}

int main(){
 heteris_rp_cpu_create_desc_v1 desc{};desc.abi_version=1;desc.struct_size=sizeof desc;desc.compute_workers=1;desc.managed_thread_limit=2;desc.max_tasks=8;desc.max_dependencies=8;desc.host_bytes_limit=1u<<20;desc.task_scratch_bytes_limit=1u<<16;
 heteris_rp_token runtime{};assert(heteris_cpu_runtime_create_v1(&desc,&runtime)==HETERIS_OK);
 heteris_runtime_provider_v1 provider{};assert(heteris_cpu_runtime_export_provider_v1(runtime,&provider)==HETERIS_OK);auto api=*provider.api;
 zynum_native_context_requirements_v1 req{};assert(zynum_native_context_query_v1(&req)==ZYNUM_NATIVE_OK);
 heteris_rp_scratch_v1 storage{},data[3]{};assert(api.scratch_allocate(provider.provider,req.bytes,req.alignment,&storage)==HETERIS_OK);
 for(auto &lease:data)assert(api.scratch_allocate(provider.provider,128,8,&lease)==HETERIS_OK);
 auto *a=::new (data[0].data) double[16]{};
 auto *b=::new (data[1].data) double[16]{};
 auto *c=::new (data[2].data) double[16]{};*a=2;*b=3;*c=4;
 zynum_native_policy_v1 policy{1,sizeof policy,1,0,{0,0}};zynum_native_report_v1 report{};
 assert(zynum_native_context_init_v1(storage.data,req.bytes,&provider,&policy,&report)==ZYNUM_NATIVE_OK);
 const auto spectral_provider=provider.provider;
 assert(api.provider_release(provider.provider)==HETERIS_OK);std::memset(&provider,0,sizeof provider);
 assert(zynum_native_gemm_f64_nn_v1(storage.data,1,1,1,2,a,1,1,b,1,1,0.5,c,1,1,&report)==ZYNUM_NATIVE_OK);assert(*c==14&&report.output_state==1&&report.submission_state==2);
 assert(zynum_native_gemm_f64_tn_v1(storage.data,1,1,1,2,a,1,1,b,1,1,0.5,c,1,1,&report)==ZYNUM_NATIVE_OK);assert(*c==19&&report.output_state==1&&report.submission_state==2);
 assert(report.reserved[0]==0&&report.reserved[1]==0);
 a[0]=2;a[1]=3;b[0]=5;b[1]=7;*c=4;
 assert(zynum_native_gemm_f64_tn_v1(storage.data,1,1,2,2,a,2,2,b,2,2,0.5,c,1,1,&report)==ZYNUM_NATIVE_OK);assert(*c==64);
 assert(zynum_native_gemm_f64_tn_v1(storage.data,1,1,2,2,a,1,2,b,2,2,0.5,c,1,1,&report)==ZYNUM_NATIVE_VIEW_INVALID);assert(*c==64);
 for(unsigned i=0;i<8;++i){a[i]=i<6?static_cast<double>(i):991;c[i]=773;}
 assert(zynum_native_laplacian_periodic_3d_f64_v1(storage.data,1,2,3,a,8,c,8,2,3,4,&report)==ZYNUM_NATIVE_OK);
 const double expected[6]={30,18,6,-6,-18,-30};
 for(unsigned i=0;i<6;++i)assert(c[i]==expected[i]);
 assert(c[6]==773&&c[7]==773&&a[6]==991&&a[7]==991);
 assert(report.output_state==ZYNUM_NATIVE_OUTPUT_COMPLETE&&report.submission_state==ZYNUM_NATIVE_SUBMISSION_QUIESCENT);
 assert(report.callbacks_started==report.callbacks_completed&&report.callbacks_completed>0&&report.reserved[0]==0&&report.reserved[1]==0);
 assert(zynum_native_laplacian_periodic_3d_f64_v1(storage.data,UINT64_MAX,0,UINT64_MAX,nullptr,0,nullptr,0,2,3,4,&report)==ZYNUM_NATIVE_OK);
 assert(report.output_state==ZYNUM_NATIVE_OUTPUT_UNCHANGED&&report.submission_state==ZYNUM_NATIVE_SUBMISSION_NOT_SUBMITTED);
 assert(zynum_native_laplacian_periodic_3d_f64_v1(storage.data,0,1,1,nullptr,1,nullptr,0,2,3,4,&report)==ZYNUM_NATIVE_VIEW_INVALID);
 assert(zynum_native_laplacian_periodic_3d_f64_v1(storage.data,0,1,1,a,8,a+7,1,2,3,4,&report)==ZYNUM_NATIVE_VIEW_INVALID);
 cpp_spectral_cases(storage.data,api,spectral_provider,a,b,c);
 assert(heteris_cpu_runtime_destroy_v1(runtime)==HETERIS_BUSY);
 assert(zynum_native_context_request_close_v1(storage.data,&report)==ZYNUM_NATIVE_OK);
 assert(zynum_native_context_progress_close_v1(storage.data,&report)==ZYNUM_NATIVE_OK);
 // double is trivially destructible; releasing each lease ends its array lifetime.
 for(auto &lease:data)assert(api.scratch_release(lease.lease)==HETERIS_OK);
 assert(api.scratch_release(storage.lease)==HETERIS_OK);assert(heteris_cpu_runtime_destroy_v1(runtime)==HETERIS_OK);return 0;
}
