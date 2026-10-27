// SPDX-License-Identifier: LGPL-3.0-or-later
#include <zynum_native.h>
#include <cassert>
#include <cmath>
#include <cstring>
#include <type_traits>
#include <new>
static_assert(sizeof(zynum_native_report_v1)==72);
using Gemm=zynum_native_status(*)(void*,uint64_t,uint64_t,uint64_t,double,const double*,uint64_t,uint64_t,const double*,uint64_t,uint64_t,double,double*,uint64_t,uint64_t,zynum_native_report_v1*);
static_assert(std::is_same_v<decltype(&zynum_native_gemm_f64_nn_v1),Gemm>);
static_assert(std::is_same_v<decltype(&zynum_native_gemm_f64_tn_v1),Gemm>);
using Laplacian=zynum_native_status(*)(void*,uint64_t,uint64_t,uint64_t,const double*,uint64_t,double*,uint64_t,double,double,double,zynum_native_report_v1*);
static_assert(std::is_same_v<decltype(&zynum_native_laplacian_periodic_3d_f64_v1),Laplacian>);
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
 assert(heteris_cpu_runtime_destroy_v1(runtime)==HETERIS_BUSY);
 assert(zynum_native_context_request_close_v1(storage.data,&report)==ZYNUM_NATIVE_OK);
 assert(zynum_native_context_progress_close_v1(storage.data,&report)==ZYNUM_NATIVE_OK);
 // double is trivially destructible; releasing each lease ends its array lifetime.
 for(auto &lease:data)assert(api.scratch_release(lease.lease)==HETERIS_OK);
 assert(api.scratch_release(storage.lease)==HETERIS_OK);assert(heteris_cpu_runtime_destroy_v1(runtime)==HETERIS_OK);return 0;
}
