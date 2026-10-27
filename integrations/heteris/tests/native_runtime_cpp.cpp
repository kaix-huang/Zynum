// SPDX-License-Identifier: LGPL-3.0-or-later
#include <zynum_native.h>
#include <cassert>
#include <cstring>
#include <type_traits>
static_assert(sizeof(zynum_native_report_v1)==72);
using Gemm=zynum_native_status(*)(void*,uint64_t,uint64_t,uint64_t,double,const double*,uint64_t,uint64_t,const double*,uint64_t,uint64_t,double,double*,uint64_t,uint64_t,zynum_native_report_v1*);
static_assert(std::is_same_v<decltype(&zynum_native_gemm_f64_nn_v1),Gemm>);
int main(){
 heteris_rp_cpu_create_desc_v1 desc{};desc.abi_version=1;desc.struct_size=sizeof desc;desc.compute_workers=1;desc.managed_thread_limit=2;desc.max_tasks=8;desc.max_dependencies=8;desc.host_bytes_limit=1u<<20;desc.task_scratch_bytes_limit=1u<<16;
 heteris_rp_token runtime{};assert(heteris_cpu_runtime_create_v1(&desc,&runtime)==HETERIS_OK);
 heteris_runtime_provider_v1 provider{};assert(heteris_cpu_runtime_export_provider_v1(runtime,&provider)==HETERIS_OK);auto api=*provider.api;
 zynum_native_context_requirements_v1 req{};assert(zynum_native_context_query_v1(&req)==ZYNUM_NATIVE_OK);
 heteris_rp_scratch_v1 storage{},data[3]{};assert(api.scratch_allocate(provider.provider,req.bytes,req.alignment,&storage)==HETERIS_OK);
 for(auto &lease:data)assert(api.scratch_allocate(provider.provider,8,8,&lease)==HETERIS_OK);
 auto *a=static_cast<double*>(data[0].data),*b=static_cast<double*>(data[1].data),*c=static_cast<double*>(data[2].data);*a=2;*b=3;*c=4;
 zynum_native_policy_v1 policy{1,sizeof policy,1,0,{0,0}};zynum_native_report_v1 report{};
 assert(zynum_native_context_init_v1(storage.data,req.bytes,&provider,&policy,&report)==ZYNUM_NATIVE_OK);
 assert(api.provider_release(provider.provider)==HETERIS_OK);std::memset(&provider,0,sizeof provider);
 assert(zynum_native_gemm_f64_nn_v1(storage.data,1,1,1,2,a,1,1,b,1,1,0.5,c,1,1,&report)==ZYNUM_NATIVE_OK);assert(*c==14&&report.output_state==1&&report.submission_state==2);
 assert(heteris_cpu_runtime_destroy_v1(runtime)==HETERIS_BUSY);
 assert(zynum_native_context_request_close_v1(storage.data,&report)==ZYNUM_NATIVE_OK);
 assert(zynum_native_context_progress_close_v1(storage.data,&report)==ZYNUM_NATIVE_OK);
 for(auto &lease:data)assert(api.scratch_release(lease.lease)==HETERIS_OK);
 assert(api.scratch_release(storage.lease)==HETERIS_OK);assert(heteris_cpu_runtime_destroy_v1(runtime)==HETERIS_OK);return 0;
}
