/* SPDX-License-Identifier: LGPL-3.0-or-later */
#include <zynum_native.h>
#include <assert.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
static int authority, retains, releases, owned=1, fail_release;
static heteris_status retain(heteris_rp_token token) { assert(token.authority==&authority);++retains;++owned;return HETERIS_OK; }
static heteris_status release(heteris_rp_token token) { assert(token.authority==&authority);++releases;if(fail_release)return HETERIS_BUSY;--owned;return HETERIS_OK; }
static heteris_status query(heteris_rp_token token,heteris_rp_cpu_caps_v1 *out) {
 memset(out,0,sizeof *out);out->abi_version=1;out->struct_size=sizeof *out;
 out->mode=HETERIS_RESOURCE_MODE_INTEGRATED;out->device_kind=HETERIS_DEVICE_CPU;
 out->flags=HETERIS_RP_CPU_BULK_SYNC|HETERIS_RP_HOST_SCRATCH|HETERIS_RP_EXPLICIT_HOST_BACKING_BUDGET|HETERIS_RP_ORIGINAL_MANAGED_THREAD_BUDGET|HETERIS_RP_QUIESCENT_RETURN;
 out->provider_instance=(heteris_rp_identity){token.serial,token.generation};
 out->runtime_instance=(heteris_rp_identity){1,1};out->memory_domain=(heteris_rp_identity){2,1};
 out->compute_workers=2;out->managed_threads=3;out->max_bulk_commands=2;return HETERIS_OK;
}
static heteris_status same(heteris_rp_token a,heteris_rp_token b,uint32_t *out){(void)a;(void)b;*out=1;return HETERIS_OK;}
static heteris_status alloc_scratch(heteris_rp_token t,uint64_t b,uint64_t a,heteris_rp_scratch_v1 *out){(void)t;(void)b;(void)a;(void)out;abort();}
static heteris_status scratch_ref(heteris_rp_token t){(void)t;abort();}
static heteris_status bulk(heteris_rp_token t,uint64_t c,uint64_t g,uint64_t z,uint32_t p,heteris_rp_cpu_range_fn f,void *u,heteris_rp_bulk_result_v1 *r){(void)t;(void)c;(void)g;(void)z;(void)p;(void)f;(void)u;(void)r;abort();}
int main(void) {
 zynum_native_context_requirements_v1 req;
 assert(zynum_native_context_query_v1(NULL)==ZYNUM_NATIVE_INVALID_ARGUMENT);
 assert(zynum_native_context_query_v1(&req)==ZYNUM_NATIVE_OK);
 assert(req.abi_version==1 && req.struct_size==sizeof req && req.bytes && req.alignment);
 size_t size=(size_t)((req.bytes+req.alignment-1)/req.alignment*req.alignment);
 void *first=aligned_alloc((size_t)req.alignment,size),*second=aligned_alloc((size_t)req.alignment,size);assert(first&&second);
 heteris_runtime_provider_api_v1 api={1,sizeof api,&authority,retain,release,query,same,alloc_scratch,scratch_ref,scratch_ref,bulk};
 heteris_runtime_provider_v1 provider={1,sizeof provider,&api,{&authority,55,1,HETERIS_RP_PROVIDER,0}};
 zynum_native_policy_v1 policy={1,sizeof policy,1,0,{0,0}};
 zynum_native_report_v1 report;
 assert(zynum_native_context_init_v1(first,req.bytes-1,&provider,&policy,&report)==ZYNUM_NATIVE_INVALID_ARGUMENT);
 policy.reserved[0]=1;assert(zynum_native_context_init_v1(first,req.bytes,&provider,&policy,&report)==ZYNUM_NATIVE_INVALID_ARGUMENT);policy.reserved[0]=0;
 assert(retains==0 && owned==1);
 assert(zynum_native_context_init_v1(first,req.bytes,&provider,&policy,&report)==ZYNUM_NATIVE_OK);
 assert(zynum_native_context_init_v1(second,req.bytes,&provider,&policy,&report)==ZYNUM_NATIVE_OK);
 assert(retains==2 && owned==3 && report.abi_version==1 && report.struct_size==sizeof report);
 /* Both contexts must own copied immutable descriptor/table prefixes. */
 memset(&provider,0,sizeof provider);memset(&api,0,sizeof api);
 assert(zynum_native_context_request_close_v1(first,&report)==ZYNUM_NATIVE_OK && report.close_state==ZYNUM_NATIVE_CLOSE_CLOSING);
 fail_release=1;
 assert(zynum_native_context_progress_close_v1(first,&report)==ZYNUM_NATIVE_RELEASE_FAILED);
 assert(report.close_state==ZYNUM_NATIVE_CLOSE_RELEASE_FAILED && report.provider_status==HETERIS_BUSY && report.cleanup_status==HETERIS_BUSY && owned==3);
 fail_release=0;
 assert(zynum_native_context_progress_close_v1(first,&report)==ZYNUM_NATIVE_OK && report.close_state==ZYNUM_NATIVE_CLOSE_CLOSED && owned==2);
 assert(zynum_native_context_progress_close_v1(first,&report)==ZYNUM_NATIVE_OK && owned==2);
 assert(zynum_native_context_request_close_v1(second,&report)==ZYNUM_NATIVE_OK);
 assert(zynum_native_context_progress_close_v1(second,&report)==ZYNUM_NATIVE_OK && owned==1 && releases==3);
 free(first);free(second);return 0;
}
