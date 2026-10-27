/* SPDX-License-Identifier: LGPL-3.0-or-later */
#define ZYNUM_NATIVE_IDENTITY_TESTING 1
#include <zynum_native.h>
#include <assert.h>
#include <math.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <signal.h>
#include <sys/resource.h>
#include <sys/wait.h>
#include <unistd.h>
#ifdef ZYNUM_NATIVE_DSO_CLIENT_TESTS
#include <dlfcn.h>
#endif
static heteris_runtime_provider_api_v1 real_api;
static pthread_t caller;
static _Atomic unsigned bulk_calls, callback_calls, caller_callbacks;
enum death_fault { DEATH_NONE, DEATH_MALFORMED, DEATH_NONQUIESCENT, DEATH_RELEASE };
static enum death_fault death_fault;
static unsigned death_phase, death_allocations, death_release_site;
static heteris_rp_token death_leases[5];
static heteris_status death_allocate(heteris_rp_token p,uint64_t bytes,uint64_t alignment,heteris_rp_scratch_v1 *out) {
 heteris_status status=real_api.scratch_allocate(p,bytes,alignment,out);
 if(status==HETERIS_OK){assert(death_allocations<5);death_leases[death_allocations++]=out->lease;}return status;
}
static heteris_status death_release(heteris_rp_token lease) {
 if(death_fault==DEATH_RELEASE&&death_allocations>=death_release_site&&lease.serial==death_leases[death_release_site-1].serial){fprintf(stderr,"TN_DEATH release_site=%u\n",death_release_site);return -1942;}
 return real_api.scratch_release(lease);
}
struct invocation { heteris_rp_cpu_range_fn callback; void *data; };
static heteris_status range(void *raw,uint64_t begin,uint64_t end) {
 struct invocation *call=raw;atomic_fetch_add(&callback_calls,1);
 if(pthread_equal(caller,pthread_self()))atomic_fetch_add(&caller_callbacks,1);
 return call->callback(call->data,begin,end);
}
static heteris_status traced_bulk(heteris_rp_token p,uint64_t count,uint64_t grain,uint64_t scratch,uint32_t cap,heteris_rp_cpu_range_fn callback,void *data,heteris_rp_bulk_result_v1 *out) {
 struct invocation call={callback,data};atomic_fetch_add(&bulk_calls,1);
 heteris_status status=real_api.cpu_bulk_sync(p,count,grain,scratch,cap,range,&call,out);
 if(death_fault!=DEATH_NONE&&atomic_load(&bulk_calls)==death_phase){
  fprintf(stderr,"TN_DEATH phase=%u fault=%u\n",death_phase,(unsigned)death_fault);
  if(death_fault==DEATH_MALFORMED)out->reserved=1;
  if(death_fault==DEATH_NONQUIESCENT){assert(out->callbacks_started>0);out->callbacks_completed=out->callbacks_started-1;}
 }return status;
}
typedef zynum_native_status (*gemm_fn)(void*,uint64_t,uint64_t,uint64_t,double,const double*,uint64_t,uint64_t,const double*,uint64_t,uint64_t,double,double*,uint64_t,uint64_t,zynum_native_report_v1*);
static void allocate(heteris_rp_token provider,uint64_t bytes,uint64_t alignment,heteris_rp_scratch_v1 *out) { assert(real_api.scratch_allocate(provider,bytes,alignment,out)==HETERIS_OK); }
static void equals(const double *a,const double *b,size_t n) { for(size_t i=0;i<n;++i)assert(fabs(a[i]-b[i])<1e-12); }
/* Each child creates its own real Runtime after fork. Only the test proxy
   corrupts a quiescent completion or refuses a valid lease release. No worker
   remains live when the proxy reports malformed/nonquiescent completion. */
static void tn_death_case(enum death_fault fault,unsigned phase,unsigned release_site,uint32_t limit) {
 struct rlimit core_limit={0,0};assert(setrlimit(RLIMIT_CORE,&core_limit)==0);alarm(15);
 death_fault=fault;death_phase=phase;death_release_site=release_site;death_allocations=0;
 caller=pthread_self();atomic_store(&bulk_calls,0);
 heteris_rp_cpu_create_desc_v1 desc={0};desc.abi_version=1;desc.struct_size=sizeof desc;
 desc.compute_workers=2;desc.managed_thread_limit=3;desc.max_tasks=8;desc.max_dependencies=8;desc.host_bytes_limit=1u<<20;
 heteris_rp_token runtime;assert(heteris_cpu_runtime_create_v1(&desc,&runtime)==HETERIS_OK);
 heteris_runtime_provider_v1 provider;assert(heteris_cpu_runtime_export_provider_v1(runtime,&provider)==HETERIS_OK);
 real_api=*provider.api;heteris_runtime_provider_api_v1 proxy=real_api;
 proxy.cpu_bulk_sync=traced_bulk;proxy.scratch_allocate=death_allocate;proxy.scratch_release=death_release;provider.api=&proxy;
 zynum_native_context_requirements_v1 req;assert(zynum_native_context_query_v1(&req)==ZYNUM_NATIVE_OK);
 heteris_rp_scratch_v1 context;assert(real_api.scratch_allocate(provider.provider,req.bytes,req.alignment,&context)==HETERIS_OK);
 zynum_native_policy_v1 policy={1,sizeof policy,1,limit,{0,0}};zynum_native_report_v1 report;
 assert(zynum_native_context_init_v1(context.data,req.bytes,&provider,&policy,&report)==ZYNUM_NATIVE_OK);
 const double a[4]={1,2,3,4};double output[4]={7,7,7,7};
 (void)zynum_native_gemm_f64_tn_v1(context.data,2,2,2,1,a,4,2,a,4,2,1,output,4,2,&report);
 _exit(99); /* A returning API is never accepted as fail-stop evidence. */
}
static void expect_tn_death(enum death_fault fault,unsigned phase,unsigned release_site,uint32_t limit) {
 int channel[2];assert(pipe(channel)==0);pid_t child=fork();assert(child>=0);
 if(child==0){close(channel[0]);assert(dup2(channel[1],STDERR_FILENO)>=0);close(channel[1]);tn_death_case(fault,phase,release_site,limit);}
 close(channel[1]);char output[16384];size_t used=0;char discard[1024];ssize_t amount;
 while((amount=read(channel[0],used<sizeof output-1?output+used:discard,used<sizeof output-1?sizeof output-1-used:sizeof discard))>0){if(used<sizeof output-1)used+=(size_t)amount;}
 assert(amount==0);close(channel[0]);output[used]=0;int status;assert(waitpid(child,&status,0)==child);
 assert(WIFSIGNALED(status));assert(WTERMSIG(status)==SIGABRT);
 char marker[96];
 if(fault==DEATH_RELEASE)snprintf(marker,sizeof marker,"TN_DEATH release_site=%u",release_site);
 else snprintf(marker,sizeof marker,"TN_DEATH phase=%u fault=%u",phase,(unsigned)fault);
 assert(strstr(output,marker)!=NULL);
 assert(strstr(output,"HETERIS_PROVIDER_CONTRACT_FAILURE")!=NULL);
 const char *reason=fault==DEATH_MALFORMED?"malformed bulk completion prefix/counts":fault==DEATH_NONQUIESCENT?"bulk returned before all callbacks completed":"valid scratch release failed";
 assert(strstr(output,reason)!=NULL);
}
static void tn_death_cases(void) {
 for(uint32_t limit=1;limit<=2;++limit){
  for(unsigned phase=1;phase<=2;++phase){expect_tn_death(DEATH_MALFORMED,phase,0,limit);expect_tn_death(DEATH_NONQUIESCENT,phase,0,limit);}
  /* Allocation identities: packed A, pack descriptor, pack Invocation,
     original NN Tasks, original NN Invocation. Every release is protected. */
  for(unsigned site=1;site<=5;++site)expect_tn_death(DEATH_RELEASE,0,site,limit);
 }
 printf("native TN fail-stop cases passed: 18 real-Runtime child processes\n");
}
int main(int argc,char **argv) {
 tn_death_cases();
 gemm_fn call_gemm=zynum_native_gemm_f64_nn_v1;
#ifdef ZYNUM_NATIVE_DSO_CLIENT_TESTS
 assert(argc==3);void *clients[2];
 for(int i=0;i<2;++i){clients[i]=dlopen(argv[i+1],RTLD_NOW|RTLD_LOCAL);if(!clients[i]){fprintf(stderr,"%s\n",dlerror());return 2;}
  const void *(*core_id)(void)=dlsym(clients[i],"zynum_client_core_identity");
  const void *(*execution_id)(void)=dlsym(clients[i],"zynum_client_execution_identity");
  assert(core_id&&execution_id);assert(core_id()==zynum_native_core_identity_v1());assert(execution_id()==zynum_native_execution_identity_v1());
 }
 call_gemm=(gemm_fn)dlsym(clients[0],"zynum_client_gemm");assert(call_gemm);
#else
 (void)argc;(void)argv;
#endif
 caller=pthread_self();
 heteris_rp_cpu_create_desc_v1 desc={0};desc.abi_version=1;desc.struct_size=sizeof desc;
 desc.compute_workers=2;desc.managed_thread_limit=3;desc.max_tasks=8;desc.max_dependencies=8;
 desc.host_bytes_limit=1u<<20;desc.task_scratch_bytes_limit=1u<<16;
 heteris_rp_token runtime;assert(heteris_cpu_runtime_create_v1(&desc,&runtime)==HETERIS_OK);
 heteris_runtime_provider_v1 provider;assert(heteris_cpu_runtime_export_provider_v1(runtime,&provider)==HETERIS_OK);
 real_api=*provider.api;heteris_runtime_provider_api_v1 traced=real_api;traced.cpu_bulk_sync=traced_bulk;provider.api=&traced;
 zynum_native_context_requirements_v1 req;assert(zynum_native_context_query_v1(&req)==ZYNUM_NATIVE_OK);
 heteris_rp_scratch_v1 contexts[2],matrices[3];
 for(int i=0;i<2;++i)allocate(provider.provider,req.bytes,req.alignment,&contexts[i]);
 for(int i=0;i<3;++i)allocate(provider.provider,128,8,&matrices[i]);
 double *a=matrices[0].data,*b=matrices[1].data,*c=matrices[2].data;
 const double av[4]={1,2,3,4},bv[4]={5,6,7,8},expected[4]={23.5,34.5,31.5,46.5};
 memcpy(a,av,sizeof av);memcpy(b,bv,sizeof bv);for(int i=0;i<4;++i)c[i]=1;
 zynum_native_policy_v1 policy={1,sizeof policy,1,0,{0,0}};zynum_native_report_v1 report;
 for(int i=0;i<2;++i)assert(zynum_native_context_init_v1(contexts[i].data,req.bytes,&provider,&policy,&report)==ZYNUM_NATIVE_OK);
 assert(real_api.provider_release(provider.provider)==HETERIS_OK);memset(&provider,0,sizeof provider);memset(&traced,0,sizeof traced);
 assert(heteris_cpu_runtime_destroy_v1(runtime)==HETERIS_BUSY);
 assert(call_gemm(contexts[0].data,2,2,2,1,a,4,2,b,4,2,0.5,c,4,2,&report)==ZYNUM_NATIVE_OK);equals(c,expected,4);
 assert(report.output_state==1&&report.submission_state==2&&report.callbacks_completed>0&&report.callbacks_started==report.callbacks_completed);
 unsigned submitted=atomic_load(&bulk_calls);double saved[4];memcpy(saved,c,sizeof saved);
 assert(call_gemm(contexts[0].data,UINT64_MAX,2,2,1,a,4,2,b,4,2,0,c,4,2,&report)==ZYNUM_NATIVE_OVERFLOW);
 assert(call_gemm(contexts[0].data,2,2,2,1,a,3,2,b,4,2,0,c,4,2,&report)==ZYNUM_NATIVE_VIEW_INVALID);
 assert(call_gemm(contexts[0].data,2,2,2,1,NULL,4,2,b,4,2,0,c,4,2,&report)==ZYNUM_NATIVE_VIEW_INVALID);
 assert(call_gemm(contexts[0].data,2,2,2,1,(const double*)((uintptr_t)a+1),4,2,b,4,2,0,c,4,2,&report)==ZYNUM_NATIVE_VIEW_INVALID);
 assert(call_gemm(contexts[0].data,2,2,2,1,(const double*)(UINTPTR_MAX-7),4,2,b,4,2,0,c,4,2,&report)==ZYNUM_NATIVE_OVERFLOW);
 assert(call_gemm(contexts[0].data,2,2,2,1,a,4,2,b,4,2,0,a,4,2,&report)==ZYNUM_NATIVE_VIEW_INVALID);
 assert(call_gemm(contexts[0].data,2,2,2,1,a,4,2,b,4,2,0,c,4,2,(zynum_native_report_v1*)c)==ZYNUM_NATIVE_INVALID_ARGUMENT);
 equals(c,saved,4);assert(atomic_load(&bulk_calls)==submitted);
 for(int i=0;i<4;++i)a[i]=b[i]=NAN;
 assert(call_gemm(contexts[0].data,2,2,2,0,a,4,2,b,4,2,2,c,4,2,&report)==ZYNUM_NATIVE_OK);
 for(int i=0;i<4;++i)assert(c[i]==saved[i]*2);
 assert(call_gemm(contexts[0].data,2,2,0,1,NULL,0,0,NULL,0,0,0.5,c,4,2,&report)==ZYNUM_NATIVE_OK);equals(c,saved,4);
 submitted=atomic_load(&bulk_calls);
 assert(call_gemm(contexts[0].data,0,2,0,1,NULL,0,0,NULL,0,0,0,NULL,0,0,&report)==ZYNUM_NATIVE_OK&&report.output_state==0);
 assert(atomic_load(&bulk_calls)==submitted);
 assert(zynum_native_context_request_close_v1(contexts[0].data,&report)==ZYNUM_NATIVE_OK);
 assert(call_gemm(contexts[0].data,2,2,0,1,NULL,0,0,NULL,0,0,1,c,4,2,&report)==ZYNUM_NATIVE_CONTEXT_CLOSING);
 assert(zynum_native_context_progress_close_v1(contexts[0].data,&report)==ZYNUM_NATIVE_OK);
 assert(heteris_cpu_runtime_destroy_v1(runtime)==HETERIS_BUSY);
#ifdef ZYNUM_NATIVE_DSO_CLIENT_TESTS
 call_gemm=(gemm_fn)dlsym(clients[1],"zynum_client_gemm");assert(call_gemm);
#endif
 assert(call_gemm(contexts[1].data,2,2,0,1,NULL,0,0,NULL,0,0,1,c,4,2,&report)==ZYNUM_NATIVE_OK);equals(c,saved,4);
 /* Rectangular physical A(k,m), padded buffers and a terminal C-only report. */
 for(int i=0;i<16;++i)a[i]=b[i]=c[i]=77;
 a[0]=1;a[1]=2;a[2]=3;a[5]=4;a[6]=5;a[7]=6;
 b[0]=7;b[1]=8;b[2]=9;b[4]=10;b[5]=11;b[6]=12;
 c[0]=c[1]=c[3]=c[4]=1;
 submitted=atomic_load(&bulk_calls);double tn_saved[6];memcpy(tn_saved,c,sizeof tn_saved);
 assert(zynum_native_gemm_f64_tn_v1(contexts[1].data,2,2,3,1,a,7,5,b,7,4,0.5,c,6,3,&report)==ZYNUM_NATIVE_VIEW_INVALID);
 assert(zynum_native_gemm_f64_tn_v1(contexts[1].data,2,2,3,0,a,8,2,b,7,4,0.5,c,6,3,&report)==ZYNUM_NATIVE_VIEW_INVALID);
 assert(zynum_native_gemm_f64_tn_v1(contexts[1].data,UINT64_MAX,2,3,1,a,8,5,b,7,4,0.5,c,6,3,&report)==ZYNUM_NATIVE_OVERFLOW);
 assert(zynum_native_gemm_f64_tn_v1(contexts[1].data,2,2,3,1,NULL,8,5,b,7,4,0.5,c,6,3,&report)==ZYNUM_NATIVE_VIEW_INVALID);
 assert(zynum_native_gemm_f64_tn_v1(contexts[1].data,2,2,3,1,(const double*)((uintptr_t)a+1),8,5,b,7,4,0.5,c,6,3,&report)==ZYNUM_NATIVE_VIEW_INVALID);
 assert(zynum_native_gemm_f64_tn_v1(contexts[1].data,2,2,3,1,(const double*)(UINTPTR_MAX-7),8,5,b,7,4,0.5,c,6,3,&report)==ZYNUM_NATIVE_OVERFLOW);
 assert(zynum_native_gemm_f64_tn_v1(contexts[1].data,2,2,3,1,a,8,5,b,7,4,0.5,a,8,3,&report)==ZYNUM_NATIVE_VIEW_INVALID);
 assert(zynum_native_gemm_f64_tn_v1(contexts[1].data,2,2,3,1,a,8,5,b,7,4,0.5,b,7,3,&report)==ZYNUM_NATIVE_VIEW_INVALID);
 assert(zynum_native_gemm_f64_tn_v1(contexts[1].data,2,2,3,1,a,8,5,b,7,4,0.5,c,6,3,(zynum_native_report_v1*)c)==ZYNUM_NATIVE_INVALID_ARGUMENT);
 assert(zynum_native_gemm_f64_tn_v1(contexts[1].data,2,2,3,INFINITY,a,8,5,b,7,4,0.5,c,6,3,&report)==ZYNUM_NATIVE_INVALID_ARGUMENT);
 assert(zynum_native_gemm_f64_tn_v1(contexts[1].data,2,2,3,1,a,8,5,b,7,4,NAN,c,6,3,&report)==ZYNUM_NATIVE_INVALID_ARGUMENT);
 assert(zynum_native_gemm_f64_tn_v1(contexts[1].data,2,2,3,1,a,UINT64_MAX,5,b,7,4,0,c,6,3,&report)==ZYNUM_NATIVE_OVERFLOW);
 assert(zynum_native_gemm_f64_tn_v1(contexts[1].data,2,2,3,1,a,8,5,b,7,4,0,c,6,3,(zynum_native_report_v1*)a)==ZYNUM_NATIVE_INVALID_ARGUMENT);
 assert(zynum_native_gemm_f64_tn_v1(contexts[1].data,1,1,0,1,NULL,0,0,NULL,0,0,0,(double*)contexts[1].data,1,1,&report)==ZYNUM_NATIVE_INVALID_ARGUMENT);
 assert(zynum_native_gemm_f64_tn_v1(contexts[1].data,0,2,3,1,NULL,0,3,b,6,4,0,NULL,0,0,&report)==ZYNUM_NATIVE_VIEW_INVALID);
 equals(c,tn_saved,6);assert(atomic_load(&bulk_calls)==submitted);
 assert(zynum_native_gemm_f64_tn_v1(contexts[1].data,0,2,0,1,NULL,0,0,NULL,0,0,0,NULL,0,0,&report)==ZYNUM_NATIVE_OK);
 assert(report.output_state==0&&atomic_load(&bulk_calls)==submitted);
 assert(zynum_native_gemm_f64_tn_v1(contexts[1].data,2,2,3,1,a,8,5,b,7,4,0.5,c,6,3,&report)==ZYNUM_NATIVE_OK);
 const double tn_expected[6]={50.5,122.5,77,68.5,167.5,77};equals(c,tn_expected,6);
 assert(report.output_state==1&&report.submission_state==2&&report.callbacks_completed==2&&report.callbacks_started==2);
 assert(report.reserved[0]==0&&report.reserved[1]==0&&atomic_load(&bulk_calls)==submitted+2);
 assert(zynum_native_context_request_close_v1(contexts[1].data,&report)==ZYNUM_NATIVE_OK);
 assert(zynum_native_gemm_f64_tn_v1(contexts[1].data,0,2,0,1,NULL,0,0,NULL,0,0,0,NULL,0,0,&report)==ZYNUM_NATIVE_CONTEXT_CLOSING);
 assert(zynum_native_gemm_f64_tn_v1(contexts[1].data,2,2,3,1,a,7,5,b,7,4,0,c,6,3,&report)==ZYNUM_NATIVE_VIEW_INVALID);
 assert(zynum_native_context_progress_close_v1(contexts[1].data,&report)==ZYNUM_NATIVE_OK);
 assert(atomic_load(&bulk_calls)==6&&atomic_load(&callback_calls)>=6&&atomic_load(&caller_callbacks)==0);
 for(int i=0;i<3;++i)assert(real_api.scratch_release(matrices[i].lease)==HETERIS_OK);
 for(int i=0;i<2;++i)assert(real_api.scratch_release(contexts[i].lease)==HETERIS_OK);
 assert(heteris_cpu_runtime_destroy_v1(runtime)==HETERIS_OK);
#ifdef ZYNUM_NATIVE_DSO_CLIENT_TESTS
 for(int i=0;i<2;++i)assert(dlclose(clients[i])==0);
#endif
 printf("native Runtime C consumer passed; bulk=%u callbacks=%u caller_callbacks=%u\n",atomic_load(&bulk_calls),atomic_load(&callback_calls),atomic_load(&caller_callbacks));return 0;
}
