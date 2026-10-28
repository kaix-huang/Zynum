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
static int laplacian_death;
static unsigned death_phase, death_allocations, death_release_site;
static heteris_rp_token death_leases[5];
static heteris_status death_allocate(heteris_rp_token p,uint64_t bytes,uint64_t alignment,heteris_rp_scratch_v1 *out) {
 heteris_status status=real_api.scratch_allocate(p,bytes,alignment,out);
 if(status==HETERIS_OK){assert(death_allocations<5);death_leases[death_allocations++]=out->lease;}return status;
}
static heteris_status death_release(heteris_rp_token lease) {
 if(death_fault==DEATH_RELEASE&&death_allocations>=death_release_site&&lease.serial==death_leases[death_release_site-1].serial){fprintf(stderr,"%s_DEATH release_site=%u\n",laplacian_death?"LAPLACIAN":"TN",death_release_site);return -1942;}
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
  fprintf(stderr,"%s_DEATH phase=%u fault=%u\n",laplacian_death?"LAPLACIAN":"TN",death_phase,(unsigned)death_fault);
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
 if(laplacian_death)(void)zynum_native_laplacian_periodic_3d_f64_v1(context.data,1,2,2,a,4,output,4,2,3,4,&report);
 else (void)zynum_native_gemm_f64_tn_v1(context.data,2,2,2,1,a,4,2,a,4,2,1,output,4,2,&report);
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
 if(fault==DEATH_RELEASE)snprintf(marker,sizeof marker,"%s_DEATH release_site=%u",laplacian_death?"LAPLACIAN":"TN",release_site);
 else snprintf(marker,sizeof marker,"%s_DEATH phase=%u fault=%u",laplacian_death?"LAPLACIAN":"TN",phase,(unsigned)fault);
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
static void laplacian_death_cases(void) {
 laplacian_death=1;
 for(uint32_t limit=1;limit<=2;++limit){
  expect_tn_death(DEATH_MALFORMED,1,0,limit);
  expect_tn_death(DEATH_NONQUIESCENT,1,0,limit);
  for(unsigned site=1;site<=2;++site)expect_tn_death(DEATH_RELEASE,0,site,limit);
 }
 laplacian_death=0;
 printf("native periodic Laplacian fail-stop cases passed: 8 real-Runtime child processes\n");
}

static void laplacian_runtime_cases(void);
static void spectral_runtime_cases(void);
int main(int argc,char **argv) {
 tn_death_cases();
 laplacian_death_cases();
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
 laplacian_runtime_cases();
 spectral_runtime_cases();
 printf("native Runtime C consumer passed; bulk=%u callbacks=%u caller_callbacks=%u\n",atomic_load(&bulk_calls),atomic_load(&callback_calls),atomic_load(&caller_callbacks));return 0;
}

/* Defensive numeric range cases below do not attest live double objects at
   fabricated/control addresses. No such address is dereferenced by the API. */
static void laplacian_runtime_cases(void) {
 caller=pthread_self();
 heteris_rp_cpu_create_desc_v1 desc={0};desc.abi_version=1;desc.struct_size=sizeof desc;
 desc.compute_workers=2;desc.managed_thread_limit=3;desc.max_tasks=8;desc.max_dependencies=8;desc.host_bytes_limit=1u<<20;
 heteris_rp_token runtime;assert(heteris_cpu_runtime_create_v1(&desc,&runtime)==HETERIS_OK);
 heteris_runtime_provider_v1 provider;assert(heteris_cpu_runtime_export_provider_v1(runtime,&provider)==HETERIS_OK);
 real_api=*provider.api;heteris_runtime_provider_api_v1 proxy=real_api;proxy.cpu_bulk_sync=traced_bulk;provider.api=&proxy;
 zynum_native_context_requirements_v1 req;assert(zynum_native_context_query_v1(&req)==ZYNUM_NATIVE_OK);
 heteris_rp_scratch_v1 context,data[2];allocate(provider.provider,req.bytes,req.alignment,&context);
 for(unsigned i=0;i<2;++i)allocate(provider.provider,64,8,&data[i]);
 double *x=data[0].data,*y=data[1].data;
 for(unsigned i=0;i<8;++i){x[i]=i<6?(double)i:991;y[i]=773;}
 zynum_native_policy_v1 policy={1,sizeof policy,1,2,{0,0}};zynum_native_report_v1 report,sentinel;
 assert(zynum_native_context_init_v1(context.data,req.bytes,&provider,&policy,&report)==ZYNUM_NATIVE_OK);
 assert(real_api.provider_release(provider.provider)==HETERIS_OK);memset(&provider,0,sizeof provider);
 unsigned before=atomic_load(&bulk_calls);
 memset(&report,0x5a,sizeof report);sentinel=report;
 assert(zynum_native_laplacian_periodic_3d_f64_v1(NULL,1,2,3,x,8,y,8,2,3,4,&report)==ZYNUM_NATIVE_INVALID_ARGUMENT);
 assert(memcmp(&report,&sentinel,sizeof report)==0);
 assert(zynum_native_laplacian_periodic_3d_f64_v1(context.data,1,2,3,x,8,y,8,2,3,4,NULL)==ZYNUM_NATIVE_INVALID_ARGUMENT);
 assert(memcmp(&report,&sentinel,sizeof report)==0);
 assert(zynum_native_laplacian_periodic_3d_f64_v1((void*)((uintptr_t)context.data+1),1,2,3,x,8,y,8,2,3,4,&report)==ZYNUM_NATIVE_INVALID_ARGUMENT);
 assert(memcmp(&report,&sentinel,sizeof report)==0);
 assert(zynum_native_laplacian_periodic_3d_f64_v1(context.data,0,1,1,x,UINT64_MAX,y,8,2,3,4,&report)==ZYNUM_NATIVE_INVALID_ARGUMENT);
 assert(memcmp(&report,&sentinel,sizeof report)==0);
 assert(zynum_native_laplacian_periodic_3d_f64_v1(context.data,0,1,1,(const double*)(UINTPTR_MAX-7),2,y,8,2,3,4,&report)==ZYNUM_NATIVE_INVALID_ARGUMENT);
 assert(memcmp(&report,&sentinel,sizeof report)==0);
 assert(zynum_native_laplacian_periodic_3d_f64_v1(context.data,0,1,1,(const double*)&report,sizeof report/sizeof(double),y,8,2,3,4,&report)==ZYNUM_NATIVE_INVALID_ARGUMENT);
 assert(memcmp(&report,&sentinel,sizeof report)==0);
 assert(zynum_native_laplacian_periodic_3d_f64_v1(context.data,0,1,1,(const double*)context.data,1,y,8,2,3,4,&report)==ZYNUM_NATIVE_INVALID_ARGUMENT);
 assert(memcmp(&report,&sentinel,sizeof report)==0);
 unsigned char context_prefix[sizeof report];size_t prefix_bytes=req.bytes<sizeof context_prefix?(size_t)req.bytes:sizeof context_prefix;memcpy(context_prefix,context.data,prefix_bytes);
 assert(zynum_native_laplacian_periodic_3d_f64_v1(context.data,0,1,1,NULL,0,NULL,0,2,3,4,(zynum_native_report_v1*)context.data)==ZYNUM_NATIVE_INVALID_ARGUMENT);
 assert(memcmp(context_prefix,context.data,prefix_bytes)==0);
 assert(zynum_native_laplacian_periodic_3d_f64_v1(context.data,UINT64_MAX,2,1,x,8,y,8,NAN,3,4,&report)==ZYNUM_NATIVE_OVERFLOW);
 assert(report.output_state==ZYNUM_NATIVE_OUTPUT_UNCHANGED&&report.submission_state==ZYNUM_NATIVE_SUBMISSION_NOT_SUBMITTED);
 assert(zynum_native_laplacian_periodic_3d_f64_v1(context.data,UINT64_MAX,1,1,x,8,y,8,2,3,4,&report)==ZYNUM_NATIVE_OVERFLOW);
 assert(zynum_native_laplacian_periodic_3d_f64_v1(context.data,0,1,1,NULL,1,y,8,2,3,4,&report)==ZYNUM_NATIVE_VIEW_INVALID);
 assert(zynum_native_laplacian_periodic_3d_f64_v1(context.data,0,1,1,(const double*)((uintptr_t)x+1),0,y,8,2,3,4,&report)==ZYNUM_NATIVE_VIEW_INVALID);
 assert(zynum_native_laplacian_periodic_3d_f64_v1(context.data,1,2,3,x,5,y,8,NAN,3,4,&report)==ZYNUM_NATIVE_VIEW_INVALID);
 assert(zynum_native_laplacian_periodic_3d_f64_v1(context.data,1,2,3,x,8,y,5,2,3,4,&report)==ZYNUM_NATIVE_VIEW_INVALID);
 const double invalid_h[5]={0,-0.0,-1,NAN,INFINITY};
 for(unsigned axis=0;axis<3;++axis)for(unsigned i=0;i<5;++i){
  double h[3]={2,3,4};h[axis]=invalid_h[i];
  assert(zynum_native_laplacian_periodic_3d_f64_v1(context.data,0,1,1,x,8,y,8,h[0],h[1],h[2],&report)==ZYNUM_NATIVE_INVALID_ARGUMENT);
 }
 assert(zynum_native_laplacian_periodic_3d_f64_v1(context.data,0,1,1,x,8,x+7,1,2,3,4,&report)==ZYNUM_NATIVE_VIEW_INVALID);
 assert(atomic_load(&bulk_calls)==before);
 for(unsigned i=0;i<8;++i)assert(y[i]==773);
 for(unsigned mask=0;mask<7;++mask){
  assert(zynum_native_laplacian_periodic_3d_f64_v1(context.data,mask&1?1:0,mask&2?1:0,mask&4?1:0,NULL,0,NULL,0,2,3,4,&report)==ZYNUM_NATIVE_OK);
  assert(report.output_state==0&&report.submission_state==0&&report.callbacks_started==0&&report.callbacks_completed==0);
 }
 assert(zynum_native_laplacian_periodic_3d_f64_v1(context.data,UINT64_MAX,0,UINT64_MAX,NULL,0,NULL,0,2,3,4,&report)==ZYNUM_NATIVE_OK);
 assert(atomic_load(&bulk_calls)==before);
 assert(zynum_native_laplacian_periodic_3d_f64_v1(context.data,1,2,3,x,8,y,8,2,3,4,&report)==ZYNUM_NATIVE_OK);
 const double expected[6]={30,18,6,-6,-18,-30};equals(y,expected,6);
 assert(y[6]==773&&y[7]==773&&x[6]==991&&x[7]==991);
 assert(report.output_state==1&&report.submission_state==2&&report.callbacks_started==report.callbacks_completed&&report.callbacks_completed>0);
 assert(report.reserved[0]==0&&report.reserved[1]==0&&atomic_load(&bulk_calls)==before+1);
 x[0]=INFINITY;
 assert(zynum_native_laplacian_periodic_3d_f64_v1(context.data,1,1,1,x,8,y,8,2,3,4,&report)==ZYNUM_NATIVE_OK);
 assert(isnan(y[0])&&report.output_state==1&&atomic_load(&bulk_calls)==before+2);
 assert(heteris_cpu_runtime_destroy_v1(runtime)==HETERIS_BUSY);
 assert(zynum_native_context_request_close_v1(context.data,&report)==ZYNUM_NATIVE_OK);
 assert(zynum_native_laplacian_periodic_3d_f64_v1(context.data,0,1,1,NULL,0,NULL,0,-1,3,4,&report)==ZYNUM_NATIVE_INVALID_ARGUMENT);
 assert(zynum_native_laplacian_periodic_3d_f64_v1(context.data,0,1,1,NULL,0,NULL,0,2,3,4,&report)==ZYNUM_NATIVE_CONTEXT_CLOSING);
 assert(zynum_native_context_progress_close_v1(context.data,&report)==ZYNUM_NATIVE_OK);
 for(unsigned i=0;i<2;++i)assert(real_api.scratch_release(data[i].lease)==HETERIS_OK);
 assert(real_api.scratch_release(context.lease)==HETERIS_OK);
 assert(heteris_cpu_runtime_destroy_v1(runtime)==HETERIS_OK);
 printf("native periodic Laplacian C validation and direct output cases passed\n");
}


/* Spectral cases use the actual formal provider for Context, passive guard,
   payload and workspace storage. Invalid numeric-address probes return before
   dereference and do not attest provenance/lifetime of fabricated objects. */
struct spectral_case;
struct spectral_state {
 unsigned calls,fail_at,nan_at,close_at,reenter,nested,child;
 int32_t code;
 void *guard,*nested_guard;uint64_t guard_bytes;
 double *work,*nested_work;
 zynum_native_report_v1 child_report;
 int32_t child_status;
 unsigned close_busy,after_child,hold;
 _Atomic unsigned hold_entered,hold_release;
};
static int32_t spectral_identity(void *raw,void *context,uint64_t n,uint64_t w,const double *x,uint64_t xb,uint64_t ldx,double *y,uint64_t yb,uint64_t ldy) {
 (void)raw;(void)context;assert(ldx==n&&ldy==n&&xb==n*w&&yb==n*w);
 if(n==0||w==0){assert(x==NULL&&y==NULL&&xb==0&&yb==0);return 0;}
 for(uint64_t j=0;j<w;++j)for(uint64_t i=0;i<n;++i)y[j*ldy+i]=x[j*ldx+i];
 return 0;
}
static int32_t spectral_apply(void *raw,void *context,uint64_t n,uint64_t w,const double *x,uint64_t xb,uint64_t ldx,double *y,uint64_t yb,uint64_t ldy) {
 struct spectral_state *state=raw;++state->calls;
 assert(ldx==n&&ldy==n&&xb==n*w&&yb==n*w);
 if(state->hold){atomic_store_explicit(&state->hold_entered,1,memory_order_release);while(!atomic_load_explicit(&state->hold_release,memory_order_acquire)){} }
 if(state->reenter){
  zynum_native_spectral_report_v1 nested_report;const double c0=1;double nested_output=773;
  assert(zynum_native_chebyshev_series_f64_v1(context,spectral_identity,NULL,0,0,0,NULL,0,UINT64_MAX,&c0,1,1,-1,1,&nested_output,1,UINT64_MAX,state->work,0,state->guard,state->guard_bytes,&nested_report)==ZYNUM_NATIVE_SPECTRAL_BUSY);
  assert(nested_report.phase==0&&nested_report.admission_attempted==0&&nested_report.callback_returned==0&&nested_output==773);
 }
 if(state->nested){
  zynum_native_spectral_report_v1 nested_report;const double c0=2,input=3;double output=773;
  assert(zynum_native_chebyshev_series_f64_v1(context,spectral_identity,NULL,0,1,1,&input,1,1,&c0,1,1,-1,1,&output,1,1,state->nested_work,4,state->nested_guard,state->guard_bytes,&nested_report)==ZYNUM_NATIVE_SPECTRAL_OK);
  assert(output==6&&nested_report.admission_succeeded==1&&nested_report.callbacks_returned==0);
 }
 if(!state->after_child&&state->close_at==state->calls){
  assert(zynum_native_context_request_close_v1(context,&state->child_report)==ZYNUM_NATIVE_OK);
  assert(zynum_native_context_progress_close_v1(context,&state->child_report)==ZYNUM_NATIVE_BUSY);
  state->close_busy=1;
 }
 if(state->child){
  const double factor=1;
  state->child_status=zynum_native_gemm_f64_nn_v1(context,n,w,1,1,x,xb,ldx,&factor,1,1,0,y,yb,ldy,&state->child_report);
  if(state->after_child&&state->child_status==ZYNUM_NATIVE_OK&&state->close_at==state->calls){
   assert(zynum_native_context_request_close_v1(context,&state->child_report)==ZYNUM_NATIVE_OK);
   assert(zynum_native_context_progress_close_v1(context,&state->child_report)==ZYNUM_NATIVE_BUSY);state->close_busy=1;
  }
  return state->child_status;
 }
 if(state->calls==state->fail_at)return state->code;
 if(n==0||w==0){assert(x==NULL&&y==NULL&&xb==0&&yb==0);return 0;}
 for(uint64_t j=0;j<w;++j)for(uint64_t i=0;i<n;++i)y[j*ldy+i]=x[j*ldx+i];
 if(state->calls==state->nan_at)y[0]=NAN;
 return 0;
}
struct spectral_case {
 void *context,*guard;uint64_t guard_bytes;
 struct spectral_state *state;
 uint64_t n,w,q,xb,ldx,cb,yb,ldy,wb;
 const double *x,*coeff;double *y,*work;
 double lower,upper;
 zynum_native_spectral_apply_f64_v1 apply;
};
static int32_t spectral_series(struct spectral_case *a,zynum_native_spectral_report_v1 *report) {
 return zynum_native_chebyshev_series_f64_v1(a->context,a->apply,a->state,a->state?sizeof *a->state:0,a->n,a->w,a->x,a->xb,a->ldx,a->coeff,a->cb,a->q,a->lower,a->upper,a->y,a->yb,a->ldy,a->work,a->wb,a->guard,a->guard_bytes,report);
}
static int32_t spectral_moments(struct spectral_case *a,zynum_native_spectral_report_v1 *report) {
 return zynum_native_chebyshev_moments_f64_v1(a->context,a->apply,a->state,a->state?sizeof *a->state:0,a->n,a->w,a->x,a->xb,a->ldx,a->q,a->lower,a->upper,a->y,a->yb,a->ldy,a->work,a->wb,a->guard,a->guard_bytes,report);
}
static void spectral_bad(struct spectral_case *a,int32_t expected) {
 zynum_native_spectral_report_v1 report,saved;memset(&report,0x5a,sizeof report);saved=report;
 assert(spectral_series(a,&report)==expected);
 if(expected==ZYNUM_NATIVE_SPECTRAL_INVALID_CONTROL)assert(memcmp(&report,&saved,sizeof report)==0);
 else {
  assert(report.abi_version==1&&report.struct_size==sizeof report&&report.phase==ZYNUM_NATIVE_SPECTRAL_VALIDATION&&report.output_state==0);
  assert(report.admission_attempted==0&&report.admission_succeeded==0&&report.order_valid==0&&report.callback_returned==0&&report.callbacks_returned==0);
  assert(report.admission_status==0&&report.callback_status==0&&report.order==0&&report.callback_order==0&&report.reserved[0]==0&&report.reserved[1]==0);
 }
}
struct spectral_worker { struct spectral_case args; int32_t q1,q2; zynum_native_spectral_report_v1 report; pthread_t owner; };
static int32_t spectral_worker_run(void *raw,uint64_t begin,uint64_t end) {
 struct spectral_worker *w=raw;assert(begin==0&&end==1);assert(!pthread_equal(pthread_self(),w->owner));
 w->args.q=1;w->q1=spectral_series(&w->args,&w->report);
 w->args.q=2;w->args.state->child=1;w->q2=spectral_series(&w->args,&w->report);
 return 0;
}
struct spectral_thread {struct spectral_case args;zynum_native_spectral_report_v1 report;int32_t status;};
static void *spectral_thread_run(void *raw){struct spectral_thread *t=raw;t->status=spectral_series(&t->args,&t->report);return NULL;}
static void spectral_runtime_cases(void) {
 uint64_t gb=777,ga=777,required=777;
 assert(zynum_native_spectral_guard_query_v1(NULL,&ga)==ZYNUM_NATIVE_SPECTRAL_INVALID_CONTROL&&ga==777);
 assert(zynum_native_spectral_guard_query_v1(&gb,&gb)==ZYNUM_NATIVE_SPECTRAL_INVALID_CONTROL&&gb==777);
 assert(zynum_native_spectral_guard_query_v1(&gb,&ga)==ZYNUM_NATIVE_SPECTRAL_OK&&gb>0&&ga>0&&gb+16<=64);
 assert(zynum_native_spectral_workspace_elements_v1(0,1,1,1,&required)==ZYNUM_NATIVE_SPECTRAL_INVALID_ARGUMENT&&required==777);
 assert(zynum_native_spectral_workspace_elements_v1(1,1,1,0,&required)==ZYNUM_NATIVE_SPECTRAL_INVALID_ARGUMENT&&required==777);
 assert(zynum_native_spectral_workspace_elements_v1(1,0,UINT64_MAX,1,&required)==ZYNUM_NATIVE_SPECTRAL_OVERFLOW&&required==777);
 assert(zynum_native_spectral_workspace_elements_v1(1,UINT64_MAX,0,1,&required)==ZYNUM_NATIVE_SPECTRAL_OVERFLOW&&required==777);
 assert(zynum_native_spectral_workspace_elements_v1(2,0,0,UINT64_MAX,&required)==ZYNUM_NATIVE_SPECTRAL_OVERFLOW&&required==777);
 assert(zynum_native_spectral_workspace_elements_v1(1,3,2,3,&required)==ZYNUM_NATIVE_SPECTRAL_OK&&required==24);
 assert(zynum_native_spectral_workspace_elements_v1(2,3,2,3,&required)==ZYNUM_NATIVE_SPECTRAL_OK&&required==24);
 heteris_rp_cpu_create_desc_v1 desc={0};desc.abi_version=1;desc.struct_size=sizeof desc;desc.compute_workers=2;desc.managed_thread_limit=3;desc.max_tasks=8;desc.max_dependencies=8;desc.host_bytes_limit=1u<<20;
 heteris_rp_token runtime;assert(heteris_cpu_runtime_create_v1(&desc,&runtime)==HETERIS_OK);
 heteris_runtime_provider_v1 provider;assert(heteris_cpu_runtime_export_provider_v1(runtime,&provider)==HETERIS_OK);
 heteris_runtime_provider_api_v1 api=*provider.api;
 zynum_native_context_requirements_v1 req;assert(zynum_native_context_query_v1(&req)==ZYNUM_NATIVE_OK);
 heteris_rp_scratch_v1 memory[8];
 assert(api.scratch_allocate(provider.provider,req.bytes,req.alignment,&memory[0])==HETERIS_OK);
 for(unsigned i=1;i<=2;++i)assert(api.scratch_allocate(provider.provider,gb+16,ga,&memory[i])==HETERIS_OK);
 for(unsigned i=3;i<8;++i)assert(api.scratch_allocate(provider.provider,128*sizeof(double),8,&memory[i])==HETERIS_OK);
 void *context=memory[0].data;double *x=memory[3].data,*co=memory[4].data,*y=memory[5].data,*work=memory[6].data,*nested_work=memory[7].data;
 unsigned char guard_before[64];
 for(unsigned i=1;i<=2;++i){
  memset(memory[i].data,0xa5,(size_t)gb+16);memcpy(guard_before,memory[i].data,(size_t)gb+16);
  assert(zynum_native_spectral_guard_init_v1(memory[i].data,gb-1)==ZYNUM_NATIVE_SPECTRAL_INVALID_CONTROL);
  assert(memcmp(guard_before,memory[i].data,(size_t)gb+16)==0);
  assert(zynum_native_spectral_guard_init_v1(memory[i].data,gb+16)==ZYNUM_NATIVE_SPECTRAL_OK);
  for(uint64_t k=gb;k<gb+16;++k)assert(((unsigned char*)memory[i].data)[k]==0xa5);
 }
 zynum_native_policy_v1 policy={1,sizeof policy,1,1,{0,0}};zynum_native_report_v1 child_report;
 assert(zynum_native_context_init_v1(context,req.bytes,&provider,&policy,&child_report)==ZYNUM_NATIVE_OK);
 assert(api.provider_release(provider.provider)==HETERIS_OK);
 for(unsigned i=0;i<128;++i){x[i]=NAN;co[i]=NAN;y[i]=773;work[i]=991;nested_work[i]=991;}
 x[0]=1;x[1]=2;x[4]=3;x[5]=4;co[0]=1;co[1]=2;co[2]=3;
 struct spectral_state state={0};state.guard=memory[1].data;state.nested_guard=memory[2].data;state.guard_bytes=gb+16;state.work=work;state.nested_work=nested_work;
 const struct spectral_case base={context,memory[1].data,gb+16,&state,2,2,3,6,4,3,7,4,128,x,co,y,work,-1,1,spectral_apply};
 struct spectral_case a=base;zynum_native_spectral_report_v1 report;
 /* Strict preflight wins over logical overflow, invalid callback and BUSY. */
 a.context=NULL;a.q=UINT64_MAX;spectral_bad(&a,ZYNUM_NATIVE_SPECTRAL_INVALID_CONTROL);
 a=base;a.x=NULL;a.xb=1;spectral_bad(&a,ZYNUM_NATIVE_SPECTRAL_INVALID_CONTROL);
 a=base;a.xb=UINT64_MAX;spectral_bad(&a,ZYNUM_NATIVE_SPECTRAL_INVALID_CONTROL);
 a=base;a.x=(const double*)(UINTPTR_MAX-7);a.xb=2;spectral_bad(&a,ZYNUM_NATIVE_SPECTRAL_INVALID_CONTROL);
 a=base;a.x=(const double*)&state;a.xb=sizeof state/8;spectral_bad(&a,ZYNUM_NATIVE_SPECTRAL_INVALID_CONTROL);
 a=base;a.guard_bytes=gb-1;spectral_bad(&a,ZYNUM_NATIVE_SPECTRAL_INVALID_CONTROL);
 a=base;a.guard=(void*)((uintptr_t)memory[1].data+1);spectral_bad(&a,ZYNUM_NATIVE_SPECTRAL_INVALID_CONTROL);
 a=base;a.guard_bytes=UINT64_MAX;spectral_bad(&a,ZYNUM_NATIVE_SPECTRAL_INVALID_CONTROL);
 memset(&report,0x5a,sizeof report);zynum_native_spectral_report_v1 saved_report=report;
 assert(zynum_native_chebyshev_series_f64_v1(context,spectral_apply,&state,0,2,2,x,6,4,co,3,3,-1,1,y,7,4,work,128,memory[1].data,gb+16,&report)==ZYNUM_NATIVE_SPECTRAL_INVALID_CONTROL);
 assert(memcmp(&report,&saved_report,sizeof report)==0);
 a=base;a.apply=NULL;spectral_bad(&a,ZYNUM_NATIVE_SPECTRAL_INVALID_ARGUMENT);
 a=base;a.q=0;spectral_bad(&a,ZYNUM_NATIVE_SPECTRAL_INVALID_ARGUMENT);
 a=base;a.lower=NAN;spectral_bad(&a,ZYNUM_NATIVE_SPECTRAL_INVALID_ARGUMENT);
 a=base;a.upper=INFINITY;spectral_bad(&a,ZYNUM_NATIVE_SPECTRAL_INVALID_ARGUMENT);
 a=base;a.lower=1;spectral_bad(&a,ZYNUM_NATIVE_SPECTRAL_INVALID_ARGUMENT);
 a=base;a.n=UINT64_MAX;a.w=0;spectral_bad(&a,ZYNUM_NATIVE_SPECTRAL_OVERFLOW);
 a=base;a.n=0;a.w=UINT64_MAX;spectral_bad(&a,ZYNUM_NATIVE_SPECTRAL_OVERFLOW);
 a=base;a.q=UINT64_MAX;spectral_bad(&a,ZYNUM_NATIVE_SPECTRAL_OVERFLOW);
 a=base;a.ldx=UINT64_MAX;spectral_bad(&a,ZYNUM_NATIVE_SPECTRAL_OVERFLOW);
 a=base;a.ldx=1;spectral_bad(&a,ZYNUM_NATIVE_SPECTRAL_VIEW_INVALID);
 a=base;a.xb=5;spectral_bad(&a,ZYNUM_NATIVE_SPECTRAL_VIEW_INVALID);
 a=base;a.yb=5;spectral_bad(&a,ZYNUM_NATIVE_SPECTRAL_VIEW_INVALID);
 a=base;a.cb=2;spectral_bad(&a,ZYNUM_NATIVE_SPECTRAL_VIEW_INVALID);
 a=base;a.wb=15;spectral_bad(&a,ZYNUM_NATIVE_SPECTRAL_VIEW_INVALID);
 a=base;a.x=(const double*)((uintptr_t)x+1);spectral_bad(&a,ZYNUM_NATIVE_SPECTRAL_VIEW_INVALID);
 a=base;a.n=0;a.w=0;a.x=(const double*)((uintptr_t)x+1);a.xb=0;a.y=NULL;a.yb=0;a.work=NULL;a.wb=0;spectral_bad(&a,ZYNUM_NATIVE_SPECTRAL_VIEW_INVALID);
 a=base;a.n=0;a.w=0;a.xb=128;a.work=x+127;a.wb=1;spectral_bad(&a,ZYNUM_NATIVE_SPECTRAL_ALIASING);
 a=base;a.work=x;a.wb=128;spectral_bad(&a,ZYNUM_NATIVE_SPECTRAL_ALIASING);
 a=base;a.y=(double*)co;spectral_bad(&a,ZYNUM_NATIVE_SPECTRAL_ALIASING);
 co[0]=NAN;a=base;spectral_bad(&a,ZYNUM_NATIVE_SPECTRAL_NONFINITE_INPUT);co[0]=1;
 x[0]=INFINITY;spectral_bad(&a,ZYNUM_NATIVE_SPECTRAL_NONFINITE_INPUT);x[0]=1;
 assert(state.calls==0);for(unsigned i=0;i<128;++i){assert(y[i]==773);assert(work[i]==991);}
 /* Identity H, [-1,1]: every T_k(H) is I; independent expected sum=6. */
 a=base;state.reenter=1;state.nested=1;
 assert(spectral_series(&a,&report)==ZYNUM_NATIVE_SPECTRAL_OK);
 assert(y[0]==6&&y[1]==12&&y[4]==18&&y[5]==24&&y[2]==773&&y[3]==773&&y[6]==773);
 assert(state.calls==2&&report.phase==7&&report.order==2&&report.callback_order==2&&report.callbacks_returned==2&&report.callback_status==0&&report.output_state==1);
 state.calls=0;state.reenter=0;state.nested=0;
 assert(spectral_moments(&a,&report)==ZYNUM_NATIVE_SPECTRAL_OK);
 assert(y[0]==5&&y[1]==5&&y[2]==5&&y[4]==25&&y[5]==25&&y[6]==25&&y[3]==773&&y[7]==773);
 assert(state.calls==2&&report.phase==7&&report.output_state==1);
 for(unsigned i=18;i<128;++i)assert(work[i]==991);
 /* Staging permits partial X/output overlap and readonly coefficient/X sharing. */
 double saved_x[128];memcpy(saved_x,x,sizeof saved_x);a=base;a.y=x+1;a.yb=127;
 state.calls=0;assert(spectral_series(&a,&report)==ZYNUM_NATIVE_SPECTRAL_OK);
 assert(x[0]==1&&x[1]==6&&x[2]==12&&x[5]==18&&x[6]==24);memcpy(x,saved_x,sizeof saved_x);
 a=base;a.y=x;a.yb=128;state.calls=0;
 assert(spectral_moments(&a,&report)==ZYNUM_NATIVE_SPECTRAL_OK);
 assert(x[0]==5&&x[1]==5&&x[2]==5&&x[4]==25&&x[5]==25&&x[6]==25);memcpy(x,saved_x,sizeof saved_x);
 a=base;x[2]=3;a.coeff=x;state.calls=0;
 assert(spectral_series(&a,&report)==ZYNUM_NATIVE_SPECTRAL_OK&&y[0]==6&&y[1]==12&&y[4]==18&&y[5]==24);
 memcpy(x,saved_x,sizeof saved_x);a=base;
 double saved_y[128];memcpy(saved_y,y,sizeof saved_y);
 const int32_t codes[]={INT32_MIN,-1,1,8,10,INT32_MAX};
 for(unsigned i=0;i<sizeof codes/sizeof codes[0];++i){
  state.calls=0;state.fail_at=2;state.code=codes[i];
  assert(spectral_series(&a,&report)==ZYNUM_NATIVE_SPECTRAL_CALLBACK_FAILURE);
  assert(report.phase==ZYNUM_NATIVE_SPECTRAL_APPLY&&report.order==2&&report.callback_order==2&&report.callback_status==codes[i]&&report.callbacks_returned==2&&report.admission_status==0&&report.output_state==0);
  assert(memcmp(saved_y,y,sizeof saved_y)==0);
 }
 state.calls=0;state.fail_at=0;state.nan_at=1;
 assert(spectral_series(&a,&report)==ZYNUM_NATIVE_SPECTRAL_NUMERICAL_FAILURE);
 assert(report.phase==ZYNUM_NATIVE_SPECTRAL_RECURRENCE&&report.callback_returned==1&&report.callback_status==0&&report.callbacks_returned==1);
 assert(memcmp(saved_y,y,sizeof saved_y)==0);state.nan_at=0;
 a=base;a.n=1;a.w=1;a.xb=1;a.yb=1;a.ldx=UINT64_MAX;a.ldy=UINT64_MAX;a.q=1;
 state.calls=0;co[0]=2;x[0]=3;
 assert(spectral_series(&a,&report)==ZYNUM_NATIVE_SPECTRAL_OK&&y[0]==6&&state.calls==0&&report.order_valid==1&&report.order==0&&report.callback_returned==0);
 x[0]=2;co[0]=0x1.fffffffffffffp1023;
 assert(spectral_series(&a,&report)==ZYNUM_NATIVE_SPECTRAL_NUMERICAL_FAILURE&&report.phase==ZYNUM_NATIVE_SPECTRAL_INITIALIZE&&report.order==0&&report.callback_returned==0&&y[0]==6);
 co[0]=1;x[0]=0x1.fffffffffffffp1023;
 assert(spectral_moments(&a,&report)==ZYNUM_NATIVE_SPECTRAL_NUMERICAL_FAILURE&&report.phase==ZYNUM_NATIVE_SPECTRAL_MOMENT&&report.callback_returned==0&&y[0]==6);
 /* A genuinely concurrent same-guard call is BUSY, while malformed control
    still wins before CAS. The first callback is held on its API caller thread. */
 a=base;x[0]=1;co[0]=1;state.calls=0;state.hold=1;
 atomic_store(&state.hold_entered,0);atomic_store(&state.hold_release,0);
 struct spectral_thread active={0};active.args=a;pthread_t thread;assert(pthread_create(&thread,NULL,spectral_thread_run,&active)==0);
 while(!atomic_load_explicit(&state.hold_entered,memory_order_acquire)){}
 assert(spectral_series(&a,&report)==ZYNUM_NATIVE_SPECTRAL_BUSY&&report.admission_attempted==0&&report.callback_returned==0);
 struct spectral_case malformed=a;malformed.apply=NULL;malformed.q=0;malformed.upper=NAN;
 spectral_bad(&malformed,ZYNUM_NATIVE_SPECTRAL_BUSY);
 malformed=a;malformed.context=NULL;spectral_bad(&malformed,ZYNUM_NATIVE_SPECTRAL_INVALID_CONTROL);
 atomic_store_explicit(&state.hold_release,1,memory_order_release);assert(pthread_join(thread,NULL)==0);
 assert(active.status==ZYNUM_NATIVE_SPECTRAL_OK);state.hold=0;state.calls=0;
 /* Fully validated empty schedules and their report validity differ by kind. */
 a=base;a.n=0;a.w=2;a.x=NULL;a.xb=0;a.ldx=UINT64_MAX;a.yb=7;a.q=3;state.calls=0;
 assert(spectral_moments(&a,&report)==ZYNUM_NATIVE_SPECTRAL_OK&&state.calls==2&&report.order==2&&report.output_state==1);
 assert(y[0]==0&&y[1]==0&&y[2]==0&&y[4]==0&&y[5]==0&&y[6]==0);
 a.y=NULL;a.yb=0;a.ldy=UINT64_MAX;state.calls=0;
 assert(spectral_series(&a,&report)==ZYNUM_NATIVE_SPECTRAL_OK&&state.calls==2&&report.output_state==0&&report.order==2);
 a=base;a.w=0;a.x=NULL;a.xb=0;a.y=NULL;a.yb=0;a.ldx=UINT64_MAX;a.ldy=UINT64_MAX;a.work=NULL;a.wb=0;state.calls=0;
 assert(spectral_series(&a,&report)==ZYNUM_NATIVE_SPECTRAL_OK&&state.calls==2&&report.callback_order==2&&report.output_state==0);
 state.calls=0;
 assert(spectral_moments(&a,&report)==ZYNUM_NATIVE_SPECTRAL_OK&&state.calls==0&&report.order_valid==0&&report.callback_returned==0&&report.phase==7&&report.output_state==0);
 state.calls=0;state.fail_at=1;state.code=INT32_MIN;
 assert(spectral_series(&a,&report)==ZYNUM_NATIVE_SPECTRAL_CALLBACK_FAILURE&&state.calls==1&&report.callback_status==INT32_MIN&&report.callbacks_returned==1&&report.output_state==0);
 state.fail_at=0;
 /* Real Runtime worker: Q1 is permitted; actual nested bulk is rejected. */
 a=base;a.n=1;a.w=1;a.q=1;a.xb=1;a.yb=1;a.ldx=1;a.ldy=1;x[0]=1;co[0]=1;co[1]=1;state.calls=0;
 struct spectral_worker worker={0};worker.args=a;worker.owner=pthread_self();heteris_rp_bulk_result_v1 outcome;
 assert(api.cpu_bulk_sync(provider.provider,0,1,1,1,spectral_worker_run,&worker,&outcome)==HETERIS_OK);
 assert(worker.q1==ZYNUM_NATIVE_SPECTRAL_OK&&worker.q2==ZYNUM_NATIVE_SPECTRAL_CALLBACK_FAILURE);
 assert(state.child_status==ZYNUM_NATIVE_WOULD_DEADLOCK&&state.child_report.provider_status==HETERIS_WOULD_DEADLOCK);
 assert(worker.report.admission_succeeded==1&&worker.report.callback_status==ZYNUM_NATIVE_WOULD_DEADLOCK&&worker.report.phase==ZYNUM_NATIVE_SPECTRAL_APPLY);
 /* Closing after the first successful child invalidates the next child's
    independent admission, while the successful outer admission stays valid. */
 heteris_rp_scratch_v1 extra_context;assert(api.scratch_allocate(provider.provider,req.bytes,req.alignment,&extra_context)==HETERIS_OK);
 assert(zynum_native_context_init_v1(extra_context.data,req.bytes,&provider,&policy,&child_report)==ZYNUM_NATIVE_OK);
 struct spectral_case between=base;between.context=extra_context.data;between.n=1;between.w=1;between.xb=1;between.yb=1;between.ldx=1;between.ldy=1;
 state.calls=0;state.child=1;state.close_at=1;state.after_child=1;state.close_busy=0;
 assert(spectral_series(&between,&report)==ZYNUM_NATIVE_SPECTRAL_CALLBACK_FAILURE&&state.calls==2&&state.close_busy==1);
 assert(report.admission_succeeded==1&&report.admission_status==0&&report.order==2&&report.callback_order==2&&report.callbacks_returned==2&&report.callback_status==ZYNUM_NATIVE_CONTEXT_CLOSING);
 assert(state.child_status==ZYNUM_NATIVE_CONTEXT_CLOSING);
 assert(zynum_native_context_progress_close_v1(extra_context.data,&child_report)==ZYNUM_NATIVE_OK);
 assert(api.scratch_release(extra_context.lease)==HETERIS_OK);state.after_child=0;
 state.child=0;state.calls=0;state.close_at=2;a=base;a.q=3;
 assert(spectral_series(&a,&report)==ZYNUM_NATIVE_SPECTRAL_OK&&state.close_busy&&report.phase==7&&report.admission_succeeded==1);
 state.calls=0;
 assert(spectral_series(&a,&report)==ZYNUM_NATIVE_SPECTRAL_CONTEXT_FAILURE&&report.phase==ZYNUM_NATIVE_SPECTRAL_ADMISSION&&report.admission_attempted==1&&report.admission_succeeded==0&&report.admission_status==ZYNUM_NATIVE_CONTEXT_CLOSING&&report.callback_returned==0);
 assert(zynum_native_context_progress_close_v1(context,&child_report)==ZYNUM_NATIVE_OK);
 for(unsigned i=1;i<=2;++i)for(uint64_t k=gb;k<gb+16;++k)assert(((unsigned char*)memory[i].data)[k]==0xa5);
 for(unsigned i=0;i<8;++i)assert(api.scratch_release(memory[i].lease)==HETERIS_OK);
 assert(heteris_cpu_runtime_destroy_v1(runtime)==HETERIS_OK);
 printf("native spectral C controls, guard, reports, empty, worker and close cases passed\n");
}
