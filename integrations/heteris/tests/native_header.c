/* SPDX-License-Identifier: LGPL-3.0-or-later */
#include <stddef.h>
#include <zynum_native.h>
_Static_assert(sizeof(zynum_native_status)==4, "fixed i32 status");
_Static_assert(sizeof(zynum_native_report_v1)==72, "report v1 layout");
_Static_assert(offsetof(zynum_native_report_v1, callbacks_started)==40, "report count offset");
_Static_assert(sizeof(zynum_native_policy_v1)==32, "policy v1 layout");
_Static_assert(sizeof(zynum_native_context_requirements_v1)==40, "requirements v1 layout");
void zynum_native_header_type_check(void) {
 zynum_native_status (*query)(zynum_native_context_requirements_v1*)=zynum_native_context_query_v1;
 zynum_native_status (*init)(void*,uint64_t,const heteris_runtime_provider_v1*,const zynum_native_policy_v1*,zynum_native_report_v1*)=zynum_native_context_init_v1;
 zynum_native_status (*gemm)(void*,uint64_t,uint64_t,uint64_t,double,const double*,uint64_t,uint64_t,const double*,uint64_t,uint64_t,double,double*,uint64_t,uint64_t,zynum_native_report_v1*)=zynum_native_gemm_f64_nn_v1;
 (void)query;(void)init;(void)gemm;
}
