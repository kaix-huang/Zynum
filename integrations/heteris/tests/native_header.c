/* SPDX-License-Identifier: LGPL-3.0-or-later */
#include <stddef.h>
#ifdef ZYNUM_NATIVE_SPECTRAL_FIRST
#include <zynum_native_spectral.h>
#include <zynum_native.h>
#else
#include <zynum_native.h>
#include <zynum_native_spectral.h>
#endif
#include <zynum_native_spectral.h>
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
 zynum_native_status (*gemm_tn)(void*,uint64_t,uint64_t,uint64_t,double,const double*,uint64_t,uint64_t,const double*,uint64_t,uint64_t,double,double*,uint64_t,uint64_t,zynum_native_report_v1*)=zynum_native_gemm_f64_tn_v1;
 (void)query;(void)init;(void)gemm;(void)gemm_tn;
}

_Static_assert(sizeof(zynum_native_spectral_status)==4,"spectral fixed i32 status");
_Static_assert(sizeof(zynum_native_spectral_report_v1)==80,"spectral report x86-64 layout");
_Static_assert(offsetof(zynum_native_spectral_report_v1,order)==40,"spectral order offset");
_Static_assert(offsetof(zynum_native_spectral_report_v1,callback_status)==36,"spectral callback offset");
_Static_assert(offsetof(zynum_native_spectral_report_v1,reserved)==64,"spectral reserved offset");
_Static_assert(ZYNUM_NATIVE_SPECTRAL_NUMERICAL_FAILURE==10,"spectral status extent");
_Static_assert(ZYNUM_NATIVE_SPECTRAL_COMPLETE==7,"spectral phase extent");
void zynum_native_spectral_header_type_check(void) {
 zynum_native_spectral_status (*guard_query)(uint64_t*,uint64_t*)=zynum_native_spectral_guard_query_v1;
 zynum_native_spectral_status (*guard_init)(void*,uint64_t)=zynum_native_spectral_guard_init_v1;
 zynum_native_spectral_status (*work_query)(uint32_t,uint64_t,uint64_t,uint64_t,uint64_t*)=zynum_native_spectral_workspace_elements_v1;
 zynum_native_spectral_status (*series)(void*,zynum_native_spectral_apply_f64_v1,void*,uint64_t,uint64_t,uint64_t,const double*,uint64_t,uint64_t,const double*,uint64_t,uint64_t,double,double,double*,uint64_t,uint64_t,double*,uint64_t,void*,uint64_t,zynum_native_spectral_report_v1*)=zynum_native_chebyshev_series_f64_v1;
 zynum_native_spectral_status (*moments)(void*,zynum_native_spectral_apply_f64_v1,void*,uint64_t,uint64_t,uint64_t,const double*,uint64_t,uint64_t,uint64_t,double,double,double*,uint64_t,uint64_t,double*,uint64_t,void*,uint64_t,zynum_native_spectral_report_v1*)=zynum_native_chebyshev_moments_f64_v1;
 (void)guard_query;(void)guard_init;(void)work_query;(void)series;(void)moments;
}

typedef int32_t (*spectral_expected_callback)(void*,void*,uint64_t,uint64_t,const double*,uint64_t,uint64_t,double*,uint64_t,uint64_t);
_Static_assert(_Generic((zynum_native_spectral_apply_f64_v1)0,spectral_expected_callback:1,default:0),"spectral callback exact C type");
