/* SPDX-License-Identifier: LGPL-3.0-or-later */
#ifndef ZYNUM_NATIVE_H
#define ZYNUM_NATIVE_H
#include <stdint.h>
#include <heteris_runtime_provider.h>
#ifdef __cplusplus
extern "C" {
#endif
#define ZYNUM_NATIVE_ABI_VERSION 1u
typedef int32_t zynum_native_status;
enum {
 ZYNUM_NATIVE_OK=0, ZYNUM_NATIVE_INVALID_ARGUMENT=1, ZYNUM_NATIVE_UNSUPPORTED=2,
 ZYNUM_NATIVE_VIEW_INVALID=3, ZYNUM_NATIVE_CONTEXT_CLOSING=4, ZYNUM_NATIVE_BUDGET_EXCEEDED=5,
 ZYNUM_NATIVE_OUT_OF_MEMORY=6, ZYNUM_NATIVE_BUSY=7, ZYNUM_NATIVE_CLOSED=8,
 ZYNUM_NATIVE_WOULD_DEADLOCK=9, ZYNUM_NATIVE_PROVIDER_REJECTED=10,
 ZYNUM_NATIVE_EXECUTION_FAILURE=11, ZYNUM_NATIVE_CALLBACK_FAILURE=12,
 ZYNUM_NATIVE_OVERFLOW=13, ZYNUM_NATIVE_PROVIDER_CONTRACT_FAILURE=14,
 ZYNUM_NATIVE_RELEASE_FAILED=15, ZYNUM_NATIVE_COUNTER_OVERFLOW=16
};
/* Fixed ABI values for policy.algorithm and the report state fields. */
enum { ZYNUM_NATIVE_ALGORITHM_PORTABLE_GENERIC=1 };
enum {
 ZYNUM_NATIVE_RESOURCE_STAGE_NONE=0, ZYNUM_NATIVE_RESOURCE_STAGE_DESCRIPTOR=1,
 ZYNUM_NATIVE_RESOURCE_STAGE_RETAIN=2, ZYNUM_NATIVE_RESOURCE_STAGE_QUERY=3,
 ZYNUM_NATIVE_RESOURCE_STAGE_SCRATCH_ALLOCATE=4, ZYNUM_NATIVE_RESOURCE_STAGE_SUBMIT=5,
 ZYNUM_NATIVE_RESOURCE_STAGE_SCRATCH_RELEASE=6, ZYNUM_NATIVE_RESOURCE_STAGE_PROVIDER_RELEASE=7,
 ZYNUM_NATIVE_RESOURCE_STAGE_ROLLBACK_RELEASE=8
};
enum { ZYNUM_NATIVE_SUBMISSION_NOT_SUBMITTED=0, ZYNUM_NATIVE_SUBMISSION_REJECTED=1,
 ZYNUM_NATIVE_SUBMISSION_QUIESCENT=2 };
enum { ZYNUM_NATIVE_OUTPUT_UNCHANGED=0, ZYNUM_NATIVE_OUTPUT_COMPLETE=1,
 ZYNUM_NATIVE_OUTPUT_PARTIAL_OR_INVALID=2 };
enum { ZYNUM_NATIVE_CLOSE_OPEN=0, ZYNUM_NATIVE_CLOSE_CLOSING=1, ZYNUM_NATIVE_CLOSE_BUSY=2,
 ZYNUM_NATIVE_CLOSE_WAITING_RESOURCES=3, ZYNUM_NATIVE_CLOSE_RELEASE_FAILED=4, ZYNUM_NATIVE_CLOSE_CLOSED=5 };
typedef struct zynum_native_report_v1 {
 uint32_t abi_version, struct_size;
 uint32_t resource_stage, submission_state, output_state, close_state;
 int32_t provider_status, execution_status, callback_status, cleanup_status;
 uint64_t callbacks_started, callbacks_completed;
 uint64_t reserved[2];
} zynum_native_report_v1;
typedef struct zynum_native_policy_v1 {
 uint32_t abi_version, struct_size, algorithm, max_parallel_tasks;
 uint64_t reserved[2];
} zynum_native_policy_v1;
typedef struct zynum_native_context_requirements_v1 {
 uint32_t abi_version, struct_size;
 uint64_t bytes, alignment, reserved[2];
} zynum_native_context_requirements_v1;
/* Context storage is caller-owned, stable and initialized exactly once. Keep it
   and the implementation image live through close and every returning API call.
   Borrowed provider/policy descriptors must be valid for initialization. Report
   storage must not overlap Context, descriptors, or mutable mathematical data.
   No function constructs a Runtime, default Context, local pool or allocator. */
zynum_native_status zynum_native_context_query_v1(zynum_native_context_requirements_v1 *out);
zynum_native_status zynum_native_context_init_v1(void *storage, uint64_t bytes,
 const heteris_runtime_provider_v1 *provider, const zynum_native_policy_v1 *policy,
 zynum_native_report_v1 *report);
zynum_native_status zynum_native_context_request_close_v1(void *storage, zynum_native_report_v1 *report);
zynum_native_status zynum_native_context_progress_close_v1(void *storage, zynum_native_report_v1 *report);
/* FP64 column-major NN, checked primitive borrowed buffers. Implementation is
   supplied by the original Core numerical seam; never an adapter-local loop. */
zynum_native_status zynum_native_gemm_f64_nn_v1(void *storage,
 uint64_t m, uint64_t n, uint64_t k, double alpha,
 const double *a, uint64_t a_backing_elements, uint64_t lda,
 const double *b, uint64_t b_backing_elements, uint64_t ldb, double beta,
 double *c, uint64_t c_backing_elements, uint64_t ldc, zynum_native_report_v1 *report);
/* FP64 column-major TN: physical A(k,m), B(k,n), C(m,n). A requires
   lda >= k when nonempty. All other borrowing and validation rules match NN. */
zynum_native_status zynum_native_gemm_f64_tn_v1(void *storage,
 uint64_t m, uint64_t n, uint64_t k, double alpha,
 const double *a, uint64_t a_backing_elements, uint64_t lda,
 const double *b, uint64_t b_backing_elements, uint64_t ldb, double beta,
 double *c, uint64_t c_backing_elements, uint64_t ldc, zynum_native_report_v1 *report);
/* Contiguous periodic 3D FP64 grid, axis 2 fastest. Positive finite inverse
   squared spacings. No in-place operation: full declared X/Y spans are disjoint.
   Control ranges are checked before any report write; an uncheckable declared
   span or control overlap returns INVALID_ARGUMENT with report untouched.
   Any zero axis still validates and admits, but allocates/submits/writes nothing.
   Success describes completed IEEE arithmetic, not finite/scientific validity.
   A quiescent execution/callback failure may leave Y partial or invalid. */
zynum_native_status zynum_native_laplacian_periodic_3d_f64_v1(void *storage,
 uint64_t n0, uint64_t n1, uint64_t n2,
 const double *x, uint64_t x_backing_elements, double *y, uint64_t y_backing_elements,
 double inv_h2_0, double inv_h2_1, double inv_h2_2, zynum_native_report_v1 *report);
#ifdef ZYNUM_NATIVE_IDENTITY_TESTING
const void *zynum_native_core_identity_v1(void);
const void *zynum_native_execution_identity_v1(void);
#endif
#ifdef __cplusplus
}
#endif
#include "zynum_native_spectral.h"
#endif
