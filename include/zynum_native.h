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
/* Complex128 row-major CN: C = alpha*adjoint(A)*B + beta*C, with physical
   A(k,m), B(k,n), C(m,n). Complex values use Heteris Data's interleaved
   double(real,imag) representation (16 bytes, alignment 8). All capacities and
   leading dimensions count complex elements, never doubles or bytes.
   alpha/beta are finite primitive components. Complete declared spans must be
   representable and C disjoint from A/B; A and B may overlap each other.
   Context/report and complete declared spans must be disjoint. Every metadata
   check precedes report writes and borrowing. Unprovable control/span metadata
   returns INVALID_ARGUMENT with report untouched. A zero m/n validates and
   admits without allocation/submission/writes. alpha=0 or k=0 does not read
   A/B; beta=0 does not read C. Every nonempty output uses the explicit provider.
   Rejected admission/allocation/submission preserves C; quiescent execution or
   callback failure may leave C partial or invalid. Scratch-release/provider
   contract violations retain the existing fail-stop ownership policy. */
zynum_native_status zynum_native_gemm_c128_row_major_cn_v1(void *storage,
 uint64_t m, uint64_t n, uint64_t k, double alpha_re, double alpha_im,
 const double *a_ri, uint64_t a_capacity_complex, uint64_t lda_complex,
 const double *b_ri, uint64_t b_capacity_complex, uint64_t ldb_complex,
 double beta_re, double beta_im, double *c_ri, uint64_t c_capacity_complex,
 uint64_t ldc_complex, zynum_native_report_v1 *report);
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
/* Single-grid negative periodic Laplacian solve, axis 2 fastest. All axes
   are positive; singleton axes and repeated neighbors retain stencil semantics.
   Query writes an output-only v1 requirements struct on success. The input
   problem requires the exact v1 struct_size and zero reserved fields. An
   unsupported struct_size leaves query output and solve report untouched.
   Workspace is 12 F64 per point;
   each stencil application also borrows transient Task and Invocation scratch
   from the admitted provider, outside these caller workspace requirements. */
enum {
 ZYNUM_NATIVE_PERIODIC_NEG_LAPLACIAN_PHASE_PREFLIGHT=0,
 ZYNUM_NATIVE_PERIODIC_NEG_LAPLACIAN_PHASE_ADMISSION=1,
 ZYNUM_NATIVE_PERIODIC_NEG_LAPLACIAN_PHASE_RHS_PROJECTION=2,
 ZYNUM_NATIVE_PERIODIC_NEG_LAPLACIAN_PHASE_ITERATION=3,
 ZYNUM_NATIVE_PERIODIC_NEG_LAPLACIAN_PHASE_FINAL_VERIFICATION=4,
 ZYNUM_NATIVE_PERIODIC_NEG_LAPLACIAN_PHASE_PUBLISHED=5
};
enum {
 ZYNUM_NATIVE_PERIODIC_NEG_LAPLACIAN_STOP_NONE=0,
 ZYNUM_NATIVE_PERIODIC_NEG_LAPLACIAN_STOP_CONVERGED=1,
 ZYNUM_NATIVE_PERIODIC_NEG_LAPLACIAN_STOP_ITERATION_LIMIT=2,
 ZYNUM_NATIVE_PERIODIC_NEG_LAPLACIAN_STOP_RHO_BREAKDOWN=3,
 ZYNUM_NATIVE_PERIODIC_NEG_LAPLACIAN_STOP_ALPHA_BREAKDOWN=4,
 ZYNUM_NATIVE_PERIODIC_NEG_LAPLACIAN_STOP_OMEGA_BREAKDOWN=5,
 ZYNUM_NATIVE_PERIODIC_NEG_LAPLACIAN_STOP_FINAL_RESIDUAL_REJECTED=6,
 ZYNUM_NATIVE_PERIODIC_NEG_LAPLACIAN_STOP_NUMERICAL_FAILURE=7
};
enum {
 ZYNUM_NATIVE_PERIODIC_NEG_LAPLACIAN_RHS_VALID=1,
 ZYNUM_NATIVE_PERIODIC_NEG_LAPLACIAN_SOLVER_VALID=2,
 ZYNUM_NATIVE_PERIODIC_NEG_LAPLACIAN_FINAL_VALID=4
};
typedef struct zynum_native_periodic_neg_laplacian_problem_v1 {
 uint32_t abi_version, struct_size;
 uint64_t shape[3];
 double inv_h2[3];
 uint32_t max_iterations, reserved0;
 double absolute_tolerance_l2, relative_tolerance;
 uint64_t reserved[2];
} zynum_native_periodic_neg_laplacian_problem_v1;
typedef struct zynum_native_periodic_neg_laplacian_requirements_v1 {
 uint32_t abi_version, struct_size;
 uint64_t points, workspace_elements, workspace_bytes, workspace_alignment;
 uint64_t reserved[2];
} zynum_native_periodic_neg_laplacian_requirements_v1;
typedef struct zynum_native_periodic_neg_laplacian_report_v1 {
 uint32_t abi_version, struct_size, phase, stop_reason;
 uint32_t valid_flags, reserved0;
 uint64_t iterations, solver_actions, residual_restarts;
 double rhs_mean, solution_mean, rhs_l2, projected_rhs_l2, residual_l2;
 zynum_native_report_v1 native;
 uint64_t reserved[2];
} zynum_native_periodic_neg_laplacian_report_v1;
zynum_native_status zynum_native_periodic_neg_laplacian_query_f64_v1(
 const zynum_native_periodic_neg_laplacian_problem_v1 *problem,
 zynum_native_periodic_neg_laplacian_requirements_v1 *out);
/* From zero, solve PAP*x=P*b by BiCGStab for A=-Delta_h and P the mean
   projection. Reapply A to the projected F64 candidate and compare the original
   ||b-A*x||_2 with max(absolute_tolerance_l2,relative_tolerance*||b||_2).
   solution is written only on OK + CONVERGED + native.output_state COMPLETE.
   All returning failures preserve solution and its tail. RHS stays read-only.
   The three control ranges and full declared backing ranges must be pairwise
   disjoint. Unprovable control/backing ranges and aliases leave report untouched.
   The report is output-only. Its native diagnostics describe the last provider
   call, with output_state describing caller solution. solver_actions excludes
   the final verification application. Numerical stops return OK, while
   admission/provider failures return their existing native status. */
zynum_native_status zynum_native_periodic_neg_laplacian_solve_f64_v1(
 void *context_storage,
 const zynum_native_periodic_neg_laplacian_problem_v1 *problem,
 const double *rhs, uint64_t rhs_backing_elements,
 double *solution, uint64_t solution_backing_elements,
 double *workspace, uint64_t workspace_backing_elements,
 zynum_native_periodic_neg_laplacian_report_v1 *report);
/* Exact, context-free summary of count finite F64 values, with count > 0.
   scale must be positive finite and limit nonnegative finite. Lower/upper
   fields enclose the exact quantities, and sum_sign is -1, 0 or 1. The
   directional half-width is toward zero relative to the exact sum.
   The complete declared backing and output report must be disjoint and
   remain valid through return. Any rejection leaves the report untouched.
   No allocation, provider, Context or runtime is used. */
enum {
 ZYNUM_NATIVE_F64_SUM_EXACT_ZERO=1,
 ZYNUM_NATIVE_F64_SUM_QUANTIZATION_ZERO_INTERIOR=2,
 ZYNUM_NATIVE_F64_SUM_SCALED_ABS_WITHIN_LIMIT=4
};
typedef struct zynum_native_f64_sum_summary_report_v1 {
 uint32_t abi_version, struct_size;
 uint64_t count;
 int32_t sum_sign;
 uint32_t flags;
 double mean_lower, mean_upper, scaled_sum_lower, scaled_sum_upper;
 double scaled_l1_lower, scaled_l1_upper;
 double scaled_direction_half_width_lower, scaled_direction_half_width_upper;
 double maximum_absolute;
 uint64_t reserved[2];
} zynum_native_f64_sum_summary_report_v1;
zynum_native_status zynum_native_f64_sum_summary_v1(
 const double *values, uint64_t backing_elements, uint64_t count,
 double scale, double limit, zynum_native_f64_sum_summary_report_v1 *out);
#ifdef ZYNUM_NATIVE_IDENTITY_TESTING
const void *zynum_native_core_identity_v1(void);
const void *zynum_native_execution_identity_v1(void);
#endif
#ifdef __cplusplus
}
#endif
#include "zynum_native_spectral.h"
#endif
