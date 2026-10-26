/* SPDX-License-Identifier: LGPL-3.0-or-later */
#ifndef ZYNUM_H
#define ZYNUM_H
#include "heteris.h"

#ifdef __cplusplus
extern "C" {
#endif

#define ZYNUM_ABI_VERSION 1u
enum { ZYNUM_MODE_NAIVE=HETERIS_RESOURCE_MODE_NAIVE,
       ZYNUM_MODE_INTEGRATED=HETERIS_RESOURCE_MODE_INTEGRATED };
typedef heteris_allocate_fn zynum_allocate_fn;
typedef heteris_release_fn zynum_release_fn;
typedef heteris_map_fn zynum_map_fn;
/* Explicit synchronous CPU provider. Integrated mode calls only this table;
 * errors propagate without fallback. No kernels allocate or create threads.
 * The caller keeps table, user_data, and storage alive for each whole call.
 * Providers must map each allocation to a stable, unique host byte range.
 * Callers serialize provider allocation/release and prevent concurrent writes. */
typedef heteris_resources zynum_resources;

/* Explicit lightweight malloc/free provider, no pools. A destroy with live
 * allocations returns BUSY and leaves the resource table valid. */
heteris_status zynum_resources_create_naive(zynum_resources *out);
heteris_status zynum_resources_destroy_naive(zynum_resources *resources);

/* Views must be host F64, rank 1 for axpy/dot or rank 3 for Laplacian.
 * Signed byte strides, read broadcasts and empty views follow Heteris rules.
 * Metadata, resources, shapes, map results and cross-view aliasing are checked
 * before writing. Output is unchanged on any returned error.
 * Axpy supports exact x==y layout; any other intersecting byte ranges fail.
 * Laplacian requires disjoint input/output ranges. Dot reads both inputs and
 * writes its scalar result only after success. Its output must not alias inputs.
 * alpha and inv_h2 components must be finite; inv_h2 must also be positive.
 * inv_h2[d] is 1/(spacing[d]^2). Laplacian computes sum_d inv_h2[d] *
 * (x[i-e_d] - 2*x[i] + x[i+e_d]) with periodic boundaries, without a matrix.
 * NaNs/Infs in array values follow IEEE arithmetic. No compensated reduction. */
heteris_status zynum_axpy_f64(const zynum_resources *, const heteris_view *x, const heteris_view *y, double alpha);
heteris_status zynum_dot_f64(const zynum_resources *, const heteris_view *x, const heteris_view *y, double *result);
heteris_status zynum_laplacian_periodic_3d_f64(const zynum_resources *, const heteris_view *x, const heteris_view *y, const double inv_h2[3]);

enum { ZYNUM_OP_NONE=0, ZYNUM_OP_TRANSPOSE=1, ZYNUM_OP_CONJUGATE_TRANSPOSE=2 };
typedef struct zynum_complex128 { double real, imag; } zynum_complex128;
/* C = alpha*op(A)*op(B) + beta*C. Rank-2 host F64/COMPLEX128 views.
 * op(A) is m*k, op(B) k*n, C m*n. Signed byte strides and read
 * broadcasts follow Heteris. For real values conjugate transpose is transpose.
 * All scalars must be finite. All metadata, resources, dimensions and mappings
 * are validated even for empty output, alpha=0 or k=0. C's conservative mapped
 * byte interval must be disjoint from A and B, including inactive operands.
 * A/B always require READ. C requires WRITE and requires READ only if beta!=0.
 * beta=0 never reads C; alpha=0 or k=0 never reads A/B elements and scales C.
 * Descriptors/resources/scalars are snapshotted before provider callbacks.
 * No allocation, packing, implicit executor or optimized dispatch is used.
 * Returned errors precede all writes. Array NaNs/Infs use IEEE arithmetic. */
heteris_status zynum_gemm_f64(const zynum_resources *, uint32_t op_a, uint32_t op_b,
 const heteris_view *a, const heteris_view *b, const heteris_view *c, double alpha, double beta);
heteris_status zynum_gemm_c128(const zynum_resources *, uint32_t op_a, uint32_t op_b,
 const heteris_view *a, const heteris_view *b, const heteris_view *c, zynum_complex128 alpha, zynum_complex128 beta);

#ifdef __cplusplus
}
#endif
#endif
