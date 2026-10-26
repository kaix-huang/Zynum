/* Copyright (C) 2026 Scientific Platform contributors
 * SPDX-License-Identifier: LGPL-3.0-or-later */
#ifndef ZYNUM_SPECTRAL_H
#define ZYNUM_SPECTRAL_H
#include "zynum.h"
#ifdef __cplusplus
extern "C" {
#endif
typedef heteris_status (*zynum_block_apply_f64)(void *user_data,
 const zynum_resources *, const heteris_view *x, const heteris_view *y);
/* Synchronous sum_k c[k] T_k((H-center I)/half_width) X; no endpoint halving.
 * lower<upper must be finite and enclose H's spectrum; this call does not prove
 * that enclosure. H and user_data remain fixed throughout all callbacks.
 * X/Y are host F64 rank2 [rows,block], arbitrary admitted strides. Coefficients
 * are a nonempty rank1 readable F64 view. Workspace is readable/writable F64
 * rank2 [rows,4*block], packed column-major (strides 8,8*rows), with checked
 * sizes. No allocation or default provider is used.
 * X/Y may overlap. Workspace must not overlap X/Y/coefficients; coefficients
 * must not overlap Y. Bounding byte intervals are conservatively checked.
 * Metadata and resources are copied before callbacks; borrowed storage and
 * callback code remain alive through return. Callbacks may alter only their
 * provided output logical elements and private state, complete synchronously,
 * and must not retain borrowed views. Callback output must be finite.
 * Output logical elements commit only on success; padding is preserved.
 * Workspace may change on any failure. Nonfinite inputs return INVALID_ARGUMENT;
 * nonfinite callback/recurrence results return NUMERICAL_FAILURE;
 * callback errors propagate unchanged. Empty X still validates all arguments
 * and invokes the callback once per requested positive polynomial order.
 * Results/moments from a changed H or a different interval cannot be combined. */
heteris_status zynum_chebyshev_series_f64(const zynum_resources *,
 zynum_block_apply_f64, void *user_data, const heteris_view *x,
 const heteris_view *coefficients, double lower, double upper,
 const heteris_view *y, const heteris_view *workspace);
/* Raw per-vector moments mu[k,j] = X_j^T T_k((H-center I)/half_width) X_j.
 * count>=1; no sample averaging, spin factor, normalization or endpoint halving.
 * X is readable host F64 rank2 [rows,block]. Moments is writable host F64
 * rank2 [count,block] with arbitrary admitted strides and may overlap X.
 * Recurrence workspace is readable/writable packed column-major F64
 * [rows,3*block]; staged workspace is the same layout [count,block].
 * Both workspaces must be disjoint from X, moments and each other, using
 * conservative mapped byte intervals. Sizes/strides are checked before use.
 * The interval, callback lifetime, metadata snapshots, finite arithmetic and
 * failure/commit rules above apply. Moments commits only after all orders.
 * Zero block validates everything and returns without callbacks. Zero rows
 * with a nonzero block produces zero moments and invokes count-1 callbacks.
 * Workspace may change on failure; no allocation or hidden context is used. */
heteris_status zynum_chebyshev_moments_f64(const zynum_resources *,
 zynum_block_apply_f64, void *user_data, const heteris_view *x, uint64_t count,
 double lower, double upper, const heteris_view *moments,
 const heteris_view *recurrence_workspace, const heteris_view *staged_workspace);
#ifdef __cplusplus
}
#endif
#endif
