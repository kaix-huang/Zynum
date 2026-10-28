/* Copyright (C) 2026 Zynum contributors
 * SPDX-License-Identifier: LGPL-3.0-or-later */
#ifndef ZYNUM_NATIVE_SPECTRAL_H
#define ZYNUM_NATIVE_SPECTRAL_H
#include "zynum_native.h"
#ifdef __cplusplus
extern "C" {
#endif
/* Borrowed FP64 panels, explicit Context and caller-owned workspace/guard.
 * Query and initialize the opaque guard exactly once before sharing it. Keep
 * its storage stable and live, with no copy/move/reinitialization, through all
 * returning calls. Only Zynum accesses its internal atomic state. Excess guard
 * and workspace bytes are untouched. No provider or allocation is owned here.
 *
 * Callbacks run synchronously on the API caller and apply the ORIGINAL H, not
 * its interval-scaled action. Nonzero int32 returns are opaque, even enum
 * collisions. Finish all accesses before return; retain no borrowed pointers
 * and do not unwind across C. Compact callback panels have ld=rows and backing
 * rows*columns; empty panels have NULL pointers/backing0, retaining rows/columns
 * and ld=rows. Series W0 still calls Q-1 times; moments W0 never calls.
 *
 * Full declared backing/control ranges are checked before report/guard writes;
 * INVALID_CONTROL leaves them untouched. Other failures initialize a validation
 * report before acquiring the guard. BUSY never releases another caller's guard.
 * Logical output is published only after every order succeeds. Recoverable
 * failure preserves every public output byte, including padding/tails.
 *
 * All payload objects/access rights, Context/provider/image, callback and model
 * remain live and fixed through return. X/output may overlap. Work is separate
 * from all payloads; series coefficients are separate from output. Controls and
 * userdata are mutually separate and separate from ALL declared payload ranges.
 * Userdata is NULL/0 or nonnull/positive, and only its declared object is checked:
 * the caller must separately exclude its reachable model/workspace alias graph.
 * Numeric range checks do not establish provenance, accessibility or lifetime.
 *
 * Consume report fields only after the function returns; the report is not a
 * live callback progress channel.
 * Reports distinguish last returned callback status from local arithmetic and
 * admission failures. Placeholder fields are valid only under their flags.
 * Output COMPLETE requires a nonempty logical output; otherwise UNCHANGED.
 * Bounds are the caller's enclosure promise; probes/H/scaling stay fixed.
 * Serial recurrence/commit runs on the caller; admission alone is no provider
 * bulk charge or claim of full resource/scientific qualification. See the SDK
 * documentation for exact validation order, sizes, phases and close semantics.
 */
typedef int32_t zynum_native_spectral_status;
enum {
 ZYNUM_NATIVE_SPECTRAL_OK=0,
 ZYNUM_NATIVE_SPECTRAL_INVALID_CONTROL=1,
 ZYNUM_NATIVE_SPECTRAL_INVALID_ARGUMENT=2,
 ZYNUM_NATIVE_SPECTRAL_VIEW_INVALID=3,
 ZYNUM_NATIVE_SPECTRAL_OVERFLOW=4,
 ZYNUM_NATIVE_SPECTRAL_ALIASING=5,
 ZYNUM_NATIVE_SPECTRAL_BUSY=6,
 ZYNUM_NATIVE_SPECTRAL_CONTEXT_FAILURE=7,
 ZYNUM_NATIVE_SPECTRAL_CALLBACK_FAILURE=8,
 ZYNUM_NATIVE_SPECTRAL_NONFINITE_INPUT=9,
 ZYNUM_NATIVE_SPECTRAL_NUMERICAL_FAILURE=10
};
enum { ZYNUM_NATIVE_SPECTRAL_SERIES=1, ZYNUM_NATIVE_SPECTRAL_MOMENTS=2 };
enum {
 ZYNUM_NATIVE_SPECTRAL_VALIDATION=0, ZYNUM_NATIVE_SPECTRAL_ADMISSION=1,
 ZYNUM_NATIVE_SPECTRAL_INITIALIZE=2, ZYNUM_NATIVE_SPECTRAL_APPLY=3,
 ZYNUM_NATIVE_SPECTRAL_RECURRENCE=4, ZYNUM_NATIVE_SPECTRAL_MOMENT=5,
 ZYNUM_NATIVE_SPECTRAL_COMMIT=6, ZYNUM_NATIVE_SPECTRAL_COMPLETE=7
};
/* output_state uses existing ZYNUM_NATIVE_OUTPUT_UNCHANGED=0 or COMPLETE=1;
   no public-output PARTIAL_OR_INVALID state is produced by this interface. */
typedef int32_t (*zynum_native_spectral_apply_f64_v1)(
 void *user_data, void *native_context,
 uint64_t rows, uint64_t columns,
 const double *x, uint64_t x_backing_elements, uint64_t ldx,
 double *y, uint64_t y_backing_elements, uint64_t ldy);
typedef struct zynum_native_spectral_report_v1 {
 uint32_t abi_version, struct_size, phase, output_state;
 uint32_t admission_attempted, admission_succeeded, order_valid, callback_returned;
 zynum_native_status admission_status;
 int32_t callback_status;
 uint64_t order, callback_order, callbacks_returned, reserved[2];
} zynum_native_spectral_report_v1;
zynum_native_spectral_status zynum_native_spectral_guard_query_v1(
 uint64_t *out_bytes, uint64_t *out_alignment);
zynum_native_spectral_status zynum_native_spectral_guard_init_v1(
 void *storage, uint64_t bytes);
zynum_native_spectral_status zynum_native_spectral_workspace_elements_v1(
 uint32_t kind, uint64_t rows, uint64_t columns, uint64_t orders,
 uint64_t *out_elements);
zynum_native_spectral_status zynum_native_chebyshev_series_f64_v1(
 void *native_context, zynum_native_spectral_apply_f64_v1 apply,
 void *user_data, uint64_t user_data_bytes,
 uint64_t rows, uint64_t columns,
 const double *x, uint64_t x_backing_elements, uint64_t ldx,
 const double *coefficients, uint64_t coefficients_backing_elements,
 uint64_t coefficient_count, double lower, double upper,
 double *y, uint64_t y_backing_elements, uint64_t ldy,
 double *work, uint64_t work_backing_elements,
 void *guard, uint64_t guard_bytes,
 zynum_native_spectral_report_v1 *report);
zynum_native_spectral_status zynum_native_chebyshev_moments_f64_v1(
 void *native_context, zynum_native_spectral_apply_f64_v1 apply,
 void *user_data, uint64_t user_data_bytes,
 uint64_t rows, uint64_t columns,
 const double *x, uint64_t x_backing_elements, uint64_t ldx,
 uint64_t orders, double lower, double upper,
 double *moments, uint64_t moments_backing_elements, uint64_t ldm,
 double *work, uint64_t work_backing_elements,
 void *guard, uint64_t guard_bytes,
 zynum_native_spectral_report_v1 *report);
#ifdef __cplusplus
}
#endif
#endif
