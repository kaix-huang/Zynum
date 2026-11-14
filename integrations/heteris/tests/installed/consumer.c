/* SPDX-License-Identifier: LGPL-3.0-or-later */
#include <zynum.h>
#include <zynum_spectral.h>
#include <zynum_native.h>
#include <zynum_native_spectral.h>
#include <assert.h>
#include <stddef.h>
#ifdef __cplusplus
#define ABI_ASSERT static_assert
#define ZERO_INIT {}
#else
#define ABI_ASSERT _Static_assert
#define ZERO_INIT {0}
#endif
ABI_ASSERT(sizeof(double) == 8, "FP64 width");
ABI_ASSERT(sizeof(zynum_complex128) == 16, "complex128 width");
ABI_ASSERT(offsetof(zynum_complex128, imag) == 8, "complex128 layout");
ABI_ASSERT(sizeof(zynum_native_context_requirements_v1) == 40, "requirements layout");
ABI_ASSERT(sizeof(zynum_native_report_v1) == 72, "report layout");
int main(void) {
    zynum_native_context_requirements_v1 requirements = ZERO_INIT;
    assert(zynum_native_context_query_v1(&requirements) == ZYNUM_NATIVE_OK);
    assert(requirements.abi_version == 1 && requirements.bytes > 0 && requirements.alignment > 0);
    const double values[] = {0x1p100, 1.0, -0x1p100};
    zynum_native_f64_sum_summary_report_v1 report = ZERO_INIT;
    assert(zynum_native_f64_sum_summary_v1(values, 3, 3, 0.5, 0.5, &report) == ZYNUM_NATIVE_OK);
    assert(report.scaled_sum_lower == 0.5 && report.scaled_sum_upper == 0.5);
    /* Missing explicit context/provider is rejected, never replaced implicitly. */
    assert(zynum_native_gemm_f64_nn_v1(NULL, 0, 0, 0, 1, NULL, 0, 0, NULL, 0, 0, 0,
                                      NULL, 0, 0, NULL) == ZYNUM_NATIVE_INVALID_ARGUMENT);
    assert(zynum_native_gemm_c128_row_major_cn_v1(NULL, 0, 0, 0, 1, 0, NULL, 0, 0,
             NULL, 0, 0, 0, 0, NULL, 0, 0, NULL) == ZYNUM_NATIVE_INVALID_ARGUMENT);
    return 0;
}
