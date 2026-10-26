/* SPDX-License-Identifier: LGPL-3.0-or-later */
#include "zynum.h"
heteris_status zynum_validate_view(const heteris_view *v, uint64_t *begin, uint64_t *end) {
    return heteris_view_validate(v,begin,end);
}

/* Preserve the public by-value C ABI despite Zig 0.17's SysV aggregate argument
 * lowering issue after six integer-register arguments. The numerical kernel
 * remains in Zig; only four scalar components cross this internal boundary. */
extern heteris_status zynum_internal_gemm_c128_components(const zynum_resources *,
    uint32_t, uint32_t, const heteris_view *, const heteris_view *, const heteris_view *,
    double, double, double, double);
heteris_status zynum_gemm_c128(const zynum_resources *resources, uint32_t op_a, uint32_t op_b,
    const heteris_view *a, const heteris_view *b, const heteris_view *c,
    zynum_complex128 alpha, zynum_complex128 beta) {
    return zynum_internal_gemm_c128_components(resources, op_a, op_b, a, b, c,
        alpha.real, alpha.imag, beta.real, beta.imag);
}
