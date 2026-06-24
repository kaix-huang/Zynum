// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later

//! AArch64 single-vector specialized kernels.

const features = @import("features.zig");

const enable_sme_dscal = true;
const enable_sve_dscal = true;
const enable_asimd_dscal = false;
const enable_asimd_dasum = false;
const enable_sme_dasum = true;
const asimd_level1_min_len = 256 * 1024;

extern fn zynum_blas_asimd_dscal_f64(n: usize, alpha: f64, x: [*]f64) callconv(.c) void;
extern fn zynum_blas_sve_dscal_f64(n: usize, alpha: f64, x: [*]f64) callconv(.c) void;
extern fn zynum_blas_asimd_dasum_f64(n: usize, x: [*]const f64) callconv(.c) f64;

pub fn scalUnitReal(comptime T: type, n: usize, alpha: T, x: [*]T) bool {
    if (comptime enable_sme_dscal and features.has_sme) {
        if (T == f64 and n >= 256 * 1024 and features.streamingVectorBytes() == 64) {
            var sm_state: features.StreamingModeState = undefined;
            sm_state.startSm();
            defer sm_state.stopSm();

            smeDscalF64Streaming(n, alpha, x);
            return true;
        }
    }
    if (comptime enable_sve_dscal and features.has_sve) {
        if (T == f64 and n >= 16) {
            zynum_blas_sve_dscal_f64(n, alpha, x);
            return true;
        }
    }
    if (comptime !enable_asimd_dscal) return false;
    if (T != f64) return false;
    if (comptime !features.has_asimd) return false;
    if (n < asimd_level1_min_len) return false;
    zynum_blas_asimd_dscal_f64(n, alpha, x);
    return true;
}

pub fn asumUnitReal(comptime T: type, n: usize, x: [*]const T) ?T {
    if (comptime enable_sme_dasum and features.has_sme) {
        if (T == f64 and n >= 64 * 1024 and features.streamingVectorBytes() == 64) {
            var sm_state: features.StreamingModeState = undefined;
            sm_state.startSm();
            const result_bits = smeDasumF64StreamingBits(n, x);
            const stopped_result_bits = sm_state.stopSmRetU64(result_bits);
            return @bitCast(stopped_result_bits);
        }
    }
    if (comptime !enable_asimd_dasum) return null;
    if (T != f64) return null;
    if (comptime !features.has_asimd) return null;
    if (n < asimd_level1_min_len) return null;
    return zynum_blas_asimd_dasum_f64(n, x);
}

noinline fn smeDscalF64Streaming(n: usize, alpha: f64, x: [*]f64) callconv(.c) void {
    asm volatile (
        \\cbz x0, 3f
        \\fmov x9, d0
        \\fmov d4, x9
        \\mov z4.d, d4
        \\ptrue p0.d
        \\cntd x6
        \\lsl x7, x6, #2
        \\
        \\0:
        \\cmp x0, x7
        \\b.lo 1f
        \\ld1d { z0.d }, p0/z, [x1]
        \\ld1d { z1.d }, p0/z, [x1, #1, MUL VL]
        \\ld1d { z2.d }, p0/z, [x1, #2, MUL VL]
        \\ld1d { z3.d }, p0/z, [x1, #3, MUL VL]
        \\fmul z0.d, p0/m, z0.d, z4.d
        \\fmul z1.d, p0/m, z1.d, z4.d
        \\fmul z2.d, p0/m, z2.d, z4.d
        \\fmul z3.d, p0/m, z3.d, z4.d
        \\st1d { z0.d }, p0, [x1]
        \\st1d { z1.d }, p0, [x1, #1, MUL VL]
        \\st1d { z2.d }, p0, [x1, #2, MUL VL]
        \\st1d { z3.d }, p0, [x1, #3, MUL VL]
        \\addvl x1, x1, #4
        \\sub x0, x0, x7
        \\b 0b
        \\
        \\1:
        \\cbz x0, 2f
        \\mov x8, #0
        \\whilelo p1.d, x8, x0
        \\ld1d { z0.d }, p1/z, [x1]
        \\fmul z0.d, p1/m, z0.d, z4.d
        \\st1d { z0.d }, p1, [x1]
        \\
        \\2:
        \\3:
        :
        : [n] "{x0}" (n),
          [x] "{x1}" (x),
          [alpha] "{d0}" (alpha),
        : .{ .memory = true });
}

noinline fn smeDasumF64StreamingBits(n: usize, x: [*]const f64) callconv(.c) u64 {
    return asm volatile (
        \\cbz x0, 3f
        \\ptrue p0.d
        \\cntd x6
        \\lsl x7, x6, #2
        \\dup z0.d, #0
        \\dup z1.d, #0
        \\dup z2.d, #0
        \\dup z3.d, #0
        \\
        \\0:
        \\cmp x0, x7
        \\b.lo 1f
        \\ld1d { z4.d }, p0/z, [x1]
        \\ld1d { z5.d }, p0/z, [x1, #1, MUL VL]
        \\ld1d { z6.d }, p0/z, [x1, #2, MUL VL]
        \\ld1d { z7.d }, p0/z, [x1, #3, MUL VL]
        \\fabs z4.d, p0/m, z4.d
        \\fabs z5.d, p0/m, z5.d
        \\fabs z6.d, p0/m, z6.d
        \\fabs z7.d, p0/m, z7.d
        \\fadd z0.d, z0.d, z4.d
        \\fadd z1.d, z1.d, z5.d
        \\fadd z2.d, z2.d, z6.d
        \\fadd z3.d, z3.d, z7.d
        \\addvl x1, x1, #4
        \\sub x0, x0, x7
        \\b 0b
        \\
        \\1:
        \\cbz x0, 2f
        \\mov x8, #0
        \\whilelo p1.d, x8, x0
        \\ld1d { z4.d }, p1/z, [x1]
        \\fabs z4.d, p1/m, z4.d
        \\fadd z0.d, z0.d, z4.d
        \\cmp x0, x6
        \\b.ls 2f
        \\addvl x1, x1, #1
        \\sub x0, x0, x6
        \\b 1b
        \\
        \\2:
        \\fadd z0.d, z0.d, z1.d
        \\fadd z2.d, z2.d, z3.d
        \\fadd z0.d, z0.d, z2.d
        \\faddv d16, p0, z0.d
        \\fmov x9, d16
        \\mov x0, x9
        \\b 4f
        \\
        \\3:
        \\mov x0, xzr
        \\
        \\4:
        : [result] "={x0}" (-> u64),
        : [n] "{x0}" (n),
          [x] "{x1}" (x),
        : .{ .memory = true });
}
