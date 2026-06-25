// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later

//! AArch64 single-vector specialized kernels.

const asm_fragments = @import("asm_fragments.zig");
const features = @import("features.zig");
const vector_matrix_asm = @import("vector_matrix_asm.zig");

const enable_asimd_dscal = false;
const enable_sme_dscal = true;
const enable_sve_dscal = true;
const enable_sme_dasum = true;
const enable_sve_dasum = true;

fn asimdDscalF64(n: usize, alpha: f64, x: [*]f64) void {
    const V = @Vector(32, f64);
    const alpha_v: V = @splat(alpha);
    var i: usize = 0;
    while (i + 64 <= n) : (i += 64) {
        inline for (0..2) |k| {
            const offset = i + 32 * k;
            const xv: V = @as(*align(1) const V, @ptrCast(x + offset)).*;
            @as(*align(1) V, @ptrCast(x + offset)).* = xv * alpha_v;
        }
    }
    while (i + 32 <= n) : (i += 32) {
        const xv: V = @as(*align(1) const V, @ptrCast(x + i)).*;
        @as(*align(1) V, @ptrCast(x + i)).* = xv * alpha_v;
    }
    while (i < n) : (i += 1) x[i] *= alpha;
}

inline fn callScalF64Kernel(comptime kernel: anytype, n: usize, alpha: f64, x: [*]f64) void {
    const Kernel = *const fn (usize, f64, [*]f64) callconv(.c) void;
    @as(Kernel, @ptrCast(&kernel))(n, alpha, x);
}

inline fn callScalF32Kernel(comptime kernel: anytype, n: usize, alpha: f32, x: [*]f32) void {
    const Kernel = *const fn (usize, f32, [*]f32) callconv(.c) void;
    @as(Kernel, @ptrCast(&kernel))(n, alpha, x);
}

inline fn callAsumF64Kernel(comptime kernel: anytype, n: usize, x: [*]const f64) u64 {
    const Kernel = *const fn (usize, [*]const f64) callconv(.c) u64;
    return @as(Kernel, @ptrCast(&kernel))(n, x);
}

inline fn callAsumF32Kernel(comptime kernel: anytype, n: usize, x: [*]const f32) u32 {
    const Kernel = *const fn (usize, [*]const f32) callconv(.c) u32;
    return @as(Kernel, @ptrCast(&kernel))(n, x);
}

inline fn scalUnitRealDisabled(comptime T: type, n: usize, alpha: T, x: [*]T) bool {
    _ = n;
    _ = alpha;
    _ = x;
    return false;
}

inline fn asumUnitRealDisabled(comptime T: type, n: usize, x: [*]const T) ?T {
    _ = n;
    _ = x;
    return null;
}

pub fn scalUnitReal(comptime T: type, n: usize, alpha: T, x: [*]T) bool {
    if (comptime !(enable_asimd_dscal and features.has_asimd) and !(enable_sme_dscal and features.has_sme) and !(enable_sve_dscal and features.has_sve)) {
        return scalUnitRealDisabled(T, n, alpha, x);
    }
    if (comptime enable_asimd_dscal and features.has_asimd) {
        if (T == f64 and n >= 16) {
            asimdDscalF64(n, alpha, x);
            return true;
        }
    }
    if (comptime enable_sme_dscal and features.has_sme) {
        if (T == f32 and n >= 64 * 1024 and features.streamingVectorBytes() == 64) {
            var sm_state: features.StreamingModeState = undefined;
            sm_state.startSmZa();
            defer sm_state.stopSmZa();

            callScalF32Kernel(smeSscalF32Streaming, n, alpha, x);
            return true;
        }
        if (T == f64 and n >= 64 * 1024 and features.streamingVectorBytes() == 64) {
            var sm_state: features.StreamingModeState = undefined;
            sm_state.startSmZa();
            defer sm_state.stopSmZa();

            callScalF64Kernel(smeDscalF64Streaming, n, alpha, x);
            return true;
        }
    }
    if (comptime enable_sve_dscal and features.has_sve) {
        if (T == f64 and n >= 16) {
            callScalF64Kernel(vector_matrix_asm.sveDscalF64, n, alpha, x);
            return true;
        }
    }
    return false;
}

pub fn asumUnitReal(comptime T: type, n: usize, x: [*]const T) ?T {
    if (comptime !(enable_sme_dasum and features.has_sme) and !(enable_sve_dasum and features.has_sve)) {
        return asumUnitRealDisabled(T, n, x);
    }
    if (comptime enable_sve_dasum and features.has_sve) {
        if (T == f32 and n >= 64 * 1024 and features.sveVectorBytes() == 64) {
            const result_bits = callAsumF32Kernel(sveSasumF32Bits, n, x);
            return @bitCast(result_bits);
        }
    }
    if (comptime enable_sme_dasum and features.has_sme2) {
        if (T == f32 and n >= 64 * 1024 and features.streamingVectorBytes() == 64) {
            var sm_state: features.StreamingModeState = undefined;
            sm_state.startSmZa();
            const result_bits = callAsumF32Kernel(smeSasumF32StreamingBits, n, x);
            const stopped_result_bits = sm_state.stopSmZaRetU32(result_bits);
            return @bitCast(stopped_result_bits);
        }
        if (T == f64 and n >= 64 * 1024 and features.streamingVectorBytes() == 64) {
            var sm_state: features.StreamingModeState = undefined;
            sm_state.startSmZa();
            const result_bits = callAsumF64Kernel(smeDasumF64StreamingBits, n, x);
            const stopped_result_bits = sm_state.stopSmZaRetU64(result_bits);
            return @bitCast(stopped_result_bits);
        }
    }
    if (comptime enable_sve_dasum and features.has_sve) {
        if (T == f64 and n >= 16) {
            const result_bits = callAsumF64Kernel(vector_matrix_asm.sveDasumF64Bits, n, x);
            return @bitCast(result_bits);
        }
    }
    return null;
}

noinline fn sveSasumF32Bits(n: usize, x: [*]const f32) callconv(.naked) u32 {
    _ = n;
    _ = x;
    asm volatile (
        \\cbz x0, 4f
    ++ asm_fragments.ptrue_p0_s ++
        \\cntw x6
        \\lsl x7, x6, #4
        \\dup z0.s, #0
        \\dup z1.s, #0
        \\dup z2.s, #0
        \\dup z3.s, #0
        \\dup z4.s, #0
        \\dup z5.s, #0
        \\dup z6.s, #0
        \\dup z7.s, #0
        \\
        \\0:
        \\cmp x0, x7
        \\b.lo 1f
        \\ld1w { z16.s }, p0/z, [x1]
        \\ld1w { z17.s }, p0/z, [x1, #1, MUL VL]
        \\ld1w { z18.s }, p0/z, [x1, #2, MUL VL]
        \\ld1w { z19.s }, p0/z, [x1, #3, MUL VL]
        \\ld1w { z20.s }, p0/z, [x1, #4, MUL VL]
        \\ld1w { z21.s }, p0/z, [x1, #5, MUL VL]
        \\ld1w { z22.s }, p0/z, [x1, #6, MUL VL]
        \\ld1w { z23.s }, p0/z, [x1, #7, MUL VL]
        \\fabs z16.s, p0/m, z16.s
        \\fabs z17.s, p0/m, z17.s
        \\fabs z18.s, p0/m, z18.s
        \\fabs z19.s, p0/m, z19.s
        \\fabs z20.s, p0/m, z20.s
        \\fabs z21.s, p0/m, z21.s
        \\fabs z22.s, p0/m, z22.s
        \\fabs z23.s, p0/m, z23.s
        \\fadd z0.s, z0.s, z16.s
        \\fadd z1.s, z1.s, z17.s
        \\fadd z2.s, z2.s, z18.s
        \\fadd z3.s, z3.s, z19.s
        \\fadd z4.s, z4.s, z20.s
        \\fadd z5.s, z5.s, z21.s
        \\fadd z6.s, z6.s, z22.s
        \\fadd z7.s, z7.s, z23.s
        \\
        \\ld1w { z16.s }, p0/z, [x1, #8, MUL VL]
        \\ld1w { z17.s }, p0/z, [x1, #9, MUL VL]
        \\ld1w { z18.s }, p0/z, [x1, #10, MUL VL]
        \\ld1w { z19.s }, p0/z, [x1, #11, MUL VL]
        \\ld1w { z20.s }, p0/z, [x1, #12, MUL VL]
        \\ld1w { z21.s }, p0/z, [x1, #13, MUL VL]
        \\ld1w { z22.s }, p0/z, [x1, #14, MUL VL]
        \\ld1w { z23.s }, p0/z, [x1, #15, MUL VL]
        \\fabs z16.s, p0/m, z16.s
        \\fabs z17.s, p0/m, z17.s
        \\fabs z18.s, p0/m, z18.s
        \\fabs z19.s, p0/m, z19.s
        \\fabs z20.s, p0/m, z20.s
        \\fabs z21.s, p0/m, z21.s
        \\fabs z22.s, p0/m, z22.s
        \\fabs z23.s, p0/m, z23.s
        \\fadd z0.s, z0.s, z16.s
        \\fadd z1.s, z1.s, z17.s
        \\fadd z2.s, z2.s, z18.s
        \\fadd z3.s, z3.s, z19.s
        \\fadd z4.s, z4.s, z20.s
        \\fadd z5.s, z5.s, z21.s
        \\fadd z6.s, z6.s, z22.s
        \\fadd z7.s, z7.s, z23.s
        \\addvl x1, x1, #16
        \\sub x0, x0, x7
        \\b 0b
        \\
        \\1:
        \\cbz x0, 2f
        \\mov x8, #0
        \\
        \\5:
        \\whilelo p1.s, x8, x0
        \\b.none 2f
        \\ld1w { z16.s }, p1/z, [x1, x8, lsl #2]
        \\fabs z16.s, p1/m, z16.s
        \\fadd z0.s, z0.s, z16.s
        \\incw x8
        \\b 5b
        \\
        \\2:
        \\fadd z0.s, z0.s, z1.s
        \\fadd z2.s, z2.s, z3.s
        \\fadd z4.s, z4.s, z5.s
        \\fadd z6.s, z6.s, z7.s
        \\fadd z0.s, z0.s, z2.s
        \\fadd z4.s, z4.s, z6.s
        \\fadd z0.s, z0.s, z4.s
        \\faddv s16, p0, z0.s
        \\fmov w0, s16
        \\ret
        \\
        \\4:
        \\mov w0, wzr
        \\ret
    ::: .{ .memory = true });
}

noinline fn smeSscalF32Streaming(n: usize, alpha: f32, x: [*]f32) callconv(.naked) void {
    _ = n;
    _ = alpha;
    _ = x;
    asm volatile (
        \\cbz x0, 3f
        \\ptrue pn8.s
        \\mov z0.s, s0
        \\mov x7, #512
        \\
        \\0:
        \\cmp x0, x7
        \\b.lo 1f
        \\zero { za }
        \\mov w8, #0
        \\mov w11, #8
        \\ld1w { z4.s - z7.s }, pn8/z, [x1]
        \\ld1w { z16.s - z19.s }, pn8/z, [x1, #4, MUL VL]
        \\fmla za.s[w8, 0, vgx4], { z4.s - z7.s }, z0.s
        \\fmla za.s[w11, 0, vgx4], { z16.s - z19.s }, z0.s
        \\mov w8, #1
        \\mov w11, #9
        \\ld1w { z4.s - z7.s }, pn8/z, [x1, #8, MUL VL]
        \\ld1w { z16.s - z19.s }, pn8/z, [x1, #12, MUL VL]
        \\fmla za.s[w8, 0, vgx4], { z4.s - z7.s }, z0.s
        \\fmla za.s[w11, 0, vgx4], { z16.s - z19.s }, z0.s
        \\mov w8, #2
        \\mov w11, #10
        \\ld1w { z4.s - z7.s }, pn8/z, [x1, #16, MUL VL]
        \\ld1w { z16.s - z19.s }, pn8/z, [x1, #20, MUL VL]
        \\fmla za.s[w8, 0, vgx4], { z4.s - z7.s }, z0.s
        \\fmla za.s[w11, 0, vgx4], { z16.s - z19.s }, z0.s
        \\mov w8, #3
        \\mov w11, #11
        \\ld1w { z4.s - z7.s }, pn8/z, [x1, #24, MUL VL]
        \\ld1w { z16.s - z19.s }, pn8/z, [x1, #28, MUL VL]
        \\fmla za.s[w8, 0, vgx4], { z4.s - z7.s }, z0.s
        \\fmla za.s[w11, 0, vgx4], { z16.s - z19.s }, z0.s
        \\
        \\mov w8, #0
        \\mov w11, #8
        \\mov { z4.s - z7.s }, za.s[w8, 0, vgx4]
        \\mov { z16.s - z19.s }, za.s[w11, 0, vgx4]
        \\st1w { z4.s - z7.s }, pn8, [x1]
        \\st1w { z16.s - z19.s }, pn8, [x1, #4, MUL VL]
        \\mov w8, #1
        \\mov w11, #9
        \\mov { z4.s - z7.s }, za.s[w8, 0, vgx4]
        \\mov { z16.s - z19.s }, za.s[w11, 0, vgx4]
        \\st1w { z4.s - z7.s }, pn8, [x1, #8, MUL VL]
        \\st1w { z16.s - z19.s }, pn8, [x1, #12, MUL VL]
        \\mov w8, #2
        \\mov w11, #10
        \\mov { z4.s - z7.s }, za.s[w8, 0, vgx4]
        \\mov { z16.s - z19.s }, za.s[w11, 0, vgx4]
        \\st1w { z4.s - z7.s }, pn8, [x1, #16, MUL VL]
        \\st1w { z16.s - z19.s }, pn8, [x1, #20, MUL VL]
        \\mov w8, #3
        \\mov w11, #11
        \\mov { z4.s - z7.s }, za.s[w8, 0, vgx4]
        \\mov { z16.s - z19.s }, za.s[w11, 0, vgx4]
        \\st1w { z4.s - z7.s }, pn8, [x1, #24, MUL VL]
        \\st1w { z16.s - z19.s }, pn8, [x1, #28, MUL VL]
        \\
        \\add x1, x1, #2048
        \\sub x0, x0, x7
        \\b 0b
        \\
        \\1:
        \\cbz x0, 2f
        \\mov x8, #0
        \\mov z0.s, s0
        \\
        \\4:
        \\whilelo p1.s, x8, x0
        \\b.none 2f
        \\ld1w { z4.s }, p1/z, [x1, x8, lsl #2]
        \\fmul z4.s, p1/m, z4.s, z0.s
        \\st1w { z4.s }, p1, [x1, x8, lsl #2]
        \\incw x8
        \\b 4b
        \\
        \\2:
        \\3:
        \\ret
        ::: .{ .memory = true });
}

noinline fn smeDscalF64Streaming(n: usize, alpha: f64, x: [*]f64) callconv(.naked) void {
    _ = n;
    _ = alpha;
    _ = x;
    asm volatile (
        \\cbz x0, 3f
        \\ptrue pn8.d
        \\mov z0.d, d0
        \\cntd x6
        \\lsl x7, x6, #5
        \\
        \\0:
        \\cmp x0, x7
        \\b.lo 1f
        \\zero { za }
        \\
        \\mov x10, x1
        \\addvl x13, x10, #4
        \\addvl x9, x10, #8
        \\addvl x12, x10, #12
        \\mov w8, #0
        \\mov w11, #8
        \\ld1d { z4.d - z7.d }, pn8/z, [x10]
        \\ld1d { z16.d - z19.d }, pn8/z, [x13]
        \\fmla za.d[w8, 0, vgx4], { z4.d - z7.d }, z0.d
        \\fmla za.d[w11, 0, vgx4], { z16.d - z19.d }, z0.d
        \\mov w8, #1
        \\mov w11, #9
        \\ld1d { z4.d - z7.d }, pn8/z, [x9]
        \\ld1d { z16.d - z19.d }, pn8/z, [x12]
        \\fmla za.d[w8, 0, vgx4], { z4.d - z7.d }, z0.d
        \\fmla za.d[w11, 0, vgx4], { z16.d - z19.d }, z0.d
        \\
        \\addvl x10, x10, #16
        \\addvl x13, x13, #16
        \\addvl x9, x9, #16
        \\addvl x12, x12, #16
        \\mov w8, #2
        \\mov w11, #10
        \\ld1d { z4.d - z7.d }, pn8/z, [x10]
        \\ld1d { z16.d - z19.d }, pn8/z, [x13]
        \\fmla za.d[w8, 0, vgx4], { z4.d - z7.d }, z0.d
        \\fmla za.d[w11, 0, vgx4], { z16.d - z19.d }, z0.d
        \\mov w8, #3
        \\mov w11, #11
        \\ld1d { z4.d - z7.d }, pn8/z, [x9]
        \\ld1d { z16.d - z19.d }, pn8/z, [x12]
        \\fmla za.d[w8, 0, vgx4], { z4.d - z7.d }, z0.d
        \\fmla za.d[w11, 0, vgx4], { z16.d - z19.d }, z0.d
        \\
        \\mov x10, x1
        \\addvl x13, x10, #4
        \\mov w8, #0
        \\mov w11, #8
        \\mov { z4.d - z7.d }, za.d[w8, 0, vgx4]
        \\mov { z16.d - z19.d }, za.d[w11, 0, vgx4]
        \\st1d { z4.d - z7.d }, pn8, [x10]
        \\st1d { z16.d - z19.d }, pn8, [x13]
        \\addvl x10, x10, #8
        \\addvl x13, x13, #8
        \\mov w8, #1
        \\mov w11, #9
        \\mov { z4.d - z7.d }, za.d[w8, 0, vgx4]
        \\mov { z16.d - z19.d }, za.d[w11, 0, vgx4]
        \\st1d { z4.d - z7.d }, pn8, [x10]
        \\st1d { z16.d - z19.d }, pn8, [x13]
        \\addvl x10, x10, #8
        \\addvl x13, x13, #8
        \\mov w8, #2
        \\mov w11, #10
        \\mov { z4.d - z7.d }, za.d[w8, 0, vgx4]
        \\mov { z16.d - z19.d }, za.d[w11, 0, vgx4]
        \\st1d { z4.d - z7.d }, pn8, [x10]
        \\st1d { z16.d - z19.d }, pn8, [x13]
        \\addvl x10, x10, #8
        \\addvl x13, x13, #8
        \\mov w8, #3
        \\mov w11, #11
        \\mov { z4.d - z7.d }, za.d[w8, 0, vgx4]
        \\mov { z16.d - z19.d }, za.d[w11, 0, vgx4]
        \\st1d { z4.d - z7.d }, pn8, [x10]
        \\st1d { z16.d - z19.d }, pn8, [x13]
        \\
        \\addvl x1, x1, #16
        \\addvl x1, x1, #16
        \\sub x0, x0, x7
        \\b 0b
        \\
        \\1:
        \\cbz x0, 2f
        \\mov x8, #0
        \\mov z0.d, d0
        \\
        \\4:
        \\whilelo p1.d, x8, x0
        \\b.none 2f
        \\ld1d { z4.d }, p1/z, [x1, x8, lsl #3]
        \\fmul z4.d, p1/m, z4.d, z0.d
        \\st1d { z4.d }, p1, [x1, x8, lsl #3]
        \\incd x8
        \\b 4b
        \\
        \\2:
        \\3:
        \\ret
        ::: .{ .memory = true });
}

noinline fn smeSasumF32StreamingBits(n: usize, x: [*]const f32) callconv(.naked) u32 {
    _ = n;
    _ = x;
    asm volatile (
        \\cbz x0, 4f
        \\ptrue pn8.s
        \\ptrue p0.s
        \\mov x7, #512
        \\zero { za }
        \\dup z24.s, #0
        \\
        \\0:
        \\cmp x0, x7
        \\b.lo 1f
        \\
        \\mov w8, #0
        \\mov w11, #8
        \\ld1w { z4.s - z7.s }, pn8/z, [x1]
        \\ld1w { z16.s - z19.s }, pn8/z, [x1, #4, MUL VL]
        \\fabs z4.s, p0/m, z4.s
        \\fabs z5.s, p0/m, z5.s
        \\fabs z6.s, p0/m, z6.s
        \\fabs z7.s, p0/m, z7.s
        \\fabs z16.s, p0/m, z16.s
        \\fabs z17.s, p0/m, z17.s
        \\fabs z18.s, p0/m, z18.s
        \\fabs z19.s, p0/m, z19.s
        \\fadd za.s[w8, 0, vgx4], { z4.s - z7.s }
        \\fadd za.s[w11, 0, vgx4], { z16.s - z19.s }
        \\mov w8, #1
        \\mov w11, #9
        \\ld1w { z4.s - z7.s }, pn8/z, [x1, #8, MUL VL]
        \\ld1w { z16.s - z19.s }, pn8/z, [x1, #12, MUL VL]
        \\fabs z4.s, p0/m, z4.s
        \\fabs z5.s, p0/m, z5.s
        \\fabs z6.s, p0/m, z6.s
        \\fabs z7.s, p0/m, z7.s
        \\fabs z16.s, p0/m, z16.s
        \\fabs z17.s, p0/m, z17.s
        \\fabs z18.s, p0/m, z18.s
        \\fabs z19.s, p0/m, z19.s
        \\fadd za.s[w8, 0, vgx4], { z4.s - z7.s }
        \\fadd za.s[w11, 0, vgx4], { z16.s - z19.s }
        \\mov w8, #2
        \\mov w11, #10
        \\ld1w { z4.s - z7.s }, pn8/z, [x1, #16, MUL VL]
        \\ld1w { z16.s - z19.s }, pn8/z, [x1, #20, MUL VL]
        \\fabs z4.s, p0/m, z4.s
        \\fabs z5.s, p0/m, z5.s
        \\fabs z6.s, p0/m, z6.s
        \\fabs z7.s, p0/m, z7.s
        \\fabs z16.s, p0/m, z16.s
        \\fabs z17.s, p0/m, z17.s
        \\fabs z18.s, p0/m, z18.s
        \\fabs z19.s, p0/m, z19.s
        \\fadd za.s[w8, 0, vgx4], { z4.s - z7.s }
        \\fadd za.s[w11, 0, vgx4], { z16.s - z19.s }
        \\mov w8, #3
        \\mov w11, #11
        \\ld1w { z4.s - z7.s }, pn8/z, [x1, #24, MUL VL]
        \\ld1w { z16.s - z19.s }, pn8/z, [x1, #28, MUL VL]
        \\fabs z4.s, p0/m, z4.s
        \\fabs z5.s, p0/m, z5.s
        \\fabs z6.s, p0/m, z6.s
        \\fabs z7.s, p0/m, z7.s
        \\fabs z16.s, p0/m, z16.s
        \\fabs z17.s, p0/m, z17.s
        \\fabs z18.s, p0/m, z18.s
        \\fabs z19.s, p0/m, z19.s
        \\fadd za.s[w8, 0, vgx4], { z4.s - z7.s }
        \\fadd za.s[w11, 0, vgx4], { z16.s - z19.s }
        \\
        \\add x1, x1, #2048
        \\sub x0, x0, x7
        \\b 0b
        \\
        \\1:
        \\cbz x0, 2f
        \\mov x8, #0
        \\
        \\5:
        \\whilelo p1.s, x8, x0
        \\b.none 2f
        \\ld1w { z25.s }, p1/z, [x1, x8, lsl #2]
        \\fabs z25.s, p1/m, z25.s
        \\fadd z24.s, z24.s, z25.s
        \\incw x8
        \\b 5b
        \\
        \\2:
        \\mov w8, #2
        \\mov w11, #10
        \\mov { z4.s - z7.s }, za.s[w8, 0, vgx4]
        \\mov { z16.s - z19.s }, za.s[w11, 0, vgx4]
        \\mov w8, #0
        \\mov w11, #8
        \\fadd za.s[w8, 0, vgx4], { z4.s - z7.s }
        \\fadd za.s[w11, 0, vgx4], { z16.s - z19.s }
        \\mov w8, #3
        \\mov w11, #11
        \\mov { z4.s - z7.s }, za.s[w8, 0, vgx4]
        \\mov { z16.s - z19.s }, za.s[w11, 0, vgx4]
        \\mov w8, #1
        \\mov w11, #9
        \\fadd za.s[w8, 0, vgx4], { z4.s - z7.s }
        \\fadd za.s[w11, 0, vgx4], { z16.s - z19.s }
        \\mov w8, #0
        \\mov w11, #8
        \\mov { z4.s - z7.s }, za.s[w8, 0, vgx4]
        \\mov { z16.s - z19.s }, za.s[w11, 0, vgx4]
        \\mov w8, #1
        \\mov w11, #9
        \\fadd za.s[w8, 0, vgx4], { z4.s - z7.s }
        \\fadd za.s[w11, 0, vgx4], { z16.s - z19.s }
        \\mov { z4.s - z7.s }, za.s[w8, 0, vgx4]
        \\mov { z16.s - z19.s }, za.s[w11, 0, vgx4]
        \\fadd z4.s, z4.s, z5.s
        \\fadd z6.s, z6.s, z7.s
        \\fadd z16.s, z16.s, z17.s
        \\fadd z18.s, z18.s, z19.s
        \\fadd z4.s, z4.s, z6.s
        \\fadd z16.s, z16.s, z18.s
        \\fadd z4.s, z4.s, z16.s
        \\fadd z4.s, z4.s, z24.s
        \\faddv s16, p0, z4.s
        \\fmov w0, s16
        \\ret
        \\
        \\4:
        \\mov w0, wzr
        \\ret
        ::: .{ .memory = true });
}

noinline fn smeDasumF64StreamingBits(n: usize, x: [*]const f64) callconv(.naked) u64 {
    _ = n;
    _ = x;
    asm volatile (
        \\cbz x0, 4f
        \\ptrue pn8.d
        \\ptrue p0.d
        \\mov x7, #256
        \\zero { za }
        \\dup z24.d, #0
        \\
        \\0:
        \\cmp x0, x7
        \\b.lo 1f
        \\
        \\mov w8, #0
        \\mov w11, #8
        \\ld1d { z4.d - z7.d }, pn8/z, [x1]
        \\ld1d { z16.d - z19.d }, pn8/z, [x1, #4, MUL VL]
        \\fabs z4.d, p0/m, z4.d
        \\fabs z5.d, p0/m, z5.d
        \\fabs z6.d, p0/m, z6.d
        \\fabs z7.d, p0/m, z7.d
        \\fabs z16.d, p0/m, z16.d
        \\fabs z17.d, p0/m, z17.d
        \\fabs z18.d, p0/m, z18.d
        \\fabs z19.d, p0/m, z19.d
        \\fadd za.d[w8, 0, vgx4], { z4.d - z7.d }
        \\fadd za.d[w11, 0, vgx4], { z16.d - z19.d }
        \\mov w8, #1
        \\mov w11, #9
        \\ld1d { z4.d - z7.d }, pn8/z, [x1, #8, MUL VL]
        \\ld1d { z16.d - z19.d }, pn8/z, [x1, #12, MUL VL]
        \\fabs z4.d, p0/m, z4.d
        \\fabs z5.d, p0/m, z5.d
        \\fabs z6.d, p0/m, z6.d
        \\fabs z7.d, p0/m, z7.d
        \\fabs z16.d, p0/m, z16.d
        \\fabs z17.d, p0/m, z17.d
        \\fabs z18.d, p0/m, z18.d
        \\fabs z19.d, p0/m, z19.d
        \\fadd za.d[w8, 0, vgx4], { z4.d - z7.d }
        \\fadd za.d[w11, 0, vgx4], { z16.d - z19.d }
        \\
        \\mov w8, #2
        \\mov w11, #10
        \\ld1d { z4.d - z7.d }, pn8/z, [x1, #16, MUL VL]
        \\ld1d { z16.d - z19.d }, pn8/z, [x1, #20, MUL VL]
        \\fabs z4.d, p0/m, z4.d
        \\fabs z5.d, p0/m, z5.d
        \\fabs z6.d, p0/m, z6.d
        \\fabs z7.d, p0/m, z7.d
        \\fabs z16.d, p0/m, z16.d
        \\fabs z17.d, p0/m, z17.d
        \\fabs z18.d, p0/m, z18.d
        \\fabs z19.d, p0/m, z19.d
        \\fadd za.d[w8, 0, vgx4], { z4.d - z7.d }
        \\fadd za.d[w11, 0, vgx4], { z16.d - z19.d }
        \\mov w8, #3
        \\mov w11, #11
        \\ld1d { z4.d - z7.d }, pn8/z, [x1, #24, MUL VL]
        \\ld1d { z16.d - z19.d }, pn8/z, [x1, #28, MUL VL]
        \\fabs z4.d, p0/m, z4.d
        \\fabs z5.d, p0/m, z5.d
        \\fabs z6.d, p0/m, z6.d
        \\fabs z7.d, p0/m, z7.d
        \\fabs z16.d, p0/m, z16.d
        \\fabs z17.d, p0/m, z17.d
        \\fabs z18.d, p0/m, z18.d
        \\fabs z19.d, p0/m, z19.d
        \\fadd za.d[w8, 0, vgx4], { z4.d - z7.d }
        \\fadd za.d[w11, 0, vgx4], { z16.d - z19.d }
        \\
        \\add x1, x1, #2048
        \\sub x0, x0, x7
        \\b 0b
        \\
        \\1:
        \\cbz x0, 2f
        \\mov x8, #0
        \\
        \\5:
        \\whilelo p1.d, x8, x0
        \\b.none 2f
        \\ld1d { z25.d }, p1/z, [x1, x8, lsl #3]
        \\fabs z25.d, p1/m, z25.d
        \\fadd z24.d, z24.d, z25.d
        \\incd x8
        \\b 5b
        \\
        \\2:
        \\mov w8, #2
        \\mov w11, #10
        \\mov { z4.d - z7.d }, za.d[w8, 0, vgx4]
        \\mov { z16.d - z19.d }, za.d[w11, 0, vgx4]
        \\mov w8, #0
        \\mov w11, #8
        \\fadd za.d[w8, 0, vgx4], { z4.d - z7.d }
        \\fadd za.d[w11, 0, vgx4], { z16.d - z19.d }
        \\mov w8, #3
        \\mov w11, #11
        \\mov { z4.d - z7.d }, za.d[w8, 0, vgx4]
        \\mov { z16.d - z19.d }, za.d[w11, 0, vgx4]
        \\mov w8, #1
        \\mov w11, #9
        \\fadd za.d[w8, 0, vgx4], { z4.d - z7.d }
        \\fadd za.d[w11, 0, vgx4], { z16.d - z19.d }
        \\mov w8, #0
        \\mov w11, #8
        \\mov { z4.d - z7.d }, za.d[w8, 0, vgx4]
        \\mov { z16.d - z19.d }, za.d[w11, 0, vgx4]
        \\mov w8, #1
        \\mov w11, #9
        \\fadd za.d[w8, 0, vgx4], { z4.d - z7.d }
        \\fadd za.d[w11, 0, vgx4], { z16.d - z19.d }
        \\mov { z4.d - z7.d }, za.d[w8, 0, vgx4]
        \\mov { z16.d - z19.d }, za.d[w11, 0, vgx4]
        \\fadd z4.d, z4.d, z5.d
        \\fadd z6.d, z6.d, z7.d
        \\fadd z16.d, z16.d, z17.d
        \\fadd z18.d, z18.d, z19.d
        \\fadd z4.d, z4.d, z6.d
        \\fadd z16.d, z16.d, z18.d
        \\fadd z4.d, z4.d, z16.d
        \\fadd z4.d, z4.d, z24.d
        \\faddv d16, p0, z4.d
        \\fmov x0, d16
        \\ret
        \\
        \\4:
        \\mov x0, xzr
        \\ret
        ::: .{ .memory = true });
}
