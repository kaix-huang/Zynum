// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later

//! AArch64 two-vector specialized kernels.

const asm_fragments = @import("asm_fragments.zig");
const features = @import("features.zig");
const types = @import("../../types.zig");
const vector_matrix_asm = @import("vector_matrix_asm.zig");

const enable_sme_copy_bytes = true;
const enable_sme_ddot = true;
const enable_sve_ddot = true;

inline fn callCopyBytesKernel(comptime kernel: anytype, n_bytes: usize, x: [*]const u8, y: [*]u8) void {
    const Kernel = *const fn (usize, [*]const u8, [*]u8) callconv(.c) void;
    @as(Kernel, @ptrCast(&kernel))(n_bytes, x, y);
}

inline fn callDotF64Kernel(comptime kernel: anytype, n: usize, x: [*]const f64, y: [*]const f64) u64 {
    const Kernel = *const fn (usize, [*]const f64, [*]const f64) callconv(.c) u64;
    return @as(Kernel, @ptrCast(&kernel))(n, x, y);
}

inline fn callDotF32Kernel(comptime kernel: anytype, n: usize, x: [*]const f32, y: [*]const f32) u32 {
    const Kernel = *const fn (usize, [*]const f32, [*]const f32) callconv(.c) u32;
    return @as(Kernel, @ptrCast(&kernel))(n, x, y);
}

inline fn callAxpyF32Kernel(comptime kernel: anytype, n: usize, alpha: f32, x: [*]const f32, y: [*]f32) void {
    const Kernel = *const fn (usize, f32, [*]const f32, [*]f32) callconv(.c) void;
    @as(Kernel, @ptrCast(&kernel))(n, alpha, x, y);
}

inline fn callZdotF64Kernel(comptime kernel: anytype, n: usize, x: [*]const types.ComplexF64, y: [*]const types.ComplexF64, out: *types.ComplexF64) void {
    const Kernel = *const fn (usize, [*]const types.ComplexF64, [*]const types.ComplexF64, *types.ComplexF64) callconv(.c) void;
    @as(Kernel, @ptrCast(&kernel))(n, x, y, out);
}

noinline fn smeDcopyBytesStreaming(n_bytes: usize, x: [*]const u8, y: [*]u8) callconv(.naked) void {
    _ = n_bytes;
    _ = x;
    _ = y;
    asm volatile (
        \\cbz x0, 4f
        \\ptrue pn8.b
        \\cntb x6
        \\lsl x7, x6, #3
        \\
        \\0:
        \\cmp x0, x7
        \\b.lo 1f
        \\ld1b { z4.b - z7.b }, pn8/z, [x1]
        \\ld1b { z16.b - z19.b }, pn8/z, [x1, #4, MUL VL]
        \\st1b { z4.b - z7.b }, pn8, [x2]
        \\st1b { z16.b - z19.b }, pn8, [x2, #4, MUL VL]
        \\addvl x1, x1, #8
        \\addvl x2, x2, #8
        \\sub x0, x0, x7
        \\b 0b
        \\
        \\1:
        \\cbz x0, 4f
        \\lsl x7, x6, #2
        \\cmp x0, x7
        \\b.lo 2f
        \\sub x8, x0, x7
        \\ptrue pn9.b
        \\whilelt pn10.b, xzr, x8, vlx4
        \\ld1b { z4.b - z7.b }, pn9/z, [x1]
        \\ld1b { z16.b - z19.b }, pn10/z, [x1, #4, MUL VL]
        \\st1b { z4.b - z7.b }, pn9, [x2]
        \\st1b { z16.b - z19.b }, pn10, [x2, #4, MUL VL]
        \\b 4f
        \\
        \\2:
        \\whilelt pn9.b, xzr, x0, vlx4
        \\ld1b { z4.b - z7.b }, pn9/z, [x1]
        \\st1b { z4.b - z7.b }, pn9, [x2]
        \\
        \\4:
        \\ret
        ::: .{ .memory = true });
}

noinline fn smeDdotF64StreamingBits(n: usize, x: [*]const f64, y: [*]const f64) callconv(.naked) u64 {
    _ = n;
    _ = x;
    _ = y;
    asm volatile (
        \\cbz x0, 4f
        \\ptrue pn8.d
        \\ptrue p0.d
        \\cntd x6
        \\lsl x7, x6, #5
        \\zero { za }
        \\dup z28.d, #0
        \\
        \\0:
        \\cmp x0, x7
        \\b.lo 1f
        \\mov w8, #0
        \\mov w11, #8
        \\ld1d { z4.d - z7.d }, pn8/z, [x1]
        \\ld1d { z16.d - z19.d }, pn8/z, [x2]
        \\ld1d { z20.d - z23.d }, pn8/z, [x1, #4, MUL VL]
        \\ld1d { z24.d - z27.d }, pn8/z, [x2, #4, MUL VL]
        \\fmla za.d[w8, 0, vgx4], { z4.d - z7.d }, { z16.d - z19.d }
        \\fmla za.d[w11, 0, vgx4], { z20.d - z23.d }, { z24.d - z27.d }
        \\mov w8, #1
        \\mov w11, #9
        \\ld1d { z4.d - z7.d }, pn8/z, [x1, #8, MUL VL]
        \\ld1d { z16.d - z19.d }, pn8/z, [x2, #8, MUL VL]
        \\ld1d { z20.d - z23.d }, pn8/z, [x1, #12, MUL VL]
        \\ld1d { z24.d - z27.d }, pn8/z, [x2, #12, MUL VL]
        \\fmla za.d[w8, 0, vgx4], { z4.d - z7.d }, { z16.d - z19.d }
        \\fmla za.d[w11, 0, vgx4], { z20.d - z23.d }, { z24.d - z27.d }
        \\mov w8, #2
        \\mov w11, #10
        \\ld1d { z4.d - z7.d }, pn8/z, [x1, #16, MUL VL]
        \\ld1d { z16.d - z19.d }, pn8/z, [x2, #16, MUL VL]
        \\ld1d { z20.d - z23.d }, pn8/z, [x1, #20, MUL VL]
        \\ld1d { z24.d - z27.d }, pn8/z, [x2, #20, MUL VL]
        \\fmla za.d[w8, 0, vgx4], { z4.d - z7.d }, { z16.d - z19.d }
        \\fmla za.d[w11, 0, vgx4], { z20.d - z23.d }, { z24.d - z27.d }
        \\mov w8, #3
        \\mov w11, #11
        \\ld1d { z4.d - z7.d }, pn8/z, [x1, #24, MUL VL]
        \\ld1d { z16.d - z19.d }, pn8/z, [x2, #24, MUL VL]
        \\ld1d { z20.d - z23.d }, pn8/z, [x1, #28, MUL VL]
        \\ld1d { z24.d - z27.d }, pn8/z, [x2, #28, MUL VL]
        \\fmla za.d[w8, 0, vgx4], { z4.d - z7.d }, { z16.d - z19.d }
        \\fmla za.d[w11, 0, vgx4], { z20.d - z23.d }, { z24.d - z27.d }
        \\addvl x1, x1, #16
        \\addvl x1, x1, #16
        \\addvl x2, x2, #16
        \\addvl x2, x2, #16
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
        \\ld1d { z4.d }, p1/z, [x1, x8, lsl #3]
        \\ld1d { z16.d }, p1/z, [x2, x8, lsl #3]
        \\fmla z28.d, p1/m, z4.d, z16.d
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
        \\fadd z4.d, z4.d, z28.d
        \\faddv d16, p0, z4.d
        \\fmov x0, d16
        \\ret
        \\
        \\4:
        \\mov x0, xzr
        \\ret
        ::: .{ .memory = true });
}

noinline fn smeSdotF32StreamingBits(n: usize, x: [*]const f32, y: [*]const f32) callconv(.naked) u32 {
    _ = n;
    _ = x;
    _ = y;
    asm volatile (
        \\cbz x0, 4f
        \\ptrue pn8.s
        \\ptrue p0.s
        \\mov x7, #512
        \\zero { za }
        \\dup z28.s, #0
        \\
        \\0:
        \\cmp x0, x7
        \\b.lo 1f
        \\mov w8, #0
        \\mov w11, #8
        \\ld1w { z4.s - z7.s }, pn8/z, [x1]
        \\ld1w { z16.s - z19.s }, pn8/z, [x2]
        \\ld1w { z20.s - z23.s }, pn8/z, [x1, #4, MUL VL]
        \\ld1w { z24.s - z27.s }, pn8/z, [x2, #4, MUL VL]
        \\fmla za.s[w8, 0, vgx4], { z4.s - z7.s }, { z16.s - z19.s }
        \\fmla za.s[w11, 0, vgx4], { z20.s - z23.s }, { z24.s - z27.s }
        \\mov w8, #1
        \\mov w11, #9
        \\ld1w { z4.s - z7.s }, pn8/z, [x1, #8, MUL VL]
        \\ld1w { z16.s - z19.s }, pn8/z, [x2, #8, MUL VL]
        \\ld1w { z20.s - z23.s }, pn8/z, [x1, #12, MUL VL]
        \\ld1w { z24.s - z27.s }, pn8/z, [x2, #12, MUL VL]
        \\fmla za.s[w8, 0, vgx4], { z4.s - z7.s }, { z16.s - z19.s }
        \\fmla za.s[w11, 0, vgx4], { z20.s - z23.s }, { z24.s - z27.s }
        \\mov w8, #2
        \\mov w11, #10
        \\ld1w { z4.s - z7.s }, pn8/z, [x1, #16, MUL VL]
        \\ld1w { z16.s - z19.s }, pn8/z, [x2, #16, MUL VL]
        \\ld1w { z20.s - z23.s }, pn8/z, [x1, #20, MUL VL]
        \\ld1w { z24.s - z27.s }, pn8/z, [x2, #20, MUL VL]
        \\fmla za.s[w8, 0, vgx4], { z4.s - z7.s }, { z16.s - z19.s }
        \\fmla za.s[w11, 0, vgx4], { z20.s - z23.s }, { z24.s - z27.s }
        \\mov w8, #3
        \\mov w11, #11
        \\ld1w { z4.s - z7.s }, pn8/z, [x1, #24, MUL VL]
        \\ld1w { z16.s - z19.s }, pn8/z, [x2, #24, MUL VL]
        \\ld1w { z20.s - z23.s }, pn8/z, [x1, #28, MUL VL]
        \\ld1w { z24.s - z27.s }, pn8/z, [x2, #28, MUL VL]
        \\fmla za.s[w8, 0, vgx4], { z4.s - z7.s }, { z16.s - z19.s }
        \\fmla za.s[w11, 0, vgx4], { z20.s - z23.s }, { z24.s - z27.s }
        \\add x1, x1, #2048
        \\add x2, x2, #2048
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
        \\ld1w { z4.s }, p1/z, [x1, x8, lsl #2]
        \\ld1w { z16.s }, p1/z, [x2, x8, lsl #2]
        \\fmla z28.s, p1/m, z4.s, z16.s
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
        \\fadd z4.s, z4.s, z28.s
        \\faddv s16, p0, z4.s
        \\fmov w0, s16
        \\ret
        \\
        \\4:
        \\mov w0, wzr
        \\ret
        ::: .{ .memory = true });
}

noinline fn sveZdotuF64(n: usize, x: [*]const types.ComplexF64, y: [*]const types.ComplexF64, out: *types.ComplexF64) callconv(.naked) void {
    _ = n;
    _ = x;
    _ = y;
    _ = out;
    asm volatile (
        \\cbz x0, 13f
    ++ asm_fragments.ptrue_p0_d ++
        \\cntd x6
        \\lsl x7, x6, #2
        \\dup z0.d, #0
        \\dup z1.d, #0
        \\dup z2.d, #0
        \\dup z3.d, #0
        \\dup z8.d, #0
        \\dup z9.d, #0
        \\dup z10.d, #0
        \\dup z11.d, #0
        \\
        \\10:
        \\cmp x0, x7
        \\b.lo 11f
        \\ld2d { z4.d, z5.d }, p0/z, [x1]
        \\ld2d { z16.d, z17.d }, p0/z, [x2]
        \\fmla z0.d, p0/m, z4.d, z16.d
        \\fmls z0.d, p0/m, z5.d, z17.d
        \\fmla z8.d, p0/m, z4.d, z17.d
        \\fmla z8.d, p0/m, z5.d, z16.d
        \\
        \\ld2d { z4.d, z5.d }, p0/z, [x1, #2, MUL VL]
        \\ld2d { z16.d, z17.d }, p0/z, [x2, #2, MUL VL]
        \\fmla z1.d, p0/m, z4.d, z16.d
        \\fmls z1.d, p0/m, z5.d, z17.d
        \\fmla z9.d, p0/m, z4.d, z17.d
        \\fmla z9.d, p0/m, z5.d, z16.d
        \\
        \\ld2d { z4.d, z5.d }, p0/z, [x1, #4, MUL VL]
        \\ld2d { z16.d, z17.d }, p0/z, [x2, #4, MUL VL]
        \\fmla z2.d, p0/m, z4.d, z16.d
        \\fmls z2.d, p0/m, z5.d, z17.d
        \\fmla z10.d, p0/m, z4.d, z17.d
        \\fmla z10.d, p0/m, z5.d, z16.d
        \\
        \\ld2d { z4.d, z5.d }, p0/z, [x1, #6, MUL VL]
        \\ld2d { z16.d, z17.d }, p0/z, [x2, #6, MUL VL]
        \\fmla z3.d, p0/m, z4.d, z16.d
        \\fmls z3.d, p0/m, z5.d, z17.d
        \\fmla z11.d, p0/m, z4.d, z17.d
        \\fmla z11.d, p0/m, z5.d, z16.d
        \\
        \\addvl x1, x1, #8
        \\addvl x2, x2, #8
        \\sub x0, x0, x7
        \\b 10b
        \\
        \\11:
        \\cbz x0, 12f
        \\
        \\14:
        \\whilelo p1.d, xzr, x0
        \\b.none 12f
        \\ld2d { z4.d, z5.d }, p1/z, [x1]
        \\ld2d { z16.d, z17.d }, p1/z, [x2]
        \\fmla z0.d, p1/m, z4.d, z16.d
        \\fmls z0.d, p1/m, z5.d, z17.d
        \\fmla z8.d, p1/m, z4.d, z17.d
        \\fmla z8.d, p1/m, z5.d, z16.d
        \\cmp x0, x6
        \\b.ls 12f
        \\sub x0, x0, x6
        \\addvl x1, x1, #2
        \\addvl x2, x2, #2
        \\b 14b
        \\
        \\12:
        \\fadd z0.d, z0.d, z1.d
        \\fadd z2.d, z2.d, z3.d
        \\fadd z0.d, z0.d, z2.d
        \\fadd z8.d, z8.d, z9.d
        \\fadd z10.d, z10.d, z11.d
        \\fadd z8.d, z8.d, z10.d
        \\faddv d0, p0, z0.d
        \\faddv d1, p0, z8.d
        \\str d0, [x3]
        \\str d1, [x3, #8]
        \\ret
        \\
        \\13:
        \\str xzr, [x3]
        \\str xzr, [x3, #8]
        \\ret
    ::: .{ .memory = true });
}

noinline fn sveZdotcF64(n: usize, x: [*]const types.ComplexF64, y: [*]const types.ComplexF64, out: *types.ComplexF64) callconv(.naked) void {
    _ = n;
    _ = x;
    _ = y;
    _ = out;
    asm volatile (
        \\cbz x0, 13f
    ++ asm_fragments.ptrue_p0_d ++
        \\cntd x6
        \\lsl x7, x6, #2
        \\dup z0.d, #0
        \\dup z1.d, #0
        \\dup z2.d, #0
        \\dup z3.d, #0
        \\dup z8.d, #0
        \\dup z9.d, #0
        \\dup z10.d, #0
        \\dup z11.d, #0
        \\
        \\10:
        \\cmp x0, x7
        \\b.lo 11f
        \\ld2d { z4.d, z5.d }, p0/z, [x1]
        \\ld2d { z16.d, z17.d }, p0/z, [x2]
        \\fmla z0.d, p0/m, z4.d, z16.d
        \\fmla z0.d, p0/m, z5.d, z17.d
        \\fmla z8.d, p0/m, z4.d, z17.d
        \\fmls z8.d, p0/m, z5.d, z16.d
        \\
        \\ld2d { z4.d, z5.d }, p0/z, [x1, #2, MUL VL]
        \\ld2d { z16.d, z17.d }, p0/z, [x2, #2, MUL VL]
        \\fmla z1.d, p0/m, z4.d, z16.d
        \\fmla z1.d, p0/m, z5.d, z17.d
        \\fmla z9.d, p0/m, z4.d, z17.d
        \\fmls z9.d, p0/m, z5.d, z16.d
        \\
        \\ld2d { z4.d, z5.d }, p0/z, [x1, #4, MUL VL]
        \\ld2d { z16.d, z17.d }, p0/z, [x2, #4, MUL VL]
        \\fmla z2.d, p0/m, z4.d, z16.d
        \\fmla z2.d, p0/m, z5.d, z17.d
        \\fmla z10.d, p0/m, z4.d, z17.d
        \\fmls z10.d, p0/m, z5.d, z16.d
        \\
        \\ld2d { z4.d, z5.d }, p0/z, [x1, #6, MUL VL]
        \\ld2d { z16.d, z17.d }, p0/z, [x2, #6, MUL VL]
        \\fmla z3.d, p0/m, z4.d, z16.d
        \\fmla z3.d, p0/m, z5.d, z17.d
        \\fmla z11.d, p0/m, z4.d, z17.d
        \\fmls z11.d, p0/m, z5.d, z16.d
        \\
        \\addvl x1, x1, #8
        \\addvl x2, x2, #8
        \\sub x0, x0, x7
        \\b 10b
        \\
        \\11:
        \\cbz x0, 12f
        \\
        \\14:
        \\whilelo p1.d, xzr, x0
        \\b.none 12f
        \\ld2d { z4.d, z5.d }, p1/z, [x1]
        \\ld2d { z16.d, z17.d }, p1/z, [x2]
        \\fmla z0.d, p1/m, z4.d, z16.d
        \\fmla z0.d, p1/m, z5.d, z17.d
        \\fmla z8.d, p1/m, z4.d, z17.d
        \\fmls z8.d, p1/m, z5.d, z16.d
        \\cmp x0, x6
        \\b.ls 12f
        \\sub x0, x0, x6
        \\addvl x1, x1, #2
        \\addvl x2, x2, #2
        \\b 14b
        \\
        \\12:
        \\fadd z0.d, z0.d, z1.d
        \\fadd z2.d, z2.d, z3.d
        \\fadd z0.d, z0.d, z2.d
        \\fadd z8.d, z8.d, z9.d
        \\fadd z10.d, z10.d, z11.d
        \\fadd z8.d, z8.d, z10.d
        \\faddv d0, p0, z0.d
        \\faddv d1, p0, z8.d
        \\str d0, [x3]
        \\str d1, [x3, #8]
        \\ret
        \\
        \\13:
        \\str xzr, [x3]
        \\str xzr, [x3, #8]
        \\ret
    ::: .{ .memory = true });
}

noinline fn smeSaxpyF32Streaming(n: usize, alpha: f32, x: [*]const f32, y: [*]f32) callconv(.naked) void {
    _ = n;
    _ = alpha;
    _ = x;
    _ = y;
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
        \\ld1w { z16.s - z19.s }, pn8/z, [x2]
        \\fadd za.s[w8, 0, vgx4], { z16.s - z19.s }
        \\fmla za.s[w8, 0, vgx4], { z4.s - z7.s }, z0.s
        \\ld1w { z4.s - z7.s }, pn8/z, [x1, #4, MUL VL]
        \\ld1w { z16.s - z19.s }, pn8/z, [x2, #4, MUL VL]
        \\fadd za.s[w11, 0, vgx4], { z16.s - z19.s }
        \\fmla za.s[w11, 0, vgx4], { z4.s - z7.s }, z0.s
        \\mov w8, #1
        \\mov w11, #9
        \\ld1w { z4.s - z7.s }, pn8/z, [x1, #8, MUL VL]
        \\ld1w { z16.s - z19.s }, pn8/z, [x2, #8, MUL VL]
        \\fadd za.s[w8, 0, vgx4], { z16.s - z19.s }
        \\fmla za.s[w8, 0, vgx4], { z4.s - z7.s }, z0.s
        \\ld1w { z4.s - z7.s }, pn8/z, [x1, #12, MUL VL]
        \\ld1w { z16.s - z19.s }, pn8/z, [x2, #12, MUL VL]
        \\fadd za.s[w11, 0, vgx4], { z16.s - z19.s }
        \\fmla za.s[w11, 0, vgx4], { z4.s - z7.s }, z0.s
        \\mov w8, #2
        \\mov w11, #10
        \\ld1w { z4.s - z7.s }, pn8/z, [x1, #16, MUL VL]
        \\ld1w { z16.s - z19.s }, pn8/z, [x2, #16, MUL VL]
        \\fadd za.s[w8, 0, vgx4], { z16.s - z19.s }
        \\fmla za.s[w8, 0, vgx4], { z4.s - z7.s }, z0.s
        \\ld1w { z4.s - z7.s }, pn8/z, [x1, #20, MUL VL]
        \\ld1w { z16.s - z19.s }, pn8/z, [x2, #20, MUL VL]
        \\fadd za.s[w11, 0, vgx4], { z16.s - z19.s }
        \\fmla za.s[w11, 0, vgx4], { z4.s - z7.s }, z0.s
        \\mov w8, #3
        \\mov w11, #11
        \\ld1w { z4.s - z7.s }, pn8/z, [x1, #24, MUL VL]
        \\ld1w { z16.s - z19.s }, pn8/z, [x2, #24, MUL VL]
        \\fadd za.s[w8, 0, vgx4], { z16.s - z19.s }
        \\fmla za.s[w8, 0, vgx4], { z4.s - z7.s }, z0.s
        \\ld1w { z4.s - z7.s }, pn8/z, [x1, #28, MUL VL]
        \\ld1w { z16.s - z19.s }, pn8/z, [x2, #28, MUL VL]
        \\fadd za.s[w11, 0, vgx4], { z16.s - z19.s }
        \\fmla za.s[w11, 0, vgx4], { z4.s - z7.s }, z0.s
        \\
        \\mov w8, #0
        \\mov w11, #8
        \\mov { z4.s - z7.s }, za.s[w8, 0, vgx4]
        \\mov { z16.s - z19.s }, za.s[w11, 0, vgx4]
        \\st1w { z4.s - z7.s }, pn8, [x2]
        \\st1w { z16.s - z19.s }, pn8, [x2, #4, MUL VL]
        \\mov w8, #1
        \\mov w11, #9
        \\mov { z4.s - z7.s }, za.s[w8, 0, vgx4]
        \\mov { z16.s - z19.s }, za.s[w11, 0, vgx4]
        \\st1w { z4.s - z7.s }, pn8, [x2, #8, MUL VL]
        \\st1w { z16.s - z19.s }, pn8, [x2, #12, MUL VL]
        \\mov w8, #2
        \\mov w11, #10
        \\mov { z4.s - z7.s }, za.s[w8, 0, vgx4]
        \\mov { z16.s - z19.s }, za.s[w11, 0, vgx4]
        \\st1w { z4.s - z7.s }, pn8, [x2, #16, MUL VL]
        \\st1w { z16.s - z19.s }, pn8, [x2, #20, MUL VL]
        \\mov w8, #3
        \\mov w11, #11
        \\mov { z4.s - z7.s }, za.s[w8, 0, vgx4]
        \\mov { z16.s - z19.s }, za.s[w11, 0, vgx4]
        \\st1w { z4.s - z7.s }, pn8, [x2, #24, MUL VL]
        \\st1w { z16.s - z19.s }, pn8, [x2, #28, MUL VL]
        \\
        \\add x1, x1, #2048
        \\add x2, x2, #2048
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
        \\ld1w { z16.s }, p1/z, [x2, x8, lsl #2]
        \\fmla z16.s, p1/m, z4.s, z0.s
        \\st1w { z16.s }, p1, [x2, x8, lsl #2]
        \\incw x8
        \\b 4b
        \\
        \\2:
        \\3:
        \\ret
        ::: .{ .memory = true });
}

pub fn copyBytes(n_bytes: usize, x: [*]const u8, y: [*]u8) bool {
    if (n_bytes == 0) return true;
    if (comptime enable_sme_copy_bytes and features.has_sme2) {
        if (n_bytes < 8 * 1024 or n_bytes >= 16 * 1024 * 1024 or features.streamingVectorBytes() != 64) return false;
        var sm_state: features.StreamingModeState = undefined;
        sm_state.startSmZa();
        defer sm_state.stopSmZa();
        callCopyBytesKernel(smeDcopyBytesStreaming, n_bytes, x, y);
        return true;
    }
    return false;
}

pub fn copyUnit(comptime T: type, n: usize, x: [*]const T, y: [*]T) bool {
    return copyBytes(n * @sizeOf(T), @ptrCast(x), @ptrCast(y));
}

pub fn copyUnitReal(comptime T: type, n: usize, x: [*]const T, y: [*]T) bool {
    return copyUnit(T, n, x, y);
}

pub fn axpyUnitReal(comptime T: type, n: usize, alpha: T, x: [*]const T, y: [*]T) bool {
    if (comptime features.has_sme2) {
        if (T == f32 and n >= 64 * 1024 and features.streamingVectorBytes() == 64) {
            var sm_state: features.StreamingModeState = undefined;
            sm_state.startSmZa();
            defer sm_state.stopSmZa();
            callAxpyF32Kernel(smeSaxpyF32Streaming, n, alpha, x, y);
            return true;
        }
    }
    return false;
}

pub fn dotUnitReal(comptime T: type, n: usize, x: [*]const T, y: [*]const T) ?T {
    if (comptime enable_sme_ddot and features.has_sme2) {
        if (T == f32 and n >= 64 * 1024 and features.streamingVectorBytes() == 64) {
            var sm_state: features.StreamingModeState = undefined;
            sm_state.startSmZa();
            const result_bits = callDotF32Kernel(smeSdotF32StreamingBits, n, x, y);
            const stopped_result_bits = sm_state.stopSmZaRetU32(result_bits);
            return @bitCast(stopped_result_bits);
        }
        if (T == f64 and n >= 64 * 1024 and features.streamingVectorBytes() == 64) {
            var sm_state: features.StreamingModeState = undefined;
            sm_state.startSmZa();
            const result_bits = callDotF64Kernel(smeDdotF64StreamingBits, n, x, y);
            const stopped_result_bits = sm_state.stopSmZaRetU64(result_bits);
            return @bitCast(stopped_result_bits);
        }
    }
    if (comptime enable_sve_ddot and features.has_sve) {
        if (T == f64 and n >= 16) {
            const result_bits = callDotF64Kernel(vector_matrix_asm.sveDdotF64Bits, n, x, y);
            return @bitCast(result_bits);
        }
    }
    return null;
}

pub fn dotUnitComplex(comptime T: type, n: usize, x: [*]const T, y: [*]const T, conjx: bool) ?T {
    if (comptime features.has_sve and T == types.ComplexF64) {
        if (n >= 64) {
            var out: T = undefined;
            if (conjx) {
                callZdotF64Kernel(sveZdotcF64, n, x, y, &out);
            } else {
                callZdotF64Kernel(sveZdotuF64, n, x, y, &out);
            }
            return out;
        }
    }
    return null;
}
