// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later

//! AArch64 matrix-vector whole-function asm entry points.

const builders = @import("builders.zig");

pub noinline fn dgemvTransSveF64(m: usize, n: usize, alpha: f64, a: [*]const f64, lda_bytes: usize, x: [*]const f64, y: [*]f64) callconv(.naked) void {
    _ = m;
    _ = n;
    _ = alpha;
    _ = a;
    _ = lda_bytes;
    _ = x;
    _ = y;
    asm volatile (builders.dgemvTransSveF64Asm() ::: .{ .memory = true });
}

pub noinline fn dgemvTransSveF64FullN8(m: usize, n: usize, alpha: f64, a: [*]const f64, lda_bytes: usize, x: [*]const f64, y: [*]f64) callconv(.naked) void {
    _ = m;
    _ = n;
    _ = alpha;
    _ = a;
    _ = lda_bytes;
    _ = x;
    _ = y;
    asm volatile (builders.dgemvTransSveF64FullN8Asm() ::: .{ .memory = true });
}

pub noinline fn dgemvTransSveF64FullN8Acc2(m: usize, n: usize, alpha: f64, a: [*]const f64, lda_bytes: usize, x: [*]const f64, y: [*]f64) callconv(.naked) void {
    _ = m;
    _ = n;
    _ = alpha;
    _ = a;
    _ = lda_bytes;
    _ = x;
    _ = y;
    asm volatile (builders.dgemvTransSveF64FullN8Acc2Asm() ::: .{ .memory = true });
}

pub noinline fn zgemvNoTransSme2C64512x1(m: usize, n: usize, alpha_re_bits: u64, alpha_im_bits: u64, a: [*]const f64, lda_bytes: usize, x: [*]const f64, y: [*]f64) callconv(.naked) void {
    _ = m;
    _ = n;
    _ = alpha_re_bits;
    _ = alpha_im_bits;
    _ = a;
    _ = lda_bytes;
    _ = x;
    _ = y;
    asm volatile (builders.zgemvNoTransSme2C64512x1Asm() ::: .{ .memory = true });
}

pub noinline fn zgemvNoTransSme2C6464x1(m: usize, n: usize, alpha_re_bits: u64, alpha_im_bits: u64, a: [*]const f64, lda_bytes: usize, x: [*]const f64, y: [*]f64) callconv(.naked) void {
    _ = m;
    _ = n;
    _ = alpha_re_bits;
    _ = alpha_im_bits;
    _ = a;
    _ = lda_bytes;
    _ = x;
    _ = y;
    asm volatile (builders.zgemvNoTransSme2C6464x1Asm() ::: .{ .memory = true });
}

pub noinline fn zgemvNoTransFcmlaF64M64N512Rows(m: usize, n: usize, alpha_re_bits: u64, alpha_im_bits: u64, a: [*]const f64, lda_bytes: usize, x: [*]const f64, y: [*]f64) callconv(.naked) void {
    _ = m;
    _ = n;
    _ = alpha_re_bits;
    _ = alpha_im_bits;
    _ = a;
    _ = lda_bytes;
    _ = x;
    _ = y;
    asm volatile (builders.zgemvNoTransFcmlaF64M64N512RowsAsm() ::: .{ .memory = true });
}

pub noinline fn cgemvTransFcmlaF32M128(alpha_re_bits: u32, alpha_im_bits: u32, beta_re_bits: u32, beta_im_bits: u32, a: [*]const f32, lda_bytes: usize, x: [*]const f32, y: [*]f32) callconv(.naked) void {
    _ = alpha_re_bits;
    _ = alpha_im_bits;
    _ = beta_re_bits;
    _ = beta_im_bits;
    _ = a;
    _ = lda_bytes;
    _ = x;
    _ = y;
    asm volatile (builders.cgemvTransFcmlaF32M128Asm() ::: .{ .memory = true });
}

pub noinline fn cgemvNoTransFcmlaF32M128(alpha_re_bits: u32, alpha_im_bits: u32, beta_re_bits: u32, beta_im_bits: u32, a: [*]const f32, lda_bytes: usize, x: [*]const f32, y: [*]f32) callconv(.naked) void {
    _ = alpha_re_bits;
    _ = alpha_im_bits;
    _ = beta_re_bits;
    _ = beta_im_bits;
    _ = a;
    _ = lda_bytes;
    _ = x;
    _ = y;
    asm volatile (builders.cgemvNoTransFcmlaF32M128Asm() ::: .{ .memory = true });
}

pub noinline fn cgemvNoTransFcmlaF32M512N64Task(alpha_re_bits: u32, alpha_im_bits: u32, beta_re_bits: u32, beta_im_bits: u32, a: [*]const f32, lda_bytes: usize, x: [*]const f32, y: [*]f32) callconv(.naked) void {
    _ = alpha_re_bits;
    _ = alpha_im_bits;
    _ = beta_re_bits;
    _ = beta_im_bits;
    _ = a;
    _ = lda_bytes;
    _ = x;
    _ = y;
    asm volatile (builders.cgemvNoTransFcmlaF32M512N64TaskNoMemsetAsm() ::: .{ .memory = true });
}

pub noinline fn zgemvNoTransFcmlaF64M128(alpha_re_bits: u64, alpha_im_bits: u64, beta_re_bits: u64, beta_im_bits: u64, a: [*]const f64, lda_bytes: usize, x: [*]const f64, y: [*]f64) callconv(.naked) void {
    _ = alpha_re_bits;
    _ = alpha_im_bits;
    _ = beta_re_bits;
    _ = beta_im_bits;
    _ = a;
    _ = lda_bytes;
    _ = x;
    _ = y;
    asm volatile (builders.zgemvNoTransFcmlaF64M128Asm() ::: .{ .memory = true });
}

pub noinline fn zgemvNoTransFcmlaF64M512N64Task(alpha_re_bits: u64, alpha_im_bits: u64, beta_re_bits: u64, beta_im_bits: u64, a: [*]const f64, lda_bytes: usize, x: [*]const f64, y: [*]f64) callconv(.naked) void {
    _ = alpha_re_bits;
    _ = alpha_im_bits;
    _ = beta_re_bits;
    _ = beta_im_bits;
    _ = a;
    _ = lda_bytes;
    _ = x;
    _ = y;
    asm volatile (builders.zgemvNoTransFcmlaF64M512N64TaskNoMemsetAsm() ::: .{ .memory = true });
}

pub noinline fn zgemvNoTransFcmlaF64M512NTask(alpha_re_bits: u64, alpha_im_bits: u64, panel_count: usize, a: [*]const f64, lda_bytes: usize, x: [*]const f64, y: [*]f64) callconv(.naked) void {
    _ = alpha_re_bits;
    _ = alpha_im_bits;
    _ = panel_count;
    _ = a;
    _ = lda_bytes;
    _ = x;
    _ = y;
    asm volatile (builders.zgemvNoTransFcmlaF64M512NTaskNoMemsetAsm() ::: .{ .memory = true });
}

pub noinline fn zgemvTransFcmlaF64M128(alpha_re_bits: u64, alpha_im_bits: u64, beta_re_bits: u64, beta_im_bits: u64, a: [*]const f64, lda_bytes: usize, x: [*]const f64, y: [*]f64) callconv(.naked) void {
    _ = alpha_re_bits;
    _ = alpha_im_bits;
    _ = beta_re_bits;
    _ = beta_im_bits;
    _ = a;
    _ = lda_bytes;
    _ = x;
    _ = y;
    asm volatile (builders.zgemvTransFcmlaF64M128Asm() ::: .{ .memory = true });
}

pub noinline fn zgemvConjTransFcmlaF64M128(alpha_re_bits: u64, alpha_im_bits: u64, beta_re_bits: u64, beta_im_bits: u64, a: [*]const f64, lda_bytes: usize, x: [*]const f64, y: [*]f64) callconv(.naked) void {
    _ = alpha_re_bits;
    _ = alpha_im_bits;
    _ = beta_re_bits;
    _ = beta_im_bits;
    _ = a;
    _ = lda_bytes;
    _ = x;
    _ = y;
    asm volatile (builders.zgemvConjTransFcmlaF64M128Asm() ::: .{ .memory = true });
}

pub noinline fn zgemvTransFcmlaF64M128Cols8(alpha_re_bits: u64, alpha_im_bits: u64, beta_re_bits: u64, beta_im_bits: u64, a: [*]const f64, lda_bytes: usize, x: [*]const f64, y: [*]f64) callconv(.naked) void {
    _ = alpha_re_bits;
    _ = alpha_im_bits;
    _ = beta_re_bits;
    _ = beta_im_bits;
    _ = a;
    _ = lda_bytes;
    _ = x;
    _ = y;
    asm volatile (builders.zgemvTransFcmlaF64M128Cols8Asm() ::: .{ .memory = true });
}

pub noinline fn zgemvConjTransFcmlaF64M128Cols8(alpha_re_bits: u64, alpha_im_bits: u64, beta_re_bits: u64, beta_im_bits: u64, a: [*]const f64, lda_bytes: usize, x: [*]const f64, y: [*]f64) callconv(.naked) void {
    _ = alpha_re_bits;
    _ = alpha_im_bits;
    _ = beta_re_bits;
    _ = beta_im_bits;
    _ = a;
    _ = lda_bytes;
    _ = x;
    _ = y;
    asm volatile (builders.zgemvConjTransFcmlaF64M128Cols8Asm() ::: .{ .memory = true });
}

pub noinline fn zgemvTransFcmlaF64M256N128Task(alpha_re_bits: u64, alpha_im_bits: u64, beta_re_bits: u64, beta_im_bits: u64, a: [*]const f64, lda_bytes: usize, x: [*]const f64, y: [*]f64) callconv(.naked) void {
    _ = alpha_re_bits;
    _ = alpha_im_bits;
    _ = beta_re_bits;
    _ = beta_im_bits;
    _ = a;
    _ = lda_bytes;
    _ = x;
    _ = y;
    asm volatile (builders.zgemvTransFcmlaF64M256N128TaskAsm() ::: .{ .memory = true });
}

pub noinline fn zgemvTransFcmlaF64M512N64Task(alpha_re_bits: u64, alpha_im_bits: u64, beta_re_bits: u64, beta_im_bits: u64, a: [*]const f64, lda_bytes: usize, x: [*]const f64, y: [*]f64) callconv(.naked) void {
    _ = alpha_re_bits;
    _ = alpha_im_bits;
    _ = beta_re_bits;
    _ = beta_im_bits;
    _ = a;
    _ = lda_bytes;
    _ = x;
    _ = y;
    asm volatile (builders.zgemvTransFcmlaF64M512N64TaskAsm() ::: .{ .memory = true });
}

pub noinline fn zgemvTransFcmlaF64M512N64TaskBeta(alpha_re_bits: u64, alpha_im_bits: u64, beta_re_bits: u64, beta_im_bits: u64, a: [*]const f64, lda_bytes: usize, x: [*]const f64, y: [*]f64) callconv(.naked) void {
    _ = alpha_re_bits;
    _ = alpha_im_bits;
    _ = beta_re_bits;
    _ = beta_im_bits;
    _ = a;
    _ = lda_bytes;
    _ = x;
    _ = y;
    asm volatile (builders.zgemvTransFcmlaF64M512N64TaskBetaAsm() ::: .{ .memory = true });
}

pub noinline fn zgemvConjTransFcmlaF64M512N64Task(alpha_re_bits: u64, alpha_im_bits: u64, beta_re_bits: u64, beta_im_bits: u64, a: [*]const f64, lda_bytes: usize, x: [*]const f64, y: [*]f64) callconv(.naked) void {
    _ = alpha_re_bits;
    _ = alpha_im_bits;
    _ = beta_re_bits;
    _ = beta_im_bits;
    _ = a;
    _ = lda_bytes;
    _ = x;
    _ = y;
    asm volatile (builders.zgemvConjTransFcmlaF64M512N64TaskAsm() ::: .{ .memory = true });
}

pub noinline fn zgemvConjTransFcmlaF64M512N64TaskBeta(alpha_re_bits: u64, alpha_im_bits: u64, beta_re_bits: u64, beta_im_bits: u64, a: [*]const f64, lda_bytes: usize, x: [*]const f64, y: [*]f64) callconv(.naked) void {
    _ = alpha_re_bits;
    _ = alpha_im_bits;
    _ = beta_re_bits;
    _ = beta_im_bits;
    _ = a;
    _ = lda_bytes;
    _ = x;
    _ = y;
    asm volatile (builders.zgemvConjTransFcmlaF64M512N64TaskBetaAsm() ::: .{ .memory = true });
}

pub noinline fn dgemvNoTransSmF64M128(m: usize, n: usize, alpha_bits: u64, beta_bits: u64, a: [*]const f64, lda_bytes: usize, x: [*]const f64, y: [*]f64) callconv(.naked) void {
    _ = m;
    _ = n;
    _ = alpha_bits;
    _ = beta_bits;
    _ = a;
    _ = lda_bytes;
    _ = x;
    _ = y;
    asm volatile (
        \\cbz x0, 4f
        \\cbz x1, 4f
    ++ builders.ptrue_p0_d ++
        \\
        \\dup z0.d, #0
        \\dup z1.d, #0
        \\dup z2.d, #0
        \\dup z3.d, #0
        \\dup z4.d, #0
        \\dup z5.d, #0
        \\dup z6.d, #0
        \\dup z7.d, #0
        \\dup z20.d, #0
        \\dup z21.d, #0
        \\dup z22.d, #0
        \\dup z23.d, #0
        \\dup z24.d, #0
        \\dup z25.d, #0
        \\dup z26.d, #0
        \\dup z27.d, #0
        \\
        \\mov x8, x4
        \\mov x9, x6
        \\mov x10, x1
        \\
        \\1:
        \\ldr d28, [x9]
        \\mov z28.d, d28
        \\ldr d29, [x9, #8]
        \\mov z29.d, d29
        \\ldr d30, [x9, #16]
        \\mov z30.d, d30
        \\ldr d31, [x9, #24]
        \\mov z31.d, d31
        \\add x11, x8, x5
        \\add x12, x11, x5
        \\add x13, x12, x5
        \\
        \\ld1d { z16.d }, p0/z, [x8]
        \\ld1d { z17.d }, p0/z, [x11]
        \\ld1d { z18.d }, p0/z, [x12]
        \\ld1d { z19.d }, p0/z, [x13]
        \\fmla z0.d, p0/m, z16.d, z28.d
        \\fmla z0.d, p0/m, z17.d, z29.d
        \\fmla z0.d, p0/m, z18.d, z30.d
        \\fmla z0.d, p0/m, z19.d, z31.d
        \\ld1d { z16.d }, p0/z, [x8, #1, mul vl]
        \\ld1d { z17.d }, p0/z, [x11, #1, mul vl]
        \\ld1d { z18.d }, p0/z, [x12, #1, mul vl]
        \\ld1d { z19.d }, p0/z, [x13, #1, mul vl]
        \\fmla z1.d, p0/m, z16.d, z28.d
        \\fmla z1.d, p0/m, z17.d, z29.d
        \\fmla z1.d, p0/m, z18.d, z30.d
        \\fmla z1.d, p0/m, z19.d, z31.d
        \\ld1d { z16.d }, p0/z, [x8, #2, mul vl]
        \\ld1d { z17.d }, p0/z, [x11, #2, mul vl]
        \\ld1d { z18.d }, p0/z, [x12, #2, mul vl]
        \\ld1d { z19.d }, p0/z, [x13, #2, mul vl]
        \\fmla z2.d, p0/m, z16.d, z28.d
        \\fmla z2.d, p0/m, z17.d, z29.d
        \\fmla z2.d, p0/m, z18.d, z30.d
        \\fmla z2.d, p0/m, z19.d, z31.d
        \\ld1d { z16.d }, p0/z, [x8, #3, mul vl]
        \\ld1d { z17.d }, p0/z, [x11, #3, mul vl]
        \\ld1d { z18.d }, p0/z, [x12, #3, mul vl]
        \\ld1d { z19.d }, p0/z, [x13, #3, mul vl]
        \\fmla z3.d, p0/m, z16.d, z28.d
        \\fmla z3.d, p0/m, z17.d, z29.d
        \\fmla z3.d, p0/m, z18.d, z30.d
        \\fmla z3.d, p0/m, z19.d, z31.d
        \\ld1d { z16.d }, p0/z, [x8, #4, mul vl]
        \\ld1d { z17.d }, p0/z, [x11, #4, mul vl]
        \\ld1d { z18.d }, p0/z, [x12, #4, mul vl]
        \\ld1d { z19.d }, p0/z, [x13, #4, mul vl]
        \\fmla z4.d, p0/m, z16.d, z28.d
        \\fmla z4.d, p0/m, z17.d, z29.d
        \\fmla z4.d, p0/m, z18.d, z30.d
        \\fmla z4.d, p0/m, z19.d, z31.d
        \\ld1d { z16.d }, p0/z, [x8, #5, mul vl]
        \\ld1d { z17.d }, p0/z, [x11, #5, mul vl]
        \\ld1d { z18.d }, p0/z, [x12, #5, mul vl]
        \\ld1d { z19.d }, p0/z, [x13, #5, mul vl]
        \\fmla z5.d, p0/m, z16.d, z28.d
        \\fmla z5.d, p0/m, z17.d, z29.d
        \\fmla z5.d, p0/m, z18.d, z30.d
        \\fmla z5.d, p0/m, z19.d, z31.d
        \\ld1d { z16.d }, p0/z, [x8, #6, mul vl]
        \\ld1d { z17.d }, p0/z, [x11, #6, mul vl]
        \\ld1d { z18.d }, p0/z, [x12, #6, mul vl]
        \\ld1d { z19.d }, p0/z, [x13, #6, mul vl]
        \\fmla z6.d, p0/m, z16.d, z28.d
        \\fmla z6.d, p0/m, z17.d, z29.d
        \\fmla z6.d, p0/m, z18.d, z30.d
        \\fmla z6.d, p0/m, z19.d, z31.d
        \\ld1d { z16.d }, p0/z, [x8, #7, mul vl]
        \\ld1d { z17.d }, p0/z, [x11, #7, mul vl]
        \\ld1d { z18.d }, p0/z, [x12, #7, mul vl]
        \\ld1d { z19.d }, p0/z, [x13, #7, mul vl]
        \\fmla z7.d, p0/m, z16.d, z28.d
        \\fmla z7.d, p0/m, z17.d, z29.d
        \\fmla z7.d, p0/m, z18.d, z30.d
        \\fmla z7.d, p0/m, z19.d, z31.d
        \\addvl x14, x8, #8
        \\addvl x15, x11, #8
        \\addvl x16, x12, #8
        \\addvl x17, x13, #8
        \\ld1d { z16.d }, p0/z, [x14]
        \\ld1d { z17.d }, p0/z, [x15]
        \\ld1d { z18.d }, p0/z, [x16]
        \\ld1d { z19.d }, p0/z, [x17]
        \\fmla z20.d, p0/m, z16.d, z28.d
        \\fmla z20.d, p0/m, z17.d, z29.d
        \\fmla z20.d, p0/m, z18.d, z30.d
        \\fmla z20.d, p0/m, z19.d, z31.d
        \\ld1d { z16.d }, p0/z, [x14, #1, mul vl]
        \\ld1d { z17.d }, p0/z, [x15, #1, mul vl]
        \\ld1d { z18.d }, p0/z, [x16, #1, mul vl]
        \\ld1d { z19.d }, p0/z, [x17, #1, mul vl]
        \\fmla z21.d, p0/m, z16.d, z28.d
        \\fmla z21.d, p0/m, z17.d, z29.d
        \\fmla z21.d, p0/m, z18.d, z30.d
        \\fmla z21.d, p0/m, z19.d, z31.d
        \\ld1d { z16.d }, p0/z, [x14, #2, mul vl]
        \\ld1d { z17.d }, p0/z, [x15, #2, mul vl]
        \\ld1d { z18.d }, p0/z, [x16, #2, mul vl]
        \\ld1d { z19.d }, p0/z, [x17, #2, mul vl]
        \\fmla z22.d, p0/m, z16.d, z28.d
        \\fmla z22.d, p0/m, z17.d, z29.d
        \\fmla z22.d, p0/m, z18.d, z30.d
        \\fmla z22.d, p0/m, z19.d, z31.d
        \\ld1d { z16.d }, p0/z, [x14, #3, mul vl]
        \\ld1d { z17.d }, p0/z, [x15, #3, mul vl]
        \\ld1d { z18.d }, p0/z, [x16, #3, mul vl]
        \\ld1d { z19.d }, p0/z, [x17, #3, mul vl]
        \\fmla z23.d, p0/m, z16.d, z28.d
        \\fmla z23.d, p0/m, z17.d, z29.d
        \\fmla z23.d, p0/m, z18.d, z30.d
        \\fmla z23.d, p0/m, z19.d, z31.d
        \\ld1d { z16.d }, p0/z, [x14, #4, mul vl]
        \\ld1d { z17.d }, p0/z, [x15, #4, mul vl]
        \\ld1d { z18.d }, p0/z, [x16, #4, mul vl]
        \\ld1d { z19.d }, p0/z, [x17, #4, mul vl]
        \\fmla z24.d, p0/m, z16.d, z28.d
        \\fmla z24.d, p0/m, z17.d, z29.d
        \\fmla z24.d, p0/m, z18.d, z30.d
        \\fmla z24.d, p0/m, z19.d, z31.d
        \\ld1d { z16.d }, p0/z, [x14, #5, mul vl]
        \\ld1d { z17.d }, p0/z, [x15, #5, mul vl]
        \\ld1d { z18.d }, p0/z, [x16, #5, mul vl]
        \\ld1d { z19.d }, p0/z, [x17, #5, mul vl]
        \\fmla z25.d, p0/m, z16.d, z28.d
        \\fmla z25.d, p0/m, z17.d, z29.d
        \\fmla z25.d, p0/m, z18.d, z30.d
        \\fmla z25.d, p0/m, z19.d, z31.d
        \\ld1d { z16.d }, p0/z, [x14, #6, mul vl]
        \\ld1d { z17.d }, p0/z, [x15, #6, mul vl]
        \\ld1d { z18.d }, p0/z, [x16, #6, mul vl]
        \\ld1d { z19.d }, p0/z, [x17, #6, mul vl]
        \\fmla z26.d, p0/m, z16.d, z28.d
        \\fmla z26.d, p0/m, z17.d, z29.d
        \\fmla z26.d, p0/m, z18.d, z30.d
        \\fmla z26.d, p0/m, z19.d, z31.d
        \\ld1d { z16.d }, p0/z, [x14, #7, mul vl]
        \\ld1d { z17.d }, p0/z, [x15, #7, mul vl]
        \\ld1d { z18.d }, p0/z, [x16, #7, mul vl]
        \\ld1d { z19.d }, p0/z, [x17, #7, mul vl]
        \\fmla z27.d, p0/m, z16.d, z28.d
        \\fmla z27.d, p0/m, z17.d, z29.d
        \\fmla z27.d, p0/m, z18.d, z30.d
        \\fmla z27.d, p0/m, z19.d, z31.d
        \\
        \\add x8, x13, x5
        \\add x9, x9, #32
        \\subs x10, x10, #4
        \\b.ne 1b
        \\
        \\fmov d28, x2
        \\fmov d29, x3
        \\mov z30.d, d29
        \\mov z31.d, d28
        \\ld1d { z16.d }, p0/z, [x7]
        \\fmul z16.d, z16.d, z30.d
        \\fmla z16.d, p0/m, z0.d, z31.d
        \\st1d { z16.d }, p0, [x7]
        \\ld1d { z16.d }, p0/z, [x7, #1, mul vl]
        \\fmul z16.d, z16.d, z30.d
        \\fmla z16.d, p0/m, z1.d, z31.d
        \\st1d { z16.d }, p0, [x7, #1, mul vl]
        \\ld1d { z16.d }, p0/z, [x7, #2, mul vl]
        \\fmul z16.d, z16.d, z30.d
        \\fmla z16.d, p0/m, z2.d, z31.d
        \\st1d { z16.d }, p0, [x7, #2, mul vl]
        \\ld1d { z16.d }, p0/z, [x7, #3, mul vl]
        \\fmul z16.d, z16.d, z30.d
        \\fmla z16.d, p0/m, z3.d, z31.d
        \\st1d { z16.d }, p0, [x7, #3, mul vl]
        \\ld1d { z16.d }, p0/z, [x7, #4, mul vl]
        \\fmul z16.d, z16.d, z30.d
        \\fmla z16.d, p0/m, z4.d, z31.d
        \\st1d { z16.d }, p0, [x7, #4, mul vl]
        \\ld1d { z16.d }, p0/z, [x7, #5, mul vl]
        \\fmul z16.d, z16.d, z30.d
        \\fmla z16.d, p0/m, z5.d, z31.d
        \\st1d { z16.d }, p0, [x7, #5, mul vl]
        \\ld1d { z16.d }, p0/z, [x7, #6, mul vl]
        \\fmul z16.d, z16.d, z30.d
        \\fmla z16.d, p0/m, z6.d, z31.d
        \\st1d { z16.d }, p0, [x7, #6, mul vl]
        \\ld1d { z16.d }, p0/z, [x7, #7, mul vl]
        \\fmul z16.d, z16.d, z30.d
        \\fmla z16.d, p0/m, z7.d, z31.d
        \\st1d { z16.d }, p0, [x7, #7, mul vl]
        \\addvl x14, x7, #8
        \\ld1d { z16.d }, p0/z, [x14]
        \\fmul z16.d, z16.d, z30.d
        \\fmla z16.d, p0/m, z20.d, z31.d
        \\st1d { z16.d }, p0, [x14]
        \\ld1d { z16.d }, p0/z, [x14, #1, mul vl]
        \\fmul z16.d, z16.d, z30.d
        \\fmla z16.d, p0/m, z21.d, z31.d
        \\st1d { z16.d }, p0, [x14, #1, mul vl]
        \\ld1d { z16.d }, p0/z, [x14, #2, mul vl]
        \\fmul z16.d, z16.d, z30.d
        \\fmla z16.d, p0/m, z22.d, z31.d
        \\st1d { z16.d }, p0, [x14, #2, mul vl]
        \\ld1d { z16.d }, p0/z, [x14, #3, mul vl]
        \\fmul z16.d, z16.d, z30.d
        \\fmla z16.d, p0/m, z23.d, z31.d
        \\st1d { z16.d }, p0, [x14, #3, mul vl]
        \\ld1d { z16.d }, p0/z, [x14, #4, mul vl]
        \\fmul z16.d, z16.d, z30.d
        \\fmla z16.d, p0/m, z24.d, z31.d
        \\st1d { z16.d }, p0, [x14, #4, mul vl]
        \\ld1d { z16.d }, p0/z, [x14, #5, mul vl]
        \\fmul z16.d, z16.d, z30.d
        \\fmla z16.d, p0/m, z25.d, z31.d
        \\st1d { z16.d }, p0, [x14, #5, mul vl]
        \\ld1d { z16.d }, p0/z, [x14, #6, mul vl]
        \\fmul z16.d, z16.d, z30.d
        \\fmla z16.d, p0/m, z26.d, z31.d
        \\st1d { z16.d }, p0, [x14, #6, mul vl]
        \\ld1d { z16.d }, p0/z, [x14, #7, mul vl]
        \\fmul z16.d, z16.d, z30.d
        \\fmla z16.d, p0/m, z27.d, z31.d
        \\st1d { z16.d }, p0, [x14, #7, mul vl]
        \\
        \\4:
        \\ret
    ::: .{ .memory = true });
}

pub noinline fn dgemvNoTransSme2F64128x1(m: usize, n: usize, alpha_bits: u64, beta_bits: u64, a: [*]const f64, lda_bytes: usize, x: [*]const f64, y: [*]f64) callconv(.naked) void {
    _ = m;
    _ = n;
    _ = alpha_bits;
    _ = beta_bits;
    _ = a;
    _ = lda_bytes;
    _ = x;
    _ = y;
    asm volatile (
        \\cbz x0, 12f
        \\cbz x1, 12f
    ++ builders.ptrue_pn8_d ++ builders.ptrue_p0_d ++
        \\
        \\zero { za }
        \\mov x15, x4
        \\mov x12, x7
        \\mov x16, x6
        \\mov x17, x1
        \\
        \\11:
        \\cmp x17, #2
        \\b.lo 13f
        \\ldr d2, [x16], #8
        \\mov z4.d, d2
        \\mov x9, x15
        \\
        \\ld1d { z16.d - z19.d }, pn8/z, [x9]
        \\ld1d { z20.d - z23.d }, pn8/z, [x9, #4, mul vl]
        \\ld1d { z24.d - z27.d }, pn8/z, [x9, #8, mul vl]
        \\ld1d { z28.d - z31.d }, pn8/z, [x9, #12, mul vl]
        \\mov w8, #0
        \\mov w11, #8
        \\fmla za.d[w8, 0, vgx4], { z16.d - z19.d }, z4.d
        \\fmla za.d[w11, 0, vgx4], { z20.d - z23.d }, z4.d
        \\mov w8, #1
        \\mov w11, #9
        \\fmla za.d[w8, 0, vgx4], { z24.d - z27.d }, z4.d
        \\fmla za.d[w11, 0, vgx4], { z28.d - z31.d }, z4.d
        \\
        \\add x15, x15, x5
        \\ldr d2, [x16], #8
        \\mov z4.d, d2
        \\mov x9, x15
        \\
        \\ld1d { z16.d - z19.d }, pn8/z, [x9]
        \\ld1d { z20.d - z23.d }, pn8/z, [x9, #4, mul vl]
        \\ld1d { z24.d - z27.d }, pn8/z, [x9, #8, mul vl]
        \\ld1d { z28.d - z31.d }, pn8/z, [x9, #12, mul vl]
        \\mov w8, #0
        \\mov w11, #8
        \\fmla za.d[w8, 0, vgx4], { z16.d - z19.d }, z4.d
        \\fmla za.d[w11, 0, vgx4], { z20.d - z23.d }, z4.d
        \\mov w8, #1
        \\mov w11, #9
        \\fmla za.d[w8, 0, vgx4], { z24.d - z27.d }, z4.d
        \\fmla za.d[w11, 0, vgx4], { z28.d - z31.d }, z4.d
        \\
        \\add x15, x15, x5
        \\subs x17, x17, #2
        \\b.ne 11b
        \\
        \\13:
        \\cbz x17, 14f
        \\ldr d2, [x16], #8
        \\mov z4.d, d2
        \\mov x9, x15
        \\
        \\ld1d { z16.d - z19.d }, pn8/z, [x9]
        \\ld1d { z20.d - z23.d }, pn8/z, [x9, #4, mul vl]
        \\ld1d { z24.d - z27.d }, pn8/z, [x9, #8, mul vl]
        \\ld1d { z28.d - z31.d }, pn8/z, [x9, #12, mul vl]
        \\mov w8, #0
        \\mov w11, #8
        \\fmla za.d[w8, 0, vgx4], { z16.d - z19.d }, z4.d
        \\fmla za.d[w11, 0, vgx4], { z20.d - z23.d }, z4.d
        \\mov w8, #1
        \\mov w11, #9
        \\fmla za.d[w8, 0, vgx4], { z24.d - z27.d }, z4.d
        \\fmla za.d[w11, 0, vgx4], { z28.d - z31.d }, z4.d
        \\
        \\14:
        \\mov x9, x12
        \\fmov d2, x3
        \\fmov d3, x2
        \\mov z4.d, d2
        \\mov z5.d, d3
        \\mov w8, #0
        \\mov w11, #8
        \\mov { z16.d - z19.d }, za.d[w8, 0, vgx4]
        \\mov { z20.d - z23.d }, za.d[w11, 0, vgx4]
        \\zero { za0.d }
        \\ld1d { z24.d - z27.d }, pn8/z, [x9]
        \\ld1d { z28.d - z31.d }, pn8/z, [x9, #4, mul vl]
        \\fmla za.d[w8, 0, vgx4], { z24.d - z27.d }, z4.d
        \\fmla za.d[w11, 0, vgx4], { z28.d - z31.d }, z4.d
        \\fmla za.d[w8, 0, vgx4], { z16.d - z19.d }, z5.d
        \\fmla za.d[w11, 0, vgx4], { z20.d - z23.d }, z5.d
        \\
        \\mov w8, #1
        \\mov w11, #9
        \\mov { z16.d - z19.d }, za.d[w8, 0, vgx4]
        \\mov { z20.d - z23.d }, za.d[w11, 0, vgx4]
        \\zero { za1.d }
        \\ld1d { z24.d - z27.d }, pn8/z, [x9, #8, mul vl]
        \\ld1d { z28.d - z31.d }, pn8/z, [x9, #12, mul vl]
        \\fmla za.d[w8, 0, vgx4], { z24.d - z27.d }, z4.d
        \\fmla za.d[w11, 0, vgx4], { z28.d - z31.d }, z4.d
        \\fmla za.d[w8, 0, vgx4], { z16.d - z19.d }, z5.d
        \\fmla za.d[w11, 0, vgx4], { z20.d - z23.d }, z5.d
        \\
        \\mov w8, #0
        \\mov w11, #8
        \\mov { z16.d - z19.d }, za.d[w8, 0, vgx4]
        \\mov { z20.d - z23.d }, za.d[w11, 0, vgx4]
        \\st1d { z16.d - z19.d }, pn8, [x9]
        \\st1d { z20.d - z23.d }, pn8, [x9, #4, mul vl]
        \\mov w8, #1
        \\mov w11, #9
        \\mov { z16.d - z19.d }, za.d[w8, 0, vgx4]
        \\mov { z20.d - z23.d }, za.d[w11, 0, vgx4]
        \\st1d { z16.d - z19.d }, pn8, [x9, #8, mul vl]
        \\st1d { z20.d - z23.d }, pn8, [x9, #12, mul vl]
        \\
        \\12:
        \\ret
    ::: .{ .memory = true });
}

pub noinline fn dgemvNoTransSme2F64256x1(m: usize, n: usize, alpha_bits: u64, beta_bits: u64, a: [*]const f64, lda_bytes: usize, x: [*]const f64, y: [*]f64) callconv(.naked) void {
    _ = m;
    _ = n;
    _ = alpha_bits;
    _ = beta_bits;
    _ = a;
    _ = lda_bytes;
    _ = x;
    _ = y;
    asm volatile (
        \\cbz x0, 12f
        \\cbz x1, 12f
    ++ builders.ptrue_pn8_d ++ builders.ptrue_p0_d ++
        \\
        \\mov x13, x4
        \\mov x14, x7
        \\mov x10, x0
        \\
        \\10:
        \\zero { za }
        \\mov x15, x13
        \\mov x16, x6
        \\mov x17, x1
        \\
        \\11:
        \\ldr d2, [x16], #8
        \\mov z4.d, d2
        \\mov x9, x15
        \\
        \\ld1d { z16.d - z19.d }, pn8/z, [x9]
        \\ld1d { z20.d - z23.d }, pn8/z, [x9, #4, mul vl]
        \\ld1d { z24.d - z27.d }, pn8/z, [x9, #8, mul vl]
        \\ld1d { z28.d - z31.d }, pn8/z, [x9, #12, mul vl]
        \\mov w8, #0
        \\mov w11, #8
        \\fmla za.d[w8, 0, vgx4], { z16.d - z19.d }, z4.d
        \\fmla za.d[w11, 0, vgx4], { z20.d - z23.d }, z4.d
        \\mov w8, #1
        \\mov w11, #9
        \\fmla za.d[w8, 0, vgx4], { z24.d - z27.d }, z4.d
        \\fmla za.d[w11, 0, vgx4], { z28.d - z31.d }, z4.d
        \\
        \\add x9, x9, #1024
        \\ld1d { z16.d - z19.d }, pn8/z, [x9]
        \\ld1d { z20.d - z23.d }, pn8/z, [x9, #4, mul vl]
        \\ld1d { z24.d - z27.d }, pn8/z, [x9, #8, mul vl]
        \\ld1d { z28.d - z31.d }, pn8/z, [x9, #12, mul vl]
        \\mov w8, #2
        \\mov w11, #10
        \\fmla za.d[w8, 0, vgx4], { z16.d - z19.d }, z4.d
        \\fmla za.d[w11, 0, vgx4], { z20.d - z23.d }, z4.d
        \\mov w8, #3
        \\mov w11, #11
        \\fmla za.d[w8, 0, vgx4], { z24.d - z27.d }, z4.d
        \\fmla za.d[w11, 0, vgx4], { z28.d - z31.d }, z4.d
        \\
        \\add x15, x15, x5
        \\subs x17, x17, #1
        \\b.ne 11b
        \\
        \\mov x9, x14
        \\fmov d2, x3
        \\fmov d3, x2
        \\mov z4.d, d2
        \\mov z5.d, d3
        \\
        \\mov w8, #0
        \\mov w11, #8
        \\mov { z16.d - z19.d }, za.d[w8, 0, vgx4]
        \\mov { z20.d - z23.d }, za.d[w11, 0, vgx4]
        \\ld1d { z24.d - z27.d }, pn8/z, [x9]
        \\ld1d { z28.d - z31.d }, pn8/z, [x9, #4, mul vl]
        \\fmul z24.d, z24.d, z4.d
        \\fmul z25.d, z25.d, z4.d
        \\fmul z26.d, z26.d, z4.d
        \\fmul z27.d, z27.d, z4.d
        \\fmul z28.d, z28.d, z4.d
        \\fmul z29.d, z29.d, z4.d
        \\fmul z30.d, z30.d, z4.d
        \\fmul z31.d, z31.d, z4.d
        \\fmla z24.d, p0/m, z16.d, z5.d
        \\fmla z25.d, p0/m, z17.d, z5.d
        \\fmla z26.d, p0/m, z18.d, z5.d
        \\fmla z27.d, p0/m, z19.d, z5.d
        \\fmla z28.d, p0/m, z20.d, z5.d
        \\fmla z29.d, p0/m, z21.d, z5.d
        \\fmla z30.d, p0/m, z22.d, z5.d
        \\fmla z31.d, p0/m, z23.d, z5.d
        \\st1d { z24.d - z27.d }, pn8, [x9]
        \\st1d { z28.d - z31.d }, pn8, [x9, #4, mul vl]
        \\
        \\mov w8, #1
        \\mov w11, #9
        \\mov { z16.d - z19.d }, za.d[w8, 0, vgx4]
        \\mov { z20.d - z23.d }, za.d[w11, 0, vgx4]
        \\ld1d { z24.d - z27.d }, pn8/z, [x9, #8, mul vl]
        \\ld1d { z28.d - z31.d }, pn8/z, [x9, #12, mul vl]
        \\fmul z24.d, z24.d, z4.d
        \\fmul z25.d, z25.d, z4.d
        \\fmul z26.d, z26.d, z4.d
        \\fmul z27.d, z27.d, z4.d
        \\fmul z28.d, z28.d, z4.d
        \\fmul z29.d, z29.d, z4.d
        \\fmul z30.d, z30.d, z4.d
        \\fmul z31.d, z31.d, z4.d
        \\fmla z24.d, p0/m, z16.d, z5.d
        \\fmla z25.d, p0/m, z17.d, z5.d
        \\fmla z26.d, p0/m, z18.d, z5.d
        \\fmla z27.d, p0/m, z19.d, z5.d
        \\fmla z28.d, p0/m, z20.d, z5.d
        \\fmla z29.d, p0/m, z21.d, z5.d
        \\fmla z30.d, p0/m, z22.d, z5.d
        \\fmla z31.d, p0/m, z23.d, z5.d
        \\st1d { z24.d - z27.d }, pn8, [x9, #8, mul vl]
        \\st1d { z28.d - z31.d }, pn8, [x9, #12, mul vl]
        \\
        \\mov w8, #2
        \\mov w11, #10
        \\mov { z16.d - z19.d }, za.d[w8, 0, vgx4]
        \\mov { z20.d - z23.d }, za.d[w11, 0, vgx4]
        \\ld1d { z24.d - z27.d }, pn8/z, [x9, #16, mul vl]
        \\ld1d { z28.d - z31.d }, pn8/z, [x9, #20, mul vl]
        \\fmul z24.d, z24.d, z4.d
        \\fmul z25.d, z25.d, z4.d
        \\fmul z26.d, z26.d, z4.d
        \\fmul z27.d, z27.d, z4.d
        \\fmul z28.d, z28.d, z4.d
        \\fmul z29.d, z29.d, z4.d
        \\fmul z30.d, z30.d, z4.d
        \\fmul z31.d, z31.d, z4.d
        \\fmla z24.d, p0/m, z16.d, z5.d
        \\fmla z25.d, p0/m, z17.d, z5.d
        \\fmla z26.d, p0/m, z18.d, z5.d
        \\fmla z27.d, p0/m, z19.d, z5.d
        \\fmla z28.d, p0/m, z20.d, z5.d
        \\fmla z29.d, p0/m, z21.d, z5.d
        \\fmla z30.d, p0/m, z22.d, z5.d
        \\fmla z31.d, p0/m, z23.d, z5.d
        \\st1d { z24.d - z27.d }, pn8, [x9, #16, mul vl]
        \\st1d { z28.d - z31.d }, pn8, [x9, #20, mul vl]
        \\
        \\mov w8, #3
        \\mov w11, #11
        \\mov { z16.d - z19.d }, za.d[w8, 0, vgx4]
        \\mov { z20.d - z23.d }, za.d[w11, 0, vgx4]
        \\ld1d { z24.d - z27.d }, pn8/z, [x9, #24, mul vl]
        \\ld1d { z28.d - z31.d }, pn8/z, [x9, #28, mul vl]
        \\fmul z24.d, z24.d, z4.d
        \\fmul z25.d, z25.d, z4.d
        \\fmul z26.d, z26.d, z4.d
        \\fmul z27.d, z27.d, z4.d
        \\fmul z28.d, z28.d, z4.d
        \\fmul z29.d, z29.d, z4.d
        \\fmul z30.d, z30.d, z4.d
        \\fmul z31.d, z31.d, z4.d
        \\fmla z24.d, p0/m, z16.d, z5.d
        \\fmla z25.d, p0/m, z17.d, z5.d
        \\fmla z26.d, p0/m, z18.d, z5.d
        \\fmla z27.d, p0/m, z19.d, z5.d
        \\fmla z28.d, p0/m, z20.d, z5.d
        \\fmla z29.d, p0/m, z21.d, z5.d
        \\fmla z30.d, p0/m, z22.d, z5.d
        \\fmla z31.d, p0/m, z23.d, z5.d
        \\st1d { z24.d - z27.d }, pn8, [x9, #24, mul vl]
        \\st1d { z28.d - z31.d }, pn8, [x9, #28, mul vl]
        \\
        \\add x13, x13, #2048
        \\add x14, x14, #2048
        \\subs x10, x10, #256
        \\b.ne 10b
        \\
        \\12:
        \\ret
    ::: .{ .memory = true });
}

pub noinline fn sgemvNoTransSme2F32512x1(m: usize, n: usize, alpha_bits: u32, beta_bits: u32, a: [*]const f32, lda_bytes: usize, x: [*]const f32, y: [*]f32) callconv(.naked) void {
    _ = m;
    _ = n;
    _ = alpha_bits;
    _ = beta_bits;
    _ = a;
    _ = lda_bytes;
    _ = x;
    _ = y;
    asm volatile (
        \\cbz x0, 12f
        \\cbz x1, 12f
    ++ builders.ptrue_pn8_s ++ builders.ptrue_p0_s ++
        \\
        \\zero { za }
        \\mov x15, x4
        \\mov x16, x6
        \\mov x17, x1
        \\
        \\11:
        \\ldr s2, [x16], #4
        \\mov z4.s, s2
        \\mov x9, x15
        \\ld1w { z16.s - z19.s }, pn8/z, [x9]
        \\ld1w { z20.s - z23.s }, pn8/z, [x9, #4, mul vl]
        \\ld1w { z24.s - z27.s }, pn8/z, [x9, #8, mul vl]
        \\ld1w { z28.s - z31.s }, pn8/z, [x9, #12, mul vl]
        \\mov w8, #0
        \\mov w11, #8
        \\fmla za.s[w8, 0, vgx4], { z16.s - z19.s }, z4.s
        \\fmla za.s[w11, 0, vgx4], { z20.s - z23.s }, z4.s
        \\mov w8, #1
        \\mov w11, #9
        \\fmla za.s[w8, 0, vgx4], { z24.s - z27.s }, z4.s
        \\fmla za.s[w11, 0, vgx4], { z28.s - z31.s }, z4.s
        \\
        \\add x9, x15, #1024
        \\ld1w { z16.s - z19.s }, pn8/z, [x9]
        \\ld1w { z20.s - z23.s }, pn8/z, [x9, #4, mul vl]
        \\ld1w { z24.s - z27.s }, pn8/z, [x9, #8, mul vl]
        \\ld1w { z28.s - z31.s }, pn8/z, [x9, #12, mul vl]
        \\mov w8, #2
        \\mov w11, #10
        \\fmla za.s[w8, 0, vgx4], { z16.s - z19.s }, z4.s
        \\fmla za.s[w11, 0, vgx4], { z20.s - z23.s }, z4.s
        \\mov w8, #3
        \\mov w11, #11
        \\fmla za.s[w8, 0, vgx4], { z24.s - z27.s }, z4.s
        \\fmla za.s[w11, 0, vgx4], { z28.s - z31.s }, z4.s
        \\add x15, x15, x5
        \\subs x17, x17, #1
        \\b.ne 11b
        \\
        \\mov x9, x7
        \\fmov s2, w3
        \\fmov s3, w2
        \\mov z4.s, s2
        \\mov z5.s, s3
        \\mov w8, #0
        \\mov { z16.s - z19.s }, za.s[w8, 0, vgx4]
        \\ld1w { z20.s - z23.s }, pn8/z, [x9]
        \\fmul z20.s, p0/m, z20.s, z4.s
        \\fmul z21.s, p0/m, z21.s, z4.s
        \\fmul z22.s, p0/m, z22.s, z4.s
        \\fmul z23.s, p0/m, z23.s, z4.s
        \\fmla z20.s, p0/m, z16.s, z5.s
        \\fmla z21.s, p0/m, z17.s, z5.s
        \\fmla z22.s, p0/m, z18.s, z5.s
        \\fmla z23.s, p0/m, z19.s, z5.s
        \\st1w { z20.s - z23.s }, pn8, [x9]
        \\
        \\mov w8, #8
        \\mov { z16.s - z19.s }, za.s[w8, 0, vgx4]
        \\ld1w { z20.s - z23.s }, pn8/z, [x9, #4, mul vl]
        \\fmul z20.s, p0/m, z20.s, z4.s
        \\fmul z21.s, p0/m, z21.s, z4.s
        \\fmul z22.s, p0/m, z22.s, z4.s
        \\fmul z23.s, p0/m, z23.s, z4.s
        \\fmla z20.s, p0/m, z16.s, z5.s
        \\fmla z21.s, p0/m, z17.s, z5.s
        \\fmla z22.s, p0/m, z18.s, z5.s
        \\fmla z23.s, p0/m, z19.s, z5.s
        \\st1w { z20.s - z23.s }, pn8, [x9, #4, mul vl]
        \\
        \\mov w8, #1
        \\mov { z16.s - z19.s }, za.s[w8, 0, vgx4]
        \\ld1w { z20.s - z23.s }, pn8/z, [x9, #8, mul vl]
        \\fmul z20.s, p0/m, z20.s, z4.s
        \\fmul z21.s, p0/m, z21.s, z4.s
        \\fmul z22.s, p0/m, z22.s, z4.s
        \\fmul z23.s, p0/m, z23.s, z4.s
        \\fmla z20.s, p0/m, z16.s, z5.s
        \\fmla z21.s, p0/m, z17.s, z5.s
        \\fmla z22.s, p0/m, z18.s, z5.s
        \\fmla z23.s, p0/m, z19.s, z5.s
        \\st1w { z20.s - z23.s }, pn8, [x9, #8, mul vl]
        \\
        \\mov w8, #9
        \\mov { z16.s - z19.s }, za.s[w8, 0, vgx4]
        \\ld1w { z20.s - z23.s }, pn8/z, [x9, #12, mul vl]
        \\fmul z20.s, p0/m, z20.s, z4.s
        \\fmul z21.s, p0/m, z21.s, z4.s
        \\fmul z22.s, p0/m, z22.s, z4.s
        \\fmul z23.s, p0/m, z23.s, z4.s
        \\fmla z20.s, p0/m, z16.s, z5.s
        \\fmla z21.s, p0/m, z17.s, z5.s
        \\fmla z22.s, p0/m, z18.s, z5.s
        \\fmla z23.s, p0/m, z19.s, z5.s
        \\st1w { z20.s - z23.s }, pn8, [x9, #12, mul vl]
        \\
        \\add x9, x7, #1024
        \\mov w8, #2
        \\mov { z16.s - z19.s }, za.s[w8, 0, vgx4]
        \\ld1w { z20.s - z23.s }, pn8/z, [x9]
        \\fmul z20.s, p0/m, z20.s, z4.s
        \\fmul z21.s, p0/m, z21.s, z4.s
        \\fmul z22.s, p0/m, z22.s, z4.s
        \\fmul z23.s, p0/m, z23.s, z4.s
        \\fmla z20.s, p0/m, z16.s, z5.s
        \\fmla z21.s, p0/m, z17.s, z5.s
        \\fmla z22.s, p0/m, z18.s, z5.s
        \\fmla z23.s, p0/m, z19.s, z5.s
        \\st1w { z20.s - z23.s }, pn8, [x9]
        \\
        \\mov w8, #10
        \\mov { z16.s - z19.s }, za.s[w8, 0, vgx4]
        \\ld1w { z20.s - z23.s }, pn8/z, [x9, #4, mul vl]
        \\fmul z20.s, p0/m, z20.s, z4.s
        \\fmul z21.s, p0/m, z21.s, z4.s
        \\fmul z22.s, p0/m, z22.s, z4.s
        \\fmul z23.s, p0/m, z23.s, z4.s
        \\fmla z20.s, p0/m, z16.s, z5.s
        \\fmla z21.s, p0/m, z17.s, z5.s
        \\fmla z22.s, p0/m, z18.s, z5.s
        \\fmla z23.s, p0/m, z19.s, z5.s
        \\st1w { z20.s - z23.s }, pn8, [x9, #4, mul vl]
        \\
        \\mov w8, #3
        \\mov { z16.s - z19.s }, za.s[w8, 0, vgx4]
        \\ld1w { z20.s - z23.s }, pn8/z, [x9, #8, mul vl]
        \\fmul z20.s, p0/m, z20.s, z4.s
        \\fmul z21.s, p0/m, z21.s, z4.s
        \\fmul z22.s, p0/m, z22.s, z4.s
        \\fmul z23.s, p0/m, z23.s, z4.s
        \\fmla z20.s, p0/m, z16.s, z5.s
        \\fmla z21.s, p0/m, z17.s, z5.s
        \\fmla z22.s, p0/m, z18.s, z5.s
        \\fmla z23.s, p0/m, z19.s, z5.s
        \\st1w { z20.s - z23.s }, pn8, [x9, #8, mul vl]
        \\
        \\mov w8, #11
        \\mov { z16.s - z19.s }, za.s[w8, 0, vgx4]
        \\ld1w { z20.s - z23.s }, pn8/z, [x9, #12, mul vl]
        \\fmul z20.s, p0/m, z20.s, z4.s
        \\fmul z21.s, p0/m, z21.s, z4.s
        \\fmul z22.s, p0/m, z22.s, z4.s
        \\fmul z23.s, p0/m, z23.s, z4.s
        \\fmla z20.s, p0/m, z16.s, z5.s
        \\fmla z21.s, p0/m, z17.s, z5.s
        \\fmla z22.s, p0/m, z18.s, z5.s
        \\fmla z23.s, p0/m, z19.s, z5.s
        \\st1w { z20.s - z23.s }, pn8, [x9, #12, mul vl]
        \\
        \\12:
        \\ret
    ::: .{ .memory = true });
}

pub noinline fn sgemvNoTransSme2F32256x1(m: usize, n: usize, alpha_bits: u32, beta_bits: u32, a: [*]const f32, lda_bytes: usize, x: [*]const f32, y: [*]f32) callconv(.naked) void {
    _ = m;
    _ = n;
    _ = alpha_bits;
    _ = beta_bits;
    _ = a;
    _ = lda_bytes;
    _ = x;
    _ = y;
    asm volatile (
        \\cbz x0, 12f
        \\cbz x1, 12f
    ++ builders.ptrue_pn8_s ++ builders.ptrue_p0_s ++
        \\
        \\mov x13, x4
        \\mov x14, x7
        \\mov x10, x0
        \\
        \\10:
        \\zero { za }
        \\mov x15, x13
        \\mov x16, x6
        \\mov x17, x1
        \\
        \\11:
        \\ldr s2, [x16], #4
        \\mov z4.s, s2
        \\mov x9, x15
        \\
        \\ld1w { z16.s - z19.s }, pn8/z, [x9]
        \\ld1w { z20.s - z23.s }, pn8/z, [x9, #4, mul vl]
        \\ld1w { z24.s - z27.s }, pn8/z, [x9, #8, mul vl]
        \\ld1w { z28.s - z31.s }, pn8/z, [x9, #12, mul vl]
        \\mov w8, #0
        \\mov w11, #8
        \\fmla za.s[w8, 0, vgx4], { z16.s - z19.s }, z4.s
        \\fmla za.s[w11, 0, vgx4], { z20.s - z23.s }, z4.s
        \\mov w8, #1
        \\mov w11, #9
        \\fmla za.s[w8, 0, vgx4], { z24.s - z27.s }, z4.s
        \\fmla za.s[w11, 0, vgx4], { z28.s - z31.s }, z4.s
        \\
        \\add x15, x15, x5
        \\subs x17, x17, #1
        \\b.ne 11b
        \\
        \\mov x9, x14
        \\fmov s2, w3
        \\fmov s3, w2
        \\mov z4.s, s2
        \\mov z5.s, s3
        \\
        \\mov w8, #0
        \\mov { z16.s - z19.s }, za.s[w8, 0, vgx4]
        \\ld1w { z20.s - z23.s }, pn8/z, [x9]
        \\fmul z20.s, p0/m, z20.s, z4.s
        \\fmul z21.s, p0/m, z21.s, z4.s
        \\fmul z22.s, p0/m, z22.s, z4.s
        \\fmul z23.s, p0/m, z23.s, z4.s
        \\fmla z20.s, p0/m, z16.s, z5.s
        \\fmla z21.s, p0/m, z17.s, z5.s
        \\fmla z22.s, p0/m, z18.s, z5.s
        \\fmla z23.s, p0/m, z19.s, z5.s
        \\st1w { z20.s - z23.s }, pn8, [x9]
        \\
        \\mov w8, #8
        \\mov { z16.s - z19.s }, za.s[w8, 0, vgx4]
        \\ld1w { z20.s - z23.s }, pn8/z, [x9, #4, mul vl]
        \\fmul z20.s, p0/m, z20.s, z4.s
        \\fmul z21.s, p0/m, z21.s, z4.s
        \\fmul z22.s, p0/m, z22.s, z4.s
        \\fmul z23.s, p0/m, z23.s, z4.s
        \\fmla z20.s, p0/m, z16.s, z5.s
        \\fmla z21.s, p0/m, z17.s, z5.s
        \\fmla z22.s, p0/m, z18.s, z5.s
        \\fmla z23.s, p0/m, z19.s, z5.s
        \\st1w { z20.s - z23.s }, pn8, [x9, #4, mul vl]
        \\
        \\mov w8, #1
        \\mov { z16.s - z19.s }, za.s[w8, 0, vgx4]
        \\ld1w { z20.s - z23.s }, pn8/z, [x9, #8, mul vl]
        \\fmul z20.s, p0/m, z20.s, z4.s
        \\fmul z21.s, p0/m, z21.s, z4.s
        \\fmul z22.s, p0/m, z22.s, z4.s
        \\fmul z23.s, p0/m, z23.s, z4.s
        \\fmla z20.s, p0/m, z16.s, z5.s
        \\fmla z21.s, p0/m, z17.s, z5.s
        \\fmla z22.s, p0/m, z18.s, z5.s
        \\fmla z23.s, p0/m, z19.s, z5.s
        \\st1w { z20.s - z23.s }, pn8, [x9, #8, mul vl]
        \\
        \\mov w8, #9
        \\mov { z16.s - z19.s }, za.s[w8, 0, vgx4]
        \\ld1w { z20.s - z23.s }, pn8/z, [x9, #12, mul vl]
        \\fmul z20.s, p0/m, z20.s, z4.s
        \\fmul z21.s, p0/m, z21.s, z4.s
        \\fmul z22.s, p0/m, z22.s, z4.s
        \\fmul z23.s, p0/m, z23.s, z4.s
        \\fmla z20.s, p0/m, z16.s, z5.s
        \\fmla z21.s, p0/m, z17.s, z5.s
        \\fmla z22.s, p0/m, z18.s, z5.s
        \\fmla z23.s, p0/m, z19.s, z5.s
        \\st1w { z20.s - z23.s }, pn8, [x9, #12, mul vl]
        \\
        \\add x13, x13, #1024
        \\add x14, x14, #1024
        \\subs x10, x10, #256
        \\b.ne 10b
        \\
        \\12:
        \\ret
    ::: .{ .memory = true });
}

pub noinline fn dgemvTransSme2F648x32(m: usize, n: usize, alpha_bits: u64, beta_bits: u64, a: [*]const f64, lda_bytes: usize, x: [*]const f64, y: [*]f64) callconv(.naked) void {
    _ = m;
    _ = n;
    _ = alpha_bits;
    _ = beta_bits;
    _ = a;
    _ = lda_bytes;
    _ = x;
    _ = y;
    asm volatile (
        \\cbz x0, 13f
        \\cbz x1, 13f
    ++ builders.ptrue_pn8_d ++
        \\
        \\mov x13, x4
        \\mov x15, x1
        \\lsl x14, x5, #3
        \\
        \\10:
        \\zero { za }
        \\mov x12, #0
        \\mov x10, x6
        \\mov x16, x13
        \\
        \\11:
        \\cmp x12, x0
        \\b.hs 12f
        \\
        \\mov x9, x16
        \\ld1d { z4.d - z7.d }, pn8/z, [x10]
        \\mov w8, #0
        \\mov w11, #8
        \\
        \\ld1d { z16.d, z20.d, z24.d, z28.d }, pn8/z, [x9]
        \\add x9, x9, x5
        \\ld1d { z17.d, z21.d, z25.d, z29.d }, pn8/z, [x9]
        \\add x9, x9, x5
        \\ld1d { z18.d, z22.d, z26.d, z30.d }, pn8/z, [x9]
        \\add x9, x9, x5
        \\ld1d { z19.d, z23.d, z27.d, z31.d }, pn8/z, [x9]
        \\add x9, x9, x5
        \\fmla za.d[w8, 0, vgx4], { z16.d - z19.d }, z4.d
        \\fmla za.d[w11, 0, vgx4], { z20.d - z23.d }, z5.d
        \\fmla za.d[w8, 1, vgx4], { z24.d - z27.d }, z6.d
        \\fmla za.d[w11, 1, vgx4], { z28.d - z31.d }, z7.d
        \\
        \\ld1d { z16.d, z20.d, z24.d, z28.d }, pn8/z, [x9]
        \\add x9, x9, x5
        \\ld1d { z17.d, z21.d, z25.d, z29.d }, pn8/z, [x9]
        \\add x9, x9, x5
        \\ld1d { z18.d, z22.d, z26.d, z30.d }, pn8/z, [x9]
        \\add x9, x9, x5
        \\ld1d { z19.d, z23.d, z27.d, z31.d }, pn8/z, [x9]
        \\fmla za.d[w8, 2, vgx4], { z16.d - z19.d }, z4.d
        \\fmla za.d[w11, 2, vgx4], { z20.d - z23.d }, z5.d
        \\fmla za.d[w8, 3, vgx4], { z24.d - z27.d }, z6.d
        \\fmla za.d[w11, 3, vgx4], { z28.d - z31.d }, z7.d
        \\
        \\add x12, x12, #32
        \\add x10, x10, #256
        \\add x16, x16, #256
        \\b 11b
        \\
        \\12:
        \\mov w8, #0
        \\mov w11, #8
        \\mov { z16.d - z19.d }, za.d[w8, 1, vgx4]
        \\mov { z20.d - z23.d }, za.d[w11, 1, vgx4]
        \\fadd za.d[w8, 0, vgx4], { z16.d - z19.d }
        \\fadd za.d[w11, 0, vgx4], { z20.d - z23.d }
        \\mov { z24.d - z27.d }, za.d[w8, 3, vgx4]
        \\mov { z28.d - z31.d }, za.d[w11, 3, vgx4]
        \\fadd za.d[w8, 2, vgx4], { z24.d - z27.d }
        \\fadd za.d[w11, 2, vgx4], { z28.d - z31.d }
        \\mov { z20.d - z23.d }, za.d[w11, 0, vgx4]
        \\fadd za.d[w8, 0, vgx4], { z20.d - z23.d }
        \\mov { z16.d - z19.d }, za.d[w11, 2, vgx4]
        \\fadd za.d[w8, 2, vgx4], { z16.d - z19.d }
        \\mov { z16.d - z19.d }, za.d[w8, 2, vgx4]
        \\mov za.d[w11, 0, vgx4], { z16.d - z19.d }
        \\mov w12, #0
        \\mov { z16.d - z19.d }, za0v.d[w12, 0:3]
        \\mov w12, #4
        \\mov { z20.d - z23.d }, za0v.d[w12, 0:3]
        \\zero { za1.d }
        \\fadd za.d[w8, 1, vgx4], { z16.d - z19.d }
        \\fadd za.d[w8, 1, vgx4], { z20.d - z23.d }
        \\mov { z20.d - z23.d }, za.d[w8, 1, vgx4]
        \\fadd z20.d, z20.d, z21.d
        \\fadd z20.d, z20.d, z22.d
        \\fadd z20.d, z20.d, z23.d
        \\adr x12, 14f
        \\ldr z4, [x12]
        \\tbl z20.d, { z20.d }, z4.d
        \\
        \\mov x12, #8
        \\whilelo p0.d, xzr, x12
        \\fmov d2, x3
        \\fmov d3, x2
        \\ld1d { z16.d }, p0/z, [x7]
        \\mov z17.d, d2
        \\fmul z16.d, z16.d, z17.d
        \\mov z17.d, d3
        \\fmla z16.d, p0/m, z20.d, z17.d
        \\st1d { z16.d }, p0, [x7]
        \\
        \\add x7, x7, #64
        \\add x13, x13, x14
        \\subs x15, x15, #8
        \\b.ne 10b
        \\
        \\13:
        \\ret
        \\
        \\14:
        \\.quad 0, 2, 4, 6, 1, 3, 5, 7
    ::: .{ .memory = true });
}

pub noinline fn sgemvTransSme2F3216x64(m: usize, n: usize, alpha_bits: u32, beta_bits: u32, a: [*]const f32, lda_bytes: usize, x: [*]const f32, y: [*]f32) callconv(.naked) void {
    _ = m;
    _ = n;
    _ = alpha_bits;
    _ = beta_bits;
    _ = a;
    _ = lda_bytes;
    _ = x;
    _ = y;
    asm volatile (
        \\cbz x0, 13f
        \\cbz x1, 13f
    ++ builders.ptrue_pn8_s ++ builders.ptrue_p0_s ++
        \\
        \\mov x13, x4
        \\mov x15, x1
        \\lsl x14, x5, #4
        \\
        \\10:
        \\zero { za }
        \\mov x12, #0
        \\mov x10, x6
        \\mov x16, x13
        \\
        \\11:
        \\cmp x12, x0
        \\b.hs 12f
        \\
        \\mov x9, x16
        \\ld1w { z4.s - z7.s }, pn8/z, [x10]
        \\mov w8, #0
        \\mov w11, #8
        \\
        \\ld1w { z16.s, z20.s, z24.s, z28.s }, pn8/z, [x9]
        \\add x9, x9, x5
        \\ld1w { z17.s, z21.s, z25.s, z29.s }, pn8/z, [x9]
        \\add x9, x9, x5
        \\ld1w { z18.s, z22.s, z26.s, z30.s }, pn8/z, [x9]
        \\add x9, x9, x5
        \\ld1w { z19.s, z23.s, z27.s, z31.s }, pn8/z, [x9]
        \\add x9, x9, x5
        \\fmla za.s[w8, 0, vgx4], { z16.s - z19.s }, z4.s
        \\fmla za.s[w11, 0, vgx4], { z20.s - z23.s }, z5.s
        \\fmla za.s[w8, 1, vgx4], { z24.s - z27.s }, z6.s
        \\fmla za.s[w11, 1, vgx4], { z28.s - z31.s }, z7.s
        \\
        \\ld1w { z16.s, z20.s, z24.s, z28.s }, pn8/z, [x9]
        \\add x9, x9, x5
        \\ld1w { z17.s, z21.s, z25.s, z29.s }, pn8/z, [x9]
        \\add x9, x9, x5
        \\ld1w { z18.s, z22.s, z26.s, z30.s }, pn8/z, [x9]
        \\add x9, x9, x5
        \\ld1w { z19.s, z23.s, z27.s, z31.s }, pn8/z, [x9]
        \\add x9, x9, x5
        \\fmla za.s[w8, 2, vgx4], { z16.s - z19.s }, z4.s
        \\fmla za.s[w11, 2, vgx4], { z20.s - z23.s }, z5.s
        \\fmla za.s[w8, 3, vgx4], { z24.s - z27.s }, z6.s
        \\fmla za.s[w11, 3, vgx4], { z28.s - z31.s }, z7.s
        \\
        \\ld1w { z16.s, z20.s, z24.s, z28.s }, pn8/z, [x9]
        \\add x9, x9, x5
        \\ld1w { z17.s, z21.s, z25.s, z29.s }, pn8/z, [x9]
        \\add x9, x9, x5
        \\ld1w { z18.s, z22.s, z26.s, z30.s }, pn8/z, [x9]
        \\add x9, x9, x5
        \\ld1w { z19.s, z23.s, z27.s, z31.s }, pn8/z, [x9]
        \\add x9, x9, x5
        \\fmla za.s[w8, 4, vgx4], { z16.s - z19.s }, z4.s
        \\fmla za.s[w11, 4, vgx4], { z20.s - z23.s }, z5.s
        \\fmla za.s[w8, 5, vgx4], { z24.s - z27.s }, z6.s
        \\fmla za.s[w11, 5, vgx4], { z28.s - z31.s }, z7.s
        \\
        \\ld1w { z16.s, z20.s, z24.s, z28.s }, pn8/z, [x9]
        \\add x9, x9, x5
        \\ld1w { z17.s, z21.s, z25.s, z29.s }, pn8/z, [x9]
        \\add x9, x9, x5
        \\ld1w { z18.s, z22.s, z26.s, z30.s }, pn8/z, [x9]
        \\add x9, x9, x5
        \\ld1w { z19.s, z23.s, z27.s, z31.s }, pn8/z, [x9]
        \\fmla za.s[w8, 6, vgx4], { z16.s - z19.s }, z4.s
        \\fmla za.s[w11, 6, vgx4], { z20.s - z23.s }, z5.s
        \\fmla za.s[w8, 7, vgx4], { z24.s - z27.s }, z6.s
        \\fmla za.s[w11, 7, vgx4], { z28.s - z31.s }, z7.s
        \\
        \\add x12, x12, #64
        \\add x10, x10, #256
        \\add x16, x16, #256
        \\b 11b
        \\
        \\12:
        \\mov w8, #0
        \\mov w11, #8
        \\mov { z16.d - z19.d }, za.d[w8, 1, vgx4]
        \\mov { z20.d - z23.d }, za.d[w11, 1, vgx4]
        \\fadd za.s[w8, 0, vgx4], { z16.s - z19.s }
        \\fadd za.s[w11, 0, vgx4], { z20.s - z23.s }
        \\mov { z24.d - z27.d }, za.d[w8, 3, vgx4]
        \\mov { z28.d - z31.d }, za.d[w11, 3, vgx4]
        \\fadd za.s[w8, 2, vgx4], { z24.s - z27.s }
        \\fadd za.s[w11, 2, vgx4], { z28.s - z31.s }
        \\mov { z16.d - z19.d }, za.d[w8, 5, vgx4]
        \\mov { z20.d - z23.d }, za.d[w11, 5, vgx4]
        \\fadd za.s[w8, 4, vgx4], { z16.s - z19.s }
        \\fadd za.s[w11, 4, vgx4], { z20.s - z23.s }
        \\mov { z24.d - z27.d }, za.d[w8, 7, vgx4]
        \\mov { z28.d - z31.d }, za.d[w11, 7, vgx4]
        \\fadd za.s[w8, 6, vgx4], { z24.s - z27.s }
        \\fadd za.s[w11, 6, vgx4], { z28.s - z31.s }
        \\mov { z20.d - z23.d }, za.d[w11, 0, vgx4]
        \\fadd za.s[w8, 0, vgx4], { z20.s - z23.s }
        \\mov { z16.d - z19.d }, za.d[w11, 2, vgx4]
        \\fadd za.s[w8, 2, vgx4], { z16.s - z19.s }
        \\mov { z24.d - z27.d }, za.d[w11, 4, vgx4]
        \\fadd za.s[w8, 4, vgx4], { z24.s - z27.s }
        \\mov { z28.d - z31.d }, za.d[w11, 6, vgx4]
        \\fadd za.s[w8, 6, vgx4], { z28.s - z31.s }
        \\mov { z16.d - z19.d }, za.d[w8, 2, vgx4]
        \\mov { z20.d - z23.d }, za.d[w8, 6, vgx4]
        \\mov za.d[w11, 0, vgx4], { z16.d - z19.d }
        \\mov za.d[w11, 4, vgx4], { z20.d - z23.d }
        \\mov w12, #0
        \\mov { z16.s - z19.s }, za0v.s[w12, 0:3]
        \\mov w12, #4
        \\mov { z20.s - z23.s }, za0v.s[w12, 0:3]
        \\mov w12, #8
        \\mov { z24.s - z27.s }, za0v.s[w12, 0:3]
        \\mov w12, #12
        \\mov { z28.s - z31.s }, za0v.s[w12, 0:3]
        \\zero { za1.s }
        \\fadd za.s[w8, 1, vgx4], { z16.s - z19.s }
        \\fadd za.s[w11, 1, vgx4], { z20.s - z23.s }
        \\fadd za.s[w8, 1, vgx4], { z24.s - z27.s }
        \\fadd za.s[w11, 1, vgx4], { z28.s - z31.s }
        \\mov { z16.d - z19.d }, za.d[w11, 1, vgx4]
        \\fadd za.s[w8, 1, vgx4], { z16.s - z19.s }
        \\mov { z20.d - z23.d }, za.d[w8, 1, vgx4]
        \\fadd z20.s, z20.s, z21.s
        \\fadd z20.s, z20.s, z22.s
        \\fadd z20.s, z20.s, z23.s
        \\adr x12, 14f
        \\ldr z4, [x12]
        \\tbl z20.s, { z20.s }, z4.s
        \\
        \\fmov s2, w3
        \\fmov s3, w2
        \\ld1w { z16.s }, p0/z, [x7]
        \\mov z17.s, s2
        \\fmul z16.s, z16.s, z17.s
        \\mov z17.s, s3
        \\fmla z16.s, p0/m, z20.s, z17.s
        \\st1w { z16.s }, p0, [x7]
        \\
        \\add x7, x7, #64
        \\add x13, x13, x14
        \\subs x15, x15, #16
        \\b.ne 10b
        \\
        \\13:
        \\ret
        \\
        \\14:
        \\.word 0, 4, 8, 12, 2, 6, 10, 14, 1, 5, 9, 13, 3, 7, 11, 15
    ::: .{ .memory = true });
}
