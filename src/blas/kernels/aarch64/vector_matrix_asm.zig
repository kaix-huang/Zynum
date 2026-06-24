// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later

//! AArch64 SVE/SME2 vector and matrix-vector microkernels expressed as Zig-owned whole-function inline asm.

const asm_fragments = @import("asm_fragments.zig");

pub noinline fn sveDgemvTF64(m: usize, n: usize, alpha: f64, a: [*]const f64, lda_bytes: usize, x: [*]const f64, y: [*]f64) callconv(.naked) void {
    _ = m;
    _ = n;
    _ = alpha;
    _ = a;
    _ = lda_bytes;
    _ = x;
    _ = y;
    asm volatile (
        \\cbz x0, 18f
        \\cbz x1, 18f
    ++ asm_fragments.ptrue_p1_d ++
        \\
        \\10:
        \\cmp x1, #8
        \\b.lo 14f
        \\
        \\11:
        \\mov x8, x2
        \\add x9, x8, x3
        \\add x10, x9, x3
        \\add x11, x10, x3
        \\add x12, x11, x3
        \\add x13, x12, x3
        \\add x14, x13, x3
        \\add x15, x14, x3
        \\
        \\dup z0.d, #0
        \\dup z1.d, #0
        \\dup z2.d, #0
        \\dup z3.d, #0
        \\dup z4.d, #0
        \\dup z5.d, #0
        \\dup z6.d, #0
        \\dup z7.d, #0
        \\mov x16, #0
        \\
        \\12:
        \\whilelo p0.d, x16, x0
        \\b.none 13f
        \\ld1d { z16.d }, p0/z, [x4, x16, lsl #3]
        \\ld1d { z17.d }, p0/z, [x8, x16, lsl #3]
        \\ld1d { z18.d }, p0/z, [x9, x16, lsl #3]
        \\ld1d { z19.d }, p0/z, [x10, x16, lsl #3]
        \\ld1d { z20.d }, p0/z, [x11, x16, lsl #3]
        \\ld1d { z21.d }, p0/z, [x12, x16, lsl #3]
        \\ld1d { z22.d }, p0/z, [x13, x16, lsl #3]
        \\ld1d { z23.d }, p0/z, [x14, x16, lsl #3]
        \\ld1d { z24.d }, p0/z, [x15, x16, lsl #3]
        \\fmla z0.d, p0/m, z17.d, z16.d
        \\fmla z1.d, p0/m, z18.d, z16.d
        \\fmla z2.d, p0/m, z19.d, z16.d
        \\fmla z3.d, p0/m, z20.d, z16.d
        \\fmla z4.d, p0/m, z21.d, z16.d
        \\fmla z5.d, p0/m, z22.d, z16.d
        \\fmla z6.d, p0/m, z23.d, z16.d
        \\fmla z7.d, p0/m, z24.d, z16.d
        \\incd x16
        \\b 12b
        \\
        \\13:
        \\faddv d1, p1, z0.d
        \\ldr d2, [x5]
        \\fmadd d2, d0, d1, d2
        \\str d2, [x5]
        \\faddv d1, p1, z1.d
        \\ldr d2, [x5, #8]
        \\fmadd d2, d0, d1, d2
        \\str d2, [x5, #8]
        \\faddv d1, p1, z2.d
        \\ldr d2, [x5, #16]
        \\fmadd d2, d0, d1, d2
        \\str d2, [x5, #16]
        \\faddv d1, p1, z3.d
        \\ldr d2, [x5, #24]
        \\fmadd d2, d0, d1, d2
        \\str d2, [x5, #24]
        \\faddv d1, p1, z4.d
        \\ldr d2, [x5, #32]
        \\fmadd d2, d0, d1, d2
        \\str d2, [x5, #32]
        \\faddv d1, p1, z5.d
        \\ldr d2, [x5, #40]
        \\fmadd d2, d0, d1, d2
        \\str d2, [x5, #40]
        \\faddv d1, p1, z6.d
        \\ldr d2, [x5, #48]
        \\fmadd d2, d0, d1, d2
        \\str d2, [x5, #48]
        \\faddv d1, p1, z7.d
        \\ldr d2, [x5, #56]
        \\fmadd d2, d0, d1, d2
        \\str d2, [x5, #56]
        \\
        \\add x2, x15, x3
        \\add x5, x5, #64
        \\sub x1, x1, #8
        \\b 10b
        \\
        \\14:
        \\cbz x1, 18f
        \\
        \\15:
        \\dup z0.d, #0
        \\mov x16, #0
        \\
        \\16:
        \\whilelo p0.d, x16, x0
        \\b.none 17f
        \\ld1d { z16.d }, p0/z, [x4, x16, lsl #3]
        \\ld1d { z17.d }, p0/z, [x2, x16, lsl #3]
        \\fmla z0.d, p0/m, z17.d, z16.d
        \\incd x16
        \\b 16b
        \\
        \\17:
        \\faddv d1, p1, z0.d
        \\ldr d2, [x5]
        \\fmadd d2, d0, d1, d2
        \\str d2, [x5]
        \\add x2, x2, x3
        \\add x5, x5, #8
        \\subs x1, x1, #1
        \\b.ne 15b
        \\
        \\18:
        \\ret
    ::: .{ .memory = true });
}

pub noinline fn sveDgemvTF64FullN8(m: usize, n: usize, alpha: f64, a: [*]const f64, lda_bytes: usize, x: [*]const f64, y: [*]f64) callconv(.naked) void {
    _ = m;
    _ = n;
    _ = alpha;
    _ = a;
    _ = lda_bytes;
    _ = x;
    _ = y;
    asm volatile (
        \\cbz x0, 15f
        \\cbz x1, 15f
    ++ asm_fragments.ptrue_p0_d ++ asm_fragments.ptrue_p1_d ++
        \\
        \\10:
        \\cmp x1, #8
        \\b.lo 14f
        \\
        \\11:
        \\mov x8, x2
        \\add x9, x8, x3
        \\add x10, x9, x3
        \\add x11, x10, x3
        \\add x12, x11, x3
        \\add x13, x12, x3
        \\add x14, x13, x3
        \\add x15, x14, x3
        \\
        \\dup z0.d, #0
        \\dup z1.d, #0
        \\dup z2.d, #0
        \\dup z3.d, #0
        \\dup z4.d, #0
        \\dup z5.d, #0
        \\dup z6.d, #0
        \\dup z7.d, #0
        \\mov x16, #0
        \\
        \\12:
        \\cmp x16, x0
        \\b.hs 13f
        \\ld1d { z16.d }, p0/z, [x4, x16, lsl #3]
        \\ld1d { z17.d }, p0/z, [x8, x16, lsl #3]
        \\ld1d { z18.d }, p0/z, [x9, x16, lsl #3]
        \\ld1d { z19.d }, p0/z, [x10, x16, lsl #3]
        \\ld1d { z20.d }, p0/z, [x11, x16, lsl #3]
        \\ld1d { z21.d }, p0/z, [x12, x16, lsl #3]
        \\ld1d { z22.d }, p0/z, [x13, x16, lsl #3]
        \\ld1d { z23.d }, p0/z, [x14, x16, lsl #3]
        \\ld1d { z24.d }, p0/z, [x15, x16, lsl #3]
        \\fmla z0.d, p0/m, z17.d, z16.d
        \\fmla z1.d, p0/m, z18.d, z16.d
        \\fmla z2.d, p0/m, z19.d, z16.d
        \\fmla z3.d, p0/m, z20.d, z16.d
        \\fmla z4.d, p0/m, z21.d, z16.d
        \\fmla z5.d, p0/m, z22.d, z16.d
        \\fmla z6.d, p0/m, z23.d, z16.d
        \\fmla z7.d, p0/m, z24.d, z16.d
        \\incd x16
        \\b 12b
        \\
        \\13:
        \\faddv d1, p1, z0.d
        \\ldr d2, [x5]
        \\fmadd d2, d0, d1, d2
        \\str d2, [x5]
        \\faddv d1, p1, z1.d
        \\ldr d2, [x5, #8]
        \\fmadd d2, d0, d1, d2
        \\str d2, [x5, #8]
        \\faddv d1, p1, z2.d
        \\ldr d2, [x5, #16]
        \\fmadd d2, d0, d1, d2
        \\str d2, [x5, #16]
        \\faddv d1, p1, z3.d
        \\ldr d2, [x5, #24]
        \\fmadd d2, d0, d1, d2
        \\str d2, [x5, #24]
        \\faddv d1, p1, z4.d
        \\ldr d2, [x5, #32]
        \\fmadd d2, d0, d1, d2
        \\str d2, [x5, #32]
        \\faddv d1, p1, z5.d
        \\ldr d2, [x5, #40]
        \\fmadd d2, d0, d1, d2
        \\str d2, [x5, #40]
        \\faddv d1, p1, z6.d
        \\ldr d2, [x5, #48]
        \\fmadd d2, d0, d1, d2
        \\str d2, [x5, #48]
        \\faddv d1, p1, z7.d
        \\ldr d2, [x5, #56]
        \\fmadd d2, d0, d1, d2
        \\str d2, [x5, #56]
        \\
        \\add x2, x15, x3
        \\add x5, x5, #64
        \\sub x1, x1, #8
        \\b 10b
        \\
        \\14:
        \\cbz x1, 15f
        \\b 15f
        \\
        \\15:
        \\ret
    ::: .{ .memory = true });
}

pub noinline fn sveDgemvTF64FullN8Acc2(m: usize, n: usize, alpha: f64, a: [*]const f64, lda_bytes: usize, x: [*]const f64, y: [*]f64) callconv(.naked) void {
    _ = m;
    _ = n;
    _ = alpha;
    _ = a;
    _ = lda_bytes;
    _ = x;
    _ = y;
    asm volatile (
        \\cbz x0, 15f
        \\cbz x1, 15f
    ++ asm_fragments.ptrue_p0_d ++ asm_fragments.ptrue_p1_d ++
        \\
        \\10:
        \\cmp x1, #8
        \\b.lo 14f
        \\
        \\11:
        \\mov x8, x2
        \\add x9, x8, x3
        \\add x10, x9, x3
        \\add x11, x10, x3
        \\add x12, x11, x3
        \\add x13, x12, x3
        \\add x14, x13, x3
        \\add x15, x14, x3
        \\
        \\dup z0.d, #0
        \\dup z1.d, #0
        \\dup z2.d, #0
        \\dup z3.d, #0
        \\dup z4.d, #0
        \\dup z5.d, #0
        \\dup z6.d, #0
        \\dup z7.d, #0
        \\dup z8.d, #0
        \\dup z9.d, #0
        \\dup z10.d, #0
        \\dup z11.d, #0
        \\dup z12.d, #0
        \\dup z13.d, #0
        \\dup z14.d, #0
        \\dup z15.d, #0
        \\mov x16, #0
        \\
        \\12:
        \\cmp x16, x0
        \\b.hs 13f
        \\ld1d { z16.d }, p0/z, [x4, x16, lsl #3]
        \\ld1d { z17.d }, p0/z, [x8, x16, lsl #3]
        \\ld1d { z18.d }, p0/z, [x9, x16, lsl #3]
        \\ld1d { z19.d }, p0/z, [x10, x16, lsl #3]
        \\ld1d { z20.d }, p0/z, [x11, x16, lsl #3]
        \\ld1d { z21.d }, p0/z, [x12, x16, lsl #3]
        \\ld1d { z22.d }, p0/z, [x13, x16, lsl #3]
        \\ld1d { z23.d }, p0/z, [x14, x16, lsl #3]
        \\ld1d { z24.d }, p0/z, [x15, x16, lsl #3]
        \\fmla z0.d, p0/m, z17.d, z16.d
        \\fmla z1.d, p0/m, z18.d, z16.d
        \\fmla z2.d, p0/m, z19.d, z16.d
        \\fmla z3.d, p0/m, z20.d, z16.d
        \\fmla z4.d, p0/m, z21.d, z16.d
        \\fmla z5.d, p0/m, z22.d, z16.d
        \\fmla z6.d, p0/m, z23.d, z16.d
        \\fmla z7.d, p0/m, z24.d, z16.d
        \\incd x16
        \\
        \\ld1d { z16.d }, p0/z, [x4, x16, lsl #3]
        \\ld1d { z17.d }, p0/z, [x8, x16, lsl #3]
        \\ld1d { z18.d }, p0/z, [x9, x16, lsl #3]
        \\ld1d { z19.d }, p0/z, [x10, x16, lsl #3]
        \\ld1d { z20.d }, p0/z, [x11, x16, lsl #3]
        \\ld1d { z21.d }, p0/z, [x12, x16, lsl #3]
        \\ld1d { z22.d }, p0/z, [x13, x16, lsl #3]
        \\ld1d { z23.d }, p0/z, [x14, x16, lsl #3]
        \\ld1d { z24.d }, p0/z, [x15, x16, lsl #3]
        \\fmla z8.d, p0/m, z17.d, z16.d
        \\fmla z9.d, p0/m, z18.d, z16.d
        \\fmla z10.d, p0/m, z19.d, z16.d
        \\fmla z11.d, p0/m, z20.d, z16.d
        \\fmla z12.d, p0/m, z21.d, z16.d
        \\fmla z13.d, p0/m, z22.d, z16.d
        \\fmla z14.d, p0/m, z23.d, z16.d
        \\fmla z15.d, p0/m, z24.d, z16.d
        \\incd x16
        \\b 12b
        \\
        \\13:
        \\fadd z0.d, z0.d, z8.d
        \\fadd z1.d, z1.d, z9.d
        \\fadd z2.d, z2.d, z10.d
        \\fadd z3.d, z3.d, z11.d
        \\fadd z4.d, z4.d, z12.d
        \\fadd z5.d, z5.d, z13.d
        \\fadd z6.d, z6.d, z14.d
        \\fadd z7.d, z7.d, z15.d
        \\
        \\faddv d1, p1, z0.d
        \\ldr d2, [x5]
        \\fmadd d2, d0, d1, d2
        \\str d2, [x5]
        \\faddv d1, p1, z1.d
        \\ldr d2, [x5, #8]
        \\fmadd d2, d0, d1, d2
        \\str d2, [x5, #8]
        \\faddv d1, p1, z2.d
        \\ldr d2, [x5, #16]
        \\fmadd d2, d0, d1, d2
        \\str d2, [x5, #16]
        \\faddv d1, p1, z3.d
        \\ldr d2, [x5, #24]
        \\fmadd d2, d0, d1, d2
        \\str d2, [x5, #24]
        \\faddv d1, p1, z4.d
        \\ldr d2, [x5, #32]
        \\fmadd d2, d0, d1, d2
        \\str d2, [x5, #32]
        \\faddv d1, p1, z5.d
        \\ldr d2, [x5, #40]
        \\fmadd d2, d0, d1, d2
        \\str d2, [x5, #40]
        \\faddv d1, p1, z6.d
        \\ldr d2, [x5, #48]
        \\fmadd d2, d0, d1, d2
        \\str d2, [x5, #48]
        \\faddv d1, p1, z7.d
        \\ldr d2, [x5, #56]
        \\fmadd d2, d0, d1, d2
        \\str d2, [x5, #56]
        \\
        \\add x2, x15, x3
        \\add x5, x5, #64
        \\sub x1, x1, #8
        \\b 10b
        \\
        \\14:
        \\cbz x1, 15f
        \\b 15f
        \\
        \\15:
        \\ret
    ::: .{ .memory = true });
}

pub noinline fn sme2DgemvNF64256x1(m: usize, n: usize, alpha: f64, beta: f64, a: [*]const f64, lda_bytes: usize, x: [*]const f64, y: [*]f64) callconv(.naked) void {
    _ = m;
    _ = n;
    _ = alpha;
    _ = beta;
    _ = a;
    _ = lda_bytes;
    _ = x;
    _ = y;
    asm volatile (
        \\cbz x0, 12f
        \\cbz x1, 12f
    ++ asm_fragments.ptrue_pn8_d ++
        \\
        \\mov x6, x2              // Current 256-row A panel.
        \\mov x7, x5              // Current 256-row y panel.
        \\mov x14, x0             // Remaining rows.
        \\
        \\10:
        \\zero { za }
        \\mov x15, x6             // A column pointer for this row panel.
        \\mov x16, x4             // x pointer.
        \\mov x17, x1             // Remaining columns.
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
        \\add x15, x15, x3
        \\subs x17, x17, #1
        \\b.ne 11b
        \\
        \\mov x9, x7
        \\mov w8, #0
        \\mov w11, #8
        \\mov { z16.d - z19.d }, za.d[w8, 0, vgx4]
        \\mov { z20.d - z23.d }, za.d[w11, 0, vgx4]
        \\zero { za0.d }
        \\ld1d { z24.d - z27.d }, pn8/z, [x9]
        \\ld1d { z28.d - z31.d }, pn8/z, [x9, #4, mul vl]
        \\mov z4.d, d1
        \\fmla za.d[w8, 0, vgx4], { z24.d - z27.d }, z4.d
        \\fmla za.d[w11, 0, vgx4], { z28.d - z31.d }, z4.d
        \\mov z4.d, d0
        \\fmla za.d[w8, 0, vgx4], { z16.d - z19.d }, z4.d
        \\fmla za.d[w11, 0, vgx4], { z20.d - z23.d }, z4.d
        \\
        \\mov w8, #1
        \\mov w11, #9
        \\mov { z16.d - z19.d }, za.d[w8, 0, vgx4]
        \\mov { z20.d - z23.d }, za.d[w11, 0, vgx4]
        \\zero { za1.d }
        \\ld1d { z24.d - z27.d }, pn8/z, [x9, #8, mul vl]
        \\ld1d { z28.d - z31.d }, pn8/z, [x9, #12, mul vl]
        \\mov z4.d, d1
        \\fmla za.d[w8, 0, vgx4], { z24.d - z27.d }, z4.d
        \\fmla za.d[w11, 0, vgx4], { z28.d - z31.d }, z4.d
        \\mov z4.d, d0
        \\fmla za.d[w8, 0, vgx4], { z16.d - z19.d }, z4.d
        \\fmla za.d[w11, 0, vgx4], { z20.d - z23.d }, z4.d
        \\
        \\mov w8, #2
        \\mov w11, #10
        \\mov { z16.d - z19.d }, za.d[w8, 0, vgx4]
        \\mov { z20.d - z23.d }, za.d[w11, 0, vgx4]
        \\zero { za2.d }
        \\ld1d { z24.d - z27.d }, pn8/z, [x9, #16, mul vl]
        \\ld1d { z28.d - z31.d }, pn8/z, [x9, #20, mul vl]
        \\mov z4.d, d1
        \\fmla za.d[w8, 0, vgx4], { z24.d - z27.d }, z4.d
        \\fmla za.d[w11, 0, vgx4], { z28.d - z31.d }, z4.d
        \\mov z4.d, d0
        \\fmla za.d[w8, 0, vgx4], { z16.d - z19.d }, z4.d
        \\fmla za.d[w11, 0, vgx4], { z20.d - z23.d }, z4.d
        \\
        \\mov w8, #3
        \\mov w11, #11
        \\mov { z16.d - z19.d }, za.d[w8, 0, vgx4]
        \\mov { z20.d - z23.d }, za.d[w11, 0, vgx4]
        \\zero { za3.d }
        \\ld1d { z24.d - z27.d }, pn8/z, [x9, #24, mul vl]
        \\ld1d { z28.d - z31.d }, pn8/z, [x9, #28, mul vl]
        \\mov z4.d, d1
        \\fmla za.d[w8, 0, vgx4], { z24.d - z27.d }, z4.d
        \\fmla za.d[w11, 0, vgx4], { z28.d - z31.d }, z4.d
        \\mov z4.d, d0
        \\fmla za.d[w8, 0, vgx4], { z16.d - z19.d }, z4.d
        \\fmla za.d[w11, 0, vgx4], { z20.d - z23.d }, z4.d
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
        \\mov w8, #2
        \\mov w11, #10
        \\mov { z16.d - z19.d }, za.d[w8, 0, vgx4]
        \\mov { z20.d - z23.d }, za.d[w11, 0, vgx4]
        \\st1d { z16.d - z19.d }, pn8, [x9, #16, mul vl]
        \\st1d { z20.d - z23.d }, pn8, [x9, #20, mul vl]
        \\mov w8, #3
        \\mov w11, #11
        \\mov { z16.d - z19.d }, za.d[w8, 0, vgx4]
        \\mov { z20.d - z23.d }, za.d[w11, 0, vgx4]
        \\st1d { z16.d - z19.d }, pn8, [x9, #24, mul vl]
        \\st1d { z20.d - z23.d }, pn8, [x9, #28, mul vl]
        \\
        \\add x6, x6, #2048
        \\add x7, x7, #2048
        \\subs x14, x14, #256
        \\b.ne 10b
        \\
        \\12:
        \\ret
    ::: .{ .memory = true });
}

pub noinline fn sme2DgemvTF648x32(m: usize, n: usize, alpha: f64, beta: f64, a: [*]const f64, lda_bytes: usize, x: [*]const f64, y: [*]f64) callconv(.naked) void {
    _ = m;
    _ = n;
    _ = alpha;
    _ = beta;
    _ = a;
    _ = lda_bytes;
    _ = x;
    _ = y;
    asm volatile (
        \\cbz x0, 13f
        \\cbz x1, 13f
    ++ asm_fragments.ptrue_pn8_d ++
        \\
        \\mov x7, x2              // Current 8-column A panel.
        \\mov x15, x1             // Remaining output columns.
        \\lsl x6, x3, #3          // lda_bytes * 8.
        \\
        \\10:
        \\zero { za }
        \\mov x12, #0             // Row offset in elements.
        \\mov x10, x4             // x row pointer.
        \\mov x16, x7             // A column-0 row pointer for this panel.
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
        \\add x9, x9, x3
        \\ld1d { z17.d, z21.d, z25.d, z29.d }, pn8/z, [x9]
        \\add x9, x9, x3
        \\ld1d { z18.d, z22.d, z26.d, z30.d }, pn8/z, [x9]
        \\add x9, x9, x3
        \\ld1d { z19.d, z23.d, z27.d, z31.d }, pn8/z, [x9]
        \\add x9, x9, x3
        \\fmla za.d[w8, 0, vgx4], { z16.d - z19.d }, z4.d
        \\fmla za.d[w11, 0, vgx4], { z20.d - z23.d }, z5.d
        \\fmla za.d[w8, 1, vgx4], { z24.d - z27.d }, z6.d
        \\fmla za.d[w11, 1, vgx4], { z28.d - z31.d }, z7.d
        \\
        \\ld1d { z16.d, z20.d, z24.d, z28.d }, pn8/z, [x9]
        \\add x9, x9, x3
        \\ld1d { z17.d, z21.d, z25.d, z29.d }, pn8/z, [x9]
        \\add x9, x9, x3
        \\ld1d { z18.d, z22.d, z26.d, z30.d }, pn8/z, [x9]
        \\add x9, x9, x3
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
        \\adr x13, 14f
        \\ldr z4, [x13]
        \\tbl z20.d, { z20.d }, z4.d
        \\
        \\mov x13, #8
        \\whilelo p0.d, xzr, x13
        \\ld1d { z16.d }, p0/z, [x5]
        \\mov z17.d, d1
        \\fmul z16.d, z16.d, z17.d
        \\mov z17.d, d0
        \\fmla z16.d, p0/m, z20.d, z17.d
        \\st1d { z16.d }, p0, [x5]
        \\
        \\add x5, x5, #64
        \\add x7, x7, x6
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

pub noinline fn sveDscalF64(n: usize, alpha: f64, x: [*]f64) callconv(.naked) void {
    _ = n;
    _ = alpha;
    _ = x;
    asm volatile (
        \\cbz x0, 12f
    ++ asm_fragments.ptrue_p0_d ++
        \\mov z4.d, d0
        \\cntd x6
        \\lsl x7, x6, #2
        \\
        \\10:
        \\cmp x0, x7
        \\b.lo 11f
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
        \\b 10b
        \\
        \\11:
        \\cbz x0, 12f
        \\mov x8, #0
        \\whilelo p1.d, x8, x0
        \\ld1d { z0.d }, p1/z, [x1]
        \\fmul z0.d, p1/m, z0.d, z4.d
        \\st1d { z0.d }, p1, [x1]
        \\
        \\12:
        \\ret
    ::: .{ .memory = true });
}
