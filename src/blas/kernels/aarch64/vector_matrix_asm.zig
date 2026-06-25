// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later

//! AArch64 SVE/SME2 vector and matrix-vector microkernels expressed as Zig-owned whole-function inline asm.

const asm_fragments = @import("asm_fragments.zig");

pub noinline fn asimdDgerF64x4Rows8(m: usize, n: usize, alpha: f64, x: [*]const f64, y: [*]const f64, a: [*]f64, lda_bytes: usize) callconv(.naked) void {
    _ = m;
    _ = n;
    _ = alpha;
    _ = x;
    _ = y;
    _ = a;
    _ = lda_bytes;
    asm volatile (
        \\cbz x0, 99f
        \\cbz x1, 99f
        \\mov x6, x1
        \\mov x7, x4
        \\
        \\10:
        \\cmp x6, #4
        \\b.lo 40f
        \\mov x8, x7
        \\add x9, x8, x5
        \\add x10, x9, x5
        \\add x11, x10, x5
        \\ldr d1, [x3]
        \\ldr d2, [x3, #8]
        \\ldr d3, [x3, #16]
        \\ldr d4, [x3, #24]
        \\fmul d1, d1, d0
        \\fmul d2, d2, d0
        \\fmul d3, d3, d0
        \\fmul d4, d4, d0
        \\mov x12, x2
        \\mov x13, x8
        \\mov x14, x9
        \\mov x15, x10
        \\mov x16, x11
        \\mov x17, x0
        \\
        \\11:
        \\cmp x17, #8
        \\b.lo 20f
        \\prfm pldl1keep, [x12, #256]
        \\ldp q16, q17, [x12], #32
        \\ldp q18, q19, [x12], #32
        \\ldp q20, q21, [x13]
        \\ldp q22, q23, [x13, #32]
        \\fmla v20.2d, v16.2d, v1.d[0]
        \\fmla v21.2d, v17.2d, v1.d[0]
        \\fmla v22.2d, v18.2d, v1.d[0]
        \\fmla v23.2d, v19.2d, v1.d[0]
        \\stp q20, q21, [x13]
        \\stp q22, q23, [x13, #32]
        \\ldp q20, q21, [x14]
        \\ldp q22, q23, [x14, #32]
        \\fmla v20.2d, v16.2d, v2.d[0]
        \\fmla v21.2d, v17.2d, v2.d[0]
        \\fmla v22.2d, v18.2d, v2.d[0]
        \\fmla v23.2d, v19.2d, v2.d[0]
        \\stp q20, q21, [x14]
        \\stp q22, q23, [x14, #32]
        \\ldp q20, q21, [x15]
        \\ldp q22, q23, [x15, #32]
        \\fmla v20.2d, v16.2d, v3.d[0]
        \\fmla v21.2d, v17.2d, v3.d[0]
        \\fmla v22.2d, v18.2d, v3.d[0]
        \\fmla v23.2d, v19.2d, v3.d[0]
        \\stp q20, q21, [x15]
        \\stp q22, q23, [x15, #32]
        \\ldp q20, q21, [x16]
        \\ldp q22, q23, [x16, #32]
        \\fmla v20.2d, v16.2d, v4.d[0]
        \\fmla v21.2d, v17.2d, v4.d[0]
        \\fmla v22.2d, v18.2d, v4.d[0]
        \\fmla v23.2d, v19.2d, v4.d[0]
        \\stp q20, q21, [x16]
        \\stp q22, q23, [x16, #32]
        \\add x13, x13, #64
        \\add x14, x14, #64
        \\add x15, x15, #64
        \\add x16, x16, #64
        \\sub x17, x17, #8
        \\b 11b
        \\
        \\20:
        \\cmp x17, #4
        \\b.lo 30f
        \\ldp q16, q17, [x12], #32
        \\ldp q20, q21, [x13]
        \\fmla v20.2d, v16.2d, v1.d[0]
        \\fmla v21.2d, v17.2d, v1.d[0]
        \\stp q20, q21, [x13], #32
        \\ldp q20, q21, [x14]
        \\fmla v20.2d, v16.2d, v2.d[0]
        \\fmla v21.2d, v17.2d, v2.d[0]
        \\stp q20, q21, [x14], #32
        \\ldp q20, q21, [x15]
        \\fmla v20.2d, v16.2d, v3.d[0]
        \\fmla v21.2d, v17.2d, v3.d[0]
        \\stp q20, q21, [x15], #32
        \\ldp q20, q21, [x16]
        \\fmla v20.2d, v16.2d, v4.d[0]
        \\fmla v21.2d, v17.2d, v4.d[0]
        \\stp q20, q21, [x16], #32
        \\sub x17, x17, #4
        \\
        \\30:
        \\cmp x17, #2
        \\b.lo 31f
        \\ldr q16, [x12], #16
        \\ldr q20, [x13]
        \\fmla v20.2d, v16.2d, v1.d[0]
        \\str q20, [x13], #16
        \\ldr q20, [x14]
        \\fmla v20.2d, v16.2d, v2.d[0]
        \\str q20, [x14], #16
        \\ldr q20, [x15]
        \\fmla v20.2d, v16.2d, v3.d[0]
        \\str q20, [x15], #16
        \\ldr q20, [x16]
        \\fmla v20.2d, v16.2d, v4.d[0]
        \\str q20, [x16], #16
        \\sub x17, x17, #2
        \\
        \\31:
        \\cbz x17, 32f
        \\ldr d16, [x12]
        \\ldr d20, [x13]
        \\fmadd d20, d16, d1, d20
        \\str d20, [x13]
        \\ldr d20, [x14]
        \\fmadd d20, d16, d2, d20
        \\str d20, [x14]
        \\ldr d20, [x15]
        \\fmadd d20, d16, d3, d20
        \\str d20, [x15]
        \\ldr d20, [x16]
        \\fmadd d20, d16, d4, d20
        \\str d20, [x16]
        \\
        \\32:
        \\add x7, x11, x5
        \\add x3, x3, #32
        \\sub x6, x6, #4
        \\b 10b
        \\
        \\40:
        \\cbz x6, 99f
        \\ldr d1, [x3], #8
        \\fmul d1, d1, d0
        \\mov x12, x2
        \\mov x13, x7
        \\mov x17, x0
        \\
        \\41:
        \\cmp x17, #8
        \\b.lo 50f
        \\ldp q16, q17, [x12], #32
        \\ldp q18, q19, [x12], #32
        \\ldp q20, q21, [x13]
        \\ldp q22, q23, [x13, #32]
        \\fmla v20.2d, v16.2d, v1.d[0]
        \\fmla v21.2d, v17.2d, v1.d[0]
        \\fmla v22.2d, v18.2d, v1.d[0]
        \\fmla v23.2d, v19.2d, v1.d[0]
        \\stp q20, q21, [x13]
        \\stp q22, q23, [x13, #32]
        \\add x13, x13, #64
        \\sub x17, x17, #8
        \\b 41b
        \\
        \\50:
        \\cmp x17, #4
        \\b.lo 60f
        \\ldp q16, q17, [x12], #32
        \\ldp q20, q21, [x13]
        \\fmla v20.2d, v16.2d, v1.d[0]
        \\fmla v21.2d, v17.2d, v1.d[0]
        \\stp q20, q21, [x13], #32
        \\sub x17, x17, #4
        \\
        \\60:
        \\cmp x17, #2
        \\b.lo 70f
        \\ldr q16, [x12], #16
        \\ldr q20, [x13]
        \\fmla v20.2d, v16.2d, v1.d[0]
        \\str q20, [x13], #16
        \\sub x17, x17, #2
        \\
        \\70:
        \\cbz x17, 80f
        \\ldr d16, [x12]
        \\ldr d20, [x13]
        \\fmadd d20, d16, d1, d20
        \\str d20, [x13]
        \\
        \\80:
        \\add x7, x7, x5
        \\subs x6, x6, #1
        \\b.ne 40b
        \\
        \\99:
        \\ret
        ::: .{ .memory = true });
}

pub noinline fn asimdDgerF64x8Rows8(m: usize, n: usize, alpha: f64, x: [*]const f64, y: [*]const f64, a: [*]f64, lda_bytes: usize) callconv(.naked) void {
    _ = m;
    _ = n;
    _ = alpha;
    _ = x;
    _ = y;
    _ = a;
    _ = lda_bytes;
    asm volatile (
        \\cbz x0, 99f
        \\cbz x1, 99f
        \\mov x6, x1
        \\mov x7, x4
        \\
        \\10:
        \\cmp x6, #8
        \\b.lo 99f
        \\mov x8, x7
        \\add x9, x8, x5
        \\add x10, x9, x5
        \\add x11, x10, x5
        \\add x12, x11, x5
        \\add x13, x12, x5
        \\add x14, x13, x5
        \\add x15, x14, x5
        \\add x7, x15, x5
        \\ldr d1, [x3]
        \\ldr d2, [x3, #8]
        \\ldr d3, [x3, #16]
        \\ldr d4, [x3, #24]
        \\ldr d5, [x3, #32]
        \\ldr d6, [x3, #40]
        \\ldr d7, [x3, #48]
        \\ldr d24, [x3, #56]
        \\fmul d1, d1, d0
        \\fmul d2, d2, d0
        \\fmul d3, d3, d0
        \\fmul d4, d4, d0
        \\fmul d5, d5, d0
        \\fmul d6, d6, d0
        \\fmul d7, d7, d0
        \\fmul d24, d24, d0
        \\mov x16, x2
        \\mov x17, x0
        \\
        \\11:
        \\cmp x17, #8
        \\b.lo 20f
        \\prfm pldl1keep, [x16, #256]
        \\ldp q16, q17, [x16], #32
        \\ldp q18, q19, [x16], #32
        \\ldp q20, q21, [x8]
        \\ldp q22, q23, [x8, #32]
        \\fmla v20.2d, v16.2d, v1.d[0]
        \\fmla v21.2d, v17.2d, v1.d[0]
        \\fmla v22.2d, v18.2d, v1.d[0]
        \\fmla v23.2d, v19.2d, v1.d[0]
        \\stp q20, q21, [x8]
        \\stp q22, q23, [x8, #32]
        \\ldp q20, q21, [x9]
        \\ldp q22, q23, [x9, #32]
        \\fmla v20.2d, v16.2d, v2.d[0]
        \\fmla v21.2d, v17.2d, v2.d[0]
        \\fmla v22.2d, v18.2d, v2.d[0]
        \\fmla v23.2d, v19.2d, v2.d[0]
        \\stp q20, q21, [x9]
        \\stp q22, q23, [x9, #32]
        \\ldp q20, q21, [x10]
        \\ldp q22, q23, [x10, #32]
        \\fmla v20.2d, v16.2d, v3.d[0]
        \\fmla v21.2d, v17.2d, v3.d[0]
        \\fmla v22.2d, v18.2d, v3.d[0]
        \\fmla v23.2d, v19.2d, v3.d[0]
        \\stp q20, q21, [x10]
        \\stp q22, q23, [x10, #32]
        \\ldp q20, q21, [x11]
        \\ldp q22, q23, [x11, #32]
        \\fmla v20.2d, v16.2d, v4.d[0]
        \\fmla v21.2d, v17.2d, v4.d[0]
        \\fmla v22.2d, v18.2d, v4.d[0]
        \\fmla v23.2d, v19.2d, v4.d[0]
        \\stp q20, q21, [x11]
        \\stp q22, q23, [x11, #32]
        \\ldp q20, q21, [x12]
        \\ldp q22, q23, [x12, #32]
        \\fmla v20.2d, v16.2d, v5.d[0]
        \\fmla v21.2d, v17.2d, v5.d[0]
        \\fmla v22.2d, v18.2d, v5.d[0]
        \\fmla v23.2d, v19.2d, v5.d[0]
        \\stp q20, q21, [x12]
        \\stp q22, q23, [x12, #32]
        \\ldp q20, q21, [x13]
        \\ldp q22, q23, [x13, #32]
        \\fmla v20.2d, v16.2d, v6.d[0]
        \\fmla v21.2d, v17.2d, v6.d[0]
        \\fmla v22.2d, v18.2d, v6.d[0]
        \\fmla v23.2d, v19.2d, v6.d[0]
        \\stp q20, q21, [x13]
        \\stp q22, q23, [x13, #32]
        \\ldp q20, q21, [x14]
        \\ldp q22, q23, [x14, #32]
        \\fmla v20.2d, v16.2d, v7.d[0]
        \\fmla v21.2d, v17.2d, v7.d[0]
        \\fmla v22.2d, v18.2d, v7.d[0]
        \\fmla v23.2d, v19.2d, v7.d[0]
        \\stp q20, q21, [x14]
        \\stp q22, q23, [x14, #32]
        \\ldp q20, q21, [x15]
        \\ldp q22, q23, [x15, #32]
        \\fmla v20.2d, v16.2d, v24.d[0]
        \\fmla v21.2d, v17.2d, v24.d[0]
        \\fmla v22.2d, v18.2d, v24.d[0]
        \\fmla v23.2d, v19.2d, v24.d[0]
        \\stp q20, q21, [x15]
        \\stp q22, q23, [x15, #32]
        \\add x8, x8, #64
        \\add x9, x9, #64
        \\add x10, x10, #64
        \\add x11, x11, #64
        \\add x12, x12, #64
        \\add x13, x13, #64
        \\add x14, x14, #64
        \\add x15, x15, #64
        \\sub x17, x17, #8
        \\b 11b
        \\
        \\20:
        \\cmp x17, #4
        \\b.lo 30f
        \\ldp q16, q17, [x16], #32
        \\ldp q20, q21, [x8]
        \\fmla v20.2d, v16.2d, v1.d[0]
        \\fmla v21.2d, v17.2d, v1.d[0]
        \\stp q20, q21, [x8], #32
        \\ldp q20, q21, [x9]
        \\fmla v20.2d, v16.2d, v2.d[0]
        \\fmla v21.2d, v17.2d, v2.d[0]
        \\stp q20, q21, [x9], #32
        \\ldp q20, q21, [x10]
        \\fmla v20.2d, v16.2d, v3.d[0]
        \\fmla v21.2d, v17.2d, v3.d[0]
        \\stp q20, q21, [x10], #32
        \\ldp q20, q21, [x11]
        \\fmla v20.2d, v16.2d, v4.d[0]
        \\fmla v21.2d, v17.2d, v4.d[0]
        \\stp q20, q21, [x11], #32
        \\ldp q20, q21, [x12]
        \\fmla v20.2d, v16.2d, v5.d[0]
        \\fmla v21.2d, v17.2d, v5.d[0]
        \\stp q20, q21, [x12], #32
        \\ldp q20, q21, [x13]
        \\fmla v20.2d, v16.2d, v6.d[0]
        \\fmla v21.2d, v17.2d, v6.d[0]
        \\stp q20, q21, [x13], #32
        \\ldp q20, q21, [x14]
        \\fmla v20.2d, v16.2d, v7.d[0]
        \\fmla v21.2d, v17.2d, v7.d[0]
        \\stp q20, q21, [x14], #32
        \\ldp q20, q21, [x15]
        \\fmla v20.2d, v16.2d, v24.d[0]
        \\fmla v21.2d, v17.2d, v24.d[0]
        \\stp q20, q21, [x15], #32
        \\sub x17, x17, #4
        \\
        \\30:
        \\cmp x17, #2
        \\b.lo 40f
        \\ldr q16, [x16], #16
        \\ldr q20, [x8]
        \\fmla v20.2d, v16.2d, v1.d[0]
        \\str q20, [x8], #16
        \\ldr q20, [x9]
        \\fmla v20.2d, v16.2d, v2.d[0]
        \\str q20, [x9], #16
        \\ldr q20, [x10]
        \\fmla v20.2d, v16.2d, v3.d[0]
        \\str q20, [x10], #16
        \\ldr q20, [x11]
        \\fmla v20.2d, v16.2d, v4.d[0]
        \\str q20, [x11], #16
        \\ldr q20, [x12]
        \\fmla v20.2d, v16.2d, v5.d[0]
        \\str q20, [x12], #16
        \\ldr q20, [x13]
        \\fmla v20.2d, v16.2d, v6.d[0]
        \\str q20, [x13], #16
        \\ldr q20, [x14]
        \\fmla v20.2d, v16.2d, v7.d[0]
        \\str q20, [x14], #16
        \\ldr q20, [x15]
        \\fmla v20.2d, v16.2d, v24.d[0]
        \\str q20, [x15], #16
        \\sub x17, x17, #2
        \\
        \\40:
        \\cbz x17, 50f
        \\ldr d16, [x16]
        \\ldr d20, [x8]
        \\fmadd d20, d16, d1, d20
        \\str d20, [x8]
        \\ldr d20, [x9]
        \\fmadd d20, d16, d2, d20
        \\str d20, [x9]
        \\ldr d20, [x10]
        \\fmadd d20, d16, d3, d20
        \\str d20, [x10]
        \\ldr d20, [x11]
        \\fmadd d20, d16, d4, d20
        \\str d20, [x11]
        \\ldr d20, [x12]
        \\fmadd d20, d16, d5, d20
        \\str d20, [x12]
        \\ldr d20, [x13]
        \\fmadd d20, d16, d6, d20
        \\str d20, [x13]
        \\ldr d20, [x14]
        \\fmadd d20, d16, d7, d20
        \\str d20, [x14]
        \\ldr d20, [x15]
        \\fmadd d20, d16, d24, d20
        \\str d20, [x15]
        \\
        \\50:
        \\add x3, x3, #64
        \\sub x6, x6, #8
        \\b 10b
        \\
        \\99:
        \\ret
        ::: .{ .memory = true });
}

pub noinline fn asimdDgerF64DaxpyRows8(m: usize, n: usize, alpha: f64, x: [*]const f64, y: [*]const f64, a: [*]f64, lda_bytes: usize) callconv(.naked) void {
    _ = m;
    _ = n;
    _ = alpha;
    _ = x;
    _ = y;
    _ = a;
    _ = lda_bytes;
    asm volatile (
        \\cbz x0, 99f
        \\cbz x1, 99f
        \\mov x6, x1
        \\mov x7, x4
        \\
        \\10:
        \\ldr d1, [x3], #8
        \\fmul d1, d1, d0
        \\mov x8, x2
        \\mov x9, x7
        \\mov x10, x0
        \\
        \\11:
        \\cmp x10, #8
        \\b.lo 20f
        \\prfm pldl1keep, [x8, #256]
        \\prfm pldl1keep, [x9, #512]
        \\ldp q16, q17, [x8], #32
        \\ldp q18, q19, [x8], #32
        \\ldp q20, q21, [x9]
        \\ldp q22, q23, [x9, #32]
        \\fmla v20.2d, v16.2d, v1.d[0]
        \\fmla v21.2d, v17.2d, v1.d[0]
        \\fmla v22.2d, v18.2d, v1.d[0]
        \\fmla v23.2d, v19.2d, v1.d[0]
        \\stp q20, q21, [x9]
        \\stp q22, q23, [x9, #32]
        \\add x9, x9, #64
        \\sub x10, x10, #8
        \\b 11b
        \\
        \\20:
        \\cmp x10, #4
        \\b.lo 30f
        \\ldp q16, q17, [x8], #32
        \\ldp q20, q21, [x9]
        \\fmla v20.2d, v16.2d, v1.d[0]
        \\fmla v21.2d, v17.2d, v1.d[0]
        \\stp q20, q21, [x9], #32
        \\sub x10, x10, #4
        \\
        \\30:
        \\cmp x10, #2
        \\b.lo 40f
        \\ldr q16, [x8], #16
        \\ldr q20, [x9]
        \\fmla v20.2d, v16.2d, v1.d[0]
        \\str q20, [x9], #16
        \\sub x10, x10, #2
        \\
        \\40:
        \\cbz x10, 50f
        \\ldr d16, [x8]
        \\ldr d20, [x9]
        \\fmadd d20, d16, d1, d20
        \\str d20, [x9]
        \\
        \\50:
        \\add x7, x7, x5
        \\subs x6, x6, #1
        \\b.ne 10b
        \\
        \\99:
        \\ret
        ::: .{ .memory = true });
}

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

pub noinline fn sme2SgemvNF32256x1(m: usize, n: usize, alpha: f32, beta: f32, a: [*]const f32, lda_bytes: usize, x: [*]const f32, y: [*]f32) callconv(.naked) void {
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
        \\ptrue pn8.s
        \\ptrue p0.s
        \\
        \\mov x6, x2
        \\mov x7, x5
        \\mov x14, x0
        \\
        \\10:
        \\zero { za }
        \\mov x15, x6
        \\mov x16, x4
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
        \\add x15, x15, x3
        \\subs x17, x17, #1
        \\b.ne 11b
        \\
        \\mov x9, x7
        \\mov z4.s, s1
        \\mov z5.s, s0
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
        \\add x6, x6, #1024
        \\add x7, x7, #1024
        \\subs x14, x14, #256
        \\b.ne 10b
        \\
        \\12:
        \\ret
        ::: .{ .memory = true });
}

pub noinline fn sme2SgemvNF32512x1(m: usize, n: usize, alpha: f32, beta: f32, a: [*]const f32, lda_bytes: usize, x: [*]const f32, y: [*]f32) callconv(.naked) void {
    _ = m;
    _ = n;
    _ = alpha;
    _ = beta;
    _ = a;
    _ = lda_bytes;
    _ = x;
    _ = y;
    asm volatile ("cbz x0, 12f\n" ++
            "cbz x1, 12f\n" ++
            "ptrue pn8.s\n" ++
            "ptrue p0.s\n" ++
            "zero { za }\n" ++
            "mov x15, x2\n" ++
            "mov x16, x4\n" ++
            "mov x17, x1\n" ++
            "11:\n" ++
            "ldr s2, [x16], #4\n" ++
            "mov z4.s, s2\n" ++
            "mov x9, x15\n" ++
            "ld1w { z16.s - z19.s }, pn8/z, [x9]\n" ++
            "ld1w { z20.s - z23.s }, pn8/z, [x9, #4, mul vl]\n" ++
            "ld1w { z24.s - z27.s }, pn8/z, [x9, #8, mul vl]\n" ++
            "ld1w { z28.s - z31.s }, pn8/z, [x9, #12, mul vl]\n" ++
            "mov w8, #0\n" ++
            "mov w11, #8\n" ++
            "fmla za.s[w8, 0, vgx4], { z16.s - z19.s }, z4.s\n" ++
            "fmla za.s[w11, 0, vgx4], { z20.s - z23.s }, z4.s\n" ++
            "mov w8, #1\n" ++
            "mov w11, #9\n" ++
            "fmla za.s[w8, 0, vgx4], { z24.s - z27.s }, z4.s\n" ++
            "fmla za.s[w11, 0, vgx4], { z28.s - z31.s }, z4.s\n" ++
            "add x9, x15, #1024\n" ++
            "ld1w { z16.s - z19.s }, pn8/z, [x9]\n" ++
            "ld1w { z20.s - z23.s }, pn8/z, [x9, #4, mul vl]\n" ++
            "ld1w { z24.s - z27.s }, pn8/z, [x9, #8, mul vl]\n" ++
            "ld1w { z28.s - z31.s }, pn8/z, [x9, #12, mul vl]\n" ++
            "mov w8, #2\n" ++
            "mov w11, #10\n" ++
            "fmla za.s[w8, 0, vgx4], { z16.s - z19.s }, z4.s\n" ++
            "fmla za.s[w11, 0, vgx4], { z20.s - z23.s }, z4.s\n" ++
            "mov w8, #3\n" ++
            "mov w11, #11\n" ++
            "fmla za.s[w8, 0, vgx4], { z24.s - z27.s }, z4.s\n" ++
            "fmla za.s[w11, 0, vgx4], { z28.s - z31.s }, z4.s\n" ++
            "add x15, x15, x3\n" ++
            "subs x17, x17, #1\n" ++
            "b.ne 11b\n" ++
            "mov x9, x5\n" ++
            "mov z4.s, s1\n" ++
            "mov z5.s, s0\n" ++
            "mov w8, #0\n" ++
            "mov { z16.s - z19.s }, za.s[w8, 0, vgx4]\n" ++
            "ld1w { z20.s - z23.s }, pn8/z, [x9]\n" ++
            "fmul z20.s, p0/m, z20.s, z4.s\n" ++
            "fmul z21.s, p0/m, z21.s, z4.s\n" ++
            "fmul z22.s, p0/m, z22.s, z4.s\n" ++
            "fmul z23.s, p0/m, z23.s, z4.s\n" ++
            "fmla z20.s, p0/m, z16.s, z5.s\n" ++
            "fmla z21.s, p0/m, z17.s, z5.s\n" ++
            "fmla z22.s, p0/m, z18.s, z5.s\n" ++
            "fmla z23.s, p0/m, z19.s, z5.s\n" ++
            "st1w { z20.s - z23.s }, pn8, [x9]\n" ++
            "mov w8, #8\n" ++
            "mov { z16.s - z19.s }, za.s[w8, 0, vgx4]\n" ++
            "ld1w { z20.s - z23.s }, pn8/z, [x9, #4, mul vl]\n" ++
            "fmul z20.s, p0/m, z20.s, z4.s\n" ++
            "fmul z21.s, p0/m, z21.s, z4.s\n" ++
            "fmul z22.s, p0/m, z22.s, z4.s\n" ++
            "fmul z23.s, p0/m, z23.s, z4.s\n" ++
            "fmla z20.s, p0/m, z16.s, z5.s\n" ++
            "fmla z21.s, p0/m, z17.s, z5.s\n" ++
            "fmla z22.s, p0/m, z18.s, z5.s\n" ++
            "fmla z23.s, p0/m, z19.s, z5.s\n" ++
            "st1w { z20.s - z23.s }, pn8, [x9, #4, mul vl]\n" ++
            "mov w8, #1\n" ++
            "mov { z16.s - z19.s }, za.s[w8, 0, vgx4]\n" ++
            "ld1w { z20.s - z23.s }, pn8/z, [x9, #8, mul vl]\n" ++
            "fmul z20.s, p0/m, z20.s, z4.s\n" ++
            "fmul z21.s, p0/m, z21.s, z4.s\n" ++
            "fmul z22.s, p0/m, z22.s, z4.s\n" ++
            "fmul z23.s, p0/m, z23.s, z4.s\n" ++
            "fmla z20.s, p0/m, z16.s, z5.s\n" ++
            "fmla z21.s, p0/m, z17.s, z5.s\n" ++
            "fmla z22.s, p0/m, z18.s, z5.s\n" ++
            "fmla z23.s, p0/m, z19.s, z5.s\n" ++
            "st1w { z20.s - z23.s }, pn8, [x9, #8, mul vl]\n" ++
            "mov w8, #9\n" ++
            "mov { z16.s - z19.s }, za.s[w8, 0, vgx4]\n" ++
            "ld1w { z20.s - z23.s }, pn8/z, [x9, #12, mul vl]\n" ++
            "fmul z20.s, p0/m, z20.s, z4.s\n" ++
            "fmul z21.s, p0/m, z21.s, z4.s\n" ++
            "fmul z22.s, p0/m, z22.s, z4.s\n" ++
            "fmul z23.s, p0/m, z23.s, z4.s\n" ++
            "fmla z20.s, p0/m, z16.s, z5.s\n" ++
            "fmla z21.s, p0/m, z17.s, z5.s\n" ++
            "fmla z22.s, p0/m, z18.s, z5.s\n" ++
            "fmla z23.s, p0/m, z19.s, z5.s\n" ++
            "st1w { z20.s - z23.s }, pn8, [x9, #12, mul vl]\n" ++
            "add x9, x5, #1024\n" ++
            "mov w8, #2\n" ++
            "mov { z16.s - z19.s }, za.s[w8, 0, vgx4]\n" ++
            "ld1w { z20.s - z23.s }, pn8/z, [x9]\n" ++
            "fmul z20.s, p0/m, z20.s, z4.s\n" ++
            "fmul z21.s, p0/m, z21.s, z4.s\n" ++
            "fmul z22.s, p0/m, z22.s, z4.s\n" ++
            "fmul z23.s, p0/m, z23.s, z4.s\n" ++
            "fmla z20.s, p0/m, z16.s, z5.s\n" ++
            "fmla z21.s, p0/m, z17.s, z5.s\n" ++
            "fmla z22.s, p0/m, z18.s, z5.s\n" ++
            "fmla z23.s, p0/m, z19.s, z5.s\n" ++
            "st1w { z20.s - z23.s }, pn8, [x9]\n" ++
            "mov w8, #10\n" ++
            "mov { z16.s - z19.s }, za.s[w8, 0, vgx4]\n" ++
            "ld1w { z20.s - z23.s }, pn8/z, [x9, #4, mul vl]\n" ++
            "fmul z20.s, p0/m, z20.s, z4.s\n" ++
            "fmul z21.s, p0/m, z21.s, z4.s\n" ++
            "fmul z22.s, p0/m, z22.s, z4.s\n" ++
            "fmul z23.s, p0/m, z23.s, z4.s\n" ++
            "fmla z20.s, p0/m, z16.s, z5.s\n" ++
            "fmla z21.s, p0/m, z17.s, z5.s\n" ++
            "fmla z22.s, p0/m, z18.s, z5.s\n" ++
            "fmla z23.s, p0/m, z19.s, z5.s\n" ++
            "st1w { z20.s - z23.s }, pn8, [x9, #4, mul vl]\n" ++
            "mov w8, #3\n" ++
            "mov { z16.s - z19.s }, za.s[w8, 0, vgx4]\n" ++
            "ld1w { z20.s - z23.s }, pn8/z, [x9, #8, mul vl]\n" ++
            "fmul z20.s, p0/m, z20.s, z4.s\n" ++
            "fmul z21.s, p0/m, z21.s, z4.s\n" ++
            "fmul z22.s, p0/m, z22.s, z4.s\n" ++
            "fmul z23.s, p0/m, z23.s, z4.s\n" ++
            "fmla z20.s, p0/m, z16.s, z5.s\n" ++
            "fmla z21.s, p0/m, z17.s, z5.s\n" ++
            "fmla z22.s, p0/m, z18.s, z5.s\n" ++
            "fmla z23.s, p0/m, z19.s, z5.s\n" ++
            "st1w { z20.s - z23.s }, pn8, [x9, #8, mul vl]\n" ++
            "mov w8, #11\n" ++
            "mov { z16.s - z19.s }, za.s[w8, 0, vgx4]\n" ++
            "ld1w { z20.s - z23.s }, pn8/z, [x9, #12, mul vl]\n" ++
            "fmul z20.s, p0/m, z20.s, z4.s\n" ++
            "fmul z21.s, p0/m, z21.s, z4.s\n" ++
            "fmul z22.s, p0/m, z22.s, z4.s\n" ++
            "fmul z23.s, p0/m, z23.s, z4.s\n" ++
            "fmla z20.s, p0/m, z16.s, z5.s\n" ++
            "fmla z21.s, p0/m, z17.s, z5.s\n" ++
            "fmla z22.s, p0/m, z18.s, z5.s\n" ++
            "fmla z23.s, p0/m, z19.s, z5.s\n" ++
            "st1w { z20.s - z23.s }, pn8, [x9, #12, mul vl]\n" ++
            "12:\n" ++
            "ret\n" ::: .{ .memory = true });
}

pub noinline fn sme2DgemvNF64128x1(m: usize, n: usize, alpha: f64, beta: f64, a: [*]const f64, lda_bytes: usize, x: [*]const f64, y: [*]f64) callconv(.naked) void {
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
        \\zero { za }
        \\mov x15, x2
        \\mov x7, x5
        \\mov x16, x4
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
        \\add x15, x15, x3
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
        \\add x15, x15, x3
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

pub noinline fn sme2SgemvTF3216x64(m: usize, n: usize, alpha: f32, beta: f32, a: [*]const f32, lda_bytes: usize, x: [*]const f32, y: [*]f32) callconv(.naked) void {
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
    ++ asm_fragments.ptrue_pn8_s ++
        \\
        \\mov x7, x2              // Current 16-column A panel.
        \\mov x15, x1             // Remaining output columns.
        \\lsl x6, x3, #4          // lda_bytes * 16.
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
        \\ld1w { z4.s - z7.s }, pn8/z, [x10]
        \\mov w8, #0
        \\mov w11, #8
        \\
        \\ld1w { z16.s, z20.s, z24.s, z28.s }, pn8/z, [x9]
        \\add x9, x9, x3
        \\ld1w { z17.s, z21.s, z25.s, z29.s }, pn8/z, [x9]
        \\add x9, x9, x3
        \\ld1w { z18.s, z22.s, z26.s, z30.s }, pn8/z, [x9]
        \\add x9, x9, x3
        \\ld1w { z19.s, z23.s, z27.s, z31.s }, pn8/z, [x9]
        \\add x9, x9, x3
        \\fmla za.s[w8, 0, vgx4], { z16.s - z19.s }, z4.s
        \\fmla za.s[w11, 0, vgx4], { z20.s - z23.s }, z5.s
        \\fmla za.s[w8, 1, vgx4], { z24.s - z27.s }, z6.s
        \\fmla za.s[w11, 1, vgx4], { z28.s - z31.s }, z7.s
        \\
        \\ld1w { z16.s, z20.s, z24.s, z28.s }, pn8/z, [x9]
        \\add x9, x9, x3
        \\ld1w { z17.s, z21.s, z25.s, z29.s }, pn8/z, [x9]
        \\add x9, x9, x3
        \\ld1w { z18.s, z22.s, z26.s, z30.s }, pn8/z, [x9]
        \\add x9, x9, x3
        \\ld1w { z19.s, z23.s, z27.s, z31.s }, pn8/z, [x9]
        \\add x9, x9, x3
        \\fmla za.s[w8, 2, vgx4], { z16.s - z19.s }, z4.s
        \\fmla za.s[w11, 2, vgx4], { z20.s - z23.s }, z5.s
        \\fmla za.s[w8, 3, vgx4], { z24.s - z27.s }, z6.s
        \\fmla za.s[w11, 3, vgx4], { z28.s - z31.s }, z7.s
        \\
        \\ld1w { z16.s, z20.s, z24.s, z28.s }, pn8/z, [x9]
        \\add x9, x9, x3
        \\ld1w { z17.s, z21.s, z25.s, z29.s }, pn8/z, [x9]
        \\add x9, x9, x3
        \\ld1w { z18.s, z22.s, z26.s, z30.s }, pn8/z, [x9]
        \\add x9, x9, x3
        \\ld1w { z19.s, z23.s, z27.s, z31.s }, pn8/z, [x9]
        \\add x9, x9, x3
        \\fmla za.s[w8, 4, vgx4], { z16.s - z19.s }, z4.s
        \\fmla za.s[w11, 4, vgx4], { z20.s - z23.s }, z5.s
        \\fmla za.s[w8, 5, vgx4], { z24.s - z27.s }, z6.s
        \\fmla za.s[w11, 5, vgx4], { z28.s - z31.s }, z7.s
        \\
        \\ld1w { z16.s, z20.s, z24.s, z28.s }, pn8/z, [x9]
        \\add x9, x9, x3
        \\ld1w { z17.s, z21.s, z25.s, z29.s }, pn8/z, [x9]
        \\add x9, x9, x3
        \\ld1w { z18.s, z22.s, z26.s, z30.s }, pn8/z, [x9]
        \\add x9, x9, x3
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
        \\adr x13, 14f
        \\ldr z4, [x13]
        \\tbl z20.s, { z20.s }, z4.s
        \\
        \\ptrue p0.s
        \\ld1w { z16.s }, p0/z, [x5]
        \\mov z17.s, s1
        \\fmul z16.s, z16.s, z17.s
        \\mov z17.s, s0
        \\fmla z16.s, p0/m, z20.s, z17.s
        \\st1w { z16.s }, p0, [x5]
        \\
        \\add x5, x5, #64
        \\add x7, x7, x6
        \\subs x15, x15, #16
        \\b.ne 10b
        \\
        \\13:
        \\ret
        \\
        \\14:
        \\.word 0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15
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
        \\
        \\13:
        \\whilelo p1.d, x8, x0
        \\b.none 12f
        \\ld1d { z0.d }, p1/z, [x1, x8, lsl #3]
        \\fmul z0.d, p1/m, z0.d, z4.d
        \\st1d { z0.d }, p1, [x1, x8, lsl #3]
        \\incd x8
        \\b 13b
        \\
        \\12:
        \\ret
    ::: .{ .memory = true });
}

pub noinline fn sveDgerF64N8(m: usize, n: usize, alpha: f64, x: [*]const f64, y: [*]const f64, a: [*]f64, lda_bytes: usize) callconv(.naked) void {
    _ = m;
    _ = n;
    _ = alpha;
    _ = x;
    _ = y;
    _ = a;
    _ = lda_bytes;
    asm volatile (
        \\cbz x0, 18f
        \\cbz x1, 18f
    ++ asm_fragments.ptrue_p0_d ++
        \\cntd x7
        \\
        \\10:
        \\mov x8, x5
        \\add x9, x8, x6
        \\add x10, x9, x6
        \\add x11, x10, x6
        \\add x12, x11, x6
        \\add x13, x12, x6
        \\add x14, x13, x6
        \\add x15, x14, x6
        \\
        \\ldr d1, [x4]
        \\ldr d2, [x4, #8]
        \\ldr d3, [x4, #16]
        \\ldr d5, [x4, #24]
        \\ldr d6, [x4, #32]
        \\ldr d7, [x4, #40]
        \\ldr d24, [x4, #48]
        \\ldr d25, [x4, #56]
        \\fmul d1, d0, d1
        \\fmul d2, d0, d2
        \\fmul d3, d0, d3
        \\fmul d5, d0, d5
        \\fmul d6, d0, d6
        \\fmul d7, d0, d7
        \\fmul d24, d0, d24
        \\fmul d25, d0, d25
        \\mov z1.d, d1
        \\mov z2.d, d2
        \\mov z3.d, d3
        \\mov z5.d, d5
        \\mov z6.d, d6
        \\mov z7.d, d7
        \\mov z24.d, d24
        \\mov z25.d, d25
        \\
        \\mov x16, #0
        \\11:
        \\sub x17, x0, x16
        \\cmp x17, x7
        \\b.lo 12f
        \\ld1d { z4.d }, p0/z, [x3, x16, lsl #3]
        \\ld1d { z16.d }, p0/z, [x8, x16, lsl #3]
        \\ld1d { z17.d }, p0/z, [x9, x16, lsl #3]
        \\ld1d { z18.d }, p0/z, [x10, x16, lsl #3]
        \\ld1d { z19.d }, p0/z, [x11, x16, lsl #3]
        \\ld1d { z20.d }, p0/z, [x12, x16, lsl #3]
        \\ld1d { z21.d }, p0/z, [x13, x16, lsl #3]
        \\ld1d { z22.d }, p0/z, [x14, x16, lsl #3]
        \\ld1d { z23.d }, p0/z, [x15, x16, lsl #3]
        \\fmla z16.d, p0/m, z4.d, z1.d
        \\fmla z17.d, p0/m, z4.d, z2.d
        \\fmla z18.d, p0/m, z4.d, z3.d
        \\fmla z19.d, p0/m, z4.d, z5.d
        \\fmla z20.d, p0/m, z4.d, z6.d
        \\fmla z21.d, p0/m, z4.d, z7.d
        \\fmla z22.d, p0/m, z4.d, z24.d
        \\fmla z23.d, p0/m, z4.d, z25.d
        \\st1d { z16.d }, p0, [x8, x16, lsl #3]
        \\st1d { z17.d }, p0, [x9, x16, lsl #3]
        \\st1d { z18.d }, p0, [x10, x16, lsl #3]
        \\st1d { z19.d }, p0, [x11, x16, lsl #3]
        \\st1d { z20.d }, p0, [x12, x16, lsl #3]
        \\st1d { z21.d }, p0, [x13, x16, lsl #3]
        \\st1d { z22.d }, p0, [x14, x16, lsl #3]
        \\st1d { z23.d }, p0, [x15, x16, lsl #3]
        \\incd x16
        \\b 11b
        \\
        \\12:
        \\whilelo p1.d, x16, x0
        \\b.none 13f
        \\ld1d { z4.d }, p1/z, [x3, x16, lsl #3]
        \\ld1d { z16.d }, p1/z, [x8, x16, lsl #3]
        \\ld1d { z17.d }, p1/z, [x9, x16, lsl #3]
        \\ld1d { z18.d }, p1/z, [x10, x16, lsl #3]
        \\ld1d { z19.d }, p1/z, [x11, x16, lsl #3]
        \\ld1d { z20.d }, p1/z, [x12, x16, lsl #3]
        \\ld1d { z21.d }, p1/z, [x13, x16, lsl #3]
        \\ld1d { z22.d }, p1/z, [x14, x16, lsl #3]
        \\ld1d { z23.d }, p1/z, [x15, x16, lsl #3]
        \\fmla z16.d, p1/m, z4.d, z1.d
        \\fmla z17.d, p1/m, z4.d, z2.d
        \\fmla z18.d, p1/m, z4.d, z3.d
        \\fmla z19.d, p1/m, z4.d, z5.d
        \\fmla z20.d, p1/m, z4.d, z6.d
        \\fmla z21.d, p1/m, z4.d, z7.d
        \\fmla z22.d, p1/m, z4.d, z24.d
        \\fmla z23.d, p1/m, z4.d, z25.d
        \\st1d { z16.d }, p1, [x8, x16, lsl #3]
        \\st1d { z17.d }, p1, [x9, x16, lsl #3]
        \\st1d { z18.d }, p1, [x10, x16, lsl #3]
        \\st1d { z19.d }, p1, [x11, x16, lsl #3]
        \\st1d { z20.d }, p1, [x12, x16, lsl #3]
        \\st1d { z21.d }, p1, [x13, x16, lsl #3]
        \\st1d { z22.d }, p1, [x14, x16, lsl #3]
        \\st1d { z23.d }, p1, [x15, x16, lsl #3]
        \\
        \\13:
        \\add x5, x15, x6
        \\add x4, x4, #64
        \\subs x1, x1, #8
        \\b.ne 10b
        \\
        \\18:
        \\ret
    ::: .{ .memory = true });
}

pub noinline fn asimdDgerF64x4(m: usize, n: usize, alpha: f64, x: [*]const f64, y: [*]const f64, a: [*]f64, lda_bytes: usize) callconv(.naked) void {
    _ = m;
    _ = n;
    _ = alpha;
    _ = x;
    _ = y;
    _ = a;
    _ = lda_bytes;
    asm volatile (
        \\cbz x0, 12f
        \\cbz x1, 12f
        \\lsl x17, x0, #3
        \\
        \\10:
        \\mov x8, x4
        \\add x9, x8, x5
        \\add x10, x9, x5
        \\add x11, x10, x5
        \\
        \\ldr d1, [x3]
        \\ldr d2, [x3, #8]
        \\ldr d3, [x3, #16]
        \\ldr d4, [x3, #24]
        \\fmul d1, d0, d1
        \\fmul d2, d0, d2
        \\fmul d3, d0, d3
        \\fmul d4, d0, d4
        \\
        \\mov x16, #0
        \\11:
        \\ldr q17, [x2, x16]
        \\ldr q18, [x8, x16]
        \\ldr q19, [x9, x16]
        \\ldr q20, [x10, x16]
        \\ldr q21, [x11, x16]
        \\fmla.2d v18, v17, v1[0]
        \\fmla.2d v19, v17, v2[0]
        \\fmla.2d v20, v17, v3[0]
        \\fmla.2d v21, v17, v4[0]
        \\str q18, [x8, x16]
        \\str q19, [x9, x16]
        \\str q20, [x10, x16]
        \\str q21, [x11, x16]
        \\add x16, x16, #16
        \\cmp x16, x17
        \\b.lo 11b
        \\
        \\add x4, x11, x5
        \\add x3, x3, #32
        \\subs x1, x1, #4
        \\b.ne 10b
        \\
        \\12:
        \\ret
        ::: .{ .memory = true });
}

pub noinline fn sveDcopyF64(n: usize, x: [*]const f64, y: [*]f64) callconv(.naked) void {
    _ = n;
    _ = x;
    _ = y;
    asm volatile (
        \\cbz x0, 12f
    ++ asm_fragments.ptrue_p0_d ++
        \\cntd x6
        \\lsl x7, x6, #4
        \\
        \\10:
        \\cmp x0, x7
        \\b.lo 11f
        \\ld1d { z0.d }, p0/z, [x1]
        \\ld1d { z1.d }, p0/z, [x1, #1, MUL VL]
        \\ld1d { z2.d }, p0/z, [x1, #2, MUL VL]
        \\ld1d { z3.d }, p0/z, [x1, #3, MUL VL]
        \\ld1d { z4.d }, p0/z, [x1, #4, MUL VL]
        \\ld1d { z5.d }, p0/z, [x1, #5, MUL VL]
        \\ld1d { z6.d }, p0/z, [x1, #6, MUL VL]
        \\ld1d { z7.d }, p0/z, [x1, #7, MUL VL]
        \\ld1d { z8.d }, p0/z, [x1, #8, MUL VL]
        \\ld1d { z9.d }, p0/z, [x1, #9, MUL VL]
        \\ld1d { z10.d }, p0/z, [x1, #10, MUL VL]
        \\ld1d { z11.d }, p0/z, [x1, #11, MUL VL]
        \\ld1d { z12.d }, p0/z, [x1, #12, MUL VL]
        \\ld1d { z13.d }, p0/z, [x1, #13, MUL VL]
        \\ld1d { z14.d }, p0/z, [x1, #14, MUL VL]
        \\ld1d { z15.d }, p0/z, [x1, #15, MUL VL]
        \\st1d { z0.d }, p0, [x2]
        \\st1d { z1.d }, p0, [x2, #1, MUL VL]
        \\st1d { z2.d }, p0, [x2, #2, MUL VL]
        \\st1d { z3.d }, p0, [x2, #3, MUL VL]
        \\st1d { z4.d }, p0, [x2, #4, MUL VL]
        \\st1d { z5.d }, p0, [x2, #5, MUL VL]
        \\st1d { z6.d }, p0, [x2, #6, MUL VL]
        \\st1d { z7.d }, p0, [x2, #7, MUL VL]
        \\st1d { z8.d }, p0, [x2, #8, MUL VL]
        \\st1d { z9.d }, p0, [x2, #9, MUL VL]
        \\st1d { z10.d }, p0, [x2, #10, MUL VL]
        \\st1d { z11.d }, p0, [x2, #11, MUL VL]
        \\st1d { z12.d }, p0, [x2, #12, MUL VL]
        \\st1d { z13.d }, p0, [x2, #13, MUL VL]
        \\st1d { z14.d }, p0, [x2, #14, MUL VL]
        \\st1d { z15.d }, p0, [x2, #15, MUL VL]
        \\addvl x1, x1, #16
        \\addvl x2, x2, #16
        \\sub x0, x0, x7
        \\b 10b
        \\
        \\11:
        \\cbz x0, 12f
        \\mov x8, #0
        \\
        \\13:
        \\whilelo p1.d, x8, x0
        \\b.none 12f
        \\ld1d { z0.d }, p1/z, [x1, x8, lsl #3]
        \\st1d { z0.d }, p1, [x2, x8, lsl #3]
        \\incd x8
        \\b 13b
        \\
        \\12:
        \\ret
    ::: .{ .memory = true });
}

pub noinline fn sveDasumF64Bits(n: usize, x: [*]const f64) callconv(.naked) u64 {
    _ = n;
    _ = x;
    asm volatile (
        \\cbz x0, 13f
    ++ asm_fragments.ptrue_p0_d ++
        \\cntd x6
        \\lsl x7, x6, #4
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
        \\
        \\10:
        \\cmp x0, x7
        \\b.lo 11f
        \\ld1d { z16.d }, p0/z, [x1]
        \\ld1d { z17.d }, p0/z, [x1, #1, MUL VL]
        \\ld1d { z18.d }, p0/z, [x1, #2, MUL VL]
        \\ld1d { z19.d }, p0/z, [x1, #3, MUL VL]
        \\ld1d { z20.d }, p0/z, [x1, #4, MUL VL]
        \\ld1d { z21.d }, p0/z, [x1, #5, MUL VL]
        \\ld1d { z22.d }, p0/z, [x1, #6, MUL VL]
        \\ld1d { z23.d }, p0/z, [x1, #7, MUL VL]
        \\ld1d { z24.d }, p0/z, [x1, #8, MUL VL]
        \\ld1d { z25.d }, p0/z, [x1, #9, MUL VL]
        \\ld1d { z26.d }, p0/z, [x1, #10, MUL VL]
        \\ld1d { z27.d }, p0/z, [x1, #11, MUL VL]
        \\ld1d { z28.d }, p0/z, [x1, #12, MUL VL]
        \\ld1d { z29.d }, p0/z, [x1, #13, MUL VL]
        \\ld1d { z30.d }, p0/z, [x1, #14, MUL VL]
        \\ld1d { z31.d }, p0/z, [x1, #15, MUL VL]
        \\fabs z16.d, p0/m, z16.d
        \\fabs z17.d, p0/m, z17.d
        \\fabs z18.d, p0/m, z18.d
        \\fabs z19.d, p0/m, z19.d
        \\fabs z20.d, p0/m, z20.d
        \\fabs z21.d, p0/m, z21.d
        \\fabs z22.d, p0/m, z22.d
        \\fabs z23.d, p0/m, z23.d
        \\fabs z24.d, p0/m, z24.d
        \\fabs z25.d, p0/m, z25.d
        \\fabs z26.d, p0/m, z26.d
        \\fabs z27.d, p0/m, z27.d
        \\fabs z28.d, p0/m, z28.d
        \\fabs z29.d, p0/m, z29.d
        \\fabs z30.d, p0/m, z30.d
        \\fabs z31.d, p0/m, z31.d
        \\fadd z0.d, z0.d, z16.d
        \\fadd z1.d, z1.d, z17.d
        \\fadd z2.d, z2.d, z18.d
        \\fadd z3.d, z3.d, z19.d
        \\fadd z4.d, z4.d, z20.d
        \\fadd z5.d, z5.d, z21.d
        \\fadd z6.d, z6.d, z22.d
        \\fadd z7.d, z7.d, z23.d
        \\fadd z8.d, z8.d, z24.d
        \\fadd z9.d, z9.d, z25.d
        \\fadd z10.d, z10.d, z26.d
        \\fadd z11.d, z11.d, z27.d
        \\fadd z12.d, z12.d, z28.d
        \\fadd z13.d, z13.d, z29.d
        \\fadd z14.d, z14.d, z30.d
        \\fadd z15.d, z15.d, z31.d
        \\addvl x1, x1, #16
        \\sub x0, x0, x7
        \\b 10b
        \\
        \\11:
        \\cbz x0, 12f
        \\mov x8, #0
        \\
        \\14:
        \\whilelo p1.d, x8, x0
        \\b.none 12f
        \\ld1d { z16.d }, p1/z, [x1, x8, lsl #3]
        \\fabs z16.d, p1/m, z16.d
        \\fadd z0.d, z0.d, z16.d
        \\incd x8
        \\b 14b
        \\
        \\12:
        \\fadd z0.d, z0.d, z1.d
        \\fadd z2.d, z2.d, z3.d
        \\fadd z4.d, z4.d, z5.d
        \\fadd z6.d, z6.d, z7.d
        \\fadd z8.d, z8.d, z9.d
        \\fadd z10.d, z10.d, z11.d
        \\fadd z12.d, z12.d, z13.d
        \\fadd z14.d, z14.d, z15.d
        \\fadd z0.d, z0.d, z2.d
        \\fadd z4.d, z4.d, z6.d
        \\fadd z8.d, z8.d, z10.d
        \\fadd z12.d, z12.d, z14.d
        \\fadd z0.d, z0.d, z4.d
        \\fadd z8.d, z8.d, z12.d
        \\fadd z0.d, z0.d, z8.d
        \\faddv d16, p0, z0.d
        \\fmov x0, d16
        \\ret
        \\
        \\13:
        \\mov x0, xzr
        \\ret
    ::: .{ .memory = true });
}

pub noinline fn sveDdotF64Bits(n: usize, x: [*]const f64, y: [*]const f64) callconv(.naked) u64 {
    _ = n;
    _ = x;
    _ = y;
    asm volatile (
        \\cbz x0, 13f
    ++ asm_fragments.ptrue_p0_d ++
        \\cntd x6
        \\lsl x7, x6, #2
        \\dup z0.d, #0
        \\dup z1.d, #0
        \\dup z2.d, #0
        \\dup z3.d, #0
        \\
        \\10:
        \\cmp x0, x7
        \\b.lo 11f
        \\ld1d { z4.d }, p0/z, [x1]
        \\ld1d { z5.d }, p0/z, [x1, #1, MUL VL]
        \\ld1d { z6.d }, p0/z, [x1, #2, MUL VL]
        \\ld1d { z7.d }, p0/z, [x1, #3, MUL VL]
        \\ld1d { z16.d }, p0/z, [x2]
        \\ld1d { z17.d }, p0/z, [x2, #1, MUL VL]
        \\ld1d { z18.d }, p0/z, [x2, #2, MUL VL]
        \\ld1d { z19.d }, p0/z, [x2, #3, MUL VL]
        \\fmla z0.d, p0/m, z4.d, z16.d
        \\fmla z1.d, p0/m, z5.d, z17.d
        \\fmla z2.d, p0/m, z6.d, z18.d
        \\fmla z3.d, p0/m, z7.d, z19.d
        \\addvl x1, x1, #4
        \\addvl x2, x2, #4
        \\sub x0, x0, x7
        \\b 10b
        \\
        \\11:
        \\cbz x0, 12f
        \\mov x8, #0
        \\
        \\14:
        \\whilelo p1.d, x8, x0
        \\b.none 12f
        \\ld1d { z4.d }, p1/z, [x1, x8, lsl #3]
        \\ld1d { z16.d }, p1/z, [x2, x8, lsl #3]
        \\fmla z0.d, p1/m, z4.d, z16.d
        \\incd x8
        \\b 14b
        \\
        \\12:
        \\fadd z0.d, z0.d, z1.d
        \\fadd z2.d, z2.d, z3.d
        \\fadd z0.d, z0.d, z2.d
        \\faddv d16, p0, z0.d
        \\fmov x0, d16
        \\ret
        \\
        \\13:
        \\mov x0, xzr
        \\ret
    ::: .{ .memory = true });
}
