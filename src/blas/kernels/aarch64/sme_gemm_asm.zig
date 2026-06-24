// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later

//! SME GEMM microkernels expressed as Zig-owned whole-function inline asm.

const asm_fragments = @import("asm_fragments.zig");

pub noinline fn sgemmPanelF32(a: [*]const f32, b_pack: [*]const f32, c: [*]f32, m_full: usize, k: usize, lda_bytes: usize, ldc_bytes: usize) callconv(.naked) void {
    _ = a;
    _ = b_pack;
    _ = c;
    _ = m_full;
    _ = k;
    _ = lda_bytes;
    _ = ldc_bytes;
    asm volatile (asm_fragments.read_svl_x14 ++ asm_fragments.ptrue_p0_s ++
            \\
            \\lsl x3, x3, #2
            \\mov x7, #0
            \\1:
            \\cmp x7, x3
            \\b.hs 5f
            \\zero { za0.s }
            \\
            \\add x10, x0, x7
            \\mov x11, x1
            \\mov x9, x4
            \\2:
            \\cbz x9, 3f
            \\ld1w { z0.s }, p0/z, [x10]
            \\ld1w { z1.s }, p0/z, [x11]
            \\fmopa za0.s, p0/m, p0/m, z0.s, z1.s
            \\add x10, x10, x5
            \\add x11, x11, x14
            \\sub x9, x9, #1
            \\b 2b
            \\
            \\3:
            \\add x10, x2, x7
            \\mov x11, x6
            \\mov w12, #0
            \\lsr x13, x14, #2
            \\4:
            \\cbz x13, 6f
            \\st1w { za0v.s[w12, 0] }, p0, [x10]
            \\add x10, x10, x11
            \\add w12, w12, #1
            \\sub x13, x13, #1
            \\b 4b
            \\
            \\6:
            \\add x7, x7, x14
            \\b 1b
            \\
            \\5:
            \\ret
        ::: .{ .memory = true });
}

pub noinline fn sgemmPanel4mF32(a: [*]const f32, b_pack: [*]const f32, c: [*]f32, m_full: usize, k: usize, lda_bytes: usize, ldc_bytes: usize) callconv(.naked) void {
    _ = a;
    _ = b_pack;
    _ = c;
    _ = m_full;
    _ = k;
    _ = lda_bytes;
    _ = ldc_bytes;
    asm volatile (asm_fragments.read_svl_x14 ++ asm_fragments.ptrue_p0_s ++
            \\
            \\lsl x3, x3, #2
            \\mov x7, #0
            \\13:
            \\cmp x7, x3
            \\b.hs 17f
            \\zero { za0.s, za1.s, za2.s, za3.s }
            \\
            \\add x10, x0, x7
            \\mov x11, x1
            \\mov x9, x4
            \\14:
            \\cbz x9, 15f
            \\ld1w { z0.s }, p0/z, [x10]
            \\add x13, x10, x14
            \\ld1w { z1.s }, p0/z, [x13]
            \\add x13, x10, x14, lsl #1
            \\ld1w { z2.s }, p0/z, [x13]
            \\add x13, x10, x14
            \\add x13, x13, x14, lsl #1
            \\ld1w { z3.s }, p0/z, [x13]
            \\ld1w { z4.s }, p0/z, [x11]
            \\fmopa za0.s, p0/m, p0/m, z0.s, z4.s
            \\fmopa za1.s, p0/m, p0/m, z1.s, z4.s
            \\fmopa za2.s, p0/m, p0/m, z2.s, z4.s
            \\fmopa za3.s, p0/m, p0/m, z3.s, z4.s
            \\add x10, x10, x5
            \\add x11, x11, x14
            \\sub x9, x9, #1
            \\b 14b
            \\
            \\15:
            \\add x10, x2, x7
            \\mov w12, #0
            \\lsr x13, x14, #2
            \\16:
            \\cbz x13, 18f
            \\st1w { za0v.s[w12, 0] }, p0, [x10]
            \\add x16, x10, x14
            \\st1w { za1v.s[w12, 0] }, p0, [x16]
            \\add x16, x10, x14, lsl #1
            \\st1w { za2v.s[w12, 0] }, p0, [x16]
            \\add x16, x10, x14
            \\add x16, x16, x14, lsl #1
            \\st1w { za3v.s[w12, 0] }, p0, [x16]
            \\add x10, x10, x6
            \\add w12, w12, #1
            \\sub x13, x13, #1
            \\b 16b
            \\
            \\18:
            \\add x7, x7, x14, lsl #2
            \\b 13b
            \\
            \\17:
            \\ret
        ::: .{ .memory = true });
}

pub noinline fn sgemmPanel2x2F32(a: [*]const f32, b_pack: [*]const f32, c: [*]f32, m_full: usize, k: usize, lda_bytes: usize, ldc_bytes: usize) callconv(.naked) void {
    _ = a;
    _ = b_pack;
    _ = c;
    _ = m_full;
    _ = k;
    _ = lda_bytes;
    _ = ldc_bytes;
    asm volatile (asm_fragments.read_svl_x14 ++ asm_fragments.ptrue_p0_s ++
            \\
            \\lsr x15, x14, #2
            \\mul x15, x15, x6
            \\
            \\lsl x3, x3, #2
            \\mov x7, #0
            \\31:
            \\cmp x7, x3
            \\b.hs 35f
            \\zero { za0.s, za1.s, za2.s, za3.s }
            \\
            \\add x10, x0, x7
            \\mov x11, x1
            \\mov x9, x4
            \\32:
            \\cmp x9, #2                            // Need two K steps for the unrolled body.
            \\b.lo 43f                              // Drop to the one-K tail when fewer remain.
            \\ld1w { z0.s }, p0/z, [x10]            // A panel 0: A[i : i+tile, p].
            \\add x13, x10, x14                     // Address A panel 1: one tile row below.
            \\ld1w { z1.s }, p0/z, [x13]            // A panel 1: A[i+tile : i+2*tile, p].
            \\ld1w { z2.s }, p0/z, [x11]            // B panel 0: packed B[p, j : j+tile].
            \\add x13, x11, x14                     // Address B panel 1 inside the 2N pack.
            \\ld1w { z3.s }, p0/z, [x13]            // B panel 1: packed B[p, j+tile : j+2*tile].
            \\fmopa za0.s, p0/m, p0/m, z0.s, z2.s  // C00 += A0 * B0.
            \\fmopa za1.s, p0/m, p0/m, z1.s, z2.s  // C10 += A1 * B0.
            \\fmopa za2.s, p0/m, p0/m, z0.s, z3.s  // C01 += A0 * B1.
            \\fmopa za3.s, p0/m, p0/m, z1.s, z3.s  // C11 += A1 * B1.
            \\add x10, x10, x5                      // Advance A to K p+1 using lda bytes.
            \\add x11, x11, x14, lsl #1             // Advance packed B by 2 tile columns.
            \\ld1w { z0.s }, p0/z, [x10]            // A panel 0 for K p+1.
            \\add x13, x10, x14                     // Address A panel 1 for K p+1.
            \\ld1w { z1.s }, p0/z, [x13]            // A panel 1 for K p+1.
            \\ld1w { z2.s }, p0/z, [x11]            // B panel 0 for K p+1.
            \\add x13, x11, x14                     // Address B panel 1 for K p+1.
            \\ld1w { z3.s }, p0/z, [x13]            // B panel 1 for K p+1.
            \\fmopa za0.s, p0/m, p0/m, z0.s, z2.s  // C00 += A0 * B0 for K p+1.
            \\fmopa za1.s, p0/m, p0/m, z1.s, z2.s  // C10 += A1 * B0 for K p+1.
            \\fmopa za2.s, p0/m, p0/m, z0.s, z3.s  // C01 += A0 * B1 for K p+1.
            \\fmopa za3.s, p0/m, p0/m, z1.s, z3.s  // C11 += A1 * B1 for K p+1.
            \\add x10, x10, x5                      // Advance A to the next unrolled pair.
            \\add x11, x11, x14, lsl #1             // Advance packed B to the next K pair.
            \\sub x9, x9, #2                        // Two K positions consumed.
            \\b 32b                                 // Continue until fewer than two remain.
            \\
            \\43:
            \\cbz x9, 33f
            \\ld1w { z0.s }, p0/z, [x10]
            \\add x13, x10, x14
            \\ld1w { z1.s }, p0/z, [x13]
            \\ld1w { z2.s }, p0/z, [x11]
            \\add x13, x11, x14
            \\ld1w { z3.s }, p0/z, [x13]
            \\fmopa za0.s, p0/m, p0/m, z0.s, z2.s
            \\fmopa za1.s, p0/m, p0/m, z1.s, z2.s
            \\fmopa za2.s, p0/m, p0/m, z0.s, z3.s
            \\fmopa za3.s, p0/m, p0/m, z1.s, z3.s
            \\
            \\33:
            \\add x10, x2, x7
            \\mov w12, #0
            \\lsr x13, x14, #2
            \\34:
            \\cbz x13, 36f                          // All ZA rows stored.
            \\st1w { za0v.s[w12, 0] }, p0, [x10]    // Store C00 row into column j.
            \\add x16, x10, x14                     // Address C10 row: same column, +tile rows.
            \\st1w { za1v.s[w12, 0] }, p0, [x16]    // Store C10.
            \\add x16, x10, x15                     // Address C01 row: +tile columns in C.
            \\st1w { za2v.s[w12, 0] }, p0, [x16]    // Store C01.
            \\add x16, x16, x14                     // Address C11 row: +tile rows from C01.
            \\st1w { za3v.s[w12, 0] }, p0, [x16]    // Store C11.
            \\add x10, x10, x6                      // Advance to next C row using ldc bytes.
            \\add w12, w12, #1                      // Select next ZA row vector.
            \\sub x13, x13, #1                      // One row vector stored.
            \\b 34b                                 // Continue across tile rows.
            \\
            \\36:
            \\add x7, x7, x14, lsl #1
            \\b 31b
            \\
            \\35:
            \\ret
        ::: .{ .memory = true });
}

pub noinline fn sgemmPanel1x2F32(a: [*]const f32, b_pack: [*]const f32, c: [*]f32, m_full: usize, k: usize, lda_bytes: usize, ldc_bytes: usize) callconv(.naked) void {
    _ = a;
    _ = b_pack;
    _ = c;
    _ = m_full;
    _ = k;
    _ = lda_bytes;
    _ = ldc_bytes;
    asm volatile (asm_fragments.read_svl_x14 ++ asm_fragments.ptrue_p0_s ++
            \\
            \\lsr x15, x14, #2
            \\mul x15, x15, x6
            \\
            \\lsl x3, x3, #2
            \\mov x7, #0
            \\83:
            \\cmp x7, x3
            \\b.hs 87f
            \\zero { za0.s, za1.s }
            \\
            \\add x10, x0, x7
            \\mov x11, x1
            \\mov x9, x4
            \\84:
            \\cmp x9, #2
            \\b.lo 88f
            \\ld1w { z0.s }, p0/z, [x10]
            \\ld1w { z2.s }, p0/z, [x11]
            \\add x13, x11, x14
            \\ld1w { z3.s }, p0/z, [x13]
            \\fmopa za0.s, p0/m, p0/m, z0.s, z2.s
            \\fmopa za1.s, p0/m, p0/m, z0.s, z3.s
            \\add x10, x10, x5
            \\add x11, x11, x14, lsl #1
            \\ld1w { z0.s }, p0/z, [x10]
            \\ld1w { z2.s }, p0/z, [x11]
            \\add x13, x11, x14
            \\ld1w { z3.s }, p0/z, [x13]
            \\fmopa za0.s, p0/m, p0/m, z0.s, z2.s
            \\fmopa za1.s, p0/m, p0/m, z0.s, z3.s
            \\add x10, x10, x5
            \\add x11, x11, x14, lsl #1
            \\sub x9, x9, #2
            \\b 84b
            \\
            \\88:
            \\cbz x9, 85f
            \\ld1w { z0.s }, p0/z, [x10]
            \\ld1w { z2.s }, p0/z, [x11]
            \\add x13, x11, x14
            \\ld1w { z3.s }, p0/z, [x13]
            \\fmopa za0.s, p0/m, p0/m, z0.s, z2.s
            \\fmopa za1.s, p0/m, p0/m, z0.s, z3.s
            \\
            \\85:
            \\add x10, x2, x7
            \\mov w12, #0
            \\lsr x13, x14, #2
            \\86:
            \\cbz x13, 89f
            \\st1w { za0v.s[w12, 0] }, p0, [x10]
            \\add x16, x10, x15
            \\st1w { za1v.s[w12, 0] }, p0, [x16]
            \\add x10, x10, x6
            \\add w12, w12, #1
            \\sub x13, x13, #1
            \\b 86b
            \\
            \\89:
            \\add x7, x7, x14
            \\b 83b
            \\
            \\87:
            \\ret
        ::: .{ .memory = true });
}

pub noinline fn sgemmPanels2x2F32(a: [*]const f32, b_pack: [*]const f32, c: [*]f32, m_full: usize, k: usize, lda_bytes: usize, ldc_bytes: usize, panel_count: usize) callconv(.naked) void {
    _ = a;
    _ = b_pack;
    _ = c;
    _ = m_full;
    _ = k;
    _ = lda_bytes;
    _ = ldc_bytes;
    _ = panel_count;
    asm volatile (
        \\stp x19, x20, [sp, #-16]!
    ++ asm_fragments.read_svl_x14 ++
        \\lsr x15, x14, #2
        \\mul x15, x15, x6
        \\lsl x17, x14, #1
        \\mul x17, x17, x4
        \\lsl x8, x15, #1
        \\mov x19, x7
        \\lsl x20, x3, #2
        \\
    ++ asm_fragments.ptrue_p0_s ++
        \\
        \\73:
        \\cbz x19, 81f
        \\mov x7, #0
        \\74:
        \\cmp x7, x20
        \\b.hs 80f
        \\zero { za0.s, za1.s, za2.s, za3.s }
        \\
        \\add x10, x0, x7
        \\mov x11, x1
        \\mov x9, x4
        \\75:
        \\cmp x9, #2
        \\b.lo 82f
        \\ld1w { z0.s }, p0/z, [x10]
        \\add x13, x10, x14
        \\ld1w { z1.s }, p0/z, [x13]
        \\ld1w { z2.s }, p0/z, [x11]
        \\add x13, x11, x14
        \\ld1w { z3.s }, p0/z, [x13]
        \\fmopa za0.s, p0/m, p0/m, z0.s, z2.s
        \\fmopa za1.s, p0/m, p0/m, z1.s, z2.s
        \\fmopa za2.s, p0/m, p0/m, z0.s, z3.s
        \\fmopa za3.s, p0/m, p0/m, z1.s, z3.s
        \\add x10, x10, x5
        \\add x11, x11, x14, lsl #1
        \\ld1w { z0.s }, p0/z, [x10]
        \\add x13, x10, x14
        \\ld1w { z1.s }, p0/z, [x13]
        \\ld1w { z2.s }, p0/z, [x11]
        \\add x13, x11, x14
        \\ld1w { z3.s }, p0/z, [x13]
        \\fmopa za0.s, p0/m, p0/m, z0.s, z2.s
        \\fmopa za1.s, p0/m, p0/m, z1.s, z2.s
        \\fmopa za2.s, p0/m, p0/m, z0.s, z3.s
        \\fmopa za3.s, p0/m, p0/m, z1.s, z3.s
        \\add x10, x10, x5
        \\add x11, x11, x14, lsl #1
        \\sub x9, x9, #2
        \\b 75b
        \\
        \\82:
        \\cbz x9, 76f
        \\ld1w { z0.s }, p0/z, [x10]
        \\add x13, x10, x14
        \\ld1w { z1.s }, p0/z, [x13]
        \\ld1w { z2.s }, p0/z, [x11]
        \\add x13, x11, x14
        \\ld1w { z3.s }, p0/z, [x13]
        \\fmopa za0.s, p0/m, p0/m, z0.s, z2.s
        \\fmopa za1.s, p0/m, p0/m, z1.s, z2.s
        \\fmopa za2.s, p0/m, p0/m, z0.s, z3.s
        \\fmopa za3.s, p0/m, p0/m, z1.s, z3.s
        \\
        \\76:
        \\add x10, x2, x7
        \\mov w12, #0
        \\lsr x13, x14, #2
        \\77:
        \\cbz x13, 79f
        \\st1w { za0v.s[w12, 0] }, p0, [x10]
        \\add x16, x10, x14
        \\st1w { za1v.s[w12, 0] }, p0, [x16]
        \\add x16, x10, x15
        \\st1w { za2v.s[w12, 0] }, p0, [x16]
        \\add x16, x16, x14
        \\st1w { za3v.s[w12, 0] }, p0, [x16]
        \\add x10, x10, x6
        \\add w12, w12, #1
        \\sub x13, x13, #1
        \\b 77b
        \\
        \\79:
        \\add x7, x7, x14, lsl #1
        \\b 74b
        \\
        \\80:
        \\add x1, x1, x17
        \\add x2, x2, x8
        \\sub x19, x19, #1
        \\b 73b
        \\
        \\81:
        \\ldp x19, x20, [sp], #16
        \\ret
    ::: .{ .memory = true });
}

pub noinline fn sgemmPanels2x2U4F32(a: [*]const f32, b_pack: [*]const f32, c: [*]f32, m_full: usize, k: usize, lda_bytes: usize, ldc_bytes: usize, panel_count: usize) callconv(.naked) void {
    _ = a;
    _ = b_pack;
    _ = c;
    _ = m_full;
    _ = k;
    _ = lda_bytes;
    _ = ldc_bytes;
    _ = panel_count;
    asm volatile (
        \\stp x19, x20, [sp, #-16]!
    ++ asm_fragments.read_svl_x14 ++
        \\lsr x15, x14, #2
        \\mul x15, x15, x6
        \\lsl x17, x14, #1
        \\mul x17, x17, x4
        \\lsl x8, x15, #1
        \\mov x19, x7
        \\lsl x20, x3, #2
        \\
    ++ asm_fragments.ptrue_p0_s ++
        \\
        \\90:
        \\cbz x19, 99f
        \\mov x7, #0
        \\91:
        \\cmp x7, x20
        \\b.hs 98f
        \\zero { za0.s, za1.s, za2.s, za3.s }
        \\
        \\add x10, x0, x7
        \\mov x11, x1
        \\mov x9, x4
        \\92:
        \\cmp x9, #4
        \\b.lo 93f
        \\ld1w { z0.s }, p0/z, [x10]
        \\add x13, x10, x14
        \\ld1w { z1.s }, p0/z, [x13]
        \\ld1w { z2.s }, p0/z, [x11]
        \\add x13, x11, x14
        \\ld1w { z3.s }, p0/z, [x13]
        \\fmopa za0.s, p0/m, p0/m, z0.s, z2.s
        \\fmopa za1.s, p0/m, p0/m, z1.s, z2.s
        \\fmopa za2.s, p0/m, p0/m, z0.s, z3.s
        \\fmopa za3.s, p0/m, p0/m, z1.s, z3.s
        \\add x10, x10, x5
        \\add x11, x11, x14, lsl #1
        \\ld1w { z0.s }, p0/z, [x10]
        \\add x13, x10, x14
        \\ld1w { z1.s }, p0/z, [x13]
        \\ld1w { z2.s }, p0/z, [x11]
        \\add x13, x11, x14
        \\ld1w { z3.s }, p0/z, [x13]
        \\fmopa za0.s, p0/m, p0/m, z0.s, z2.s
        \\fmopa za1.s, p0/m, p0/m, z1.s, z2.s
        \\fmopa za2.s, p0/m, p0/m, z0.s, z3.s
        \\fmopa za3.s, p0/m, p0/m, z1.s, z3.s
        \\add x10, x10, x5
        \\add x11, x11, x14, lsl #1
        \\ld1w { z0.s }, p0/z, [x10]
        \\add x13, x10, x14
        \\ld1w { z1.s }, p0/z, [x13]
        \\ld1w { z2.s }, p0/z, [x11]
        \\add x13, x11, x14
        \\ld1w { z3.s }, p0/z, [x13]
        \\fmopa za0.s, p0/m, p0/m, z0.s, z2.s
        \\fmopa za1.s, p0/m, p0/m, z1.s, z2.s
        \\fmopa za2.s, p0/m, p0/m, z0.s, z3.s
        \\fmopa za3.s, p0/m, p0/m, z1.s, z3.s
        \\add x10, x10, x5
        \\add x11, x11, x14, lsl #1
        \\ld1w { z0.s }, p0/z, [x10]
        \\add x13, x10, x14
        \\ld1w { z1.s }, p0/z, [x13]
        \\ld1w { z2.s }, p0/z, [x11]
        \\add x13, x11, x14
        \\ld1w { z3.s }, p0/z, [x13]
        \\fmopa za0.s, p0/m, p0/m, z0.s, z2.s
        \\fmopa za1.s, p0/m, p0/m, z1.s, z2.s
        \\fmopa za2.s, p0/m, p0/m, z0.s, z3.s
        \\fmopa za3.s, p0/m, p0/m, z1.s, z3.s
        \\add x10, x10, x5
        \\add x11, x11, x14, lsl #1
        \\sub x9, x9, #4
        \\b 92b
        \\
        \\93:
        \\cmp x9, #2
        \\b.lo 94f
        \\ld1w { z0.s }, p0/z, [x10]
        \\add x13, x10, x14
        \\ld1w { z1.s }, p0/z, [x13]
        \\ld1w { z2.s }, p0/z, [x11]
        \\add x13, x11, x14
        \\ld1w { z3.s }, p0/z, [x13]
        \\fmopa za0.s, p0/m, p0/m, z0.s, z2.s
        \\fmopa za1.s, p0/m, p0/m, z1.s, z2.s
        \\fmopa za2.s, p0/m, p0/m, z0.s, z3.s
        \\fmopa za3.s, p0/m, p0/m, z1.s, z3.s
        \\add x10, x10, x5
        \\add x11, x11, x14, lsl #1
        \\ld1w { z0.s }, p0/z, [x10]
        \\add x13, x10, x14
        \\ld1w { z1.s }, p0/z, [x13]
        \\ld1w { z2.s }, p0/z, [x11]
        \\add x13, x11, x14
        \\ld1w { z3.s }, p0/z, [x13]
        \\fmopa za0.s, p0/m, p0/m, z0.s, z2.s
        \\fmopa za1.s, p0/m, p0/m, z1.s, z2.s
        \\fmopa za2.s, p0/m, p0/m, z0.s, z3.s
        \\fmopa za3.s, p0/m, p0/m, z1.s, z3.s
        \\add x10, x10, x5
        \\add x11, x11, x14, lsl #1
        \\sub x9, x9, #2
        \\94:
        \\cbz x9, 95f
        \\ld1w { z0.s }, p0/z, [x10]
        \\add x13, x10, x14
        \\ld1w { z1.s }, p0/z, [x13]
        \\ld1w { z2.s }, p0/z, [x11]
        \\add x13, x11, x14
        \\ld1w { z3.s }, p0/z, [x13]
        \\fmopa za0.s, p0/m, p0/m, z0.s, z2.s
        \\fmopa za1.s, p0/m, p0/m, z1.s, z2.s
        \\fmopa za2.s, p0/m, p0/m, z0.s, z3.s
        \\fmopa za3.s, p0/m, p0/m, z1.s, z3.s
        \\
        \\95:
        \\add x10, x2, x7
        \\mov w12, #0
        \\lsr x13, x14, #2
        \\96:
        \\cbz x13, 97f
        \\st1w { za0v.s[w12, 0] }, p0, [x10]
        \\add x16, x10, x14
        \\st1w { za1v.s[w12, 0] }, p0, [x16]
        \\add x16, x10, x15
        \\st1w { za2v.s[w12, 0] }, p0, [x16]
        \\add x16, x16, x14
        \\st1w { za3v.s[w12, 0] }, p0, [x16]
        \\add x10, x10, x6
        \\add w12, w12, #1
        \\sub x13, x13, #1
        \\b 96b
        \\
        \\97:
        \\add x7, x7, x14, lsl #1
        \\b 91b
        \\
        \\98:
        \\add x1, x1, x17
        \\add x2, x2, x8
        \\sub x19, x19, #1
        \\b 90b
        \\
        \\99:
        \\ldp x19, x20, [sp], #16
        \\ret
    ::: .{ .memory = true });
}

pub noinline fn dgemmPanelF64(a: [*]const f64, b_pack: [*]const f64, c: [*]f64, m_full: usize, k: usize, lda_bytes: usize, ldc_bytes: usize) callconv(.naked) void {
    _ = a;
    _ = b_pack;
    _ = c;
    _ = m_full;
    _ = k;
    _ = lda_bytes;
    _ = ldc_bytes;
    asm volatile (asm_fragments.read_svl_x14 ++ asm_fragments.ptrue_p0_d ++
            \\
            \\lsl x3, x3, #3
            \\mov x7, #0
            \\19:
            \\cmp x7, x3
            \\b.hs 23f
            \\zero { za0.d }
            \\
            \\add x10, x0, x7
            \\mov x11, x1
            \\mov x9, x4
            \\20:
            \\cbz x9, 21f
            \\ld1d { z0.d }, p0/z, [x10]
            \\ld1d { z1.d }, p0/z, [x11]
            \\fmopa za0.d, p0/m, p0/m, z0.d, z1.d
            \\add x10, x10, x5
            \\add x11, x11, x14
            \\sub x9, x9, #1
            \\b 20b
            \\
            \\21:
            \\add x10, x2, x7
            \\mov w12, #0
            \\lsr x13, x14, #3
            \\22:
            \\cbz x13, 24f
            \\st1d { za0v.d[w12, 0] }, p0, [x10]
            \\add x10, x10, x6
            \\add w12, w12, #1
            \\sub x13, x13, #1
            \\b 22b
            \\
            \\24:
            \\add x7, x7, x14
            \\b 19b
            \\
            \\23:
            \\ret
        ::: .{ .memory = true });
}

pub noinline fn dgemmPanel4mF64(a: [*]const f64, b_pack: [*]const f64, c: [*]f64, m_full: usize, k: usize, lda_bytes: usize, ldc_bytes: usize) callconv(.naked) void {
    _ = a;
    _ = b_pack;
    _ = c;
    _ = m_full;
    _ = k;
    _ = lda_bytes;
    _ = ldc_bytes;
    asm volatile (asm_fragments.read_svl_x14 ++ asm_fragments.ptrue_p0_d ++
            \\
            \\lsl x3, x3, #3
            \\mov x7, #0
            \\25:
            \\cmp x7, x3
            \\b.hs 29f
            \\zero { za0.d, za1.d, za2.d, za3.d }
            \\
            \\add x10, x0, x7
            \\mov x11, x1
            \\mov x9, x4
            \\26:
            \\cbz x9, 27f
            \\ld1d { z0.d }, p0/z, [x10]
            \\add x13, x10, x14
            \\ld1d { z1.d }, p0/z, [x13]
            \\add x13, x10, x14, lsl #1
            \\ld1d { z2.d }, p0/z, [x13]
            \\add x13, x10, x14
            \\add x13, x13, x14, lsl #1
            \\ld1d { z3.d }, p0/z, [x13]
            \\ld1d { z4.d }, p0/z, [x11]
            \\fmopa za0.d, p0/m, p0/m, z0.d, z4.d
            \\fmopa za1.d, p0/m, p0/m, z1.d, z4.d
            \\fmopa za2.d, p0/m, p0/m, z2.d, z4.d
            \\fmopa za3.d, p0/m, p0/m, z3.d, z4.d
            \\add x10, x10, x5
            \\add x11, x11, x14
            \\sub x9, x9, #1
            \\b 26b
            \\
            \\27:
            \\add x10, x2, x7
            \\mov w12, #0
            \\lsr x13, x14, #3
            \\28:
            \\cbz x13, 30f
            \\st1d { za0v.d[w12, 0] }, p0, [x10]
            \\add x16, x10, x14
            \\st1d { za1v.d[w12, 0] }, p0, [x16]
            \\add x16, x10, x14, lsl #1
            \\st1d { za2v.d[w12, 0] }, p0, [x16]
            \\add x16, x10, x14
            \\add x16, x16, x14, lsl #1
            \\st1d { za3v.d[w12, 0] }, p0, [x16]
            \\add x10, x10, x6
            \\add w12, w12, #1
            \\sub x13, x13, #1
            \\b 28b
            \\
            \\30:
            \\add x7, x7, x14, lsl #2
            \\b 25b
            \\
            \\29:
            \\ret
        ::: .{ .memory = true });
}

pub noinline fn dgemmPanel2x2F64(a: [*]const f64, b_pack: [*]const f64, c: [*]f64, m_full: usize, k: usize, lda_bytes: usize, ldc_bytes: usize) callconv(.naked) void {
    _ = a;
    _ = b_pack;
    _ = c;
    _ = m_full;
    _ = k;
    _ = lda_bytes;
    _ = ldc_bytes;
    asm volatile (asm_fragments.read_svl_x14 ++ asm_fragments.ptrue_p0_d ++
            \\
            \\lsr x15, x14, #3
            \\mul x15, x15, x6
            \\
            \\lsl x3, x3, #3
            \\mov x7, #0
            \\37:
            \\cmp x7, x3
            \\b.hs 41f
            \\zero { za0.d, za1.d, za2.d, za3.d }
            \\
            \\add x10, x0, x7
            \\mov x11, x1
            \\mov x9, x4
            \\38:
            \\cmp x9, #2
            \\b.lo 44f
            \\ld1d { z0.d }, p0/z, [x10]
            \\add x13, x10, x14
            \\ld1d { z1.d }, p0/z, [x13]
            \\ld1d { z2.d }, p0/z, [x11]
            \\add x13, x11, x14
            \\ld1d { z3.d }, p0/z, [x13]
            \\fmopa za0.d, p0/m, p0/m, z0.d, z2.d
            \\fmopa za1.d, p0/m, p0/m, z1.d, z2.d
            \\fmopa za2.d, p0/m, p0/m, z0.d, z3.d
            \\fmopa za3.d, p0/m, p0/m, z1.d, z3.d
            \\add x10, x10, x5
            \\add x11, x11, x14, lsl #1
            \\ld1d { z0.d }, p0/z, [x10]
            \\add x13, x10, x14
            \\ld1d { z1.d }, p0/z, [x13]
            \\ld1d { z2.d }, p0/z, [x11]
            \\add x13, x11, x14
            \\ld1d { z3.d }, p0/z, [x13]
            \\fmopa za0.d, p0/m, p0/m, z0.d, z2.d
            \\fmopa za1.d, p0/m, p0/m, z1.d, z2.d
            \\fmopa za2.d, p0/m, p0/m, z0.d, z3.d
            \\fmopa za3.d, p0/m, p0/m, z1.d, z3.d
            \\add x10, x10, x5
            \\add x11, x11, x14, lsl #1
            \\sub x9, x9, #2
            \\b 38b
            \\
            \\44:
            \\cbz x9, 39f
            \\ld1d { z0.d }, p0/z, [x10]
            \\add x13, x10, x14
            \\ld1d { z1.d }, p0/z, [x13]
            \\ld1d { z2.d }, p0/z, [x11]
            \\add x13, x11, x14
            \\ld1d { z3.d }, p0/z, [x13]
            \\fmopa za0.d, p0/m, p0/m, z0.d, z2.d
            \\fmopa za1.d, p0/m, p0/m, z1.d, z2.d
            \\fmopa za2.d, p0/m, p0/m, z0.d, z3.d
            \\fmopa za3.d, p0/m, p0/m, z1.d, z3.d
            \\
            \\39:
            \\add x10, x2, x7
            \\mov w12, #0
            \\lsr x13, x14, #3
            \\40:
            \\cbz x13, 42f
            \\st1d { za0v.d[w12, 0] }, p0, [x10]
            \\add x16, x10, x14
            \\st1d { za1v.d[w12, 0] }, p0, [x16]
            \\add x16, x10, x15
            \\st1d { za2v.d[w12, 0] }, p0, [x16]
            \\add x16, x16, x14
            \\st1d { za3v.d[w12, 0] }, p0, [x16]
            \\add x10, x10, x6
            \\add w12, w12, #1
            \\sub x13, x13, #1
            \\b 40b
            \\
            \\42:
            \\add x7, x7, x14, lsl #1
            \\b 37b
            \\
            \\41:
            \\ret
        ::: .{ .memory = true });
}

pub noinline fn dgemmPanel4x2F64(a: [*]const f64, b_pack: [*]const f64, c: [*]f64, m_full: usize, k: usize, lda_bytes: usize, ldc_bytes: usize) callconv(.naked) void {
    _ = a;
    _ = b_pack;
    _ = c;
    _ = m_full;
    _ = k;
    _ = lda_bytes;
    _ = ldc_bytes;
    asm volatile (asm_fragments.read_svl_x14 ++ asm_fragments.ptrue_p0_d ++
            \\
            \\lsr x15, x14, #3
            \\mul x15, x15, x6
            \\
            \\lsl x3, x3, #3
            \\mov x7, #0
            \\45:
            \\cmp x7, x3
            \\b.hs 49f
            \\zero { za0.d, za1.d, za2.d, za3.d, za4.d, za5.d, za6.d, za7.d }
            \\
            \\add x10, x0, x7
            \\mov x11, x1
            \\mov x9, x4
            \\46:
            \\cmp x9, #2                            // Need two K steps for the unrolled body.
            \\b.lo 51f                              // Drop to the one-K tail when fewer remain.
            \\prfm pldl1keep, [x11, #512]           // Pull future packed B into L1 for high-K f64.
            \\ld1d { z0.d }, p0/z, [x10]            // A panel 0: A[i : i+tile, p].
            \\add x13, x10, x14                     // Address A panel 1.
            \\ld1d { z1.d }, p0/z, [x13]            // A panel 1: +1 tile row.
            \\add x13, x10, x14, lsl #1             // Address A panel 2.
            \\ld1d { z2.d }, p0/z, [x13]            // A panel 2: +2 tile rows.
            \\add x13, x10, x14                     // Rebuild +1 tile row address.
            \\add x13, x13, x14, lsl #1             // Address A panel 3: +3 tile rows.
            \\ld1d { z3.d }, p0/z, [x13]            // A panel 3.
            \\ld1d { z4.d }, p0/z, [x11]            // B panel 0: packed B[p, j : j+tile].
            \\add x13, x11, x14                     // Address B panel 1 inside the 2N pack.
            \\ld1d { z5.d }, p0/z, [x13]            // B panel 1: packed B[p, j+tile : j+2*tile].
            \\fmopa za0.d, p0/m, p0/m, z0.d, z4.d  // C00 += A0 * B0.
            \\fmopa za1.d, p0/m, p0/m, z1.d, z4.d  // C10 += A1 * B0.
            \\fmopa za2.d, p0/m, p0/m, z2.d, z4.d  // C20 += A2 * B0.
            \\fmopa za3.d, p0/m, p0/m, z3.d, z4.d  // C30 += A3 * B0.
            \\fmopa za4.d, p0/m, p0/m, z0.d, z5.d  // C01 += A0 * B1.
            \\fmopa za5.d, p0/m, p0/m, z1.d, z5.d  // C11 += A1 * B1.
            \\fmopa za6.d, p0/m, p0/m, z2.d, z5.d  // C21 += A2 * B1.
            \\fmopa za7.d, p0/m, p0/m, z3.d, z5.d  // C31 += A3 * B1.
            \\add x10, x10, x5                      // Advance A to K p+1 using lda bytes.
            \\add x11, x11, x14, lsl #1             // Advance packed B by 2 tile columns.
            \\ld1d { z0.d }, p0/z, [x10]            // A panel 0 for K p+1.
            \\add x13, x10, x14                     // Address A panel 1 for K p+1.
            \\ld1d { z1.d }, p0/z, [x13]            // A panel 1 for K p+1.
            \\add x13, x10, x14, lsl #1             // Address A panel 2 for K p+1.
            \\ld1d { z2.d }, p0/z, [x13]            // A panel 2 for K p+1.
            \\add x13, x10, x14                     // Rebuild +1 tile row address.
            \\add x13, x13, x14, lsl #1             // Address A panel 3 for K p+1.
            \\ld1d { z3.d }, p0/z, [x13]            // A panel 3 for K p+1.
            \\ld1d { z4.d }, p0/z, [x11]            // B panel 0 for K p+1.
            \\add x13, x11, x14                     // Address B panel 1 for K p+1.
            \\ld1d { z5.d }, p0/z, [x13]            // B panel 1 for K p+1.
            \\fmopa za0.d, p0/m, p0/m, z0.d, z4.d  // C00 += A0 * B0 for K p+1.
            \\fmopa za1.d, p0/m, p0/m, z1.d, z4.d  // C10 += A1 * B0 for K p+1.
            \\fmopa za2.d, p0/m, p0/m, z2.d, z4.d  // C20 += A2 * B0 for K p+1.
            \\fmopa za3.d, p0/m, p0/m, z3.d, z4.d  // C30 += A3 * B0 for K p+1.
            \\fmopa za4.d, p0/m, p0/m, z0.d, z5.d  // C01 += A0 * B1 for K p+1.
            \\fmopa za5.d, p0/m, p0/m, z1.d, z5.d  // C11 += A1 * B1 for K p+1.
            \\fmopa za6.d, p0/m, p0/m, z2.d, z5.d  // C21 += A2 * B1 for K p+1.
            \\fmopa za7.d, p0/m, p0/m, z3.d, z5.d  // C31 += A3 * B1 for K p+1.
            \\add x10, x10, x5                      // Advance A to the next unrolled pair.
            \\add x11, x11, x14, lsl #1             // Advance packed B to the next K pair.
            \\sub x9, x9, #2                        // Two K positions consumed.
            \\b 46b                                 // Continue until fewer than two remain.
            \\
            \\51:
            \\cbz x9, 47f
            \\prfm pldl1keep, [x11, #512]
            \\ld1d { z0.d }, p0/z, [x10]
            \\add x13, x10, x14
            \\ld1d { z1.d }, p0/z, [x13]
            \\add x13, x10, x14, lsl #1
            \\ld1d { z2.d }, p0/z, [x13]
            \\add x13, x10, x14
            \\add x13, x13, x14, lsl #1
            \\ld1d { z3.d }, p0/z, [x13]
            \\ld1d { z4.d }, p0/z, [x11]
            \\add x13, x11, x14
            \\ld1d { z5.d }, p0/z, [x13]
            \\fmopa za0.d, p0/m, p0/m, z0.d, z4.d
            \\fmopa za1.d, p0/m, p0/m, z1.d, z4.d
            \\fmopa za2.d, p0/m, p0/m, z2.d, z4.d
            \\fmopa za3.d, p0/m, p0/m, z3.d, z4.d
            \\fmopa za4.d, p0/m, p0/m, z0.d, z5.d
            \\fmopa za5.d, p0/m, p0/m, z1.d, z5.d
            \\fmopa za6.d, p0/m, p0/m, z2.d, z5.d
            \\fmopa za7.d, p0/m, p0/m, z3.d, z5.d
            \\
            \\47:
            \\add x10, x2, x7
            \\mov w12, #0
            \\lsr x13, x14, #3
            \\48:
            \\cbz x13, 50f                          // All ZA rows stored.
            \\st1d { za0v.d[w12, 0] }, p0, [x10]    // Store C00 row into column j.
            \\add x16, x10, x14                     // Address C10: +1 tile row.
            \\st1d { za1v.d[w12, 0] }, p0, [x16]    // Store C10.
            \\add x16, x10, x14, lsl #1             // Address C20: +2 tile rows.
            \\st1d { za2v.d[w12, 0] }, p0, [x16]    // Store C20.
            \\add x16, x10, x14                     // Rebuild +1 tile row address.
            \\add x16, x16, x14, lsl #1             // Address C30: +3 tile rows.
            \\st1d { za3v.d[w12, 0] }, p0, [x16]    // Store C30.
            \\add x16, x10, x15                     // Address C01: +tile columns.
            \\st1d { za4v.d[w12, 0] }, p0, [x16]    // Store C01.
            \\add x16, x16, x14                     // Address C11.
            \\st1d { za5v.d[w12, 0] }, p0, [x16]    // Store C11.
            \\add x16, x10, x15                     // Rebuild C01 base.
            \\add x16, x16, x14, lsl #1             // Address C21.
            \\st1d { za6v.d[w12, 0] }, p0, [x16]    // Store C21.
            \\add x16, x10, x15                     // Rebuild C01 base.
            \\add x16, x16, x14                     // Add one tile row.
            \\add x16, x16, x14, lsl #1             // Address C31.
            \\st1d { za7v.d[w12, 0] }, p0, [x16]    // Store C31.
            \\add x10, x10, x6                      // Advance to next C row using ldc bytes.
            \\add w12, w12, #1                      // Select next ZA row vector.
            \\sub x13, x13, #1                      // One row vector stored.
            \\b 48b                                 // Continue across tile rows.
            \\
            \\50:
            \\add x7, x7, x14, lsl #2
            \\b 45b
            \\
            \\49:
            \\ret
        ::: .{ .memory = true });
}

pub noinline fn dgemmPanels4x2F64(a: [*]const f64, b_pack: [*]const f64, c: [*]f64, m_full: usize, k: usize, lda_bytes: usize, ldc_bytes: usize, panel_count: usize) callconv(.naked) void {
    _ = a;
    _ = b_pack;
    _ = c;
    _ = m_full;
    _ = k;
    _ = lda_bytes;
    _ = ldc_bytes;
    _ = panel_count;
    asm volatile (
        \\stp x19, x20, [sp, #-16]!
    ++ asm_fragments.read_svl_x14 ++
        \\lsr x15, x14, #3
        \\mul x15, x15, x6
        \\lsl x17, x14, #1
        \\mul x17, x17, x4
        \\lsl x8, x15, #1
        \\mov x19, x7
        \\lsl x20, x3, #3
        \\
    ++ asm_fragments.ptrue_p0_d ++
        \\
        \\64:
        \\cbz x19, 72f
        \\mov x7, #0
        \\65:
        \\cmp x7, x20
        \\b.hs 71f
        \\zero { za0.d, za1.d, za2.d, za3.d, za4.d, za5.d, za6.d, za7.d }
        \\
        \\add x10, x0, x7
        \\mov x11, x1
        \\mov x9, x4
        \\66:
        \\cmp x9, #2
        \\b.lo 69f
        \\prfm pldl1keep, [x11, #512]
        \\ld1d { z0.d }, p0/z, [x10]
        \\add x13, x10, x14
        \\ld1d { z1.d }, p0/z, [x13]
        \\add x13, x10, x14, lsl #1
        \\ld1d { z2.d }, p0/z, [x13]
        \\add x13, x10, x14
        \\add x13, x13, x14, lsl #1
        \\ld1d { z3.d }, p0/z, [x13]
        \\ld1d { z4.d }, p0/z, [x11]
        \\add x13, x11, x14
        \\ld1d { z5.d }, p0/z, [x13]
        \\fmopa za0.d, p0/m, p0/m, z0.d, z4.d
        \\fmopa za1.d, p0/m, p0/m, z1.d, z4.d
        \\fmopa za2.d, p0/m, p0/m, z2.d, z4.d
        \\fmopa za3.d, p0/m, p0/m, z3.d, z4.d
        \\fmopa za4.d, p0/m, p0/m, z0.d, z5.d
        \\fmopa za5.d, p0/m, p0/m, z1.d, z5.d
        \\fmopa za6.d, p0/m, p0/m, z2.d, z5.d
        \\fmopa za7.d, p0/m, p0/m, z3.d, z5.d
        \\add x10, x10, x5
        \\add x11, x11, x14, lsl #1
        \\ld1d { z0.d }, p0/z, [x10]
        \\add x13, x10, x14
        \\ld1d { z1.d }, p0/z, [x13]
        \\add x13, x10, x14, lsl #1
        \\ld1d { z2.d }, p0/z, [x13]
        \\add x13, x10, x14
        \\add x13, x13, x14, lsl #1
        \\ld1d { z3.d }, p0/z, [x13]
        \\ld1d { z4.d }, p0/z, [x11]
        \\add x13, x11, x14
        \\ld1d { z5.d }, p0/z, [x13]
        \\fmopa za0.d, p0/m, p0/m, z0.d, z4.d
        \\fmopa za1.d, p0/m, p0/m, z1.d, z4.d
        \\fmopa za2.d, p0/m, p0/m, z2.d, z4.d
        \\fmopa za3.d, p0/m, p0/m, z3.d, z4.d
        \\fmopa za4.d, p0/m, p0/m, z0.d, z5.d
        \\fmopa za5.d, p0/m, p0/m, z1.d, z5.d
        \\fmopa za6.d, p0/m, p0/m, z2.d, z5.d
        \\fmopa za7.d, p0/m, p0/m, z3.d, z5.d
        \\add x10, x10, x5
        \\add x11, x11, x14, lsl #1
        \\sub x9, x9, #2
        \\b 66b
        \\
        \\69:
        \\cbz x9, 67f
        \\prfm pldl1keep, [x11, #512]
        \\ld1d { z0.d }, p0/z, [x10]
        \\add x13, x10, x14
        \\ld1d { z1.d }, p0/z, [x13]
        \\add x13, x10, x14, lsl #1
        \\ld1d { z2.d }, p0/z, [x13]
        \\add x13, x10, x14
        \\add x13, x13, x14, lsl #1
        \\ld1d { z3.d }, p0/z, [x13]
        \\ld1d { z4.d }, p0/z, [x11]
        \\add x13, x11, x14
        \\ld1d { z5.d }, p0/z, [x13]
        \\fmopa za0.d, p0/m, p0/m, z0.d, z4.d
        \\fmopa za1.d, p0/m, p0/m, z1.d, z4.d
        \\fmopa za2.d, p0/m, p0/m, z2.d, z4.d
        \\fmopa za3.d, p0/m, p0/m, z3.d, z4.d
        \\fmopa za4.d, p0/m, p0/m, z0.d, z5.d
        \\fmopa za5.d, p0/m, p0/m, z1.d, z5.d
        \\fmopa za6.d, p0/m, p0/m, z2.d, z5.d
        \\fmopa za7.d, p0/m, p0/m, z3.d, z5.d
        \\
        \\67:
        \\add x10, x2, x7
        \\mov w12, #0
        \\lsr x13, x14, #3
        \\68:
        \\cbz x13, 70f
        \\st1d { za0v.d[w12, 0] }, p0, [x10]
        \\add x16, x10, x14
        \\st1d { za1v.d[w12, 0] }, p0, [x16]
        \\add x16, x10, x14, lsl #1
        \\st1d { za2v.d[w12, 0] }, p0, [x16]
        \\add x16, x10, x14
        \\add x16, x16, x14, lsl #1
        \\st1d { za3v.d[w12, 0] }, p0, [x16]
        \\add x16, x10, x15
        \\st1d { za4v.d[w12, 0] }, p0, [x16]
        \\add x16, x16, x14
        \\st1d { za5v.d[w12, 0] }, p0, [x16]
        \\add x16, x10, x15
        \\add x16, x16, x14, lsl #1
        \\st1d { za6v.d[w12, 0] }, p0, [x16]
        \\add x16, x10, x15
        \\add x16, x16, x14
        \\add x16, x16, x14, lsl #1
        \\st1d { za7v.d[w12, 0] }, p0, [x16]
        \\add x10, x10, x6
        \\add w12, w12, #1
        \\sub x13, x13, #1
        \\b 68b
        \\
        \\70:
        \\add x7, x7, x14, lsl #2
        \\b 65b
        \\
        \\71:
        \\add x1, x1, x17
        \\add x2, x2, x8
        \\sub x19, x19, #1
        \\b 64b
        \\
        \\72:
        \\ldp x19, x20, [sp], #16
        \\ret
    ::: .{ .memory = true });
}
