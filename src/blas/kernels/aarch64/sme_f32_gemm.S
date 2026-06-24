// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later

.text
// This file only needs base SME.  It does not use SME2-only forms; SME2
// feature detection stays in Zig so the same symbols can run on base-SME
// machines that support f32 FMOPA.
.arch_extension sme

#ifdef __APPLE__
#define ZYNUM_SYMBOL(name) _##name
#else
#define ZYNUM_SYMBOL(name) name
#endif

// Common ABI for the f32 SME kernels:
//   x0=A, x1=packed B, x2=C, x3=m_full rows, x4=K,
//   x5=lda bytes, x6=ldc bytes, x7=batch panel count when present.
// x14 is the streaming vector length in bytes.  One f32 tile has
// x14/4 rows/columns; C is column-major, so stores walk ZA rows.
// These are body symbols: the Zig SME worker wrapper owns SM/ZA entry,
// callee-save preservation, and SM/ZA exit.  Callers must enter streaming
// mode and enable ZA before using any symbol in this file.

.globl ZYNUM_SYMBOL(zynum_blas_sme_sgemm_panel_f32)
.p2align 2
ZYNUM_SYMBOL(zynum_blas_sme_sgemm_panel_f32):
    rdsvl x14, #1         // x14 = SVL bytes; all tile strides derive from it.
    ptrue p0.s            // All f32 lanes active; full tiles only in assembly.

    lsl x3, x3, #2
    mov x7, #0
1:
    cmp x7, x3
    b.hs 5f
    zero { za0.s }

    add x10, x0, x7
    mov x11, x1
    mov x9, x4
2:
    cbz x9, 3f
    ld1w { z0.s }, p0/z, [x10]
    ld1w { z1.s }, p0/z, [x11]
    fmopa za0.s, p0/m, p0/m, z0.s, z1.s
    add x10, x10, x5
    add x11, x11, x14
    sub x9, x9, #1
    b 2b

3:
    add x10, x2, x7
    mov x11, x6
    mov w12, #0
    lsr x13, x14, #2
4:
    cbz x13, 6f
    st1w { za0v.s[w12, 0] }, p0, [x10]
    add x10, x10, x11
    add w12, w12, #1
    sub x13, x13, #1
    b 4b

6:
    add x7, x7, x14
    b 1b

5:
    ret

.globl ZYNUM_SYMBOL(zynum_blas_sme_sgemm_panel4m_f32)
.p2align 2
ZYNUM_SYMBOL(zynum_blas_sme_sgemm_panel4m_f32):
    rdsvl x14, #1         // x14 = SVL bytes; one f32 tile is x14/4 rows.
    ptrue p0.s            // Use all lanes; no partial-row predicate here.

    lsl x3, x3, #2
    mov x7, #0
13:
    cmp x7, x3
    b.hs 17f
    zero { za0.s, za1.s, za2.s, za3.s }

    add x10, x0, x7
    mov x11, x1
    mov x9, x4
14:
    cbz x9, 15f
    ld1w { z0.s }, p0/z, [x10]
    add x13, x10, x14
    ld1w { z1.s }, p0/z, [x13]
    add x13, x10, x14, lsl #1
    ld1w { z2.s }, p0/z, [x13]
    add x13, x10, x14
    add x13, x13, x14, lsl #1
    ld1w { z3.s }, p0/z, [x13]
    ld1w { z4.s }, p0/z, [x11]
    fmopa za0.s, p0/m, p0/m, z0.s, z4.s
    fmopa za1.s, p0/m, p0/m, z1.s, z4.s
    fmopa za2.s, p0/m, p0/m, z2.s, z4.s
    fmopa za3.s, p0/m, p0/m, z3.s, z4.s
    add x10, x10, x5
    add x11, x11, x14
    sub x9, x9, #1
    b 14b

15:
    add x10, x2, x7
    mov w12, #0
    lsr x13, x14, #2
16:
    cbz x13, 18f
    st1w { za0v.s[w12, 0] }, p0, [x10]
    add x16, x10, x14
    st1w { za1v.s[w12, 0] }, p0, [x16]
    add x16, x10, x14, lsl #1
    st1w { za2v.s[w12, 0] }, p0, [x16]
    add x16, x10, x14
    add x16, x16, x14, lsl #1
    st1w { za3v.s[w12, 0] }, p0, [x16]
    add x10, x10, x6
    add w12, w12, #1
    sub x13, x13, #1
    b 16b

18:
    add x7, x7, x14, lsl #2
    b 13b

17:
    ret

.globl ZYNUM_SYMBOL(zynum_blas_sme_sgemm_panel2x2_f32)
.p2align 2
ZYNUM_SYMBOL(zynum_blas_sme_sgemm_panel2x2_f32):
    rdsvl x14, #1         // x14 = SVL bytes, reused for A/B/C tile offsets.
    ptrue p0.s            // Full f32 tile predicate.

    lsr x15, x14, #2
    mul x15, x15, x6

    // 2M x 2N kernel.  za0/za1 accumulate the first N panel,
    // za2/za3 the second.  This symbol uses a K-unroll2 main loop; the
    // K-unroll4 variant is exposed as a separate symbol for the outer plan.
    lsl x3, x3, #2
    mov x7, #0
31:
    cmp x7, x3
    b.hs 35f
    zero { za0.s, za1.s, za2.s, za3.s }

    add x10, x0, x7
    mov x11, x1
    mov x9, x4
32:
    cmp x9, #2                            // Need two K steps for the unrolled body.
    b.lo 43f                              // Drop to the one-K tail when fewer remain.
    ld1w { z0.s }, p0/z, [x10]            // A panel 0: A[i : i+tile, p].
    add x13, x10, x14                     // Address A panel 1: one tile row below.
    ld1w { z1.s }, p0/z, [x13]            // A panel 1: A[i+tile : i+2*tile, p].
    ld1w { z2.s }, p0/z, [x11]            // B panel 0: packed B[p, j : j+tile].
    add x13, x11, x14                     // Address B panel 1 inside the 2N pack.
    ld1w { z3.s }, p0/z, [x13]            // B panel 1: packed B[p, j+tile : j+2*tile].
    fmopa za0.s, p0/m, p0/m, z0.s, z2.s  // C00 += A0 * B0.
    fmopa za1.s, p0/m, p0/m, z1.s, z2.s  // C10 += A1 * B0.
    fmopa za2.s, p0/m, p0/m, z0.s, z3.s  // C01 += A0 * B1.
    fmopa za3.s, p0/m, p0/m, z1.s, z3.s  // C11 += A1 * B1.
    add x10, x10, x5                      // Advance A to K p+1 using lda bytes.
    add x11, x11, x14, lsl #1             // Advance packed B by 2 tile columns.
    ld1w { z0.s }, p0/z, [x10]            // A panel 0 for K p+1.
    add x13, x10, x14                     // Address A panel 1 for K p+1.
    ld1w { z1.s }, p0/z, [x13]            // A panel 1 for K p+1.
    ld1w { z2.s }, p0/z, [x11]            // B panel 0 for K p+1.
    add x13, x11, x14                     // Address B panel 1 for K p+1.
    ld1w { z3.s }, p0/z, [x13]            // B panel 1 for K p+1.
    fmopa za0.s, p0/m, p0/m, z0.s, z2.s  // C00 += A0 * B0 for K p+1.
    fmopa za1.s, p0/m, p0/m, z1.s, z2.s  // C10 += A1 * B0 for K p+1.
    fmopa za2.s, p0/m, p0/m, z0.s, z3.s  // C01 += A0 * B1 for K p+1.
    fmopa za3.s, p0/m, p0/m, z1.s, z3.s  // C11 += A1 * B1 for K p+1.
    add x10, x10, x5                      // Advance A to the next unrolled pair.
    add x11, x11, x14, lsl #1             // Advance packed B to the next K pair.
    sub x9, x9, #2                        // Two K positions consumed.
    b 32b                                 // Continue until fewer than two remain.

43:
    cbz x9, 33f
    ld1w { z0.s }, p0/z, [x10]
    add x13, x10, x14
    ld1w { z1.s }, p0/z, [x13]
    ld1w { z2.s }, p0/z, [x11]
    add x13, x11, x14
    ld1w { z3.s }, p0/z, [x13]
    fmopa za0.s, p0/m, p0/m, z0.s, z2.s
    fmopa za1.s, p0/m, p0/m, z1.s, z2.s
    fmopa za2.s, p0/m, p0/m, z0.s, z3.s
    fmopa za3.s, p0/m, p0/m, z1.s, z3.s

33:
    // Store each ZA tile row into two column-major C panels.
    add x10, x2, x7
    mov w12, #0
    lsr x13, x14, #2
34:
    cbz x13, 36f                          // All ZA rows stored.
    st1w { za0v.s[w12, 0] }, p0, [x10]    // Store C00 row into column j.
    add x16, x10, x14                     // Address C10 row: same column, +tile rows.
    st1w { za1v.s[w12, 0] }, p0, [x16]    // Store C10.
    add x16, x10, x15                     // Address C01 row: +tile columns in C.
    st1w { za2v.s[w12, 0] }, p0, [x16]    // Store C01.
    add x16, x16, x14                     // Address C11 row: +tile rows from C01.
    st1w { za3v.s[w12, 0] }, p0, [x16]    // Store C11.
    add x10, x10, x6                      // Advance to next C row using ldc bytes.
    add w12, w12, #1                      // Select next ZA row vector.
    sub x13, x13, #1                      // One row vector stored.
    b 34b                                 // Continue across tile rows.

36:
    add x7, x7, x14, lsl #1
    b 31b

35:
    ret

.globl ZYNUM_SYMBOL(zynum_blas_sme_sgemm_panel1x2_f32)
.p2align 2
ZYNUM_SYMBOL(zynum_blas_sme_sgemm_panel1x2_f32):
    rdsvl x14, #1         // x14 = SVL bytes; one f32 tile is x14/4 rows.
    ptrue p0.s            // Full f32 tile predicate.

    lsr x15, x14, #2
    mul x15, x15, x6

    // Residual 1M x 2N kernel for shapes with one full f32 SME tile
    // left after the 2M path, e.g. M=16/48/80 on M-series SVL=64.
    lsl x3, x3, #2
    mov x7, #0
83:
    cmp x7, x3
    b.hs 87f
    zero { za0.s, za1.s }

    add x10, x0, x7
    mov x11, x1
    mov x9, x4
84:
    cmp x9, #2
    b.lo 88f
    ld1w { z0.s }, p0/z, [x10]
    ld1w { z2.s }, p0/z, [x11]
    add x13, x11, x14
    ld1w { z3.s }, p0/z, [x13]
    fmopa za0.s, p0/m, p0/m, z0.s, z2.s
    fmopa za1.s, p0/m, p0/m, z0.s, z3.s
    add x10, x10, x5
    add x11, x11, x14, lsl #1
    ld1w { z0.s }, p0/z, [x10]
    ld1w { z2.s }, p0/z, [x11]
    add x13, x11, x14
    ld1w { z3.s }, p0/z, [x13]
    fmopa za0.s, p0/m, p0/m, z0.s, z2.s
    fmopa za1.s, p0/m, p0/m, z0.s, z3.s
    add x10, x10, x5
    add x11, x11, x14, lsl #1
    sub x9, x9, #2
    b 84b

88:
    cbz x9, 85f
    ld1w { z0.s }, p0/z, [x10]
    ld1w { z2.s }, p0/z, [x11]
    add x13, x11, x14
    ld1w { z3.s }, p0/z, [x13]
    fmopa za0.s, p0/m, p0/m, z0.s, z2.s
    fmopa za1.s, p0/m, p0/m, z0.s, z3.s

85:
    add x10, x2, x7
    mov w12, #0
    lsr x13, x14, #2
86:
    cbz x13, 89f
    st1w { za0v.s[w12, 0] }, p0, [x10]
    add x16, x10, x15
    st1w { za1v.s[w12, 0] }, p0, [x16]
    add x10, x10, x6
    add w12, w12, #1
    sub x13, x13, #1
    b 86b

89:
    add x7, x7, x14
    b 83b

87:
    ret

.globl ZYNUM_SYMBOL(zynum_blas_sme_sgemm_panels2x2_f32)
.p2align 2
ZYNUM_SYMBOL(zynum_blas_sme_sgemm_panels2x2_f32):
    stp x19, x20, [sp, #-16]!
    rdsvl x14, #1         // x14 = SVL bytes; shared by all batched panels.
    lsr x15, x14, #2
    mul x15, x15, x6
    lsl x17, x14, #1
    mul x17, x17, x4
    lsl x8, x15, #1
    mov x19, x7
    lsl x20, x3, #2

    // Batched panels share one caller-owned SM/ZA region across several
    // adjacent 2N panels while preserving the packed-B layout:
    // [panel][K][2*tile cols].
    ptrue p0.s            // Full f32 tile predicate.

73:
    cbz x19, 81f
    mov x7, #0
74:
    cmp x7, x20
    b.hs 80f
    zero { za0.s, za1.s, za2.s, za3.s }

    add x10, x0, x7
    mov x11, x1
    mov x9, x4
    // Same 2M x 2N K loop as panel2x2 above.  The only difference is
    // that x1/x2 are advanced after each batched N panel is completed.
75:
    cmp x9, #2
    b.lo 82f
    ld1w { z0.s }, p0/z, [x10]
    add x13, x10, x14
    ld1w { z1.s }, p0/z, [x13]
    ld1w { z2.s }, p0/z, [x11]
    add x13, x11, x14
    ld1w { z3.s }, p0/z, [x13]
    fmopa za0.s, p0/m, p0/m, z0.s, z2.s
    fmopa za1.s, p0/m, p0/m, z1.s, z2.s
    fmopa za2.s, p0/m, p0/m, z0.s, z3.s
    fmopa za3.s, p0/m, p0/m, z1.s, z3.s
    add x10, x10, x5
    add x11, x11, x14, lsl #1
    ld1w { z0.s }, p0/z, [x10]
    add x13, x10, x14
    ld1w { z1.s }, p0/z, [x13]
    ld1w { z2.s }, p0/z, [x11]
    add x13, x11, x14
    ld1w { z3.s }, p0/z, [x13]
    fmopa za0.s, p0/m, p0/m, z0.s, z2.s
    fmopa za1.s, p0/m, p0/m, z1.s, z2.s
    fmopa za2.s, p0/m, p0/m, z0.s, z3.s
    fmopa za3.s, p0/m, p0/m, z1.s, z3.s
    add x10, x10, x5
    add x11, x11, x14, lsl #1
    sub x9, x9, #2
    b 75b

82:
    cbz x9, 76f
    ld1w { z0.s }, p0/z, [x10]
    add x13, x10, x14
    ld1w { z1.s }, p0/z, [x13]
    ld1w { z2.s }, p0/z, [x11]
    add x13, x11, x14
    ld1w { z3.s }, p0/z, [x13]
    fmopa za0.s, p0/m, p0/m, z0.s, z2.s
    fmopa za1.s, p0/m, p0/m, z1.s, z2.s
    fmopa za2.s, p0/m, p0/m, z0.s, z3.s
    fmopa za3.s, p0/m, p0/m, z1.s, z3.s

76:
    add x10, x2, x7
    mov w12, #0
    lsr x13, x14, #2
77:
    cbz x13, 79f
    st1w { za0v.s[w12, 0] }, p0, [x10]
    add x16, x10, x14
    st1w { za1v.s[w12, 0] }, p0, [x16]
    add x16, x10, x15
    st1w { za2v.s[w12, 0] }, p0, [x16]
    add x16, x16, x14
    st1w { za3v.s[w12, 0] }, p0, [x16]
    add x10, x10, x6
    add w12, w12, #1
    sub x13, x13, #1
    b 77b

79:
    add x7, x7, x14, lsl #1
    b 74b

80:
    add x1, x1, x17
    add x2, x2, x8
    sub x19, x19, #1
    b 73b

81:
    ldp x19, x20, [sp], #16
    ret

.globl ZYNUM_SYMBOL(zynum_blas_sme_sgemm_panels2x2_u4_f32)
.p2align 2
ZYNUM_SYMBOL(zynum_blas_sme_sgemm_panels2x2_u4_f32):
    stp x19, x20, [sp, #-16]!
    rdsvl x14, #1
    lsr x15, x14, #2
    mul x15, x15, x6
    lsl x17, x14, #1
    mul x17, x17, x4
    lsl x8, x15, #1
    mov x19, x7
    lsl x20, x3, #2

    // Alternate 2M x 2N f32 variant with a K-unroll4 main loop.  The caller
    // selects this symbol explicitly through the higher-level execution plan.
    ptrue p0.s

90:
    cbz x19, 99f
    mov x7, #0
91:
    cmp x7, x20
    b.hs 98f
    zero { za0.s, za1.s, za2.s, za3.s }

    add x10, x0, x7
    mov x11, x1
    mov x9, x4
92:
    cmp x9, #4
    b.lo 93f
    ld1w { z0.s }, p0/z, [x10]
    add x13, x10, x14
    ld1w { z1.s }, p0/z, [x13]
    ld1w { z2.s }, p0/z, [x11]
    add x13, x11, x14
    ld1w { z3.s }, p0/z, [x13]
    fmopa za0.s, p0/m, p0/m, z0.s, z2.s
    fmopa za1.s, p0/m, p0/m, z1.s, z2.s
    fmopa za2.s, p0/m, p0/m, z0.s, z3.s
    fmopa za3.s, p0/m, p0/m, z1.s, z3.s
    add x10, x10, x5
    add x11, x11, x14, lsl #1
    ld1w { z0.s }, p0/z, [x10]
    add x13, x10, x14
    ld1w { z1.s }, p0/z, [x13]
    ld1w { z2.s }, p0/z, [x11]
    add x13, x11, x14
    ld1w { z3.s }, p0/z, [x13]
    fmopa za0.s, p0/m, p0/m, z0.s, z2.s
    fmopa za1.s, p0/m, p0/m, z1.s, z2.s
    fmopa za2.s, p0/m, p0/m, z0.s, z3.s
    fmopa za3.s, p0/m, p0/m, z1.s, z3.s
    add x10, x10, x5
    add x11, x11, x14, lsl #1
    ld1w { z0.s }, p0/z, [x10]
    add x13, x10, x14
    ld1w { z1.s }, p0/z, [x13]
    ld1w { z2.s }, p0/z, [x11]
    add x13, x11, x14
    ld1w { z3.s }, p0/z, [x13]
    fmopa za0.s, p0/m, p0/m, z0.s, z2.s
    fmopa za1.s, p0/m, p0/m, z1.s, z2.s
    fmopa za2.s, p0/m, p0/m, z0.s, z3.s
    fmopa za3.s, p0/m, p0/m, z1.s, z3.s
    add x10, x10, x5
    add x11, x11, x14, lsl #1
    ld1w { z0.s }, p0/z, [x10]
    add x13, x10, x14
    ld1w { z1.s }, p0/z, [x13]
    ld1w { z2.s }, p0/z, [x11]
    add x13, x11, x14
    ld1w { z3.s }, p0/z, [x13]
    fmopa za0.s, p0/m, p0/m, z0.s, z2.s
    fmopa za1.s, p0/m, p0/m, z1.s, z2.s
    fmopa za2.s, p0/m, p0/m, z0.s, z3.s
    fmopa za3.s, p0/m, p0/m, z1.s, z3.s
    add x10, x10, x5
    add x11, x11, x14, lsl #1
    sub x9, x9, #4
    b 92b

93:
    cmp x9, #2
    b.lo 94f
    ld1w { z0.s }, p0/z, [x10]
    add x13, x10, x14
    ld1w { z1.s }, p0/z, [x13]
    ld1w { z2.s }, p0/z, [x11]
    add x13, x11, x14
    ld1w { z3.s }, p0/z, [x13]
    fmopa za0.s, p0/m, p0/m, z0.s, z2.s
    fmopa za1.s, p0/m, p0/m, z1.s, z2.s
    fmopa za2.s, p0/m, p0/m, z0.s, z3.s
    fmopa za3.s, p0/m, p0/m, z1.s, z3.s
    add x10, x10, x5
    add x11, x11, x14, lsl #1
    ld1w { z0.s }, p0/z, [x10]
    add x13, x10, x14
    ld1w { z1.s }, p0/z, [x13]
    ld1w { z2.s }, p0/z, [x11]
    add x13, x11, x14
    ld1w { z3.s }, p0/z, [x13]
    fmopa za0.s, p0/m, p0/m, z0.s, z2.s
    fmopa za1.s, p0/m, p0/m, z1.s, z2.s
    fmopa za2.s, p0/m, p0/m, z0.s, z3.s
    fmopa za3.s, p0/m, p0/m, z1.s, z3.s
    add x10, x10, x5
    add x11, x11, x14, lsl #1
    sub x9, x9, #2
94:
    cbz x9, 95f
    ld1w { z0.s }, p0/z, [x10]
    add x13, x10, x14
    ld1w { z1.s }, p0/z, [x13]
    ld1w { z2.s }, p0/z, [x11]
    add x13, x11, x14
    ld1w { z3.s }, p0/z, [x13]
    fmopa za0.s, p0/m, p0/m, z0.s, z2.s
    fmopa za1.s, p0/m, p0/m, z1.s, z2.s
    fmopa za2.s, p0/m, p0/m, z0.s, z3.s
    fmopa za3.s, p0/m, p0/m, z1.s, z3.s

95:
    add x10, x2, x7
    mov w12, #0
    lsr x13, x14, #2
96:
    cbz x13, 97f
    st1w { za0v.s[w12, 0] }, p0, [x10]
    add x16, x10, x14
    st1w { za1v.s[w12, 0] }, p0, [x16]
    add x16, x10, x15
    st1w { za2v.s[w12, 0] }, p0, [x16]
    add x16, x16, x14
    st1w { za3v.s[w12, 0] }, p0, [x16]
    add x10, x10, x6
    add w12, w12, #1
    sub x13, x13, #1
    b 96b

97:
    add x7, x7, x14, lsl #1
    b 91b

98:
    add x1, x1, x17
    add x2, x2, x8
    sub x19, x19, #1
    b 90b

99:
    ldp x19, x20, [sp], #16
    ret
