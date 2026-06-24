// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later

.text
// Base SME provides streaming mode, ZA tiles, and FMOPA.  The f64 kernels
// additionally need SME_F64F64 so .d FMOPA accumulates are architectural.
// They do not currently use SME2-only instructions; SME2/SME2.1 feature
// policy stays in Zig dispatch so these symbols describe their real ISA need.
.arch_extension sme
.arch_extension sme-f64f64

#ifdef __APPLE__
#define ZYNUM_SYMBOL(name) _##name
#else
#define ZYNUM_SYMBOL(name) name
#endif

// Common ABI for the f64 SME kernels:
//   x0=A, x1=packed B, x2=C, x3=m_full rows, x4=K,
//   x5=lda bytes, x6=ldc bytes, x7=batch panel count when present.
// x14 is the streaming vector length in bytes; one f64 tile has x14/8
// rows/columns.  These kernels require SME_F64F64 accumulate support.
// These are body symbols: the Zig SME worker wrapper owns SM/ZA entry,
// callee-save preservation, and SM/ZA exit.  Callers must enter streaming
// mode and enable ZA before using any symbol in this file.

.globl ZYNUM_SYMBOL(zynum_blas_sme_dgemm_panel_f64)
.p2align 2
ZYNUM_SYMBOL(zynum_blas_sme_dgemm_panel_f64):
    rdsvl x14, #1         // x14 = SVL bytes; all f64 tile strides derive from it.
    ptrue p0.d            // All f64 lanes active; only full tiles enter assembly.

    lsl x3, x3, #3
    mov x7, #0
19:
    cmp x7, x3
    b.hs 23f
    zero { za0.d }

    add x10, x0, x7
    mov x11, x1
    mov x9, x4
20:
    cbz x9, 21f
    ld1d { z0.d }, p0/z, [x10]
    ld1d { z1.d }, p0/z, [x11]
    fmopa za0.d, p0/m, p0/m, z0.d, z1.d
    add x10, x10, x5
    add x11, x11, x14
    sub x9, x9, #1
    b 20b

21:
    add x10, x2, x7
    mov w12, #0
    lsr x13, x14, #3
22:
    cbz x13, 24f
    st1d { za0v.d[w12, 0] }, p0, [x10]
    add x10, x10, x6
    add w12, w12, #1
    sub x13, x13, #1
    b 22b

24:
    add x7, x7, x14
    b 19b

23:
    ret

.globl ZYNUM_SYMBOL(zynum_blas_sme_dgemm_panel2x2_f64)
.p2align 2
ZYNUM_SYMBOL(zynum_blas_sme_dgemm_panel2x2_f64):
    rdsvl x14, #1         // x14 = SVL bytes; one f64 tile is x14/8 rows.
    ptrue p0.d            // Full f64 tile predicate.

    lsr x15, x14, #3
    mul x15, x15, x6

    // 2M x 2N residual kernel.  The main f64 path uses 4M x 2N, and this
    // avoids sending 2 full SME tile rows to scalar tails.
    lsl x3, x3, #3
    mov x7, #0
37:
    cmp x7, x3
    b.hs 41f
    zero { za0.d, za1.d, za2.d, za3.d }

    add x10, x0, x7
    mov x11, x1
    mov x9, x4
38:
    cmp x9, #2
    b.lo 44f
    ld1d { z0.d }, p0/z, [x10]
    add x13, x10, x14
    ld1d { z1.d }, p0/z, [x13]
    ld1d { z2.d }, p0/z, [x11]
    add x13, x11, x14
    ld1d { z3.d }, p0/z, [x13]
    fmopa za0.d, p0/m, p0/m, z0.d, z2.d
    fmopa za1.d, p0/m, p0/m, z1.d, z2.d
    fmopa za2.d, p0/m, p0/m, z0.d, z3.d
    fmopa za3.d, p0/m, p0/m, z1.d, z3.d
    add x10, x10, x5
    add x11, x11, x14, lsl #1
    ld1d { z0.d }, p0/z, [x10]
    add x13, x10, x14
    ld1d { z1.d }, p0/z, [x13]
    ld1d { z2.d }, p0/z, [x11]
    add x13, x11, x14
    ld1d { z3.d }, p0/z, [x13]
    fmopa za0.d, p0/m, p0/m, z0.d, z2.d
    fmopa za1.d, p0/m, p0/m, z1.d, z2.d
    fmopa za2.d, p0/m, p0/m, z0.d, z3.d
    fmopa za3.d, p0/m, p0/m, z1.d, z3.d
    add x10, x10, x5
    add x11, x11, x14, lsl #1
    sub x9, x9, #2
    b 38b

44:
    cbz x9, 39f
    ld1d { z0.d }, p0/z, [x10]
    add x13, x10, x14
    ld1d { z1.d }, p0/z, [x13]
    ld1d { z2.d }, p0/z, [x11]
    add x13, x11, x14
    ld1d { z3.d }, p0/z, [x13]
    fmopa za0.d, p0/m, p0/m, z0.d, z2.d
    fmopa za1.d, p0/m, p0/m, z1.d, z2.d
    fmopa za2.d, p0/m, p0/m, z0.d, z3.d
    fmopa za3.d, p0/m, p0/m, z1.d, z3.d

39:
    add x10, x2, x7
    mov w12, #0
    lsr x13, x14, #3
40:
    cbz x13, 42f
    st1d { za0v.d[w12, 0] }, p0, [x10]
    add x16, x10, x14
    st1d { za1v.d[w12, 0] }, p0, [x16]
    add x16, x10, x15
    st1d { za2v.d[w12, 0] }, p0, [x16]
    add x16, x16, x14
    st1d { za3v.d[w12, 0] }, p0, [x16]
    add x10, x10, x6
    add w12, w12, #1
    sub x13, x13, #1
    b 40b

42:
    add x7, x7, x14, lsl #1
    b 37b

41:
    ret

.globl ZYNUM_SYMBOL(zynum_blas_sme_dgemm_panel4m_f64)
.p2align 2
ZYNUM_SYMBOL(zynum_blas_sme_dgemm_panel4m_f64):
    rdsvl x14, #1         // x14 = SVL bytes; one f64 tile is x14/8 rows.
    ptrue p0.d            // Use all f64 lanes in each tile row.

    lsl x3, x3, #3
    mov x7, #0
25:
    cmp x7, x3
    b.hs 29f
    zero { za0.d, za1.d, za2.d, za3.d }

    add x10, x0, x7
    mov x11, x1
    mov x9, x4
26:
    cbz x9, 27f
    ld1d { z0.d }, p0/z, [x10]
    add x13, x10, x14
    ld1d { z1.d }, p0/z, [x13]
    add x13, x10, x14, lsl #1
    ld1d { z2.d }, p0/z, [x13]
    add x13, x10, x14
    add x13, x13, x14, lsl #1
    ld1d { z3.d }, p0/z, [x13]
    ld1d { z4.d }, p0/z, [x11]
    fmopa za0.d, p0/m, p0/m, z0.d, z4.d
    fmopa za1.d, p0/m, p0/m, z1.d, z4.d
    fmopa za2.d, p0/m, p0/m, z2.d, z4.d
    fmopa za3.d, p0/m, p0/m, z3.d, z4.d
    add x10, x10, x5
    add x11, x11, x14
    sub x9, x9, #1
    b 26b

27:
    add x10, x2, x7
    mov w12, #0
    lsr x13, x14, #3
28:
    cbz x13, 30f
    st1d { za0v.d[w12, 0] }, p0, [x10]
    add x16, x10, x14
    st1d { za1v.d[w12, 0] }, p0, [x16]
    add x16, x10, x14, lsl #1
    st1d { za2v.d[w12, 0] }, p0, [x16]
    add x16, x10, x14
    add x16, x16, x14, lsl #1
    st1d { za3v.d[w12, 0] }, p0, [x16]
    add x10, x10, x6
    add w12, w12, #1
    sub x13, x13, #1
    b 28b

30:
    add x7, x7, x14, lsl #2
    b 25b

29:
    ret

.globl ZYNUM_SYMBOL(zynum_blas_sme_dgemm_panel4x2_f64)
.p2align 2
ZYNUM_SYMBOL(zynum_blas_sme_dgemm_panel4x2_f64):
    rdsvl x14, #1         // x14 = SVL bytes, reused for A/B/C tile offsets.
    ptrue p0.d            // Full f64 tile predicate.

    lsr x15, x14, #3
    mul x15, x15, x6

    // 4M x 2N kernel uses za0..za7: four M panels times two N panels.
    // f64 can encode this tile group on Apple M5; f32 could not use
    // the same za4.s..za7.s form with the current assembler/ISA.
    lsl x3, x3, #3
    mov x7, #0
45:
    cmp x7, x3
    b.hs 49f
    zero { za0.d, za1.d, za2.d, za3.d, za4.d, za5.d, za6.d, za7.d }

    add x10, x0, x7
    mov x11, x1
    mov x9, x4
46:
    cmp x9, #2                            // Need two K steps for the unrolled body.
    b.lo 51f                              // Drop to the one-K tail when fewer remain.
    prfm pldl1keep, [x11, #512]           // Pull future packed B into L1 for high-K f64.
    ld1d { z0.d }, p0/z, [x10]            // A panel 0: A[i : i+tile, p].
    add x13, x10, x14                     // Address A panel 1.
    ld1d { z1.d }, p0/z, [x13]            // A panel 1: +1 tile row.
    add x13, x10, x14, lsl #1             // Address A panel 2.
    ld1d { z2.d }, p0/z, [x13]            // A panel 2: +2 tile rows.
    add x13, x10, x14                     // Rebuild +1 tile row address.
    add x13, x13, x14, lsl #1             // Address A panel 3: +3 tile rows.
    ld1d { z3.d }, p0/z, [x13]            // A panel 3.
    ld1d { z4.d }, p0/z, [x11]            // B panel 0: packed B[p, j : j+tile].
    add x13, x11, x14                     // Address B panel 1 inside the 2N pack.
    ld1d { z5.d }, p0/z, [x13]            // B panel 1: packed B[p, j+tile : j+2*tile].
    fmopa za0.d, p0/m, p0/m, z0.d, z4.d  // C00 += A0 * B0.
    fmopa za1.d, p0/m, p0/m, z1.d, z4.d  // C10 += A1 * B0.
    fmopa za2.d, p0/m, p0/m, z2.d, z4.d  // C20 += A2 * B0.
    fmopa za3.d, p0/m, p0/m, z3.d, z4.d  // C30 += A3 * B0.
    fmopa za4.d, p0/m, p0/m, z0.d, z5.d  // C01 += A0 * B1.
    fmopa za5.d, p0/m, p0/m, z1.d, z5.d  // C11 += A1 * B1.
    fmopa za6.d, p0/m, p0/m, z2.d, z5.d  // C21 += A2 * B1.
    fmopa za7.d, p0/m, p0/m, z3.d, z5.d  // C31 += A3 * B1.
    add x10, x10, x5                      // Advance A to K p+1 using lda bytes.
    add x11, x11, x14, lsl #1             // Advance packed B by 2 tile columns.
    ld1d { z0.d }, p0/z, [x10]            // A panel 0 for K p+1.
    add x13, x10, x14                     // Address A panel 1 for K p+1.
    ld1d { z1.d }, p0/z, [x13]            // A panel 1 for K p+1.
    add x13, x10, x14, lsl #1             // Address A panel 2 for K p+1.
    ld1d { z2.d }, p0/z, [x13]            // A panel 2 for K p+1.
    add x13, x10, x14                     // Rebuild +1 tile row address.
    add x13, x13, x14, lsl #1             // Address A panel 3 for K p+1.
    ld1d { z3.d }, p0/z, [x13]            // A panel 3 for K p+1.
    ld1d { z4.d }, p0/z, [x11]            // B panel 0 for K p+1.
    add x13, x11, x14                     // Address B panel 1 for K p+1.
    ld1d { z5.d }, p0/z, [x13]            // B panel 1 for K p+1.
    fmopa za0.d, p0/m, p0/m, z0.d, z4.d  // C00 += A0 * B0 for K p+1.
    fmopa za1.d, p0/m, p0/m, z1.d, z4.d  // C10 += A1 * B0 for K p+1.
    fmopa za2.d, p0/m, p0/m, z2.d, z4.d  // C20 += A2 * B0 for K p+1.
    fmopa za3.d, p0/m, p0/m, z3.d, z4.d  // C30 += A3 * B0 for K p+1.
    fmopa za4.d, p0/m, p0/m, z0.d, z5.d  // C01 += A0 * B1 for K p+1.
    fmopa za5.d, p0/m, p0/m, z1.d, z5.d  // C11 += A1 * B1 for K p+1.
    fmopa za6.d, p0/m, p0/m, z2.d, z5.d  // C21 += A2 * B1 for K p+1.
    fmopa za7.d, p0/m, p0/m, z3.d, z5.d  // C31 += A3 * B1 for K p+1.
    add x10, x10, x5                      // Advance A to the next unrolled pair.
    add x11, x11, x14, lsl #1             // Advance packed B to the next K pair.
    sub x9, x9, #2                        // Two K positions consumed.
    b 46b                                 // Continue until fewer than two remain.

51:
    cbz x9, 47f
    prfm pldl1keep, [x11, #512]
    ld1d { z0.d }, p0/z, [x10]
    add x13, x10, x14
    ld1d { z1.d }, p0/z, [x13]
    add x13, x10, x14, lsl #1
    ld1d { z2.d }, p0/z, [x13]
    add x13, x10, x14
    add x13, x13, x14, lsl #1
    ld1d { z3.d }, p0/z, [x13]
    ld1d { z4.d }, p0/z, [x11]
    add x13, x11, x14
    ld1d { z5.d }, p0/z, [x13]
    fmopa za0.d, p0/m, p0/m, z0.d, z4.d
    fmopa za1.d, p0/m, p0/m, z1.d, z4.d
    fmopa za2.d, p0/m, p0/m, z2.d, z4.d
    fmopa za3.d, p0/m, p0/m, z3.d, z4.d
    fmopa za4.d, p0/m, p0/m, z0.d, z5.d
    fmopa za5.d, p0/m, p0/m, z1.d, z5.d
    fmopa za6.d, p0/m, p0/m, z2.d, z5.d
    fmopa za7.d, p0/m, p0/m, z3.d, z5.d

47:
    add x10, x2, x7
    mov w12, #0
    lsr x13, x14, #3
48:
    cbz x13, 50f                          // All ZA rows stored.
    st1d { za0v.d[w12, 0] }, p0, [x10]    // Store C00 row into column j.
    add x16, x10, x14                     // Address C10: +1 tile row.
    st1d { za1v.d[w12, 0] }, p0, [x16]    // Store C10.
    add x16, x10, x14, lsl #1             // Address C20: +2 tile rows.
    st1d { za2v.d[w12, 0] }, p0, [x16]    // Store C20.
    add x16, x10, x14                     // Rebuild +1 tile row address.
    add x16, x16, x14, lsl #1             // Address C30: +3 tile rows.
    st1d { za3v.d[w12, 0] }, p0, [x16]    // Store C30.
    add x16, x10, x15                     // Address C01: +tile columns.
    st1d { za4v.d[w12, 0] }, p0, [x16]    // Store C01.
    add x16, x16, x14                     // Address C11.
    st1d { za5v.d[w12, 0] }, p0, [x16]    // Store C11.
    add x16, x10, x15                     // Rebuild C01 base.
    add x16, x16, x14, lsl #1             // Address C21.
    st1d { za6v.d[w12, 0] }, p0, [x16]    // Store C21.
    add x16, x10, x15                     // Rebuild C01 base.
    add x16, x16, x14                     // Add one tile row.
    add x16, x16, x14, lsl #1             // Address C31.
    st1d { za7v.d[w12, 0] }, p0, [x16]    // Store C31.
    add x10, x10, x6                      // Advance to next C row using ldc bytes.
    add w12, w12, #1                      // Select next ZA row vector.
    sub x13, x13, #1                      // One row vector stored.
    b 48b                                 // Continue across tile rows.

50:
    add x7, x7, x14, lsl #2
    b 45b

49:
    ret

.globl ZYNUM_SYMBOL(zynum_blas_sme_dgemm_panels4x2_f64)
.p2align 2
ZYNUM_SYMBOL(zynum_blas_sme_dgemm_panels4x2_f64):
    stp x19, x20, [sp, #-16]!
    rdsvl x14, #1         // x14 = SVL bytes; shared by all batched panels.
    lsr x15, x14, #3
    mul x15, x15, x6
    lsl x17, x14, #1
    mul x17, x17, x4
    lsl x8, x15, #1
    mov x19, x7
    lsl x20, x3, #3

    // Batched 4M x 2N panels share the caller-owned SM/ZA region.  x17
    // advances packed B by one panel and x8 advances C.
    ptrue p0.d            // Full f64 tile predicate.

64:
    cbz x19, 72f
    mov x7, #0
65:
    cmp x7, x20
    b.hs 71f
    zero { za0.d, za1.d, za2.d, za3.d, za4.d, za5.d, za6.d, za7.d }

    add x10, x0, x7
    mov x11, x1
    mov x9, x4
    // Same 4M x 2N K loop as panel4x2 above.  Batched mode only keeps
    // SM/ZA enabled while x1/x2 advance across adjacent packed panels.
66:
    cmp x9, #2
    b.lo 69f
    prfm pldl1keep, [x11, #512]
    ld1d { z0.d }, p0/z, [x10]
    add x13, x10, x14
    ld1d { z1.d }, p0/z, [x13]
    add x13, x10, x14, lsl #1
    ld1d { z2.d }, p0/z, [x13]
    add x13, x10, x14
    add x13, x13, x14, lsl #1
    ld1d { z3.d }, p0/z, [x13]
    ld1d { z4.d }, p0/z, [x11]
    add x13, x11, x14
    ld1d { z5.d }, p0/z, [x13]
    fmopa za0.d, p0/m, p0/m, z0.d, z4.d
    fmopa za1.d, p0/m, p0/m, z1.d, z4.d
    fmopa za2.d, p0/m, p0/m, z2.d, z4.d
    fmopa za3.d, p0/m, p0/m, z3.d, z4.d
    fmopa za4.d, p0/m, p0/m, z0.d, z5.d
    fmopa za5.d, p0/m, p0/m, z1.d, z5.d
    fmopa za6.d, p0/m, p0/m, z2.d, z5.d
    fmopa za7.d, p0/m, p0/m, z3.d, z5.d
    add x10, x10, x5
    add x11, x11, x14, lsl #1
    ld1d { z0.d }, p0/z, [x10]
    add x13, x10, x14
    ld1d { z1.d }, p0/z, [x13]
    add x13, x10, x14, lsl #1
    ld1d { z2.d }, p0/z, [x13]
    add x13, x10, x14
    add x13, x13, x14, lsl #1
    ld1d { z3.d }, p0/z, [x13]
    ld1d { z4.d }, p0/z, [x11]
    add x13, x11, x14
    ld1d { z5.d }, p0/z, [x13]
    fmopa za0.d, p0/m, p0/m, z0.d, z4.d
    fmopa za1.d, p0/m, p0/m, z1.d, z4.d
    fmopa za2.d, p0/m, p0/m, z2.d, z4.d
    fmopa za3.d, p0/m, p0/m, z3.d, z4.d
    fmopa za4.d, p0/m, p0/m, z0.d, z5.d
    fmopa za5.d, p0/m, p0/m, z1.d, z5.d
    fmopa za6.d, p0/m, p0/m, z2.d, z5.d
    fmopa za7.d, p0/m, p0/m, z3.d, z5.d
    add x10, x10, x5
    add x11, x11, x14, lsl #1
    sub x9, x9, #2
    b 66b

69:
    cbz x9, 67f
    prfm pldl1keep, [x11, #512]
    ld1d { z0.d }, p0/z, [x10]
    add x13, x10, x14
    ld1d { z1.d }, p0/z, [x13]
    add x13, x10, x14, lsl #1
    ld1d { z2.d }, p0/z, [x13]
    add x13, x10, x14
    add x13, x13, x14, lsl #1
    ld1d { z3.d }, p0/z, [x13]
    ld1d { z4.d }, p0/z, [x11]
    add x13, x11, x14
    ld1d { z5.d }, p0/z, [x13]
    fmopa za0.d, p0/m, p0/m, z0.d, z4.d
    fmopa za1.d, p0/m, p0/m, z1.d, z4.d
    fmopa za2.d, p0/m, p0/m, z2.d, z4.d
    fmopa za3.d, p0/m, p0/m, z3.d, z4.d
    fmopa za4.d, p0/m, p0/m, z0.d, z5.d
    fmopa za5.d, p0/m, p0/m, z1.d, z5.d
    fmopa za6.d, p0/m, p0/m, z2.d, z5.d
    fmopa za7.d, p0/m, p0/m, z3.d, z5.d

67:
    add x10, x2, x7
    mov w12, #0
    lsr x13, x14, #3
68:
    cbz x13, 70f
    st1d { za0v.d[w12, 0] }, p0, [x10]
    add x16, x10, x14
    st1d { za1v.d[w12, 0] }, p0, [x16]
    add x16, x10, x14, lsl #1
    st1d { za2v.d[w12, 0] }, p0, [x16]
    add x16, x10, x14
    add x16, x16, x14, lsl #1
    st1d { za3v.d[w12, 0] }, p0, [x16]
    add x16, x10, x15
    st1d { za4v.d[w12, 0] }, p0, [x16]
    add x16, x16, x14
    st1d { za5v.d[w12, 0] }, p0, [x16]
    add x16, x10, x15
    add x16, x16, x14, lsl #1
    st1d { za6v.d[w12, 0] }, p0, [x16]
    add x16, x10, x15
    add x16, x16, x14
    add x16, x16, x14, lsl #1
    st1d { za7v.d[w12, 0] }, p0, [x16]
    add x10, x10, x6
    add w12, w12, #1
    sub x13, x13, #1
    b 68b

70:
    add x7, x7, x14, lsl #2
    b 65b

71:
    add x1, x1, x17
    add x2, x2, x8
    sub x19, x19, #1
    b 64b

72:
    ldp x19, x20, [sp], #16
    ret
