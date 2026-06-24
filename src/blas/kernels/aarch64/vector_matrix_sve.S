// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later

.text
.arch_extension sve
.arch_extension sme
.arch_extension sme2
.arch_extension sme-f64f64

#ifdef __APPLE__
#define ZYNUM_SYMBOL(name) _##name
#else
#define ZYNUM_SYMBOL(name) name
#endif

// SVE/ASIMD symbols in this file run in the normal non-streaming state.
// SME/SME2 body symbols assume their Zig caller has already entered SM and,
// for ZA users, enabled ZA through features.StreamingModeState.

// void zynum_blas_sve_dgemv_t_f64(
//     usize m, usize n, double alpha, const double *a, usize lda_bytes,
//     const double *x, double *y)
//
// AArch64 ABI:
//   x0=m, x1=n, d0=alpha, x2=a, x3=lda_bytes, x4=x, x5=y.
//
// Computes y[j] += alpha * dot(A[:, j], x) for column-major A.  The hot path
// accumulates eight output columns so each SVE load of x feeds eight FMLA
// chains, matching the existing GEMM convention of isolating higher-ISA
// prologue/epilogue in architecture code.
.globl ZYNUM_SYMBOL(zynum_blas_sve_dgemv_t_f64)
.p2align 2
ZYNUM_SYMBOL(zynum_blas_sve_dgemv_t_f64):
    cbz x0, .Ldone
    cbz x1, .Ldone
    ptrue p1.d

.Lblock8_check:
    cmp x1, #8
    b.lo .Ltail1_check

.Lblock8:
    mov x8, x2
    add x9, x8, x3
    add x10, x9, x3
    add x11, x10, x3
    add x12, x11, x3
    add x13, x12, x3
    add x14, x13, x3
    add x15, x14, x3

    dup z0.d, #0
    dup z1.d, #0
    dup z2.d, #0
    dup z3.d, #0
    dup z4.d, #0
    dup z5.d, #0
    dup z6.d, #0
    dup z7.d, #0
    mov x16, #0

.Lblock8_rows:
    whilelo p0.d, x16, x0
    b.none .Lblock8_reduce
    ld1d { z16.d }, p0/z, [x4, x16, lsl #3]
    ld1d { z17.d }, p0/z, [x8, x16, lsl #3]
    ld1d { z18.d }, p0/z, [x9, x16, lsl #3]
    ld1d { z19.d }, p0/z, [x10, x16, lsl #3]
    ld1d { z20.d }, p0/z, [x11, x16, lsl #3]
    ld1d { z21.d }, p0/z, [x12, x16, lsl #3]
    ld1d { z22.d }, p0/z, [x13, x16, lsl #3]
    ld1d { z23.d }, p0/z, [x14, x16, lsl #3]
    ld1d { z24.d }, p0/z, [x15, x16, lsl #3]
    fmla z0.d, p0/m, z17.d, z16.d
    fmla z1.d, p0/m, z18.d, z16.d
    fmla z2.d, p0/m, z19.d, z16.d
    fmla z3.d, p0/m, z20.d, z16.d
    fmla z4.d, p0/m, z21.d, z16.d
    fmla z5.d, p0/m, z22.d, z16.d
    fmla z6.d, p0/m, z23.d, z16.d
    fmla z7.d, p0/m, z24.d, z16.d
    incd x16
    b .Lblock8_rows

.Lblock8_reduce:
    faddv d1, p1, z0.d
    ldr d2, [x5]
    fmadd d2, d0, d1, d2
    str d2, [x5]
    faddv d1, p1, z1.d
    ldr d2, [x5, #8]
    fmadd d2, d0, d1, d2
    str d2, [x5, #8]
    faddv d1, p1, z2.d
    ldr d2, [x5, #16]
    fmadd d2, d0, d1, d2
    str d2, [x5, #16]
    faddv d1, p1, z3.d
    ldr d2, [x5, #24]
    fmadd d2, d0, d1, d2
    str d2, [x5, #24]
    faddv d1, p1, z4.d
    ldr d2, [x5, #32]
    fmadd d2, d0, d1, d2
    str d2, [x5, #32]
    faddv d1, p1, z5.d
    ldr d2, [x5, #40]
    fmadd d2, d0, d1, d2
    str d2, [x5, #40]
    faddv d1, p1, z6.d
    ldr d2, [x5, #48]
    fmadd d2, d0, d1, d2
    str d2, [x5, #48]
    faddv d1, p1, z7.d
    ldr d2, [x5, #56]
    fmadd d2, d0, d1, d2
    str d2, [x5, #56]

    add x2, x15, x3
    add x5, x5, #64
    sub x1, x1, #8
    b .Lblock8_check

.Ltail1_check:
    cbz x1, .Ldone

.Ltail1:
    dup z0.d, #0
    mov x16, #0

.Ltail1_rows:
    whilelo p0.d, x16, x0
    b.none .Ltail1_reduce
    ld1d { z16.d }, p0/z, [x4, x16, lsl #3]
    ld1d { z17.d }, p0/z, [x2, x16, lsl #3]
    fmla z0.d, p0/m, z17.d, z16.d
    incd x16
    b .Ltail1_rows

.Ltail1_reduce:
    faddv d1, p1, z0.d
    ldr d2, [x5]
    fmadd d2, d0, d1, d2
    str d2, [x5]
    add x2, x2, x3
    add x5, x5, #8
    subs x1, x1, #1
    b.ne .Ltail1

.Ldone:
    ret

// void zynum_blas_sve_dgemv_t_f64_full_n8(
//     usize m, usize n, double alpha, const double *a, usize lda_bytes,
//     const double *x, double *y)
//
// Same n8 panel as zynum_blas_sve_dgemv_t_f64, specialized for m being an
// exact multiple of the SVE double-lane count.  The row loop can use a single
// ptrue predicate and a count compare instead of rebuilding whilelo each step.
.globl ZYNUM_SYMBOL(zynum_blas_sve_dgemv_t_f64_full_n8)
.p2align 2
ZYNUM_SYMBOL(zynum_blas_sve_dgemv_t_f64_full_n8):
    cbz x0, .Lfull_done
    cbz x1, .Lfull_done
    ptrue p0.d
    ptrue p1.d

.Lfull_block8_check:
    cmp x1, #8
    b.lo .Lfull_tail

.Lfull_block8:
    mov x8, x2
    add x9, x8, x3
    add x10, x9, x3
    add x11, x10, x3
    add x12, x11, x3
    add x13, x12, x3
    add x14, x13, x3
    add x15, x14, x3

    dup z0.d, #0
    dup z1.d, #0
    dup z2.d, #0
    dup z3.d, #0
    dup z4.d, #0
    dup z5.d, #0
    dup z6.d, #0
    dup z7.d, #0
    mov x16, #0

.Lfull_block8_rows:
    cmp x16, x0
    b.hs .Lfull_block8_reduce
    ld1d { z16.d }, p0/z, [x4, x16, lsl #3]
    ld1d { z17.d }, p0/z, [x8, x16, lsl #3]
    ld1d { z18.d }, p0/z, [x9, x16, lsl #3]
    ld1d { z19.d }, p0/z, [x10, x16, lsl #3]
    ld1d { z20.d }, p0/z, [x11, x16, lsl #3]
    ld1d { z21.d }, p0/z, [x12, x16, lsl #3]
    ld1d { z22.d }, p0/z, [x13, x16, lsl #3]
    ld1d { z23.d }, p0/z, [x14, x16, lsl #3]
    ld1d { z24.d }, p0/z, [x15, x16, lsl #3]
    fmla z0.d, p0/m, z17.d, z16.d
    fmla z1.d, p0/m, z18.d, z16.d
    fmla z2.d, p0/m, z19.d, z16.d
    fmla z3.d, p0/m, z20.d, z16.d
    fmla z4.d, p0/m, z21.d, z16.d
    fmla z5.d, p0/m, z22.d, z16.d
    fmla z6.d, p0/m, z23.d, z16.d
    fmla z7.d, p0/m, z24.d, z16.d
    incd x16
    b .Lfull_block8_rows

.Lfull_block8_reduce:
    faddv d1, p1, z0.d
    ldr d2, [x5]
    fmadd d2, d0, d1, d2
    str d2, [x5]
    faddv d1, p1, z1.d
    ldr d2, [x5, #8]
    fmadd d2, d0, d1, d2
    str d2, [x5, #8]
    faddv d1, p1, z2.d
    ldr d2, [x5, #16]
    fmadd d2, d0, d1, d2
    str d2, [x5, #16]
    faddv d1, p1, z3.d
    ldr d2, [x5, #24]
    fmadd d2, d0, d1, d2
    str d2, [x5, #24]
    faddv d1, p1, z4.d
    ldr d2, [x5, #32]
    fmadd d2, d0, d1, d2
    str d2, [x5, #32]
    faddv d1, p1, z5.d
    ldr d2, [x5, #40]
    fmadd d2, d0, d1, d2
    str d2, [x5, #40]
    faddv d1, p1, z6.d
    ldr d2, [x5, #48]
    fmadd d2, d0, d1, d2
    str d2, [x5, #48]
    faddv d1, p1, z7.d
    ldr d2, [x5, #56]
    fmadd d2, d0, d1, d2
    str d2, [x5, #56]

    add x2, x15, x3
    add x5, x5, #64
    sub x1, x1, #8
    b .Lfull_block8_check

.Lfull_tail:
    cbz x1, .Lfull_done
    b ZYNUM_SYMBOL(zynum_blas_sve_dgemv_t_f64)

.Lfull_done:
    ret

// void zynum_blas_sve_dgemv_t_f64_full_n8_acc2(
//     usize m, usize n, double alpha, const double *a, usize lda_bytes,
//     const double *x, double *y)
//
// Full-vector n8 panel with two independent accumulation chains per output
// column. The caller gates this to m being a multiple of twice the SVE
// double-lane count.
.globl ZYNUM_SYMBOL(zynum_blas_sve_dgemv_t_f64_full_n8_acc2)
.p2align 2
ZYNUM_SYMBOL(zynum_blas_sve_dgemv_t_f64_full_n8_acc2):
    cbz x0, .Lacc2_done
    cbz x1, .Lacc2_done
    ptrue p0.d
    ptrue p1.d

.Lacc2_block8_check:
    cmp x1, #8
    b.lo .Lacc2_tail

.Lacc2_block8:
    mov x8, x2
    add x9, x8, x3
    add x10, x9, x3
    add x11, x10, x3
    add x12, x11, x3
    add x13, x12, x3
    add x14, x13, x3
    add x15, x14, x3

    dup z0.d, #0
    dup z1.d, #0
    dup z2.d, #0
    dup z3.d, #0
    dup z4.d, #0
    dup z5.d, #0
    dup z6.d, #0
    dup z7.d, #0
    dup z8.d, #0
    dup z9.d, #0
    dup z10.d, #0
    dup z11.d, #0
    dup z12.d, #0
    dup z13.d, #0
    dup z14.d, #0
    dup z15.d, #0
    mov x16, #0

.Lacc2_block8_rows:
    cmp x16, x0
    b.hs .Lacc2_block8_reduce
    ld1d { z16.d }, p0/z, [x4, x16, lsl #3]
    ld1d { z17.d }, p0/z, [x8, x16, lsl #3]
    ld1d { z18.d }, p0/z, [x9, x16, lsl #3]
    ld1d { z19.d }, p0/z, [x10, x16, lsl #3]
    ld1d { z20.d }, p0/z, [x11, x16, lsl #3]
    ld1d { z21.d }, p0/z, [x12, x16, lsl #3]
    ld1d { z22.d }, p0/z, [x13, x16, lsl #3]
    ld1d { z23.d }, p0/z, [x14, x16, lsl #3]
    ld1d { z24.d }, p0/z, [x15, x16, lsl #3]
    fmla z0.d, p0/m, z17.d, z16.d
    fmla z1.d, p0/m, z18.d, z16.d
    fmla z2.d, p0/m, z19.d, z16.d
    fmla z3.d, p0/m, z20.d, z16.d
    fmla z4.d, p0/m, z21.d, z16.d
    fmla z5.d, p0/m, z22.d, z16.d
    fmla z6.d, p0/m, z23.d, z16.d
    fmla z7.d, p0/m, z24.d, z16.d
    incd x16

    ld1d { z16.d }, p0/z, [x4, x16, lsl #3]
    ld1d { z17.d }, p0/z, [x8, x16, lsl #3]
    ld1d { z18.d }, p0/z, [x9, x16, lsl #3]
    ld1d { z19.d }, p0/z, [x10, x16, lsl #3]
    ld1d { z20.d }, p0/z, [x11, x16, lsl #3]
    ld1d { z21.d }, p0/z, [x12, x16, lsl #3]
    ld1d { z22.d }, p0/z, [x13, x16, lsl #3]
    ld1d { z23.d }, p0/z, [x14, x16, lsl #3]
    ld1d { z24.d }, p0/z, [x15, x16, lsl #3]
    fmla z8.d, p0/m, z17.d, z16.d
    fmla z9.d, p0/m, z18.d, z16.d
    fmla z10.d, p0/m, z19.d, z16.d
    fmla z11.d, p0/m, z20.d, z16.d
    fmla z12.d, p0/m, z21.d, z16.d
    fmla z13.d, p0/m, z22.d, z16.d
    fmla z14.d, p0/m, z23.d, z16.d
    fmla z15.d, p0/m, z24.d, z16.d
    incd x16
    b .Lacc2_block8_rows

.Lacc2_block8_reduce:
    fadd z0.d, z0.d, z8.d
    fadd z1.d, z1.d, z9.d
    fadd z2.d, z2.d, z10.d
    fadd z3.d, z3.d, z11.d
    fadd z4.d, z4.d, z12.d
    fadd z5.d, z5.d, z13.d
    fadd z6.d, z6.d, z14.d
    fadd z7.d, z7.d, z15.d

    faddv d1, p1, z0.d
    ldr d2, [x5]
    fmadd d2, d0, d1, d2
    str d2, [x5]
    faddv d1, p1, z1.d
    ldr d2, [x5, #8]
    fmadd d2, d0, d1, d2
    str d2, [x5, #8]
    faddv d1, p1, z2.d
    ldr d2, [x5, #16]
    fmadd d2, d0, d1, d2
    str d2, [x5, #16]
    faddv d1, p1, z3.d
    ldr d2, [x5, #24]
    fmadd d2, d0, d1, d2
    str d2, [x5, #24]
    faddv d1, p1, z4.d
    ldr d2, [x5, #32]
    fmadd d2, d0, d1, d2
    str d2, [x5, #32]
    faddv d1, p1, z5.d
    ldr d2, [x5, #40]
    fmadd d2, d0, d1, d2
    str d2, [x5, #40]
    faddv d1, p1, z6.d
    ldr d2, [x5, #48]
    fmadd d2, d0, d1, d2
    str d2, [x5, #48]
    faddv d1, p1, z7.d
    ldr d2, [x5, #56]
    fmadd d2, d0, d1, d2
    str d2, [x5, #56]

    add x2, x15, x3
    add x5, x5, #64
    sub x1, x1, #8
    b .Lacc2_block8_check

.Lacc2_tail:
    cbz x1, .Lacc2_done
    b ZYNUM_SYMBOL(zynum_blas_sve_dgemv_t_f64)

.Lacc2_done:
    ret

// void zynum_blas_sve_dgemv_t_f64_n16(
//     usize m, usize n, double alpha, const double *a, usize lda_bytes,
//     const double *x, double *y)
//
// Wider SVE transpose GEMV panel.  The main loop accumulates sixteen output
// columns per x vector.  Remaining columns tail-call the n8 kernel above.
.globl ZYNUM_SYMBOL(zynum_blas_sve_dgemv_t_f64_n16)
.p2align 2
ZYNUM_SYMBOL(zynum_blas_sve_dgemv_t_f64_n16):
    cbz x0, .Ln16_done
    cbz x1, .Ln16_done
    ptrue p1.d

.Ln16_block_check:
    cmp x1, #16
    b.lo .Ln16_tail

.Ln16_block:
    mov x8, x2
    add x9, x8, x3
    add x10, x9, x3
    add x11, x10, x3
    add x12, x11, x3
    add x13, x12, x3
    add x14, x13, x3
    add x15, x14, x3

    dup z0.d, #0
    dup z1.d, #0
    dup z2.d, #0
    dup z3.d, #0
    dup z4.d, #0
    dup z5.d, #0
    dup z6.d, #0
    dup z7.d, #0
    dup z8.d, #0
    dup z9.d, #0
    dup z10.d, #0
    dup z11.d, #0
    dup z12.d, #0
    dup z13.d, #0
    dup z14.d, #0
    dup z15.d, #0
    mov x16, #0

.Ln16_rows:
    whilelo p0.d, x16, x0
    b.none .Ln16_reduce
    ld1d { z16.d }, p0/z, [x4, x16, lsl #3]

    ld1d { z17.d }, p0/z, [x8, x16, lsl #3]
    fmla z0.d, p0/m, z17.d, z16.d
    ld1d { z17.d }, p0/z, [x9, x16, lsl #3]
    fmla z1.d, p0/m, z17.d, z16.d
    ld1d { z17.d }, p0/z, [x10, x16, lsl #3]
    fmla z2.d, p0/m, z17.d, z16.d
    ld1d { z17.d }, p0/z, [x11, x16, lsl #3]
    fmla z3.d, p0/m, z17.d, z16.d
    ld1d { z17.d }, p0/z, [x12, x16, lsl #3]
    fmla z4.d, p0/m, z17.d, z16.d
    ld1d { z17.d }, p0/z, [x13, x16, lsl #3]
    fmla z5.d, p0/m, z17.d, z16.d
    ld1d { z17.d }, p0/z, [x14, x16, lsl #3]
    fmla z6.d, p0/m, z17.d, z16.d
    ld1d { z17.d }, p0/z, [x15, x16, lsl #3]
    fmla z7.d, p0/m, z17.d, z16.d

    add x6, x15, x3
    ld1d { z17.d }, p0/z, [x6, x16, lsl #3]
    fmla z8.d, p0/m, z17.d, z16.d
    add x6, x6, x3
    ld1d { z17.d }, p0/z, [x6, x16, lsl #3]
    fmla z9.d, p0/m, z17.d, z16.d
    add x6, x6, x3
    ld1d { z17.d }, p0/z, [x6, x16, lsl #3]
    fmla z10.d, p0/m, z17.d, z16.d
    add x6, x6, x3
    ld1d { z17.d }, p0/z, [x6, x16, lsl #3]
    fmla z11.d, p0/m, z17.d, z16.d
    add x6, x6, x3
    ld1d { z17.d }, p0/z, [x6, x16, lsl #3]
    fmla z12.d, p0/m, z17.d, z16.d
    add x6, x6, x3
    ld1d { z17.d }, p0/z, [x6, x16, lsl #3]
    fmla z13.d, p0/m, z17.d, z16.d
    add x6, x6, x3
    ld1d { z17.d }, p0/z, [x6, x16, lsl #3]
    fmla z14.d, p0/m, z17.d, z16.d
    add x6, x6, x3
    ld1d { z17.d }, p0/z, [x6, x16, lsl #3]
    fmla z15.d, p0/m, z17.d, z16.d

    incd x16
    b .Ln16_rows

.Ln16_reduce:
    faddv d1, p1, z0.d
    ldr d2, [x5]
    fmadd d2, d0, d1, d2
    str d2, [x5]
    faddv d1, p1, z1.d
    ldr d2, [x5, #8]
    fmadd d2, d0, d1, d2
    str d2, [x5, #8]
    faddv d1, p1, z2.d
    ldr d2, [x5, #16]
    fmadd d2, d0, d1, d2
    str d2, [x5, #16]
    faddv d1, p1, z3.d
    ldr d2, [x5, #24]
    fmadd d2, d0, d1, d2
    str d2, [x5, #24]
    faddv d1, p1, z4.d
    ldr d2, [x5, #32]
    fmadd d2, d0, d1, d2
    str d2, [x5, #32]
    faddv d1, p1, z5.d
    ldr d2, [x5, #40]
    fmadd d2, d0, d1, d2
    str d2, [x5, #40]
    faddv d1, p1, z6.d
    ldr d2, [x5, #48]
    fmadd d2, d0, d1, d2
    str d2, [x5, #48]
    faddv d1, p1, z7.d
    ldr d2, [x5, #56]
    fmadd d2, d0, d1, d2
    str d2, [x5, #56]
    faddv d1, p1, z8.d
    ldr d2, [x5, #64]
    fmadd d2, d0, d1, d2
    str d2, [x5, #64]
    faddv d1, p1, z9.d
    ldr d2, [x5, #72]
    fmadd d2, d0, d1, d2
    str d2, [x5, #72]
    faddv d1, p1, z10.d
    ldr d2, [x5, #80]
    fmadd d2, d0, d1, d2
    str d2, [x5, #80]
    faddv d1, p1, z11.d
    ldr d2, [x5, #88]
    fmadd d2, d0, d1, d2
    str d2, [x5, #88]
    faddv d1, p1, z12.d
    ldr d2, [x5, #96]
    fmadd d2, d0, d1, d2
    str d2, [x5, #96]
    faddv d1, p1, z13.d
    ldr d2, [x5, #104]
    fmadd d2, d0, d1, d2
    str d2, [x5, #104]
    faddv d1, p1, z14.d
    ldr d2, [x5, #112]
    fmadd d2, d0, d1, d2
    str d2, [x5, #112]
    faddv d1, p1, z15.d
    ldr d2, [x5, #120]
    fmadd d2, d0, d1, d2
    str d2, [x5, #120]

    add x2, x15, x3, lsl #3
    add x2, x2, x3
    add x5, x5, #128
    sub x1, x1, #16
    b .Ln16_block_check

.Ln16_tail:
    cbz x1, .Ln16_done
    b ZYNUM_SYMBOL(zynum_blas_sve_dgemv_t_f64)

.Ln16_done:
    ret

// void zynum_blas_sme2_dgemv_n_f64_256x1(
//     usize m, usize n, double alpha, double beta, const double *a,
//     usize lda_bytes, const double *x, double *y)
//
// SME2/ZA no-transpose GEMV for full 256-row panels. The facade only calls this
// when the row count is a multiple of 256 and the streaming vector length is
// 64 bytes, so the kernel can avoid row-tail handling.
.globl ZYNUM_SYMBOL(zynum_blas_sme2_dgemv_n_f64_256x1)
.p2align 2
ZYNUM_SYMBOL(zynum_blas_sme2_dgemv_n_f64_256x1):
    cbz x0, .Lsme2_n_done_return
    cbz x1, .Lsme2_n_done_return
    ptrue pn8.d

    mov x6, x2              // Current 256-row A panel.
    mov x7, x5              // Current 256-row y panel.
    mov x14, x0             // Remaining rows.

.Lsme2_n_rows256:
    zero { za }
    mov x15, x6             // A column pointer for this row panel.
    mov x16, x4             // x pointer.
    mov x17, x1             // Remaining columns.

.Lsme2_n_cols:
    ldr d2, [x16], #8
    mov z4.d, d2
    mov x9, x15

    ld1d { z16.d - z19.d }, pn8/z, [x9]
    ld1d { z20.d - z23.d }, pn8/z, [x9, #4, mul vl]
    ld1d { z24.d - z27.d }, pn8/z, [x9, #8, mul vl]
    ld1d { z28.d - z31.d }, pn8/z, [x9, #12, mul vl]
    mov w8, #0
    mov w11, #8
    fmla za.d[w8, 0, vgx4], { z16.d - z19.d }, z4.d
    fmla za.d[w11, 0, vgx4], { z20.d - z23.d }, z4.d
    mov w8, #1
    mov w11, #9
    fmla za.d[w8, 0, vgx4], { z24.d - z27.d }, z4.d
    fmla za.d[w11, 0, vgx4], { z28.d - z31.d }, z4.d

    add x9, x9, #1024
    ld1d { z16.d - z19.d }, pn8/z, [x9]
    ld1d { z20.d - z23.d }, pn8/z, [x9, #4, mul vl]
    ld1d { z24.d - z27.d }, pn8/z, [x9, #8, mul vl]
    ld1d { z28.d - z31.d }, pn8/z, [x9, #12, mul vl]
    mov w8, #2
    mov w11, #10
    fmla za.d[w8, 0, vgx4], { z16.d - z19.d }, z4.d
    fmla za.d[w11, 0, vgx4], { z20.d - z23.d }, z4.d
    mov w8, #3
    mov w11, #11
    fmla za.d[w8, 0, vgx4], { z24.d - z27.d }, z4.d
    fmla za.d[w11, 0, vgx4], { z28.d - z31.d }, z4.d

    add x15, x15, x3
    subs x17, x17, #1
    b.ne .Lsme2_n_cols

    mov x9, x7
    mov w8, #0
    mov w11, #8
    mov { z16.d - z19.d }, za.d[w8, 0, vgx4]
    mov { z20.d - z23.d }, za.d[w11, 0, vgx4]
    zero { za0.d }
    ld1d { z24.d - z27.d }, pn8/z, [x9]
    ld1d { z28.d - z31.d }, pn8/z, [x9, #4, mul vl]
    mov z4.d, d1
    fmla za.d[w8, 0, vgx4], { z24.d - z27.d }, z4.d
    fmla za.d[w11, 0, vgx4], { z28.d - z31.d }, z4.d
    mov z4.d, d0
    fmla za.d[w8, 0, vgx4], { z16.d - z19.d }, z4.d
    fmla za.d[w11, 0, vgx4], { z20.d - z23.d }, z4.d

    mov w8, #1
    mov w11, #9
    mov { z16.d - z19.d }, za.d[w8, 0, vgx4]
    mov { z20.d - z23.d }, za.d[w11, 0, vgx4]
    zero { za1.d }
    ld1d { z24.d - z27.d }, pn8/z, [x9, #8, mul vl]
    ld1d { z28.d - z31.d }, pn8/z, [x9, #12, mul vl]
    mov z4.d, d1
    fmla za.d[w8, 0, vgx4], { z24.d - z27.d }, z4.d
    fmla za.d[w11, 0, vgx4], { z28.d - z31.d }, z4.d
    mov z4.d, d0
    fmla za.d[w8, 0, vgx4], { z16.d - z19.d }, z4.d
    fmla za.d[w11, 0, vgx4], { z20.d - z23.d }, z4.d

    mov w8, #2
    mov w11, #10
    mov { z16.d - z19.d }, za.d[w8, 0, vgx4]
    mov { z20.d - z23.d }, za.d[w11, 0, vgx4]
    zero { za2.d }
    ld1d { z24.d - z27.d }, pn8/z, [x9, #16, mul vl]
    ld1d { z28.d - z31.d }, pn8/z, [x9, #20, mul vl]
    mov z4.d, d1
    fmla za.d[w8, 0, vgx4], { z24.d - z27.d }, z4.d
    fmla za.d[w11, 0, vgx4], { z28.d - z31.d }, z4.d
    mov z4.d, d0
    fmla za.d[w8, 0, vgx4], { z16.d - z19.d }, z4.d
    fmla za.d[w11, 0, vgx4], { z20.d - z23.d }, z4.d

    mov w8, #3
    mov w11, #11
    mov { z16.d - z19.d }, za.d[w8, 0, vgx4]
    mov { z20.d - z23.d }, za.d[w11, 0, vgx4]
    zero { za3.d }
    ld1d { z24.d - z27.d }, pn8/z, [x9, #24, mul vl]
    ld1d { z28.d - z31.d }, pn8/z, [x9, #28, mul vl]
    mov z4.d, d1
    fmla za.d[w8, 0, vgx4], { z24.d - z27.d }, z4.d
    fmla za.d[w11, 0, vgx4], { z28.d - z31.d }, z4.d
    mov z4.d, d0
    fmla za.d[w8, 0, vgx4], { z16.d - z19.d }, z4.d
    fmla za.d[w11, 0, vgx4], { z20.d - z23.d }, z4.d

    mov w8, #0
    mov w11, #8
    mov { z16.d - z19.d }, za.d[w8, 0, vgx4]
    mov { z20.d - z23.d }, za.d[w11, 0, vgx4]
    st1d { z16.d - z19.d }, pn8, [x9]
    st1d { z20.d - z23.d }, pn8, [x9, #4, mul vl]
    mov w8, #1
    mov w11, #9
    mov { z16.d - z19.d }, za.d[w8, 0, vgx4]
    mov { z20.d - z23.d }, za.d[w11, 0, vgx4]
    st1d { z16.d - z19.d }, pn8, [x9, #8, mul vl]
    st1d { z20.d - z23.d }, pn8, [x9, #12, mul vl]
    mov w8, #2
    mov w11, #10
    mov { z16.d - z19.d }, za.d[w8, 0, vgx4]
    mov { z20.d - z23.d }, za.d[w11, 0, vgx4]
    st1d { z16.d - z19.d }, pn8, [x9, #16, mul vl]
    st1d { z20.d - z23.d }, pn8, [x9, #20, mul vl]
    mov w8, #3
    mov w11, #11
    mov { z16.d - z19.d }, za.d[w8, 0, vgx4]
    mov { z20.d - z23.d }, za.d[w11, 0, vgx4]
    st1d { z16.d - z19.d }, pn8, [x9, #24, mul vl]
    st1d { z20.d - z23.d }, pn8, [x9, #28, mul vl]

    add x6, x6, #2048
    add x7, x7, #2048
    subs x14, x14, #256
    b.ne .Lsme2_n_rows256

.Lsme2_n_done_return:
    ret

// void zynum_blas_sme2_dgemv_t_f64_8x32(
//     usize m, usize n, double alpha, double beta, const double *a,
//     usize lda_bytes, const double *x, double *y)
//
// SME2/ZA transpose GEMV for full 32-row x 8-column panels.  The facade only
// calls this when m is a multiple of 32 and n is a multiple of 8, matching the
// fast Accelerate shape for medium square matrices.
.globl ZYNUM_SYMBOL(zynum_blas_sme2_dgemv_t_f64_8x32)
.p2align 2
ZYNUM_SYMBOL(zynum_blas_sme2_dgemv_t_f64_8x32):
    cbz x0, .Lsme2_t_done_return
    cbz x1, .Lsme2_t_done_return
    ptrue pn8.d

    mov x7, x2              // Current 8-column A panel.
    mov x15, x1             // Remaining output columns.
    lsl x6, x3, #3          // lda_bytes * 8.

.Lsme2_t_cols:
    zero { za }
    mov x12, #0             // Row offset in elements.
    mov x10, x4             // x row pointer.
    mov x16, x7             // A column-0 row pointer for this panel.

.Lsme2_t_rows32:
    cmp x12, x0
    b.hs .Lsme2_t_reduce

    mov x9, x16
    ld1d { z4.d - z7.d }, pn8/z, [x10]
    mov w8, #0
    mov w11, #8

    ld1d { z16.d, z20.d, z24.d, z28.d }, pn8/z, [x9]
    add x9, x9, x3
    ld1d { z17.d, z21.d, z25.d, z29.d }, pn8/z, [x9]
    add x9, x9, x3
    ld1d { z18.d, z22.d, z26.d, z30.d }, pn8/z, [x9]
    add x9, x9, x3
    ld1d { z19.d, z23.d, z27.d, z31.d }, pn8/z, [x9]
    add x9, x9, x3
    fmla za.d[w8, 0, vgx4], { z16.d - z19.d }, z4.d
    fmla za.d[w11, 0, vgx4], { z20.d - z23.d }, z5.d
    fmla za.d[w8, 1, vgx4], { z24.d - z27.d }, z6.d
    fmla za.d[w11, 1, vgx4], { z28.d - z31.d }, z7.d

    ld1d { z16.d, z20.d, z24.d, z28.d }, pn8/z, [x9]
    add x9, x9, x3
    ld1d { z17.d, z21.d, z25.d, z29.d }, pn8/z, [x9]
    add x9, x9, x3
    ld1d { z18.d, z22.d, z26.d, z30.d }, pn8/z, [x9]
    add x9, x9, x3
    ld1d { z19.d, z23.d, z27.d, z31.d }, pn8/z, [x9]
    fmla za.d[w8, 2, vgx4], { z16.d - z19.d }, z4.d
    fmla za.d[w11, 2, vgx4], { z20.d - z23.d }, z5.d
    fmla za.d[w8, 3, vgx4], { z24.d - z27.d }, z6.d
    fmla za.d[w11, 3, vgx4], { z28.d - z31.d }, z7.d

    add x12, x12, #32
    add x10, x10, #256
    add x16, x16, #256
    b .Lsme2_t_rows32

.Lsme2_t_reduce:
    mov w8, #0
    mov w11, #8
    mov { z16.d - z19.d }, za.d[w8, 1, vgx4]
    mov { z20.d - z23.d }, za.d[w11, 1, vgx4]
    fadd za.d[w8, 0, vgx4], { z16.d - z19.d }
    fadd za.d[w11, 0, vgx4], { z20.d - z23.d }
    mov { z24.d - z27.d }, za.d[w8, 3, vgx4]
    mov { z28.d - z31.d }, za.d[w11, 3, vgx4]
    fadd za.d[w8, 2, vgx4], { z24.d - z27.d }
    fadd za.d[w11, 2, vgx4], { z28.d - z31.d }
    mov { z20.d - z23.d }, za.d[w11, 0, vgx4]
    fadd za.d[w8, 0, vgx4], { z20.d - z23.d }
    mov { z16.d - z19.d }, za.d[w11, 2, vgx4]
    fadd za.d[w8, 2, vgx4], { z16.d - z19.d }
    mov { z16.d - z19.d }, za.d[w8, 2, vgx4]
    mov za.d[w11, 0, vgx4], { z16.d - z19.d }
    mov w12, #0
    mov { z16.d - z19.d }, za0v.d[w12, 0:3]
    mov w12, #4
    mov { z20.d - z23.d }, za0v.d[w12, 0:3]
    zero { za1.d }
    fadd za.d[w8, 1, vgx4], { z16.d - z19.d }
    fadd za.d[w8, 1, vgx4], { z20.d - z23.d }
    mov { z20.d - z23.d }, za.d[w8, 1, vgx4]
    fadd z20.d, z20.d, z21.d
    fadd z20.d, z20.d, z22.d
    fadd z20.d, z20.d, z23.d
    adr x13, .Lsme2_t_permute
    ldr z4, [x13]
    tbl z20.d, { z20.d }, z4.d

    mov x13, #8
    whilelo p0.d, xzr, x13
    ld1d { z16.d }, p0/z, [x5]
    mov z17.d, d1
    fmul z16.d, z16.d, z17.d
    mov z17.d, d0
    fmla z16.d, p0/m, z20.d, z17.d
    st1d { z16.d }, p0, [x5]

    add x5, x5, #64
    add x7, x7, x6
    subs x15, x15, #8
    b.ne .Lsme2_t_cols

.Lsme2_t_done_return:
    ret

.p2align 3
.Lsme2_t_permute:
    .quad 0, 2, 4, 6, 1, 3, 5, 7

// void zynum_blas_asimd_dgemv_t_f64_n1(
//     usize m, usize n, double alpha, const double *a, usize lda_bytes,
//     const double *x, double *y)
//
// Single-column ASIMD transpose GEMV.  This streams one A column and x at a
// time with a 32-double row unroll, matching OpenBLAS' prefetch-friendly hot
// loop shape for medium square matrices.
.globl ZYNUM_SYMBOL(zynum_blas_asimd_dgemv_t_f64_n1)
.p2align 2
ZYNUM_SYMBOL(zynum_blas_asimd_dgemv_t_f64_n1):
    cbz x0, .Lasimd_t_n1_done
    cbz x1, .Lasimd_t_n1_done

.Lasimd_t_n1_col:
    movi v1.2d, #0
    movi v2.2d, #0
    movi v3.2d, #0
    movi v4.2d, #0
    mov x9, x2
    mov x10, x4

    lsr x12, x0, #5
    cbz x12, .Lasimd_t_n1_tail8_count

.Lasimd_t_n1_loop32:
    ldp q16, q17, [x9]
    ldp q18, q19, [x9, #32]
    ldp q20, q21, [x10]
    ldp q22, q23, [x10, #32]
    fmla v1.2d, v16.2d, v20.2d
    fmla v2.2d, v17.2d, v21.2d
    fmla v3.2d, v18.2d, v22.2d
    fmla v4.2d, v19.2d, v23.2d
    prfm pldl1keep, [x9, #768]
    prfm pldl1keep, [x10, #768]

    ldp q16, q17, [x9, #64]
    ldp q18, q19, [x9, #96]
    ldp q20, q21, [x10, #64]
    ldp q22, q23, [x10, #96]
    fmla v1.2d, v16.2d, v20.2d
    fmla v2.2d, v17.2d, v21.2d
    fmla v3.2d, v18.2d, v22.2d
    fmla v4.2d, v19.2d, v23.2d

    ldp q16, q17, [x9, #128]
    ldp q18, q19, [x9, #160]
    ldp q20, q21, [x10, #128]
    ldp q22, q23, [x10, #160]
    fmla v1.2d, v16.2d, v20.2d
    fmla v2.2d, v17.2d, v21.2d
    fmla v3.2d, v18.2d, v22.2d
    fmla v4.2d, v19.2d, v23.2d
    prfm pldl1keep, [x9, #1024]
    prfm pldl1keep, [x10, #1024]

    ldp q16, q17, [x9, #192]
    ldp q18, q19, [x9, #224]
    ldp q20, q21, [x10, #192]
    ldp q22, q23, [x10, #224]
    fmla v1.2d, v16.2d, v20.2d
    fmla v2.2d, v17.2d, v21.2d
    fmla v3.2d, v18.2d, v22.2d
    fmla v4.2d, v19.2d, v23.2d

    add x9, x9, #256
    add x10, x10, #256
    subs x12, x12, #1
    b.ne .Lasimd_t_n1_loop32

.Lasimd_t_n1_tail8_count:
    and x12, x0, #31
    lsr x12, x12, #3
    cbz x12, .Lasimd_t_n1_reduce

.Lasimd_t_n1_tail8:
    ldp q16, q17, [x9]
    ldp q18, q19, [x9, #32]
    ldp q20, q21, [x10]
    ldp q22, q23, [x10, #32]
    fmla v1.2d, v16.2d, v20.2d
    fmla v2.2d, v17.2d, v21.2d
    fmla v3.2d, v18.2d, v22.2d
    fmla v4.2d, v19.2d, v23.2d
    add x9, x9, #64
    add x10, x10, #64
    subs x12, x12, #1
    b.ne .Lasimd_t_n1_tail8

.Lasimd_t_n1_reduce:
    faddp d1, v1.2d
    faddp d2, v2.2d
    faddp d3, v3.2d
    faddp d4, v4.2d
    fadd d1, d1, d2
    fadd d3, d3, d4
    fadd d1, d1, d3

    and x12, x0, #7
    cbz x12, .Lasimd_t_n1_store

.Lasimd_t_n1_tail1:
    ldr d16, [x9], #8
    ldr d17, [x10], #8
    fmadd d1, d16, d17, d1
    subs x12, x12, #1
    b.ne .Lasimd_t_n1_tail1

.Lasimd_t_n1_store:
    ldr d2, [x5]
    fmadd d2, d0, d1, d2
    str d2, [x5]
    add x2, x2, x3
    add x5, x5, #8
    subs x1, x1, #1
    b.ne .Lasimd_t_n1_col

.Lasimd_t_n1_done:
    ret

// void zynum_blas_asimd_dgemv_t_f64(
//     usize m, usize n, double alpha, const double *a, usize lda_bytes,
//     const double *x, double *y)
//
// ASIMD transpose GEMV for 128-bit machines.  It computes eight output columns
// per block and unrolls the row loop by four, so each pair of x loads feeds
// eight independent dot products.
.globl ZYNUM_SYMBOL(zynum_blas_asimd_dgemv_t_f64)
.p2align 2
ZYNUM_SYMBOL(zynum_blas_asimd_dgemv_t_f64):
    cbz x0, .Lasimd_t_done
    cbz x1, .Lasimd_t_done

.Lasimd_t_block8_check:
    cmp x1, #8
    b.lo .Lasimd_t_tail1_check

.Lasimd_t_block8:
    mov x8, x2
    add x9, x8, x3
    add x10, x9, x3
    add x11, x10, x3
    add x12, x11, x3
    add x13, x12, x3
    add x14, x13, x3
    add x15, x14, x3

    movi v1.2d, #0
    movi v2.2d, #0
    movi v3.2d, #0
    movi v4.2d, #0
    movi v5.2d, #0
    movi v6.2d, #0
    movi v7.2d, #0
    movi v20.2d, #0
    mov x6, x4
    mov x7, x0

.Lasimd_t_rows4_check:
    cmp x7, #4
    b.lo .Lasimd_t_rows2_check

.Lasimd_t_rows4:
    ldr q16, [x6]
    ldr q17, [x6, #16]

    ldr q18, [x8]
    ldr q19, [x8, #16]
    fmla v1.2d, v18.2d, v16.2d
    fmla v1.2d, v19.2d, v17.2d
    add x8, x8, #32

    ldr q18, [x9]
    ldr q19, [x9, #16]
    fmla v2.2d, v18.2d, v16.2d
    fmla v2.2d, v19.2d, v17.2d
    add x9, x9, #32

    ldr q18, [x10]
    ldr q19, [x10, #16]
    fmla v3.2d, v18.2d, v16.2d
    fmla v3.2d, v19.2d, v17.2d
    add x10, x10, #32

    ldr q18, [x11]
    ldr q19, [x11, #16]
    fmla v4.2d, v18.2d, v16.2d
    fmla v4.2d, v19.2d, v17.2d
    add x11, x11, #32

    ldr q18, [x12]
    ldr q19, [x12, #16]
    fmla v5.2d, v18.2d, v16.2d
    fmla v5.2d, v19.2d, v17.2d
    add x12, x12, #32

    ldr q18, [x13]
    ldr q19, [x13, #16]
    fmla v6.2d, v18.2d, v16.2d
    fmla v6.2d, v19.2d, v17.2d
    add x13, x13, #32

    ldr q18, [x14]
    ldr q19, [x14, #16]
    fmla v7.2d, v18.2d, v16.2d
    fmla v7.2d, v19.2d, v17.2d
    add x14, x14, #32

    ldr q18, [x15]
    ldr q19, [x15, #16]
    fmla v20.2d, v18.2d, v16.2d
    fmla v20.2d, v19.2d, v17.2d
    add x15, x15, #32

    add x6, x6, #32
    sub x7, x7, #4
    b .Lasimd_t_rows4_check

.Lasimd_t_rows2_check:
    cmp x7, #2
    b.lo .Lasimd_t_rows1_check

    ldr q16, [x6]
    ldr q18, [x8], #16
    fmla v1.2d, v18.2d, v16.2d
    ldr q18, [x9], #16
    fmla v2.2d, v18.2d, v16.2d
    ldr q18, [x10], #16
    fmla v3.2d, v18.2d, v16.2d
    ldr q18, [x11], #16
    fmla v4.2d, v18.2d, v16.2d
    ldr q18, [x12], #16
    fmla v5.2d, v18.2d, v16.2d
    ldr q18, [x13], #16
    fmla v6.2d, v18.2d, v16.2d
    ldr q18, [x14], #16
    fmla v7.2d, v18.2d, v16.2d
    ldr q18, [x15], #16
    fmla v20.2d, v18.2d, v16.2d
    add x6, x6, #16
    sub x7, x7, #2

.Lasimd_t_rows1_check:
    cbz x7, .Lasimd_t_block8_reduce

    movi v16.2d, #0
    ldr d16, [x6]
    movi v18.2d, #0
    ldr d18, [x8]
    fmla v1.2d, v18.2d, v16.2d
    movi v18.2d, #0
    ldr d18, [x9]
    fmla v2.2d, v18.2d, v16.2d
    movi v18.2d, #0
    ldr d18, [x10]
    fmla v3.2d, v18.2d, v16.2d
    movi v18.2d, #0
    ldr d18, [x11]
    fmla v4.2d, v18.2d, v16.2d
    movi v18.2d, #0
    ldr d18, [x12]
    fmla v5.2d, v18.2d, v16.2d
    movi v18.2d, #0
    ldr d18, [x13]
    fmla v6.2d, v18.2d, v16.2d
    movi v18.2d, #0
    ldr d18, [x14]
    fmla v7.2d, v18.2d, v16.2d
    movi v18.2d, #0
    ldr d18, [x15]
    fmla v20.2d, v18.2d, v16.2d

.Lasimd_t_block8_reduce:
    faddp d16, v1.2d
    ldr d17, [x5]
    fmadd d17, d0, d16, d17
    str d17, [x5]
    faddp d16, v2.2d
    ldr d17, [x5, #8]
    fmadd d17, d0, d16, d17
    str d17, [x5, #8]
    faddp d16, v3.2d
    ldr d17, [x5, #16]
    fmadd d17, d0, d16, d17
    str d17, [x5, #16]
    faddp d16, v4.2d
    ldr d17, [x5, #24]
    fmadd d17, d0, d16, d17
    str d17, [x5, #24]
    faddp d16, v5.2d
    ldr d17, [x5, #32]
    fmadd d17, d0, d16, d17
    str d17, [x5, #32]
    faddp d16, v6.2d
    ldr d17, [x5, #40]
    fmadd d17, d0, d16, d17
    str d17, [x5, #40]
    faddp d16, v7.2d
    ldr d17, [x5, #48]
    fmadd d17, d0, d16, d17
    str d17, [x5, #48]
    faddp d16, v20.2d
    ldr d17, [x5, #56]
    fmadd d17, d0, d16, d17
    str d17, [x5, #56]

    add x2, x2, x3, lsl #3
    add x5, x5, #64
    sub x1, x1, #8
    b .Lasimd_t_block8_check

.Lasimd_t_tail1_check:
    cbz x1, .Lasimd_t_done

.Lasimd_t_tail1:
    mov x8, x2
    mov x6, x4
    mov x7, x0
    movi v1.2d, #0

.Lasimd_t_tail1_rows4_check:
    cmp x7, #4
    b.lo .Lasimd_t_tail1_rows2_check
    ldr q16, [x6]
    ldr q17, [x6, #16]
    ldr q18, [x8]
    ldr q19, [x8, #16]
    fmla v1.2d, v18.2d, v16.2d
    fmla v1.2d, v19.2d, v17.2d
    add x6, x6, #32
    add x8, x8, #32
    sub x7, x7, #4
    b .Lasimd_t_tail1_rows4_check

.Lasimd_t_tail1_rows2_check:
    cmp x7, #2
    b.lo .Lasimd_t_tail1_rows1_check
    ldr q16, [x6]
    ldr q18, [x8]
    fmla v1.2d, v18.2d, v16.2d
    add x6, x6, #16
    add x8, x8, #16
    sub x7, x7, #2

.Lasimd_t_tail1_rows1_check:
    cbz x7, .Lasimd_t_tail1_reduce
    movi v16.2d, #0
    movi v18.2d, #0
    ldr d16, [x6]
    ldr d18, [x8]
    fmla v1.2d, v18.2d, v16.2d

.Lasimd_t_tail1_reduce:
    faddp d16, v1.2d
    ldr d17, [x5]
    fmadd d17, d0, d16, d17
    str d17, [x5]
    add x2, x2, x3
    add x5, x5, #8
    subs x1, x1, #1
    b.ne .Lasimd_t_tail1

.Lasimd_t_done:
    ret

// void zynum_blas_asimd_dgemv_n_f64(
//     usize m, usize n, double alpha, const double *a, usize lda_bytes,
//     const double *x, double *y)
//
// Computes y[i] += alpha * sum_j A[i, j] * x[j] for column-major A.  Rows are
// processed in 8/4/2/1 panels so each y element is loaded and stored once per
// panel while A is streamed by columns.
.globl ZYNUM_SYMBOL(zynum_blas_asimd_dgemv_n_f64)
.p2align 2
ZYNUM_SYMBOL(zynum_blas_asimd_dgemv_n_f64):
    cbz x0, .Ln_done
    cbz x1, .Ln_done

    mov x6, x0

.Ln_rows8_check:
    cmp x6, #8
    b.lo .Ln_rows4_check

.Ln_rows8:
    mov x7, x2
    mov x8, x4
    mov x9, x1
    movi v1.2d, #0
    movi v2.2d, #0
    movi v3.2d, #0
    movi v4.2d, #0

.Ln_rows8_cols:
    ldr d16, [x8], #8
    fmul d16, d16, d0
    dup v16.2d, v16.d[0]
    ldr q20, [x7]
    ldr q21, [x7, #16]
    ldr q22, [x7, #32]
    ldr q23, [x7, #48]
    fmla v1.2d, v20.2d, v16.2d
    fmla v2.2d, v21.2d, v16.2d
    fmla v3.2d, v22.2d, v16.2d
    fmla v4.2d, v23.2d, v16.2d
    add x7, x7, x3
    subs x9, x9, #1
    b.ne .Ln_rows8_cols

    ldr q20, [x5]
    ldr q21, [x5, #16]
    ldr q22, [x5, #32]
    ldr q23, [x5, #48]
    fadd v20.2d, v20.2d, v1.2d
    fadd v21.2d, v21.2d, v2.2d
    fadd v22.2d, v22.2d, v3.2d
    fadd v23.2d, v23.2d, v4.2d
    str q20, [x5]
    str q21, [x5, #16]
    str q22, [x5, #32]
    str q23, [x5, #48]
    add x2, x2, #64
    add x5, x5, #64
    sub x6, x6, #8
    b .Ln_rows8_check

.Ln_rows4_check:
    cmp x6, #4
    b.lo .Ln_rows2_check

.Ln_rows4:
    mov x7, x2
    mov x8, x4
    mov x9, x1
    movi v1.2d, #0
    movi v2.2d, #0

.Ln_rows4_cols:
    ldr d16, [x8], #8
    fmul d16, d16, d0
    dup v16.2d, v16.d[0]
    ldr q20, [x7]
    ldr q21, [x7, #16]
    fmla v1.2d, v20.2d, v16.2d
    fmla v2.2d, v21.2d, v16.2d
    add x7, x7, x3
    subs x9, x9, #1
    b.ne .Ln_rows4_cols

    ldr q20, [x5]
    ldr q21, [x5, #16]
    fadd v20.2d, v20.2d, v1.2d
    fadd v21.2d, v21.2d, v2.2d
    str q20, [x5]
    str q21, [x5, #16]
    add x2, x2, #32
    add x5, x5, #32
    sub x6, x6, #4

.Ln_rows2_check:
    cmp x6, #2
    b.lo .Ln_rows1_check

.Ln_rows2:
    mov x7, x2
    mov x8, x4
    mov x9, x1
    movi v1.2d, #0

.Ln_rows2_cols:
    ldr d16, [x8], #8
    fmul d16, d16, d0
    dup v16.2d, v16.d[0]
    ldr q20, [x7]
    fmla v1.2d, v20.2d, v16.2d
    add x7, x7, x3
    subs x9, x9, #1
    b.ne .Ln_rows2_cols

    ldr q20, [x5]
    fadd v20.2d, v20.2d, v1.2d
    str q20, [x5]
    add x2, x2, #16
    add x5, x5, #16
    sub x6, x6, #2

.Ln_rows1_check:
    cbz x6, .Ln_done

.Ln_rows1:
    mov x7, x2
    mov x8, x4
    mov x9, x1
    fmov d1, xzr

.Ln_rows1_cols:
    ldr d16, [x8], #8
    fmul d16, d16, d0
    ldr d17, [x7]
    fmadd d1, d17, d16, d1
    add x7, x7, x3
    subs x9, x9, #1
    b.ne .Ln_rows1_cols

    ldr d18, [x5]
    fadd d18, d18, d1
    str d18, [x5]

.Ln_done:
    ret

// void zynum_blas_asimd_dcopy_f64(usize n, const double *x, double *y)
.globl ZYNUM_SYMBOL(zynum_blas_asimd_dcopy_f64)
.p2align 2
ZYNUM_SYMBOL(zynum_blas_asimd_dcopy_f64):
    cbz x0, .Lcopy_done

.Lcopy32_check:
    cmp x0, #32
    b.lo .Lcopy16_check

.Lcopy32:
    ldp q0, q1, [x1]
    ldp q2, q3, [x1, #32]
    ldp q4, q5, [x1, #64]
    ldp q6, q7, [x1, #96]
    stp q0, q1, [x2]
    stp q2, q3, [x2, #32]
    stp q4, q5, [x2, #64]
    stp q6, q7, [x2, #96]
    ldp q0, q1, [x1, #128]
    ldp q2, q3, [x1, #160]
    ldp q4, q5, [x1, #192]
    ldp q6, q7, [x1, #224]
    stp q0, q1, [x2, #128]
    stp q2, q3, [x2, #160]
    stp q4, q5, [x2, #192]
    stp q6, q7, [x2, #224]
    add x1, x1, #256
    add x2, x2, #256
    sub x0, x0, #32
    b .Lcopy32_check

.Lcopy16_check:
    cmp x0, #16
    b.lo .Lcopy8_check

.Lcopy16:
    ldp q0, q1, [x1]
    ldp q2, q3, [x1, #32]
    ldp q4, q5, [x1, #64]
    ldp q6, q7, [x1, #96]
    stp q0, q1, [x2]
    stp q2, q3, [x2, #32]
    stp q4, q5, [x2, #64]
    stp q6, q7, [x2, #96]
    add x1, x1, #128
    add x2, x2, #128
    sub x0, x0, #16
    b .Lcopy16_check

.Lcopy8_check:
    cmp x0, #8
    b.lo .Lcopy4_check
    ldp q0, q1, [x1]
    ldp q2, q3, [x1, #32]
    stp q0, q1, [x2]
    stp q2, q3, [x2, #32]
    add x1, x1, #64
    add x2, x2, #64
    sub x0, x0, #8

.Lcopy4_check:
    cmp x0, #4
    b.lo .Lcopy2_check
    ldp q0, q1, [x1]
    stp q0, q1, [x2]
    add x1, x1, #32
    add x2, x2, #32
    sub x0, x0, #4

.Lcopy2_check:
    cmp x0, #2
    b.lo .Lcopy1_check
    ldr q0, [x1]
    str q0, [x2]
    add x1, x1, #16
    add x2, x2, #16
    sub x0, x0, #2

.Lcopy1_check:
    cbz x0, .Lcopy_done
    ldr d0, [x1]
    str d0, [x2]

.Lcopy_done:
    ret

// void zynum_blas_sve_dcopy_f64(usize n, const double *x, double *y)
//
// SVE copy uses full-vector groups for the main body and a single predicated
// tail.  The AArch64 facade decides the length gate so this routine only owns
// the ISA-specific prologue/epilogue.
.globl ZYNUM_SYMBOL(zynum_blas_sve_dcopy_f64)
.p2align 2
ZYNUM_SYMBOL(zynum_blas_sve_dcopy_f64):
    cbz x0, .Lsve_copy_done
    ptrue p0.d
    cntd x6
    lsl x7, x6, #2

.Lsve_copy4_check:
    cmp x0, x7
    b.lo .Lsve_copy_tail
    ld1d { z0.d }, p0/z, [x1]
    ld1d { z1.d }, p0/z, [x1, #1, MUL VL]
    ld1d { z2.d }, p0/z, [x1, #2, MUL VL]
    ld1d { z3.d }, p0/z, [x1, #3, MUL VL]
    st1d { z0.d }, p0, [x2]
    st1d { z1.d }, p0, [x2, #1, MUL VL]
    st1d { z2.d }, p0, [x2, #2, MUL VL]
    st1d { z3.d }, p0, [x2, #3, MUL VL]
    addvl x1, x1, #4
    addvl x2, x2, #4
    sub x0, x0, x7
    b .Lsve_copy4_check

.Lsve_copy_tail:
    cbz x0, .Lsve_copy_done
    mov x9, #0
    whilelo p1.d, x9, x0
    ld1d { z0.d }, p1/z, [x1]
    st1d { z0.d }, p1, [x2]

.Lsve_copy_done:
    ret

// void zynum_blas_asimd_dscal_f64(usize n, double alpha, double *x)
.globl ZYNUM_SYMBOL(zynum_blas_asimd_dscal_f64)
.p2align 2
ZYNUM_SYMBOL(zynum_blas_asimd_dscal_f64):
    cbz x0, .Lscal_done
    dup v0.2d, v0.d[0]

.Lscal8_check:
    cmp x0, #8
    b.lo .Lscal4_check

.Lscal8:
    ldr q1, [x1]
    ldr q2, [x1, #16]
    ldr q3, [x1, #32]
    ldr q4, [x1, #48]
    fmul v1.2d, v1.2d, v0.2d
    fmul v2.2d, v2.2d, v0.2d
    fmul v3.2d, v3.2d, v0.2d
    fmul v4.2d, v4.2d, v0.2d
    str q1, [x1]
    str q2, [x1, #16]
    str q3, [x1, #32]
    str q4, [x1, #48]
    add x1, x1, #64
    sub x0, x0, #8
    b .Lscal8_check

.Lscal4_check:
    cmp x0, #4
    b.lo .Lscal2_check
    ldr q1, [x1]
    ldr q2, [x1, #16]
    fmul v1.2d, v1.2d, v0.2d
    fmul v2.2d, v2.2d, v0.2d
    str q1, [x1]
    str q2, [x1, #16]
    add x1, x1, #32
    sub x0, x0, #4

.Lscal2_check:
    cmp x0, #2
    b.lo .Lscal1_check
    ldr q1, [x1]
    fmul v1.2d, v1.2d, v0.2d
    str q1, [x1]
    add x1, x1, #16
    sub x0, x0, #2

.Lscal1_check:
    cbz x0, .Lscal_done
    ldr d1, [x1]
    fmul d1, d1, d0
    str d1, [x1]

.Lscal_done:
    ret

// void zynum_blas_sve_dscal_f64(usize n, double alpha, double *x)
.globl ZYNUM_SYMBOL(zynum_blas_sve_dscal_f64)
.p2align 2
ZYNUM_SYMBOL(zynum_blas_sve_dscal_f64):
    cbz x0, .Lsve_scal_done
    ptrue p0.d
    mov z4.d, d0
    cntd x6
    lsl x7, x6, #2

.Lsve_scal4_check:
    cmp x0, x7
    b.lo .Lsve_scal_tail
    ld1d { z0.d }, p0/z, [x1]
    ld1d { z1.d }, p0/z, [x1, #1, MUL VL]
    ld1d { z2.d }, p0/z, [x1, #2, MUL VL]
    ld1d { z3.d }, p0/z, [x1, #3, MUL VL]
    fmul z0.d, p0/m, z0.d, z4.d
    fmul z1.d, p0/m, z1.d, z4.d
    fmul z2.d, p0/m, z2.d, z4.d
    fmul z3.d, p0/m, z3.d, z4.d
    st1d { z0.d }, p0, [x1]
    st1d { z1.d }, p0, [x1, #1, MUL VL]
    st1d { z2.d }, p0, [x1, #2, MUL VL]
    st1d { z3.d }, p0, [x1, #3, MUL VL]
    addvl x1, x1, #4
    sub x0, x0, x7
    b .Lsve_scal4_check

.Lsve_scal_tail:
    cbz x0, .Lsve_scal_done
    mov x8, #0
    whilelo p1.d, x8, x0
    ld1d { z0.d }, p1/z, [x1]
    fmul z0.d, p1/m, z0.d, z4.d
    st1d { z0.d }, p1, [x1]

.Lsve_scal_done:
    ret

// void zynum_blas_sme_dscal_f64(usize n, double alpha, double *x)
.globl ZYNUM_SYMBOL(zynum_blas_sme_dscal_f64)
.p2align 2
ZYNUM_SYMBOL(zynum_blas_sme_dscal_f64):
    cbz x0, .Lsme_scal_done_return
    fmov x9, d0
    fmov d4, x9
    mov z4.d, d4
    ptrue p0.d
    cntd x6
    lsl x7, x6, #2

.Lsme_scal4_check:
    cmp x0, x7
    b.lo .Lsme_scal_tail
    ld1d { z0.d }, p0/z, [x1]
    ld1d { z1.d }, p0/z, [x1, #1, MUL VL]
    ld1d { z2.d }, p0/z, [x1, #2, MUL VL]
    ld1d { z3.d }, p0/z, [x1, #3, MUL VL]
    fmul z0.d, p0/m, z0.d, z4.d
    fmul z1.d, p0/m, z1.d, z4.d
    fmul z2.d, p0/m, z2.d, z4.d
    fmul z3.d, p0/m, z3.d, z4.d
    st1d { z0.d }, p0, [x1]
    st1d { z1.d }, p0, [x1, #1, MUL VL]
    st1d { z2.d }, p0, [x1, #2, MUL VL]
    st1d { z3.d }, p0, [x1, #3, MUL VL]
    addvl x1, x1, #4
    sub x0, x0, x7
    b .Lsme_scal4_check

.Lsme_scal_tail:
    cbz x0, .Lsme_scal_done
    mov x8, #0
    whilelo p1.d, x8, x0
    ld1d { z0.d }, p1/z, [x1]
    fmul z0.d, p1/m, z0.d, z4.d
    st1d { z0.d }, p1, [x1]

.Lsme_scal_done:

.Lsme_scal_done_return:
    ret

// double zynum_blas_sme_ddot_f64(usize n, const double *x, const double *y)
.globl ZYNUM_SYMBOL(zynum_blas_sme_ddot_f64)
.p2align 2
ZYNUM_SYMBOL(zynum_blas_sme_ddot_f64):
    cbz x0, .Lsme_ddot_zero
    ptrue p0.d
    cntd x6
    lsl x7, x6, #2
    dup z0.d, #0
    dup z1.d, #0
    dup z2.d, #0
    dup z3.d, #0

.Lsme_ddot4_check:
    cmp x0, x7
    b.lo .Lsme_ddot_tail
    ld1d { z4.d }, p0/z, [x1]
    ld1d { z5.d }, p0/z, [x1, #1, MUL VL]
    ld1d { z6.d }, p0/z, [x1, #2, MUL VL]
    ld1d { z7.d }, p0/z, [x1, #3, MUL VL]
    ld1d { z16.d }, p0/z, [x2]
    ld1d { z17.d }, p0/z, [x2, #1, MUL VL]
    ld1d { z18.d }, p0/z, [x2, #2, MUL VL]
    ld1d { z19.d }, p0/z, [x2, #3, MUL VL]
    fmla z0.d, p0/m, z4.d, z16.d
    fmla z1.d, p0/m, z5.d, z17.d
    fmla z2.d, p0/m, z6.d, z18.d
    fmla z3.d, p0/m, z7.d, z19.d
    addvl x1, x1, #4
    addvl x2, x2, #4
    sub x0, x0, x7
    b .Lsme_ddot4_check

.Lsme_ddot_tail:
    cbz x0, .Lsme_ddot_reduce
    mov x8, #0
    whilelo p1.d, x8, x0
    ld1d { z4.d }, p1/z, [x1]
    ld1d { z16.d }, p1/z, [x2]
    fmla z0.d, p1/m, z4.d, z16.d
    cmp x0, x6
    b.ls .Lsme_ddot_reduce
    addvl x1, x1, #1
    addvl x2, x2, #1
    sub x0, x0, x6
    b .Lsme_ddot_tail

.Lsme_ddot_reduce:
    fadd z0.d, z0.d, z1.d
    fadd z2.d, z2.d, z3.d
    fadd z0.d, z0.d, z2.d
    faddv d16, p0, z0.d
    fmov x9, d16
    fmov d0, x9
    ret

.Lsme_ddot_zero:
    fmov d0, xzr
    ret

// void zynum_blas_asimd_daxpy_f64(usize n, double alpha, const double *x, double *y)
.globl ZYNUM_SYMBOL(zynum_blas_asimd_daxpy_f64)
.p2align 2
ZYNUM_SYMBOL(zynum_blas_asimd_daxpy_f64):
    cbz x0, .Laxpy_done
    dup v0.2d, v0.d[0]

.Laxpy8_check:
    cmp x0, #8
    b.lo .Laxpy4_check

.Laxpy8:
    ldr q1, [x1]
    ldr q2, [x1, #16]
    ldr q3, [x1, #32]
    ldr q4, [x1, #48]
    ldr q5, [x2]
    ldr q6, [x2, #16]
    ldr q7, [x2, #32]
    ldr q16, [x2, #48]
    fmla v5.2d, v1.2d, v0.2d
    fmla v6.2d, v2.2d, v0.2d
    fmla v7.2d, v3.2d, v0.2d
    fmla v16.2d, v4.2d, v0.2d
    str q5, [x2]
    str q6, [x2, #16]
    str q7, [x2, #32]
    str q16, [x2, #48]
    add x1, x1, #64
    add x2, x2, #64
    sub x0, x0, #8
    b .Laxpy8_check

.Laxpy4_check:
    cmp x0, #4
    b.lo .Laxpy2_check
    ldr q1, [x1]
    ldr q2, [x1, #16]
    ldr q5, [x2]
    ldr q6, [x2, #16]
    fmla v5.2d, v1.2d, v0.2d
    fmla v6.2d, v2.2d, v0.2d
    str q5, [x2]
    str q6, [x2, #16]
    add x1, x1, #32
    add x2, x2, #32
    sub x0, x0, #4

.Laxpy2_check:
    cmp x0, #2
    b.lo .Laxpy1_check
    ldr q1, [x1]
    ldr q5, [x2]
    fmla v5.2d, v1.2d, v0.2d
    str q5, [x2]
    add x1, x1, #16
    add x2, x2, #16
    sub x0, x0, #2

.Laxpy1_check:
    cbz x0, .Laxpy_done
    ldr d1, [x1]
    ldr d5, [x2]
    fmadd d5, d1, d0, d5
    str d5, [x2]

.Laxpy_done:
    ret

// double zynum_blas_asimd_ddot_f64(usize n, const double *x, const double *y)
.globl ZYNUM_SYMBOL(zynum_blas_asimd_ddot_f64)
.p2align 2
ZYNUM_SYMBOL(zynum_blas_asimd_ddot_f64):
    fmov d0, xzr
    cbz x0, .Ldot_done
    movi v1.2d, #0
    movi v2.2d, #0
    movi v3.2d, #0
    movi v4.2d, #0

.Ldot8_check:
    cmp x0, #8
    b.lo .Ldot4_check

.Ldot8:
    ldr q5, [x1]
    ldr q6, [x1, #16]
    ldr q7, [x1, #32]
    ldr q16, [x1, #48]
    ldr q17, [x2]
    ldr q18, [x2, #16]
    ldr q19, [x2, #32]
    ldr q20, [x2, #48]
    fmla v1.2d, v5.2d, v17.2d
    fmla v2.2d, v6.2d, v18.2d
    fmla v3.2d, v7.2d, v19.2d
    fmla v4.2d, v16.2d, v20.2d
    add x1, x1, #64
    add x2, x2, #64
    sub x0, x0, #8
    b .Ldot8_check

.Ldot4_check:
    cmp x0, #4
    b.lo .Ldot2_check
    ldr q5, [x1]
    ldr q6, [x1, #16]
    ldr q17, [x2]
    ldr q18, [x2, #16]
    fmla v1.2d, v5.2d, v17.2d
    fmla v2.2d, v6.2d, v18.2d
    add x1, x1, #32
    add x2, x2, #32
    sub x0, x0, #4

.Ldot2_check:
    cmp x0, #2
    b.lo .Ldot_reduce
    ldr q5, [x1]
    ldr q17, [x2]
    fmla v1.2d, v5.2d, v17.2d
    add x1, x1, #16
    add x2, x2, #16
    sub x0, x0, #2

.Ldot_reduce:
    fadd v1.2d, v1.2d, v2.2d
    fadd v3.2d, v3.2d, v4.2d
    fadd v1.2d, v1.2d, v3.2d
    faddp d0, v1.2d
    cbz x0, .Ldot_done
    ldr d5, [x1]
    ldr d17, [x2]
    fmadd d0, d5, d17, d0

.Ldot_done:
    ret

// double zynum_blas_asimd_dasum_f64(usize n, const double *x)
.globl ZYNUM_SYMBOL(zynum_blas_asimd_dasum_f64)
.p2align 2
ZYNUM_SYMBOL(zynum_blas_asimd_dasum_f64):
    fmov d0, xzr
    cbz x0, .Lasum_done
    movi v1.2d, #0
    movi v2.2d, #0
    movi v3.2d, #0
    movi v4.2d, #0

.Lasum8_check:
    cmp x0, #8
    b.lo .Lasum4_check

.Lasum8:
    ldr q5, [x1]
    ldr q6, [x1, #16]
    ldr q7, [x1, #32]
    ldr q16, [x1, #48]
    fabs v5.2d, v5.2d
    fabs v6.2d, v6.2d
    fabs v7.2d, v7.2d
    fabs v16.2d, v16.2d
    fadd v1.2d, v1.2d, v5.2d
    fadd v2.2d, v2.2d, v6.2d
    fadd v3.2d, v3.2d, v7.2d
    fadd v4.2d, v4.2d, v16.2d
    add x1, x1, #64
    sub x0, x0, #8
    b .Lasum8_check

.Lasum4_check:
    cmp x0, #4
    b.lo .Lasum2_check
    ldr q5, [x1]
    ldr q6, [x1, #16]
    fabs v5.2d, v5.2d
    fabs v6.2d, v6.2d
    fadd v1.2d, v1.2d, v5.2d
    fadd v2.2d, v2.2d, v6.2d
    add x1, x1, #32
    sub x0, x0, #4

.Lasum2_check:
    cmp x0, #2
    b.lo .Lasum_reduce
    ldr q5, [x1]
    fabs v5.2d, v5.2d
    fadd v1.2d, v1.2d, v5.2d
    add x1, x1, #16
    sub x0, x0, #2

.Lasum_reduce:
    fadd v1.2d, v1.2d, v2.2d
    fadd v3.2d, v3.2d, v4.2d
    fadd v1.2d, v1.2d, v3.2d
    faddp d0, v1.2d
    cbz x0, .Lasum_done
    ldr d5, [x1]
    fabs d5, d5
    fadd d0, d0, d5

.Lasum_done:
    ret

// double zynum_blas_sme_dasum_f64(usize n, const double *x)
.globl ZYNUM_SYMBOL(zynum_blas_sme_dasum_f64)
.p2align 2
ZYNUM_SYMBOL(zynum_blas_sme_dasum_f64):
    cbz x0, .Lsme_asum_zero
    ptrue p0.d
    cntd x6
    lsl x7, x6, #2
    dup z0.d, #0
    dup z1.d, #0
    dup z2.d, #0
    dup z3.d, #0

.Lsme_asum4_check:
    cmp x0, x7
    b.lo .Lsme_asum_tail
    ld1d { z4.d }, p0/z, [x1]
    ld1d { z5.d }, p0/z, [x1, #1, MUL VL]
    ld1d { z6.d }, p0/z, [x1, #2, MUL VL]
    ld1d { z7.d }, p0/z, [x1, #3, MUL VL]
    fabs z4.d, p0/m, z4.d
    fabs z5.d, p0/m, z5.d
    fabs z6.d, p0/m, z6.d
    fabs z7.d, p0/m, z7.d
    fadd z0.d, z0.d, z4.d
    fadd z1.d, z1.d, z5.d
    fadd z2.d, z2.d, z6.d
    fadd z3.d, z3.d, z7.d
    addvl x1, x1, #4
    sub x0, x0, x7
    b .Lsme_asum4_check

.Lsme_asum_tail:
    cbz x0, .Lsme_asum_reduce
    mov x8, #0
    whilelo p1.d, x8, x0
    ld1d { z4.d }, p1/z, [x1]
    fabs z4.d, p1/m, z4.d
    fadd z0.d, z0.d, z4.d
    cmp x0, x6
    b.ls .Lsme_asum_reduce
    addvl x1, x1, #1
    sub x0, x0, x6
    b .Lsme_asum_tail

.Lsme_asum_reduce:
    fadd z0.d, z0.d, z1.d
    fadd z2.d, z2.d, z3.d
    fadd z0.d, z0.d, z2.d
    faddv d16, p0, z0.d
    fmov x9, d16
    fmov d0, x9
    ret

.Lsme_asum_zero:
    fmov d0, xzr
    ret

// void zynum_blas_asimd_dger_f64(
//     usize m, usize n, double alpha, const double *x, const double *y,
//     double *a, usize lda_bytes)
.globl ZYNUM_SYMBOL(zynum_blas_asimd_dger_f64)
.p2align 2
ZYNUM_SYMBOL(zynum_blas_asimd_dger_f64):
    cbz x0, .Lger_done
    cbz x1, .Lger_done

.Lger4_check:
    cmp x1, #4
    b.lo .Lger1_check

.Lger4:
    mov x8, x4
    add x9, x8, x5
    add x10, x9, x5
    add x11, x10, x5

    ldr d1, [x3]
    ldr d2, [x3, #8]
    ldr d3, [x3, #16]
    ldr d4, [x3, #24]
    fmul d1, d1, d0
    fmul d2, d2, d0
    fmul d3, d3, d0
    fmul d4, d4, d0
    dup v1.2d, v1.d[0]
    dup v2.2d, v2.d[0]
    dup v3.2d, v3.d[0]
    dup v4.2d, v4.d[0]

    mov x12, x0
    mov x13, x2

.Lger4_rows8_check:
    cmp x12, #8
    b.lo .Lger4_rows4_check

.Lger4_rows8:
    ldr q5, [x13]
    ldr q6, [x13, #16]
    ldr q7, [x13, #32]
    ldr q16, [x13, #48]

    ldr q17, [x8]
    ldr q18, [x8, #16]
    ldr q19, [x8, #32]
    ldr q20, [x8, #48]
    fmla v17.2d, v5.2d, v1.2d
    fmla v18.2d, v6.2d, v1.2d
    fmla v19.2d, v7.2d, v1.2d
    fmla v20.2d, v16.2d, v1.2d
    str q17, [x8]
    str q18, [x8, #16]
    str q19, [x8, #32]
    str q20, [x8, #48]

    ldr q17, [x9]
    ldr q18, [x9, #16]
    ldr q19, [x9, #32]
    ldr q20, [x9, #48]
    fmla v17.2d, v5.2d, v2.2d
    fmla v18.2d, v6.2d, v2.2d
    fmla v19.2d, v7.2d, v2.2d
    fmla v20.2d, v16.2d, v2.2d
    str q17, [x9]
    str q18, [x9, #16]
    str q19, [x9, #32]
    str q20, [x9, #48]

    ldr q17, [x10]
    ldr q18, [x10, #16]
    ldr q19, [x10, #32]
    ldr q20, [x10, #48]
    fmla v17.2d, v5.2d, v3.2d
    fmla v18.2d, v6.2d, v3.2d
    fmla v19.2d, v7.2d, v3.2d
    fmla v20.2d, v16.2d, v3.2d
    str q17, [x10]
    str q18, [x10, #16]
    str q19, [x10, #32]
    str q20, [x10, #48]

    ldr q17, [x11]
    ldr q18, [x11, #16]
    ldr q19, [x11, #32]
    ldr q20, [x11, #48]
    fmla v17.2d, v5.2d, v4.2d
    fmla v18.2d, v6.2d, v4.2d
    fmla v19.2d, v7.2d, v4.2d
    fmla v20.2d, v16.2d, v4.2d
    str q17, [x11]
    str q18, [x11, #16]
    str q19, [x11, #32]
    str q20, [x11, #48]

    add x13, x13, #64
    add x8, x8, #64
    add x9, x9, #64
    add x10, x10, #64
    add x11, x11, #64
    sub x12, x12, #8
    b .Lger4_rows8_check

.Lger4_rows4_check:
    cmp x12, #4
    b.lo .Lger4_rows2_check
    ldr q5, [x13]
    ldr q6, [x13, #16]
    ldr q17, [x8]
    ldr q18, [x8, #16]
    fmla v17.2d, v5.2d, v1.2d
    fmla v18.2d, v6.2d, v1.2d
    str q17, [x8]
    str q18, [x8, #16]
    ldr q17, [x9]
    ldr q18, [x9, #16]
    fmla v17.2d, v5.2d, v2.2d
    fmla v18.2d, v6.2d, v2.2d
    str q17, [x9]
    str q18, [x9, #16]
    ldr q17, [x10]
    ldr q18, [x10, #16]
    fmla v17.2d, v5.2d, v3.2d
    fmla v18.2d, v6.2d, v3.2d
    str q17, [x10]
    str q18, [x10, #16]
    ldr q17, [x11]
    ldr q18, [x11, #16]
    fmla v17.2d, v5.2d, v4.2d
    fmla v18.2d, v6.2d, v4.2d
    str q17, [x11]
    str q18, [x11, #16]
    add x13, x13, #32
    add x8, x8, #32
    add x9, x9, #32
    add x10, x10, #32
    add x11, x11, #32
    sub x12, x12, #4

.Lger4_rows2_check:
    cmp x12, #2
    b.lo .Lger4_rows1_check
    ldr q5, [x13]
    ldr q17, [x8]
    fmla v17.2d, v5.2d, v1.2d
    str q17, [x8]
    ldr q17, [x9]
    fmla v17.2d, v5.2d, v2.2d
    str q17, [x9]
    ldr q17, [x10]
    fmla v17.2d, v5.2d, v3.2d
    str q17, [x10]
    ldr q17, [x11]
    fmla v17.2d, v5.2d, v4.2d
    str q17, [x11]
    add x13, x13, #16
    add x8, x8, #16
    add x9, x9, #16
    add x10, x10, #16
    add x11, x11, #16
    sub x12, x12, #2

.Lger4_rows1_check:
    cbz x12, .Lger4_next
    ldr d5, [x13]
    ldr d17, [x8]
    fmadd d17, d5, d1, d17
    str d17, [x8]
    ldr d17, [x9]
    fmadd d17, d5, d2, d17
    str d17, [x9]
    ldr d17, [x10]
    fmadd d17, d5, d3, d17
    str d17, [x10]
    ldr d17, [x11]
    fmadd d17, d5, d4, d17
    str d17, [x11]

.Lger4_next:
    add x4, x4, x5, lsl #2
    add x3, x3, #32
    sub x1, x1, #4
    b .Lger4_check

.Lger1_check:
    cbz x1, .Lger_done

.Lger1:
    ldr d1, [x3], #8
    fmul d1, d1, d0
    dup v1.2d, v1.d[0]
    mov x8, x4
    mov x12, x0
    mov x13, x2

.Lger1_rows8_check:
    cmp x12, #8
    b.lo .Lger1_rows4_check
    ldr q5, [x13]
    ldr q6, [x13, #16]
    ldr q7, [x13, #32]
    ldr q16, [x13, #48]
    ldr q17, [x8]
    ldr q18, [x8, #16]
    ldr q19, [x8, #32]
    ldr q20, [x8, #48]
    fmla v17.2d, v5.2d, v1.2d
    fmla v18.2d, v6.2d, v1.2d
    fmla v19.2d, v7.2d, v1.2d
    fmla v20.2d, v16.2d, v1.2d
    str q17, [x8]
    str q18, [x8, #16]
    str q19, [x8, #32]
    str q20, [x8, #48]
    add x13, x13, #64
    add x8, x8, #64
    sub x12, x12, #8
    b .Lger1_rows8_check

.Lger1_rows4_check:
    cmp x12, #4
    b.lo .Lger1_rows2_check
    ldr q5, [x13]
    ldr q6, [x13, #16]
    ldr q17, [x8]
    ldr q18, [x8, #16]
    fmla v17.2d, v5.2d, v1.2d
    fmla v18.2d, v6.2d, v1.2d
    str q17, [x8]
    str q18, [x8, #16]
    add x13, x13, #32
    add x8, x8, #32
    sub x12, x12, #4

.Lger1_rows2_check:
    cmp x12, #2
    b.lo .Lger1_rows1_check
    ldr q5, [x13]
    ldr q17, [x8]
    fmla v17.2d, v5.2d, v1.2d
    str q17, [x8]
    add x13, x13, #16
    add x8, x8, #16
    sub x12, x12, #2

.Lger1_rows1_check:
    cbz x12, .Lger1_next
    ldr d5, [x13]
    ldr d17, [x8]
    fmadd d17, d5, d1, d17
    str d17, [x8]

.Lger1_next:
    add x4, x4, x5
    subs x1, x1, #1
    b.ne .Lger1

.Lger_done:
    ret

// void zynum_blas_sve_dger_f64(
//     usize m, usize n, double alpha, const double *x, const double *y,
//     double *a, usize lda_bytes)
//
// SVE rank-1 update for column-major A:
//     A[:, j] += alpha * y[j] * x
//
// Processes four columns per panel.  The common Level 2 layer owns argument
// normalization and parallel column partitioning; this kernel owns only the
// SVE row loop and tail predicate.
.globl ZYNUM_SYMBOL(zynum_blas_sve_dger_f64)
.p2align 2
ZYNUM_SYMBOL(zynum_blas_sve_dger_f64):
    cbz x0, .Lsve_ger_done
    cbz x1, .Lsve_ger_done
    ptrue p0.d

.Lsve_ger4_check:
    cmp x1, #4
    b.lo .Lsve_ger1_check

.Lsve_ger4:
    mov x8, x4
    add x9, x8, x5
    add x10, x9, x5
    add x11, x10, x5

    ldr d1, [x3]
    ldr d2, [x3, #8]
    ldr d3, [x3, #16]
    ldr d4, [x3, #24]
    fmul d1, d1, d0
    fmul d2, d2, d0
    fmul d3, d3, d0
    fmul d4, d4, d0
    mov z1.d, d1
    mov z2.d, d2
    mov z3.d, d3
    mov z4.d, d4

    mov x12, #0
.Lsve_ger4_rows:
    cmp x12, x0
    b.hs .Lsve_ger4_next
    ld1d { z0.d }, p0/z, [x2, x12, lsl #3]
    ld1d { z5.d }, p0/z, [x8, x12, lsl #3]
    ld1d { z6.d }, p0/z, [x9, x12, lsl #3]
    ld1d { z7.d }, p0/z, [x10, x12, lsl #3]
    ld1d { z16.d }, p0/z, [x11, x12, lsl #3]
    fmla z5.d, p0/m, z0.d, z1.d
    fmla z6.d, p0/m, z0.d, z2.d
    fmla z7.d, p0/m, z0.d, z3.d
    fmla z16.d, p0/m, z0.d, z4.d
    st1d { z5.d }, p0, [x8, x12, lsl #3]
    st1d { z6.d }, p0, [x9, x12, lsl #3]
    st1d { z7.d }, p0, [x10, x12, lsl #3]
    st1d { z16.d }, p0, [x11, x12, lsl #3]
    incd x12
    b .Lsve_ger4_rows

.Lsve_ger4_next:
    add x4, x4, x5, lsl #2
    add x3, x3, #32
    sub x1, x1, #4
    b .Lsve_ger4_check

.Lsve_ger1_check:
    cbz x1, .Lsve_ger_done

.Lsve_ger1:
    ldr d1, [x3], #8
    fmul d1, d1, d0
    mov z1.d, d1
    mov x8, x4
    mov x12, #0
.Lsve_ger1_rows:
    cmp x12, x0
    b.hs .Lsve_ger1_next
    ld1d { z0.d }, p0/z, [x2, x12, lsl #3]
    ld1d { z5.d }, p0/z, [x8, x12, lsl #3]
    fmla z5.d, p0/m, z0.d, z1.d
    st1d { z5.d }, p0, [x8, x12, lsl #3]
    incd x12
    b .Lsve_ger1_rows

.Lsve_ger1_next:
    add x4, x4, x5
    subs x1, x1, #1
    b.ne .Lsve_ger1

.Lsve_ger_done:
    ret

// void zynum_blas_sme_dger_f64(
//     usize m, usize n, double alpha, const double *x, const double *y,
//     double *a, usize lda_bytes)
//
// Full-tile SME rank-1 update for column-major A:
//     A += alpha * x * y^T
// The Zig wrapper only calls this when m and n are multiples of the f64 SME
// tile.  Each ZA tile is initialized from the old A tile, receives one FMOPA,
// then is stored back to A, preserving GER's additive semantics.
.globl ZYNUM_SYMBOL(zynum_blas_sme_dger_f64)
.p2align 2
ZYNUM_SYMBOL(zynum_blas_sme_dger_f64):
    cbz x0, .Lsme_ger_return
    cbz x1, .Lsme_ger_return

    rdsvl x14, #1
    ptrue p0.d

    lsr x15, x14, #3          // f64 elements per SME tile.
    lsl x16, x0, #3           // m in bytes.
    mul x13, x15, x5          // byte stride from one column tile to the next.
    mov x7, x4                // Current A column-tile base.
    mov x8, x3                // Current y tile pointer.
    mov x17, x1               // Remaining columns in elements.
    mov z31.d, d0             // alpha vector.

.Lsme_ger_cols:
    cbz x17, .Lsme_ger_done
    ld1d { z1.d }, p0/z, [x8]
    fmul z1.d, z1.d, z31.d
    mov x9, #0                // Current row offset in bytes.

.Lsme_ger_rows:
    cmp x9, x16
    b.hs .Lsme_ger_next_col

    add x10, x7, x9           // A tile base.
    mov w12, #0
    mov x11, x15
.Lsme_ger_load_a:
    cbz x11, .Lsme_ger_loaded_a
    ld1d { za0v.d[w12, 0] }, p0/z, [x10]
    add x10, x10, x5
    add w12, w12, #1
    sub x11, x11, #1
    b .Lsme_ger_load_a

.Lsme_ger_loaded_a:
    add x10, x2, x9
    ld1d { z0.d }, p0/z, [x10]
    fmopa za0.d, p0/m, p0/m, z0.d, z1.d

    add x10, x7, x9
    mov w12, #0
    mov x11, x15
.Lsme_ger_store_a:
    cbz x11, .Lsme_ger_stored_a
    st1d { za0v.d[w12, 0] }, p0, [x10]
    add x10, x10, x5
    add w12, w12, #1
    sub x11, x11, #1
    b .Lsme_ger_store_a

.Lsme_ger_stored_a:
    add x9, x9, x14
    b .Lsme_ger_rows

.Lsme_ger_next_col:
    add x7, x7, x13
    add x8, x8, x14
    sub x17, x17, x15
    b .Lsme_ger_cols

.Lsme_ger_done:

.Lsme_ger_return:
    ret
