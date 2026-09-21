// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later

//! Core dispatch entries for packed and banded triangular operations.

const std = @import("std");
const catalog = @import("../../kernels/shared/matrix_vector/catalog.zig");
const level2_tuning = @import("../../kernels/shared/matrix_vector/tuning.zig");

const builtin = @import("builtin");

const scalar = @import("../shared/scalar.zig");
const triangular = @import("triangular.zig");
const isolated = @import("../../kernels/isolated/x86_64_compact_triangular_bridge.zig");

const BlasInt = scalar.BlasInt;
const Order = scalar.Order;
const Uplo = scalar.Uplo;
const Diag = scalar.Diag;

pub noinline fn tpmv(comptime T: type, uplo: Uplo, trans_: Order, diag: Diag, n: BlasInt, ap: [*]const T, x: [*]T, incx: BlasInt) void {
    if (comptime builtin.cpu.arch == .x86_64) {
        if (n > 0 and incx != 0 and isolated.tryTpmv(T, uplo, trans_, diag, n, ap, x, incx)) return;
    }
    if (comptime builtin.cpu.arch == .aarch64 and builtin.os.tag == .macos and (T == f32 or T == f64)) {
        const profile = level2_tuning.active.triangular;
        if (profile.selectFiniteTpmv(T, n) == .compact_triangular_packed_finite)
            return tpmvFiniteCandidate(T, uplo, trans_, diag, n, ap, x, incx);
    }
    return legacyTpmv(T, uplo, trans_, diag, n, ap, x, incx);
}

// Keep the non-tail candidate attempt out of the public dispatch entry. Both
// dispatch targets have the same argument ABI, so off-gate calls need not retain
// their arguments across an attempt that they cannot use.
noinline fn tpmvFiniteCandidate(comptime T: type, uplo: Uplo, trans_: Order, diag: Diag, n: BlasInt, ap: [*]const T, x: [*]T, incx: BlasInt) void {
    const profile = level2_tuning.active.triangular;
    if (tryFiniteTpmv(T, true, std.heap.c_allocator, profile.finite_tpmv_workspace_max_bytes, uplo, trans_, diag, n, ap, x, incx)) return;
    return legacyTpmv(T, uplo, trans_, diag, n, ap, x, incx);
}

// Isolate the original full-range body from candidate-call register pressure.
// This leaf deliberately retains the old arguments and arithmetic unchanged.
// Compile-time selection adds no runtime wrapper or branch. Measured Mac
// leaves have explicit layouts; other leaves retain the default.
inline fn legacyTpmv(comptime T: type, uplo: Uplo, trans_: Order, diag: Diag, n: BlasInt, ap: [*]const T, x: [*]T, incx: BlasInt) void {
    if (comptime builtin.cpu.arch == .aarch64 and builtin.os.tag == .macos and (T == f32 or T == f64)) {
        return legacyTpmvAligned(T, uplo, trans_, diag, n, ap, x, incx);
    }
    if (comptime builtin.cpu.arch == .aarch64 and builtin.os.tag == .macos and T == scalar.ComplexF64) {
        return legacyComplexTpmvAligned(T, uplo, trans_, diag, n, ap, x, incx);
    }
    return legacyTpmvDefault(T, uplo, trans_, diag, n, ap, x, incx);
}

noinline fn legacyTpmvAligned(comptime T: type, uplo: Uplo, trans_: Order, diag: Diag, n: BlasInt, ap: [*]const T, x: [*]T, incx: BlasInt) align(64) void {
    triangular.tpmv(T, uplo, trans_, diag, n, ap, x, incx);
}

// On Apple M5, shifting this full-row body four bytes avoids the measured
// lower/no-transpose and upper/transpose unit-diagonal layout regressions.
// Anchor the body independently of preceding kernels; preserve all arithmetic.
noinline fn legacyComplexTpmvAligned(comptime T: type, uplo: Uplo, trans_: Order, diag: Diag, n: BlasInt, ap: [*]const T, x: [*]T, incx: BlasInt) align(64) void {
    asm volatile ("nop");
    triangular.tpmv(T, uplo, trans_, diag, n, ap, x, incx);
}

noinline fn legacyTpmvDefault(comptime T: type, uplo: Uplo, trans_: Order, diag: Diag, n: BlasInt, ap: [*]const T, x: [*]T, incx: BlasInt) void {
    triangular.tpmv(T, uplo, trans_, diag, n, ap, x, incx);
}

pub noinline fn tpsv(comptime T: type, uplo: Uplo, trans_: Order, diag: Diag, n: BlasInt, ap: [*]const T, x: [*]T, incx: BlasInt) void {
    if (comptime builtin.cpu.arch == .x86_64) {
        if (n > 0 and incx != 0 and isolated.tryTpsv(T, uplo, trans_, diag, n, ap, x, incx)) return;
    }
    triangular.tpsv(T, uplo, trans_, diag, n, ap, x, incx);
}

pub noinline fn tbsv(comptime T: type, uplo: Uplo, trans_: Order, diag: Diag, n: BlasInt, k: BlasInt, a: [*]const T, lda: BlasInt, x: [*]T, incx: BlasInt) void {
    if (comptime builtin.cpu.arch == .x86_64) {
        if (n > 0 and incx != 0 and k >= 0 and isolated.tryTbsv(T, uplo, trans_, diag, n, k, a, lda, x, incx)) return;
    }
    if (comptime T == f32 or T == f64) {
        const profile = level2_tuning.active.triangular;
        if (profile.selectFiniteTbsv(T, n, k) == .compact_triangular_band_finite and
            tryFiniteTbsv(T, std.heap.c_allocator, profile.finite_tbsv_workspace_max_bytes, uplo, trans_, diag, n, k, a, lda, x, incx)) return;
    }
    triangular.tbsv(T, uplo, trans_, diag, n, k, a, lda, x, incx);
}

// Experimental whole operation. All refusals precede caller writes.
// Retrying the fallback does not promise floating-point trap/flag atomicity.
noinline fn tryFiniteTbsv(
    comptime T: type,
    allocator: std.mem.Allocator,
    max_bytes: usize,
    uplo: Uplo,
    trans_: Order,
    diag: Diag,
    n_: BlasInt,
    k_: BlasInt,
    a: [*]const T,
    lda: BlasInt,
    x: [*]T,
    incx: BlasInt,
) bool {
    if (k_ == 0 and diag == .unit and tryNormalUnitDiagonal(T, max_bytes, n_, a, lda, x, incx)) return true;
    const effective_upper = (trans_ == .no_trans and uplo == .upper) or
        (trans_ != .no_trans and uplo == .lower);
    if (n_ >= 128 and k_ >= (@as(BlasInt, 8) << @intFromBool(diag == .non_unit)) and k_ <= @divTrunc(n_, 4) and !effective_upper) {
        return tryFiniteTbsvImpl(T, 4, false, allocator, max_bytes, uplo, trans_, diag, n_, k_, a, lda, x, incx);
    }
    if (effective_upper and diag == .unit and k_ >= 4 and @mod(k_, 2) != 0) return tryFiniteTbsvImpl(T, 4, true, allocator, max_bytes, uplo, trans_, diag, n_, k_, a, lda, x, incx);
    if (effective_upper and (diag == .unit or @mod(k_, 2) == 0)) return tryFiniteTbsvImpl(T, 1, false, allocator, max_bytes, uplo, trans_, diag, n_, k_, a, lda, x, incx);
    if (@mod(k_, 2) == 0) return tryFiniteTbsvImpl(T, 3, false, allocator, max_bytes, uplo, trans_, diag, n_, k_, a, lda, x, incx);
    return tryFiniteTbsvImpl(T, 2, false, allocator, max_bytes, uplo, trans_, diag, n_, k_, a, lda, x, incx);
}

// A unit diagonal with no off-diagonal entries leaves normal finite inputs
// unchanged. Positive zero is also unchanged except under downward rounding.
// Negative zero, subnormals and nonfinite values retain their original arithmetic.
noinline fn tryNormalUnitDiagonal(comptime T: type, max_bytes: usize, n_: BlasInt, a: [*]const T, lda: BlasInt, x: [*]const T, incx: BlasInt) bool {
    if (comptime T != f32 and T != f64) return false;
    if (builtin.cpu.arch != .aarch64 or builtin.os.tag != .macos or n_ < 128 or lda < 1 or incx == 0) return false;
    const n: usize = @intCast(n_);
    const bytes = std.math.mul(usize, n, @sizeOf(T)) catch return false;
    if (bytes > max_bytes) return false;
    const signed_stride: i64 = incx;
    const stride: usize = @intCast(if (signed_stride < 0) -signed_stride else signed_stride);
    const last = std.math.mul(usize, n - 1, stride) catch return false;
    const span = std.math.add(usize, last, 1) catch return false;
    const x_bytes = std.math.mul(usize, span, @sizeOf(T)) catch return false;
    const matrix_count = std.math.mul(usize, @intCast(lda), n) catch return false;
    const a_bytes = std.math.mul(usize, matrix_count, @sizeOf(T)) catch return false;
    const a_addr = @intFromPtr(a);
    const x_addr = @intFromPtr(x);
    if (x_addr >= a_addr) {
        if (x_addr - a_addr < a_bytes) return false;
    } else if (a_addr - x_addr < x_bytes) return false;
    const Bits = std.meta.Int(.unsigned, @bitSizeOf(T));
    const exponent: Bits = if (T == f32) 0x7f800000 else 0x7ff0000000000000;
    // The predicate is independent of logical order; scan physical strides.
    var i: usize = 0;
    var has_positive_zero = false;
    if (stride == 1) {
        const width = 16;
        const V = @Vector(width, Bits);
        while (n - i >= width) : (i += width) {
            const raw: V = @bitCast(@as(@Vector(width, T), x[i..][0..width].*));
            const exps = raw & @as(V, @splat(exponent));
            // Unsigned range test excludes exponent zero and all-ones together.
            if (!@reduce(.And, (exps -% @as(V, @splat(1))) < @as(V, @splat(exponent - 1)))) {
                // Only positive zero may repair an excluded exponent. Keep the
                // exceptional block check vectorized instead of reloading lanes.
                const repaired = @select(Bits, raw == @as(V, @splat(0)), @as(V, @splat(1)), exps);
                if (!@reduce(.And, (repaired -% @as(V, @splat(1))) < @as(V, @splat(exponent - 1)))) return false;
                has_positive_zero = true;
            }
        }
    }
    while (i < n) : (i += 1) {
        const bits: Bits = @bitCast(x[i * stride]);
        const exp = bits & exponent;
        if (exp == 0 or exp == exponent) {
            if (bits != 0) return false;
            has_positive_zero = true;
        }
    }
    return !has_positive_zero or identityPreservesPositiveZero();
}

inline fn identityPreservesPositiveZero() bool {
    // ARM FPCR.RMode=2 is round toward minus infinity: +0 - +0 becomes -0.
    // Other modes leave +0 unchanged after subtraction of either signed zero.
    const fpcr = asm volatile ("mrs %[result], fpcr"
        : [result] "=r" (-> u64),
    );
    return ((fpcr >> 22) & 3) != 2;
}

inline fn initTbsvSolvedSigns(comptime T: type, upper: bool, solved: usize, n: usize, work: []const T, first_iteration: *[2]usize, first_index: *[2]usize) void {
    const Bits = std.meta.Int(.unsigned, @bitSizeOf(T));
    var found: usize = 0;
    for (0..solved) |iteration| {
        const i = if (upper) n - iteration - 1 else iteration;
        const sign: usize = @intCast(@as(Bits, @bitCast(work[i])) >> (@bitSizeOf(T) - 1));
        if (first_iteration[sign] == n) {
            first_iteration[sign] = iteration;
            first_index[sign] = i;
            found += 1;
            if (found == 2) break;
        }
    }
}

noinline fn tryFiniteTbsvImpl(
    comptime T: type,
    comptime unroll: usize,
    comptime upper_four: bool,
    allocator: std.mem.Allocator,
    max_bytes: usize,
    uplo: Uplo,
    trans_: Order,
    diag: Diag,
    n_: BlasInt,
    k_: BlasInt,
    a: [*]const T,
    lda: BlasInt,
    x: [*]T,
    incx: BlasInt,
) bool {
    @setFloatMode(.strict);
    if (comptime T != f32 and T != f64) return false;
    if (builtin.cpu.arch != .aarch64 or builtin.os.tag != .macos or n_ < 128 or k_ < 0 or incx == 0 or lda < 1) return false;
    if (k_ > @divTrunc(n_, 4)) return false;
    const n: usize = @intCast(n_);
    const k: usize = @intCast(k_);
    const ld: usize = @intCast(lda);
    if (ld <= k) return false;
    const bytes = std.math.mul(usize, n, @sizeOf(T)) catch return false;
    if (bytes > max_bytes) return false;
    // Widen before negation so the minimum BlasInt stride is representable.
    const stride_signed: i64 = @intCast(incx);
    const stride: usize = @intCast(if (stride_signed < 0) -stride_signed else stride_signed);
    const last = std.math.mul(usize, n - 1, stride) catch return false;
    const span = std.math.add(usize, last, 1) catch return false;
    const x_bytes = std.math.mul(usize, span, @sizeOf(T)) catch return false;
    const matrix_count = std.math.mul(usize, ld, n) catch return false;
    const a_bytes = std.math.mul(usize, matrix_count, @sizeOf(T)) catch return false;
    const a_addr = @intFromPtr(a);
    const x_addr = @intFromPtr(x);
    if (x_addr >= a_addr) {
        if (x_addr - a_addr < a_bytes) return false;
    } else if (a_addr - x_addr < x_bytes) return false;
    const work = allocator.alloc(T, n) catch return false;
    defer allocator.free(work);
    if (incx == 1) {
        @memcpy(work, x[0..n]);
    } else {
        for (0..n) |i| work[i] = x[if (incx > 0) i * stride else last - i * stride];
    }
    const upper = if (comptime unroll == 4) upper_four else (trans_ == .no_trans and uplo == .upper) or
        (trans_ != .no_trans and uplo == .lower);
    // Previously solved values are finite, including signed zeros. At a zero
    // accumulator, each sign present in a skipped region needs one subtraction:
    // repeated same-sign zero subtractions are idempotent under IEEE arithmetic.
    var first_iteration = [_]usize{ n, n };
    var first_index = [_]usize{ 0, 0 };
    const lazy_signs = T == f32 and unroll == 4 and !upper_four;
    var track_signs = !lazy_signs;
    for (0..n) |iteration| {
        const i = if (upper) n - iteration - 1 else iteration;
        const begin = if (upper) i + 1 else i - @min(i, k);
        const end = if (upper) i + 1 + @min(k, n - i - 1) else i;
        var value = work[i];
        if (!upper and value == 0) {
            if (lazy_signs and begin > 0 and !track_signs) {
                initTbsvSolvedSigns(T, upper, iteration, n, work, &first_iteration, &first_index);
                track_signs = true;
            }
            for (0..2) |sign| {
                if (first_iteration[sign] < begin) value = value - @as(T, 0) * work[first_index[sign]];
            }
        }
        if (begin < end) {
            const row = if (trans_ == .no_trans) i else begin;
            const col = if (trans_ == .no_trans) begin else i;
            var offset = if (uplo == .upper) k + row - col + col * ld else row - col + col * ld;
            const step = if (trans_ == .no_trans) ld - 1 else 1;
            if (comptime unroll == 4) {
                var j = begin;
                // Independent loads/products may overlap, but each subtraction
                // retains the original ascending-j dependency and strict ordering.
                while (end - j >= 4) : (j += 4) {
                    const product0 = a[offset] * work[j];
                    value = value - product0;
                    offset += step;
                    const product1 = a[offset] * work[j + 1];
                    value = value - product1;
                    offset += step;
                    const product2 = a[offset] * work[j + 2];
                    value = value - product2;
                    offset += step;
                    const product3 = a[offset] * work[j + 3];
                    value = value - product3;
                    offset += step;
                }
                while (j < end) : (j += 1) {
                    const product = a[offset] * work[j];
                    value = value - product;
                    offset += step;
                }
                // The final unused index is bounded by ld*n+ld. Checked ld*n*T
                // bytes, sizeof(T)>=4 and n>=128 leave ample usize headroom.

            } else if (comptime unroll == 1) {
                for (begin..end) |j| {
                    const product = a[offset] * work[j];
                    value = value - product;
                    if (j + 1 < end) offset += step;
                }
            } else if (comptime unroll == 2) {
                var j = begin;
                while (end - j >= 2) : (j += 2) {
                    const product0 = a[offset] * work[j];
                    const product1 = a[offset + step] * work[j + 1];
                    value = value - product0;
                    value = value - product1;
                    if (j + 2 < end) offset += 2 * step;
                }
                if (j < end) value = value - a[offset] * work[j];
            } else {
                var j = begin;
                while (end - j >= 3) : (j += 2) {
                    const product0 = a[offset] * work[j];
                    const product1 = a[offset + step] * work[j + 1];
                    value = value - product0;
                    value = value - product1;
                    offset += 2 * step;
                }
                // begin < end and the pair loop leaves one or two terms.
                value = value - a[offset] * work[j];
                if (j + 1 < end) value = value - a[offset + step] * work[j + 1];
            }
        }
        if (upper and value == 0) {
            if (lazy_signs and end < n and !track_signs) {
                initTbsvSolvedSigns(T, upper, iteration, n, work, &first_iteration, &first_index);
                track_signs = true;
            }
            for (0..2) |sign| {
                if (first_iteration[sign] < n - end) value = value - @as(T, 0) * work[first_index[sign]];
            }
        }
        // Real conjugate-transpose equals transpose. Unit diagonal is never read.
        if (diag == .non_unit) value = value / a[i * ld + (if (uplo == .upper) k else @as(usize, 0))];
        const Bits = std.meta.Int(.unsigned, @bitSizeOf(T));
        const exponent: Bits = if (T == f32) 0x7f800000 else 0x7ff0000000000000;
        if ((@as(Bits, @bitCast(value)) & exponent) == exponent) return false;
        work[i] = value;
        if (track_signs) {
            const sign: usize = @intCast(@as(Bits, @bitCast(value)) >> (@bitSizeOf(T) - 1));
            if (first_iteration[sign] == n) {
                first_iteration[sign] = iteration;
                first_index[sign] = i;
            }
        }
    }
    if (incx == 1) {
        @memcpy(x[0..n], work);
    } else {
        for (0..n) |i| x[if (incx > 0) i * stride else last - i * stride] = work[i];
    }
    return true;
}

pub const testing = struct {
    pub fn forceTpmv(comptime T: type, implementation: catalog.Implementation, allocator: std.mem.Allocator, max_bytes: usize, uplo: Uplo, trans_: Order, diag: Diag, n: BlasInt, ap: [*]const T, x: [*]T, incx: BlasInt) bool {
        if (implementation == .portable_scalar) {
            legacyTpmv(T, uplo, trans_, diag, n, ap, x, incx);
            return true;
        }
        if (implementation != .compact_triangular_packed_finite) return false;
        return tryFiniteTpmv(T, false, allocator, max_bytes, uplo, trans_, diag, n, ap, x, incx);
    }

    pub fn forceTbsv(comptime T: type, implementation: catalog.Implementation, allocator: std.mem.Allocator, max_bytes: usize, uplo: Uplo, trans_: Order, diag: Diag, n: BlasInt, k: BlasInt, a: [*]const T, lda: BlasInt, x: [*]T, incx: BlasInt) bool {
        if (implementation == .portable_scalar) {
            triangular.tbsv(T, uplo, trans_, diag, n, k, a, lda, x, incx);
            return true;
        }
        if (implementation != .compact_triangular_band_finite) return false;
        return tryFiniteTbsv(T, allocator, max_bytes, uplo, trans_, diag, n, k, a, lda, x, incx);
    }
};

// Reuse contiguous kernels for bounded strided input. Commit only after the
// inner computation succeeds; its unit stride prevents repeated staging.
noinline fn stagedSmallTpmv(comptime T: type, allocator: std.mem.Allocator, max_bytes: usize, uplo: Uplo, trans_: Order, diag: Diag, n_: BlasInt, ap: [*]const T, x: [*]T, incx: BlasInt, stride: usize, last: usize) bool {
    const n: usize = @intCast(n_);
    var values: [128]T = undefined;
    for (0..n) |i| values[i] = x[if (incx > 0) i * stride else last - i * stride];
    if (!tryFiniteTpmv(T, true, allocator, max_bytes, uplo, trans_, diag, n_, ap, &values, 1)) return false;
    for (0..n) |i| x[if (incx > 0) i * stride else last - i * stride] = values[i];
    return true;
}

// Ordered packed rows. No caller output is changed on refusal.
noinline fn tryFiniteTpmv(comptime T: type, comptime stack_small: bool, allocator: std.mem.Allocator, max_bytes: usize, uplo: Uplo, trans_: Order, diag: Diag, n_: BlasInt, ap: [*]const T, x: [*]T, incx: BlasInt) bool {
    @setFloatMode(.strict);
    if (comptime T != f32 and T != f64) return false;
    if (builtin.cpu.arch != .aarch64 or builtin.os.tag != .macos or n_ < 64 or incx == 0) return false;
    const n: usize = @intCast(n_);
    const bytes = std.math.mul(usize, n, @sizeOf(T)) catch return false;
    if (bytes > max_bytes) return false;
    const next_n = std.math.add(usize, n, 1) catch return false;
    const count = (std.math.mul(usize, n, next_n) catch return false) / 2;
    const a_bytes = std.math.mul(usize, count, @sizeOf(T)) catch return false;
    const signed_stride: i64 = @intCast(incx);
    const stride: usize = @intCast(if (signed_stride < 0) -signed_stride else signed_stride);
    const last = std.math.mul(usize, n - 1, stride) catch return false;
    const span = std.math.add(usize, last, 1) catch return false;
    const x_bytes = std.math.mul(usize, span, @sizeOf(T)) catch return false;
    const a_addr = @intFromPtr(ap);
    const x_addr = @intFromPtr(x);
    if (x_addr >= a_addr) {
        if (x_addr - a_addr < a_bytes) return false;
    } else if (a_addr - x_addr < x_bytes) return false;
    // Reserve both logical input and output workspace before copying input.
    if (stack_small and T == f32 and n <= 128 and trans_ != .no_trans and incx != 1 and bytes <= max_bytes / 2) {
        return stagedSmallTpmv(T, allocator, max_bytes - bytes, uplo, trans_, diag, n_, ap, x, incx, stride, last);
    }
    const Bits = std.meta.Int(.unsigned, @bitSizeOf(T));
    const exponent: Bits = if (T == f32) 0x7f800000 else 0x7ff0000000000000;
    if (stride == 1) {
        var i: usize = 0;
        while (n - i >= 16) : (i += 16) {
            const raw: @Vector(16, Bits) = @bitCast(@as(@Vector(16, T), x[i..][0..16].*));
            if (@reduce(.Or, (raw & @as(@Vector(16, Bits), @splat(exponent))) == @as(@Vector(16, Bits), @splat(exponent)))) return false;
        }
        while (i < n) : (i += 1) {
            if ((@as(Bits, @bitCast(x[i])) & exponent) == exponent) return false;
        }
    } else {
        for (0..n) |i| {
            const value = x[if (incx > 0) i * stride else last - i * stride];
            if ((@as(Bits, @bitCast(value)) & exponent) == exponent) return false;
        }
    }
    var local_output: [if (stack_small) 128 else 0]T = undefined;
    const use_local = stack_small and n <= local_output.len;
    const output = if (use_local) local_output[0..n] else allocator.alloc(T, n) catch return false;
    defer if (!use_local) allocator.free(output);
    const upper = (trans_ == .no_trans and uplo == .upper) or (trans_ != .no_trans and uplo == .lower);
    const block_rows: usize = 8;
    const paired_end = n - n % block_rows;
    if (trans_ == .no_trans) {
        if (block_rows > 2) {
            if (T == f32 and n >= 512) {
                if (!finiteTpmvNoTransRows(T, 16, 0, uplo, diag, n, ap, x, incx, stride, last, output)) return false;
                const tail_begin = n - n % 16;
                if (n - tail_begin >= 8) {
                    if (!finiteTpmvNoTransRows(T, 8, tail_begin, uplo, diag, n, ap, x, incx, stride, last, output)) return false;
                }
            } else if (!finiteTpmvNoTransRows(T, 8, 0, uplo, diag, n, ap, x, incx, stride, last, output)) return false;
        } else if (!finiteTpmvNoTransPairs(T, 0, uplo, diag, n, ap, x, incx, stride, last, output)) return false;
    } else if (block_rows > 2) {
        if (incx == 1 and ((T == f32 and (n >= 128 or diag == .unit)) or (T == f64 and n >= 512))) {
            if (!finiteTpmvTransRows(T, 16, true, 0, uplo, diag, n, ap, x, incx, stride, last, output)) return false;
            const tail_begin = n - n % 16;
            if (n - tail_begin >= 8) {
                if (!finiteTpmvTransRows(T, 8, true, tail_begin, uplo, diag, n, ap, x, incx, stride, last, output)) return false;
            }
        } else if (incx == 1) {
            if (T == f32 and n >= 64) {
                if (!finiteTpmvTransRows(T, 16, true, 0, uplo, diag, n, ap, x, incx, stride, last, output)) return false;
                const tail_begin = n - n % 16;
                if (n - tail_begin >= 8) {
                    if (!finiteTpmvTransRows(T, 8, true, tail_begin, uplo, diag, n, ap, x, incx, stride, last, output)) return false;
                }
            } else if (!finiteTpmvTransRows(T, 8, true, 0, uplo, diag, n, ap, x, incx, stride, last, output)) return false;
        } else if (!finiteTpmvTransRows(T, 8, false, 0, uplo, diag, n, ap, x, incx, stride, last, output)) return false;
    } else {
        if (!finiteTpmvTransPairs(T, 0, uplo, diag, n, ap, x, incx, stride, last, output)) return false;
    }
    var scalar_begin = paired_end;
    if (!upper and n - scalar_begin >= 4) {
        if (trans_ == .no_trans) {
            if (!finiteTpmvLowerTailFour(T, false, scalar_begin, diag, n, ap, x, incx, stride, last, output)) return false;
        } else if (!finiteTpmvLowerTailFour(T, true, scalar_begin, diag, n, ap, x, incx, stride, last, output)) return false;
        scalar_begin += 4;
    }
    if (!upper and n - scalar_begin >= 2) {
        if (trans_ == .no_trans) {
            if (!finiteTpmvNoTransPairs(T, scalar_begin, uplo, diag, n, ap, x, incx, stride, last, output)) return false;
        } else if (!finiteTpmvTransPairs(T, scalar_begin, uplo, diag, n, ap, x, incx, stride, last, output)) return false;
        scalar_begin = n - n % 2;
    }
    for (scalar_begin..n) |iteration| {
        const i = iteration;
        const begin = if (upper) i + 1 else 0;
        const end = if (upper) n else i;
        var offset = if (trans_ == .no_trans)
            (if (uplo == .upper) i * (i + 1) / 2 + i else i)
        else
            (if (uplo == .upper) i * (i + 1) / 2 else i * (2 * n - i + 1) / 2);
        var sum: T = 0;
        // The diagonal is the first term of an upper row and the last term of
        // a lower row. Peel it without changing the ascending addition order,
        // including the initial +0 addition and the unit-diagonal multiply.
        if (upper) {
            const av: T = if (diag == .unit) 1 else ap[offset];
            const product = av * x[if (incx > 0) i * stride else last - i * stride];
            sum = sum + product;
            offset += if (trans_ != .no_trans) 1 else i + 1;
        }
        for (begin..end) |j| {
            const product = ap[offset] * x[if (incx > 0) j * stride else last - j * stride];
            sum = sum + product;
            // Even the unused final upper-row offset is at most count + n.
            // Checked count * sizeof(T), sizeof(T) >= 4 and n <= count make
            // this update safe without a per-element final-term branch.
            offset += if (trans_ != .no_trans) 1 else if (uplo == .upper) j + 1 else n - j - 1;
        }
        if (!upper) {
            const av: T = if (diag == .unit) 1 else ap[offset];
            const product = av * x[if (incx > 0) i * stride else last - i * stride];
            sum = sum + product;
        }
        if ((@as(Bits, @bitCast(sum)) & exponent) == exponent or sum == 0) return false;
        output[i] = sum;
    }
    for (0..n) |i| x[if (incx > 0) i * stride else last - i * stride] = output[i];
    return true;
}

// Four independent long-tail outputs retain each row's ascending term order.
noinline fn finiteTpmvLowerTailFour(comptime T: type, comptime transposed: bool, first: usize, diag: Diag, n: usize, ap: [*]const T, x: [*]const T, incx: BlasInt, stride: usize, last: usize, output: []T) align(64) bool {
    @setFloatMode(.strict);
    const Bits = std.meta.Int(.unsigned, @bitSizeOf(T));
    const exponent: Bits = if (T == f32) 0x7f800000 else 0x7ff0000000000000;
    var sums: @Vector(4, T) = @splat(0);
    var bases: [4]usize = undefined;
    inline for (0..4) |r| {
        const row = first + r;
        bases[r] = if (transposed) row * (row + 1) / 2 else row;
    }
    for (0..first) |j| {
        var coefficients: [4]T = undefined;
        inline for (0..4) |r| coefficients[r] = ap[bases[r]];
        const xj = x[if (incx > 0) j * stride else last - j * stride];
        sums = sums + @as(@Vector(4, T), coefficients) * @as(@Vector(4, T), @splat(xj));
        inline for (0..4) |r| bases[r] += if (transposed) 1 else n - j - 1;
    }
    inline for (0..4) |r| {
        inline for (0..r) |c| {
            const j = first + c;
            sums[r] = sums[r] + ap[bases[r]] * x[if (incx > 0) j * stride else last - j * stride];
            bases[r] += if (transposed) 1 else n - j - 1;
        }
        const row = first + r;
        const diagonal: T = if (diag == .unit) 1 else ap[bases[r]];
        sums[r] = sums[r] + diagonal * x[if (incx > 0) row * stride else last - row * stride];
        if ((@as(Bits, @bitCast(sums[r])) & exponent) == exponent or sums[r] == 0) return false;
        output[row] = sums[r];
    }
    return true;
}

// Transposed rows occupy adjacent packed columns. Share each X load across
// two independent, ordered sums; neither a reduction nor FMA is introduced.
noinline fn finiteTpmvTransPairs(comptime T: type, first_row: usize, uplo: Uplo, diag: Diag, n: usize, ap: [*]const T, x: [*]const T, incx: BlasInt, stride: usize, last: usize, output: []T) align(64) bool {
    @setFloatMode(.strict);
    const Bits = std.meta.Int(.unsigned, @bitSizeOf(T));
    const exponent: Bits = if (T == f32) 0x7f800000 else 0x7ff0000000000000;
    var i: usize = first_row;
    while (i + 1 < n) : (i += 2) {
        var sum0: T = 0;
        var sum1: T = 0;
        const xi = x[if (incx > 0) i * stride else last - i * stride];
        const xi1 = x[if (incx > 0) (i + 1) * stride else last - (i + 1) * stride];
        if (uplo == .upper) {
            const base0 = i * (i + 1) / 2;
            const base1 = base0 + i + 1;
            for (0..i) |j| {
                const xj = x[if (incx > 0) j * stride else last - j * stride];
                const product0 = ap[base0 + j] * xj;
                const product1 = ap[base1 + j] * xj;
                sum0 = sum0 + product0;
                sum1 = sum1 + product1;
            }
            const diagonal0: T = if (diag == .unit) 1 else ap[base0 + i];
            sum0 = sum0 + diagonal0 * xi;
            sum1 = sum1 + ap[base1 + i] * xi;
            const diagonal1: T = if (diag == .unit) 1 else ap[base1 + i + 1];
            sum1 = sum1 + diagonal1 * xi1;
        } else {
            const base0 = i * (2 * n - i + 1) / 2;
            const base1 = base0 + n - i;
            const diagonal0: T = if (diag == .unit) 1 else ap[base0];
            sum0 = sum0 + diagonal0 * xi;
            sum0 = sum0 + ap[base0 + 1] * xi1;
            const diagonal1: T = if (diag == .unit) 1 else ap[base1];
            sum1 = sum1 + diagonal1 * xi1;
            for (i + 2..n) |j| {
                const xj = x[if (incx > 0) j * stride else last - j * stride];
                const product0 = ap[base0 + j - i] * xj;
                const product1 = ap[base1 + j - i - 1] * xj;
                sum0 = sum0 + product0;
                sum1 = sum1 + product1;
            }
        }
        if ((@as(Bits, @bitCast(sum0)) & exponent) == exponent or sum0 == 0 or
            (@as(Bits, @bitCast(sum1)) & exponent) == exponent or sum1 == 0) return false;
        output[i] = sum0;
        output[i + 1] = sum1;
    }
    return true;
}

// Adjacent output rows share X loads and packed-column address arithmetic.
// They remain separate ordered sums, including each initial +0 addition. Only
// private output is written here; the caller commits after every row succeeds.
noinline fn finiteTpmvNoTransPairs(comptime T: type, first_row: usize, uplo: Uplo, diag: Diag, n: usize, ap: [*]const T, x: [*]const T, incx: BlasInt, stride: usize, last: usize, output: []T) align(64) bool {
    @setFloatMode(.strict);
    const Bits = std.meta.Int(.unsigned, @bitSizeOf(T));
    const exponent: Bits = if (T == f32) 0x7f800000 else 0x7ff0000000000000;
    var i: usize = first_row;
    while (i + 1 < n) : (i += 2) {
        var sum0: T = 0;
        var sum1: T = 0;
        const xi = x[if (incx > 0) i * stride else last - i * stride];
        const xi1 = x[if (incx > 0) (i + 1) * stride else last - (i + 1) * stride];
        if (uplo == .upper) {
            var offset = i * (i + 1) / 2 + i;
            const diagonal0: T = if (diag == .unit) 1 else ap[offset];
            const first0 = diagonal0 * xi;
            sum0 = sum0 + first0;
            offset += i + 1;
            const second0 = ap[offset] * xi1;
            sum0 = sum0 + second0;
            const diagonal1: T = if (diag == .unit) 1 else ap[offset + 1];
            const first1 = diagonal1 * xi1;
            sum1 = sum1 + first1;
            offset += i + 2;
            for (i + 2..n) |j| {
                const xj = x[if (incx > 0) j * stride else last - j * stride];
                const product0 = ap[offset] * xj;
                const product1 = ap[offset + 1] * xj;
                sum0 = sum0 + product0;
                sum1 = sum1 + product1;
                // The unused final offset is <= count + n, already bounded
                // by the caller's checked packed byte span.
                offset += j + 1;
            }
        } else {
            var offset = i;
            for (0..i) |j| {
                const xj = x[if (incx > 0) j * stride else last - j * stride];
                const product0 = ap[offset] * xj;
                const product1 = ap[offset + 1] * xj;
                sum0 = sum0 + product0;
                sum1 = sum1 + product1;
                offset += n - j - 1;
            }
            const diagonal0: T = if (diag == .unit) 1 else ap[offset];
            const last0 = diagonal0 * xi;
            sum0 = sum0 + last0;
            const penultimate1 = ap[offset + 1] * xi;
            sum1 = sum1 + penultimate1;
            offset += n - i;
            const diagonal1: T = if (diag == .unit) 1 else ap[offset];
            const last1 = diagonal1 * xi1;
            sum1 = sum1 + last1;
        }
        if ((@as(Bits, @bitCast(sum0)) & exponent) == exponent or sum0 == 0 or
            (@as(Bits, @bitCast(sum1)) & exponent) == exponent or sum1 == 0) return false;
        output[i] = sum0;
        output[i + 1] = sum1;
    }
    return true;
}

// Eight independent output rows hide accumulation latency without reordering
// terms within a row. Packed columns share the same logical vector interval.
inline fn splitTpmvColumnPairs(comptime T: type, comptime width: usize, pairs: [width]@Vector(2, T)) struct { first: @Vector(width, T), second: @Vector(width, T) } {
    var first: [width / 8]@Vector(8, T) = undefined;
    var second: [width / 8]@Vector(8, T) = undefined;
    inline for (0..width / 8) |group| {
        const r = group * 8;
        const join2 = @Vector(4, i32){ 0, 1, -1, -2 };
        const join4 = @Vector(8, i32){ 0, 1, 2, 3, -1, -2, -3, -4 };
        const low = @shuffle(T, @shuffle(T, pairs[r], pairs[r + 1], join2), @shuffle(T, pairs[r + 2], pairs[r + 3], join2), join4);
        const high = @shuffle(T, @shuffle(T, pairs[r + 4], pairs[r + 5], join2), @shuffle(T, pairs[r + 6], pairs[r + 7], join2), join4);
        first[group] = @shuffle(T, low, high, @Vector(8, i32){ 0, 2, 4, 6, -1, -3, -5, -7 });
        second[group] = @shuffle(T, low, high, @Vector(8, i32){ 1, 3, 5, 7, -2, -4, -6, -8 });
    }
    if (width == 8) return .{ .first = first[0], .second = second[0] };
    const join8 = @Vector(16, i32){ 0, 1, 2, 3, 4, 5, 6, 7, -1, -2, -3, -4, -5, -6, -7, -8 };
    return .{ .first = @shuffle(T, first[0], first[1], join8), .second = @shuffle(T, second[0], second[1], join8) };
}

inline fn transposedBoundaryFour(comptime T: type, comptime width: usize, comptime upper: bool, i: usize, diag: Diag, bases: [width]usize, ap: [*]const T, x: [*]const T, initial: @Vector(width, T)) @Vector(width, T) {
    @setFloatMode(.strict);
    var groups: [width / 4]@Vector(4, T) = undefined;
    inline for (0..width / 4) |group| {
        const offset = group * 4;
        const start = i + offset;
        var sums = @shuffle(T, initial, undefined, @Vector(4, i32){ offset, offset + 1, offset + 2, offset + 3 });
        if (upper) {
            var column = i;
            while (column < start) : (column += 2) {
                var pairs: [4]@Vector(2, T) = undefined;
                inline for (0..4) |r| pairs[r] = ap[bases[offset + r] + column ..][0..2].*;
                const low = @shuffle(T, pairs[0], pairs[1], @Vector(4, i32){ 0, 1, -1, -2 });
                const high = @shuffle(T, pairs[2], pairs[3], @Vector(4, i32){ 0, 1, -1, -2 });
                const split = .{
                    .first = @shuffle(T, low, high, @Vector(4, i32){ 0, 2, -1, -3 }),
                    .second = @shuffle(T, low, high, @Vector(4, i32){ 1, 3, -2, -4 }),
                };
                sums = sums + split.first * @as(@Vector(4, T), @splat(x[column]));
                sums = sums + split.second * @as(@Vector(4, T), @splat(x[column + 1]));
            }
        }
        inline for (0..4) |r| {
            const row = start + r;
            if (upper) {
                inline for (0..r) |c| sums[r] = sums[r] + ap[bases[offset + r] + start + c] * x[start + c];
            }
            const diagonal: T = if (diag == .unit) 1 else ap[bases[offset + r] + (if (upper) row else 0)];
            sums[r] = sums[r] + diagonal * x[row];
            if (!upper) {
                inline for (r + 1..4) |c| sums[r] = sums[r] + ap[bases[offset + r] + c - r] * x[start + c];
            }
        }
        if (!upper) {
            var column = start + 4;
            while (column < i + width) : (column += 2) {
                var pairs: [4]@Vector(2, T) = undefined;
                inline for (0..4) |r| pairs[r] = ap[bases[offset + r] + column - start - r ..][0..2].*;
                const low = @shuffle(T, pairs[0], pairs[1], @Vector(4, i32){ 0, 1, -1, -2 });
                const high = @shuffle(T, pairs[2], pairs[3], @Vector(4, i32){ 0, 1, -1, -2 });
                const split = .{
                    .first = @shuffle(T, low, high, @Vector(4, i32){ 0, 2, -1, -3 }),
                    .second = @shuffle(T, low, high, @Vector(4, i32){ 1, 3, -2, -4 }),
                };
                sums = sums + split.first * @as(@Vector(4, T), @splat(x[column]));
                sums = sums + split.second * @as(@Vector(4, T), @splat(x[column + 1]));
            }
        }
        groups[group] = sums;
    }
    const join = @Vector(8, i32){ 0, 1, 2, 3, -1, -2, -3, -4 };
    const low = @shuffle(T, groups[0], groups[1], join);
    if (width == 8) return low;
    const high = @shuffle(T, groups[2], groups[3], join);
    return @shuffle(T, low, high, @Vector(16, i32){ 0, 1, 2, 3, 4, 5, 6, 7, -1, -2, -3, -4, -5, -6, -7, -8 });
}

inline fn transposedBoundary16(comptime T: type, comptime upper: bool, i: usize, diag: Diag, bases: [16]usize, ap: [*]const T, x: [*]const T, initial: @Vector(16, T)) @Vector(16, T) {
    if (T == f32) return transposedBoundaryFour(T, 16, upper, i, diag, bases, ap, x, initial);
    @setFloatMode(.strict);
    var groups: [2]@Vector(8, T) = undefined;
    inline for (0..2) |group| {
        const offset = group * 8;
        const start = i + offset;
        var sums = @shuffle(T, initial, undefined, @Vector(8, i32){ offset, offset + 1, offset + 2, offset + 3, offset + 4, offset + 5, offset + 6, offset + 7 });
        if (upper) {
            var column = i;
            while (column < start) : (column += 2) {
                var pairs: [8]@Vector(2, T) = undefined;
                inline for (0..8) |r| pairs[r] = ap[bases[offset + r] + column ..][0..2].*;
                const split = splitTpmvColumnPairs(T, 8, pairs);
                sums = sums + split.first * @as(@Vector(8, T), @splat(x[column]));
                sums = sums + split.second * @as(@Vector(8, T), @splat(x[column + 1]));
            }
        }
        inline for (0..8) |r| {
            const row = start + r;
            if (upper) {
                inline for (0..r) |c| sums[r] = sums[r] + ap[bases[offset + r] + start + c] * x[start + c];
            }
            const diagonal: T = if (diag == .unit) 1 else ap[bases[offset + r] + (if (upper) row else 0)];
            sums[r] = sums[r] + diagonal * x[row];
            if (!upper) {
                inline for (r + 1..8) |c| sums[r] = sums[r] + ap[bases[offset + r] + c - r] * x[start + c];
            }
        }
        if (!upper) {
            var column = start + 8;
            while (column < i + 16) : (column += 2) {
                var pairs: [8]@Vector(2, T) = undefined;
                inline for (0..8) |r| pairs[r] = ap[bases[offset + r] + column - start - r ..][0..2].*;
                const split = splitTpmvColumnPairs(T, 8, pairs);
                sums = sums + split.first * @as(@Vector(8, T), @splat(x[column]));
                sums = sums + split.second * @as(@Vector(8, T), @splat(x[column + 1]));
            }
        }
        groups[group] = sums;
    }
    return @shuffle(T, groups[0], groups[1], @Vector(16, i32){ 0, 1, 2, 3, 4, 5, 6, 7, -1, -2, -3, -4, -5, -6, -7, -8 });
}

inline fn finiteTpmvTransRows(comptime T: type, comptime width: usize, comptime unit_stride: bool, first_row: usize, uplo: Uplo, diag: Diag, n: usize, ap: [*]const T, x: [*]const T, incx: BlasInt, stride: usize, last: usize, output: []T) bool {
    // Resolve the diagonal case once for contiguous f32 and eight-row f64
    // blocks so the boundary kernels can specialize their diagonal handling.
    if ((T == f32 or (T == f64 and width == 8)) and unit_stride and diag == .unit) return finiteTpmvTransRowsPrefetch(T, width, unit_stride, 16, true, first_row, uplo, diag, n, ap, x, incx, stride, last, output);
    if (T == f64 and width == 16 and n >= 4096) return finiteTpmvTransRowsPrefetch(T, width, unit_stride, 32, false, first_row, uplo, diag, n, ap, x, incx, stride, last, output);
    return finiteTpmvTransRowsPrefetch(T, width, unit_stride, 16, false, first_row, uplo, diag, n, ap, x, incx, stride, last, output);
}

inline fn finiteTpmvTransRowsPrefetch(comptime T: type, comptime width: usize, comptime unit_stride: bool, comptime prefetch_distance: usize, comptime unit_diagonal: bool, first_row: usize, uplo: Uplo, diagonal_arg: Diag, n: usize, ap: [*]const T, x: [*]const T, incx: BlasInt, stride: usize, last: usize, output: []T) align(64) bool {
    if (T == f32 and unit_stride and (width == 16 or unit_diagonal)) {
        if (uplo == .upper) return finiteTpmvTransRowsSelected(T, width, unit_stride, prefetch_distance, unit_diagonal, .upper, first_row, uplo, diagonal_arg, n, ap, x, incx, stride, last, output);
        return finiteTpmvTransRowsSelected(T, width, unit_stride, prefetch_distance, unit_diagonal, .lower, first_row, uplo, diagonal_arg, n, ap, x, incx, stride, last, output);
    }
    return finiteTpmvTransRowsSelected(T, width, unit_stride, prefetch_distance, unit_diagonal, null, first_row, uplo, diagonal_arg, n, ap, x, incx, stride, last, output);
}

inline fn finiteTpmvTransRowsSelected(comptime T: type, comptime width: usize, comptime unit_stride: bool, comptime prefetch_distance: usize, comptime unit_diagonal: bool, comptime selected_uplo: ?Uplo, first_row: usize, uplo_arg: Uplo, diagonal_arg: Diag, n: usize, ap: [*]const T, x: [*]const T, incx: BlasInt, stride: usize, last: usize, output: []T) align(64) bool {
    if (T == f32 and width == 16 and unit_stride and selected_uplo == .upper and n <= 256) {
        return finiteTpmvTransRowsSelectedImpl(T, width, unit_stride, prefetch_distance, unit_diagonal, selected_uplo, true, first_row, uplo_arg, diagonal_arg, n, ap, x, incx, stride, last, output);
    }
    return finiteTpmvTransRowsSelectedImpl(T, width, unit_stride, prefetch_distance, unit_diagonal, selected_uplo, false, first_row, uplo_arg, diagonal_arg, n, ap, x, incx, stride, last, output);
}

noinline fn finiteTpmvTransRowsSelectedImpl(comptime T: type, comptime width: usize, comptime unit_stride: bool, comptime prefetch_distance: usize, comptime unit_diagonal: bool, comptime selected_uplo: ?Uplo, comptime grouped_rows: bool, first_row: usize, uplo_arg: Uplo, diagonal_arg: Diag, n: usize, ap: [*]const T, x: [*]const T, incx: BlasInt, stride: usize, last: usize, output: []T) align(64) bool {
    const uplo = selected_uplo orelse uplo_arg;
    @setFloatMode(.strict);
    const diag: Diag = if (unit_diagonal) .unit else diagonal_arg;
    const Bits = std.meta.Int(.unsigned, @bitSizeOf(T));
    const exponent: Bits = if (T == f32) 0x7f800000 else 0x7ff0000000000000;
    var i: usize = first_row;
    while (i + width - 1 < n) : (i += width) {
        var sums: @Vector(width, T) = @splat(0);
        var bases: [width]usize = undefined;
        if (T == f64 and width == 8) {
            if (uplo == .upper) {
                bases[0] = i * (i + 1) / 2;
                inline for (1..width) |r| bases[r] = bases[r - 1] + i + r;
            } else {
                bases[0] = i * (2 * n - i + 1) / 2;
                inline for (1..width) |r| bases[r] = bases[r - 1] + n - i - r + 1;
            }
        } else if (T == f32 and width == 16 and selected_uplo == .upper) {
            bases[0] = i * (i + 1) / 2;
            inline for (1..width) |r| bases[r] = bases[r - 1] + i + r;
        } else if (T == f32 and width == 16 and uplo == .lower) {
            bases[0] = i * (2 * n - i + 1) / 2;
            inline for (1..width) |r| bases[r] = bases[r - 1] + n - i - r + 1;
        } else {
            inline for (0..width) |r| {
                const row = i + r;
                bases[r] = if (uplo == .upper) row * (row + 1) / 2 else row * (2 * n - row + 1) / 2;
            }
        }
        if (uplo == .upper) {
            var column: usize = 0;
            const paired_columns = i & ~@as(usize, 1);
            while (column < paired_columns) : (column += 2) {
                if (T == f64 and width == 16 and (column & (if (prefetch_distance == 32) @as(usize, 15) else 7)) == 0 and column + prefetch_distance < i) {
                    inline for (0..width) |r| @prefetch(ap + bases[r] + column + prefetch_distance, .{ .rw = .read, .locality = 3, .cache = .data });
                }
                if (grouped_rows) {
                    // Share one input-pair load across the row groups.
                    const input_pair: @Vector(2, T) = if (comptime builtin.cpu.arch == .aarch64 and builtin.cpu.hasAll(.aarch64, &.{ .neon, .fp_armv8 }) and T == f32)
                        asm volatile ("ldr %[pair:d], [%[ptr]]"
                            : [pair] "=w" (-> @Vector(2, f32)),
                            : [ptr] "r" (x + column),
                            : .{ .memory = true })
                    else
                        .{ x[column], x[column + 1] };
                    var groups: [2]@Vector(8, T) = undefined;
                    inline for (0..2) |group| {
                        const offset = group * 8;
                        var split: struct { first: @Vector(8, T), second: @Vector(8, T) } = undefined;
                        if (comptime T == f32 and builtin.cpu.arch == .aarch64 and builtin.cpu.hasAll(.aarch64, &.{ .neon, .fp_armv8 })) {
                            var joined: [4]@Vector(4, T) = undefined;
                            inline for (0..4) |pair| {
                                var address: usize = undefined;
                                var value: @Vector(4, T) = undefined;
                                asm volatile (
                                    \\ldr %[value:d], [%[first], %[index]]
                                    \\add %[address], %[second], %[index]
                                    \\ld1 {%[value].d}[1], [%[address]]
                                    : [value] "=&w" (value),
                                      [address] "=&r" (address),
                                    : [first] "r" (ap + bases[offset + 2 * pair]),
                                      [second] "r" (ap + bases[offset + 2 * pair + 1]),
                                      [index] "r" (column * 4),
                                    : .{ .memory = true });
                                joined[pair] = value;
                            }
                            const first_low = @shuffle(T, joined[0], joined[1], @Vector(4, i32){ 0, 2, -1, -3 });
                            const first_high = @shuffle(T, joined[2], joined[3], @Vector(4, i32){ 0, 2, -1, -3 });
                            const second_low = @shuffle(T, joined[0], joined[1], @Vector(4, i32){ 1, 3, -2, -4 });
                            const second_high = @shuffle(T, joined[2], joined[3], @Vector(4, i32){ 1, 3, -2, -4 });
                            split.first = @shuffle(T, first_low, first_high, @Vector(8, i32){ 0, 1, 2, 3, -1, -2, -3, -4 });
                            split.second = @shuffle(T, second_low, second_high, @Vector(8, i32){ 0, 1, 2, 3, -1, -2, -3, -4 });
                        } else {
                            var pairs: [8]@Vector(2, T) = undefined;
                            inline for (0..8) |r| pairs[r] = ap[bases[offset + r] + column ..][0..2].*;
                            const loaded = splitTpmvColumnPairs(T, 8, pairs);
                            split = .{ .first = loaded.first, .second = loaded.second };
                        }
                        var part = @shuffle(T, sums, undefined, @Vector(8, i32){ offset, offset + 1, offset + 2, offset + 3, offset + 4, offset + 5, offset + 6, offset + 7 });
                        const second_product = split.second * @as(@Vector(8, T), @splat(input_pair[1]));
                        part = part + split.first * @as(@Vector(8, T), @splat(input_pair[0]));
                        part = part + second_product;
                        groups[group] = part;
                    }
                    sums = @shuffle(T, groups[0], groups[1], @Vector(16, i32){ 0, 1, 2, 3, 4, 5, 6, 7, -1, -2, -3, -4, -5, -6, -7, -8 });
                } else {
                    if (T == f64 and width == 8 and unit_stride and !unit_diagonal) {
                        const input_pair: @Vector(2, T) = if (comptime builtin.cpu.arch == .aarch64 and builtin.cpu.hasAll(.aarch64, &.{ .neon, .fp_armv8 }))
                            asm volatile ("ldr %[pair:q], [%[ptr]]"
                                : [pair] "=w" (-> @Vector(2, f64)),
                                : [ptr] "r" (x + column),
                                : .{ .memory = true })
                        else
                            .{ x[column], x[column + 1] };
                        var groups: [2]@Vector(4, T) = undefined;
                        inline for (0..2) |group| {
                            const offset = group * 4;
                            var pairs: [4]@Vector(2, T) = undefined;
                            inline for (0..4) |r| pairs[r] = ap[bases[offset + r] + column ..][0..2].*;
                            const join = @Vector(4, i32){ 0, 1, -1, -2 };
                            const low = @shuffle(T, pairs[0], pairs[1], join);
                            const high = @shuffle(T, pairs[2], pairs[3], join);
                            const first = @shuffle(T, low, high, @Vector(4, i32){ 0, 2, -1, -3 });
                            const second = @shuffle(T, low, high, @Vector(4, i32){ 1, 3, -2, -4 });
                            var value = @shuffle(T, sums, undefined, @Vector(4, i32){ offset, offset + 1, offset + 2, offset + 3 });
                            const second_product = second * @as(@Vector(4, T), @splat(input_pair[1]));
                            value = value + first * @as(@Vector(4, T), @splat(input_pair[0]));
                            value = value + second_product;
                            groups[group] = value;
                        }
                        sums = @shuffle(T, groups[0], groups[1], @Vector(8, i32){ 0, 1, 2, 3, -1, -2, -3, -4 });
                    } else {
                        var pairs: [width]@Vector(2, T) = undefined;
                        inline for (0..width) |r| pairs[r] = ap[bases[r] + column ..][0..2].*;
                        const split = splitTpmvColumnPairs(T, width, pairs);
                        const first = split.first;
                        const second = split.second;
                        if (comptime builtin.cpu.arch == .aarch64 and builtin.cpu.hasAll(.aarch64, &.{ .neon, .fp_armv8 }) and T == f32 and width == 16 and unit_stride) {
                            const input_pair = asm volatile ("ldr %[pair:d], [%[ptr]]"
                                : [pair] "=w" (-> @Vector(2, f32)),
                                : [ptr] "r" (x + column),
                                : .{ .memory = true });
                            sums = sums + @as(@Vector(width, T), first) * @as(@Vector(width, T), @splat(input_pair[0]));
                            sums = sums + @as(@Vector(width, T), second) * @as(@Vector(width, T), @splat(input_pair[1]));
                        } else if (comptime builtin.cpu.arch == .aarch64 and builtin.cpu.hasAll(.aarch64, &.{ .neon, .fp_armv8 }) and T == f64 and (width == 8 or width == 16) and unit_stride) {
                            const input_pair = asm volatile ("ldr %[pair:q], [%[ptr]]"
                                : [pair] "=w" (-> @Vector(2, f64)),
                                : [ptr] "r" (x + column),
                                : .{ .memory = true });
                            sums = sums + @as(@Vector(width, T), first) * @as(@Vector(width, T), @splat(input_pair[0]));
                            sums = sums + @as(@Vector(width, T), second) * @as(@Vector(width, T), @splat(input_pair[1]));
                        } else {
                            const x0 = x[if (unit_stride) column else if (incx > 0) column * stride else last - column * stride];
                            const x1 = x[if (unit_stride) column + 1 else if (incx > 0) (column + 1) * stride else last - (column + 1) * stride];
                            sums = sums + @as(@Vector(width, T), first) * @as(@Vector(width, T), @splat(x0));
                            sums = sums + @as(@Vector(width, T), second) * @as(@Vector(width, T), @splat(x1));
                        }
                    }
                }
            }
            if (column < i) {
                const xj = x[if (unit_stride) column else if (incx > 0) column * stride else last - column * stride];
                var coefficients: [width]T = undefined;
                inline for (0..width) |r| coefficients[r] = ap[bases[r] + column];
                sums = sums + @as(@Vector(width, T), coefficients) * @as(@Vector(width, T), @splat(xj));
            }
            if (width == 16 and unit_stride) {
                sums = transposedBoundary16(T, true, i, diag, bases, ap, x, sums);
            } else if ((T == f32 or T == f64) and width == 8 and unit_stride) {
                sums = transposedBoundaryFour(T, 8, true, i, diag, bases, ap, x, sums);
            } else {
                inline for (0..width) |r| {
                    inline for (0..r) |c| {
                        const j = i + c;
                        sums[r] = sums[r] + ap[bases[r] + j] * x[if (unit_stride) j else if (incx > 0) j * stride else last - j * stride];
                    }
                    const row = i + r;
                    const diagonal: T = if (diag == .unit) 1 else ap[bases[r] + row];
                    sums[r] = sums[r] + diagonal * x[if (unit_stride) row else if (incx > 0) row * stride else last - row * stride];
                }
            }
        } else {
            if (width == 16 and unit_stride) {
                sums = transposedBoundary16(T, false, i, diag, bases, ap, x, sums);
            } else if ((T == f32 or T == f64) and width == 8 and unit_stride) {
                sums = transposedBoundaryFour(T, 8, false, i, diag, bases, ap, x, sums);
            } else {
                inline for (0..width) |r| {
                    const row = i + r;
                    const diagonal: T = if (diag == .unit) 1 else ap[bases[r]];
                    sums[r] = sums[r] + diagonal * x[if (unit_stride) row else if (incx > 0) row * stride else last - row * stride];
                    inline for (r + 1..width) |c| {
                        const j = i + c;
                        sums[r] = sums[r] + ap[bases[r] + c - r] * x[if (unit_stride) j else if (incx > 0) j * stride else last - j * stride];
                    }
                }
            }
            var column: usize = i + width;
            const paired_columns = n - ((n - column) & 1);
            while (column < paired_columns) : (column += 2) {
                if (T == f64 and width == 16 and (column & (if (prefetch_distance == 32) @as(usize, 15) else 7)) == 0 and column + prefetch_distance < n) {
                    inline for (0..width) |r| @prefetch(ap + bases[r] + column + prefetch_distance - i - r, .{ .rw = .read, .locality = 3, .cache = .data });
                }
                if (T == f64 and width == 16 and unit_stride) {
                    var groups: [2]@Vector(8, T) = undefined;
                    inline for (0..2) |group| {
                        const offset = group * 8;
                        var pairs: [8]@Vector(2, T) = undefined;
                        inline for (0..8) |r| pairs[r] = ap[bases[offset + r] + column - i - offset - r ..][0..2].*;
                        const split = splitTpmvColumnPairs(T, 8, pairs);
                        var part = @shuffle(T, sums, undefined, @Vector(8, i32){ offset, offset + 1, offset + 2, offset + 3, offset + 4, offset + 5, offset + 6, offset + 7 });
                        part = part + split.first * @as(@Vector(8, T), @splat(x[column]));
                        part = part + split.second * @as(@Vector(8, T), @splat(x[column + 1]));
                        groups[group] = part;
                    }
                    sums = @shuffle(T, groups[0], groups[1], @Vector(16, i32){ 0, 1, 2, 3, 4, 5, 6, 7, -1, -2, -3, -4, -5, -6, -7, -8 });
                } else if (T == f64 and width == 8 and unit_stride and !unit_diagonal) {
                    const input_pair: @Vector(2, T) = if (comptime builtin.cpu.arch == .aarch64 and builtin.cpu.hasAll(.aarch64, &.{ .neon, .fp_armv8 }))
                        asm volatile ("ldr %[pair:q], [%[ptr]]"
                            : [pair] "=w" (-> @Vector(2, f64)),
                            : [ptr] "r" (x + column),
                            : .{ .memory = true })
                    else
                        .{ x[column], x[column + 1] };
                    var groups: [2]@Vector(4, T) = undefined;
                    inline for (0..2) |group| {
                        const offset = group * 4;
                        var pairs: [4]@Vector(2, T) = undefined;
                        inline for (0..4) |r| pairs[r] = ap[bases[offset + r] + column - i - offset - r ..][0..2].*;
                        const join = @Vector(4, i32){ 0, 1, -1, -2 };
                        const low = @shuffle(T, pairs[0], pairs[1], join);
                        const high = @shuffle(T, pairs[2], pairs[3], join);
                        const first = @shuffle(T, low, high, @Vector(4, i32){ 0, 2, -1, -3 });
                        const second = @shuffle(T, low, high, @Vector(4, i32){ 1, 3, -2, -4 });
                        var value = @shuffle(T, sums, undefined, @Vector(4, i32){ offset, offset + 1, offset + 2, offset + 3 });
                        const second_product = second * @as(@Vector(4, T), @splat(input_pair[1]));
                        value = value + first * @as(@Vector(4, T), @splat(input_pair[0]));
                        value = value + second_product;
                        groups[group] = value;
                    }
                    sums = @shuffle(T, groups[0], groups[1], @Vector(8, i32){ 0, 1, 2, 3, -1, -2, -3, -4 });
                } else {
                    var pairs: [width]@Vector(2, T) = undefined;
                    inline for (0..width) |r| pairs[r] = ap[bases[r] + column - i - r ..][0..2].*;
                    const split = splitTpmvColumnPairs(T, width, pairs);
                    const first = split.first;
                    const second = split.second;
                    if (comptime builtin.cpu.arch == .aarch64 and builtin.cpu.hasAll(.aarch64, &.{ .neon, .fp_armv8 }) and T == f32 and width == 16 and unit_stride) {
                        // Keep the pair in one load instead of separate broadcast/scalar
                        // loads. The paired-column bound guarantees both inputs exist.
                        const input_pair = asm volatile ("ldr %[pair:d], [%[ptr]]"
                            : [pair] "=w" (-> @Vector(2, f32)),
                            : [ptr] "r" (x + column),
                            : .{ .memory = true });
                        sums = sums + @as(@Vector(width, T), first) * @as(@Vector(width, T), @splat(input_pair[0]));
                        sums = sums + @as(@Vector(width, T), second) * @as(@Vector(width, T), @splat(input_pair[1]));
                    } else if (comptime builtin.cpu.arch == .aarch64 and builtin.cpu.hasAll(.aarch64, &.{ .neon, .fp_armv8 }) and T == f64 and width == 8 and unit_stride) {
                        const input_pair = asm volatile ("ldr %[pair:q], [%[ptr]]"
                            : [pair] "=w" (-> @Vector(2, f64)),
                            : [ptr] "r" (x + column),
                            : .{ .memory = true });
                        sums = sums + @as(@Vector(width, T), first) * @as(@Vector(width, T), @splat(input_pair[0]));
                        sums = sums + @as(@Vector(width, T), second) * @as(@Vector(width, T), @splat(input_pair[1]));
                    } else {
                        const x0 = x[if (unit_stride) column else if (incx > 0) column * stride else last - column * stride];
                        const x1 = x[if (unit_stride) column + 1 else if (incx > 0) (column + 1) * stride else last - (column + 1) * stride];
                        sums = sums + @as(@Vector(width, T), first) * @as(@Vector(width, T), @splat(x0));
                        sums = sums + @as(@Vector(width, T), second) * @as(@Vector(width, T), @splat(x1));
                    }
                }
            }
            if (column < n) {
                const xj = x[if (unit_stride) column else if (incx > 0) column * stride else last - column * stride];
                var coefficients: [width]T = undefined;
                inline for (0..width) |r| coefficients[r] = ap[bases[r] + column - i - r];
                sums = sums + @as(@Vector(width, T), coefficients) * @as(@Vector(width, T), @splat(xj));
            }
        }
        // Normal results need no floating comparison. Preserve the scalar
        // checks for zero, subnormal and non-finite lanes, including FP flags.
        if (T == f32 and unit_stride) {
            const exponents = @as(@Vector(width, Bits), @bitCast(sums)) & @as(@Vector(width, Bits), @splat(exponent));
            const exceptional = (exponents == @as(@Vector(width, Bits), @splat(0))) | (exponents == @as(@Vector(width, Bits), @splat(exponent)));
            if (!@reduce(.Or, exceptional)) {
                output[i..][0..width].* = sums;
                continue;
            }
        }
        inline for (0..width) |r| {
            if ((@as(Bits, @bitCast(sums[r])) & exponent) == exponent or sums[r] == 0) return false;
            output[i + r] = sums[r];
        }
    }
    return true;
}

// Adjacent rows share packed-column loads while preserving each sum's order.
inline fn finiteTpmvNoTransRows(comptime T: type, comptime width: usize, first_row: usize, uplo: Uplo, diag: Diag, n: usize, ap: [*]const T, x: [*]const T, incx: BlasInt, stride: usize, last: usize, output: []T) bool {
    if (comptime T == f64 or T == f32) {
        if (incx == 1 and uplo == .upper) return finiteTpmvNoTransRowsImpl(T, width, true, first_row, uplo, diag, n, ap, x, incx, stride, last, output);
        // Lower f32 rows below 512 benefit from fixed-stride addressing and
        // unrolling; larger calls showed no consistent gain over the generic leaf.
        if (T == f32 and n <= 511 and incx == 1 and uplo == .lower) return finiteTpmvNoTransRowsImpl(T, width, true, first_row, uplo, diag, n, ap, x, incx, stride, last, output);
    }
    return finiteTpmvNoTransRowsImpl(T, width, false, first_row, uplo, diag, n, ap, x, incx, stride, last, output);
}

inline fn finiteTpmvNoTransRowsImpl(comptime T: type, comptime width: usize, comptime unit_stride: bool, first_row: usize, uplo: Uplo, diag: Diag, n: usize, ap: [*]const T, x: [*]const T, incx: BlasInt, stride: usize, last: usize, output: []T) bool {
    return FiniteTpmvRowsLeaf(T, width, unit_stride).run(first_row, uplo, diag, n, ap, x, incx, stride, last, output);
}

// Separate the strided f32 eight-row triangles so their boundary calculations
// do not share register live ranges with the other triangle's common loop.
fn FiniteTpmvLowerStridedRowsLeaf(comptime width: usize) type {
    return struct {
        noinline fn run(first_row: usize, diag: Diag, n: usize, ap: [*]const f32, x: [*]const f32, incx: BlasInt, stride: usize, last: usize, output: []f32) align(128) bool {
            @setFloatMode(.strict);
            // The caller checked the input span; wrapping represents a negative
            // step without a direction test in each common-column iteration.
            const input_step: usize = if (incx > 0) stride else 0 -% stride;
            var i: usize = first_row;
            while (i + width - 1 < n) : (i += width) {
                var sums: @Vector(width, f32) = @splat(0);
                var offset = i;
                var input_index: usize = if (incx > 0) 0 else last;
                var column_step = n - 1;
                var remaining = i;
                while (remaining != 0) : (remaining -= 1) {
                    // Keep a scalar load: the compiler's post-index LD1R form
                    // was slower in the measured strided lower-triangle cases.
                    const xj = if (comptime builtin.cpu.arch == .aarch64 and builtin.cpu.hasAll(.aarch64, &.{ .neon, .fp_armv8 }))
                        asm volatile ("ldr %[value:s], [%[source]]"
                            : [value] "=w" (-> f32),
                            : [source] "r" (x + input_index),
                            : .{ .memory = true })
                    else
                        x[input_index];
                    sums = sums + @as(@Vector(width, f32), ap[offset..][0..width].*) * @as(@Vector(width, f32), @splat(xj));
                    offset += column_step;
                    column_step -= 1;
                    input_index +%= input_step;
                }
                inline for (0..width) |c| {
                    const j = i + c;
                    const xj = x[if (incx > 0) j * stride else last - j * stride];
                    const base = j * (2 * n - j + 1) / 2;
                    inline for (c..width) |r| {
                        const av: f32 = if (r == c and diag == .unit) 1 else ap[base + r - c];
                        sums[r] = sums[r] + av * xj;
                    }
                }
                inline for (0..width) |r| {
                    if ((@as(u32, @bitCast(sums[r])) & 0x7f800000) == 0x7f800000 or sums[r] == 0) return false;
                    output[i + r] = sums[r];
                }
            }
            return true;
        }
    };
}

fn FiniteTpmvUpperStridedRowsLeaf(comptime width: usize, comptime unit_diagonal: bool) type {
    return struct {
        noinline fn run(first_row: usize, n: usize, ap: [*]const f32, x: [*]const f32, incx: BlasInt, stride: usize, last: usize, output: []f32) align(128) bool {
            @setFloatMode(.strict);
            var i: usize = first_row;
            while (i + width - 1 < n) : (i += width) {
                var sums: @Vector(width, f32) = @splat(0);
                inline for (0..width) |c| {
                    const j = i + c;
                    const xj = x[if (incx > 0) j * stride else last - j * stride];
                    const base = j * (j + 1) / 2 + i;
                    inline for (0..c + 1) |r| {
                        const av: f32 = if (unit_diagonal and r == c) 1 else ap[base + r];
                        sums[r] = sums[r] + av * xj;
                    }
                }
                var offset = (i + width) * (i + width + 1) / 2 + i;
                for (i + width..n) |j| {
                    const xj = x[if (incx > 0) j * stride else last - j * stride];
                    sums = sums + @as(@Vector(width, f32), ap[offset..][0..width].*) * @as(@Vector(width, f32), @splat(xj));
                    offset += j + 1;
                }
                inline for (0..width) |r| {
                    if ((@as(u32, @bitCast(sums[r])) & 0x7f800000) == 0x7f800000 or sums[r] == 0) return false;
                    output[i + r] = sums[r];
                }
            }
            return true;
        }
    };
}

fn FiniteTpmvRowsLeaf(comptime T: type, comptime width: usize, comptime unit_stride: bool) type {
    return struct {
        noinline fn run(first_row: usize, uplo: Uplo, diag: Diag, n: usize, ap: [*]const T, x: [*]const T, incx: BlasInt, stride: usize, last: usize, output: []T) align(if (T == f32 and width == 8 and !unit_stride) 128 else 64) bool {
            @setFloatMode(.strict);
            const Bits = std.meta.Int(.unsigned, @bitSizeOf(T));
            const exponent: Bits = if (T == f32) 0x7f800000 else 0x7ff0000000000000;
            var i: usize = first_row;
            while (i + width - 1 < n) : (i += width) {
                var sums: @Vector(width, T) = @splat(0);
                if (uplo == .upper) {
                    if (comptime T == f32 and width == 8 and !unit_stride) {
                        if (diag == .unit) return FiniteTpmvUpperStridedRowsLeaf(width, true).run(i, n, ap, x, incx, stride, last, output);
                        return FiniteTpmvUpperStridedRowsLeaf(width, false).run(i, n, ap, x, incx, stride, last, output);
                    }
                    inline for (0..width) |c| {
                        const j = i + c;
                        const xj = x[if (unit_stride) j else if (incx > 0) j * stride else last - j * stride];
                        const base = j * (j + 1) / 2 + i;
                        inline for (0..c + 1) |r| {
                            const av: T = if (r == c and diag == .unit) 1 else ap[base + r];
                            sums[r] = sums[r] + av * xj;
                        }
                    }
                    var offset = (i + width) * (i + width + 1) / 2 + i;
                    if (comptime unit_stride) {
                        var j = i + width;
                        while (n - j >= 4) : (j += 4) {
                            inline for (0..2) |pair| {
                                const term = 2 * pair;
                                const first = @as(@Vector(width, T), ap[offset..][0..width].*) * @as(@Vector(width, T), @splat(x[j + term]));
                                offset += j + term + 1;
                                const second = @as(@Vector(width, T), ap[offset..][0..width].*) * @as(@Vector(width, T), @splat(x[j + term + 1]));
                                offset += j + term + 2;
                                sums = sums + first;
                                sums = sums + second;
                            }
                        }
                        while (j < n) : (j += 1) {
                            sums = sums + @as(@Vector(width, T), ap[offset..][0..width].*) * @as(@Vector(width, T), @splat(x[j]));
                            offset += j + 1;
                        }
                    } else {
                        for (i + width..n) |j| {
                            const xj = x[if (unit_stride) j else if (incx > 0) j * stride else last - j * stride];
                            sums = sums + @as(@Vector(width, T), ap[offset..][0..width].*) * @as(@Vector(width, T), @splat(xj));
                            offset += j + 1;
                        }
                    }
                } else {
                    if (comptime T == f32 and width == 8 and !unit_stride) {
                        return FiniteTpmvLowerStridedRowsLeaf(width).run(i, diag, n, ap, x, incx, stride, last, output);
                    }
                    var offset = i;
                    var column: usize = 0;
                    if (comptime T == f32 and unit_stride) {
                        while (i - column >= 4) : (column += 4) {
                            inline for (0..4) |term| {
                                sums = sums + @as(@Vector(width, T), ap[offset..][0..width].*) * @as(@Vector(width, T), @splat(x[column + term]));
                                offset += n - column - term - 1;
                            }
                        }
                    }
                    for (column..i) |j| {
                        const xj = x[if (unit_stride) j else if (incx > 0) j * stride else last - j * stride];
                        sums = sums + @as(@Vector(width, T), ap[offset..][0..width].*) * @as(@Vector(width, T), @splat(xj));
                        offset += n - j - 1;
                    }
                    inline for (0..width) |c| {
                        const j = i + c;
                        const xj = x[if (unit_stride) j else if (incx > 0) j * stride else last - j * stride];
                        const base = j * (2 * n - j + 1) / 2;
                        inline for (c..width) |r| {
                            const av: T = if (r == c and diag == .unit) 1 else ap[base + r - c];
                            sums[r] = sums[r] + av * xj;
                        }
                    }
                }
                inline for (0..width) |r| {
                    if ((@as(Bits, @bitCast(sums[r])) & exponent) == exponent or sums[r] == 0) return false;
                    output[i + r] = sums[r];
                }
            }
            return true;
        }
    };
}
