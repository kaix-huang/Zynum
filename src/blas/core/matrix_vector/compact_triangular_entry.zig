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
    if (tryFiniteTpmv(T, std.heap.c_allocator, profile.finite_tpmv_workspace_max_bytes, uplo, trans_, diag, n, ap, x, incx)) return;
    return legacyTpmv(T, uplo, trans_, diag, n, ap, x, incx);
}

// Isolate the original full-range body from candidate-call register pressure.
// This leaf deliberately retains the old arguments and arithmetic unchanged.
// Compile-time selection adds no runtime wrapper or branch. Only the measured
// Mac real leaves request 64-byte alignment; other leaves retain the default.
inline fn legacyTpmv(comptime T: type, uplo: Uplo, trans_: Order, diag: Diag, n: BlasInt, ap: [*]const T, x: [*]T, incx: BlasInt) void {
    if (comptime builtin.cpu.arch == .aarch64 and builtin.os.tag == .macos and (T == f32 or T == f64)) {
        return legacyTpmvAligned(T, uplo, trans_, diag, n, ap, x, incx);
    }
    return legacyTpmvDefault(T, uplo, trans_, diag, n, ap, x, incx);
}

noinline fn legacyTpmvAligned(comptime T: type, uplo: Uplo, trans_: Order, diag: Diag, n: BlasInt, ap: [*]const T, x: [*]T, incx: BlasInt) align(64) void {
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
    const effective_upper = (trans_ == .no_trans and uplo == .upper) or
        (trans_ != .no_trans and uplo == .lower);
    if (n_ >= 128 and k_ >= 16 and k_ <= @divTrunc(n_, 4) and !effective_upper) {
        return tryFiniteTbsvImpl(T, true, allocator, max_bytes, uplo, trans_, diag, n_, k_, a, lda, x, incx);
    }
    return tryFiniteTbsvImpl(T, false, allocator, max_bytes, uplo, trans_, diag, n_, k_, a, lda, x, incx);
}

noinline fn tryFiniteTbsvImpl(
    comptime T: type,
    comptime unroll: bool,
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
    for (0..n) |i| work[i] = x[if (incx > 0) i * stride else last - i * stride];
    const upper = (trans_ == .no_trans and uplo == .upper) or
        (trans_ != .no_trans and uplo == .lower);
    // Previously solved values are finite, including signed zeros. At a zero
    // accumulator, each sign present in a skipped region needs one subtraction:
    // repeated same-sign zero subtractions are idempotent under IEEE arithmetic.
    var first_iteration = [_]usize{ n, n };
    var first_index = [_]usize{ 0, 0 };
    for (0..n) |iteration| {
        const i = if (upper) n - iteration - 1 else iteration;
        const begin = if (upper) i + 1 else i - @min(i, k);
        const end = if (upper) i + 1 + @min(k, n - i - 1) else i;
        var value = work[i];
        if (!upper and value == 0) {
            for (0..2) |sign| {
                if (first_iteration[sign] < begin) value = value - @as(T, 0) * work[first_index[sign]];
            }
        }
        if (begin < end) {
            const row = if (trans_ == .no_trans) i else begin;
            const col = if (trans_ == .no_trans) begin else i;
            var offset = if (uplo == .upper) k + row - col + col * ld else row - col + col * ld;
            const step = if (trans_ == .no_trans) ld - 1 else 1;
            if (comptime unroll) {
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

            } else {
                for (begin..end) |j| {
                    const product = a[offset] * work[j];
                    value = value - product;
                    if (j + 1 < end) offset += step;
                }
            }
        }
        if (upper and value == 0) {
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
        const sign: usize = @intCast(@as(Bits, @bitCast(value)) >> (@bitSizeOf(T) - 1));
        if (first_iteration[sign] == n) {
            first_iteration[sign] = iteration;
            first_index[sign] = i;
        }
    }
    for (0..n) |i| x[if (incx > 0) i * stride else last - i * stride] = work[i];
    return true;
}

pub const testing = struct {
    pub fn forceTpmv(comptime T: type, implementation: catalog.Implementation, allocator: std.mem.Allocator, max_bytes: usize, uplo: Uplo, trans_: Order, diag: Diag, n: BlasInt, ap: [*]const T, x: [*]T, incx: BlasInt) bool {
        if (implementation == .portable_scalar) {
            legacyTpmv(T, uplo, trans_, diag, n, ap, x, incx);
            return true;
        }
        if (implementation != .compact_triangular_packed_finite) return false;
        return tryFiniteTpmv(T, allocator, max_bytes, uplo, trans_, diag, n, ap, x, incx);
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

// Experimental ordered packed rows. No caller output is changed on refusal.
noinline fn tryFiniteTpmv(comptime T: type, allocator: std.mem.Allocator, max_bytes: usize, uplo: Uplo, trans_: Order, diag: Diag, n_: BlasInt, ap: [*]const T, x: [*]T, incx: BlasInt) bool {
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
    const Bits = std.meta.Int(.unsigned, @bitSizeOf(T));
    const exponent: Bits = if (T == f32) 0x7f800000 else 0x7ff0000000000000;
    for (0..n) |i| {
        const value = x[if (incx > 0) i * stride else last - i * stride];
        if ((@as(Bits, @bitCast(value)) & exponent) == exponent) return false;
    }
    const output = allocator.alloc(T, n) catch return false;
    defer allocator.free(output);
    const upper = (trans_ == .no_trans and uplo == .upper) or (trans_ != .no_trans and uplo == .lower);
    const paired_end = if (trans_ == .no_trans) n - n % 2 else 0;
    if (paired_end != 0 and !finiteTpmvNoTransPairs(T, uplo, diag, n, ap, x, incx, stride, last, output)) return false;
    for (paired_end..n) |iteration| {
        const i = if (trans_ == .no_trans or upper) iteration else n - iteration - 1;
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

// Adjacent output rows share X loads and packed-column address arithmetic.
// They remain separate ordered sums, including each initial +0 addition. Only
// private output is written here; the caller commits after every row succeeds.
noinline fn finiteTpmvNoTransPairs(comptime T: type, uplo: Uplo, diag: Diag, n: usize, ap: [*]const T, x: [*]const T, incx: BlasInt, stride: usize, last: usize, output: []T) bool {
    @setFloatMode(.strict);
    const Bits = std.meta.Int(.unsigned, @bitSizeOf(T));
    const exponent: Bits = if (T == f32) 0x7f800000 else 0x7ff0000000000000;
    var i: usize = 0;
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
