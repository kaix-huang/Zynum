// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later

const std = @import("std");

const scalar = @import("../shared/scalar.zig");
const indexing = @import("../shared/indexing.zig");
const core_pool = @import("../execution/thread_pool.zig");
const matrix_vector_ops = @import("../matrix_vector.zig");
const planner = @import("planner.zig");

pub const BlasInt = scalar.BlasInt;
pub const ComplexF32 = scalar.ComplexF32;
pub const ComplexF64 = scalar.ComplexF64;
pub const Order = scalar.Order;

const zero = scalar.zero;
const add = scalar.add;
const mul = scalar.mul;
const isZero = scalar.isZero;
const isOne = scalar.isOne;

const toUsize = indexing.toUsize;
const matIndex = indexing.matIndex;
const matrixValue = matrix_vector_ops.matrixValue;

const F32x4 = @Vector(4, f32);
const F64x2 = @Vector(2, f64);

const C32x4 = struct {
    re: F32x4,
    im: F32x4,
};

const C64x2 = struct {
    re: F64x2,
    im: F64x2,
};

const max_cached_complex_workspace_bytes = 64 * 1024 * 1024;

threadlocal var complex_f32_workspace_ptr: ?[*]f32 = null;
threadlocal var complex_f32_workspace_len: usize = 0;
threadlocal var complex_f64_workspace_ptr: ?[*]f64 = null;
threadlocal var complex_f64_workspace_len: usize = 0;

fn ComplexWorkspace(comptime T: type) type {
    return struct {
        data: []T,
        cached: bool,

        fn deinit(self: @This()) void {
            if (!self.cached) std.heap.c_allocator.free(self.data);
        }
    };
}

fn acquireComplexWorkspace(comptime T: type, len: usize) ?ComplexWorkspace(T) {
    if (len * @sizeOf(T) > max_cached_complex_workspace_bytes) {
        const data = std.heap.c_allocator.alloc(T, len) catch return null;
        return .{ .data = data, .cached = false };
    }

    if (T == f32) {
        if (complex_f32_workspace_len < len) {
            const data = std.heap.c_allocator.alloc(f32, len) catch return null;
            if (complex_f32_workspace_ptr) |old| std.heap.c_allocator.free(old[0..complex_f32_workspace_len]);
            complex_f32_workspace_ptr = data.ptr;
            complex_f32_workspace_len = len;
        }
        return .{ .data = complex_f32_workspace_ptr.?[0..len], .cached = true };
    }
    if (T == f64) {
        if (complex_f64_workspace_len < len) {
            const data = std.heap.c_allocator.alloc(f64, len) catch return null;
            if (complex_f64_workspace_ptr) |old| std.heap.c_allocator.free(old[0..complex_f64_workspace_len]);
            complex_f64_workspace_ptr = data.ptr;
            complex_f64_workspace_len = len;
        }
        return .{ .data = complex_f64_workspace_ptr.?[0..len], .cached = true };
    }
    @compileError("complex GEMM workspace supports f32 and f64");
}

pub fn freeCurrentThreadCaches() void {
    if (complex_f32_workspace_ptr) |ptr| std.heap.c_allocator.free(ptr[0..complex_f32_workspace_len]);
    complex_f32_workspace_ptr = null;
    complex_f32_workspace_len = 0;
    if (complex_f64_workspace_ptr) |ptr| std.heap.c_allocator.free(ptr[0..complex_f64_workspace_len]);
    complex_f64_workspace_ptr = null;
    complex_f64_workspace_len = 0;
}

inline fn takeWorkspace(comptime T: type, workspace: []T, offset: *usize, len: usize) []T {
    const start = offset.*;
    offset.* = start + len;
    return workspace[start..offset.*];
}

inline fn loadF32x4(src: []const f32, index: usize) F32x4 {
    return @as(*align(1) const F32x4, @ptrCast(&src[index])).*;
}

inline fn storeF32x4(dst: []f32, index: usize, value: F32x4) void {
    @as(*align(1) F32x4, @ptrCast(&dst[index])).* = value;
}

inline fn storeF64x2(dst: []f64, index: usize, value: F64x2) void {
    @as(*align(1) F64x2, @ptrCast(&dst[index])).* = value;
}

inline fn loadF64x2(src: []const f64, index: usize) F64x2 {
    return @as(*align(1) const F64x2, @ptrCast(&src[index])).*;
}

noinline fn gemmNoTransRealShortWideTailN2048(
    comptime T: type,
    m_: BlasInt,
    aligned_rows: BlasInt,
    k_: BlasInt,
    alpha: T,
    a: [*]const T,
    lda: BlasInt,
    b: [*]const T,
    ldb: BlasInt,
    beta: T,
    c: [*]T,
    ldc: BlasInt,
) void {
    planner.noTransReal(T, aligned_rows, 2048, k_, alpha, a, lda, b, ldb, beta, c, ldc);

    var x_tail: [1024]T = undefined;
    var y_tail: [2048]T = undefined;
    const m = toUsize(m_);
    const aligned = toUsize(aligned_rows);
    const k = toUsize(k_);
    for (aligned..m) |row| {
        for (0..k) |p| x_tail[p] = a[matIndex(lda, row, p)];
        for (0..y_tail.len) |j| y_tail[j] = c[matIndex(ldc, row, j)];
        matrix_vector_ops.gemv(T, .trans, k_, 2048, alpha, b, ldb, &x_tail, 1, beta, &y_tail, 1);
        for (0..y_tail.len) |j| c[matIndex(ldc, row, j)] = y_tail[j];
    }
}

pub fn gemmNoTransReal(comptime T: type, m_: BlasInt, n_: BlasInt, k_: BlasInt, alpha: T, a: [*]const T, lda: BlasInt, b: [*]const T, ldb: BlasInt, beta: T, c: [*]T, ldc: BlasInt) void {
    if (m_ <= 0 or n_ <= 0) return;
    if (k_ <= 0 or isZero(T, alpha)) {
        const m = toUsize(m_);
        const n = toUsize(n_);
        for (0..n) |j| {
            for (0..m) |i| {
                const idxc = matIndex(ldc, i, j);
                c[idxc] = if (isZero(T, beta)) zero(T) else mul(T, beta, c[idxc]);
            }
        }
        return;
    }
    if (m_ == 17 and n_ == 2048 and k_ >= 128 and k_ <= 1024) {
        gemmNoTransRealShortWideTailN2048(T, m_, 16, k_, alpha, a, lda, b, ldb, beta, c, ldc);
        return;
    }
    if (m_ == 34 and n_ == 2048 and k_ >= 256 and k_ <= 1024) {
        gemmNoTransRealShortWideTailN2048(T, m_, 32, k_, alpha, a, lda, b, ldb, beta, c, ldc);
        return;
    }
    if (n_ == 1 and k_ > 0) {
        matrix_vector_ops.gemv(T, .no_trans, m_, k_, alpha, a, lda, b, 1, beta, c, 1);
        return;
    }
    if (T == f32 and m_ == 1 and lda == 1 and ldc == 1 and k_ >= 128) {
        matrix_vector_ops.gemv(T, .trans, k_, n_, alpha, b, ldb, a, 1, beta, c, 1);
        return;
    }
    if (T == f64 and m_ == 1 and lda == 1 and ldc == 1 and k_ >= 128) {
        matrix_vector_ops.gemv(T, .trans, k_, n_, alpha, b, ldb, a, 1, beta, c, 1);
        return;
    }
    if (T == f32 and n_ == 17 and m_ >= 1024 and k_ >= 128) {
        planner.noTransReal(T, m_, 16, k_, alpha, a, lda, b, ldb, beta, c, ldc);
        matrix_vector_ops.gemv(T, .no_trans, m_, k_, alpha, a, lda, b + matIndex(ldb, 0, 16), 1, beta, c + matIndex(ldc, 0, 16), 1);
        return;
    }
    if (T == f32 and n_ > 1 and n_ <= 17 and n_ != 16 and m_ >= 1024 and k_ >= 128) {
        const n = toUsize(n_);
        for (0..n) |j| {
            matrix_vector_ops.gemv(T, .no_trans, m_, k_, alpha, a, lda, b + matIndex(ldb, 0, j), 1, beta, c + matIndex(ldc, 0, j), 1);
        }
        return;
    }
    if (T == f64 and n_ == 17 and m_ >= 1024 and k_ >= 128) {
        planner.noTransReal(T, m_, 16, k_, alpha, a, lda, b, ldb, beta, c, ldc);
        matrix_vector_ops.gemv(T, .no_trans, m_, k_, alpha, a, lda, b + matIndex(ldb, 0, 16), 1, beta, c + matIndex(ldc, 0, 16), 1);
        return;
    }
    if (T == f64 and n_ > 1 and n_ <= 17 and n_ != 16 and m_ >= 1024 and k_ >= 128) {
        const n = toUsize(n_);
        for (0..n) |j| {
            matrix_vector_ops.gemv(T, .no_trans, m_, k_, alpha, a, lda, b + matIndex(ldb, 0, j), 1, beta, c + matIndex(ldc, 0, j), 1);
        }
        return;
    }
    planner.noTransReal(T, m_, n_, k_, alpha, a, lda, b, ldb, beta, c, ldc);
}

fn tryGemmNoTransTransposedBReal(comptime T: type, m_: BlasInt, n_: BlasInt, k_: BlasInt, alpha: T, a: [*]const T, lda: BlasInt, b: [*]const T, ldb: BlasInt, beta: T, c: [*]T, ldc: BlasInt) bool {
    if (k_ >= 128 and !isZero(T, alpha)) {
        if (n_ == 1 and m_ >= 128) {
            matrix_vector_ops.gemv(T, .no_trans, m_, k_, alpha, a, lda, b, ldb, beta, c, 1);
            return true;
        }
        if (m_ == 1 and n_ >= 128) {
            matrix_vector_ops.gemv(T, .no_trans, n_, k_, alpha, b, ldb, a, lda, beta, c, ldc);
            return true;
        }
    }
    return planner.noTransTransposedBReal(T, m_, n_, k_, alpha, a, lda, b, ldb, beta, c, ldc);
}

fn tryGemmTransposedAReal(comptime T: type, transb: Order, m_: BlasInt, n_: BlasInt, k_: BlasInt, alpha: T, a: [*]const T, lda: BlasInt, b: [*]const T, ldb: BlasInt, beta: T, c: [*]T, ldc: BlasInt) bool {
    if (transb == .trans) {
        return planner.transposedAReal(T, m_, n_, k_, alpha, a, lda, b, ldb, .trans, beta, c, ldc);
    }
    return planner.transposedAReal(T, m_, n_, k_, alpha, a, lda, b, ldb, .no_trans, beta, c, ldc);
}

inline fn loadC32x4(ptr: [*]const ComplexF32, ld: BlasInt, row: usize, col: usize) C32x4 {
    return .{
        .re = .{
            ptr[matIndex(ld, row + 0, col)].re,
            ptr[matIndex(ld, row + 1, col)].re,
            ptr[matIndex(ld, row + 2, col)].re,
            ptr[matIndex(ld, row + 3, col)].re,
        },
        .im = .{
            ptr[matIndex(ld, row + 0, col)].im,
            ptr[matIndex(ld, row + 1, col)].im,
            ptr[matIndex(ld, row + 2, col)].im,
            ptr[matIndex(ld, row + 3, col)].im,
        },
    };
}

inline fn transposeF32x4(v0: F32x4, v1: F32x4, v2: F32x4, v3: F32x4) [4]F32x4 {
    const t0 = @shuffle(f32, v0, v1, @Vector(4, i32){ 0, ~@as(i32, 0), 2, ~@as(i32, 2) });
    const t1 = @shuffle(f32, v0, v1, @Vector(4, i32){ 1, ~@as(i32, 1), 3, ~@as(i32, 3) });
    const t2 = @shuffle(f32, v2, v3, @Vector(4, i32){ 0, ~@as(i32, 0), 2, ~@as(i32, 2) });
    const t3 = @shuffle(f32, v2, v3, @Vector(4, i32){ 1, ~@as(i32, 1), 3, ~@as(i32, 3) });
    return .{
        @shuffle(f32, t0, t2, @Vector(4, i32){ 0, 1, ~@as(i32, 0), ~@as(i32, 1) }),
        @shuffle(f32, t1, t3, @Vector(4, i32){ 0, 1, ~@as(i32, 0), ~@as(i32, 1) }),
        @shuffle(f32, t0, t2, @Vector(4, i32){ 2, 3, ~@as(i32, 2), ~@as(i32, 3) }),
        @shuffle(f32, t1, t3, @Vector(4, i32){ 2, 3, ~@as(i32, 2), ~@as(i32, 3) }),
    };
}

inline fn storeC32x4(ptr: [*]ComplexF32, ld: BlasInt, row: usize, col: usize, value: C32x4) void {
    inline for (0..4) |r| {
        ptr[matIndex(ld, row + r, col)] = .{ .re = value.re[r], .im = value.im[r] };
    }
}

inline fn finishC32x4(alpha: ComplexF32, beta: ComplexF32, c: [*]const ComplexF32, ldc: BlasInt, row: usize, col: usize, acc: C32x4) C32x4 {
    var out = acc;
    if (!isOne(ComplexF32, alpha)) {
        out = .{
            .re = acc.re * @as(F32x4, @splat(alpha.re)) - acc.im * @as(F32x4, @splat(alpha.im)),
            .im = acc.re * @as(F32x4, @splat(alpha.im)) + acc.im * @as(F32x4, @splat(alpha.re)),
        };
    }
    if (!isZero(ComplexF32, beta)) {
        const old = loadC32x4(c, ldc, row, col);
        out.re += old.re * @as(F32x4, @splat(beta.re)) - old.im * @as(F32x4, @splat(beta.im));
        out.im += old.re * @as(F32x4, @splat(beta.im)) + old.im * @as(F32x4, @splat(beta.re));
    }
    return out;
}

fn kernelC32x4xN(comptime tile_n: usize, k: usize, alpha: ComplexF32, a: [*]const ComplexF32, lda: BlasInt, b: [*]const ComplexF32, ldb: BlasInt, beta: ComplexF32, c: [*]ComplexF32, ldc: BlasInt, i: usize, j: usize) void {
    const zero_v: F32x4 = @splat(0);
    var acc: [tile_n]C32x4 = [_]C32x4{.{ .re = zero_v, .im = zero_v }} ** tile_n;

    var p: usize = 0;
    while (p < k) : (p += 1) {
        const av = loadC32x4(a, lda, i, p);
        inline for (0..tile_n) |col| {
            const bv = b[matIndex(ldb, p, j + col)];
            acc[col].re = @mulAdd(F32x4, av.re, @as(F32x4, @splat(bv.re)), acc[col].re);
            acc[col].re = @mulAdd(F32x4, -av.im, @as(F32x4, @splat(bv.im)), acc[col].re);
            acc[col].im = @mulAdd(F32x4, av.re, @as(F32x4, @splat(bv.im)), acc[col].im);
            acc[col].im = @mulAdd(F32x4, av.im, @as(F32x4, @splat(bv.re)), acc[col].im);
        }
    }

    inline for (0..tile_n) |col| {
        const out = if (isOne(ComplexF32, alpha) and isZero(ComplexF32, beta)) acc[col] else finishC32x4(alpha, beta, c, ldc, i, j + col, acc[col]);
        storeC32x4(c, ldc, i, j + col, out);
    }
}

fn tailRowsC32(comptime tile_n: usize, m: usize, k: usize, alpha: ComplexF32, a: [*]const ComplexF32, lda: BlasInt, b: [*]const ComplexF32, ldb: BlasInt, beta: ComplexF32, c: [*]ComplexF32, ldc: BlasInt, row_start: usize, j: usize) void {
    var tail_i = row_start;
    while (tail_i < m) : (tail_i += 1) {
        inline for (0..tile_n) |col| {
            var sum = zero(ComplexF32);
            for (0..k) |pp| {
                sum = add(ComplexF32, sum, mul(ComplexF32, a[matIndex(lda, tail_i, pp)], b[matIndex(ldb, pp, j + col)]));
            }
            const idxc = matIndex(ldc, tail_i, j + col);
            c[idxc] = add(ComplexF32, mul(ComplexF32, alpha, sum), if (isZero(ComplexF32, beta)) zero(ComplexF32) else mul(ComplexF32, beta, c[idxc]));
        }
    }
}

fn gemmNoTransComplexF32(m_: BlasInt, n_: BlasInt, k_: BlasInt, alpha: ComplexF32, a: [*]const ComplexF32, lda: BlasInt, b: [*]const ComplexF32, ldb: BlasInt, beta: ComplexF32, c: [*]ComplexF32, ldc: BlasInt) void {
    if (n_ == 1 and k_ > 0) {
        matrix_vector_ops.gemv(ComplexF32, .no_trans, m_, k_, alpha, a, lda, b, 1, beta, c, 1);
        return;
    }
    if (m_ == 1 and lda == 1 and ldc == 1 and k_ >= 128) {
        matrix_vector_ops.gemv(ComplexF32, .trans, k_, n_, alpha, b, ldb, a, 1, beta, c, 1);
        return;
    }
    const m = toUsize(m_);
    const n = toUsize(n_);
    const k = toUsize(k_);
    var j: usize = 0;
    while (j + 4 <= n) : (j += 4) {
        var i: usize = 0;
        while (i + 4 <= m) : (i += 4) kernelC32x4xN(4, k, alpha, a, lda, b, ldb, beta, c, ldc, i, j);
        if (i < m) tailRowsC32(4, m, k, alpha, a, lda, b, ldb, beta, c, ldc, i, j);
    }
    if (j < n) {
        switch (n - j) {
            1 => {
                var i: usize = 0;
                while (i + 4 <= m) : (i += 4) kernelC32x4xN(1, k, alpha, a, lda, b, ldb, beta, c, ldc, i, j);
                if (i < m) tailRowsC32(1, m, k, alpha, a, lda, b, ldb, beta, c, ldc, i, j);
            },
            2 => {
                var i: usize = 0;
                while (i + 4 <= m) : (i += 4) kernelC32x4xN(2, k, alpha, a, lda, b, ldb, beta, c, ldc, i, j);
                if (i < m) tailRowsC32(2, m, k, alpha, a, lda, b, ldb, beta, c, ldc, i, j);
            },
            3 => {
                var i: usize = 0;
                while (i + 4 <= m) : (i += 4) kernelC32x4xN(3, k, alpha, a, lda, b, ldb, beta, c, ldc, i, j);
                if (i < m) tailRowsC32(3, m, k, alpha, a, lda, b, ldb, beta, c, ldc, i, j);
            },
            else => unreachable,
        }
    }
}

inline fn useExpandedComplexF32Real(m: usize, n: usize, k: usize) bool {
    // Default square CGEMM up to 384 is faster on the compact 3M path: it
    // avoids expanded-real's larger materialization/scatter cost.
    if ((m > 1 and m <= 64 and n >= 1024 and k >= 256 and k <= 1024)) return true;
    if (m == n and m == 128 and k >= 2048) return true;
    // Expanded-real maps one CGEMM to one larger SGEMM.  It avoids the
    // 3M path's three real GEMM planner calls and three B-pack passes, but
    // pays for bigger A/B/C workspaces and a final AoS scatter.  Keep the
    // extra gate under roughly the M5 P-core L2 and only for shapes that
    // beat 3M in complete sweeps.
    const min_work = 128 * 1024;
    if (m *| n *| k < min_work) return false;
    const expanded_real_elems = 4 *| m *| k + 2 *| k *| n + 2 *| m *| n;
    if (expanded_real_elems > 16 * 1024 * 1024 / @sizeOf(f32)) return false;
    if (m == n and m >= 512 and k <= 64) return true;
    if (m != n and m <= 512 and n <= 512 and k >= 512 and (m >= 128 or k <= 1024)) return true;
    return n >= 4 *| m and k < 256;
}

inline fn useExpandedComplexF64Real(m: usize, n: usize, k: usize) bool {
    // Keep expanded-real only for the smallest default square ZGEMM cases;
    // sq128/sq192 are faster on the compact 3M path.
    return (m == n and n == k and m < 128) or (m > 1 and m <= 32 and n >= 2048 and k <= 512);
}

inline fn complex3mRowCompute(m: usize, n: usize, k: usize) usize {
    // The compact 3M path otherwise gives all three real products an odd
    // 127-row stride.  One zero row makes the representative 127x129 family
    // a regular 128-row problem while adding less than one percent work.
    // Keep the experiment exact until its surrounding shape boundaries have
    // been swept; all other 3M calls retain their original layout.
    return if (m == 127 and n == 129 and k >= 31 and k <= 129) 128 else m;
}

noinline fn gemmComplexF32ViaExpandedRealWorkspacePaddedSmall(transa: Order, transb: Order, m_: BlasInt, n_: BlasInt, k_: BlasInt, a: [*]const ComplexF32, lda: BlasInt, b: [*]const ComplexF32, ldb: BlasInt, c: [*]ComplexF32, ldc: BlasInt, workspace: []f32) void {
    const m = toUsize(m_);
    const n = toUsize(n_);
    const k = toUsize(k_);

    // The real planner's regular 128x64 AMX tile is much faster than its
    // 2*m by n fringe path for the small non-NN complex shapes routed here.
    // Keep the logical real/imaginary rows contiguous and initialize only
    // the discarded padding rows/columns.
    const m_valid = 2 * m;
    const m_compute = 128;
    const n_compute = 64;
    const k2 = 2 * k;
    const a2_len = m_compute * k2;
    const b2_len = k2 * n_compute;
    const c2_len = m_compute * n_compute;
    var workspace_offset: usize = 0;
    const a2 = takeWorkspace(f32, workspace, &workspace_offset, a2_len);
    const b2 = takeWorkspace(f32, workspace, &workspace_offset, b2_len);
    const c2 = takeWorkspace(f32, workspace, &workspace_offset, c2_len);
    for (0..k2) |p| @memset(a2[p * m_compute + m_valid .. (p + 1) * m_compute], 0);
    @memset(b2[n * k2 .. n_compute * k2], 0);

    for (0..k) |p| {
        const p_lo = p * m_compute;
        const p_hi = (k + p) * m_compute;
        var i: usize = 0;
        if (transa == .no_trans) {
            while (i + 4 <= m) : (i += 4) {
                const value = loadC32x4(a, lda, i, p);
                storeF32x4(a2, p_lo + i, value.re);
                storeF32x4(a2, p_lo + m + i, value.im);
                storeF32x4(a2, p_hi + i, -value.im);
                storeF32x4(a2, p_hi + m + i, value.re);
            }
        }
        while (i < m) : (i += 1) {
            const value = complexOperandValue(ComplexF32, transa, a, lda, i, p);
            a2[p_lo + i] = value.re;
            a2[p_lo + m + i] = value.im;
            a2[p_hi + i] = -value.im;
            a2[p_hi + m + i] = value.re;
        }
    }
    for (0..n) |j| {
        const j_base = j * k2;
        var p: usize = 0;
        if (transb == .no_trans) {
            while (p + 4 <= k) : (p += 4) {
                const value = loadC32x4(b, ldb, p, j);
                storeF32x4(b2, j_base + p, value.re);
                storeF32x4(b2, j_base + k + p, value.im);
            }
        }
        while (p < k) : (p += 1) {
            const value = complexOperandValue(ComplexF32, transb, b, ldb, p, j);
            b2[j_base + p] = value.re;
            b2[j_base + k + p] = value.im;
        }
    }

    const m_compute_i: BlasInt = @intCast(m_compute);
    const n_compute_i: BlasInt = @intCast(n_compute);
    const k2_i: BlasInt = @intCast(k2);
    const m_compute_ld: BlasInt = @intCast(m_compute);
    const k2_ld: BlasInt = @intCast(k2);
    gemmNoTransReal(f32, m_compute_i, n_compute_i, k2_i, 1, a2.ptr, m_compute_ld, b2.ptr, k2_ld, 0, c2.ptr, m_compute_ld);

    for (0..n) |j| {
        const base = j * m_compute;
        var i: usize = 0;
        while (i + 4 <= m) : (i += 4) {
            storeC32x4(c, ldc, i, j, .{ .re = loadF32x4(c2, base + i), .im = loadF32x4(c2, base + m + i) });
        }
        while (i < m) : (i += 1) {
            c[matIndex(ldc, i, j)] = .{ .re = c2[base + i], .im = c2[base + m + i] };
        }
    }
}

fn gemmComplexF32ViaExpandedRealWorkspace(transa: Order, transb: Order, m_: BlasInt, n_: BlasInt, k_: BlasInt, a: [*]const ComplexF32, lda: BlasInt, b: [*]const ComplexF32, ldb: BlasInt, c: [*]ComplexF32, ldc: BlasInt, workspace: []f32) void {
    const m = toUsize(m_);
    const n = toUsize(n_);
    const k = toUsize(k_);

    const m2 = 2 * m;
    const k2 = 2 * k;
    const a2_len = m2 * k2;
    const b2_len = k2 * n;
    const c2_len = m2 * n;
    var workspace_offset: usize = 0;
    const a2 = takeWorkspace(f32, workspace, &workspace_offset, a2_len);
    const b2 = takeWorkspace(f32, workspace, &workspace_offset, b2_len);
    const c2 = takeWorkspace(f32, workspace, &workspace_offset, c2_len);

    for (0..k) |p| {
        const p_lo = p * m2;
        const p_hi = (k + p) * m2;
        var i: usize = 0;
        if (transa == .no_trans) {
            while (i + 4 <= m) : (i += 4) {
                const value = loadC32x4(a, lda, i, p);
                storeF32x4(a2, p_lo + i, value.re);
                storeF32x4(a2, p_lo + m + i, value.im);
                storeF32x4(a2, p_hi + i, -value.im);
                storeF32x4(a2, p_hi + m + i, value.re);
            }
        }
        while (i < m) : (i += 1) {
            const value = complexOperandValue(ComplexF32, transa, a, lda, i, p);
            a2[p_lo + i] = value.re;
            a2[p_lo + m + i] = value.im;
            a2[p_hi + i] = -value.im;
            a2[p_hi + m + i] = value.re;
        }
    }
    for (0..n) |j| {
        const j_base = j * k2;
        var p: usize = 0;
        if (transb == .no_trans) {
            while (p + 4 <= k) : (p += 4) {
                const value = loadC32x4(b, ldb, p, j);
                storeF32x4(b2, j_base + p, value.re);
                storeF32x4(b2, j_base + k + p, value.im);
            }
        }
        while (p < k) : (p += 1) {
            const value = complexOperandValue(ComplexF32, transb, b, ldb, p, j);
            b2[j_base + p] = value.re;
            b2[j_base + k + p] = value.im;
        }
    }

    const m2_i: BlasInt = @intCast(m2);
    const k2_i: BlasInt = @intCast(k2);
    const m2_ld: BlasInt = @intCast(m2);
    const k2_ld: BlasInt = @intCast(k2);
    gemmNoTransReal(f32, m2_i, n_, k2_i, 1, a2.ptr, m2_ld, b2.ptr, k2_ld, 0, c2.ptr, m2_ld);

    for (0..n) |j| {
        const base = j * m2;
        var i: usize = 0;
        while (i + 4 <= m) : (i += 4) {
            storeC32x4(c, ldc, i, j, .{ .re = loadF32x4(c2, base + i), .im = loadF32x4(c2, base + m + i) });
        }
        while (i < m) : (i += 1) {
            c[matIndex(ldc, i, j)] = .{ .re = c2[base + i], .im = c2[base + m + i] };
        }
    }
}

fn tryGemmComplexF32ViaExpandedReal(transa: Order, transb: Order, m_: BlasInt, n_: BlasInt, k_: BlasInt, alpha: ComplexF32, a: [*]const ComplexF32, lda: BlasInt, b: [*]const ComplexF32, ldb: BlasInt, beta: ComplexF32, c: [*]ComplexF32, ldc: BlasInt) bool {
    if (!isOne(ComplexF32, alpha) or !isZero(ComplexF32, beta)) return false;
    const m = toUsize(m_);
    const n = toUsize(n_);
    const k = toUsize(k_);
    if (transa == .no_trans and transb == .no_trans) {
        if (!useExpandedComplexF32Real(m, n, k)) return false;
    } else {
        const work = m *| n *| k;
        if (!(m <= 64 and n <= 64 and k >= 128 and k <= 256 and work >= 128 * 1024)) return false;
    }

    const pad_small_non_nn = (transa != .no_trans or transb != .no_trans) and m <= 64 and n <= 64 and (m < 64 or n < 64);
    const m2 = if (pad_small_non_nn) @as(usize, 128) else 2 * m;
    const n_compute = if (pad_small_non_nn) @as(usize, 64) else n;
    const workspace_len = m2 * (2 * k) + (2 * k) * n_compute + m2 * n_compute;
    const workspace = acquireComplexWorkspace(f32, workspace_len) orelse return false;
    defer workspace.deinit();
    if (pad_small_non_nn) {
        gemmComplexF32ViaExpandedRealWorkspacePaddedSmall(transa, transb, m_, n_, k_, a, lda, b, ldb, c, ldc, workspace.data);
    } else {
        gemmComplexF32ViaExpandedRealWorkspace(transa, transb, m_, n_, k_, a, lda, b, ldb, c, ldc, workspace.data);
    }
    return true;
}

inline fn complexOperandValue(comptime T: type, trans: Order, matrix: [*]const T, ld: BlasInt, row: usize, col: usize) T {
    const value = switch (trans) {
        .no_trans => matrix[matIndex(ld, row, col)],
        .trans, .conj_trans => matrix[matIndex(ld, col, row)],
    };
    return if (trans == .conj_trans) .{ .re = value.re, .im = -value.im } else value;
}

inline fn materializeComplexF32TransposedB4x4(transb: Order, n: usize, k: usize, b: [*]const ComplexF32, ldb: BlasInt, br: []f32, bi: []f32, bp: []f32) void {
    var j: usize = 0;
    while (j + 4 <= n) : (j += 4) {
        var p: usize = 0;
        while (p + 4 <= k) : (p += 4) {
            const v0 = loadC32x4(b, ldb, j, p + 0);
            const v1 = loadC32x4(b, ldb, j, p + 1);
            const v2 = loadC32x4(b, ldb, j, p + 2);
            const v3 = loadC32x4(b, ldb, j, p + 3);
            const re = transposeF32x4(v0.re, v1.re, v2.re, v3.re);
            var im = transposeF32x4(v0.im, v1.im, v2.im, v3.im);
            if (transb == .conj_trans) {
                inline for (0..4) |lane| im[lane] = -im[lane];
            }
            inline for (0..4) |lane| {
                const idx = p + (j + lane) * k;
                storeF32x4(br, idx, re[lane]);
                storeF32x4(bi, idx, im[lane]);
                storeF32x4(bp, idx, re[lane] + im[lane]);
            }
        }
        while (p < k) : (p += 1) {
            inline for (0..4) |lane| {
                const jj = j + lane;
                const value = complexOperandValue(ComplexF32, transb, b, ldb, p, jj);
                const idx = p + jj * k;
                br[idx] = value.re;
                bi[idx] = value.im;
                bp[idx] = value.re + value.im;
            }
        }
    }
    while (j < n) : (j += 1) {
        for (0..k) |p| {
            const value = complexOperandValue(ComplexF32, transb, b, ldb, p, j);
            const idx = p + j * k;
            br[idx] = value.re;
            bi[idx] = value.im;
            bp[idx] = value.re + value.im;
        }
    }
}

fn gemmComplexF32ViaRealBuffers(transa: Order, transb: Order, m_: BlasInt, n_: BlasInt, k_: BlasInt, a: [*]const ComplexF32, lda: BlasInt, b: [*]const ComplexF32, ldb: BlasInt, c: [*]ComplexF32, ldc: BlasInt, m_compute: usize, ar: []f32, ai: []f32, am: []f32, br: []f32, bi: []f32, bp: []f32, cr: []f32, ci: []f32, tmp: []f32) void {
    const m = toUsize(m_);
    const n = toUsize(n_);
    const k = toUsize(k_);

    if (m_compute > m) {
        for (0..k) |p| {
            const pad_start = p * m_compute + m;
            const pad_end = (p + 1) * m_compute;
            @memset(ar[pad_start..pad_end], 0);
            @memset(ai[pad_start..pad_end], 0);
            @memset(am[pad_start..pad_end], 0);
        }
    }

    if (transa != .no_trans and (m_compute > m or m >= 256 or k >= 512)) {
        var p: usize = 0;
        while (p + 4 <= k) : (p += 4) {
            var i: usize = 0;
            while (i + 4 <= m) : (i += 4) {
                const v0 = loadC32x4(a, lda, p, i + 0);
                const v1 = loadC32x4(a, lda, p, i + 1);
                const v2 = loadC32x4(a, lda, p, i + 2);
                const v3 = loadC32x4(a, lda, p, i + 3);
                const re = transposeF32x4(v0.re, v1.re, v2.re, v3.re);
                var im = transposeF32x4(v0.im, v1.im, v2.im, v3.im);
                if (transa == .conj_trans) {
                    inline for (0..4) |lane| im[lane] = -im[lane];
                }
                inline for (0..4) |lane| {
                    const idx = i + (p + lane) * m_compute;
                    storeF32x4(ar, idx, re[lane]);
                    storeF32x4(ai, idx, re[lane] + im[lane]);
                    storeF32x4(am, idx, im[lane] - re[lane]);
                }
            }
            while (i < m) : (i += 1) {
                inline for (0..4) |lane| {
                    const pp = p + lane;
                    const value = complexOperandValue(ComplexF32, transa, a, lda, i, pp);
                    const idx = i + pp * m_compute;
                    ar[idx] = value.re;
                    ai[idx] = value.re + value.im;
                    am[idx] = value.im - value.re;
                }
            }
        }
        while (p < k) : (p += 1) {
            for (0..m) |i| {
                const value = complexOperandValue(ComplexF32, transa, a, lda, i, p);
                const idx = i + p * m_compute;
                ar[idx] = value.re;
                ai[idx] = value.re + value.im;
                am[idx] = value.im - value.re;
            }
        }
    } else {
        for (0..k) |p| {
            var i: usize = 0;
            if (transa == .no_trans) {
                while (i + 4 <= m) : (i += 4) {
                    const value = loadC32x4(a, lda, i, p);
                    const idx = i + p * m_compute;
                    storeF32x4(ar, idx, value.re);
                    storeF32x4(ai, idx, value.re + value.im);
                    storeF32x4(am, idx, value.im - value.re);
                }
            }
            while (i < m) : (i += 1) {
                const value = complexOperandValue(ComplexF32, transa, a, lda, i, p);
                const idx = i + p * m_compute;
                ar[idx] = value.re;
                ai[idx] = value.re + value.im;
                am[idx] = value.im - value.re;
            }
        }
    }
    if (transb != .no_trans and (m_compute > m or n >= 256 or k >= 512)) {
        materializeComplexF32TransposedB4x4(transb, n, k, b, ldb, br, bi, bp);
    } else {
        for (0..n) |j| {
            var p: usize = 0;
            if (transb == .no_trans) {
                while (p + 4 <= k) : (p += 4) {
                    const value = loadC32x4(b, ldb, p, j);
                    const idx = p + j * k;
                    storeF32x4(br, idx, value.re);
                    storeF32x4(bi, idx, value.im);
                    storeF32x4(bp, idx, value.re + value.im);
                }
            }
            while (p < k) : (p += 1) {
                const value = complexOperandValue(ComplexF32, transb, b, ldb, p, j);
                const idx = p + j * k;
                br[idx] = value.re;
                bi[idx] = value.im;
                bp[idx] = value.re + value.im;
            }
        }
    }

    const m_compute_i: BlasInt = @intCast(m_compute);
    const lda_r: BlasInt = @intCast(m_compute);
    const ldb_r: BlasInt = @intCast(k);
    const ldc_r: BlasInt = @intCast(m_compute);
    gemmNoTransReal(f32, m_compute_i, n_, k_, 1, ar.ptr, lda_r, bp.ptr, ldb_r, 0, cr.ptr, ldc_r);
    gemmNoTransReal(f32, m_compute_i, n_, k_, 1, ai.ptr, lda_r, bi.ptr, ldb_r, 0, tmp.ptr, ldc_r);
    gemmNoTransReal(f32, m_compute_i, n_, k_, 1, am.ptr, lda_r, br.ptr, ldb_r, 0, ci.ptr, ldc_r);

    for (0..n) |j| {
        var i: usize = 0;
        while (i + 4 <= m) : (i += 4) {
            const src = i + j * m_compute;
            const crv = loadF32x4(cr, src);
            storeC32x4(c, ldc, i, j, .{ .re = crv - loadF32x4(tmp, src), .im = crv + loadF32x4(ci, src) });
        }
        while (i < m) : (i += 1) {
            const src = i + j * m_compute;
            c[matIndex(ldc, i, j)] = .{ .re = cr[src] - tmp[src], .im = cr[src] + ci[src] };
        }
    }
}

fn tryGemmComplexF32ViaReal(transa: Order, transb: Order, m_: BlasInt, n_: BlasInt, k_: BlasInt, alpha: ComplexF32, a: [*]const ComplexF32, lda: BlasInt, b: [*]const ComplexF32, ldb: BlasInt, beta: ComplexF32, c: [*]ComplexF32, ldc: BlasInt) bool {
    if (!isOne(ComplexF32, alpha) or !isZero(ComplexF32, beta)) return false;
    const m = toUsize(m_);
    const n = toUsize(n_);
    const k = toUsize(k_);
    if (m == 1 or n == 1) {
        // Materialization only amortizes on the edge layouts whose real GEMMs
        // become contiguous GEMV calls; the other transpose pairs stay direct.
        const row_edge = m == 1 and transa != .no_trans and transb == .no_trans;
        const column_edge = n == 1 and transa == .no_trans and transb != .no_trans;
        if (!row_edge and !column_edge) return false;
    }
    if (m *| n *| k < 128 * 1024) return false;

    const m_compute = complex3mRowCompute(m, n, k);
    const a_len = m_compute * k;
    const b_len = k * n;
    const c_len = m_compute * n;
    const plane_padding = 64;

    const workspace = acquireComplexWorkspace(f32, 3 * a_len + 3 * b_len + 3 * c_len + 8 * plane_padding) orelse return false;
    defer workspace.deinit();
    var workspace_offset: usize = 0;
    const ar = takeWorkspace(f32, workspace.data, &workspace_offset, a_len);
    workspace_offset += plane_padding;
    const ai = takeWorkspace(f32, workspace.data, &workspace_offset, a_len);
    workspace_offset += plane_padding;
    const am = takeWorkspace(f32, workspace.data, &workspace_offset, a_len);
    workspace_offset += plane_padding;
    const br = takeWorkspace(f32, workspace.data, &workspace_offset, b_len);
    workspace_offset += plane_padding;
    const bi = takeWorkspace(f32, workspace.data, &workspace_offset, b_len);
    workspace_offset += plane_padding;
    const bp = takeWorkspace(f32, workspace.data, &workspace_offset, b_len);
    workspace_offset += plane_padding;
    const cr = takeWorkspace(f32, workspace.data, &workspace_offset, c_len);
    workspace_offset += plane_padding;
    const ci = takeWorkspace(f32, workspace.data, &workspace_offset, c_len);
    workspace_offset += plane_padding;
    const tmp = takeWorkspace(f32, workspace.data, &workspace_offset, c_len);
    gemmComplexF32ViaRealBuffers(transa, transb, m_, n_, k_, a, lda, b, ldb, c, ldc, m_compute, ar, ai, am, br, bi, bp, cr, ci, tmp);
    return true;
}

inline fn loadC64x2(ptr: [*]const ComplexF64, ld: BlasInt, row: usize, col: usize) C64x2 {
    return .{
        .re = .{
            ptr[matIndex(ld, row + 0, col)].re,
            ptr[matIndex(ld, row + 1, col)].re,
        },
        .im = .{
            ptr[matIndex(ld, row + 0, col)].im,
            ptr[matIndex(ld, row + 1, col)].im,
        },
    };
}

inline fn transposeF64x2(v0: F64x2, v1: F64x2) [2]F64x2 {
    return .{
        @shuffle(f64, v0, v1, @Vector(2, i32){ 0, ~@as(i32, 0) }),
        @shuffle(f64, v0, v1, @Vector(2, i32){ 1, ~@as(i32, 1) }),
    };
}

inline fn storeC64x2(ptr: [*]ComplexF64, ld: BlasInt, row: usize, col: usize, value: C64x2) void {
    inline for (0..2) |r| {
        ptr[matIndex(ld, row + r, col)] = .{ .re = value.re[r], .im = value.im[r] };
    }
}

inline fn finishC64x2(alpha: ComplexF64, beta: ComplexF64, c: [*]const ComplexF64, ldc: BlasInt, row: usize, col: usize, acc: C64x2) C64x2 {
    var out = acc;
    if (!isOne(ComplexF64, alpha)) {
        out = .{
            .re = acc.re * @as(F64x2, @splat(alpha.re)) - acc.im * @as(F64x2, @splat(alpha.im)),
            .im = acc.re * @as(F64x2, @splat(alpha.im)) + acc.im * @as(F64x2, @splat(alpha.re)),
        };
    }
    if (!isZero(ComplexF64, beta)) {
        const old = loadC64x2(c, ldc, row, col);
        out.re += old.re * @as(F64x2, @splat(beta.re)) - old.im * @as(F64x2, @splat(beta.im));
        out.im += old.re * @as(F64x2, @splat(beta.im)) + old.im * @as(F64x2, @splat(beta.re));
    }
    return out;
}

fn kernelC64x2xN(comptime tile_n: usize, k: usize, alpha: ComplexF64, a: [*]const ComplexF64, lda: BlasInt, b: [*]const ComplexF64, ldb: BlasInt, beta: ComplexF64, c: [*]ComplexF64, ldc: BlasInt, i: usize, j: usize) void {
    const zero_v: F64x2 = @splat(0);
    var acc: [tile_n]C64x2 = [_]C64x2{.{ .re = zero_v, .im = zero_v }} ** tile_n;

    var p: usize = 0;
    while (p < k) : (p += 1) {
        const av = loadC64x2(a, lda, i, p);
        inline for (0..tile_n) |col| {
            const bv = b[matIndex(ldb, p, j + col)];
            acc[col].re = @mulAdd(F64x2, av.re, @as(F64x2, @splat(bv.re)), acc[col].re);
            acc[col].re = @mulAdd(F64x2, -av.im, @as(F64x2, @splat(bv.im)), acc[col].re);
            acc[col].im = @mulAdd(F64x2, av.re, @as(F64x2, @splat(bv.im)), acc[col].im);
            acc[col].im = @mulAdd(F64x2, av.im, @as(F64x2, @splat(bv.re)), acc[col].im);
        }
    }

    inline for (0..tile_n) |col| {
        const out = if (isOne(ComplexF64, alpha) and isZero(ComplexF64, beta)) acc[col] else finishC64x2(alpha, beta, c, ldc, i, j + col, acc[col]);
        storeC64x2(c, ldc, i, j + col, out);
    }
}

fn tailRowsC64(comptime tile_n: usize, m: usize, k: usize, alpha: ComplexF64, a: [*]const ComplexF64, lda: BlasInt, b: [*]const ComplexF64, ldb: BlasInt, beta: ComplexF64, c: [*]ComplexF64, ldc: BlasInt, row_start: usize, j: usize) void {
    var tail_i = row_start;
    while (tail_i < m) : (tail_i += 1) {
        inline for (0..tile_n) |col| {
            var sum = zero(ComplexF64);
            for (0..k) |pp| {
                sum = add(ComplexF64, sum, mul(ComplexF64, a[matIndex(lda, tail_i, pp)], b[matIndex(ldb, pp, j + col)]));
            }
            const idxc = matIndex(ldc, tail_i, j + col);
            c[idxc] = add(ComplexF64, mul(ComplexF64, alpha, sum), if (isZero(ComplexF64, beta)) zero(ComplexF64) else mul(ComplexF64, beta, c[idxc]));
        }
    }
}

fn gemmNoTransComplexF64(m_: BlasInt, n_: BlasInt, k_: BlasInt, alpha: ComplexF64, a: [*]const ComplexF64, lda: BlasInt, b: [*]const ComplexF64, ldb: BlasInt, beta: ComplexF64, c: [*]ComplexF64, ldc: BlasInt) void {
    if (n_ == 1 and k_ > 0) {
        matrix_vector_ops.gemv(ComplexF64, .no_trans, m_, k_, alpha, a, lda, b, 1, beta, c, 1);
        return;
    }
    if (m_ == 1 and lda == 1 and ldc == 1 and k_ >= 128) {
        matrix_vector_ops.gemv(ComplexF64, .trans, k_, n_, alpha, b, ldb, a, 1, beta, c, 1);
        return;
    }
    const m = toUsize(m_);
    const n = toUsize(n_);
    const k = toUsize(k_);
    var j: usize = 0;
    while (j + 4 <= n) : (j += 4) {
        var i: usize = 0;
        while (i + 2 <= m) : (i += 2) kernelC64x2xN(4, k, alpha, a, lda, b, ldb, beta, c, ldc, i, j);
        if (i < m) tailRowsC64(4, m, k, alpha, a, lda, b, ldb, beta, c, ldc, i, j);
    }
    if (j < n) {
        switch (n - j) {
            1 => {
                var i: usize = 0;
                while (i + 2 <= m) : (i += 2) kernelC64x2xN(1, k, alpha, a, lda, b, ldb, beta, c, ldc, i, j);
                if (i < m) tailRowsC64(1, m, k, alpha, a, lda, b, ldb, beta, c, ldc, i, j);
            },
            2 => {
                var i: usize = 0;
                while (i + 2 <= m) : (i += 2) kernelC64x2xN(2, k, alpha, a, lda, b, ldb, beta, c, ldc, i, j);
                if (i < m) tailRowsC64(2, m, k, alpha, a, lda, b, ldb, beta, c, ldc, i, j);
            },
            3 => {
                var i: usize = 0;
                while (i + 2 <= m) : (i += 2) kernelC64x2xN(3, k, alpha, a, lda, b, ldb, beta, c, ldc, i, j);
                if (i < m) tailRowsC64(3, m, k, alpha, a, lda, b, ldb, beta, c, ldc, i, j);
            },
            else => unreachable,
        }
    }
}

fn gemmNoTransComplexF64ViaExpandedRealWorkspace(m_: BlasInt, n_: BlasInt, k_: BlasInt, a: [*]const ComplexF64, lda: BlasInt, b: [*]const ComplexF64, ldb: BlasInt, c: [*]ComplexF64, ldc: BlasInt, workspace: []f64) void {
    const m = toUsize(m_);
    const n = toUsize(n_);
    const k = toUsize(k_);

    const m2 = 2 * m;
    const k2 = 2 * k;
    const a2_len = m2 * k2;
    const b2_len = k2 * n;
    const c2_len = m2 * n;
    var workspace_offset: usize = 0;
    const a2 = takeWorkspace(f64, workspace, &workspace_offset, a2_len);
    const b2 = takeWorkspace(f64, workspace, &workspace_offset, b2_len);
    const c2 = takeWorkspace(f64, workspace, &workspace_offset, c2_len);

    for (0..k) |p| {
        const p_lo = p * m2;
        const p_hi = (k + p) * m2;
        var i: usize = 0;
        while (i + 2 <= m) : (i += 2) {
            const value = loadC64x2(a, lda, i, p);
            storeF64x2(a2, p_lo + i, value.re);
            storeF64x2(a2, p_lo + m + i, value.im);
            storeF64x2(a2, p_hi + i, -value.im);
            storeF64x2(a2, p_hi + m + i, value.re);
        }
        while (i < m) : (i += 1) {
            const value = a[matIndex(lda, i, p)];
            a2[p_lo + i] = value.re;
            a2[p_lo + m + i] = value.im;
            a2[p_hi + i] = -value.im;
            a2[p_hi + m + i] = value.re;
        }
    }
    for (0..n) |j| {
        const j_base = j * k2;
        var p: usize = 0;
        while (p + 2 <= k) : (p += 2) {
            const value = loadC64x2(b, ldb, p, j);
            storeF64x2(b2, j_base + p, value.re);
            storeF64x2(b2, j_base + k + p, value.im);
        }
        while (p < k) : (p += 1) {
            const value = b[matIndex(ldb, p, j)];
            b2[j_base + p] = value.re;
            b2[j_base + k + p] = value.im;
        }
    }

    const m2_i: BlasInt = @intCast(m2);
    const k2_i: BlasInt = @intCast(k2);
    const m2_ld: BlasInt = @intCast(m2);
    const k2_ld: BlasInt = @intCast(k2);
    gemmNoTransReal(f64, m2_i, n_, k2_i, 1, a2.ptr, m2_ld, b2.ptr, k2_ld, 0, c2.ptr, m2_ld);

    for (0..n) |j| {
        for (0..m) |i| {
            c[matIndex(ldc, i, j)] = .{ .re = c2[i + j * m2], .im = c2[m + i + j * m2] };
        }
    }
}

fn tryGemmNoTransComplexF64ViaExpandedReal(m_: BlasInt, n_: BlasInt, k_: BlasInt, alpha: ComplexF64, a: [*]const ComplexF64, lda: BlasInt, b: [*]const ComplexF64, ldb: BlasInt, beta: ComplexF64, c: [*]ComplexF64, ldc: BlasInt) bool {
    if (!isOne(ComplexF64, alpha) or !isZero(ComplexF64, beta)) return false;
    const m = toUsize(m_);
    const n = toUsize(n_);
    const k = toUsize(k_);
    if (!useExpandedComplexF64Real(m, n, k)) return false;

    const workspace_len = (2 * m) * (2 * k) + (2 * k) * n + (2 * m) * n;
    const workspace = acquireComplexWorkspace(f64, workspace_len) orelse return false;
    defer workspace.deinit();
    gemmNoTransComplexF64ViaExpandedRealWorkspace(m_, n_, k_, a, lda, b, ldb, c, ldc, workspace.data);
    return true;
}

const ComplexF64RealGemmTask = struct {
    m: BlasInt,
    n: BlasInt,
    k: BlasInt,
    a: [*]const f64,
    lda: BlasInt,
    b: [*]const f64,
    ldb: BlasInt,
    c: [*]f64,
    ldc: BlasInt,
};

fn runComplexF64RealGemmTask(raw_tasks: *const anyopaque, index: usize) void {
    const tasks: [*]const ComplexF64RealGemmTask = @ptrCast(@alignCast(raw_tasks));
    const task = tasks[index];
    gemmNoTransReal(f64, task.m, task.n, task.k, 1, task.a, task.lda, task.b, task.ldb, 0, task.c, task.ldc);
}

fn gemmComplexF64ViaRealBuffers(transa: Order, transb: Order, m_: BlasInt, n_: BlasInt, k_: BlasInt, a: [*]const ComplexF64, lda: BlasInt, b: [*]const ComplexF64, ldb: BlasInt, c: [*]ComplexF64, ldc: BlasInt, m_compute: usize, ar: []f64, ai: []f64, am: []f64, br: []f64, bi: []f64, bp: []f64, cr: []f64, ci: []f64, tmp: []f64) void {
    const m = toUsize(m_);
    const n = toUsize(n_);
    const k = toUsize(k_);

    if (m_compute > m) {
        for (0..k) |p| {
            const pad_start = p * m_compute + m;
            const pad_end = (p + 1) * m_compute;
            @memset(ar[pad_start..pad_end], 0);
            @memset(ai[pad_start..pad_end], 0);
            @memset(am[pad_start..pad_end], 0);
        }
    }

    if (transa == .no_trans) {
        for (0..k) |p| {
            var i: usize = 0;
            while (i + 2 <= m) : (i += 2) {
                const value = loadC64x2(a, lda, i, p);
                const idx = i + p * m_compute;
                storeF64x2(ar, idx, value.re);
                storeF64x2(ai, idx, value.re + value.im);
                storeF64x2(am, idx, value.im - value.re);
            }
            while (i < m) : (i += 1) {
                const value = a[matIndex(lda, i, p)];
                const idx = i + p * m_compute;
                ar[idx] = value.re;
                ai[idx] = value.re + value.im;
                am[idx] = value.im - value.re;
            }
        }
    } else {
        var p: usize = 0;
        while (p + 2 <= k) : (p += 2) {
            var i: usize = 0;
            while (i + 2 <= m) : (i += 2) {
                const v0 = loadC64x2(a, lda, p, i + 0);
                const v1 = loadC64x2(a, lda, p, i + 1);
                const re = transposeF64x2(v0.re, v1.re);
                var im = transposeF64x2(v0.im, v1.im);
                if (transa == .conj_trans) {
                    im[0] = -im[0];
                    im[1] = -im[1];
                }
                inline for (0..2) |lane| {
                    const idx = i + (p + lane) * m_compute;
                    storeF64x2(ar, idx, re[lane]);
                    storeF64x2(ai, idx, re[lane] + im[lane]);
                    storeF64x2(am, idx, im[lane] - re[lane]);
                }
            }
            while (i < m) : (i += 1) {
                inline for (0..2) |lane| {
                    const pp = p + lane;
                    const value = complexOperandValue(ComplexF64, transa, a, lda, i, pp);
                    const idx = i + pp * m_compute;
                    ar[idx] = value.re;
                    ai[idx] = value.re + value.im;
                    am[idx] = value.im - value.re;
                }
            }
        }
        while (p < k) : (p += 1) {
            for (0..m) |i| {
                const value = complexOperandValue(ComplexF64, transa, a, lda, i, p);
                const idx = i + p * m_compute;
                ar[idx] = value.re;
                ai[idx] = value.re + value.im;
                am[idx] = value.im - value.re;
            }
        }
    }
    if (transb == .no_trans) {
        for (0..n) |j| {
            var p: usize = 0;
            while (p + 2 <= k) : (p += 2) {
                const value = loadC64x2(b, ldb, p, j);
                const idx = p + j * k;
                storeF64x2(br, idx, value.re);
                storeF64x2(bi, idx, value.im);
                storeF64x2(bp, idx, value.re + value.im);
            }
            while (p < k) : (p += 1) {
                const value = b[matIndex(ldb, p, j)];
                const idx = p + j * k;
                br[idx] = value.re;
                bi[idx] = value.im;
                bp[idx] = value.re + value.im;
            }
        }
    } else {
        var j: usize = 0;
        while (j + 2 <= n) : (j += 2) {
            var p: usize = 0;
            while (p + 2 <= k) : (p += 2) {
                const v0 = loadC64x2(b, ldb, j, p + 0);
                const v1 = loadC64x2(b, ldb, j, p + 1);
                const re = transposeF64x2(v0.re, v1.re);
                var im = transposeF64x2(v0.im, v1.im);
                if (transb == .conj_trans) {
                    im[0] = -im[0];
                    im[1] = -im[1];
                }
                inline for (0..2) |lane| {
                    const idx = p + (j + lane) * k;
                    storeF64x2(br, idx, re[lane]);
                    storeF64x2(bi, idx, im[lane]);
                    storeF64x2(bp, idx, re[lane] + im[lane]);
                }
            }
            while (p < k) : (p += 1) {
                inline for (0..2) |lane| {
                    const jj = j + lane;
                    const value = complexOperandValue(ComplexF64, transb, b, ldb, p, jj);
                    const idx = p + jj * k;
                    br[idx] = value.re;
                    bi[idx] = value.im;
                    bp[idx] = value.re + value.im;
                }
            }
        }
        while (j < n) : (j += 1) {
            for (0..k) |p| {
                const value = complexOperandValue(ComplexF64, transb, b, ldb, p, j);
                const idx = p + j * k;
                br[idx] = value.re;
                bi[idx] = value.im;
                bp[idx] = value.re + value.im;
            }
        }
    }

    const m_compute_i: BlasInt = @intCast(m_compute);
    const lda_r: BlasInt = @intCast(m_compute);
    const ldb_r: BlasInt = @intCast(k);
    const ldc_r: BlasInt = @intCast(m_compute);
    var parallel_real_products = false;
    if (m == 127 and n == 129 and k == 32) {
        const real_tasks = [_]ComplexF64RealGemmTask{
            .{ .m = m_compute_i, .n = n_, .k = k_, .a = ar.ptr, .lda = lda_r, .b = bp.ptr, .ldb = ldb_r, .c = cr.ptr, .ldc = ldc_r },
            .{ .m = m_compute_i, .n = n_, .k = k_, .a = ai.ptr, .lda = lda_r, .b = bi.ptr, .ldb = ldb_r, .c = tmp.ptr, .ldc = ldc_r },
            .{ .m = m_compute_i, .n = n_, .k = k_, .a = am.ptr, .lda = lda_r, .b = br.ptr, .ldb = ldb_r, .c = ci.ptr, .ldc = ldc_r },
        };
        parallel_real_products = core_pool.runLowLatency(runComplexF64RealGemmTask, @ptrCast(&real_tasks), real_tasks.len);
    }
    if (!parallel_real_products) {
        gemmNoTransReal(f64, m_compute_i, n_, k_, 1, ar.ptr, lda_r, bp.ptr, ldb_r, 0, cr.ptr, ldc_r);
        gemmNoTransReal(f64, m_compute_i, n_, k_, 1, ai.ptr, lda_r, bi.ptr, ldb_r, 0, tmp.ptr, ldc_r);
        gemmNoTransReal(f64, m_compute_i, n_, k_, 1, am.ptr, lda_r, br.ptr, ldb_r, 0, ci.ptr, ldc_r);
    }

    for (0..n) |j| {
        var i: usize = 0;
        while (i + 2 <= m) : (i += 2) {
            const src = i + j * m_compute;
            const crv = loadF64x2(cr, src);
            storeC64x2(c, ldc, i, j, .{ .re = crv - loadF64x2(tmp, src), .im = crv + loadF64x2(ci, src) });
        }
        while (i < m) : (i += 1) {
            const src = i + j * m_compute;
            c[matIndex(ldc, i, j)] = .{ .re = cr[src] - tmp[src], .im = cr[src] + ci[src] };
        }
    }
}

fn tryGemmComplexF64ViaReal(transa: Order, transb: Order, m_: BlasInt, n_: BlasInt, k_: BlasInt, alpha: ComplexF64, a: [*]const ComplexF64, lda: BlasInt, b: [*]const ComplexF64, ldb: BlasInt, beta: ComplexF64, c: [*]ComplexF64, ldc: BlasInt) bool {
    if (!isOne(ComplexF64, alpha) or !isZero(ComplexF64, beta)) return false;
    const m = toUsize(m_);
    const n = toUsize(n_);
    const k = toUsize(k_);
    // f64 only amortizes 3M materialization on the contiguous column-edge form.
    if ((m == 1 or n == 1) and !(n == 1 and transa == .no_trans and transb != .no_trans)) return false;
    if (m *| n *| k < 128 * 1024) return false;

    const m_compute = complex3mRowCompute(m, n, k);
    const a_len = m_compute * k;
    const b_len = k * n;
    const c_len = m_compute * n;

    const workspace = acquireComplexWorkspace(f64, 3 * a_len + 3 * b_len + 3 * c_len) orelse return false;
    defer workspace.deinit();
    var workspace_offset: usize = 0;
    const ar = takeWorkspace(f64, workspace.data, &workspace_offset, a_len);
    const ai = takeWorkspace(f64, workspace.data, &workspace_offset, a_len);
    const am = takeWorkspace(f64, workspace.data, &workspace_offset, a_len);
    const br = takeWorkspace(f64, workspace.data, &workspace_offset, b_len);
    const bi = takeWorkspace(f64, workspace.data, &workspace_offset, b_len);
    const bp = takeWorkspace(f64, workspace.data, &workspace_offset, b_len);
    const cr = takeWorkspace(f64, workspace.data, &workspace_offset, c_len);
    const ci = takeWorkspace(f64, workspace.data, &workspace_offset, c_len);
    const tmp = takeWorkspace(f64, workspace.data, &workspace_offset, c_len);
    gemmComplexF64ViaRealBuffers(transa, transb, m_, n_, k_, a, lda, b, ldb, c, ldc, m_compute, ar, ai, am, br, bi, bp, cr, ci, tmp);
    return true;
}

pub fn gemm(comptime T: type, transa: Order, transb: Order, m_: BlasInt, n_: BlasInt, k_: BlasInt, alpha: T, a: [*]const T, lda: BlasInt, b: [*]const T, ldb: BlasInt, beta: T, c: [*]T, ldc: BlasInt) void {
    if (m_ <= 0 or n_ <= 0) return;
    if ((T == f32 or T == f64) and transa == .no_trans and transb == .no_trans) {
        gemmNoTransReal(T, m_, n_, k_, alpha, a, lda, b, ldb, beta, c, ldc);
        return;
    }
    if ((T == f32 or T == f64) and transa == .no_trans and transb == .trans) {
        if (tryGemmNoTransTransposedBReal(T, m_, n_, k_, alpha, a, lda, b, ldb, beta, c, ldc)) return;
    }
    if ((T == f32 or T == f64) and transa == .trans and (transb == .no_trans or transb == .trans)) {
        if (tryGemmTransposedAReal(T, transb, m_, n_, k_, alpha, a, lda, b, ldb, beta, c, ldc)) return;
    }
    const m = toUsize(m_);
    const n = toUsize(n_);
    const k = toUsize(k_);
    if (k == 0 or isZero(T, alpha)) {
        for (0..n) |j| {
            for (0..m) |i| {
                const idxc = matIndex(ldc, i, j);
                c[idxc] = if (isZero(T, beta)) zero(T) else mul(T, beta, c[idxc]);
            }
        }
        return;
    }
    if (transa == .no_trans and transb == .no_trans) {
        if (T == ComplexF32) {
            if ((m_ == 1 or n_ == 1) and k_ >= 128) {
                gemmNoTransComplexF32(m_, n_, k_, alpha, a, lda, b, ldb, beta, c, ldc);
                return;
            }
            if (tryGemmComplexF32ViaExpandedReal(.no_trans, .no_trans, m_, n_, k_, alpha, a, lda, b, ldb, beta, c, ldc)) return;
            if (tryGemmComplexF32ViaReal(.no_trans, .no_trans, m_, n_, k_, alpha, a, lda, b, ldb, beta, c, ldc)) return;
            gemmNoTransComplexF32(m_, n_, k_, alpha, a, lda, b, ldb, beta, c, ldc);
            return;
        } else if (T == ComplexF64) {
            if ((m_ == 1 or n_ == 1) and k_ >= 128) {
                gemmNoTransComplexF64(m_, n_, k_, alpha, a, lda, b, ldb, beta, c, ldc);
                return;
            }
            if (tryGemmNoTransComplexF64ViaExpandedReal(m_, n_, k_, alpha, a, lda, b, ldb, beta, c, ldc)) return;
            if (tryGemmComplexF64ViaReal(.no_trans, .no_trans, m_, n_, k_, alpha, a, lda, b, ldb, beta, c, ldc)) return;
            gemmNoTransComplexF64(m_, n_, k_, alpha, a, lda, b, ldb, beta, c, ldc);
            return;
        }
    }
    if (T == ComplexF32) {
        if (tryGemmComplexF32ViaExpandedReal(transa, transb, m_, n_, k_, alpha, a, lda, b, ldb, beta, c, ldc)) return;
        if (tryGemmComplexF32ViaReal(transa, transb, m_, n_, k_, alpha, a, lda, b, ldb, beta, c, ldc)) return;
    } else if (T == ComplexF64) {
        if (tryGemmComplexF64ViaReal(transa, transb, m_, n_, k_, alpha, a, lda, b, ldb, beta, c, ldc)) return;
    }
    for (0..n) |j| {
        for (0..m) |i| {
            var sum = zero(T);
            for (0..k) |p| sum = add(T, sum, mul(T, matrixValue(T, transa, a, lda, i, p), matrixValue(T, transb, b, ldb, p, j)));
            const idxc = matIndex(ldc, i, j);
            c[idxc] = add(T, mul(T, alpha, sum), if (isZero(T, beta)) zero(T) else mul(T, beta, c[idxc]));
        }
    }
}
