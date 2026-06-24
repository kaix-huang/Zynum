// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later

//! General, symmetric, Hermitian, and packed rank-update BLAS Level 2 kernels.

const scalar = @import("../scalar.zig");
const indexing = @import("../indexing.zig");
const level1 = @import("../level1.zig");
const core_pool = @import("../pool.zig");
const matrix_vector_kernels = @import("../../kernels/matrix_vector.zig");

const BlasInt = scalar.BlasInt;
const Uplo = scalar.Uplo;
const Real = scalar.Real;

const realScalar = scalar.realScalar;
const add = scalar.add;
const mul = scalar.mul;
const conj = scalar.conj;
const maybeConj = scalar.maybeConj;
const isZero = scalar.isZero;

const toUsize = indexing.toUsize;
const startIndex = indexing.startIndex;
const matIndex = indexing.matIndex;
const packedIndex = indexing.packedIndex;
const vectorGet = indexing.vectorGet;

fn isReal(comptime T: type) bool {
    return T == f32 or T == f64;
}

fn lanes(comptime T: type) comptime_int {
    if (T == f32) return 8;
    if (T == f64) return 4;
    @compileError("real GER vector lanes support f32 and f64");
}

fn unroll(comptime T: type) comptime_int {
    return 4 * lanes(T);
}

fn unrollVectors(comptime T: type) comptime_int {
    return unroll(T) / lanes(T);
}

inline fn loadVec(comptime T: type, comptime lane_count: comptime_int, ptr: [*]const T, index: usize) @Vector(lane_count, T) {
    const V = @Vector(lane_count, T);
    return @as(*align(1) const V, @ptrCast(ptr + index)).*;
}

inline fn storeVec(comptime T: type, comptime lane_count: comptime_int, ptr: [*]T, index: usize, value: @Vector(lane_count, T)) void {
    const V = @Vector(lane_count, T);
    @as(*align(1) V, @ptrCast(ptr + index)).* = value;
}

fn gerUnitReal(comptime T: type, m: usize, n: usize, alpha: T, x: [*]const T, y: [*]const T, a: [*]T, lda: BlasInt) void {
    if (matrix_vector_kernels.gerUnitReal(T, m, n, alpha, x, y, a, lda)) return;
    const lane_count = lanes(T);
    const unroll_count = unroll(T);
    const V = @Vector(lane_count, T);
    var j: usize = 0;
    while (j + 4 <= n) : (j += 4) {
        const s0: V = @splat(alpha * y[j]);
        const s1: V = @splat(alpha * y[j + 1]);
        const s2: V = @splat(alpha * y[j + 2]);
        const s3: V = @splat(alpha * y[j + 3]);
        const c0 = a + matIndex(lda, 0, j);
        const c1 = a + matIndex(lda, 0, j + 1);
        const c2 = a + matIndex(lda, 0, j + 2);
        const c3 = a + matIndex(lda, 0, j + 3);
        var i: usize = 0;
        while (i + unroll_count <= m) : (i += unroll_count) {
            inline for (0..unrollVectors(T)) |k| {
                const offset = i + k * lane_count;
                const xv = loadVec(T, lane_count, x, offset);
                storeVec(T, lane_count, c0, offset, @mulAdd(V, xv, s0, loadVec(T, lane_count, c0, offset)));
                storeVec(T, lane_count, c1, offset, @mulAdd(V, xv, s1, loadVec(T, lane_count, c1, offset)));
                storeVec(T, lane_count, c2, offset, @mulAdd(V, xv, s2, loadVec(T, lane_count, c2, offset)));
                storeVec(T, lane_count, c3, offset, @mulAdd(V, xv, s3, loadVec(T, lane_count, c3, offset)));
            }
        }
        while (i + lane_count <= m) : (i += lane_count) {
            const xv = loadVec(T, lane_count, x, i);
            storeVec(T, lane_count, c0, i, @mulAdd(V, xv, s0, loadVec(T, lane_count, c0, i)));
            storeVec(T, lane_count, c1, i, @mulAdd(V, xv, s1, loadVec(T, lane_count, c1, i)));
            storeVec(T, lane_count, c2, i, @mulAdd(V, xv, s2, loadVec(T, lane_count, c2, i)));
            storeVec(T, lane_count, c3, i, @mulAdd(V, xv, s3, loadVec(T, lane_count, c3, i)));
        }
        while (i < m) : (i += 1) {
            const xv = x[i];
            c0[i] = @mulAdd(T, xv, alpha * y[j], c0[i]);
            c1[i] = @mulAdd(T, xv, alpha * y[j + 1], c1[i]);
            c2[i] = @mulAdd(T, xv, alpha * y[j + 2], c2[i]);
            c3[i] = @mulAdd(T, xv, alpha * y[j + 3], c3[i]);
        }
    }
    while (j < n) : (j += 1) {
        const temp = alpha * y[j];
        if (temp != 0) level1.axpyUnitReal(T, m, temp, x, a + matIndex(lda, 0, j));
    }
}

fn GerTask(comptime T: type) type {
    return struct {
        m: usize,
        n0: usize,
        n1: usize,
        alpha: T,
        x: [*]const T,
        y: [*]const T,
        a: [*]T,
        lda: BlasInt,
    };
}

fn runGerTask(comptime T: type, raw_tasks: *const anyopaque, index: usize) void {
    const tasks: [*]const GerTask(T) = @ptrCast(@alignCast(raw_tasks));
    const task = tasks[index];
    gerUnitReal(T, task.m, task.n1 - task.n0, task.alpha, task.x, task.y + task.n0, task.a + matIndex(task.lda, 0, task.n0), task.lda);
}

fn runGerTaskF32(raw_tasks: *const anyopaque, index: usize) void {
    runGerTask(f32, raw_tasks, index);
}

fn runGerTaskF64(raw_tasks: *const anyopaque, index: usize) void {
    runGerTask(f64, raw_tasks, index);
}

fn parallelGerUnitReal(comptime T: type, m: usize, n: usize, alpha: T, x: [*]const T, y: [*]const T, a: [*]T, lda: BlasInt) bool {
    const min_cols_per_task: usize = if (n >= 1024) 128 else 256;
    const task_count = core_pool.taskCount(n, min_cols_per_task);
    if (task_count <= 1) return false;

    var tasks: [core_pool.max_tasks]GerTask(T) = undefined;
    for (0..task_count) |task_index| {
        const n0 = task_index * n / task_count;
        const n1 = (task_index + 1) * n / task_count;
        tasks[task_index] = .{
            .m = m,
            .n0 = n0,
            .n1 = n1,
            .alpha = alpha,
            .x = x,
            .y = y,
            .a = a,
            .lda = lda,
        };
    }

    const runner = if (T == f32) runGerTaskF32 else runGerTaskF64;
    return core_pool.run(runner, @ptrCast(&tasks), task_count);
}

pub fn ger(comptime T: type, m_: BlasInt, n_: BlasInt, alpha: T, x: [*]const T, incx_: BlasInt, y: [*]const T, incy_: BlasInt, a: [*]T, lda: BlasInt, conj_y: bool) void {
    if (m_ <= 0 or n_ <= 0 or incx_ == 0 or incy_ == 0 or isZero(T, alpha)) return;
    const m = toUsize(m_);
    const n = toUsize(n_);
    if (comptime isReal(T)) {
        if (!conj_y and incx_ == 1 and incy_ == 1) {
            if (parallelGerUnitReal(T, m, n, alpha, x, y, a, lda)) return;
            return gerUnitReal(T, m, n, alpha, x, y, a, lda);
        }
    }
    const sx = startIndex(m_, incx_);
    const sy = startIndex(n_, incy_);
    for (0..n) |j| {
        const yj = maybeConj(T, vectorGet(T, y, sy, j, incy_), conj_y);
        const temp = mul(T, alpha, yj);
        for (0..m) |i| {
            const idxa = matIndex(lda, i, j);
            a[idxa] = add(T, a[idxa], mul(T, vectorGet(T, x, sx, i, incx_), temp));
        }
    }
}

pub fn syr(comptime T: type, uplo: Uplo, n_: BlasInt, alpha: T, x: [*]const T, incx_: BlasInt, a: [*]T, lda: BlasInt) void {
    if (n_ <= 0 or incx_ == 0 or isZero(T, alpha)) return;
    const n = toUsize(n_);
    const sx = startIndex(n_, incx_);
    for (0..n) |j| {
        const xj = vectorGet(T, x, sx, j, incx_);
        const temp = mul(T, alpha, xj);
        const row0: usize = if (uplo == .upper) 0 else j;
        const row1: usize = if (uplo == .upper) j + 1 else n;
        for (row0..row1) |i| {
            const idxa = matIndex(lda, i, j);
            a[idxa] = add(T, a[idxa], mul(T, vectorGet(T, x, sx, i, incx_), temp));
        }
    }
}

pub fn spr(comptime T: type, uplo: Uplo, n_: BlasInt, alpha: T, x: [*]const T, incx_: BlasInt, ap: [*]T) void {
    if (n_ <= 0 or incx_ == 0 or isZero(T, alpha)) return;
    const n = toUsize(n_);
    const sx = startIndex(n_, incx_);
    for (0..n) |j| {
        const xj = vectorGet(T, x, sx, j, incx_);
        const temp = mul(T, alpha, xj);
        const row0: usize = if (uplo == .upper) 0 else j;
        const row1: usize = if (uplo == .upper) j + 1 else n;
        for (row0..row1) |i| {
            const idxa = packedIndex(uplo, n, i, j);
            ap[idxa] = add(T, ap[idxa], mul(T, vectorGet(T, x, sx, i, incx_), temp));
        }
    }
}

pub fn syr2(comptime T: type, uplo: Uplo, n_: BlasInt, alpha: T, x: [*]const T, incx_: BlasInt, y: [*]const T, incy_: BlasInt, a: [*]T, lda: BlasInt) void {
    if (n_ <= 0 or incx_ == 0 or incy_ == 0 or isZero(T, alpha)) return;
    const n = toUsize(n_);
    const sx = startIndex(n_, incx_);
    const sy = startIndex(n_, incy_);
    for (0..n) |j| {
        const temp1 = mul(T, alpha, vectorGet(T, y, sy, j, incy_));
        const temp2 = mul(T, alpha, vectorGet(T, x, sx, j, incx_));
        const row0: usize = if (uplo == .upper) 0 else j;
        const row1: usize = if (uplo == .upper) j + 1 else n;
        for (row0..row1) |i| {
            const upd = add(T, mul(T, vectorGet(T, x, sx, i, incx_), temp1), mul(T, vectorGet(T, y, sy, i, incy_), temp2));
            const idxa = matIndex(lda, i, j);
            a[idxa] = add(T, a[idxa], upd);
        }
    }
}

pub fn spr2(comptime T: type, uplo: Uplo, n_: BlasInt, alpha: T, x: [*]const T, incx_: BlasInt, y: [*]const T, incy_: BlasInt, ap: [*]T) void {
    if (n_ <= 0 or incx_ == 0 or incy_ == 0 or isZero(T, alpha)) return;
    const n = toUsize(n_);
    const sx = startIndex(n_, incx_);
    const sy = startIndex(n_, incy_);
    for (0..n) |j| {
        const temp1 = mul(T, alpha, vectorGet(T, y, sy, j, incy_));
        const temp2 = mul(T, alpha, vectorGet(T, x, sx, j, incx_));
        const row0: usize = if (uplo == .upper) 0 else j;
        const row1: usize = if (uplo == .upper) j + 1 else n;
        for (row0..row1) |i| {
            const upd = add(T, mul(T, vectorGet(T, x, sx, i, incx_), temp1), mul(T, vectorGet(T, y, sy, i, incy_), temp2));
            const idxa = packedIndex(uplo, n, i, j);
            ap[idxa] = add(T, ap[idxa], upd);
        }
    }
}

pub fn her(comptime T: type, uplo: Uplo, n_: BlasInt, alpha: Real(T), x: [*]const T, incx_: BlasInt, a: [*]T, lda: BlasInt) void {
    if (n_ <= 0 or incx_ == 0 or alpha == 0) return;
    const n = toUsize(n_);
    const sx = startIndex(n_, incx_);
    for (0..n) |j| {
        const temp = mul(T, realScalar(T, alpha), conj(T, vectorGet(T, x, sx, j, incx_)));
        const row0: usize = if (uplo == .upper) 0 else j;
        const row1: usize = if (uplo == .upper) j + 1 else n;
        for (row0..row1) |i| {
            const idxa = matIndex(lda, i, j);
            a[idxa] = add(T, a[idxa], mul(T, vectorGet(T, x, sx, i, incx_), temp));
            if (i == j) a[idxa].im = 0;
        }
    }
}

pub fn hpr(comptime T: type, uplo: Uplo, n_: BlasInt, alpha: Real(T), x: [*]const T, incx_: BlasInt, ap: [*]T) void {
    if (n_ <= 0 or incx_ == 0 or alpha == 0) return;
    const n = toUsize(n_);
    const sx = startIndex(n_, incx_);
    for (0..n) |j| {
        const temp = mul(T, realScalar(T, alpha), conj(T, vectorGet(T, x, sx, j, incx_)));
        const row0: usize = if (uplo == .upper) 0 else j;
        const row1: usize = if (uplo == .upper) j + 1 else n;
        for (row0..row1) |i| {
            const idxa = packedIndex(uplo, n, i, j);
            ap[idxa] = add(T, ap[idxa], mul(T, vectorGet(T, x, sx, i, incx_), temp));
            if (i == j) ap[idxa].im = 0;
        }
    }
}

pub fn her2(comptime T: type, uplo: Uplo, n_: BlasInt, alpha: T, x: [*]const T, incx_: BlasInt, y: [*]const T, incy_: BlasInt, a: [*]T, lda: BlasInt) void {
    if (n_ <= 0 or incx_ == 0 or incy_ == 0 or isZero(T, alpha)) return;
    const n = toUsize(n_);
    const sx = startIndex(n_, incx_);
    const sy = startIndex(n_, incy_);
    for (0..n) |j| {
        const temp1 = mul(T, alpha, conj(T, vectorGet(T, y, sy, j, incy_)));
        const temp2 = mul(T, conj(T, alpha), conj(T, vectorGet(T, x, sx, j, incx_)));
        const row0: usize = if (uplo == .upper) 0 else j;
        const row1: usize = if (uplo == .upper) j + 1 else n;
        for (row0..row1) |i| {
            const upd = add(T, mul(T, vectorGet(T, x, sx, i, incx_), temp1), mul(T, vectorGet(T, y, sy, i, incy_), temp2));
            const idxa = matIndex(lda, i, j);
            a[idxa] = add(T, a[idxa], upd);
            if (i == j) a[idxa].im = 0;
        }
    }
}

pub fn hpr2(comptime T: type, uplo: Uplo, n_: BlasInt, alpha: T, x: [*]const T, incx_: BlasInt, y: [*]const T, incy_: BlasInt, ap: [*]T) void {
    if (n_ <= 0 or incx_ == 0 or incy_ == 0 or isZero(T, alpha)) return;
    const n = toUsize(n_);
    const sx = startIndex(n_, incx_);
    const sy = startIndex(n_, incy_);
    for (0..n) |j| {
        const temp1 = mul(T, alpha, conj(T, vectorGet(T, y, sy, j, incy_)));
        const temp2 = mul(T, conj(T, alpha), conj(T, vectorGet(T, x, sx, j, incx_)));
        const row0: usize = if (uplo == .upper) 0 else j;
        const row1: usize = if (uplo == .upper) j + 1 else n;
        for (row0..row1) |i| {
            const upd = add(T, mul(T, vectorGet(T, x, sx, i, incx_), temp1), mul(T, vectorGet(T, y, sy, i, incy_), temp2));
            const idxa = packedIndex(uplo, n, i, j);
            ap[idxa] = add(T, ap[idxa], upd);
            if (i == j) ap[idxa].im = 0;
        }
    }
}
