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
const isComplex = scalar.isComplex;
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

inline fn loadVec(comptime T: type, comptime lane_count: comptime_int, ptr: [*]const T, index: usize) @Vector(lane_count, T) {
    const V = @Vector(lane_count, T);
    return @as(*align(1) const V, @ptrCast(ptr + index)).*;
}

inline fn storeVec(comptime T: type, comptime lane_count: comptime_int, ptr: [*]T, index: usize, value: @Vector(lane_count, T)) void {
    const V = @Vector(lane_count, T);
    @as(*align(1) V, @ptrCast(ptr + index)).* = value;
}

fn gerUnitRealBlocked(comptime T: type, comptime lane_count: comptime_int, comptime vector_unroll: comptime_int, m: usize, n: usize, alpha: T, x: [*]const T, y: [*]const T, a: [*]T, lda: BlasInt) void {
    const unroll_count = vector_unroll * lane_count;
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
            inline for (0..vector_unroll) |k| {
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

fn gerUnitReal(comptime T: type, m: usize, n: usize, alpha: T, x: [*]const T, y: [*]const T, a: [*]T, lda: BlasInt) void {
    if (matrix_vector_kernels.gerUnitReal(T, m, n, alpha, x, y, a, lda)) return;
    if (T == f64 and m >= 512 and m < 1024) return gerUnitRealBlocked(T, 8, 8, m, n, alpha, x, y, a, lda);
    return gerUnitRealBlocked(T, lanes(T), 4, m, n, alpha, x, y, a, lda);
}

fn GerTask(comptime T: type) type {
    return struct {
        m0: usize,
        m1: usize,
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
    gerUnitReal(
        T,
        task.m1 - task.m0,
        task.n1 - task.n0,
        task.alpha,
        task.x + task.m0,
        task.y + task.n0,
        task.a + matIndex(task.lda, task.m0, task.n0),
        task.lda,
    );
}

fn runGerTaskF32(raw_tasks: *const anyopaque, index: usize) void {
    runGerTask(f32, raw_tasks, index);
}

fn runGerTaskF64(raw_tasks: *const anyopaque, index: usize) void {
    runGerTask(f64, raw_tasks, index);
}

fn parallelGerUnitReal(comptime T: type, m: usize, n: usize, alpha: T, x: [*]const T, y: [*]const T, a: [*]T, lda: BlasInt) bool {
    const min_cols_per_task: usize = if (n >= 768) 32 else if (n >= 512) 64 else if (n >= 256) 80 else 256;
    const max_task_count: usize = if (n >= 1536) 10 else if (n >= 512) 4 else if (n >= 384) 10 else 8;
    const task_count = @min(core_pool.taskCount(n, min_cols_per_task), max_task_count);
    if (task_count <= 1) return false;

    const block_cols: usize = 4;
    const block_count = n / block_cols;
    var tasks: [core_pool.max_tasks]GerTask(T) = undefined;
    for (0..task_count) |task_index| {
        const n0, const n1 = if (block_count > 0) .{
            (task_index * block_count / task_count) * block_cols,
            if (task_index + 1 == task_count) n else ((task_index + 1) * block_count / task_count) * block_cols,
        } else .{
            task_index * n / task_count,
            (task_index + 1) * n / task_count,
        };
        tasks[task_index] = .{
            .m0 = 0,
            .m1 = m,
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
    if ((T == f32 or T == f64) and n < 1536) return core_pool.runLowLatency(runner, @ptrCast(&tasks), task_count);
    return core_pool.run(runner, @ptrCast(&tasks), task_count);
}

inline fn c64GerTerm(xv: @Vector(4, f64), coeff: scalar.ComplexF64) @Vector(4, f64) {
    const swap_mask: @Vector(4, i32) = .{ 1, 0, 3, 2 };
    const re_v: @Vector(4, f64) = @splat(coeff.re);
    const im_v: @Vector(4, f64) = .{ -coeff.im, coeff.im, -coeff.im, coeff.im };
    return @mulAdd(@Vector(4, f64), xv, re_v, @shuffle(f64, xv, undefined, swap_mask) * im_v);
}

fn c64Ger4(m: usize, c0: scalar.ComplexF64, c1: scalar.ComplexF64, c2: scalar.ComplexF64, c3: scalar.ComplexF64, x: [*]const scalar.ComplexF64, a0: [*]scalar.ComplexF64, a1: [*]scalar.ComplexF64, a2: [*]scalar.ComplexF64, a3: [*]scalar.ComplexF64) void {
    const real_x: [*]const f64 = @ptrCast(x);
    const r0: [*]f64 = @ptrCast(a0);
    const r1: [*]f64 = @ptrCast(a1);
    const r2: [*]f64 = @ptrCast(a2);
    const r3: [*]f64 = @ptrCast(a3);
    const real_n = 2 * m;
    var i: usize = 0;
    while (i + 4 <= real_n) : (i += 4) {
        const xv = loadVec(f64, 4, real_x, i);
        storeVec(f64, 4, r0, i, loadVec(f64, 4, r0, i) + c64GerTerm(xv, c0));
        storeVec(f64, 4, r1, i, loadVec(f64, 4, r1, i) + c64GerTerm(xv, c1));
        storeVec(f64, 4, r2, i, loadVec(f64, 4, r2, i) + c64GerTerm(xv, c2));
        storeVec(f64, 4, r3, i, loadVec(f64, 4, r3, i) + c64GerTerm(xv, c3));
    }
    while (i < real_n) : (i += 2) {
        const xr = real_x[i];
        const xi = real_x[i + 1];
        r0[i] += c0.re * xr - c0.im * xi;
        r0[i + 1] += c0.re * xi + c0.im * xr;
        r1[i] += c1.re * xr - c1.im * xi;
        r1[i + 1] += c1.re * xi + c1.im * xr;
        r2[i] += c2.re * xr - c2.im * xi;
        r2[i + 1] += c2.re * xi + c2.im * xr;
        r3[i] += c3.re * xr - c3.im * xi;
        r3[i + 1] += c3.re * xi + c3.im * xr;
    }
}

fn gerUnitComplexC64(m: usize, n: usize, alpha: scalar.ComplexF64, x: [*]const scalar.ComplexF64, y: [*]const scalar.ComplexF64, a: [*]scalar.ComplexF64, lda: BlasInt, conj_y: bool) void {
    var j: usize = 0;
    while (j + 4 <= n) : (j += 4) {
        c64Ger4(
            m,
            mul(scalar.ComplexF64, alpha, maybeConj(scalar.ComplexF64, y[j], conj_y)),
            mul(scalar.ComplexF64, alpha, maybeConj(scalar.ComplexF64, y[j + 1], conj_y)),
            mul(scalar.ComplexF64, alpha, maybeConj(scalar.ComplexF64, y[j + 2], conj_y)),
            mul(scalar.ComplexF64, alpha, maybeConj(scalar.ComplexF64, y[j + 3], conj_y)),
            x,
            a + matIndex(lda, 0, j),
            a + matIndex(lda, 0, j + 1),
            a + matIndex(lda, 0, j + 2),
            a + matIndex(lda, 0, j + 3),
        );
    }
    while (j < n) : (j += 1) {
        const temp = mul(scalar.ComplexF64, alpha, maybeConj(scalar.ComplexF64, y[j], conj_y));
        if (!isZero(scalar.ComplexF64, temp)) level1.axpy(scalar.ComplexF64, @intCast(m), temp, x, 1, a + matIndex(lda, 0, j), 1);
    }
}

fn gerUnitComplex(comptime T: type, m_: BlasInt, n: usize, alpha: T, x: [*]const T, y: [*]const T, a: [*]T, lda: BlasInt, conj_y: bool) void {
    if (T == scalar.ComplexF64 and !conj_y and m_ == 256) {
        return gerUnitComplexC64(toUsize(m_), n, alpha, x, y, a, lda, false);
    }
    for (0..n) |j| {
        const yj = maybeConj(T, y[j], conj_y);
        const temp = mul(T, alpha, yj);
        if (!isZero(T, temp)) level1.axpy(T, m_, temp, x, 1, a + matIndex(lda, 0, j), 1);
    }
}

fn ComplexGerTask(comptime T: type) type {
    return struct {
        m: BlasInt,
        n0: usize,
        n1: usize,
        alpha: T,
        x: [*]const T,
        y: [*]const T,
        a: [*]T,
        lda: BlasInt,
        conj_y: bool,
    };
}

fn runComplexGerTask(comptime T: type, raw_tasks: *const anyopaque, index: usize) void {
    const tasks: [*]const ComplexGerTask(T) = @ptrCast(@alignCast(raw_tasks));
    const task = tasks[index];
    gerUnitComplex(
        T,
        task.m,
        task.n1 - task.n0,
        task.alpha,
        task.x,
        task.y + task.n0,
        task.a + matIndex(task.lda, 0, task.n0),
        task.lda,
        task.conj_y,
    );
}

fn runComplexGerTaskC32(raw_tasks: *const anyopaque, index: usize) void {
    runComplexGerTask(scalar.ComplexF32, raw_tasks, index);
}

fn runComplexGerTaskC64(raw_tasks: *const anyopaque, index: usize) void {
    runComplexGerTask(scalar.ComplexF64, raw_tasks, index);
}

fn parallelGerUnitComplex(comptime T: type, m: usize, n: usize, alpha: T, x: [*]const T, y: [*]const T, a: [*]T, lda: BlasInt, conj_y: bool) bool {
    if (m *| n < 256 * 256) return false;
    const min_cols_per_task: usize = if (T == scalar.ComplexF32) 64 else 48;
    var task_count = core_pool.taskCount(n, min_cols_per_task);
    const max_task_count: usize = if (n < 512) 4 else 8;
    task_count = @min(task_count, max_task_count);
    if (task_count <= 1) return false;

    const block_cols: usize = 4;
    const block_count = n / block_cols;
    var tasks: [core_pool.max_tasks]ComplexGerTask(T) = undefined;
    for (0..task_count) |task_index| {
        const n0, const n1 = if (block_count > 0) .{
            (task_index * block_count / task_count) * block_cols,
            if (task_index + 1 == task_count) n else ((task_index + 1) * block_count / task_count) * block_cols,
        } else .{
            task_index * n / task_count,
            (task_index + 1) * n / task_count,
        };
        tasks[task_index] = .{
            .m = @intCast(m),
            .n0 = n0,
            .n1 = n1,
            .alpha = alpha,
            .x = x,
            .y = y,
            .a = a,
            .lda = lda,
            .conj_y = conj_y,
        };
    }

    const runner = if (T == scalar.ComplexF32) runComplexGerTaskC32 else runComplexGerTaskC64;
    return core_pool.runLowLatency(runner, @ptrCast(&tasks), task_count);
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
    } else if (comptime isComplex(T)) {
        if (incx_ == 1 and incy_ == 1) {
            if (parallelGerUnitComplex(T, m, n, alpha, x, y, a, lda, conj_y)) return;
            return gerUnitComplex(T, m_, n, alpha, x, y, a, lda, conj_y);
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
