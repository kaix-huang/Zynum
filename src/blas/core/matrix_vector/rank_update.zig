// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later

//! General, symmetric, Hermitian, and packed rank-update BLAS Level 2 kernels.

const std = @import("std");
const builtin = @import("builtin");

const scalar = @import("../shared/scalar.zig");
const indexing = @import("../shared/indexing.zig");
const vector_ops = @import("../vector.zig");
const core_pool = @import("../execution/thread_pool.zig");
const matrix_vector_kernels = @import("../../kernels/dispatch/matrix_vector.zig");
const level2_tuning = @import("../../kernels/shared/matrix_vector/tuning.zig");
const runtime = @import("../../runtime.zig");

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
const tuning = level2_tuning.active.rank_update;

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
        if (temp != 0) vector_ops.axpyUnitReal(T, m, temp, x, a + matIndex(lda, 0, j));
    }
}

fn gerUnitReal(comptime T: type, m: usize, n: usize, alpha: T, x: [*]const T, y: [*]const T, a: [*]T, lda: BlasInt) void {
    if (matrix_vector_kernels.gerUnitReal(T, m, n, alpha, x, y, a, lda)) return;
    if (tuning.preferRealGerWideF64(T, m)) return gerUnitRealBlocked(T, 8, 8, m, n, alpha, x, y, a, lda);
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
    if (!tuning.preferRealGerParallel(m, n)) return false;
    const task_count = tuning.capRealGerTasks(n, core_pool.taskCount(n, tuning.realGerMinColumns(n)));
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
    if (tuning.useRealGerLowLatency(T, n)) return core_pool.runLowLatency(runner, @ptrCast(&tasks), task_count);
    if (comptime builtin.cpu.arch == .x86_64) return core_pool.runLowLatency(runner, @ptrCast(&tasks), task_count);
    return core_pool.run(runner, @ptrCast(&tasks), task_count);
}

inline fn c64GerTerm(xv: @Vector(4, f64), coeff: scalar.ComplexF64) @Vector(4, f64) {
    const swap_mask: @Vector(4, i32) = .{ 1, 0, 3, 2 };
    const re_v: @Vector(4, f64) = @splat(coeff.re);
    const im_v: @Vector(4, f64) = .{ -coeff.im, coeff.im, -coeff.im, coeff.im };
    return @mulAdd(@Vector(4, f64), xv, re_v, @shuffle(f64, xv, undefined, swap_mask) * im_v);
}

inline fn c32GerTerm(xv: @Vector(8, f32), coeff: scalar.ComplexF32) @Vector(8, f32) {
    const swap_mask: @Vector(8, i32) = .{ 1, 0, 3, 2, 5, 4, 7, 6 };
    const re_v: @Vector(8, f32) = @splat(coeff.re);
    const im_v: @Vector(8, f32) = .{ -coeff.im, coeff.im, -coeff.im, coeff.im, -coeff.im, coeff.im, -coeff.im, coeff.im };
    return @mulAdd(@Vector(8, f32), xv, re_v, @shuffle(f32, xv, undefined, swap_mask) * im_v);
}

fn c32Ger4(m: usize, c0: scalar.ComplexF32, c1: scalar.ComplexF32, c2: scalar.ComplexF32, c3: scalar.ComplexF32, x: [*]const scalar.ComplexF32, a0: [*]scalar.ComplexF32, a1: [*]scalar.ComplexF32, a2: [*]scalar.ComplexF32, a3: [*]scalar.ComplexF32) void {
    const real_x: [*]const f32 = @ptrCast(x);
    const r0: [*]f32 = @ptrCast(a0);
    const r1: [*]f32 = @ptrCast(a1);
    const r2: [*]f32 = @ptrCast(a2);
    const r3: [*]f32 = @ptrCast(a3);
    const real_n = 2 * m;
    var i: usize = 0;
    while (i + 16 <= real_n) : (i += 16) {
        const xv0 = loadVec(f32, 8, real_x, i);
        const xv1 = loadVec(f32, 8, real_x, i + 8);
        storeVec(f32, 8, r0, i, loadVec(f32, 8, r0, i) + c32GerTerm(xv0, c0));
        storeVec(f32, 8, r0, i + 8, loadVec(f32, 8, r0, i + 8) + c32GerTerm(xv1, c0));
        storeVec(f32, 8, r1, i, loadVec(f32, 8, r1, i) + c32GerTerm(xv0, c1));
        storeVec(f32, 8, r1, i + 8, loadVec(f32, 8, r1, i + 8) + c32GerTerm(xv1, c1));
        storeVec(f32, 8, r2, i, loadVec(f32, 8, r2, i) + c32GerTerm(xv0, c2));
        storeVec(f32, 8, r2, i + 8, loadVec(f32, 8, r2, i + 8) + c32GerTerm(xv1, c2));
        storeVec(f32, 8, r3, i, loadVec(f32, 8, r3, i) + c32GerTerm(xv0, c3));
        storeVec(f32, 8, r3, i + 8, loadVec(f32, 8, r3, i + 8) + c32GerTerm(xv1, c3));
    }
    while (i + 8 <= real_n) : (i += 8) {
        const xv = loadVec(f32, 8, real_x, i);
        storeVec(f32, 8, r0, i, loadVec(f32, 8, r0, i) + c32GerTerm(xv, c0));
        storeVec(f32, 8, r1, i, loadVec(f32, 8, r1, i) + c32GerTerm(xv, c1));
        storeVec(f32, 8, r2, i, loadVec(f32, 8, r2, i) + c32GerTerm(xv, c2));
        storeVec(f32, 8, r3, i, loadVec(f32, 8, r3, i) + c32GerTerm(xv, c3));
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

fn gerUnitComplexC32(m: usize, n: usize, alpha: scalar.ComplexF32, x: [*]const scalar.ComplexF32, y: [*]const scalar.ComplexF32, a: [*]scalar.ComplexF32, lda: BlasInt, conj_y: bool) void {
    var j: usize = 0;
    while (j + 4 <= n) : (j += 4) {
        c32Ger4(
            m,
            mul(scalar.ComplexF32, alpha, maybeConj(scalar.ComplexF32, y[j], conj_y)),
            mul(scalar.ComplexF32, alpha, maybeConj(scalar.ComplexF32, y[j + 1], conj_y)),
            mul(scalar.ComplexF32, alpha, maybeConj(scalar.ComplexF32, y[j + 2], conj_y)),
            mul(scalar.ComplexF32, alpha, maybeConj(scalar.ComplexF32, y[j + 3], conj_y)),
            x,
            a + matIndex(lda, 0, j),
            a + matIndex(lda, 0, j + 1),
            a + matIndex(lda, 0, j + 2),
            a + matIndex(lda, 0, j + 3),
        );
    }
    while (j < n) : (j += 1) {
        const temp = mul(scalar.ComplexF32, alpha, maybeConj(scalar.ComplexF32, y[j], conj_y));
        if (!isZero(scalar.ComplexF32, temp)) vector_ops.axpy(scalar.ComplexF32, @intCast(m), temp, x, 1, a + matIndex(lda, 0, j), 1);
    }
}

fn c64Ger4(m: usize, c0: scalar.ComplexF64, c1: scalar.ComplexF64, c2: scalar.ComplexF64, c3: scalar.ComplexF64, x: [*]const scalar.ComplexF64, a0: [*]scalar.ComplexF64, a1: [*]scalar.ComplexF64, a2: [*]scalar.ComplexF64, a3: [*]scalar.ComplexF64) void {
    const real_x: [*]const f64 = @ptrCast(x);
    const r0: [*]f64 = @ptrCast(a0);
    const r1: [*]f64 = @ptrCast(a1);
    const r2: [*]f64 = @ptrCast(a2);
    const r3: [*]f64 = @ptrCast(a3);
    const real_n = 2 * m;
    var i: usize = 0;
    while (i + 8 <= real_n) : (i += 8) {
        const xv0 = loadVec(f64, 4, real_x, i);
        const xv1 = loadVec(f64, 4, real_x, i + 4);
        storeVec(f64, 4, r0, i, loadVec(f64, 4, r0, i) + c64GerTerm(xv0, c0));
        storeVec(f64, 4, r0, i + 4, loadVec(f64, 4, r0, i + 4) + c64GerTerm(xv1, c0));
        storeVec(f64, 4, r1, i, loadVec(f64, 4, r1, i) + c64GerTerm(xv0, c1));
        storeVec(f64, 4, r1, i + 4, loadVec(f64, 4, r1, i + 4) + c64GerTerm(xv1, c1));
        storeVec(f64, 4, r2, i, loadVec(f64, 4, r2, i) + c64GerTerm(xv0, c2));
        storeVec(f64, 4, r2, i + 4, loadVec(f64, 4, r2, i + 4) + c64GerTerm(xv1, c2));
        storeVec(f64, 4, r3, i, loadVec(f64, 4, r3, i) + c64GerTerm(xv0, c3));
        storeVec(f64, 4, r3, i + 4, loadVec(f64, 4, r3, i + 4) + c64GerTerm(xv1, c3));
    }
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
        if (!isZero(scalar.ComplexF64, temp)) vector_ops.axpy(scalar.ComplexF64, @intCast(m), temp, x, 1, a + matIndex(lda, 0, j), 1);
    }
}

fn gerUnitComplex(comptime T: type, m_: BlasInt, n: usize, alpha: T, x: [*]const T, y: [*]const T, a: [*]T, lda: BlasInt, conj_y: bool) void {
    if (tuning.enable_fixed_complex_ger and
        matrix_vector_kernels.gerUnitComplex(T, toUsize(m_), n, alpha, x, y, a, lda, conj_y))
    {
        return;
    }
    if (T == scalar.ComplexF32 and tuning.preferComplexGerSpecialized(T, toUsize(m_))) {
        return gerUnitComplexC32(toUsize(m_), n, alpha, x, y, a, lda, conj_y);
    }
    if (T == scalar.ComplexF64 and tuning.preferComplexGerSpecialized(T, toUsize(m_))) {
        return gerUnitComplexC64(toUsize(m_), n, alpha, x, y, a, lda, conj_y);
    }
    for (0..n) |j| {
        const yj = maybeConj(T, y[j], conj_y);
        const temp = mul(T, alpha, yj);
        if (!isZero(T, temp)) vector_ops.axpy(T, m_, temp, x, 1, a + matIndex(lda, 0, j), 1);
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
    const exact_c64_ger128 = T == scalar.ComplexF64 and m == 128 and n == 128;
    if (!tuning.preferComplexGerParallel(exact_c64_ger128, m, n)) return false;
    var task_count = core_pool.taskCount(n, tuning.complexGerMinColumns(T, exact_c64_ger128));
    task_count = tuning.capComplexGerTasks(T, exact_c64_ger128, n, task_count);
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
        if (!conj_y and level2_tuning.coreUnitSelected(T, .ger, .{ .m = m, .n = n, .incx = incx_, .incy = incy_ })) {
            if (parallelGerUnitReal(T, m, n, alpha, x, y, a, lda)) return;
            return gerUnitReal(T, m, n, alpha, x, y, a, lda);
        }
    } else if (comptime isComplex(T)) {
        const unit_selected = if (conj_y)
            level2_tuning.coreUnitSelected(T, .gerc, .{ .m = m, .n = n, .incx = incx_, .incy = incy_ })
        else
            level2_tuning.coreUnitSelected(T, .geru, .{ .m = m, .n = n, .incx = incx_, .incy = incy_ });
        if (unit_selected) {
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

const DenseRankOperation = enum {
    syr,
    her,
    syr2,
    her2,
};

fn preferPackedColumns(n: usize) bool {
    if (comptime builtin.cpu.arch == .x86_64) return n >= 128;
    return tuning.preferPackedStructuredParallel(n);
}

fn rankUpdateTaskLimit(comptime T: type, comptime operation: DenseRankOperation, comptime is_packed: bool, n: usize, requested: usize) usize {
    if (comptime builtin.cpu.arch != .x86_64) return requested;
    var limit = requested;
    if (n >= 4096) {
        if (comptime T == scalar.ComplexF64 and operation == .her2) limit = @min(limit, 8);
        if (comptime is_packed and T == scalar.ComplexF64 and operation == .her) limit = @min(limit, 8);
        if (comptime is_packed and T == f32 and operation == .syr2) limit = @min(limit, 8);
        if (comptime is_packed and T == f64 and operation == .syr2) limit = @min(limit, 16);
        if (comptime !is_packed and T == scalar.ComplexF32 and operation == .her2) limit = @min(limit, 16);
    }
    return limit;
}

// Rank-two updates share one read/modify/write of the destination column.
// In particular, avoid a second pass over a column larger than L1.
fn rankTwoColumn(comptime T: type, n: usize, cx: T, x: [*]const T, cy: T, y: [*]const T, dest: [*]T) void {
    if (comptime builtin.cpu.arch != .x86_64) {
        if (!isZero(T, cx)) vector_ops.axpy(T, @intCast(n), cx, x, 1, dest, 1);
        if (!isZero(T, cy)) vector_ops.axpy(T, @intCast(n), cy, y, 1, dest, 1);
        return;
    }
    if (isZero(T, cx)) return vector_ops.axpy(T, @intCast(n), cy, y, 1, dest, 1);
    if (isZero(T, cy)) return vector_ops.axpy(T, @intCast(n), cx, x, 1, dest, 1);
    const R = if (T == f32 or T == scalar.ComplexF32) f32 else f64;
    const width = 32 / @sizeOf(R);
    const V = @Vector(width, R);
    const xp: [*]const R = @ptrCast(x);
    const yp: [*]const R = @ptrCast(y);
    const dp: [*]R = @ptrCast(dest);
    const components = if (comptime isReal(T)) 1 else 2;
    const limit = n * components;
    var i: usize = 0;
    while (i + width <= limit) : (i += width) {
        const xv = loadVec(R, width, xp, i);
        const yv = loadVec(R, width, yp, i);
        var acc = loadVec(R, width, dp, i);
        if (comptime isReal(T)) {
            acc = @mulAdd(V, xv, @as(V, @splat(cx)), acc);
            acc = @mulAdd(V, yv, @as(V, @splat(cy)), acc);
        } else {
            const swap: @Vector(width, i32) = comptime blk: {
                var mask: @Vector(width, i32) = undefined;
                for (0..width) |lane| mask[lane] = @intCast(lane ^ 1);
                break :blk mask;
            };
            var xi: V = @splat(cx.im);
            var yi: V = @splat(cy.im);
            inline for (0..width / 2) |pair| {
                xi[2 * pair] = -cx.im;
                yi[2 * pair] = -cy.im;
            }
            acc = @mulAdd(V, xv, @as(V, @splat(cx.re)), acc);
            acc = @mulAdd(V, @shuffle(R, xv, undefined, swap), xi, acc);
            acc = @mulAdd(V, yv, @as(V, @splat(cy.re)), acc);
            acc = @mulAdd(V, @shuffle(R, yv, undefined, swap), yi, acc);
        }
        storeVec(R, width, dp, i, acc);
    }
    var row = i / components;
    while (row < n) : (row += 1) {
        dest[row] = if (comptime isReal(T))
            @mulAdd(T, cy, y[row], @mulAdd(T, cx, x[row], dest[row]))
        else
            add(T, add(T, dest[row], mul(T, cx, x[row])), mul(T, cy, y[row]));
    }
}

// Reuse x/y across four real or two complex destination columns. The complex
// tile is narrower so that its real/imaginary coefficients fit AVX2 registers.
inline fn rankTwoPanel(comptime T: type, comptime columns: usize, comptime is_packed: bool, n: usize, cx: [columns]T, x: [*]const T, cy: [columns]T, y: [*]const T, dest: [columns][*]T) void {
    const R = if (T == f32 or T == scalar.ComplexF32) f32 else f64;
    const width = 32 / @sizeOf(R);
    const V = @Vector(width, R);
    const components = if (comptime isReal(T)) 1 else 2;
    const xp: [*]const R = @ptrCast(x);
    const yp: [*]const R = @ptrCast(y);
    var i: usize = 0;
    // Narrow tiles leave enough registers for a longer row loop. Dense f64
    // four-column panels retain the shorter loop after the large-size holdout.
    if (comptime isReal(T) and (columns == 2 or (T == f64 and is_packed))) {
        const row_vectors = if (columns == 2) 4 else 2;
        while (i + row_vectors * width <= n) : (i += row_vectors * width) {
            inline for (0..row_vectors) |step| {
                const offset = i + step * width;
                const xv = loadVec(R, width, xp, offset);
                const yv = loadVec(R, width, yp, offset);
                inline for (0..columns) |col| {
                    var acc = loadVec(R, width, dest[col], offset);
                    acc = @mulAdd(V, xv, @as(V, @splat(cx[col])), acc);
                    acc = @mulAdd(V, yv, @as(V, @splat(cy[col])), acc);
                    storeVec(R, width, dest[col], offset, acc);
                }
            }
        }
    }
    while (i + width <= n * components) : (i += width) {
        const xv = loadVec(R, width, xp, i);
        const yv = loadVec(R, width, yp, i);
        inline for (0..columns) |col| {
            const dp: [*]R = @ptrCast(dest[col]);
            var acc = loadVec(R, width, dp, i);
            if (comptime isReal(T)) {
                acc = @mulAdd(V, xv, @as(V, @splat(cx[col])), acc);
                acc = @mulAdd(V, yv, @as(V, @splat(cy[col])), acc);
            } else {
                const swap: @Vector(width, i32) = comptime blk: {
                    var mask: @Vector(width, i32) = undefined;
                    for (0..width) |lane| mask[lane] = @intCast(lane ^ 1);
                    break :blk mask;
                };
                var xi: V = @splat(cx[col].im);
                var yi: V = @splat(cy[col].im);
                inline for (0..width / 2) |pair| {
                    xi[2 * pair] = -cx[col].im;
                    yi[2 * pair] = -cy[col].im;
                }
                acc = @mulAdd(V, xv, @as(V, @splat(cx[col].re)), acc);
                acc = @mulAdd(V, @shuffle(R, xv, undefined, swap), xi, acc);
                acc = @mulAdd(V, yv, @as(V, @splat(cy[col].re)), acc);
                acc = @mulAdd(V, @shuffle(R, yv, undefined, swap), yi, acc);
            }
            storeVec(R, width, dp, i, acc);
        }
    }
    if (comptime T == f32) {
        const half = width / 2;
        const H = @Vector(half, R);
        if (i + half <= n) {
            const xv = loadVec(R, half, xp, i);
            const yv = loadVec(R, half, yp, i);
            inline for (0..columns) |col| {
                const dp: [*]R = @ptrCast(dest[col]);
                var acc = loadVec(R, half, dp, i);
                acc = @mulAdd(H, xv, @as(H, @splat(cx[col])), acc);
                acc = @mulAdd(H, yv, @as(H, @splat(cy[col])), acc);
                storeVec(R, half, dp, i, acc);
            }
            i += half;
        }
    }
    var row = i / components;
    while (row < n) : (row += 1) {
        inline for (0..columns) |col| {
            dest[col][row] = if (comptime isReal(T))
                @mulAdd(T, cy[col], y[row], @mulAdd(T, cx[col], x[row], dest[col][row]))
            else
                add(T, add(T, dest[col][row], mul(T, cx[col], x[row])), mul(T, cy[col], y[row]));
        }
    }
}

fn tiledRankTwoColumns(comptime T: type, comptime is_packed: bool, uplo: Uplo, n: usize, j0: usize, j1: usize, alpha: T, x: [*]const T, y: [*]const T, a: [*]T, lda: BlasInt) usize {
    if (comptime isReal(T) and (!is_packed or T == f32)) {
        if (n <= 256) return tiledRankTwoColumnsImpl(T, is_packed, 2, uplo, n, j0, j1, alpha, x, y, a, lda);
    }
    return tiledRankTwoColumnsImpl(T, is_packed, if (isReal(T)) 4 else 2, uplo, n, j0, j1, alpha, x, y, a, lda);
}

fn tiledRankTwoColumnsImpl(comptime T: type, comptime is_packed: bool, comptime columns: usize, uplo: Uplo, n: usize, j0: usize, j1: usize, alpha: T, x: [*]const T, y: [*]const T, a: [*]T, lda: BlasInt) usize {
    var j = j0;
    while (j + columns <= j1) : (j += columns) {
        const row0 = if (uplo == .upper) 0 else j + columns - 1;
        const count = if (uplo == .upper) j + 1 else n - row0;
        var cx: [columns]T = undefined;
        var cy: [columns]T = undefined;
        var dest: [columns][*]T = undefined;
        inline for (0..columns) |col| {
            const jj = j + col;
            cx[col] = mul(T, alpha, conj(T, y[jj]));
            cy[col] = mul(T, conj(T, alpha), conj(T, x[jj]));
            dest[col] = a + if (is_packed) packedIndex(uplo, n, row0, jj) else matIndex(lda, row0, jj);
        }
        rankTwoPanel(T, columns, is_packed, count, cx, x + row0, cy, y + row0, dest);
        inline for (0..columns) |col| {
            const jj = j + col;
            const fringe_row = if (uplo == .upper) j + 1 else jj;
            const fringe_count = if (uplo == .upper) col else columns - 1 - col;
            if (fringe_count > 0) {
                const fringe = a + if (is_packed) packedIndex(uplo, n, fringe_row, jj) else matIndex(lda, fringe_row, jj);
                rankTwoColumn(T, fringe_count, cx[col], x + fringe_row, cy[col], y + fringe_row, fringe);
            }
            if (comptime !isReal(T)) {
                const diag = if (is_packed) packedIndex(uplo, n, jj, jj) else matIndex(lda, jj, jj);
                a[diag].im = 0;
            }
        }
    }
    return j;
}

fn denseRankColumns(comptime T: type, comptime operation: DenseRankOperation, uplo: Uplo, n: usize, j0: usize, j1: usize, alpha: T, x: [*]const T, y: [*]const T, a: [*]T, lda: BlasInt) void {
    const first = if (comptime builtin.cpu.arch == .x86_64 and (operation == .syr2 or operation == .her2))
        tiledRankTwoColumns(T, false, uplo, n, j0, j1, alpha, x, y, a, lda)
    else
        j0;
    for (first..j1) |j| {
        const row0: usize = if (uplo == .upper) 0 else j;
        const row1: usize = if (uplo == .upper) j + 1 else n;
        const count = row1 - row0;
        const column = a + matIndex(lda, row0, j);

        switch (operation) {
            .syr => {
                const temp = alpha * x[j];
                if (temp != 0) vector_ops.axpyUnitReal(T, count, temp, x + row0, column);
            },
            .her => {
                const temp = mul(T, alpha, conj(T, x[j]));
                if (!isZero(T, temp)) vector_ops.axpy(T, @intCast(count), temp, x + row0, 1, column, 1);
                a[matIndex(lda, j, j)].im = 0;
            },
            .syr2 => {
                const temp1 = alpha * y[j];
                const temp2 = alpha * x[j];
                rankTwoColumn(T, count, temp1, x + row0, temp2, y + row0, column);
            },
            .her2 => {
                const temp1 = mul(T, alpha, conj(T, y[j]));
                const temp2 = mul(T, conj(T, alpha), conj(T, x[j]));
                rankTwoColumn(T, count, temp1, x + row0, temp2, y + row0, column);
                a[matIndex(lda, j, j)].im = 0;
            },
        }
    }
}

fn DenseRankTask(comptime T: type) type {
    return struct {
        uplo: Uplo,
        n: usize,
        j0: usize,
        j1: usize,
        alpha: T,
        x: [*]const T,
        y: [*]const T,
        a: [*]T,
        lda: BlasInt,
    };
}

fn runDenseRankTask(comptime T: type, comptime operation: DenseRankOperation, raw_tasks: *const anyopaque, index: usize) void {
    const tasks: [*]const DenseRankTask(T) = @ptrCast(@alignCast(raw_tasks));
    const task = tasks[index];
    denseRankColumns(T, operation, task.uplo, task.n, task.j0, task.j1, task.alpha, task.x, task.y, task.a, task.lda);
}

fn runSyrTaskF32(raw_tasks: *const anyopaque, index: usize) void {
    runDenseRankTask(f32, .syr, raw_tasks, index);
}

fn runSyrTaskF64(raw_tasks: *const anyopaque, index: usize) void {
    runDenseRankTask(f64, .syr, raw_tasks, index);
}

fn runHerTaskC32(raw_tasks: *const anyopaque, index: usize) void {
    runDenseRankTask(scalar.ComplexF32, .her, raw_tasks, index);
}

fn runHerTaskC64(raw_tasks: *const anyopaque, index: usize) void {
    runDenseRankTask(scalar.ComplexF64, .her, raw_tasks, index);
}

fn runSyr2TaskF32(raw_tasks: *const anyopaque, index: usize) void {
    runDenseRankTask(f32, .syr2, raw_tasks, index);
}

fn runSyr2TaskF64(raw_tasks: *const anyopaque, index: usize) void {
    runDenseRankTask(f64, .syr2, raw_tasks, index);
}

fn runHer2TaskC32(raw_tasks: *const anyopaque, index: usize) void {
    runDenseRankTask(scalar.ComplexF32, .her2, raw_tasks, index);
}

fn runHer2TaskC64(raw_tasks: *const anyopaque, index: usize) void {
    runDenseRankTask(scalar.ComplexF64, .her2, raw_tasks, index);
}

fn denseRankTaskBoundary(uplo: Uplo, n: usize, task_count: usize, task_index: usize) usize {
    if (task_index == 0) return 0;
    if (task_index >= task_count) return n;
    const fraction = @as(f64, @floatFromInt(task_index)) / @as(f64, @floatFromInt(task_count));
    const boundary = if (uplo == .upper)
        @sqrt(fraction) * @as(f64, @floatFromInt(n))
    else
        (1.0 - @sqrt(1.0 - fraction)) * @as(f64, @floatFromInt(n));
    return @min(n, @as(usize, @intFromFloat(boundary)));
}

fn parallelDenseRankUpdate(comptime T: type, comptime operation: DenseRankOperation, uplo: Uplo, n: usize, alpha: T, x: [*]const T, y: [*]const T, a: [*]T, lda: BlasInt) bool {
    if (!tuning.preferDenseStructuredParallel(n)) return false;

    var task_count = core_pool.taskCount(n, tuning.structuredMinColumns());
    if (comptime builtin.cpu.arch == .x86_64) {
        task_count = level2_tuning.capTaskCountByWork(task_count, n *| n, 64 * 1024);
    }
    task_count = rankUpdateTaskLimit(T, operation, false, n, tuning.capStructuredTasks(n, task_count));
    if (task_count <= 1) return false;

    var tasks: [core_pool.max_tasks]DenseRankTask(T) = undefined;
    for (0..task_count) |task_index| {
        tasks[task_index] = .{
            .uplo = uplo,
            .n = n,
            .j0 = denseRankTaskBoundary(uplo, n, task_count, task_index),
            .j1 = denseRankTaskBoundary(uplo, n, task_count, task_index + 1),
            .alpha = alpha,
            .x = x,
            .y = y,
            .a = a,
            .lda = lda,
        };
    }

    const runner: core_pool.TaskFn = switch (operation) {
        .syr => if (T == f32) runSyrTaskF32 else runSyrTaskF64,
        .her => if (T == scalar.ComplexF32) runHerTaskC32 else runHerTaskC64,
        .syr2 => if (T == f32) runSyr2TaskF32 else runSyr2TaskF64,
        .her2 => if (T == scalar.ComplexF32) runHer2TaskC32 else runHer2TaskC64,
    };
    if (comptime builtin.cpu.arch == .x86_64) return core_pool.runLowLatency(runner, @ptrCast(&tasks), task_count);
    return core_pool.run(runner, @ptrCast(&tasks), task_count);
}

fn packedRankColumns(comptime T: type, comptime operation: DenseRankOperation, uplo: Uplo, n: usize, j0: usize, j1: usize, alpha: T, x: [*]const T, y: [*]const T, ap: [*]T) void {
    const first = if (comptime builtin.cpu.arch == .x86_64 and (operation == .syr2 or operation == .her2))
        tiledRankTwoColumns(T, true, uplo, n, j0, j1, alpha, x, y, ap, 0)
    else
        j0;
    for (first..j1) |j| {
        const row0: usize = if (uplo == .upper) 0 else j;
        const count: usize = if (uplo == .upper) j + 1 else n - j;
        const segment = ap + packedIndex(uplo, n, row0, j);

        switch (operation) {
            .syr => {
                const temp = alpha * x[j];
                if (temp != 0) vector_ops.axpyUnitReal(T, count, temp, x + row0, segment);
            },
            .her => {
                const temp = mul(T, alpha, conj(T, x[j]));
                if (!isZero(T, temp)) vector_ops.axpy(T, @intCast(count), temp, x + row0, 1, segment, 1);
                ap[packedIndex(uplo, n, j, j)].im = 0;
            },
            .syr2 => {
                const temp1 = alpha * y[j];
                const temp2 = alpha * x[j];
                rankTwoColumn(T, count, temp1, x + row0, temp2, y + row0, segment);
            },
            .her2 => {
                const temp1 = mul(T, alpha, conj(T, y[j]));
                const temp2 = mul(T, conj(T, alpha), conj(T, x[j]));
                rankTwoColumn(T, count, temp1, x + row0, temp2, y + row0, segment);
                ap[packedIndex(uplo, n, j, j)].im = 0;
            },
        }
    }
}

fn PackedRankTask(comptime T: type) type {
    return struct {
        uplo: Uplo,
        n: usize,
        j0: usize,
        j1: usize,
        alpha: T,
        x: [*]const T,
        y: [*]const T,
        ap: [*]T,
    };
}

fn runPackedRankTask(comptime T: type, comptime operation: DenseRankOperation, raw_tasks: *const anyopaque, index: usize) void {
    const tasks: [*]const PackedRankTask(T) = @ptrCast(@alignCast(raw_tasks));
    const task = tasks[index];
    packedRankColumns(T, operation, task.uplo, task.n, task.j0, task.j1, task.alpha, task.x, task.y, task.ap);
}

fn runPackedSyrTaskF32(raw_tasks: *const anyopaque, index: usize) void {
    runPackedRankTask(f32, .syr, raw_tasks, index);
}

fn runPackedSyrTaskF64(raw_tasks: *const anyopaque, index: usize) void {
    runPackedRankTask(f64, .syr, raw_tasks, index);
}

fn runPackedHerTaskC32(raw_tasks: *const anyopaque, index: usize) void {
    runPackedRankTask(scalar.ComplexF32, .her, raw_tasks, index);
}

fn runPackedHerTaskC64(raw_tasks: *const anyopaque, index: usize) void {
    runPackedRankTask(scalar.ComplexF64, .her, raw_tasks, index);
}

fn runPackedSyr2TaskF32(raw_tasks: *const anyopaque, index: usize) void {
    runPackedRankTask(f32, .syr2, raw_tasks, index);
}

fn runPackedSyr2TaskF64(raw_tasks: *const anyopaque, index: usize) void {
    runPackedRankTask(f64, .syr2, raw_tasks, index);
}

fn runPackedHer2TaskC32(raw_tasks: *const anyopaque, index: usize) void {
    runPackedRankTask(scalar.ComplexF32, .her2, raw_tasks, index);
}

fn runPackedHer2TaskC64(raw_tasks: *const anyopaque, index: usize) void {
    runPackedRankTask(scalar.ComplexF64, .her2, raw_tasks, index);
}

fn packedRankElementCount(n: usize) usize {
    const next = n +| 1;
    return if (n % 2 == 0) (n / 2) *| next else n *| (next / 2);
}

noinline fn parallelPackedRankUpdate(comptime T: type, comptime operation: DenseRankOperation, uplo: Uplo, n: usize, alpha: T, x: [*]const T, y: [*]const T, ap: [*]T) bool {
    if (!tuning.preferPackedTaskComposition(n)) return false;

    var task_count = core_pool.taskCount(n, tuning.structuredMinColumns());
    if (comptime builtin.cpu.arch == .x86_64) {
        task_count = level2_tuning.capTaskCountByWork(task_count, packedRankElementCount(n), 64 * 1024);
    }
    task_count = rankUpdateTaskLimit(T, operation, true, n, tuning.capStructuredTasks(n, task_count));
    if (task_count <= 1) return false;

    var tasks: [core_pool.max_tasks]PackedRankTask(T) = undefined;
    for (0..task_count) |task_index| {
        tasks[task_index] = .{
            .uplo = uplo,
            .n = n,
            .j0 = denseRankTaskBoundary(uplo, n, task_count, task_index),
            .j1 = denseRankTaskBoundary(uplo, n, task_count, task_index + 1),
            .alpha = alpha,
            .x = x,
            .y = y,
            .ap = ap,
        };
    }

    const runner: core_pool.TaskFn = switch (operation) {
        .syr => if (T == f32) runPackedSyrTaskF32 else runPackedSyrTaskF64,
        .her => if (T == scalar.ComplexF32) runPackedHerTaskC32 else runPackedHerTaskC64,
        .syr2 => if (T == f32) runPackedSyr2TaskF32 else runPackedSyr2TaskF64,
        .her2 => if (T == scalar.ComplexF32) runPackedHer2TaskC32 else runPackedHer2TaskC64,
    };
    if (comptime builtin.cpu.arch == .x86_64) return core_pool.runLowLatency(runner, @ptrCast(&tasks), task_count);
    return core_pool.run(runner, @ptrCast(&tasks), task_count);
}

pub fn syr(comptime T: type, uplo: Uplo, n_: BlasInt, alpha: T, x: [*]const T, incx_: BlasInt, a: [*]T, lda: BlasInt) void {
    if (n_ <= 0 or incx_ == 0 or isZero(T, alpha)) return;
    const n = toUsize(n_);
    if (comptime isReal(T)) {
        if (level2_tuning.coreUnitSelected(T, .syr, .{ .m = n, .n = n, .incx = incx_ })) {
            if (parallelDenseRankUpdate(T, .syr, uplo, n, alpha, x, x, a, lda)) return;
            return denseRankColumns(T, .syr, uplo, n, 0, n, alpha, x, x, a, lda);
        }
    }
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
    if (comptime isReal(T)) {
        if (level2_tuning.coreUnitSelected(T, .spr, .{ .m = n, .n = n, .incx = incx_ }) and
            preferPackedColumns(n))
        {
            if (tuning.preferPackedStructuredParallel(n) and parallelPackedRankUpdate(T, .syr, uplo, n, alpha, x, x, ap)) return;
            return packedRankColumns(T, .syr, uplo, n, 0, n, alpha, x, x, ap);
        }
    }
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
    if (comptime isReal(T)) {
        if (level2_tuning.coreUnitSelected(T, .syr2, .{ .m = n, .n = n, .incx = incx_, .incy = incy_ })) {
            if (parallelDenseRankUpdate(T, .syr2, uplo, n, alpha, x, y, a, lda)) return;
            return denseRankColumns(T, .syr2, uplo, n, 0, n, alpha, x, y, a, lda);
        }
    }
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
    if (comptime isReal(T)) {
        if (level2_tuning.coreUnitSelected(T, .spr2, .{ .m = n, .n = n, .incx = incx_, .incy = incy_ }) and
            preferPackedColumns(n))
        {
            if (tuning.preferPackedStructuredParallel(n) and parallelPackedRankUpdate(T, .syr2, uplo, n, alpha, x, y, ap)) return;
            return packedRankColumns(T, .syr2, uplo, n, 0, n, alpha, x, y, ap);
        }
    }
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
    if (comptime isComplex(T)) {
        if (level2_tuning.coreUnitSelected(T, .her, .{ .m = n, .n = n, .incx = incx_ })) {
            const complex_alpha = realScalar(T, alpha);
            if (parallelDenseRankUpdate(T, .her, uplo, n, complex_alpha, x, x, a, lda)) return;
            return denseRankColumns(T, .her, uplo, n, 0, n, complex_alpha, x, x, a, lda);
        }
    }
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
    if (comptime isComplex(T)) {
        if (level2_tuning.coreUnitSelected(T, .hpr, .{ .m = n, .n = n, .incx = incx_ }) and
            preferPackedColumns(n))
        {
            const complex_alpha = realScalar(T, alpha);
            if (tuning.preferPackedStructuredParallel(n) and parallelPackedRankUpdate(T, .her, uplo, n, complex_alpha, x, x, ap)) return;
            return packedRankColumns(T, .her, uplo, n, 0, n, complex_alpha, x, x, ap);
        }
    }
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
    if (comptime isComplex(T)) {
        if (level2_tuning.coreUnitSelected(T, .her2, .{ .m = n, .n = n, .incx = incx_, .incy = incy_ })) {
            if (parallelDenseRankUpdate(T, .her2, uplo, n, alpha, x, y, a, lda)) return;
            return denseRankColumns(T, .her2, uplo, n, 0, n, alpha, x, y, a, lda);
        }
    }
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
    if (comptime isComplex(T)) {
        if (level2_tuning.coreUnitSelected(T, .hpr2, .{ .m = n, .n = n, .incx = incx_, .incy = incy_ }) and
            preferPackedColumns(n))
        {
            if (tuning.preferPackedStructuredParallel(n) and parallelPackedRankUpdate(T, .her2, uplo, n, alpha, x, y, ap)) return;
            return packedRankColumns(T, .her2, uplo, n, 0, n, alpha, x, y, ap);
        }
    }
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

pub const testing = struct {
    pub fn gerColumns(
        comptime T: type,
        m: usize,
        n: usize,
        alpha: T,
        x: [*]const T,
        y: [*]const T,
        a: [*]T,
        lda: BlasInt,
        conjugate_y: bool,
    ) void {
        if (comptime isReal(T)) return gerUnitReal(T, m, n, alpha, x, y, a, lda);
        return gerUnitComplex(T, @intCast(m), n, alpha, x, y, a, lda, conjugate_y);
    }
};

fn packedRankTestValue(comptime T: type, index: usize, phase: usize) T {
    const re = @as(f64, @floatFromInt((index * 17 + phase * 11) % 37)) / 19.0 - 0.75;
    if (T == f32 or T == f64) return @floatCast(re);
    const im = @as(f64, @floatFromInt((index * 13 + phase * 7) % 29)) / 23.0 - 0.5;
    return .{ .re = @floatCast(re), .im = @floatCast(im) };
}

fn expectPackedRankApprox(comptime T: type, expected: []const T, actual: []const T) !void {
    try std.testing.expectEqual(expected.len, actual.len);
    if (comptime isReal(T)) {
        const tolerance: T = if (T == f32) 2e-5 else 2e-12;
        for (expected, actual) |want, got| try std.testing.expectApproxEqAbs(want, got, tolerance);
    } else {
        const Component = if (T == scalar.ComplexF32) f32 else f64;
        const tolerance: Component = if (T == scalar.ComplexF32) 3e-5 else 3e-12;
        for (expected, actual) |want, got| {
            try std.testing.expectApproxEqAbs(want.re, got.re, tolerance);
            try std.testing.expectApproxEqAbs(want.im, got.im, tolerance);
        }
    }
}

fn expectParallelPackedRankMatchesSingle(comptime T: type, comptime operation: DenseRankOperation, uplo: Uplo) !void {
    const n: usize = 512;
    const packed_len = packedRankElementCount(n);
    const allocator = std.testing.allocator;
    const x = try allocator.alloc(T, n);
    defer allocator.free(x);
    const y = try allocator.alloc(T, n);
    defer allocator.free(y);
    const expected = try allocator.alloc(T, packed_len);
    defer allocator.free(expected);
    const actual = try allocator.alloc(T, packed_len);
    defer allocator.free(actual);

    for (x, 0..) |*value, i| value.* = packedRankTestValue(T, i, 1);
    for (y, 0..) |*value, i| value.* = packedRankTestValue(T, i, 2);
    for (expected, actual, 0..) |*expected_value, *actual_value, i| {
        const value = packedRankTestValue(T, i, 3);
        expected_value.* = value;
        actual_value.* = value;
    }

    const alpha = if (operation == .her) blk: {
        const value = packedRankTestValue(T, 5, 4);
        break :blk realScalar(T, value.re);
    } else packedRankTestValue(T, 5, 4);
    packedRankColumns(T, operation, uplo, n, 0, n, alpha, x.ptr, y.ptr, expected.ptr);
    try std.testing.expect(parallelPackedRankUpdate(T, operation, uplo, n, alpha, x.ptr, y.ptr, actual.ptr));
    try expectPackedRankApprox(T, expected, actual);

    for (actual, 0..) |*value, i| value.* = packedRankTestValue(T, i, 3);
    runtime.setMaxThreads(1);
    switch (operation) {
        .syr => spr(T, uplo, @intCast(n), alpha, x.ptr, 1, actual.ptr),
        .her => hpr(T, uplo, @intCast(n), alpha.re, x.ptr, 1, actual.ptr),
        .syr2 => spr2(T, uplo, @intCast(n), alpha, x.ptr, 1, y.ptr, 1, actual.ptr),
        .her2 => hpr2(T, uplo, @intCast(n), alpha, x.ptr, 1, y.ptr, 1, actual.ptr),
    }
    runtime.setMaxThreads(4);
    try expectPackedRankApprox(T, expected, actual);
}

test "packed rank update parallel columns match the single-task body" {
    runtime.setMaxThreads(4);
    defer {
        runtime.setMaxThreads(0);
        core_pool.shutdown();
    }
    if (runtime.maxThreads() <= 1) return error.SkipZigTest;

    inline for (.{ Uplo.upper, Uplo.lower }) |uplo| {
        try expectParallelPackedRankMatchesSingle(f32, .syr, uplo);
        try expectParallelPackedRankMatchesSingle(f64, .syr, uplo);
        try expectParallelPackedRankMatchesSingle(scalar.ComplexF32, .her, uplo);
        try expectParallelPackedRankMatchesSingle(scalar.ComplexF64, .her, uplo);
        try expectParallelPackedRankMatchesSingle(f32, .syr2, uplo);
        try expectParallelPackedRankMatchesSingle(f64, .syr2, uplo);
        try expectParallelPackedRankMatchesSingle(scalar.ComplexF32, .her2, uplo);
        try expectParallelPackedRankMatchesSingle(scalar.ComplexF64, .her2, uplo);
    }
}

test "fused rank-two columns preserve tails guards and zero coefficients" {
    inline for (.{ f32, f64, scalar.ComplexF32, scalar.ComplexF64 }) |T| {
        var x: [34]T = undefined;
        var y: [34]T = undefined;
        for (&x, &y, 0..) |*xv, *yv, i| {
            xv.* = packedRankTestValue(T, i, 1);
            yv.* = packedRankTestValue(T, i, 2);
        }
        for ([_]usize{ 0, 1, 2, 3, 7, 8, 9, 16, 17, 31, 33 }) |n| {
            for (0..4) |coefficient_case| {
                const cx = if (coefficient_case & 1 == 0) packedRankTestValue(T, 2, 4) else scalar.zero(T);
                const cy = if (coefficient_case & 2 == 0) packedRankTestValue(T, 3, 5) else scalar.zero(T);
                var expected: [34]T = undefined;
                for (&expected, 0..) |*v, i| v.* = packedRankTestValue(T, i, 3);
                var actual = expected;
                for (0..n) |i| expected[i] = add(T, add(T, expected[i], mul(T, cx, x[i])), mul(T, cy, y[i]));
                rankTwoColumn(T, n, cx, &x, cy, &y, &actual);
                try expectPackedRankApprox(T, &expected, &actual);
            }
        }
    }
}
