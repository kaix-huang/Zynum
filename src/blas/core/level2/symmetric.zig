// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later

//! Symmetric, Hermitian, banded, and packed matrix-vector BLAS Level 2 kernels.

const std = @import("std");

const scalar = @import("../scalar.zig");
const indexing = @import("../indexing.zig");
const access = @import("access.zig");
const core_pool = @import("../pool.zig");

const BlasInt = scalar.BlasInt;
const Uplo = scalar.Uplo;

const zero = scalar.zero;
const add = scalar.add;
const mul = scalar.mul;
const conj = scalar.conj;
const isComplex = scalar.isComplex;
const isZero = scalar.isZero;

const toUsize = indexing.toUsize;
const startIndex = indexing.startIndex;
const ix = indexing.ix;
const symBandIndex = indexing.symBandIndex;
const vectorGet = indexing.vectorGet;

const symValue = access.symValue;
const symPackedValue = access.symPackedValue;

threadlocal var symv_workspace_f32_ptr: ?[*]f32 = null;
threadlocal var symv_workspace_f32_len: usize = 0;
threadlocal var symv_workspace_f64_ptr: ?[*]f64 = null;
threadlocal var symv_workspace_f64_len: usize = 0;

fn symvWorkspace(comptime T: type, len: usize) ?[]T {
    if (T == f32) {
        if (symv_workspace_f32_len < len) {
            const data = std.heap.c_allocator.alloc(f32, len) catch return null;
            if (symv_workspace_f32_ptr) |old| std.heap.c_allocator.free(old[0..symv_workspace_f32_len]);
            symv_workspace_f32_ptr = data.ptr;
            symv_workspace_f32_len = len;
        }
        return symv_workspace_f32_ptr.?[0..len];
    }
    if (T == f64) {
        if (symv_workspace_f64_len < len) {
            const data = std.heap.c_allocator.alloc(f64, len) catch return null;
            if (symv_workspace_f64_ptr) |old| std.heap.c_allocator.free(old[0..symv_workspace_f64_len]);
            symv_workspace_f64_ptr = data.ptr;
            symv_workspace_f64_len = len;
        }
        return symv_workspace_f64_ptr.?[0..len];
    }
    return null;
}

fn isReal(comptime T: type) bool {
    return T == f32 or T == f64;
}

fn lanes(comptime T: type) comptime_int {
    if (T == f32) return 8;
    if (T == f64) return 4;
    @compileError("real SYMV vector lanes support f32 and f64");
}

fn unroll(comptime T: type) comptime_int {
    return 4 * lanes(T);
}

inline fn loadVec(comptime T: type, comptime lane_count: comptime_int, ptr: [*]const T, index: usize) @Vector(lane_count, T) {
    const V = @Vector(lane_count, T);
    return @as(*align(1) const V, @ptrCast(ptr + index)).*;
}

inline fn storeVec(comptime T: type, comptime lane_count: comptime_int, ptr: [*]T, index: usize, value: @Vector(lane_count, T)) void {
    const V = @Vector(lane_count, T);
    @as(*align(1) V, @ptrCast(ptr + index)).* = value;
}

fn scaleUnitReal(comptime T: type, n: usize, beta: T, y: [*]T) void {
    if (beta == 1) return;
    if (beta == 0) {
        @memset(y[0..n], 0);
        return;
    }
    const lane_count = lanes(T);
    const V = @Vector(lane_count, T);
    const beta_v: V = @splat(beta);
    var i: usize = 0;
    while (i + lane_count <= n) : (i += lane_count) {
        storeVec(T, lane_count, y, i, loadVec(T, lane_count, y, i) * beta_v);
    }
    while (i < n) : (i += 1) y[i] *= beta;
}

fn addUnitReal(comptime T: type, n: usize, x: [*]const T, y: [*]T) void {
    const lane_count = lanes(T);
    var i: usize = 0;
    while (i + lane_count <= n) : (i += lane_count) {
        storeVec(T, lane_count, y, i, loadVec(T, lane_count, y, i) + loadVec(T, lane_count, x, i));
    }
    while (i < n) : (i += 1) y[i] += x[i];
}

fn symvColumnsUnitReal(comptime T: type, uplo: Uplo, n: usize, j0: usize, j1: usize, alpha: T, a: [*]const T, lda: BlasInt, x: [*]const T, y: [*]T) void {
    const lane_count = lanes(T);
    const unroll_count = unroll(T);
    const V = @Vector(lane_count, T);
    if (uplo == .upper) {
        for (j0..j1) |j| {
            const col = a + indexing.matIndex(lda, 0, j);
            const temp1 = alpha * x[j];
            const temp1_v: V = @splat(temp1);
            var temp20_v: V = @splat(0);
            var temp21_v: V = @splat(0);
            var temp22_v: V = @splat(0);
            var temp23_v: V = @splat(0);
            var i: usize = 0;
            while (i + unroll_count <= j) : (i += unroll_count) {
                inline for (0..4) |k| {
                    const offset = i + k * lane_count;
                    const av = loadVec(T, lane_count, col, offset);
                    const xv = loadVec(T, lane_count, x, offset);
                    storeVec(T, lane_count, y, offset, @mulAdd(V, av, temp1_v, loadVec(T, lane_count, y, offset)));
                    switch (k) {
                        0 => temp20_v = @mulAdd(V, av, xv, temp20_v),
                        1 => temp21_v = @mulAdd(V, av, xv, temp21_v),
                        2 => temp22_v = @mulAdd(V, av, xv, temp22_v),
                        3 => temp23_v = @mulAdd(V, av, xv, temp23_v),
                        else => unreachable,
                    }
                }
            }
            var temp2_v = temp20_v + temp21_v + temp22_v + temp23_v;
            while (i + lane_count <= j) : (i += lane_count) {
                const av = loadVec(T, lane_count, col, i);
                const xv = loadVec(T, lane_count, x, i);
                storeVec(T, lane_count, y, i, @mulAdd(V, av, temp1_v, loadVec(T, lane_count, y, i)));
                temp2_v = @mulAdd(V, av, xv, temp2_v);
            }
            var temp2: T = @reduce(.Add, temp2_v);
            while (i < j) : (i += 1) {
                y[i] = @mulAdd(T, temp1, col[i], y[i]);
                temp2 = @mulAdd(T, col[i], x[i], temp2);
            }
            y[j] = @mulAdd(T, temp1, col[j], y[j] + alpha * temp2);
        }
    } else {
        for (j0..j1) |j| {
            const col = a + indexing.matIndex(lda, 0, j);
            const temp1 = alpha * x[j];
            const temp1_v: V = @splat(temp1);
            y[j] = @mulAdd(T, temp1, col[j], y[j]);
            var temp20_v: V = @splat(0);
            var temp21_v: V = @splat(0);
            var temp22_v: V = @splat(0);
            var temp23_v: V = @splat(0);
            var i = j + 1;
            while (i + unroll_count <= n) : (i += unroll_count) {
                inline for (0..4) |k| {
                    const offset = i + k * lane_count;
                    const av = loadVec(T, lane_count, col, offset);
                    const xv = loadVec(T, lane_count, x, offset);
                    storeVec(T, lane_count, y, offset, @mulAdd(V, av, temp1_v, loadVec(T, lane_count, y, offset)));
                    switch (k) {
                        0 => temp20_v = @mulAdd(V, av, xv, temp20_v),
                        1 => temp21_v = @mulAdd(V, av, xv, temp21_v),
                        2 => temp22_v = @mulAdd(V, av, xv, temp22_v),
                        3 => temp23_v = @mulAdd(V, av, xv, temp23_v),
                        else => unreachable,
                    }
                }
            }
            var temp2_v = temp20_v + temp21_v + temp22_v + temp23_v;
            while (i + lane_count <= n) : (i += lane_count) {
                const av = loadVec(T, lane_count, col, i);
                const xv = loadVec(T, lane_count, x, i);
                storeVec(T, lane_count, y, i, @mulAdd(V, av, temp1_v, loadVec(T, lane_count, y, i)));
                temp2_v = @mulAdd(V, av, xv, temp2_v);
            }
            var temp2: T = @reduce(.Add, temp2_v);
            while (i < n) : (i += 1) {
                y[i] = @mulAdd(T, temp1, col[i], y[i]);
                temp2 = @mulAdd(T, col[i], x[i], temp2);
            }
            y[j] += alpha * temp2;
        }
    }
}

fn SymvTask(comptime T: type) type {
    return struct {
        uplo: Uplo,
        n: usize,
        j0: usize,
        j1: usize,
        alpha: T,
        a: [*]const T,
        lda: BlasInt,
        x: [*]const T,
        y_delta: [*]T,
    };
}

fn runSymvTask(comptime T: type, raw_tasks: *const anyopaque, index: usize) void {
    const tasks: [*]const SymvTask(T) = @ptrCast(@alignCast(raw_tasks));
    const task = tasks[index];
    symvColumnsUnitReal(T, task.uplo, task.n, task.j0, task.j1, task.alpha, task.a, task.lda, task.x, task.y_delta);
}

fn runSymvTaskF32(raw_tasks: *const anyopaque, index: usize) void {
    runSymvTask(f32, raw_tasks, index);
}

fn runSymvTaskF64(raw_tasks: *const anyopaque, index: usize) void {
    runSymvTask(f64, raw_tasks, index);
}

fn parallelSymvUnitReal(comptime T: type, uplo: Uplo, n: usize, alpha: T, a: [*]const T, lda: BlasInt, x: [*]const T, y: [*]T) bool {
    if (n *| n < 768 * 768) return false;
    var task_count = core_pool.taskCount(n, 128);
    if (n <= 1536) task_count = @min(task_count, 6);
    if (task_count <= 1) return false;

    const workspace_len = task_count * n;
    if (workspace_len * @sizeOf(T) > 64 * 1024 * 1024) return false;
    const workspace = symvWorkspace(T, workspace_len) orelse return false;
    @memset(workspace, 0);

    var tasks: [core_pool.max_tasks]SymvTask(T) = undefined;
    for (0..task_count) |task_index| {
        const j0 = task_index * n / task_count;
        const j1 = (task_index + 1) * n / task_count;
        tasks[task_index] = .{
            .uplo = uplo,
            .n = n,
            .j0 = j0,
            .j1 = j1,
            .alpha = alpha,
            .a = a,
            .lda = lda,
            .x = x,
            .y_delta = workspace.ptr + task_index * n,
        };
    }

    const runner = if (T == f32) runSymvTaskF32 else runSymvTaskF64;
    if (!core_pool.run(runner, @ptrCast(&tasks), task_count)) return false;

    for (0..task_count) |task_index| {
        addUnitReal(T, n, workspace.ptr + task_index * n, y);
    }
    return true;
}

fn symvUnitReal(comptime T: type, uplo: Uplo, n: usize, alpha: T, a: [*]const T, lda: BlasInt, x: [*]const T, beta: T, y: [*]T) void {
    scaleUnitReal(T, n, beta, y);
    if (alpha == 0) return;
    if (parallelSymvUnitReal(T, uplo, n, alpha, a, lda, x, y)) return;
    symvColumnsUnitReal(T, uplo, n, 0, n, alpha, a, lda, x, y);
}

pub fn symv(comptime T: type, uplo: Uplo, n_: BlasInt, alpha: T, a: [*]const T, lda: BlasInt, x: [*]const T, incx_: BlasInt, beta: T, y: [*]T, incy_: BlasInt, herm: bool) void {
    if (n_ <= 0 or incx_ == 0 or incy_ == 0) return;
    const n = toUsize(n_);
    if (comptime isReal(T)) {
        if (!herm and incx_ == 1 and incy_ == 1) return symvUnitReal(T, uplo, n, alpha, a, lda, x, beta, y);
    }
    const sx = startIndex(n_, incx_);
    const sy = startIndex(n_, incy_);
    for (0..n) |i| {
        const py = ix(sy, i, incy_);
        y[py] = if (isZero(T, beta)) zero(T) else mul(T, beta, y[py]);
    }
    if (isZero(T, alpha)) return;
    for (0..n) |i| {
        var sum = zero(T);
        for (0..n) |j| sum = add(T, sum, mul(T, symValue(T, uplo, a, lda, i, j, herm), vectorGet(T, x, sx, j, incx_)));
        const py = ix(sy, i, incy_);
        y[py] = add(T, y[py], mul(T, alpha, sum));
    }
}

pub fn sbmv(comptime T: type, uplo: Uplo, n_: BlasInt, k_: BlasInt, alpha: T, a: [*]const T, lda: BlasInt, x: [*]const T, incx_: BlasInt, beta: T, y: [*]T, incy_: BlasInt, herm: bool) void {
    if (n_ <= 0 or incx_ == 0 or incy_ == 0) return;
    const n = toUsize(n_);
    const k = toUsize(k_);
    const sx = startIndex(n_, incx_);
    const sy = startIndex(n_, incy_);
    for (0..n) |i| {
        const py = ix(sy, i, incy_);
        y[py] = if (isZero(T, beta)) zero(T) else mul(T, beta, y[py]);
    }
    if (isZero(T, alpha)) return;
    for (0..n) |i| {
        var sum = zero(T);
        const j0 = if (i > k) i - k else 0;
        const j1 = @min(n, i + k + 1);
        for (j0..j1) |j| {
            if (symBandIndex(uplo, n, k, lda, i, j)) |idxa| {
                var av = a[idxa];
                const direct = (uplo == .upper and i <= j) or (uplo == .lower and i >= j);
                if (herm and !direct) av = conj(T, av);
                if (herm and i == j) {
                    if (comptime isComplex(T)) av.im = 0;
                }
                sum = add(T, sum, mul(T, av, vectorGet(T, x, sx, j, incx_)));
            }
        }
        const py = ix(sy, i, incy_);
        y[py] = add(T, y[py], mul(T, alpha, sum));
    }
}

pub fn spmv(comptime T: type, uplo: Uplo, n_: BlasInt, alpha: T, ap: [*]const T, x: [*]const T, incx_: BlasInt, beta: T, y: [*]T, incy_: BlasInt, herm: bool) void {
    if (n_ <= 0 or incx_ == 0 or incy_ == 0) return;
    const n = toUsize(n_);
    const sx = startIndex(n_, incx_);
    const sy = startIndex(n_, incy_);
    for (0..n) |i| {
        const py = ix(sy, i, incy_);
        y[py] = if (isZero(T, beta)) zero(T) else mul(T, beta, y[py]);
    }
    if (isZero(T, alpha)) return;
    for (0..n) |i| {
        var sum = zero(T);
        for (0..n) |j| sum = add(T, sum, mul(T, symPackedValue(T, uplo, n, ap, i, j, herm), vectorGet(T, x, sx, j, incx_)));
        const py = ix(sy, i, incy_);
        y[py] = add(T, y[py], mul(T, alpha, sum));
    }
}
