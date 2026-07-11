// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later

//! Symmetric, Hermitian, banded, and packed matrix-vector BLAS Level 2 kernels.

const std = @import("std");

const scalar = @import("../shared/scalar.zig");
const indexing = @import("../shared/indexing.zig");
const access = @import("access.zig");
const vector_ops = @import("../vector.zig");
const core_pool = @import("../execution/thread_pool.zig");

const BlasInt = scalar.BlasInt;
const Uplo = scalar.Uplo;

const zero = scalar.zero;
const one = scalar.one;
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
threadlocal var symv_workspace_c32_ptr: ?[*]scalar.ComplexF32 = null;
threadlocal var symv_workspace_c32_len: usize = 0;
threadlocal var symv_workspace_c64_ptr: ?[*]scalar.ComplexF64 = null;
threadlocal var symv_workspace_c64_len: usize = 0;

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
    if (T == scalar.ComplexF32) {
        if (symv_workspace_c32_len < len) {
            const data = std.heap.c_allocator.alloc(scalar.ComplexF32, len) catch return null;
            if (symv_workspace_c32_ptr) |old| std.heap.c_allocator.free(old[0..symv_workspace_c32_len]);
            symv_workspace_c32_ptr = data.ptr;
            symv_workspace_c32_len = len;
        }
        return symv_workspace_c32_ptr.?[0..len];
    }
    if (T == scalar.ComplexF64) {
        if (symv_workspace_c64_len < len) {
            const data = std.heap.c_allocator.alloc(scalar.ComplexF64, len) catch return null;
            if (symv_workspace_c64_ptr) |old| std.heap.c_allocator.free(old[0..symv_workspace_c64_len]);
            symv_workspace_c64_ptr = data.ptr;
            symv_workspace_c64_len = len;
        }
        return symv_workspace_c64_ptr.?[0..len];
    }
    return null;
}

pub fn freeCurrentThreadCaches() void {
    if (symv_workspace_f32_ptr) |ptr| std.heap.c_allocator.free(ptr[0..symv_workspace_f32_len]);
    symv_workspace_f32_ptr = null;
    symv_workspace_f32_len = 0;
    if (symv_workspace_f64_ptr) |ptr| std.heap.c_allocator.free(ptr[0..symv_workspace_f64_len]);
    symv_workspace_f64_ptr = null;
    symv_workspace_f64_len = 0;
    if (symv_workspace_c32_ptr) |ptr| std.heap.c_allocator.free(ptr[0..symv_workspace_c32_len]);
    symv_workspace_c32_ptr = null;
    symv_workspace_c32_len = 0;
    if (symv_workspace_c64_ptr) |ptr| std.heap.c_allocator.free(ptr[0..symv_workspace_c64_len]);
    symv_workspace_c64_ptr = null;
    symv_workspace_c64_len = 0;
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

fn mergeSymvWorkspacesUnitReal(comptime T: type, n: usize, task_count: usize, beta: T, workspace: [*]const T, y: [*]T) void {
    const lane_count = lanes(T);
    const V = @Vector(lane_count, T);
    const beta_v: V = @splat(beta);
    var i: usize = 0;
    while (i + lane_count <= n) : (i += lane_count) {
        var sum_v: V = @splat(0);
        for (0..task_count) |task_index| {
            sum_v += loadVec(T, lane_count, workspace + task_index * n, i);
        }
        const out_v = if (beta == 0)
            sum_v
        else if (beta == 1)
            loadVec(T, lane_count, y, i) + sum_v
        else
            @mulAdd(V, loadVec(T, lane_count, y, i), beta_v, sum_v);
        storeVec(T, lane_count, y, i, out_v);
    }
    while (i < n) : (i += 1) {
        var sum: T = 0;
        for (0..task_count) |task_index| {
            sum += workspace[task_index * n + i];
        }
        y[i] = if (beta == 0) sum else @mulAdd(T, y[i], beta, sum);
    }
}

fn mergeSymvUpperWorkspacesUnitReal(comptime T: type, n: usize, task_count: usize, beta: T, workspace: [*]const T, ends: []const usize, y: [*]T) void {
    const lane_count = lanes(T);
    const V = @Vector(lane_count, T);
    const beta_v: V = @splat(beta);
    var i: usize = 0;
    while (i + lane_count <= n) : (i += lane_count) {
        var sum_v: V = @splat(0);
        for (0..task_count) |task_index| {
            const end = ends[task_index];
            if (i + lane_count <= end) {
                sum_v += loadVec(T, lane_count, workspace + task_index * n, i);
            } else if (i < end) {
                var partial_v: V = @splat(0);
                inline for (0..lane_count) |lane| {
                    if (i + lane < end) partial_v[lane] = workspace[task_index * n + i + lane];
                }
                sum_v += partial_v;
            }
        }
        const out_v = if (beta == 0)
            sum_v
        else if (beta == 1)
            loadVec(T, lane_count, y, i) + sum_v
        else
            @mulAdd(V, loadVec(T, lane_count, y, i), beta_v, sum_v);
        storeVec(T, lane_count, y, i, out_v);
    }
    while (i < n) : (i += 1) {
        var sum: T = 0;
        for (0..task_count) |task_index| {
            if (i < ends[task_index]) sum += workspace[task_index * n + i];
        }
        y[i] = if (beta == 0) sum else @mulAdd(T, y[i], beta, sum);
    }
}

fn symvColumnsUnitReal(comptime T: type, uplo: Uplo, n: usize, j0: usize, j1: usize, alpha: T, a: [*]const T, lda: BlasInt, x: [*]const T, y: [*]T) void {
    const lane_count = lanes(T);
    const unroll_count = unroll(T);
    const V = @Vector(lane_count, T);
    if (uplo == .upper) {
        var j = j0;
        while (n >= 256 and j + 4 <= j1) : (j += 4) {
            const c0 = a + indexing.matIndex(lda, 0, j);
            const c1 = a + indexing.matIndex(lda, 0, j + 1);
            const c2 = a + indexing.matIndex(lda, 0, j + 2);
            const c3 = a + indexing.matIndex(lda, 0, j + 3);
            const temp1_0 = alpha * x[j];
            const temp1_1 = alpha * x[j + 1];
            const temp1_2 = alpha * x[j + 2];
            const temp1_3 = alpha * x[j + 3];
            const temp1_0_v: V = @splat(temp1_0);
            const temp1_1_v: V = @splat(temp1_1);
            const temp1_2_v: V = @splat(temp1_2);
            const temp1_3_v: V = @splat(temp1_3);
            var temp2_0_v: V = @splat(0);
            var temp2_1_v: V = @splat(0);
            var temp2_2_v: V = @splat(0);
            var temp2_3_v: V = @splat(0);
            var i: usize = 0;
            while (i + unroll_count <= j) : (i += unroll_count) {
                inline for (0..4) |k| {
                    const offset = i + k * lane_count;
                    const av0 = loadVec(T, lane_count, c0, offset);
                    const av1 = loadVec(T, lane_count, c1, offset);
                    const av2 = loadVec(T, lane_count, c2, offset);
                    const av3 = loadVec(T, lane_count, c3, offset);
                    const xv = loadVec(T, lane_count, x, offset);
                    var yv = loadVec(T, lane_count, y, offset);
                    yv = @mulAdd(V, av0, temp1_0_v, yv);
                    yv = @mulAdd(V, av1, temp1_1_v, yv);
                    yv = @mulAdd(V, av2, temp1_2_v, yv);
                    yv = @mulAdd(V, av3, temp1_3_v, yv);
                    storeVec(T, lane_count, y, offset, yv);
                    temp2_0_v = @mulAdd(V, av0, xv, temp2_0_v);
                    temp2_1_v = @mulAdd(V, av1, xv, temp2_1_v);
                    temp2_2_v = @mulAdd(V, av2, xv, temp2_2_v);
                    temp2_3_v = @mulAdd(V, av3, xv, temp2_3_v);
                }
            }
            while (i + lane_count <= j) : (i += lane_count) {
                const av0 = loadVec(T, lane_count, c0, i);
                const av1 = loadVec(T, lane_count, c1, i);
                const av2 = loadVec(T, lane_count, c2, i);
                const av3 = loadVec(T, lane_count, c3, i);
                const xv = loadVec(T, lane_count, x, i);
                var yv = loadVec(T, lane_count, y, i);
                yv = @mulAdd(V, av0, temp1_0_v, yv);
                yv = @mulAdd(V, av1, temp1_1_v, yv);
                yv = @mulAdd(V, av2, temp1_2_v, yv);
                yv = @mulAdd(V, av3, temp1_3_v, yv);
                storeVec(T, lane_count, y, i, yv);
                temp2_0_v = @mulAdd(V, av0, xv, temp2_0_v);
                temp2_1_v = @mulAdd(V, av1, xv, temp2_1_v);
                temp2_2_v = @mulAdd(V, av2, xv, temp2_2_v);
                temp2_3_v = @mulAdd(V, av3, xv, temp2_3_v);
            }
            var temp2_0: T = @reduce(.Add, temp2_0_v);
            var temp2_1: T = @reduce(.Add, temp2_1_v);
            var temp2_2: T = @reduce(.Add, temp2_2_v);
            var temp2_3: T = @reduce(.Add, temp2_3_v);
            while (i < j) : (i += 1) {
                const av0 = c0[i];
                const av1 = c1[i];
                const av2 = c2[i];
                const av3 = c3[i];
                y[i] = @mulAdd(T, av0, temp1_0, y[i]);
                y[i] = @mulAdd(T, av1, temp1_1, y[i]);
                y[i] = @mulAdd(T, av2, temp1_2, y[i]);
                y[i] = @mulAdd(T, av3, temp1_3, y[i]);
                temp2_0 = @mulAdd(T, av0, x[i], temp2_0);
                temp2_1 = @mulAdd(T, av1, x[i], temp2_1);
                temp2_2 = @mulAdd(T, av2, x[i], temp2_2);
                temp2_3 = @mulAdd(T, av3, x[i], temp2_3);
            }

            y[j] = @mulAdd(T, temp1_0, c0[j], y[j] + alpha * temp2_0);
            const a01 = c1[j];
            y[j] = @mulAdd(T, a01, temp1_1, y[j]);
            temp2_1 = @mulAdd(T, a01, x[j], temp2_1);
            const a02 = c2[j];
            y[j] = @mulAdd(T, a02, temp1_2, y[j]);
            temp2_2 = @mulAdd(T, a02, x[j], temp2_2);
            const a03 = c3[j];
            y[j] = @mulAdd(T, a03, temp1_3, y[j]);
            temp2_3 = @mulAdd(T, a03, x[j], temp2_3);

            y[j + 1] = @mulAdd(T, temp1_1, c1[j + 1], y[j + 1] + alpha * temp2_1);
            const a12 = c2[j + 1];
            y[j + 1] = @mulAdd(T, a12, temp1_2, y[j + 1]);
            temp2_2 = @mulAdd(T, a12, x[j + 1], temp2_2);
            const a13 = c3[j + 1];
            y[j + 1] = @mulAdd(T, a13, temp1_3, y[j + 1]);
            temp2_3 = @mulAdd(T, a13, x[j + 1], temp2_3);

            y[j + 2] = @mulAdd(T, temp1_2, c2[j + 2], y[j + 2] + alpha * temp2_2);
            const a23 = c3[j + 2];
            y[j + 2] = @mulAdd(T, a23, temp1_3, y[j + 2]);
            temp2_3 = @mulAdd(T, a23, x[j + 2], temp2_3);

            y[j + 3] = @mulAdd(T, temp1_3, c3[j + 3], y[j + 3] + alpha * temp2_3);
        }
        while (j < j1) : (j += 1) {
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

fn parallelSymvUnitReal(comptime T: type, uplo: Uplo, n: usize, alpha: T, a: [*]const T, lda: BlasInt, x: [*]const T, beta: T, y: [*]T) bool {
    if (n *| n < 512 * 512) return false;
    const min_cols_per_task: usize = if (T == f32 and n == 512) 64 else if (n >= 768) 64 else 128;
    var task_count = core_pool.taskCount(n, min_cols_per_task);
    const small_task_cap: usize = if (T == f32 and n == 512) 8 else 6;
    if (n <= 1536) task_count = @min(task_count, small_task_cap);
    if (task_count <= 1) return false;

    const workspace_len = task_count * n;
    if (workspace_len * @sizeOf(T) > 64 * 1024 * 1024) return false;
    const workspace = symvWorkspace(T, workspace_len) orelse return false;
    const use_upper_ranged_workspace = T == f32 and uplo == .upper and n == 512;
    if (!use_upper_ranged_workspace) @memset(workspace, 0);

    var tasks: [core_pool.max_tasks]SymvTask(T) = undefined;
    var upper_ends: [core_pool.max_tasks]usize = undefined;
    for (0..task_count) |task_index| {
        const j0 = symvTaskBoundary(uplo, n, task_count, task_index);
        const j1 = symvTaskBoundary(uplo, n, task_count, task_index + 1);
        if (use_upper_ranged_workspace) {
            upper_ends[task_index] = j1;
            @memset((workspace.ptr + task_index * n)[0..j1], 0);
        }
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
    const ran = if ((T == f32 or T == f64) and n <= 1024)
        core_pool.runLowLatency(runner, @ptrCast(&tasks), task_count)
    else
        core_pool.run(runner, @ptrCast(&tasks), task_count);
    if (!ran) return false;

    if (use_upper_ranged_workspace) {
        mergeSymvUpperWorkspacesUnitReal(T, n, task_count, beta, workspace.ptr, upper_ends[0..task_count], y);
    } else {
        mergeSymvWorkspacesUnitReal(T, n, task_count, beta, workspace.ptr, y);
    }
    return true;
}

fn symvTaskBoundary(uplo: Uplo, n: usize, task_count: usize, task_index: usize) usize {
    if (uplo == .upper) return upperTriangularTaskBoundary(n, task_count, task_index);
    return lowerTriangularTaskBoundary(n, task_count, task_index);
}

fn symvUnitReal(comptime T: type, uplo: Uplo, n: usize, alpha: T, a: [*]const T, lda: BlasInt, x: [*]const T, beta: T, y: [*]T) void {
    if (alpha != 0 and parallelSymvUnitReal(T, uplo, n, alpha, a, lda, x, beta, y)) return;
    scaleUnitReal(T, n, beta, y);
    if (alpha == 0) return;
    symvColumnsUnitReal(T, uplo, n, 0, n, alpha, a, lda, x, y);
}

fn hermDiag(comptime T: type, value: T) T {
    return .{ .re = value.re, .im = 0 };
}

inline fn c32HermAxpyTerm(xv: @Vector(8, f32), coeff: scalar.ComplexF32) @Vector(8, f32) {
    const swap_mask: @Vector(8, i32) = .{ 1, 0, 3, 2, 5, 4, 7, 6 };
    const re_v: @Vector(8, f32) = @splat(coeff.re);
    const im_v: @Vector(8, f32) = .{ -coeff.im, coeff.im, -coeff.im, coeff.im, -coeff.im, coeff.im, -coeff.im, coeff.im };
    return @mulAdd(@Vector(8, f32), xv, re_v, @shuffle(f32, xv, undefined, swap_mask) * im_v);
}

inline fn c32HermDotAccumulate(
    col_real: [*]const f32,
    x_real: [*]const f32,
    offset: usize,
    re_acc: *@Vector(8, f32),
    im_acc: *@Vector(8, f32),
) void {
    const swap_mask: @Vector(8, i32) = .{ 1, 0, 3, 2, 5, 4, 7, 6 };
    const im_sign: @Vector(8, f32) = .{ 1, -1, 1, -1, 1, -1, 1, -1 };
    const av = loadVec(f32, 8, col_real, offset);
    const xv = loadVec(f32, 8, x_real, offset);
    re_acc.* = @mulAdd(@Vector(8, f32), av, xv, re_acc.*);
    im_acc.* = @mulAdd(@Vector(8, f32), av, @shuffle(f32, xv, undefined, swap_mask) * im_sign, im_acc.*);
}

fn hermvUpperColumnC32(j: usize, alpha: scalar.ComplexF32, col: [*]const scalar.ComplexF32, x: [*]const scalar.ComplexF32, y_delta: [*]scalar.ComplexF32) void {
    const xj = mul(scalar.ComplexF32, alpha, x[j]);
    const col_real: [*]const f32 = @ptrCast(col);
    const x_real: [*]const f32 = @ptrCast(x);
    const y_real: [*]f32 = @ptrCast(y_delta);
    const real_n = 2 * j;
    var re_acc0: @Vector(8, f32) = @splat(0);
    var re_acc1: @Vector(8, f32) = @splat(0);
    var im_acc0: @Vector(8, f32) = @splat(0);
    var im_acc1: @Vector(8, f32) = @splat(0);
    var i: usize = 0;
    while (i + 16 <= real_n) : (i += 16) {
        var yv0 = loadVec(f32, 8, y_real, i);
        var yv1 = loadVec(f32, 8, y_real, i + 8);
        const av0 = loadVec(f32, 8, col_real, i);
        const av1 = loadVec(f32, 8, col_real, i + 8);
        yv0 += c32HermAxpyTerm(av0, xj);
        yv1 += c32HermAxpyTerm(av1, xj);
        storeVec(f32, 8, y_real, i, yv0);
        storeVec(f32, 8, y_real, i + 8, yv1);
        c32HermDotAccumulate(col_real, x_real, i, &re_acc0, &im_acc0);
        c32HermDotAccumulate(col_real, x_real, i + 8, &re_acc1, &im_acc1);
    }
    var re_acc = re_acc0 + re_acc1;
    var im_acc = im_acc0 + im_acc1;
    while (i + 8 <= real_n) : (i += 8) {
        const av = loadVec(f32, 8, col_real, i);
        const yv = loadVec(f32, 8, y_real, i) + c32HermAxpyTerm(av, xj);
        storeVec(f32, 8, y_real, i, yv);
        c32HermDotAccumulate(col_real, x_real, i, &re_acc, &im_acc);
    }
    var sum = scalar.ComplexF32{ .re = @reduce(.Add, re_acc), .im = @reduce(.Add, im_acc) };
    while (i < real_n) : (i += 2) {
        const ar = col_real[i];
        const ai = col_real[i + 1];
        const xr = x_real[i];
        const xi = x_real[i + 1];
        y_real[i] += xj.re * ar - xj.im * ai;
        y_real[i + 1] += xj.re * ai + xj.im * ar;
        sum.re = @mulAdd(f32, ai, xi, @mulAdd(f32, ar, xr, sum.re));
        sum.im = @mulAdd(f32, -ai, xr, @mulAdd(f32, ar, xi, sum.im));
    }
    y_delta[j] = add(scalar.ComplexF32, y_delta[j], mul(scalar.ComplexF32, alpha, sum));
    y_delta[j] = add(scalar.ComplexF32, y_delta[j], mul(scalar.ComplexF32, xj, hermDiag(scalar.ComplexF32, col[j])));
}

inline fn c64HermAxpyTerm(xv: @Vector(4, f64), coeff: scalar.ComplexF64) @Vector(4, f64) {
    const swap_mask: @Vector(4, i32) = .{ 1, 0, 3, 2 };
    const re_v: @Vector(4, f64) = @splat(coeff.re);
    const im_v: @Vector(4, f64) = .{ -coeff.im, coeff.im, -coeff.im, coeff.im };
    return @mulAdd(@Vector(4, f64), xv, re_v, @shuffle(f64, xv, undefined, swap_mask) * im_v);
}

inline fn c64HermDotAccumulate(
    col_real: [*]const f64,
    x_real: [*]const f64,
    offset: usize,
    re_acc: *@Vector(4, f64),
    im_acc: *@Vector(4, f64),
) void {
    const swap_mask: @Vector(4, i32) = .{ 1, 0, 3, 2 };
    const im_sign: @Vector(4, f64) = .{ 1, -1, 1, -1 };
    const av = loadVec(f64, 4, col_real, offset);
    const xv = loadVec(f64, 4, x_real, offset);
    re_acc.* = @mulAdd(@Vector(4, f64), av, xv, re_acc.*);
    im_acc.* = @mulAdd(@Vector(4, f64), av, @shuffle(f64, xv, undefined, swap_mask) * im_sign, im_acc.*);
}

fn hermvUpperColumnC64(j: usize, alpha: scalar.ComplexF64, col: [*]const scalar.ComplexF64, x: [*]const scalar.ComplexF64, y_delta: [*]scalar.ComplexF64) void {
    const xj = mul(scalar.ComplexF64, alpha, x[j]);
    const col_real: [*]const f64 = @ptrCast(col);
    const x_real: [*]const f64 = @ptrCast(x);
    const y_real: [*]f64 = @ptrCast(y_delta);
    const real_n = 2 * j;
    var re_acc0: @Vector(4, f64) = @splat(0);
    var re_acc1: @Vector(4, f64) = @splat(0);
    var im_acc0: @Vector(4, f64) = @splat(0);
    var im_acc1: @Vector(4, f64) = @splat(0);
    var i: usize = 0;
    while (i + 8 <= real_n) : (i += 8) {
        var yv0 = loadVec(f64, 4, y_real, i);
        var yv1 = loadVec(f64, 4, y_real, i + 4);
        const av0 = loadVec(f64, 4, col_real, i);
        const av1 = loadVec(f64, 4, col_real, i + 4);
        yv0 += c64HermAxpyTerm(av0, xj);
        yv1 += c64HermAxpyTerm(av1, xj);
        storeVec(f64, 4, y_real, i, yv0);
        storeVec(f64, 4, y_real, i + 4, yv1);
        c64HermDotAccumulate(col_real, x_real, i, &re_acc0, &im_acc0);
        c64HermDotAccumulate(col_real, x_real, i + 4, &re_acc1, &im_acc1);
    }
    var re_acc = re_acc0 + re_acc1;
    var im_acc = im_acc0 + im_acc1;
    while (i + 4 <= real_n) : (i += 4) {
        const av = loadVec(f64, 4, col_real, i);
        const yv = loadVec(f64, 4, y_real, i) + c64HermAxpyTerm(av, xj);
        storeVec(f64, 4, y_real, i, yv);
        c64HermDotAccumulate(col_real, x_real, i, &re_acc, &im_acc);
    }
    var sum = scalar.ComplexF64{ .re = @reduce(.Add, re_acc), .im = @reduce(.Add, im_acc) };
    while (i < real_n) : (i += 2) {
        const ar = col_real[i];
        const ai = col_real[i + 1];
        const xr = x_real[i];
        const xi = x_real[i + 1];
        y_real[i] += xj.re * ar - xj.im * ai;
        y_real[i + 1] += xj.re * ai + xj.im * ar;
        sum.re = @mulAdd(f64, ai, xi, @mulAdd(f64, ar, xr, sum.re));
        sum.im = @mulAdd(f64, -ai, xr, @mulAdd(f64, ar, xi, sum.im));
    }
    y_delta[j] = add(scalar.ComplexF64, y_delta[j], mul(scalar.ComplexF64, alpha, sum));
    y_delta[j] = add(scalar.ComplexF64, y_delta[j], mul(scalar.ComplexF64, xj, hermDiag(scalar.ComplexF64, col[j])));
}

fn hermvColumnsUnitComplex(comptime T: type, uplo: Uplo, n: usize, j0: usize, j1: usize, alpha: T, a: [*]const T, lda: BlasInt, x: [*]const T, y_delta: [*]T) void {
    if (uplo == .upper) {
        for (j0..j1) |j| {
            const col = a + indexing.matIndex(lda, 0, j);
            if (T == scalar.ComplexF32) {
                hermvUpperColumnC32(j, alpha, @ptrCast(col), @ptrCast(x), @ptrCast(y_delta));
                continue;
            }
            if (T == scalar.ComplexF64) {
                hermvUpperColumnC64(j, alpha, @ptrCast(col), @ptrCast(x), @ptrCast(y_delta));
                continue;
            }
            const xj = mul(T, alpha, x[j]);
            if (j > 0) {
                if (!isZero(T, xj)) vector_ops.axpy(T, @intCast(j), xj, col, 1, y_delta, 1);
                const sum = vector_ops.dot(T, @intCast(j), col, 1, x, 1, true);
                y_delta[j] = add(T, y_delta[j], mul(T, alpha, sum));
            }
            y_delta[j] = add(T, y_delta[j], mul(T, xj, hermDiag(T, col[j])));
        }
    } else {
        for (j0..j1) |j| {
            const col = a + indexing.matIndex(lda, 0, j);
            const xj = mul(T, alpha, x[j]);
            y_delta[j] = add(T, y_delta[j], mul(T, xj, hermDiag(T, col[j])));
            if (j + 1 < n) {
                const tail_len = n - j - 1;
                const tail_col = col + j + 1;
                const tail_x = x + j + 1;
                if (!isZero(T, xj)) vector_ops.axpy(T, @intCast(tail_len), xj, tail_col, 1, y_delta + j + 1, 1);
                const sum = vector_ops.dot(T, @intCast(tail_len), tail_col, 1, tail_x, 1, true);
                y_delta[j] = add(T, y_delta[j], mul(T, alpha, sum));
            }
        }
    }
}

fn HermvTask(comptime T: type) type {
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

fn runHermvTask(comptime T: type, raw_tasks: *const anyopaque, index: usize) void {
    const tasks: [*]const HermvTask(T) = @ptrCast(@alignCast(raw_tasks));
    const task = tasks[index];
    hermvColumnsUnitComplex(T, task.uplo, task.n, task.j0, task.j1, task.alpha, task.a, task.lda, task.x, task.y_delta);
}

fn runHermvTaskC32(raw_tasks: *const anyopaque, index: usize) void {
    runHermvTask(scalar.ComplexF32, raw_tasks, index);
}

fn runHermvTaskC64(raw_tasks: *const anyopaque, index: usize) void {
    runHermvTask(scalar.ComplexF64, raw_tasks, index);
}

fn mergeHermvUpperWorkspacesUnitComplex(comptime T: type, n: usize, task_count: usize, workspace: [*]const T, ends: []const usize, y: [*]T) void {
    const add_alpha = one(T);
    for (0..task_count) |task_index| {
        const end = ends[task_index];
        if (end == 0) continue;
        vector_ops.axpy(T, @intCast(end), add_alpha, workspace + task_index * n, 1, y, 1);
    }
}

fn upperTriangularTaskBoundary(n: usize, task_count: usize, task_index: usize) usize {
    if (task_index == 0) return 0;
    if (task_index >= task_count) return n;
    const fraction = @as(f64, @floatFromInt(task_index)) / @as(f64, @floatFromInt(task_count));
    const boundary = @sqrt(fraction) * @as(f64, @floatFromInt(n));
    return @min(n, @as(usize, @intFromFloat(boundary)));
}

fn lowerTriangularTaskBoundary(n: usize, task_count: usize, task_index: usize) usize {
    if (task_index == 0) return 0;
    if (task_index >= task_count) return n;
    const fraction = @as(f64, @floatFromInt(task_index)) / @as(f64, @floatFromInt(task_count));
    const boundary = (1.0 - @sqrt(1.0 - fraction)) * @as(f64, @floatFromInt(n));
    return @min(n, @as(usize, @intFromFloat(boundary)));
}

fn parallelHermvUnitComplex(comptime T: type, uplo: Uplo, n: usize, n_: BlasInt, alpha: T, a: [*]const T, lda: BlasInt, x: [*]const T, y: [*]T) bool {
    if (n *| n < 256 * 256) return false;
    const min_cols_per_task: usize = 48;
    var task_count = core_pool.taskCount(n, min_cols_per_task);
    if (n <= 512) task_count = @min(task_count, 10);
    const balance_upper = uplo == .upper and n >= 256;
    if (balance_upper and T == scalar.ComplexF64 and n == 512) task_count = @min(task_count, 8);
    if (task_count <= 1) return false;

    const workspace_len = task_count * n;
    if (workspace_len * @sizeOf(T) > 64 * 1024 * 1024) return false;
    const workspace = symvWorkspace(T, workspace_len) orelse return false;
    const use_upper_ranged_workspace = uplo == .upper;
    if (!use_upper_ranged_workspace) @memset(workspace, zero(T));

    var tasks: [core_pool.max_tasks]HermvTask(T) = undefined;
    var upper_ends: [core_pool.max_tasks]usize = undefined;
    for (0..task_count) |task_index| {
        const j0 = if (balance_upper) upperTriangularTaskBoundary(n, task_count, task_index) else task_index * n / task_count;
        const j1 = if (balance_upper) upperTriangularTaskBoundary(n, task_count, task_index + 1) else (task_index + 1) * n / task_count;
        if (use_upper_ranged_workspace) {
            upper_ends[task_index] = j1;
            @memset((workspace.ptr + task_index * n)[0..j1], zero(T));
        }
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

    const runner = if (T == scalar.ComplexF32) runHermvTaskC32 else runHermvTaskC64;
    if (!core_pool.runLowLatency(runner, @ptrCast(&tasks), task_count)) return false;

    if (use_upper_ranged_workspace) {
        mergeHermvUpperWorkspacesUnitComplex(T, n, task_count, workspace.ptr, upper_ends[0..task_count], y);
    } else {
        const add_alpha = one(T);
        for (0..task_count) |task_index| {
            vector_ops.axpy(T, n_, add_alpha, workspace.ptr + task_index * n, 1, y, 1);
        }
    }
    return true;
}

fn hermvUnitComplex(comptime T: type, uplo: Uplo, n_: BlasInt, alpha: T, a: [*]const T, lda: BlasInt, x: [*]const T, beta: T, y: [*]T) void {
    const n = toUsize(n_);
    vector_ops.scal(T, n_, beta, y, 1);
    if (isZero(T, alpha)) return;
    if (parallelHermvUnitComplex(T, uplo, n, n_, alpha, a, lda, x, y)) return;
    hermvColumnsUnitComplex(T, uplo, n, 0, n, alpha, a, lda, x, y);
}

pub fn symv(comptime T: type, uplo: Uplo, n_: BlasInt, alpha: T, a: [*]const T, lda: BlasInt, x: [*]const T, incx_: BlasInt, beta: T, y: [*]T, incy_: BlasInt, herm: bool) void {
    if (n_ <= 0 or incx_ == 0 or incy_ == 0) return;
    const n = toUsize(n_);
    if (comptime isReal(T)) {
        if (!herm and incx_ == 1 and incy_ == 1) return symvUnitReal(T, uplo, n, alpha, a, lda, x, beta, y);
    } else if (comptime isComplex(T)) {
        if (herm and incx_ == 1 and incy_ == 1) return hermvUnitComplex(T, uplo, n_, alpha, a, lda, x, beta, y);
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
    if (incx_ == 1 and incy_ == 1 and n >= 512 and k >= 8) {
        if (comptime isReal(T)) {
            scaleUnitReal(T, n, beta, y);
        } else {
            vector_ops.scal(T, n_, beta, y, 1);
        }
        if (isZero(T, alpha)) return;

        const lda_u = toUsize(lda);
        for (0..n) |j| {
            const row0 = if (uplo == .upper) (if (j > k) j - k else 0) else j + 1;
            const row1 = if (uplo == .upper) j else @min(n, j + k + 1);
            const len = row1 - row0;
            const col = if (uplo == .upper)
                a + j * lda_u + (k + row0 - j)
            else
                a + j * lda_u + 1;
            const xj = mul(T, alpha, x[j]);
            if (len != 0) {
                if (!isZero(T, xj)) {
                    if (comptime isReal(T)) {
                        vector_ops.axpyUnitReal(T, len, xj, col, y + row0);
                    } else {
                        vector_ops.axpy(T, @intCast(len), xj, col, 1, y + row0, 1);
                    }
                }
                const sum = if (comptime isReal(T))
                    vector_ops.dotUnitReal(T, len, col, x + row0)
                else
                    vector_ops.dot(T, @intCast(len), col, 1, x + row0, 1, herm);
                y[j] = add(T, y[j], mul(T, alpha, sum));
            }

            var diag_value = a[j * lda_u + (if (uplo == .upper) k else 0)];
            if (herm and comptime isComplex(T)) diag_value.im = 0;
            y[j] = add(T, y[j], mul(T, xj, diag_value));
        }
        return;
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

fn packedMvColumnsUnit(comptime T: type, uplo: Uplo, n: usize, j0: usize, j1: usize, alpha: T, ap: [*]const T, x: [*]const T, y_delta: [*]T, herm: bool) void {
    for (j0..j1) |j| {
        const xj = x[j];
        const scaled_xj = mul(T, alpha, xj);
        var mirrored_sum = zero(T);
        var diag_value: T = undefined;

        if (uplo == .upper) {
            const column_start = j * (j + 1) / 2;
            for (0..j) |i| {
                const value = ap[column_start + i];
                y_delta[i] = add(T, y_delta[i], mul(T, value, scaled_xj));
                const mirrored = if (herm) conj(T, value) else value;
                mirrored_sum = add(T, mirrored_sum, mul(T, mirrored, x[i]));
            }
            diag_value = ap[column_start + j];
        } else {
            const column_start = j * (2 * n - j + 1) / 2;
            diag_value = ap[column_start];
            for (j + 1..n) |i| {
                const value = ap[column_start + (i - j)];
                y_delta[i] = add(T, y_delta[i], mul(T, value, scaled_xj));
                const mirrored = if (herm) conj(T, value) else value;
                mirrored_sum = add(T, mirrored_sum, mul(T, mirrored, x[i]));
            }
        }

        if (herm and comptime isComplex(T)) diag_value.im = 0;
        const diagonal_sum = add(T, mirrored_sum, mul(T, diag_value, xj));
        y_delta[j] = add(T, y_delta[j], mul(T, alpha, diagonal_sum));
    }
}

fn PackedMvTask(comptime T: type) type {
    return struct {
        uplo: Uplo,
        n: usize,
        j0: usize,
        j1: usize,
        alpha: T,
        ap: [*]const T,
        x: [*]const T,
        y_delta: [*]T,
        herm: bool,
    };
}

fn runPackedMvTask(comptime T: type, raw_tasks: *const anyopaque, index: usize) void {
    const tasks: [*]const PackedMvTask(T) = @ptrCast(@alignCast(raw_tasks));
    const task = tasks[index];
    packedMvColumnsUnit(T, task.uplo, task.n, task.j0, task.j1, task.alpha, task.ap, task.x, task.y_delta, task.herm);
}

fn runPackedMvTaskF32(raw_tasks: *const anyopaque, index: usize) void {
    runPackedMvTask(f32, raw_tasks, index);
}

fn runPackedMvTaskF64(raw_tasks: *const anyopaque, index: usize) void {
    runPackedMvTask(f64, raw_tasks, index);
}

fn runPackedMvTaskC32(raw_tasks: *const anyopaque, index: usize) void {
    runPackedMvTask(scalar.ComplexF32, raw_tasks, index);
}

fn runPackedMvTaskC64(raw_tasks: *const anyopaque, index: usize) void {
    runPackedMvTask(scalar.ComplexF64, raw_tasks, index);
}

fn mergePackedMvWorkspacesUnit(comptime T: type, n: usize, task_count: usize, beta: T, workspace: [*]const T, y: [*]T) void {
    if (comptime isReal(T)) {
        mergeSymvWorkspacesUnitReal(T, n, task_count, beta, workspace, y);
        return;
    }
    for (0..n) |i| {
        var sum = zero(T);
        for (0..task_count) |task_index| {
            sum = add(T, sum, workspace[task_index * n + i]);
        }
        y[i] = if (isZero(T, beta)) sum else add(T, mul(T, beta, y[i]), sum);
    }
}

noinline fn parallelPackedMvUnit(comptime T: type, uplo: Uplo, n: usize, alpha: T, ap: [*]const T, x: [*]const T, beta: T, y: [*]T, herm: bool) bool {
    const min_cols_per_task: usize = 64;
    const task_count = core_pool.taskCount(n, min_cols_per_task);
    if (task_count <= 1) return false;

    const max_workspace_bytes: usize = 64 * 1024 * 1024;
    if (n > max_workspace_bytes / @sizeOf(T) / task_count) return false;
    const workspace_len = task_count * n;
    const workspace = symvWorkspace(T, workspace_len) orelse return false;
    @memset(workspace, zero(T));

    var tasks: [core_pool.max_tasks]PackedMvTask(T) = undefined;
    for (0..task_count) |task_index| {
        const j0 = if (uplo == .upper)
            upperTriangularTaskBoundary(n, task_count, task_index)
        else
            lowerTriangularTaskBoundary(n, task_count, task_index);
        const j1 = if (uplo == .upper)
            upperTriangularTaskBoundary(n, task_count, task_index + 1)
        else
            lowerTriangularTaskBoundary(n, task_count, task_index + 1);
        tasks[task_index] = .{
            .uplo = uplo,
            .n = n,
            .j0 = j0,
            .j1 = j1,
            .alpha = alpha,
            .ap = ap,
            .x = x,
            .y_delta = workspace.ptr + task_index * n,
            .herm = herm,
        };
    }

    const runner = if (T == f32)
        runPackedMvTaskF32
    else if (T == f64)
        runPackedMvTaskF64
    else if (T == scalar.ComplexF32)
        runPackedMvTaskC32
    else if (T == scalar.ComplexF64)
        runPackedMvTaskC64
    else
        return false;
    // A false result means core_pool ran no task, so the caller can use the unchanged serial body.
    if (!core_pool.runLowLatency(runner, @ptrCast(&tasks), task_count)) return false;

    mergePackedMvWorkspacesUnit(T, n, task_count, beta, workspace.ptr, y);
    return true;
}

fn spmvSerial(comptime T: type, uplo: Uplo, n_: BlasInt, alpha: T, ap: [*]const T, x: [*]const T, incx_: BlasInt, beta: T, y: [*]T, incy_: BlasInt, herm: bool) void {
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

pub fn spmv(comptime T: type, uplo: Uplo, n_: BlasInt, alpha: T, ap: [*]const T, x: [*]const T, incx_: BlasInt, beta: T, y: [*]T, incy_: BlasInt, herm: bool) void {
    if (n_ <= 0 or incx_ == 0 or incy_ == 0) return;
    const n = toUsize(n_);
    if (n >= 512 and incx_ == 1 and incy_ == 1 and !isZero(T, alpha)) {
        if (parallelPackedMvUnit(T, uplo, n, alpha, ap, x, beta, y, herm)) return;
    }
    spmvSerial(T, uplo, n_, alpha, ap, x, incx_, beta, y, incy_, herm);
}
