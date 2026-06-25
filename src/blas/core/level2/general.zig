// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later

//! General dense and banded matrix-vector BLAS Level 2 kernels.

const scalar = @import("../scalar.zig");
const indexing = @import("../indexing.zig");
const level1 = @import("../level1.zig");
const core_pool = @import("../pool.zig");
const matrix_vector_kernels = @import("../../kernels/matrix_vector.zig");
const std = @import("std");

const BlasInt = scalar.BlasInt;
const Order = scalar.Order;

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
const matIndex = indexing.matIndex;
const bandGeneralIndex = indexing.bandGeneralIndex;
const vectorGet = indexing.vectorGet;

threadlocal var gemv_workspace_f32_ptr: ?[*]f32 = null;
threadlocal var gemv_workspace_f32_len: usize = 0;
threadlocal var gemv_workspace_f64_ptr: ?[*]f64 = null;
threadlocal var gemv_workspace_f64_len: usize = 0;
threadlocal var gemv_workspace_c32_ptr: ?[*]scalar.ComplexF32 = null;
threadlocal var gemv_workspace_c32_len: usize = 0;
threadlocal var gemv_workspace_c64_ptr: ?[*]scalar.ComplexF64 = null;
threadlocal var gemv_workspace_c64_len: usize = 0;

fn gemvWorkspace(comptime T: type, len: usize) ?[]T {
    if (T == f32) {
        if (gemv_workspace_f32_len < len) {
            const data = std.heap.c_allocator.alloc(f32, len) catch return null;
            if (gemv_workspace_f32_ptr) |old| std.heap.c_allocator.free(old[0..gemv_workspace_f32_len]);
            gemv_workspace_f32_ptr = data.ptr;
            gemv_workspace_f32_len = len;
        }
        return gemv_workspace_f32_ptr.?[0..len];
    }
    if (T == f64) {
        if (gemv_workspace_f64_len < len) {
            const data = std.heap.c_allocator.alloc(f64, len) catch return null;
            if (gemv_workspace_f64_ptr) |old| std.heap.c_allocator.free(old[0..gemv_workspace_f64_len]);
            gemv_workspace_f64_ptr = data.ptr;
            gemv_workspace_f64_len = len;
        }
        return gemv_workspace_f64_ptr.?[0..len];
    }
    if (T == scalar.ComplexF32) {
        if (gemv_workspace_c32_len < len) {
            const data = std.heap.c_allocator.alloc(scalar.ComplexF32, len) catch return null;
            if (gemv_workspace_c32_ptr) |old| std.heap.c_allocator.free(old[0..gemv_workspace_c32_len]);
            gemv_workspace_c32_ptr = data.ptr;
            gemv_workspace_c32_len = len;
        }
        return gemv_workspace_c32_ptr.?[0..len];
    }
    if (T == scalar.ComplexF64) {
        if (gemv_workspace_c64_len < len) {
            const data = std.heap.c_allocator.alloc(scalar.ComplexF64, len) catch return null;
            if (gemv_workspace_c64_ptr) |old| std.heap.c_allocator.free(old[0..gemv_workspace_c64_len]);
            gemv_workspace_c64_ptr = data.ptr;
            gemv_workspace_c64_len = len;
        }
        return gemv_workspace_c64_ptr.?[0..len];
    }
    return null;
}

fn isReal(comptime T: type) bool {
    return T == f32 or T == f64;
}

fn lanes(comptime T: type) comptime_int {
    if (T == f32) return 8;
    if (T == f64) return 4;
    @compileError("real GEMV vector lanes support f32 and f64");
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

inline fn loadF64x2(ptr: [*]const f64, index: usize) @Vector(2, f64) {
    return @as(*align(1) const @Vector(2, f64), @ptrCast(ptr + index)).*;
}

fn scaleUnitReal(comptime T: type, n: usize, beta: T, y: [*]T) void {
    if (beta == 1) return;
    if (beta == 0) {
        @memset(y[0..n], 0);
        return;
    }
    level1.scalUnitReal(T, n, beta, y);
}

fn axpyUnitReal(comptime T: type, n: usize, alpha: T, x: [*]const T, y: [*]T) void {
    level1.axpyUnitReal(T, n, alpha, x, y);
}

fn dotUnitReal(comptime T: type, n: usize, x: [*]const T, y: [*]const T) T {
    return level1.dotUnitReal(T, n, x, y);
}

fn gemvNoTransUnitReal(comptime T: type, m: usize, n: usize, alpha: T, a: [*]const T, lda: BlasInt, x: [*]const T, y: [*]T) void {
    if (matrix_vector_kernels.gemvNoTransUnitReal(T, m, n, alpha, a, lda, x, y)) return;
    const lane_count = lanes(T);
    const unroll_count = unroll(T);
    const V = @Vector(lane_count, T);
    var j: usize = 0;
    while (j + 8 <= n) : (j += 8) {
        const s0: V = @splat(alpha * x[j]);
        const s1: V = @splat(alpha * x[j + 1]);
        const s2: V = @splat(alpha * x[j + 2]);
        const s3: V = @splat(alpha * x[j + 3]);
        const s4: V = @splat(alpha * x[j + 4]);
        const s5: V = @splat(alpha * x[j + 5]);
        const s6: V = @splat(alpha * x[j + 6]);
        const s7: V = @splat(alpha * x[j + 7]);
        const c0 = a + matIndex(lda, 0, j);
        const c1 = a + matIndex(lda, 0, j + 1);
        const c2 = a + matIndex(lda, 0, j + 2);
        const c3 = a + matIndex(lda, 0, j + 3);
        const c4 = a + matIndex(lda, 0, j + 4);
        const c5 = a + matIndex(lda, 0, j + 5);
        const c6 = a + matIndex(lda, 0, j + 6);
        const c7 = a + matIndex(lda, 0, j + 7);
        var i: usize = 0;
        while (i + unroll_count <= m) : (i += unroll_count) {
            inline for (0..4) |k| {
                const offset = i + k * lane_count;
                var yv = loadVec(T, lane_count, y, offset);
                yv = @mulAdd(V, loadVec(T, lane_count, c0, offset), s0, yv);
                yv = @mulAdd(V, loadVec(T, lane_count, c1, offset), s1, yv);
                yv = @mulAdd(V, loadVec(T, lane_count, c2, offset), s2, yv);
                yv = @mulAdd(V, loadVec(T, lane_count, c3, offset), s3, yv);
                yv = @mulAdd(V, loadVec(T, lane_count, c4, offset), s4, yv);
                yv = @mulAdd(V, loadVec(T, lane_count, c5, offset), s5, yv);
                yv = @mulAdd(V, loadVec(T, lane_count, c6, offset), s6, yv);
                yv = @mulAdd(V, loadVec(T, lane_count, c7, offset), s7, yv);
                storeVec(T, lane_count, y, offset, yv);
            }
        }
        while (i + lane_count <= m) : (i += lane_count) {
            var yv = loadVec(T, lane_count, y, i);
            yv = @mulAdd(V, loadVec(T, lane_count, c0, i), s0, yv);
            yv = @mulAdd(V, loadVec(T, lane_count, c1, i), s1, yv);
            yv = @mulAdd(V, loadVec(T, lane_count, c2, i), s2, yv);
            yv = @mulAdd(V, loadVec(T, lane_count, c3, i), s3, yv);
            yv = @mulAdd(V, loadVec(T, lane_count, c4, i), s4, yv);
            yv = @mulAdd(V, loadVec(T, lane_count, c5, i), s5, yv);
            yv = @mulAdd(V, loadVec(T, lane_count, c6, i), s6, yv);
            yv = @mulAdd(V, loadVec(T, lane_count, c7, i), s7, yv);
            storeVec(T, lane_count, y, i, yv);
        }
        while (i < m) : (i += 1) {
            var yi = y[i];
            yi = @mulAdd(T, a[matIndex(lda, i, j)], alpha * x[j], yi);
            yi = @mulAdd(T, a[matIndex(lda, i, j + 1)], alpha * x[j + 1], yi);
            yi = @mulAdd(T, a[matIndex(lda, i, j + 2)], alpha * x[j + 2], yi);
            yi = @mulAdd(T, a[matIndex(lda, i, j + 3)], alpha * x[j + 3], yi);
            yi = @mulAdd(T, a[matIndex(lda, i, j + 4)], alpha * x[j + 4], yi);
            yi = @mulAdd(T, a[matIndex(lda, i, j + 5)], alpha * x[j + 5], yi);
            yi = @mulAdd(T, a[matIndex(lda, i, j + 6)], alpha * x[j + 6], yi);
            yi = @mulAdd(T, a[matIndex(lda, i, j + 7)], alpha * x[j + 7], yi);
            y[i] = yi;
        }
    }
    while (j < n) : (j += 1) {
        const xj = alpha * x[j];
        if (xj != 0) axpyUnitReal(T, m, xj, a + matIndex(lda, 0, j), y);
    }
}

fn gemvNoTransRowsUnitReal(comptime T: type, row_count: usize, n: usize, row0: usize, alpha: T, a: [*]const T, lda: BlasInt, x: [*]const T, y: [*]T) void {
    for (0..n) |j| {
        const xj = alpha * x[j];
        if (xj != 0) axpyUnitReal(T, row_count, xj, a + matIndex(lda, row0, j), y + row0);
    }
}

fn GemvNoTransTask(comptime T: type) type {
    return struct {
        m0: usize,
        m1: usize,
        n: usize,
        alpha: T,
        a: [*]const T,
        lda: BlasInt,
        x: [*]const T,
        y: [*]T,
    };
}

fn runGemvNoTransTask(comptime T: type, raw_tasks: *const anyopaque, index: usize) void {
    const tasks: [*]const GemvNoTransTask(T) = @ptrCast(@alignCast(raw_tasks));
    const task = tasks[index];
    gemvNoTransRowsUnitReal(T, task.m1 - task.m0, task.n, task.m0, task.alpha, task.a, task.lda, task.x, task.y);
}

fn runGemvNoTransTaskF32(raw_tasks: *const anyopaque, index: usize) void {
    runGemvNoTransTask(f32, raw_tasks, index);
}

fn runGemvNoTransTaskF64(raw_tasks: *const anyopaque, index: usize) void {
    runGemvNoTransTask(f64, raw_tasks, index);
}

fn GemvNoTransPackedRowsTask(comptime T: type) type {
    return struct {
        block0: usize,
        block1: usize,
        n: usize,
        a: [*]const T,
        lda: BlasInt,
        pack: [*]const T,
        scratch: [*]T,
        y: [*]T,
    };
}

fn runGemvNoTransPackedRowsTask(comptime T: type, raw_tasks: *const anyopaque, index: usize) void {
    const tasks: [*]const GemvNoTransPackedRowsTask(T) = @ptrCast(@alignCast(raw_tasks));
    const task = tasks[index];
    const row0 = task.block0 * 8;
    const row_count = (task.block1 - task.block0) * 8;
    _ = matrix_vector_kernels.gemvNoTransPackedRowsUnitReal(
        T,
        row_count,
        task.n,
        task.a + row0,
        task.lda,
        task.pack,
        task.scratch + row0,
        task.y + row0,
    );
}

fn runGemvNoTransPackedRowsTaskF32(raw_tasks: *const anyopaque, index: usize) void {
    runGemvNoTransPackedRowsTask(f32, raw_tasks, index);
}

fn runGemvNoTransPackedRowsTaskF64(raw_tasks: *const anyopaque, index: usize) void {
    runGemvNoTransPackedRowsTask(f64, raw_tasks, index);
}

fn parallelGemvNoTransPackedRowsUnitReal(comptime T: type, m: usize, n: usize, alpha: T, a: [*]const T, lda: BlasInt, x: [*]const T, y: [*]T) bool {
    if (m *| n < 768 * 768) return false;
    const pack_len = matrix_vector_kernels.gemvNoTransPackLenUnitReal(T, m, n, lda) orelse return false;
    const block_count = m / 8;
    const task_count = core_pool.taskCount(block_count, 8);
    if (task_count <= 1) return false;

    const workspace = gemvWorkspace(T, pack_len + m) orelse return false;
    const pack = workspace[0..pack_len];
    const scratch = workspace[pack_len .. pack_len + m];
    if (!matrix_vector_kernels.gemvNoTransPackUnitReal(T, n, alpha, x, pack)) return false;

    var tasks: [core_pool.max_tasks]GemvNoTransPackedRowsTask(T) = undefined;
    for (0..task_count) |task_index| {
        tasks[task_index] = .{
            .block0 = task_index * block_count / task_count,
            .block1 = (task_index + 1) * block_count / task_count,
            .n = n,
            .a = a,
            .lda = lda,
            .pack = pack.ptr,
            .scratch = scratch.ptr,
            .y = y,
        };
    }

    const runner = if (T == f32) runGemvNoTransPackedRowsTaskF32 else runGemvNoTransPackedRowsTaskF64;
    return core_pool.run(runner, @ptrCast(&tasks), task_count);
}

fn parallelGemvNoTransUnitReal(comptime T: type, m: usize, n: usize, alpha: T, a: [*]const T, lda: BlasInt, x: [*]const T, y: [*]T) bool {
    if (m *| n < 512 * 512) return false;
    const task_count = core_pool.taskCount(m, 128);
    if (task_count <= 1) return false;

    var tasks: [core_pool.max_tasks]GemvNoTransTask(T) = undefined;
    for (0..task_count) |task_index| {
        tasks[task_index] = .{
            .m0 = task_index * m / task_count,
            .m1 = (task_index + 1) * m / task_count,
            .n = n,
            .alpha = alpha,
            .a = a,
            .lda = lda,
            .x = x,
            .y = y,
        };
    }

    const runner = if (T == f32) runGemvNoTransTaskF32 else runGemvNoTransTaskF64;
    return core_pool.run(runner, @ptrCast(&tasks), task_count);
}

fn GemvNoTransColumnTask(comptime T: type) type {
    return struct {
        m: usize,
        n0: usize,
        n1: usize,
        alpha: T,
        a: [*]const T,
        lda: BlasInt,
        x: [*]const T,
        y_delta: [*]T,
    };
}

fn runGemvNoTransColumnTask(comptime T: type, raw_tasks: *const anyopaque, index: usize) void {
    const tasks: [*]const GemvNoTransColumnTask(T) = @ptrCast(@alignCast(raw_tasks));
    const task = tasks[index];
    gemvNoTransUnitReal(T, task.m, task.n1 - task.n0, task.alpha, task.a + matIndex(task.lda, 0, task.n0), task.lda, task.x + task.n0, task.y_delta);
}

fn runGemvNoTransColumnTaskF32(raw_tasks: *const anyopaque, index: usize) void {
    runGemvNoTransColumnTask(f32, raw_tasks, index);
}

fn runGemvNoTransColumnTaskF64(raw_tasks: *const anyopaque, index: usize) void {
    runGemvNoTransColumnTask(f64, raw_tasks, index);
}

fn addUnitReal(comptime T: type, n: usize, x: [*]const T, y: [*]T) void {
    const lane_count = lanes(T);
    var i: usize = 0;
    while (i + lane_count <= n) : (i += lane_count) {
        storeVec(T, lane_count, y, i, loadVec(T, lane_count, y, i) + loadVec(T, lane_count, x, i));
    }
    while (i < n) : (i += 1) y[i] += x[i];
}

fn parallelGemvNoTransColumnsUnitReal(comptime T: type, m: usize, n: usize, alpha: T, a: [*]const T, lda: BlasInt, x: [*]const T, y: [*]T) bool {
    if (m *| n < 512 * 512) return false;
    const task_count = core_pool.taskCount(n, 256);
    if (task_count <= 1) return false;

    const workspace_len = task_count * m;
    if (workspace_len * @sizeOf(T) > 64 * 1024 * 1024) return false;
    const workspace = gemvWorkspace(T, workspace_len) orelse return false;
    @memset(workspace, 0);

    var tasks: [core_pool.max_tasks]GemvNoTransColumnTask(T) = undefined;
    for (0..task_count) |task_index| {
        const n0 = task_index * n / task_count;
        const n1 = (task_index + 1) * n / task_count;
        tasks[task_index] = .{
            .m = m,
            .n0 = n0,
            .n1 = n1,
            .alpha = alpha,
            .a = a,
            .lda = lda,
            .x = x,
            .y_delta = workspace.ptr + task_index * m,
        };
    }

    const runner = if (T == f32) runGemvNoTransColumnTaskF32 else runGemvNoTransColumnTaskF64;
    if (!core_pool.run(runner, @ptrCast(&tasks), task_count)) return false;

    for (0..task_count) |task_index| {
        addUnitReal(T, m, workspace.ptr + task_index * m, y);
    }
    return true;
}

fn gemvTransUnitRealF64(m: usize, n: usize, alpha: f64, a: [*]const f64, lda: BlasInt, x: [*]const f64, y: [*]f64) void {
    const V = @Vector(2, f64);
    var j: usize = 0;
    while (j + 8 <= n) : (j += 8) {
        const cols = [_][*]const f64{
            a + matIndex(lda, 0, j + 0),
            a + matIndex(lda, 0, j + 1),
            a + matIndex(lda, 0, j + 2),
            a + matIndex(lda, 0, j + 3),
            a + matIndex(lda, 0, j + 4),
            a + matIndex(lda, 0, j + 5),
            a + matIndex(lda, 0, j + 6),
            a + matIndex(lda, 0, j + 7),
        };
        const zero_v: V = @splat(0);
        var acc0 = [_]V{zero_v} ** 8;
        var acc1 = [_]V{zero_v} ** 8;
        var i: usize = 0;
        while (i + 4 <= m) : (i += 4) {
            const x0 = loadF64x2(x, i);
            const x1 = loadF64x2(x, i + 2);
            inline for (0..8) |col| {
                acc0[col] = @mulAdd(V, loadF64x2(cols[col], i), x0, acc0[col]);
                acc1[col] = @mulAdd(V, loadF64x2(cols[col], i + 2), x1, acc1[col]);
            }
        }
        var sums: [8]f64 = undefined;
        inline for (0..8) |col| {
            sums[col] = @reduce(.Add, acc0[col] + acc1[col]);
        }
        while (i + 2 <= m) : (i += 2) {
            const xv = loadF64x2(x, i);
            inline for (0..8) |col| {
                sums[col] += @reduce(.Add, loadF64x2(cols[col], i) * xv);
            }
        }
        while (i < m) : (i += 1) {
            inline for (0..8) |col| {
                sums[col] = @mulAdd(f64, cols[col][i], x[i], sums[col]);
            }
        }
        inline for (0..8) |col| {
            y[j + col] = @mulAdd(f64, alpha, sums[col], y[j + col]);
        }
    }
    while (j < n) : (j += 1) {
        y[j] = @mulAdd(f64, alpha, dotUnitReal(f64, m, a + matIndex(lda, 0, j), x), y[j]);
    }
}

fn gemvTransUnitReal(comptime T: type, m: usize, n: usize, alpha: T, a: [*]const T, lda: BlasInt, x: [*]const T, y: [*]T) void {
    if (matrix_vector_kernels.gemvTransUnitReal(T, m, n, alpha, a, lda, x, y)) return;
    if (comptime T == f64) return gemvTransUnitRealF64(m, n, alpha, a, lda, x, y);

    const lane_count = lanes(T);
    const unroll_count = unroll(T);
    const V = @Vector(lane_count, T);
    var j: usize = 0;
    while (j + 4 <= n) : (j += 4) {
        const c0 = a + matIndex(lda, 0, j);
        const c1 = a + matIndex(lda, 0, j + 1);
        const c2 = a + matIndex(lda, 0, j + 2);
        const c3 = a + matIndex(lda, 0, j + 3);
        var acc00: V = @splat(0);
        var acc01: V = @splat(0);
        var acc02: V = @splat(0);
        var acc03: V = @splat(0);
        var acc10: V = @splat(0);
        var acc11: V = @splat(0);
        var acc12: V = @splat(0);
        var acc13: V = @splat(0);
        var acc20: V = @splat(0);
        var acc21: V = @splat(0);
        var acc22: V = @splat(0);
        var acc23: V = @splat(0);
        var acc30: V = @splat(0);
        var acc31: V = @splat(0);
        var acc32: V = @splat(0);
        var acc33: V = @splat(0);
        var i: usize = 0;
        while (i + unroll_count <= m) : (i += unroll_count) {
            const x0 = loadVec(T, lane_count, x, i);
            const x1 = loadVec(T, lane_count, x, i + lane_count);
            const x2 = loadVec(T, lane_count, x, i + 2 * lane_count);
            const x3 = loadVec(T, lane_count, x, i + 3 * lane_count);
            acc00 = @mulAdd(V, loadVec(T, lane_count, c0, i), x0, acc00);
            acc01 = @mulAdd(V, loadVec(T, lane_count, c1, i), x0, acc01);
            acc02 = @mulAdd(V, loadVec(T, lane_count, c2, i), x0, acc02);
            acc03 = @mulAdd(V, loadVec(T, lane_count, c3, i), x0, acc03);
            acc10 = @mulAdd(V, loadVec(T, lane_count, c0, i + lane_count), x1, acc10);
            acc11 = @mulAdd(V, loadVec(T, lane_count, c1, i + lane_count), x1, acc11);
            acc12 = @mulAdd(V, loadVec(T, lane_count, c2, i + lane_count), x1, acc12);
            acc13 = @mulAdd(V, loadVec(T, lane_count, c3, i + lane_count), x1, acc13);
            acc20 = @mulAdd(V, loadVec(T, lane_count, c0, i + 2 * lane_count), x2, acc20);
            acc21 = @mulAdd(V, loadVec(T, lane_count, c1, i + 2 * lane_count), x2, acc21);
            acc22 = @mulAdd(V, loadVec(T, lane_count, c2, i + 2 * lane_count), x2, acc22);
            acc23 = @mulAdd(V, loadVec(T, lane_count, c3, i + 2 * lane_count), x2, acc23);
            acc30 = @mulAdd(V, loadVec(T, lane_count, c0, i + 3 * lane_count), x3, acc30);
            acc31 = @mulAdd(V, loadVec(T, lane_count, c1, i + 3 * lane_count), x3, acc31);
            acc32 = @mulAdd(V, loadVec(T, lane_count, c2, i + 3 * lane_count), x3, acc32);
            acc33 = @mulAdd(V, loadVec(T, lane_count, c3, i + 3 * lane_count), x3, acc33);
        }
        var sum0: T = @reduce(.Add, acc00 + acc10 + acc20 + acc30);
        var sum1: T = @reduce(.Add, acc01 + acc11 + acc21 + acc31);
        var sum2: T = @reduce(.Add, acc02 + acc12 + acc22 + acc32);
        var sum3: T = @reduce(.Add, acc03 + acc13 + acc23 + acc33);
        while (i + lane_count <= m) : (i += lane_count) {
            const xv = loadVec(T, lane_count, x, i);
            sum0 += @reduce(.Add, loadVec(T, lane_count, c0, i) * xv);
            sum1 += @reduce(.Add, loadVec(T, lane_count, c1, i) * xv);
            sum2 += @reduce(.Add, loadVec(T, lane_count, c2, i) * xv);
            sum3 += @reduce(.Add, loadVec(T, lane_count, c3, i) * xv);
        }
        while (i < m) : (i += 1) {
            sum0 = @mulAdd(T, c0[i], x[i], sum0);
            sum1 = @mulAdd(T, c1[i], x[i], sum1);
            sum2 = @mulAdd(T, c2[i], x[i], sum2);
            sum3 = @mulAdd(T, c3[i], x[i], sum3);
        }
        y[j] = @mulAdd(T, alpha, sum0, y[j]);
        y[j + 1] = @mulAdd(T, alpha, sum1, y[j + 1]);
        y[j + 2] = @mulAdd(T, alpha, sum2, y[j + 2]);
        y[j + 3] = @mulAdd(T, alpha, sum3, y[j + 3]);
    }
    while (j < n) : (j += 1) {
        y[j] = @mulAdd(T, alpha, dotUnitReal(T, m, a + matIndex(lda, 0, j), x), y[j]);
    }
}

fn GemvTransTask(comptime T: type) type {
    return struct {
        m: usize,
        n0: usize,
        n1: usize,
        alpha: T,
        a: [*]const T,
        lda: BlasInt,
        x: [*]const T,
        y: [*]T,
    };
}

fn runGemvTransTask(comptime T: type, raw_tasks: *const anyopaque, index: usize) void {
    const tasks: [*]const GemvTransTask(T) = @ptrCast(@alignCast(raw_tasks));
    const task = tasks[index];
    gemvTransUnitReal(T, task.m, task.n1 - task.n0, task.alpha, task.a + matIndex(task.lda, 0, task.n0), task.lda, task.x, task.y + task.n0);
}

fn runGemvTransTaskF32(raw_tasks: *const anyopaque, index: usize) void {
    runGemvTransTask(f32, raw_tasks, index);
}

fn runGemvTransTaskF64(raw_tasks: *const anyopaque, index: usize) void {
    runGemvTransTask(f64, raw_tasks, index);
}

fn parallelGemvTransUnitReal(comptime T: type, m: usize, n: usize, alpha: T, a: [*]const T, lda: BlasInt, x: [*]const T, y: [*]T) bool {
    const min_work: usize = if (T == f32) 512 * 512 else 768 * 768;
    if (m *| n < min_work) return false;
    const min_cols_per_task: usize = if (T == f32 and n <= 256) 64 else 64;
    var task_count = core_pool.taskCount(n, min_cols_per_task);
    if (n <= 1536) task_count = @min(task_count, 8);
    const block_cols: usize = 16;
    const block_count = n / block_cols;
    if (block_count > 0) task_count = @min(task_count, block_count);
    if (task_count <= 1) return false;

    var tasks: [core_pool.max_tasks]GemvTransTask(T) = undefined;
    for (0..task_count) |task_index| {
        const n0, const n1 = if (block_count > 0) .{
            (task_index * block_count / task_count) * block_cols,
            if (task_index + 1 == task_count) n else ((task_index + 1) * block_count / task_count) * block_cols,
        } else .{
            task_index * n / task_count,
            (task_index + 1) * n / task_count,
        };
        tasks[task_index] = .{
            .m = m,
            .n0 = n0,
            .n1 = n1,
            .alpha = alpha,
            .a = a,
            .lda = lda,
            .x = x,
            .y = y,
        };
    }

    const runner = if (T == f32) runGemvTransTaskF32 else runGemvTransTaskF64;
    if (T == f32 and n < 1536) return core_pool.runLowLatency(runner, @ptrCast(&tasks), task_count);
    return core_pool.run(runner, @ptrCast(&tasks), task_count);
}

fn gemvUnitReal(comptime T: type, trans_: Order, m: usize, n: usize, alpha: T, a: [*]const T, lda: BlasInt, x: [*]const T, beta: T, y: [*]T) void {
    if (trans_ == .no_trans) {
        if (matrix_vector_kernels.gemvNoTransFullUnitReal(T, m, n, alpha, a, lda, x, beta, y)) return;
    } else {
        if (matrix_vector_kernels.gemvTransFullUnitReal(T, m, n, alpha, a, lda, x, beta, y)) return;
    }

    const leny = if (trans_ == .no_trans) m else n;
    scaleUnitReal(T, leny, beta, y);
    if (alpha == 0) return;

    if (trans_ == .no_trans) {
        if (matrix_vector_kernels.gemvNoTransUnitReal(T, m, n, alpha, a, lda, x, y)) return;
        if (parallelGemvNoTransPackedRowsUnitReal(T, m, n, alpha, a, lda, x, y)) return;
        if (parallelGemvNoTransColumnsUnitReal(T, m, n, alpha, a, lda, x, y)) return;
        if (parallelGemvNoTransUnitReal(T, m, n, alpha, a, lda, x, y)) return;
        gemvNoTransUnitReal(T, m, n, alpha, a, lda, x, y);
    } else {
        if (parallelGemvTransUnitReal(T, m, n, alpha, a, lda, x, y)) return;
        gemvTransUnitReal(T, m, n, alpha, a, lda, x, y);
    }
}

fn gemvNoTransUnitComplex(comptime T: type, m_: BlasInt, n: usize, alpha: T, a: [*]const T, lda: BlasInt, x: [*]const T, y: [*]T) void {
    if (T == scalar.ComplexF32) return gemvNoTransUnitComplexC32(toUsize(m_), n, alpha, a, lda, x, y);
    if (T == scalar.ComplexF64) return gemvNoTransUnitComplexC64(toUsize(m_), n, alpha, a, lda, x, y);
    for (0..n) |j| {
        const xj = mul(T, alpha, x[j]);
        if (!isZero(T, xj)) level1.axpy(T, m_, xj, a + matIndex(lda, 0, j), 1, y, 1);
    }
}

inline fn c32AxpyTerm(xv: @Vector(8, f32), coeff: scalar.ComplexF32) @Vector(8, f32) {
    const swap_mask: @Vector(8, i32) = .{ 1, 0, 3, 2, 5, 4, 7, 6 };
    const re_v: @Vector(8, f32) = @splat(coeff.re);
    const im_v: @Vector(8, f32) = .{ -coeff.im, coeff.im, -coeff.im, coeff.im, -coeff.im, coeff.im, -coeff.im, coeff.im };
    return @mulAdd(@Vector(8, f32), xv, re_v, @shuffle(f32, xv, undefined, swap_mask) * im_v);
}

fn c32Axpy4(m: usize, c0: scalar.ComplexF32, a0: [*]const scalar.ComplexF32, c1: scalar.ComplexF32, a1: [*]const scalar.ComplexF32, c2: scalar.ComplexF32, a2: [*]const scalar.ComplexF32, c3: scalar.ComplexF32, a3: [*]const scalar.ComplexF32, y: [*]scalar.ComplexF32) void {
    const real_y: [*]f32 = @ptrCast(y);
    const r0: [*]const f32 = @ptrCast(a0);
    const r1: [*]const f32 = @ptrCast(a1);
    const r2: [*]const f32 = @ptrCast(a2);
    const r3: [*]const f32 = @ptrCast(a3);
    const real_n = 2 * m;
    var i: usize = 0;
    while (i + 16 <= real_n) : (i += 16) {
        var yv0 = loadVec(f32, 8, real_y, i);
        var yv1 = loadVec(f32, 8, real_y, i + 8);
        yv0 += c32AxpyTerm(loadVec(f32, 8, r0, i), c0);
        yv1 += c32AxpyTerm(loadVec(f32, 8, r0, i + 8), c0);
        yv0 += c32AxpyTerm(loadVec(f32, 8, r1, i), c1);
        yv1 += c32AxpyTerm(loadVec(f32, 8, r1, i + 8), c1);
        yv0 += c32AxpyTerm(loadVec(f32, 8, r2, i), c2);
        yv1 += c32AxpyTerm(loadVec(f32, 8, r2, i + 8), c2);
        yv0 += c32AxpyTerm(loadVec(f32, 8, r3, i), c3);
        yv1 += c32AxpyTerm(loadVec(f32, 8, r3, i + 8), c3);
        storeVec(f32, 8, real_y, i, yv0);
        storeVec(f32, 8, real_y, i + 8, yv1);
    }
    while (i + 8 <= real_n) : (i += 8) {
        var yv = loadVec(f32, 8, real_y, i);
        yv += c32AxpyTerm(loadVec(f32, 8, r0, i), c0);
        yv += c32AxpyTerm(loadVec(f32, 8, r1, i), c1);
        yv += c32AxpyTerm(loadVec(f32, 8, r2, i), c2);
        yv += c32AxpyTerm(loadVec(f32, 8, r3, i), c3);
        storeVec(f32, 8, real_y, i, yv);
    }
    while (i < real_n) : (i += 2) {
        const ar0 = r0[i];
        const ai0 = r0[i + 1];
        const ar1 = r1[i];
        const ai1 = r1[i + 1];
        const ar2 = r2[i];
        const ai2 = r2[i + 1];
        const ar3 = r3[i];
        const ai3 = r3[i + 1];
        real_y[i] += c0.re * ar0 - c0.im * ai0 + c1.re * ar1 - c1.im * ai1 + c2.re * ar2 - c2.im * ai2 + c3.re * ar3 - c3.im * ai3;
        real_y[i + 1] += c0.re * ai0 + c0.im * ar0 + c1.re * ai1 + c1.im * ar1 + c2.re * ai2 + c2.im * ar2 + c3.re * ai3 + c3.im * ar3;
    }
}

fn c32Axpy8(m: usize, coeffs: *const [8]scalar.ComplexF32, cols: *const [8][*]const scalar.ComplexF32, y: [*]scalar.ComplexF32) void {
    const real_y: [*]f32 = @ptrCast(y);
    const r0: [*]const f32 = @ptrCast(cols[0]);
    const r1: [*]const f32 = @ptrCast(cols[1]);
    const r2: [*]const f32 = @ptrCast(cols[2]);
    const r3: [*]const f32 = @ptrCast(cols[3]);
    const r4: [*]const f32 = @ptrCast(cols[4]);
    const r5: [*]const f32 = @ptrCast(cols[5]);
    const r6: [*]const f32 = @ptrCast(cols[6]);
    const r7: [*]const f32 = @ptrCast(cols[7]);
    const real_n = 2 * m;
    var i: usize = 0;
    while (i + 16 <= real_n) : (i += 16) {
        var yv0 = loadVec(f32, 8, real_y, i);
        var yv1 = loadVec(f32, 8, real_y, i + 8);
        yv0 += c32AxpyTerm(loadVec(f32, 8, r0, i), coeffs[0]);
        yv1 += c32AxpyTerm(loadVec(f32, 8, r0, i + 8), coeffs[0]);
        yv0 += c32AxpyTerm(loadVec(f32, 8, r1, i), coeffs[1]);
        yv1 += c32AxpyTerm(loadVec(f32, 8, r1, i + 8), coeffs[1]);
        yv0 += c32AxpyTerm(loadVec(f32, 8, r2, i), coeffs[2]);
        yv1 += c32AxpyTerm(loadVec(f32, 8, r2, i + 8), coeffs[2]);
        yv0 += c32AxpyTerm(loadVec(f32, 8, r3, i), coeffs[3]);
        yv1 += c32AxpyTerm(loadVec(f32, 8, r3, i + 8), coeffs[3]);
        yv0 += c32AxpyTerm(loadVec(f32, 8, r4, i), coeffs[4]);
        yv1 += c32AxpyTerm(loadVec(f32, 8, r4, i + 8), coeffs[4]);
        yv0 += c32AxpyTerm(loadVec(f32, 8, r5, i), coeffs[5]);
        yv1 += c32AxpyTerm(loadVec(f32, 8, r5, i + 8), coeffs[5]);
        yv0 += c32AxpyTerm(loadVec(f32, 8, r6, i), coeffs[6]);
        yv1 += c32AxpyTerm(loadVec(f32, 8, r6, i + 8), coeffs[6]);
        yv0 += c32AxpyTerm(loadVec(f32, 8, r7, i), coeffs[7]);
        yv1 += c32AxpyTerm(loadVec(f32, 8, r7, i + 8), coeffs[7]);
        storeVec(f32, 8, real_y, i, yv0);
        storeVec(f32, 8, real_y, i + 8, yv1);
    }
    while (i + 8 <= real_n) : (i += 8) {
        var yv = loadVec(f32, 8, real_y, i);
        yv += c32AxpyTerm(loadVec(f32, 8, r0, i), coeffs[0]);
        yv += c32AxpyTerm(loadVec(f32, 8, r1, i), coeffs[1]);
        yv += c32AxpyTerm(loadVec(f32, 8, r2, i), coeffs[2]);
        yv += c32AxpyTerm(loadVec(f32, 8, r3, i), coeffs[3]);
        yv += c32AxpyTerm(loadVec(f32, 8, r4, i), coeffs[4]);
        yv += c32AxpyTerm(loadVec(f32, 8, r5, i), coeffs[5]);
        yv += c32AxpyTerm(loadVec(f32, 8, r6, i), coeffs[6]);
        yv += c32AxpyTerm(loadVec(f32, 8, r7, i), coeffs[7]);
        storeVec(f32, 8, real_y, i, yv);
    }
    while (i < real_n) : (i += 2) {
        const ar0 = r0[i];
        const ai0 = r0[i + 1];
        const ar1 = r1[i];
        const ai1 = r1[i + 1];
        const ar2 = r2[i];
        const ai2 = r2[i + 1];
        const ar3 = r3[i];
        const ai3 = r3[i + 1];
        const ar4 = r4[i];
        const ai4 = r4[i + 1];
        const ar5 = r5[i];
        const ai5 = r5[i + 1];
        const ar6 = r6[i];
        const ai6 = r6[i + 1];
        const ar7 = r7[i];
        const ai7 = r7[i + 1];
        real_y[i] += coeffs[0].re * ar0 - coeffs[0].im * ai0 + coeffs[1].re * ar1 - coeffs[1].im * ai1 + coeffs[2].re * ar2 - coeffs[2].im * ai2 + coeffs[3].re * ar3 - coeffs[3].im * ai3 + coeffs[4].re * ar4 - coeffs[4].im * ai4 + coeffs[5].re * ar5 - coeffs[5].im * ai5 + coeffs[6].re * ar6 - coeffs[6].im * ai6 + coeffs[7].re * ar7 - coeffs[7].im * ai7;
        real_y[i + 1] += coeffs[0].re * ai0 + coeffs[0].im * ar0 + coeffs[1].re * ai1 + coeffs[1].im * ar1 + coeffs[2].re * ai2 + coeffs[2].im * ar2 + coeffs[3].re * ai3 + coeffs[3].im * ar3 + coeffs[4].re * ai4 + coeffs[4].im * ar4 + coeffs[5].re * ai5 + coeffs[5].im * ar5 + coeffs[6].re * ai6 + coeffs[6].im * ar6 + coeffs[7].re * ai7 + coeffs[7].im * ar7;
    }
}

fn c32Axpy1(m: usize, coeff: scalar.ComplexF32, a_col: [*]const scalar.ComplexF32, y: [*]scalar.ComplexF32) void {
    const real_y: [*]f32 = @ptrCast(y);
    const real_a: [*]const f32 = @ptrCast(a_col);
    const real_n = 2 * m;
    var i: usize = 0;
    while (i + 8 <= real_n) : (i += 8) {
        const yv = loadVec(f32, 8, real_y, i);
        storeVec(f32, 8, real_y, i, yv + c32AxpyTerm(loadVec(f32, 8, real_a, i), coeff));
    }
    while (i < real_n) : (i += 2) {
        const ar = real_a[i];
        const ai = real_a[i + 1];
        real_y[i] += coeff.re * ar - coeff.im * ai;
        real_y[i + 1] += coeff.re * ai + coeff.im * ar;
    }
}

fn gemvNoTransUnitComplexC32(m: usize, n: usize, alpha: scalar.ComplexF32, a: [*]const scalar.ComplexF32, lda: BlasInt, x: [*]const scalar.ComplexF32, y: [*]scalar.ComplexF32) void {
    var j: usize = 0;
    if (m == 128 and n == 128) {
        while (j + 8 <= n) : (j += 8) {
            const coeffs = [8]scalar.ComplexF32{
                mul(scalar.ComplexF32, alpha, x[j]),
                mul(scalar.ComplexF32, alpha, x[j + 1]),
                mul(scalar.ComplexF32, alpha, x[j + 2]),
                mul(scalar.ComplexF32, alpha, x[j + 3]),
                mul(scalar.ComplexF32, alpha, x[j + 4]),
                mul(scalar.ComplexF32, alpha, x[j + 5]),
                mul(scalar.ComplexF32, alpha, x[j + 6]),
                mul(scalar.ComplexF32, alpha, x[j + 7]),
            };
            const cols = [8][*]const scalar.ComplexF32{
                a + matIndex(lda, 0, j),
                a + matIndex(lda, 0, j + 1),
                a + matIndex(lda, 0, j + 2),
                a + matIndex(lda, 0, j + 3),
                a + matIndex(lda, 0, j + 4),
                a + matIndex(lda, 0, j + 5),
                a + matIndex(lda, 0, j + 6),
                a + matIndex(lda, 0, j + 7),
            };
            c32Axpy8(m, &coeffs, &cols, y);
        }
        return;
    }
    while (j + 4 <= n) : (j += 4) {
        c32Axpy4(
            m,
            mul(scalar.ComplexF32, alpha, x[j]),
            a + matIndex(lda, 0, j),
            mul(scalar.ComplexF32, alpha, x[j + 1]),
            a + matIndex(lda, 0, j + 1),
            mul(scalar.ComplexF32, alpha, x[j + 2]),
            a + matIndex(lda, 0, j + 2),
            mul(scalar.ComplexF32, alpha, x[j + 3]),
            a + matIndex(lda, 0, j + 3),
            y,
        );
    }
    while (j < n) : (j += 1) c32Axpy1(m, mul(scalar.ComplexF32, alpha, x[j]), a + matIndex(lda, 0, j), y);
}

inline fn c64AxpyTerm(xv: @Vector(4, f64), coeff: scalar.ComplexF64) @Vector(4, f64) {
    const swap_mask: @Vector(4, i32) = .{ 1, 0, 3, 2 };
    const re_v: @Vector(4, f64) = @splat(coeff.re);
    const im_v: @Vector(4, f64) = .{ -coeff.im, coeff.im, -coeff.im, coeff.im };
    return @mulAdd(@Vector(4, f64), xv, re_v, @shuffle(f64, xv, undefined, swap_mask) * im_v);
}

fn c64Axpy4(m: usize, c0: scalar.ComplexF64, a0: [*]const scalar.ComplexF64, c1: scalar.ComplexF64, a1: [*]const scalar.ComplexF64, c2: scalar.ComplexF64, a2: [*]const scalar.ComplexF64, c3: scalar.ComplexF64, a3: [*]const scalar.ComplexF64, y: [*]scalar.ComplexF64) void {
    const real_y: [*]f64 = @ptrCast(y);
    const r0: [*]const f64 = @ptrCast(a0);
    const r1: [*]const f64 = @ptrCast(a1);
    const r2: [*]const f64 = @ptrCast(a2);
    const r3: [*]const f64 = @ptrCast(a3);
    const real_n = 2 * m;
    var i: usize = 0;
    while (i + 8 <= real_n) : (i += 8) {
        var yv0 = loadVec(f64, 4, real_y, i);
        var yv1 = loadVec(f64, 4, real_y, i + 4);
        yv0 += c64AxpyTerm(loadVec(f64, 4, r0, i), c0);
        yv1 += c64AxpyTerm(loadVec(f64, 4, r0, i + 4), c0);
        yv0 += c64AxpyTerm(loadVec(f64, 4, r1, i), c1);
        yv1 += c64AxpyTerm(loadVec(f64, 4, r1, i + 4), c1);
        yv0 += c64AxpyTerm(loadVec(f64, 4, r2, i), c2);
        yv1 += c64AxpyTerm(loadVec(f64, 4, r2, i + 4), c2);
        yv0 += c64AxpyTerm(loadVec(f64, 4, r3, i), c3);
        yv1 += c64AxpyTerm(loadVec(f64, 4, r3, i + 4), c3);
        storeVec(f64, 4, real_y, i, yv0);
        storeVec(f64, 4, real_y, i + 4, yv1);
    }
    while (i + 4 <= real_n) : (i += 4) {
        var yv = loadVec(f64, 4, real_y, i);
        yv += c64AxpyTerm(loadVec(f64, 4, r0, i), c0);
        yv += c64AxpyTerm(loadVec(f64, 4, r1, i), c1);
        yv += c64AxpyTerm(loadVec(f64, 4, r2, i), c2);
        yv += c64AxpyTerm(loadVec(f64, 4, r3, i), c3);
        storeVec(f64, 4, real_y, i, yv);
    }
    while (i < real_n) : (i += 2) {
        const ar0 = r0[i];
        const ai0 = r0[i + 1];
        const ar1 = r1[i];
        const ai1 = r1[i + 1];
        const ar2 = r2[i];
        const ai2 = r2[i + 1];
        const ar3 = r3[i];
        const ai3 = r3[i + 1];
        real_y[i] += c0.re * ar0 - c0.im * ai0 + c1.re * ar1 - c1.im * ai1 + c2.re * ar2 - c2.im * ai2 + c3.re * ar3 - c3.im * ai3;
        real_y[i + 1] += c0.re * ai0 + c0.im * ar0 + c1.re * ai1 + c1.im * ar1 + c2.re * ai2 + c2.im * ar2 + c3.re * ai3 + c3.im * ar3;
    }
}

fn c64Axpy1(m: usize, coeff: scalar.ComplexF64, a_col: [*]const scalar.ComplexF64, y: [*]scalar.ComplexF64) void {
    const real_y: [*]f64 = @ptrCast(y);
    const real_a: [*]const f64 = @ptrCast(a_col);
    const real_n = 2 * m;
    var i: usize = 0;
    while (i + 4 <= real_n) : (i += 4) {
        const yv = loadVec(f64, 4, real_y, i);
        storeVec(f64, 4, real_y, i, yv + c64AxpyTerm(loadVec(f64, 4, real_a, i), coeff));
    }
    while (i < real_n) : (i += 2) {
        const ar = real_a[i];
        const ai = real_a[i + 1];
        real_y[i] += coeff.re * ar - coeff.im * ai;
        real_y[i + 1] += coeff.re * ai + coeff.im * ar;
    }
}

fn gemvNoTransUnitComplexC64(m: usize, n: usize, alpha: scalar.ComplexF64, a: [*]const scalar.ComplexF64, lda: BlasInt, x: [*]const scalar.ComplexF64, y: [*]scalar.ComplexF64) void {
    var j: usize = 0;
    while (j + 4 <= n) : (j += 4) {
        c64Axpy4(
            m,
            mul(scalar.ComplexF64, alpha, x[j]),
            a + matIndex(lda, 0, j),
            mul(scalar.ComplexF64, alpha, x[j + 1]),
            a + matIndex(lda, 0, j + 1),
            mul(scalar.ComplexF64, alpha, x[j + 2]),
            a + matIndex(lda, 0, j + 2),
            mul(scalar.ComplexF64, alpha, x[j + 3]),
            a + matIndex(lda, 0, j + 3),
            y,
        );
    }
    while (j < n) : (j += 1) c64Axpy1(m, mul(scalar.ComplexF64, alpha, x[j]), a + matIndex(lda, 0, j), y);
}

fn GemvNoTransComplexTask(comptime T: type) type {
    return struct {
        m: BlasInt,
        n0: usize,
        n1: usize,
        alpha: T,
        a: [*]const T,
        lda: BlasInt,
        x: [*]const T,
        y_delta: [*]T,
    };
}

fn runGemvNoTransComplexTask(comptime T: type, raw_tasks: *const anyopaque, index: usize) void {
    const tasks: [*]const GemvNoTransComplexTask(T) = @ptrCast(@alignCast(raw_tasks));
    const task = tasks[index];
    gemvNoTransUnitComplex(T, task.m, task.n1 - task.n0, task.alpha, task.a + matIndex(task.lda, 0, task.n0), task.lda, task.x + task.n0, task.y_delta);
}

fn runGemvNoTransComplexTaskC32(raw_tasks: *const anyopaque, index: usize) void {
    runGemvNoTransComplexTask(scalar.ComplexF32, raw_tasks, index);
}

fn runGemvNoTransComplexTaskC64(raw_tasks: *const anyopaque, index: usize) void {
    runGemvNoTransComplexTask(scalar.ComplexF64, raw_tasks, index);
}

fn parallelGemvNoTransUnitComplex(comptime T: type, m: usize, n: usize, m_: BlasInt, alpha: T, a: [*]const T, lda: BlasInt, x: [*]const T, y: [*]T) bool {
    if (T == scalar.ComplexF32 and m == 128 and n == 128) return false;
    if (m *| n < 128 * 128) return false;
    const min_cols_per_task: usize = if (T == scalar.ComplexF32) 64 else 48;
    var task_count = core_pool.taskCount(n, min_cols_per_task);
    const max_task_count: usize = if (T == scalar.ComplexF64 and n >= 256 and n < 512) 10 else if (n < 512) 4 else if (T == scalar.ComplexF32) 10 else 8;
    task_count = @min(task_count, max_task_count);
    if (task_count <= 1) return false;

    const workspace_len = task_count * m;
    if (workspace_len * @sizeOf(T) > 64 * 1024 * 1024) return false;
    const workspace = gemvWorkspace(T, workspace_len) orelse return false;
    @memset(workspace, zero(T));

    var tasks: [core_pool.max_tasks]GemvNoTransComplexTask(T) = undefined;
    for (0..task_count) |task_index| {
        const n0 = task_index * n / task_count;
        const n1 = (task_index + 1) * n / task_count;
        tasks[task_index] = .{
            .m = m_,
            .n0 = n0,
            .n1 = n1,
            .alpha = alpha,
            .a = a,
            .lda = lda,
            .x = x,
            .y_delta = workspace.ptr + task_index * m,
        };
    }

    const runner = if (T == scalar.ComplexF32) runGemvNoTransComplexTaskC32 else runGemvNoTransComplexTaskC64;
    if (!core_pool.runLowLatency(runner, @ptrCast(&tasks), task_count)) return false;

    const add_alpha = one(T);
    for (0..task_count) |task_index| {
        level1.axpy(T, m_, add_alpha, workspace.ptr + task_index * m, 1, y, 1);
    }
    return true;
}

fn gemvTransUnitComplex(comptime T: type, m_: BlasInt, n: usize, alpha: T, a: [*]const T, lda: BlasInt, x: [*]const T, y: [*]T, do_conj: bool) void {
    if (T == scalar.ComplexF32) return gemvTransUnitComplexC32(toUsize(m_), n, alpha, a, lda, x, y, do_conj);
    if (T == scalar.ComplexF64) return gemvTransUnitComplexC64(toUsize(m_), n, alpha, a, lda, x, y, do_conj);
    for (0..n) |j| {
        const sum = level1.dot(T, m_, a + matIndex(lda, 0, j), 1, x, 1, do_conj);
        y[j] = add(T, y[j], mul(T, alpha, sum));
    }
}

inline fn c32DotAccumulateVec(
    a_real: [*]const f32,
    x_real: [*]const f32,
    offset: usize,
    re_sign: @Vector(8, f32),
    im_sign: @Vector(8, f32),
    re_acc: *@Vector(8, f32),
    im_acc: *@Vector(8, f32),
) void {
    const swap_mask: @Vector(8, i32) = .{ 1, 0, 3, 2, 5, 4, 7, 6 };
    const av = loadVec(f32, 8, a_real, offset);
    const xv = loadVec(f32, 8, x_real, offset);
    const x_swap = @shuffle(f32, xv, undefined, swap_mask);
    re_acc.* = @mulAdd(@Vector(8, f32), av * xv, re_sign, re_acc.*);
    im_acc.* = @mulAdd(@Vector(8, f32), av * x_swap, im_sign, im_acc.*);
}

inline fn c32Dot2AccumulateVec(
    a0_real: [*]const f32,
    a1_real: [*]const f32,
    x_real: [*]const f32,
    offset: usize,
    re_sign: @Vector(8, f32),
    im_sign: @Vector(8, f32),
    re_acc0: *@Vector(8, f32),
    im_acc0: *@Vector(8, f32),
    re_acc1: *@Vector(8, f32),
    im_acc1: *@Vector(8, f32),
) void {
    const swap_mask: @Vector(8, i32) = .{ 1, 0, 3, 2, 5, 4, 7, 6 };
    const xv = loadVec(f32, 8, x_real, offset);
    const x_swap = @shuffle(f32, xv, undefined, swap_mask);
    const a0v = loadVec(f32, 8, a0_real, offset);
    const a1v = loadVec(f32, 8, a1_real, offset);
    re_acc0.* = @mulAdd(@Vector(8, f32), a0v * xv, re_sign, re_acc0.*);
    im_acc0.* = @mulAdd(@Vector(8, f32), a0v * x_swap, im_sign, im_acc0.*);
    re_acc1.* = @mulAdd(@Vector(8, f32), a1v * xv, re_sign, re_acc1.*);
    im_acc1.* = @mulAdd(@Vector(8, f32), a1v * x_swap, im_sign, im_acc1.*);
}

inline fn c32Dot4AccumulateVec(
    a0_real: [*]const f32,
    a1_real: [*]const f32,
    a2_real: [*]const f32,
    a3_real: [*]const f32,
    x_real: [*]const f32,
    offset: usize,
    re_sign: @Vector(8, f32),
    im_sign: @Vector(8, f32),
    re_acc0: *@Vector(8, f32),
    im_acc0: *@Vector(8, f32),
    re_acc1: *@Vector(8, f32),
    im_acc1: *@Vector(8, f32),
    re_acc2: *@Vector(8, f32),
    im_acc2: *@Vector(8, f32),
    re_acc3: *@Vector(8, f32),
    im_acc3: *@Vector(8, f32),
) void {
    const swap_mask: @Vector(8, i32) = .{ 1, 0, 3, 2, 5, 4, 7, 6 };
    const xv = loadVec(f32, 8, x_real, offset);
    const x_swap = @shuffle(f32, xv, undefined, swap_mask);
    const a0v = loadVec(f32, 8, a0_real, offset);
    const a1v = loadVec(f32, 8, a1_real, offset);
    const a2v = loadVec(f32, 8, a2_real, offset);
    const a3v = loadVec(f32, 8, a3_real, offset);
    re_acc0.* = @mulAdd(@Vector(8, f32), a0v * xv, re_sign, re_acc0.*);
    im_acc0.* = @mulAdd(@Vector(8, f32), a0v * x_swap, im_sign, im_acc0.*);
    re_acc1.* = @mulAdd(@Vector(8, f32), a1v * xv, re_sign, re_acc1.*);
    im_acc1.* = @mulAdd(@Vector(8, f32), a1v * x_swap, im_sign, im_acc1.*);
    re_acc2.* = @mulAdd(@Vector(8, f32), a2v * xv, re_sign, re_acc2.*);
    im_acc2.* = @mulAdd(@Vector(8, f32), a2v * x_swap, im_sign, im_acc2.*);
    re_acc3.* = @mulAdd(@Vector(8, f32), a3v * xv, re_sign, re_acc3.*);
    im_acc3.* = @mulAdd(@Vector(8, f32), a3v * x_swap, im_sign, im_acc3.*);
}

fn c32DotUnit(m: usize, a_col: [*]const scalar.ComplexF32, x: [*]const scalar.ComplexF32, do_conj: bool) scalar.ComplexF32 {
    const re_sign: @Vector(8, f32) = if (do_conj) @splat(1) else .{ 1, -1, 1, -1, 1, -1, 1, -1 };
    const im_sign: @Vector(8, f32) = if (do_conj) .{ 1, -1, 1, -1, 1, -1, 1, -1 } else @splat(1);
    const a_real: [*]const f32 = @ptrCast(a_col);
    const x_real: [*]const f32 = @ptrCast(x);
    const real_n = 2 * m;
    var re_acc0: @Vector(8, f32) = @splat(0);
    var re_acc1: @Vector(8, f32) = @splat(0);
    var im_acc0: @Vector(8, f32) = @splat(0);
    var im_acc1: @Vector(8, f32) = @splat(0);
    var i: usize = 0;
    while (i + 16 <= real_n) : (i += 16) {
        c32DotAccumulateVec(a_real, x_real, i, re_sign, im_sign, &re_acc0, &im_acc0);
        c32DotAccumulateVec(a_real, x_real, i + 8, re_sign, im_sign, &re_acc1, &im_acc1);
    }
    var re_acc = re_acc0 + re_acc1;
    var im_acc = im_acc0 + im_acc1;
    while (i + 8 <= real_n) : (i += 8) {
        c32DotAccumulateVec(a_real, x_real, i, re_sign, im_sign, &re_acc, &im_acc);
    }
    var re_sum: f32 = @reduce(.Add, re_acc);
    var im_sum: f32 = @reduce(.Add, im_acc);
    while (i < real_n) : (i += 2) {
        const ar = a_real[i];
        const ai = a_real[i + 1];
        const xr = x_real[i];
        const xi = x_real[i + 1];
        if (do_conj) {
            re_sum = @mulAdd(f32, ai, xi, @mulAdd(f32, ar, xr, re_sum));
            im_sum = @mulAdd(f32, -ai, xr, @mulAdd(f32, ar, xi, im_sum));
        } else {
            re_sum = @mulAdd(f32, -ai, xi, @mulAdd(f32, ar, xr, re_sum));
            im_sum = @mulAdd(f32, ai, xr, @mulAdd(f32, ar, xi, im_sum));
        }
    }
    return .{ .re = re_sum, .im = im_sum };
}

fn c32Dot2Unit(m: usize, a0_col: [*]const scalar.ComplexF32, a1_col: [*]const scalar.ComplexF32, x: [*]const scalar.ComplexF32, do_conj: bool) [2]scalar.ComplexF32 {
    const re_sign: @Vector(8, f32) = if (do_conj) @splat(1) else .{ 1, -1, 1, -1, 1, -1, 1, -1 };
    const im_sign: @Vector(8, f32) = if (do_conj) .{ 1, -1, 1, -1, 1, -1, 1, -1 } else @splat(1);
    const a0_real: [*]const f32 = @ptrCast(a0_col);
    const a1_real: [*]const f32 = @ptrCast(a1_col);
    const x_real: [*]const f32 = @ptrCast(x);
    const real_n = 2 * m;
    var re_acc00: @Vector(8, f32) = @splat(0);
    var re_acc01: @Vector(8, f32) = @splat(0);
    var im_acc00: @Vector(8, f32) = @splat(0);
    var im_acc01: @Vector(8, f32) = @splat(0);
    var re_acc10: @Vector(8, f32) = @splat(0);
    var re_acc11: @Vector(8, f32) = @splat(0);
    var im_acc10: @Vector(8, f32) = @splat(0);
    var im_acc11: @Vector(8, f32) = @splat(0);
    var i: usize = 0;
    while (i + 16 <= real_n) : (i += 16) {
        c32Dot2AccumulateVec(a0_real, a1_real, x_real, i, re_sign, im_sign, &re_acc00, &im_acc00, &re_acc10, &im_acc10);
        c32Dot2AccumulateVec(a0_real, a1_real, x_real, i + 8, re_sign, im_sign, &re_acc01, &im_acc01, &re_acc11, &im_acc11);
    }
    var re_acc0 = re_acc00 + re_acc01;
    var im_acc0 = im_acc00 + im_acc01;
    var re_acc1 = re_acc10 + re_acc11;
    var im_acc1 = im_acc10 + im_acc11;
    while (i + 8 <= real_n) : (i += 8) {
        c32Dot2AccumulateVec(a0_real, a1_real, x_real, i, re_sign, im_sign, &re_acc0, &im_acc0, &re_acc1, &im_acc1);
    }
    var re_sum0: f32 = @reduce(.Add, re_acc0);
    var im_sum0: f32 = @reduce(.Add, im_acc0);
    var re_sum1: f32 = @reduce(.Add, re_acc1);
    var im_sum1: f32 = @reduce(.Add, im_acc1);
    while (i < real_n) : (i += 2) {
        const ar0 = a0_real[i];
        const ai0 = a0_real[i + 1];
        const ar1 = a1_real[i];
        const ai1 = a1_real[i + 1];
        const xr = x_real[i];
        const xi = x_real[i + 1];
        if (do_conj) {
            re_sum0 = @mulAdd(f32, ai0, xi, @mulAdd(f32, ar0, xr, re_sum0));
            im_sum0 = @mulAdd(f32, -ai0, xr, @mulAdd(f32, ar0, xi, im_sum0));
            re_sum1 = @mulAdd(f32, ai1, xi, @mulAdd(f32, ar1, xr, re_sum1));
            im_sum1 = @mulAdd(f32, -ai1, xr, @mulAdd(f32, ar1, xi, im_sum1));
        } else {
            re_sum0 = @mulAdd(f32, -ai0, xi, @mulAdd(f32, ar0, xr, re_sum0));
            im_sum0 = @mulAdd(f32, ai0, xr, @mulAdd(f32, ar0, xi, im_sum0));
            re_sum1 = @mulAdd(f32, -ai1, xi, @mulAdd(f32, ar1, xr, re_sum1));
            im_sum1 = @mulAdd(f32, ai1, xr, @mulAdd(f32, ar1, xi, im_sum1));
        }
    }
    return .{
        .{ .re = re_sum0, .im = im_sum0 },
        .{ .re = re_sum1, .im = im_sum1 },
    };
}

fn c32Dot4Unit(m: usize, a0_col: [*]const scalar.ComplexF32, a1_col: [*]const scalar.ComplexF32, a2_col: [*]const scalar.ComplexF32, a3_col: [*]const scalar.ComplexF32, x: [*]const scalar.ComplexF32, do_conj: bool) [4]scalar.ComplexF32 {
    const re_sign: @Vector(8, f32) = if (do_conj) @splat(1) else .{ 1, -1, 1, -1, 1, -1, 1, -1 };
    const im_sign: @Vector(8, f32) = if (do_conj) .{ 1, -1, 1, -1, 1, -1, 1, -1 } else @splat(1);
    const a0_real: [*]const f32 = @ptrCast(a0_col);
    const a1_real: [*]const f32 = @ptrCast(a1_col);
    const a2_real: [*]const f32 = @ptrCast(a2_col);
    const a3_real: [*]const f32 = @ptrCast(a3_col);
    const x_real: [*]const f32 = @ptrCast(x);
    const real_n = 2 * m;
    var re_acc0: @Vector(8, f32) = @splat(0);
    var im_acc0: @Vector(8, f32) = @splat(0);
    var re_acc1: @Vector(8, f32) = @splat(0);
    var im_acc1: @Vector(8, f32) = @splat(0);
    var re_acc2: @Vector(8, f32) = @splat(0);
    var im_acc2: @Vector(8, f32) = @splat(0);
    var re_acc3: @Vector(8, f32) = @splat(0);
    var im_acc3: @Vector(8, f32) = @splat(0);
    var i: usize = 0;
    while (i + 8 <= real_n) : (i += 8) {
        c32Dot4AccumulateVec(a0_real, a1_real, a2_real, a3_real, x_real, i, re_sign, im_sign, &re_acc0, &im_acc0, &re_acc1, &im_acc1, &re_acc2, &im_acc2, &re_acc3, &im_acc3);
    }
    var re_sum0: f32 = @reduce(.Add, re_acc0);
    var im_sum0: f32 = @reduce(.Add, im_acc0);
    var re_sum1: f32 = @reduce(.Add, re_acc1);
    var im_sum1: f32 = @reduce(.Add, im_acc1);
    var re_sum2: f32 = @reduce(.Add, re_acc2);
    var im_sum2: f32 = @reduce(.Add, im_acc2);
    var re_sum3: f32 = @reduce(.Add, re_acc3);
    var im_sum3: f32 = @reduce(.Add, im_acc3);
    while (i < real_n) : (i += 2) {
        const ar0 = a0_real[i];
        const ai0 = a0_real[i + 1];
        const ar1 = a1_real[i];
        const ai1 = a1_real[i + 1];
        const ar2 = a2_real[i];
        const ai2 = a2_real[i + 1];
        const ar3 = a3_real[i];
        const ai3 = a3_real[i + 1];
        const xr = x_real[i];
        const xi = x_real[i + 1];
        if (do_conj) {
            re_sum0 = @mulAdd(f32, ai0, xi, @mulAdd(f32, ar0, xr, re_sum0));
            im_sum0 = @mulAdd(f32, -ai0, xr, @mulAdd(f32, ar0, xi, im_sum0));
            re_sum1 = @mulAdd(f32, ai1, xi, @mulAdd(f32, ar1, xr, re_sum1));
            im_sum1 = @mulAdd(f32, -ai1, xr, @mulAdd(f32, ar1, xi, im_sum1));
            re_sum2 = @mulAdd(f32, ai2, xi, @mulAdd(f32, ar2, xr, re_sum2));
            im_sum2 = @mulAdd(f32, -ai2, xr, @mulAdd(f32, ar2, xi, im_sum2));
            re_sum3 = @mulAdd(f32, ai3, xi, @mulAdd(f32, ar3, xr, re_sum3));
            im_sum3 = @mulAdd(f32, -ai3, xr, @mulAdd(f32, ar3, xi, im_sum3));
        } else {
            re_sum0 = @mulAdd(f32, -ai0, xi, @mulAdd(f32, ar0, xr, re_sum0));
            im_sum0 = @mulAdd(f32, ai0, xr, @mulAdd(f32, ar0, xi, im_sum0));
            re_sum1 = @mulAdd(f32, -ai1, xi, @mulAdd(f32, ar1, xr, re_sum1));
            im_sum1 = @mulAdd(f32, ai1, xr, @mulAdd(f32, ar1, xi, im_sum1));
            re_sum2 = @mulAdd(f32, -ai2, xi, @mulAdd(f32, ar2, xr, re_sum2));
            im_sum2 = @mulAdd(f32, ai2, xr, @mulAdd(f32, ar2, xi, im_sum2));
            re_sum3 = @mulAdd(f32, -ai3, xi, @mulAdd(f32, ar3, xr, re_sum3));
            im_sum3 = @mulAdd(f32, ai3, xr, @mulAdd(f32, ar3, xi, im_sum3));
        }
    }
    return .{
        .{ .re = re_sum0, .im = im_sum0 },
        .{ .re = re_sum1, .im = im_sum1 },
        .{ .re = re_sum2, .im = im_sum2 },
        .{ .re = re_sum3, .im = im_sum3 },
    };
}

fn gemvTransUnitComplexC32(m: usize, n: usize, alpha: scalar.ComplexF32, a: [*]const scalar.ComplexF32, lda: BlasInt, x: [*]const scalar.ComplexF32, y: [*]scalar.ComplexF32, do_conj: bool) void {
    var j: usize = 0;
    if (m == 128 or m == 256 or m >= 512) {
        while (j + 4 <= n) : (j += 4) {
            const sums = c32Dot4Unit(m, a + matIndex(lda, 0, j), a + matIndex(lda, 0, j + 1), a + matIndex(lda, 0, j + 2), a + matIndex(lda, 0, j + 3), x, do_conj);
            y[j] = add(scalar.ComplexF32, y[j], mul(scalar.ComplexF32, alpha, sums[0]));
            y[j + 1] = add(scalar.ComplexF32, y[j + 1], mul(scalar.ComplexF32, alpha, sums[1]));
            y[j + 2] = add(scalar.ComplexF32, y[j + 2], mul(scalar.ComplexF32, alpha, sums[2]));
            y[j + 3] = add(scalar.ComplexF32, y[j + 3], mul(scalar.ComplexF32, alpha, sums[3]));
        }
        while (j + 2 <= n) : (j += 2) {
            const sums = c32Dot2Unit(m, a + matIndex(lda, 0, j), a + matIndex(lda, 0, j + 1), x, do_conj);
            y[j] = add(scalar.ComplexF32, y[j], mul(scalar.ComplexF32, alpha, sums[0]));
            y[j + 1] = add(scalar.ComplexF32, y[j + 1], mul(scalar.ComplexF32, alpha, sums[1]));
        }
    } else {
        while (j + 2 <= n) : (j += 2) {
            const sum0 = c32DotUnit(m, a + matIndex(lda, 0, j), x, do_conj);
            const sum1 = c32DotUnit(m, a + matIndex(lda, 0, j + 1), x, do_conj);
            y[j] = add(scalar.ComplexF32, y[j], mul(scalar.ComplexF32, alpha, sum0));
            y[j + 1] = add(scalar.ComplexF32, y[j + 1], mul(scalar.ComplexF32, alpha, sum1));
        }
    }
    while (j < n) : (j += 1) {
        const sum = c32DotUnit(m, a + matIndex(lda, 0, j), x, do_conj);
        y[j] = add(scalar.ComplexF32, y[j], mul(scalar.ComplexF32, alpha, sum));
    }
}

inline fn c64DotAccumulateVec(
    a_real: [*]const f64,
    x_real: [*]const f64,
    offset: usize,
    re_sign: @Vector(4, f64),
    im_sign: @Vector(4, f64),
    re_acc: *@Vector(4, f64),
    im_acc: *@Vector(4, f64),
) void {
    const swap_mask: @Vector(4, i32) = .{ 1, 0, 3, 2 };
    const av = loadVec(f64, 4, a_real, offset);
    const xv = loadVec(f64, 4, x_real, offset);
    const x_swap = @shuffle(f64, xv, undefined, swap_mask);
    re_acc.* = @mulAdd(@Vector(4, f64), av * xv, re_sign, re_acc.*);
    im_acc.* = @mulAdd(@Vector(4, f64), av * x_swap, im_sign, im_acc.*);
}

inline fn c64Dot2AccumulateVec(
    a0_real: [*]const f64,
    a1_real: [*]const f64,
    x_real: [*]const f64,
    offset: usize,
    re_sign: @Vector(4, f64),
    im_sign: @Vector(4, f64),
    re_acc0: *@Vector(4, f64),
    im_acc0: *@Vector(4, f64),
    re_acc1: *@Vector(4, f64),
    im_acc1: *@Vector(4, f64),
) void {
    const swap_mask: @Vector(4, i32) = .{ 1, 0, 3, 2 };
    const xv = loadVec(f64, 4, x_real, offset);
    const x_swap = @shuffle(f64, xv, undefined, swap_mask);
    const a0v = loadVec(f64, 4, a0_real, offset);
    const a1v = loadVec(f64, 4, a1_real, offset);
    re_acc0.* = @mulAdd(@Vector(4, f64), a0v * xv, re_sign, re_acc0.*);
    im_acc0.* = @mulAdd(@Vector(4, f64), a0v * x_swap, im_sign, im_acc0.*);
    re_acc1.* = @mulAdd(@Vector(4, f64), a1v * xv, re_sign, re_acc1.*);
    im_acc1.* = @mulAdd(@Vector(4, f64), a1v * x_swap, im_sign, im_acc1.*);
}

inline fn c64Dot4AccumulateVec(
    a0_real: [*]const f64,
    a1_real: [*]const f64,
    a2_real: [*]const f64,
    a3_real: [*]const f64,
    x_real: [*]const f64,
    offset: usize,
    re_sign: @Vector(4, f64),
    im_sign: @Vector(4, f64),
    re_acc0: *@Vector(4, f64),
    im_acc0: *@Vector(4, f64),
    re_acc1: *@Vector(4, f64),
    im_acc1: *@Vector(4, f64),
    re_acc2: *@Vector(4, f64),
    im_acc2: *@Vector(4, f64),
    re_acc3: *@Vector(4, f64),
    im_acc3: *@Vector(4, f64),
) void {
    const swap_mask: @Vector(4, i32) = .{ 1, 0, 3, 2 };
    const xv = loadVec(f64, 4, x_real, offset);
    const x_swap = @shuffle(f64, xv, undefined, swap_mask);
    const a0v = loadVec(f64, 4, a0_real, offset);
    const a1v = loadVec(f64, 4, a1_real, offset);
    const a2v = loadVec(f64, 4, a2_real, offset);
    const a3v = loadVec(f64, 4, a3_real, offset);
    re_acc0.* = @mulAdd(@Vector(4, f64), a0v * xv, re_sign, re_acc0.*);
    im_acc0.* = @mulAdd(@Vector(4, f64), a0v * x_swap, im_sign, im_acc0.*);
    re_acc1.* = @mulAdd(@Vector(4, f64), a1v * xv, re_sign, re_acc1.*);
    im_acc1.* = @mulAdd(@Vector(4, f64), a1v * x_swap, im_sign, im_acc1.*);
    re_acc2.* = @mulAdd(@Vector(4, f64), a2v * xv, re_sign, re_acc2.*);
    im_acc2.* = @mulAdd(@Vector(4, f64), a2v * x_swap, im_sign, im_acc2.*);
    re_acc3.* = @mulAdd(@Vector(4, f64), a3v * xv, re_sign, re_acc3.*);
    im_acc3.* = @mulAdd(@Vector(4, f64), a3v * x_swap, im_sign, im_acc3.*);
}

fn c64DotUnit(m: usize, a_col: [*]const scalar.ComplexF64, x: [*]const scalar.ComplexF64, do_conj: bool) scalar.ComplexF64 {
    const re_sign: @Vector(4, f64) = if (do_conj) @splat(1) else .{ 1, -1, 1, -1 };
    const im_sign: @Vector(4, f64) = if (do_conj) .{ 1, -1, 1, -1 } else @splat(1);
    const a_real: [*]const f64 = @ptrCast(a_col);
    const x_real: [*]const f64 = @ptrCast(x);
    const real_n = 2 * m;
    var re_acc0: @Vector(4, f64) = @splat(0);
    var re_acc1: @Vector(4, f64) = @splat(0);
    var im_acc0: @Vector(4, f64) = @splat(0);
    var im_acc1: @Vector(4, f64) = @splat(0);
    var i: usize = 0;
    while (i + 8 <= real_n) : (i += 8) {
        c64DotAccumulateVec(a_real, x_real, i, re_sign, im_sign, &re_acc0, &im_acc0);
        c64DotAccumulateVec(a_real, x_real, i + 4, re_sign, im_sign, &re_acc1, &im_acc1);
    }
    var re_acc = re_acc0 + re_acc1;
    var im_acc = im_acc0 + im_acc1;
    while (i + 4 <= real_n) : (i += 4) {
        c64DotAccumulateVec(a_real, x_real, i, re_sign, im_sign, &re_acc, &im_acc);
    }
    var re_sum: f64 = @reduce(.Add, re_acc);
    var im_sum: f64 = @reduce(.Add, im_acc);
    while (i < real_n) : (i += 2) {
        const ar = a_real[i];
        const ai = a_real[i + 1];
        const xr = x_real[i];
        const xi = x_real[i + 1];
        if (do_conj) {
            re_sum = @mulAdd(f64, ai, xi, @mulAdd(f64, ar, xr, re_sum));
            im_sum = @mulAdd(f64, -ai, xr, @mulAdd(f64, ar, xi, im_sum));
        } else {
            re_sum = @mulAdd(f64, -ai, xi, @mulAdd(f64, ar, xr, re_sum));
            im_sum = @mulAdd(f64, ai, xr, @mulAdd(f64, ar, xi, im_sum));
        }
    }
    return .{ .re = re_sum, .im = im_sum };
}

fn c64Dot2Unit(m: usize, a0_col: [*]const scalar.ComplexF64, a1_col: [*]const scalar.ComplexF64, x: [*]const scalar.ComplexF64, do_conj: bool) [2]scalar.ComplexF64 {
    const re_sign: @Vector(4, f64) = if (do_conj) @splat(1) else .{ 1, -1, 1, -1 };
    const im_sign: @Vector(4, f64) = if (do_conj) .{ 1, -1, 1, -1 } else @splat(1);
    const a0_real: [*]const f64 = @ptrCast(a0_col);
    const a1_real: [*]const f64 = @ptrCast(a1_col);
    const x_real: [*]const f64 = @ptrCast(x);
    const real_n = 2 * m;
    var re_acc00: @Vector(4, f64) = @splat(0);
    var re_acc01: @Vector(4, f64) = @splat(0);
    var im_acc00: @Vector(4, f64) = @splat(0);
    var im_acc01: @Vector(4, f64) = @splat(0);
    var re_acc10: @Vector(4, f64) = @splat(0);
    var re_acc11: @Vector(4, f64) = @splat(0);
    var im_acc10: @Vector(4, f64) = @splat(0);
    var im_acc11: @Vector(4, f64) = @splat(0);
    var i: usize = 0;
    while (i + 8 <= real_n) : (i += 8) {
        c64Dot2AccumulateVec(a0_real, a1_real, x_real, i, re_sign, im_sign, &re_acc00, &im_acc00, &re_acc10, &im_acc10);
        c64Dot2AccumulateVec(a0_real, a1_real, x_real, i + 4, re_sign, im_sign, &re_acc01, &im_acc01, &re_acc11, &im_acc11);
    }
    var re_acc0 = re_acc00 + re_acc01;
    var im_acc0 = im_acc00 + im_acc01;
    var re_acc1 = re_acc10 + re_acc11;
    var im_acc1 = im_acc10 + im_acc11;
    while (i + 4 <= real_n) : (i += 4) {
        c64Dot2AccumulateVec(a0_real, a1_real, x_real, i, re_sign, im_sign, &re_acc0, &im_acc0, &re_acc1, &im_acc1);
    }
    var re_sum0: f64 = @reduce(.Add, re_acc0);
    var im_sum0: f64 = @reduce(.Add, im_acc0);
    var re_sum1: f64 = @reduce(.Add, re_acc1);
    var im_sum1: f64 = @reduce(.Add, im_acc1);
    while (i < real_n) : (i += 2) {
        const ar0 = a0_real[i];
        const ai0 = a0_real[i + 1];
        const ar1 = a1_real[i];
        const ai1 = a1_real[i + 1];
        const xr = x_real[i];
        const xi = x_real[i + 1];
        if (do_conj) {
            re_sum0 = @mulAdd(f64, ai0, xi, @mulAdd(f64, ar0, xr, re_sum0));
            im_sum0 = @mulAdd(f64, -ai0, xr, @mulAdd(f64, ar0, xi, im_sum0));
            re_sum1 = @mulAdd(f64, ai1, xi, @mulAdd(f64, ar1, xr, re_sum1));
            im_sum1 = @mulAdd(f64, -ai1, xr, @mulAdd(f64, ar1, xi, im_sum1));
        } else {
            re_sum0 = @mulAdd(f64, -ai0, xi, @mulAdd(f64, ar0, xr, re_sum0));
            im_sum0 = @mulAdd(f64, ai0, xr, @mulAdd(f64, ar0, xi, im_sum0));
            re_sum1 = @mulAdd(f64, -ai1, xi, @mulAdd(f64, ar1, xr, re_sum1));
            im_sum1 = @mulAdd(f64, ai1, xr, @mulAdd(f64, ar1, xi, im_sum1));
        }
    }
    return .{
        .{ .re = re_sum0, .im = im_sum0 },
        .{ .re = re_sum1, .im = im_sum1 },
    };
}

fn c64Dot4Unit(m: usize, a0_col: [*]const scalar.ComplexF64, a1_col: [*]const scalar.ComplexF64, a2_col: [*]const scalar.ComplexF64, a3_col: [*]const scalar.ComplexF64, x: [*]const scalar.ComplexF64, do_conj: bool) [4]scalar.ComplexF64 {
    const re_sign: @Vector(4, f64) = if (do_conj) @splat(1) else .{ 1, -1, 1, -1 };
    const im_sign: @Vector(4, f64) = if (do_conj) .{ 1, -1, 1, -1 } else @splat(1);
    const a0_real: [*]const f64 = @ptrCast(a0_col);
    const a1_real: [*]const f64 = @ptrCast(a1_col);
    const a2_real: [*]const f64 = @ptrCast(a2_col);
    const a3_real: [*]const f64 = @ptrCast(a3_col);
    const x_real: [*]const f64 = @ptrCast(x);
    const real_n = 2 * m;
    var re_acc0: @Vector(4, f64) = @splat(0);
    var im_acc0: @Vector(4, f64) = @splat(0);
    var re_acc1: @Vector(4, f64) = @splat(0);
    var im_acc1: @Vector(4, f64) = @splat(0);
    var re_acc2: @Vector(4, f64) = @splat(0);
    var im_acc2: @Vector(4, f64) = @splat(0);
    var re_acc3: @Vector(4, f64) = @splat(0);
    var im_acc3: @Vector(4, f64) = @splat(0);
    var i: usize = 0;
    while (i + 4 <= real_n) : (i += 4) {
        c64Dot4AccumulateVec(a0_real, a1_real, a2_real, a3_real, x_real, i, re_sign, im_sign, &re_acc0, &im_acc0, &re_acc1, &im_acc1, &re_acc2, &im_acc2, &re_acc3, &im_acc3);
    }
    var re_sum0: f64 = @reduce(.Add, re_acc0);
    var im_sum0: f64 = @reduce(.Add, im_acc0);
    var re_sum1: f64 = @reduce(.Add, re_acc1);
    var im_sum1: f64 = @reduce(.Add, im_acc1);
    var re_sum2: f64 = @reduce(.Add, re_acc2);
    var im_sum2: f64 = @reduce(.Add, im_acc2);
    var re_sum3: f64 = @reduce(.Add, re_acc3);
    var im_sum3: f64 = @reduce(.Add, im_acc3);
    while (i < real_n) : (i += 2) {
        const ar0 = a0_real[i];
        const ai0 = a0_real[i + 1];
        const ar1 = a1_real[i];
        const ai1 = a1_real[i + 1];
        const ar2 = a2_real[i];
        const ai2 = a2_real[i + 1];
        const ar3 = a3_real[i];
        const ai3 = a3_real[i + 1];
        const xr = x_real[i];
        const xi = x_real[i + 1];
        if (do_conj) {
            re_sum0 = @mulAdd(f64, ai0, xi, @mulAdd(f64, ar0, xr, re_sum0));
            im_sum0 = @mulAdd(f64, -ai0, xr, @mulAdd(f64, ar0, xi, im_sum0));
            re_sum1 = @mulAdd(f64, ai1, xi, @mulAdd(f64, ar1, xr, re_sum1));
            im_sum1 = @mulAdd(f64, -ai1, xr, @mulAdd(f64, ar1, xi, im_sum1));
            re_sum2 = @mulAdd(f64, ai2, xi, @mulAdd(f64, ar2, xr, re_sum2));
            im_sum2 = @mulAdd(f64, -ai2, xr, @mulAdd(f64, ar2, xi, im_sum2));
            re_sum3 = @mulAdd(f64, ai3, xi, @mulAdd(f64, ar3, xr, re_sum3));
            im_sum3 = @mulAdd(f64, -ai3, xr, @mulAdd(f64, ar3, xi, im_sum3));
        } else {
            re_sum0 = @mulAdd(f64, -ai0, xi, @mulAdd(f64, ar0, xr, re_sum0));
            im_sum0 = @mulAdd(f64, ai0, xr, @mulAdd(f64, ar0, xi, im_sum0));
            re_sum1 = @mulAdd(f64, -ai1, xi, @mulAdd(f64, ar1, xr, re_sum1));
            im_sum1 = @mulAdd(f64, ai1, xr, @mulAdd(f64, ar1, xi, im_sum1));
            re_sum2 = @mulAdd(f64, -ai2, xi, @mulAdd(f64, ar2, xr, re_sum2));
            im_sum2 = @mulAdd(f64, ai2, xr, @mulAdd(f64, ar2, xi, im_sum2));
            re_sum3 = @mulAdd(f64, -ai3, xi, @mulAdd(f64, ar3, xr, re_sum3));
            im_sum3 = @mulAdd(f64, ai3, xr, @mulAdd(f64, ar3, xi, im_sum3));
        }
    }
    return .{
        .{ .re = re_sum0, .im = im_sum0 },
        .{ .re = re_sum1, .im = im_sum1 },
        .{ .re = re_sum2, .im = im_sum2 },
        .{ .re = re_sum3, .im = im_sum3 },
    };
}

fn gemvTransUnitComplexC64(m: usize, n: usize, alpha: scalar.ComplexF64, a: [*]const scalar.ComplexF64, lda: BlasInt, x: [*]const scalar.ComplexF64, y: [*]scalar.ComplexF64, do_conj: bool) void {
    var j: usize = 0;
    if (m == 128) {
        while (j + 4 <= n) : (j += 4) {
            const sums = c64Dot4Unit(m, a + matIndex(lda, 0, j), a + matIndex(lda, 0, j + 1), a + matIndex(lda, 0, j + 2), a + matIndex(lda, 0, j + 3), x, do_conj);
            y[j] = add(scalar.ComplexF64, y[j], mul(scalar.ComplexF64, alpha, sums[0]));
            y[j + 1] = add(scalar.ComplexF64, y[j + 1], mul(scalar.ComplexF64, alpha, sums[1]));
            y[j + 2] = add(scalar.ComplexF64, y[j + 2], mul(scalar.ComplexF64, alpha, sums[2]));
            y[j + 3] = add(scalar.ComplexF64, y[j + 3], mul(scalar.ComplexF64, alpha, sums[3]));
        }
    }
    while (j + 2 <= n) : (j += 2) {
        const sums = c64Dot2Unit(m, a + matIndex(lda, 0, j), a + matIndex(lda, 0, j + 1), x, do_conj);
        y[j] = add(scalar.ComplexF64, y[j], mul(scalar.ComplexF64, alpha, sums[0]));
        y[j + 1] = add(scalar.ComplexF64, y[j + 1], mul(scalar.ComplexF64, alpha, sums[1]));
    }
    while (j < n) : (j += 1) {
        const sum = c64DotUnit(m, a + matIndex(lda, 0, j), x, do_conj);
        y[j] = add(scalar.ComplexF64, y[j], mul(scalar.ComplexF64, alpha, sum));
    }
}

fn GemvTransComplexTask(comptime T: type) type {
    return struct {
        m: BlasInt,
        n0: usize,
        n1: usize,
        alpha: T,
        a: [*]const T,
        lda: BlasInt,
        x: [*]const T,
        y: [*]T,
        do_conj: bool,
    };
}

fn runGemvTransComplexTask(comptime T: type, raw_tasks: *const anyopaque, index: usize) void {
    const tasks: [*]const GemvTransComplexTask(T) = @ptrCast(@alignCast(raw_tasks));
    const task = tasks[index];
    gemvTransUnitComplex(T, task.m, task.n1 - task.n0, task.alpha, task.a + matIndex(task.lda, 0, task.n0), task.lda, task.x, task.y + task.n0, task.do_conj);
}

fn runGemvTransComplexTaskC32(raw_tasks: *const anyopaque, index: usize) void {
    runGemvTransComplexTask(scalar.ComplexF32, raw_tasks, index);
}

fn runGemvTransComplexTaskC64(raw_tasks: *const anyopaque, index: usize) void {
    runGemvTransComplexTask(scalar.ComplexF64, raw_tasks, index);
}

fn parallelGemvTransUnitComplex(comptime T: type, m: usize, n: usize, m_: BlasInt, alpha: T, a: [*]const T, lda: BlasInt, x: [*]const T, y: [*]T, do_conj: bool) bool {
    if (T == scalar.ComplexF32 and m == 128 and n == 128) return false;
    if (m *| n < 128 * 128) return false;
    const min_cols_per_task: usize = if (T == scalar.ComplexF32) 64 else 48;
    var task_count = core_pool.taskCount(n, min_cols_per_task);
    const max_task_count: usize = if (T == scalar.ComplexF64 and n >= 256 and n < 512) 10 else if (n < 512) 4 else if (T == scalar.ComplexF32) 10 else 8;
    task_count = @min(task_count, max_task_count);
    if (task_count <= 1) return false;

    var tasks: [core_pool.max_tasks]GemvTransComplexTask(T) = undefined;
    for (0..task_count) |task_index| {
        const n0 = task_index * n / task_count;
        const n1 = (task_index + 1) * n / task_count;
        tasks[task_index] = .{
            .m = m_,
            .n0 = n0,
            .n1 = n1,
            .alpha = alpha,
            .a = a,
            .lda = lda,
            .x = x,
            .y = y,
            .do_conj = do_conj,
        };
    }

    const runner = if (T == scalar.ComplexF32) runGemvTransComplexTaskC32 else runGemvTransComplexTaskC64;
    return core_pool.runLowLatency(runner, @ptrCast(&tasks), task_count);
}

fn gemvUnitComplex(comptime T: type, trans_: Order, m_: BlasInt, n_: BlasInt, alpha: T, a: [*]const T, lda: BlasInt, x: [*]const T, beta: T, y: [*]T) void {
    const m = toUsize(m_);
    const n = toUsize(n_);
    const leny = if (trans_ == .no_trans) m_ else n_;
    level1.scal(T, leny, beta, y, 1);
    if (isZero(T, alpha)) return;

    if (trans_ == .no_trans) {
        if (parallelGemvNoTransUnitComplex(T, m, n, m_, alpha, a, lda, x, y)) return;
        gemvNoTransUnitComplex(T, m_, n, alpha, a, lda, x, y);
    } else {
        const do_conj = trans_ == .conj_trans;
        if (parallelGemvTransUnitComplex(T, m, n, m_, alpha, a, lda, x, y, do_conj)) return;
        gemvTransUnitComplex(T, m_, n, alpha, a, lda, x, y, do_conj);
    }
}

pub fn gemv(comptime T: type, trans_: Order, m_: BlasInt, n_: BlasInt, alpha: T, a: [*]const T, lda: BlasInt, x: [*]const T, incx_: BlasInt, beta: T, y: [*]T, incy_: BlasInt) void {
    if (m_ <= 0 or n_ <= 0 or incx_ == 0 or incy_ == 0) return;
    const m = toUsize(m_);
    const n = toUsize(n_);
    if (comptime isReal(T)) {
        if (incx_ == 1 and incy_ == 1) return gemvUnitReal(T, trans_, m, n, alpha, a, lda, x, beta, y);
    } else if (comptime isComplex(T)) {
        if (incx_ == 1 and incy_ == 1) return gemvUnitComplex(T, trans_, m_, n_, alpha, a, lda, x, beta, y);
    }
    const lenx: BlasInt = if (trans_ == .no_trans) n_ else m_;
    const leny: BlasInt = if (trans_ == .no_trans) m_ else n_;
    const sx = startIndex(lenx, incx_);
    const sy = startIndex(leny, incy_);
    for (0..toUsize(leny)) |i| {
        const py = ix(sy, i, incy_);
        y[py] = if (isZero(T, beta)) zero(T) else mul(T, beta, y[py]);
    }
    if (isZero(T, alpha)) return;
    if (trans_ == .no_trans) {
        for (0..n) |j| {
            const xj = mul(T, alpha, vectorGet(T, x, sx, j, incx_));
            if (isZero(T, xj)) continue;
            for (0..m) |i| {
                const py = ix(sy, i, incy_);
                y[py] = add(T, y[py], mul(T, a[matIndex(lda, i, j)], xj));
            }
        }
    } else {
        for (0..n) |j| {
            var sum = zero(T);
            for (0..m) |i| {
                var av = a[matIndex(lda, i, j)];
                if (trans_ == .conj_trans) av = conj(T, av);
                sum = add(T, sum, mul(T, av, vectorGet(T, x, sx, i, incx_)));
            }
            const py = ix(sy, j, incy_);
            y[py] = add(T, y[py], mul(T, alpha, sum));
        }
    }
}

pub fn gbmv(comptime T: type, trans_: Order, m_: BlasInt, n_: BlasInt, kl_: BlasInt, ku_: BlasInt, alpha: T, a: [*]const T, lda: BlasInt, x: [*]const T, incx_: BlasInt, beta: T, y: [*]T, incy_: BlasInt) void {
    if (m_ <= 0 or n_ <= 0 or incx_ == 0 or incy_ == 0) return;
    const m = toUsize(m_);
    const n = toUsize(n_);
    const kl = toUsize(kl_);
    const ku = toUsize(ku_);
    const lenx: BlasInt = if (trans_ == .no_trans) n_ else m_;
    const leny: BlasInt = if (trans_ == .no_trans) m_ else n_;
    const sx = startIndex(lenx, incx_);
    const sy = startIndex(leny, incy_);
    for (0..toUsize(leny)) |i| {
        const py = ix(sy, i, incy_);
        y[py] = if (isZero(T, beta)) zero(T) else mul(T, beta, y[py]);
    }
    if (isZero(T, alpha)) return;
    if (trans_ == .no_trans) {
        for (0..n) |j| {
            const xj = mul(T, alpha, vectorGet(T, x, sx, j, incx_));
            const row0 = if (j > ku) j - ku else 0;
            const row1 = @min(m, j + kl + 1);
            for (row0..row1) |i| {
                const idxa = bandGeneralIndex(m, n, kl, ku, lda, i, j).?;
                const py = ix(sy, i, incy_);
                y[py] = add(T, y[py], mul(T, a[idxa], xj));
            }
        }
    } else {
        for (0..n) |j| {
            var sum = zero(T);
            const row0 = if (j > ku) j - ku else 0;
            const row1 = @min(m, j + kl + 1);
            for (row0..row1) |i| {
                var av = a[bandGeneralIndex(m, n, kl, ku, lda, i, j).?];
                if (trans_ == .conj_trans) av = conj(T, av);
                sum = add(T, sum, mul(T, av, vectorGet(T, x, sx, i, incx_)));
            }
            const py = ix(sy, j, incy_);
            y[py] = add(T, y[py], mul(T, alpha, sum));
        }
    }
}
