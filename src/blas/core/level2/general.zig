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
const add = scalar.add;
const mul = scalar.mul;
const conj = scalar.conj;
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
    for (0..n) |j| {
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
    if (m *| n < 768 * 768) return false;
    var task_count = core_pool.taskCount(n, 64);
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

pub fn gemv(comptime T: type, trans_: Order, m_: BlasInt, n_: BlasInt, alpha: T, a: [*]const T, lda: BlasInt, x: [*]const T, incx_: BlasInt, beta: T, y: [*]T, incy_: BlasInt) void {
    if (m_ <= 0 or n_ <= 0 or incx_ == 0 or incy_ == 0) return;
    const m = toUsize(m_);
    const n = toUsize(n_);
    if (comptime isReal(T)) {
        if (incx_ == 1 and incy_ == 1) return gemvUnitReal(T, trans_, m, n, alpha, a, lda, x, beta, y);
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
