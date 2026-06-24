// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later

const std = @import("std");

const scalar = @import("scalar.zig");
const indexing = @import("indexing.zig");
const core_pool = @import("pool.zig");
const vector_binary_kernels = @import("../kernels/vector_binary.zig");
const vector_unary_kernels = @import("../kernels/vector_unary.zig");

pub const BlasInt = scalar.BlasInt;
pub const ComplexF32 = scalar.ComplexF32;
pub const ComplexF64 = scalar.ComplexF64;

const Real = scalar.Real;
const zero = scalar.zero;
const one = scalar.one;
const realScalar = scalar.realScalar;
const add = scalar.add;
const sub = scalar.sub;
const mul = scalar.mul;
const divv = scalar.divv;
const conj = scalar.conj;
const maybeConj = scalar.maybeConj;
const realPart = scalar.realPart;
const imagPart = scalar.imagPart;
const isComplex = scalar.isComplex;
const isZero = scalar.isZero;
const abs1 = scalar.abs1;
const abs2 = scalar.abs2;

const toUsize = indexing.toUsize;
const startIndex = indexing.startIndex;
const ix = indexing.ix;
const vectorGet = indexing.vectorGet;
const vectorSet = indexing.vectorSet;

extern fn memcpy(noalias dest: [*]u8, noalias src: [*]const u8, n: usize) callconv(.c) [*]u8;

fn isReal(comptime T: type) bool {
    return T == f32 or T == f64;
}

fn lanes(comptime T: type) comptime_int {
    if (T == f32) return 8;
    if (T == f64) return 4;
    @compileError("real Level 1 vector lanes support f32 and f64");
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

pub fn scalUnitReal(comptime T: type, n: usize, alpha: T, x: [*]T) void {
    if (vector_unary_kernels.scalUnitReal(T, n, alpha, x)) return;
    const lane_count = lanes(T);
    const unroll_count = unroll(T);
    const V = @Vector(lane_count, T);
    const alpha_v: V = @splat(alpha);
    var i: usize = 0;
    while (i + unroll_count <= n) : (i += unroll_count) {
        inline for (0..4) |k| {
            const offset = i + k * lane_count;
            storeVec(T, lane_count, x, offset, loadVec(T, lane_count, x, offset) * alpha_v);
        }
    }
    while (i + lane_count <= n) : (i += lane_count) {
        storeVec(T, lane_count, x, i, loadVec(T, lane_count, x, i) * alpha_v);
    }
    while (i < n) : (i += 1) x[i] *= alpha;
}

fn copyUnit(comptime T: type, n: usize, x: [*]const T, y: [*]T) void {
    if (n == 0) return;
    _ = memcpy(@ptrCast(y), @ptrCast(x), n * @sizeOf(T));
}

pub fn copyUnitReal(comptime T: type, n: usize, x: [*]const T, y: [*]T) void {
    if (vector_binary_kernels.copyUnitReal(T, n, x, y)) return;
    copyUnit(T, n, x, y);
}

fn swapUnit(comptime T: type, n: usize, x: [*]T, y: [*]T) void {
    if (comptime isReal(T)) {
        const lane_count = lanes(T);
        var i: usize = 0;
        while (i + lane_count <= n) : (i += lane_count) {
            const xv = loadVec(T, lane_count, x, i);
            const yv = loadVec(T, lane_count, y, i);
            storeVec(T, lane_count, x, i, yv);
            storeVec(T, lane_count, y, i, xv);
        }
        while (i < n) : (i += 1) {
            const t = x[i];
            x[i] = y[i];
            y[i] = t;
        }
    } else {
        for (0..n) |i| {
            const t = x[i];
            x[i] = y[i];
            y[i] = t;
        }
    }
}

pub fn axpyUnitReal(comptime T: type, n: usize, alpha: T, x: [*]const T, y: [*]T) void {
    if (vector_binary_kernels.axpyUnitReal(T, n, alpha, x, y)) return;
    const lane_count = lanes(T);
    const unroll_count = unroll(T);
    const V = @Vector(lane_count, T);
    const alpha_v: V = @splat(alpha);
    var i: usize = 0;
    while (i + unroll_count <= n) : (i += unroll_count) {
        inline for (0..4) |k| {
            const offset = i + k * lane_count;
            const xv = loadVec(T, lane_count, x, offset);
            const yv = loadVec(T, lane_count, y, offset);
            storeVec(T, lane_count, y, offset, @mulAdd(V, xv, alpha_v, yv));
        }
    }
    while (i + lane_count <= n) : (i += lane_count) {
        const xv = loadVec(T, lane_count, x, i);
        const yv = loadVec(T, lane_count, y, i);
        storeVec(T, lane_count, y, i, @mulAdd(V, xv, alpha_v, yv));
    }
    while (i < n) : (i += 1) y[i] = @mulAdd(T, alpha, x[i], y[i]);
}

fn axpbyUnitReal(comptime T: type, n: usize, alpha: T, x: [*]const T, beta: T, y: [*]T) void {
    const lane_count = lanes(T);
    const unroll_count = unroll(T);
    const V = @Vector(lane_count, T);
    const alpha_v: V = @splat(alpha);
    const beta_v: V = @splat(beta);
    var i: usize = 0;
    while (i + unroll_count <= n) : (i += unroll_count) {
        inline for (0..4) |k| {
            const offset = i + k * lane_count;
            const xv = loadVec(T, lane_count, x, offset);
            const yv = loadVec(T, lane_count, y, offset);
            storeVec(T, lane_count, y, offset, @mulAdd(V, xv, alpha_v, yv * beta_v));
        }
    }
    while (i + lane_count <= n) : (i += lane_count) {
        const xv = loadVec(T, lane_count, x, i);
        const yv = loadVec(T, lane_count, y, i);
        storeVec(T, lane_count, y, i, @mulAdd(V, xv, alpha_v, yv * beta_v));
    }
    while (i < n) : (i += 1) y[i] = @mulAdd(T, alpha, x[i], beta * y[i]);
}

pub fn dotUnitReal(comptime T: type, n: usize, x: [*]const T, y: [*]const T) T {
    if (vector_binary_kernels.dotUnitReal(T, n, x, y)) |result| return result;
    const lane_count = lanes(T);
    const unroll_count = unroll(T);
    const V = @Vector(lane_count, T);
    var acc0: V = @splat(0);
    var acc1: V = @splat(0);
    var acc2: V = @splat(0);
    var acc3: V = @splat(0);
    var i: usize = 0;
    while (i + unroll_count <= n) : (i += unroll_count) {
        acc0 = @mulAdd(V, loadVec(T, lane_count, x, i), loadVec(T, lane_count, y, i), acc0);
        acc1 = @mulAdd(V, loadVec(T, lane_count, x, i + lane_count), loadVec(T, lane_count, y, i + lane_count), acc1);
        acc2 = @mulAdd(V, loadVec(T, lane_count, x, i + 2 * lane_count), loadVec(T, lane_count, y, i + 2 * lane_count), acc2);
        acc3 = @mulAdd(V, loadVec(T, lane_count, x, i + 3 * lane_count), loadVec(T, lane_count, y, i + 3 * lane_count), acc3);
    }
    var acc = acc0 + acc1 + acc2 + acc3;
    while (i + lane_count <= n) : (i += lane_count) {
        acc = @mulAdd(V, loadVec(T, lane_count, x, i), loadVec(T, lane_count, y, i), acc);
    }
    var sum: T = @reduce(.Add, acc);
    while (i < n) : (i += 1) sum = @mulAdd(T, x[i], y[i], sum);
    return sum;
}

fn asumUnitReal(comptime T: type, n: usize, x: [*]const T) T {
    if (vector_unary_kernels.asumUnitReal(T, n, x)) |result| return result;
    const lane_count = lanes(T);
    const unroll_count = unroll(T);
    const V = @Vector(lane_count, T);
    var acc0: V = @splat(0);
    var acc1: V = @splat(0);
    var acc2: V = @splat(0);
    var acc3: V = @splat(0);
    var i: usize = 0;
    while (i + unroll_count <= n) : (i += unroll_count) {
        acc0 += @abs(loadVec(T, lane_count, x, i));
        acc1 += @abs(loadVec(T, lane_count, x, i + lane_count));
        acc2 += @abs(loadVec(T, lane_count, x, i + 2 * lane_count));
        acc3 += @abs(loadVec(T, lane_count, x, i + 3 * lane_count));
    }
    var acc = acc0 + acc1 + acc2 + acc3;
    while (i + lane_count <= n) : (i += lane_count) {
        acc += @abs(loadVec(T, lane_count, x, i));
    }
    var sum: T = @reduce(.Add, acc);
    while (i < n) : (i += 1) sum += @abs(x[i]);
    return sum;
}

fn nrm2UnitReal(comptime T: type, n: usize, x: [*]const T) ?T {
    const lane_count = lanes(T);
    const unroll_count = unroll(T);
    const V = @Vector(lane_count, T);
    var max_v: V = @splat(0);
    var i: usize = 0;
    while (i + unroll_count <= n) : (i += unroll_count) {
        inline for (0..4) |k| {
            max_v = @max(max_v, @abs(loadVec(T, lane_count, x, i + k * lane_count)));
        }
    }
    while (i + lane_count <= n) : (i += lane_count) {
        max_v = @max(max_v, @abs(loadVec(T, lane_count, x, i)));
    }
    var scale: T = @reduce(.Max, max_v);
    while (i < n) : (i += 1) scale = @max(scale, @abs(x[i]));
    if (scale == 0) return 0;
    if (!std.math.isFinite(scale)) return null;

    const inv_scale_v: V = @splat(1 / scale);
    var acc0: V = @splat(0);
    var acc1: V = @splat(0);
    var acc2: V = @splat(0);
    var acc3: V = @splat(0);
    i = 0;
    while (i + unroll_count <= n) : (i += unroll_count) {
        const v0 = loadVec(T, lane_count, x, i) * inv_scale_v;
        const v1 = loadVec(T, lane_count, x, i + lane_count) * inv_scale_v;
        const v2 = loadVec(T, lane_count, x, i + 2 * lane_count) * inv_scale_v;
        const v3 = loadVec(T, lane_count, x, i + 3 * lane_count) * inv_scale_v;
        acc0 = @mulAdd(V, v0, v0, acc0);
        acc1 = @mulAdd(V, v1, v1, acc1);
        acc2 = @mulAdd(V, v2, v2, acc2);
        acc3 = @mulAdd(V, v3, v3, acc3);
    }
    var acc = acc0 + acc1 + acc2 + acc3;
    while (i + lane_count <= n) : (i += lane_count) {
        const v = loadVec(T, lane_count, x, i) * inv_scale_v;
        acc = @mulAdd(V, v, v, acc);
    }
    var ssq: T = @reduce(.Add, acc);
    while (i < n) : (i += 1) {
        const v = x[i] / scale;
        ssq = @mulAdd(T, v, v, ssq);
    }
    return scale * @sqrt(ssq);
}

fn parallelTaskCount(n: usize, min_items_per_task: usize, max_task_count: usize) usize {
    if (n < 512 * 1024) return 1;
    return @min(core_pool.taskCount(n, min_items_per_task), max_task_count);
}

fn parallelScalTaskCount(n: usize) usize {
    if (n < 2 * 1024 * 1024) return parallelTaskCount(n, 128 * 1024, 8);
    return parallelTaskCount(n, 170 * 1024, 6);
}

fn RangeTask(comptime T: type) type {
    return struct {
        n0: usize,
        n1: usize,
        alpha: T,
        beta: T,
        x: [*]const T,
        y: [*]T,
        out: *T,
    };
}

fn runScalTask(comptime T: type, raw_tasks: *const anyopaque, index: usize) void {
    const tasks: [*]const RangeTask(T) = @ptrCast(@alignCast(raw_tasks));
    const task = tasks[index];
    scalUnitReal(T, task.n1 - task.n0, task.alpha, task.y + task.n0);
}

fn runScalTaskF32(raw_tasks: *const anyopaque, index: usize) void {
    runScalTask(f32, raw_tasks, index);
}

fn runScalTaskF64(raw_tasks: *const anyopaque, index: usize) void {
    runScalTask(f64, raw_tasks, index);
}

fn parallelScalUnitReal(comptime T: type, n: usize, alpha: T, x: [*]T) bool {
    const task_count = parallelScalTaskCount(n);
    if (task_count <= 1) return false;

    var tasks: [core_pool.max_tasks]RangeTask(T) = undefined;
    for (0..task_count) |task_index| {
        tasks[task_index] = .{
            .n0 = task_index * n / task_count,
            .n1 = (task_index + 1) * n / task_count,
            .alpha = alpha,
            .beta = 0,
            .x = x,
            .y = x,
            .out = undefined,
        };
    }
    const runner = if (T == f32) runScalTaskF32 else runScalTaskF64;
    return core_pool.run(runner, @ptrCast(&tasks), task_count);
}

fn runCopyTask(comptime T: type, raw_tasks: *const anyopaque, index: usize) void {
    const tasks: [*]const RangeTask(T) = @ptrCast(@alignCast(raw_tasks));
    const task = tasks[index];
    copyUnitReal(T, task.n1 - task.n0, task.x + task.n0, task.y + task.n0);
}

fn runCopyTaskF32(raw_tasks: *const anyopaque, index: usize) void {
    runCopyTask(f32, raw_tasks, index);
}

fn runCopyTaskF64(raw_tasks: *const anyopaque, index: usize) void {
    runCopyTask(f64, raw_tasks, index);
}

fn parallelCopyUnitReal(comptime T: type, n: usize, x: [*]const T, y: [*]T) bool {
    const task_count = parallelTaskCount(n, 128 * 1024, 10);
    if (task_count <= 1) return false;

    var tasks: [core_pool.max_tasks]RangeTask(T) = undefined;
    const block_items: usize = if (T == f64) 32 else 64;
    const block_count = n / block_items;
    for (0..task_count) |task_index| {
        const n0, const n1 = if (block_count > 0) .{
            (task_index * block_count / task_count) * block_items,
            if (task_index + 1 == task_count) n else ((task_index + 1) * block_count / task_count) * block_items,
        } else .{
            task_index * n / task_count,
            (task_index + 1) * n / task_count,
        };
        tasks[task_index] = .{
            .n0 = n0,
            .n1 = n1,
            .alpha = 0,
            .beta = 0,
            .x = x,
            .y = y,
            .out = undefined,
        };
    }
    const runner = if (T == f32) runCopyTaskF32 else runCopyTaskF64;
    return core_pool.run(runner, @ptrCast(&tasks), task_count);
}

fn runAxpyTask(comptime T: type, raw_tasks: *const anyopaque, index: usize) void {
    const tasks: [*]const RangeTask(T) = @ptrCast(@alignCast(raw_tasks));
    const task = tasks[index];
    axpyUnitReal(T, task.n1 - task.n0, task.alpha, task.x + task.n0, task.y + task.n0);
}

fn runAxpyTaskF32(raw_tasks: *const anyopaque, index: usize) void {
    runAxpyTask(f32, raw_tasks, index);
}

fn runAxpyTaskF64(raw_tasks: *const anyopaque, index: usize) void {
    runAxpyTask(f64, raw_tasks, index);
}

fn parallelAxpyUnitReal(comptime T: type, n: usize, alpha: T, x: [*]const T, y: [*]T) bool {
    const task_count = parallelTaskCount(n, 170 * 1024, 6);
    if (task_count <= 1) return false;

    var tasks: [core_pool.max_tasks]RangeTask(T) = undefined;
    for (0..task_count) |task_index| {
        tasks[task_index] = .{
            .n0 = task_index * n / task_count,
            .n1 = (task_index + 1) * n / task_count,
            .alpha = alpha,
            .beta = 0,
            .x = x,
            .y = y,
            .out = undefined,
        };
    }
    const runner = if (T == f32) runAxpyTaskF32 else runAxpyTaskF64;
    return core_pool.run(runner, @ptrCast(&tasks), task_count);
}

fn runAxpbyTask(comptime T: type, raw_tasks: *const anyopaque, index: usize) void {
    const tasks: [*]const RangeTask(T) = @ptrCast(@alignCast(raw_tasks));
    const task = tasks[index];
    axpbyUnitReal(T, task.n1 - task.n0, task.alpha, task.x + task.n0, task.beta, task.y + task.n0);
}

fn runAxpbyTaskF32(raw_tasks: *const anyopaque, index: usize) void {
    runAxpbyTask(f32, raw_tasks, index);
}

fn runAxpbyTaskF64(raw_tasks: *const anyopaque, index: usize) void {
    runAxpbyTask(f64, raw_tasks, index);
}

fn parallelAxpbyUnitReal(comptime T: type, n: usize, alpha: T, x: [*]const T, beta: T, y: [*]T) bool {
    const task_count = parallelTaskCount(n, 170 * 1024, 6);
    if (task_count <= 1) return false;

    var tasks: [core_pool.max_tasks]RangeTask(T) = undefined;
    for (0..task_count) |task_index| {
        tasks[task_index] = .{
            .n0 = task_index * n / task_count,
            .n1 = (task_index + 1) * n / task_count,
            .alpha = alpha,
            .beta = beta,
            .x = x,
            .y = y,
            .out = undefined,
        };
    }
    const runner = if (T == f32) runAxpbyTaskF32 else runAxpbyTaskF64;
    return core_pool.run(runner, @ptrCast(&tasks), task_count);
}

fn runDotTask(comptime T: type, raw_tasks: *const anyopaque, index: usize) void {
    const tasks: [*]const RangeTask(T) = @ptrCast(@alignCast(raw_tasks));
    const task = tasks[index];
    task.out.* = dotUnitReal(T, task.n1 - task.n0, task.x + task.n0, task.y + task.n0);
}

fn runDotTaskF32(raw_tasks: *const anyopaque, index: usize) void {
    runDotTask(f32, raw_tasks, index);
}

fn runDotTaskF64(raw_tasks: *const anyopaque, index: usize) void {
    runDotTask(f64, raw_tasks, index);
}

fn parallelDotUnitReal(comptime T: type, n: usize, x: [*]const T, y: [*]const T) ?T {
    const task_count = parallelTaskCount(n, 128 * 1024, 10);
    if (task_count <= 1) return null;

    var partial: [core_pool.max_tasks]T = undefined;
    var tasks: [core_pool.max_tasks]RangeTask(T) = undefined;
    for (0..task_count) |task_index| {
        partial[task_index] = 0;
        tasks[task_index] = .{
            .n0 = task_index * n / task_count,
            .n1 = (task_index + 1) * n / task_count,
            .alpha = 0,
            .beta = 0,
            .x = x,
            .y = @constCast(y),
            .out = &partial[task_index],
        };
    }
    const runner = if (T == f32) runDotTaskF32 else runDotTaskF64;
    if (!core_pool.run(runner, @ptrCast(&tasks), task_count)) return null;

    var result: T = 0;
    for (partial[0..task_count]) |v| result += v;
    return result;
}

fn runAsumTask(comptime T: type, raw_tasks: *const anyopaque, index: usize) void {
    const tasks: [*]const RangeTask(T) = @ptrCast(@alignCast(raw_tasks));
    const task = tasks[index];
    task.out.* = asumUnitReal(T, task.n1 - task.n0, task.x + task.n0);
}

fn runAsumTaskF32(raw_tasks: *const anyopaque, index: usize) void {
    runAsumTask(f32, raw_tasks, index);
}

fn runAsumTaskF64(raw_tasks: *const anyopaque, index: usize) void {
    runAsumTask(f64, raw_tasks, index);
}

fn parallelAsumUnitReal(comptime T: type, n: usize, x: [*]const T) ?T {
    const task_count = parallelTaskCount(n, 96 * 1024, 10);
    if (task_count <= 1) return null;

    var partial: [core_pool.max_tasks]T = undefined;
    var tasks: [core_pool.max_tasks]RangeTask(T) = undefined;
    for (0..task_count) |task_index| {
        partial[task_index] = 0;
        tasks[task_index] = .{
            .n0 = task_index * n / task_count,
            .n1 = (task_index + 1) * n / task_count,
            .alpha = 0,
            .beta = 0,
            .x = x,
            .y = undefined,
            .out = &partial[task_index],
        };
    }
    const runner = if (T == f32) runAsumTaskF32 else runAsumTaskF64;
    if (!core_pool.run(runner, @ptrCast(&tasks), task_count)) return null;

    var result: T = 0;
    for (partial[0..task_count]) |v| result += v;
    return result;
}

pub fn scal(comptime T: type, n_: BlasInt, alpha: T, x: [*]T, incx_: BlasInt) void {
    if (n_ <= 0 or incx_ == 0) return;
    const n = toUsize(n_);
    if (comptime isReal(T)) {
        if (incx_ == 1) {
            if (parallelScalUnitReal(T, n, alpha, x)) return;
            return scalUnitReal(T, n, alpha, x);
        }
    }
    const sx = startIndex(n_, incx_);
    for (0..n) |i| {
        const p = ix(sx, i, incx_);
        x[p] = mul(T, alpha, x[p]);
    }
}

pub fn rscal(comptime T: type, n_: BlasInt, alpha: Real(T), x: [*]T, incx_: BlasInt) void {
    scal(T, n_, realScalar(T, alpha), x, incx_);
}

pub fn copy(comptime T: type, n_: BlasInt, x: [*]const T, incx_: BlasInt, y: [*]T, incy_: BlasInt) void {
    if (n_ <= 0 or incx_ == 0 or incy_ == 0) return;
    const n = toUsize(n_);
    if (incx_ == 1 and incy_ == 1) {
        if (comptime isReal(T)) {
            if (parallelCopyUnitReal(T, n, x, y)) return;
            return copyUnitReal(T, n, x, y);
        }
        return copyUnit(T, n, x, y);
    }
    const sx = startIndex(n_, incx_);
    const sy = startIndex(n_, incy_);
    for (0..n) |i| vectorSet(T, y, sy, i, incy_, vectorGet(T, x, sx, i, incx_));
}

pub fn swap(comptime T: type, n_: BlasInt, x: [*]T, incx_: BlasInt, y: [*]T, incy_: BlasInt) void {
    if (n_ <= 0 or incx_ == 0 or incy_ == 0) return;
    const n = toUsize(n_);
    if (incx_ == 1 and incy_ == 1) return swapUnit(T, n, x, y);
    const sx = startIndex(n_, incx_);
    const sy = startIndex(n_, incy_);
    for (0..n) |i| {
        const px = ix(sx, i, incx_);
        const py = ix(sy, i, incy_);
        const t = x[px];
        x[px] = y[py];
        y[py] = t;
    }
}

pub fn axpy(comptime T: type, n_: BlasInt, alpha: T, x: [*]const T, incx_: BlasInt, y: [*]T, incy_: BlasInt) void {
    if (n_ <= 0 or incx_ == 0 or incy_ == 0 or isZero(T, alpha)) return;
    const n = toUsize(n_);
    if (comptime isReal(T)) {
        if (incx_ == 1 and incy_ == 1) {
            if (parallelAxpyUnitReal(T, n, alpha, x, y)) return;
            return axpyUnitReal(T, n, alpha, x, y);
        }
    }
    const sx = startIndex(n_, incx_);
    const sy = startIndex(n_, incy_);
    for (0..n) |i| {
        const py = ix(sy, i, incy_);
        y[py] = add(T, y[py], mul(T, alpha, vectorGet(T, x, sx, i, incx_)));
    }
}

pub fn axpby(comptime T: type, n_: BlasInt, alpha: T, x: [*]const T, incx_: BlasInt, beta: T, y: [*]T, incy_: BlasInt) void {
    if (n_ <= 0 or incx_ == 0 or incy_ == 0) return;
    const n = toUsize(n_);
    if (comptime isReal(T)) {
        if (incx_ == 1 and incy_ == 1) {
            if (parallelAxpbyUnitReal(T, n, alpha, x, beta, y)) return;
            return axpbyUnitReal(T, n, alpha, x, beta, y);
        }
    }
    const sx = startIndex(n_, incx_);
    const sy = startIndex(n_, incy_);
    for (0..n) |i| {
        const py = ix(sy, i, incy_);
        const xv = vectorGet(T, x, sx, i, incx_);
        y[py] = add(T, mul(T, alpha, xv), mul(T, beta, y[py]));
    }
}

pub fn dot(comptime T: type, n_: BlasInt, x: [*]const T, incx_: BlasInt, y: [*]const T, incy_: BlasInt, conjx: bool) T {
    if (n_ <= 0 or incx_ == 0 or incy_ == 0) return zero(T);
    const n = toUsize(n_);
    if (comptime isReal(T)) {
        if (incx_ == 1 and incy_ == 1) {
            if (vector_binary_kernels.dotUnitReal(T, n, x, y)) |result| return result;
            return parallelDotUnitReal(T, n, x, y) orelse dotUnitReal(T, n, x, y);
        }
    }
    const sx = startIndex(n_, incx_);
    const sy = startIndex(n_, incy_);
    var sum = zero(T);
    for (0..n) |i| {
        const a = maybeConj(T, vectorGet(T, x, sx, i, incx_), conjx);
        sum = add(T, sum, mul(T, a, vectorGet(T, y, sy, i, incy_)));
    }
    return sum;
}

pub fn asum(comptime T: type, n_: BlasInt, x: [*]const T, incx_: BlasInt) Real(T) {
    if (n_ <= 0 or incx_ == 0) return 0;
    const n = toUsize(n_);
    if (comptime isReal(T)) {
        if (incx_ == 1) return parallelAsumUnitReal(T, n, x) orelse asumUnitReal(T, n, x);
    }
    const sx = startIndex(n_, incx_);
    var sum: Real(T) = 0;
    for (0..n) |i| sum += abs1(T, vectorGet(T, x, sx, i, incx_));
    return sum;
}

pub fn nrm2(comptime T: type, n_: BlasInt, x: [*]const T, incx_: BlasInt) Real(T) {
    if (n_ <= 0 or incx_ == 0) return 0;
    const n = toUsize(n_);
    if (comptime isReal(T)) {
        if (incx_ == 1) {
            if (nrm2UnitReal(T, n, x)) |result| return result;
        }
    }
    const sx = startIndex(n_, incx_);
    var scale: Real(T) = 0;
    var ssq: Real(T) = 1;
    for (0..n) |i| {
        const v = vectorGet(T, x, sx, i, incx_);
        if (comptime isComplex(T)) {
            inline for (.{ realPart(T, v), imagPart(T, v) }) |component| {
                const ax = @abs(component);
                if (ax != 0) {
                    if (scale < ax) {
                        const r = scale / ax;
                        ssq = 1 + ssq * r * r;
                        scale = ax;
                    } else {
                        const r = ax / scale;
                        ssq += r * r;
                    }
                }
            }
        } else {
            const ax = @abs(v);
            if (ax != 0) {
                if (scale < ax) {
                    const r = scale / ax;
                    ssq = 1 + ssq * r * r;
                    scale = ax;
                } else {
                    const r = ax / scale;
                    ssq += r * r;
                }
            }
        }
    }
    return scale * @sqrt(ssq);
}

pub fn iamax(comptime T: type, n_: BlasInt, x: [*]const T, incx_: BlasInt) BlasInt {
    if (n_ < 1 or incx_ <= 0) return 0;
    const n = toUsize(n_);
    var best: usize = 0;
    var best_abs = abs1(T, x[0]);
    var p: usize = @intCast(incx_);
    for (1..n) |i| {
        const a = abs1(T, x[p]);
        if (a > best_abs) {
            best_abs = a;
            best = i;
        }
        p += @intCast(incx_);
    }
    return @intCast(best + 1);
}

pub fn rot(comptime T: type, n_: BlasInt, x: [*]T, incx_: BlasInt, y: [*]T, incy_: BlasInt, c: Real(T), s: T) void {
    if (n_ <= 0 or incx_ == 0 or incy_ == 0) return;
    const n = toUsize(n_);
    const sx = startIndex(n_, incx_);
    const sy = startIndex(n_, incy_);
    const cc = realScalar(T, c);
    for (0..n) |i| {
        const px = ix(sx, i, incx_);
        const py = ix(sy, i, incy_);
        const xv = x[px];
        const yv = y[py];
        x[px] = add(T, mul(T, cc, xv), mul(T, s, yv));
        y[py] = sub(T, mul(T, cc, yv), mul(T, s, xv));
    }
}

pub fn rotgReal(comptime T: type, a: *T, b: *T, c: *T, s: *T) void {
    const aa = a.*;
    const bb = b.*;
    const roe = if (@abs(bb) > @abs(aa)) bb else aa;
    const scale = @abs(aa) + @abs(bb);
    if (scale == 0) {
        c.* = 1;
        s.* = 0;
        a.* = 0;
        b.* = 0;
        return;
    }
    var r = scale * @sqrt((aa / scale) * (aa / scale) + (bb / scale) * (bb / scale));
    r = std.math.copysign(r, roe);
    c.* = aa / r;
    s.* = bb / r;
    var z: T = 1;
    if (@abs(aa) > @abs(bb)) z = s.*;
    if (@abs(bb) >= @abs(aa) and c.* != 0) z = 1 / c.*;
    a.* = r;
    b.* = z;
}

pub fn rotgComplex(comptime T: type, ca: *T, cb: *T, c: *Real(T), s: *T) void {
    const a = ca.*;
    const b = cb.*;
    const abs_a = abs2(T, a);
    if (abs_a == 0) {
        c.* = 0;
        s.* = one(T);
        ca.* = b;
        return;
    }
    const scale = abs_a + abs2(T, b);
    const norm = scale * @sqrt((abs_a / scale) * (abs_a / scale) + (abs2(T, b) / scale) * (abs2(T, b) / scale));
    const alpha = divv(T, a, realScalar(T, abs_a));
    c.* = abs_a / norm;
    s.* = divv(T, mul(T, alpha, conj(T, b)), realScalar(T, norm));
    ca.* = mul(T, alpha, realScalar(T, norm));
}

pub fn rotm(comptime T: type, n_: BlasInt, x: [*]T, incx_: BlasInt, y: [*]T, incy_: BlasInt, param: [*]const T) void {
    if (n_ <= 0 or incx_ == 0 or incy_ == 0) return;
    const flag = param[0];
    if (flag == -2) return;
    const h11 = param[1];
    const h21 = param[2];
    const h12 = param[3];
    const h22 = param[4];
    const n = toUsize(n_);
    const sx = startIndex(n_, incx_);
    const sy = startIndex(n_, incy_);
    for (0..n) |i| {
        const px = ix(sx, i, incx_);
        const py = ix(sy, i, incy_);
        const w = x[px];
        const z = y[py];
        if (flag < 0) {
            x[px] = w * h11 + z * h12;
            y[py] = w * h21 + z * h22;
        } else if (flag == 0) {
            x[px] = w + z * h12;
            y[py] = w * h21 + z;
        } else {
            x[px] = w * h11 + z;
            y[py] = -w + z * h22;
        }
    }
}

pub fn rotmg(comptime T: type, d1: *T, d2: *T, x1: *T, y1: *const T, param: [*]T) void {
    const gam: T = 4096;
    const gamsq = gam * gam;
    const rgamsq: T = 1 / gamsq;
    var flag: T = undefined;
    var h11: T = 0;
    var h12: T = 0;
    var h21: T = 0;
    var h22: T = 0;

    if (d1.* < 0) {
        flag = -1;
        d1.* = 0;
        d2.* = 0;
        x1.* = 0;
    } else {
        const p2 = d2.* * y1.*;
        if (p2 == 0) {
            flag = -2;
            param[0] = flag;
            return;
        }
        const p1 = d1.* * x1.*;
        const q2 = p2 * y1.*;
        const q1 = p1 * x1.*;
        if (@abs(q1) > @abs(q2)) {
            h21 = -y1.* / x1.*;
            h12 = p2 / p1;
            const u = 1 - h12 * h21;
            if (u > 0) {
                flag = 0;
                d1.* /= u;
                d2.* /= u;
                x1.* *= u;
            } else {
                flag = -1;
                d1.* = 0;
                d2.* = 0;
                x1.* = 0;
            }
        } else {
            if (q2 < 0) {
                flag = -1;
                d1.* = 0;
                d2.* = 0;
                x1.* = 0;
            } else {
                flag = 1;
                h11 = p1 / p2;
                h22 = x1.* / y1.*;
                const u = 1 + h11 * h22;
                const tmp = d2.* / u;
                d2.* = d1.* / u;
                d1.* = tmp;
                x1.* = y1.* * u;
            }
        }
        if (d1.* != 0) {
            while (d1.* <= rgamsq or d1.* >= gamsq) {
                if (flag == 0) {
                    h11 = 1;
                    h22 = 1;
                    flag = -1;
                } else if (flag > 0) {
                    h21 = -1;
                    h12 = 1;
                    flag = -1;
                }
                if (d1.* <= rgamsq) {
                    d1.* *= gamsq;
                    x1.* /= gam;
                    h11 /= gam;
                    h12 /= gam;
                } else {
                    d1.* /= gamsq;
                    x1.* *= gam;
                    h11 *= gam;
                    h12 *= gam;
                }
            }
        }
        if (d2.* != 0) {
            while (@abs(d2.*) <= rgamsq or @abs(d2.*) >= gamsq) {
                if (flag == 0) {
                    h11 = 1;
                    h22 = 1;
                    flag = -1;
                } else if (flag > 0) {
                    h21 = -1;
                    h12 = 1;
                    flag = -1;
                }
                if (@abs(d2.*) <= rgamsq) {
                    d2.* *= gamsq;
                    h21 /= gam;
                    h22 /= gam;
                } else {
                    d2.* /= gamsq;
                    h21 *= gam;
                    h22 *= gam;
                }
            }
        }
    }
    param[0] = flag;
    if (flag < 0) {
        param[1] = h11;
        param[2] = h21;
        param[3] = h12;
        param[4] = h22;
    } else if (flag == 0) {
        param[2] = h21;
        param[3] = h12;
    } else {
        param[1] = h11;
        param[4] = h22;
    }
}
