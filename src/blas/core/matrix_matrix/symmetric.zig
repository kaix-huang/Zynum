// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later

const scalar = @import("../shared/scalar.zig");
const indexing = @import("../shared/indexing.zig");
const matrix_vector_ops = @import("../matrix_vector.zig");
const core_pool = @import("../execution/thread_pool.zig");

pub const BlasInt = scalar.BlasInt;
pub const Order = scalar.Order;
pub const Uplo = scalar.Uplo;
pub const Side = scalar.Side;

const zero = scalar.zero;
const add = scalar.add;
const mul = scalar.mul;
const conj = scalar.conj;
const isComplex = scalar.isComplex;
const isZero = scalar.isZero;

const toUsize = indexing.toUsize;
const matIndex = indexing.matIndex;
const matrixValue = matrix_vector_ops.matrixValue;
const symValue = matrix_vector_ops.symValue;

fn SymmTask(comptime T: type) type {
    return struct {
        side: Side,
        uplo: Uplo,
        m: usize,
        n: usize,
        alpha: T,
        a: [*]const T,
        lda: BlasInt,
        b: [*]const T,
        ldb: BlasInt,
        beta: T,
        c: [*]T,
        ldc: BlasInt,
        herm: bool,
        task_index: usize,
        task_count: usize,
    };
}

fn runSymmColumns(comptime T: type, task: SymmTask(T)) void {
    var j = task.task_index;
    while (j < task.n) : (j += task.task_count) {
        for (0..task.m) |i| {
            var sum = zero(T);
            if (task.side == .left) {
                for (0..task.m) |p| sum = add(T, sum, mul(T, symValue(T, task.uplo, task.a, task.lda, i, p, task.herm), task.b[matIndex(task.ldb, p, j)]));
            } else {
                for (0..task.n) |p| sum = add(T, sum, mul(T, task.b[matIndex(task.ldb, i, p)], symValue(T, task.uplo, task.a, task.lda, p, j, task.herm)));
            }
            const idxc = matIndex(task.ldc, i, j);
            task.c[idxc] = add(T, mul(T, task.alpha, sum), if (isZero(T, task.beta)) zero(T) else mul(T, task.beta, task.c[idxc]));
        }
    }
}

fn runSymmTask(comptime T: type, raw_tasks: *const anyopaque, index: usize) void {
    const tasks: [*]const SymmTask(T) = @ptrCast(@alignCast(raw_tasks));
    runSymmColumns(T, tasks[index]);
}

fn runSymmTaskF32(raw_tasks: *const anyopaque, index: usize) void {
    runSymmTask(f32, raw_tasks, index);
}

fn runSymmTaskF64(raw_tasks: *const anyopaque, index: usize) void {
    runSymmTask(f64, raw_tasks, index);
}

fn runSymmTaskC32(raw_tasks: *const anyopaque, index: usize) void {
    runSymmTask(scalar.ComplexF32, raw_tasks, index);
}

fn runSymmTaskC64(raw_tasks: *const anyopaque, index: usize) void {
    runSymmTask(scalar.ComplexF64, raw_tasks, index);
}

fn runParallelSymm(comptime T: type, tasks: []const SymmTask(T)) bool {
    const runner = if (T == f32)
        runSymmTaskF32
    else if (T == f64)
        runSymmTaskF64
    else if (T == scalar.ComplexF32)
        runSymmTaskC32
    else if (T == scalar.ComplexF64)
        runSymmTaskC64
    else
        @compileError("parallel SYMM supports BLAS scalar types");
    return core_pool.runLowLatency(runner, @ptrCast(tasks.ptr), tasks.len);
}

pub fn symm(comptime T: type, side: Side, uplo: Uplo, m_: BlasInt, n_: BlasInt, alpha: T, a: [*]const T, lda: BlasInt, b: [*]const T, ldb: BlasInt, beta: T, c: [*]T, ldc: BlasInt, herm: bool) void {
    if (m_ <= 0 or n_ <= 0) return;
    const m = toUsize(m_);
    const n = toUsize(n_);
    var tasks: [core_pool.max_tasks]SymmTask(T) = undefined;
    const order = if (side == .left) m else n;
    const work = m *| n *| order;
    const task_count = if (work >= 8 * 1024 * 1024) @min(core_pool.taskCount(n, 4), 32) else 1;
    if (task_count > 1) {
        for (tasks[0..task_count], 0..) |*task, task_index| {
            task.* = .{
                .side = side,
                .uplo = uplo,
                .m = m,
                .n = n,
                .alpha = alpha,
                .a = a,
                .lda = lda,
                .b = b,
                .ldb = ldb,
                .beta = beta,
                .c = c,
                .ldc = ldc,
                .herm = herm,
                .task_index = task_index,
                .task_count = task_count,
            };
        }
        if (runParallelSymm(T, tasks[0..task_count])) return;
    }
    runSymmColumns(T, .{
        .side = side,
        .uplo = uplo,
        .m = m,
        .n = n,
        .alpha = alpha,
        .a = a,
        .lda = lda,
        .b = b,
        .ldb = ldb,
        .beta = beta,
        .c = c,
        .ldc = ldc,
        .herm = herm,
        .task_index = 0,
        .task_count = 1,
    });
}

fn SyrkTask(comptime T: type) type {
    return struct {
        uplo: Uplo,
        trans: Order,
        n: usize,
        k: usize,
        alpha: T,
        a: [*]const T,
        lda: BlasInt,
        beta: T,
        c: [*]T,
        ldc: BlasInt,
        herm: bool,
        task_index: usize,
        task_count: usize,
    };
}

fn runSyrkColumns(comptime T: type, task: SyrkTask(T)) void {
    var j = task.task_index;
    while (j < task.n) : (j += task.task_count) {
        const row0: usize = if (task.uplo == .upper) 0 else j;
        const row1: usize = if (task.uplo == .upper) j + 1 else task.n;
        for (row0..row1) |i| {
            var sum = zero(T);
            for (0..task.k) |p| {
                const ai = if (task.trans == .no_trans) task.a[matIndex(task.lda, i, p)] else matrixValue(T, task.trans, task.a, task.lda, i, p);
                var aj = if (task.trans == .no_trans) task.a[matIndex(task.lda, j, p)] else matrixValue(T, task.trans, task.a, task.lda, j, p);
                if (task.herm) aj = conj(T, aj);
                sum = add(T, sum, mul(T, ai, aj));
            }
            const idxc = matIndex(task.ldc, i, j);
            task.c[idxc] = add(T, mul(T, task.alpha, sum), if (isZero(T, task.beta)) zero(T) else mul(T, task.beta, task.c[idxc]));
            if (task.herm and i == j) {
                if (comptime isComplex(T)) task.c[idxc].im = 0;
            }
        }
    }
}

fn runSyrkTask(comptime T: type, raw_tasks: *const anyopaque, index: usize) void {
    const tasks: [*]const SyrkTask(T) = @ptrCast(@alignCast(raw_tasks));
    runSyrkColumns(T, tasks[index]);
}

fn runSyrkTaskF32(raw_tasks: *const anyopaque, index: usize) void {
    runSyrkTask(f32, raw_tasks, index);
}

fn runSyrkTaskF64(raw_tasks: *const anyopaque, index: usize) void {
    runSyrkTask(f64, raw_tasks, index);
}

fn runSyrkTaskC32(raw_tasks: *const anyopaque, index: usize) void {
    runSyrkTask(scalar.ComplexF32, raw_tasks, index);
}

fn runSyrkTaskC64(raw_tasks: *const anyopaque, index: usize) void {
    runSyrkTask(scalar.ComplexF64, raw_tasks, index);
}

fn runParallelSyrk(comptime T: type, tasks: []const SyrkTask(T)) bool {
    const runner = if (T == f32)
        runSyrkTaskF32
    else if (T == f64)
        runSyrkTaskF64
    else if (T == scalar.ComplexF32)
        runSyrkTaskC32
    else if (T == scalar.ComplexF64)
        runSyrkTaskC64
    else
        @compileError("parallel SYRK supports BLAS scalar types");
    return core_pool.runLowLatency(runner, @ptrCast(tasks.ptr), tasks.len);
}

pub fn syrk(comptime T: type, uplo: Uplo, trans_: Order, n_: BlasInt, k_: BlasInt, alpha: T, a: [*]const T, lda: BlasInt, beta: T, c: [*]T, ldc: BlasInt, herm: bool) void {
    if (n_ <= 0) return;
    const n = toUsize(n_);
    const k = toUsize(k_);
    var tasks: [core_pool.max_tasks]SyrkTask(T) = undefined;
    const work = n *| n *| k;
    const task_count = if (work >= 128 * 1024) @min(core_pool.taskCount(n, 8), 32) else 1;
    if (task_count > 1) {
        for (tasks[0..task_count], 0..) |*task, task_index| {
            task.* = .{
                .uplo = uplo,
                .trans = trans_,
                .n = n,
                .k = k,
                .alpha = alpha,
                .a = a,
                .lda = lda,
                .beta = beta,
                .c = c,
                .ldc = ldc,
                .herm = herm,
                .task_index = task_index,
                .task_count = task_count,
            };
        }
        if (runParallelSyrk(T, tasks[0..task_count])) return;
    }
    runSyrkColumns(T, .{
        .uplo = uplo,
        .trans = trans_,
        .n = n,
        .k = k,
        .alpha = alpha,
        .a = a,
        .lda = lda,
        .beta = beta,
        .c = c,
        .ldc = ldc,
        .herm = herm,
        .task_index = 0,
        .task_count = 1,
    });
}

fn Syr2kTask(comptime T: type) type {
    return struct {
        uplo: Uplo,
        trans: Order,
        n: usize,
        k: usize,
        alpha: T,
        a: [*]const T,
        lda: BlasInt,
        b: [*]const T,
        ldb: BlasInt,
        beta: T,
        c: [*]T,
        ldc: BlasInt,
        herm: bool,
        task_index: usize,
        task_count: usize,
    };
}

fn runSyr2kColumns(comptime T: type, task: Syr2kTask(T)) void {
    var j = task.task_index;
    while (j < task.n) : (j += task.task_count) {
        const row0: usize = if (task.uplo == .upper) 0 else j;
        const row1: usize = if (task.uplo == .upper) j + 1 else task.n;
        for (row0..row1) |i| {
            var sum = zero(T);
            for (0..task.k) |p| {
                const ai = matrixValue(T, task.trans, task.a, task.lda, i, p);
                const bi = matrixValue(T, task.trans, task.b, task.ldb, i, p);
                var aj = matrixValue(T, task.trans, task.a, task.lda, j, p);
                var bj = matrixValue(T, task.trans, task.b, task.ldb, j, p);
                if (task.herm) {
                    aj = conj(T, aj);
                    bj = conj(T, bj);
                    sum = add(T, sum, add(T, mul(T, task.alpha, mul(T, ai, bj)), mul(T, conj(T, task.alpha), mul(T, bi, aj))));
                } else {
                    sum = add(T, sum, add(T, mul(T, ai, bj), mul(T, bi, aj)));
                }
            }
            const idxc = matIndex(task.ldc, i, j);
            const prod = if (task.herm) sum else mul(T, task.alpha, sum);
            task.c[idxc] = add(T, prod, if (isZero(T, task.beta)) zero(T) else mul(T, task.beta, task.c[idxc]));
            if (task.herm and i == j) {
                if (comptime isComplex(T)) task.c[idxc].im = 0;
            }
        }
    }
}

fn runSyr2kTask(comptime T: type, raw_tasks: *const anyopaque, index: usize) void {
    const tasks: [*]const Syr2kTask(T) = @ptrCast(@alignCast(raw_tasks));
    runSyr2kColumns(T, tasks[index]);
}

fn runSyr2kTaskF32(raw_tasks: *const anyopaque, index: usize) void {
    runSyr2kTask(f32, raw_tasks, index);
}

fn runSyr2kTaskF64(raw_tasks: *const anyopaque, index: usize) void {
    runSyr2kTask(f64, raw_tasks, index);
}

fn runSyr2kTaskC32(raw_tasks: *const anyopaque, index: usize) void {
    runSyr2kTask(scalar.ComplexF32, raw_tasks, index);
}

fn runSyr2kTaskC64(raw_tasks: *const anyopaque, index: usize) void {
    runSyr2kTask(scalar.ComplexF64, raw_tasks, index);
}

fn runParallelSyr2k(comptime T: type, tasks: []const Syr2kTask(T)) bool {
    const runner = if (T == f32)
        runSyr2kTaskF32
    else if (T == f64)
        runSyr2kTaskF64
    else if (T == scalar.ComplexF32)
        runSyr2kTaskC32
    else if (T == scalar.ComplexF64)
        runSyr2kTaskC64
    else
        @compileError("parallel SYR2K supports BLAS scalar types");
    return core_pool.runLowLatency(runner, @ptrCast(tasks.ptr), tasks.len);
}

pub fn syr2k(comptime T: type, uplo: Uplo, trans_: Order, n_: BlasInt, k_: BlasInt, alpha: T, a: [*]const T, lda: BlasInt, b: [*]const T, ldb: BlasInt, beta: T, c: [*]T, ldc: BlasInt, herm: bool) void {
    if (n_ <= 0) return;
    const n = toUsize(n_);
    const k = toUsize(k_);
    var tasks: [core_pool.max_tasks]Syr2kTask(T) = undefined;
    const work = n *| n *| k;
    const task_count = if (work >= 128 * 1024) @min(core_pool.taskCount(n, 8), 32) else 1;
    if (task_count > 1) {
        for (tasks[0..task_count], 0..) |*task, task_index| {
            task.* = .{
                .uplo = uplo,
                .trans = trans_,
                .n = n,
                .k = k,
                .alpha = alpha,
                .a = a,
                .lda = lda,
                .b = b,
                .ldb = ldb,
                .beta = beta,
                .c = c,
                .ldc = ldc,
                .herm = herm,
                .task_index = task_index,
                .task_count = task_count,
            };
        }
        if (runParallelSyr2k(T, tasks[0..task_count])) return;
    }
    runSyr2kColumns(T, .{
        .uplo = uplo,
        .trans = trans_,
        .n = n,
        .k = k,
        .alpha = alpha,
        .a = a,
        .lda = lda,
        .b = b,
        .ldb = ldb,
        .beta = beta,
        .c = c,
        .ldc = ldc,
        .herm = herm,
        .task_index = 0,
        .task_count = 1,
    });
}
