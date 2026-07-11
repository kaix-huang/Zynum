// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later

const std = @import("std");
const builtin = @import("builtin");

const scalar = @import("../shared/scalar.zig");
const indexing = @import("../shared/indexing.zig");
const matrix_vector_ops = @import("../matrix_vector.zig");
const core_pool = @import("../execution/thread_pool.zig");
const runtime = @import("../../runtime.zig");

const BlasInt = scalar.BlasInt;
const Order = scalar.Order;
const Uplo = scalar.Uplo;
const Diag = scalar.Diag;

const zero = scalar.zero;
const add = scalar.add;
const mul = scalar.mul;
const conj = scalar.conj;
const isOne = scalar.isOne;

const matIndex = indexing.matIndex;
const triValue = matrix_vector_ops.triValue;
const trsv = matrix_vector_ops.trsv;

const parallel_min_work = 8 * 1024 * 1024;
const parallel_max_tasks = 32;
const parallel_min_rows_per_task = 4;

pub const Operation = enum {
    multiply,
    solve,
};

fn opIsUpper(uplo: Uplo, trans_: Order) bool {
    return (trans_ == .no_trans and uplo == .upper) or (trans_ != .no_trans and uplo == .lower);
}

fn scaleDenseRows(comptime T: type, n: usize, first_row: usize, end_row: usize, alpha: T, b: [*]T, ldb: BlasInt) void {
    if (isOne(T, alpha)) return;
    for (first_row..end_row) |i| {
        for (0..n) |j| {
            b[matIndex(ldb, i, j)] = mul(T, alpha, b[matIndex(ldb, i, j)]);
        }
    }
}

fn conjugateDenseRow(comptime T: type, n: usize, row: usize, b: [*]T, ldb: BlasInt) void {
    for (0..n) |j| {
        const index = matIndex(ldb, row, j);
        b[index] = conj(T, b[index]);
    }
}

fn solveRightRow(comptime T: type, uplo: Uplo, trans_: Order, diag: Diag, n_: BlasInt, n: usize, a: [*]const T, lda: BlasInt, b: [*]T, ldb: BlasInt, row: usize) void {
    const x = b + matIndex(ldb, row, 0);
    switch (trans_) {
        .no_trans => trsv(T, uplo, .trans, diag, n_, a, lda, x, ldb),
        .trans => trsv(T, uplo, .no_trans, diag, n_, a, lda, x, ldb),
        .conj_trans => {
            // Transposing X * A^H = B requires a conj(A) solve.
            if (comptime scalar.isComplex(T)) conjugateDenseRow(T, n, row, b, ldb);
            trsv(T, uplo, .no_trans, diag, n_, a, lda, x, ldb);
            if (comptime scalar.isComplex(T)) conjugateDenseRow(T, n, row, b, ldb);
        },
    }
}

fn multiplyRightRows(comptime T: type, uplo: Uplo, trans_: Order, diag: Diag, n: usize, alpha: T, a: [*]const T, lda: BlasInt, b: [*]T, ldb: BlasInt, first_row: usize, end_row: usize) void {
    for (first_row..end_row) |i| {
        if (opIsUpper(uplo, trans_)) {
            var cc: usize = n;
            while (cc > 0) {
                cc -= 1;
                var sum = zero(T);
                for (0..n) |p| sum = add(T, sum, mul(T, b[matIndex(ldb, i, p)], triValue(T, uplo, diag, trans_, a, lda, p, cc)));
                b[matIndex(ldb, i, cc)] = mul(T, alpha, sum);
            }
        } else {
            for (0..n) |j| {
                var sum = zero(T);
                for (0..n) |p| sum = add(T, sum, mul(T, b[matIndex(ldb, i, p)], triValue(T, uplo, diag, trans_, a, lda, p, j)));
                b[matIndex(ldb, i, j)] = mul(T, alpha, sum);
            }
        }
    }
}

fn solveRightRows(comptime T: type, uplo: Uplo, trans_: Order, diag: Diag, n: usize, a: [*]const T, lda: BlasInt, b: [*]T, ldb: BlasInt, first_row: usize, end_row: usize) void {
    for (first_row..end_row) |i| {
        solveRightRow(T, uplo, trans_, diag, @intCast(n), n, a, lda, b, ldb, i);
    }
}

fn RightTask(comptime T: type) type {
    return struct {
        operation: Operation,
        uplo: Uplo,
        trans: Order,
        diag: Diag,
        n: usize,
        alpha: T,
        a: [*]const T,
        lda: BlasInt,
        b: [*]T,
        ldb: BlasInt,
        first_row: usize,
        end_row: usize,
    };
}

fn runRightRows(comptime T: type, task: RightTask(T)) void {
    switch (task.operation) {
        .multiply => multiplyRightRows(T, task.uplo, task.trans, task.diag, task.n, task.alpha, task.a, task.lda, task.b, task.ldb, task.first_row, task.end_row),
        .solve => {
            scaleDenseRows(T, task.n, task.first_row, task.end_row, task.alpha, task.b, task.ldb);
            solveRightRows(T, task.uplo, task.trans, task.diag, task.n, task.a, task.lda, task.b, task.ldb, task.first_row, task.end_row);
        },
    }
}

fn runRightTask(comptime T: type, raw_tasks: *const anyopaque, index: usize) void {
    const tasks: [*]const RightTask(T) = @ptrCast(@alignCast(raw_tasks));
    runRightRows(T, tasks[index]);
}

fn runRightTaskF32(raw_tasks: *const anyopaque, index: usize) void {
    runRightTask(f32, raw_tasks, index);
}

fn runRightTaskF64(raw_tasks: *const anyopaque, index: usize) void {
    runRightTask(f64, raw_tasks, index);
}

fn runRightTaskC32(raw_tasks: *const anyopaque, index: usize) void {
    runRightTask(scalar.ComplexF32, raw_tasks, index);
}

fn runRightTaskC64(raw_tasks: *const anyopaque, index: usize) void {
    runRightTask(scalar.ComplexF64, raw_tasks, index);
}

fn runParallelRight(comptime T: type, tasks: []const RightTask(T)) bool {
    const runner = if (T == f32)
        runRightTaskF32
    else if (T == f64)
        runRightTaskF64
    else if (T == scalar.ComplexF32)
        runRightTaskC32
    else if (T == scalar.ComplexF64)
        runRightTaskC64
    else
        @compileError("parallel TRMM/TRSM supports BLAS scalar types");
    return core_pool.runLowLatency(runner, @ptrCast(tasks.ptr), tasks.len);
}

fn rightTaskCount(m: usize, n: usize) usize {
    if (comptime builtin.cpu.arch != .x86_64) return 1;
    const work = m *| n *| n;
    if (work < parallel_min_work) return 1;
    return @min(core_pool.taskCount(m, parallel_min_rows_per_task), parallel_max_tasks);
}

pub noinline fn tryDispatch(comptime T: type, operation: Operation, uplo: Uplo, trans_: Order, diag: Diag, m: usize, n: usize, alpha: T, a: [*]const T, lda: BlasInt, b: [*]T, ldb: BlasInt) bool {
    const task_count = rightTaskCount(m, n);
    if (task_count <= 1) return false;

    var tasks: [core_pool.max_tasks]RightTask(T) = undefined;
    const base_rows = m / task_count;
    const extra_rows = m % task_count;
    var first_row: usize = 0;
    for (tasks[0..task_count], 0..) |*task, task_index| {
        const row_count = base_rows + @intFromBool(task_index < extra_rows);
        task.* = .{
            .operation = operation,
            .uplo = uplo,
            .trans = trans_,
            .diag = diag,
            .n = n,
            .alpha = alpha,
            .a = a,
            .lda = lda,
            .b = b,
            .ldb = ldb,
            .first_row = first_row,
            .end_row = first_row + row_count,
        };
        first_row += row_count;
    }
    return runParallelRight(T, tasks[0..task_count]);
}

fn testValue(comptime T: type, real: f64, imaginary: f64) T {
    if (comptime scalar.isComplex(T)) {
        return .{ .re = @floatCast(real), .im = @floatCast(imaginary) };
    }
    return @floatCast(real);
}

fn fillTestOperands(comptime T: type, a: []T, lda: usize, n: usize, b: []T, ldb: usize, column_count: usize) void {
    @memset(a, zero(T));
    for (0..n) |j| {
        for (0..n) |i| {
            const raw_real = @as(f64, @floatFromInt((i * 7 + j * 11 + 3) % 19)) - 9.0;
            const raw_imaginary = @as(f64, @floatFromInt((i * 13 + j * 5 + 1) % 17)) - 8.0;
            a[i + j * lda] = if (i == j)
                testValue(T, 2.0 + raw_real / 64.0, raw_imaginary / 96.0)
            else
                testValue(T, raw_real / 64.0, raw_imaginary / 96.0);
        }
    }
    for (0..column_count) |j| {
        for (0..ldb) |i| {
            const raw_real = @as(f64, @floatFromInt((i * 5 + j * 3 + 2) % 23)) - 11.0;
            const raw_imaginary = @as(f64, @floatFromInt((i * 3 + j * 7 + 4) % 21)) - 10.0;
            b[i + j * ldb] = testValue(T, raw_real / 16.0, raw_imaginary / 24.0);
        }
    }
}

fn expectParallelMatchesSingle(comptime T: type, operation: Operation, uplo: Uplo, trans_: Order, diag: Diag) !void {
    const m = 17;
    const n = 13;
    const lda = n + 2;
    const ldb = m + 3;
    const alpha = testValue(T, -0.625, 0.1875);
    var a: [lda * n]T = undefined;
    var initial_b: [ldb * n]T = undefined;
    fillTestOperands(T, &a, lda, n, &initial_b, ldb, n);
    var expected = initial_b;
    var actual = initial_b;

    switch (operation) {
        .multiply => multiplyRightRows(T, uplo, trans_, diag, n, alpha, &a, lda, &expected, ldb, 0, m),
        .solve => {
            scaleDenseRows(T, n, 0, m, alpha, &expected, ldb);
            solveRightRows(T, uplo, trans_, diag, n, &a, lda, &expected, ldb, 0, m);
        },
    }

    const task_count: usize = @min(runtime.maxThreads(), 4);
    var tasks: [4]RightTask(T) = undefined;
    const base_rows = m / task_count;
    const extra_rows = m % task_count;
    var first_row: usize = 0;
    for (tasks[0..task_count], 0..) |*task, task_index| {
        const row_count = base_rows + @intFromBool(task_index < extra_rows);
        task.* = .{
            .operation = operation,
            .uplo = uplo,
            .trans = trans_,
            .diag = diag,
            .n = n,
            .alpha = alpha,
            .a = &a,
            .lda = @intCast(lda),
            .b = &actual,
            .ldb = @intCast(ldb),
            .first_row = first_row,
            .end_row = first_row + row_count,
        };
        first_row += row_count;
    }
    try std.testing.expect(runParallelRight(T, tasks[0..task_count]));
    try std.testing.expectEqualSlices(T, &expected, &actual);
}

const BusyDispatchTask = struct {
    a: [*]const f32,
    b: [*]f32,
    result: *std.atomic.Value(u8),

    fn run(raw_tasks: *const anyopaque, index: usize) void {
        if (index != 0) return;
        const tasks: [*]const BusyDispatchTask = @ptrCast(@alignCast(raw_tasks));
        const task = tasks[0];
        const dispatched = tryDispatch(f32, .multiply, .upper, .no_trans, .non_unit, 512, 128, 0.75, task.a, 129, task.b, 514);
        task.result.store(@intFromBool(dispatched), .release);
    }
};

test "right TRMM and TRSM parallel tasks preserve every type and triangular mode" {
    runtime.setMaxThreads(4);
    defer {
        runtime.setMaxThreads(0);
        core_pool.shutdown();
    }
    if (runtime.maxThreads() <= 1) return error.SkipZigTest;

    inline for (.{ f32, f64, scalar.ComplexF32, scalar.ComplexF64 }) |T| {
        inline for (.{ Operation.multiply, Operation.solve }) |operation| {
            inline for (.{ Uplo.upper, Uplo.lower }) |uplo| {
                inline for (.{ Order.no_trans, Order.trans, Order.conj_trans }) |trans_| {
                    inline for (.{ Diag.non_unit, Diag.unit }) |diag| {
                        try expectParallelMatchesSingle(T, operation, uplo, trans_, diag);
                    }
                }
            }
        }
    }
}

test "right row-parallel gate switches at 511 and 512 rows" {
    runtime.setMaxThreads(4);
    defer runtime.setMaxThreads(0);
    if (runtime.maxThreads() <= 1) return error.SkipZigTest;

    try std.testing.expectEqual(@as(usize, 1), rightTaskCount(128, 128));
    if (comptime builtin.cpu.arch != .x86_64) {
        try std.testing.expectEqual(@as(usize, 1), rightTaskCount(512, 128));
        return;
    }
    try std.testing.expectEqual(@as(usize, 1), rightTaskCount(511, 128));
    try std.testing.expect(rightTaskCount(512, 128) > 1);
    try std.testing.expect(rightTaskCount(512, 128) <= parallel_max_tasks);

    runtime.setMaxThreads(1);
    try std.testing.expectEqual(@as(usize, 1), rightTaskCount(512, 128));
}

test "right dispatch returns false without writes for one thread" {
    const allocator = std.testing.allocator;
    const a = try allocator.alloc(f32, 129 * 128);
    defer allocator.free(a);
    const b = try allocator.alloc(f32, 514 * 128);
    defer allocator.free(b);
    const original = try allocator.alloc(f32, b.len);
    defer allocator.free(original);
    fillTestOperands(f32, a, 129, 128, b, 514, 128);
    @memcpy(original, b);

    runtime.setMaxThreads(1);
    defer {
        runtime.setMaxThreads(0);
        core_pool.shutdown();
    }
    try std.testing.expect(!tryDispatch(f32, .solve, .lower, .trans, .non_unit, 512, 128, -0.75, a.ptr, 129, b.ptr, 514));
    try std.testing.expectEqualSlices(f32, original, b);
}

test "right dispatch failure returns false without writes" {
    const allocator = std.testing.allocator;
    const a = try allocator.alloc(f32, 129 * 128);
    defer allocator.free(a);
    const b = try allocator.alloc(f32, 514 * 128);
    defer allocator.free(b);
    const original = try allocator.alloc(f32, b.len);
    defer allocator.free(original);
    fillTestOperands(f32, a, 129, 128, b, 514, 128);
    @memcpy(original, b);

    runtime.setMaxThreads(2);
    defer {
        runtime.setMaxThreads(0);
        core_pool.shutdown();
    }
    var result = std.atomic.Value(u8).init(2);
    const tasks = [_]BusyDispatchTask{
        .{ .a = a.ptr, .b = b.ptr, .result = &result },
        .{ .a = a.ptr, .b = b.ptr, .result = &result },
    };
    try std.testing.expect(core_pool.runLowLatency(BusyDispatchTask.run, @ptrCast(&tasks), tasks.len));
    try std.testing.expectEqual(@as(u8, 0), result.load(.acquire));
    try std.testing.expectEqualSlices(f32, original, b);
}
