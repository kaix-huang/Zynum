// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later

const std = @import("std");

const scalar = @import("../shared/scalar.zig");
const indexing = @import("../shared/indexing.zig");
const matrix_vector_ops = @import("../matrix_vector.zig");
const core_pool = @import("../execution/thread_pool.zig");
const runtime = @import("../../runtime.zig");

pub const BlasInt = scalar.BlasInt;
pub const Order = scalar.Order;
pub const Uplo = scalar.Uplo;
pub const Diag = scalar.Diag;
pub const Side = scalar.Side;

const zero = scalar.zero;
const add = scalar.add;
const mul = scalar.mul;
const conj = scalar.conj;
const isOne = scalar.isOne;

const toUsize = indexing.toUsize;
const matIndex = indexing.matIndex;
const triValue = matrix_vector_ops.triValue;
const trmv = matrix_vector_ops.trmv;
const trsv = matrix_vector_ops.trsv;

const parallel_left_min_work = 8 * 1024 * 1024;
const parallel_left_max_tasks = 32;
const parallel_left_min_columns_per_task = 4;

const LeftOperation = enum {
    multiply,
    solve,
};

fn opIsUpper(uplo: Uplo, trans_: Order) bool {
    return (trans_ == .no_trans and uplo == .upper) or (trans_ != .no_trans and uplo == .lower);
}

fn scaleDenseColumns(comptime T: type, m: usize, first_col: usize, end_col: usize, alpha: T, b: [*]T, ldb: BlasInt) void {
    if (isOne(T, alpha)) return;
    for (first_col..end_col) |j| {
        for (0..m) |i| {
            b[matIndex(ldb, i, j)] = mul(T, alpha, b[matIndex(ldb, i, j)]);
        }
    }
}

fn scaleDenseMatrix(comptime T: type, m: usize, n: usize, alpha: T, b: [*]T, ldb: BlasInt) void {
    scaleDenseColumns(T, m, 0, n, alpha, b, ldb);
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

fn LeftTask(comptime T: type) type {
    return struct {
        operation: LeftOperation,
        uplo: Uplo,
        trans: Order,
        diag: Diag,
        m: usize,
        alpha: T,
        a: [*]const T,
        lda: BlasInt,
        b: [*]T,
        ldb: BlasInt,
        first_col: usize,
        end_col: usize,
    };
}

fn runLeftColumns(comptime T: type, task: LeftTask(T)) void {
    switch (task.operation) {
        .multiply => {
            for (task.first_col..task.end_col) |j| {
                trmv(T, task.uplo, task.trans, task.diag, @intCast(task.m), task.a, task.lda, task.b + matIndex(task.ldb, 0, j), 1);
                scaleDenseColumns(T, task.m, j, j + 1, task.alpha, task.b, task.ldb);
            }
        },
        .solve => {
            scaleDenseColumns(T, task.m, task.first_col, task.end_col, task.alpha, task.b, task.ldb);
            for (task.first_col..task.end_col) |j| {
                trsv(T, task.uplo, task.trans, task.diag, @intCast(task.m), task.a, task.lda, task.b + matIndex(task.ldb, 0, j), 1);
            }
        },
    }
}

fn runLeftTask(comptime T: type, raw_tasks: *const anyopaque, index: usize) void {
    const tasks: [*]const LeftTask(T) = @ptrCast(@alignCast(raw_tasks));
    runLeftColumns(T, tasks[index]);
}

fn runLeftTaskF32(raw_tasks: *const anyopaque, index: usize) void {
    runLeftTask(f32, raw_tasks, index);
}

fn runLeftTaskF64(raw_tasks: *const anyopaque, index: usize) void {
    runLeftTask(f64, raw_tasks, index);
}

fn runLeftTaskC32(raw_tasks: *const anyopaque, index: usize) void {
    runLeftTask(scalar.ComplexF32, raw_tasks, index);
}

fn runLeftTaskC64(raw_tasks: *const anyopaque, index: usize) void {
    runLeftTask(scalar.ComplexF64, raw_tasks, index);
}

fn runParallelLeft(comptime T: type, tasks: []const LeftTask(T)) bool {
    const runner = if (T == f32)
        runLeftTaskF32
    else if (T == f64)
        runLeftTaskF64
    else if (T == scalar.ComplexF32)
        runLeftTaskC32
    else if (T == scalar.ComplexF64)
        runLeftTaskC64
    else
        @compileError("parallel TRMM/TRSM supports BLAS scalar types");
    return core_pool.runLowLatency(runner, @ptrCast(tasks.ptr), tasks.len);
}

fn runLeft(comptime T: type, operation: LeftOperation, uplo: Uplo, trans_: Order, diag: Diag, m: usize, n: usize, alpha: T, a: [*]const T, lda: BlasInt, b: [*]T, ldb: BlasInt) void {
    var tasks: [core_pool.max_tasks]LeftTask(T) = undefined;
    const work = m *| m *| n;
    const task_count = if (work >= parallel_left_min_work)
        @min(core_pool.taskCount(n, parallel_left_min_columns_per_task), parallel_left_max_tasks)
    else
        1;
    if (task_count > 1) {
        const base_columns = n / task_count;
        const extra_columns = n % task_count;
        var first_col: usize = 0;
        for (tasks[0..task_count], 0..) |*task, task_index| {
            const column_count = base_columns + @intFromBool(task_index < extra_columns);
            task.* = .{
                .operation = operation,
                .uplo = uplo,
                .trans = trans_,
                .diag = diag,
                .m = m,
                .alpha = alpha,
                .a = a,
                .lda = lda,
                .b = b,
                .ldb = ldb,
                .first_col = first_col,
                .end_col = first_col + column_count,
            };
            first_col += column_count;
        }
        if (runParallelLeft(T, tasks[0..task_count])) return;
    }
    runLeftColumns(T, .{
        .operation = operation,
        .uplo = uplo,
        .trans = trans_,
        .diag = diag,
        .m = m,
        .alpha = alpha,
        .a = a,
        .lda = lda,
        .b = b,
        .ldb = ldb,
        .first_col = 0,
        .end_col = n,
    });
}

pub fn trmm(comptime T: type, side: Side, uplo: Uplo, trans_: Order, diag: Diag, m_: BlasInt, n_: BlasInt, alpha: T, a: [*]const T, lda: BlasInt, b: [*]T, ldb: BlasInt) void {
    if (m_ <= 0 or n_ <= 0) return;
    const m = toUsize(m_);
    const n = toUsize(n_);
    if (side == .left) {
        runLeft(T, .multiply, uplo, trans_, diag, m, n, alpha, a, lda, b, ldb);
    } else {
        for (0..m) |i| {
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
}

pub fn trsm(comptime T: type, side: Side, uplo: Uplo, trans_: Order, diag: Diag, m_: BlasInt, n_: BlasInt, alpha: T, a: [*]const T, lda: BlasInt, b: [*]T, ldb: BlasInt) void {
    if (m_ <= 0 or n_ <= 0) return;
    const m = toUsize(m_);
    const n = toUsize(n_);
    if (side == .left) {
        runLeft(T, .solve, uplo, trans_, diag, m, n, alpha, a, lda, b, ldb);
    } else {
        scaleDenseMatrix(T, m, n, alpha, b, ldb);
        for (0..m) |i| {
            solveRightRow(T, uplo, trans_, diag, n_, n, a, lda, b, ldb, i);
        }
    }
}

fn triangularTestValue(comptime T: type, real: f64, imaginary: f64) T {
    if (comptime scalar.isComplex(T)) {
        return .{ .re = @floatCast(real), .im = @floatCast(imaginary) };
    }
    return @floatCast(real);
}

fn fillTriangularTestOperands(comptime T: type, a: []T, lda: usize, m: usize, b: []T, ldb: usize, n: usize) void {
    @memset(a, zero(T));
    for (0..m) |j| {
        for (0..m) |i| {
            const raw_real = @as(f64, @floatFromInt((i * 7 + j * 11 + 3) % 19)) - 9.0;
            const raw_imaginary = @as(f64, @floatFromInt((i * 13 + j * 5 + 1) % 17)) - 8.0;
            a[i + j * lda] = if (i == j)
                triangularTestValue(T, 2.0 + raw_real / 64.0, raw_imaginary / 96.0)
            else
                triangularTestValue(T, raw_real / 64.0, raw_imaginary / 96.0);
        }
    }
    for (0..n) |j| {
        for (0..ldb) |i| {
            const raw_real = @as(f64, @floatFromInt((i * 5 + j * 3 + 2) % 23)) - 11.0;
            const raw_imaginary = @as(f64, @floatFromInt((i * 3 + j * 7 + 4) % 21)) - 10.0;
            b[i + j * ldb] = triangularTestValue(T, raw_real / 16.0, raw_imaginary / 24.0);
        }
    }
}

fn expectParallelLeftMatchesSingle(comptime T: type, operation: LeftOperation, uplo: Uplo, trans_: Order, diag: Diag) !void {
    const m = 17;
    const n = 13;
    const lda = m + 2;
    const ldb = m + 3;
    const alpha = triangularTestValue(T, -0.625, 0.1875);
    var a: [lda * m]T = undefined;
    var initial_b: [ldb * n]T = undefined;
    fillTriangularTestOperands(T, &a, lda, m, &initial_b, ldb, n);
    var expected = initial_b;
    var actual = initial_b;

    runLeftColumns(T, .{
        .operation = operation,
        .uplo = uplo,
        .trans = trans_,
        .diag = diag,
        .m = m,
        .alpha = alpha,
        .a = &a,
        .lda = @intCast(lda),
        .b = &expected,
        .ldb = @intCast(ldb),
        .first_col = 0,
        .end_col = n,
    });

    const task_count: usize = @min(runtime.maxThreads(), 4);
    var tasks: [4]LeftTask(T) = undefined;
    const base_columns = n / task_count;
    const extra_columns = n % task_count;
    var first_col: usize = 0;
    for (tasks[0..task_count], 0..) |*task, task_index| {
        const column_count = base_columns + @intFromBool(task_index < extra_columns);
        task.* = .{
            .operation = operation,
            .uplo = uplo,
            .trans = trans_,
            .diag = diag,
            .m = m,
            .alpha = alpha,
            .a = &a,
            .lda = @intCast(lda),
            .b = &actual,
            .ldb = @intCast(ldb),
            .first_col = first_col,
            .end_col = first_col + column_count,
        };
        first_col += column_count;
    }
    try std.testing.expect(runParallelLeft(T, tasks[0..task_count]));
    try std.testing.expectEqualSlices(T, &expected, &actual);
}

fn expectPublicLeftGateMatchesSingle(comptime T: type) !void {
    const m = 128;
    const n = 512;
    const lda = m + 1;
    const ldb = m + 2;
    const alpha = triangularTestValue(T, -0.75, 0);
    const allocator = std.testing.allocator;
    const a = try allocator.alloc(T, lda * m);
    defer allocator.free(a);
    const initial_b = try allocator.alloc(T, ldb * n);
    defer allocator.free(initial_b);
    const expected = try allocator.alloc(T, ldb * n);
    defer allocator.free(expected);
    const actual = try allocator.alloc(T, ldb * n);
    defer allocator.free(actual);
    fillTriangularTestOperands(T, a, lda, m, initial_b, ldb, n);

    inline for (.{ LeftOperation.multiply, LeftOperation.solve }) |operation| {
        @memcpy(expected, initial_b);
        @memcpy(actual, initial_b);
        runLeftColumns(T, .{
            .operation = operation,
            .uplo = .lower,
            .trans = .trans,
            .diag = .non_unit,
            .m = m,
            .alpha = alpha,
            .a = a.ptr,
            .lda = @intCast(lda),
            .b = expected.ptr,
            .ldb = @intCast(ldb),
            .first_col = 0,
            .end_col = n,
        });
        if (operation == .multiply) {
            trmm(T, .left, .lower, .trans, .non_unit, m, n, alpha, a.ptr, lda, actual.ptr, ldb);
        } else {
            trsm(T, .left, .lower, .trans, .non_unit, m, n, alpha, a.ptr, lda, actual.ptr, ldb);
        }
        try std.testing.expectEqualSlices(T, expected, actual);
    }
}

test "left TRMM and TRSM parallel tasks preserve every triangular mode" {
    runtime.setMaxThreads(4);
    defer {
        runtime.setMaxThreads(0);
        core_pool.shutdown();
    }
    if (runtime.maxThreads() <= 1) return error.SkipZigTest;

    inline for (.{ f64, scalar.ComplexF64 }) |T| {
        inline for (.{ LeftOperation.multiply, LeftOperation.solve }) |operation| {
            inline for (.{ Uplo.upper, Uplo.lower }) |uplo| {
                inline for (.{ Order.no_trans, Order.trans, Order.conj_trans }) |trans_| {
                    inline for (.{ Diag.non_unit, Diag.unit }) |diag| {
                        try expectParallelLeftMatchesSingle(T, operation, uplo, trans_, diag);
                    }
                }
            }
        }
    }
}

test "left TRMM and TRSM public path crosses conservative parallel gate" {
    runtime.setMaxThreads(4);
    defer {
        runtime.setMaxThreads(0);
        core_pool.shutdown();
    }
    if (runtime.maxThreads() <= 1) return error.SkipZigTest;

    try expectPublicLeftGateMatchesSingle(f32);
}
