// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later

//! Separately compiled x86_64 dense-SYMM and right-triangular candidates.

const std = @import("std");

const abi = @import("x86_64_structured_abi.zig");
const task_runtime = @import("task_runtime_client.zig");
const scalar = @import("../../core/shared/scalar.zig");
const indexing = @import("../../core/shared/indexing.zig");
const structured_tuning = @import("../tuning/structured.zig");

extern var zynum_internal_x86_64_structured_enabled: u8;
extern fn zynum_internal_structured_gemm_execute(request: *abi.GemmRequest) callconv(.c) void;

const max_tasks = 32;
const dense_max_workspace_bytes = 64 * 1024 * 1024;

fn realValue(comptime T: type, raw: u64) T {
    if (T == f32) return @bitCast(@as(u32, @truncate(raw)));
    if (T == f64) return @bitCast(raw);
    @compileError("structured object expects a real component");
}

fn scalarValue(comptime T: type, re: u64, im: u64) T {
    if (T == f32 or T == f64) return realValue(T, re);
    const Real = scalar.Real(T);
    return .{ .re = realValue(Real, re), .im = realValue(Real, im) };
}

fn bits(comptime T: type, value: T) u64 {
    if (T == f32) return @as(u32, @bitCast(value));
    if (T == f64) return @bitCast(value);
    @compileError("bits expects a real scalar component");
}

fn components(comptime T: type, value: T) [2]u64 {
    if (T == f32 or T == f64) return .{ bits(T, value), 0 };
    return .{ bits(@TypeOf(value.re), value.re), bits(@TypeOf(value.im), value.im) };
}

fn inputPtr(comptime T: type, pointer: *const anyopaque) [*]const T {
    return @ptrCast(@alignCast(pointer));
}

fn outputPtr(comptime T: type, pointer: *anyopaque) [*]T {
    return @ptrCast(@alignCast(pointer));
}

fn coreSide(raw: u8) scalar.Side {
    return if (@as(abi.Side, @enumFromInt(raw)) == .left) .left else .right;
}

fn coreUplo(raw: u8) scalar.Uplo {
    return if (@as(abi.Uplo, @enumFromInt(raw)) == .upper) .upper else .lower;
}

fn coreTranspose(raw: u8) scalar.Order {
    return switch (@as(abi.Transpose, @enumFromInt(raw))) {
        .no_trans => .no_trans,
        .trans => .trans,
        .conj_trans => .conj_trans,
    };
}

fn coreDiagonal(raw: u8) scalar.Diag {
    return if (@as(abi.Diagonal, @enumFromInt(raw)) == .unit) .unit else .non_unit;
}

fn scalarTag(comptime T: type) abi.Scalar {
    if (T == f32) return .f32;
    if (T == f64) return .f64;
    if (T == scalar.ComplexF32) return .complex_f32;
    if (T == scalar.ComplexF64) return .complex_f64;
    unreachable;
}

fn callHostGemm(comptime T: type, transa: scalar.Order, transb: scalar.Order, m: i32, n: i32, k: i32, alpha: T, a: [*]const T, lda: i32, b: [*]const T, ldb: i32, beta: T, c: [*]T, ldc: i32) void {
    const alpha_parts = components(T, alpha);
    const beta_parts = components(T, beta);
    var request: abi.GemmRequest = .{
        .transa = @intFromEnum(switch (transa) {
            .no_trans => abi.Transpose.no_trans,
            .trans => abi.Transpose.trans,
            .conj_trans => abi.Transpose.conj_trans,
        }),
        .transb = @intFromEnum(switch (transb) {
            .no_trans => abi.Transpose.no_trans,
            .trans => abi.Transpose.trans,
            .conj_trans => abi.Transpose.conj_trans,
        }),
        .scalar = @intFromEnum(scalarTag(T)),
        .m = m,
        .n = n,
        .k = k,
        .lda = lda,
        .ldb = ldb,
        .ldc = ldc,
        .alpha_re = alpha_parts[0],
        .alpha_im = alpha_parts[1],
        .beta_re = beta_parts[0],
        .beta_im = beta_parts[1],
        .a = @ptrCast(a),
        .b = @ptrCast(b),
        .c = @ptrCast(c),
    };
    zynum_internal_structured_gemm_execute(&request);
}

fn symmetricValue(comptime T: type, uplo: scalar.Uplo, a: [*]const T, lda: i32, row: usize, col: usize, hermitian: bool) T {
    const direct = if (uplo == .upper) row <= col else row >= col;
    var value = if (direct) a[indexing.matIndex(lda, row, col)] else a[indexing.matIndex(lda, col, row)];
    if (hermitian and !direct) value = scalar.conj(T, value);
    if (hermitian and row == col) {
        if (comptime scalar.isComplex(T)) value.im = 0;
    }
    return value;
}

fn scaleOutput(comptime T: type, m: usize, n: usize, beta: T, c: [*]T, ldc: i32) void {
    for (0..n) |j| {
        for (0..m) |i| {
            const index = indexing.matIndex(ldc, i, j);
            c[index] = if (scalar.isZero(T, beta)) scalar.zero(T) else scalar.mul(T, beta, c[index]);
        }
    }
}

fn denseSymm(comptime T: type, request: *abi.Request) bool {
    const m: usize = @intCast(request.m);
    const n: usize = @intCast(request.n);
    const side = coreSide(request.side);
    const uplo = coreUplo(request.uplo);
    const hermitian = request.flags & 1 != 0;
    const alpha = scalarValue(T, request.alpha_re, request.alpha_im);
    const beta = scalarValue(T, request.beta_re, request.beta_im);
    const a = inputPtr(T, request.a);
    const b = inputPtr(T, request.b);
    const c = outputPtr(T, request.c);
    const tuning_side: structured_tuning.Side = if (side == .left) .left else .right;
    if (!structured_tuning.x86_64_object_profile.denseCandidate(structured_tuning.scalarKind(T), tuning_side, hermitian, m, n)) return false;
    if (scalar.isZero(T, alpha)) {
        scaleOutput(T, m, n, beta, c, request.ldc);
        return true;
    }
    const order = if (side == .left) m else n;

    const dense_len = std.math.mul(usize, order, order) catch return false;
    const saved_len = if (comptime scalar.isComplex(T))
        if (scalar.isZero(T, beta)) 0 else std.math.mul(usize, m, n) catch return false
    else
        0;
    const workspace_len = std.math.add(usize, dense_len, saved_len) catch return false;
    const workspace_bytes = std.math.mul(usize, workspace_len, @sizeOf(T)) catch return false;
    if (workspace_bytes > dense_max_workspace_bytes) return false;
    const workspace = std.heap.c_allocator.alloc(T, workspace_len) catch return false;
    defer std.heap.c_allocator.free(workspace);
    const dense_a = workspace[0..dense_len];
    const saved_c = workspace[dense_len..];

    for (0..order) |j| {
        for (0..order) |i| dense_a[i + j * order] = symmetricValue(T, uplo, a, request.lda, i, j, hermitian);
    }
    if (comptime scalar.isComplex(T)) {
        if (saved_c.len != 0) {
            for (0..n) |j| for (0..m) |i| {
                saved_c[i + j * m] = c[indexing.matIndex(request.ldc, i, j)];
            };
        }
        if (side == .left) {
            callHostGemm(T, .no_trans, .no_trans, request.m, request.n, request.m, scalar.one(T), dense_a.ptr, request.m, b, request.ldb, scalar.zero(T), c, request.ldc);
        } else {
            callHostGemm(T, .no_trans, .no_trans, request.m, request.n, request.n, scalar.one(T), b, request.ldb, dense_a.ptr, request.n, scalar.zero(T), c, request.ldc);
        }
        for (0..n) |j| for (0..m) |i| {
            const index = indexing.matIndex(request.ldc, i, j);
            const product = scalar.mul(T, alpha, c[index]);
            c[index] = if (saved_c.len == 0) product else scalar.add(T, product, scalar.mul(T, beta, saved_c[i + j * m]));
        };
    } else if (side == .left) {
        callHostGemm(T, .no_trans, .no_trans, request.m, request.n, request.m, alpha, dense_a.ptr, request.m, b, request.ldb, beta, c, request.ldc);
    } else {
        callHostGemm(T, .no_trans, .no_trans, request.m, request.n, request.n, alpha, b, request.ldb, dense_a.ptr, request.n, beta, c, request.ldc);
    }
    return true;
}

fn RightTask(comptime T: type) type {
    return struct {
        operation: abi.Operation,
        uplo: scalar.Uplo,
        transpose: scalar.Order,
        diagonal: scalar.Diag,
        n: usize,
        alpha: T,
        a: [*]const T,
        lda: i32,
        b: [*]T,
        ldb: i32,
        first_row: usize,
        end_row: usize,
    };
}

fn effectiveUpper(uplo: scalar.Uplo, transpose: scalar.Order) bool {
    return (transpose == .no_trans and uplo == .upper) or (transpose != .no_trans and uplo == .lower);
}

fn triangularValue(comptime T: type, uplo: scalar.Uplo, diagonal: scalar.Diag, transpose: scalar.Order, a: [*]const T, lda: i32, row: usize, column: usize) T {
    const stored_row = if (transpose == .no_trans) row else column;
    const stored_column = if (transpose == .no_trans) column else row;
    if (stored_row == stored_column and diagonal == .unit) return scalar.one(T);
    if (uplo == .upper and stored_row > stored_column) return scalar.zero(T);
    if (uplo == .lower and stored_row < stored_column) return scalar.zero(T);
    const result = a[indexing.matIndex(lda, stored_row, stored_column)];
    return if (transpose == .conj_trans) scalar.conj(T, result) else result;
}

fn solveStrided(comptime T: type, uplo: scalar.Uplo, transpose: scalar.Order, diagonal: scalar.Diag, n: usize, a: [*]const T, lda: i32, x: [*]T, stride: i32) void {
    const upper = effectiveUpper(uplo, transpose);
    if (upper) {
        var row = n;
        while (row > 0) {
            row -= 1;
            const output_index = indexing.vectorIndex(0, row, stride);
            var result = x[output_index];
            for (row + 1..n) |column| {
                result = scalar.sub(T, result, scalar.mul(T, triangularValue(T, uplo, diagonal, transpose, a, lda, row, column), x[indexing.vectorIndex(0, column, stride)]));
            }
            if (diagonal == .non_unit) result = scalar.divv(T, result, triangularValue(T, uplo, diagonal, transpose, a, lda, row, row));
            x[output_index] = result;
        }
        return;
    }
    for (0..n) |row| {
        const output_index = indexing.vectorIndex(0, row, stride);
        var result = x[output_index];
        for (0..row) |column| {
            result = scalar.sub(T, result, scalar.mul(T, triangularValue(T, uplo, diagonal, transpose, a, lda, row, column), x[indexing.vectorIndex(0, column, stride)]));
        }
        if (diagonal == .non_unit) result = scalar.divv(T, result, triangularValue(T, uplo, diagonal, transpose, a, lda, row, row));
        x[output_index] = result;
    }
}

fn conjugateRow(comptime T: type, n: usize, row: usize, b: [*]T, ldb: i32) void {
    for (0..n) |j| {
        const index = indexing.matIndex(ldb, row, j);
        b[index] = scalar.conj(T, b[index]);
    }
}

fn solveRightRow(comptime T: type, task: RightTask(T), row: usize) void {
    const x = task.b + indexing.matIndex(task.ldb, row, 0);
    switch (task.transpose) {
        .no_trans => solveStrided(T, task.uplo, .trans, task.diagonal, task.n, task.a, task.lda, x, task.ldb),
        .trans => solveStrided(T, task.uplo, .no_trans, task.diagonal, task.n, task.a, task.lda, x, task.ldb),
        .conj_trans => {
            if (comptime scalar.isComplex(T)) conjugateRow(T, task.n, row, task.b, task.ldb);
            solveStrided(T, task.uplo, .no_trans, task.diagonal, task.n, task.a, task.lda, x, task.ldb);
            if (comptime scalar.isComplex(T)) conjugateRow(T, task.n, row, task.b, task.ldb);
        },
    }
}

fn runRightRows(comptime T: type, task: RightTask(T)) void {
    if (task.operation == .trsm_right) {
        for (task.first_row..task.end_row) |i| {
            if (!scalar.isOne(T, task.alpha)) {
                for (0..task.n) |j| {
                    const index = indexing.matIndex(task.ldb, i, j);
                    task.b[index] = scalar.mul(T, task.alpha, task.b[index]);
                }
            }
            solveRightRow(T, task, i);
        }
        return;
    }
    for (task.first_row..task.end_row) |i| {
        if (effectiveUpper(task.uplo, task.transpose)) {
            var col = task.n;
            while (col > 0) {
                col -= 1;
                var sum = scalar.zero(T);
                for (0..task.n) |p| sum = scalar.add(T, sum, scalar.mul(T, task.b[indexing.matIndex(task.ldb, i, p)], triangularValue(T, task.uplo, task.diagonal, task.transpose, task.a, task.lda, p, col)));
                task.b[indexing.matIndex(task.ldb, i, col)] = scalar.mul(T, task.alpha, sum);
            }
        } else {
            for (0..task.n) |j| {
                var sum = scalar.zero(T);
                for (0..task.n) |p| sum = scalar.add(T, sum, scalar.mul(T, task.b[indexing.matIndex(task.ldb, i, p)], triangularValue(T, task.uplo, task.diagonal, task.transpose, task.a, task.lda, p, j)));
                task.b[indexing.matIndex(task.ldb, i, j)] = scalar.mul(T, task.alpha, sum);
            }
        }
    }
}

fn runTask(comptime T: type, raw_tasks: *const anyopaque, index: usize) void {
    const tasks: [*]const RightTask(T) = @ptrCast(@alignCast(raw_tasks));
    runRightRows(T, tasks[index]);
}

fn runTaskF32(raw_tasks: *const anyopaque, index: usize) void {
    runTask(f32, raw_tasks, index);
}
fn runTaskF64(raw_tasks: *const anyopaque, index: usize) void {
    runTask(f64, raw_tasks, index);
}
fn runTaskC32(raw_tasks: *const anyopaque, index: usize) void {
    runTask(scalar.ComplexF32, raw_tasks, index);
}
fn runTaskC64(raw_tasks: *const anyopaque, index: usize) void {
    runTask(scalar.ComplexF64, raw_tasks, index);
}

fn rightDispatch(comptime T: type, request: *abi.Request) bool {
    const m: usize = @intCast(request.m);
    const n: usize = @intCast(request.n);
    if (!structured_tuning.x86_64_object_profile.rightTriangularCandidate(m, n)) return false;
    const task_count = @min(task_runtime.taskCount(m, 4), max_tasks);
    if (task_count <= 1) return false;
    var tasks: [task_runtime.max_tasks]RightTask(T) = undefined;
    const base_rows = m / task_count;
    const extra_rows = m % task_count;
    var first_row: usize = 0;
    for (tasks[0..task_count], 0..) |*task, task_index| {
        const rows = base_rows + @intFromBool(task_index < extra_rows);
        task.* = .{
            .operation = @enumFromInt(request.operation),
            .uplo = coreUplo(request.uplo),
            .transpose = coreTranspose(request.transpose),
            .diagonal = coreDiagonal(request.diagonal),
            .n = n,
            .alpha = scalarValue(T, request.alpha_re, request.alpha_im),
            .a = inputPtr(T, request.a),
            .lda = request.lda,
            .b = outputPtr(T, request.c),
            .ldb = request.ldb,
            .first_row = first_row,
            .end_row = first_row + rows,
        };
        first_row += rows;
    }
    const runner = if (T == f32) runTaskF32 else if (T == f64) runTaskF64 else if (T == scalar.ComplexF32) runTaskC32 else runTaskC64;
    return task_runtime.runLowLatency(runner, @ptrCast(tasks[0..task_count].ptr), task_count);
}

fn dispatchScalar(comptime T: type, operation: abi.Operation, request: *abi.Request) bool {
    return switch (operation) {
        .symm_dense => denseSymm(T, request),
        .trmm_right, .trsm_right => rightDispatch(T, request),
    };
}

fn execute(request: *abi.Request) callconv(.c) u8 {
    if (zynum_internal_x86_64_structured_enabled == 0) return 0;
    if (request.m <= 0 or request.n <= 0) return 0;
    const operation: abi.Operation = @enumFromInt(request.operation);
    const scalar_tag: abi.Scalar = @enumFromInt(request.scalar);
    const handled = switch (scalar_tag) {
        .f32 => dispatchScalar(f32, operation, request),
        .f64 => dispatchScalar(f64, operation, request),
        .complex_f32 => dispatchScalar(scalar.ComplexF32, operation, request),
        .complex_f64 => dispatchScalar(scalar.ComplexF64, operation, request),
    };
    return @intFromBool(handled);
}

comptime {
    @export(&execute, .{ .name = "zynum_internal_x86_64_structured_execute", .visibility = .hidden });
}
