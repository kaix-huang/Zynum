// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// Checked ABI adapters. Declarations come from tools/checked_abi_schema.py.
const std = @import("std");
const execution = @import("zynum-execution");
const abi = @import("abi_types.zig");
pub const Wire = abi;
// Internal interface generators reuse the same pointer/length/status rules.
pub const readBuffer = constSlice;
pub const writeBuffer = mutableSlice;
pub const countElements = count;
pub const statusCode = status;
const lapack = @import("zynum-core").lapack;
const lapack_checked = @import("zynum-core").internal.lapack_checked;
const sparse = @import("zynum-core").sparse;

fn status(err: anyerror) i32 {
    return switch (err) {
        error.InvalidFlag, error.InvalidMatrixOperation => abi.INVALID_FLAG,
        error.InvalidDimension => abi.INVALID_DIMENSION,
        error.InvalidLeadingDimension => abi.INVALID_LEADING_DIMENSION,
        error.BufferTooSmall => abi.BUFFER_TOO_SMALL,
        error.NullPointer => abi.NULL_POINTER,
        error.MisalignedPointer => abi.MISALIGNED_POINTER,
        error.SizeOverflow => abi.SIZE_OVERFLOW,
        error.DimensionMismatch => abi.DIMENSION_MISMATCH,
        error.AliasingNotAllowed => abi.ALIASING_NOT_ALLOWED,
        error.InvalidRowOffsets, error.InvalidColumnIndex => abi.INVALID_SPARSE_STRUCTURE,
        error.NonFiniteInput => abi.NONFINITE_INPUT,
        // Additive Cholesky profile owns this status. Existing real/Sparse
        // checked-v1 operations cannot produce a Hermitian diagonal error.
        error.InvalidHermitianDiagonal => @import("cholesky_status.zig").INVALID_HERMITIAN_DIAGONAL,
        error.InvalidLength => abi.INVALID_LENGTH,
        else => abi.INTERNAL_ERROR,
    };
}

fn result(value: anyerror!usize) abi.result {
    const info = value catch |err| return .{ .status = status(err), .reserved = 0, .info = 0 };
    return .{ .status = if (info == 0) abi.OK else abi.NUMERICAL_FAILURE, .reserved = 0, .info = @intCast(info) };
}

fn triangle(flag: i32) !lapack.Triangle {
    return switch (flag) {
        abi.LOWER => .lower,
        abi.UPPER => .upper,
        else => error.InvalidFlag,
    };
}

fn transform(flag: i32) !sparse.MatrixTransform {
    return switch (flag) {
        abi.NORMAL => .normal,
        abi.TRANSPOSED => .transposed,
        else => error.InvalidFlag,
    };
}

fn count(value: i64) !usize {
    if (value < 0) return error.InvalidLength;
    return std.math.cast(usize, value) orelse error.SizeOverflow;
}

fn dimension(value: i64) !usize {
    if (value < 0) return error.InvalidDimension;
    return std.math.cast(usize, value) orelse error.SizeOverflow;
}

// Validates arithmetic and representation before constructing any Zig slice.
// Actual ownership, allocation extent, readability/writability remain caller obligations.
fn pointer(comptime T: type, data: ?*const anyopaque, length: i64) !usize {
    const len = try count(length);
    const bytes = std.math.mul(usize, len, @sizeOf(T)) catch return error.SizeOverflow;
    if (bytes > std.math.maxInt(isize)) return error.SizeOverflow;
    if (len == 0) return 0;
    const p = data orelse return error.NullPointer;
    const start = @intFromPtr(p);
    if (start % @alignOf(T) != 0) return error.MisalignedPointer;
    _ = std.math.add(usize, start, bytes) catch return error.SizeOverflow;
    return len;
}

fn constSlice(comptime T: type, b: abi.const_buffer) ![]const T {
    const len = try pointer(T, b.data, b.length);
    if (len == 0) return &.{};
    const p: [*]const T = @ptrCast(@alignCast(b.data.?));
    return p[0..len];
}

fn mutableSlice(comptime T: type, b: abi.buffer) ![]T {
    const len = try pointer(T, b.data, b.length);
    if (len == 0) return &.{};
    const p: [*]T = @ptrCast(@alignCast(b.data.?));
    return p[0..len];
}

fn densePreflight(a: anytype) !void {
    const rows = try dimension(a.rows);
    const cols = try dimension(a.cols);
    if (rows > std.math.maxInt(lapack.PivotInt) or cols > std.math.maxInt(lapack.PivotInt)) return error.InvalidDimension;
    if (a.ld < 0 or a.ld > std.math.maxInt(lapack.PivotInt)) return error.InvalidLeadingDimension;
    const len = try count(a.storage.length);
    if (rows == 0 or cols == 0) return;
    const ld: usize = @intCast(a.ld);
    if (ld < rows) return error.InvalidLeadingDimension;
    const preceding = std.math.mul(usize, cols - 1, ld) catch return error.SizeOverflow;
    const span = std.math.add(usize, preceding, rows) catch return error.SizeOverflow;
    if (len < span) return error.BufferTooSmall;
}

fn matrix(comptime T: type, a: abi.dense) !lapack.Matrix(T) {
    try densePreflight(a);
    return .{ .values = try mutableSlice(T, a.storage), .row_count = @intCast(a.rows), .column_count = @intCast(a.cols), .leading_dimension = @intCast(a.ld) };
}

fn constMatrix(comptime T: type, a: abi.const_dense) !lapack.ConstMatrix(T) {
    try densePreflight(a);
    return .{ .values = try constSlice(T, a.storage), .row_count = @intCast(a.rows), .column_count = @intCast(a.cols), .leading_dimension = @intCast(a.ld) };
}

fn factorInner(comptime T: type, operation: *const execution.context.Operation, a: abi.dense, flag: i32) !usize {
    const uplo = try triangle(flag);
    return lapack_checked.choleskyFactorizeAdmitted(operation, T, try matrix(T, a), uplo);
}

pub fn factor(comptime T: type, a: abi.dense, flag: i32) abi.result {
    var operation = execution.beginDefault();
    defer operation.end();
    return result(factorInner(T, &operation, a, flag));
}

fn solveInner(comptime T: type, operation: *const execution.context.Operation, a: abi.const_dense, b: abi.dense, flag: i32) !usize {
    const uplo = try triangle(flag);
    // Both descriptor ranges are checked before public operations may read memory.
    const factor_matrix = try constMatrix(T, a);
    const rhs = try matrix(T, b);
    return lapack_checked.choleskySolveAdmitted(operation, T, factor_matrix, rhs, uplo);
}

pub fn solve(comptime T: type, a: abi.const_dense, b: abi.dense, flag: i32) abi.result {
    var operation = execution.beginDefault();
    defer operation.end();
    return result(solveInner(T, &operation, a, b, flag));
}

fn spmvInner(operation: *const execution.context.Operation, comptime T: type, comptime O: type, comptime I: type, comptime extended_transform: bool, a: abi.csr, flag: i32, alpha: T, x: abi.const_buffer, beta: T, y: abi.buffer) !usize {
    const op = if (extended_transform) try sparseTransform(flag) else try transform(flag);
    const rows = try dimension(a.rows);
    const cols = try dimension(a.cols);
    const offsets_len = std.math.add(usize, rows, 1) catch return error.SizeOverflow;
    if (try count(a.row_offsets.length) != offsets_len or try count(a.column_indices.length) != try count(a.values.length)) return error.InvalidLength;
    const nx = if (op == .normal) cols else rows;
    const ny = if (op == .normal) rows else cols;
    if (try count(x.length) != nx or try count(y.length) != ny) return error.DimensionMismatch;
    // Every caller pointer is checked before Sparse.check reads CSR structure.
    const offsets = try constSlice(O, a.row_offsets);
    const indices = try constSlice(I, a.column_indices);
    const values = try constSlice(T, a.values);
    const input = try constSlice(T, x);
    const output = try mutableSlice(T, y);
    const csr: sparse.ConstCsrMatrixWithIndexTypes(T, O, I) = .{ .row_offsets = offsets, .column_indices = indices, .values = values, .row_count = rows, .column_count = cols, .operation = op };
    // Preserve CSR preflight precedence before the generic operator checks.
    // Its fused backend revalidates borrowed metadata and all sparse overlap.
    try csr.check();
    const action = @import("zynum-core").internal.sparse_checked.operator.CsrBackend(@TypeOf(csr)).borrow(&csr);
    try action.applyScaledAdmitted(operation, alpha, input, beta, output, &.{});
    return 0;
}

pub fn spmv(comptime T: type, comptime I: type, a: abi.csr, flag: i32, alpha: T, x: abi.const_buffer, beta: T, y: abi.buffer) abi.result {
    var operation = execution.beginDefault();
    defer operation.end();
    return result(spmvInner(&operation, T, I, I, false, a, flag, alpha, x, beta, y));
}

// Extended Sparse adapters preserve the existing checked-v1 flag contract.
// The additive wire profile will name adjoint explicitly; no exports yet.
fn sparseTransform(flag: i32) !sparse.MatrixTransform {
    return if (flag == 2) .adjoint else transform(flag);
}

/// Callback adapters borrow the outer algorithm admission. No nested default
/// Context operation is opened; the existing wire preflight/status stays intact.
pub fn spmvWithIndexTypesAdmitted(operation: *const execution.context.Operation, comptime T: type, comptime O: type, comptime I: type, a: abi.csr, flag: i32, alpha: T, x: abi.const_buffer, beta: T, y: abi.buffer) abi.result {
    _ = operation.borrowedContext();
    return result(spmvInner(operation, T, O, I, true, a, flag, alpha, x, beta, y));
}
pub fn spmvWithIndexTypes(comptime T: type, comptime O: type, comptime I: type, a: abi.csr, flag: i32, alpha: T, x: abi.const_buffer, beta: T, y: abi.buffer) abi.result {
    var operation = execution.beginDefault();
    defer operation.end();
    return result(spmvInner(&operation, T, O, I, true, a, flag, alpha, x, beta, y));
}

fn spmmInner(operation: *const execution.context.Operation, comptime T: type, comptime O: type, comptime I: type, a: abi.csr, flag: i32, alpha: T, b: abi.const_dense, beta: T, c: abi.dense) !usize {
    const op = try sparseTransform(flag);
    const rows = try dimension(a.rows);
    const cols = try dimension(a.cols);
    const offsets_len = std.math.add(usize, rows, 1) catch return error.SizeOverflow;
    if (try count(a.row_offsets.length) != offsets_len or try count(a.column_indices.length) != try count(a.values.length)) return error.InvalidLength;
    try densePreflight(b);
    try densePreflight(c);
    if (try dimension(b.rows) != (if (op == .normal) cols else rows) or try dimension(c.rows) != (if (op == .normal) rows else cols) or b.cols != c.cols) return error.DimensionMismatch;
    const offsets = try constSlice(O, a.row_offsets);
    const indices = try constSlice(I, a.column_indices);
    const values = try constSlice(T, a.values);
    const input = try constSlice(T, b.storage);
    const output = try mutableSlice(T, c.storage);
    const csr: sparse.ConstCsrMatrixWithIndexTypes(T, O, I) = .{ .row_offsets = offsets, .column_indices = indices, .values = values, .row_count = rows, .column_count = cols, .operation = op };
    const right = try sparse.constMatrix(T, input, .{ .row_count = @as(usize, @intCast(b.rows)), .column_count = @as(usize, @intCast(b.cols)), .leading_dimension = @as(usize, @intCast(b.ld)) });
    const output_matrix = try sparse.matrix(T, output, .{ .row_count = @as(usize, @intCast(c.rows)), .column_count = @as(usize, @intCast(c.cols)), .leading_dimension = @as(usize, @intCast(c.ld)) });
    try csr.check();
    const action = @import("zynum-core").internal.sparse_checked.operator.CsrBackend(@TypeOf(csr)).borrow(&csr);
    try action.applyPanelAdmitted(operation, alpha, .{ .values = right.values, .shape = .{ .row_count = right.row_count, .column_count = right.column_count, .leading_dimension = right.leading_dimension, .layout = .column_major } }, beta, .{ .values = output_matrix.values, .shape = .{ .row_count = output_matrix.row_count, .column_count = output_matrix.column_count, .leading_dimension = output_matrix.leading_dimension, .layout = .column_major } }, &.{});
    return 0;
}

pub fn spmmWithIndexTypesAdmitted(operation: *const execution.context.Operation, comptime T: type, comptime O: type, comptime I: type, a: abi.csr, flag: i32, alpha: T, b: abi.const_dense, beta: T, c: abi.dense) abi.result {
    _ = operation.borrowedContext();
    return result(spmmInner(operation, T, O, I, a, flag, alpha, b, beta, c));
}
pub fn spmmWithIndexTypes(comptime T: type, comptime O: type, comptime I: type, a: abi.csr, flag: i32, alpha: T, b: abi.const_dense, beta: T, c: abi.dense) abi.result {
    var operation = execution.beginDefault();
    defer operation.end();
    return result(spmmInner(&operation, T, O, I, a, flag, alpha, b, beta, c));
}
