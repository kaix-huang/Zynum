// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later

//! Serial CSR execution with validation and no-alias checks before mutation.

const std = @import("std");
const execution = @import("zynum-execution");
const storage = @import("../numerics/storage.zig");
const dense = @import("../numerics/dense.zig");
const views = @import("views.zig");
const scalar = @import("scalar.zig");
const pattern = @import("pattern.zig");
const product = @import("core/product.zig");
const Error = views.Error;

const ByteRange = storage.ByteRange;

fn offsetType(comptime Matrix: type) type {
    return if (@hasDecl(Matrix, "Offset")) Matrix.Offset else Matrix.Index;
}

fn byteRange(comptime T: type, values: []const T) Error!ByteRange {
    const start = @intFromPtr(values.ptr);
    const size = storage.byteLength(values.len, @sizeOf(T)) catch return error.SizeOverflow;
    return storage.byteRange(start, size);
}

fn ensureDisjoint(output: ByteRange, input: ByteRange) Error!void {
    if (storage.overlaps(output, input)) return error.AliasingNotAllowed;
}

fn checkSparseAliasing(comptime T: type, output: ByteRange, input_matrix: anytype) Error!void {
    const I = @TypeOf(input_matrix).Index;
    try ensureDisjoint(output, try byteRange(T, input_matrix.values));
    try ensureDisjoint(output, try byteRange(offsetType(@TypeOf(input_matrix)), input_matrix.row_offsets));
    try ensureDisjoint(output, try byteRange(I, input_matrix.column_indices));
}

const scaleOutput = product.scaleOutput;

fn csrOperand(comptime T: type, input_matrix: anytype) pattern.Operand(T, offsetType(@TypeOf(input_matrix)), @TypeOf(input_matrix).Index) {
    return .{
        .pattern = .{
            .ranges = .{ .contiguous = input_matrix.row_offsets },
            .ordinals = input_matrix.column_indices,
            .row_count = input_matrix.row_count,
            .column_count = input_matrix.column_count,
        },
        .values = input_matrix.values,
        .operation = switch (input_matrix.operation) {
            .normal => .normal,
            .transposed => .transposed,
            .adjoint => .adjoint,
        },
    };
}

fn productColumn(comptime T: type, input_matrix: anytype, input: []const T, output: []T, scale: T, result_scale: T) void {
    const I = @TypeOf(input_matrix).Index;
    product.executeValidated(T, offsetType(@TypeOf(input_matrix)), I, csrOperand(T, input_matrix), input, output, scale, result_scale);
}

/// Compute `result_vector = product_scale * op(matrix) * input_vector +
/// result_scale * result_vector` (CSR SpMV). Scales default to one and zero.
/// Input and result vectors are contiguous slices with exactly the effective
/// column and row counts, respectively. The CSR view carries the transform.
///
/// In all builds, validate the CSR structure, dimensions, and no-alias contract
/// before changing any result. Output must not overlap input or any CSR buffer.
/// Structural checks still run when a scale is zero; when `product_scale == 0`
/// no input numerical values are read, and when `result_scale == 0` no old
/// result values are read. An empty product only scales the result.
///
/// CSR validation costs O(rows + nnz) on every call; the serial numerical
/// execution costs O(rows + nnz + effective output length). No setup cache or
/// allocation is hidden in this checked entry point.
pub fn matrixVectorMultiply(arguments: anytype) Error!void {
    var operation = execution.beginDefault();
    defer operation.end();
    return matrixVectorMultiplyAdmitted(&operation, arguments);
}

/// Internal same-call borrow; the caller owns one live Operation through return.
/// Revalidates all operands on each call; does not begin, end or retain the token.
pub fn matrixVectorMultiplyAdmitted(operation: *const execution.context.Operation, arguments: anytype) Error!void {
    _ = operation.borrowedContext();
    const input_matrix = arguments.matrix;
    const T = @TypeOf(input_matrix).Scalar;
    const input: []const T = arguments.input_vector;
    const output: []T = arguments.result_vector;
    const product_scale: T = views.optionField(arguments, "product_scale", scalar.one(T));
    const result_scale: T = views.optionField(arguments, "result_scale", scalar.zero(T));

    try input_matrix.check();
    if (input.len != input_matrix.effectiveColumnCount() or output.len != input_matrix.effectiveRowCount()) {
        return error.DimensionMismatch;
    }
    const output_range = try byteRange(T, output);
    try ensureDisjoint(output_range, try byteRange(T, input));
    try checkSparseAliasing(T, output_range, input_matrix);

    if (scalar.isZero(T, product_scale) or input_matrix.values.len == 0 or output.len == 0) {
        scaleOutput(T, output, result_scale);
        return;
    }
    productColumn(T, input_matrix, input, output, product_scale, result_scale);
}

/// Compute `result_matrix = product_scale * op(left_matrix) * right_matrix +
/// result_scale * result_matrix` (CSR SpMM). The left input is CSR; the right
/// input and output are column-major dense views. Scales default to one and zero.
/// Padding and the unused tail of dense output storage are preserved.
///
/// All validation precedes mutation, including when the product scale is zero.
/// Output must not overlap either input or any CSR metadata. Dense alias checks
/// include the bounding storage span (inter-column padding included, unused
/// trailing storage excluded). Zero scales have the same unused-value semantics
/// as `matrixVectorMultiply`.
///
/// Validation costs O(rows + nnz) once per call, independently of the number of
/// dense columns. Numerical execution reuses sparse entries across blocks of
/// two dense columns and handles a remaining column serially; construction
/// costs are separate and there is no allocation or worker pool.
pub fn matrixMultiply(arguments: anytype) Error!void {
    var operation = execution.beginDefault();
    defer operation.end();
    return matrixMultiplyAdmitted(&operation, arguments);
}

/// Internal same-call borrow; keeps the caller's single live admission.
pub fn matrixMultiplyAdmitted(operation: *const execution.context.Operation, arguments: anytype) Error!void {
    _ = operation.borrowedContext();
    const left = arguments.left_matrix;
    const right = arguments.right_matrix;
    const output = arguments.result_matrix;
    const T = @TypeOf(left).Scalar;
    const product_scale: T = views.optionField(arguments, "product_scale", scalar.one(T));
    const result_scale: T = views.optionField(arguments, "result_scale", scalar.zero(T));
    // Existing Sparse views and general operators borrow the same native panels.
    const input_panel: dense.ConstPanel(T) = if (@hasField(@TypeOf(right), "shape")) right else .{
        .values = right.values,
        .shape = .{ .row_count = right.row_count, .column_count = right.column_count, .leading_dimension = right.leading_dimension, .layout = .column_major },
    };
    const output_panel: dense.Panel(T) = if (@hasField(@TypeOf(output), "shape")) output else .{
        .values = output.values,
        .shape = .{ .row_count = output.row_count, .column_count = output.column_count, .leading_dimension = output.leading_dimension, .layout = .column_major },
    };
    try left.check();
    _ = try right.check();
    _ = try output.check();
    const right_length = try input_panel.check();
    const output_length = try output_panel.check();
    if (input_panel.shape.row_count != left.effectiveColumnCount() or output_panel.shape.row_count != left.effectiveRowCount() or output_panel.shape.column_count != input_panel.shape.column_count) return error.DimensionMismatch;
    const output_range = try byteRange(T, output_panel.values[0..output_length]);
    try ensureDisjoint(output_range, try byteRange(T, input_panel.values[0..right_length]));
    try checkSparseAliasing(T, output_range, left);
    const I = @TypeOf(left).Index;
    product.executePanelValidated(T, offsetType(@TypeOf(left)), I, csrOperand(T, left), .general, input_panel, output_panel, product_scale, result_scale);
}
