// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later

//! User-facing checked BLAS operations for the Zig API.
//!
//! These functions translate checked public views into unchecked core operands,
//! enforce cheap shape contracts in every build and aliasing contracts in checked
//! builds, and keep BLAS ABI spellings out of the descriptive Zig facade. Default
//! output operations use a no-alias contract; `WithWorkspace` variants provide
//! explicit temporary storage for caller-approved aliasing.

const aliasing = @import("aliasing.zig");
const core = @import("../core.zig");
const std = @import("std");
const views = @import("views.zig");

/// Errors returned by checked public Zig BLAS operations.
pub const Error = views.Error;
/// Backwards-compatible alias for the public BLAS API error set.
pub const BlasError = views.BlasError;

fn assertSameScalar(comptime Expected: type, value: anytype) void {
    const Actual = @TypeOf(value).Scalar;
    if (Actual != Expected) @compileError("all BLAS views in one operation must use the same scalar type");
}

fn checkedPairLength(first_length: views.BlasInt, second_length: views.BlasInt) Error!views.BlasInt {
    if (first_length < 0 or second_length < 0) return error.InvalidLength;
    return @min(first_length, second_length);
}

fn requireEqualLength(first_length: views.BlasInt, second_length: views.BlasInt) Error!void {
    if (first_length < 0 or second_length < 0) return error.InvalidLength;
    if (first_length != second_length) return error.DimensionMismatch;
}

fn checkedNonNegativeLength(length: views.BlasInt) Error!usize {
    if (length < 0) return error.InvalidLength;
    return @intCast(length);
}

fn constVectorOperand(comptime T: type, vector: anytype) core.ConstVector(T) {
    return .{ .values = vector.values.ptr, .length = vector.length, .stride = vector.stride };
}

fn vectorOperand(comptime T: type, vector: anytype) core.Vector(T) {
    return .{ .values = vector.values.ptr, .length = vector.length, .stride = vector.stride };
}

fn constMatrixOperand(comptime T: type, matrix: anytype) core.ConstMatrix(T) {
    return .{
        .values = matrix.values.ptr,
        .row_count = matrix.row_count,
        .column_count = matrix.column_count,
        .leading_dimension = matrix.leading_dimension,
        .transpose = views.toCoreTranspose(matrix.operation),
    };
}

fn matrixOperand(comptime T: type, matrix: anytype) core.Matrix(T) {
    return .{
        .values = matrix.values.ptr,
        .row_count = matrix.row_count,
        .column_count = matrix.column_count,
        .leading_dimension = matrix.leading_dimension,
    };
}

fn workspaceVector(comptime T: type, values: []T, length: views.BlasInt) core.Vector(T) {
    return .{ .values = values.ptr, .length = length, .stride = 1 };
}

fn workspaceMatrix(comptime T: type, values: []T, row_count: views.BlasInt, column_count: views.BlasInt) core.Matrix(T) {
    return .{
        .values = values.ptr,
        .row_count = row_count,
        .column_count = column_count,
        .leading_dimension = row_count,
    };
}

fn ensureWorkspaceLength(actual: usize, required: usize) Error!void {
    if (actual < required) return error.WorkspaceTooSmall;
}

fn ensureVectorIntoAliases(comptime T: type, input_vector: anytype, result_vector: anytype) Error!void {
    if (!views.runtime_checks_enabled) return;
    if (aliasing.vectorsExactlyMatch(input_vector, result_vector)) return;
    try aliasing.ensureNoVectorOverlap(T, input_vector, result_vector);
}

fn copyVectorIntoResult(comptime T: type, input_vector: anytype, result_vector: anytype) void {
    core.copyVectorView(T, .{
        .source = constVectorOperand(T, input_vector),
        .destination = .{ .values = result_vector.values.ptr, .length = input_vector.length, .stride = result_vector.stride },
    });
}

fn copyMatrixToWorkspace(comptime T: type, source_matrix: anytype, workspace: []T) void {
    const rows: usize = @intCast(source_matrix.row_count);
    const cols: usize = @intCast(source_matrix.column_count);
    for (0..cols) |col| {
        for (0..rows) |row| {
            workspace[row + col * rows] = source_matrix.values[@as(usize, @intCast(source_matrix.leading_dimension)) * col + row];
        }
    }
}

fn copyWorkspaceToMatrix(comptime T: type, workspace: []const T, result_matrix: anytype) void {
    const rows: usize = @intCast(result_matrix.row_count);
    const cols: usize = @intCast(result_matrix.column_count);
    for (0..cols) |col| {
        for (0..rows) |row| {
            result_matrix.values[@as(usize, @intCast(result_matrix.leading_dimension)) * col + row] = workspace[row + col * rows];
        }
    }
}

fn matrixVectorExpectedResultLength(matrix: anytype) Error!views.BlasInt {
    const length = matrix.effectiveRowCount();
    if (length < 0) return error.InvalidLength;
    return length;
}

fn matrixProductElementCount(result_matrix: anytype) Error!usize {
    const rows = try checkedNonNegativeLength(result_matrix.row_count);
    const cols = try checkedNonNegativeLength(result_matrix.column_count);
    return std.math.mul(usize, rows, cols) catch error.InvalidLength;
}

fn checkMatrixVectorShapes(input_matrix: anytype, input_vector: anytype, result_vector: anytype) Error!views.BlasInt {
    const expected_input_length = input_matrix.effectiveColumnCount();
    const expected_result_length = input_matrix.effectiveRowCount();
    if (expected_input_length < 0 or expected_result_length < 0) return error.InvalidLength;
    if (input_vector.length < expected_input_length or result_vector.length < expected_result_length) return error.DimensionMismatch;
    return expected_result_length;
}

fn checkMatrixProductShapes(left_matrix: anytype, right_matrix: anytype, result_matrix: anytype) Error!void {
    if (left_matrix.effectiveColumnCount() != right_matrix.effectiveRowCount()) return error.DimensionMismatch;
    if (result_matrix.row_count != left_matrix.effectiveRowCount()) return error.DimensionMismatch;
    if (result_matrix.column_count != right_matrix.effectiveColumnCount()) return error.DimensionMismatch;
}

fn executeScaleVectorIntoNoAlias(comptime T: type, input_vector: anytype, result_vector: anytype, scale: T) void {
    copyVectorIntoResult(T, input_vector, result_vector);
    core.scaleVectorView(T, .{ .vector = vectorOperand(T, result_vector), .scale = scale });
}

fn executeAddScaledVectorIntoNoAlias(comptime T: type, source_vector: anytype, input_vector: anytype, result_vector: anytype, scale: T) void {
    copyVectorIntoResult(T, input_vector, result_vector);
    core.addScaledVectorView(T, .{
        .source = constVectorOperand(T, source_vector),
        .destination = vectorOperand(T, result_vector),
        .scale = scale,
    });
}

fn executeCombineVectorsIntoNoAlias(comptime T: type, source_vector: anytype, input_vector: anytype, result_vector: anytype, source_scale: T, input_scale: T) void {
    copyVectorIntoResult(T, input_vector, result_vector);
    core.combineVectorViews(T, .{
        .source = constVectorOperand(T, source_vector),
        .destination = vectorOperand(T, result_vector),
        .source_scale = source_scale,
        .destination_scale = input_scale,
    });
}

fn executeMatrixVectorNoAlias(comptime T: type, input_matrix: anytype, input_vector: anytype, result_vector: anytype, product_scale: T, result_scale: T) void {
    core.multiplyMatrixVector(T, .{
        .matrix = constMatrixOperand(T, input_matrix),
        .input = constVectorOperand(T, input_vector),
        .output = vectorOperand(T, result_vector),
        .product_scale = product_scale,
        .output_scale = result_scale,
    });
}

fn executeMatrixVectorWithWorkspace(comptime T: type, input_matrix: anytype, input_vector: anytype, result_vector: anytype, workspace: []T, result_length: views.BlasInt, product_scale: T, result_scale: T) void {
    const temp = workspaceVector(T, workspace, result_length);
    if (!core.isZero(T, result_scale)) {
        core.copyVectorView(T, .{
            .source = constVectorOperand(T, result_vector),
            .destination = temp,
        });
    }
    core.multiplyMatrixVector(T, .{
        .matrix = constMatrixOperand(T, input_matrix),
        .input = constVectorOperand(T, input_vector),
        .output = temp,
        .product_scale = product_scale,
        .output_scale = result_scale,
    });
    core.copyVectorView(T, .{
        .source = temp.asConst(),
        .destination = .{ .values = result_vector.values.ptr, .length = result_length, .stride = result_vector.stride },
    });
}

fn executeMatrixProductNoAlias(comptime T: type, left_matrix: anytype, right_matrix: anytype, result_matrix: anytype, product_scale: T, result_scale: T) void {
    core.multiplyMatrices(T, .{
        .left = constMatrixOperand(T, left_matrix),
        .right = constMatrixOperand(T, right_matrix),
        .output = matrixOperand(T, result_matrix),
        .product_scale = product_scale,
        .output_scale = result_scale,
    });
}

fn executeMatrixProductWithWorkspace(comptime T: type, left_matrix: anytype, right_matrix: anytype, result_matrix: anytype, workspace: []T, product_scale: T, result_scale: T) void {
    if (!core.isZero(T, result_scale)) copyMatrixToWorkspace(T, result_matrix, workspace);
    const temp = workspaceMatrix(T, workspace, result_matrix.row_count, result_matrix.column_count);
    core.multiplyMatrices(T, .{
        .left = constMatrixOperand(T, left_matrix),
        .right = constMatrixOperand(T, right_matrix),
        .output = temp,
        .product_scale = product_scale,
        .output_scale = result_scale,
    });
    copyWorkspaceToMatrix(T, workspace, result_matrix);
}

/// Return the workspace element count required by `matrixVectorMultiplyWithWorkspace`.
pub fn matrixVectorMultiplyWorkspaceLength(arguments: anytype) Error!usize {
    return checkedNonNegativeLength(try matrixVectorExpectedResultLength(arguments.matrix));
}

/// Return the workspace element count required by `matrixMultiplyWithWorkspace`.
pub fn matrixMultiplyWorkspaceLength(arguments: anytype) Error!usize {
    return matrixProductElementCount(arguments.result_matrix);
}

/// Swap elements between two mutable vector views over their shared length.
///
/// Exact self-swaps are allowed; partial overlap is rejected in checked builds.
pub fn swapVectors(arguments: anytype) Error!void {
    const T = @TypeOf(arguments.first_vector).Scalar;
    assertSameScalar(T, arguments.second_vector);
    var first_vector = arguments.first_vector;
    var second_vector = arguments.second_vector;
    try first_vector.check();
    try second_vector.check();
    try aliasing.ensureNoPartialVectorOverlap(T, first_vector, second_vector);
    const shared_length = try checkedPairLength(first_vector.length, second_vector.length);
    core.swapVectorViews(T, .{
        .first = .{ .values = first_vector.values.ptr, .length = shared_length, .stride = first_vector.stride },
        .second = .{ .values = second_vector.values.ptr, .length = shared_length, .stride = second_vector.stride },
    });
}

/// Copy from `source_vector` to `destination_vector` over their shared length.
///
/// Exact self-copies are no-ops; partial overlap is rejected in checked builds.
pub fn copyVector(arguments: anytype) Error!void {
    const T = @TypeOf(arguments.destination_vector).Scalar;
    assertSameScalar(T, arguments.source_vector);
    const source_vector = arguments.source_vector;
    var destination_vector = arguments.destination_vector;
    try source_vector.check();
    try destination_vector.check();
    if (aliasing.vectorsExactlyMatch(source_vector, destination_vector)) return;
    try aliasing.ensureNoPartialVectorOverlap(T, source_vector, destination_vector);
    const shared_length = try checkedPairLength(source_vector.length, destination_vector.length);
    core.copyVectorView(T, .{
        .source = .{ .values = source_vector.values.ptr, .length = shared_length, .stride = source_vector.stride },
        .destination = .{ .values = destination_vector.values.ptr, .length = shared_length, .stride = destination_vector.stride },
    });
}

/// Scale `target_vector` in place by `scale`.
pub fn scaleVector(arguments: anytype) Error!void {
    const T = @TypeOf(arguments.target_vector).Scalar;
    var target_vector = arguments.target_vector;
    try target_vector.check();
    core.scaleVectorView(T, .{ .vector = vectorOperand(T, target_vector), .scale = arguments.scale });
}

/// Write `result_vector = scale * input_vector`.
///
/// `input_vector` and `result_vector` must have equal lengths. Exact in-place
/// use is allowed; other overlap is rejected in checked builds.
pub fn scaleVectorInto(arguments: anytype) Error!void {
    const T = @TypeOf(arguments.result_vector).Scalar;
    assertSameScalar(T, arguments.input_vector);
    const input_vector = arguments.input_vector;
    var result_vector = arguments.result_vector;
    try input_vector.check();
    try result_vector.check();
    try requireEqualLength(input_vector.length, result_vector.length);
    if (aliasing.vectorsExactlyMatch(input_vector, result_vector)) {
        try scaleVector(.{ .target_vector = result_vector, .scale = arguments.scale });
        return;
    }
    try ensureVectorIntoAliases(T, input_vector, result_vector);
    executeScaleVectorIntoNoAlias(T, input_vector, result_vector, arguments.scale);
}

/// Accumulate `destination_vector += scale * source_vector` over the shared
/// length.
///
/// Exact source/destination aliasing is allowed; partial overlap is rejected in
/// checked builds.
pub fn addScaledVector(arguments: anytype) Error!void {
    const T = @TypeOf(arguments.destination_vector).Scalar;
    assertSameScalar(T, arguments.source_vector);
    const source_vector = arguments.source_vector;
    var destination_vector = arguments.destination_vector;
    try source_vector.check();
    try destination_vector.check();
    try aliasing.ensureNoPartialVectorOverlap(T, source_vector, destination_vector);
    const shared_length = try checkedPairLength(source_vector.length, destination_vector.length);
    core.addScaledVectorView(T, .{
        .source = .{ .values = source_vector.values.ptr, .length = shared_length, .stride = source_vector.stride },
        .destination = .{ .values = destination_vector.values.ptr, .length = shared_length, .stride = destination_vector.stride },
        .scale = arguments.scale,
    });
}

/// Write `result_vector = input_vector + scale * source_vector`.
///
/// All three vectors must have equal lengths. `result_vector` may exactly match
/// `input_vector`; other result/input or result/source overlap is rejected in
/// checked builds.
pub fn addScaledVectorInto(arguments: anytype) Error!void {
    const T = @TypeOf(arguments.result_vector).Scalar;
    assertSameScalar(T, arguments.source_vector);
    assertSameScalar(T, arguments.input_vector);
    const source_vector = arguments.source_vector;
    const input_vector = arguments.input_vector;
    var result_vector = arguments.result_vector;
    try source_vector.check();
    try input_vector.check();
    try result_vector.check();
    try requireEqualLength(source_vector.length, input_vector.length);
    try requireEqualLength(input_vector.length, result_vector.length);
    if (aliasing.vectorsExactlyMatch(input_vector, result_vector)) {
        try addScaledVector(.{ .source_vector = source_vector, .destination_vector = result_vector, .scale = arguments.scale });
        return;
    }
    try aliasing.ensureNoVectorOverlap(T, result_vector, input_vector);
    try aliasing.ensureNoVectorOverlap(T, result_vector, source_vector);
    executeAddScaledVectorIntoNoAlias(T, source_vector, input_vector, result_vector, arguments.scale);
}

/// Accumulate `destination = source_scale * source + destination_scale * destination`.
///
/// The operation runs over the source/destination shared length. Exact aliasing
/// is allowed; partial overlap is rejected in checked builds.
pub fn combineVectors(arguments: anytype) Error!void {
    const T = @TypeOf(arguments.destination_vector).Scalar;
    assertSameScalar(T, arguments.source_vector);
    const source_vector = arguments.source_vector;
    var destination_vector = arguments.destination_vector;
    try source_vector.check();
    try destination_vector.check();
    try aliasing.ensureNoPartialVectorOverlap(T, source_vector, destination_vector);
    const shared_length = try checkedPairLength(source_vector.length, destination_vector.length);
    core.combineVectorViews(T, .{
        .source = .{ .values = source_vector.values.ptr, .length = shared_length, .stride = source_vector.stride },
        .destination = .{ .values = destination_vector.values.ptr, .length = shared_length, .stride = destination_vector.stride },
        .source_scale = arguments.source_scale,
        .destination_scale = arguments.destination_scale,
    });
}

/// Write `result = source_scale * source + input_scale * input`.
///
/// All three vectors must have equal lengths. `result_vector` may exactly match
/// `input_vector`; other result/input or result/source overlap is rejected in
/// checked builds.
pub fn combineVectorsInto(arguments: anytype) Error!void {
    const T = @TypeOf(arguments.result_vector).Scalar;
    assertSameScalar(T, arguments.source_vector);
    assertSameScalar(T, arguments.input_vector);
    const source_vector = arguments.source_vector;
    const input_vector = arguments.input_vector;
    var result_vector = arguments.result_vector;
    try source_vector.check();
    try input_vector.check();
    try result_vector.check();
    try requireEqualLength(source_vector.length, input_vector.length);
    try requireEqualLength(input_vector.length, result_vector.length);
    if (aliasing.vectorsExactlyMatch(input_vector, result_vector)) {
        try combineVectors(.{
            .source_vector = source_vector,
            .destination_vector = result_vector,
            .source_scale = arguments.source_scale,
            .destination_scale = arguments.input_scale,
        });
        return;
    }
    try aliasing.ensureNoVectorOverlap(T, result_vector, input_vector);
    try aliasing.ensureNoVectorOverlap(T, result_vector, source_vector);
    executeCombineVectorsIntoNoAlias(T, source_vector, input_vector, result_vector, arguments.source_scale, arguments.input_scale);
}

/// Return the unconjugated dot product over the left/right shared length.
pub fn dotProduct(arguments: anytype) Error!@TypeOf(arguments.left_vector).Scalar {
    const T = @TypeOf(arguments.left_vector).Scalar;
    assertSameScalar(T, arguments.right_vector);
    const left_vector = arguments.left_vector;
    const right_vector = arguments.right_vector;
    try left_vector.check();
    try right_vector.check();
    const shared_length = try checkedPairLength(left_vector.length, right_vector.length);
    return core.dotProductView(T, .{
        .left = .{ .values = left_vector.values.ptr, .length = shared_length, .stride = left_vector.stride },
        .right = .{ .values = right_vector.values.ptr, .length = shared_length, .stride = right_vector.stride },
    });
}

/// Return the dot product with the left vector conjugated for complex scalars.
pub fn conjugatedDotProduct(arguments: anytype) Error!@TypeOf(arguments.left_vector).Scalar {
    const T = @TypeOf(arguments.left_vector).Scalar;
    assertSameScalar(T, arguments.right_vector);
    const left_vector = arguments.left_vector;
    const right_vector = arguments.right_vector;
    try left_vector.check();
    try right_vector.check();
    const shared_length = try checkedPairLength(left_vector.length, right_vector.length);
    return core.dotProductView(T, .{
        .left = .{ .values = left_vector.values.ptr, .length = shared_length, .stride = left_vector.stride },
        .right = .{ .values = right_vector.values.ptr, .length = shared_length, .stride = right_vector.stride },
        .conjugate_left = true,
    });
}

/// Return the Euclidean norm of `input_vector`.
pub fn euclideanNorm(arguments: anytype) Error!core.Real(@TypeOf(arguments.input_vector).Scalar) {
    const T = @TypeOf(arguments.input_vector).Scalar;
    const input_vector = arguments.input_vector;
    try input_vector.check();
    return core.euclideanNormView(T, constVectorOperand(T, input_vector));
}

/// Compute `result = product_scale * op(matrix) * input + result_scale * result`.
///
/// `product_scale` defaults to one and `result_scale` defaults to zero. The
/// result vector must not overlap the input vector or matrix in checked builds.
pub fn matrixVectorMultiply(arguments: anytype) Error!void {
    const T = @TypeOf(arguments.result_vector).Scalar;
    assertSameScalar(T, arguments.matrix);
    assertSameScalar(T, arguments.input_vector);

    const input_matrix = arguments.matrix;
    const input_vector = arguments.input_vector;
    var result_vector = arguments.result_vector;
    try input_matrix.check();
    try input_vector.check();
    try result_vector.check();

    _ = try checkMatrixVectorShapes(input_matrix, input_vector, result_vector);
    if (views.runtime_checks_enabled) {
        try aliasing.ensureNoVectorOverlap(T, result_vector, input_vector);
        try aliasing.ensureNoVectorMatrixOverlap(T, result_vector, input_matrix);
    }

    const product_scale: T = views.optionField(arguments, "product_scale", core.one(T));
    const result_scale: T = views.optionField(arguments, "result_scale", core.zero(T));
    executeMatrixVectorNoAlias(T, input_matrix, input_vector, result_vector, product_scale, result_scale);
}

/// Workspace-backed matrix-vector multiply for caller-approved result/input or
/// result/matrix aliasing.
///
/// `workspace` must contain at least `matrixVectorMultiplyWorkspaceLength`
/// elements and must not overlap the matrix, input, or result in checked builds.
pub fn matrixVectorMultiplyWithWorkspace(arguments: anytype) Error!void {
    const T = @TypeOf(arguments.result_vector).Scalar;
    assertSameScalar(T, arguments.matrix);
    assertSameScalar(T, arguments.input_vector);

    const input_matrix = arguments.matrix;
    const input_vector = arguments.input_vector;
    var result_vector = arguments.result_vector;
    const workspace = arguments.workspace;
    try input_matrix.check();
    try input_vector.check();
    try result_vector.check();

    const result_length = try checkMatrixVectorShapes(input_matrix, input_vector, result_vector);
    const required_workspace = try checkedNonNegativeLength(result_length);
    try ensureWorkspaceLength(workspace.len, required_workspace);

    if (views.runtime_checks_enabled) {
        const workspace_view = try views.vector(T, workspace, .{ .length = result_length });
        try aliasing.ensureNoVectorOverlap(T, workspace_view, input_vector);
        try aliasing.ensureNoVectorOverlap(T, workspace_view, result_vector);
        try aliasing.ensureNoVectorMatrixOverlap(T, workspace_view, input_matrix);
    }

    const product_scale: T = views.optionField(arguments, "product_scale", core.one(T));
    const result_scale: T = views.optionField(arguments, "result_scale", core.zero(T));
    executeMatrixVectorWithWorkspace(T, input_matrix, input_vector, result_vector, workspace[0..required_workspace], result_length, product_scale, result_scale);
}

/// Compute `result = product_scale * op(left) * op(right) + result_scale * result`.
///
/// `product_scale` defaults to one and `result_scale` defaults to zero. The
/// result matrix must not overlap either input matrix in checked builds.
pub fn matrixMultiply(arguments: anytype) Error!void {
    const T = @TypeOf(arguments.result_matrix).Scalar;
    assertSameScalar(T, arguments.left_matrix);
    assertSameScalar(T, arguments.right_matrix);

    const left_matrix = arguments.left_matrix;
    const right_matrix = arguments.right_matrix;
    var result_matrix = arguments.result_matrix;
    try left_matrix.check();
    try right_matrix.check();
    try result_matrix.check();

    try checkMatrixProductShapes(left_matrix, right_matrix, result_matrix);
    if (views.runtime_checks_enabled) {
        try aliasing.ensureNoMatrixOverlap(T, result_matrix, left_matrix);
        try aliasing.ensureNoMatrixOverlap(T, result_matrix, right_matrix);
    }

    const product_scale: T = views.optionField(arguments, "product_scale", core.one(T));
    const result_scale: T = views.optionField(arguments, "result_scale", core.zero(T));
    executeMatrixProductNoAlias(T, left_matrix, right_matrix, result_matrix, product_scale, result_scale);
}

/// Workspace-backed matrix multiply for caller-approved result/input-matrix
/// aliasing.
///
/// `workspace` must contain at least `matrixMultiplyWorkspaceLength` elements
/// and must not overlap either input matrix or the result in checked builds.
pub fn matrixMultiplyWithWorkspace(arguments: anytype) Error!void {
    const T = @TypeOf(arguments.result_matrix).Scalar;
    assertSameScalar(T, arguments.left_matrix);
    assertSameScalar(T, arguments.right_matrix);

    const left_matrix = arguments.left_matrix;
    const right_matrix = arguments.right_matrix;
    var result_matrix = arguments.result_matrix;
    const workspace = arguments.workspace;
    try left_matrix.check();
    try right_matrix.check();
    try result_matrix.check();
    try checkMatrixProductShapes(left_matrix, right_matrix, result_matrix);

    const required_workspace = try matrixProductElementCount(result_matrix);
    try ensureWorkspaceLength(workspace.len, required_workspace);
    if (views.runtime_checks_enabled) {
        const workspace_view = try views.matrix(T, workspace[0..required_workspace], .{
            .row_count = result_matrix.row_count,
            .column_count = result_matrix.column_count,
            .leading_dimension = result_matrix.row_count,
        });
        try aliasing.ensureNoMatrixOverlap(T, workspace_view, left_matrix);
        try aliasing.ensureNoMatrixOverlap(T, workspace_view, right_matrix);
        try aliasing.ensureNoMatrixOverlap(T, workspace_view, result_matrix);
    }

    const product_scale: T = views.optionField(arguments, "product_scale", core.one(T));
    const result_scale: T = views.optionField(arguments, "result_scale", core.zero(T));
    executeMatrixProductWithWorkspace(T, left_matrix, right_matrix, result_matrix, workspace[0..required_workspace], product_scale, result_scale);
}
