// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later

const vector_ops = @import("../vector.zig");
const matrix_vector_ops = @import("../matrix_vector.zig");
const matrix_matrix_ops = @import("../matrix_matrix.zig");
const operands = @import("operands.zig");
const scalar = @import("../shared/scalar.zig");

pub const BlasInt = scalar.BlasInt;

fn sharedLength(first_length: BlasInt, second_length: BlasInt) BlasInt {
    return @min(first_length, second_length);
}

pub fn swapVectors(comptime T: type, args: operands.VectorSwap(T)) void {
    const length = sharedLength(args.first.length, args.second.length);
    vector_ops.swap(T, length, args.first.values, args.first.stride, args.second.values, args.second.stride);
}

/// Borrow the caller's live operation without admitting or owning another one.
pub fn swapVectorsAdmitted(comptime T: type, operation: *const @import("zynum-execution").context.Operation, args: operands.VectorSwap(T)) void {
    const length = sharedLength(args.first.length, args.second.length);
    vector_ops.swapWithResources(T, .{ .admitted = operation }, length, args.first.values, args.first.stride, args.second.values, args.second.stride);
}

pub fn copyVector(comptime T: type, args: operands.VectorCopy(T)) void {
    const length = sharedLength(args.source.length, args.destination.length);
    vector_ops.copy(T, length, args.source.values, args.source.stride, args.destination.values, args.destination.stride);
}

/// Borrow admission; used by checked GEMV's workspace copy-in/copyback.
pub fn copyVectorAdmitted(comptime T: type, operation: *const @import("zynum-execution").context.Operation, args: operands.VectorCopy(T)) void {
    const length = sharedLength(args.source.length, args.destination.length);
    vector_ops.copyWithResources(T, .{ .admitted = operation }, length, args.source.values, args.source.stride, args.destination.values, args.destination.stride);
}

pub fn scaleVector(comptime T: type, args: operands.VectorScale(T)) void {
    vector_ops.scal(T, args.vector.length, args.scale, args.vector.values, args.vector.stride);
}

/// Borrow the caller's live operation without admitting or owning another one.
pub fn scaleVectorAdmitted(comptime T: type, operation: *const @import("zynum-execution").context.Operation, args: operands.VectorScale(T)) void {
    vector_ops.scalWithResources(T, .{ .admitted = operation }, args.vector.length, args.scale, args.vector.values, args.vector.stride);
}

pub fn addScaledVector(comptime T: type, args: operands.ScaledVectorAdd(T)) void {
    const length = sharedLength(args.source.length, args.destination.length);
    vector_ops.axpy(T, length, args.scale, args.source.values, args.source.stride, args.destination.values, args.destination.stride);
}

/// Borrow the caller's live operation without admitting or owning another one.
pub fn addScaledVectorAdmitted(comptime T: type, operation: *const @import("zynum-execution").context.Operation, args: operands.ScaledVectorAdd(T)) void {
    const length = sharedLength(args.source.length, args.destination.length);
    vector_ops.axpyWithResources(T, .{ .admitted = operation }, length, args.scale, args.source.values, args.source.stride, args.destination.values, args.destination.stride);
}

pub fn combineVectors(comptime T: type, args: operands.VectorLinearCombination(T)) void {
    const length = sharedLength(args.source.length, args.destination.length);
    vector_ops.axpby(T, length, args.source_scale, args.source.values, args.source.stride, args.destination_scale, args.destination.values, args.destination.stride);
}

/// Borrow the caller's live operation without admitting or owning another one.
pub fn combineVectorsAdmitted(comptime T: type, operation: *const @import("zynum-execution").context.Operation, args: operands.VectorLinearCombination(T)) void {
    const length = sharedLength(args.source.length, args.destination.length);
    vector_ops.axpbyWithResources(T, .{ .admitted = operation }, length, args.source_scale, args.source.values, args.source.stride, args.destination_scale, args.destination.values, args.destination.stride);
}

pub fn dotProduct(comptime T: type, args: operands.VectorDot(T)) T {
    const length = sharedLength(args.left.length, args.right.length);
    return vector_ops.dot(T, length, args.left.values, args.left.stride, args.right.values, args.right.stride, args.conjugate_left);
}

/// Borrow the caller's live operation without admitting or owning another one.
pub fn dotProductAdmitted(comptime T: type, operation: *const @import("zynum-execution").context.Operation, args: operands.VectorDot(T)) T {
    const length = sharedLength(args.left.length, args.right.length);
    return vector_ops.dotWithResources(T, .{ .admitted = operation }, length, args.left.values, args.left.stride, args.right.values, args.right.stride, args.conjugate_left);
}

pub fn euclideanNorm(comptime T: type, vector: operands.ConstVector(T)) scalar.Real(T) {
    return vector_ops.nrm2(T, vector.length, vector.values, vector.stride);
}

/// Borrow the caller's live operation without admitting or owning another one.
pub fn euclideanNormAdmitted(comptime T: type, operation: *const @import("zynum-execution").context.Operation, vector: operands.ConstVector(T)) scalar.Real(T) {
    return vector_ops.nrm2WithResources(T, .{ .admitted = operation }, vector.length, vector.values, vector.stride);
}

pub fn multiplyMatrixVector(comptime T: type, args: operands.MatrixVectorProduct(T)) void {
    matrix_vector_ops.gemv(
        T,
        args.matrix.transpose,
        args.matrix.row_count,
        args.matrix.column_count,
        args.product_scale,
        args.matrix.values,
        args.matrix.leading_dimension,
        args.input.values,
        args.input.stride,
        args.output_scale,
        args.output.values,
        args.output.stride,
    );
}
/// Borrow the outer operation through GEMV scheduling and scratch.
pub fn multiplyMatrixVectorAdmitted(comptime T: type, operation: *const @import("zynum-execution").context.Operation, args: operands.MatrixVectorProduct(T)) void {
    matrix_vector_ops.gemvAdmitted(
        T,
        operation,
        args.matrix.transpose,
        args.matrix.row_count,
        args.matrix.column_count,
        args.product_scale,
        args.matrix.values,
        args.matrix.leading_dimension,
        args.input.values,
        args.input.stride,
        args.output_scale,
        args.output.values,
        args.output.stride,
    );
}

pub fn multiplyMatrices(comptime T: type, args: operands.MatrixProduct(T)) void {
    matrix_matrix_ops.gemm(
        T,
        args.left.transpose,
        args.right.transpose,
        args.output.row_count,
        args.output.column_count,
        args.left.effectiveColumnCount(),
        args.product_scale,
        args.left.values,
        args.left.leading_dimension,
        args.right.values,
        args.right.leading_dimension,
        args.output_scale,
        args.output.values,
        args.output.leading_dimension,
    );
}

/// Borrow the outer operation through real matrix scheduling and scratch.
pub fn multiplyRealMatricesAdmitted(comptime T: type, operation: *const @import("zynum-execution").context.Operation, args: operands.MatrixProduct(T)) void {
    matrix_matrix_ops.gemmRealAdmitted(
        T,
        operation,
        args.left.transpose,
        args.right.transpose,
        args.output.row_count,
        args.output.column_count,
        args.left.effectiveColumnCount(),
        args.product_scale,
        args.left.values,
        args.left.leading_dimension,
        args.right.values,
        args.right.leading_dimension,
        args.output_scale,
        args.output.values,
        args.output.leading_dimension,
    );
}

/// Borrow the outer operation through complex matrix scheduling and scratch.
pub fn multiplyComplexMatricesAdmitted(comptime T: type, operation: *const @import("zynum-execution").context.Operation, args: operands.MatrixProduct(T)) void {
    matrix_matrix_ops.gemmComplexAdmitted(
        T,
        operation,
        args.left.transpose,
        args.right.transpose,
        args.output.row_count,
        args.output.column_count,
        args.left.effectiveColumnCount(),
        args.product_scale,
        args.left.values,
        args.left.leading_dimension,
        args.right.values,
        args.right.leading_dimension,
        args.output_scale,
        args.output.values,
        args.output.leading_dimension,
    );
}

/// Pointer operands have already passed the public view, shape and alias checks.
pub fn multiplyF64NNExplicit(operation: *const @import("zynum-execution").Operation, args: operands.MatrixProduct(f64), report: *@import("zynum-execution").CallReport) @import("zynum-execution").ExecutionError!void {
    try @import("../matrix_matrix/gemm.zig").gemmF64NNExplicit(operation, args, report);
}
