// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later

//! Explicit-provider checked operations on the original mathematical Core.
const execution = @import("zynum-execution");
const views = @import("views.zig");
const aliasing = @import("aliasing.zig");
const core = @import("../core/checked.zig");

pub const MatrixProductF64 = struct {
    left_matrix: views.ConstMatrix(f64),
    right_matrix: views.ConstMatrix(f64),
    result_matrix: views.Matrix(f64),
    product_scale: f64 = 1,
    result_scale: f64 = 0,
};
pub const NativeError = views.Error || execution.ExecutionError;

/// Column-major FP64 NN GEMM. Validation precedes admission, allocation, and
/// every output write in all optimization modes. The borrowed buffers must
/// remain valid until this synchronous call returns.
pub fn matrixMultiplyF64NN(context: *execution.Context, args: MatrixProductF64, report: *execution.CallReport) NativeError!void {
    report.* = .{};
    const a = args.left_matrix;
    const b = args.right_matrix;
    const c = args.result_matrix;
    try a.check();
    try b.check();
    try c.check();
    if (a.operation != .normal or b.operation != .normal) return error.Unsupported;
    if (a.column_count != b.row_count or a.row_count != c.row_count or b.column_count != c.column_count) return error.DimensionMismatch;
    try aliasing.ensureNoMatrixOverlap(f64, a, c);
    try aliasing.ensureNoMatrixOverlap(f64, b, c);
    var operation = context.begin() catch |err| return switch (err) {
        error.Closing => error.ContextClosing,
        error.CounterOverflow => error.CounterOverflow,
    };
    defer operation.end();
    try core.multiplyF64NNExplicit(&operation, .{
        .left = .{ .values = a.values.ptr, .row_count = a.row_count, .column_count = a.column_count, .leading_dimension = a.leading_dimension },
        .right = .{ .values = b.values.ptr, .row_count = b.row_count, .column_count = b.column_count, .leading_dimension = b.leading_dimension },
        .output = .{ .values = c.values.ptr, .row_count = c.row_count, .column_count = c.column_count, .leading_dimension = c.leading_dimension },
        .product_scale = args.product_scale,
        .output_scale = args.result_scale,
    }, report);
}
