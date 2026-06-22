// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later

const std = @import("std");
const builtin = @import("builtin");
const zynum = @import("zynum");

test "top-level package exposes Zynum BLAS namespace" {
    var values = [_]f64{ 1, 2, 3 };
    const vector = try zynum.blas.constVector(f64, &values, .{});
    try std.testing.expectEqual(@as(zynum.blas.BlasInt, 3), vector.length);
}

test "modern typed gemm API" {
    var a = [_]f64{
        1, 2,
        3, 4,
        5, 6,
    };
    var b = [_]f64{
        7,  8,  9,
        10, 11, 12,
    };
    var c = [_]f64{ 0, 0, 0, 0 };
    const left_matrix = try zynum.constMatrix(f64, &a, .{ .row_count = 2, .column_count = 3 });
    const right_matrix = try zynum.constMatrix(f64, &b, .{ .row_count = 3, .column_count = 2 });
    const result_matrix = try zynum.matrix(f64, &c, .{ .row_count = 2, .column_count = 2 });
    try zynum.matrixMultiply(.{
        .left_matrix = left_matrix,
        .right_matrix = right_matrix,
        .result_matrix = result_matrix,
    });
    try std.testing.expectApproxEqAbs(@as(f64, 76), c[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 100), c[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 103), c[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 136), c[3], 1e-12);
}

test "modern checked API rejects invalid shapes and short strides" {
    if (builtin.mode == .ReleaseFast) return error.SkipZigTest;

    var short = [_]f64{ 1, 2, 3 };
    try std.testing.expectError(error.BufferTooSmall, zynum.constVector(f64, &short, .{ .length = 3, .stride = 2 }));

    var a_data = [_]f64{ 1, 2, 3, 4, 5, 6 };
    var b_data = [_]f64{ 1, 2, 3, 4, 5, 6 };
    var c_data = [_]f64{ 0, 0, 0, 0 };
    const left_matrix = try zynum.constMatrix(f64, &a_data, .{ .row_count = 2, .column_count = 3 });
    const right_matrix = try zynum.constMatrix(f64, &b_data, .{ .row_count = 2, .column_count = 3 });
    const result_matrix = try zynum.matrix(f64, &c_data, .{ .row_count = 2, .column_count = 2 });
    try std.testing.expectError(error.DimensionMismatch, zynum.matrixMultiply(.{
        .left_matrix = left_matrix,
        .right_matrix = right_matrix,
        .result_matrix = result_matrix,
    }));
}

test "modern vector Into operations write out of place" {
    var input = [_]f64{ 1, 2, 3 };
    var result = [_]f64{ 0, 0, 0 };
    const input_vector = try zynum.constVector(f64, &input, .{});
    const result_vector = try zynum.vector(f64, &result, .{});

    try zynum.scaleVectorInto(.{
        .input_vector = input_vector,
        .result_vector = result_vector,
        .scale = 4,
    });

    try std.testing.expectEqualSlices(f64, &.{ 4, 8, 12 }, &result);
    try std.testing.expectEqualSlices(f64, &.{ 1, 2, 3 }, &input);
}

test "modern matrix vector multiply rejects aliasing without workspace" {
    if (builtin.mode == .ReleaseFast) return error.SkipZigTest;

    var matrix_data = [_]f64{ 1, 2, 3, 4 };
    var vector_data = [_]f64{ 20, 10 };
    const matrix_view = try zynum.constMatrix(f64, &matrix_data, .{ .row_count = 2, .column_count = 2 });
    const input_vector = try zynum.constVector(f64, &vector_data, .{});
    const result_vector = try zynum.vector(f64, &vector_data, .{});

    try std.testing.expectError(error.AliasingNotAllowed, zynum.matrixVectorMultiply(.{
        .matrix = matrix_view,
        .input_vector = input_vector,
        .result_vector = result_vector,
    }));
}

test "modern matrix vector multiply workspace supports in-place result" {
    var matrix_data = [_]f64{ 1, 2, 3, 4 };
    var vector_data = [_]f64{ 20, 10 };
    var workspace_data = [_]f64{ 0, 0 };
    const matrix_view = try zynum.constMatrix(f64, &matrix_data, .{ .row_count = 2, .column_count = 2 });
    const input_vector = try zynum.constVector(f64, &vector_data, .{});
    const result_vector = try zynum.vector(f64, &vector_data, .{});

    try std.testing.expectEqual(@as(usize, 2), try zynum.matrixVectorMultiplyWorkspaceLength(.{ .matrix = matrix_view }));
    try zynum.matrixVectorMultiplyWithWorkspace(.{
        .matrix = matrix_view,
        .input_vector = input_vector,
        .result_vector = result_vector,
        .workspace = &workspace_data,
    });

    try std.testing.expectApproxEqAbs(@as(f64, 50), vector_data[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 80), vector_data[1], 1e-12);
}

test "modern matrix multiply rejects aliasing without workspace" {
    if (builtin.mode == .ReleaseFast) return error.SkipZigTest;

    var left_data = [_]f64{ 1, 2, 3, 4 };
    var right_data = [_]f64{ 5, 6, 7, 8 };
    const left_matrix = try zynum.constMatrix(f64, &left_data, .{ .row_count = 2, .column_count = 2 });
    const right_matrix = try zynum.constMatrix(f64, &right_data, .{ .row_count = 2, .column_count = 2 });
    const result_matrix = try zynum.matrix(f64, &left_data, .{ .row_count = 2, .column_count = 2 });

    try std.testing.expectError(error.AliasingNotAllowed, zynum.matrixMultiply(.{
        .left_matrix = left_matrix,
        .right_matrix = right_matrix,
        .result_matrix = result_matrix,
    }));
}

test "modern matrix multiply workspace supports in-place result" {
    var left_data = [_]f64{ 1, 2, 3, 4 };
    var right_data = [_]f64{ 5, 6, 7, 8 };
    var workspace_data = [_]f64{ 0, 0, 0, 0 };
    const left_matrix = try zynum.constMatrix(f64, &left_data, .{ .row_count = 2, .column_count = 2 });
    const right_matrix = try zynum.constMatrix(f64, &right_data, .{ .row_count = 2, .column_count = 2 });
    const result_matrix = try zynum.matrix(f64, &left_data, .{ .row_count = 2, .column_count = 2 });

    try std.testing.expectEqual(@as(usize, 4), try zynum.matrixMultiplyWorkspaceLength(.{ .result_matrix = result_matrix }));
    try zynum.matrixMultiplyWithWorkspace(.{
        .left_matrix = left_matrix,
        .right_matrix = right_matrix,
        .result_matrix = result_matrix,
        .workspace = &workspace_data,
    });

    try std.testing.expectApproxEqAbs(@as(f64, 23), left_data[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 34), left_data[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 31), left_data[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 46), left_data[3], 1e-12);
}
