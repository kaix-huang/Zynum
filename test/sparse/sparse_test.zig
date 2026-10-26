// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later

const std = @import("std");
const sparse = @import("zynum-sparse");

test {
    _ = @import("operator_test.zig");
    _ = @import("checked_adapter_test.zig");
    _ = @import("mixed_index_test.zig");
    _ = @import("triangular_test.zig");
    _ = @import("cg_test.zig");
    _ = @import("complex_test.zig");
    _ = @import("formats_test.zig");
    _ = @import("optimization_test.zig");
    _ = @import("randomized_conversion_test.zig");
    _ = @import("object_test.zig");
    _ = @import("context_object_test.zig");
    _ = @import("interop/handle_test.zig");
    _ = @import("interop/mkl_sparse_adapter_test.zig");
    _ = @import("interop/raw_lifecycle_test.zig");
    _ = @import("interop/mkl_sparse_mv_test.zig");
    _ = @import("interop/mkl_sparse_mm_test.zig");
}

fn tolerance(comptime T: type) T {
    return if (T == f32) 0.00001 else 0.000000000001;
}

fn expectClose(comptime T: type, expected: []const T, actual: []const T) !void {
    try std.testing.expectEqual(expected.len, actual.len);
    for (expected, actual) |want, got| {
        try std.testing.expectApproxEqAbs(want, got, tolerance(T));
    }
}

// Independent row-major dense reference for the logical matrix. Deliberately
// does not expand or traverse CSR arrays, so CSR addressing mistakes are visible.
fn referenceProduct(comptime T: type, transform: sparse.MatrixTransform, dense: []const T, rows: usize, columns: usize, input: []const T, alpha: T, beta: T, output: []T) void {
    const effective_rows = if (transform == .normal) rows else columns;
    const effective_columns = if (transform == .normal) columns else rows;
    for (0..effective_rows) |row| {
        var sum: T = 0;
        for (0..effective_columns) |column| {
            const entry = if (transform == .normal) dense[row * columns + column] else dense[column * columns + row];
            sum += entry * input[column];
        }
        output[row] = alpha * sum + beta * output[row];
    }
}

test "Sparse CSR SpMV matches dense reference for all scalar and index types" {
    inline for (.{ f32, f64 }) |T| {
        inline for (.{ i32, u32, i64, u64 }) |I| {
            const offsets = [_]I{ 0, 3, 3, 5, 7 };
            const indices = [_]I{ 2, 0, 2, 1, 0, 2, 1 };
            const values = [_]T{ 2, 1, -0.5, 3, -2, 4, 1 };
            const dense = [_]T{ 1, 0, 1.5, 0, 0, 0, -2, 3, 0, 0, 1, 4 };
            const a = try sparse.constCsrMatrix(T, I, &offsets, &indices, &values, .{ .row_count = 4, .column_count = 3 });
            const input = [_]T{ 2, -1, 3 };
            var output = [_]T{ 4, 6, 8, 10 };
            var expected = output;
            referenceProduct(T, .normal, &dense, 4, 3, &input, -2, 0.5, &expected);
            try sparse.matrixVectorMultiply(.{
                .matrix = a,
                .input_vector = &input,
                .result_vector = &output,
                .product_scale = -2,
                .result_scale = 0.5,
            });
            try expectClose(T, &expected, &output);

            const transpose_input = [_]T{ 1, 2, -1, 0.5 };
            var transpose_output = [_]T{ 7, 8, 9 };
            var transpose_expected = transpose_output;
            referenceProduct(T, .transposed, &dense, 4, 3, &transpose_input, 1.5, -1, &transpose_expected);
            try sparse.matrixVectorMultiply(.{
                .matrix = a.transposed(),
                .input_vector = &transpose_input,
                .result_vector = &transpose_output,
                .product_scale = 1.5,
                .result_scale = -1,
            });
            try expectClose(T, &transpose_expected, &transpose_output);
        }
    }
}

test "Sparse CSR permits empty rows, explicit zero values and stored-order duplicates" {
    // Reordering these entries can change the f32 result. The API promises that
    // duplicates are evaluated in storage order, without an implicit conversion.
    const a = try sparse.constCsrMatrix(f32, u32, &.{ 0, 0, 3, 3 }, &.{ 0, 0, 0 }, &.{ 16777216, -16777216, 1 }, .{ .row_count = 3, .column_count = 1 });
    var output = [_]f32{ 9, 9, 9 };
    try sparse.matrixVectorMultiply(.{ .matrix = a, .input_vector = &[_]f32{1}, .result_vector = &output });
    try std.testing.expectEqualSlices(f32, &.{ 0, 1, 0 }, &output);

    const zero = try sparse.constCsrMatrix(f64, i64, &.{ 0, 1 }, &.{0}, &.{0}, .{ .row_count = 1, .column_count = 1 });
    var zero_output = [_]f64{9};
    try sparse.matrixVectorMultiply(.{ .matrix = zero, .input_vector = &[_]f64{2}, .result_vector = &zero_output });
    try std.testing.expectEqual(@as(f64, 0), zero_output[0]);
}

test "Sparse CSR SpMV zero scales do not read unused numerical inputs" {
    inline for (.{ f32, f64 }) |T| {
        const nan = std.math.nan(T);
        const a = try sparse.constCsrMatrix(T, i32, &.{ 0, 1, 2 }, &.{ 1, 0 }, &.{ nan, nan }, .{ .row_count = 2, .column_count = 2 });
        inline for (.{ sparse.MatrixTransform.normal, sparse.MatrixTransform.transposed }) |operation| {
            var op_a = a;
            op_a.operation = operation;
            var output = [_]T{ nan, nan };
            try sparse.matrixVectorMultiply(.{
                .matrix = op_a,
                .input_vector = &[_]T{ nan, nan },
                .result_vector = &output,
                .product_scale = 0,
                .result_scale = 0,
            });
            try std.testing.expectEqualSlices(T, &.{ 0, 0 }, &output);
            output = .{ 2, -3 };
            try sparse.matrixVectorMultiply(.{
                .matrix = op_a,
                .input_vector = &[_]T{ nan, nan },
                .result_vector = &output,
                .product_scale = -@as(T, 0),
                .result_scale = 2,
            });
            try std.testing.expectEqualSlices(T, &.{ 4, -6 }, &output);
        }
        const finite = try sparse.constCsrMatrix(T, i32, &.{ 0, 1 }, &.{0}, &.{3}, .{ .row_count = 1, .column_count = 1 });
        var output = [_]T{nan};
        try sparse.matrixVectorMultiply(.{ .matrix = finite, .input_vector = &[_]T{2}, .result_vector = &output });
        try std.testing.expectEqual(@as(T, 6), output[0]);
    }
}

test "Sparse CSR SpMV handles rectangular zero dimensions and zero nnz" {
    const no_rows = try sparse.constCsrMatrix(f64, u64, &.{0}, &.{}, &.{}, .{ .row_count = 0, .column_count = 3 });
    var empty: [0]f64 = .{};
    try sparse.matrixVectorMultiply(.{ .matrix = no_rows, .input_vector = &[_]f64{ 1, 2, 3 }, .result_vector = &empty });
    var transpose_output = [_]f64{ 1, 2, 3 };
    try sparse.matrixVectorMultiply(.{ .matrix = no_rows.transposed(), .input_vector = &[_]f64{}, .result_vector = &transpose_output, .result_scale = -2 });
    try std.testing.expectEqualSlices(f64, &.{ -2, -4, -6 }, &transpose_output);

    const no_columns = try sparse.constCsrMatrix(f64, i64, &.{ 0, 0, 0 }, &.{}, &.{}, .{ .row_count = 2, .column_count = 0 });
    var output = [_]f64{ std.math.nan(f64), std.math.nan(f64) };
    try sparse.matrixVectorMultiply(.{ .matrix = no_columns, .input_vector = &[_]f64{}, .result_vector = &output });
    try std.testing.expectEqualSlices(f64, &.{ 0, 0 }, &output);
    const all_empty = try sparse.constCsrMatrix(f64, i32, &.{ 0, 0, 0 }, &.{}, &.{}, .{ .row_count = 2, .column_count = 4 });
    output = .{ 3, 5 };
    try sparse.matrixVectorMultiply(.{ .matrix = all_empty, .input_vector = &[_]f64{ 1, 2, 3, 4 }, .result_vector = &output, .result_scale = 1 });
    try std.testing.expectEqualSlices(f64, &.{ 3, 5 }, &output);
}

test "Sparse CSR rejects invalid offsets, lengths and columns" {
    const options = .{ .row_count = 2, .column_count = 3 };
    try std.testing.expectError(error.InvalidLength, sparse.constCsrMatrix(f64, i32, &.{ 0, 1 }, &.{0}, &.{1}, options));
    try std.testing.expectError(error.InvalidLength, sparse.constCsrMatrix(f64, i32, &.{ 0, 0, 1 }, &.{}, &.{1}, options));
    try std.testing.expectError(error.InvalidRowOffsets, sparse.constCsrMatrix(f64, i32, &.{ 1, 1, 1 }, &.{0}, &.{1}, options));
    try std.testing.expectError(error.InvalidRowOffsets, sparse.constCsrMatrix(f64, i32, &.{ 0, 1, 0 }, &.{0}, &.{1}, options));
    try std.testing.expectError(error.InvalidRowOffsets, sparse.constCsrMatrix(f64, i32, &.{ 0, 0, 0 }, &.{0}, &.{1}, options));
    try std.testing.expectError(error.InvalidRowOffsets, sparse.constCsrMatrix(f64, i32, &.{ 0, 2, 2 }, &.{0}, &.{1}, options));
    try std.testing.expectError(error.InvalidRowOffsets, sparse.constCsrMatrix(f64, i32, &.{ 0, -1, 1 }, &.{0}, &.{1}, options));
    try std.testing.expectError(error.InvalidColumnIndex, sparse.constCsrMatrix(f64, i32, &.{ 0, 0, 1 }, &.{-1}, &.{1}, options));
    try std.testing.expectError(error.InvalidColumnIndex, sparse.constCsrMatrix(f64, u64, &.{ 0, 0, 1 }, &.{3}, &.{1}, options));
    try std.testing.expectError(error.InvalidColumnIndex, sparse.constCsrMatrix(f64, u64, &.{ 0, 0, 1 }, &.{std.math.maxInt(u64)}, &.{1}, options));
    try std.testing.expectError(error.SizeOverflow, sparse.constCsrMatrix(f64, u64, &.{}, &.{}, &.{}, .{ .row_count = std.math.maxInt(usize), .column_count = 1 }));
}

test "Sparse CSR borrowed structure is rechecked before mutation, even for zero alpha" {
    var offsets = [_]u32{ 0, 1, 2 };
    var indices = [_]u32{ 0, 1 };
    const values = [_]f64{ 2, 3 };
    const a = try sparse.constCsrMatrix(f64, u32, &offsets, &indices, &values, .{ .row_count = 2, .column_count = 2 });
    var output = [_]f64{ 17, 19 };
    indices[1] = 2;
    try std.testing.expectError(error.InvalidColumnIndex, sparse.matrixVectorMultiply(.{ .matrix = a, .input_vector = &[_]f64{ 1, 2 }, .result_vector = &output, .product_scale = 0 }));
    try std.testing.expectEqualSlices(f64, &.{ 17, 19 }, &output);
    indices[1] = 1;
    offsets[2] = 1;
    try std.testing.expectError(error.InvalidRowOffsets, sparse.matrixVectorMultiply(.{ .matrix = a, .input_vector = &[_]f64{ 1, 2 }, .result_vector = &output }));
    try std.testing.expectEqualSlices(f64, &.{ 17, 19 }, &output);
    offsets[2] = 2;
    try std.testing.expectError(error.DimensionMismatch, sparse.matrixVectorMultiply(.{ .matrix = a, .input_vector = &[_]f64{1}, .result_vector = &output }));
    try std.testing.expectEqualSlices(f64, &.{ 17, 19 }, &output);
    try std.testing.expectError(error.DimensionMismatch, sparse.matrixVectorMultiply(.{ .matrix = a, .input_vector = &[_]f64{ 1, 2 }, .result_vector = output[0..1] }));
    try std.testing.expectEqualSlices(f64, &.{ 17, 19 }, &output);
}

test "Sparse CSR SpMV rejects full and partial numerical aliasing before mutation" {
    const a = try sparse.constCsrMatrix(f64, i32, &.{ 0, 1, 2 }, &.{ 0, 1 }, &.{ 2, 3 }, .{ .row_count = 2, .column_count = 2 });
    var shared = [_]f64{ 3, 5, 7 };
    try std.testing.expectError(error.AliasingNotAllowed, sparse.matrixVectorMultiply(.{ .matrix = a, .input_vector = shared[0..2], .result_vector = shared[0..2], .product_scale = 0 }));
    try std.testing.expectError(error.AliasingNotAllowed, sparse.matrixVectorMultiply(.{ .matrix = a, .input_vector = shared[0..2], .result_vector = shared[1..3] }));
    try std.testing.expectEqualSlices(f64, &.{ 3, 5, 7 }, &shared);
    const aliased_a = try sparse.constCsrMatrix(f64, i32, &.{ 0, 1, 2 }, &.{ 0, 1 }, shared[0..2], .{ .row_count = 2, .column_count = 2 });
    try std.testing.expectError(error.AliasingNotAllowed, sparse.matrixVectorMultiply(.{ .matrix = aliased_a, .input_vector = &[_]f64{ 1, 2 }, .result_vector = shared[1..3] }));
    try std.testing.expectEqualSlices(f64, &.{ 3, 5, 7 }, &shared);
}

test "Sparse CSR SpMV rejects output overlap with index and offset storage" {
    var indices = [_]u32{0};
    const index_output = std.mem.bytesAsSlice(f32, std.mem.sliceAsBytes(&indices));
    const a = try sparse.constCsrMatrix(f32, u32, &.{ 0, 1 }, &indices, &.{2}, .{ .row_count = 1, .column_count = 1 });
    try std.testing.expectError(error.AliasingNotAllowed, sparse.matrixVectorMultiply(.{ .matrix = a, .input_vector = &[_]f32{3}, .result_vector = index_output }));
    try std.testing.expectEqual(@as(u32, 0), indices[0]);

    var offsets = [_]u64{ 0, 1 };
    const offset_output = std.mem.bytesAsSlice(f64, std.mem.sliceAsBytes(&offsets));
    const b = try sparse.constCsrMatrix(f64, u64, &offsets, &.{0}, &.{2}, .{ .row_count = 1, .column_count = 1 });
    try std.testing.expectError(error.AliasingNotAllowed, sparse.matrixVectorMultiply(.{ .matrix = b, .input_vector = &[_]f64{3}, .result_vector = offset_output[1..2] }));
    try std.testing.expectEqualSlices(u64, &.{ 0, 1 }, &offsets);
}

test "Sparse CSR SpMM matches dense reference with transpose and padded columns" {
    inline for (.{ f32, f64 }) |T| {
        inline for (.{ u32, i64 }) |I| {
            const a = try sparse.constCsrMatrix(T, I, &.{ 0, 3, 3, 5, 7 }, &.{ 2, 0, 2, 1, 0, 2, 1 }, &.{ 2, 1, -0.5, 3, -2, 4, 1 }, .{ .row_count = 4, .column_count = 3 });
            const dense = [_]T{ 1, 0, 1.5, 0, 0, 0, -2, 3, 0, 0, 1, 4 };
            inline for (.{ sparse.MatrixTransform.normal, sparse.MatrixTransform.transposed }) |operation| {
                var op_a = a;
                op_a.operation = operation;
                const input_rows = op_a.effectiveColumnCount();
                const output_rows = op_a.effectiveRowCount();
                const input_ld = input_rows + 2;
                const output_ld = output_rows + 1;
                var input_values: [20]T = @splat(std.math.nan(T));
                var output_values: [20]T = @splat(12345);
                var expected = output_values;
                for (0..3) |column| {
                    for (0..input_rows) |row| input_values[column * input_ld + row] = @as(T, @floatFromInt(column * 3 + row)) - 2;
                    for (0..output_rows) |row| {
                        output_values[column * output_ld + row] = @floatFromInt(column + row + 1);
                        expected[column * output_ld + row] = output_values[column * output_ld + row];
                    }
                    referenceProduct(T, operation, &dense, 4, 3, input_values[column * input_ld ..][0..input_rows], -1.5, 0.25, expected[column * output_ld ..][0..output_rows]);
                }
                const input_snapshot = input_values;
                const right = try sparse.constMatrix(T, &input_values, .{ .row_count = input_rows, .column_count = 3, .leading_dimension = input_ld });
                const output = try sparse.matrix(T, &output_values, .{ .row_count = output_rows, .column_count = 3, .leading_dimension = output_ld });
                try sparse.matrixMultiply(.{ .left_matrix = op_a, .right_matrix = right, .result_matrix = output, .product_scale = -1.5, .result_scale = 0.25 });
                try expectClose(T, &expected, &output_values);
                try std.testing.expectEqualSlices(u8, std.mem.sliceAsBytes(&input_snapshot), std.mem.sliceAsBytes(&input_values));
            }
        }
    }
}

test "Sparse CSR SpMM zero scales preserve padding and ignore unused NaNs" {
    inline for (.{ f32, f64 }) |T| {
        const nan = std.math.nan(T);
        const a = try sparse.constCsrMatrix(T, u64, &.{ 0, 1, 2 }, &.{ 1, 0 }, &.{ nan, nan }, .{ .row_count = 2, .column_count = 2 });
        const right = try sparse.constMatrix(T, &.{ nan, nan, nan, nan }, .{ .row_count = 2, .column_count = 2 });
        var output_values = [_]T{ nan, nan, 77, nan, nan, 88 };
        const output = try sparse.matrix(T, &output_values, .{ .row_count = 2, .column_count = 2, .leading_dimension = 3 });
        try sparse.matrixMultiply(.{ .left_matrix = a, .right_matrix = right, .result_matrix = output, .product_scale = 0 });
        try std.testing.expectEqualSlices(T, &.{ 0, 0, 77, 0, 0, 88 }, &output_values);
        output_values = .{ 1, 2, 77, 3, 4, 88 };
        try sparse.matrixMultiply(.{ .left_matrix = a.transposed(), .right_matrix = right, .result_matrix = output, .product_scale = 0, .result_scale = -2 });
        try std.testing.expectEqualSlices(T, &.{ -2, -4, 77, -6, -8, 88 }, &output_values);

        const finite = try sparse.constCsrMatrix(T, u64, &.{ 0, 1, 2 }, &.{ 1, 0 }, &.{ 2, 3 }, .{ .row_count = 2, .column_count = 2 });
        const finite_right = try sparse.constMatrix(T, &.{ 1, 2, 3, 4 }, .{ .row_count = 2, .column_count = 2 });
        output_values = .{ nan, nan, 77, nan, nan, 88 };
        try sparse.matrixMultiply(.{ .left_matrix = finite, .right_matrix = finite_right, .result_matrix = output });
        try std.testing.expectEqualSlices(T, &.{ 4, 3, 77, 8, 9, 88 }, &output_values);
    }
}

test "Sparse dense view checks capacity, leading dimensions and arithmetic overflow" {
    try std.testing.expectError(error.InvalidLeadingDimension, sparse.constMatrix(f64, &.{ 1, 2 }, .{ .row_count = 2, .column_count = 1, .leading_dimension = 1 }));
    try std.testing.expectError(error.BufferTooSmall, sparse.constMatrix(f64, &.{ 1, 2, 3, 4 }, .{ .row_count = 2, .column_count = 2, .leading_dimension = 3 }));
    try std.testing.expectError(error.SizeOverflow, sparse.constMatrix(f64, &.{}, .{ .row_count = 1, .column_count = 3, .leading_dimension = std.math.maxInt(usize) }));
    try std.testing.expectError(error.SizeOverflow, sparse.constMatrix(f64, &.{}, .{ .row_count = 2, .column_count = 2, .leading_dimension = std.math.maxInt(usize) }));
    _ = try sparse.constMatrix(f64, &.{ 1, 2, 0, 3, 4 }, .{ .row_count = 2, .column_count = 2, .leading_dimension = 3 });
    _ = try sparse.constMatrix(f64, &.{}, .{ .row_count = 0, .column_count = 3, .leading_dimension = std.math.maxInt(usize) });
}

test "Sparse CSR SpMM empty dimensions avoid unused leading-dimension arithmetic" {
    var empty: [0]f64 = .{};
    const no_columns = try sparse.constCsrMatrix(f64, u32, &.{ 0, 0, 0 }, &.{}, &.{}, .{ .row_count = 2, .column_count = 0 });
    const right = try sparse.constMatrix(f64, &.{}, .{ .row_count = 0, .column_count = 3, .leading_dimension = std.math.maxInt(usize) });
    var values = [_]f64{ 1, 2, 99, 3, 4, 99, 5, 6 };
    const output = try sparse.matrix(f64, &values, .{ .row_count = 2, .column_count = 3, .leading_dimension = 3 });
    try sparse.matrixMultiply(.{ .left_matrix = no_columns, .right_matrix = right, .result_matrix = output, .result_scale = -1 });
    try std.testing.expectEqualSlices(f64, &.{ -1, -2, 99, -3, -4, 99, -5, -6 }, &values);

    const no_rows = try sparse.constCsrMatrix(f64, u32, &.{0}, &.{}, &.{}, .{ .row_count = 0, .column_count = 0 });
    const empty_output = try sparse.matrix(f64, &empty, .{ .row_count = 0, .column_count = 3, .leading_dimension = std.math.maxInt(usize) });
    try sparse.matrixMultiply(.{ .left_matrix = no_rows, .right_matrix = right, .result_matrix = empty_output });

    const nonempty = try sparse.constCsrMatrix(f64, u32, &.{ 0, 1 }, &.{0}, &.{2}, .{ .row_count = 1, .column_count = 1 });
    const no_rhs = try sparse.constMatrix(f64, &.{}, .{ .row_count = 1, .column_count = 0, .leading_dimension = std.math.maxInt(usize) });
    const no_output = try sparse.matrix(f64, &empty, .{ .row_count = 1, .column_count = 0, .leading_dimension = std.math.maxInt(usize) });
    try sparse.matrixMultiply(.{ .left_matrix = nonempty, .right_matrix = no_rhs, .result_matrix = no_output });
}

test "Sparse CSR SpMM validates all operands before scaling any output" {
    var indices = [_]i32{ 0, 1 };
    const a = try sparse.constCsrMatrix(f64, i32, &.{ 0, 1, 2 }, &indices, &.{ 2, 3 }, .{ .row_count = 2, .column_count = 2 });
    const right = try sparse.constMatrix(f64, &.{ 1, 2, 3, 4 }, .{ .row_count = 2, .column_count = 2 });
    var output_values = [_]f64{ 5, 6, 7, 8 };
    const output = try sparse.matrix(f64, &output_values, .{ .row_count = 2, .column_count = 2 });
    indices[1] = 2;
    try std.testing.expectError(error.InvalidColumnIndex, sparse.matrixMultiply(.{ .left_matrix = a, .right_matrix = right, .result_matrix = output, .product_scale = 0 }));
    indices[1] = 1;
    var bad_right = right;
    bad_right.row_count = 1;
    try std.testing.expectError(error.DimensionMismatch, sparse.matrixMultiply(.{ .left_matrix = a, .right_matrix = bad_right, .result_matrix = output }));
    var bad_output = output;
    bad_output.values = output_values[0..3];
    try std.testing.expectError(error.BufferTooSmall, sparse.matrixMultiply(.{ .left_matrix = a, .right_matrix = right, .result_matrix = bad_output }));
    bad_output = output;
    bad_output.leading_dimension = 1;
    try std.testing.expectError(error.InvalidLeadingDimension, sparse.matrixMultiply(.{ .left_matrix = a, .right_matrix = right, .result_matrix = bad_output }));
    bad_output = output;
    bad_output.leading_dimension = std.math.maxInt(usize);
    try std.testing.expectError(error.SizeOverflow, sparse.matrixMultiply(.{ .left_matrix = a, .right_matrix = right, .result_matrix = bad_output }));
    try std.testing.expectEqualSlices(f64, &.{ 5, 6, 7, 8 }, &output_values);
}

test "Sparse CSR SpMM rejects dense input, padding and sparse storage aliasing" {
    const a = try sparse.constCsrMatrix(f64, u32, &.{ 0, 1 }, &.{0}, &.{2}, .{ .row_count = 1, .column_count = 1 });
    var shared = [_]f64{ 3, 5, 7, 11 };
    const right = try sparse.constMatrix(f64, &shared, .{ .row_count = 1, .column_count = 2, .leading_dimension = 2 });
    const exact_output = try sparse.matrix(f64, &shared, .{ .row_count = 1, .column_count = 2, .leading_dimension = 2 });
    try std.testing.expectError(error.AliasingNotAllowed, sparse.matrixMultiply(.{ .left_matrix = a, .right_matrix = right, .result_matrix = exact_output, .product_scale = 0 }));
    const padding_output = try sparse.matrix(f64, shared[1..], .{ .row_count = 1, .column_count = 2, .leading_dimension = 2 });
    try std.testing.expectError(error.AliasingNotAllowed, sparse.matrixMultiply(.{ .left_matrix = a, .right_matrix = right, .result_matrix = padding_output }));
    try std.testing.expectEqualSlices(f64, &.{ 3, 5, 7, 11 }, &shared);

    const alias_a = try sparse.constCsrMatrix(f64, u32, &.{ 0, 1 }, &.{0}, shared[1..2], .{ .row_count = 1, .column_count = 1 });
    const independent_right = try sparse.constMatrix(f64, &.{ 1, 2 }, .{ .row_count = 1, .column_count = 2 });
    try std.testing.expectError(error.AliasingNotAllowed, sparse.matrixMultiply(.{ .left_matrix = alias_a, .right_matrix = independent_right, .result_matrix = exact_output }));
    try std.testing.expectEqualSlices(f64, &.{ 3, 5, 7, 11 }, &shared);
}

test "Sparse multiplication allows shared read-only inputs and disjoint dense tails" {
    const shared_input = [_]f64{ 2, 3 };
    const a = try sparse.constCsrMatrix(f64, i32, &.{ 0, 1, 2 }, &.{ 0, 1 }, &shared_input, .{ .row_count = 2, .column_count = 2 });
    var output = [_]f64{ 0, 0 };
    try sparse.matrixVectorMultiply(.{ .matrix = a, .input_vector = &shared_input, .result_vector = &output });
    try std.testing.expectEqualSlices(f64, &.{ 4, 9 }, &output);

    const one = try sparse.constCsrMatrix(f64, i32, &.{ 0, 1 }, &.{0}, &.{2}, .{ .row_count = 1, .column_count = 1 });
    var backing = [_]f64{ 3, 5, 7, 11 };
    // Only the first two entries belong to the input's logical storage span.
    const right = try sparse.constMatrix(f64, &backing, .{ .row_count = 1, .column_count = 2 });
    const result = try sparse.matrix(f64, backing[2..], .{ .row_count = 1, .column_count = 2 });
    try sparse.matrixMultiply(.{ .left_matrix = one, .right_matrix = right, .result_matrix = result });
    try std.testing.expectEqualSlices(f64, &.{ 3, 5, 6, 10 }, &backing);
}

test "Sparse storage geometry checks empty leading dimensions before capacity" {
    try std.testing.expectError(error.InvalidLeadingDimension, sparse.constMatrix(f64, &.{}, .{ .row_count = 2, .column_count = 0, .leading_dimension = 0 }));
    _ = try sparse.constMatrix(f64, &.{}, .{ .row_count = 2, .column_count = 0, .leading_dimension = 2 });
    _ = try sparse.constMatrix(f64, &.{}, .{ .row_count = 0, .column_count = std.math.maxInt(usize), .leading_dimension = std.math.maxInt(usize) });
    try std.testing.expectError(error.InvalidLeadingDimension, sparse.constMatrix(f64, &.{}, .{ .row_count = 2, .column_count = 3, .leading_dimension = 1 }));
}

test "Sparse byte geometry rejects wrapping endpoints before vector access" {
    const a = try sparse.constCsrMatrix(f64, u32, &.{ 0, 1 }, &.{0}, &.{2}, .{ .row_count = 1, .column_count = 1 });
    const invalid_output: [*]f64 = @ptrFromInt(std.math.maxInt(usize) - (@sizeOf(f64) - 1));
    try std.testing.expectError(error.SizeOverflow, sparse.matrixVectorMultiply(.{ .matrix = a, .input_vector = &[_]f64{3}, .result_vector = invalid_output[0..1] }));
    // An empty result still validates the input byte count before the numerical
    // empty-product return. No element of this deliberately huge slice is read.
    const count = std.math.maxInt(usize) / @sizeOf(f64) + 1;
    const empty = try sparse.constCsrMatrix(f64, u64, &.{0}, &.{}, &.{}, .{ .row_count = 0, .column_count = count });
    const input_base: [*]const f64 = @ptrFromInt(@alignOf(f64));
    var output: [0]f64 = .{};
    try std.testing.expectError(error.SizeOverflow, sparse.matrixVectorMultiply(.{ .matrix = empty, .input_vector = input_base[0..count], .result_vector = &output }));
}

test "Sparse multiple invalid operands preserve structural and shape precedence" {
    var offsets = [_]u32{ 0, 1 };
    const a = try sparse.constCsrMatrix(f64, u32, &offsets, &.{0}, &.{2}, .{ .row_count = 1, .column_count = 1 });
    const good_right = try sparse.constMatrix(f64, &.{3}, .{ .row_count = 1, .column_count = 1 });
    var bad_right = good_right;
    bad_right.leading_dimension = 0;
    var output = [_]f64{7};
    const out = try sparse.matrix(f64, &output, .{ .row_count = 1, .column_count = 1 });
    offsets[0] = 1;
    try std.testing.expectError(error.InvalidRowOffsets, sparse.matrixMultiply(.{ .left_matrix = a, .right_matrix = bad_right, .result_matrix = out }));
    offsets[0] = 0;
    const rectangular = try sparse.constCsrMatrix(f64, u32, &.{ 0, 1 }, &.{0}, &.{2}, .{ .row_count = 1, .column_count = 2 });
    var bad_out = out;
    bad_out.values = output[0..0];
    var workspace: [1]f64 = undefined;
    try std.testing.expectError(error.BufferTooSmall, sparse.triangularSolveInPlace(.{ .matrix = rectangular, .rhs = bad_out, .triangle = .lower, .diagonal = .non_unit, .workspace = &workspace }));
    try std.testing.expectEqual(@as(f64, 7), output[0]);
}

test "Sparse CG workspace and tolerance errors precede mixed storage aliases" {
    const a = try sparse.constCsrMatrix(f64, u32, &.{ 0, 1 }, &.{0}, &.{2}, .{ .row_count = 1, .column_count = 1 });
    var shared = [_]f64{ 7, 11, 13, 17 };
    const before = shared;
    try std.testing.expectError(error.WorkspaceTooSmall, sparse.conjugateGradient(.{ .matrix = a, .rhs = &[_]f64{3}, .solution = shared[0..1], .workspace = shared[0..3], .initial_guess = .zero, .relative_tolerance = -1, .absolute_tolerance = 0, .max_iterations = 1 }));
    try std.testing.expectError(error.InvalidTolerance, sparse.conjugateGradient(.{ .matrix = a, .rhs = &[_]f64{3}, .solution = shared[0..1], .workspace = &shared, .initial_guess = .zero, .relative_tolerance = -1, .absolute_tolerance = 0, .max_iterations = 1 }));
    try std.testing.expectError(error.AliasingNotAllowed, sparse.conjugateGradient(.{ .matrix = a, .rhs = &[_]f64{3}, .solution = shared[0..1], .workspace = &shared, .initial_guess = .zero, .relative_tolerance = 0, .absolute_tolerance = 0, .max_iterations = 1 }));
    try std.testing.expectEqualSlices(f64, &before, &shared);
}

test "CSR checked adapter retains structural error order over the shared pattern" {
    inline for (.{ i32, u32, i64, u64 }) |I| {
        const invalid: I = if (@typeInfo(I).int.signedness == .signed) -1 else std.math.maxInt(I);
        const ordinals = [_]I{invalid};
        const values = [_]f64{std.math.nan(f64)};
        var a: sparse.ConstCsrMatrix(f64, I) = .{
            .row_offsets = &.{0},
            .column_indices = &ordinals,
            .values = &.{},
            .row_count = std.math.maxInt(usize),
            .column_count = 1,
        };
        try std.testing.expectError(error.SizeOverflow, a.check());
        a.row_count = 0;
        try std.testing.expectError(error.InvalidLength, a.check());
        a.row_count = 1;
        a.values = &values;
        a.row_offsets = &.{ 0, 2 };
        try std.testing.expectError(error.InvalidRowOffsets, a.check());
        a.row_offsets = &.{ 0, 1 };
        try std.testing.expectError(error.InvalidColumnIndex, a.check());
        a.row_count = 2;
        a.row_offsets = &.{ 0, 2, 1 };
        try std.testing.expectError(error.InvalidRowOffsets, a.check());
        a.row_count = 1;
        a.row_offsets = &.{ 0, 1 };
        a.column_indices = &.{0};
        try a.check();
        a.column_count = 0;
        try std.testing.expectError(error.InvalidColumnIndex, a.check());
        try std.testing.expect(std.math.isNan(values[0]));
    }
}

test "Sparse checked three RHS retains literal rectangular arithmetic and full physical padding" {
    inline for (.{ f32, f64 }) |T| {
        inline for (.{ i32, u32, i64, u64 }) |I| {
            const a = try sparse.constCsrMatrix(T, I, &.{ 0, 3, 5 }, &.{ 2, 0, 0, 1, 0 }, &.{ -1, 3, -1, 4, 3 }, .{ .row_count = 2, .column_count = 3 });
            const b = [_]T{ 1, 4, 7, 91, 2, 5, 8, 92, 3, 6, 9, 93 };
            var c = [_]T{ 1, 4, 81, 2, 5, 82, 3, 6, 83, 84 };
            const input = try sparse.constMatrix(T, &b, .{ .row_count = 3, .column_count = 3, .leading_dimension = 4 });
            const output = try sparse.matrix(T, &c, .{ .row_count = 2, .column_count = 3, .leading_dimension = 3 });
            try sparse.matrixMultiply(.{ .left_matrix = a, .right_matrix = input, .result_matrix = output, .product_scale = 2, .result_scale = -1 });
            try std.testing.expectEqualSlices(T, &.{ -11, 34, 81, -10, 47, 82, -9, 60, 83, 84 }, &c);
        }
    }
}

test "Sparse checked panel delegation retains exact preflight priority and no mutation on errors" {
    var offsets = [_]i64{ 0, 1 };
    var indices = [_]i64{0};
    const a = try sparse.constCsrMatrix(f64, i64, &offsets, &indices, &.{2}, .{ .row_count = 1, .column_count = 1 });
    var shared = [_]f64{ 3, 7, 11, 13, 17, 19 };
    const before = shared;
    var input = try sparse.constMatrix(f64, shared[0..5], .{ .row_count = 1, .column_count = 3, .leading_dimension = 2 });
    var output = try sparse.matrix(f64, shared[1..6], .{ .row_count = 1, .column_count = 3, .leading_dimension = 2 });
    offsets[0] = 1;
    input.leading_dimension = 0;
    output.values = &.{};
    try std.testing.expectError(error.InvalidRowOffsets, sparse.matrixMultiply(.{ .left_matrix = a, .right_matrix = input, .result_matrix = output, .product_scale = 0 }));
    offsets[0] = 0;
    try std.testing.expectError(error.InvalidLeadingDimension, sparse.matrixMultiply(.{ .left_matrix = a, .right_matrix = input, .result_matrix = output, .product_scale = 0 }));
    input.leading_dimension = 2;
    try std.testing.expectError(error.BufferTooSmall, sparse.matrixMultiply(.{ .left_matrix = a, .right_matrix = input, .result_matrix = output, .product_scale = 0 }));
    output.values = shared[1..6];
    try std.testing.expectError(error.AliasingNotAllowed, sparse.matrixMultiply(.{ .left_matrix = a, .right_matrix = input, .result_matrix = output, .product_scale = 0 }));
    indices[0] = 1;
    try std.testing.expectError(error.InvalidColumnIndex, sparse.matrixMultiply(.{ .left_matrix = a, .right_matrix = input, .result_matrix = output, .product_scale = 0 }));
    try std.testing.expectEqualSlices(f64, &before, &shared);
}

test "Sparse checked zero K three RHS scales only logical output with arbitrary input stride" {
    const a = try sparse.constCsrMatrix(f64, u64, &.{ 0, 0, 0 }, &.{}, &.{}, .{ .row_count = 2, .column_count = 0 });
    const input = try sparse.constMatrix(f64, &.{}, .{ .row_count = 0, .column_count = 3, .leading_dimension = std.math.maxInt(usize) });
    var y = [_]f64{ 1, 2, 91, 3, 4, 92, 5, 6, 93 };
    const output = try sparse.matrix(f64, &y, .{ .row_count = 2, .column_count = 3, .leading_dimension = 3 });
    try sparse.matrixMultiply(.{ .left_matrix = a, .right_matrix = input, .result_matrix = output, .product_scale = std.math.inf(f64), .result_scale = -2 });
    try std.testing.expectEqualSlices(f64, &.{ -2, -4, 91, -6, -8, 92, -10, -12, 93 }, &y);
}

// Direct checked calls share the existing Core default, not a facade-local owner.
const admission_execution = @import("zynum-execution");
const admission_core = @import("zynum-core");
const AdmissionWatch = struct { actual: []const u8, expected: []const u8 };
fn admissionWatch(actual: anytype, expected: anytype) AdmissionWatch {
    return .{ .actual = std.mem.asBytes(actual), .expected = std.mem.asBytes(expected) };
}
const AdmissionProbe = struct {
    var watches: ?[]const AdmissionWatch = null;
    var admitted: usize = 0;
    var finished: usize = 0;
    fn observe(ctx: *admission_execution.context.Context, event: admission_execution.context.TestEvent) void {
        if (ctx != admission_execution.getContext() or &ctx.executor != admission_execution.get() or ctx.snapshot().active_operations != 1)
            @panic("direct Sparse admission owner/count mismatch");
        const active = watches orelse @panic("unexpected nested or unarmed default call");
        switch (event) {
            .admitted => admitted += 1,
            .finishing => {
                for (active) |watch| if (!std.mem.eql(u8, watch.actual, watch.expected))
                    @panic("Sparse storage differs at finishing, before Operation.end");
                finished += 1;
            },
        }
    }
    fn install() void {
        watches = null;
        admitted = 0;
        finished = 0;
        admission_execution.context.Context.testing.configure(admission_execution.getContext(), 1, observe);
    }
    fn restore() void {
        admission_execution.context.Context.testing.configure(admission_execution.getContext(), std.math.maxInt(usize), null);
        watches = null;
    }
    fn arm(items: []const AdmissionWatch) void {
        if (watches != null or admission_execution.getContext().snapshot().active_operations != 0)
            @panic("probe must be armed only between completed calls");
        watches = items;
        admitted = 0;
        finished = 0;
    }
    fn complete() !void {
        try std.testing.expectEqual(@as(usize, 1), admitted);
        try std.testing.expectEqual(@as(usize, 1), finished);
        try std.testing.expectEqual(@as(usize, 0), admission_execution.getContext().snapshot().active_operations);
        watches = null;
    }
};

test "direct Sparse SpMV pairs one permit through success structural error and empty return" {
    AdmissionProbe.install();
    defer AdmissionProbe.restore();
    try std.testing.expect(sparse.matrixVectorMultiply == admission_core.sparse.matrixVectorMultiply);
    var offsets = [_]u32{ 0, 1 };
    const indices = [_]u32{0};
    const values = [_]f64{3};
    const input = [_]f64{2};
    const saved_indices = indices;
    const saved_values = values;
    const saved_input = input;
    var output = [_]f64{ 5, 91 };
    const a: sparse.ConstCsrMatrix(f64, u32) = .{ .row_offsets = &offsets, .column_indices = &indices, .values = &values, .row_count = 1, .column_count = 1 };
    {
        const expected = [_]f64{ 7, 91 };
        const saved_offsets = offsets;
        const watches = [_]AdmissionWatch{ admissionWatch(&output, &expected), admissionWatch(&offsets, &saved_offsets), admissionWatch(&indices, &saved_indices), admissionWatch(&values, &saved_values), admissionWatch(&input, &saved_input) };
        AdmissionProbe.arm(&watches);
        try sparse.matrixVectorMultiply(.{ .matrix = a, .input_vector = &input, .result_vector = output[0..1], .product_scale = 2, .result_scale = -1 });
        try AdmissionProbe.complete();
    }
    {
        offsets[0] = 1;
        const saved_offsets = offsets;
        const expected = output;
        const watches = [_]AdmissionWatch{ admissionWatch(&output, &expected), admissionWatch(&offsets, &saved_offsets), admissionWatch(&values, &saved_values), admissionWatch(&input, &saved_input) };
        AdmissionProbe.arm(&watches);
        try std.testing.expectError(error.InvalidRowOffsets, sparse.matrixVectorMultiply(.{ .matrix = a, .input_vector = input[0..0], .result_vector = output[0..1], .product_scale = 0 }));
        try AdmissionProbe.complete();
    }
    {
        offsets[0] = 0;
        const empty: sparse.ConstCsrMatrix(f64, u32) = .{ .row_offsets = offsets[0..1], .column_indices = indices[0..0], .values = values[0..0], .row_count = 0, .column_count = 0 };
        const expected = output;
        const saved_offsets = offsets;
        const watches = [_]AdmissionWatch{ admissionWatch(&output, &expected), admissionWatch(&offsets, &saved_offsets), admissionWatch(&values, &saved_values), admissionWatch(&input, &saved_input) };
        AdmissionProbe.arm(&watches);
        try sparse.matrixVectorMultiply(.{ .matrix = empty, .input_vector = input[0..0], .result_vector = output[0..0] });
        try AdmissionProbe.complete();
    }
}

test "direct Sparse SpMM permit includes padded three RHS and zero K huge stride" {
    AdmissionProbe.install();
    defer AdmissionProbe.restore();
    const offsets = [_]u32{ 0, 1 };
    const indices = [_]u32{0};
    const values = [_]f64{3};
    const input = [_]f64{ 2, 71, 4, 72, 6, 73 };
    const saved_input = input;
    const saved_values = values;
    var output = [_]f64{ 5, 91, 7, 92, 9, 93 };
    const a: sparse.ConstCsrMatrix(f64, u32) = .{ .row_offsets = &offsets, .column_indices = &indices, .values = &values, .row_count = 1, .column_count = 1 };
    const b: sparse.ConstMatrix(f64) = .{ .values = &input, .row_count = 1, .column_count = 3, .leading_dimension = 2 };
    const c: sparse.Matrix(f64) = .{ .values = &output, .row_count = 1, .column_count = 3, .leading_dimension = 2 };
    {
        const expected = [_]f64{ 7, 91, 17, 92, 27, 93 };
        const watches = [_]AdmissionWatch{ admissionWatch(&output, &expected), admissionWatch(&input, &saved_input), admissionWatch(&values, &saved_values) };
        AdmissionProbe.arm(&watches);
        try sparse.matrixMultiply(.{ .left_matrix = a, .right_matrix = b, .result_matrix = c, .product_scale = 2, .result_scale = -1 });
        try AdmissionProbe.complete();
    }
    {
        var bad_b = b;
        bad_b.leading_dimension = 0;
        var bad_c = c;
        bad_c.row_count = 2;
        const expected = output;
        const watches = [_]AdmissionWatch{ admissionWatch(&output, &expected), admissionWatch(&input, &saved_input) };
        AdmissionProbe.arm(&watches);
        try std.testing.expectError(error.InvalidLeadingDimension, sparse.matrixMultiply(.{ .left_matrix = a, .right_matrix = bad_b, .result_matrix = bad_c }));
        try AdmissionProbe.complete();
    }
    {
        const empty_offsets = [_]u32{ 0, 0 };
        const empty: sparse.ConstCsrMatrix(f64, u32) = .{ .row_offsets = &empty_offsets, .column_indices = indices[0..0], .values = values[0..0], .row_count = 1, .column_count = 0 };
        const empty_b: sparse.ConstMatrix(f64) = .{ .values = input[0..0], .row_count = 0, .column_count = 3, .leading_dimension = std.math.maxInt(usize) };
        const expected = [_]f64{ 14, 91, 34, 92, 54, 93 };
        const watches = [_]AdmissionWatch{ admissionWatch(&output, &expected), admissionWatch(&input, &saved_input), admissionWatch(&values, &saved_values) };
        AdmissionProbe.arm(&watches);
        try sparse.matrixMultiply(.{ .left_matrix = empty, .right_matrix = empty_b, .result_matrix = c, .result_scale = 2 });
        try AdmissionProbe.complete();
    }
}

test "direct CSC conversion permit covers complete result and capacity error before workspace" {
    AdmissionProbe.install();
    defer AdmissionProbe.restore();
    const input_offsets = [_]u32{ 0, 2, 3 };
    const rows = [_]u32{ 1, 0, 1 };
    const input_values = [_]f64{ 2, 3, 4 };
    const saved_rows = rows;
    const saved_input_values = input_values;
    const a: sparse.ConstCscMatrix(f64, u32) = .{ .column_offsets = &input_offsets, .row_indices = &rows, .values = &input_values, .row_count = 2, .column_count = 2 };
    var offsets = [_]u32{ 9, 9, 9, 99 };
    var columns = [_]u32{ 8, 8, 8, 88 };
    var values = [_]f64{ 6, 6, 6, 66 };
    var work = [_]usize{ 7, 7, 77 };
    {
        const want_offsets = [_]u32{ 0, 1, 3, 99 };
        const want_columns = [_]u32{ 0, 0, 1, 88 };
        const want_values = [_]f64{ 3, 2, 4, 66 };
        const tail: usize = 77;
        const watches = [_]AdmissionWatch{ admissionWatch(&offsets, &want_offsets), admissionWatch(&columns, &want_columns), admissionWatch(&values, &want_values), admissionWatch(&work[2], &tail), admissionWatch(&rows, &saved_rows), admissionWatch(&input_values, &saved_input_values) };
        AdmissionProbe.arm(&watches);
        const result = try sparse.cscToCsr(.{ .matrix = a, .row_offsets = &offsets, .column_indices = &columns, .values = &values, .workspace = &work });
        try AdmissionProbe.complete();
        try std.testing.expect(result.values.ptr == @as([]const f64, &values).ptr and result.values.len == 3);
    }
    {
        const before_offsets = offsets;
        const before_columns = columns;
        const before_values = values;
        const before_work = work;
        const watches = [_]AdmissionWatch{ admissionWatch(&offsets, &before_offsets), admissionWatch(&columns, &before_columns), admissionWatch(&values, &before_values), admissionWatch(&work, &before_work) };
        AdmissionProbe.arm(&watches);
        try std.testing.expectError(error.BufferTooSmall, sparse.cscToCsr(.{ .matrix = a, .row_offsets = offsets[0..1], .column_indices = &columns, .values = &values, .workspace = work[0..0] }));
        try AdmissionProbe.complete();
    }
    {
        const empty: sparse.ConstCscMatrix(f64, u32) = .{ .column_offsets = input_offsets[0..1], .row_indices = rows[0..0], .values = input_values[0..0], .row_count = 2, .column_count = 0 };
        const want_offsets = [_]u32{ 0, 0, 0, 99 };
        const before_columns = columns;
        const before_values = values;
        const before_work = work;
        const watches = [_]AdmissionWatch{ admissionWatch(&offsets, &want_offsets), admissionWatch(&columns, &before_columns), admissionWatch(&values, &before_values), admissionWatch(&work, &before_work) };
        AdmissionProbe.arm(&watches);
        _ = try sparse.cscToCsr(.{ .matrix = empty, .row_offsets = &offsets, .column_indices = &columns, .values = &values, .workspace = work[0..0] });
        try AdmissionProbe.complete();
    }
}

test "direct COO conversion permit retains stable duplicates error atomicity and empty result" {
    AdmissionProbe.install();
    defer AdmissionProbe.restore();
    const rows = [_]u32{ 1, 0, 1 };
    const input_columns = [_]u32{ 0, 1, 0 };
    const input_values = [_]f64{ 2, 3, 4 };
    const saved_rows = rows;
    const saved_columns = input_columns;
    const saved_values = input_values;
    const a: sparse.ConstCooMatrix(f64, u32) = .{ .row_indices = &rows, .column_indices = &input_columns, .values = &input_values, .row_count = 2, .column_count = 2 };
    var offsets = [_]u32{ 9, 9, 9, 99 };
    var columns = [_]u32{ 8, 8, 8, 88 };
    var values = [_]f64{ 6, 6, 6, 66 };
    var work: [265]usize = @splat(77);
    {
        const want_offsets = [_]u32{ 0, 1, 3, 99 };
        const want_columns = [_]u32{ 1, 0, 0, 88 };
        const want_values = [_]f64{ 3, 2, 4, 66 };
        const tail: usize = 77;
        const watches = [_]AdmissionWatch{ admissionWatch(&offsets, &want_offsets), admissionWatch(&columns, &want_columns), admissionWatch(&values, &want_values), admissionWatch(&work[264], &tail), admissionWatch(&rows, &saved_rows), admissionWatch(&input_columns, &saved_columns), admissionWatch(&input_values, &saved_values) };
        AdmissionProbe.arm(&watches);
        _ = try sparse.cooToCsr(.{ .matrix = a, .row_offsets = &offsets, .column_indices = &columns, .values = &values, .workspace = &work });
        try AdmissionProbe.complete();
    }
    {
        const before_offsets = offsets;
        const before_columns = columns;
        const before_values = values;
        const before_work = work;
        const watches = [_]AdmissionWatch{ admissionWatch(&offsets, &before_offsets), admissionWatch(&columns, &before_columns), admissionWatch(&values, &before_values), admissionWatch(&work, &before_work) };
        AdmissionProbe.arm(&watches);
        try std.testing.expectError(error.WorkspaceTooSmall, sparse.cooToCsr(.{ .matrix = a, .row_offsets = &offsets, .column_indices = &columns, .values = &values, .workspace = work[0..0] }));
        try AdmissionProbe.complete();
    }
    {
        const empty: sparse.ConstCooMatrix(f64, u32) = .{ .row_indices = rows[0..0], .column_indices = input_columns[0..0], .values = input_values[0..0], .row_count = 2, .column_count = 2 };
        const want_offsets = [_]u32{ 0, 0, 0, 99 };
        const before_columns = columns;
        const before_values = values;
        const before_work = work;
        const watches = [_]AdmissionWatch{ admissionWatch(&offsets, &want_offsets), admissionWatch(&columns, &before_columns), admissionWatch(&values, &before_values), admissionWatch(&work, &before_work) };
        AdmissionProbe.arm(&watches);
        _ = try sparse.cooToCsr(.{ .matrix = empty, .row_offsets = &offsets, .column_indices = &columns, .values = &values, .workspace = work[0..0] });
        try AdmissionProbe.complete();
    }
}

test "direct Sparse triangular permit spans diagonal dry run solve and fixed empty workspace" {
    AdmissionProbe.install();
    defer AdmissionProbe.restore();
    const offsets = [_]u32{ 0, 1 };
    const indices = [_]u32{0};
    var values = [_]f64{2};
    var rhs = [_]f64{ 6, 91 };
    var work = [_]f64{ -1, 73 };
    const a: sparse.ConstCsrMatrix(f64, u32) = .{ .row_offsets = &offsets, .column_indices = &indices, .values = &values, .row_count = 1, .column_count = 1 };
    const b: sparse.Matrix(f64) = .{ .values = &rhs, .row_count = 1, .column_count = 1, .leading_dimension = 2 };
    {
        const want_rhs = [_]f64{ 3, 91 };
        const want_work = [_]f64{ 2, 73 };
        const saved_values = values;
        const watches = [_]AdmissionWatch{ admissionWatch(&rhs, &want_rhs), admissionWatch(&work, &want_work), admissionWatch(&values, &saved_values) };
        AdmissionProbe.arm(&watches);
        const result = try sparse.triangularSolveInPlace(.{ .matrix = a, .rhs = b, .triangle = sparse.Triangle.lower, .diagonal = sparse.Diagonal.non_unit, .workspace = &work });
        try AdmissionProbe.complete();
        try std.testing.expectEqual(@as(usize, 0), result.info);
        try std.testing.expectEqual(@as(usize, 1), result.completed_rhs);
    }
    {
        var bad_b = b;
        bad_b.row_count = 2;
        bad_b.leading_dimension = 2;
        var empty_work: [0]f64 = .{};
        const before_rhs = rhs;
        const before_work = work;
        const watches = [_]AdmissionWatch{ admissionWatch(&rhs, &before_rhs), admissionWatch(&work, &before_work) };
        AdmissionProbe.arm(&watches);
        try std.testing.expectError(error.DimensionMismatch, sparse.triangularSolveInPlace(.{ .matrix = a, .rhs = bad_b, .triangle = sparse.Triangle.lower, .diagonal = sparse.Diagonal.non_unit, .workspace = &empty_work }));
        try AdmissionProbe.complete();
    }
    {
        values[0] = 0;
        const before_rhs = rhs;
        const before_work = work;
        const watches = [_]AdmissionWatch{ admissionWatch(&rhs, &before_rhs), admissionWatch(&work, &before_work) };
        AdmissionProbe.arm(&watches);
        const result = try sparse.triangularSolveInPlace(.{ .matrix = a, .rhs = b, .triangle = sparse.Triangle.lower, .diagonal = sparse.Diagonal.non_unit, .workspace = &work });
        try AdmissionProbe.complete();
        try std.testing.expectEqual(@as(usize, 1), result.info);
        try std.testing.expectEqual(sparse.TriangularSolveFailure.singular_diagonal, result.failure.?);
    }
    {
        var empty_b = b;
        empty_b.column_count = 0;
        var empty_work: [0]f64 = .{};
        const before_rhs = rhs;
        const before_work = work;
        const watches = [_]AdmissionWatch{ admissionWatch(&rhs, &before_rhs), admissionWatch(&work, &before_work) };
        AdmissionProbe.arm(&watches);
        const result = try sparse.triangularSolveInPlace(.{ .matrix = a, .rhs = empty_b, .triangle = sparse.Triangle.lower, .diagonal = sparse.Diagonal.non_unit, .workspace = &empty_work });
        try AdmissionProbe.complete();
        try std.testing.expectEqual(@as(usize, 0), result.info);
    }
}

test "direct Sparse CG permit includes widened true residual argument error breakdown and empty solve" {
    AdmissionProbe.install();
    defer AdmissionProbe.restore();
    const offsets = [_]u32{ 0, 1 };
    const indices = [_]u32{0};
    var values = [_]f64{2};
    const rhs = [_]f64{6};
    var x = [_]f64{ 77, 91 };
    var work = [_]f64{ 0, 0, 0, 0, 73 };
    const a: sparse.ConstCsrMatrix(f64, u32) = .{ .row_offsets = &offsets, .column_indices = &indices, .values = &values, .row_count = 1, .column_count = 1 };
    {
        const want_x = [_]f64{ 3, 91 };
        const saved_rhs = rhs;
        const saved_values = values;
        const tail: f64 = 73;
        const watches = [_]AdmissionWatch{ admissionWatch(&x, &want_x), admissionWatch(&work[4], &tail), admissionWatch(&rhs, &saved_rhs), admissionWatch(&values, &saved_values) };
        AdmissionProbe.arm(&watches);
        const result = try sparse.conjugateGradient(.{ .matrix = a, .rhs = &rhs, .solution = x[0..1], .workspace = &work, .initial_guess = sparse.InitialGuess.zero, .relative_tolerance = 0, .absolute_tolerance = 0, .max_iterations = @as(usize, 2) });
        try AdmissionProbe.complete();
        try std.testing.expectEqual(sparse.ConjugateGradientStatus.converged, result.status);
        try std.testing.expectEqual(@as(usize, 1), result.iterations);
        try std.testing.expectEqual(@as(f128, 0), result.true_residual_norm.?);
    }
    {
        const before_x = x;
        const before_work = work;
        const watches = [_]AdmissionWatch{ admissionWatch(&x, &before_x), admissionWatch(&work, &before_work) };
        AdmissionProbe.arm(&watches);
        try std.testing.expectError(error.InvalidTolerance, sparse.conjugateGradient(.{ .matrix = a, .rhs = x[0..1], .solution = x[0..1], .workspace = &work, .initial_guess = sparse.InitialGuess.provided, .relative_tolerance = -1, .absolute_tolerance = 0, .max_iterations = @as(usize, 2) }));
        try AdmissionProbe.complete();
    }
    {
        values[0] = -2;
        const want_x = [_]f64{ 0, 91 };
        const tail: f64 = 73;
        const watches = [_]AdmissionWatch{ admissionWatch(&x, &want_x), admissionWatch(&work[4], &tail) };
        AdmissionProbe.arm(&watches);
        const result = try sparse.conjugateGradient(.{ .matrix = a, .rhs = &rhs, .solution = x[0..1], .workspace = &work, .initial_guess = sparse.InitialGuess.zero, .relative_tolerance = 0, .absolute_tolerance = 0, .max_iterations = @as(usize, 2) });
        try AdmissionProbe.complete();
        try std.testing.expectEqual(sparse.ConjugateGradientStatus.nonpositive_curvature, result.status);
        try std.testing.expectEqual(@as(usize, 0), result.iterations);
    }
    {
        const empty: sparse.ConstCsrMatrix(f64, u32) = .{ .row_offsets = offsets[0..1], .column_indices = indices[0..0], .values = values[0..0], .row_count = 0, .column_count = 0 };
        const before_x = x;
        const before_work = work;
        const watches = [_]AdmissionWatch{ admissionWatch(&x, &before_x), admissionWatch(&work, &before_work) };
        AdmissionProbe.arm(&watches);
        const result = try sparse.conjugateGradient(.{ .matrix = empty, .rhs = rhs[0..0], .solution = x[0..0], .workspace = work[0..0], .initial_guess = sparse.InitialGuess.zero, .relative_tolerance = 0, .absolute_tolerance = 0, .max_iterations = @as(usize, 0) });
        try AdmissionProbe.complete();
        try std.testing.expectEqual(sparse.ConjugateGradientStatus.converged, result.status);
        try std.testing.expectEqual(@as(usize, 0), result.iterations);
    }
}

test "checked Sparse SpMV borrows one closable operation after close request without touching default" {
    AdmissionProbe.install();
    defer AdmissionProbe.restore(); // Unarmed: any accidental default entry panics.
    const Context = admission_execution.context.Context;
    var ctx = Context.init(std.heap.c_allocator, admission_execution.test_support.policy, .internal_closable);
    const offsets = [_]u32{ 0, 1 };
    const indices = [_]u32{0};
    const values = [_]f64{3};
    const input = [_]f64{2};
    var output = [_]f64{ 5, 91 };
    const a: sparse.ConstCsrMatrix(f64, u32) = .{ .row_offsets = &offsets, .column_indices = &indices, .values = &values, .row_count = 1, .column_count = 1 };
    Context.testing.configure(&ctx, 1, null);
    var operation = try ctx.begin();
    var ended = false;
    defer if (!ended) operation.end();
    try std.testing.expectEqual(admission_execution.context.RequestResult.pending, ctx.requestClose());
    try std.testing.expectEqual(admission_execution.context.CloseResult.busy, ctx.tryProgressClose());
    try admission_core.internal.sparse_checked.matrixVectorMultiplyAdmitted(&operation, .{ .matrix = a, .input_vector = &input, .result_vector = output[0..1], .product_scale = 2, .result_scale = -1 });
    try std.testing.expectEqualSlices(f64, &.{ 7, 91 }, &output);
    try std.testing.expectEqual(@as(usize, 1), ctx.snapshot().active_operations);
    try std.testing.expectEqual(@as(usize, 0), admission_execution.getContext().snapshot().active_operations);
    try std.testing.expectError(error.DimensionMismatch, admission_core.internal.sparse_checked.matrixVectorMultiplyAdmitted(&operation, .{ .matrix = a, .input_vector = input[0..0], .result_vector = output[0..1] }));
    try std.testing.expectEqualSlices(f64, &.{ 7, 91 }, &output);
    operation.end();
    ended = true;
    try std.testing.expectEqual(admission_execution.context.CloseResult.closed, ctx.tryProgressClose());
    try std.testing.expectError(error.Closing, ctx.begin());
    try std.testing.expectEqualSlices(f64, &.{ 7, 91 }, &output);
}
