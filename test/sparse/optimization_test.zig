// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later

const std = @import("std");
const sparse = @import("zynum-sparse");

fn expectBitsOrNan(comptime T: type, expected: []const T, actual: []const T) !void {
    const U = if (T == f32) u32 else u64;
    try std.testing.expectEqual(expected.len, actual.len);
    for (expected, actual) |want, got| {
        if (std.math.isNan(want)) {
            try std.testing.expect(std.math.isNan(got));
        } else {
            try std.testing.expectEqual(@as(U, @bitCast(want)), @as(U, @bitCast(got)));
        }
    }
}

test "metadata reductions validate every position across vector and scalar tails" {
    inline for (.{ i32, u32, i64, u64 }) |I| {
        var offsets: [259]I = undefined;
        var columns: [258]I = undefined;
        const values: [258]f32 = @splat(1);
        for (&offsets, 0..) |*offset, index| offset.* = @intCast(index);
        for (&columns, 0..) |*column, index| column.* = @intCast(index % 17);
        for ([_]usize{ 0, 1, 2, 7, 8, 9, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 128, 129, 255, 256, 257, 258 }) |length| {
            const a = try sparse.constCsrMatrix(f32, I, offsets[0 .. length + 1], columns[0..length], values[0..length], .{ .row_count = length, .column_count = 17 });
            for (0..length) |entry| {
                const saved = columns[entry];
                columns[entry] = 17;
                try std.testing.expectError(error.InvalidColumnIndex, a.check());
                if (I == i32 or I == i64) {
                    columns[entry] = std.math.minInt(I);
                    try std.testing.expectError(error.InvalidColumnIndex, a.check());
                }
                columns[entry] = saved;
            }
            if (length > 1) {
                for (1..length) |position| {
                    const saved = offsets[position];
                    offsets[position] = @intCast(length + 1);
                    try std.testing.expectError(error.InvalidRowOffsets, a.check());
                    if (I == i32 or I == i64) {
                        offsets[position] = -1;
                        try std.testing.expectError(error.InvalidRowOffsets, a.check());
                    }
                    offsets[position] = saved;
                }
            }
            try a.check();
        }
    }
}

test "metadata reduction preserves dimensions larger than signed and narrow index ranges" {
    inline for (.{ i32, u32, i64 }) |I| {
        if (comptime @as(u128, std.math.maxInt(I)) < @as(u128, std.math.maxInt(usize))) {
            const maximum = std.math.maxInt(I);
            const columns: usize = @as(usize, @intCast(maximum)) + 1;
            const a = try sparse.constCsrMatrix(f64, I, &.{ 0, 1 }, &.{maximum}, &.{2}, .{ .row_count = 1, .column_count = columns });
            try a.check();
            var at_limit = a;
            at_limit.column_count = @intCast(maximum);
            try std.testing.expectError(error.InvalidColumnIndex, at_limit.check());
            if (I == i32 or I == i64) {
                var negative = a;
                negative.column_indices = &.{-1};
                try std.testing.expectError(error.InvalidColumnIndex, negative.check());
            }
        }
    }
    const wide = try sparse.constCsrMatrix(f64, u64, &.{ 0, 1 }, &.{@as(u64, std.math.maxInt(usize)) - 1}, &.{2}, .{ .row_count = 1, .column_count = std.math.maxInt(usize) });
    try wide.check();
}

test "fused SpMV preserves scale zero nonreads signed zeros and nonfinite arithmetic" {
    inline for (.{ f32, f64 }) |T| {
        const offsets = [_]i32{ 0, 3, 3, 4 };
        const indices = [_]i32{ 1, 0, 1, 0 };
        var values = [_]T{ 2, -1, -2, 0 };
        const a = try sparse.constCsrMatrix(T, i32, &offsets, &indices, &values, .{ .row_count = 3, .column_count = 2 });
        const x = [_]T{ 3, 4 };
        var actual: [3]T = @splat(std.math.nan(T));
        try sparse.matrixVectorMultiply(.{ .matrix = a, .input_vector = &x, .result_vector = &actual, .product_scale = 2 });
        try expectBitsOrNan(T, &.{ -6, 0, 0 }, &actual);

        actual = .{ 1, -@as(T, 0), 2 };
        try sparse.matrixVectorMultiply(.{ .matrix = a, .input_vector = &x, .result_vector = &actual, .product_scale = -1, .result_scale = 1 });
        try expectBitsOrNan(T, &.{ 4, -@as(T, 0), 2 }, &actual);

        actual = @splat(std.math.nan(T));
        try sparse.matrixVectorMultiply(.{ .matrix = a, .input_vector = &x, .result_vector = &actual, .product_scale = std.math.inf(T) });
        try expectBitsOrNan(T, &.{ -std.math.inf(T), std.math.nan(T), std.math.nan(T) }, &actual);

        const zero_input = [_]T{ -@as(T, 0), 0 };
        for ([_]T{ 1, -1 }) |alpha| {
            try sparse.matrixVectorMultiply(.{ .matrix = a, .input_vector = &zero_input, .result_vector = &actual, .product_scale = alpha });
            try expectBitsOrNan(T, &.{ 0, 0, 0 }, &actual);
        }

        // Both matrix and dense numerical values are unused when alpha is zero.
        @memset(&values, std.math.nan(T));
        const unused_input: [2]T = @splat(std.math.nan(T));
        for ([_]T{ 0, -@as(T, 0) }) |alpha| {
            actual = @splat(std.math.nan(T));
            try sparse.matrixVectorMultiply(.{ .matrix = a, .input_vector = &unused_input, .result_vector = &actual, .product_scale = alpha });
            try expectBitsOrNan(T, &.{ 0, 0, 0 }, &actual);
            actual = .{ -@as(T, 0), 2, 3 };
            try sparse.matrixVectorMultiply(.{ .matrix = a, .input_vector = &unused_input, .result_vector = &actual, .product_scale = alpha, .result_scale = 1 });
            try expectBitsOrNan(T, &.{ -@as(T, 0), 2, 3 }, &actual);
        }
    }
}

test "two RHS blocks and odd tails match independent dense products and preserve padding" {
    inline for (.{ f32, f64 }) |T| {
        inline for (.{ i32, u32, i64, u64 }) |I| {
            const offsets = [_]I{ 0, 4, 4, 6, 9, 9 };
            const indices = [_]I{ 3, 0, 3, 3, 2, 1, 0, 0, 2 };
            const initial_values = [_]T{ 2, 1, -0.5, 1, -2, 3, 4, -4, 0.25 };
            // Literal logical row-major matrix: independent of CSR metadata.
            // Binary fractions keep both accumulation orders exactly representable.
            const dense = [_][4]T{
                .{ 1, 0, 0, 2.5 },
                .{ 0, 0, 0, 0 },
                .{ 0, 3, -2, 0 },
                .{ 0, 0, 0.25, 0 },
                .{ 0, 0, 0, 0 },
            };
            for ([_]bool{ false, true }) |transpose| {
                const input_rows: usize = if (transpose) 5 else 4;
                const output_rows: usize = if (transpose) 4 else 5;
                const input_ld = input_rows + 2;
                const output_ld = output_rows + 3;
                for ([_]usize{ 0, 1, 2, 3, 7, 8, 9, 15, 16, 17 }) |rhs| {
                    for ([_]T{ 0, -@as(T, 0), 1, -1.5 }) |alpha| {
                        for ([_]T{ 0, -@as(T, 0), 1, -0.5 }) |beta| {
                            var values = initial_values;
                            if (alpha == 0) @memset(&values, std.math.nan(T));
                            const a = try sparse.constCsrMatrix(T, I, &offsets, &indices, &values, .{ .row_count = 5, .column_count = 4, .operation = if (transpose) sparse.MatrixTransform.transposed else sparse.MatrixTransform.normal });
                            var input: [160]T = @splat(12345);
                            var actual: [160]T = @splat(12345);
                            for (0..rhs) |column| {
                                for (0..input_rows) |row| input[column * input_ld + row] = if (alpha == 0) std.math.nan(T) else @as(T, @floatFromInt(@as(i32, @intCast((column * 7 + row * 3) % 17)) - 8)) / 4;
                                for (0..output_rows) |row| actual[column * output_ld + row] = if (beta == 0) std.math.nan(T) else @floatFromInt((column + 1) * (row + 2));
                            }
                            var expected = actual;
                            for (0..rhs) |column| {
                                for (0..output_rows) |row| {
                                    const destination = &expected[column * output_ld + row];
                                    const old: T = if (beta == 0) 0 else beta * destination.*;
                                    if (alpha == 0) {
                                        destination.* = old;
                                        continue;
                                    }
                                    var sum: T = 0;
                                    for (0..input_rows) |inner| {
                                        const coefficient = if (transpose) dense[inner][row] else dense[row][inner];
                                        sum += coefficient * input[column * input_ld + inner];
                                    }
                                    destination.* = old + alpha * sum;
                                }
                            }
                            const right = try sparse.constMatrix(T, &input, .{ .row_count = input_rows, .column_count = rhs, .leading_dimension = input_ld });
                            const result = try sparse.matrix(T, &actual, .{ .row_count = output_rows, .column_count = rhs, .leading_dimension = output_ld });
                            try sparse.matrixMultiply(.{ .left_matrix = a, .right_matrix = right, .result_matrix = result, .product_scale = alpha, .result_scale = beta });
                            try expectBitsOrNan(T, &expected, &actual);
                        }
                    }
                }
            }
        }
    }
}

test "two RHS blocks and tails retain duplicate storage-order accumulation" {
    inline for (.{ f32, f64 }) |T| {
        const large: T = if (T == f32) 16777216 else 9007199254740992;
        const values = [_]T{ large, -large, 1 };
        const a = try sparse.constCsrMatrix(T, u32, &.{ 0, 3 }, &.{ 0, 0, 0 }, &values, .{ .row_count = 1, .column_count = 1 });
        const input: [9]T = @splat(1);
        for ([_]bool{ false, true }) |transpose| {
            for ([_]usize{ 1, 2, 3, 8, 9 }) |rhs| {
                var output: [9]T = @splat(std.math.nan(T));
                const right = try sparse.constMatrix(T, &input, .{ .row_count = 1, .column_count = rhs });
                const result = try sparse.matrix(T, &output, .{ .row_count = 1, .column_count = rhs });
                try sparse.matrixMultiply(.{ .left_matrix = if (transpose) a.transposed() else a, .right_matrix = right, .result_matrix = result });
                for (output[0..rhs]) |value| try std.testing.expectEqual(@as(T, 1), value);
                for (output[rhs..]) |value| try std.testing.expect(std.math.isNan(value));
            }
        }
    }
}

test "two RHS blocks reject late invalid indices before touching any output" {
    var offsets: [66]u32 = undefined;
    var indices: [65]u32 = undefined;
    const values: [65]f64 = @splat(1);
    for (&offsets, 0..) |*offset, index| offset.* = @intCast(index);
    @memset(&indices, 0);
    const a = try sparse.constCsrMatrix(f64, u32, &offsets, &indices, &values, .{ .row_count = 65, .column_count = 1 });
    const input: [8]f64 = @splat(std.math.nan(f64));
    var output: [520]f64 = @splat(77);
    const right = try sparse.constMatrix(f64, &input, .{ .row_count = 1, .column_count = 8 });
    const result = try sparse.matrix(f64, &output, .{ .row_count = 65, .column_count = 8 });
    for (0..65) |bad_position| {
        indices[bad_position] = 1;
        try std.testing.expectError(error.InvalidColumnIndex, sparse.matrixMultiply(.{ .left_matrix = a, .right_matrix = right, .result_matrix = result, .product_scale = 0 }));
        indices[bad_position] = 0;
    }
    for (output) |value| try std.testing.expectEqual(@as(f64, 77), value);
}

test "two RHS blocks retain empty-product semantics and zero output dimensions" {
    var empty: [0]f64 = .{};
    const a = try sparse.constCsrMatrix(f64, i32, &.{ 0, 0, 0 }, &.{}, &.{}, .{ .row_count = 2, .column_count = 0 });
    const right = try sparse.constMatrix(f64, &.{}, .{ .row_count = 0, .column_count = 17, .leading_dimension = std.math.maxInt(usize) });
    var output: [51]f64 = @splat(12345);
    for (0..17) |column| {
        output[column * 3] = std.math.nan(f64);
        output[column * 3 + 1] = std.math.nan(f64);
    }
    const result = try sparse.matrix(f64, &output, .{ .row_count = 2, .column_count = 17, .leading_dimension = 3 });
    try sparse.matrixMultiply(.{ .left_matrix = a, .right_matrix = right, .result_matrix = result, .product_scale = std.math.inf(f64) });
    for (0..17) |column| {
        try std.testing.expectEqual(@as(f64, 0), output[column * 3]);
        try std.testing.expectEqual(@as(f64, 0), output[column * 3 + 1]);
        try std.testing.expectEqual(@as(f64, 12345), output[column * 3 + 2]);
    }
    const zero = try sparse.constCsrMatrix(f64, i32, &.{0}, &.{}, &.{}, .{ .row_count = 0, .column_count = 0 });
    const zero_output = try sparse.matrix(f64, &empty, .{ .row_count = 0, .column_count = 17, .leading_dimension = std.math.maxInt(usize) });
    try sparse.matrixMultiply(.{ .left_matrix = zero, .right_matrix = right, .result_matrix = zero_output });
}
