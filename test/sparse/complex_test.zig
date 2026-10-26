// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
const std = @import("std");
const sparse = @import("zynum-sparse");

fn Real(comptime T: type) type {
    return if (T == sparse.ComplexF32) f32 else f64;
}
fn z(comptime T: type, re: Real(T), im: Real(T)) T {
    return .{ .re = re, .im = im };
}
fn same(comptime T: type, expected: []const T, actual: []const T) !void {
    try std.testing.expectEqualSlices(u8, std.mem.sliceAsBytes(expected), std.mem.sliceAsBytes(actual));
}
fn near(comptime T: type, expected: T, actual: T) !void {
    const tolerance: Real(T) = if (T == sparse.ComplexF32) 0.00001 else 1e-12;
    try std.testing.expectApproxEqAbs(expected.re, actual.re, tolerance);
    try std.testing.expectApproxEqAbs(expected.im, actual.im, tolerance);
}

test "complex SpMV and paired SpMM match independent dense real imaginary products" {
    @setEvalBranchQuota(10000);
    inline for (.{ sparse.ComplexF32, sparse.ComplexF64 }) |T| {
        const R = Real(T);
        inline for (.{ i32, u32, i64, u64 }) |I| {
            const offsets = [_]I{ 0, 3, 3, 5 };
            const columns = [_]I{ 1, 0, 1, 0, 1 };
            const initial = [_]T{ z(T, 2, 1), z(T, 1, -2), z(T, -0.5, 0.5), z(T, 0, 3), z(T, -1, 1) };
            // Literal row-major dense matrix, independent of CSR construction.
            const ar = [_]R{ 1, 1.5, 0, 0, 0, -1 };
            const ai = [_]R{ -2, 1.5, 0, 0, 3, 1 };
            for ([_]sparse.MatrixTransform{ .normal, .transposed, .adjoint }) |operation| {
                const input_rows: usize = if (operation == .normal) 2 else 3;
                const output_rows: usize = if (operation == .normal) 3 else 2;
                const ldx = input_rows + 2;
                const ldy = output_rows + 3;
                for ([_]usize{ 0, 1, 2, 3, 7, 8, 9 }) |rhs_count| {
                    for ([_]T{ z(T, 0, 0), z(T, -@as(R, 0), 0), z(T, 1, 0), z(T, 0.5, -1) }) |alpha| {
                        for ([_]T{ z(T, 0, 0), z(T, 0, -@as(R, 0)), z(T, 1, 0), z(T, -0.5, 0.25) }) |beta| {
                            const zero_alpha = alpha.re == 0 and alpha.im == 0;
                            const zero_beta = beta.re == 0 and beta.im == 0;
                            var values = initial;
                            if (zero_alpha) @memset(&values, z(T, std.math.nan(R), std.math.nan(R)));
                            const a = try sparse.constCsrMatrix(T, I, &offsets, &columns, &values, .{ .row_count = 3, .column_count = 2, .operation = operation });
                            var input: [128]T = @splat(z(T, 123, -77));
                            var output: [128]T = @splat(z(T, 123, -77));
                            for (0..rhs_count) |column| {
                                for (0..input_rows) |row| input[column * ldx + row] = if (zero_alpha) z(T, std.math.nan(R), std.math.nan(R)) else z(T, @as(R, @floatFromInt(@as(i32, @intCast((column * 7 + row) % 9)) - 4)) / 4, @as(R, @floatFromInt(@as(i32, @intCast((column + row * 3) % 7)) - 3)) / 4);
                                for (0..output_rows) |row| output[column * ldy + row] = if (zero_beta) z(T, std.math.nan(R), std.math.nan(R)) else z(T, @as(R, @floatFromInt(column + row + 1)) / 4, -@as(R, @floatFromInt(column + row + 2)) / 4);
                            }
                            const before = output;
                            const input_before = input;
                            var expected = output;
                            for (0..rhs_count) |column| {
                                for (0..output_rows) |row| {
                                    const old = before[column * ldy + row];
                                    const prior_re: R = if (zero_beta) 0 else beta.re * old.re - beta.im * old.im;
                                    const prior_im: R = if (zero_beta) 0 else beta.re * old.im + beta.im * old.re;
                                    var rr: R = 0;
                                    var ii: R = 0;
                                    var ri: R = 0;
                                    var ir: R = 0;
                                    if (!zero_alpha) {
                                        for (0..input_rows) |inner| {
                                            const coordinate = if (operation == .normal) row * 2 + inner else inner * 2 + row;
                                            const real = ar[coordinate];
                                            const imag = if (operation == .adjoint) -ai[coordinate] else ai[coordinate];
                                            const x = input[column * ldx + inner];
                                            rr += real * x.re;
                                            ii += imag * x.im;
                                            ri += real * x.im;
                                            ir += imag * x.re;
                                        }
                                    }
                                    const product_re = rr - ii;
                                    const product_im = ri + ir;
                                    expected[column * ldy + row] = if (zero_alpha) z(T, prior_re, prior_im) else z(T, prior_re + alpha.re * product_re - alpha.im * product_im, prior_im + alpha.re * product_im + alpha.im * product_re);
                                }
                            }
                            const right = try sparse.constMatrix(T, &input, .{ .row_count = input_rows, .column_count = rhs_count, .leading_dimension = ldx });
                            const result = try sparse.matrix(T, &output, .{ .row_count = output_rows, .column_count = rhs_count, .leading_dimension = ldy });
                            try sparse.matrixMultiply(.{ .left_matrix = a, .right_matrix = right, .result_matrix = result, .product_scale = alpha, .result_scale = beta });
                            for (0..rhs_count) |column| for (0..output_rows) |row| try near(T, expected[column * ldy + row], output[column * ldy + row]);
                            for (output, 0..) |_, index| {
                                if (index / ldy >= rhs_count or index % ldy >= output_rows) try same(T, before[index..][0..1], output[index..][0..1]);
                            }
                            try same(T, &input_before, &input);
                            if (rhs_count == 1) {
                                var vector = before;
                                try sparse.matrixVectorMultiply(.{ .matrix = a, .input_vector = input[0..input_rows], .result_vector = vector[0..output_rows], .product_scale = alpha, .result_scale = beta });
                                for (0..output_rows) |row| try near(T, expected[row], vector[row]);
                            }
                        }
                    }
                }
            }
        }
    }
}

test "adjoint conjugates only matrix values and preserves defined signed zero behavior" {
    inline for (.{ sparse.ComplexF32, sparse.ComplexF64 }) |T| {
        const R = Real(T);
        const a = try sparse.constCsrMatrix(T, u32, &.{ 0, 1 }, &.{0}, &.{z(T, 2, 3)}, .{ .row_count = 1, .column_count = 1 });
        var y = [_]T{z(T, 0, 0)};
        try sparse.matrixVectorMultiply(.{ .matrix = a.transposed(), .input_vector = &[_]T{z(T, 4, -5)}, .result_vector = &y });
        try near(T, z(T, 23, 2), y[0]);
        try sparse.matrixVectorMultiply(.{ .matrix = a.adjoint(), .input_vector = &[_]T{z(T, 4, -5)}, .result_vector = &y });
        try near(T, z(T, -7, -22), y[0]);
        try std.testing.expectEqual(sparse.MatrixTransform.adjoint, a.transposed().adjoint().operation);
        try std.testing.expectEqual(sparse.MatrixTransform.transposed, a.adjoint().transposed().operation);
        const signed = try sparse.constCsrMatrix(T, i64, &.{ 0, 1 }, &.{0}, &.{z(T, -1, 0)}, .{ .row_count = 1, .column_count = 1 });
        for ([_]bool{ false, true }) |adjoint| {
            y[0] = z(T, 0, -@as(R, 0));
            try sparse.matrixVectorMultiply(.{ .matrix = if (adjoint) signed.adjoint() else signed.transposed(), .input_vector = &[_]T{z(T, 1, 0)}, .result_vector = &y, .result_scale = z(T, 1, 0) });
            const expected = [_]T{z(T, -1, if (adjoint) -@as(R, 0) else 0)};
            try same(T, &expected, &y);
            const input: [3]T = @splat(z(T, 1, 0));
            var output: [3]T = @splat(z(T, 0, -@as(R, 0)));
            const right = try sparse.constMatrix(T, &input, .{ .row_count = 1, .column_count = 3 });
            const result = try sparse.matrix(T, &output, .{ .row_count = 1, .column_count = 3 });
            try sparse.matrixMultiply(.{ .left_matrix = if (adjoint) signed.adjoint() else signed.transposed(), .right_matrix = right, .result_matrix = result, .result_scale = z(T, 1, 0) });
            for (0..3) |column| try same(T, &expected, output[column..][0..1]);
        }
    }
}

test "complex duplicate multiplication retains stored order for vector blocks and tails" {
    inline for (.{ sparse.ComplexF32, sparse.ComplexF64 }) |T| {
        const R = Real(T);
        const large: R = if (R == f32) 16777216 else 9007199254740992;
        const values = [_]T{ z(T, large, 0), z(T, -large, 0), z(T, 1, 0) };
        const a = try sparse.constCsrMatrix(T, u64, &.{ 0, 3 }, &.{ 0, 0, 0 }, &values, .{ .row_count = 1, .column_count = 1 });
        const input: [9]T = @splat(z(T, 1, 0));
        for ([_]sparse.MatrixTransform{ .normal, .transposed, .adjoint }) |operation| {
            var transformed = a;
            transformed.operation = operation;
            var vector = [_]T{z(T, 0, 0)};
            try sparse.matrixVectorMultiply(.{ .matrix = transformed, .input_vector = input[0..1], .result_vector = &vector });
            try near(T, z(T, 1, 0), vector[0]);
            for ([_]usize{ 1, 2, 3, 8, 9 }) |columns| {
                var output: [9]T = @splat(z(T, std.math.nan(R), std.math.nan(R)));
                const right = try sparse.constMatrix(T, &input, .{ .row_count = 1, .column_count = columns });
                const result = try sparse.matrix(T, &output, .{ .row_count = 1, .column_count = columns });
                try sparse.matrixMultiply(.{ .left_matrix = transformed, .right_matrix = right, .result_matrix = result });
                for (output[0..columns]) |value| try near(T, z(T, 1, 0), value);
                for (output[columns..]) |value| try std.testing.expect(std.math.isNan(value.re) and std.math.isNan(value.im));
            }
        }
    }
}

test "complex CSC and COO conversion preserve component bits and duplicate value order" {
    inline for (.{ sparse.ComplexF32, sparse.ComplexF64 }) |T| {
        const R = Real(T);
        const U = if (R == f32) u32 else u64;
        const nan_bits: U = if (R == f32) 0x7fc00001 else 0x7ff8000000000001;
        const values = [_]T{ z(T, @bitCast(nan_bits), -@as(R, 0)), z(T, 0, @bitCast(nan_bits + 7)), z(T, -@as(R, 0), 0), z(T, 3, -4), z(T, @bitCast(nan_bits + 13), std.math.inf(R)), z(T, 2, -@as(R, 0)) };
        const expected = [_]T{ values[4], values[1], values[2], values[5], values[0], values[3] };
        inline for (.{ i32, u32, i64, u64 }) |I| {
            const coo = try sparse.constCooMatrix(T, I, &.{ 2, 0, 1, 2, 0, 2 }, &.{ 1, 1, 0, 1, 0, 0 }, &values, .{ .row_count = 3, .column_count = 2 });
            const csc_values = [_]T{ values[2], values[4], values[5], values[0], values[1], values[3] };
            const csc = try sparse.constCscMatrix(T, I, &.{ 0, 3, 6 }, &.{ 1, 0, 2, 2, 0, 2 }, &csc_values, .{ .row_count = 3, .column_count = 2 });
            for ([_]bool{ false, true }) |use_csc| {
                var offsets: [5]I = @splat(99);
                var columns: [7]I = @splat(99);
                var output: [7]T = @splat(z(T, 11, -12));
                var workspace: [272]usize = @splat(99);
                const result = if (use_csc)
                    try sparse.cscToCsr(.{ .matrix = csc, .row_offsets = &offsets, .column_indices = &columns, .values = &output, .workspace = &workspace })
                else
                    try sparse.cooToCsr(.{ .matrix = coo, .row_offsets = &offsets, .column_indices = &columns, .values = &output, .workspace = &workspace });
                try result.check();
                try std.testing.expectEqual(sparse.MatrixTransform.normal, result.operation);
                try std.testing.expectEqualSlices(I, &.{ 0, 2, 3, 6 }, result.row_offsets);
                try std.testing.expectEqualSlices(I, &.{ 0, 1, 0, 0, 1, 1 }, result.column_indices);
                try same(T, &expected, result.values);
                try same(T, &[_]T{z(T, 11, -12)}, output[6..]);
                try std.testing.expectEqual(@as(I, 99), offsets[4]);
                try std.testing.expectEqual(@as(I, 99), columns[6]);
                const active: usize = if (use_csc) 3 else 271;
                for (workspace[active..]) |value| try std.testing.expectEqual(@as(usize, 99), value);
            }
        }
    }
}

test "complex empty products and zero scales preserve unused NaNs and padding semantics" {
    inline for (.{ sparse.ComplexF32, sparse.ComplexF64 }) |T| {
        const R = Real(T);
        const empty = try sparse.constCsrMatrix(T, u32, &.{ 0, 0, 0 }, &.{}, &.{}, .{ .row_count = 2, .column_count = 0 });
        var vector = [_]T{ z(T, 1, 2), z(T, -3, 4) };
        try sparse.matrixVectorMultiply(.{ .matrix = empty, .input_vector = &[_]T{}, .result_vector = &vector, .product_scale = z(T, std.math.nan(R), std.math.inf(R)), .result_scale = z(T, 2, -1) });
        try near(T, z(T, 4, 3), vector[0]);
        try near(T, z(T, -2, 11), vector[1]);
        const right = try sparse.constMatrix(T, &.{}, .{ .row_count = 0, .column_count = 3, .leading_dimension = std.math.maxInt(usize) });
        var output: [9]T = @splat(z(T, std.math.nan(R), std.math.nan(R)));
        for (0..3) |column| output[column * 3 + 2] = z(T, 7, -9);
        const result = try sparse.matrix(T, &output, .{ .row_count = 2, .column_count = 3, .leading_dimension = 3 });
        try sparse.matrixMultiply(.{ .left_matrix = empty, .right_matrix = right, .result_matrix = result, .product_scale = z(T, std.math.inf(R), std.math.nan(R)), .result_scale = z(T, -@as(R, 0), -@as(R, 0)) });
        for (0..3) |column| {
            try same(T, &[_]T{ z(T, 0, 0), z(T, 0, 0), z(T, 7, -9) }, output[column * 3 ..][0..3]);
        }
        const a = try sparse.constCsrMatrix(T, i32, &.{ 0, 1 }, &.{0}, &.{z(T, std.math.nan(R), std.math.nan(R))}, .{ .row_count = 1, .column_count = 1 });
        var y = [_]T{z(T, -@as(R, 0), -@as(R, 0))};
        const before = y;
        try sparse.matrixVectorMultiply(.{ .matrix = a.adjoint(), .input_vector = &[_]T{z(T, std.math.nan(R), std.math.nan(R))}, .result_vector = &y, .product_scale = z(T, -@as(R, 0), 0), .result_scale = z(T, 1, -@as(R, 0)) });
        try same(T, &before, &y);
        const coo = try sparse.constCooMatrix(T, u64, &.{}, &.{}, &.{}, .{ .row_count = 0, .column_count = 3 });
        var offsets = [_]u64{99};
        var no_indices: [0]u64 = .{};
        var no_values: [0]T = .{};
        var no_workspace: [0]usize = .{};
        const converted = try sparse.cooToCsr(.{ .matrix = coo, .row_offsets = &offsets, .column_indices = &no_indices, .values = &no_values, .workspace = &no_workspace });
        try converted.check();
        try std.testing.expectEqual(@as(u64, 0), offsets[0]);
        const zero_result = try sparse.matrix(T, &no_values, .{ .row_count = 0, .column_count = 3, .leading_dimension = std.math.maxInt(usize) });
        const normal_input = try sparse.constMatrix(T, &output, .{ .row_count = 3, .column_count = 3 });
        try sparse.matrixMultiply(.{ .left_matrix = converted, .right_matrix = normal_input, .result_matrix = zero_result });
    }
}

test "complex invalid structure dimensions and aliases fail before mutation even at zero alpha" {
    inline for (.{ sparse.ComplexF32, sparse.ComplexF64 }) |T| {
        var offsets: [8]u64 align(@max(@alignOf(T), @alignOf(u64))) = .{ 0, 1, 2, 99, 99, 99, 99, 99 };
        var indices: [8]u64 align(@max(@alignOf(T), @alignOf(u64))) = .{ 0, 1, 99, 99, 99, 99, 99, 99 };
        var values = [_]T{ z(T, 1, 0), z(T, 1, 0) };
        const a = try sparse.constCsrMatrix(T, u64, offsets[0..3], indices[0..2], &values, .{ .row_count = 2, .column_count = 2 });
        var input: [8]T = @splat(z(T, 2, 3));
        var output: [8]T = @splat(z(T, 7, -4));
        const before = output;
        indices[1] = 2;
        try std.testing.expectError(error.InvalidColumnIndex, sparse.matrixVectorMultiply(.{ .matrix = a.adjoint(), .input_vector = input[0..2], .result_vector = output[0..2], .product_scale = z(T, 0, 0) }));
        indices[1] = 1;
        offsets[1] = 3;
        try std.testing.expectError(error.InvalidRowOffsets, sparse.matrixVectorMultiply(.{ .matrix = a, .input_vector = input[0..2], .result_vector = output[0..2] }));
        offsets[1] = 1;
        try std.testing.expectError(error.DimensionMismatch, sparse.matrixVectorMultiply(.{ .matrix = a, .input_vector = input[0..1], .result_vector = output[0..2] }));
        const metadata_output = [_][]T{
            std.mem.bytesAsSlice(T, std.mem.sliceAsBytes(&indices))[0..2],
            std.mem.bytesAsSlice(T, std.mem.sliceAsBytes(&offsets))[0..2],
            &values,
            input[0..2],
            input[1..3],
        };
        for (metadata_output) |aliased| try std.testing.expectError(error.AliasingNotAllowed, sparse.matrixVectorMultiply(.{ .matrix = a, .input_vector = input[0..2], .result_vector = aliased, .product_scale = z(T, 0, 0) }));
        const right = try sparse.constMatrix(T, input[0..5], .{ .row_count = 2, .column_count = 2, .leading_dimension = 3 });
        const aliased_result = try sparse.matrix(T, input[2..7], .{ .row_count = 2, .column_count = 2, .leading_dimension = 3 });
        try std.testing.expectError(error.AliasingNotAllowed, sparse.matrixMultiply(.{ .left_matrix = a.adjoint(), .right_matrix = right, .result_matrix = aliased_result, .product_scale = z(T, 0, 0) }));
        try same(T, &before, &output);
        for (input) |value| try near(T, z(T, 2, 3), value);
        try std.testing.expectEqualSlices(u64, &.{ 0, 1, 2, 99, 99, 99, 99, 99 }, &offsets);
        try std.testing.expectEqualSlices(u64, &.{ 0, 1, 99, 99, 99, 99, 99, 99 }, &indices);
    }
}

test "complex conversion capacities workspace and mixed metadata aliases preflight before writes" {
    inline for (.{ sparse.ComplexF32, sparse.ComplexF64 }) |T| {
        var source_values: [2]T align(@alignOf(usize)) = .{ z(T, 1, 2), z(T, 3, 4) };
        const a = try sparse.constCscMatrix(T, u64, &.{ 0, 1, 2 }, &.{ 1, 0 }, &source_values, .{ .row_count = 2, .column_count = 2 });
        var offsets: [3]u64 = @splat(99);
        var indices: [2]u64 = @splat(99);
        var output: [3]T align(@alignOf(u64)) = @splat(z(T, 7, -4));
        var workspace: [2]usize = @splat(99);
        const before = output;
        const args = .{ .matrix = a, .row_offsets = @as([]u64, &offsets), .column_indices = @as([]u64, &indices), .values = @as([]T, &output), .workspace = @as([]usize, &workspace) };
        var bad = args;
        bad.workspace = workspace[0..1];
        try std.testing.expectError(error.WorkspaceTooSmall, sparse.cscToCsr(bad));
        bad = args;
        bad.values = output[0..1];
        try std.testing.expectError(error.BufferTooSmall, sparse.cscToCsr(bad));
        bad = args;
        bad.values = &source_values;
        try std.testing.expectError(error.AliasingNotAllowed, sparse.cscToCsr(bad));
        bad = args;
        bad.column_indices = std.mem.bytesAsSlice(u64, std.mem.sliceAsBytes(&output))[0..2];
        try std.testing.expectError(error.AliasingNotAllowed, sparse.cscToCsr(bad));
        bad = args;
        bad.workspace = std.mem.bytesAsSlice(usize, std.mem.sliceAsBytes(&source_values))[0..2];
        try std.testing.expectError(error.AliasingNotAllowed, sparse.cscToCsr(bad));
        try same(T, &before, &output);
        try same(T, &[_]T{ z(T, 1, 2), z(T, 3, 4) }, &source_values);
        try std.testing.expectEqualSlices(u64, &.{ 99, 99, 99 }, &offsets);
        try std.testing.expectEqualSlices(u64, &.{ 99, 99 }, &indices);
        try std.testing.expectEqualSlices(usize, &.{ 99, 99 }, &workspace);
    }
}
