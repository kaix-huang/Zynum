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

test "mixed-index complex SpMV and paired SpMM match independent dense real imaginary products" {
    @setEvalBranchQuota(10000);
    inline for (.{ sparse.ComplexF32, sparse.ComplexF64 }) |T| {
        const R = Real(T);
        inline for (.{ .{ i64, i32 }, .{ u64, u32 }, .{ i32, i64 }, .{ u32, u64 } }) |types| {
            const O = types[0];
            const I = types[1];
            const offsets = [_]O{ 0, 3, 3, 5 };
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
                            const a = try sparse.constCsrMatrixWithIndexTypes(T, O, I, &offsets, &columns, &values, .{ .row_count = 3, .column_count = 2, .operation = operation });
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

test "mixed CSR rejects invalid metadata before changing output even with zero alpha" {
    inline for (.{ .{ i64, i32 }, .{ u64, u32 }, .{ i32, i64 }, .{ u32, u64 } }) |types| {
        const O = types[0];
        const I = types[1];
        try std.testing.expect(sparse.ConstCsrMatrix(f64, I) == sparse.ConstCsrMatrixWithIndexTypes(f64, I, I));
        var offsets = [_]O{ 0, 1, 2 };
        var columns = [_]I{ 0, 1 };
        const values = [_]f64{ 2, 3 };
        const a = try sparse.constCsrMatrixWithIndexTypes(f64, O, I, &offsets, &columns, &values, .{ .row_count = 2, .column_count = 2 });
        const input = [_]f64{ 5, 7 };
        var output = [_]f64{ 91, 92 };
        offsets[1] = std.math.maxInt(O);
        try std.testing.expectError(error.InvalidRowOffsets, sparse.matrixVectorMultiply(.{ .matrix = a, .input_vector = &input, .result_vector = &output, .product_scale = @as(f64, 0) }));
        try std.testing.expectEqualSlices(f64, &.{ 91, 92 }, &output);
        offsets[1] = 1;
        columns[1] = std.math.maxInt(I);
        try std.testing.expectError(error.InvalidColumnIndex, sparse.matrixVectorMultiply(.{ .matrix = a, .input_vector = &input, .result_vector = &output }));
        try std.testing.expectEqualSlices(f64, &.{ 91, 92 }, &output);
        columns[1] = 1;
        if (@typeInfo(O).int.signedness == .signed) {
            offsets[1] = -1;
            try std.testing.expectError(error.InvalidRowOffsets, a.check());
            offsets[1] = 1;
        }
        var huge = a;
        huge.row_count = std.math.maxInt(usize);
        try std.testing.expectError(error.SizeOverflow, huge.check());
    }
}

test "mixed CSR alias preflight uses actual offset and ordinal byte widths" {
    var offsets = [_]i64{ 0, 2, 4, 99 };
    var columns: [4]i32 align(8) = .{ 0, 1, 0, 1 };
    const values = [_]f64{ 1, 2, 3, 4 };
    const a = try sparse.constCsrMatrixWithIndexTypes(f64, i64, i32, offsets[0..3], &columns, &values, .{ .row_count = 2, .column_count = 2 });
    const input = [_]f64{ 5, 7 };
    const old_offsets = offsets;
    const old_columns = columns;
    const offset_output: [*]f64 = @ptrCast(&offsets[2]);
    const column_output: [*]f64 = @ptrCast(&columns);
    for ([_][]f64{ offset_output[0..2], column_output[0..2] }) |output| {
        try std.testing.expectError(error.AliasingNotAllowed, sparse.matrixVectorMultiply(.{ .matrix = a, .input_vector = &input, .result_vector = output }));
        const right = try sparse.constMatrix(f64, &input, .{ .row_count = 2, .column_count = 1, .leading_dimension = 2 });
        const result = try sparse.matrix(f64, output, .{ .row_count = 2, .column_count = 1, .leading_dimension = 2 });
        try std.testing.expectError(error.AliasingNotAllowed, sparse.matrixMultiply(.{ .left_matrix = a, .right_matrix = right, .result_matrix = result }));
        try std.testing.expectEqualSlices(i64, &old_offsets, &offsets);
        try std.testing.expectEqualSlices(i32, &old_columns, &columns);
    }
}

test "mixed SpMM borrows one admission and finishes while context closes" {
    const Core = @import("zynum-core");
    const execution = @import("zynum-execution");
    var ctx = execution.context.Context.init(std.testing.allocator, execution.test_support.policy, .internal_closable);
    var operation = try ctx.begin();
    try std.testing.expectEqual(execution.context.RequestResult.pending, ctx.requestClose());
    const offsets = [_]i64{ 0, 1, 2 };
    const indices = [_]i32{ 0, 1 };
    const values = [_]f64{ 2, 3 };
    const a = try sparse.constCsrMatrixWithIndexTypes(f64, i64, i32, &offsets, &indices, &values, .{ .row_count = 2, .column_count = 2 });
    const input = [_]f64{ 5, 7 };
    var output = [_]f64{ 91, 92 };
    const right = try sparse.constMatrix(f64, &input, .{ .row_count = 2, .column_count = 1, .leading_dimension = 2 });
    const result = try sparse.matrix(f64, &output, .{ .row_count = 2, .column_count = 1, .leading_dimension = 2 });
    try Core.internal.sparse_checked.matrixMultiplyAdmitted(&operation, .{ .left_matrix = a, .right_matrix = right, .result_matrix = result });
    try std.testing.expectEqualSlices(f64, &.{ 10, 21 }, &output);
    try std.testing.expectEqual(@as(usize, 1), ctx.snapshot().active_operations);
    try std.testing.expectEqual(@as(usize, 0), ctx.snapshot().resource_pins);
    try std.testing.expectError(error.Closing, ctx.begin());
    operation.end();
    try std.testing.expectEqual(execution.context.CloseResult.closed, ctx.tryProgressClose());
}
