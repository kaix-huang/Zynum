// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later

const std = @import("std");
const sparse = @import("zynum-sparse");

fn near(comptime T: type, expected: T, actual: T) !void {
    const tolerance: T = if (T == f32) 0.0001 else 0.00000000001;
    try std.testing.expectApproxEqAbs(expected, actual, tolerance * @max(1, @abs(expected)));
}

fn success(result: sparse.TriangularSolveResult, columns: usize) !void {
    try std.testing.expectEqualDeep(sparse.TriangularSolveResult{ .completed_rhs = columns }, result);
}

fn finiteSelection(row: usize, column: usize, triangle: sparse.Triangle, diagonal: sparse.Diagonal) bool {
    if (row == column) return diagonal == .non_unit;
    return if (triangle == .lower) column < row else column > row;
}

test "triangular solve matches independent dense products across types flags and padded RHS" {
    inline for (.{ f32, f64 }) |T| {
        inline for (.{ i32, u32, i64, u64 }) |I| {
            const dense = [_][4]T{
                .{ 4, 0.5, -1, 2 },
                .{ 1, 5, 0.5, -1 },
                .{ -0.5, 2, 6, 0.25 },
                .{ 0.75, -1, 1, 7 },
            };
            const offsets = [_]I{ 0, 6, 12, 18, 24 };
            var columns: [24]I = undefined;
            var values: [24]T = undefined;
            inline for (.{ sparse.Triangle.lower, sparse.Triangle.upper }) |triangle| {
                inline for (.{ sparse.Diagonal.non_unit, sparse.Diagonal.unit }) |diagonal| {
                    for (0..4) |row| {
                        const duplicate = (row + 1) % 4;
                        for ([_]usize{ 3, 1, 0, 2, row, duplicate }, 0..) |column, position| {
                            const part: T = if (column == row) 0.5 else if (column == duplicate) (if (position == 5) 0.25 else 0.75) else 1;
                            columns[row * 6 + position] = @intCast(column);
                            values[row * 6 + position] = if (finiteSelection(row, column, triangle, diagonal)) dense[row][column] * part else std.math.nan(T);
                        }
                    }
                    const original = try sparse.constCsrMatrix(T, I, &offsets, &columns, &values, .{ .row_count = 4, .column_count = 4 });
                    for ([_]sparse.MatrixTransform{ .normal, .transposed, .adjoint }) |operation| {
                        const transpose = operation != .normal;
                        var transformed = original;
                        transformed.operation = operation;
                        for ([_]usize{ 1, 2, 3, 8 }) |nrhs| {
                            var b: [50]T = @splat(-12345);
                            var expected: [32]T = undefined;
                            for (0..nrhs) |column| {
                                for (0..4) |row| expected[column * 4 + row] = @as(T, @floatFromInt((row + 1) * (column + 1))) / 4;
                                for (0..4) |row| {
                                    var sum: T = 0;
                                    for (0..4) |inner| {
                                        const source_row = if (transpose) inner else row;
                                        const source_column = if (transpose) row else inner;
                                        const coefficient: T = if (source_row == source_column and diagonal == .unit) 1 else if (finiteSelection(source_row, source_column, triangle, diagonal)) dense[source_row][source_column] else 0;
                                        sum += coefficient * expected[column * 4 + inner];
                                    }
                                    b[column * 6 + row] = sum;
                                }
                            }
                            const original_b = b;
                            const rhs = try sparse.matrix(T, &b, .{ .row_count = 4, .column_count = nrhs, .leading_dimension = 6 });
                            var workspace: [6]T = @splat(99);
                            const required = try sparse.triangularSolveWorkspaceLength(T, 4, nrhs, diagonal);
                            try std.testing.expectEqual(@as(usize, if (diagonal == .unit) 0 else 4), required);
                            const result = try sparse.triangularSolveInPlace(.{ .matrix = transformed, .triangle = triangle, .diagonal = diagonal, .rhs = rhs, .workspace = &workspace });
                            try success(result, nrhs);
                            for (0..nrhs) |column| {
                                for (0..4) |row| {
                                    try near(T, expected[column * 4 + row], b[column * 6 + row]);
                                    var product: T = 0;
                                    for (0..4) |inner| {
                                        const source_row = if (transpose) inner else row;
                                        const source_column = if (transpose) row else inner;
                                        const coefficient: T = if (source_row == source_column and diagonal == .unit) 1 else if (finiteSelection(source_row, source_column, triangle, diagonal)) dense[source_row][source_column] else 0;
                                        product += coefficient * b[column * 6 + inner];
                                    }
                                    try near(T, original_b[column * 6 + row], product);
                                }
                            }
                            for (b, 0..) |value, index| {
                                if (index / 6 >= nrhs or index % 6 >= 4) try std.testing.expectEqual(@as(T, -12345), value);
                            }
                            for (workspace, 0..) |value, row| {
                                try std.testing.expectEqual(if (row < required) dense[row][row] else @as(T, 99), value);
                            }
                        }
                    }
                }
            }
        }
    }
}

test "duplicate diagonal sum and strict triangle contributions retain source order" {
    inline for (.{ f32, f64 }) |T| {
        const large: T = if (T == f32) 16777216 else 9007199254740992;
        const a = try sparse.constCsrMatrix(T, u32, &.{ 0, 3 }, &.{ 0, 0, 0 }, &.{ large, -large, 2 }, .{ .row_count = 1, .column_count = 1 });
        var b = [_]T{4};
        var work = [_]T{99};
        const rhs = try sparse.matrix(T, &b, .{ .row_count = 1, .column_count = 1 });
        try success(try sparse.triangularSolveInPlace(.{ .matrix = a, .triangle = sparse.Triangle.lower, .diagonal = sparse.Diagonal.non_unit, .rhs = rhs, .workspace = &work }), 1);
        try std.testing.expectEqual(@as(T, 2), b[0]);
        try std.testing.expectEqual(@as(T, 2), work[0]);
        const lower = try sparse.constCsrMatrix(T, u32, &.{ 0, 0, 3 }, &.{ 0, 0, 0 }, &.{ large, -large, 1 }, .{ .row_count = 2, .column_count = 2 });
        var empty: [0]T = .{};
        for ([_]bool{ false, true }) |transpose| {
            var pair = if (transpose) [_]T{ 0, 1 } else [_]T{ 1, 0 };
            const output = try sparse.matrix(T, &pair, .{ .row_count = 2, .column_count = 1 });
            try success(try sparse.triangularSolveInPlace(.{ .matrix = if (transpose) lower.transposed() else lower, .triangle = sparse.Triangle.lower, .diagonal = sparse.Diagonal.unit, .rhs = output, .workspace = &empty }), 1);
            const expected = if (transpose) [_]T{ -1, 1 } else [_]T{ 1, -1 };
            try std.testing.expectEqualSlices(T, &expected, &pair);
        }
    }
}

test "missing zero and overflowing aggregate diagonals fail before all mutation" {
    inline for (.{ f32, f64 }) |T| {
        const missing = try sparse.constCsrMatrix(T, i64, &.{ 0, 1, 1, 2 }, &.{ 0, 2 }, &.{ 1, 2 }, .{ .row_count = 3, .column_count = 3 });
        var b: [6]T = @splat(7);
        var workspace: [3]T = @splat(99);
        const rhs = try sparse.matrix(T, &b, .{ .row_count = 3, .column_count = 2 });
        const result = try sparse.triangularSolveInPlace(.{ .matrix = missing, .triangle = sparse.Triangle.upper, .diagonal = sparse.Diagonal.non_unit, .rhs = rhs, .workspace = &workspace });
        try std.testing.expectEqualDeep(sparse.TriangularSolveResult{ .info = 2, .failure = .singular_diagonal }, result);
        for (b) |value| try std.testing.expectEqual(@as(T, 7), value);
        for (workspace) |value| try std.testing.expectEqual(@as(T, 99), value);
        const max = std.math.floatMax(T);
        for ([_]bool{ false, true }) |overflow| {
            const values = if (overflow) [_]T{ max, max } else [_]T{ 1, -1 };
            const a = try sparse.constCsrMatrix(T, i64, &.{ 0, 2 }, &.{ 0, 0 }, &values, .{ .row_count = 1, .column_count = 1 });
            var single = [_]T{4};
            const output = try sparse.matrix(T, &single, .{ .row_count = 1, .column_count = 1 });
            const status = try sparse.triangularSolveInPlace(.{ .matrix = a.transposed(), .triangle = sparse.Triangle.lower, .diagonal = sparse.Diagonal.non_unit, .rhs = output, .workspace = &workspace });
            try std.testing.expectEqualDeep(sparse.TriangularSolveResult{ .info = 1, .failure = if (overflow) .non_finite_diagonal else .singular_diagonal }, status);
            try std.testing.expectEqual(@as(T, 4), single[0]);
            for (workspace) |value| try std.testing.expectEqual(@as(T, 99), value);
        }
    }
}

test "empty rows are valid for implicit unit diagonals and singular otherwise" {
    const a = try sparse.constCsrMatrix(f64, i32, &.{ 0, 0, 0 }, &.{}, &.{}, .{ .row_count = 2, .column_count = 2 });
    var b = [_]f64{ 2, 3 };
    var workspace = [_]f64{ 99, 99 };
    const rhs = try sparse.matrix(f64, &b, .{ .row_count = 2, .column_count = 1 });
    try success(try sparse.triangularSolveInPlace(.{ .matrix = a, .triangle = sparse.Triangle.lower, .diagonal = sparse.Diagonal.unit, .rhs = rhs, .workspace = workspace[0..0] }), 1);
    try std.testing.expectEqualSlices(f64, &.{ 2, 3 }, &b);
    const result = try sparse.triangularSolveInPlace(.{ .matrix = a, .triangle = sparse.Triangle.lower, .diagonal = sparse.Diagonal.non_unit, .rhs = rhs, .workspace = &workspace });
    try std.testing.expectEqualDeep(sparse.TriangularSolveResult{ .info = 1, .failure = .singular_diagonal }, result);
    try std.testing.expectEqualSlices(f64, &.{ 2, 3 }, &b);
    try std.testing.expectEqualSlices(f64, &.{ 99, 99 }, &workspace);
}

test "used nonfinite A and RHS values reject before diagonal status or mutation" {
    var values = [_]f64{std.math.nan(f64)};
    const a = try sparse.constCsrMatrix(f64, u64, &.{ 0, 0, 1 }, &.{0}, &values, .{ .row_count = 2, .column_count = 2 });
    var b = [_]f64{ 2, 3, 4, 5 };
    var workspace = [_]f64{ 99, 99 };
    const rhs = try sparse.matrix(f64, &b, .{ .row_count = 2, .column_count = 2 });
    const args = .{ .matrix = a, .triangle = sparse.Triangle.lower, .diagonal = sparse.Diagonal.non_unit, .rhs = rhs, .workspace = &workspace };
    try std.testing.expectError(error.NonFiniteInput, sparse.triangularSolveInPlace(args));
    try std.testing.expectEqualSlices(f64, &.{ 2, 3, 4, 5 }, &b);
    values[0] = 1;
    b[3] = std.math.inf(f64);
    try std.testing.expectError(error.NonFiniteInput, sparse.triangularSolveInPlace(args));
    try std.testing.expectEqualSlices(f64, &.{ 2, 3, 4, std.math.inf(f64) }, &b);
    try std.testing.expectEqualSlices(f64, &.{ 99, 99 }, &workspace);
    // The diagonal is used in non-unit mode, even if a zero RHS could mask it.
    const nonfinite_diagonal = try sparse.constCsrMatrix(f64, u64, &.{ 0, 1 }, &.{0}, &.{std.math.inf(f64)}, .{ .row_count = 1, .column_count = 1 });
    var zero = [_]f64{0};
    const single = try sparse.matrix(f64, &zero, .{ .row_count = 1, .column_count = 1 });
    try std.testing.expectError(error.NonFiniteInput, sparse.triangularSolveInPlace(.{ .matrix = nonfinite_diagonal, .triangle = sparse.Triangle.lower, .diagonal = sparse.Diagonal.non_unit, .rhs = single, .workspace = &workspace }));
    try std.testing.expectEqual(@as(f64, 0), zero[0]);
}

test "zero RHS and zero size have no numerical reads but revalidate all structure" {
    var columns = [_]u32{0};
    const a = try sparse.constCsrMatrix(f64, u32, &.{ 0, 1, 1 }, &columns, &.{std.math.nan(f64)}, .{ .row_count = 2, .column_count = 2 });
    var storage = [_]f64{ std.math.nan(f64), 99 };
    const rhs = try sparse.matrix(f64, &storage, .{ .row_count = 2, .column_count = 0, .leading_dimension = std.math.maxInt(usize) });
    try std.testing.expectEqual(@as(usize, 0), try sparse.triangularSolveWorkspaceLength(f64, 2, 0, .non_unit));
    try success(try sparse.triangularSolveInPlace(.{ .matrix = a, .triangle = sparse.Triangle.lower, .diagonal = sparse.Diagonal.non_unit, .rhs = rhs, .workspace = storage[0..0] }), 0);
    try std.testing.expect(std.math.isNan(storage[0]));
    try std.testing.expectEqual(@as(f64, 99), storage[1]);
    columns[0] = 2;
    try std.testing.expectError(error.InvalidColumnIndex, sparse.triangularSolveInPlace(.{ .matrix = a, .triangle = sparse.Triangle.upper, .diagonal = sparse.Diagonal.unit, .rhs = rhs, .workspace = storage[0..0] }));
    const empty = try sparse.constCsrMatrix(f64, i32, &.{0}, &.{}, &.{}, .{ .row_count = 0, .column_count = 0 });
    const no_rows = try sparse.matrix(f64, &storage, .{ .row_count = 0, .column_count = 5, .leading_dimension = std.math.maxInt(usize) });
    try success(try sparse.triangularSolveInPlace(.{ .matrix = empty, .triangle = sparse.Triangle.lower, .diagonal = sparse.Diagonal.non_unit, .rhs = no_rows, .workspace = storage[0..0] }), 5);
}

fn unchanged(b: []const f64, workspace: []const f64) !void {
    for (b) |value| try std.testing.expectEqual(@as(f64, 7), value);
    for (workspace) |value| try std.testing.expectEqual(@as(f64, 99), value);
}

test "malformed mutable structure dimensions capacities and arithmetic reject before mutation" {
    var offsets = [_]i64{ 0, 1, 2 };
    var columns = [_]i64{ 0, 1 };
    const a = try sparse.constCsrMatrix(f64, i64, &offsets, &columns, &.{ 1, 1 }, .{ .row_count = 2, .column_count = 2 });
    var b: [5]f64 = @splat(7);
    var workspace: [2]f64 = @splat(99);
    const rhs = try sparse.matrix(f64, &b, .{ .row_count = 2, .column_count = 2, .leading_dimension = 3 });
    const args = .{ .matrix = a, .triangle = sparse.Triangle.lower, .diagonal = sparse.Diagonal.non_unit, .rhs = rhs, .workspace = @as([]f64, &workspace) };
    columns[1] = -1;
    try std.testing.expectError(error.InvalidColumnIndex, sparse.triangularSolveInPlace(args));
    columns[1] = 2;
    try std.testing.expectError(error.InvalidColumnIndex, sparse.triangularSolveInPlace(args));
    columns[1] = 1;
    offsets[1] = 3;
    try std.testing.expectError(error.InvalidRowOffsets, sparse.triangularSolveInPlace(args));
    offsets[1] = 1;
    var bad = args;
    bad.matrix.column_count = 3;
    try std.testing.expectError(error.DimensionMismatch, sparse.triangularSolveInPlace(bad));
    bad = args;
    bad.rhs.row_count = 1;
    try std.testing.expectError(error.DimensionMismatch, sparse.triangularSolveInPlace(bad));
    bad = args;
    bad.workspace = workspace[0..1];
    try std.testing.expectError(error.WorkspaceTooSmall, sparse.triangularSolveInPlace(bad));
    bad = args;
    bad.rhs.values = b[0..4];
    try std.testing.expectError(error.BufferTooSmall, sparse.triangularSolveInPlace(bad));
    bad = args;
    bad.rhs.leading_dimension = 1;
    try std.testing.expectError(error.InvalidLeadingDimension, sparse.triangularSolveInPlace(bad));
    bad = args;
    bad.rhs.leading_dimension = std.math.maxInt(usize);
    try std.testing.expectError(error.SizeOverflow, sparse.triangularSolveInPlace(bad));
    try std.testing.expectError(error.SizeOverflow, sparse.triangularSolveWorkspaceLength(f64, std.math.maxInt(usize), 1, .non_unit));
    try std.testing.expectEqual(@as(usize, 0), try sparse.triangularSolveWorkspaceLength(f64, std.math.maxInt(usize), 0, .non_unit));
    try unchanged(&b, &workspace);
}

test "active RHS workspace and sparse numerical or metadata aliases are rejected" {
    var offsets = [_]u64{ 0, 1, 2, 99, 99 };
    var columns = [_]u64{ 0, 1, 99, 99, 99 };
    var values = [_]f64{ 1, 1, 99, 99, 99 };
    const a = try sparse.constCsrMatrix(f64, u64, offsets[0..3], columns[0..2], values[0..2], .{ .row_count = 2, .column_count = 2 });
    var b: [5]f64 = @splat(7);
    var workspace: [2]f64 = @splat(99);
    const rhs = try sparse.matrix(f64, &b, .{ .row_count = 2, .column_count = 2, .leading_dimension = 3 });
    const args = .{ .matrix = a, .triangle = sparse.Triangle.lower, .diagonal = sparse.Diagonal.non_unit, .rhs = rhs, .workspace = @as([]f64, &workspace) };
    var bad = args;
    bad.workspace = b[0..2];
    try std.testing.expectError(error.AliasingNotAllowed, sparse.triangularSolveInPlace(bad));
    bad = args;
    // Includes padding at index 2 and the start of the second RHS.
    bad.workspace = b[2..4];
    try std.testing.expectError(error.AliasingNotAllowed, sparse.triangularSolveInPlace(bad));
    bad = args;
    bad.workspace = values[0..2];
    try std.testing.expectError(error.AliasingNotAllowed, sparse.triangularSolveInPlace(bad));
    bad = args;
    bad.workspace = std.mem.bytesAsSlice(f64, std.mem.sliceAsBytes(columns[0..2]));
    try std.testing.expectError(error.AliasingNotAllowed, sparse.triangularSolveInPlace(bad));
    bad = args;
    bad.workspace = std.mem.bytesAsSlice(f64, std.mem.sliceAsBytes(offsets[0..2]));
    try std.testing.expectError(error.AliasingNotAllowed, sparse.triangularSolveInPlace(bad));
    for ([_][]f64{ &values, std.mem.bytesAsSlice(f64, std.mem.sliceAsBytes(&columns)), std.mem.bytesAsSlice(f64, std.mem.sliceAsBytes(&offsets)) }) |aliased| {
        bad = args;
        bad.rhs.values = aliased;
        try std.testing.expectError(error.AliasingNotAllowed, sparse.triangularSolveInPlace(bad));
    }
    try unchanged(&b, &workspace);
    try std.testing.expectEqualSlices(u64, &.{ 0, 1, 2, 99, 99 }, &offsets);
    try std.testing.expectEqualSlices(u64, &.{ 0, 1, 99, 99, 99 }, &columns);
    try std.testing.expectEqualSlices(f64, &.{ 1, 1, 99, 99, 99 }, &values);
}

test "disjoint active prefixes share backing buffers and padding numerical values are unread" {
    var backing = [_]f64{ 2, 4, 6, std.math.nan(f64), 8, 10, 99, 99, 99 };
    const a = try sparse.constCsrMatrix(f64, u32, &.{ 0, 1 }, &.{0}, backing[0..1], .{ .row_count = 1, .column_count = 1 });
    const rhs = try sparse.matrix(f64, backing[2..], .{ .row_count = 1, .column_count = 2, .leading_dimension = 2 });
    const result = try sparse.triangularSolveInPlace(.{ .matrix = a, .triangle = sparse.Triangle.lower, .diagonal = sparse.Diagonal.non_unit, .rhs = rhs, .workspace = backing[5..] });
    try success(result, 2);
    try std.testing.expectEqual(@as(f64, 3), backing[2]);
    try std.testing.expect(std.math.isNan(backing[3]));
    try std.testing.expectEqual(@as(f64, 4), backing[4]);
    try std.testing.expectEqual(@as(f64, 2), backing[5]);
    try std.testing.expectEqualSlices(f64, &.{ 99, 99, 99 }, backing[6..]);
}

test "execution division overflow reports completed RHS and normal partial row semantics" {
    inline for (.{ f32, f64 }) |T| {
        const a = try sparse.constCsrMatrix(T, u32, &.{ 0, 1, 2 }, &.{ 0, 1 }, &.{ 2, 0.5 }, .{ .row_count = 2, .column_count = 2 });
        const max = std.math.floatMax(T);
        for ([_]bool{ false, true }) |transpose| {
            var b = [_]T{ 2, 2, 99, max, max, 99, 3, 3, 99 };
            var workspace: [2]T = @splat(88);
            const rhs = try sparse.matrix(T, &b, .{ .row_count = 2, .column_count = 3, .leading_dimension = 3 });
            const status = try sparse.triangularSolveInPlace(.{ .matrix = if (transpose) a.transposed() else a, .triangle = if (transpose) sparse.Triangle.upper else sparse.Triangle.lower, .diagonal = sparse.Diagonal.non_unit, .rhs = rhs, .workspace = &workspace });
            try std.testing.expectEqualDeep(sparse.TriangularSolveResult{ .info = 2, .failure = .non_finite_result, .rhs_column = 2, .completed_rhs = 1 }, status);
            try std.testing.expectEqualSlices(T, &.{ 1, 4, 99, max / 2, max, 99, 3, 3, 99 }, &b);
            try std.testing.expectEqualSlices(T, &.{ 2, 0.5 }, &workspace);
        }
    }
}

test "transposed scatter failure retains preceding finite updates and leaves failing update untouched" {
    inline for (.{ f32, f64 }) |T| {
        const a = try sparse.constCsrMatrix(T, i32, &.{ 0, 0, 0, 2 }, &.{ 0, 1 }, &.{ 1, std.math.floatMax(T) }, .{ .row_count = 3, .column_count = 3 });
        var b = [_]T{ 3, 5, 2 };
        var empty: [0]T = .{};
        const rhs = try sparse.matrix(T, &b, .{ .row_count = 3, .column_count = 1 });
        const status = try sparse.triangularSolveInPlace(.{ .matrix = a.transposed(), .triangle = sparse.Triangle.lower, .diagonal = sparse.Diagonal.unit, .rhs = rhs, .workspace = &empty });
        try std.testing.expectEqualDeep(sparse.TriangularSolveResult{ .info = 2, .failure = .non_finite_result, .rhs_column = 1 }, status);
        try std.testing.expectEqualSlices(T, &.{ 1, 5, 2 }, &b);
    }
}

test "finite product subtraction overflow never stores a nonfinite result" {
    inline for (.{ f32, f64 }) |T| {
        const max = std.math.floatMax(T);
        const a = try sparse.constCsrMatrix(T, i32, &.{ 0, 0, 1 }, &.{0}, &.{max}, .{ .row_count = 2, .column_count = 2 });
        var empty: [0]T = .{};
        for ([_]bool{ false, true }) |transpose| {
            var b = if (transpose) [_]T{ -max, 1 } else [_]T{ 1, -max };
            const original = b;
            const rhs = try sparse.matrix(T, &b, .{ .row_count = 2, .column_count = 1 });
            const result = try sparse.triangularSolveInPlace(.{ .matrix = if (transpose) a.transposed() else a, .triangle = sparse.Triangle.lower, .diagonal = sparse.Diagonal.unit, .rhs = rhs, .workspace = &empty });
            try std.testing.expectEqualDeep(sparse.TriangularSolveResult{ .info = if (transpose) 1 else 2, .failure = .non_finite_result, .rhs_column = 1 }, result);
            try std.testing.expectEqualSlices(T, &original, &b);
        }
    }
}

test "zero-valued RHS solves to zero while selected diagonal validation still applies" {
    inline for (.{ f32, f64 }) |T| {
        const max = std.math.floatMax(T);
        const a = try sparse.constCsrMatrix(T, u64, &.{ 0, 1, 3 }, &.{ 0, 1, 0 }, &.{ 2, 3, max }, .{ .row_count = 2, .column_count = 2 });
        for ([_]bool{ false, true }) |transpose| {
            var b: [6]T = @splat(0);
            var workspace: [2]T = @splat(std.math.nan(T));
            const rhs = try sparse.matrix(T, &b, .{ .row_count = 2, .column_count = 3 });
            try success(try sparse.triangularSolveInPlace(.{ .matrix = if (transpose) a.transposed() else a, .triangle = sparse.Triangle.lower, .diagonal = sparse.Diagonal.non_unit, .rhs = rhs, .workspace = &workspace }), 3);
            for (b) |value| try std.testing.expectEqual(@as(T, 0), value);
            try std.testing.expectEqualSlices(T, &.{ 2, 3 }, &workspace);
        }
    }
}

test "fixed empty workspace preserves earlier RHS capacity and shape errors" {
    const a = try sparse.constCsrMatrix(f64, u32, &.{ 0, 1 }, &.{0}, &.{2}, .{ .row_count = 1, .column_count = 2 });
    var storage = [_]f64{7};
    var rhs = try sparse.matrix(f64, &storage, .{ .row_count = 1, .column_count = 1 });
    rhs.values = storage[0..0];
    try std.testing.expectError(error.BufferTooSmall, sparse.triangularSolveInPlace(.{
        .matrix = a,
        .rhs = rhs,
        .triangle = .lower,
        .diagonal = .non_unit,
        .workspace = &[_]f64{},
    }));
    rhs.values = &storage;
    try std.testing.expectError(error.DimensionMismatch, sparse.triangularSolveInPlace(.{
        .matrix = a,
        .rhs = rhs,
        .triangle = .lower,
        .diagonal = .non_unit,
        .workspace = &[_]f64{},
    }));
    try std.testing.expectEqual(@as(f64, 7), storage[0]);
}

test "fixed empty array workspaces report shortage before numerical input errors" {
    inline for (.{ f32, f64 }) |T| {
        const a = try sparse.constCsrMatrix(T, i64, &.{ 0, 1 }, &.{0}, &.{std.math.nan(T)}, .{ .row_count = 1, .column_count = 1 });
        var storage = [_]T{7};
        const rhs = try sparse.matrix(T, &storage, .{ .row_count = 1, .column_count = 1 });
        var empty: [0]T = .{};
        inline for (.{ &empty, &[_]T{} }) |workspace| {
            try std.testing.expectError(error.WorkspaceTooSmall, sparse.triangularSolveInPlace(.{
                .matrix = a,
                .rhs = rhs,
                .triangle = .lower,
                .diagonal = .non_unit,
                .workspace = workspace,
            }));
        }
        try std.testing.expectEqual(@as(T, 7), storage[0]);
    }
}

test "fixed empty workspace supports empty and zero RHS nonunit solves" {
    inline for (.{ f32, f64 }) |T| {
        var empty: [0]T = .{};
        const no_rows = try sparse.constCsrMatrix(T, u64, &.{0}, &.{}, &.{}, .{ .row_count = 0, .column_count = 0 });
        const no_rows_rhs = try sparse.matrix(T, &empty, .{ .row_count = 0, .column_count = 3, .leading_dimension = std.math.maxInt(usize) });
        try success(try sparse.triangularSolveInPlace(.{ .matrix = no_rows, .rhs = no_rows_rhs, .triangle = .upper, .diagonal = .non_unit, .workspace = &[_]T{} }), 3);
        const no_rhs_a = try sparse.constCsrMatrix(T, u64, &.{ 0, 1, 2 }, &.{ 0, 1 }, &.{ std.math.nan(T), std.math.inf(T) }, .{ .row_count = 2, .column_count = 2 });
        const no_rhs = try sparse.matrix(T, &empty, .{ .row_count = 2, .column_count = 0 });
        try success(try sparse.triangularSolveInPlace(.{ .matrix = no_rhs_a, .rhs = no_rhs, .triangle = .lower, .diagonal = .non_unit, .workspace = &empty }), 0);
    }
}

test "runtime unit diagonal uses fixed empty workspace for padded multiple RHS" {
    inline for (.{ f32, f64 }) |T| {
        const a = try sparse.constCsrMatrix(T, i32, &.{ 0, 1, 3 }, &.{ 0, 0, 1 }, &.{ std.math.nan(T), 2, std.math.inf(T) }, .{ .row_count = 2, .column_count = 2 });
        for ([_]sparse.Diagonal{.unit}) |diagonal| {
            var rhs_values = [_]T{ 3, 10, 91, -1, 3, 93 };
            const rhs = try sparse.matrix(T, &rhs_values, .{ .row_count = 2, .column_count = 2, .leading_dimension = 3 });
            try success(try sparse.triangularSolveInPlace(.{ .matrix = a, .rhs = rhs, .triangle = .lower, .diagonal = diagonal, .workspace = &[_]T{} }), 2);
            try std.testing.expectEqualSlices(T, &.{ 3, 4, 91, -1, 5, 93 }, &rhs_values);
        }
    }
}

fn cvalue(comptime T: type, re: f64, im: f64) T {
    return .{ .re = @floatCast(re), .im = @floatCast(im) };
}
fn cadd(comptime T: type, x: T, y: T) T {
    return .{ .re = x.re + y.re, .im = x.im + y.im };
}
fn cmul(comptime T: type, x: T, y: T) T {
    return .{ .re = x.re * y.re - x.im * y.im, .im = x.re * y.im + x.im * y.re };
}
fn cnear(comptime T: type, x: T, y: T) !void {
    const tol: @TypeOf(x.re) = if (T == sparse.ComplexF32) 3e-5 else 3e-12;
    try std.testing.expectApproxEqAbs(x.re, y.re, tol);
    try std.testing.expectApproxEqAbs(x.im, y.im, tol);
}

test "complex CSR triangular solve independent dense residual all mixed indices N T H triangle diagonal and padding" {
    @setEvalBranchQuota(20000);
    inline for (.{ sparse.ComplexF32, sparse.ComplexF64 }) |T| {
        const R = @TypeOf(@as(T, undefined).re);
        inline for (.{ i32, u32, i64, u64 }) |O| inline for (.{ i32, u32, i64, u64 }) |I| {
            const offsets = [_]O{ 0, 3, 6, 9 };
            const cols = [_]I{ 2, 0, 1, 2, 0, 1, 2, 0, 1 };
            const dense = [_][3]T{
                .{ cvalue(T, 3, 2), cvalue(T, 0.5, -0.2), cvalue(T, -1, 0.7) },
                .{ cvalue(T, 1, 0.3), cvalue(T, 4, -1), cvalue(T, 0.2, 0.6) },
                .{ cvalue(T, -0.4, 0.5), cvalue(T, 0.7, -0.2), cvalue(T, 2.5, 0.5) },
            };
            for ([_]sparse.Triangle{ .lower, .upper }) |triangle| for ([_]sparse.Diagonal{ .unit, .non_unit }) |diagonal| for ([_]sparse.MatrixTransform{ .normal, .transposed, .adjoint }) |op| {
                var values: [9]T = undefined;
                for (0..3) |i| for (0..3) |e| {
                    const j: usize = @intCast(cols[i * 3 + e]);
                    values[i * 3 + e] = if (finiteSelection(i, j, triangle, diagonal)) dense[i][j] else cvalue(T, std.math.nan(R), std.math.nan(R));
                };
                const a: sparse.ConstCsrMatrixWithIndexTypes(T, O, I) = .{ .values = &values, .column_indices = &cols, .row_offsets = &offsets, .row_count = 3, .column_count = 3, .operation = op };
                const wanted = [_][3]T{ .{ cvalue(T, 1, 2), cvalue(T, -0.5, 0.4), cvalue(T, 0.7, -0.3) }, .{ cvalue(T, -0.2, 0.6), cvalue(T, 0.4, 0.9), cvalue(T, 1.3, -0.5) } };
                var rhs: [8]T = @splat(cvalue(T, 77, 9));
                for (0..2) |col| for (0..3) |i| {
                    var sum = cvalue(T, 0, 0);
                    for (0..3) |j| {
                        const row = if (op == .normal) i else j;
                        const column = if (op == .normal) j else i;
                        var coefficient = if (row == column and diagonal == .unit) cvalue(T, 1, 0) else if (finiteSelection(row, column, triangle, diagonal)) dense[row][column] else cvalue(T, 0, 0);
                        if (op == .adjoint) coefficient.im = -coefficient.im;
                        sum = cadd(T, sum, cmul(T, coefficient, wanted[col][j]));
                    }
                    rhs[col * 4 + i] = sum;
                };
                var work: [3]T = undefined;
                try success(try sparse.triangularSolveInPlace(.{ .matrix = a, .triangle = triangle, .diagonal = diagonal, .rhs = sparse.Matrix(T){ .values = &rhs, .row_count = 3, .column_count = 2, .leading_dimension = 4 }, .workspace = &work }), 2);
                for (0..2) |col| for (0..3) |i| {
                    try cnear(T, wanted[col][i], rhs[col * 4 + i]);
                };
                try std.testing.expectEqual(cvalue(T, 77, 9), rhs[3]);
                try std.testing.expectEqual(cvalue(T, 77, 9), rhs[7]);
            };
        };
    }
}

test "real CSR triangular solve permits independent offset ordinal widths" {
    @setEvalBranchQuota(20000);
    inline for (.{ f32, f64 }) |T| inline for (.{ i32, u32, i64, u64 }) |O| inline for (.{ i32, u32, i64, u64 }) |I| {
        const offsets = [_]O{ 0, 1, 3 };
        const cols = [_]I{ 0, 1, 0 };
        const values = [_]T{ 2, 3, 1 };
        const a: sparse.ConstCsrMatrixWithIndexTypes(T, O, I) = .{ .values = &values, .row_offsets = &offsets, .column_indices = &cols, .row_count = 2, .column_count = 2 };
        var rhs = [_]T{ 4, 11 };
        var work: [2]T = undefined;
        try success(try sparse.triangularSolveInPlace(.{ .matrix = a, .triangle = sparse.Triangle.lower, .diagonal = sparse.Diagonal.non_unit, .rhs = sparse.Matrix(T){ .values = &rhs, .row_count = 2, .column_count = 1, .leading_dimension = 2 }, .workspace = &work }), 1);
        try near(T, 2, rhs[0]);
        try near(T, 3, rhs[1]);
    };
}

test "complex triangular admitted solve completes after close without extra admission or resource pin" {
    const Core = @import("zynum-core");
    const E = Core.internal.numerics.Execution;
    inline for (.{ sparse.ComplexF32, sparse.ComplexF64 }) |T| {
        var ctx = E.context.Context.init(std.testing.allocator, E.test_support.policy, .internal_closable);
        var op = try ctx.begin();
        _ = ctx.requestClose();
        const offsets = [_]i64{ 0, 1 };
        const cols = [_]i32{0};
        const values = [_]T{cvalue(T, 2, 1)};
        var rhs = [_]T{cvalue(T, 3, 4)};
        var work: [1]T = undefined;
        const a: sparse.ConstCsrMatrixWithIndexTypes(T, i64, i32) = .{ .values = &values, .row_offsets = &offsets, .column_indices = &cols, .row_count = 1, .column_count = 1 };
        try success(try Core.internal.sparse_checked.triangularSolveInPlaceAdmitted(&op, .{ .matrix = a, .triangle = sparse.Triangle.lower, .diagonal = sparse.Diagonal.non_unit, .rhs = sparse.Matrix(T){ .values = &rhs, .row_count = 1, .column_count = 1, .leading_dimension = 1 }, .workspace = &work }), 1);
        try cnear(T, cvalue(T, 2, 1), rhs[0]);
        try std.testing.expectEqual(@as(usize, 1), ctx.snapshot().active_operations);
        try std.testing.expectEqual(@as(usize, 0), ctx.snapshot().resource_pins);
        try std.testing.expectError(error.Closing, ctx.begin());
        op.end();
        try std.testing.expectEqual(E.context.CloseResult.closed, ctx.tryProgressClose());
    }
}

test "complex triangular singular duplicate diagonal and nonfinite preflight do not mutate active storage" {
    inline for (.{ sparse.ComplexF32, sparse.ComplexF64 }) |T| {
        const R = @TypeOf(@as(T, undefined).re);
        const offsets = [_]i64{ 0, 2 };
        const cols = [_]u32{ 0, 0 };
        var values = [_]T{ cvalue(T, 2, 1), cvalue(T, -2, -1) };
        const a: sparse.ConstCsrMatrixWithIndexTypes(T, i64, u32) = .{ .values = &values, .row_offsets = &offsets, .column_indices = &cols, .row_count = 1, .column_count = 1 };
        var rhs = [_]T{cvalue(T, 3, 4)};
        var work = [_]T{cvalue(T, 99, 7)};
        const before_rhs = rhs;
        const before_work = work;
        const args = .{ .matrix = a, .triangle = sparse.Triangle.lower, .diagonal = sparse.Diagonal.non_unit, .rhs = sparse.Matrix(T){ .values = &rhs, .row_count = 1, .column_count = 1, .leading_dimension = 1 }, .workspace = &work };
        const failed = try sparse.triangularSolveInPlace(args);
        try std.testing.expectEqual(@as(usize, 1), failed.info);
        try std.testing.expectEqual(sparse.TriangularSolveFailure.singular_diagonal, failed.failure.?);
        try std.testing.expectEqual(@as(usize, 0), failed.rhs_column);
        try std.testing.expectEqualSlices(T, &before_rhs, &rhs);
        try std.testing.expectEqualSlices(T, &before_work, &work);
        values[1] = cvalue(T, std.math.nan(R), 0);
        try std.testing.expectError(error.NonFiniteInput, sparse.triangularSolveInPlace(args));
        try std.testing.expectEqualSlices(T, &before_rhs, &rhs);
        try std.testing.expectEqualSlices(T, &before_work, &work);
        values[1] = cvalue(T, 0, 0);
        rhs[0] = cvalue(T, 0, std.math.inf(R));
        const nonfinite_rhs = rhs;
        try std.testing.expectError(error.NonFiniteInput, sparse.triangularSolveInPlace(args));
        try std.testing.expectEqualSlices(T, &nonfinite_rhs, &rhs);
        try std.testing.expectEqualSlices(T, &before_work, &work);
        rhs = before_rhs;
        values[1] = cvalue(T, 2, 1);
        rhs[0] = cvalue(T, 6, 8);
        try success(try sparse.triangularSolveInPlace(args), 1);
        try cnear(T, cvalue(T, 2, 1), rhs[0]);
    }
}

test "complex triangular scaled division and arithmetic failure preserve pending store and earlier RHS completion" {
    inline for (.{ sparse.ComplexF32, sparse.ComplexF64 }) |T| {
        const R = @TypeOf(@as(T, undefined).re);
        const big = std.math.floatMax(R);
        const offsets1 = [_]u64{ 0, 1 };
        const cols1 = [_]i32{0};
        const val1 = [_]T{.{ .re = big, .im = big }};
        var rhs1 = val1;
        var work1: [1]T = undefined;
        const a1: sparse.ConstCsrMatrixWithIndexTypes(T, u64, i32) = .{ .values = &val1, .row_offsets = &offsets1, .column_indices = &cols1, .row_count = 1, .column_count = 1 };
        try success(try sparse.triangularSolveInPlace(.{ .matrix = a1, .triangle = sparse.Triangle.lower, .diagonal = sparse.Diagonal.non_unit, .rhs = sparse.Matrix(T){ .values = &rhs1, .row_count = 1, .column_count = 1, .leading_dimension = 1 }, .workspace = &work1 }), 1);
        try cnear(T, cvalue(T, 1, 0), rhs1[0]);
        const offsets = [_]i64{ 0, 1, 3 };
        const cols = [_]i32{ 0, 0, 1 };
        const values = [_]T{ cvalue(T, 1, 0), .{ .re = big, .im = 0 }, cvalue(T, 1, 0) };
        const a: sparse.ConstCsrMatrixWithIndexTypes(T, i64, i32) = .{ .values = &values, .row_offsets = &offsets, .column_indices = &cols, .row_count = 2, .column_count = 2 };
        var rhs = [_]T{ cvalue(T, 0, 0), cvalue(T, 1, 0), cvalue(T, 2, 0), cvalue(T, 3, 0), cvalue(T, 99, 0), cvalue(T, 99, 0) };
        var work: [2]T = undefined;
        const result = try sparse.triangularSolveInPlace(.{ .matrix = a, .triangle = sparse.Triangle.lower, .diagonal = sparse.Diagonal.non_unit, .rhs = sparse.Matrix(T){ .values = &rhs, .row_count = 2, .column_count = 3, .leading_dimension = 2 }, .workspace = &work });
        try std.testing.expectEqual(@as(usize, 2), result.info);
        try std.testing.expectEqual(sparse.TriangularSolveFailure.non_finite_result, result.failure.?);
        try std.testing.expectEqual(@as(usize, 2), result.rhs_column);
        try std.testing.expectEqual(@as(usize, 1), result.completed_rhs);
        try cnear(T, cvalue(T, 0, 0), rhs[0]);
        try cnear(T, cvalue(T, 1, 0), rhs[1]);
        try cnear(T, cvalue(T, 3, 0), rhs[3]);
        try cnear(T, cvalue(T, 99, 0), rhs[4]);
        try cnear(T, cvalue(T, 99, 0), rhs[5]);
    }
}

test "complex triangular zero RHS ignores numbers and typed alias and workspace errors precede mutation" {
    inline for (.{ sparse.ComplexF32, sparse.ComplexF64 }) |T| {
        const R = @TypeOf(@as(T, undefined).re);
        const offsets = [_]i64{ 0, 1 };
        const cols = [_]i32{0};
        const values = [_]T{cvalue(T, std.math.nan(R), std.math.inf(R))};
        const a: sparse.ConstCsrMatrixWithIndexTypes(T, i64, i32) = .{ .values = &values, .row_offsets = &offsets, .column_indices = &cols, .row_count = 1, .column_count = 1 };
        var empty: [0]T = .{};
        try success(try sparse.triangularSolveInPlace(.{ .matrix = a, .triangle = sparse.Triangle.lower, .diagonal = sparse.Diagonal.non_unit, .rhs = sparse.Matrix(T){ .values = &empty, .row_count = 1, .column_count = 0, .leading_dimension = 1 }, .workspace = &empty }), 0);
        var rhs = [_]T{cvalue(T, 3, 4)};
        const before = rhs;
        const b = sparse.Matrix(T){ .values = &rhs, .row_count = 1, .column_count = 1, .leading_dimension = 1 };
        try std.testing.expectError(error.WorkspaceTooSmall, sparse.triangularSolveInPlace(.{ .matrix = a, .triangle = sparse.Triangle.lower, .diagonal = sparse.Diagonal.non_unit, .rhs = b, .workspace = &empty }));
        try std.testing.expectError(error.AliasingNotAllowed, sparse.triangularSolveInPlace(.{ .matrix = a, .triangle = sparse.Triangle.lower, .diagonal = sparse.Diagonal.non_unit, .rhs = b, .workspace = &rhs }));
        try std.testing.expectEqualSlices(T, &before, &rhs);
    }
}

test "mixed offset byte spans reject RHS alias into wide offsets before any numerical read" {
    inline for (.{ sparse.ComplexF32, sparse.ComplexF64 }) |T| {
        var offsets = [_]i64{ 0, 1, 0, 0 };
        const cols = [_]i32{0};
        const values = [_]T{cvalue(T, 2, 1)};
        var work: [1]T = undefined;
        const before = offsets;
        const out: [*]T = @ptrCast(&offsets[1]);
        const a: sparse.ConstCsrMatrixWithIndexTypes(T, i64, i32) = .{ .values = &values, .row_offsets = offsets[0..2], .column_indices = &cols, .row_count = 1, .column_count = 1 };
        try std.testing.expectError(error.AliasingNotAllowed, sparse.triangularSolveInPlace(.{ .matrix = a, .triangle = sparse.Triangle.lower, .diagonal = sparse.Diagonal.non_unit, .rhs = sparse.Matrix(T){ .values = out[0..1], .row_count = 1, .column_count = 1, .leading_dimension = 1 }, .workspace = &work }));
        try std.testing.expectEqualSlices(i64, &before, &offsets);
    }
}
