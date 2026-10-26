// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
const std = @import("std");
const sparse = @import("zynum-sparse");

fn denseResidual(comptime T: type, dense: []const T, b: []const T, x: []const T) f64 {
    var squared: f64 = 0;
    for (0..b.len) |row| {
        var product: f64 = 0;
        for (0..b.len) |column| product += @as(f64, dense[row * b.len + column]) * @as(f64, x[column]);
        const residual = @as(f64, b[row]) - product;
        squared += residual * residual;
    }
    return @sqrt(squared);
}

fn checkConverged(comptime T: type, result: sparse.ConjugateGradientResult(T), dense: []const T, b: []const T, x: []const T) !void {
    try std.testing.expectEqual(sparse.ConjugateGradientStatus.converged, result.status);
    try std.testing.expect(result.true_residual_norm.? <= result.tolerance);
    const independent = denseResidual(T, dense, b, x);
    try std.testing.expect(independent <= @as(f64, @floatCast(result.tolerance)) * 1.01 + 1e-12);
}

test "CG solves independent dense SPD products with unsorted duplicates all types and warm starts" {
    inline for (.{ f32, f64 }) |T| {
        inline for (.{ i32, u32, i64, u64 }) |I| {
            const dense = [_]T{ 4, 1, 0.5, 0, 1, 5, 1, 0.25, 0.5, 1, 6, 1, 0, 0.25, 1, 3 };
            const expected = [_]T{ 1, -2, 0.5, 3 };
            var b: [4]T = undefined;
            for (0..4) |row| {
                b[row] = 0;
                for (0..4) |column| b[row] += dense[row * 4 + column] * expected[column];
            }
            const offsets = [_]I{ 0, 6, 12, 18, 24 };
            var columns: [24]I = undefined;
            var values: [24]T = undefined;
            for (0..4) |row| {
                const duplicate = (row + 1) % 4;
                for ([_]usize{ 3, 1, 0, 2, row, duplicate }, 0..) |column, entry| {
                    columns[row * 6 + entry] = @intCast(column);
                    const part: T = if (column == row) 0.5 else if (column == duplicate) (if (entry == 5) 0.25 else 0.75) else 1;
                    values[row * 6 + entry] = part * dense[row * 4 + column];
                }
            }
            const a = try sparse.constCsrMatrix(T, I, &offsets, &columns, &values, .{ .row_count = 4, .column_count = 4 });
            for ([_]sparse.InitialGuess{ .zero, .provided }) |guess| {
                var x = if (guess == .zero) [_]T{ std.math.nan(T), std.math.nan(T), std.math.nan(T), std.math.nan(T) } else [_]T{ 0.25, -0.5, 2, 1 };
                const initial = x;
                var workspace: [18]T = @splat(99);
                const args = .{ .matrix = a, .rhs = &b, .solution = &x, .workspace = &workspace, .initial_guess = guess, .relative_tolerance = @as(T, if (T == f32) 2e-6 else 1e-12), .absolute_tolerance = @as(T, 0), .max_iterations = @as(usize, 32) };
                const result = try sparse.conjugateGradient(args);
                try checkConverged(T, result, &dense, &b, &x);
                for (x, expected) |actual, wanted| try std.testing.expectApproxEqAbs(wanted, actual, @as(T, if (T == f32) 0.00002 else 1e-10));
                try std.testing.expectEqualSlices(T, &.{ 99, 99 }, workspace[16..]);
                const solved = x;
                x = initial;
                const repeated = try sparse.conjugateGradient(args);
                try std.testing.expectEqualDeep(result, repeated);
                try std.testing.expectEqualSlices(T, &solved, &x);
            }
            var exact = expected;
            var workspace: [16]T = undefined;
            const result = try sparse.conjugateGradient(.{ .matrix = a, .rhs = &b, .solution = &exact, .workspace = &workspace, .initial_guess = sparse.InitialGuess.provided, .relative_tolerance = @as(T, 0), .absolute_tolerance = @as(T, 0), .max_iterations = @as(usize, 0) });
            try checkConverged(T, result, &dense, &b, &exact);
            try std.testing.expectEqual(@as(usize, 0), result.iterations);
        }
    }
}

test "CG handles empty systems zero RHS and explicit additive tolerances" {
    inline for (.{ f32, f64 }) |T| {
        const empty = try sparse.constCsrMatrix(T, u32, &.{0}, &.{}, &.{}, .{ .row_count = 0, .column_count = 0 });
        var x0: [0]T = .{};
        var w0: [0]T = .{};
        try std.testing.expectEqual(@as(usize, 0), try sparse.conjugateGradientWorkspaceLength(T, 0));
        const empty_result = try sparse.conjugateGradient(.{ .matrix = empty, .rhs = &x0, .solution = &x0, .workspace = &w0, .initial_guess = sparse.InitialGuess.zero, .relative_tolerance = @as(T, 0), .absolute_tolerance = @as(T, 0), .max_iterations = @as(usize, 0) });
        try std.testing.expectEqual(sparse.ConjugateGradientStatus.converged, empty_result.status);
        try std.testing.expectEqual(@as(usize, 0), empty_result.iterations);
        const identity = try sparse.constCsrMatrix(T, i32, &.{ 0, 1, 2 }, &.{ 0, 1 }, &.{ 1, 1 }, .{ .row_count = 2, .column_count = 2 });
        var x = [_]T{ 3, -4 };
        var work: [8]T = undefined;
        const zero = try sparse.conjugateGradient(.{ .matrix = identity, .rhs = &[_]T{ 0, 0 }, .solution = &x, .workspace = &work, .initial_guess = sparse.InitialGuess.provided, .relative_tolerance = @as(T, 0), .absolute_tolerance = @as(T, 0), .max_iterations = @as(usize, 4) });
        try std.testing.expectEqual(sparse.ConjugateGradientStatus.converged, zero.status);
        try std.testing.expectEqual(@as(usize, 1), zero.iterations);
        try std.testing.expectEqualSlices(T, &.{ 0, 0 }, &x);
        x = @splat(std.math.nan(T));
        const already_zero = try sparse.conjugateGradient(.{ .matrix = identity, .rhs = &[_]T{ 0, 0 }, .solution = &x, .workspace = &work, .initial_guess = sparse.InitialGuess.zero, .relative_tolerance = @as(T, 0), .absolute_tolerance = @as(T, 0), .max_iterations = @as(usize, 0) });
        try std.testing.expectEqual(sparse.ConjugateGradientStatus.converged, already_zero.status);
        try std.testing.expectEqual(@as(usize, 0), already_zero.iterations);
        const relaxed = try sparse.conjugateGradient(.{ .matrix = identity, .rhs = &[_]T{ 3, 4 }, .solution = &x, .workspace = &work, .initial_guess = sparse.InitialGuess.zero, .relative_tolerance = @as(T, 0.25), .absolute_tolerance = @as(T, 3.75), .max_iterations = @as(usize, 0) });
        try std.testing.expectEqual(sparse.ConjugateGradientStatus.converged, relaxed.status);
        try std.testing.expectEqual(@as(sparse.ConjugateGradientAccumulator(T), 5), relaxed.tolerance);
        try std.testing.expectEqual(@as(sparse.ConjugateGradientAccumulator(T), 5), relaxed.true_residual_norm.?);
    }
}

test "CG iteration cap retains last full iterate and reports a fresh residual" {
    inline for (.{ f32, f64 }) |T| {
        const a = try sparse.constCsrMatrix(T, u64, &.{ 0, 1, 2 }, &.{ 0, 1 }, &.{ 1, 4 }, .{ .row_count = 2, .column_count = 2 });
        const b = [_]T{ 1, 1 };
        var x = [_]T{ 2, 3 };
        var work: [8]T = undefined;
        const initial = try sparse.conjugateGradient(.{ .matrix = a, .rhs = &b, .solution = &x, .workspace = &work, .initial_guess = sparse.InitialGuess.provided, .relative_tolerance = @as(T, 0), .absolute_tolerance = @as(T, 0), .max_iterations = @as(usize, 0) });
        try std.testing.expectEqual(sparse.ConjugateGradientStatus.max_iterations, initial.status);
        try std.testing.expectEqual(@as(usize, 0), initial.iterations);
        try std.testing.expectEqualSlices(T, &.{ 2, 3 }, &x);
        const limited = try sparse.conjugateGradient(.{ .matrix = a, .rhs = &b, .solution = &x, .workspace = &work, .initial_guess = sparse.InitialGuess.zero, .relative_tolerance = @as(T, 0), .absolute_tolerance = @as(T, 0), .max_iterations = @as(usize, 1) });
        try std.testing.expectEqual(sparse.ConjugateGradientStatus.max_iterations, limited.status);
        try std.testing.expectEqual(@as(usize, 1), limited.iterations);
        try std.testing.expectEqualSlices(T, &.{ 0.4, 0.4 }, &x);
        const independent = denseResidual(T, &.{ 1, 0, 0, 4 }, &b, &x);
        try std.testing.expectApproxEqAbs(independent, @as(f64, @floatCast(limited.true_residual_norm.?)), 1e-14);
    }
}

test "CG never claims convergence from a zero recursive residual alone" {
    inline for (.{ f32, f64 }) |T| {
        const a = try sparse.constCsrMatrix(T, i32, &.{ 0, 1 }, &.{0}, &.{3}, .{ .row_count = 1, .column_count = 1 });
        var x = [_]T{0};
        var work: [4]T = undefined;
        const result = try sparse.conjugateGradient(.{ .matrix = a, .rhs = &[_]T{1}, .solution = &x, .workspace = &work, .initial_guess = sparse.InitialGuess.zero, .relative_tolerance = @as(T, 0), .absolute_tolerance = @as(T, 0), .max_iterations = @as(usize, 3) });
        try std.testing.expectEqual(sparse.ConjugateGradientStatus.max_iterations, result.status);
        try std.testing.expectEqual(@as(usize, 3), result.iterations);
        const residual = @abs(@as(sparse.ConjugateGradientAccumulator(T), 1) - @as(sparse.ConjugateGradientAccumulator(T), 3) * @as(sparse.ConjugateGradientAccumulator(T), x[0]));
        try std.testing.expect(residual > 0);
        try std.testing.expectEqual(residual, result.true_residual_norm.?);
    }
}

test "CG ill conditioned diagonal SPD convergence is checked by independent residual" {
    inline for (.{ f32, f64 }) |T| {
        const diagonal = [_]T{ 0.001, 0.1, 10, 1000 };
        const a = try sparse.constCsrMatrix(T, i64, &.{ 0, 1, 2, 3, 4 }, &.{ 0, 1, 2, 3 }, &diagonal, .{ .row_count = 4, .column_count = 4 });
        const expected = [_]T{ 1, 2, 3, 4 };
        var b: [4]T = undefined;
        var dense: [16]T = @splat(0);
        for (0..4) |i| {
            b[i] = diagonal[i] * expected[i];
            dense[5 * i] = diagonal[i];
        }
        var x: [4]T = undefined;
        var work: [16]T = undefined;
        const result = try sparse.conjugateGradient(.{ .matrix = a, .rhs = &b, .solution = &x, .workspace = &work, .initial_guess = sparse.InitialGuess.zero, .relative_tolerance = @as(T, if (T == f32) 1e-7 else 1e-13), .absolute_tolerance = @as(T, 0), .max_iterations = @as(usize, 100) });
        try checkConverged(T, result, &dense, &b, &x);
    }
}

test "CG nonpositive curvature covers indefinite and empty row breakdown without false SPD claims" {
    inline for (.{ f32, f64 }) |T| {
        const indefinite = try sparse.constCsrMatrix(T, u32, &.{ 0, 1, 2 }, &.{ 0, 1 }, &.{ 1, -1 }, .{ .row_count = 2, .column_count = 2 });
        var x: [2]T = undefined;
        var work: [8]T = undefined;
        const result = try sparse.conjugateGradient(.{ .matrix = indefinite, .rhs = &[_]T{ 1, 1 }, .solution = &x, .workspace = &work, .initial_guess = sparse.InitialGuess.zero, .relative_tolerance = @as(T, 0), .absolute_tolerance = @as(T, 0), .max_iterations = @as(usize, 10) });
        try std.testing.expectEqual(sparse.ConjugateGradientStatus.nonpositive_curvature, result.status);
        try std.testing.expectEqual(@as(usize, 0), result.iterations);
        try std.testing.expectEqualSlices(T, &.{ 0, 0 }, &x);
        const empty_rows = try sparse.constCsrMatrix(T, u32, &.{ 0, 0, 0 }, &.{}, &.{}, .{ .row_count = 2, .column_count = 2 });
        const singular = try sparse.conjugateGradient(.{ .matrix = empty_rows, .rhs = &[_]T{ 1, 1 }, .solution = &x, .workspace = &work, .initial_guess = sparse.InitialGuess.zero, .relative_tolerance = @as(T, 0), .absolute_tolerance = @as(T, 0), .max_iterations = @as(usize, 10) });
        try std.testing.expectEqual(sparse.ConjugateGradientStatus.nonpositive_curvature, singular.status);
    }
}

test "CG widened norms and products handle representable large and tiny identity solves" {
    inline for (.{ f32, f64 }) |T| {
        const identity = try sparse.constCsrMatrix(T, u32, &.{ 0, 1, 2, 3, 4 }, &.{ 0, 1, 2, 3 }, &.{ 1, 1, 1, 1 }, .{ .row_count = 4, .column_count = 4 });
        const large = std.math.floatMax(T) * @as(T, 0.75);
        const tiny: T = if (T == f32) 1e-30 else 1e-300;
        for ([_]T{ large, tiny }) |value| {
            const b: [4]T = @splat(value);
            var x: [4]T = undefined;
            var work: [16]T = undefined;
            const result = try sparse.conjugateGradient(.{ .matrix = identity, .rhs = &b, .solution = &x, .workspace = &work, .initial_guess = sparse.InitialGuess.zero, .relative_tolerance = @as(T, 0), .absolute_tolerance = @as(T, 0), .max_iterations = @as(usize, 4) });
            try std.testing.expectEqual(sparse.ConjugateGradientStatus.converged, result.status);
            try std.testing.expectEqual(@as(usize, 1), result.iterations);
            try std.testing.expectEqualSlices(T, &b, &x);
            try std.testing.expect(result.rhs_norm > 0);
            try std.testing.expectEqual(@as(sparse.ConjugateGradientAccumulator(T), 0), result.true_residual_norm.?);
            if (value == large) try std.testing.expect(result.rhs_norm > std.math.floatMax(T));
        }
    }
}

test "CG representability failures preserve the last complete solution iterate" {
    inline for (.{ f32, f64 }) |T| {
        const max = std.math.floatMax(T);
        const minimum: T = if (T == f32) 1.1754943508222875e-38 else 2.2250738585072014e-308;
        for ([_]bool{ false, true }) |update_overflow| {
            const a = try sparse.constCsrMatrix(T, u32, &.{ 0, 1 }, &.{0}, &.{if (update_overflow) minimum else max}, .{ .row_count = 1, .column_count = 1 });
            var x = [_]T{99};
            var work: [4]T = undefined;
            const result = try sparse.conjugateGradient(.{ .matrix = a, .rhs = &[_]T{if (update_overflow) max else 2}, .solution = &x, .workspace = &work, .initial_guess = sparse.InitialGuess.zero, .relative_tolerance = @as(T, 0), .absolute_tolerance = @as(T, 0), .max_iterations = @as(usize, 2) });
            try std.testing.expectEqual(sparse.ConjugateGradientStatus.nonfinite, result.status);
            try std.testing.expectEqual(@as(usize, 0), result.iterations);
            try std.testing.expectEqual(@as(T, 0), x[0]);
            try std.testing.expect(result.true_residual_norm != null);
        }
    }
}

test "CG rejects every invalid argument before any solution or workspace mutation" {
    var offsets = [_]i64{ 0, 1, 2 };
    var columns = [_]i64{ 0, 1 };
    var values = [_]f64{ 1, 2 };
    const a = try sparse.constCsrMatrix(f64, i64, &offsets, &columns, &values, .{ .row_count = 2, .column_count = 2 });
    var b = [_]f64{ 1, 1 };
    var x = [_]f64{ 7, 7 };
    var work: [8]f64 = @splat(99);
    const Arguments = struct {
        matrix: @TypeOf(a),
        rhs: []const f64,
        solution: []f64,
        workspace: []f64,
        initial_guess: sparse.InitialGuess,
        relative_tolerance: f64,
        absolute_tolerance: f64,
        max_iterations: usize,
    };
    const args: Arguments = .{ .matrix = a, .rhs = &b, .solution = &x, .workspace = &work, .initial_guess = .zero, .relative_tolerance = 1e-10, .absolute_tolerance = 0, .max_iterations = 4 };
    columns[1] = 2;
    try std.testing.expectError(error.InvalidColumnIndex, sparse.conjugateGradient(args));
    columns[1] = -1;
    try std.testing.expectError(error.InvalidColumnIndex, sparse.conjugateGradient(args));
    columns[1] = 1;
    offsets[1] = 3;
    try std.testing.expectError(error.InvalidRowOffsets, sparse.conjugateGradient(args));
    offsets[1] = 1;
    var bad = args;
    bad.matrix.column_count = 3;
    try std.testing.expectError(error.DimensionMismatch, sparse.conjugateGradient(bad));
    bad = args;
    bad.matrix = a.transposed();
    try std.testing.expectError(error.UnsupportedTransform, sparse.conjugateGradient(bad));
    bad.matrix = a.adjoint();
    try std.testing.expectError(error.UnsupportedTransform, sparse.conjugateGradient(bad));
    bad = args;
    bad.rhs = b[0..1];
    try std.testing.expectError(error.DimensionMismatch, sparse.conjugateGradient(bad));
    bad = args;
    bad.solution = x[0..1];
    try std.testing.expectError(error.DimensionMismatch, sparse.conjugateGradient(bad));
    bad = args;
    bad.workspace = work[0..7];
    try std.testing.expectError(error.WorkspaceTooSmall, sparse.conjugateGradient(bad));
    for ([_]f64{ -1, std.math.nan(f64), std.math.inf(f64) }) |invalid| {
        bad = args;
        bad.relative_tolerance = invalid;
        try std.testing.expectError(error.InvalidTolerance, sparse.conjugateGradient(bad));
        bad = args;
        bad.absolute_tolerance = invalid;
        try std.testing.expectError(error.InvalidTolerance, sparse.conjugateGradient(bad));
    }
    values[1] = std.math.nan(f64);
    try std.testing.expectError(error.NonFiniteInput, sparse.conjugateGradient(args));
    values[1] = 2;
    b[1] = std.math.inf(f64);
    try std.testing.expectError(error.NonFiniteInput, sparse.conjugateGradient(args));
    b[1] = 1;
    x[1] = std.math.nan(f64);
    bad = args;
    bad.initial_guess = .provided;
    try std.testing.expectError(error.NonFiniteInput, sparse.conjugateGradient(bad));
    try std.testing.expect(std.math.isNan(x[1]));
    x[1] = 7;
    try std.testing.expectError(error.SizeOverflow, sparse.conjugateGradientWorkspaceLength(f64, std.math.maxInt(usize)));
    try std.testing.expectError(error.SizeOverflow, sparse.conjugateGradientWorkspaceLength(f64, std.math.maxInt(usize) / 4));
    try std.testing.expectEqualSlices(f64, &.{ 7, 7 }, &x);
    for (work) |value| try std.testing.expectEqual(@as(f64, 99), value);
}

test "CG rejects full partial numerical and metadata active overlaps" {
    var offsets = [_]u64{ 0, 1, 2, 99, 99, 99, 99, 99 };
    var columns = [_]u64{ 0, 1, 99, 99, 99, 99, 99, 99 };
    var values = [_]f64{ 1, 2, 99, 99, 99, 99, 99, 99 };
    const a = try sparse.constCsrMatrix(f64, u64, offsets[0..3], columns[0..2], values[0..2], .{ .row_count = 2, .column_count = 2 });
    var b: [9]f64 = @splat(1);
    var x: [9]f64 = @splat(7);
    var work: [8]f64 = @splat(99);
    const args = .{ .matrix = a, .rhs = @as([]const f64, b[0..2]), .solution = @as([]f64, x[0..2]), .workspace = @as([]f64, &work), .initial_guess = sparse.InitialGuess.zero, .relative_tolerance = @as(f64, 1e-10), .absolute_tolerance = @as(f64, 0), .max_iterations = @as(usize, 4) };
    for ([_][]f64{ b[0..2], b[1..3], values[0..2], work[0..2], std.mem.bytesAsSlice(f64, std.mem.sliceAsBytes(offsets[0..2])), std.mem.bytesAsSlice(f64, std.mem.sliceAsBytes(columns[0..2])) }) |aliased| {
        var bad = args;
        bad.solution = aliased;
        try std.testing.expectError(error.AliasingNotAllowed, sparse.conjugateGradient(bad));
    }
    for ([_][]f64{ b[1..9], x[1..9], &values, std.mem.bytesAsSlice(f64, std.mem.sliceAsBytes(&offsets)), std.mem.bytesAsSlice(f64, std.mem.sliceAsBytes(&columns)) }) |aliased| {
        var bad = args;
        bad.workspace = aliased;
        try std.testing.expectError(error.AliasingNotAllowed, sparse.conjugateGradient(bad));
    }
    for (b) |value| try std.testing.expectEqual(@as(f64, 1), value);
    for (x) |value| try std.testing.expectEqual(@as(f64, 7), value);
    for (work) |value| try std.testing.expectEqual(@as(f64, 99), value);
    try std.testing.expectEqualSlices(u64, &.{ 0, 1, 2, 99, 99, 99, 99, 99 }, &offsets);
}

test "CG permits disjoint active prefixes and shared read only inputs" {
    var backing = [_]f64{ 2, 6, std.math.nan(f64), 99, 99, 99, 99, 88 };
    const a = try sparse.constCsrMatrix(f64, u32, &.{ 0, 1 }, &.{0}, backing[0..1], .{ .row_count = 1, .column_count = 1 });
    const result = try sparse.conjugateGradient(.{ .matrix = a, .rhs = backing[1..2], .solution = backing[2..3], .workspace = backing[3..], .initial_guess = sparse.InitialGuess.zero, .relative_tolerance = @as(f64, 0), .absolute_tolerance = @as(f64, 0), .max_iterations = @as(usize, 2) });
    try std.testing.expectEqual(sparse.ConjugateGradientStatus.converged, result.status);
    try std.testing.expectEqual(@as(f64, 3), backing[2]);
    try std.testing.expectEqual(@as(f64, 88), backing[7]);
    // CSR values and RHS can be the same read-only storage.
    const shared = try sparse.conjugateGradient(.{ .matrix = a, .rhs = backing[0..1], .solution = backing[2..3], .workspace = backing[3..], .initial_guess = sparse.InitialGuess.zero, .relative_tolerance = @as(f64, 0), .absolute_tolerance = @as(f64, 0), .max_iterations = @as(usize, 2) });
    try std.testing.expectEqual(sparse.ConjugateGradientStatus.converged, shared.status);
    try std.testing.expectEqual(@as(f64, 1), backing[2]);
}
