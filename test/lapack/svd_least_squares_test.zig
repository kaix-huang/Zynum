// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
const std = @import("std");
const core = @import("zynum-core");
const solver = core.internal.numerics.SvdLeastSquares;
const scalar = core.internal.numerics.Scalar;
const execution = core.internal.numerics.Execution;
const dense = core.internal.numerics.Dense;
const scalars = .{ f32, f64, scalar.ComplexF32, scalar.ComplexF64 };
const Z = struct { re: f128 = 0, im: f128 = 0 };
fn add(a: Z, b: Z) Z {
    return .{ .re = a.re + b.re, .im = a.im + b.im };
}
fn scale(a: Z, factor: f128) Z {
    return .{ .re = a.re * factor, .im = a.im * factor };
}
fn multiply(a: Z, b: Z) Z {
    return .{ .re = a.re * b.re - a.im * b.im, .im = a.re * b.im + a.im * b.re };
}
fn conjugate(a: Z) Z {
    return .{ .re = a.re, .im = -a.im };
}
fn value(comptime T: type, z: Z) T {
    return if (comptime scalar.isComplex(T)) .{ .re = @floatCast(z.re), .im = @floatCast(z.im) } else @floatCast(z.re);
}
fn wide(comptime T: type, x: T) Z {
    return if (comptime scalar.isComplex(T)) .{ .re = x.re, .im = x.im } else .{ .re = x };
}
fn close(comptime T: type, expected: Z, actual: T, denominator: f128) !void {
    const observed = wide(T, actual);
    const tolerance = 1024 * @as(f128, std.math.floatEps(scalar.Real(T)));
    try std.testing.expect(@abs(observed.re - expected.re) / denominator <= tolerance);
    try std.testing.expect(@abs(observed.im - expected.im) / denominator <= tolerance);
}
fn basis(comptime T: type, dimension: usize, row: usize, column: usize, right: bool) Z {
    const hadamard = [4][4]f128{ .{ 1, 1, 1, 1 }, .{ 1, -1, 1, -1 }, .{ 1, 1, -1, -1 }, .{ 1, -1, -1, 1 } };
    const amplitude: f128 = if (dimension == 4) hadamard[row][column] * 0.5 else if (row == column) 1 else 0;
    if (comptime !scalar.isComplex(T)) return .{ .re = amplitude * (if ((row + @intFromBool(right)) % 2 == 0) @as(f128, 1) else -1) };
    const phases = [4]Z{ .{ .re = 1 }, .{ .im = 1 }, .{ .re = -1 }, .{ .im = -1 } };
    return scale(phases[(row + @intFromBool(right)) % 4], amplitude);
}
fn matrix(comptime T: type, values: []T, m: usize, n: usize, ld: usize) dense.Matrix(T) {
    return .{ .values = values, .row_count = m, .column_count = n, .leading_dimension = ld };
}
fn outside(comptime T: type, values: []const T, rows: usize, cols: usize, ld: usize, marker: T) !void {
    for (values, 0..) |x, i| {
        if (i / ld < cols and i % ld < rows) continue;
        try std.testing.expectEqual(marker, x);
    }
}
fn fixture(comptime T: type, m: usize, n: usize, deficient: bool, relative: f128, a_scale: f128, b_scale: f128, options: solver.Options) !void {
    const R = scalar.Real(T);
    const marker = value(T, .{ .re = 73, .im = -19 });
    var av: [64]T = @splat(marker);
    var bv: [48]T = @splat(marker);
    var xv: [48]T = @splat(marker);
    var vv: [64]T = @splat(marker);
    var rv: [32]T = @splat(marker);
    var singular: [6]R = @splat(37);
    const sigma = [3]f128{ 4, 1, if (deficient) @as(f128, 0) else 0.25 };
    const coordinates = [3][2]f128{ .{ 2, -1 }, .{ -1, 3 }, .{ 0.5, 2 } };
    const threshold = relative * 4;
    var rank: usize = 0;
    for (sigma) |s| if (s > threshold) {
        rank += 1;
    };
    const a = matrix(T, &av, m, n, m + 2);
    const b = matrix(T, &bv, m, 2, m + 2);
    const output: solver.Outputs(T) = .{ .solution = matrix(T, &xv, n, 2, n + 2), .right_vectors = matrix(T, &vv, n, 3, n + 2), .residual = matrix(T, &rv, m -| n, 2, (m -| n) + 2), .singular = &singular };
    for (0..n) |j| for (0..m) |i| {
        var entry: Z = .{};
        for (0..3) |q| entry = add(entry, scale(multiply(basis(T, m, i, q, false), conjugate(basis(T, n, j, q, true))), sigma[q] * a_scale));
        a.values[i + j * a.leading_dimension] = value(T, entry);
    };
    for (0..2) |j| for (0..m) |i| {
        var entry: Z = .{};
        for (0..3) |q| entry = add(entry, scale(basis(T, m, i, q, false), sigma[q] * coordinates[q][j] * b_scale));
        if (m == 4) entry = add(entry, scale(basis(T, 4, i, 3, false), @as(f128, @floatFromInt(j + 1)) * b_scale));
        b.values[i + j * b.leading_dimension] = value(T, entry);
    };
    const old_a = av;
    const old_b = bv;
    var context = execution.context.Context.init(std.testing.allocator, execution.test_support.policy, .internal_closable);
    const result = try solver.solveManaged(&context, T, a.asConst(), b.asConst(), output, .{ .relative = @floatCast(relative) }, options);
    try std.testing.expectEqual(.converged, result.status);
    try std.testing.expectEqual(rank, result.rank);
    try std.testing.expectEqual(@as(usize, 0), result.unconverged);
    for (0..2) |j| for (0..n) |i| {
        var expected: Z = .{};
        for (0..3) |q| if (sigma[q] > threshold) {
            expected = add(expected, scale(basis(T, n, i, q, true), coordinates[q][j] * b_scale / a_scale));
        };
        try close(T, expected, xv[i + j * output.solution.leading_dimension], @abs(b_scale / a_scale));
    };
    for (sigma, singular[0..3]) |expected, actual| try std.testing.expect(@abs(@as(f128, actual) / @abs(a_scale) - expected) <= 4096 * @as(f128, std.math.floatEps(R)));
    if (m > n and rank == n) {
        for (0..2) |j| {
            var norm: f128 = 0;
            for (0..m - n) |i| {
                const z = wide(T, rv[i + j * output.residual.leading_dimension]);
                norm += (z.re / b_scale) * (z.re / b_scale) + (z.im / b_scale) * (z.im / b_scale);
            }
            const expected = @as(f128, @floatFromInt((j + 1) * (j + 1)));
            try std.testing.expect(@abs(norm - expected) <= 8192 * @as(f128, std.math.floatEps(R)));
        }
    } else {
        for (rv) |x| try std.testing.expectEqual(marker, x);
    }
    for (0..3) |p| for (0..3) |q| {
        var dot: Z = .{};
        for (0..n) |i| dot = add(dot, multiply(conjugate(wide(T, vv[i + p * output.right_vectors.leading_dimension])), wide(T, vv[i + q * output.right_vectors.leading_dimension])));
        try std.testing.expect(@abs(dot.re - @as(f128, if (p == q) 1 else 0)) <= 1024 * @as(f128, std.math.floatEps(R)));
        try std.testing.expect(@abs(dot.im) <= 1024 * @as(f128, std.math.floatEps(R)));
    };
    try std.testing.expectEqualSlices(T, &old_a, &av);
    try std.testing.expectEqualSlices(T, &old_b, &bv);
    try outside(T, &xv, n, 2, n + 2, marker);
    try outside(T, &vv, n, 3, n + 2, marker);
    try outside(T, &rv, m -| n, 2, (m -| n) + 2, marker);
    for (singular[3..]) |x| try std.testing.expectEqual(@as(R, 37), x);
    try std.testing.expectEqual(@as(usize, 0), context.snapshot().active_operations);
    try std.testing.expectEqual(@as(usize, 0), context.snapshot().resource_pins);
    _ = context.requestClose();
    try std.testing.expectEqual(execution.context.CloseResult.closed, context.tryProgressClose());
}
test "SVD least squares shared Core real complex shapes ranks cutoff and independent scales" {
    var configurations: usize = 0;
    inline for (scalars) |T| {
        const tiny: f128 = if (scalar.Real(T) == f32) 1e-10 else 1e-100;
        const huge: f128 = if (scalar.Real(T) == f32) 1e10 else 1e100;
        const scales = [7][2]f128{ .{ 1, 1 }, .{ tiny, 1 }, .{ huge, 1 }, .{ 1, tiny }, .{ 1, huge }, .{ tiny, huge }, .{ huge, tiny } };
        for ([_][2]usize{ .{ 4, 3 }, .{ 3, 4 }, .{ 3, 3 } }) |shape| for ([_]bool{ false, true }) |deficient| for ([_]f128{ 0.01, 0.125, 0.75, 2 }) |cutoff| for (scales) |pair| {
            inline for (.{ false, true }) |ieee| {
                try fixture(T, shape[0], shape[1], deficient, cutoff, pair[0], pair[1], .{ .numerical_policy = if (ieee) .ieee else .finite });
                configurations += 1;
            }
        };
    }
    try std.testing.expectEqual(@as(usize, 1344), configurations);
}
test "SVD least squares exact cutoff equality zero relative absolute and singleton semantics" {
    inline for (scalars) |T| {
        var av: [9]T = @splat(scalar.zero(T));
        av[0] = value(T, .{ .re = 4 });
        av[4] = value(T, .{ .re = 1 });
        av[8] = value(T, .{ .re = 0.25 });
        var bv = [3]T{ value(T, .{ .re = 4 }), value(T, .{ .re = 2 }), value(T, .{ .re = 1 }) };
        var xv: [3]T = undefined;
        var vv: [9]T = undefined;
        var singular: [3]scalar.Real(T) = undefined;
        const cuts = [5]solver.Cutoff(T){ .{}, .{ .relative = 0.25 }, .{ .relative = 1 }, .{ .absolute = 1 }, .{ .absolute = 0.25 } };
        for (cuts, [_]usize{ 3, 1, 0, 1, 2 }) |cut, expected| {
            var context = execution.context.Context.init(std.testing.allocator, execution.test_support.policy, .internal_closable);
            const result = try solver.solveManaged(&context, T, matrix(T, &av, 3, 3, 3).asConst(), matrix(T, &bv, 3, 1, 3).asConst(), .{ .solution = matrix(T, &xv, 3, 1, 3), .right_vectors = matrix(T, &vv, 3, 3, 3), .singular = &singular, .residual = matrix(T, &.{}, 0, 1, 0) }, cut, .{});
            try std.testing.expectEqual(.converged, result.status);
            try std.testing.expectEqual(expected, result.rank);
            _ = context.requestClose();
            try std.testing.expectEqual(execution.context.CloseResult.closed, context.tryProgressClose());
        }
        var context = execution.context.Context.init(std.testing.allocator, execution.test_support.policy, .internal_closable);
        var one_a = [1]T{value(T, .{ .re = 2 })};
        var one_b = [1]T{value(T, .{ .re = 6 })};
        const result = try solver.solveManaged(&context, T, matrix(T, &one_a, 1, 1, 1).asConst(), matrix(T, &one_b, 1, 1, 1).asConst(), .{ .solution = matrix(T, xv[0..1], 1, 1, 1), .right_vectors = matrix(T, vv[0..1], 1, 1, 1), .singular = singular[0..1], .residual = matrix(T, &.{}, 0, 1, 0) }, .{ .relative = 1 }, .{});
        try std.testing.expectEqual(@as(usize, 0), result.rank);
        try close(T, .{}, xv[0], 1);
        _ = context.requestClose();
        try std.testing.expectEqual(execution.context.CloseResult.closed, context.tryProgressClose());
    }
}
test "SVD least squares all scratch allocations borrow closing operation and unwind on failure" {
    inline for (scalars) |T| for (0..4) |budget| {
        var failing = std.testing.FailingAllocator.init(std.testing.allocator, .{ .fail_index = budget });
        var context = execution.context.Context.init(failing.allocator(), execution.test_support.policy, .internal_closable);
        var operation = try context.begin();
        _ = context.requestClose();
        var av = [4]T{ value(T, .{ .re = 2 }), scalar.zero(T), scalar.zero(T), value(T, .{ .re = 4 }) };
        var bv = [2]T{ value(T, .{ .re = 6 }), value(T, .{ .re = 8 }) };
        const marker = value(T, .{ .re = 19, .im = -23 });
        var xv: [2]T = @splat(marker);
        var vv: [4]T = @splat(marker);
        var singular: [2]scalar.Real(T) = @splat(31);
        const out: solver.Outputs(T) = .{ .solution = matrix(T, &xv, 2, 1, 2), .right_vectors = matrix(T, &vv, 2, 2, 2), .singular = &singular, .residual = matrix(T, &.{}, 0, 1, 0) };
        if (budget < 3) {
            try std.testing.expectError(error.OutOfMemory, solver.solveAdmitted(&operation, T, matrix(T, &av, 2, 2, 2).asConst(), matrix(T, &bv, 2, 1, 2).asConst(), out, .{}, .{}));
            for (xv) |x| try std.testing.expectEqual(marker, x);
            for (vv) |x| try std.testing.expectEqual(marker, x);
            for (singular) |x| try std.testing.expectEqual(@as(scalar.Real(T), 31), x);
        } else {
            const result = try solver.solveAdmitted(&operation, T, matrix(T, &av, 2, 2, 2).asConst(), matrix(T, &bv, 2, 1, 2).asConst(), out, .{}, .{});
            try std.testing.expectEqual(.converged, result.status);
            try close(T, .{ .re = 3 }, xv[0], 1);
            try close(T, .{ .re = 2 }, xv[1], 1);
        }
        try std.testing.expectEqual(@as(usize, 1), context.snapshot().active_operations);
        try std.testing.expectEqual(@as(usize, 0), context.snapshot().resource_pins);
        try std.testing.expectEqual(failing.allocated_bytes, failing.freed_bytes);
        try std.testing.expectError(error.Closing, context.begin());
        operation.end();
        try std.testing.expectEqual(execution.context.CloseResult.closed, context.tryProgressClose());
    };
}
test "SVD least squares resource count overflow unwinds original closing operation" {
    inline for (scalars) |T| for ([_]usize{ 1, 2 }) |limit| {
        var context = execution.context.Context.init(std.testing.allocator, execution.test_support.policy, .internal_closable);
        execution.context.Context.testing.configure(&context, limit, null);
        var operation = try context.begin();
        _ = context.requestClose();
        var av = [1]T{value(T, .{ .re = 2 })};
        var bv = [1]T{value(T, .{ .re = 6 })};
        var xv = [1]T{value(T, .{ .re = 19 })};
        var vv = [1]T{value(T, .{ .re = 23 })};
        var singular = [1]scalar.Real(T){31};
        try std.testing.expectError(error.CounterOverflow, solver.solveAdmitted(&operation, T, matrix(T, &av, 1, 1, 1).asConst(), matrix(T, &bv, 1, 1, 1).asConst(), .{ .solution = matrix(T, &xv, 1, 1, 1), .right_vectors = matrix(T, &vv, 1, 1, 1), .singular = &singular, .residual = matrix(T, &.{}, 0, 1, 0) }, .{}, .{}));
        try close(T, .{ .re = 19 }, xv[0], 1);
        try close(T, .{ .re = 23 }, vv[0], 1);
        try std.testing.expectEqual(@as(scalar.Real(T), 31), singular[0]);
        try std.testing.expectEqual(@as(usize, 1), context.snapshot().active_operations);
        try std.testing.expectEqual(@as(usize, 0), context.snapshot().resource_pins);
        operation.end();
        try std.testing.expectEqual(execution.context.CloseResult.closed, context.tryProgressClose());
    };
}
test "SVD least squares preflight and closed Context reject without allocating or writing outputs" {
    inline for (scalars) |T| {
        var failing = std.testing.FailingAllocator.init(std.testing.allocator, .{ .fail_index = 0 });
        var context = execution.context.Context.init(failing.allocator(), execution.test_support.policy, .internal_closable);
        var av: [4]T = @splat(value(T, .{ .re = 2 }));
        var bv: [2]T = @splat(value(T, .{ .re = 1 }));
        const marker = value(T, .{ .re = 19, .im = -23 });
        var xv: [2]T = @splat(marker);
        var vv: [4]T = @splat(marker);
        var singular: [2]scalar.Real(T) = @splat(31);
        const a = matrix(T, &av, 2, 2, 2);
        const b = matrix(T, &bv, 2, 1, 2);
        const out: solver.Outputs(T) = .{ .solution = matrix(T, &xv, 2, 1, 2), .right_vectors = matrix(T, &vv, 2, 2, 2), .singular = &singular, .residual = matrix(T, &.{}, 0, 1, 0) };
        var wrong = out;
        wrong.solution.row_count = 1;
        try std.testing.expectError(error.DimensionMismatch, solver.solveManaged(&context, T, a.asConst(), b.asConst(), wrong, .{}, .{}));
        wrong = out;
        wrong.singular = singular[0..1];
        try std.testing.expectError(error.BufferTooSmall, solver.solveManaged(&context, T, a.asConst(), b.asConst(), wrong, .{}, .{}));
        wrong = out;
        wrong.solution.values = &av;
        try std.testing.expectError(error.AliasingNotAllowed, solver.solveManaged(&context, T, a.asConst(), b.asConst(), wrong, .{}, .{}));
        wrong = out;
        wrong.solution.values = &bv;
        try std.testing.expectError(error.AliasingNotAllowed, solver.solveManaged(&context, T, a.asConst(), b.asConst(), wrong, .{}, .{}));
        const mixed: [*]scalar.Real(T) = @ptrCast(&av);
        wrong = out;
        wrong.singular = mixed[0..2];
        try std.testing.expectError(error.AliasingNotAllowed, solver.solveManaged(&context, T, a.asConst(), b.asConst(), wrong, .{}, .{}));
        try std.testing.expectError(error.InvalidCutoff, solver.solveManaged(&context, T, a.asConst(), b.asConst(), out, .{ .relative = -1 }, .{}));
        try std.testing.expectError(error.InvalidCutoff, solver.solveManaged(&context, T, a.asConst(), b.asConst(), out, .{ .absolute = std.math.nan(solver.Wide(T)) }, .{}));
        try std.testing.expectError(error.InvalidTolerance, solver.solveManaged(&context, T, a.asConst(), b.asConst(), out, .{}, .{ .relative_tolerance = 0 }));
        av[0] = value(T, .{ .re = std.math.nan(f128) });
        try std.testing.expectError(error.NonFiniteInput, solver.solveManaged(&context, T, a.asConst(), b.asConst(), out, .{}, .{}));
        for (xv) |x| try std.testing.expectEqual(marker, x);
        for (vv) |x| try std.testing.expectEqual(marker, x);
        for (singular) |x| try std.testing.expectEqual(@as(scalar.Real(T), 31), x);
        try std.testing.expectEqual(@as(usize, 0), failing.allocated_bytes);
        try std.testing.expectEqual(@as(usize, 0), context.snapshot().active_operations);
        try std.testing.expectEqual(@as(usize, 0), context.snapshot().resource_pins);
        _ = context.requestClose();
        try std.testing.expectEqual(execution.context.CloseResult.closed, context.tryProgressClose());
        try std.testing.expectError(error.Closing, solver.solveManaged(&context, T, a.asConst(), b.asConst(), out, .{}, .{}));
        try std.testing.expectEqual(@as(usize, 0), failing.allocated_bytes);
        try std.testing.expectError(error.SizeOverflow, solver.plan(T, std.math.maxInt(usize), 1, 1));
    }
}
test "SVD least squares bounded factorization and checked reconstruction failures preserve publication" {
    inline for (scalars) |T| {
        var av: [9]T = undefined;
        for (0..3) |j| for (0..3) |i| {
            av[i + j * 3] = value(T, .{ .re = @floatFromInt(1 + (i + j * 2) % 5), .im = @floatFromInt(i + j + 1) });
        };
        var bv: [3]T = @splat(value(T, .{ .re = 1 }));
        const marker = value(T, .{ .re = 19, .im = -23 });
        var xv: [3]T = @splat(marker);
        var vv: [9]T = @splat(marker);
        var singular: [3]scalar.Real(T) = @splat(31);
        var context = execution.context.Context.init(std.testing.allocator, execution.test_support.policy, .internal_closable);
        const result = try solver.solveManaged(&context, T, matrix(T, &av, 3, 3, 3).asConst(), matrix(T, &bv, 3, 1, 3).asConst(), .{ .solution = matrix(T, &xv, 3, 1, 3), .right_vectors = matrix(T, &vv, 3, 3, 3), .singular = &singular, .residual = matrix(T, &.{}, 0, 1, 0) }, .{}, .{ .max_steps = 0 });
        try std.testing.expectEqual(.no_convergence, result.status);
        try std.testing.expectEqual(solver.Stage.factorization, result.stage);
        try std.testing.expect(result.unconverged > 0 and result.unconverged <= 2);
        for (xv) |x| try std.testing.expectEqual(marker, x);
        for (vv) |x| try std.testing.expectEqual(marker, x);
        for (singular) |x| try std.testing.expectEqual(@as(scalar.Real(T), 31), x);
        var tiny_a = [1]T{value(T, .{ .re = std.math.floatTrueMin(scalar.Real(T)) })};
        const out: solver.Outputs(T) = .{ .solution = matrix(T, xv[0..1], 1, 1, 1), .right_vectors = matrix(T, vv[0..1], 1, 1, 1), .singular = singular[0..1], .residual = matrix(T, &.{}, 0, 1, 0) };
        const failed = try solver.solveManaged(&context, T, matrix(T, &tiny_a, 1, 1, 1).asConst(), matrix(T, bv[0..1], 1, 1, 1).asConst(), out, .{}, .{});
        try std.testing.expectEqual(.non_finite_result, failed.status);
        try std.testing.expectEqual(solver.Stage.reconstruction, failed.stage);
        try std.testing.expectEqual(marker, xv[0]);
        try std.testing.expectEqual(marker, vv[0]);
        try std.testing.expectEqual(@as(scalar.Real(T), 31), singular[0]);
        const ieee = try solver.solveManaged(&context, T, matrix(T, &tiny_a, 1, 1, 1).asConst(), matrix(T, bv[0..1], 1, 1, 1).asConst(), out, .{}, .{ .numerical_policy = .ieee });
        try std.testing.expectEqual(.converged, ieee.status);
        try std.testing.expectEqual(@as(usize, 0), ieee.unconverged);
        try std.testing.expect(std.math.isInf(wide(T, xv[0]).re));
        const maximum = std.math.floatMax(scalar.Real(T));
        var publication_a = [2]T{ value(T, .{ .re = maximum }), value(T, .{ .re = maximum }) };
        var publication_b = [2]T{ value(T, .{ .re = 1 }), value(T, .{ .re = 1 }) };
        var publication_x = [1]T{marker};
        var publication_v = [1]T{marker};
        var publication_residual = [1]T{marker};
        var publication_singular = [1]scalar.Real(T){31};
        const publication = try solver.solveManaged(&context, T, matrix(T, &publication_a, 2, 1, 2).asConst(), matrix(T, &publication_b, 2, 1, 2).asConst(), .{ .solution = matrix(T, &publication_x, 1, 1, 1), .right_vectors = matrix(T, &publication_v, 1, 1, 1), .singular = &publication_singular, .residual = matrix(T, &publication_residual, 1, 1, 1) }, .{}, .{});
        try std.testing.expectEqual(.non_finite_result, publication.status);
        try std.testing.expectEqual(solver.Stage.publication, publication.stage);
        try std.testing.expectEqual(marker, publication_x[0]);
        try std.testing.expectEqual(marker, publication_v[0]);
        try std.testing.expectEqual(marker, publication_residual[0]);
        try std.testing.expectEqual(@as(scalar.Real(T), 31), publication_singular[0]);
        try std.testing.expectEqual(@as(usize, 0), context.snapshot().active_operations);
        try std.testing.expectEqual(@as(usize, 0), context.snapshot().resource_pins);
        _ = context.requestClose();
        try std.testing.expectEqual(execution.context.CloseResult.closed, context.tryProgressClose());
    }
}
test "SVD least squares zero RHS still factors and empty or zero matrices retain defined outputs" {
    inline for (scalars) |T| {
        for ([_][2]usize{ .{ 3, 2 }, .{ 2, 3 }, .{ 0, 2 }, .{ 2, 0 }, .{ 0, 0 }, .{ std.math.maxInt(usize), 0 } }) |shape| {
            const m = shape[0];
            const n = shape[1];
            const k = @min(m, n);
            var av: [12]T = @splat(scalar.zero(T));
            for (0..k) |i| av[i + i * m] = value(T, .{ .re = @floatFromInt(i + 2) });
            var xv: [1]T = @splat(value(T, .{ .re = 17 }));
            var vv: [12]T = undefined;
            var singular: [3]scalar.Real(T) = @splat(31);
            var context = execution.context.Context.init(std.testing.allocator, execution.test_support.policy, .internal_closable);
            const result = try solver.solveManaged(&context, T, matrix(T, &av, m, n, m).asConst(), matrix(T, &.{}, m, 0, m).asConst(), .{ .solution = matrix(T, &xv, n, 0, n), .right_vectors = matrix(T, &vv, n, k, n), .singular = &singular, .residual = matrix(T, &.{}, m -| n, 0, m -| n) }, .{}, .{});
            try std.testing.expectEqual(.converged, result.status);
            try std.testing.expectEqual(k, result.rank);
            if (k > 0) try std.testing.expectEqual(@as(scalar.Real(T), @floatFromInt(k + 1)), singular[0]);
            try close(T, .{ .re = 17 }, xv[0], 1);
            _ = context.requestClose();
            try std.testing.expectEqual(execution.context.CloseResult.closed, context.tryProgressClose());
        }
        for ([_][2]usize{ .{ 0, 2 }, .{ 2, 0 }, .{ 3, 2 } }) |shape| {
            const m = shape[0];
            const n = shape[1];
            const k = @min(m, n);
            var av: [6]T = @splat(scalar.zero(T));
            var bv = [3]T{ value(T, .{ .re = 1 }), value(T, .{ .re = 2 }), value(T, .{ .re = 3 }) };
            var xv: [2]T = @splat(value(T, .{ .re = 19 }));
            var vv: [4]T = undefined;
            var rv: [2]T = @splat(value(T, .{ .re = 23 }));
            var singular: [2]scalar.Real(T) = @splat(31);
            var context = execution.context.Context.init(std.testing.allocator, execution.test_support.policy, .internal_closable);
            const result = try solver.solveManaged(&context, T, matrix(T, &av, m, n, m).asConst(), matrix(T, &bv, m, 1, m).asConst(), .{ .solution = matrix(T, &xv, n, 1, n), .right_vectors = matrix(T, &vv, n, k, n), .singular = &singular, .residual = matrix(T, &rv, m -| n, 1, m -| n) }, .{}, .{});
            try std.testing.expectEqual(.converged, result.status);
            try std.testing.expectEqual(@as(usize, 0), result.rank);
            for (xv[0..n]) |x| try close(T, .{}, x, 1);
            for (singular[0..k]) |s| try std.testing.expectEqual(@as(scalar.Real(T), 0), s);
            if (n == 0) for (0..m) |i| try std.testing.expectEqual(bv[i], rv[i]);
            if (m > n and n != 0) for (rv) |x| try close(T, .{ .re = 23 }, x, 1);
            _ = context.requestClose();
            try std.testing.expectEqual(execution.context.CloseResult.closed, context.tryProgressClose());
        }
    }
}
test "SVD least squares Core keeps existing scalar workspace operation and option identities" {
    try std.testing.expect(solver.Options == core.internal.numerics.BidiagonalSvd.Options);
    try std.testing.expect(solver.Options == core.internal.numerics.ComplexBidiagonalSvd.Options);
    try std.testing.expect(solver.WorkspaceScalar(f64) == core.internal.numerics.BidiagonalSvd.WorkspaceScalar(f64));
    try std.testing.expect(solver.WorkspaceScalar(scalar.ComplexF64) == core.internal.numerics.ComplexBidiagonalSvd.WorkspaceScalar(scalar.ComplexF64));
    try std.testing.expect(execution.context.Operation == @import("zynum-execution").context.Operation);
}
test "SVD least squares FFT descriptor and mixed index Sparse control retain one closing Context" {
    inline for (scalars) |T| {
        const C = if (scalar.Real(T) == f32) scalar.ComplexF32 else scalar.ComplexF64;
        const Descriptor = core.internal.numerics.FFT.Persistent(execution, C);
        const Control = core.internal.numerics.SparseObject.ManagedMatrixControl(execution.context, T, i64, i32);
        const offsets = [_]i64{ 0, 1, 2 };
        const ordinals = [_]i32{ 0, 1 };
        const numbers = [_]T{ value(T, .{ .re = 2 }), value(T, .{ .re = 4 }) };
        const pattern: core.internal.numerics.SparsePattern.Pattern(i64, i32) = .{ .ranges = .{ .contiguous = &offsets }, .ordinals = &ordinals, .row_count = 2, .column_count = 2 };
        var context = execution.context.Context.init(std.testing.allocator, execution.test_support.policy, .internal_closable);
        var operation = try context.begin();
        var descriptor = try Descriptor.createAdmitted(&operation, .{ .complex_1d = 2 });
        try descriptor.commitAdmitted(&operation);
        const sparse = try Control.borrow(&operation, pattern, &numbers);
        try std.testing.expectEqual(@as(usize, 2), context.snapshot().resource_pins);
        _ = context.requestClose();
        const exported = try sparse.exportView(&operation, .row);
        try std.testing.expectEqualSlices(T, &numbers, exported.values);
        var av = [4]T{ value(T, .{ .re = 2 }), scalar.zero(T), scalar.zero(T), value(T, .{ .re = 4 }) };
        var xv: [2]T = undefined;
        var vv: [4]T = undefined;
        var singular: [2]scalar.Real(T) = undefined;
        const result = try solver.solveAdmitted(&operation, T, matrix(T, &av, 2, 2, 2).asConst(), .{ .values = exported.values, .row_count = 2, .column_count = 1, .leading_dimension = 2 }, .{ .solution = matrix(T, &xv, 2, 1, 2), .right_vectors = matrix(T, &vv, 2, 2, 2), .singular = &singular, .residual = matrix(T, &.{}, 0, 1, 0) }, .{}, .{});
        try std.testing.expectEqual(.converged, result.status);
        for (xv) |x| try close(T, .{ .re = 1 }, x, 1);
        try std.testing.expectEqual(@as(usize, 2), context.snapshot().resource_pins);
        const x0 = wide(T, xv[0]);
        const x1 = wide(T, xv[1]);
        const input = [2]C{ .{ .re = @floatCast(x0.re), .im = @floatCast(x0.im) }, .{ .re = @floatCast(x1.re), .im = @floatCast(x1.im) } };
        var output: [2]C = undefined;
        try descriptor.runScratchAdmitted(.complex_forward, &operation, &input, &output);
        try close(C, .{ .re = 2 }, output[0], 1);
        try close(C, .{}, output[1], 1);
        try std.testing.expectEqual(@as(usize, 1), context.snapshot().active_operations);
        try std.testing.expectEqual(@as(usize, 2), context.snapshot().resource_pins);
        try std.testing.expectError(error.Closing, context.begin());
        operation.end();
        try std.testing.expectEqual(execution.context.CloseResult.waiting_resources, context.tryProgressClose());
        sparse.release();
        try std.testing.expectEqual(@as(usize, 1), context.snapshot().resource_pins);
        try std.testing.expectEqual(execution.context.CloseResult.waiting_resources, context.tryProgressClose());
        descriptor.deinit();
        try std.testing.expectEqual(@as(usize, 0), context.snapshot().resource_pins);
        try std.testing.expectEqual(execution.context.CloseResult.closed, context.tryProgressClose());
    }
}
