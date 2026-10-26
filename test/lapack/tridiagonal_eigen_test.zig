const std = @import("std");
const core = @import("numerical-owner").Provider.TridiagonalEigen;
const dense = @import("numerical-owner").Provider.Dense;

fn verify(comptime T: type, original_d: []const T, original_e: []const T, d: []const T, e: []const T, z: []const T, ld: usize, tolerance: f64) !void {
    const n = d.len;
    var scale: f64 = 0;
    for (original_d) |x| scale = @max(scale, @abs(@as(f64, x)));
    for (original_e) |x| scale = @max(scale, @abs(@as(f64, x)));
    scale = @max(scale, std.math.floatMin(f64));
    for (0..n) |j| for (0..n) |i| {
        var left = (@as(f64, original_d[i]) / scale) * @as(f64, z[i + j * ld]);
        if (i > 0) left += (@as(f64, original_e[i - 1]) / scale) * @as(f64, z[i - 1 + j * ld]);
        if (i + 1 < n) left += (@as(f64, original_e[i]) / scale) * @as(f64, z[i + 1 + j * ld]);
        var right = @as(f64, z[i + j * ld]) * (@as(f64, d[j]) / scale);
        if (j > 0) right += @as(f64, z[i + (j - 1) * ld]) * (@as(f64, e[j - 1]) / scale);
        if (j + 1 < n) right += @as(f64, z[i + (j + 1) * ld]) * (@as(f64, e[j]) / scale);
        try std.testing.expectApproxEqAbs(left, right, tolerance);
        var dot: f64 = 0;
        for (0..n) |k| dot += @as(f64, z[k + i * ld]) * @as(f64, z[k + j * ld]);
        try std.testing.expectApproxEqAbs(@as(f64, if (i == j) 1 else 0), dot, tolerance);
    };
}

fn matrix(comptime T: type, values: []T, n: usize, ld: usize) dense.Matrix(T) {
    return .{ .values = values, .row_count = n, .column_count = n, .leading_dimension = ld };
}

test "tridiagonal Toeplitz analytic eigenvalues and padded orthogonal vectors" {
    inline for (.{ f32, f64 }) |T| {
        const n = 7;
        var d: [n]T = @splat(2);
        var e: [n - 1]T = @splat(-1);
        const original_d = d;
        const original_e = e;
        var z: [9 * n + 2]T = @splat(73);
        const result = core.solve(T, &d, &e, matrix(T, &z, n, 9), .{});
        try std.testing.expectEqual(core.Status.converged, result.status);
        for (d, 0..) |value, i| {
            const expected = 2 - 2 * @cos(std.math.pi * @as(f64, @floatFromInt(i + 1)) / (n + 1));
            try std.testing.expectApproxEqAbs(expected, @as(f64, value), 40 * std.math.floatEps(T));
        }
        try verify(T, &original_d, &original_e, &d, &e, &z, 9, 80 * std.math.floatEps(T));
        for (0..n) |j| for (n..9) |i| try std.testing.expectEqual(@as(T, 73), z[i + j * 9]);
        try std.testing.expectEqual(@as(T, 73), z[z.len - 1]);
    }
}

test "tridiagonal values-only agrees with vector path for deterministic signed matrices" {
    inline for (.{ f32, f64 }) |T| {
        var seed: u64 = 1729;
        for (0..32) |_| {
            var d: [9]T = undefined;
            var e: [8]T = undefined;
            for (&d) |*x| {
                seed = seed *% 6364136223846793005 +% 1;
                x.* = @floatCast(@as(f64, @floatFromInt(seed >> 32)) / 2147483648.0 - 1);
            }
            for (&e) |*x| {
                seed = seed *% 6364136223846793005 +% 1;
                x.* = @floatCast(@as(f64, @floatFromInt(seed >> 32)) / 2147483648.0 - 1);
            }
            const original_d = d;
            const original_e = e;
            var values = d;
            var edges = e;
            var z: [81]T = undefined;
            try std.testing.expectEqual(core.Status.converged, core.solve(T, &d, &e, matrix(T, &z, 9, 9), .{}).status);
            try std.testing.expectEqual(core.Status.converged, core.solve(T, &values, &edges, null, .{}).status);
            try std.testing.expectEqualSlices(T, &d, &values);
            try verify(T, &original_d, &original_e, &d, &e, &z, 9, 200 * std.math.floatEps(T));
            for (1..d.len) |i| try std.testing.expect(d[i - 1] <= d[i]);
            for (e) |x| try std.testing.expectEqual(@as(T, 0), x);
        }
    }
}

test "tridiagonal disconnected extreme scales and repeated eigenvalues" {
    inline for (.{ f32, f64 }) |T| {
        const exponent: i32 = if (T == f32) 100 else 900;
        const big = std.math.ldexp(@as(T, 1), exponent);
        const small = std.math.ldexp(@as(T, 1), -exponent);
        var d = [_]T{ 2 * big, 2 * big, 2 * small, 2 * small, 3 * small };
        var e = [_]T{ big, 0, small, 0 };
        const od = d;
        const oe = e;
        var z: [25]T = undefined;
        try std.testing.expectEqual(core.Status.converged, core.solve(T, &d, &e, matrix(T, &z, 5, 5), .{}).status);
        const expected = [_]T{ small, 3 * small, 3 * small, big, 3 * big };
        for (expected, d) |x, y| try std.testing.expectApproxEqRel(x, y, 20 * std.math.floatEps(T));
        try verify(T, &od, &oe, &d, &e, &z, 5, 100 * std.math.floatEps(T));
    }
}

test "tridiagonal bounded iteration retains similar tridiagonal and counts edges" {
    inline for (.{ f32, f64 }) |T| inline for (.{ @as(usize, 0), @as(usize, 1) }) |budget| {
        var d = [_]T{ 4, -1, 5, 2 };
        var e = [_]T{ 2, 3, -1 };
        const od = d;
        const oe = e;
        var z: [16]T = undefined;
        const result = core.solve(T, &d, &e, matrix(T, &z, 4, 4), .{ .max_iterations = budget });
        try std.testing.expectEqual(core.Status.iteration_limit, result.status);
        try std.testing.expectEqual(budget, result.iterations);
        var count: usize = 0;
        for (e) |x| if (x != 0) {
            count += 1;
        };
        try std.testing.expectEqual(count, result.unconverged);
        try verify(T, &od, &oe, &d, &e, &z, 4, 100 * std.math.floatEps(T));
    };
}

test "tridiagonal empty singleton suffix and nonfinite preflight" {
    var d: [0]f64 = .{};
    var empty_e = [_]f64{31};
    try std.testing.expectEqual(core.Status.converged, core.solve(f64, &d, &empty_e, null, .{}).status);
    try std.testing.expectEqual(@as(f64, 31), empty_e[0]);
    var one = [_]f64{-7};
    var z = [_]f64{ 91, 91 };
    try std.testing.expectEqual(core.Status.converged, core.solve(f64, &one, &empty_e, matrix(f64, &z, 1, 2), .{}).status);
    try std.testing.expectEqualSlices(f64, &.{ 1, 91 }, &z);
    var bad_d = [_]f64{ 2, 3 };
    var bad_e = [_]f64{std.math.inf(f64)};
    var bad_z: [4]f64 = @splat(91);
    try std.testing.expectEqual(core.Status.nonfinite_input, core.solve(f64, &bad_d, &bad_e, matrix(f64, &bad_z, 2, 2), .{}).status);
    try std.testing.expectEqualSlices(f64, &.{ 2, 3 }, &bad_d);
    try std.testing.expectEqualSlices(f64, &@as([4]f64, @splat(91)), &bad_z);
}

test "tridiagonal zero diagonal pairs retain subnormal and large eigenvalues" {
    inline for (.{ f32, f64 }) |T| {
        const minimum = std.math.floatTrueMin(T);
        const maximum = std.math.floatMax(T) / 4;
        for ([_]T{ minimum, std.math.floatMin(T), 1, maximum }) |edge| {
            var d = [_]T{ 0, 0 };
            var e = [_]T{edge};
            const od = d;
            const oe = e;
            var z: [4]T = undefined;
            try std.testing.expectEqual(core.Status.converged, core.solve(T, &d, &e, matrix(T, &z, 2, 2), .{}).status);
            try std.testing.expectApproxEqAbs(@as(f64, -edge), @as(f64, d[0]), @as(f64, edge) * 10 * std.math.floatEps(T));
            try std.testing.expectApproxEqAbs(@as(f64, edge), @as(f64, d[1]), @as(f64, edge) * 10 * std.math.floatEps(T));
            try verify(T, &od, &oe, &d, &e, &z, 2, 80 * std.math.floatEps(T));
        }
    }
}

test "tridiagonal exact zero and diagonal inputs need no iterations" {
    inline for (.{ f32, f64 }) |T| {
        var d = [_]T{ 2, -3, 0, 2, 0 };
        var e: [4]T = @splat(0);
        const od = d;
        const oe = e;
        var z: [25]T = undefined;
        const result = core.solve(T, &d, &e, matrix(T, &z, 5, 5), .{ .max_iterations = 0 });
        try std.testing.expectEqual(core.Status.converged, result.status);
        try std.testing.expectEqual(@as(usize, 0), result.iterations);
        try std.testing.expectEqualSlices(T, &.{ -3, 0, 0, 2, 2 }, &d);
        try verify(T, &od, &oe, &d, &e, &z, 5, 0);
    }
}

test "tridiagonal unfinished independent blocks count all remaining edges" {
    var d = [_]f64{ 4, -1, 5, 2, 1, 1 };
    var e = [_]f64{ 2, 3, -1, 0, 0.25 };
    const od = d;
    const oe = e;
    var z: [36]f64 = undefined;
    const result = core.solve(f64, &d, &e, matrix(f64, &z, 6, 6), .{ .max_iterations = 1 });
    try std.testing.expectEqual(core.Status.iteration_limit, result.status);
    try std.testing.expectEqual(@as(usize, 4), result.unconverged);
    try std.testing.expectEqual(@as(f64, 0.25), e[4]);
    try verify(f64, &od, &oe, &d, &e, &z, 6, 100 * std.math.floatEps(f64));
}

test "tridiagonal clustered spectrum resolves with orthogonal vectors" {
    inline for (.{ f32, f64 }) |T| {
        const n = 24;
        const edge: T = if (T == f32) 0.0001 else 0.0000000001;
        var d: [n]T = @splat(1);
        var e: [n - 1]T = @splat(-edge);
        const od = d;
        const oe = e;
        var z: [n * n]T = undefined;
        const result = core.solve(T, &d, &e, matrix(T, &z, n, n), .{});
        try std.testing.expectEqual(core.Status.converged, result.status);
        for (d, 0..) |value, i| {
            const expected = 1 - 2 * @as(f64, edge) * @cos(std.math.pi * @as(f64, @floatFromInt(i + 1)) / (n + 1));
            try std.testing.expectApproxEqAbs(expected, @as(f64, value), 80 * std.math.floatEps(T));
        }
        try verify(T, &od, &oe, &d, &e, &z, n, 200 * std.math.floatEps(T));
    }
}

test "tridiagonal unrepresentable eigenvalue is distinct from convergence limit" {
    inline for (.{ f32, f64 }) |T| {
        const large = std.math.floatMax(T);
        var d = [_]T{ large, large };
        var e = [_]T{large};
        const result = core.solve(T, &d, &e, null, .{});
        try std.testing.expectEqual(core.Status.range_limit, result.status);
        try std.testing.expectEqual(@as(usize, 0), result.unconverged);
        try std.testing.expect(std.math.isInf(d[0]) or std.math.isInf(d[1]));
    }
}

test "tridiagonal standard IEEE policy bounds nonfinite iterations without checked status" {
    inline for (.{ f32, f64 }) |T| {
        var singleton = [_]T{std.math.inf(T)};
        var no_edges: [0]T = .{};
        try std.testing.expectEqual(core.Status.converged, core.solveWithContract(T, &singleton, &no_edges, null, .{}, .stev_ieee_v1).status);
        for ([_]T{ std.math.inf(T), std.math.nan(T) }) |edge| {
            var d = [_]T{ 1, 2 };
            var e = [_]T{edge};
            const result = core.solveWithContract(T, &d, &e, null, .{}, .stev_ieee_v1);
            try std.testing.expectEqual(core.Status.iteration_limit, result.status);
            try std.testing.expectEqual(@as(usize, 1), result.unconverged);
            try std.testing.expectEqual(@as(usize, 60), result.iterations);
        }
        var d = [_]T{ std.math.floatMax(T), std.math.floatMax(T) };
        var e = [_]T{std.math.floatMax(T)};
        const result = core.solveWithContract(T, &d, &e, null, .{}, .stev_ieee_v1);
        try std.testing.expectEqual(core.Status.converged, result.status);
        try std.testing.expectEqual(@as(usize, 0), result.unconverged);
    }
}
