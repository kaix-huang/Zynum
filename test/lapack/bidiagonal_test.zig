// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
const std = @import("std");
const core = @import("zynum-core").internal.numerics.Bidiagonal;

test "bidiagonal reduction reconstructs scaled tall matrices with orthogonal factors" {
    inline for (.{ f64, f128 }) |T| {
        for ([_]T{ 1, 1e-200, 1e200 }) |scale| {
            const original = [_]T{ 1, -2, 3, 4, 2, 0, -1, 5, -3, 6, 2, 1 };
            var b: [15]T = @splat(77);
            var u: [20]T = @splat(88);
            var v: [12]T = @splat(99);
            for (0..3) |j| for (0..4) |i| {
                b[i + j * 5] = original[i + j * 4] * scale;
            };
            try core.reduce(T, .{ .values = &b, .row_count = 4, .column_count = 3, .leading_dimension = 5 }, .{ .values = &u, .row_count = 4, .column_count = 4, .leading_dimension = 5 }, .{ .values = &v, .row_count = 3, .column_count = 3, .leading_dimension = 4 });
            const tol = 512 * std.math.floatEps(T);
            for (0..4) |i| for (0..3) |j| {
                if (i != j and j != i + 1) try std.testing.expectEqual(@as(T, 0), b[i + j * 5]);
                var reconstructed: T = 0;
                for (0..4) |r| for (0..3) |c| {
                    reconstructed += u[i + r * 5] * (b[r + c * 5] / scale) * v[j + c * 4];
                };
                try std.testing.expectApproxEqAbs(original[i + j * 4], reconstructed, tol);
            };
            for (0..4) |i| for (0..4) |j| {
                var dot: T = 0;
                for (0..4) |r| dot += u[r + i * 5] * u[r + j * 5];
                try std.testing.expectApproxEqAbs(@as(T, if (i == j) 1 else 0), dot, tol);
            };
            for (0..3) |i| for (0..3) |j| {
                var dot: T = 0;
                for (0..3) |r| dot += v[r + i * 4] * v[r + j * 4];
                try std.testing.expectApproxEqAbs(@as(T, if (i == j) 1 else 0), dot, tol);
            };
            for (0..3) |j| try std.testing.expectEqual(@as(T, 77), b[4 + j * 5]);
            for (0..4) |j| try std.testing.expectEqual(@as(T, 88), u[4 + j * 5]);
            for (0..3) |j| try std.testing.expectEqual(@as(T, 99), v[3 + j * 4]);
        }
    }
}

test "bidiagonal reduction preflight and zero singleton geometry" {
    const Dense = @import("zynum-core").internal.numerics.Dense;
    var b = [_]f64{ 1, 2, 3, std.math.nan(f64) };
    var u: [4]f64 = @splat(88);
    var v: [4]f64 = @splat(99);
    const matrix: Dense.Matrix(f64) = .{ .values = &b, .row_count = 2, .column_count = 2, .leading_dimension = 2 };
    const left: Dense.Matrix(f64) = .{ .values = &u, .row_count = 2, .column_count = 2, .leading_dimension = 2 };
    const right: Dense.Matrix(f64) = .{ .values = &v, .row_count = 2, .column_count = 2, .leading_dimension = 2 };
    try std.testing.expectError(error.AliasingNotAllowed, core.reduce(f64, matrix, matrix, right));
    try std.testing.expectError(error.NonFiniteInput, core.reduce(f64, matrix, left, right));
    for (u) |value| try std.testing.expectEqual(@as(f64, 88), value);
    for (v) |value| try std.testing.expectEqual(@as(f64, 99), value);
    try std.testing.expectEqualSlices(f64, &.{ 1, 2, 3 }, b[0..3]);
    @memset(&b, 0);
    try core.reduce(f64, matrix, left, right);
    try std.testing.expectEqualSlices(f64, &.{ 1, 0, 0, 1 }, &u);
    try std.testing.expectEqualSlices(f64, &.{ 1, 0, 0, 1 }, &v);
    const empty: Dense.Matrix(f64) = .{ .values = &.{}, .row_count = 0, .column_count = 0, .leading_dimension = 0 };
    try core.reduce(f64, empty, empty, empty);
    b[0] = -3;
    try core.reduce(f64, .{ .values = b[0..1], .row_count = 1, .column_count = 1, .leading_dimension = 1 }, .{ .values = u[0..1], .row_count = 1, .column_count = 1, .leading_dimension = 1 }, .{ .values = v[0..1], .row_count = 1, .column_count = 1, .leading_dimension = 1 });
    try std.testing.expectEqual(@as(f64, -3), b[0]);
    try std.testing.expectEqual(@as(f64, 1), u[0]);
    try std.testing.expectEqual(@as(f64, 1), v[0]);
}

test "bidiagonal QR preserves partial reconstruction and reports real unfinished edges" {
    inline for (.{ f64, f128 }) |T| {
        for ([_]usize{ 0, 1, 4096 }) |budget| {
            const original = [_]T{ 3, 0, 0, 4, 2, 0, 0, 1, -1 };
            var b = original;
            var u: [9]T = undefined;
            var v: [9]T = undefined;
            const Dense = @import("zynum-core").internal.numerics.Dense;
            const bm: Dense.Matrix(T) = .{ .values = &b, .row_count = 3, .column_count = 3, .leading_dimension = 3 };
            const um: Dense.Matrix(T) = .{ .values = &u, .row_count = 3, .column_count = 3, .leading_dimension = 3 };
            const vm: Dense.Matrix(T) = .{ .values = &v, .row_count = 3, .column_count = 3, .leading_dimension = 3 };
            try core.reduce(T, bm, um, vm);
            const result = try core.diagonalize(T, bm, um, vm, .{ .max_steps = budget });
            var edges: usize = 0;
            for (0..2) |i| if (b[i + (i + 1) * 3] != 0) {
                edges += 1;
            };
            try std.testing.expectEqual(edges, result.unconverged);
            try std.testing.expectEqual(budget == 4096, result.converged);
            const tol = 4096 * std.math.floatEps(T);
            for (0..3) |i| for (0..3) |j| {
                if (i != j and j != i + 1) try std.testing.expectEqual(@as(T, 0), b[i + j * 3]);
                var value: T = 0;
                for (0..3) |r| for (0..3) |c| {
                    value += u[i + r * 3] * b[r + c * 3] * v[j + c * 3];
                };
                try std.testing.expectApproxEqAbs(original[i + j * 3], value, tol);
                var uu: T = 0;
                var vv: T = 0;
                for (0..3) |r| {
                    uu += u[r + i * 3] * u[r + j * 3];
                    vv += v[r + i * 3] * v[r + j * 3];
                }
                try std.testing.expectApproxEqAbs(@as(T, if (i == j) 1 else 0), uu, tol);
                try std.testing.expectApproxEqAbs(@as(T, if (i == j) 1 else 0), vv, tol);
            };
            if (result.converged) {
                try std.testing.expect(b[0] >= b[4] and b[4] >= b[8] and b[8] >= 0);
                try std.testing.expectApproxEqAbs(@as(T, 31), b[0] * b[0] + b[4] * b[4] + b[8] * b[8], tol);
                try std.testing.expectApproxEqAbs(@as(T, 6), b[0] * b[4] * b[8], tol);
            }
        }
    }
}

test "bidiagonal QR resolves zero diagonals repeated values and scaled rank deficiency" {
    const Dense = @import("zynum-core").internal.numerics.Dense;
    inline for (.{ f64, f128 }) |T| {
        for ([_]T{ 1, 1e-200, 1e200 }) |scale| {
            for ([_][4]T{ .{ 0, 0, 1, 0 }, .{ 0, 0, 1, 2 }, .{ 1, 0, 0, 1 } }, 0..) |original, kind| {
                var b: [4]T = undefined;
                for (&b, original) |*value, input| value.* = input * scale;
                var u: [4]T = undefined;
                var v: [4]T = undefined;
                const bm: Dense.Matrix(T) = .{ .values = &b, .row_count = 2, .column_count = 2, .leading_dimension = 2 };
                const um: Dense.Matrix(T) = .{ .values = &u, .row_count = 2, .column_count = 2, .leading_dimension = 2 };
                const vm: Dense.Matrix(T) = .{ .values = &v, .row_count = 2, .column_count = 2, .leading_dimension = 2 };
                try core.reduce(T, bm, um, vm);
                const result = try core.diagonalize(T, bm, um, vm, .{});
                try std.testing.expect(result.converged);
                try std.testing.expectEqual(@as(usize, 0), result.unconverged);
                const tol = 1024 * std.math.floatEps(T);
                try std.testing.expectApproxEqAbs(if (kind == 1) @sqrt(@as(T, 5)) else @as(T, 1), b[0] / scale, tol);
                try std.testing.expectApproxEqAbs(@as(T, if (kind == 2) 1 else 0), b[3] / scale, tol);
                for (0..2) |i| for (0..2) |j| {
                    var value: T = 0;
                    for (0..2) |k| value += u[i + k * 2] * (b[k + k * 2] / scale) * v[j + k * 2];
                    try std.testing.expectApproxEqAbs(original[i + j * 2], value, tol);
                };
            }
        }
    }
}

test "bidiagonal QR reconstructs deterministic dense and rank deficient ensembles" {
    const Dense = @import("zynum-core").internal.numerics.Dense;
    inline for (.{ f64, f128 }) |T| {
        var seed: u64 = 0x7ab91fc31;
        for (0..24) |case| {
            var original: [35]T = undefined;
            for (&original) |*value| {
                seed = seed *% 6364136223846793005 +% 1442695040888963407;
                value.* = @as(T, @floatFromInt(@as(i32, @intCast(seed >> 48)) - 32768)) / 32768;
            }
            if (case % 2 == 0) for (0..7) |i| {
                original[i + 28] = original[i] + 2 * original[i + 7];
            };
            var b = original;
            var u: [49]T = undefined;
            var v: [25]T = undefined;
            const bm: Dense.Matrix(T) = .{ .values = &b, .row_count = 7, .column_count = 5, .leading_dimension = 7 };
            const um: Dense.Matrix(T) = .{ .values = &u, .row_count = 7, .column_count = 7, .leading_dimension = 7 };
            const vm: Dense.Matrix(T) = .{ .values = &v, .row_count = 5, .column_count = 5, .leading_dimension = 5 };
            try core.reduce(T, bm, um, vm);
            const result = try core.diagonalize(T, bm, um, vm, .{});
            try std.testing.expect(result.converged);
            const tol = 32768 * std.math.floatEps(T);
            for (0..7) |i| for (0..5) |j| {
                var value: T = 0;
                for (0..5) |k| value += u[i + k * 7] * b[k + k * 7] * v[j + k * 5];
                try std.testing.expectApproxEqAbs(original[i + j * 7], value, tol);
            };
            for (0..7) |i| for (0..7) |j| {
                var dot: T = 0;
                for (0..7) |r| dot += u[r + i * 7] * u[r + j * 7];
                try std.testing.expectApproxEqAbs(@as(T, if (i == j) 1 else 0), dot, tol);
            };
            for (0..5) |i| for (0..5) |j| {
                var dot: T = 0;
                for (0..5) |r| dot += v[r + i * 5] * v[r + j * 5];
                try std.testing.expectApproxEqAbs(@as(T, if (i == j) 1 else 0), dot, tol);
            };
            for (0..4) |i| try std.testing.expect(b[i + i * 7] >= b[i + 1 + (i + 1) * 7]);
            try std.testing.expect(b[32] >= 0);
            if (case % 2 == 0) try std.testing.expect(b[32] < tol);
        }
    }
}

test "bidiagonal QR retains tiny singular values across extreme diagonal scales" {
    const Dense = @import("zynum-core").internal.numerics.Dense;
    inline for (.{ f64, f128 }) |T| {
        var b = [_]T{ 1e200, 0, 0, 1e200, 1e-200, 0, 0, 0, 1 };
        var u: [9]T = undefined;
        var v: [9]T = undefined;
        const bm: Dense.Matrix(T) = .{ .values = &b, .row_count = 3, .column_count = 3, .leading_dimension = 3 };
        const um: Dense.Matrix(T) = .{ .values = &u, .row_count = 3, .column_count = 3, .leading_dimension = 3 };
        const vm: Dense.Matrix(T) = .{ .values = &v, .row_count = 3, .column_count = 3, .leading_dimension = 3 };
        try core.reduce(T, bm, um, vm);
        const result = try core.diagonalize(T, bm, um, vm, .{});
        try std.testing.expect(result.converged);
        const tol = 4096 * std.math.floatEps(T);
        try std.testing.expectApproxEqAbs(@sqrt(@as(T, 2)), b[0] / 1e200, tol);
        try std.testing.expectApproxEqAbs(@as(T, 1), b[4], tol);
        try std.testing.expectApproxEqAbs(1 / @sqrt(@as(T, 2)), b[8] / 1e-200, tol);
    }
}

test "bidiagonal QR resolves clustered analytic two by two spectra" {
    const Dense = @import("zynum-core").internal.numerics.Dense;
    inline for (.{ f64, f128 }) |T| {
        for ([_]T{ 1e-8, 32 * std.math.floatEps(T) }) |edge| {
            var b = [_]T{ 1, 0, edge, 1 };
            var u: [4]T = undefined;
            var v: [4]T = undefined;
            const bm: Dense.Matrix(T) = .{ .values = &b, .row_count = 2, .column_count = 2, .leading_dimension = 2 };
            const um: Dense.Matrix(T) = .{ .values = &u, .row_count = 2, .column_count = 2, .leading_dimension = 2 };
            const vm: Dense.Matrix(T) = .{ .values = &v, .row_count = 2, .column_count = 2, .leading_dimension = 2 };
            try core.reduce(T, bm, um, vm);
            const result = try core.diagonalize(T, bm, um, vm, .{});
            try std.testing.expect(result.converged);
            const radius = std.math.hypot(@as(T, 2), edge);
            try std.testing.expectApproxEqAbs((radius + edge) / 2, b[0], 8 * std.math.floatEps(T));
            try std.testing.expectApproxEqAbs((radius - edge) / 2, b[3], 8 * std.math.floatEps(T));
            try std.testing.expect(b[0] > b[3]);
        }
    }
}

test "bidiagonal IEEE policy is bounded and reports actual unfinished edges" {
    const dense = @import("zynum-core").internal.numerics.Dense;
    inline for (.{ f64, f128 }) |T| {
        for ([_]T{ std.math.nan(T), std.math.inf(T), -std.math.inf(T) }) |bad| {
            var b = [_]T{ 1, 0, bad, 2, 77 };
            var u = [_]T{ 1, 0, 0, 1, 88 };
            var v = [_]T{ 1, 0, 0, 1, 99 };
            const bm: dense.Matrix(T) = .{ .values = &b, .row_count = 2, .column_count = 2, .leading_dimension = 2 };
            const um: dense.Matrix(T) = .{ .values = &u, .row_count = 2, .column_count = 2, .leading_dimension = 2 };
            const vm: dense.Matrix(T) = .{ .values = &v, .row_count = 2, .column_count = 2, .leading_dimension = 2 };
            try std.testing.expectError(error.NonFiniteInput, core.diagonalize(T, bm, um, vm, .{}));
            const result = try core.diagonalize(T, bm, um, vm, .{ .max_steps = 3, .numerical_policy = .ieee });
            try std.testing.expect(result.steps <= 3);
            try std.testing.expectEqual(@as(usize, if (b[2] != 0) 1 else 0), result.unconverged);
            try std.testing.expectEqual(result.unconverged == 0, result.converged);
            try std.testing.expectEqual(@as(T, 77), b[4]);
            try std.testing.expectEqual(@as(T, 88), u[4]);
            try std.testing.expectEqual(@as(T, 99), v[4]);
            b = .{ 1, bad, 2, 3, 77 };
            try core.reduceWithPolicy(T, bm, um, vm, .ieee);
            try std.testing.expectEqual(@as(T, 0), b[1]);
            const reduced = try core.diagonalize(T, bm, um, vm, .{ .max_steps = 2, .numerical_policy = .ieee });
            try std.testing.expect(reduced.steps <= 2);
            try std.testing.expectEqual(@as(usize, if (b[2] != 0) 1 else 0), reduced.unconverged);
        }
    }
}
