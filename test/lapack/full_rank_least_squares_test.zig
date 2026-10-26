// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
const std = @import("std");
const Core = @import("zynum-core");
const gels = Core.internal.numerics.FullRankLeastSquares;
const views = Core.lapack;
const types = Core.lapack;
const scalar = Core.internal.numerics.Scalar;
const execution = Core.internal.numerics.Execution;
const scalars = .{ f32, f64, types.ComplexF32, types.ComplexF64 };
const Z = struct { re: f128 = 0, im: f128 = 0 };
fn add(a: Z, b: Z) Z {
    return .{ .re = a.re + b.re, .im = a.im + b.im };
}
fn sub(a: Z, b: Z) Z {
    return .{ .re = a.re - b.re, .im = a.im - b.im };
}
fn mul(a: Z, b: Z) Z {
    return .{ .re = a.re * b.re - a.im * b.im, .im = a.re * b.im + a.im * b.re };
}
fn conj(a: Z) Z {
    return .{ .re = a.re, .im = -a.im };
}
fn scale(a: Z, s: f128) Z {
    return .{ .re = a.re * s, .im = a.im * s };
}
fn abs(a: Z) f128 {
    return @sqrt(a.re * a.re + a.im * a.im);
}
fn val(comptime T: type, a: Z) T {
    return if (comptime scalar.isComplex(T)) .{ .re = @floatCast(a.re), .im = @floatCast(a.im) } else @floatCast(a.re);
}
fn wide(comptime T: type, a: T) Z {
    return if (comptime scalar.isComplex(T)) .{ .re = a.re, .im = a.im } else .{ .re = a };
}
fn epsilon(comptime T: type) f128 {
    return if (scalar.Real(T) == f32) 3e-4 else 3e-11;
}
fn close(comptime T: type, expected: Z, actual: T, unit: f128) !void {
    const got = wide(T, actual);
    const tol = epsilon(T) * @max(@abs(unit), abs(expected));
    try std.testing.expect(std.math.isFinite(got.re) and std.math.isFinite(got.im));
    try std.testing.expectApproxEqAbs(expected.re, got.re, tol);
    try std.testing.expectApproxEqAbs(expected.im, got.im, tol);
}
fn same(comptime T: type, expected: []const T, actual: []const T) !void {
    try std.testing.expectEqualSlices(u8, std.mem.sliceAsBytes(expected), std.mem.sliceAsBytes(actual));
}
fn mat(comptime T: type, a: []T, m: usize, n: usize, ld: usize) views.Matrix(T) {
    return .{ .values = a, .row_count = @intCast(m), .column_count = @intCast(n), .leading_dimension = @intCast(ld) };
}
fn success(result: gels.Result) !void {
    try std.testing.expectEqualDeep(gels.Result{}, result);
}
fn sentinel(comptime T: type) T {
    return val(T, .{ .re = -987, .im = 65 });
}
fn nan(comptime T: type) T {
    return val(T, .{ .re = std.math.nan(f128) });
}
fn base(comptime T: type, i: usize, j: usize) Z {
    const real = [2][3]f128{ .{ 1, 2, -1 }, .{ 2, -1, 3 } };
    const imag = [2][3]f128{ .{ 0.5, -0.25, 0.75 }, .{ -0.5, 0.25, 0.5 } };
    return .{ .re = real[j][i], .im = if (scalar.isComplex(T)) imag[j][i] else 0 };
}
fn originalA(comptime T: type, m: usize, i: usize, j: usize) Z {
    return if (m == 2 and j < 3) conj(base(T, j, i)) else base(T, i, j);
}
fn coefficient(a: []const Z, m: usize, i: usize, j: usize, problem: gels.Operation) Z {
    return if (problem == .normal) a[i + j * m] else conj(a[j + i * m]);
}
fn targetNorm(comptime T: type, a: f128) f128 {
    const R = scalar.Real(T);
    const lo = @as(f128, std.math.floatMin(R)) / @as(f128, std.math.floatEps(R));
    return @min(@max(a, lo), 1 / lo);
}

// Independent dense reflector product. QR uses H_1...H_k; row-stored LQ
// uses H_k^H...H_1^H. The implementation's Q-application routine is not used.
fn packedQ(comptime T: type, a: []const T, tau: []const T, m: usize, n: usize, ld: usize, q: []Z) void {
    const dimension = @max(m, n);
    const k = @min(m, n);
    const is_lq = m < n;
    for (0..dimension) |j| for (0..dimension) |i| {
        q[i + j * dimension] = .{ .re = if (i == j) 1 else 0 };
    };
    for (0..k) |ordinal| {
        const step = if (is_lq) ordinal else k - 1 - ordinal;
        var v: [4]Z = @splat(.{});
        v[step] = .{ .re = 1 };
        for (step + 1..dimension) |i| v[i] = if (is_lq) conj(wide(T, a[step + i * ld])) else wide(T, a[i + step * ld]);
        const t = if (is_lq) conj(wide(T, tau[step])) else wide(T, tau[step]);
        for (0..dimension) |j| {
            var dot: Z = .{};
            for (step..dimension) |i| dot = add(dot, mul(conj(v[i]), q[i + j * dimension]));
            for (step..dimension) |i| q[i + j * dimension] = sub(q[i + j * dimension], mul(mul(t, v[i]), dot));
        }
    }
}

fn branchCase(comptime T: type, comptime m: usize, comptime n: usize, problem: gels.Operation, a_units: f128, b_units: f128, comptime contract: gels.Contract) !void {
    const rows: usize = @max(m, n);
    const k: usize = @min(m, n);
    const lda: usize = m + 2;
    const ldb: usize = rows + 2;
    const nrhs: usize = 2;
    const p: usize = if (problem == .normal) m else n;
    const unknowns: usize = if (problem == .normal) n else m;
    var original: [m * n]Z = undefined;
    for (0..n) |j| for (0..m) |i| {
        original[i + j * m] = originalA(T, m, i, j);
    };
    var x: [rows * nrhs]Z = @splat(.{});
    var rhs: [rows * nrhs]Z = @splat(.{});
    var residual: [rows * nrhs]Z = @splat(.{});
    if (p >= unknowns) {
        for (0..nrhs) |j| for (0..unknowns) |i| {
            x[i + j * rows] = .{ .re = @as(f128, @floatFromInt(1 + i + j)) / 8, .im = if (scalar.isComplex(T)) @as(f128, @floatFromInt(1 + 2 * i + j)) / 16 else 0 };
        };
        if (p > unknowns) {
            var u: [3]Z = undefined;
            var v: [3]Z = undefined;
            for (0..3) |i| {
                u[i] = conj(coefficient(&original, m, i, 0, problem));
                v[i] = conj(coefficient(&original, m, i, 1, problem));
            }
            var w = [3]Z{ sub(mul(u[1], v[2]), mul(u[2], v[1])), sub(mul(u[2], v[0]), mul(u[0], v[2])), sub(mul(u[0], v[1]), mul(u[1], v[0])) };
            var length: f128 = 0;
            for (w) |z| length += abs(z) * abs(z);
            length = @sqrt(length);
            for (&w) |*z| z.* = scale(z.*, 1 / length);
            for (0..nrhs) |j| for (0..p) |i| {
                residual[i + j * rows] = scale(w[i], if (j == 0) 0.125 else -0.25);
            };
        }
    } else {
        // Known X = D^H*Y lies in the minimum-norm subspace of full-row-rank D.
        for (0..nrhs) |j| for (0..unknowns) |i| {
            for (0..p) |r| {
                const y: Z = .{ .re = @as(f128, @floatFromInt(1 + r + j)) / 8, .im = if (scalar.isComplex(T)) @as(f128, @floatFromInt(2 + r + j)) / 16 else 0 };
                x[i + j * rows] = add(x[i + j * rows], mul(conj(coefficient(&original, m, r, i, problem)), y));
            }
        };
    }
    for (0..nrhs) |j| for (0..p) |i| {
        rhs[i + j * rows] = residual[i + j * rows];
        for (0..unknowns) |r| rhs[i + j * rows] = add(rhs[i + j * rows], mul(coefficient(&original, m, i, r, problem), x[r + j * rows]));
    };
    var a: [lda * n + 3]T = @splat(sentinel(T));
    var b: [ldb * nrhs + 3]T = @splat(sentinel(T));
    for (0..n) |j| for (0..m) |i| {
        a[i + j * lda] = val(T, scale(original[i + j * m], a_units));
    };
    for (0..nrhs) |j| for (0..rows) |i| {
        b[i + j * ldb] = if (i < p) val(T, scale(rhs[i + j * rows], b_units)) else nan(T);
    };
    const before_a = a;
    const before_b = b;
    const capacity: usize = k + (if (m < n) m * n + k else 0);
    var work: [capacity + 4]T = @splat(sentinel(T));
    const requirement = try gels.plan(T, m, n, nrhs);
    try std.testing.expectEqual(capacity, requirement.total_elements);
    if (contract == .checked) {
        try success(try gels.solve(T, mat(T, &a, m, n, lda), mat(T, &b, rows, nrhs, ldb), problem, &work));
    } else {
        var operation = execution.beginDefault();
        defer operation.end();
        try success(try gels.solveWithContractAdmitted(&operation, T, mat(T, &a, m, n, lda), mat(T, &b, rows, nrhs, ldb), problem, &work, contract));
    }
    const solution_units = b_units / a_units;
    for (0..nrhs) |j| for (0..unknowns) |i| try close(T, scale(x[i + j * rows], solution_units), b[i + j * ldb], solution_units);
    for (0..n) |j| try same(T, before_a[j * lda + m .. (j + 1) * lda], a[j * lda + m .. (j + 1) * lda]);
    for (0..nrhs) |j| try same(T, before_b[j * ldb + rows .. (j + 1) * ldb], b[j * ldb + rows .. (j + 1) * ldb]);
    try same(T, before_a[lda * n ..], a[lda * n ..]);
    try same(T, before_b[ldb * nrhs ..], b[ldb * nrhs ..]);
    for (work[capacity..]) |z| try close(T, wide(T, sentinel(T)), z, 1);
    var q: [rows * rows]Z = undefined;
    packedQ(T, &a, work[0..k], m, n, lda, &q);
    var norm_a: f128 = 0;
    for (original) |z| norm_a = @max(norm_a, abs(z));
    const target_a = targetNorm(T, norm_a * a_units);
    for (0..n) |j| for (0..m) |i| {
        var got: Z = .{};
        if (m >= n) {
            for (0..@min(j + 1, m)) |r| got = add(got, mul(q[i + r * rows], scale(wide(T, a[r + j * lda]), 1 / target_a)));
        } else {
            for (0..@min(i + 1, n)) |r| got = add(got, mul(scale(wide(T, a[i + r * lda]), 1 / target_a), q[r + j * rows]));
        }
        try std.testing.expect(abs(sub(got, scale(original[i + j * m], 1 / norm_a))) < epsilon(T));
    };
    if (p > unknowns) for (0..nrhs) |j| {
        var rhs_norm: f128 = 0;
        for (0..nrhs) |col| for (0..p) |i| {
            rhs_norm = @max(rhs_norm, abs(rhs[i + col * rows]));
        };
        const target_b = targetNorm(T, rhs_norm * b_units);
        var got: f128 = 0;
        var expected: f128 = 0;
        for (unknowns..rows) |i| {
            const z = scale(wide(T, b[i + j * ldb]), 1 / target_b);
            got += abs(z) * abs(z);
        }
        for (0..p) |i| {
            const z = scale(residual[i + j * rows], 1 / rhs_norm);
            expected += abs(z) * abs(z);
        }
        try std.testing.expectApproxEqAbs(expected, got, epsilon(T));
    };
}

test "GELS all four real complex problem branches preserve analytic solutions factors and scaled residual tails" {
    inline for (scalars) |T| {
        const tiny: f128 = if (scalar.Real(T) == f32) 0x1p-120 else 0x1p-1020;
        const huge: f128 = if (scalar.Real(T) == f32) 0x1p120 else 0x1p1000;
        for ([7][2]f128{ .{ 1, 1 }, .{ tiny, tiny }, .{ huge, huge }, .{ tiny, 1 }, .{ 1, tiny }, .{ huge, 1 }, .{ 1, huge } }) |units| {
            inline for (.{ gels.Operation.normal, gels.Operation.adjoint }) |op| inline for (.{ gels.Contract.checked, gels.Contract.lapack_ieee_v1 }) |contract| {
                try branchCase(T, 3, 2, op, units[0], units[1], contract);
                try branchCase(T, 2, 3, op, units[0], units[1], contract);
                try branchCase(T, 2, 2, op, units[0], units[1], contract);
            };
        }
    }
}

test "GELS singular stage keeps minimum norm tail and publishes transformed least squares RHS" {
    inline for (scalars) |T| inline for (.{ false, true }) |is_lq| inline for (.{ gels.Operation.normal, gels.Operation.adjoint }) |problem| {
        const m: usize = if (is_lq) 2 else 3;
        const n: usize = if (is_lq) 3 else 2;
        const k: usize = 2;
        const rows: usize = 3;
        var a: [6]T = @splat(val(T, .{}));
        if (is_lq) {
            a[1 + 1 * m] = val(T, .{ .re = 1, .im = 0.5 });
            a[1 + 2 * m] = val(T, .{ .re = 2, .im = -0.25 });
        } else {
            a[1 + 1 * m] = val(T, .{ .re = 1, .im = 0.5 });
            a[2 + 1 * m] = val(T, .{ .re = 2, .im = -0.25 });
        }
        const least_squares = (!is_lq and problem == .normal) or (is_lq and problem == .adjoint);
        var b = [3]T{ val(T, .{ .re = 1 }), val(T, .{ .re = 2, .im = 0.25 }), if (least_squares) val(T, .{ .re = 3, .im = -0.5 }) else nan(T) };
        const original = b;
        var work: [10]T = @splat(sentinel(T));
        const result = try gels.solve(T, mat(T, &a, m, n, m), mat(T, &b, rows, 1, rows), problem, &work);
        try std.testing.expectEqualDeep(gels.Result{ .info = 1, .failure = .singular, .stage = .triangular_solve }, result);
        if (!least_squares) try same(T, &original, &b) else {
            var q: [9]Z = undefined;
            packedQ(T, &a, work[0..k], m, n, m, &q);
            for (0..3) |i| {
                var expected: Z = .{};
                for (0..3) |j| expected = add(expected, mul(if (is_lq) q[i + j * 3] else conj(q[j + i * 3]), wide(T, original[j])));
                try close(T, expected, b[i], 1);
            }
        }
    };
}

test "GELS all zero and empty computations clear complete B without factoring or scratch mutation" {
    inline for (scalars) |T| inline for (.{ gels.Operation.normal, gels.Operation.adjoint }) |problem| {
        var a: [6]T = @splat(val(T, .{}));
        var b: [3]T = @splat(val(T, .{ .re = 2 }));
        var work: [10]T = @splat(sentinel(T));
        const ow = work;
        const oa = a;
        try success(try gels.solve(T, mat(T, &a, 2, 3, 2), mat(T, &b, 3, 1, 3), problem, &work));
        for (b) |z| try close(T, .{}, z, 1);
        try same(T, &oa, &a);
        try same(T, &ow, &work);
        var empty: [0]T = .{};
        var output: [4]T = @splat(nan(T));
        try success(try gels.solve(T, mat(T, &empty, 0, 4, 0), mat(T, &output, 4, 1, 4), problem, &empty));
        for (output) |z| try close(T, .{}, z, 1);
        @memset(&a, nan(T));
        const poisoned = a;
        try success(try gels.solve(T, mat(T, &a, 3, 2, 3), mat(T, &empty, 3, 0, 0), problem, &empty));
        try same(T, &poisoned, &a);
    };
}

test "GELS geometry capacity aliases and used nonfinite values reject before normalization" {
    inline for (scalars) |T| {
        var a: [6]T = @splat(val(T, .{ .re = 1 }));
        var b: [3]T = @splat(val(T, .{ .re = 2 }));
        var work: [4]T = @splat(sentinel(T));
        const oa = a;
        const ob = b;
        const ow = work;
        try std.testing.expectError(error.BufferTooSmall, gels.solve(T, mat(T, &a, 3, 2, 3), mat(T, &b, 3, 1, 3), .normal, work[0..1]));
        try std.testing.expectError(error.DimensionMismatch, gels.solve(T, mat(T, &a, 3, 2, 3), mat(T, &b, 2, 1, 3), .normal, &work));
        try std.testing.expectError(error.AliasingNotAllowed, gels.solve(T, mat(T, &a, 3, 2, 3), mat(T, a[0..3], 3, 1, 3), .normal, &work));
        try std.testing.expectError(error.AliasingNotAllowed, gels.solve(T, mat(T, &a, 3, 2, 3), mat(T, &b, 3, 1, 3), .normal, a[0..2]));
        try same(T, &oa, &a);
        try same(T, &ob, &b);
        try same(T, &ow, &work);
        a[0] = nan(T);
        const poisoned = a;
        try std.testing.expectError(error.NonFiniteInput, gels.solve(T, mat(T, &a, 3, 2, 3), mat(T, &b, 3, 1, 3), .normal, &work));
        try same(T, &poisoned, &a);
        try same(T, &ob, &b);
        try same(T, &ow, &work);
        a = oa;
        b[0] = nan(T);
        const poisoned_b = b;
        try std.testing.expectError(error.NonFiniteInput, gels.solve(T, mat(T, &a, 3, 2, 3), mat(T, &b, 3, 1, 3), .normal, &work));
        try same(T, &oa, &a);
        try same(T, &poisoned_b, &b);
        try same(T, &ow, &work);
        try std.testing.expectError(error.InvalidLength, gels.plan(T, std.math.maxInt(usize) - 1, std.math.maxInt(usize), 1));
    }
}

fn smallest(comptime T: type) scalar.Real(T) {
    return if (scalar.Real(T) == f32) @bitCast(@as(u32, 1)) else @bitCast(@as(u64, 1));
}
test "GELS smallest subnormal scalar equations solve in original units for real complex N H" {
    inline for (scalars) |T| inline for (.{ gels.Operation.normal, gels.Operation.adjoint }) |problem| {
        const a_value: Z = .{ .re = smallest(T), .im = if (scalar.isComplex(T)) smallest(T) else 0 };
        var a = [1]T{val(T, a_value)};
        var b = [1]T{val(T, if (problem == .normal) a_value else conj(a_value))};
        var work: [1]T = undefined;
        try success(try gels.solve(T, mat(T, &a, 1, 1, 1), mat(T, &b, 1, 1, 1), problem, &work));
        try close(T, .{ .re = 1 }, b[0], 1);
    };
}

test "GELS all four branches finish on one admitted operation after Context begins closing" {
    inline for (scalars) |T| inline for (.{ false, true }) |is_lq| inline for (.{ gels.Operation.normal, gels.Operation.adjoint }) |problem| {
        var context = execution.context.Context.init(std.testing.allocator, execution.test_support.policy, .internal_closable);
        var operation = try context.begin();
        _ = context.requestClose();
        const m: usize = if (is_lq) 2 else 3;
        const n: usize = if (is_lq) 3 else 2;
        const p: usize = if (problem == .normal) m else n;
        var a: [6]T = undefined;
        for (0..n) |j| for (0..m) |i| {
            a[i + j * m] = val(T, originalA(T, m, i, j));
        };
        var b: [3]T = @splat(nan(T));
        for (0..p) |i| b[i] = val(T, .{ .re = @floatFromInt(i + 1) });
        var work: [10]T = undefined;
        try success(try gels.solveWithContractAdmitted(&operation, T, mat(T, &a, m, n, m), mat(T, &b, 3, 1, 3), problem, &work, .checked));
        try std.testing.expectEqual(@as(usize, 1), context.snapshot().active_operations);
        try std.testing.expectEqual(@as(usize, 0), context.snapshot().resource_pins);
        try std.testing.expectError(error.Closing, context.begin());
        operation.end();
        try std.testing.expectEqual(execution.context.CloseResult.closed, context.tryProgressClose());
    };
}

test "GELS checked unrepresentable rescale remains explicit while IEEE returns no invented singular INFO" {
    inline for (scalars) |T| {
        var a = [1]T{val(T, .{ .re = smallest(T) })};
        var b = [1]T{val(T, .{ .re = 1 })};
        var work: [1]T = undefined;
        const failure = try gels.solve(T, mat(T, &a, 1, 1, 1), mat(T, &b, 1, 1, 1), .normal, &work);
        try std.testing.expectEqualDeep(gels.Result{ .info = 1, .failure = .arithmetic, .stage = .rescaling, .rhs_column = 1 }, failure);
        try std.testing.expect(std.math.isFinite(wide(T, b[0]).re));
        var context = execution.context.Context.init(std.testing.allocator, execution.test_support.policy, .internal_closable);
        var operation = try context.begin();
        a[0] = val(T, .{ .re = smallest(T) });
        b[0] = val(T, .{ .re = 1 });
        try success(try gels.solveWithContractAdmitted(&operation, T, mat(T, &a, 1, 1, 1), mat(T, &b, 1, 1, 1), .normal, &work, .lapack_ieee_v1));
        try std.testing.expect(std.math.isInf(wide(T, b[0]).re));
        operation.end();
        _ = context.requestClose();
        try std.testing.expectEqual(execution.context.CloseResult.closed, context.tryProgressClose());
    }
}

test "GELS IEEE nonfinite real complex N H paths execute without numerical readmission or invented INFO" {
    inline for (scalars) |T| inline for (.{ gels.Operation.normal, gels.Operation.adjoint }) |problem| {
        var context = execution.context.Context.init(std.testing.allocator, execution.test_support.policy, .internal_closable);
        var operation = try context.begin();
        _ = context.requestClose();
        var a = [2]T{ nan(T), val(T, .{ .re = 1 }) };
        var b = [2]T{ val(T, .{ .re = 1 }), nan(T) };
        var work: [4]T = undefined;
        try success(try gels.solveWithContractAdmitted(&operation, T, mat(T, &a, 1, 2, 1), mat(T, &b, 2, 1, 2), problem, &work, .lapack_ieee_v1));
        try std.testing.expectEqual(@as(usize, 1), context.snapshot().active_operations);
        try std.testing.expectEqual(@as(usize, 0), context.snapshot().resource_pins);
        operation.end();
        try std.testing.expectEqual(execution.context.CloseResult.closed, context.tryProgressClose());
    };
}
