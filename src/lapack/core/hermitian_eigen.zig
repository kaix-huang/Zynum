// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later

//! Allocation-free complex Hermitian eigensystems over common borrowed storage.
//! The default checked contract commits only converged, finite narrowed outputs.
//! The explicit IEEE contract uses real tridiagonal iteration, permits nonfinite
//! outputs and commits unfinished iteration data with actual edge counts.
//! Inputs remain readonly. Scratch is caller-owned, disjoint and may be overwritten.
const std = @import("std");
const dense = @import("../../numerics/dense.zig");
const storage = @import("../../numerics/storage.zig");
const cx = @import("../../numerics/complex.zig");
const validation = @import("../validation.zig");
pub const LapackTridiagonal = @import("hermitian_tridiagonal.zig");
pub const Contract = enum { checked_finite, heev_ieee_v1 };
pub const Error = validation.Error || dense.Error;
pub const Triangle = enum { upper, lower };
pub const Problem = enum { ax_lambda_bx, abx_lambda_x, bax_lambda_x };
pub const Status = enum { converged, iteration_limit, not_positive_definite, range_limit };
pub const UnconvergedBlock = LapackTridiagonal.UnconvergedBlock;
pub const Result = struct {
    status: Status,
    sweeps: usize = 0,
    failing_pivot: usize = 0,
    unconverged: usize = 0,
    failed_block: ?UnconvergedBlock = null,
};
pub const GeneralizedFailureVectors = enum { backtransform_available, keep_reduced };
pub const Options = struct {
    max_sweeps: usize = 96,
    relative_tolerance: ?f64 = null,
    // A boundary chooses how unfinished standard-problem vectors are exposed.
    // Preserve the existing partial recovery unless the caller requests them
    // in the reduced basis. This does not change iteration or convergence.
    generalized_failure_vectors: GeneralizedFailureVectors = .backtransform_available,
};
pub fn WorkspaceScalar(comptime C: type) type {
    return cx.WideComplex(C);
}
pub fn workspaceLength(comptime C: type, n: usize) Error!usize {
    const square = std.math.mul(usize, n, n) catch return error.InvalidLength;
    const count = std.math.mul(usize, square, 4) catch return error.InvalidLength;
    _ = storage.byteLength(count, @sizeOf(WorkspaceScalar(C))) catch return error.InvalidLength;
    return count;
}
fn conjugate(z: anytype) @TypeOf(z) {
    return .{ .re = z.re, .im = -z.im };
}
fn product(a: anytype, b: @TypeOf(a)) @TypeOf(a) {
    return .{ .re = a.re * b.re - a.im * b.im, .im = a.re * b.im + a.im * b.re };
}
fn subtract(a: anytype, b: @TypeOf(a)) @TypeOf(a) {
    return .{ .re = a.re - b.re, .im = a.im - b.im };
}
fn scaled(z: anytype, x: @TypeOf(z.re)) @TypeOf(z) {
    return .{ .re = z.re * x, .im = z.im * x };
}
fn magnitude(z: anytype) @TypeOf(z.re) {
    return std.math.hypot(z.re, z.im);
}
fn get(comptime C: type, a: dense.ConstMatrix(C), triangle: Triangle, i: usize, j: usize) C {
    if (i == j) return .{ .re = a.values[i + j * a.leading_dimension].re, .im = 0 };
    const stored = if (triangle == .lower) i >= j else i <= j;
    const z = a.values[if (stored) i + j * a.leading_dimension else j + i * a.leading_dimension];
    return if (stored) z else .{ .re = z.re, .im = -z.im };
}
fn scan(comptime C: type, a: dense.ConstMatrix(C), triangle: Triangle, comptime contract: Contract) Error!cx.Wide(C) {
    var largest: cx.Wide(C) = 0;
    for (0..a.column_count) |j| for (0..a.row_count) |i| {
        if ((triangle == .lower and i < j) or (triangle == .upper and i > j)) continue;
        const z = get(C, a, triangle, i, j);
        if (contract == .checked_finite and !cx.finite(C, z)) return error.NonFiniteInput;
        largest = @max(largest, @max(@abs(@as(cx.Wide(C), z.re)), @abs(@as(cx.Wide(C), z.im))));
    };
    return if (largest == 0 or (contract == .heev_ieee_v1 and !std.math.isFinite(largest))) 1 else largest;
}
fn load(comptime C: type, a: dense.ConstMatrix(C), triangle: Triangle, scale: cx.Wide(C), out: []WorkspaceScalar(C)) void {
    const n = a.row_count;
    for (0..n) |j| for (0..n) |i| {
        out[i + j * n] = scaled(cx.widen(C, get(C, a, triangle, i, j)), 1 / scale);
    };
}
fn hermitianize(h: anytype, n: usize) void {
    for (0..n) |j| {
        h[j + j * n].im = 0;
        for (j + 1..n) |i| {
            const a = h[i + j * n];
            const b = conjugate(h[j + i * n]);
            h[i + j * n] = .{ .re = (a.re + b.re) / 2, .im = (a.im + b.im) / 2 };
            h[j + i * n] = conjugate(h[i + j * n]);
        }
    }
}
fn jacobi(comptime C: type, h: []WorkspaceScalar(C), v: []WorkspaceScalar(C), n: usize, options: Options) Result {
    const W = cx.Wide(C);
    for (v, 0..) |*z, k| z.* = .{ .re = if (k % n == k / n) 1 else 0, .im = 0 };
    const tolerance: W = if (options.relative_tolerance) |t| t else std.math.floatEps(W) * @as(W, @floatFromInt(@max(n, 1))) * 8;
    var sweeps: usize = 0;
    while (true) {
        var needs_rotation = false;
        for (0..n) |q| for (0..q) |p| {
            const threshold = tolerance * @sqrt(@abs(h[p + p * n].re)) * @sqrt(@abs(h[q + q * n].re));
            if (magnitude(h[p + q * n]) > threshold) needs_rotation = true;
        };
        if (!needs_rotation) return .{ .status = .converged, .sweeps = sweeps };
        if (sweeps == options.max_sweeps) return .{ .status = .iteration_limit, .sweeps = sweeps };
        sweeps += 1;
        for (0..n) |p| for (p + 1..n) |q| {
            const b = h[p + q * n];
            const r = magnitude(b);
            const threshold = tolerance * @sqrt(@abs(h[p + p * n].re)) * @sqrt(@abs(h[q + q * n].re));
            if (r == 0 or r <= threshold) continue;
            const phase = scaled(b, 1 / r);
            const tau = (h[q + q * n].re - h[p + p * n].re) / (2 * r);
            const t = (if (tau < 0) @as(W, -1) else @as(W, 1)) / (@abs(tau) + std.math.hypot(@as(W, 1), tau));
            const c = 1 / @sqrt(1 + t * t);
            const s = t * c;
            for (0..n) |k| {
                if (k != p and k != q) {
                    const a = h[k + p * n];
                    const d = h[k + q * n];
                    const first = subtract(scaled(a, c), scaled(product(conjugate(phase), d), s));
                    const second = subtract(scaled(d, c), scaled(product(phase, a), -s));
                    h[k + p * n] = first;
                    h[p + k * n] = conjugate(first);
                    h[k + q * n] = second;
                    h[q + k * n] = conjugate(second);
                }
                const a = v[k + p * n];
                const d = v[k + q * n];
                v[k + p * n] = subtract(scaled(a, c), scaled(product(conjugate(phase), d), s));
                v[k + q * n] = subtract(scaled(d, c), scaled(product(phase, a), -s));
            }
            h[p + p * n].re -= t * r;
            h[q + q * n].re += t * r;
            h[p + q * n] = .{ .re = 0, .im = 0 };
            h[q + p * n] = .{ .re = 0, .im = 0 };
        };
    }
}
fn cholesky(l: anytype, n: usize, comptime contract: Contract) usize {
    for (0..n) |j| {
        var d = l[j + j * n].re;
        for (0..j) |k| {
            const z = l[j + k * n];
            d -= z.re * z.re + z.im * z.im;
        }
        if (!(d > 0) or (contract == .checked_finite and !std.math.isFinite(d))) return j + 1;
        const root = @sqrt(d);
        l[j + j * n] = .{ .re = root, .im = 0 };
        for (j + 1..n) |i| {
            var z = l[i + j * n];
            for (0..j) |k| z = subtract(z, product(l[i + k * n], conjugate(l[j + k * n])));
            l[i + j * n] = scaled(z, 1 / root);
        }
        for (0..j) |i| l[i + j * n] = .{ .re = 0, .im = 0 };
    }
    return 0;
}
fn transform(h: anytype, l: @TypeOf(h), tmp: @TypeOf(h), n: usize, problem: Problem) void {
    if (problem == .ax_lambda_bx) {
        for (0..n) |j| for (0..n) |i| {
            var z = h[i + j * n];
            for (0..i) |k| z = subtract(z, product(l[i + k * n], tmp[k + j * n]));
            tmp[i + j * n] = scaled(z, 1 / l[i + i * n].re);
        };
        for (0..n) |i| for (0..n) |j| {
            var z = tmp[i + j * n];
            for (0..j) |k| z = subtract(z, product(h[i + k * n], conjugate(l[j + k * n])));
            h[i + j * n] = scaled(z, 1 / l[j + j * n].re);
        };
    } else {
        for (0..n) |j| for (0..n) |i| {
            var z: @TypeOf(h[0]) = .{ .re = 0, .im = 0 };
            for (j..n) |k| z = subtract(z, scaled(product(h[i + k * n], l[k + j * n]), -1));
            tmp[i + j * n] = z;
        };
        for (0..n) |j| for (0..n) |i| {
            var z: @TypeOf(h[0]) = .{ .re = 0, .im = 0 };
            for (i..n) |k| z = subtract(z, scaled(product(conjugate(l[k + i * n]), tmp[k + j * n]), -1));
            h[i + j * n] = z;
        };
    }
    hermitianize(h, n);
}

pub fn solve(comptime C: type, a: dense.ConstMatrix(C), triangle: Triangle, eigenvalues: []cx.scalar.Real(C), vectors: ?dense.Matrix(C), workspace: []WorkspaceScalar(C), options: Options) Error!Result {
    return solveImpl(C, a, null, .ax_lambda_bx, triangle, eigenvalues, vectors, workspace, options, .checked_finite);
}
pub fn solveGeneralized(comptime C: type, a: dense.ConstMatrix(C), b: dense.ConstMatrix(C), problem: Problem, triangle: Triangle, eigenvalues: []cx.scalar.Real(C), vectors: ?dense.Matrix(C), workspace: []WorkspaceScalar(C), options: Options) Error!Result {
    return solveImpl(C, a, b, problem, triangle, eigenvalues, vectors, workspace, options, .checked_finite);
}
pub fn solveWithContract(comptime C: type, a: dense.ConstMatrix(C), triangle: Triangle, eigenvalues: []cx.scalar.Real(C), vectors: ?dense.Matrix(C), workspace: []WorkspaceScalar(C), options: Options, comptime contract: Contract) Error!Result {
    return solveImpl(C, a, null, .ax_lambda_bx, triangle, eigenvalues, vectors, workspace, options, contract);
}
pub fn solveGeneralizedWithContract(comptime C: type, a: dense.ConstMatrix(C), b: dense.ConstMatrix(C), problem: Problem, triangle: Triangle, eigenvalues: []cx.scalar.Real(C), vectors: ?dense.Matrix(C), workspace: []WorkspaceScalar(C), options: Options, comptime contract: Contract) Error!Result {
    return solveImpl(C, a, b, problem, triangle, eigenvalues, vectors, workspace, options, contract);
}
fn solveImpl(comptime C: type, a: dense.ConstMatrix(C), b: ?dense.ConstMatrix(C), problem: Problem, triangle: Triangle, eigenvalues: []cx.scalar.Real(C), vectors: ?dense.Matrix(C), workspace: []WorkspaceScalar(C), options: Options, comptime contract: Contract) Error!Result {
    const W = cx.Wide(C);
    const R = cx.scalar.Real(C);
    const n = a.row_count;
    if (a.column_count != n) return error.DimensionMismatch;
    const alen = try a.check();
    const count = try workspaceLength(C, n);
    if (eigenvalues.len < n or workspace.len < count) return error.BufferTooSmall;
    if (options.relative_tolerance) |t| {
        if (!(t > 0 and t < 1) or !std.math.isFinite(t)) return error.InvalidTolerance;
    }
    const w = workspace[0..count];
    const values = eigenvalues[0..n];
    try validation.noOverlap(C, a.values[0..alen], R, values);
    try validation.noOverlap(C, a.values[0..alen], WorkspaceScalar(C), w);
    try validation.noOverlap(R, values, WorkspaceScalar(C), w);
    var vlen: usize = 0;
    if (vectors) |z| {
        if (z.row_count != n or z.column_count != n) return error.DimensionMismatch;
        vlen = try z.check();
        try validation.noOverlap(C, z.values[0..vlen], C, a.values[0..alen]);
        try validation.noOverlap(C, z.values[0..vlen], R, values);
        try validation.noOverlap(C, z.values[0..vlen], WorkspaceScalar(C), w);
    }
    if (b) |m| {
        if (m.row_count != n or m.column_count != n) return error.DimensionMismatch;
        const blen = try m.check();
        try validation.noOverlap(C, m.values[0..blen], R, values);
        try validation.noOverlap(C, m.values[0..blen], WorkspaceScalar(C), w);
        if (vectors) |z| try validation.noOverlap(C, m.values[0..blen], C, z.values[0..vlen]);
    }
    const sa = try scan(C, a, triangle, contract);
    const sb: W = if (b) |m| try scan(C, m, triangle, contract) else 1;
    if (n == 0) return .{ .status = .converged };
    const nn = n * n;
    const h = w[0..nn];
    const v = w[nn .. 2 * nn];
    const l = w[2 * nn .. 3 * nn];
    const tmp = w[3 * nn .. 4 * nn];
    load(C, a, triangle, sa, h);
    if (b) |m| {
        load(C, m, triangle, sb, l);
        const pivot = cholesky(l, n, contract);
        if (pivot != 0) return .{ .status = .not_positive_definite, .failing_pivot = pivot };
        transform(h, l, tmp, n, problem);
    }
    // A second normalization bounds Jacobi intermediates after reduction.
    var reduced_scale: W = 0;
    for (h) |z| {
        reduced_scale = @max(reduced_scale, @max(@abs(z.re), @abs(z.im)));
    }
    if (!std.math.isFinite(reduced_scale)) {
        if (contract == .checked_finite) return .{ .status = .range_limit };
        reduced_scale = 1;
    }
    if (reduced_scale == 0) reduced_scale = 1;
    for (h) |*z| z.* = scaled(z.*, 1 / reduced_scale);
    const result: Result = if (contract == .checked_finite) jacobi(C, h, v, n, options) else blk: {
        const t = LapackTridiagonal.solve(C, h, v, tmp, n, options.max_sweeps *| n);
        break :blk .{ .status = if (t.unconverged == 0) .converged else .iteration_limit, .sweeps = t.iterations, .unconverged = t.unconverged, .failed_block = t.failed_block };
    };
    if (contract == .checked_finite and result.status != .converged) return result;
    if (result.status == .converged) {
        for (0..n) |i| {
            var smallest = i;
            for (i + 1..n) |j| {
                if (h[j + j * n].re < h[smallest + smallest * n].re) smallest = j;
            }
            if (smallest != i) {
                std.mem.swap(WorkspaceScalar(C), &h[i + i * n], &h[smallest + smallest * n]);
                for (0..n) |k| std.mem.swap(WorkspaceScalar(C), &v[k + i * n], &v[k + smallest * n]);
            }
        }
    }
    if (b != null and vectors != null) {
        const neig = if (contract == .heev_ieee_v1 and result.status == .iteration_limit)
            (if (options.generalized_failure_vectors == .keep_reduced) 0 else result.unconverged -| 1)
        else
            n;
        if (problem == .bax_lambda_x) {
            for (0..neig) |j| for (0..n) |i| {
                var z: WorkspaceScalar(C) = .{ .re = 0, .im = 0 };
                for (0..i + 1) |k| z = subtract(z, scaled(product(l[i + k * n], v[k + j * n]), -1));
                tmp[i + j * n] = scaled(z, @sqrt(sb));
            };
            @memcpy(v[0 .. neig * n], tmp[0 .. neig * n]);
        } else {
            for (0..neig) |j| {
                var i = n;
                while (i > 0) {
                    i -= 1;
                    var z = v[i + j * n];
                    for (i + 1..n) |k| z = subtract(z, product(conjugate(l[k + i * n]), v[k + j * n]));
                    v[i + j * n] = scaled(z, 1 / l[i + i * n].re);
                }
            }
            for (v[0 .. neig * n]) |*z| z.* = scaled(z.*, 1 / @sqrt(sb));
        }
    }
    const eigen_scale = if (b == null) sa else if (problem == .ax_lambda_bx) sa / sb else sa * sb;
    if (contract == .checked_finite) {
        for (0..n) |j| {
            const value: R = @floatCast(h[j + j * n].re * reduced_scale * eigen_scale);
            if (!std.math.isFinite(value)) return .{ .status = .range_limit, .sweeps = result.sweeps };
            if (vectors != null) for (0..n) |i| {
                if (cx.narrow(C, v[i + j * n]) == null) return .{ .status = .range_limit, .sweeps = result.sweeps };
            };
        }
    }
    for (0..n) |j| {
        values[j] = @floatCast(h[j + j * n].re * reduced_scale * eigen_scale);
        if (vectors) |z| for (0..n) |i| {
            z.values[i + j * z.leading_dimension] = if (contract == .checked_finite) cx.narrow(C, v[i + j * n]).? else .{ .re = @floatCast(v[i + j * n].re), .im = @floatCast(v[i + j * n].im) };
        };
    }
    return result;
}
